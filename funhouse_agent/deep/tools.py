"""LangChain tools for the deepagents (v5.0) port of the geotech agent.

Wraps the same dispatch surface the v1 agent uses
(:mod:`funhouse_agent.dispatch` + :mod:`funhouse_agent.vision_tools`) as
LangChain :class:`~langchain_core.tools.StructuredTool` objects so they can be
handed to ``deepagents.create_deep_agent``.

Design goals (mirrors v1):

* Every tool returns a **JSON string** (``json.dumps(..., default=str)``)
  exactly like ``dispatch.dispatch_tool`` / ``native_tools.dispatch_native_tool``.
* ``call_agent`` preserves the flattened-parameter auto-nesting quirk from
  ``native_tools.dispatch_native_tool`` (LLMs frequently hoist method params to
  the top level instead of nesting them under ``parameters``).
* Tools are produced by factories bound to an ``allowed_agents`` scope so the
  references sub-agent sees ``REFERENCE_MODULES`` while the primary sees
  ``ANALYSIS_MODULES`` — the same scoping v1 enforces at the dispatcher.
* Tool descriptions are kept close to the (tuned) wording in
  ``native_tools.OPENAI_TOOLS`` and ``vision_tools.VISION_TOOL_DESCRIPTIONS``.

The vision tools need a GenAIEngine + attachments + save_fn. For offline
construction those are injected at build time (closured); ``read_reference_figure``
and ``save_file`` work without a live engine where possible, and ``analyze_*``
return a clear error if no engine was wired.
"""

import json
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from funhouse_agent.dispatch import (
    REFERENCE_MODULES,
    call_agent as _call_agent,
    describe_method as _describe_method,
    list_agents as _list_agents,
    list_methods as _list_methods,
    # Resolution primitives (underscore-prefixed but importable). Used to make
    # describe_method redirect a guessed method name the same way call_agent
    # already does, instead of bouncing with a bare "Unknown method".
    _METHOD_ALIASES,
    _load_adapter,
    _selector_value_candidates,
)
from funhouse_agent.vision_tools import (
    dispatch_extended_tool as _dispatch_extended_tool,
    _default_save_fn,
)


# ---------------------------------------------------------------------------
# Tool-result size cap (token-bloat guard)
# ---------------------------------------------------------------------------

#: Default cap on the size of a tool result fed back to the model, mirroring
#: v1's ``GeotechAgent._max_result_chars`` (``funhouse_agent/agent.py``). Large
#: reference-text dumps from the references sub-agent would otherwise be
#: re-sent on every ReAct round and compound the input-token cost.
DEFAULT_MAX_RESULT_CHARS = 8000

#: Larger default cap for REFERENCE reads (``call_agent`` on a reference
#: module + ``read_reference_figure``): for those tools the reference text IS
#: the payload, so cutting it at the general cap loses the answer and provokes
#: a wasteful re-read. Numeric/calc results are NOT capped at all (they are
#: tiny) — see :func:`_result_cap_for_module`.
DEFAULT_REFERENCE_RESULT_CHARS = 16000

#: Appended after the truncation marker so the agent does a NARROWER follow-up
#: search instead of re-requesting (and re-truncating) the same large item.
SEARCH_NARROWER_NUDGE = (
    " Do NOT re-request this same item — it will truncate again. Run a "
    "NARROWER follow-up search instead (more specific terms, or one specific "
    "section/table/figure) to retrieve just the part you need."
)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate a tool-result string to ``max_chars``, marking any cut.

    Mirrors :func:`funhouse_agent.react_support._truncate` (the v1 behavior),
    but appends a count of the dropped characters so the marker is informative,
    plus :data:`SEARCH_NARROWER_NUDGE` so the agent's recovery move is a
    narrower search rather than a re-read of the same oversized item.

    Parameters
    ----------
    text : str
        The tool result (already a JSON string in this module).
    max_chars : int
        Maximum number of characters to keep. A value ``<= 0`` disables
        truncation and the full ``text`` is returned unchanged.

    Returns
    -------
    str
        ``text`` unchanged when it is short enough (or truncation is disabled),
        otherwise the first ``max_chars`` characters followed by a
        ``\\n...[truncated N chars]`` marker and the search-narrower nudge.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n...[truncated {len(text) - max_chars} chars]."
        + SEARCH_NARROWER_NUDGE
    )


def _resolve_reference_cap(max_result_chars: int,
                           reference_result_chars: Optional[int]) -> int:
    """Resolve the cap used for REFERENCE reads.

    * Truncation disabled globally (``max_result_chars <= 0``) → disabled here
      too (``0``).
    * Explicit ``reference_result_chars`` → used as-is.
    * Default (``None``) → ``DEFAULT_REFERENCE_RESULT_CHARS``, but never
      SMALLER than the general cap (a caller raising the general cap above
      16000 should not silently shrink reference reads).
    """
    if max_result_chars <= 0:
        return 0
    if reference_result_chars is not None:
        return reference_result_chars
    return max(DEFAULT_REFERENCE_RESULT_CHARS, max_result_chars)


def _result_cap_for_module(agent_name: str, max_result_chars: int,
                           reference_cap: int) -> int:
    """Per-call cap for a ``call_agent`` result, by target module.

    * Reference modules (DM7/GEC/UFC/... text + ``reference_db``/``figure_db``)
      get the LARGER ``reference_cap`` — the reference text is the payload.
    * Numeric/calc modules are UNCAPPED (``0``): their JSON results are tiny,
      and capping them risks cutting a valid result mid-number for no token
      benefit.
    """
    if agent_name in REFERENCE_MODULES:
        return reference_cap
    return 0


# ---------------------------------------------------------------------------
# Core dispatch tools (the 4 meta-tools), scoped to allowed_agents
# ---------------------------------------------------------------------------

# Keys that belong to call_agent itself; anything else the model passes at the
# top level is treated as a flattened method parameter and auto-nested.
_CALL_AGENT_KEYS = {"agent_name", "method", "parameters"}


def _resolve_describe_method(agent_name, method, allowed_agents):
    """Resolve a guessed ``describe_method`` method name to real docs.

    Mirrors the auto-resolution ``call_agent`` already performs (via the
    dispatch ``_METHOD_ALIASES`` map and selector-value handling) so that
    ``describe_method`` *redirects* a guess instead of bouncing it with a bare
    "Unknown method" error — saving the agent a wasted recovery round.

    The happy path (a real method) is left to :func:`dispatch.describe_method`;
    this only runs when that returned an "Unknown method" error.

    Parameters
    ----------
    agent_name : str
        The module name (already confirmed visible/allowed by the caller, since
        ``dispatch.describe_method`` returned an *Unknown method* — not an
        *Unknown module* — error).
    method : str
        The guessed method name to resolve.
    allowed_agents : iterable of str or None
        The active scope, passed through to :func:`dispatch.describe_method` so
        the returned docs respect the same visibility the caller enforces.

    Returns
    -------
    dict or None
        On success, the real method's documentation dict with a top-level
        ``"_note"`` explaining the redirect. ``None`` when ``method`` cannot be
        resolved to a real method of ``agent_name`` (the caller then builds an
        enriched error). Defensive: any failure loading the adapter or probing
        selector candidates yields ``None``.
    """
    guess = (method or "").strip().lower()

    # (a) Curated alias map: the value is either the real method name (str) or a
    # ``(real_method, {param: value})`` tuple when a selector value is implied.
    entry = _METHOD_ALIASES.get((agent_name, guess))
    if entry is not None:
        real = entry if isinstance(entry, str) else entry[0]
        inject = {} if isinstance(entry, str) else (entry[1] or {})
        docs = _describe_method(
            agent_name=agent_name, method=real, allowed_agents=allowed_agents,
        )
        if "error" not in docs:
            out = dict(docs)
            if inject:
                pairs = ", ".join(f"{p}='{v}'" for p, v in inject.items())
                out["_note"] = (
                    f"'{method}' is not a method name — it maps to "
                    f"'{real}' with {pairs}. Showing that method's docs; "
                    f"call it with {pairs}."
                )
            else:
                out["_note"] = (
                    f"'{method}' is not a method name — it maps to the "
                    f"'{real}' method. Showing that method's docs."
                )
            return out

    # (b) Selector value: ``method`` is an allowed VALUE of a selector parameter
    # (e.g. 'vesic' for the 'factor_method' parameter), not a method name.
    try:
        mod = _load_adapter(agent_name)
        cands = _selector_value_candidates(mod, method)
    except Exception:
        return None
    if cands:
        real, selector = cands[0]
        docs = _describe_method(
            agent_name=agent_name, method=real, allowed_agents=allowed_agents,
        )
        if "error" not in docs:
            out = dict(docs)
            out["_note"] = (
                f"'{method}' is not a method name — it is a value for the "
                f"'{selector}' parameter of '{real}'. Showing that method's "
                f"docs; call it with {selector}='{method}'."
            )
            return out

    return None


def _enriched_unknown_method_error(agent_name, method, allowed_agents):
    """Build a recovery-friendly error listing each real method's brief.

    Returned when a guessed method cannot be resolved. Instead of just naming
    the available methods (which invites another guess), this pairs each with
    its one-line ``brief`` from ``METHOD_INFO`` so the agent can pick the
    closest one in a single reliable step.

    Returns
    -------
    dict
        ``{"error": ..., "available_methods": {method: brief, ...},
        "directive": ...}``. Falls back to ``None`` if the adapter / method
        info cannot be loaded, so the caller keeps the original bare error.
    """
    try:
        mod = _load_adapter(agent_name)
        briefs = {
            m: info.get("brief", "")
            for m, info in mod.METHOD_INFO.items()
            if not info.get("alias_of")
        }
    except Exception:
        return None
    if not briefs:
        return None
    return {
        "error": f"Unknown method '{method}' for module '{agent_name}'.",
        "available_methods": briefs,
        "directive": (
            "These are the available methods and what they do; pick the "
            "closest one. Theory/qualifier names (e.g. vesic, ultimate) are "
            "parameter values, not methods."
        ),
    }


class _CallAgentArgs(BaseModel):
    """Args schema for the call_agent tool.

    ``extra="allow"`` is the key piece: LLMs frequently flatten the method
    parameters to the top level instead of nesting them under ``parameters``.
    A plain ``**kwargs`` function signature does NOT survive
    ``StructuredTool.from_function`` (it collapses to a single ``extra`` dict
    field), so we declare the schema explicitly and let pydantic keep the
    extras — they flow into the function's ``**extra`` and get auto-nested,
    mirroring ``native_tools.dispatch_native_tool``.
    """

    model_config = ConfigDict(extra="allow")

    agent_name: str = Field(description="Module name.")
    method: str = Field(description="Method name.")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Nested dict of method-specific inputs (all SI units). Example: "
            '{"width": 2.0, "depth": 1.5, "friction_angle": 30, '
            '"unit_weight": 18.0}. Use describe_method first to see the '
            "required keys."
        ),
    )


def make_core_tools(
    allowed_agents=None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
) -> list:
    """Build the 4 core dispatch tools bound to an ``allowed_agents`` scope.

    Parameters
    ----------
    allowed_agents : iterable of str, optional
        Whitelist of module names. If provided, modules outside this set are
        invisible to ``list_agents`` / ``list_methods`` / ``describe_method``
        and refused by ``call_agent`` (same semantics as v1
        ``dispatch.dispatch_tool``). ``None`` exposes the full registry.
    max_result_chars : int, optional
        Cap on the size of the catalog/doc tools' results (``list_agents`` /
        ``list_methods`` / ``describe_method``) fed back to the model (default
        ``8000``, mirroring v1's ``GeotechAgent._max_result_chars``). Results
        longer than this are truncated with a clear marker + a
        "search narrower" nudge. A value ``<= 0`` disables ALL truncation in
        this factory (including reference reads).
    reference_result_chars : int, optional
        Cap for ``call_agent`` results from REFERENCE modules, where the
        reference text is the payload. Defaults to the larger
        ``DEFAULT_REFERENCE_RESULT_CHARS`` (16000; never below
        ``max_result_chars``). ``call_agent`` results from numeric/calc
        modules are NOT capped at all — they are tiny.

    Returns
    -------
    list[StructuredTool]
        ``[list_agents, list_methods, describe_method, call_agent]``.
    """
    reference_cap = _resolve_reference_cap(max_result_chars,
                                           reference_result_chars)

    def list_agents() -> str:
        """List all available geotechnical analysis modules with brief
        descriptions."""
        return _truncate(
            json.dumps(_list_agents(allowed_agents=allowed_agents), default=str),
            max_result_chars,
        )

    def list_methods(agent_name: str, category: str = "") -> str:
        """List available methods for a specific analysis module.

        ``agent_name`` is the module name (e.g. 'bearing_capacity',
        'settlement', 'subsurface'). ``category`` is an optional category
        filter; empty string for all.
        """
        return _truncate(
            json.dumps(
                _list_methods(
                    agent_name=agent_name,
                    category=category or "",
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            max_result_chars,
        )

    def describe_method(agent_name: str, method: str) -> str:
        """Get full parameter documentation for a method. Always call this
        before using a method for the first time.

        ``agent_name`` is the module name; ``method`` is the method name
        within that module.

        If ``method`` is a guessed name that is not real (e.g. a theory name
        like 'vesic' or a selector value), this resolves/redirects it to the
        module's real method — adding a ``_note`` explaining the mapping —
        the same way ``call_agent`` already auto-resolves such guesses. When it
        truly cannot resolve, it returns an enriched error listing each
        available method's brief so recovery is one reliable step, not a guess.
        """
        result = _describe_method(
            agent_name=agent_name,
            method=method,
            allowed_agents=allowed_agents,
        )
        # Happy path (real method, or an Unknown-MODULE / scope error): return
        # as-is. Only an "Unknown method" error (the module is visible but the
        # method name is wrong) is worth resolving — an Unknown-module error
        # means the module is out of scope and must stay refused.
        err = result.get("error", "") if isinstance(result, dict) else ""
        if "Unknown method" in err:
            resolved = _resolve_describe_method(
                agent_name, method, allowed_agents
            )
            if resolved is not None:
                result = resolved
            else:
                enriched = _enriched_unknown_method_error(
                    agent_name, method, allowed_agents
                )
                if enriched is not None:
                    result = enriched
        return _truncate(json.dumps(result, default=str), max_result_chars)

    def call_agent(
        agent_name: str,
        method: str,
        parameters: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> str:
        """Execute a geotechnical calculation. All method-specific inputs
        (width, depth, friction_angle, unit_weight, etc.) MUST go inside the
        'parameters' object (all SI units: meters, kPa, kN, kN/m3, degrees).
        Use describe_method first to see the required keys.

        ``agent_name`` is the module name; ``method`` is the method name.
        ``parameters`` is a nested dict of method-specific inputs, e.g.
        ``{"width": 2.0, "depth": 1.5, "friction_angle": 30,
        "unit_weight": 18.0}``.
        """
        # LLMs sometimes flatten method parameters to the top level instead of
        # nesting them under "parameters". Auto-nest any extra keys — mirrors
        # native_tools.dispatch_native_tool's call_agent handling.
        params = dict(parameters or {})
        extras = {k: v for k, v in extra.items() if k not in _CALL_AGENT_KEYS}
        if extras:
            params.update(extras)
        # Smart per-tool budget: reference modules get the LARGER reference
        # cap (their text is the payload); calc modules are uncapped (tiny).
        cap = _result_cap_for_module(agent_name, max_result_chars,
                                     reference_cap)
        return _truncate(
            json.dumps(
                _call_agent(
                    agent_name=agent_name,
                    method=method,
                    parameters=params,
                    attachments=None,
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            cap,
        )

    return [
        StructuredTool.from_function(
            list_agents,
            name="list_agents",
            description=(
                "List all available geotechnical analysis modules with "
                "brief descriptions."
            ),
        ),
        StructuredTool.from_function(
            list_methods,
            name="list_methods",
            description=(
                "List available methods for a specific analysis module."
            ),
        ),
        StructuredTool.from_function(
            describe_method,
            name="describe_method",
            description=(
                "Get full parameter documentation for a method. Always call "
                "this before using a method for the first time."
            ),
        ),
        StructuredTool.from_function(
            call_agent,
            name="call_agent",
            description=(
                "Execute a geotechnical calculation. All method-specific "
                "inputs (width, depth, friction_angle, unit_weight, etc.) "
                "MUST go inside the 'parameters' object — never as top-level "
                "arguments. All units are SI."
            ),
            args_schema=_CallAgentArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# Vision / file-output tools (wrap vision_tools.dispatch_extended_tool)
# ---------------------------------------------------------------------------

def make_vision_tools(
    engine=None,
    attachments: Optional[Dict[str, bytes]] = None,
    save_fn: Optional[Callable] = None,
    include: Optional[set] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
) -> list:
    """Build the vision / file-output tools as LangChain tools.

    The engine, attachments, and save_fn are closured in at build time so the
    tools match the no-extra-args calling convention deepagents expects.

    Parameters
    ----------
    engine : GenAIEngine, optional
        Vision-capable engine (``analyze_image``). Required for
        ``analyze_image`` / ``analyze_pdf_page`` / ``read_reference_figure``;
        when ``None`` those tools return a clear "vision not available" error
        instead of raising (so offline construction still works).
    attachments : dict, optional
        ``{key: bytes}`` of attached files (for ``analyze_image`` /
        ``analyze_pdf_page``). May be a live reference the host mutates.
    save_fn : callable, optional
        ``(path, content) -> saved_path``. Defaults to local filesystem write.
    include : set of str, optional
        Subset of tool names to build. Defaults to all four
        (``analyze_image``, ``analyze_pdf_page``, ``read_reference_figure``,
        ``save_file``).
    max_result_chars : int, optional
        Cap on the size of each tool's result fed back to the model (default
        ``8000``, mirroring v1). Results longer than this are truncated with a
        clear marker + a "search narrower" nudge; a value ``<= 0`` disables
        truncation. Applied to every vision/file tool via the shared
        ``_dispatch`` helper.
    reference_result_chars : int, optional
        Larger cap for ``read_reference_figure`` (a REFERENCE read — the chart
        read-off text is the payload). Defaults to
        ``DEFAULT_REFERENCE_RESULT_CHARS`` (16000; never below
        ``max_result_chars``); disabled when ``max_result_chars <= 0``.

    Returns
    -------
    list[StructuredTool]
    """
    attachments = attachments if attachments is not None else {}
    save_fn = save_fn or _default_save_fn
    include = include if include is not None else {
        "analyze_image", "analyze_pdf_page", "read_reference_figure",
        "save_file",
    }
    reference_cap = _resolve_reference_cap(max_result_chars,
                                           reference_result_chars)

    def _dispatch(tool_name: str, arguments: dict) -> str:
        # read_reference_figure is a reference read: larger budget.
        cap = (reference_cap if tool_name == "read_reference_figure"
               else max_result_chars)
        return _truncate(
            _dispatch_extended_tool(
                tool_name=tool_name,
                arguments=arguments,
                engine=engine,
                attachments=attachments,
                save_fn=save_fn,
            ),
            cap,
        )

    def analyze_image(attachment_key: str,
                      prompt: str = "Describe this image.") -> str:
        """Analyze an attached image using vision. Returns text
        description/analysis of the image content.

        ``attachment_key`` is the key of the attached image file; ``prompt``
        is what to extract or analyze from the image.
        """
        return _dispatch(
            "analyze_image",
            {"attachment_key": attachment_key, "prompt": prompt},
        )

    def analyze_pdf_page(
        attachment_key: str,
        page: int = 0,
        prompt: str = "Describe the content of this page.",
    ) -> str:
        """Render a PDF page and analyze it using vision.

        ``attachment_key`` is the key of the attached PDF file; ``page`` is the
        0-indexed page number; ``prompt`` is what to extract from the page.
        """
        return _dispatch(
            "analyze_pdf_page",
            {"attachment_key": attachment_key, "page": page, "prompt": prompt},
        )

    def read_reference_figure(
        reference: str,
        figure_number: str,
        prompt: str = "",
    ) -> str:
        """Render a digitized reference figure (e.g. a DM7 design chart) and
        read a value off it with vision. Use this whenever a numeric value must
        come from a chart — do not read values off a chart from the caption or
        from memory. Find the figure first with ``call_agent`` →
        ``figure_db.figure_search``, then pass its ``reference`` +
        ``figure_number`` here with a ``prompt`` describing the value(s) you
        need. Returns a chart read-off estimate — verify it against a
        closed-form/digitized method where one exists.
        """
        return _dispatch(
            "read_reference_figure",
            {
                "reference": reference,
                "figure_number": figure_number,
                "prompt": prompt,
            },
        )

    def save_file(path: str, content: str, encoding: str = "text") -> str:
        """Save raw text or data to a file. Returns the saved file path. For
        formatted calculation documents, use the ``calc_package`` module via
        ``call_agent`` instead.

        ``path`` is the output file path; ``content`` is the file content
        (text or base64); ``encoding`` is 'text' (default) or 'base64' for
        binary.
        """
        return _dispatch(
            "save_file",
            {"path": path, "content": content, "encoding": encoding},
        )

    _builders = {
        "analyze_image": (
            analyze_image,
            "Analyze an attached image using vision. Returns text "
            "description/analysis of the image content.",
        ),
        "analyze_pdf_page": (
            analyze_pdf_page,
            "Render a PDF page and analyze it using vision.",
        ),
        "read_reference_figure": (
            read_reference_figure,
            "Render a digitized reference figure (e.g. a DM7 design chart) "
            "and read a value off it with vision. Find the figure first via "
            "figure_db.figure_search, then read the value off the actual "
            "chart. Returns a chart read-off estimate.",
        ),
        "save_file": (
            save_file,
            "Save raw text or data to a file. Returns the saved file path. "
            "For formatted calculation documents, use the calc_package module "
            "via call_agent instead.",
        ),
    }

    tools = []
    for name, (fn, desc) in _builders.items():
        if name in include:
            tools.append(
                StructuredTool.from_function(fn, name=name, description=desc)
            )
    return tools


__all__ = [
    "make_core_tools",
    "make_vision_tools",
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_REFERENCE_RESULT_CHARS",
    "SEARCH_NARROWER_NUDGE",
]
