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

The vision / file-output tools are built by the vision kit
(:mod:`funhouse_agent.deep.vision_kit`, which both pages use; its names are
re-exported here). This module's :func:`make_vision_tools` adds the geotech
page's two reference readers, ``read_reference_figure`` and
``view_worked_example_source``, to the kit's tools.
"""

import json
import os
from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import StructuredTool

from funhouse_agent import document_tools as _document_tools
from funhouse_agent.deep.fit_results import (fit_catalog, fit_method_choices,
                                             fit_method_doc, fit_method_list)
from funhouse_agent.deep.strict_args import strict_tool, strict_tools
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from funhouse_agent.dispatch import (
    REFERENCE_MODULES,
    call_agent as _call_agent,
    describe_method as _describe_method,
    list_agents as _list_agents,
    list_methods as _list_methods,
    parameters_object as _parameters_object,
    # Resolution primitives (underscore-prefixed but importable). Used to make
    # describe_method redirect a guessed method name the same way call_agent
    # already does, instead of bouncing with a bare "Unknown method".
    _METHOD_ALIASES,
    _closest_methods,
    _cross_module_redirect,
    _load_adapter,
    _selector_value_candidates,
)
# The vision kit's names, re-exported (INF-30 step 4: the kit moved to
# funhouse_agent.deep.vision_kit); make_vision_tools is below.
from funhouse_agent.deep import vision_kit as _kit
from funhouse_agent.deep.vision_kit import (  # noqa: F401
    ANNOTATE_DXF_DESCRIPTION,
    DEFAULT_MAX_RESULT_CHARS,
    DEFAULT_REFERENCE_RESULT_CHARS,
    DEFAULT_VISION_RESULT_CHARS,
    IMAGE_FIND_NOTE,
    IMAGE_LOOK_NOTE,
    IMAGE_MARKUP_NOTE,
    IMAGE_ZOOM_NOTE,
    KEPT_READINGS_DESCRIPTION,
    MIN_SCORE_UNUSED_NOTE,
    NO_ZOOM_NOTE,
    PLOT_DATA_REPLACES_NOTE,
    SEARCH_NARROWER_NUDGE,
    TURN_ARG_NOTE,
    WRITE_XLSX_DESCRIPTION,
    _DOCUMENT_TOOL_NOTES,
    _PDF_PAGES_NOTE,
    _PDF_PAGE_NOTE,
    _PLOT_APP_NOTE,
    _app_document_description,
    _default_save_fn,
    _dispatch_extended_tool,
    _document_tool_schemas,
    _docx_available,
    _dxf_notes_available,
    _find_like_available,
    _json_with,
    _plot_available,
    _plot_data_retiring,
    _plot_description,
    _plot_takes_chart,
    _resolve_reference_cap,
    _shown_as,
    _truncate,
    _with_saved_note,
    _without_argument,
    _without_zoom_tool,
    _write_xlsx,
    _xlsx_available,
)


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

    # (a2) A text-tool name on a reference module ('search', 'text_search',
    # 'Text Retrieval') -> that module's own search / section method (G13).
    try:
        from funhouse_agent.dispatch import _text_tool_for
        text_tool = _text_tool_for(_load_adapter(agent_name), method)
    except Exception:
        text_tool = None
    if text_tool is not None:
        docs = _describe_method(agent_name=agent_name, method=text_tool,
                                allowed_agents=allowed_agents)
        if "error" not in docs:
            out = dict(docs)
            out["_note"] = (f"'{method}' is not a method name — this "
                            f"module's text tool is '{text_tool}'. Showing "
                            "its docs.")
            return out

    # (b) A method of ANOTHER module (e.g. an apparent-pressure envelope asked
    # of retaining_walls): describing is read-only, so show that method's docs
    # and say where it lives — call_agent still refuses to run it here.
    try:
        redirect = _cross_module_redirect(agent_name, method, allowed_agents)
    except Exception:
        redirect = None
    if redirect is not None:
        right_agent, right_method = redirect
        docs = _describe_method(
            agent_name=right_agent, method=right_method,
            allowed_agents=allowed_agents,
        )
        if "error" not in docs:
            out = dict(docs)
            out["_note"] = (
                f"'{method}' is not a '{agent_name}' method — it lives on "
                f"module '{right_agent}' as '{right_method}'. Showing that "
                f"method's docs; call it with call_agent('{right_agent}', "
                f"'{right_method}', {{...}})."
            )
            return out

    # (c) Selector value: ``method`` is an allowed VALUE of a selector parameter
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

    # (d) A guess that can only mean ONE method: the module's only method,
    # or the one method whose name contains the guess (or all its words) --
    # 'infinite_slope' -> 'infinite_slope_fos', 'basal_heave' ->
    # 'check_basal_heave'. Describing is read-only, so it is answered with
    # that method's docs (live smoke wave 1, G12: 33 of 106 questions spent
    # a model call on a guessed describe_method name).
    only = _unique_method_match(mod, method)
    if only is not None:
        docs = _describe_method(
            agent_name=agent_name, method=only, allowed_agents=allowed_agents,
        )
        if "error" not in docs:
            out = dict(docs)
            out["_note"] = (
                f"'{method}' is not a '{agent_name}' method name; the one "
                f"method it can mean is '{only}'. Showing its docs; call "
                f"call_agent('{agent_name}', '{only}', {{...}}).")
            return out

    return None


def _unique_method_match(mod, guess) -> Optional[str]:
    """The one listed method ``guess`` can mean, else ``None``: the module's
    only method, or the single method whose name contains the guess or all
    of its words."""
    from funhouse_agent.dispatch import _name_tokens
    names = [m for m, info in mod.METHOD_INFO.items()
             if not info.get("alias_of")]
    if len(names) == 1:
        return names[0]
    g = str(guess or "").strip().lower().replace("-", "_").replace(" ", "_")
    if len(g) < 4:
        return None
    contains = [m for m in names if g in m.lower()]
    if len(contains) == 1:
        return contains[0]
    if contains:
        return None
    words = _name_tokens(g)
    if not words:
        return None
    by_words = [m for m in names if words <= _name_tokens(m)]
    return by_words[0] if len(by_words) == 1 else None


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
        closest = _closest_methods(mod, method)
    except Exception:
        return None
    if not briefs:
        return None
    # The closest names come FIRST: a large module's brief list can be cut by
    # the result cap, and the nearest real methods must survive the cut.
    return {
        "error": (f"Unknown method '{method}' for module '{agent_name}'. "
                  f"Closest: {closest}."),
        "closest": closest,
        "directive": (
            "Describe the closest method that fits, or pick from the full "
            "list below. Theory/qualifier names (e.g. vesic, ultimate) are "
            "parameter values, not methods."
        ),
        "available_methods": briefs,
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

    ``parameters`` is shown to the model as an object but validated as
    anything (GM-189): a string reaches ``call_agent``, which reads a JSON
    object out of it or refuses it with the shape expected and the
    method's parameter names. Pydantic's own refusal ("Input should be a
    valid dictionary", with the whole argument echoed back) said neither.
    """

    model_config = ConfigDict(extra="allow")

    agent_name: str = Field(description="Module name.")
    method: str = Field(description="Method name.")
    parameters: Annotated[Any, WithJsonSchema({
        "anyOf": [{"additionalProperties": True, "type": "object"},
                  {"type": "null"}]})] = Field(
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
    attachments: Optional[Dict[str, bytes]] = None,
) -> list:
    """Build the 4 core dispatch tools bound to an ``allowed_agents`` scope.

    ``attachments`` is the conversation's ``{key: bytes}`` (the live dict the
    host mutates): ``call_agent`` hands it to the dispatcher so a method
    given ``attachment_key`` (``parse_diggs``...) reads the uploaded file.
    Until 2026-10-09 it passed ``None`` and the key never resolved (live
    smoke wave 1, A8); a key naming a file in the working folder resolves
    either way.

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
        # Fitted by structure, never cut mid-JSON (live smoke wave 1, G1).
        return fit_catalog(_list_agents(allowed_agents=allowed_agents),
                           max_result_chars)

    def list_methods(agent_name: str = "", category: str = "") -> str:
        """List available methods for a specific analysis module.

        ``agent_name`` is the module name (e.g. 'bearing_capacity',
        'settlement', 'subsurface'); left empty, the answer is the list of
        modules to pick from. ``category`` is an optional category filter;
        empty string for all.
        """
        return fit_method_list(
            _list_methods(
                agent_name=agent_name,
                category=category or "",
                allowed_agents=allowed_agents,
            ),
            max_result_chars, agent_name)

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
        # Never cut mid-JSON: a long doc keeps every parameter and shortens
        # its prose; a long method list drops briefs (live smoke wave 1, G1).
        if isinstance(result, dict) and "available_methods" in result:
            return fit_method_choices(result, max_result_chars, agent_name,
                                      result.get("closest") or ())
        # A method's own documentation is a reference read: it gets the
        # reference budget, so the eight methods whose docs run past 8,000
        # characters (six slope methods, downdrag's two) keep their prose
        # (BACKLOG GM-16). 0 = no limit.
        doc_cap = (0 if max_result_chars <= 0
                   else max(max_result_chars, reference_cap or 0))
        return fit_method_doc(result, doc_cap)

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
        # native_tools.dispatch_native_tool's call_agent handling. Any shape
        # but an object is left for the dispatcher to refuse (GM-189).
        params = _parameters_object(parameters)
        extras = {k: v for k, v in extra.items() if k not in _CALL_AGENT_KEYS}
        if extras and isinstance(params, dict):
            params = {**params, **extras}
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
                    attachments=attachments,
                    allowed_agents=allowed_agents,
                ),
                default=str,
            ),
            cap,
        )

    # Every tool refuses an argument it does not take, by name (brief 5,
    # N3); call_agent's schema takes extra keys on purpose and is unchanged.
    return strict_tools([
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
                "arguments. All units are SI. A result's 'choices' say what "
                "the module chose (given, default or module) and "
                "'defaults_used' the declared defaults the run used."
            ),
            args_schema=_CallAgentArgs,
        ),
    ])


# ---------------------------------------------------------------------------
# The vision kit with the geotech page's two reference readers added
# ---------------------------------------------------------------------------

def _read_reference_figure_tool(dispatch):
    """``read_reference_figure`` over the vision kit's ``dispatch``."""
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
        need. Returns a chart read-off estimate and, where code can find the
        chart's axes and curve in the drawing, a measured value beside it
        (``code_reading``) — verify against a closed-form/digitized method
        where one exists.
        """
        return dispatch(
            "read_reference_figure",
            {
                "reference": reference,
                "figure_number": figure_number,
                "prompt": prompt,
            },
        )
    return read_reference_figure, (
        "Render a digitized reference figure (e.g. a DM7 design chart) "
        "and read a value off it with vision. Find the figure first via "
        "figure_db.figure_search, then read the value off the actual "
        "chart. Returns the vision read-off estimate and, where the "
        "chart's axes and curve can be found in the drawing, a value "
        "MEASURED from it with its +/- beside the estimate "
        "(code_reading; flagged where the two disagree)."
    )


def _view_worked_example_source_tool(dispatch):
    """``view_worked_example_source`` over the vision kit's ``dispatch``."""
    def view_worked_example_source(
        example_id: str,
        pdf_page: int = 0,
        prompt: str = "",
    ) -> str:
        """Render a worked example's PRINTED SOURCE PAGE (design-report
        example, incl. its figures) and analyze it with vision. Use this after
        ``find_worked_examples``/``get_worked_example`` when following an
        exemplar whose entry lists ``source_pdf_pages`` — seeing the printed
        page (chart, cross-section, tabulated steps) beats re-deriving it from
        the text summary. ``pdf_page`` is 1-based; omit (0) for the first
        catalogued page; ``prompt`` says what to extract. Page numbers were
        located by text search and may be off by one — page around if needed.
        Values read off charts are estimates; verify against a digitized
        method where one exists.
        """
        args = {"example_id": example_id, "prompt": prompt}
        if pdf_page:
            args["pdf_page"] = pdf_page
        return dispatch("view_worked_example_source", args)
    return view_worked_example_source, (
        "Render a worked example's PRINTED SOURCE PAGE (design-report "
        "example incl. figures) and analyze it with vision. Use after "
        "find_worked_examples when the entry lists source_pdf_pages; "
        "pdf_page is 1-based (0 = first catalogued page)."
    )


#: The geotech page's reference readers, added to the vision kit: they
#: render a reference library figure and a worked example's printed
#: source page, which the Document Review page does not have (DR-04).
GEOTECH_VISION_TOOLS = {
    "read_reference_figure": _read_reference_figure_tool,
    "view_worked_example_source": _view_worked_example_source_tool,
}


def make_vision_tools(*args, **kwargs) -> list:
    """The vision kit's tools with the geotech page's two reference readers
    (:data:`GEOTECH_VISION_TOOLS`) added: the arguments of
    :func:`funhouse_agent.deep.vision_kit.make_vision_tools`, whose
    ``added_tools`` defaults here to those readers. The Document Review
    page builds the kit's own."""
    kwargs.setdefault("added_tools", GEOTECH_VISION_TOOLS)
    return _kit.make_vision_tools(*args, **kwargs)


__all__ = [
    "make_core_tools",
    "make_vision_tools",
    "make_report_ingest_tool",
    "make_report_library_tool",
    "REPORT_INGEST_DESCRIPTION",
    "REPORT_LIBRARY_DESCRIPTION",
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_REFERENCE_RESULT_CHARS",
    "SEARCH_NARROWER_NUDGE",
]


# ---------------------------------------------------------------------------
# the whole-report ingest (report_ingest), as ONE primary tool
# ---------------------------------------------------------------------------

#: What the model is told the tool does. Deliberately about the SHAPE of the
#: job -- a whole report, not a lookup -- because the failure this tool exists
#: to prevent is the primary agent reading a 400-page report page by page.
REPORT_INGEST_DESCRIPTION = (
    "Read a WHOLE geotechnical report PDF into one organised, cited record: "
    "the standing questions about the report (what it is, who wrote it, how "
    "many borings, what foundations and bearing pressures were recommended, "
    "site class, seismic code, natural hazards), every boring and test pit "
    "log as data, every laboratory test, and a note of what could not be "
    "read. Writes the record, a one-page summary, a library page and a DIGGS "
    "2.6 file, and returns counts, the paths and the first lines of the "
    "summary -- not the whole record. USE THIS for any whole-report question; "
    "do not read a long report page by page with the document tools. "
    "'questions' is anything else you want asked of the report, one per line."
)

#: The tier the ingest's page-label vision voter runs on. It looks at every
#: page of the report, so it goes on the cheapest tier the deployment has --
#: about $0.05 a report, against $0.45 for the label review it now replaces
#: on the pages the voters agree about.
VISION_TIER = "funhouse-gpt-low"


def make_report_ingest_tool(
    engine=None,
    attachments: Optional[Dict[str, bytes]] = None,
    out_dir: Optional[str] = None,
    budgets=None,
    db_path: Optional[str] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    label_policy: str = "structural",
    review_mode: str = "disagreements",
    max_model_calls: Optional[int] = None,
    checklist: bool = False,
) -> list:
    """The ``report_ingest`` primary tool, or an empty list.

    Empty when the installed planlens cannot serve the ingest: it needs the
    page roles and the log grid, which arrived in planlens 0.5. The cluster
    installs planlens from PyPI and has resolved older than the pin before,
    so this is checked against the INSTALLED package's own published specs
    rather than trusted from a version number — the same rule the document
    tools follow.

    ``engine`` is whatever the host built the agent with; the page's chat
    model inside it (on Foundry, BACKLOG RI-03), or a live Prompter, becomes
    the ingest's engine through :func:`report_ingest.engine.engine_for`.
    With no engine to be found the tool is still built and says so when
    called, which is a better answer than a missing tool the model then
    invents a workaround for.

    ``label_policy`` and ``review_mode`` are the ingest's page vote, passed
    straight through to :func:`report_ingest.graph.ingest_report`. The
    defaults are its own: three cheap voters on every page, and the
    expensive label review shown only the pages they split on.

    ``max_model_calls`` is one run's ceiling on model calls, the same one
    the sub-agent holds (:class:`report_ingest.subagent.CallCeiling`;
    ``None``: no ceiling beyond the readers' own budgets). ``checklist``
    runs the report review checklist's code checks on the record written
    and returns them in the answer (BACKLOG RI-35).
    """
    if not _document_tools.report_ingest_supported():
        return []

    attachments = attachments if attachments is not None else {}

    def report_ingest(source: str, questions: str = "") -> str:
        from report_ingest.engine import engine_for
        from report_ingest.subagent import run_ingest

        try:
            resolved = _document_tools.resolve_document_source(
                source, attachments)
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"{type(exc).__name__}: {exc}",
                               "hint": "give the attachment key of the "
                                       "uploaded PDF or a real file path"})
        ingest_engine = engine_for(engine)
        # The page-label vote's second voter looks at EVERY page, so it runs
        # on the cheapest tier the deployment has rather than on the tier
        # the readers use; on a host whose engine is not a Prompter (the
        # page's chat model) this is None and the graph falls back to the
        # one engine it has.
        vision_engine = engine_for(engine, model=VISION_TIER)
        if ingest_engine is None:
            return json.dumps({
                "error": "no ingest engine is configured in this deployment",
                "hint": "the ingest runs on the app's own model; tell the "
                        "user it is unavailable here rather than reading the "
                        "report another way"})
        folder = out_dir or report_ingest_out_dir(source, resolved)
        asked = [line.strip() for line in str(questions or "").splitlines()
                 if line.strip()]
        try:
            answer = run_ingest(resolved, asked, engine=ingest_engine,
                                out_dir=folder, budgets=budgets,
                                db_path=db_path,
                                label_policy=label_policy,
                                review_mode=review_mode,
                                vision_engine=vision_engine,
                                report_id=os.path.splitext(
                                    os.path.basename(str(source)))[0],
                                max_model_calls=max_model_calls,
                                checklist=checklist)
        except Exception as exc:  # noqa: BLE001 - one report, not the turn
            return json.dumps({"error": f"the ingest failed: "
                                        f"{type(exc).__name__}: {exc}"})
        return _truncate(json.dumps(answer.model_dump()), max_result_chars)

    report_ingest.__doc__ = REPORT_INGEST_DESCRIPTION
    return [strict_tool(StructuredTool.from_function(
        report_ingest, name="report_ingest",
        description=REPORT_INGEST_DESCRIPTION))]


#: The folder of the SHARED report library, when the deployment has one. The
#: ingest then writes each report's folder here (``<library>/<file name>/``)
#: whichever conversation reads it, and the report_library tool reads here.
REPORT_LIBRARY_ENV = "GEOTECH_REPORT_LIBRARY"


def shared_report_library() -> str:
    """The shared library's folder from ``GEOTECH_REPORT_LIBRARY`` (``~``
    expanded, made absolute), or "" when none is set."""
    value = (os.environ.get(REPORT_LIBRARY_ENV) or "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else ""


def report_ingest_out_dir(source: str, resolved: Any = None) -> str:
    """Where a report's outputs go: ``<file name>-<8 hex of the file's
    hash>`` in the shared library (``GEOTECH_REPORT_LIBRARY``) when one is
    set, else ``report_ingest/<file name>`` in the conversation's own folder
    (the working folder bound to the turn), for the tool and the sub-agent
    alike.

    The hash is the one the record's own key is made of
    (``report_ingest.writers.source_digest``, of the readable file
    ``resolved``, else of ``source``), so two different PDFs sharing a file
    name get two folders in the shared library while the same file read
    again resumes in its own. Where the file cannot be read to hash it, the
    folder is the bare ``<file name>``.
    """
    from funhouse_agent._fileio import default_output_dir

    stem = os.path.splitext(os.path.basename(str(source)))[0] or "report"
    shared = shared_report_library()
    if shared:
        from report_ingest.writers import source_digest
        try:
            digest = (source_digest(resolved) if resolved is not None
                      else "") or source_digest(source)
        except Exception:                            # noqa: BLE001
            digest = ""
        return os.path.join(shared, f"{stem}-{digest[:8]}" if digest
                            else stem)
    return os.path.join(default_output_dir(), "report_ingest", stem)


# ---------------------------------------------------------------------------
# the report LIBRARY (report_library), as ONE primary tool
# ---------------------------------------------------------------------------

#: What the model is told the library tool does. About the SHAPE of the job
#: again, and about the boundary with the ingest: this one answers from the
#: reports already read and cannot read a new one.
REPORT_LIBRARY_DESCRIPTION = (
    "Ask a question ACROSS the geotechnical reports that have already been "
    "read into records -- which reports belong to a post, what each one "
    "recommended, how a value compares between them, which report and page "
    "prints something, where two readings of one report disagree, what the "
    "library holds altogether. It answers from those records ONLY and cites "
    "the report id and the page behind every fact; where the library holds "
    "no answer it says so rather than guessing. It CANNOT read a new PDF: a "
    "report nobody has ingested is not in the library, and 'report_ingest' "
    "is what puts it there."
)


def make_report_library_tool(
    model=None,
    library_root: Optional[str] = None,
    db_path: Optional[str] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    max_tool_calls: Optional[int] = None,
) -> list:
    """The ``report_library`` primary tool, or an empty list.

    Empty when there is no library to ask: ``library_root`` unset, or a
    folder holding no ``report.record.json``. The feature detection is the
    FOLDER rather than a version, because a deployment either has reports
    read into it or does not, and advertising a library agent over an empty
    folder would only teach the primary to ask it things it cannot answer.

    ``model`` is the app's own chat model, which is what the library
    sub-agent reads query results with; without one the tool is still built
    and says so when called, the same choice the ingest tool makes.
    """
    from report_ingest.library_agent import library_available

    if not library_available(library_root):
        return []

    def report_library(question: str) -> str:
        from report_ingest.library_agent import (
            MAX_TOOL_CALLS, answer_as_json, answer_question,
        )

        if model is None:
            return json.dumps({
                "error": "no model is configured for the report library in "
                         "this deployment",
                "hint": "the library agent answers with the app's own model; "
                        "tell the user it is unavailable here rather than "
                        "answering the question from memory"})
        try:
            answer = answer_question(
                question, model=model, library_root=library_root,
                db_path=db_path,
                max_tool_calls=max_tool_calls or MAX_TOOL_CALLS)
        except Exception as exc:  # noqa: BLE001 - one question, not the turn
            return json.dumps({"error": f"the library query failed: "
                                        f"{type(exc).__name__}: {exc}"})
        return _truncate(answer_as_json(answer, max_result_chars),
                         max_result_chars)

    report_library.__doc__ = REPORT_LIBRARY_DESCRIPTION
    return [strict_tool(StructuredTool.from_function(
        report_library, name="report_library",
        description=REPORT_LIBRARY_DESCRIPTION))]
