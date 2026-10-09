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
import os
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import StructuredTool

from funhouse_agent import document_tools as _document_tools
from funhouse_agent.deep.fit_results import (fit_catalog, fit_method_choices,
                                             fit_method_doc, fit_method_list)
from funhouse_agent.deep.strict_args import strict_tool, strict_tools
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
    _closest_methods,
    _cross_module_redirect,
    _load_adapter,
    _selector_value_candidates,
)
from funhouse_agent.vision_tools import (
    dispatch_extended_tool as _dispatch_extended_tool,
    docx_available as _docx_available,
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

#: Cap for the two vision tools whose result IS a report over many images:
#: ``analyze_pdf_page`` with tiles (an overview + up to 16 tile readings) and
#: ``find_like`` (instances by page across a whole set). Owner, 2026-09-25:
#: fine to raise the caps for robustness. Each tool budgets itself just under
#: this (``vision_tools.TILED_RESULT_CHARS``, ``find_like.RESULT_BUDGET_CHARS``)
#: so a result is shortened as valid JSON, never cut mid-string here.
DEFAULT_VISION_RESULT_CHARS = 32000

#: Appended after the truncation marker so the agent does a NARROWER follow-up
#: search instead of re-requesting (and re-truncating) the same large item.
SEARCH_NARROWER_NUDGE = (
    " Do NOT re-request this same item — it will truncate again. Run a "
    "NARROWER follow-up search instead (more specific terms, or one specific "
    "section/table/figure) to retrieve just the part you need."
)


def _with_saved_note(result: str, save_fn) -> str:
    """Let the host annotate a successful ``save_file`` result.

    A host ``save_fn`` may expose ``saved_note(saved_path) -> str | None`` to
    say what its destination means for the user. The webapp uses it to state
    that a file written into the conversation folder is now attached to the
    chat — field feedback 2026-09-04 caught the agent saving reports there and
    telling the user in the same turn that it could not attach files, because
    nothing in the tool result said otherwise.

    Best-effort by design: no hook, a hook that raises, an errored save, or a
    result that is not parseable JSON (e.g. truncated) all leave ``result``
    exactly as it was.
    """
    hook = getattr(save_fn, "saved_note", None)
    if not callable(hook):
        return result
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return result
    if not isinstance(data, dict) or data.get("error") or not data.get("saved"):
        return result
    try:
        note = hook(data["saved"])
    except Exception:
        return result
    if not note:
        return result
    existing = data.get("note")
    data["note"] = f"{existing} {note}" if existing else str(note)
    return json.dumps(data)


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


def _find_like_available() -> bool:
    """Whether the installed planlens can search a drawing set for a mark on
    this host. planlens answers that itself once it has a matcher that needs
    no OpenCV (``planlens.document.findlike.available``: OpenCV where it
    loads, numpy anywhere else — the FIPS hosts, where loading OpenCV aborts
    the interpreter: Foundry 2026-10-03, Funhouse 2026-10-07). planlens 0.11
    has only OpenCV, so there the question is whether OpenCV loads (it
    test-loads it in a child process first); before 0.11 there is nothing to
    ask."""
    import importlib.util
    try:
        if importlib.util.find_spec("planlens.document.findlike") is None:
            return False
    except (ImportError, ValueError):
        return False
    try:
        from planlens.document.findlike import available
    except ImportError:          # planlens 0.11: OpenCV is the only matcher
        try:
            from planlens.opencv import available
        except ImportError:      # planlens before 0.11: no test-load to ask
            return True
    ok, why = available()
    if not ok:
        import logging
        logging.getLogger(__name__).info("find_like left out: %s", why)
    return ok


def _plot_available() -> bool:
    """Whether the data-plot module can be imported (matplotlib is a core
    app dependency, so this is a guard for a stripped library install)."""
    try:
        import profile_figure  # noqa: F401
    except Exception:                                  # noqa: BLE001
        return False
    return True


def _xlsx_available() -> bool:
    try:
        from funhouse_agent.xlsx_writer import available
    except Exception:                                  # noqa: BLE001
        return False
    return available()


#: What the app adds to the module's own description of ``plot_data``.
_PLOT_APP_NOTE = (
    " In this app the chart appears as an INTERACTIVE card under your reply "
    "(zoom and hover); the PNG it also saves is the copy for a report "
    "(html_img_tag) or a Word document (![title](<output_path>) in "
    "write_docx). Give output_path a bare file name, or leave it out. Never "
    "draw a chart by hand as SVG or as text.")


def _plot_description() -> str:
    """``plot_data``'s model-facing description, in the module's own words
    (``profile_figure`` METHOD_INFO), with the app's note."""
    try:
        from funhouse_agent.adapters.profile_figure_adapter import METHOD_INFO
        info = METHOD_INFO["plot_data"]
        params = info.get("parameters") or {}
        lines = [str(info.get("brief") or "").strip()]
        for key in ("series", "xlabel", "ylabel", "depth_axis", "logx",
                    "hlines", "vlines", "legend", "interactive"):
            desc = (params.get(key) or {}).get("description")
            if desc:
                lines.append(f"{key}: {desc}")
        return " ".join(lines) + _PLOT_APP_NOTE
    except Exception:                                  # noqa: BLE001
        return ("Plot one or more x/y data series (series: [{x: [...], y: "
                "[...], label}]) -- SPT or CPT vs depth (depth_axis=true), "
                "settlement vs time, a sweep, quantities from a document."
                + _PLOT_APP_NOTE)


#: ``write_xlsx``'s model-facing description.
WRITE_XLSX_DESCRIPTION = (
    "Write an Excel workbook (.xlsx), one sheet per table -- a schedule, a "
    "quantity take-off, a comment log, test results, a comparison table: "
    "anything the reader will sort, filter or calculate with. Give the "
    "tables EITHER as 'sheets' = [{name, rows}] with rows a list of lists "
    "(the first row is the header) or a list of objects (their keys become "
    "the header), OR as 'markdown' holding pipe tables (each table becomes "
    "its own sheet, named by the heading above it); both may be given. "
    "Numbers are written as numbers (an id with a leading zero stays text), "
    "the header row is bold and frozen, columns are sized to fit. 'path' is "
    "the file name (.xlsx is added); a bare name lands in the working folder "
    "and appears as a download card. Never save SpreadsheetML, an .xls or a "
    "CSV in its place when the user asked for Excel.")


def _write_xlsx(path: str, sheets, markdown: str, save_fn) -> str:
    """The ``write_xlsx`` tool: build the workbook, save it through the
    host's ``save_fn`` (so the web app records it and gives it a card), and
    name it by its conversation-relative name."""
    from funhouse_agent import xlsx_writer
    from funhouse_agent._fileio import (conversation_name, host_output_dir,
                                        into_working_folder,
                                        resolve_output_path)
    name = str(path or "").strip() or "tables.xlsx"
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".xlsx":
        name = (stem if ext.lower() in (".xls", ".csv", ".xml", ".txt")
                else name) + ".xlsx"
    try:
        data, summary, warnings = xlsx_writer.build_workbook(
            sheets=sheets if isinstance(sheets, list) else None,
            markdown=str(markdown or ""))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:                           # noqa: BLE001
        return json.dumps({"error": f"the workbook could not be built: "
                                    f"{type(exc).__name__}: {exc}"})
    target = (into_working_folder(name) if host_output_dir()
              else resolve_output_path(name))
    try:
        saved = save_fn(target, data)
    except Exception as exc:                           # noqa: BLE001
        return json.dumps({"error": f"the workbook could not be saved: "
                                    f"{type(exc).__name__}: {exc}"})
    saved_abs = os.path.abspath(str(saved or target))
    size = os.path.getsize(saved_abs) if os.path.isfile(saved_abs) else 0
    if size < len(data):
        return json.dumps({"error": f"the workbook was written but the file "
                                    f"holds {size} of {len(data)} bytes",
                           "saved": conversation_name(saved_abs)})
    out: Dict[str, Any] = {"saved": conversation_name(saved_abs),
                           "file_size_bytes": size, "sheets": summary}
    if warnings:
        out["warnings"] = warnings
    hook = getattr(save_fn, "saved_note", None)
    if callable(hook):
        try:
            note = hook(saved_abs)
        except Exception:                              # noqa: BLE001
            note = None
        if note:
            out["note"] = str(note)
    return json.dumps(out, default=str)


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
        return fit_method_doc(result, max_result_chars)

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
                "arguments. All units are SI."
            ),
            args_schema=_CallAgentArgs,
        ),
    ])


# ---------------------------------------------------------------------------
# Vision / file-output tools (wrap vision_tools.dispatch_extended_tool)
# ---------------------------------------------------------------------------

#: Appended to planlens' own description where THIS app changes the contract.
#: planlens is framework-neutral and cannot know where a file belongs or who
#: is signing the review; both are settled here, so the model is told.
_DOCUMENT_TOOL_NOTES = {
    "annotate_document": (
        " In this app give output_path a bare file name: it is written into "
        "this conversation's working folder, attached to the reply as a "
        "download, and it defaults to <document>_marked.pdf. The app signs "
        "every comment itself — the person using it, via this app, as an AI "
        "draft, which is what the reviewer must be able to see — so there is "
        "no author argument. For a thing you found by looking, pass the view and "
        "image_box of the ZOOMED look in which the thing is legible (a view "
        "of 300 pt or less; a whole-page look only says where to zoom, and "
        "a small mark from it is refused) — never a box from memory or an "
        "estimate. Every mark is then CHECKED, whatever its anchor (box, "
        "point, quote or note): a crop of the marked copy is looked at and "
        "the result's `check` says which marks are on the thing their "
        "target, label or quoted words name — or, naming none, the thing "
        "their comment is about — and which are misplaced. Never hand over "
        "a file with misplaced marks: find those things again and rewrite "
        "the copy with append=false."),
}


def _app_document_description(name: str) -> str:
    """A document tool's model-facing description: planlens' own words, with
    the app's note where the app changes the contract — and, for the two
    visual-scale tools, the app's own words (they take a look's view +
    image_box and read a scan's labels, which planlens' text cannot say)."""
    if name == "measure":
        from funhouse_agent.measure_tool import MEASURE_DESCRIPTION
        return MEASURE_DESCRIPTION
    if name == "log_grid":
        from funhouse_agent.measure_tool import log_grid_description
        return log_grid_description()
    return (_document_tools.tool_description(name)
            + _DOCUMENT_TOOL_NOTES.get(name, ""))


def make_vision_tools(
    engine=None,
    attachments: Optional[Dict[str, bytes]] = None,
    save_fn: Optional[Callable] = None,
    include: Optional[set] = None,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    reference_result_chars: Optional[int] = None,
    markup_author: Optional[str] = None,
    description_overrides: Optional[Dict[str, str]] = None,
    inline_images: bool = False,
    inline_image_files: bool = False,
) -> list:
    """Build the vision / file-output tools as LangChain tools.

    ``description_overrides`` replaces a tool's model-facing description by
    name — the lean Document Review agent describes its tools for the page it
    is on (:mod:`funhouse_agent.deep.review_agent`). ``inline_images`` makes
    ``analyze_pdf_page`` / ``render_region`` return the rendered image for the
    main model to look at instead of a one-shot vision call's description
    (only meaningful with the image middleware that shows it; see
    :mod:`funhouse_agent.deep.inline_images`). ``inline_image_files`` makes
    ``analyze_image`` do the same for an image FILE given by its path (a
    contact sheet); the lean agent sets it only with both
    ``GEOTECH_VISION_INLINE`` and ``GEOTECH_REVIEW_OVERVIEW`` on.

    The engine, attachments, and save_fn are closured in at build time so the
    tools match the no-extra-args calling convention deepagents expects.
    ``markup_author`` is who ``annotate_document`` signs comments as when the
    call names nobody — the signed-in person on a multi-user host (the webapp
    passes ``Identity.markup_author``); ``None`` falls back to the
    deployment-wide :func:`funhouse_agent.document_tools.markup_author`.

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
    if include is None:
        include = {
            "list_files", "read_pdf_text", "read_text_file", "analyze_image",
            "analyze_pdf_page", "render_region", "read_reference_figure",
            "view_worked_example_source", "save_file",
        }
        # Whole-document review tools (planlens.tools), when the installed
        # planlens has them — never advertised to the model otherwise.
        if _document_tools.available():
            include |= set(_document_tools.document_tool_names())
        # Word output, on the same rule: offered only where it can be produced.
        if _docx_available():
            include |= {"write_docx"}
        # A plot and a spreadsheet on BOTH pages (live smoke wave 1, A12: the
        # review page had no plot tool and neither page wrote Excel).
        if _plot_available():
            include |= {"plot_data"}
        if _xlsx_available():
            include |= {"write_xlsx"}
        # find_like needs planlens 0.10 (planlens.document.findlike).
        if _find_like_available():
            include |= {"find_like"}
    reference_cap = _resolve_reference_cap(max_result_chars,
                                           reference_result_chars)
    # Resolved once per build: the newer tools depend on the installed planlens.
    document_names = set(_document_tools.document_tool_names())

    def _dispatch(tool_name: str, arguments: dict) -> str:
        if tool_name in document_names:
            # Text-payload reads, budgeted by planlens itself: its limit sits
            # just under the reference cap, so results page through cursors
            # as valid JSON instead of being string-truncated here.
            return _truncate(
                _document_tools.dispatch_document_tool(
                    tool_name, arguments, attachments=attachments,
                    max_chars=_document_tools.budget_for_cap(reference_cap),
                    cap=reference_cap or None),
                reference_cap,
            )
        # read_reference_figure / read_pdf_text are text-payload reads and
        # list_files can be a long directory dump: the content IS the answer,
        # so they get the larger reference budget.
        if tool_name in ("analyze_pdf_page", "find_like"):
            cap = (0 if max_result_chars <= 0
                   else max(DEFAULT_VISION_RESULT_CHARS, reference_cap))
        elif tool_name in ("read_reference_figure", "read_pdf_text",
                           "read_text_file", "list_files",
                           "view_worked_example_source"):
            cap = reference_cap
        else:
            cap = max_result_chars
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

    def list_files(path: str = ".", max_entries: int = 200,
                   depth: int = 0) -> str:
        """List the files this conversation holds (read-only): each entry's
        name, type (dir/file), size and modified time.

        ``path`` defaults to ``.``, the conversation's working folder (the
        user's uploads, the files fetched or produced, and ``.scratch`` for
        tool scratch images); a relative path is inside it. The read tools
        reach only this conversation's files, the reference documents and any
        folder the deployment opens -- not the server's disk. The scratch
        filesystem's ``ls`` / ``read_file`` see none of these. ``max_entries``
        caps the count (default 200); ``depth`` descends sub-directories (0 =
        children only, max 2).
        """
        return _dispatch(
            "list_files",
            {"path": path, "max_entries": max_entries, "depth": depth},
        )

    def read_pdf_text(source: str, pages: str = "") -> str:
        """Extract the TEXT LAYER of a PDF (PyMuPDF — cheap, no vision). Use
        this FIRST for a text-based report (boring logs, lab summaries,
        recommendations, specs) instead of vision-reading every page.

        ``source`` is an attachment key or a file name in the working folder
        (a relative path is inside it). ``pages`` is an int, a list, or a "start-end" range like
        "0-9"; omit for the first several pages. A page with no text layer
        (scanned image) is flagged per-page — use analyze_pdf_page for those.
        """
        args = {"source": source}
        if pages != "":
            args["pages"] = pages
        return _dispatch("read_pdf_text", args)

    def read_text_file(path: str, offset: int = 0, max_chars: int = 6000) -> str:
        """Read a text file this conversation holds -- HTML, TXT, CSV, JSON,
        MD, such as a report source written earlier. The scratch filesystem's
        ``read_file`` cannot see it. ``path`` is a file name in the working
        folder (a relative path is inside it); long files page with
        ``offset`` (the result gives ``next_offset``)."""
        return _dispatch("read_text_file",
                         {"path": path, "offset": offset, "max_chars": max_chars})

    def analyze_image(attachment_key: str,
                      prompt: str = "Describe this image.") -> str:
        """Analyze an image using vision. Returns text description/analysis
        of the image content.

        ``attachment_key`` is an attached image's key, an image file's name
        in the working folder, or the name a tool returned for a scratch
        image (``.scratch/...`` -- render_page_thumbnails' contact sheets,
        find_like's sheets); ``prompt`` is what to extract or analyze.
        """
        args = {"attachment_key": attachment_key, "prompt": prompt}
        if inline_image_files:
            # An image FILE (a contact sheet) is shown to the main model like
            # the inline page tools' images; an upload keeps the vision call.
            args["_inline"] = True
        return _dispatch("analyze_image", args)

    def analyze_pdf_page(
        attachment_key: str,
        page: int = 0,
        prompt: str = "Describe the content of this page.",
        tiles: str = "auto",
    ) -> str:
        """Render a PDF page and analyze it using vision.

        ``attachment_key`` is the key of the attached PDF file; ``page`` is the
        0-indexed page number; ``prompt`` is what to extract from the page.
        ``tiles``: ``"auto"`` (default) ALSO reads the page in overlapping
        tiles when its small lettering is too small in the whole-page image
        (the result then carries every tile's reading and view); ``"off"``;
        or N / ``"NxN"`` with N 2-4 (``"3"`` or ``"3x3"``) for a fixed split.
        """
        args = {"attachment_key": attachment_key, "page": page,
                "prompt": prompt, "tiles": tiles}
        if inline_images:
            args["_inline"] = True
        return _dispatch("analyze_pdf_page", args)

    def find_like(
        attachment_key: str,
        page: int = 0,
        bbox: Optional[list] = None,
        text: str = "",
        pages: Optional[str] = None,
        include_legend: bool = False,
        view: Optional[list] = None,
        image_box: Optional[list] = None,
    ) -> str:
        """Find EVERY copy of one mark — a tag, a code, a symbol — across the
        document, including drawing sheets whose lettering is drawn as lines
        and cannot be searched as text; every candidate is then READ by vision
        at a large size, so look-alikes (GCG vs GCE) are rejected.

        First zoom (``render_region``) until ONE copy is legible — a legend
        row is a fine example — and confirm what it reads. Then pass that
        copy's ``bbox`` [x0, y0, x1, y1] in PDF points (tight round its
        lettering, no leader or table rule inside) — or the zoom result's
        ``view`` + the copy's 0-999 ``image_box`` — and ``text``, what it
        reads (e.g. ``"GCE"``). ``pages`` like ``"0-84"`` (default all).
        Returns the instances by page, callouts (a leader is drawn from the
        tag; ``points_to`` is where it points) apart from legend entries
        (``include_legend`` to list them), uncertain reads to zoom on, and
        contact sheets of the candidates, written to the conversation's
        scratch folder (``.scratch/...``, no download cards; look at one with
        analyze_image by the name given).
        Use it when the user wants EVERY occurrence of one repeated mark
        across many sheets; for anything else the reading and zoom tools are
        the way.
        """
        args = {"attachment_key": attachment_key, "page": page,
                "include_legend": include_legend}
        for k, v in (("bbox", bbox), ("text", text or None), ("pages", pages),
                     ("view", view), ("image_box", image_box)):
            if v is not None:
                args[k] = v
        return _dispatch("find_like", args)

    def render_region(
        attachment_key: str,
        page: int = 0,
        bbox: Optional[list] = None,
        marks: Optional[list] = None,
        prompt: str = "Describe what this zoomed-in region shows.",
        view: Optional[list] = None,
        image_box: Optional[list] = None,
    ) -> str:
        """Render a ZOOMED-IN crop of a PDF page and analyze it with vision —
        the "geometry says WHERE, vision says WHAT" primitive for drawings.

        Get exact coordinates first from the ``drawing_ir`` module
        (digitize_drawing → query_drawing — e.g. a leader's tip_xy, a title
        block's region_bbox), then zoom here to see WHAT is at that location.
        ``bbox`` is [x0, y0, x1, y1] in PDF points, TOP-LEFT origin with y
        DOWN (drawing_ir query coordinates are bottom-left/y-up — convert
        with y_pdf = page_height − y_ir, or use drawing_ir's ``snip_region``
        method, which converts for you and saves a PNG instead). Optional
        ``marks`` = [[x, y, label], ...] draws numbered circles at points of
        interest so the question becomes "what is mark 1 pointing at?".
        ``attachment_key`` is an attachment key or a real PDF path.

        To zoom on something an earlier vision result LOCATED, pass that
        result's ``view`` plus the 0-999 ``image_box`` its analysis gave
        (instead of ``bbox``); same ``attachment_key`` and ``page``. The
        window is the box padded by how far a box from that view can be off
        (a tenth of the view each way), so the thing is in it. Every
        vision result carries a ``view``. The zoom is always drawn as large
        as the vision model reads: to see more detail, zoom on a smaller
        box.
        """
        args = {"attachment_key": attachment_key, "page": page,
                "prompt": prompt}
        if bbox is not None:
            args["bbox"] = bbox
        if marks is not None:
            args["marks"] = marks
        if view is not None:
            args["view"] = view
        if image_box is not None:
            args["image_box"] = image_box
        if inline_images:
            args["_inline"] = True
        return _dispatch("render_region", args)

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
        return _dispatch(
            "read_reference_figure",
            {
                "reference": reference,
                "figure_number": figure_number,
                "prompt": prompt,
            },
        )

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
        return _dispatch("view_worked_example_source", args)

    def open_document(source: str) -> str:
        """Open a PDF for review; returns a handle and a map of the document."""
        return _dispatch("open_document", {"source": source})

    def document_structure(handle: str, offset: int = 0) -> str:
        """The constituent documents of a stapled PDF, with the page
        numbers printed on them."""
        return _dispatch("document_structure",
                         {"handle": handle, "offset": offset})

    def render_page_thumbnails(handle: str, pages: Any = None,
                               columns: int = 6) -> str:
        """Contact sheets of every page (thumbnail + page number + kind),
        written as PNG files to look at with analyze_image."""
        args = {"handle": handle, "columns": columns}
        if pages not in (None, ""):
            args["pages"] = pages
        return _dispatch("render_page_thumbnails", args)

    def document_page_map(handle: str, pages: Any = None, kind: str = "",
                          with_evidence: bool = False) -> str:
        """One row per page: kind, label, heading, counts."""
        args = {"handle": handle}
        if pages not in (None, ""):
            args["pages"] = pages
        if kind:
            args["kind"] = kind
        if with_evidence:
            args["with_evidence"] = True
        return _dispatch("document_page_map", args)

    def read_document(handle: str, pages: Any = None, start_line: int = 0,
                      with_locations: bool = False,
                      include_tables: bool = True,
                      include_markups: bool = True) -> str:
        """Read pages: text (optionally with boxes), tables, markups."""
        args = {"handle": handle, "start_line": start_line,
                "with_locations": with_locations,
                "include_tables": include_tables,
                "include_markups": include_markups}
        if pages not in (None, ""):
            args["pages"] = pages
        return _dispatch("read_document", args)

    def search_document(handle: str, pattern: str, pages: Any = None,
                        regex: bool = False, case_sensitive: bool = False,
                        include_markups: bool = True,
                        max_hits: int = 100, fuzzy: bool = False,
                        min_score: int = 80) -> str:
        """Find text, hidden CAD text and markup comments. ``fuzzy=True``
        matches approximately (score 0-100, best first) for text read
        optically or plotted as strokes; ``min_score`` defaults to 80."""
        if fuzzy and not _document_tools.search_supports_fuzzy():
            return json.dumps({
                "error": "the installed planlens does not support fuzzy "
                         "search (it arrived in planlens 0.4)",
                "hint": "search again without fuzzy; an exact miss on a "
                        "scan, a figure or a drawing sheet is not absence — "
                        "look at the page instead"})
        args = {"handle": handle, "pattern": pattern, "regex": regex,
                "case_sensitive": case_sensitive,
                "include_markups": include_markups, "max_hits": max_hits}
        if pages not in (None, ""):
            args["pages"] = pages
        if fuzzy:
            args["fuzzy"] = True
            args["min_score"] = min_score
        return _dispatch("search_document", args)

    def find_quantities(handle: str, pages: Any = None, kinds: Any = None,
                        units: Any = None, include_markups: bool = True,
                        offset: int = 0) -> str:
        """Every number WITH A UNIT the document states, with its wording,
        qualifier, page and box."""
        args = {"handle": handle, "include_markups": include_markups,
                "offset": offset}
        if pages not in (None, ""):
            args["pages"] = pages
        if kinds not in (None, "", [], ()):
            args["kinds"] = kinds
        if units not in (None, "", [], ()):
            args["units"] = units
        return _dispatch("find_quantities", args)

    def document_markups(handle: str, pages: Any = None, author: str = "",
                         offset: int = 0) -> str:
        """The review record: every markup with author, date and target."""
        args = {"handle": handle, "offset": offset}
        if pages not in (None, ""):
            args["pages"] = pages
        if author:
            args["author"] = author
        return _dispatch("document_markups", args)

    def annotate_document(handle: str, markups: Optional[list] = None,
                          output_path: str = "",
                          append: bool = True, check: bool = True) -> str:
        """Write review comments onto a COPY of the PDF (planlens' own words
        describe the markups; this app resolves the output file and signs
        them). ``check`` (default on) looks at every mark on the marked copy,
        however it was anchored, and reports which are misplaced.

        The signature is ALWAYS the app's (``markup_author``: the signed-in
        person via this app, as an AI draft), never the model's: brief 5
        caught an agent signing "AI Draft Review" unasked, which dropped the
        reviewer's name from every comment. So there is no ``author``
        argument."""
        args = {
            "handle": handle,
            "output_path": _document_tools.markup_output_path(output_path,
                                                              handle),
            "markups": markups or [],
            "author": markup_author or _document_tools.markup_author(),
            "append": append,
        }
        raw = _dispatch("annotate_document", args)
        if not check:
            return raw
        try:
            result = json.loads(raw)
        except ValueError:
            return raw                  # an error string, or a cut result
        out_pdf = result.get("output_path") if isinstance(result, dict) else None
        if not out_pdf or not os.path.isfile(out_pdf) or "error" in result:
            return raw
        try:
            from funhouse_agent import markup_check
            block = markup_check.check_marks(out_pdf, result,
                                             list(markups or []), engine)
        except Exception as exc:  # noqa: BLE001 - the file is written either way
            block = {"error": f"the placement check failed: "
                              f"{type(exc).__name__}: {exc}"}
        if block is None:
            return raw
        result["check"] = block
        return json.dumps(result)

    def measure(source: str, page: int = 0, kind: str = "line",
                bbox: Optional[list] = None, view: Optional[list] = None,
                image_box: Optional[list] = None,
                at: Optional[Dict[str, float]] = None,
                to: Optional[list] = None, scale: str = "",
                side: str = "top") -> str:
        """Measure a position through the page's own scale (planlens'
        ``measure``; the app converts a look's view + image_box and reads a
        scan's label values itself)."""
        from funhouse_agent.measure_tool import run_measure
        args: Dict[str, Any] = {"source": source, "page": page, "kind": kind,
                                "side": side}
        for k, v in (("bbox", bbox), ("view", view), ("image_box", image_box),
                     ("at", at), ("to", to), ("scale", scale or None)):
            if v is not None:
                args[k] = v
        return _truncate(
            run_measure(args, attachments, engine,
                        max_chars=_document_tools.budget_for_cap(reference_cap),
                        cap=reference_cap or None),
            reference_cap)

    def log_grid(handle: str, pages: Any = None, rows: bool = True,
                 offset: int = 0) -> str:
        """One boring or test-pit log as its grid (planlens' ``log_grid``;
        the app reads a textless scan's depth labels itself)."""
        from funhouse_agent.measure_tool import run_log_grid
        args: Dict[str, Any] = {"handle": handle, "rows": rows,
                                "offset": offset}
        if pages not in (None, ""):
            args["pages"] = pages
        return _truncate(
            run_log_grid(args, attachments, engine,
                         max_chars=_document_tools.budget_for_cap(reference_cap),
                         cap=reference_cap or None),
            reference_cap)

    def save_file(path: str, content: str, encoding: str = "text") -> str:
        """Save raw text or data to a file. Returns the saved file path. For
        formatted calculation documents, use the ``calc_package`` module via
        ``call_agent`` instead.

        ``path`` is the output file path; ``content`` is the file content
        (text or base64); ``encoding`` is 'text' (default) or 'base64' for
        binary.
        """
        return _with_saved_note(
            _dispatch(
                "save_file",
                {"path": path, "content": content, "encoding": encoding},
            ),
            save_fn,
        )

    def write_docx(path: str, markdown: str, title: str = "") -> str:
        """Write a Word (.docx) document from markdown — a memo, a findings
        summary, a review response: anything the reader opens in Word, tracks
        changes in, or pastes into a report template.

        ``markdown`` is ordinary markdown: headings ``#`` to ``####``,
        **bold**, *italic*, `code`, bullet and numbered lists (one level of
        nesting), pipe tables (an italic line right after a table becomes its
        caption), ``>`` quotes, fenced code, and ``![alt](figure.png)`` for a
        figure already saved in the working folder. ``---`` becomes a PAGE
        BREAK. ``title`` is optional and is rendered as the document title.
        ``path`` is the output file (``.docx`` is added if missing); a bare
        name lands in the working folder. A figure that cannot be found is
        reported in ``warnings`` and the document is still written — read them
        and tell the user. For a Mathcad-style CALCULATION package use the
        ``calc_package`` module via ``call_agent`` instead.
        """
        return _with_saved_note(
            _dispatch(
                "write_docx",
                {"path": path, "markdown": markdown, "title": title},
            ),
            save_fn,
        )

    def plot_data(series: list, title: str = "", xlabel: str = "",
                  ylabel: str = "", depth_axis: bool = False,
                  logx: bool = False, logy: bool = False,
                  hlines: Optional[list] = None,
                  vlines: Optional[list] = None,
                  legend: Optional[bool] = None, output_path: str = "",
                  interactive: bool = True) -> str:
        """Plot x/y data (the ``profile_figure`` module's ``plot_data``)."""
        params: Dict[str, Any] = {
            "series": series, "title": title, "xlabel": xlabel,
            "ylabel": ylabel, "depth_axis": depth_axis, "logx": logx,
            "logy": logy, "interactive": interactive}
        for k, v in (("hlines", hlines), ("vlines", vlines),
                     ("legend", legend), ("output_path", output_path or None)):
            if v is not None:
                params[k] = v
        return _truncate(json.dumps(
            _call_agent(agent_name="profile_figure", method="plot_data",
                        parameters=params, attachments=attachments,
                        allowed_agents=None),
            default=str), max_result_chars)

    def write_xlsx(path: str, sheets: Optional[list] = None,
                   markdown: str = "") -> str:
        """Write an Excel workbook (one sheet per table)."""
        return _write_xlsx(path, sheets, markdown, save_fn)

    # Each document tool is described to the model in planlens' own words, and
    # the ones a newer planlens added appear only where they exist.
    document_review_builders = [
        ("open_document", open_document),
        ("document_structure", document_structure),
        ("document_page_map", document_page_map),
        ("read_document", read_document),
        ("search_document", search_document),
        ("document_markups", document_markups),
        ("render_page_thumbnails", render_page_thumbnails),
    ]
    if _document_tools.has_tool("find_quantities"):
        document_review_builders.append(("find_quantities", find_quantities))
    if _document_tools.has_tool("annotate_document"):
        document_review_builders.append(("annotate_document", annotate_document))
    # Visual scales (planlens' find_scales behind both): described in the
    # app's own words, because the app takes a look's view + image_box and
    # reads a scan's label values itself.
    if _document_tools.has_tool("measure"):
        document_review_builders.append(("measure", measure))
    if _document_tools.has_tool("log_grid"):
        document_review_builders.append(("log_grid", log_grid))

    _builders = {
        "list_files": (
            list_files,
            "List the files this conversation holds (read-only): name, type "
            "(dir/file), size and modified time per entry. path defaults to "
            "'.', the conversation's working folder (uploads, fetched and "
            "produced files, and .scratch for tool scratch images); a "
            "relative path is inside it. The read tools reach only this "
            "conversation's files, the reference documents and any folder the "
            "deployment opens. The scratch filesystem's ls does not see them.",
        ),
        "read_pdf_text": (
            read_pdf_text,
            "Extract the TEXT LAYER of a PDF (PyMuPDF — cheap, no vision). "
            "First-choice reader for a text-based report; a scanned page with "
            "no text layer is flagged per-page (use analyze_pdf_page there). "
            "source is an attachment key or a file name in the working "
            "folder.",
        ),
        "read_text_file": (
            read_text_file,
            "Read a text file this conversation holds (HTML, TXT, CSV, JSON, "
            "MD -- e.g. a report source written earlier), by its name in the "
            "working folder. The scratch read_file cannot see it. Pages with "
            "offset / next_offset.",
        ),
        "analyze_image": (
            analyze_image,
            "Analyze an image using vision: an attached image (its key), an "
            "image file in the working folder, or a scratch image a tool "
            "returned by name (.scratch/... from render_page_thumbnails or "
            "find_like). Returns text "
            "description/analysis of the image content.",
        ),
        "analyze_pdf_page": (
            analyze_pdf_page,
            "Render a PDF page and analyze it using vision. tiles: 'auto' "
            "(default: the page is ALSO read in overlapping tiles when its "
            "lettering is too small in the whole-page image), 'off', or N or "
            "'NxN' with N from 2 to 4 (e.g. '3x3') for a fixed split. Boxes "
            "come back on a 0-999 grid with the result's view; a whole-page "
            "box says where to zoom, not where to put a mark.",
        ),
        "find_like": (
            find_like,
            "Find EVERY copy of one mark (tag, code, symbol) across a drawing "
            "set — even lettering drawn as lines — from one zoomed, confirmed "
            "example box + the text it reads; every candidate is verified by "
            "vision. Returns instances by page, callouts (with where each "
            "leader points) apart from legend entries, and uncertain reads.",
        ),
        "render_region": (
            render_region,
            "Render a ZOOMED-IN crop of a PDF page and analyze it with "
            "vision — for drawings: get exact coordinates from the "
            "drawing_ir module first (geometry says WHERE), then zoom here "
            "to see WHAT is there. bbox is [x0,y0,x1,y1] in PDF points "
            "(top-left origin, y down); optional marks = [[x,y,label],...] "
            "draws numbered circles at points of interest. To zoom on "
            "something an earlier vision result located, pass its view + "
            "the 0-999 image_box its analysis gave instead of bbox: the "
            "window is padded by that view's location error, so the thing "
            "is in it.",
        ),
        "read_reference_figure": (
            read_reference_figure,
            "Render a digitized reference figure (e.g. a DM7 design chart) "
            "and read a value off it with vision. Find the figure first via "
            "figure_db.figure_search, then read the value off the actual "
            "chart. Returns the vision read-off estimate and, where the "
            "chart's axes and curve can be found in the drawing, a value "
            "MEASURED from it with its +/- beside the estimate "
            "(code_reading; flagged where the two disagree).",
        ),
        "view_worked_example_source": (
            view_worked_example_source,
            "Render a worked example's PRINTED SOURCE PAGE (design-report "
            "example incl. figures) and analyze it with vision. Use after "
            "find_worked_examples when the entry lists source_pdf_pages; "
            "pdf_page is 1-based (0 = first catalogued page).",
        ),
        **({name: (fn, _app_document_description(name))
            for name, fn in document_review_builders}
           if _document_tools.available() else {}),
        "save_file": (
            save_file,
            "Save raw text or data to a file. Returns the saved file path. "
            "For formatted calculation documents, use the calc_package module "
            "via call_agent instead.",
        ),
        **({"write_docx": (
            write_docx,
            "Write a Word (.docx) document from markdown — a memo, a findings "
            "summary, a review response. Headings, bold/italic/code, bullet "
            "and numbered lists, pipe tables (an italic line after a table is "
            "its caption), block quotes, fenced code, and "
            "![alt](figure.png) for a figure already in the working folder; "
            "--- is a page break. A figure that is not found is reported in "
            "warnings and the document is still written. For a Mathcad-style "
            "calculation package use the calc_package module instead.",
        )} if _docx_available() else {}),
        **({"plot_data": (plot_data, _plot_description())}
           if _plot_available() else {}),
        **({"write_xlsx": (write_xlsx, WRITE_XLSX_DESCRIPTION)}
           if _xlsx_available() else {}),
    }

    overrides = dict(description_overrides or {})
    tools = []
    for name, (fn, desc) in _builders.items():
        if name in include:
            # An unknown argument (analyze_pdf_page(pages=12)) is refused by
            # name rather than dropped (brief 5, N3).
            tools.append(strict_tool(
                StructuredTool.from_function(
                    fn, name=name, description=overrides.get(name, desc))
            ))
    return tools


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
) -> list:
    """The ``report_ingest`` primary tool, or an empty list.

    Empty when the installed planlens cannot serve the ingest: it needs the
    page roles and the log grid, which arrived in planlens 0.5. The cluster
    installs planlens from PyPI and has resolved older than the pin before,
    so this is checked against the INSTALLED package's own published specs
    rather than trusted from a version number — the same rule the document
    tools follow.

    ``engine`` is whatever the host built the agent with; the live Prompter
    inside it becomes the ingest's engine through
    :func:`report_ingest.engine.engine_for`. With no engine to be found the
    tool is still built and says so when called, which is a better answer
    than a missing tool the model then invents a workaround for.

    ``label_policy`` and ``review_mode`` are the ingest's page vote, passed
    straight through to :func:`report_ingest.graph.ingest_report`. The
    defaults are its own: three cheap voters on every page, and the
    expensive label review shown only the pages they split on.
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
        # the readers use; on a host whose engine is not a Prompter this is
        # None and the graph falls back to the one engine it has.
        vision_engine = engine_for(engine, model=VISION_TIER)
        if ingest_engine is None:
            return json.dumps({
                "error": "no ingest engine is configured in this deployment",
                "hint": "the ingest runs on the app's own model; tell the "
                        "user it is unavailable here rather than reading the "
                        "report another way"})
        folder = out_dir or _report_ingest_out_dir(source)
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
                                    os.path.basename(str(source)))[0])
        except Exception as exc:  # noqa: BLE001 - one report, not the turn
            return json.dumps({"error": f"the ingest failed: "
                                        f"{type(exc).__name__}: {exc}"})
        return _truncate(json.dumps(answer.model_dump()), max_result_chars)

    report_ingest.__doc__ = REPORT_INGEST_DESCRIPTION
    return [strict_tool(StructuredTool.from_function(
        report_ingest, name="report_ingest",
        description=REPORT_INGEST_DESCRIPTION))]


def _report_ingest_out_dir(source: str) -> str:
    """Where a report's outputs go: the conversation's own folder."""
    from funhouse_agent._fileio import default_output_dir

    stem = os.path.splitext(os.path.basename(str(source)))[0] or "report"
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
