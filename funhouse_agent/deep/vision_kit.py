"""The vision kit: both pages' looking, reading and file-output tools.

:func:`make_vision_tools` builds the vision, document and file-output tools
as LangChain :class:`~langchain_core.tools.StructuredTool` objects (each
wrapping :func:`funhouse_agent.vision_tools.dispatch_extended_tool` or a
planlens document tool), with the tool-result caps. It moved out of
:mod:`funhouse_agent.deep.tools` (INF-30 step 4) so the Document Review
page builds its tools without loading the geotechnical dispatch module:
nothing here imports it at load (``plot_data`` reaches it when it draws,
until the chart stack moves in step 6). :mod:`funhouse_agent.deep.tools`
re-exports these names; its ``make_vision_tools`` adds the geotech page's
two reference readers.

Every tool returns a JSON string. The engine, attachments and save_fn are
closured in at build time; for offline construction ``analyze_*`` return a
clear error if no engine was wired.
"""

import json
import os
from typing import (Annotated, Any, Callable, Dict, List, Optional, Sequence,
                    Tuple)

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, WithJsonSchema, create_model

from funhouse_agent import document_tools as _document_tools
from funhouse_agent.deep.limits import DEFAULT_MAX_RESULT_CHARS
from funhouse_agent.deep.strict_args import strict_tool
from funhouse_agent.vision_tools import (
    IMAGE_FIND_NOTE,
    IMAGE_LOOK_NOTE,
    IMAGE_MARKUP_NOTE,
    IMAGE_ZOOM_NOTE,
    KEPT_READINGS_DESCRIPTION,
    TURN_ARG_NOTE,
    dispatch_extended_tool as _dispatch_extended_tool,
    docx_available as _docx_available,
    _default_save_fn,
)


# ---------------------------------------------------------------------------
# Tool-result size cap (token-bloat guard)
# ---------------------------------------------------------------------------

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


def _dxf_notes_available() -> bool:
    """Whether DXF notes can be written (ezdxf, a core app dependency
    through planlens; a guard for a stripped library install)."""
    try:
        from funhouse_agent.dxf_notes import available
    except Exception:                                  # noqa: BLE001
        return False
    return available()


ANNOTATE_DXF_DESCRIPTION = (
    "Add review notes to a COPY of a DXF drawing - each note a TEXT (MTEXT "
    "for several lines) on a markup layer of its own (default REVIEW, red), "
    "optionally with a leader to the point it is about - written by a CAD "
    "library, so nothing else in the drawing changes; the copy is read back "
    "and the result says every original entity is still there. source = "
    "the DXF's file name; notes = [{text, x, y, leader_to?: [x, y], "
    "height?}] in the drawing's own units and coordinates, as "
    "open_document / read_document list them (a CIRCLE's centre, a TEXT's "
    "position; put the note a little beside the thing and point at it with "
    "leader_to). output_path = a bare file name (default "
    "<name>_marked.dxf). Use it for any note, comment or label on a DXF; "
    "never retype a DXF with save_file.")


#: Said in place of the zoom hint where ``render_region`` is not offered
#: (live wave w8: the figure and calc helpers were told to zoom with a tool
#: they did not have; BACKLOG VIS-28).
NO_ZOOM_NOTE = ("render_region is not offered here: for a closer read, ask "
                "again with tiles='3x3' (or '4x4') and a narrower prompt.")


def _without_zoom_tool(raw):
    """A vision result for an agent without ``render_region``: no zoom
    hint naming it."""
    import re
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(out, dict):
        return raw
    changed = out.pop("zoom_hint", None) is not None
    tiling = out.get("tiling")
    if isinstance(tiling, str) and "render_region" in tiling:
        out["tiling"] = re.sub(
            r"\s*Each tile's boxes are on that tile's own 0-999 grid:[^.]*\.",
            "", tiling)
        changed = True
    if not changed:
        return raw
    out["zoom_note"] = NO_ZOOM_NOTE
    return json.dumps(out, ensure_ascii=False)


#: Said by search_document given min_score without fuzzy, which planlens
#: drops without a word (BACKLOG DR-32).
MIN_SCORE_UNUSED_NOTE = ("min_score was not used: it is the lowest score of "
                         "a fuzzy match, and this search was exact "
                         "(fuzzy=false)")


def _json_with(raw, **keys):
    """A JSON tool result with ``keys`` added; an error, or a result that
    is not a JSON object (cut short), is left as it is."""
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(out, dict) or "error" in out:
        return raw
    out.update(keys)
    return json.dumps(out, ensure_ascii=False)


#: What the app adds to the module's own description of ``plot_data``.
_PLOT_APP_NOTE = (
    " In this app the chart appears as an INTERACTIVE card under your reply "
    "(zoom and hover); the PNG it also saves is the copy for a report "
    "(html_img_tag) or a Word document (![title](<output_path>) in "
    "write_docx). The PNG and the chart are drawn from the same series, so "
    "analyze_image on the PNG checks what the chat shows; look before you "
    "call a figure done. Give output_path a bare file name, or leave it "
    "out, and redraw under the same name. Never draw a chart by hand as SVG, "
    "as text or as a .plotly.json. What it draws: x/y series, filled "
    "outlines and named points (a name is placed where it stands clear of "
    "the data, near the point, not exactly on it), and notes placed where "
    "you put them (annotations). It cannot draw a "
    "structure to scale with diagrams beside it, or several panels sharing "
    "one axis. plan_view is for plans only (it adds a north arrow and a "
    "scale bar), never for a section.")


def _plot_description(reference_chart: bool = True) -> str:
    """``plot_data``'s model-facing description, in the module's own words
    (``profile_figure`` METHOD_INFO), with the app's note.
    ``reference_chart=False`` leaves that argument out (the Document Review
    page, which draws no soil-classification chart: BACKLOG RI-34)."""
    try:
        from funhouse_agent.adapters.profile_figure_adapter import METHOD_INFO
        info = METHOD_INFO["plot_data"]
        params = info.get("parameters") or {}
        lines = [str(info.get("brief") or "").strip()]
        for key in ("series", "annotations",
                    *(("reference_chart",) if reference_chart else ()),
                    "plan_view", "north_deg",
                    "equal_aspect", "xlabel", "ylabel", "depth_axis",
                    "x_on_top", "logx",
                    "hlines", "vlines", "xlim", "ylim", "legend",
                    "legend_position", "interactive"):
            desc = (params.get(key) or {}).get("description")
            if desc:
                lines.append(f"{key}: {desc}")
        return " ".join(lines) + _PLOT_APP_NOTE
    except Exception:                                  # noqa: BLE001
        return ("Plot one or more x/y data series (series: [{x: [...], y: "
                "[...], label}]) -- an SPT or CPT profile against elevation "
                "(depth with depth_axis=true only when no elevations are "
                "given), settlement vs time, a sweep, quantities from a "
                "document."
                + _PLOT_APP_NOTE)


#: ``write_xlsx``'s model-facing description.
WRITE_XLSX_DESCRIPTION = (
    "Write an Excel workbook (.xlsx), one sheet per table -- a schedule, a "
    "quantity take-off, a comment log, test results, a comparison table: "
    "anything the reader will sort, filter or calculate with. Give the "
    "tables EITHER as 'sheets' = [{name, rows}] with rows a list of lists "
    "(the first row is the header, or give it as 'columns') or a list of "
    "objects (their keys become the header); any other key in a sheet is "
    "refused, OR as 'markdown' holding pipe tables (each table becomes "
    "its own sheet, named by the heading above it); both may be given. "
    "Numbers are written as numbers (an id with a leading zero stays text), "
    "the header row is bold and frozen, columns are sized to their text (at "
    "most 60 characters wide) and a longer cell wraps, its row made tall "
    "enough to show it all. 'path' is "
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


# ---------------------------------------------------------------------------
# Vision / file-output tools (wrap vision_tools.dispatch_extended_tool)
# ---------------------------------------------------------------------------

#: Appended to planlens' own description where THIS app changes the contract.
#: planlens is framework-neutral and cannot know where a file belongs or who
#: is signing the review; both are settled here, so the model is told.
#: Every page-range document tool also takes the viewer's 1-based numbers
#: (live smoke wave 2c: a model passed PDF page 5 of a 5-page document).
_PDF_PAGES_NOTE = (" pages counts from 0; or give pdf_pages instead, "
                   "1-based as a PDF viewer shows (e.g. '1-3,6').")
#: The same for a one-page argument.
_PDF_PAGE_NOTE = (" page counts from 0; or give pdf_page instead, 1-based "
                  "as a PDF viewer shows.")

_DOCUMENT_TOOL_NOTES = {
    # Live smoke wave 2b, C2 / C8: MuPDF read a workbook as nine digits and
    # could not open a DXF; both are read as text by the app instead.
    "open_document": (
        " In this app it also opens a Word (.docx), Excel (.xlsx, .xlsm) or "
        "DXF file -- not as pages but as text: a Word or Excel file as "
        "Markdown (one table per sheet), a DXF as its text, leaders and "
        "dimensions with their positions, layers and entity counts. Its file "
        "name is then its handle for read_document (start_line pages "
        "through it) and search_document; the page, markup and vision tools "
        "do not apply to it."),
    "read_document": (
        " For a Word, Excel or DXF file, handle is its file name and "
        "start_line pages through its lines (pages does not apply)."
        + _PDF_PAGES_NOTE),
    "search_document": (
        " For a Word, Excel or DXF file, handle is its file name; hits are "
        "its lines." + _PDF_PAGES_NOTE),
    "document_page_map": _PDF_PAGES_NOTE,
    "document_markups": _PDF_PAGES_NOTE,
    "find_quantities": _PDF_PAGES_NOTE,
    "render_page_thumbnails": _PDF_PAGES_NOTE,
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
        "target or quoted words name — or, naming neither, the thing their "
        "comment is about — and which are misplaced. A mark put on blank "
        "paper on purpose (a stamp in an empty corner) names that area as "
        "its target; the check reports it under on_blank_paper, which a "
        "look cannot confirm: tell the user where it sits. A label is only the "
        "text the reader sees beside a mark (a tag, a verdict such as "
        "'460.1?'): it is never compared with what is under the mark, so it "
        "need not match it. Never hand over a file with misplaced marks: "
        "find those things again and rewrite the copy with append=false "
        "(a misplaced ring or box may come back with found_bbox, where the "
        "check saw the thing: pass it as that mark's bbox). Keep correcting "
        "without stopping to ask, up to the number of checks the note "
        "gives; after the last, hand the file over and say which marks are "
        "unconfirmed. "
        "To change or delete ONE mark of a marked copy you wrote, call again "
        "with output_path = that copy and remove = [its id as "
        "document_markups lists it on that copy, e.g. 'p0.m1', or words "
        "from its comment or its label]; the marks you give in markups are "
        "added, every other mark stays (only this app's marks can be "
        "removed; ids change after a removal, so list them again before "
        "removing another). The result's in_file counts what the copy "
        "holds: the marks this call added and those kept from before, by "
        "author — the document's own markups are always kept. A markup may "
        "give pdf_page (1-based, as a PDF viewer shows) instead of page "
        "(0-based). " + IMAGE_MARKUP_NOTE),
}


def _app_document_description(name: str) -> str:
    """A document tool's model-facing description: planlens' own words, with
    the app's note where the app changes the contract — and, for the two
    visual-scale tools, the app's own words (they take a look's view +
    image_box and read a scan's labels, which planlens' text cannot say)."""
    if name == "measure":
        from funhouse_agent.measure_tool import MEASURE_DESCRIPTION
        return MEASURE_DESCRIPTION + _PDF_PAGE_NOTE
    if name == "log_grid":
        from funhouse_agent.measure_tool import log_grid_description
        return log_grid_description() + _PDF_PAGES_NOTE
    return (_document_tools.tool_description(name)
            + _DOCUMENT_TOOL_NOTES.get(name, ""))


def _document_tool_schemas(name: str) -> Dict[str, Dict[str, Any]]:
    """Arguments of a document tool shown to the model with a schema of
    their own rather than the one read off the function (BACKLOG DR-32):
    ``annotate_document``'s ``markups`` with planlens' markup fields (it
    was an array of anything) and ``remove`` as strings; ``search_document``'s
    ``min_score`` with planlens' default, 80, though the function's is
    ``None`` so a score given without ``fuzzy`` can be said."""
    if name == "annotate_document":
        shown = {"remove": {"type": "array", "items": {"type": "string"}}}
        item = _document_tools.annotate_markups_schema()
        if item is not None:
            shown["markups"] = item
        return shown
    if name == "search_document":
        return {"min_score": {"type": "integer", "default": 80}}
    return {}


def _shown_as(tool, shown: Dict[str, Dict[str, Any]]):
    """``tool`` with the JSON schema the model is shown for some arguments
    replaced (``{name: schema}``). What a call is checked against is
    unchanged (a list stays any list), so nothing the tool takes today is
    refused. A schema with its own ``default`` is shown with it, not the
    function's."""
    base = tool.args_schema
    if not shown or not (isinstance(base, type)
                         and issubclass(base, BaseModel)):
        return tool
    fields = {}
    for name, schema in shown.items():
        info = base.model_fields.get(name)
        if info is None:
            continue
        default = info.default
        # A default_factory keeps pydantic from writing the function's
        # default over the schema's own.
        fields[name] = (Annotated[info.annotation, WithJsonSchema(schema)],
                        Field(default_factory=lambda d=default: d)
                        if "default" in schema else default)
    if not fields:
        return tool
    model = create_model(base.__name__, __base__=base,
                         __module__=base.__module__, **fields)
    return tool.model_copy(update={"args_schema": model})


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
    plot_data_replaces: bool = False,
    description_rewrites: Optional[
        Dict[str, Sequence[Tuple[str, str]]]] = None,
    plot_reference_charts: bool = True,
    added_tools: Optional[Dict[str, Callable]] = None,
) -> list:
    """Build the vision / file-output tools as LangChain tools.

    ``plot_data_replaces`` gives ``plot_data`` a ``replaces`` argument that
    retires a draft drawn earlier in the turn under another name (the figure
    helper's copy, behind its switch; live wave w9 left a rejected draft's
    card beside the final figure).

    ``description_overrides`` replaces a tool's model-facing description by
    name — the lean Document Review agent describes its tools for the page it
    is on (:mod:`funhouse_agent.deep.review_agent`). ``description_rewrites``
    swaps phrases inside a tool's description, ``{name: [(old, new), ...]}``,
    after any override: the Document Review page's wording for a description
    built elsewhere (planlens' own, or one with feature-detected parts), so
    its examples name no kind of geotechnical report (BACKLOG RI-34).
    ``plot_reference_charts=False`` leaves ``plot_data``'s
    ``reference_chart`` (the soil-classification charts) out, argument and
    description both. ``inline_images`` makes
    ``analyze_pdf_page`` / ``render_region`` return the rendered image for the
    main model to look at instead of a one-shot vision call's description
    (only meaningful with the image middleware that shows it; see
    :mod:`funhouse_agent.deep.inline_images`). ``inline_image_files`` makes
    ``analyze_image`` do the same for an image FILE given by its path (a
    contact sheet); the lean agent sets it only with both
    ``GEOTECH_VISION_INLINE`` and ``GEOTECH_REVIEW_OVERVIEW`` on.

    ``added_tools`` are tools another app adds to the kit, ``{name:
    builder}``: ``builder(dispatch)`` returns ``(function, description)``,
    the function reading through ``dispatch(name, arguments)`` (the kit's
    call into :func:`funhouse_agent.vision_tools.dispatch_extended_tool`,
    under the reference cap). They are built where ``include`` names them
    (and by default), after ``kept_readings``. The geotech page adds its
    reference-figure and worked-example readers this way
    (:func:`funhouse_agent.deep.tools.make_vision_tools`).

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
        ``analyze_image`` / ``analyze_pdf_page`` and the added readers;
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
        Larger cap for REFERENCE reads (an added reader's chart read-off,
        ``read_pdf_text``: the text is the payload). Defaults to
        ``DEFAULT_REFERENCE_RESULT_CHARS`` (16000; never below
        ``max_result_chars``); disabled when ``max_result_chars <= 0``.

    Returns
    -------
    list[StructuredTool]
    """
    attachments = attachments if attachments is not None else {}
    save_fn = save_fn or _default_save_fn
    added = dict(added_tools or {})
    if include is None:
        include = {
            "list_files", "read_pdf_text", "read_text_file", "analyze_image",
            "analyze_pdf_page", "render_region", "save_file",
            # The conversation's kept page readings, for every agent with
            # the looking tools -- helpers too (BACKLOG TA-28).
            "kept_readings",
        }
        # The tools another app adds (added_tools), by default.
        include |= set(added)
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
        # Notes on a DXF, written by a CAD library -- never retyped by the
        # model (live smoke wave 2c, E4).
        if _dxf_notes_available():
            include |= {"annotate_dxf"}
        # find_like needs planlens 0.10 (planlens.document.findlike).
        if _find_like_available():
            include |= {"find_like"}
    reference_cap = _resolve_reference_cap(max_result_chars,
                                           reference_result_chars)
    # Resolved once per build: the newer tools depend on the installed planlens.
    document_names = set(_document_tools.document_tool_names())
    # plot_data draws per figure name, for the redraw note.
    draws: Dict[str, int] = {}

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
        # read_pdf_text (and an added reference reader) are text-payload
        # reads and list_files can be a long directory dump: the content IS
        # the answer, so they get the larger reference budget.
        # analyze_image too: an image too large for one look is read in
        # tiles (VIS-09), and the tiled result is held under its own budget.
        # kept_readings returns one page's reading whole on request, under
        # the same budget as the look that made it (BACKLOG TA-28).
        if tool_name in ("analyze_pdf_page", "find_like", "analyze_image",
                         "kept_readings"):
            cap = (0 if max_result_chars <= 0
                   else max(DEFAULT_VISION_RESULT_CHARS, reference_cap))
        elif tool_name in ("read_pdf_text", "read_text_file",
                           "list_files") or tool_name in added:
            cap = reference_cap
        else:
            cap = max_result_chars
        raw = _dispatch_extended_tool(
            tool_name=tool_name,
            arguments=arguments,
            engine=engine,
            attachments=attachments,
            save_fn=save_fn,
        )
        if tool_name in ("analyze_image", "analyze_pdf_page") \
                and "render_region" not in include:
            raw = _without_zoom_tool(raw)
        return _truncate(raw, cap)

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

    def read_pdf_text(source: str, pages: str = "",
                      pdf_pages: Any = None) -> str:
        """Extract the TEXT LAYER of a PDF (PyMuPDF — cheap, no vision). Use
        this FIRST for a text-based report (boring logs, lab summaries,
        recommendations, specs) instead of vision-reading every page.

        ``source`` is an attachment key or a file name in the working folder
        (a relative path is inside it). ``pages`` counts from 0: an int, a
        list, or a range like "0-9"; or give ``pdf_pages`` instead, 1-based
        as a PDF viewer shows (e.g. "1-10"); omit both for the first several
        pages. Each page in the result gives both (``page``, ``pdf_page``).
        A page with no text layer (scanned image) is flagged per-page — use
        analyze_pdf_page for those.
        """
        args = {"source": source}
        if pages != "":
            args["pages"] = pages
        if pdf_pages not in ("", None, [], ()):
            args["pdf_pages"] = pdf_pages
        return _dispatch("read_pdf_text", args)

    def read_text_file(path: str, offset: int = 0, max_chars: int = 6000,
                       limit: Optional[int] = None) -> str:
        """Read a file this conversation holds as text -- HTML, TXT, CSV,
        JSON, MD, such as a report source written earlier -- and a Word
        (.docx) or Excel (.xlsx) file as Markdown: a .docx in write_docx's
        own dialect (headings, numbered/bulleted lists, tables, bold/italic,
        line breaks), so a memo can be read, edited and written back with
        write_docx without its layout drifting; an .xlsx as one table per
        sheet with the sheet names. The scratch filesystem's ``read_file``
        cannot see these files. ``path`` is a file name in the working
        folder (a relative path is inside it). ``limit`` (or ``max_chars``)
        is how many characters to return, 6,000 by default and at most
        12,000 per call; ``offset`` is where to start. The result's
        ``showing`` says which characters came back of how many, and
        ``next_offset`` where the rest starts."""
        args = {"path": path, "offset": offset, "max_chars": max_chars}
        if limit is not None:
            args["limit"] = limit
        return _dispatch("read_text_file", args)

    def analyze_image(attachment_key: str,
                      prompt: str = "Describe this image.",
                      tiles: str = "auto",
                      turn: Optional[int] = None) -> str:
        """Analyze an image using vision. Returns text description/analysis
        of the image content.

        ``attachment_key`` is an attached image's key (an upload, a pasted
        screenshot), an image file's name in the working folder, or the name
        a tool returned for a scratch image (``.scratch/...`` --
        render_page_thumbnails' contact sheets, find_like's sheets);
        ``prompt`` is what to extract or analyze. The result gives the
        image's ``view`` (its own pixels) and 0-999 boxes, to zoom with
        render_region. ``tiles``: ``"auto"`` (default) ALSO reads an image
        larger than one look holds in overlapping close-up tiles; ``"off"``;
        or N / ``"NxN"`` with N 2-4. ``turn``: 90, 180 or 270 sends it
        turned that many degrees clockwise.
        """
        args = {"attachment_key": attachment_key, "prompt": prompt}
        if tiles not in (None, "", "auto"):
            args["tiles"] = tiles
        if turn is not None:
            args["turn"] = turn
        if inline_image_files:
            # An image FILE (a contact sheet) is shown to the main model like
            # the inline page tools' images; an upload keeps the vision call.
            args["_inline"] = True
        # A file that is not an image is answered with the tools that read
        # it, of those this agent has (GM-175: the calc helper was told to
        # use analyze_pdf_page, which it did not have).
        args["_tools"] = sorted(include)
        _wait_for_drawing(attachment_key)
        return _dispatch("analyze_image", args)

    def _wait_for_drawing(name) -> None:
        """A look at a figure file waits while a drawing tool is writing in
        the conversation folder (live wave w8: a draw and a look issued in
        the same step read the PNG from before the redraw)."""
        if not isinstance(name, str) or name in attachments or not \
                name.lower().endswith((".png", ".jpg", ".jpeg", ".gif",
                                       ".webp", ".svg")):
            return
        try:
            from funhouse_agent._fileio import host_output_dir
            from funhouse_agent.drawing_wait import (drawing_in,
                                                     wait_until_drawn)
            folder = host_output_dir()
        except Exception:                              # noqa: BLE001
            return
        if not folder:
            return
        if not drawing_in(folder):
            # a draw issued in the same step may not have started yet
            import time as _time
            _time.sleep(0.5)
        if drawing_in(folder):
            wait_until_drawn(folder)

    def analyze_pdf_page(
        attachment_key: str,
        page: Optional[int] = None,
        prompt: str = "Describe the content of this page.",
        tiles: str = "auto",
        pdf_page: Optional[int] = None,
        reuse: bool = False,
        turn: Optional[int] = None,
    ) -> str:
        """Render a PDF page and analyze it using vision.

        ``attachment_key`` is the key of the attached PDF file; ``page`` is the
        0-indexed page number (default 0) -- or give ``pdf_page`` instead,
        1-based, the number a PDF viewer shows; ``prompt`` is what to extract
        from the page.
        ``tiles``: ``"auto"`` (default) ALSO reads the page in overlapping
        tiles when its small lettering is too small in the whole-page image
        (the result then carries every tile's reading and view); ``"off"``;
        or N / ``"NxN"`` with N 2-4 (``"3"`` or ``"3x3"``) for a fixed split.
        ``reuse``: true hands back this conversation's latest reading of the
        page (it says what it was asked) instead of a new look -- for a
        follow-up the earlier reading already answers; look again when it
        does not.
        ``turn``: 90, 180 or 270 sends the page (and its tiles) turned that
        many degrees clockwise, for lettering that runs up the page (90),
        down it (270) or upside down (180); 0 sends it as the page shows.
        """
        args = {"attachment_key": attachment_key, "prompt": prompt,
                "tiles": tiles}
        if page is not None:
            args["page"] = page
        if pdf_page is not None:
            args["pdf_page"] = pdf_page
        if reuse:
            args["reuse"] = True
        if turn is not None:
            args["turn"] = turn
        if inline_images:
            args["_inline"] = True
        return _dispatch("analyze_pdf_page", args)

    def find_like(
        attachment_key: str,
        page: Optional[int] = None,
        bbox: Optional[list] = None,
        text: str = "",
        pages: Optional[str] = None,
        include_legend: bool = False,
        view: Optional[list] = None,
        image_box: Optional[list] = None,
        pdf_page: Optional[int] = None,
        pdf_pages: Optional[str] = None,
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
        the way. ``page`` / ``pages`` count from 0; ``pdf_page`` /
        ``pdf_pages`` are the same 1-based, as a PDF viewer shows.
        """
        args = {"attachment_key": attachment_key,
                "include_legend": include_legend}
        for k, v in (("page", page), ("bbox", bbox), ("text", text or None),
                     ("pages", pages), ("view", view),
                     ("image_box", image_box), ("pdf_page", pdf_page),
                     ("pdf_pages", pdf_pages)):
            if v is not None:
                args[k] = v
        return _dispatch("find_like", args)

    def render_region(
        attachment_key: str,
        page: Optional[int] = None,
        bbox: Optional[list] = None,
        marks: Optional[list] = None,
        prompt: str = "Describe what this zoomed-in region shows.",
        view: Optional[list] = None,
        image_box: Optional[list] = None,
        pdf_page: Optional[int] = None,
        reuse: bool = False,
        turn: Optional[int] = None,
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
        ``attachment_key`` is an attachment key or a real PDF path -- or an
        IMAGE (an upload, a pasted screenshot): then ``bbox`` and ``view``
        are in the image's own pixels, ``page`` is not used, and the zoom
        is cut from the original at full resolution.

        To zoom on something an earlier vision result LOCATED, pass that
        result's ``view`` plus the 0-999 ``image_box`` its analysis gave
        (instead of ``bbox``); same ``attachment_key`` and ``page``. The
        window is the box padded by how far a box from that view can be off
        (a tenth of the view each way), so the thing is in it. Every
        vision result carries a ``view``. The zoom is always drawn as large
        as the vision model reads: to see more detail, zoom on a smaller
        box. ``page`` counts from 0 (default 0); or give ``pdf_page``,
        1-based, as a PDF viewer shows.
        ``reuse``: true hands back this conversation's kept zoom of the same
        region (it says what it was asked) instead of a new look -- for a
        follow-up that zoom already answers; look again when it does not.
        ``turn``: 90, 180 or 270 sends the zoom turned that many degrees
        clockwise, for lettering that runs up the page (90), down it (270)
        or upside down (180); boxes still come back on the view as the page
        shows it. 0 sends it as the page shows.
        """
        args = {"attachment_key": attachment_key, "prompt": prompt}
        if page is not None:
            args["page"] = page
        if pdf_page is not None:
            args["pdf_page"] = pdf_page
        if bbox is not None:
            args["bbox"] = bbox
        if marks is not None:
            args["marks"] = marks
        if view is not None:
            args["view"] = view
        if image_box is not None:
            args["image_box"] = image_box
        if reuse:
            args["reuse"] = True
        if turn is not None:
            args["turn"] = turn
        if inline_images:
            args["_inline"] = True
        return _dispatch("render_region", args)

    def kept_readings(document: str = "", pdf_page: Optional[int] = None,
                      pdf_pages: Any = None, kind: str = "",
                      bbox: Optional[list] = None, offset: int = 0) -> str:
        """The page readings this conversation keeps (see
        ``vision_tools.KEPT_READINGS_DESCRIPTION``): by reference without
        ``pdf_page``, one page's reading whole with it. Read-only."""
        args: Dict[str, Any] = {}
        for k, v in (("document", document or None), ("pdf_page", pdf_page),
                     ("pdf_pages", pdf_pages), ("kind", kind or None),
                     ("bbox", bbox), ("offset", offset or None)):
            if v not in (None, "", [], ()):
                args[k] = v
        return _dispatch("kept_readings", args)

    def open_document(source: str) -> str:
        """Open a PDF for review; returns a handle and a map of the document."""
        return _dispatch("open_document", {"source": source})

    def document_structure(handle: str, offset: int = 0) -> str:
        """The constituent documents of a stapled PDF, with the page
        numbers printed on them."""
        return _dispatch("document_structure",
                         {"handle": handle, "offset": offset})

    def _with_pdf_pages(args: dict, pdf_pages: Any) -> dict:
        """``pdf_pages`` (1-based, as a viewer shows) handed on; the
        document tools' dispatch turns it into ``pages`` (live smoke 2c)."""
        if pdf_pages not in (None, "", [], ()):
            args["pdf_pages"] = pdf_pages
        return args

    def render_page_thumbnails(handle: str, pages: Any = None,
                               columns: int = 6,
                               pdf_pages: Any = None) -> str:
        """Contact sheets of every page (thumbnail + page number + kind),
        written as PNG files to look at with analyze_image."""
        args = {"handle": handle, "columns": columns}
        if pages not in (None, ""):
            args["pages"] = pages
        return _dispatch("render_page_thumbnails",
                         _with_pdf_pages(args, pdf_pages))

    def document_page_map(handle: str, pages: Any = None, kind: str = "",
                          with_evidence: bool = False,
                          pdf_pages: Any = None) -> str:
        """One row per page: kind, label, heading, counts."""
        args = {"handle": handle}
        if pages not in (None, ""):
            args["pages"] = pages
        if kind:
            args["kind"] = kind
        if with_evidence:
            args["with_evidence"] = True
        return _dispatch("document_page_map", _with_pdf_pages(args, pdf_pages))

    def read_document(handle: str, pages: Any = None, start_line: int = 0,
                      with_locations: bool = False,
                      include_tables: bool = True,
                      include_markups: bool = True,
                      pdf_pages: Any = None) -> str:
        """Read pages: text (optionally with boxes), tables, markups."""
        args = {"handle": handle, "start_line": start_line,
                "with_locations": with_locations,
                "include_tables": include_tables,
                "include_markups": include_markups}
        if pages not in (None, ""):
            args["pages"] = pages
        return _dispatch("read_document", _with_pdf_pages(args, pdf_pages))

    def search_document(handle: str, pattern: str, pages: Any = None,
                        regex: bool = False, case_sensitive: bool = False,
                        include_markups: bool = True,
                        max_hits: int = 100, fuzzy: bool = False,
                        min_score: Optional[int] = None,
                        pdf_pages: Any = None) -> str:
        """Find text, hidden CAD text and markup comments. ``fuzzy=True``
        matches approximately (score 0-100, best first) for text read
        optically or plotted as strokes; ``min_score`` defaults to 80
        (planlens' own; the model is shown 80). Given without ``fuzzy`` it
        is not used, and the result says so (BACKLOG DR-32)."""
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
            if min_score is not None:
                args["min_score"] = min_score
        raw = _dispatch("search_document", _with_pdf_pages(args, pdf_pages))
        if min_score is not None and not fuzzy:
            raw = _json_with(raw, min_score_unused=MIN_SCORE_UNUSED_NOTE)
        return raw

    def find_quantities(handle: str, pages: Any = None, kinds: Any = None,
                        units: Any = None, include_markups: bool = True,
                        offset: int = 0, pdf_pages: Any = None) -> str:
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
        return _dispatch("find_quantities", _with_pdf_pages(args, pdf_pages))

    def document_markups(handle: str, pages: Any = None, author: str = "",
                         offset: int = 0, pdf_pages: Any = None) -> str:
        """The review record: every markup with author, date and target."""
        args = {"handle": handle, "offset": offset}
        if pages not in (None, ""):
            args["pages"] = pages
        if author:
            args["author"] = author
        return _dispatch("document_markups", _with_pdf_pages(args, pdf_pages))

    def annotate_document(handle: str, markups: Optional[list] = None,
                          output_path: str = "",
                          append: bool = True, check: bool = True,
                          remove: Optional[list] = None) -> str:
        """Write review comments onto a COPY of the PDF (planlens' own words
        describe the markups; this app resolves the output file and signs
        them). ``check`` (default on) looks at every mark on the marked copy,
        however it was anchored, and reports which are misplaced.

        ``remove`` takes marks this app wrote out of an existing marked copy
        (``output_path``) first -- each a markup id as document_markups lists
        it on that copy (``"p0.m1"``) or words from its comment OR its
        visible label naming one mark -- so ``remove`` plus ``markups``
        changes one mark and keeps the rest (live smoke wave 2b, C4: one
        comment was changed by rebuilding all nineteen; 2c, E9: a label's
        words were refused, as only comments were matched).

        The copy is written to a temporary file and swapped in, after any
        handle this conversation holds on it is closed (C4: MuPDF could not
        save over a copy the toolkit held open on Windows, and the error
        ended the turn).

        The result's ``in_file`` and the opening of its ``note`` say what the
        copy holds -- the marks this call added, the ones kept from before
        and who wrote those (live smoke 2c, E10: told only "a NEW file: the
        document you opened is unchanged" and ``n_written: 1``, a model said
        the document's five markups were missing from the copy).

        The signature is ALWAYS the app's (``markup_author``: the signed-in
        person via this app, as an AI draft), never the model's: brief 5
        caught an agent signing "AI Draft Review" unasked, which dropped the
        reviewer's name from every comment. So there is no ``author``
        argument."""
        from funhouse_agent import markup_check
        from funhouse_agent._fileio import hide_working_folder
        from funhouse_agent.error_text import tool_error
        # A markup may give pdf_page (1-based) instead of page (live smoke
        # 2c); converted once, so the check reads the same pages.
        markups, problem = _document_tools.markups_with_pages(markups)
        if problem is None:
            # A remove object with a key other than id/text is refused here,
            # before its words are matched (BACKLOG DR-32).
            problem = _document_tools.remove_problem(remove)
        if problem is not None:
            return json.dumps(problem)
        signer = markup_author or _document_tools.markup_author()
        # Words in remove name a mark by its comment OR its label (E9): the
        # ones naming exactly one mark become its id here; the rest are
        # refused here, never guessed.
        refused_here = []
        if remove:
            try:
                final = _document_tools.markup_output_path(output_path,
                                                           handle)
                if os.path.isfile(final):
                    with open(final, "rb") as fh:
                        named, refused_here = \
                            markup_check.name_marks_by_words(
                                fh.read(), remove, signer)
                    # Nothing left to take out: [""] still makes the copy
                    # the base for any new marks (the remover skips an
                    # empty entry).
                    remove = named or [""]
            except Exception:  # noqa: BLE001 - the remover's own matching stands
                refused_here = []
        try:
            # An IMAGE's marks are in its own pixels, as its looks give them
            # (VIS-26); None for a PDF.
            image = _document_tools.image_of(handle, attachments)
            result = _document_tools.write_marked_copy(
                handle, markups, output_path, append, signer,
                lambda args: _dispatch("annotate_document", args),
                remove=remove, image=image)
        except Exception as exc:  # noqa: BLE001 - a tool error, not the turn's
            return json.dumps(hide_working_folder(
                tool_error("annotate_document", exc)))
        # The check opens the marked copy by its ABSOLUTE path; the model is
        # handed the conversation-relative name, AFTER the check, so no
        # server path reaches it (live smoke 2a, B9: the last A6 leak). The
        # file tools resolve that name in the working folder.
        if not isinstance(result, dict):
            return json.dumps(hide_working_folder({"error": str(result)}))
        if refused_here and "error" not in result:
            result["not_removed"] = (list(result.get("not_removed") or [])
                                     + refused_here)
        out_pdf = result.get("output_path")
        if out_pdf and os.path.isfile(out_pdf) and "error" not in result:
            try:
                markup_check.describe_copy(result, signer)
            except Exception:  # noqa: BLE001 - planlens' own note stands
                pass
        if check and out_pdf and os.path.isfile(out_pdf) \
                and "error" not in result:
            try:
                block = markup_check.check_marks(out_pdf, result,
                                                 list(markups or []), engine)
            except Exception as exc:  # noqa: BLE001 - the file is written either way
                from funhouse_agent.error_text import error_line
                block = {"error": "the placement check failed: "
                                  + error_line(exc)}
            if block is not None:
                result["check"] = block
        # An image's marks were taken in its own pixels: the frame said, and
        # any box on a copy at another scale given back in pixels (VIS-26).
        _document_tools.image_frame_result(result)
        return json.dumps(hide_working_folder(result))

    def measure(source: str, page: Optional[int] = None, kind: str = "line",
                bbox: Optional[list] = None, view: Optional[list] = None,
                image_box: Optional[list] = None,
                at: Optional[Dict[str, float]] = None,
                to: Optional[list] = None, scale: str = "",
                side: str = "top", pdf_page: Optional[int] = None) -> str:
        """Measure a position through the page's own scale (planlens'
        ``measure``; the app converts a look's view + image_box and reads a
        scan's label values itself). ``page`` counts from 0 (default 0);
        ``pdf_page`` is the same, 1-based."""
        from funhouse_agent.measure_tool import run_measure
        from funhouse_agent.page_numbers import resolve_page
        page, problem = resolve_page(page, pdf_page)
        if problem is not None:
            return json.dumps(problem)
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
                 offset: int = 0, pdf_pages: Any = None,
                 row_detail: str = "") -> str:
        """One boring or test-pit log as its grid (planlens' ``log_grid``;
        the app reads a textless scan's depth labels itself). ``row_detail``
        is compact (the default) or full."""
        from funhouse_agent.measure_tool import run_log_grid
        args: Dict[str, Any] = {"handle": handle, "rows": rows,
                                "offset": offset}
        if pages not in (None, ""):
            args["pages"] = pages
        if pdf_pages not in (None, ""):
            args["pdf_pages"] = pdf_pages
        if row_detail not in (None, ""):
            args["row_detail"] = row_detail
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
        args = {"path": path, "markdown": markdown, "title": title}
        if markup_author:
            # File > Info names the signed-in person (via the app), not
            # "python-docx" (live smoke wave 2a, B4).
            args["_author"] = markup_author
        return _with_saved_note(_dispatch("write_docx", args), save_fn)

    def plot_data(series: list, title: str = "", xlabel: str = "",
                  ylabel: str = "", depth_axis: bool = False,
                  logx: bool = False, logy: bool = False,
                  hlines: Optional[list] = None,
                  vlines: Optional[list] = None,
                  legend: Optional[bool] = None, output_path: str = "",
                  interactive: bool = True, equal_aspect: bool = False,
                  xlim: Optional[list] = None, ylim: Optional[list] = None,
                  legend_position: str = "auto", plan_view: bool = False,
                  north_deg: Optional[float] = None,
                  scale_units: str = "m", x_on_top: bool = False,
                  annotations: Optional[list] = None,
                  reference_chart: str = "") -> str:
        """Plot x/y data (the ``profile_figure`` module's ``plot_data``)."""
        params: Dict[str, Any] = {
            "series": series, "title": title, "xlabel": xlabel,
            "ylabel": ylabel, "depth_axis": depth_axis, "logx": logx,
            "logy": logy, "interactive": interactive,
            "equal_aspect": equal_aspect,
            "legend_position": legend_position or "auto",
            "plan_view": plan_view, "scale_units": scale_units or "m",
            "x_on_top": bool(x_on_top)}
        for k, v in (("hlines", hlines), ("vlines", vlines),
                     ("legend", legend), ("output_path", output_path or None),
                     ("xlim", xlim), ("ylim", ylim), ("north_deg", north_deg),
                     ("annotations", annotations),
                     ("reference_chart", reference_chart or None)):
            if v is not None:
                params[k] = v
        from funhouse_agent.drawing_wait import drawing
        try:
            from funhouse_agent._fileio import host_output_dir
            folder = host_output_dir()
        except Exception:                              # noqa: BLE001
            folder = None
        # Through GSE's dispatch until the chart stack moves (INF-30 step
        # 6): it adds defaults_used and the output note, and puts the file
        # in the working folder. Imported here, never at load.
        from funhouse_agent.dispatch import call_agent as _call_agent
        with drawing(folder):
            result = _call_agent(agent_name="profile_figure",
                                 method="plot_data", parameters=params,
                                 attachments=attachments, allowed_agents=None)
        # one figure's versions share a name: plan.png, plan2.png, plan_v3
        import re as _re
        key = os.path.splitext(os.path.basename(
            str(output_path or "").strip().lower()))[0]
        key = _re.sub(r"[_\- ]*(v|rev|version)?[_\- ]*\d+$", "", key)
        if key and isinstance(result, dict):
            draws[key] = draws.get(key, 0) + 1
            if draws[key] >= 3:
                # live wave w8 (F55): five redraws of a wall drawn as x/y
                # series, then the turn ran out of steps
                result["redraw_note"] = (
                    f"This figure has now been drawn {draws[key]} times. If "
                    "plot_data cannot draw what it needs (a structure to "
                    "scale with diagrams beside it, panels sharing one axis), "
                    "draw it with plot_code where you have it, or hand it to "
                    "the figure helper where there is one; otherwise say "
                    "what it cannot show rather than drawing it again.")
        from funhouse_agent import review_flags as _rf
        if (isinstance(result, dict) and result.get("status") == "success"
                and (_rf.figure_helper() or _rf.plot_code())):
            # Inside a helper, draws are counted across plot_data and
            # plot_code and across file names (live wave w9: six draws
            # against a cap of three, two of them here); and each figure's
            # draws across the turn, whoever draws it, with the arguments
            # that drew it kept beside it (live wave w12, GM-151); behind
            # the figure switches.
            from funhouse_agent import plot_box as _box
            before = _box.draws_so_far()
            drawn = [result.get("output_path"),
                     result.get("plotly_json_path")]
            booked = _box.figures.record_draw(drawn, "plot_data", folder,
                                              args=params)
            draw = _box.drawn_count(None if before is None else before + 1,
                                    booked["draw"])
            if draw is not None:
                result["draw"] = draw
                if draw >= _box.DRAW_CAP:
                    result["redraw_note"] = _box.draw_cap_note(draw)
                result["next"] = _box.next_step(
                    draw, renamed_drafts=plot_data_replaces,
                    checked_in="warnings")
            fig = (booked["figures"] or [{}])[0]
            if fig.get("code_file"):
                result["code_file"] = fig["code_file"]
            if fig.get("continues"):
                result["continues"] = fig["continues"]
            notes = _box.booked_notes(booked)
            if notes:
                result["figure_notes"] = notes
        return _truncate(json.dumps(result, default=str), max_result_chars)

    def write_xlsx(path: str, sheets: Optional[list] = None,
                   markdown: str = "") -> str:
        """Write an Excel workbook (one sheet per table)."""
        return _write_xlsx(path, sheets, markdown, save_fn)

    def annotate_dxf(source: str, notes: list, output_path: str = "",
                     layer: str = "REVIEW") -> str:
        """Add review notes to a COPY of a DXF drawing (a CAD library
        writes it; see the tool description)."""
        args: Dict[str, Any] = {"source": source, "notes": notes,
                                "layer": layer}
        if output_path:
            args["output_path"] = output_path
        return _with_saved_note(_dispatch("annotate_dxf", args), save_fn)

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
            "folder." + _PDF_PAGES_NOTE + " Each page in the result gives "
            "both (page, pdf_page).",
        ),
        "read_text_file": (
            read_text_file,
            "Read a file this conversation holds, by its name in the working "
            "folder: text (HTML, TXT, CSV, JSON, MD -- e.g. a report source "
            "written earlier) as it is; a Word .docx as Markdown in "
            "write_docx's own dialect (headings, lists, tables, bold/italic, "
            "line breaks), so a memo can be read, edited and written back "
            "with write_docx without drifting; an Excel .xlsx as one "
            "Markdown table per sheet, with the sheet names. The scratch "
            "read_file cannot see these files. limit = characters to return "
            "(6,000 by default, at most 12,000 a call), offset = where to "
            "start; the result's 'showing' says which characters came back "
            "of how many, and next_offset where the rest starts.",
        ),
        "analyze_image": (
            analyze_image,
            "Analyze an image using vision: an attached image (its key: an "
            "upload or a pasted screenshot), an image file in the working "
            "folder, or a scratch image a tool returned by name (.scratch/... "
            "from render_page_thumbnails or find_like). Returns text "
            "description/analysis of the image content. " + IMAGE_LOOK_NOTE,
        ),
        "analyze_pdf_page": (
            analyze_pdf_page,
            "Render a PDF page and analyze it using vision. tiles: 'auto' "
            "(default: the page is ALSO read in overlapping tiles when its "
            "lettering is too small in the whole-page image), 'off', or N or "
            "'NxN' with N from 2 to 4 (e.g. '3x3') for a fixed split. Boxes "
            "come back on a 0-999 grid with the result's view; a whole-page "
            "box says where to zoom, not where to put a mark. reuse=true "
            "returns this conversation's latest reading of the page (saying "
            "what it was asked) instead of a new look. " + TURN_ARG_NOTE
            + _PDF_PAGE_NOTE,
        ),
        "find_like": (
            find_like,
            "Find EVERY copy of one mark (tag, code, symbol) across a drawing "
            "set — even lettering drawn as lines — from one zoomed, confirmed "
            "example box + the text it reads; every candidate is verified by "
            "vision. Returns instances by page, callouts (with where each "
            "leader points) apart from legend entries, and uncertain reads. "
            "page / pages count from 0; pdf_page / pdf_pages are the same "
            "1-based, as a PDF viewer shows. " + IMAGE_FIND_NOTE,
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
            "is in it. reuse=true returns this conversation's kept zoom of "
            "the same region (saying what it was asked) instead of a new "
            "look. " + IMAGE_ZOOM_NOTE + " " + TURN_ARG_NOTE
            + _PDF_PAGE_NOTE,
        ),
        "kept_readings": (kept_readings, KEPT_READINGS_DESCRIPTION),
        # The tools another app adds (added_tools): GSE's reference
        # readers on the geotech page.
        **{name: build(_dispatch) for name, build in added.items()},
        **({name: (fn, _app_document_description(name))
            for name, fn in document_review_builders}
           if _document_tools.available() else {}),
        "save_file": (
            save_file,
            "Save raw text or data to a file. Returns the saved file path. "
            "For formatted calculation documents, use the calc_package module "
            "via call_agent instead. Saving code (a script, a notebook cell) "
            "does not run it: say a saved script was not run here, never "
            "that it was checked.",
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
        **({"plot_data": (plot_data,
                          _plot_description(plot_reference_charts))}
           if _plot_available() else {}),
        **({"write_xlsx": (write_xlsx, WRITE_XLSX_DESCRIPTION)}
           if _xlsx_available() else {}),
        **({"annotate_dxf": (annotate_dxf, ANNOTATE_DXF_DESCRIPTION)}
           if _dxf_notes_available() else {}),
    }

    overrides = dict(description_overrides or {})
    rewrites = dict(description_rewrites or {})
    tools = []
    for name, (fn, desc) in _builders.items():
        if name in include:
            text = overrides.get(name, desc)
            for old, new in rewrites.get(name, ()):
                text = text.replace(old, new)
            tool = StructuredTool.from_function(
                fn, name=name, description=text)
            if name == "plot_data" and (not plot_reference_charts
                                        or not _plot_takes_chart()):
                # a reference library without chart_lines (1.4.1 and
                # before): the argument is not offered at all (GM-153)
                tool = _without_argument(tool, "reference_chart")
            if name == "plot_data" and plot_data_replaces:
                tool = _plot_data_retiring(tool)
            if name in document_names:
                tool = _shown_as(tool, _document_tool_schemas(name))
            # An unknown argument (analyze_pdf_page(pages=12)) is refused by
            # name rather than dropped (brief 5, N3).
            tools.append(strict_tool(tool))
    return tools


def _plot_takes_chart() -> bool:
    """Whether plot_data takes ``reference_chart`` here: the adapter
    accepts it only where the reference library has chart_lines."""
    try:
        from funhouse_agent.adapters.profile_figure_adapter import (
            _VALID_PLOT)
    except Exception:                                  # noqa: BLE001
        return False
    return "reference_chart" in _VALID_PLOT


def _without_argument(tool, arg: str):
    """``tool`` with ``arg`` left out of its arguments (its function keeps
    the default)."""
    from pydantic import create_model
    base = tool.args_schema
    fields = {k: (f.annotation, f) for k, f in base.model_fields.items()
              if k != arg}
    schema = create_model(base.__name__, __config__=base.model_config,
                          **fields)
    return StructuredTool(name=tool.name, description=tool.description,
                          args_schema=schema, func=tool.func)


#: Added to plot_data's description where it takes ``replaces``.
PLOT_DATA_REPLACES_NOTE = (
    " replaces: file names of drafts this turn drew under another name, by "
    "any helper (a picture or chart, or the name without its extension), "
    "removed once this figure is drawn, so the chat shows only the final "
    "figure.")


def _plot_data_retiring(tool):
    """``plot_data`` that also takes ``replaces``: the drafts a renamed
    redraw retires (:func:`funhouse_agent.plot_box.retire_drafts`), removed
    only when the new figure was drawn."""
    from pydantic import create_model
    schema = create_model(
        tool.args_schema.__name__, __base__=tool.args_schema,
        replaces=(Optional[List[str]], Field(
            default=None, description=PLOT_DATA_REPLACES_NOTE.strip())))
    inner = tool.func

    def plot_data(**kwargs) -> str:
        replaces = kwargs.pop("replaces", None)
        from funhouse_agent import plot_box
        # the figure this draw replaces is counted on (GM-151)
        with plot_box.figures.replacing(replaces):
            out = inner(**kwargs)
        if not replaces:
            return out
        try:
            res = json.loads(out)
        except (TypeError, ValueError):
            res = None
        if not isinstance(res, dict):
            # cut short (or not JSON): the draw cannot be confirmed, so
            # nothing is removed, and the result says so
            return (str(out) + "\n" + json.dumps({
                "replaces": "nothing was retired: this result was cut "
                            "short, so the draw could not be confirmed; "
                            "name the drafts again on the next draw"}))
        if res.get("status") != "success":
            res["replaces"] = "nothing was retired, because this draw failed"
            return json.dumps(res, default=str)
        res.update(plot_box.retire_drafts(
            replaces, keep=[res.get("output_path"),
                            res.get("plotly_json_path")]))
        return json.dumps(res, default=str)

    return StructuredTool(name=tool.name,
                          description=tool.description
                          + PLOT_DATA_REPLACES_NOTE,
                          args_schema=schema, func=plot_data)


__all__ = [
    "make_vision_tools",
    "DEFAULT_MAX_RESULT_CHARS",
    "DEFAULT_REFERENCE_RESULT_CHARS",
    "DEFAULT_VISION_RESULT_CHARS",
    "SEARCH_NARROWER_NUDGE",
    "ANNOTATE_DXF_DESCRIPTION",
    "MIN_SCORE_UNUSED_NOTE",
    "NO_ZOOM_NOTE",
    "PLOT_DATA_REPLACES_NOTE",
    "WRITE_XLSX_DESCRIPTION",
]
