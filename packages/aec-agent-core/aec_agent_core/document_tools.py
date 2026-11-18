"""Whole-document review tools for the agent: the app's binding to
``planlens.review``.

BACKLOG PL-29 (b), the app split's step 9 (b). The document tools that only
wrapped planlens -- one toolkit per conversation, handles and sources
interchangeable, viewer page numbers, Word/Excel/DXF read as text, marked
copies, the markup check, ``find_like``, page numbers and the view frames --
moved into planlens 0.15 as ``planlens.review`` (part (a);
``module_work/pl29/CHECK.md``). planlens still makes no model call and
imports nothing of this app: what the app supplied is handed over here, in
one :class:`planlens.review.ReviewHost`:

- :func:`app_host` -- the fields that are the same for every conversation
  (the app's file confinement and its refusal words, the working folder the
  host bound for the turn, where the markup check keeps its verdicts,
  ``find_like``'s budget from ``GEOTECH_FIND_LIKE_BUDGET_S``, the Word/Excel
  reader, the image support, the error and record hooks, the hints naming
  this app's vision tools). It is installed for the process when this module
  is imported (``planlens.review.configure``), so a library caller (the
  cited-page check, ``measure_tool``) resolves and
  opens documents exactly as the agent's tools do;
- :func:`host_for` -- one host per agent build (per ``make_vision_tools``
  call): the base with that build's LOOK (``vision_tools.ask_vision`` with
  the build's engine), its SIGNER (the signed-in person, else
  :func:`markup_author`, read at call time) and its crop render (the
  engine's budget-sized ``vision_view.render_view``, unpadded). Every call
  the build's tools make passes it (``host=``), so two conversations served
  by one process never share a look or a signer.

The tools the agent is shown keep the app's own functions
(:mod:`funhouse_agent.deep.vision_kit`), so their names, descriptions and
schemas are what they were; their bodies call :func:`dispatch_document_tool`,
:func:`annotate_document` and :func:`find_like` here, which run planlens'
code under the build's host.

Tools and parameters planlens gained after the version this app pins are
FEATURE-DETECTED from the installed package's own specs (:func:`has_tool`,
:func:`search_supports_fuzzy`) and stay off the surface otherwise; the
detection is kept one release after the move (plan step 9 (b)).

The names other app modules and tests import from here (``handle_source``,
``write_marked_copy``, ``resolve_document_source``, ...) are planlens.review's
own objects, imported below; any other moved name is looked up there too
(:func:`__getattr__`). These aliases go at the split's step 16. A test that
patches moved code patches it in ``planlens.review``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional

from planlens import review as _review
from planlens.review import conversation as _conv
from planlens.review import marked_copy as _marks
from planlens.review import specs as _rspecs

# -- moved to planlens.review, imported here for the app's callers ------------
from planlens.review.conversation import (  # noqa: F401 - aliases
    CAD_DOCUMENT_EXTENSIONS, OFFICE_DOCUMENT_EXTENSIONS, SIDE_READ_TOOLS,
    UNCAPPED_BUDGET, _ATTACHMENTS, available, budget_for_cap,
    document_entry, forget_path, handle_source, looks_like_handle,
    open_document_entry, resolve_document_source, side_kind,
    with_viewer_pages, _toolkit,
)
from planlens.review.marked_copy import (  # noqa: F401 - aliases
    AUTHOR_IGNORED_NOTE, MarkedCopyNotWritten, REMOVE_ENTRY_KEYS,
    WRITE_FAILED_HINT, _document_name, empty_comments_note, image_frame_result,
    image_of, markup_output_path, markups_with_pages, remove_problem,
    write_marked_copy,
)
from planlens.review.specs import (  # noqa: F401 - aliases
    APP_MARKUP_FIELDS, DESCRIBED_MARKUP_FIELDS, WITHHELD_MARKUP_FIELDS,
    annotate_markups_schema,
)

#: The tools every planlens with the tool layer serves (0.3 and later). Kept as
#: a module constant because callers import it; ``document_tool_names()`` is
#: what the INSTALLED planlens actually offers.
DOCUMENT_TOOL_NAMES = (
    "open_document",
    "document_structure",
    "document_page_map",
    "read_document",
    "search_document",
    "document_markups",
    "render_page_thumbnails",
)

#: Served only by a newer planlens. Each one is advertised to the model only
#: when the installed package publishes a spec for it.
OPTIONAL_DOCUMENT_TOOL_NAMES = (
    "find_quantities",
    "annotate_document",
    # Visual scales (planlens main a3f2a1d+): a position measured through the
    # page's own scale, and a boring log read as its grid. The app wraps both
    # (funhouse_agent.measure_tool): a look's view + image_box, and a scan's
    # label values read by one vision call.
    "measure",
    "log_grid",
)

#: planlens' sheet tools (a drawing set's sheets read from their title
#: blocks, and its callouts checked against them; planlens after 0.15.1).
#: planlens offers them only to a host that asks, and this app asks only
#: where a page was built with them (``GEOTECH_REVIEW_SHEET_TOOLS``, BACKLOG
#: PL-37, DR-07): not measured yet, so not on by default.
SHEET_TOOL_NAMES = ("sheet_index", "check_cross_references")

#: Sheet tools a newer planlens added (the marks on a set's sheets checked
#: against its schedules, BACKLOG PL-42): offered with the two above where
#: the installed planlens publishes them, and never asked for otherwise.
#: With them, planlens' ``open_document`` also brings the sheet index and
#: that check for a host that asked for the sheet tools.
OPTIONAL_SHEET_TOOL_NAMES = ("check_schedule_marks",)

#: planlens' plan tools (a vector plan's symbols with their exact centres
#: and labels, and page points turned into local easting / northing;
#: BACKLOG PL-25, PL-27). Every planlens the core takes publishes them, but
#: they are NOT in :data:`OPTIONAL_DOCUMENT_TOOL_NAMES`: a page offers them
#: only where it was built with them (``GEOTECH_REVIEW_PLAN_TOOLS``), not
#: measured yet, so not on by default.
PLAN_TOOL_NAMES = ("plan_symbols", "plan_grid")

#: planlens' drawing tool (one page's dimensions, leaders, callouts,
#: revision clouds and title block read from its drawn geometry; BACKLOG
#: DR-44, PL-13; planlens after 0.16.0). A page offers it only where it was
#: built with it (``GEOTECH_REVIEW_DRAWING_TOOLS``) and the installed
#: planlens has it.
DRAWING_TOOL_NAMES = ("drawing_geometry",)

#: Who a comment this app writes onto a PDF is signed by. A reviewer opening
#: the marked-up file has to be able to tell a drafted comment from a person's
#: without asking, so the default says so in the name; a deployment that signs
#: its reviews differently sets the env var.
MARKUP_AUTHOR_ENV = "GEOTECH_MARKUP_AUTHOR"
# DEFAULT_MARKUP_AUTHOR: "<the application's name> (AI draft)", and
# OWN_MARK_WORDS: the words in a markup's author that make it one this app
# wrote, so ``remove`` may take it out (besides the signer itself; another
# reviewer's markup in the copy is left alone). Both follow the application
# the page handed in (aec_agent_core.app_details; INF-33 step 13), so they
# are read at each use through the module's __getattr__ (at its end).


def _app_named(attr: str):
    """DEFAULT_MARKUP_AUTHOR and OWN_MARK_WORDS, from the application's
    details; None for any other name."""
    if attr == "DEFAULT_MARKUP_AUTHOR":
        from aec_agent_core.app_details import markup_author as _author
        return _author()
    if attr == "OWN_MARK_WORDS":
        from aec_agent_core.app_details import own_mark_words
        return own_mark_words()
    return None

#: How the model views a PNG planlens wrote (the thumbnail contact sheets):
#: the app's analyze_image accepts a real path as its attachment_key.
IMAGE_VIEW_HINT = ("view the image with analyze_image(attachment_key="
                   "<image_path>, prompt=<what to look for>)")

#: Appended by planlens to every "! look:" line. It must name the app's own
#: vision tools and their argument conventions, because that line is the
#: model's cue to switch from reading to looking.
VISION_HINT = (
    "to look: analyze_pdf_page(attachment_key=<the source you opened>, "
    "page=N, prompt=<what to find>) views the whole page; "
    "render_region(attachment_key=<source>, page=N, bbox=[x0,y0,x1,y1], "
    "prompt=...) zooms on a box from read_document or a markup (same frame, "
    "no conversion); marks=[[x,y,label],...] numbers spots to ask about; "
    "a vision result's view + a 0-999 image_box from its analysis zooms on "
    "what it found"
)

def _number_setting(name: str, default: float) -> float:
    """The number in the environment variable ``name``, else ``default``: a
    value that is not a number (a typing slip in an app setting) must not
    stop this module importing, which stops the app starting."""
    try:
        value = float((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return float(default)
    # "inf" and "nan" are floats, and neither is a count or a time
    return value if value == value and abs(value) != float("inf") \
        else float(default)


#: Conversations whose toolkits are kept open at once; past it the least
#: recently used idle one is closed (its handles reopen on their next use).
MAX_LIVE_TOOLKITS = int(_number_setting("GEOTECH_DOCUMENT_TOOLKITS", 8))
#: A conversation's toolkit idle this long (seconds) is closed.
IDLE_CLOSE_S = _number_setting("GEOTECH_DOCUMENT_TOOLKIT_IDLE_S", 1800)

#: Seconds one ``find_like`` call may spend in all, read on every call
#: (unset: planlens' default, 300 s; ``0`` or less: no limit).
FIND_LIKE_BUDGET_ENV = "GEOTECH_FIND_LIKE_BUDGET_S"


# ---------------------------------------------------------------------------
# Feature detection (kept one release after the move; plan step 9 (b))
# ---------------------------------------------------------------------------

def tool_description(name: str) -> str:
    """The model-facing description planlens publishes for ``name``."""
    spec = _spec(name)
    if spec is None:
        raise KeyError(name)
    return spec["description"]


def _spec(name: str) -> Optional[Dict[str, Any]]:
    """The installed planlens' spec for ``name``, or ``None``."""
    try:
        from planlens.tools import specs as _specs
    except ImportError:
        return None
    # The sheet tools' specs are kept in a list of their own (a host that
    # reads TOOL_SPECS gets no tool it did not ask for).
    # ... and so are the drawing tools' (planlens after 0.16.0).
    for spec in list(_specs.TOOL_SPECS) + list(
            getattr(_specs, "SHEET_TOOL_SPECS", ())) + list(
            getattr(_specs, "DRAWING_TOOL_SPECS", ())):
        if spec.get("name") == name:
            return spec
    return None


def has_tool(name: str) -> bool:
    """Whether the installed planlens publishes a tool called ``name``."""
    return _spec(name) is not None


def document_tool_names() -> tuple:
    """The document tools to put on the agent's surface.

    The fixed seven, plus every optional tool the installed planlens actually
    publishes. Computed per call, not at import, so the set follows the package
    that is installed rather than the one this app was written against.
    """
    return DOCUMENT_TOOL_NAMES + tuple(
        name for name in OPTIONAL_DOCUMENT_TOOL_NAMES if has_tool(name))


def sheet_tools_available() -> bool:
    """Whether the installed planlens has the sheet tools
    (:data:`SHEET_TOOL_NAMES`) and a host can ask for them."""
    return (all(has_tool(name) for name in SHEET_TOOL_NAMES)
            and "toolkit_options" in {
                f.name for f in dataclasses.fields(_review.ReviewHost)})


def sheet_tool_names() -> tuple:
    """The sheet tools to put on a page built with them: the two of
    :data:`SHEET_TOOL_NAMES`, then each of
    :data:`OPTIONAL_SHEET_TOOL_NAMES` the installed planlens publishes.
    Empty where :func:`sheet_tools_available` is not."""
    if not sheet_tools_available():
        return ()
    return SHEET_TOOL_NAMES + tuple(
        name for name in OPTIONAL_SHEET_TOOL_NAMES if has_tool(name))


def plan_tools_available() -> bool:
    """Whether the installed planlens publishes the plan tools
    (:data:`PLAN_TOOL_NAMES`)."""
    return all(has_tool(name) for name in PLAN_TOOL_NAMES)


def plan_tools_need_asking() -> bool:
    """Whether the installed planlens.review serves the plan tools only to
    a host that asks (``toolkit_options={"plan_tools": True}``, planlens
    after 0.16.0, BACKLOG PL-27). An older one serves them to every host
    and must not be handed the option (its toolkit would refuse it)."""
    return hasattr(_rspecs, "PLAN_TOOLS")


#: The plan tools' arguments shown with planlens' own schema of them (type,
#: limits and words, read off its spec): all but ``handle`` and ``page`` /
#: ``pdf_page``. The same rule as ``planlens.review.specs.
#: PLAN_SHOWN_FROM_TOOLKIT``, kept here for a planlens without it.
PLAN_SHOWN_FROM_PLANLENS = {
    "plan_symbols": ("label_pattern", "shapes", "min_size", "max_size",
                     "max_label_gap", "bbox"),
    "plan_grid": ("controls", "north", "scale", "origin", "points",
                  "on_mirror", "units"),
}


def plan_tool_schemas(name: str) -> Dict[str, Dict[str, Any]]:
    """``{argument: schema}`` for plan tool ``name``: each argument of
    :data:`PLAN_SHOWN_FROM_PLANLENS` as the installed planlens' spec shows
    it (a copy). Empty for any other tool."""
    import copy
    spec = _spec(name) or {}
    props = (spec.get("parameters") or {}).get("properties") or {}
    return {arg: copy.deepcopy(props[arg])
            for arg in PLAN_SHOWN_FROM_PLANLENS.get(name, ())
            if isinstance(props.get(arg), dict)}


def _host_takes_options() -> bool:
    """Whether a host can ask planlens.review for anything
    (``ReviewHost.toolkit_options``)."""
    return "toolkit_options" in {
        f.name for f in dataclasses.fields(_review.ReviewHost)}


def sheet_address_args(name: str) -> list:
    """``[(argument, schema)]``: the sheet-number arguments document tool
    ``name`` takes for a host that asked for the sheet tools (``sheets``
    where it takes ``pages``, ``sheet`` where it takes ``page``; BACKLOG
    PL-37), each with planlens' own schema and words
    (``planlens.review.specs.sheet_args``). Empty where the installed
    planlens does not offer them (0.16.0 and before), and for a tool that
    takes no page. Read from planlens' own specs, never from its version."""
    offered = getattr(_rspecs, "sheet_args", None)
    if not callable(offered):
        return []
    import copy
    try:
        return [(str(arg), copy.deepcopy(schema))
                for arg, schema in offered(name, True)
                if isinstance(schema, dict)]
    except Exception:  # noqa: BLE001 - an unreadable spec offers nothing
        return []


def drawing_tools_available() -> bool:
    """Whether the installed planlens has the drawing tools
    (:data:`DRAWING_TOOL_NAMES`) and its host layer offers them to a host
    that asks (``planlens.review.specs.DRAWING_TOOLS``)."""
    offered = set(getattr(_rspecs, "DRAWING_TOOLS", ()) or ())
    return (_host_takes_options()
            and all(name in offered and has_tool(name)
                    for name in DRAWING_TOOL_NAMES))


def drawing_tool_schemas(name: str) -> Dict[str, Dict[str, Any]]:
    """``{argument: schema}`` for drawing tool ``name``: each argument
    planlens' host layer shows with the toolkit's own schema
    (``planlens.review.specs.DRAWING_SHOWN_FROM_TOOLKIT``), as the
    installed planlens' spec shows it (a copy). Empty for any other tool."""
    import copy
    shown = (getattr(_rspecs, "DRAWING_SHOWN_FROM_TOOLKIT", None) or {})
    spec = _spec(name) or {}
    props = (spec.get("parameters") or {}).get("properties") or {}
    return {arg: copy.deepcopy(props[arg])
            for arg in shown.get(name, ())
            if isinstance(props.get(arg), dict)}


def dxf_plots_available() -> bool:
    """Whether the installed planlens can open a DXF as a PDF plot of its
    model space for a host that asks (BACKLOG DR-19; planlens after
    0.16.0): its toolkit takes ``dxf_plots`` and it has the plot itself
    (``planlens.dxf.plot``, which needs ezdxf, and Pillow for ezdxf's
    drawing add-on: without Pillow planlens refuses every plot,
    ``PlotUnavailable``, so the page must not say a DXF is a plot).
    planlens publishes no list of the options a host may ask with, so the
    toolkit's own signature is read; an older toolkit would refuse the
    option."""
    if not _host_takes_options():
        return False
    import importlib.util
    import inspect
    try:
        from planlens.tools import ReviewToolkit
        takes = "dxf_plots" in inspect.signature(
            ReviewToolkit.__init__).parameters
        return bool(takes
                    and importlib.util.find_spec("planlens.dxf.plot")
                    and importlib.util.find_spec("ezdxf")
                    and importlib.util.find_spec("PIL"))
    except Exception:  # noqa: BLE001 - not there is not offered
        return False


def search_supports_fuzzy() -> bool:
    """Whether the installed ``search_document`` takes ``fuzzy``."""
    spec = _spec("search_document")
    if spec is None:
        return False
    schema = spec.get("parameters") or spec.get("input_schema") or {}
    return "fuzzy" in (schema.get("properties") or {})


def markup_author() -> str:
    """The author ``annotate_document`` signs this app's comments with."""
    from aec_agent_core.app_details import markup_author as _default
    return (os.environ.get(MARKUP_AUTHOR_ENV) or "").strip() or _default()


REPORT_INGEST_TOOLS = ("document_roles", "log_grid")


def report_ingest_supported() -> bool:
    """Whether the installed planlens can serve the whole-report ingest."""
    return available() and all(has_tool(name) for name in REPORT_INGEST_TOOLS)


# ---------------------------------------------------------------------------
# The host: what the app hands planlens.review
# ---------------------------------------------------------------------------

def _resolve_source(source: str):
    """A source the document tools could not find among the call's
    attachments or the conversation's handles (planlens takes those steps
    first): a file this conversation may read -> its path. A path outside
    what this conversation may read is refused, naming the conversation's
    files rather than any server path."""
    from planlens.tools import ToolError
    from aec_agent_core.vision_tools import (PathRefused, files_here_text,
                                             find_readable_file, read_roots)
    attachments = _review.host.current_attachments()
    try:
        found = find_readable_file(source)
    except PathRefused as exc:
        raise ToolError(str(exc), hint=(
            f"attachment keys: {sorted(attachments) or 'none'}"))
    if found:
        return found
    if read_roots() is not None:
        raise ToolError(
            f"'{source}' is not an attachment key, a document handle this "
            "conversation opened, or a file in this conversation's working "
            "folder",
            hint=(f"attachment keys: {sorted(attachments) or 'none'}; files "
                  f"here: {files_here_text()}"))
    raise ToolError(
        f"'{source}' is not an attachment key or a readable file path",
        hint=(f"attachment keys: {sorted(attachments) or 'none'}; real paths "
              f"(/tmp/..., /Volumes/...) also work"))


def _read_office(path: str, name: str, image_dir=None, image_ref=None):
    """A Word or Excel file as ``(markdown, fields)``
    (``funhouse_agent.office_text``, the reader read_text_file uses); a
    reader without the picture arguments falls back to plain Markdown."""
    from aec_agent_core import office_text
    try:
        return office_text.read_office(path, image_dir=image_dir,
                                       image_ref=image_ref)
    except TypeError:            # a reader without the picture arguments
        return office_text.office_to_markdown(path), {}


def _record(event: str, data: Dict[str, Any]) -> None:
    """A tool's event on the run's record (e.g. a wait for a document's
    lock; BACKLOG EV-14). Outside a run there is no record: nothing."""
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(event, data)
    except Exception:  # noqa: BLE001 - outside a run, or no logger
        pass


def _error_json(tool: str, exc: BaseException) -> str:
    from aec_agent_core.error_text import tool_error_json
    return tool_error_json(tool, exc)


def _after_marked_copy(result: Dict[str, Any], markups: list):
    """The documents a marked copy's notes cite, against what the
    conversation looked up (GEOTECH_CITED_DOCUMENTS; BACKLOG EV-21)."""
    from aec_agent_core import cited_documents
    return cited_documents.add_to_result(
        result, cited_documents.deliverable_texts(list(markups or [])),
        where="annotate_document")


def _working_folder():
    from aec_agent_core._fileio import host_output_dir
    return host_output_dir()


def _verdict_key():
    from aec_agent_core.vision_tools import reads_file
    return reads_file()


def _find_like_budget():
    return os.environ.get(FIND_LIKE_BUDGET_ENV)


def _display_path(path):
    from aec_agent_core.vision_tools import display_path
    return display_path(path)


def _readable_file(name):
    from aec_agent_core.vision_tools import find_readable_file
    return find_readable_file(name)


def _output_path(name):
    from aec_agent_core._fileio import resolve_output_path
    return resolve_output_path(name)


def _hide_paths(obj):
    from aec_agent_core._fileio import hide_working_folder
    return hide_working_folder(obj)


def _find_like_source(key: str, attachments: Dict[str, bytes]) -> bytes:
    """The bytes ``find_like`` searches, by the vision tools' own resolver
    (their words for a key they cannot find)."""
    from aec_agent_core.vision_tools import _resolve_attachment_or_path
    return _resolve_attachment_or_path(key, attachments)[0]


def _open_image(data: bytes, name: str):
    from aec_agent_core.vision_tools import _open_image as open_image
    return open_image(data, name)


def _as_pdf(picture) -> bytes:
    from aec_agent_core.image_view import as_pdf
    return as_pdf(picture)


def _page_scale(picture) -> float:
    from aec_agent_core.image_view import page_scale
    return page_scale(picture)


def _shown_size(data: bytes):
    from aec_agent_core.image_view import shown_size
    return shown_size(data)


def _takes_sent_scale(images) -> bool:
    """Whether the installed planlens' ``ImageSupport`` takes ``sent_scale``
    (planlens after 0.16.0)."""
    try:
        return images is not None and "sent_scale" in {
            f.name for f in dataclasses.fields(images)}
    except TypeError:                # not a dataclass: an older planlens
        return False


def _sent_scale_for(engine):
    """``sent_scale(w, h)`` for planlens: the scale (pixels sent per pixel
    of the image) at which this engine's looks send a ``w`` x ``h`` pixel
    view of an image, 1 when it goes whole (``image_view.sent_size``)."""
    def sent_scale(w: float, h: float) -> float:
        from aec_agent_core import image_view
        w, h = max(1, int(round(float(w)))), max(1, int(round(float(h))))
        sw, sh = image_view.sent_size(w, h, engine)
        return min(sw / float(w), sh / float(h))

    return sent_scale


def app_host() -> "_review.ReviewHost":
    """The host fields that are the same for every conversation (CHECK §3).
    The per-build ones (look, signer, crop render) are :func:`host_for`'s."""
    from aec_agent_core.vision_tools import SCRATCH_DIR
    return _review.ReviewHost(
        resolve_source=_resolve_source,
        look=None,
        working_folder=_working_folder,
        signer=markup_author,
        verdict_key=_verdict_key,
        find_like_budget_s=_find_like_budget,
        display_path=_display_path,
        readable_file=_readable_file,
        output_path=_output_path,
        hide_paths=_hide_paths,
        render_crop=None,
        find_like_source=_find_like_source,
        images=_review.ImageSupport(open_image=_open_image, as_pdf=_as_pdf,
                                    page_scale=_page_scale,
                                    shown_size=_shown_size),
        read_office=_read_office,
        after_marked_copy=_after_marked_copy,
        record=_record,
        error_json=_error_json,
        vision_hint=VISION_HINT,
        image_view_hint=IMAGE_VIEW_HINT,
        scratch_dir=SCRATCH_DIR,
        own_mark_words=_app_named("OWN_MARK_WORDS"),
        max_live_toolkits=MAX_LIVE_TOOLKITS,
        idle_close_s=IDLE_CLOSE_S,
    )


def host_for(engine=None, markup_author: Optional[str] = None,
             sheet_tools: bool = False,
             plan_tools: bool = False,
             drawing_tools: bool = False,
             dxf_plots: bool = False) -> "_review.ReviewHost":
    """One agent build's host: the process's (:func:`app_host`, as
    installed) with that build's look, signer and crop render.

    ``look`` is ``vision_tools.ask_vision`` with ``engine`` (``None`` with no
    engine: ``find_like`` then returns unverified matches, as before);
    ``signer`` is ``markup_author`` (the signed-in person) or, read at call
    time, the deployment's :func:`markup_author`; the markup check's crop is
    ``vision_view.render_view`` sized to ``engine``, unpadded (the check's
    box already holds its margin). ``sheet_tools`` asks planlens for the
    sheet tools on this build's conversations (:data:`SHEET_TOOL_NAMES`);
    ``plan_tools`` for the plan tools (:data:`PLAN_TOOL_NAMES`), where the
    installed planlens serves them only to a host that asks
    (:func:`plan_tools_need_asking`; an older one serves them unasked);
    ``drawing_tools`` for the drawing tools (:data:`DRAWING_TOOL_NAMES`)
    and ``dxf_plots`` for a DXF opened also as a plot, each only where the
    installed planlens has it (:func:`drawing_tools_available`,
    :func:`dxf_plots_available`): an older one is not handed an option its
    toolkit would refuse."""
    who = markup_author

    def signer() -> str:
        return who or _deployment_author()

    def render_crop(pdf: bytes, page: int, bbox):
        from aec_agent_core import vision_view
        return vision_view.render_view(pdf, page=int(page), bbox=bbox,
                                       pad_frac=0.0, engine=engine)

    look = None
    if engine is not None:
        def look(image: bytes, question: str, kind: str) -> str:
            from aec_agent_core.vision_tools import ask_vision
            return ask_vision(engine, image, question, kind)

    host = dataclasses.replace(BASE_HOST, look=look, signer=signer,
                               render_crop=render_crop)
    images = getattr(host, "images", None)
    if engine is not None and _takes_sent_scale(images):
        # A mark anchored on an image view is judged by the scale this
        # engine's looks send that view at (BACKLOG VIS-26); a planlens
        # without the field judges it by its width, as before.
        host = dataclasses.replace(host, images=dataclasses.replace(
            images, sent_scale=_sent_scale_for(engine)))
    if sheet_tools:
        host = dataclasses.replace(host, toolkit_options={
            **(host.toolkit_options or {}), "sheet_tools": True})
    if plan_tools and plan_tools_need_asking():
        host = dataclasses.replace(host, toolkit_options={
            **(host.toolkit_options or {}), "plan_tools": True})
    if drawing_tools and drawing_tools_available():
        host = dataclasses.replace(host, toolkit_options={
            **(host.toolkit_options or {}), "drawing_tools": True})
    if dxf_plots and dxf_plots_available():
        host = dataclasses.replace(host, toolkit_options={
            **(host.toolkit_options or {}), "dxf_plots": True})
    return host


def checking_host(host: "_review.ReviewHost", engine) -> "_review.ReviewHost":
    """``host`` for ``annotate_document``: marks are checked only by an
    engine that can see images (no ``check`` block otherwise, as before)."""
    if engine is None or not hasattr(engine, "analyze_image"):
        return dataclasses.replace(host, look=None)
    return host


# ---------------------------------------------------------------------------
# The calls the agent's tools make
# ---------------------------------------------------------------------------

def dispatch_document_tool(name: str, arguments: Dict[str, Any],
                           attachments: Optional[Dict[str, bytes]] = None,
                           max_chars: Optional[int] = None,
                           cap: Optional[int] = None,
                           host: Optional["_review.ReviewHost"] = None) -> str:
    """Run one document tool (planlens.review's dispatcher) under ``host``
    (an agent build's, :func:`host_for`), else the process's. Never raises;
    see :func:`planlens.review.conversation.dispatch_document_tool`."""
    if host is None:
        return _conv.dispatch_document_tool(name, arguments, attachments,
                                            max_chars, cap)
    with _review.using(host):
        return _conv.dispatch_document_tool(name, arguments, attachments,
                                            max_chars, cap)


def annotate_document(arguments: Dict[str, Any],
                      attachments: Optional[Dict[str, bytes]] = None,
                      max_chars: Optional[int] = None,
                      host: Optional["_review.ReviewHost"] = None) -> str:
    """``annotate_document`` as planlens.review runs it: a marked copy
    signed by ``host``'s signer, every mark checked by its look."""
    return _review.call("annotate_document", arguments, attachments,
                        max_chars, host=host)


def find_like(arguments: Dict[str, Any],
              attachments: Optional[Dict[str, bytes]] = None,
              host: Optional["_review.ReviewHost"] = None) -> str:
    """``find_like`` as planlens.review runs it: every copy of one mark,
    each candidate read by ``host``'s look (unverified with none)."""
    from planlens.review import tools as _rtools
    if host is None:
        return _rtools.find_like(dict(arguments or {}), attachments)
    with _review.using(host):
        return _rtools.find_like(dict(arguments or {}), attachments)


def __getattr__(name: str):
    """DEFAULT_MARKUP_AUTHOR and OWN_MARK_WORDS (the application's; INF-33
    step 13), then any other name that moved to planlens.review (an alias
    until step 16): the toolkit per conversation, then the marked copy's."""
    named = _app_named(name)
    if named is not None:
        return named
    for module in (_conv, _marks, _rspecs):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_deployment_author = markup_author

#: The process's host: every planlens.review call that no agent build serves
#: with its own (the cited-page check, library callers).
BASE_HOST = _review.configure(app_host())
