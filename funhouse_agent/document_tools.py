"""Whole-document review tools for the agent, served by ``planlens.tools``.

The deep agent's primary tool surface gets ``open_document``,
``document_structure``, ``document_page_map``, ``read_document``,
``search_document``, ``document_markups`` and ``render_page_thumbnails`` —
tools that read a whole PDF as located, attributed data: a page map, text with
boxes, tables, the review markups and the hidden text CAD programs leave
behind. planlens owns the behaviour and the size discipline (every result valid
JSON inside the budget, longer results paged through cursors). This module only
connects it to the app:

- a ``source`` resolves against the conversation's attachments first, then
  the conversation's own files — the same order and the same confinement as
  ``read_pdf_text`` (:func:`funhouse_agent.vision_tools.find_readable_file`);
- one toolkit PER CONVERSATION (keyed by the working folder the host bound
  for the turn), so a document opened in one turn is still open, by handle,
  in the next, and a handle, the open-document limit and the "open handles"
  hint belong to the conversation that opened them. Until live smoke 1
  (2026-10-08, A4) one toolkit served the whole process: the hint listed
  other people's handles, any session could read them, and a ninth document
  opened by anyone closed someone else's. Idle conversations' toolkits are
  closed (:data:`MAX_LIVE_TOOLKITS`, :data:`IDLE_CLOSE_S`); a handle they
  had reopens from its source the next time it is used;
- a handle and a source are interchangeable (A7): a document tool given a
  file name or attachment key where it expects a handle opens it, and
  ``open_document`` / the vision tools given a handle read the document it
  stands for — opening and reading in one parallel batch works;
- the page thumbnails go to the conversation's scratch folder
  (:data:`funhouse_agent.vision_tools.SCRATCH_DIR`), named by a short name
  the vision tools resolve, never by their server path (A9);
- attachments reach the resolver through a context variable set for the
  duration of each call, so concurrent conversations resolve only their own
  uploads;
- the "how to look" instruction planlens appends to every ``! look:`` line
  names THIS app's vision tools (``analyze_pdf_page``, ``render_region``),
  which take the same ``source`` and the same displayed-frame boxes. The
  toolkit's own ``render_page`` / ``render_region`` are not exposed here —
  the app already routes images to the model through its vision engine;
- ``annotate_document`` is the one tool that WRITES. Its ``output_path`` is
  resolved into the conversation's working folder
  (:func:`markup_output_path`), so the marked-up PDF gets a download card and
  rides the SharePoint mirror like anything else the agent saves, and every
  comment is signed :data:`DEFAULT_MARKUP_AUTHOR` unless the deployment or the
  call says otherwise. It is written to a temporary file and swapped in
  (:func:`write_marked_copy`), and one mark can be removed or replaced by its
  id (live smoke wave 2b, C4);
- a Word, Excel or DXF file is read as TEXT (:func:`_side_document`): Word
  and Excel as Markdown, a DXF as its text and entities -- never through
  MuPDF (wave 2b, C2 / C8);
- no document tool raises: an error comes back as JSON with no server path
  (wave 2b, C3).

Tools and parameters planlens gained after the version this app pins are
FEATURE-DETECTED from the installed package's own specs
(:func:`has_tool`, :func:`search_supports_fuzzy`) and stay off the surface
otherwise — the cluster installs planlens from PyPI, which can be older than
the development checkout.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Dict, Optional

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

#: Who a comment this app writes onto a PDF is signed by. A reviewer opening
#: the marked-up file has to be able to tell a drafted comment from a person's
#: without asking, so the default says so in the name; a deployment that signs
#: its reviews differently sets the env var.
MARKUP_AUTHOR_ENV = "GEOTECH_MARKUP_AUTHOR"
DEFAULT_MARKUP_AUTHOR = "GeotechStaffEngineer (AI draft)"

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

#: Budget used when the host has disabled truncation: generous, still a bound.
UNCAPPED_BUDGET = 60000

#: Room left under the host's cap, so planlens' own limit is what binds and the
#: host's string truncation (which breaks JSON) never fires.
#: It also holds the viewer page numbers :func:`with_viewer_pages` adds (about
#: 14 characters a page reference; a 100-hit search needs ~1,400).
_CAP_MARGIN = 1500

_ATTACHMENTS: ContextVar[Optional[Dict[str, bytes]]] = ContextVar(
    "gse_document_attachments", default=None)

#: Conversations whose toolkits are kept open at once; past it the least
#: recently used idle one is closed (its handles reopen on their next use).
MAX_LIVE_TOOLKITS = int(os.environ.get("GEOTECH_DOCUMENT_TOOLKITS", "8") or 8)
#: A conversation's toolkit idle this long (seconds) is closed.
IDLE_CLOSE_S = float(os.environ.get("GEOTECH_DOCUMENT_TOOLKIT_IDLE_S", "1800")
                     or 1800)
#: Conversations remembered at all (handle -> source names; a few strings).
MAX_SPACES = 512
#: Handles a conversation remembers the source of.
MAX_REMEMBERED_HANDLES = 256


class _Space:
    """One conversation's documents: its toolkit (``None`` once closed for
    idleness) and the source each handle it opened came from."""

    def __init__(self, key: str, folder: Optional[str]):
        self.key = key
        self.folder = folder
        self.kit = None
        self.sources: "OrderedDict[str, str]" = OrderedDict()
        self.last_used = time.monotonic()
        self.busy = 0


_SPACES: "OrderedDict[str, _Space]" = OrderedDict()
_SPACES_LOCK = threading.Lock()

#: What a planlens document handle looks like (``doc_`` + a content hash).
_HANDLE = re.compile(r"^doc_[0-9a-f]{6,40}$")


def looks_like_handle(value: str) -> bool:
    """Whether ``value`` has the shape of a document handle."""
    return bool(_HANDLE.match(str(value or "").strip()))


def _space_key() -> tuple:
    """``(key, folder)`` of the conversation in this context: the working
    folder the host bound for the turn; ``("", None)`` for a library caller
    with none (one shared toolkit, as before)."""
    from funhouse_agent._fileio import host_output_dir
    folder = host_output_dir()
    if not folder:
        return "", None
    return os.path.normcase(os.path.realpath(folder)), folder


def _new_toolkit(folder: Optional[str]):
    from planlens.tools import ReviewToolkit
    kw: Dict[str, Any] = dict(resolve_source=_resolve, vision_hint=VISION_HINT,
                              image_view_hint=IMAGE_VIEW_HINT)
    if folder:
        from funhouse_agent.vision_tools import SCRATCH_DIR
        kw["output_dir"] = os.path.join(folder, SCRATCH_DIR)
    return ReviewToolkit(**kw)


def _close_quietly(kit) -> None:
    try:
        kit.close()
    except Exception:  # noqa: BLE001 - closing is housekeeping
        pass


def _space(create: bool = True, hold: bool = False) -> Optional[_Space]:
    """This conversation's space, its toolkit open; idle ones closed.
    ``hold`` marks a call in flight (release with :func:`_release`), so the
    toolkit cannot be closed under it."""
    key, folder = _space_key()
    to_close = []
    with _SPACES_LOCK:
        space = _SPACES.get(key)
        if space is None:
            if not create:
                return None
            space = _Space(key, folder)
            _SPACES[key] = space
        _SPACES.move_to_end(key)
        space.last_used = time.monotonic()
        if hold:
            space.busy += 1
        if space.kit is None:
            space.kit = _new_toolkit(folder)
        # Housekeeping: idle toolkits closed, then the oldest beyond the cap,
        # never one with a call in flight.
        now = time.monotonic()
        live = [s for s in _SPACES.values() if s.kit is not None]
        for s in live:
            if s is not space and not s.busy \
                    and now - s.last_used > IDLE_CLOSE_S:
                to_close.append(s.kit)
                s.kit = None
        live = [s for s in _SPACES.values() if s.kit is not None]
        for s in live:                       # oldest first
            if len(live) - len(to_close) <= MAX_LIVE_TOOLKITS:
                break
            if s is not space and not s.busy and s.kit is not None:
                to_close.append(s.kit)
                s.kit = None
        while len(_SPACES) > MAX_SPACES:
            oldest = next(iter(_SPACES.values()))
            if oldest is space or oldest.busy:
                break
            _SPACES.popitem(last=False)
            if oldest.kit is not None:
                to_close.append(oldest.kit)
    for kit in to_close:                     # outside the lock: may wait
        _close_quietly(kit)
    return space


def _remember(space: _Space) -> None:
    """Note the source of every document open in ``space``'s toolkit, so its
    handle keeps working after the document is closed (the conversation's
    limit, or an idle toolkit closed)."""
    kit = space.kit
    entries = getattr(kit, "_entries", None) if kit is not None else None
    if not entries:
        return
    with _SPACES_LOCK:
        for handle, entry in list(entries.items()):
            src = getattr(entry, "source", None) or getattr(entry, "path", None)
            if src and not looks_like_handle(str(src)):
                space.sources[handle] = str(src)
                space.sources.move_to_end(handle)
        while len(space.sources) > MAX_REMEMBERED_HANDLES:
            space.sources.popitem(last=False)


def handle_source(handle: str) -> Optional[str]:
    """The source (attachment key or file) a document ``handle`` was opened
    from IN THIS CONVERSATION, else ``None`` -- another conversation's
    handle means nothing here."""
    if not handle:
        return None
    space = _space(create=False)
    if space is None:
        return None
    _remember(space)
    with _SPACES_LOCK:
        return space.sources.get(str(handle))


def _release(space: _Space) -> None:
    """End a call :func:`_space` was asked to ``hold`` for."""
    with _SPACES_LOCK:
        space.busy = max(0, space.busy - 1)
        space.last_used = time.monotonic()
    _remember(space)


def available() -> bool:
    """Whether the installed planlens has the tool layer."""
    try:
        import planlens.tools  # noqa: F401
    except ImportError:
        return False
    return True


def budget_for_cap(cap: int) -> int:
    """planlens' size limit for a host result cap (``<= 0`` = uncapped)."""
    if cap <= 0:
        return UNCAPPED_BUDGET
    return max(1000, int(cap) - _CAP_MARGIN)


def tool_description(name: str) -> str:
    """The model-facing description planlens publishes for ``name``."""
    spec = _spec(name)
    if spec is None:
        raise KeyError(name)
    return spec["description"]


def _spec(name: str) -> Optional[Dict[str, Any]]:
    """The installed planlens' spec for ``name``, or ``None``."""
    try:
        from planlens.tools.specs import TOOL_SPECS
    except ImportError:
        return None
    for spec in TOOL_SPECS:
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


def search_supports_fuzzy() -> bool:
    """Whether the installed ``search_document`` takes ``fuzzy``."""
    spec = _spec("search_document")
    if spec is None:
        return False
    schema = spec.get("parameters") or spec.get("input_schema") or {}
    return "fuzzy" in (schema.get("properties") or {})


def markup_author() -> str:
    """The author ``annotate_document`` signs this app's comments with."""
    return (os.environ.get(MARKUP_AUTHOR_ENV) or "").strip() \
        or DEFAULT_MARKUP_AUTHOR


def _document_name(handle: str) -> Optional[str]:
    """What the open document behind ``handle`` was called, if anything is."""
    if not handle:
        return None
    getter = getattr(_toolkit(), "document_name", None)
    name = None
    if getter is not None:       # a planlens older than the accessor has none
        try:
            name = getter(handle)
        except Exception:        # pragma: no cover - a stale handle
            name = None
    if name is None:             # closed since: the name it was opened under
        src = (handle_source(handle) if str(handle).startswith("doc_")
               else handle)          # A7: a file name given as the handle
        name = os.path.basename(src.replace("\\", "/")) if src else None
    return name


def markup_output_path(output_path: str = "", handle: str = "") -> str:
    """Where a marked-up copy goes: this conversation's working folder.

    planlens is framework-neutral and writes ``output_path`` as given, so the
    app resolves it the way every other tool output is resolved — a bare name
    lands in the working folder this conversation writes to (where it gets a
    download card and rides the SharePoint mirror), an absolute path is
    honoured. With no name at all the copy is named after the document it
    marks up, which is what a reviewer expects to find beside the original.
    While a host has bound a working folder, an absolute path outside it
    keeps its file name and lands in the working folder (A2: a tool never
    writes into another conversation's folder).
    """
    from funhouse_agent._fileio import (host_output_dir, into_working_folder,
                                        resolve_output_path)
    name = str(output_path or "").strip()
    if not name:
        stem = os.path.splitext(_document_name(handle) or "document")[0]
        name = f"{stem}_marked.pdf"
    if not os.path.splitext(name)[1]:
        name += ".pdf"
    folder = host_output_dir()
    if folder:
        return into_working_folder(name, folder)
    return resolve_output_path(name)


def _resolve(source: str):
    """An attachment key, a handle this conversation opened, or a file of
    this conversation -> PDF bytes or a readable path (planlens' resolver).
    A path outside what this conversation may read is refused, naming the
    conversation's files rather than any server path."""
    from planlens.tools import ToolError
    from funhouse_agent.vision_tools import (PathRefused, files_here_text,
                                             find_readable_file, read_roots)
    attachments = _ATTACHMENTS.get() or {}
    if source in attachments:
        return attachments[source]
    remembered = handle_source(source) if looks_like_handle(source) else None
    if remembered and remembered != source:
        if remembered in attachments:
            return attachments[remembered]
        source = remembered
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


def _toolkit():
    """The toolkit of the conversation in this context (one per bound
    working folder; one shared toolkit for a library caller with none)."""
    return _space().kit


def dispatch_document_tool(name: str, arguments: Dict[str, Any],
                           attachments: Optional[Dict[str, bytes]] = None,
                           max_chars: Optional[int] = None,
                           cap: Optional[int] = None) -> str:
    """Run one document tool; returns a JSON string within ``max_chars``
    (``cap``, when given, is the host's own limit: the viewer page numbers
    are added only while the result stays under it).

    It never raises (live smoke wave 2b, C3: MuPDF's ``FzErrorSystem`` out
    of ``annotate_document`` ended a tester's whole turn and showed the
    server path): an error comes back as ``{"error", "hint"}`` with no
    server path in it, and the conversation goes on. A Word, Excel or DXF
    file is read here as text (:func:`_side_document`), never through
    MuPDF, which read a workbook as nine digits (C2) and could not open a
    DXF at all (C8).

    Pages: ``pdf_pages`` (1-based, as a viewer shows) is taken in place of
    ``pages`` and ``pdf_page`` in place of ``page``; a page the document
    does not have comes back as an error with a hint ("pages are 0-based
    here; PDF page 5 is page 4") -- live smoke wave 2c."""
    if not available():
        return json.dumps({
            "error": "document tools need planlens with planlens.tools "
                     "(0.3 or later)",
            "hint": "pip install -U planlens"})
    # A shallow snapshot: the host mutates its attachments dict between turns.
    token = _ATTACHMENTS.set(dict(attachments or {}))
    space = None
    opened_note = None
    one_based = False
    try:
        space = _space(hold=True)
        arguments = dict(arguments or {})
        one_based, refused = _one_based_pages(arguments)
        if refused is not None:
            return refused
        side = _side_document(name, arguments, max_chars)
        if side is not None:
            return side
        refused, opened_note = _as_handle_or_source(space, name, arguments)
        if refused is not None:
            return refused
        out = space.kit.call_json(name, arguments, max_chars=max_chars)
    except Exception as exc:  # noqa: BLE001 - a tool error, never the turn's
        from funhouse_agent.error_text import tool_error_json
        return _with_range_hint(tool_error_json(name, exc), one_based)
    finally:
        if space is not None:
            _release(space)
        _ATTACHMENTS.reset(token)
    out = _with_range_hint(out, one_based)
    # The margin planlens was given below the host's cap is where the viewer
    # page numbers fit; a result they would push past it goes out as it came.
    if cap and cap > 0:
        limit: Optional[int] = int(cap) - 20
    elif max_chars:
        limit = max_chars + _CAP_MARGIN - 20
    else:
        limit = None
    out = _short_image_paths(out)
    out = with_viewer_pages(out, limit)
    if opened_note:
        out = _with_note(out, "opened", opened_note, limit)
    return out


def _one_based_pages(arguments: Dict[str, Any]):
    """Rewrite ``pdf_pages`` / ``pdf_page`` (1-based) in ``arguments`` into
    the 0-based ``pages`` / ``page`` planlens takes, in place. Returns
    ``(one_based, error_json or None)``: giving both forms of one argument
    is refused unless they agree (never a guess)."""
    from funhouse_agent.page_numbers import (PAGE_ARGS_NOTE,
                                             pdf_pages_to_pages,
                                             resolve_page)
    one_based = False
    if "pdf_pages" in arguments:
        spec = arguments.pop("pdf_pages")
        if spec not in (None, "", [], ()):
            if arguments.get("pages") not in (None, "", [], ()):
                return True, json.dumps({
                    "error": "give pages (0-based) or pdf_pages (1-based), "
                             "not both",
                    "hint": "pdf_pages counts as a PDF viewer does; pages "
                            "counts from 0"})
            pages, problem = pdf_pages_to_pages(spec)
            if problem is not None:
                return True, json.dumps(problem)
            arguments["pages"] = pages
            one_based = True
    if "pdf_page" in arguments:
        given = arguments.pop("pdf_page")
        if given not in (None, ""):
            page, problem = resolve_page(arguments.get("page"), given)
            if problem is not None:
                problem.setdefault("hint", PAGE_ARGS_NOTE)
                return True, json.dumps(problem)
            arguments["page"] = page
            one_based = True
    return one_based, None


def _with_range_hint(out: str, one_based: bool = False) -> str:
    """``out`` (a tool's JSON) with a hint added when its error is a page
    out of range (:func:`funhouse_agent.page_numbers.range_hint`)."""
    if not isinstance(out, str) or '"error"' not in out[:400]:
        return out
    try:
        data = json.loads(out)
    except ValueError:
        return out
    if not isinstance(data, dict) or "error" not in data:
        return out
    from funhouse_agent.page_numbers import range_hint
    hint = range_hint(str(data["error"]), one_based)
    if not hint:
        return out
    data["hint"] = f"{hint}. {data['hint']}" if data.get("hint") else hint
    return json.dumps(data, ensure_ascii=False)


def _as_handle_or_source(space: _Space, name: str,
                         arguments: Dict[str, Any]):
    """A7 (live smoke 1: 29 of 43 tool errors): a handle and a source are
    interchangeable. ``open_document`` given a handle this conversation
    opened reads the document it stands for; any other document tool given
    a file name or attachment key where its ``handle`` goes opens that
    document first. Returns ``(error_json or None, note or None)`` and
    rewrites ``arguments`` in place."""
    kit = space.kit
    if name == "open_document":
        src = str(arguments.get("source") or "")
        if looks_like_handle(src):
            remembered = handle_source(src)
            if remembered:
                arguments["source"] = remembered
        return None, None
    if "handle" not in arguments:
        return None, None
    given = str(arguments.get("handle") or "").strip()
    if not given or given in getattr(kit, "_entries", {}):
        return None, None
    # A handle-shaped value (``doc_`` and no file extension) is a handle --
    # planlens answers an unknown one itself, naming this conversation's
    # open handles; anything else is a file name or attachment key.
    handle_like = looks_like_handle(given) or (
        given.startswith("doc_") and not os.path.splitext(given)[1])
    source = handle_source(given) if handle_like else given
    if not source:
        return None, None            # planlens says "unknown handle" itself
    from planlens.tools import ToolError
    try:
        entry = kit.open(source)
    except Exception as exc:  # noqa: BLE001 - ToolError, or a resolver slip
        if not isinstance(exc, ToolError):
            exc = ToolError(f"{type(exc).__name__}: {exc}")
        if handle_like:
            return None, None
        out = {"error": (f"'{given}' is not a document handle this "
                         f"conversation opened, and it did not open as a "
                         f"document: {exc}"),
               "hint": (getattr(exc, "hint", None)
                        or "call open_document with the file's name first")}
        return json.dumps(out, ensure_ascii=False), None
    arguments["handle"] = entry.handle
    if handle_like:
        return None, None            # reopened after it was closed: silent
    return None, (f"'{given}' was opened as document handle "
                  f"{entry.handle}; either name works in later calls")


def _with_note(result: str, key: str, note: str,
               limit: Optional[int]) -> str:
    """``result`` with ``{key: note}`` added, while it stays under ``limit``."""
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return result
    if not isinstance(data, dict) or key in data:
        return result
    data[key] = note
    out = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return result if limit is not None and len(out) > limit else out


def _short_image_paths(result: str) -> str:
    """Images the toolkit wrote (page thumbnails) named by their place in
    the conversation (``.scratch/doc_…_thumbs_0-29.png``), which the vision
    tools resolve -- never by their server path (A6, A9). Unchanged with no
    working folder bound."""
    if '"image_path"' not in result:
        return result
    from funhouse_agent.vision_tools import display_path, read_roots
    if read_roots() is None:
        return result
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return result

    def walk(obj):
        if isinstance(obj, dict):
            return {k: (display_path(v) if k == "image_path"
                        and isinstance(v, str) else walk(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        return obj

    return json.dumps(walk(data), ensure_ascii=False, separators=(",", ":"))


_PAGE_HEADER = re.compile(r"=== page (\d+)( continued)?")

#: A planlens page-range string ("0-2,5,7-9").
_RANGES = re.compile(r"^\s*\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\s*$")


def _viewer_ranges(text: str) -> str:
    """``"0-2,5"`` -> ``"1-3,6"``: a 0-based range string as a viewer counts."""
    return re.sub(r"\d+", lambda m: str(int(m.group(0)) + 1),
                  re.sub(r"\s+", "", text))


def _is_pages_key(key: Any) -> bool:
    """A key that holds page indexes (``pages``, ``pages_to_view``,
    ``duplicate_pages``, ``pages_with_hits`` ...), never a cursor."""
    k = str(key)
    return k != "next" and not k.startswith("pdf_") and (
        k == "pages" or k.startswith("pages_") or k.endswith("_pages"))


def _viewer_pages_value(value: Any) -> Any:
    """The viewer's numbering of a page-range value, or ``None`` when it is
    not one: a range string, a ``{kind: ranges}`` map, or a
    ``{"<page>": n}`` map."""
    if isinstance(value, str) and _RANGES.match(value):
        return _viewer_ranges(value)
    if isinstance(value, dict) and value:
        if all(str(k).isdigit() for k in value):
            return {str(int(k) + 1): v for k, v in value.items()}
        if all(isinstance(v, str) and _RANGES.match(v)
               for v in value.values()):
            return {k: _viewer_ranges(v) for k, v in value.items()}
    return None


def _viewer_pages(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            # A cursor ("next") is handed back to the tool as it is: no
            # viewer numbers in it to be mistaken for an argument.
            out[k] = v if k == "next" else _viewer_pages(v)
            if (k == "page" and isinstance(v, int) and not isinstance(v, bool)
                    and "pdf_page" not in obj):
                out["pdf_page"] = v + 1
            elif _is_pages_key(k) and f"pdf_{k}" not in obj:
                # Live smoke wave 2b (C9): a single page had its viewer
                # number, a range did not, and chapter starts were cited one
                # page low. Every range now carries the viewer's numbers too.
                shown = _viewer_pages_value(v)
                if shown is not None:
                    out[f"pdf_{k}"] = shown
        return out
    if isinstance(obj, list):
        return [_viewer_pages(v) for v in obj]
    if isinstance(obj, str) and "=== page " in obj:
        return _PAGE_HEADER.sub(
            lambda m: f"=== page {m.group(1)} [pdf_page "
                      f"{int(m.group(1)) + 1}]{m.group(2) or ''}", obj)
    return obj


def with_viewer_pages(result: str, limit: Optional[int] = None) -> str:
    """``result`` with the page a PDF VIEWER shows beside every page index.

    planlens (like every tool here) numbers pages from 0, and its results
    cite "page 28"; a reader opening the file in Bluebeam or Acrobat finds
    that content on page 29. Every citation drawn from a result with no
    printed page number was one page early (review of 2026-09-26). So each
    ``"page": n`` gains ``"pdf_page": n + 1``, each ``=== page n`` header
    in read text gains ``[pdf_page n+1]`` — the number the prompt says to cite
    — and each page RANGE (``"pages": "0-12"``, ``pages_by_kind``,
    ``pages_to_view``, ``pages_with_hits`` ...) gains a ``pdf_`` twin in the
    viewer's numbers (``"pdf_pages": "1-13"``; live smoke wave 2b, C9). The
    tools still TAKE 0-based pages, as their descriptions say.
    Unparseable results, and results the additions would push past
    ``limit``, are returned unchanged.
    """
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return result
    # Serialized the way planlens serializes, so the length compares like for
    # like with the budget it was given.
    out = json.dumps(_viewer_pages(data), ensure_ascii=False,
                     separators=(",", ":"))
    if limit is not None and len(out) > limit:
        return result
    return out


# ---------------------------------------------------------------------------
# Word, Excel and DXF files: read as text, never as pages
# ---------------------------------------------------------------------------
# Live smoke wave 2b. C2: an .xlsx fetched from SharePoint went to MuPDF's
# Office conversion, which dropped every shared-string cell; read_document
# returned "0 0 0 0 0 1 0 1 0" and the tester was told three times their log
# was "probably damaged". C8: a DXF could not be opened at all ("could not
# open ... as a PDF or image: FileDataError"), though planlens reads DXF
# exactly (planlens.ir.ingest.from_dxf). Both are now read here, by name:
# open_document returns the text (and, for a DXF, its layers and entities),
# read_document pages through it by line, search_document finds lines in it.
# The file's NAME stands in for a handle -- there are no pages to hand to the
# page, markup and vision tools, and those say so instead of guessing.

#: Word and Excel files (the older .doc / .xls are named so the reader can
#: say what to save them as).
OFFICE_DOCUMENT_EXTENSIONS = (".docx", ".xlsx", ".xlsm", ".doc", ".xls")
#: CAD drawings read as data.
CAD_DOCUMENT_EXTENSIONS = (".dxf",)
#: Tools that work on such a file (the rest need pages).
SIDE_READ_TOOLS = ("open_document", "read_document", "search_document")
#: Files read this way kept in memory (by path, mtime and size).
_SIDE_CACHE_MAX = 16
_SIDE_CACHE: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
_SIDE_LOCK = threading.Lock()


def side_kind(name: Any) -> Optional[str]:
    """``"office"`` / ``"cad"`` for a file read as text rather than pages
    (by its extension), else ``None``."""
    ext = os.path.splitext(str(name or "").strip())[1].lower()
    if ext in OFFICE_DOCUMENT_EXTENSIONS:
        return "office"
    if ext in CAD_DOCUMENT_EXTENSIONS:
        return "cad"
    return None


def _side_source(name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """The Word/Excel/DXF file this call is about, as the model named it (a
    handle this conversation opened from one counts), or ``None``."""
    key = "source" if name == "open_document" else "handle"
    given = str(arguments.get(key) or "").strip()
    if not given:
        return None
    if side_kind(given):
        return given
    if looks_like_handle(given):
        remembered = handle_source(given)
        if remembered and side_kind(remembered):
            return remembered
    return None


def _side_path(source: str):
    """``(path, temporary)`` of a readable copy of ``source``: the file in
    the working folder, or the attachment's bytes written to a temporary
    file of the same type. Raises planlens' ``ToolError`` as the document
    tools do."""
    resolved = _resolve(source)
    if isinstance(resolved, (bytes, bytearray)):
        import tempfile
        ext = os.path.splitext(source)[1] or ".bin"
        fd, tmp = tempfile.mkstemp(suffix=ext, prefix="gse_side_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(bytes(resolved))
        return tmp, True
    return str(resolved), False


def _dxf_units_in(path: str) -> Optional[int]:
    """The ``$INSUNITS`` code a text DXF's header states, if any."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(400_000).decode("latin-1", errors="replace")
    except OSError:
        return None
    m = re.search(r"\$INSUNITS\s*\r?\n\s*70\s*\r?\n\s*(-?\d+)", head)
    return int(m.group(1)) if m else None


#: $INSUNITS codes (DXF reference) -> unit names.
_INSUNITS_NAMES = {0: None, 1: "in", 2: "ft", 3: "mi", 4: "mm", 5: "cm",
                   6: "m", 7: "km", 8: "microinches", 9: "mils", 10: "yd"}


def _fmt(v: float) -> str:
    return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.1f}"


# -- A DXF's geometry, listed with the text it sits by (wave 2c, E4) ----------
# F40: the boring circles, the building outline and the arrow line were
# counted but not listed, so `search_document("CIRCLE")` found nothing and
# the model read coordinates off raw DXF pages, 3-4 calls a turn.

#: Geometry lines listed per DXF at most (the text is always listed whole).
CAD_GEOMETRY_MAX = 20000
#: Vertices written out per polyline or hatch boundary.
CAD_VERTICES_SHOWN = 12
#: Grid cells one entity may search for nearby text (a sheet border's
#: outline is not "near" any one label).
_NEAR_MAX_CELLS = 2500
#: Geometry kinds listed, in this order.
_CAD_GEOMETRY_KINDS = ("circle", "arc", "polyline", "line", "region")


def _seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx
                                               + (p[1] - ay) * dy) / L2))
    return ((p[0] - ax - t * dx) ** 2 + (p[1] - ay - t * dy) ** 2) ** 0.5


def _inside(p, ring) -> bool:
    """Whether point ``p`` lies inside the closed ``ring`` (ray casting)."""
    x, y, hit = p[0], p[1], False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (y1 > y) != (y2 > y) and \
                x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            hit = not hit
    return hit


def _ring_of(e):
    kind = getattr(e, "KIND", "")
    if kind == "polyline":
        return list(getattr(e, "vertices", None) or []), \
            bool(getattr(e, "closed", False))
    if kind == "region":
        return list(getattr(e, "boundary", None) or []), True
    return [], False


def _distance_to(e, p) -> float:
    """Distance from point ``p`` to entity ``e``'s drawn geometry."""
    kind = getattr(e, "KIND", "")
    if kind in ("circle", "arc"):
        c, r = getattr(e, "center", (0.0, 0.0)), getattr(e, "radius", 0.0)
        return abs(((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5 - r)
    if kind == "line":
        return _seg_dist(p, e.start, e.end)
    pts, closed = _ring_of(e)
    if not pts:
        return float("inf")
    if len(pts) == 1:
        return _seg_dist(p, pts[0], pts[0])
    segs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
    return min(_seg_dist(p, a, b) for a, b in segs)


class _TextGrid:
    """The drawing's text items in grid cells of the "near" radius."""

    def __init__(self, texts, radius):
        self.texts, self.r = texts, radius
        self.cells: Dict[tuple, list] = {}
        for i, t in enumerate(texts):
            self.cells.setdefault(self._cell(t[0]), []).append(i)

    def _cell(self, p):
        return (int(p[0] // self.r), int(p[1] // self.r))

    def candidates(self, box):
        """Text indices within the radius of ``box``, or ``None`` when the
        box is too large to say one label is near it."""
        x0, y0 = self._cell((box[0] - self.r, box[1] - self.r))
        x1, y1 = self._cell((box[2] + self.r, box[3] + self.r))
        if (x1 - x0 + 1) * (y1 - y0 + 1) > _NEAR_MAX_CELLS:
            return None
        out = []
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                out.extend(self.cells.get((ix, iy), ()))
        return out


def _near_text(e, grid: Optional[_TextGrid]) -> str:
    """`` near "B-2"`` (the closest text within the radius) and, for a
    closed outline, `` encloses "BUILDING"``; ``""`` when neither."""
    if grid is None or getattr(e, "bbox", None) is None:
        return ""
    idx = grid.candidates(e.bbox)
    if not idx:
        return ""
    bits = []
    ring, closed = _ring_of(e)
    if closed and 3 <= len(ring) <= 2000:
        # The label of an outline sits near its middle: the texts whose
        # middles are nearest its middle first.
        cx, cy = (e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2
        inside = sorted(
            (grid.texts[i] for i in idx if _inside(grid.texts[i][0], ring)),
            key=lambda t: (t[2][0] - cx) ** 2 + (t[2][1] - cy) ** 2)
        if inside:
            more = len(inside) - 2
            bits.append("encloses " + ", ".join(
                f'"{t[1]}"' for t in inside[:2]) + (
                f" and {more} more text{'s' if more > 1 else ''}"
                if more > 0 else ""))
    best, best_d = None, grid.r
    for i in idx:
        d = _distance_to(e, grid.texts[i][0])
        if d <= best_d:
            best, best_d = grid.texts[i][1], d
    if best is not None and not any(best in b for b in bits):
        bits.append(f'near "{best}"')
    return (" " + "; ".join(bits)) if bits else ""


def _cad_geometry(ir, pt, scale, texts):
    """``(rows, n_listed, n_all)``: one line per circle, arc, polyline, line
    and hatch, in drawing units, with the text it sits by."""
    heights = sorted(float(getattr(e, "height", 0) or 0) for e in ir.entities
                     if getattr(e, "KIND", "") == "text"
                     and (getattr(e, "height", 0) or 0) > 0)
    box = ir.bbox()
    diag = (((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5
            if box is not None else 0.0)
    radius = max(4 * heights[len(heights) // 2] if heights else 0.0,
                 0.02 * diag)
    grid = _TextGrid(texts, radius) if texts and radius > 0 else None

    def num(v) -> str:
        return _fmt(v * scale)

    def where(e) -> str:
        layer = getattr(e, "layer", None) or "0"
        style = str(getattr(e, "style", None) or "")
        block = (f" [block {style.split('|')[0][len('block:'):]}]"
                 if style.startswith("block:") else "")
        return f" [layer {layer}]{block}"

    def verts(points) -> str:
        shown = " ".join(pt(p) for p in points[:CAD_VERTICES_SHOWN])
        more = len(points) - CAD_VERTICES_SHOWN
        return shown + (f" ... (+{more} more)" if more > 0 else "")

    picked = [e for e in ir.entities
              if getattr(e, "KIND", "") in _CAD_GEOMETRY_KINDS]
    order = {k: i for i, k in enumerate(_CAD_GEOMETRY_KINDS)}

    def key(e):
        b = getattr(e, "bbox", None) or (0.0, 0.0, 0.0, 0.0)
        return (order[e.KIND], -round(b[3], 6), round(b[0], 6))

    picked.sort(key=key)
    rows = []
    for e in picked[:CAD_GEOMETRY_MAX]:
        kind = e.KIND
        if kind == "circle":
            text = (f"CIRCLE centre {pt(e.center)} radius {num(e.radius)}")
        elif kind == "arc":
            text = (f"ARC centre {pt(e.center)} radius {num(e.radius)}, "
                    f"{_fmt(e.start_angle)} to {_fmt(e.end_angle)} deg")
        elif kind == "line":
            text = (f"LINE {pt(e.start)} to {pt(e.end)}, length "
                    f"{num(e.length())}")
        elif kind == "polyline":
            vs = list(e.vertices or [])
            text = (f"POLYLINE {'closed' if e.closed else 'open'}, "
                    f"{len(vs)} vertices: {verts(vs)}")
        else:
            vs = list(getattr(e, "boundary", None) or [])
            area = e.area() * scale * scale
            text = (f"HATCH area {_fmt(area)}, boundary of {len(vs)} "
                    f"vertices: {verts(vs)}")
        rows.append(text + where(e) + _near_text(e, grid))
    return rows, len(rows), len(picked)


def _read_cad(path: str, name: str) -> Dict[str, Any]:
    """A DXF as its text lines (every TEXT/MTEXT/ATTRIB, leader and
    dimension, in reading order, in DRAWING units) and a summary."""
    from planlens.dxf.units import UNIT_FACTORS
    from planlens.ir.ingest import from_dxf
    code = _dxf_units_in(path)
    unit = _INSUNITS_NAMES.get(code) if code is not None else None
    factor = UNIT_FACTORS.get(unit or "", None)
    # planlens converts to metres by the header's units; reading in the
    # drawing's own units is what the user's CAD program shows, so the
    # conversion is undone (a unit planlens does not convert stays as read).
    ir = from_dxf(filepath=path, units=(unit if factor else "m"))
    scale = 1.0 / factor if factor else 1.0

    def pt(p) -> str:
        return f"({_fmt(p[0] * scale)}, {_fmt(p[1] * scale)})"

    rows = []
    texts = []         # (position, text, middle): what geometry sits by
    for e in ir.entities:
        kind = getattr(e, "KIND", "")
        layer = getattr(e, "layer", None) or "0"
        if kind == "text":
            text = " ".join(str(getattr(e, "content", "") or "").split())
            if text:
                pos = getattr(e, "position", (0.0, 0.0))
                b = getattr(e, "bbox", None)
                mid = (((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b
                       else tuple(pos))
                texts.append((pos, text[:60], mid))
                rows.append((pos, f"TEXT \"{text}\" at {pt(pos)} "
                                  f"[layer {layer}]"))
        elif kind == "leader":
            verts = list(getattr(e, "vertices", None) or [])
            tip = verts[0] if verts else (0.0, 0.0)
            text = " ".join(str(getattr(e, "text", "") or "").split())
            rows.append((tip, f"LEADER \"{text}\" pointing at {pt(tip)} "
                              f"[layer {layer}]"))
        elif kind == "dimension":
            mid = getattr(e, "text_midpoint", None) or (0.0, 0.0)
            text = " ".join(str(getattr(e, "text", "") or "").split())
            meas = getattr(e, "measurement", None)
            val = (f" = {_fmt(meas * scale)}" if isinstance(meas, (int, float))
                   and meas else "")
            rows.append((mid, f"DIMENSION \"{text or '<>'}\"{val} at "
                              f"{pt(mid)} [layer {layer}]"))
    # Reading order on a plan: top to bottom, then left to right.
    rows.sort(key=lambda r: (-round(r[0][1] * scale, 3),
                             round(r[0][0] * scale, 3)))
    geometry, n_listed, n_geometry = _cad_geometry(ir, pt, scale, texts)
    layers = sorted(ir.counts_by_layer().items(), key=lambda kv: -kv[1])
    by_layer: Dict[str, Dict[str, int]] = {}
    for e in ir.entities:
        d = by_layer.setdefault(getattr(e, "layer", None) or "0", {})
        k = getattr(e, "KIND", "") or "other"
        d[k] = d.get(k, 0) + 1
    box = ir.bbox()
    header: Dict[str, Any] = {
        "kind": "dxf drawing",
        "read_as": "cad data (exact text and entities, drawing units)",
        "drawing_units": (unit or "not stated in the file"),
        "n_entities": len(ir.entities),
        "entities_by_type": ir.counts_by_type(),
        "layers": {k: v for k, v in layers[:40]},
        "entities_by_layer": {k: by_layer.get(k, {}) for k, _ in layers[:40]},
        "n_text_lines": len(rows),
        "n_geometry_lines": n_listed,
    }
    if n_geometry > n_listed:
        header["geometry_not_listed"] = n_geometry - n_listed
    if len(layers) > 40:
        header["layers_not_listed"] = len(layers) - 40
    if box is not None:
        header["extent"] = [float(_fmt(v * scale)) for v in box]
    if ir.warnings:
        header["warnings"] = [str(w)[:200] for w in ir.warnings[:5]]
    header["note"] = (
        f"'{name}' is a DXF drawing, read as CAD data, in drawing units: "
        "every piece of text, leader and dimension with its position, then "
        "the geometry - each CIRCLE (centre, radius), ARC, POLYLINE "
        "(vertices, closed or open), LINE (ends) and HATCH - with the text "
        "it sits by ('near', 'encloses'); and the entity counts per layer. "
        f"read_document(handle='{name}', start_line=N) pages through it all; "
        f"search_document(handle='{name}', pattern=...) finds a line (e.g. "
        "'CIRCLE', or a label). To add a note or comment to the drawing, "
        "use annotate_dxf (a copy, written by a CAD library) - never retype "
        "the DXF with save_file. There are no pages here: to LOOK at the "
        "sheet, or for a marked-up PDF, ask the user for a PDF plot of it.")
    lines = [r[1] for r in rows]
    if geometry:
        lines += ["GEOMETRY (drawing units; 'near' = the closest text):"] \
            + geometry
    return {"lines": lines, "header": header}


def _read_office(path: str, name: str) -> Dict[str, Any]:
    """A Word or Excel file as Markdown lines (``funhouse_agent.office_text``,
    the reader read_text_file uses) and what it is."""
    from funhouse_agent import office_text
    image_dir = image_ref = None
    try:
        from funhouse_agent.vision_tools import SCRATCH_DIR, scratch_dir
        image_dir = scratch_dir(create=False)
        image_ref = SCRATCH_DIR if image_dir else None
    except Exception:  # noqa: BLE001 - pictures are named, not saved
        image_dir = image_ref = None
    try:
        markdown, fields = office_text.read_office(
            path, image_dir=image_dir, image_ref=image_ref)
    except TypeError:            # a reader without the picture arguments
        markdown = office_text.office_to_markdown(path)
        fields = {}
    ext = os.path.splitext(name)[1].lower()
    header: Dict[str, Any] = {
        "kind": ("word document" if ext in (".docx", ".doc")
                 else "excel workbook"),
        "read_as": "markdown"}
    note = fields.pop("note", None) if isinstance(fields, dict) else None
    if isinstance(fields, dict):
        header.update({k: v for k, v in fields.items()
                       if k not in ("read_as",)})
    tail = (f" There are no pages here, so the page, markup and vision tools "
            f"do not apply: read_document(handle='{name}', start_line=N) "
            f"pages through it, search_document(handle='{name}', "
            f"pattern=...) finds lines in it, and read_text_file reads it "
            f"the same way.")
    header["note"] = (note or f"'{name}' read as Markdown.") + tail
    return {"lines": str(markdown or "").splitlines(), "header": header}


def _side_read(source: str) -> Dict[str, Any]:
    """The text of a Word/Excel/DXF ``source`` (cached by path, mtime and
    size). Raises a ``ToolError`` / ``OfficeReadError`` with words fit to
    show."""
    path, temporary = _side_path(source)
    name = os.path.basename(str(source).replace("\\", "/")) or str(source)
    try:
        st = os.stat(path)
        key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns,
               st.st_size)
        if not temporary:
            with _SIDE_LOCK:
                hit = _SIDE_CACHE.get(key)
                if hit is not None:
                    _SIDE_CACHE.move_to_end(key)
                    return hit
        kind = side_kind(name) or side_kind(path)
        read = _read_cad(path, name) if kind == "cad" \
            else _read_office(path, name)
        read["name"] = name
        if not temporary:
            with _SIDE_LOCK:
                _SIDE_CACHE[key] = read
                while len(_SIDE_CACHE) > _SIDE_CACHE_MAX:
                    _SIDE_CACHE.popitem(last=False)
        return read
    finally:
        if temporary:
            try:
                os.remove(path)
            except OSError:
                pass


def _lines_within(lines, start: int, budget: int):
    """``(text, next_start)``: lines from ``start`` whose JSON fits
    ``budget``; ``next_start`` is ``None`` at the end."""
    used, out = 0, []
    i = start
    while i < len(lines):
        size = len(json.dumps(lines[i], ensure_ascii=False)) + 2
        if out and used + size > budget:
            return "\n".join(out), i
        if not out and size > budget:      # one line longer than the room
            out.append(lines[i][:max(200, budget - 80)] + " ...[line cut]")
            return "\n".join(out), (i + 1 if i + 1 < len(lines) else None)
        out.append(lines[i])
        used += size
        i += 1
    return "\n".join(out), None


def _side_search(lines, pattern: str, regex: bool, case_sensitive: bool,
                 fuzzy: bool, min_score: int, max_hits: int):
    hits = []
    if regex:
        rx = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        test = lambda s: rx.search(s) is not None  # noqa: E731
    else:
        needle = pattern if case_sensitive else pattern.lower()
        test = lambda s: needle in (s if case_sensitive  # noqa: E731
                                    else s.lower())
    scorer = None
    if fuzzy and not regex:
        try:
            from rapidfuzz import fuzz
            scorer = fuzz.partial_ratio
        except ImportError:
            scorer = None
    for i, line in enumerate(lines):
        if scorer is not None:
            score = scorer(pattern.lower(), line.lower())
            if score >= min_score:
                hits.append({"line": i, "text": line[:400],
                             "score": round(float(score), 1)})
        elif test(line):
            hits.append({"line": i, "text": line[:400]})
    if scorer is not None:
        hits.sort(key=lambda h: -h["score"])
    return hits[:max_hits], len(hits)


def _side_document(name: str, arguments: Dict[str, Any],
                   max_chars: Optional[int]) -> Optional[str]:
    """The result of a document tool called on a Word, Excel or DXF file,
    or ``None`` when the call is about a PDF or an image (planlens' own)."""
    source = _side_source(name, arguments)
    if source is None:
        return None
    shown = os.path.basename(source.replace("\\", "/")) or source
    kind = side_kind(source)
    if name not in SIDE_READ_TOOLS:
        what = ("a DXF drawing, read as CAD data" if kind == "cad"
                else "a Word or Excel file, read as Markdown")
        hint = (f"read_document(handle='{shown}') reads it and "
                f"search_document(handle='{shown}', pattern=...) searches "
                "it.")
        hint += (" To add a note to the drawing, use annotate_dxf (it "
                 "writes a copy); to look at the sheet or for a marked-up "
                 "PDF, ask the user for a PDF plot of it." if kind == "cad"
                 else
                 " Comments on it go in a document you write (write_docx, "
                 "write_xlsx).")
        return json.dumps({
            "error": f"'{shown}' is {what}: it has no pages here, so "
                     f"{name} does not apply to it.",
            "hint": hint}, ensure_ascii=False)
    try:
        read = _side_read(source)
    except Exception as exc:  # noqa: BLE001 - ToolError / OfficeReadError
        from funhouse_agent.error_text import scrub_paths
        out = {"error": scrub_paths(str(exc)) or type(exc).__name__}
        hint = getattr(exc, "hint", None)
        if hint:
            out["hint"] = scrub_paths(hint)
        return json.dumps(out, ensure_ascii=False)
    lines = read["lines"]
    budget = max(1000, int(max_chars or 12000) - 600)
    if name == "search_document":
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            return json.dumps({"error": "pattern is empty"})
        try:
            hits, n = _side_search(
                lines, pattern, bool(arguments.get("regex")),
                bool(arguments.get("case_sensitive")),
                bool(arguments.get("fuzzy")),
                int(arguments.get("min_score") or 80),
                max(1, min(int(arguments.get("max_hits") or 100), 500)))
        except re.error as exc:
            return json.dumps({"error": f"invalid regular expression: {exc}",
                               "hint": "pass regex=false for a literal "
                                       "search"})
        out: Dict[str, Any] = {"handle": shown, "pattern": pattern,
                               "n_hits": n, "hits": []}
        room = budget - len(json.dumps(out))
        for h in hits:
            size = len(json.dumps(h, ensure_ascii=False)) + 1
            if size > room:
                out["hits_omitted_for_size"] = len(hits) - len(out["hits"])
                break
            out["hits"].append(h)
            room -= size
        out["note"] = ("hits are LINES of the file's text (line = the "
                       "start_line read_document takes); a Word/Excel/DXF "
                       "file has no pages")
        return json.dumps(out, ensure_ascii=False)
    try:
        start = max(0, int(arguments.get("start_line") or 0))
    except (TypeError, ValueError):
        start = 0
    if lines and start >= len(lines):
        return json.dumps({"error": f"start_line {start} is past the end "
                                    f"({len(lines)} lines)"})
    out = {"handle": shown, "source": source, "name": read.get("name", shown)}
    if name == "open_document" or start == 0:
        out.update({k: v for k, v in read["header"].items() if k != "note"})
    out["n_lines"] = len(lines)
    note = read["header"].get("note") if name == "open_document" else None
    reserve = len(json.dumps(out, ensure_ascii=False)) + len(
        json.dumps(note or "", ensure_ascii=False)) + 120
    text, nxt = _lines_within(lines, start, max(400, budget - reserve))
    out["lines_returned"] = (f"{start}-{start + max(0, text.count(chr(10)))}"
                             if text else "none")
    out["text"] = text
    if nxt is not None:
        out["next"] = {"start_line": nxt}
        out["more"] = (f"call read_document(handle='{shown}', "
                       f"start_line={nxt}) for the rest")
    if note:
        out["note"] = note
    return json.dumps(out, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Writing a marked-up copy: never over a file in use, one mark at a time
# ---------------------------------------------------------------------------
# Live smoke wave 2b, C4 (F31): to change one comment the model rebuilt all 19
# marks with append=false onto review_set_marked.pdf, which this
# conversation's toolkit held OPEN (the model had opened the marked copy to
# read its marks). MuPDF saved straight onto it; Windows refused to remove a
# file in use and the turn died. On Linux the save succeeds and the open
# handle silently keeps serving the old marks; a save that fails midway
# leaves the delivered copy broken. Now the copy is written to a temporary
# file beside it and swapped in with os.replace, after every handle this
# conversation holds on it is closed (and reopens, by name, on next use);
# and one mark can be removed or replaced by its id without rebuilding the
# rest.

#: How a markup id that planlens' document_markups lists looks ("p0.m1").
_MARKUP_ID = re.compile(r"^p(\d+)\.m(\d+)$")


def forget_path(path: str) -> int:
    """Close every document THIS conversation's toolkit holds open from the
    file ``path`` (so it can be replaced, and is re-read when next used: its
    handle still resolves, by the name it was opened under). Returns how
    many were closed. planlens has no public call for one document; this
    reaches into the toolkit's own entries under its lock."""
    space = _space(create=False)
    kit = space.kit if space is not None else None
    if kit is None or not path:
        return 0
    _remember(space)
    target = os.path.normcase(os.path.abspath(str(path)))
    entries = getattr(kit, "_entries", None)
    if not isinstance(entries, dict):
        return 0
    guard = getattr(kit, "_guard", None)
    closed = []
    if guard is not None:
        guard.acquire()
    try:
        for handle, entry in list(entries.items()):
            p = getattr(entry, "path", None)
            if p and os.path.normcase(os.path.abspath(str(p))) == target:
                entries.pop(handle, None)
                closed.append(entry)
        by_key = getattr(kit, "_by_key", None)
        if closed and isinstance(by_key, dict):
            gone = {e.handle for e in closed}
            for k in [k for k, h in by_key.items() if h in gone]:
                by_key.pop(k, None)
    finally:
        if guard is not None:
            guard.release()
    for entry in closed:
        try:
            with entry.lock:
                entry.doc.close()
        except Exception:  # noqa: BLE001 - closing is housekeeping
            pass
    return len(closed)


def _temp_beside(final: str) -> str:
    """A temporary file name for writing ``final``: in the conversation's
    scratch folder when ``final`` is in the working folder (never a
    download card, never mirrored), else beside it as a dot-file.

    The name is SHORT -- ``<8 hex>.part.pdf``, no stem (live smoke wave 2c,
    D5): ``.scratch\\<the final name>.<8 hex>.part.pdf`` added 27
    characters to an already long name and pushed the path past Windows'
    260-character limit (F46 at 265, F16), where MuPDF could not open it."""
    import uuid
    tag = uuid.uuid4().hex[:8]
    folder = os.path.dirname(os.path.abspath(final))
    try:
        from funhouse_agent._fileio import host_output_dir
        from funhouse_agent.vision_tools import SCRATCH_DIR
        host = host_output_dir()
        if host and os.path.normcase(os.path.abspath(host)) == \
                os.path.normcase(folder):
            folder = os.path.join(folder, SCRATCH_DIR)
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, f"{tag}.part.pdf")
    except Exception:  # noqa: BLE001 - beside it, then
        pass
    return os.path.join(folder, f".{tag}.part.pdf")


#: Longest full path Windows opens without long-path support.
WINDOWS_MAX_PATH = 260


class MarkedCopyNotWritten(OSError):
    """The marked copy could not be written on the server: a file-system
    failure, not a problem with the marks, so the same call fails again."""


#: The hint a write failure carries (D5: F16 retried the same call five
#: times on the generic "try it again once").
WRITE_FAILED_HINT = (
    "This is a file problem on the server, not a problem with the marks: "
    "calling annotate_document again the same way will fail the same way, "
    "and check=false will not help. If the cause says the path is too long, "
    "a shorter output_path may work, once; otherwise tell the user plainly "
    "that the marked copy could not be saved, with the cause, and stop "
    "retrying.")


def _write_failure(path: str, cause: Any = None) -> MarkedCopyNotWritten:
    """A :class:`MarkedCopyNotWritten` naming what failed and, where it can
    be told, why -- a path too long for Windows is said as such, since
    MuPDF's own message is cut off mid-path and keeps no cause."""
    from funhouse_agent.error_text import scrub_paths
    why = ""
    if isinstance(cause, BaseException):
        why = " ".join(scrub_paths(f"{type(cause).__name__}: {cause}").split())
    elif cause:
        why = " ".join(scrub_paths(str(cause)).split())
    if len(why) > 240:
        why = why[:240] + " …"
    n = len(os.path.abspath(path))
    if os.name == "nt" and n >= WINDOWS_MAX_PATH - 1:
        why = (f"the server path is {n} characters long, over Windows' "
               f"{WINDOWS_MAX_PATH}-character limit"
               + (f" ({why})" if why else ""))
    return MarkedCopyNotWritten(
        "could not write the marked copy on the server"
        + (f": {why}" if why else ""))


def _probe_writable(tmp: str) -> None:
    """Create and remove ``tmp`` before planlens writes there: a path the
    server cannot write fails HERE, with its cause, instead of deep inside
    MuPDF with a message cut off mid-path. Raises
    :class:`MarkedCopyNotWritten`."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
        with open(tmp, "wb"):
            pass
        os.remove(tmp)
    except OSError as exc:
        raise _write_failure(tmp, exc) from exc


def _is_write_failure(error_text: str, tmp: str) -> bool:
    """Whether planlens' error is the temporary copy failing to save (a
    MuPDF system error, or one naming the temporary file) rather than a
    problem with the marks."""
    text = str(error_text or "")
    return ("FzErrorSystem" in text or os.path.basename(tmp) in text
            or "cannot open file" in text or "cannot remove file" in text)


def _replace_into(tmp: str, final: str):
    """``os.replace(tmp, final)`` once nothing here holds ``final`` open;
    ``(path written, note or None)``. When ``final`` is still in use
    elsewhere (open in the user's viewer on a Windows host) the copy is
    saved under the next free name instead, and the note says so."""
    import time as _t
    last = None
    for attempt in range(3):
        forget_path(final)
        try:
            os.replace(tmp, final)
            return final, None
        except PermissionError as exc:
            last = exc
            _t.sleep(0.2 * (attempt + 1))
    stem, ext = os.path.splitext(final)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    alt = f"{stem}_{n}{ext}"
    os.replace(tmp, alt)
    return alt, (f"'{os.path.basename(final)}' is in use elsewhere "
                 f"({type(last).__name__ if last else 'locked'}), so the "
                 f"marked copy was saved as '{os.path.basename(alt)}'; tell "
                 "the user which file to open")


def _ours(author: Optional[str], signer: str) -> bool:
    """Whether a markup was written by this app (signed by it), so it may be
    removed; another reviewer's markup in the copy is left alone."""
    a = str(author or "")
    return bool(a) and (a == signer or "GeotechStaffEngineer" in a
                        or "(AI draft)" in a)


def _is_label(doc, xref) -> bool:
    """Whether an annotation is a label grouped with another (``/RT
    /Group``), as planlens writes a mark's visible label."""
    try:
        kind, val = doc.xref_get_key(int(xref), "RT")
    except Exception:  # noqa: BLE001
        return False
    return kind == "name" and str(val).lstrip("/") == "Group"


def remove_markups(pdf: bytes, remove, signer: str):
    """``(new_pdf_bytes, removed, not_removed)``: the marks named in
    ``remove`` taken out of a marked copy, each with its label and any
    reply to it. An entry is a markup id as document_markups lists it
    (``"p0.m1"``) or words from its comment, which must name exactly one
    mark this app wrote. Only this app's marks are removed."""
    import fitz
    from planlens.document.annotations import extract_annotations

    doc = fitz.open(stream=bytes(pdf), filetype="pdf")
    try:
        per_page = {i: extract_annotations(doc[i], i)[0]
                    for i in range(doc.page_count)}
        everything = [m for ms in per_page.values() for m in ms]
        chosen, removed, refused = {}, [], []
        for raw in ([remove] if isinstance(remove, (str, dict))
                    else list(remove or [])):
            want = (raw.get("id") or raw.get("text") or "") \
                if isinstance(raw, dict) else str(raw or "")
            want = want.strip()
            if not want:
                continue
            m_id = _MARKUP_ID.match(want)
            if m_id:
                hits = [m for m in per_page.get(int(m_id.group(1)), [])
                        if m.id == want]
                if not hits:
                    refused.append({"remove": want, "reason": (
                        "no markup has this id in the copy now (ids are "
                        "positions on a page and change after a removal: "
                        "list them again with document_markups)")})
                    continue
            else:
                # A mark's label rides on it (removed with it), so only the
                # marks themselves are matched by their words.
                low = want.lower()
                hits = [m for m in everything if _ours(m.author, signer)
                        and not _is_label(doc, m.xref)
                        and low in (m.text or "").lower()]
                if len(hits) != 1:
                    refused.append({"remove": want, "reason": (
                        "no mark this app wrote says that" if not hits else
                        f"{len(hits)} marks say that "
                        f"({', '.join(m.id for m in hits[:8])}): give the "
                        "id of the one meant")})
                    continue
            m = hits[0]
            if not _ours(m.author, signer):
                refused.append({"remove": want, "reason": (
                    f"{m.id} is {m.author or 'another reviewer'}'s markup, "
                    "not one this app wrote; it is left as it is")})
                continue
            chosen[(m.page, m.xref)] = m
        for (page_no, xref), m in chosen.items():
            page = doc[page_no]
            doomed = {xref}
            grew = True
            while grew:                 # its label, and replies to either
                grew = False
                for a in page.annots() or []:
                    if a.xref in doomed:
                        continue
                    kind, val = doc.xref_get_key(a.xref, "IRT")
                    if kind == "xref" and val:
                        try:
                            parent = int(str(val).split()[0])
                        except ValueError:
                            continue
                        if parent in doomed:
                            doomed.add(a.xref)
                            grew = True
            for x in sorted(doomed, reverse=True):
                try:
                    page.delete_annot(page.load_annot(x))
                except Exception:  # noqa: BLE001 - already gone with a parent
                    pass
            removed.append({"id": m.id, "page": m.page, "pdf_page": m.page + 1,
                            "kind": m.kind, "says": (m.text or "")[:120],
                            "with_attached": len(doomed) - 1})
        return doc.tobytes(garbage=1), removed, refused
    finally:
        doc.close()


def _handle_path(handle: str) -> Optional[str]:
    """The file a document handle (or a file name given as one) stands for
    in this conversation, when it is a file; ``None`` otherwise."""
    try:
        if looks_like_handle(handle):
            entry = document_entry(handle)
            if getattr(entry, "path", None):
                return str(entry.path)
            handle = handle_source(handle) or ""
        if not handle:
            return None
        from funhouse_agent.vision_tools import find_readable_file
        return find_readable_file(handle)
    except Exception:  # noqa: BLE001 - not a file, then
        return None


def write_marked_copy(handle: str, markups: Optional[list], output_path: str,
                      append: bool, author: str, call, remove=None
                      ) -> Dict[str, Any]:
    """Write ``annotate_document``'s marked copy safely (see the section
    comment): to a temporary file, then swapped in. ``call(args) -> str`` runs
    planlens' annotate_document (the host's dispatch). ``remove`` takes marks
    out of an existing marked copy first, so ``remove`` + ``markups``
    replaces one mark and keeps the rest. Returns the result dict, its
    ``output_path`` the final file's absolute path."""
    try:
        return _write_marked_copy(handle, markups, output_path, append,
                                  author, call, remove)
    except MarkedCopyNotWritten as exc:
        return {"error": str(exc), "hint": WRITE_FAILED_HINT}


def markups_with_pages(markups) -> tuple:
    """``(markups, None)`` with each markup's ``pdf_page`` (1-based, the
    number a viewer shows) turned into planlens' 0-based ``page``, or
    ``([], error)``: a markup whose ``page`` and ``pdf_page`` name
    different pages is refused, never guessed (live smoke wave 2c)."""
    from funhouse_agent.page_numbers import resolve_page
    out = []
    for i, m in enumerate(list(markups or [])):
        if isinstance(m, dict) and "pdf_page" in m:
            m = dict(m)
            page, problem = resolve_page(m.get("page"), m.pop("pdf_page"),
                                         default=None)
            if problem is not None:
                return [], {"error": f"markup {i}: {problem['error']}",
                            "hint": problem.get("hint", "")}
            m["page"] = page
        out.append(m)
    return out, None


def _nothing_written(result: Dict[str, Any], final: str, existed: bool,
                     had_marks: bool, removed, refused) -> Dict[str, Any]:
    """``result`` for a call that placed no mark and removed none: no
    ``output_path`` (nothing to download), and a note saying nothing was
    written, why, and which copy -- if any -- is still there unchanged."""
    out = {k: v for k, v in result.items()
           if k not in ("output_path", "appended_to_existing", "note")}
    name = os.path.basename(final)
    why = ("every mark was skipped: read each skipped row's reason"
           if had_marks else "no mark named in remove could be removed")
    kept = (f"'{name}' is unchanged, as it was before this call" if existed
            else f"'{name}' was not created")
    out["nothing_written"] = True
    out["note"] = (f"Nothing was written: {why}. No marked copy was saved; "
                   f"{kept}. Fix what the reasons say and call again; do not "
                   "offer the user a marked copy from this call.")
    if removed is not None:
        out["removed"] = removed
    if refused:
        out["not_removed"] = refused
    return out


def _write_marked_copy(handle, markups, output_path, append, author, call,
                       remove) -> Dict[str, Any]:
    """:func:`write_marked_copy`; a file-system failure raises
    :class:`MarkedCopyNotWritten` (D5)."""
    import shutil
    marks, problem = markups_with_pages(markups)
    if problem is not None:
        return problem
    final = markup_output_path(output_path, handle)
    existed = os.path.isfile(final)
    tmp = _temp_beside(final)
    try:
        removed = refused = None
        if remove:
            if not existed:
                return {"error": (
                    f"nothing to remove from: '{os.path.basename(final)}' "
                    "does not exist yet"),
                    "hint": ("remove takes marks out of a marked copy "
                             "written earlier; name it in output_path")}
            with open(final, "rb") as fh:
                data = fh.read()
            data, removed, refused = remove_markups(data, remove, author)
            try:
                with open(tmp, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                raise _write_failure(tmp, exc) from exc
        elif append and existed:
            try:
                shutil.copyfile(final, tmp)
            except OSError as exc:
                raise _write_failure(tmp, exc) from exc
        else:
            _probe_writable(tmp)
            src = _handle_path(handle)
            if src and os.path.normcase(os.path.abspath(src)) == \
                    os.path.normcase(os.path.abspath(final)):
                return {"error": (
                    "this handle is the marked copy itself, and append=false "
                    "would write every mark again on top of the ones it has"),
                    "hint": ("to rebuild the copy, pass the ORIGINAL "
                             "document's handle with append=false; to change "
                             "or delete one mark, pass remove=[its id] (and "
                             "the new mark in markups) with output_path = "
                             "this copy")}
        if marks:
            raw = call({"handle": handle, "output_path": tmp,
                        "markups": marks, "author": author,
                        # onto the temporary copy when there is one
                        "append": bool(remove or append)})
            try:
                result = json.loads(raw)
            except (TypeError, ValueError):
                return {"error": str(raw)[:500]}
            if isinstance(result, dict) and "error" in result \
                    and _is_write_failure(result["error"], tmp):
                # MuPDF could not save the temporary copy: said as the
                # server-side failure it is, not "try it again once" (D5).
                raise _write_failure(tmp, result["error"])
            if not isinstance(result, dict) or "error" in result:
                return result if isinstance(result, dict) else \
                    {"error": str(result)[:500]}
            from funhouse_agent.page_numbers import range_hint
            for row in result.get("skipped") or ():
                hint = range_hint(str(row.get("reason") or "")) \
                    if isinstance(row, dict) else None
                if hint:
                    row["hint"] = hint
        elif remove:
            result = {"handle": handle, "author": author, "n_written": 0,
                      "n_skipped": 0}
        else:
            return {"error": "markups is empty: nothing would be written",
                    "hint": ("each markup is {kind, page, comment} plus ONE "
                             "anchor; to take marks out, pass remove")}
        if not int(result.get("n_written") or 0) and not removed:
            # Every mark skipped and nothing taken out: no file is written,
            # so no "marked" copy without marks becomes a download card
            # (live smoke wave 3, F3: F04's first call saved a 0-mark copy).
            return _nothing_written(result, final, existed,
                                    bool(marks), removed, refused)
        try:
            written, moved = _replace_into(tmp, final)
        except OSError as exc:
            raise _write_failure(final, exc) from exc
        result["output_path"] = written
        result["appended_to_existing"] = bool(existed and (append or remove))
        if moved:
            result["saved_elsewhere"] = moved
        if removed is not None:
            result["removed"] = removed
            if refused:
                result["not_removed"] = refused
            result["ids_note"] = (
                "markup ids are positions on a page: after a removal the "
                "ids on that page change; list them with document_markups "
                "before removing another by id")
        return result
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


#: The planlens tools the report ingest is built on: the page roles and work
#: items it walks, and the log grid its log reader reads rows off. Both
#: arrived in planlens 0.5; on anything older the ingest cannot run at all,
#: so the app's tool hides itself rather than failing in front of the user.
REPORT_INGEST_TOOLS = ("document_roles", "log_grid")


def report_ingest_supported() -> bool:
    """Whether the installed planlens can serve the whole-report ingest."""
    return available() and all(has_tool(name) for name in REPORT_INGEST_TOOLS)


def open_document_entry(source: str,
                        attachments: Optional[Dict[str, bytes]] = None):
    """The toolkit's open entry for ``source`` (an attachment key or a path),
    opening it if it is not open yet. Raises planlens' ``ToolError`` when the
    source resolves to nothing. The coverage ledger
    (:mod:`funhouse_agent.coverage`) uses it to know which document a vision
    tool's ``attachment_key`` points at; the entry carries ``handle``,
    ``name``, ``doc`` (the planlens Document) and ``lock``. A handle this
    conversation opened is accepted too (A7)."""
    token = _ATTACHMENTS.set(dict(attachments or {}))
    try:
        if looks_like_handle(source):
            entry = document_entry(source)
            if entry is not None:
                return entry
            source = handle_source(source) or source
        return _toolkit().open(source)
    finally:
        _ATTACHMENTS.reset(token)


def document_entry(handle: str):
    """The open entry behind a document ``handle`` THIS conversation opened,
    or ``None``. A handle whose document was closed since (the
    conversation's open-document limit, or an idle conversation's toolkit)
    is reopened from the source it came from, where that still resolves."""
    if not handle or not available():
        return None
    kit = _toolkit()
    try:
        return kit._entry(handle)
    except Exception:            # an unknown or closed handle
        pass
    source = handle_source(handle)
    if not source:
        return None
    try:
        entry = kit.open(source)
    except Exception:            # noqa: BLE001 - its source is gone
        return None
    return entry if entry.handle == handle else None


def resolve_document_source(source: str,
                            attachments: Optional[Dict[str, bytes]] = None):
    """An attachment key or a path, resolved the way the document tools do.

    Returns the attachment's BYTES or a real path, which is what planlens'
    ``open_document`` takes either of. Raises planlens' ``ToolError`` when it
    is neither, with the same wording the document tools give.
    """
    token = _ATTACHMENTS.set(dict(attachments or {}))
    try:
        return _resolve(source)
    finally:
        _ATTACHMENTS.reset(token)
