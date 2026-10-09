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
  call says otherwise.

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
    are added only while the result stays under it)."""
    if not available():
        return json.dumps({
            "error": "document tools need planlens with planlens.tools "
                     "(0.3 or later)",
            "hint": "pip install -U planlens"})
    # A shallow snapshot: the host mutates its attachments dict between turns.
    token = _ATTACHMENTS.set(dict(attachments or {}))
    space = _space(hold=True)
    opened_note = None
    try:
        arguments = dict(arguments or {})
        refused, opened_note = _as_handle_or_source(space, name, arguments)
        if refused is not None:
            return refused
        out = space.kit.call_json(name, arguments, max_chars=max_chars)
    finally:
        _release(space)
        _ATTACHMENTS.reset(token)
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


def _viewer_pages(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k] = _viewer_pages(v)
            if (k == "page" and isinstance(v, int) and not isinstance(v, bool)
                    and "pdf_page" not in obj):
                out["pdf_page"] = v + 1
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
    ``"page": n`` gains ``"pdf_page": n + 1`` and each ``=== page n`` header
    in read text gains ``[pdf_page n+1]`` — the number the prompt says to cite.
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
