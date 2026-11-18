"""Pages an answer or a written file cites, checked against the pages this
conversation actually opened (BACKLOG EV-18).

WHY. In the 2026-10-14 field session (``module_work/field_feedback/
2026-10-14_report-to-diggs_v5.33.0/TRACE_REVIEW.md`` Part 3, C5 and P3.8) the
answer gave four page lists as "source traceability" for placeholder borings,
and the DIGGS file carried them. They were even splits of a page range, not
the pages anything had read: the logs were on other pages, which a helper's
own calls had named. Nothing compared a cited page with the pages read.

WHAT IS CHECKED, AND WHERE

* ``subsurface.write_diggs`` - every page a record cites (its ``pages`` and
  the ``page`` of any ``prov``) - adds one cross-check entry of kind
  :data:`ENTRY_KIND` per record that cites a page no tool or helper read or
  looked at in this conversation (:func:`record_entries`). The file is
  written all the same: the check never refuses it.
* The ANSWER: after each turn, every page citation in the reply ("PDF page
  24", "pp. 12-14", "page A-3", a table column headed "Pages") is checked
  (:func:`check_answer`). The result goes into the turn's record
  (``activity.jsonl``, event :data:`RECORD_EVENT`) whenever the reply cites
  pages. Behind ``GEOTECH_CITED_PAGES`` (:func:`funhouse_agent.review_flags.
  cited_pages`, OFF by default): ``note`` appends a short note to the answer
  listing the cited pages no tool opened; ``retry`` tells the model once,
  before the answer goes out, and the reply after that note is the answer
  (a note is appended if it still cites pages no tool opened). The
  :class:`TurnPageCheck` object carries both for ``webapp.core.stream_turn``.

WHAT "OPENED" MEANS. The record is the app's own, never the model's account
(:func:`pages_opened`): every tool call of the primary and its helpers in the
conversation's ``activity.jsonl`` (a page read as text by ``read_document`` /
``read_pdf_text`` / ``log_grid``, looked at by ``analyze_pdf_page`` /
``render_region``, or asked about by ``sweep_pages`` in records from before
DR-39), the vision tools' read record (``reads.json``) and kept readings
(``page_readings.json``). A page
reached only by a text tool that found no text layer was not read (there was
nothing to read). A page named only in a search hit (``search_document``,
``find_quantities``...) is reported apart: the model saw a line of it, not
the page. Without the activity log nothing knows the text reads, so nothing
is checked and the result says so.

CITATIONS. A citation that says PDF ("PDF page 24", "PDF pp. 108-119",
"pdf_page 24", "PDF 133-134", "page 24 of the PDF") counts pages as a PDF
viewer does (1-based). A bare "page 12" / "p. 12" may be the number printed on
the page or the viewer's: it counts as opened when either reading is (the
printed number off planlens' page map: ``printed_page``, the PDF page label,
or a page number standing alone in the header or footer band). A printed
label ("page A-3", "p. 5-2", "p. iv") counts when a page printing it was
opened, and is left unchecked when no page prints it. A run of pages cited
exactly as a structure tool named it ("the logs, PDF pages 41-90": a segment,
a run of one page kind, a work item) says where something is, read off the
page map, and is recorded apart (``page_map``). A page cited beside the
name of a published code, manual or work (a designation such as an agency
manual's, or an author and year) is another document's page and is not
checked here. Pages are 0-based inside this module, as in every tool.

WHAT DOES NOT COUNT AS A PAGE NO TOOL OPENED (repairs of 2026-11-18, from
an offline pass over saved runs; none of them adds a firing). Only
``never_opened`` makes the note; each of these is recorded under its own
list instead:

* ``page_map``, also for a page whose statement (the sentence part, list
  item or table row that cites it) says only what a structure result
  (``open_document``, ``document_page_map``, ``document_structure``...)
  said of that page: :func:`_described` gives the rule ("PDF p. 4 is
  blank", "PDF pp. 7-11: Appendix A, boring logs", "PDF page 8 duplicates
  page 3"). The row keeps the words that matched (``matched``).
* ``named_unread``: a page the answer itself names as not opened, not
  needed, or offered for a look ("PDF pages 10 and 14 were not viewed",
  "Not read: PDF page 6", "I can look at the scan (PDF p. 5)").
* ``unverified``, also for a citation said to be printed ("printed page
  49", "chapter page 12-16"), or written as a chapter-page number under a
  one-page lead ("p. 1-3", a column headed "Page"), that no page of the
  documents read is known to print: in a document whose page map knows no
  printed numbers, "1-3" is one printed page, not PDF pages 1 to 3.
* nothing at all for a printed number given beside the PDF page of the
  same place ("printed page 1-3; PDF page 13", a row with a "Page" and a
  "PDF page" column): the PDF page decides.

A page whose review markups ``document_markups`` listed counts as opened
for a citation (the markup was read whole), and so does a page a reference
reader (:data:`REFERENCE_TOOLS`, the geotech page's) rendered and read,
where the conversation has no document of its own or the citation's line
names that reference (its figure or table number, its example id). A run
of printed labels ("pp. 4-1 to 4-3") is one citation of three pages. These
are read only by the answer check: what ``write_diggs`` is told
(:func:`record_entries`) and the photograph-page check are as before.

PHOTOGRAPH PAGES NOTHING LOOKED AT (BACKLOG DR-48). At turn end the same
record answers a second question: which photograph pages of the attached
documents did no tool or helper look at (``analyze_pdf_page``,
``render_region``, the vision records), and were not on a
``render_page_thumbnails`` contact sheet that a later call then looked at
(:func:`photo_coverage`). "Photograph page" is planlens' own page role
(``photos``, :mod:`planlens.document.roles`: a photograph caption on an
image page), never a file name; a document whose page kinds cannot be told
here (not a PDF, not in the conversation's folder, a planlens without page
roles) is left out and the record says why. The attached documents are the
page's upload list (``attachments.json`` beside the activity log), else the
documents the call log opened. It costs no model call, but one planlens pass
over each attached PDF (cached per file while the process lives), so at turn
end it runs only behind ``GEOTECH_PHOTO_PAGES``
(:func:`aec_agent_core.review_flags.photo_pages`, OFF by default: nothing of
it runs and the turn's record is as before). ``record`` keeps the answer
under :data:`PHOTO_KEY` in the turn's :data:`RECORD_EVENT` event (any error
in it swallowed and recorded); ``retry`` also tells the model once, before
the answer goes out, through the same :class:`TurnPageCheck` ``feedback``
the cited-pages retry uses; pages it was told about are not told again later
in the conversation. :func:`photo_coverage` also reads a saved run's folder
by itself, with no switch.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple)

#: The activity-log event a turn's check is recorded under.
RECORD_EVENT = "cited_pages"
#: The ``write_diggs`` cross-check kind.
ENTRY_KIND = "page_not_read"
#: How the feedback note to the model opens (``GEOTECH_CITED_PAGES=retry``).
NOTE_PREFIX = "[Page check]"
#: The status line shown while the answer is written again.
RETRY_STATUS = ("page check: the answer cites pages no tool opened; it is "
                "being written again")

ACTIVITY_FILE = "activity.jsonl"

#: Tools whose results name the pages they READ or LOOKED AT (the list
#: :func:`funhouse_agent.coverage.pages_from_call` reads).
TEXT_TOOLS = ("read_document", "read_pdf_text")
LOOK_TOOLS = ("analyze_pdf_page", "render_region")
SWEEP_TOOLS = ("sweep_pages",)
GRID_TOOLS = ("log_grid",)
OPENING_TOOLS = TEXT_TOOLS + LOOK_TOOLS + SWEEP_TOOLS + GRID_TOOLS
#: Tools whose results name pages in HITS: a line of the page was seen, the
#: page was not opened.
HIT_TOOLS = ("search_document", "find_quantities", "digest_search",
             "document_markups", "find_like")
#: The tool that lists a document's review markups, each with its page:
#: an answer citing a markup's page cites what the markup says, read whole.
MARKUP_TOOLS = ("document_markups",)
#: The geotech page's readers of a reference library page: each renders a
#: page of a reference PDF (a figure, a table, a worked example's printed
#: source) and reads it with vision (BACKLOG EV-18, 2026-10-20 note).
REFERENCE_TOOLS = ("read_reference_figure", "read_reference_table",
                   "view_worked_example_source")

#: How a page was reached (``hows``): read as text, looked at, asked about by
#: a sweep, reached by a text tool and found to have no text layer, named in
#: a search hit.
OPENED_HOWS = frozenset({"text", "look", "sweep"})
TEXT_EMPTY = "text_empty"
HIT = "hit"

#: The planlens page role of a page of photographs (DR-48).
PHOTO_ROLE = "photos"
#: The key of the photograph-page check in the turn's :data:`RECORD_EVENT`.
PHOTO_KEY = "photo_pages"
#: How the photograph-page note to the model opens (``GEOTECH_PHOTO_PAGES``).
PHOTO_NOTE_PREFIX = "[Photo-page check]"
#: The status line the page shows while the model answers that note.
PHOTO_RETRY_STATUS = ("photo-page check: attached documents have photograph "
                      "pages nothing looked at; the answer is held back once")
#: The page's list of the conversation's uploads, beside the activity log.
ATTACHMENTS_FILE = "attachments.json"
#: Tools that write contact sheets of a document's pages (PNG files that a
#: later call looks at).
CONTACT_SHEET_TOOLS = ("render_page_thumbnails",)
#: The tools that look at an image file: only a call of one of these that
#: names a contact sheet counts as seeing the pages on it.
SHEET_LOOK_TOOLS = ("analyze_image",) + LOOK_TOOLS

#: Most citations / entries kept in one result.
MAX_LISTED = 40
#: A cited range longer than this is taken as its two ends only.
MAX_RANGE = 2000
#: A sidecar holding a whole tool result is read up to this size.
MAX_SIDECAR_BYTES = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Where the conversation's record is
# ---------------------------------------------------------------------------

_RECORD_DIR: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "geotech_cited_pages_record_dir", default=None)


def bind_record_dir(path: Optional[str]):
    """Name the folder holding this conversation's ``activity.jsonl`` for the
    calls made in this context (the turn binds it; tools read it). Returns
    the token for :func:`unbind_record_dir`."""
    return _RECORD_DIR.set(os.path.abspath(str(path)) if path else None)


def unbind_record_dir(token) -> None:
    try:
        _RECORD_DIR.reset(token)
    except (ValueError, LookupError, RuntimeError):
        pass  # a token from another context: nothing to undo here


@contextlib.contextmanager
def record_bound(path: Optional[str]):
    """``with record_bound(conv_dir):`` - :func:`bind_record_dir` for a
    block (a no-op for ``None``)."""
    if not path:
        yield
        return
    token = bind_record_dir(path)
    try:
        yield
    finally:
        unbind_record_dir(token)


def _host_folder(folder=None) -> Optional[str]:
    if folder:
        return os.path.abspath(str(folder))
    try:
        from aec_agent_core._fileio import host_output_dir
        got = host_output_dir()
    except Exception:  # noqa: BLE001 - no folder, then
        return None
    return os.path.abspath(str(got)) if got else None


def record_dir(folder=None) -> Optional[str]:
    """The folder holding this conversation's ``activity.jsonl``: the one
    the turn bound, else the one the working
    folder sits in (the web app's ``<conversation>/files``, the review
    suite's ``<run>/files``), else the working folder itself. ``None`` when
    there is none."""
    if folder is None:
        bound = _RECORD_DIR.get()
        if bound and os.path.isdir(bound):
            return bound
    host = _host_folder(folder)
    if not host:
        return None
    candidates = []
    if os.path.basename(host.rstrip("\\/")) == "files":
        candidates.append(os.path.dirname(host.rstrip("\\/")))
    candidates.append(host)
    for c in candidates:
        if os.path.isfile(os.path.join(c, ACTIVITY_FILE)):
            return c
    return None


# ---------------------------------------------------------------------------
# Page lists
# ---------------------------------------------------------------------------

def _pages(spec: Any) -> List[int]:
    from aec_agent_core.coverage import parse_pages
    return parse_pages(spec)


def compact(pages: Iterable[int], offset: int = 0) -> str:
    from aec_agent_core.coverage import compact as _compact
    return _compact(pages, offset)


def _json(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def doc_key(name: Any) -> str:
    """One key per document: its file name, case folded (a handle stays as
    it is until the record says which file it stands for)."""
    text = str(name or "").strip().replace("\\", "/").rstrip("/")
    return os.path.basename(text).lower() if text else ""


# ---------------------------------------------------------------------------
# The record of pages opened
# ---------------------------------------------------------------------------

@dataclass
class PagesOpened:
    """Every page any tool or helper reached in one conversation."""
    #: document key -> 0-based page -> how it was reached (``hows``).
    pages: Dict[str, Dict[int, Set[str]]] = field(default_factory=dict)
    #: document key -> the name a reader knows it by.
    names: Dict[str, str] = field(default_factory=dict)
    #: document key -> what it can be opened again by (attachment keys,
    #: paths, document handles).
    sources: Dict[str, Set[str]] = field(default_factory=dict)
    #: document key -> page count, where a result said.
    n_pages: Dict[str, int] = field(default_factory=dict)
    #: The records read ("activity log", "vision reads", ...).
    records: List[str] = field(default_factory=list)
    #: Whether a record of TEXT reads was found (the activity log); without
    #: one nothing is checked.
    knows_text: bool = False
    #: Page runs ``(first, last)``, 0-based, that a structure tool named
    #: (a segment, a run of one page kind, a work item): an answer naming
    #: exactly one describes where something is, from the page map.
    runs: Set[Tuple[int, int]] = field(default_factory=set)
    #: document key -> 0-based pages that were on a contact sheet a later
    #: call looked at (kept apart: a cited page is not "opened" by it).
    sheet_seen: Dict[str, Set[int]] = field(default_factory=dict)
    #: document key -> 0-based photograph pages the model was already told
    #: nothing looked at (``GEOTECH_PHOTO_PAGES``), earlier in the record.
    photos_told: Dict[str, Set[int]] = field(default_factory=dict)
    #: document key -> 0-based page -> what a structure tool's result said
    #: of the page (its kind, its title or heading, its duplicate flag...),
    #: as text. Only the answer check reads it (:func:`check_answer`).
    facts: Dict[str, Dict[int, List[str]]] = field(default_factory=dict)
    #: document key -> 0-based pages whose review markups a markup listing
    #: returned. Only the answer check reads it.
    markups: Dict[str, Set[int]] = field(default_factory=dict)
    #: reference PDF key -> 0-based page a reference reader rendered and
    #: read -> the printed page labels its result gave. Kept apart from
    #: ``pages``: a reference is not one of the conversation's documents.
    references: Dict[str, Dict[int, Set[str]]] = field(default_factory=dict)
    #: reference PDF key -> what the answer may name it by (its figure or
    #: table number, its example id, its reference, its file name).
    reference_names: Dict[str, Set[str]] = field(default_factory=dict)

    def looked(self, key: str) -> Set[int]:
        """The pages of one document something looked at: a look, or a
        contact sheet that was looked at."""
        return ({p for p, hows in self.pages.get(key, {}).items()
                 if "look" in hows} | set(self.sheet_seen.get(key, ())))

    def add(self, key: str, page: int, how: str, name: str = "",
            source: str = "") -> None:
        if not key or not isinstance(page, int) or isinstance(page, bool) \
                or page < 0:
            return
        self.pages.setdefault(key, {}).setdefault(page, set()).add(how)
        if name and key not in self.names:
            self.names[key] = name
        if source:
            self.sources.setdefault(key, set()).add(str(source))

    def documents(self, name: Any = None) -> List[str]:
        """The keys of the documents read; with ``name``, those it names
        (a file name, with or without its extension, or a part of one)
        when any does, else all of them."""
        keys = sorted(set(self.pages) | set(self.names))
        want = doc_key(name)
        if not want:
            return keys
        stem = os.path.splitext(want)[0]
        hit = [k for k in keys
               if k == want or os.path.splitext(k)[0] == stem
               or (len(stem) >= 4 and (stem in k or os.path.splitext(k)[0]
                                       in stem))]
        return hit or keys

    def hows(self, page: int, docs: Optional[Sequence[str]] = None
             ) -> Set[str]:
        out: Set[str] = set()
        for key in (docs if docs is not None else list(self.pages)):
            out |= self.pages.get(key, {}).get(page, set())
        return out

    def status(self, page: int, docs: Optional[Sequence[str]] = None) -> str:
        """``opened`` / ``text_empty`` (a text tool found no text layer, and
        nothing looked) / ``hit`` (named only in a search hit) / ``never``."""
        hows = self.hows(page, docs)
        if hows & OPENED_HOWS:
            return "opened"
        if TEXT_EMPTY in hows:
            return TEXT_EMPTY
        if HIT in hows:
            return HIT
        return "never"

    def opened(self, docs: Optional[Sequence[str]] = None) -> Set[int]:
        return {p for key in (docs if docs is not None else list(self.pages))
                for p, hows in self.pages.get(key, {}).items()
                if hows & OPENED_HOWS}


# -- the activity log, read incrementally ------------------------------------

class _ActivityState:
    """What one ``activity.jsonl`` says so far: it is append-only, so each
    read starts where the last one stopped."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.size_seen = 0
        self.starts: Dict[str, Tuple[str, Any]] = {}
        #: (ref kind, ref value, page, how) of every page reached.
        self.reached: List[Tuple[str, str, int, str]] = []
        self.handle_names: Dict[str, str] = {}
        self.n_pages: Dict[str, int] = {}
        self.sources: Dict[str, Set[str]] = {}
        self.runs: Set[Tuple[int, int]] = set()
        #: contact-sheet file (base name, case folded) -> (ref kind, ref
        #: value, 0-based pages on it).
        self.sheets: Dict[str, Tuple[str, str, List[int]]] = {}
        #: (ref kind, ref value, page) of every page on a sheet looked at.
        self.sheet_seen: List[Tuple[str, str, int]] = []
        #: document key -> 0-based photograph pages already told.
        self.photos_told: Dict[str, Set[int]] = {}
        #: (ref kind, ref value) -> 0-based page -> structure facts.
        self.facts: Dict[Tuple[str, str], Dict[int, List[str]]] = {}
        #: (ref kind, ref value, page) of every page a markup listing named.
        self.markups: List[Tuple[str, str, int]] = []
        #: reference key -> page -> printed labels, and its names.
        self.references: Dict[str, Dict[int, Set[str]]] = {}
        self.reference_names: Dict[str, Set[str]] = {}


_ACTIVITY:"OrderedDict[str, _ActivityState]" = OrderedDict()
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY_KEEP = 32


def _state_for(path: str) -> _ActivityState:
    key = os.path.normcase(os.path.abspath(path))
    with _ACTIVITY_LOCK:
        st = _ACTIVITY.get(key)
        if st is None:
            st = _ACTIVITY[key] = _ActivityState()
        _ACTIVITY.move_to_end(key)
        while len(_ACTIVITY) > _ACTIVITY_KEEP:
            _ACTIVITY.popitem(last=False)
        return st


def clear_cache() -> None:
    """Forget what the activity logs said, and the photograph pages measured
    (tests; a host that rewrites one)."""
    with _ACTIVITY_LOCK:
        _ACTIVITY.clear()
    with _PHOTO_LOCK:
        _PHOTO_CACHE.clear()


def _args_dict(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    data = _json(args)
    return data if data is not None else {}


def _ref_of(args: dict, data: Optional[dict]) -> Tuple[str, str]:
    handle = str(args.get("handle") or (data or {}).get("handle") or "")
    if handle:
        return "handle", handle
    src = str(args.get("attachment_key") or args.get("source")
              or args.get("path") or (data or {}).get("source") or "")
    return "source", src


def _grid_pages(data: dict) -> List[int]:
    """A ``log_grid`` result that returned any of a log's content (rows,
    layers, samples, the header fields) read its pages, as far as a citation
    is concerned (:func:`funhouse_agent.coverage.pages_from_call`, which
    asks whether a page was read WHOLE, is stricter)."""
    if any(data.get(k) for k in ("rows", "layers", "samples", "fields",
                                 "columns")):
        return _pages(data.get("pages"))
    return []


_PAGES_RETURNED = re.compile(r'"pages_returned"\s*:\s*"([^"]*)"')

#: Tools that describe a document's STRUCTURE: the page runs their results
#: name are where things are, read off the page map, not pages read.
STRUCTURE_TOOLS = ("open_document", "document_structure", "document_roles",
                   "document_page_map", "document_inventory", "digest_pages")


def _runs_of(value: Any) -> List[Tuple[int, int]]:
    """The runs ``(first, last)`` of a page list: ``"0-5,9"``, ``[3, 4,
    5]``, ``"[3, 4]"``."""
    if isinstance(value, str) and not value.strip().startswith("["):
        out = []
        for part in value.split(","):
            got = _pages(part)
            if got:
                out.append((got[0], got[-1]))
        return out
    got = _pages(value if not isinstance(value, str) else _json_list(value))
    out = []
    for p in got:
        if out and p == out[-1][1] + 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


def _json_list(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _named_runs(obj: Any, out: Set[Tuple[int, int]], depth: int = 0) -> None:
    """Every 0-based page run a structure result names (``pages``,
    ``*_pages``, ``pages_by_kind``; never the viewer's ``pdf_*`` twins)."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if key.startswith("pdf_") or key in ("next", "hint"):
                continue
            if key == "pages_by_kind" and isinstance(v, dict):
                for runs in v.values():
                    out.update(_runs_of(runs))
            elif key == "pages" or key.endswith("_pages"):
                if isinstance(v, (str, list)):
                    out.update(_runs_of(v))
            elif isinstance(v, (dict, list)):
                _named_runs(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _named_runs(v, out, depth + 1)


def _hit_pages(obj: Any, out: Set[int], depth: int = 0) -> None:
    """Every ``page`` of a hit in a search-like result."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        page = obj.get("page")
        if isinstance(page, int) and not isinstance(page, bool):
            out.add(page)
        for k, v in obj.items():
            if k in ("next", "hint"):
                continue
            if isinstance(v, (dict, list)):
                _hit_pages(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _hit_pages(v, out, depth + 1)


def _reached_by_call(name: str, args: dict, result: Any
                     ) -> Tuple[Optional[Tuple[str, str]],
                                List[Tuple[int, str]]]:
    """``(ref, [(page, how)])`` for one finished tool call."""
    from aec_agent_core.coverage import pages_from_call
    data = _json(result)
    if data is not None and data.get("error"):
        return None, []
    if name in OPENING_TOOLS:
        try:
            ref, pages = pages_from_call(name, args, result)
        except Exception:  # noqa: BLE001 - a result of an unknown shape
            ref, pages = None, []
        if pages and (not ref or not ref[1]):
            ref = _ref_of(args, data)          # a look given a handle
        if name in GRID_TOOLS and data is not None:
            extra = [(p, "text") for p in _grid_pages(data)]
            if extra:
                ref = ref or _ref_of(args, data)
                pages = list(pages) + extra
        if not pages and data is None and name == "read_document" \
                and isinstance(result, str):
            # A result cut short in the record, with no whole copy beside it.
            m = _PAGES_RETURNED.search(result)
            if m:
                ref = ("handle", str(args.get("handle") or ""))
                pages = [(p, "text") for p in _pages(m.group(1))]
        return ref, list(pages)
    if name in HIT_TOOLS and data is not None:
        got: Set[int] = set()
        _hit_pages(data, got)
        return _ref_of(args, data), [(p, HIT) for p in sorted(got)]
    return None, []


def _whole_result(rec: dict, base: str) -> Any:
    """A tool result, from its sidecar when the record cut it short."""
    result = rec.get("result")
    if rec.get("truncated") and rec.get("full"):
        path = os.path.join(base, str(rec["full"]))
        try:
            if os.path.getsize(path) <= MAX_SIDECAR_BYTES:
                with open(path, encoding="utf-8") as fh:
                    return fh.read()
        except OSError:
            pass
    return result


def _file_key(value: Any) -> str:
    """A file named in a result or an argument, by its base name (case
    folded): the record shortens paths one way and a call names them
    another."""
    text = str(value or "").strip().replace("\\", "/").rstrip("/")
    return os.path.basename(text).lower() if text else ""


def _note_sheets(st: _ActivityState, args: dict, result: Any) -> None:
    """The contact sheets a ``render_page_thumbnails`` call wrote, and the
    pages on each (seen only once a later call looks at the sheet)."""
    data = _json(result)
    if data is None or data.get("error"):
        return
    ref = _ref_of(args, data)
    if not ref[1]:
        return
    for sheet in data.get("sheets") or []:
        if not isinstance(sheet, dict):
            continue
        key = _file_key(sheet.get("image_path") or sheet.get("path"))
        pages = _pages(sheet.get("pages"))
        if key and pages:
            st.sheets[key] = (ref[0], ref[1], pages)


def _note_sheet_looks(st: _ActivityState, args: dict, result: Any) -> None:
    """A finished call of a looking tool (:data:`SHEET_LOOK_TOOLS`) whose
    arguments name a contact sheet written earlier (``analyze_image`` on
    its PNG) saw the pages on that sheet."""
    data = _json(result)
    if data is not None and data.get("error"):
        return
    values: List[Any] = []
    for v in args.values():
        values.extend(v if isinstance(v, (list, tuple)) else [v])
    for v in values:
        if not isinstance(v, str):
            continue
        got = st.sheets.get(_file_key(v))
        if got:
            st.sheet_seen.extend((got[0], got[1], p) for p in got[2])


def _note_told(st: _ActivityState, photos: Any) -> None:
    """The photograph pages an earlier note told the model about (the
    ``told`` of a :data:`PHOTO_KEY` record: 1-based PDF pages)."""
    told = photos.get("told") if isinstance(photos, dict) else None
    if not isinstance(told, dict):
        return
    for name, spec in told.items():
        got = {p - 1 for p in _pages(spec) if p >= 1}
        if got:
            st.photos_told.setdefault(doc_key(name), set()).update(got)


# -- what structure results say of each page (the answer check) --------------

#: A page kind (planlens' page map) in the words an answer uses for it.
_KIND_WORDS = {"blank": "blank empty", "scanned": "scanned scan image",
               "mixed": "mixed text graphics figure",
               "drawing_sheet": "drawing sheet", "photos": "photos photograph"}
#: Numbers a structure row gives that a page prints or is counted by (a
#: printed page number, a sheet number, a word count); other numbers in a
#: row (a segment id, a rotation) are not facts about the page.
_FACT_NUMBERS = ("printed_page", "printed_of", "label", "sheet", "words",
                 "n_pages", "sheet_count", "printed_pages")
#: What a page flagged as a duplicate (and the page it repeats) is said to
#: be, in the words an answer uses for it.
_DUPLICATE_WORDS = "duplicate duplicates copy repeat repeats same identical " \
                   "matches exactly"
#: Row keys that are bookkeeping, not facts about the page.
_FACT_SKIP = ("handle", "source", "look", "evidence", "duplicate_rule",
              "rule", "bbox", "confidence", "text_source")
#: Most facts kept for one page, and the length of each.
MAX_FACTS = 16
MAX_FACT_CHARS = 300


def _kind_words(kind: Any) -> str:
    k = str(kind or "").strip().lower()
    return _KIND_WORDS.get(k, k.replace("_", " "))


def _key_words(key: Any) -> str:
    """A page list's key as words: ``duplicate_pages`` -> "duplicate",
    ``pages_without_text_layer`` -> "without text layer"."""
    return " ".join(w for w in str(key).lower().split("_")
                    if w and w not in ("pages", "page", "by", "with"))


def _add_fact(out: Dict[int, List[str]], page: Any, text: Any) -> None:
    if not isinstance(page, int) or isinstance(page, bool) or page < 0:
        return
    text = str(text or "").strip()[:MAX_FACT_CHARS]
    if not text:
        return
    got = out.setdefault(page, [])
    if len(got) < MAX_FACTS and text not in got:
        got.append(text)


def _row_pages(obj: dict) -> List[int]:
    """The 0-based pages one structure row is about: its ``page``, or its
    ``pages`` when that is a page list (not a list of rows)."""
    page = obj.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        return [page]
    pages = obj.get("pages")
    if isinstance(pages, list) and not all(
            isinstance(p, int) and not isinstance(p, bool) for p in pages):
        return []
    if isinstance(pages, (str, list)) and pages:
        got = _pages(pages)
        return got if len(got) <= MAX_RANGE else []
    return []


#: Keys of an outline (bookmark) list: entries of ``level``, ``title`` and
#: the 0-based ``page`` each starts on.
_OUTLINE_KEYS = ("toc", "outline", "bookmarks")


def _outline_spans(entries: List[Any], n_pages: Optional[int],
                   out: Dict[int, List[str]]) -> None:
    """An outline entry names the page its part starts on; the pages after
    it, up to the next entry of the same or a higher level, are in that
    part too (the last entry's run to the document's end, when the page
    count is known and the list was not cut short)."""
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        p, title = e.get("page"), e.get("title")
        if not isinstance(p, int) or isinstance(p, bool) or p < 0 \
                or not title:
            continue
        lvl = e.get("level")
        rows.append((p, lvl if isinstance(lvl, int) else 1, str(title)))
    for i, (p, lvl, title) in enumerate(rows):
        end = next((q - 1 for q, l2, _t in rows[i + 1:] if l2 <= lvl), None)
        if end is None:
            end = n_pages - 1 if isinstance(n_pages, int) else p
        for x in range(p + 1, min(end, p + MAX_RANGE) + 1):
            _add_fact(out, x, title)


def _structure_facts(obj: Any, out: Dict[int, List[str]], depth: int = 0,
                     under: str = "", n_pages: Optional[int] = None) -> None:
    """What a structure result says of each 0-based page it names: a row's
    own words (title, heading, kind, label...) for the pages the row is
    about, a page list's key for the pages on it ("duplicate_pages",
    ``pages_by_kind``), an outline entry's title for the pages of its part,
    never the viewer's ``pdf_*`` twins."""
    if depth > 7:
        return
    if isinstance(obj, dict):
        n = obj.get("n_pages")
        if isinstance(n, int) and not isinstance(n, bool) and n > 0:
            n_pages = n
        for key in _OUTLINE_KEYS:
            if isinstance(obj.get(key), list):
                _outline_spans(obj[key], None if obj.get(key + "_truncated")
                               else n_pages, out)
        rows = _row_pages(obj)
        if rows:
            bits = [_key_words(under)] if under else []
            for k, v in obj.items():
                key = str(k)
                if key in ("page", "pages") or key in _FACT_SKIP \
                        or key.startswith(("pdf_", "next")) \
                        or isinstance(v, (dict, list)) \
                        or v is None or v == "" or v is False:
                    continue
                if key in ("kind", "role"):
                    bits.append(_kind_words(v))
                elif key == "duplicate_of":
                    bits.append(_DUPLICATE_WORDS + " second")
                    _add_fact(out, v, _DUPLICATE_WORDS + " original first")
                elif key == "needs_ocr":
                    bits.append("no text layer scan")
                elif key == "n_markups":
                    if isinstance(v, int) and v > 0:
                        bits.append("markups")
                elif v is True:
                    bits.append(_key_words(key))
                elif isinstance(v, (int, float)):
                    if key in _FACT_NUMBERS:
                        bits.append(f"{_key_words(key)} {v}")
                else:
                    bits.append(str(v))
            text = " ".join(b for b in bits if b)
            for p in rows:
                _add_fact(out, p, text)
        for k, v in obj.items():
            key = str(k)
            if key.startswith(("pdf_", "next")) or key in ("hint", "note",
                                                           "look"):
                continue
            if key == "pages_by_kind" and isinstance(v, dict):
                for kind, runs in v.items():
                    got = _pages(runs)
                    for p in got[:MAX_RANGE]:
                        _add_fact(out, p, _kind_words(kind))
            elif isinstance(v, str) and key != "pages" and (
                    key.endswith("_pages") or key.startswith("pages_")):
                for p in _pages(v)[:MAX_RANGE]:
                    _add_fact(out, p, _key_words(key))
            elif isinstance(v, (dict, list)):
                _structure_facts(v, out, depth + 1, key, n_pages)
    elif isinstance(obj, list):
        for v in obj[:5000]:
            _structure_facts(v, out, depth + 1, under, n_pages)


def _note_facts(st: _ActivityState, ref: Tuple[str, str], data: dict
                ) -> None:
    if not ref[1]:
        return
    got: Dict[int, List[str]] = {}
    _structure_facts(data, got)
    if not got:
        return
    mine = st.facts.setdefault(ref, {})
    for page, texts in got.items():
        for text in texts:
            _add_fact(mine, page, text)


#: A markup listing's compact row: "p3 (A-2) callout: ..." (0-based page).
_MARKUP_ROW = re.compile(r"^\s*p(\d{1,5})\b")


def _note_markups(st: _ActivityState, args: dict, result: Any) -> None:
    """The pages of the review markups a markup listing returned (a row is
    a compact string "p3 ..." or a dict with its ``page``)."""
    data = _json(result)
    if data is None or data.get("error"):
        return
    rows = data.get("markups")
    if not isinstance(rows, list):
        return
    ref = _ref_of(args, data)
    if not ref[1]:
        return
    for row in rows:
        page = None
        if isinstance(row, str):
            m = _MARKUP_ROW.match(row)
            page = int(m.group(1)) if m else None
        elif isinstance(row, dict):
            page = row.get("page")
        if isinstance(page, int) and not isinstance(page, bool) \
                and page >= 0:
            st.markups.append((ref[0], ref[1], page))


def _note_reference(st: _ActivityState, args: dict, result: Any) -> None:
    """The reference page(s) a reference reader rendered and read: its
    ``pdf_page_index`` (a figure, a worked example's source page) or each
    row of ``pages_read`` that has no error (a table), with the printed
    page its result gives, under the reference PDF's file name."""
    data = _json(result)
    if data is None or data.get("error"):
        return
    key = doc_key(data.get("source") or data.get("source_doc")
                  or args.get("reference") or args.get("example_id") or "")
    if not key:
        return
    pages: List[int] = []
    rows = data.get("pages_read")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and not row.get("error"):
                idx = row.get("pdf_page_index")
                if isinstance(idx, int) and not isinstance(idx, bool):
                    pages.append(idx)
    else:
        idx = data.get("pdf_page_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            pdf = data.get("pdf_page")
            idx = pdf - 1 if isinstance(pdf, int) \
                and not isinstance(pdf, bool) else None
        if isinstance(idx, int):
            pages.append(idx)
    printed = data.get("printed_page")
    mine = st.references.setdefault(key, {})
    for i, page in enumerate(p for p in pages if p >= 0):
        labels = mine.setdefault(page, set())
        if i == 0 and printed not in (None, ""):
            labels.add(_norm_label(printed))
    names = st.reference_names.setdefault(key, set())
    names.add(_norm_label(os.path.splitext(key)[0]))
    for k in ("figure_number", "table_number", "example_id", "reference"):
        v = data.get(k) or args.get(k)
        if v not in (None, ""):
            names.add(_norm_label(v))


def _feed_activity(st: _ActivityState, path: str) -> None:
    """Read what was added to ``path`` since the last call."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size < st.size_seen:            # rewritten: start over
        st.reset()
    if size == st.offset:
        st.size_seen = size
        return
    base = os.path.dirname(path)
    with open(path, "rb") as fh:
        fh.seek(st.offset)
        chunk = fh.read(size - st.offset)
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return                         # half a line: wait for the rest
    st.offset += cut + 1
    st.size_seen = size
    for raw in chunk[:cut + 1].splitlines():
        try:
            rec = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        ev = rec.get("event")
        rid = str(rec.get("run_id") or "")
        if ev == RECORD_EVENT:
            _note_told(st, rec.get(PHOTO_KEY))
            continue
        if ev == "tool_start" and rid:
            st.starts[rid] = (str(rec.get("name") or ""), rec.get("args"))
            continue
        if ev not in ("tool_end", "tool_error") or not rid:
            continue
        start = st.starts.pop(rid, None)
        if ev == "tool_error" or start is None:
            continue
        name, args = start
        args = _args_dict(args)
        result = _whole_result(rec, base)
        if name in CONTACT_SHEET_TOOLS:
            _note_sheets(st, args, result)
        elif st.sheets and name in SHEET_LOOK_TOOLS:
            _note_sheet_looks(st, args, result)
        if name == "open_document":
            data = _json(result) or {}
            handle = str(data.get("handle") or "")
            shown = str(data.get("name") or data.get("source")
                        or args.get("source") or "")
            if handle and shown:
                st.handle_names[handle] = shown
                st.sources.setdefault(doc_key(shown), set()).add(handle)
            src = str(args.get("source") or "")
            if src and shown:
                st.sources.setdefault(doc_key(shown), set()).add(src)
            n = data.get("n_pages")
            if shown and isinstance(n, int):
                st.n_pages[doc_key(shown)] = n
        if name in STRUCTURE_TOOLS:
            data = _json(result)
            if data is not None and not data.get("error"):
                _named_runs(data, st.runs)
                try:
                    _note_facts(st, _ref_of(args, data), data)
                except Exception:  # noqa: BLE001 - a fact the check lacks
                    pass
            continue
        if name in REFERENCE_TOOLS:
            try:
                _note_reference(st, args, result)
            except Exception:  # noqa: BLE001
                pass
            continue
        if name in MARKUP_TOOLS:
            try:
                _note_markups(st, args, result)
            except Exception:  # noqa: BLE001
                pass
        ref, pages = _reached_by_call(name, args, result)
        if not ref or not ref[1]:
            continue
        for page, how in pages:
            st.reached.append((ref[0], ref[1], page, how))


def _from_activity(path: str, out: PagesOpened) -> bool:
    if not os.path.isfile(path):
        return False
    st = _state_for(path)
    with st.lock:
        try:
            _feed_activity(st, path)
        except Exception:  # noqa: BLE001 - a record problem, not the turn's
            pass
        reached = list(st.reached)
        handle_names = dict(st.handle_names)
        n_pages = dict(st.n_pages)
        sources = {k: set(v) for k, v in st.sources.items()}
        out.runs |= st.runs
        sheet_seen = list(st.sheet_seen)
        for key, told in st.photos_told.items():
            out.photos_told.setdefault(key, set()).update(told)
        facts = {ref: {p: list(v) for p, v in rows.items()}
                 for ref, rows in st.facts.items()}
        markups = list(st.markups)
        for key, rows in st.references.items():
            mine = out.references.setdefault(key, {})
            for page, labels in rows.items():
                mine.setdefault(page, set()).update(labels)
        for key, names in st.reference_names.items():
            out.reference_names.setdefault(key, set()).update(names)

    def key_of(kind: str, value: str) -> str:
        if kind == "handle":
            shown = handle_names.get(value) or _handle_name(value)
            return doc_key(shown) if shown else value
        return doc_key(value)

    for (kind, value), rows in facts.items():
        key = key_of(kind, value)
        if key:
            mine = out.facts.setdefault(key, {})
            for page, texts in rows.items():
                for text in texts:
                    _add_fact(mine, page, text)
    for kind, value, page in markups:
        key = key_of(kind, value)
        if key:
            out.markups.setdefault(key, set()).add(page)
    for kind, value, page in sheet_seen:
        key = key_of(kind, value)
        if key:
            out.sheet_seen.setdefault(key, set()).add(page)
    for kind, value, page, how in reached:
        if kind == "handle":
            shown = handle_names.get(value) or _handle_name(value)
            key = doc_key(shown) if shown else value
            out.add(key, page, how, name=shown or value, source=value)
        else:
            out.add(doc_key(value), page, how, name=os.path.basename(
                value.replace("\\", "/")), source=value)
    for key, n in n_pages.items():
        out.n_pages.setdefault(key, n)
    for key, srcs in sources.items():
        out.sources.setdefault(key, set()).update(srcs)
    return True


def _handle_name(handle: str) -> Optional[str]:
    """The file a document handle stands for in THIS conversation, as the
    document tools remember it (``None`` when they do not)."""
    try:
        from aec_agent_core import document_tools
        got = document_tools.handle_source(handle)
    except Exception:  # noqa: BLE001
        return None
    return os.path.basename(str(got).replace("\\", "/")) if got else None


def _from_vision_records(folder, out: PagesOpened) -> List[str]:
    used = []
    try:
        from aec_agent_core.vision_tools import (readings_for_conversation,
                                                 reads_for_conversation)
    except Exception:  # noqa: BLE001
        return used
    for label, fn in (("vision reads", reads_for_conversation),
                      ("kept readings", readings_for_conversation)):
        try:
            rows = fn(folder)
        except Exception:  # noqa: BLE001
            rows = []
        n = 0
        for r in rows or []:
            page = r.get("page")
            name = str(r.get("document") or "")
            if isinstance(page, int) and name:
                out.add(doc_key(name), page, "look", name=name)
                n += 1
        if n:
            used.append(label)
    return used


def pages_opened(record_folder: Optional[str] = None,
                 folder=None) -> PagesOpened:
    """Every page any tool or helper reached in this conversation, from the
    app's records (never from what a model says it read).

    ``record_folder`` holds ``activity.jsonl``
    (:func:`record_dir` when ``None``); ``folder`` is the working folder the
    vision records are kept for (the one bound for the turn when ``None``).
    """
    out = PagesOpened()
    base = record_folder or record_dir(folder)
    if base:
        if _from_activity(os.path.join(base, ACTIVITY_FILE), out):
            out.records.append("activity log")
            out.knows_text = True
    out.records.extend(_from_vision_records(folder, out))
    return out


# ---------------------------------------------------------------------------
# A written file's pages (write_diggs, record_finding)
# ---------------------------------------------------------------------------

def _prov_pages(obj: Any, out: Set[int], depth: int = 0) -> None:
    """The ``page`` of every ``prov`` (one, or a list) anywhere in a record."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "prov":
                for p in (v if isinstance(v, list) else [v]):
                    page = p.get("page") if isinstance(p, dict) else None
                    if isinstance(page, int) and not isinstance(page, bool) \
                            and page >= 0:
                        out.add(page)
            elif isinstance(v, (dict, list)):
                _prov_pages(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _prov_pages(v, out, depth + 1)


def record_pages(record: Any) -> List[int]:
    """The 0-based pages one record cites: its ``pages`` and the page of any
    ``prov`` in it (a pydantic model or a plain dict)."""
    data = record.model_dump() if hasattr(record, "model_dump") else record
    if not isinstance(data, dict):
        return []
    got: Set[int] = set(_pages(data.get("pages")))
    _prov_pages(data, got)
    return sorted(got)


def _record_label(data: dict, group: str, i: int) -> str:
    inv = str(data.get("investigation_id") or "").strip()
    if group == "investigations":
        return inv or f"investigations[{i}]"
    sample = str(data.get("sample_id") or "").strip()
    kind = str(data.get("kind") or "test")
    who = " ".join(x for x in (inv, sample) if x)
    return f"{who} ({kind})" if who else f"lab_tests[{i}] ({kind})"


def record_entries(groups: Dict[str, Sequence[Any]], document: Any = None,
                   opened: Optional[PagesOpened] = None,
                   record_folder: Optional[str] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """``(entries, summary)`` for the records a writer was given
    (``{"investigations": [...], "lab_tests": [...]}``).

    One entry per record that cites a page no tool or helper read or looked
    at in this conversation; ``summary`` says what was checked (or, with no
    record of the text reads here, why nothing was). ``document`` (a writer's
    ``document_id``) narrows the pages read to the document it names, when it
    names one that was read."""
    cited_any = False
    per_record = []
    for group in ("investigations", "lab_tests"):
        for i, rec in enumerate(groups.get(group) or []):
            data = rec.model_dump() if hasattr(rec, "model_dump") else rec
            if not isinstance(data, dict):
                continue
            pages = record_pages(data)
            if pages:
                cited_any = True
                per_record.append((_record_label(data, group, i), pages))
    if not cited_any:
        return [], {"checked": False, "why": "no record cites a page"}
    if opened is None:
        opened = pages_opened(record_folder)
    if not opened.knows_text:
        return [], {"checked": False,
                    "why": "this host keeps no record of the pages read "
                           "(no activity log), so cited pages were not "
                           "checked"}
    docs = opened.documents(document) if opened.pages or opened.names \
        else []
    entries: List[Dict[str, Any]] = []
    n_cited = n_unread = 0
    for label, pages in per_record:
        n_cited += len(pages)
        never = [p for p in pages if opened.status(p, docs) == "never"]
        empty = [p for p in pages if opened.status(p, docs) == TEXT_EMPTY]
        hits = [p for p in pages if opened.status(p, docs) == HIT]
        unread = never + empty + hits
        if not unread:
            continue
        n_unread += len(unread)
        read = [p for p in pages if p not in set(unread)]
        cited = (f"PDF page {pages[0] + 1}" if len(pages) == 1
                 else f"PDF pages {compact(pages, 1)}")
        if read:
            what = (f"PDF page{'s' if len(unread) > 1 else ''} "
                    f"{compact(unread, 1)} of them")
        else:
            what = "it" if len(pages) == 1 else "any of them"
        parts = [f"{label} cites {cited} as where its values came from; no "
                 f"tool or helper read or looked at {what} in this "
                 "conversation"]
        if empty:
            parts.append(f"PDF pages {compact(empty, 1)} were reached only as "
                         "text and have no text layer")
        if hits:
            parts.append(f"PDF pages {compact(hits, 1)} appeared only in "
                         "search results")
        if read:
            parts.append(f"PDF {compact(read, 1)} were read")
        entries.append({"kind": ENTRY_KIND, "where": label,
                        "detail": "; ".join(parts)
                                  + ". Check those pages, or give the pages "
                                    "the values were read from.",
                        "pages": sorted(unread)})
    summary = {"checked": True, "pages_cited": n_cited,
               "not_read": n_unread,
               "against": ", ".join(opened.records) or "no record"}
    if len(docs) > 1 and document in (None, ""):
        summary["documents"] = ("pages read in any of the conversation's "
                                f"{len(docs)} documents count")
    return entries, summary


# ---------------------------------------------------------------------------
# Citations in an answer
# ---------------------------------------------------------------------------

_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-",
                         "\u2013": "-", "\u2014": "-", "\u2212": "-",
                         "\u00a0": " ", "\u202f": " ", "\u2009": " "})

#: "PDF page 24", "PDF pp. 108-119", "pdf_page 24", "PDF 133-134",
#: "PDF viewer page 7", "viewer page 7".
_PDF_LEAD = re.compile(
    r"(?i)(?<![\w-])(?:"
    r"pdf[ _]?(?:viewer[ \t]+)?(?:pages?\b|pgs?\b\.?|pp\b\.?|p\.)[ \t]*[:=]?"
    r"[ \t]*"
    r"|pdf[ \t]+(?=\d)"
    r"|(?:pdf[ \t]+)?viewer[ \t]+pages?\b[ \t]*"
    r")")
#: "page 12", "pages 3, 5 and 7", "pp. 12-14", "p. 5", "p.5", "pg 3",
#: "Page: 12".
_GEN_LEAD = re.compile(
    r"(?i)(?<![\w-])(?:pages?\b|pgs?\b\.?|pp\b\.?|p\.)[ \t]*[:#]?[ \t]*")
#: One cited page or range. A printed label: "A-3", "B-12"; a number or a
#: range "12", "12-14", "12 to 14"; a lower-case roman numeral (front
#: matter: "iv"), or an upper-case one of two letters or more ("XII"; "I"
#: alone is the pronoun).
_ITEM = re.compile(
    r"(?:(?P<lab>[A-Z]{1,2}-\d{1,4})"
    r"|(?P<a>\d{1,4})(?:[ \t]*(?:-|(?i:to|through|thru)[ \t])[ \t]*"
    r"(?P<b>\d{1,4}))?"
    r"|(?P<rom>[ivxlc]{1,7}|[IVXLC]{2,7})"
    r")(?![\w])(?!\.\d)")
_SEP = re.compile(r"(?i)[ \t]*(?:,[ \t]*(?:and\b|&|or\b)?|\band\b|&|\bor\b"
                  r"|\bto\b|\bthrough\b|\bthru\b)[ \t]*")
#: "page 3 of 10": a page of a count, not two pages.
_OF_COUNT = re.compile(r"(?i)[ \t]*(?:of|/)[ \t]*\d")
#: "pages 27 of 213 read", "12 of 40 pages viewed": how many, not which.
_COUNT_READ = re.compile(
    r"(?i)[ \t]*(?:of|/)[ \t]*\d[\d,]*[ \t]+(?:pages[ \t]+)?"
    r"(?:(?:were|are|have[ \t]+been|been)[ \t]+)?"
    r"(?:read|viewed|opened|skipped|looked|rendered|checked|seen|done)\b")
#: "page 24 of the PDF", "page 24 in the PDF viewer".
_OF_PDF = re.compile(r"(?i)\s*(?:of|in)\s+(?:the\s+)?pdf\b")
#: Where a clause starts, looking back from a citation.
_CLAUSE_END = re.compile(r"[.!?;]\s|\n|\|")

#: A published code, manual or work cited beside a page: the designation of
#: an agency or standards body's document, or an author and a year. A page
#: cited next to one belongs to that document, not to this conversation's.
_PUBLISHED = re.compile(
    r"(?<![\w-])(?:"
    r"(?:GEC|UFC|FHWA|NHI|AASHTO|ASTM|ACI|ASCE|SEI|IBC|NAVFAC|USACE|NCHRP|"
    r"TRB|CGPR|API|AISC|NEHRP|FEMA|NFPA|Eurocode|Caltrans|USBR|NRCS|USDA|"
    r"FPL|NAVDOCKS|PTI|DFI|ADSC|CIRIA)(?![a-z])"
    # Short designations only with their number ("DM7.2", "EM 1110",
    # "TM 5-818-1", "EN 1997"): "TM-1" is as likely a boring as a manual.
    r"|(?:DM|EM|TM|ETL|EC|EN|BS|ISO|NDS|PCA)\s?\d|DM-\d"
    r")|\bet\s+al\b")
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
           "sept", "oct", "nov", "dec")
_AUTHOR_YEAR = re.compile(
    r"\b(?P<who>[A-Z][A-Za-z'\-]+)(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?"
    r",?\s*\(?(?:19|20)\d{2}[a-z]?\)?")


@dataclass
class Cited:
    """One page citation in an answer."""
    shown: str
    #: ``pdf`` (viewer pages), ``page`` (a number that may be printed or
    #: the viewer's) or ``label`` (a printed label: "A-3", "5-2", "iv").
    kind: str
    #: 1-based numbers (``pdf`` and ``page``).
    numbers: List[int] = field(default_factory=list)
    #: The printed label (``label``), or the printed reading of a hyphenated
    #: number ("12-14": pages 12 to 14, or the page printed "12-14").
    label: str = ""
    start: int = 0
    #: Where the citation's phrase starts ("p. 12": at the "p."; a table
    #: cell: at the cell).
    lead: int = 0
    #: Set when the citation sits beside another document's name (a code, a
    #: manual, an author and year): not one of this conversation's pages.
    elsewhere: str = ""
    #: A run of printed labels ("pp. 4-1 to 4-3": 4-1, 4-2, 4-3), kind
    #: ``label``; each is one page.
    labels: List[str] = field(default_factory=list)
    #: Where the citation's item ends (a table cell: the cell's end).
    end: int = 0
    #: In a Markdown table row (the row is the citation's statement).
    table: bool = False
    #: Its lead names one page ("page", "p.", a column headed "Page"), not
    #: several ("pages", "pp.").
    singular: bool = False
    #: Said to be a printed number ("printed page 12", "chapter page 3-4",
    #: "manual page 5-2", a column headed "Printed page").
    printed: bool = False
    #: Where the lead it is under starts: the citation's own (``lead``), or
    #: a lead repeated in the same phrase ("PDF page 10, and PDF page 14":
    #: the second "PDF page"). Pages under one group are one subject.
    group: int = -1

    def all_labels(self) -> List[str]:
        return list(self.labels) or ([self.label] if self.label else [])


def _norm_label(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").translate(_DASHES)).lower()


def _item_cited(m: "re.Match", pdf: bool, start: int) -> Optional[Cited]:
    if m.group("lab"):
        lab = m.group("lab")
        return Cited(shown=f"page {lab}", kind="label",
                     label=_norm_label(lab), start=start)
    if m.group("rom"):
        rom = m.group("rom")
        return Cited(shown=f"page {rom}", kind="label",
                     label=_norm_label(rom), start=start)
    a = int(m.group("a"))
    b = int(m.group("b")) if m.group("b") else None
    word = "PDF page" if pdf else "page"
    if b is None:
        if a <= 0:
            return None
        return Cited(shown=f"{word} {a}", kind="pdf" if pdf else "page",
                     numbers=[a], start=start)
    if b < a:
        if pdf:          # "PDF 5-2": two pages, as written
            nums = [n for n in (a, b) if n > 0]
            return Cited(shown=f"{word}s {a}, {b}", kind="pdf",
                         numbers=sorted(set(nums)), start=start) \
                if nums else None
        return Cited(shown=f"page {a}-{b}", kind="label",
                     label=f"{a}-{b}", start=start)
    nums = (list(range(max(a, 1), b + 1)) if b - a <= MAX_RANGE
            else [max(a, 1), b])
    return Cited(shown=f"{word}s {a}-{b}", kind="pdf" if pdf else "page",
                 numbers=nums, label="" if pdf else f"{a}-{b}", start=start)


def _scan_items(text: str, pos: int, pdf: bool) -> Tuple[List[Cited], int]:
    """The items of one citation starting at ``pos`` (after its lead): each
    with whether it counts as a PDF page (a lead repeated after a separator,
    "PDF pages 3 and page 7", sets that for the items after it), and where
    that repeated lead starts (``group``; -1 under the citation's own)."""
    found: List[Tuple["re.Match", bool]] = []
    #: the separator before each item after the first ("to", ",", ...)
    seps: List[str] = [""]
    groups: List[int] = [-1]
    group = -1
    while True:
        m = _ITEM.match(text, pos)
        if not m:
            break
        found.append((m, pdf))
        pos = m.end()
        if _OF_COUNT.match(text, pos):
            break
        s = _SEP.match(text, pos)
        if not s or s.end() == pos:
            break
        nxt = s.end()
        pdf_lead = _PDF_LEAD.match(text, nxt)
        lead = pdf_lead or _GEN_LEAD.match(text, nxt)
        if lead:
            nxt = lead.end()
        if not _ITEM.match(text, nxt):
            break
        if lead:
            pdf = bool(pdf_lead)
            group = lead.start()
        seps.append(s.group(0).strip().lower())
        groups.append(group)
        pos = nxt
    if found and _OF_PDF.match(text, found[-1][0].end()):
        found = [(m, True) for m, _p in found]
    if len(found) == 1 and _COUNT_READ.match(text, found[0][0].end()):
        return [], pos        # "pages 27 of 213 read": a count, no page
    out: List[Cited] = []
    for (m, p), sep, grp in zip(found, seps, groups):
        c = _item_cited(m, p, m.start())
        if c is None:
            continue
        c.end = m.end()
        c.group = grp
        if out and sep in _RANGE_SEPS and _merge_labels(out[-1], c):
            out[-1].end = c.end
            continue
        out.append(c)
    return out, pos


#: Separators that make two printed labels one run ("4-1 to 4-3").
_RANGE_SEPS = ("to", "through", "thru")
#: A printed label of a numbered part: "4-1", "12-15", "A-3".
_PART_LABEL = re.compile(r"(?i)([a-z]{1,2}|\d{1,4})-(\d{1,4})")


def _merge_labels(first: Cited, second: Cited) -> bool:
    """``first`` becomes the run of printed labels "P-a" to "P-b" when the
    two name pages of one part P, the second after the first ("4-1 to 4-2",
    "12-15 through 12-17"); False, and nothing changed, otherwise."""
    if first.kind == "pdf" or second.kind == "pdf" or first.labels \
            or second.labels:
        return False
    a = _PART_LABEL.fullmatch(first.label or "")
    b = _PART_LABEL.fullmatch(second.label or "")
    if not a or not b or a.group(1).lower() != b.group(1).lower():
        return False
    lo, hi = int(a.group(2)), int(b.group(2))
    if hi <= lo or hi - lo > MAX_RANGE:
        return False
    part = a.group(1).lower()
    first.labels = [f"{part}-{n}" for n in range(lo, hi + 1)]
    first.shown = (f"pages {first.shown.split(' ', 1)[-1]} to "
                   f"{second.shown.split(' ', 1)[-1]}")
    first.kind = "label"
    first.numbers = []
    first.label = ""
    return True


def _table_citations(text: str) -> List[Cited]:
    """Citations in a Markdown table column headed "Page(s)" / "PDF page"
    (bare numbers in its cells)."""
    out: List[Cited] = []
    lines = text.split("\n")
    offsets = []
    at = 0
    for ln in lines:
        offsets.append(at)
        at += len(ln) + 1
    i = 0
    head_re = re.compile(r"(?i)\b(?:pdf[ _]?)?(?:pages?|pp|pgs?)\b")
    sep_re = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
    while i + 1 < len(lines):
        head, sep = lines[i], lines[i + 1]
        if head.count("|") < 2 or not sep_re.match(sep):
            i += 1
            continue
        cells = [c.strip() for c in head.strip().strip("|").split("|")]
        cols = {j: "pdf" in c.lower() for j, c in enumerate(cells)
                if head_re.search(c)}
        heads = {j: cells[j] for j in cols}
        j = i + 2
        while j < len(lines) and lines[j].count("|") >= 1 \
                and lines[j].strip():
            row = lines[j]
            parts = row.strip().strip("|").split("|")
            # where each cell starts in the whole text
            base = offsets[j] + (len(row) - len(row.lstrip()))
            if row.lstrip().startswith("|"):
                base += 1
            pos_in_row = 0
            for k, cell in enumerate(parts):
                if k in cols:
                    lead_ws = len(cell) - len(cell.lstrip())
                    start = base + pos_in_row + lead_ws
                    sub = cell.strip()
                    pdf = cols[k]
                    lead = _PDF_LEAD.match(sub) or _GEN_LEAD.match(sub)
                    if lead and _PDF_LEAD.match(sub):
                        pdf = True
                    items, _ = _scan_items(sub, lead.end() if lead else 0,
                                           pdf)
                    head = heads.get(k, "")
                    for c in items:
                        c.start += start
                        c.lead = start
                        c.group = start + c.group if c.group >= 0 else start
                        c.end = start + len(sub)
                        c.table = True
                        c.singular = not _PLURAL_HEAD.search(
                            lead.group(0) if lead else head)
                        c.printed = bool(_PRINTED_WORD.search(head))
                        out.append(c)
                pos_in_row += len(cell) + 1
            j += 1
        i = j
    return out


def _elsewhere(text: str, start: int, doc_names: Sequence[str]) -> str:
    """The other document a citation at ``start`` sits beside, or ``""``:
    a published code's or manual's designation, or an author and year, in
    the same clause before it or in the same parentheses after it - unless
    the clause names one of this conversation's documents."""
    lo = max(0, start - 100)
    before = text[lo:start]
    ends = list(_CLAUSE_END.finditer(before))
    if ends:
        before = before[ends[-1].end():]
    # After it: only inside the parentheses it sits in ("(p. 4 of GEC-12)").
    after = text[start:start + 80]
    close = after.find(")")
    stop = re.search(r"[;\n|(]", after)
    after = after[:close] if close >= 0 and (stop is None or stop.start()
                                             > close) else ""
    window = before + " " + after
    low = window.lower()
    for name in doc_names:
        stem = os.path.splitext(str(name).lower())[0]
        if len(stem) >= 4 and stem in low:
            return ""
    m = _PUBLISHED.search(window)
    if m:
        return m.group(0)
    # An author and year just before the page ("Das (2010), p. 350",
    # "(Bowles 1996, p. 512)"); a date ("drilled 3 Mar 2021") is not one.
    for m in _AUTHOR_YEAR.finditer(before[-40:]):
        who = m.group("who").lower()
        if who[:3] not in _MONTHS and who[:4] not in _MONTHS:
            return m.group(0).strip()
    return ""


#: A lead or a column head that names several pages.
_PLURAL_HEAD = re.compile(r"(?i)\b(?:pages|pp|pgs)\b")
#: A word that makes a page number a printed one, as a column head or just
#: before the lead: "printed page 12", "chapter page 3-4", "manual p. 5-2".
_PRINTED_WORD = re.compile(r"(?i)\b(?:printed|chapter|manual)\b")
_PRINTED_BEFORE = re.compile(r"(?i)\b(?:printed|chapter|manual)[ \t]+$")


def find_citations(text: Any, doc_names: Sequence[str] = ()) -> List[Cited]:
    """Every page citation in ``text``, in order, one per page or range."""
    t = str(text or "").translate(_DASHES)
    out: List[Cited] = []
    taken: List[Tuple[int, int]] = []
    for pdf, leads in ((True, _PDF_LEAD), (False, _GEN_LEAD)):
        for lead in leads.finditer(t):
            if any(a <= lead.start() < b for a, b in taken):
                continue
            items, end = _scan_items(t, lead.end(), pdf)
            singular = not _PLURAL_HEAD.search(lead.group(0))
            printed = bool(_PRINTED_BEFORE.search(
                t[max(0, lead.start() - 12):lead.start()]))
            for c in items:
                c.lead = lead.start()
                if c.group < 0:
                    c.group = c.lead
                c.singular = singular
                c.printed = printed
            if items:
                out.extend(items)
                taken.append((lead.start(), end))
    for c in _table_citations(t):
        if not any(a <= c.start < b for a, b in taken):
            out.append(c)
    seen: Set[Tuple] = set()
    uniq: List[Cited] = []
    for c in sorted(out, key=lambda c: c.start):
        c.elsewhere = _elsewhere(t, c.lead, doc_names)
        key = (c.kind, tuple(c.numbers), c.label, tuple(c.labels),
               bool(c.elsewhere))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# -- printed page numbers ----------------------------------------------------

#: A page number standing alone in a header or footer line: "12", "5-2",
#: "A-3", "iii", "Page 7" (the review digest's rule, read off planlens' page
#: map bands).
_PRINTED = re.compile(
    r"^(?:(?i:page|pg\.?)\s+)?(?P<v>(?:[A-Z]{1,2}-)?\d{1,4}(?:-\d{1,4})?"
    r"|[ivxlc]{1,7}|[IVXLC]{2,7})$")


def summary_labels(summary: Any) -> Set[str]:
    """What one planlens page-map row says the page is numbered: its
    printed ``Page N``, its PDF page label, and a number standing alone in
    its header or footer band."""
    out: Set[str] = set()
    printed = getattr(summary, "printed_page", None)
    if isinstance(printed, int):
        out.add(str(printed))
    label = getattr(summary, "label", None)
    if label:
        out.add(_norm_label(label))
    for band in (getattr(summary, "header", None),
                 getattr(summary, "footer", None)):
        for line in str(band or "").split(" | "):
            m = _PRINTED.match(line.strip().translate(_DASHES))
            if m:
                out.add(_norm_label(m.group("v")))
    return {x for x in out if x}


#: ``(opened, document key) -> {page: labels}`` or ``None``.
LabelSource = Callable[[PagesOpened, str], Optional[Dict[int, Set[str]]]]


def planlens_labels(opened: PagesOpened, key: str
                    ) -> Optional[Dict[int, Set[str]]]:
    """A document's printed page numbers from its planlens page map, the
    document opened the way the document tools open it; ``None`` when it
    cannot be opened here."""
    try:
        from aec_agent_core import document_tools
    except Exception:  # noqa: BLE001
        return None
    for src in sorted(opened.sources.get(key, ())):
        try:
            if document_tools.looks_like_handle(src):
                entry = document_tools.document_entry(src)
            else:
                entry = document_tools.open_document_entry(src)
            if entry is None:
                continue
            lock = getattr(entry, "lock", None) or threading.RLock()
            with lock:
                rows = list(entry.doc.page_map())
        except Exception:  # noqa: BLE001 - try the next way in
            continue
        opened.n_pages.setdefault(key, len(rows))
        return {r.page: summary_labels(r) for r in rows}
    return None


# -- the check ----------------------------------------------------------------

def _range_text(pages: Iterable[int]) -> str:
    ps = sorted(set(pages))
    return (f"PDF page {ps[0] + 1}" if len(ps) == 1
            else f"PDF pages {compact(ps, 1)}")


# -- what the answer says of a cited page -------------------------------------

#: Where a statement ends: a sentence end, a semicolon, a line break, or a
#: turn to another subject (", but", ", then").
_STATEMENT_END = re.compile(
    r"[.!?](?=[\s*_)\]]|$|[A-Z])|;|\n"
    r"|(?i:,?[ \t]+(?:but|while|whereas|although|though|however)\b)"
    r"|(?i:,[ \t]+then\b)")

#: The answer says a page was not opened, or offers it for a look: it names
#: the page, it does not cite it as where something came from. Anywhere in
#: the citation's statement ("PDF pages 10 and 14 were not viewed", "Not
#: read: PDF page 6", "if the stamp is on the scan (PDF p. 5), I can look
#: there").
_UNREAD = re.compile(
    r"(?i)(?:"
    r"\b(?:not|never)\s+(?:yet\s+|been\s+|be\s+)*"
    r"(?:viewed|read|opened|looked\s+at|rendered)\b"
    r"|\b(?:did|do|does|have|has|had|was|were|is|are|could|would|will)"
    r"(?:n['’]t|\s+not)\s+(?:yet\s+)?(?:been\s+)?"
    r"(?:view|read|open|look|render)\w*"
    r"|\b(?:cannot|can['’]t)\s+(?:view|read|open|look|see|render)\w*"
    r"|\bun(?:read|viewed|opened)\b"
    r"|\bno\s+tool\s+(?:opened|read|looked)\b"
    r"|\b(?:i|we)(?:\s+(?:can|could|will|would|may|might)|['’]ll)\s+"
    r"(?:also\s+)?(?:look|view|open|check|read|render|zoom)\b"
    r"|\b(?:want|like)\s+me\s+to\s+(?:look|view|open|check|read|render)\b"
    r"|\bshall\s+i\s+(?:look|view|open|check|read|render)\b"
    r"|\bneeds?\s+(?:to\s+be\s+)?(?:viewed|looked\s+at|opened|rendered"
    r"|viewing|looking\s+at|a\s+look)\b"
    r"|\b(?:should|must|could|can)\s+(?:also\s+)?(?:still\s+)?be\s+"
    r"(?:viewed|looked\s+at|opened|rendered)\b"
    r")")
#: The same in words that also say other things ("not needed", "should be
#: checked"): only right after the citation, of which it is said ("PDF page
#: 6 (the site plan) was not needed", "the blank p. 4 should be checked").
_UNREAD_AFTER = re.compile(
    r"(?i)[\s)\]*,:]*(?:\([^)\n]{0,120}\)[\s,]*)?"
    r"(?:(?:which|that|were|was|is|are|has|have|had|been|being|also|still"
    r"|yet|i|we)\s+)*"
    r"(?:(?:not|\w*n['’]t)\s+(?:yet\s+)?(?:been\s+)?"
    r"(?:needed|checked|reviewed|used|consulted)"
    r"|should\s+(?:also\s+)?(?:still\s+)?be\s+(?:checked|reviewed|confirmed)"
    r"|needs?\s+(?:to\s+be\s+)?(?:checked|reviewed|confirmed|checking))\b")

#: Words of a statement that say nothing of what is on a page.
_PLAIN_WORDS = frozenset("""
page pages printed print viewer this that these those there here with from
into onto than then them they their have been were which what when where
while each both also only just very more most some such other another about
after before between under over same like does will would could should
shows show shown holds hold contains contain carries carry gives give runs
starts start begins begin ends follow follows following appears appear
seems look looks numbered number numbers title titled heading headed
headings labelled labeled label named called reads read says said lists
listed flagged flags flag marks marked tools tool probably probable likely
apparently document documents file continue continues continued""".split())
_WORDS = re.compile(r"[A-Za-z][A-Za-z'’]*[A-Za-z]")
#: A number or identifier in a statement: "B-3", "2026", "1:100", "4.0".
_IDENT = re.compile(r"(?<![\w])(?:[A-Za-z]{1,3}[-.]?)?\d[\w.:/×-]*")
_FACT_WORDS = re.compile(r"[a-z][a-z']+")
#: Most pages one citation may name and still be read as "what the page map
#: said".
MAX_DESCRIBED = 200
#: Words that may stand between a printed page number and the PDF page the
#: answer gives for the same place ("printed page 12 (PDF page 14)", "PDF
#: page 13, printed page 1-3").
_PAIR_GAP = re.compile(
    r"(?i)(?:[\s,;:=()\[\]/-]|\b(?:printed|pdf|viewer|manual|chapter|in|a"
    r"|the|of|on|at|or|is)\b)*")


def _grp(c: Cited) -> int:
    """Where the lead a citation's page is under starts."""
    return c.group if c.group >= 0 else c.lead


def _masked(text: str, cites: Sequence[Cited]) -> str:
    """``text`` with every citation's phrase blanked (same length): a
    statement's words and numbers are then its own, not the page numbers."""
    chars = list(text)
    for c in cites:
        for i in range(max(0, c.lead), min(len(chars), max(c.end, c.start))):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def _statement_span(masked: str, c: Cited) -> Tuple[int, int]:
    """Where the statement a citation sits in starts and ends: its table
    row, else from the last statement end before it to the first after."""
    if c.table:
        lo = masked.rfind("\n", 0, c.lead) + 1
        hi = masked.find("\n", c.lead)
        return lo, hi if hi >= 0 else len(masked)
    lo = max(0, c.lead - 400)
    for m in _STATEMENT_END.finditer(masked, lo, c.lead):
        lo = m.end()
    end = max(c.end, c.lead)
    m = _STATEMENT_END.search(masked, end, end + 400)
    return lo, m.start() if m else min(len(masked), end + 400)


def _statement(masked: str, c: Cited) -> str:
    lo, hi = _statement_span(masked, c)
    return masked[lo:hi]


def _unread_cue(masked: str, c: Cited, cites: Sequence[Cited]
                ) -> Optional["re.Match"]:
    """A cue in the citation's statement that the answer names its page as
    unread, said of this page: after it with no other citation between
    ("PDF page 4 is blank, and PDF page 14 was not viewed" names only page
    14), or before it with no other citation before the cue ("Not read:
    PDF page 6, PDF page 9"; not "PDF page 14 was not viewed, and the stamp
    is on PDF page 10")."""
    lo, hi = _statement_span(masked, c)
    end = max(c.end, c.start)
    for m in _UNREAD.finditer(masked, lo, hi):
        if m.start() < _grp(c):
            between = (lo, m.start())
        else:
            between = (end, m.start())
        if not any(_grp(d) != _grp(c)
                   and between[0] <= _grp(d) < between[1] for d in cites):
            return m
    return _UNREAD_AFTER.match(masked, end)


#: Short words that turn a title into a finding ("has no drainage details"):
#: counted with a statement's other words though under four letters.
_NEGATIONS = frozenset({"no", "not", "nor"})


def _word_in(word: str, facts: Set[str]) -> bool:
    if word in facts or (word.endswith("s") and word[:-1] in facts):
        return True
    if len(word) < 4:
        return False
    return any(len(f) >= 4 and (word.startswith(f) or f.startswith(word))
               for f in facts)


def _ident_in(ident: str, text: str) -> bool:
    return re.search(r"(?<![0-9a-z])" + re.escape(ident) + r"(?![0-9a-z])",
                     text) is not None


def _described(statement: str, units: Sequence[Set[Tuple[str, int]]],
               facts_of: Callable[[str, int], str]) -> Optional[List[str]]:
    """The words a statement shares with what structure results said of
    every page it cites, or ``None`` when it says more than they did.

    The statement must (a) say of EACH cited page something a structure
    result said of it (a word of its kind, title, heading, label or flag,
    or one of its numbers), (b) give no number or identifier those results
    did not give for those pages, and (c) have every one of its other
    words (page vocabulary left out) among their words: one word more is a
    claim about the page ("the slope stability results on PDF page 20
    fail"), and a claim about a page no tool opened must fire."""
    if not units or len(units) > MAX_DESCRIBED:
        return None
    words = []
    for w in _WORDS.findall(statement):
        w = re.sub(r"['’]s$", "", w.lower())
        if (len(w) >= 4 and w not in _PLAIN_WORDS) or w in _NEGATIONS:
            words.append(w)
    idents = []
    for tok in _IDENT.findall(statement):
        tok = tok.lower().replace("×", "x").rstrip(".:/-")
        if any(ch.isdigit() for ch in tok):
            idents.append(tok)
    if not words and not idents:
        return None
    said: Set[str] = set()
    every: List[str] = []
    for unit in units:
        ok = False
        for k, p in unit:
            text = facts_of(k, p)
            if not text:
                continue
            every.append(text)
            fw = set(_FACT_WORDS.findall(text))
            hit = [w for w in words if _word_in(w, fw)]
            hit += [i for i in idents if _ident_in(i, text)]
            if hit:
                ok = True
                said.update(hit)
        if not ok:
            return None
    union = " ".join(every)
    if any(not _ident_in(i, union) for i in idents):
        return None
    if words:
        fw = set(_FACT_WORDS.findall(union))
        if any(not _word_in(w, fw) for w in words):
            return None        # a word the page map did not give: a claim
    return sorted(said)


def _paired(text: str, cites: Sequence[Cited]) -> Set[int]:
    """The citations of a page number that sit beside a PDF page given for
    the same place: in the same table row, or next to it with nothing
    between but punctuation and words such as "printed" ("printed page
    1-3; PDF page 13", "page 238 (PDF page 284)")."""
    out: Set[int] = set()
    for i, c in enumerate(cites):
        if c.kind == "pdf" or c.elsewhere:
            continue
        for d in cites:
            if d.kind != "pdf" or _grp(d) == _grp(c) or d.elsewhere:
                continue
            if c.table and d.table:
                if text.rfind("\n", 0, c.lead) == text.rfind("\n", 0,
                                                              d.lead):
                    out.add(i)
                    break
                continue
            a, b = (c, d) if _grp(c) < _grp(d) else (d, c)
            gap = text[max(a.end, a.start):_grp(b)]
            if len(gap) <= 40 and "\n" not in gap \
                    and _PAIR_GAP.fullmatch(gap):
                out.add(i)
                break
    return out


def check_answer(text: Any, opened: Optional[PagesOpened] = None,
                 record_folder: Optional[str] = None, folder=None,
                 label_source: Optional[LabelSource] = None
                 ) -> Dict[str, Any]:
    """Every page the answer cites, against the pages opened.

    Returns ``{"checked", "n_cited", "cited", "never_opened",
    "partly_opened", "hit_only", "page_map", "named_unread", "unverified",
    "other_documents", "against"}`` (``checked`` False, with ``why``, when
    there is no record of the text reads here). Each listed citation is
    ``{"cited": as written, "pages": the PDF pages it resolved to, ...}``.
    Only ``never_opened`` is a page cited that no tool opened; the module's
    docstring says what goes in each other list."""
    if opened is None and not _CHEAP_HINT.search(str(text or "")):
        return {"checked": True, "n_cited": 0}
    if opened is None:
        opened = pages_opened(record_folder, folder)
    names = [opened.names.get(k, k) for k in opened.documents()]
    t = str(text or "").translate(_DASHES)
    cites = find_citations(t, names)
    if not cites:
        return {"checked": True, "n_cited": 0}
    result: Dict[str, Any] = {
        "n_cited": len(cites),
        "cited": [c.shown for c in cites][:MAX_LISTED]}
    if not opened.knows_text:
        result.update(checked=False,
                      why="this host keeps no record of the pages read (no "
                          "activity log), so cited pages were not checked")
        return result
    result["checked"] = True
    result["against"] = ", ".join(opened.records) or "no record"
    docs = opened.documents()
    #: every document the record knows, read or only mapped (open_document)
    known = sorted(set(docs) | set(opened.n_pages))
    source = label_source or planlens_labels
    label_state: Dict[str, Any] = {"loaded": False, "value": None}
    masked = _masked(t, cites)
    paired = _paired(t, cites)
    all_markups: Set[int] = set().union(*opened.markups.values()) \
        if opened.markups else set()
    ref_keys = sorted(opened.references)
    #: the references a reference reader rendered that count for the
    #: citation being checked (set for each citation)
    cur_refs: List[str] = []
    REF = "ref:"

    def labels() -> Optional[Dict[str, Dict[int, Set[str]]]]:
        """Each document's printed page numbers; ``None`` unless every
        document's could be read (loaded once, and only when needed)."""
        if not docs:
            return None
        if not label_state["loaded"]:
            got: Dict[str, Dict[int, Set[str]]] = {}
            ok = True
            for key in docs:
                try:
                    rows = source(opened, key)
                except Exception:  # noqa: BLE001
                    rows = None
                if rows is None:
                    ok = False
                else:
                    got[key] = rows
            label_state.update(loaded=True, value=got if ok else None)
        return label_state["value"]

    def numbering() -> str:
        """How the documents read are numbered in print, as far as their
        page maps say: ``chapter`` (a page prints "4-2" or "A-3"),
        ``plain`` (at least half the pages of a document print a plain
        number), ``none`` (the printed numbers of most pages are not
        known)."""
        got = "none"
        for rows in (labels() or {}).values():
            plain = 0
            for ls in rows.values():
                if any(_PART_LABEL.fullmatch(x) for x in ls):
                    return "chapter"
                plain += any(x.isdigit() for x in ls)
            if rows and 2 * plain >= len(rows):
                got = "plain"
        return got

    def exists(idx: int) -> bool:
        ns = [opened.n_pages.get(k) for k in docs]
        return bool(docs) and any(n is None or idx < n for n in ns)

    def exists_known(idx: int) -> bool:
        ns = [opened.n_pages.get(k) for k in known]
        return bool(known) and any(n is None or idx < n for n in ns)

    def refs_for(c: Cited) -> List[str]:
        """The rendered references a citation may mean: every one where the
        conversation has no document of its own, else those its line names
        (a figure or table number, an example id, the reference)."""
        if not ref_keys or not docs:
            return list(ref_keys)
        lo = t.rfind("\n", 0, c.lead) + 1
        hi = t.find("\n", max(c.end, c.lead))
        line = _norm_label(t[lo:hi if hi >= 0 else len(t)])
        return [r for r in ref_keys
                if any(len(n) >= 3 and n in line
                       for n in opened.reference_names.get(r, ()))]

    # A reading is a list of UNITS, one per page (or printed number) the
    # citation names; a unit is the (document, page) pairs it could be, and
    # is opened when any of them was. ("", page) = that page of any document;
    # ("ref:<file>", page) = a page a reference reader rendered.
    Unit = Set[Tuple[str, int]]

    def status(k: str, p: int) -> str:
        if k.startswith(REF):
            return ("opened" if p in opened.references.get(k[len(REF):], {})
                    else "never")
        st = opened.status(p, [k] if k else None)
        if st == "opened":
            return st
        if p in (opened.markups.get(k, ()) if k else all_markups):
            return "opened"            # its review markups were listed
        if not k and any(p in opened.references.get(r, {})
                         for r in cur_refs):
            return "opened"            # a reference reader rendered it
        return st

    def is_open(k: str, p: int, how: str = "opened") -> bool:
        return status(k, p) == how

    def score(units: List[Unit]) -> Tuple[float, int]:
        n = sum(1 for u in units if any(is_open(k, p) for k, p in u))
        return (n / len(units) if units else 0.0, n)

    def printed_units(wanted: Sequence[str]) -> List[Unit]:
        lab = labels() or {}
        units = []
        for w in wanted:
            u = {(k, p) for k, rows in lab.items() for p, ls in rows.items()
                 if w in ls}
            u |= {(REF + r, p) for r in cur_refs
                  for p, ls in opened.references.get(r, {}).items()
                  if w in ls}
            if u:
                units.append(u)
        return units

    fact_cache: Dict[Tuple[str, int], str] = {}

    def facts_of(k: str, p: int) -> str:
        """What structure results said of page ``p`` (of document ``k``,
        or of any document), with its document's page count and the
        printed numbers its page map gives it, lower case."""
        if (k, p) not in fact_cache:
            bits: List[str] = []
            if not k.startswith(REF):
                for d in ([k] if k else sorted(set(opened.facts)
                                                | set(opened.n_pages))):
                    said = list(opened.facts.get(d, {}).get(p, ()))
                    if not said:
                        continue      # no structure result named the page
                    bits.extend(said)
                    bits.extend(sorted((labels() or {}).get(d, {})
                                       .get(p, ())))
                    n = opened.n_pages.get(d)
                    if isinstance(n, int) and 0 <= p < n:
                        bits.append(f"of {n} pages")
                        if p == n - 1:
                            bits.append("last final end through until")
            fact_cache[(k, p)] = " ".join(bits).lower().translate(
                _DASHES).replace("×", "x")
        return fact_cache[(k, p)]

    lists: Dict[str, List[Dict[str, Any]]] = {
        "never_opened": [], "partly_opened": [], "hit_only": [],
        "page_map": [], "named_unread": [], "unverified": [],
        "other_documents": []}
    for i, c in enumerate(cites):
        if c.elsewhere:
            lists["other_documents"].append({"cited": c.shown,
                                             "beside": c.elsewhere})
            continue
        if i in paired:
            continue          # the PDF page given beside it decides
        cur_refs[:] = refs_for(c)
        readings: List[Tuple[str, List[Unit]]] = []
        if c.kind in ("pdf", "page") and c.numbers:
            readings.append(("pdf", [{("", n - 1)} for n in c.numbers]))
        pdf_done = bool(readings) and score(readings[0][1])[0] >= 1.0
        if pdf_done:
            continue                                       # opened
        # Exactly a run of pages a structure tool named ("the logs, PDF
        # pages 41-90"): where something is, read off the page map.
        idx = sorted({n - 1 for n in c.numbers})
        mapped = (len(idx) >= 2 and idx[-1] - idx[0] + 1 == len(idx)
                  and (idx[0], idx[-1]) in opened.runs)
        how = "pdf"

        def file(name: str, row: Dict[str, Any]) -> None:
            if mapped and how == "pdf" and name in (
                    "never_opened", "partly_opened", "hit_only"):
                row["why"] = ("exactly a page run the page map or the "
                              "document's structure names: where something "
                              "is, not what a page says")
                name = "page_map"
            lists[name].append(row)

        def fire(row: Dict[str, Any], c: Cited = c,
                 readings: List[Tuple[str, List[Unit]]] = readings) -> None:
            """File a citation nothing it could mean was opened: not a
            citation where the answer names the page as unread, "what the
            page map said" where that is all it says of the page, else
            never opened."""
            if not (mapped and how == "pdf"):
                stmt = _statement(masked, c)
                cue = _unread_cue(masked, c, cites)
                if cue:
                    row["said"] = re.sub(r"\s+", " ", cue.group(0)).strip(
                        " )],:*")
                    row["why"] = ("the answer names this page as not opened, "
                                  "not needed or still to be looked at: not "
                                  "a citation of where something came from")
                    lists["named_unread"].append(row)
                    return
                for _how, units in readings:
                    if _how == "pdf":
                        # the other pages of the same phrase ("PDF pages 6
                        # and 7 are the logs of B-1 and B-2"): the statement
                        # is of all of them
                        units = units + [
                            {("", n - 1)} for d in cites
                            if d is not c and _grp(d) == _grp(c)
                            and not d.elsewhere and d.kind in ("pdf", "page")
                            for n in d.numbers]
                    # what the statement says of ANOTHER page being unread
                    # ("..., and PDF page 14 was not viewed") is not said of
                    # this one
                    said = _described(_UNREAD.sub(" ", stmt), units,
                                      facts_of)
                    if said:
                        row["matched"] = said
                        row["why"] = ("what the answer says of the page is "
                                      "what the page map or the document's "
                                      "structure said of it: where something "
                                      "is, not what the page says")
                        lists["page_map"].append(row)
                        return
            file("never_opened", row)

        chapter = (c.kind == "page" and (c.singular or c.printed)
                   and re.match(r"\d{1,4}-\d", t[c.start:c.start + 10])
                   is not None)
        if c.kind in ("page", "label") and (labels() is not None
                                            or cur_refs):
            if chapter and numbering() != "plain":
                # "p. 1-3" in documents numbered by chapter, or whose
                # printed numbers are not known: one printed page, never
                # a run of pages.
                readings = [("printed", printed_units([c.label]))]
            else:
                if c.kind == "page":
                    readings.append(("printed", printed_units(
                        [str(n) for n in c.numbers])))
                if c.labels:
                    readings.append(("printed", printed_units(c.labels)))
                elif c.label:
                    readings.append(("printed", printed_units([c.label])))
        readings = [(how, units) for how, units in readings if units]
        row: Dict[str, Any] = {"cited": c.shown}
        printed_found = any(how == "printed" for how, _u in readings)
        if c.kind != "pdf" and labels() is not None and not printed_found \
                and (c.printed or (chapter and numbering() != "plain")):
            # Said to be a printed number, or "p. 1-3" in documents numbered
            # by chapter or whose printed numbers are not known, and no page
            # of the documents read is known to print it: it cannot be
            # checked here, and its reading as PDF pages is not what it
            # says.
            row["why"] = ("reads as a printed page number that no page of "
                          "the documents read is known to print")
            lists["unverified"].append(row)
            continue
        if readings:
            # The reading with the most opened; on a tie a printed number
            # (what the prompt asks a citation to give first).
            how, units = max(readings, key=lambda r: (score(r[1]),
                                                      r[0] == "printed"))
            frac, n_got = score(units)
            pages = sorted({p for u in units for _k, p in u})
            row["pages"] = _range_text(pages)
            if how == "printed":
                row["read_as"] = "printed page number"
            if frac >= 1.0:
                continue                                   # opened
            if n_got:
                missing = sorted({p for u in units
                                  if not any(is_open(k, q) for k, q in u)
                                  for _k, p in u})
                row["not_opened"] = _range_text(missing)
                row["opened"] = f"{n_got} of {len(units)}"
                file("partly_opened", row)
                continue
            if any(is_open(k, p, HIT) for u in units for k, p in u):
                file("hit_only", row)
                continue
        # Nothing the citation could mean was opened.
        if c.kind == "pdf":
            if not any(exists_known(n - 1) for n in c.numbers):
                row["note"] = "no document read here has that page"
            fire(row, c, readings)
            continue
        if labels() is None:
            row["why"] = ("the printed page numbers of the documents read "
                          "could not be had here")
            lists["unverified"].append(row)
            continue
        if not printed_found and not any(exists(n - 1) for n in c.numbers):
            row.pop("pages", None)
            row["why"] = "no page of the documents read has that number"
            lists["unverified"].append(row)
            continue
        fire(row, c, readings)
    for k, v in lists.items():
        if v:
            result[k] = v[:MAX_LISTED]
            if len(v) > MAX_LISTED:
                result[f"n_{k}"] = len(v)
    return result


#: Cheap test before any record is read: does the text cite a page at all?
_CHEAP_HINT = re.compile(r"(?i)\b(?:pages?|pp|pgs?|pdf)\b|\bp\.")


def note_for_user(result: Dict[str, Any]) -> Optional[str]:
    """The note appended to an answer, or ``None``."""
    rows = result.get("never_opened") or []
    if not rows:
        return None
    listed = "; ".join(_row_text(r) for r in rows[:12])
    more = len(rows) - 12
    if more > 0:
        listed += f"; and {more} more"
    return ("**Page check (by the app).** This answer cites pages that no "
            "tool or helper read or looked at in this conversation: "
            f"{listed}. Treat those citations as unchecked until the pages "
            "are looked at.")


def note_for_model(result: Dict[str, Any]) -> Optional[str]:
    """The note the model is given once (``GEOTECH_CITED_PAGES=retry``)."""
    rows = result.get("never_opened") or []
    if not rows:
        return None
    lines = "\n".join(f"- {_row_text(r)}" for r in rows[:20])
    return (f"{NOTE_PREFIX} Before your answer goes to the user, the app "
            "checked every page it cites against its record of the pages any "
            "tool or helper read or looked at in this conversation. These "
            "citations name pages no tool opened:\n" + lines + "\n"
            "The answer you wrote is held back: the user will not see it. "
            "Open those pages and check what you say about them, or cite the "
            "pages the information came from, or say plainly where it came "
            "from if not from a page that was read. This note comes once. "
            "Then write your whole answer to the user's request again, as "
            "they will read it: it replaces the one held back.")


def _row_text(row: Dict[str, Any]) -> str:
    cited = str(row.get("cited") or "")
    pages = str(row.get("pages") or "")
    text = cited
    if pages and pages.lower() != cited.lower():
        text += f" ({pages})"
    if row.get("note"):
        text += f" - {row['note']}"
    return text


# ---------------------------------------------------------------------------
# Photograph pages nothing looked at (BACKLOG DR-48)
# ---------------------------------------------------------------------------

#: ``path -> (0-based photograph pages, "")`` or ``(None, why)``.
PhotoSource = Callable[[str], Tuple[Optional[List[int]], str]]

_PHOTO_CACHE: "OrderedDict[tuple, Tuple[Optional[List[int]], str]]" = \
    OrderedDict()
_PHOTO_LOCK = threading.Lock()
_PHOTO_KEEP = 64


def photograph_pages(path: str) -> Tuple[Optional[List[int]], str]:
    """The 0-based photograph pages of the PDF at ``path`` by planlens' own
    page roles (:data:`PHOTO_ROLE`), with ``""``; or ``(None, why)`` when
    its page kinds cannot be told here. Measured once per file (its path,
    size and time), with no model call."""
    if os.path.splitext(str(path))[1].lower() != ".pdf":
        return None, "not a PDF: its page kinds are not measured"
    try:
        info = os.stat(path)
    except OSError:
        return None, "the file is not in the conversation's folder"
    key = (os.path.normcase(os.path.realpath(path)), info.st_size,
           info.st_mtime_ns)
    with _PHOTO_LOCK:
        got = _PHOTO_CACHE.get(key)
        if got is not None:
            _PHOTO_CACHE.move_to_end(key)
            return got
    try:
        from planlens.document import open_document
        from planlens.document.roles import page_roles
    except Exception:  # noqa: BLE001 - an older planlens
        return None, "the installed planlens names no page roles"
    try:
        with open_document(str(path)) as doc:
            roles = page_roles(doc)
        got = (sorted({int(r.page) for r in roles
                       if getattr(r, "role", None) == PHOTO_ROLE}), "")
    except Exception as exc:  # noqa: BLE001 - a file planlens cannot read
        got = (None, "planlens could not measure its pages: "
                     f"{type(exc).__name__}")
    with _PHOTO_LOCK:
        _PHOTO_CACHE[key] = got
        while len(_PHOTO_CACHE) > _PHOTO_KEEP:
            _PHOTO_CACHE.popitem(last=False)
    return got


def _under(path: str, root: str) -> bool:
    try:
        top = os.path.realpath(root)
        return os.path.commonpath([os.path.realpath(path), top]) == top
    except ValueError:
        return False


def attached_documents(record_folder: str, opened: PagesOpened
                       ) -> Tuple[List[Tuple[str, Optional[str]]], str]:
    """The conversation's attached documents as ``(name, path or None)``,
    and where the list came from: the page's upload list
    (:data:`ATTACHMENTS_FILE` beside the activity log, each key a file in
    its ``files`` folder), else the documents the call log opened that are
    files in the conversation's own folder (a reference read by its path
    elsewhere is not an attachment)."""
    files = os.path.join(record_folder, "files")
    try:
        with open(os.path.join(record_folder, ATTACHMENTS_FILE),
                  encoding="utf-8") as fh:
            keys = json.load(fh)
    except (OSError, ValueError):
        keys = None
    out: List[Tuple[str, Optional[str]]] = []
    if isinstance(keys, list):
        for k in keys:
            name = str(k or "").strip()
            if not name:
                continue
            path = os.path.join(files, name)
            ok = _under(path, files) and os.path.isfile(path)
            out.append((name, path if ok else None))
        return out, "attachment list"
    for key in sorted(set(opened.documents()) | set(opened.sources)):
        for name in [opened.names.get(key)] + sorted(
                opened.sources.get(key, ())) + [key]:
            base = os.path.basename(str(name or "").replace("\\", "/"))
            if not os.path.splitext(base)[1]:
                continue             # a handle, not a file
            path = next((c for c in (os.path.join(files, base),
                                     os.path.join(record_folder, base))
                         if os.path.isfile(c)), None)
            if path:
                out.append((base, path))
                break
    return out, "call log"


def photo_coverage(record_folder: Optional[str] = None, folder=None,
                   opened: Optional[PagesOpened] = None,
                   photo_source: Optional[PhotoSource] = None
                   ) -> Optional[Dict[str, Any]]:
    """Per attached document, its photograph pages that nothing looked at
    (no look, and no contact sheet holding them that a later call looked
    at). No model call.

    ``None`` when the conversation has no record folder or no attached
    document. Else ``{"checked", "attached", "against", "documents" (those
    whose page kinds were told), "photos" (each one's photograph pages),
    "unseen" ({} when every photograph page was looked at)}``, with
    ``"told_before"`` (unseen pages an earlier note named) and ``"left_out"``
    (a document whose page kinds cannot be told here, and why) when any.
    Pages are PDF page numbers, 1-based as a viewer shows them."""
    base = record_folder or record_dir(folder)
    if not base:
        return None
    if opened is None:
        opened = pages_opened(base, folder)
    docs, where = attached_documents(base, opened)
    if not docs:
        return None
    if not opened.knows_text:
        return {"checked": False, "attached": where,
                "why": "this host keeps no record of the pages looked at "
                       "(no activity log)"}
    source = photo_source or photograph_pages
    result: Dict[str, Any] = {
        "checked": True, "attached": where,
        "against": ", ".join(opened.records) or "no record",
        "documents": [], "photos": {}, "unseen": {}}
    told_before: Dict[str, str] = {}
    left_out: Dict[str, str] = {}
    for name, path in docs[:MAX_LISTED]:
        if path is None:
            left_out[name] = "the file is not in the conversation's folder"
            continue
        try:
            pages, why = source(path)
        except Exception as exc:  # noqa: BLE001 - that document only
            pages, why = None, ("its pages could not be measured: "
                                f"{type(exc).__name__}")
        if pages is None:
            left_out[name] = why or "its page kinds cannot be told here"
            continue
        result["documents"].append(name)
        if not pages:
            continue
        key = doc_key(name)
        result["photos"][name] = compact(pages, 1)
        missing = sorted(set(pages) - opened.looked(key))
        if missing:
            result["unseen"][name] = compact(missing, 1)
            before = set(missing) & opened.photos_told.get(key, set())
            if before:
                told_before[name] = compact(before, 1)
    if told_before:
        result["told_before"] = told_before
    if left_out:
        result["left_out"] = left_out
    if len(docs) > MAX_LISTED:
        result["n_attached"] = len(docs)
    return result


def _pdf_pages_text(spec: str) -> str:
    return (f"PDF pages {spec}" if any(c in spec for c in "-,")
            else f"PDF page {spec}")


def photo_note_for_model(result: Optional[Dict[str, Any]]
                         ) -> Tuple[Optional[str], Dict[str, str]]:
    """The note the model is given once (``GEOTECH_PHOTO_PAGES=retry``),
    and the pages it names per document; ``(None, {})`` when every
    photograph page was looked at or was named by an earlier note."""
    unseen = (result or {}).get("unseen") or {}
    before = (result or {}).get("told_before") or {}
    tell: Dict[str, str] = {}
    for name, spec in unseen.items():
        left = sorted(set(_pages(spec)) - set(_pages(before.get(name))))
        if left:
            tell[name] = compact(left)
    if not tell:
        return None, {}
    lines = "\n".join(f"- {name}: {_pdf_pages_text(spec)}"
                      for name, spec in list(tell.items())[:20])
    return (f"{PHOTO_NOTE_PREFIX} Before your answer goes to the user, the "
            "app compared the photograph pages of the attached documents "
            "(the page kinds the document tools measure) with its record of "
            "the pages any tool or helper looked at in this conversation, a "
            "contact sheet counting once it was looked at. Nothing looked "
            "at these photograph pages:\n" + lines + "\n"
            "The answer you wrote is held back: the user will not see it. "
            "If your answer rests on these documents or names them as its "
            "basis, look at those pages and check what you say; otherwise "
            "say plainly that their photographs were not looked at. This "
            "note comes once. Then write your whole answer to the user's "
            "request again, as they will read it: it replaces the one held "
            "back."), tell


# ---------------------------------------------------------------------------
# One turn's check (webapp.core.stream_turn)
# ---------------------------------------------------------------------------

def _no_look_left() -> bool:
    """The running turn has met its ceiling on image calls (BACKLOG TA-43),
    so a note asking for a look at a page would only have that look
    refused. ``False`` with no ceiling set, outside a turn, and wherever
    the turn's scope cannot be read."""
    try:
        from aec_agent_core.deep.keep_work import image_ceiling_met
        return bool(image_ceiling_met())
    except Exception:  # noqa: BLE001 - never the turn
        return False


class TurnPageCheck:
    """The answer check of one turn: ``feedback`` before the answer goes out
    (``retry`` mode, once), ``finish`` when it does (records the check and,
    per the mode, notes the answer). Never raises.

    It carries the photograph-page check too (BACKLOG DR-48), by
    ``photo_mode`` (``GEOTECH_PHOTO_PAGES``): ``off``, nothing of it runs;
    ``record``, ``finish`` records it (:data:`PHOTO_KEY` in the same event);
    ``retry``, that and ``feedback`` tells the model once, in the same note
    as the page check's when both speak. The pages a note named are
    recorded as told (a ``photos_fed_back`` event) once the note is handed
    to the model: when ``feedback`` returns it, or, for a page built with
    ``confirms_notes`` (one that may drop a note it was given, as Document
    Review does when the turn's budget has ended), when the page calls
    :meth:`note_handed`."""

    def __init__(self, record_folder: Optional[str] = None, folder=None,
                 log: Optional[Callable[[Dict[str, Any]], None]] = None,
                 mode: Optional[str] = None,
                 label_source: Optional[LabelSource] = None,
                 photo_mode: Optional[str] = None,
                 photo_source: Optional[PhotoSource] = None,
                 confirms_notes: bool = False):
        self.record_folder = record_folder
        self.folder = folder
        self.log = log
        if mode is None:
            try:
                from aec_agent_core import review_flags
                mode = review_flags.cited_pages()
            except Exception:  # noqa: BLE001
                mode = "off"
        self.mode = mode
        if photo_mode is None:
            try:
                from aec_agent_core import review_flags
                photo_mode = review_flags.photo_pages()
            except Exception:  # noqa: BLE001
                photo_mode = "off"
        self.photo_mode = photo_mode
        self.label_source = label_source
        self.photo_source = photo_source
        self.confirms_notes = bool(confirms_notes)
        self.fed_back = False
        self.photos_told = False
        #: The photograph-page record of a note not yet handed to the model.
        self._photo_pending: Optional[Dict[str, Any]] = None
        #: What the last note was about: "pages", "photos" or both.
        self._told: Tuple[str, ...] = ()

    @property
    def retry_status(self) -> str:
        """The status line for the last note (the page check's alone is
        :data:`RETRY_STATUS`, as before)."""
        parts = [RETRY_STATUS] if "pages" in self._told else []
        if "photos" in self._told:
            parts.append(PHOTO_RETRY_STATUS)
        return "; ".join(parts) or RETRY_STATUS

    def _check(self, answer: str) -> Dict[str, Any]:
        try:
            return check_answer(answer, record_folder=self.record_folder,
                                folder=self.folder,
                                label_source=self.label_source)
        except Exception as exc:  # noqa: BLE001 - never the turn
            return {"checked": False, "n_cited": 0,
                    "why": f"the check failed: {type(exc).__name__}: {exc}"}

    def _photos(self) -> Optional[Dict[str, Any]]:
        """The photograph-page check (``None`` with no attached document);
        a failure is the record's, never the turn's."""
        if not self.record_folder:
            return None
        try:
            return photo_coverage(self.record_folder, self.folder,
                                  photo_source=self.photo_source)
        except Exception as exc:  # noqa: BLE001 - never the turn
            return {"checked": False,
                    "why": f"the check failed: {type(exc).__name__}: {exc}"}

    def _record(self, result: Dict[str, Any], action: str,
                photos: Optional[Dict[str, Any]] = None) -> None:
        if self.log is None or not (result.get("n_cited") or photos):
            return
        data = dict(result)
        if photos:
            data[PHOTO_KEY] = photos
            data["photo_mode"] = self.photo_mode
        try:
            self.log({**data, "mode": self.mode, "action": action})
        except Exception:  # noqa: BLE001
            pass

    def feedback(self, answer: str) -> Optional[str]:
        """The note to give the model before this answer goes out, once a
        turn for each check that has one (the page check in ``retry``
        mode, the photograph-page check with ``photo_mode`` ``retry``);
        else ``None``."""
        if not str(answer).strip():
            return None
        notes: List[str] = []
        told: List[str] = []
        result: Dict[str, Any] = {}
        if self.mode == "retry" and not self.fed_back:
            result = self._check(answer)
            note = note_for_model(result)
            if note:
                self.fed_back = True
                notes.append(note)
                told.append("pages")
        if (self.photo_mode == "retry" and not self.photos_told
                and not _no_look_left()):
            photos = self._photos()
            try:
                note, named = photo_note_for_model(photos)
            except Exception:  # noqa: BLE001 - never the turn
                note, named = None, {}
            if note:
                self.photos_told = True
                self._photo_pending = {**photos, "told": named}
                notes.append(note)
                told.append("photos")
        if not notes:
            return None
        self._told = tuple(told)
        if "pages" in told:
            self._record(result, "fed_back")
        if "photos" in told and not self.confirms_notes:
            self.note_handed()
        return "\n\n".join(notes)

    def note_handed(self) -> None:
        """The page handed this turn's note to the model: the photograph
        pages it named are recorded as told, so a later turn of the
        conversation does not tell them again. Nothing when no photograph
        note is waiting."""
        pending, self._photo_pending = self._photo_pending, None
        if pending:
            self._record({}, "photos_fed_back", pending)

    def finish(self, answer: str) -> str:
        """``answer`` as the user gets it: checked, recorded, and in
        ``note`` mode (or after a ``retry`` that still cites pages no tool
        opened) with the note appended. The photograph-page check is
        recorded with it when ``photo_mode`` is ``record`` or ``retry``."""
        try:
            photos = (self._photos()
                      if self.photo_mode in ("record", "retry") else None)
            if not str(answer or "").strip():
                if photos:
                    self._record({}, "none", photos)
                return answer
            result = self._check(answer)
            note = (note_for_user(result)
                    if self.mode in ("note", "retry") else None)
            self._record(result, "noted" if note else "none", photos)
            if note:
                return f"{str(answer).rstrip()}\n\n---\n{note}"
            return answer
        except Exception:  # noqa: BLE001 - never the turn
            return answer


__all__ = ["RECORD_EVENT", "ENTRY_KIND", "NOTE_PREFIX", "RETRY_STATUS",
           "PagesOpened", "Cited", "TurnPageCheck", "bind_record_dir",
           "unbind_record_dir", "record_bound", "record_dir", "pages_opened",
           "record_pages", "record_entries",
           "find_citations", "check_answer", "summary_labels",
           "planlens_labels", "note_for_user", "note_for_model",
           "clear_cache", "doc_key",
           "PHOTO_ROLE", "PHOTO_KEY", "PHOTO_NOTE_PREFIX",
           "PHOTO_RETRY_STATUS", "ATTACHMENTS_FILE", "CONTACT_SHEET_TOOLS",
           "SHEET_LOOK_TOOLS",
           "photograph_pages", "attached_documents", "photo_coverage",
           "photo_note_for_model"]
