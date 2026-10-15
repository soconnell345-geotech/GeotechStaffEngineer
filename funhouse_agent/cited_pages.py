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
* ``record_finding`` (the findings tools) says which of a finding's cited
  pages no tool opened (:func:`unopened_citations`).
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
``render_region``, or asked about by ``sweep_pages``), the coverage ledger
(``coverage.json``, with ``GEOTECH_COVERAGE`` on), the vision tools' read
record (``reads.json``) and kept readings (``page_readings.json``). A page
reached only by a text tool that found no text layer was not read (there was
nothing to read). A page named only in a search hit (``search_document``,
``find_quantities``...) is reported apart: the model saw a line of it, not
the page. Without the activity log or the coverage ledger nothing knows the
text reads, so nothing is checked and the result says so.

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
COVERAGE_FILE = "coverage.json"

#: Tools whose results name the pages they READ or LOOKED AT (the coverage
#: ledger's own list: :mod:`funhouse_agent.coverage`).
TEXT_TOOLS = ("read_document", "read_pdf_text")
LOOK_TOOLS = ("analyze_pdf_page", "render_region")
SWEEP_TOOLS = ("sweep_pages",)
GRID_TOOLS = ("log_grid",)
OPENING_TOOLS = TEXT_TOOLS + LOOK_TOOLS + SWEEP_TOOLS + GRID_TOOLS
#: Tools whose results name pages in HITS: a line of the page was seen, the
#: page was not opened.
HIT_TOOLS = ("search_document", "find_quantities", "digest_search",
             "document_markups", "find_like")

#: How a page was reached (``hows``): read as text, looked at, asked about by
#: a sweep, reached by a text tool and found to have no text layer, named in
#: a search hit.
OPENED_HOWS = frozenset({"text", "look", "sweep"})
TEXT_EMPTY = "text_empty"
HIT = "hit"

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
        from funhouse_agent._fileio import host_output_dir
        got = host_output_dir()
    except Exception:  # noqa: BLE001 - no folder, then
        return None
    return os.path.abspath(str(got)) if got else None


def record_dir(folder=None) -> Optional[str]:
    """The folder holding this conversation's ``activity.jsonl`` (or its
    ``coverage.json``): the one the turn bound, else the one the working
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
        if os.path.isfile(os.path.join(c, ACTIVITY_FILE)) or \
                os.path.isfile(os.path.join(c, COVERAGE_FILE)):
            return c
    return None


# ---------------------------------------------------------------------------
# Page lists
# ---------------------------------------------------------------------------

def _pages(spec: Any) -> List[int]:
    from funhouse_agent.coverage import parse_pages
    return parse_pages(spec)


def compact(pages: Iterable[int], offset: int = 0) -> str:
    from funhouse_agent.coverage import compact as _compact
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
    #: The records read ("activity log", "coverage ledger", ...).
    records: List[str] = field(default_factory=list)
    #: Whether a record of TEXT reads was found (the activity log or the
    #: coverage ledger); without one nothing is checked.
    knows_text: bool = False
    #: Page runs ``(first, last)``, 0-based, that a structure tool named
    #: (a segment, a run of one page kind, a work item): an answer naming
    #: exactly one describes where something is, from the page map.
    runs: Set[Tuple[int, int]] = field(default_factory=set)

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


_ACTIVITY: "OrderedDict[str, _ActivityState]" = OrderedDict()
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
    """Forget what the activity logs said (tests; a host that rewrites one)."""
    with _ACTIVITY_LOCK:
        _ACTIVITY.clear()


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
    is concerned (the coverage ledger, which asks whether a page was read
    WHOLE, is stricter)."""
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
    from funhouse_agent.coverage import pages_from_call
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
            continue
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
        from funhouse_agent import document_tools
        got = document_tools.handle_source(handle)
    except Exception:  # noqa: BLE001
        return None
    return os.path.basename(str(got).replace("\\", "/")) if got else None


def _from_coverage(path: str, out: PagesOpened) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    docs = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(docs, dict):
        return False
    for d in docs.values():
        if not isinstance(d, dict):
            continue
        shown = str(d.get("name") or d.get("source") or "")
        key = doc_key(shown)
        src = str(d.get("source") or "")
        inv = d.get("inventory") or {}
        if isinstance(inv, dict) and isinstance(inv.get("n_pages"), int):
            out.n_pages.setdefault(key, inv["n_pages"])
        for page, row in (d.get("reads") or {}).items():
            try:
                p = int(page)
            except (TypeError, ValueError):
                continue
            for how in (row or {}).get("how") or []:
                out.add(key, p, str(how), name=shown, source=src)
    return True


def _from_vision_records(folder, out: PagesOpened) -> List[str]:
    used = []
    try:
        from funhouse_agent.vision_tools import (readings_for_conversation,
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

    ``record_folder`` holds ``activity.jsonl`` / ``coverage.json``
    (:func:`record_dir` when ``None``); ``folder`` is the working folder the
    vision records are kept for (the one bound for the turn when ``None``).
    """
    out = PagesOpened()
    base = record_folder or record_dir(folder)
    if base:
        if _from_activity(os.path.join(base, ACTIVITY_FILE), out):
            out.records.append("activity log")
            out.knows_text = True
        cov = os.path.join(base, COVERAGE_FILE)
        if os.path.isfile(cov) and _from_coverage(cov, out):
            out.records.append("coverage ledger")
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


def unopened_citations(citations: Iterable[Tuple[Any, int]],
                       record_folder: Optional[str] = None,
                       opened: Optional[PagesOpened] = None
                       ) -> Optional[List[Dict[str, Any]]]:
    """``[{"document", "page", "pdf_page", "status"}]`` for each
    ``(document, 0-based page)`` that no tool opened in this conversation;
    ``None`` when there is no record of the text reads to check against."""
    cits = [(d, p) for d, p in citations
            if isinstance(p, int) and not isinstance(p, bool) and p >= 0]
    if not cits:
        return []
    if opened is None:
        opened = pages_opened(record_folder)
    if not opened.knows_text:
        return None
    out = []
    for document, page in cits:
        docs = opened.documents(document)
        st = opened.status(page, docs)
        if st != "opened":
            out.append({"document": str(document or ""), "page": page,
                        "pdf_page": page + 1, "status": st})
    return out


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
    "PDF pages 3 and page 7", sets that for the items after it)."""
    found: List[Tuple["re.Match", bool]] = []
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
        pos = nxt
    if found and _OF_PDF.match(text, found[-1][0].end()):
        found = [(m, True) for m, _p in found]
    out = [c for c in (_item_cited(m, p, m.start()) for m, p in found)
           if c is not None]
    return out, pos


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
                    for c in items:
                        c.start += start
                        c.lead = start
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
            for c in items:
                c.lead = lead.start()
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
        key = (c.kind, tuple(c.numbers), c.label, bool(c.elsewhere))
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
        from funhouse_agent import document_tools
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


def check_answer(text: Any, opened: Optional[PagesOpened] = None,
                 record_folder: Optional[str] = None, folder=None,
                 label_source: Optional[LabelSource] = None
                 ) -> Dict[str, Any]:
    """Every page the answer cites, against the pages opened.

    Returns ``{"checked", "n_cited", "cited", "never_opened",
    "partly_opened", "hit_only", "page_map", "unverified",
    "other_documents", "against"}`` (``checked`` False, with ``why``, when there is no record
    of the text reads here). Each listed citation is ``{"cited": as written,
    "pages": the PDF pages it resolved to, ...}``."""
    if opened is None and not _CHEAP_HINT.search(str(text or "")):
        return {"checked": True, "n_cited": 0}
    if opened is None:
        opened = pages_opened(record_folder, folder)
    names = [opened.names.get(k, k) for k in opened.documents()]
    cites = find_citations(text, names)
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
    source = label_source or planlens_labels
    label_state: Dict[str, Any] = {"loaded": False, "value": None}

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

    def exists(idx: int) -> bool:
        ns = [opened.n_pages.get(k) for k in docs]
        return bool(docs) and any(n is None or idx < n for n in ns)

    # A reading is a list of UNITS, one per page (or printed number) the
    # citation names; a unit is the (document, page) pairs it could be, and
    # is opened when any of them was. ("", page) = that page of any document.
    Unit = Set[Tuple[str, int]]

    def is_open(k: str, p: int, how: str = "opened") -> bool:
        return opened.status(p, [k] if k else None) == how

    def score(units: List[Unit]) -> Tuple[float, int]:
        n = sum(1 for u in units if any(is_open(k, p) for k, p in u))
        return (n / len(units) if units else 0.0, n)

    def printed_units(wanted: Sequence[str]) -> List[Unit]:
        lab = labels() or {}
        units = []
        for w in wanted:
            u = {(k, p) for k, rows in lab.items() for p, ls in rows.items()
                 if w in ls}
            if u:
                units.append(u)
        return units

    lists: Dict[str, List[Dict[str, Any]]] = {
        "never_opened": [], "partly_opened": [], "hit_only": [],
        "page_map": [], "unverified": [], "other_documents": []}
    for c in cites:
        if c.elsewhere:
            lists["other_documents"].append({"cited": c.shown,
                                             "beside": c.elsewhere})
            continue
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
        if c.kind in ("page", "label") and labels() is not None:
            if c.kind == "page":
                readings.append(("printed", printed_units(
                    [str(n) for n in c.numbers])))
            if c.label:
                readings.append(("printed", printed_units([c.label])))
        readings = [(how, units) for how, units in readings if units]
        row: Dict[str, Any] = {"cited": c.shown}
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
            if not any(exists(n - 1) for n in c.numbers):
                row["note"] = "no document read here has that page"
            file("never_opened", row)
            continue
        if labels() is None:
            row["why"] = ("the printed page numbers of the documents read "
                          "could not be had here")
            lists["unverified"].append(row)
            continue
        printed_found = any(how == "printed" for how, _u in readings)
        if not printed_found and not any(exists(n - 1) for n in c.numbers):
            row.pop("pages", None)
            row["why"] = "no page of the documents read has that number"
            lists["unverified"].append(row)
            continue
        file("never_opened", row)
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
# One turn's check (webapp.core.stream_turn)
# ---------------------------------------------------------------------------

class TurnPageCheck:
    """The answer check of one turn: ``feedback`` before the answer goes out
    (``retry`` mode, once), ``finish`` when it does (records the check and,
    per the mode, notes the answer). Never raises."""

    def __init__(self, record_folder: Optional[str] = None, folder=None,
                 log: Optional[Callable[[Dict[str, Any]], None]] = None,
                 mode: Optional[str] = None,
                 label_source: Optional[LabelSource] = None):
        self.record_folder = record_folder
        self.folder = folder
        self.log = log
        if mode is None:
            try:
                from funhouse_agent import review_flags
                mode = review_flags.cited_pages()
            except Exception:  # noqa: BLE001
                mode = "off"
        self.mode = mode
        self.label_source = label_source
        self.fed_back = False

    def _check(self, answer: str) -> Dict[str, Any]:
        try:
            return check_answer(answer, record_folder=self.record_folder,
                                folder=self.folder,
                                label_source=self.label_source)
        except Exception as exc:  # noqa: BLE001 - never the turn
            return {"checked": False, "n_cited": 0,
                    "why": f"the check failed: {type(exc).__name__}: {exc}"}

    def _record(self, result: Dict[str, Any], action: str) -> None:
        if self.log is None or not result.get("n_cited"):
            return
        try:
            self.log({**result, "mode": self.mode, "action": action})
        except Exception:  # noqa: BLE001
            pass

    def feedback(self, answer: str) -> Optional[str]:
        """The note to give the model before this answer goes out, once a
        turn, in ``retry`` mode; else ``None``."""
        if self.mode != "retry" or self.fed_back or not str(answer).strip():
            return None
        result = self._check(answer)
        note = note_for_model(result)
        if note:
            self.fed_back = True
            self._record(result, "fed_back")
        return note

    def finish(self, answer: str) -> str:
        """``answer`` as the user gets it: checked, recorded, and in
        ``note`` mode (or after a ``retry`` that still cites pages no tool
        opened) with the note appended."""
        try:
            if not str(answer or "").strip():
                return answer
            result = self._check(answer)
            note = (note_for_user(result)
                    if self.mode in ("note", "retry") else None)
            self._record(result, "noted" if note else "none")
            if note:
                return f"{str(answer).rstrip()}\n\n---\n{note}"
            return answer
        except Exception:  # noqa: BLE001 - never the turn
            return answer


__all__ = ["RECORD_EVENT", "ENTRY_KIND", "NOTE_PREFIX", "RETRY_STATUS",
           "PagesOpened", "Cited", "TurnPageCheck", "bind_record_dir",
           "unbind_record_dir", "record_bound", "record_dir", "pages_opened",
           "record_pages", "record_entries", "unopened_citations",
           "find_citations", "check_answer", "summary_labels",
           "planlens_labels", "note_for_user", "note_for_model",
           "clear_cache", "doc_key"]
