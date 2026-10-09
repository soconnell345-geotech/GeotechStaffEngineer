"""Coverage of a document, kept by the app rather than by the model.

WHY. In the 2026-10-06 field session (``module_work/field_feedback/
2026-10-06_geotech-report-session_v5.32.0/FINDINGS.md`` P3 and section 8) the
geotech agent was asked to take a report's subsurface data out. It read 10 of
23 classification sheets and none of the chemistry, compaction or density
sheets, never extracted one boring's lab results, and skipped the newest
borings' logs because it took "boring log" to mean "scanned page". It had
handed page ranges to a helper that said "none found". The prompt already
said to look at every page in scope. The owner, 2026-10-08: "If it has a task
that consists of extracting data, it should check *all* pages" - enforced in
CODE, generally, not as another prompt rule.

WHAT THIS MODULE IS (no LangChain in it; the middleware and the tools that
use it are :mod:`funhouse_agent.deep.coverage_tools`):

* an **inventory** of a document - every page with what it IS (a log, a lab
  sheet, a plan, text...) from the document's own structure, through
  planlens' page roles (:mod:`planlens.document.roles`, the same rules report
  ingest builds its work items from), and whether it has a usable text layer;
* a **ledger** of what was actually read: every page any tool or any helper
  read as text or looked at, derived from the tool calls themselves (their
  arguments and their results), never from what the model says it did; plus
  the pages the model marks ``extracted`` or ``skipped`` with a reason;
* **coverage as counts** from the ledger ("laboratory sheets: 10 of 23 read;
  not read: PDF pages 96-103");
* the **gate's note**: when a turn that took data out of a document is about
  to finish with data pages unread, the list the agent is told once.

"Read" means: looked at by a vision tool, or read as text by a text tool on a
page that HAS a usable text layer (``log_grid`` counts as a text read of each
page whose rows it returned in full). A scanned page read only as text was
not read - there was nothing to read - and the coverage says so.

Pages are 0-based here, as in every tool; ``pdf_pages`` (one higher) are what
a reader cites.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: The file the ledger is kept in, in the conversation's folder (beside
#: ``activity.jsonl``), so it survives an agent rebuild and is mirrored with
#: the rest of the conversation.
FILE_NAME = "coverage.json"

#: How the coverage report groups page roles: (key, what a reader calls
#: them, the planlens roles in it). Every role planlens can give is in one.
GROUPS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("logs", "exploration logs",
     ("boring_log", "test_pit_log", "cpt_log", "dcp_log")),
    ("lab", "laboratory sheets", ("lab_test",)),
    ("field_tests", "field test sheets", ("field_test",)),
    ("drawings", "plans, profiles and figures", ("plan", "profile", "figure")),
    ("calculations", "calculation pages", ("calculation",)),
    ("narrative", "text pages", ("narrative", "letter")),
    ("appended", "pages of reports bound inside it", ("appended_report",)),
    ("photos", "photograph pages", ("photos",)),
    ("other", "other pages", ("other",)),
    ("front", "cover, contents and divider pages", ("cover", "toc", "divider")),
)
GROUP_LABEL = {k: label for k, label, _ in GROUPS}
ROLE_GROUP = {role: k for k, _, roles in GROUPS for role in roles}

#: Groups that a whole-document task must cover (everything but the cover,
#: the contents and the tabs).
TARGET_GROUPS = tuple(k for k, _, _ in GROUPS if k != "front")
#: Groups that hold DATA. Reading some pages of one of these in a turn is
#: itself the shape of an extraction, so it holds the turn to the rest of
#: that group (the gate's automatic arming).
DATA_GROUPS = ("logs", "lab", "field_tests")

#: Tools whose results say which pages were READ AS TEXT.
TEXT_TOOLS = ("read_document", "read_pdf_text")
#: Tools that LOOK at a page (whole or zoomed).
LOOK_TOOLS = ("analyze_pdf_page", "render_region")
#: A tool that reads or looks at each page of a range itself.
SWEEP_TOOLS = ("sweep_pages",)
#: Tools that return a log's printed content row by row, each row naming its
#: page (planlens' ``log_grid``): a page whose rows all came back was read,
#: as text. Foundry brief 5 (N6): a run that took four log pages through
#: ``log_grid`` alone, every printed value returned, was told by the gate
#: that it had not read them.
GRID_TOOLS = ("log_grid",)
#: Not reads, by design: ``measure`` returns a position measured through the
#: page's scale, not the page's content - like a search, it answers one
#: question about a page; the look or the text read that found the thing is
#: the read. Nor are searches, thumbnails, ``find_like``, ``find_quantities``
#: or the digest's page rows.

#: Fewest planlens role confidence that is the page's own evidence: a page
#: that names itself (planlens "named" 0.8, "strong" 0.9), a log's
#: continuation sheet (0.7), a page of a bound report (0.85). Below it are
#: planlens' "weak" cues (0.6: a test named on a page of working, a few test
#: words, a form's shape), roles inherited from a tab (0.5-0.6) and guesses
#: from the page's shape (0.35-0.4).
STRONG_CONFIDENCE = 0.7
#: planlens evidence tags for a role INHERITED from the page's appendix tab
#: (the page says nothing about itself).
INHERITED_TAGS = ("tab-declares", "between-pages-of-one-log")
#: Version of the rules an inventory was built with; a saved inventory from
#: older rules is rebuilt the next time its document is touched.
INVENTORY_RULES = 2

#: Fewest characters of text for a page to count as having a text layer
#: (the same threshold ``read_pdf_text`` uses to call a page scanned).
MIN_TEXT_CHARS = 20

#: The gate note's fixed opening: also how the gate recognises its own note.
GATE_PREFIX = "[Coverage check]"

#: "10 of 23", "10/23", "all 23", "23 of 23": what stating coverage as a
#: count looks like in an answer.
_COUNT_RE = re.compile(
    r"\b\d{1,4}\s*(?:of|out of|/)\s*\d{1,4}\b|\ball\s+\d{1,4}\b"
    r"|\bevery one of the\s+\d{1,4}\b", re.IGNORECASE)


def states_coverage(text: str) -> bool:
    """Whether ``text`` states coverage as a count ("10 of 23 sheets")."""
    return bool(_COUNT_RE.search(str(text or "")))


# ---------------------------------------------------------------------------
# Page lists
# ---------------------------------------------------------------------------

def parse_pages(spec: Any, n_pages: Optional[int] = None) -> List[int]:
    """0-based pages from an int, a list, a string like ``"2-4,7"``, or a
    dict of those (``{"logs": "7-10", "lab": [16, 17]}``, or
    ``{"pages": [...]}``: every value is read). Anything unreadable is
    dropped; ``n_pages`` bounds the result."""
    out: List[int] = []
    if spec is None or spec == "":
        return out
    if isinstance(spec, bool):
        return out
    if isinstance(spec, int):
        out = [spec]
    elif isinstance(spec, dict):
        for v in spec.values():
            out.extend(parse_pages(v, n_pages))
    elif isinstance(spec, (list, tuple, set)):
        for v in spec:
            out.extend(parse_pages(v, n_pages))
    else:
        for part in str(spec).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(\d+)\s*(?:-|–|to)\s*(\d+)$", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a <= b and b - a <= 5000:
                    out.extend(range(a, b + 1))
            elif part.isdigit():
                out.append(int(part))
    if n_pages is not None:
        out = [p for p in out if 0 <= p < n_pages]
    return sorted(set(p for p in out if p >= 0))


def compact(pages: Iterable[int], offset: int = 0) -> str:
    """``[4,5,6,9]`` as ``"4-6,9"`` (``offset=1`` for viewer pages)."""
    ps = sorted(set(int(p) + offset for p in pages))
    if not ps:
        return ""
    out: List[str] = []
    start = prev = ps[0]
    for p in ps[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = p
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


# ---------------------------------------------------------------------------
# What a tool call read
# ---------------------------------------------------------------------------

def _json(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def pages_from_call(name: str, args: Optional[dict], result: Any
                    ) -> Tuple[Optional[Tuple[str, str]],
                               List[Tuple[int, str]]]:
    """``(source_ref, [(page, how), ...])`` for one tool call.

    ``source_ref`` is ``("handle", h)`` or ``("source", key_or_path)``: how
    the call named its document. ``how`` is ``"text"`` (the result carried
    that page's text), ``"text_empty"`` (a text tool reached the page and it
    had no text layer), ``"look"`` (a vision tool looked at it) or
    ``"sweep"`` (a sweep asked about it, by text or by looking). Pages come
    from the RESULT where the result says which pages it returned (a read cut
    short by its size limit did not read the rest), from the arguments where
    a successful look is about one page. A call that failed read nothing.
    """
    args = args or {}
    data = _json(result)
    if data is not None and data.get("error"):
        return None, []
    if name == "read_document":
        if data is None:
            return None, []
        ref = ("handle", str(args.get("handle") or data.get("handle") or ""))
        return ref, [(p, "text") for p in
                     parse_pages(data.get("pages_returned"))]
    if name == "read_pdf_text":
        if data is None:
            return None, []
        src = str(args.get("source") or args.get("attachment_key")
                  or args.get("path") or data.get("source") or "")
        out = []
        for row in data.get("pages") or []:
            if not isinstance(row, dict) or not isinstance(row.get("page"),
                                                           int):
                continue
            how = "text" if row.get("has_text_layer") else "text_empty"
            out.append((row["page"], how))
        return ("source", src), out
    if name in LOOK_TOOLS:
        if result is None or (isinstance(result, str) and not result.strip()):
            return None, []
        src = str(args.get("attachment_key") or args.get("source") or "")
        # The page the tool READ (its result names it), else the one asked
        # for -- a 1-based pdf_page included (live smoke wave 2c).
        page = data.get("page") if data is not None else None
        if not isinstance(page, int) or isinstance(page, bool):
            try:
                pdf_page = args.get("pdf_page")
                page = (int(pdf_page) - 1 if pdf_page not in (None, "")
                        else int(args.get("page", 0) or 0))
            except (TypeError, ValueError):
                return None, []
        return ("source", src), [(page, "look")]
    if name in SWEEP_TOOLS:
        if data is None:
            return None, []
        src = str(args.get("source") or "")
        checked = set(parse_pages(data.get("pages_checked")))
        failed = {r.get("page") for r in data.get("unanswered") or []
                  if isinstance(r, dict)}
        return ("source", src), [(p, "sweep") for p in sorted(checked - failed)]
    if name in GRID_TOOLS:
        if data is None:
            return None, []
        ref = ("handle", str(args.get("handle") or data.get("handle") or ""))
        return ref, [(p, "text") for p in _grid_pages_returned(args, data)]
    return None, []


def _grid_pages_returned(args: dict, data: dict) -> List[int]:
    """The pages whose rows a ``log_grid`` result has returned IN FULL.

    The rows come a window at a time (``offset`` in, ``next_offset`` out),
    in page order, each naming its page. A page is complete in this window
    when a later page's rows follow it, or when this is the last window
    (no ``next_offset``); the last page of a window that goes on may go on
    in the next. A window after the first (``offset`` > 0) also completes
    the pages of the call that come before its first row's page: their rows
    were the earlier windows' (a page whose rows ended exactly at a window's
    end shows only here). A call with no rows (``rows=false``, a scan with
    no text) returns no page's content and reads nothing."""
    rows = data.get("rows")
    if not isinstance(rows, list):
        return []
    order: List[int] = []
    for row in rows:
        page = row.get("page") if isinstance(row, dict) else None
        if isinstance(page, int) and not isinstance(page, bool) \
                and (not order or order[-1] != page):
            order.append(page)
    if not order:
        return []
    more = data.get("next_offset") not in (None, "", False)
    done = order[:-1] if more else list(order)
    try:
        offset = int(args.get("offset") or data.get("offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    if offset > 0:
        listed = parse_pages(data.get("pages"))
        if order[0] in listed:
            done.extend(listed[:listed.index(order[0])])
    return sorted(set(done))


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------

@dataclass
class PageInfo:
    """One page of the inventory."""
    page: int
    role: str
    group: str
    kind: str
    has_text: bool
    item: Optional[str] = None
    #: True for a page of a report bound inside this one; ``group`` is then
    #: what the page itself is (its inner role), when planlens could tell.
    bound: bool = False
    #: planlens' confidence in ``role`` and the tag of its evidence
    #: ("page-title", "tab-declares", ...).
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    #: Set when planlens calls the page a log, a lab sheet or a field-test
    #: sheet and the inventory does NOT count it as one (``group`` is then
    #: "other"): why - the evidence is weak, or the page stands alone.
    not_data: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"page": self.page, "role": self.role, "group": self.group,
             "kind": self.kind, "has_text": self.has_text,
             "item": self.item, "bound": self.bound}
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 2)
        if self.evidence:
            d["evidence"] = self.evidence
        if self.not_data:
            d["not_data"] = self.not_data
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageInfo":
        conf = d.get("confidence")
        return cls(page=int(d["page"]), role=str(d.get("role", "other")),
                   group=str(d.get("group", "other")),
                   kind=str(d.get("kind", "mixed")),
                   has_text=bool(d.get("has_text", True)),
                   item=d.get("item"), bound=bool(d.get("bound", False)),
                   confidence=float(conf) if conf is not None else None,
                   evidence=d.get("evidence"), not_data=d.get("not_data"))


#: Pages passed over when looking for a data page's neighbour: a tab, the
#: contents, a cover or a blank page sits between two appendices of data.
_PASS_OVER_ROLES = ("divider", "toc", "cover")


def _data_evidence(role_row: Any, has_text: bool) -> Optional[str]:
    """``None`` when planlens' evidence that a page IS a log, a lab sheet
    or a field-test sheet is strong; otherwise why it is not.

    Strong: the page names itself (planlens 0.7 and up), or it is a scan
    under a tab that declares it - a page with no text cannot name itself,
    so its tab is all the evidence there can be (the field session's older
    logs were scans behind a tab). Weak: one of planlens' own "weak" cues
    (0.6 - a test named on a page of working, a few test words, a form's
    shape), a role inherited from a tab by a page that HAS text and says
    nothing about itself (it could have, and did not), or a guess from the
    page's shape."""
    conf = float(getattr(role_row, "confidence", 0.0) or 0.0)
    ev = getattr(role_row, "evidence", None) or {}
    tag = ev.get("tag")
    if conf >= STRONG_CONFIDENCE:
        return None
    if tag in INHERITED_TAGS and not has_text:
        return None
    if tag in INHERITED_TAGS:
        return (f"weak evidence (planlens {conf:.2f}): the role is its "
                "appendix tab's, and the page, which has text, says nothing "
                "about itself")
    why = ev.get("why") or ev.get("rule") or "no title of its own"
    return f"weak evidence (planlens {conf:.2f}): {str(why)[:90]}"


def classify_pages(summaries: Sequence[Any], role_rows: Sequence[Any]
                   ) -> List[PageInfo]:
    """The inventory's pages from planlens' page map and page roles.

    Only STRONG evidence makes a page a log, a lab sheet or a field-test
    sheet here (Foundry brief 5, CV1: planlens' weak roles called 33 of the
    60 pages of a backfill manual laboratory sheets, the gate then held a
    narrow question to them and the answers told users "laboratory sheets:
    3 of 33 read"). A data page counts when:

    * its own evidence is strong (:func:`_data_evidence`), or it belongs to
      a planlens work item - a log's sheets, one test's pages - that has a
      page with strong evidence of the same kind (a continuation sheet); and
    * it stands with other data pages: the next data-like page before or
      after it (passing over tabs, contents, covers and blank pages) is one
      too. Logs and laboratory sheets come in sets - a log's sheets, an
      appendix of logs, a run of test sheets; a page the rules call a lab
      sheet, alone among pages of prose, is a page that names a test.

    A page that fails either is counted as ``other`` (``not_data`` says
    why): a declared or written-out extraction still covers it, and the
    implicit "data pages were read" arming does not see it.
    """
    roles: Dict[int, Any] = {getattr(r, "page", None): r for r in role_rows}
    pages: List[PageInfo] = []
    for s in summaries:
        r = roles.get(s.page)
        role = getattr(r, "role", None) or "other"
        group = ROLE_GROUP.get(role, "other")
        bound = role == "appended_report"
        evidence = getattr(r, "evidence", None) or {}
        if bound:
            inner = evidence.get("inner_role")
            if inner in ROLE_GROUP and ROLE_GROUP[inner] not in ("front",):
                group = ROLE_GROUP[inner]
        has_text = (int(getattr(s, "n_text_chars", 0) or 0)
                    >= MIN_TEXT_CHARS
                    and bool(getattr(s, "text_reliable", True))
                    and getattr(s, "kind", "") != "scanned")
        conf = getattr(r, "confidence", None)
        pages.append(PageInfo(page=s.page, role=role, group=group,
                              kind=str(getattr(s, "kind", "mixed")),
                              has_text=has_text,
                              item=getattr(r, "item_id", None), bound=bound,
                              confidence=(float(conf) if conf is not None
                                          else None),
                              evidence=evidence.get("tag")))

    # 1. Each data page's own evidence (a bound report's page: the binding
    #    is strong, and what the page is inside it is planlens' own reading).
    weak: Dict[int, str] = {}
    for p in pages:
        if p.group not in DATA_GROUPS or p.bound:
            continue
        why = _data_evidence(roles.get(p.page), p.has_text)
        if why:
            weak[p.page] = why
    # 2. A weak page of a work item that has a strong page of its group.
    strong_items = {(p.item, p.group) for p in pages
                    if p.group in DATA_GROUPS and p.item
                    and p.page not in weak}
    for p in pages:
        if p.page in weak and (p.item, p.group) in strong_items:
            del weak[p.page]
    # 3. Data pages stand with other data pages.
    by_page = {p.page: p for p in pages}

    def is_data(q: int) -> bool:
        info = by_page.get(q)
        return (info is not None and info.group in DATA_GROUPS
                and q not in weak)

    def neighbour(page: int, step: int) -> Optional[int]:
        q = page + step
        while q in by_page and (by_page[q].role in _PASS_OVER_ROLES
                                or by_page[q].kind == "blank"):
            q += step
        return q if q in by_page else None

    alone = {}
    for p in pages:
        if not is_data(p.page):
            continue
        if not any(q is not None and is_data(q)
                   for q in (neighbour(p.page, -1), neighbour(p.page, 1))):
            alone[p.page] = ("stands alone: no other log, laboratory or "
                             "field-test page beside it")
    for p in pages:
        why = weak.get(p.page) or alone.get(p.page)
        if why and p.group in DATA_GROUPS:
            p.not_data = why
            p.group = "other"
    return pages


@dataclass
class Inventory:
    """Every page of one document, with what it is."""
    name: str
    n_pages: int
    pages: List[PageInfo]
    items: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""
    #: :data:`INVENTORY_RULES` when built; 1 for an inventory saved before
    #: the rules were versioned.
    rules: int = INVENTORY_RULES

    @classmethod
    def from_document(cls, doc, name: str = "") -> "Inventory":
        """Built from a planlens Document: its page map (kind, text layer)
        and its page roles and work items (:func:`classify_pages`). Where
        the roles cannot be had (an old planlens, a document they fail on)
        every page is ``other`` and the inventory says so."""
        summaries = list(doc.page_map())
        n = len(summaries)
        role_rows: List[Any] = []
        items: List[Dict[str, Any]] = []
        note = ""
        try:
            from planlens.document.roles import roles_and_items
            role_rows, item_rows = roles_and_items(doc)
            for it in item_rows:
                items.append({"id": it.id, "kind": it.kind,
                              "title": it.title, "pages": list(it.pages)})
        except Exception as exc:  # noqa: BLE001 - an inventory without roles
            role_rows = []
            note = (f"page roles unavailable ({type(exc).__name__}); every "
                    "page is counted as 'other'")
        pages = classify_pages(summaries, role_rows)
        return cls(name=name or getattr(doc, "name", "") or "document",
                   n_pages=n, pages=pages, items=items, note=note)

    def info(self, page: int) -> Optional[PageInfo]:
        if 0 <= page < len(self.pages):
            return self.pages[page]
        return None

    def targets(self, groups: Optional[Iterable[str]] = None) -> List[int]:
        """The pages a task over these ``groups`` (default: every target
        group) must cover. Blank pages are never targets."""
        want = set(groups if groups is not None else TARGET_GROUPS)
        return [p.page for p in self.pages
                if p.group in want and p.kind != "blank"]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "n_pages": self.n_pages,
                "rules": self.rules,
                "pages": [p.to_dict() for p in self.pages],
                "items": list(self.items), "note": self.note}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Inventory":
        return cls(name=str(d.get("name", "")), n_pages=int(d.get("n_pages", 0)),
                   pages=[PageInfo.from_dict(p) for p in d.get("pages") or []],
                   items=list(d.get("items") or []),
                   note=str(d.get("note", "")),
                   rules=int(d.get("rules", 1) or 1))


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def _doc_text_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _sample_label(obj: Dict[str, Any]) -> str:
    """"B-2 S-3" for a lab test or a summary row (exploration and sample as
    given)."""
    inv = str(obj.get("investigation_id") or "?").strip()
    sample = str(obj.get("sample_id") or "").strip()
    return f"{inv} {sample}".strip()


class CoverageLedger:
    """What every tool and helper read of every document, in one place.

    One ledger per conversation (the builder makes it and binds it to the
    conversation's folder, ``path``); the middleware feeds it from every
    tool call of the primary agent AND its helpers, and the tools read it.
    Thread-safe: tool calls run in parallel. Never raises out of
    :meth:`record_call` - a coverage problem must not fail a tool call.
    """

    def __init__(self, path: Optional[str] = None,
                 attachments: Optional[Dict[str, bytes]] = None):
        self.path = path
        self.attachments = attachments if attachments is not None else {}
        self._lock = threading.RLock()
        self.seq = 0
        #: doc_key -> {"name", "inventory", "reads", "marks", "declared",
        #: "scope", "source"}; ``scope`` is the declared task's pages (None:
        #: the whole document).
        self.docs: Dict[str, Dict[str, Any]] = {}
        self._handles: Dict[str, str] = {}       # handle -> doc_key
        self._sources: Dict[str, Tuple[str, str]] = {}  # source -> (handle, key)
        self.turn_starts: Dict[str, int] = {}
        #: The user turn begun last (:meth:`begin_turn`).
        self.current_turn: Optional[str] = None
        self.fired: Dict[str, Dict[str, Any]] = {}
        #: Extraction outputs written (write_diggs), newest last.
        self.outputs: List[Dict[str, Any]] = []
        self.last_error: Optional[str] = None
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self.last_error = f"could not read {self.path}: {exc}"
            return
        self.seq = int(data.get("seq", 0))
        for key, d in (data.get("documents") or {}).items():
            inv = Inventory.from_dict(d["inventory"]) \
                if d.get("inventory") else None
            if inv is not None and inv.rules < INVENTORY_RULES:
                inv = None        # older rules: rebuilt when next touched
            scope = d.get("scope")
            self.docs[key] = {
                "name": d.get("name", ""),
                "inventory": inv,
                "reads": {int(p): r for p, r in (d.get("reads") or {}).items()},
                "marks": {int(p): m for p, m in (d.get("marks") or {}).items()},
                "declared": d.get("declared"),
                "scope": parse_pages(scope) if scope else None,
                "source": d.get("source"),
            }
        self.turn_starts = dict(data.get("turn_starts") or {})
        self.fired = dict(data.get("gate") or {})
        self.outputs = list(data.get("outputs") or [])

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            data = {
                "about": ("The app's own record of which pages of each "
                          "document any tool or helper read (text) or looked "
                          "at, the pages the agent marked extracted or "
                          "skipped, and each coverage check the agent was "
                          "given. Pages are 0-based."),
                "seq": self.seq,
                "documents": {
                    k: {"name": d["name"], "source": d.get("source"),
                        "declared": d.get("declared"),
                        "scope": (compact(d["scope"]) if d.get("scope")
                                  else None),
                        "inventory": (d["inventory"].to_dict()
                                      if d.get("inventory") else None),
                        "reads": {str(p): r for p, r in
                                  sorted(d["reads"].items())},
                        "marks": {str(p): m for p, m in
                                  sorted(d["marks"].items())}}
                    for k, d in self.docs.items()},
                "turn_starts": self.turn_starts,
                "gate": self.fired,
                "outputs": self.outputs[-10:],
            }
            # Written under the lock: parallel tool calls each save, and two
            # writers of one ".part" file would corrupt it.
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                            exist_ok=True)
                tmp = self.path + ".part"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=1,
                              default=str)
                os.replace(tmp, self.path)
            except OSError as exc:
                self.last_error = f"could not write {self.path}: {exc}"

    # -- which document ----------------------------------------------------
    def _entry_for(self, ref: Tuple[str, str]):
        from funhouse_agent import document_tools
        kind, value = ref
        if not value:
            return None
        if kind == "handle":
            return document_tools.document_entry(value)
        return document_tools.open_document_entry(value, self.attachments)

    def _identity(self, entry) -> str:
        """A key for a document's CONTENT, so an attachment key, a path to
        the same file and a planlens handle all land in one place."""
        handle = getattr(entry, "handle", "")
        if handle in self._handles:
            return self._handles[handle]
        key = None
        try:
            from funhouse_agent import document_tools
            path = getattr(entry, "path", None)
            data = None
            if path and os.path.isfile(path):
                with open(path, "rb") as fh:
                    data = fh.read()
            else:
                src = getattr(entry, "source", None)
                if src:
                    got = document_tools.resolve_document_source(
                        src, self.attachments)
                    if isinstance(got, (bytes, bytearray)):
                        data = bytes(got)
                    elif got and os.path.isfile(got):
                        with open(got, "rb") as fh:
                            data = fh.read()
            if data:
                key = "sha1:" + _doc_text_sha1(data)
        except Exception:  # noqa: BLE001 - fall back to the handle
            key = None
        key = key or f"handle:{handle}"
        self._handles[handle] = key
        return key

    def _source_sig(self, source: str) -> Tuple:
        """What a source string points at NOW: an attachment's size, or a
        file's size and time. A re-upload under the same name is a new
        document."""
        data = self.attachments.get(source)
        if isinstance(data, (bytes, bytearray)):
            return (source, "bytes", len(data))
        try:
            st = os.stat(source)
            return (source, "path", st.st_size, st.st_mtime_ns)
        except OSError:
            return (source, "unknown")

    def resolve(self, ref: Optional[Tuple[str, str]]) -> Optional[str]:
        """The document key for a call's ``source_ref``, registering the
        document (and building its inventory) on first sight."""
        if not ref or not ref[1]:
            return None
        sig = self._source_sig(ref[1]) if ref[0] == "source" else None
        if sig is not None and sig in self._sources:
            handle, key = self._sources[sig]
            if key in self.docs and self.docs[key].get("inventory"):
                return key
        entry = self._entry_for(ref)
        if entry is None:
            return None
        key = self._identity(entry)
        if sig is not None:
            self._sources[sig] = (entry.handle, key)
        with self._lock:
            d = self.docs.get(key)
            if d is None:
                d = self.docs[key] = {"name": getattr(entry, "name", ""),
                                      "inventory": None, "reads": {},
                                      "marks": {}, "declared": None,
                                      "scope": None, "source": None}
            if not d.get("source"):
                d["source"] = (getattr(entry, "source", None)
                               or getattr(entry, "path", None) or ref[1])
            if d.get("inventory") is None:
                lock = getattr(entry, "lock", None) or threading.RLock()
                with lock:
                    d["inventory"] = Inventory.from_document(
                        entry.doc, name=getattr(entry, "name", ""))
                d["name"] = d["inventory"].name
        return key

    # -- recording ---------------------------------------------------------
    def next_seq(self) -> int:
        with self._lock:
            self.seq += 1
            return self.seq

    def record_call(self, name: str, args: Optional[dict], result: Any,
                    agent: str = "primary") -> List[int]:
        """Record what one finished tool call read. Returns the pages
        recorded (for tests); never raises."""
        try:
            if name == "call_agent":
                self._record_output(args or {}, result)
                return []
            ref, pages = pages_from_call(name, args, result)
            if not pages:
                return []
            key = self.resolve(ref)
            if key is None:
                return []
            seq = self.next_seq()
            with self._lock:
                d = self.docs[key]
                n = d["inventory"].n_pages if d.get("inventory") else None
                done = []
                for page, how in pages:
                    if n is not None and not 0 <= page < n:
                        continue
                    row = d["reads"].setdefault(
                        page, {"how": [], "by": [], "tools": [],
                               "first": seq, "last": seq})
                    for k, v in (("how", how), ("by", agent), ("tools", name)):
                        if v not in row[k]:
                            row[k].append(v)
                    row["last"] = seq
                    done.append(page)
            self.save()
            return done
        except Exception as exc:  # noqa: BLE001 - never fail a tool call
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []

    def _record_output(self, args: dict, result: Any) -> None:
        """A data file written from what was read (``subsurface.write_diggs``):
        the turn took data out of a document. Its cross-checks are kept for
        the checklist, and the pages its data cite are marked extracted."""
        agent_name = str(args.get("agent_name") or "")
        method = str(args.get("method") or "")
        if method != "write_diggs" or "subsurface" not in agent_name:
            return
        data = _json(result) or {}
        if data.get("error") and not data.get("output_path"):
            return
        params = args.get("parameters") or {}
        if isinstance(params, str):
            params = _json(params) or {}
        params = dict(params) if isinstance(params, dict) else {}
        # call_agent hoists stray top-level keys into parameters; so do we.
        for k in ("investigations", "lab_tests"):
            if k not in params and isinstance(args.get(k), list):
                params[k] = args[k]
        cited: List[int] = []
        rows = 0
        uncited: List[str] = []
        summary_rows: List[str] = []
        sheets: List[str] = []
        for group in ("investigations", "lab_tests"):
            for obj in params.get(group) or []:
                if not isinstance(obj, dict):
                    continue
                rows += 1
                got = parse_pages(obj.get("pages"))
                if got:
                    cited.extend(got)
                else:
                    uncited.append(str(obj.get("investigation_id")
                                       or obj.get("sample_id")
                                       or obj.get("kind") or group))
                if group != "lab_tests":
                    continue
                if obj.get("kind") == "summary_table":
                    result_ = obj.get("result") or {}
                    for row in (result_.get("rows") or []
                                if isinstance(result_, dict) else []):
                        if isinstance(row, dict):
                            summary_rows.append(_sample_label(row))
                else:
                    sheets.append(_sample_label(obj)
                                  + f" ({obj.get('kind') or 'test'})")
        seq = self.next_seq()
        with self._lock:
            self.outputs.append({
                "seq": seq, "time": round(time.time(), 1),
                "tool": "subsurface.write_diggs",
                "output_path": data.get("output_path"),
                "verdict": data.get("verdict"),
                "cross_checks": data.get("cross_checks"),
                "rows": rows, "rows_without_pages": uncited[:40],
                "pages_cited": compact(cited),
                "summary_rows": summary_rows[:200],
                "sheets": sheets[:200]})
            # The data cite pages of ONE document only if one is in play.
            touched = [k for k, d in self.docs.items() if d["reads"]]
            if len(touched) == 1 and cited:
                d = self.docs[touched[0]]
                for p in set(cited):
                    d["marks"].setdefault(p, {"status": "extracted",
                                              "by": "write_diggs",
                                              "seq": seq})
        self.save()

    def declared_this_turn(self, key: str) -> bool:
        """Whether the document has a coverage task opened in the current
        user turn."""
        d = self.docs.get(key) or {}
        start = self.turn_starts.get(self.current_turn or "", 0)
        return d.get("declared") is not None and int(d["declared"]) >= start

    def declare(self, key: str, pages: Optional[Iterable[int]] = None
                ) -> None:
        """The agent opened a coverage task on this document, over ``pages``
        (``None``: the whole document).

        A declaration STAYS PUT (Foundry brief 5, CV3: every call of the
        tool re-declared the whole document, so a question about one chapter
        of a 538-page manual was held to 400 pages it never touched). Within
        one user turn, a call that names no ``pages`` keeps the task's scope;
        ``pages`` sets it. A new turn's task starts over: the whole document
        unless it names its pages. Marking pages is not a declaration
        (:meth:`mark`)."""
        with self._lock:
            d = self.docs[key]
            scope = sorted(set(pages)) if pages is not None else None
            if self.declared_this_turn(key):
                if pages is not None:
                    d["scope"] = scope or None
            else:
                d["declared"] = self.next_seq()
                d["scope"] = scope or None
        self.save()

    def scope(self, key: str) -> Optional[List[int]]:
        """The declared task's pages on this document (``None``: all)."""
        return (self.docs.get(key) or {}).get("scope") or None

    def mark(self, key: str, pages: Iterable[int], status: str,
             reason: str = "") -> List[int]:
        """Mark pages ``extracted`` or ``skipped`` (with ``reason``)."""
        seq = self.next_seq()
        with self._lock:
            d = self.docs[key]
            n = d["inventory"].n_pages if d.get("inventory") else None
            done = []
            for p in pages:
                if n is not None and not 0 <= p < n:
                    continue
                row = {"status": status, "seq": seq, "by": "agent"}
                if reason:
                    row["reason"] = reason
                d["marks"][p] = row
                done.append(p)
        self.save()
        return done

    # -- coverage ----------------------------------------------------------
    def covered(self, key: str, page: int) -> bool:
        """Read: looked at, swept, or read as text where there is text."""
        d = self.docs.get(key) or {}
        row = (d.get("reads") or {}).get(page)
        if not row:
            return False
        how = set(row.get("how") or [])
        if how & {"look", "sweep"}:
            return True
        inv = d.get("inventory")
        info = inv.info(page) if inv else None
        return "text" in how and (info is None or info.has_text)

    def _status(self, key: str, page: int) -> str:
        """``read`` / ``skipped`` / ``text_only`` (read as text, has none) /
        ``unread``."""
        d = self.docs[key]
        if self.covered(key, page):
            return "read"
        mark = d["marks"].get(page) or {}
        if mark.get("status") == "skipped":
            return "skipped"
        if page in d["reads"]:
            return "text_only"
        return "unread"

    def group_rows(self, key: str, groups: Optional[Iterable[str]] = None,
                   pages: Optional[Iterable[int]] = None
                   ) -> List[Dict[str, Any]]:
        """Per group: pages, read, extracted, skipped, and what is not read
        (over ``pages`` only, when given: a declared task's scope)."""
        d = self.docs[key]
        inv: Inventory = d["inventory"]
        want = list(groups) if groups is not None else [k for k, _, _ in GROUPS]
        within = set(pages) if pages is not None else None
        rows = []
        for g in want:
            pages = [p.page for p in inv.pages if p.group == g
                     and p.kind != "blank"
                     and (within is None or p.page in within)]
            if not pages:
                continue
            status = {p: self._status(key, p) for p in pages}
            read = [p for p in pages if status[p] == "read"]
            skipped = [p for p in pages if status[p] == "skipped"]
            unread = [p for p in pages if status[p] in ("unread", "text_only")]
            text_only = [p for p in pages if status[p] == "text_only"]
            extracted = [p for p in pages
                         if (d["marks"].get(p) or {}).get("status")
                         == "extracted"]
            claimed = [p for p in extracted if p not in read]
            row = {"group": g, "label": GROUP_LABEL[g], "pages": len(pages),
                   "read": len(read), "extracted": len(extracted),
                   "skipped": len(skipped)}
            if g not in TARGET_GROUPS:
                # Counted, never asked for: a cover or a tab holds no data.
                rows.append(row)
                continue
            if unread:
                row["not_read"] = compact(unread)
                row["not_read_pdf"] = compact(unread, 1)
            if text_only:
                row["read_as_text_but_no_text_layer"] = compact(text_only)
            if claimed:
                row["marked_extracted_but_never_read"] = compact(claimed)
            if skipped:
                row["skipped_pages"] = compact(skipped)
            bound = [p for p in pages if inv.pages[p].bound]
            if bound:
                row["in_bound_reports"] = compact(bound)
            rows.append(row)
        return rows

    def statement(self, key: str,
                  groups: Optional[Iterable[str]] = None,
                  pages: Optional[Iterable[int]] = None) -> str:
        """Coverage in one line, for the answer: counts, and the PDF pages
        not read (over ``groups``, default every target group; over
        ``pages`` only, when given)."""
        d = self.docs[key]
        pages = list(pages) if pages is not None else None
        parts = []
        for row in self.group_rows(key, tuple(groups) if groups is not None
                                   else TARGET_GROUPS, pages):
            s = f"{row['label']} {row['read']} of {row['pages']} read"
            if row.get("not_read_pdf"):
                s += f" (not read: PDF pages {row['not_read_pdf']})"
            if row["skipped"]:
                s += f", {row['skipped']} skipped"
            parts.append(s)
        where = f" (PDF pages {compact(pages, 1)})" if pages else ""
        if not parts:
            return f"Coverage of {d['name']}{where}: no pages to read."
        return f"Coverage of {d['name']}{where}: " + "; ".join(parts) + "."

    def report(self, key: str, max_items: int = 40) -> Dict[str, Any]:
        """What the coverage tool returns: the inventory by group, what was
        read, and the line to state in the answer - over the declared
        task's pages when it named them."""
        d = self.docs[key]
        inv: Inventory = d["inventory"]
        scope = self.scope(key) if self.declared_this_turn(key) else None
        out: Dict[str, Any] = {"document": d["name"], "n_pages": inv.n_pages}
        if scope:
            out["task_pages"] = compact(scope)
            out["task_pages_pdf"] = compact(scope, 1)
            out["pages_outside_the_task"] = inv.n_pages - len(scope)
        out.update({
            "groups": self.group_rows(key, pages=scope),
            "statement": self.statement(key, pages=scope),
            "how_read_is_counted": (
                "From the app's record of every tool call in this "
                "conversation, yours and your helpers': a page is read when "
                "a vision tool looked at it, or a text tool read it and it "
                "has a text layer. A page with no text layer read only as "
                "text was not read. Pages are 0-based; *_pdf lists are what "
                "a reader cites."),
        })
        not_data = [p.page for p in inv.pages if p.not_data
                    and (scope is None or p.page in scope)]
        if not_data:
            out["counted_as_other"] = (
                f"pages {compact(not_data)}: the page roles call them logs "
                "or test sheets on weak evidence, or they stand alone among "
                "pages of another kind, so they are counted as other pages")
        within = set(scope) if scope else None
        counted = {p.page for p in inv.pages if not p.not_data}
        items = [it for it in inv.items
                 if it.get("kind") not in ("front_matter",)
                 and any(p in counted for p in it.get("pages") or [])
                 and (within is None
                      or any(p in within for p in it.get("pages") or []))]
        if items:
            rows = []
            for it in items[:max_items]:
                ps = it.get("pages") or []
                read = sum(1 for p in ps if self.covered(key, p))
                rows.append({"kind": it.get("kind"), "title": it.get("title"),
                             "pages": compact(ps), "read": f"{read}/{len(ps)}"})
            out["items"] = rows
            if len(items) > max_items:
                out["items_more"] = len(items) - max_items
        if inv.note:
            out["note"] = inv.note
        return out

    # -- the gate ----------------------------------------------------------
    def begin_turn(self, turn_key: str) -> None:
        with self._lock:
            if turn_key not in self.turn_starts:
                self.turn_starts[turn_key] = self.seq + 1
            self.current_turn = turn_key

    def gate_fired(self, turn_key: str) -> bool:
        return turn_key in self.fired

    def armed(self, turn_key: str
              ) -> Dict[str, Tuple[Tuple[str, ...], str]]:
        """Documents this turn is held to: ``{key: (groups, why)}``.

        * ``declared``: the agent opened a coverage task on the document
          this turn (the coverage tool) - every target group, over the
          task's pages (:meth:`held_pages`);
        * ``output``: a data file was written from what was read
          (``write_diggs``) and the document was read this turn - every
          target group;
        * ``data``: otherwise, when pages of any DATA group (logs, lab
          sheets, field tests) were read this turn - every data group. A
          turn that reads data pages is taking data out; the field session
          read old logs and some lab sheets and missed the newest logs and
          the rest of the sheets, which are the other pages of exactly
          these groups.
        """
        start = self.turn_starts.get(turn_key, 0)
        wrote = any(o.get("seq", 0) >= start for o in self.outputs)
        out: Dict[str, Tuple[Tuple[str, ...], str]] = {}
        with self._lock:
            for key, d in self.docs.items():
                inv = d.get("inventory")
                if inv is None:
                    continue
                touched = [p for p, r in d["reads"].items()
                           if int(r.get("last", 0)) >= start]
                declared = d.get("declared")
                if declared is not None and int(declared) >= start:
                    out[key] = (TARGET_GROUPS, "declared")
                elif wrote and touched:
                    out[key] = (TARGET_GROUPS, "output")
                else:
                    groups = {inv.info(p).group for p in touched
                              if inv.info(p) is not None}
                    if groups & set(DATA_GROUPS):
                        out[key] = (DATA_GROUPS, "data")
        return out

    def held_pages(self, key: str, why: str) -> Optional[List[int]]:
        """The pages a turn armed for ``why`` is held to on this document:
        a declared task's own pages, else ``None`` (the whole document)."""
        return self.scope(key) if why == "declared" else None

    def gate_note(self, turn_key: str, answer: str = "",
                  extra: Sequence[str] = ()) -> Optional[str]:
        """The note the agent is given once, or ``None``.

        It is given when a document the turn is held to (:meth:`armed`) has
        pages nobody read; or, for a declared or written-out extraction with
        everything read, when the answer does not state coverage as counts;
        or when ``extra`` (failed checklist checks) has anything in it.

        The note goes to the model INSTEAD of its answer: the gate holds the
        reply back (:class:`funhouse_agent.deep.coverage_tools.CoverageGate`)
        and the reply after the note is the one the user gets. So the note
        asks for the whole answer, not a continuation (Foundry brief 5,
        CV2/N4: "This note comes once; then finish your answer" was read
        two ways, and every gated answer reached the user as the first reply
        glued to the second)."""
        armed = self.armed(turn_key)
        if not armed:
            return None
        lines: List[str] = []
        any_unread = False
        explicit = any(why != "data" for _g, why in armed.values())
        for key, (groups, why) in armed.items():
            held = self.held_pages(key, why)
            rows = [r for r in self.group_rows(key, groups, held)]
            unread_rows = [r for r in rows if r.get("not_read")]
            name = self.docs[key]["name"]
            if held:
                name += f" (the task's pages: PDF pages {compact(held, 1)})"
            if unread_rows:
                any_unread = True
                lines.append(f"In {name} these pages have not been read or "
                             "looked at by any tool or helper:")
                for r in unread_rows:
                    n_unread = len(parse_pages(r["not_read"]))
                    line = (f"- {r['label']}: {n_unread} of {r['pages']} not "
                            f"read - pages {r['not_read']} (PDF pages "
                            f"{r['not_read_pdf']})")
                    if r.get("read_as_text_but_no_text_layer"):
                        line += (f"; pages {r['read_as_text_but_no_text_layer']}"
                                 " were read as text but have no text layer, "
                                 "so they must be looked at")
                    else:
                        inv = self.docs[key]["inventory"]
                        scans = [p for p in parse_pages(r["not_read"])
                                 if not inv.pages[p].has_text]
                        if scans:
                            line += (f"; pages {compact(scans)} have no text "
                                     "layer, so they must be looked at")
                    lines.append(line)
        if not any_unread and not extra and (not explicit
                                             or states_coverage(answer)):
            return None
        head = (f"{GATE_PREFIX} Before your answer goes to the user, the app "
                "has checked it against its record of every page any tool or "
                "helper read or looked at in this conversation. The answer "
                "you had written is held back: the user will not see it.")
        if any_unread:
            body = ("\n".join(lines) + "\nRead or look at them now, or mark "
                    "the ones you leave out as skipped with a reason "
                    "(document_coverage), or say in your answer why they "
                    "were not needed.")
        else:
            body = "Every page this task covers has been read."
        statements = "\n".join(
            f"- {self.statement(k, g, self.held_pages(k, w))}"
            for k, (g, w) in armed.items())
        tail = ("State the coverage in your answer as counts, from this "
                "record:\n" + statements)
        parts = [head, body]
        if extra:
            parts.append("\n".join(extra))
        parts.append(tail)
        parts.append("This note comes once. Then write your whole answer to "
                     "the user's request, as they will read it: it replaces "
                     "the one held back, so it must stand on its own, with "
                     "the coverage in it.")
        return "\n".join(parts)

    def note_fired(self, turn_key: str, note: str) -> None:
        with self._lock:
            self.fired[turn_key] = {"seq": self.seq,
                                    "time": round(time.time(), 1),
                                    "note": note}
        self.save()

    def note_skipped(self, turn_key: str, why: str) -> None:
        """The gate stood down (too few steps left): kept in the record."""
        with self._lock:
            self.fired[turn_key] = {"seq": self.seq,
                                    "time": round(time.time(), 1),
                                    "skipped": why}
        self.save()


def ledger_path(folder: Optional[str]) -> Optional[str]:
    """Where a conversation's ledger lives: ``<folder>/coverage.json``."""
    return os.path.join(folder, FILE_NAME) if folder else None


__all__ = ["CoverageLedger", "Inventory", "PageInfo", "GROUPS",
           "GROUP_LABEL", "ROLE_GROUP", "TARGET_GROUPS", "DATA_GROUPS",
           "GATE_PREFIX", "FILE_NAME", "TEXT_TOOLS", "LOOK_TOOLS",
           "SWEEP_TOOLS", "GRID_TOOLS", "STRONG_CONFIDENCE",
           "INVENTORY_RULES", "classify_pages", "parse_pages", "compact",
           "pages_from_call", "states_coverage", "ledger_path"]
