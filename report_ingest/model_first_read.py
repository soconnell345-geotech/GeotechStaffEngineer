"""Model-first report ingest: the readers (BACKLOG RI-87, piece 7).

``module_work/REPORT_INGEST_PLAN.md`` section 8.2, steps 7 to 9. Under
``GEOTECH_INGEST_MODEL_FIRST=on`` each group of pages that
:func:`report_ingest.model_first_pages.group_pages` drew is read by ONE
call, and the model's answer is the record. Nothing on today's path imports
this module: the graph wiring (piece 8) does, and only under ``on``.

ONE CALL PER GROUP. The call carries the group's pages, as a PDF where the
engine can send one and as pictures with their text otherwise
(``GEOTECH_INGEST_MODEL_FIRST_PAGES``, :func:`~report_ingest.
model_first_pages.resolve_route`), then the stage's instruction
(:func:`~report_ingest.model_first.instruction`, with the sentence that maps
each PDF page or picture to the report's page), and offers the stage's
answer schema (:func:`~report_ingest.model_first.answer_schema`) as the
answer tool. No system prompt, no brief, no grid rows, no floor and no zoom
tool: the call the plain baseline measured (RI-86). Every word the model
reads comes from piece 4's instructions and schemas and piece 2's page
blocks; this module adds none.

A CALL THAT DOES NOT FIT. A reply that stopped at its output cap
(:attr:`~report_ingest.engine.Reply.cut_off`) gives no answer, and pages too
large for one request (:func:`~report_ingest.model_first_pages.pdf_problem`,
:func:`~report_ingest.model_first_pages.pictures_problem`) are never sent.
Either way the group is asked once more as two halves, cut at the sheet
boundary nearest its middle (for laboratory pages, where a description says
a new sheet begins, if one does in the middle half of the pages; else at
the middle page). A half that still does not fit, a single page that
does not, and a call that gives no structured answer each become a
``skipped`` entry naming the pages (``refused_by: "budget"``, which the
reconciler files as ``skipped``). Nothing is asked a third time.

A REQUEST OVER THE LIMIT (BACKLOG RI-121). Each reader takes
``max_request_tokens_in``, a per-request input-token limit (the graph's
``Budgets.max_request_tokens_in``); None, the default, changes nothing.
Set, every call's request is estimated before it is sent
(:func:`report_ingest.engine.estimate_tokens`): a picture by its own pixels
(:func:`~report_ingest.engine.picture_tokens_est`), a PDF page as its text
layer and one page picture
(:func:`~report_ingest.engine.pdf_tokens_est`), the words, and the answer
schema offered as its tool. A group over the limit is cut in two by
:func:`_halves` before anything is sent, and each half again while it is
still over, at no cost: the pieces are then read as groups of their own
(each asked once more as two halves if its reply is cut off). A log's
pieces are its own pages, so a piece never holds another hole's, and its
answers are joined into the one log as a long log's parts are. A single
page still over the limit is sent as it is. Each cut, and each page sent
over the limit, is one line in the result's ``warnings`` and one row in its
``over_limit`` (the graph makes each row a QA note), and each call in
``calls`` carries its ``estimated_tokens_in`` beside the tokens the reply
says it took, so the estimate can be checked against a run. With no limit
the result has no ``over_limit`` and is what it was.

BUILD. Each answer is turned into record objects by its stage's own
``_Builder`` with ``checks="flag"`` (:mod:`report_ingest.checks`): a value
that cannot be true stays out of its slot with an open QA entry, a doubtful
one stays at a lowered confidence. Before that, every box in the answer is
cleared: the model was shown no rows or cells to copy one from (plan section
8.4), so a box is placement's to give. Each answer is built over the pages
of its OWN call, so its page numbers are read the way it gave them
(:func:`~report_ingest.checks.pdf_local`), and the schema's added fields go
into the record on that answer's own build
(:func:`~report_ingest.model_first.apply_log_additions`,
:func:`~report_ingest.model_first.apply_lab_additions`), before any join;
the entries those return name the page as that answer's pages are read.

* **Logs**: the grid of planlens (``log_grid``, no call) gives each call's
  depth window and slack (``_depth_window``, ``_depth_slack``); a page with
  no ruler leaves the depths unchecked, with one note. The parts of a long
  log (groups sharing a key) and the halves of a call are joined into one
  log: the first answer's header, with any header slot it left empty taken
  from a later one and any slot a later answer gives otherwise listed as
  read; the remarks of each answer appended; the layers, samples, blow
  counts and water readings of each later answer appended; and one entry
  that names the join.
* **Lab**: one result per window; the tests of two halves are listed
  together.
* **Calc**: one result per printout. A printout longer than one call is
  read in the two windows of today's reader (``calc_calls``); the answers
  are put on the report's pages first, each against its own call's pages
  (an answer from a call of one page on that page, whatever it cites: the
  RI-74 rule), then folded by today's rule (``calc_reader._fold_readings``,
  two rows with no box one value only when they read the same value) and
  built once over the printout.
* **Narrative**: the groups are its consecutive parts. Each answer's
  citations are put on the report's pages against its own call's pages,
  then the answer is added to one builder, whose own rule merges the parts
  (the first answer kept, lists joined, every disagreement recorded; an
  answer mixing its call's pages with others gets one note); a narrative
  read in more than one call then gets today's one summary call
  (``narrative_reader._summarise``). The builder gets the caption count of
  the code (``count_captions``, no call), which under ``flag`` is a QA note
  and never the record's value, and no logs. The quote check runs last,
  as today.

PLACE. ``place`` is a callable, default None (no placement, and no box on
any value). Given, it is called once per answer's record, after the build
and the additions, as ``place(stage, record, pages, doc, rulers=...)``:
``record`` is the log's ``Investigation``, the window's list of
``LabTest``, the printout's ``Calculation`` or the narrative's citations
(``field -> [Citation]``); ``pages`` are the call's pages; ``rulers`` the
log's depth rulers (logs only). It changes the record's provenance in place
and returns what it found (or None), which the result keeps under
``placement``.

THE RESULT. Each stage's result is a dict in the shape of today's reader's
``to_dict()`` (``LogReadResult``, ``LabReadResult``, ``CalcReadResult``,
``NarrativeReadResult``), with no floor: ``disagreements``, ``kept`` and
``added`` are empty, and the floor's slots are None or empty. The model's
record is also its ``model_*`` slot, so a scorecard's model column reads it.
Added keys: ``key`` (the group's key, for the graph's ``items/mf-<key>.json``),
``route``, ``prompt_version`` (:data:`~report_ingest.model_first.
READ_PROMPT_VERSION`), ``calls`` (each call's pages and how it ended),
``placement`` (what ``place`` returned, one per answer) and ``error``
(empty, or why nothing was read: the record slot is then None, or an empty
list for lab tests, and the ``skipped`` entries say which pages).

An error a call raises (a refused request, the call ceiling) is not caught:
the caller decides, as it does for today's readers.

THE MODELS are the caller's: each reader takes the engine of its stage.
:data:`STAGE_MODELS` holds the plan's defaults for the graph wiring and the
measurement plumbing (pieces 8 and 9) to build those engines from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from pydantic import BaseModel

from report_ingest.checks import (IMPOSSIBLE, LISTED, SURFACE_TOL_M, _noted,
                                  _on_call_pages, _span, as_read,
                                  ground_investigation, pdf_local)
from report_ingest.engine import (estimate_tokens, pdf_block,
                                  picture_tokens_est, schema_tokens_est,
                                  text_block, user)
from report_ingest.model_first import (READ_PROMPT_VERSION, _put_field,
                                       answer_schema, apply_lab_additions,
                                       apply_log_additions, instruction)
from report_ingest.model_first_pages import (PageGroup, calc_calls, cut_pdf,
                                             lab_starts, pdf_problem,
                                             picture_blocks, pictures_problem,
                                             resolve_route)

__all__ = ["STAGE_MODELS", "Place", "read_logs", "read_lab", "read_calcs",
           "read_narrative"]

#: The model each stage reads on by default (plan section 8.1): laboratory
#: sheets and the narrative on Opus 5.5, logs and calculations on Sonnet 5.5
#: (RI-78; the narrative with every question in one call, owner
#: 2026-10-25). For the graph wiring and the measurement plumbing (pieces 8
#: and 9) to build each stage's engine from; the readers here take the
#: engine they are handed.
STAGE_MODELS: Dict[str, str] = {
    "logs": "claude-sonnet-5-5",
    "lab": "claude-opus-5-5",
    "calc": "claude-sonnet-5-5",
    "narrative": "claude-opus-5-5",
}

#: ``place(stage, record, pages, doc, rulers=None)``: puts one answer's
#: record on its pages (piece 5) and returns what it found, or None.
Place = Callable[..., Optional[Dict[str, Any]]]

#: Where a PDF is cut from: a path, or the report's bytes.
Source = Union[str, Path, bytes, None]


# ---------------------------------------------------------------------------
# the calls
# ---------------------------------------------------------------------------

@dataclass
class _Spent:
    """What one result's calls cost, in today's readers' ``cost`` shape, and
    how each call ended."""

    cost: Dict[str, Any] = field(default_factory=lambda: {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0})
    calls: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    #: Each request estimated over the request limit (RI-121): its pages,
    #: the estimate, the limit, and the two groups it was cut into (none:
    #: one page, sent as it is).
    over_limit: List[Dict[str, Any]] = field(default_factory=list)

    def charge(self, reply: Any, pages: Sequence[int],
               estimated: Optional[int] = None) -> None:
        self.cost["calls"] += 1
        self.cost["input_tokens"] += reply.usage.input_tokens
        self.cost["output_tokens"] += reply.usage.output_tokens
        self.cost["cache_read_tokens"] += reply.usage.cache_read_tokens
        self.cost["seconds"] += reply.seconds
        self.cost["dollars"] += reply.usage.dollars(reply.model)
        self.model = reply.model or self.model
        row = {"pages": list(pages), "stop_reason": reply.stop_reason,
               "cut_off": bool(reply.cut_off),
               "answered": reply.parsed is not None and not reply.cut_off}
        if estimated is not None:
            # Only under a request limit (RI-121): the estimate the call was
            # sized by, beside what the reply says the request took.
            row.update(estimated_tokens_in=int(estimated),
                       input_tokens=int(reply.usage.input_tokens))
        self.calls.append(row)

    def rounded(self) -> Dict[str, Any]:
        out = dict(self.cost)
        out["seconds"] = round(out["seconds"], 2)
        out["dollars"] = round(out["dollars"], 5)
        return out


def _skipped(stage: str, pages: Sequence[int], why: str) -> Dict[str, Any]:
    """The entry for pages no answer came back for: ``skipped`` in QA."""
    pages = [int(p) for p in pages]
    return {"what": f"{stage} {_span(pages)}", "page": pages[0],
            "pages": pages,
            "why": f"{why}; {_span(pages)} were not read" if len(pages) > 1
            else f"{why}; {_span(pages)} was not read",
            "refused_by": "budget"}


def _halves(pages: Sequence[int], starts: Sequence[int] = ()
            ) -> List[List[int]]:
    """``pages`` in two, cut at the sheet boundary nearest the middle: the
    page in ``starts`` (where a new sheet begins) nearest it, if one is in
    the middle half of the pages, else the middle page. A start next to
    either end would leave one half nearly the size of the call that did
    not fit."""
    pages = [int(p) for p in pages]
    middle = len(pages) / 2.0
    wanted = {int(p) for p in starts}
    cuts = [n for n, page in enumerate(pages) if n and page in wanted
            and len(pages) / 4.0 <= n <= 3.0 * len(pages) / 4.0]
    cut = (min(cuts, key=lambda n: (abs(n - middle), n)) if cuts
           else max(1, len(pages) // 2))
    return [pages[:cut], pages[cut:]]


class _Caller:
    """One stage's calls on one engine: what each call carries and what
    comes back."""

    def __init__(self, stage: str, doc: Any, engine: Any, report_id: str,
                 route: Optional[str], pdf: Source,
                 limit: Optional[int] = None) -> None:
        self.stage = stage
        self.doc = doc
        self.engine = engine
        self.report_id = str(report_id or "")
        self.route = resolve_route(
            route, bool(getattr(engine, "accepts_pdf", False)))
        self.schema = answer_schema(stage)
        self._pdf = pdf
        #: The per-request input-token limit (RI-121), or None.
        self.limit: Optional[int] = int(limit) if limit else None
        self._schema_tokens: Optional[int] = None

    def size(self, blocks: Sequence[Dict[str, Any]]) -> int:
        """One call's request, estimated in input tokens: its blocks (each
        picture by its own size, a PDF page as its text and one picture)
        and the answer schema it offers as its tool."""
        if self._schema_tokens is None:
            self._schema_tokens = schema_tokens_est(self.schema)
        return self._schema_tokens + estimate_tokens(
            [user(*blocks)], picture=picture_tokens_est)

    def _source(self) -> Union[str, Path, bytes]:
        """The report to cut a call's PDF from: what the caller gave, else
        the open document's own bytes, taken once."""
        if self._pdf is None:
            self._pdf = self.doc.tobytes()
        return self._pdf

    def blocks(self, pages: Sequence[int]
               ) -> Tuple[List[Dict[str, Any]], str]:
        """The user turn's blocks for one call, and why they cannot be sent
        ("" when they can): the pages first, then the instruction."""
        said = text_block(instruction(self.stage, self.report_id, pages,
                                      route=self.route))
        if self.route == "pdf":
            # A sheet scanned sideways is sent rotated upright (RI-149).
            from report_ingest.turned import turn_of
            turns = {p: turn_of(self.doc, p).quarter for p in pages}
            data, n_pages, _sanitized = cut_pdf(
                self._source(), pages,
                turns={p: q for p, q in turns.items() if q})
            return [pdf_block(data), said], pdf_problem(n_pages, len(data))
        shown = picture_blocks(self.doc, pages,
                               model=getattr(self.engine, "name", None))
        blocks = shown + [said]
        return blocks, pictures_problem(blocks)

    def read(self, pages: Sequence[int], spent: _Spent,
             starts: Sequence[int] = (), split: bool = True
             ) -> Tuple[List[Tuple[List[int], Any]], List[Dict[str, Any]]]:
        """``([(the call's pages, its answer), ...], skipped entries)``.

        One call over ``pages``. A reply cut off at its output cap, or pages
        too large to send, are asked once more as two halves (:func:`_halves`
        at ``starts``); a half that still does not fit, and a call with no
        structured answer, are skipped entries.

        Under a request limit (:attr:`limit`, RI-121) a request estimated
        over it is cut in two before it is sent, each half read as
        ``pages`` is (``split`` as given), and a single page over it is sent
        with a warning."""
        pages = [int(p) for p in pages]
        blocks, problem = self.blocks(pages)
        size = self.size(blocks) if self.limit is not None else None
        over = size is not None and size > self.limit
        if over and len(pages) > 1:
            halves = _halves(pages, starts)
            spent.over_limit.append({
                "pages": pages, "estimated_tokens_in": size,
                "limit": self.limit, "cut_into": halves})
            spent.warnings.append(
                f"{_span(pages)}: the request was estimated at {size:,} "
                f"input tokens, over the limit of {self.limit:,} a request; "
                f"split before it was sent, into {_span(halves[0])} and "
                f"{_span(halves[1])}")
            pieces: List[Tuple[List[int], Any]] = []
            unread: List[Dict[str, Any]] = []
            for half in halves:
                answers, rows = self.read(half, spent, starts, split=split)
                pieces.extend(answers)
                unread.extend(rows)
            return pieces, unread
        if problem:
            why = f"too large for one call ({problem})"
        else:
            if over:
                spent.over_limit.append({
                    "pages": pages, "estimated_tokens_in": size,
                    "limit": self.limit, "cut_into": []})
                spent.warnings.append(
                    f"{_span(pages)}: the request was estimated at {size:,} "
                    f"input tokens, over the limit of {self.limit:,} a "
                    f"request, and one page cannot be split; it was sent as "
                    f"it is")
            reply = self.engine.complete([user(*blocks)],
                                         output_format=self.schema)
            spent.charge(reply, pages, size)
            if not reply.cut_off and reply.parsed is not None:
                return [(pages, _without_boxes(reply.parsed))], []
            if not reply.cut_off:
                notes = "; ".join(getattr(reply, "notes", None) or [])
                return [], [_skipped(self.stage, pages, (
                    f"the call gave no structured answer (stop_reason "
                    f"{reply.stop_reason!r}"
                    + (f"; {notes[:300]}" if notes else "") + ")"))]
            why = (f"the reply stopped at its output cap (stop_reason "
                   f"{reply.stop_reason!r})")
        if split and len(pages) > 1:
            halves = _halves(pages, starts)
            spent.warnings.append(
                f"{_span(pages)}: {why}; asked again as two halves, "
                f"{_span(halves[0])} and {_span(halves[1])}")
            got: List[Tuple[List[int], Any]] = []
            skipped: List[Dict[str, Any]] = []
            for half in halves:
                answers, rows = self.read(half, spent, starts, split=False)
                got.extend(answers)
                skipped.extend(rows)
            return got, skipped
        if not split:
            why += (" when these pages were asked as one half of pages that "
                    "did not fit in one call")
        elif len(pages) == 1:
            why += ", and one page cannot be split"
        return [], [_skipped(self.stage, pages, why)]


def _without_boxes(reading: Any) -> Any:
    """A copy of ``reading`` with every box cleared: the model was shown no
    rows or cells to copy a box from (plan section 8.4)."""
    out = reading.model_copy(deep=True)

    def walk(node: Any) -> None:
        if isinstance(node, BaseModel):
            fields = type(node).model_fields
            if "bbox" in fields and getattr(node, "bbox", None) is not None:
                node.bbox = None
            for name in fields:
                walk(getattr(node, name, None))
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(out)
    return out


def _result(stage: str, key: str, caller: _Caller, spent: _Spent,
            pages: Sequence[int], unresolved: List[Dict[str, Any]],
            placement: List[Any], warnings: Sequence[str] = (),
            error: str = "") -> Dict[str, Any]:
    """The keys every stage's result adds to today's shape; under a request
    limit (RI-121) also ``over_limit``, each request estimated over it."""
    out = {"key": key, "route": caller.route,
           "prompt_version": READ_PROMPT_VERSION,
           "calls": list(spent.calls),
           "placement": [p for p in placement if p is not None],
           "error": error, "pages": [int(p) for p in pages],
           "unresolved": unresolved, "cost": spent.rounded(),
           "model_calls": int(spent.cost["calls"]),
           "model": spent.model or str(getattr(caller.engine, "name", "")),
           "warnings": list(warnings) + list(spent.warnings),
           "disagreements": [], "kept": [], "added": []}
    if caller.limit is not None:
        out["over_limit"] = [dict(row) for row in spent.over_limit]
    return out


def _nothing_read(skipped: Sequence[Dict[str, Any]]) -> str:
    return ("no answer came back for any page of this group"
            + (f" ({skipped[0]['why']})" if skipped else ""))


def _entries_on_pages(rows: List[Dict[str, Any]], builder: Any
                      ) -> List[Dict[str, Any]]:
    """The entries the schema's additions return, each page read the way
    its builder read the answer's pages (its ``PageNumbers`` under flag)."""
    numbers = getattr(builder, "_numbers", None)
    if numbers is not None:
        for row in rows:
            if row.get("page") is not None:
                row["page"] = numbers.page(row["page"])[0]
    return rows


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------

#: The parts of a log that a later answer's are appended to.
_LOG_LISTS: Tuple[str, ...] = ("layers", "samples", "spt", "water", "prov")


def _by_key(groups: Sequence[PageGroup]) -> List[List[PageGroup]]:
    """The groups by key, in order of first appearance; a group with no key
    is its own."""
    out: Dict[str, List[PageGroup]] = {}
    for n, group in enumerate(groups):
        out.setdefault(group.key or f"#{n}", []).append(group)
    return [sorted(parts, key=lambda g: (g.part or (1, 1))[0])
            for parts in out.values()]


#: Header slots the join does not set side by side: the lists it appends,
#: the header fields and remarks it adds to, the pages it joins, the depth
#: unit it compares on its own, and each sheet's own "page 1 of 3".
_JOINED_APART: Tuple[str, ...] = _LOG_LISTS + (
    "fields", "remarks", "pages", "depth_unit", "units_known", "sheet")
#: Header slots whose own slots are joined one by one.
_JOINED_BY_SLOT: Tuple[str, ...] = ("drilling",)


def _plainly(value: Any) -> Any:
    """A slot's value to compare: a record object without its provenance."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude=(
            {"prov"} if "prov" in type(value).model_fields else None))
    return value


def _join_slots(out: Any, later: Any, skip: Sequence[str], what: str,
                pages: Sequence[int], rows: List[Dict[str, Any]],
                prefix: str = "") -> None:
    """``later``'s header slots into ``out``: one ``out`` left at its
    default takes ``later``'s (the kind excepted), and one that differs is
    kept as ``out`` has it, with ``later``'s value listed as read."""
    for slot, info in type(out).model_fields.items():
        if slot in skip:
            continue
        ours, theirs = getattr(out, slot), getattr(later, slot)
        if slot in _JOINED_BY_SLOT:
            _join_slots(ours, theirs, (), what, pages, rows, f"{slot} ")
            continue
        default = _plainly(info.get_default(call_default_factory=True))
        if _plainly(theirs) in (default, _plainly(ours)):
            continue
        if _plainly(ours) == default and slot != "kind":
            setattr(out, slot, theirs)
            continue
        rows.append({
            "what": f"{what}: {prefix}{slot}", "page": int(pages[0]),
            "read": as_read(theirs),
            "why": f"the answer for {_span(pages)} gives another "
                   f"{prefix}{slot} than the first call's; the record keeps "
                   f"the first call's, and this one is {LISTED}",
            "refused_by": "python"})


def _join_logs(built: Sequence[Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    """One log from the records of its answers, in page order.

    The first answer's header, kind and depth unit; a header slot it left at
    its default takes the first later answer's that has one, and a slot a
    later answer gives otherwise is kept as the first gives it, with the
    later value listed as read (one entry each). Each answer's remarks are
    appended; its header fields added (one printed under a label already
    there with another value is kept beside it, numbered); its layers,
    samples, blow counts, water readings and provenance appended; the pages
    joined. Returns the log and the entries: those above, one when two
    answers give depths in different units, and one naming the join."""
    out = built[0].model_copy(deep=True)
    rows: List[Dict[str, Any]] = []
    what = f"log {out.investigation_id}".strip()
    for later in built[1:]:
        later = later.model_copy(deep=True)
        if later.depth_unit and out.depth_unit \
                and later.depth_unit != out.depth_unit:
            rows.append({
                "what": f"{what}: depth unit", "page": later.pages[0],
                "why": f"the answer for {_span(later.pages)} gives depths "
                       f"in {later.depth_unit}, the first in "
                       f"{out.depth_unit}; its depths are joined as read"})
        elif later.depth_unit and not out.depth_unit:
            out.depth_unit = later.depth_unit
            out.units_known = later.units_known
        for name in _LOG_LISTS:
            getattr(out, name).extend(getattr(later, name))
        fields = dict(out.fields)
        for label, value in later.fields.items():
            _put_field(fields, label, value)
        out.fields = fields
        remarks = later.remarks.strip()
        if remarks and remarks not in out.remarks:
            out.remarks = "; ".join(r for r in (out.remarks.strip(), remarks)
                                    if r)
        _join_slots(out, later, _JOINED_APART, what, later.pages, rows)
        out.pages = sorted(set(out.pages) | set(later.pages))
    parts = "; ".join(_span(b.pages) for b in built)
    rows.append({
        "what": what, "page": built[1].pages[0],
        "why": f"read in {len(built)} calls ({parts}) and joined: the "
               f"header is the first call's, and each later call's layers, "
               f"samples, blow counts and water readings are appended"})
    return out, rows


def read_logs(doc: Any, groups: Sequence[PageGroup], engine: Any, *,
              report_id: str = "", route: Optional[str] = None,
              pdf: Source = None, place: Optional[Place] = None,
              max_request_tokens_in: Optional[int] = None
              ) -> List[Dict[str, Any]]:
    """Every log of ``groups`` (``group_pages(...)["logs"]``), one call per
    group: a list of results in ``LogReadResult.to_dict()``'s shape, one per
    log. The groups of one log (one ``key``, the parts of a log longer than
    a call) give one result.

    ``engine`` is the logs' engine; ``route`` the page route (None: the
    variable); ``pdf`` the report to cut each call's PDF from (a path or its
    bytes; None: ``doc.tobytes()``, taken once); ``place`` and
    ``max_request_tokens_in`` (a group whose request is estimated over it is
    cut by its pages before it is sent) as the module says. A log no answer
    came back for has ``investigation`` None and an ``error``."""
    from report_ingest.log_reader import (LogReadResult, _Builder,
                                          _depth_slack, _depth_window)
    from report_ingest.visual_scales import log_grid_for

    caller = _Caller("logs", doc, engine, report_id, route, pdf,
                     max_request_tokens_in)
    out: List[Dict[str, Any]] = []
    for parts in _by_key(groups):
        spent = _Spent()
        answers: List[Tuple[List[int], Any]] = []
        skipped: List[Dict[str, Any]] = []
        for group in parts:
            got, rows = caller.read(group.pages, spent)
            answers.extend(got)
            skipped.extend(rows)
        pages = sorted({p for g in parts for p in g.pages})
        notes = [n for g in parts for n in g.notes]
        key = parts[0].key or f"log-{pages[0]}"
        built: List[Any] = []
        unresolved: List[Dict[str, Any]] = []
        changes: List[Dict[str, Any]] = []
        grid_warnings: List[str] = []
        placement: List[Any] = []
        for call_pages, reading in sorted(answers, key=lambda a: a[0][0]):
            # Planlens' grid, no call (visual scales are given no engine):
            # each call's depth window and its slack.
            grid, _info = log_grid_for(doc, call_pages)
            builder = _Builder(
                reading, _depth_window(doc, grid, call_pages), call_pages,
                report_id, no_ruler=not bool(getattr(grid, "rulers", None)),
                slack=_depth_slack(grid, call_pages), checks="flag")
            investigation, extra = apply_log_additions(reading,
                                                       builder.build())
            extra = _entries_on_pages(extra, builder)
            if place is not None:
                placement.append(place("logs", investigation, call_pages, doc,
                                       rulers=getattr(grid, "rulers", None)))
            built.append(investigation)
            unresolved.extend(builder.unresolved + extra)
            changes.extend(builder.changes)
            grid_warnings.extend(getattr(grid, "warnings", None) or [])
        if not built:
            blob = _result("logs", key, caller, spent, pages, skipped,
                           placement, notes, _nothing_read(skipped))
            blob.update(investigation=None, model_investigation=None,
                        floor=None, changes=[], reconciled=0,
                        follow_up=False, template=None)
            out.append(blob)
            continue
        investigation = built[0]
        if len(built) > 1:
            investigation, joined = _join_logs(built)
            unresolved.extend(joined)
        # The log's last gate (RI-123): each builder checked its own depths
        # against the ground surface, so this finds what the additions or
        # the join brought, which is nothing today.
        above, near = ground_investigation(investigation, refused=IMPOSSIBLE)
        unresolved.extend(above)
        if near:
            unresolved.append({
                "what": "depths at the ground surface", "page": pages[0],
                "why": f"{near} joined depth(s) above the ground surface by "
                       f"no more than {SURFACE_TOL_M:g} m were recorded as 0",
                "flag": "surface_snap"})
        result = LogReadResult(
            investigation=investigation, changes=changes,
            unresolved=unresolved + skipped, pages=pages,
            model_investigation=investigation,
            warnings=grid_warnings)
        blob = result.to_dict()
        blob.update(_result("logs", key, caller, spent, pages,
                            unresolved + skipped, placement,
                            notes + grid_warnings))
        out.append(blob)
    return out


# ---------------------------------------------------------------------------
# laboratory sheets
# ---------------------------------------------------------------------------

def read_lab(doc: Any, groups: Sequence[PageGroup], engine: Any, *,
             report_id: str = "", route: Optional[str] = None,
             pdf: Source = None,
             descriptions: Optional[Mapping[int, Mapping[str, Any]]] = None,
             place: Optional[Place] = None,
             max_request_tokens_in: Optional[int] = None
             ) -> List[Dict[str, Any]]:
    """Every laboratory window of ``groups`` (``group_pages(...)["lab"]``),
    one call each: a list of results in ``LabReadResult.to_dict()``'s
    shape, one per window.

    ``descriptions`` (the page descriptions the groups were drawn from) say
    where a new sheet begins, which is where a window whose reply did not
    fit, or whose request is over ``max_request_tokens_in``, is cut in two;
    without them it is cut at its middle page. The rest as
    :func:`read_logs`. A window no answer came back for has no tests and an
    ``error``."""
    from report_ingest.lab_reader import LabReadResult, _Builder

    caller = _Caller("lab", doc, engine, report_id, route, pdf,
                     max_request_tokens_in)
    out: List[Dict[str, Any]] = []
    for group in groups:
        spent = _Spent()
        starts = (lab_starts(group.pages, descriptions)
                  if descriptions else [])
        answers, skipped = caller.read(group.pages, spent, starts)
        tests: List[Any] = []
        unresolved: List[Dict[str, Any]] = []
        changes: List[Dict[str, Any]] = []
        placement: List[Any] = []
        for call_pages, reading in answers:
            builder = _Builder(reading, call_pages, report_id, [],
                               checks="flag")
            built, extra = apply_lab_additions(reading, builder.build())
            extra = _entries_on_pages(extra, builder)
            if place is not None:
                placement.append(place("lab", built, call_pages, doc))
            tests.extend(built)
            unresolved.extend(builder.unresolved + extra)
            changes.extend(builder.changes)
        key = group.key or f"lab-{group.pages[0]}"
        if group.part:
            key = f"{key}-{group.part[0]}"
        result = LabReadResult(
            tests=tests, changes=changes, unresolved=unresolved + skipped,
            pages=list(group.pages), model_tests=tests,
            windows=([list(a[0]) for a in answers] if len(answers) > 1
                     else []))
        blob = result.to_dict()
        blob.update(_result("lab", key, caller, spent, group.pages,
                            unresolved + skipped, placement, group.notes,
                            "" if answers else _nothing_read(skipped)))
        out.append(blob)
    return out


# ---------------------------------------------------------------------------
# calculation printouts
# ---------------------------------------------------------------------------

def _calc_on_report_pages(reading: Any, pages: Sequence[int]
                          ) -> Tuple[Any, List[Dict[str, Any]], str]:
    """A calculation answer with every page it cites read against the pages
    of its own call (:func:`_on_call_pages`, with the one-page rule), so two
    answers can be folded into one: a copy, the entries and the warning."""
    out = reading.model_copy(deep=True)
    holders = ([v.prov for v in (*out.inputs, *out.results)]
               + list(out.unsettled))
    rows, said = _on_call_pages(holders, pages, one_page=True)
    return out, rows, said


def _narrative_on_report_pages(reading: Any, pages: Sequence[int]
                               ) -> Tuple[Any, List[Dict[str, Any]], str]:
    """A narrative answer with the pages its citations and unsettled rows
    cite read against the pages of its own call (:func:`_on_call_pages`): a
    copy, the entries and the warning."""
    out = reading.model_copy(deep=True)
    rows, said = _on_call_pages([*out.citations, *out.unsettled], pages)
    return out, rows, said


def read_calcs(doc: Any, groups: Sequence[PageGroup], engine: Any, *,
               report_id: str = "", route: Optional[str] = None,
               pdf: Source = None, place: Optional[Place] = None,
               max_request_tokens_in: Optional[int] = None
               ) -> List[Dict[str, Any]]:
    """Every calculation printout of ``groups`` (``group_pages(...)
    ["calc"]``): a list of results in ``CalcReadResult.to_dict()``'s shape,
    one per printout.

    A printout of up to ``calc_reader.MAX_CALC_PAGES`` pages is one call; a
    longer one two (``calc_calls``), whose answers are folded into one
    reading by today's rule before it is built over the whole printout. The
    rest as :func:`read_logs`. A printout no answer came back for has
    ``calculation`` None and an ``error``."""
    from report_ingest.calc_reader import (CalcReadResult, _Builder,
                                           _fold_readings)

    caller = _Caller("calc", doc, engine, report_id, route, pdf,
                     max_request_tokens_in)
    out: List[Dict[str, Any]] = []
    for group in groups:
        spent = _Spent()
        answers: List[Tuple[List[int], Any]] = []
        skipped: List[Dict[str, Any]] = []
        for call_pages in calc_calls(group):
            got, rows = caller.read(call_pages, spent)
            answers.extend(got)
            skipped.extend(rows)
        pages = list(group.pages)
        key = group.key or f"calc-{pages[0]}"
        if not answers:
            blob = _result("calc", key, caller, spent, pages, skipped, [],
                           group.notes, _nothing_read(skipped))
            blob.update(calculation=None, model_calculation=None,
                        floor_calculation=None, floor_values=0, changes=[],
                        tool_calls=0, reconciled=0)
            out.append(blob)
            continue
        unresolved: List[Dict[str, Any]] = []
        warnings = list(group.notes)
        # One answer over the whole printout: the builder reads its pages.
        # Otherwise each answer is read against its own call's pages first,
        # and the builder is told so.
        mapped = not (len(answers) == 1 and answers[0][0] == pages)
        if not mapped:
            reading = answers[0][1]
        else:
            readings = []
            for call_pages, answer in answers:
                answer, rows, said = _calc_on_report_pages(answer,
                                                           call_pages)
                readings.append(answer)
                unresolved.extend(rows)
                warnings.extend([said] if said else [])
            reading = readings[0]
            seen = set(answers[0][0])
            for (call_pages, _a), later in zip(answers[1:], readings[1:]):
                reading = _fold_readings(
                    reading, later, fresh=True, same_value=True,
                    later_pages=[p for p in call_pages if p not in seen])
                seen.update(call_pages)
        builder = _Builder(reading, pages, report_id, [], [], checks="flag",
                           mapped=mapped)
        calculation = builder.build()
        placement = ([place("calc", calculation, pages, doc)]
                     if place is not None else [])
        unresolved.extend(builder.unresolved)
        result = CalcReadResult(
            calculation=calculation, changes=builder.changes,
            unresolved=unresolved + skipped, pages=pages,
            model_calculation=calculation)
        blob = result.to_dict()
        blob.update(_result("calc", key, caller, spent, pages,
                            unresolved + skipped, placement,
                            warnings + builder.warnings))
        out.append(blob)
    return out


# ---------------------------------------------------------------------------
# the narrative
# ---------------------------------------------------------------------------

def read_narrative(doc: Any, groups: Sequence[PageGroup], engine: Any, *,
                   report_id: str = "", route: Optional[str] = None,
                   pdf: Source = None, outline: Any = None,
                   body_pages: Optional[Sequence[int]] = None,
                   place: Optional[Place] = None,
                   max_request_tokens_in: Optional[int] = None
                   ) -> Dict[str, Any]:
    """The narrative (``group_pages(...)["narrative"]``: its consecutive
    parts), one call per part: one result in
    ``NarrativeReadResult.to_dict()``'s shape.

    ``body_pages`` are the pages the caption count runs over (the main
    body, as the graph gives today's reader; default: the narrative's own
    pages); ``outline`` gives the summary call its section headings. The
    rest as :func:`read_logs`. A narrative no answer came back for has
    ``general``, ``natural_hazards`` and ``facts`` None and an ``error``."""
    from report_ingest.narrative_reader import (NarrativeReadResult,
                                                _apply_summaries, _Builder,
                                                _headings_text, _summarise,
                                                check_quotes, count_captions)

    groups = [g for g in groups if g.pages]
    if not groups:
        raise ValueError("read_narrative needs at least one group of pages")
    caller = _Caller("narrative", doc, engine, report_id, route, pdf,
                     max_request_tokens_in)
    spent = _Spent()
    pages = sorted({int(p) for g in groups for p in g.pages})
    notes = [n for g in groups for n in g.notes]
    builder = _Builder(pages, checks="flag")
    skipped: List[Dict[str, Any]] = []
    answered = 0
    for group in groups:
        got, rows = caller.read(group.pages, spent)
        skipped.extend(rows)
        for call_pages, reading in got:
            answered += 1
            reading, mixed, said = _narrative_on_report_pages(reading,
                                                              call_pages)
            builder.unresolved.extend(mixed)
            spent.warnings.extend([said] if said else [])
            builder.add(reading, answered)
    if not answered:
        blob = _result("narrative", "narrative", caller, spent, pages,
                       skipped, [], notes, _nothing_read(skipped))
        blob.update(general=None, natural_hazards=None, facts=None,
                    answered=[], n_answered=0, confidence={})
        return blob
    counted = count_captions(doc, list(body_pages) if body_pages else pages)
    general, hazards, facts = builder.build(pages, counted, None)
    if answered > 1:
        # Read in parts: one summary call over the merged answers, as
        # today's reader makes it.
        reply = _summarise(general, hazards, _headings_text(outline, pages),
                           engine)
        spent.charge(reply, [])
        if reply.parsed is not None and not reply.cut_off:
            _apply_summaries(reply.parsed, general, hazards, builder)
        else:
            spent.warnings.append("the summary call returned no structured "
                                  "answer; the per-part summaries stand")
    confidence = check_quotes(doc, builder.citations,
                              general.answered() + hazards.answered(),
                              builder)
    placement = ([place("narrative", builder.citations, pages, doc)]
                 if place is not None else [])
    builder.unresolved.extend(skipped)
    facts.unresolved = [dict(u) for u in builder.unresolved]
    result = NarrativeReadResult(
        general=general, natural_hazards=hazards, facts=facts,
        citations=dict(builder.citations),
        unresolved=[dict(u) for u in builder.unresolved],
        confidence=confidence, pages=pages)
    blob = result.to_dict()
    blob.update(_result("narrative", "narrative", caller, spent, pages,
                        [dict(u) for u in builder.unresolved], placement,
                        notes))
    return blob
