"""The whole ingest, end to end: one PDF in, one record and its exports out.

A DETERMINISTIC LOOP, NOT A PLANNING AGENT. The order below is Python's, and
the models are called one bounded thing at a time: one call over the whole
document to say what it is, one reviewing loop over the page labels, then one
reader per log, one per laboratory sheet and one for the narrative. Nothing
decides at runtime how many calls to make.

That is a deliberate choice and the reasons are the plan's. It is budgetable
-- a 150-page report is about a hundred calls and you can say so before it
runs. It is testable offline: every reader takes an engine, so the whole graph
runs against scripted replies with no network. It is restartable: each item
writes its own file under ``out_dir/items/`` as it finishes, and a second run
picks up where the first stopped. And it cannot forget an appendix, which a
model deciding its own fan-out over 455 pages eventually will.

THE WORKFLOW TRIAGE CHOSE DECIDES WHAT RUNS.

``standard``        everything below.
``appendix_only``   no narrative reader: there is no narrative to read, and a
                    reader pointed at an appendix would answer the owner's
                    questions from a boring log.
``multi_document``  everything, plus a QA entry naming the parts, because the
                    answers will mix two documents' facts until somebody
                    splits them.
``partial``         everything, plus a QA entry: what is missing is missing.
``scanned``         everything the pages support. The log and laboratory
                    readers look at the page image and work; the narrative
                    reader is TEXT ONLY, so on a scanned narrative with no
                    Azure Document Intelligence result it is skipped with a QA
                    entry rather than run against nothing.
``needs_person``    stops after triage, with the profile and a QA entry. A
                    file nobody can identify is not improved by reading it
                    wrong for twenty minutes.

THE PAGE LABELS ARE A VOTE, AND THE EXPENSIVE LOOK GOES WHERE THE VOTERS
SPLIT. Everything downstream hangs on what each page IS, and until this
train one voter decided it -- planlens' rules -- while an agent loop costing
about $0.45 a report then checked every page. The corpus run of 2026-09-20
says both halves of that were wrong: the rules and a cheap vision pass are
COMPLEMENTARY by label class (the rules own the structural labels, vision
owns the visual ones), and the review breaks nearly as many labels as it
fixes on the reports its prompt was tuned against. So now three cheap voters
label every page -- the rules, one vision pass over the pages as pictures at
about $0.05 a report, and the printed FORM where a fingerprint file is in
force -- :mod:`report_ingest.label_vote` combines them under ``label_policy``,
and ``review_mode="disagreements"`` sends the review ONLY the pages they
split on, with their neighbours for context and a budget that scales with
that count. A split the review does not settle becomes a
``QAEntry(kind="label_disagreement")``; every page's label in the record
carries its confidence, its voters and whether they agreed.

A REPORT BOUND INSIDE THIS ONE IS ITS OWN RECORD. An earlier firm's whole
investigation reproduced as an appendix, a bridging report bound into the
design-build report that answers it: 19 of the corpus's 38 reports have at
least one. Its pages stay labelled ``appended_report`` in THIS record and are
listed on ``ReportRecord.bound_documents``; the pages themselves go round
this same loop again, into their own record under ``out_dir/bound/<id>/``
with ``parent`` set. Nothing crosses the boundary -- a boring in the child is
the child's, the parent's narrative reader is TOLD the ranges so it answers
``previousInvestigationCount`` with them, and the parent's counts and
dictionaries come from the parent's own logs. ``ingest_bound=False`` turns it
off and the pages are listed and skipped, as they were before. See
:mod:`report_ingest.bound` for how the boundary is decided.

THE CALCULATIONS ARE READ. A quarter of the hand-labelled pages of this
corpus are calculation printouts and until this train every one of them was
a QA entry saying the pages existed and had been skipped. Each run of
``calculation`` pages is now one work item for
:func:`report_ingest.calc_reader.read_calculation`, which returns one
:class:`~report_ingest.model.Calculation` -- what it works out, what printed
it, what it was given and what it concluded -- onto
``ReportRecord.calculations``.

WHAT IS STILL NOT READ. ``figures``, ``photos``, ``plan``, ``profile`` and
``front_matter`` items: the narrative reader is shown those pages, and
nothing reads them as items of their own.

MODEL-FIRST INGEST IS A SWITCH, OFF BY DEFAULT (BACKLOG RI-87,
``module_work/REPORT_INGEST_PLAN.md`` section 8).
``GEOTECH_INGEST_MODEL_FIRST`` (or ``ingest_report(model_first=...)``) is
``off``, and the graph is the one described above, byte for byte; no
model-first module is imported. ``on``
reads each section's pages in one call and makes the model's answer the
record: every page is described and the report labelled by sections (the
range voter, RI-73), and those labels replace the vision voter and the label
review, which do not run; the logs, laboratory sheets, printouts and
narrative are grouped from the labels and the descriptions and read by
:mod:`report_ingest.model_first_read`, with no floor; each value is placed
on its page by the words printed there (:mod:`report_ingest.locate`). A
bound document labels its own pages by sections and reads them in its own
window, and this report's narrative is given none of them. Cone soundings
and dynamic probes keep the sounding reader. Each stage's result is saved
as ``items/mf-<key>.json``; one no answer came back for is not saved, so a
resumed run asks again.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

from report_ingest.bound import APPENDED_LABEL, compact_pages
from report_ingest.calc_reader import split_calc_items
from report_ingest.engine import LARGE_REQUEST_FRACTION, CallCeilingReached
from report_ingest.lab_reader import split_lab_items
from report_ingest.log_holes import missing_holes_qa, split_log_items
from report_ingest.label_vote import (
    DEFAULT_POLICY, POLICIES, PageChoice, Voter, combine, load_trust_table,
    neighbourhood, split_pages,
)
from report_ingest.model import (
    RESUMED_STEPS, BoundReport, DocumentFacts, LabelVote, PageLabel,
    ParentReport, Project, QAEntry, ReportRecord, ResumedSteps,
)

__all__ = [
    "ingest_report", "Budgets", "output_paths", "ITEM_READERS",
    "OUTPUT_NAMES", "REVIEW_MODES", "BOUND_DIR", "MODEL_FIRST_ENV",
    "model_first_setting",
]

#: Where a bound document's own folder goes, under the parent's ``out_dir``.
#: One level, named by the child's handle: ``bound/bound1/``.
BOUND_DIR = "bound"

#: Which pages the label review is given.
#:
#: ``disagreements``
#:     only the pages the voters split on, plus two either side for context.
#:     The default, and the one the measurements argue for: the review's
#:     value is concentrated on pages the cheap voters cannot settle, and on
#:     a tuned report the rest of its changes are as likely to break a label
#:     as to fix one.
#: ``all``
#:     every page, which is what every run before this train did.
#: ``none``
#:     no review; the vote's labels stand.
REVIEW_MODES: Tuple[str, ...] = ("disagreements", "all", "none")

#: What the five exploration answers (``boringCount``, ``testPitCount``,
#: ``cptCount``, ``boringDictionary``, ``testPitDictionary``) take from the
#: logs this record read, once everything is read (BACKLOG RI-14). The rule
#: is the narrative reader's own, :func:`report_ingest.narrative_reader.
#: settled_explorations`; this says how much of it runs. Each mode does what
#: the one before it does, and more.
#:
#: ``off``
#:     the default: the narrative's answers only, as every run before RI-14.
#: ``zeros``
#:     a count the narrative left unanswered is 0 where logs of
#:     other kinds were read and none of this one -- the house convention that
#:     none means 0 (``narrative_glossary`` convention 1, DRAFT), with the
#:     appendix as the evidence. The miss run 7 measured most (cptCount null
#:     on 7 of 8 reports, testPitCount on 5, where the hand writes 0).
#: ``fill``
#:     every answer the narrative left open takes the logs' answer: the
#:     identifiers read and the count of holes, as well as the zeros.
#: ``settle``
#:     ``fill``, and a stated answer the logs contradict is overruled too --
#:     unless the logs' answer is zero or empty, because no cone logs read is
#:     not the same as no cones pushed. This is the reader's lever 4 as
#:     5.24.0 wrote it.
#:
#: Why ``off`` by default: new behaviour stays off until it is measured on
#: real reads (the owner's standing rule), and none of these has been.
#: ``zeros`` rests on a convention still marked DRAFT. RI-14's offline
#: measure only set the log ITEMS planlens' rules find (a stand-in for the
#: logs read) against the eight hand narrative truths: the zeros were right
#: 6 of 6; a non-zero count or list off the logs 5 of 17 -- an appendix holds
#: earlier investigations' logs a bound range did not catch, and a hole can
#: be read as two items -- and overruling a narrative that was right was
#: wrong 12 times in 13. ``zeros`` is the first to measure on real reads.
#:
#: Whichever runs, the reconciler has already compared the narrative's own
#: answer with what was read and recorded every difference as a
#: ``count_mismatch``, and ``narrative.stated_counts`` keeps the numbers the
#: narrative stated.
COUNT_MODES: Tuple[str, ...] = ("off", "zeros", "fill", "settle")

#: Which reader takes which kind of work item. A kind that is not here is
#: recorded and not read -- ``figures``, ``photos``, ``front_matter`` and
#: ``appended_report``, which becomes a record of its own instead.
ITEM_READERS: Dict[str, str] = {
    "boring_log": "log", "test_pit_log": "log",
    # A sounding is not a log. A cone sounding and a dynamic probe measure a
    # SERIES against depth and print no strata, no samples and no drives, so
    # they go to the sounding reader, which reads a series; a test pit IS a
    # log -- a ruler, a description column and samples -- and stays with the
    # log reader, which now seeds a pit's dimensions as well.
    "cpt_log": "cpt", "dcp_log": "dcp",
    "lab_test": "lab", "narrative": "narrative",
    "calculation": "calc",
}

#: Today's reader of each stage model first reads (BACKLOG RI-87), by the
#: stage names ``stage_engines`` takes. With model first off, a stage named
#: there is read by this reader on the engine given, so the two arms can be
#: measured on the same model for each stage (:func:`ingest_report`).
TODAY_READERS: Dict[str, str] = {
    "logs": "log", "lab": "lab", "calc": "calc", "narrative": "narrative"}

#: The files :func:`report_ingest.writers.write_outputs` leaves behind.
OUTPUT_NAMES: Tuple[str, ...] = (
    "report.record.json", "report.summary.md", "report.page.md",
    "report.diggs.xml", "reports.db",
)

#: The model-first switch (BACKLOG RI-87): ``off``, the default (unset or
#: ``0``), or ``on`` (``1``). See :func:`model_first_setting`.
MODEL_FIRST_ENV = "GEOTECH_INGEST_MODEL_FIRST"

#: The readers model-first ingest replaces (``ITEM_READERS``' names), and
#: the stage of :mod:`report_ingest.model_first_read` each becomes.
_MODEL_FIRST_STAGES: Dict[str, str] = {
    "log": "logs", "lab": "lab", "calc": "calc", "narrative": "narrative"}


def model_first_setting(value: Any = None) -> str:
    """``off`` or ``on``: ``value``, or :data:`MODEL_FIRST_ENV` when
    ``value`` is None. Unset or empty is ``off``; a typo is refused."""
    if value is None:
        value = os.environ.get(MODEL_FIRST_ENV)
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value or "").strip().lower()
    if text in ("", "0", "off", "false", "no"):
        return "off"
    if text in ("1", "on", "true", "yes"):
        return "on"
    raise ValueError(f"{MODEL_FIRST_ENV}={value!r}: use off or on")


@dataclass
class Budgets:
    """The model-call ceilings, one per pass, and the size of the run.

    Every number here is a ceiling, not a target: a laboratory sheet whose
    values are tabulated costs one call of its four, and a narrative that fits
    in one call costs one of its eight. What the ceilings buy is the ability
    to say, before a 400-page report runs, what the worst case costs.
    """

    narrative: int = 8
    log: int = 3
    lab: int = 4
    #: The calculation reader's ceiling. Most printouts cost ONE of its two;
    #: the second is spent only on a run too long to show at once or on the
    #: reader's own unsettled list.
    calc: int = 2
    #: The sounding readers' ceiling, shared by the cone sounding and the
    #: dynamic probe. A TABULATED sheet costs one of its two -- the table is
    #: the record and the call only checks the header. A PLOTTED one costs
    #: both: one call per plot panel group for the traces, one for the
    #: header and whatever the first call could not settle.
    sounding: int = 2
    #: The label review's tool-call budget. None takes the one that fits the
    #: mode: ``max(60, 0.25 x pages)`` over a whole report, and
    #: ``max(20, 0.5 x split pages)`` over the pages the voters split on,
    #: which is a far smaller set and needs a budget that says so.
    review_tool_calls: Optional[int] = None
    review_model_calls: int = 60
    #: A ceiling on the vision pass's MODEL CALLS; None is no ceiling and
    #: the mode decides (sheet mode is about one call per six pages).
    vision_model_calls: Optional[int] = None
    #: A ceiling on the range voter's DESCRIBER calls (BACKLOG RI-73, off
    #: unless its switch is on); None is no ceiling: one call a page, and a
    #: second where the first answer was not taken. Reached, the pages left
    #: are described on the next run and the voter sits this one out.
    range_describe_calls: Optional[int] = None
    #: Skip a pass entirely. Triage off means no workflow and the standard
    #: path; review off is ``review_mode="none"`` however it was asked for,
    #: and is kept because callers already spell it this way.
    triage: bool = True
    review: bool = True
    #: A ceiling on how many work items are read, for a smoke run over a big
    #: report. None reads them all. Counted after a log item is cut into
    #: one item per hole (RI-117) and a calculation item into its parts:
    #: each part is an item.
    max_items: Optional[int] = None
    #: A per-request input-token limit (BACKLOG RI-110, RI-121): set, a
    #: log's conversation drops earlier pictures, or its page window is
    #: split, before a request nears it; and model first, a page group whose
    #: request is estimated over it is cut into smaller groups before it is
    #: sent (:mod:`report_ingest.model_first_read`). None, the default, sets
    #: none.
    max_request_tokens_in: Optional[int] = None
    #: The lab reader's sheets per call (BACKLOG RI-116): set to N, a lab
    #: item of more than N pages is read in calls of at most N pages, as a
    #: long run's windows are (``lab_reader.read_lab_sheet``), with ``lab``
    #: the ceiling per call's window. None, the default, changes nothing.
    lab_sheets_per_call: Optional[int] = None

    #: A per-minute input-token budget per model (BACKLOG RI-127): set, the
    #: run's engines share the process's
    #: :class:`~report_ingest.engine.RequestPacer` for it, and after a
    #: request estimated above ``large_request_fraction`` of it, the next
    #: request to that model waits out the minute rather than meet a rate
    #: limit. None, the default, paces nothing.
    tokens_per_minute: Optional[int] = None
    large_request_fraction: float = LARGE_REQUEST_FRACTION

    def __post_init__(self) -> None:
        if self.lab_sheets_per_call is not None:  # refused before any call
            from report_ingest.lab_reader import split_size
            split_size(1, self.lab_sheets_per_call)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def output_paths(out_dir: Any) -> Dict[str, str]:
    """The outputs in ``out_dir`` that exist, by their short names."""
    out = str(out_dir)
    names = {"record": "report.record.json", "summary": "report.summary.md",
             "page": "report.page.md", "diggs": "report.diggs.xml",
             "db": "reports.db"}
    return {key: os.path.join(out, name) for key, name in names.items()
            if os.path.isfile(os.path.join(out, name))}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

class _Spend:
    """What the run has cost so far, across every pass.

    Each step hands in its own cost when it finishes. One that does not
    finish -- the model-call ceiling refused a call it made, or it failed --
    never does, so :meth:`settle` reads what it spent off the run's
    :class:`_Tally` instead: every call made is counted (BACKLOG RI-02's
    review).

    A step taken from the folder made no call, so it hands in nothing:
    :meth:`reuse` names it and keeps what its saved file says it cost apart
    (BACKLOG RI-125). The counts above are this run's; :meth:`resumed` is
    the record's account of the steps that came off disk and of the whole
    report's cost.
    """

    def __init__(self, tally: Optional["_Tally"] = None) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.dollars = 0.0
        self.seconds = 0.0
        self.tally = tally
        #: The steps taken from the folder (:data:`RESUMED_STEPS` names).
        self.reused: List[str] = []
        #: What those steps cost when they were made, as their files say.
        self.saved: Dict[str, Any] = {"calls": 0, "input_tokens": 0,
                                      "output_tokens": 0, "dollars": 0.0}
        #: Readers' items with a reading in the record, and those of them
        #: whose reading came off disk.
        self.readings = 0
        self.readings_reused = 0

    def add(self, cost: Optional[Dict[str, Any]]) -> None:
        if not cost:
            return
        self.calls += int(cost.get("calls") or 0)
        self.input_tokens += int(cost.get("input_tokens") or 0)
        self.output_tokens += int(cost.get("output_tokens") or 0)
        self.dollars += float(cost.get("dollars") or 0.0)
        self.seconds += float(cost.get("seconds") or 0.0)

    def reuse(self, step: str, cost: Optional[Dict[str, Any]]) -> None:
        """A step taken from the folder: no call of this run. Its saved
        file's cost is kept apart, for the whole report's."""
        if step not in self.reused:
            self.reused.append(step)
        cost = cost if isinstance(cost, dict) else {}
        for key in ("calls", "input_tokens", "output_tokens"):
            self.saved[key] += int(cost.get(key) or 0)
        self.saved["dollars"] += float(cost.get("dollars") or 0.0)

    def hand_in(self, step: str, blob: Dict[str, Any],
                cached: Optional[Dict[str, Any]]) -> None:
        """A step's cost: this run's when it made the calls, the folder's
        when ``blob`` is the saved file it read (``cached``)."""
        if cached is not None and blob is cached:
            self.reuse(step, blob.get("cost"))
        else:
            self.add(blob.get("cost"))

    def reading(self, reused: bool) -> None:
        """One reader's item with a reading in the record."""
        self.readings += 1
        self.readings_reused += 1 if reused else 0

    def absorb(self, inner: "_Spend") -> None:
        """A bound document's run, into its parent's: its calls are this
        run's, and the steps it took from its folder are counted too."""
        self.add({"calls": inner.calls,
                  "input_tokens": inner.input_tokens,
                  "output_tokens": inner.output_tokens,
                  "dollars": inner.dollars,
                  "seconds": inner.seconds})
        for step in inner.reused:
            self.reuse(step, None)
        for key in self.saved:
            self.saved[key] += inner.saved[key]
        self.readings += inner.readings
        self.readings_reused += inner.readings_reused

    def resumed(self) -> Optional[ResumedSteps]:
        """The record's account of what came off disk, or None for a run
        that read everything itself (its record is then written as one
        always was)."""
        if not self.reused:
            return None
        saved = self.saved
        return ResumedSteps(
            steps=[step for step in RESUMED_STEPS if step in self.reused],
            items=self.readings_reused, items_of=self.readings,
            saved_calls=saved["calls"],
            saved_input_tokens=saved["input_tokens"],
            saved_output_tokens=saved["output_tokens"],
            saved_dollars=round(saved["dollars"], 5),
            report_calls=self.calls + saved["calls"],
            report_input_tokens=self.input_tokens + saved["input_tokens"],
            report_output_tokens=self.output_tokens + saved["output_tokens"],
            report_dollars=round(self.dollars + saved["dollars"], 5))

    def mark(self) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Where a step starts, for :meth:`settle` and :meth:`cut_short`."""
        return (self.calls,
                self.tally.mark() if self.tally is not None else None)

    def settle(self, mark: Tuple[int, Optional[Dict[str, Any]]]) -> None:
        """Count what a step spent that it did not hand in itself.

        A step that did not finish handed nothing in: all it spent is
        counted. One that did is topped up with the calls it left out --
        sent and failed, which have no reply to count -- so no call is
        counted twice and none is missed.
        """
        before, tallied = mark
        if tallied is None or self.tally is None:
            return
        spent = self.tally.since(tallied)
        if self.calls == before:
            self.add(spent)
            return
        missed = int(spent["calls"]) - (self.calls - before)
        if missed > 0:
            self.calls += missed

    def cut_short(self, mark: Tuple[int, Optional[Dict[str, Any]]]) -> bool:
        """Did the ceiling refuse a call since ``mark``? A voter inside a
        reader may take the refusal as its own failure and the reader then
        finish; what it read is used, but the step is not finished."""
        tallied = mark[1]
        return bool(tallied is not None and self.tally is not None
                    and self.tally.refused > tallied["refused"])


class _Tally:
    """Every call this run's engines made, counted as each one returns.

    The steps' own counts stay what the record reports; this is what a step
    cut short spent (:meth:`_Spend.settle`) and whether the ceiling refused
    a call inside it (:meth:`_Spend.cut_short`). A call that went out and
    failed is a call, as the ceiling counts it; a refused one never went out.
    """

    FIELDS = ("calls", "input_tokens", "output_tokens", "dollars", "seconds")

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.dollars = 0.0
        self.seconds = 0.0
        self.refused = 0

    def wrap(self, engine: Any) -> Any:
        return None if engine is None else _TalliedEngine(engine, self)

    def charge(self, reply: Any) -> None:
        self.calls += 1
        usage = getattr(reply, "usage", None)
        if usage is not None:
            self.input_tokens += int(usage.input_tokens or 0)
            self.output_tokens += int(usage.output_tokens or 0)
            self.dollars += float(usage.dollars(getattr(reply, "model", "")
                                                or ""))
        self.seconds += float(getattr(reply, "seconds", 0.0) or 0.0)

    def mark(self) -> Dict[str, Any]:
        out = {name: getattr(self, name) for name in self.FIELDS}
        out["refused"] = self.refused
        return out

    def since(self, mark: Dict[str, Any]) -> Dict[str, Any]:
        return {name: getattr(self, name) - mark[name]
                for name in self.FIELDS}


class _TalliedEngine:
    """An engine whose calls count on a :class:`_Tally`; everything else is
    the engine it wraps."""

    def __init__(self, inner: Any, tally: _Tally) -> None:
        self._inner = inner
        self._tally = tally

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_inner", "_tally"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def complete(self, messages: Any, **kwargs: Any) -> Any:
        try:
            reply = self._inner.complete(messages, **kwargs)
        except CallCeilingReached:
            self._tally.refused += 1
            raise
        except Exception:
            self._tally.calls += 1
            raise
        self._tally.charge(reply)
        return reply


def _keep(path: str, blob: Dict[str, Any], spend: _Spend,
          mark: Tuple[int, Optional[Dict[str, Any]]], qa: List[QAEntry],
          where: str, pages: Sequence[int] = ()) -> bool:
    """Save a step's fresh result so a resumed run does not pay for it again
    -- unless the ceiling cut the step short. Then what it read is used in
    THIS record and not saved, with a ``partial`` entry saying so: asking
    for the report again reads the step again. True when saved.
    """
    if spend.cut_short(mark):
        qa.append(QAEntry(
            kind="partial", where=where,
            detail="the run's ceiling of model calls refused a call this "
                   "step made, so what it read is in this record but was "
                   "not saved as finished; asking for the report again "
                   "reads it again",
            pages=list(pages)[:200]))
        return False
    _save(path, blob)
    return True


def _is_document(source: Any) -> bool:
    """Is ``source`` an already-open planlens document?

    The cluster stage opens each report through the corpus, which knows
    which pages need the Azure Document Intelligence text and wires it in;
    the graph then reads that document rather than opening the file again
    without it. The caller owns such a document and closes it.
    """
    return all(hasattr(source, name)
               for name in ("page_map", "render", "n_pages", "page"))


def _open(source: Any, di_result: Any, report_id: str):
    from planlens.document import open_document
    if _is_document(source):
        return source
    if isinstance(source, (bytes, bytearray)):
        return open_document(bytes(source), text_source=di_result,
                             name=report_id or "report")
    return open_document(str(source), text_source=di_result,
                         name=report_id or os.path.basename(str(source)))


def _planlens_version() -> str:
    try:
        from importlib.metadata import version
        return version("planlens")
    except Exception:                            # a source checkout
        return ""


def _cached(path: str, resume: bool) -> Optional[Dict[str, Any]]:
    if not resume or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _save(path: str, blob: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(blob, handle, indent=2)


@dataclass
class _Labelling:
    """How this run settles what each page is. Built from the parameters."""

    policy: str = DEFAULT_POLICY
    review_mode: str = "disagreements"
    vision_engine: Any = None
    vision_mode: str = "sheet"
    vision_detail: Optional[str] = None
    trust_table: Any = None
    templates: Any = None
    #: The document triage's engine (``None`` = the main engine).
    triage_engine: Any = None
    note: str = ""                    # printed once, and kept for the run file
    #: What each pass cost, whether this run paid it or read it off disk.
    #: A resumed run's ``paid`` is False and the numbers are still the
    #: report's, which is what a cost-per-report table wants.
    vision_cost: Dict[str, Any] = field(default_factory=dict)
    review_cost: Dict[str, Any] = field(default_factory=dict)
    vision_paid: bool = False
    review_paid: bool = False
    #: The range voter (BACKLOG RI-73): its arm, ``None`` for off (the
    #: default), and the engines that describe the pages and label them by
    #: sections (``None`` = the main engine).
    range_arm: Optional[str] = None
    describe_engine: Any = None
    range_engine: Any = None
    describe_cost: Dict[str, Any] = field(default_factory=dict)
    range_cost: Dict[str, Any] = field(default_factory=dict)
    range_paid: bool = False
    #: Model-first ingest (BACKLOG RI-87): the range labels replace the
    #: vision voter and the label review, which do not run.
    model_first: bool = False

    def resolve(self) -> Dict[str, Any]:
        """The trust table, with the fallback the ``trust`` policy needs."""
        table = load_trust_table(self.trust_table)
        if self.policy == "trust" and not table:
            self.policy = "structural"
            self.note = (
                "label_policy='trust' needs a trust table learned in sample "
                "(the vote stage writes one to vote/trust_table.json); none "
                "was given, so this run falls back to 'structural'.")
            print(f"  {self.note}")
        return table


@dataclass
class _ModelFirst:
    """How a model-first run reads (BACKLOG RI-87): each stage's engine and
    page route, the page descriptions its groups are drawn from, the report
    to cut each call's PDF from, and the placement."""

    #: ``stage -> engine`` for ``logs``, ``lab``, ``calc`` and
    #: ``narrative``.
    engines: Dict[str, Any]
    #: ``stage -> pdf | pictures``: the route each stage's engine takes.
    routes: Dict[str, str]
    #: The route as it was asked for: ``auto``, ``pdf`` or ``pictures``.
    asked: str = "auto"
    #: A path or the report's bytes; None takes the open document's bytes,
    #: once, when a stage first sends a PDF.
    pdf: Any = None
    descriptions: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    #: ``place(stage, record, pages, doc, rulers=None)``
    #: (:func:`_placer`).
    place: Any = None
    #: ``Budgets.max_request_tokens_in`` (BACKLOG RI-121): each reader's
    #: per-request input-token limit, or None.
    max_request_tokens_in: Optional[int] = None
    #: How many groups this run cut for the limit, the bound reports'
    #: included, and the run's QA note, which says so (:func:`_note_cuts`).
    cut: int = 0
    note: Any = None

    def source(self, doc: Any) -> Any:
        if self.pdf is None:
            self.pdf = doc.tobytes()
        return self.pdf


@dataclass
class _Unit:
    """What one model-first result read -- a log, a laboratory window, a
    printout, the narrative -- in the shape the graph's checks read off a
    planlens work item (``kind``, ``id``, ``pages``). ``shown``: every page
    its calls carry, where that is more than ``pages`` (the narrative's
    front pages, shown whatever their label)."""

    kind: str
    id: str
    pages: List[int]
    title: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    shown: List[int] = field(default_factory=list)


def ingest_report(source: Any, engine: Any, *,
                  di_result: Any = None,
                  questions: Optional[Sequence[str]] = None,
                  out_dir: Any = ".",
                  budgets: Optional[Budgets] = None,
                  report_id: str = "",
                  resume: bool = True,
                  write: bool = True,
                  db_path: Any = None,
                  label_policy: str = DEFAULT_POLICY,
                  review_mode: str = "disagreements",
                  vision_engine: Any = None,
                  vision_mode: str = "sheet",
                  vision_detail: Optional[str] = None,
                  trust_table: Any = None,
                  templates: Any = None,
                  ingest_bound: bool = True,
                  triage_engine: Any = None,
                  counts_from_logs: str = "off",
                  narrative_skips_bound: bool = False,
                  range_labels: Optional[str] = None,
                  describe_engine: Any = None,
                  range_engine: Any = None,
                  model_first: Any = None,
                  stage_engines: Any = None,
                  page_route: Optional[str] = None) -> ReportRecord:
    """Read one report end to end and write its record and its exports.

    ``source`` is a PDF path, its bytes, or an OPEN planlens document (the
    caller then owns it and closes it; the cluster stage passes the corpus's
    own document, which already carries the Azure text where it is needed).
    ``di_result`` is an Azure Document Intelligence layout for the scanned
    pages, when one exists --
    ``planlens.document.azure_di`` builds it and planlens puts its text into
    the same frame as everything else. ``questions`` are the caller's own
    questions, answered off the narrative beside the owner's two schemas.

    Everything is written under ``out_dir``: the record and its exports at the
    top, and one file per work item under ``items/`` so a second run resumes
    rather than paying again. Pass ``resume=False`` to re-read everything.

    Parameters
    ----------
    label_policy
        How the page voters are combined: ``structural`` (the default),
        ``trust``, ``confidence``, or ``rules`` for the rules alone, which
        reproduces what this graph did before the vote and calls no vision
        pass at all. See :mod:`report_ingest.label_vote`.
    review_mode
        Which pages the label review is given: ``disagreements`` (the
        default -- the pages the voters split on, plus two either side),
        ``all``, or ``none``. ``Budgets(review=False)`` is ``none`` however
        it was asked for.
    vision_engine, vision_mode, vision_detail
        The second voter. ``vision_engine`` defaults to ``engine``; give it
        a CHEAP tier, which is where the $0.05-a-report figure comes from.
        ``vision_mode`` is ``sheet`` by default -- one call per contact
        sheet, which is the mode the corpus was measured in.
    triage_engine
        The document triage's engine; defaults to ``engine``. The label
        review and the readers stay on ``engine``.
    trust_table
        A path or a dict for the ``trust`` policy, as the ``vote`` stage
        writes to ``vote/trust_table.json``. Without one, ``trust`` falls
        back to ``structural`` and prints a note.
    templates
        Log-template fingerprints for the third voter: a path, an iterable
        of :class:`~report_ingest.log_templates.Fingerprint`, or ``None`` to
        use whatever ``log_templates.use_templates`` put in force -- which
        is nothing unless something set it, and then the voter is a no-op.
    ingest_bound
        Read a report bound inside this one as its OWN record, under
        ``out_dir/bound/<id>/``, and list it on
        ``ReportRecord.bound_documents``. On by default. ``False`` lists the
        pages in a QA entry and does not read them, which is what every run
        before this did. See :mod:`report_ingest.bound`. A work item of this
        report whose pages lie inside a bound document -- a log, a sheet, a
        printout or a sounding on pages triage called another report's while
        the page labels did not -- is that document's: its pages are handed
        to that document's pass, which keeps their labels and reads them into
        its record only, never into this one as well; a page it does not read
        is a ``skipped`` entry here.
    counts_from_logs
        ``off`` (the default), ``zeros``, ``fill`` or ``settle``: how much
        of the five exploration answers the logs read decide. See
        :data:`COUNT_MODES`.
    narrative_skips_bound
        OFF by default, because it changes what the narrative reader is
        shown. On, this report's narrative reader is not given a bound
        document's pages at all -- not as narrative, not as the front of the
        report, not as a searched passage -- and the captions it counts are
        this report's only. Off, it is shown them and TOLD whose they are,
        which is what every run before RI-14 did. Unmeasured either way.
    range_labels, describe_engine, range_engine
        The range voter (BACKLOG RI-73), OFF by default because its prompts
        are new model-facing behaviour. ``range_labels`` is ``plain`` or
        ``hinted`` (the rules' labels given as a hint) to run it, ``off`` to
        keep it off, or ``None`` to follow ``GEOTECH_INGEST_RANGE_LABELS``.
        On, every page is described by ``describe_engine`` (one vision call
        a page, saved under ``out_dir/descriptions/``) and the report is
        labelled by sections by ``range_engine`` (one call, or one per
        window of a long report, saved to ``out_dir/ranges-<arm>.json``);
        that label settles every page it gave one. The describer defaults
        to ``vision_engine``, then ``engine``; the labeller to ``engine``.
        ``Budgets.range_describe_calls`` caps the describer's calls.
        ``label_policy="rules"`` does not run it. See
        :mod:`report_ingest.range_labels`.
    model_first, stage_engines, page_route
        Model-first ingest (BACKLOG RI-87), OFF by default: ``model_first``
        is ``off`` or ``on``, or ``None`` to follow
        ``GEOTECH_INGEST_MODEL_FIRST`` (:func:`model_first_setting`). On,
        the range voter runs (``plain`` unless ``range_labels`` names an
        arm), the vision voter and the label review do not, and the logs,
        laboratory sheets, printouts and narrative are read model first.
        ``stage_engines`` gives each stage's engine: a mapping from
        ``logs``, ``lab``, ``calc`` and ``narrative`` to an engine (a stage
        not in it reads on ``engine``), or a callable that takes a model
        name and returns an engine, called with each stage's model in
        :data:`report_ingest.model_first_read.STAGE_MODELS`; ``None`` reads
        every stage on ``engine``. ``page_route`` says how the pages are
        sent: ``auto``, ``pdf`` or ``pictures``, or ``None`` to follow
        ``GEOTECH_INGEST_MODEL_FIRST_PAGES``. ``label_policy="rules"`` is
        refused: the groups are drawn from the range labels.
        ``Budgets.max_request_tokens_in`` sizes each group's request before
        it is sent (BACKLOG RI-121). Off,
        ``stage_engines`` may still be a mapping: today's reader of each
        stage it names (:data:`TODAY_READERS`) reads on that engine, so the
        two arms can be measured on the same models; the soundings, the
        triage, the vision voter and the label review stay on their own
        engines. ``page_route`` is ignored when off.

    The cost on ``record.document`` is the WHOLE run's, the children
    included; each child's own block says what that child cost. It is THIS
    run's calls only: a step taken from the folder (a saved triage, vision
    pass, review, identity call or item reading) made none, and
    ``record.document.resumed`` names those steps, says how many items came
    off disk of how many read, and gives the whole report's cost -- this
    run's plus what the saved files say their steps cost (BACKLOG RI-125).
    A run that read everything itself leaves it unset.
    """
    from report_ingest.writers import record_key, write_outputs

    budgets = budgets or Budgets()
    if label_policy not in POLICIES:
        raise ValueError(f"unknown label_policy {label_policy!r}; the "
                         f"policies are {list(POLICIES)}")
    if review_mode not in REVIEW_MODES:
        raise ValueError(f"unknown review_mode {review_mode!r}; the modes "
                         f"are {list(REVIEW_MODES)}")
    if counts_from_logs not in COUNT_MODES:
        raise ValueError(f"unknown counts_from_logs {counts_from_logs!r}; "
                         f"the modes are {list(COUNT_MODES)}")
    first = model_first_setting(model_first) == "on"
    if first and label_policy == "rules":
        raise ValueError("model-first ingest draws its groups from the range "
                         "labels; label_policy='rules' takes the rules alone")
    from report_ingest.range_labels import range_arm_of, range_labels_arm

    labelling = _Labelling(
        policy=label_policy,
        review_mode="none" if not budgets.review else review_mode,
        vision_engine=vision_engine, vision_mode=vision_mode,
        vision_detail=vision_detail, trust_table=trust_table,
        templates=templates, triage_engine=triage_engine,
        range_arm=(range_labels_arm() if range_labels is None
                   else range_arm_of(range_labels)),
        describe_engine=describe_engine, range_engine=range_engine)
    out = str(out_dir)
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    started = time.time()
    tally = _Tally()
    engine = tally.wrap(engine)
    labelling.vision_engine = tally.wrap(labelling.vision_engine)
    labelling.triage_engine = tally.wrap(labelling.triage_engine)
    labelling.describe_engine = tally.wrap(labelling.describe_engine)
    labelling.range_engine = tally.wrap(labelling.range_engine)
    spend = _Spend(tally)
    qa: List[QAEntry] = []
    unresolved: List[Dict[str, Any]] = []
    children: List[Tuple[ReportRecord, str]] = []
    first_run = (_model_first_run(
        source, engine, tally, labelling, stage_engines, page_route,
        questions, qa,
        max_request_tokens_in=budgets.max_request_tokens_in)
        if first else None)
    readers = None if first else _today_readers(stage_engines, tally, qa)

    doc = _open(source, di_result, report_id)
    paced: List[Any] = []
    try:
        paced = _pace(budgets, engine, labelling, readers, first_run)
        if first_run is not None:
            first_run.place = _placer(doc)
        record = _run(doc, engine, budgets, out, resume, questions or [],
                      report_id, spend, qa, unresolved, di_result, labelling,
                      children, ingest_bound, counts_from_logs,
                      narrative_skips_bound, first_run, readers)
    finally:
        for real in paced:
            real.pacer = None
        if not _is_document(source):
            doc.close()

    if first_run is not None:
        _note_cuts(first_run, qa)      # the bound reports' cuts counted too
    record.qa.extend(qa)
    record.document.model_calls = spend.calls
    record.document.input_tokens = spend.input_tokens
    record.document.output_tokens = spend.output_tokens
    record.document.dollars = round(spend.dollars, 5)
    record.document.seconds = round(time.time() - started, 1)
    record.document.resumed = spend.resumed()

    if write:
        if _is_document(source):
            key_source: Any = getattr(source, "path", None) or report_id \
                or None
        elif isinstance(source, (bytes, bytearray)):
            key_source = None
        else:
            key_source = source
        # The children FIRST: their records carry the DIGGS verdicts their
        # own writers add, and the parent's summary counts what each of them
        # came to. Both are settled by the time the parent is written.
        by_id = {child.parent.bound_id: child for child, _ in children
                 if child.parent is not None}
        # A bound document belongs in the SAME library as the report it came
        # out of. With no db_path the parent's default is beside its own
        # outputs, so the children are pointed at that rather than each
        # making a one-row database in its own folder.
        library = db_path or os.path.join(out, "reports.db")
        base = record_key(record, key_source)
        for child, child_dir in children:
            write_outputs(child, child_dir, source=key_source,
                          db_path=library, parent_base=base)
        for row in record.bound_documents:
            child = by_id.get(row.bound_id)
            if child is not None:
                row.counts = child.counts()
        write_outputs(record, out, source=key_source, db_path=db_path)
    return record


def _pace(budgets: Budgets, engine: Any, labelling: "_Labelling",
          readers: Optional[Dict[str, Any]], first_run: Any) -> List[Any]:
    """With ``Budgets.tokens_per_minute`` set (BACKLOG RI-127), the
    process's pacer for it on every engine of this run that has none: the
    run's, the labelling's, today's readers' and the model-first stages'.
    Returns the engines it was put on, which the run takes it off again
    when it ends; nothing when the budget is unset."""
    if not budgets.tokens_per_minute:
        return []
    from report_ingest.engine import pace_engines, shared_pacer

    engines = [engine, labelling.vision_engine, labelling.triage_engine,
               labelling.describe_engine, labelling.range_engine]
    engines += list((readers or {}).values())
    if first_run is not None:
        engines += list(first_run.engines.values())
    return pace_engines(engines, shared_pacer(
        budgets.tokens_per_minute,
        large_fraction=budgets.large_request_fraction))


def _today_readers(stage_engines: Any, tally: _Tally,
                   qa: List[QAEntry]) -> Optional[Dict[str, Any]]:
    """With model first off, today's readers on the engines
    ``stage_engines`` names (:data:`TODAY_READERS`), counted on the run's
    tally, and one QA note that names them. None when it names none: every
    reader reads on the run's engine, as before.

    A mapping only: a callable is given the model names of
    :mod:`report_ingest.model_first_read`, which a run with model first off
    does not import."""
    if stage_engines is None:
        return None
    if not isinstance(stage_engines, dict):
        raise ValueError("with model first off, stage_engines is a mapping "
                         "from logs, lab, calc or narrative to an engine")
    unknown = sorted(set(stage_engines) - set(TODAY_READERS))
    if unknown:
        raise ValueError(f"stage_engines names {unknown}; the stages are "
                         f"{list(TODAY_READERS)}")
    named = [stage for stage in TODAY_READERS
             if stage_engines.get(stage) is not None]
    if not named:
        return None
    qa.append(QAEntry(
        kind="note", where="readers",
        detail="today's readers, each stage named here on its own engine "
               "(stage_engines); the rest of the run on the run's engine",
        values=[f"{stage}: "
                f"{getattr(stage_engines[stage], 'name', '') or 'engine'}"
                for stage in named]))
    return {TODAY_READERS[stage]: tally.wrap(stage_engines[stage])
            for stage in named}


def _run(doc: Any, engine: Any, budgets: Budgets, out: str, resume: bool,
         questions: Sequence[str], report_id: str, spend: _Spend,
         qa: List[QAEntry], unresolved: List[Dict[str, Any]],
         di_result: Any, labelling: "_Labelling",
         children: List[Tuple[ReportRecord, str]],
         ingest_bound: bool = True, counts_from_logs: str = "off",
         narrative_skips_bound: bool = False,
         model_first: Optional[_ModelFirst] = None,
         readers: Optional[Dict[str, Any]] = None) -> ReportRecord:
    """The loop itself, with the document open.

    ``children`` is filled with ``(record, folder)`` for every report bound
    inside this one, for the caller to write; they are not written here
    because the library key and the database both belong to the caller's
    ``source``, which this function does not have. ``readers`` puts today's
    reader of a stage on its own engine (:func:`_today_readers`), in this
    report and in the reports bound inside it.
    """
    from planlens.document.model import SOURCE_AZURE_DI
    from planlens.document.roles import (
        assign, build_items, document_outline, facts_from_document,
    )
    from report_ingest.reconciler import reconcile
    from report_ingest.triage import (
        document_facts, settle_workflow, triage as run_triage)

    facts = facts_from_document(doc)
    roles = assign(facts)
    build_items(facts, roles)
    outline = document_outline(doc)
    counted = document_facts(doc, roles)

    summaries = list(doc.page_map())
    di_pages = [s.page for s in summaries
                if s.evidence.get("text_source") == SOURCE_AZURE_DI]
    no_text = [s.page for s in summaries
               if not s.text_reliable or s.n_text_chars == 0]

    record = ReportRecord()
    record.document = DocumentFacts(
        report_id=report_id,
        n_pages=doc.n_pages,
        page_roles=dict(counted.role_counts),
        scan_fraction=counted.scan_fraction,
        di_pages=len(di_pages),
        planlens_version=_planlens_version(),
        workflow="standard")

    # -- 0b. triage ---------------------------------------------------------
    profile = None
    workflow_note = ""
    if budgets.triage:
        cached = _cached(os.path.join(out, "triage.json"), resume)
        if cached is not None:
            # A triage.json written before RI-20 holds the answer as the
            # model gave it; it is settled the same way a fresh one is.
            record.document.workflow, workflow_note = settle_workflow(
                cached.get("workflow", "standard"))
            workflow_note = cached.get("workflow_note") or workflow_note
            profile = _Profile(cached)
            spend.reuse("triage", cached.get("cost"))
        else:
            # Triage is one call over a ledger: a cheaper tier is usually
            # enough, and the scoring stage names one. Until 5.32 the graph
            # had no way to take it, so ``triage_model`` on the ingest stage
            # was silently the main model (Foundry run, 2026-10-01).
            mark = spend.mark()
            profile = run_triage(doc, roles, outline,
                                 engine=labelling.triage_engine or engine,
                                 facts=counted)
            spend.add(profile.cost)
            spend.settle(mark)
            _save(os.path.join(out, "triage.json"), profile.to_dict())
            record.document.workflow = profile.workflow
            workflow_note = profile.workflow_note
    workflow = record.document.workflow
    if workflow_note:
        qa.append(QAEntry(kind="note", where="workflow",
                          detail=workflow_note))

    if profile is not None:
        for note in getattr(profile, "anomalies", ()) or ():
            qa.append(QAEntry(kind="note", where="triage", detail=str(note)))
        bound = getattr(profile, "bound_together", ()) or ()
        if bound:
            qa.append(QAEntry(
                kind="note", where="triage.bound_together",
                detail=f"{len(bound)} separately-bound document(s) are in "
                       f"this file; each is read into its own record and "
                       f"nothing of theirs is counted as this report's"
                       if ingest_bound else
                       f"{len(bound)} separately-bound document(s) are in "
                       f"this file; the answers below mix them until somebody "
                       f"splits it",
                values=[f"{b.get('kind')} {b.get('pages')}" for b in bound]))

    if workflow == "needs_person":
        qa.append(QAEntry(
            kind="skipped", where="workflow",
            detail="triage could not tell what this file is, so nothing was "
                   "read. " + (getattr(profile, "rationale", "") or "")))
        return record

    # -- 0c. the vote: the rules, the picture and the printed form ----------
    choices, vision, templates = _vote_labels(
        doc, engine, budgets, out, resume, roles, labelling, spend, qa,
        outline)
    labels = {c.page: c.label for c in choices}
    split = split_pages(choices)
    confidence = {c.page: c.confidence for c in choices}
    if model_first is not None:
        from report_ingest.range_labels import load_descriptions
        model_first.descriptions = load_descriptions(
            os.path.join(out, "descriptions"))
    # The review checks the labels the record CURRENTLY has, which after the
    # vote are the vote's and not the rules'. So the roles it is handed --
    # its ledger, its weak-spot list and the labels it validates a change
    # against -- carry the vote's answer and the vote's confidence.
    roles = _relabel(roles, labels, confidence)

    # -- 0d. the label review, on the pages the voters could not settle -----
    changed = _review(doc, engine, budgets, out, resume, roles, outline,
                      profile, labelling, choices, labels, split, spend, qa,
                      unresolved)

    record.page_labels = [_page_label(c, changed) for c in choices]
    record.document.label_policy = labelling.policy
    record.document.label_split_pages = len(split)
    record.document.review_mode = labelling.review_mode
    record.document.review_changed = len(changed)
    _save(os.path.join(out, "labels.json"),
          _labels_blob(choices, split, changed, labelling, doc.n_pages))

    roles = _relabel(roles, labels, confidence)
    # One log item per hole (RI-117): planlens' items run on over pages whose
    # headers name another hole where its title-block code is not the
    # hole's name. One lab item per sheet (RI-150): a summary table and the
    # grading sheets after it are cut apart where their titles differ.
    items = split_lab_items(doc, split_log_items(doc, split_calc_items(
        doc, build_items(facts, roles))))

    # -- 0e. the reports bound inside this one ------------------------------
    # Decided BEFORE the readers, because the parent's narrative reader is
    # told the ranges: the borings behind that tab are a previous
    # investigation's and the counts it answers are about this report.
    ranges = _bound_ranges(profile, labels, doc.n_pages, qa, ingest_bound)
    told = _identify_bound(doc, engine, out, resume, ranges, spend, qa)

    # A bound document's pages are ITS pages. The work items were built off
    # the page labels, and a range triage drew where the labels did not say
    # ``appended_report`` still has logs and sheets labelled as such inside
    # it: read here as well, they would be in this record AND the child's.
    # They are HANDED to the bound document's pass, which keeps these
    # labels (:func:`_one_bound`), and what it does not read is said below.
    bound = {int(page) for row in ranges for page in row.pages}
    # Model first, the narrative is given none of a bound document's pages
    # (``narrative_skips_bound`` always on): its groups are drawn from the
    # labels, and a range triage drew round pages labelled narrative would
    # otherwise be read as this report's.
    skips_bound = narrative_skips_bound or model_first is not None
    items, handed = _outside_bound(items, bound, qa, skips_bound)
    every = list(items)
    if budgets.max_items:
        items = items[:int(budgets.max_items)]
    own = {page: label for page, label in labels.items()
           if page not in bound}
    window = ([page for page in range(doc.n_pages) if page not in bound]
              if bound and skips_bound else None)

    # -- 1-2. the readers ---------------------------------------------------
    if model_first is None:
        read = _read_items(doc, engine, budgets, out, resume, questions,
                           report_id, items, own if narrative_skips_bound
                           else labels, outline, record, spend, qa,
                           unresolved, workflow, no_text, di_pages, told,
                           bound, window,
                           narrative_key="-skips-bound" if window else "",
                           readers=readers)
    else:
        mine, every, items, rest = _model_first_plan(
            doc, model_first, own, window, every, budgets.max_items)
        read = _read_model_first(
            doc, model_first, out, resume, report_id, own, outline, record,
            spend, qa, unresolved, workflow, mine, window)
        read += _read_items(doc, engine, budgets, out, resume, questions,
                            report_id, rest, own, outline, record, spend, qa,
                            unresolved, workflow, no_text, di_pages, told,
                            bound, window)

    # -- 3. the reconciler --------------------------------------------------
    # Over this report's own pages: the bound documents' are checked in
    # their own records.
    reconcile(record, labels=own, items=items,
              no_text_pages=[p for p in no_text if p not in bound],
              di_pages=[p for p in di_pages if p not in bound],
              reader_unresolved=unresolved, doc=doc, outline=outline)
    if model_first is None:
        # The holes the log items' page headers print in a field or a title
        # that no log in the record carries (RI-117): today's log reader
        # returns one hole per item.
        qa.extend(missing_holes_qa(doc, items, record))
    _counts_from_logs(record, qa, counts_from_logs, every, read)
    record.project = _project_from(record)

    # -- 4. each bound document, round this same loop, into its own record --
    if ranges:
        reads = _read_bound(doc, engine, budgets, out, resume, report_id,
                            facts, vision, templates, labelling, ranges,
                            told, record, children, spend, qa, no_text,
                            di_pages, counts_from_logs, model_first,
                            readers)
        _handed_over(every, handed, ranges, reads, qa)
    return record


class _Profile:
    """A cached triage profile, read back off disk."""

    def __init__(self, blob: Dict[str, Any]) -> None:
        self.__dict__.update(blob)
        self.cost = {}

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "cost"}


# ---------------------------------------------------------------------------
# 0c. the vote
# ---------------------------------------------------------------------------

def _relabel(roles: Sequence[Any], labels: Dict[int, str],
             confidence: Dict[int, float]) -> List[Any]:
    """planlens' page roles, carrying the labels the vote settled on.

    The EVIDENCE is the rules' own and is kept: it is what the review's
    weak-spot list reads to say why a label is doubtful, and a vote does
    not make the tab that declared a page stop having declared it.
    """
    from planlens.document.roles import PageRole

    return [PageRole(page=r.page, role=labels.get(r.page, r.role),
                     confidence=confidence.get(r.page, r.confidence),
                     evidence=r.evidence)
            for r in roles]

def _vote_labels(doc: Any, engine: Any, budgets: Budgets, out: str,
                 resume: bool, roles: Sequence[Any], labelling: "_Labelling",
                 spend: _Spend, qa: List[QAEntry], outline: Any = None
                 ) -> Tuple[List[PageChoice], Dict[int, Tuple[str, float]],
                            Dict[int, Any]]:
    """One :class:`PageChoice` per page, from the voters this run has.

    The rules always vote. The vision pass votes unless the policy is
    ``rules``, and a pass that will not run is a QA note and one voter
    fewer rather than a failed report. The template votes only where a
    fingerprint file is in force, which is not the default state. The range
    voter votes only where its switch is on (:func:`_range_labels`).

    The two non-rules voters' answers come back with the choices so that a
    bound document, whose RULES answer has to be recomputed over its own
    pages, can be voted on without paying for a second vision pass over
    pages this one has already looked at.
    """
    table = labelling.resolve()
    rules = {int(r.page): (r.role, float(r.confidence)) for r in roles}
    vision = _vision_labels(doc, engine, budgets, out, resume, labelling,
                            spend, qa)
    templates = _template_labels(doc, labelling, qa)
    ranged = _range_labels(doc, engine, out, resume, rules, outline,
                           labelling, spend, qa, budgets)
    pages = sorted(set(rules) | set(vision) | set(ranged))
    out_rows: List[PageChoice] = []
    for page in pages:
        role, confidence = rules.get(page, ("", None))
        seen = vision.get(page)
        match = templates.get(page)
        out_rows.append(combine(
            Voter("rules", role, confidence),
            Voter("vision", seen[0], seen[1]) if seen else None,
            Voter("template", "", match.confidence, match.family)
            if match else None,
            policy=labelling.policy, trust_table=table, page=page,
            ranges=Voter("ranges", ranged[page]) if page in ranged
            else None))
    return out_rows, vision, templates


def _range_labels(doc: Any, engine: Any, out: str, resume: bool,
                  rules: Dict[int, Tuple[str, float]], outline: Any,
                  labelling: "_Labelling", spend: _Spend,
                  qa: List[QAEntry],
                  budgets: Optional[Budgets] = None) -> Dict[int, str]:
    """``page -> label`` from the range voter (BACKLOG RI-73), or nothing.

    Nothing unless the switch names an arm. The descriptions are saved one
    page at a time as they are made, so a run cut short pays again only for
    the pages it had not reached; the sections are saved once the labeller
    has answered over a COMPLETE set of descriptions, so a page that could
    not be described is asked about again next time rather than labelled
    from a gap for good. A pass that fails is a QA note and one voter fewer.

    The saved sections record the describer, the labeller, the describer's
    prompt version, the conventions and a fingerprint of the labeller's
    instructions; a saved file that differs in any of them is not used. A
    describing pass stopped by its ceiling labels nothing: the voter sits
    this run out, and the next one describes the pages left.
    """
    arm = labelling.range_arm
    if arm is None:
        return {}
    if labelling.policy == "rules":
        qa.append(QAEntry(
            kind="note", where="labels.ranges",
            detail="the range voter is switched on, but label_policy='rules' "
                   "takes the rules alone, so it did not run"))
        return {}
    from report_ingest.range_labels import (
        DESCRIBE_PROMPT_VERSION, _engine_name, conventions_version,
        describe_pages, label_by_ranges, labeller_fingerprint,
    )

    describer = labelling.describe_engine or labelling.vision_engine or engine
    labeller = labelling.range_engine or engine
    made_by = {"describer": _engine_name(describer),
               "labeller": _engine_name(labeller),
               "describe_prompt": DESCRIBE_PROMPT_VERSION,
               "conventions": conventions_version(),
               "labeller_prompt": labeller_fingerprint(arm == "hinted")}
    path = os.path.join(out, f"ranges-{arm}.json")
    blob = _cached(path, resume)
    if blob is not None and any(blob.get(key) != value
                                for key, value in made_by.items()):
        qa.append(QAEntry(
            kind="note", where="labels.ranges",
            detail="the saved sections came from another describer or "
                   "labeller (or set of instructions) and were not used"))
        blob = None
    if blob is not None:
        # The sections and every page description they were drawn from
        # came off disk (``saved_cost`` is every description's own).
        spend.reuse("ranges", blob.get("cost"))
        spend.reuse("descriptions",
                    (blob.get("describe") or {}).get("saved_cost"))
    if blob is None:
        from report_ingest.triage import outline_text

        mark = spend.mark()
        try:
            described = describe_pages(
                doc, describer, folder=os.path.join(out, "descriptions"),
                redo=not resume,
                max_calls=(budgets.range_describe_calls if budgets
                           else None))
            spend.add(described.cost)
            if described.reused:
                spend.reuse("descriptions", described.reused_cost)
            if described.stopped:
                spend.settle(mark)
                left = doc.n_pages - len(described.pages)
                qa.append(QAEntry(
                    kind="partial", where="labels.ranges",
                    detail=f"the describer stopped ({described.stopped}) "
                           f"with {left} page(s) not described, so the "
                           f"range voter did not label this run; asking "
                           f"for the report again describes the rest"))
                return {}
            found = label_by_ranges(
                described.pages, doc.n_pages, labeller,
                outline=outline_text(outline) if outline is not None else "",
                hints=({page: role for page, (role, _c) in rules.items()}
                       if arm == "hinted" else None))
        except Exception as exc:              # one voter, not the report
            spend.settle(mark)
            qa.append(QAEntry(
                kind="note", where="labels.ranges",
                detail=f"the range voter did not run ({type(exc).__name__}: "
                       f"{exc}), so the page labels are the other voters'"))
            return {}
        spend.add(found.cost)
        blob = found.to_dict()
        blob.update(made_by)
        blob["describe"] = described.to_dict()
        missing = [row["page"] for row in described.unresolved]
        if missing:
            qa.append(QAEntry(
                kind="partial", where="labels.ranges",
                detail=f"{len(missing)} page(s) could not be described, so "
                       f"the sections were drawn with a gap there and are "
                       f"not saved as finished; asking for the report again "
                       f"describes those pages and draws them again",
                pages=missing[:200]))
        else:
            _keep(path, blob, spend, mark, qa, "labels.ranges")
        spend.settle(mark)
        labelling.range_paid = True
    labelling.describe_cost = dict(
        (blob.get("describe") or {}).get("saved_cost") or {})
    labelling.range_cost = dict(blob.get("cost") or {})
    if blob.get("unresolved"):
        qa.append(QAEntry(
            kind="note", where="labels.ranges",
            detail=f"{len(blob['unresolved'])} page(s) no section settled; "
                   f"the other voters label them",
            pages=list(blob["unresolved"])[:200]))
    return {int(k): str(v) for k, v in (blob.get("labels") or {}).items()}


def _vision_labels(doc: Any, engine: Any, budgets: Budgets, out: str,
                   resume: bool, labelling: "_Labelling", spend: _Spend,
                   qa: List[QAEntry]) -> Dict[int, Tuple[str, float]]:
    """``page -> (label, confidence)`` from one pass over the pictures.
    None under ``rules``, and none model first: the range labels replace
    it."""
    if labelling.policy == "rules" or labelling.model_first:
        return {}
    path = os.path.join(out, "vision.json")
    blob = _cached(path, resume)
    if blob is not None:
        spend.reuse("vision", blob.get("cost"))
    else:
        from report_ingest.vision_labels import classify_pages_by_vision
        mark = spend.mark()
        try:
            seen = classify_pages_by_vision(
                doc, labelling.vision_engine or engine,
                mode=labelling.vision_mode, detail=labelling.vision_detail,
                budget=budgets.vision_model_calls)
        except Exception as exc:              # one voter, not the report
            spend.settle(mark)
            qa.append(QAEntry(
                kind="note", where="labels.vision",
                detail=f"the vision voter did not run "
                       f"({type(exc).__name__}: {exc}), so the page labels "
                       f"are the rules' alone and nothing was flagged as a "
                       f"disagreement"))
            return {}
        blob = seen.to_dict()
        _keep(path, blob, spend, mark, qa, "labels.vision")
        spend.add(blob.get("cost"))
        spend.settle(mark)
        labelling.vision_paid = True
    labelling.vision_cost = dict(blob.get("cost") or {})
    labels = {int(k): str(v) for k, v in (blob.get("labels") or {}).items()}
    out_rows: Dict[int, Tuple[str, float]] = {}
    for entry in (blob.get("detail") or []):
        try:
            page = int(entry["page"])
        except (KeyError, TypeError, ValueError):
            continue
        if page in labels:
            out_rows[page] = (labels[page],
                              float(entry.get("confidence") or 0.0))
    for page, label in labels.items():
        out_rows.setdefault(page, (label, 0.0))
    return out_rows


def _template_labels(doc: Any, labelling: "_Labelling",
                     qa: List[QAEntry]) -> Dict[int, Any]:
    """``page -> TemplateMatch`` where a fingerprint claims the page."""
    from report_ingest import log_templates

    try:
        rows = log_templates.resolve_templates(labelling.templates)
    except (OSError, ValueError):             # a file that will not parse
        rows = []
    if not rows:
        return {}
    matches = log_templates.recognise_pages(doc, range(doc.n_pages),
                                            templates=rows)
    if matches:
        qa.append(QAEntry(
            kind="note", where="labels.template",
            detail=f"{len(matches)} page(s) were recognised as a printed log "
                   f"form and voted as exploration logs",
            pages=sorted(matches)))
    return matches


# ---------------------------------------------------------------------------
# 0d. the review, on the pages the voters split on
# ---------------------------------------------------------------------------

def _review(doc: Any, engine: Any, budgets: Budgets, out: str, resume: bool,
            roles: Sequence[Any], outline: Any, profile: Any,
            labelling: "_Labelling", choices: List[PageChoice],
            labels: Dict[int, str], split: Sequence[int], spend: _Spend,
            qa: List[QAEntry],
            unresolved: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Run the label review under its mode and apply what it changed.

    Returns ``page -> the change``, and mutates ``labels`` and ``choices``
    so that what the readers see and what the record says are the same
    thing. Every split the review did not settle becomes a
    ``label_disagreement``; a change on a page the voters AGREED on is
    applied -- the agent looked and it may well be right -- and flagged,
    because it is outside what this mode asked for.
    """
    from report_ingest.label_review import (
        SPLIT_CONTEXT, budget_for_split, review_labels,
    )

    mode = labelling.review_mode
    asked: Optional[List[int]] = None
    if mode == "disagreements":
        asked = neighbourhood(split, doc.n_pages,
                              either_side=SPLIT_CONTEXT)
    changed: Dict[int, Dict[str, Any]] = {}
    targeted = False

    blob = None if mode == "none" else _cached(
        os.path.join(out, "review.json"), resume)
    if blob is not None:
        # Kept by an earlier attempt, or seeded from a separate label run.
        spend.reuse("review", blob.get("cost"))
    if blob is None and mode != "none" \
            and not (mode == "disagreements" and not split):
        budget = budgets.review_tool_calls
        if budget is None and mode == "disagreements":
            budget = budget_for_split(len(split))
        mark = spend.mark()
        try:
            review = review_labels(doc, roles, outline, profile,
                                   budget=budget, engine=engine,
                                   max_model_calls=budgets.review_model_calls,
                                   pages=asked, choices=choices)
        except Exception as exc:              # the review, not the report
            # A review that could not be had -- on the Foundry run of
            # 2026-10-04 a long report's ran out of output room and took the
            # whole report with it. The VOTE's labels stand, every split it
            # did not settle is a label_disagreement below as always, and
            # this says why none was settled. Nothing is saved, so a resumed
            # run asks again.
            cost = dict(getattr(exc, "cost", None) or {})
            if cost:
                spend.add(cost)
                labelling.review_paid = True
                labelling.review_cost = cost
            spend.settle(mark)
            qa.append(QAEntry(
                kind="partial", where="labels.review",
                detail=(f"the label review did not finish "
                        f"({type(exc).__name__}: {exc}); the page labels are "
                        f"the vote's, and every page the voters split on is "
                        f"left as a disagreement"),
                pages=list(asked or [])[:200]))
            review = None
        if review is not None:
            spend.add(review.cost)
            spend.settle(mark)
            labelling.review_paid = True
            blob = review.to_dict()
            _keep(os.path.join(out, "review.json"), blob, spend, mark, qa,
                  "labels.review", asked or ())
    if blob is not None:
        targeted = bool(blob.get("asked_pages"))
        labelling.review_cost = dict(blob.get("cost") or {})
        for row in _review_changes(blob, choices):
            page = int(row["page"])
            if page not in labels:
                continue
            labels[page] = str(row.get("to") or labels[page])
            changed[page] = dict(row)
        for row in (blob.get("unresolved") or []):
            unresolved.append({"what": f"page {row.get('page')}",
                               "why": row.get("why", ""),
                               "page": row.get("page")})

    settled = set(changed)
    for choice in choices:
        if choice.agreed or choice.page in settled:
            continue
        qa.append(QAEntry(
            kind="label_disagreement", where=f"labels.page{choice.page}",
            detail=(("the page labellers did not agree, and the label review "
                     "does not run model first" if labelling.model_first else
                     "the page labellers did not agree and the review did "
                     "not settle it")
                    + f"; the record calls page {choice.page} "
                    f"'{choice.label}' under the '{choice.policy}' policy"),
            values=[_voter_line(v) for v in choice.voters],
            pages=[choice.page]))
    for page, row in sorted(changed.items()):
        choice = next((c for c in choices if c.page == page), None)
        if choice is None or not choice.agreed or not targeted:
            continue
        qa.append(QAEntry(
            kind="note", where=f"labels.page{page}",
            detail=(f"the voters agreed on page {page}, so it was not one of "
                    f"the pages the review was asked about; the review moved "
                    f"it from '{row.get('from')}' to '{row.get('to')}' "
                    f"anyway"),
            values=[str(row.get("reason") or "")],
            pages=[page]))
    return changed


def _review_changes(blob: Dict[str, Any],
                    choices: Sequence[PageChoice]) -> List[Dict[str, Any]]:
    """The review's changes, whichever shape the saved file is in.

    A review this graph ran lists them. A review REUSED from a cluster
    ``labels`` run carries only its final labels, because that is all the
    label stage saved -- so the changes are what those labels say that the
    RULES did not, which is exactly what the review had changed.

    A saved change whose evidence is missing, empty or names no look at the
    page is left out: a review run now refuses it (BACKLOG RI-15), and a
    resumed run must not apply what a fresh one would refuse. One saved
    before that refusal existed is the case this catches. Only the
    ``reused`` mark, which the label-reuse path writes, is exempt.
    """
    from report_ingest.label_review import names_a_look

    rows: List[Dict[str, Any]] = []
    if blob.get("changes") is not None:
        for row in (blob.get("changes") or []):
            try:
                int(row["page"])
            except (KeyError, TypeError, ValueError):
                continue
            evidence = row.get("evidence")
            if evidence != "reused" and not names_a_look(evidence or ""):
                continue
            rows.append(dict(row))
        return rows
    rules = {}
    for choice in choices:
        voter = choice.voter("rules")
        if voter is not None and voter.label:
            rules[choice.page] = voter.label
    for key, label in sorted((blob.get("final_labels") or {}).items()):
        try:
            page = int(key)
        except (TypeError, ValueError):
            continue
        if rules.get(page) and str(label) != rules[page]:
            rows.append({"page": page, "from": rules[page], "to": str(label),
                         "reason": "a saved label run's review said so",
                         "evidence": "reused"})
    return rows


def _voter_line(voter: Voter) -> str:
    """One voter as a line of a QA entry: labels and numbers only."""
    said = voter.label or f"template {voter.family}"
    return f"{voter.name} {said} ({float(voter.confidence or 0.0):.2f})"


def _page_label(choice: PageChoice,
                changed: Dict[int, Dict[str, Any]]) -> PageLabel:
    """One page's entry in the record, after the review has had its say."""
    row = changed.get(choice.page)
    return PageLabel(
        page=choice.page,
        label=str(row.get("to")) if row else choice.label,
        confidence=choice.confidence,
        agreed=choice.agreed,
        policy=choice.policy,
        settled_by="review" if row else "vote",
        voters=[LabelVote(voter=v.name, label=v.label,
                          confidence=float(v.confidence or 0.0),
                          family=v.family) for v in choice.voters])


def _labels_blob(choices: Sequence[PageChoice], split: Sequence[int],
                 changed: Dict[int, Dict[str, Any]],
                 labelling: "_Labelling", n_pages: int) -> Dict[str, Any]:
    """The run's own account of the vote, for a scorecard to read back."""
    return {
        "policy": labelling.policy,
        "review_mode": labelling.review_mode,
        "vision_mode": (labelling.vision_mode
                        if labelling.policy != "rules"
                        and not labelling.model_first else ""),
        "note": labelling.note,
        **({"model_first": "on"} if labelling.model_first else {}),
        **({"range_arm": labelling.range_arm} if labelling.range_arm
           else {}),
        "n_pages": int(n_pages),
        "pages_voted": len(choices),
        "split_pages": list(split),
        "n_split": len(split),
        "n_agreed": sum(1 for c in choices if c.agreed),
        "review_changed": sorted(int(p) for p in changed),
        "n_review_changed": len(changed),
        "n_review_changed_off_split": sum(
            1 for c in choices if c.agreed and c.page in changed),
        "cost": {"vision": dict(labelling.vision_cost),
                 "review": dict(labelling.review_cost),
                 "vision_paid": labelling.vision_paid,
                 "review_paid": labelling.review_paid,
                 **({"describe": dict(labelling.describe_cost),
                     "ranges": dict(labelling.range_cost),
                     "ranges_paid": labelling.range_paid}
                    if labelling.range_arm else {})},
        "labels": {str(c.page): (str(changed[c.page].get("to"))
                                 if c.page in changed else c.label)
                   for c in choices},
        "pages": [c.to_dict() for c in choices],
    }


# ---------------------------------------------------------------------------
# 0e / 4. the reports bound inside this one
# ---------------------------------------------------------------------------

def _bound_ranges(profile: Any, labels: Dict[int, str], n_pages: int,
                  qa: List[QAEntry], ingest_bound: bool) -> List[Any]:
    """The bound documents' page ranges, or nothing when the pass is off."""
    from report_ingest.bound import bound_ranges

    if not ingest_bound:
        return []
    ranges, notes = bound_ranges(profile, labels, n_pages)
    qa.extend(notes)
    return ranges


def _outside_bound(items: Sequence[Any], bound: Collection[int],
                   qa: List[QAEntry], narrative_too: bool = False
                   ) -> Tuple[List[Any], List[Tuple[Any, List[int]]]]:
    """This report's work items, with the bound documents' pages taken out.

    Only the items a reader would read: a log, a sheet, a printout, a
    sounding -- and the narrative when ``narrative_too``. Each loses the
    pages that are a bound document's, and an item left with none is not
    this report's at all. The rest (figures, front matter, the
    ``appended_report`` run itself) are read by nobody and stay listed.

    Returns the items kept and, for each item that lost pages, the item and
    the pages it lost: those are HANDED to the bound document's own pass,
    which keeps their labels (:func:`_one_bound`) so that the same reader
    reads them there, and :func:`_handed_over` says which it did not read.
    A trimmed item is marked, so that its reading is not taken off a
    folder written before the trim, when it was read whole.
    """
    import dataclasses

    if not bound:
        return list(items), []
    kept: List[Any] = []
    handed: List[Tuple[Any, List[int]]] = []
    for item in items:
        reader = ITEM_READERS.get(item.kind)
        inside = [int(p) for p in item.pages if int(p) in bound]
        if not inside or reader is None or (reader == "narrative"
                                            and not narrative_too):
            kept.append(item)
            continue
        handed.append((item, inside))
        outside = [int(p) for p in item.pages if int(p) not in bound]
        if outside:
            kept.append(dataclasses.replace(
                item, pages=outside,
                evidence={**dict(item.evidence), "first_page": outside[0],
                          "bound_pages_handed": compact_pages(inside)}))
    if handed:
        pages = sorted({p for _item, inside in handed for p in inside})
        qa.append(QAEntry(
            kind="note", where="bound.items",
            detail=(f"{len(handed)} work item(s) labelled as this report's "
                    f"lie on the pages of a report bound inside it; those "
                    f"pages are handed to that report's own pass, which "
                    f"keeps their labels, and are not read as this "
                    f"report's"),
            values=[f"{item.id} {item.kind} {compact_pages(inside)}"
                    for item, inside in handed],
            pages=pages))
    return kept, handed


def _handed_over(items: Sequence[Any],
                 handed: Sequence[Tuple[Any, List[int]]],
                 ranges: Sequence[Any],
                 reads: Dict[str, Optional[Set[Tuple[int, str]]]],
                 qa: List[QAEntry]) -> None:
    """What the bound documents' passes did with the pages handed to them.

    Said only AFTER they ran, never assumed. ``reads`` is, per bound
    document, every ``(page, reader)`` its pass read to the end -- ``None``
    when the pass failed. A page handed over that no bound document read
    with the reader this report would have used (that pass failed, its item
    failed, or its item ceiling cut it) is ``skipped`` here, in the shape
    the reconciler gives a labelled page nothing read: nothing of it is in
    either record. And each run of ``appended_report`` pages inside a range
    is said to be read only when the bound document it is in was.
    """
    owner = {int(p): row.bound_id for row in ranges for p in row.pages}
    for item, inside in handed:
        reader = ITEM_READERS.get(item.kind)
        missed = [p for p in inside
                  if (p, reader) not in (reads.get(owner.get(p, "")) or ())]
        if not missed:
            continue
        qa.append(QAEntry(
            kind="skipped", where=f"pages.{item.kind}",
            detail=(f"{len(missed)} page(s) labelled {item.kind} lie in a "
                    f"report bound inside this one "
                    f"({', '.join(sorted({owner[p] for p in missed}))}) and "
                    f"were handed to its own pass, which did not read them "
                    f"with the {reader} reader (the pass failed, its item "
                    f"failed or its item ceiling cut it): nothing read them"),
            values=[item.id],
            pages=missed))

    for item in items:
        pages = [int(p) for p in item.pages]
        if item.kind != APPENDED_LABEL or not pages \
                or not set(pages) <= set(owner):
            continue
        failed = sorted({owner[p] for p in pages
                         if reads.get(owner[p]) is None})
        qa.append(QAEntry(
            kind="skipped" if failed else "note",
            where="items.appended_report",
            detail=("a report bound inside this one; its pages were read "
                    "into their own record and are listed on "
                    "bound_documents" if not failed else
                    f"a report bound inside this one ({', '.join(failed)}) "
                    f"could not be read as its own document; its pages are "
                    f"listed here"),
            pages=pages))


def _identify_bound(doc: Any, engine: Any, out: str, resume: bool,
                    ranges: Sequence[Any], spend: _Spend,
                    qa: List[QAEntry]) -> List[Dict[str, Any]]:
    """One small call per bound document: what it says it IS.

    Cached under the child's own folder, so a resumed run does not pay for
    it again, and the answer is needed BEFORE the parent's narrative reader
    runs -- a narrative told "pages 112-184 are a 2009 study by another
    firm" answers the previous-investigation questions differently from one
    told only that some pages are bound in.
    """
    from report_ingest.bound import identify_bound

    told: List[Dict[str, Any]] = []
    for row in ranges:
        path = os.path.join(out, BOUND_DIR, row.bound_id, "identity.json")
        blob = _cached(path, resume)
        if blob is not None:
            spend.reuse("identity", blob.get("cost"))
        else:
            mark = spend.mark()
            result = identify_bound(doc, row.pages, engine,
                                    title_hint=row.title)
            blob = result.to_dict()
            # A call that never answered is no answer to keep: the next run
            # asks again (a refusal by the ceiling, or any failure).
            if not result.failed:
                _keep(path, blob, spend, mark, qa,
                      f"bound.{row.bound_id}.identity",
                      [row.first_page, row.last_page])
            spend.add(blob.get("cost"))
            # A call sent that failed has no cost of its own to hand in.
            spend.settle(mark)
        identity = dict(blob.get("identity") or {})
        warning = str(blob.get("warning") or "")
        if warning:
            qa.append(QAEntry(
                kind="partial", where=f"bound.{row.bound_id}",
                detail=f"{warning}; the bound document is still read, with "
                       f"whatever the report it is bound into called it",
                pages=[row.first_page, row.last_page]))
        told.append({
            "bound_id": row.bound_id,
            "pages": row.spec,
            "title": identity.get("title") or row.title,
            "firm": identity.get("firm") or "",
            "date": identity.get("date") or "",
            "document_type": identity.get("document_type") or "",
            "kind": identity.get("kind") or row.kind,
            "same_site": identity.get("same_site") or "",
        })
    return told


def _read_bound(doc: Any, engine: Any, budgets: Budgets, out: str,
                resume: bool, report_id: str, facts: Sequence[Any],
                vision: Dict[int, Tuple[str, float]],
                templates: Dict[int, Any], labelling: "_Labelling",
                ranges: Sequence[Any], told: Sequence[Dict[str, Any]],
                record: ReportRecord,
                children: List[Tuple[ReportRecord, str]], spend: _Spend,
                qa: List[QAEntry], no_text: Sequence[int],
                di_pages: Sequence[int],
                counts_from_logs: str = "off",
                model_first: Optional[_ModelFirst] = None,
                readers: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Optional[Set[Tuple[int, str]]]]:
    """Each bound document round this same loop, into its own record.

    Returns, per bound document, every ``(page, reader)`` its pass read to
    the end, or ``None`` for one that could not be read at all.
    """
    by_id = {row["bound_id"]: row for row in told}
    reads: Dict[str, Optional[Set[Tuple[int, str]]]] = {}
    for row in ranges:
        folder = os.path.join(out, BOUND_DIR, row.bound_id)
        identity = by_id.get(row.bound_id, {})
        entry = BoundReport(
            bound_id=row.bound_id,
            title=str(identity.get("title") or row.title),
            firm=str(identity.get("firm") or ""),
            date=str(identity.get("date") or ""),
            kind=str(identity.get("kind") or row.kind),
            document_type=str(identity.get("document_type") or ""),
            pages=row.spec, first_page=row.first_page,
            last_page=row.last_page, n_pages=row.n_pages,
            said_by=list(row.said_by),
            folder=f"{BOUND_DIR}/{row.bound_id}",
            record_path=f"{BOUND_DIR}/{row.bound_id}/{OUTPUT_NAMES[0]}")
        mark = spend.mark()
        reads[row.bound_id] = None
        try:
            child, done = _one_bound(doc, engine, budgets, folder, resume,
                                     report_id, facts, vision, templates,
                                     labelling, row, identity, record, spend,
                                     no_text, di_pages, counts_from_logs,
                                     model_first, readers)
        except Exception as exc:              # one child, not the report
            spend.settle(mark)
            entry.read = False
            entry.report_id = _bound_report_id(report_id, row.bound_id)
            qa.append(QAEntry(
                kind="skipped", where=f"bound.{row.bound_id}",
                detail=f"the report bound in at pages {row.spec} could not "
                       f"be read as its own document: "
                       f"{type(exc).__name__}: {exc}; its pages stay listed "
                       f"here and nothing of it is in this record",
                pages=[row.first_page, row.last_page]))
            record.bound_documents.append(entry)
            continue
        # A model-first narrative read every page it was shown.
        reads[row.bound_id] = {(int(page), ITEM_READERS.get(item.kind, ""))
                               for item in done
                               for page in (getattr(item, "shown", None)
                                            or item.pages)}
        entry.report_id = child.document.report_id
        entry.counts = child.counts()
        record.bound_documents.append(entry)
        children.append((child, folder))
        qa.append(QAEntry(
            kind="note", where=f"bound.{row.bound_id}",
            detail=(f"pages {row.spec} are a report bound inside this one "
                    f"({', '.join(row.said_by) or 'triage'} said so) and "
                    f"were read into their own record at "
                    f"{entry.record_path}; its "
                    f"{entry.counts.get('investigations', 0)} "
                    f"exploration(s) and {entry.counts.get('lab_tests', 0)} "
                    f"laboratory test(s) are ITS and are not in this "
                    f"record"),
            pages=[row.first_page, row.last_page]))
    return reads


def _bound_report_id(report_id: str, bound_id: str) -> str:
    """The child's own handle: the parent's, then its own, or just its own."""
    return f"{report_id}.{bound_id}" if report_id else bound_id


def _one_bound(doc: Any, engine: Any, budgets: Budgets, folder: str,
               resume: bool, report_id: str, facts: Sequence[Any],
               vision: Dict[int, Tuple[str, float]],
               templates: Dict[int, Any], labelling: "_Labelling",
               row: Any, identity: Dict[str, Any], parent: ReportRecord,
               spend: _Spend, no_text: Sequence[int],
               di_pages: Sequence[int],
               counts_from_logs: str = "off",
               model_first: Optional[_ModelFirst] = None,
               readers: Optional[Dict[str, Any]] = None
               ) -> Tuple[ReportRecord, List[Any]]:
    """One bound document as its own record, on the parent's page numbers.

    THE RULES ARE RE-RUN, THE OTHER VOTERS ARE NOT. planlens labels every
    page of a nested document ``appended_report`` -- that is the rules'
    answer about the BINDING, and it is right about the parent. So the rules
    are re-run over the child's pages REBASED to 0, which is what makes a
    cover a cover and a boring log a boring log again; the vision pass and
    the printed-form recogniser already answered about these pages and their
    answers are reused verbatim, because a page's picture does not change
    when you stop calling it an appendix.

    A PAGE THE FILE'S OWN PASS GAVE A READER KEEPS ITS LABEL. Where the
    parent's final label (after its vote AND its label review) is a log, a
    laboratory sheet, a printout or a sounding, the parent handed the page
    over to be read here (:func:`_outside_bound`), and the re-run would
    otherwise decide it again without the review: rules rebased onto a short
    window call a run of sheets near its start prose (R15, BACKLOG RI-14),
    and nobody would read them. The re-run decides only the pages the parent
    called ``appended_report`` or something no reader reads. Pipeline
    judgment, not a printed rule: the parent's label had more evidence.

    Everything the child stores is on the PARENT file's page numbers. There
    is one PDF. Returns the record and the work items its readers read to
    the end.

    MODEL FIRST (BACKLOG RI-87), the child labels its pages by sections in a
    call of its own, over the descriptions its pages already have
    (:func:`_bound_range_labels`); that label settles each page it gives,
    and the rules re-run decide the rest. Its pages are then read model
    first in its own window, as the parent's are.
    """
    import dataclasses

    from planlens.document.roles import assign, build_items, outline_from
    from report_ingest.label_vote import Voter, combine
    from report_ingest.reconciler import reconcile

    os.makedirs(os.path.join(folder, "items"), exist_ok=True)
    window = [int(p) for p in row.pages]
    at = {page: index for index, page in enumerate(window)}

    inner = [dataclasses.replace(f, page=at[f.page]) for f in facts
             if f.page in at]
    inner_roles = assign(inner)
    rules = {window[r.page]: (r.role, float(r.confidence))
             for r in inner_roles}

    def lines_of(index: int) -> List[str]:
        return [line.text for line in doc.page(index, tables=False).lines
                if (line.text or "").strip()]

    inner_spend = _Spend(spend.tally)
    qa_labels: List[QAEntry] = []
    ranged: Dict[int, str] = {}
    if model_first is not None:
        ranged = _bound_range_labels(
            engine, folder, resume, window, rules,
            outline_from([f for f in facts if f.page in at],
                         [dataclasses.replace(r, page=window[r.page])
                          for r in inner_roles], lines_of),
            labelling, model_first.descriptions, inner_spend, qa_labels)

    table = labelling.resolve()
    choices = []
    for page in window:
        role, confidence = rules.get(page, ("", None))
        if role == APPENDED_LABEL:
            role, confidence = "", None
        seen = vision.get(page)
        # INSIDE a bound document, ``appended_report`` is not an answer about
        # what a page is -- it is the whole document's situation, and it is
        # this record's premise. A voter that says it has no view here, which
        # is different from voting for it and being outvoted: it must not
        # count as a disagreement either.
        if seen and seen[0] == APPENDED_LABEL:
            seen = None
        match = templates.get(page)
        choices.append(combine(
            Voter("rules", role, confidence),
            Voter("vision", seen[0], seen[1]) if seen else None,
            Voter("template", "", match.confidence, match.family)
            if match else None,
            policy=labelling.policy, trust_table=table, page=page,
            ranges=Voter("ranges", ranged[page]) if page in ranged
            else None))
    held = {pl.page: pl.label for pl in parent.page_labels
            if pl.page in at and ITEM_READERS.get(pl.label)
            not in (None, "narrative")}
    kept: List[Tuple[int, str, str]] = []
    for index, choice in enumerate(choices):
        label = held.get(choice.page)
        if label is None or label == choice.label:
            continue
        kept.append((choice.page, choice.label or "none", label))
        choices[index] = dataclasses.replace(
            choice, label=label,
            why=f"kept the whole file's label {label}, a page a reader reads")
    labels = {c.page: c.label for c in choices}
    split = split_pages(choices)

    # The items, built on the child's own numbering and translated back.
    voted = [dataclasses.replace(f, page=at[f.page]) for f in facts
             if f.page in at]
    roles = _relabel([r for r in inner_roles],
                     {at[p]: label for p, label in labels.items()},
                     {at[c.page]: c.confidence for c in choices})
    # The split runs on the PARENT file's numbering, because that is where
    # the pages the signature is read off actually live.
    items = split_lab_items(doc, split_log_items(doc, split_calc_items(
        doc, [_rebase_item(item, window)
              for item in build_items(voted, roles)])))
    every = list(items)
    if budgets.max_items:
        # The same ceiling as the parent's, applied to this document: a
        # smoke run over a big report must not turn into a full run of the
        # report bound inside it.
        items = items[:int(budgets.max_items)]

    outline = outline_from(
        [f for f in facts if f.page in at],
        _relabel([dataclasses.replace(r, page=window[r.page])
                  for r in inner_roles], labels,
                 {c.page: c.confidence for c in choices}),
        lines_of)

    child = ReportRecord()
    child.document = DocumentFacts(
        report_id=_bound_report_id(report_id, row.bound_id),
        n_pages=len(window),
        page_roles=_role_counts(labels),
        scan_fraction=_scan_fraction(facts, at),
        di_pages=len([p for p in di_pages if p in at]),
        planlens_version=_planlens_version(),
        workflow=_bound_workflow(labels),
        label_policy=labelling.policy,
        label_split_pages=len(split),
        review_mode="none")
    child.parent = ParentReport(
        report_id=parent.document.report_id,
        bound_id=row.bound_id,
        pages=row.spec, first_page=row.first_page,
        last_page=row.last_page, n_pages=row.n_pages,
        record_path=f"../../{OUTPUT_NAMES[0]}")
    child.page_labels = [_page_label(c, {}) for c in choices]
    for entry in child.page_labels:
        if entry.page in held:
            entry.settled_by = "parent"

    qa: List[QAEntry] = [QAEntry(
        kind="note", where="bound",
        detail=(f"this record is a report bound inside another one, at pages "
                f"{row.spec} of that file; every page number in it is a page "
                f"of that same file. The page labels are the vote's "
                + ("(its own pages labelled by sections, and planlens' rules "
                   "re-run over these pages alone where the sections left a "
                   "page unsettled)" if model_first is not None else
                   "(planlens' rules re-run over these pages alone, with the "
                   "whole file's vision pass reused)")
                + ", except that a page the whole file's pass gave a reader "
                  "keeps that label; the label review does not run over a "
                  "bound document"),
        pages=[row.first_page, row.last_page])] + qa_labels
    for name, value in (("title", identity.get("title")),
                        ("firm", identity.get("firm")),
                        ("date", identity.get("date"))):
        if not value:
            qa.append(QAEntry(
                kind="partial", where=f"bound.identity.{name}",
                detail=f"this bound report prints no {name} on the pages it "
                       f"was identified from",
                pages=[row.first_page]))
    if kept:
        qa.append(QAEntry(
            kind="note", where="bound.labels",
            detail=(f"{len(kept)} page(s) keep the label the whole file's "
                    f"pass gave them (its vote and its label review), not "
                    f"the one the re-run over these pages alone would: the "
                    f"file handed them over to be read here by that reader"),
            values=[f"{page} {was} -> {now}" for page, was, now in kept],
            pages=[page for page, _was, _now in kept]))
    for choice in choices:
        if choice.agreed:
            continue
        settled = (f"keeps the whole file's label '{choice.label}'"
                   if choice.page in held else
                   f"calls page {choice.page} '{choice.label}' under the "
                   f"'{choice.policy}' policy")
        qa.append(QAEntry(
            kind="label_disagreement", where=f"labels.page{choice.page}",
            detail=(f"the page labellers did not agree, and a bound document "
                    f"does not get the label review; the record {settled}"),
            values=[_voter_line(v) for v in choice.voters],
            pages=[choice.page]))

    unresolved: List[Dict[str, Any]] = []
    started = time.time()
    # A report bound inside a report bound inside a report is where this
    # stops: the child's own ``appended_report`` pages are listed in its QA
    # and not read again. One level is what the corpus has and what a record
    # of a record of a record would be worth.
    if model_first is None:
        done = _read_items(doc, engine, budgets, folder, resume, (),
                           child.document.report_id, items, labels, outline,
                           child, inner_spend, qa, unresolved,
                           child.document.workflow,
                           [p for p in no_text if p in at],
                           [p for p in di_pages if p in at],
                           window=window, readers=readers)
    else:
        mine, every, items, rest = _model_first_plan(
            doc, model_first, labels, window, every, budgets.max_items)
        done = _read_model_first(
            doc, model_first, folder, resume, child.document.report_id,
            labels, outline, child, inner_spend, qa, unresolved,
            child.document.workflow, mine, window)
        done += _read_items(doc, engine, budgets, folder, resume, (),
                            child.document.report_id, rest, labels, outline,
                            child, inner_spend, qa, unresolved,
                            child.document.workflow,
                            [p for p in no_text if p in at],
                            [p for p in di_pages if p in at],
                            window=window)
    _save(os.path.join(folder, "labels.json"),
          _labels_blob(choices, split, {},
                       dataclasses.replace(labelling, review_mode="none"),
                       len(window)))
    _seed_from_identity(child, identity, qa, row)

    reconcile(child, labels=labels, items=items,
              no_text_pages=[p for p in no_text if p in at],
              di_pages=[p for p in di_pages if p in at],
              reader_unresolved=unresolved, doc=doc, outline=outline)
    if model_first is None:
        qa.extend(missing_holes_qa(doc, items, child))      # RI-117
    _counts_from_logs(child, qa, counts_from_logs, every, done)
    child.project = _project_from(child)
    child.qa.extend(qa)
    child.document.model_calls = inner_spend.calls
    child.document.input_tokens = inner_spend.input_tokens
    child.document.output_tokens = inner_spend.output_tokens
    child.document.dollars = round(inner_spend.dollars, 5)
    child.document.seconds = round(time.time() - started, 1)
    child.document.resumed = inner_spend.resumed()
    spend.absorb(inner_spend)
    return child, done


def _seed_from_identity(child: ReportRecord, identity: Dict[str, Any],
                        qa: List[QAEntry], row: Any) -> None:
    """What the bound document's own cover said, where the prose said nothing.

    A data report bound into a design report often has no narrative at all --
    its workflow is ``appendix_only`` and the reader that answers the owner's
    schemas never runs. Its cover still names the firm and the date, and the
    identity call read them off it. Filling ONLY the nulls keeps the
    narrative's answer wherever there is one, and the QA entry says which
    fields came off a cover rather than out of the prose, because those two
    are not the same kind of answer.
    """
    from report_ingest.model import DOCUMENT_TYPE_VALUES

    filled: List[str] = []
    kind = str(identity.get("document_type") or "")
    if child.general.documentType is None and kind in DOCUMENT_TYPE_VALUES:
        child.general.documentType = kind          # type: ignore[assignment]
        filled.append("documentType")
    firm = str(identity.get("firm") or "").strip()
    if not child.general.geotechnicalEngineerFirm and firm:
        child.general.geotechnicalEngineerFirm = firm
        filled.append("geotechnicalEngineerFirm")
    date = str(identity.get("date") or "").strip()
    if not child.natural_hazards.reportDate and date:
        child.natural_hazards.reportDate = date
        filled.append("reportDate")
    if filled:
        qa.append(QAEntry(
            kind="note", where="bound.identity",
            detail=f"{', '.join(filled)} came off this bound report's own "
                   f"cover or letterhead rather than out of its narrative",
            pages=[row.first_page]))


def _rebase_item(item: Any, window: Sequence[int]) -> Any:
    """One work item's pages back onto the parent file's numbering."""
    import dataclasses

    return dataclasses.replace(
        item, pages=[int(window[p]) for p in item.pages],
        evidence={**dict(item.evidence),
                  "first_page": int(window[item.pages[0]])}
        if item.pages else dict(item.evidence))


def _role_counts(labels: Dict[int, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label in labels.values():
        out[label] = out.get(label, 0) + 1
    return out


def _scan_fraction(facts: Sequence[Any], at: Dict[int, int]) -> float:
    rows = [f for f in facts if f.page in at]
    if not rows:
        return 0.0
    scanned = sum(1 for f in rows if getattr(f, "kind", "") == "scanned")
    return scanned / len(rows)


def _bound_workflow(labels: Dict[int, str]) -> str:
    """A bound document's workflow, from its own pages and no model call.

    There is nothing for a triage pass to choose here: the file was already
    triaged, and what is left to decide is whether this document has a
    narrative to read. ``appendix_only`` is the honest answer for a data
    report bound into a design report, and it is what stops the narrative
    reader answering the owner's schemas off a boring log.
    """
    return "standard" if "narrative" in set(labels.values()) \
        else "appendix_only"


def _project_from(record: ReportRecord) -> Project:
    """The project block, from what the narrative answered about it."""
    general = record.general
    return Project(name=general.projectName or "",
                   number=general.projectNumber or "",
                   client=general.primeAe or general.primeContractor or "")


def _read_items(doc: Any, engine: Any, budgets: Budgets, out: str,
                resume: bool, questions: Sequence[str], report_id: str,
                items: Sequence[Any], labels: Dict[int, str], outline: Any,
                record: ReportRecord, spend: _Spend, qa: List[QAEntry],
                unresolved: List[Dict[str, Any]], workflow: str,
                no_text: Sequence[int], di_pages: Sequence[int],
                bound_told: Sequence[Dict[str, Any]] = (),
                bound_pages: Collection[int] = (),
                window: Optional[Sequence[int]] = None,
                narrative_key: str = "",
                readers: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Every work item through its reader, into ``record``.

    Returns the items whose reader ran to the end (or whose reading came
    off disk): what a caller may say was READ, as against what was only
    listed. ``narrative_key`` marks the narrative's cache file when what
    its reader is shown differs from the default (``narrative_skips_bound``),
    so a resumed run never takes one reading for the other. ``readers``
    gives today's reader of a stage its own engine (:func:`_today_readers`);
    a reader not in it reads on ``engine``.
    """
    from report_ingest.model import (
        Calculation, GeneralFacts, Investigation, LabTest, NarrativeFacts,
        NaturalHazardFacts,
    )

    body = [page for page, role in sorted(labels.items())
            if role in ("narrative", "figure", "plan", "profile", "toc")]
    blind = set(int(p) for p in no_text) - set(int(p) for p in di_pages)
    bound_pages = set(int(p) for p in bound_pages)
    prose, prose_pages = _narrative_items(items, workflow, blind, qa)
    done: List[Any] = []

    for item in items:
        reader = ITEM_READERS.get(item.kind)
        pages = [int(p) for p in item.pages]
        if reader is None:
            if item.kind == "appended_report":
                # Whether these pages became a record of their own is
                # decided by the bound pass, not here: this item is one run
                # of ``appended_report`` pages and a bound document may be
                # the union of this run and what triage said around it. A
                # run inside a range is said to be read only once its bound
                # document was (:func:`_handed_over`). A run under the page
                # floor for a document is in no range, stays this report's
                # and is read by nobody.
                if pages and set(pages) <= bound_pages:
                    continue
                qa.append(QAEntry(
                    kind="skipped", where="items.appended_report",
                    detail="a report bound inside this one was not read as "
                           "its own document; its pages are listed here",
                    pages=pages))
            continue

        if reader == "narrative":
            # Read ONCE, at the first of its items, over all of them; the
            # ones that are not read were given their QA entry above.
            if not prose or item is not prose[0]:
                continue
            pages = prose_pages

        key = item.id if reader != "narrative" or len(prose) == 1 \
            else f"{prose[0].id}-{prose[-1].id}"
        if reader == "narrative":
            key += narrative_key
        elif item.evidence.get("bound_pages_handed"):
            # Its pages inside a bound document were handed over; a reading
            # saved before that, of the whole item, is not this one.
            key += "-own"
        if reader == "lab" and budgets.lab_sheets_per_call is not None:
            # A reading split by sheets per call (RI-116) is not the whole
            # item's, and a resumed run must not take one for the other.
            from report_ingest.lab_reader import split_key
            key += split_key(len(pages), budgets.lab_sheets_per_call)
        path = os.path.join(out, "items", f"{key}.json")
        cached = _cached(path, resume)
        mark = spend.mark()
        where = f"items.{item.kind}.{key}"
        item_engine = (readers or {}).get(reader, engine)
        try:
            if reader == "narrative":
                blob = cached or _read_narrative(
                    doc, pages, item_engine, budgets, outline, report_id, body,
                    questions, bound_told, window)
                if cached is None:
                    _keep(path, blob, spend, mark, qa, where, pages)
                spend.hand_in("items", blob, cached)
                record.general = GeneralFacts.model_validate(blob["general"])
                record.natural_hazards = NaturalHazardFacts.model_validate(
                    blob["natural_hazards"])
                record.narrative = NarrativeFacts.model_validate(
                    blob["facts"])
                unresolved.extend(blob.get("unresolved") or [])
            elif reader == "log":
                blob = cached or _read_log(doc, pages, item_engine, budgets,
                                           item, report_id)
                if cached is None:
                    _keep(path, blob, spend, mark, qa, where, pages)
                spend.hand_in("items", blob, cached)
                record.investigations.append(
                    Investigation.model_validate(blob["investigation"]))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
                votes = (blob.get("visual_scales") or {}).get("layer_votes")
                if votes:
                    from report_ingest.visual_scales import layer_votes_qa
                    inv = blob.get("investigation") or {}
                    qa.extend(layer_votes_qa(
                        votes, f"investigations["
                               f"{inv.get('investigation_id', '')}].layers",
                        inv.get("depth_unit") or "", pages))
            elif reader in ("cpt", "dcp"):
                blob = cached or _read_sounding(doc, pages, engine, budgets,
                                                item, report_id, reader)
                if cached is None:
                    _keep(path, blob, spend, mark, qa, where, pages)
                spend.hand_in("items", blob, cached)
                if blob.get("investigation"):
                    record.investigations.append(
                        Investigation.model_validate(blob["investigation"]))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
                if blob.get("code_trace"):
                    from report_ingest.visual_scales import trace_check_qa
                    qa.extend(trace_check_qa(blob["code_trace"], pages))
            elif reader == "calc":
                blob = cached or _read_calc(doc, pages, item_engine, budgets,
                                            item, report_id)
                if cached is None:
                    _keep(path, blob, spend, mark, qa, where, pages)
                spend.hand_in("items", blob, cached)
                if blob.get("calculation"):
                    record.calculations.append(
                        Calculation.model_validate(blob["calculation"]))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
            else:
                blob = cached or _read_lab(doc, pages, item_engine, budgets,
                                           item, report_id)
                if cached is None:
                    _keep(path, blob, spend, mark, qa, where, pages)
                spend.hand_in("items", blob, cached)
                for test in blob.get("tests") or []:
                    record.lab_tests.append(LabTest.model_validate(test))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
                if blob.get("plot_check"):
                    from report_ingest.visual_scales import plot_check_qa
                    qa.extend(plot_check_qa(blob["plot_check"], pages))
            spend.reading(cached is not None and blob is cached)
            done.extend(prose if reader == "narrative" else [item])
        except Exception as exc:                 # one item must not stop the run
            qa.append(QAEntry(
                kind="skipped", where=f"items.{item.kind}",
                detail=f"{key} could not be read: "
                       f"{type(exc).__name__}: {exc}",
                pages=pages))
        finally:
            spend.settle(mark)
    return done


def _narrative_items(items: Sequence[Any], workflow: str, blind: set,
                     qa: List[QAEntry]) -> Tuple[List[Any], List[int]]:
    """The narrative work items to read, and all their pages as one set.

    THE NARRATIVE IS READ ONCE. planlens starts a new work item wherever a
    run of narrative pages is broken -- by a figure, a table page, a tab --
    so one report's prose is often several items. Each used to be read on
    its own, over nearly the same pages (the reader adds the front matter
    and the narrative pages it finds itself), and each answer REPLACED the
    one before: the record kept the last item's, paid for every one and
    carried every read's unsettled lines over again (BACKLOG RI-14). Now
    they are one read. A narrative too long for one call is cut into parts
    by the reader and merged by its own rule: the first answer kept, lists
    unioned, every disagreement between two parts recorded.

    The items that are not read get their QA entry here, each as before: no
    narrative reader on an appendix-only file, and none over items whose
    every page is a scan with no text.
    """
    readable: List[Any] = []
    for item in items:
        if ITEM_READERS.get(item.kind) != "narrative":
            continue
        pages = [int(p) for p in item.pages]
        if workflow == "appendix_only":
            qa.append(QAEntry(
                kind="skipped", where="items.narrative",
                detail="this file is appendix or figure material, so the "
                       "narrative reader did not run",
                pages=pages))
            continue
        if set(pages) <= blind:
            qa.append(QAEntry(
                kind="unreadable", where="items.narrative",
                detail="every narrative page is a scan with no reliable "
                       "text and no Azure Document Intelligence result; "
                       "the narrative reader reads text and was not run",
                pages=pages))
            continue
        readable.append(item)
    pages = sorted({int(p) for item in readable for p in item.pages})
    if len(readable) > 1:
        qa.append(QAEntry(
            kind="note", where="items.narrative",
            detail=(f"the narrative is {len(readable)} work items, broken by "
                    f"other pages ({', '.join(i.id for i in readable)}); "
                    f"it was read once, over all of them"),
            pages=pages))
    return readable, pages


#: The work-item kind whose logs answer each of the five exploration fields.
_FIELD_ITEMS: Dict[str, str] = {
    "boringCount": "boring_log", "boringDictionary": "boring_log",
    "testPitCount": "test_pit_log", "testPitDictionary": "test_pit_log",
    "cptCount": "cpt_log",
}


def _counts_from_logs(record: ReportRecord, qa: List[QAEntry], mode: str,
                      items: Sequence[Any], read: Sequence[Any]) -> None:
    """The five exploration answers, against the logs this record read.

    The narrative reader's own rule
    (:func:`~report_ingest.narrative_reader.settled_explorations`), run HERE
    rather than inside the reader: the narrative is read before the logs,
    and a resumed run takes it off disk, so only here are both in hand on
    every run. And AFTER the reconciler, so that its ``count_mismatch``
    entries compare what the narrative SAID with what was read; the numbers
    the narrative stated stay in ``narrative.stated_counts``. Nothing runs
    on a record whose narrative was not read: these are the narrative's
    answers, and an appendix-only record does not answer the schemas.
    See :data:`COUNT_MODES`.

    ONLY WHAT THE LOGS READ CAN VOUCH FOR. ``items`` are every work item of
    this record (the item ceiling's cut included) and ``read`` the ones read
    to the end. A 0 goes in only where the record has NO work item of that
    kind: a test-pit log that failed to read, a cone item that gave nothing
    and one the ceiling cut are all pits or cones that exist. Any other
    answer goes in only where every item of its kind was read. A field held
    back is listed in the note, with why.
    """
    from report_ingest.narrative_reader import (
        DETERMINISTIC_FIELDS, explorations_found, settled_explorations,
    )

    if mode == "off" or not record.narrative.pages:
        return
    stated = {name: getattr(record.general, name, None)
              for name in DETERMINISTIC_FIELDS}
    changed = settled_explorations(
        stated, explorations_found(record.investigations),
        overrule=mode == "settle")
    if mode == "zeros":
        changed = {name: value for name, value in changed.items()
                   if value == 0}
    done = {item.id for item in read}
    held: Dict[str, str] = {}
    for name, value in list(changed.items()):
        kind = _FIELD_ITEMS[name]
        mine = [item for item in items if item.kind == kind]
        unread = [item for item in mine if item.id not in done]
        if value == 0 and mine:
            held[name] = (f"the file has {len(mine)} {kind} item(s), "
                          f"{len(unread)} not read")
        elif value != 0 and unread:
            held[name] = (f"{len(unread)} of {len(mine)} {kind} item(s) "
                          f"not read")
        else:
            continue
        del changed[name]
    if not changed and not held:
        return
    for name, value in changed.items():
        setattr(record.general, name, value)

    def show(value: Any) -> str:
        if value in (None, []):
            return "not stated"
        return ", ".join(value) if isinstance(value, list) else str(value)

    filled = [name for name in changed if stated[name] in (None, [])]
    overruled = [name for name in changed if name not in filled]
    zeros = [name for name in filled if changed[name] == 0]
    others = [name for name in filled if name not in zeros]
    said = []
    if zeros:
        said.append(f"{', '.join(zeros)} is 0: the narrative did not say, "
                    f"and the logs read hold none of that kind while they "
                    f"hold others (none means 0)")
    if others:
        said.append(f"{', '.join(others)} came from the logs read, where the "
                    f"narrative did not say")
    if overruled:
        said.append(f"{', '.join(overruled)} are the logs' answer over the "
                    f"narrative's ({mode!r}); the narrative's is in the "
                    f"count_mismatch entries and narrative.stated_counts")
    if held:
        said.append(f"{', '.join(held)} stay as the narrative gave them: "
                    f"logs of that kind are in the file that the logs read "
                    f"do not account for")
    qa.append(QAEntry(
        kind="note", where="general.explorations",
        detail="; ".join(said),
        values=[f"{name}: {show(stated[name])} -> {show(value)}"
                for name, value in changed.items()]
        + [f"{name}: held ({why})" for name, why in held.items()]))


def _vote_qa(qa: List[QAEntry], blob: Dict[str, Any],
             pages: Sequence[int]) -> None:
    """The two voters' splits, and what was kept from the floor, into QA.

    A reader's record is the merge of its floor (the grid, the tables) and
    the model's answer since 5.23.0. Every slot the two split on becomes a
    ``disagreement`` entry carrying both values, which one the record holds
    and the two confidences -- the trigger for a second look the owner
    asked for. A floor value the model did not return is a ``note``: it is
    in the record, and a reviewer should know the model never saw it.
    """
    for row in blob.get("disagreements") or []:
        conf = row.get("confidence") or {}
        why = str(row.get("why") or "").strip()
        why = (why[0].upper() + why[1:] + ("" if why.endswith(".") else ".")
               if why else "")
        kept_side = str(row.get("kept") or "floor")
        # Neither voter's reading: both were ruled out (RI-47). Both: the
        # two stand side by side in the record, two water readings or two
        # lab kinds (RI-70).
        carries = (kept_side if kept_side in ("neither", "both")
                   else f"the {kept_side}'s")
        qa.append(QAEntry(
            kind="disagreement", where=str(row.get("where") or ""),
            detail=(f"{row.get('what', '')}: {row.get('floor_method', 'floor')} "
                    f"read {row.get('floor', '')!s}, "
                    f"{row.get('model_method', 'model')} read "
                    f"{row.get('model', '')!s}; the record carries "
                    f"{carries}. {why}").strip(),
            values=[f"{row.get('floor_method', 'floor')} {row.get('floor', '')} "
                    f"({float(conf.get('floor', 0.0)):.2f})",
                    f"{row.get('model_method', 'model')} "
                    f"{row.get('model', '')} "
                    f"({float(conf.get('model', 0.0)):.2f})"],
            pages=[int(row["page"])] if row.get("page") is not None
            else [int(p) for p in pages]))
    kept = blob.get("kept") or []
    if kept:
        qa.append(QAEntry(
            kind="note", where="floor",
            detail=(f"{len(kept)} value(s) the model did not return were "
                    f"kept from the floor (the grid's rows or the page's "
                    f"tables): " + "; ".join(
                        str(k.get("what") or "") for k in kept[:8])
                    + (" ..." if len(kept) > 8 else "")),
            pages=sorted({int(k["page"]) for k in kept
                          if k.get("page") is not None})
            or [int(p) for p in pages]))
    # A blow record of more counts than drives, refused (RI-130), and a
    # depth the model copied off the starting record (RI-131): both said,
    # so a reviewer knows what the record does not hold and which depths
    # rest on the floor alone.
    for name, detail in (
            ("refused", "blow record(s) of more counts than a driven record "
                        "has drives were refused (not kept as the record's "
                        "blows or as an alternative): "),
            ("echoed", "depth(s) the model returned as the floor's own "
                       "worked-out value (to more decimals than a log "
                       "prints) are one reading, the floor's, at its method "
                       "and confidence: ")):
        rows = blob.get(name) or []
        if not rows:
            continue
        qa.append(QAEntry(
            kind="note", where="floor",
            detail=(f"{len(rows)} {detail}" + "; ".join(
                str(r.get("what") or "") for r in rows[:8])
                + (" ..." if len(rows) > 8 else "")),
            pages=sorted({int(r["page"]) for r in rows
                          if r.get("page") is not None})
            or [int(p) for p in pages]))


def _read_narrative(doc, pages, engine, budgets, outline, report_id, body,
                    questions, bound_told=(), window=None) -> Dict[str, Any]:
    from report_ingest.narrative_reader import read_narrative
    result = read_narrative(doc, pages, engine, budget=budgets.narrative,
                            outline=outline, report_id=report_id,
                            body_pages=body or pages, questions=questions,
                            bound_documents=bound_told, window=window)
    return result.to_dict()


def _read_log(doc, pages, engine, budgets, item, report_id) -> Dict[str, Any]:
    from report_ingest.log_reader import read_log
    extra = ({"max_request_tokens_in": budgets.max_request_tokens_in}
             if budgets.max_request_tokens_in else {})
    result = read_log(doc, pages, engine, budget=budgets.log,
                      item_title=item.title or "", report_id=report_id,
                      **extra)
    return result.to_dict()


def _read_lab(doc, pages, engine, budgets, item, report_id) -> Dict[str, Any]:
    from report_ingest.lab_reader import read_lab_sheet
    # A part cut from a lab item carries the boring its first part prints
    # where its own pages print none (RI-150, split_lab_items).
    carried = (getattr(item, "evidence", None) or {}).get("carried_link")
    extra = {"carried_link": carried} if carried else {}
    result = read_lab_sheet(doc, pages, engine, budget=budgets.lab,
                            item_title=item.title or "", report_id=report_id,
                            sheets_per_call=budgets.lab_sheets_per_call,
                            **extra)
    return result.to_dict()


def _read_calc(doc, pages, engine, budgets, item, report_id
               ) -> Dict[str, Any]:
    from report_ingest.calc_reader import read_calculation
    result = read_calculation(doc, pages, engine, budget=budgets.calc,
                              item_title=item.title or "",
                              report_id=report_id)
    return result.to_dict()


def _read_sounding(doc, pages, engine, budgets, item, report_id, kind
                   ) -> Dict[str, Any]:
    from report_ingest.sounding_reader import read_sounding
    result = read_sounding(doc, pages, engine, kind=kind,
                           budget=budgets.sounding,
                           item_title=item.title or "", report_id=report_id)
    return result.to_dict()


# ---------------------------------------------------------------------------
# model-first ingest (BACKLOG RI-87), only under GEOTECH_INGEST_MODEL_FIRST=on
# ---------------------------------------------------------------------------

def _model_first_run(source: Any, engine: Any, tally: _Tally,
                     labelling: "_Labelling", stage_engines: Any,
                     page_route: Optional[str],
                     questions: Optional[Sequence[str]],
                     qa: List[QAEntry],
                     max_request_tokens_in: Optional[int] = None
                     ) -> _ModelFirst:
    """The run's model-first settings, and the one QA note that names them.

    The range voter is switched on (``plain`` unless an arm was named), and
    the vision voter and the label review off. Each stage's engine is the
    one ``stage_engines`` gives (:func:`ingest_report`), counted on the
    run's tally; its route is resolved here, so a ``pdf`` route an engine
    cannot send is refused before anything is read. A per-request limit
    (``Budgets.max_request_tokens_in``, RI-121) goes to every reader."""
    from report_ingest.model_first_pages import page_route as route_setting
    from report_ingest.model_first_pages import resolve_route
    from report_ingest.model_first_read import STAGE_MODELS

    asked = route_setting(page_route)
    labelling.model_first = True
    labelling.review_mode = "none"
    labelling.range_arm = labelling.range_arm or "plain"
    engines: Dict[str, Any] = {}
    routes: Dict[str, str] = {}
    for stage, model in STAGE_MODELS.items():
        if callable(stage_engines):
            given = stage_engines(model)
        else:
            given = (stage_engines or {}).get(stage)
        engines[stage] = tally.wrap(given) if given is not None else engine
        routes[stage] = resolve_route(
            asked, bool(getattr(engines[stage], "accepts_pdf", False)))
    if _is_document(source):
        pdf = None
    elif isinstance(source, (bytes, bytearray)):
        pdf = bytes(source)
    else:
        pdf = str(source)
    limit = int(max_request_tokens_in) if max_request_tokens_in else None
    note = QAEntry(
        kind="note", where="model_first",
        detail=(f"read model first ({MODEL_FIRST_ENV}=on): each group of "
                f"pages was read in one call and the model's answer is the "
                f"record; the pages were labelled by sections (the range "
                f"voter, arm {labelling.range_arm!r}), and the vision voter, "
                f"the label review and the floors did not run"
                + ("; the caller's own questions were not asked: the "
                   "narrative's call carries its stage's instruction only"
                   if questions else "")
                + (f"; each request was sized against a limit of {limit:,} "
                   f"input tokens before it was sent" if limit else "")),
        values=[f"{stage}: {getattr(engines[stage], 'name', '') or 'engine'}"
                f", pages as {routes[stage]}" for stage in STAGE_MODELS])
    qa.append(note)
    return _ModelFirst(engines=engines, routes=routes, asked=asked, pdf=pdf,
                       max_request_tokens_in=limit, note=note)


def _note_cuts(model_first: _ModelFirst, qa: List[QAEntry]) -> None:
    """The run's QA note, once a run has cut groups for the request limit
    (RI-121): it says how many, beside the limit it names. A run that cut
    none keeps the note as it is."""
    if not model_first.cut or model_first.note is None:
        return
    n = model_first.cut
    said = (f", and {n} group{'s' if n > 1 else ''} estimated over it "
            f"{'were' if n > 1 else 'was'} cut into smaller groups, each "
            f"cut listed in a note of its own")
    for at, entry in enumerate(qa):
        if entry is model_first.note:
            qa[at] = QAEntry(kind=entry.kind, where=entry.where,
                             detail=entry.detail + said,
                             values=list(entry.values),
                             pages=list(entry.pages))
            model_first.note = qa[at]
            return


def _over_limit_qa(blob: Dict[str, Any], where: str) -> List[QAEntry]:
    """One note per request a model-first result found over the request
    limit (RI-121; ``over_limit``, none without a limit): a group cut in
    two before it was sent, or one page sent as it is."""
    from report_ingest.checks import _span

    out: List[QAEntry] = []
    for row in blob.get("over_limit") or []:
        pages = [int(p) for p in row.get("pages") or []]
        if not pages:
            continue
        said = (f"{_span(pages)}: the request was estimated at "
                f"{int(row.get('estimated_tokens_in') or 0):,} input "
                f"tokens, over the limit of {int(row.get('limit') or 0):,} "
                f"a request")
        into = [list(p) for p in row.get("cut_into") or [] if p]
        if into:
            said += (", so it was cut into "
                     + " and ".join(_span(p) for p in into)
                     + " before it was sent, each read as a group of its "
                       "own")
        else:
            said += "; one page cannot be cut, so it was sent as it is"
        out.append(QAEntry(kind="note", where=f"{where}.request_limit",
                           detail=said, pages=pages))
    return out


def _placer(doc: Any) -> Any:
    """``place(stage, record, pages, doc, rulers=None)`` for the readers:
    each answer's record placed on its pages by the words printed there
    (:mod:`report_ingest.locate`), over one reading of each page's words.

    The record a reader hands it is its builder's, whose pages are already
    the report's: the builder read the answer's cited pages once, as pages
    or as positions in the call (:func:`report_ingest.checks.pdf_local`).
    So locate is told to take them as they stand, rather than decide again
    from the cites the record kept, which could decide otherwise, or turn a
    position twice. A laboratory test placed on another page of its call
    takes that page as its ``pages``. Returns what locate found
    (``Placements.to_dict()``)."""
    from report_ingest import locate
    from report_ingest.checks import PageNumbers

    words = locate.PageWords(doc)

    def place(stage: str, record: Any, pages: Sequence[int], _doc: Any,
              rulers: Any = None) -> Dict[str, Any]:
        as_built = PageNumbers(tuple(int(p) for p in pages))
        if stage == "logs":
            found = locate.place_investigation(words, record, pages,
                                               rulers=rulers,
                                               numbers=as_built)
        elif stage == "lab":
            found = locate.place_lab_tests(words, record, pages,
                                           numbers=as_built)
            # One placement per test, in order: a test found on another
            # page of its call is on that page.
            for test, placed in zip(record, found.items):
                if placed.status == locate.MOVED:
                    test.pages = [int(placed.page)]
        elif stage == "calc":
            found = locate.place_calculation(words, record, pages,
                                             numbers=as_built)
        else:
            found = locate.place_citations(words, record, pages)
        return found.to_dict()

    return place


def _model_first_units(doc: Any, model_first: _ModelFirst,
                       labels: Dict[int, str],
                       window: Optional[Sequence[int]] = None
                       ) -> List[Tuple[_Unit, List[Any]]]:
    """The logs, laboratory windows, printouts and narrative to read model
    first, each with the groups of pages its calls carry, in page order.

    The pages are grouped from the labels and the page descriptions
    (:func:`report_ingest.model_first_pages.group_pages`, the narrative's
    parts cut to its route's page limit and its text budget), within
    ``window`` when one is given. A unit's pages are every page its calls
    carry, but the narrative's: its pages are those labelled its own
    (cover, letter, contents, narrative and figures), and the front pages
    it is also shown, whatever their label, are its ``shown``."""
    from report_ingest.model_first_pages import (FIGURE_LABELS,
                                                 NARRATIVE_LABELS,
                                                 group_pages)
    from report_ingest.narrative_reader import (CHARS_PER_TOKEN,
                                                INPUT_TOKEN_BUDGET)

    inside = None if window is None else {int(p) for p in window}
    page_chars = {int(s.page): int(s.n_text_chars) for s in doc.page_map()
                  if inside is None or int(s.page) in inside}
    groups = group_pages(labels, model_first.descriptions,
                         n_pages=doc.n_pages, doc=doc,
                         route=model_first.routes["narrative"],
                         window=window, page_chars=page_chars,
                         max_chars=INPUT_TOKEN_BUDGET * CHARS_PER_TOKEN)
    units: List[Tuple[_Unit, List[Any]]] = []
    if groups["narrative"]:
        shown = _group_pages(groups["narrative"])
        own = [p for p in shown
               if labels.get(p) in NARRATIVE_LABELS + FIGURE_LABELS]
        units.append((_Unit("narrative", "narrative", own or shown,
                            shown=shown), groups["narrative"]))
    logs: Dict[str, List[Any]] = {}
    for group in groups["logs"]:
        logs.setdefault(group.key, []).append(group)
    for key, parts in logs.items():
        pages = _group_pages(parts)
        units.append((_Unit(labels.get(pages[0], "boring_log"), key, pages),
                      parts))
    for group in groups["lab"]:
        key = group.key + (f"-{group.part[0]}" if group.part else "")
        units.append((_Unit("lab_test", key, _group_pages([group])),
                      [group]))
    for group in groups["calc"]:
        units.append((_Unit("calculation", group.key,
                            _group_pages([group])), [group]))
    units.sort(key=lambda u: min(u[0].pages))
    return units


def _group_pages(parts: Sequence[Any]) -> List[int]:
    """Every page the groups carry, in order."""
    return sorted({int(p) for g in parts for p in g.pages})


def _model_first_plan(doc: Any, model_first: _ModelFirst,
                      labels: Dict[int, str],
                      window: Optional[Sequence[int]], every: Sequence[Any],
                      max_items: Optional[int]
                      ) -> Tuple[List[Tuple[_Unit, List[Any]]], List[Any],
                                 List[Any], List[Any]]:
    """What a model-first pass reads: the model-first units
    (:func:`_model_first_units`) in place of the work items of their
    stages, beside today's items of the other readers (the soundings), and
    the item ceiling applied once over all of them in page order. The
    items no reader reads (figures, the ``appended_report`` run) stay
    listed and do not count against it.

    Returns the units to read, every item, the items given a reader (what
    the reconciler counts as read) and today's items to read."""
    def first(item: Any) -> int:
        return min(item.pages) if item.pages else 0

    units = _model_first_units(doc, model_first, labels, window)
    soundings = [item for item in every if ITEM_READERS.get(item.kind)
                 not in (None, *_MODEL_FIRST_STAGES)]
    listed = [item for item in every if ITEM_READERS.get(item.kind) is None]
    combined = sorted([unit for unit, _parts in units] + soundings,
                      key=first)
    read = combined[:int(max_items)] if max_items else combined
    kept = {id(item) for item in read}
    return ([(unit, parts) for unit, parts in units if id(unit) in kept],
            sorted(combined + listed, key=first),
            sorted(read + listed, key=first),
            [item for item in soundings if id(item) in kept])


def _still_current(blob: Dict[str, Any], parts: Sequence[Any],
                   route: str) -> bool:
    """Is a saved model-first result one this run would ask for: the same
    pages, the same instruction and schema (``READ_PROMPT_VERSION``) and
    the same route? Any other is read again, and is new spend on resume: a
    result saved under mf-1, before a sheet scanned sideways was sent
    upright (RI-149), and one whose pages are no longer one unit's (a lab
    item cut by :func:`report_ingest.lab_reader.split_lab_items`, RI-150,
    may change a unit's pages)."""
    from report_ingest.model_first import READ_PROMPT_VERSION

    return (sorted(int(p) for p in blob.get("pages") or [])
            == _group_pages(parts)
            and blob.get("prompt_version") == READ_PROMPT_VERSION
            and blob.get("route") == route)


def _part_skipped(blob: Dict[str, Any]) -> bool:
    """Did part of a result's pages get no answer (a ``skipped`` row,
    :func:`report_ingest.model_first_read._skipped`)?"""
    return any(row.get("refused_by") == "budget" and row.get("pages")
               for row in blob.get("unresolved") or [])


def _read_model_first(doc: Any, model_first: _ModelFirst, out: str,
                      resume: bool, report_id: str, labels: Dict[int, str],
                      outline: Any, record: ReportRecord, spend: _Spend,
                      qa: List[QAEntry], unresolved: List[Dict[str, Any]],
                      workflow: str,
                      units: Sequence[Tuple[_Unit, List[Any]]],
                      window: Optional[Sequence[int]] = None) -> List[_Unit]:
    """The logs, laboratory sheets, printouts and narrative, model first:
    each of ``units`` (:func:`_model_first_plan`) read in its own calls and
    saved as ``items/mf-<key>.json``.

    A saved result is used again only while it is still current
    (:func:`_still_current`). A result no answer came back for (``error``
    set) puts its ``skipped`` entries, with all their pages, on the
    record's list and nothing else; one with part of its pages unanswered
    is used, its entries carried, but neither is saved, so a resumed run
    asks for the whole group again. What placement could not place is one
    QA note per answer, and so is each request a reader found over the
    request limit (:func:`_over_limit_qa`), the run's note counting the
    groups cut (:func:`_note_cuts`). One hole's logs read in more than one
    place are joined (:func:`_join_by_hole`). On an ``appendix_only`` file
    the narrative is not read, as today.

    Returns the units read to the end.
    """
    inside = None if window is None else {int(p) for p in window}
    body = [page for page, role in sorted(labels.items())
            if role in ("narrative", "figure", "plan", "profile", "toc")
            and (inside is None or page in inside)]

    done: List[_Unit] = []
    placed: Dict[str, List[Any]] = {}
    try:                       # an earlier run's file must never be read
        os.remove(os.path.join(out, "placement.json"))
    except OSError:
        pass
    first_log = len(record.investigations)
    for unit, parts in units:
        stage = _MODEL_FIRST_STAGES[ITEM_READERS[unit.kind]]
        if stage == "narrative" and workflow == "appendix_only":
            prose = [p for p in unit.pages if labels.get(p) == "narrative"]
            if prose:
                qa.append(QAEntry(
                    kind="skipped", where="items.narrative",
                    detail="this file is appendix or figure material, so "
                           "the narrative reader did not run",
                    pages=prose))
            continue
        key = f"mf-{unit.id}"
        path = os.path.join(out, "items", f"{key}.json")
        cached = _cached(path, resume)
        if cached is not None and not _still_current(
                cached, parts, model_first.routes[stage]):
            cached = None
        mark = spend.mark()
        where = f"items.{unit.kind}.{key}"
        try:
            blob = cached or _read_unit(doc, model_first, stage, parts,
                                        report_id, outline, body)
            spend.hand_in("items", blob, cached)
            # Each request found over the request limit (RI-121; none
            # without one), before anything else, so a cut is listed even
            # when the groups it was cut into came back with no answer.
            qa.extend(_over_limit_qa(blob, where))
            model_first.cut += sum(1 for row in blob.get("over_limit") or []
                                   if row.get("cut_into"))
            # The skipped entries first: a result with nothing read has no
            # record to validate, and its pages must still be listed.
            unresolved.extend(blob.get("unresolved") or [])
            if blob.get("error"):
                continue
            if cached is None and not _part_skipped(blob):
                _keep(path, blob, spend, mark, qa, where, unit.pages)
            _into_record(stage, blob, record)
            placed[key] = list(blob.get("placement") or [])
            qa.extend(_placement_qa(blob, where, unit.pages))
            spend.reading(cached is not None and blob is cached)
            done.append(unit)
        except Exception as exc:                 # one item must not stop the run
            qa.append(QAEntry(
                kind="skipped", where=f"items.{unit.kind}",
                detail=f"{key} could not be read: "
                       f"{type(exc).__name__}: {exc}",
                pages=unit.pages))
        finally:
            spend.settle(mark)
    _write_placement(out, placed)
    _join_by_hole(record, first_log, unresolved, qa, _ruler_slack(doc, qa))
    return done


def _write_placement(out: str, placed: Dict[str, List[Any]]) -> None:
    """``placement.json`` beside ``items/``: the placement of each unit
    this run put on the record, by its key (BACKLOG RI-120). The item files
    are not that -- a group with part of its pages unanswered is never
    saved, and a regroup leaves the old group's files behind -- so a count
    of what was placed reads this file. :func:`_read_model_first` removes
    any earlier file when it starts and writes this one as soon as its
    units are read, so a file on disk is always from the latest run that
    got that far and a run that stopped sooner leaves none."""
    try:
        with open(os.path.join(out, "placement.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(placed, fh)
    except OSError:                       # a count must not stop the run
        pass


def _ruler_slack(doc: Any, qa: List[QAEntry]) -> Any:
    """``slack(pages)`` for :func:`_join_by_hole`: the logs' ruler slack
    on those pages, from planlens' grid (no model call;
    :func:`report_ingest.log_reader._depth_slack`). Where planlens cannot
    read the grid, 0.0 and one QA note: the run goes on."""
    def slack(pages: Sequence[int]) -> float:
        from report_ingest.log_reader import _depth_slack
        from report_ingest.model_first_read import _span
        from report_ingest.visual_scales import log_grid_for

        try:
            grid, _info = log_grid_for(doc, list(pages))
            return _depth_slack(grid, pages)
        except Exception as exc:          # a planlens error must not end it
            qa.append(QAEntry(
                kind="note", where="investigations",
                detail=(f"the depth rulers on {_span(pages)} could not be "
                        f"read ({type(exc).__name__}: {exc}); two logs of "
                        f"one hole there were compared with no slack"),
                pages=list(pages)))
            return 0.0

    return slack


def _join_by_hole(record: ReportRecord, start: int,
                  unresolved: List[Dict[str, Any]], qa: List[QAEntry],
                  slack: Any = None) -> None:
    """One exploration's logs read in more than one place (model first).

    Among the logs from ``start`` on, those whose identifiers are one
    written two ways (:func:`report_ingest.reconciler.fold_id`) are taken in
    page order, and a later one is joined to the log before it only when it
    continues that log (:func:`_continues_log`, within ``slack(pages)``, the
    logs' ruler slack; none when not given), by the rule that joins a long
    log's parts (``model_first_read._join_logs``: the first's header, each
    later header value that differs listed as read, the rest appended). A
    join is one note, "logged in two places". A later log that does not
    continue the one before is kept apart, with one note naming the
    possible duplicate. Identifiers that differ only in leading zeros
    ("B-01" and "B-1") are never joined, as a report can number two holes
    so, and one note says so. A log with no identifier is joined to none.
    """
    from report_ingest.model_first_read import _join_logs, _span
    from report_ingest.reconciler import fold_id, hole_key

    logs: Dict[str, List[Any]] = {}
    for n, inv in enumerate(record.investigations[start:]):
        logs.setdefault(fold_id(inv.investigation_id) or f"#{n}",
                        []).append(inv)
    out: List[Any] = []
    for invs in logs.values():
        invs = sorted(invs, key=lambda inv: min(inv.pages) if inv.pages
                      else 0)
        kept: List[Tuple[Any, List[Any]]] = [(invs[0], [invs[0]])]
        for later in invs[1:]:
            first, parts = kept[-1]
            pages = sorted(set(first.pages) | set(later.pages))
            why = _continues_log(first, later,
                                 slack(pages) if slack else 0.0)
            if not why:
                joined, rows = _join_logs([first, later])
                # The last row names a join of calls; the note below says
                # what this join is.
                unresolved.extend(rows[:-1])
                kept[-1] = (joined, parts + [later])
                continue
            kept.append((later, [later]))
            qa.append(QAEntry(
                kind="note",
                where=f"investigations.{later.investigation_id}",
                detail=(f"log {later.investigation_id} on "
                        f"{_span(later.pages)} has the identifier of the log "
                        f"on {_span(first.pages)}, but does not continue it "
                        f"({why}); the two are kept apart: one may be the "
                        f"other printed again, or two holes given one name"),
                pages=pages))
        for inv, parts in kept:
            if len(parts) > 1:
                places = {2: "two", 3: "three"}.get(len(parts),
                                                     str(len(parts)))
                qa.append(QAEntry(
                    kind="note",
                    where=f"investigations.{inv.investigation_id}",
                    detail=(f"log {inv.investigation_id} is logged in "
                            f"{places} places ("
                            + "; ".join(_span(p.pages) for p in parts)
                            + "), each continuing the one before, and they "
                              "are joined: the header is the first's, and "
                              "each later one's layers, samples, blow "
                              "counts and water readings are appended"),
                    pages=list(inv.pages)))
            out.append(inv)
    record.investigations[start:] = out

    padded: Dict[str, List[str]] = {}
    for folded, invs in logs.items():
        if not folded.startswith("#"):
            padded.setdefault(hole_key(folded), []).append(folded)
    for folds in padded.values():
        if len(folds) < 2:
            continue
        names = [logs[f][0].investigation_id for f in folds]
        qa.append(QAEntry(
            kind="note", where=f"investigations.{names[0]}",
            detail=(f"the logs {' and '.join(names)} have identifiers that "
                    f"differ only in leading zeros; they are kept as "
                    f"separate logs, as a report can number two holes so"),
            pages=sorted({p for f in folds for inv in logs[f]
                          for p in inv.pages})))


def _continues_log(first: Any, later: Any, slack: float = 0.0) -> str:
    """Why ``later`` does not continue ``first``, or ``""`` when it does:
    its depths, in the same unit, start below the surface and at or below
    the deepest depth ``first`` gives, less ``slack``. A log that starts at
    the surface continues none."""
    if (first.depth_unit or "") != (later.depth_unit or ""):
        return (f"its depths are in {later.depth_unit or 'no stated unit'}, "
                f"the first's in {first.depth_unit or 'no stated unit'}")
    above, below = _log_depths(first), _log_depths(later)
    if not above or not below:
        return "one of them gives no depths"
    if min(below) <= 0:
        return f"it starts at the surface, {min(below):g}"
    if min(below) >= max(above) - slack:
        return ""
    return (f"its depths start at {min(below):g}, above the first's "
            f"deepest, {max(above):g}")


def _log_depths(inv: Any) -> List[float]:
    """Every depth a log gives: its layers', samples' and drives'."""
    quantities = ([q for ly in inv.layers for q in (ly.top, ly.bottom)]
                  + [q for sm in inv.samples for q in (sm.top, sm.bottom)]
                  + [q for d in inv.spt for q in (d.depth_top,
                                                    d.depth_bottom)])
    return [float(q.value) for q in quantities if q is not None]


def _read_unit(doc: Any, model_first: _ModelFirst, stage: str,
               groups: Sequence[Any], report_id: str, outline: Any,
               body: Sequence[int]) -> Dict[str, Any]:
    """One model-first result: a log (its parts), a laboratory window, a
    printout or the narrative (its parts)."""
    from report_ingest import model_first_read as readers

    engine = model_first.engines[stage]
    route = model_first.routes[stage]
    common = dict(report_id=report_id, route=route,
                  pdf=model_first.source(doc) if route == "pdf" else None,
                  place=model_first.place)
    if model_first.max_request_tokens_in:
        # Only when set (RI-121): unset, each reader is called as before.
        common["max_request_tokens_in"] = model_first.max_request_tokens_in
    if stage == "logs":
        return readers.read_logs(doc, groups, engine, **common)[0]
    if stage == "lab":
        return readers.read_lab(doc, groups, engine,
                                descriptions=model_first.descriptions,
                                **common)[0]
    if stage == "calc":
        return readers.read_calcs(doc, groups, engine, **common)[0]
    return readers.read_narrative(doc, groups, engine, outline=outline,
                                  body_pages=list(body) or None, **common)


def _into_record(stage: str, blob: Dict[str, Any],
                 record: ReportRecord) -> None:
    """A model-first result's record objects, into ``record``."""
    from report_ingest.model import (
        Calculation, GeneralFacts, Investigation, LabTest, NarrativeFacts,
        NaturalHazardFacts,
    )

    if stage == "narrative":
        record.general = GeneralFacts.model_validate(blob["general"])
        record.natural_hazards = NaturalHazardFacts.model_validate(
            blob["natural_hazards"])
        record.narrative = NarrativeFacts.model_validate(blob["facts"])
    elif stage == "logs":
        if blob.get("investigation"):
            record.investigations.append(
                Investigation.model_validate(blob["investigation"]))
    elif stage == "lab":
        for test in blob.get("tests") or []:
            record.lab_tests.append(LabTest.model_validate(test))
    elif blob.get("calculation"):
        record.calculations.append(
            Calculation.model_validate(blob["calculation"]))


def _placement_qa(blob: Dict[str, Any], where: str,
                  pages: Sequence[int]) -> List[QAEntry]:
    """One note per answer whose values were not all placed where cited
    ("N of M not found printed on the cited page"), on the pages of the
    values it names."""
    out: List[QAEntry] = []
    for found in blob.get("placement") or []:
        note = str((found or {}).get("note") or "").strip()
        if not note:
            continue
        named = sorted({int(item["page"]) for item in found.get("items") or []
                        if item.get("page") is not None
                        and (item.get("status") != "placed"
                             or item.get("page_level"))})
        out.append(QAEntry(kind="note", where=f"{where}.placement",
                           detail=note,
                           pages=named or [int(p) for p in pages][:200]))
    return out


def _bound_range_labels(engine: Any, folder: str, resume: bool,
                        window: Sequence[int],
                        rules: Dict[int, Tuple[str, float]], outline: Any,
                        labelling: "_Labelling",
                        descriptions: Dict[int, Dict[str, Any]],
                        spend: _Spend, qa: List[QAEntry]) -> Dict[int, str]:
    """A bound document's own range labels, model first (plan section 8.2
    step 5): its pages' descriptions, already made for the whole file,
    labelled by sections in one call of their own, on the parent file's
    page numbers. Saved in the child's folder when every page had a
    description; a call that fails is a QA note and no labels."""
    from report_ingest.range_labels import (
        DESCRIBE_PROMPT_VERSION, _engine_name, conventions_version,
        label_by_ranges, labeller_fingerprint,
    )
    from report_ingest.triage import outline_text

    arm = labelling.range_arm or "plain"
    describer = labelling.describe_engine or labelling.vision_engine or engine
    labeller = labelling.range_engine or engine
    made_by = {"describer": _engine_name(describer),
               "labeller": _engine_name(labeller),
               "describe_prompt": DESCRIBE_PROMPT_VERSION,
               "conventions": conventions_version(),
               "labeller_prompt": labeller_fingerprint(arm == "hinted")}
    window = [int(p) for p in window]
    ends = [window[0], window[-1]]
    path = os.path.join(folder, f"ranges-{arm}.json")
    blob = _cached(path, resume)
    if blob is not None and any(blob.get(key) != value
                                for key, value in made_by.items()):
        blob = None
    if blob is not None:
        spend.reuse("ranges", blob.get("cost"))
    if blob is None:
        local = {n: descriptions[p] for n, p in enumerate(window)
                 if p in descriptions}
        mark = spend.mark()
        try:
            found = label_by_ranges(
                local, len(window), labeller,
                outline=outline_text(outline) if outline is not None else "",
                hints=({n: rules.get(p, ("", 0.0))[0]
                        for n, p in enumerate(window)}
                       if arm == "hinted" else None))
        except Exception as exc:              # the labels, not the document
            spend.settle(mark)
            qa.append(QAEntry(
                kind="note", where="labels.ranges",
                detail=f"the range labeller did not run over this bound "
                       f"document ({type(exc).__name__}: {exc}), so its page "
                       f"labels are the rules' re-run over its pages",
                pages=ends))
            return {}
        spend.add(found.cost)
        blob = found.to_dict()
        blob.update(made_by)
        if len(local) == len(window):
            _keep(path, blob, spend, mark, qa, "labels.ranges", ends)
        spend.settle(mark)
    left = [window[int(p)] for p in (blob.get("unresolved") or [])
            if 0 <= int(p) < len(window)]
    if left:
        qa.append(QAEntry(
            kind="note", where="labels.ranges",
            detail=f"{len(left)} page(s) no section settled; the rules "
                   f"re-run over this document's pages label them",
            pages=left[:200]))
    return {window[int(k)]: str(v)
            for k, v in (blob.get("labels") or {}).items()
            if 0 <= int(k) < len(window)}
