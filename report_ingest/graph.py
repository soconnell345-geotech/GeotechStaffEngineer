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

WHAT IS NOT READ YET. Calculation printouts are work package 5; every calc
item is recorded as a QA entry saying it was not read, so a reviewer of the
record knows the pages exist and were skipped on purpose.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.label_vote import (
    DEFAULT_POLICY, POLICIES, PageChoice, Voter, combine, load_trust_table,
    neighbourhood, split_pages,
)
from report_ingest.model import (
    DocumentFacts, LabelVote, PageLabel, Project, QAEntry, ReportRecord,
)

__all__ = [
    "ingest_report", "Budgets", "output_paths", "ITEM_READERS",
    "OUTPUT_NAMES", "REVIEW_MODES",
]

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

#: Which reader takes which kind of work item. A kind that is not here is
#: recorded and not read -- ``figures``, ``photos``, ``front_matter``,
#: ``appended_report`` and (until work package 5) ``calculation``.
ITEM_READERS: Dict[str, str] = {
    "boring_log": "log", "test_pit_log": "log", "cpt_log": "log",
    "dcp_log": "log", "lab_test": "lab", "narrative": "narrative",
}

#: The files :func:`report_ingest.writers.write_outputs` leaves behind.
OUTPUT_NAMES: Tuple[str, ...] = (
    "report.record.json", "report.summary.md", "report.page.md",
    "report.diggs.xml", "reports.db",
)


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
    #: The label review's tool-call budget. None takes the one that fits the
    #: mode: ``max(60, 0.25 x pages)`` over a whole report, and
    #: ``max(20, 0.5 x split pages)`` over the pages the voters split on,
    #: which is a far smaller set and needs a budget that says so.
    review_tool_calls: Optional[int] = None
    review_model_calls: int = 60
    #: A ceiling on the vision pass's MODEL CALLS; None is no ceiling and
    #: the mode decides (sheet mode is about one call per six pages).
    vision_model_calls: Optional[int] = None
    #: Skip a pass entirely. Triage off means no workflow and the standard
    #: path; review off is ``review_mode="none"`` however it was asked for,
    #: and is kept because callers already spell it this way.
    triage: bool = True
    review: bool = True
    #: A ceiling on how many work items are read, for a smoke run over a big
    #: report. None reads them all.
    max_items: Optional[int] = None

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
    """What the run has cost so far, across every pass."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.dollars = 0.0
        self.seconds = 0.0

    def add(self, cost: Optional[Dict[str, Any]]) -> None:
        if not cost:
            return
        self.calls += int(cost.get("calls") or 0)
        self.input_tokens += int(cost.get("input_tokens") or 0)
        self.output_tokens += int(cost.get("output_tokens") or 0)
        self.dollars += float(cost.get("dollars") or 0.0)
        self.seconds += float(cost.get("seconds") or 0.0)


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
    note: str = ""                    # printed once, and kept for the run file
    #: What each pass cost, whether this run paid it or read it off disk.
    #: A resumed run's ``paid`` is False and the numbers are still the
    #: report's, which is what a cost-per-report table wants.
    vision_cost: Dict[str, Any] = field(default_factory=dict)
    review_cost: Dict[str, Any] = field(default_factory=dict)
    vision_paid: bool = False
    review_paid: bool = False

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
                  templates: Any = None) -> ReportRecord:
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
    trust_table
        A path or a dict for the ``trust`` policy, as the ``vote`` stage
        writes to ``vote/trust_table.json``. Without one, ``trust`` falls
        back to ``structural`` and prints a note.
    templates
        Log-template fingerprints for the third voter: a path, an iterable
        of :class:`~report_ingest.log_templates.Fingerprint`, or ``None`` to
        use whatever ``log_templates.use_templates`` put in force -- which
        is nothing unless something set it, and then the voter is a no-op.
    """
    from report_ingest.writers import write_outputs

    budgets = budgets or Budgets()
    if label_policy not in POLICIES:
        raise ValueError(f"unknown label_policy {label_policy!r}; the "
                         f"policies are {list(POLICIES)}")
    if review_mode not in REVIEW_MODES:
        raise ValueError(f"unknown review_mode {review_mode!r}; the modes "
                         f"are {list(REVIEW_MODES)}")
    labelling = _Labelling(
        policy=label_policy,
        review_mode="none" if not budgets.review else review_mode,
        vision_engine=vision_engine, vision_mode=vision_mode,
        vision_detail=vision_detail, trust_table=trust_table,
        templates=templates)
    out = str(out_dir)
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    started = time.time()
    spend = _Spend()
    qa: List[QAEntry] = []
    unresolved: List[Dict[str, Any]] = []

    doc = _open(source, di_result, report_id)
    try:
        record = _run(doc, engine, budgets, out, resume, questions or [],
                      report_id, spend, qa, unresolved, di_result, labelling)
    finally:
        if not _is_document(source):
            doc.close()

    record.qa.extend(qa)
    record.document.model_calls = spend.calls
    record.document.input_tokens = spend.input_tokens
    record.document.output_tokens = spend.output_tokens
    record.document.dollars = round(spend.dollars, 5)
    record.document.seconds = round(time.time() - started, 1)

    if write:
        if _is_document(source):
            key_source: Any = getattr(source, "path", None) or report_id \
                or None
        elif isinstance(source, (bytes, bytearray)):
            key_source = None
        else:
            key_source = source
        write_outputs(record, out, source=key_source, db_path=db_path)
    return record


def _run(doc: Any, engine: Any, budgets: Budgets, out: str, resume: bool,
         questions: Sequence[str], report_id: str, spend: _Spend,
         qa: List[QAEntry], unresolved: List[Dict[str, Any]],
         di_result: Any, labelling: "_Labelling") -> ReportRecord:
    """The loop itself, with the document open."""
    from planlens.document.model import SOURCE_AZURE_DI
    from planlens.document.roles import (
        assign, build_items, document_outline, facts_from_document,
    )
    from report_ingest.reconciler import reconcile
    from report_ingest.triage import document_facts, triage as run_triage

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
    if budgets.triage:
        cached = _cached(os.path.join(out, "triage.json"), resume)
        if cached is not None:
            record.document.workflow = cached.get("workflow", "standard")
            profile = _Profile(cached)
        else:
            profile = run_triage(doc, roles, outline, engine=engine,
                                 facts=counted)
            spend.add(profile.cost)
            _save(os.path.join(out, "triage.json"), profile.to_dict())
            record.document.workflow = profile.workflow
    workflow = record.document.workflow

    if profile is not None:
        for note in getattr(profile, "anomalies", ()) or ():
            qa.append(QAEntry(kind="note", where="triage", detail=str(note)))
        bound = getattr(profile, "bound_together", ()) or ()
        if bound:
            qa.append(QAEntry(
                kind="note", where="triage.bound_together",
                detail=f"{len(bound)} separately-bound document(s) are in "
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
    choices = _vote_labels(doc, engine, budgets, out, resume, roles,
                           labelling, spend, qa)
    labels = {c.page: c.label for c in choices}
    split = split_pages(choices)
    confidence = {c.page: c.confidence for c in choices}
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
    items = build_items(facts, roles)

    if budgets.max_items:
        items = items[:int(budgets.max_items)]

    # -- 1-2. the readers ---------------------------------------------------
    _read_items(doc, engine, budgets, out, resume, questions, report_id,
                items, labels, outline, record, spend, qa, unresolved,
                workflow, no_text, di_pages)

    # -- 3. the reconciler --------------------------------------------------
    reconcile(record, labels=labels, items=items, no_text_pages=no_text,
              di_pages=di_pages, reader_unresolved=unresolved)
    record.project = _project_from(record)
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
                 spend: _Spend, qa: List[QAEntry]) -> List[PageChoice]:
    """One :class:`PageChoice` per page, from the voters this run has.

    The rules always vote. The vision pass votes unless the policy is
    ``rules``, and a pass that will not run is a QA note and one voter
    fewer rather than a failed report. The template votes only where a
    fingerprint file is in force, which is not the default state.
    """
    table = labelling.resolve()
    rules = {int(r.page): (r.role, float(r.confidence)) for r in roles}
    vision = _vision_labels(doc, engine, budgets, out, resume, labelling,
                            spend, qa)
    templates = _template_labels(doc, labelling, qa)
    pages = sorted(set(rules) | set(vision))
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
            policy=labelling.policy, trust_table=table, page=page))
    return out_rows


def _vision_labels(doc: Any, engine: Any, budgets: Budgets, out: str,
                   resume: bool, labelling: "_Labelling", spend: _Spend,
                   qa: List[QAEntry]) -> Dict[int, Tuple[str, float]]:
    """``page -> (label, confidence)`` from one pass over the pictures."""
    if labelling.policy == "rules":
        return {}
    path = os.path.join(out, "vision.json")
    blob = _cached(path, resume)
    if blob is None:
        from report_ingest.vision_labels import classify_pages_by_vision
        try:
            seen = classify_pages_by_vision(
                doc, labelling.vision_engine or engine,
                mode=labelling.vision_mode, detail=labelling.vision_detail,
                budget=budgets.vision_model_calls)
        except Exception as exc:              # one voter, not the report
            qa.append(QAEntry(
                kind="note", where="labels.vision",
                detail=f"the vision voter did not run "
                       f"({type(exc).__name__}: {exc}), so the page labels "
                       f"are the rules' alone and nothing was flagged as a "
                       f"disagreement"))
            return {}
        blob = seen.to_dict()
        _save(path, blob)
        spend.add(blob.get("cost"))
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
    if blob is None and mode != "none" \
            and not (mode == "disagreements" and not split):
        budget = budgets.review_tool_calls
        if budget is None and mode == "disagreements":
            budget = budget_for_split(len(split))
        review = review_labels(doc, roles, outline, profile,
                               budget=budget, engine=engine,
                               max_model_calls=budgets.review_model_calls,
                               pages=asked, choices=choices)
        spend.add(review.cost)
        labelling.review_paid = True
        blob = review.to_dict()
        _save(os.path.join(out, "review.json"), blob)
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
            detail=(f"the page labellers did not agree and the review did "
                    f"not settle it; the record calls page {choice.page} "
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
    """
    rows: List[Dict[str, Any]] = []
    if blob.get("changes") is not None:
        for row in (blob.get("changes") or []):
            try:
                int(row["page"])
            except (KeyError, TypeError, ValueError):
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
                        if labelling.policy != "rules" else ""),
        "note": labelling.note,
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
                 "review_paid": labelling.review_paid},
        "labels": {str(c.page): (str(changed[c.page].get("to"))
                                 if c.page in changed else c.label)
                   for c in choices},
        "pages": [c.to_dict() for c in choices],
    }


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
                no_text: Sequence[int], di_pages: Sequence[int]) -> None:
    from report_ingest.model import (
        GeneralFacts, Investigation, LabTest, NarrativeFacts,
        NaturalHazardFacts,
    )

    body = [page for page, role in sorted(labels.items())
            if role in ("narrative", "figure", "plan", "profile", "toc")]
    blind = set(int(p) for p in no_text) - set(int(p) for p in di_pages)

    for item in items:
        reader = ITEM_READERS.get(item.kind)
        pages = [int(p) for p in item.pages]
        if reader is None:
            if item.kind in ("calculation", "appended_report"):
                qa.append(QAEntry(
                    kind="skipped", where=f"items.{item.kind}",
                    detail=(
                        "calculation printouts are not read yet (work package "
                        "5); the pages are listed here so they are not "
                        "forgotten" if item.kind == "calculation" else
                        "a report bound inside this one was not read as its "
                        "own document; its pages are listed here"),
                    pages=pages))
            continue

        if reader == "narrative":
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

        path = os.path.join(out, "items", f"{item.id}.json")
        cached = _cached(path, resume)
        try:
            if reader == "narrative":
                blob = cached or _read_narrative(
                    doc, pages, engine, budgets, outline, report_id, body,
                    questions)
                if cached is None:
                    _save(path, blob)
                spend.add(blob.get("cost"))
                record.general = GeneralFacts.model_validate(blob["general"])
                record.natural_hazards = NaturalHazardFacts.model_validate(
                    blob["natural_hazards"])
                record.narrative = NarrativeFacts.model_validate(
                    blob["facts"])
                unresolved.extend(blob.get("unresolved") or [])
            elif reader == "log":
                blob = cached or _read_log(doc, pages, engine, budgets,
                                           item, report_id)
                if cached is None:
                    _save(path, blob)
                spend.add(blob.get("cost"))
                record.investigations.append(
                    Investigation.model_validate(blob["investigation"]))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
            else:
                blob = cached or _read_lab(doc, pages, engine, budgets,
                                           item, report_id)
                if cached is None:
                    _save(path, blob)
                spend.add(blob.get("cost"))
                for test in blob.get("tests") or []:
                    record.lab_tests.append(LabTest.model_validate(test))
                unresolved.extend(blob.get("unresolved") or [])
                _vote_qa(qa, blob, pages)
        except Exception as exc:                 # one item must not stop the run
            qa.append(QAEntry(
                kind="skipped", where=f"items.{item.kind}",
                detail=f"{item.id} could not be read: "
                       f"{type(exc).__name__}: {exc}",
                pages=pages))


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
        qa.append(QAEntry(
            kind="disagreement", where=str(row.get("where") or ""),
            detail=(f"{row.get('what', '')}: {row.get('floor_method', 'floor')} "
                    f"read {row.get('floor', '')!s}, "
                    f"{row.get('model_method', 'model')} read "
                    f"{row.get('model', '')!s}; the record carries the "
                    f"{row.get('kept', 'floor')}'s. {why}").strip(),
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


def _read_narrative(doc, pages, engine, budgets, outline, report_id, body,
                    questions) -> Dict[str, Any]:
    from report_ingest.narrative_reader import read_narrative
    result = read_narrative(doc, pages, engine, budget=budgets.narrative,
                            outline=outline, report_id=report_id,
                            body_pages=body or pages, questions=questions)
    return result.to_dict()


def _read_log(doc, pages, engine, budgets, item, report_id) -> Dict[str, Any]:
    from report_ingest.log_reader import read_log
    result = read_log(doc, pages, engine, budget=budgets.log,
                      item_title=item.title or "", report_id=report_id)
    return result.to_dict()


def _read_lab(doc, pages, engine, budgets, item, report_id) -> Dict[str, Any]:
    from report_ingest.lab_reader import read_lab_sheet
    result = read_lab_sheet(doc, pages, engine, budget=budgets.lab,
                            item_title=item.title or "", report_id=report_id)
    return result.to_dict()
