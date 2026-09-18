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

from report_ingest.model import (
    DocumentFacts, Project, QAEntry, ReportRecord,
)

__all__ = [
    "ingest_report", "Budgets", "output_paths", "ITEM_READERS",
    "OUTPUT_NAMES",
]

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
    #: The label review's tool-call budget; None takes planlens' own, which
    #: scales with the page count.
    review_tool_calls: Optional[int] = None
    review_model_calls: int = 60
    #: Skip a pass entirely. Triage off means no workflow and the standard
    #: path; review off means the rule labels stand.
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


def _open(source: Any, di_result: Any, report_id: str):
    from planlens.document import open_document
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


def ingest_report(source: Any, engine: Any, *,
                  di_result: Any = None,
                  questions: Optional[Sequence[str]] = None,
                  out_dir: Any = ".",
                  budgets: Optional[Budgets] = None,
                  report_id: str = "",
                  resume: bool = True,
                  write: bool = True,
                  db_path: Any = None) -> ReportRecord:
    """Read one report end to end and write its record and its exports.

    ``source`` is a PDF path or its bytes. ``di_result`` is an Azure Document
    Intelligence layout for the scanned pages, when one exists --
    ``planlens.document.azure_di`` builds it and planlens puts its text into
    the same frame as everything else. ``questions`` are the caller's own
    questions, answered off the narrative beside the owner's two schemas.

    Everything is written under ``out_dir``: the record and its exports at the
    top, and one file per work item under ``items/`` so a second run resumes
    rather than paying again. Pass ``resume=False`` to re-read everything.
    """
    from report_ingest.writers import write_outputs

    budgets = budgets or Budgets()
    out = str(out_dir)
    os.makedirs(os.path.join(out, "items"), exist_ok=True)
    started = time.time()
    spend = _Spend()
    qa: List[QAEntry] = []
    unresolved: List[Dict[str, Any]] = []

    doc = _open(source, di_result, report_id)
    try:
        record = _run(doc, engine, budgets, out, resume, questions or [],
                      report_id, spend, qa, unresolved, di_result)
    finally:
        doc.close()

    record.qa.extend(qa)
    record.document.model_calls = spend.calls
    record.document.input_tokens = spend.input_tokens
    record.document.output_tokens = spend.output_tokens
    record.document.dollars = round(spend.dollars, 5)
    record.document.seconds = round(time.time() - started, 1)

    if write:
        write_outputs(record, out,
                      source=None if isinstance(source, (bytes, bytearray))
                      else source,
                      db_path=db_path)
    return record


def _run(doc: Any, engine: Any, budgets: Budgets, out: str, resume: bool,
         questions: Sequence[str], report_id: str, spend: _Spend,
         qa: List[QAEntry], unresolved: List[Dict[str, Any]],
         di_result: Any) -> ReportRecord:
    """The loop itself, with the document open."""
    from planlens.document.model import SOURCE_AZURE_DI
    from planlens.document.roles import (
        PageRole, assign, build_items, document_outline, facts_from_document,
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

    # -- 0c. label review ---------------------------------------------------
    labels = {r.page: r.role for r in roles}
    if budgets.review:
        from report_ingest.label_review import review_labels
        cached = _cached(os.path.join(out, "review.json"), resume)
        if cached is not None:
            labels = {int(k): v for k, v in
                      (cached.get("final_labels") or {}).items()} or labels
        else:
            review = review_labels(doc, roles, outline, profile,
                                   budget=budgets.review_tool_calls,
                                   engine=engine,
                                   max_model_calls=budgets.review_model_calls)
            spend.add(review.cost)
            _save(os.path.join(out, "review.json"), review.to_dict())
            labels = review.final_labels
            for row in review.unresolved:
                unresolved.append({"what": f"page {row.get('page')}",
                                   "why": row.get("why", ""),
                                   "page": row.get("page")})
        roles = [PageRole(page=r.page, role=labels.get(r.page, r.role),
                          confidence=r.confidence, evidence=r.evidence)
                 for r in roles]
        items = build_items(facts, roles)
    else:
        items = build_items(facts, list(roles))

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
            else:
                blob = cached or _read_lab(doc, pages, engine, budgets,
                                           item, report_id)
                if cached is None:
                    _save(path, blob)
                spend.add(blob.get("cost"))
                for test in blob.get("tests") or []:
                    record.lab_tests.append(LabTest.model_validate(test))
                unresolved.extend(blob.get("unresolved") or [])
        except Exception as exc:                 # one item must not stop the run
            qa.append(QAEntry(
                kind="skipped", where=f"items.{item.kind}",
                detail=f"{item.id} could not be read: "
                       f"{type(exc).__name__}: {exc}",
                pages=pages))


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
