"""The geotechnical report review checklist, run on a report's record.

One place holds a geotechnical report's questions: the owner's two standing
query schemas (:class:`~report_ingest.model.GeneralFacts`,
:class:`~report_ingest.model.NaturalHazardFacts`), their conventions
(:mod:`report_ingest.narrative_glossary`) and this checklist. It moved here
from ``funhouse_agent.report_checklist`` with BACKLOG RI-34 (owner,
2026-10-20: "move all questions related to geotech reports to
report_ingest"); the Document Review page no longer offers it.

The checklist is DATA, in ``report_review_checklist.json`` beside this
module, marked DRAFT until the owner has marked it up (BACKLOG OWN-31). Each
item is either:

* ``code`` - a check made here from the record report ingest wrote
  (:class:`~report_ingest.model.ReportRecord`): what each page was labelled,
  the explorations and laboratory tests read, the narrative's answers, and
  the reconciler's QA entries (its conflicts, links and counts); or
* ``judge`` - an item the agent works through against the report and
  reports on.

:func:`run_checklist` runs every ``code`` item it has a check for and returns
the ``judge`` items for the agent. A ``code`` item whose check cannot run on
the record it is given (no record, no page labels, no summary table) is
returned with the judge items, with what the code found, so nothing on the
list is dropped in silence.

Adding a check: write ``fn(ctx, item) -> {"status", "detail", ...}`` and
:func:`register_check` it under the name the JSON item gives as ``check``.
``status`` is ``pass``, ``fail``, ``unsure`` (the code found something it
cannot settle: a look is needed) or ``not_run`` (nothing to check here).

Importing this module costs pydantic (the record's models) and nothing else;
the tool imports LangChain when it is built.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "report_review_checklist.json")

#: The file report ingest writes the record to (``writers.RECORD_NAME``).
RECORD_NAME = "report.record.json"

STATUSES = ("pass", "fail", "unsure", "not_run")

#: Page labels that are exploration logs, and laboratory sheets.
LOG_LABELS = ("boring_log", "test_pit_log", "cpt_log", "dcp_log")
LAB_LABELS = ("lab_test",)

#: The narrative's answers about the explorations (the owner's field names).
EXPLORATION_FIELDS = ("boringCount", "testPitCount", "cptCount",
                      "boringDictionary", "testPitDictionary")

#: Narrative answers the reader counts from the record itself rather than
#: reads off a page (the explorations from the logs read, the tables and
#: figures from the captions: ``narrative_reader.DETERMINISTIC_FIELDS`` and
#: its caption counts). They carry no quote by design, so they are not
#: "values with no page".
COUNTED_BY_CODE = EXPLORATION_FIELDS + ("tableCount", "figureCount")

#: What the agent is asked to do with the result.
HOW_TO_REPORT = (
    "Report the code checks as they stand (a failed one is a finding: give "
    "its detail and pages). Work through every item for you to judge "
    "against the report and its record, and report each as met, not met or "
    "cannot tell, with the pages you looked at; where code_said is given, "
    "start from it.")


def load_checklist(path: Optional[str] = None) -> Dict[str, Any]:
    """The checklist data (the shipped DRAFT unless ``path`` is given)."""
    with open(path or DATA_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    for item in data.get("items") or []:
        if item.get("mode") not in ("code", "judge"):
            raise ValueError(f"checklist item {item.get('id')!r}: mode must "
                             "be 'code' or 'judge'")
    return data


def record_path(where: str) -> str:
    """The record file for ``where``: the file itself, or the folder report
    ingest wrote it into."""
    where = os.fspath(where)
    if os.path.isdir(where):
        return os.path.join(where, RECORD_NAME)
    return where


def load_record(record: Any):
    """A :class:`~report_ingest.model.ReportRecord` from a record, a dict, or
    the path of a record file or of the folder holding one. ``None`` stays
    ``None``; a path that does not load raises ``ValueError`` saying why."""
    from report_ingest.model import ReportRecord

    if record is None or isinstance(record, ReportRecord):
        return record
    if isinstance(record, dict):
        return ReportRecord.model_validate(record)
    if not isinstance(record, (str, os.PathLike)):
        # e.g. the coverage ledger the checks read before RI-34
        raise TypeError(f"the checklist reads a report's record (a "
                        f"ReportRecord, its dict, or the path of "
                        f"{RECORD_NAME} or its folder), not a "
                        f"{type(record).__name__}")
    path = record_path(str(record))
    if not os.path.isfile(path):
        raise ValueError(f"no record at {path!r}: give the path of the "
                         f"{RECORD_NAME} report ingest wrote, or of the "
                         f"folder it wrote it into")
    try:
        with open(path, encoding="utf-8") as fh:
            return ReportRecord.model_validate(json.load(fh))
    except Exception as exc:  # noqa: BLE001 - said to the caller
        raise ValueError(f"the record at {path!r} did not load: "
                         f"{type(exc).__name__}: {str(exc)[:200]}") from exc


# ---------------------------------------------------------------------------
# Pages, written the way a reader cites them
# ---------------------------------------------------------------------------

def compact(pages: Iterable[int], offset: int = 0) -> str:
    """``[3, 4, 5, 9]`` as ``"3-5, 9"``; ``offset=1`` gives PDF pages (a
    record's pages are 0-based, a reader counts from 1)."""
    nums = sorted({int(p) + offset for p in pages})
    runs: List[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        runs.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(runs)


def _pdf(pages: Iterable[int]) -> str:
    return compact(pages, 1)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class CheckContext:
    """What a check may use: the record (``None`` when none was given)."""
    record: Any = None

    def qa(self, kinds: Optional[Sequence[str]] = None,
           where: str = "") -> List[Any]:
        """The record's QA entries, narrowed by kind and by a word of their
        ``where`` or ``detail``."""
        out = list(getattr(self.record, "qa", None) or [])
        if kinds:
            out = [e for e in out if e.kind in set(kinds)]
        if where:
            w = where.lower()
            out = [e for e in out if w in str(e.where).lower()
                   or w in str(e.detail).lower()]
        return out

    def labelled(self, labels: Sequence[str]) -> List[int]:
        """Pages the record labels as one of ``labels``.

        Not the pages of a report bound inside this one that was read as its
        own record: they are read there and checked there, and this record
        holds nothing off them (BACKLOG RI-14).
        """
        from report_ingest.bound import parse_range

        n_pages = int(getattr(self.record.document, "n_pages", 0) or 0)
        theirs = {page for row in (self.record.bound_documents or [])
                  if row.read
                  for page in parse_range(row.pages, n_pages or 10 ** 6)}
        return sorted(pl.page for pl in self.record.page_labels
                      if pl.label in set(labels) and pl.page not in theirs)

    def sheets(self) -> List[Any]:
        return [t for t in self.record.lab_tests if t.kind != "summary_table"]

    def tables(self) -> List[Any]:
        return [t for t in self.record.lab_tests if t.kind == "summary_table"]


def _entry_line(e: Any) -> str:
    line = f"{e.kind} ({e.where}): {e.detail}" if e.where else \
        f"{e.kind}: {e.detail}"
    if e.values:
        line += f" (values {', '.join(str(v) for v in e.values[:6])})"
    if e.pages:
        line += f" [PDF pages {_pdf(e.pages)}]"
    return line


def _pages_of(entry: Any) -> List[int]:
    """Every page a record entry came from: its own list and its
    provenance."""
    pages = {int(p) for p in (getattr(entry, "pages", None) or [])}
    prov = getattr(entry, "prov", None)
    for p in (prov if isinstance(prov, list) else [prov] if prov else []):
        if getattr(p, "page", None) is not None:
            pages.add(int(p.page))
    return sorted(pages)


def _no_record(_item: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "not_run",
            "detail": f"no record of the report was given (the {RECORD_NAME} "
                      "report ingest writes when it reads a report), so code "
                      "could not check this"}


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_logs_and_lab_read(ctx: CheckContext, item: Dict[str, Any]
                            ) -> Dict[str, Any]:
    """Every page labelled a log or a laboratory sheet has a record entry
    that came from it, stated as counts."""
    rec = ctx.record
    if rec is None:
        return _no_record(item)
    if not rec.page_labels:
        roles = {k: v for k, v in (rec.document.page_roles or {}).items()
                 if k in LOG_LABELS + LAB_LABELS}
        return {"status": "not_run",
                "detail": "the record has no per-page labels (written before "
                          "the page labels were a vote), so the pages cannot "
                          "be listed; page roles counted: "
                          + (", ".join(f"{k} {v}" for k, v in roles.items())
                             or "none of logs or laboratory sheets")
                          + f"; {len(rec.investigations)} explorations and "
                          f"{len(rec.lab_tests)} laboratory tests read"}
    groups = (("exploration logs", ctx.labelled(LOG_LABELS),
               rec.investigations),
              ("laboratory sheets", ctx.labelled(LAB_LABELS), rec.lab_tests))
    if not any(pages for _l, pages, _e in groups):
        return {"status": "not_run",
                "detail": "no page of the record is labelled an exploration "
                          "log or a laboratory sheet"}
    counts: List[str] = []
    missing: List[str] = []
    for label, pages, entries in groups:
        if not pages:
            continue
        read = {p for e in entries for p in _pages_of(e)}
        unread = [p for p in pages if p not in read]
        counts.append(f"{label} {len(pages) - len(unread)} of {len(pages)} "
                      "pages read into the record")
        if unread:
            missing.append(f"{label} PDF pages {_pdf(unread)}")
    skipped = [_entry_line(e) for e in ctx.qa(("skipped",), "pages.")]
    detail = "; ".join(counts)
    if not missing:
        return {"status": "pass", "detail": detail}
    out = {"status": "fail",
           "detail": detail + ". No record entry came from: "
                     + "; ".join(missing)}
    if skipped:
        out["entries"] = skipped[:12]
    return out


def check_explorations_vs_logs(ctx: CheckContext, item: Dict[str, Any]
                               ) -> Dict[str, Any]:
    """The explorations the narrative names and counts, against the logs
    read: the reconciler's count and list checks, and the narrative
    reader's own differences with the logs, as the record carries them."""
    rec = ctx.record
    if rec is None:
        return _no_record(item)
    cites = rec.general.citations or {}
    # An answer with a citation is one the narrative STATED; one without was
    # filled in from the logs read, and comparing it with them proves nothing.
    stated = [f for f in EXPLORATION_FIELDS
              if getattr(rec.general, f, None) not in (None, [])
              and cites.get(f)]
    by_kind: Dict[str, List[str]] = {}
    for inv in rec.investigations:
        by_kind.setdefault(inv.kind, []).append(inv.investigation_id or "?")
    logs = "; ".join(f"{kind} {', '.join(ids)}"
                     for kind, ids in by_kind.items()) or "none"
    plan_note = ("names printed only on the location plan are not in the "
                 "record (BACKLOG RI-24), so they are not compared")
    if not stated and not rec.investigations:
        return {"status": "not_run",
                "detail": "the record holds no explorations: the narrative "
                          "names none and no log was read"}
    if not stated:
        return {"status": "unsure",
                "detail": f"logs read: {logs}; the narrative's list and "
                          "counts of explorations are not stated in the "
                          f"record; {plan_note}"}
    entries = ([e for e in ctx.qa(("count_mismatch",))
                if str(e.where).startswith("general.")]
               + [e for e in ctx.qa(("partial",))
                  if e.where in EXPLORATION_FIELDS])
    found = (f"the narrative states {', '.join(stated)}; logs read: {logs}; "
             f"{plan_note}")
    if not entries:
        return {"status": "pass", "detail": found}
    return {"status": "fail",
            "detail": f"{len(entries)} difference(s) between the narrative "
                      f"and the logs read ({found})",
            "entries": [_entry_line(e) for e in entries[:12]]}


def check_lab_ids_vs_logs(ctx: CheckContext, item: Dict[str, Any]
                          ) -> Dict[str, Any]:
    """Each laboratory test linked to an exploration and a sample on its log
    (the reconciler's link, within 0.15 m)."""
    rec = ctx.record
    if rec is None:
        return _no_record(item)
    sheets = ctx.sheets()
    if not sheets:
        return {"status": "not_run",
                "detail": "the record holds no laboratory test sheets"}
    linked = [t for t in sheets if t.linked_sample_id]
    entries = [e for e in ctx.qa(("partial",), "lab_tests.")
               if "summary" not in str(e.where).lower()]
    detail = (f"{len(linked)} of {len(sheets)} laboratory tests linked to an "
              "exploration and a sample on its log")
    if len(linked) == len(sheets) and not entries:
        return {"status": "pass", "detail": detail}
    return {"status": "fail", "detail": detail,
            "entries": [_entry_line(e) for e in entries[:12]]}


def _sheet_kinds_on(rows: Sequence[Any], sheet: Any) -> bool:
    """Whether the summary table carries any value this sheet reports: a
    sheet whose kind the summary does not tabulate at all (chemistry on a
    table of index tests) is not missing from it (BACKLOG DR-21)."""
    from report_ingest.reconciler import CROSS_CHECKS

    carried = {col for col in CROSS_CHECKS
               if any(getattr(r, col, None) is not None for r in rows)}
    result = getattr(sheet, "result", None)
    if result is None:
        return False
    for col in carried:
        if any(getattr(result, name, None) is not None
               for name in CROSS_CHECKS[col]):
            return True
    return False


def check_summary_and_sheets(ctx: CheckContext, item: Dict[str, Any]
                             ) -> Dict[str, Any]:
    """Each summary-table row has a sheet, and each sheet of a kind the
    summary tabulates is on it."""
    from report_ingest.reconciler import _sheets_for

    rec = ctx.record
    if rec is None:
        return _no_record(item)
    tables = ctx.tables()
    if not tables:
        return {"status": "not_run",
                "detail": "the record holds no laboratory summary table"}
    rows = [r for t in tables for r in (getattr(t.result, "rows", None) or [])]
    sheets = ctx.sheets()

    def name(x: Any) -> str:
        depth = getattr(x, "depth_top", None)
        at = f" at {depth.value:g} {depth.unit}".rstrip() if depth else ""
        return f"{x.investigation_id or '?'} {x.sample_id or ''}".strip() + at

    no_sheet = [name(r) for r in rows if not _sheets_for(r, sheets)]
    off = [s for s in sheets if not any(_sheets_for(r, [s]) for r in rows)]
    not_on = [f"{name(s)} ({s.kind})" for s in off if _sheet_kinds_on(rows, s)]
    other_kinds = sorted({s.kind for s in off if not _sheet_kinds_on(rows, s)})
    problems: List[str] = []
    if no_sheet:
        problems.append(f"summary rows with no sheet: "
                        f"{', '.join(no_sheet[:20])}")
    if not_on:
        problems.append(f"sheets not on the summary: {', '.join(not_on[:20])}")
    entries = [_entry_line(e) for e in ctx.qa(("partial",),
                                               "lab_tests.summary_table")]
    note = (f"; sheets of kinds the summary does not tabulate, not counted: "
            f"{', '.join(other_kinds)}" if other_kinds else "")
    if problems or entries:
        out = {"status": "fail",
               "detail": "; ".join(problems or ["the summary names "
                                                "explorations not read"])
                         + note}
        if entries:
            out["entries"] = entries[:5]
        return out
    return {"status": "pass",
            "detail": f"all {len(rows)} summary rows have a sheet and every "
                      f"sheet of a kind the summary tabulates is on it{note}"}


def check_cross_checks(ctx: CheckContext, item: Dict[str, Any]
                       ) -> Dict[str, Any]:
    """The reconciler's QA entries of the kinds the item names."""
    if ctx.record is None:
        return _no_record(item)
    entries = ctx.qa(item.get("kinds"), item.get("where", ""))
    if not entries:
        return {"status": "pass",
                "detail": "the reconciler raised nothing of this kind on the "
                          "record"}
    return {"status": "fail",
            "detail": f"{len(entries)} point(s) the reconciler raised on the "
                      "record",
            "entries": [_entry_line(e) for e in entries[:12]]}


def check_values_cite_pages(ctx: CheckContext, item: Dict[str, Any]
                            ) -> Dict[str, Any]:
    """Every exploration, laboratory test and calculation in the record
    names its pages, and every narrative answer its citation."""
    rec = ctx.record
    if rec is None:
        return _no_record(item)
    entries = ([("exploration", i.investigation_id, i)
                for i in rec.investigations]
               + [("laboratory test", f"{t.kind} {t.investigation_id} "
                   f"{t.sample_id}".strip(), t) for t in rec.lab_tests]
               + [("calculation", c.subject or c.kind, c)
                  for c in rec.calculations])
    uncited = [f"{kind} {label}".strip() for kind, label, e in entries
               if not _pages_of(e)]
    answers: List[str] = []
    for facts in (rec.general, rec.natural_hazards):
        cites = getattr(facts, "citations", None) or {}
        for field in facts.answered():
            if not cites.get(field) and field not in COUNTED_BY_CODE:
                answers.append(field)
    total = len(entries)
    problems: List[str] = []
    if uncited:
        problems.append(f"{len(uncited)} of {total} records name no page: "
                        f"{', '.join(uncited[:15])}")
    if answers:
        problems.append(f"narrative answers with no citation: "
                        f"{', '.join(answers[:20])}")
    if not total and not answers:
        return {"status": "not_run",
                "detail": "the record holds no extracted values"}
    if problems:
        return {"status": "fail", "detail": "; ".join(problems)}
    return {"status": "pass",
            "detail": f"all {total} records name their pages and every "
                      "narrative answer carries a citation"}


CHECKS: Dict[str, Callable[[CheckContext, Dict[str, Any]], Dict[str, Any]]] = {
    "logs_and_lab_read": check_logs_and_lab_read,
    "explorations_vs_logs": check_explorations_vs_logs,
    "lab_ids_vs_logs": check_lab_ids_vs_logs,
    "cross_checks": check_cross_checks,
    "summary_and_sheets": check_summary_and_sheets,
    "values_cite_pages": check_values_cite_pages,
}


def register_check(name: str, fn: Callable[[CheckContext, Dict[str, Any]],
                                           Dict[str, Any]]) -> None:
    """Add (or replace) a code check under ``name``. The hook for a check
    the app does not have yet: a JSON item naming ``check: name`` runs it."""
    CHECKS[str(name)] = fn


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def _run_one(ctx: CheckContext, item: Dict[str, Any]) -> Dict[str, Any]:
    fn = CHECKS.get(str(item.get("check")))
    if fn is None:
        return {"status": "not_run",
                "detail": f"no code check named {item.get('check')!r} is "
                          "wired; judge it"}
    try:
        return fn(ctx, item)
    except Exception as exc:  # noqa: BLE001 - a broken check is judged
        return {"status": "not_run",
                "detail": f"the check failed to run ({type(exc).__name__}: "
                          f"{exc}); judge it"}


def _title(record: Any) -> str:
    if record is None:
        return ""
    try:
        from report_ingest.writers import title_of
        return title_of(record)
    except Exception:  # noqa: BLE001 - a name, never an error
        return record.document.report_id or record.project.name or ""


def run_checklist(record: Any = None, data: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Run the code items on one report's record (a ``ReportRecord``, its
    dict, or the path of its file or folder; ``None`` runs none of them) and
    return them with the judge items for the agent."""
    data = data or load_checklist()
    rec = load_record(record)
    ctx = CheckContext(record=rec)
    code_rows: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    for item in data.get("items") or []:
        row = {"id": item["id"], "section": item.get("section", ""),
               "text": item["text"]}
        if item.get("mode") != "code":
            judge_rows.append(row)
            continue
        row.update(_run_one(ctx, item))
        code_rows.append(row)
        if row.get("status") in ("not_run", "unsure"):
            judge_rows.append({"id": item["id"],
                               "section": item.get("section", ""),
                               "text": item["text"],
                               "code_said": row.get("detail")})
    out = {
        "checklist": data.get("title"),
        "status": data.get("status"),
        "status_note": data.get("status_note"),
        "report": _title(rec),
        "code_checks": code_rows,
        "for_you_to_judge": judge_rows,
        "how_to_report": HOW_TO_REPORT,
    }
    if isinstance(record, (str, os.PathLike)):
        out["record"] = record_path(str(record))
    return out


def failed_lines(record: Any = None, data: Optional[Dict[str, Any]] = None
                 ) -> List[str]:
    """One line per FAILED code check on the record, with its entries."""
    data = data or load_checklist()
    ctx = CheckContext(record=load_record(record))
    out = []
    for item in data.get("items") or []:
        if item.get("mode") != "code":
            continue
        result = _run_one(ctx, item)
        if result.get("status") == "fail":
            line = f"- {item['text']} -> {result.get('detail')}"
            for e in result.get("entries") or []:
                line += f"\n    {e}"
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# The tool (the GeotechStaffEngineer page, behind GEOTECH_REVIEW_CHECKLIST)
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "The review checklist for a geotechnical report (a DRAFT the owner is "
    "still marking up), run on the report's record: the "
    f"{RECORD_NAME} report ingest writes when it reads a report (what each "
    "page is, the explorations and laboratory tests read, the answers taken "
    "from the text, each with its pages). From the record it checks that "
    "every log and laboratory page was read, the explorations the text names "
    "and counts against the logs, each laboratory test against a sample on "
    "its log, the summary table against the sheets (values that differ, "
    "sheets missing either way), units that could not be converted, and "
    "that every value cites its page. It returns the items that need your "
    "judgement: campaigns and dates, groundwater, locations and datum, "
    "consistency, currency of the codes cited, plausibility, "
    "recommendations. {record} With no record, every item comes back for "
    "you to judge. Report against every item.")

#: ``record`` where the ``report_ingest`` tool is offered beside it, and
#: where it is not (no pointer at a tool the agent does not have).
_RECORD_WITH_INGEST = ("`record`: the record path the report_ingest tool "
                       "returned (paths.record), or the folder it wrote.")
_RECORD_ALONE = (f"`record`: the path of a {RECORD_NAME}, or of the folder "
                 "holding one.")

REPORT_CHECKLIST_DESCRIPTION = _DESCRIPTION.format(record=_RECORD_WITH_INGEST)


def checklist_description(ingest_tool: bool = True) -> str:
    """The tool's description, naming the ``report_ingest`` tool only where
    the agent has it."""
    return _DESCRIPTION.format(
        record=_RECORD_WITH_INGEST if ingest_tool else _RECORD_ALONE)


def make_checklist_tool(max_result_chars: int = 16000,
                        ingest_tool: bool = True):
    """The ``report_checklist`` tool: the checklist on a record.
    ``ingest_tool``: whether the agent also has the ``report_ingest`` tool
    (its description names it only then)."""
    from langchain_core.tools import StructuredTool

    def report_checklist(record: str = "") -> str:
        try:
            out = run_checklist(record or None)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"the checklist failed: "
                                        f"{type(exc).__name__}: {exc}"})
        text = json.dumps(out, ensure_ascii=False)
        if max_result_chars > 0 and len(text) > max_result_chars:
            for row in out.get("code_checks") or []:
                row.pop("entries", None)
            text = json.dumps(out, ensure_ascii=False)[:max_result_chars]
        return text

    return StructuredTool.from_function(
        report_checklist, name="report_checklist",
        description=checklist_description(ingest_tool))


__all__ = ["load_checklist", "load_record", "record_path", "run_checklist",
           "failed_lines", "register_check", "make_checklist_tool",
           "checklist_description",
           "compact", "CHECKS", "CheckContext", "DATA_FILE", "RECORD_NAME",
           "STATUSES", "LOG_LABELS", "LAB_LABELS", "COUNTED_BY_CODE",
           "EXPLORATION_FIELDS", "HOW_TO_REPORT",
           "REPORT_CHECKLIST_DESCRIPTION"]
