"""The report-review checklist: one data file, used two ways.

The checklist itself is DATA, in ``report_review_checklist.json`` beside
this module, marked DRAFT until the owner has marked it up (plan of record
``module_work/SCALES_COVERAGE_CROSSCHECKS.md``, W4 b). Each item is either:

* ``code`` - a check the app makes itself, from what it has recorded: the
  coverage ledger of what was read (:mod:`funhouse_agent.coverage`), the
  report's own text layer, and the cross-checks ``subsurface.write_diggs``
  runs on the data it is given (report ingest's reconciler, wired there by
  W2); or
* ``judge`` - an item the agent works through and reports against.

:func:`run_checklist` runs every ``code`` item it has a check for and returns
the ``judge`` items for the agent. A ``code`` item whose check cannot run here
(no data written yet, no text layer) is returned with the judgement items, so
nothing on the list is dropped in silence.

Adding a check: write ``fn(ctx, item) -> {"status", "detail", ...}`` and
:func:`register_check` it under the name the JSON item gives as ``check``.
``status`` is ``pass``, ``fail``, ``unsure`` (code found something it cannot
settle: a look is needed) or ``not_run`` (nothing to check here).

The precedent is :mod:`funhouse_agent.review_checklists` (the calculation
reviewers' checklists), which keeps prose; this one keeps rows, because code
runs half of it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from funhouse_agent import coverage as _cov

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "report_review_checklist.json")

STATUSES = ("pass", "fail", "unsure", "not_run")


def load_checklist(path: Optional[str] = None) -> Dict[str, Any]:
    """The checklist data (the shipped DRAFT unless ``path`` is given)."""
    with open(path or DATA_FILE, encoding="utf-8") as fh:
        data = json.load(fh)
    for item in data.get("items") or []:
        if item.get("mode") not in ("code", "judge"):
            raise ValueError(f"checklist item {item.get('id')!r}: mode must "
                             "be 'code' or 'judge'")
    return data


# ---------------------------------------------------------------------------
# Exploration ids in a text layer
# ---------------------------------------------------------------------------

#: Prefixes an exploration id is printed with. "B" only with a hyphen
#: ("B-1"); the others with a hyphen, a space or nothing ("BH3").
_ID_RE = re.compile(
    r"(?<![A-Za-z0-9.\-/])(BH|TB|TP|TH|CPTU|SCPT|CPT|DCP|MW|DH|HA|B(?=-))"
    r"[-\s]?(\d{1,3}[A-Z]?)(?![A-Za-z0-9])")
#: Words that make "B-1" something else: "Table B-1", "Figure B-2".
_NOT_AN_EXPLORATION = re.compile(
    r"(?:table|tab\.|figure|fig\.?|appendix|app\.|plate|sheet|section|"
    r"drawing|dwg\.?|exhibit|annex|attachment|page|item|note|detail|"
    r"type|grade|zone|class)\s*$", re.IGNORECASE)


def exploration_ids(text: str) -> Set[str]:
    """Exploration ids in ``text``, normalised to ``PREFIX-N``."""
    out: Set[str] = set()
    for m in _ID_RE.finditer(text or ""):
        before = (text or "")[max(0, m.start() - 14):m.start()]
        if _NOT_AN_EXPLORATION.search(before):
            continue
        out.add(f"{m.group(1).upper()}-{m.group(2).upper()}")
    return out


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class CheckContext:
    """What a check may use."""
    ledger: Any
    key: str
    _doc: Any = None
    _opened: bool = False
    _lines: Dict[int, List[Any]] = field(default_factory=dict)

    @property
    def inventory(self):
        return self.ledger.docs[self.key]["inventory"]

    @property
    def outputs(self) -> List[Dict[str, Any]]:
        return list(self.ledger.outputs or [])

    def document(self):
        """The planlens Document, reopened from the source it was read
        from; ``None`` when that is no longer to hand."""
        if not self._opened:
            self._opened = True
            src = self.ledger.docs[self.key].get("source")
            try:
                from funhouse_agent import document_tools
                entry = document_tools.open_document_entry(
                    src, self.ledger.attachments)
                self._doc = entry.doc
            except Exception:  # noqa: BLE001 - text checks then do not run
                self._doc = None
        return self._doc

    def lines(self, page: int) -> List[Any]:
        if page not in self._lines:
            doc = self.document()
            try:
                self._lines[page] = list(doc.page(page, tables=False).lines) \
                    if doc is not None else []
            except Exception:  # noqa: BLE001
                self._lines[page] = []
        return self._lines[page]

    def text(self, page: int) -> str:
        return "\n".join(ln.text for ln in self.lines(page) if ln.text)

    def top_text(self, page: int, fraction: float = 0.3) -> str:
        """The text in the top ``fraction`` of a page (a log's title block)."""
        doc = self.document()
        try:
            height = float(doc.page_map()[page].height)
        except Exception:  # noqa: BLE001
            height = 792.0
        return "\n".join(ln.text for ln in self.lines(page)
                         if ln.text and ln.bbox[1] < fraction * height)

    def latest_output(self) -> Optional[Dict[str, Any]]:
        outs = [o for o in self.outputs if o.get("cross_checks") is not None
                or o.get("rows")]
        return outs[-1] if outs else None


def _pages_of(ctx: CheckContext, groups) -> List[int]:
    return [p.page for p in ctx.inventory.pages if p.group in groups]


def _logged_ids(ctx: CheckContext):
    """(ids printed at the top of the log pages, log pages with no text)."""
    ids: Set[str] = set()
    blind: List[int] = []
    for p in _pages_of(ctx, ("logs",)):
        info = ctx.inventory.info(p)
        if info is not None and not info.has_text:
            blind.append(p)
            continue
        ids |= exploration_ids(ctx.top_text(p))
    for it in ctx.inventory.items:
        if it.get("kind") in ("boring_log", "test_pit_log", "cpt_log",
                              "dcp_log") and it.get("title"):
            ids |= exploration_ids(str(it["title"]))
    return ids, blind


#: Groups whose pages NAME explorations (a text, a plan, a list), as opposed
#: to the logs that record them and the sheets that test their samples.
_NAMING_GROUPS = ("narrative", "drawings", "other", "photos", "front")


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def check_logs_and_lab_read(ctx: CheckContext, item: Dict[str, Any]
                            ) -> Dict[str, Any]:
    rows = ctx.ledger.group_rows(ctx.key, ("logs", "lab"))
    if not rows:
        return {"status": "not_run",
                "detail": "the document has no pages read as exploration "
                          "logs or laboratory sheets"}
    counts = "; ".join(f"{r['label']} {r['read']} of {r['pages']} read"
                       + (f", {r['skipped']} skipped" if r["skipped"] else "")
                       for r in rows)
    missing = [r for r in rows if r.get("not_read")]
    if not missing:
        return {"status": "pass", "detail": counts}
    return {"status": "fail",
            "detail": counts + ". Not read: " + "; ".join(
                f"{r['label']} PDF pages {r['not_read_pdf']}"
                for r in missing)}


def check_explorations_vs_logs(ctx: CheckContext, item: Dict[str, Any]
                               ) -> Dict[str, Any]:
    if ctx.document() is None:
        return {"status": "not_run",
                "detail": "the document is no longer open to read its text"}
    named: Set[str] = set()
    for p in _pages_of(ctx, _NAMING_GROUPS):
        named |= exploration_ids(ctx.text(p))
    logged, blind = _logged_ids(ctx)
    if not named and not logged:
        return {"status": "not_run",
                "detail": "no exploration ids were found in the text layer"
                          + (f"; log pages {_cov.compact(blind)} have no "
                             "text layer" if blind else "")}
    no_log = sorted(named - logged)
    not_named = sorted(logged - named)
    found = (f"named in the text: {', '.join(sorted(named)) or 'none'}; "
             f"logs found: {', '.join(sorted(logged)) or 'none'}")
    if not no_log and not not_named:
        return {"status": "pass", "detail": found}
    problems = []
    if no_log:
        problems.append(f"named with no log in the text layer: "
                        f"{', '.join(no_log)}")
    if not_named:
        problems.append(f"logs never named in the text or on the plan: "
                        f"{', '.join(not_named)}")
    if no_log and blind:
        return {"status": "unsure",
                "detail": "; ".join(problems) + f". Log pages "
                          f"{_cov.compact(blind)} (PDF pages "
                          f"{_cov.compact(blind, 1)}) have no text layer and "
                          "may hold them: look at them to settle it. ("
                          + found + ")"}
    return {"status": "fail", "detail": "; ".join(problems) + f". ({found})"}


def _cross_entries(ctx: CheckContext, kinds=None, where: str = ""):
    out = ctx.latest_output()
    if out is None:
        return None, ("not run: these cross-checks run inside "
                      "subsurface.write_diggs on the data it is given, and no "
                      "data has been written in this conversation")
    cc = out.get("cross_checks")
    if not isinstance(cc, dict):
        return None, "not run: the data written carried no cross-checks"
    if cc.get("error"):
        return None, f"not run: {cc['error']}"
    entries = [e for e in cc.get("entries") or [] if isinstance(e, dict)]
    if kinds:
        entries = [e for e in entries if e.get("kind") in set(kinds)]
    if where:
        w = where.lower()
        entries = [e for e in entries
                   if w in str(e.get("where", "")).lower()
                   or w in str(e.get("detail", "")).lower()]
    return entries, None


def _entry_line(e: Dict[str, Any]) -> str:
    line = f"{e.get('kind')}: {e.get('detail', '')}"
    if e.get("values"):
        line += f" (values {', '.join(str(v) for v in e['values'])})"
    if e.get("pages"):
        line += f" [PDF pages {_cov.compact(e['pages'], 1)}]"
    return line


def check_cross_checks(ctx: CheckContext, item: Dict[str, Any]
                       ) -> Dict[str, Any]:
    entries, why = _cross_entries(ctx, item.get("kinds"),
                                  item.get("where", ""))
    if entries is None:
        return {"status": "not_run", "detail": why}
    if not entries:
        return {"status": "pass",
                "detail": "the cross-checks of the data written raised "
                          "nothing of this kind"}
    return {"status": "fail",
            "detail": f"{len(entries)} point(s) from the cross-checks of the "
                      "data written",
            "entries": [_entry_line(e) for e in entries[:12]]}


def check_lab_ids_vs_logs(ctx: CheckContext, item: Dict[str, Any]
                          ) -> Dict[str, Any]:
    problems: List[str] = []
    notes: List[str] = []
    unsure = False
    if ctx.document() is not None:
        logged, blind = _logged_ids(ctx)
        on_sheets: Set[str] = set()
        for p in _pages_of(ctx, ("lab",)):
            on_sheets |= exploration_ids(ctx.text(p))
        orphans = sorted(on_sheets - logged)
        if orphans:
            if blind:
                unsure = True
                notes.append(f"lab sheets name {', '.join(orphans)}, with no "
                             f"log found in the text layer; log pages "
                             f"{_cov.compact(blind)} have no text layer")
            else:
                problems.append(f"lab sheets name explorations with no log: "
                                f"{', '.join(orphans)}")
    # The reconciler's link of each lab test to its boring and sample is a
    # 'partial' on lab_tests.<kind>; its summary-table points are not this.
    entries, why = _cross_entries(ctx, ("partial",), "lab_tests")
    if entries:
        entries = [e for e in entries
                   if "summary" not in str(e.get("where", "")).lower()]
        problems += [_entry_line(e) for e in entries[:10]]
    if problems:
        return {"status": "fail", "detail": "; ".join(problems + notes)}
    if unsure:
        return {"status": "unsure", "detail": "; ".join(notes)}
    if entries is None and ctx.document() is None:
        return {"status": "not_run", "detail": why}
    return {"status": "pass" if entries is not None else "unsure",
            "detail": ("the lab sheets name only explorations that have logs"
                       + ("" if entries is not None else
                          "; sample ids and depths are matched only by the "
                          "cross-checks of written data (" + str(why) + ")"))}


def _fold(label: str) -> str:
    """"B-2 S-3" and "B2 S3" are the same sample."""
    return re.sub(r"[\s\-_.]+", "", str(label).split(" (")[0]).upper()


def check_summary_and_sheets(ctx: CheckContext, item: Dict[str, Any]
                             ) -> Dict[str, Any]:
    """Each summary-table row has a sheet, and each sheet is on the summary
    - from the data given to ``write_diggs``, plus the cross-checks' own
    point about summary rows naming explorations that were not read."""
    out = ctx.latest_output()
    if out is None or not out.get("summary_rows"):
        return {"status": "not_run",
                "detail": ("no laboratory summary table has been written as "
                           "data in this conversation (subsurface.write_diggs "
                           "takes it as a lab test of kind summary_table)"
                           if out is not None else
                           "no data has been written in this conversation")}
    rows = {_fold(r): r for r in out.get("summary_rows") or []}
    sheets = {_fold(s): s for s in out.get("sheets") or []}
    no_sheet = sorted(rows[k] for k in rows if k not in sheets)
    not_on = sorted(sheets[k] for k in sheets if k not in rows)
    entries, _why = _cross_entries(ctx, ("partial",), "summary")
    problems = []
    if no_sheet:
        problems.append(f"summary rows with no sheet: {', '.join(no_sheet[:20])}")
    if not_on:
        problems.append(f"sheets not on the summary: {', '.join(not_on[:20])}")
    problems += [_entry_line(e) for e in (entries or [])[:5]]
    if problems:
        return {"status": "fail", "detail": "; ".join(problems)}
    return {"status": "pass",
            "detail": f"all {len(rows)} summary rows have a sheet and every "
                      "sheet is on the summary"}


def check_values_cite_pages(ctx: CheckContext, item: Dict[str, Any]
                            ) -> Dict[str, Any]:
    out = ctx.latest_output()
    if out is None:
        return {"status": "not_run",
                "detail": "no data has been written in this conversation "
                          "(subsurface.write_diggs keeps each value's pages)"}
    uncited = out.get("rows_without_pages") or []
    cited = _cov.parse_pages(out.get("pages_cited"))
    unread = [p for p in cited if not ctx.ledger.covered(ctx.key, p)]
    problems = []
    if uncited:
        problems.append(f"{len(uncited)} of {out.get('rows', 0)} records "
                        f"written cite no page: {', '.join(uncited[:15])}")
    if unread:
        problems.append(f"pages cited that no tool read: PDF pages "
                        f"{_cov.compact(unread, 1)}")
    if problems:
        return {"status": "fail", "detail": "; ".join(problems)}
    return {"status": "pass",
            "detail": f"all {out.get('rows', 0)} records written cite pages, "
                      "and every page cited was read"}


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

def run_checklist(ledger, key: str, data: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Run the code items for one document; return them with the judgement
    items for the agent."""
    data = data or load_checklist()
    ctx = CheckContext(ledger=ledger, key=key)
    code_rows: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    for item in data.get("items") or []:
        row = {"id": item["id"], "section": item.get("section", ""),
               "text": item["text"]}
        if item.get("mode") != "code":
            judge_rows.append(row)
            continue
        fn = CHECKS.get(str(item.get("check")))
        if fn is None:
            result = {"status": "not_run",
                      "detail": f"no code check named {item.get('check')!r} "
                                "is wired; judge it"}
        else:
            try:
                result = fn(ctx, item)
            except Exception as exc:  # noqa: BLE001 - a broken check
                result = {"status": "not_run",
                          "detail": f"the check failed to run "
                                    f"({type(exc).__name__}: {exc}); judge it"}
        row.update(result)
        code_rows.append(row)
        if row.get("status") in ("not_run", "unsure"):
            judge_rows.append({"id": item["id"],
                               "section": item.get("section", ""),
                               "text": item["text"],
                               "code_said": row.get("detail")})
    return {
        "checklist": data.get("title"),
        "status": data.get("status"),
        "status_note": data.get("status_note"),
        "document": ledger.docs[key]["name"],
        "code_checks": code_rows,
        "for_you_to_judge": judge_rows,
        "how_to_report": (
            "Report the code checks as they stand (a failed one is a "
            "finding: give its detail and pages). Work through every item "
            "for you to judge against the report and report each as met, "
            "not met or cannot tell, with the pages you looked at; where "
            "code_said is given, start from it."),
    }


def failed_lines(ledger, key: str, data: Optional[Dict[str, Any]] = None
                 ) -> List[str]:
    """One line per FAILED code check (for the coverage gate's note)."""
    data = data or load_checklist()
    ctx = CheckContext(ledger=ledger, key=key)
    out = []
    for item in data.get("items") or []:
        if item.get("mode") != "code":
            continue
        if item.get("check") == "logs_and_lab_read":
            continue          # the gate's own list says this already
        fn = CHECKS.get(str(item.get("check")))
        if fn is None:
            continue
        try:
            result = fn(ctx, item)
        except Exception:  # noqa: BLE001
            continue
        if result.get("status") == "fail":
            line = f"- {item['text']} -> {result.get('detail')}"
            for e in result.get("entries") or []:
                line += f"\n    {e}"
            out.append(line)
    return out


__all__ = ["load_checklist", "run_checklist", "failed_lines",
           "register_check", "exploration_ids", "CHECKS", "CheckContext",
           "DATA_FILE", "STATUSES"]
