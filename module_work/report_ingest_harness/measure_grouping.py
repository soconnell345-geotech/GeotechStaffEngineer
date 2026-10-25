"""BACKLOG RI-87 piece 3: the model-first grouping, scored offline.

Run from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_grouping
    ... measure_grouping --describers haiku --only R01 R02
    ... measure_grouping --describers haiku --only R01 --overlay <folder>

No model is called. For each describer's saved RI-73 run it rebuilds every
reader's groups (:func:`report_ingest.model_first_pages.group_pages`) from
that run's range labels and page descriptions, and scores them against the
answer sheets' ITEM PAGES -- which pages each hand-truthed log, laboratory
sheet and calculation sits on, and nothing else of the answer sheet.

WHERE AN ITEM'S PAGES COME FROM: the rule the stage's scorers use
(``cluster_scoring._score_ingest_record``): a log's ``pages``, a lab sheet's
page out of its id (``lab_scoring.pages_of``), a calculation's ``pages``
(``calc_scoring.pages_of``). For a report in the stage's open set
(``OPEN.txt``) the file is read for its ``id`` and page keys ONLY; a blind
report's file is not opened, and its item is the page its file name gives
(``R01_p45`` is page 45), which is what the lab rule reads anyway and, in
this corpus, every log answer sheet's one page. The reports held out of
every rule (``measure_oos.BLIND``) are blind in every stage, whatever a
stage's ``OPEN.txt`` says.

WHAT IT MEASURES, per stage, per report and in total:

* **whole**: one group holds every page of the item. The parts of one long
  log count as one group (they are joined after reading).
* **missed**: item pages in no group of the stage's reader -- the range label
  sent them elsewhere (the label each one got is listed);
* **split**: item pages spread over two groups or more;
* **mixed**: a group holding the item also holds another item's pages. A
  fault for logs and calculations, where one call reads one item; a lab
  window reads several sheets by design, so lab mixing is counted, not
  faulted;
* **extra**: pages in the item's group(s) that are not the item's (a
  calculation: another printout read in the same call). A log answer sheet
  covers one sheet, so a log's extra pages are its other sheets, counted
  and not faulted;
* **pages in no-item groups**: the reader's other pages, which no answer
  sheet covers (coverage, not error).

Logs have one-page answer sheets, so ``whole`` cannot tell a good log
grouping from a bad one. Proxies that use no answer sheet are added, from
the sheet numbers and explorations the descriptions give: a group whose
pages are sheets 1, 2, .., n of n is **complete**; one that starts past
sheet 1 or stops before sheet n is **cut**; one with numbers out of that
order is **broken**; a group with **more than one sheet 1**, or whose first
sheet says n but which holds another number of pages, joined or cut a set;
a group naming **more than one exploration** joined logs, and an
exploration found in **more than one group** was cut or sits in two places.
And the log boundaries the two describers agree on.

``--overlay`` lays descriptions from another folder (``<ID>/pNNNN.json``,
as ``run_range_labels probe`` saves them) over a describer's own, and
scores that as one more describer: a re-describe of some pages, with the
same range labels.

Calculations are scored three ways: the module's rule (a new printout
where the descriptions' sheet numbers OR today's page heads say so), today's
heads alone (``split_calc_runs`` on the opened report) and the descriptions
alone, to say what each signal catches that the other misses.

WHERE IT WRITES: ``<main checkout>/module_work/field_feedback/
2026-10-24_claude_local/raw/grouping/`` (gitignored; refused otherwise):
``results.json`` and ``RESULTS.md``, which carry report IDs and page numbers
only. Nothing here prints a description, a title or any answer-sheet value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from module_work.report_ingest_harness import run_claude_local as rcl
from module_work.report_ingest_harness.measure_oos import BLIND
from report_ingest import model_first_pages as mfp

#: The describers' saved runs, under the claude-local output folder.
DESCRIBERS: Dict[str, str] = {"haiku": "ri73_full_haiku",
                              "sonnet": "ri73_full_sonnet"}
#: The range-label arm the plan names (RI-73's conventions B).
DEFAULT_ARM = "hinted_B"
#: The stages with answer sheets that carry item pages.
STAGES: Tuple[str, ...] = ("logs", "lab", "calc")
#: Where the results go, under the claude-local output folder.
OUT_NAME = "grouping"
#: The log proxies listed group by group, and their headings.
PROXY_LISTS: Tuple[Tuple[str, str], ...] = (
    ("cut", "whose sheet set is cut"),
    ("broken", "whose sheet numbers are out of order"),
    ("sheet1_many", "with more than one sheet 1"),
    ("count_differs", "whose first sheet gives another count of pages"),
    ("holes_many", "whose pages name more than one exploration"),
    ("ids_many_groups", "sharing one exploration"))
#: The score's columns, in the order they are printed.
CALC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("logs", "logs"), ("lab", "lab"), ("calc", "calc (both)"),
    ("calc_heads", "calc (heads)"), ("calc_descriptions", "calc (descr.)"))


# ---------------------------------------------------------------------------
# the items
# ---------------------------------------------------------------------------

def item_pages(stage: str, path: Path, is_open: bool) -> List[int]:
    """One answer sheet's pages by the stage's own rule, reading an open
    file's ``id`` and page keys only and a blind file not at all."""
    from report_ingest.calc_scoring import pages_of as calc_pages_of
    from report_ingest.lab_scoring import pages_of as lab_pages_of

    keys: Dict[str, Any] = {"id": path.stem}
    if is_open:
        blob = json.loads(path.read_text(encoding="utf-8"))
        keys = {k: blob[k] for k in ("id", "pages", "first_page", "n_pages")
                if k in blob}
        keys.setdefault("id", path.stem)
    if stage == "logs":
        pages = [int(p) for p in keys.get("pages") or []]
        return pages or lab_pages_of(keys)
    if stage == "lab":
        return lab_pages_of(keys)
    return calc_pages_of(keys)


def load_items(truth_root: Path, rids: Sequence[str]
               ) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """``stage -> report -> item id -> pages`` for the reports named."""
    out: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for stage in STAGES:
        folder = truth_root / stage
        openset = set(rcl.open_reports(stage, folder)) - set(BLIND)
        out[stage] = {}
        for path in sorted(folder.glob("*.json")):
            rid = rcl.report_of(path.stem)
            if rid not in rids:
                continue
            pages = item_pages(stage, path, rid in openset)
            out[stage].setdefault(rid, {})[path.stem] = pages
    return out


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

def units_of(stage: str, groups: Sequence[mfp.PageGroup]
             ) -> List[List[int]]:
    """The groups as the score sees them: a long log's parts joined, every
    other group as it is."""
    if stage != "logs":
        return [list(g.pages) for g in groups]
    joined: Dict[str, List[int]] = {}
    order: List[str] = []
    for g in groups:
        key = g.key or f"log-{g.pages[0]}"
        if key not in joined:
            joined[key] = []
            order.append(key)
        joined[key].extend(g.pages)
    return [sorted(joined[k]) for k in order]


def score_stage(stage: str, units: Sequence[Sequence[int]],
                items: Mapping[str, Sequence[int]],
                labels: Mapping[int, str]) -> Dict[str, Any]:
    """One stage of one report: a row per item, and the pages of groups
    that hold none."""
    where: Dict[int, int] = {}
    for n, unit in enumerate(units):
        for page in unit:
            where[int(page)] = n
    holders: Dict[int, List[str]] = {}
    for item, pages in items.items():
        for n in {where[p] for p in pages if p in where}:
            holders.setdefault(n, []).append(item)
    rows = []
    for item, pages in sorted(items.items()):
        mine = sorted({where[p] for p in pages if p in where})
        missed = [p for p in pages if p not in where]
        groups = [list(units[n]) for n in mine]
        others = sorted({o for n in mine for o in holders.get(n, [])
                         if o != item})
        extra = sorted(p for g in groups for p in g if p not in set(pages))
        rows.append({
            "item": item, "pages": list(pages),
            "whole": len(mine) == 1 and not missed,
            "missed": [{"page": p, "label": labels.get(p, "")}
                       for p in missed],
            "split": len(mine) > 1,
            "groups": [[g[0], g[-1], len(g)] for g in groups],
            "mixed_with": others,
            "extra": extra})
    empty = [list(u) for n, u in enumerate(units) if n not in holders]
    return {"items": rows, "n_groups": len(units),
            "n_pages": sum(len(u) for u in units),
            "empty_groups": len(empty),
            "empty_pages": sum(len(u) for u in empty)}


def log_proxies(groups: Sequence[mfp.PageGroup],
                descriptions: Mapping[int, Mapping[str, Any]]
                ) -> Dict[str, Any]:
    """What the printed sheet numbers and the explorations say about the
    log groups, with no answer sheet (the module docstring's proxies)."""
    units = units_of("logs", groups)
    named = mfp.page_identifiers([p for u in units for p in u], descriptions)
    counts: Counter = Counter()
    lists: Dict[str, List[Any]] = {"cut": [], "broken": [], "sheet1_many": [],
                                   "count_differs": [], "holes_many": [],
                                   "ids_many_groups": []}
    seen: Dict[str, List[int]] = {}
    for n, unit in enumerate(units):
        sheets = [mfp.sheet_of(descriptions.get(p)) for p in unit]
        holes = sorted({named.get(p) for p in unit} - {None})
        for hole in holes:
            seen.setdefault(hole, []).append(n)
        span = [unit[0], unit[-1], len(unit)]
        if len(holes) > 1:
            counts["holes>1"] += 1
            lists["holes_many"].append(span[:2] + [holes])
        ones = sum(1 for s in sheets if s and s[0] == 1)
        if ones > 1:
            counts["sheet1_many"] += 1
            lists["sheet1_many"].append(span + [ones])
        first = sheets[0]
        if first and first[0] == 1 and first[1] != len(unit):
            counts["count_differs"] += 1
            lists["count_differs"].append(span + [first[1]])
        if not any(sheets):
            counts["no_sheet_numbers"] += 1
            continue
        whole_set = [(k + 1, len(unit)) for k in range(len(unit))]
        last = sheets[-1]
        if sheets == whole_set:
            counts["complete_set"] += 1
        elif (first and first[0] > 1) or (last and last[0] < last[1]):
            counts["cut_set"] += 1
            lists["cut"].append(span)
        elif all(sheets):
            counts["broken_set"] += 1
            lists["broken"].append(span)
        else:
            counts["unclear"] += 1
    for hole, where in sorted(seen.items()):
        if len(where) > 1:
            apart = any(b - a > 1 for a, b in zip(where, where[1:]))
            counts["ids_many_groups"] += 1
            counts["ids_many_groups_apart"] += apart
            lists["ids_many_groups"].append(
                [hole, [units[n][0] for n in where], apart])
    return {"counts": dict(counts), **lists}


def boundaries(groups: Sequence[mfp.PageGroup]) -> Dict[int, bool]:
    """``page -> starts a log`` over every log page."""
    out: Dict[int, bool] = {}
    for unit in units_of("logs", groups):
        for n, page in enumerate(unit):
            out[page] = n == 0
    return out


# ---------------------------------------------------------------------------
# one describer
# ---------------------------------------------------------------------------

def run_describer(name: str, folder: Path, arm: str, rids: Sequence[str],
                  items: Mapping[str, Mapping[str, Mapping[str, List[int]]]],
                  docs: Mapping[str, Any], overlay: Optional[Path] = None
                  ) -> Dict[str, Any]:
    """One describer's run scored; ``overlay`` (``<ID>/pNNNN.json``) lays
    a re-describe of some pages over its own descriptions."""
    from report_ingest.range_labels import load_descriptions

    out: Dict[str, Any] = {"describer": name, "folder": folder.name,
                           "arm": arm, "reports": {}}
    for rid in rids:
        run_file = folder / "ranges" / arm / f"{rid}.json"
        if not run_file.is_file():
            continue
        blob = json.loads(run_file.read_text(encoding="utf-8"))
        labels = {int(p): str(l) for p, l in (blob.get("labels")
                                               or {}).items() if l}
        n_pages = int(blob.get("n_pages") or (max(labels) + 1))
        descriptions = load_descriptions(
            str(folder / "ranges" / "descriptions" / rid))
        if overlay is not None:
            laid = load_descriptions(str(overlay / rid))
            if not laid:
                continue
            descriptions.update(laid)
        doc = docs.get(rid)
        groups = mfp.group_pages(labels, descriptions, n_pages=n_pages,
                                 doc=doc)
        report: Dict[str, Any] = {"n_pages": n_pages,
                                  "described": len(descriptions),
                                  "calc_heads": doc is not None}
        for stage in STAGES:
            report[stage] = score_stage(
                stage, units_of(stage, groups[stage]),
                items[stage].get(rid, {}), labels)
        calc_pages = mfp.stage_pages(labels, "calc")
        for key, groups_of in (
                ("calc_heads", lambda: mfp.calc_groups(calc_pages, doc)),
                ("calc_descriptions", lambda: mfp.calc_groups(
                    calc_pages, None, descriptions))):
            report[key] = score_stage("calc", units_of("calc", groups_of()),
                                      items["calc"].get(rid, {}), labels)
        report["log_proxies"] = log_proxies(groups["logs"], descriptions)
        report["log_boundaries"] = {str(p): s for p, s in
                                    boundaries(groups["logs"]).items()}
        report["log_groups"] = [g.to_dict() for g in groups["logs"]]
        report["lab_windows"] = [[g.pages[0], g.pages[-1], len(g.pages),
                                  g.why] for g in groups["lab"]]
        report["calc_groups"] = [[g.pages[0], g.pages[-1], len(g.pages)]
                                 for g in groups["calc"]]
        report["narrative"] = [[g.pages[0], g.pages[-1], len(g.pages)]
                               for g in groups["narrative"]]
        out["reports"][rid] = report
    return out


# ---------------------------------------------------------------------------
# totals and the page
# ---------------------------------------------------------------------------

def totals(part: Mapping[str, Any], key: str) -> Dict[str, int]:
    """One stage's counts over every report of a describer."""
    acc: Counter = Counter()
    for report in part["reports"].values():
        block = report[key]
        for row in block["items"]:
            acc["items"] += 1
            acc["whole"] += row["whole"]
            acc["missed_items"] += bool(row["missed"])
            acc["missed_pages"] += len(row["missed"])
            acc["split"] += row["split"]
            acc["mixed"] += bool(row["mixed_with"])
            acc["extra_items"] += bool(row["extra"])
            acc["extra_pages"] += len(row["extra"])
            acc["item_pages"] += len(row["pages"])
        acc["groups"] += block["n_groups"]
        acc["pages"] += block["n_pages"]
        acc["empty_groups"] += block["empty_groups"]
        acc["empty_pages"] += block["empty_pages"]
    return dict(acc)


def _line(stage: str, t: Mapping[str, int]) -> str:
    return (f"  {stage:<18} items {t.get('items', 0):>3}  whole "
            f"{t.get('whole', 0):>3}  missed {t.get('missed_items', 0):>2} "
            f"({t.get('missed_pages', 0)} pp)  split {t.get('split', 0):>2}"
            f"  mixed {t.get('mixed', 0):>2}  extra "
            f"{t.get('extra_items', 0):>2} ({t.get('extra_pages', 0)} pp)"
            f"  | groups {t.get('groups', 0):>4}  pages "
            f"{t.get('pages', 0):>5}  no-item groups "
            f"{t.get('empty_groups', 0):>4} ({t.get('empty_pages', 0)} pp)")


def _worst(part: Mapping[str, Any], key: str) -> List[str]:
    out: List[str] = []
    for rid, report in sorted(part["reports"].items()):
        for row in report[key]["items"]:
            faults = []
            if row["missed"]:
                faults.append("missed " + ", ".join(
                    f"p{m['page']} ({m['label'] or 'no label'})"
                    for m in row["missed"]))
            if row["split"]:
                faults.append("split over " + ", ".join(
                    f"p{a}-{b}" for a, b, _n in row["groups"]))
            if row["mixed_with"] and key != "lab":
                faults.append("mixed with " + ", ".join(row["mixed_with"]))
            if row["extra"] and key not in ("lab", "logs"):
                faults.append(f"{len(row['extra'])} extra page(s) "
                              f"{_spans(row['extra'])}")
            if faults:
                out.append(f"  {row['item']} (pages "
                           f"{_spans(row['pages'])}): " + "; ".join(faults))
    return out


def _spans(pages: Sequence[int]) -> str:
    out: List[str] = []
    for run in mfp.runs(pages):
        out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ", ".join(out)


def agreement(left: Mapping[str, Any], right: Mapping[str, Any]
              ) -> Dict[str, int]:
    """Log boundaries two describers agree on, over the pages both group as
    logs."""
    acc: Counter = Counter()
    for rid, report in left["reports"].items():
        other = right["reports"].get(rid)
        if not other:
            continue
        a, b = report["log_boundaries"], other["log_boundaries"]
        for page in set(a) & set(b):
            acc["pages"] += 1
            acc["starts_left"] += a[page]
            acc["starts_right"] += b[page]
            acc["starts_both"] += a[page] and b[page]
            acc["disagree"] += a[page] != b[page]
    return dict(acc)


def write_results(out: Path, results: Mapping[str, Any]) -> str:
    lines = ["# RI-87 piece 3: model-first grouping, scored offline", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} by "
             f"`measure_grouping`. No model call. Range-label arm "
             f"`{results['arm']}`; item pages from the answer sheets' file "
             f"names (open reports: the stage's own page rule over the "
             f"file's id and page keys).", ""]
    parts = results["describers"]
    for name, part in parts.items():
        rids = sorted(part["reports"])
        lines += [f"## {name} ({part['folder']}): {len(rids)} reports", "",
                  "```"]
        for key, title in CALC_COLUMNS:
            lines.append(_line(title, totals(part, key)))
        lines.append("```")
        lines += ["", "Per report (whole/items; groups):", "", "```"]
        for rid in rids:
            report = part["reports"][rid]
            cells = []
            for key, short in (("logs", "logs"), ("lab", "lab"),
                               ("calc", "calc"), ("calc_heads", "calcH"),
                               ("calc_descriptions", "calcD")):
                rows = report[key]["items"]
                cells.append(f"{short} {sum(r['whole'] for r in rows)}/"
                             f"{len(rows)} ({report[key]['n_groups']} g)")
            narrative = report["narrative"]
            cells.append(f"narrative {sum(n for _a, _b, n in narrative)} pp "
                         f"in {len(narrative)} part(s)")
            lines.append(f"  {rid} ({report['n_pages']} pp)  "
                         + "  ".join(cells))
        lines.append("```")
        for key, title in CALC_COLUMNS:
            worst = _worst(part, key)
            lines += ["", f"Items with a fault, {title}:", "", "```"]
            lines += worst or ["  none"]
            lines.append("```")
        proxy: Counter = Counter()
        listed: Dict[str, List[str]] = {key: [] for key, _t in PROXY_LISTS}
        for rid, report in sorted(part["reports"].items()):
            proxies = report["log_proxies"]
            proxy.update(proxies["counts"])
            for row in proxies["cut"]:
                listed["cut"].append(f"  {rid} p{row[0]}-{row[1]} "
                                     f"({row[2]} pp)")
            for row in proxies["broken"]:
                listed["broken"].append(f"  {rid} p{row[0]}-{row[1]} "
                                        f"({row[2]} pp)")
            for row in proxies["sheet1_many"]:
                listed["sheet1_many"].append(
                    f"  {rid} p{row[0]}-{row[1]} ({row[2]} pp): {row[3]} "
                    f"pages say sheet 1")
            for row in proxies["count_differs"]:
                listed["count_differs"].append(
                    f"  {rid} p{row[0]}-{row[1]} ({row[2]} pp): its first "
                    f"sheet says {row[3]}")
            for row in proxies["holes_many"]:
                listed["holes_many"].append(f"  {rid} p{row[0]}-{row[1]}: "
                                            f"{len(row[2])} explorations")
            for row in proxies["ids_many_groups"]:
                listed["ids_many_groups"].append(
                    f"  {rid}: one exploration in groups starting "
                    + ", ".join(f"p{p}" for p in row[1])
                    + (" (apart)" if row[2] else " (side by side)"))
        lines += ["", "Log groups by their sheet numbers and explorations "
                      "(no answer sheet):", "", "```",
                  "  " + "  ".join(f"{k} {v}" for k, v in
                                   sorted(proxy.items())), "```"]
        for key, title in PROXY_LISTS:
            lines += ["", f"Log groups {title}:", "", "```"]
            lines += listed[key] or ["  none"]
            lines.append("```")
        lines.append("")
    names = list(parts)
    for n, a in enumerate(names):
        for b in names[n + 1:]:
            agree = agreement(parts[a], parts[b])
            lines += [f"## Log boundaries, {a} against {b}", "", "```",
                      "  " + "  ".join(f"{k} {v}" for k, v in
                                       sorted(agree.items())), "```", ""]
    text = "\n".join(lines)
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1),
                                      encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def open_docs(raw: Path, rids: Sequence[str]) -> Dict[str, Any]:
    """The reports as planlens documents, for the calculation heads."""
    from report_ingest.corpus import Corpus

    corpus = Corpus(raw / "corpus", di_dir=raw / "di")
    out: Dict[str, Any] = {}
    for rid in rids:
        try:
            out[rid] = corpus.open_report(rid, di="auto", warn=False)
        except Exception as exc:                             # noqa: BLE001
            print(f"{rid}: not opened ({type(exc).__name__}); its "
                  f"calculation runs are cut by length only",
                  file=sys.stderr)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--describers", nargs="*", default=list(DESCRIBERS),
                    choices=list(DESCRIBERS))
    ap.add_argument("--arm", default=DEFAULT_ARM,
                    help="the range-label arm (default hinted_B)")
    ap.add_argument("--only", nargs="*", default=None, help="report IDs")
    ap.add_argument("--no-doc", action="store_true",
                    help="do not open the reports: calculation runs are "
                         "cut by length only")
    ap.add_argument("--overlay", default=None,
                    help="a folder of re-described pages (<ID>/pNNNN.json) "
                         "laid over each describer's own, scored as one "
                         "more describer on the reports it covers")
    args = ap.parse_args(argv)

    main_root = rcl.main_checkout()
    raw = main_root / rcl.RAW_REL
    claude = main_root / rcl.OUT_REL
    out = claude / OUT_NAME
    if rcl.git_ignored(out, main_root) is not True:
        print(f"refused: {out} is not ignored by git", file=sys.stderr)
        return 2
    folders = {name: claude / DESCRIBERS[name] for name in args.describers}
    rids = sorted({p.stem for f in folders.values()
                   for p in (f / "ranges" / args.arm).glob("R*.json")})
    if args.only:
        rids = [r for r in rids if r in args.only]
    items = load_items(raw / "truth", rids)
    docs = {} if args.no_doc else open_docs(
        raw, [r for r in rids if r in items["calc"]])
    try:
        parts = {name: run_describer(name, folder, args.arm, rids, items,
                                     docs)
                 for name, folder in folders.items()}
        if args.overlay:
            overlay = Path(args.overlay)
            for name, folder in folders.items():
                part = run_describer(name, folder, args.arm, rids, items,
                                     docs, overlay=overlay)
                part["overlay"] = overlay.name
                parts[f"{name}+{overlay.name}"] = part
        results = {"arm": args.arm, "reports": rids, "describers": parts}
    finally:
        for doc in docs.values():
            try:
                doc.close()
            except Exception:                                # noqa: BLE001
                pass
    out.mkdir(parents=True, exist_ok=True)
    text = write_results(out, results)
    print(text)
    print(f"\nwritten: {out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
