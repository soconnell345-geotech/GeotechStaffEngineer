"""WP3 measurement: the page's tables, then the lab reader, on 31 sheets.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp3_lab
    ... --only R36,R28        # a subset, by report
    ... --kind gradation      # a subset, by test kind
    ... --tables-only         # no model at all: the BEFORE column by itself
    ... --detail              # every miss on the OPEN sheets
    ... --append              # write the scorecard into MEASUREMENTS.md

WHAT IT MEASURES. For each hand-truthed sheet it scores the numbers in the
page's DETECTED TABLES (BEFORE), then runs
``report_ingest.lab_reader.read_lab_sheet`` over the same open document and
scores the records it built (AFTER). The two columns are the same document
read two ways, so the gap between them is the model and nothing else.

A table cannot say what test it is on or which boring a number belongs to, so
``kind`` and ``link`` have no BEFORE column. That is the honest line, and it
is why the OVERALL columns are not directly comparable: read the metrics.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse
against OpenAI models through Prompter; a score measured on Claude measures a
model that will never do the work. This script drives the DEVELOPMENT engine
so prompts can be iterated here at a keystroke, and its figures go into the
ledger marked as a checkpoint. The real run is
``report_ingest.cluster_scoring.score_on_cluster(..., stages=("lab",),
lab_truth_dir=...)``, which scores through the same
``report_ingest.lab_scoring``.

``--tables-only`` needs no engine, no key and no network. It is the honest
baseline and it is worth re-running whenever planlens changes.

PRIVACY. Reports are IDs and sheets are ``<kind>__<ID>_p<page>``. Nothing here
prints a project name, a firm or a file name, and MEASUREMENTS.md, which IS
committed, gets only IDs, kinds, counts and rates.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from module_work.report_ingest_harness import lab_truth_records as T
from report_ingest.lab_scoring import (
    DEPTH_TOL_M, EXACT_TOL, METRICS, PASSING_TOL, LabScore, Score, pages_of,
    report_of, score_one_sheet, score_tables,
)

LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"

#: The development model. Never quoted as the system's accuracy.
DEV_MODEL = "claude-opus-5"
#: The reports whose lab pages were open to the builder, when the truth
#: folder carries no OPEN.txt.
DEFAULT_OPEN = ("R36", "R28", "R17", "R06")


def run_one(truth: Dict[str, Any], engine: Any, budget: int, di: str
            ) -> Tuple[LabScore, Optional[LabScore]]:
    """``(before, after)``; ``after`` is None when no engine was given."""
    report = report_of(truth)
    pages = pages_of(truth)
    doc = None
    try:
        doc = C.open_report(report, di=di, warn=False)
        if engine is None:
            return score_tables(truth, doc, pages), None
        return score_one_sheet(truth, doc, engine, budget=budget,
                               report_id=report)
    finally:
        if doc is not None:
            doc.close()


def _rate(score: Optional[Score]) -> str:
    if score is None or not score.total:
        return "     -"
    return f"{score.rate:5.0%} {score.found}/{score.total}"


def _sum(scores: Sequence[LabScore], metric: str) -> Score:
    out = Score()
    for score in scores:
        got = score.scores.get(metric)
        if got is not None:
            out += got
    return out


def _block(name: str, group: Sequence[Tuple[LabScore, Optional[LabScore]]],
           lines: List[str]) -> None:
    group = [row for row in group if not row[0].error]
    if not group:
        return
    befores = [b for b, _a in group]
    afters = [a for _b, a in group if a is not None and not a.error]
    header = f"{'metric':<10}{'before':>14}{'after':>14}"
    lines += [f"{name} -- {len(group)} sheet(s)", header, "-" * len(header)]
    for metric in METRICS:
        before = _sum(befores, metric)
        after = _sum(afters, metric) if afters else None
        if not before.total and not (after and after.total):
            continue
        lines.append(f"{metric:<10}{_rate(before):>14}{_rate(after):>14}")
    before_total = _totals(befores)
    after_total = _totals(afters) if afters else None
    lines.append(f"{'OVERALL':<10}{_rate(before_total):>14}"
                 + (f"{_rate(after_total):>14}" if after_total else ""))
    lines.append("")


def _totals(scores: Sequence[LabScore]) -> Score:
    out = Score()
    for score in scores:
        out += score.total
    return out


def report(rows: Sequence[Tuple[LabScore, Optional[LabScore]]],
           openset: Sequence[str], detail: bool = False) -> str:
    lines: List[str] = []
    groups: Dict[str, List[Tuple[LabScore, Optional[LabScore]]]] = {
        "open": [], "blind": []}
    for before, after in rows:
        groups["open" if before.report in openset else "blind"].append(
            (before, after))
    for name in ("open", "blind"):
        _block(name, groups[name], lines)
    _block("all", rows, lines)

    kinds = sorted({before.kind for before, _a in rows})
    if len(kinds) > 1:
        header = f"{'kind':<20}{'sheets':>7}{'before':>14}{'after':>14}"
        lines += [header, "-" * len(header)]
        for kind in kinds:
            group = [r for r in rows if r[0].kind == kind]
            befores = [b for b, _a in group if not b.error]
            afters = [a for _b, a in group if a is not None and not a.error]
            lines.append(
                f"{kind:<20}{len(group):>7}{_rate(_totals(befores)):>14}"
                + (f"{_rate(_totals(afters)):>14}" if afters else ""))
        lines.append("")

    header = (f"{'sheet':<28}{'set':<7}{'before':>12}{'after':>12}"
              f"{'calls':>7}{'zoom':>6}{'unres':>7}{'look':>6}{'s':>7}")
    lines += [header, "-" * len(header)]
    for before, after in rows:
        name = "open" if before.report in openset else "blind"
        if before.error:
            lines.append(f"{before.sheet_id:<28}{name:<7}ERROR "
                         f"{before.error[:50]}")
            continue
        if after is not None and after.error:
            lines.append(f"{before.sheet_id:<28}{name:<7}"
                         f"{_rate(before.total):>12}  READER FAILED")
            continue
        lines.append(
            f"{before.sheet_id:<28}{name:<7}{_rate(before.total):>12}"
            + (f"{_rate(after.total):>12}{after.model_calls:>7}"
               f"{after.tool_calls:>6}{after.unresolved:>7}"
               f"{after.changes:>6}{after.cost.get('seconds', 0.0):>7.0f}"
               if after is not None
               else f"{'-':>12}{'-':>7}{'-':>6}{'-':>7}{'-':>6}{'-':>7}"))
    lines.append("")

    # A run in which every reader call failed must be LOUD, or a scorecard
    # reports a reader that never ran as a reader that found nothing.
    failed = [(b.sheet_id, a.error) for b, a in rows
              if a is not None and a.error]
    if failed:
        lines.append(f"THE READER FAILED ON {len(failed)} OF {len(rows)} "
                     f"SHEET(S). Those sheets are in NO after number above.")
        for sheet_id, error in failed:
            lines.append(f"  {sheet_id}: {str(error)[:160]}")
        lines.append("")

    afters = [a for _b, a in rows if a is not None and not a.error]
    if afters:
        calls = sum(a.model_calls for a in afters)
        zooms = sum(a.tool_calls for a in afters)
        tokens_in = sum(a.cost.get("input_tokens", 0) for a in afters)
        tokens_out = sum(a.cost.get("output_tokens", 0) for a in afters)
        dollars = sum(a.cost.get("dollars", 0.0) for a in afters)
        seconds = sum(a.cost.get("seconds", 0.0) for a in afters)
        n = len(afters)
        lines += [
            f"cost: {calls} model call(s), {zooms} zoom(s), {tokens_in:,} in, "
            f"{tokens_out:,} out, {seconds:.0f} s, ${dollars:.2f}",
            f"per sheet: {calls / n:.1f} call(s), {zooms / n:.1f} zoom(s), "
            f"{tokens_in / n:,.0f} in, {tokens_out / n:,.0f} out, "
            f"{seconds / n:.0f} s, ${dollars / n:.2f}",
            ""]

    if detail:
        lines.append("misses on the OPEN sheets (the blind ones are not "
                     "listed):")
        for before, after in rows:
            if before.report not in openset:
                continue
            for stage, score in (("tables", before), ("record", after)):
                if score is None:
                    continue
                for metric in METRICS:
                    got = score.scores.get(metric)
                    for miss in (got.misses if got else ()):
                        lines.append(
                            f"  {score.sheet_id} {stage} {metric}: {miss}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--kind", default="", help="comma-separated test kinds")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--tables-only", action="store_true",
                    help="score the page's tables alone; no engine, no key, "
                         "no network")
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--budget", type=int, default=4,
                    help="model calls per sheet, the answer included")
    ap.add_argument("--detail", action="store_true",
                    help="list every miss on the OPEN sheets")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
    openset = T.open_reports(DEFAULT_OPEN)
    truths = T.load_truth(only)
    if kinds:
        truths = [t for t in truths if str(t.get("kind") or "") in kinds]
    if not truths:
        print("no truth sheets matched")
        return 1

    engine = None
    if not args.tables_only:
        from report_ingest.engine import ClaudeEngine
        engine = ClaudeEngine(args.model)

    rows: List[Tuple[LabScore, Optional[LabScore]]] = []
    for n, truth in enumerate(truths, 1):
        print(f"[{n}/{len(truths)}] {truth['id']} ...", flush=True)
        rows.append(run_one(truth, engine, args.budget, args.di))

    text = report(rows, openset, detail=args.detail)
    print(text)
    failed = [b.sheet_id for b, a in rows if a is not None and a.error]
    if failed and len(failed) == len(rows):
        print(f"\nEVERY reader call failed ({len(failed)} sheet(s)). Nothing "
              f"was measured; the before column is the tables alone.",
              file=sys.stderr)
    if args.append:
        stamp = datetime.date.today().isoformat()
        head = ("the page's tables only" if args.tables_only
                else f"tables then lab_reader, DEVELOPMENT engine "
                     f"{args.model} (a checkpoint, never a result)")
        block = (f"\n### WP3 lab scorecard, {stamp} -- {head}\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"Tolerances: a depth links within {DEPTH_TOL_M} m, "
                   f"compared in metres whatever the sheet prints; an index "
                   f"value exact to {EXACT_TOL}; a grading within "
                   f"{PASSING_TOL} percent; a curve within the tolerance its "
                   f"own truth file states. `kind` and `link` have no before "
                   f"column: a table cannot answer them. Open set = "
                   f"{', '.join(openset)}.\n\n```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    if failed and len(failed) == len(rows) and not args.tables_only:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
