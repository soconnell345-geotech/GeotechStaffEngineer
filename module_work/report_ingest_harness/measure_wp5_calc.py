"""WP5 measurement: the floor, then the calculation reader, on ten runs.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp5_calc
    ... --only R18,R23          # a subset, by report
    ... --kind settlement       # a subset, by calculation kind
    ... --floor-only            # no model at all: the BEFORE column by itself
    ... --detail                # every miss
    ... --append                # write the scorecard into MEASUREMENTS.md

WHAT IT MEASURES. For each hand-truthed calculation it scores the FLOOR --
every (label, value) a pattern reads off the pages' tables and lines with no
model at all (``report_ingest.calc_reader.floor_from_pages``) -- and then
runs ``read_calculation`` over the same open document and scores the record
it built. The two columns are the same pages read two ways, so the gap
between them is the model and nothing else.

A pattern over a page cannot say what a calculation works out, what method it
used or what it is for, so ``kind``, ``method`` and ``subject`` have no
BEFORE column; ``program`` does, because a banner is a pattern. That is why
the two OVERALL columns are not directly comparable: read the metrics.

THERE IS NO BLIND SET. All ten runs were read while the reader's prompt was
written, so every number this prints is IN SAMPLE and none of it is evidence
about an unseen report. A blind set is owed and the scorecard says so.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse
against OpenAI models through Prompter; a score measured on Claude measures a
model that will never do the work. This script drives the DEVELOPMENT engine
so prompts can be iterated here at a keystroke, and its figures go into the
ledger marked as a checkpoint. The real run is
``report_ingest.cluster_scoring.score_on_cluster(..., stages=("calc",),
truth_dir=...)``, which scores through the same
``report_ingest.calc_scoring``.

``--floor-only`` needs no engine, no key and no network. It is the honest
baseline and it is worth re-running whenever planlens changes.

PRIVACY. Reports are IDs and calculations are ``<kind>__<ID>_p<first page>``.
Nothing here prints a project name, a firm or a file name, and
MEASUREMENTS.md, which IS committed, gets only IDs, kinds, counts and rates.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.calc_scoring import (
    METRICS, MODEL_ONLY, NAME_RATIO, PROGRAM_RATIO, VALUE_TOL, CalcScore,
    Score, pages_of, report_of, score_floor, score_one_calc,
)

LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"
TRUTH_DIR = C.RAW_DIR / "truth" / "calc"

#: The development model. Never quoted as the system's accuracy.
DEV_MODEL = "claude-opus-5"


def truth_files() -> List[Path]:
    return sorted(TRUTH_DIR.glob("*.json"))


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every hand-truthed calculation, or the ones whose report is in ``only``."""
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed calculations ({TRUTH_DIR} is gitignored and "
            f"exists only where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in truth_files():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if only and report_of(blob) not in only:
            continue
        out.append(blob)
    return out


def open_reports(default: Sequence[str] = ()) -> Tuple[str, ...]:
    """The reports whose calculation pages the builder was allowed to see."""
    path = TRUTH_DIR / "OPEN.txt"
    if path.is_file():
        names = tuple(line.strip() for line
                      in path.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return tuple(default)


def run_one(truth: Dict[str, Any], engine: Any, budget: int, di: str
            ) -> Tuple[CalcScore, Optional[CalcScore]]:
    """``(before, after)``; ``after`` is None when no engine was given."""
    from report_ingest.calc_reader import floor_from_pages

    report = report_of(truth)
    doc = None
    try:
        doc = C.open_report(report, di=di, warn=False)
        if engine is None:
            pages = pages_of(truth)
            floor = floor_from_pages(doc, pages, report)
            score = score_floor(truth, floor.as_calculation(report))
            score.n_pages = len(pages)
            score.floor_values = floor.n_values
            return score, None
        return score_one_calc(truth, doc, engine, budget=budget,
                              report_id=report)
    finally:
        if doc is not None:
            doc.close()


def _rate(score: Optional[Score]) -> str:
    if score is None or not score.total:
        return "     -"
    return f"{score.rate:5.0%} {score.found}/{score.total}"


def _sum(scores: Sequence[CalcScore], metric: str) -> Score:
    out = Score()
    for score in scores:
        got = score.scores.get(metric)
        if got is not None:
            out += got
    return out


def _totals(scores: Sequence[CalcScore], skip: Sequence[str] = ()) -> Score:
    out = Score()
    for score in scores:
        for metric in METRICS:
            if metric in skip:
                continue
            got = score.scores.get(metric)
            if got is not None:
                out += got
    return out


def _block(name: str, group: Sequence[Tuple[CalcScore, Optional[CalcScore]]],
           lines: List[str]) -> None:
    group = [row for row in group if not row[0].error]
    if not group:
        return
    befores = [b for b, _a in group]
    afters = [a for _b, a in group if a is not None and not a.error]
    header = f"{'metric':<10}{'before':>14}{'after':>14}"
    lines += [f"{name} -- {len(group)} run(s)", header, "-" * len(header)]
    for metric in METRICS:
        before = _sum(befores, metric)
        after = _sum(afters, metric) if afters else None
        if not before.total and not (after and after.total):
            continue
        shown = "-" if metric in MODEL_ONLY else _rate(before)
        lines.append(f"{metric:<10}{shown:>14}"
                     + (f"{_rate(after):>14}" if after else ""))
    before_total = _totals(befores, skip=MODEL_ONLY)
    after_total = _totals(afters) if afters else None
    lines.append(f"{'OVERALL':<10}{_rate(before_total):>14}"
                 + (f"{_rate(after_total):>14}" if after_total else ""))
    lines.append("(the before OVERALL leaves out the metrics the floor is "
                 "not asked; read the metrics)")
    lines.append("")


def report(rows: Sequence[Tuple[CalcScore, Optional[CalcScore]]],
           openset: Sequence[str], detail: bool = False) -> str:
    lines: List[str] = []
    groups: Dict[str, List[Tuple[CalcScore, Optional[CalcScore]]]] = {
        "open": [], "blind": []}
    for before, after in rows:
        groups["open" if before.report in openset
               else "blind"].append((before, after))
    for name in ("open", "blind"):
        _block(name, groups[name], lines)
    _block("all", rows, lines)

    kinds = sorted({before.kind for before, _a in rows})
    if len(kinds) > 1:
        header = (f"{'kind':<28}{'runs':>6}{'pages':>7}{'before':>14}"
                  f"{'after':>14}")
        lines += [header, "-" * len(header)]
        for kind in kinds:
            group = [r for r in rows if r[0].kind == kind]
            befores = [b for b, _a in group if not b.error]
            afters = [a for _b, a in group if a is not None and not a.error]
            pages = sum(b.n_pages for b in befores)
            lines.append(
                f"{kind:<28}{len(group):>6}{pages:>7}"
                f"{_rate(_totals(befores, skip=MODEL_ONLY)):>14}"
                + (f"{_rate(_totals(afters)):>14}" if afters else ""))
        lines.append("")

    header = (f"{'calculation':<40}{'set':<7}{'pp':>4}{'floor':>6}"
              f"{'before':>12}{'after':>12}{'calls':>6}{'zoom':>5}"
              f"{'unres':>6}{'misp':>5}{'s':>6}")
    lines += [header, "-" * len(header)]
    for before, after in rows:
        name = "open" if before.report in openset else "blind"
        if before.error:
            lines.append(f"{before.calc_id:<40}{name:<7}ERROR "
                         f"{before.error[:50]}")
            continue
        if after is not None and after.error:
            lines.append(
                f"{before.calc_id:<40}{name:<7}{before.n_pages:>4}"
                f"{before.floor_values:>6}"
                f"{_rate(_totals([before], skip=MODEL_ONLY)):>12}"
                f"  READER FAILED: {str(after.error)[:60]}")
            continue
        lines.append(
            f"{before.calc_id:<40}{name:<7}{before.n_pages:>4}"
            f"{before.floor_values:>6}"
            f"{_rate(_totals([before], skip=MODEL_ONLY)):>12}"
            + (f"{_rate(after.total):>12}{after.model_calls:>6}"
               f"{after.tool_calls:>5}{after.unresolved:>6}"
               f"{after.misplaced:>5}{after.cost.get('seconds', 0.0):>6.0f}"
               if after is not None
               else f"{'-':>12}{'-':>6}{'-':>5}{'-':>6}{'-':>5}{'-':>6}"))
    lines.append("")

    failed = [(b.calc_id, a.error) for b, a in rows
              if a is not None and a.error]
    if failed:
        lines.append(f"THE READER FAILED ON {len(failed)} OF {len(rows)} "
                     f"RUN(S). Those runs are in NO after number above.")
        for calc_id, error in failed:
            lines.append(f"  {calc_id}: {str(error)[:160]}")
        lines.append("")

    afters = [a for _b, a in rows if a is not None and not a.error]
    if afters:
        calls = sum(a.model_calls for a in afters)
        zooms = sum(a.tool_calls for a in afters)
        tokens_in = sum(a.cost.get("input_tokens", 0) for a in afters)
        tokens_out = sum(a.cost.get("output_tokens", 0) for a in afters)
        dollars = sum(a.cost.get("dollars", 0.0) for a in afters)
        seconds = sum(a.cost.get("seconds", 0.0) for a in afters)
        merged = (sum(a.disagreements for a in afters),
                  sum(a.kept for a in afters), sum(a.added for a in afters),
                  sum(a.reconciled for a in afters))
        n = len(afters)
        lines += [
            f"merge: {merged[0]} disagreement(s), {merged[1]} kept from the "
            f"floor, {merged[2]} added by the reader, {merged[3]} reconciled",
            f"cost: {calls} model call(s), {zooms} zoom(s), {tokens_in:,} in, "
            f"{tokens_out:,} out, {seconds:.0f} s, ${dollars:.2f}",
            f"per run: {calls / n:.1f} call(s), {zooms / n:.1f} zoom(s), "
            f"{tokens_in / n:,.0f} in, {tokens_out / n:,.0f} out, "
            f"{seconds / n:.0f} s, ${dollars / n:.2f}",
            ""]

    if detail:
        lines.append("misses:")
        for before, after in rows:
            for stage, score in (("floor", before), ("record", after)):
                if score is None:
                    continue
                for metric in METRICS:
                    got = score.scores.get(metric)
                    for miss in (got.misses if got else ()):
                        lines.append(
                            f"  {score.calc_id} {stage} {metric}: {miss}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--kind", default="",
                    help="comma-separated calculation kinds")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--floor-only", action="store_true",
                    help="score the floor alone; no engine, no key, no "
                         "network")
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--budget", type=int, default=2,
                    help="model calls per calculation, the answer included")
    ap.add_argument("--detail", action="store_true", help="list every miss")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    kinds = {x.strip() for x in args.kind.split(",") if x.strip()}
    openset = open_reports()
    truths = load_truth(only)
    if kinds:
        truths = [t for t in truths if str(t.get("kind") or "") in kinds]
    if not truths:
        print("no truth calculations matched")
        return 1

    engine = None
    if not args.floor_only:
        from report_ingest.engine import ClaudeEngine
        engine = ClaudeEngine(args.model)

    rows: List[Tuple[CalcScore, Optional[CalcScore]]] = []
    for n, truth in enumerate(truths, 1):
        print(f"[{n}/{len(truths)}] {truth['id']} ...", flush=True)
        rows.append(run_one(truth, engine, args.budget, args.di))

    text = report(rows, openset, detail=args.detail)
    print(text)
    failed = [b.calc_id for b, a in rows if a is not None and a.error]
    if failed and len(failed) == len(rows):
        print(f"\nEVERY reader call failed ({len(failed)} run(s)). Nothing "
              f"was measured; the before column is the floor alone.",
              file=sys.stderr)
    if args.append:
        stamp = datetime.date.today().isoformat()
        head = ("the floor alone" if args.floor_only
                else f"floor then calc_reader, DEVELOPMENT engine "
                     f"{args.model} (a checkpoint, never a result)")
        block = (f"\n### WP5 calc reader -- {head}, {stamp}\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"A value matches when its printed LABEL matches at "
                   f"partial ratio {NAME_RATIO:g} and its VALUE within "
                   f"{VALUE_TOL:.0%} or the last printed digit, compared in "
                   f"SI wherever both units convert; a program name matches "
                   f"at {PROGRAM_RATIO:g}. "
                   f"{', '.join('`' + m + '`' for m in MODEL_ONLY)} have no "
                   f"before column: a pattern over a page cannot answer "
                   f"them. THERE IS NO BLIND SET -- all "
                   f"{len(openset)} report(s) ({', '.join(openset)}) were "
                   f"read while the reader's prompt was written, so every "
                   f"number here is in sample.\n\n```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    if failed and len(failed) == len(rows) and not args.floor_only:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
