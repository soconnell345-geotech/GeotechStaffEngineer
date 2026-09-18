"""WP2b measurement: the log grid, then the log reader, on the truthed logs.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp2b_logs
    ... --only R36,R37        # a subset
    ... --grid-only           # no model at all: the BEFORE column by itself
    ... --detail              # every miss on the OPEN logs
    ... --append              # write the scorecard into MEASUREMENTS.md

WHAT IT MEASURES. For each hand-truthed log it runs ``log_grid`` once, scores
what the grid alone recovered (BEFORE), hands that same grid to
``report_ingest.log_reader.read_log``, and scores the record the reader built
(AFTER). One grid, two scores, so the gap between the columns is the model
and nothing else.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse
against OpenAI models through Prompter; a score measured on Claude measures a
model that will never do the work. This script drives the DEVELOPMENT engine
so prompts can be iterated here at a keystroke, and its figures go into the
ledger marked as a checkpoint. The real run is
``report_ingest.cluster_scoring.score_on_cluster(..., stages=("logs",),
truth_dir=...)``, which does exactly the same scoring through
``report_ingest.log_scoring``.

``--grid-only`` needs no engine, no key and no network. It is the honest
baseline and it is worth re-running whenever planlens changes.

PRIVACY. Reports are IDs and logs are ``<ID>_p<page>``. Nothing here prints a
project name, a firm or a file name, and MEASUREMENTS.md, which IS committed,
gets only IDs, counts and rates.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.log_scoring import (
    LAYER_TOL_M, METRICS, SAMPLE_TOL_M, WATER_TOL_M, LogScore, score_grid,
    score_one_log,
)

TRUTH_DIR = C.RAW_DIR / "truth" / "logs"
OPEN_FILE = TRUTH_DIR / "OPEN.txt"
LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"

#: The development model. Never quoted as the system's accuracy.
DEV_MODEL = "claude-opus-5"
#: The open set, when no OPEN.txt is there.
DEFAULT_OPEN = ("R36", "R37", "R06", "R07", "R15", "R28")


def open_reports() -> Tuple[str, ...]:
    if OPEN_FILE.is_file():
        names = tuple(line.strip() for line
                      in OPEN_FILE.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return DEFAULT_OPEN


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs ({TRUTH_DIR} is gitignored and exists only "
            f"where the owner put it)")
    out = []
    for path in sorted(TRUTH_DIR.glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if only and truth["id"].split("_")[0] not in only:
            continue
        out.append(truth)
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_one(truth: Dict[str, Any], engine: Any, budget: int, di: str
            ) -> Tuple[LogScore, Optional[LogScore]]:
    """``(before, after)``; ``after`` is None when no engine was given."""
    from planlens.document.loggrid import log_grid

    report = truth["id"].split("_")[0]
    doc = None
    try:
        doc = C.open_report(report, di=di, warn=False)
        if engine is None:
            return score_grid(truth, log_grid(doc, list(truth["pages"]))), None
        return score_one_log(truth, doc, engine, budget=budget,
                             report_id=report)
    finally:
        if doc is not None:
            doc.close()


def _rate(score) -> str:
    if score is None or not score.total:
        return "     -"
    return f"{score.rate:5.0%} {score.found}/{score.total}"


def _sum(scores: Sequence[LogScore], metric: str):
    from report_ingest.log_scoring import Score

    out = Score()
    for score in scores:
        got = score.scores.get(metric)
        if got is not None:
            out += got
    return out


def report(rows: Sequence[Tuple[LogScore, Optional[LogScore]]],
           openset: Sequence[str], detail: bool = False) -> str:
    lines: List[str] = []
    groups: Dict[str, List[Tuple[LogScore, Optional[LogScore]]]] = {
        "open": [], "blind": []}
    for before, after in rows:
        groups["open" if before.report in openset else "blind"].append(
            (before, after))

    header = (f"{'metric':<16}{'before':>14}{'after':>14}")
    for name in ("open", "blind", "all"):
        group = rows if name == "all" else groups[name]
        group = [r for r in group if not r[0].error]
        if not group:
            continue
        befores = [b for b, _a in group]
        afters = [a for _b, a in group
                  if a is not None and not a.error]
        lines += [f"{name} -- {len(group)} log(s)", header,
                  "-" * len(header)]
        for metric in METRICS:
            before = _sum(befores, metric)
            after = _sum(afters, metric) if afters else None
            if not before.total and not (after and after.total):
                continue
            lines.append(f"{metric:<16}{_rate(before):>14}"
                         f"{_rate(after):>14}")
        total_before = sum((b.total.found for b in befores))
        count_before = sum((b.total.total for b in befores))
        total_after = sum((a.total.found for a in afters))
        count_after = sum((a.total.total for a in afters))
        lines.append(
            f"{'OVERALL':<16}"
            f"{(total_before / count_before if count_before else 0):5.0%} "
            f"{total_before}/{count_before:<6}"
            + (f"{(total_after / count_after if count_after else 0):5.0%} "
               f"{total_after}/{count_after}" if afters else ""))
        lines.append("")

    per_log = (f"{'log':<14}{'set':<7}{'before':>13}{'after':>13}"
               f"{'calls':>7}{'unres':>7}{'look':>6}{'s':>7}")
    lines += [per_log, "-" * len(per_log)]
    for before, after in rows:
        name = "open" if before.report in openset else "blind"
        if before.error:
            lines.append(f"{before.log_id:<14}{name:<7}ERROR "
                         f"{before.error[:60]}")
            continue
        if after is not None and after.error:
            lines.append(f"{before.log_id:<14}{name:<7}"
                         f"{_rate(before.total):>13}  READER FAILED")
            continue
        lines.append(
            f"{before.log_id:<14}{name:<7}{_rate(before.total):>13}"
            + (f"{_rate(after.total):>13}{after.model_calls:>7}"
               f"{after.unresolved:>7}{after.changes:>6}"
               f"{after.cost.get('seconds', 0.0):>7.0f}"
               if after is not None
               else f"{'-':>13}{'-':>7}{'-':>7}{'-':>6}{'-':>7}"))
    lines.append("")

    # A run in which every reader call failed used to print a tidy "0% 0/0"
    # after column and a row of dashes. A failure has to be LOUD, or a
    # scorecard reports a reader that never ran as a reader that found
    # nothing.
    failed = [(b.log_id, a.error) for b, a in rows
              if a is not None and a.error]
    if failed:
        lines.append(f"THE READER FAILED ON {len(failed)} OF {len(rows)} "
                     f"LOG(S). Those logs are in NO after number above.")
        for log_id, error in failed:
            lines.append(f"  {log_id}: {error[:160]}")
        lines.append("")

    afters = [a for _b, a in rows if a is not None and not a.error]
    if afters:
        calls = sum(a.model_calls for a in afters)
        tokens_in = sum(a.cost.get("input_tokens", 0) for a in afters)
        tokens_out = sum(a.cost.get("output_tokens", 0) for a in afters)
        dollars = sum(a.cost.get("dollars", 0.0) for a in afters)
        seconds = sum(a.cost.get("seconds", 0.0) for a in afters)
        n = len(afters)
        lines += [
            f"cost: {calls} model call(s), {tokens_in:,} in, "
            f"{tokens_out:,} out, {seconds:.0f} s, ${dollars:.2f}",
            f"per log: {calls / n:.1f} call(s), {tokens_in / n:,.0f} in, "
            f"{tokens_out / n:,.0f} out, {seconds / n:.0f} s, "
            f"${dollars / n:.2f}",
            ""]

    warned = [b for b, _a in rows if b.warnings]
    if warned:
        lines.append("warnings the grid raised:")
        for before in warned:
            for warning in before.warnings:
                lines.append(f"  {before.log_id}: {warning}")
        lines.append("")

    if detail:
        lines.append("misses on the OPEN logs (the blind ones are not "
                     "listed):")
        for before, after in rows:
            if before.report not in openset:
                continue
            for stage, score in (("grid", before), ("record", after)):
                if score is None:
                    continue
                for metric in METRICS:
                    got = score.scores.get(metric)
                    for miss in (got.misses if got else ()):
                        lines.append(
                            f"  {score.log_id} {stage} {metric}: {miss}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--grid-only", action="store_true",
                    help="score the grid alone; no engine, no key, no network")
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--budget", type=int, default=6,
                    help="model calls per log")
    ap.add_argument("--detail", action="store_true",
                    help="list every miss on the OPEN logs")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    openset = open_reports()
    truths = load_truth(only)
    if not truths:
        print("no truth files matched")
        return 1

    engine = None
    if not args.grid_only:
        from report_ingest.engine import ClaudeEngine
        engine = ClaudeEngine(args.model)

    rows: List[Tuple[LogScore, Optional[LogScore]]] = []
    for n, truth in enumerate(truths, 1):
        print(f"[{n}/{len(truths)}] {truth['id']} ...", flush=True)
        rows.append(run_one(truth, engine, args.budget, args.di))

    text = report(rows, openset, detail=args.detail)
    print(text)
    failed = [b.log_id for b, a in rows if a is not None and a.error]
    if failed and len(failed) == len(rows):
        print(f"\nEVERY reader call failed ({len(failed)} log(s)). Nothing "
              f"was measured; the before column is the grid alone.",
              file=sys.stderr)
    if args.append:
        stamp = datetime.date.today().isoformat()
        head = ("log_grid only" if args.grid_only
                else f"log_grid then log_reader, DEVELOPMENT engine "
                     f"{args.model} (a checkpoint, never a result)")
        block = (f"\n### WP2b log scorecard, {stamp} -- {head}\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"Tolerances: samples, index values and water "
                   f"{SAMPLE_TOL_M} m ({WATER_TOL_M} m for water), layer tops "
                   f"{LAYER_TOL_M} m; depths compared in metres whatever the "
                   f"log prints; N values exact. Open set = "
                   f"{', '.join(openset)}.\n\n```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    if failed and len(failed) == len(rows) and not args.grid_only:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
