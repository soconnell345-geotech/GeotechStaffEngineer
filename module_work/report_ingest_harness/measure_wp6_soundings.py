"""WP6 measurement: the floor, then the reader, on test pits and soundings.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp6_soundings
    ... --only R20,R28          # a subset, by report
    ... --kind cpt              # a subset: test_pit, cpt or dcp
    ... --floor-only            # no model at all: the BEFORE column alone
    ... --detail                # every miss
    ... --append                # write the scorecard into MEASUREMENTS.md

WHAT IT MEASURES. For each hand-truthed sheet it scores the FLOOR -- the log
grid on a test pit, and on a sounding either the sheet's own TABULATED table
or, where the sheet plots its traces, the axis ranges and nothing else -- and
then runs the reader over the same open document and scores the record it
built. The two columns are the same pages read two ways, so the gap between
them is the model and nothing else.

THE ONE SPLIT THAT MATTERS, and the scorecard prints it. A TABULATED sounding
already has its series in the floor, so its before column is a real baseline
and the reader's job is the header. A PLOTTED sounding's floor holds NO points
at all, so its before column is zero by construction and its after column IS
the digitising. Averaging the two hides the only number anyone wants.

THERE IS NO BLIND SET. Every hand-truthed sheet was read while these readers
were written, so every number this prints is IN SAMPLE and none of it is
evidence about an unseen report. A blind set is owed and the scorecard says
so.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse against
OpenAI models through Prompter; a score measured on Claude measures a model
that will never do the work. This script drives the DEVELOPMENT engine so
prompts can be iterated here at a keystroke, and its figures go into the
ledger marked as a checkpoint. The real run is
``report_ingest.cluster_scoring.score_on_cluster(..., stages=("soundings",),
truth_dir=...)``, which scores through the same
``report_ingest.sounding_scoring``.

``--floor-only`` needs no engine, no key and no network. It is the honest
baseline and it is worth re-running whenever planlens changes.

PRIVACY. Reports are IDs and sheets are ``<kind>__<ID>_p<page>``. Nothing here
prints a project name, a firm or a file name, and MEASUREMENTS.md, which IS
committed, gets only IDs, kinds, counts and rates.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.sounding_scoring import (
    CHANNEL_TOL, DEPTH_TOL, METRICS, U2_TOL, Score, SoundingScore,
    kind_of, pages_of, report_of, score_one_sounding,
)

LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"
TRUTH_DIR = C.RAW_DIR / "truth" / "soundings"

#: The development model. Never quoted as the system's accuracy.
DEV_MODEL = "claude-opus-5"


def truth_files() -> List[Path]:
    return sorted(TRUTH_DIR.glob("*.json"))


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every hand-truthed sheet, or the ones whose report is in ``only``."""
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed soundings ({TRUTH_DIR} is gitignored and "
            f"exists only where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in truth_files():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if only and report_of(blob) not in only:
            continue
        out.append(blob)
    return out


def open_reports(default: Sequence[str] = ()) -> Tuple[str, ...]:
    """The reports whose sounding pages the builder was allowed to see."""
    path = TRUTH_DIR / "OPEN.txt"
    if path.is_file():
        names = tuple(line.strip() for line
                      in path.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return tuple(default)


def run_one(truth: Dict[str, Any], engine: Any, budget: int, di: str
            ) -> Tuple[SoundingScore, Optional[SoundingScore]]:
    """``(before, after)``; ``after`` is None when no engine was given."""
    report = report_of(truth)
    doc = None
    try:
        doc = C.open_report(report, di=di, warn=False)
        before, after = score_one_sounding(truth, doc, engine, budget=budget,
                                           report_id=report)
        if engine is None:
            return before, None
        return before, after
    finally:
        if doc is not None:
            doc.close()


def _rate(score: Optional[Score]) -> str:
    if score is None or not score.total:
        return "     -"
    return f"{score.rate:5.0%} {score.found}/{score.total}"


def _sum(scores: Sequence[SoundingScore], metric: str) -> Score:
    out = Score()
    for score in scores:
        got = score.scores.get(metric)
        if got is not None:
            out += got
    return out


def _totals(scores: Sequence[SoundingScore]) -> Score:
    out = Score()
    for score in scores:
        for metric in METRICS:
            got = score.scores.get(metric)
            if got is not None:
                out += got
    return out


def _block(name: str,
           group: Sequence[Tuple[SoundingScore, Optional[SoundingScore]]],
           lines: List[str]) -> None:
    group = [row for row in group if not row[0].error]
    if not group:
        return
    befores = [b for b, _a in group]
    afters = [a for _b, a in group if a is not None and not a.error]
    header = f"{'metric':<12}{'before':>14}{'after':>14}"
    lines += [f"{name} -- {len(group)} sheet(s)", header, "-" * len(header)]
    for metric in METRICS:
        before = _sum(befores, metric)
        after = _sum(afters, metric) if afters else None
        if not before.total and not (after and after.total):
            continue
        lines.append(f"{metric:<12}{_rate(before):>14}"
                     + (f"{_rate(after):>14}" if after else ""))
    lines.append(f"{'OVERALL':<12}{_rate(_totals(befores)):>14}"
                 + (f"{_rate(_totals(afters)):>14}" if afters else ""))
    lines.append("")


def report(rows: Sequence[Tuple[SoundingScore, Optional[SoundingScore]]],
           openset: Sequence[str], detail: bool = False) -> str:
    lines: List[str] = []
    groups: Dict[str, List[Tuple[SoundingScore,
                                 Optional[SoundingScore]]]] = {
        "open": [], "blind": []}
    for before, after in rows:
        groups["open" if before.report in openset
               else "blind"].append((before, after))
    for name in ("open", "blind"):
        _block(name, groups[name], lines)
    _block("all", rows, lines)

    # The split the whole scorecard turns on.
    for shape, wanted in (("tabulated", True), ("plotted", False)):
        group = [r for r in rows if bool(r[0].tabulated) is wanted]
        if group:
            _block(shape, group, lines)

    kinds = sorted({before.kind for before, _a in rows})
    if len(kinds) > 1:
        header = (f"{'kind':<12}{'sheets':>8}{'pages':>7}{'before':>14}"
                  f"{'after':>14}")
        lines += [header, "-" * len(header)]
        for kind in kinds:
            group = [r for r in rows if r[0].kind == kind]
            befores = [b for b, _a in group if not b.error]
            afters = [a for _b, a in group if a is not None and not a.error]
            pages = sum(b.n_pages for b in befores)
            lines.append(
                f"{kind:<12}{len(group):>8}{pages:>7}"
                f"{_rate(_totals(befores)):>14}"
                + (f"{_rate(_totals(afters)):>14}" if afters else ""))
        lines.append("")

    header = (f"{'sheet':<34}{'set':<7}{'kind':<9}{'shape':<11}{'pp':>4}"
              f"{'floor':>7}{'before':>12}{'after':>12}{'calls':>6}"
              f"{'zoom':>5}{'unres':>6}{'s':>6}")
    lines += [header, "-" * len(header)]
    for before, after in rows:
        name = "open" if before.report in openset else "blind"
        shape = "tabulated" if before.tabulated else "plotted"
        if before.error:
            lines.append(f"{before.sounding_id:<34}{name:<7}ERROR "
                         f"{before.error[:50]}")
            continue
        if after is not None and after.error:
            lines.append(
                f"{before.sounding_id:<34}{name:<7}{before.kind:<9}"
                f"{shape:<11}{before.n_pages:>4}{before.floor_points:>7}"
                f"{_rate(_totals([before])):>12}"
                f"  READER FAILED: {str(after.error)[:50]}")
            continue
        lines.append(
            f"{before.sounding_id:<34}{name:<7}{before.kind:<9}"
            f"{shape:<11}{before.n_pages:>4}{before.floor_points:>7}"
            f"{_rate(_totals([before])):>12}"
            + (f"{_rate(after.total):>12}{after.model_calls:>6}"
               f"{after.tool_calls:>5}{after.unresolved:>6}"
               f"{after.cost.get('seconds', 0.0):>6.0f}"
               if after is not None
               else f"{'-':>12}{'-':>6}{'-':>5}{'-':>6}{'-':>6}"))
    lines.append("")

    failed = [(b.sounding_id, a.error) for b, a in rows
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
        merged = (sum(a.disagreements for a in afters),
                  sum(a.kept for a in afters), sum(a.added for a in afters),
                  sum(a.reconciled for a in afters))
        n = len(afters)
        lines += [
            f"merge: {merged[0]} disagreement(s), {merged[1]} kept from the "
            f"floor, {merged[2]} added by the reader, {merged[3]} reconciled",
            f"cost: {calls} model call(s), {zooms} zoom(s), {tokens_in:,} in, "
            f"{tokens_out:,} out, {seconds:.0f} s, ${dollars:.2f}",
            f"per sheet: {calls / n:.1f} call(s), {zooms / n:.1f} zoom(s), "
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
                            f"  {score.sounding_id} {stage} {metric}: {miss}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--kind", default="",
                    help="comma-separated kinds: test_pit, cpt, dcp")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--floor-only", action="store_true",
                    help="score the floor alone; no engine, no key, no "
                         "network")
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--budget", type=int, default=2,
                    help="model calls per sheet, the answer included")
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
        truths = [t for t in truths if kind_of(t) in kinds]
    if not truths:
        print("no truth sheets matched")
        return 1

    engine = None
    if not args.floor_only:
        from report_ingest.engine import ClaudeEngine
        engine = ClaudeEngine(args.model)

    rows: List[Tuple[SoundingScore, Optional[SoundingScore]]] = []
    for n, truth in enumerate(truths, 1):
        print(f"[{n}/{len(truths)}] {truth['id']} "
              f"(pages {pages_of(truth)}) ...", flush=True)
        rows.append(run_one(truth, engine, args.budget, args.di))

    text = report(rows, openset, detail=args.detail)
    print(text)
    failed = [b.sounding_id for b, a in rows if a is not None and a.error]
    if failed and len(failed) == len(rows):
        print(f"\nEVERY reader call failed ({len(failed)} sheet(s)). Nothing "
              f"was measured; the before column is the floor alone.",
              file=sys.stderr)
    if args.append:
        stamp = datetime.date.today().isoformat()
        head = ("the floor alone" if args.floor_only
                else f"floor then the readers, DEVELOPMENT engine "
                     f"{args.model} (a checkpoint, never a result)")
        block = (f"\n### WP6 soundings -- {head}, {stamp}\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"A truth point counts as found when the reader has a "
                   f"reading within {DEPTH_TOL:g} m of that depth and its "
                   f"value is within {CHANNEL_TOL:.0%} or one axis tick "
                   f"(tip resistance, sleeve friction, the printed index), "
                   f"{U2_TOL:.0%} (pore pressure), or EXACTLY (a blow "
                   f"count). A test pit is scored by the log scorer's own "
                   f"metrics plus its plan dimensions. THERE IS NO BLIND "
                   f"SET -- every sheet was read while the readers were "
                   f"written, so every number here is in sample.\n\n"
                   f"```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    if failed and len(failed) == len(rows) and not args.floor_only:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
