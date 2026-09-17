"""WP4 measurement: the narrative reader against the hand answers.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp4_narrative
    ... --only R36,R05       # a subset, by report
    ... --fields             # the per-question table as well
    ... --detail             # every miss on the OPEN reports
    ... --append             # write the scorecard into MEASUREMENTS.md

WHAT IT MEASURES. For each report with a hand answer
(``raw/truth/narrative/<ID>.json``) it opens the report, finds the narrative
work item, runs ``report_ingest.narrative_reader.read_narrative`` over it, and
scores the two schemas field by field through
``report_ingest.narrative_scoring`` -- the same module the cluster run uses, so
the two cannot drift.

THREE NUMBERS, AND THE THIRD IS THERE TO BE DISTRUSTED. Most reports answer
most of the general list and only part of the hazards list, so a reader that
says nothing agrees with the hand on a great many fields. ``recall`` is of the
questions the report DOES answer; ``precision`` is of the answers the reader
gave; ``agreement`` counts every field including the ones both sides left
null, and is printed because it is the figure a naive scorer would report.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse against
OpenAI models through Prompter; a score measured on Claude measures a model
that will never do the work. This script drives the DEVELOPMENT engine so the
prompt can be iterated here at a keystroke, and its figures go into the ledger
marked as a checkpoint. The real run is
``report_ingest.cluster_scoring.score_on_cluster(..., stages=("narrative",),
narrative_truth_dir=...)``.

PRIVACY. Reports are IDs. Nothing here prints a project name, a firm or a file
name; the per-question table carries field names, counts and rates, and
MEASUREMENTS.md, which IS committed, gets only those.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from module_work.report_ingest_harness import corpus as C
from report_ingest.narrative_scoring import (
    FUZZY_RATIO, KINDS, LIST_HIT_JACCARD, NarrativeScore, Score,
    score_one_report,
)

LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"
TRUTH_DIR = C.RAW_DIR / "truth" / "narrative"

#: The development model. Never quoted as the system's accuracy.
DEV_MODEL = "claude-opus-5"
#: The reports whose answers were written with the reader's output in view,
#: when the truth folder carries no OPEN.txt.
DEFAULT_OPEN = ("R36", "R05")


def truth_files() -> List[Path]:
    return sorted(TRUTH_DIR.glob("*.json")) if TRUTH_DIR.is_dir() else []


def open_reports(default: Sequence[str] = ()) -> Sequence[str]:
    path = TRUTH_DIR / "OPEN.txt"
    if path.is_file():
        names = [line.strip() for line
                 in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if names:
            return names
    return list(default)


def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Every hand answer, or the ones whose report is in ``only``.

    The report's ID is the file's stem, and it is written into the blob so
    everything downstream can name it without carrying the path around.
    """
    if not truth_files():
        raise FileNotFoundError(
            f"no hand answers ({TRUTH_DIR} is gitignored and exists only "
            f"where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in truth_files():
        rid = path.stem
        if only and rid not in only:
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob.setdefault("id", rid)
        out.append(blob)
    return out


def run_one(truth: Dict[str, Any], engine: Any, budget: int,
            di: str) -> NarrativeScore:
    rid = str(truth.get("id") or "")
    doc = None
    try:
        doc = C.open_report(rid, di=di, warn=False)
        return score_one_report(truth, doc, engine, budget=budget, report=rid)
    except Exception as exc:                     # score the rest of them
        return NarrativeScore(report=rid, error=f"{type(exc).__name__}: {exc}")
    finally:
        if doc is not None:
            doc.close()


def _rate(score: Optional[Score]) -> str:
    if score is None or not score.total:
        return "     -"
    return f"{score.found / score.total:5.0%} {score.found}/{score.total}"


def _sum(scores: Sequence[NarrativeScore], name: str) -> Score:
    out = Score()
    for score in scores:
        out += getattr(score, name)
    return out


def _sum_kind(scores: Sequence[NarrativeScore], kind: str) -> Score:
    out = Score()
    for score in scores:
        got = score.by_kind.get(kind)
        if got is not None:
            out += got
    return out


def _block(name: str, group: Sequence[NarrativeScore]) -> List[str]:
    if not group:
        return []
    out = ["", f"{name} -- {len(group)} report(s)",
           f"{'metric':<26}{'rate':>16}"]
    for metric in ("recall", "precision", "agreement"):
        out.append(f"{metric:<26}{_rate(_sum(group, metric)):>16}")
    out.append("")
    for kind in KINDS:
        got = _sum_kind(group, kind)
        if got.total:
            out.append(f"{'  recall, ' + kind:<26}{_rate(got):>16}")
    for label, attr in (("list items, precision", "list_items"),
                        ("list items, recall", "list_item_recall"),
                        ("summaries written", "summaries_present"),
                        ("summaries in limit", "summaries_within_limit")):
        got = _sum(group, attr)
        if got.total:
            out.append(f"{'  ' + label:<26}{_rate(got):>16}")
    return out


def _fields_table(scores: Sequence[NarrativeScore]) -> List[str]:
    cells: Dict[str, Dict[str, int]] = {}
    for score in scores:
        for row in score.fields:
            cell = cells.setdefault(row.field, {"right": 0, "asked": 0,
                                                "missed": 0, "invented": 0,
                                                "wrong": 0})
            if row.verdict in ("missed", "wrong", "right", "too_long"):
                cell["asked"] += 1
            if row.verdict == "right":
                cell["right"] += 1
            elif row.verdict in ("missed", "invented", "wrong"):
                cell[row.verdict] += 1
    asked = {name: cell for name, cell in cells.items() if cell["asked"]}
    if not asked:
        return []
    out = ["", "Per question",
           f"{'field':<30}{'asked':>7}{'right':>7}{'missed':>8}{'wrong':>7}"
           f"{'invented':>10}"]
    for name in sorted(asked, key=lambda k: (-asked[k]["asked"], k)):
        cell = asked[name]
        out.append(f"{name:<30}{cell['asked']:>7}{cell['right']:>7}"
                   f"{cell['missed']:>8}{cell['wrong']:>7}"
                   f"{cell['invented']:>10}")
    return out


def report(scores: Sequence[NarrativeScore], openset: Sequence[str], *,
           fields: bool = False, detail: bool = False) -> str:
    good = [s for s in scores if s.error is None]
    out: List[str] = [
        f"{'report':<8}{'set':<7}{'recall':>14}{'precision':>14}"
        f"{'agree':>14}{'calls':>7}{'unres':>7}{'$':>8}"]
    for score in scores:
        which = "open" if score.report in openset else "blind"
        if score.error:
            out.append(f"{score.report:<8}{which:<7}ERROR "
                       f"{score.error[:60]}")
            continue
        out.append(
            f"{score.report:<8}{which:<7}{_rate(score.recall):>14}"
            f"{_rate(score.precision):>14}{_rate(score.agreement):>14}"
            f"{score.model_calls:>7}{score.unresolved:>7}"
            f"{score.cost.get('dollars', 0.0):>8.3f}")

    out += _block("OPEN", [s for s in good if s.report in openset])
    out += _block("BLIND", [s for s in good if s.report not in openset])
    out += _block("ALL", good)
    if fields:
        out += _fields_table(good)
    if detail:
        for score in good:
            if score.report not in openset:
                continue
            misses = [row for row in score.fields
                      if row.verdict in ("missed", "wrong", "invented")]
            if not misses:
                continue
            out += ["", f"{score.report} -- {len(misses)} miss(es)"]
            out += [f"  {row.field:<28}{row.verdict:<10}hand={row.hand!r} "
                    f"got={row.got!r}" for row in misses[:40]]
    spent = sum(s.cost.get("dollars", 0.0) for s in good)
    calls = sum(s.model_calls for s in good)
    out += ["", f"{calls} model call(s) over {len(good)} report(s), "
                f"${spent:.2f} at list price."]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--budget", type=int, default=8,
                    help="model calls per report, the answer included")
    ap.add_argument("--fields", action="store_true",
                    help="print the per-question table")
    ap.add_argument("--detail", action="store_true",
                    help="list every miss on the OPEN reports")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    openset = open_reports(DEFAULT_OPEN)
    truths = load_truth(only)
    if not truths:
        print("no hand answers matched")
        return 1

    from report_ingest.engine import ClaudeEngine
    engine = ClaudeEngine(args.model)

    scores: List[NarrativeScore] = []
    for n, truth in enumerate(truths, 1):
        print(f"[{n}/{len(truths)}] {truth['id']} ...", flush=True)
        scores.append(run_one(truth, engine, args.budget, args.di))

    text = report(scores, openset, fields=args.fields, detail=args.detail)
    print(text)
    failed = [s.report for s in scores if s.error]
    if failed and len(failed) == len(scores):
        print(f"\nEVERY report failed ({len(failed)}). Nothing was measured.",
              file=sys.stderr)
    if args.append:
        stamp = datetime.date.today().isoformat()
        block = (f"\n### WP4 narrative scorecard, {stamp} -- DEVELOPMENT "
                 f"engine {args.model} (a checkpoint, never a result)\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"recall is of the questions the report DOES answer; "
                   f"precision is of the answers the reader gave; agreement "
                   f"counts every field including the ones both sides left "
                   f"null, and flatters. Enumerations and counts exact; a "
                   f"string at a partial ratio of {FUZZY_RATIO} or on a "
                   f"shared proper noun; a list at a Jaccard overlap of "
                   f"{LIST_HIT_JACCARD}; the four summaries on presence and "
                   f"word limit only. Open set = "
                   f"{', '.join(openset)}.\n\n```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    if failed and len(failed) == len(scores):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
