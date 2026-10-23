"""BACKLOG RI-15: what the label review changed, against the hand labels.

Run from the repo root, on an ``ingest`` stage folder a cluster run left::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_label_review <folder>
    ... --raw-dir <the private corpus folder>     # default: the harness's own

``<folder>`` holds one ``<ID>/`` per report with the ``labels.json`` and
``review.json`` the graph wrote. Nothing is run and no model is called.

WHAT IT COUNTS. ``labels.json`` carries the vote's label for every page
(``pages[].label``, which is what the review was asked to check) and each
voter's own label; ``review.json`` carries every change the review made, with
the evidence it named. For each change, :func:`report_ingest.scoring.
verdict_for` says what it did against the hand label: ``fixed`` (wrong to
right), ``broke`` (right to wrong) or ``still_wrong`` (wrong to another
wrong). The counts are given per report, by the vote's label, by the move,
by the evidence named, and by which voter's label the review moved the page
to. Then everything again under the RI-15 rule (a change whose evidence
names no look at the page is refused, :func:`report_ingest.label_review.
names_a_look`), with each report's accuracy before the review, after it,
and after it under the rule.

WHICH HAND LABELS. Only the spreadsheet's (every page of the in-sample
reports). A report in the blind set (:data:`report_ingest.scoring.
OOS_BLIND`) is refused before anything about it is read, and so is a report
the spreadsheet does not cover.

PRIVACY. Reports are IDs, pages are numbers and labels are labels. A change's
``reason`` is never read, so nothing printed can carry a page's own words.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, \
    Sequence

from report_ingest.label_review import names_a_look
from report_ingest.scoring import OOS_BLIND, disputed_drop, verdict_for

#: The verdicts a change can get, in the order they are printed.
VERDICTS = ("fixed", "broke", "still_wrong", "disputed", "unscored")


@dataclass
class ReportRun:
    """One report's saved vote and review, and its hand labels."""

    rid: str
    vote: Dict[int, str]
    hand: Dict[int, str]
    changes: List[Dict[str, Any]]
    rules: Dict[int, str] = field(default_factory=dict)
    vision: Dict[int, str] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    asked: int = 0
    split: int = 0


def read_run(folder: Path, rid: str, hand: Mapping[int, str]) -> ReportRun:
    """One report's ``labels.json`` and ``review.json``, as counts need them."""
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    vote: Dict[int, str] = {}
    rules: Dict[int, str] = {}
    vision: Dict[int, str] = {}
    for row in labels.get("pages") or []:
        page = int(row["page"])
        vote[page] = str(row.get("label") or "other")
        for voter in row.get("voters") or []:
            if voter.get("voter") == "rules":
                rules[page] = str(voter.get("label") or "")
            elif voter.get("voter") == "vision":
                vision[page] = str(voter.get("label") or "")
    changes = [{"page": int(c["page"]), "to": str(c["to"]),
                "evidence": str(c.get("evidence") or "")}
               for c in review.get("changes") or []]
    return ReportRun(rid=rid, vote=vote, hand=dict(hand), changes=changes,
                     rules=rules, vision=vision,
                     cost=dict(review.get("cost") or {}),
                     asked=len(review.get("asked_pages") or []),
                     split=len(labels.get("split_pages") or []))


def side(run: ReportRun, page: int, to_label: str) -> str:
    """Whose label the review moved the page to."""
    if to_label == run.rules.get(page):
        return "the rules'"
    if to_label == run.vision.get(page):
        return "vision's"
    return "neither voter's"


def verdicts(run: ReportRun,
             keep: Callable[[Dict[str, Any]], bool] = lambda c: True
             ) -> List[Dict[str, Any]]:
    """Each kept change with what it did against the hand label."""
    out: List[Dict[str, Any]] = []
    for change in run.changes:
        if not keep(change):
            continue
        page = change["page"]
        before = run.vote.get(page, "other")
        out.append({
            "rid": run.rid, "page": page, "before": before,
            "after": change["to"], "evidence": change["evidence"] or "none",
            "side": side(run, page, change["to"]),
            "verdict": verdict_for(run.rid, page, before, change["to"],
                                   run.hand.get(page)),
        })
    return out


def accuracy(run: ReportRun, applied: Iterable[Dict[str, Any]]
             ) -> Dict[str, Any]:
    """Strict accuracy over the hand-labelled pages, before and after."""
    after = dict(run.vote)
    for row in applied:
        after[row["page"]] = row["after"]
    before_hits = after_hits = scored = 0
    for page, want in sorted(run.hand.items()):
        now = after.get(page, "other")
        if disputed_drop(run.rid, page, now):
            continue
        scored += 1
        before_hits += int(run.vote.get(page, "other") == want)
        after_hits += int(now == want)
    return {"scored": scored, "before": before_hits, "after": after_hits}


def by(rows: Iterable[Dict[str, Any]], key: Callable[[Dict[str, Any]], Any]
       ) -> Dict[Any, Counter]:
    """Verdict counts grouped by ``key``."""
    groups: Dict[Any, Counter] = defaultdict(Counter)
    for row in rows:
        groups[key(row)][row["verdict"]] += 1
    return dict(groups)


def _table(title: str, groups: Mapping[Any, Counter], width: int = 34
           ) -> List[str]:
    out = ["", title,
           f"{'':<{width}}{'changes':>8}{'fixed':>7}{'broke':>7}"
           f"{'still':>7}{'other':>7}"]
    for key, c in sorted(groups.items(), key=lambda kv: (-sum(kv[1].values()),
                                                         str(kv[0]))):
        other = c["disputed"] + c["unscored"]
        out.append(f"{str(key)[:width - 1]:<{width}}{sum(c.values()):>8}"
                   f"{c['fixed']:>7}{c['broke']:>7}{c['still_wrong']:>7}"
                   f"{other:>7}")
    return out


def report(runs: Sequence[ReportRun]) -> List[str]:
    """The whole printout: per report, the groupings, then under the rule."""
    out = ["Label review changes against the hand labels "
           "(still = wrong to another wrong; other = disputed or unscored)"]
    rows_all: List[Dict[str, Any]] = []
    rows_rule: List[Dict[str, Any]] = []
    out.append(f"{'report':<8}{'split':>6}{'asked':>6}{'changes':>8}"
               f"{'fixed':>6}{'broke':>6}{'still':>6}{'hand':>6}"
               f"{'before':>8}{'after':>8}{'rule':>8}"
               f"{'calls':>6}{'in tok':>9}{'per call':>9}")
    totals: Counter = Counter()
    for run in runs:
        rows = verdicts(run)
        kept = verdicts(run, lambda c: names_a_look(c["evidence"]))
        rows_all += rows
        rows_rule += kept
        base = accuracy(run, rows)
        ruled = accuracy(run, kept)
        c = Counter(r["verdict"] for r in rows)
        calls = int(run.cost.get("calls") or 0)
        tokens = int(run.cost.get("input_tokens") or 0)
        n = base["scored"]
        out.append(
            f"{run.rid:<8}{run.split:>6}{run.asked:>6}{len(rows):>8}"
            f"{c['fixed']:>6}{c['broke']:>6}{c['still_wrong']:>6}{n:>6}"
            f"{(base['before'] / n if n else float('nan')):>8.3f}"
            f"{(base['after'] / n if n else float('nan')):>8.3f}"
            f"{(ruled['after'] / n if n else float('nan')):>8.3f}"
            f"{calls:>6}{tokens:>9,}"
            f"{(tokens // calls if calls else 0):>9,}")
        totals.update({"hand": n, "before": base["before"],
                       "after": base["after"], "rule": ruled["after"]})
    n = totals["hand"]
    if n:
        out.append(f"ALL: {n} hand-labelled pages; accuracy before the review "
                   f"{totals['before'] / n:.3f}, after it "
                   f"{totals['after'] / n:.3f}, after it under the rule "
                   f"{totals['rule'] / n:.3f}")
    for name, rows in (("", rows_all), (" -- under the RI-15 rule", rows_rule)):
        c = Counter(r["verdict"] for r in rows)
        out += ["", f"ALL changes{name}: {len(rows)}; fixed {c['fixed']}, "
                    f"broke {c['broke']}, still wrong {c['still_wrong']}"]
    out += _table("by the evidence the change named", by(rows_all,
                  lambda r: r["evidence"]))
    out += _table("by whose label the review moved the page to",
                  by(rows_all, lambda r: r["side"]))
    out += _table("by the vote's label (what the review changed)",
                  by(rows_all, lambda r: r["before"]))
    out += _table("by the move", by(rows_all,
                  lambda r: f"{r['before']} -> {r['after']}"))
    out += _table("by the move, under the RI-15 rule", by(rows_rule,
                  lambda r: f"{r['before']} -> {r['after']}"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="an ingest stage folder: <ID>/labels.json "
                                   "and <ID>/review.json per report")
    ap.add_argument("--raw-dir", default=None,
                    help="the private corpus folder holding corpus/ and the "
                         "label spreadsheet; default the harness's own")
    ap.add_argument("--only", nargs="*", default=None, help="report IDs")
    args = ap.parse_args(argv)

    from report_ingest.corpus import Corpus
    from module_work.report_ingest_harness import corpus as harness

    raw = Path(args.raw_dir) if args.raw_dir else harness.RAW_DIR
    sheets = Corpus(raw / "corpus", di_dir=raw / "di",
                    labels_xlsx=raw / "corpus" / harness.LABELS_XLSX.name)
    covered = set(sheets.mapped_ids())
    runs: List[ReportRun] = []
    for folder in sorted(Path(args.folder).iterdir()):
        rid = folder.name
        if args.only and rid not in args.only:
            continue
        if not (folder / "review.json").is_file():
            continue
        if rid in OOS_BLIND:
            print(f"{rid}: in the blind set, not read", file=sys.stderr)
            continue
        if rid not in covered:
            print(f"{rid}: no spreadsheet labels, skipped", file=sys.stderr)
            continue
        hand = {pl.page0: pl.label for pl in sheets.labels_for(rid)}
        runs.append(read_run(folder, rid, hand))
    print("\n".join(report(runs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
