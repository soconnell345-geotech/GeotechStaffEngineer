"""BACKLOG RI-73: page labels scored by the reader each page is sent to.

Run from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_range_labels \\
        --stage-b <an ingest stage folder> [--runs <folder> ...] [--only R18]

``--stage-b`` is an ``ingest`` stage folder a cluster run left (one
``<ID>/`` per report holding the ``labels.json`` and ``review.json`` the graph
wrote), and gives five label sets: the rules alone, vision alone, the vote,
master (the vote plus the review's changes that name a look at the page,
the RI-15 rule) and the labels as stored. ``--runs`` is a folder of label
files, one ``<ID>.json`` per report, holding ``labels`` (``page -> label``,
as the ``range_labels`` stage writes them) or that mapping bare; each folder
is one more column, named after the folder. Nothing is run and no model is
called.

WHAT IT PRINTS, per label set, over the hand-labelled pages of the reports
every set covers: strict accuracy; the wrong pages; the HARMFUL ones, which
send a page to another reader than its hand label would, or to none, or to
one where it should go to none (:func:`report_ingest.scoring.consequence`,
on the graph's own reader table); the wrong pages by consequence; and the run
structure -- where each wrong page sits in the run of pages its hand label
makes (:data:`report_ingest.scoring.STRUCTURE`) -- with the number of runs
each set draws against the hand labels' number. Then harmful pages per
report, and the commonest harmful moves.

DIVIDERS APART. Whether a tab or divider page takes its section's label is
the owner's open question (BACKLOG OWN-92 (3)), and the hand labels answer
it both ways. So strict accuracy and the harmful count are printed again
with every divider page left out -- a page the hand labels call
``divider`` and a page the label set calls ``divider`` alike -- per set and
per report.

``--harmful-runs SET`` lists that set's largest runs of harmful pages: in
one report, consecutive pages, one hand label and one given label each.

WHICH HAND LABELS. Only the spreadsheet's (every page of the in-sample
reports), through the harness's corpus. A report in the blind set
(:data:`report_ingest.scoring.OOS_BLIND`) is refused before anything about it
is read.

PRIVACY. Reports are IDs, pages are numbers and labels are labels. No reason,
description or page text is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_ingest.label_review import names_a_look
from report_ingest.scoring import (CONSEQUENCES, HARMLESS, OOS_BLIND,
                                   STRUCTURE, consequence, harm_score,
                                   hand_sections, is_harmful)

#: The label sets an ingest folder gives, in the order they are printed.
STAGE_B_SETS: Tuple[str, ...] = ("rules", "vision", "vote", "master",
                                 "stored")


def stage_b_labels(folder: Path) -> Dict[str, Dict[int, str]]:
    """One report's five label sets out of its ingest folder.

    ``master`` is the vote with the review's changes applied where the
    change names a look at the page (BACKLOG RI-15), which is what the graph
    does now; ``stored`` is the run's own final labels, every change applied.
    A voter's missing label is left out, and scores as ``other``.
    """
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, str]] = {name: {} for name in STAGE_B_SETS}
    for row in labels.get("pages") or []:
        page = int(row["page"])
        out["vote"][page] = str(row.get("label") or "other")
        for voter in row.get("voters") or []:
            name = voter.get("voter")
            if name in ("rules", "vision") and voter.get("label"):
                out[name][page] = str(voter["label"])
    out["master"] = dict(out["vote"])
    review = folder / "review.json"
    if review.is_file():
        blob = json.loads(review.read_text(encoding="utf-8"))
        for change in blob.get("changes") or []:
            if names_a_look(str(change.get("evidence") or "")):
                out["master"][int(change["page"])] = str(change["to"])
    out["stored"] = {int(k): str(v)
                     for k, v in (labels.get("labels") or {}).items()}
    return out


def run_labels(path: Path) -> Dict[int, str]:
    """``page -> label`` from one label file: a ``range_labels`` run file
    (its ``labels``) or the mapping itself."""
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, dict) and isinstance(blob.get("labels"), dict):
        blob = blob["labels"]
    return {int(k): str(v) for k, v in blob.items() if v}


#: The label OWN-92 (3) is about.
DIVIDER = "divider"


def without_dividers(hand: Mapping[int, str], given: Mapping[int, str]
                     ) -> Dict[int, str]:
    """The hand labels with every divider page left out: one the hand
    labels call a divider, and one ``given`` does."""
    return {page: label for page, label in hand.items()
            if label != DIVIDER and given.get(page) != DIVIDER}


def harmful_runs(hand: Mapping[str, Mapping[int, str]],
                 given: Mapping[str, Mapping[int, str]]
                 ) -> List[Tuple[str, int, int, str, str, str]]:
    """``(report, first, last, hand label, given label, consequence)`` for
    every run of harmful pages, largest first: consecutive pages of one
    report with one hand label and one given label, every one harmful."""
    out: List[Tuple[str, int, int, str, str, str]] = []
    for rid, truth in hand.items():
        labels = given.get(rid, {})
        run: Optional[List[Any]] = None
        for page in sorted(truth) + [None]:
            pair = None
            if page is not None:
                want, got = truth[page], labels.get(page, "other")
                if is_harmful(want, got):
                    pair = (want, got)
            if run and (pair is None or pair != run[3]
                        or page != run[2] + 1):
                out.append((rid, run[1], run[2], run[3][0], run[3][1],
                            consequence(*run[3])))
                run = None
            if pair is not None:
                if run is None:
                    run = [rid, page, page, pair]
                else:
                    run[2] = page
    out.sort(key=lambda r: (-(r[2] - r[1] + 1), r[0], r[1]))
    return out


def runs_column(arg: str) -> Tuple[str, Path]:
    """A ``--runs`` argument as (column name, folder): ``NAME=folder``, or
    a bare folder named after itself."""
    name, _, path = arg.rpartition("=")
    folder = Path(path)
    return name or folder.name, folder


def n_runs(labels: Mapping[int, str], pages: Sequence[int]) -> int:
    """How many runs of one label ``labels`` draws over ``pages``."""
    return len(hand_sections({p: labels.get(p, "other") for p in pages}))


def score_sets(hand: Mapping[str, Dict[int, str]],
               sets: Mapping[str, Mapping[str, Dict[int, str]]]
               ) -> Dict[str, Dict[str, Any]]:
    """Every label set scored over every report: the totals, and per report.

    ``hand`` is ``report -> page -> label``; ``sets`` is ``set -> report ->
    page -> label``. The run structure is worked out report by report (a run
    never crosses two reports) and added up.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, per_report in sets.items():
        total: Dict[str, Any] = {"pages": 0, "right": 0, "wrong": 0,
                                 "harmful": 0, "runs": 0, "hand_runs": 0,
                                 "pages_nodiv": 0, "right_nodiv": 0,
                                 "harmful_nodiv": 0,
                                 "by_consequence": Counter(),
                                 "by_structure": Counter(),
                                 "harmful_by_structure": Counter(),
                                 "harmful_moves": Counter(), "reports": {}}
        for rid, truth in hand.items():
            given = per_report.get(rid, {})
            one = harm_score(truth, given)
            apart = harm_score(without_dividers(truth, given), given)
            one["strict_nodiv"] = apart["strict"]
            one["harmful_nodiv"] = apart["harmful"]
            total["pages_nodiv"] += apart["pages"]
            total["right_nodiv"] += apart["right"]
            total["harmful_nodiv"] += apart["harmful"]
            for key in ("pages", "right", "wrong", "harmful"):
                total[key] += one[key]
            for key in ("by_consequence", "by_structure",
                        "harmful_by_structure", "harmful_moves"):
                total[key].update(one[key])
            pages = sorted(truth)
            total["runs"] += n_runs(given, pages)
            total["hand_runs"] += len(hand_sections(truth))
            total["reports"][rid] = one
        total["strict"] = (total["right"] / total["pages"]
                           if total["pages"] else float("nan"))
        total["strict_nodiv"] = (total["right_nodiv"] / total["pages_nodiv"]
                                 if total["pages_nodiv"] else float("nan"))
        out[name] = total
    return out


def report_lines(scored: Mapping[str, Dict[str, Any]],
                 top: int = 12) -> List[str]:
    """The printout: one column per label set."""
    names = list(scored)
    width = max([13] + [len(n) + 2 for n in names])
    head = f"{'':<30}" + "".join(f"{n:>{width}}" for n in names)

    def row(title: str, values: Sequence[Any], fmt: str = "{}") -> str:
        return f"{title:<30}" + "".join(
            f"{fmt.format(v):>{width}}" for v in values)

    first = next(iter(scored.values()))
    out = [f"{len(first['reports'])} report(s), {first['pages']} "
           f"hand-labelled pages; a page with no label counts as 'other'",
           "", head,
           row("strict accuracy", [s["strict"] for s in scored.values()],
               "{:.3f}"),
           row("wrong pages", [s["wrong"] for s in scored.values()]),
           row("HARMFUL (change a reader)",
               [s["harmful"] for s in scored.values()]),
           row("harmful, % of pages",
               [100.0 * s["harmful"] / s["pages"] if s["pages"] else 0.0
                for s in scored.values()], "{:.1f}"),
           row("strict, no dividers",
               [s["strict_nodiv"] for s in scored.values()], "{:.3f}"),
           row("HARMFUL, no dividers",
               [s["harmful_nodiv"] for s in scored.values()]),
           row("pages, no dividers",
               [s["pages_nodiv"] for s in scored.values()]),
           "", "wrong pages by consequence"]
    for what in CONSEQUENCES:
        mark = "" if what in HARMLESS else " *"
        out.append(row(f"  {what}{mark}", [s["by_consequence"].get(what, 0)
                                           for s in scored.values()]))
    out += ["  (* changes a reader)", "",
            "run structure of the wrong pages (all / harmful)"]
    for what in STRUCTURE:
        out.append(row(f"  {what}", [
            f"{s['by_structure'].get(what, 0)}/"
            f"{s['harmful_by_structure'].get(what, 0)}"
            for s in scored.values()]))
    out.append(row("runs drawn (hand: "
                   f"{first['hand_runs']})", [s["runs"]
                                              for s in scored.values()]))
    out += ["", "harmful pages per report (all / no dividers)",
            f"{'report':<30}" + "".join(f"{n:>{width}}" for n in names)]
    for rid in first["reports"]:
        out.append(row(f"  {rid} ({first['reports'][rid]['pages']} pp)",
                       [f"{s['reports'][rid]['harmful']}/"
                        f"{s['reports'][rid]['harmful_nodiv']}"
                        for s in scored.values()]))
    out += ["", "strict accuracy per report (all / no dividers)",
            f"{'report':<30}" + "".join(f"{n:>{width}}" for n in names)]
    for rid in first["reports"]:
        out.append(row(f"  {rid}", [
            f"{s['reports'][rid]['strict']:.3f}/"
            f"{s['reports'][rid]['strict_nodiv']:.3f}"
            for s in scored.values()]))
    for name, s in scored.items():
        moves = Counter(s["harmful_moves"]).most_common(top)
        out += ["", f"{name}: commonest harmful moves (hand -> given)"]
        out += [f"  {move:<36}{count:>6}" for move, count in moves]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage-b", default=None,
                    help="an ingest stage folder: <ID>/labels.json and "
                         "<ID>/review.json per report")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="folders of <ID>.json label files, one column each, "
                         "named after the folder or NAME=folder")
    ap.add_argument("--raw-dir", default=None,
                    help="the private corpus folder holding corpus/ and the "
                         "label spreadsheet; default the harness's own")
    ap.add_argument("--only", nargs="*", default=None, help="report IDs")
    ap.add_argument("--harmful-runs", nargs="*", default=[],
                    help="list these sets' largest runs of harmful pages")
    ap.add_argument("--top", type=int, default=10,
                    help="how many runs --harmful-runs lists")
    ap.add_argument("--sets", nargs="*", default=list(STAGE_B_SETS),
                    help=f"which stage-b sets to print "
                         f"(default all: {' '.join(STAGE_B_SETS)})")
    args = ap.parse_args(argv)
    if not args.stage_b and not args.runs:
        ap.error("pass --stage-b, --runs or both")

    from module_work.report_ingest_harness import corpus as harness
    from report_ingest.corpus import Corpus

    raw = Path(args.raw_dir) if args.raw_dir else harness.RAW_DIR
    sheets = Corpus(raw / "corpus", di_dir=raw / "di",
                    labels_xlsx=raw / "corpus" / harness.LABELS_XLSX.name,
                    cache_dir=raw)
    covered = set(sheets.mapped_ids())

    sets: Dict[str, Dict[str, Dict[int, str]]] = {}
    found: List[set] = []
    if args.stage_b:
        per_set: Dict[str, Dict[str, Dict[int, str]]] = {
            name: {} for name in args.sets}
        for folder in sorted(Path(args.stage_b).iterdir()):
            if not (folder / "labels.json").is_file():
                continue
            labels = stage_b_labels(folder)
            for name in args.sets:
                per_set[name][folder.name] = labels[name]
        sets.update(per_set)
        found.append(set(next(iter(per_set.values()), {})))
    for runs in args.runs:
        name, folder = runs_column(runs)
        while name in sets:
            name += "'"
        sets[name] = {path.stem: run_labels(path)
                      for path in sorted(folder.glob("R*.json"))}
        found.append(set(sets[name]))

    rids = sorted(set.intersection(*found)) if found else []
    if args.only:
        rids = [r for r in rids if r in args.only]
    hand: Dict[str, Dict[int, str]] = {}
    for rid in rids:
        if rid in OOS_BLIND:
            print(f"{rid}: in the blind set, not read", file=sys.stderr)
            continue
        if rid not in covered:
            print(f"{rid}: no spreadsheet labels, skipped", file=sys.stderr)
            continue
        hand[rid] = {pl.page0: pl.label for pl in sheets.labels_for(rid)}
    if not hand:
        print("no report is covered by every label set and the spreadsheet")
        return 1
    print("\n".join(report_lines(score_sets(hand, sets))))
    for name in args.harmful_runs:
        if name not in sets:
            print(f"\n--harmful-runs {name}: no such label set")
            continue
        print(f"\n{name}: the {args.top} largest harmful runs "
              f"(report, pages, hand -> given, what it costs)")
        for rid, first, last, want, got, what in harmful_runs(
                hand, sets[name])[:args.top]:
            span = f"p{first}" if first == last else f"p{first}-{last}"
            print(f"  {rid} {span:<12}{last - first + 1:>4} pp  "
                  f"{want} -> {got}  ({what})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
