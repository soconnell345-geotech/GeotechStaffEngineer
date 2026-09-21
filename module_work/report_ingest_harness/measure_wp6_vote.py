"""WP6 measurement: the voters set against each other, with NO model at all.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp6_vote
    ... --set checkpoint            # which reports
    ... --vision-dir <folder>       # saved vision runs; default is WP5's own
    ... --append --note "what changed this round"

WHAT IT MEASURES. Nothing new is run: the rules' labels and the vision
labels come off the WP5 run files the development engine already produced,
the review's labels off the WP1b run files beside them, and the hand labels
out of the same spreadsheet and out-of-sample file every other scorecard
uses. So this costs nothing, can be re-run after every change to the
arithmetic, and is free to be wrong the first few times.

THE QUESTION. The corpus run of 2026-09-20 said the two cheap voters are
COMPLEMENTARY rather than one being better: the rules own the structural
labels (``appended_report``, ``other``, ``calculation``, ``lab_test``) and
vision owns the visual ones (``plan``, ``profile``, ``photos``, ``cover``,
``toc``, ``figure`` recall). So: how often do they agree, how right are they
when they do, which of them to believe where they split, and how good would
a targeted review of the splits have to be?

THE ARITHMETIC IS THE SHIPPED ONE. :mod:`report_ingest.vote` holds it, and
:func:`report_ingest.cluster_scoring._run_vote` is the cluster twin of this
script. Two copies of this would drift and the second copy's numbers would
be the ones nobody checked.

PRIVACY. Reports are IDs, pages are numbers and labels are labels. Nothing
here reads a reason or a heading, so nothing it prints can carry one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus
from module_work.report_ingest_harness import measure_wp1b as wp1b
from module_work.report_ingest_harness import measure_wp5_vision as wp5
from report_ingest.scoring import (   # the cluster scores with these too
    CHECKPOINT, GATE, KEY_CONTENT, Scores, columns_label_table,
    disputed_drop,
)
from report_ingest.vote import (
    POLICIES, STRUCTURAL_RULES_WIN, PageVote, agreement, build_votes,
    policy_labels, required_review_accuracy, trust_table, vision_confidences,
)

LEDGER = corpus.RAW_DIR.parent / "MEASUREMENTS.md"
SECTION = "## WP6 -- the vote"
#: Where WP5 left its runs. Gitignored, like all of ``raw/``.
VISION_RUNS_DIR = wp5.RUNS_DIR
#: Where WP1b left its runs, for the review column.
REVIEW_RUNS_DIR = wp1b.RUNS_DIR
#: Where the per-report disagreement lists go.
OUT_DIR = corpus.CHECKS_DIR / "wp6_vote"

#: The display name each policy gets in the per-label table, where a column
#: has ten characters to fit into beside its ``P ``/``R `` prefix.
POLICY_COLUMN = {"trust": "trust", "structural": "struct",
                 "confidence": "conf"}


# -- the inputs --------------------------------------------------------------

def load_vision_runs(dirs: Sequence[Path]) -> Dict[str, dict]:
    """``id -> saved vision run``. The first folder that holds one wins."""
    runs: Dict[str, dict] = {}
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            rid = path.stem
            if rid in runs or rid.startswith("detail_"):
                continue
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if (blob.get("vision") or {}).get("labels"):
                runs[rid] = blob
    return runs


def votes_for(rid: str, blob: dict, oos: Dict[str, Dict[int, dict]]
              ) -> Tuple[List[PageVote], int]:
    """One report's pages as votes, and how many were dropped as disputed."""
    rules = {int(k): v for k, v in (blob.get("rules_labels") or {}).items()}
    confidence = {int(k): float(v) for k, v
                  in (blob.get("rules_confidence") or {}).items()}
    seen = blob.get("vision") or {}
    vision = {int(k): v for k, v in (seen.get("labels") or {}).items()}
    review = wp5.review_labels_for(rid)
    hand, alternates, _source = wp1b.truth_for(rid, oos)
    rows = build_votes(rid, rules, vision, rules_confidence=confidence,
                       vision_confidence=vision_confidences(seen),
                       review=review, hand=hand, alternates=alternates)
    kept, dropped = [], 0
    for vote in rows:
        if vote.review and disputed_drop(rid, vote.page, vote.review):
            dropped += 1
            continue
        kept.append(vote)
    return kept, dropped


# -- scoring -----------------------------------------------------------------

def score_set(rows: Sequence[PageVote],
              table: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One set: the agreement, every voter's score and the three policies."""
    scorers: Dict[str, Scores] = {"rules": Scores(), "vision": Scores(),
                                  "review": Scores()}
    for policy in POLICIES:
        scorers[policy] = Scores()
    for vote in rows:
        if vote.hand is None:
            continue
        alts = vote.alternates
        scorers["rules"].add(vote.hand, vote.rules, alts)
        scorers["vision"].add(vote.hand, vote.vision, alts)
        if vote.review is not None:
            scorers["review"].add(vote.hand, vote.review, alts)
        for policy, label in policy_labels(vote, table).items():
            scorers[policy].add(vote.hand, label, alts)
    agree = agreement(rows)
    return {
        "agreement": agree,
        "scorers": scorers,
        "required_review_accuracy": required_review_accuracy(
            agree["agreed"]["pages"], agree["agreed"]["correct"],
            agree["disagreed"]["pages"], GATE),
    }


def columns(scored: Dict[str, Any]) -> List[Tuple[str, Scores]]:
    """The columns this set actually has, in the order they are read."""
    scorers = scored["scorers"]
    out = [("rules", scorers["rules"])]
    if scorers["review"].n:
        out.append(("+review", scorers["review"]))
    out.append(("vision", scorers["vision"]))
    out += [(POLICY_COLUMN[name], scorers[name]) for name in POLICIES]
    return [(name, s) for name, s in out if s.n]


def _pct(value: Optional[float]) -> str:
    return "     -" if value is None else f"{value:6.3f}"


def report(by_set: Dict[str, Dict[str, Any]],
           table: Dict[str, Dict[str, Any]],
           learned_on: Sequence[str]) -> str:
    """The scorecard. IDs, labels, counts and rates; never a reason."""
    out: List[str] = [
        "agreement, and what it is worth "
        "(the rate is over every page; the accuracies over hand-labelled "
        "pages alone)",
        f"{'set':<14}{'pages':>8}{'agree':>8}{'rate':>8}{'scored':>8}"
        f"{'agreed':>8}{'right':>9}{'split':>7}{'rules':>8}{'vision':>8}"
        f"{'either':>8}",
    ]
    for name, scored in by_set.items():
        agree = scored["agreement"]
        agreed, split = agree["agreed"], agree["disagreed"]
        out.append(
            f"{name:<14}{agree['pages']:>8}{agree['agree']:>8}"
            f"{_pct(agree['agreement']):>8}{agree['scored']:>8}"
            f"{agreed['pages']:>8}{_pct(agreed['accuracy']):>9}"
            f"{split['pages']:>7}{_pct(split['rules_accuracy']):>8}"
            f"{_pct(split['vision_accuracy']):>8}"
            f"{_pct(split['ceiling']):>8}")

    out += ["", f"per-label trust, learned on {len(learned_on)} in-sample "
                f"report(s) and NEVER on the blind set",
            f"{'rules label':<18}{'splits':>8}{'rules right':>13}"
            f"{'vision right':>14}{'believe':>10}"]
    for label in sorted(table, key=lambda k: (-table[k]["pages"], k)):
        cell = table[label]
        out.append(f"{label:<18}{cell['pages']:>8}{cell['rules']:>13}"
                   f"{cell['vision']:>14}{cell['winner']:>10}")
    if not table:
        out.append("(nothing learned: no in-sample report in this run, so "
                   "`trust` sends every disagreement to vision)")

    for name, scored in by_set.items():
        cols = columns(scored)
        if not cols:
            continue
        out += ["", f"{name} -- {scored['scorers']['rules'].n} scored pages",
                f"{'':<24}" + "".join(f"{n:>12}" for n, _s in cols),
                f"{'strict accuracy':<24}"
                + "".join(f"{s.accuracy:>12.3f}" for _n, s in cols),
                f"{'accepting alternates':<24}"
                + "".join(f"{s.lenient_accuracy:>12.3f}" for _n, s in cols),
                "", f"key content (the gate is {GATE:.2f} on both rates)"]
        out += columns_label_table(cols, KEY_CONTENT)

    out += ["", f"the disagreement set: what a review of the splits alone "
                f"would have to reach for the set to clear {GATE:.2f}",
            f"{'set':<14}{'scored':>8}{'split':>8}{'fraction':>10}"
            f"{'agreed right':>14}{'must reach':>12}"]
    for name, scored in by_set.items():
        agree = scored["agreement"]
        need = scored["required_review_accuracy"]
        text = ("       -" if need is None
                else "  >1.000" if need > 1.0
                else "   0.000" if need <= 0.0 else f"{need:8.3f}")
        out.append(f"{name:<14}{agree['scored']:>8}"
                   f"{agree['disagreed']['pages']:>8}"
                   f"{_pct(agree['disagreed']['fraction']):>10}"
                   f"{agree['agreed']['correct']:>14}{text:>12}")
    out += ["", "trust believes whoever the in-sample table favours for the "
                "RULES' label class and vision where it says nothing; "
                "structural gives the rules "
            + ", ".join(STRUCTURAL_RULES_WIN)
            + " and vision everything else; confidence believes whoever said "
              "it more confidently, ties to the rules. Read the "
              "out-of-sample rows: trust is scored in sample with a table "
              "learned on those very pages."]
    return "\n".join(out)


# -- the command line --------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="", dest="set_name",
                    choices=("",) + wp1b.SET_NAMES,
                    help="one set; the default reports every set that has "
                         "a saved vision run in it")
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--vision-dir", action="append", default=[],
                    help="a folder of saved vision runs; repeatable, and "
                         "the first that holds a report wins")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    dirs = [Path(d) for d in args.vision_dir] or [VISION_RUNS_DIR]
    runs = load_vision_runs(dirs)
    if not runs:
        print(f"no saved vision runs in {', '.join(str(d) for d in dirs)}; "
              f"run measure_wp5_vision first", file=sys.stderr)
        return 1

    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    names = ((args.set_name,) if args.set_name
             else ("insample", "oos_open", "oos_blind"))
    members: Dict[str, List[str]] = {}
    for name in names:
        members[name] = [rid for rid in wp1b.set_ids(name)
                         if rid in runs and (not only or rid in only)]
    if "oos_blind" in members:
        members["honest_blind"] = [rid for rid in members["oos_blind"]
                                   if rid not in CHECKPOINT]

    oos = wp5.oos_labels()
    votes: Dict[str, List[PageVote]] = {}
    dropped = 0
    for name, ids in members.items():
        for rid in ids:
            if rid in votes:
                continue
            votes[rid], lost = votes_for(rid, runs[rid], oos)
            dropped += lost
    if not votes:
        print("no report in the chosen set has a saved vision run",
              file=sys.stderr)
        return 1

    learned_on = [rid for rid in wp1b.set_ids("insample") if rid in votes]
    table = trust_table([v for rid in learned_on for v in votes[rid]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rid, rows in votes.items():
        (OUT_DIR / f"{rid}.json").write_text(json.dumps({
            "id": rid,
            "run_date": datetime.date.today().isoformat(),
            "pages_compared": len(rows),
            "agreement": agreement(rows),
            "trust_table_learned_on": list(learned_on),
            "disagreements": [v.to_row(policy_labels(v, table))
                              for v in rows if not v.agree],
        }, indent=2), encoding="utf-8")

    by_set = {}
    for name, ids in members.items():
        rows = [v for rid in ids for v in votes.get(rid, ())]
        if rows:
            by_set[name] = score_set(rows, table)

    text = report(by_set, table, learned_on)
    print(text)
    if dropped:
        print(f"\n{dropped} page(s) dropped from every column as confirmed "
              f"disputed hand labels.")
    print(f"\nper-report disagreement lists in {OUT_DIR}")

    if args.append:
        stamp = datetime.date.today().isoformat()
        block = (f"\n### WP6 vote scorecard, {stamp} -- NO model calls\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"Over {len(votes)} report(s) with a saved vision run. "
                   f"The rules' labels and the vision labels come off those "
                   f"runs, the review's off the WP1b runs beside them, and "
                   f"all of them are scored against the same hand labels by "
                   f"the same scorer as every other scorecard here.\n\n"
                   f"```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
