"""WP1 measurement: planlens page roles against the hand-labelled pages.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1_labels
    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1_labels \
        --only R18 --no-append

Opens every report that has hand labels (``labels.labels_for``) with
``di="auto"``, runs :func:`planlens.document.roles.page_roles`, and scores it
TWICE: once over the development reports and once over the held-out ones.
Prints precision, recall, F1 and support per role, the full confusion matrix
and the per-report accuracy for each set, and appends the lot to the private
ledger with the trajectory of both sets round by round.

**The split.** Rules are developed against :data:`DEV_IDS` only. The five
reports in :data:`HELDOUT_IDS` are a 2001 optical-character-over-scan report,
a pure scan read through Azure, a short second volume, a 729-page report from
a different firm and a report in a non-US format -- deliberately the hardest
and least alike. **The gate is on the HELD-OUT set**: precision AND recall
>= 0.90 on ``boring_log``, ``test_pit_log``, ``lab_test``, ``narrative`` and
``calculation``. The other thirteen roles are reported, not gated.

Where held-out lags development by more than :data:`LAG` on a gated role the
section says so and names the confusion pair. It does NOT list the held-out
misses page by page: closing a gap by reading the held-out pages is how a
held-out set stops being one. Development misses are listed in full.

PRIVACY, and one deliberate departure from the brief. The ledger is a tracked
file in a PUBLIC repository, so nothing this script appends may name a
report, a project, a place, a firm or a person. A page's HEADING does name
them -- it is the largest type on the page, which on a log form is the firm's
title block and on a lab sheet the laboratory's letterhead. So the miss lines
in the ledger carry the page's KIND and the evidence planlens itself produced
(a rule name and the phrase it matched, both from the module's own
vocabulary) instead of the heading. The same miss list WITH the headings is
written to ``raw/checks/wp1_misses.txt``, which is gitignored.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus, labels

LEDGER = corpus.RAW_DIR.parent / "MEASUREMENTS.md"
SECTION = "## WP1 -- page roles"
#: Where the rounds so far are kept, so a run can print the trajectory.
ROUNDS_JSON = corpus.RAW_DIR / "wp1_rounds.json"
#: The miss list that carries page headings. Gitignored, like everything in
#: ``raw/``; the ledger gets the same misses without them.
MISSES_TXT = corpus.CHECKS_DIR / "wp1_misses.txt"

#: Precision AND recall must reach this on each gated role, on HELD_OUT.
GATE = 0.90
#: How far held-out may fall behind development before the section says so.
LAG = 0.05
GATED: Tuple[str, ...] = ("boring_log", "test_pit_log", "lab_test",
                          "narrative", "calculation")

#: The reports the rules are developed against.
DEV_IDS: Tuple[str, ...] = ("R09", "R12", "R15", "R16", "R20", "R23", "R28",
                            "R29", "R30")
#: The reports nothing may be tuned against.
HELDOUT_IDS: Tuple[str, ...] = ("R11", "R13", "R18", "R21", "R24")

DEV, HELD = "dev", "held_out"
SPLITS = (DEV, HELD)
SPLIT_TITLE = {DEV: "development set", HELD: "held-out set"}


def split_of(rid: str) -> Optional[str]:
    if rid in DEV_IDS:
        return DEV
    if rid in HELDOUT_IDS:
        return HELD
    return None


class Scores:
    """Per-label counts for one set of reports, and the rates they make."""

    def __init__(self, split: str = DEV) -> None:
        self.split = split
        self.cm: Counter = Counter()          # (hand, predicted) -> pages
        self.per_report: Dict[str, List[int]] = {}
        self.misses: List[dict] = []
        self.seconds = 0.0

    # -- accumulation ------------------------------------------------------
    def add(self, rid: str, hand: str, pred: str, page: int,
            kind: str, heading: Optional[str], evidence: dict) -> None:
        self.cm[(hand, pred)] += 1
        row = self.per_report.setdefault(rid, [0, 0])
        row[1] += 1
        if hand == pred:
            row[0] += 1
        elif hand in GATED or pred in GATED:
            why = evidence.get("why") or evidence.get("declared") or ""
            self.misses.append({
                "rid": rid, "page": page, "hand": hand, "pred": pred,
                "kind": kind, "heading": heading or "",
                "rule": str(evidence.get("rule") or "")[:60],
                "why": str(why)[:60],
            })

    # -- rates -------------------------------------------------------------
    def rates(self, label: str) -> Tuple[float, float, float, int]:
        tp = self.cm[(label, label)]
        fp = sum(v for (h, p), v in self.cm.items() if p == label and h != label)
        fn = sum(v for (h, p), v in self.cm.items() if h == label and p != label)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return precision, recall, f1, support

    @property
    def n_pages(self) -> int:
        return sum(self.cm.values())

    @property
    def accuracy(self) -> float:
        right = sum(v for (h, p), v in self.cm.items() if h == p)
        return right / self.n_pages if self.n_pages else 0.0

    def gated(self) -> Dict[str, Tuple[float, float]]:
        return {label: self.rates(label)[:2] for label in GATED}

    def passes(self) -> bool:
        return all(p >= GATE and r >= GATE for p, r in self.gated().values())

    def worst_pairs(self, label: str, n: int = 3) -> List[Tuple[str, int]]:
        """The confusions that cost this label most, as "hand -> pred"."""
        pairs = Counter()
        for (h, p), v in self.cm.items():
            if h == p:
                continue
            if h == label:
                pairs[f"{h} -> {p}"] += v
            elif p == label:
                pairs[f"{h} -> {p}"] += v
        return pairs.most_common(n)


def measure(rid: str, scores: Scores) -> None:
    """Score one report into one set. Opens it once, ``di="auto"``."""
    from planlens.document.roles import page_roles

    hand = {pl.page0: pl.label for pl in labels.labels_for(rid)}
    start = time.perf_counter()
    with corpus.open_report(rid, di="auto", warn=False) as doc:
        summaries = {s.page: s for s in doc.page_map()}
        for role in page_roles(doc):
            truth = hand.get(role.page)
            if truth is None:
                continue                       # a page the sheet never labelled
            s = summaries[role.page]
            scores.add(rid, truth, role.role, role.page, s.kind, s.heading,
                       role.evidence)
    scores.seconds += time.perf_counter() - start
    gc.collect()


# -- output -----------------------------------------------------------------

def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _load_rounds() -> List[dict]:
    if not ROUNDS_JSON.is_file():
        return []
    try:
        return json.loads(ROUNDS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _round_row(scores: Dict[str, Scores], note: str, n: int) -> dict:
    return {
        "round": n,
        "date": date.today().isoformat(),
        "note": note,
        "sets": {
            split: {
                "accuracy": round(scores[split].accuracy, 4),
                "n_pages": scores[split].n_pages,
                "gated": {k: [round(p, 4), round(r, 4)]
                          for k, (p, r) in scores[split].gated().items()},
            } for split in SPLITS
        },
    }


def _save_round(rounds: List[dict], row: dict) -> List[dict]:
    rounds = list(rounds) + [row]
    try:
        ROUNDS_JSON.parent.mkdir(parents=True, exist_ok=True)
        ROUNDS_JSON.write_text(json.dumps(rounds, indent=2), encoding="utf-8")
    except OSError:                                   # pragma: no cover
        pass
    return rounds


def _gated_cells(entry: dict, split: str) -> List[str]:
    """The five gated P/R for one set of one round, or dashes."""
    sets = entry.get("sets")
    if sets is None:
        # A round measured before the split existed: one set, all 14 reports.
        if split != DEV:
            return ["-"] * len(GATED) + ["-"]
        gated = entry.get("gated", {})
        cells = [f"{gated.get(g, (0, 0))[0]:.3f} / {gated.get(g, (0, 0))[1]:.3f}"
                 for g in GATED]
        return cells + [f"{entry.get('accuracy', 0.0):.3f}"]
    got = sets.get(split, {})
    gated = got.get("gated", {})
    cells = [f"{gated.get(g, (0, 0))[0]:.3f} / {gated.get(g, (0, 0))[1]:.3f}"
             for g in GATED]
    return cells + [f"{got.get('accuracy', 0.0):.3f}"]


def _trajectory(rounds: List[dict]) -> List[str]:
    lines: List[str] = []
    for split in SPLITS:
        lines += ["", f"**{SPLIT_TITLE[split].title()}**, round by round", ""]
        header = ["round"] + [f"{g} P/R" for g in GATED] + ["accuracy",
                                                            "what changed"]
        rows = []
        for entry in rounds:
            note = entry.get("note") or ""
            if entry.get("sets") is None:
                note = (note + " (all 14 reports; measured before the "
                                "split)").strip()
            rows.append([str(entry["round"])] + _gated_cells(entry, split)
                        + [note])
        lines += _table(header, rows)
    return lines


def _lag_lines(scores: Dict[str, Scores]) -> List[str]:
    """Where held-out falls behind development, and on what confusion."""
    out: List[str] = []
    for label in GATED:
        dp, dr = scores[DEV].gated()[label]
        hp, hr = scores[HELD].gated()[label]
        gaps = []
        if dp - hp > LAG:
            gaps.append(f"precision {dp:.3f} to {hp:.3f}")
        if dr - hr > LAG:
            gaps.append(f"recall {dr:.3f} to {hr:.3f}")
        if not gaps:
            continue
        pairs = scores[HELD].worst_pairs(label)
        named = ", ".join(f"`{p}` ({n} pages)" for p, n in pairs) or "none"
        out.append(f"- **`{label}` lags by more than {LAG:.2f}**: "
                   f"{'; '.join(gaps)}. The confusions that cost it on the "
                   f"held-out set: {named}.")
    if not out:
        out.append(f"- No gated role lags development by more than "
                   f"{LAG:.2f} on either rate.")
    return out


def _set_section(scores: Scores, label_order: Sequence[str],
                 with_misses: bool) -> List[str]:
    lines: List[str] = []
    title = SPLIT_TITLE[scores.split].title()
    ids = ", ".join(sorted(scores.per_report))
    lines += [f"#### {title}", "",
              f"{len(scores.per_report)} reports ({ids}), "
              f"{scores.n_pages} pages, {scores.seconds:.0f} s.", ""]
    rows = []
    for label in label_order:
        p, r, f1, support = scores.rates(label)
        mark = " **(gated)**" if label in GATED else ""
        rows.append([f"`{label}`{mark}", f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}",
                     str(support)])
    lines += _table(["role", "precision", "recall", "F1", "hand-labelled"],
                    rows)
    lines += ["", f"Page accuracy **{scores.accuracy:.3f}** over "
                  f"{scores.n_pages} pages."]
    if scores.split == HELD:
        lines[-1] += (f" The gate "
                      f"{'PASSES' if scores.passes() else 'does NOT pass'}.")

    lines += ["", "Confusion matrix (rows = hand label, columns = predicted):",
              ""]
    short = {lab: lab[:4] for lab in label_order}
    lines += _table(["hand \\ pred"] + [short[c] for c in label_order],
                    [[f"`{h}`"] + [str(scores.cm[(h, c)] or "") or "."
                                   for c in label_order]
                     for h in label_order])
    lines += ["", "Column keys: " + ", ".join(
        f"`{short[c]}` = {c}" for c in label_order) + ".", ""]

    lines += ["Per-report accuracy:", ""]
    lines += _table(["ID", "pages", "accuracy"],
                    [[rid, str(total), f"{right / total:.3f}" if total else "-"]
                     for rid, (right, total) in sorted(scores.per_report.items())])
    lines.append("")

    if not with_misses:
        pairs = Counter()
        for m in scores.misses:
            pairs[f"{m['hand']} -> {m['pred']}"] += 1
        lines.append(
            f"{len(scores.misses)} pages where a gated role is involved and "
            f"the rules and the hand label disagree. They are NOT listed page "
            f"by page: closing a gap by reading the held-out pages is how a "
            f"held-out set stops being one. The confusions, by size:")
        lines.append("")
        lines += _table(["hand -> predicted", "pages"],
                        [[f"`{k}`", str(v)] for k, v in pairs.most_common()])
        lines.append("")
        return lines

    lines.append(
        f"{len(scores.misses)} pages where a gated role is involved and the "
        f"rules and the hand label disagree. `rule` and `why` are the "
        f"evidence planlens itself recorded; the page HEADING is omitted on "
        f"purpose -- the largest type on a log or a laboratory sheet is a "
        f"firm's title block, and this file is tracked in a public "
        f"repository. The same list WITH headings is written to "
        f"`raw/checks/wp1_misses.txt`, which is gitignored.")
    lines.append("")
    by_rid: Dict[str, List[dict]] = {}
    for m in scores.misses:
        by_rid.setdefault(m["rid"], []).append(m)
    for rid in sorted(by_rid):
        lines.append(f"**{rid}** ({len(by_rid[rid])})")
        lines.append("")
        lines.append("```")
        for m in by_rid[rid]:
            lines.append(
                f"{m['rid']} p{m['page']:<4d} hand={m['hand']:<15s} "
                f"pred={m['pred']:<15s} kind={m['kind']:<13s} "
                f"rule={m['rule']!r} why={m['why']!r}")
        lines.append("```")
        lines.append("")
    return lines


def build_report(scores: Dict[str, Scores], rounds: List[dict], note: str,
                 label_order: Sequence[str]) -> List[str]:
    n = len(rounds)
    total = sum(scores[s].n_pages for s in SPLITS)
    lines = [f"{SECTION} -- round {n}", ""]
    lines.append(
        f"Measured {date.today().isoformat()} with the app venv and the "
        f"editable planlens checkout. {total} hand-labelled pages, opened "
        f"`di=\"auto\"`, scored against `labels.labels_for`, split into a "
        f"development set ({', '.join(DEV_IDS)}) and a held-out set "
        f"({', '.join(HELDOUT_IDS)}). **The gate is on the held-out set**: "
        f"precision AND recall >= {GATE:.2f} on "
        f"{', '.join('`%s`' % g for g in GATED)}.")
    if note:
        lines += ["", f"**This round:** {note}"]
    lines += _trajectory(rounds)
    lines += ["", "### Where held-out lags development", ""]
    lines += _lag_lines(scores)
    lines += ["", "### This round in full", ""]
    lines += _set_section(scores[HELD], label_order, with_misses=False)
    lines += _set_section(scores[DEV], label_order, with_misses=True)
    return lines


def write_private_misses(scores: Dict[str, Scores]) -> Optional[str]:
    """The same misses WITH page headings, into the gitignored corpus folder."""
    try:
        MISSES_TXT.parent.mkdir(parents=True, exist_ok=True)
        with MISSES_TXT.open("w", encoding="utf-8") as fh:
            fh.write(f"WP1 gated-role misses, {date.today().isoformat()}. "
                     f"PRIVATE: page headings quoted. Never commit this "
                     f"file.\n\n")
            for split in SPLITS:
                fh.write(f"=== {SPLIT_TITLE[split]} ===\n")
                for m in scores[split].misses:
                    fh.write(
                        f"{m['rid']} p{m['page']:<4d} hand={m['hand']:<15s} "
                        f"pred={m['pred']:<15s} kind={m['kind']:<13s} "
                        f"heading={m['heading'][:40]!r} rule={m['rule']!r} "
                        f"why={m['why']!r}\n")
                fh.write("\n")
        return str(MISSES_TXT)
    except OSError:                                   # pragma: no cover
        return None


def main(argv: Optional[List[str]] = None) -> int:
    from planlens.document.roles import ROLES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", metavar="ID",
                        help="score just these IDs (default: every report "
                             "that has hand labels)")
    parser.add_argument("--no-append", action="store_true",
                        help="print the section but do not touch the ledger")
    parser.add_argument("--note", default="",
                        help="what changed since the previous round; goes in "
                             "the trajectory table")
    args = parser.parse_args(argv)

    if not corpus.raw_available() or not labels.labels_available():
        print("the private corpus or the hand labels are not on this "
              "machine; nothing to measure", file=sys.stderr)
        return 1

    warnings.simplefilter("ignore", RuntimeWarning)
    wanted = args.only or labels.mapped_ids()
    scores = {split: Scores(split) for split in SPLITS}
    print(f"{'ID':5s} {'set':9s} {'pages':>6s} {'accuracy':>9s}  s")
    for rid in wanted:
        split = split_of(rid)
        if split is None:
            print(f"{rid:5s} {'(no set)':9s} -- not in either set, skipped",
                  file=sys.stderr)
            continue
        before = scores[split].seconds
        measure(rid, scores[split])
        right, total = scores[split].per_report.get(rid, (0, 0))
        print(f"{rid:5s} {split:9s} {total:6d} {right / total:9.3f}  "
              f"{scores[split].seconds - before:.1f}", flush=True)

    rounds = _load_rounds()
    row = _round_row(scores, args.note, len(rounds) + 1)
    if not args.no_append and not args.only:
        rounds = _save_round(rounds, row)
    else:
        rounds = rounds + [row]

    lines = build_report(scores, rounds, args.note, ROLES)
    print()
    print("\n".join(lines))
    private = write_private_misses(scores)
    if private:
        print(f"\nmisses with headings (PRIVATE, gitignored): {private}")

    if not args.no_append and not args.only:
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write("\n---\n\n")
            fh.write("\n".join(lines))
        print(f"appended to {LEDGER}")
    return 0 if scores[HELD].passes() else 2


if __name__ == "__main__":
    raise SystemExit(main())
