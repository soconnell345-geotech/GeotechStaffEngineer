"""WP1 measurement: planlens page roles against the hand-labelled pages.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1_labels
    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1_labels \
        --only R18 R36 --no-append

Opens every report that has hand labels (``labels.labels_for``) with
``di="auto"``, runs :func:`planlens.document.roles.page_roles`, and scores it:
precision, recall, F1 and support per label; the full confusion matrix; the
accuracy of each report; and every miss on the five GATED roles. Prints the
lot and appends it to the private ledger as a ``## WP1 -- page roles``
section, with the trajectory of the gated numbers across rounds so the effect
of each fix is visible.

**Gate**: precision AND recall >= 0.90 on ``boring_log``, ``test_pit_log``,
``lab_test``, ``narrative`` and ``calculation``. The other thirteen roles are
reported, not gated.

PRIVACY, and one deliberate departure from the brief. The ledger is a tracked
file in a PUBLIC repository, so nothing this script appends may name a
report, a project, a place, a firm or a person. A page's HEADING does name
them — it is the largest type on the page, which on a log form is the firm's
title block and on a lab sheet the laboratory's letterhead. So the miss lines
in the ledger carry the page's KIND and the evidence planlens itself produced
(a rule name and the phrase it matched, both from the module's own vocabulary)
instead of the heading. The same miss list WITH the headings is written to
``raw/checks/wp1_misses.txt``, which is gitignored, for a human to read
beside the rendered pages.
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

#: Precision AND recall must reach this on each gated role.
GATE = 0.90
GATED: Tuple[str, ...] = ("boring_log", "test_pit_log", "lab_test",
                          "narrative", "calculation")


class Scores:
    """Per-label counts, and the rates they make."""

    def __init__(self) -> None:
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


def measure(rid: str, scores: Scores) -> None:
    """Score one report. Opens it once, ``di="auto"``."""
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


def _save_round(rounds: List[dict], scores: Scores, note: str) -> List[dict]:
    rounds = list(rounds)
    rounds.append({
        "round": len(rounds) + 1,
        "date": date.today().isoformat(),
        "note": note,
        "accuracy": round(scores.accuracy, 4),
        "gated": {k: [round(p, 4), round(r, 4)]
                  for k, (p, r) in scores.gated().items()},
    })
    try:
        ROUNDS_JSON.parent.mkdir(parents=True, exist_ok=True)
        ROUNDS_JSON.write_text(json.dumps(rounds, indent=2), encoding="utf-8")
    except OSError:                                   # pragma: no cover
        pass
    return rounds


def build_report(scores: Scores, rounds: List[dict], note: str,
                 label_order: Sequence[str]) -> List[str]:
    n = len(rounds)
    lines = [f"{SECTION} -- round {n}", ""]
    lines.append(
        f"Measured {date.today().isoformat()} with the app venv and the "
        f"editable planlens checkout. {len(scores.per_report)} reports with "
        f"hand labels, {scores.n_pages} pages, opened `di=\"auto\"`, scored "
        f"against `labels.labels_for`. {scores.seconds:.0f} s. "
        f"Gate: precision AND recall >= {GATE:.2f} on "
        f"{', '.join('`%s`' % g for g in GATED)}.")
    if note:
        lines += ["", f"**This round:** {note}"]

    lines += ["", "### Gated roles, round by round", ""]
    header = ["round"] + [f"{g} P/R" for g in GATED] + ["accuracy", "what changed"]
    rows = []
    for r in rounds:
        cells = [str(r["round"])]
        for g in GATED:
            p, rc = r["gated"].get(g, (0.0, 0.0))
            cells.append(f"{p:.3f} / {rc:.3f}")
        cells.append(f"{r['accuracy']:.3f}")
        cells.append(r.get("note") or "")
        rows.append(cells)
    lines += _table(header, rows)

    lines += ["", "### Every role, this round", ""]
    rows = []
    for label in label_order:
        p, r, f1, support = scores.rates(label)
        mark = " **(gated)**" if label in GATED else ""
        rows.append([f"`{label}`{mark}", f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}",
                     str(support)])
    lines += _table(["role", "precision", "recall", "F1", "hand-labelled"],
                    rows)
    lines.append("")
    verdict = "PASSES" if scores.passes() else "does NOT pass"
    lines.append(f"Overall page accuracy **{scores.accuracy:.3f}** over "
                 f"{scores.n_pages} pages. The gate {verdict}.")

    lines += ["", "### Confusion matrix (rows = hand label, columns = "
                  "predicted)", ""]
    short = {lab: lab[:4] for lab in label_order}
    lines += _table(["hand \\ pred"] + [short[c] for c in label_order],
                    [[f"`{h}`"] + [str(scores.cm[(h, c)] or "") or "."
                                   for c in label_order]
                     for h in label_order])
    lines.append("")
    lines.append("Column keys: " + ", ".join(
        f"`{short[c]}` = {c}" for c in label_order) + ".")

    lines += ["", "### Per-report accuracy", ""]
    rows = [[rid, str(total), f"{right / total:.3f}" if total else "-"]
            for rid, (right, total) in sorted(scores.per_report.items())]
    lines += _table(["ID", "pages", "accuracy"], rows)

    lines += ["", "### Misses on the gated roles", ""]
    lines.append(
        f"{len(scores.misses)} pages where a gated role was involved and the "
        f"rules and the hand label disagree, by report. `rule` and `why` are "
        f"the evidence planlens itself recorded; the page HEADING is omitted "
        f"on purpose -- the largest type on a log or a laboratory sheet is a "
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


def write_private_misses(scores: Scores) -> Optional[str]:
    """The same misses WITH page headings, into the gitignored corpus folder."""
    try:
        MISSES_TXT.parent.mkdir(parents=True, exist_ok=True)
        with MISSES_TXT.open("w", encoding="utf-8") as fh:
            fh.write(f"WP1 gated-role misses, {date.today().isoformat()}. "
                     f"PRIVATE: page headings quoted. Never commit this "
                     f"file.\n\n")
            for m in scores.misses:
                fh.write(f"{m['rid']} p{m['page']:<4d} hand={m['hand']:<15s} "
                         f"pred={m['pred']:<15s} kind={m['kind']:<13s} "
                         f"heading={m['heading'][:40]!r} rule={m['rule']!r} "
                         f"why={m['why']!r}\n")
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
    scores = Scores()
    print(f"{'ID':5s} {'pages':>6s} {'accuracy':>9s}  s")
    for rid in wanted:
        before = scores.seconds
        measure(rid, scores)
        right, total = scores.per_report.get(rid, (0, 0))
        print(f"{rid:5s} {total:6d} {right / total:9.3f}  "
              f"{scores.seconds - before:.1f}", flush=True)

    rounds = _load_rounds()
    if not args.no_append and not args.only:
        rounds = _save_round(rounds, scores, args.note)
    else:
        rounds = rounds + [{
            "round": len(rounds) + 1, "date": date.today().isoformat(),
            "note": args.note or "(not recorded: partial run)",
            "accuracy": round(scores.accuracy, 4),
            "gated": {k: [round(p, 4), round(r, 4)]
                      for k, (p, r) in scores.gated().items()}}]

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
    return 0 if scores.passes() else 2


if __name__ == "__main__":
    raise SystemExit(main())
