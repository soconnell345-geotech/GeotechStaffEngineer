"""WP1b measurement: the rules, then triage, then the label review.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
        --set checkpoint --note "cost checkpoint"
    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
        --set insample
    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
        --set oos_blind --no-append

For every report in a set it opens the PDF with ``di="auto"``, runs
:func:`planlens.document.roles.page_roles`, then
:func:`report_ingest.triage.triage` and
:func:`report_ingest.label_review.review_labels`, and scores the labels
BEFORE and AFTER the review against the hand labels: accuracy, precision,
recall and F1 per label, the key-content table, and every change the review
made marked against the hand label as one of

``fixed``
    the rules were wrong, the review put it right;
``broke``
    the rules were right, the review made it wrong;
``still_wrong``
    both are wrong, but differently;
``unscored``
    no hand label for that page, so nobody can say.

That last column is the point of the whole file. A review that raises
accuracy while breaking three correct labels has not earned the raise, and
a total alone would hide it.

THE SETS, AND THE BLIND ONE.

``insample``    the fourteen reports with spreadsheet labels (4,147 pages).
``oos_open``    ten reports the lead hand-labelled from contact sheets and
                then showed to the builder: R01-R08, R10, R14, five pages
                each.
``oos_blind``   the other fourteen of the lead's out-of-sample reports.
                **Nothing about this set may be read except its summary.**
                ``--set oos_blind`` therefore prints one table and writes no
                miss list, no change list and no per-report detail, and the
                script refuses ``--changes`` on it. A held-out set stops
                being held out the moment somebody reads its misses.
``checkpoint``  the six reports of the cost checkpoint: two public and
                short, one 455-page, one with a scanned appendix, one from
                2006, one public with a mixed appendix.

GROUND TRUTH comes from the spreadsheet when the report has one (every page
labelled) and from the lead's out-of-sample file otherwise (five pages).
``--truth`` says which, per report, in the output.

PRIVACY. This appends to a ledger tracked in a PUBLIC repository, so what it
appends is IDs, label names, counts and rates -- never a page heading, a
change's free-text reason, a triage rationale or an anomaly, any of which
can name a firm, a project or a person. All of that goes to
``raw/checks/wp1b/`` and ``raw/checks/triage/``, which are gitignored.

COST. Every report's calls, tokens, seconds and dollars at list price are
printed and recorded. ``--reuse`` re-scores from the saved run instead of
calling the model again, so tuning the scorecard costs nothing.
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus, labels

LEDGER = corpus.RAW_DIR.parent / "MEASUREMENTS.md"
SECTION = "## WP1b -- triage and label review"
#: Where the full, private detail of a run goes. Gitignored, like all of
#: ``raw/``.
RUNS_DIR = corpus.CHECKS_DIR / "wp1b"
#: Where each report's document profile goes, for the lead's audit.
TRIAGE_DIR = corpus.CHECKS_DIR / "triage"
#: The lead's out-of-sample hand labels.
OOS_LABELS = corpus.CHECKS_DIR / "oos" / "labels.json"

#: The labels a downstream reader depends on. The gate after review is 0.98
#: precision AND recall on these.
KEY_CONTENT: Tuple[str, ...] = (
    "narrative", "plan", "profile", "boring_log", "test_pit_log", "cpt_log",
    "dcp_log", "lab_test", "calculation",
)
GATE = 0.98

OOS_OPEN: Tuple[str, ...] = ("R01", "R02", "R03", "R04", "R05", "R06", "R07",
                             "R08", "R10", "R14")
OOS_BLIND: Tuple[str, ...] = ("R17", "R19", "R22", "R25", "R26", "R27",
                              "R31", "R32", "R33", "R34", "R35", "R36",
                              "R37", "R38")
#: The six the cost checkpoint runs on: two short public reports, the
#: 455-page one, the one with a 75-page scanned appendix, and a small 2006
#: report.
CHECKPOINT: Tuple[str, ...] = ("R36", "R37", "R05", "R28", "R15", "R14")

TRIAGE_MODEL = "claude-sonnet-5"
REVIEW_MODEL = "claude-opus-5"


def set_ids(name: str) -> Tuple[str, ...]:
    if name == "insample":
        return tuple(labels.mapped_ids())
    if name == "oos_open":
        return OOS_OPEN
    if name == "oos_blind":
        return OOS_BLIND
    if name == "checkpoint":
        return CHECKPOINT
    raise ValueError(f"unknown set {name!r}")


SET_NAMES = ("insample", "oos_open", "oos_blind", "checkpoint")


# -- ground truth -----------------------------------------------------------

def _oos_labels() -> Dict[str, Dict[int, dict]]:
    if not OOS_LABELS.is_file():
        return {}
    blob = json.loads(OOS_LABELS.read_text(encoding="utf-8"))
    return {rid: {int(p): v for p, v in pages.items()}
            for rid, pages in blob.items()}


def truth_for(rid: str, oos: Dict[str, Dict[int, dict]]
              ) -> Tuple[Dict[int, str], Dict[int, Tuple[str, ...]], str]:
    """``(label per page, acceptable alternates, where it came from)``.

    The spreadsheet wins when a report has one: it labels every page, and
    the out-of-sample file only samples five. A report in neither returns
    nothing and is run but not scored.
    """
    try:
        if rid in labels.mapped_ids():
            hand = {pl.page0: pl.label for pl in labels.labels_for(rid)}
            return hand, {}, "spreadsheet"
    except (FileNotFoundError, ImportError):
        pass
    rows = oos.get(rid)
    if rows:
        return ({p: v["label"] for p, v in rows.items()},
                {p: tuple(v.get("alternates") or ()) for p, v in rows.items()},
                "lead, 5 pages")
    return {}, {}, "none"


# -- scoring ----------------------------------------------------------------

class Scores:
    """Per-label counts over a set of reports, and the rates they make."""

    def __init__(self) -> None:
        self.cm: Counter = Counter()             # (hand, predicted) -> pages
        self.lenient_hits = 0
        self.n = 0

    def add(self, hand: str, pred: str,
            alternates: Sequence[str] = ()) -> None:
        self.cm[(hand, pred)] += 1
        self.n += 1
        if pred == hand or pred in alternates:
            self.lenient_hits += 1

    @property
    def correct(self) -> int:
        return sum(v for (h, p), v in self.cm.items() if h == p)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else float("nan")

    @property
    def lenient_accuracy(self) -> float:
        return self.lenient_hits / self.n if self.n else float("nan")

    def rates(self, label: str) -> Tuple[float, float, float, int]:
        tp = self.cm[(label, label)]
        fp = sum(v for (h, p), v in self.cm.items() if p == label and h != label)
        fn = sum(v for (h, p), v in self.cm.items() if h == label and p != label)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / support if support else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision == precision and recall == recall
              and (precision + recall) else float("nan"))
        return precision, recall, f1, support

    def present_labels(self) -> List[str]:
        seen = {h for h, _ in self.cm} | {p for _, p in self.cm}
        return [x for x in labels.LABELS if x in seen]


def _rate(value: float) -> str:
    return "  --  " if value != value else f"{value:6.3f}"


def _label_table(before: Scores, after: Scores, only: Sequence[str] = ()
                 ) -> List[str]:
    rows = [f"{'label':<16}{'n':>6}"
            f"{'P before':>10}{'R before':>10}{'F1 before':>11}"
            f"{'P after':>10}{'R after':>10}{'F1 after':>10}"]
    names = list(only) if only else after.present_labels()
    for name in names:
        pb, rb, fb, support = before.rates(name)
        pa, ra, fa, support_a = after.rates(name)
        support = max(support, support_a)
        if not support and not only:
            continue
        rows.append(f"{name:<16}{support:>6}"
                    f"{_rate(pb):>10}{_rate(rb):>10}{_rate(fb):>11}"
                    f"{_rate(pa):>10}{_rate(ra):>10}{_rate(fa):>10}")
    return rows


# -- running one report -----------------------------------------------------

def run_report(rid: str, *, reuse: bool = False,
               triage_model: str = TRIAGE_MODEL,
               review_model: str = REVIEW_MODEL,
               set_name: str = "") -> dict:
    """Rules, triage and review on one report; the saved blob either way."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TRIAGE_DIR.mkdir(parents=True, exist_ok=True)
    saved = RUNS_DIR / f"{rid}.json"
    if reuse and saved.is_file():
        blob = json.loads(saved.read_text(encoding="utf-8"))
        blob["reused"] = True
        return blob

    from planlens.document.roles import document_outline, page_roles
    from report_ingest.engine import ClaudeEngine, CostMeter
    from report_ingest.label_review import review_labels
    from report_ingest.triage import document_facts, triage

    started = time.time()
    meter = CostMeter()
    doc = corpus.open_report(rid, di="auto")
    try:
        roles = page_roles(doc)
        outline = document_outline(doc)
        facts = document_facts(doc, roles)
        profile = triage(doc, roles, outline,
                         engine=ClaudeEngine(triage_model, meter=meter),
                         facts=facts)
        review = review_labels(doc, roles, outline, profile,
                               engine=ClaudeEngine(review_model, meter=meter))
        rules = {r.page: r.role for r in roles}
        n_pages = doc.n_pages
    finally:
        doc.close()
        gc.collect()

    blob = {
        "id": rid,
        "set": set_name,
        "run_date": date.today().isoformat(),
        "n_pages": n_pages,
        "triage_model": triage_model,
        "review_model": review_model,
        "rules_labels": {str(k): v for k, v in sorted(rules.items())},
        "profile": profile.to_dict(),
        "review": review.to_dict(),
        "cost": meter.to_dict(),
        "seconds": round(time.time() - started, 1),
        "reused": False,
    }
    saved.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    (TRIAGE_DIR / f"{rid}.json").write_text(
        json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return blob


# -- the report ---------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", dest="set_name", default="checkpoint",
                    choices=SET_NAMES)
    ap.add_argument("--only", default="",
                    help="comma-separated IDs, inside the set")
    ap.add_argument("--reuse", action="store_true",
                    help="score the saved runs instead of calling the model")
    ap.add_argument("--changes", action="store_true",
                    help="print every change with its verdict (never on the "
                         "blind set)")
    ap.add_argument("--triage-model", default=TRIAGE_MODEL)
    ap.add_argument("--review-model", default=REVIEW_MODEL)
    ap.add_argument("--no-append", action="store_true",
                    help="print, do not touch the ledger")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    if not corpus.raw_available():
        print("the report-ingest corpus is not on this machine", file=sys.stderr)
        return 2

    blind = args.set_name == "oos_blind"
    if blind and args.changes:
        print("--changes is refused on the blind set: reading its misses is "
              "how a held-out set stops being one", file=sys.stderr)
        return 2

    ids = set_ids(args.set_name)
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        ids = tuple(x for x in ids if x in wanted)
    if not ids:
        print("no reports selected", file=sys.stderr)
        return 2

    oos = _oos_labels()
    before, after = Scores(), Scores()
    verdicts: Counter = Counter()
    per_report: List[dict] = []
    change_rows: List[dict] = []
    triage_rows: List[dict] = []
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
              "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}

    print(f"WP1b: {args.set_name} -- {len(ids)} report(s); triage "
          f"{args.triage_model}, review {args.review_model}"
          + ("  [REUSING SAVED RUNS]" if args.reuse else ""))
    if blind:
        print("blind set: summary only, by design")

    for rid in ids:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blob = run_report(rid, reuse=args.reuse,
                                  triage_model=args.triage_model,
                                  review_model=args.review_model,
                                  set_name=args.set_name)
        except Exception as exc:                    # keep the set going
            print(f"  {rid}: FAILED -- {type(exc).__name__}: {exc}")
            per_report.append({"id": rid, "failed": f"{type(exc).__name__}"})
            continue

        rules = {int(k): v for k, v in blob["rules_labels"].items()}
        final = {int(k): v for k, v in blob["review"]["final_labels"].items()}
        hand, alternates, source = truth_for(rid, oos)
        cost = blob["cost"]
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            totals[key] += cost.get(key, 0)
        totals["dollars"] += cost.get("dollars", 0.0)
        totals["seconds"] += blob.get("seconds", 0.0)

        rb = ra = 0
        for page, want in sorted(hand.items()):
            alts = alternates.get(page, ())
            was, now = rules.get(page, "other"), final.get(page, "other")
            before.add(want, was, alts)
            after.add(want, now, alts)
            rb += int(was == want)
            ra += int(now == want)

        for change in blob["review"]["changes"]:
            page = int(change["page"])
            want = hand.get(page)
            if want is None:
                verdict = "unscored"
            elif change["to"] == want:
                verdict = "fixed" if change["from"] != want else "unscored"
            elif change["from"] == want:
                verdict = "broke"
            else:
                verdict = "still_wrong"
            verdicts[verdict] += 1
            change_rows.append({"id": rid, "page": page, **change,
                                "hand": want or "-", "verdict": verdict})

        profile = blob["profile"]
        triage_rows.append({
            "id": rid,
            "document_type": profile["document_type"],
            "workflow": profile["workflow"],
            "bound": len(profile["bound_together"]),
            "toc": profile["toc_agreement"],
            "scan": profile["scan_fraction"],
        })
        review = blob["review"]
        per_report.append({
            "id": rid, "pages": blob["n_pages"], "scored": len(hand),
            "source": source,
            "before": rb / len(hand) if hand else float("nan"),
            "after": ra / len(hand) if hand else float("nan"),
            "changes": len(review["changes"]),
            "rejected": len(review["rejected_changes"]),
            "unresolved": len(review["unresolved"]),
            "tool_calls": review["tool_calls"],
            "budget": review["budget"],
            "calls": cost.get("calls", 0),
            "dollars": cost.get("dollars", 0.0),
            "seconds": blob.get("seconds", 0.0),
        })
        if not blind:
            row = per_report[-1]
            print(f"  {rid}: {row['pages']:>4} pp, {row['scored']:>4} scored "
                  f"({source}); accuracy {row['before']:.3f} -> "
                  f"{row['after']:.3f}; {row['changes']} change(s), "
                  f"{row['tool_calls']}/{row['budget']} tool calls, "
                  f"{row['calls']} model calls, ${row['dollars']:.3f}, "
                  f"{row['seconds']:.0f} s")

    lines = _render(args, ids, before, after, verdicts, per_report,
                    triage_rows, totals, blind)
    print("\n".join(lines))

    if args.changes and change_rows:
        print("\nevery change, with its verdict against the hand label")
        for row in change_rows:
            print(f"  {row['id']} p{row['page']:<4} {row['from']:<14} -> "
                  f"{row['to']:<14} hand={row['hand']:<14} "
                  f"{row['verdict']:<12} [{row['evidence']}] {row['reason']}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    detail = RUNS_DIR / f"detail_{args.set_name}.json"
    detail.write_text(json.dumps({
        "set": args.set_name, "date": date.today().isoformat(),
        "ids": list(ids), "per_report": per_report,
        "verdicts": dict(verdicts), "totals": totals,
        # The blind set's changes are written nowhere, not even privately:
        # a file that exists is a file somebody reads.
        "changes": [] if blind else change_rows,
    }, indent=2), encoding="utf-8")
    print(f"\nprivate detail: {detail}")
    print(f"document profiles: {TRIAGE_DIR}")

    if not args.no_append:
        _append(lines, args)
        print(f"appended to {LEDGER}")
    return 0


def _render(args, ids, before: Scores, after: Scores, verdicts: Counter,
            per_report: List[dict], triage_rows: List[dict],
            totals: dict, blind: bool) -> List[str]:
    out: List[str] = []
    out.append("")
    out.append(f"{args.set_name}: {len(ids)} report(s), {after.n} scored pages")
    out.append(f"{'':<24}{'before':>10}{'after':>10}")
    out.append(f"{'strict accuracy':<24}{before.accuracy:>10.3f}"
               f"{after.accuracy:>10.3f}")
    out.append(f"{'accepting alternates':<24}"
               f"{before.lenient_accuracy:>10.3f}"
               f"{after.lenient_accuracy:>10.3f}")
    out.append("")
    out.append(f"key content (the gate is {GATE:.2f} on both rates after "
               f"review)")
    out.extend(_label_table(before, after, KEY_CONTENT))
    failed: List[str] = []
    for name in KEY_CONTENT:
        precision, recall, _f1, support = after.rates(name)
        if not support:
            continue                      # the set has no page of that label
        if precision != precision or recall != recall or min(
                precision, recall) < GATE:
            failed.append(name)
    out.append("below the gate after review: "
               + (", ".join(failed) if failed else "none"))
    if not blind:
        out.append("")
        out.append("every label")
        out.extend(_label_table(before, after))
        out.append("")
        out.append("what the review's changes did, against the hand labels")
        for verdict in ("fixed", "broke", "still_wrong", "unscored"):
            out.append(f"  {verdict:<14}{verdicts.get(verdict, 0):>5}")
        out.append("")
        out.append(f"{'report':<8}{'pages':>7}{'scored':>8}{'before':>9}"
                   f"{'after':>8}{'chg':>5}{'tools':>7}{'calls':>7}"
                   f"{'$':>8}{'s':>7}")
        for row in per_report:
            if row.get("failed"):
                out.append(f"{row['id']:<8}  FAILED: {row['failed']}")
                continue
            out.append(f"{row['id']:<8}{row['pages']:>7}{row['scored']:>8}"
                       f"{row['before']:>9.3f}{row['after']:>8.3f}"
                       f"{row['changes']:>5}"
                       f"{row['tool_calls']:>4}/{row['budget']:<2}"
                       f"{row['calls']:>7}{row['dollars']:>8.3f}"
                       f"{row['seconds']:>7.0f}")
        out.append("")
        out.append("what triage said (enumerated fields only; the rationale "
                   "and anomalies stay in raw/checks/triage/)")
        out.append(f"{'report':<8}{'document_type':<30}{'workflow':<16}"
                   f"{'bound':>6}{'toc':>10}{'scan':>7}")
        for row in triage_rows:
            out.append(f"{row['id']:<8}{row['document_type']:<30}"
                       f"{row['workflow']:<16}{row['bound']:>6}"
                       f"{row['toc']:>10}{row['scan']:>7.2f}")
    out.append("")
    n = max(1, len([r for r in per_report if not r.get("failed")]))
    out.append(f"cost: {totals['calls']} model calls, "
               f"{totals['input_tokens']:,} input tokens "
               f"(+{totals['cache_read_tokens']:,} cached), "
               f"{totals['output_tokens']:,} output, "
               f"${totals['dollars']:.2f}, {totals['seconds']:.0f} s "
               f"-- ${totals['dollars'] / n:.3f} and "
               f"{totals['seconds'] / n:.0f} s a report")
    return out


def _append(lines: Sequence[str], args) -> None:
    head = [
        "", "---", "",
        f"{SECTION} -- {args.set_name} ({date.today().isoformat()})", "",
        f"Triage {args.triage_model}; review {args.review_model}. "
        + (args.note or "No note given."),
        "",
        "```",
    ]
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(head) + "\n")
        fh.write("\n".join(lines).strip("\n") + "\n")
        fh.write("```\n")


if __name__ == "__main__":
    sys.exit(main())
