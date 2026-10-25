"""Scoring the page labels: the sets, the rates, and what a change did.

Shared by the development scorecard
(``module_work/report_ingest_harness/measure_wp1b.py``) and the cluster
entry point (:mod:`report_ingest.cluster_scoring`), because two copies of
this arithmetic would drift and the second copy's numbers would be the ones
nobody checked.

THE VERDICT ON EACH CHANGE is what the file is really for. A review that
raises accuracy while breaking correct labels has not earned the raise, and
an accuracy figure alone cannot show that. So every change the review makes
is graded against the hand label: ``fixed``, ``broke``, ``still_wrong``,
``disputed`` or ``unscored``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.corpus import LABELS

__all__ = [
    "KEY_CONTENT", "GATE", "OOS_OPEN", "OOS_BLIND", "CHECKPOINT", "DISPUTED",
    "Scores", "disputed_drop", "verdict_for", "label_table",
    "columns_label_table", "rate", "gate_failures",
    "NARRATIVE_FRONT", "BOUND_READER", "CONSEQUENCES", "HARMLESS",
    "STRUCTURE", "reader_for", "consequence", "is_harmful",
    "hand_sections", "miss_structure", "harm_score",
]

#: The labels a downstream reader depends on. The gate after review is 0.98
#: precision AND recall on these. The other nine are reported, not gated.
KEY_CONTENT: Tuple[str, ...] = (
    "narrative", "plan", "profile", "boring_log", "test_pit_log", "cpt_log",
    "dcp_log", "lab_test", "calculation",
)
GATE = 0.98

#: The lead's out-of-sample reports, hand-labelled five pages each from
#: contact sheets. The open half was shown to the builder; the blind half
#: was not.
OOS_OPEN: Tuple[str, ...] = ("R01", "R02", "R03", "R04", "R05", "R06", "R07",
                             "R08", "R10", "R14")
OOS_BLIND: Tuple[str, ...] = ("R17", "R19", "R22", "R25", "R26", "R27",
                              "R31", "R32", "R33", "R34", "R35", "R36",
                              "R37", "R38")
#: The six of the cost checkpoint. Two of them, R36 and R37, are also in the
#: blind set: the checkpoint brief named them and their changes were read, so
#: they are no longer blind and the blind set has to be reported twice.
CHECKPOINT: Tuple[str, ...] = ("R36", "R37", "R05", "R28", "R15", "R14")

#: Pages where the review contradicted the hand label and the lead, looking
#: at the page, judged the HAND label the doubtful one. The spreadsheet is
#: never edited: a hand label records what a person decided, and rewriting it
#: to suit a model would destroy the only independent thing in the
#: measurement. A confirmed entry drops the page from the before AND the
#: after score, so it counts as neither a rule hit nor a review miss.
DISPUTED: Dict[Tuple[str, int], dict] = {
    ("R37", 47): {
        "review": "other", "hand": "calculation", "confirmed": True,
        "note": "a web-tool disclaimer page inside the calculation appendix; "
                "no inputs or results on it"},
    ("R28", 217): {
        "review": "letter", "hand": "lab_test", "confirmed": True,
        "note": "a laboratory's transmittal cover letter inside the "
                "laboratory appendix"},
    ("R15", 8): {
        "review": "cover", "hand": "narrative", "confirmed": True,
        "note": "a one-line volume title sheet inside the narrative run"},
    ("R15", 19): {
        "review": "cover", "hand": "narrative", "confirmed": True,
        "note": "the second volume's title sheet, the same one-line form"},
}


def disputed_drop(rid: str, page: int, after_label: str) -> bool:
    """Should this page be dropped from the score as a confirmed dispute?

    Only when the lead has confirmed it AND the review still says what it
    said when the dispute was raised. A later run that moves the page
    somewhere else is a new answer, and a new answer is scored rather than
    excused by an old dispute.
    """
    row = DISPUTED.get((rid, int(page)))
    return bool(row and row["confirmed"] and after_label == row["review"])


def verdict_for(rid: str, page: int, from_label: str, to_label: str,
                hand: Optional[str]) -> str:
    """What one change did, against the hand label."""
    row = DISPUTED.get((rid, int(page)))
    if row is not None and to_label == row["review"]:
        return "disputed"
    if hand is None:
        return "unscored"
    if to_label == hand:
        return "fixed" if from_label != hand else "unscored"
    if from_label == hand:
        return "broke"
    return "still_wrong"


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
        """``(precision, recall, f1, support)``.

        A label the set contains no page of comes back as NaN rather than
        zero: it is unmeasured, and an unmeasured label must not read as
        either a perfect score or a failing one.
        """
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
        return [x for x in LABELS if x in seen]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": self.n,
            "accuracy": None if self.n == 0 else round(self.accuracy, 4),
            "lenient_accuracy": (None if self.n == 0
                                 else round(self.lenient_accuracy, 4)),
            "per_label": {
                label: {"precision": _none_for_nan(p), "recall": _none_for_nan(r),
                        "f1": _none_for_nan(f), "support": s}
                for label in self.present_labels()
                for p, r, f, s in [self.rates(label)]},
        }


def _none_for_nan(value: float) -> Optional[float]:
    return None if value != value else round(value, 4)


def rate(value: float) -> str:
    """One rate, or a dash where there was nothing to measure."""
    return "  --  " if value != value else f"{value:6.3f}"


def label_table(before: Scores, after: Scores,
                only: Sequence[str] = ()) -> List[str]:
    """The per-label before-and-after table, as lines."""
    rows = [f"{'label':<16}{'n':>6}"
            f"{'P before':>10}{'R before':>10}{'F1 before':>11}"
            f"{'P after':>10}{'R after':>10}{'F1 after':>10}"]
    names = list(only) if only else after.present_labels()
    for name in names:
        pb, rb, fb, support = before.rates(name)
        pa, ra, fa, support_after = after.rates(name)
        support = max(support, support_after)
        if not support and not only:
            continue
        rows.append(f"{name:<16}{support:>6}"
                    f"{rate(pb):>10}{rate(rb):>10}{rate(fb):>11}"
                    f"{rate(pa):>10}{rate(ra):>10}{rate(fa):>10}")
    return rows


def columns_label_table(columns: Sequence[Tuple[str, Scores]],
                        only: Sequence[str] = ()) -> List[str]:
    """Per-label precision and recall, one PAIR of columns per run.

    :func:`label_table` is the two-column before-and-after form the label
    scorecards print. This is the same table widened to however many runs
    there are, so three ways of labelling a page -- the rules alone, the
    rules the review corrected, and a model looking at the picture -- can be
    read off one set of rows against one set of hand labels.

    A run that did not happen simply has no columns: a report with no review
    saved beside it prints two, and one with a review prints three. The row
    set is the union of the labels the runs produced, so a label only one of
    them ever emits is still visible.
    """
    if not columns:
        return []
    header = f"{'label':<16}{'n':>6}"
    for name, _scores in columns:
        header += f"{'P ' + name:>10}{'R ' + name:>10}"
    rows = [header]
    if only:
        names: List[str] = list(only)
    else:
        seen: List[str] = []
        for _name, scores in columns:
            for label in scores.present_labels():
                if label not in seen:
                    seen.append(label)
        names = [x for x in LABELS if x in seen]
    for label in names:
        support = max(scores.rates(label)[3] for _n, scores in columns)
        if not support and not only:
            continue
        line = f"{label:<16}{support:>6}"
        for _name, scores in columns:
            precision, recall, _f1, _support = scores.rates(label)
            line += f"{rate(precision):>10}{rate(recall):>10}"
        rows.append(line)
    return rows


def gate_failures(after: Scores, gate: float = GATE) -> List[str]:
    """Key-content labels below the gate, ignoring ones with no pages."""
    failed: List[str] = []
    for name in KEY_CONTENT:
        precision, recall, _f1, support = after.rates(name)
        if not support:
            continue
        if precision != precision or recall != recall or min(
                precision, recall) < gate:
            failed.append(name)
    return failed


# ---------------------------------------------------------------------------
# what a wrong label costs: the reader it sends the page to (BACKLOG RI-73)
# ---------------------------------------------------------------------------

#: The labels the narrative reader is handed as the front of the report
#: beside ``narrative`` (:data:`report_ingest.narrative_reader.FRONT_LABELS`),
#: so a wrong one of these reaches that reader too.
NARRATIVE_FRONT: Tuple[str, ...] = ("cover", "letter", "toc")

#: The reader a page of a bound document goes to: its own record.
BOUND_READER = "bound"

#: Every way a wrong label can land, from the hand label's reader and the
#: given label's. The first two change no reader; the rest do, and
#: ``harmful`` counts them. ``bound/parent swap`` is a page read into the
#: wrong one of a bound document and its parent.
CONSEQUENCES: Tuple[str, ...] = (
    "same reader", "same (neither read)", "different reader",
    "should be read, is not", "read, should not be", "bound/parent swap",
    "bound page not read", "page read as bound",
)
HARMLESS: Tuple[str, ...] = CONSEQUENCES[:2]

#: How a wrong page sits in the run of pages its hand label makes (a run =
#: the longest stretch of consecutive pages with one hand label): every page
#: of a run of two or more wrong; a block of wrong pages touching the run's
#: first or last page, one page or more; a block strictly inside the run;
#: a run of one page.
STRUCTURE: Tuple[str, ...] = (
    "whole section wrong", "edge, 1 page", "edge, 2+ pages",
    "inside, 1 page", "inside, 2+ pages", "one-page section",
)


def reader_for(label: Optional[str]) -> Optional[str]:
    """The reader a page with this label goes to, or None for none.

    The graph's own table (:data:`report_ingest.graph.ITEM_READERS`), with the
    front matter the narrative reader is handed and a bound document's pages,
    which are read into a record of their own. Imported where it is used:
    the graph imports the readers and this module is imported by scorers that
    open no document.
    """
    from report_ingest.bound import APPENDED_LABEL
    from report_ingest.graph import ITEM_READERS

    if not label:
        return None
    if label == APPENDED_LABEL:
        return BOUND_READER
    if label in NARRATIVE_FRONT:
        return "narrative"
    return ITEM_READERS.get(label)


def consequence(hand: str, given: str) -> str:
    """What a page labelled ``given`` instead of ``hand`` costs its reading.

    Called on a WRONG page only (``given != hand``); see
    :data:`CONSEQUENCES`.
    """
    want, got = reader_for(hand), reader_for(given)
    if want == got:
        return "same reader" if want else "same (neither read)"
    if BOUND_READER in (want, got):
        if want and got:
            return "bound/parent swap"
        return ("bound page not read" if want == BOUND_READER
                else "page read as bound")
    if want is None:
        return "read, should not be"
    if got is None:
        return "should be read, is not"
    return "different reader"


def is_harmful(hand: str, given: str) -> bool:
    """Does ``given`` send this page to another reader than ``hand`` does,
    or to none, or to one where it should go to none?"""
    return given != hand and consequence(hand, given) not in HARMLESS


def hand_sections(hand: Dict[int, str]) -> List[Tuple[int, int, str]]:
    """``(first, last, label)`` for every run of one hand label, in order;
    a page missing from ``hand`` ends a run."""
    out: List[Tuple[int, int, str]] = []
    for page in sorted(hand):
        label = hand[page]
        if out and out[-1][2] == label and out[-1][1] == page - 1:
            out[-1] = (out[-1][0], page, label)
        else:
            out.append((page, page, label))
    return out


def miss_structure(hand: Dict[int, str], given: Dict[int, str]
                   ) -> Dict[int, str]:
    """``page -> one of`` :data:`STRUCTURE` for every wrong page.

    A page missing from ``given`` is ``other``, as everywhere in scoring.
    """
    out: Dict[int, str] = {}
    for first, last, label in hand_sections(hand):
        pages = list(range(first, last + 1))
        wrong = [p for p in pages if given.get(p, "other") != label]
        if not wrong:
            continue
        if len(pages) == 1:
            out[first] = "one-page section"
            continue
        if len(wrong) == len(pages):
            for page in pages:
                out[page] = "whole section wrong"
            continue
        block: List[int] = []
        for page in wrong + [None]:
            if block and (page is None or page != block[-1] + 1):
                edge = block[0] == first or block[-1] == last
                size = "1 page" if len(block) == 1 else "2+ pages"
                for p in block:
                    out[p] = f"{'edge' if edge else 'inside'}, {size}"
                block = []
            if page is not None:
                block.append(page)
    return out


def harm_score(hand: Dict[int, str], given: Dict[int, str]
               ) -> Dict[str, Any]:
    """Strict accuracy, the wrong pages by consequence and by structure,
    and how many of them change a reader, for one report or many joined.

    ``hand`` and ``given`` are ``page -> label``; only hand-labelled pages
    are scored, and a page ``given`` has no label for is ``other``.
    """
    by_consequence: Counter = Counter()
    by_structure: Counter = Counter()
    harmful_structure: Counter = Counter()
    moves: Counter = Counter()
    structure = miss_structure(hand, given)
    hits = harmful = 0
    for page, want in hand.items():
        got = given.get(page, "other")
        if got == want:
            hits += 1
            continue
        what = consequence(want, got)
        by_consequence[what] += 1
        by_structure[structure[page]] += 1
        if what not in HARMLESS:
            harmful += 1
            harmful_structure[structure[page]] += 1
            moves[f"{want} -> {got}"] += 1
    n = len(hand)
    return {
        "pages": n, "right": hits,
        "strict": (hits / n) if n else float("nan"),
        "wrong": n - hits, "harmful": harmful,
        "by_consequence": dict(by_consequence),
        "by_structure": dict(by_structure),
        "harmful_by_structure": dict(harmful_structure),
        "harmful_moves": dict(moves),
    }
