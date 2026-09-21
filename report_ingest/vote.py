"""Three voters on one page, and what their disagreement is worth.

WHY. The owner's standing direction, 2026-09-18: *"even if something scores
worse, it could still be useful. Something like 'wisdom of the crowd' or
random forests; if multiple methods say different things, it could trigger an
extra review or something. Would be good to have confidence values associated
with the classifications and data extractions."*

The corpus run of 2026-09-20 (ledger runs 7 and 10) says the two cheap
voters are COMPLEMENTARY by label class rather than one being better:

* the RULES own the structural labels -- ``appended_report`` (494 in-sample
  pages, which vision never once emits), ``other`` (216, the same), and
  ``calculation`` (1,009, of which vision misses 41 %);
* VISION owns the visual ones -- ``plan``, ``profile``, ``photos``,
  ``cover``, ``toc`` and ``figure`` recall, where it beats the rules
  outright;
* overall: rules 0.908 in sample and 0.767 honest blind, rules plus the
  label review 0.916 / 0.850, vision (sheet mode, $0.05 a report) 0.655 /
  0.867.

So the question is not which to keep. It is what a page they disagree on is
worth looking at, and which of them to believe where they split.

WHAT IS HERE. The arithmetic only, over plain dicts, with NO model call and
no document opened -- so it runs on the run files already on disk, and the
cluster stage and the development harness share one copy of it.

:func:`agreement` is the confidence claim: how often the two agree, and how
right they are when they do against when they do not. :func:`trust_table`
learns, from the IN-SAMPLE reports alone, which voter is right more often
per label class when they split. :func:`combine` applies one of three
policies to one page. :func:`required_review_accuracy` says how good a
targeted review of the disagreements would have to be to carry the whole set
over the gate.

NOTHING HERE LEARNS ON THE BLIND SET. :func:`trust_table` is handed the
in-sample votes and nothing else; the caller is what keeps that promise, and
the scorecard says so in the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from report_ingest.label_vote import (
    MISSING, STRUCTURAL_RULES_WIN, Voter, combine as combine_page,
)

__all__ = [
    "PageVote", "POLICIES", "STRUCTURAL_RULES_WIN", "MISSING", "agreement",
    "combine", "trust_table", "required_review_accuracy", "policy_labels",
]

#: The three ways of settling a disagreement this module SCORES.
#:
#: ``trust``
#:     believe whichever voter the in-sample trust table favours for the
#:     RULES' label class, and vision where the table has nothing to say.
#: ``structural``
#:     vision everywhere except the four labels the rules demonstrably own.
#:     No learning at all: it is the ledger's reading written down as a rule,
#:     and it is the policy to beat.
#: ``confidence``
#:     believe whichever voter said so more confidently, ties to the rules.
#:
#: The arithmetic itself lives in :mod:`report_ingest.label_vote`, which the
#: production graph uses too, so a policy cannot mean one thing here and
#: another there. That module also knows a fourth, ``rules`` -- the rules'
#: label untouched -- which the graph offers so a run can reproduce what the
#: ingest did before the vote. It is not scored here: the ``rules`` column of
#: every table below already IS that answer.
POLICIES: Tuple[str, ...] = ("trust", "structural", "confidence")


@dataclass(frozen=True)
class PageVote:
    """One page, as every voter saw it.

    ``hand`` is ``None`` for a page nobody hand-labelled: it still counts
    towards the agreement rate -- which is the number a production run would
    have -- and towards nothing that is scored.
    """

    report: str
    page: int
    rules: str
    vision: str
    rules_confidence: Optional[float] = None
    vision_confidence: Optional[float] = None
    review: Optional[str] = None
    hand: Optional[str] = None
    alternates: Tuple[str, ...] = ()

    @property
    def agree(self) -> bool:
        return self.rules == self.vision

    @property
    def scored(self) -> bool:
        return self.hand is not None

    def right(self, label: Optional[str]) -> bool:
        """Is ``label`` the hand label? Strictly, alternates aside."""
        return self.hand is not None and label == self.hand

    def to_row(self, chosen: Dict[str, str]) -> Dict[str, Any]:
        """One line of the QA list: labels, numbers and nothing else."""
        row: Dict[str, Any] = {
            "page": self.page,
            "rules": self.rules,
            "vision": self.vision,
            "review": self.review,
            "hand": self.hand,
        }
        if self.rules_confidence is not None:
            row["rules_confidence"] = round(float(self.rules_confidence), 3)
        if self.vision_confidence is not None:
            row["vision_confidence"] = round(float(self.vision_confidence), 3)
        row["chosen"] = dict(chosen)
        return row


# ---------------------------------------------------------------------------
# 1. agreement, and what it is worth
# ---------------------------------------------------------------------------

def agreement(votes: Iterable[PageVote]) -> Dict[str, Any]:
    """How often the rules and vision agree, and how right that makes them.

    The agreement rate is over EVERY page both voters covered, hand-labelled
    or not, because that is the number a production run can compute for
    itself. The accuracies beside it are over the hand-labelled pages alone.
    """
    rows = list(votes)
    agreed = [v for v in rows if v.agree]
    scored = [v for v in rows if v.scored]
    agreed_scored = [v for v in scored if v.agree]
    split = [v for v in scored if not v.agree]
    agreed_right = sum(1 for v in agreed_scored if v.right(v.rules))
    rules_right = sum(1 for v in split if v.right(v.rules))
    vision_right = sum(1 for v in split if v.right(v.vision))
    either = sum(1 for v in split
                 if v.right(v.rules) or v.right(v.vision))
    return {
        "pages": len(rows),
        "agree": len(agreed),
        "agreement": _fraction(len(agreed), len(rows)),
        "scored": len(scored),
        "agreed": {
            "pages": len(agreed_scored),
            "correct": agreed_right,
            "accuracy": _fraction(agreed_right, len(agreed_scored)),
        },
        "disagreed": {
            "pages": len(split),
            "fraction": _fraction(len(split), len(scored)),
            "rules_correct": rules_right,
            "vision_correct": vision_right,
            "either_correct": either,
            "neither_correct": len(split) - either,
            "rules_accuracy": _fraction(rules_right, len(split)),
            "vision_accuracy": _fraction(vision_right, len(split)),
            "ceiling": _fraction(either, len(split)),
        },
    }


def _fraction(part: int, whole: int) -> Optional[float]:
    """``part / whole``, or ``None`` where there was nothing to divide.

    ``None`` rather than 0.0: a rate over no pages is unmeasured, and an
    unmeasured rate must not read as a failing one.
    """
    return round(part / whole, 4) if whole else None


# ---------------------------------------------------------------------------
# 2. the per-label trust table, learned in sample
# ---------------------------------------------------------------------------

def trust_table(votes: Iterable[PageVote]) -> Dict[str, Dict[str, Any]]:
    """Which voter to believe per label class, from disagreements alone.

    Keyed by the RULES' label, because that is what a production run has
    before it knows the answer: the rules label every page, and the question
    at a disagreement is whether to let vision overrule the class the rules
    put the page in.

    Hand it the IN-SAMPLE votes and nothing else. A tie goes to vision,
    which is also what an unseen class gets, so ``winner == "rules"`` always
    means the rules were strictly better on the pages that were looked at.
    """
    table: Dict[str, Dict[str, Any]] = {}
    for vote in votes:
        if vote.agree or not vote.scored:
            continue
        cell = table.setdefault(vote.rules, {"pages": 0, "rules": 0,
                                             "vision": 0, "winner": "vision"})
        cell["pages"] += 1
        cell["rules"] += int(vote.right(vote.rules))
        cell["vision"] += int(vote.right(vote.vision))
    for cell in table.values():
        cell["winner"] = "rules" if cell["rules"] > cell["vision"] else "vision"
    return table


# ---------------------------------------------------------------------------
# 3. the three policies
# ---------------------------------------------------------------------------

def combine(vote: PageVote, policy: str,
            table: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The label one policy settles on for one page.

    ONE COPY OF THE ARITHMETIC. This is
    :func:`report_ingest.label_vote.combine` over a :class:`PageVote`, and
    the production graph calls the same function over the same voters -- so
    what ``structural`` scores here is what ``structural`` does there. This
    stage has no template voter: it runs off saved run files, and a
    fingerprint match is not saved in one.
    """
    if policy not in POLICIES:
        raise ValueError(
            f"unknown policy {policy!r}; the policies are {list(POLICIES)}")
    return combine_page(
        Voter("rules", vote.rules, vote.rules_confidence),
        Voter("vision", vote.vision, vote.vision_confidence),
        policy=policy, trust_table=table, page=vote.page).label


def policy_labels(vote: PageVote,
                  table: Optional[Dict[str, Dict[str, Any]]] = None
                  ) -> Dict[str, str]:
    """What all three policies say about one page."""
    return {name: combine(vote, name, table) for name in POLICIES}


# ---------------------------------------------------------------------------
# 4. what a targeted review of the disagreements would have to manage
# ---------------------------------------------------------------------------

def required_review_accuracy(agreed_pages: int, agreed_correct: int,
                             disagreed_pages: int,
                             gate: float) -> Optional[float]:
    """The accuracy a review of the disagreements alone would need.

    Everything the two voters agree on is left as it is; the pages they split
    on go to a third look. Solve for the accuracy that look needs so the
    whole set reaches ``gate``::

        (agreed_correct + a * disagreed) / (agreed + disagreed) = gate

    ``None`` where there is nothing to review or nothing to score. A value
    above 1.0 means the gate is out of reach even with a perfect review,
    which is a real answer and is printed as one.
    """
    total = agreed_pages + disagreed_pages
    if not total or not disagreed_pages:
        return None
    return round((gate * total - agreed_correct) / disagreed_pages, 4)


# ---------------------------------------------------------------------------
# building the votes
# ---------------------------------------------------------------------------

def build_votes(report: str, rules: Dict[int, str], vision: Dict[int, str], *,
                rules_confidence: Optional[Dict[int, float]] = None,
                vision_confidence: Optional[Dict[int, float]] = None,
                review: Optional[Dict[int, str]] = None,
                hand: Optional[Dict[int, str]] = None,
                alternates: Optional[Dict[int, Sequence[str]]] = None,
                pages: Optional[Sequence[int]] = None) -> List[PageVote]:
    """One :class:`PageVote` per page of one report.

    ``pages`` defaults to every page either voter has a label for, so a page
    the vision pass left unresolved is still a vote -- for ``other``, the
    same way the label scorer counts it -- and therefore still a
    disagreement. That is the honest reading: a page nothing could answer
    for is exactly the page that wants a second look.
    """
    rules_confidence = rules_confidence or {}
    vision_confidence = vision_confidence or {}
    review = review or {}
    hand = hand or {}
    alternates = alternates or {}
    if pages is None:
        pages = sorted(set(rules) | set(vision) | set(hand))
    out: List[PageVote] = []
    for page in pages:
        out.append(PageVote(
            report=report,
            page=int(page),
            rules=rules.get(page, MISSING),
            vision=vision.get(page, MISSING),
            rules_confidence=rules_confidence.get(page),
            vision_confidence=vision_confidence.get(page),
            review=review.get(page),
            hand=hand.get(page),
            alternates=tuple(alternates.get(page) or ()),
        ))
    return out


def vision_confidences(seen: Dict[str, Any]) -> Dict[int, float]:
    """``page -> confidence`` out of a saved vision run's ``detail`` list."""
    out: Dict[int, float] = {}
    for entry in (seen.get("detail") or []):
        try:
            page = int(entry["page"])
        except (KeyError, TypeError, ValueError):
            continue
        value = entry.get("confidence")
        if value is None:
            continue
        try:
            out[page] = float(value)
        except (TypeError, ValueError):
            continue
    return out
