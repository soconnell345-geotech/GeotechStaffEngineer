"""One page, three cheap voters, and the rule that settles them.

WHY THIS MODULE EXISTS AT ALL. Until 5.24.0 the page labels the whole ingest
hangs on came from ONE voter -- planlens' rules -- and an expensive agent
loop then checked all of them. The corpus run of 2026-09-20 (ledger runs 7
and 10) says that is the wrong shape twice over:

* the rules and the vision pass are COMPLEMENTARY by label class rather
  than one being better. The rules own ``appended_report`` (494 in-sample
  pages the vision pass never once emits), ``other`` (216, the same) and
  ``calculation`` (1,009, of which vision misses 41 %); vision owns
  ``plan``, ``profile``, ``photos``, ``cover``, ``toc`` and ``figure``
  recall outright. Overall: rules 0.908 in sample and 0.767 honest blind,
  vision (sheet mode, about $0.05 a report) 0.655 and 0.867;
* the review -- rules plus review is 0.916 / 0.850 at about $0.45 a report
  -- breaks nearly as many labels as it fixes on the reports its prompt was
  tuned against, and earns its money on the ones it has never seen.

So the expensive look belongs where the cheap voters DISAGREE, which is the
owner's own standing direction of 2026-09-18: *"if multiple methods say
different things, it could trigger an extra review ... would be good to have
confidence values associated with the classifications and data
extractions."*

WHAT IS HERE. The arithmetic, over plain values, with no model call, no
document opened and no import of anything that opens one. Two callers share
it and neither owns it:

* :mod:`report_ingest.graph`, which labels a page in production and then
  sends the splits to :func:`report_ingest.label_review.review_labels`;
* :mod:`report_ingest.vote` and the ``vote`` stage of
  :mod:`report_ingest.cluster_scoring`, which score the same policies
  against hand labels.

Two copies of a policy would drift, and the second copy's numbers would be
the ones nobody checked.

THE THIRD VOTER IS NOT LIKE THE OTHER TWO. The rules and the vision pass
each name a page label. The log-template recogniser
(:mod:`report_ingest.log_templates`) names a FORM -- who printed it -- and a
form is not a label: a fingerprint says "this page came off that firm's
boring-log template", never which of the four exploration labels the page
should carry. So the template votes for the CLASS: a page a fingerprint
claims is an exploration log, and the rule below is the whole of what that
buys. It is a no-op wherever no fingerprint file is in force, which is the
default state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "POLICIES", "DEFAULT_POLICY", "STRUCTURAL_RULES_WIN", "LOG_LABELS",
    "MISSING", "Voter", "PageChoice", "combine", "split_pages",
    "neighbourhood", "load_trust_table",
]

#: Every way of settling a disagreement this module knows.
#:
#: ``rules``
#:     the rules' label, always. No vote is taken and nothing else is
#:     consulted: this is what the ingest did before the vote, kept so a
#:     run can reproduce it exactly.
#: ``structural``
#:     vision everywhere except the labels the rules demonstrably own
#:     (:data:`STRUCTURAL_RULES_WIN`). It learns nothing -- it is the
#:     ledger's reading written down as a rule -- which is what makes it the
#:     policy to beat and the sensible default until a cluster run says
#:     otherwise.
#: ``confidence``
#:     whichever of the two said so more confidently, ties to the rules.
#: ``trust``
#:     whichever voter a per-label trust table, learned on the in-sample
#:     reports alone, favours for the class the RULES put the page in.
#:     Needs that table; see :func:`load_trust_table`.
POLICIES: Tuple[str, ...] = ("trust", "structural", "confidence", "rules")

#: What the graph uses when nothing says otherwise.
DEFAULT_POLICY = "structural"

#: The labels the rules win on, for the ``structural`` policy. A page is
#: ``appended_report`` or ``other`` because of where it SITS in the document,
#: which a picture of it cannot show; ``calculation`` and ``lab_test`` are
#: printouts and forms that read as tables or narrative when seen alone.
STRUCTURAL_RULES_WIN: Tuple[str, ...] = ("appended_report", "other",
                                         "calculation", "lab_test")

#: The four labels that are one exploration's own record. A log-template
#: match says the page is one of these and cannot say which.
LOG_LABELS: Tuple[str, ...] = ("boring_log", "test_pit_log", "cpt_log",
                               "dcp_log")

#: What a voter's label is taken to be where it has none -- the same rule
#: the label scorer uses, because a non-answer is scored, not excused.
MISSING = "other"


@dataclass(frozen=True)
class Voter:
    """What one voter said about one page.

    ``label`` is a page label for the rules and the vision pass, and EMPTY
    for the template voter, which recognises the form rather than the label:
    its claim is ``family``, and what that claim is worth is the template
    rule in :func:`combine`.
    """

    name: str
    label: str = ""
    confidence: Optional[float] = None
    family: str = ""

    @property
    def spoke(self) -> bool:
        """Did this voter give a page label at all?"""
        return bool(self.label)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"voter": self.name, "label": self.label,
                               "confidence": _round(self.confidence)}
        if self.family:
            out["family"] = self.family
        return out


@dataclass(frozen=True)
class PageChoice:
    """One page's label, and every voter that had a view on it."""

    page: int
    label: str
    confidence: float
    agreed: bool
    policy: str
    voters: Tuple[Voter, ...] = ()
    #: Why the label is what it is, in a few words, for a reader of the run
    #: file. Never a page's own words -- a policy name and a label only.
    why: str = ""

    def voter(self, name: str) -> Optional[Voter]:
        for row in self.voters:
            if row.name == name:
                return row
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "label": self.label,
            "confidence": _round(self.confidence),
            "agreed": self.agreed,
            "policy": self.policy,
            "why": self.why,
            "voters": [v.to_dict() for v in self.voters],
        }


def _round(value: Optional[float]) -> float:
    return round(float(value or 0.0), 3)


def _clip(value: Optional[float]) -> float:
    """A confidence into 0 to 1; a missing one is 0.0, which is honest."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:                          # NaN
        return 0.0
    return max(0.0, min(1.0, number))


# ---------------------------------------------------------------------------
# the policies
# ---------------------------------------------------------------------------

def _by_policy(rules: Voter, vision: Voter, policy: str,
               trust_table: Optional[Mapping[str, Mapping[str, Any]]]
               ) -> Tuple[str, str]:
    """``(label, why)`` from the two labelling voters under one policy."""
    if policy == "rules" or not vision.spoke:
        return rules.label, "rules"
    if not rules.spoke:
        return vision.label, "vision"
    if policy == "structural":
        if rules.label in STRUCTURAL_RULES_WIN:
            return rules.label, f"structural: the rules own {rules.label}"
        return vision.label, "structural: vision"
    if policy == "confidence":
        if _clip(vision.confidence) > _clip(rules.confidence):
            return vision.label, "confidence: vision said so more firmly"
        return rules.label, "confidence: the rules said so at least as firmly"
    # trust
    if rules.label == vision.label:
        return rules.label, "trust: the voters agreed"
    cell = (trust_table or {}).get(rules.label)
    if cell and str(cell.get("winner")) == "rules":
        return rules.label, f"trust: the table believes the rules on " \
                            f"{rules.label}"
    return vision.label, "trust: vision, which the table favours here"


def combine(rules: Any, vision: Any = None, template: Any = None, *,
            policy: str = DEFAULT_POLICY,
            trust_table: Optional[Mapping[str, Mapping[str, Any]]] = None,
            page: int = 0) -> PageChoice:
    """Settle one page under one policy, and say whether the voters agreed.

    ``rules``, ``vision`` and ``template`` are :class:`Voter` instances, or
    ``None`` for a voter that did not run. A bare string is accepted for the
    first two and read as a label with no confidence, which is what makes a
    hand-made table in a test short.

    THE TEMPLATE RULE, and it is the whole of what the third voter does.
    A fingerprint match means the page is one exploration's own record. So
    where a template claims the page and the policy's own answer is NOT one
    of :data:`LOG_LABELS`, a log label that either of the other two voters
    gave WINS -- the template breaks the tie towards the class it recognised.
    Where neither of them offered a log label there is nothing to break the
    tie towards: the policy's answer stands and ``agreed`` is False, which
    is exactly the page a second look is for. ``policy="rules"`` reproduces
    the old behaviour and is not subject to the rule.

    ``agreed`` is True when every voter that had a view concurs with the
    chosen label -- the template concurring when the chosen label is a log
    label. A page with only one voter agrees with itself, which is right:
    there is nothing there to disagree.

    ``confidence`` is the highest confidence among the voters that gave the
    chosen label, and 0.0 where none of them carried one. Agreement never
    lowers a voter's own certainty, and a record should not read as less
    sure because a second voter was less sure than the first.
    """
    if policy not in POLICIES:
        raise ValueError(
            f"unknown label policy {policy!r}; the policies are "
            f"{list(POLICIES)}")
    rules_v = _as_voter(rules, "rules")
    vision_v = _as_voter(vision, "vision")
    template_v = _as_voter(template, "template")

    label, why = _by_policy(rules_v, vision_v, policy, trust_table)
    claimed = bool(template_v.family) and policy != "rules"
    if claimed and label not in LOG_LABELS:
        for other in (rules_v, vision_v):
            if other.label in LOG_LABELS:
                label = other.label
                why = (f"template {template_v.family}: the page came off a "
                       f"log form, so {other.name}'s {other.label} wins")
                break

    voters = tuple(v for v in (rules_v, vision_v, template_v)
                   if v.spoke or v.family)
    spoken = [v for v in voters if v.spoke]
    agreed = all(v.label == label for v in spoken)
    if claimed and label not in LOG_LABELS:
        agreed = False
    confidence = max([_clip(v.confidence) for v in spoken
                      if v.label == label] or [0.0])
    return PageChoice(page=int(page), label=label or MISSING,
                      confidence=confidence, agreed=agreed, policy=policy,
                      voters=voters, why=why)


def _as_voter(value: Any, name: str) -> Voter:
    if value is None:
        return Voter(name=name)
    if isinstance(value, Voter):
        return value if value.name == name else Voter(
            name=name, label=value.label, confidence=value.confidence,
            family=value.family)
    if isinstance(value, str):
        return Voter(name=name, label=value)
    raise TypeError(
        f"a voter is a Voter, a label string or None, not "
        f"{type(value).__name__}")


# ---------------------------------------------------------------------------
# what the graph asks next
# ---------------------------------------------------------------------------

def split_pages(choices: Sequence[PageChoice]) -> list:
    """The pages the voters did not agree on, in page order."""
    return sorted(int(c.page) for c in choices if not c.agreed)


def neighbourhood(pages: Sequence[int], n_pages: int, *,
                  either_side: int = 2) -> list:
    """``pages`` plus their neighbours, clipped to the document, in order.

    A page is decided by what it sits between at least as often as by what
    it prints -- a form page inside a laboratory appendix, a plan two pages
    after the divider that names it -- so a review asked about one page
    alone is being asked the hard version of the question. Two either side
    is what a reviewer needs to see the run the page belongs to.
    """
    want: set = set()
    for page in pages:
        page = int(page)
        for other in range(page - either_side, page + either_side + 1):
            if 0 <= other < int(n_pages):
                want.add(other)
    return sorted(want)


def load_trust_table(source: Any) -> Dict[str, Dict[str, Any]]:
    """A trust table out of a dict or a JSON file the ``vote`` stage wrote.

    The stage saves ``{"learned_on": [...], "table": {...}}`` so a reader of
    the file can see which reports it came from; a bare table is accepted
    too, and so is ``None``, which is an empty table rather than an error --
    the ``trust`` policy then falls back and says so.
    """
    if not source:
        return {}
    blob: Any = source
    if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
        import json
        import os
        path = os.fspath(source) if hasattr(source, "__fspath__") else source
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                blob = json.load(handle)
        except (OSError, ValueError):
            return {}
    if isinstance(blob, dict) and "table" in blob:
        blob = blob.get("table") or {}
    if not isinstance(blob, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for label, cell in blob.items():
        if isinstance(cell, dict) and cell.get("winner"):
            out[str(label)] = dict(cell)
    return out
