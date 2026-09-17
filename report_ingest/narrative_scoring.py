"""Scoring the narrative reading, question by question, against a hand answer.

THE TRUTH FILE is one JSON per report: ``{"general": {...}, "natural_hazards":
{...}, "notes": "..."}``, the owner's field names, and **null where the report
does not say**. That null is the whole difficulty of scoring this pass. Most
reports answer most of the general list and only part of the hazards list, so
a reader that answers nothing agrees with the truth on a great many fields,
and a reader that answers everything is right about some of them by accident.

SO THREE NUMBERS ARE REPORTED AND NONE OF THEM IS "ACCURACY".

``recall``     of the questions the report DOES answer, how many came back
               right. This is the one that says whether the reader reads.
``precision``  of the answers the reader GAVE, how many were right. This is
               the one that catches a reader inventing plausible facts.
``agreement``  over every field, how often the reader and the hand agree,
               including both saying nothing. Reported because it is the
               figure a naive scorer would print, and it flatters.

HOW EACH KIND OF FIELD IS JUDGED.

``enum``     exact, after folding case and punctuation. The vocabularies are
             fixed and an answer outside one was already refused by the
             reader, so anything here is a real disagreement.
``int``      exact. A count is a count.
``string``   a normalised token match: :mod:`rapidfuzz`'s partial ratio at 85
             or better, OR a shared proper noun -- a capitalised word or an
             identifier the two answers have in common. "Soil & Rock
             Consulting Engineers" and "Soil and Rock Consulting Engineers,
             Inc." are one firm and a scorer that said otherwise would be
             measuring punctuation. ``siteClass`` and ``reportDate`` also
             pass on their NORMALISED forms, because "Site Class D" and "D"
             are the same answer and so are "14 March 2026" and "2026-03-14".
``list``     set overlap. The items' own precision and recall are accumulated
             across every list field, and the field counts as right when the
             Jaccard overlap reaches :data:`LIST_HIT_JACCARD`.
``summary``  presence and length ONLY. Whether a summary is a good summary is
             a person's judgement, and a scorer that pretended to make it
             would be scoring its own opinion. What can be checked is that one
             was written where the report supports one, and that it is inside
             its word limit.

Everything here is shared by the development scorecard
(``module_work/report_ingest_harness/measure_wp4_narrative.py``) and the
cluster run (:mod:`report_ingest.cluster_scoring`), because two copies of this
arithmetic would drift and the second copy's numbers would be the ones nobody
checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import (
    GENERAL_FIELDS, NATURAL_HAZARD_FIELDS, SUMMARY_FIELDS,
    SUMMARY_WORD_LIMITS,
)

__all__ = [
    "FIELD_KINDS", "KINDS", "LIST_HIT_JACCARD", "FUZZY_RATIO",
    "Score", "FieldResult", "NarrativeScore", "kind_of", "same_value",
    "score_narrative", "score_one_report", "narrative_pages_of",
]

#: A string answer matches when rapidfuzz's partial ratio reaches this.
FUZZY_RATIO = 85
#: A list answer counts as right when the overlap reaches this.
LIST_HIT_JACCARD = 0.6
#: The kinds a field is judged as, in the order a scorecard prints them.
KINDS: Tuple[str, ...] = ("enum", "int", "string", "list", "summary")

#: Which fields are answered from a fixed vocabulary.
_ENUM_FIELDS = frozenset((
    "documentType", "propertyType", "projectPhase", "outsideProject",
    "liquefactionPotential", "geophysicalTestingMention", "soilCorrosion",
    "siteResponseMention", "hazardAnalysisMention",
))
#: Which are whole numbers.
_INT_FIELDS = frozenset((
    "boringCount", "testPitCount", "cptCount", "tableCount", "figureCount",
    "previousInvestigationCount", "structureCount",
))
#: Which are lists.
_LIST_FIELDS = frozenset((
    "structureList", "boringDictionary", "testPitDictionary",
    "recommendedFoundations", "bearingCapacity", "earthHazardsExposed",
))

#: ``field -> kind``, for every field of both schemas.
FIELD_KINDS: Dict[str, str] = {}
for _name in GENERAL_FIELDS + NATURAL_HAZARD_FIELDS:
    if _name in SUMMARY_FIELDS:
        FIELD_KINDS[_name] = "summary"
    elif _name in _ENUM_FIELDS:
        FIELD_KINDS[_name] = "enum"
    elif _name in _INT_FIELDS:
        FIELD_KINDS[_name] = "int"
    elif _name in _LIST_FIELDS:
        FIELD_KINDS[_name] = "list"
    else:
        FIELD_KINDS[_name] = "string"
del _name


def kind_of(field_name: str) -> str:
    """How this field is judged."""
    return FIELD_KINDS.get(field_name, "string")


# ---------------------------------------------------------------------------
# comparing two answers
# ---------------------------------------------------------------------------

def _fold(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _tokens(value: Any) -> List[str]:
    return [token for token in _fold(value).split() if token]


#: A word that is proper-noun-ish: a capitalised word of four letters or more,
#: or an identifier with a digit in it (a project number, a boring name).
_RE_PROPER = re.compile(r"\b(?:[A-Z][A-Za-z]{3,}|[A-Za-z]*\d[\w\-./]*)\b")


def _proper(value: Any) -> set:
    return {match.lower() for match in _RE_PROPER.findall(str(value or ""))}


def _partial_ratio(left: str, right: str) -> float:
    """rapidfuzz's partial ratio, or a containment test without it.

    rapidfuzz arrives with planlens, so it is normally here; the fallback is
    deliberately blunt rather than clever, so that a missing package shows up
    as a different number rather than as a silently different definition of
    "the same answer".
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:                          # pragma: no cover - fallback
        if not left or not right:
            return 0.0
        return 100.0 if (left in right or right in left) else 0.0
    return float(fuzz.partial_ratio(left, right))


def _strings_match(field_name: str, hand: Any, got: Any) -> bool:
    left, right = _fold(hand), _fold(got)
    if not left or not right:
        return False
    if left == right:
        return True
    if field_name == "siteClass":
        from report_ingest.narrative_reader import normalise_site_class
        one, two = normalise_site_class(hand), normalise_site_class(got)
        if one and two:
            return one == two
    if field_name == "reportDate":
        from report_ingest.narrative_reader import iso_date
        one, two = iso_date(hand), iso_date(got)
        if one and two:
            return one == two
    if field_name in ("asceSevenVersion", "seismicCodeUsed"):
        from report_ingest.narrative_reader import normalise_asce_version
        one, two = normalise_asce_version(hand), normalise_asce_version(got)
        if one and two:
            return one == two
    if _partial_ratio(left, right) >= FUZZY_RATIO:
        return True
    shared = _proper(hand) & _proper(got)
    return bool(shared)


def _list_overlap(hand: Any, got: Any) -> Tuple[float, int, int, int]:
    """``(jaccard, hits, in the answer, in the truth)`` over folded items."""
    left = {_fold(item) for item in (hand or []) if _fold(item)}
    right = {_fold(item) for item in (got or []) if _fold(item)}
    if not left and not right:
        return 1.0, 0, 0, 0
    hits = len(left & right)
    union = len(left | right)
    return (hits / union if union else 1.0), hits, len(right), len(left)


def same_value(field_name: str, hand: Any, got: Any) -> bool:
    """Do these two answers to one question agree?"""
    kind = kind_of(field_name)
    if hand is None or got is None:
        return hand is None and got is None
    if kind == "enum":
        return _fold(hand) == _fold(got)
    if kind == "int":
        try:
            return int(hand) == int(got)
        except (TypeError, ValueError):
            return False
    if kind == "list":
        jaccard, *_ = _list_overlap(hand, got)
        return jaccard >= LIST_HIT_JACCARD
    if kind == "summary":
        return bool(str(got or "").strip())
    return _strings_match(field_name, hand, got)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------

@dataclass
class Score:
    """A tally of right out of asked, with what went wrong."""

    found: int = 0
    total: int = 0
    misses: List[str] = field(default_factory=list)

    def add(self, ok: bool, what: str = "") -> None:
        self.total += 1
        if ok:
            self.found += 1
        elif what:
            self.misses.append(what)

    @property
    def rate(self) -> Optional[float]:
        return self.found / self.total if self.total else None

    def __iadd__(self, other: "Score") -> "Score":
        self.found += other.found
        self.total += other.total
        self.misses.extend(other.misses)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"found": self.found, "total": self.total, "rate": self.rate}


@dataclass
class FieldResult:
    """One question, scored."""

    field: str
    kind: str
    hand: Any
    got: Any
    ok: bool
    #: What went wrong, in a word: ``missed`` (the report answers, the reader
    #: did not), ``invented`` (the reader answered, the report does not),
    #: ``wrong`` (both answered, differently), ``right``, ``both_silent``.
    verdict: str = ""
    jaccard: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"field": self.field, "kind": self.kind,
               "verdict": self.verdict, "ok": self.ok,
               "hand": _short(self.hand), "got": _short(self.got)}
        if self.jaccard is not None:
            out["jaccard"] = round(self.jaccard, 3)
        return out


def _short(value: Any, n: int = 120) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = " ".join(str(value).split()) if not isinstance(value, list) \
        else "; ".join(" ".join(str(v).split()) for v in value)
    return text if len(text) <= n else text[:n - 1] + "…"


@dataclass
class NarrativeScore:
    """One report's narrative reading, scored field by field."""

    report: str
    fields: List[FieldResult] = field(default_factory=list)
    #: ``kind -> Score`` over the fields the HAND answered (recall).
    by_kind: Dict[str, Score] = field(default_factory=dict)
    recall: Score = field(default_factory=Score)
    precision: Score = field(default_factory=Score)
    agreement: Score = field(default_factory=Score)
    #: The items inside every list field, pooled.
    list_items: Score = field(default_factory=Score)
    list_item_recall: Score = field(default_factory=Score)
    #: The four prose summaries: written where the report supports one, and
    #: inside the word limit.
    summaries_present: Score = field(default_factory=Score)
    summaries_within_limit: Score = field(default_factory=Score)
    unresolved: int = 0
    model_calls: int = 0
    cost: Dict[str, Any] = field(default_factory=dict)
    pages: List[int] = field(default_factory=list)
    error: Optional[str] = None

    def kind(self, name: str) -> Score:
        return self.by_kind.setdefault(name, Score())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report": self.report,
            "recall": self.recall.to_dict(),
            "precision": self.precision.to_dict(),
            "agreement": self.agreement.to_dict(),
            "by_kind": {name: self.kind(name).to_dict()
                        for name in KINDS if self.by_kind.get(name)},
            "list_items": {"precision": self.list_items.to_dict(),
                           "recall": self.list_item_recall.to_dict()},
            "summaries": {"present": self.summaries_present.to_dict(),
                          "within_limit":
                              self.summaries_within_limit.to_dict()},
            "fields": [row.to_dict() for row in self.fields],
            "unresolved": self.unresolved,
            "model_calls": self.model_calls,
            "cost": dict(self.cost),
            "pages": list(self.pages),
            "error": self.error,
        }


def _value(blob: Dict[str, Any], name: str) -> Any:
    value = blob.get(name)
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, list) and not value:
        return None
    return value


def score_narrative(truth: Dict[str, Any], general: Any, hazards: Any,
                    report: str = "") -> NarrativeScore:
    """Score one reading against one hand answer, question by question."""
    out = NarrativeScore(report=report or str(truth.get("id") or ""))
    sections = (("general", GENERAL_FIELDS, general),
                ("natural_hazards", NATURAL_HAZARD_FIELDS, hazards))
    for section, names, answers in sections:
        hand_section = truth.get(section) or {}
        for name in names:
            hand = _value(hand_section, name)
            got = getattr(answers, name, None)
            if isinstance(got, list) and not got:
                got = None
            kind = kind_of(name)

            if kind == "summary":
                _score_summary(out, name, hand, got)
                continue

            ok = same_value(name, hand, got)
            jaccard = None
            if kind == "list" and hand is not None and got is not None:
                # The ITEMS' own precision and recall, pooled across every
                # list field: a boring dictionary that names eight of ten
                # holes is not the same failure as one that names two.
                jaccard, hits, given, wanted = _list_overlap(hand, got)
                out.list_items.found += hits
                out.list_items.total += given
                out.list_item_recall.found += hits
                out.list_item_recall.total += wanted

            verdict = _verdict(hand, got, ok)
            out.fields.append(FieldResult(field=name, kind=kind, hand=hand,
                                          got=got, ok=ok, verdict=verdict,
                                          jaccard=jaccard))
            out.agreement.add(ok, f"{name}: {verdict}")
            if hand is not None:
                out.recall.add(ok, f"{name}: {verdict}")
                out.kind(kind).add(ok, f"{name}: {verdict}")
            if got is not None:
                out.precision.add(ok, f"{name}: {verdict}")
    return out


def _score_summary(out: NarrativeScore, name: str, hand: Any,
                   got: Any) -> None:
    """Presence and length; never the words themselves."""
    text = " ".join(str(got or "").split())
    present = bool(text)
    limit = SUMMARY_WORD_LIMITS.get(name, 100)
    within = present and len(text.split()) <= limit
    # A summary is expected where the hand wrote one; where the hand left it
    # null the report gave nothing to summarise and a written one is not
    # counted against the reader either way.
    if hand is not None:
        out.summaries_present.add(present, f"{name}: not written")
    if present:
        out.summaries_within_limit.add(
            within, f"{name}: {len(text.split())} words, limit {limit}")
    out.fields.append(FieldResult(
        field=name, kind="summary", hand=_short(hand), got=_short(got),
        ok=within if present else (hand is None),
        verdict=("right" if within else
                 ("too_long" if present else
                  ("both_silent" if hand is None else "missed")))))


def _verdict(hand: Any, got: Any, ok: bool) -> str:
    if hand is None and got is None:
        return "both_silent"
    if hand is None:
        return "invented"
    if got is None:
        return "missed"
    return "right" if ok else "wrong"


# ---------------------------------------------------------------------------
# running the reader over one report
# ---------------------------------------------------------------------------

def narrative_pages_of(doc: Any) -> Tuple[List[int], List[int]]:
    """``(the narrative pages, the main body pages)`` of an open document.

    The narrative is every page of every ``narrative`` work item -- a report
    bound inside another is its own item and is deliberately not one of them.
    The body is the narrative plus the figure, plan, profile and contents
    pages before the appendices, which is where the captions Python counts
    live.
    """
    from planlens.document.roles import roles_and_items

    roles, items = roles_and_items(doc)
    narrative = sorted(int(page) for item in items
                       if item.kind == "narrative" for page in item.pages)
    body_roles = ("narrative", "figure", "plan", "profile", "toc")
    body = sorted(int(r.page) for r in roles if r.role in body_roles)
    return narrative, body


def score_one_report(truth: Dict[str, Any], doc: Any, engine: Any, *,
                     budget: int = 8, report: str = "") -> NarrativeScore:
    """Read one report's narrative and score it against the hand answer.

    A report whose narrative cannot be found comes back as a score with an
    ``error`` rather than an exception: the cluster run scores what it can and
    says what it could not.
    """
    from report_ingest.narrative_reader import read_narrative

    report = report or str(truth.get("id") or "")
    pages, body = narrative_pages_of(doc)
    if not pages:
        out = NarrativeScore(report=report,
                             error="no narrative work item in this document")
        return out
    try:
        result = read_narrative(doc, pages, engine, budget=budget,
                                body_pages=body, report_id=report)
    except Exception as exc:                     # score what can be scored
        return NarrativeScore(report=report, pages=pages,
                              error=f"{type(exc).__name__}: {exc}")
    out = score_narrative(truth, result.general, result.natural_hazards,
                          report=report)
    out.pages = pages
    out.unresolved = len(result.unresolved)
    out.model_calls = result.model_calls
    out.cost = dict(result.cost)
    return out
