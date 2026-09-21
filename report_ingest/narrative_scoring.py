"""Scoring the narrative reading, question by question, against a hand answer.

THE TRUTH FILE is one JSON per report: ``{"id", "source", "general": {...},
"natural_hazards": {...}, "_alternates": {...}, "_skip": [...], "notes"}``, the
owner's field names, and **null where the report does not say**.

``_alternates`` is the hand's fairness valve: other answers it will accept for
this report, keyed by field, always as a LIST. A report can say a thing in more
than one defensible way -- 14 borings or 18 counting the ones abandoned -- and
the hand is one reading of it, not the only one. A ``null`` among them means
"not stated is acceptable too"; a LIST among them is a whole alternative list
answer. ``_skip`` names the fields not to score at all for this report, for a
question the report genuinely does not settle. Neither is a thing the reader
ever sees.

That null is the whole difficulty of scoring this pass. Most
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
``list``     set overlap, and TWO KINDS of list. ``boringDictionary`` and
             ``testPitDictionary`` hold identifiers and are compared exactly
             (folded), because B-1 and B-12 are two holes however alike they
             look. The others hold prose -- the hand's ``bearingCapacity``
             runs to fifteen words an item -- and are compared item by item by
             the string rule, paired one to one. The items' own precision and
             recall are accumulated across every list field, and the field
             counts as right at :data:`LIST_HIT_JACCARD`.
``verdict``  the four questions answered "yes"/"no"/"mixed"/"unclear" and then
             a reason are scored on the VERDICT alone. Two readers who both
             find the geophysical survey will not word the reason the same
             way, and the question asked was whether the report mentions it.
``summary``  presence and length ONLY. Whether a summary is a good summary is
             a person's judgement, and a scorer that pretended to make it
             would be scoring its own opinion. What can be checked is that one
             was written where the report supports one, and that it is inside
             its word limit.

AND EVERY NUMBER IS REPORTED TWICE, STRICT AND LENIENT. Four of the questions
are answered with long free text -- the recommended foundations, the
structures, the bearing-pressure sentences, the profile -- and two people
writing the same report's answer never word them alike. The strict view is
what a machine consumer would get; the lenient view (:data:`LENIENT_FIELDS`,
:data:`LENIENT_RATIO`, a number and its unit in common) is what a reviewer
reading both answers would say. Neither is "the" score and a scorecard prints
them side by side, because the gap between them is itself the finding: a wide
gap means the reader found the thing and worded it differently, a narrow one
means it did not find it.

And every miss carries a KIND (:func:`miss_kind`): ``convention`` when the
field is one a house rule in :mod:`report_ingest.narrative_glossary` governs,
because that is the failure a better model does not fix, and otherwise the
plain verdict. :func:`dominant_miss` reduces a column of them to the one word
a per-question table prints in its "why" column.

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
    "LENIENT_FIELDS", "LENIENT_RATIO", "MISS_KINDS",
    "Score", "FieldResult", "NarrativeScore", "kind_of", "same_value",
    "score_narrative", "score_one_report", "narrative_pages_of",
    "alternates_for", "skipped_fields", "miss_kind", "dominant_miss",
]

#: A string answer matches when rapidfuzz's partial ratio reaches this.
FUZZY_RATIO = 85

#: THE LENIENT VIEW, and what it is for. Four of the questions are answered
#: with long free text -- a list of recommended foundations, the structures,
#: the bearing-pressure sentences, the soil profile in one string -- and two
#: readers of the same report word them differently every time. At the strict
#: ratio those four score around a third and the number says almost nothing
#: about whether the reader FOUND the recommendation; it says the two hands
#: do not write alike. So both views are computed and both are printed, and
#: neither is "the" score: strict is what a machine consumer would get,
#: lenient is what a reviewer reading both would say.
LENIENT_FIELDS: frozenset = frozenset((
    "recommendedFoundations", "structureList", "bearingCapacity", "strata",
))
#: The lenient bar: the same partial ratio, lower, OR a number and its unit
#: in common. "3,000 psf for spread footings" and "spread footings, 3000 psf
#: net allowable" are one recommendation by the second test whatever the
#: first says.
LENIENT_RATIO = 70
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
#: The two list fields whose items are IDENTIFIERS rather than prose, and are
#: therefore compared exactly (folded) rather than by resemblance. B-1 and
#: B-12 are two holes, however alike they look to a fuzzy matcher.
_IDENTIFIER_LISTS = frozenset(("boringDictionary", "testPitDictionary"))

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


#: The four ways an answer can be wrong, in the order a scorecard prefers to
#: name them. ``convention`` is the one that matters for deciding what to do
#: next: the reader found the thing and answered it under a different house
#: rule from the hand's, which is fixed in
#: :mod:`report_ingest.narrative_glossary` and never by a better model.
MISS_KINDS: Tuple[str, ...] = ("convention", "wrong", "missed", "invented")


def miss_kind(field_name: str, verdict: str) -> str:
    """What KIND of miss this is -- the "why" column of the per-question table.

    A miss on a field a house rule governs is a ``convention`` miss, because
    that is what the evidence says it usually is: null where the rule says 0,
    the wrong sense of a "mention" question, a hazard list with shaking in it.
    Everything else keeps its own verdict. A right answer has no kind.
    """
    if verdict not in ("missed", "wrong", "invented"):
        return ""
    try:
        from report_ingest.narrative_glossary import CONVENTIONS
    except ImportError:                          # pragma: no cover
        return verdict
    for row in CONVENTIONS:
        if field_name in row.fields:
            return "convention"
    return verdict


def dominant_miss(kinds: Sequence[str]) -> str:
    """The commonest miss kind in a column, ties broken by :data:`MISS_KINDS`.

    ``""`` where nothing went wrong, which is the answer a field that is
    always right should give.
    """
    counts: Dict[str, int] = {}
    for kind in kinds:
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    if not counts:
        return ""
    best = max(counts.values())
    for name in MISS_KINDS:
        if counts.get(name) == best:
            return name
    return sorted(counts)[0]


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


#: A number and the unit printed beside it. The unit list is the one the two
#: schemas actually carry: pressures, depths, densities, angles, percentages.
_RE_NUMBER_UNIT = re.compile(
    r"(-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(psf|ksf|tsf|kpa|mpa|kn/m2|kg/cm2|pcf|kcf|ft|feet|m|in|inches|cm|mm|%|"
    r"degrees|degree|deg)\b",
    re.IGNORECASE)
#: Spellings of one unit, folded together, so "feet" and "ft" are one unit.
_UNIT_ALIASES: Dict[str, str] = {
    "feet": "ft", "inches": "in", "degree": "deg", "degrees": "deg",
    "kn/m2": "kpa",
}


def _numbers_with_units(value: Any) -> set:
    """``{(number, unit)}`` a string prints, for the lenient comparison.

    A number without a unit is not collected: two sentences that share the
    bare number 3 share nothing, and two that share ``3000 psf`` are about
    the same recommendation whatever else they say.
    """
    out: set = set()
    for text in ([value] if not isinstance(value, (list, tuple))
                 else list(value)):
        for match in _RE_NUMBER_UNIT.finditer(str(text or "")):
            try:
                number = float(match.group(1).replace(",", ""))
            except ValueError:                   # pragma: no cover - regex
                continue
            unit = match.group(2).lower()
            out.add((number, _UNIT_ALIASES.get(unit, unit)))
    return out


def _strings_match(field_name: str, hand: Any, got: Any,
                   lenient: bool = False) -> bool:
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
    if _proper(hand) & _proper(got):
        return True
    if not lenient:
        return False
    # THE LENIENT BAR: a lower ratio, or a number and its unit in common.
    if _partial_ratio(left, right) >= LENIENT_RATIO:
        return True
    return bool(_numbers_with_units(hand) & _numbers_with_units(got))


def _list_overlap(field_name: str, hand: Any, got: Any,
                  lenient: bool = False) -> Tuple[float, int, int, int]:
    """``(jaccard, hits, in the answer, in the truth)``.

    TWO KINDS OF LIST, judged differently, because they are two kinds of
    thing. ``boringDictionary`` and ``testPitDictionary`` are IDENTIFIERS and
    are compared exactly, folded the way the reconciler folds a hole's name,
    so that B-1, "B 1" and b1 are one hole and B-1 and B-12 are two. The rest
    are PROSE -- the hand-written ``bearingCapacity`` runs to fifteen words an
    item and ``recommendedFoundations`` to nineteen -- and are compared item by
    item with the same rule strings get, paired one to one so a single truth
    item cannot be matched twice and flatter the precision.
    """
    left = [item for item in (hand or []) if str(item).strip()]
    right = [item for item in (got or []) if str(item).strip()]
    if not left and not right:
        return 1.0, 0, 0, 0
    if field_name in _IDENTIFIER_LISTS:
        from report_ingest.reconciler import fold_id
        wanted = {fold_id(item) for item in left}
        given = {fold_id(item) for item in right}
        hits = len(wanted & given)
        union = len(wanted | given)
        return (hits / union if union else 1.0), hits, len(given), len(wanted)

    taken: List[int] = []
    hits = 0
    for item in left:
        for index, candidate in enumerate(right):
            if index in taken:
                continue
            if _strings_match(field_name, item, candidate, lenient):
                taken.append(index)
                hits += 1
                break
    union = len(left) + len(right) - hits
    return (hits / union if union else 1.0), hits, len(right), len(left)


def _matches(field_name: str, hand: Any, got: Any,
             lenient: bool = False) -> bool:
    """Does this ONE accepted answer agree with the prediction?

    ``lenient`` only loosens the four free-text fields in
    :data:`LENIENT_FIELDS`; an enumeration, a count and an identifier list
    are exact under both views, because nothing about those is a matter of
    wording.
    """
    from report_ingest.narrative_reader import VERDICT_FIELDS, verdict_token

    lenient = lenient and field_name in LENIENT_FIELDS
    kind = kind_of(field_name)
    if hand is None or got is None:
        return hand is None and got is None
    if field_name in VERDICT_FIELDS:
        # THE VERDICT IS THE ANSWER. The hand writes "yes - six seismic
        # refraction and MASW lines across the site" and a reader that finds
        # the same testing will not word its reason the same way. Scoring the
        # reason would be scoring a paraphrase; the question asked was whether
        # the report mentions it.
        one, two = verdict_token(hand), verdict_token(got)
        return one is not None and one == two
    if kind == "enum":
        return _fold(hand) == _fold(got)
    if kind == "int":
        try:
            return int(hand) == int(got)
        except (TypeError, ValueError):
            return False
    if kind == "list":
        jaccard, *_ = _list_overlap(field_name, hand, got, lenient)
        return jaccard >= LIST_HIT_JACCARD
    if kind == "summary":
        return bool(str(got or "").strip())
    return _strings_match(field_name, hand, got, lenient)


def same_value(field_name: str, hand: Any, got: Any,
               alternates: Sequence[Any] = (),
               lenient: bool = False) -> bool:
    """Do these two answers to one question agree?

    ``alternates`` are the truth file's own ``_alternates[field]``: other
    answers the hand will accept for this report, because a report can say a
    thing in more than one defensible way and the hand is not the only reading
    of it. A ``None`` among them means "not stated" is acceptable too, and a
    LIST among them is a whole alternative list answer. Any one of them
    matching is a hit.
    """
    if _matches(field_name, hand, got, lenient):
        return True
    return any(_matches(field_name, _normalise(alt), got, lenient)
               for alt in alternates)


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
    #: The same comparison under the LENIENT view. Equal to ``ok`` for every
    #: field outside :data:`LENIENT_FIELDS`, which is most of them.
    lenient_ok: bool = False
    #: What kind of miss this is, for the per-question table's "why" column:
    #: ``convention``, ``wrong``, ``missed``, ``invented`` or ``""``.
    miss: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {"field": self.field, "kind": self.kind,
               "verdict": self.verdict, "ok": self.ok,
               "lenient_ok": self.lenient_ok, "miss": self.miss,
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
    #: THE SAME THREE UNDER THE LENIENT VIEW, printed beside the strict ones
    #: and never instead of them. They differ only on the four free-text
    #: fields of :data:`LENIENT_FIELDS`.
    lenient_recall: Score = field(default_factory=Score)
    lenient_precision: Score = field(default_factory=Score)
    lenient_agreement: Score = field(default_factory=Score)
    #: The items inside every list field, pooled.
    list_items: Score = field(default_factory=Score)
    list_item_recall: Score = field(default_factory=Score)
    #: The four prose summaries: written where the report supports one, and
    #: inside the word limit.
    summaries_present: Score = field(default_factory=Score)
    summaries_within_limit: Score = field(default_factory=Score)
    #: Fields this report's hand said not to score (``_skip``), excluded
    #: from every count above.
    skipped: List[str] = field(default_factory=list)
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
            "lenient": {"recall": self.lenient_recall.to_dict(),
                        "precision": self.lenient_precision.to_dict(),
                        "agreement": self.lenient_agreement.to_dict()},
            "by_kind": {name: self.kind(name).to_dict()
                        for name in KINDS if self.by_kind.get(name)},
            "list_items": {"precision": self.list_items.to_dict(),
                           "recall": self.list_item_recall.to_dict()},
            "summaries": {"present": self.summaries_present.to_dict(),
                          "within_limit":
                              self.summaries_within_limit.to_dict()},
            "fields": [row.to_dict() for row in self.fields],
            "skipped": list(self.skipped),
            "unresolved": self.unresolved,
            "model_calls": self.model_calls,
            "cost": dict(self.cost),
            "pages": list(self.pages),
            "error": self.error,
        }


def _normalise(value: Any) -> Any:
    """One answer as the scorer holds it: a blank or an empty list is None.

    AN EMPTY LIST IS READ AS "NOT STATED", and that is a decision rather than
    a fact about the schema. The hand writes ``"testPitDictionary": []``
    beside ``"testPitCount": 0`` -- plainly "there are none" rather than "the
    report is silent" -- but the reader cannot express an empty list at all
    (:mod:`report_ingest.narrative_reader` stores an empty list as None), so
    scoring the two apart would make those fields permanent misses for a
    reader doing exactly the right thing. Both sides are collapsed the same
    way, which also makes ``[[]]`` in ``_alternates`` mean what the hand
    intends: saying nothing is acceptable here.
    """
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (list, tuple)) and not value:
        return None
    return value


def _value(blob: Dict[str, Any], name: str) -> Any:
    return _normalise(blob.get(name))


def alternates_for(truth: Dict[str, Any], field_name: str) -> List[Any]:
    """The other answers this report's hand will accept for this field.

    ``_alternates`` is the truth file's own, keyed by field, and is always a
    LIST of accepted values: a string, a number, ``null`` for "not stated is
    acceptable too", or a whole list where the answer is a list.
    """
    block = truth.get("_alternates") or {}
    value = block.get(field_name)
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def skipped_fields(truth: Dict[str, Any]) -> frozenset:
    """The fields this report's hand says not to score at all.

    ``_skip`` is for a question whose answer this report genuinely does not
    settle -- what counts as a table, most often -- where any score would be
    measuring the ambiguity rather than the reader.
    """
    return frozenset(str(name) for name in (truth.get("_skip") or []))


def score_narrative(truth: Dict[str, Any], general: Any, hazards: Any,
                    report: str = "") -> NarrativeScore:
    """Score one reading against one hand answer, question by question."""
    out = NarrativeScore(report=report or str(truth.get("id") or ""))
    skip = skipped_fields(truth)
    out.skipped = sorted(skip)
    sections = (("general", GENERAL_FIELDS, general),
                ("natural_hazards", NATURAL_HAZARD_FIELDS, hazards))
    for section, names, answers in sections:
        hand_section = truth.get(section) or {}
        for name in names:
            hand = _value(hand_section, name)
            got = _normalise(getattr(answers, name, None))
            kind = kind_of(name)

            if name in skip:
                # Not scored anywhere: not in recall, not in precision, not
                # in agreement. Recorded so a reader of the per-report detail
                # can see it was excluded rather than passed.
                out.fields.append(FieldResult(
                    field=name, kind=kind, hand=hand, got=got, ok=False,
                    verdict="skipped"))
                continue

            alternates = alternates_for(truth, name)
            if kind == "summary":
                _score_summary(out, name, hand, got)
                continue

            ok = same_value(name, hand, got, alternates)
            soft = ok or same_value(name, hand, got, alternates, lenient=True)
            jaccard = None
            if kind == "list" and hand is not None and got is not None:
                # The ITEMS' own precision and recall, pooled across every
                # list field: a boring dictionary that names eight of ten
                # holes is not the same failure as one that names two. Scored
                # against whichever accepted answer the prediction matched
                # best, so an alternate list answer counts its items too.
                jaccard, hits, given, wanted = _best_overlap(
                    name, hand, got, alternates)
                out.list_items.found += hits
                out.list_items.total += given
                out.list_item_recall.found += hits
                out.list_item_recall.total += wanted

            verdict = _verdict(hand, got, ok)
            out.fields.append(FieldResult(
                field=name, kind=kind, hand=hand, got=got, ok=ok,
                verdict=verdict, jaccard=jaccard, lenient_ok=soft,
                miss=miss_kind(name, verdict)))
            out.agreement.add(ok, f"{name}: {verdict}")
            out.lenient_agreement.add(soft)
            if hand is not None:
                out.recall.add(ok, f"{name}: {verdict}")
                out.lenient_recall.add(soft)
                out.kind(kind).add(ok, f"{name}: {verdict}")
            if got is not None:
                out.precision.add(ok, f"{name}: {verdict}")
                out.lenient_precision.add(soft)
    return out


def _best_overlap(field_name: str, hand: Any, got: Any,
                  alternates: Sequence[Any]) -> Tuple[float, int, int, int]:
    """The overlap against whichever accepted list answer fits best."""
    best = _list_overlap(field_name, hand, got)
    for alternate in alternates:
        value = _normalise(alternate)
        if not isinstance(value, (list, tuple)):
            continue
        candidate = _list_overlap(field_name, value, got)
        if candidate[0] > best[0]:
            best = candidate
    return best


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
                     budget: int = 8, report: str = "",
                     front_pages: Optional[int] = None,
                     investigations: Optional[Sequence[Any]] = None
                     ) -> NarrativeScore:
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
    extra: Dict[str, Any] = {}
    if front_pages is not None:
        extra["front_pages"] = int(front_pages)
    if investigations is not None:
        extra["investigations"] = list(investigations)
    try:
        result = read_narrative(doc, pages, engine, budget=budget,
                                body_pages=body, report_id=report, **extra)
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
