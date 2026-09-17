"""The narrative -> the owner's two query schemas, answered with citations.

THE STANDING QUESTIONS ARE THE WORK. The owner has been asking the same two
lists of questions of geotechnical reports for years -- what kind of document
this is, who wrote it, how many borings, what foundations were recommended,
what bearing pressure, which seismic code, what natural hazards. This reader
answers those lists off the report's own prose and nothing else, and every
answer it gives carries the page and the page's own words, so a reviewer can
go and check it in one step.

FOUR RULES THE PROMPT IS BUILT AROUND.

**Answer only from the text in front of you.** Not from what a firm of that
name usually recommends, not from what a site of that description usually is.
A geotechnical report is a document about one place, and a plausible answer
that the report does not contain is the failure this reader exists to avoid.

**Null is an answer and the commonest one.** Most reports answer most of one
schema and almost none of the other. A field the report does not address
stays null; it never becomes an empty string or the words "not stated",
because a scorer has to tell a report that did not say from a reader that did
not read.

**The enumerations are the owner's words.** A document type, a project phase,
a liquefaction verdict and a hazard list are answered from fixed vocabularies
and in the owner's spelling, so a query run today is comparable with one run
three years ago. An answer outside the vocabulary is REFUSED into
``unresolved`` rather than stored, because a value nothing downstream
recognises is worse than a blank.

**A public report about somebody else's project names no post.** ``postName``
and ``propertyType`` are questions about this owner's estate. A report written
for another client is marked ``outsideProject = "yes"`` and those two fields
stay null -- inferring a post from a city name is exactly the guess that makes
a library of answers untrustworthy.

WHAT PYTHON DOES RATHER THAN ASK. The table and figure counts are counted off
the captions in the main body and off the printed lists of tables and figures,
and handed to the model as FACTS; the model is asked anyway and a disagreement
is recorded, but the count that is stored is the one that was counted. The
report date is normalised to ISO here, the site class is reduced to its letter
here, and the ASCE 7 edition to its year here -- three small deterministic
jobs that a model does not need to be spent on.

THE BUDGET is eight model calls and a normal report costs ONE. The narrative
is sent whole when it fits; a narrative too long for one call is sent in page
chunks, one call each, and merged in Python -- first answer wins, lists are
unioned, and every disagreement between two chunks is recorded rather than
settled. Only when there was more than one chunk is a final call spent, and
only on the four prose summaries, which are the one thing a merge cannot do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest.engine import Engine, text_block, user
from report_ingest.model import (
    BearingValue, Citation, DOCUMENT_TYPE_VALUES, EARTH_HAZARD_VALUES,
    GENERAL_FIELDS, GeneralFacts, LIQUEFACTION_VALUES, Mention,
    NATURAL_HAZARD_FIELDS, NarrativeFacts, NaturalHazardFacts,
    PROJECT_PHASE_VALUES, PROPERTY_TYPE_VALUES, Quantity, SUMMARY_FIELDS,
    SUMMARY_WORD_LIMITS, Stratum, YES_NO_UNCLEAR,
)

__all__ = [
    "read_narrative", "NarrativeReadResult", "NarrativeReading", "ReadExtra",
    "SummaryReading", "NARRATIVE_SYSTEM", "MAX_MODEL_CALLS",
    "MAX_CHUNK_CHARS", "MAX_QUOTE_WORDS", "count_captions",
    "counts_from_outline", "serialise_page", "normalise_site_class",
    "normalise_asce_version", "iso_date", "VOCABULARIES",
]

#: The ceiling in the brief: eight model calls for one narrative. A narrative
#: that fits in one call -- almost all of them -- costs ONE.
MAX_MODEL_CALLS = 8
#: How much narrative text goes into one call. A geotechnical narrative runs
#: to about 2,500 characters a page, so this is roughly forty pages of prose:
#: comfortably inside every context these engines offer, and small enough that
#: a model is not asked to hold a whole volume in mind while answering thirty
#: seven questions.
MAX_CHUNK_CHARS = 90000
#: Text lines serialised per page, and how much of one line is sent.
MAX_LINES_PER_PAGE = 400
MAX_LINE_CHARS = 400
#: How much of a page's tables travels with it. A narrative page's table is
#: often where the bearing pressures and the seismic parameters actually sit,
#: so tables are worth their room.
MAX_TABLE_CHARS = 6000
#: A citation quote is the page's own words, and twenty of them is enough to
#: find the sentence again. Longer is cut here rather than argued about.
MAX_QUOTE_WORDS = 20

#: The vocabularies the reader must answer from, by the owner's field name.
#: One table: the prompt prints it, the builder validates against it, and a
#: change to an enumeration is a change in :mod:`report_ingest.model` alone.
VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "documentType": DOCUMENT_TYPE_VALUES,
    "propertyType": PROPERTY_TYPE_VALUES,
    "projectPhase": PROJECT_PHASE_VALUES,
    "outsideProject": YES_NO_UNCLEAR,
    "liquefactionPotential": LIQUEFACTION_VALUES,
    "earthHazardsExposed": EARTH_HAZARD_VALUES,
    "geophysicalTestingMention": YES_NO_UNCLEAR,
    "soilCorrosion": YES_NO_UNCLEAR,
    "siteResponseMention": YES_NO_UNCLEAR,
    "hazardAnalysisMention": YES_NO_UNCLEAR,
}

#: The fields whose answer is a list of strings as the report words them.
_LIST_FIELDS: Tuple[str, ...] = (
    "structureList", "boringDictionary", "testPitDictionary",
    "recommendedFoundations", "bearingCapacity",
)
#: The fields whose answer is a whole number.
_COUNT_FIELDS: Tuple[str, ...] = (
    "boringCount", "testPitCount", "cptCount", "tableCount", "figureCount",
    "previousInvestigationCount", "structureCount",
)
#: ``field -> the key it takes in NarrativeFacts.stated_counts``.
_STATED_COUNT_KEY: Dict[str, str] = {
    "boringCount": "borings", "testPitCount": "test_pits",
    "cptCount": "cpts", "tableCount": "tables", "figureCount": "figures",
    "previousInvestigationCount": "previous_investigations",
    "structureCount": "structures",
}
#: The two counts Python counts for itself off the pages.
_COUNTED_FIELD: Dict[str, str] = {"tableCount": "tables",
                                  "figureCount": "figures"}

#: What a model writes when it means null. Every one of them is a non-answer
#: and is stored as one, because a scorer counting "N/A" as an answer would
#: credit a reading that read nothing.
_NON_ANSWERS = frozenset((
    "n a", "na", "not stated", "unknown", "none stated", "not applicable",
    "not provided", "not specified", "not mentioned", "null", "none given",
))


# ---------------------------------------------------------------------------
# what Python counts rather than asks
# ---------------------------------------------------------------------------

#: A caption line: "Figure 3 - Boring Location Plan", "TABLE 2. Summary of
#: Laboratory Results", "Tableau 4 :", "Figura 1-2". The number may carry a
#: letter prefix or a section part ("Figure A-2", "Table 4.1"); the whole
#: label is what identifies it, so "Figure 4.1" and "Figure 4.2" are two.
_RE_CAPTION = re.compile(
    r"^\s*(figure|fig\.?|table|tableau|tabla|figura|plate|exhibit)\s+"
    r"([A-Za-z]{0,2}[-.]?\d+(?:[-.]\d+)*)\s*[-–—:.)]?\s*(\S.*)?$",
    re.IGNORECASE)

#: What each caption word counts towards.
_CAPTION_KIND = {
    "figure": "figures", "fig": "figures", "figura": "figures",
    "plate": "figures", "exhibit": "figures",
    "table": "tables", "tableau": "tables", "tabla": "tables",
}


def count_captions(doc: Any, pages: Sequence[int]) -> Dict[str, Any]:
    """Count the tables and figures the MAIN BODY captions, deterministically.

    A caption is a line that LEADS with "Figure N" or "Table N" AND goes on to
    name the thing -- a sentence that refers to table 3 in passing does not,
    and neither does a bare cross-reference that a line break happened to end.
    The label is what is counted, not the line, so a figure captioned on two
    pages counts once and "Figure 4.1" and "Figure 4.2" count twice.

    Returns the counts and the labels themselves, because a reviewer checking
    a count against a report wants to know WHICH ones were seen.
    """
    labels: Dict[str, List[str]] = {"figures": [], "tables": []}
    for page in pages:
        try:
            content = doc.page(page, tables=False)
        except Exception:                    # a page that will not read
            continue
        for line in content.lines:
            text = " ".join((line.text or "").split())
            if not text or len(text) > 200:
                continue
            match = _RE_CAPTION.match(text)
            if match is None:
                continue
            word = match.group(1).lower().rstrip(".")
            kind = _CAPTION_KIND.get(word)
            if kind is None:
                continue
            if not (match.group(3) or "").strip():
                continue
            label = f"{kind[:-1]} {match.group(2).upper()}"
            if label not in labels[kind]:
                labels[kind].append(label)
    return {"tables": len(labels["tables"]),
            "figures": len(labels["figures"]),
            "table_labels": labels["tables"],
            "figure_labels": labels["figures"]}


def counts_from_outline(outline: Any) -> Dict[str, int]:
    """What the printed lists of tables and figures say there are."""
    if outline is None:
        return {}
    try:
        figures = outline.of_kind("figure")
        tables = outline.of_kind("table")
    except AttributeError:                   # not an Outline
        return {}
    out: Dict[str, int] = {}
    if figures:
        out["figures"] = len(figures)
    if tables:
        out["tables"] = len(tables)
    return out


# ---------------------------------------------------------------------------
# the small deterministic normalisations
# ---------------------------------------------------------------------------

_RE_SITE_CLASS = re.compile(
    r"\b(?:site\s+)?class\s*[:\-]?\s*([A-F])\s*(?:[/\-]\s*([A-F]))?\b",
    re.IGNORECASE)
_RE_BARE_CLASS = re.compile(r"^\s*([A-F])\s*(?:[/\-]\s*([A-F]))?\s*$",
                            re.IGNORECASE)


def normalise_site_class(text: Optional[str]) -> Optional[str]:
    """``"Site Class D (stiff soil)"`` -> ``"D"``; ``"C/D"`` -> ``"CD"``.

    None when the text names no class this can read. The caller records that:
    a site class is a letter, and text that is not one is a misreading rather
    than an unusual answer.
    """
    if not text:
        return None
    raw = " ".join(str(text).split())
    match = _RE_BARE_CLASS.match(raw) or _RE_SITE_CLASS.search(raw)
    if match is None:
        return None
    first = match.group(1).upper()
    second = (match.group(2) or "").upper()
    return first + second if second else first


_RE_ASCE = re.compile(r"\b7\s*[-–]\s*(\d{2})\b")


def normalise_asce_version(text: Optional[str]) -> Optional[str]:
    """``"ASCE 7-16 (2016)"`` -> ``"7-16"``. None when no edition is named."""
    if not text:
        return None
    match = _RE_ASCE.search(str(text))
    return f"7-{match.group(1)}" if match else None


_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"), start=1)}
for _name, _index in list(_MONTHS.items()):
    _MONTHS[_name[:3]] = _index
_MONTHS["sept"] = 9
del _name, _index

_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_RE_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b")
_RE_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def iso_date(text: Optional[str]) -> Optional[str]:
    """A printed date as ``YYYY-MM-DD``, or None when it cannot be read.

    Reads the four spellings these reports print: an ISO date, "30 August
    2023", "August 30, 2023" and "8/30/2023". A slash date is read
    month-first, which is the convention of every report in this corpus; a
    date this cannot read comes back None and the caller says so rather than
    storing a guess.
    """
    if not text:
        return None
    raw = " ".join(str(text).split())
    match = _RE_ISO.search(raw)
    if match:
        year, month, day = (int(g) for g in match.groups())
        return _ymd(year, month, day)
    match = _RE_DMY.search(raw)
    if match:
        month = _MONTHS.get(match.group(2).lower())
        if month:
            return _ymd(int(match.group(3)), month, int(match.group(1)))
    match = _RE_MDY.search(raw)
    if match:
        month = _MONTHS.get(match.group(1).lower())
        if month:
            return _ymd(int(match.group(3)), month, int(match.group(2)))
    match = _RE_SLASH.search(raw)
    if match:
        return _ymd(int(match.group(3)), int(match.group(1)),
                    int(match.group(2)))
    return None


def _ymd(year: int, month: int, day: int) -> Optional[str]:
    if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


# ---------------------------------------------------------------------------
# what the model returns
# ---------------------------------------------------------------------------

class ReadCitation(BaseModel):
    """One answer's evidence: the field, the page and the page's words."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        description="the field this supports, by its exact name, e.g. "
                    "'boringCount' or 'siteClass'")
    page: int = Field(description="0-based PDF page index, as shown")
    quote: str = Field(
        description="the page's own words that say it, 20 words or fewer, "
                    "copied exactly")


class ReadBearing(BaseModel):
    """One recommended bearing pressure, as a number with its unit."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="the number as printed")
    unit: str = Field(
        description="the unit as printed: psf, ksf, tsf, kPa, kg/cm2")
    foundation_type: str = Field(
        default="",
        description="what it is for, as printed: spread footing, mat, "
                    "drilled shaft end bearing")
    condition: str = Field(
        default="",
        description="the condition, as printed: 'net allowable', 'on "
                    "engineered fill', 'at 1.5 m embedment'")
    page: Optional[int] = Field(default=None)
    quote: str = Field(default="", description="20 words or fewer")


class ReadStratum(BaseModel):
    """One stratum of the profile the report describes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="the report's own name for it")
    description: str = Field(default="", description="as printed")
    top: Optional[float] = Field(
        default=None,
        description="depth to its top, when the report gives one")
    bottom: Optional[float] = Field(default=None)
    depth_unit: str = Field(
        default="",
        description="'ft' or 'm' -- the unit top and bottom are in; empty "
                    "when the report gives no depth")
    uscs: str = Field(
        default="",
        description="the group symbol the report prints for it; empty when "
                    "it prints none")
    page: Optional[int] = Field(default=None)


class ReadExtra(BaseModel):
    """One question asked of this report beyond the two schemas."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="the question, as it was asked")
    answer: str = Field(
        description="what the narrative says, in 60 words or fewer; an EMPTY "
                    "string when it does not say")
    page: Optional[int] = Field(default=None)
    quote: str = Field(default="", description="20 words or fewer")


class Unsettled(BaseModel):
    """Something in the narrative you could not settle."""

    model_config = ConfigDict(extra="forbid")

    what: str = Field(description="what it is, in 15 words or fewer")
    page: Optional[int] = Field(default=None)
    why: str = Field(description="what stopped you, in 20 words or fewer")


class NarrativeReading(BaseModel):
    """Both of the owner's schemas as the model read them.

    The names are the owner's and are answered exactly as spelled. Every one
    of them may be null, and null MEANS the narrative does not say.
    """

    model_config = ConfigDict(extra="forbid")

    # -- the general schema ------------------------------------------------
    documentType: Optional[str] = None
    quickSummary: Optional[str] = None
    postName: Optional[str] = None
    propertyType: Optional[str] = None
    projectNumber: Optional[str] = None
    projectName: Optional[str] = None
    projectPhase: Optional[str] = None
    primeContractor: Optional[str] = None
    primeAe: Optional[str] = None
    geotechnicalEngineerFirm: Optional[str] = None
    testingProgramSummary: Optional[str] = None
    boringCount: Optional[int] = None
    testPitCount: Optional[int] = None
    cptCount: Optional[int] = None
    tableCount: Optional[int] = None
    figureCount: Optional[int] = None
    previousInvestigationCount: Optional[int] = None
    strata: Optional[str] = None
    structureCount: Optional[int] = None
    structureList: Optional[List[str]] = None
    outsideProject: Optional[str] = None
    boringDictionary: Optional[List[str]] = None
    testPitDictionary: Optional[List[str]] = None
    recommendedFoundations: Optional[List[str]] = None
    bearingCapacity: Optional[List[str]] = None

    # -- the natural hazards schema ---------------------------------------
    liquefactionPotential: Optional[str] = None
    asceSevenVersion: Optional[str] = None
    earthHazardsExposed: Optional[List[str]] = None
    seismicCodeUsed: Optional[str] = None
    geophysicalTestingMention: Optional[str] = None
    soilCorrosion: Optional[str] = None
    siteResponseMention: Optional[str] = None
    hazardAnalysisMention: Optional[str] = None
    siteClass: Optional[str] = None
    seismicParameterSummary: Optional[str] = None
    naturalHazardSummary: Optional[str] = None
    reportDate: Optional[str] = None
    reportDateISO: Optional[str] = Field(
        default=None,
        description="the same date as YYYY-MM-DD, when you can tell")

    # -- the typed twins ---------------------------------------------------
    bearingCapacityValues: List[ReadBearing] = Field(default_factory=list)
    strataList: List[ReadStratum] = Field(default_factory=list)

    extra_answers: List[ReadExtra] = Field(
        default_factory=list,
        description="one entry for each question in THE CALLER'S OWN "
                    "QUESTIONS, in the order asked; empty when there were "
                    "none")
    citations: List[ReadCitation] = Field(
        default_factory=list,
        description="one entry for EVERY field you answered")
    unsettled: List[Unsettled] = Field(default_factory=list)


class SummaryReading(BaseModel):
    """The four prose summaries, written once over the whole narrative."""

    model_config = ConfigDict(extra="forbid")

    quickSummary: Optional[str] = Field(
        default=None, description="100 words or fewer")
    testingProgramSummary: Optional[str] = Field(
        default=None, description="100 words or fewer")
    naturalHazardSummary: Optional[str] = Field(
        default=None, description="200 words or fewer")
    seismicParameterSummary: Optional[str] = Field(
        default=None, description="100 words or fewer")


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclass
class NarrativeReadResult:
    """One narrative read, and everything needed to audit the reading.

    ``general`` and ``natural_hazards`` are the two schemas answered; they go
    straight onto the record's own sections of those names. ``facts`` is what
    sits BESIDE them -- what the narrative stated, what Python counted -- and
    goes onto ``record.narrative``.
    """

    general: GeneralFacts = field(default_factory=GeneralFacts)
    natural_hazards: NaturalHazardFacts = field(
        default_factory=NaturalHazardFacts)
    facts: NarrativeFacts = field(default_factory=NarrativeFacts)
    #: Every citation, keyed by the owner's field name. The same entries are
    #: on the two schemas; this is the flat view a writer walks.
    citations: Dict[str, List[Citation]] = field(default_factory=dict)
    #: What could not be settled: the model's own list, plus everything
    #: Python refused -- a value outside an enumeration, a citation on a page
    #: that is not in this narrative, a summary past its word limit, a
    #: disagreement between two chunks of a long narrative.
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)

    @property
    def answered(self) -> List[str]:
        """Every owner field this reading answered, general first."""
        return self.general.answered() + self.natural_hazards.answered()

    def apply(self, record: Any) -> Any:
        """Write this reading onto a record's three narrative sections."""
        record.general = self.general
        record.natural_hazards = self.natural_hazards
        record.narrative = self.facts
        return record

    def to_dict(self) -> Dict[str, Any]:
        return {
            "general": self.general.model_dump(mode="json"),
            "natural_hazards": self.natural_hazards.model_dump(mode="json"),
            "facts": self.facts.model_dump(mode="json"),
            "answered": self.answered,
            "n_answered": len(self.answered),
            "unresolved": [dict(u) for u in self.unresolved],
            "cost": dict(self.cost),
            "model_calls": self.model_calls,
            "model": self.model,
            "warnings": list(self.warnings),
            "pages": list(self.pages),
        }


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

NARRATIVE_SYSTEM = """\
You are reading the narrative of ONE geotechnical report and answering two
fixed lists of questions about it. The lists have been asked of thousands of
reports over years, so they are answered in their own words and their own
spelling every time: an answer is comparable with an answer given three years
ago or it is worth nothing.

THE FOUR RULES.

1. ANSWER ONLY FROM THE TEXT YOU ARE GIVEN. Not from the firm's usual
   practice, not from what a site of that description usually is, not from
   the project's name. If the pages in front of you do not say it, you do not
   know it.

2. NULL IS AN ANSWER, AND THE RIGHT ONE MORE OFTEN THAN NOT. A report that
   does not discuss liquefaction leaves liquefactionPotential null. Never
   write "not stated", "N/A", "unknown" or an empty string into a field:
   write null. Most reports answer most of the general list and only part of
   the natural-hazards list, and that is the expected shape of an answer.

3. THE ENUMERATED FIELDS TAKE ONE OF THEIR LISTED WORDS, EXACTLY AS SPELLED
   in the vocabulary below. An answer that is not one of those words is
   thrown away, so where none of them fits use the vocabulary's own
   "unclear" or "other" if it has one, and null if it does not.

4. A REPORT ABOUT SOMEBODY ELSE'S PROJECT NAMES NO POST. postName and
   propertyType are questions about one estate of government property. If
   this report was written for another client -- a private developer, a
   municipality, another agency -- set outsideProject to "yes" and leave
   postName and propertyType null. Never infer a post from a city, a country
   or a project name.

CITATIONS. Every field you answer needs at least one citation: the field's
exact name, the 0-based page index as shown in the "=== page N ===" headers,
and 20 words or fewer of the page's OWN text, copied, that say it. A citation
whose page is not one of the pages you were given is thrown away with the
answer it supports. Cite the page that states the fact, not a contents page
that lists the section.

COUNTS. boringCount, testPitCount and cptCount are what the NARRATIVE SAYS
was done -- the sentence that says "four borings were advanced" -- not what
you can infer from anything else. tableCount and figureCount have already
been counted from the captions and are given to you as facts below; answer
them anyway, from the report's own lists of tables and figures, and a
disagreement will be recorded rather than argued.

VALUES WITH UNITS. bearingCapacity keeps the report's own sentences, one per
recommendation. bearingCapacityValues is the same recommendations as numbers:
the number as printed, the unit as printed (psf, ksf, tsf, kPa), what
foundation it is for and under what condition. Never convert a unit, and
never give a value whose unit the report does not print.

THE PROFILE. strata is the report's description of the soil profile in one
string, as it words it. strataList is the same profile as records, one per
stratum, with the depths the report gives and the USCS symbol IT prints --
never a symbol you inferred from a description.

THE SUMMARIES are prose, written for an engineer who has not read the report:
quickSummary (what this report is and what it concluded, 100 words or fewer),
testingProgramSummary (what was drilled, dug, sounded and tested, 100 words
or fewer), naturalHazardSummary (what it says about natural hazards, 200
words or fewer), seismicParameterSummary (the seismic design parameters, 100
words or fewer). Over the limit is a defect; under it is fine. Leave one null
rather than padding it out of nothing.

Anything you could not settle goes in unsettled, with the page and what
stopped you.
"""

_FINAL_INSTRUCTION = (
    "Now answer both lists as data: every field the narrative answers, null "
    "for every field it does not, a citation for each answer, and anything "
    "you could not settle."
)

_SUMMARY_SYSTEM = """\
You are writing the four prose summaries of one geotechnical report. You are
given the answers already read off it and the report's own section headings.
Write the summaries for an engineer who has not read the report, FROM THOSE
ANSWERS ALONE, and keep inside the word limits: 100 words for quickSummary,
100 for testingProgramSummary, 200 for naturalHazardSummary, 100 for
seismicParameterSummary. Say only what the answers say. Leave one null rather
than writing a summary of nothing.
"""


def _vocabulary_block() -> str:
    lines = ["THE ENUMERATED FIELDS AND THEIR WORDS"]
    for name, words in VOCABULARIES.items():
        many = (" (a list; give every one that applies)"
                if name == "earthHazardsExposed" else "")
        lines.append(f"  {name}{many}: " + " | ".join(words))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def _clip(text: str, n: int = MAX_LINE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def serialise_page(doc: Any, page: int) -> str:
    """One narrative page as page-tagged text, with its tables.

    Prose, not geometry: a narrative answer comes off a sentence and is cited
    by page and words, so the boxes that matter on a log form are noise here.
    The page's tables come with it, because the bearing pressures and the
    seismic parameters are as often in a table on a narrative page as in its
    prose.
    """
    out: List[str] = [f"=== page {page} ==="]
    try:
        content = doc.page(page)
    except Exception as exc:                     # a page that will not read
        out.append(f"[could not be read: {type(exc).__name__}: {exc}]")
        return "\n".join(out)
    for warning in content.warnings:
        out.append(f"[WARNING: {_clip(warning, 160)}]")
    if not content.lines:
        out.append("[this page carries no text layer]")
    lines = sorted(content.lines,
                   key=lambda ln: (round(ln.bbox[1], 1), ln.bbox[0]))
    for line in lines[:MAX_LINES_PER_PAGE]:
        text = _clip(line.text)
        if text:
            out.append(text)
    if len(lines) > MAX_LINES_PER_PAGE:
        out.append(f"[{len(lines) - MAX_LINES_PER_PAGE} further line(s) not "
                   f"listed]")
    budget = MAX_TABLE_CHARS
    for table in content.tables:
        if budget <= 0:
            out.append("[further table(s) on this page not listed]")
            break
        markdown = table.to_markdown()
        if len(markdown) > budget:
            markdown = markdown[:budget] + "\n[table truncated]"
        budget -= len(markdown)
        out.append(f"TABLE {table.id}, {table.n_rows}x{table.n_cols}")
        out.append(markdown)
    return "\n".join(out)


def _chunks(doc: Any, pages: Sequence[int],
            max_chars: int) -> List[Tuple[List[int], str]]:
    """The narrative as one block of text, or as several when it is long.

    A chunk never splits a page, and a single page longer than the limit is
    its own chunk rather than being cut: a page cut in half loses the sentence
    the answer was in, and the answer with it.
    """
    out: List[Tuple[List[int], str]] = []
    current: List[int] = []
    body: List[str] = []
    used = 0
    for page in pages:
        text = serialise_page(doc, page)
        if current and used + len(text) > max_chars:
            out.append((current, "\n".join(body)))
            current, body, used = [], [], 0
        current.append(int(page))
        body.append(text)
        used += len(text)
    if current:
        out.append((current, "\n".join(body)))
    return out


def _headings_text(outline: Any, pages: Sequence[int]) -> str:
    if outline is None:
        return ""
    wanted = set(int(p) for p in pages)
    marks = [m for m in getattr(outline, "headings", ())
             if m.page in wanted]
    return "\n".join(f"  page {m.page}: {m.text}" for m in marks)


def _brief(pages: Sequence[int], all_pages: Sequence[int], text: str,
           counted: Dict[str, Any], outline_counts: Dict[str, int],
           headings: str, report_id: str, chunk: int, n_chunks: int,
           questions: Optional[Sequence[str]] = None) -> str:
    parts: List[str] = [
        "THE NARRATIVE of one geotechnical report"
        + (f" ({report_id})" if report_id else "") + "."]
    if n_chunks > 1:
        parts.append(
            f"This is part {chunk} of {n_chunks}. You are seeing pages "
            f"{pages[0]}-{pages[-1]} of a narrative that runs over pages "
            f"{all_pages[0]}-{all_pages[-1]}. Answer what THESE pages answer "
            f"and leave the rest null; the parts are merged afterwards, so a "
            f"field you leave null here is not lost if another part answers "
            f"it.")
    else:
        parts.append("The narrative is pages "
                     + ", ".join(str(p) for p in pages) + ", in full.")
    parts += [
        "",
        _vocabulary_block(),
        "",
        "FACTS ALREADY COUNTED FROM THE PAGES",
        f"  tables captioned in the main body: {counted.get('tables', 0)}",
        f"  figures captioned in the main body: {counted.get('figures', 0)}",
    ]
    if outline_counts:
        parts.append(
            "  the report's own printed lists: "
            + ", ".join(f"{k} {v}" for k, v in sorted(outline_counts.items())))
    if headings:
        parts += ["", "THE NARRATIVE'S OWN SECTION HEADINGS", headings]
    asked = [" ".join(str(q).split()) for q in (questions or [])
             if str(q).strip()]
    if asked:
        parts += ["",
                  "THE CALLER'S OWN QUESTIONS. Answer each of these as well, "
                  "in extra_answers, under the same rules: only from the "
                  "text, with a citation, and an EMPTY answer where the "
                  "narrative does not say."]
        parts += [f"  {n}. {q}" for n, q in enumerate(asked, start=1)]
    parts += ["", "THE PAGES", text]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# turning the reading into the record
# ---------------------------------------------------------------------------

def _fold(value: Any) -> str:
    """A word folded for comparison against a vocabulary."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _quote(text: str) -> str:
    words = " ".join(str(text or "").split()).split()
    if len(words) <= MAX_QUOTE_WORDS:
        return " ".join(words)
    return " ".join(words[:MAX_QUOTE_WORDS]) + "…"


class _Builder:
    """Turns one or more :class:`NarrativeReading` into the record's facts.

    Everything a real narrative cannot support is refused here rather than
    stored: a value outside an enumeration, a citation on a page that is not
    part of this narrative, a bearing pressure with no unit, a summary past
    its word limit, and -- on a narrative long enough to have been read in
    parts -- two parts that answer the same field differently.
    """

    def __init__(self, pages: Sequence[int]) -> None:
        self.pages = set(int(p) for p in pages)
        self.unresolved: List[Dict[str, Any]] = []
        self.citations: Dict[str, List[Citation]] = {}
        self.values: Dict[str, Any] = {}
        self.stated: Dict[str, int] = {}
        self.bearing: List[BearingValue] = []
        self.strata: List[Stratum] = []
        self.extra: List[Dict[str, Any]] = []
        self._from_chunk: Dict[str, int] = {}

    # -- refusals ----------------------------------------------------------
    def refuse(self, what: str, why: str, value: Any = None,
               page: Optional[int] = None) -> None:
        row: Dict[str, Any] = {"what": what, "why": why}
        if value is not None:
            row["value"] = str(value)[:200]
        if page is not None:
            row["page"] = int(page)
        self.unresolved.append(row)

    # -- one reading -------------------------------------------------------
    def add(self, reading: NarrativeReading, chunk: int = 1) -> None:
        """Fold one chunk's answers in.

        First answer wins; a second, DIFFERENT answer to the same question is
        a disagreement between two parts of one narrative and is recorded
        rather than settled. Lists are the exception: two parts each naming
        some of the borings are both right, so lists are unioned.
        """
        for name in GENERAL_FIELDS + NATURAL_HAZARD_FIELDS:
            value = self._clean(name, getattr(reading, name, None))
            if value is None:
                continue
            if name in self.values:
                if self.values[name] == value:
                    continue
                if isinstance(value, list):
                    merged = list(self.values[name])
                    for item in value:
                        if item not in merged:
                            merged.append(item)
                    self.values[name] = merged
                    continue
                self.refuse(
                    name,
                    f"part {self._from_chunk.get(name)} and part {chunk} of "
                    f"the narrative answer this differently; the first "
                    f"answer is kept",
                    f"{self.values[name]!r} vs {value!r}")
                continue
            self.values[name] = value
            self._from_chunk[name] = chunk
            key = _STATED_COUNT_KEY.get(name)
            if key is not None and isinstance(value, int):
                self.stated[key] = value

        if reading.reportDateISO and "reportDateISO" not in self.values:
            iso = iso_date(reading.reportDateISO)
            if iso:
                self.values["reportDateISO"] = iso

        for read_bearing in reading.bearingCapacityValues:
            self._bearing(read_bearing)
        for read_stratum in reading.strataList:
            self._stratum(read_stratum)
        for extra in reading.extra_answers:
            self._extra(extra)
        for cite in reading.citations:
            self._citation(cite)
        for row in reading.unsettled:
            self.refuse(row.what, row.why, page=row.page)

    # -- one value ---------------------------------------------------------
    def _clean(self, name: str, value: Any) -> Any:
        if value is None:
            return None
        if name in VOCABULARIES:
            return self._enum(name, value)
        if name in _COUNT_FIELDS:
            try:
                number = int(value)
            except (TypeError, ValueError):
                self.refuse(name, "not a whole number", value)
                return None
            if number < 0:
                self.refuse(name, "a count cannot be negative", value)
                return None
            return number
        if name in _LIST_FIELDS:
            items: List[str] = []
            for item in (value or []):
                text = " ".join(str(item).split())
                if text and _fold(text) not in _NON_ANSWERS \
                        and text not in items:
                    items.append(text)
            return items or None
        text = " ".join(str(value).split())
        if not text or _fold(text) in _NON_ANSWERS:
            return None
        if name in SUMMARY_FIELDS:
            limit = SUMMARY_WORD_LIMITS[name]
            words = len(text.split())
            if words > limit:
                self.refuse(name, f"{words} words, past the {limit}-word "
                                  f"limit; kept as written")
        return text

    def _enum(self, name: str, value: Any) -> Any:
        words = VOCABULARIES[name]
        folded = {_fold(w): w for w in words}
        if name == "earthHazardsExposed":
            out: List[str] = []
            for item in (value or []):
                got = folded.get(_fold(item))
                if got is None:
                    self.refuse(name, "not one of the listed hazards", item)
                elif got not in out:
                    out.append(got)
            return out or None
        got = folded.get(_fold(value))
        if got is None:
            self.refuse(name, f"not one of {', '.join(words)}", value)
            return None
        return got

    def _bearing(self, read: ReadBearing) -> None:
        if not (read.unit or "").strip():
            self.refuse("bearingCapacityValues",
                        "a pressure with no unit is not a value",
                        f"{read.value} for {read.foundation_type}")
            return
        self.bearing.append(BearingValue(
            value=Quantity(value=float(read.value),
                           unit=" ".join(read.unit.split())),
            foundation_type=" ".join((read.foundation_type or "").split()),
            condition=" ".join((read.condition or "").split()),
            citation=self._cites_for(read.page, read.quote,
                                     "bearingCapacityValues")))

    def _stratum(self, read: ReadStratum) -> None:
        unit = " ".join((read.depth_unit or "").split())
        top = bottom = None
        if read.top is not None:
            if not unit:
                self.refuse("strataList",
                            "a depth with no unit cannot be stored",
                            f"{read.name}: top {read.top}")
            else:
                top = Quantity(value=float(read.top), unit=unit)
        if read.bottom is not None and unit:
            bottom = Quantity(value=float(read.bottom), unit=unit)
        if top is not None and bottom is not None and bottom.value < top.value:
            self.refuse("strataList", "the base is above the top",
                        f"{read.name}: {read.top} to {read.bottom} {unit}")
            bottom = None
        self.strata.append(Stratum(
            name=" ".join((read.name or "").split()),
            description=" ".join((read.description or "").split()),
            top=top, bottom=bottom,
            uscs=" ".join((read.uscs or "").split()).upper(),
            citation=self._cites_for(read.page, "", "strataList")))

    def _extra(self, read: ReadExtra) -> None:
        """One answer to a question the caller asked.

        An empty answer is KEPT, with its question: "the narrative does not
        say" is the answer to most questions asked of most reports, and a
        caller who gets silence back cannot tell that from a reader that
        forgot to ask.
        """
        question = " ".join((read.question or "").split())
        if not question:
            return
        answer = " ".join((read.answer or "").split())
        if _fold(answer) in _NON_ANSWERS:
            answer = ""
        row: Dict[str, Any] = {"question": question, "answer": answer}
        cites = self._cites_for(read.page, read.quote, "extra_answers")             if answer else []
        if cites:
            row["page"] = cites[0].page
            row["quote"] = cites[0].quote
        for seen in self.extra:
            if seen["question"] == question:
                if not seen.get("answer") and answer:
                    seen.update(row)
                return
        self.extra.append(row)

    def _cites_for(self, page: Optional[int], quote: str,
                   what: str) -> List[Citation]:
        if page is None:
            return []
        if int(page) not in self.pages:
            self.refuse(what, "cited a page that is not in this narrative",
                        page=page)
            return []
        return [Citation(page=int(page), quote=_quote(quote))]

    def _citation(self, cite: ReadCitation) -> None:
        name = (cite.field or "").strip()
        if name not in GENERAL_FIELDS and name not in NATURAL_HAZARD_FIELDS:
            self.refuse("citation",
                        f"cites a field that does not exist: {name!r}",
                        page=cite.page)
            return
        if int(cite.page) not in self.pages:
            self.refuse(name, "cited a page that is not in this narrative",
                        page=cite.page)
            return
        row = Citation(page=int(cite.page), quote=_quote(cite.quote))
        here = self.citations.setdefault(name, [])
        if not any(c.page == row.page and c.quote == row.quote for c in here):
            here.append(row)

    # -- the answer --------------------------------------------------------
    def build(self, pages: Sequence[int], counted: Dict[str, Any]
              ) -> Tuple[GeneralFacts, NaturalHazardFacts, NarrativeFacts]:
        values = dict(self.values)

        # The two counts Python counted for itself. The counted number is
        # what is stored when there was anything to count; the model's answer
        # is kept in stated_counts either way and a disagreement is recorded,
        # because a report's own list of figures disagreeing with the captions
        # on its pages is a fact about the report.
        for name, key in _COUNTED_FIELD.items():
            ours = int(counted.get(key) or 0)
            theirs = values.get(name)
            if ours:
                if theirs is not None and theirs != ours:
                    self.refuse(
                        name,
                        f"the captions in the body count {ours}; the reading "
                        f"says {theirs}. The counted number is stored",
                        theirs)
                values[name] = ours

        general = GeneralFacts(
            **{k: v for k, v in values.items() if k in GENERAL_FIELDS},
            bearingCapacityValues=self.bearing,
            strataList=self.strata,
            outsideProjectAnswer=self._mention("outsideProject", values),
            citations={k: v for k, v in self.citations.items()
                       if k in GENERAL_FIELDS})
        hazards = NaturalHazardFacts(
            **{k: v for k, v in values.items() if k in NATURAL_HAZARD_FIELDS},
            siteClassNormalized=normalise_site_class(values.get("siteClass")),
            asceSevenVersionNormalized=normalise_asce_version(
                values.get("asceSevenVersion")),
            reportDateISO=(iso_date(values.get("reportDate"))
                           or values.get("reportDateISO")),
            geophysicalTestingAnswer=self._mention(
                "geophysicalTestingMention", values),
            soilCorrosionAnswer=self._mention("soilCorrosion", values),
            siteResponseAnswer=self._mention("siteResponseMention", values),
            hazardAnalysisAnswer=self._mention("hazardAnalysisMention",
                                               values),
            citations={k: v for k, v in self.citations.items()
                       if k in NATURAL_HAZARD_FIELDS})

        if values.get("siteClass") and hazards.siteClassNormalized is None:
            self.refuse("siteClass",
                        "names no site class letter this can read",
                        values["siteClass"])
        if values.get("reportDate") and hazards.reportDateISO is None:
            self.refuse("reportDate", "not a date this can normalise",
                        values["reportDate"])
        for name in general.answered() + hazards.answered():
            if name not in self.citations:
                self.refuse(name, "answered with no citation")

        facts = NarrativeFacts(
            stated_counts=dict(self.stated),
            counted={k: int(v) for k, v in counted.items()
                     if isinstance(v, int)},
            pages=[int(p) for p in pages],
            extra_answers=[dict(row) for row in self.extra],
            unresolved=[dict(u) for u in self.unresolved])
        return general, hazards, facts

    def _mention(self, name: str,
                 values: Dict[str, Any]) -> Optional[Mention]:
        answer = values.get(name)
        if answer not in YES_NO_UNCLEAR:
            return None
        return Mention(answer=answer,
                       citation=list(self.citations.get(name, [])))


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

def read_narrative(doc: Any, narrative_pages: Sequence[int], engine: Engine,
                   *, budget: int = MAX_MODEL_CALLS,
                   outline: Any = None,
                   report_id: str = "",
                   counted: Optional[Dict[str, Any]] = None,
                   body_pages: Optional[Sequence[int]] = None,
                   questions: Optional[Sequence[str]] = None,
                   max_chunk_chars: Optional[int] = None
                   ) -> NarrativeReadResult:
    """Read the narrative and answer the owner's two schemas.

    ``narrative_pages`` are the pages of the narrative work item --
    ``planlens.document.roles.document_items`` gives them. ``budget`` is the
    model-call ceiling; a narrative that fits in one call costs one.

    ``outline`` is :func:`planlens.document.roles.document_outline` when the
    caller already has it: its printed lists of tables and figures and its
    section headings go into the brief. ``counted`` is
    :func:`count_captions` when a caller has already run it; otherwise it is
    run here over ``body_pages`` -- the main body, which is the narrative and
    the figure pages before the appendices -- or over the narrative pages.
    """
    pages = [int(p) for p in narrative_pages]
    if not pages:
        raise ValueError("read_narrative needs at least one page")
    budget = max(1, min(int(budget), MAX_MODEL_CALLS))

    if counted is None:
        counted = count_captions(
            doc, list(body_pages) if body_pages else pages)
    outline_counts = counts_from_outline(outline)
    headings = _headings_text(outline, pages)

    spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0}

    def charge(reply: Any) -> None:
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)

    warnings: List[str] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:                 # a page that will not read
            warnings.append(f"page {page}: {type(exc).__name__}: {exc}")
        else:
            warnings.extend(f"page {page}: {w}" for w in content.warnings)

    blocks = _chunks(doc, pages, int(max_chunk_chars or MAX_CHUNK_CHARS))
    # One call is kept back for the summaries whenever the narrative had to
    # be read in parts: a summary written over part three of five is a
    # summary of a fifth of the report, and merging five of them in Python
    # would be worse than any one of them.
    max_chunks = budget - 1 if len(blocks) > 1 else budget
    if len(blocks) > max_chunks:
        warnings.append(
            f"the narrative is {len(blocks)} chunks and the budget allows "
            f"{max_chunks}; pages {blocks[max_chunks][0][0]}-"
            f"{blocks[-1][0][-1]} were not read")
        blocks = blocks[:max_chunks]

    builder = _Builder(pages)
    model_calls = 0
    final: Any = None
    read_pages: List[int] = []
    for index, (chunk_pages, text) in enumerate(blocks, start=1):
        brief = _brief(chunk_pages, pages, text, counted, outline_counts,
                       headings, report_id, index, len(blocks), questions)
        messages = [user(text_block(brief)),
                    user(text_block(_FINAL_INSTRUCTION))]
        reply = engine.complete(messages, system=NARRATIVE_SYSTEM,
                                output_format=NarrativeReading)
        model_calls += 1
        charge(reply)
        final = reply
        if reply.parsed is None:
            warnings.append(
                f"pages {chunk_pages[0]}-{chunk_pages[-1]}: no structured "
                f"answer (stop_reason {reply.stop_reason!r})")
            continue
        read_pages.extend(chunk_pages)
        builder.add(reply.parsed, index)

    if not read_pages:
        raise RuntimeError(
            f"the narrative reader returned no structured reading for pages "
            f"{pages} in {model_calls} call(s) (stop_reason "
            f"{getattr(final, 'stop_reason', None)!r})")

    general, hazards, facts = builder.build(pages, counted)

    if len(blocks) > 1 and model_calls < budget:
        reply = _summarise(general, hazards, headings, engine)
        model_calls += 1
        charge(reply)
        if reply.parsed is not None:
            _apply_summaries(reply.parsed, general, hazards, builder)
        else:
            warnings.append("the summary call returned no structured answer; "
                            "the per-part summaries stand")

    facts.unresolved = [dict(u) for u in builder.unresolved]
    spent["seconds"] = round(spent["seconds"], 2)
    spent["dollars"] = round(spent["dollars"], 5)
    return NarrativeReadResult(
        general=general,
        natural_hazards=hazards,
        facts=facts,
        citations=dict(builder.citations),
        unresolved=[dict(u) for u in builder.unresolved],
        cost=spent,
        model_calls=model_calls,
        model=getattr(final, "model", "") or getattr(engine, "name", ""),
        warnings=warnings,
        pages=pages)


def _summarise(general: GeneralFacts, hazards: NaturalHazardFacts,
               headings: str, engine: Engine) -> Any:
    """One call for the four prose summaries over a multi-part narrative."""
    answers: List[str] = []
    for name in GENERAL_FIELDS:
        value = getattr(general, name, None)
        if value is not None and name not in SUMMARY_FIELDS:
            answers.append(f"  {name}: {value}")
    for name in NATURAL_HAZARD_FIELDS:
        value = getattr(hazards, name, None)
        if value is not None and name not in SUMMARY_FIELDS:
            answers.append(f"  {name}: {value}")
    for item in general.bearingCapacityValues:
        answers.append(f"  bearing: {item.value} {item.foundation_type} "
                       f"{item.condition}".rstrip())
    for item in general.strataList:
        answers.append(f"  stratum: {item.name} {item.description}".rstrip())
    parts = ["THE ANSWERS ALREADY READ OFF THIS REPORT",
             "\n".join(answers) or "  (nothing was answered)"]
    if headings:
        parts += ["", "THE NARRATIVE'S OWN SECTION HEADINGS", headings]
    parts += ["", "Write the four summaries now."]
    return engine.complete([user(text_block("\n".join(parts)))],
                           system=_SUMMARY_SYSTEM,
                           output_format=SummaryReading)


def _apply_summaries(reading: SummaryReading, general: GeneralFacts,
                     hazards: NaturalHazardFacts, builder: _Builder) -> None:
    for name in SUMMARY_FIELDS:
        text = getattr(reading, name, None)
        if not text:
            continue
        text = " ".join(str(text).split())
        limit = SUMMARY_WORD_LIMITS[name]
        if len(text.split()) > limit:
            builder.refuse(name, f"{len(text.split())} words, past the "
                                 f"{limit}-word limit; kept as written")
        target = general if name in GENERAL_FIELDS else hazards
        setattr(target, name, text)
