"""The Document Review suite's tasks: questions a reviewer asks, with checks.

WHY THIS EXISTS. Until 2026-09-26 every change to the Document Review page was
judged on the one example that prompted it (the rebar sheet, the GCE callouts),
so a change could fix one question and quietly break three others. These tasks
are the fixed, varied set every change is now scored on.

WHAT A TASK IS. A question as a reviewer would type it, the document(s) it is
asked of, and CHECKS on the answer (and on any file the agent produced). The
checks are deterministic wherever they can be — a value the answer must state,
a set of sheets it must list, a file it must write — so a score means the same
thing on every run and needs no model to grade it.

WHERE THE TRUTH COMES FROM. Every expected value below was read by hand off the
document and the source is recorded in the task's ``truth`` note:

* the ten Mecklenburg County standard-detail sheets
  (``module_work/drawing_ground_truth/mecklenburg/``, public documents) are
  CAD plots whose lettering is drawn as lines — no text layer at all — and the
  truth is the text of the DWG they were plotted from (``*.truth.json``);
* the three UFCs are public-domain DoD criteria (``geotech-references/docs``):
  one text manual whose key table is an image, one scanned manual whose lists
  and figures have no text layer, one 228-page manual;
* the two calculation packages are the app's own sample output
  (``sample_calc_package.pdf``, ``sample_pdfs/retaining_walls.pdf``);
* two SYNTHETIC documents come from ``planlens.testing`` because no public
  document carries review markups or a duplicated page. They are the smallest
  part of the suite on purpose: planlens' tools were built against them, so a
  suite made of them would repeat the overfitting it exists to catch;
* one synthetic geotechnical REPORT (``review_eval/report_fixture.py``,
  2026-10-08) for the coverage tasks: whether every data page of a report
  was read can only be scored on a report whose every page is known by
  construction, and no public report the suite may carry has the shape that
  was lost (new logs as vector pages, old ones as scans, a long laboratory
  appendix). Its three tasks run on both pages (``Task.page``).

SPLITS. Everything here is ``open``: builders may read it. A BLIND set — tasks
the people changing the harness never see — belongs in a private task file on
the owner's SharePoint (``load_tasks(extra=...)``), written by someone who is
not tuning the harness. Tester feedback becomes a new task there, not a new
prompt rule.

Checks that every task gets unless it says otherwise (``AUTO_CHECKS``): the
answer never gives up with "too blurry" / "send a higher-resolution file", and
never cites "page 0" (a tool's 0-based index leaking into a citation).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from funhouse_agent.review_eval import report_fixture as _RF

#: Categories a task can be in (what the reviewer is doing).
CATEGORIES = ("orient", "summarize", "locate", "count", "check", "compare",
              "markups", "produce", "extract")

#: What kind of document the task is asked of.
DOC_TYPES = ("drawing_stroke", "drawing_set", "criteria_text",
             "criteria_scanned", "long_text", "calc_package", "markup_set",
             "submittal", "report")

#: Which page of the app a task is asked on: the Document Review page (the
#: suite's home) or the GeotechStaffEngineer page (the geotech agent, its
#: analysis modules and its DIGGS writer) - plan W4 asks for both harnesses.
PAGES = ("review", "geotech")

#: Giving up on legibility: calling the file too blurry, or asking for a
#: better one. ("I zoomed in at higher resolution" is not giving up.)
GIVE_UP_PHRASES = (
    "too blurry",
    {"re": r"(?:send|provide|upload|share|supply|need|request|obtain|get)"
           r"\w*\b[^.\n]{0,60}\b(?:higher[- ]resolution|better(?:[- ]quality)?"
           r"|clearer|sharper|higher[- ]quality|more legible)\s+"
           r"(?:copy|scan|version|file|pdf|image|print)"},
)

#: Added to every task unless ``auto_checks=False``.
AUTO_CHECKS: List[Dict[str, Any]] = [
    {"type": "not_contains", "terms": list(GIVE_UP_PHRASES),
     "label": "does not give up on legibility"},
    {"type": "not_contains", "terms": [{"re": r"\b(?:page|p\.|pg\.?)\s*0\b"}],
     "label": "no 0-based page citation"},
]


@dataclass
class Task:
    """One question, the document(s) it is asked of, and its checks."""
    id: str
    question: str
    documents: List[str]
    category: str
    doc_type: str
    checks: List[Dict[str, Any]]
    truth: str = ""
    split: str = "open"
    followups: List[str] = field(default_factory=list)
    auto_checks: bool = True
    #: The app page the task is asked on (:data:`PAGES`).
    page: str = "review"

    def all_checks(self) -> List[Dict[str, Any]]:
        return list(self.checks) + (list(AUTO_CHECKS) if self.auto_checks
                                    else [])

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "question": self.question,
                "documents": list(self.documents), "category": self.category,
                "doc_type": self.doc_type, "checks": list(self.checks),
                "truth": self.truth, "split": self.split,
                "followups": list(self.followups),
                "auto_checks": self.auto_checks, "page": self.page}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(id=str(d["id"]), question=str(d["question"]),
                   documents=list(d.get("documents") or []),
                   category=str(d.get("category", "locate")),
                   doc_type=str(d.get("doc_type", "other")),
                   checks=list(d.get("checks") or []),
                   truth=str(d.get("truth", "")),
                   split=str(d.get("split", "open")),
                   followups=list(d.get("followups") or []),
                   auto_checks=bool(d.get("auto_checks", True)),
                   page=str(d.get("page", "review")))


# ---------------------------------------------------------------------------
# Term helpers (alternatives for the ways an answer writes the same thing)
# ---------------------------------------------------------------------------

#: A number must stand on its own: not the tail of a bigger number, an id or a
#: range written "5-4" (a quote before it is fine: 'ALL CONCRETE ... 3600').
_LEAD = r"(?<![\w.,/-])"


def _num(n: str) -> Dict[str, str]:
    """``n`` as a whole number or code ("3600" is not in "36000", "8-4" is not
    in "5-818-4", "1.5" is not in "1.55")."""
    return {"re": rf"{_LEAD}{re.escape(n)}(?!\d)"}


def _ft(n: str) -> Dict[str, str]:
    """``n`` feet, in any of the ways an answer writes it (and not as the tail
    of a bigger number: "12 ft" is not "2 ft"; "note 4'" is not 4 ft)."""
    return {"re": rf"{_LEAD}(?<!note )(?<!notes ){re.escape(n)}\s*"
                  rf"(?:'|ft\b|-ft\b|feet|-foot|foot)"}


def _inch(n: str) -> Dict[str, str]:
    """``n`` inches, in any of the ways an answer writes it ("5-4 in the
    manual" is not 4 inches)."""
    return {"re": rf"{_LEAD}{re.escape(n)}\s*(?:\"|inch(?:es)?\b|-inch\b|"
                  rf"in\.|-in\b|in\b(?!\s+(?:the|a|an|this|that|these|those|"
                  rf"which|each|all|order|addition|accordance|place|front)\b))"}


def _pct(n: str) -> Dict[str, str]:
    """``n`` percent ("5%" must not match inside "15%")."""
    return {"re": rf"{_LEAD}{re.escape(n)}\s*(?:%|percent)"}


def _min(n: str, unit: str = "ft") -> Dict[str, str]:
    """``n`` feet (``unit="in"``: inches) stated as a MINIMUM, in the ways an
    answer writes it: "10' MIN.", "10'-0\\" MIN." (drawing notation), "10 ft
    (minimum)", "min. 10 ft", "at least 10 feet" ("5' MAX" is not a minimum
    of 5 ft; "110' min" is not 10'; "10'-6\\" MIN." is not 10'; in "L=10'
    MIN., 5' MAX" the MIN. belongs to the 10)."""
    if unit == "ft":
        # feet, then zero inches as a drawing writes it ("10'-0\"") or none;
        # any other inches make it a different length.
        u = (r"(?:'|ft\b|-ft\b|feet|-foot|foot)"
             r"(?:\s*-?\s*0\s*(?:\"|inch(?:es)?\b|in\.|in\b))?"
             r"(?!\s*-?\s*\d)")
    else:
        u = r"(?:\"|inch(?:es)?\b|-inch\b|in\.|-in\b|in\b)"
    num = rf"{_LEAD}{re.escape(n)}\s*{u}"
    return {"re": rf"{num}[^\w\n]{{0,4}}(?:min\b|min\.|minimum)|"
                  rf"(?:\bmin\b\.?|\bminimum|\bat least|\bno less than|"
                  rf"\bnot less than)[^.,;)\]\n\d]{{0,25}}{num}"
                  rf"(?![^\w\n]{{0,4}}max)"}


_FT_UNIT = r"(?:'|ft\b|-ft\b|feet|-foot|foot)"


def _lettered_min(letter: str, n: str) -> Dict[str, str]:
    """A lettered value ("W = 5 ft") in a run of lettered values that the
    answer then calls minimums all together ("L = 10 ft, W = 5 ft, X = 7 ft
    (all minimums)"). "W=5' MAX" or "W=5' (as drawn)" is not."""
    one = rf"\b{letter}\s*=\s*{re.escape(n)}\s*{_FT_UNIT}"
    other = rf"(?:\w+\s+)?[a-z]\s*=\s*\d+(?:\.\d+)?\s*{_FT_UNIT}"
    run = rf"(?:\s*(?:,|;|\band\b|&)?\s*{other})*"
    together = (r"\s*[,;:-]?\s*\(?\s*(?:(?:all|each|both)\s+(?:are\s+)?"
                r"(?:a\s+)?(?:minimums?\b|min\b)|(?:are\s+)?minimums\b)")
    return {"re": one + run + together}


#: 3600 psi, written either way.
_3600 = [_num("3600"), _num("3,600")]

_MECK_IDS = ["10.17A", "10.25A", "10.31A", "11.01", "20.00A", "20.00B",
             "21.01", "30.00", "30.01", "50.03"]

#: How an answer may write a sheet id: with or without the letter's case, with
#: "STD. NO." around it. Matching is case-insensitive already, so aliases only
#: cover real spelling variants.
_MECK_ALIASES = {"20.00A": ["20.00A", "20.00 A", "20.00-A"],
                 "20.00B": ["20.00B", "20.00 B", "20.00-B", "20.00A/B",
                            "20.00A & B", "20.00A and B", "20.00A-B"]}

# Coverage tasks on the long manuals (check type ``labelled_set``). Each item
# counts only when the answer names it by its ID TOGETHER WITH ITS TITLE
# (the title in the text the id owns, before the next id): a topic list with
# no ids, letters matched to the wrong topics and an answer that ids half
# the items all fall short. A bare number is an id only where it cannot be
# something else: in a manual "12-3" is also a printed page and a paragraph,
# and "A" is a word.

def _bare(token: str) -> str:
    """A bare letter or number used as an id in a list or table ("- A:",
    "A. ", "| A |", "A - ", "A ("), never inside a word or "(a)"."""
    return (rf"(?:^|(?<=[\s|*•]))(?<!\(){token}"
            rf"(?=\s*(?:[.:)|](?:\s|$)|[-–]\s|\())")


#: Words before a "12-3" that make it a page, figure or paragraph number.
_NOT_A_TABLE = "".join(
    rf"(?<!{re.escape(w)} )" for w in (
        "p.", "pp.", "pg.", "page", "pages", "printed", "sheet", "figure",
        "fig.", "figures", "paragraph", "para.", "section", "sec.",
        "equation", "eq.", "chapter"))

#: UFC 3-260-02's fourteen appendices: id forms and title words.
_UFC260_APPENDIX_TITLES = {
    "A": ["references"],
    "B": ["design analysis"],
    "C": ["contract drawing", "contract drawings"],
    "D": ["waiver", "waivers"],
    "E": ["flexural strength"],
    "F": ["strain repetitions", "strain repetition"],
    "G": ["cylindrical specimens", "cylindrical specimen",
          "specimen preparation", "preparation of bituminous"],
    "H": ["dynamic modulus"],
    "I": ["estimating the modulus", "estimating modulus",
          "estimate the modulus", "estimation of the modulus",
          "estimation of modulus", "estimated modulus"],
    "J": ["unbound"],
    "K": ["stabilized soils", "stabilized soil", "stabilised soils",
          "stabilised soil", "fatigue characteristics"],
    "L": ["modulus of subgrade", "subgrade resilient modulus",
          "resilient modulus of subgrade", "subgrade material",
          "subgrade materials"],
    "M": ["fatigue life"],
    "N": ["resilient modulus of granular",
          "resilient modulus of the granular", "granular base resilient",
          "granular base material"],
}
_UFC260_APPENDICES = {
    f"appendix {c}": {
        "ids": [rf"\b(?:appendix|app\.|appx\.?)\s*{c.lower()}\b(?!-\d)",
                _bare(c.lower())],
        "titles": titles}
    for c, titles in _UFC260_APPENDIX_TITLES.items()}

#: The eight tables of UFC 3-260-02 Chapter 12: id forms and title words.
_UFC260_CH12_TABLE_TITLES = {
    1: ["mixed traffic design", "example of mixed traffic"],
    2: ["stress-strength ratios", "stress-strength ratio",
        "stress strength ratios", "stress strength ratio",
        "allowable coverages"],
    3: ["fatigue damage summary", "summary sheet"],
    4: ["pass-to-coverage", "pass to coverage", "pass-coverage"],
    5: ["channelized", "primary traffic"],
    6: ["unchannelized", "un-channelized", "secondary traffic"],
    7: ["transverse contraction joints", "transverse contraction joint",
        "joint spacing", "spacing of transverse"],
    8: ["dowel", "dowels"],
}
_UFC260_CH12_TABLES = {
    f"table 12-{n}": {
        "ids": [rf"\b(?:tables?|tbl\.?)\s*12-{n}(?![\d]|\.\d)",
                rf"(?<![\w.-]){_NOT_A_TABLE}12-{n}(?![\d]|\.\d|-\d)"],
        "titles": titles}
    for n, titles in _UFC260_CH12_TABLE_TITLES.items()}

#: The nine ASCE 7 chapters UFC 3-301-01 Chapter 3 modifies (its section
#: headings 3-1 to 3-9, checked against the PDF's text): id forms and title
#: words. "chapter 1" is not "chapter 11".
_ASCE7_CHAPTER_TITLES = {
    1: ["basic requirements", "classification of buildings",
        "risk categor"],
    2: ["combinations of loads", "load combinations"],
    6: ["tsunami"],
    7: ["snow"],
    11: ["seismic design criteria"],
    12: ["building structures"],
    13: ["nonstructural", "non-structural"],
    15: ["nonbuilding", "non-building"],
    26: ["wind", "enclosure classification"],
}
_ASCE7_CHAPTERS = {
    f"ASCE 7 chapter {n}": {
        "ids": [rf"\b(?:chapter|ch\.?)\s*{n}(?![\d]|\.\d|-\d)",
                rf"(?<=\|)\s*{n}(?=\s*\|)"],
        "titles": titles}
    for n, titles in _ASCE7_CHAPTER_TITLES.items()}
#: A bare list of chapter numbers ("ASCE 7 chapters 1, 2, 6, 7, 11, 12, 13,
#: 15 and 26") names them too.
_ASCE7_ENUM_LEAD = r"\b(?:chapters?|chs?\.)\s*"
_ASCE7_ENUM_TOKENS = {f"ASCE 7 chapter {n}": str(n)
                      for n in _ASCE7_CHAPTER_TITLES}


# ---------------------------------------------------------------------------
# The open set
# ---------------------------------------------------------------------------

OPEN_TASKS: List[Task] = [
    # --- single drawing sheets, lettering drawn as lines (no text layer) ----
    Task(
        id="meck-driveway-notes",
        question=("I'm checking this driveway detail. What concrete strength "
                  "does it require, and what is the limit on the breakover "
                  "'A'?"),
        documents=["meck_10.25a"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": [_3600, _pct("8")]}],
        truth="10.25A NOTES 1 and 4 (DWG text): 'ALL CONCRETE TO BE 3600 "
              "P.S.I.'; '\"A\" BREAKOVER SHALL BE 8% OR LESS'."),
    Task(
        id="meck-ramp-slopes",
        question=("For this curb ramp standard, what are the maximum ramp "
                  "(running) slope and the maximum cross slope?"),
        documents=["meck_10.31a"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": [_num("8.33"), _pct("2.1")]}],
        truth="10.31A NOTE 4: 'CROSS SLOPE CANNOT EXCEED 2.1% MAX. RAMP SLOPE "
              "CANNOT EXCEED 8.33% MAX.'"),
    Task(
        id="meck-ramp-warning-mat",
        question=("What does this sheet call for at the detectable warning "
                  "surface, and which other standard does it point to for "
                  "it?"),
        documents=["meck_10.31a"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": ["10.35B"]},
                {"type": "contains_any", "terms": [{"re": r"\bmats?\b"}]}],
        truth="10.31A: 'DETECTABLE WARNING MAT PER MCLDS #10.35B'; '2\" MAX "
              "FROM BACK OF CURB TO DETECTABLE WARNING SURFACE MAT'."),
    Task(
        id="meck-pavement-section",
        question=("What pavement section does this standard require for a "
                  "local residential street — surface, intermediate and base "
                  "courses?"),
        documents=["meck_11.01"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": ["S9.5B", "B25.0C", _inch("8")]}],
        truth="11.01 TYPICAL PAVEMENT SECTION: 1 1/2\" S9.5B surface and "
              "intermediate; '8\" COMPACTED AGGREGATE BASE COURSE, OR 4\" ACBC "
              "TYPE B25.0C (OR 4\" BCBC TYPE HB, INDIVIDUAL APPROVAL "
              "REQUIRED)'."),
    Task(
        id="meck-revision-block",
        question=("List the revisions recorded in this sheet's revision "
                  "block, with their dates and what changed."),
        documents=["meck_11.01"], category="summarize",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [[_num("5/09"), _num("05/09"), "may 2009"],
                           [_num("11/16"), "nov 2016", "november 2016"],
                           [_num("9/22"), _num("09/22"), "sept 2022",
                            "september 2022"]]}],
        truth="11.01 REVISIONS: 1 5/09 REVISED PVMT. SECT.; 2 11/16 REVISED "
              "SURFACE COURSE AMOUNT; 3 9/22 REVISED PAVEMENT MIX TYPES "
              "(S9.5B, B25.0C), ADDED NOTE #3."),
    Task(
        id="meck-row-sidewalk",
        question=("What is the minimum right-of-way width for a local "
                  "residential street on this sheet, and how far must the "
                  "sidewalk be from the back of curb?"),
        documents=["meck_11.01"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": [_ft("50"), _ft("4")]}],
        truth="11.01: '50' R/W (MINIMUM)'; NOTE 1 'SIDEWALK SHALL BE PROVIDED "
              "ON BOTH SIDES OF STREET, MINIMUM 4' FROM BACK OF CURB.'"),
    Task(
        id="meck-underdrain",
        question=("What pipe does this bioretention detail specify for the "
                  "underdrain, and how are the perforations arranged?"),
        documents=["meck_21.01"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [_inch("6"), "schedule 40", ["m278", "m 278"],
                           ["3/8"]]},
                {"type": "contains_any", "terms": ["hdpe", "m252", "m 252"]}],
        truth="21.01 NOTE 5: 'MIN. 6\" PERFORATED SCHEDULE 40 PVC (PER AASHTO "
              "M278) OR DOUBLE WALL HDPE (PER AASHTO M252). PERFORATIONS "
              "SHOULD BE 3/8\" SPACED 3\" ON CENTER ALONG 4 LONGITUDINAL ROWS "
              "SPACED 90 DEGREES APART.'"),
    Task(
        id="meck-bioretention-access",
        question=("What access and easement requirements does this "
                  "bioretention standard set?"),
        documents=["meck_21.01"], category="summarize",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [_ft("20"), _ft("12"), _pct("15"), _pct("5")]}],
        truth="21.01 NOTE 1: min 20 foot access easement to a dedicated public "
              "right of way; access road min 12' stabilized width, max long. "
              "grade 15%, max cross-slope 5%; 10-foot perimeter maintenance "
              "easement."),
    Task(
        id="meck-sediment-trap-criteria",
        question=("What are the design criteria in this sheet's data block "
                  "for a temporary sediment trap?"),
        documents=["meck_30.01"], category="summarize",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [_3600, _num("435"), _num("2:1")]},
                {"type": "contains_any",
                 "terms": [{"re": r"(?<![\d.])5\s*(?:ac\b|acres?\b)"},
                           {"re": r"(?:<|less than|under|up to)\s*5(?![\d.])"}]}],
        truth="30.01 TEMPORARY SEDIMENT TRAP DESIGN CRITERIA: drainage area "
              "< 5 AC.; min length to width ratio 2:1; min volume 3600 cu ft "
              "per acre disturbed; surface area 435 sq ft per cfs Q10."),
    Task(
        id="meck-monument",
        question=("How deep is the concrete control monument on this sheet, "
                  "and what materials go into it?"),
        documents=["meck_50.03"], category="summarize",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": [_inch("30")]},
                {"type": "contains_any",
                 "terms": ["brass", "iron pin", "reinforcing"]}],
        truth="50.03 TYPICAL CONCRETE CONTROL MONUMENT: 30\" deep; brass plate "
              "with grooved dowel, iron pin, steel reinforcing rods; ferrous "
              "materials required."),
    Task(
        id="meck-curb-types",
        question="Which curb and gutter types are detailed on this sheet?",
        documents=["meck_10.17a"], category="summarize",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [[_num("1'-6"), _num("1-6"), _inch("18")],
                           [_num("2'-6"), _num("2-6"), _inch("30")],
                           [_num("2'-0"), _num("2-0"), _inch("24")]]}],
        truth="10.17A titles: 1'-6\" STANDARD CURB AND GUTTER; STANDARD 2'-6\" "
              "CURB AND GUTTER; 2'-0\" STANDARD CURB & GUTTER."),

    # --- dimensions and leaders: found by geometry, read by looking ---------
    Task(
        id="meck-trap-dimensions",
        question=("What minimum dimensions does this sediment trap detail "
                  "show? List each one with how it is labelled."),
        documents=["meck_30.01"], category="locate",
        doc_type="drawing_stroke",
        # Each lettered value must be given AS A MINIMUM, as it is labelled:
        # "W=5' MAX." or "W=5' (as drawn)" is not what the sheet says.
        checks=[{"type": "contains_all",
                 "terms": [[_min("10"), _lettered_min("l", "10")],
                           [_min("5"), _lettered_min("w", "5")],
                           [_min("7"), _lettered_min("x", "7")]],
                 "label": "the lettered minimums L, W and X"},
                {"type": "contains_all",
                 "terms": [_min("1.5"), _min("21", "in")],
                 "label": "the two unlettered minimums"}],
        truth="30.01 DIMENSION entities (DWG): 'L=10' MIN.', 'W=5' MIN.', "
              "'X=7' MIN.', '1.5' MIN.', '21\" MIN.'; the others are '5' "
              "MAX', '5' MAX FILL', '2' TO 3.5'', 'H' and 'T='. No text "
              "layer: every value is read by looking."),
    Task(
        id="meck-bioretention-section-dims",
        question=("What minimum dimensions are shown on Section A-A of this "
                  "bioretention detail?"),
        documents=["meck_21.01"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all", "terms": [_min("10"), _min("4")]}],
        truth="21.01 SECTION A-A DIMENSION entities (DWG): '10' MIN.' and "
              "'4' MIN.' across the section; its three other dimensions "
              "carry no text of their own (the notes beside them give 1'-0\" "
              "ponding, 2'-0\" to 4'-0\" filter media and a 1'-0\" gravel "
              "layer). No text layer: every value is read by looking."),
    Task(
        id="meck-ramp-detail-callouts",
        question=("In the 2'-6\" curb and gutter ramp detail on this sheet, "
                  "what do the leader callouts point out? Give the flowline "
                  "depth it calls for."),
        documents=["meck_10.31a"], category="locate",
        doc_type="drawing_stroke",
        checks=[{"type": "contains_all",
                 "terms": [[_inch("3/4"), {"re": "¾"}, _inch("0.75")],
                           "flowline", "edge of pavement",
                           {"re": r"match(?:es|ing)?\s+(?:the\s+)?ramp\s+"
                                  r"slope"}]},
                {"type": "not_contains",
                 "terms": [{"re": r"(?<![\d/.-])34\s*(?:\"|in\b|inch)"}],
                 "label": "does not read the stacked 3/4 as 34"}],
        truth="10.31A 2'-6\" CURB AND GUTTER RAMP DETAIL MULTILEADER entities "
              "(DWG): '3/4\" FLOWLINE DEPTH AT RAMP LOCATION', 'EDGE OF "
              "PAVEMENT ELEVATION', 'TYP. MATCH RAMP SLOPE' (the flowline "
              "callout repeats on the ramp section). The sheet's hidden CAD "
              "text drops the stacked fraction's slash, so only looking gives "
              "the depth right."),

    # --- the ten sheets as one set --------------------------------------------
    Task(
        id="set-sheet-index",
        question=("Make me a sheet index for this set: each page's standard "
                  "number and title."),
        documents=["meck_set"], category="summarize",
        doc_type="drawing_set",
        checks=[{"type": "set_match", "vocabulary": _MECK_IDS,
                 "expected": _MECK_IDS, "aliases": _MECK_ALIASES,
                 "min_recall": 0.9, "min_precision": 0.9}],
        truth="STD. NO. in each sheet's title block (DWG text): 10.17A curb and "
              "gutter; 10.25A residential driveway; 10.31A perpendicular curb "
              "ramp; 11.01 local residential street; 20.00A and 20.00B NCDOT "
              "drainage standards approved for the county; 21.01 bioretention; "
              "30.00 special erosion control requirements; 30.01 temporary "
              "sediment trap; 50.03 concrete control monument."),
    Task(
        id="set-3600-psi",
        question="Which sheets in this set specify 3600 psi concrete?",
        documents=["meck_set"], category="count", doc_type="drawing_set",
        checks=[{"type": "set_match", "vocabulary": _MECK_IDS,
                 "expected": ["10.25A", "20.00A", "20.00B"],
                 "aliases": _MECK_ALIASES,
                 "min_recall": 1.0, "min_precision": 0.75}],
        truth="10.25A NOTE 1; 20.00A and 20.00B NOTE 1. Trap: 30.01 prints "
              "'3600 (CU. FT. PER AC.)', a volume, not a concrete strength."),
    Task(
        id="set-cross-references",
        question=("List the other standards or documents these sheets refer "
                  "the reader to, and say which of them are not included in "
                  "this set."),
        documents=["meck_set"], category="compare", doc_type="drawing_set",
        checks=[{"type": "set_match",
                 "vocabulary": ["10.35B", "20.17", "30.19", "11.51", "840.54",
                                "6.60"],
                 "expected": ["10.35B", "20.17", "30.19", "11.51", "840.54",
                              "6.60"],
                 "min_recall": 0.66, "min_precision": 0.0}],
        truth="10.25A -> STD 10.17 (in the set as 10.17A); 10.31A -> MCLDS "
              "#10.35B; 20.00A/B -> MCLDS 20.17 and STD 840.54; 30.01 -> STD "
              "#30.19 and NCESCPDM #6.60; 11.01 -> DETAIL 11.51 and sections "
              "1.A.18/1.E.4/1.F; 21.01 -> BMP design manual. Only 10.17 is in "
              "the set."),
    Task(
        id="set-find-bioretention",
        question=("Which sheet covers bioretention, and what depth of filter "
                  "media does it show?"),
        documents=["meck_set"], category="locate", doc_type="drawing_set",
        checks=[{"type": "contains_all",
                 "terms": [_num("21.01"), [_num("4'-0"), _ft("4")]]}],
        truth="21.01 SECTION A-A: '2'-0\" TO 4'-0\" FILTER MEDIA'."),
    Task(
        id="set-ncdot-vs-county",
        question=("What concrete strength does NCDOT require for drainage "
                  "structures compared with the county, according to this "
                  "set? Cite the sheet."),
        documents=["meck_set"], category="check", doc_type="drawing_set",
        checks=[{"type": "contains_all",
                 "terms": [[_num("2500"), _num("2,500")], _3600]},
                {"type": "contains_any",
                 "terms": ["20.00A", "20.00B", "20.00 A", "20.00 B"]}],
        truth="20.00A/20.00B NOTE 1: 'NCDOT REQUIRES CLASS B CONCRETE "
              "(2500PSI). THE COUNTY REQUIRES 3600 PSI CONCRETE STRENGTH @ 28 "
              "DAYS.'"),

    # --- a text manual whose key table is an image --------------------------
    Task(
        id="ufc04-density",
        question=("What compaction does this manual require for backfill "
                  "beneath structures, compared with open areas where no "
                  "structures will be built? Cite the page."),
        documents=["ufc_3_220_04fa"], category="locate",
        doc_type="criteria_text",
        checks=[{"type": "contains_all", "terms": [_num("90"), _num("95")]},
                {"type": "cites", "pages": ["29"], "printed": ["5-2"]}],
        truth="PDF page 29 (printed 5-2), para 5-x(4) Density requirements: "
              "open areas 90 percent of CE 55; beneath structures "
              "cohesionless 95-100 percent, cohesive at least 95 percent."),
    Task(
        id="ufc04-table-5-1",
        question=("According to Table 5-1, for compacted semipervious and "
                  "impervious soils using a sheepsfoot roller, how many "
                  "passes and what compacted lift thickness are "
                  "recommended?"),
        documents=["ufc_3_220_04fa"], category="locate",
        doc_type="criteria_text",
        checks=[{"type": "contains_all",
                 "terms": [[_num("4-8"), _num("4 to 8"), "four to eight"],
                           _inch("6")]}],
        truth="PDF page 30 (printed 5-3), Table 5-1 is an IMAGE (no text "
              "layer): Semipervious and Impervious, Compacted, Sheepsfoot "
              "roller: 4-8 passes, compacted lift 6 in."),
    Task(
        id="ufc04-supersedes",
        question="What does this UFC supersede, and when was that dated?",
        documents=["ufc_3_220_04fa"], category="orient",
        doc_type="criteria_text",
        checks=[{"type": "contains_all", "terms": ["5-818-4", _num("1983")]}],
        truth="PDF page 2: 'This UFC supersedes TM 5-818-4, dated 1 June "
              "1983.'"),
    Task(
        id="ufc04-confined-zones",
        question=("For cohesive backfill in confined zones, what loose-lift "
                  "thickness and compaction equipment should be specified, "
                  "and what equipment does the manual say does not work?"),
        documents=["ufc_3_220_04fa"], category="locate",
        doc_type="criteria_text",
        checks=[{"type": "contains_all", "terms": [_inch("4"), "rammer"]},
                {"type": "contains_any",
                 "terms": ["two-by-four", "2x4", "2 x 4", "air tamper",
                           "pogo", "powder puff"]}],
        truth="PDF page 29: 'use of rammer compactors and a loose-lift "
              "thickness of not more than 4 inches should be specified... "
              "\"two-by-four\" wood rammers, or single air tampers... do not "
              "produce sufficient compaction.'"),

    # --- a scanned manual: lists and figures with no text layer --------------
    Task(
        id="ufc07-figure-1-1",
        question=("What does Figure 1-1 show, and what ground slope does the "
                  "text on that page say puts walls and foundations at risk "
                  "from downhill creep?"),
        documents=["ufc_3_220_07"], category="locate",
        doc_type="criteria_scanned",
        checks=[{"type": "contains_all", "terms": ["vertical", "diagonal"]},
                {"type": "contains_any",
                 "terms": [{"re": r"(?<![\d.])5\s*(?:°|degrees?\b|deg\b)"},
                           _pct("9")]}],
        truth="PDF page 13 (printed 1-3), image-only: Figure 1-1 'Examples of "
              "cracks in an exterior wall' (a. Vertical cracks; b. Diagonal "
              "and vertical cracks); text 'slopes greater than 5 degrees (9 "
              "percent)'."),
    Task(
        id="ufc07-drilled-shaft-table",
        question=("Which table covers inspection of drilled shafts, and on "
                  "what page does it appear?"),
        documents=["ufc_3_220_07"], category="locate",
        doc_type="criteria_scanned",
        checks=[{"type": "contains_all", "terms": [_num("8-3"), _num("8-4")]}],
        truth="PDF page 10 (printed iii), image-only list of tables: 'Table "
              "8-3 Inspection of Drilled Shafts ... 8-4'."),

    # --- a long manual -------------------------------------------------------
    Task(
        id="ufc301-changes",
        question=("How many changes have been issued to this UFC, with their "
                  "dates, and what earlier edition does it supersede?"),
        documents=["ufc_3_301_01"], category="orient", doc_type="long_text",
        checks=[{"type": "contains_all",
                 "terms": [_num("2019"), ["june 3, 2025", "3 june 2025",
                                    "june 2025", "2025-06-03"]]},
                {"type": "contains_any",
                 "terms": [{"re": r"\b(?:four|4)\s+(?:changes|amendments)"},
                           {"re": r"change\s*(?:no\.?\s*)?(?:4|four)\b"}]}],
        truth="PDF page 3 Record of Changes lists four changes: Change 1 Oct 2, "
              "2023; Change 2 Sept 4, 2024; Change 3 Feb 3, 2025; Change 4 "
              "June 3, 2025. Supersedes UFC 3-301-01 dated 1 October 2019."),

    # --- coverage: the honest answer needs most of a long manual ----------
    # (shape 2 of module_work/REVIEW_ARCHITECTURE.md). Each truth was read
    # off the PDF's own text layer and checked page by page; an answer that
    # lists only the first few items fails the recall floor.
    Task(
        id="ufc260-appendices",
        question=("Which appendices does this manual have, and what does each "
                  "one cover?"),
        documents=["ufc_3_260_02"], category="orient", doc_type="long_text",
        checks=[{"type": "labelled_set", "items": _UFC260_APPENDICES,
                 "min_recall": 0.85, "require_title": True,
                 "label": "names at least 12 of the 14 appendices by letter "
                          "with what each covers"}],
        truth="ufc_3_260_02_2001.pdf (538 pp), the APPENDIX heading on each "
              "appendix's first page (text layer): Appendix A References "
              "(PDF p. 431, printed A-1); Appendix B Airfield/heliport design "
              "analysis outline (439, B-1); Appendix C Recommended contract "
              "drawing outline for airfield/heliport pavements (446, C-1); "
              "Appendix D Waiver processing procedures (452, D-1); Appendix E "
              "Determination of flexural strength and modulus of elasticity "
              "of bituminous concrete (457, E-1); Appendix F Curves for "
              "determining effective strain repetitions (460, F-1); Appendix "
              "G Procedure for preparation of bituminous cylindrical "
              "specimens (482, G-1); Appendix H Procedure for determining the "
              "dynamic modulus of bituminous concrete mixtures (484, H-1); "
              "Appendix I Procedure for estimating the modulus of elasticity "
              "of bituminous concrete (487, I-1); Appendix J Procedure for "
              "determining the modulus of elasticity of unbound granular "
              "base and subbase course materials (491, J-1); Appendix K "
              "Procedure for determining the flexural modulus and fatigue "
              "characteristics of stabilized soils (495, K-1); Appendix L "
              "Procedure for determining resilient modulus of subgrade "
              "material (500, L-1); Appendix M Procedures for determining "
              "the fatigue life of bituminous concrete (522, M-1); Appendix "
              "N Procedure for determining the resilient modulus of granular "
              "base material (528, N-1). No other appendix letter appears "
              "anywhere in the text."),
    Task(
        id="ufc260-ch12-tables",
        question=("List every table in Chapter 12 (plain concrete pavements) "
                  "of this manual, with its number and title."),
        documents=["ufc_3_260_02"], category="summarize",
        doc_type="long_text",
        checks=[{"type": "labelled_set", "items": _UFC260_CH12_TABLES,
                 "min_recall": 0.85, "require_title": True,
                 "label": "numbers and titles at least 7 of the chapter's 8 "
                          "tables"}],
        truth="ufc_3_260_02_2001.pdf, Chapter 12 runs PDF pp. 191-267 "
              "(printed 12-1 to 12-77), mostly design-curve figures; its "
              "table captions (text layer, 'Table 12-N' with the title on "
              "the next line): 12-1 Example of Mixed Traffic Design (PDF p. "
              "196, printed 12-6); 12-2 Stress-Strength Ratios and Allowable "
              "Coverages (198, 12-8); 12-3 Fatigue Damage Summary Sheet for "
              "Mixed Traffic (200, 12-10); 12-4 Pass-to-Coverage Ratios (201, "
              "12-11); 12-5 Design Example for Primary (Channelized) Traffic "
              "Areas (205, 12-15); 12-6 Design Example for Secondary "
              "(Unchannelized) Traffic Areas (207, 12-17); 12-7 Recommended "
              "Spacing of Transverse Contraction Joints (211, 12-21); 12-8 "
              "Dowel Size and Spacing for Construction, Contraction, and "
              "Expansion Joints (212, 12-22). No 'Table 12-9' or higher is "
              "mentioned anywhere in the text."),
    Task(
        id="ufc301-asce7-chapters",
        question=("Chapter 3 of this UFC modifies ASCE 7. Which ASCE 7 "
                  "chapters does it modify, what is each one about, and "
                  "which section of the UFC covers it?"),
        documents=["ufc_3_301_01"], category="summarize",
        doc_type="long_text",
        # A chapter number alone counts (the list is what is asked for
        # first), but not one said with another chapter's subject.
        checks=[{"type": "labelled_set", "items": _ASCE7_CHAPTERS,
                 "min_recall": 0.85, "require_title": False,
                 "enum_lead": _ASCE7_ENUM_LEAD,
                 "enum_tokens": _ASCE7_ENUM_TOKENS,
                 "label": "names at least 8 of the 9 ASCE 7 chapters, none "
                          "with another chapter's subject"}],
        truth="UFC_3-301-01_2023_c4.pdf, Chapter 3 runs PDF pp. 65-89 "
              "(printed 43-67); its section headings (text layer, 'ASCE 7-22 "
              "CHAPTER N'): 3-1 ASCE 7-22 Chapter 1 General (basic "
              "requirements, classification of buildings; PDF p. 65); 3-2 "
              "Chapter 2 Combinations of Loads (p. 65); 3-3 Chapter 6 "
              "Tsunami Loads (p. 68); 3-4 Chapter 7 Snow Loads (p. 69); 3-5 "
              "Chapter 11 Seismic Design Criteria (p. 69); 3-6 Chapter 12 "
              "Seismic Design Requirements for Building Structures (p. 70); "
              "3-7 Chapter 13 Seismic Design Requirements for Nonstructural "
              "Components (p. 81); 3-8 Chapter 15 Seismic Design "
              "Requirements for Nonbuilding Structures (p. 87); 3-9 Chapter "
              "26 Wind Loads: General Requirements (enclosure "
              "classification; p. 89)."),

    # --- calculation packages ------------------------------------------------
    Task(
        id="calc-wall-check",
        question=("Check this retaining wall calculation package: does the "
                  "design meet its own stability requirements? Point out "
                  "anything in the summary the calculations do not "
                  "support."),
        documents=["calc_retaining_wall"], category="check",
        doc_type="calc_package",
        checks=[{"type": "contains_all", "terms": [_num("1.185"), _num("1.5")]},
                {"type": "contains_any",
                 "terms": ["fail", "does not meet", "not meet", "inadequate",
                           "not satisfied", "below the required"]},
                {"type": "contains_any",
                 "terms": [{"re": r"(?<![\d.])99\.9"}],
                 "label": "flags the unsupported bearing value"}],
        truth="retaining_walls.pdf: sliding FOS 1.185 < required 1.5 (FAIL); "
              "overturning 2.733 >= 2.0 (OK); the summary lists Bearing "
              "'99.900 / N/A / OK' with no bearing calculation behind it."),
    Task(
        id="calc-bearing-consistency",
        question=("Does the allowable bearing pressure reported in this "
                  "package follow from its own ultimate capacity and factor "
                  "of safety? Show the check."),
        documents=["calc_bearing"], category="check",
        doc_type="calc_package",
        checks=[{"type": "contains_all",
                 "terms": [_num("398"), [_num("1195"), _num("1,195")]]}],
        truth="sample_calc_package.pdf page 4: q_ult 1,195.3 kPa / FS 3.0 = "
              "q_all 398.4 kPa (consistent)."),

    # --- synthetic: review markups, a stapled submittal -----------------------
    Task(
        id="fixture-markups",
        question=("List every reviewer comment and markup in this document: "
                  "who made it, what it says, and whether anyone has "
                  "responded."),
        documents=["fixture_review_document"], category="markups",
        doc_type="markup_set",
        checks=[{"type": "contains_all",
                 "terms": ["pile embedment", ["6 m", "6m", "6-m"],
                           "contractor b", "reviewer a"]}],
        truth="planlens.testing.document_fixtures: Reviewer A 'CONFIRM THE "
              "PILE EMBEDMENT SHOWN HERE.'; Contractor B reply 'Embedment "
              "revised to 6 m per updated calcs.'; Reviewer A stamp "
              "APPROVED."),
    Task(
        id="fixture-duplicate-page",
        question=("Is any page of this submittal a duplicate of another page? "
                  "Give the page numbers as a PDF viewer shows them."),
        documents=["fixture_submittal"], category="check",
        doc_type="submittal",
        checks=[{"type": "contains_all",
                 "terms": [["duplicat", "repeat", "identical", "same as"]]},
                {"type": "cites", "pages": [8], "label": "cites viewer page 8"},
                {"type": "cites", "pages": [3], "label": "cites viewer page 3"},
                {"type": "not_contains",
                 "terms": [{"re": r"\bpages?\s+7\b"}],
                 "label": "does not cite the 0-based index 7"}],
        truth="PDF page 8 repeats PDF page 3 (both carry the report's printed "
              "'Page 2'); planlens.testing.submittal_fixtures has them at "
              "0-based indexes 7 and 2."),

    # --- producing files -----------------------------------------------------
    Task(
        id="produce-markup",
        question=("Mark up a copy of this sheet for the designer: add one "
                  "comment on the ramp slope note asking them to confirm the "
                  "8.33% maximum applies over the full ramp length. Keep it "
                  "a draft."),
        documents=["meck_10.31a"], category="produce",
        doc_type="drawing_stroke",
        # No file-name check: the question names no file, and the Foundry run
        # (2026-10-02) failed three arms on '10.31A_designer_draft_markup.pdf'
        # for not containing "marked" — a check fault, not the agent's.
        checks=[{"type": "file_produced", "ext": ".pdf"},
                {"type": "pdf_markups", "min": 1, "pages": [0],
                 "text_contains": "8.33"}],
        truth="A marked-up copy with at least one comment on page 1 that "
              "mentions 8.33."),
    Task(
        id="set-long-rare-tag",
        question=("This is a 24-sheet set whose lettering is drawn as lines. "
                  "On which sheets is there an FPG penetration callout - an "
                  "FPG tag with a leader drawn from it, not the legend row? "
                  "Give the page numbers."),
        documents=["fixture_tags_long"], category="count",
        doc_type="drawing_set",
        checks=[{"type": "pages_listed", "expected": [4, 12, 20],
                 "n_pages": 24, "min_recall": 1.0, "min_precision": 0.75,
                 "label": "names pages 4, 12 and 20 and little else"}],
        truth=("Pages 4, 12 and 20 each carry one FPG callout. Every sheet "
               "repeats the legend (with an FPG row) and carries FBG callouts, "
               "the look-alike; the lettering is 0.06 in drawn as lines, so "
               "the only way to answer is to look at every sheet "
               "(planlens.testing.tag_fixtures, 24 pages, gce_growth 0). Field "
               "session 2026-10-01: an 85-sheet set where sheets nobody "
               "looked at were missed.")),
    Task(
        id="produce-circle-tags",
        question=("On the first sheet of this set, circle in red every GCE "
                  "penetration tag that has a leader drawn from it (the "
                  "callouts, not the legend row), each labelled GCE, in a "
                  "marked-up copy for the designer."),
        documents=["fixture_tags"], category="produce",
        doc_type="drawing_stroke",
        checks=[{"type": "file_produced", "ext": ".pdf"},
                {"type": "markups_on_targets", "fixture": "tags",
                 "text": "GCE", "kind": "callout", "page": 0,
                 "min_recall": 0.8, "min_precision": 0.8,
                 "label": "the rings sit on the GCE callouts"}],
        truth=("Sheet 1 (PDF page 1) carries 7 GCE callouts with leaders, "
               "plus a bare GCE, the legend's GCE row and look-alikes (GCG, "
               "QCE) that must not be circled; lettering 0.06 in, drawn as "
               "lines, so the tags are found by looking and each ring placed "
               "from a zoom in which its tag is legible (whole-sheet boxes "
               "were 20-90 pt off, 2026-10-07; planlens.testing.tag_fixtures, "
               "seed 11). Field session 2026-10-01: rings drawn at invented "
               "coordinates.")),
    Task(
        id="produce-memo",
        question=("Write a one-page Word memo summarizing the concrete "
                  "strength, curb ramp slope and street pavement requirements "
                  "in this set, citing the sheet for each."),
        documents=["meck_set"], category="produce", doc_type="drawing_set",
        checks=[{"type": "file_produced", "ext": ".docx"},
                {"type": "docx_contains",
                 "terms": [_3600, _num("8.33"), "S9.5B"]}],
        truth="10.25A/20.00A/20.00B 3600 psi; 10.31A 8.33% ramp slope; 11.01 "
              "S9.5B surface course."),

    # --- coverage: taking the data out of a whole report (plan W4) -----------
    # One synthetic report (review_eval/report_fixture.py) in the shape of
    # the 2026-10-06 field session's: new logs as vector pages, older logs
    # as scans, a laboratory appendix of a summary and 13 sheets. What is
    # scored is whether EVERY data page was read (from the run's activity,
    # not from anything the app's own coverage ledger says), whether the
    # answer states its coverage as counts, and values from pages a partial
    # reading misses: the scans, the newest boring, the last sheets.
    Task(
        id="report-extract-all",
        question=("Put the subsurface data in this geotechnical report into "
                  "tables: one row per boring (date, total depth, "
                  "groundwater, SPT N values with depth) and one row per "
                  "laboratory test result (boring, sample, depth, test, "
                  "values)."),
        documents=["fixture_report"], category="extract", doc_type="report",
        checks=[
            {"type": "pages_covered", "pages": list(_RF.TARGET_PAGES),
             "look_pages": list(_RF.SCANNED_LOG_PAGES), "min_fraction": 1.0,
             "label": "every log page and laboratory page was read"},
            {"type": "states_coverage",
             "label": "the answer states its coverage as counts"},
            {"type": "set_match", "vocabulary": list(_RF.NEW_BORINGS
                                                     + _RF.OLD_BORINGS),
             "expected": list(_RF.NEW_BORINGS + _RF.OLD_BORINGS),
             "min_recall": 1.0,
             "label": "names all six borings, 2026 and 2011"},
            {"type": "contains_all",
             "terms": [_num("43"), "shell", _num("41.7"), _num("1.94"),
                       ["1,850", _num("1850")]],
             "label": "values from the newest log, a scan and the last "
                      "sheets"},
        ],
        truth=("Borings of 2026 (vector logs, PDF pages 8-11): B-1 (two "
               "sheets, 14.9 m, water 3.1 m, N 9 to 52), B-2 (7.4 m, water "
               "2.8 m), B-3 (7.4 m, water 3.6 m, N 43 at 7.0 m). Borings of "
               "2011 (scanned logs, PDF pages 13-15): BH-1, BH-2 (grey silty "
               "SAND with shell fragments; water 3.4 m), BH-3. Laboratory "
               "(PDF pages 17-30): Atterberg B-1 S-2, B-2 S-3 (LL 32, PL 20), "
               "B-3 S-2; particle size incl. B-2 S-5 fines 41.7 %; compaction "
               "B-1 BULK-1 1.94 Mg/m3 at 11.6 %; in-place density B-2 S-2; "
               "soil chemistry B-1 S-3 and B-3 S-3; groundwater B-2 W-1 "
               "sulfate 1,850 mg/L. Coverage: 7 of 7 log pages and 14 of 14 "
               "laboratory pages read. (review_eval/report_fixture.py)")),
    Task(
        id="report-extract-diggs",
        question=("Extract the subsurface data in this geotechnical report - "
                  "its borings and its laboratory results - as a DIGGS "
                  "file."),
        documents=["fixture_report"], category="extract", doc_type="report",
        page="geotech",
        checks=[
            {"type": "pages_covered", "pages": list(_RF.TARGET_PAGES),
             "look_pages": list(_RF.SCANNED_LOG_PAGES), "min_fraction": 1.0,
             "label": "every log page and laboratory page was read"},
            {"type": "states_coverage",
             "label": "the answer states its coverage as counts"},
            {"type": "file_produced", "ext": ".xml"},
            {"type": "set_match", "vocabulary": list(_RF.NEW_BORINGS
                                                     + _RF.OLD_BORINGS),
             "expected": list(_RF.NEW_BORINGS + _RF.OLD_BORINGS),
             "min_recall": 1.0,
             "label": "names all six borings, 2026 and 2011"},
        ],
        truth=("A DIGGS 2.6 file of the six borings (B-1, B-2, B-3 of 2026 "
               "on vector logs; BH-1, BH-2, BH-3 of 2011 on scans) and the "
               "laboratory results of the 13 sheets; coverage 7 of 7 log "
               "pages and 14 of 14 laboratory pages read. Asked on the "
               "GEOTECH page, where subsurface.write_diggs writes and checks "
               "the file. (review_eval/report_fixture.py)")),
    Task(
        id="report-summary-vs-sheets",
        question=("Check this report's summary table of laboratory results "
                  "against the individual laboratory test sheets. Does every "
                  "value on the summary agree with its sheet? Name any "
                  "sample where they differ, with both values and the "
                  "pages."),
        documents=["fixture_report"], category="check", doc_type="report",
        checks=[
            {"type": "contains_all",
             "terms": ["B-2", "S-3", _num("32"), _num("20")],
             "label": "names B-2 S-3 and the limits 32 and 20"},
            {"type": "cites", "pages": [19],
             "label": "cites the Atterberg sheet, PDF page 19"},
            {"type": "pages_covered", "pages": list(_RF.LAB_PAGES),
             "min_fraction": 1.0,
             "label": "every laboratory page was read"},
        ],
        truth=("B-2 S-3 differs: Table B-1 (PDF page 17) gives LL 20 and PL "
               "32, the Atterberg sheet (PDF page 19) gives LL 32 and PL 20 "
               "(PI 12) - the summary's liquid and plastic limits are "
               "swapped. Every other value agrees; 14 of 14 laboratory pages "
               "read. (review_eval/report_fixture.py: the planted error has "
               "the shape of the field session's, where the summary table's "
               "PL exceeded its LL.)")),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tasks(extra: Optional[Iterable[Any]] = None,
               include_open: bool = True) -> List[Task]:
    """The shipped open tasks plus any from ``extra``.

    ``extra`` is a list of paths to JSON files (each a list of task dicts, or
    ``{"tasks": [...]}``) and/or task dicts — how a private blind set on the
    owner's SharePoint is added. A later task with an id already seen
    replaces the earlier one.
    """
    out: Dict[str, Task] = {}
    if include_open:
        for t in OPEN_TASKS:
            out[t.id] = t
    for item in extra or ():
        for d in _read_extra(item):
            t = Task.from_dict(d)
            out[t.id] = t
    return list(out.values())


def _read_extra(item: Any) -> List[Dict[str, Any]]:
    if isinstance(item, dict):
        return [item]
    if isinstance(item, Task):
        return [item.to_dict()]
    path = os.fspath(item)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    return list(data)


def select(tasks: Sequence[Task], ids: Optional[Iterable[str]] = None,
           categories: Optional[Iterable[str]] = None,
           split: Optional[str] = None) -> List[Task]:
    """Filter by id prefix, category and split (``None`` keeps all)."""
    ids = [str(i) for i in (ids or [])]
    cats = set(categories or [])
    out = []
    for t in tasks:
        if ids and not any(t.id == i or t.id.startswith(i) for i in ids):
            continue
        if cats and t.category not in cats:
            continue
        if split and t.split != split:
            continue
        out.append(t)
    return out


__all__ = ["Task", "OPEN_TASKS", "AUTO_CHECKS", "CATEGORIES", "DOC_TYPES",
           "PAGES", "GIVE_UP_PHRASES", "load_tasks", "select"]
