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
  suite made of them would repeat the overfitting it exists to catch.

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

#: Categories a task can be in (what the reviewer is doing).
CATEGORIES = ("orient", "summarize", "locate", "count", "check", "compare",
              "markups", "produce")

#: What kind of document the task is asked of.
DOC_TYPES = ("drawing_stroke", "drawing_set", "criteria_text",
             "criteria_scanned", "long_text", "calc_package", "markup_set",
             "submittal")

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

    def all_checks(self) -> List[Dict[str, Any]]:
        return list(self.checks) + (list(AUTO_CHECKS) if self.auto_checks
                                    else [])

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "question": self.question,
                "documents": list(self.documents), "category": self.category,
                "doc_type": self.doc_type, "checks": list(self.checks),
                "truth": self.truth, "split": self.split,
                "followups": list(self.followups),
                "auto_checks": self.auto_checks}

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
                   auto_checks=bool(d.get("auto_checks", True)))


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
        checks=[{"type": "file_produced", "ext": ".pdf",
                 "name_contains": "marked"},
                {"type": "pdf_markups", "min": 1, "pages": [0],
                 "text_contains": "8.33"}],
        truth="A marked-up copy with at least one comment on page 1 that "
              "mentions 8.33."),
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
           "GIVE_UP_PHRASES", "load_tasks", "select"]
