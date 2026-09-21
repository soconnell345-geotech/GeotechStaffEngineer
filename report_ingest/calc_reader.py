"""One calculation printout -> one typed :class:`~report_ingest.model.Calculation`.

THE LARGEST UNREAD CLASS OF PAGE IN THE CORPUS. A quarter of the 4,300
hand-labelled pages are calculations -- 1,009 of them, over eight of the
fourteen labelled reports -- and until this train every one was listed in a
QA entry and skipped. They are also the pages a reviewer most wants: a boring
log says what the ground is, a laboratory sheet says what a specimen did, and
a calculation says what the engineer ASSUMED, what they WORKED OUT and what
they CONCLUDED. A record with the ground in it and none of the design is half
a record.

WHAT A CALCULATION PAGE IS, AND WHY IT IS NOT A LABORATORY SHEET. A lab sheet
is a form: the same laboratory prints the same boxes every time, so a title
names the test and a detected table holds the numbers. A calculation is not a
form at all. It is one of three things and they look nothing alike:

* a PROGRAM PRINTOUT -- a banner line naming the program and its version,
  then fixed-width columns of echoed input and computed output, often for
  several load cases, often running to a dozen pages;
* a SPREADSHEET printed to PDF -- a title block, a project block, a table of
  inputs with the unit in the column HEADING and the number in the cell
  beside it, and a total at the bottom;
* a HAND CALCULATION -- a scanned sheet with a printed form around a
  handwritten working, where the only reliable text is the form.

So there is no title vocabulary to classify from and no single table to read.
What all three DO have is labelled numbers: the label printed beside the
value is what says what the value means, and that pair is the whole of what
this record holds.

THE FOUR RULES THE PROMPT IS BUILT AROUND.

**The kind is what the calculation WORKS OUT, not what printed it.** A
settlement worked on a spreadsheet and a settlement worked by a commercial
program are both ``settlement``. The vocabulary is
:data:`~report_ingest.model.CALC_KINDS` and a printout that is none of them
is ``other``, which is an answer.

**An input is what the calculation was given; a result is what it worked
out.** The line between them is the page's own layout -- a project block, a
"Design Inputs" panel and an echoed profile are inputs; a summary line, a
total, a thickness, a capacity, a factor of safety are results. Where the
page does not make the line clear the value is an input, because claiming
something was concluded when it was assumed is the worse error.

**Nothing is computed and nothing is converted.** A factor of safety the
sheet did not print stays absent. A thickness in inches stays in inches. The
label keeps the sheet's own words, parentheses and all, because a reviewer
goes looking for what the page says.

**A number that is not on the page is not a reading.** Every result the model
returns is checked back against every number these pages print, to the
printed precision. One that is nowhere on the paper keeps its place in the
record -- it may be right and the text layer wrong -- but drops to confidence
0.3 and is listed for review.

THE FLOOR. Before any call, the pages' own DETECTED TABLES and their text
lines are read for (label, value, unit) triples with nothing but a pattern:
a ``label | value`` table row, a one-row table under its own headings, a line
reading ``Pile-head deflection = 0.025 meters``, and planlens' own
``quantities`` pass over the prose. The program name comes off a running
banner where a known program-name pattern matches. That is the STARTING
RECORD the model is shown, and its answer is merged back under the same rule
as the log and the laboratory readers (:mod:`report_ingest.floor`): it may
ADD, it may CORRECT only with evidence, and it may never DROP. Every split is
a disagreement for the QA section.

THE BUDGET is :data:`MAX_CALC_CALLS` calls and most printouts cost ONE. A run
longer than :data:`MAX_CALC_PAGES` is shown a window of that many pages with
the FIRST and the LAST always in it -- the first carries the banner and the
inputs, the last carries the answer -- and the second call is spent only on
the rest of the run or on the model's own unsettled list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest.engine import (
    Engine, image_block, text_block, tool_result_block, user,
)
from report_ingest.floor import MergeLog, fold, has_evidence, settle, show
from report_ingest.model import (
    CALC_KINDS, Calculation, NamedQuantity, Provenance, Quantity,
    _canonical_unit,
)

__all__ = [
    "read_calculation", "CalcReadResult", "CalcReading", "ReadValue",
    "CALC_READER_SYSTEM", "CALC_TOOLS", "CALC_KIND_DEFINITIONS",
    "MAX_CALC_CALLS", "MAX_CALC_PAGES", "PAGE_DPI", "ZOOM_DPI",
    "PROGRAM_PATTERNS", "FloorValue", "CalcFloor", "floor_from_pages",
    "serialise_floor", "merge_calculation", "page_window", "printed_numbers",
    "NAME_RATIO", "SUMMARY_WORDS",
]

#: The ceiling in the brief: two model calls for one printout, the first of
#: them the answer. A second is spent only on a run too long to show at once
#: or on the model's own unsettled list.
MAX_CALC_CALLS = 2
#: How many of a run's pages one call may carry. A calculation appendix run
#: can be forty pages of load cases; a call that tried to carry them all
#: would spend its context on echoed columns and answer worse than one shown
#: the banner, the inputs and the summary.
MAX_CALC_PAGES = 12
#: About 110 dpi, as the laboratory reader uses: enough to read a fixed-width
#: printout and to see the shape of a plotted section.
PAGE_DPI = 110.0
#: What ``zoom_plot`` renders at. A plotted slope section or an influence
#: diagram has to be read against its own axis ticks.
ZOOM_DPI = 300.0
#: Text lines serialised per page, and how much of one line is sent. A
#: fixed-width program printout runs to a few hundred short lines.
MAX_LINES_PER_PAGE = 320
MAX_LINE_CHARS = 200
#: Table cells serialised per page.
MAX_TABLE_CHARS = 7000
#: Floor values shown to the model. A long printout yields hundreds of
#: labelled numbers and the starting record has to stay readable.
MAX_FLOOR_VALUES = 140
#: What the reader's own summary may run to.
SUMMARY_WORDS = 60
#: Two printed labels are the same label at this partial ratio. The scorer's
#: own figure (:data:`report_ingest.calc_scoring.NAME_RATIO`), so a
#: difference the scorecard would never see is not a disagreement either.
NAME_RATIO = 80.0

#: One line per kind, in wording that belongs to no firm and no program.
#: These are what the model chooses between, and they are about WHAT IS
#: WORKED OUT rather than how the printout looks -- the printouts look like
#: everything.
CALC_KIND_DEFINITIONS: Dict[str, str] = {
    "lateral_pile": "a pile, shaft or wall element loaded SIDEWAYS: "
                    "deflection, shear and bending moment against depth for "
                    "a head load, usually from a p-y analysis",
    "axial_pile": "the capacity of ONE pile or drilled shaft along its axis: "
                  "side friction and end bearing against depth, compression "
                  "and uplift, allowable or factored",
    "pile_group": "a GROUP of piles or shafts together: the load each takes, "
                  "group efficiency, group settlement, a pile cap",
    "shallow_foundation_bearing": "the bearing capacity of a footing, mat or "
                                  "raft: ultimate and allowable pressure, "
                                  "bearing-capacity factors, a footing size "
                                  "chosen to carry a load",
    "settlement": "how far something SETTLES: immediate, consolidation or "
                  "secondary, of a footing, a mat, an embankment or a fill",
    "slope_stability": "the factor of safety of a slope, an embankment or a "
                       "cut: a searched surface, a method of slices, static "
                       "or seismic, sometimes a required reinforcement",
    "retaining_wall": "earth pressure on a wall and what the wall does with "
                      "it: active, at-rest, passive or seismic pressure, "
                      "sliding, overturning, embedment, anchor or strut "
                      "loads",
    "liquefaction": "whether saturated soil will liquefy in an earthquake: "
                    "cyclic stress ratio against cyclic resistance ratio, a "
                    "factor of safety against triggering, and the settlement "
                    "or lateral spread that follows",
    "site_response": "what the ground does to the shaking: a seismic site "
                     "class, spectral accelerations, a design spectrum, a "
                     "peak ground acceleration at the surface, a "
                     "one-dimensional response analysis",
    "seepage": "water moving through the ground: a flow net, a quantity of "
               "flow, an exit gradient, a drawdown, a dewatering system",
    "pavement": "the thickness of a pavement and what it is made of: traffic "
                "in equivalent axle loads, a structural number, a slab "
                "thickness, a layer schedule",
    "ground_improvement": "making the ground better before building on it: "
                          "stone columns or aggregate piers, wick drains and "
                          "surcharge, compaction grouting, deep mixing, "
                          "dynamic compaction",
    "other": "a calculation that is none of the above -- an earthwork "
             "quantity, a structural member, a survey reduction, a page of "
             "reference tables printed as backup",
}

#: Program names a running banner may carry. GENERIC and deliberately small:
#: every one of these is a commercial or public geotechnical program named in
#: the trade press and in textbooks, and nothing here is derived from any
#: particular report. A printout whose program is not in this list is NOT
#: refused -- the model reads the banner and says what it says; this list
#: only lets the FLOOR name one before a call is spent.
PROGRAM_PATTERNS: Tuple[str, ...] = (
    r"lpile", r"\bgroup\s*\d", r"apile", r"shaft\s*\d", r"driven",
    r"\bweap\b", r"gr[lw]weap",
    r"stabl\s*\d*[a-z]*", r"\bslide\d?\b", r"slope/w", r"sigma/w", r"seep/w",
    r"quake/w", r"geostudio", r"\bplaxis\b", r"\bflac\b", r"\brs\s*[23]\b",
    r"settle\s*3", r"\bunisettle\b", r"\bembank\b", r"\bfossa\b",
    r"\bmsew\b", r"\bressa\b", r"\bsnail[sz]?\b", r"\bclara\b",
    r"\bshake\s*\d*\b", r"deepsoil", r"\bproshake\b", r"\bstrata\b",
    r"\bliquefy\b", r"\bcliq\b", r"cpet[- ]?it", r"\bnovo\b",
    r"\bwinpas\b", r"\bdarwin\b", r"\bpca\s*spreadsheet\b", r"\bpcapave\b",
    r"\bfb[- ]?pier\b", r"\bfb[- ]?multipier\b", r"\ballpile\b",
    r"\bgint\b", r"\bgeo5\b", r"\bgeostru\b", r"\bslope\s*w\b",
)
_PROGRAM_RE = re.compile("|".join(PROGRAM_PATTERNS), re.I)
#: A version printed beside the banner: ``Version 2018.11.3``, ``v6.2``,
#: ``Release 9``.
_VERSION_RE = re.compile(
    r"\b(?:version|ver\.?|release|rel\.?|v)\s*"
    r"([0-9]+(?:\.[0-9]+){0,3}[a-z]?)\b", re.I)

#: A number, however a printout writes it: ``-0.00975``, ``1,200``,
#: ``8.22E-06``, ``2.5%``. The thousands separator is kept out of the value.
_NUMBER_RE = re.compile(
    r"[-+‐‑‒–−]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+‐‑‒–−]?\d+)?")
#: A unit token as a printout spells one. Short, starts with a letter or a
#: sign, and made of the characters a unit is made of. Anything longer is
#: prose.
_UNIT_TOKEN = r"[A-Za-z%°'\"][A-Za-z0-9%°'\"./^·\-]{0,11}"
_UNIT_LIKE = re.compile(r"^" + _UNIT_TOKEN + r"$")
#: Words that trail a number and are NOT units. Short, because the loose
#: rule above only has to keep out what a printed line actually ends with.
#: ``in``, ``ft`` and ``m`` are units and are deliberately absent.
_NOT_UNITS = frozenset((
    "of", "to", "and", "or", "at", "the", "a", "an", "for", "from", "is",
    "was", "are", "were", "by", "on", "with", "no", "not", "yes", "total",
    "max", "min", "avg", "sum", "each", "case", "layer", "number", "value",
    "type", "load", "page", "sheet", "note", "notes", "see", "use", "using",
    "than", "then", "if", "as", "per",
))
#: ``label = value unit`` and ``label: value unit`` -- the shapes a program
#: printout and a spreadsheet use. The name is GREEDY, so a formula label
#: that carries its own equals sign keeps it and the LAST one separates the
#: label from the number.
_PAIR_RE = re.compile(
    r"^(?P<name>[A-Za-z].{1,70})\s*[=:]\s*"
    r"(?P<value>" + _NUMBER_RE.pattern + r")"
    r"\s*(?P<unit>" + _UNIT_TOKEN + r")?\s*$")
#: ``label ..........  value unit`` -- dot leaders, or a wide gap, which is
#: how a fixed-width printout lines a value up with its label.
_GAP_RE = re.compile(
    r"^(?P<name>[A-Za-z].{1,70}?)[\s.]{3,}"
    r"(?P<value>" + _NUMBER_RE.pattern + r")"
    r"\s*(?P<unit>" + _UNIT_TOKEN + r")?\s*$")
#: A span that is nothing but a number and, perhaps, its unit -- which is
#: what a spreadsheet's value cell is, printed as its own text span beside
#: the label span that names it.
_VALUE_SPAN_RE = re.compile(
    r"^(?P<value>" + _NUMBER_RE.pattern + r")"
    r"\s*(?P<unit>" + _UNIT_TOKEN + r")?\s*$")
#: A unit printed in the label, which is what a spreadsheet column does:
#: ``Footing Width B (ft)``, ``Layer Settlement (inches)``.
_PAREN_RE = re.compile(r"\(([^)]{1,14})\)")
#: A clock time or a date, which a printout stamps in its footer and which a
#: ``label : value`` pattern would otherwise read as a labelled value --
#: ``11:53:13 AM`` becomes "Thursday, June 29, 2006 11:53" = 13.
_STAMP_RE = re.compile(
    r"\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.I)
#: Two text spans are on the same printed LINE when their tops are within
#: this many points. A spreadsheet sets a label and its value in different
#: sizes, so their boxes rarely start at exactly the same y.
BAND_TOL = 4.0
#: How far to the right of a label its value may sit and still be its value.
#: A page set in two columns puts an unrelated table on the same printed line
#: as a label, and without this the label of one column names the value of
#: the other.
MAX_PAIR_GAP = 260.0
#: How many labelled values one printout's floor may carry. A twelve-page
#: program printout states the same twenty labels on every page for every
#: load case; past this the starting record stops being a record and becomes
#: a transcription, and the cap is recorded as a warning rather than hidden.
MAX_FLOOR_KEEP = 300

#: How sure the floor is of a value. A ruled table on a vector text layer is
#: exact; one read optically is not; a pattern over a line of prose is a
#: pattern over a line of prose.
CONF_TABLE = 0.85
CONF_TABLE_OCR = 0.6
CONF_LINE = 0.7
CONF_QUANTITY = 0.65
CONF_PROGRAM = 0.8


# ---------------------------------------------------------------------------
# what the model returns
# ---------------------------------------------------------------------------
#
# These mirror report_ingest.model but are NOT the same classes, for the same
# reason the log and lab readers keep them apart: a schema the model fills
# cannot also be the schema that has already been checked. Every field is
# flat and optional, because a strict JSON schema has to list every property.

class ReadProv(BaseModel):
    """Where the model says it read a value."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="the box of the line or table cell this came from, "
                    "copied exactly; null when you read it off the picture")
    from_image: bool = Field(
        default=False,
        description="true when the picture, not the text, is what told you")
    note: str = Field(
        default="",
        description="what was ambiguous and what settled it, in 15 words or "
                    "fewer; empty when nothing was")


class ReadValue(BaseModel):
    """One labelled value the printout carries."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="the label printed beside the value, AS PRINTED, with "
                    "its own parentheses and unit text: 'Footing Width B "
                    "(ft)', 'Maximum bending moment', 'Total ESALs'")
    value: Optional[float] = Field(
        default=None,
        description="the number as printed; null when the page prints a word")
    unit: str = Field(
        default="",
        description="the unit as printed: ft, in, m, mm, kPa, psf, ksf, tsf, "
                    "psi, MPa, pcf, kN/m3, kN, kips, deg, %. Empty string "
                    "when the value has no unit -- a count, a ratio, a "
                    "factor of safety, a structural number")
    text: str = Field(
        default="",
        description="the value as WORDS where that is what is printed: 'C', "
                    "'OK', 'Adequate', 'Site Class D'. Leave value null then")
    prov: ReadProv


class Unsettled(BaseModel):
    """Something on these pages you could not read."""

    model_config = ConfigDict(extra="forbid")

    what: str = Field(description="what it is, in 15 words or fewer")
    page: Optional[int] = Field(default=None)
    why: str = Field(description="what stopped you, in 20 words or fewer")


class CalcReading(BaseModel):
    """ONE calculation printout as the model read it."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description="one of the kind vocabulary you were given")
    program: str = Field(
        default="",
        description="the program that printed this, as its banner spells it. "
                    "Empty when no program is named -- a spreadsheet or a "
                    "hand calculation usually names none, and empty is the "
                    "right answer then")
    program_version: str = Field(
        default="",
        description="its version, when the banner prints one; empty "
                    "otherwise")
    method: str = Field(
        default="",
        description="the method or the standard the page names, in the "
                    "page's own words and a dozen words at most")
    subject: str = Field(
        default="",
        description="what this calculation is FOR, as printed: the "
                    "structure, the footing, the boring, the section, the "
                    "load case. Empty when the page names none")
    inputs: List[ReadValue] = Field(
        default_factory=list,
        description="every labelled value the calculation was GIVEN")
    results: List[ReadValue] = Field(
        default_factory=list,
        description="every labelled value it WORKED OUT")
    summary: str = Field(
        default="",
        description="what this calculation does and what it concluded, in "
                    "your own words, 60 words or fewer")
    continues: bool = Field(
        default=False,
        description="true when this printout carries values on pages you "
                    "were NOT shown and they matter")
    language: str = Field(
        default="",
        description="a two-letter code when the pages are not in English: "
                    "fr, es, pt. Empty for English")
    pages_read: List[int] = Field(default_factory=list)
    unsettled: List[Unsettled] = Field(
        default_factory=list,
        description="everything on these pages you could not read, and why")


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

@dataclass
class FloorValue:
    """One labelled value read off the pages with a pattern and no meaning."""

    name: str
    value: Optional[float] = None
    unit: str = ""
    text: str = ""
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    method: str = "tables"
    confidence: float = CONF_TABLE

    @property
    def key(self) -> str:
        return fold(self.name)

    def as_named(self) -> NamedQuantity:
        """This floor value as a record value."""
        prov = Provenance(
            page=int(self.page), bbox=self.bbox,
            method="tables" if self.method == "tables" else "text",
            confidence=float(self.confidence),
            note=f"read off the page by pattern ({self.method})")
        if self.value is None:
            return NamedQuantity(name=self.name, text=self.text, prov=prov)
        return NamedQuantity(
            name=self.name,
            value=Quantity(value=float(self.value), unit=self.unit),
            prov=prov)


@dataclass
class CalcFloor:
    """What the pages' own tables, lines and quantities say, before a call."""

    program: Optional[str] = None
    program_page: Optional[int] = None
    title: str = ""
    values: List[FloorValue] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_values(self) -> int:
        return len(self.values)

    def as_calculation(self, report_id: str = "") -> Calculation:
        """The floor alone as a record, for the scorecard's BEFORE column.

        ``kind`` is ``other``: a pattern over a page cannot say what a
        calculation works out, and crediting the floor with the kind would
        flatter it into meaninglessness. Every value goes in ``inputs``,
        because nothing here can tell an input from a result either.
        """
        return Calculation(
            kind="other", program=self.program, subject=self.title,
            inputs=[v.as_named() for v in self.values],
            pages=list(self.pages), source_report=report_id,
            prov=[Provenance(
                page=self.pages[0] if self.pages else 0, method="tables",
                confidence=CONF_TABLE,
                note=f"the floor: {len(self.values)} labelled value(s) read "
                     f"off these pages by pattern")])


def _clip(text: str, n: int = MAX_LINE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _number(token: Any) -> Optional[float]:
    """The number a token prints, or None.

    A printout writes a minus as a hyphen, a non-breaking hyphen or a Unicode
    minus, groups thousands with commas and writes exponents in either case.
    All of those are the same number; ``<10`` and ``N/A`` are not numbers.
    """
    raw = str(token or "").strip().replace(" ", " ")
    for dash in ("‐", "‑", "‒", "–", "−"):
        raw = raw.replace(dash, "-")
    raw = raw.replace(",", "").replace(" ", "").rstrip("%")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _unit_in_label(label: str) -> str:
    """The unit a label carries in parentheses, or ``""``.

    Only a unit the record can convert counts: a spreadsheet writes
    ``Footing Width B (ft)`` and also ``Modulus (tsf) or Uncorrected
    N-value``, and reading the second as a unit for every cell under it would
    put tsf on a blow count.
    """
    for token in _PAREN_RE.findall(str(label or "")):
        if _canonical_unit(token.strip()):
            return token.strip()
    return ""


def _unit_token(token: Optional[str]) -> str:
    """A trailing token that really is a unit, or ``""``.

    A unit the record can CONVERT is obviously one. A unit it cannot is
    still one -- ``kN-m``, ``psi/in``, ``kips``, ``blows/ft`` -- and dropping
    it would leave a bending moment in the floor as a bare number, which is
    worse than carrying a unit nothing converts. What is kept out is the
    handful of English words a printed line ends with.
    """
    text = str(token or "").strip().strip(".,;")
    if not text:
        return ""
    if _canonical_unit(text):
        return text
    if text.lower() in _NOT_UNITS or not _UNIT_LIKE.match(text):
        return ""
    return text


def _clean_name(name: str) -> str:
    """A label as the page prints it, trimmed of leaders and punctuation."""
    text = " ".join(str(name or "").replace(" ", " ").split())
    return text.strip(" .:=-–—_|")


def _is_label(text: str) -> bool:
    """Does this span read as the NAME of a value rather than as data?

    Two letters is enough -- a worked sheet labels things ``Fa``, ``SS`` and
    ``qu`` -- but they have to be most of what the span says. Mostly digits
    is a column of a data row, and ``19.6000 0.00206 1.41E-04`` names
    nothing.
    """
    name = _clean_name(text)
    if len(name) < 2 or len(name) > 80:
        return False
    letters = sum(1 for ch in name if ch.isalpha())
    return letters >= 2 and letters >= len(name) // 3


def _pair_in_line(text: str) -> Optional[Tuple[str, float, str]]:
    """``(label, value, unit)`` a single printed span states, or None.

    Two shapes, and both of them are how a printout puts a number beside its
    own name in ONE span: ``Pile-head deflection = 0.02500000 meters`` and
    ``Design ESALs ........ 137,774``. A span of several numbers is a row of
    a table and is left alone.
    """
    line = str(text or "").replace(" ", " ").strip()
    if not line or len(line) > 160 or _STAMP_RE.search(line):
        return None
    for pattern in (_PAIR_RE, _GAP_RE):
        match = pattern.match(line)
        if match is None:
            continue
        name = _clean_name(match.group("name"))
        value = _number(match.group("value"))
        if value is None or not _is_label(name):
            continue
        unit = _unit_token(match.group("unit")) or _unit_in_label(name)
        return name, value, unit
    return None


def _label_before(line: str, mention: str) -> str:
    """The words a line prints IN FRONT of one quantity mention, or ``""``.

    ``we considered a maximum bearing pressure of 144 kPa`` labels its 144
    with everything to the left of it; a line that IS the mention labels
    nothing. The label is trimmed to its last dozen words, because the
    sentence in front of a number in a paragraph is not a label.
    """
    text = " ".join(str(line or "").replace(" ", " ").split())
    needle = " ".join(str(mention or "").replace(" ", " ").split())
    if not text or not needle:
        return ""
    at = text.find(needle)
    head = text[:at] if at > 0 else ""
    words = head.split()[-12:]
    # A sentence often states the same value twice -- "144 kPa (3 ksf)" --
    # and the words right in front of the second one are the first one. The
    # label is what follows the last number in the head, and where that is
    # not words enough to be a label, what came BEFORE that number is.
    numbers = [i for i, word in enumerate(words)
               if re.match(r"^[-+]?\d", word)]
    if numbers:
        tail = words[numbers[-1] + 1:]
        words = tail if len(tail) >= 3 else words[:numbers[-1]]
    name = _clean_name(" ".join(words)).rstrip(" (")
    return name if _is_label(name) else ""


def _bands(lines: Sequence[Any]) -> List[List[Any]]:
    """The page's text spans grouped into the printed LINES they sit on.

    planlens returns a span per run of text, so a spreadsheet's ``Footing
    Width B (ft)`` and its ``25.8`` are two spans at the same height and a
    reader that looked at one span at a time would never see the pair. They
    are grouped by the top of their box, because a label and its value are
    often set in different sizes and their boxes rarely start level.
    """
    ordered = sorted(lines, key=lambda ln: (round(ln.bbox[1], 1), ln.bbox[0]))
    out: List[List[Any]] = []
    for line in ordered:
        if out and abs(float(line.bbox[1])
                       - float(out[-1][0].bbox[1])) <= BAND_TOL:
            out[-1].append(line)
        else:
            out.append([line])
    for band in out:
        band.sort(key=lambda ln: ln.bbox[0])
    return out


def _pairs_in_band(band: Sequence[Any]) -> List[Tuple[str, Any, Any]]:
    """``(label, value span, label span)`` for each pair on one printed line.

    A label span followed by a value span is a labelled value: that is how a
    spreadsheet, a title block and a project block are set. A line of THREE
    or more value spans is a data row -- a soil profile, a load schedule --
    and means something only as a row, so nothing is taken from it.

    A label span that ends in a COLON also pairs with the words that follow
    it, because that is how a printout states what it is for: ``Project
    Description:`` and then the name of the thing. Only with the colon: a
    rule that paired any two word spans on a line would read a two-column
    page of prose as a hundred labelled values.
    """
    texts = [" ".join(str(ln.text or "").replace(" ", " ").split())
             for ln in band]
    values = [bool(_VALUE_SPAN_RE.match(t)) for t in texts]
    if sum(values) > 2:
        return []
    out: List[Tuple[str, Any, Any]] = []
    label: Optional[int] = None
    sep = False

    def near(i: int, j: int) -> bool:
        return (float(band[i].bbox[0])
                - float(band[j].bbox[2])) <= MAX_PAIR_GAP

    for i, text in enumerate(texts):
        # A lone equals sign or colon set as its OWN span between a label
        # and its value, which is how a worked sheet lines up
        # "Seismic Site Class  =  C". It keeps the label pending.
        if text in ("=", ":", "-->", "→"):
            sep = True
            continue
        if values[i]:
            if label is not None and near(i, label):
                out.append((_clean_name(texts[label]), band[i], band[label]))
            label, sep = None, False
            continue
        wanted = label is not None and (
            sep or texts[label].rstrip().endswith((":", "=")))
        if wanted and len(text) <= 80 and not _STAMP_RE.search(text) \
                and near(i, label):
            out.append((_clean_name(texts[label]), band[i], band[label]))
            label, sep = None, False
            continue
        label = i if _is_label(text) else None
        sep = False
    return out


def _table_conf(table: Any) -> float:
    source = str(getattr(table, "source", "") or "").lower()
    return CONF_TABLE_OCR if ("ocr" in source or "di" in source
                              or "azure" in source) else CONF_TABLE


def _table_rows(table: Any) -> List[List[str]]:
    rows = [[str(c or "").strip() for c in row] for row in (table.rows or [])]
    if table.header:
        rows.insert(0, [str(c or "").strip() for c in table.header])
    return [r for r in rows if any(r)]


def _values_from_table(table: Any, page: int) -> List[FloorValue]:
    """Every ``label -> value`` pair one detected table states.

    TWO SHAPES, and nothing cleverer. A PROPERTY table puts the label in one
    cell and the value beside it, which is what a spreadsheet's project block
    and foundation block do. A ONE-ROW table puts the labels in the header
    and the values under them. A table with several body rows is a grid of
    data -- a soil profile, a load schedule -- whose rows mean something only
    together, and reading one cell of it as a labelled value would produce
    twenty values named ``Layer`` that are all different numbers.
    """
    rows = _table_rows(table)
    if not rows:
        return []
    conf = _table_conf(table)
    bbox = tuple(table.bbox) if getattr(table, "bbox", None) else None
    out: List[FloorValue] = []

    # A table whose first row names its columns. With ONE body row that row
    # holds the values; with several it is a grid of data -- a soil profile,
    # a layer-by-layer settlement, a load schedule -- whose rows mean
    # something only together. The LAST row of such a grid is still worth
    # taking, because a printed table's last row is where the total, the
    # cumulative settlement and the governing case are: it is labelled as
    # the last row so nothing downstream mistakes it for the whole column.
    if len(rows) >= 2 and len([c for c in rows[0] if c]) >= 2 \
            and len(rows[0]) <= 20:
        header = rows[0]
        # The last row with more than one cell filled. A spreadsheet
        # rules twenty rows and fills five, so "the last row" has to mean
        # the last row that carries VALUES rather than the last ruled line
        # with a row number in it.
        body = [r for r in rows[1:]
                if sum(1 for c in r if str(c).strip()) >= 2]
        if body:
            row = body[-1]
            suffix = "" if len(body) == 1 else ", last row of the table"
            for i, label in enumerate(header):
                if i >= len(row):
                    break
                name = _clean_name(label)
                value = _number(row[i])
                if not name or value is None:
                    continue
                tail = str(row[i]).split()
                out.append(FloorValue(
                    name=f"{name}{suffix}", value=value,
                    unit=(_unit_in_label(label)
                          or (_unit_token(tail[-1]) if len(tail) > 1 else "")),
                    page=page, bbox=bbox, method="tables", confidence=conf))
        if out:
            return out

    for row in rows:
        cells = [c for c in row if c]
        if len(cells) != 2:
            continue
        label, cell = cells[0], cells[1]
        name = _clean_name(label)
        if not name or not any(ch.isalpha() for ch in name):
            continue
        value = _number(cell)
        if value is None:
            words = " ".join(str(cell).split())
            if words and len(words) <= 40 and _number(words) is None:
                out.append(FloorValue(name=name, text=words, page=page,
                                      bbox=bbox, method="tables",
                                      confidence=conf))
            continue
        unit = _unit_in_label(label)
        if not unit:
            tail = str(cell).split()
            unit = _unit_token(tail[-1]) if len(tail) > 1 else ""
        out.append(FloorValue(name=name, value=value, unit=unit, page=page,
                              bbox=bbox, method="tables", confidence=conf))
    return out


def _program_in(text: str) -> Optional[Tuple[str, str]]:
    """``(program, version)`` a line's banner names, or None."""
    line = " ".join(str(text or "").replace(" ", " ").split())
    if not line or len(line) > 120:
        return None
    match = _PROGRAM_RE.search(line)
    if match is None:
        return None
    version = _VERSION_RE.search(line)
    return match.group(0).strip(), (version.group(1) if version else "")


def floor_from_pages(doc: Any, pages: Sequence[int],
                     report_id: str = "") -> CalcFloor:
    """The pages' own labelled values, before any model call is spent.

    Nothing here is inferred beyond the label printed beside the number: a
    table row that reads ``Footing Width B (ft) | 25.8``, a header table's
    one row of values, a line reading ``Design ESALs ... 137,774``, and
    planlens' own ``quantities`` pass for a value stated in prose. The
    program name comes off a running banner only where a known program-name
    pattern matches, and is ``None`` otherwise -- which, on a spreadsheet, is
    the right answer.
    """
    pages = [int(p) for p in pages]
    floor = CalcFloor(pages=pages)
    if not pages:
        return floor

    seen: set = set()
    capped = False
    #: ``(page, line ids) -> the text of those lines``, so a quantity mention
    #: can be given the label printed in front of it.
    line_text: Dict[Tuple[int, Tuple[str, ...]], str] = {}

    def put(value: FloorValue) -> None:
        nonlocal capped
        key = (value.key, None if value.value is None else round(value.value, 6),
               fold(value.text))
        if not value.key or key in seen:
            return
        if len(floor.values) >= MAX_FLOOR_KEEP:
            capped = True
            return
        seen.add(key)
        floor.values.append(value)

    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:                 # a page that will not read
            floor.warnings.append(
                f"page {page}: {type(exc).__name__}: {exc}")
            continue
        lines = sorted(content.lines,
                       key=lambda ln: (round(ln.bbox[1], 1), ln.bbox[0]))
        for line in lines:
            key = (page, (str(getattr(line, "id", "")),))
            line_text[key] = str(line.text or "")
        if page == pages[0]:
            for line in lines[:12]:
                text = " ".join(str(line.text or "").split())
                if text and not floor.title and len(text) >= 4 \
                        and any(ch.isalpha() for ch in text):
                    floor.title = _clip(text, 90)
                    break
        if floor.program is None:
            for line in lines[:8] + lines[-4:]:
                got = _program_in(line.text)
                if got is None:
                    continue
                name, version = got
                floor.program = f"{name} {version}".strip()
                floor.program_page = page
                break
        for table in content.tables:
            for value in _values_from_table(table, page):
                put(value)
        for line in lines:
            got = _pair_in_line(line.text)
            if got is None:
                continue
            name, value, unit = got
            put(FloorValue(name=name, value=value, unit=unit, page=page,
                           bbox=tuple(line.bbox), method="text",
                           confidence=CONF_LINE))
        # A label span and its value span on the same printed line, which is
        # what a spreadsheet, a title block and a project block are made of.
        for band in _bands(lines):
            for name, value_span, label_span in _pairs_in_band(band):
                text = " ".join(str(value_span.text or "")
                                .replace(" ", " ").split())
                match = _VALUE_SPAN_RE.match(text)
                if match is None:
                    put(FloorValue(name=name, text=text, page=page,
                                   bbox=tuple(label_span.bbox),
                                   method="text", confidence=CONF_LINE))
                    continue
                number = _number(match.group("value"))
                if number is None:
                    continue
                unit = _unit_token(match.group("unit")) \
                    or _unit_in_label(str(label_span.text or ""))
                put(FloorValue(name=name, value=number, unit=unit, page=page,
                               bbox=tuple(label_span.bbox), method="text",
                               confidence=CONF_LINE))

    # planlens' own quantities pass, for a value stated in PROSE rather than
    # printed in a table or lined up against its own label. The mention
    # itself is only the number and its unit, so the label is the words that
    # come before it on the same line; a mention with no words in front of it
    # -- an axis tick, a column of sieve sizes -- names nothing and is left.
    try:
        mentions = doc.quantities(pages=list(pages), include_markups=False)
    except Exception:                            # a document without it
        mentions = []
    for mention in mentions:
        unit = _unit_token(getattr(mention, "units", ""))
        if not unit:
            continue
        page = int(getattr(mention, "page", pages[0]))
        name = _label_before(line_text.get((page, tuple(
            getattr(mention, "line_ids", ()) or ())), ""),
            str(getattr(mention, "text", "") or ""))
        if not name:
            continue
        put(FloorValue(name=name, value=float(mention.value), unit=unit,
                       page=page,
                       bbox=(tuple(mention.bbox)
                             if getattr(mention, "bbox", None) else None),
                       method="quantities", confidence=CONF_QUANTITY))
    if capped:
        floor.warnings.append(
            f"more than {MAX_FLOOR_KEEP} labelled values were readable off "
            f"these pages; the floor kept the first {MAX_FLOOR_KEEP} and the "
            f"rest are the reader's to find")
    return floor


def serialise_floor(floor: CalcFloor) -> str:
    """The floor as compact lines the model builds on."""
    out: List[str] = [
        f"program read off a banner: {floor.program or 'none read'}",
        f"the first line of the first page: {floor.title or '(none)'}",
        f"{floor.n_values} labelled value(s) read by pattern"
        + (f", the first {MAX_FLOOR_VALUES} listed"
           if floor.n_values > MAX_FLOOR_VALUES else "") + ":",
    ]
    for value in floor.values[:MAX_FLOOR_VALUES]:
        shown = (value.text if value.value is None
                 else f"{value.value:g} {value.unit}".strip())
        box = (" box " + ",".join(f"{v:.0f}" for v in value.bbox)
               if value.bbox else "")
        out.append(f"  {value.name}: {shown}  [page {value.page}{box}, "
                   f"{value.method}, confidence {value.confidence:.2f}]")
    if not floor.values:
        out.append("  (no labelled value could be read by pattern; every "
                   "value on these pages is yours to find)")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _ratio(left: str, right: str) -> float:
    """rapidfuzz's partial ratio, or containment without it.

    rapidfuzz arrives with planlens, so it is normally here. The fallback is
    blunt rather than clever: a missing package must show up as a different
    NUMBER, never as a quietly different definition of "the same label".
    """
    if not left or not right:
        return 0.0
    try:
        from rapidfuzz import fuzz
    except ImportError:                      # pragma: no cover - fallback
        return 100.0 if left in right or right in left else 0.0
    return float(fuzz.partial_ratio(left, right))


def _same_name(left: str, right: str) -> bool:
    a, b = fold(left), fold(right)
    if not a or not b:
        return False
    return a == b or _ratio(a, b) >= NAME_RATIO


def _same_value(left: NamedQuantity, right: NamedQuantity) -> Optional[bool]:
    """Are two readings of one label the same reading?

    Compared in SI where both units convert, and as printed where they do
    not -- a printout in kips beside a truth in kN is one number, and a
    printout in a unit the record cannot convert is compared to itself.
    """
    if left.value is None or right.value is None:
        if left.value is None and right.value is None:
            return fold(left.text) == fold(right.text) if (
                left.text and right.text) else None
        return None
    a, b = left.value, right.value
    if fold(a.unit) != fold(b.unit):
        sa, sb = a.si_value, b.si_value
        if sa is not None and sb is not None:
            return abs(sa - sb) <= max(1e-9, abs(sb) * 0.005)
    return abs(a.value - b.value) <= max(1e-9, abs(b.value) * 0.005)


def _slot(rows: Sequence[NamedQuantity],
          name: str) -> Optional[NamedQuantity]:
    for row in rows:
        if _same_name(name, row.name):
            return row
    return None


def merge_calculation(floor: CalcFloor, model: Calculation,
                      log: Optional[MergeLog] = None
                      ) -> Tuple[Calculation, MergeLog]:
    """The floor's values folded onto the model's calculation.

    THE SAME RULE AS THE OTHER TWO READERS. A value the model added is kept.
    A value both read the same way is ``reconciled``. A value they read
    differently keeps the FLOOR'S unless the model named a box and said what
    it read, and either way both are in the record -- one in the slot, one on
    the provenance -- and the split is a disagreement for QA. A floor value
    the model did not return at all is kept, in ``inputs``, with a note: a
    pattern cannot tell an input from a result, and calling a value an input
    when it might be a result is the smaller claim.
    """
    log = log if log is not None else MergeLog()
    out = model.model_copy(deep=True)
    where = f"calculations[{out.kind}]"
    # The READER'S OWN rows, snapshotted before anything is appended. A floor
    # value kept because the reader did not return it must not then become
    # the thing the NEXT floor value is settled against: two patterns reading
    # the same label off the same page are one voter, not two, and settling
    # them against each other would invent a disagreement.
    model_inputs = list(out.inputs)
    model_results = list(out.results)

    if out.program is None and floor.program:
        out.program = floor.program
        log.keep(where, "program", floor.program_page, floor.program)
    elif floor.program and out.program and \
            not _same_name(floor.program, out.program):
        log.disagree(
            where, "program", floor.program, out.program, kept="model",
            why="the banner pattern and the reader named different "
                "programs; the reader's stands and the pattern's is beside "
                "it",
            page=floor.program_page, floor_method="text",
            floor_confidence=CONF_PROGRAM, model_confidence=0.9,
            model_method="model")

    for value in floor.values:
        named = value.as_named()
        twin = (_slot(model_results, value.name)
                or _slot(model_inputs, value.name))
        if twin is None:
            out.inputs.append(named)
            log.keep(where, value.name, value.page, named.value or named.text)
            continue
        same = _same_value(named, twin)
        prov = twin.prov or Provenance(page=value.page, method="model")
        s = settle(value.name, named.value or named.text,
                   twin.value or twin.text, same,
                   floor_method="tables" if value.method == "tables"
                   else "text",
                   floor_confidence=value.confidence,
                   model_method=prov.method if prov.method in (
                       "model", "model_from_picture") else "model",
                   model_confidence=float(prov.confidence),
                   evidence=has_evidence(prov, value.bbox),
                   floor_unit=value.unit,
                   model_unit=twin.value.unit if twin.value else "",
                   model_note=prov.note)
        if s.verdict == "reconciled":
            log.reconciled += 1
        elif s.verdict in ("floor_wins", "model_wins"):
            log.disagree(
                where, value.name, named.value or named.text,
                twin.value or twin.text,
                kept="model" if s.verdict == "model_wins" else "floor",
                why=("the reader named a box and a note, so its reading "
                     "replaces the pattern's; the pattern's is kept beside it"
                     if s.verdict == "model_wins" else
                     "the reader's value differs from the one printed beside "
                     "this label and it named no box or note for it; the "
                     "page's stands and the reader's is kept beside it"),
                page=value.page,
                floor_method="tables" if value.method == "tables" else "text",
                floor_confidence=value.confidence,
                model_confidence=float(prov.confidence),
                model_method=prov.method)
            if s.verdict == "floor_wins":
                twin.value, twin.text = named.value, named.text
            if s.alternative is not None:
                prov.alternatives.append(s.alternative)
                twin.prov = prov
        elif s.verdict == "floor_only":
            twin.value, twin.text = named.value, named.text
            log.keep(where, value.name, value.page, named.value or named.text)

    floor_keys = [v.key for v in floor.values]
    for row in model_inputs + model_results:
        if not any(_same_name(row.name, k) for k in floor_keys):
            page = row.prov.page if row.prov else None
            log.add(where, row.name, page, row.value or row.text,
                    method=row.prov.method if row.prov else "model")
    return out, log


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclass
class CalcReadResult:
    """One printout read, and everything needed to audit the reading."""

    calculation: Optional[Calculation] = None
    #: Every value the reader took from the PICTURE rather than the text.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    #: What was not settled: the model's own list, plus everything Python
    #: refused -- a result that is on no page, a kind outside the list, a
    #: unit the record cannot convert.
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    #: The two voters kept apart, so the scorecard can score each alone.
    floor_calculation: Optional[Calculation] = None
    model_calculation: Optional[Calculation] = None
    floor_values: int = 0
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    kept: List[Dict[str, Any]] = field(default_factory=list)
    added: List[Dict[str, Any]] = field(default_factory=list)
    reconciled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calculation": (self.calculation.model_dump(mode="json")
                            if self.calculation is not None else None),
            "changes": [dict(c) for c in self.changes],
            "unresolved": [dict(u) for u in self.unresolved],
            "cost": dict(self.cost),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "model": self.model,
            "warnings": list(self.warnings),
            "pages": list(self.pages),
            "floor_calculation": (
                self.floor_calculation.model_dump(mode="json")
                if self.floor_calculation is not None else None),
            "model_calculation": (
                self.model_calculation.model_dump(mode="json")
                if self.model_calculation is not None else None),
            "floor_values": self.floor_values,
            "disagreements": [dict(d) for d in self.disagreements],
            "kept": [dict(k) for k in self.kept],
            "added": [dict(a) for a in self.added],
            "reconciled": self.reconciled,
        }


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

CALC_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "zoom_plot",
        "description": (
            "A magnified crop of one region of the page, rendered far finer "
            "than the whole-page picture you were given. Use it on a "
            "PLOTTED result -- a slope section with its factor of safety "
            "written on it, a deflection or moment diagram, an influence "
            "chart -- and only when the value is not printed in text "
            "anywhere. Include the axes and their tick labels in the box: a "
            "curve read without them is a guess. Give the box in the same "
            "coordinates as the text lines you were shown."),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer",
                         "description": "0-based PDF page index"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "x0, y0, x1, y1 of the region, including "
                                   "the axes and their tick labels",
                },
                "why": {"type": "string",
                        "description": "what you are trying to read, in ten "
                                       "words or fewer"},
            },
            "required": ["page", "bbox"],
        },
    },
]


class _Tools:
    """The one tool the reader has, bound to one open document."""

    def __init__(self, doc: Any, pages: Sequence[int]) -> None:
        self.doc = doc
        self.pages = list(pages)
        self.zooms: List[Dict[str, Any]] = []

    def zoom_plot(self, arguments: Dict[str, Any]) -> Any:
        page = int(arguments.get("page", self.pages[0]))
        if page not in self.pages:
            raise ValueError(
                f"page {page} is not part of this printout ({self.pages})")
        bbox = arguments.get("bbox")
        if not bbox or len(list(bbox)) != 4:
            raise ValueError("bbox must be four numbers: x0, y0, x1, y1")
        box = [float(v) for v in bbox]
        png, info = self.doc.render(page, bbox=box, dpi=ZOOM_DPI)
        self.zooms.append({"page": page, "bbox": box,
                           "why": str(arguments.get("why") or "")})
        return [text_block(
            f"page {page}, box {[round(v, 1) for v in info['clip']]} at "
            f"{info['dpi']} dpi ({info['width_px']}x{info['height_px']} px). "
            f"Read the value against what you can see, and say in the note "
            f"that you read it off the picture."),
            image_block(png)]

    def run(self, name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool]:
        """``(content, is_error)`` -- a tool mistake is an answer, not a stop."""
        handler = getattr(self, name, None)
        if handler is None or name not in {t["name"] for t in CALC_TOOLS}:
            return (f"unknown tool {name!r}; the tools are "
                    f"{[t['name'] for t in CALC_TOOLS]}", True)
        try:
            return handler(arguments), False
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

CALC_READER_SYSTEM = """\
You are reading ONE calculation out of an engineering report and returning
what it says as data. A calculation printout is one of three things and they
look nothing alike: a PROGRAM PRINTOUT with a banner and fixed-width columns,
a SPREADSHEET printed to PDF with a title block and tables, or a HAND
CALCULATION on a printed form. Nothing about the layout can be assumed. What
CAN be assumed is that every number on the page has a label printed beside it
saying what it is, and that pair is what you are collecting.

WHAT YOU ARE GIVEN. Every line of text on the pages with the box it occupies,
any tables that were detected, the ledger line the document reader wrote for
these pages, the STARTING RECORD read off the pages by pattern before you
were called, and the first page as a picture. On a scanned printout the text
comes from an optical reader and may be imperfect; the picture is then worth
more.

THE KIND IS WHAT THE CALCULATION WORKS OUT. Not what printed it, not which
appendix it sits in. A settlement worked on a spreadsheet and a settlement
worked by a program are both settlement. Choose from the kind vocabulary you
were given, and choose 'other' when it is none of them -- a page of reference
tables printed as backup, an earthwork quantity, a structural member. 'other'
is an answer; a wrong kind is not.

THE PROGRAM IS WHAT THE BANNER PRINTS, OR NOTHING. Copy the program's name
and its version exactly as the page spells them. A spreadsheet and a hand
calculation usually name no program at all, and leaving it empty is the right
answer -- never name the spreadsheet's template, the firm, or a program you
think was probably used.

AN INPUT IS WHAT IT WAS GIVEN; A RESULT IS WHAT IT WORKED OUT. The page's own
layout tells you: a project block, a 'Design Inputs' panel, an echoed soil
profile and a load schedule are inputs; a summary line, a total, a chosen
thickness, a capacity, a settlement, a factor of safety are results. Where the
page does not make it clear, call it an INPUT -- saying something was
concluded when it was assumed is the worse mistake.

KEEP THE LABEL THE PAGE'S OWN WORDS. 'Footing Width B (ft)', not 'width'.
'Total Cumulative Settlement (inches)', not 'settlement'. A reviewer opens the
page and looks for the words that are on it. Keep the parentheses, keep the
symbol, keep the capitals.

UNITS STAY AS PRINTED AND NOTHING IS COMPUTED. A thickness in inches is
inches. A pressure in ksf is ksf. Do not convert, do not add up a column the
page did not add up, do not work out a factor of safety the page did not
print, and do not supply a unit the page does not state -- a count, a ratio,
a structural number and a factor of safety have no unit, and an empty unit is
correct for them.

A VALUE PRINTED AS A WORD IS A RESULT. A seismic site class is 'C'. A check
prints 'OK' or 'Adequate' or 'Does not govern'. Put those in text and leave
the number null: turning 'C' into a number would be a lie.

THE STARTING RECORD, AND THE THREE THINGS YOU MAY DO TO IT. The pages'
detected tables and their lines have already been read for every (label,
value, unit) a pattern can find, and you are given them as THE STARTING
RECORD with the page and box each came from. Build on it; do not start again.
- You may ADD what it lacks: a value printed where no pattern would find it,
  a value read off the picture, a label the pattern mangled.
- You may CORRECT a value it has, but only with evidence: give the box of
  the line or cell you read it from and say in the note what the page prints
  there. A correction with no box and no note is not accepted; the starting
  value stands and yours is kept beside it for review.
- You may never DROP one. A value you leave out is kept anyway, as an input,
  so leaving it out gains nothing and loses the chance to say it is a result.

WHAT TO LEAVE OUT. A printout echoes its inputs on every page and prints
thousands of intermediate rows -- the depth-by-depth table of a p-y analysis,
every trial circle of a slope search, every axle class of a traffic count.
Do not transcribe those. Report the values a reviewer would look for: what
was assumed, what governs, and what came out. Twenty to sixty values is a
full reading of a printout; six hundred is a transcription.

SEVERAL LOAD CASES ARE ONE CALCULATION. A printout that runs the same
analysis for four load cases or three footing sizes is ONE calculation; name
the case in the label ('Maximum bending moment, Load Case 3') so the results
stay apart.

THE SUBJECT IS WHAT IT IS FOR, AS PRINTED: the structure, the footing, the
boring, the section, the wall, the load case. If the page names a boring or a
test pit, put that identifier in the subject exactly as printed -- something
downstream links this calculation to the ground with it.

PAGES YOU WERE NOT SHOWN. A long printout is shown to you as a window, with
its first and last pages always in it. If values you would report are on
pages you cannot see, say so by setting continues true rather than guessing
at them.

ANYTHING YOU CANNOT SETTLE goes in unsettled with the reason, and that is a
good answer. So is an empty subject, an empty method and an empty program.

PROVENANCE. Every value carries the page it came from and, where you can, the
box of the line or cell that named it. Where the picture is what told you, set
from_image true.
"""

_FINAL_INSTRUCTION = (
    "Now give this calculation as data: what it works out, what printed it, "
    "the method it names, what it is for, the values it was given, the values "
    "it worked out, a summary in sixty words or fewer, and anything you could "
    "not settle."
)

_CONTINUE_INSTRUCTION = (
    "Here are the rest of this printout's pages. Report ONLY what they add: "
    "return the whole calculation again with the values from these pages "
    "folded into it, keeping everything you already reported."
)

_UNSETTLED_INSTRUCTION = (
    "You left {n} thing(s) unsettled. Look again at the pages and the "
    "picture, settle what you can, and return the whole calculation with "
    "those values in it. What is still unsettled stays in the list with the "
    "reason."
)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def page_window(pages: Sequence[int], start: int = 0,
                size: int = MAX_CALC_PAGES) -> List[int]:
    """The pages one call carries: the FIRST and the LAST always in it.

    A calculation's banner, its project block and its inputs are on the first
    page and its answer is on the last; a window cut off the front or the
    back of a long printout would miss one or the other. So the window is the
    first page, then ``size - 2`` pages from ``start``, then the last.
    ``start`` is an offset into the pages BETWEEN those two, so a second call
    continues where the first stopped.
    """
    pages = [int(p) for p in pages]
    if len(pages) <= size:
        return pages if start == 0 else []
    first, last, middle = pages[0], pages[-1], pages[1:-1]
    take = middle[int(start):int(start) + max(0, size - 2)]
    if not take and start:
        return []
    return [first] + take + [last]


def serialise_page(doc: Any, page: int) -> str:
    """One page's text lines and tables, with the boxes, as compact lines."""
    out: List[str] = []
    try:
        content = doc.page(page)
    except Exception as exc:                      # a page that will not read
        return (f"--- page {page}: could not be read "
                f"({type(exc).__name__}: {exc}) ---")
    out.append(f"--- page {page}: {len(content.lines)} text line(s), "
               f"{len(content.tables)} table(s) ---")
    sources = ", ".join(content.text_sources) or "no text layer"
    out.append(f"  text source: {sources}")
    for warning in content.warnings:
        out.append(f"  WARNING: {_clip(warning, 160)}")
    if not content.lines:
        out.append("  THIS PAGE CARRIES NO TEXT. Read it from the picture, "
                   "and say so on every value.")
    out.append("  x0,y0,x1,y1 | text")
    lines = sorted(content.lines, key=lambda ln: (round(ln.bbox[1], 1),
                                                  ln.bbox[0]))
    for line in lines[:MAX_LINES_PER_PAGE]:
        x0, y0, x1, y1 = line.bbox
        out.append(f"  {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f} | "
                   f"{_clip(line.text)}")
    if len(lines) > MAX_LINES_PER_PAGE:
        out.append(f"  [{len(lines) - MAX_LINES_PER_PAGE} further line(s) "
                   f"not listed]")
    budget = MAX_TABLE_CHARS
    for table in content.tables:
        if budget <= 0:
            out.append("  [further table(s) not listed]")
            break
        x0, y0, x1, y1 = table.bbox
        markdown = table.to_markdown()
        if len(markdown) > budget:
            markdown = markdown[:budget] + "\n  [table truncated]"
        budget -= len(markdown)
        out.append(f"  TABLE {table.id}, {table.n_rows}x{table.n_cols}, box "
                   f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}")
        out.extend("  " + row for row in markdown.splitlines())
    return "\n".join(out)


def _brief(doc: Any, window: Sequence[int], all_pages: Sequence[int],
           ledger: Sequence[str], item_title: str, report_id: str,
           budget: int, floor: CalcFloor) -> str:
    vocabulary = "\n".join(f"  {name}: {text}"
                           for name, text in CALC_KIND_DEFINITIONS.items())
    parts: List[str] = [
        f"ONE CALCULATION, printed on page(s) "
        f"{', '.join(str(p) for p in all_pages)}"
        + (f" of report {report_id}" if report_id else "") + ".",
        f"You may make {budget} model call(s) in all, the last of them your "
        f"answer.",
    ]
    if len(window) < len(all_pages):
        parts.append(
            f"You are shown {len(window)} of its {len(all_pages)} pages -- "
            f"{', '.join(str(p) for p in window)} -- its first and its last "
            f"among them. Set continues true if values you would report are "
            f"on the pages you cannot see.")
    if item_title:
        parts.append(f"The document titles it: {item_title}")
    parts += [
        "",
        "THE KIND VOCABULARY",
        vocabulary,
        "",
        "WHAT THE PAGE LEDGER SAYS ABOUT THESE PAGES",
        "\n".join(ledger) if ledger else "(no ledger line)",
        "",
        "THE STARTING RECORD (read off these pages by pattern before you "
        "were called, with the page and box each came from and the reader's "
        "confidence -- build on it; add, correct with the box and a note, "
        "never drop. Nothing here knows whether a value is an input or a "
        "result, and that is yours to say)",
        serialise_floor(floor),
        "",
        "THE PAGES",
    ]
    parts.extend(serialise_page(doc, page) for page in window)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# turning the reading into the record
# ---------------------------------------------------------------------------

def printed_numbers(doc: Any, pages: Sequence[int]) -> List[float]:
    """Every number these pages print, in text or in a table.

    What the result gate checks against. No meaning is attached to any of
    them: the only question is whether a number the reader reports is on the
    paper at all.
    """
    out: List[float] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception:                        # a page that will not read
            continue
        texts: List[str] = [str(ln.text or "") for ln in content.lines]
        for table in content.tables:
            grid = ([table.header] if table.header else []) + (table.rows
                                                               or [])
            for row in grid:
                texts.extend(str(c or "") for c in (row or ()))
        for text in texts:
            cleaned = str(text).replace(" ", " ")
            for token in _NUMBER_RE.findall(cleaned):
                value = _number(token)
                if value is not None:
                    out.append(value)
    return out


def _precision_tol(value: float) -> float:
    """Half of the last printed digit's place, from the value as printed.

    A page printing 0.74 has said "between 0.735 and 0.745"; a page printing
    137,774 has said that number exactly. Matching a reported value to the
    page at the PRINTED precision is the honest test, and a bare relative
    tolerance would let 0.7 pass for 0.74 on a page that printed two
    decimals.
    """
    text = f"{value!r}"
    if "e" in text or "E" in text:
        return max(abs(value) * 1e-6, 1e-12)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * (10.0 ** -decimals)


def _on_the_page(value: float, pool: Sequence[float]) -> bool:
    tol = _precision_tol(value)
    return any(abs(printed - value) <= tol for printed in pool)


class _Builder:
    """Turns one :class:`CalcReading` into a :class:`Calculation`.

    Three gates, and each refusal costs one value and nothing else: a
    printout with one hallucinated total still has thirty good numbers on it
    and the record should carry them.
    """

    def __init__(self, reading: CalcReading, pages: Sequence[int],
                 report_id: str, zooms: Sequence[Dict[str, Any]],
                 pool: Sequence[float]) -> None:
        self.reading = reading
        self.pages = list(pages)
        self.report_id = report_id
        self.zooms = list(zooms)
        self.pool = list(pool)
        self.unresolved: List[Dict[str, Any]] = []
        self.changes: List[Dict[str, Any]] = []

    def _refuse(self, what: str, why: str, value: Any = None,
                page: Optional[int] = None) -> None:
        row: Dict[str, Any] = {"what": what, "why": why,
                               "refused_by": "python"}
        if value is not None:
            row["value"] = show(value)
        if page is not None:
            row["page"] = int(page)
        self.unresolved.append(row)

    # -- the gates ---------------------------------------------------------
    def kind_of(self) -> str:
        """The kind, or ``other`` with a note saying what was asked for."""
        kind = (self.reading.kind or "").strip().lower().replace(" ", "_")
        if kind in CALC_KINDS:
            return kind
        alias = {
            "bearing_capacity": "shallow_foundation_bearing",
            "shallow_foundation": "shallow_foundation_bearing",
            "footing": "shallow_foundation_bearing",
            "deep_foundation": "axial_pile", "pile": "axial_pile",
            "drilled_shaft": "axial_pile", "micropile": "axial_pile",
            "p_y": "lateral_pile", "lateral_load": "lateral_pile",
            "consolidation": "settlement",
            "slope": "slope_stability", "stability": "slope_stability",
            "earth_pressure": "retaining_wall", "wall": "retaining_wall",
            "shoring": "retaining_wall", "excavation_support":
                "retaining_wall",
            "seismic": "site_response", "site_class": "site_response",
            "response_spectrum": "site_response",
            "dewatering": "seepage", "groundwater": "seepage",
            "pavement_design": "pavement",
            "stone_columns": "ground_improvement",
        }.get(kind)
        if alias:
            return alias
        if kind:
            self._refuse(
                "calculation kind",
                f"{kind!r} is not in the kind vocabulary; recorded as "
                f"'other' rather than invented", kind)
        return "other"

    def provenance(self, prov: ReadProv, what: str) -> Provenance:
        page = int(prov.page)
        if page not in self.pages:
            self._refuse(
                what, f"the reader put this on page {page}, which is not "
                      f"part of this printout ({self.pages}); the box was "
                      f"dropped", page=page)
            return Provenance(page=self.pages[0] if self.pages else page,
                              method=("model_from_picture" if prov.from_image
                                      else "model"),
                              confidence=0.4, note=prov.note)
        method = "model_from_picture" if prov.from_image else "model"
        if prov.from_image:
            self.changes.append({
                "what": what, "page": page,
                "why": prov.note or "read off the picture, not the text"})
        return Provenance(page=page,
                          bbox=tuple(prov.bbox) if prov.bbox else None,
                          method=method,
                          confidence=0.8 if prov.from_image else 0.9,
                          note=prov.note)

    def value(self, read: ReadValue, role: str) -> Optional[NamedQuantity]:
        """One read value as a record value, or None when it is not one."""
        name = _clean_name(read.name)
        if not name:
            self._refuse(f"an unnamed {role}",
                         "the reader returned a value with no printed label; "
                         "a value with no label cannot be checked and was "
                         "not recorded",
                         read.value, page=read.prov.page)
            return None
        prov = self.provenance(read.prov, f"{role} {name}")
        text = " ".join(str(read.text or "").split())
        if read.value is None:
            if not text:
                self._refuse(f"{role} {name}",
                             "the reader returned neither a number nor any "
                             "words for this label", page=prov.page)
                return None
            return NamedQuantity(name=name, text=text, prov=prov)

        unit = str(read.unit or "").strip()
        note = ""
        if unit and _canonical_unit(unit) is None:
            # NEVER DROPPED, and never silently kept either: the value stays
            # in the record with the unit the page printed -- that is the
            # record's own rule -- and the fact that nothing downstream can
            # convert it is on the unsettled list where a reviewer sees it.
            note = (f"the unit {unit!r} is not one the record can convert; "
                    f"the value is kept as printed and nothing converts it")
            self._refuse(f"{role} {name}", note,
                         f"{read.value:g} {unit}", page=prov.page)
        quantity = Quantity(value=float(read.value), unit=unit)

        if role == "result" and not _on_the_page(float(read.value),
                                                 self.pool):
            prov.confidence = 0.3
            flag = ("this number is on none of these pages to the precision "
                    "it was reported at; it is kept at confidence 0.3 for "
                    "review rather than dropped")
            note = f"{note}; {flag}".strip("; ") if note else flag
            self._refuse(f"result {name}", flag, f"{read.value:g} {unit}",
                         page=prov.page)
        return NamedQuantity(name=name, value=quantity, note=note, prov=prov)

    def _rows(self, reads: Sequence[ReadValue],
              role: str) -> List[NamedQuantity]:
        out: List[NamedQuantity] = []
        for read in reads:
            got = self.value(read, role)
            if got is not None:
                out.append(got)
        return out

    def summary(self) -> str:
        words = str(self.reading.summary or "").split()
        if len(words) <= SUMMARY_WORDS:
            return " ".join(words)
        self._refuse("summary",
                     f"the reader's summary ran to {len(words)} words; it "
                     f"was cut to {SUMMARY_WORDS}")
        return " ".join(words[:SUMMARY_WORDS]) + " …"

    def build(self) -> Calculation:
        kind = self.kind_of()
        program = " ".join(
            x for x in (str(self.reading.program or "").strip(),
                        str(self.reading.program_version or "").strip()) if x)
        prov = Provenance(
            page=self.pages[0] if self.pages else 0, method="model",
            confidence=0.9,
            note=f"read from page(s) "
                 f"{', '.join(str(p) for p in self.pages)}")
        for item in self.reading.unsettled:
            self.unresolved.append({
                "what": item.what, "page": item.page, "why": item.why,
                "refused_by": "reader"})
        for zoom in self.zooms:
            self.changes.append({
                "what": "zoom_plot", "page": zoom["page"],
                "why": zoom["why"] or "looked closely at a plotted result"})
        return Calculation(
            kind=kind,
            program=program or None,
            method=" ".join(str(self.reading.method or "").split()),
            subject=" ".join(str(self.reading.subject or "").split()),
            inputs=self._rows(self.reading.inputs, "input"),
            results=self._rows(self.reading.results, "result"),
            summary=self.summary(),
            pages=list(self.pages),
            source_report=self.report_id,
            prov=[prov],
            unsettled=[dict(u) for u in self.unresolved])


def _fold_readings(first: CalcReading, second: CalcReading) -> CalcReading:
    """Two windows of one printout as one reading.

    The later window wins on the scalar fields only where the first left
    them empty -- the banner and the project block are on the first page --
    and the value lists are the union, keyed by the printed label, with the
    later reading's value kept where both carry a label.
    """
    out = first.model_copy(deep=True)
    out.kind = second.kind or first.kind
    out.program = first.program or second.program
    out.program_version = first.program_version or second.program_version
    out.method = first.method or second.method
    out.subject = first.subject or second.subject
    out.summary = second.summary or first.summary
    out.continues = bool(second.continues)
    for name in ("inputs", "results"):
        merged: List[ReadValue] = list(getattr(out, name))
        for row in getattr(second, name):
            twin = next((r for r in merged if _same_name(r.name, row.name)),
                        None)
            if twin is None:
                merged.append(row)
            else:
                merged[merged.index(twin)] = row
        setattr(out, name, merged)
    out.pages_read = sorted(set(list(first.pages_read)
                                + list(second.pages_read)))
    out.unsettled = list(second.unsettled)
    return out


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

def read_calculation(doc, item_pages: Sequence[int], engine: Engine, *,
                     budget: int = MAX_CALC_CALLS,
                     ledger: Optional[Sequence[str]] = None,
                     item_title: str = "", report_id: str = "",
                     dpi: float = PAGE_DPI) -> CalcReadResult:
    """Read ONE calculation printout into a :class:`Calculation`.

    ``item_pages`` are the pages of one printout --
    ``planlens.document.roles.build_items`` groups them, splitting a
    calculation appendix on program banners and on printed titles, so a run
    of pages IS one printout. ``budget`` is the model-call ceiling, the
    answer included; a second call is spent only on pages the first was not
    shown or on the reader's own unsettled list.
    """
    pages = [int(p) for p in item_pages]
    if not pages:
        raise ValueError("read_calculation needs at least one page")
    budget = max(1, min(int(budget), MAX_CALC_CALLS))

    if ledger is None:
        try:
            from planlens.document.roles import page_ledger
            lines = page_ledger(doc)
            ledger = [ln for ln in lines
                      if any(ln.startswith(f"p{p:03d} ") for p in pages)]
        except Exception:                        # a ledger is a nicety
            ledger = []

    spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0}

    def charge(reply) -> None:
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

    # THE FLOOR, over EVERY page of the run -- the pattern is free, so a
    # window narrower than the printout costs the model's eyes and not the
    # starting record.
    floor = floor_from_pages(doc, pages, report_id)
    warnings.extend(floor.warnings)
    pool = printed_numbers(doc, pages)

    window = page_window(pages)
    images: List[bytes] = []
    try:
        png, _info = doc.render(window[0], dpi=dpi)
    except Exception:                            # a page that will not draw
        pass                                     # is read from its text alone
    else:
        images.append(png)

    brief = _brief(doc, window, pages, ledger, item_title, report_id, budget,
                   floor)
    tools = _Tools(doc, pages)
    messages: List[Dict[str, Any]] = [
        user(text_block(brief), *[image_block(png) for png in images])]

    model_calls = 0
    tool_calls = 0
    reading: Optional[CalcReading] = None
    final: Any = None
    next_start = max(0, MAX_CALC_PAGES - 2)
    followed_up = False

    while model_calls < budget:
        last = model_calls == budget - 1
        reply = engine.complete(messages, system=CALC_READER_SYSTEM,
                                tools=None if last else CALC_TOOLS,
                                output_format=CalcReading)
        model_calls += 1
        charge(reply)
        final = reply
        if reply.tool_calls and not last:
            messages.append({"role": "assistant", "content": reply.content})
            results: List[Dict[str, Any]] = []
            for call in reply.tool_calls:
                tool_calls += 1
                content, is_error = tools.run(call.name, call.arguments)
                results.append(tool_result_block(call.id, content,
                                                 is_error=is_error))
            messages.append({"role": "user", "content": results})
            continue
        if reply.parsed is not None:
            reading = (reply.parsed if reading is None
                       else _fold_readings(reading, reply.parsed))
            if last or followed_up:
                break
            # The one follow-up this reader spends: the rest of a printout it
            # could not be shown at once, or its own unsettled list. Anything
            # else and the budget goes on re-reading pages already read.
            rest = page_window(pages, next_start)
            if reading.continues and rest:
                followed_up = True
                messages = [user(
                    text_block(_CONTINUE_INSTRUCTION), text_block(
                        "\n".join(serialise_page(doc, p) for p in rest)))]
                continue
            if reading.unsettled:
                followed_up = True
                messages.append({"role": "assistant",
                                 "content": reply.content or [text_block("")]})
                messages.append(user(text_block(_UNSETTLED_INSTRUCTION.format(
                    n=len(reading.unsettled)))))
                continue
            break
        # Neither a tool call nor an answer: say what is wanted, once, and
        # spend another call on it rather than returning nothing.
        messages.append({"role": "assistant",
                         "content": reply.content or [text_block("")]})
        messages.append(user(text_block(_FINAL_INSTRUCTION)))

    if reading is None:
        raise RuntimeError(
            f"the calculation reader returned no structured reading for "
            f"pages {pages} in {model_calls} call(s) (stop_reason "
            f"{getattr(final, 'stop_reason', None)!r})")

    builder = _Builder(reading, pages, report_id, tools.zooms, pool)
    model_calc = builder.build()
    merged, merge_log = merge_calculation(floor, model_calc, MergeLog())
    merged.unsettled = [dict(u) for u in builder.unresolved]
    spent["seconds"] = round(spent["seconds"], 2)
    spent["dollars"] = round(spent["dollars"], 5)
    return CalcReadResult(
        calculation=merged,
        changes=builder.changes,
        unresolved=builder.unresolved,
        cost=spent,
        model_calls=model_calls,
        tool_calls=tool_calls,
        model=getattr(final, "model", "") or getattr(engine, "name", ""),
        warnings=warnings,
        pages=pages,
        floor_calculation=floor.as_calculation(report_id),
        model_calculation=model_calc,
        floor_values=floor.n_values,
        disagreements=merge_log.disagreements,
        kept=merge_log.kept,
        added=merge_log.added,
        reconciled=merge_log.reconciled)
