"""One laboratory sheet -> typed :class:`~report_ingest.model.LabTest` records.

A boring log is a form with a depth ruler, and ``log_grid`` can say WHERE
every number on it sits before any model sees the page. A lab sheet is not
that. Every laboratory prints its own form, half of them are a plot with a
results box beside it, and the thing that says what a number means is the
word printed next to it. So the geometry here is weaker and the reading is
more of the work -- which is exactly why the rules below are strict about
what may be believed.

THE FOUR RULES THE PROMPT IS BUILT AROUND.

**The sheet's own title says what the test is.** Not the appendix tab, not
the file name, not what the numbers look like. A page headed ATTERBERG LIMITS
is an Atterberg test even when it also prints a sieve column, and a page
headed CERTIFICATE OF ANALYSIS that lists samples and reports nothing is not
a test at all -- it is recorded as such, with ``no_results``, because a page
that was read and found empty and a page that was skipped are different
things.

**A tabulated value beats the plot every time.** Half these sheets print the
grading curve AND the percent-passing table; reading the table is exact and
reading the plot is not. A curve is digitised only where the values are not
tabulated anywhere on the sheet, and then the whole test is flagged
``curves_digitised`` -- flagged as a whole, because a reviewer checks the
sheet, not one number.

**The link to the ground is what the sheet PRINTS.** The boring identifier
and the depth, copied. This reader never matches a sample to a log; the
reconciler does that later and records a conflict when it cannot. A sheet
that names no boring gets an empty identifier, which is an answer.

**Units stay as printed.** A confining pressure in psf stays in psf and a
sieve opening in inches stays in inches. The DIGGS writer converts, once.

WHAT PYTHON CHECKS AFTERWARDS. Four things that cannot be true of a real
sheet, each of which a model does produce now and then: a depth outside
0-300 m, a percentage outside 0-100, a liquid limit below the plastic limit,
and a grading series in which MORE passes a smaller sieve. Each is refused
into ``unresolved`` rather than accepted, and the rest of the sheet is kept:
one bad series is not a reason to lose the eleven good values beside it.

THE BUDGET is four model calls per sheet, and most sheets cost ONE. Every
call asks for the answer and offers :func:`zoom_plot` at the same time, so a
sheet whose values are tabulated is read and answered in a single call; a
further call is spent only when the model asks to magnify a plot. On its last
allowed call the tool is withdrawn, so a reader that keeps zooming runs out
of looking rather than out of answering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest.engine import (
    Engine, image_block, text_block, tool_result_block, user,
)
from report_ingest.model import (
    AtterbergResult, CBRResult, ChemicalResult, CompactionPoint,
    CompactionResult, ConsolidationPoint, ConsolidationResult,
    GradationResult, LabTest, MoistureDensityResult, OtherResult, Provenance,
    Quantity, RESULT_CLASS, ShearPoint, SievePoint, StrengthResult,
    StrengthSpecimen, SummaryRow, SummaryTableResult,
)

__all__ = [
    "read_lab_sheet", "LabReadResult", "LabSheetReading", "ReadTest",
    "LAB_READER_SYSTEM", "LAB_TOOLS", "KIND_DEFINITIONS", "MAX_MODEL_CALLS",
    "PAGE_DPI", "ZOOM_DPI", "MAX_DEPTH_M", "SERIES_NAMES",
]

#: The ceiling in the brief: four model calls for one sheet, the last of them
#: the structured answer. A sheet whose values are all tabulated needs one.
MAX_MODEL_CALLS = 4
#: About 110 dpi: a letter page at roughly 1200 px on its long side. Enough
#: to read a results box and see the shape of a curve.
PAGE_DPI = 110.0
#: What ``zoom_plot`` renders at. A plotted curve has to be read against its
#: own axis ticks, so the crop is rendered far finer than the page.
ZOOM_DPI = 300.0
#: No borehole in this corpus is deeper than a few hundred metres, and a
#: depth beyond this is a misread column, not a reading.
MAX_DEPTH_M = 300.0
#: Text lines serialised per page, and how much of one line is sent. A dense
#: certificate page runs to a few hundred lines; a summary table to more.
MAX_LINES_PER_PAGE = 400
MAX_LINE_CHARS = 200
#: Table cells serialised per page. A wide summary table is the reason this
#: is generous.
MAX_TABLE_CHARS = 9000

#: The series a lab sheet plots or tabulates, and what the pair means. The
#: model picks a name from this list; Python knows what to do with each.
SERIES_NAMES: Dict[str, str] = {
    "percent_passing": "particle size against percent finer: the grading "
                       "curve. x = the sieve opening, y = percent passing",
    "consolidation": "pressure against strain or void ratio: the e-log p or "
                     "strain-log p curve. x = pressure, y = strain % "
                     "(or void ratio)",
    "shear_envelope": "normal stress against shear stress: the failure "
                      "envelope of a direct shear test",
    "stress_strain": "axial strain against deviator stress, one curve per "
                     "specimen. x = strain %, y = deviator stress",
    "compaction": "water content against dry density: the Proctor curve. "
                  "x = water content %, y = dry density",
    "flow_curve": "blow count against water content: the Casagrande trials "
                  "of a liquid limit",
    "cbr": "penetration against load or stress",
}

#: One line per kind, in wording that belongs to no single laboratory. These
#: are what the model chooses between, and they are deliberately about WHAT
#: WAS MEASURED rather than how the sheet looks -- the sheets look like
#: everything.
KIND_DEFINITIONS: Dict[str, str] = {
    "atterberg": "the water contents at which a fine soil changes state: "
                 "liquid limit, plastic limit, plasticity index, sometimes a "
                 "shrinkage limit",
    "gradation": "how much of a sample is each size: a sieve and/or "
                 "hydrometer analysis, reported as percent passing each size "
                 "and usually as gravel, sand, silt and clay fractions",
    "swell_consolidation": "how much a confined specimen moves under "
                           "pressure: a swell, collapse, oedometer or "
                           "consolidation test, reported as strain or void "
                           "ratio against applied pressure",
    "triaxial": "a cylindrical specimen sheared inside a cell under a "
                "confining pressure: UU, CU or CD",
    "direct_shear": "a specimen sheared along a forced plane under a normal "
                    "stress, usually three specimens giving an envelope",
    "unconfined": "a SOIL specimen compressed with no confinement, giving an "
                  "unconfined compressive strength",
    "unconfined_rock": "a ROCK core compressed with no confinement, giving a "
                       "uniaxial compressive strength, usually in MPa",
    "compaction": "how dense a soil packs at different water contents: a "
                  "Proctor, reported as a curve with a maximum dry density "
                  "and an optimum water content",
    "cbr": "the resistance of a compacted specimen to a penetrating piston, "
           "reported as a percentage",
    "moisture_content": "how much water a specimen holds, and nothing else",
    "density": "the bulk and/or dry density of a specimen, and nothing else",
    "organic_content": "how much of a specimen burns away: loss on ignition "
                       "or ash content",
    "chemical": "what a soil would do to buried metal and concrete: pH, "
                "resistivity, sulfate, chloride, sulfide, redox potential",
    "specific_gravity": "the specific gravity of the solid particles",
    "permeability": "how fast water flows through a specimen",
    "summary_table": "a TABLE of many specimens' results together, usually "
                     "headed a summary of laboratory testing: one row per "
                     "specimen, one column per kind of result",
    "other": "a laboratory page that is none of the above, including one "
             "that reports no result at all -- a list of samples received, a "
             "chain of custody, a cover sheet",
}


# ---------------------------------------------------------------------------
# what the model returns
# ---------------------------------------------------------------------------
#
# These mirror report_ingest.model but are NOT the same classes, for the same
# reason the log reader keeps them apart: a schema the model fills cannot also
# be the schema that has already been checked. Every field is flat and
# optional, because a strict JSON schema has to list every property and a
# union of ten result shapes is where structured output goes wrong.

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


class ReadQuantity(BaseModel):
    """A number that has a unit, and the unit exactly as the sheet prints it."""

    model_config = ConfigDict(extra="forbid")

    value: float
    unit: str = Field(
        description="as printed: psf, kPa, tsf, pcf, kN/m3, g/cm3, Mg/m3, "
                    "MPa, mm, in, ohm-cm, mg/kg, mV, deg. Empty string only "
                    "when the sheet prints no unit at all")


class ReadReported(BaseModel):
    """A value a sheet may print as a number OR as words.

    ``<10``, ``Nil``, ``trace``, ``N.P.``: set ``text`` and leave ``value``
    null. Never turn ``<10`` into 10 -- below the reporting limit is not the
    number ten.
    """

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = Field(
        default=None, description="the number, when the sheet prints one")
    text: str = Field(
        default="",
        description="the words, when the sheet prints words instead: '<10', "
                    "'Nil', 'trace to positive', 'N.P.'")
    unit: str = Field(default="", description="as printed; empty if none")


class ReadPoint(BaseModel):
    """One point of a series."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    label: str = Field(
        default="",
        description="which specimen, which stage (load, unload, rebound), or "
                    "the sieve's printed designation")


class ReadSeries(BaseModel):
    """One curve or one table of pairs the sheet carries."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="percent_passing, consolidation, shear_envelope, "
                    "stress_strain, compaction, flow_curve or cbr")
    x_unit: str = Field(
        default="",
        description="the unit of x as printed: mm, in, psf, kPa, '%'")
    y_unit: str = Field(
        default="", description="the unit of y as printed; '%' for a "
                                "percentage, empty for a void ratio")
    digitised: bool = Field(
        default=False,
        description="true when you read these points off a PLOT because the "
                    "sheet tabulates them nowhere")
    points: List[ReadPoint] = Field(default_factory=list)


class ReadSpecimen(BaseModel):
    """One specimen of a strength test."""

    model_config = ConfigDict(extra="forbid")

    specimen_id: str = Field(default="")
    confining: Optional[ReadQuantity] = Field(
        default=None, description="cell pressure, or normal stress")
    peak_deviator: Optional[ReadQuantity] = Field(
        default=None,
        description="peak deviator stress, peak shear stress, or the failure "
                    "stress")
    strain_at_peak_percent: Optional[float] = None
    pore_pressure: Optional[ReadQuantity] = Field(
        default=None, description="pore pressure at failure, or its change")
    stress_ratio: Optional[float] = None
    c: Optional[ReadQuantity] = None
    phi_deg: Optional[float] = None
    wc: Optional[float] = Field(default=None, description="water content, %")
    dry_density: Optional[ReadQuantity] = None
    wet_density: Optional[ReadQuantity] = None
    height: Optional[ReadQuantity] = None
    diameter: Optional[ReadQuantity] = None
    note: str = Field(default="")


class ReadPassing(BaseModel):
    """One sieve of a grading, as a summary table prints it."""

    model_config = ConfigDict(extra="forbid")

    sieve: str = Field(
        description="as printed: 'No. 200', '3/4 in', '0.075 mm'")
    percent_passing: float
    size_mm: Optional[float] = Field(
        default=None,
        description="the opening in millimetres, ONLY when the sheet prints "
                    "it; never worked out from the sieve number")


class ReadField(BaseModel):
    """Something printed that has no field of its own here."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="the sheet's own heading for it")
    value: str = Field(description="the cell's text, verbatim")


class ReadRow(BaseModel):
    """One row of a summary-of-laboratory-tests table."""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(default="")
    sample_id: str = Field(default="")
    depth_top: Optional[float] = None
    depth_bottom: Optional[float] = None
    elevation_top: Optional[ReadQuantity] = None
    sample_type: str = Field(default="")
    description: str = Field(default="")
    uscs: str = Field(default="", description="as printed; never inferred")
    stratum: str = Field(default="")
    lab: str = Field(default="")
    wc: Optional[float] = None
    ll: Optional[float] = None
    pl: Optional[ReadReported] = None
    pi: Optional[float] = None
    passing: List[ReadPassing] = Field(default_factory=list)
    fines_percent: Optional[float] = None
    sand_percent: Optional[float] = None
    gravel_percent: Optional[float] = None
    silt_clay_percent: Optional[float] = None
    wet_density: Optional[ReadQuantity] = None
    dry_density: Optional[ReadQuantity] = None
    max_dry_density: Optional[ReadQuantity] = None
    optimum_wc: Optional[float] = None
    qu: Optional[ReadQuantity] = None
    su: Optional[ReadQuantity] = None
    c: Optional[ReadQuantity] = None
    phi_deg: Optional[float] = None
    swell_percent: Optional[float] = None
    organic_percent: Optional[float] = None
    pH: Optional[ReadReported] = None
    resistivity: Optional[ReadReported] = None
    sulfate: Optional[ReadReported] = None
    chloride: Optional[ReadReported] = None
    sulfides: Optional[ReadReported] = None
    redox: Optional[ReadReported] = None
    other: List[ReadField] = Field(default_factory=list)


class ReadTest(BaseModel):
    """One test on one specimen, as the model read it.

    Flat on purpose: every kind of test fills the fields that belong to it
    and leaves the rest null. Python sorts them into the typed result.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description="one of the kind vocabulary you were given")
    test_type: str = Field(
        default="",
        description="the sub-kind the sheet names: UU, CU, CD, CDS, swell, "
                    "oedometer, standard, modified")
    investigation_id: str = Field(
        default="",
        description="the boring, pit or sounding PRINTED on the sheet, "
                    "exactly as printed. Empty when the sheet names none")
    sample_id: str = Field(default="", description="as printed")
    depth_top: Optional[float] = Field(
        default=None, description="in the depth unit of this sheet")
    depth_bottom: Optional[float] = None
    elevation: Optional[ReadQuantity] = None
    standard: str = Field(
        default="", description="ASTM D4318, BS 1377 Part 2 ... as printed")
    lab: str = Field(default="", description="who ran it, when printed")
    date: str = Field(default="", description="as printed; do not reformat")
    description: str = Field(
        default="", description="the sheet's description of the soil or rock")
    uscs: str = Field(
        default="", description="the group symbol PRINTED; never inferred")

    # -- the index values a sheet prints ----------------------------------
    ll: Optional[float] = Field(default=None, description="liquid limit, %")
    pl: Optional[float] = Field(default=None, description="plastic limit, %")
    pi: Optional[float] = Field(default=None, description="plasticity index, %")
    non_plastic: bool = Field(
        default=False, description="the sheet printed NP instead of limits")
    shrinkage_limit: Optional[float] = None
    pl_trials: List[float] = Field(
        default_factory=list, description="each plastic-limit trial, %")
    wc: Optional[float] = Field(default=None, description="water content, %")
    water_contents: List[float] = Field(
        default_factory=list,
        description="each determination, where the sheet prints several and "
                    "an average")
    wet_density: Optional[ReadQuantity] = None
    dry_density: Optional[ReadQuantity] = None
    specific_gravity: Optional[float] = None
    void_ratio: Optional[float] = None
    saturation_percent: Optional[float] = None
    ash_percent: Optional[float] = None
    organic_percent: Optional[float] = None

    # -- gradation ---------------------------------------------------------
    cobbles_percent: Optional[float] = None
    gravel_percent: Optional[float] = None
    sand_percent: Optional[float] = None
    silt_percent: Optional[float] = None
    clay_percent: Optional[float] = None
    fines_percent: Optional[float] = Field(
        default=None,
        description="silt and clay together, where the sheet prints one "
                    "number for them")
    d10: Optional[ReadQuantity] = None
    d30: Optional[ReadQuantity] = None
    d50: Optional[ReadQuantity] = None
    d60: Optional[ReadQuantity] = None
    d85: Optional[ReadQuantity] = None
    d90: Optional[ReadQuantity] = None
    d100: Optional[ReadQuantity] = None
    cu: Optional[float] = Field(
        default=None, description="coefficient of uniformity")
    cc: Optional[float] = Field(
        default=None, description="coefficient of curvature")
    hydrometer: bool = Field(default=False)

    # -- consolidation -----------------------------------------------------
    swell_percent: Optional[float] = None
    swell_at: Optional[ReadQuantity] = Field(
        default=None, description="the pressure the swell was measured at")
    swell_pressure: Optional[ReadQuantity] = None
    pc: Optional[ReadQuantity] = Field(
        default=None, description="preconsolidation pressure")
    compression_index: Optional[float] = Field(
        default=None, description="Cc of a consolidation test")
    recompression_index: Optional[float] = Field(
        default=None, description="Cr of a consolidation test")
    cv: Optional[ReadQuantity] = None
    e0: Optional[float] = None

    # -- strength ----------------------------------------------------------
    c: Optional[ReadQuantity] = Field(
        default=None, description="cohesion intercept of the envelope")
    phi_deg: Optional[float] = None
    c_residual: Optional[ReadQuantity] = None
    phi_residual_deg: Optional[float] = None
    qu: Optional[ReadQuantity] = Field(
        default=None, description="unconfined or uniaxial strength")
    su: Optional[ReadQuantity] = None
    strain_at_failure_percent: Optional[float] = None
    rock_type: str = Field(default="")
    weathering: str = Field(default="")
    specimens: List[ReadSpecimen] = Field(default_factory=list)

    # -- compaction and CBR ------------------------------------------------
    max_dry_density: Optional[ReadQuantity] = None
    optimum_wc: Optional[float] = None
    method: str = Field(
        default="", description="standard, modified, or as the sheet says")
    mould_volume: Optional[ReadQuantity] = None
    blows_per_layer: Optional[int] = None
    layers: Optional[int] = None
    rammer_mass: Optional[ReadQuantity] = None
    cbr_percent: Optional[float] = None
    cbr_at_0_1in: Optional[float] = None
    cbr_at_0_2in: Optional[float] = None
    soaked: Optional[bool] = None
    surcharge: Optional[ReadQuantity] = None
    compaction_percent: Optional[float] = None

    # -- chemical ----------------------------------------------------------
    pH: Optional[ReadReported] = None
    resistivity: Optional[ReadReported] = None
    resistivity_minimum: Optional[ReadReported] = None
    sulfate: Optional[ReadReported] = None
    chloride: Optional[ReadReported] = None
    sulfides: Optional[ReadReported] = None
    redox: Optional[ReadReported] = None
    total_salts: Optional[ReadReported] = None
    conductivity: Optional[ReadReported] = None
    temperature: Optional[ReadReported] = None
    reporting_limit: Optional[ReadReported] = None
    lab_sample_id: str = Field(default="")

    # -- the rest ----------------------------------------------------------
    series: List[ReadSeries] = Field(
        default_factory=list,
        description="every curve or table of pairs this test carries")
    rows: List[ReadRow] = Field(
        default_factory=list,
        description="the rows, when and only when kind is summary_table")
    no_results: bool = Field(
        default=False,
        description="this page reports no test result at all: a list of "
                    "samples received, a chain of custody, a cover sheet")
    fields: List[ReadField] = Field(
        default_factory=list,
        description="anything else the sheet printed that has no field here")
    prov: ReadProv


class Unsettled(BaseModel):
    """Something on this sheet you could not read."""

    model_config = ConfigDict(extra="forbid")

    what: str = Field(description="what it is, in 15 words or fewer")
    page: Optional[int] = Field(default=None)
    why: str = Field(description="what stopped you, in 20 words or fewer")


class LabSheetReading(BaseModel):
    """One laboratory sheet as the model read it."""

    model_config = ConfigDict(extra="forbid")

    depth_unit: str = Field(
        default="",
        description="'ft' or 'm' -- the unit EVERY depth you report is in. "
                    "Empty when the sheet prints no depth at all")
    language: str = Field(
        default="",
        description="a two-letter code when the sheet is not in English: "
                    "fr, es, pt. Empty for English")
    lab: str = Field(
        default="", description="the laboratory, when the sheet names one")
    standard: str = Field(
        default="",
        description="the standard the WHOLE sheet cites, when it cites one")
    tests: List[ReadTest] = Field(
        description="one entry per specimen tested, except a summary table, "
                    "which is ONE entry holding a row per specimen")
    pages_read: List[int] = Field(default_factory=list)
    unsettled: List[Unsettled] = Field(
        default_factory=list,
        description="everything on this sheet you could not read, and why")


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclass
class LabReadResult:
    """One sheet read, and everything needed to audit the reading."""

    tests: List[LabTest] = field(default_factory=list)
    #: Every value the reader took from the PICTURE rather than the text, and
    #: every curve it digitised. This is what a reviewer checks first.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    #: What was not settled: the model's own list, plus everything Python
    #: refused (an impossible depth, a percentage past 100, a liquid limit
    #: below the plastic limit, a grading series running the wrong way).
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)

    @property
    def kinds(self) -> List[str]:
        """The kinds this sheet turned out to hold, in order."""
        return [t.kind for t in self.tests]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tests": [t.model_dump(mode="json") for t in self.tests],
            "kinds": self.kinds,
            "changes": [dict(c) for c in self.changes],
            "unresolved": [dict(u) for u in self.unresolved],
            "cost": dict(self.cost),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "model": self.model,
            "warnings": list(self.warnings),
            "pages": list(self.pages),
        }


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

LAB_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "zoom_plot",
        "description": (
            "A magnified crop of one region of the page, rendered far finer "
            "than the whole-page picture you were given. Use it to read a "
            "PLOTTED curve, and only when the sheet tabulates those values "
            "nowhere. Include the axes in the box: a curve read without its "
            "own tick labels in view is a guess. Give the box in the same "
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
                f"page {page} is not part of this sheet ({self.pages})")
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
            f"Read the values against the axis ticks you can see, and set "
            f"digitised true on the series you build from them."),
            image_block(png)]

    def run(self, name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool]:
        """``(content, is_error)`` -- a tool mistake is an answer, not a stop."""
        handler = getattr(self, name, None)
        if handler is None or name not in {t["name"] for t in LAB_TOOLS}:
            return (f"unknown tool {name!r}; the tools are "
                    f"{[t['name'] for t in LAB_TOOLS]}", True)
        try:
            return handler(arguments), False
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

LAB_READER_SYSTEM = """\
You are reading ONE laboratory test sheet out of an engineering report and
returning what it says as data. Every laboratory prints its own form, in its
own language, so nothing about the layout can be assumed. What CAN be assumed
is that the sheet says what it did and what it found, in words, next to the
numbers.

WHAT YOU ARE GIVEN. Every line of text on the page with the box it occupies,
any tables that were detected, the ledger line the document reader wrote for
these pages, and the page as a picture. On a scanned sheet the text comes
from an optical reader and the tables come with it; it may be imperfect and
the picture is then worth more.

THE TEST KIND COMES FROM THE SHEET'S OWN TITLE. Not from the appendix it sits
in, not from what the numbers look like. Read the heading, the standard it
cites and the words on the results box, and choose from the kind vocabulary
you were given. A sheet that reports two different tests on the same specimen
-- a grading curve AND Atterberg limits, which is common -- is TWO entries,
one per kind, both carrying the same boring and depth. A sheet that reports
the same test on several specimens is one entry per specimen.

THE LINK TO THE GROUND IS WHAT THE SHEET PRINTS. Copy the boring, pit or
sounding identifier and the depth exactly as printed. Do not reconcile them
with anything, do not tidy them, do not guess one from a sample number. If
the sheet names no boring, leave it empty: that is an answer and something
else will deal with it.

READ THE TABLE BEFORE THE PLOT. Most of these sheets print a curve AND the
values that curve was drawn from, in a table or a results box beside it.
Those printed values are exact and the plot is not, so take the table every
time -- including when the table is harder to find. Digitise a curve ONLY
when the values appear nowhere in text on the sheet. When you do:
- call zoom_plot on the plot with its AXES AND TICK LABELS inside the box,
  and read the points against those ticks;
- report the points in the series with digitised set true;
- do not invent points between the ones that are marked; report the plotted
  points only.

UNITS STAY AS PRINTED. A pressure in psf is reported as psf, a density in
Mg/m3 as Mg/m3, a uniaxial strength in MPa as MPa. Never convert anything.
State the sheet's depth unit once, as 'ft' or 'm', and report every depth in
it unchanged.

VALUES THE SHEET PRINTS AS WORDS ARE RESULTS. '<10' is not the number ten:
put '<10' in the text field and leave the number null. So with 'Nil',
'trace', 'positive', 'N.P.', 'non-plastic'. A liquid limit that the sheet
leaves blank is null; a plastic limit printed NP sets non_plastic true.

A SUMMARY TABLE IS ONE ENTRY. A table headed something like "Summary of
Laboratory Test Results", with one row per specimen and a column per kind of
result, is a single entry of kind summary_table with one row per specimen --
not one entry per row and not one entry per column. Keep every column: a
column with no field of its own goes in that row's other list, under the
heading the table prints.

A PAGE WITH NO RESULTS IS STILL AN ANSWER. A laboratory certificate that
lists which samples were received, a chain of custody, a cover sheet: one
entry of kind other with no_results true, carrying whatever identifiers it
does print. Do not force it into a test kind.

WHAT NOT TO DO. Do not compute: no plasticity index the sheet did not print,
no percent fines added up from a sieve column, no unit conversion. Do not
give a specimen a USCS symbol the sheet does not print. Do not fill a field
from what sheets of this kind usually say. Anything you cannot settle goes in
unsettled with the reason, and that is a good answer.

PROVENANCE. Every entry carries the page it came from and, where you can, the
box of the line or cell that named it. Where the picture is what told you,
set from_image true.
"""

_FINAL_INSTRUCTION = (
    "Now give the sheet as data: every test on it, with the boring and depth "
    "printed on the sheet, the values in the units printed, the series you "
    "read or digitised, and anything you could not settle."
)

_NUDGE = (
    "You have {left} model call(s) left, including the one that gives the "
    "answer. Finish looking and get ready to report."
)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def _clip(text: str, n: int = MAX_LINE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


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
        turned = " [turned]" if (line.rotation or 0) else ""
        out.append(f"  {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}{turned} | "
                   f"{_clip(line.text)}")
    if len(lines) > MAX_LINES_PER_PAGE:
        out.append(f"  [{len(lines) - MAX_LINES_PER_PAGE} further line(s) "
                   f"not listed]")
    budget = MAX_TABLE_CHARS
    for table in content.tables:
        if budget <= 0:
            out.append("  [further table(s) not listed: the tables on this "
                       "page are longer than one sheet's worth]")
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


def _brief(doc: Any, pages: Sequence[int], ledger: Sequence[str],
           item_title: str, report_id: str, hint_kind: Optional[str],
           budget: int) -> str:
    vocabulary = "\n".join(f"  {name}: {text}"
                           for name, text in KIND_DEFINITIONS.items())
    series = "\n".join(f"  {name}: {text}"
                       for name, text in SERIES_NAMES.items())
    parts: List[str] = [
        f"ONE LABORATORY SHEET, on page(s) "
        f"{', '.join(str(p) for p in pages)}"
        + (f" of report {report_id}" if report_id else "") + ".",
        f"You may make {budget} model call(s) in all, the last of them your "
        f"answer.",
    ]
    if item_title:
        parts.append(f"The document titles it: {item_title}")
    if hint_kind:
        parts.append(
            f"The page labels suggest this is a {hint_kind!r} sheet. That is "
            f"a HINT from a rule that read the page's shape, not a fact: if "
            f"the sheet's own title says otherwise, the title wins.")
    parts += [
        "",
        "THE KIND VOCABULARY",
        vocabulary,
        "",
        "THE SERIES NAMES",
        series,
        "",
        "WHAT THE PAGE LEDGER SAYS ABOUT THESE PAGES",
        "\n".join(ledger) if ledger else "(no ledger line)",
        "",
        "THE PAGE",
    ]
    parts.extend(serialise_page(doc, page) for page in pages)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# turning the reading into the record
# ---------------------------------------------------------------------------

def _pct_ok(value: Optional[float]) -> bool:
    return value is None or 0.0 <= float(value) <= 100.0


class _Builder:
    """Turns one :class:`LabSheetReading` into :class:`LabTest` records.

    Every value passes one of four gates on the way, and a refusal costs that
    value and nothing else: a sheet with one impossible percentage still has
    twenty good numbers on it and the record should carry them.
    """

    def __init__(self, reading: LabSheetReading, pages: Sequence[int],
                 report_id: str, zooms: Sequence[Dict[str, Any]]) -> None:
        self.reading = reading
        self.pages = list(pages)
        self.report_id = report_id
        self.zooms = list(zooms)
        self.unit = (reading.depth_unit or "").strip()
        self.unresolved: List[Dict[str, Any]] = []
        self.changes: List[Dict[str, Any]] = []
        self._what = ""

    # -- the gates ---------------------------------------------------------
    def _refuse(self, what: str, why: str, value: Any = None,
                page: Optional[int] = None) -> None:
        row: Dict[str, Any] = {"what": f"{self._what}: {what}" if self._what
                               else what, "why": why, "refused_by": "python"}
        if value is not None:
            row["value"] = value
        if page is not None:
            row["page"] = page
        self.unresolved.append(row)

    def depth(self, value: Optional[float], what: str) -> Optional[Quantity]:
        """A depth as a Quantity, or None with the refusal recorded.

        The window is 0 to 300 m, converted: a laboratory specimen comes out
        of a hole, a hole has a positive depth, and no exploration in this
        corpus goes past a few hundred metres. A number outside it is a
        column read wrong, not a depth.
        """
        if value is None:
            return None
        value = float(value)
        quantity = Quantity(value=value, unit=self.unit)
        metres = quantity.si_value
        if metres is None:
            # An unrecognised depth unit cannot be checked against a window
            # in metres. The value is kept as printed -- that is the record's
            # rule -- and the fact that nothing checked it is recorded.
            self._refuse(what, f"depth unit {self.unit!r} is not one the "
                               f"record can convert, so the 0-{MAX_DEPTH_M:g} m "
                               f"check could not be made; the value is kept "
                               f"as printed", value)
            return quantity
        if not (0.0 <= metres <= MAX_DEPTH_M):
            self._refuse(what, f"depth {value:g} {self.unit or '(no unit)'} "
                               f"is {metres:.1f} m, outside 0 to "
                               f"{MAX_DEPTH_M:g} m; refused rather than "
                               f"accepted", value)
            return None
        return quantity

    def percent(self, value: Optional[float], what: str) -> Optional[float]:
        """A percentage, or None with the refusal recorded."""
        if value is None:
            return None
        if not _pct_ok(value):
            self._refuse(what, f"{value:g} is not a percentage; refused "
                               f"rather than accepted", value)
            return None
        return float(value)

    def quantity(self, read: Optional[ReadQuantity]) -> Optional[Quantity]:
        if read is None:
            return None
        return Quantity(value=float(read.value), unit=(read.unit or "").strip())

    def reported(self, read: Optional[ReadReported]) -> Any:
        """A value the sheet printed as a number or as words.

        The words win when there are words: a sheet that prints ``<10`` has
        NOT measured ten, and the string is the only honest record of it.
        """
        if read is None:
            return None
        text = (read.text or "").strip()
        if text:
            return text
        if read.value is None:
            return None
        unit = (read.unit or "").strip()
        if unit:
            return Quantity(value=float(read.value), unit=unit)
        return float(read.value)

    def atterberg_ok(self, ll: Optional[float], pl: Optional[float],
                     pi: Optional[float]) -> Tuple[Optional[float],
                                                   Optional[float],
                                                   Optional[float]]:
        """The three limits, or nothing, when the three cannot all be true.

        A liquid limit below the plastic limit is impossible, and the sheet
        did not print it: two columns have been swapped or one has been read
        off the wrong row. Which of the three is wrong cannot be known from
        here, so all three go -- keeping two of a bad triple would leave a
        plausible pair on the record.
        """
        if ll is None or pl is None:
            return ll, pl, pi
        if float(ll) < float(pl):
            self._refuse(
                "Atterberg limits",
                f"liquid limit {ll:g} is below plastic limit {pl:g}, which "
                f"no soil does; all three limits refused rather than one "
                f"guessed at", [ll, pl, pi])
            return None, None, None
        return ll, pl, pi

    # -- series ------------------------------------------------------------
    def sieve_points(self, series: ReadSeries) -> List[SievePoint]:
        """A grading series, checked for running the right way.

        More material cannot pass a SMALLER sieve than passed a larger one.
        A series that does is the curve read backwards, or two curves read as
        one, and it is refused whole: a partly-reversed grading is worse than
        none, because it looks like a reading.
        """
        points: List[Tuple[Optional[float], float, str]] = []
        for point in series.points:
            percent = point.y
            size = point.x
            if not _pct_ok(percent):
                self._refuse(
                    "grading series",
                    f"percent passing {percent:g} is not a percentage; the "
                    f"series was refused", percent)
                return []
            points.append((size, float(percent), point.label))
        with_size = [(s, p) for s, p, _lbl in points if s is not None]
        ordered = sorted(with_size, key=lambda sp: -sp[0])
        for (_s1, p1), (_s2, p2) in zip(ordered, ordered[1:]):
            if p2 > p1 + 0.51:
                self._refuse(
                    "grading series",
                    f"{p2:g}% passes a smaller sieve than the {p1:g}% that "
                    f"passed a larger one; the series runs the wrong way and "
                    f"was refused whole", [p1, p2])
                return []
        unit = (series.x_unit or "mm").strip()
        out: List[SievePoint] = []
        for size, percent, label in points:
            out.append(SievePoint(
                percent_passing=percent,
                size=(Quantity(value=float(size), unit=unit)
                      if size is not None else None),
                sieve=label,
                method="hydrometer" if "hydro" in label.lower() else ""))
        return out

    def series_named(self, test: ReadTest, name: str) -> Optional[ReadSeries]:
        for series in test.series:
            if (series.name or "").strip().lower() == name:
                return series
        return None

    # -- one test ----------------------------------------------------------
    def kind_of(self, test: ReadTest) -> str:
        kind = (test.kind or "").strip().lower().replace(" ", "_")
        if kind in RESULT_CLASS:
            return kind
        alias = {
            "consolidation": "swell_consolidation", "swell": "swell_consolidation",
            "oedometer": "swell_consolidation", "collapse": "swell_consolidation",
            "sieve": "gradation", "hydrometer": "gradation",
            "particle_size": "gradation", "grading": "gradation",
            "proctor": "compaction",
            "moisture": "moisture_content", "water_content": "moisture_content",
            "corrosivity": "chemical", "corrosion": "chemical",
            "unconfined_compression": "unconfined", "ucs": "unconfined_rock",
            "uu": "triaxial", "cu": "triaxial", "cd": "triaxial",
            "summary": "summary_table", "loss_on_ignition": "organic_content",
            "organic": "organic_content",
        }.get(kind)
        if alias:
            return alias
        if kind:
            self._refuse(
                "test kind", f"{test.kind!r} is not in the kind vocabulary; "
                             f"recorded as 'other'")
        return "other"

    def provenance(self, prov: ReadProv, digitised: bool) -> Provenance:
        page = int(prov.page)
        if page not in self.pages:
            self._refuse(
                "provenance",
                f"page {page} is not part of this sheet ({self.pages}); the "
                f"box was dropped", page=page)
            return Provenance(page=self.pages[0] if self.pages else page,
                              method="vision" if prov.from_image else "text",
                              confidence=0.4, note=prov.note)
        method = "vision" if (prov.from_image or digitised) else "text"
        if prov.from_image:
            self.changes.append({
                "what": self._what, "page": page,
                "why": prov.note or "read off the picture, not the text"})
        return Provenance(page=page,
                          bbox=tuple(prov.bbox) if prov.bbox else None,
                          method=method,
                          confidence=0.8 if method == "vision" else 0.9,
                          note=prov.note)

    def result_for(self, kind: str, test: ReadTest) -> Any:
        builder = {
            "atterberg": self._atterberg,
            "gradation": self._gradation,
            "swell_consolidation": self._consolidation,
            "triaxial": self._strength, "direct_shear": self._strength,
            "unconfined": self._strength, "unconfined_rock": self._strength,
            "compaction": self._compaction,
            "cbr": self._cbr,
            "moisture_content": self._moisture, "density": self._moisture,
            "organic_content": self._moisture,
            "chemical": self._chemical,
            "summary_table": self._summary,
        }.get(kind, self._other)
        return builder(kind, test)

    def _atterberg(self, kind: str, test: ReadTest) -> AtterbergResult:
        ll, pl, pi = self.atterberg_ok(
            self.percent(test.ll, "liquid limit"),
            self.percent(test.pl, "plastic limit"),
            self.percent(test.pi, "plasticity index"))
        flow = self.series_named(test, "flow_curve")
        return AtterbergResult(
            ll=ll, pl=pl, pi=pi,
            non_plastic=bool(test.non_plastic),
            shrinkage_limit=self.percent(test.shrinkage_limit,
                                         "shrinkage limit"),
            flow_curve=[(int(round(p.x)), float(p.y))
                        for p in (flow.points if flow else ())],
            pl_trials=[float(v) for v in test.pl_trials],
            water_content=self.percent(test.wc, "water content"),
            uscs=test.uscs.strip(), description=test.description)

    def _gradation(self, kind: str, test: ReadTest) -> GradationResult:
        series = self.series_named(test, "percent_passing")
        return GradationResult(
            percent_passing=(self.sieve_points(series) if series else []),
            d10=self.quantity(test.d10), d30=self.quantity(test.d30),
            d50=self.quantity(test.d50), d60=self.quantity(test.d60),
            d85=self.quantity(test.d85), d90=self.quantity(test.d90),
            d100=self.quantity(test.d100),
            cu=test.cu, cc=test.cc,
            cobbles_percent=self.percent(test.cobbles_percent, "cobbles"),
            gravel_percent=self.percent(test.gravel_percent, "gravel"),
            sand_percent=self.percent(test.sand_percent, "sand"),
            silt_percent=self.percent(test.silt_percent, "silt"),
            clay_percent=self.percent(test.clay_percent, "clay"),
            fines_percent=self.percent(test.fines_percent, "fines"),
            hydrometer=bool(test.hydrometer),
            water_content=self.percent(test.wc, "water content"),
            uscs=test.uscs.strip(), description=test.description)

    def _consolidation(self, kind: str, test: ReadTest) -> ConsolidationResult:
        series = self.series_named(test, "consolidation")
        points: List[ConsolidationPoint] = []
        for point in (series.points if series else ()):
            unit = (series.x_unit or "").strip()
            y_unit = (series.y_unit or "%").strip()
            points.append(ConsolidationPoint(
                stress=Quantity(value=float(point.x), unit=unit),
                strain_percent=(float(point.y) if y_unit == "%" else None),
                void_ratio=(None if y_unit == "%" else float(point.y)),
                stage=(point.label or "load")))
        return ConsolidationResult(
            test_type=test.test_type,
            points=points,
            swell_percent=test.swell_percent,
            swell_at=self.quantity(test.swell_at),
            swell_pressure=self.quantity(test.swell_pressure),
            pc=self.quantity(test.pc),
            cc=test.compression_index, cr=test.recompression_index,
            cv=self.quantity(test.cv), e0=test.e0,
            dry_unit_weight=self.quantity(test.dry_density),
            wc=self.percent(test.wc, "water content"),
            saturation_percent=self.percent(test.saturation_percent,
                                            "saturation"),
            uscs=test.uscs.strip(), description=test.description)

    def _strength(self, kind: str, test: ReadTest) -> StrengthResult:
        specimens = [
            StrengthSpecimen(
                specimen_id=s.specimen_id,
                confining=self.quantity(s.confining),
                peak_deviator=self.quantity(s.peak_deviator),
                strain_at_peak_percent=s.strain_at_peak_percent,
                pore_pressure=self.quantity(s.pore_pressure),
                stress_ratio=s.stress_ratio,
                c=self.quantity(s.c), phi_deg=s.phi_deg,
                wc=self.percent(s.wc, "specimen water content"),
                dry_density=self.quantity(s.dry_density),
                wet_density=self.quantity(s.wet_density),
                height=self.quantity(s.height),
                diameter=self.quantity(s.diameter), note=s.note)
            for s in test.specimens]
        points: List[ShearPoint] = []
        for name in ("shear_envelope", "stress_strain"):
            series = self.series_named(test, name)
            if series is None:
                continue
            x_unit = (series.x_unit or "").strip()
            y_unit = (series.y_unit or "").strip()
            points.extend(
                ShearPoint(x=Quantity(value=float(p.x), unit=x_unit),
                           y=Quantity(value=float(p.y), unit=y_unit),
                           specimen=p.label)
                for p in series.points)
        return StrengthResult(
            kind=kind, test_type=test.test_type or kind,
            specimens=specimens,
            c=self.quantity(test.c), phi_deg=test.phi_deg,
            c_residual=self.quantity(test.c_residual),
            phi_residual_deg=test.phi_residual_deg,
            qu=self.quantity(test.qu), su=self.quantity(test.su),
            strain_at_failure_percent=test.strain_at_failure_percent,
            points=points,
            wc=self.percent(test.wc, "water content"),
            dry_density=self.quantity(test.dry_density),
            wet_density=self.quantity(test.wet_density),
            rock_type=test.rock_type, weathering=test.weathering,
            uscs=test.uscs.strip(), description=test.description)

    def _compaction(self, kind: str, test: ReadTest) -> CompactionResult:
        series = self.series_named(test, "compaction")
        points: List[CompactionPoint] = []
        for point in (series.points if series else ()):
            water = self.percent(point.x, "compaction point water content")
            if water is None:
                continue
            points.append(CompactionPoint(
                water_content=water,
                dry_density=Quantity(value=float(point.y),
                                     unit=(series.y_unit or "").strip())))
        return CompactionResult(
            points=points,
            max_dry_density=self.quantity(test.max_dry_density),
            optimum_wc=self.percent(test.optimum_wc, "optimum water content"),
            method=test.method or test.test_type,
            mould_volume=self.quantity(test.mould_volume),
            blows_per_layer=test.blows_per_layer, layers=test.layers,
            rammer_mass=self.quantity(test.rammer_mass),
            uscs=test.uscs.strip(), description=test.description)

    def _cbr(self, kind: str, test: ReadTest) -> CBRResult:
        series = self.series_named(test, "cbr")
        return CBRResult(
            cbr_percent=test.cbr_percent,
            cbr_at_0_1in=test.cbr_at_0_1in, cbr_at_0_2in=test.cbr_at_0_2in,
            swell_percent=test.swell_percent, soaked=test.soaked,
            surcharge=self.quantity(test.surcharge),
            dry_density=self.quantity(test.dry_density),
            wc=self.percent(test.wc, "water content"),
            compaction_percent=test.compaction_percent,
            points=[(float(p.x), float(p.y))
                    for p in (series.points if series else ())],
            description=test.description)

    def _moisture(self, kind: str, test: ReadTest) -> MoistureDensityResult:
        return MoistureDensityResult(
            kind=kind,
            wc=self.percent(test.wc, "water content"),
            water_contents=[float(v) for v in test.water_contents],
            wet_density=self.quantity(test.wet_density),
            dry_density=self.quantity(test.dry_density),
            specific_gravity=test.specific_gravity,
            void_ratio=test.void_ratio,
            saturation_percent=self.percent(test.saturation_percent,
                                            "saturation"),
            ash_percent=self.percent(test.ash_percent, "ash"),
            organic_percent=self.percent(test.organic_percent, "organic"),
            uscs=test.uscs.strip(), description=test.description)

    def _chemical(self, kind: str, test: ReadTest) -> ChemicalResult:
        return ChemicalResult(
            pH=self.reported(test.pH),
            resistivity=self.reported(test.resistivity),
            resistivity_minimum=self.reported(test.resistivity_minimum),
            sulfate=self.reported(test.sulfate),
            chloride=self.reported(test.chloride),
            sulfides=self.reported(test.sulfides),
            redox=self.reported(test.redox),
            total_salts=self.reported(test.total_salts),
            conductivity=self.reported(test.conductivity),
            organic_percent=(self.percent(test.organic_percent, "organic")),
            temperature=self.reported(test.temperature),
            wc=(float(test.wc) if test.wc is not None else None),
            reporting_limit=self.reported(test.reporting_limit),
            lab_sample_id=test.lab_sample_id,
            description=test.description)

    def _summary(self, kind: str, test: ReadTest) -> SummaryTableResult:
        rows: List[SummaryRow] = []
        for n, row in enumerate(test.rows, start=1):
            self._what = f"summary row {n} ({row.investigation_id or '?'})"
            ll, pl_number, pi = self.atterberg_ok(
                self.percent(row.ll, "liquid limit"),
                (row.pl.value if row.pl is not None else None),
                self.percent(row.pi, "plasticity index"))
            passing: List[SievePoint] = []
            for entry in row.passing:
                percent = self.percent(entry.percent_passing,
                                       f"percent passing {entry.sieve}")
                if percent is None:
                    continue
                passing.append(SievePoint(
                    percent_passing=percent, sieve=entry.sieve,
                    size=(Quantity(value=float(entry.size_mm), unit="mm")
                          if entry.size_mm is not None else None)))
            pl_value: Any = self.reported(row.pl)
            if isinstance(pl_value, float) and pl_number is None:
                pl_value = None                  # refused with its triple
            rows.append(SummaryRow(
                investigation_id=row.investigation_id,
                sample_id=row.sample_id,
                depth_top=self.depth(row.depth_top, "row depth"),
                depth_bottom=self.depth(row.depth_bottom, "row base"),
                elevation_top=self.quantity(row.elevation_top),
                sample_type=row.sample_type, description=row.description,
                uscs=row.uscs.strip(), stratum=row.stratum, lab=row.lab,
                wc=self.percent(row.wc, "water content"),
                ll=ll, pl=pl_value, pi=pi,
                percent_passing=passing,
                fines_percent=self.percent(row.fines_percent, "fines"),
                sand_percent=self.percent(row.sand_percent, "sand"),
                gravel_percent=self.percent(row.gravel_percent, "gravel"),
                silt_clay_percent=self.percent(row.silt_clay_percent,
                                               "silt and clay"),
                wet_density=self.quantity(row.wet_density),
                dry_density=self.quantity(row.dry_density),
                max_dry_density=self.quantity(row.max_dry_density),
                optimum_wc=self.percent(row.optimum_wc, "optimum"),
                qu=self.quantity(row.qu), su=self.quantity(row.su),
                c=self.quantity(row.c), phi_deg=row.phi_deg,
                swell_percent=row.swell_percent,
                organic_percent=self.percent(row.organic_percent, "organic"),
                pH=self.reported(row.pH),
                resistivity=self.reported(row.resistivity),
                sulfate=self.reported(row.sulfate),
                chloride=self.reported(row.chloride),
                sulfides=self.reported(row.sulfides),
                redox=self.reported(row.redox),
                other=[(f.name, f.value) for f in row.other]))
        self._what = ""
        return SummaryTableResult(rows=rows, title=test.description,
                                  sheet=test.test_type)

    def _other(self, kind: str, test: ReadTest) -> OtherResult:
        fields: Dict[str, Any] = {f.name: f.value for f in test.fields}
        if test.lab_sample_id:
            fields.setdefault("lab_sample_id", test.lab_sample_id)
        return OtherResult(kind=(kind if kind in ("other", "specific_gravity",
                                                  "permeability") else "other"),
                           no_results=bool(test.no_results), fields=fields)

    # -- the whole sheet ---------------------------------------------------
    def build(self) -> List[LabTest]:
        out: List[LabTest] = []
        for n, test in enumerate(self.reading.tests, start=1):
            kind = self.kind_of(test)
            label = (f"{kind} {n} "
                     f"({test.investigation_id or 'no boring named'})")
            self._what = label
            digitised = any(s.digitised for s in test.series)
            result = self.result_for(kind, test)
            fields = {f.name: f.value for f in test.fields}
            lab_test = LabTest(
                kind=kind,
                investigation_id=test.investigation_id.strip(),
                sample_id=test.sample_id.strip(),
                depth_top=self.depth(test.depth_top, "depth"),
                depth_bottom=self.depth(test.depth_bottom, "base depth"),
                elevation=self.quantity(test.elevation),
                standard=(test.standard or self.reading.standard).strip(),
                lab=(test.lab or self.reading.lab).strip(),
                date=test.date.strip(),
                language=(self.reading.language or "").strip().lower()[:2],
                pages=list(self.pages),
                source_report=self.report_id,
                curves_digitised=digitised,
                result=result,
                fields={k: str(v) for k, v in fields.items()
                        if str(v or "").strip()},
                prov=[self.provenance(test.prov, digitised)])
            if digitised:
                self.changes.append({
                    "what": label, "page": test.prov.page,
                    "why": "a curve was digitised: the sheet tabulates these "
                           "values nowhere"})
            out.append(lab_test)
        self._what = ""
        for item in self.reading.unsettled:
            self.unresolved.append({
                "what": item.what, "page": item.page, "why": item.why,
                "refused_by": "reader"})
        for zoom in self.zooms:
            self.changes.append({
                "what": "zoom_plot", "page": zoom["page"],
                "why": zoom["why"] or "looked closely at a plot"})
        return out


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

def read_lab_sheet(doc, item_pages: Sequence[int], engine: Engine, *,
                   budget: int = MAX_MODEL_CALLS,
                   hint_kind: Optional[str] = None,
                   ledger: Optional[Sequence[str]] = None,
                   item_title: str = "", report_id: str = "",
                   dpi: float = PAGE_DPI) -> LabReadResult:
    """Read ONE laboratory sheet and return its tests as records.

    ``item_pages`` are the pages of one sheet or one multi-page test --
    ``planlens.document.roles.document_items`` gives them. ``budget`` is the
    model-call ceiling for this sheet, the answer included; the reader spends
    a call on :func:`zoom_plot` only when it asks for one.

    ``hint_kind`` is what the page labels think the sheet is. It is passed to
    the model as a hint and never as a fact: the label is a rule reading the
    page's shape, and the sheet's own title outranks it.
    """
    pages = [int(p) for p in item_pages]
    if not pages:
        raise ValueError("read_lab_sheet needs at least one page")
    budget = max(1, min(int(budget), MAX_MODEL_CALLS))

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
    images: List[bytes] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:                 # a page that will not read
            warnings.append(f"page {page}: {type(exc).__name__}: {exc}")
        else:
            warnings.extend(f"page {page}: {w}" for w in content.warnings)
        try:
            png, _info = doc.render(page, dpi=dpi)
        except Exception:                        # a page that will not draw
            continue                             # is read from its text alone
        images.append(png)

    brief = _brief(doc, pages, ledger, item_title, report_id, hint_kind,
                   budget)
    tools = _Tools(doc, pages)
    messages: List[Dict[str, Any]] = [
        user(text_block(brief), *[image_block(png) for png in images])]

    model_calls = 0
    tool_calls = 0
    reading: Optional[LabSheetReading] = None
    final: Any = None
    # EVERY call asks for the answer AND offers the tool, so a sheet whose
    # values are all tabulated -- which is most of them -- costs ONE call and
    # not two. A call is spent on looking only when the model asks to look.
    # On its last allowed call the tool is withdrawn, so a reader that keeps
    # zooming runs out of looking rather than out of answering.
    while model_calls < budget:
        last = model_calls == budget - 1
        reply = engine.complete(messages, system=LAB_READER_SYSTEM,
                                tools=None if last else LAB_TOOLS,
                                output_format=LabSheetReading)
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
            left = budget - model_calls
            if left <= 2:
                results.append(text_block(_NUDGE.format(left=left)))
            messages.append({"role": "user", "content": results})
            continue
        if reply.parsed is not None:
            reading = reply.parsed
            break
        # Neither a tool call nor an answer: say what is wanted, once, and
        # spend another call on it rather than returning nothing.
        messages.append({"role": "assistant",
                         "content": reply.content or [text_block("")]})
        messages.append(user(text_block(_FINAL_INSTRUCTION)))

    if reading is None:
        raise RuntimeError(
            f"the lab reader returned no structured reading for pages "
            f"{pages} in {model_calls} call(s) (stop_reason "
            f"{getattr(final, 'stop_reason', None)!r})")

    builder = _Builder(reading, pages, report_id, tools.zooms)
    tests = builder.build()
    spent["seconds"] = round(spent["seconds"], 2)
    spent["dollars"] = round(spent["dollars"], 5)
    return LabReadResult(
        tests=tests,
        changes=builder.changes,
        unresolved=builder.unresolved,
        cost=spent,
        model_calls=model_calls,
        tool_calls=tool_calls,
        model=getattr(final, "model", "") or getattr(engine, "name", ""),
        warnings=warnings,
        pages=pages)
