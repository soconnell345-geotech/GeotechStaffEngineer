"""The record: one report's contents as typed, cited data.

This is the product of the ingest. The summary page, the library page and the
DIGGS file are three exports of it, and nothing downstream reads a model's
prose -- it reads these fields.

THREE RULES HOLD THE WHOLE MODEL TOGETHER.

**A number keeps the unit it was printed in.** A log that prints 21.5 ft is
recorded as :class:`Quantity` ``(21.5, "ft")``, not as 6.5534 m. Conversion
happens once, in the writer that needs SI (:meth:`Quantity.to_si`), so a
reviewer setting the record beside the page sees the page's own numbers and a
rounding cannot compound through three hands. The unit is part of the value:
a :class:`Quantity` cannot be built without one.

**Every value-bearing field carries where it came from.** :class:`Provenance`
is the page, the box on that page, and HOW it was read -- the embedded text
layer, an Azure Document Intelligence result, optical character recognition,
the log grid's geometry, or a model looking at the picture. A reviewer can go
to the page and check; a QA pass can ask which values rest on vision alone.

**What could not be read is recorded, not guessed.** Every section has room
for the case where the page did not say: an optional field stays ``None``, a
:class:`QAEntry` says what was skipped and why, and
``Investigation.units_known`` is False when no depth unit was printed. A
missing value and a wrong value are different failures and the record keeps
them different.

WHAT IS A STUB AND WHAT IS NOT. WP2 fills :class:`Investigation` from the
boring logs, and that part is complete. :class:`LabTest` carries its result
as a free ``dict`` until WP3 gives each test kind a typed result;
:class:`NarrativeFacts` names the owner's two query schemas but leaves them to
WP4; :class:`CalcEntry` is WP5. They are here so the record's shape, its JSON
schema and its version do not change under the later packages -- a consumer
written against this schema keeps working as the stubs fill in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "SCHEMA_VERSION", "SI_UNITS", "UNIT_TO_SI",
    "Provenance", "Quantity", "Project", "DrillingDetails", "Layer", "Sample",
    "SPT", "WaterLevel", "LabTest", "Investigation", "NarrativeFacts",
    "GeneralFacts", "NaturalHazardFacts", "CalcEntry", "QAEntry",
    "DocumentFacts", "ReportRecord", "to_si", "record_json_schema",
]

#: The record's own version. A consumer stores it and can tell whether a file
#: predates a field it wants. Bumped when a field changes MEANING; adding an
#: optional field is not a bump.
SCHEMA_VERSION = "2.0"


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

#: The SI unit each quantity KIND converts to, and the kind's name.
SI_UNITS: Dict[str, str] = {
    "length": "m",
    "stress": "kPa",
    "unit_weight": "kN/m3",
    "blow_rate": "blows/0.3m",
    "angle": "deg",
    "ratio": "1",
    "percent": "%",
}

#: Standard gravity, for the three unit-of-mass-per-volume spellings a log
#: uses where it means a unit WEIGHT. g/cm3, Mg/m3 and t/m3 are densities;
#: multiplying by g is the geotechnical convention and the only way a density
#: and a pcf column can be compared, so it is done here and said out loud.
G = 9.80665

#: ``printed unit -> (kind, factor to the kind's SI unit)``. Deliberately
#: small: every unit in it appears on a log or a lab sheet in the corpus. A
#: unit that is not here is NOT converted -- :meth:`Quantity.to_si` returns
#: None rather than inventing a factor, and the caller records that.
UNIT_TO_SI: Dict[str, Tuple[str, float]] = {
    # length
    "m": ("length", 1.0),
    "mm": ("length", 0.001),
    "cm": ("length", 0.01),
    "ft": ("length", 0.3048),
    "in": ("length", 0.0254),
    # stress and pressure
    "kpa": ("stress", 1.0),
    "mpa": ("stress", 1000.0),
    "psf": ("stress", 0.04788025898033584),
    "psi": ("stress", 6.894757293168361),
    "tsf": ("stress", 95.76051796067168),
    "ksf": ("stress", 47.88025898033584),
    "kg/cm2": ("stress", 98.0665),
    "bar": ("stress", 100.0),
    # unit weight, and the densities a log means as one
    "kn/m3": ("unit_weight", 1.0),
    "pcf": ("unit_weight", 0.1570874606),
    "g/cm3": ("unit_weight", G),
    "mg/m3": ("unit_weight", G),
    "t/m3": ("unit_weight", G),
    # blow rate
    "blows/0.3m": ("blow_rate", 1.0),
    "blows/30cm": ("blow_rate", 1.0),
    "blows/ft": ("blow_rate", 0.9842519685039370),
    "blows/m": ("blow_rate", 0.3),
    # dimensionless and already-SI
    "deg": ("angle", 1.0),
    "%": ("percent", 1.0),
    "pct": ("percent", 1.0),
    "": ("ratio", 1.0),
}

#: Spellings that mean a unit already in the table. Applied after case
#: folding and after stripping spaces, so "Kg/cm²" and "KG / CM2" both land.
_UNIT_ALIASES: Dict[str, str] = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "millimeter": "mm", "millimeters": "mm", "millimetre": "mm",
    "centimeter": "cm", "centimeters": "cm",
    "feet": "ft", "foot": "ft", "'": "ft", "ft.": "ft",
    "inch": "in", "inches": "in", '"': "in", "in.": "in",
    "kn/m^3": "kn/m3", "kn/m³": "kn/m3", "knm3": "kn/m3",
    "lb/ft3": "pcf", "lb/ft^3": "pcf", "lbs/ft3": "pcf", "pounds/ft3": "pcf",
    "g/cc": "g/cm3", "g/cm^3": "g/cm3", "g/cm³": "g/cm3",
    "mg/m^3": "mg/m3", "mg/m³": "mg/m3",
    "kg/cm^2": "kg/cm2", "kg/cm²": "kg/cm2", "ksc": "kg/cm2",
    "kn/m2": "kpa", "kn/m^2": "kpa",
    "tons/ft2": "tsf", "tsf.": "tsf",
    "blows/foot": "blows/ft", "bpf": "blows/ft",
    "blows per foot": "blows/ft", "blows per 0.3m": "blows/0.3m",
    "degrees": "deg", "degree": "deg", "°": "deg",
    "percent": "%", "pct.": "%",
}


def _canonical_unit(unit: str) -> Optional[str]:
    """A printed unit folded to a key of :data:`UNIT_TO_SI`, or None.

    None means NOT RECOGNISED, which is different from the empty string: an
    empty unit is the dimensionless one and converts to itself, while
    "furlongs" has no factor here and must not be converted to anything.
    """
    text = str(unit or "").strip().lower()
    text = text.replace("²", "2").replace("³", "3")
    for candidate in (text, _UNIT_ALIASES.get(text),
                      text.replace(" ", ""),
                      _UNIT_ALIASES.get(text.replace(" ", ""))):
        if candidate is not None and candidate in UNIT_TO_SI:
            return candidate
    return None


def to_si(value: float, unit: str) -> Optional[Tuple[float, str]]:
    """``(value in SI, the SI unit)``, or None for a unit not in the table.

    None is an answer, not a failure: a writer that needs SI records that it
    could not convert rather than shipping a number in the wrong unit.
    """
    key = _canonical_unit(unit)
    if key is None:
        return None
    kind, factor = UNIT_TO_SI[key]
    return float(value) * factor, SI_UNITS[kind]


# ---------------------------------------------------------------------------
# provenance and quantities
# ---------------------------------------------------------------------------

#: How a value was read off the page, in falling order of exactness.
Method = Literal["text", "di", "ocr", "grid", "vision", "derived"]


class Provenance(BaseModel):
    """Where a value came from: the page, the box, and how it was read.

    ``bbox`` is ``(x0, y0, x1, y1)`` in displayed-page points with a top-left
    origin -- planlens' one frame, the same one ``render_region`` takes -- so
    a reviewer can be shown the exact patch of paper a number came off.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="x0, y0, x1, y1 in displayed-page points, top-left origin")
    method: Method = Field(
        default="text",
        description="text: the PDF's own text layer. di: Azure Document "
                    "Intelligence. ocr: on-machine optical reading. grid: "
                    "placed by the log grid's geometry. vision: a model read "
                    "it off the picture. derived: computed from other "
                    "recorded values, not printed")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="how sure the reader is, 0 to 1")
    note: str = Field(
        default="",
        description="anything a reviewer needs: what was ambiguous, what "
                    "settled it")


class Quantity(BaseModel):
    """A number with the unit it was printed in, and where it came from.

    Anything with a unit is a Quantity. A count, a ratio and a percentage are
    plain numbers elsewhere in the record; a depth, a pressure, a unit weight
    and a blow rate are Quantities, because the log's unit is the only thing
    that says what the number means.
    """

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="the number as printed")
    unit: str = Field(
        description="the unit as printed: ft, m, in, mm, psf, tsf, ksf, kPa, "
                    "kg/cm2, pcf, kN/m3, g/cm3, Mg/m3, blows/ft, blows/30cm")
    prov: Optional[Provenance] = Field(
        default=None, description="where this number was read")

    def to_si(self) -> Optional["Quantity"]:
        """The same value in SI, or None when the unit is not in the table.

        The provenance is carried through unchanged: converting does not
        change where the number came from.
        """
        got = to_si(self.value, self.unit)
        if got is None:
            return None
        value, unit = got
        return Quantity(value=value, unit=unit, prov=self.prov)

    @property
    def si_value(self) -> Optional[float]:
        """The number in SI, or None for an unconvertible unit."""
        converted = self.to_si()
        return None if converted is None else converted.value

    def __str__(self) -> str:                       # pragma: no cover - repr
        return f"{self.value:g} {self.unit}".strip()


# ---------------------------------------------------------------------------
# the report's identity
# ---------------------------------------------------------------------------

class Project(BaseModel):
    """The job the report is about, as the report prints it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="the project name")
    number: str = Field(
        default="", description="the project or contract number")
    client: str = Field(default="", description="who the report was for")
    location: str = Field(
        default="", description="the site, as printed; never inferred")
    coordinate_system: str = Field(
        default="",
        description="the CRS the investigation coordinates are in, when "
                    "printed (e.g. 'WGS 84', 'NAD83 / California zone 6')")
    elevation_datum: str = Field(
        default="", description="the vertical datum, when printed")
    prov: List[Provenance] = Field(
        default_factory=list, description="pages these came from")


class DocumentFacts(BaseModel):
    """What the FILE is: filled deterministically, never by a model."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(
        default="",
        description="the corpus ID (R01-R38) or another stable handle; never "
                    "a file name in a committed record")
    n_pages: int = Field(default=0, ge=0)
    page_roles: Dict[str, int] = Field(
        default_factory=dict,
        description="how many pages carry each role, from planlens")
    scan_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="fraction of pages with no reliable text layer")
    di_pages: int = Field(
        default=0, ge=0,
        description="pages read through Azure Document Intelligence")
    ocr_pages: int = Field(default=0, ge=0)
    planlens_version: str = Field(default="")
    workflow: str = Field(
        default="",
        description="the workflow triage chose: standard, appendix_only, "
                    "partial, multi_document, scanned, needs_a_person")
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    dollars: float = Field(default=0.0, ge=0.0)
    seconds: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# one exploration
# ---------------------------------------------------------------------------

#: What was put in the ground. The log's own word decides; a sounding that
#: the page does not name is ``other``.
InvestigationKind = Literal[
    "boring", "test_pit", "cpt", "dcp", "hand_auger", "well", "other"]


class DrillingDetails(BaseModel):
    """How the hole was made, as the log prints it.

    Every field is optional because every field is optional on a real log.
    The hammer matters most: an SPT N value means a different thing behind a
    safety hammer and behind an automatic one, and a record that drops the
    hammer has thrown away the correction.
    """

    model_config = ConfigDict(extra="forbid")

    method: str = Field(
        default="",
        description="hollow stem auger, mud rotary, wash boring, hand "
                    "excavation ... as printed")
    equipment: str = Field(
        default="", description="the rig, as printed")
    hole_diameter: Optional[Quantity] = None
    hammer_type: str = Field(
        default="",
        description="automatic, safety, donut, cathead ... as printed")
    hammer_energy_ratio: Optional[float] = Field(
        default=None, ge=0.0,
        description="the hammer's energy ratio as a PERCENT (e.g. 82 for an "
                    "82 % automatic hammer), when the log prints one")
    hammer_mass: Optional[Quantity] = None
    hammer_drop: Optional[Quantity] = None
    sampler: str = Field(
        default="", description="split spoon, ring, Shelby ... as printed")
    driller: str = Field(default="")
    contractor: str = Field(default="")
    logged_by: str = Field(default="")
    prov: List[Provenance] = Field(default_factory=list)


class Layer(BaseModel):
    """One described stratum: its top, its bottom and what it is."""

    model_config = ConfigDict(extra="forbid")

    top: Quantity = Field(description="depth to the top of the layer")
    bottom: Optional[Quantity] = Field(
        default=None,
        description="depth to the base; None on the last layer of a "
                    "continuation page, where the base is not printed")
    description: str = Field(
        default="", description="the log's own words, verbatim")
    uscs: str = Field(
        default="",
        description="the USCS group symbol the log prints (CL, SM, SP-SM); "
                    "empty when the log prints none -- never inferred from "
                    "the description")
    consistency: str = Field(
        default="", description="stiff, medium dense ... as printed")
    color: str = Field(default="", description="as printed")
    moisture: str = Field(default="", description="as printed")
    prov: Optional[Provenance] = None


#: What the sampler was. ``spt`` is a driven split spoon whose blows define an
#: N value; ``ring`` is a driven lined sampler whose blows do not.
SampleKind = Literal[
    "spt", "ring", "shelby", "bulk", "grab", "core", "cuttings", "other"]


class Sample(BaseModel):
    """One sample taken from the hole, and what was measured on it there."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(
        default="", description="the log's own label for it (S-1, 4, R-2)")
    top: Quantity = Field(description="depth to the top of the sample")
    bottom: Optional[Quantity] = Field(
        default=None, description="depth to the base, when printed")
    kind: SampleKind = Field(
        default="other",
        description="what the sampler was; 'other' when the log's symbol "
                    "cannot be read")
    recovery: Optional[Quantity] = Field(
        default=None, description="length recovered, when printed")
    recovery_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="recovery as a percentage, when the log prints one")
    rqd_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="rock quality designation, when printed")
    #: Index values printed ON THE LOG FACE beside the sample. A lab sheet's
    #: own copy of the same test is a LabTest and may disagree; the
    #: reconciler records the conflict rather than picking a winner.
    water_content: Optional[float] = Field(
        default=None, description="natural water content, percent")
    dry_unit_weight: Optional[Quantity] = None
    liquid_limit: Optional[float] = None
    plastic_limit: Optional[float] = None
    plasticity_index: Optional[float] = None
    fines_percent: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="percent passing the No. 200 sieve")
    qu: Optional[Quantity] = Field(
        default=None, description="unconfined compressive strength")
    pocket_pen: Optional[Quantity] = Field(
        default=None, description="pocket penetrometer reading")
    uscs: str = Field(
        default="", description="the group symbol printed against the sample")
    note: str = Field(default="")
    prov: Optional[Provenance] = None


class SPT(BaseModel):
    """One driven penetration record: the drives, and the N they define.

    ``blows`` is the record AS PRINTED, one entry per increment, and an entry
    is a string when the log printed one -- ``"50/5\\""`` is a refusal, not
    the number fifty. ``n`` is recorded only when the log PRINTS it; the
    record does no arithmetic, because half the templates print the drives
    and half print the N, and a reader that adds the second and third drives
    on a four-drive rock core would invent a number.
    """

    model_config = ConfigDict(extra="forbid")

    depth_top: Quantity = Field(description="depth at the start of the drive")
    depth_bottom: Optional[Quantity] = Field(
        default=None, description="depth at the end, when printed")
    blows: List[Union[int, str]] = Field(
        default_factory=list,
        description="one entry per increment, as printed; a string where the "
                    "log printed one, e.g. '50/5\"'")
    n: Optional[int] = Field(
        default=None, ge=0,
        description="the N value the log PRINTS; None when it prints only "
                    "the drives")
    refusal: bool = Field(
        default=False,
        description="the log records refusal at this depth (50 blows for "
                    "less than the full increment, or the word itself)")
    hammer: str = Field(
        default="",
        description="the hammer, when this record names one of its own; "
                    "otherwise the hole's hammer in DrillingDetails")
    sample_id: str = Field(
        default="", description="the sample this drive belongs to, when one")
    increment: Optional[Quantity] = Field(
        default=None, description="the length of each increment, when "
                                  "printed (typically 6 in or 0.15 m)")
    prov: Optional[Provenance] = None


#: When a water level was read. ``while_drilling`` and ``at_completion`` are
#: the two nearly every log prints; ``after_hours`` carries its own delay.
WaterWhen = Literal[
    "while_drilling", "at_completion", "after_hours", "not_encountered",
    "unknown"]


class WaterLevel(BaseModel):
    """One water observation in one hole."""

    model_config = ConfigDict(extra="forbid")

    depth: Optional[Quantity] = Field(
        default=None,
        description="depth to water; None when the log says none was "
                    "encountered")
    when: WaterWhen = Field(default="unknown")
    hours: Optional[float] = Field(
        default=None, ge=0.0,
        description="hours after completion, when 'after_hours'")
    date: str = Field(default="", description="as printed")
    casing_depth: Optional[Quantity] = None
    caved_depth: Optional[Quantity] = None
    elevation: Optional[Quantity] = None
    note: str = Field(default="")
    prov: Optional[Provenance] = None


class LabTest(BaseModel):
    """One laboratory test on one sample.

    WP3 gives each ``kind`` a typed result model. Until then ``kind`` is a
    string from a named list and ``result`` is a free dict, so a lab reader
    can fill the record today and a consumer written against this schema
    keeps working when the types arrive.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        description="atterberg, moisture, gradation, hydrometer, "
                    "consolidation, triaxial_uu, triaxial_cu, triaxial_cd, "
                    "direct_shear, unconfined, proctor, cbr, permeability, "
                    "specific_gravity, corrosivity, swell, collapse, "
                    "organic_content, other")
    investigation_id: str = Field(
        default="", description="the hole the sample came from, as printed")
    sample_id: str = Field(default="")
    depth: Optional[Quantity] = None
    depth_bottom: Optional[Quantity] = None
    standard: str = Field(
        default="", description="ASTM D4318, AASHTO T89, NF P94-051 ...")
    lab: str = Field(default="", description="who ran it, when printed")
    result: Dict[str, Any] = Field(
        default_factory=dict,
        description="WP3 STUB: the test's values, keyed by the name the "
                    "sheet gives them. Typed per kind in WP3")
    prov: List[Provenance] = Field(default_factory=list)


class Investigation(BaseModel):
    """One hole, pit or sounding: everything one log says about it.

    This is what the log reader returns and what the DIGGS writer turns into
    a sampling feature. ``units_known`` is the load-bearing flag: when the
    page printed no depth unit and none could be read, every depth in here is
    in whatever the ruler printed and must not be converted.
    """

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(
        default="", description="the log's own identifier (B-2, TP-4, CPT-1)")
    kind: InvestigationKind = Field(default="boring")
    depth_unit: str = Field(
        default="",
        description="the unit every depth in this investigation is in: 'ft' "
                    "or 'm'; empty when the page printed none")
    units_known: bool = Field(
        default=True,
        description="False when no depth unit was printed and none could be "
                    "read. Depths are then in the ruler's own numbers and "
                    "MUST NOT be converted")
    x: Optional[float] = Field(
        default=None, description="easting or longitude, as printed")
    y: Optional[float] = Field(
        default=None, description="northing or latitude, as printed")
    coordinate_system: str = Field(default="")
    elevation: Optional[Quantity] = Field(
        default=None, description="ground surface elevation, when printed")
    total_depth: Optional[Quantity] = None
    date_started: str = Field(default="", description="as printed")
    date_finished: str = Field(default="", description="as printed")
    drilling: DrillingDetails = Field(default_factory=DrillingDetails)
    layers: List[Layer] = Field(default_factory=list)
    samples: List[Sample] = Field(default_factory=list)
    spt: List[SPT] = Field(default_factory=list)
    water: List[WaterLevel] = Field(default_factory=list)
    remarks: str = Field(
        default="", description="the log's own notes, verbatim")
    station: str = Field(default="", description="as printed")
    offset: str = Field(default="", description="as printed")
    pages: List[int] = Field(
        default_factory=list,
        description="0-based PDF pages this log occupies, in order")
    source_report: str = Field(
        default="", description="the report ID this log came out of")
    sheet: str = Field(
        default="", description="the log's own 'page 1 of 3', as printed")
    fields: Dict[str, str] = Field(
        default_factory=dict,
        description="every other key-value the log's header printed, as "
                    "printed; nothing is dropped because the model has no "
                    "field for it")
    prov: List[Provenance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# the narrative (WP4 stubs) and the calcs (WP5 stub)
# ---------------------------------------------------------------------------

class GeneralFacts(BaseModel):
    """WP4 STUB: the owner's GENERAL query schema.

    The field names are the owner's, verbatim, so outputs stay comparable
    with the runs they have been making for years. WP4 fills them and adds
    the typed twins beside the strings (plan sections 3). Everything is
    optional; a report that does not answer a question leaves it None.
    """

    model_config = ConfigDict(extra="allow")

    answers: Dict[str, Any] = Field(
        default_factory=dict,
        description="WP4 STUB: the owner's general schema, field names "
                    "verbatim (documentType, quickSummary, projectNumber, "
                    "boringCount, recommendedFoundations, bearingCapacity "
                    "...)")
    citations: Dict[str, List[Provenance]] = Field(
        default_factory=dict,
        description="the pages each answer was read from, keyed by field")


class NaturalHazardFacts(BaseModel):
    """WP4 STUB: the owner's NATURAL HAZARDS query schema, names verbatim."""

    model_config = ConfigDict(extra="allow")

    answers: Dict[str, Any] = Field(
        default_factory=dict,
        description="WP4 STUB: liquefactionPotential, asceSevenVersion, "
                    "seismicCodeUsed, siteClass, reportDate ...")
    citations: Dict[str, List[Provenance]] = Field(default_factory=dict)


class NarrativeFacts(BaseModel):
    """WP4 STUB: both query schemas, and what the narrative counted."""

    model_config = ConfigDict(extra="forbid")

    general: GeneralFacts = Field(default_factory=GeneralFacts)
    natural_hazards: NaturalHazardFacts = Field(
        default_factory=NaturalHazardFacts)
    stated_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="counts the narrative STATES (borings, test pits, CPTs, "
                    "tables, figures). The reconciler sets them beside what "
                    "was found; a mismatch is a QA entry, not an error")


class CalcEntry(BaseModel):
    """WP5 STUB: one calculation printout."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="")
    method: str = Field(default="", description="what was calculated, and how")
    program: str = Field(default="", description="the program's own banner")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    pages: List[int] = Field(default_factory=list)
    prov: List[Provenance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# quality assurance
# ---------------------------------------------------------------------------

#: What a QA entry says happened. ``conflict`` is the important one: two
#: sources of the same value that disagree are RECORDED, never silently
#: resolved.
QAKind = Literal[
    "skipped", "partial", "conflict", "unreadable", "unconverted",
    "count_mismatch", "out_of_range", "note"]


class QAEntry(BaseModel):
    """One thing a reviewer needs to know about this record."""

    model_config = ConfigDict(extra="forbid")

    kind: QAKind
    where: str = Field(
        default="",
        description="what it is about: a page, an investigation id, a field "
                    "path")
    detail: str = Field(description="what happened, in a sentence")
    values: List[str] = Field(
        default_factory=list,
        description="for a conflict, the disagreeing values as printed")
    pages: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class ReportRecord(BaseModel):
    """One geotechnical report as data.

    Written as ``report.record.json``. The summary page, the library page and
    the DIGGS file are exports of this and add nothing that is not here.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    document: DocumentFacts = Field(default_factory=DocumentFacts)
    project: Project = Field(default_factory=Project)
    general: GeneralFacts = Field(default_factory=GeneralFacts)
    natural_hazards: NaturalHazardFacts = Field(
        default_factory=NaturalHazardFacts)
    narrative: NarrativeFacts = Field(default_factory=NarrativeFacts)
    investigations: List[Investigation] = Field(default_factory=list)
    lab_tests: List[LabTest] = Field(default_factory=list)
    calcs: List[CalcEntry] = Field(default_factory=list)
    qa: List[QAEntry] = Field(default_factory=list)

    def investigation(self, investigation_id: str) -> Optional[Investigation]:
        """The investigation with that id, or None."""
        for inv in self.investigations:
            if inv.investigation_id == investigation_id:
                return inv
        return None

    def counts(self) -> Dict[str, int]:
        """What is in this record, for the compact tool result."""
        return {
            "investigations": len(self.investigations),
            "layers": sum(len(i.layers) for i in self.investigations),
            "samples": sum(len(i.samples) for i in self.investigations),
            "spt": sum(len(i.spt) for i in self.investigations),
            "water_levels": sum(len(i.water) for i in self.investigations),
            "lab_tests": len(self.lab_tests),
            "calcs": len(self.calcs),
            "qa": len(self.qa),
        }


def record_json_schema() -> Dict[str, Any]:
    """The record's JSON schema, for the library agent and for a consumer.

    Exported rather than hand-written so it cannot drift from the models.
    """
    schema = ReportRecord.model_json_schema()
    schema["title"] = "ReportRecord"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema
