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
boring logs and WP3 fills :class:`LabTest` from the laboratory sheets; both
are complete, and a lab test's values are now a TYPED result per test kind
rather than a free dict. :class:`NarrativeFacts` names the owner's two query
schemas but leaves them to WP4; :class:`CalcEntry` is WP5. They are here so
the record's shape, its JSON schema and its version do not change under the
later packages -- a consumer written against this schema keeps working as the
stubs fill in.
"""

from __future__ import annotations

from typing import (
    Annotated, Any, Dict, List, Literal, Optional, Tuple, Union,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SCHEMA_VERSION", "SI_UNITS", "UNIT_TO_SI",
    "Provenance", "Alternative", "Quantity", "Project", "DrillingDetails",
    "Layer", "Sample",
    "SPT", "WaterLevel", "LabTest", "Investigation", "NarrativeFacts",
    "GeneralFacts", "NaturalHazardFacts", "CalcEntry", "QAEntry",
    "CALC_KINDS", "CalcKind", "NamedQuantity", "Calculation",
    "DocumentFacts", "LabelVote", "PageLabel", "ParentReport", "BoundReport",
    "ReportRecord", "to_si", "si_numbers",
    "record_json_schema",
    # the narrative (WP4): the owner's two schemas and their typed twins
    "Citation", "Mention", "BearingValue", "Stratum",
    "DOCUMENT_TYPE_VALUES", "PROPERTY_TYPE_VALUES", "PROJECT_PHASE_VALUES",
    "LIQUEFACTION_VALUES", "EARTH_HAZARD_VALUES", "YES_NO_UNCLEAR",
    "GENERAL_FIELDS", "NATURAL_HAZARD_FIELDS", "SUMMARY_FIELDS",
    "SUMMARY_WORD_LIMITS",
    # the typed laboratory results (WP3)
    "LabKind", "Reported", "LabResult", "RESULT_CLASS",
    "SievePoint", "AtterbergResult", "GradationResult",
    "ConsolidationPoint", "ConsolidationResult", "ShearPoint",
    "StrengthSpecimen", "StrengthResult", "CompactionPoint",
    "CompactionResult", "CBRResult", "MoistureDensityResult",
    "ChemicalResult", "SummaryRow", "SummaryTableResult", "OtherResult",
]

#: The record's own version. A consumer stores it and can tell whether a file
#: predates a field it wants. Bumped when a field changes MEANING; adding an
#: optional field is not a bump.
#:
#: 3.0 (WP3): ``LabTest.result`` stopped being a free dict and became a typed
#: result discriminated on ``kind``, and ``LabTest.depth`` became
#: ``depth_top``. Both are changes of MEANING, so the version moves; a 2.0
#: file's lab tests do not load as 3.0 ones.
#:
#: 4.0 (WP4): ``GeneralFacts`` and ``NaturalHazardFacts`` stopped being a free
#: ``answers`` dict and became the owner's two query schemas as REAL FIELDS,
#: field name for field name, with typed twins beside the strings; and
#: ``NarrativeFacts`` stopped carrying its own second copy of them -- the
#: record's own ``general`` and ``natural_hazards`` are the answers, and
#: ``narrative`` is what the reading produced BESIDE them (what the narrative
#: stated, what Python counted, what the appendix turned out to hold). Both
#: are changes of meaning, so the version moves.
#:
#: ``bound_documents`` and ``parent`` (a report bound inside another report,
#: read into its own record) are NOT a bump: both are optional, an older
#: record simply has an empty list and a null parent, and no existing field
#: changed meaning. What changed is what the pipeline now DOES with pages it
#: used to list and skip; see :mod:`report_ingest.bound`.
SCHEMA_VERSION = "4.0"


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
    # what a laboratory sheet adds (WP3)
    "resistivity": "ohm.m",
    "potential": "mV",
    "mass_fraction": "mg/kg",
    "temperature": "degC",
    "volume": "m3",
    "mass": "kg",
    "velocity": "m/s",
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
    # what a laboratory sheet adds. A corrosivity suite reports resistivity
    # in ohm-cm in the United States and in ohm.m almost everywhere else; a
    # redox potential is always millivolts; an ion is milligrams per kilogram
    # or, identically, parts per million by mass.
    "ohm.m": ("resistivity", 1.0),
    "ohm-cm": ("resistivity", 0.01),
    "kohm-cm": ("resistivity", 10.0),
    "mv": ("potential", 1.0),
    "v": ("potential", 1000.0),
    "mg/kg": ("mass_fraction", 1.0),
    "ppm": ("mass_fraction", 1.0),
    "degc": ("temperature", 1.0),
    "m3": ("volume", 1.0),
    "cm3": ("volume", 1e-6),
    "ml": ("volume", 1e-6),
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "lb": ("mass", 0.45359237),
    "m/s": ("velocity", 1.0),
    "cm/s": ("velocity", 0.01),
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
    # the laboratory's own spellings
    "ohm.cm": "ohm-cm", "ohmcm": "ohm-cm", "ohm cm": "ohm-cm",
    "ohm-centimeter": "ohm-cm", "ohm·cm": "ohm-cm", "ω-cm": "ohm-cm",
    "ohm-m": "ohm.m", "ohmm": "ohm.m", "ohm m": "ohm.m", "ohm·m": "ohm.m",
    "kohm.cm": "kohm-cm", "kohmcm": "kohm-cm", "kohm cm": "kohm-cm",
    "kilohm-cm": "kohm-cm", "k-ohm-cm": "kohm-cm",
    "millivolt": "mv", "millivolts": "mv", "volt": "v", "volts": "v",
    "mg/kilogram": "mg/kg", "milligram/kg": "mg/kg", "mg kg": "mg/kg",
    "parts per million": "ppm",
    "c": "degc", "°c": "degc", "deg c": "degc", "celsius": "degc",
    "cc": "cm3", "cm^3": "cm3", "cm³": "cm3", "millilitre": "ml",
    "milliliter": "ml", "m^3": "m3", "m³": "m3",
    "gram": "g", "grams": "g", "gm": "g",
    "kilogram": "kg", "kilograms": "kg",
    "lbs": "lb", "pound": "lb", "pounds": "lb",
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
#:
#: The readers VOTE since 5.23.0. ``grid`` and ``tables`` are the
#: deterministic first voter (the log grid's geometry, the page's detected
#: tables); ``model`` and ``model_from_picture`` are the second voter, a model
#: reading the rows or the picture; ``reconciled`` is a value both voters
#: gave, within the scorer's own tolerance. ``vision`` is kept for the older
#: files and the whole-page vision passes.
Method = Literal["text", "di", "ocr", "grid", "tables", "vision", "model",
                 "model_from_picture", "reconciled", "derived"]


class Alternative(BaseModel):
    """A value a second voter gave for the same slot, and did not win.

    A disagreement between the floor and the model is never settled in
    silence: the value in the record's slot is one voter's, and the other's
    stands here beside it with its own method and confidence, so a reviewer
    sees both and a QA entry sends them to look.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        default="",
        description="the slot this is an alternative for: 'n', 'top', "
                    "'uscs', 'll' ...; empty when the whole object is meant")
    value: str = Field(description="the other voter's value, as text")
    unit: str = Field(default="", description="its unit, when it had one")
    method: Method = Field(description="which voter gave it")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    note: str = Field(default="", description="what that voter said about it")


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
                    "placed by the log grid's geometry. tables: a detected "
                    "table on the page. model: a model read it off the rows "
                    "or the text. model_from_picture: a model read it off "
                    "the picture. reconciled: the grid or tables and the "
                    "model gave the same value. vision: a whole-page vision "
                    "pass. derived: computed from other recorded values, "
                    "not printed")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="how sure the reader is, 0 to 1")
    note: str = Field(
        default="",
        description="anything a reviewer needs: what was ambiguous, what "
                    "settled it")
    alternatives: List[Alternative] = Field(
        default_factory=list,
        description="what another voter said for this slot, when the voters "
                    "disagreed; the record keeps both and QA flags it")


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
    #: How the page labels were settled. See :mod:`report_ingest.label_vote`
    #: for the policies and :attr:`ReportRecord.page_labels` for the pages.
    label_policy: str = Field(
        default="",
        description="which policy combined the voters: trust, structural, "
                    "confidence, or rules for the rules alone")
    label_split_pages: int = Field(
        default=0, ge=0,
        description="pages the voters did not agree on")
    review_mode: str = Field(
        default="",
        description="which pages the label review was given: disagreements, "
                    "all, or none")
    review_changed: int = Field(
        default=0, ge=0,
        description="pages the label review moved")


# ---------------------------------------------------------------------------
# what each page IS, and who said so
# ---------------------------------------------------------------------------

class LabelVote(BaseModel):
    """One voter's view of one page.

    ``label`` is a page label for the rules and for the vision pass. It is
    EMPTY for the ``template`` voter, which recognises the printed FORM
    rather than the label -- its claim is ``family``, the firm whose log
    template the page came off, and what that claim is worth is the template
    rule in :func:`report_ingest.label_vote.combine`.
    """

    model_config = ConfigDict(extra="forbid")

    voter: str = Field(
        description="rules (planlens), vision (a model looking at the page), "
                    "or template (the printed form it was recognised as)")
    label: str = Field(
        default="",
        description="the page label this voter gave; empty for the template "
                    "voter, which names a form and not a label")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="how sure this voter was, 0 to 1")
    family: str = Field(
        default="",
        description="the template voter's claim: whose printed form this is")


class PageLabel(BaseModel):
    """What one page IS, how sure the record is, and who said so.

    The label the work items were built from, after the label review has had
    its say. ``agreed`` is the honest signal a production run can compute
    without any hand labels: the cheap voters concurring is what makes a
    label trustworthy, and their splitting is what sends the page to the
    expensive look.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index", ge=0)
    label: str = Field(description="what the record took this page to be")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="the highest confidence among the voters that gave this "
                    "label")
    agreed: bool = Field(
        default=True,
        description="did every voter with a view concur")
    policy: str = Field(
        default="",
        description="the policy that settled it: trust, structural, "
                    "confidence or rules")
    settled_by: str = Field(
        default="vote",
        description="vote, or review where the label review moved the page")
    voters: List[LabelVote] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# laboratory tests
# ---------------------------------------------------------------------------
#
# WHAT A TYPED RESULT IS FOR. A lab sheet prints numbers whose MEANING is the
# column they sit under: 31 under LIQUID LIMIT is a liquid limit, 31 under
# PERCENT PASSING is a proportion of a sample. Until WP3 a lab test's values
# were a free dict, so nothing downstream could ask for a liquid limit
# without first learning what this particular sheet had called it. Each kind
# of test now has a result class, and the class names the values.
#
# THE UNION IS DISCRIMINATED ON ``kind``, and every result class repeats the
# test kinds it answers for. So a record read back off disk rebuilds the
# right class without guessing, and a test whose kind and result disagree is
# refused at construction rather than discovered later by a reader of the
# JSON.
#
# THE UNIT RULE OF THE WHOLE RECORD HOLDS HERE. A value with a unit is a
# Quantity in the unit the sheet printed -- a confining pressure in psf stays
# in psf, a sieve opening in inches stays in inches. A value with no unit is
# a plain number: a percentage, a pH, a blow count, a ratio such as Cu or Cc.
# The DIGGS writer converts, once.

#: What a laboratory sheet IS. The vocabulary is the sheet's own title; a
#: sheet that does not say is ``other``, which is an answer.
LabKind = Literal[
    "atterberg",            # liquid, plastic and shrinkage limits
    "gradation",            # sieve and/or hydrometer grading
    "swell_consolidation",  # one-dimensional swell, collapse or oedometer
    "triaxial",             # UU, CU or CD
    "direct_shear",
    "unconfined",           # unconfined compression on soil
    "unconfined_rock",      # unconfined compression on a rock core
    "compaction",           # Proctor
    "cbr",
    "moisture_content",
    "density",
    "organic_content",      # loss on ignition, ash content
    "chemical",             # corrosivity: pH, resistivity, sulfate, chloride
    "specific_gravity",
    "permeability",
    "summary_table",        # a table of many samples' results
    "other",
]

#: A value a sheet prints as words rather than as a number -- ``"<10"``,
#: ``"Nil"``, ``"trace to positive"``, ``"N.P."``. It is kept as the string
#: the sheet printed, because "below the reporting limit" is not the number
#: ten, and a record that stored ten would be WRONG rather than incomplete.
Reported = Union[Quantity, float, str]


class SievePoint(BaseModel):
    """One point of a grading curve: a sieve, and what passed it."""

    model_config = ConfigDict(extra="forbid")

    percent_passing: float = Field(
        ge=0.0, le=100.0, description="percent finer than this size")
    size: Optional[Quantity] = Field(
        default=None,
        description="the opening, as printed (mm or in); None when the sheet "
                    "gives only a sieve designation")
    sieve: str = Field(
        default="",
        description="the sieve as the sheet names it: 'No. 200', '3/4 in', "
                    "'0.075 mm', '80 um'")
    method: str = Field(
        default="",
        description="sieve or hydrometer, when the sheet distinguishes them")


class AtterbergResult(BaseModel):
    """Liquid limit, plastic limit and plasticity index, as printed.

    All three are percentages and therefore plain numbers. ``non_plastic`` is
    the sheet printing NP rather than a figure, which is a RESULT -- this
    soil has no plastic limit -- and not a missing value.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["atterberg"] = "atterberg"
    ll: Optional[float] = Field(default=None, description="liquid limit, %")
    pl: Optional[float] = Field(default=None, description="plastic limit, %")
    pi: Optional[float] = Field(
        default=None, description="plasticity index, %")
    non_plastic: bool = Field(
        default=False, description="the sheet printed NP rather than limits")
    shrinkage_limit: Optional[float] = None
    flow_curve: List[Tuple[int, float]] = Field(
        default_factory=list,
        description="the Casagrande trials, (blow count, water content %), "
                    "in the order printed")
    pl_trials: List[float] = Field(
        default_factory=list,
        description="each plastic-limit determination, %")
    water_content: Optional[float] = Field(
        default=None, description="natural water content, %, when printed")
    uscs: str = Field(default="", description="as printed; never inferred")
    description: str = Field(default="", description="the sheet's own words")


class GradationResult(BaseModel):
    """A grading curve, and the numbers the sheet derives from it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["gradation"] = "gradation"
    percent_passing: List[SievePoint] = Field(
        default_factory=list,
        description="the curve, coarsest first, as printed or digitised")
    d10: Optional[Quantity] = None
    d30: Optional[Quantity] = None
    d50: Optional[Quantity] = None
    d60: Optional[Quantity] = None
    d85: Optional[Quantity] = None
    d90: Optional[Quantity] = None
    d100: Optional[Quantity] = Field(
        default=None, description="the largest particle size, when printed")
    cu: Optional[float] = Field(
        default=None, description="coefficient of uniformity, dimensionless")
    cc: Optional[float] = Field(
        default=None, description="coefficient of curvature, dimensionless")
    cobbles_percent: Optional[float] = None
    gravel_percent: Optional[float] = None
    sand_percent: Optional[float] = None
    silt_percent: Optional[float] = None
    clay_percent: Optional[float] = None
    fines_percent: Optional[float] = Field(
        default=None,
        description="silt and clay together, where the sheet prints one "
                    "number for them")
    hydrometer: bool = Field(
        default=False, description="a hydrometer was run as well as sieves")
    water_content: Optional[float] = None
    uscs: str = Field(default="", description="as printed; never inferred")
    description: str = Field(default="", description="the sheet's own words")


class ConsolidationPoint(BaseModel):
    """One load step: the pressure, and how far the specimen moved."""

    model_config = ConfigDict(extra="forbid")

    stress: Quantity = Field(description="the applied pressure, as printed")
    strain_percent: Optional[float] = Field(
        default=None,
        description="axial strain, %, keeping the SIGN the sheet plots it "
                    "with: swell one way, compression the other")
    void_ratio: Optional[float] = Field(
        default=None,
        description="e, where the sheet plots e rather than strain")
    stage: str = Field(
        default="load", description="load, unload or rebound, as printed")


class ConsolidationResult(BaseModel):
    """A one-dimensional swell, collapse or oedometer test.

    ``test_type`` is the sub-kind the sheet names -- a swell test and an
    oedometer are the same apparatus run for different answers -- and is a
    field of its own rather than the discriminator, which belongs to
    :attr:`LabTest.kind`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["swell_consolidation"] = "swell_consolidation"
    test_type: str = Field(
        default="",
        description="swell, collapse, oedometer or consolidation, as the "
                    "sheet names it")
    points: List[ConsolidationPoint] = Field(
        default_factory=list,
        description="the pressure-strain or pressure-void-ratio curve")
    swell_percent: Optional[float] = Field(
        default=None, description="percent swell, at the pressure below")
    swell_at: Optional[Quantity] = Field(
        default=None,
        description="the seating pressure the swell was measured at")
    swell_pressure: Optional[Quantity] = None
    pc: Optional[Quantity] = Field(
        default=None, description="preconsolidation pressure")
    cc: Optional[float] = Field(
        default=None, description="compression index, dimensionless")
    cr: Optional[float] = Field(
        default=None, description="recompression index, dimensionless")
    cv: Optional[Quantity] = Field(
        default=None, description="coefficient of consolidation")
    e0: Optional[float] = Field(default=None, description="initial void ratio")
    dry_unit_weight: Optional[Quantity] = None
    wc: Optional[float] = Field(default=None, description="water content, %")
    saturation_percent: Optional[float] = None
    uscs: str = Field(default="")
    description: str = Field(default="")


class ShearPoint(BaseModel):
    """One point of a failure envelope or of a stress-strain curve."""

    model_config = ConfigDict(extra="forbid")

    x: Quantity = Field(
        description="normal stress on an envelope; axial strain (unit '%') "
                    "on a stress-strain curve")
    y: Quantity = Field(
        description="shear stress on an envelope; deviator stress on a "
                    "stress-strain curve")
    specimen: str = Field(
        default="", description="which specimen this point belongs to")


class StrengthSpecimen(BaseModel):
    """One specimen of a strength test, as the sheet reports it."""

    model_config = ConfigDict(extra="forbid")

    specimen_id: str = Field(default="", description="1, 2, 3 or as printed")
    confining: Optional[Quantity] = Field(
        default=None, description="cell or normal pressure")
    peak_deviator: Optional[Quantity] = Field(
        default=None,
        description="peak deviator stress (triaxial), peak shear stress "
                    "(direct shear), or the failure stress")
    strain_at_peak_percent: Optional[float] = None
    pore_pressure: Optional[Quantity] = Field(
        default=None, description="pore pressure at failure, or its change")
    stress_ratio: Optional[float] = Field(
        default=None, description="maximum effective stress ratio")
    c: Optional[Quantity] = Field(
        default=None, description="cohesion, where reported per specimen")
    phi_deg: Optional[float] = None
    wc: Optional[float] = Field(default=None, description="water content, %")
    dry_density: Optional[Quantity] = None
    wet_density: Optional[Quantity] = None
    height: Optional[Quantity] = None
    diameter: Optional[Quantity] = None
    note: str = Field(default="")


class StrengthResult(BaseModel):
    """A triaxial, direct shear or unconfined compression test.

    One class for four kinds because a strength test has one shape of answer:
    specimens, each sheared under a confinement, and an envelope or a single
    strength read off them.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["triaxial", "direct_shear", "unconfined",
                  "unconfined_rock"] = "triaxial"
    test_type: str = Field(
        default="",
        description="UU, CU, CD, CDS, direct_shear, unconfined or "
                    "unconfined_rock, as the sheet names it")
    specimens: List[StrengthSpecimen] = Field(default_factory=list)
    c: Optional[Quantity] = Field(
        default=None, description="cohesion intercept of the envelope")
    phi_deg: Optional[float] = Field(
        default=None, description="friction angle of the envelope, degrees")
    c_residual: Optional[Quantity] = None
    phi_residual_deg: Optional[float] = None
    qu: Optional[Quantity] = Field(
        default=None, description="unconfined compressive strength")
    su: Optional[Quantity] = Field(
        default=None,
        description="undrained shear strength, where reported instead of qu")
    strain_at_failure_percent: Optional[float] = None
    points: List[ShearPoint] = Field(
        default_factory=list,
        description="the envelope, or the stress-strain curve, when the "
                    "sheet plots one")
    wc: Optional[float] = None
    dry_density: Optional[Quantity] = None
    wet_density: Optional[Quantity] = None
    rock_type: str = Field(default="", description="for a rock core")
    weathering: str = Field(default="")
    uscs: str = Field(default="")
    description: str = Field(default="")


class CompactionPoint(BaseModel):
    """One compaction trial: how wet it was, and how dense it came out."""

    model_config = ConfigDict(extra="forbid")

    water_content: float = Field(description="%")
    dry_density: Quantity = Field(description="as printed")


class CompactionResult(BaseModel):
    """A Proctor: the curve, its peak, and how it was run."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["compaction"] = "compaction"
    points: List[CompactionPoint] = Field(default_factory=list)
    max_dry_density: Optional[Quantity] = None
    optimum_wc: Optional[float] = Field(default=None, description="%")
    method: str = Field(
        default="",
        description="standard, modified, or the standard's own name")
    mould_volume: Optional[Quantity] = None
    blows_per_layer: Optional[int] = None
    layers: Optional[int] = None
    rammer_mass: Optional[Quantity] = None
    oversize_percent: Optional[float] = None
    uscs: str = Field(default="")
    description: str = Field(default="")


class CBRResult(BaseModel):
    """A laboratory California bearing ratio."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cbr"] = "cbr"
    cbr_percent: Optional[float] = Field(
        default=None, description="the reported CBR, %")
    cbr_at_0_1in: Optional[float] = None
    cbr_at_0_2in: Optional[float] = None
    swell_percent: Optional[float] = None
    soaked: Optional[bool] = None
    surcharge: Optional[Quantity] = None
    dry_density: Optional[Quantity] = None
    wc: Optional[float] = Field(default=None, description="%")
    compaction_percent: Optional[float] = Field(
        default=None, description="percent of maximum dry density")
    points: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="(penetration, load or stress) in the sheet's own units")
    description: str = Field(default="")


class MoistureDensityResult(BaseModel):
    """Water content, density, and the loss-on-ignition pair.

    One class for three kinds because these are the measurements a laboratory
    makes on a specimen before it does anything else to it, and a sheet that
    prints one usually prints the others beside it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["moisture_content", "density",
                  "organic_content"] = "moisture_content"
    wc: Optional[float] = Field(default=None, description="water content, %")
    water_contents: List[float] = Field(
        default_factory=list,
        description="each determination, where the sheet prints them and an "
                    "average")
    wet_density: Optional[Quantity] = None
    dry_density: Optional[Quantity] = None
    specific_gravity: Optional[float] = None
    void_ratio: Optional[float] = None
    saturation_percent: Optional[float] = None
    ash_percent: Optional[float] = Field(
        default=None, description="ash remaining after ignition, %")
    organic_percent: Optional[float] = Field(
        default=None, description="loss on ignition, %")
    uscs: str = Field(default="")
    description: str = Field(default="")


class ChemicalResult(BaseModel):
    """A corrosivity suite: pH, resistivity and the ions.

    Every field takes a string as well as a number, because these sheets
    print ``<10``, ``Nil``, ``trace`` and ``positive`` as often as they print
    figures, and those are results.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["chemical"] = "chemical"
    pH: Optional[Reported] = None
    resistivity: Optional[Reported] = Field(
        default=None, description="as printed: ohm-cm, kohm-cm, ohm.m")
    resistivity_minimum: Optional[Reported] = Field(
        default=None,
        description="the minimum-resistivity result, where the sheet reports "
                    "as-received and minimum")
    sulfate: Optional[Reported] = None
    chloride: Optional[Reported] = None
    sulfides: Optional[Reported] = None
    redox: Optional[Reported] = Field(
        default=None, description="redox potential, mV")
    total_salts: Optional[Reported] = None
    conductivity: Optional[Reported] = None
    organic_percent: Optional[Reported] = None
    temperature: Optional[Reported] = None
    wc: Optional[Reported] = Field(default=None, description="water content, %")
    reporting_limit: Optional[Reported] = Field(
        default=None,
        description="the laboratory's reporting limit, when printed, so a "
                    "'<' value can be read")
    lab_sample_id: str = Field(
        default="", description="the laboratory's own sample number")
    description: str = Field(default="")


class SummaryRow(BaseModel):
    """One specimen's line of a summary-of-laboratory-tests table.

    A row is not a test: it is one sample's results gathered from several.
    It is kept as a row because the table is a thing the report PRINTS, and
    the reconciler's job is to set it beside the per-sheet values, not to
    decide in advance which of the two is right.
    """

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(default="")
    sample_id: str = Field(default="")
    depth_top: Optional[Quantity] = None
    depth_bottom: Optional[Quantity] = None
    elevation_top: Optional[Quantity] = None
    sample_type: str = Field(default="")
    description: str = Field(default="")
    uscs: str = Field(default="")
    stratum: str = Field(default="")
    lab: str = Field(default="")
    wc: Optional[float] = Field(default=None, description="%")
    ll: Optional[float] = None
    pl: Optional[Reported] = Field(
        default=None, description="a number, or 'N.P.' as printed")
    pi: Optional[float] = None
    percent_passing: List[SievePoint] = Field(
        default_factory=list,
        description="the sieve columns this table carries, e.g. No. 4, "
                    "No. 40, No. 200")
    fines_percent: Optional[float] = None
    sand_percent: Optional[float] = None
    gravel_percent: Optional[float] = None
    silt_clay_percent: Optional[float] = None
    wet_density: Optional[Quantity] = None
    dry_density: Optional[Quantity] = None
    max_dry_density: Optional[Quantity] = None
    optimum_wc: Optional[float] = None
    qu: Optional[Quantity] = None
    su: Optional[Quantity] = None
    c: Optional[Quantity] = None
    phi_deg: Optional[float] = None
    swell_percent: Optional[float] = None
    organic_percent: Optional[float] = None
    pH: Optional[Reported] = None
    resistivity: Optional[Reported] = None
    sulfate: Optional[Reported] = None
    chloride: Optional[Reported] = None
    sulfides: Optional[Reported] = None
    redox: Optional[Reported] = None
    other: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="any column this row carries that has no field here, as "
                    "(the column's printed heading, the cell's text)")


class SummaryTableResult(BaseModel):
    """A whole summary-of-laboratory-tests table: one row per specimen."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["summary_table"] = "summary_table"
    rows: List[SummaryRow] = Field(default_factory=list)
    title: str = Field(default="", description="the table's own heading")
    sheet: str = Field(
        default="", description="its 'Sheet 1 of 4', as printed")


class OtherResult(BaseModel):
    """A sheet with no result class of its own, and the page that has none.

    ``no_results`` is the certificate page that lists which samples a
    laboratory received and reports nothing about them. It is a fact about
    the page, and recording it is how a reconciler knows the page was read
    and found empty rather than skipped.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["other", "specific_gravity", "permeability"] = "other"
    no_results: bool = Field(
        default=False,
        description="the page carries no test result: a sample list, a chain "
                    "of custody, a cover sheet")
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="everything the sheet printed, keyed by the name the "
                    "sheet gives it")


#: The union, discriminated on ``kind``. A result read back off disk rebuilds
#: its own class; a result whose kind does not match its test is refused by
#: :class:`LabTest`.
LabResult = Annotated[
    Union[AtterbergResult, GradationResult, ConsolidationResult,
          StrengthResult, CompactionResult, CBRResult, MoistureDensityResult,
          ChemicalResult, SummaryTableResult, OtherResult],
    Field(discriminator="kind"),
]

#: ``LabTest.kind -> the result class that answers for it``. Built from the
#: classes rather than written out a second time, so a result class cannot be
#: added without this table knowing about it.
RESULT_CLASS: Dict[str, Any] = {}
for _cls in (AtterbergResult, GradationResult, ConsolidationResult,
             StrengthResult, CompactionResult, CBRResult,
             MoistureDensityResult, ChemicalResult, SummaryTableResult,
             OtherResult):
    for _kind in _cls.model_fields["kind"].annotation.__args__:
        RESULT_CLASS[_kind] = _cls
del _cls, _kind


class LabTest(BaseModel):
    """One laboratory test, on one specimen, as one sheet reports it.

    A sheet reporting several specimens becomes several LabTests -- except a
    summary table, which is ONE test whose result holds a row per specimen,
    because the table is a document the report prints and splitting it would
    lose which values were printed together.

    The link to the ground is ``investigation_id`` plus ``depth_top``, both
    AS PRINTED ON THE SHEET. The reconciler matches them against the logs;
    this record never invents the link, because a lab sheet that names no
    boring is a real thing and a guessed link is worse than none.
    """

    model_config = ConfigDict(extra="forbid")

    kind: LabKind = Field(
        description="what the sheet's own title says the test is")
    investigation_id: str = Field(
        default="", description="the hole the sample came from, as printed")
    sample_id: str = Field(
        default="", description="the sample's own label, as printed")
    depth_top: Optional[Quantity] = Field(
        default=None, description="depth to the top of the specimen")
    depth_bottom: Optional[Quantity] = None
    elevation: Optional[Quantity] = None
    standard: str = Field(
        default="", description="ASTM D4318, BS 1377 Part 2, NF P94-051 ...")
    lab: str = Field(default="", description="who ran it, when printed")
    date: str = Field(default="", description="as printed; never normalised")
    language: str = Field(
        default="",
        description="the sheet's language as a two-letter code where it is "
                    "not English: fr, es, pt")
    pages: List[int] = Field(
        default_factory=list,
        description="0-based PDF pages this test was read from")
    source_report: str = Field(default="")
    curves_digitised: bool = Field(
        default=False,
        description="a value in the result was read off a PLOT rather than "
                    "a table. The whole test is flagged, because a reviewer "
                    "checks the sheet, not one number")
    result: Optional[LabResult] = Field(
        default=None,
        description="the typed result; None when the kind is known and the "
                    "values are not")
    fields: Dict[str, str] = Field(
        default_factory=dict,
        description="every other key-value the sheet printed, as printed")
    note: str = Field(default="")
    #: What the RECONCILER matched this sheet to, which is not the same thing
    #: as what the sheet printed. ``investigation_id`` above is the sheet's own
    #: words and never changes; these say which hole and which sample of the
    #: record they turned out to name, and stay empty when nothing matched.
    linked_investigation_id: str = Field(
        default="",
        description="the investigation this test was matched to, or empty")
    linked_sample_id: str = Field(
        default="", description="the sample it was matched to, or empty")
    linked_depth_delta_m: Optional[float] = Field(
        default=None,
        description="how far the sheet's depth sat from the sample's, in "
                    "metres; None when no sample matched")
    prov: List[Provenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def _result_matches_kind(self) -> "LabTest":
        """A result must answer for the kind it is attached to.

        Checked rather than trusted because ``kind`` is what every consumer
        dispatches on: an Atterberg result filed under a gradation test would
        be read as a gradation by everything downstream, and the mistake
        would be invisible in the JSON.
        """
        if self.result is not None and self.result.kind != self.kind:
            raise ValueError(
                f"a {self.kind!r} test cannot carry a {self.result.kind!r} "
                f"result; the result class for {self.kind!r} is "
                f"{RESULT_CLASS.get(self.kind, OtherResult).__name__}")
        return self


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

#: The owner's ``documentType`` enumeration, verbatim, and the same tuple
#: :mod:`report_ingest.triage` answers with -- one list, so the triage pass
#: and the narrative reader cannot drift apart on what a document may be.
DOCUMENT_TYPE_VALUES: Tuple[str, ...] = (
    "geotechnical report",
    "environmental report",
    "recommendation letter",
    "report addendum",
    "report appendix or figure(s)",
    "partial report",
    "other",
)

#: KNOWN ANSWERS, not a closed list. The owner's own word lists for these
#: questions were written for runs whose outputs are gone and are not in this
#: tree (plan section 7, item 1); these are the values the HAND ANSWERS for
#: eight reports actually use, read off them on 2026-09-17.
#:
#: The fields that take them are plain strings rather than Literals on
#: purpose. A first draft of this file GUESSED these lists -- government
#: owned / leased, planning / feasibility / design, high / moderate / low --
#: and every guess was wrong against the first hand answers that arrived. A
#: guessed vocabulary does not merely mislabel: it makes the schema refuse the
#: true answer, so the reader stores nothing and the scorer marks the field
#: wrong for a reader that read it correctly. A known-values list the reader
#: is shown, and folds a spelling variant onto, cannot fail that way.
PROPERTY_TYPE_VALUES: Tuple[str, ...] = (
    "New embassy or consulate compound",
    "Existing embassy or consulate compound",
    "other",
)
PROJECT_PHASE_VALUES: Tuple[str, ...] = (
    "Technical due diligence", "Bridging", "Design-build", "other",
)
LIQUEFACTION_VALUES: Tuple[str, ...] = (
    "not liquefiable", "potentially liquefiable",
)
#: Hazards as the hand names them: a short phrase for what the report says the
#: site is exposed to, in the report's own terms ("liquefaction-induced
#: settlement" rather than "liquefaction"). Examples for the reader, never a
#: filter on the answer.
EARTH_HAZARD_VALUES: Tuple[str, ...] = (
    "seismic shaking", "liquefaction", "liquefaction-induced settlement",
    "fault rupture", "landslide", "rockfall", "subsidence",
    "expansive soil", "collapsible soil", "karst", "erosion", "flooding",
    "tsunami", "volcanic",
)
#: What a question asked as a question gets answered with. ``mixed`` is the
#: hand's own: soil that is not corrosive to one thing and is to another is
#: neither a yes nor a no, and flattening it either way loses the finding.
YES_NO_UNCLEAR: Tuple[str, ...] = ("yes", "no", "mixed", "unclear")


class Citation(BaseModel):
    """Where an answer was read: the page, and the page's own words.

    A short quote rather than a box, because a narrative answer comes off a
    sentence and a reviewer checks it by finding that sentence. The quote is
    the page's own wording, never a paraphrase.
    """

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index")
    quote: str = Field(
        default="",
        description="the page's own words, 20 words or fewer, verbatim")


class Mention(BaseModel):
    """A yes/no/mixed/unclear verdict with the sentence that settles it."""

    model_config = ConfigDict(extra="forbid")

    answer: Literal["yes", "no", "mixed", "unclear"]
    citation: List[Citation] = Field(default_factory=list)


class BearingValue(BaseModel):
    """One bearing pressure the report recommends, as a number with a unit.

    The typed twin of the ``bearingCapacity`` strings. A pressure means
    nothing without what it is for -- a strip footing on engineered fill and
    a mat on residual soil are different recommendations -- so the foundation
    and the condition travel with the number.
    """

    model_config = ConfigDict(extra="forbid")

    value: Quantity = Field(description="the pressure, in the unit printed")
    foundation_type: str = Field(
        default="",
        description="what it is for, as printed: spread footing, mat, "
                    "drilled shaft end bearing ...")
    condition: str = Field(
        default="",
        description="the condition it applies under, as printed: 'on "
                    "engineered fill', 'net allowable', 'at 1.5 m depth'")
    citation: List[Citation] = Field(default_factory=list)


class Stratum(BaseModel):
    """One stratum of the ``strata`` answer, as a record rather than prose."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="", description="the report's own name for it, as printed")
    description: str = Field(default="", description="as printed")
    top: Optional[Quantity] = Field(
        default=None, description="depth or elevation to its top, when given")
    bottom: Optional[Quantity] = Field(default=None)
    uscs: str = Field(
        default="",
        description="the group symbol the report prints for it; never "
                    "inferred from the description")
    citation: List[Citation] = Field(default_factory=list)


#: The owner's general schema, in the owner's order. Kept as a tuple so a
#: writer, a scorer and a prompt can walk the same list rather than three.
GENERAL_FIELDS: Tuple[str, ...] = (
    "documentType", "quickSummary", "postName", "propertyType",
    "projectNumber", "projectName", "projectPhase", "primeContractor",
    "primeAe", "geotechnicalEngineerFirm", "testingProgramSummary",
    "boringCount", "testPitCount", "cptCount", "tableCount", "figureCount",
    "previousInvestigationCount", "strata", "structureCount", "structureList",
    "outsideProject", "boringDictionary", "testPitDictionary",
    "recommendedFoundations", "bearingCapacity",
)

#: The owner's natural-hazards schema, in the owner's order.
NATURAL_HAZARD_FIELDS: Tuple[str, ...] = (
    "liquefactionPotential", "asceSevenVersion", "earthHazardsExposed",
    "seismicCodeUsed", "geophysicalTestingMention", "soilCorrosion",
    "siteResponseMention", "hazardAnalysisMention", "siteClass",
    "seismicParameterSummary", "naturalHazardSummary", "reportDate",
)

#: The four fields that are PROSE. They are scored for presence and length
#: only: whether a summary is a good summary is a person's call, and a
#: scorer that pretended otherwise would be scoring its own opinion.
SUMMARY_FIELDS: Tuple[str, ...] = (
    "quickSummary", "testingProgramSummary", "naturalHazardSummary",
    "seismicParameterSummary",
)
#: How long each may run, in words.
SUMMARY_WORD_LIMITS: Dict[str, int] = {
    "quickSummary": 100,
    "testingProgramSummary": 100,
    "naturalHazardSummary": 200,
    "seismicParameterSummary": 100,
}


class GeneralFacts(BaseModel):
    """The owner's GENERAL query schema, field name for field name.

    The names are the owner's, verbatim and camel-cased against every
    convention in the rest of this package, so that outputs stay comparable
    with the runs they have been making for years. Renaming them to suit this
    codebase would break the only continuity the answers have.

    EVERY FIELD IS OPTIONAL AND NONE MEANS NOT STATED. A report that does not
    say who the prime contractor was leaves ``primeContractor`` None; it never
    carries an empty string, a "not stated" string or a guess, because a
    scorer has to be able to tell a report that did not say from a reader
    that did not read.

    THE TYPED TWINS SIT BESIDE THE STRINGS, NEVER INSTEAD OF THEM.
    ``bearingCapacity`` keeps the sentences the report printed and
    ``bearingCapacityValues`` carries the same recommendations as numbers with
    units; ``strata`` keeps the one-string dictionary and ``strataList``
    carries it as records. A consumer that wants arithmetic reads the twin; a
    reviewer checking the reading reads the string.
    """

    model_config = ConfigDict(extra="forbid")

    # -- what the document is ---------------------------------------------
    documentType: Optional[Literal[
        "geotechnical report", "environmental report",
        "recommendation letter", "report addendum",
        "report appendix or figure(s)", "partial report", "other"]] = Field(
        default=None, description="what kind of document this is")
    quickSummary: Optional[str] = Field(
        default=None,
        description="what the report is and what it concluded, 100 words or "
                    "fewer")
    postName: Optional[str] = Field(
        default=None,
        description="the post the site belongs to, as the report names it; "
                    "never inferred from a place name")
    propertyType: Optional[str] = Field(
        default=None,
        description="what the property is, in the owner's own words; "
                    "PROPERTY_TYPE_VALUES lists the ones seen so far")
    projectNumber: Optional[str] = Field(
        default=None, description="the project or job number, as printed")
    projectName: Optional[str] = Field(default=None, description="as printed")
    projectPhase: Optional[str] = Field(
        default=None,
        description="the phase the work was done for, in the owner's own "
                    "words; PROJECT_PHASE_VALUES lists the ones seen so far")
    primeContractor: Optional[str] = Field(default=None)
    primeAe: Optional[str] = Field(
        default=None, description="the prime architect-engineer")
    geotechnicalEngineerFirm: Optional[str] = Field(
        default=None, description="the firm that wrote the report")

    # -- what was done -----------------------------------------------------
    testingProgramSummary: Optional[str] = Field(
        default=None,
        description="what was drilled, dug, sounded and tested, 100 words or "
                    "fewer")
    boringCount: Optional[int] = Field(default=None, ge=0)
    testPitCount: Optional[int] = Field(default=None, ge=0)
    cptCount: Optional[int] = Field(default=None, ge=0)
    tableCount: Optional[int] = Field(default=None, ge=0)
    figureCount: Optional[int] = Field(default=None, ge=0)
    previousInvestigationCount: Optional[int] = Field(default=None, ge=0)
    strata: Optional[str] = Field(
        default=None,
        description="the soil profile the report describes, as one string; "
                    "the twin is strataList")
    structureCount: Optional[int] = Field(default=None, ge=0)
    structureList: Optional[List[str]] = Field(
        default=None, description="the structures the report is about")
    outsideProject: Optional[Literal["yes", "no", "unclear"]] = Field(
        default=None,
        description="is this a report about somebody else's project, from "
                    "outside")
    boringDictionary: Optional[List[str]] = Field(
        default=None,
        description="the boring identifiers the narrative names, as printed")
    testPitDictionary: Optional[List[str]] = Field(default=None)
    recommendedFoundations: Optional[List[str]] = Field(
        default=None, description="the foundation types recommended")
    bearingCapacity: Optional[List[str]] = Field(
        default=None,
        description="the bearing recommendations as the report words them; "
                    "the twin is bearingCapacityValues")

    # -- the typed twins ---------------------------------------------------
    bearingCapacityValues: List[BearingValue] = Field(
        default_factory=list,
        description="the same recommendations as numbers with units")
    strataList: List[Stratum] = Field(
        default_factory=list, description="the same profile as records")
    outsideProjectAnswer: Optional[Mention] = Field(
        default=None, description="outsideProject with its citation")

    #: The pages and words each answer was read from, keyed by the owner's
    #: field name. A field with no entry here was not answered.
    citations: Dict[str, List[Citation]] = Field(default_factory=dict)

    def answered(self) -> List[str]:
        """The owner's fields this report actually answered, in order."""
        return [name for name in GENERAL_FIELDS
                if getattr(self, name, None) is not None]


class NaturalHazardFacts(BaseModel):
    """The owner's NATURAL HAZARDS query schema, names verbatim.

    Same rules as :class:`GeneralFacts`: None means the report did not say,
    the owner's enumerations are kept as written, and the typed twins sit
    beside the strings rather than replacing them.
    """

    model_config = ConfigDict(extra="forbid")

    liquefactionPotential: Optional[str] = Field(
        default=None,
        description="what the report concluded about liquefaction, in the "
                    "owner's words: 'not liquefiable', 'potentially "
                    "liquefiable', or the report's own verdict")
    asceSevenVersion: Optional[str] = Field(
        default=None, description="the ASCE 7 edition cited, as printed")
    earthHazardsExposed: Optional[List[str]] = Field(
        default=None,
        description="the hazards the report says the site is exposed to, one "
                    "short phrase each in the report's own terms; an empty "
                    "list means it names none")
    seismicCodeUsed: Optional[str] = Field(
        default=None, description="the seismic code or standard, as printed")
    # THE FOUR VERDICT QUESTIONS. Each is answered "yes", "no", "mixed" or
    # "unclear", optionally followed by " - " and what was found: the hand's
    # own answers read "yes - six seismic refraction lines across the site".
    # The verdict is the answer and the reason is what makes it useful, so the
    # field keeps both and the bare verdict is carried in the twin beside it.
    geophysicalTestingMention: Optional[str] = Field(
        default=None,
        description="does the report mention geophysical testing: yes / no / "
                    "mixed / unclear, optionally ' - ' and what was done")
    soilCorrosion: Optional[str] = Field(
        default=None,
        description="does it address soil corrosivity: yes / no / mixed / "
                    "unclear, optionally ' - ' and what it found")
    siteResponseMention: Optional[str] = Field(
        default=None,
        description="does it mention a site response analysis: yes / no / "
                    "mixed / unclear, optionally ' - ' and which")
    hazardAnalysisMention: Optional[str] = Field(
        default=None,
        description="does it mention a seismic hazard analysis, "
                    "probabilistic or deterministic: yes / no / mixed / "
                    "unclear, optionally ' - ' and which")
    siteClass: Optional[str] = Field(
        default=None, description="the site class, as printed")
    seismicParameterSummary: Optional[str] = Field(
        default=None,
        description="the seismic design parameters, 100 words or fewer")
    naturalHazardSummary: Optional[str] = Field(
        default=None,
        description="what the report says about natural hazards, 200 words "
                    "or fewer")
    reportDate: Optional[str] = Field(
        default=None,
        description="the date on the report, as printed; the twin is "
                    "reportDateISO")

    # -- the typed twins ---------------------------------------------------
    siteClassNormalized: Optional[str] = Field(
        default=None,
        description="the site class as a bare letter or letter pair: A, B, "
                    "BC, C, CD, D, DE, E, F")
    asceSevenVersionNormalized: Optional[str] = Field(
        default=None,
        description="the ASCE 7 edition as '7-16', '7-22'; None when the "
                    "printed text names no edition")
    reportDateISO: Optional[str] = Field(
        default=None, description="the report date as YYYY-MM-DD")
    geophysicalTestingAnswer: Optional[Mention] = None
    soilCorrosionAnswer: Optional[Mention] = None
    siteResponseAnswer: Optional[Mention] = None
    hazardAnalysisAnswer: Optional[Mention] = None

    citations: Dict[str, List[Citation]] = Field(default_factory=dict)

    def answered(self) -> List[str]:
        """The owner's fields this report actually answered, in order."""
        return [name for name in NATURAL_HAZARD_FIELDS
                if getattr(self, name, None) is not None]


class NarrativeFacts(BaseModel):
    """What the narrative reading produced BESIDE the two schemas.

    The answers themselves live on the record as ``general`` and
    ``natural_hazards``; there is one copy of them and this is not it. What
    is here is the arithmetic around them: what the narrative SAID was done,
    what Python COUNTED off the pages, and what the appendix turned out to
    hold. The reconciler sets the three side by side, and a disagreement is a
    QA entry rather than a correction -- a report that says four borings and
    carries five logs is telling a reviewer something.
    """

    model_config = ConfigDict(extra="forbid")

    stated_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="counts the narrative STATES: borings, test_pits, cpts, "
                    "tables, figures, previous_investigations, structures")
    counted: Dict[str, int] = Field(
        default_factory=dict,
        description="what Python counted off the pages deterministically: "
                    "tables and figures from the captions in the main body, "
                    "and from the printed lists of tables and figures")
    found_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="what the appendix turned out to hold, filled by the "
                    "reconciler: borings, test_pits, cpts, dcps, lab_tests")
    found_ids: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="the identifiers actually found, by kind, filled by the "
                    "reconciler")
    pages: List[int] = Field(
        default_factory=list,
        description="the narrative pages that were read")
    extra_answers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="questions asked of THIS report beyond the two schemas -- "
                    "the caller's own -- each as {question, answer, page, "
                    "quote}, with an empty answer where the narrative does "
                    "not say")
    unresolved: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="what the reader could not settle, and why")


class CalcEntry(BaseModel):
    """The WP5 STUB, superseded by :class:`Calculation` and kept for files.

    Records written before the calculation reader carry ``calcs``; nothing
    writes it now. :attr:`ReportRecord.calculations` is where a calculation
    read off the pages goes, and this stays so a file written against the
    stub still loads.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="")
    method: str = Field(default="", description="what was calculated, and how")
    program: str = Field(default="", description="the program's own banner")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    pages: List[int] = Field(default_factory=list)
    prov: List[Provenance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# calculations (WP5)
# ---------------------------------------------------------------------------

#: What a calculation printout WORKS OUT. The vocabulary is about the
#: QUESTION, not about the program that answered it: a settlement worked on a
#: spreadsheet and one worked by a commercial program are the same kind of
#: calculation, and a reviewer asking "what did they assume for settlement"
#: wants both. A printout that is none of these is ``other``, which is an
#: answer rather than a failure.
CALC_KINDS: Tuple[str, ...] = (
    "lateral_pile", "axial_pile", "pile_group",
    "shallow_foundation_bearing", "settlement", "slope_stability",
    "retaining_wall", "liquefaction", "site_response", "seepage",
    "pavement", "ground_improvement", "other",
)

CalcKind = Literal[
    "lateral_pile", "axial_pile", "pile_group",
    "shallow_foundation_bearing", "settlement", "slope_stability",
    "retaining_wall", "liquefaction", "site_response", "seepage",
    "pavement", "ground_improvement", "other"]


class NamedQuantity(BaseModel):
    """One labelled value a calculation printed: the label, and the value.

    A calculation sheet is a list of labelled numbers -- ``Footing Width B
    (ft)  25.8``, ``Total Settlement (inches)  0.74``, ``PCC Thickness
    5.46 inches`` -- so the PRINTED LABEL paired with the value is the whole
    of what this record holds. The label stays the sheet's own words: a
    reviewer goes looking for what the page says, not for a name this
    package invented for it.

    A value is either a number with its unit (:attr:`value`) or a word
    (:attr:`text`): a seismic site class is ``C``, a stability check prints
    ``OK``, a bearing check prints ``Adequate``. One of the two is always
    set, and turning ``C`` into a number would be a lie.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="the label printed beside the value, as printed")
    value: Optional[Quantity] = Field(
        default=None,
        description="the number and the unit as printed; None when the page "
                    "prints a word instead")
    text: str = Field(
        default="",
        description="the value as WORDS, when that is what the page prints: "
                    "'C', 'OK', 'Adequate', 'Site Class D'")
    note: str = Field(
        default="", description="anything a reviewer needs about this value")
    prov: Optional[Provenance] = Field(
        default=None, description="the page and the box it came off")

    @model_validator(mode="after")
    def _has_a_value(self) -> "NamedQuantity":
        if self.value is None and not self.text.strip():
            raise ValueError(
                "a NamedQuantity carries either a Quantity or the words the "
                "page printed; one with neither is not a value")
        return self

    def __str__(self) -> str:                       # pragma: no cover - repr
        shown = str(self.value) if self.value is not None else self.text
        return f"{self.name}: {shown}".strip()


class Calculation(BaseModel):
    """One calculation printout, as its pages print it.

    A quarter of the hand-labelled pages of this corpus are calculations: a
    program's own output, a spreadsheet printed to PDF, a sheet worked by
    hand and scanned. They are the design itself -- what was assumed, what
    was worked out, what was concluded -- and until this train the whole
    class was listed in a QA entry and skipped.

    NOTHING IS COMPUTED HERE. A factor of safety the printout did not print
    stays absent, a unit the page did not state is not supplied, and a kind
    the page does not support is ``other``. The record says what the paper
    says.
    """

    model_config = ConfigDict(extra="forbid")

    kind: CalcKind = Field(
        description="what this calculation works out, from the controlled "
                    "list; 'other' when it is none of them")
    program: Optional[str] = Field(
        default=None,
        description="the program that printed it, with its version, AS "
                    "PRINTED; None when the page names none -- a spreadsheet "
                    "or a hand calculation usually does not")
    method: str = Field(
        default="",
        description="the method or standard the page names: 'Schmertmann "
                    "strain influence', 'AASHTO 1993', 'Meyerhof', "
                    "'Bishop simplified'")
    subject: str = Field(
        default="",
        description="what it is FOR, as printed: the structure, the boring, "
                    "the section, the load case")
    inputs: List[NamedQuantity] = Field(
        default_factory=list,
        description="the labelled values the calculation was GIVEN")
    results: List[NamedQuantity] = Field(
        default_factory=list,
        description="the labelled values it WORKED OUT")
    summary: str = Field(
        default="",
        description="what this calculation does and what it concluded, in "
                    "the reader's own words, 60 words or fewer")
    pages: List[int] = Field(
        default_factory=list,
        description="0-based PDF pages this calculation was read from")
    source_report: str = Field(default="")
    prov: List[Provenance] = Field(
        default_factory=list, description="where it was read")
    unsettled: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="what could not be read off these pages, and why, each "
                    "as {what, why, page}")

    #: What the RECONCILER matched this calculation to. ``subject`` above is
    #: the page's own words and never changes; this says which exploration of
    #: the record it turned out to name, and stays empty when nothing did.
    linked_investigation_id: str = Field(
        default="",
        description="the investigation this calculation names, or empty")

    def result(self, *names: str) -> Optional[NamedQuantity]:
        """The first result whose printed label contains one of ``names``."""
        for wanted in names:
            key = wanted.strip().lower()
            for row in self.results:
                if key and key in row.name.lower():
                    return row
        return None


# ---------------------------------------------------------------------------
# quality assurance
# ---------------------------------------------------------------------------

#: What a QA entry says happened. ``conflict`` is the important one: two
#: sources of the same value that disagree are RECORDED, never silently
#: resolved. ``disagreement`` is its twin for the two voters inside one
#: reader -- the grid or the tables against the model -- and is the entry
#: the owner asked for: a place where methods say different things, so a
#: reviewer is sent to look.
QAKind = Literal[
    "skipped", "partial", "conflict", "disagreement", "label_disagreement",
    "unreadable", "unconverted", "count_mismatch", "out_of_range", "note"]


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
# reports bound inside reports
# ---------------------------------------------------------------------------

class ParentReport(BaseModel):
    """The report this record was bound inside, and where in it.

    Present only on the record of a BOUND document: an earlier firm's
    investigation reproduced whole as an appendix, a bridging report bound
    into the design-build report that answers it. ``pages`` are 0-based
    indexes into the PARENT file, which is the same PDF this record's own
    provenance points into -- there is one file, and a reviewer sent to
    check a value opens it at the page the record names.
    """

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(
        default="",
        description="the parent record's report_id; empty when it has none")
    bound_id: str = Field(
        default="",
        description="this document's handle inside the parent: bound1, "
                    "bound2, in page order")
    pages: str = Field(
        default="",
        description="the pages of the PARENT file this report occupied, "
                    "0-based, e.g. '112-184'")
    first_page: int = Field(default=0, ge=0)
    last_page: int = Field(default=0, ge=0)
    n_pages: int = Field(default=0, ge=0)
    record_path: str = Field(
        default="",
        description="where the parent's record.json is, relative to this "
                    "record's own folder")


class BoundReport(BaseModel):
    """One report bound inside this one, read as its own record.

    Nothing of it is in THIS record: its borings are its borings and its
    laboratory tests are its laboratory tests. What is here is the pointer --
    what it is, which pages of this file it occupied, what its own record
    turned out to hold, and where that record was written.
    """

    model_config = ConfigDict(extra="forbid")

    bound_id: str = Field(
        description="its handle inside this report: bound1, bound2, in page "
                    "order")
    report_id: str = Field(
        default="",
        description="the report_id its own record carries")
    title: str = Field(
        default="", description="the title it prints for itself, as printed")
    firm: str = Field(
        default="", description="the firm that wrote it, as printed")
    date: str = Field(
        default="", description="the date it prints for itself, as printed")
    kind: str = Field(
        default="",
        description="how it is bound in: volume, appended_prior_report, "
                    "data_report or other")
    document_type: str = Field(
        default="", description="what kind of document it is, in the owner's "
                                "documentType vocabulary")
    pages: str = Field(
        default="",
        description="the pages of THIS file it occupied, 0-based, e.g. "
                    "'112-184'")
    first_page: int = Field(default=0, ge=0)
    last_page: int = Field(default=0, ge=0)
    n_pages: int = Field(default=0, ge=0)
    said_by: List[str] = Field(
        default_factory=list,
        description="which sources called this range a bound document: "
                    "triage, planlens, or both")
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="what its own record holds: investigations, layers, "
                    "samples, spt, water_levels, lab_tests, calculations, "
                    "qa")
    folder: str = Field(
        default="",
        description="its output folder, relative to this record's own")
    record_path: str = Field(
        default="",
        description="its record.json, relative to this record's own folder")
    read: bool = Field(
        default=True,
        description="False when its pages were listed and not read as their "
                    "own document -- ingest_bound was off, or the reading "
                    "failed; the reason is then a QA entry")


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

class ReportRecord(BaseModel):
    """One geotechnical report as data.

    Written as ``report.record.json``. The summary page, the library page and
    the DIGGS file are exports of this and add nothing that is not here.

    A report with another report bound inside it is TWO records, not one
    record holding two investigations: ``bound_documents`` lists the children
    and ``parent`` is set on each child. See :mod:`report_ingest.bound` for
    why, and for how the boundary between them is decided.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    document: DocumentFacts = Field(default_factory=DocumentFacts)
    page_labels: List[PageLabel] = Field(
        default_factory=list,
        description="what each page IS, with its confidence and every voter "
                    "that had a view; empty on a record written before the "
                    "page labels became a vote")
    project: Project = Field(default_factory=Project)
    general: GeneralFacts = Field(default_factory=GeneralFacts)
    natural_hazards: NaturalHazardFacts = Field(
        default_factory=NaturalHazardFacts)
    narrative: NarrativeFacts = Field(default_factory=NarrativeFacts)
    investigations: List[Investigation] = Field(default_factory=list)
    lab_tests: List[LabTest] = Field(default_factory=list)
    calculations: List[Calculation] = Field(
        default_factory=list,
        description="the calculation printouts, read off their own pages; "
                    "empty on a record written before the calculation "
                    "reader and on one whose report carries none")
    calcs: List[CalcEntry] = Field(
        default_factory=list,
        description="the WP5 stub, kept so an older file still loads; "
                    "nothing writes it")
    qa: List[QAEntry] = Field(default_factory=list)
    bound_documents: List[BoundReport] = Field(
        default_factory=list,
        description="reports bound inside this one, each read into its own "
                    "record; nothing of theirs is anywhere else in this "
                    "record")
    parent: Optional[ParentReport] = Field(
        default=None,
        description="set when THIS record is a report bound inside another "
                    "one: which report, and the pages of it this one "
                    "occupied")

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
            "calculations": len(self.calculations),
            "calcs": len(self.calcs),
            "qa": len(self.qa),
        }


def si_numbers(value: Any, path: str = "",
               skip: Tuple[str, ...] = ()) -> List[Tuple[str, float, str]]:
    """``(field, value in SI, the SI unit)`` for every number a model holds.

    Walks any part of the record -- a whole test, one result, one row -- and
    returns the numbers in it and nothing else. A :class:`Quantity` converts;
    a plain number is dimensionless and is taken as it stands; a string, a
    boolean and a None are not numbers and do not appear. A quantity whose
    printed unit the table cannot convert does not appear either, because
    there is no SI value for it to have.

    It exists so that a check on a record can ask "did these numbers survive"
    without a second copy of whatever mapping a writer or a scorer uses. That
    independence is the point: a walk over the model cannot agree with a
    writer by sharing its mistakes.
    """
    out: List[Tuple[str, float, str]] = []
    if value is None or isinstance(value, (str, bool)):
        return out
    if isinstance(value, Quantity):
        converted = value.to_si()
        if converted is not None:
            out.append((path, converted.value, converted.unit))
        return out
    if isinstance(value, (int, float)):
        out.append((path, float(value), ""))
        return out
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.extend(si_numbers(item, f"{path}[{i}]"))
        return out
    fields = getattr(type(value), "model_fields", None)
    if fields:
        for name in fields:
            if name in skip:
                continue
            here = f"{path}.{name}" if path else name
            out.extend(si_numbers(getattr(value, name, None), here))
    return out


def record_json_schema() -> Dict[str, Any]:
    """The record's JSON schema, for the library agent and for a consumer.

    Exported rather than hand-written so it cannot drift from the models.
    """
    schema = ReportRecord.model_json_schema()
    schema["title"] = "ReportRecord"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema
