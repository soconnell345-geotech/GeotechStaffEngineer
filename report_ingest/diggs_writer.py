"""The record's investigations as DIGGS 2.6, and the two gates on the file.

DIGGS is how this record leaves the building. Every plot and every tool in
``subsurface_characterization`` already takes a DIGGS file, so writing one
means the report's logs arrive in the app's own subsurface stack with nothing
new to learn, and in an interchange format the industry reads.

TWO GATES, AND WHY NEITHER ALONE IS ENOUGH.

:func:`diggs_schema_gate` runs the file against the DIGGS 2.6 XSD that pydiggs
bundles. It answers "is this DIGGS": the right elements in the right
namespaces in the right order, every measure carrying a ``uom``. It cannot
tell whether the numbers in it are the log's numbers.

:func:`diggs_roundtrip_gate` reads the file back with the app's own
``parse_diggs`` and compares every depth, blow count, water level and index
value against the investigations that went in. It answers "does it still say
what the log said". A file can be XSD-valid and wrong -- a depth written into
the wrong element validates perfectly and comes back as nothing -- and that is
exactly the failure this catches.

WHAT THE 2.6 SCHEMA ACTUALLY WANTS, because it is not what a reader of the
app's older test fixtures would guess. In real DIGGS:

* the test PROCEDURES (``DrivenPenetrationTest``, ``AtterbergLimitsTest``,
  ``WaterContentTest``, ``LabDensityTest``, ``ParticleSizeTest``,
  ``UnconfinedCompressiveStrengthTest``, ``PocketPenetrometerTest``) live in
  the ``.../2.6/geotechnical`` namespace, not the DIGGS namespace, and they
  describe HOW a test was done. They carry no result;
* the VALUE lives in ``Test/outcome/TestResult/results/ResultSet``, named by a
  ``propertyClass`` from the DIGGS property dictionary (``n_value``,
  ``water_content_natural``, ``dry_density``, ``liquid_limit``,
  ``percent_fines``, ``compressive_strength_unconfined``, ``water_depth``);
* a DEPTH is not an element. It is a position along the borehole's own linear
  reference system, written as a ``LinearExtent`` whose ``gml:posList`` holds
  the top and the base;
* lithology is an ``observation/LithologySystem`` at the document root that
  points back at its borehole, not a child of the borehole;
* there is no ``WaterLevelObservation`` and no ``MoistureContent`` element at
  all. Water goes in the borehole's own ``waterStrike``.

SI, ONCE, HERE. The record keeps every number in the unit it was printed in.
This is the one place that converts, because DIGGS wants a ``uom`` on every
measure and a consumer wants one system: metres, kPa, kN/m3. A value whose
printed unit is not in the record's conversion table is NOT written with a
guessed unit -- it is left out and named in the writer's own notes, which the
round-trip gate then does not look for.

THE CALCULATIONS ARE NOT WRITTEN, AND THAT IS DELIBERATE. DIGGS 2.6 has no
concept of a calculation: it is an interchange format for what was OBSERVED
in the ground -- holes, samples, tests and their results -- and there is no
element for what an engineer worked out from them, no way to say which method
was used and no home for a chosen footing thickness. So
``ReportRecord.calculations`` is IGNORED here. It reaches a reader through
the record itself, through the summary page's "Calculations" section and
through the library page, all three of which carry the page each value was
printed on. A writer that squeezed a design calculation into an observation
element would produce a file that validates and lies, which is the exact
failure the round-trip gate exists to catch.

THE LABORATORY TESTS (WP3) follow the same shape and add four facts about the
2.6 schema that are worth knowing before reading the code:

* every lab procedure lives in the geotechnical namespace -- except
  ``TriaxialTest``, which the schema declares in ``TestProceduresAll.xsd``
  and therefore in the DIGGS namespace itself. One element out of thirteen,
  and a file that puts it in the other namespace is silently empty;
* a ``Test`` has exactly ONE ``outcome``. A gradation with both derived
  fractions and a grading curve is therefore two Tests against the same hole
  at the same depth, not one Test with two result sets;
* a curve is a result set of MANY ROWS, whatever kind of curve it is. 2.6 has
  native homes for some of them, and every one of those homes demands a value
  these sheets do not print: a ``Grading`` requires a particle size and half
  the forms label their sieves by number alone, a consolidation increment
  requires a final axial deformation, a direct shear increment an elapsed
  time, a triaxial shear stage a cell pressure. Filling a required sibling
  with a made-up number to reach a nicer element is the one thing this writer
  will not do, so every curve goes where no invention is needed and the
  procedure carries the settings instead;
* a result is POSITIONED. ``TestResult/location`` is required and is a
  position along a hole's own linear reference system, so a lab test that
  names no boring, or names one and no depth, CANNOT be written: there is
  nowhere in the file for it to be. Those are named in the writer's notes and
  stay in the record, which does not require a value to have a place.

A lab test that names a boring this record has no log for gets a minimal
``Borehole`` of its own, so the test has something to hang off. That is
recorded in :attr:`DiggsWriteNotes.synthesised`: the hole is an identifier
the lab sheet printed and nothing more, and a reader of the file should know
which holes those are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape, quoteattr

from report_ingest.model import (
    AtterbergResult, CBRResult, CPTData, ChemicalResult, CompactionResult,
    ConsolidationResult, DCPData, GradationResult, Investigation, LabTest,
    Layer, MoistureDensityResult, OtherResult, Project, Quantity,
    ReportRecord, Sample, SPT, StrengthResult, SummaryRow,
    SummaryTableResult, WaterLevel, si_numbers,
)

__all__ = [
    "write_diggs", "diggs_schema_gate", "diggs_roundtrip_gate",
    "DiggsWriteNotes", "NS", "PROPERTY_CLASS", "TOLERANCE",
    "LAB_PROCEDURE", "DICTIONARY_CODESPACE", "APP_CODESPACE",
    "UOM_IN_FILE",
]

#: The namespaces a DIGGS 2.6 instance declares. ``diggs_geo`` is the one that
#: trips people up: every test PROCEDURE is in it.
NS: Dict[str, str] = {
    "diggs": "http://diggsml.org/schemas/2.6",
    "diggs_geo": "http://diggsml.org/schemas/2.6/geotechnical",
    "gml": "http://www.opengis.net/gml/3.2",
    "glr": "http://www.opengis.net/gml/3.3/lr",
    "xlink": "http://www.w3.org/1999/xlink",
}

#: ``record field -> (dictionary propertyClass, SI uom, the 2.6 procedure)``.
#: This table IS the DIGGS element map. Every propertyClass is one pydiggs
#: publishes in ``dictionaries/properties.xml``; every procedure is one
#: declared in the bundled 2.6 schema.
PROPERTY_CLASS: Dict[str, Tuple[str, str, str]] = {
    "n_value": ("n_value", "", "diggs_geo:DrivenPenetrationTest"),
    "blow_count": ("blow_count", "", "diggs_geo:DrivenPenetrationTest"),
    "water_content": ("water_content_natural", "%",
                      "diggs_geo:WaterContentTest"),
    "dry_unit_weight": ("dry_density", "kN/m3", "diggs_geo:LabDensityTest"),
    "liquid_limit": ("liquid_limit", "%", "diggs_geo:AtterbergLimitsTest"),
    "plastic_limit": ("plastic_limit", "%", "diggs_geo:AtterbergLimitsTest"),
    "plasticity_index": ("plasticity_index", "%",
                         "diggs_geo:AtterbergLimitsTest"),
    "fines_percent": ("percent_fines", "%", "diggs_geo:ParticleSizeTest"),
    "qu": ("compressive_strength_unconfined", "kPa",
           "diggs_geo:UnconfinedCompressiveStrengthTest"),
    "pocket_pen": ("compressive_strength_unconfined", "kPa",
                   "diggs_geo:PocketPenetrometerTest"),
}

#: The codespace of the DIGGS property dictionary pydiggs publishes
#: (``dictionaries/properties.xml``). A ``propertyClass`` in it is a term a
#: consumer can look up.
DICTIONARY_CODESPACE = "urn:diggs:def:codelist:DIGGS:properties"
#: The codespace for the handful of values a lab sheet prints that the 2.6
#: dictionary has no term for: d90 and d100, a void ratio, a swell
#: percentage, an ash content, a sulfide content, total salts, a specimen's
#: confining pressure. They are written rather than dropped -- the sheet
#: printed them -- and this codespace says out loud that they are OURS and
#: not dictionary terms, so nothing mistakes one for a published property.
APP_CODESPACE = "urn:x-diggs:def:codelist:report_ingest:properties"

#: ``LabTest.kind -> the 2.6 procedure element``. Twelve of the thirteen are
#: in the geotechnical namespace; ``TriaxialTest`` is declared in
#: ``TestProceduresAll.xsd`` and is therefore in the DIGGS namespace, which
#: is the single fact most likely to make a conformant file read as empty.
#: A kind with no procedure (a summary table, an ``other`` page) writes its
#: values with no procedure element at all, which the schema allows.
LAB_PROCEDURE: Dict[str, str] = {
    "atterberg": "diggs_geo:AtterbergLimitsTest",
    "gradation": "diggs_geo:ParticleSizeTest",
    "swell_consolidation": "diggs_geo:ConsolidationTest",
    "triaxial": "TriaxialTest",
    "direct_shear": "diggs_geo:DirectShearTest",
    "unconfined": "diggs_geo:UnconfinedCompressiveStrengthTest",
    "unconfined_rock": "diggs_geo:UnconfinedCompressiveStrengthTest",
    "compaction": "diggs_geo:LabCompactionTest",
    "cbr": "diggs_geo:LabCBRTest",
    "moisture_content": "diggs_geo:WaterContentTest",
    "density": "diggs_geo:LabDensityTest",
    "organic_content": "diggs_geo:LossOnIgnitionTest",
    "chemical": "diggs_geo:LabChemicalTest",
    "specific_gravity": "diggs_geo:SpecificGravityTest",
    "permeability": "diggs_geo:LabPermeabilityTest",
}

#: How close a value has to come back for the round trip to pass, per kind.
#: These are NOT measurement tolerances -- nothing is being measured. They
#: absorb the unit conversion and the decimal places the file is written to,
#: and nothing else, so they are tight on purpose.
TOLERANCE: Dict[str, float] = {
    "depth_m": 0.005,          # 5 mm: the file writes depths to 4 decimals
    "n_value": 0.0,            # a blow count is an integer both ways
    "percent": 0.05,
    "stress_kPa": 0.05,
    "unit_weight_kNm3": 0.005,
}

#: The record's SI unit -> the code the DIGGS schema accepts for it. Only
#: one entry, and it is the one that fails loudly rather than quietly: the
#: schema's plane-angle unit is ``dega``, ``deg`` is not in its list at all,
#: and a friction angle written in degrees fails validation on the whole
#: file. Everything else the record converts to -- m, mm, kPa, kN/m3, %, mV,
#: mg/kg, ohm.m, degC, m3, kg, m/s -- is a code the schema already knows.
UOM_IN_FILE: Dict[str, str] = {"deg": "dega"}

#: Depths and measures are written to this many decimals. Four is about
#: 0.1 mm on a depth and is well inside every tolerance above.
_DP = 4


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ncname(text: str, fallback: str = "x") -> str:
    """A string turned into something that can be a ``gml:id``.

    An id must be an XML NCName: it cannot start with a digit and cannot hold
    a space, a slash or most punctuation. Boring names routinely do all three
    ("B-1 (offset)", "SB/01"), so they are folded here rather than in the
    record, which keeps the log's own spelling.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(text or "")).strip("_")
    if not safe:
        safe = fallback
    if not re.match(r"[A-Za-z_]", safe[0]):
        safe = f"{fallback}_{safe}"
    return safe


def _num(value: float) -> str:
    """A measure, written the way the file writes every measure.

    Four decimals is a tenth of a millimetre on a depth and is plenty for
    every value a log or a sheet prints -- until the value is SMALL. A sieve
    opening of 0.075 mm is 7.5e-05 m, which four decimals round to 0.0001,
    and a coefficient of consolidation in square metres per second rounds to
    nothing at all. Below a thousandth the number is written to six
    significant figures instead, which is exact for anything printed.
    """
    number = float(value)
    if number != 0.0 and abs(number) < 0.001:
        return f"{number:.6g}"
    text = f"{number:.{_DP}f}".rstrip("0").rstrip(".")
    return text or "0"


def _coord(value: float) -> str:
    """A coordinate, which needs more decimals than a measure does.

    A depth to four decimals is a tenth of a millimetre. A LATITUDE to four
    decimals is eleven metres, which would move a boring to the next street,
    so a position is written to nine -- about a tenth of a millimetre again,
    and still an exact decimal for every value a log prints.
    """
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return text or "0"


def _si(q: Optional[Quantity]) -> Optional[Tuple[float, str]]:
    """``(value, SI unit)`` for a quantity, or None when it cannot convert."""
    if q is None:
        return None
    converted = q.to_si()
    if converted is None:
        return None
    return converted.value, converted.unit


@dataclass
class _Column:
    """One column of a result set: what it is, and in what unit.

    ``dictionary`` says whether ``klass`` is a term of the DIGGS property
    dictionary. The handful that are not are still written -- the sheet
    printed them -- under a codespace that says they are ours.
    """

    name: str
    klass: str
    uom: str = ""
    type_data: str = "double"
    dictionary: bool = True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell(value: Any) -> str:
    """One cell of a data table.

    A number is written the way every measure in this file is written. Text
    is written as the sheet printed it, with the two separator characters
    turned into spaces so a cell cannot split the row it sits in.
    """
    if _is_number(value):
        return _num(value)
    text = " ".join(str(value).replace(",", " ").replace(";", " ").split())
    return text or "-"


@dataclass
class DiggsWriteNotes:
    """What the writer could not write, and what it had to leave behind.

    Handed back on the writer's own object rather than raised: a log with one
    unconvertible unit still has fifty good values in it, and the file should
    carry them. What is missing is SAID, so the round-trip gate is not asked
    to find something that was never written.
    """

    skipped: List[str] = field(default_factory=list)
    #: Holes that exist in this file only because a lab sheet named them.
    #: They carry an identifier and nothing else -- no depth, no layers, no
    #: position -- and a reader of the file should know which they are.
    synthesised: List[str] = field(default_factory=list)
    investigations: int = 0
    layers: int = 0
    samples: int = 0
    spt: int = 0
    water: int = 0
    tests: int = 0
    lab_tests: int = 0
    #: Soundings written as one positioned result set each: a cone sounding
    #: as a StaticConePenetrationTest, a dynamic probe as a DynamicProbeTest.
    cpt: int = 0
    dcp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"skipped": list(self.skipped),
                "synthesised": list(self.synthesised),
                "investigations": self.investigations, "layers": self.layers,
                "samples": self.samples, "spt": self.spt, "water": self.water,
                "tests": self.tests, "lab_tests": self.lab_tests,
                "cpt": self.cpt, "dcp": self.dcp}


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

class _Writer:
    """Builds one DIGGS 2.6 document, element by element."""

    def __init__(self, investigations: Sequence[Investigation],
                 project: Optional[Project], document_id: str,
                 lab_tests: Sequence[LabTest] = ()) -> None:
        self.investigations = list(investigations)
        self.lab = list(lab_tests)
        self.project = project or Project()
        self.doc_id = _ncname(document_id or "report", "report")
        self.project_id = _ncname(
            self.project.number or self.project.name or "project", "project")
        self.notes = DiggsWriteNotes()
        self.lines: List[str] = []
        self._seen_ids: Dict[str, int] = {}
        self._synthesise_holes()

    def _synthesise_holes(self) -> None:
        """A hole for every boring a lab sheet names and no log describes.

        Every result in DIGGS is a position along a sampling feature's own
        linear reference system, so a lab test whose boring is not in this
        record has nowhere to be written. Rather than drop the test, the hole
        is created with its identifier and nothing else, and said out loud in
        the notes -- an identifier a lab sheet printed is a real fact, and a
        depth, a position and a layer sequence it does not have are not
        invented to go with it.
        """
        known = {inv.investigation_id.strip()
                 for inv in self.investigations if inv.investigation_id}
        wanted: Dict[str, str] = {}
        for name, depth in self._lab_positions():
            name = name.strip()
            if not name or name in known or name in wanted:
                continue
            wanted[name] = depth.unit if depth is not None else ""
        for name, unit in wanted.items():
            self.investigations.append(Investigation(
                investigation_id=name, depth_unit=unit,
                units_known=bool(unit)))
            self.notes.synthesised.append(
                f"{name}: named by a laboratory sheet and described by no "
                f"log in this record; written as a Borehole carrying its "
                f"identifier only")

    def _lab_positions(self) -> List[Tuple[str, Optional[Quantity]]]:
        """``(boring, depth)`` for every lab result this file will position."""
        out: List[Tuple[str, Optional[Quantity]]] = []
        for test in self.lab:
            if isinstance(test.result, SummaryTableResult):
                out.extend((row.investigation_id, row.depth_top)
                           for row in test.result.rows)
            else:
                out.append((test.investigation_id, test.depth_top))
        return out

    # -- id bookkeeping ----------------------------------------------------
    def uid(self, stem: str) -> str:
        """A gml:id unique in this document.

        Two borings called B-1 in one report (it happens: two volumes bound
        together) would otherwise collide, and a collision makes every xlink
        into one of them ambiguous.
        """
        stem = _ncname(stem)
        n = self._seen_ids.get(stem, 0)
        self._seen_ids[stem] = n + 1
        return stem if n == 0 else f"{stem}_{n + 1}"

    # -- emitting ----------------------------------------------------------
    def out(self, text: str, indent: int = 0) -> None:
        self.lines.append("  " * indent + text)

    def measure(self, tag: str, value: float, uom: str, indent: int) -> None:
        """One eml measure: a number that always carries its unit."""
        self.out(f"<{tag} uom={quoteattr(uom)}>{_num(value)}</{tag}>", indent)

    def text_element(self, tag: str, value: str, indent: int) -> None:
        if str(value or "").strip():
            self.out(f"<{tag}>{escape(str(value).strip())}</{tag}>", indent)

    # -- the document ------------------------------------------------------
    def build(self) -> str:
        self.out('<?xml version="1.0" encoding="UTF-8"?>')
        attrs = " ".join(
            f'xmlns:{prefix}={quoteattr(uri)}'
            for prefix, uri in NS.items() if prefix != "diggs")
        self.out(f'<Diggs xmlns={quoteattr(NS["diggs"])} {attrs} '
                 f'gml:id={quoteattr(self.doc_id)}>')
        self.document_information()
        self.project_element()
        plans: List[Tuple[Investigation, str]] = []
        for inv in self.investigations:
            stem = _ncname(inv.investigation_id or "exploration",
                           "exploration")
            plans.append((inv, self.uid(f"bh_{stem}")))
        for inv, gml_id in plans:
            self.sampling_feature(inv, gml_id)
        for inv, gml_id in plans:
            self.sampling_activities(inv, gml_id)
        for inv, gml_id in plans:
            self.samples(inv, gml_id)
        for inv, gml_id in plans:
            self.lithology(inv, gml_id)
        for inv, gml_id in plans:
            self.measurements(inv, gml_id)
        self.lab_measurements({inv.investigation_id.strip(): gml_id
                               for inv, gml_id in plans})
        self.out("</Diggs>")
        return "\n".join(self.lines) + "\n"

    def document_information(self) -> None:
        self.out("<documentInformation>", 1)
        self.out(f'<DocumentInformation gml:id='
                 f'{quoteattr(self.uid("docinfo"))}>', 2)
        self.out(f"<creationDate>{date.today().isoformat()}</creationDate>", 3)
        self.out("</DocumentInformation>", 2)
        self.out("</documentInformation>", 1)

    def project_element(self) -> None:
        p = self.project
        self.out("<project>", 1)
        self.out(f"<Project gml:id={quoteattr(self.project_id)}>", 2)
        if str(p.number or "").strip():
            self.out(f"<gml:identifier codeSpace=\"urn:x-diggs:def:project\">"
                     f"{escape(p.number.strip())}</gml:identifier>", 3)
        self.out(f"<gml:name>{escape(p.name.strip() or 'Unnamed project')}"
                 f"</gml:name>", 3)
        self.out("</Project>", 2)
        self.out("</project>", 1)

    # -- one exploration ---------------------------------------------------
    def sampling_feature(self, inv: Investigation, gml_id: str) -> None:
        # A pit is a TrialPit and a hole is a Borehole; everything else the
        # record can hold is written as a Borehole, which is what DIGGS 2.6
        # gives a linear exploration that is not a pit.
        element = "TrialPit" if inv.kind == "test_pit" else "Borehole"
        depth = _si(inv.total_depth)
        elev = _si(inv.elevation)
        z = elev[0] if elev else 0.0
        x = inv.x if inv.x is not None else 0.0
        y = inv.y if inv.y is not None else 0.0
        bottom = z - (depth[0] if depth else 0.0)

        self.out("<samplingFeature>", 1)
        self.out(f"<{element} gml:id={quoteattr(gml_id)}>", 2)
        self.out(f"<gml:name>"
                 f"{escape(inv.investigation_id or gml_id)}</gml:name>", 3)
        # The schema's order, which is not the order a person would choose:
        # remark comes BEFORE investigationTarget and projectRef.
        if inv.remarks.strip():
            self.out("<remark>", 3)
            self.out("<Remark>", 4)
            self.text_element("content", inv.remarks, 5)
            self.out("</Remark>", 4)
            self.out("</remark>", 3)
        self.out("<investigationTarget>Natural Ground</investigationTarget>", 3)
        self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
        for key, value in sorted(inv.fields.items()):
            if not str(value or "").strip():
                continue
            self.out("<otherSamplingFeatureProperty>", 3)
            self.out(f'<Parameter gml:id='
                     f'{quoteattr(self.uid(gml_id + "_p_" + key))}>', 4)
            self.text_element("parameterName", key, 5)
            self.text_element("parameterValue", value, 5)
            self.out("</Parameter>", 4)
            self.out("</otherSamplingFeatureProperty>", 3)
        if inv.station.strip() or inv.offset.strip():
            self.out("<locality>", 3)
            self.out(f'<Locality gml:id='
                     f'{quoteattr(self.uid(gml_id + "_loc"))}>', 4)
            self.text_element("station", inv.station, 5)
            self.out("</Locality>", 4)
            self.out("</locality>", 3)
        self.out("<referencePoint>", 3)
        self.out(f'<PointLocation gml:id={quoteattr(gml_id + "_rp")}>', 4)
        srs = (f' srsName={quoteattr(inv.coordinate_system)}'
               if inv.coordinate_system.strip() else "")
        self.out(f'<gml:pos srsDimension="3"{srs}>'
                 f'{_coord(x)} {_coord(y)} {_coord(z)}</gml:pos>', 5)
        self.out("</PointLocation>", 4)
        self.out("</referencePoint>", 3)
        self.out("<centerLine>", 3)
        self.out(f'<LinearExtent gml:id={quoteattr(gml_id + "_cl")}>', 4)
        self.out(f'<gml:posList srsDimension="3">'
                 f'{_coord(x)} {_coord(y)} {_coord(z)} '
                 f'{_coord(x)} {_coord(y)} {_coord(bottom)}</gml:posList>', 5)
        self.out("</LinearExtent>", 4)
        self.out("</centerLine>", 3)
        self.linear_referencing(gml_id)
        if inv.date_started.strip() or inv.date_finished.strip():
            self.when_constructed(inv, gml_id)
        if depth is not None:
            self.measure("totalMeasuredDepth", depth[0], depth[1], 3)
        elif inv.total_depth is not None:
            self.notes.skipped.append(
                f"{inv.investigation_id}: total depth "
                f"{inv.total_depth.value:g} {inv.total_depth.unit!r} -- unit "
                f"not in the conversion table")
        if element == "Borehole":
            self.drilling(inv, gml_id)
            self.water_strikes(inv, gml_id)
        self.out(f"</{element}>", 2)
        self.out("</samplingFeature>", 1)
        self.notes.investigations += 1

    def linear_referencing(self, gml_id: str) -> None:
        """The hole's own 1-D system: what every depth below is measured in.

        A depth in DIGGS is a position along this, not a number in an
        element, which is why it has to be written even though nothing in the
        record corresponds to it.
        """
        lsr = f"{gml_id}_lsr"
        self.out("<linearReferencing>", 3)
        self.out(f'<LinearSpatialReferenceSystem gml:id={quoteattr(lsr)}>', 4)
        self.out(f'<gml:identifier codeSpace="urn:x-def:authority:DIGGSINC">'
                 f'urn:x-diggs:def:fi:DIGGSINC:{lsr}</gml:identifier>', 5)
        self.out(f'<glr:linearElement xlink:href="#{gml_id}_cl"/>', 5)
        self.out("<glr:lrm>", 5)
        self.out(f'<glr:LinearReferencingMethod gml:id='
                 f'{quoteattr(gml_id + "_lrm")}>', 6)
        self.out("<glr:name>chainage</glr:name>", 7)
        self.out("<glr:type>absolute</glr:type>", 7)
        self.out("<glr:units>m</glr:units>", 7)
        self.out("</glr:LinearReferencingMethod>", 6)
        self.out("</glr:lrm>", 5)
        self.out("</LinearSpatialReferenceSystem>", 4)
        self.out("</linearReferencing>", 3)

    def when_constructed(self, inv: Investigation, gml_id: str) -> None:
        started = _iso_date(inv.date_started)
        finished = _iso_date(inv.date_finished) or started
        if not started:
            if inv.date_started.strip() or inv.date_finished.strip():
                self.notes.skipped.append(
                    f"{inv.investigation_id}: dates "
                    f"{inv.date_started!r}/{inv.date_finished!r} -- not a date "
                    f"DIGGS can carry; kept in the header fields instead")
            return
        self.out("<whenConstructed>", 3)
        self.out(f'<TimeInterval gml:id={quoteattr(gml_id + "_when")}>', 4)
        self.out(f"<start>{started}</start>", 5)
        self.out(f"<end>{finished or started}</end>", 5)
        self.out("</TimeInterval>", 4)
        self.out("</whenConstructed>", 3)

    def drilling(self, inv: Investigation, gml_id: str) -> None:
        d = inv.drilling
        if not (d.method.strip() or d.equipment.strip()):
            return
        self.out("<constructionMethod>", 3)
        self.out(f'<BoreholeConstructionMethod gml:id='
                 f'{quoteattr(gml_id + "_cm")}>', 4)
        self.out(f"<gml:name>{escape(d.method.strip() or 'Drilling')}"
                 f"</gml:name>", 5)
        if d.method.strip():
            self.out("<constructionMethod>", 5)
            self.out(f'<Specification gml:id={quoteattr(gml_id + "_cms")}>', 6)
            self.out(f"<gml:name>{escape(d.method.strip())}</gml:name>", 7)
            self.out("</Specification>", 6)
            self.out("</constructionMethod>", 5)
        if d.equipment.strip():
            self.out("<constructionEquipment>", 5)
            self.out(f'<Equipment gml:id={quoteattr(gml_id + "_cme")}>', 6)
            self.out(f"<gml:name>{escape(d.equipment.strip())}</gml:name>", 7)
            self.out("<class>Drill Rig</class>", 7)
            self.out("</Equipment>", 6)
            self.out("</constructionEquipment>", 5)
        self.out("</BoreholeConstructionMethod>", 4)
        self.out("</constructionMethod>", 3)

    def water_strikes(self, inv: Investigation, gml_id: str) -> None:
        """Water, in the only place DIGGS 2.6 has for it.

        There is no ``WaterLevelObservation`` element in 2.6. A reading is a
        ``WaterStrikeReading`` whose ``waterLocation`` is a position along the
        hole; a log that says none was encountered is ``notEncountered``.
        """
        readings = [w for w in inv.water if _si(w.depth) is not None]
        none_found = [w for w in inv.water if w.when == "not_encountered"]
        if not readings and not none_found:
            return
        self.out("<waterStrike>", 3)
        self.out(f'<WaterStrike gml:id={quoteattr(self.uid(gml_id + "_ws"))}>',
                 4)
        if not readings:
            self.out("<notEncountered>true</notEncountered>", 5)
        else:
            for n, water in enumerate(readings):
                tag = ("initialWaterStrikeReading" if n == 0
                       else "postStrikeReading")
                self.out(f"<{tag}>", 5)
                self.water_reading(water, f"{gml_id}_wr{n}", 6)
                self.out(f"</{tag}>", 5)
        self.out("</WaterStrike>", 4)
        self.out("</waterStrike>", 3)
        self.notes.water += len(readings) or 1

    def water_reading(self, water: WaterLevel, wid: str, indent: int) -> None:
        depth = _si(water.depth)
        self.out(f'<WaterStrikeReading gml:id={quoteattr(self.uid(wid))}>',
                 indent)
        if water.hours is not None:
            self.measure("elapsedTime", water.hours, "h", indent + 1)
        self.out("<waterLocation>", indent + 1)
        self.out(f'<PointLocation gml:id={quoteattr(self.uid(wid + "_pl"))} '
                 f'srsName="#{wid.rsplit("_wr", 1)[0]}_lsr" '
                 f'srsDimension="1">', indent + 2)
        self.out(f"<gml:pos>{_num(depth[0])}</gml:pos>", indent + 3)
        self.out("</PointLocation>", indent + 2)
        self.out("</waterLocation>", indent + 1)
        casing = _si(water.casing_depth)
        if casing is not None:
            self.out("<bottomCasing>", indent + 1)
            self.out(f'<PointLocation gml:id='
                     f'{quoteattr(self.uid(wid + "_bc"))} '
                     f'srsName="#{wid.rsplit("_wr", 1)[0]}_lsr" '
                     f'srsDimension="1">', indent + 2)
            self.out(f"<gml:pos>{_num(casing[0])}</gml:pos>", indent + 3)
            self.out("</PointLocation>", indent + 2)
            self.out("</bottomCasing>", indent + 1)
        self.out("</WaterStrikeReading>", indent)

    # -- samples -----------------------------------------------------------
    def _interval(self, top: float, bottom: Optional[float], lsr: str,
                  eid: str, indent: int) -> None:
        """A depth interval as a position along the hole's own system."""
        values = f"{_num(top)} {_num(bottom)}" if bottom is not None \
            else _num(top)
        self.out(f'<LinearExtent gml:id={quoteattr(self.uid(eid))} '
                 f'srsName="#{lsr}" srsDimension="1">', indent)
        self.out(f"<gml:posList>{values}</gml:posList>", indent + 1)
        self.out("</LinearExtent>", indent)

    def sampling_activities(self, inv: Investigation, gml_id: str) -> None:
        for n, sample in enumerate(inv.samples):
            top = _si(sample.top)
            if top is None:
                self.notes.skipped.append(
                    f"{inv.investigation_id}: sample "
                    f"{sample.sample_id or n} depth {sample.top.value:g} "
                    f"{sample.top.unit!r} -- unit not in the conversion table")
                continue
            bottom = _si(sample.bottom)
            sa_id = self.uid(f"sa_{_ncname(gml_id)}_{n}")
            self.out("<samplingActivity>", 1)
            self.out(f"<SamplingActivity gml:id={quoteattr(sa_id)}>", 2)
            self.out(f"<gml:name>{escape(sample.sample_id or str(n + 1))}"
                     f"</gml:name>", 3)
            self.out("<investigationTarget>Natural Ground"
                     "</investigationTarget>", 3)
            self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
            self.out(f'<samplingFeatureRef xlink:href="#{gml_id}"/>', 3)
            self.out("<samplingLocation>", 3)
            self._interval(top[0], bottom[0] if bottom else None,
                           f"{gml_id}_lsr", f"{sa_id}_le", 4)
            self.out("</samplingLocation>", 3)
            self.out("<activityType>collect</activityType>", 3)
            if sample.rqd_percent is not None:
                # The schema's own order: RQD before sampleProduced. It is a
                # length-per-length ratio, so a percentage is what it takes.
                self.measure("samplingActivityRQD", sample.rqd_percent, "%", 3)
            # The SampleProduced object: what this activity produced, and
            # where it came from. The Sample element points at THIS, which is
            # why it has to exist and why it carries the location again.
            self.out("<sampleProduced>", 3)
            self.out(f'<SampleProduced gml:id={quoteattr(sa_id + "_sp")}>', 4)
            self.out("<location>", 5)
            self._interval(top[0], bottom[0] if bottom else None,
                           f"{gml_id}_lsr", f"{sa_id}_sple", 6)
            self.out("</location>", 5)
            self.out("</SampleProduced>", 4)
            self.out("</sampleProduced>", 3)
            if sample.kind and sample.kind != "other":
                self.out("<samplingMethod>", 3)
                self.out(f'<Specification gml:id={quoteattr(sa_id + "_sm")}>',
                         4)
                self.out(f"<gml:name>{escape(sample.kind)}</gml:name>", 5)
                self.out("</Specification>", 4)
                self.out("</samplingMethod>", 3)
            recovery = _si(sample.recovery)
            if recovery is not None:
                self.measure("totalSampleRecoveryLength", recovery[0],
                             recovery[1], 3)
            if sample.recovery_percent is not None:
                # DIGGS carries recovery as a LENGTH and has no element for
                # the percentage a log prints. Dropping it would lose the
                # number most logs actually print, so it goes in the
                # activity's own other-property slot, named.
                self.out("<otherSamplingActivityProperty>", 3)
                self.out(f'<Parameter gml:id={quoteattr(sa_id + "_rec")}>', 4)
                self.out("<parameterName>recovery_percent</parameterName>", 5)
                self.out(f"<parameterValue>{_num(sample.recovery_percent)}"
                         f"</parameterValue>", 5)
                self.out("</Parameter>", 4)
                self.out("</otherSamplingActivityProperty>", 3)
            self.out("</SamplingActivity>", 2)
            self.out("</samplingActivity>", 1)
            self.notes.samples += 1

    def samples(self, inv: Investigation, gml_id: str) -> None:
        for n, sample in enumerate(inv.samples):
            if _si(sample.top) is None:
                continue
            s_id = self.uid(f"smp_{_ncname(gml_id)}_{n}")
            self.out("<sample>", 1)
            self.out(f"<Sample gml:id={quoteattr(s_id)}>", 2)
            self.out(f"<gml:name>{escape(sample.sample_id or str(n + 1))}"
                     f"</gml:name>", 3)
            self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
            # Where this sample came from, in the order the schema wants it:
            # the activity, then the SampleProduced object inside it that
            # carries the depth. A DIGGS sample with neither came from
            # nowhere.
            self.out(f'<samplingActivityRef '
                     f'xlink:href="#sa_{_ncname(gml_id)}_{n}"/>', 3)
            self.out(f'<sampleProducedRef '
                     f'xlink:href="#sa_{_ncname(gml_id)}_{n}_sp"/>', 3)
            if sample.note.strip():
                self.text_element("purpose", sample.note, 3)
            self.out("</Sample>", 2)
            self.out("</sample>", 1)

    # -- lithology ---------------------------------------------------------
    def lithology(self, inv: Investigation, gml_id: str) -> None:
        layers = [ly for ly in inv.layers if _si(ly.top) is not None]
        for ly in inv.layers:
            if _si(ly.top) is None:
                self.notes.skipped.append(
                    f"{inv.investigation_id}: layer top {ly.top.value:g} "
                    f"{ly.top.unit!r} -- unit not in the conversion table")
        if not layers:
            return
        sys_id = self.uid(f"litho_{_ncname(gml_id)}")
        self.out("<observation>", 1)
        self.out(f"<LithologySystem gml:id={quoteattr(sys_id)}>", 2)
        self.out(f"<gml:name>{escape(inv.investigation_id or gml_id)}"
                 f"</gml:name>", 3)
        self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
        self.out(f'<samplingFeatureRef xlink:href="#{gml_id}"/>', 3)
        self.out("<lithologyClassificationType>SOIL"
                 "</lithologyClassificationType>", 3)
        for n, layer in enumerate(layers):
            self.lithology_observation(layer, gml_id, sys_id, n)
        self.out("</LithologySystem>", 2)
        self.out("</observation>", 1)

    def lithology_observation(self, layer: Layer, gml_id: str, sys_id: str,
                              n: int) -> None:
        top = _si(layer.top)
        bottom = _si(layer.bottom)
        obs_id = self.uid(f"{sys_id}_obs{n}")
        self.out(f"<lithologyObservation>", 3)
        self.out(f"<LithologyObservation gml:id={quoteattr(obs_id)}>", 4)
        self.out("<location>", 5)
        self._interval(top[0], bottom[0] if bottom else None,
                       f"{gml_id}_lsr", f"{obs_id}_le", 6)
        self.out("</location>", 5)
        self.out("<primaryLithology>", 5)
        self.out(f'<Lithology gml:id={quoteattr(obs_id + "_l")}>', 6)
        if layer.uscs.strip():
            self.out(f'<classificationCode codeSpace="USCS">'
                     f"{escape(layer.uscs.strip())}</classificationCode>", 7)
        self.text_element("lithDescription", layer.description, 7)
        if layer.color.strip():
            self.out("<color>", 7)
            self.out(f'<Color gml:id={quoteattr(obs_id + "_c")}>', 8)
            self.text_element("colorName", layer.color, 9)
            self.out("</Color>", 8)
            self.out("</color>", 7)
        if layer.consistency.strip() or layer.moisture.strip():
            self.out("<fieldProperties>", 7)
            self.out(f'<FieldProperties gml:id={quoteattr(obs_id + "_fp")}>',
                     8)
            self.text_element("consistency", layer.consistency, 9)
            self.text_element("moistureCondition", layer.moisture, 9)
            self.out("</FieldProperties>", 8)
            self.out("</fieldProperties>", 7)
        self.out("</Lithology>", 6)
        self.out("</primaryLithology>", 5)
        self.out("</LithologyObservation>", 4)
        self.out("</lithologyObservation>", 3)
        self.notes.layers += 1
        if bottom is None:
            # The last layer of a sheet often has no printed base. DIGGS can
            # carry the contact -- it is written above as a one-value
            # position -- but the app's SiteModel has only INTERVALS, so it
            # will not come back. Said here so the round-trip gate is not
            # asked to find something the reader cannot hold.
            self.notes.skipped.append(
                f"layer at {top[0]:.3f} m has no printed base, so it is "
                f"written as a contact and does not come back through "
                f"parse_diggs, which reads intervals only")

    # -- measurements ------------------------------------------------------
    def measurements(self, inv: Investigation, gml_id: str) -> None:
        for n, record in enumerate(inv.spt):
            self.spt_test(inv, record, gml_id, n)
        for n, sample in enumerate(inv.samples):
            self.index_tests(inv, sample, gml_id, n)
        if inv.cpt is not None and inv.cpt.points:
            self.cone_test(inv, gml_id)
        if inv.dcp is not None and inv.dcp.points:
            self.probe_test(inv, gml_id)

    # -- the two soundings -------------------------------------------------
    def _sounding_extent(self, inv: Investigation, data: Any
                         ) -> Optional[Tuple[float, Optional[float]]]:
        """``(top, base)`` of a sounding in metres, off its own series.

        A sounding's result is POSITIONED like every other result in the
        file, and what it is positioned at is the RUN it covers: the linear
        extent from its shallowest reading to its deepest. The depths of the
        individual readings live in the result set's own depth column, which
        is what a table of many rows against one position is for.
        """
        depths = [q.si_value for q in (p.depth for p in data.points)]
        depths = [d for d in depths if d is not None]
        if not depths:
            self.notes.skipped.append(
                f"{inv.investigation_id}: the sounding's depths are in "
                f"{inv.depth_unit!r}, which is not in the conversion table; "
                f"the series was not written")
            return None
        if data.vertical_axis == "elevation" if isinstance(data, CPTData) \
                else False:
            # An elevation is not a depth along the hole. Turning one into a
            # depth needs the ground level, and where the sheet printed one
            # the conversion is the record's, not this writer's -- so the
            # extent is written from the elevations' own span and the file
            # says which in the procedure's notes.
            return min(depths), max(depths)
        return min(depths), max(depths)

    def cone_test(self, inv: Investigation, gml_id: str) -> None:
        """One cone sounding: its series as ONE positioned result set.

        THE SHAPE, and why it is this one. DIGGS 2.6 has a real home for a
        cone sounding -- ``diggs_geo:StaticConePenetrationTest``, whose
        dictionary entry names ``tip_resistance``, ``sleeve_friction`` and
        ``pore_pressure_u2`` as the properties that occur under it -- and a
        ResultSet is a TABLE, so a sounding of four hundred readings is one
        Test with four hundred ROWS and not four hundred Tests. The first
        column is the depth each row stands at; the rest are the channels.
        The test's own ``location`` is the linear extent the sounding ran
        over, because a result must be positioned and the run is where it is.
        """
        data = inv.cpt
        extent = self._sounding_extent(inv, data)
        if extent is None:
            return
        top, bottom = extent
        columns: List[_Column] = [
            _Column("Depth", "sounding_depth", "m", dictionary=False)]
        channels: List[Tuple[str, str, str, str]] = [
            ("Tip resistance", "tip_resistance", "qc", "kPa"),
            ("Sleeve friction", "sleeve_friction", "fs", "kPa"),
            ("Pore pressure u2", "pore_pressure_u2", "u2", "kPa"),
        ]
        present = [c for c in channels
                   if any(getattr(p, c[2]) is not None for p in data.points)]
        for label, klass, _field, uom in present:
            columns.append(_Column(label, klass, uom))
        ratio = any(p.rf_percent is not None for p in data.points)
        if ratio:
            columns.append(_Column("Friction ratio", "friction_ratio", "%"))
        if len(columns) == 1:
            self.notes.skipped.append(
                f"{inv.investigation_id}: the cone sounding carries no "
                f"channel this writer can put in a result set")
            return
        rows: List[List[Any]] = []
        dropped = 0
        for point in data.points:
            depth = _si(point.depth)
            if depth is None:
                dropped += 1
                continue
            row: List[Any] = [depth[0]]
            for _label, _klass, field_name, _uom in present:
                value = _si(getattr(point, field_name))
                row.append(value[0] if value is not None else "-")
            if ratio:
                row.append(point.rf_percent if point.rf_percent is not None
                           else "-")
            rows.append(row)
        if dropped:
            self.notes.skipped.append(
                f"{inv.investigation_id}: {dropped} cone reading(s) are in a "
                f"unit not in the conversion table and are not in the file")
        if not rows:
            return
        test_id = self.uid(f"cpt_{_ncname(gml_id)}")
        self._test_open("CPT", test_id, gml_id)
        self._result_table(test_id, gml_id, top, bottom, columns, rows)
        self.out("<procedure>", 3)
        self.out(f'<diggs_geo:StaticConePenetrationTest gml:id='
                 f'{quoteattr(test_id + "_proc")}>', 4)
        if data.cone_type.strip():
            self.out(f"<diggs_geo:penetrometerType>"
                     f"{escape(data.cone_type.strip())}"
                     f"</diggs_geo:penetrometerType>", 5)
        rate = _si(data.penetration_rate)
        if rate is not None and rate[1] == "m/s":
            self.measure("diggs_geo:penetrationRate", rate[0], rate[1], 5)
        sleeve = _si(data.sleeve_area)
        if sleeve is not None and sleeve[1] == "m3":
            sleeve = None        # a volume is not an area; say nothing
        if sleeve is not None:
            self.measure("diggs_geo:frictionSleeveArea", sleeve[0],
                         sleeve[1], 5)
        tip = _si(data.cone_area)
        if tip is not None:
            self.measure("diggs_geo:tipArea", tip[0], tip[1], 5)
        self.out("</diggs_geo:StaticConePenetrationTest>", 4)
        self.out("</procedure>", 3)
        self.out("</Test>", 2)
        self.out("</measurement>", 1)
        self.notes.cpt += 1
        self.notes.tests += 1

    def probe_test(self, inv: Investigation, gml_id: str) -> None:
        """One dynamic cone or dynamic probe record, as its series.

        WHICH PROCEDURE, AND WHY. **DIGGS 2.6 has no
        ``DynamicConePenetrometerTest``** -- the name does not appear
        anywhere in the published schema. What it HAS is
        ``diggs_geo:DynamicProbeTest``, whose own documentation is "all
        methods that involve driving a rod by impact hammer" and whose
        elements are exactly what a DCP record needs: a required
        ``penetrationTestType``, then ``hammerMass``, ``hammerDropHeight``,
        ``selfWeightPenetration`` and ``totalPenetration``. So a DCP is a
        ``DynamicProbeTest`` with its printed test type on it, and the
        blows, the penetration and any printed index go in the result set
        beside the depth. Nothing generic is needed and nothing is invented.

        The blow count is the dictionary's own ``blow_count``; a penetration
        and a printed index have no dictionary term and are written under
        this package's codespace, which says out loud that they are ours.
        """
        data = inv.dcp
        extent = self._sounding_extent(inv, data)
        if extent is None:
            return
        top, bottom = extent
        columns: List[_Column] = [
            _Column("Depth", "sounding_depth", "m", dictionary=False)]
        wanted: List[Tuple[str, str, str, str, bool]] = [
            ("Blow count", "blow_count", "blows", "", True),
            ("Penetration", "penetration", "penetration", "m", False),
            ("Penetration index", "penetration_index", "index", "", False),
            ("CBR", "cbr_estimated", "cbr_percent", "%", False),
        ]
        present = [w for w in wanted
                   if any(getattr(p, w[2]) is not None for p in data.points)]
        for label, klass, _f, uom, in_dictionary in present:
            columns.append(_Column(label, klass, uom,
                                   dictionary=in_dictionary))
        if len(columns) == 1:
            self.notes.skipped.append(
                f"{inv.investigation_id}: the dynamic probe record carries "
                f"no value this writer can put in a result set")
            return
        rows: List[List[Any]] = []
        index_uom = ""
        for point in data.points:
            depth = _si(point.depth)
            if depth is None:
                continue
            row: List[Any] = [depth[0]]
            for _label, _klass, field_name, _uom, _d in present:
                raw = getattr(point, field_name)
                if raw is None:
                    row.append("-")
                elif isinstance(raw, Quantity):
                    got = _si(raw)
                    if got is None:
                        row.append("-")
                    else:
                        row.append(got[0])
                        if field_name == "index":
                            index_uom = got[1]
                else:
                    row.append(float(raw))
            rows.append(row)
        if not rows:
            return
        # The index column's unit is whatever the sheet's own index
        # converted to, and it is only known after the rows are walked --
        # a mm/blow index and an MPa dynamic resistance are both "the index
        # column" and convert to different things.
        for column in columns:
            if column.klass == "penetration_index":
                column.uom = index_uom
        test_id = self.uid(f"dcp_{_ncname(gml_id)}")
        self._test_open("DCP", test_id, gml_id)
        self._result_table(test_id, gml_id, top, bottom, columns, rows)
        self.out("<procedure>", 3)
        self.out(f'<diggs_geo:DynamicProbeTest gml:id='
                 f'{quoteattr(test_id + "_proc")}>', 4)
        # penetrationTestType is REQUIRED by the schema, so a record whose
        # sheet named no type still says what it is rather than failing the
        # whole file.
        self.out(f"<diggs_geo:penetrationTestType>"
                 f"{escape(data.test_type.strip() or 'DCP')}"
                 f"</diggs_geo:penetrationTestType>", 5)
        mass = _si(data.hammer_mass)
        if mass is not None and mass[1] == "kg":
            self.measure("diggs_geo:hammerMass", mass[0], mass[1], 5)
        drop = _si(data.hammer_drop)
        if drop is not None and drop[1] == "m":
            self.measure("diggs_geo:hammerDropHeight", drop[0], drop[1], 5)
        total = _si(data.refusal_depth)
        if total is not None and total[1] == "m":
            self.measure("diggs_geo:totalPenetration", total[0], total[1], 5)
        self.out("</diggs_geo:DynamicProbeTest>", 4)
        self.out("</procedure>", 3)
        self.out("</Test>", 2)
        self.out("</measurement>", 1)
        self.notes.dcp += 1
        self.notes.tests += 1

    def _test_open(self, name: str, test_id: str, gml_id: str) -> None:
        self.out("<measurement>", 1)
        self.out(f"<Test gml:id={quoteattr(test_id)}>", 2)
        self.out(f"<gml:name>{escape(name)}</gml:name>", 3)
        self.out("<investigationTarget>Natural Ground</investigationTarget>", 3)
        self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
        self.out(f'<samplingFeatureRef xlink:href="#{gml_id}"/>', 3)

    def _result_set(self, test_id: str, gml_id: str, top: float,
                    bottom: Optional[float],
                    values: Sequence[Tuple[str, str, float]]) -> None:
        """``outcome`` for one test: where it was, and what came out.

        ``values`` is ``(propertyName, propertyClass, value)``. All of them go
        in ONE ResultSet, so an Atterberg test writes its three limits as
        three properties of one result rather than as three tests.
        """
        self._result_table(
            test_id, gml_id, top, bottom,
            [_Column(name, klass) for name, klass, _v in values],
            [[value for _n, _k, value in values]])

    def _result_table(self, test_id: str, gml_id: str, top: float,
                      bottom: Optional[float], columns: Sequence["_Column"],
                      rows: Sequence[Sequence[Any]]) -> None:
        """``outcome`` for one test, as a table of one or more rows.

        ONE row is a set of scalar results -- an Atterberg test's three
        limits. MANY rows are a curve: a grading, a consolidation, a failure
        envelope, one row per point, the columns naming the axes. The two are
        the same element because DIGGS makes no distinction between them; the
        reader tells them apart by counting rows, which is why a scalar set is
        never written as two rows of one value.

        A value that is not a number is written as the text the sheet printed,
        with ``typeData`` string. ``<10`` is not the number ten and the file
        says so.
        """
        self.out("<outcome>", 3)
        self.out(f'<TestResult gml:id={quoteattr(test_id + "_r")}>', 4)
        self.out("<location>", 5)
        self._interval(top, bottom, f"{gml_id}_lsr", f"{test_id}_le", 6)
        self.out("</location>", 5)
        self.out("<results>", 5)
        self.out("<ResultSet>", 6)
        self.out("<parameters>", 7)
        self.out(f'<PropertyParameters gml:id={quoteattr(test_id + "_pp")}>',
                 8)
        self.out("<properties>", 9)
        for i, column in enumerate(columns, start=1):
            self.out(f'<Property index="{i}" '
                     f'gml:id={quoteattr(f"{test_id}_prop{i}")}>', 10)
            self.out(f"<propertyName>{escape(column.name)}</propertyName>", 11)
            self.out(f"<typeData>{column.type_data}</typeData>", 11)
            space = (DICTIONARY_CODESPACE if column.dictionary
                     else APP_CODESPACE)
            self.out(f'<propertyClass codeSpace={quoteattr(space)}>'
                     f"{escape(column.klass)}</propertyClass>", 11)
            if column.uom:
                self.out(f"<uom>"
                         f"{escape(UOM_IN_FILE.get(column.uom, column.uom))}"
                         f"</uom>", 11)
            self.out("</Property>", 10)
        self.out("</properties>", 9)
        self.out("</PropertyParameters>", 8)
        self.out("</parameters>", 7)
        cells = [[_cell(value) for value in row] for row in rows]
        # The tuple separator is a space when a space cannot be anything else
        # -- one row of numbers, which is every result set this writer wrote
        # before WP3, so those files are byte-for-byte what they were. As soon
        # as there is a second row or a cell of text, a space is ambiguous and
        # the separator becomes a semicolon, which no cell may contain.
        plain = len(cells) == 1 and all(
            _is_number(value) for row in rows for value in row)
        sep = " " if plain else ";"
        data = sep.join(",".join(row) for row in cells)
        self.out(f'<dataValues cs="," ts={quoteattr(sep)} '
                 f'decimal=".">{escape(data)}</dataValues>', 7)
        self.out("</ResultSet>", 6)
        self.out("</results>", 5)
        self.out("</TestResult>", 4)
        self.out("</outcome>", 3)

    def spt_test(self, inv: Investigation, record: SPT, gml_id: str,
                 n: int) -> None:
        top = _si(record.depth_top)
        if top is None:
            self.notes.skipped.append(
                f"{inv.investigation_id}: driven record at "
                f"{record.depth_top.value:g} {record.depth_top.unit!r} -- "
                f"unit not in the conversion table")
            return
        bottom = _si(record.depth_bottom)
        drives = _drive_sets(record)
        # The N value goes in the result when the log PRINTS one. When it
        # prints only the drives, the result carries the drives themselves
        # rather than an N this writer invented -- and the round-trip gate
        # then checks the drives, which is what the page actually says.
        if record.n is not None:
            values = [("N-Value", "n_value", float(record.n))]
        elif drives:
            values = [(f"Blow count {i + 1}", "blow_count", float(count))
                      for i, (count, _pen) in enumerate(drives)]
        else:
            return
        test_id = self.uid(f"spt_{_ncname(gml_id)}_{n}")
        self._test_open("SPT", test_id, gml_id)
        self._result_set(test_id, gml_id, top[0],
                         bottom[0] if bottom else None, values)
        self.out("<procedure>", 3)
        self.out(f'<diggs_geo:DrivenPenetrationTest gml:id='
                 f'{quoteattr(test_id + "_proc")}>', 4)
        self.out("<diggs_geo:penetrationTestType>SPT"
                 "</diggs_geo:penetrationTestType>", 5)
        hammer = record.hammer or inv.drilling.hammer_type
        if hammer.strip():
            self.out(f"<diggs_geo:hammerType>{escape(hammer.strip())}"
                     f"</diggs_geo:hammerType>", 5)
        if inv.drilling.hammer_energy_ratio is not None:
            self.out(f'<diggs_geo:hammerEfficiency uom="%">'
                     f"{_num(inv.drilling.hammer_energy_ratio)}"
                     f"</diggs_geo:hammerEfficiency>", 5)
        for i, (count, penetration) in enumerate(drives, start=1):
            self.out("<diggs_geo:driveSet>", 5)
            self.out(f'<diggs_geo:DriveSet gml:id='
                     f'{quoteattr(f"{test_id}_ds{i}")}>', 6)
            self.out(f"<diggs_geo:index>{i}</diggs_geo:index>", 7)
            self.out(f"<diggs_geo:blowCount>{int(count)}"
                     f"</diggs_geo:blowCount>", 7)
            self.out(f'<diggs_geo:penetration uom="m">{_num(penetration)}'
                     f"</diggs_geo:penetration>", 7)
            self.out("</diggs_geo:DriveSet>", 6)
            self.out("</diggs_geo:driveSet>", 5)
        self.out("</diggs_geo:DrivenPenetrationTest>", 4)
        self.out("</procedure>", 3)
        self.out("</Test>", 2)
        self.out("</measurement>", 1)
        self.notes.spt += 1
        self.notes.tests += 1

    def index_tests(self, inv: Investigation, sample: Sample, gml_id: str,
                    n: int) -> None:
        """The index values printed on the log face, one Test per procedure.

        Atterberg's three limits share a Test because they ARE one test;
        everything else stands alone because each is its own procedure.
        """
        top = _si(sample.top)
        if top is None:
            return
        bottom = _si(sample.bottom)
        groups: List[Tuple[str, str, List[Tuple[str, str, float]]]] = []

        def plain(field: str, label: str) -> None:
            value = getattr(sample, field)
            if value is None:
                return
            klass, _uom, proc = PROPERTY_CLASS[field]
            groups.append((label, proc, [(label, klass, float(value))]))

        def quantity(field: str, label: str) -> None:
            q: Optional[Quantity] = getattr(sample, field)
            if q is None:
                return
            got = _si(q)
            klass, uom, proc = PROPERTY_CLASS[field]
            if got is None:
                self.notes.skipped.append(
                    f"{inv.investigation_id}/{sample.sample_id}: {label} "
                    f"{q.value:g} {q.unit!r} -- unit not in the conversion "
                    f"table")
                return
            if got[1] != uom:
                self.notes.skipped.append(
                    f"{inv.investigation_id}/{sample.sample_id}: {label} "
                    f"{q.value:g} {q.unit!r} converts to {got[1]}, not the "
                    f"{uom} this property is written in")
                return
            groups.append((label, proc, [(label, klass, got[0])]))

        plain("water_content", "Water content")
        quantity("dry_unit_weight", "Dry unit weight")
        atterberg = [
            (label, PROPERTY_CLASS[field][0], float(getattr(sample, field)))
            for field, label in (("liquid_limit", "Liquid limit"),
                                 ("plastic_limit", "Plastic limit"),
                                 ("plasticity_index", "Plasticity index"))
            if getattr(sample, field) is not None]
        if atterberg:
            groups.append(("Atterberg limits",
                           "diggs_geo:AtterbergLimitsTest", atterberg))
        plain("fines_percent", "Percent fines")
        quantity("qu", "Unconfined compressive strength")
        quantity("pocket_pen", "Pocket penetrometer")

        for i, (label, procedure, values) in enumerate(groups):
            test_id = self.uid(f"lab_{_ncname(gml_id)}_{n}_{i}")
            self._test_open(label, test_id, gml_id)
            self._result_set(test_id, gml_id, top[0],
                             bottom[0] if bottom else None, values)
            self.out("<procedure>", 3)
            self.out(f'<{procedure} gml:id={quoteattr(test_id + "_proc")}/>',
                     4)
            self.out("</procedure>", 3)
            self.out("</Test>", 2)
            self.out("</measurement>", 1)
            self.notes.tests += 1


    # -- laboratory tests --------------------------------------------------
    def lab_measurements(self, holes: Dict[str, str]) -> None:
        """Every laboratory test in the record, positioned in its own hole."""
        for n, test in enumerate(self.lab):
            if isinstance(test.result, SummaryTableResult):
                self.summary_rows(test, n, holes)
            else:
                self.lab_test(test, n, holes)

    def _place(self, holes: Dict[str, str], name: str,
               depth: Optional[Quantity], bottom: Optional[Quantity],
               where: str) -> Optional[Tuple[str, float, Optional[float]]]:
        """``(hole id, top in m, base in m)``, or None with the reason noted.

        A result in DIGGS is a position along a hole. No hole and no depth
        means no position, and no position means the file has nowhere to put
        the result -- so it is left out and SAID, rather than written at a
        depth nobody measured.
        """
        name = (name or "").strip()
        if not name:
            self.notes.skipped.append(
                f"{where}: the sheet names no boring, and a DIGGS result is a "
                f"position along one; kept in the record, not written here")
            return None
        gml_id = holes.get(name)
        if gml_id is None:
            self.notes.skipped.append(
                f"{where}: boring {name!r} is not in this file")
            return None
        if depth is None:
            self.notes.skipped.append(
                f"{where}: boring {name!r} is named but no depth is printed, "
                f"and a DIGGS result is a position along the hole; kept in "
                f"the record, not written here")
            return None
        top = _si(depth)
        if top is None:
            self.notes.skipped.append(
                f"{where}: depth {depth.value:g} {depth.unit!r} -- unit not "
                f"in the conversion table")
            return None
        base = _si(bottom) if bottom is not None else None
        return gml_id, top[0], (base[0] if base else None)

    def lab_test(self, test: LabTest, n: int,
                 holes: Dict[str, str]) -> None:
        """One laboratory test: its scalars, then a Test per curve it holds."""
        where = (f"lab test {n + 1} ({test.kind}"
                 + (f", {test.investigation_id}" if test.investigation_id
                    else "") + ")")
        placed = self._place(holes, test.investigation_id, test.depth_top,
                             test.depth_bottom, where)
        if placed is None:
            return
        gml_id, top, bottom = placed
        values = _Values(self, where)
        values.quantity("Elevation", "elevation", test.elevation,
                        dictionary=False)
        tables = self._lab_values(test, values)
        stem = f"labtest_{_ncname(gml_id)}_{n}"
        name = test.kind.replace("_", " ")
        if values:
            test_id = self.uid(stem)
            self._lab_open(name, test_id, gml_id, test)
            self._result_table(test_id, gml_id, top, bottom, values.columns,
                               [values.row])
            self.lab_procedure(test, test_id)
            self._lab_close()
        elif not tables:
            self.notes.skipped.append(
                f"{where}: the test carries no value this writer can put in "
                f"a result set")
        for label, columns, rows in tables:
            if not rows:
                continue
            test_id = self.uid(f"{stem}_{_ncname(label)}")
            self._lab_open(f"{name}: {label.replace('_', ' ')}", test_id,
                           gml_id, test)
            self._result_table(test_id, gml_id, top, bottom, columns, rows)
            self.lab_procedure(test, test_id)
            self._lab_close()

    def _lab_open(self, name: str, test_id: str, gml_id: str,
                  test: LabTest) -> None:
        self.out("<measurement>", 1)
        self.out(f"<Test gml:id={quoteattr(test_id)}>", 2)
        self.out(f"<gml:name>{escape(name)}</gml:name>", 3)
        self.out("<investigationTarget>Natural Ground</investigationTarget>", 3)
        self.out(f'<projectRef xlink:href="#{self.project_id}"/>', 3)
        self.out(f'<samplingFeatureRef xlink:href="#{gml_id}"/>', 3)
        # The sample is named as a PROPERTY, not as a sampleRef: the sheet
        # printed a label, and an xlink to a Sample element this file may not
        # contain would be a link to nothing.
        for key, value in (("sample_id", test.sample_id),
                           ("laboratory", test.lab),
                           ("test_date", test.date),
                           ("source_page",
                            ",".join(str(p) for p in test.pages))):
            if not str(value or "").strip():
                continue
            self.out("<otherMeasurementProperty>", 3)
            self.out(f'<Parameter gml:id='
                     f'{quoteattr(self.uid(test_id + "_" + key))}>', 4)
            self.text_element("parameterName", key, 5)
            self.text_element("parameterValue", value, 5)
            self.out("</Parameter>", 4)
            self.out("</otherMeasurementProperty>", 3)

    def _lab_close(self) -> None:
        self.out("</Test>", 2)
        self.out("</measurement>", 1)
        self.notes.lab_tests += 1
        self.notes.tests += 1

    def lab_procedure(self, test: LabTest, test_id: str) -> None:
        """How the test was done: the standard, and the settings it printed.

        The VALUES are in the result set. This element carries what the sheet
        says about the method, and only the parts of it that need nothing the
        sheet did not print.
        """
        element = LAB_PROCEDURE.get(test.kind)
        if element is None:
            return
        pid = self.uid(f"{test_id}_proc")
        result = test.result
        self.out("<procedure>", 3)
        self.out(f"<{element} gml:id={quoteattr(pid)}>", 4)
        self.out(f"<gml:name>{escape(test.kind.replace('_', ' '))}"
                 f"</gml:name>", 5)
        if test.standard.strip():
            self.out("<testProcedureMethod>", 5)
            self.out(f'<Specification gml:id={quoteattr(pid + "_spec")}>', 6)
            self.out(f"<gml:name>{escape(test.standard.strip())}</gml:name>",
                     7)
            self.text_element("standardReferenceNumber", test.standard, 7)
            self.out("</Specification>", 6)
            self.out("</testProcedureMethod>", 5)
        for key, value in self._procedure_notes(test).items():
            self.out("<otherTestProperty>", 5)
            self.out(f'<Parameter gml:id='
                     f'{quoteattr(self.uid(pid + "_" + _ncname(key)))}>', 6)
            self.text_element("parameterName", key, 7)
            self.text_element("parameterValue", value, 7)
            self.out("</Parameter>", 6)
            self.out("</otherTestProperty>", 5)
        # The concrete elements each procedure has of its own, where the
        # sheet printed what they need.
        if isinstance(result, ConsolidationResult):
            if result.test_type.strip():
                self.text_element("diggs_geo:consolidationTestType",
                                  result.test_type, 5)
            pressure = _si(result.swell_pressure)
            if pressure is not None:
                self.measure("diggs_geo:swellingPressure", pressure[0],
                             pressure[1], 5)
            pc = _si(result.pc)
            if pc is not None:
                self.measure("diggs_geo:estimatedPreConsolidationStress",
                             pc[0], pc[1], 5)
        elif isinstance(result, StrengthResult):
            if test.kind == "triaxial" and result.test_type.strip():
                self.text_element("triaxialTestType", result.test_type, 5)
            elif test.kind == "direct_shear" and result.test_type.strip():
                self.text_element("diggs_geo:directShearTestType",
                                  result.test_type, 5)
            elif test.kind in ("unconfined", "unconfined_rock") \
                    and result.strain_at_failure_percent is not None:
                self.measure("diggs_geo:axialStrainAtFailure",
                             result.strain_at_failure_percent, "%", 5)
        elif isinstance(result, CompactionResult):
            volume = _si(result.mould_volume)
            if volume is not None:
                self.measure("diggs_geo:mouldVolume", volume[0], volume[1], 5)
            if result.layers is not None:
                self.out(f"<diggs_geo:numberOfLayers>{int(result.layers)}"
                         f"</diggs_geo:numberOfLayers>", 5)
            if result.blows_per_layer is not None:
                self.out(f"<diggs_geo:blowsPerLayer>"
                         f"{int(result.blows_per_layer)}"
                         f"</diggs_geo:blowsPerLayer>", 5)
        elif isinstance(result, CBRResult):
            self.cbr_trial(result, pid)
        self.out(f"</{element}>", 4)
        self.out("</procedure>", 3)

    def cbr_trial(self, result: CBRResult, pid: str) -> None:
        """The CBR's own trial element, which every field of is optional."""
        if (result.cbr_at_0_1in is None and result.cbr_at_0_2in is None
                and result.soaked is None and result.surcharge is None):
            return
        self.out("<diggs_geo:trial>", 5)
        self.out(f'<diggs_geo:LabCBRTestTrial gml:id='
                 f'{quoteattr(pid + "_trial")}>', 6)
        if result.cbr_at_0_1in is not None:
            self.measure("diggs_geo:cbr_0.1", result.cbr_at_0_1in, "%", 7)
        if result.cbr_at_0_2in is not None:
            self.measure("diggs_geo:cbr_0.2", result.cbr_at_0_2in, "%", 7)
        if result.soaked is not None:
            self.out(f"<diggs_geo:soaking>"
                     f"{'true' if result.soaked else 'false'}"
                     f"</diggs_geo:soaking>", 7)
        surcharge = _si(result.surcharge)
        if surcharge is not None and surcharge[1] == "kPa":
            self.measure("diggs_geo:surchargePressure", surcharge[0],
                         surcharge[1], 7)
        elif surcharge is not None:
            # DIGGS calls this a pressure and a laboratory calls it a stack
            # of weights: a CBR surcharge is printed in kilograms as often as
            # in kilopascals. A mass written into a pressure element fails
            # the schema on the whole file, so it goes in the result set as
            # itself and is named here.
            self.notes.skipped.append(
                f"CBR surcharge {result.surcharge.value:g} "
                f"{result.surcharge.unit!r} is a {surcharge[1]}, not a "
                f"pressure; written as a result and not as "
                f"surchargePressure")
        self.out("</diggs_geo:LabCBRTestTrial>", 6)
        self.out("</diggs_geo:trial>", 5)

    @staticmethod
    def _procedure_notes(test: LabTest) -> Dict[str, str]:
        """What the sheet said about the method that has no element for it."""
        out: Dict[str, str] = {}
        result = test.result
        if test.language.strip():
            out["sheet_language"] = test.language.strip()
        if test.curves_digitised:
            out["curve_digitised"] = ("a value in this test was read off a "
                                      "plot, not a table")
        if isinstance(result, CompactionResult) and result.method.strip():
            out["compaction_method"] = result.method.strip()
        if isinstance(result, StrengthResult) and result.test_type.strip():
            out["test_type"] = result.test_type.strip()
        return out

    # -- the values, by kind ----------------------------------------------
    def _lab_values(self, test: LabTest, values: "_Values"
                    ) -> List[Tuple[str, List[_Column], List[List[Any]]]]:
        """Fill ``values`` with the scalars, and return the curve tables.

        One method rather than ten small ones because what it IS is a table:
        every kind of sheet, every value it can print, and the DIGGS property
        each value is written as. Read it as that table.
        """
        result = test.result
        tables: List[Tuple[str, List[_Column], List[List[Any]]]] = []
        if result is None:
            return tables

        if isinstance(result, AtterbergResult):
            values.number("Liquid limit", "liquid_limit", result.ll, "%")
            values.number("Plastic limit", "plastic_limit", result.pl, "%")
            values.number("Plasticity index", "plasticity_index", result.pi,
                          "%")
            values.number("Shrinkage limit", "shrinkage_limit",
                          result.shrinkage_limit, "%")
            values.flag("Non-plastic", "non_plastic", result.non_plastic)
            values.number("Water content", "water_content_natural",
                          result.water_content, "%")
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            if result.flow_curve:
                tables.append((
                    "flow_curve",
                    [_Column("Blow count", "blow_count", "", "double"),
                     _Column("Water content", "water_content_natural", "%")],
                    [[float(blows), float(water)]
                     for blows, water in result.flow_curve]))
            if result.pl_trials:
                tables.append((
                    "plastic_limit_trials",
                    [_Column("Water content", "water_content_natural", "%")],
                    [[float(v)] for v in result.pl_trials]))

        elif isinstance(result, GradationResult):
            values.number("Percent cobbles", "percent_cobbles",
                          result.cobbles_percent, "%")
            values.number("Percent gravel", "percent_gravel",
                          result.gravel_percent, "%")
            values.number("Percent sand", "percent_sand",
                          result.sand_percent, "%")
            values.number("Percent silt", "percent_silt",
                          result.silt_percent, "%")
            # The 2.6 dictionary's only clay term is "percent finer than two
            # microns", which is a narrower claim than a hydrometer sheet's
            # clay fraction, so the clay fraction is written as ours.
            values.number("Percent clay", "percent_clay",
                          result.clay_percent, "%", dictionary=False)
            values.number("Percent fines", "percent_fines",
                          result.fines_percent, "%")
            for name, klass, quantity, in_dictionary in (
                    ("D10", "d10", result.d10, True),
                    ("D30", "d30", result.d30, True),
                    ("D50", "d50", result.d50, True),
                    ("D60", "d60", result.d60, True),
                    ("D85", "d85", result.d85, True),
                    ("D90", "d90", result.d90, False),
                    ("D100", "d100", result.d100, False)):
                values.length_mm(name, klass, quantity,
                                 dictionary=in_dictionary)
            values.number("Coefficient of uniformity", "coef_uniformity",
                          result.cu)
            values.number("Coefficient of curvature", "coef_curvature",
                          result.cc)
            values.number("Water content", "water_content_natural",
                          result.water_content, "%")
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            rows = []
            for point in result.percent_passing:
                size = _si(point.size) if point.size is not None else None
                rows.append([size[0] * 1000.0 if size else "-",
                             float(point.percent_passing),
                             point.sieve or "-"])
            if rows:
                tables.append((
                    "grading",
                    [_Column("Particle size", "particle_size", "mm", "double",
                             False),
                     _Column("Percent passing", "percent_passing", "%",
                             "double", False),
                     _Column("Sieve", "sieve_designation", "", "string",
                             False)],
                    rows))

        elif isinstance(result, ConsolidationResult):
            values.quantity("Preconsolidation pressure",
                            "preconsolidation_pressure", result.pc, "kPa")
            values.number("Compression index", "compression_index", result.cc)
            values.number("Recompression index", "recompression_index",
                          result.cr)
            values.quantity("Coefficient of consolidation",
                            "coef_consolidation_vertical", result.cv)
            values.number("Initial void ratio", "void_ratio", result.e0,
                          dictionary=False)
            values.number("Swell", "swell_percent", result.swell_percent, "%",
                          dictionary=False)
            values.quantity("Swell seating pressure",
                            "swell_seating_pressure", result.swell_at, "kPa",
                            dictionary=False)
            values.quantity("Swelling pressure", "swell_pressure",
                            result.swell_pressure, "kPa", dictionary=False)
            values.quantity("Dry density", "dry_density",
                            result.dry_unit_weight, "kN/m3")
            values.number("Water content", "water_content_natural", result.wc,
                          "%")
            values.number("Degree of saturation", "degree_of_saturation",
                          result.saturation_percent, "%")
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            rows = []
            has_strain = any(p.strain_percent is not None
                             for p in result.points)
            has_e = any(p.void_ratio is not None for p in result.points)
            for point in result.points:
                stress = _si(point.stress)
                if stress is None:
                    self.notes.skipped.append(
                        f"consolidation point {point.stress.value:g} "
                        f"{point.stress.unit!r} -- unit not in the conversion "
                        f"table")
                    continue
                row: List[Any] = [stress[0]]
                if has_strain:
                    row.append(point.strain_percent
                               if point.strain_percent is not None else "-")
                if has_e:
                    row.append(point.void_ratio
                               if point.void_ratio is not None else "-")
                row.append(point.stage or "load")
                rows.append(row)
            if rows:
                columns = [_Column("Applied pressure", "applied_pressure",
                                   "kPa", "double", False)]
                if has_strain:
                    columns.append(_Column("Axial strain", "axial_strain",
                                           "%", "double", False))
                if has_e:
                    columns.append(_Column("Void ratio", "void_ratio", "",
                                           "double", False))
                columns.append(_Column("Stage", "load_stage", "", "string",
                                       False))
                tables.append(("curve", columns, rows))

        elif isinstance(result, StrengthResult):
            values.quantity("Unconfined compressive strength",
                            "compressive_strength_unconfined", result.qu,
                            "kPa")
            values.quantity("Undrained shear strength",
                            "shear_strength_undrained", result.su, "kPa")
            values.quantity("Cohesion", "cohesion_peak", result.c, "kPa")
            values.number("Friction angle", "friction_angle_peak",
                          result.phi_deg, "deg")
            values.quantity("Residual cohesion", "cohesion_residual",
                            result.c_residual, "kPa")
            values.number("Residual friction angle",
                          "friction_angle_residual",
                          result.phi_residual_deg, "deg")
            values.number("Axial strain at failure",
                          "axial_strain_at_failure",
                          result.strain_at_failure_percent, "%",
                          dictionary=False)
            values.number("Water content", "water_content_natural", result.wc,
                          "%")
            values.quantity("Dry density", "dry_density", result.dry_density,
                            "kN/m3")
            values.quantity("Bulk density", "bulk_density",
                            result.wet_density, "kN/m3")
            values.text("Rock type", "rock_type", result.rock_type,
                        dictionary=False)
            values.text("Weathering grade", "weathering_grade",
                        result.weathering, dictionary=False)
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            tables.extend(self._specimen_table(result))
            tables.extend(self._strength_curves(result))

        elif isinstance(result, CompactionResult):
            values.quantity("Maximum dry density", "dry_density_max",
                            result.max_dry_density, "kN/m3")
            values.number("Optimum water content", "water_content_optimum",
                          result.optimum_wc, "%")
            values.quantity("Rammer mass", "rammer_mass", result.rammer_mass,
                            "kg", dictionary=False)
            values.quantity("Mould volume", "mould_volume",
                            result.mould_volume, "m3", dictionary=False)
            values.number("Layers", "compaction_layers", result.layers,
                          dictionary=False)
            values.number("Blows per layer", "blows_per_layer",
                          result.blows_per_layer, dictionary=False)
            values.number("Oversize fraction", "oversize_percent",
                          result.oversize_percent, "%", dictionary=False)
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            rows = []
            for point in result.points:
                density = _si(point.dry_density)
                if density is None:
                    self.notes.skipped.append(
                        f"compaction point {point.dry_density.value:g} "
                        f"{point.dry_density.unit!r} -- unit not in the "
                        f"conversion table")
                    continue
                rows.append([float(point.water_content), density[0]])
            if rows:
                tables.append((
                    "curve",
                    [_Column("Water content", "water_content_natural", "%"),
                     _Column("Dry density", "dry_density", "kN/m3")],
                    rows))

        elif isinstance(result, CBRResult):
            values.number("CBR", "cbr", result.cbr_percent, "%",
                          dictionary=False)
            values.number("CBR at 0.1 in", "cbr_0.1", result.cbr_at_0_1in,
                          "%")
            values.number("CBR at 0.2 in", "cbr_0.2", result.cbr_at_0_2in,
                          "%")
            values.number("Swell", "swell_percent", result.swell_percent, "%",
                          dictionary=False)
            values.quantity("Dry density", "dry_density", result.dry_density,
                            "kN/m3")
            values.number("Water content", "water_content_natural", result.wc,
                          "%")
            values.number("Percent compaction", "percent_compaction",
                          result.compaction_percent, "%", dictionary=False)
            # The uom follows the VALUE here: a surcharge is printed in
            # kilograms on one form and kilopascals on the next, and both
            # are what that laboratory measured.
            values.quantity("Surcharge", "surcharge", result.surcharge,
                            dictionary=False)
            if result.points:
                tables.append((
                    "penetration",
                    [_Column("Penetration", "penetration", "", "double",
                             False),
                     _Column("Load", "penetration_load", "", "double",
                             False)],
                    [[float(x), float(y)] for x, y in result.points]))

        elif isinstance(result, MoistureDensityResult):
            values.number("Water content", "water_content_natural", result.wc,
                          "%")
            values.quantity("Bulk density", "bulk_density",
                            result.wet_density, "kN/m3")
            values.quantity("Dry density", "dry_density", result.dry_density,
                            "kN/m3")
            values.number("Specific gravity", "specific_gravity_solids",
                          result.specific_gravity)
            values.number("Void ratio", "void_ratio", result.void_ratio,
                          dictionary=False)
            values.number("Degree of saturation", "degree_of_saturation",
                          result.saturation_percent, "%")
            values.number("Loss on ignition", "LOI", result.organic_percent,
                          "%")
            values.number("Ash content", "ash_content", result.ash_percent,
                          "%", dictionary=False)
            values.text("USCS symbol", "uscs_symbol", result.uscs)
            if result.water_contents:
                tables.append((
                    "determinations",
                    [_Column("Water content", "water_content_natural", "%")],
                    [[float(v)] for v in result.water_contents]))

        elif isinstance(result, ChemicalResult):
            values.reported("pH", "pH", result.pH)
            values.reported("Resistivity", "resistivity", result.resistivity,
                            "ohm.m")
            values.reported("Minimum resistivity", "resistivity_minimum",
                            result.resistivity_minimum, "ohm.m",
                            dictionary=False)
            values.reported("Sulfate", "sulfate_content", result.sulfate)
            values.reported("Chloride", "chloride_content", result.chloride)
            values.reported("Sulfide", "sulfide_content", result.sulfides,
                            dictionary=False)
            values.reported("Redox potential", "redox_potential",
                            result.redox, "mV")
            values.reported("Total salts", "total_salts", result.total_salts,
                            dictionary=False)
            values.reported("Conductivity", "conductivity",
                            result.conductivity)
            values.reported("Temperature", "temperature", result.temperature,
                            "degC")
            values.reported("Water content", "water_content_natural",
                            result.wc, "%")
            values.reported("Reporting limit", "reporting_limit",
                            result.reporting_limit, dictionary=False)
            values.text("Laboratory sample", "lab_sample_id",
                        result.lab_sample_id, dictionary=False)

        elif isinstance(result, OtherResult):
            for key, value in sorted(result.fields.items()):
                values.text(key, _ncname(key).lower(), str(value),
                            dictionary=False)
            values.flag("No results on this page", "no_results",
                        result.no_results, dictionary=False)
        return tables

    def _specimen_table(self, result: StrengthResult
                        ) -> List[Tuple[str, List[_Column], List[List[Any]]]]:
        """A strength test's specimens, one row each."""
        if not result.specimens:
            return []
        columns = [_Column("Specimen", "specimen_id", "", "string", False)]
        wanted: List[Tuple[str, str, str, Any]] = [
            ("Confining pressure", "confining_pressure", "kPa", "confining"),
            ("Peak deviator stress", "deviator_stress_peak", "kPa",
             "peak_deviator"),
            ("Pore pressure", "pore_pressure_at_failure", "kPa",
             "pore_pressure"),
            ("Cohesion", "cohesion_peak", "kPa", "c"),
            ("Dry density", "dry_density", "kN/m3", "dry_density"),
            ("Bulk density", "bulk_density", "kN/m3", "wet_density"),
            ("Specimen height", "specimen_height", "m", "height"),
            ("Specimen diameter", "specimen_diameter", "m", "diameter"),
        ]
        present = [(name, klass, uom, attr) for name, klass, uom, attr
                   in wanted
                   if any(getattr(s, attr) is not None
                          for s in result.specimens)]
        plain = [(name, klass, uom, attr) for name, klass, uom, attr in (
            ("Axial strain at peak", "axial_strain_at_peak", "%",
             "strain_at_peak_percent"),
            ("Friction angle", "friction_angle_peak", "deg", "phi_deg"),
            ("Water content", "water_content_natural", "%", "wc"),
            ("Effective stress ratio", "effective_stress_ratio", "",
             "stress_ratio"))
            if any(getattr(s, attr) is not None for s in result.specimens)]
        for name, klass, uom, _attr in present:
            columns.append(_Column(name, klass, uom, "double",
                                   klass in ("cohesion_peak", "dry_density",
                                             "bulk_density")))
        for name, klass, uom, _attr in plain:
            columns.append(_Column(name, klass, uom, "double",
                                   klass in ("friction_angle_peak",
                                             "water_content_natural")))
        rows: List[List[Any]] = []
        for n, specimen in enumerate(result.specimens, start=1):
            row: List[Any] = [specimen.specimen_id or str(n)]
            for _name, _klass, uom, attr in present:
                got = _si(getattr(specimen, attr))
                row.append(got[0] if got is not None else "-")
            for _name, _klass, _uom, attr in plain:
                value = getattr(specimen, attr)
                row.append(float(value) if value is not None else "-")
            rows.append(row)
        return [("specimens", columns, rows)]

    def _strength_curves(self, result: StrengthResult
                         ) -> List[Tuple[str, List[_Column], List[List[Any]]]]:
        """The envelope and the stress-strain curve, told apart by their x.

        The record keeps both in one list of points. A point whose x is a
        percentage is a strain and belongs to a stress-strain curve; a point
        whose x is a pressure is a normal stress and belongs to an envelope.
        The unit is the only thing that says which, and it is enough.
        """
        envelope: List[List[Any]] = []
        strain: List[List[Any]] = []
        for point in result.points:
            x, y = _si(point.x), _si(point.y)
            if x is None or y is None:
                self.notes.skipped.append(
                    f"strength curve point ({point.x}, {point.y}) -- a unit "
                    f"not in the conversion table")
                continue
            row = [x[0], y[0], point.specimen or "-"]
            (strain if x[1] == "%" else envelope).append(row)
        out: List[Tuple[str, List[_Column], List[List[Any]]]] = []
        if envelope:
            out.append((
                "envelope",
                [_Column("Normal stress", "normal_stress", "kPa", "double",
                         False),
                 _Column("Shear stress", "shear_stress", "kPa", "double",
                         False),
                 _Column("Specimen", "specimen_id", "", "string", False)],
                envelope))
        if strain:
            out.append((
                "stress_strain",
                [_Column("Axial strain", "axial_strain", "%", "double",
                         False),
                 _Column("Deviator stress", "deviator_stress", "kPa",
                         "double", False),
                 _Column("Specimen", "specimen_id", "", "string", False)],
                strain))
        return out

    # -- the summary table -------------------------------------------------
    def summary_rows(self, test: LabTest, n: int,
                     holes: Dict[str, str]) -> None:
        """A summary table, one Test per row.

        Its rows belong to different holes at different depths, so the table
        cannot be one positioned result. Each row becomes its own Test with
        no procedure element: a row is one sample's results gathered out of
        several tests, and naming one procedure for it would say the
        laboratory ran a test it did not.
        """
        result = test.result
        if not isinstance(result, SummaryTableResult):
            return
        for i, row in enumerate(result.rows):
            where = (f"summary row {i + 1} of lab test {n + 1} "
                     f"({row.investigation_id or 'no boring named'})")
            placed = self._place(holes, row.investigation_id, row.depth_top,
                                 row.depth_bottom, where)
            if placed is None:
                continue
            gml_id, top, bottom = placed
            values = _Values(self, where)
            self._summary_values(row, values)
            stem = f"labsummary_{_ncname(gml_id)}_{n}_{i}"
            if values:
                test_id = self.uid(stem)
                self._lab_open("summary of laboratory tests", test_id,
                               gml_id, test)
                self._result_table(test_id, gml_id, top, bottom,
                                   values.columns, [values.row])
                self._lab_close()
            passing = [[(_si(p.size)[0] * 1000.0 if p.size is not None
                         and _si(p.size) is not None else "-"),
                        float(p.percent_passing), p.sieve or "-"]
                       for p in row.percent_passing]
            if passing:
                test_id = self.uid(f"{stem}_grading")
                self._lab_open("summary of laboratory tests: grading",
                               test_id, gml_id, test)
                self._result_table(
                    test_id, gml_id, top, bottom,
                    [_Column("Particle size", "particle_size", "mm",
                             "double", False),
                     _Column("Percent passing", "percent_passing", "%",
                             "double", False),
                     _Column("Sieve", "sieve_designation", "", "string",
                             False)],
                    passing)
                self._lab_close()

    @staticmethod
    def _summary_values(row: SummaryRow, values: "_Values") -> None:
        values.quantity("Elevation", "elevation", row.elevation_top,
                        dictionary=False)
        values.number("Water content", "water_content_natural", row.wc, "%")
        values.number("Liquid limit", "liquid_limit", row.ll, "%")
        values.reported("Plastic limit", "plastic_limit", row.pl, "%")
        values.number("Plasticity index", "plasticity_index", row.pi, "%")
        values.number("Percent fines", "percent_fines", row.fines_percent,
                      "%")
        values.number("Percent sand", "percent_sand", row.sand_percent, "%")
        values.number("Percent gravel", "percent_gravel", row.gravel_percent,
                      "%")
        values.number("Percent silt and clay", "percent_silt_and_clay",
                      row.silt_clay_percent, "%", dictionary=False)
        values.quantity("Bulk density", "bulk_density", row.wet_density,
                        "kN/m3")
        values.quantity("Dry density", "dry_density", row.dry_density,
                        "kN/m3")
        values.quantity("Maximum dry density", "dry_density_max",
                        row.max_dry_density, "kN/m3")
        values.number("Optimum water content", "water_content_optimum",
                      row.optimum_wc, "%")
        values.quantity("Unconfined compressive strength",
                        "compressive_strength_unconfined", row.qu, "kPa")
        values.quantity("Undrained shear strength",
                        "shear_strength_undrained", row.su, "kPa")
        values.quantity("Cohesion", "cohesion_peak", row.c, "kPa")
        values.number("Friction angle", "friction_angle_peak", row.phi_deg,
                      "deg")
        values.number("Swell", "swell_percent", row.swell_percent, "%",
                      dictionary=False)
        values.number("Loss on ignition", "LOI", row.organic_percent, "%")
        values.reported("pH", "pH", row.pH)
        values.reported("Resistivity", "resistivity", row.resistivity,
                        "ohm.m")
        values.reported("Sulfate", "sulfate_content", row.sulfate)
        values.reported("Chloride", "chloride_content", row.chloride)
        values.reported("Sulfide", "sulfide_content", row.sulfides,
                        dictionary=False)
        values.reported("Redox potential", "redox_potential", row.redox, "mV")
        values.text("USCS symbol", "uscs_symbol", row.uscs)
        values.text("Stratum", "stratum", row.stratum, dictionary=False)
        values.text("Sample type", "sample_type", row.sample_type,
                    dictionary=False)
        values.text("Laboratory", "laboratory", row.lab, dictionary=False)


class _Values:
    """The columns and the one row of a scalar result set, built up by kind.

    Every ``add`` is a no-op for a value the record does not hold, so a
    builder reads as a list of everything a sheet of that kind CAN print and
    writes only what this one did.
    """

    def __init__(self, writer: "_Writer", where: str) -> None:
        self.writer = writer
        self.where = where
        self.columns: List[_Column] = []
        self.row: List[Any] = []

    def __bool__(self) -> bool:
        return bool(self.columns)

    def number(self, name: str, klass: str, value: Optional[float],
               uom: str = "", dictionary: bool = True) -> None:
        if value is None:
            return
        self.columns.append(_Column(name, klass, uom, "double", dictionary))
        self.row.append(float(value))

    def text(self, name: str, klass: str, value: str,
             dictionary: bool = True) -> None:
        if not str(value or "").strip():
            return
        self.columns.append(_Column(name, klass, "", "string", dictionary))
        self.row.append(str(value).strip())

    def flag(self, name: str, klass: str, value: bool,
             dictionary: bool = True) -> None:
        """A boolean result. Written only when TRUE.

        ``non_plastic`` false is not a result, it is the absence of one, and
        a file full of false flags would say a laboratory measured something
        it did not.
        """
        if not value:
            return
        self.columns.append(_Column(name, klass, "", "boolean", dictionary))
        self.row.append("true")

    def quantity(self, name: str, klass: str, quantity: Optional[Quantity],
                 uom: Optional[str] = None, dictionary: bool = True) -> None:
        """A value with a unit, converted once, or a note saying why not.

        ``uom`` pins the unit the property must be written in; leave it None
        where the sheet's own kind decides -- a sulfate content is a
        percentage on one form and milligrams per kilogram on the next, and
        both are right.
        """
        if quantity is None:
            return
        got = _si(quantity)
        if got is None:
            self.writer.notes.skipped.append(
                f"{self.where}: {name} {quantity.value:g} "
                f"{quantity.unit!r} -- unit not in the conversion table")
            return
        if uom is not None and got[1] != uom:
            self.writer.notes.skipped.append(
                f"{self.where}: {name} {quantity.value:g} {quantity.unit!r} "
                f"converts to {got[1]}, not the {uom} this property is "
                f"written in")
            return
        self.number(name, klass, got[0], got[1], dictionary)

    def length_mm(self, name: str, klass: str, quantity,
                  dictionary: bool = True) -> None:
        """A length written in MILLIMETRES rather than in metres.

        SI once means metres everywhere -- except for a particle size, where
        metres is the wrong unit for a person reading the file and a hair's
        breadth from the precision it is written at. Every grading here is in
        millimetres and says so in its uom, which is what SI-once actually
        asks for: one unit, stated.
        """
        if quantity is None:
            return
        got = _si(quantity)
        if got is None:
            self.writer.notes.skipped.append(
                f"{self.where}: {name} {quantity.value:g} "
                f"{quantity.unit!r} -- unit not in the conversion table")
            return
        if got[1] != "m":
            self.writer.notes.skipped.append(
                f"{self.where}: {name} {quantity.value:g} {quantity.unit!r} "
                f"is not a length")
            return
        self.number(name, klass, got[0] * 1000.0, "mm", dictionary)

    def reported(self, name: str, klass: str, value: Any,
                 uom: Optional[str] = None, dictionary: bool = True) -> None:
        """A value a sheet printed as a number, with a unit, or as words."""
        if value is None:
            return
        if isinstance(value, str):
            self.text(name, klass, value, dictionary)
            return
        if isinstance(value, Quantity):
            self.quantity(name, klass, value, uom, dictionary)
            return
        self.number(name, klass, float(value), "", dictionary)


def _drive_sets(record: SPT) -> List[Tuple[int, float]]:
    """``(blow count, penetration in metres)`` for each increment.

    A plain count takes the record's own increment, or the 0.1524 m (6 in)
    every driven-sample standard uses. A refusal -- ``50/5"``, ``100/0.1m`` --
    is split into its count and the distance it actually went, which is what
    DIGGS wants and what the notation means.
    """
    default = 0.1524
    increment = record.increment.to_si() if record.increment else None
    step = increment.value if increment is not None else default
    out: List[Tuple[int, float]] = []
    for entry in record.blows:
        if isinstance(entry, (int, float)):
            out.append((int(entry), step))
            continue
        text = str(entry).strip()
        if text.isdigit():
            out.append((int(text), step))
            continue
        match = re.match(
            r'^\s*(\d+)\s*/\s*([0-9]*\.?[0-9]+)\s*(cm|mm|m|in|")?\s*$', text)
        if match:
            count = int(match.group(1))
            distance = float(match.group(2))
            unit = (match.group(3) or '"').replace('"', "in")
            from report_ingest.model import to_si as _convert
            got = _convert(distance, unit)
            out.append((count, got[0] if got else distance * 0.0254))
            continue
        # Anything else -- a word, a range, a note -- is not a blow count and
        # is left out rather than forced into an integer.
    return out


def _iso_date(text: str) -> str:
    """An ISO date from what a log prints, or ``''``.

    Deliberately narrow: m/d/Y, d-m-Y, Y-m-d and the two-digit-year forms
    that a US log uses. A date this cannot read is NOT guessed -- a log whose
    ``3/4/2025`` might be March or April keeps its own spelling in the header
    fields and writes no ``whenConstructed``.
    """
    from datetime import datetime
    raw = str(text or "").strip()
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%d-%b-%y",
                "%Y/%m/%d", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def write_diggs(record_or_investigations: Any,
                project: Optional[Project] = None, *,
                document_id: str = "",
                lab_tests: Optional[Sequence[LabTest]] = None,
                notes: Optional[List[DiggsWriteNotes]] = None) -> str:
    """DIGGS 2.6 XML for a record, an investigation, a lab test, or a list.

    ``project`` fills the ``Project`` element; when a whole
    :class:`~report_ingest.model.ReportRecord` is passed its own project and
    its own lab tests are used unless they are given here. A list may mix
    investigations and lab tests, which is what an ingest actually has in
    hand. Pass ``notes`` as an empty list to be handed the writer's
    :class:`DiggsWriteNotes` -- what it had to leave out and why, and which
    holes exist only because a lab sheet named them.
    """
    given_lab = list(lab_tests) if lab_tests is not None else None
    lab: List[LabTest] = []
    if isinstance(record_or_investigations, ReportRecord):
        record = record_or_investigations
        investigations = list(record.investigations)
        lab = list(record.lab_tests)
        project = project or record.project
        document_id = document_id or record.document.report_id
    elif isinstance(record_or_investigations, Investigation):
        investigations = [record_or_investigations]
    elif isinstance(record_or_investigations, LabTest):
        investigations = []
        lab = [record_or_investigations]
    elif isinstance(record_or_investigations, Iterable):
        items = list(record_or_investigations)
        investigations = [i for i in items if isinstance(i, Investigation)]
        lab = [i for i in items if isinstance(i, LabTest)]
        other = [i for i in items
                 if not isinstance(i, (Investigation, LabTest))]
        if other:
            raise TypeError(
                f"every item must be a model.Investigation or a "
                f"model.LabTest, not {type(other[0]).__name__}")
    else:
        raise TypeError(
            f"write_diggs takes a ReportRecord, an Investigation, a LabTest "
            f"or a list of them, not "
            f"{type(record_or_investigations).__name__}")
    if given_lab is not None:
        lab = given_lab
    for test in lab:
        if not isinstance(test, LabTest):
            raise TypeError(
                f"every lab test must be a model.LabTest, not "
                f"{type(test).__name__}")
    for inv in investigations:
        if not inv.units_known and any(
                (inv.layers, inv.samples, inv.spt, inv.water)):
            raise ValueError(
                f"investigation {inv.investigation_id!r} carries depths in an "
                f"unknown unit (units_known is False). A DIGGS file states a "
                f"uom on every measure, so writing one would mean inventing "
                f"a unit. Fix the unit or write no DIGGS for this log.")
    writer = _Writer(investigations, project, document_id, lab)
    xml = writer.build()
    if notes is not None:
        notes.append(writer.notes)
    return xml


# ---------------------------------------------------------------------------
# gate one: is it DIGGS
# ---------------------------------------------------------------------------

def diggs_schema_gate(xml: str, *, schema_version: str = "2.6"
                      ) -> Tuple[bool, List[str]]:
    """``(ok, errors)`` from the DIGGS XSD that pydiggs bundles.

    ``(False, ["pydiggs is not installed"])`` when the optional package is
    absent, so a caller can tell "the file is wrong" from "nothing checked
    it" -- which a bare False could not.
    """
    from subsurface_characterization.formats.diggs_validation import (
        has_pydiggs, validate_diggs_schema,
    )
    if not has_pydiggs():
        return False, ["pydiggs is not installed, so nothing checked this "
                       "file against the DIGGS schema"]
    result = validate_diggs_schema(content=xml, schema_version=schema_version)
    return bool(result.is_valid), list(result.errors)


# ---------------------------------------------------------------------------
# gate two: does it still say what the log said
# ---------------------------------------------------------------------------

def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def _si_value(q: Optional[Quantity]) -> Optional[float]:
    got = _si(q)
    return None if got is None else got[0]


def _nearest(values: Sequence[Tuple[float, float]], depth: float,
             want: float, depth_tol: float, value_tol: float) -> bool:
    """Is ``want`` among the values standing within ``depth_tol`` of depth?"""
    return any(_close(d, depth, depth_tol) and _close(v, want, value_tol)
               for d, v in values)


def diggs_roundtrip_gate(xml: str, investigations: Any, *,
                         project: Optional[Project] = None,
                         lab_tests: Optional[Sequence[LabTest]] = None
                         ) -> Tuple[bool, List[str]]:
    """``(ok, diffs)`` -- read the file back and compare it, value by value.

    What is checked, for every investigation: its identifier, its coordinates,
    its ground elevation, its total depth; every layer's top, base, USCS
    symbol and description; every sample depth; every driven record's N value
    or drives; every water level; and every index value printed on the log
    face. Depths in metres, stresses in kPa, unit weights in kN/m3, to the
    tolerances in :data:`TOLERANCE`.

    For every LABORATORY test: that the file holds a positioned result at that
    hole and that depth, and that EVERY NUMBER the record's typed result
    carries -- a limit, a fraction, a curve's every point, a specimen's every
    reading -- is in it. The lab check walks the record's own result models
    rather than a second copy of the writer's property table, so it cannot
    agree with the writer by sharing its mistakes: it asks whether the numbers
    survived, not whether they were filed under the names the writer chose.

    A value the writer said it could not write (its notes) is not looked for.
    Everything else that does not come back is a diff.
    """
    from subsurface_characterization import parse_diggs

    lab: List[LabTest] = []
    if isinstance(investigations, ReportRecord):
        lab = list(investigations.lab_tests)
        investigations = list(investigations.investigations)
    elif isinstance(investigations, Investigation):
        investigations = [investigations]
    elif isinstance(investigations, LabTest):
        lab = [investigations]
        investigations = []
    else:
        items = list(investigations)
        lab = [i for i in items if isinstance(i, LabTest)]
        investigations = [i for i in items if isinstance(i, Investigation)]
    if lab_tests is not None:
        lab = list(lab_tests)

    try:
        parsed = parse_diggs(content=xml)
    except Exception as exc:                        # a file that will not open
        return False, [f"parse_diggs could not read the file: "
                       f"{type(exc).__name__}: {exc}"]

    diffs: List[str] = []
    by_id = {inv.investigation_id: inv for inv in parsed.site.investigations}
    dt = TOLERANCE["depth_m"]

    for want in investigations:
        name = want.investigation_id
        got = by_id.get(name)
        if got is None:
            diffs.append(f"{name}: not in the file at all (the file has "
                         f"{sorted(by_id) or 'nothing'})")
            continue
        _compare_header(name, want, got, diffs)
        _compare_layers(name, want, got, diffs, dt)
        _compare_water(name, want, got, diffs, dt)
        _compare_measurements(name, want, got, diffs, dt)
        _compare_soundings(name, want, got, diffs, dt)
    if lab:
        _compare_lab(xml, lab, diffs)
    return (not diffs), diffs


# ---------------------------------------------------------------------------
# the laboratory half of the round trip
# ---------------------------------------------------------------------------

#: How close a number has to come back. The file writes four decimals, or six
#: significant figures below a thousandth, so this absorbs the writing and
#: nothing else.
_VALUE_TOL = 5e-4


#: Fields of a summary row that say WHERE the row is rather than what it
#: says. They are the result's position in the file, checked as a position,
#: and are not looked for again among its values.
_POSITION_FIELDS = ("depth_top", "depth_bottom")



def _numbers_under(element) -> List[float]:
    """Every number written anywhere under one element, positions aside.

    A position is where the result IS, not what it says, so the location is
    skipped: otherwise a depth in the pool could stand in for a value that
    never got written.
    """
    out: List[float] = []
    for node in element.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in ("pos", "posList"):
            continue
        for token in str(node.text or "").replace(",", " ")\
                .replace(";", " ").split():
            try:
                out.append(float(token))
            except ValueError:
                continue
    return out


def _file_tests(xml: str) -> List[Tuple[str, Optional[float], List[float]]]:
    """``(hole name, depth in m, every number in it)`` for each Test."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    gml = "{http://www.opengis.net/gml/3.2}"
    xlink = "{http://www.w3.org/1999/xlink}"
    ns = NS["diggs"]
    d = f"{{{ns}}}"
    names: Dict[str, str] = {}
    for feature in ("Borehole", "TrialPit"):
        for element in root.iter(f"{d}{feature}"):
            label = element.find(f"{gml}name")
            names[element.get(f"{gml}id", "")] = (
                (label.text or "").strip() if label is not None else "")
    out: List[Tuple[str, Optional[float], List[float]]] = []
    for test in root.iter(f"{d}Test"):
        ref = test.find(f"{d}samplingFeatureRef")
        hole = ""
        if ref is not None:
            hole = names.get(
                (ref.get(f"{xlink}href", "") or "").lstrip("#"), "")
        depth: Optional[float] = None
        position = test.find(f".//{d}location//{gml}posList")
        if position is None:
            position = test.find(f".//{d}location//{gml}pos")
        if position is not None:
            tokens = str(position.text or "").split()
            if tokens:
                try:
                    depth = float(tokens[0])
                except ValueError:
                    depth = None
        out.append((hole, depth, _numbers_under(test)))
    return out


def _lab_targets(test: LabTest) -> List[Tuple[str, str, Optional[Quantity],
                                              Any]]:
    """What to look for: one entry per positioned result this test becomes.

    One for an ordinary test. One PER ROW for a summary table, because its
    rows sit in different holes at different depths and are written as
    different results.
    """
    result = test.result
    if isinstance(result, SummaryTableResult):
        return [(f"{test.kind} row {i + 1}", row.investigation_id,
                 row.depth_top, row)
                for i, row in enumerate(result.rows)]
    return [(test.kind, test.investigation_id, test.depth_top, result)]


def _found(pool: Sequence[float], value: float, unit: str) -> bool:
    """Is this value in the pool, in the unit the file would have used?

    Everything is written in SI -- with one stated exception. A particle size
    in metres is 7.5e-05 for a No. 200 sieve, which is the wrong unit for a
    person and a hair from the precision the file writes at, so every grading
    is written in millimetres and says ``mm`` in its uom. A LENGTH therefore
    counts as found at either scale, and nothing else does.
    """
    wanted = [value] if unit != "m" else [value, value * 1000.0]
    return any(_close(candidate, seen, max(_VALUE_TOL, abs(candidate) * 1e-6))
               for candidate in wanted for seen in pool)


def _compare_lab(xml: str, lab: Sequence[LabTest],
                 diffs: List[str]) -> None:
    """Every laboratory value, against what the file gives back."""
    in_file = _file_tests(xml)
    dt = TOLERANCE["depth_m"]
    for test in lab:
        for label, hole, depth, payload in _lab_targets(test):
            hole = (hole or "").strip()
            want_depth = _si_value(depth)
            if not hole or want_depth is None:
                continue     # the writer said why; nothing to look for
            pool: List[float] = []
            found = False
            for name, at, numbers in in_file:
                if name != hole or at is None:
                    continue
                if abs(at - want_depth) > dt:
                    continue
                found = True
                pool.extend(numbers)
            wanted = si_numbers(payload, skip=_POSITION_FIELDS)
            if not found:
                if not wanted:
                    continue    # nothing was expected, so nothing is missing
                diffs.append(
                    f"{label} ({hole} at {want_depth:.3f} m): no result at "
                    f"that hole and depth came back")
                continue
            for path, value, unit in wanted:
                if not _found(pool, value, unit):
                    diffs.append(
                        f"{label} ({hole} at {want_depth:.3f} m): "
                        f"{path} = {value:.6g} did not come back")


def _compare_header(name: str, want: Investigation, got: Any,
                    diffs: List[str]) -> None:
    dt = TOLERANCE["depth_m"]
    if want.x is not None and not _close(got.x, want.x, 1e-6):
        diffs.append(f"{name}: x came back {got.x} not {want.x}")
    if want.y is not None and not _close(got.y, want.y, 1e-6):
        diffs.append(f"{name}: y came back {got.y} not {want.y}")
    elevation = _si_value(want.elevation)
    if elevation is not None and not _close(got.elevation_m, elevation, dt):
        diffs.append(f"{name}: ground elevation came back "
                     f"{got.elevation_m} not {elevation:.4f} m")
    total = _si_value(want.total_depth)
    if total is not None and not _close(got.total_depth_m, total, dt):
        diffs.append(f"{name}: total depth came back {got.total_depth_m} "
                     f"not {total:.4f} m")


def _compare_layers(name: str, want: Investigation, got: Any,
                    diffs: List[str], dt: float) -> None:
    # Only layers with BOTH a top and a base are expected back. A layer whose
    # base the sheet never printed is written as a contact, and the app's
    # LithologyInterval holds intervals only -- the writer's notes say so. A
    # base that is not below its top is not an interval either.
    written = [ly for ly in want.layers
               if _si_value(ly.top) is not None
               and _si_value(ly.bottom) is not None
               and _si_value(ly.bottom) > _si_value(ly.top)]
    if len(got.lithology) != len(written):
        diffs.append(f"{name}: {len(got.lithology)} layer(s) came back, "
                     f"{len(written)} full interval(s) went in")
    for layer, back in zip(written, got.lithology):
        top = _si_value(layer.top)
        if not _close(back.top_depth_m, top, dt):
            diffs.append(f"{name}: layer top came back {back.top_depth_m} "
                         f"not {top:.4f} m")
        bottom = _si_value(layer.bottom)
        if bottom is not None and not _close(back.bottom_depth_m, bottom, dt):
            diffs.append(f"{name}: layer base came back "
                         f"{back.bottom_depth_m} not {bottom:.4f} m")
        if layer.uscs.strip() and back.uscs.strip() != layer.uscs.strip():
            diffs.append(f"{name}: layer at {top:.2f} m came back with USCS "
                         f"{back.uscs!r} not {layer.uscs!r}")
        if layer.description.strip() and not back.description.strip():
            diffs.append(f"{name}: layer at {top:.2f} m came back with no "
                         f"description")


def _compare_water(name: str, want: Investigation, got: Any,
                   diffs: List[str], dt: float) -> None:
    depths = [_si_value(w.depth) for w in want.water]
    depths = [d for d in depths if d is not None]
    if not depths:
        return
    # parse_diggs keeps ONE water level per investigation. The first reading
    # is the one written as the initial strike, so that is the one compared;
    # the rest are in the file and are read by a consumer that wants them.
    if got.gwl_depth_m is None:
        diffs.append(f"{name}: water level {depths[0]:.4f} m came back as "
                     f"nothing")
    elif not _close(got.gwl_depth_m, depths[0], dt):
        diffs.append(f"{name}: water level came back {got.gwl_depth_m} not "
                     f"{depths[0]:.4f} m")


#: What each channel of a sounding must come back as, and how close. The
#: parameter names are the ones ``parse_diggs`` puts on a PointMeasurement;
#: the tolerances absorb the four decimals the file is written to and
#: nothing else.
_CPT_CHANNELS: Tuple[Tuple[str, str, str, float], ...] = (
    ("qc", "qc_kPa", "kPa", TOLERANCE["stress_kPa"]),
    ("fs", "fs_kPa", "kPa", TOLERANCE["stress_kPa"]),
    ("u2", "u2_kPa", "kPa", TOLERANCE["stress_kPa"]),
)
_DCP_CHANNELS: Tuple[Tuple[str, str, str, float], ...] = (
    ("blows", "blow_count", "", TOLERANCE["n_value"]),
    ("penetration", "penetration_m", "m", TOLERANCE["depth_m"]),
    ("index", "DPI", "", 0.05),
    ("cbr_percent", "CBR_pct", "", TOLERANCE["percent"]),
)


def _compare_soundings(name: str, want: Investigation, got: Any,
                       diffs: List[str], dt: float) -> None:
    """Every reading of a cone sounding or a dynamic probe, value for value.

    A sounding's whole point is the series, so every point of it is checked
    at its own depth -- not a count, not a spot check. A file that validates
    and comes back with half a sounding is the failure this catches.
    """
    data = want.cpt or want.dcp
    if data is None or not data.points:
        return
    channels = _CPT_CHANNELS if want.cpt is not None else _DCP_CHANNELS
    by_parameter: Dict[str, List[Tuple[float, float]]] = {}
    for m in got.measurements:
        by_parameter.setdefault(m.parameter, []).append((m.depth_m, m.value))
    missing = 0
    first = ""
    for point in data.points:
        depth = _si_value(point.depth)
        if depth is None:
            continue                  # the writer's notes say it was skipped
        for field_name, parameter, unit, tol in channels:
            raw = getattr(point, field_name, None)
            if raw is None:
                continue
            # A channel is a Quantity on some kinds and a plain number on
            # others -- a blow count has no unit, a penetration index is
            # printed in whatever the sheet chose. Both are compared in SI,
            # which is what the file holds.
            value = _si_value(raw) if isinstance(raw, Quantity) \
                else float(raw)
            if value is None:
                continue              # the writer said it could not convert
            if not _nearest(by_parameter.get(parameter, []), depth, value,
                            dt, tol):
                missing += 1
                if not first:
                    first = (f"{field_name} {value:.4g} at {depth:.3f} m did "
                             f"not come back as {parameter}")
    if missing:
        diffs.append(
            f"{name}: {missing} sounding reading(s) did not come back; the "
            f"first is {first}")


def _compare_measurements(name: str, want: Investigation, got: Any,
                          diffs: List[str], dt: float) -> None:
    """Every N value, drive and index value, against what came back."""
    by_parameter: Dict[str, List[Tuple[float, float]]] = {}
    for m in got.measurements:
        by_parameter.setdefault(m.parameter, []).append((m.depth_m, m.value))

    for record in want.spt:
        top = _si_value(record.depth_top)
        if top is None:
            continue
        if record.n is not None:
            if not _nearest(by_parameter.get("N_spt", []), top,
                            float(record.n), dt, TOLERANCE["n_value"]):
                diffs.append(f"{name}: N={record.n} at {top:.3f} m did not "
                             f"come back")
        else:
            drives = _drive_sets(record)
            back = by_parameter.get("blow_count", [])
            for count, _pen in drives:
                if not _nearest(back, top, float(count), dt,
                                TOLERANCE["n_value"]):
                    diffs.append(f"{name}: drive {count} at {top:.3f} m did "
                                 f"not come back")

    checks = (
        ("recovery_percent", "recovery_pct", None, TOLERANCE["percent"]),
        ("rqd_percent", "RQD_pct", None, TOLERANCE["percent"]),
        ("water_content", "wn_pct", None, TOLERANCE["percent"]),
        ("liquid_limit", "LL_pct", None, TOLERANCE["percent"]),
        ("plastic_limit", "PL_pct", None, TOLERANCE["percent"]),
        ("plasticity_index", "PI_pct", None, TOLERANCE["percent"]),
        ("fines_percent", "pct_fines", None, TOLERANCE["percent"]),
        ("dry_unit_weight", "gamma_d_kNm3", "kN/m3",
         TOLERANCE["unit_weight_kNm3"]),
        ("qu", "qu_kPa", "kPa", TOLERANCE["stress_kPa"]),
        ("pocket_pen", "qu_kPa", "kPa", TOLERANCE["stress_kPa"]),
    )
    for sample in want.samples:
        top = _si_value(sample.top)
        if top is None:
            continue
        for field, parameter, unit, tol in checks:
            raw = getattr(sample, field)
            if raw is None:
                continue
            value = _si_value(raw) if unit else float(raw)
            if value is None:
                continue                       # the writer said it skipped it
            if not _nearest(by_parameter.get(parameter, []), top, value, dt,
                            tol):
                diffs.append(
                    f"{name}: {field} {value:.4g} at {top:.3f} m did not come "
                    f"back as {parameter}")
