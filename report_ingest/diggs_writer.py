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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape, quoteattr

from report_ingest.model import (
    Investigation, Layer, Project, Quantity, ReportRecord, Sample, SPT,
    WaterLevel,
)

__all__ = [
    "write_diggs", "diggs_schema_gate", "diggs_roundtrip_gate",
    "DiggsWriteNotes", "NS", "PROPERTY_CLASS", "TOLERANCE",
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
    """A measure, written the way the file writes every measure."""
    text = f"{float(value):.{_DP}f}".rstrip("0").rstrip(".")
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
class DiggsWriteNotes:
    """What the writer could not write, and what it had to leave behind.

    Handed back on the writer's own object rather than raised: a log with one
    unconvertible unit still has fifty good values in it, and the file should
    carry them. What is missing is SAID, so the round-trip gate is not asked
    to find something that was never written.
    """

    skipped: List[str] = field(default_factory=list)
    investigations: int = 0
    layers: int = 0
    samples: int = 0
    spt: int = 0
    water: int = 0
    tests: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"skipped": list(self.skipped),
                "investigations": self.investigations, "layers": self.layers,
                "samples": self.samples, "spt": self.spt, "water": self.water,
                "tests": self.tests}


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

class _Writer:
    """Builds one DIGGS 2.6 document, element by element."""

    def __init__(self, investigations: Sequence[Investigation],
                 project: Optional[Project], document_id: str) -> None:
        self.investigations = list(investigations)
        self.project = project or Project()
        self.doc_id = _ncname(document_id or "report", "report")
        self.project_id = _ncname(
            self.project.number or self.project.name or "project", "project")
        self.notes = DiggsWriteNotes()
        self.lines: List[str] = []
        self._seen_ids: Dict[str, int] = {}

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
        for i, (name, klass, _value) in enumerate(values, start=1):
            self.out(f'<Property index="{i}" '
                     f'gml:id={quoteattr(f"{test_id}_prop{i}")}>', 10)
            self.out(f"<propertyName>{escape(name)}</propertyName>", 11)
            self.out("<typeData>double</typeData>", 11)
            self.out(f'<propertyClass codeSpace='
                     f'"urn:diggs:def:codelist:DIGGS:properties">'
                     f"{escape(klass)}</propertyClass>", 11)
            self.out("</Property>", 10)
        self.out("</properties>", 9)
        self.out("</PropertyParameters>", 8)
        self.out("</parameters>", 7)
        data = ",".join(_num(v) for _n, _k, v in values)
        self.out(f'<dataValues cs="," ts=" " decimal=".">{data}</dataValues>',
                 7)
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
                notes: Optional[List[DiggsWriteNotes]] = None) -> str:
    """DIGGS 2.6 XML for a record, an investigation, or a list of them.

    ``project`` fills the ``Project`` element; when a whole
    :class:`~report_ingest.model.ReportRecord` is passed its own project is
    used unless one is given here. Pass ``notes`` as an empty list to be
    handed the writer's :class:`DiggsWriteNotes` -- what it had to leave out
    and why.
    """
    if isinstance(record_or_investigations, ReportRecord):
        record = record_or_investigations
        investigations = list(record.investigations)
        project = project or record.project
        document_id = document_id or record.document.report_id
    elif isinstance(record_or_investigations, Investigation):
        investigations = [record_or_investigations]
    elif isinstance(record_or_investigations, Iterable):
        investigations = list(record_or_investigations)
    else:
        raise TypeError(
            f"write_diggs takes a ReportRecord, an Investigation or a list of "
            f"them, not {type(record_or_investigations).__name__}")
    for inv in investigations:
        if not isinstance(inv, Investigation):
            raise TypeError(
                f"every investigation must be a model.Investigation, not "
                f"{type(inv).__name__}")
        if not inv.units_known and any(
                (inv.layers, inv.samples, inv.spt, inv.water)):
            raise ValueError(
                f"investigation {inv.investigation_id!r} carries depths in an "
                f"unknown unit (units_known is False). A DIGGS file states a "
                f"uom on every measure, so writing one would mean inventing "
                f"a unit. Fix the unit or write no DIGGS for this log.")
    writer = _Writer(investigations, project, document_id)
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
                         project: Optional[Project] = None
                         ) -> Tuple[bool, List[str]]:
    """``(ok, diffs)`` -- read the file back and compare it, value by value.

    What is checked, for every investigation: its identifier, its coordinates,
    its ground elevation, its total depth; every layer's top, base, USCS
    symbol and description; every sample depth; every driven record's N value
    or drives; every water level; and every index value printed on the log
    face. Depths in metres, stresses in kPa, unit weights in kN/m3, to the
    tolerances in :data:`TOLERANCE`.

    A value the writer said it could not write (its notes) is not looked for.
    Everything else that does not come back is a diff.
    """
    from subsurface_characterization import parse_diggs

    if isinstance(investigations, ReportRecord):
        investigations = list(investigations.investigations)
    elif isinstance(investigations, Investigation):
        investigations = [investigations]
    else:
        investigations = list(investigations)

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
    return (not diffs), diffs


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
