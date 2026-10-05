"""The DIGGS 2.6 writer and its two gates, on a synthetic investigation.

The synthetic log here carries one of everything the writer knows how to
write, so the same path the fifteen hand-truthed logs take is walked in CI
where the private corpus is not. The truth-driven gate lives beside the
harness, in ``module_work/report_ingest_harness/tests/test_diggs_truth.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from report_ingest.diggs_writer import (
    NS, PROPERTY_CLASS, DiggsWriteNotes, diggs_roundtrip_gate,
    diggs_schema_gate, write_diggs,
)
from report_ingest.model import (
    DrillingDetails, Investigation, Layer, Project, Quantity, ReportRecord,
    SPT, Sample, WaterLevel,
)


def _ft(value: float) -> Quantity:
    return Quantity(value=value, unit="ft")


@pytest.fixture
def boring() -> Investigation:
    """One imperial boring with one of everything."""
    return Investigation(
        investigation_id="B-2", kind="boring", depth_unit="ft",
        x=-117.88735, y=33.75754, coordinate_system="WGS 84",
        elevation=_ft(104.5), total_depth=_ft(21.5),
        date_started="1/23/2015", date_finished="1/23/2015",
        drilling=DrillingDetails(method="Hollow Stem Auger", equipment="B-61",
                                 hammer_type="Automatic SPT Hammer",
                                 hammer_energy_ratio=82.0,
                                 driller="Jet Drilling"),
        layers=[
            Layer(top=_ft(0.0), bottom=_ft(5.0), uscs="CL",
                  description="SANDY LEAN CLAY (CL), dark brown, stiff",
                  color="dark brown", consistency="stiff", moisture="moist"),
            Layer(top=_ft(5.0), bottom=_ft(21.5), uscs="SM",
                  description="SILTY SAND (SM), brown, medium dense"),
        ],
        samples=[
            Sample(sample_id="1", top=_ft(2.5), bottom=_ft(4.0), kind="ring",
                   water_content=17.0,
                   dry_unit_weight=Quantity(value=108.0, unit="pcf"),
                   liquid_limit=42.0, plastic_limit=21.0,
                   plasticity_index=21.0, fines_percent=52.0,
                   qu=Quantity(value=2.0, unit="tsf"),
                   pocket_pen=Quantity(value=1.5, unit="tsf"),
                   recovery=Quantity(value=12.0, unit="in"), uscs="CL"),
            Sample(sample_id="4", top=_ft(10.0), bottom=_ft(11.5),
                   kind="spt"),
        ],
        spt=[
            SPT(depth_top=_ft(10.0), depth_bottom=_ft(11.5), blows=[2, 3, 4],
                n=7, sample_id="4"),
            SPT(depth_top=_ft(20.0), blows=[12, 30, '50/5"'], refusal=True),
        ],
        water=[
            WaterLevel(depth=_ft(12.0), when="while_drilling"),
            WaterLevel(depth=_ft(9.0), when="after_hours", hours=24.0),
        ],
        remarks="Boring backfilled with cuttings.",
        fields={"bit": "4-inch tricone"},
        pages=[38], source_report="RXX", sheet="1 of 1")


@pytest.fixture
def xml(boring) -> str:
    return write_diggs([boring],
                       Project(name="A project", number="12345"),
                       document_id="RXX")


def _root(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def _all(root: ET.Element, path: str):
    return root.findall(path, NS)


# ---------------------------------------------------------------------------
# gate one
# ---------------------------------------------------------------------------

class TestSchemaGate:
    def test_the_file_validates_against_the_bundled_two_six_schema(self, xml):
        ok, errors = diggs_schema_gate(xml)
        assert ok, errors[:3]
        assert errors == []

    def test_a_record_with_several_logs_validates(self, boring):
        second = boring.model_copy(deep=True)
        second.investigation_id = "TP-4"
        second.kind = "test_pit"
        record = ReportRecord(investigations=[boring, second])
        record.project = Project(name="A project")
        ok, errors = diggs_schema_gate(write_diggs(record))
        assert ok, errors[:3]

    def test_a_broken_file_fails_the_gate(self):
        ok, errors = diggs_schema_gate("<Diggs/>")
        assert ok is False and errors

    def test_an_identifier_that_cannot_be_an_xml_id_still_validates(self):
        inv = Investigation(investigation_id="1 / B (offset)",
                            depth_unit="m",
                            layers=[Layer(top=Quantity(value=0.0, unit="m"),
                                          bottom=Quantity(value=2.0,
                                                          unit="m"),
                                          description="FILL")])
        ok, errors = diggs_schema_gate(write_diggs([inv]))
        assert ok, errors[:3]

    def test_two_logs_with_the_same_name_do_not_collide(self):
        one = Investigation(investigation_id="B-1", depth_unit="m",
                            total_depth=Quantity(value=5.0, unit="m"))
        two = Investigation(investigation_id="B-1", depth_unit="m",
                            total_depth=Quantity(value=8.0, unit="m"))
        text = write_diggs([one, two])
        ok, errors = diggs_schema_gate(text)
        assert ok, errors[:3]
        ids = [e.get(f'{{{NS["gml"]}}}id')
               for e in _all(_root(text), ".//diggs:Borehole")]
        assert len(ids) == len(set(ids)) == 2


# ---------------------------------------------------------------------------
# gate two
# ---------------------------------------------------------------------------

class TestRoundTripGate:
    def test_everything_written_comes_back(self, xml, boring):
        ok, diffs = diggs_roundtrip_gate(xml, [boring])
        assert ok, diffs

    def test_the_gate_takes_a_record_as_well_as_a_list(self, boring):
        record = ReportRecord(investigations=[boring],
                              project=Project(name="A project"))
        ok, diffs = diggs_roundtrip_gate(write_diggs(record), record)
        assert ok, diffs

    def test_the_gate_notices_a_value_that_did_not_come_back(self, xml,
                                                             boring):
        # Ask the gate for an N value the file was never given. This is the
        # failure the gate exists for: the file is still XSD-valid.
        wrong = boring.model_copy(deep=True)
        wrong.spt.append(SPT(depth_top=_ft(15.0), blows=[9], n=99))
        ok, diffs = diggs_roundtrip_gate(xml, [wrong])
        assert ok is False
        assert any("N=99" in d for d in diffs)
        assert diggs_schema_gate(xml)[0] is True

    def test_the_gate_notices_a_missing_investigation(self, xml, boring):
        other = boring.model_copy(deep=True)
        other.investigation_id = "B-9"
        ok, diffs = diggs_roundtrip_gate(xml, [other])
        assert ok is False
        assert any("not in the file at all" in d for d in diffs)

    def test_the_gate_reports_a_file_it_cannot_read(self, boring):
        ok, diffs = diggs_roundtrip_gate("not xml at all", [boring])
        assert ok is False and "could not read" in diffs[0]

    def test_a_test_pit_round_trips(self):
        pit = Investigation(
            investigation_id="TP-4", kind="test_pit", depth_unit="m",
            total_depth=Quantity(value=3.5, unit="m"),
            layers=[Layer(top=Quantity(value=0.0, unit="m"),
                          bottom=Quantity(value=1.2, unit="m"),
                          description="FILL", uscs="SM")],
            samples=[Sample(sample_id="B1",
                            top=Quantity(value=1.0, unit="m"), kind="bulk",
                            water_content=22.0)])
        text = write_diggs([pit])
        assert "<TrialPit" in text
        assert diggs_schema_gate(text)[0] is True
        ok, diffs = diggs_roundtrip_gate(text, [pit])
        assert ok, diffs


# ---------------------------------------------------------------------------
# the element map
# ---------------------------------------------------------------------------

class TestWhatItWrites:
    def test_the_procedures_are_in_the_geotechnical_namespace(self, xml):
        root = _root(xml)
        for procedure in ("DrivenPenetrationTest", "WaterContentTest",
                          "LabDensityTest", "AtterbergLimitsTest",
                          "ParticleSizeTest",
                          "UnconfinedCompressiveStrengthTest",
                          "PocketPenetrometerTest"):
            assert _all(root, f".//diggs_geo:{procedure}"), procedure

    def test_every_property_class_is_one_the_dictionary_publishes(self):
        published = _dictionary_ids()
        for field, (klass, _uom, _proc) in PROPERTY_CLASS.items():
            assert klass in published, f"{field} -> {klass}"

    def test_the_value_lives_in_the_result_set_not_the_procedure(self, xml):
        root = _root(xml)
        classes = [e.text for e in _all(root, ".//diggs:propertyClass")]
        assert "n_value" in classes
        assert "water_content_natural" in classes
        assert "dry_density" in classes
        assert {"liquid_limit", "plastic_limit",
                "plasticity_index"} <= set(classes)
        assert "percent_fines" in classes
        assert "compressive_strength_unconfined" in classes

    def test_a_depth_is_a_position_on_the_holes_own_reference_system(self,
                                                                    xml):
        root = _root(xml)
        assert _all(root, ".//diggs:LinearSpatialReferenceSystem")
        extents = _all(root, ".//diggs:LinearExtent")
        referenced = [e for e in extents
                      if (e.get("srsName") or "").endswith("_lsr")]
        assert referenced, "no depth was written against the hole's system"

    def test_every_measure_carries_a_uom(self, xml):
        root = _root(xml)
        for tag in ("totalMeasuredDepth", "totalSampleRecoveryLength"):
            for element in _all(root, f".//diggs:{tag}"):
                assert element.get("uom"), tag
        for element in _all(root, ".//diggs_geo:penetration"):
            assert element.get("uom")

    def test_si_conversion_happens_once_here(self, xml):
        root = _root(xml)
        depth = _all(root, ".//diggs:totalMeasuredDepth")[0]
        assert depth.get("uom") == "m"
        assert float(depth.text) == pytest.approx(21.5 * 0.3048, abs=1e-4)

    def test_the_hammer_is_written_because_an_n_value_needs_it(self, xml):
        root = _root(xml)
        hammers = [e.text for e in _all(root, ".//diggs_geo:hammerType")]
        assert "Automatic SPT Hammer" in hammers
        efficiency = _all(root, ".//diggs_geo:hammerEfficiency")
        assert efficiency and efficiency[0].get("uom") == "%"

    def test_a_refusal_becomes_a_drive_of_fifty_over_the_distance_it_went(
            self, xml):
        root = _root(xml)
        counts = [int(e.text) for e in _all(root, ".//diggs_geo:blowCount")]
        assert 50 in counts
        # 50/5" is fifty blows for five inches, which is 0.127 m.
        penetrations = [float(e.text)
                        for e in _all(root, ".//diggs_geo:penetration")]
        assert any(abs(p - 0.127) < 1e-4 for p in penetrations)

    def test_drives_are_written_when_the_log_printed_no_n_value(self):
        inv = Investigation(
            investigation_id="B-3", depth_unit="ft",
            spt=[SPT(depth_top=_ft(5.0), blows=[6, 9, 14])])
        text = write_diggs([inv])
        assert diggs_schema_gate(text)[0] is True
        classes = [e.text for e in _all(_root(text), ".//diggs:propertyClass")]
        assert classes == ["blow_count"] * 3
        ok, diffs = diggs_roundtrip_gate(text, [inv])
        assert ok, diffs

    def test_water_goes_in_the_holes_own_water_strike(self, xml):
        root = _root(xml)
        strikes = _all(root, ".//diggs:WaterStrike")
        assert len(strikes) == 1
        assert _all(root, ".//diggs:initialWaterStrikeReading")
        assert _all(root, ".//diggs:postStrikeReading")

    def test_a_log_that_found_no_water_says_so(self):
        inv = Investigation(investigation_id="B-4", depth_unit="ft",
                            water=[WaterLevel(when="not_encountered")])
        text = write_diggs([inv])
        assert "<notEncountered>true</notEncountered>" in text
        assert diggs_schema_gate(text)[0] is True

    def test_the_header_fields_the_model_has_no_slot_for_are_carried(self,
                                                                    xml):
        assert "4-inch tricone" in xml

    def test_a_coordinate_keeps_the_decimals_a_latitude_needs(self, xml):
        assert "33.75754" in xml and "-117.88735" in xml


# ---------------------------------------------------------------------------
# what it refuses, and what it says it left out
# ---------------------------------------------------------------------------

class TestWhatItRefuses:
    def test_depths_in_an_unknown_unit_are_refused_outright(self):
        inv = Investigation(
            investigation_id="B-5", depth_unit="", units_known=False,
            layers=[Layer(top=Quantity(value=4.0, unit=""),
                          description="SAND")])
        with pytest.raises(ValueError, match="units_known is False"):
            write_diggs([inv])

    def test_a_log_with_no_depths_at_all_still_writes_its_header(self):
        inv = Investigation(investigation_id="B-6", depth_unit="",
                            units_known=False,
                            drilling=DrillingDetails(method="Hand auger"))
        text = write_diggs([inv])
        assert diggs_schema_gate(text)[0] is True
        assert "B-6" in text

    def test_an_unconvertible_index_unit_is_left_out_and_named(self):
        inv = Investigation(
            investigation_id="B-7", depth_unit="ft",
            samples=[Sample(sample_id="1", top=_ft(3.0),
                            qu=Quantity(value=2.0, unit="smoots"))])
        notes = []
        text = write_diggs([inv], notes=notes)
        assert diggs_schema_gate(text)[0] is True
        assert any("smoots" in s for s in notes[0].skipped)
        # ... and the gate does not then ask for it back
        ok, diffs = diggs_roundtrip_gate(text, [inv])
        assert ok, diffs

    def test_a_layer_with_no_printed_base_is_written_and_named(self):
        inv = Investigation(
            investigation_id="B-8", depth_unit="ft",
            layers=[Layer(top=_ft(16.0), description="LEAN CLAY (CL)",
                          uscs="CL")])
        notes = []
        text = write_diggs([inv], notes=notes)
        assert diggs_schema_gate(text)[0] is True
        assert any("no printed base" in s for s in notes[0].skipped)
        ok, diffs = diggs_roundtrip_gate(text, [inv])
        assert ok, diffs

    def test_the_notes_count_what_was_written(self, boring):
        notes = []
        write_diggs([boring], notes=notes)
        assert notes[0].investigations == 1
        assert notes[0].layers == 2
        assert notes[0].samples == 2
        assert notes[0].spt == 2
        assert notes[0].water == 2

    def test_something_that_is_not_an_investigation_is_refused(self):
        with pytest.raises(TypeError):
            write_diggs([{"investigation_id": "B-1"}])
        with pytest.raises(TypeError):
            write_diggs(42)


def _dictionary_ids():
    """Every propertyClass pydiggs' own dictionary publishes."""
    import os
    import re
    pydiggs = pytest.importorskip("pydiggs")
    path = os.path.join(os.path.dirname(pydiggs.__file__), "dictionaries",
                        "properties.xml")
    with open(path, encoding="utf-8", errors="replace") as fh:
        return set(re.findall(r'gml:id="([^"]+)"', fh.read()))


class TestRecoveryAndRqd:
    """The two numbers a rock-core log prints most often.

    DIGGS carries recovery as a LENGTH and has no element for the percentage
    a log actually prints, and rock quality designation belongs to the
    sampling ACTIVITY rather than to any test. Before both were written they
    were dropped silently, which is the failure the record exists to prevent.
    """

    @pytest.fixture
    def cored(self) -> Investigation:
        return Investigation(
            investigation_id="B-10", depth_unit="ft",
            samples=[Sample(sample_id="R-1", top=_ft(20.0),
                            bottom=_ft(25.0), kind="core",
                            recovery_percent=88.0, rqd_percent=62.0)])

    def test_both_are_written_where_the_schema_puts_them(self, cored):
        text = write_diggs([cored])
        ok, errors = diggs_schema_gate(text)
        assert ok, errors[:3]
        root = _root(text)
        rqd = _all(root, ".//diggs:samplingActivityRQD")
        assert rqd and rqd[0].get("uom") == "%"
        assert float(rqd[0].text) == pytest.approx(62.0)
        names = [e.text for e in _all(root, ".//diggs:parameterName")]
        assert "recovery_percent" in names

    def test_both_come_back(self, cored):
        text = write_diggs([cored])
        ok, diffs = diggs_roundtrip_gate(text, [cored])
        assert ok, diffs

    def test_the_gate_notices_when_they_do_not(self, cored):
        text = write_diggs([cored])
        wrong = cored.model_copy(deep=True)
        wrong.samples[0].rqd_percent = 30.0
        ok, diffs = diggs_roundtrip_gate(text, [wrong])
        assert ok is False
        assert any("rqd_percent" in d for d in diffs)


# ---------------------------------------------------------------------------
# the soundings
# ---------------------------------------------------------------------------

class TestTheSoundings:
    """A cone sounding and a dynamic probe through both gates.

    DIGGS 2.6 has a real home for a cone sounding
    (``diggs_geo:StaticConePenetrationTest``) and none at all for something
    called a DynamicConePenetrometerTest -- the name is nowhere in the
    published schema -- so a dynamic probe is written as the
    ``diggs_geo:DynamicProbeTest`` the schema DOES declare, whose own
    documentation is "all methods that involve driving a rod by impact
    hammer".
    """

    @staticmethod
    def _cone(**kwargs):
        from report_ingest.model import CPTData, CPTPoint
        points = [CPTPoint(depth=Quantity(value=d, unit="m"),
                           qc=Quantity(value=5.0 + d, unit="MPa"),
                           fs=Quantity(value=0.05 * d + 0.01, unit="MPa"),
                           u2=Quantity(value=10.0 * d, unit="kPa"),
                           rf_percent=1.2)
                  for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)]
        return Investigation(
            investigation_id="CPT-1", kind="cpt", depth_unit="m",
            elevation=Quantity(value=12.5, unit="m"),
            total_depth=Quantity(value=3.0, unit="m"),
            cpt=CPTData(points=points, cone_type="piezocone S15",
                        standard="EN ISO 22476-1", qc_unit="MPa",
                        fs_unit="MPa", u2_unit="kPa",
                        step=Quantity(value=0.5, unit="m"), **kwargs))

    @staticmethod
    def _probe(**kwargs):
        from report_ingest.model import DCPData, DCPPoint
        points = [DCPPoint(depth=Quantity(value=d, unit="m"),
                           blows=float(int(10 + 10 * d)),
                           index=Quantity(value=7.0 + d, unit="MPa"),
                           cbr_percent=12.0)
                  for d in (0.2, 0.4, 0.6, 0.8, 1.0)]
        return Investigation(
            investigation_id="DPT-1", kind="dcp", depth_unit="m",
            dcp=DCPData(points=points,
                        increment=Quantity(value=0.2, unit="m"),
                        test_type="DPT",
                        hammer_mass=Quantity(value=63.5, unit="kg"),
                        hammer_drop=Quantity(value=0.75, unit="m"),
                        index_name="Rd (MPa)", **kwargs))

    def test_a_cone_sounding_is_one_positioned_result_set(self):
        xml = write_diggs([self._cone()])
        assert xml.count("<Test ") == 1
        assert "diggs_geo:StaticConePenetrationTest" in xml
        assert "tip_resistance" in xml
        assert "sleeve_friction" in xml
        assert "pore_pressure_u2" in xml

    def test_a_dynamic_probe_uses_the_procedure_the_schema_has(self):
        xml = write_diggs([self._probe()])
        assert "diggs_geo:DynamicProbeTest" in xml
        # The name a reader might expect is not in the 2.6 schema at all.
        assert "DynamicConePenetrometerTest" not in xml
        assert "diggs_geo:penetrationTestType" in xml
        assert "diggs_geo:hammerMass" in xml

    def test_the_depth_of_each_reading_is_a_column_of_the_set(self):
        xml = write_diggs([self._cone()])
        # A sounding is one Test with many ROWS, so the depth cannot be the
        # test's own position; it is a column.
        assert "sounding_depth" in xml
        assert 'ts=";"' in xml

    def test_both_soundings_pass_the_schema_gate(self):
        record = ReportRecord(investigations=[self._cone(), self._probe()])
        xml = write_diggs(record)
        ok, errors = diggs_schema_gate(xml)
        assert ok, errors[:3]

    def test_both_soundings_come_back_value_for_value(self):
        record = ReportRecord(investigations=[self._cone(), self._probe()])
        xml = write_diggs(record)
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs[:5]

    def test_the_parser_gives_every_reading_its_own_depth(self):
        from subsurface_characterization import parse_diggs
        xml = write_diggs([self._cone()])
        site = parse_diggs(content=xml).site
        qc = [(m.depth_m, m.value) for m in site.investigations[0].measurements
              if m.parameter == "qc_kPa"]
        assert len(qc) == 6
        assert sorted(d for d, _v in qc) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    def test_a_changed_value_fails_the_round_trip(self):
        # The gate has to be able to FAIL, or it is measuring nothing.
        cone = self._cone()
        record = ReportRecord(investigations=[cone])
        xml = write_diggs(record)
        cone.cpt.points[2].qc = Quantity(value=99.0, unit="MPa")
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok
        assert any("sounding reading" in d for d in diffs)

    def test_the_writers_notes_count_what_it_wrote(self):
        notes = []
        write_diggs(ReportRecord(
            investigations=[self._cone(), self._probe()]), notes=notes)
        assert notes[0].cpt == 1
        assert notes[0].dcp == 1

    def test_a_sounding_whose_unit_cannot_convert_is_said_not_guessed(self):
        from report_ingest.model import DCPData, DCPPoint
        inv = Investigation(
            investigation_id="DPT-2", kind="dcp", depth_unit="m",
            dcp=DCPData(points=[
                DCPPoint(depth=Quantity(value=d, unit="m"),
                         blows=5.0,
                         index=Quantity(value=44.8, unit="daN/cm2"))
                for d in (0.2, 0.4, 0.6)]))
        xml = write_diggs([inv])
        # The blows are written; the unconvertible index is not guessed at.
        assert "blow_count" in xml
        ok, _errors = diggs_schema_gate(xml)
        assert ok


# ---------------------------------------------------------------------------
# what the Foundry run of 2026-10-04 found
# ---------------------------------------------------------------------------

def _m(value: float) -> Quantity:
    return Quantity(value=value, unit="m")


def _log(name: str, top: float, bottom: float, **over) -> Investigation:
    """A metric log with two layers and one sample between top and bottom."""
    middle = round((top + bottom) / 2.0, 2)
    data = dict(
        investigation_id=name, depth_unit="m", total_depth=_m(bottom),
        layers=[Layer(top=_m(top), bottom=_m(middle), description="CLAY",
                      uscs="CL"),
                Layer(top=_m(middle), bottom=_m(bottom), description="SAND",
                      uscs="SP")],
        samples=[Sample(sample_id="S-1", top=_m(top + 0.5),
                        bottom=_m(top + 0.95), kind="spt",
                        water_content=21.0)],
        spt=[SPT(depth_top=_m(top + 0.5), blows=[4, 6, 7], n=13)])
    data.update(over)
    return Investigation(**data)


class TestALogInAnUnknownUnit:
    """One log whose depths are in a unit nobody could read used to cost
    the whole report its DIGGS file (R11, test pit TP-2)."""

    def _record(self):
        lost = Investigation(
            investigation_id="TP-2", kind="test_pit", depth_unit="",
            units_known=False, pages=[44],
            layers=[Layer(top=Quantity(value=0.0, unit=""),
                          bottom=Quantity(value=1.2, unit=""),
                          description="FILL")])
        return ReportRecord(investigations=[_log("B-1", 0.0, 6.0), lost],
                            project=Project(name="A project"))

    def test_the_rest_of_the_report_is_written_and_the_log_is_named(self):
        record = self._record()
        notes = []
        xml = write_diggs(record, notes=notes)
        assert "B-1" in xml and "TP-2" not in xml
        (row,) = notes[0].left_out
        assert row["investigation_id"] == "TP-2" and row["pages"] == [44]
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs

    def test_the_record_gets_a_qa_entry_naming_it(self, tmp_path):
        from report_ingest.writers import write_outputs
        record = self._record()
        written = write_outputs(record, tmp_path)
        assert written.diggs and written.roundtrip_ok is True
        (entry,) = [e for e in record.qa
                    if e.where == "diggs.investigations[TP-2]"]
        assert entry.kind == "skipped" and "TP-2" in entry.detail
        assert entry.pages == [44]
        # A plain "diggs" entry is the "no file at all" verdict; this
        # report HAS a file.
        assert not [e for e in record.qa if e.where == "diggs"]

    def test_a_lab_test_on_that_log_still_has_a_hole_to_hang_off(self):
        from report_ingest.model import AtterbergResult, LabTest
        record = self._record()
        record.lab_tests = [LabTest(kind="atterberg", investigation_id="TP-2",
                                    depth_top=_m(0.6),
                                    result=AtterbergResult(ll=40, pl=20,
                                                           pi=20))]
        notes = []
        xml = write_diggs(record, notes=notes)
        assert any("left out" in s for s in notes[0].synthesised)
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs


class TestAnUnnamedLog:
    """A log that printed no identifier is named in the file by its id, and
    the gate used to look it up by its empty name (R09, R24, R29)."""

    def test_two_unnamed_logs_round_trip_and_keep_their_empty_ids(self):
        record = ReportRecord(investigations=[_log("", 0.0, 6.0),
                                              _log("", 0.0, 9.0)],
                              project=Project(name="A project"))
        xml = write_diggs(record)
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs
        assert [inv.investigation_id for inv in record.investigations] \
            == ["", ""]

    def test_the_gate_still_catches_a_wrong_value_on_one(self):
        record = ReportRecord(investigations=[_log("", 0.0, 6.0),
                                              _log("", 0.0, 9.0)])
        xml = write_diggs(record)
        record.investigations[1].spt[0].n = 99
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok and any("N=99" in d for d in diffs)


class TestTwoLogsWithOneName:
    """Two logs printing the same identifier -- a continuation sheet read as
    a log of its own -- are two features with one name; the app's reader
    merges them by name, and the gate compared each log with the merge
    (R09 '2', R16 '3': "8 layer(s) came back, 7 went in")."""

    def _record(self):
        return ReportRecord(investigations=[
            _log("3", 0.0, 11.0), _log("3", 10.5, 15.1)],
            project=Project(name="A project"))

    def test_each_is_compared_with_its_own_feature(self):
        record = self._record()
        notes = []
        xml = write_diggs(record, notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs
        assert notes[0].shared_names == ["3"]

    def test_a_wrong_value_on_the_second_is_still_caught(self):
        record = self._record()
        xml = write_diggs(record)
        record.investigations[1].layers[0].top = _m(11.0)
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok and any("layer top" in d for d in diffs)

    def test_a_shared_name_is_a_note_in_the_qa(self, tmp_path):
        from report_ingest.writers import write_outputs
        record = self._record()
        written = write_outputs(record, tmp_path)
        assert written.roundtrip_ok is True
        (entry,) = [e for e in record.qa if e.where == "diggs.shared_names"]
        assert entry.kind == "note" and entry.values == ["3"]

    def test_a_literal_suffix_cannot_collide_with_a_planned_one(self):
        record = ReportRecord(investigations=[
            _log("B-1", 0.0, 5.0), _log("B-1", 0.0, 6.0),
            _log("B-1_2", 0.0, 7.0)])
        xml = write_diggs(record)
        ids = [e.get(f'{{{NS["gml"]}}}id')
               for e in _all(_root(xml), ".//diggs:Borehole")]
        assert len(ids) == len(set(ids)) == 3
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs

    def test_a_gate_handed_one_log_of_several_finds_it_by_name(self):
        record = ReportRecord(investigations=[_log("B-1", 0.0, 5.0),
                                              _log("B-2", 0.0, 6.0)])
        xml = write_diggs(record)
        ok, diffs = diggs_roundtrip_gate(xml, [record.investigations[1]])
        assert ok, diffs


class TestWaterInAPit:
    """A TrialPit has no waterStrike in 2.6, so a pit's water level was
    written nowhere and "came back as nothing" (R13, R28, R30)."""

    def _pit(self, **over):
        data = dict(investigation_id="TP-1", kind="test_pit",
                    depth_unit="m", total_depth=_m(3.0),
                    layers=[Layer(top=_m(0.0), bottom=_m(3.0),
                                  description="SILTY SAND", uscs="SM")],
                    water=[WaterLevel(depth=_m(2.1), when="while_drilling"),
                           WaterLevel(depth=_m(1.8), when="after_hours",
                                      hours=24.0)])
        data.update(over)
        return Investigation(**data)

    def test_it_is_written_validates_and_comes_back(self):
        from subsurface_characterization import parse_diggs
        pit = self._pit()
        notes = []
        xml = write_diggs([pit], notes=notes)
        ok, errors = diggs_schema_gate(xml)
        assert ok, errors[:3]
        assert "water_depth" in xml and "<waterStrike>" not in xml
        assert notes[0].water == 2
        ok, diffs = diggs_roundtrip_gate(xml, [pit])
        assert ok, diffs
        (back,) = parse_diggs(content=xml).site.investigations
        assert back.gwl_depth_m == pytest.approx(2.1)

    def test_a_wrong_water_level_is_caught(self):
        pit = self._pit()
        xml = write_diggs([pit])
        pit.water[0].depth = _m(2.6)
        ok, diffs = diggs_roundtrip_gate(xml, [pit])
        assert not ok and any("water level" in d for d in diffs)

    def test_a_pit_that_found_no_water_is_named_not_invented(self):
        pit = self._pit(water=[WaterLevel(depth=None,
                                          when="not_encountered")])
        notes = []
        xml = write_diggs([pit], notes=notes)
        assert "water_depth" not in xml
        assert any("no water encountered" in s for s in notes[0].skipped)
        assert diggs_schema_gate(xml)[0] is True


class TestAValuePrintedWithNoUnit:
    """A value with no printed unit where its property needs one (R15
    D-values, R23 pocket penetrometers and a summary row's cohesion)."""

    def test_a_d_value_with_no_unit_is_millimetres_and_comes_back(self):
        from subsurface_characterization import parse_diggs
        from report_ingest.model import GradationResult, LabTest
        test = LabTest(kind="gradation", investigation_id="SB-02",
                       depth_top=_m(7.0),
                       result=GradationResult(
                           d30=Quantity(value=0.091, unit=""),
                           d60=Quantity(value=0.148, unit=""),
                           d100=Quantity(value=12.5, unit="")))
        notes = []
        xml = write_diggs([test], Project(name="R"), notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        assert len(notes[0].assumed) == 3
        ok, diffs = diggs_roundtrip_gate(xml, [test])
        assert ok, diffs
        (hole,) = parse_diggs(content=xml).site.investigations
        d60 = [m.value for m in hole.measurements if m.parameter == "D60_mm"]
        assert d60 == [pytest.approx(0.148)]

    def test_a_pocket_penetrometer_with_no_unit_is_left_out_not_guessed(
            self):
        pit = Investigation(
            investigation_id="STP-02", kind="test_pit", depth_unit="m",
            samples=[Sample(sample_id="1", top=_m(0.65), kind="bulk",
                            pocket_pen=Quantity(value=8.0, unit="")),
                     Sample(sample_id="2", top=_m(0.9), kind="bulk",
                            pocket_pen=Quantity(value=1.5, unit="tsf"))])
        notes = []
        xml = write_diggs([pit], notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        assert notes[0].unwritten == [("STP-02", pytest.approx(0.65), 8.0)]
        ok, diffs = diggs_roundtrip_gate(xml, [pit])
        assert ok, diffs
        # what WAS written is still checked
        pit.samples[1].pocket_pen = Quantity(value=2.5, unit="tsf")
        ok, diffs = diggs_roundtrip_gate(xml, [pit])
        assert not ok and any("pocket_pen" in d for d in diffs)

    def test_a_summary_rows_cohesion_with_no_unit_is_left_out(self):
        from report_ingest.diggs_writer import DiggsWriteNotes
        from report_ingest.model import (
            LabTest, SummaryRow, SummaryTableResult,
        )
        table = LabTest(kind="summary_table", result=SummaryTableResult(rows=[
            SummaryRow(investigation_id="LB-12", depth_top=_m(0.0),
                       wc=12.0, c=Quantity(value=21.5, unit=""))]))
        notes = []
        xml = write_diggs([table], Project(name="R"), notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        assert notes[0].unwritten
        ok, diffs = diggs_roundtrip_gate(xml, [table])
        assert ok, diffs
        # notes that do NOT excuse it: the gate says so
        ok, diffs = diggs_roundtrip_gate(xml, [table],
                                         notes=DiggsWriteNotes())
        assert not ok and any("21.5" in d for d in diffs)
