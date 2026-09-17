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
