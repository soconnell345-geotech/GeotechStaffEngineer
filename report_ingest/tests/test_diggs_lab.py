"""The DIGGS lab writer on synthetic records: the schema, and the round trip.

The gate that matters runs on thirty-one real sheets and is in the harness
(``module_work/report_ingest_harness/tests/test_lab_diggs_truth.py``), where
it is skipped everywhere but the owner's machine. This walks the same path on
records written here, so CI covers every kind of test, every place a value
can go and every reason the writer has for leaving one out.

What it asserts, beyond "it validates": that a lab test is POSITIONED in a
hole and refuses to be written when it cannot be; that a value printed as
words stays words; that a curve is a table of rows and a set of scalars is
one row; that the one namespace trap in 2.6 is not fallen into; and that
every number in the record comes back through the app's own readers.
"""

from __future__ import annotations

import pytest

from report_ingest.diggs_writer import (
    APP_CODESPACE, DICTIONARY_CODESPACE, LAB_PROCEDURE, UOM_IN_FILE,
    diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
)
from report_ingest.model import (
    AtterbergResult, CBRResult, ChemicalResult, CompactionPoint,
    CompactionResult, ConsolidationPoint, ConsolidationResult,
    GradationResult, Investigation, LabTest, MoistureDensityResult,
    OtherResult, Project, Quantity, ShearPoint, SievePoint, StrengthResult,
    StrengthSpecimen, SummaryRow, SummaryTableResult,
)


def q(value, unit):
    return Quantity(value=value, unit=unit)


def _one(kind, result, **over):
    base = dict(kind=kind, investigation_id="B-1", depth_top=q(5.0, "ft"),
                result=result)
    base.update(over)
    return LabTest(**base)


ONE_OF_EACH = [
    _one("atterberg",
         AtterbergResult(ll=48, pl=22, pi=26, uscs="CL",
                         flow_curve=[(35, 51.2), (25, 48.4), (18, 45.9)],
                         pl_trials=[21.8, 22.2], water_content=27.4),
         standard="ASTM D4318"),
    _one("gradation",
         GradationResult(
             gravel_percent=13, sand_percent=44, fines_percent=43,
             d10=q(0.002, "mm"), d60=q(0.42, "mm"), d90=q(3.1, "mm"),
             cu=210.0, cc=1.4,
             percent_passing=[
                 SievePoint(percent_passing=100.0, size=q(75.0, "mm"),
                            sieve="3 in"),
                 SievePoint(percent_passing=87.0, size=q(4.75, "mm"),
                            sieve="No. 4"),
                 SievePoint(percent_passing=43.0, size=q(0.075, "mm"),
                            sieve="No. 200")]),
         depth_top=q(12.0, "ft")),
    _one("swell_consolidation",
         ConsolidationResult(
             test_type="swell", swell_percent=0.5, swell_at=q(100.0, "psf"),
             pc=q(4000.0, "psf"), cc=0.21, cr=0.03, e0=0.68,
             dry_unit_weight=q(96.0, "pcf"), wc=25.0,
             points=[ConsolidationPoint(stress=q(100.0, "psf"),
                                        strain_percent=0.5),
                     ConsolidationPoint(stress=q(1000.0, "psf"),
                                        strain_percent=-0.8),
                     ConsolidationPoint(stress=q(8000.0, "psf"),
                                        strain_percent=-4.0),
                     ConsolidationPoint(stress=q(2000.0, "psf"),
                                        strain_percent=-3.2,
                                        stage="rebound")]),
         standard="ASTM D4546"),
    _one("triaxial",
         StrengthResult(
             kind="triaxial", test_type="CU", c=q(12.0, "kPa"), phi_deg=28.5,
             specimens=[
                 StrengthSpecimen(specimen_id="1", confining=q(100.0, "kPa"),
                                  peak_deviator=q(185.0, "kPa"),
                                  strain_at_peak_percent=10.7,
                                  pore_pressure=q(42.0, "kPa"),
                                  stress_ratio=3.7, wc=31.2,
                                  height=q(148.0, "mm"),
                                  diameter=q(74.5, "mm")),
                 StrengthSpecimen(specimen_id="2", confining=q(200.0, "kPa"),
                                  peak_deviator=q(360.0, "kPa"),
                                  strain_at_peak_percent=12.0)],
             points=[ShearPoint(x=q(0.5, "%"), y=q(40.0, "kPa"),
                                specimen="1"),
                     ShearPoint(x=q(10.7, "%"), y=q(185.0, "kPa"),
                                specimen="1")])),
    _one("direct_shear",
         StrengthResult(
             kind="direct_shear", test_type="direct_shear",
             c=q(539.0, "psf"), phi_deg=30.0,
             points=[ShearPoint(x=q(500.0, "psf"), y=q(850.0, "psf")),
                     ShearPoint(x=q(1000.0, "psf"), y=q(1085.0, "psf")),
                     ShearPoint(x=q(2000.0, "psf"), y=q(1720.0, "psf"))]),
         standard="ASTM D3080"),
    _one("unconfined_rock",
         StrengthResult(kind="unconfined_rock", test_type="unconfined_rock",
                        qu=q(60.0, "MPa"), wet_density=q(2.65, "Mg/m3"),
                        rock_type="Limestone", weathering="Fresh",
                        specimens=[StrengthSpecimen(
                            specimen_id="1", height=q(229.0, "mm"),
                            diameter=q(88.0, "mm"))]),
         depth_top=q(20.8, "m")),
    _one("compaction",
         CompactionResult(
             method="standard", max_dry_density=q(1.62, "g/cm3"),
             optimum_wc=11.6, mould_volume=q(903.21, "cm3"), layers=3,
             blows_per_layer=25,
             points=[CompactionPoint(water_content=10.09,
                                     dry_density=q(1.54, "g/cm3")),
                     CompactionPoint(water_content=11.6,
                                     dry_density=q(1.62, "g/cm3")),
                     CompactionPoint(water_content=14.44,
                                     dry_density=q(1.57, "g/cm3"))])),
    _one("cbr",
         CBRResult(cbr_percent=14.0, cbr_at_0_1in=14.0, cbr_at_0_2in=16.5,
                   swell_percent=0.4, soaked=True,
                   surcharge=q(4.54, "kg"), dry_density=q(1.8, "g/cm3"),
                   wc=12.0, points=[(0.1, 1520.0), (0.2, 1810.0)])),
    _one("moisture_content",
         MoistureDensityResult(kind="moisture_content", wc=9.1,
                               water_contents=[11.7, 6.6])),
    _one("density",
         MoistureDensityResult(kind="density", wet_density=q(2.11, "g/cm3"),
                               dry_density=q(1.85, "g/cm3"),
                               specific_gravity=2.68)),
    _one("organic_content",
         MoistureDensityResult(kind="organic_content", wc=48.0,
                               ash_percent=93.3, organic_percent=6.7),
         standard="ASTM D2974"),
    _one("chemical",
         ChemicalResult(pH=8.43, resistivity=q(1261.0, "ohm-cm"),
                        sulfate=q(0.02, "%"), chloride=q(25.0, "mg/kg"),
                        sulfides="Nil", redox=q(574.0, "mV"),
                        total_salts=q(1204.0, "mg/kg"),
                        temperature=q(24.5, "degC"),
                        lab_sample_id="15-0071")),
    _one("specific_gravity", OtherResult(kind="specific_gravity",
                                         fields={"Gs": "2.71"})),
    _one("other", OtherResult(no_results=True,
                              fields={"lab_sample_id": "24D0646-01"})),
    LabTest(kind="summary_table",
            result=SummaryTableResult(rows=[
                SummaryRow(investigation_id="A-1", depth_top=q(2.0, "m"),
                           wc=18.2, ll=41, pl=20.0, pi=21,
                           elevation_top=q(275.0, "m"), uscs="CL",
                           percent_passing=[SievePoint(
                               percent_passing=72.0, size=q(0.075, "mm"),
                               sieve="No. 200")]),
                SummaryRow(investigation_id="A-2", depth_top=q(3.5, "m"),
                           pH=6.0, resistivity=q(17000.0, "ohm-cm"),
                           sulfides="0", other=[("stratum", "B1")])])),
]


@pytest.fixture(scope="module")
def written():
    notes = []
    xml = write_diggs(ONE_OF_EACH, Project(name="Report", number="12345"),
                      document_id="R00", notes=notes)
    return xml, notes[0]


def test_every_kind_writes_valid_diggs(written):
    xml, _notes = written
    ok, errors = diggs_schema_gate(xml)
    assert ok, "; ".join(e[:300] for e in errors[:3])


def test_every_number_comes_back(written):
    xml, _notes = written
    ok, diffs = diggs_roundtrip_gate(xml, ONE_OF_EACH)
    assert ok, "; ".join(diffs[:8])


def test_a_test_per_kind_and_a_curve_per_curve(written):
    xml, notes = written
    # Fourteen tests plus a summary table of two rows, each carrying scalars
    # and, where it has one, a curve of its own: more Tests than records.
    assert notes.lab_tests > len(ONE_OF_EACH)
    assert notes.tests == notes.lab_tests


def test_the_triaxial_procedure_is_in_the_diggs_namespace_not_the_geo_one():
    """The one namespace trap in 2.6, pinned.

    Twelve of the thirteen laboratory procedures are in the geotechnical
    namespace and ``TriaxialTest`` is not, because the schema declares it in
    ``TestProceduresAll.xsd``. A file that puts it in the other one validates
    against nothing and reads as empty.
    """
    assert LAB_PROCEDURE["triaxial"] == "TriaxialTest"
    for kind, element in LAB_PROCEDURE.items():
        if kind != "triaxial":
            assert element.startswith("diggs_geo:"), kind


def test_the_procedures_are_in_the_file(written):
    xml, _notes = written
    for element in ("diggs_geo:AtterbergLimitsTest",
                    "diggs_geo:ParticleSizeTest",
                    "diggs_geo:ConsolidationTest", "<TriaxialTest",
                    "diggs_geo:DirectShearTest",
                    "diggs_geo:UnconfinedCompressiveStrengthTest",
                    "diggs_geo:LabCompactionTest", "diggs_geo:LabCBRTest",
                    "diggs_geo:WaterContentTest", "diggs_geo:LabDensityTest",
                    "diggs_geo:LossOnIgnitionTest",
                    "diggs_geo:LabChemicalTest",
                    "diggs_geo:SpecificGravityTest"):
        assert element in xml, element


def test_the_standard_is_written_as_a_specification(written):
    xml, _notes = written
    assert "<standardReferenceNumber>ASTM D4318" in xml
    assert "<standardReferenceNumber>ASTM D4546" in xml


def test_a_value_printed_as_words_is_written_as_text(written):
    xml, _notes = written
    assert "<typeData>string</typeData>" in xml
    assert "Nil" in xml


def test_a_property_the_dictionary_has_no_term_for_says_so(written):
    """D90 is real and is not a DIGGS term. The file says which it is."""
    xml, _notes = written
    assert DICTIONARY_CODESPACE in xml and APP_CODESPACE in xml
    block = xml[xml.index(">d90<") - 400:xml.index(">d90<") + 10]
    assert APP_CODESPACE in block
    block = xml[xml.index(">d10<") - 400:xml.index(">d10<") + 10]
    assert DICTIONARY_CODESPACE in block


def test_a_particle_size_is_written_in_millimetres(written):
    """In metres a No. 200 sieve is 7.5e-05 and rounds away."""
    xml, _notes = written
    assert "<uom>mm</uom>" in xml
    assert "0.075" in xml


def test_a_friction_angle_uses_the_schemas_own_code_for_degrees(written):
    xml, _notes = written
    assert UOM_IN_FILE["deg"] == "dega"
    assert "<uom>dega</uom>" in xml
    assert "<uom>deg</uom>" not in xml


def test_a_curve_is_many_rows_and_a_scalar_set_is_one():
    from subsurface_characterization.diggs26 import parse_diggs26_result_sets

    xml = write_diggs(ONE_OF_EACH, Project(name="Report"), document_id="R00")
    sets = parse_diggs26_result_sets(content=xml)
    grading = [s for s in sets if s.name.endswith("grading")]
    assert grading and grading[0].n_rows == 3
    assert grading[0].column("percent_passing") is not None
    scalars = [s for s in sets if s.name == "atterberg"]
    assert scalars and scalars[0].n_rows == 1
    assert scalars[0].scalar("liquid_limit") == 48.0


def test_the_app_reads_the_scalars_back_into_its_own_site_model():
    from subsurface_characterization import parse_diggs

    xml = write_diggs(ONE_OF_EACH, Project(name="Report"), document_id="R00")
    parsed = parse_diggs(content=xml)
    by_id = {inv.investigation_id: inv for inv in parsed.site.investigations}
    assert "B-1" in by_id and "A-1" in by_id
    names = {m.parameter for m in by_id["B-1"].measurements}
    assert {"LL_pct", "PL_pct", "PI_pct", "pct_fines", "pH",
            "resistivity_ohm_m", "qu_kPa", "organic_pct"} <= names


def test_a_lab_test_with_no_boring_is_not_written_and_says_why():
    test = LabTest(kind="atterberg", depth_top=q(2.0, "m"),
                   result=AtterbergResult(ll=40, pl=20, pi=20))
    notes = []
    xml = write_diggs([test], Project(name="R"), notes=notes)
    assert notes[0].lab_tests == 0
    assert any("names no boring" in s for s in notes[0].skipped)
    ok, _errors = diggs_schema_gate(xml)
    assert ok, "a file with nothing in it is still a DIGGS file"


def test_a_lab_test_with_no_depth_is_not_written_and_says_why():
    test = LabTest(kind="chemical", investigation_id="LB-5",
                   result=ChemicalResult(pH=7.9))
    notes = []
    write_diggs([test], Project(name="R"), notes=notes)
    assert notes[0].lab_tests == 0
    assert any("no depth is printed" in s for s in notes[0].skipped)


def test_a_hole_only_a_lab_sheet_named_is_recorded_as_one():
    notes = []
    write_diggs(ONE_OF_EACH, Project(name="R"), notes=notes)
    assert any("B-1" in s for s in notes[0].synthesised)
    # A hole a LOG describes is not synthesised.
    notes = []
    write_diggs([Investigation(investigation_id="B-1", depth_unit="ft"),
                 ONE_OF_EACH[0]], Project(name="R"), notes=notes)
    assert not notes[0].synthesised


def test_an_unconvertible_unit_is_left_out_and_named():
    test = LabTest(kind="unconfined", investigation_id="B-1",
                   depth_top=q(3.0, "m"),
                   result=StrengthResult(kind="unconfined",
                                         qu=Quantity(value=12.0,
                                                     unit="furlongs")))
    notes = []
    xml = write_diggs([test], Project(name="R"), notes=notes)
    assert any("furlongs" in s for s in notes[0].skipped)
    # And the gate does not go looking for what the writer said it dropped.
    ok, diffs = diggs_roundtrip_gate(xml, [test])
    assert ok, diffs


def test_a_record_may_mix_logs_and_lab_tests():
    from report_ingest.model import Layer, Sample

    log = Investigation(
        investigation_id="B-1", depth_unit="ft",
        layers=[Layer(top=q(0.0, "ft"), bottom=q(8.0, "ft"),
                      description="SANDY LEAN CLAY", uscs="CL")],
        samples=[Sample(sample_id="S-1", top=q(5.0, "ft"),
                        bottom=q(6.5, "ft"), kind="spt")])
    items = [log, ONE_OF_EACH[0]]
    notes = []
    xml = write_diggs(items, Project(name="R"), notes=notes)
    assert notes[0].investigations == 1 and notes[0].lab_tests >= 1
    ok, errors = diggs_schema_gate(xml)
    assert ok, errors[:2]
    ok, diffs = diggs_roundtrip_gate(xml, items)
    assert ok, diffs


def test_a_report_record_writes_its_lab_tests_too():
    from report_ingest.model import ReportRecord

    record = ReportRecord(lab_tests=[ONE_OF_EACH[0]])
    record.project = Project(name="Report")
    notes = []
    xml = write_diggs(record, notes=notes)
    assert notes[0].lab_tests >= 1
    ok, diffs = diggs_roundtrip_gate(xml, record)
    assert ok, diffs


def test_something_that_is_neither_is_refused():
    with pytest.raises(TypeError):
        write_diggs([ONE_OF_EACH[0], "not a record"], Project(name="R"))
