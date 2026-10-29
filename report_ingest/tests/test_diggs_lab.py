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


@pytest.mark.parametrize("printed", ["B-1", "B-1 S-1"])
def test_a_test_the_reconciler_linked_is_written_in_the_hole_it_names(
        printed):
    """RI-07: a sheet printing "B-1" for the log's "B-01", or "B-1 S-1" for
    its specimen, is written in the logged hole the reconciler linked it
    to -- not in a second, empty hole synthesised for the printed words --
    and the round-trip gate finds it there."""
    log = Investigation(investigation_id="B-01", depth_unit="ft")
    test = _one("atterberg", AtterbergResult(ll=40, pl=20, pi=20),
                investigation_id=printed)
    test.linked_investigation_id = "B-01"
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert not notes[0].synthesised
    assert notes[0].lab_tests == 1
    ok, errors = diggs_schema_gate(xml)
    assert ok, errors[:2]
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs
    assert test.investigation_id == printed          # the record untouched


def _row_table(*rows):
    return LabTest(kind="summary_table",
                   result=SummaryTableResult(rows=list(rows)))


def test_a_summary_row_naming_no_log_gets_a_hole_named_as_printed():
    """A row printing a boring no log describes is written in a hole
    carrying that name, as a sheet printing it would be, and the note says
    a summary row named it."""
    table = _row_table(SummaryRow(investigation_id="QZ-7",
                                  depth_top=q(2.0, "m"), ll=40.0))
    log = Investigation(investigation_id="B-1", depth_unit="m")
    notes = []
    xml = write_diggs([log, table], Project(name="R"), notes=notes)
    (made,) = notes[0].synthesised
    assert made.startswith("QZ-7: named by a row of a laboratory summary")
    assert notes[0].lab_tests == 1
    assert diggs_schema_gate(xml)[0] is True
    ok, diffs = diggs_roundtrip_gate(xml, [log, table])
    assert ok, diffs


def test_rows_and_a_sheet_printing_one_name_share_one_hole():
    sheet = _one("atterberg", AtterbergResult(ll=40, pl=20, pi=20),
                 investigation_id="QZ-7")
    table = _row_table(
        SummaryRow(investigation_id="QZ-7", depth_top=q(3.0, "m"), ll=41.0),
        SummaryRow(investigation_id="QZ-7", depth_top=q(5.0, "m"), ll=46.5))
    notes = []
    xml = write_diggs([table, sheet], Project(name="R"), notes=notes)
    (made,) = notes[0].synthesised
    assert made.startswith("QZ-7: named by a laboratory sheet")
    assert notes[0].lab_tests == 3
    ok, diffs = diggs_roundtrip_gate(xml, [table, sheet])
    assert ok, diffs


def test_a_summary_row_printing_no_boring_stays_in_the_record():
    table = _row_table(SummaryRow(depth_top=q(2.0, "m"), ll=40.0))
    log = Investigation(investigation_id="B-1", depth_unit="m")
    notes = []
    write_diggs([log, table], Project(name="R"), notes=notes)
    assert notes[0].synthesised == []
    assert notes[0].lab_tests == 0
    assert any("no boring named" in s and "kept in the record" in s
               for s in notes[0].skipped), notes[0].skipped


def test_a_placed_summary_row_with_nothing_to_write_says_so():
    """RI-71: a row with a hole and a depth but nothing the writer can put
    in a result set is named in the notes, as a sheet is."""
    bare = SummaryRow(investigation_id="B-1", depth_top=q(2.0, "m"),
                      other=[("Shelf", "kept dry")])
    unit = SummaryRow(investigation_id="B-1", depth_top=q(3.0, "m"),
                      qu=q(1.0, "furlongs"))
    kept = SummaryRow(investigation_id="B-1", depth_top=q(4.0, "m"), ll=40.0)
    log = Investigation(investigation_id="B-1", depth_unit="m")
    notes = []
    xml = write_diggs([log, _row_table(bare, unit, kept)], Project(name="R"),
                      notes=notes)
    assert notes[0].lab_tests == 1
    (first,) = [s for s in notes[0].skipped
                if s.startswith("summary row 1 of")]
    assert first.endswith("the row carries no value this writer can put in "
                          "a result set; kept in the record, not written "
                          "here")
    second = [s for s in notes[0].skipped
              if s.startswith("summary row 2 of")]
    assert len(second) == 2 and "furlongs" in second[0]
    assert "none of the row's values could be written" in second[1]
    assert not [s for s in notes[0].skipped
                if s.startswith("summary row 3 of")]
    ok, diffs = diggs_roundtrip_gate(xml, [log, _row_table(bare, unit, kept)])
    assert ok, diffs


def test_a_summary_row_is_written_in_the_hole_it_was_linked_to():
    row = SummaryRow(investigation_id="B-1", depth_top=q(2.0, "m"), ll=40.0)
    row.linked_investigation_id = "B-01"
    log = Investigation(investigation_id="B-01", depth_unit="m")
    notes = []
    xml = write_diggs([log, _row_table(row)], Project(name="R"),
                      notes=notes)
    assert notes[0].synthesised == []
    assert notes[0].lab_tests == 1
    ok, diffs = diggs_roundtrip_gate(xml, [log, _row_table(row)])
    assert ok, diffs
    assert row.investigation_id == "B-1"             # the record untouched


def test_a_summary_row_shares_the_hole_made_for_a_sheet():
    """A sheet naming a hole no log describes still gets a hole carrying its
    name; a row printing that name is written in it, and adds none."""
    sheet = _one("atterberg", AtterbergResult(ll=40, pl=20, pi=20),
                 investigation_id="QZ-7")
    row = SummaryRow(investigation_id="QZ-7", depth_top=q(3.0, "m"), ll=41.0)
    notes = []
    xml = write_diggs([sheet, _row_table(row)], Project(name="R"),
                      notes=notes)
    assert [s.split(":")[0] for s in notes[0].synthesised] == ["QZ-7"]
    assert notes[0].lab_tests == 2
    ok, diffs = diggs_roundtrip_gate(xml, [sheet, _row_table(row)])
    assert ok, diffs


def test_a_test_nobody_linked_is_placed_by_its_printed_hole():
    log = Investigation(investigation_id="B-01", depth_unit="ft")
    test = _one("atterberg", AtterbergResult(ll=40, pl=20, pi=20),
                investigation_id="B-2")
    notes = []
    write_diggs([log, test], Project(name="R"), notes=notes)
    assert any(s.startswith("B-2:") for s in notes[0].synthesised)


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


def test_a_water_analysis_with_unitless_values_is_valid_and_comes_back():
    """RI-20 review: a pH and an organic content printed with no unit (they
    have none), a water analysis in mg/L and a conductivity in uS/cm pass the
    schema and read back; the organic content was held and never written."""
    test = _one("chemical", ChemicalResult(
        pH=q(7.2, ""), sulfate=q(120.0, "mg/L"), chloride=q(35.0, "mg/l"),
        conductivity=q(350.0, "µS/cm"), organic_percent=q(3.1, ""),
        wc=q(14.0, "")))
    notes = []
    xml = write_diggs([test], Project(name="R"), notes=notes)
    assert diggs_schema_gate(xml)[0] is True
    assert "<uom>mg/L</uom>" in xml and "<uom>S/m</uom>" in xml
    assert ">LOI</propertyClass>" in xml
    assert not notes[0].skipped
    ok, diffs = diggs_roundtrip_gate(xml, [test])
    assert ok, diffs


@pytest.mark.parametrize("unit,value", [("kg/m3", 1850.0), ("Mg/m3", 1.85),
                                        ("g/cm3", 1.85)])
def test_a_mass_density_written_as_a_unit_weight_says_g(unit, value):
    test = _one("density", MoistureDensityResult(
        kind="density", dry_density=q(value, unit)))
    notes = []
    xml = write_diggs([test], Project(name="R"), notes=notes)
    (said,) = [s for s in notes[0].normalised if "standard gravity" in s]
    assert unit in said and "kN/m3" in said and "9.80665" in said
    assert diggs_roundtrip_gate(xml, [test])[0]


def test_a_unit_weight_printed_as_one_needs_no_word_about_g():
    test = _one("density", MoistureDensityResult(
        kind="density", dry_density=q(115.0, "pcf")))
    notes = []
    write_diggs([test], Project(name="R"), notes=notes)
    assert not any("gravity" in s for s in notes[0].normalised)


# ---------------------------------------------------------------------------
# a sheet's spelling of a log's id, and a pit named as one (RI-96)
# ---------------------------------------------------------------------------

def _atterberg(name):
    return _one("atterberg", AtterbergResult(ll=40, pl=20, pi=20),
                investigation_id=name)


@pytest.mark.parametrize("printed", ["B1", "b-1", "B 1", "B-001", "B_01"])
def test_a_spelling_of_a_logs_id_is_written_in_that_log(printed):
    log = Investigation(investigation_id="B-01", depth_unit="ft")
    test = _atterberg(printed)
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert not notes[0].synthesised
    assert notes[0].lab_tests == 1
    ok, errors = diggs_schema_gate(xml)
    assert ok, errors[:2]
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs[:4]


def test_a_summary_row_spelling_is_written_in_the_logs_hole():
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    row = SummaryRow(investigation_id="b01", depth_top=q(5.0, "ft"), ll=40.0)
    test = LabTest(kind="summary_table", result=SummaryTableResult(rows=[row]))
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert not notes[0].synthesised
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs[:4]


def test_a_letter_prefix_is_not_stripped_so_lb7_and_b7_stay_apart():
    log = Investigation(investigation_id="B-7", depth_unit="ft")
    notes = []
    write_diggs([log, _atterberg("XB-7")], Project(name="R"), notes=notes)
    assert [s.split(":")[0] for s in notes[0].synthesised] == ["XB-7"]


def test_a_key_two_logs_share_makes_no_guess():
    logs = [Investigation(investigation_id="B-1", depth_unit="ft"),
            Investigation(investigation_id="B1", depth_unit="ft")]
    notes = []
    write_diggs(logs + [_atterberg("b 01")], Project(name="R"), notes=notes)
    assert [s.split(":")[0] for s in notes[0].synthesised] == ["b 01"]


@pytest.mark.parametrize("name", ["TP-1", "Test Pit 1", "trial pit A"])
def test_a_stand_in_named_as_a_pit_is_written_as_a_trial_pit(name):
    notes = []
    xml = write_diggs([_atterberg(name)], Project(name="R"), notes=notes)
    assert "<TrialPit" in xml and "<Borehole" not in xml
    assert "TrialPit carrying" in notes[0].synthesised[0]
    ok, errors = diggs_schema_gate(xml)
    assert ok, errors[:2]


@pytest.mark.parametrize("name", ["B-1", "TPL-3", "Tpa"])
def test_a_stand_in_not_named_as_a_pit_stays_a_borehole(name):
    xml = write_diggs([_atterberg(name)], Project(name="R"))
    assert "<Borehole" in xml and "<TrialPit" not in xml


@pytest.mark.parametrize("log_id, printed", [
    ("B-11", "B-1-1"), ("B-101", "B-1-01"), ("B-15", "B-1.5"),
    ("TP-12", "TP-1-2")])
def test_a_name_with_two_joined_numbers_is_not_another_holes_spelling(
        log_id, printed):
    log = Investigation(investigation_id=log_id, depth_unit="ft")
    notes = []
    write_diggs([log, _atterberg(printed)], Project(name="R"), notes=notes)
    assert [s.split(":")[0] for s in notes[0].synthesised] == [printed]


def test_a_spelling_of_a_left_out_log_is_noted_as_left_out():
    from report_ingest.model import Layer
    left = Investigation(
        investigation_id="B-5", depth_unit="", units_known=False,
        layers=[Layer(top=Quantity(value=4.0, unit=""), description="SAND")])
    keep = Investigation(investigation_id="B-9", depth_unit="ft")
    notes = []
    write_diggs([left, keep, _atterberg("b05")], Project(name="R"),
                notes=notes)
    (made,) = notes[0].synthesised
    assert made.startswith("b05:") and "left out" in made


def _qu_row(qu_unit, **over):
    row = SummaryRow(investigation_id="B-1", depth_top=q(5.0, "ft"),
                     qu=q(3.0, qu_unit), **over)
    return LabTest(kind="summary_table", result=SummaryTableResult(rows=[row]))


def test_a_row_whose_only_value_is_unwritable_does_not_fail_the_gate():
    """RI-97: the writer names a value it cannot write; the gate does not
    call the row's hole missing for the want of it."""
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    test = _qu_row("kN/m3")
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert notes[0].unwritten
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs


def test_the_gate_still_fails_a_hole_that_is_really_missing():
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    xml = write_diggs([log])
    ok, diffs = diggs_roundtrip_gate(xml, [log, _qu_row("kPa")])
    assert not ok and any("no result" in d for d in diffs)


def test_the_gate_still_fails_a_written_value_that_is_missing():
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    xml = write_diggs([log, _qu_row("kN/m3")])
    ok, diffs = diggs_roundtrip_gate(xml, [log, _qu_row("kN/m3", ll=40.0)])
    assert not ok


def test_n_per_mm_on_a_strength_is_written_as_a_stress_and_said():
    """RI-93: the square of the millimetre lost in printing is put back, and
    the notes say so."""
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    test = _qu_row("N/mm")
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert not notes[0].unwritten
    assert any("N/mm" in s and "read as N/mm2" in s
               for s in notes[0].assumed)
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs


@pytest.mark.parametrize("make", ["summary", "sheet"])
def test_a_resistivity_in_the_laboratory_s_mohm_per_cm_is_written(make):
    """RI-194: "Mohm/cm" is megohm-cm on a resistivity; the writer reads it
    as the round-trip gate does, on a summary row and on a sheet."""
    log = Investigation(investigation_id="B-1", depth_unit="ft")
    value = q(0.03, "Mohm/cm")
    if make == "summary":
        test = LabTest(kind="summary_table", result=SummaryTableResult(
            rows=[SummaryRow(investigation_id="B-1", depth_top=q(5.0, "ft"),
                             resistivity=value)]))
    else:
        test = _one("chemical", ChemicalResult(resistivity=value,
                                               resistivity_minimum=value))
    notes = []
    xml = write_diggs([log, test], Project(name="R"), notes=notes)
    assert not notes[0].unwritten and not notes[0].skipped
    ok, diffs = diggs_roundtrip_gate(xml, [log, test])
    assert ok, diffs
