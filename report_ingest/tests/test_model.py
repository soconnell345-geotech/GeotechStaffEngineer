"""The record model: units, provenance, and the schema a consumer reads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from report_ingest.model import (
    SCHEMA_VERSION, UNIT_TO_SI, Calculation, DocumentFacts, Investigation,
    LabTest, Layer, NamedQuantity, Provenance, QAEntry, Quantity,
    ReportRecord, SPT, Sample, WaterLevel, record_json_schema, si_numbers,
    to_si,
)


class TestQuantity:
    def test_a_quantity_cannot_be_built_without_a_unit(self):
        with pytest.raises(ValidationError):
            Quantity(value=21.5)

    def test_the_value_is_kept_as_printed(self):
        q = Quantity(value=21.5, unit="ft")
        assert q.value == 21.5 and q.unit == "ft"

    @pytest.mark.parametrize("value,unit,want,si_unit", [
        (21.5, "ft", 6.5532, "m"),
        (18.0, "in", 0.4572, "m"),
        (1500.0, "mm", 1.5, "m"),
        (2.0, "tsf", 191.5210, "kPa"),
        (2000.0, "psf", 95.7605, "kPa"),
        (1.0, "ksf", 47.8803, "kPa"),
        (1.0, "kg/cm2", 98.0665, "kPa"),
        (116.0, "pcf", 18.2221, "kN/m3"),
        (1.85, "g/cm3", 18.1423, "kN/m3"),
        (1.85, "Mg/m3", 18.1423, "kN/m3"),
        (12.0, "blows/ft", 11.8110, "blows/0.3m"),
        (12.0, "blows/30cm", 12.0, "blows/0.3m"),
    ])
    def test_the_small_table_converts_what_a_log_prints(self, value, unit,
                                                        want, si_unit):
        got = Quantity(value=value, unit=unit).to_si()
        assert got is not None
        assert got.value == pytest.approx(want, rel=1e-4)
        assert got.unit == si_unit

    def test_a_unit_not_in_the_table_is_refused_not_guessed(self):
        assert Quantity(value=5.0, unit="furlongs").to_si() is None
        assert to_si(5.0, "cubits") is None

    # RI-20: a blank unit is a unit the page did not print, not the
    # dimensionless one. It converts to nothing -- a depth of "4" is not
    # 4 m -- unless the caller says the quantity's kind has no unit.
    @pytest.mark.parametrize("blank", ["", "  "])
    def test_an_empty_unit_is_unknown_and_converts_to_nothing(self, blank):
        assert Quantity(value=4.0, unit=blank).to_si() is None
        assert to_si(4.0, blank) is None
        assert "" not in UNIT_TO_SI

    def test_a_blank_converts_only_for_a_kind_that_has_no_unit(self):
        got = Quantity(value=7.2, unit="").to_si(kind="ratio")
        assert got is not None and (got.value, got.unit) == (7.2, "1")
        assert to_si(12.0, "", kind="percent") == (12.0, "%")
        # a kind with a unit does not make a blank complete
        assert to_si(4.0, "", kind="length") is None

    def test_the_record_walk_recognises_a_ph_by_its_field_not_its_blank(
            self):
        from report_ingest.model import ChemicalResult
        result = ChemicalResult(pH=Quantity(value=7.2, unit=""),
                                sulfate=Quantity(value=120.0, unit=""))
        got = {path: (v, u) for path, v, u in si_numbers(result)}
        assert got["pH"] == (7.2, "1")
        assert "sulfate" not in got          # unknown unit: no SI value
        printed = {path: (v, u) for path, v, u in
                   si_numbers(result, unitless_as_printed=True)}
        assert printed["sulfate"] == (120.0, "")

    @pytest.mark.parametrize("value,unit,want,si_unit", [
        # calculation printouts
        (10.0, "kips", 44.4822, "kN"), (5.0, "KN.", 5.0, "kN"),
        (2.0, "kip-ft", 2.71164, "kN.m"), (3.0, "kN-m", 3.0, "kN.m"),
        (3.0, "m-kN", 3.0, "kN.m"), (1.0, "kN-m^2", 1.0, "kN.m2"),
        (1.0, "kN/m", 1.0, "kN/m"), (1.0, "N/mm", 1.0, "kN/m"),
        (100.0, "lb/ft", 1.45939, "kN/m"), (1.0, "ksi", 6894.757, "kPa"),
        (1.0, "N/mm2", 1000.0, "kPa"), (1.0, "kips/ft2", 47.8803, "kPa"),
        (1.0, "KN/SQ M", 1.0, "kPa"), (2.0, "bars", 200.0, "kPa"),
        (1.0, "radians", 57.29578, "deg"), (30.0, "DEG.", 30.0, "deg"),
        (3.0, "M.", 3.0, "m"), (2.0, "SQ. M", 2.0, "m2"),
        (144.0, "in2", 0.092903, "m2"), (1.0, "CU. M", 1.0, "m3"),
        (2.0, "yrs", 63115200.0, "s"), (1.0, "m/m", 1.0, "1"),
        # unit weights written as a density or a gradient
        (1800.0, "Kg/m3", 17.65197, "kN/m3"),
        (1800.0, "KG/M 3", 17.65197, "kN/m3"),
        (18.0, "KN /CU.M", 18.0, "kN/m3"), (0.12, "kips/ft3", 18.8505, "kN/m3"),
        (9.8, "kPa/m", 9.8, "kN/m3"), (62.4, "psf/ft", 9.80226, "kN/m3"),
        (100.0, "pci", 27144.71, "kN/m3"),
        # a water analysis and a corrosivity suite
        (250.0, "mg/l", 250.0, "mg/L"), (250.0, "mg/dm3", 250.0, "mg/L"),
        (1500.0, "uS/cm", 0.15, "S/m"), (1.5, "mmhos/cm", 0.15, "S/m"),
        (120.0, "mg/Kg-dry", 120.0, "mg/kg"), (2000.0, "ohms*cm", 20.0, "ohm.m"),
        (4.0, "kΩ - cm", 40.0, "ohm.m"), (7.0, "pH units", 7.0, "1"),
        # a dynamic cone's index
        (25.0, "mm/blow", 25.0, "mm/blow"), (1.0, "in/blow", 25.4, "mm/blow"),
    ])
    def test_the_units_a_printout_and_a_sheet_add_convert(
            self, value, unit, want, si_unit):
        got = Quantity(value=value, unit=unit).to_si()
        assert got is not None, unit
        assert got.value == pytest.approx(want, rel=1e-4)
        assert got.unit == si_unit

    @pytest.mark.parametrize("unit", [
        # a newton or a blow count; minutes or a minimum; hours or a
        # height; seconds or a plural; days or a diameter
        "N", "min", "h", "s", "d",
        # short or metric; a moment or an energy; an offset, not a factor
        "tons", "tons/ft", "ft-lb", "°F",
        # needs the ion's weight
        "meq/L",
        # garbled on the page, or not a unit
        "Kg/m13", "KG/CM*", "kPa*", "H2 kN/m", "kN or kN/m", "PP",
        # "cum" is as often cumulative as cubic metres
        "cum",
    ])
    def test_an_ambiguous_unit_stays_unconverted(self, unit):
        assert to_si(1.0, unit) is None

    # RI-20 review: the factors the table above does not pin, each worked
    # from its definition here rather than from the module's constants
    # (1 lbf = 4.4482216 N, 1 ft = 0.3048 m, 1 in = 0.0254 m, 1 daN = 10 N).
    @pytest.mark.parametrize("value,unit,want,si_unit", [
        (1.0, "kcf", 4.4482216 / 0.3048 ** 3, "kN/m3"),
        (1.0, "ksf/ft", 4.4482216 / 0.3048 ** 3, "kN/m3"),
        (1.0, "daN/cm2", 10.0 / 1e-4 / 1000.0, "kPa"),
        (1.0, "ft/s", 0.3048, "m/s"),
        (1.0, "kips/ft", 4.4482216 / 0.3048, "kN/m"),
        (1.0, "kip-in2", 4.4482216 * 0.0254 ** 2, "kN.m2"),
        (1.0, "lb-in2", 0.0044482216 * 0.0254 ** 2, "kN.m2"),
        (1.0, "psi/in", 0.0044482216 / 0.0254 ** 3, "kN/m3"),
        (1.0, "dS/m", 0.1, "S/m"),
        (1.0, "kg/l", 9.80665, "kN/m3"),
    ])
    def test_the_factors_worked_from_their_definitions(
            self, value, unit, want, si_unit):
        got = Quantity(value=value, unit=unit).to_si()
        assert got is not None, unit
        assert got.value == pytest.approx(want, rel=1e-6)
        assert got.unit == si_unit

    @pytest.mark.parametrize("unit,mass", [
        ("kg/m3", True), ("Kg/m³", True), ("g/cm3", True), ("Mg/m3", True),
        ("t/m3", True), ("kg/l", True), ("pcf", False), ("kN/m3", False),
        ("kPa/m", False), ("", False)])
    def test_a_mass_per_volume_is_named(self, unit, mass):
        from report_ingest.model import is_mass_density
        assert is_mass_density(unit) is mass

    @pytest.mark.parametrize("spelling", [
        "FT", "ft.", "Feet", "lb/ft3", "KG / CM2", "kg/cm²", "kN/m^3",
        "Percent", "blows per foot",
    ])
    def test_spellings_a_column_header_uses_are_recognised(self, spelling):
        assert Quantity(value=1.0, unit=spelling).to_si() is not None

    def test_converting_carries_the_provenance_through(self):
        prov = Provenance(page=37, bbox=(1, 2, 3, 4), method="grid")
        got = Quantity(value=10.0, unit="ft", prov=prov).to_si()
        assert got.prov is not None and got.prov.page == 37
        assert got.prov.bbox == (1.0, 2.0, 3.0, 4.0)

    def test_every_table_entry_names_a_known_kind(self):
        from report_ingest.model import SI_UNITS
        for unit, (kind, factor) in UNIT_TO_SI.items():
            assert kind in SI_UNITS, unit
            assert factor > 0, unit


class TestProvenance:
    def test_the_method_vocabulary_is_closed(self):
        with pytest.raises(ValidationError):
            Provenance(page=1, method="guessed")

    def test_confidence_is_bounded(self):
        with pytest.raises(ValidationError):
            Provenance(page=1, confidence=1.5)

    def test_a_value_read_by_looking_says_so(self):
        p = Provenance(page=4, method="vision", confidence=0.8,
                       note="sample symbol")
        assert p.method == "vision" and p.bbox is None


class TestRecord:
    def test_the_json_schema_exports_and_names_its_version(self):
        schema = record_json_schema()
        assert schema["title"] == "ReportRecord"
        assert schema["x-schema-version"] == SCHEMA_VERSION
        # It has to survive the trip a consumer will make it take.
        again = json.loads(json.dumps(schema))
        assert set(again["properties"]) == {
            "schema_version", "document", "page_labels", "project", "general",
            "natural_hazards", "narrative", "investigations", "lab_tests",
            "calculations", "calcs", "qa", "bound_documents", "parent"}

    def test_the_schema_carries_the_investigation_shape(self):
        schema = record_json_schema()
        defs = schema["$defs"]
        for name in ("Investigation", "Layer", "Sample", "SPT", "WaterLevel",
                     "Quantity", "Provenance", "LabTest", "QAEntry"):
            assert name in defs, name
        inv = defs["Investigation"]["properties"]
        for field in ("investigation_id", "kind", "depth_unit", "units_known",
                      "layers", "samples", "spt", "water", "drilling",
                      "pages", "source_report"):
            assert field in inv, field

    def test_an_unknown_top_level_field_is_refused(self):
        with pytest.raises(ValidationError):
            ReportRecord(investigation=[])

    def test_counts_report_what_is_in_the_record(self):
        record = ReportRecord(
            document=DocumentFacts(report_id="R36", n_pages=94),
            investigations=[Investigation(
                investigation_id="B-2", depth_unit="ft",
                layers=[Layer(top=Quantity(value=0.0, unit="ft"),
                              description="ASPHALT")],
                samples=[Sample(sample_id="1",
                                top=Quantity(value=2.5, unit="ft"))],
                spt=[SPT(depth_top=Quantity(value=10.0, unit="ft"),
                         blows=[2, 3, 4], n=7)],
                water=[WaterLevel(when="not_encountered")])],
            lab_tests=[LabTest(kind="atterberg", sample_id="1")],
            qa=[QAEntry(kind="partial", detail="one sheet unreadable")])
        assert record.counts() == {
            "investigations": 1, "layers": 1, "samples": 1, "spt": 1,
            "water_levels": 1, "cpt_points": 0, "dcp_points": 0,
            "lab_tests": 1, "calculations": 0, "calcs": 0, "qa": 1,
            "qa_open": 0, "qa_changed": 0, "qa_note": 1}
        assert record.investigation("B-2") is not None
        assert record.investigation("B-9") is None

    def test_the_qa_by_rank_switch_is_off_until_it_is_set(self, monkeypatch):
        from report_ingest.model import (
            QA_BY_RANK_ENV, qa_by_rank_enabled, use_qa_by_rank,
        )
        monkeypatch.delenv(QA_BY_RANK_ENV, raising=False)
        assert qa_by_rank_enabled() is False
        with use_qa_by_rank():
            assert qa_by_rank_enabled() is True
        monkeypatch.setenv(QA_BY_RANK_ENV, "1")
        assert qa_by_rank_enabled() is True
        with use_qa_by_rank(False):
            assert qa_by_rank_enabled() is False

    def test_the_record_round_trips_through_json(self):
        record = ReportRecord(
            investigations=[Investigation(
                investigation_id="TP-1", kind="test_pit", depth_unit="m",
                samples=[Sample(top=Quantity(value=1.2, unit="m"),
                                kind="bulk", fines_percent=42.0)])])
        blob = record.model_dump_json()
        again = ReportRecord.model_validate_json(blob)
        assert again.investigations[0].samples[0].fines_percent == 42.0
        assert again.investigations[0].kind == "test_pit"


class TestCalculations:
    """What a calculation printout becomes, and what the record refuses."""

    def test_a_calculation_validates_and_round_trips(self):
        calc = Calculation(
            kind="settlement", program="PILEWORKS 2024 Version 11.2.3",
            method="Schmertmann strain influence",
            subject="Column footing F-3",
            inputs=[NamedQuantity(
                name="Footing Width B (ft)",
                value=Quantity(value=8.5, unit="ft"),
                prov=Provenance(page=43, method="tables", confidence=0.85))],
            results=[NamedQuantity(
                name="Total Cumulative Settlement (inches)",
                value=Quantity(value=0.52, unit="in"))],
            summary="A spread-footing settlement.", pages=[41, 42, 43],
            source_report="R18",
            unsettled=[{"what": "the modulus column", "why": "cut off"}])

        again = Calculation.model_validate_json(calc.model_dump_json())

        assert again.results[0].value.unit == "in"
        assert again.results[0].value.to_si().unit == "m"
        assert again.inputs[0].prov.page == 43
        assert again.result("cumulative settlement").value.value == 0.52
        assert again.result("nothing of the kind") is None

    def test_a_kind_outside_the_controlled_list_is_refused(self):
        with pytest.raises(ValidationError):
            Calculation(kind="pile_driveability")

    def test_a_named_quantity_needs_a_number_or_words(self):
        with pytest.raises(ValidationError, match="not a value"):
            NamedQuantity(name="Seismic Site Class")
        assert NamedQuantity(name="Seismic Site Class", text="C").text == "C"

    def test_a_program_the_pages_do_not_name_is_null_not_empty(self):
        """Null and "" would be the same to a reader of the JSON and they
        are not the same claim, so the field is Optional and defaults to
        None."""
        assert Calculation(kind="other").program is None

    def test_the_record_carries_the_calculations_and_counts_them(self):
        record = ReportRecord(calculations=[
            Calculation(kind="slope_stability", pages=[386])])
        again = ReportRecord.model_validate_json(record.model_dump_json())

        assert again.calculations[0].kind == "slope_stability"
        assert again.counts()["calculations"] == 1

    def test_the_si_walk_reaches_a_calculations_numbers(self):
        calc = Calculation(kind="settlement", results=[NamedQuantity(
            name="Total settlement", value=Quantity(value=1.0, unit="in"))])
        numbers = dict((path, (round(value, 6), unit))
                       for path, value, unit in si_numbers(calc))

        assert numbers["results[0].value"] == (0.0254, "m")


class TestSPT:
    def test_a_refusal_stays_the_string_the_log_printed(self):
        rec = SPT(depth_top=Quantity(value=20.0, unit="ft"),
                  blows=[12, 30, "50/5\""], refusal=True)
        assert rec.blows[2] == '50/5"'
        assert rec.n is None

    def test_n_is_only_what_the_log_prints(self):
        rec = SPT(depth_top=Quantity(value=5.0, unit="ft"), blows=[6, 9, 14])
        assert rec.n is None


class TestInvestigation:
    def test_units_unknown_is_a_first_class_state(self):
        inv = Investigation(investigation_id="B-1", depth_unit="",
                            units_known=False)
        assert inv.units_known is False and inv.depth_unit == ""

    def test_header_fields_the_model_has_no_slot_for_are_kept(self):
        inv = Investigation(investigation_id="B-1",
                            fields={"bit": "4-inch tricone",
                                    "hoja": "1 de 2"})
        assert inv.fields["hoja"] == "1 de 2"

    def test_the_kind_vocabulary_is_closed(self):
        with pytest.raises(ValidationError):
            Investigation(investigation_id="B-1", kind="trench")


# ---------------------------------------------------------------------------
# the typed laboratory results (WP3)
# ---------------------------------------------------------------------------

class TestLabResults:
    def test_a_result_rebuilds_its_own_class_off_disk(self):
        from report_ingest.model import AtterbergResult, GradationResult

        test = LabTest(kind="atterberg",
                       result=AtterbergResult(ll=48, pl=22, pi=26))
        back = LabTest.model_validate_json(test.model_dump_json())
        assert isinstance(back.result, AtterbergResult)
        assert back.result.ll == 48.0
        grading = LabTest(kind="gradation",
                          result=GradationResult(fines_percent=43))
        back = LabTest.model_validate_json(grading.model_dump_json())
        assert isinstance(back.result, GradationResult)

    def test_a_result_that_does_not_answer_for_its_kind_is_refused(self):
        from report_ingest.model import AtterbergResult

        with pytest.raises(ValidationError):
            LabTest(kind="gradation", result=AtterbergResult(ll=48))

    def test_every_kind_has_a_result_class(self):
        from report_ingest.model import LabKind, RESULT_CLASS

        for kind in LabKind.__args__:
            assert kind in RESULT_CLASS, kind

    def test_one_class_answers_for_the_four_strength_kinds(self):
        from report_ingest.model import RESULT_CLASS, StrengthResult

        for kind in ("triaxial", "direct_shear", "unconfined",
                     "unconfined_rock"):
            assert RESULT_CLASS[kind] is StrengthResult

    def test_a_kind_alone_is_a_record(self):
        """A sheet whose kind is known and whose values are not."""
        test = LabTest(kind="atterberg", investigation_id="B-1")
        assert test.result is None

    def test_a_value_printed_as_words_stays_a_string(self):
        from report_ingest.model import ChemicalResult

        result = ChemicalResult(chloride="<10", pH=8.43,
                                resistivity=Quantity(value=1261,
                                                     unit="ohm-cm"))
        assert result.chloride == "<10"
        assert result.pH == 8.43
        assert result.resistivity.to_si().unit == "ohm.m"

    def test_the_lab_units_convert(self):
        for unit, si, value in (("ohm-cm", "ohm.m", 0.01),
                                ("kohm-cm", "ohm.m", 10.0),
                                ("mV", "mV", 1.0),
                                ("ppm", "mg/kg", 1.0),
                                ("cm3", "m3", 1e-6),
                                ("Mg/m3", "kN/m3", 9.80665)):
            got = Quantity(value=1.0, unit=unit).to_si()
            assert got is not None, unit
            assert got.unit == si
            assert got.value == pytest.approx(value)

    def test_si_numbers_walks_a_result_and_finds_only_numbers(self):
        from report_ingest.model import (
            SievePoint, StrengthResult, StrengthSpecimen, si_numbers,
        )

        result = StrengthResult(
            kind="triaxial", test_type="CU", description="a clay",
            phi_deg=28.5,
            specimens=[StrengthSpecimen(specimen_id="1",
                                        peak_deviator=Quantity(value=185,
                                                               unit="kPa"))])
        found = {path: value for path, value, _unit in si_numbers(result)}
        assert found["phi_deg"] == 28.5
        assert found["specimens[0].peak_deviator"] == 185.0
        assert not any("description" in path for path in found)
        assert not any("test_type" in path for path in found)
        # A unit the table cannot convert has no SI value and is not claimed.
        point = SievePoint(percent_passing=43.0,
                           size=Quantity(value=1.0, unit="furlongs"))
        paths = {path for path, _v, _u in si_numbers(point)}
        assert "percent_passing" in paths and "size" not in paths


# ---------------------------------------------------------------------------
# what a pit, a cone sounding and a dynamic probe add
# ---------------------------------------------------------------------------

class TestTheSoundings:
    """Three kinds ``InvestigationKind`` could always NAME and the record
    could not HOLD until this train."""

    @staticmethod
    def _cone():
        from report_ingest.model import CPTData, CPTPoint
        return Investigation(
            investigation_id="CPT-1", kind="cpt", depth_unit="m",
            cpt=CPTData(
                points=[CPTPoint(depth=Quantity(value=0.5, unit="m"),
                                 qc=Quantity(value=5.5, unit="MPa"),
                                 fs=Quantity(value=0.06, unit="MPa"),
                                 u2=Quantity(value=5.0, unit="kPa"),
                                 rf_percent=1.1, sbt="sand")],
                vertical_axis="elevation", datum="m TAW",
                cone_area=Quantity(value=15.0, unit="cm3"),
                qc_unit="MPa", digitised=True))

    @staticmethod
    def _probe():
        from report_ingest.model import DCPData, DCPPoint
        return Investigation(
            investigation_id="DP-1", kind="dcp", depth_unit="m",
            dcp=DCPData(
                points=[DCPPoint(depth=Quantity(value=0.2, unit="m"),
                                 blows=4.0,
                                 index=Quantity(value=21.5, unit="MPa"),
                                 cbr_percent=12.0, refusal=False)],
                increment=Quantity(value=0.2, unit="m"),
                hammer_mass=Quantity(value=63.5, unit="kg"),
                index_name="Rd (MPa)"))

    def test_a_cone_sounding_round_trips_through_json(self):
        from report_ingest.model import CPTData
        inv = self._cone()
        again = Investigation.model_validate(inv.model_dump(mode="json"))
        assert isinstance(again.cpt, CPTData)
        assert again.cpt.n_points == 1
        assert again.cpt.points[0].qc.unit == "MPa"
        assert again.cpt.vertical_axis == "elevation"

    def test_a_dynamic_probe_round_trips_through_json(self):
        from report_ingest.model import DCPData
        inv = self._probe()
        again = Investigation.model_validate(inv.model_dump(mode="json"))
        assert isinstance(again.dcp, DCPData)
        assert again.dcp.points[0].blows == 4.0
        assert again.dcp.hammer_mass.value == 63.5

    def test_a_pit_carries_its_plan_size(self):
        from report_ingest.model import PitDimensions
        inv = Investigation(
            investigation_id="TP-1", kind="test_pit", depth_unit="m",
            pit=PitDimensions(length=Quantity(value=2.5, unit="m"),
                              width=Quantity(value=1.0, unit="m"),
                              method="Backhoe"))
        again = Investigation.model_validate(inv.model_dump(mode="json"))
        assert again.pit.length.value == 2.5
        assert again.pit.depth is None

    def test_an_exploration_of_another_kind_carries_none_of_them(self):
        boring = Investigation(investigation_id="B-1", kind="boring")
        assert boring.pit is None and boring.cpt is None
        assert boring.dcp is None

    def test_the_counts_report_the_series(self):
        record = ReportRecord(investigations=[self._cone(), self._probe()])
        counts = record.counts()
        assert counts["cpt_points"] == 1
        assert counts["dcp_points"] == 1

    def test_the_si_walk_reaches_a_soundings_numbers(self):
        numbers = dict((path, value)
                       for path, value, _unit in si_numbers(self._cone()))
        assert any(p.endswith("qc.value") or p.endswith(".qc")
                   for p in numbers)
        assert any(abs(v - 5500.0) < 1e-6 for v in numbers.values())

    def test_the_schema_carries_the_sounding_shape(self):
        schema = record_json_schema()
        names = set(schema["$defs"])
        assert {"CPTData", "CPTPoint", "DCPData", "DCPPoint",
                "PitDimensions"} <= names
        investigation = schema["$defs"]["Investigation"]["properties"]
        assert {"pit", "cpt", "dcp"} <= set(investigation)

    def test_every_new_model_is_walkable_and_forbids_extras(self):
        from report_ingest.model import (
            CPTData, CPTPoint, DCPData, DCPPoint, PitDimensions,
        )
        for cls in (PitDimensions, CPTPoint, CPTData, DCPPoint, DCPData):
            assert cls.model_config.get("extra") == "forbid", cls.__name__
            for name, field in cls.model_fields.items():
                assert field.description or name in (
                    "kind", "points", "prov"), f"{cls.__name__}.{name}"

    def test_adding_these_did_not_move_the_schema_version(self):
        # All three are optional and nothing changed meaning, so a record
        # written before them loads as one whose explorations carry none.
        from report_ingest.model import SCHEMA_VERSION
        assert SCHEMA_VERSION == "4.0"
        old = {"schema_version": "4.0",
               "investigations": [{"investigation_id": "B-1",
                                   "kind": "boring"}]}
        again = ReportRecord.model_validate(old)
        assert again.investigations[0].cpt is None
