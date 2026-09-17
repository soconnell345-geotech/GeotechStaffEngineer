"""The record model: units, provenance, and the schema a consumer reads."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from report_ingest.model import (
    SCHEMA_VERSION, UNIT_TO_SI, DocumentFacts, Investigation, LabTest, Layer,
    Provenance, QAEntry, Quantity, ReportRecord, SPT, Sample, WaterLevel,
    record_json_schema, to_si,
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

    def test_an_empty_unit_is_dimensionless_and_is_not_a_failure(self):
        got = Quantity(value=0.6, unit="").to_si()
        assert got is not None and got.value == pytest.approx(0.6)

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
            "schema_version", "document", "project", "general",
            "natural_hazards", "narrative", "investigations", "lab_tests",
            "calcs", "qa"}

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
            "water_levels": 1, "lab_tests": 1, "calcs": 0, "qa": 1}
        assert record.investigation("B-2") is not None
        assert record.investigation("B-9") is None

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
