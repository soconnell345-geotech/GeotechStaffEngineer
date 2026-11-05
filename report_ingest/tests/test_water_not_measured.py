"""Groundwater 'not measured' kept apart from 'not encountered' (BACKLOG RI-106).

A boring drilled with water or mud and no well has no groundwater reading;
that is not water below the bottom of the hole. These tests check that the
record, the floor, both log readers (today's and model first), the
reconciler, the DIGGS file and the checklist keep the two apart, and that the
prompt text sits behind ``GEOTECH_INGEST_WATER_NOT_MEASURED`` with the
default prompts unchanged. Every log page, hole name and depth here is made
up.
"""

from __future__ import annotations

import pytest

from report_ingest.model import (
    DrillingDetails, Investigation, Layer, Quantity, ReportRecord, WaterLevel,
)
from report_ingest.water_status import (
    LOG_READER_PARAGRAPH, MODEL_FIRST_SENTENCE, fluid_method, settle_water,
    says_not_measured, use_water_not_measured,
)

pymupdf = pytest.importorskip("pymupdf", reason="the log pages are drawn")


def _m(v: float) -> Quantity:
    return Quantity(value=v, unit="m")


def _hole(**over) -> Investigation:
    data = dict(investigation_id="QB-1", depth_unit="m", total_depth=_m(9.0),
                pages=[2],
                layers=[Layer(top=_m(0.0), bottom=_m(9.0),
                              description="MADE-UP SILTY SAND", uscs="SM")])
    data.update(over)
    return Investigation(**data)


# ---------------------------------------------------------------------------
# made-up log pages: the planlens fixture with its groundwater note and its
# advancement method rewritten
# ---------------------------------------------------------------------------

def _log_page(tmp_path, groundwater: str, method: str):
    from planlens.document import open_document
    from planlens.testing.loggrid_fixtures import build_imperial_log

    doc = pymupdf.open(stream=build_imperial_log().pdf, filetype="pdf")
    page = doc[0]
    for old, new, size in (("Groundwater encountered at", groundwater, 8),
                           ("Hollow Stem Auger", method, 7)):
        box = page.search_for(old)[0]
        if size == 8:
            box = pymupdf.Rect(box.x0, box.y0, box.x0 + 200, box.y1)
        page.add_redact_annot(box)
        page.apply_redactions()
        page.insert_text((box.x0, box.y1 - 2), new, fontsize=size)
    path = tmp_path / "made_up_log.pdf"
    path.write_bytes(doc.tobytes())
    doc.close()
    return open_document(str(path))


def _floor(tmp_path, groundwater, method="Hollow Stem Auger"):
    from planlens.document.loggrid import log_grid
    from report_ingest.log_floor import seed_from_grid

    doc = _log_page(tmp_path, groundwater, method)
    try:
        return seed_from_grid(log_grid(doc, [0]), [0], "RXX")
    finally:
        doc.close()


class TestTheFloorReadsThePrintedWords:
    def test_water_level_not_measured_is_not_measured(self, tmp_path):
        inv = _floor(tmp_path, "Groundwater: water level not measured",
                     "Mud Rotary")
        assert [(w.when, w.depth) for w in inv.water] == [
            ("not_measured", None)]
        assert inv.drilling.method == "Mud Rotary"

    def test_not_encountered_stays_not_encountered(self, tmp_path):
        inv = _floor(tmp_path, "Groundwater not encountered")
        assert [(w.when, w.depth) for w in inv.water] == [
            ("not_encountered", None)]

    def test_not_recorded_mud_rotary_is_not_measured(self, tmp_path):
        inv = _floor(tmp_path, "Groundwater not recorded (mud rotary)",
                     "Mud Rotary")
        assert [w.when for w in inv.water] == ["not_measured"]


# ---------------------------------------------------------------------------
# the printed words
# ---------------------------------------------------------------------------

class TestThePrintedWords:
    @pytest.mark.parametrize("text", [
        "Water level not measured",
        "Groundwater not recorded due to drilling method",
        "GWL not obtained - mud rotary",
        "Water level obscured by drilling fluid",
        "Unable to measure groundwater; hole drilled with bentonite slurry",
        "No water level readings taken",
    ])
    def test_says_not_measured(self, text):
        assert says_not_measured(text)

    @pytest.mark.parametrize("text", [
        "Groundwater not encountered",
        "Groundwater not observed",          # logs mean "none seen"
        "Dry at completion",
        "Samples not recovered below 6 m",   # no water word
        "Water encountered at 3.2 m",
    ])
    def test_does_not_say_not_measured(self, text):
        assert not says_not_measured(text)

    @pytest.mark.parametrize("method,fluid", [
        ("Mud Rotary", True), ("Wash boring", True), ("Rotary wash", True),
        ("HSA to 3 m, then drilling fluid", True), ("Water added", True),
        ("Hollow Stem Auger", False), ("Solid flight auger", False),
        ("Air rotary", False), ("Hand auger", False),
    ])
    def test_fluid_method(self, method, fluid):
        assert fluid_method(method) is fluid


# ---------------------------------------------------------------------------
# one hole, whichever reader filled it
# ---------------------------------------------------------------------------

class TestSettleWater:
    def test_a_row_whose_note_says_not_measured_is_refiled(self):
        inv = _hole(water=[WaterLevel(when="not_encountered",
                                      note="Water level not measured")])
        assert settle_water(inv)
        assert inv.water[0].when == "not_measured"

    def test_a_remark_adds_a_not_measured_row(self):
        inv = _hole(remarks="Boring backfilled with grout. Groundwater not "
                            "measured due to mud rotary drilling.")
        settle_water(inv)
        (row,) = inv.water
        assert (row.when, row.depth) == ("not_measured", None)
        assert row.prov.method == "derived" and row.prov.page == 2
        assert "not measured" in row.note

    def test_a_header_field_adds_a_not_measured_row(self):
        inv = _hole(fields={"groundwater": "N/M"})
        settle_water(inv)
        assert [w.when for w in inv.water] == ["not_measured"]

    def test_a_mud_rotary_boring_with_no_water_row_is_not_measured(self):
        inv = _hole(drilling=DrillingDetails(method="Mud Rotary"))
        lines = settle_water(inv)
        (row,) = inv.water
        assert row.when == "not_measured" and row.depth is None
        assert "Mud Rotary" in row.note and row.prov.confidence == 0.5
        assert "Mud Rotary" in lines[0]

    def test_an_auger_boring_with_no_water_row_is_left_alone(self):
        inv = _hole(drilling=DrillingDetails(method="Hollow Stem Auger"))
        assert settle_water(inv) == [] and inv.water == []

    def test_a_reading_is_never_overruled(self):
        inv = _hole(drilling=DrillingDetails(method="Mud Rotary"),
                    remarks="Water level not measured at completion",
                    water=[WaterLevel(depth=_m(3.1), when="after_hours",
                                      hours=24)])
        assert settle_water(inv) == []
        assert [w.when for w in inv.water] == ["after_hours"]

    def test_not_encountered_on_a_mud_rotary_boring_is_kept(self):
        inv = _hole(drilling=DrillingDetails(method="Mud Rotary"),
                    water=[WaterLevel(when="not_encountered")])
        assert settle_water(inv) == []
        assert [w.when for w in inv.water] == ["not_encountered"]

    def test_a_pit_is_not_given_a_fluid_method_row(self):
        inv = _hole(kind="test_pit",
                    drilling=DrillingDetails(method="Excavator, water added"))
        assert settle_water(inv) == []

    def test_the_reconciler_settles_every_hole_and_says_so(self):
        from report_ingest.reconciler import reconcile

        record = ReportRecord(investigations=[
            _hole(drilling=DrillingDetails(method="Wash boring")),
            _hole(investigation_id="QB-2",
                  water=[WaterLevel(when="not_encountered")])])
        reconcile(record)
        assert [w.when for w in record.investigations[0].water] == [
            "not_measured"]
        assert [w.when for w in record.investigations[1].water] == [
            "not_encountered"]
        (note,) = [e for e in record.qa
                   if e.where == "investigations.water"]
        assert note.kind == "note" and "QB-1" in note.values[0]


# ---------------------------------------------------------------------------
# the readers: the record field is kept; the prompt text is switched
# ---------------------------------------------------------------------------

@pytest.fixture
def imperial(tmp_path):
    from planlens.document import open_document
    from planlens.testing.loggrid_fixtures import build_imperial_log

    path = tmp_path / "log.pdf"
    path.write_bytes(build_imperial_log().pdf)
    doc = open_document(str(path))
    yield doc
    doc.close()


def _reading(**over):
    from report_ingest.log_reader import LogReading, ReadProv, ReadWater

    prov = ReadProv(page=0, bbox=(100.0, 200.0, 180.0, 212.0))
    base = dict(investigation_id="B-12", kind="boring", depth_unit="ft",
                units_known=True, total_depth=25.0,
                water=[ReadWater(depth=None, when="not_measured",
                                 note="Water level not measured",
                                 prov=prov)],
                pages_read=[0])
    base.update(over)
    return LogReading(**base)


class TestTodaysLogReader:
    def test_not_measured_from_the_model_is_kept(self, imperial):
        from report_ingest.log_reader import read_log
        from report_ingest.tests.fake_engine import FakeEngine

        result = read_log(imperial, [0], FakeEngine([{"final": _reading()}]))
        whens = [w.when for w in result.model_investigation.water]
        assert whens == ["not_measured"]
        assert not [u for u in result.unresolved
                    if "water observation" in str(u.get("what", ""))]

    def test_the_default_system_prompt_is_unchanged(self, imperial):
        from report_ingest.log_reader import LOG_READER_SYSTEM, read_log
        from report_ingest.tests.fake_engine import FakeEngine

        engine = FakeEngine([{"final": _reading()}])
        read_log(imperial, [0], engine)
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM
        assert LOG_READER_PARAGRAPH not in LOG_READER_SYSTEM

    def test_the_switch_adds_the_paragraph_after_the_water_bullet(
            self, imperial):
        from report_ingest.log_reader import LOG_READER_SYSTEM, read_log
        from report_ingest.tests.fake_engine import FakeEngine

        engine = FakeEngine([{"final": _reading()}])
        with use_water_not_measured():
            read_log(imperial, [0], engine)
        system = engine.calls[0]["system"]
        assert system != LOG_READER_SYSTEM
        assert system.replace(LOG_READER_PARAGRAPH, "") == LOG_READER_SYSTEM
        assert system.index("not an empty list.") < system.index(
            "NOT MEASURED IS NOT NOT ENCOUNTERED")

    def test_the_switch_reads_the_environment(self, monkeypatch):
        from report_ingest.water_status import water_not_measured_enabled

        monkeypatch.delenv("GEOTECH_INGEST_WATER_NOT_MEASURED",
                           raising=False)
        assert water_not_measured_enabled() is False
        monkeypatch.setenv("GEOTECH_INGEST_WATER_NOT_MEASURED", "1")
        assert water_not_measured_enabled() is True


class TestModelFirstLogReader:
    def test_the_default_instruction_is_unchanged(self):
        from report_ingest import model_first as mf

        said = mf.instruction("logs", "R00", [3])
        assert MODEL_FIRST_SENTENCE not in said
        assert said == mf.INSTRUCTIONS["logs"].format(
            where="report R00, page 3",
            pages_note=mf.page_map_sentence([3], route="pdf", whole=False))

    def test_the_switch_adds_the_sentence_after_the_water_levels(self):
        from report_ingest import model_first as mf

        off = mf.instruction("logs", "R00", [3])
        with use_water_not_measured():
            on = mf.instruction("logs", "R00", [3])
            plain = mf.instruction("logs", "R00", [3],
                                   texts=mf.PLAIN_INSTRUCTIONS)
        assert on == off.replace("and the water levels. ",
                                 "and the water levels. "
                                 + MODEL_FIRST_SENTENCE + " ")
        assert MODEL_FIRST_SENTENCE not in plain     # the measured baseline

    @staticmethod
    def _read_one_log():
        from planlens.document import open_document
        from report_ingest import model_first as mf
        from report_ingest import model_first_read as mfr
        from report_ingest.model_first_pages import PageGroup
        from report_ingest.tests.fake_engine import FakeEngine

        made = pymupdf.open()
        for n in range(3):
            made.new_page().insert_text((72, 72), f"MADE-UP PAGE {n}")
        doc = open_document(made.tobytes(), name="made_up")
        made.close()
        answer = mf.LogReading.model_validate({
            "investigation_id": "QX-1", "kind": "boring", "depth_unit": "ft",
            "total_depth": 30.0,
            "water": [{"depth": None, "when": "not_measured",
                       "note": "Water level not measured", "prov": {
                           "page": 1}}]})
        engine = FakeEngine([{"final": answer}])
        engine.accepts_pdf = True
        try:
            return mfr.read_logs(doc, [PageGroup("logs", [1], key="log-1")],
                                 engine, report_id="R00", route="auto")
        finally:
            doc.close()

    def test_not_measured_from_the_model_is_kept(self):
        got = self._read_one_log()
        record = Investigation.model_validate(got[0]["investigation"])
        assert [w.when for w in record.water] == ["not_measured"]

    def test_a_log_answer_is_saved_under_the_switch_it_was_read_with(self):
        from report_ingest import model_first as mf

        assert self._read_one_log()[0]["prompt_version"] \
            == mf.READ_PROMPT_VERSION
        with use_water_not_measured():
            on = self._read_one_log()
        assert on[0]["prompt_version"] == mf.READ_PROMPT_VERSION + "+wnm"

    @pytest.mark.parametrize("stage", ["logs", "lab", "calc", "narrative"])
    def test_a_saved_answer_resumes_only_under_its_own_switch(self, stage):
        """A log answer saved with the switch off is read again with it on,
        and the other way; the other stages' answers are current either
        way."""
        from report_ingest import model_first as mf
        from report_ingest.graph import _still_current
        from report_ingest.model_first_pages import PageGroup

        parts = [PageGroup(stage, [7])]
        off = {"pages": [7], "route": "pdf",
               "prompt_version": mf.prompt_version(stage)}
        assert off["prompt_version"] == mf.READ_PROMPT_VERSION
        assert _still_current(off, parts, "pdf", stage)
        with use_water_not_measured():
            on = {**off, "prompt_version": mf.prompt_version(stage)}
            assert _still_current(on, parts, "pdf", stage)
            assert _still_current(off, parts, "pdf", stage) is (
                stage != "logs")
        assert _still_current(on, parts, "pdf", stage) is (stage != "logs")


# ---------------------------------------------------------------------------
# DIGGS and the checklist
# ---------------------------------------------------------------------------

def _read_back(xml: str):
    from subsurface_characterization import parse_diggs
    return {inv.investigation_id: inv
            for inv in parse_diggs(content=xml).site.investigations}


class TestDiggs:
    def test_a_borehole_not_measured_is_a_remark_never_not_encountered(self):
        from report_ingest.diggs_writer import (
            NOT_MEASURED_REMARK, diggs_roundtrip_gate, diggs_schema_gate,
            write_diggs,
        )

        hole = _hole(water=[WaterLevel(when="not_measured")])
        notes = []
        xml = write_diggs([hole], notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        assert "notEncountered" not in xml
        assert "WaterStrike" not in xml
        assert NOT_MEASURED_REMARK in xml
        assert any("not measured" in s for s in notes[0].skipped)
        back = _read_back(xml)["QB-1"]
        assert back.gwl_depth_m is None
        assert back.gwl_not_encountered is False
        assert back.gwl_not_measured is True
        assert diggs_roundtrip_gate(xml, [hole])[0]

    def test_a_pit_not_measured_reads_back(self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
        )

        pit = _hole(investigation_id="QTP-1", kind="test_pit",
                    total_depth=_m(1.5),
                    layers=[Layer(top=_m(0.0), bottom=_m(1.5),
                                  description="MADE-UP FILL")],
                    water=[WaterLevel(when="not_measured")])
        xml = write_diggs([pit])
        assert diggs_schema_gate(xml)[0] is True
        back = _read_back(xml)["QTP-1"]
        assert back.gwl_not_measured is True and not back.gwl_not_encountered
        assert diggs_roundtrip_gate(xml, [pit])[0]

    def test_not_encountered_does_not_read_back_as_not_measured(self):
        from report_ingest.diggs_writer import write_diggs

        xml = write_diggs([_hole(water=[WaterLevel(when="not_encountered")])])
        back = _read_back(xml)["QB-1"]
        assert back.gwl_not_encountered is True
        assert back.gwl_not_measured is False

    def test_the_gate_says_when_not_measured_is_lost(self):
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, write_diggs,
        )

        xml = write_diggs([_hole()])
        ok, diffs = diggs_roundtrip_gate(
            xml, [_hole(water=[WaterLevel(when="not_measured")])])
        assert not ok
        assert any("'groundwater not measured' did not come back" in d
                   for d in diffs)

    def test_a_not_measured_row_with_a_depth_is_not_a_level(self):
        from report_ingest.diggs_writer import write_diggs

        xml = write_diggs([_hole(water=[WaterLevel(depth=_m(9.0),
                                                   when="not_measured")])])
        back = _read_back(xml)["QB-1"]
        assert back.gwl_depth_m is None and back.gwl_not_measured is True

    def test_the_site_model_keeps_it(self):
        from subsurface_characterization.site_model import (
            Investigation as SiteInvestigation,
        )

        inv = SiteInvestigation(investigation_id="QB-1",
                                gwl_not_measured=True)
        data = inv.to_dict()
        assert data["gwl_not_measured"] is True
        assert SiteInvestigation.from_dict(data).gwl_not_measured is True
        assert "GWL not measured" in inv.summary()
        assert "gwl_not_measured" not in SiteInvestigation(
            investigation_id="QB-2").to_dict()


class TestChecklist:
    def _record(self):
        return ReportRecord(investigations=[
            _hole(investigation_id="QB-1",
                  water=[WaterLevel(depth=_m(2.0), when="while_drilling")]),
            _hole(investigation_id="QB-2",
                  water=[WaterLevel(when="not_encountered")]),
            _hole(investigation_id="QB-3",
                  water=[WaterLevel(when="not_measured")]),
            _hole(investigation_id="QB-4")])

    def test_the_check_keeps_the_two_apart(self):
        from report_ingest.checklist import CheckContext, check_groundwater

        got = check_groundwater(CheckContext(record=self._record()), {})
        assert got["status"] == "unsure"
        detail = got["detail"]
        assert "groundwater not encountered 1 (QB-2)" in detail
        assert "NOT MEASURED" in detail and "(QB-3)" in detail
        assert "nothing recorded about water 1 (QB-4)" in detail

    def test_the_judge_item_carries_what_code_found(self):
        from report_ingest.checklist import run_checklist

        out = run_checklist(self._record())
        (row,) = [r for r in out["for_you_to_judge"]
                  if r["id"] == "completeness.groundwater"]
        assert "QB-3" in row["code_said"]
        assert not [r for r in out["code_checks"]
                    if r["id"] == "completeness.groundwater"]

    def test_with_no_record_the_judge_item_is_bare(self):
        from report_ingest.checklist import run_checklist

        (row,) = [r for r in run_checklist(None)["for_you_to_judge"]
                  if r["id"] == "completeness.groundwater"]
        assert "code_said" not in row
