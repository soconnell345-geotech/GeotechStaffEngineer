"""The log reader, offline: what it shows the model and what it refuses.

Every test runs the real reader over a real ``log_grid`` of a real (synthetic)
log page, with a fake engine replaying a canned structured reply. So the brief
that is asserted on is the brief a model would actually be sent, and the
refusals are the refusals that would actually happen.

The pages come from ``planlens.testing.loggrid_fixtures``: a form drawn twice,
once in feet and once in metres, plus the same form with its depth scale taken
off. Nothing in them is copied from a real log.
"""

from __future__ import annotations

import pytest

from report_ingest.log_reader import (
    LOG_READER_SYSTEM, LogReading, MAX_MODEL_CALLS, ReadDrilling, ReadLayer,
    ReadProv, ReadSPT, ReadSample, ReadWater, Unsettled, read_log,
    serialise_rows,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted

fitz = pytest.importorskip("fitz", reason="the fixtures are drawn with PyMuPDF")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _open(pdf: bytes, tmp_path):
    from planlens.document import open_document
    path = tmp_path / "log.pdf"
    path.write_bytes(pdf)
    return open_document(str(path))


@pytest.fixture
def imperial(tmp_path):
    from planlens.testing.loggrid_fixtures import build_imperial_log
    gt = build_imperial_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def metric(tmp_path):
    from planlens.testing.loggrid_fixtures import build_metric_log
    gt = build_metric_log()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def no_ruler(tmp_path):
    from planlens.testing.loggrid_fixtures import build_log_without_ruler
    gt = build_log_without_ruler()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


def _prov(page=0, bbox=(100.0, 200.0, 180.0, 212.0), from_image=False,
          note=""):
    return ReadProv(page=page, bbox=bbox, from_image=from_image, note=note)


def _good_imperial_reading(**over) -> LogReading:
    """What a careful reader would return for the imperial fixture."""
    base = dict(
        investigation_id="B-12",
        kind="boring",
        depth_unit="ft",
        units_known=True,
        elevation=104.5,
        elevation_unit="ft",
        total_depth=25.0,
        date_started="3/14/2025",
        drilling=ReadDrilling(method="Hollow Stem Auger",
                              hammer_type="Automatic SPT Hammer",
                              driller="Regional Drilling"),
        layers=[
            ReadLayer(top=0.0, bottom=4.0, uscs="CL",
                      description="SANDY LEAN CLAY (CL), dark brown, "
                                  "very stiff", prov=_prov()),
            ReadLayer(top=4.0, bottom=16.0, uscs="SP-SM",
                      description="POORLY GRADED SAND WITH SILT (SP-SM), "
                                  "tan, medium dense", prov=_prov()),
            ReadLayer(top=16.0, bottom=None, uscs="CL",
                      description="LEAN CLAY (CL), gray, stiff",
                      prov=_prov()),
        ],
        samples=[
            ReadSample(sample_id="1", top=2.0, bottom=3.5, kind="spt",
                       water_content=18.0, dry_unit_weight=112.0,
                       dry_unit_weight_unit="pcf", liquid_limit=42.0,
                       plastic_limit=21.0, plasticity_index=21.0,
                       prov=_prov()),
            ReadSample(sample_id="2", top=7.0, bottom=8.5, kind="spt",
                       prov=_prov()),
        ],
        spt=[
            ReadSPT(depth_top=2.0, depth_bottom=3.5, blows=["5", "9", "12"],
                    n=21, sample_id="1", prov=_prov()),
            ReadSPT(depth_top=7.0, depth_bottom=8.5, blows=["4", "6", "8"],
                    n=14, sample_id="2", prov=_prov()),
        ],
        water=[ReadWater(depth=10.0, when="while_drilling", prov=_prov())],
        pages_read=[0],
    )
    base.update(over)
    return LogReading(**base)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

class TestWhatTheModelIsShown:
    def test_the_brief_carries_the_rows_with_their_boxes(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "col | depth | conf | x0,y0,x1,y1 | text" in brief
        assert "5-9-12" in brief
        assert "dry_unit_weight" in brief

    def test_the_brief_carries_the_grid_layers_fields_and_warnings(
            self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "HEADER FIELDS" in brief and "boring_id = B-12" in brief
        assert "LAYERS THE FORM READER BOUND TO DEPTHS" in brief
        assert "WHAT THE FORM READER WARNS ABOUT THESE PAGES" in brief
        assert "SANDY LEAN CLAY" in brief

    def test_the_brief_states_the_unit_the_grid_settled_on(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE DEPTH UNIT THE FORM READER SETTLED ON: ft" in brief

    def test_the_page_picture_is_attached(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        blocks = engine.calls[0]["messages"][0]["content"]
        images = [b for b in blocks if b.get("type") == "image"]
        assert len(images) == 1
        assert images[0]["png"][:4] == b"\x89PNG"

    def test_a_structured_answer_is_asked_for_and_the_rules_are_sent(
            self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        assert engine.calls[0]["output_format"] is LogReading
        assert engine.calls[0]["system"] == LOG_READER_SYSTEM
        assert "NEVER INVENT A DEPTH" in LOG_READER_SYSTEM
        assert "Do NOT add drives together" in LOG_READER_SYSTEM

    def test_a_page_with_no_scale_says_so_in_its_rows(self, no_ruler):
        doc, _gt = no_ruler
        from planlens.document.loggrid import log_grid
        text = serialise_rows(log_grid(doc, [0]), [0])
        assert "NO DEPTH SCALE ON THIS PAGE" in text


# ---------------------------------------------------------------------------
# what comes back
# ---------------------------------------------------------------------------

class TestTheInvestigationItBuilds:
    def test_a_clean_reading_becomes_an_investigation(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine, report_id="RXX")
        inv = result.investigation
        assert inv.investigation_id == "B-12"
        assert inv.depth_unit == "ft" and inv.units_known is True
        assert inv.source_report == "RXX" and inv.pages == [0]
        assert [ly.top.value for ly in inv.layers] == [0.0, 4.0, 16.0]
        assert inv.layers[0].uscs == "CL"
        assert inv.layers[2].bottom is None
        assert [s.sample_id for s in inv.samples] == ["1", "2"]
        assert inv.samples[0].dry_unit_weight.unit == "pcf"
        assert inv.samples[0].dry_unit_weight.value == 112.0
        assert [r.n for r in inv.spt] == [21, 14]
        assert inv.water[0].when == "while_drilling"
        assert inv.drilling.hammer_type == "Automatic SPT Hammer"
        assert result.unresolved == []

    def test_depths_are_kept_in_the_unit_printed_never_converted(
            self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        inv = read_log(doc, [0], engine).investigation
        assert inv.total_depth.value == 25.0 and inv.total_depth.unit == "ft"
        # ... and the record can still be asked for SI when a writer needs it
        assert inv.total_depth.to_si().value == pytest.approx(7.62)

    def test_a_metric_log_reads_in_metres(self, metric):
        doc, _gt = metric
        reading = _good_imperial_reading(
            depth_unit="m", total_depth=6.5, elevation_unit="m",
            layers=[ReadLayer(top=0.0, bottom=1.2, description="FILL",
                              prov=_prov())],
            samples=[ReadSample(sample_id="1", top=0.6, kind="spt",
                                prov=_prov())],
            spt=[ReadSPT(depth_top=0.6, blows=["4", "7", "9"], n=16,
                         prov=_prov())],
            water=[])
        engine = FakeEngine([{"final": reading}])
        inv = read_log(doc, [0], engine).investigation
        assert inv.depth_unit == "m"
        assert inv.layers[0].top.unit == "m"
        assert inv.spt[0].depth_top.value == 0.6

    def test_a_refusal_stays_a_string_and_a_count_becomes_a_number(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            spt=[ReadSPT(depth_top=18.0, blows=["12", "30", "50/5\""],
                         refusal=True, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        inv = read_log(doc, [0], engine).investigation
        assert inv.spt[0].blows == [12, 30, '50/5"']
        assert inv.spt[0].refusal is True
        assert inv.spt[0].n is None

    def test_a_water_entry_records_that_none_was_encountered(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            water=[ReadWater(depth=None, when="not_encountered",
                             prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        inv = read_log(doc, [0], engine).investigation
        assert len(inv.water) == 1
        assert inv.water[0].depth is None
        assert inv.water[0].when == "not_encountered"

    def test_a_vocabulary_the_reader_gets_wrong_falls_back_not_crashes(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            kind="trench",
            samples=[ReadSample(sample_id="1", top=2.0, kind="spoon",
                                prov=_prov())],
            water=[ReadWater(depth=10.0, when="later", prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        inv = read_log(doc, [0], engine).investigation
        assert inv.kind == "other"
        assert inv.samples[0].kind == "other"
        assert inv.water[0].when == "unknown"


# ---------------------------------------------------------------------------
# the refusals -- the point of the whole build
# ---------------------------------------------------------------------------

class TestWhatItRefuses:
    def test_a_depth_past_the_ruler_is_flagged_not_accepted(self, imperial):
        doc, _gt = imperial
        # The imperial fixture's scale runs 5 to 20 ft. 250 is a decimal
        # point in the wrong place, the classic reader slip.
        reading = _good_imperial_reading(
            layers=[ReadLayer(top=250.0, description="LEAN CLAY",
                              prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert result.investigation.layers == []
        assert len(result.unresolved) == 1
        entry = result.unresolved[0]
        assert entry["value"] == 250.0
        assert entry["refused_by"] == "python"
        assert "outside what the scale" in entry["why"]

    def test_a_sample_past_the_ruler_takes_its_whole_sample_with_it(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            samples=[ReadSample(sample_id="9", top=-40.0, prov=_prov()),
                     ReadSample(sample_id="1", top=2.0, prov=_prov())],
            spt=[ReadSPT(depth_top=99.0, blows=["5"], prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert [s.sample_id for s in result.investigation.samples] == ["1"]
        assert result.investigation.spt == []
        assert len(result.unresolved) == 2

    def test_a_depth_just_past_the_last_tick_is_still_believed(self, imperial):
        doc, _gt = imperial
        # 25 ft is the log's own total depth, a little below the 20 ft tick.
        # Refusing that would throw away the base of every real log.
        reading = _good_imperial_reading(
            layers=[ReadLayer(top=21.0, description="LEAN CLAY",
                              prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert [ly.top.value for ly in result.investigation.layers] == [21.0]

    def test_a_log_with_no_scale_yields_no_depths_at_all(self, no_ruler):
        doc, _gt = no_ruler
        reading = _good_imperial_reading(units_known=False, depth_unit="")
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.investigation
        assert inv.layers == [] and inv.samples == [] and inv.spt == []
        assert inv.units_known is False
        # ... and the header fields survive: the page still says whose log it is
        assert inv.investigation_id == "B-12"
        assert inv.drilling.hammer_type == "Automatic SPT Hammer"
        assert all(u["refused_by"] == "python" for u in result.unresolved)
        assert any("no depth scale was found" in u["why"]
                   for u in result.unresolved)

    def test_provenance_naming_a_page_outside_this_log_loses_its_box(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            layers=[ReadLayer(top=4.0, description="SAND",
                              prov=_prov(page=77))])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        layer = result.investigation.layers[0]
        assert layer.prov.bbox is None
        assert layer.prov.confidence < 0.5
        assert any(u["page"] == 77 for u in result.unresolved)

    def test_the_readers_own_unsettled_list_is_carried_through(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            unsettled=[Unsettled(what="the symbol at 12 ft", page=0,
                                 why="too faint to read")])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert any(u["refused_by"] == "reader"
                   and "symbol" in u["what"] for u in result.unresolved)

    def test_a_percentage_out_of_range_is_clamped_not_fatal(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            samples=[ReadSample(sample_id="1", top=2.0, recovery_percent=140.0,
                                rqd_percent=-3.0, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        inv = read_log(doc, [0], engine).investigation
        assert inv.samples[0].recovery_percent == 100.0
        assert inv.samples[0].rqd_percent == 0.0


class TestWhatItRecordsAsAChange:
    def test_a_value_read_off_the_picture_is_recorded_as_one(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            samples=[ReadSample(sample_id="1", top=2.0, kind="ring",
                                prov=_prov(bbox=None, from_image=True,
                                           note="ring symbol, not a spoon"))])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert len(result.changes) == 1
        assert result.changes[0]["why"] == "ring symbol, not a spoon"
        prov = result.investigation.samples[0].prov
        assert prov.method == "vision" and prov.bbox is None

    def test_a_value_read_off_the_rows_is_not_a_change(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        assert result.changes == []
        assert result.investigation.layers[0].prov.method == "grid"


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_one_sheet_costs_one_call(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        assert engine.n_calls == 1 and result.model_calls == 1
        assert result.cost["calls"] == 1

    def test_a_reader_with_pages_left_gets_a_second_call(self, imperial):
        doc, _gt = imperial
        first = _good_imperial_reading(pages_left=[0])
        second = _good_imperial_reading(pages_left=[])
        engine = FakeEngine([{"final": first}, {"final": second}])
        result = read_log(doc, [0], engine)
        assert result.model_calls == 2
        follow_up = engine.calls[1]["messages"][-1]["content"][0]["text"]
        assert "Return the WHOLE log again" in follow_up

    def test_a_reader_that_makes_no_progress_is_stopped(self, imperial):
        doc, _gt = imperial
        stuck = _good_imperial_reading(pages_left=[0])
        # Four turns on the script; the reader must stop after the second,
        # because the second says exactly what the first did.
        engine = FakeEngine([{"final": stuck}] * 4)
        result = read_log(doc, [0], engine)
        assert result.model_calls == 2

    def test_the_budget_is_never_more_than_the_ceiling(self, imperial):
        doc, _gt = imperial
        readings = [_good_imperial_reading(pages_left=[0], total_depth=25 - i)
                    for i in range(10)]
        engine = FakeEngine([{"final": r} for r in readings])
        result = read_log(doc, [0], engine, budget=99)
        assert result.model_calls <= MAX_MODEL_CALLS

    def test_pages_the_reader_never_reached_are_unresolved(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(pages_left=[0])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine, budget=1)
        assert engine.n_calls == 1
        assert any(u["refused_by"] == "budget" for u in result.unresolved)

    def test_no_structured_answer_is_an_error_not_an_empty_log(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"text": "I could not read this page."}])
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_log(doc, [0], engine)

    def test_no_pages_is_refused_before_a_call_is_spent(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([])
        with pytest.raises(ValueError, match="at least one page"):
            read_log(doc, [], engine)
        assert engine.n_calls == 0

    def test_a_second_unasked_call_would_exhaust_the_script(self, imperial):
        """The fake engine is strict on purpose: one call means one call."""
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        with pytest.raises(ScriptExhausted):
            engine.complete([], output_format=LogReading)
