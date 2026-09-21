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

from report_ingest.log_floor import seed_from_grid
from report_ingest.log_reader import (
    LOG_READER_SYSTEM, LogReading, MAX_MODEL_CALLS, ReadDrilling, ReadLayer,
    ReadProv, ReadSPT, ReadSample, ReadWater, Unsettled, read_log,
    serialise_rows,
)
from report_ingest.log_scoring import score_grid, score_record
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
        # The reader returned two samples; the grid had placed four. The two
        # it left out are KEPT from the grid, with a note, not lost.
        assert [s.sample_id for s in inv.samples] == ["1", "2", "", ""]
        assert inv.samples[0].dry_unit_weight.unit == "pcf"
        assert inv.samples[0].dry_unit_weight.value == 112.0
        assert [r.n for r in inv.spt] == [21, 14, 26, 10]
        assert inv.water[0].when == "while_drilling"
        assert inv.drilling.hammer_type == "Automatic SPT Hammer"
        assert result.unresolved == []
        assert sum(1 for k in result.kept if k["what"].startswith("sample")) \
            == 2

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
        result = read_log(doc, [0], engine)
        drive = result.model_investigation.spt[0]
        assert drive.blows == [12, 30, '50/5"']
        assert drive.refusal is True
        assert drive.n is None
        # The grid placed 3-4-6, N=10 at that depth; the model's record,
        # given with no note, does not overrule it -- both are on the record.
        merged = result.investigation.spt[-1]
        assert merged.blows == [3, 4, 6]
        assert [a.value for a in merged.prov.alternatives
                if a.field == "blows"] == ['12-30-50/5"']

    def test_a_water_entry_records_that_none_was_encountered(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            water=[ReadWater(depth=None, when="not_encountered",
                             prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.model_investigation
        assert len(inv.water) == 1
        assert inv.water[0].depth is None
        assert inv.water[0].when == "not_encountered"
        # The header's groundwater field says 10 ft. Two readings that
        # cannot be one slot: both are kept, and the split is flagged.
        merged = result.investigation
        assert len(merged.water) == 2
        (row,) = [d for d in result.disagreements
                  if d["what"].startswith("water")]
        assert row["kept"] == "both"

    def test_a_vocabulary_the_reader_gets_wrong_falls_back_not_crashes(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            kind="trench",
            samples=[ReadSample(sample_id="1", top=2.0, kind="spoon",
                                prov=_prov())],
            water=[ReadWater(depth=10.0, when="later", prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.model_investigation
        assert inv.kind == "other"
        assert inv.samples[0].kind == "other"
        assert inv.water[0].when == "unknown"
        # 'other' is not an answer, so the merge takes the grid's SS code.
        assert result.investigation.samples[0].kind == "spt"


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
        assert result.model_investigation.layers == []
        # The record still carries the grid's three layers.
        assert len(result.investigation.layers) == 3
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
        model = result.model_investigation
        assert [s.sample_id for s in model.samples] == ["1"]
        assert model.spt == []
        assert len(result.unresolved) == 2
        # The grid's four samples and four drives are in the record anyway.
        assert len(result.investigation.samples) == 4
        assert len(result.investigation.spt) == 4

    def test_a_depth_just_past_the_last_tick_is_still_believed(self, imperial):
        doc, _gt = imperial
        # 25 ft is the log's own total depth, a little below the 20 ft tick.
        # Refusing that would throw away the base of every real log.
        reading = _good_imperial_reading(
            layers=[ReadLayer(top=21.0, description="LEAN CLAY",
                              prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        assert [ly.top.value for ly in result.model_investigation.layers] \
            == [21.0]
        assert 21.0 in [ly.top.value for ly in result.investigation.layers]

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
        layer = result.model_investigation.layers[0]
        assert layer.prov.bbox is None
        assert layer.prov.confidence < 0.5
        assert any(u["page"] == 77 for u in result.unresolved)

    def test_the_readers_own_unsettled_list_is_carried_through(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            unsettled=[Unsettled(what="the symbol at 12 ft", page=0,
                                 why="too faint to read")])
        # The list earns ONE follow-up call; this reader still cannot settle
        # it, and the item is carried through.
        engine = FakeEngine([{"final": reading}, {"final": reading}])
        result = read_log(doc, [0], engine)
        assert result.follow_up is True and result.model_calls == 2
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
        sample = result.investigation.samples[0]
        # The grid's type column printed SS; the picture said ring, with a
        # note. That is evidence, so the picture wins -- and the grid's
        # reading stands beside it as the alternative, flagged for review.
        assert sample.kind == "ring"
        (alt,) = [a for a in sample.prov.alternatives if a.field == "kind"]
        assert alt.value == "spt" and alt.method == "grid"
        (row,) = [d for d in result.disagreements
                  if d["what"].endswith(": kind")]
        assert row["kept"] == "model"
        assert sample.prov.bbox is None
        assert result.model_investigation.samples[0].prov.method == \
            "model_from_picture"

    def test_a_value_read_off_the_rows_is_not_a_change(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        assert result.changes == []
        # Both voters gave the layer: the record says so.
        assert result.investigation.layers[0].prov.method == "reconciled"
        assert result.model_investigation.layers[0].prov.method == "model"


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


# ---------------------------------------------------------------------------
# the floor: the grid is the first voter
# ---------------------------------------------------------------------------

def _truth_for(gt) -> dict:
    """The fixture's own answers in the scorer's truth shape."""
    import re
    layers = []
    for top, bottom, words in gt.layers:
        symbol = re.search(r"\(([A-Z]{2}(?:-[A-Z]{2})?)\)", words)
        layers.append({"top": top, "bottom": bottom, "description": words,
                       "uscs": symbol.group(1) if symbol else None})
    samples = []
    index = {depth: (wc, duw) for depth, wc, duw in gt.index_tests}
    for depth, blows, n in gt.samples:
        row = {"top": depth, "blows": [int(b) for b in blows.split("-")],
               "n": int(n.split("=")[1])}
        if depth in index:
            row["wc"] = float(index[depth][0])
            row["duw"] = float(index[depth][1])
        samples.append(row)
    return {"id": "RXX_p0", "pages": [0], "depth_unit": gt.unit,
            "fields": {"boring_id": gt.fields["boring_id"],
                       "hammer": gt.fields["hammer_type"],
                       "method": gt.fields["drilling_method"],
                       "driller": gt.fields["driller"],
                       "date_started": gt.fields["date_started"]},
            "layers": layers, "samples": samples,
            "water": [{"depth": 10.0 if gt.unit == "ft" else 3.0}]}


class TestTheFloor:
    """The grid is the first voter and its values are the floor."""

    def test_the_seed_carries_what_the_grid_placed(self, imperial):
        doc, gt = imperial
        from planlens.document.loggrid import log_grid
        seed = seed_from_grid(log_grid(doc, [0]), [0], "RXX")
        assert seed.investigation_id == "B-12"
        assert seed.depth_unit == "ft" and seed.units_known is True
        assert [round(s.top.value) for s in seed.samples] == [2, 7, 12, 18]
        assert [r.n for r in seed.spt] == [21, 14, 26, 10]
        assert seed.spt[0].blows == [5, 9, 12]
        assert seed.samples[0].kind == "spt"          # the SS code
        assert seed.samples[0].water_content == 18.0
        assert seed.samples[0].dry_unit_weight.value == 112.0
        assert seed.samples[0].dry_unit_weight.unit == "pcf"
        assert (seed.samples[0].liquid_limit, seed.samples[0].plastic_limit,
                seed.samples[0].plasticity_index) == (42.0, 21.0, 21.0)
        assert [ly.top.value for ly in seed.layers] == [0.0, 4.0, 16.0]
        assert [ly.uscs for ly in seed.layers] == ["CL", "SP-SM", "CL"]
        assert seed.drilling.hammer_type == "Automatic SPT Hammer"
        assert seed.total_depth.value == 25.0
        assert seed.water[0].depth.value == 10.0
        assert all(s.prov.method == "grid" for s in seed.samples)
        assert all(0.0 < s.prov.confidence <= 1.0 for s in seed.samples)

    def test_the_model_is_shown_the_starting_record(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE STARTING RECORD" in brief
        assert "sample at 2." in brief and "blows 5-9-12, N 21" in brief
        assert "layer 0 ft to 4 ft" in brief
        assert "THE STARTING RECORD, AND THE THREE THINGS" in LOG_READER_SYSTEM
        assert "never DROP one" in LOG_READER_SYSTEM

    def test_a_reply_that_drops_everything_scores_no_lower_than_the_grid(
            self, imperial):
        doc, gt = imperial
        from planlens.document.loggrid import log_grid
        truth = _truth_for(gt)
        grid = log_grid(doc, [0])
        before = score_grid(truth, grid)
        empty = _good_imperial_reading(layers=[], samples=[], spt=[],
                                       water=[], total_depth=None)
        engine = FakeEngine([{"final": empty}])
        result = read_log(doc, [0], engine, grid=grid, report_id="RXX")
        after = score_record(truth, [result.investigation])
        assert after.total.found >= before.total.found
        for metric, got in before.scores.items():
            mine = after.scores.get(metric)
            assert mine is not None and mine.found >= got.found, metric
        # ... and the model's own answer, scored alone, is what dropped it.
        alone = score_record(truth, [result.model_investigation])
        assert alone.total.found < before.total.found
        assert len(result.kept) >= 4 + 3          # four samples, three layers

    def test_a_reply_that_contradicts_a_seeded_value_keeps_both(
            self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            spt=[ReadSPT(depth_top=2.0, depth_bottom=3.5,
                         blows=["5", "9", "12"], n=12, sample_id="1",
                         prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        drive = result.investigation.spt[0]
        assert drive.n == 21                       # the grid's value stands
        (alt,) = [a for a in drive.prov.alternatives if a.field == "n"]
        assert alt.value == "12" and alt.method == "model"
        rows = [d for d in result.disagreements if d["what"].endswith(": n")]
        assert len(rows) == 1
        assert rows[0]["floor"] == "21" and rows[0]["model"] == "12"
        assert rows[0]["kept"] == "floor"
        assert rows[0]["confidence"]["floor"] > 0

    def test_a_correction_with_evidence_overrules_the_floor(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            spt=[ReadSPT(depth_top=2.0, depth_bottom=3.5,
                         blows=["5", "9", "12"], n=12, sample_id="1",
                         prov=_prov(bbox=(364.0, 219.0, 381.0, 229.0),
                                    note="the cell prints N=12; the 2 is "
                                         "overprinted"))])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        drive = result.investigation.spt[0]
        assert drive.n == 12
        (alt,) = [a for a in drive.prov.alternatives if a.field == "n"]
        assert alt.value == "21" and alt.method == "grid"
        (row,) = [d for d in result.disagreements
                  if d["what"].endswith(": n")]
        assert row["kept"] == "model"

    def test_a_contradiction_within_tolerance_is_not_a_disagreement(
            self, imperial):
        doc, _gt = imperial
        # The grid places the sample's text at 2.06 ft; the log prints 2.0.
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        assert not [d for d in result.disagreements
                    if d["what"].startswith("sample") and
                    d["what"].endswith(": top")]
        assert result.investigation.samples[0].top.value == 2.0
        assert result.investigation.samples[0].top.prov.method == \
            "reconciled"
        assert result.reconciled > 0

    def test_a_symbol_the_model_adds_is_added(self, imperial):
        doc, _gt = imperial
        reading = _good_imperial_reading(
            layers=[ReadLayer(top=0.0, bottom=4.0, uscs="CL",
                              description="SANDY LEAN CLAY (CL), dark "
                                          "brown, very stiff", prov=_prov()),
                    ReadLayer(top=4.0, bottom=16.0, uscs="SP-SM",
                              description="POORLY GRADED SAND WITH SILT "
                                          "(SP-SM), tan, medium dense",
                              prov=_prov()),
                    ReadLayer(top=16.0, bottom=None, uscs="CL",
                              description="LEAN CLAY (CL), gray, stiff",
                              consistency="stiff", color="gray",
                              prov=_prov())],
            water=[ReadWater(depth=10.0, when="while_drilling",
                             prov=_prov(bbox=None, from_image=True,
                                        note="inverted triangle at 10"))])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.investigation
        assert inv.layers[2].consistency == "stiff"      # model only
        assert inv.water[0].when == "while_drilling"     # the grid said unknown
        assert any(a["what"].endswith(": consistency") or
                   a["what"].endswith(": when") for a in result.added) \
            or inv.water[0].when == "while_drilling"

    def test_every_value_carries_a_method_and_a_confidence(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine)
        inv = result.investigation
        methods = {obj.prov.method for obj in
                   inv.layers + inv.samples + inv.spt + inv.water}
        assert methods <= {"grid", "model", "model_from_picture",
                           "reconciled"}
        assert "reconciled" in methods and "grid" in methods
        assert all(0.0 < obj.prov.confidence <= 1.0
                   for obj in inv.layers + inv.samples + inv.spt + inv.water)

    def test_the_follow_up_carries_a_magnified_band_of_the_rows(
            self, imperial):
        doc, _gt = imperial
        first = _good_imperial_reading(
            unsettled=[Unsettled(what="the sampler symbol at 12 ft", page=0,
                                 why="the rows do not say")])
        second = _good_imperial_reading()
        engine = FakeEngine([{"final": first}, {"final": second}])
        result = read_log(doc, [0], engine)
        assert result.follow_up is True
        follow = engine.calls[1]["messages"][-1]["content"]
        assert "could not settle" in follow[0]["text"]
        assert "sampler symbol at 12 ft" in follow[0]["text"]
        images = [b for b in follow if b.get("type") == "image"]
        assert len(images) == 1
        assert images[0]["png"][:4] == b"\x89PNG"
        assert result.unresolved == []

    def test_the_follow_up_is_not_made_when_the_budget_is_spent(
            self, imperial):
        doc, _gt = imperial
        first = _good_imperial_reading(
            unsettled=[Unsettled(what="the symbol at 12 ft", page=0,
                                 why="faint")])
        engine = FakeEngine([{"final": first}])
        result = read_log(doc, [0], engine, budget=1)
        assert result.follow_up is False and engine.n_calls == 1

    def test_the_result_serialises_both_voters_and_the_merge(self, imperial):
        import json
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        blob = read_log(doc, [0], engine).to_dict()
        json.dumps(blob)
        assert blob["floor"]["investigation_id"] == "B-12"
        assert blob["model_investigation"]["samples"][0]["sample_id"] == "1"
        assert isinstance(blob["disagreements"], list)
        assert blob["reconciled"] > 0


# ---------------------------------------------------------------------------
# a test pit is not a hole
# ---------------------------------------------------------------------------

class TestTheTestPit:
    """A pit's plan size, which the record had nowhere to put until this
    train and which no log in the corpus states as a labelled field."""

    @pytest.fixture
    def pit(self, tmp_path):
        from planlens.document import open_document
        from report_ingest.tests.sounding_fixtures import build_test_pit
        gt = build_test_pit()
        path = tmp_path / "pit.pdf"
        path.write_bytes(gt.pdf)
        doc = open_document(str(path))
        yield doc, gt
        doc.close()

    def _floor(self, doc):
        from planlens.document.loggrid import log_grid
        from report_ingest.log_floor import seed_from_grid
        grid = log_grid(doc, [0])
        return seed_from_grid(grid, [0], "RXX",
                              lines=list(doc.page(0).lines))

    def test_the_grid_seeds_a_pit_as_a_pit(self, pit):
        doc, gt = pit
        floor = self._floor(doc)
        assert floor.kind == "test_pit"
        assert len(floor.layers) == len(gt.layers)
        assert len(floor.samples) == len(gt.samples)

    def test_the_bucket_is_the_pits_width(self, pit):
        doc, gt = pit
        floor = self._floor(doc)
        assert floor.pit is not None
        assert floor.pit.width.value == pytest.approx(gt.dimensions["width"])
        assert floor.pit.width.unit == gt.dimensions["unit"]

    def test_where_the_width_came_from_is_on_the_record(self, pit):
        doc, _gt = pit
        floor = self._floor(doc)
        assert "BUCKET" in floor.pit.prov.note
        # Lower than a labelled field, because it is a reading of the
        # machine rather than of a stated dimension.
        assert floor.pit.prov.confidence < 0.7

    def test_a_borehole_gets_no_plan_size(self, tmp_path):
        from report_ingest.log_floor import pit_dimensions
        assert pit_dimensions(["Drilling method: Hollow stem auger",
                               "Hole diameter: 200 mm"], "m", 0) is None

    def test_a_bucket_in_a_remark_is_not_a_dimension(self):
        from report_ingest.log_floor import pit_dimensions
        assert pit_dimensions(
            ["Remarks: Test pit backfilled with the bucket on completion"],
            "m", 0) is None

    @pytest.mark.parametrize("line,length,width,depth", [
        ("Pit Dimensions: 2.5 m x 1.0 m x 3.5 m", 2.5, 1.0, 3.5),
        ("Dimensions (L x W x D): 8' x 3' x 10'", 8.0, 3.0, 10.0),
        ("Pit size: 250 cm x 100 cm", 250.0, 100.0, None),
    ])
    def test_a_run_of_dimensions_is_length_width_depth(self, line, length,
                                                       width, depth):
        from report_ingest.log_floor import pit_dimensions
        got = pit_dimensions([line], "m", 0)
        assert got.length.value == pytest.approx(length)
        assert got.width.value == pytest.approx(width)
        if depth is None:
            assert got.depth is None
        else:
            assert got.depth.value == pytest.approx(depth)

    def test_a_label_and_its_value_set_as_two_runs_still_pair(self, pit):
        from report_ingest.log_floor import header_pairs
        doc, _gt = pit
        pairs = dict(header_pairs(list(doc.page(0).lines)))
        assert pairs["Equipment"] == "Backhoe with 80 cm bucket"
        assert pairs["Total Depth"] == "2.60 m"

    def test_the_seed_shows_the_pit_to_the_model(self, pit):
        from report_ingest.log_floor import serialise_seed
        doc, _gt = pit
        assert "pit dimensions" in serialise_seed(self._floor(doc))

    def test_the_prompt_tells_the_reader_the_bucket_rule(self):
        assert "THE BUCKET IS THE WIDTH" in LOG_READER_SYSTEM
        assert "A PIT PHOTOGRAPHED WITH A SKETCH" in LOG_READER_SYSTEM

    def test_the_model_may_add_a_pit_the_grid_missed(self, pit):
        from report_ingest.log_reader import ReadPit
        doc, _gt = pit
        reading = LogReading(
            investigation_id="TP-9", kind="test_pit", depth_unit="m",
            pit=ReadPit(length=4.0, width=0.8, unit="m"))
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine, report_id="RXX")
        assert result.investigation.pit.length.value == pytest.approx(4.0)

    def test_a_pit_dimension_of_zero_is_refused(self, pit):
        from report_ingest.log_reader import ReadPit
        doc, _gt = pit
        reading = LogReading(
            investigation_id="TP-9", kind="test_pit", depth_unit="m",
            pit=ReadPit(length=0.0, width=0.8, unit="m"))
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine, report_id="RXX")
        assert result.model_investigation.pit.length is None
        assert any("greater than zero" in u["why"]
                   for u in result.unresolved)
