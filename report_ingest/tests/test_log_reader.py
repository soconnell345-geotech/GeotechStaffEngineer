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

pymupdf = pytest.importorskip("pymupdf", reason="the fixtures are drawn with PyMuPDF")


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

    def test_a_phone_shaped_item_title_is_not_quoted(self, imperial):
        # RI-156: planlens read a letterhead's telephone line as an id.
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine, item_title="TÉL-55")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "The document titles it" not in brief

    def test_a_hole_item_title_is_still_quoted(self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        read_log(doc, [0], engine, item_title="BH-12")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "The document titles it: BH-12" in brief

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

    def test_a_log_with_no_scale_keeps_only_the_depths_it_prints(
            self, no_ruler):
        # No ruler, and no depth printed in a column, a sample or a
        # description: the only depths on this page are the two its header
        # prints, "25 ft" and "Groundwater encountered at 10 ft" (RI-18).
        # Those stand; every row depth the reader returns is refused.
        doc, _gt = no_ruler
        reading = _good_imperial_reading(units_known=False, depth_unit="")
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.investigation
        assert inv.layers == [] and inv.samples == [] and inv.spt == []
        assert inv.depth_unit == "ft"                 # printed with the depths
        assert inv.total_depth is not None and inv.total_depth.value == 25.0
        assert 10.0 in [w.depth.value for w in inv.water if w.depth]
        # ... and the header fields survive: the page still says whose log it is
        assert inv.investigation_id == "B-12"
        assert inv.drilling.hammer_type == "Automatic SPT Hammer"
        scale = [u for u in result.unresolved if u.get("flag") == "no_ruler"]
        assert len(scale) == 1 and "could be fitted" in scale[0]["why"]
        refused = [u for u in result.unresolved
                   if u.get("refused_by") == "python"]
        assert sorted(u["value"] for u in refused) == [0.0, 2.0, 2.0, 4.0,
                                                       7.0, 7.0, 10.0, 16.0]
        # The rows' depths are printed nowhere; the water depth is printed,
        # in ft, but this reading gave no unit, so it does not stand on it.
        water = [u for u in refused if u["value"] == 10.0]
        assert "the reader gave no unit" in water[0]["why"]
        assert all("nothing printed on it reads" in u["why"]
                   for u in refused if u["value"] != 10.0)
        assert len(result.unresolved) == len(scale) + len(refused)

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

    def test_a_percentage_out_of_range_is_kept_out_not_fatal(self, imperial):
        """RI-89: never clamped. A recovery of 140 is a reading a log may
        print (FHWA-NHI-06-088 p. 3-44) that the 0-100 field cannot hold;
        an RQD of -3 is refused. Each says so, and the log still builds."""
        doc, _gt = imperial
        reading = _good_imperial_reading(
            samples=[ReadSample(sample_id="1", top=2.0, recovery_percent=140.0,
                                rqd_percent=-3.0, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, [0], engine)
        inv = result.investigation
        assert inv.samples[0].recovery_percent is None
        assert inv.samples[0].rqd_percent is None
        rows = {u["what"]: u for u in result.unresolved}
        recovery = rows["sample 1 recovery percent"]
        assert recovery["value"] == 140.0
        assert ("recoveries greater than 100 percent are recorded "
                "(FHWA-NHI-06-088 Vol. I p. 3-44)") in recovery["why"]
        assert recovery["why"].endswith("listed here as read")
        assert rows["sample 1 RQD"]["why"] == (
            "-3 is not a percentage; refused rather than accepted")


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

    def test_drawn_sample_boxes_seat_the_rows(self, imperial):
        """RI-32: planlens' sample intervals (PL-24) give the floor's rows
        a top, a bottom and, where the log prints none, the sample id; a
        grid with no samples seeds exactly as before."""
        doc, _gt = imperial
        from planlens.document import loggrid
        if not hasattr(loggrid, "Sample"):
            pytest.skip("planlens without PL-24's Sample (released after 0.13.0)")
        Sample, log_grid = loggrid.Sample, loggrid.log_grid
        grid = log_grid(doc, [0])
        before = seed_from_grid(grid, [0], "RXX")
        grid.samples = [
            Sample(page=0, top=t - 0.5, bottom=t + 1.5, label=f"S-{i + 1}",
                   plus_minus=0.05)
            for i, t in enumerate(s.top.value for s in before.samples)]
        after = seed_from_grid(grid, [0], "RXX")
        for b, a in zip(before.samples, after.samples):
            assert b.bottom is None and b.sample_id == ""
            assert a.top.value == pytest.approx(b.top.value - 0.5, abs=0.1)
            assert a.bottom.value == pytest.approx(b.top.value + 1.5, abs=0.1)
            assert a.sample_id.startswith("S-")
            assert "sample box drawn" in a.prov.note
        assert [r.n for r in after.spt] == [21, 14, 26, 10]   # nothing else moved
        grid.samples = []
        again = seed_from_grid(grid, [0], "RXX")
        assert [(s.top.value, s.bottom) for s in again.samples] ==             [(s.top.value, s.bottom) for s in before.samples]

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

    def test_a_scored_run_keeps_the_three_records_and_rescores_the_same(
            self, imperial):
        """``score_one_log`` keeps the merged record, the model's and the
        floor's in the run file, and re-scoring that file with no model
        gives the numbers the run printed."""
        import json
        from report_ingest.log_scoring import rescore_saved, score_one_log
        doc, gt = imperial
        truth = _truth_for(gt)
        empty = _good_imperial_reading(layers=[], samples=[], spt=[],
                                       water=[], total_depth=None)
        engine = FakeEngine([{"final": empty}])
        _before, after = score_one_log(truth, doc, engine, report_id="RXX")
        blob = json.loads(json.dumps(after.to_dict()))
        assert blob["record"] and blob["model_record"] and blob["floor_record"]
        assert blob["kept"] > 0
        again = rescore_saved(truth, blob)
        assert again["overall"]["found"] == blob["overall"]["found"]
        assert again["model_alone"]["overall"]["found"] == \
            blob["model_alone"]["overall"]["found"]
        # the floor scored alone is what the merge kept: nothing was lost
        assert again["floor_alone"]["overall"]["found"] <= \
            again["overall"]["found"]
        # ... and every run scores it, not only a rescore (brief 4: the
        # RESULTS could not say whether the floor ever held the index)
        assert blob["floor_alone"] == again["floor_alone"]

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


class TestRI37PrintedNAndDriveLengths:
    """RI-37: the reader carries the N as printed and each drive's length."""

    def test_a_model_row_with_a_printed_refusal_and_unequal_drives_lands(
            self, metric):
        doc, _gt = metric
        reading = _good_imperial_reading(
            depth_unit="m", total_depth=6.5, elevation_unit="m",
            layers=[ReadLayer(top=0.0, bottom=1.2, description="FILL",
                              prov=_prov())],
            samples=[],
            spt=[ReadSPT(depth_top=5.0, blows=["20", "50/8cm"],
                         n_printed=">50", drive_lengths=[15.0, 8.0],
                         drive_length_unit="cm", refusal=True,
                         prov=_prov())],
            water=[])
        result = read_log(doc, [0], FakeEngine([{"final": reading}]))
        rec = [r for r in result.investigation.spt
               if abs(r.depth_top.value - 5.0) < 0.01][0]
        assert rec.n_printed == ">50" and rec.n is None and rec.refusal
        assert [(q.value, q.unit) for q in rec.increments] == [
            (15.0, "cm"), (8.0, "cm")]

    def test_drive_lengths_that_do_not_line_up_are_left_out(self, metric):
        doc, _gt = metric
        reading = _good_imperial_reading(
            depth_unit="m", total_depth=6.5, elevation_unit="m",
            layers=[ReadLayer(top=0.0, bottom=1.2, description="FILL",
                              prov=_prov())],
            samples=[],
            spt=[ReadSPT(depth_top=5.0, blows=["4", "6", "8"],
                         drive_lengths=[15.0, 15.0], drive_length_unit="cm",
                         prov=_prov())],
            water=[])
        result = read_log(doc, [0], FakeEngine([{"final": reading}]))
        rec = [r for r in result.investigation.spt
               if abs(r.depth_top.value - 5.0) < 0.01][0]
        assert rec.increments == []

    def test_a_printed_n_that_differs_from_the_floors_keeps_both(self):
        from report_ingest.log_floor import merge_investigations
        from report_ingest.model import Investigation, Provenance, SPT
        from report_ingest.model import Quantity

        def inv(n_printed, method):
            return Investigation(
                investigation_id="B-1", depth_unit="m", units_known=True,
                spt=[SPT(depth_top=Quantity(value=5.0, unit="m"),
                         n_printed=n_printed, refusal=True,
                         prov=Provenance(page=0, method=method))])
        merged, log = merge_investigations(inv(">50", "grid"),
                                           inv("50/3\"", "model"))
        rows = [d for d in log.disagreements if d["what"].endswith(
            ": n_printed")]
        assert len(rows) == 1
        assert rows[0]["floor"] == ">50" and rows[0]["model"] == "50/3\""
        assert rows[0]["kept"] == "floor"
        (alt,) = [a for a in merged.spt[0].prov.alternatives
                  if a.field == "n_printed"]
        assert alt.value == "50/3\""

    def test_a_model_only_increment_list_is_added_and_written(self):
        from report_ingest.diggs_writer import write_diggs
        from report_ingest.log_floor import merge_investigations
        from report_ingest.model import Investigation, Provenance, SPT
        from report_ingest.model import Quantity
        floor = Investigation(
            investigation_id="B-1", depth_unit="m", units_known=True,
            spt=[SPT(depth_top=Quantity(value=3.0, unit="m"),
                     blows=[5, 7, 9],
                     prov=Provenance(page=0, method="grid"))])
        model = Investigation(
            investigation_id="B-1", depth_unit="m", units_known=True,
            spt=[SPT(depth_top=Quantity(value=3.0, unit="m"),
                     blows=[5, 7, 9],
                     increments=[Quantity(value=20, unit="cm"),
                                 Quantity(value=15, unit="cm"),
                                 Quantity(value=15, unit="cm")],
                     prov=Provenance(page=0, method="model"))])
        merged, _log = merge_investigations(floor, model)
        assert [q.value for q in merged.spt[0].increments] == [20, 15, 15]
        merged.x, merged.y = None, None
        xml = write_diggs([merged])
        assert "<diggs_geo:penetration uom=\"m\">0.2</" in xml


# ---------------------------------------------------------------------------
# the builder's check mode (RI-87 piece 6, plan section 8.6)
# ---------------------------------------------------------------------------

def _ranked(rows):
    """The QA entries the reconciler makes of ``rows``, with their ranks."""
    from report_ingest.reconciler import _carry_unresolved
    qa = []
    _carry_unresolved(qa, rows)
    return qa


def _flag_build(reading, window=None, pages=(10, 11), **kw):
    from report_ingest.log_reader import _Builder
    builder = _Builder(reading, window, list(pages), "R-made-up",
                       checks="flag", **kw)
    return builder, builder.build()


class TestChecksDefaultIsMaster:
    """``checks`` left out builds exactly what master built, on made-up
    answers covering every check (the snapshot was written on master)."""

    def test_every_log_case_builds_as_on_master(self):
        from report_ingest.tests.checks_fixtures import (
            build_default, load_snapshot, log_cases,
        )
        now, then = build_default(), load_snapshot()
        for case in log_cases():
            assert now[f"log/{case}"] == then[f"log/{case}"], case

    def test_refuse_named_is_the_default(self):
        from report_ingest.log_reader import _Builder
        from report_ingest.tests.checks_fixtures import (
            LOG_PAGES, LOG_WINDOW, log_window_reading,
        )
        a = _Builder(log_window_reading(), LOG_WINDOW, LOG_PAGES, "R")
        b = _Builder(log_window_reading(), LOG_WINDOW, LOG_PAGES, "R",
                     checks="refuse")
        assert a.build() == b.build()
        assert a.unresolved == b.unresolved and a.changes == b.changes

    def test_an_unknown_mode_is_refused(self):
        from report_ingest.log_reader import _Builder
        from report_ingest.tests.checks_fixtures import log_window_reading
        with pytest.raises(ValueError, match="checks must be one of"):
            _Builder(log_window_reading(), None, [10], "R", checks="lenient")


def _log_case(case, **kw):
    from report_ingest.tests.checks_fixtures import log_cases
    reading, window, pages, no_ruler = log_cases()[case]
    return _flag_build(reading, window, pages, no_ruler=no_ruler, **kw)


class TestChecksFlag:

    def test_a_depth_outside_the_window_is_kept_and_flagged(self):
        builder, inv = _log_case("window")
        # Today the third layer, whose top is past the ruler, is dropped
        # whole; here it stays, at the flag confidence.
        assert [ly.top.value for ly in inv.layers] == [0.0, 4.0, 24.0]
        assert inv.layers[1].bottom.value == 24.0
        assert inv.layers[1].prov.confidence == 0.5
        assert [s.sample_id for s in inv.samples] == ["S-1", "S-3", "S-4"]
        assert inv.samples[1].prov.confidence == 0.5
        assert inv.samples[0].prov.confidence == 0.9
        assert [r.depth_top.value for r in inv.spt] == [2.0, 22.0]
        rows = [u for u in builder.unresolved
                if "outside what the scale" in u["why"]]
        assert rows and all("kept at confidence 0.5" in u["why"]
                            and "(-0.5 to 20.5)" in u["why"] for u in rows)
        assert {e.rank for e in _ranked(rows)} == {"open"}
        assert {e.kind for e in _ranked(rows)} == {"out_of_range"}

    def test_water_is_checked_against_the_hole_not_the_window(self):
        """RI-76 (5): water depths printed in a table on the sheet lay
        outside the ruler's window and were refused. Here the hole bounds
        them: 25 ft (in a 30 ft hole) stands, 33 ft and a 40 ft cave are
        kept and flagged."""
        builder, inv = _log_case("window")
        depths = [(w.depth.value, w.prov.confidence) for w in inv.water]
        assert depths == [(12.0, 0.9), (25.0, 0.9), (33.0, 0.5), (10.0, 0.9)]
        assert inv.water[1].casing_depth.value == 18.0
        assert inv.water[2].caved_depth.value == 40.0
        below = [u for u in builder.unresolved
                 if "below the hole's total depth 30" in u["why"]]
        assert [u["value"] for u in below] == [33.0, 40.0]
        # nothing about water mentions the ruler's window
        assert not any(u["what"].startswith("water")
                       and "scale" in u["why"] for u in builder.unresolved)

    def test_the_window_slack_carries_to_the_hole_bottom(self):
        builder, inv = _log_case("window", slack=3.5)
        # 33 ft is within the slack now; the 40 ft cave of the same
        # observation is not, and the observation carries it.
        assert [u["value"] for u in builder.unresolved
                if "total depth" in u["why"]] == [40.0]
        assert [w.prov.confidence for w in inv.water] == [0.9, 0.9, 0.5, 0.9]

    def test_water_with_no_total_depth_is_kept_with_one_note(self):
        reading = LogReading.model_validate({
            "investigation_id": "B-1", "depth_unit": "ft",
            "water": [{"depth": 55.0, "prov": {"page": 10}},
                      {"depth": 50.0, "caved_depth": 52.0,
                       "prov": {"page": 10}}]})
        builder, inv = _flag_build(reading, (0.0, 20.0))
        assert [w.depth.value for w in inv.water] == [55.0, 50.0]
        assert inv.water[0].prov.confidence == 0.9
        (note,) = builder.unresolved
        assert note == {
            "what": "total depth", "flag": "no_total_depth",
            "why": "the 3 water, casing or caved depth(s) are as read, not "
                   "checked against the hole's total depth, which the log "
                   "does not give"}
        assert _ranked([note])[0].rank == "note"

    def test_no_ruler_keeps_every_depth_with_one_note(self):
        builder, inv = _log_case("no_ruler")
        assert [ly.top.value for ly in inv.layers] == [0.0, 2.5]
        assert inv.samples[0].top.value == 3.0
        assert inv.water[0].depth.value == 4.2
        assert all(ly.prov.confidence == 0.9 for ly in inv.layers)
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "no_ruler"]
        # 4.2 m of water is checked against the 6 m hole, not counted
        assert note["why"] == ("no depth scale found; the 5 depth(s) of "
                               "layers, samples and drives are as read, not "
                               "checked against a ruler")
        (entry,) = _ranked([note])
        assert entry.rank == "note"

    def test_a_pit_with_no_scale_is_the_same(self):
        builder, inv = _log_case("pit")
        assert [ly.top.value for ly in inv.layers] == [0.0]
        assert builder.changes == []          # no per-depth pit changes
        notes = [u for u in builder.unresolved
                 if u.get("flag") == "no_ruler"]
        assert len(notes) == 1

    def test_a_pit_dimension_with_no_unit_is_kept_with_one_note(self):
        """Under refuse each is refused ("the number has no meaning"); under
        flag they are kept as read, as the total depth is, with one note,
        and a water depth with no total depth to check it is counted in the
        no-ruler note."""
        builder, inv = _log_case("pit_no_unit")
        assert (inv.pit.length.value, inv.pit.length.unit) == (3.0, "")
        assert inv.pit.width.value == 2.0 and inv.pit.depth is None
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "no_unit"]
        assert note["page"] == 10
        assert "length 3, width 2" in note["why"]
        assert note["why"].endswith("kept as read, in no unit")
        (ruler,) = [u for u in builder.unresolved
                    if u.get("flag") == "no_ruler"]
        assert ruler["why"] == (
            "no depth scale found; the 2 depth(s) of layers, samples and "
            "drives are as read, not checked against a ruler; the 1 water, "
            "casing or caved depth(s) are as read, not checked against the "
            "hole's total depth, which the log does not give")
        assert {e.rank for e in _ranked([note, ruler])} == {"note"}

    def test_impossible_depths_stay_out_with_what_was_read(self):
        builder, inv = _log_case("window")
        assert "S-2" not in [s.sample_id for s in inv.samples]
        (row,) = [u for u in builder.unresolved
                  if u["what"] == "sample S-2 top"]
        assert row["value"] == -2.0 and row["page"] == 10
        assert "depth -2 is negative" in row["why"]
        assert "the whole sample goes with it" in row["why"]
        assert '"kind": "shelby"' in row["read"]
        (entry,) = _ranked([row])
        assert entry.kind == "out_of_range" and entry.rank == "open"
        assert entry.values == ["-2.0", row["read"]]
        # a negative total depth, and a layer whose top is negative
        builder, inv = _log_case("pit")
        assert inv.total_depth is None
        whats = {u["what"]: u for u in builder.unresolved}
        assert whats["total depth"]["value"] == -1.0
        assert "listed here as read" in whats["total depth"]["why"]
        assert "Red sandy CLAY" in whats["layer 2 (Red sandy CLAY) top"]["read"]
        assert "listed here as read" in whats["pit width"]["why"]

    def test_a_depth_past_300_m_is_impossible_even_inside_the_window(self):
        builder, inv = _log_case("deep")
        assert [ly.top.value for ly in inv.layers] == [0.0]
        assert inv.layers[0].bottom is None
        assert inv.total_depth is None
        past = [u for u in builder.unresolved if "past 300 m" in u["why"]]
        assert [u["value"] for u in past] == [1250.0, 1100.0, 1100.0]
        assert {e.rank for e in _ranked(past)} == {"open"}

    def test_with_no_unit_a_depth_is_bounded_as_feet(self):
        reading = LogReading.model_validate({
            "investigation_id": "B-1", "depth_unit": "", "units_known": False,
            "total_depth": 950.0,
            "layers": [{"top": 0.0, "bottom": 990.0, "description": "x",
                        "prov": {"page": 10}}]})
        builder, inv = _flag_build(reading, None)
        assert inv.total_depth.value == 950.0 and inv.total_depth.unit == ""
        assert inv.layers[0].bottom is None
        (row,) = [u for u in builder.unresolved if "value" in u]
        assert "past 300 m even read as feet" in row["why"]

    def test_a_percentage_outside_0_to_100_is_kept_out_not_clamped(self):
        builder, inv = _log_case("window")
        sample = inv.samples[0]
        assert sample.recovery_percent is None and sample.rqd_percent is None
        assert sample.fines_percent == 40.0
        rows = {u["what"]: u for u in builder.unresolved
                if u["what"].startswith("sample S-1 ")}
        rqd = rows["sample S-1 RQD"]
        assert rqd["value"] == -5.0 and rqd["page"] == 10
        assert "-5 is not a percentage; no real sheet prints" in rqd["why"]
        assert {e.rank for e in _ranked(list(rows.values()))} == {"open"}

    def test_a_recovery_over_100_is_listed_as_a_reading(self):
        """FHWA-NHI-06-088 p. 3-44: recoveries greater than 100 percent are
        recorded. The record's field holds 0 to 100, so it cannot go in,
        and its entry says why without calling it impossible."""
        builder, inv = _log_case("window")
        (row,) = [u for u in builder.unresolved
                  if u["what"] == "sample S-1 recovery percent"]
        assert row["value"] == 105.0 and row["page"] == 10
        assert row["why"] == (
            "105 is over 100; recoveries greater than 100 percent are "
            "recorded (FHWA-NHI-06-088 Vol. I p. 3-44); the record's field "
            "holds 0 to 100, so it is kept out of the record and listed "
            "here as read")
        (entry,) = _ranked([row])
        assert entry.kind == "out_of_range" and entry.rank == "open"
        # a negative recovery is impossible
        reading = LogReading.model_validate({
            "investigation_id": "B-1", "depth_unit": "ft",
            "samples": [{"sample_id": "S-1", "top": 2.0,
                         "recovery_percent": -10.0, "prov": {"page": 10}}]})
        builder, inv = _flag_build(reading, (0.0, 20.0))
        (row,) = builder.unresolved
        assert "not a percentage; no real sheet prints" in row["why"]

    def test_drive_lengths_that_cannot_be_placed_are_listed(self):
        import json
        builder, inv = _log_case("window")
        assert [r.increments for r in inv.spt] == [[], []]
        rows = {u["what"]: u for u in builder.unresolved
                if u["what"].endswith("drive lengths")}
        first = rows["driven record 1 at 2 drive lengths"]
        assert first["value"] == "6, 6" and first["page"] == 10
        assert first["why"].startswith(
            "2 drive length(s) do not line up with the 3 blow count(s)")
        assert json.loads(first["read"]) == {
            "blows": ["3", "5", "7"], "drive_lengths": [6.0, 6.0],
            "drive_length_unit": "in"}
        second = rows["driven record 2 at 22 drive lengths"]
        assert second["why"].startswith(
            "no unit was given for the drive lengths")
        assert {e.rank for e in _ranked(list(rows.values()))} == {"open"}
        # under refuse they go without a word, as today
        from report_ingest.log_reader import _Builder
        from report_ingest.tests.checks_fixtures import (
            LOG_PAGES, LOG_WINDOW, log_window_reading,
        )
        today = _Builder(log_window_reading(), LOG_WINDOW, LOG_PAGES, "R")
        today.build()
        assert not any("drive length" in u["what"] for u in today.unresolved)

    def test_a_word_outside_a_vocabulary_is_listed(self):
        builder, inv = _log_case("window")
        assert inv.samples[2].kind == "other"
        assert inv.water[3].when == "unknown"
        rows = {u["what"]: u for u in builder.unresolved
                if "vocabulary" in u["why"]}
        assert rows["sample S-4 kind"]["value"] == "piston"
        assert rows["sample S-4 kind"]["why"] == (
            "'piston' is not in the sample kind vocabulary; recorded as "
            "'other' rather than invented")
        assert rows["water observation 4 when"]["value"] == "next morning"
        assert rows["water observation 4 when"]["page"] == 11
        builder, inv = _log_case("no_ruler")
        assert inv.kind == "other"
        (row,) = [u for u in builder.unresolved if u["what"] == "log kind"]
        assert row["value"] == "auger boring"
        assert _ranked([row])[0].rank == "changed"

    def test_an_answer_citing_positions_is_read_as_positions(self):
        builder, inv = _log_case("positions")    # cites 0, 1 of 10, 11
        assert [ly.prov.page for ly in inv.layers] == [10, 11]
        assert "a position in the 2 page(s) it was sent" in (
            inv.layers[0].prov.note)
        assert inv.water[0].prov.page == 11
        assert [c["page"] for c in builder.changes] == [10, 11, 11]
        (row,) = builder.unresolved             # the reader's own note
        assert row["what"] == "a smudge" and row["page"] == 10

    def test_an_answer_is_read_one_way_not_value_by_value(self):
        """Pages 3 to 5 cited 1, 2, 3 (counted from 1): 3 is a page of the
        call, so the answer is the report's pages, and 1 and 2 are pages
        outside it, as today; none is moved to a position on its own."""
        builder, inv = _log_case("one_based")
        assert [ly.prov.page for ly in inv.layers] == [3, 3, 3]
        assert [ly.prov.confidence for ly in inv.layers] == [0.4, 0.4, 0.9]
        assert builder.changes == []
        outside = [u for u in builder.unresolved
                   if "provenance names page" in u["why"]]
        assert [u["page"] for u in outside] == [1, 2]
        # the mix is noted (RI-89 review), without the positions clause:
        # pages from 3 cannot be positions among 3
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "page_numbers"]
        assert note["why"].endswith("1, 2 are not; they were read as the "
                                    "report's pages")

    def test_an_ambiguous_answer_is_read_as_positions_with_a_note(self):
        """Pages 1 to 3 cited 0, 1, 2: 0 is no page of the call and every
        cite is a position, so all are positions, and since 1 and 2 are
        also pages of the call the answer gets one note saying so."""
        builder, inv = _log_case("zero_based")
        assert [ly.prov.page for ly in inv.layers] == [1, 2, 3]
        assert inv.water[0].prov.page == 3
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "page_numbers"]
        assert note["what"] == "page numbers"
        assert note["why"].startswith(
            "the page numbers this answer cites mix this call's pages and "
            "others: 1, 2 are among this call's pages and 0 is not, and "
            "with the call's pages 1, 2, 3")
        assert note["why"].endswith("all were read as positions")
        assert _ranked([note])[0].rank == "note"

    def test_an_entry_names_the_report_page_not_the_position(self):
        reading = LogReading.model_validate({
            "investigation_id": "B-1", "depth_unit": "ft",
            "total_depth": 30.0,
            "layers": [{"top": 0.0, "bottom": 5.0, "description": "a",
                        "prov": {"page": 0}},
                       {"top": 25.0, "bottom": 30.0, "description": "b",
                        "prov": {"page": 1}}],
            "samples": [{"sample_id": "S-1", "top": -1.0,
                         "prov": {"page": 1}}]})
        builder, inv = _flag_build(reading, (0.0, 20.0))
        pages = {u["what"]: u["page"] for u in builder.unresolved}
        assert pages["layer 2 (b) top"] == 11
        assert pages["sample S-1 top"] == 11

    def test_depth_slack_is_the_window_slack(self):
        from types import SimpleNamespace
        from report_ingest.log_reader import DEPTH_SLACK, _depth_slack
        grid = SimpleNamespace(rulers={
            10: SimpleNamespace(step=-5.0), 11: SimpleNamespace(step=2.0)})
        assert _depth_slack(grid, [10, 11]) == DEPTH_SLACK * 5.0
        assert _depth_slack(grid, [12]) == 0.0
        assert _depth_slack(SimpleNamespace(rulers={}), [10]) == 0.0


def _sized(call) -> int:
    """A recorded request's size as the log reader sizes it: its messages,
    the system prompt and the answer schema (RI-110, the RI-121 review)."""
    from report_ingest.log_reader import estimate_tokens, request_overhead
    return estimate_tokens(call["messages"]) + request_overhead(call["system"])


class TestRequestSizeLimit:
    """BACKLOG RI-110: a conversation is trimmed or split before a request
    nears a per-request input-token limit."""

    @staticmethod
    def _convo(n_images_each):
        from report_ingest.engine import image_block, text_block, user
        return [
            user(text_block("brief"), *[image_block(b"x")] * n_images_each),
            {"role": "assistant", "content": [text_block("answer")]},
            user(text_block("again"), *[image_block(b"x")] * n_images_each),
        ]

    def test_no_limit_changes_nothing(self):
        from report_ingest.log_reader import fit_conversation
        msgs = self._convo(5)
        out, note = fit_conversation(msgs, None)
        assert out == msgs and note is None

    def test_a_conversation_over_the_limit_loses_its_earlier_pictures(self):
        from report_ingest.log_reader import (
            estimate_tokens, fit_conversation)
        msgs = self._convo(4)                     # about 13,600 tokens
        limit = 9000
        assert estimate_tokens(msgs) > limit
        out, note = fit_conversation(msgs, limit)
        assert note and "dropped" in note
        assert estimate_tokens(out) <= limit
        first = out[0]["content"]
        assert not any(b.get("type") == "image" for b in first)
        last = out[-1]["content"]
        assert sum(b.get("type") == "image" for b in last) == 4

    def test_the_log_reader_drops_earlier_pictures_before_a_follow_up(
            self, imperial):
        doc, _gt = imperial
        first = _good_imperial_reading(pages_left=[0])
        second = _good_imperial_reading(pages_left=[])
        free = FakeEngine([{"final": first}, {"final": second}])
        read_log(doc, [0], free)
        # One picture counts 1,700, so a limit 800 under the unlimited
        # second request is met by dropping the first request's picture.
        limit = _sized(free.calls[1]) - 800
        engine = FakeEngine([{"final": first}, {"final": second}])
        result = read_log(doc, [0], engine, max_request_tokens_in=limit)
        assert result.model_calls == 2
        sent = engine.calls[1]["messages"]
        pictures = [b for m in sent for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "image"]
        assert len(pictures) <= 1
        assert any("dropped" in w for w in result.warnings)

    def test_with_no_limit_the_log_reader_is_unchanged(self, imperial):
        doc, _gt = imperial
        first = _good_imperial_reading(pages_left=[0])
        second = _good_imperial_reading(pages_left=[])
        engine = FakeEngine([{"final": first}, {"final": second}])
        result = read_log(doc, [0], engine)
        sent = engine.calls[1]["messages"]
        pictures = [b for m in sent for b in m["content"]
                    if isinstance(b, dict) and b.get("type") == "image"]
        assert len(pictures) == 2
        assert not any("dropped" in w for w in result.warnings)

    def test_pages_over_one_request_are_split_across_calls(self):
        from report_ingest.engine import text_block, user
        from report_ingest.log_reader import _images_fit
        from report_ingest.engine import image_block

        def build(pages):
            return user(text_block("brief"),
                        *[image_block(b"x") for _ in pages])

        pages, _trial = _images_fit(build, [1, 2, 3, 4, 5], [], 5000)
        assert pages == [1, 2]                      # 2 x 1,700 + words
        pages, _trial = _images_fit(build, [1, 2, 3], [], 10 ** 6)
        assert pages == [1, 2, 3]
        pages, _trial = _images_fit(build, [1, 2, 3], [], 10)
        assert pages == [1]                         # never fewer than one


class TestSplitAcrossCalls:
    """BACKLOG RI-110 review: pages cut from a size-limited request are read
    in a later one, and a cut page never read is said."""

    @pytest.fixture
    def two_sheets(self, tmp_path):
        from planlens.testing.loggrid_fixtures import build_imperial_log
        gt = build_imperial_log()
        src = pymupdf.open(stream=gt.pdf, filetype="pdf")
        again = pymupdf.open(stream=gt.pdf, filetype="pdf")
        src.insert_pdf(again)                    # the same sheet twice
        path = tmp_path / "two.pdf"
        path.write_bytes(src.tobytes())
        from planlens.document import open_document
        doc = open_document(str(path))
        yield doc
        doc.close()

    @staticmethod
    def _pictures(call):
        return sum(1 for m in call["messages"] for b in m["content"]
                   if isinstance(b, dict) and b.get("type") == "image")

    def test_a_split_first_request_reads_every_page_eventually(
            self, two_sheets):
        free = FakeEngine([{"final": _good_imperial_reading(pages_left=[])}])
        read_log(two_sheets, [0, 1], free)
        both = _sized(free.calls[0])
        limit = both - 800                       # one picture too many
        engine = FakeEngine([
            {"final": _good_imperial_reading(pages_left=[])},
            {"final": _good_imperial_reading(pages_left=[])}])
        result = read_log(two_sheets, [0, 1], engine,
                          max_request_tokens_in=limit)
        assert result.model_calls == 2           # page 1 followed up
        assert self._pictures(engine.calls[0]) == 1
        follow = engine.calls[1]["messages"][-1]["content"][0]["text"]
        assert "1" in follow
        assert not any("never read" in w for w in result.warnings)

    def test_a_cut_page_the_budget_never_reaches_is_said(self, two_sheets):
        free = FakeEngine([{"final": _good_imperial_reading(pages_left=[])}])
        read_log(two_sheets, [0, 1], free)
        limit = _sized(free.calls[0]) - 800
        engine = FakeEngine([
            {"final": _good_imperial_reading(pages_left=[])}])
        result = read_log(two_sheets, [0, 1], engine, budget=1,
                          max_request_tokens_in=limit)
        assert result.model_calls == 1
        assert any("never read" in w and "[1]" in w
                   for w in result.warnings)

    def test_a_page_named_only_in_pages_left_is_said_after_a_size_stop(
            self, two_sheets):
        """BACKLOG RI-121 (5): the first request carries both sheets and
        the reader names sheet 1 in ``pages_left``; the request for it is
        over the limit even with the earlier pictures dropped, so the log
        is read no further and sheet 1 is said never read."""
        from report_ingest.log_reader import (estimate_tokens,
                                              fit_conversation,
                                              request_overhead)
        # The reader says something beside its answer: words the next
        # request carries back, which the dropped picture does not make up.
        first = {"final": _good_imperial_reading(pages_left=[1]),
                 "text": "made-up words " * 500}
        free = FakeEngine([first,
                           {"final": _good_imperial_reading(pages_left=[])}])
        read_log(two_sheets, [0, 1], free)
        limit = _sized(free.calls[0])                       # just fits
        room = limit - request_overhead(free.calls[1]["system"])
        trimmed, _note = fit_conversation(free.calls[1]["messages"], room)
        assert estimate_tokens(trimmed) > room
        engine = FakeEngine([first])
        result = read_log(two_sheets, [0, 1], engine,
                          max_request_tokens_in=limit)
        assert result.model_calls == 1
        assert self._pictures(engine.calls[0]) == 2
        assert any("read no further" in w for w in result.warnings)
        assert any("[1]" in w and "never read" in w
                   and "over its size limit" in w for w in result.warnings)

    def test_the_system_prompt_and_schema_count_toward_the_limit(
            self, two_sheets):
        """The RI-121 review: a limit the first request's messages fit
        under, both pictures and all, is passed once the system prompt and
        the answer schema sent with it are counted, so the request carries
        one sheet and the other is read in a later call."""
        from report_ingest.engine import schema_tokens_est
        from report_ingest.log_reader import (CHARS_PER_TOKEN_EST,
                                              estimate_tokens,
                                              request_overhead)
        free = FakeEngine([{"final": _good_imperial_reading(pages_left=[])}])
        read_log(two_sheets, [0, 1], free)
        system = free.calls[0]["system"]
        overhead = request_overhead(system)
        assert overhead == (int(len(system) / CHARS_PER_TOKEN_EST)
                            + schema_tokens_est(LogReading))
        assert overhead > 2000                 # more than one picture
        limit = estimate_tokens(free.calls[0]["messages"]) + overhead - 100
        assert estimate_tokens(free.calls[0]["messages"]) < limit
        engine = FakeEngine([
            {"final": _good_imperial_reading(pages_left=[])},
            {"final": _good_imperial_reading(pages_left=[])}])
        result = read_log(two_sheets, [0, 1], engine,
                          max_request_tokens_in=limit)
        assert self._pictures(engine.calls[0]) == 1
        assert result.model_calls == 2
        assert any("were not sent with the first request" in w
                   for w in result.warnings)
        assert not any("never read" in w for w in result.warnings)

    def test_a_first_request_over_the_limit_with_one_page_is_said(
            self, imperial):
        doc, _gt = imperial
        engine = FakeEngine([{"final": _good_imperial_reading()}])
        result = read_log(doc, [0], engine, max_request_tokens_in=10)
        assert result.model_calls == 1
        assert any("over its size limit" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# an item handed in whole that runs on into another hole's log (RI-117)
# ---------------------------------------------------------------------------

class TestAnotherHolesPage:

    @pytest.fixture(scope="class")
    @classmethod
    def multi(cls, tmp_path_factory):
        from planlens.document import open_document
        from report_ingest.tests.multi_hole_fixtures import (
            build_multi_hole_log)
        gt = build_multi_hole_log()
        path = tmp_path_factory.mktemp("multi") / "multi.pdf"
        path.write_bytes(gt.pdf)
        doc = open_document(str(path))
        yield doc, gt
        doc.close()

    @staticmethod
    def _reading(page: int) -> LogReading:
        prov = ReadProv(page=page, bbox=(60.0, 220.0, 300.0, 240.0))
        return LogReading(
            investigation_id="HX-3", kind="boring", depth_unit="ft",
            units_known=True,
            layers=[ReadLayer(top=0.0, bottom=9.0, uscs="CL",
                              description="SANDY LEAN CLAY (CL), brown, "
                                          "stiff", prov=prov)],
            pages_read=[page])

    def test_the_floor_and_the_brief_leave_the_other_hole_out(self, multi):
        doc, gt = multi
        hx3, hx4 = gt.holes["HX-3"], gt.holes["HX-4"]
        engine = FakeEngine([{"final": self._reading(hx3[0])}])
        result = read_log(doc, hx3 + hx4, engine, item_title="QZ-7")
        assert result.floor.samples
        assert all(s.prov.page in hx3 for s in result.floor.samples)
        assert all(s.prov.page in hx3 for s in result.floor.spt)
        assert any(f"page {hx4[0]} names the hole HX-4" in w
                   for w in result.warnings)
        assert any(row.get("flag") == "floor_other_hole"
                   for row in result.unresolved)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert (f"The headers name this log HX-3, but page {hx4[0]} names "
                f"HX-4.") in brief

    def test_one_hole_s_pages_are_read_as_before(self, multi):
        doc, gt = multi
        hx3 = gt.holes["HX-3"]
        engine = FakeEngine([{"final": self._reading(hx3[0])}])
        result = read_log(doc, hx3, engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "The headers name this log" not in brief
        assert not any("names the hole" in w for w in result.warnings)
        assert not any(row.get("flag") == "reader_other_hole"
                       for row in result.unresolved)

    def test_rows_the_model_read_off_the_other_hole_s_page_are_dropped(
            self, multi):
        """RI-145: told to leave page HX-4 alone, the model still returns a
        layer off it. The layer is not in the log, the page leaves the
        log's pages, and one unresolved row keeps the layer as read."""
        doc, gt = multi
        hx3, hx4 = gt.holes["HX-3"], gt.holes["HX-4"]
        reading = self._reading(hx3[0])
        reading.layers.append(ReadLayer(
            top=0.0, bottom=6.0, uscs="SM",
            description="Made-up SILTY SAND of the other hole",
            prov=ReadProv(page=hx4[0], bbox=(60.0, 220.0, 300.0, 240.0))))
        engine = FakeEngine([{"final": reading}])
        result = read_log(doc, hx3 + hx4, engine)
        log = result.investigation
        assert all(layer.prov.page in hx3 for layer in log.layers)
        assert not any("other hole" in layer.description
                       for layer in log.layers)
        assert not set(log.pages) & set(hx4)
        (row,) = [r for r in result.unresolved
                  if r.get("flag") == "reader_other_hole"]
        assert row["page"] == hx4[0] and row["refused_by"] == "python"
        assert "names the hole HX-4, not this log's (HX-3)" in row["why"]
        assert "Made-up SILTY SAND of the other hole" in row["read"]
