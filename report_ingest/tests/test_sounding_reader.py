"""The sounding readers, offline: the floor, the gates and the merge.

Every test runs the real reader over a real synthetic sheet with a fake
engine replaying a canned structured reply, so the brief that is asserted on
is the brief a model would actually be sent and the refusals are the ones
that would actually happen.

The pages come from ``report_ingest.tests.sounding_fixtures``: a cone
sounding printed as an unruled column table, the same kind of sounding drawn
as traces with nothing but its axes in text, and a dynamic probe whose column
header runs over three printed lines. Nothing in them is copied from a real
sheet.
"""

from __future__ import annotations

import pytest

from report_ingest.model import (
    CPTData, CPTPoint, DCPData, DCPPoint, Investigation, Provenance, Quantity,
)
from report_ingest.sounding_reader import (
    CPT_STEP_M, MAX_SOUNDING_CALLS, MIN_STEP_M, SOUNDING_KINDS,
    SOUNDING_READER_SYSTEM, SOUNDING_TOOLS, ReadCone, ReadPoint, ReadProv,
    SoundingReading, Unsettled, axes_from_text, column_role, default_step,
    head_unit, merge_sounding, read_sounding, read_tabulated,
    serialise_floor, sounding_floor,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted

fitz = pytest.importorskip("fitz", reason="the fixtures are drawn with PyMuPDF")


def _open(pdf: bytes, tmp_path, name: str):
    from planlens.document import open_document
    path = tmp_path / f"{name}.pdf"
    path.write_bytes(pdf)
    return open_document(str(path))


@pytest.fixture
def tabulated(tmp_path):
    from report_ingest.tests.sounding_fixtures import build_tabulated_cpt
    gt = build_tabulated_cpt()
    doc = _open(gt.pdf, tmp_path, "cpt_tab")
    yield doc, gt
    doc.close()


@pytest.fixture
def plotted(tmp_path):
    from report_ingest.tests.sounding_fixtures import build_plotted_cpt
    gt = build_plotted_cpt()
    doc = _open(gt.pdf, tmp_path, "cpt_plot")
    yield doc, gt
    doc.close()


@pytest.fixture
def probe(tmp_path):
    from report_ingest.tests.sounding_fixtures import build_tabulated_dcp
    gt = build_tabulated_dcp()
    doc = _open(gt.pdf, tmp_path, "dcp_tab")
    yield doc, gt
    doc.close()


def _reading(**kwargs) -> SoundingReading:
    base = dict(investigation_id="CPT-4", kind="cpt", depth_unit="m",
                qc_unit="MPa", fs_unit="MPa", u2_unit="kPa")
    base.update(kwargs)
    return SoundingReading(**base)


def _point(depth: float, **kwargs) -> ReadPoint:
    return ReadPoint(depth=depth, **kwargs)


# ---------------------------------------------------------------------------
# what a column heading names
# ---------------------------------------------------------------------------

class TestTheColumnVocabulary:

    @pytest.mark.parametrize("heading,role", [
        ("Depth (m)", "depth"), ("d(m)", "depth"), ("Profondeur", "depth"),
        ("dref (m)", "elevation"),
        ("Qc (Mpa)", "qc"), ("Cone resistance", "qc"),
        ("Fs (Mpa)", "fs"), ("Sleeve friction", "fs"),
        ("u2 (kPa)", "u2"), ("Rf (%)", "rf"),
    ])
    def test_a_cone_sheets_headings(self, heading, role):
        assert column_role(heading, "cpt") == role

    @pytest.mark.parametrize("heading,role", [
        ("Number of Blows", "blows"), ("Nbre de coups", "blows"),
        ("Rd (MPa)", "index"), ("Depth (m)", "depth"),
        ("CBR (%)", "cbr"),
    ])
    def test_a_dynamic_probes_headings(self, heading, role):
        assert column_role(heading, "dcp") == role

    def test_an_accented_heading_is_read(self):
        # Half the corpus is French and the vocabulary is written unaccented.
        assert column_role("Résistance dynamique apparente en Bar",
                           "dcp") == "index"

    def test_a_kind_narrows_the_vocabulary(self):
        # A blow count is a dynamic probe's column and never a cone's.
        assert column_role("Number of Blows", "cpt") is None
        assert column_role("Cone resistance", "dcp") is None

    def test_a_sentence_is_not_a_heading(self):
        # The one that put a tip-resistance axis on a plot of undrained
        # shear strength.
        assert column_role(
            "Undrained shear strength interpreted from cone resistance and "
            "pore pressure response", "cpt") is None

    @pytest.mark.parametrize("heading,unit", [
        ("Qc (Mpa)", "Mpa"), ("Depth, m", "m"), ("Rd (MPa)", "MPa"),
        ("Cone resistance (qc) in MPa", "MPa"),
        ("Depth in m to reference level (TAW)", "m"),
        ("mm/blow", "mm/blow"),
    ])
    def test_the_unit_a_heading_prints(self, heading, unit):
        assert head_unit(heading) == unit

    def test_a_symbol_in_parentheses_is_not_a_unit(self):
        # "(qc)" and "(TAW)" are what sit in a heading's brackets far more
        # often than a unit does.
        assert head_unit("Friction ratio (Rf)") == ""
        assert head_unit("Reference level (TAW)") == ""


# ---------------------------------------------------------------------------
# the floor on a tabulated sheet
# ---------------------------------------------------------------------------

class TestTheTabulatedFloor:

    def test_the_table_is_the_floor(self, tabulated):
        doc, gt = tabulated
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        assert floor.tabulated
        assert floor.n_points == len(gt.series)

    def test_every_row_came_back_with_its_channels(self, tabulated):
        doc, gt = tabulated
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        got = {round(p.depth.value, 2): p for p in floor.investigation.cpt.points}
        assert set(got) == {round(d, 2) for d in gt.series}
        for depth, wanted in gt.series.items():
            point = got[round(depth, 2)]
            assert point.qc.value == pytest.approx(wanted["qc"])
            assert point.fs.value == pytest.approx(wanted["fs"])
            assert point.u2.value == pytest.approx(wanted["u2"])

    def test_the_units_are_the_ones_the_headings_print(self, tabulated):
        doc, gt = tabulated
        data = sounding_floor(doc, [0], "cpt", "RXX").investigation.cpt
        assert data.qc_unit == "MPa"
        assert data.u2_unit == "kPa"
        assert data.points[0].qc.unit == "MPa"

    def test_the_step_is_the_one_the_rows_stand_at(self, tabulated):
        doc, _gt = tabulated
        data = sounding_floor(doc, [0], "cpt", "RXX").investigation.cpt
        assert data.step.value == pytest.approx(0.20)
        assert data.depth_interval.value == pytest.approx(0.20)

    def test_the_title_block_names_the_sounding(self, tabulated):
        doc, gt = tabulated
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        assert floor.investigation.investigation_id == gt.investigation_id

    def test_a_header_over_three_printed_lines_is_still_a_header(self, probe):
        # The French dynamic-probe sheets set their column names on one
        # line, the rest of the name on the next and the unit on a third.
        doc, gt = probe
        series, _warnings = read_tabulated(doc, [0], "dcp")
        assert series is not None
        assert set(series.roles) >= {"depth", "blows", "index"}
        assert series.units["depth"] == "m"
        assert series.units["index"] == "MPa"

    def test_a_dynamic_probe_yields_blows_per_increment(self, probe):
        doc, gt = probe
        data = sounding_floor(doc, [0], "dcp", "RXX").investigation.dcp
        got = {round(p.depth.value, 2): p for p in data.points}
        assert set(got) == {round(d, 2) for d in gt.series}
        for depth, wanted in gt.series.items():
            assert got[round(depth, 2)].blows == pytest.approx(
                wanted["blows"])
            assert got[round(depth, 2)].index.value == pytest.approx(
                wanted["index"])

    def test_the_increment_is_taken_from_the_rows_not_invented(self, probe):
        doc, _gt = probe
        data = sounding_floor(doc, [0], "dcp", "RXX").investigation.dcp
        assert data.increment.value == pytest.approx(0.20)
        assert data.increment.unit == "m"


# ---------------------------------------------------------------------------
# the floor on a plotted sheet
# ---------------------------------------------------------------------------

class TestThePlottedFloor:

    def test_a_plotted_sheet_has_no_points(self, plotted):
        doc, _gt = plotted
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        assert not floor.tabulated
        assert floor.n_points == 0

    def test_the_axes_come_off_the_sheets_own_text(self, plotted):
        doc, gt = plotted
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        by_role = {a.role: a for a in floor.axes}
        for role, (low, high, tick) in gt.axes.items():
            assert role in by_role, f"{role} axis not found"
            assert by_role[role].low == pytest.approx(low)
            assert by_role[role].high == pytest.approx(high)
            assert by_role[role].tick == pytest.approx(tick)

    def test_an_axis_carries_the_unit_its_title_prints(self, plotted):
        doc, _gt = plotted
        axes, _warnings = axes_from_text(doc, [0], "cpt")
        assert {a.role: a.unit for a in axes}["qc"] == "MPa"

    def test_the_floor_says_the_series_is_plotted(self, plotted):
        doc, _gt = plotted
        text = serialise_floor(sounding_floor(doc, [0], "cpt", "RXX"))
        assert "PLOTS ITS SERIES" in text
        assert "Cone resistance" in text

    def test_a_sheet_with_no_axis_warns(self, tmp_path):
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(60, 60), "A PAGE WITH NOTHING ON IT")
        opened = _open(doc.tobytes(), tmp_path, "blank")
        try:
            floor = sounding_floor(opened, [0], "cpt", "RXX")
            assert any("no axis" in w for w in floor.warnings)
        finally:
            opened.close()


# ---------------------------------------------------------------------------
# the digitising step
# ---------------------------------------------------------------------------

class TestTheDigitisingStep:

    def test_the_step_is_the_finer_of_the_grid_and_the_default(self, plotted):
        doc, _gt = plotted
        floor = sounding_floor(doc, [0], "cpt", "RXX")
        # The sheet's depth grid is 1 m and the default is half a metre; the
        # traces swing faster than the grid, so the finer one wins.
        assert default_step(floor) == pytest.approx(CPT_STEP_M)

    def test_the_step_never_goes_below_the_floor(self, tmp_path):
        from report_ingest.sounding_reader import PlotAxis, SoundingFloor
        floor = SoundingFloor(kind="cpt", axes=[
            PlotAxis(role="depth", unit="m", ticks=[0.0, 0.02, 0.04, 0.06])])
        assert default_step(floor) == pytest.approx(MIN_STEP_M)


# ---------------------------------------------------------------------------
# the reader end to end
# ---------------------------------------------------------------------------

class TestTheReader:

    def test_a_tabulated_sheet_costs_one_call(self, tabulated):
        doc, gt = tabulated
        engine = FakeEngine([{"final": _reading(points=[])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        assert result.model_calls == 1
        assert result.tabulated
        # Nothing was dropped: the floor's series is the record's.
        assert result.investigation.cpt.n_points == len(gt.series)

    def test_the_brief_says_which_shape_the_sheet_is(self, tabulated):
        doc, _gt = tabulated
        engine = FakeEngine([{"final": _reading(points=[])}])
        read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "TABULATES ITS SERIES" in brief
        assert "Do NOT re-type it" in brief

    def test_a_plotted_sheet_is_told_the_step_and_the_axes(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(investigation_id="CPT-7",
                                                points=[])}])
        read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "PLOTS ITS SERIES" in brief
        assert "Digitise it through zoom_plot" in brief
        assert "from 5 to 25" in brief        # the qc axis it must check

    def test_a_digitised_series_becomes_the_record(self, plotted):
        doc, _gt = plotted
        reading = _reading(
            investigation_id="CPT-7", digitised=True, step=0.5,
            points=[_point(0.5, qc=8.0, fs=0.12),
                    _point(1.0, qc=12.0, fs=0.18),
                    _point(1.5, qc=15.0, fs=0.21, crossed=True,
                           note="the friction trace crosses the tip trace")])
        engine = FakeEngine([{"final": reading}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        data = result.investigation.cpt
        assert data.digitised
        assert [p.depth.value for p in data.points] == [0.5, 1.0, 1.5]
        assert data.step.value == pytest.approx(0.5)

    def test_a_crossing_lowers_that_points_confidence(self, plotted):
        doc, _gt = plotted
        reading = _reading(
            investigation_id="CPT-7", digitised=True,
            points=[_point(0.5, qc=8.0), _point(1.0, qc=12.0, crossed=True)])
        engine = FakeEngine([{"final": reading}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        clean, crossed = result.investigation.cpt.points
        assert crossed.prov.confidence < clean.prov.confidence
        assert "crosses" in crossed.prov.note

    def test_a_digitised_point_is_a_change_a_reviewer_sees(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", digitised=True,
            points=[_point(0.5, qc=8.0)])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        assert result.changes
        assert result.changes[0]["what"].startswith("point at depth")

    def test_the_zoom_tool_is_offered_and_its_use_is_recorded(self, plotted):
        doc, gt = plotted
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 0, "bbox": list(gt.plot_bbox),
                                      "why": "read the tip trace"})]},
            {"final": _reading(investigation_id="CPT-7", digitised=True,
                               points=[_point(1.0, qc=9.0)])},
        ])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                               budget=2)
        assert result.tool_calls == 1
        assert engine.calls[0]["tools"] == ["zoom_plot"]

    def test_a_zoom_off_this_sheet_is_an_error_not_a_stop(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 7, "bbox": [0, 0, 10, 10]})]},
            {"final": _reading(investigation_id="CPT-7", points=[])},
        ])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                               budget=2)
        assert result.model_calls == 2
        blocks = engine.calls[1]["messages"][-1]["content"]
        assert any(b.get("is_error") for b in blocks)

    def test_the_budget_is_a_ceiling(self, tabulated):
        doc, _gt = tabulated
        engine = FakeEngine([{"final": _reading(points=[])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                               budget=99)
        assert result.model_calls <= MAX_SOUNDING_CALLS

    def test_a_reader_that_never_answers_raises(self, tabulated):
        doc, _gt = tabulated
        engine = FakeEngine([{"text": "I would rather not"},
                             {"text": "still not"}])
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                          budget=2)

    def test_an_unsettled_list_buys_one_follow_up(self, plotted):
        doc, _gt = plotted
        first = _reading(investigation_id="CPT-7", points=[],
                         unsettled=[Unsettled(what="the pore pressure trace",
                                              why="it is drawn over the tip")])
        engine = FakeEngine([{"final": first},
                             {"final": _reading(investigation_id="CPT-7",
                                                points=[])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                               budget=2)
        assert result.model_calls == 2


# ---------------------------------------------------------------------------
# what Python refuses
# ---------------------------------------------------------------------------

class TestTheGates:

    def test_a_depth_outside_the_sheets_own_axis_is_refused(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", digitised=True,
            points=[_point(1.0, qc=9.0), _point(44.0, qc=9.0)])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        assert result.investigation.cpt.n_points == 1
        assert any("outside the sheet's own depth axis" in u["why"]
                   for u in result.unresolved)

    def test_a_channel_outside_its_printed_range_is_refused(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", digitised=True,
            points=[_point(1.0, qc=900.0, fs=0.2)])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        point = result.investigation.cpt.points[0]
        assert point.qc is None
        assert point.fs is not None
        assert any("the scale was misread" in u["why"]
                   for u in result.unresolved)

    def test_a_negative_tip_resistance_is_refused(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", digitised=True,
            points=[_point(1.0, qc=-4.0)])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        assert result.investigation.cpt.n_points == 0
        assert any("cannot be negative" in u["why"]
                   for u in result.unresolved)

    def test_a_series_with_no_depth_unit_places_nothing(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", depth_unit="", units_known=False,
            digitised=True, points=[_point(1.0, qc=9.0)])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX")
        assert result.model_investigation.cpt.n_points == 0
        assert any("no depth unit" in u["why"] for u in result.unresolved)

    def test_the_readers_own_unsettled_list_reaches_the_record(self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            investigation_id="CPT-7", points=[],
            unsettled=[Unsettled(what="the u2 trace", why="not drawn")])},
            {"final": _reading(investigation_id="CPT-7", points=[])}])
        result = read_sounding(doc, [0], engine, kind="cpt", report_id="RXX",
                               budget=2)
        assert result.model_calls == 2


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _series(points, kind="cpt"):
    inv = Investigation(investigation_id="X-1", kind=kind, depth_unit="m",
                        pages=[0])
    if kind == "cpt":
        inv.cpt = CPTData(points=points, qc_unit="MPa")
    else:
        inv.dcp = DCPData(points=points)
    return inv


def _cpt_point(depth, qc, method="tables", confidence=0.9, note=""):
    return CPTPoint(
        depth=Quantity(value=depth, unit="m"),
        qc=Quantity(value=qc, unit="MPa"),
        prov=Provenance(page=0, method=method, confidence=confidence,
                        note=note))


class TestTheMerge:

    def test_a_floor_point_the_reader_omitted_is_kept(self):
        floor = _series([_cpt_point(0.2, 1.8), _cpt_point(0.4, 3.4)])
        model = _series([])
        merged, log = merge_sounding(floor, model)
        assert merged.cpt.n_points == 2
        assert log.kept

    def test_a_point_the_reader_adds_is_added(self):
        floor = _series([_cpt_point(0.2, 1.8)])
        model = _series([_cpt_point(0.2, 1.8, method="model"),
                         _cpt_point(0.4, 3.4, method="model")])
        merged, log = merge_sounding(floor, model)
        assert merged.cpt.n_points == 2
        assert log.added

    def test_the_two_voters_agreeing_is_reconciled(self):
        floor = _series([_cpt_point(0.2, 1.80)])
        model = _series([_cpt_point(0.2, 1.82, method="model")])
        merged, log = merge_sounding(floor, model)
        assert log.reconciled >= 1
        assert merged.cpt.n_points == 1

    def test_a_split_without_evidence_keeps_the_floors_value(self):
        floor = _series([_cpt_point(0.2, 1.80)])
        model = _series([_cpt_point(0.2, 9.90, method="model")])
        merged, log = merge_sounding(floor, model)
        assert merged.cpt.points[0].qc.value == pytest.approx(1.80)
        assert log.disagreements
        assert log.disagreements[0]["kept"] == "floor"

    def test_a_split_with_evidence_takes_the_readers_value(self):
        floor = _series([_cpt_point(0.2, 1.80)])
        model = _series([_cpt_point(
            0.2, 9.90, method="model_from_picture",
            note="the table row is 9.90; the floor read the wrong column")])
        merged, log = merge_sounding(floor, model)
        assert merged.cpt.points[0].qc.value == pytest.approx(9.90)
        assert log.disagreements[0]["kept"] == "model"
        # Both values stay on the record.
        assert merged.cpt.points[0].prov.alternatives

    def test_the_instrument_the_reader_named_fills_an_empty_floor(self):
        floor = _series([_cpt_point(0.2, 1.8)])
        model = _series([_cpt_point(0.2, 1.8, method="model")])
        model.cpt.cone_type = "piezocone S15"
        model.cpt.standard = "EN ISO 22476-1"
        merged, _log = merge_sounding(floor, model)
        assert merged.cpt.cone_type == "piezocone S15"
        assert merged.cpt.standard == "EN ISO 22476-1"

    def test_a_dynamic_probes_blows_merge_the_same_way(self):
        def point(depth, blows, method="tables"):
            return DCPPoint(depth=Quantity(value=depth, unit="m"),
                            blows=blows,
                            prov=Provenance(page=0, method=method,
                                            confidence=0.9))
        floor = _series([point(0.2, 4.0), point(0.4, 3.0)], kind="dcp")
        model = _series([point(0.2, 4.0, "model")], kind="dcp")
        merged, log = merge_sounding(floor, model)
        assert merged.dcp.n_points == 2
        assert log.kept


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

class TestThePrompt:

    def test_it_says_a_sounding_is_a_series(self):
        assert "SERIES" in SOUNDING_READER_SYSTEM

    def test_it_forbids_re_typing_a_tabulated_series(self):
        assert "must NOT re-type it" in SOUNDING_READER_SYSTEM

    def test_it_says_the_axis_may_be_an_elevation(self):
        assert "NOT ALWAYS A DEPTH" in SOUNDING_READER_SYSTEM
        assert "vertical_axis" in SOUNDING_READER_SYSTEM

    def test_it_forbids_computing_a_friction_ratio(self):
        assert "Do not work out a friction ratio" in SOUNDING_READER_SYSTEM

    def test_it_asks_for_the_hammer_on_a_dynamic_probe(self):
        assert "hammer's mass and drop" in SOUNDING_READER_SYSTEM

    def test_the_only_tool_is_the_zoom(self):
        assert [t["name"] for t in SOUNDING_TOOLS] == ["zoom_plot"]

    def test_the_kinds_are_the_two_this_module_reads(self):
        assert SOUNDING_KINDS == ("cpt", "dcp")
