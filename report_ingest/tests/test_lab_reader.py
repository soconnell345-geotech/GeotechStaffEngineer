"""The lab reader, offline: what it shows the model and what it refuses.

Every test runs the real reader over a real synthetic lab sheet with a fake
engine replaying a canned structured reply. So the brief that is asserted on
is the brief a model would actually be sent, and the refusals are the
refusals that would actually happen.

The pages come from ``report_ingest.tests.lab_fixtures``: a plasticity chart
with its limits in a box, a grading curve with its values tabulated beneath
it, a summary table of four specimens, and a laboratory certificate that
reports nothing. Nothing in them is copied from a real sheet.
"""

from __future__ import annotations

import pytest

from report_ingest.lab_reader import (
    KIND_DEFINITIONS, LAB_READER_SYSTEM, LAB_TOOLS, LabSheetReading,
    MAX_MODEL_CALLS, ReadPassing, ReadPoint, ReadProv, ReadQuantity,
    ReadReported, ReadRow, ReadSeries, ReadSpecimen, ReadTest, SERIES_NAMES,
    Unsettled, read_lab_sheet, serialise_page,
)
from report_ingest.model import (
    AtterbergResult, ChemicalResult, GradationResult, OtherResult,
    StrengthResult, SummaryTableResult,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted

fitz = pytest.importorskip("fitz", reason="the fixtures are drawn with PyMuPDF")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _open(pdf: bytes, tmp_path):
    from planlens.document import open_document
    path = tmp_path / "sheet.pdf"
    path.write_bytes(pdf)
    return open_document(str(path))


@pytest.fixture
def atterberg(tmp_path):
    from report_ingest.tests.lab_fixtures import build_atterberg_sheet
    gt = build_atterberg_sheet()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def gradation(tmp_path):
    from report_ingest.tests.lab_fixtures import build_gradation_sheet
    gt = build_gradation_sheet()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def summary(tmp_path):
    from report_ingest.tests.lab_fixtures import build_summary_table
    gt = build_summary_table()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def certificate(tmp_path):
    from report_ingest.tests.lab_fixtures import build_certificate_page
    gt = build_certificate_page()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


def _prov(page=0, bbox=(60.0, 150.0, 380.0, 182.0), from_image=False,
          note=""):
    return ReadProv(page=page, bbox=bbox, from_image=from_image, note=note)


def _atterberg_reading(**over) -> LabSheetReading:
    """What a careful reader would return for the Atterberg fixture."""
    test = dict(kind="atterberg", investigation_id="B-4", sample_id="S-2",
                depth_top=7.5, standard="ASTM D4318", uscs="CL",
                ll=48.0, pl=22.0, pi=26.0, wc=27.4, prov=_prov())
    test.update(over.pop("test", {}))
    base = dict(depth_unit="ft", tests=[ReadTest(**test)], pages_read=[0])
    base.update(over)
    return LabSheetReading(**base)


def _gradation_reading(**over) -> LabSheetReading:
    """The gradation fixture: two tests on one specimen."""
    common = dict(investigation_id="SB-11", sample_id="3", depth_top=12.0,
                  uscs="SC", prov=_prov())
    grading = ReadTest(
        kind="gradation", standard="ASTM D6913", gravel_percent=13.0,
        sand_percent=44.0, fines_percent=43.0,
        series=[ReadSeries(
            name="percent_passing", x_unit="mm", y_unit="%", digitised=False,
            points=[ReadPoint(x=75.0, y=100.0, label="3 in"),
                    ReadPoint(x=19.0, y=98.0, label="3/4 in"),
                    ReadPoint(x=4.75, y=87.0, label="No. 4"),
                    ReadPoint(x=0.425, y=61.0, label="No. 40"),
                    ReadPoint(x=0.075, y=43.0, label="No. 200")])],
        **common)
    limits = ReadTest(kind="atterberg", ll=31.0, pl=19.0, pi=12.0, **common)
    base = dict(depth_unit="ft", tests=[grading, limits], pages_read=[0])
    base.update(over)
    return LabSheetReading(**base)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

class TestWhatTheModelIsShown:
    def test_the_brief_carries_the_lines_with_their_boxes(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        read_lab_sheet(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "x0,y0,x1,y1 | text" in brief
        assert "ATTERBERG LIMITS RESULTS" in brief
        assert "Depth: 7.5 ft" in brief

    def test_the_brief_carries_the_detected_tables(self, gradation):
        doc, _gt = gradation
        engine = FakeEngine([{"final": _gradation_reading()}])
        read_lab_sheet(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "TABLE" in brief
        assert "PERCENT FINER" in brief
        assert "0.075" in brief

    def test_the_brief_carries_the_kind_vocabulary_and_series_names(
            self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        read_lab_sheet(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE KIND VOCABULARY" in brief
        for kind in KIND_DEFINITIONS:
            assert kind in brief
        assert "THE SERIES NAMES" in brief
        for name in SERIES_NAMES:
            assert name in brief

    def test_the_page_picture_is_attached(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        read_lab_sheet(doc, [0], engine)
        blocks = engine.calls[0]["messages"][0]["content"]
        assert any(b.get("type") == "image" for b in blocks)

    def test_a_hint_is_offered_as_a_hint_and_not_as_a_fact(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        read_lab_sheet(doc, [0], engine, hint_kind="gradation")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "'gradation'" in brief
        assert "HINT" in brief and "the title wins" in brief

    def test_the_system_prompt_states_the_four_rules(self):
        for rule in ("SHEET'S OWN TITLE", "READ THE TABLE BEFORE THE PLOT",
                     "UNITS STAY AS PRINTED", "A SUMMARY TABLE IS ONE ENTRY",
                     "A PAGE WITH NO RESULTS IS STILL AN ANSWER",
                     "ANOTHER LANGUAGE"):
            assert rule in LAB_READER_SYSTEM

    def test_the_prompt_names_the_words_a_sheet_uses_in_other_languages(self):
        """A third of these laboratories do not work in English, and what
        says a number's meaning is the word printed next to it."""
        for word in ("GRANULOMETRIQUE", "TENEUR EN EAU", "CORTE DIRECTO",
                     "SONDEO"):
            assert word in LAB_READER_SYSTEM
        assert "do not" in LAB_READER_SYSTEM
        assert "translate it" in LAB_READER_SYSTEM

    def test_a_page_with_no_text_says_so(self, tmp_path):
        """A blank page is a page to look at, and the brief says so."""
        doc_ = fitz.open()
        doc_.new_page(width=612, height=792)
        blank = doc_.tobytes()
        doc_.close()
        doc = _open(blank, tmp_path)
        try:
            text = serialise_page(doc, 0)
            assert "THIS PAGE CARRIES NO TEXT" in text
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# the records that come out
# ---------------------------------------------------------------------------

class TestWhatComesBack:
    def test_a_sheet_becomes_a_typed_record(self, atterberg):
        doc, gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine, report_id="R99")
        assert len(result.tests) == 1
        test = result.tests[0]
        assert test.kind == "atterberg"
        assert isinstance(test.result, AtterbergResult)
        assert test.result.ll == 48.0 and test.result.pi == 26.0
        assert test.investigation_id == gt.investigation_id
        assert test.depth_top.value == gt.depth_top
        assert test.depth_top.unit == "ft"
        assert test.source_report == "R99"
        assert test.pages == [0]

    def test_one_sheet_can_be_two_tests(self, gradation):
        doc, _gt = gradation
        engine = FakeEngine([{"final": _gradation_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.kinds == ["gradation", "atterberg"]
        grading, limits = result.tests
        assert isinstance(grading.result, GradationResult)
        assert isinstance(limits.result, AtterbergResult)
        # Both belong to the same specimen, and say so the same way.
        assert grading.investigation_id == limits.investigation_id == "SB-11"
        assert grading.depth_top.value == limits.depth_top.value == 12.0

    def test_a_grading_keeps_its_sizes_and_its_sieve_names(self, gradation):
        doc, _gt = gradation
        engine = FakeEngine([{"final": _gradation_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        points = result.tests[0].result.percent_passing
        assert [p.sieve for p in points] == [
            "3 in", "3/4 in", "No. 4", "No. 40", "No. 200"]
        assert points[-1].percent_passing == 43.0
        assert points[-1].size.value == 0.075
        assert points[-1].size.unit == "mm"

    def test_a_summary_table_is_one_test_with_rows(self, summary):
        doc, gt = summary
        rows = [ReadRow(investigation_id=r["investigation_id"],
                        sample_id=r["sample_id"], depth_top=r["depth_top"],
                        wc=r.get("wc"), ll=r.get("ll"),
                        pl=(ReadReported(value=r["pl"]) if r.get("pl")
                            else None),
                        pi=r.get("pi"),
                        passing=[ReadPassing(sieve="No. 200",
                                             percent_passing=r["passing_200"],
                                             size_mm=0.075)])
                for r in gt.rows]
        reading = LabSheetReading(
            depth_unit="m", pages_read=[0],
            tests=[ReadTest(kind="summary_table", rows=rows, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert len(result.tests) == 1
        table = result.tests[0].result
        assert isinstance(table, SummaryTableResult)
        assert len(table.rows) == 4
        assert table.rows[0].investigation_id == "A-1"
        assert table.rows[0].ll == 41.0
        assert table.rows[3].percent_passing[0].percent_passing == 94.0

    def test_a_certificate_page_is_recorded_as_holding_no_results(
            self, certificate):
        doc, _gt = certificate
        reading = LabSheetReading(
            depth_unit="m", pages_read=[0],
            tests=[ReadTest(kind="other", no_results=True,
                            investigation_id="TP-3", depth_top=0.5,
                            lab_sample_id="25A0123-01", prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        test = result.tests[0]
        assert isinstance(test.result, OtherResult)
        assert test.result.no_results is True
        assert test.result.fields["lab_sample_id"] == "25A0123-01"

    def test_a_value_printed_as_words_stays_words(self, atterberg):
        doc, _gt = atterberg
        reading = LabSheetReading(
            depth_unit="ft", pages_read=[0],
            tests=[ReadTest(kind="chemical", investigation_id="B-4",
                            depth_top=7.5,
                            chloride=ReadReported(text="<10", unit="mg/kg"),
                            pH=ReadReported(value=8.4),
                            resistivity=ReadReported(value=1261.0,
                                                     unit="ohm-cm"),
                            prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        chemistry = result.tests[0].result
        assert isinstance(chemistry, ChemicalResult)
        assert chemistry.chloride == "<10"
        assert chemistry.pH == 8.4
        assert chemistry.resistivity.value == 1261.0
        assert chemistry.resistivity.unit == "ohm-cm"

    def test_an_unknown_kind_becomes_other_and_is_recorded(self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(test={"kind": "vane_shear"})
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].kind == "other"
        assert any("vocabulary" in u["why"] for u in result.unresolved)

    def test_a_kind_the_trade_spells_differently_still_lands(self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(test={"kind": "consolidation"})
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].kind == "swell_consolidation"

    def test_a_strength_test_keeps_its_specimens(self, atterberg):
        doc, _gt = atterberg
        reading = LabSheetReading(
            depth_unit="m", pages_read=[0],
            tests=[ReadTest(
                kind="triaxial", test_type="CU", investigation_id="B-4",
                depth_top=4.0, prov=_prov(),
                specimens=[
                    ReadSpecimen(specimen_id="1",
                                 confining=ReadQuantity(value=100.0,
                                                        unit="kPa"),
                                 peak_deviator=ReadQuantity(value=185.0,
                                                            unit="kPa"),
                                 strain_at_peak_percent=10.7),
                    ReadSpecimen(specimen_id="2",
                                 peak_deviator=ReadQuantity(value=195.0,
                                                            unit="kPa"))])])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        strength = result.tests[0].result
        assert isinstance(strength, StrengthResult)
        assert strength.kind == "triaxial" and strength.test_type == "CU"
        assert len(strength.specimens) == 2
        assert strength.specimens[0].confining.value == 100.0


# ---------------------------------------------------------------------------
# the four gates
# ---------------------------------------------------------------------------

class TestWhatPythonRefuses:
    def test_a_depth_off_any_borehole_is_refused(self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(test={"depth_top": 4000.0})
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].depth_top is None
        assert any("outside 0 to" in u["why"] for u in result.unresolved)
        # The rest of the sheet survives the refusal.
        assert result.tests[0].result.ll == 48.0

    def test_a_negative_depth_is_refused(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"depth_top": -3.0})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].depth_top is None

    def test_a_percentage_past_a_hundred_is_refused(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 148.0, "pl": 22.0, "pi": 26.0})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].result.ll is None
        assert result.tests[0].result.pl == 22.0
        assert any("not a percentage" in u["why"] for u in result.unresolved)

    def test_a_liquid_limit_below_the_plastic_limit_takes_all_three(
            self, atterberg):
        """No soil does it, so which of the three is wrong cannot be known."""
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 22.0, "pl": 48.0, "pi": 26.0})}])
        result = read_lab_sheet(doc, [0], engine)
        limits = result.tests[0].result
        assert (limits.ll, limits.pl, limits.pi) == (None, None, None)
        assert any("below plastic limit" in u["why"]
                   for u in result.unresolved)

    def test_a_grading_that_runs_the_wrong_way_is_refused_whole(
            self, gradation):
        doc, _gt = gradation
        reading = _gradation_reading()
        reading.tests[0].series[0].points[-1].y = 99.0   # more passes 0.075
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].result.percent_passing == []
        assert any("runs the wrong way" in u["why"]
                   for u in result.unresolved)

    def test_a_provenance_naming_another_page_loses_its_box(self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(test={"prov": _prov(page=7)})
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].prov[0].bbox is None
        assert any("not part of this sheet" in u["why"]
                   for u in result.unresolved)

    def test_what_the_reader_could_not_settle_is_kept(self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(
            unsettled=[Unsettled(what="the shrinkage limit", page=0,
                                 why="the box is blank")])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert any(u["refused_by"] == "reader" and "shrinkage" in u["what"]
                   for u in result.unresolved)


# ---------------------------------------------------------------------------
# looking at a plot
# ---------------------------------------------------------------------------

class TestZoomPlot:
    def test_the_tool_is_offered_and_renders_the_region(self, gradation):
        doc, gt = gradation
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 0, "bbox": list(gt.plot_bbox),
                                      "why": "read the curve"})]},
            {"final": _gradation_reading()},
        ])
        result = read_lab_sheet(doc, [0], engine)
        assert engine.calls[0]["tools"] == ["zoom_plot"]
        assert result.tool_calls == 1
        blocks = engine.tool_results()[0]["content"]
        assert any(b.get("type") == "image" for b in blocks)
        assert any("dpi" in b.get("text", "") for b in blocks)

    def test_a_page_outside_the_sheet_is_an_error_not_a_stop(self, gradation):
        doc, gt = gradation
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 9, "bbox": list(gt.plot_bbox)})]},
            {"final": _gradation_reading()},
        ])
        result = read_lab_sheet(doc, [0], engine)
        assert engine.tool_results()[0]["is_error"] is True
        assert len(result.tests) == 2          # the reading still happened

    def test_a_digitised_curve_flags_the_whole_test(self, gradation):
        doc, _gt = gradation
        reading = _gradation_reading()
        reading.tests[0].series[0].digitised = True
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].curves_digitised is True
        assert result.tests[0].prov[0].method == "vision"
        assert any("digitised" in c["why"] for c in result.changes)
        # The test that did NOT digitise anything is not flagged.
        assert result.tests[1].curves_digitised is False

    def test_a_value_read_off_the_picture_is_recorded_as_a_change(
            self, atterberg):
        doc, _gt = atterberg
        reading = _atterberg_reading(
            test={"prov": _prov(from_image=True, note="the box is a scan")})
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].prov[0].method == "vision"
        assert any("scan" in c["why"] for c in result.changes)


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_a_tabulated_sheet_costs_one_call(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        assert engine.n_calls == 1
        assert result.model_calls == 1

    def test_the_answer_call_is_always_reserved(self, gradation):
        """A reader that keeps zooming runs out of LOOKING, never of
        answering."""
        doc, gt = gradation
        zoom = {"tools": [("zoom_plot", {"page": 0,
                                         "bbox": list(gt.plot_bbox)})]}
        engine = FakeEngine([zoom, zoom, zoom, {"final": _gradation_reading()}])
        result = read_lab_sheet(doc, [0], engine, budget=4)
        assert result.model_calls == 4
        assert result.tool_calls == 3
        assert len(result.tests) == 2

    def test_the_ceiling_holds_whatever_the_caller_asks_for(self, gradation):
        doc, gt = gradation
        zoom = {"tools": [("zoom_plot", {"page": 0,
                                         "bbox": list(gt.plot_bbox)})]}
        engine = FakeEngine([zoom] * 3 + [{"final": _gradation_reading()}])
        result = read_lab_sheet(doc, [0], engine, budget=99)
        assert result.model_calls == MAX_MODEL_CALLS
        # On the last allowed call the tool is withdrawn, so the model has
        # nothing left to do but answer.
        assert engine.calls[-1]["tools"] == []
        assert engine.calls[0]["tools"] == ["zoom_plot"]

    def test_a_reader_that_never_answers_is_an_error_not_an_empty_sheet(
            self, gradation):
        doc, gt = gradation
        zoom = {"tools": [("zoom_plot", {"page": 0,
                                         "bbox": list(gt.plot_bbox)})]}
        engine = FakeEngine([zoom] * 4)
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_lab_sheet(doc, [0], engine)

    def test_running_past_the_script_is_a_failure_not_a_guess(self, atterberg):
        doc, gt = atterberg
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 0, "bbox": list(gt.plot_bbox)})]},
        ])
        with pytest.raises(ScriptExhausted):
            read_lab_sheet(doc, [0], engine)

    def test_no_structured_answer_is_an_error(self, atterberg):
        """Prose is not an answer, and the reader says so rather than
        returning an empty sheet."""
        doc, _gt = atterberg
        engine = FakeEngine([{"text": "I could not read this sheet"}] * 4)
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_lab_sheet(doc, [0], engine)

    def test_a_sheet_with_no_pages_is_refused_up_front(self, atterberg):
        doc, _gt = atterberg
        with pytest.raises(ValueError, match="at least one page"):
            read_lab_sheet(doc, [], FakeEngine([]))


def test_the_tool_surface_is_the_one_tool_it_should_be():
    assert [t["name"] for t in LAB_TOOLS] == ["zoom_plot"]
    schema = LAB_TOOLS[0]["input_schema"]
    assert schema["required"] == ["page", "bbox"]
