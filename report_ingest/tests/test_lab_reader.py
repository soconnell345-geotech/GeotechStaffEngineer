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

from report_ingest.floor import MergeLog
from report_ingest.lab_floor import (
    floor_from_tables, kind_from_title, merge_lab_tests,
)
from report_ingest.lab_reader import (
    KIND_DEFINITIONS, LAB_READER_SYSTEM, LAB_TOOLS, LINK_FROM_LABEL,
    FLOOR_PER_PAGE_ENV, LINK_FROM_LABEL_ENV, MAX_LAB_PAGES, LabSheetReading,
    MAX_MODEL_CALLS, floor_per_page_enabled, use_floor_per_page,
    ReadPassing, ReadPoint, ReadProv, ReadQuantity, ReadReported, ReadRow,
    ReadSeries, ReadSpecimen, ReadTest, SERIES_NAMES, Unsettled, lab_windows,
    link_from_label_enabled, read_lab_sheet, reader_system, serialise_page,
    use_link_from_label,
)
from report_ingest.lab_scoring import score_record, score_tables
from report_ingest.model import (
    AtterbergResult, ChemicalResult, GradationResult, OtherResult,
    StrengthResult, SummaryTableResult,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted

pymupdf = pytest.importorskip("pymupdf", reason="the fixtures are drawn with PyMuPDF")


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
        doc_ = pymupdf.open()
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
    def test_an_answer_given_beside_a_look_is_taken(self, atterberg):
        # The app's chat model can answer and ask to look in one turn; the
        # answer is whole, so the look is not paid for (the review of RI-03).
        doc, _gt = atterberg
        engine = FakeEngine([{"tools": [("zoom_plot", {"page": 0})],
                              "final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        assert engine.n_calls == 1
        assert result.tests[0].result.ll == 48.0

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
        assert result.model_tests[0].depth_top is None
        assert any("outside 0 to" in u["why"] for u in result.unresolved)
        # The rest of the sheet survives the refusal ...
        assert result.model_tests[0].result.ll == 48.0
        # ... and the record carries the depth the sheet PRINTS, from the
        # floor, since the model's was refused and left the slot empty.
        assert result.tests[0].depth_top.value == 7.5

    def test_a_negative_depth_is_refused(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"depth_top": -3.0})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.model_tests[0].depth_top is None
        assert result.tests[0].depth_top.value == 7.5     # the floor's

    def test_a_limit_past_a_hundred_is_kept_and_the_table_still_stands(
            self, atterberg):
        """RI-89: a liquid limit can pass 100 (UFS 3-220-10 section
        8-2.2.2 prints clays to 143), so the model's 148 is not refused.
        The table prints 48: the record carries the table's, and the
        model's stands beside it as a disagreement."""
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 148.0, "pl": 22.0, "pi": 26.0})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.model_tests[0].result.ll == 148.0
        assert not any("not a percentage" in u["why"]
                       for u in result.unresolved)
        assert result.tests[0].result.ll == 48.0
        (row,) = [d for d in result.disagreements
                  if d["what"] == "atterberg: ll"]
        assert (row["floor"], row["model"], row["kept"]) == (
            "48", "148", "floor")

    def test_a_liquid_limit_below_the_plastic_limit_takes_all_three(
            self, atterberg):
        """No soil does it, so which of the three is wrong cannot be known."""
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 22.0, "pl": 48.0, "pi": 26.0})}])
        result = read_lab_sheet(doc, [0], engine)
        limits = result.model_tests[0].result
        assert (limits.ll, limits.pl, limits.pi) == (None, None, None)
        assert any("below plastic limit" in u["why"]
                   for u in result.unresolved)
        # The floor's three limits fill the empty slots.
        merged = result.tests[0].result
        assert (merged.ll, merged.pl, merged.pi) == (48.0, 22.0, 26.0)

    def test_a_grading_that_runs_the_wrong_way_is_refused_whole(
            self, gradation):
        doc, _gt = gradation
        reading = _gradation_reading()
        reading.tests[0].series[0].points[-1].y = 99.0   # more passes 0.075
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.model_tests[0].result.percent_passing == []
        assert any("runs the wrong way" in u["why"]
                   for u in result.unresolved)
        # The table under the plot tabulates the series; the record has it.
        assert [p.percent_passing
                for p in result.tests[0].result.percent_passing] == \
            [100.0, 98.0, 87.0, 61.0, 43.0]

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
        assert result.tests[0].prov[0].method == "model_from_picture"
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
        assert result.tests[0].prov[0].method == "model_from_picture"
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


# ---------------------------------------------------------------------------
# a long run of sheets is read in windows (RI-07)
# ---------------------------------------------------------------------------

@pytest.fixture
def run_of(tmp_path):
    """A run of ``n`` water-content sheets, as planlens makes one work item
    of consecutive sheets with the same title. Each page names its own
    boring, or, with ``starts``, only those pages do and the rest are a
    test's continuation sheets."""
    opened = []

    def build(n: int, starts=None):
        pdf = pymupdf.open()
        for i in range(n):
            page = pdf.new_page(width=612, height=792)
            page.insert_text((72, 72), "WATER CONTENT OF SOIL", fontsize=14)
            if starts is None or i in starts:
                page.insert_text((72, 110), f"Boring: B-{i + 1}",
                                 fontsize=10)
                page.insert_text((72, 126), "Depth: 5.0 ft", fontsize=10)
            else:
                page.insert_text((72, 110), "(continued)", fontsize=10)
            page.insert_text((72, 142), f"Water content: {20 + i}.0 %",
                             fontsize=10)
        data = pdf.tobytes()
        pdf.close()
        path = tmp_path / f"run{n}_{len(opened)}.pdf"
        path.write_bytes(data)
        from planlens.document import open_document
        doc = open_document(str(path))
        opened.append(doc)
        return doc

    yield build
    for doc in opened:
        doc.close()


def _run_reading(pages) -> LabSheetReading:
    """A careful reading of the run's sheets on ``pages``."""
    return LabSheetReading(
        depth_unit="ft", pages_read=list(pages),
        tests=[ReadTest(kind="water_content", investigation_id=f"B-{p + 1}",
                        depth_top=5.0, wc=20.0 + p,
                        prov=_prov(page=p, bbox=(72.0, 130.0, 300.0, 146.0)))
               for p in pages])


def _brief_of(call) -> str:
    return call["messages"][0]["content"][0]["text"]


class TestTheWindow:
    def test_the_windows_are_consecutive_and_near_equal(self):
        assert lab_windows(range(12)) == [list(range(12))]
        assert [len(w) for w in lab_windows(range(13))] == [7, 6]
        assert [len(w) for w in lab_windows(range(17))] == [9, 8]
        assert [len(w) for w in lab_windows(range(24))] == [12, 12]
        assert [len(w) for w in lab_windows(range(25))] == [9, 8, 8]
        assert sum(lab_windows(range(40, 57)), []) == list(range(40, 57))
        assert lab_windows([]) == []
        assert MAX_LAB_PAGES == 12

    def test_a_cut_moves_to_the_nearest_page_that_starts_a_specimen(self):
        # The even cut is before page 7; a specimen starts on 5 and on 10.
        assert lab_windows(range(14), starts=[0, 5, 10]) == \
            [list(range(5)), list(range(5, 14))]
        # ... but never so far that a window outgrows the size.
        assert [len(w) for w in lab_windows(range(24), starts=[3])] == \
            [12, 12]
        # No start in reach: the even split.
        assert [len(w) for w in lab_windows(range(14), starts=[0])] == [7, 7]

    def test_a_cut_at_a_new_specimen_needs_no_note(self, run_of):
        doc = run_of(14, starts={0, 5, 10})
        engine = FakeEngine([{"final": _run_reading([0])},
                             {"final": _run_reading([5, 10])}])
        result = read_lab_sheet(doc, list(range(14)), engine)
        assert [c["n_images"] for c in engine.calls] == [5, 9]
        assert result.windows == [list(range(5)), list(range(5, 14))]
        assert not [u for u in result.unresolved
                    if "separate windows" in u["why"]]

    def test_a_cut_inside_one_specimen_is_said(self, run_of):
        """No page in reach starts a specimen, so the cut stays at the even
        split; the two halves are kept as read, and the boundary is a line
        in ``unresolved`` rather than a guessed merge."""
        doc = run_of(14, starts={0})
        engine = FakeEngine([{"final": _run_reading([0])},
                             {"final": _run_reading([7])}])
        result = read_lab_sheet(doc, list(range(14)), engine)
        assert result.windows == [list(range(7)), list(range(7, 14))]
        (note,) = [u for u in result.unresolved
                   if "separate windows" in u["why"]]
        assert note["what"] == "pages 6-7" and note["page"] == 7
        assert len(result.model_tests) == 2          # both halves kept

    def test_a_run_that_fits_is_read_whole_as_before(self, run_of):
        doc = run_of(MAX_LAB_PAGES)
        pages = list(range(MAX_LAB_PAGES))
        engine = FakeEngine([{"final": _run_reading(pages)}])
        result = read_lab_sheet(doc, pages, engine)
        assert engine.n_calls == 1
        assert engine.calls[0]["n_images"] == MAX_LAB_PAGES
        assert "one window of a run" not in _brief_of(engine.calls[0])
        assert result.windows == []
        assert "windows" not in result.to_dict()
        assert len(result.tests) == MAX_LAB_PAGES

    def test_a_longer_run_is_read_in_windows_and_put_end_to_end(
            self, run_of):
        doc = run_of(14)
        pages = list(range(14))
        engine = FakeEngine([{"final": _run_reading(pages[:7])},
                             {"final": _run_reading(pages[7:])}])
        result = read_lab_sheet(doc, pages, engine)
        assert engine.n_calls == 2
        assert [c["n_images"] for c in engine.calls] == [7, 7]
        first, second = (_brief_of(c) for c in engine.calls)
        assert "page(s) 0, 1, 2, 3, 4, 5, 6." in first
        assert "page(s) 7, 8, 9, 10, 11, 12, 13." in second
        for brief in (first, second):
            assert "one window of a run of laboratory sheets on pages 0-13" \
                in brief
        assert [t.investigation_id for t in result.tests] == \
            [f"B-{p + 1}" for p in pages]
        assert result.pages == pages
        assert result.model_calls == 2 and result.cost["calls"] == 2
        assert result.windows == [pages[:7], pages[7:]]
        assert result.to_dict()["windows"] == [pages[:7], pages[7:]]

    def test_each_window_has_the_whole_budget(self, run_of):
        """A window is a sheet: the budget is a ceiling for each, so a
        long run is not starved because its first window needed a nudge."""
        doc = run_of(14)
        pages = list(range(14))
        engine = FakeEngine([{"text": "thinking"},
                             {"final": _run_reading(pages[:7])},
                             {"text": "thinking"},
                             {"final": _run_reading(pages[7:])}])
        result = read_lab_sheet(doc, pages, engine, budget=2)
        assert result.model_calls == 4
        assert len(result.tests) == 14

    def test_a_window_with_no_answer_is_named_and_the_rest_kept(self, run_of):
        doc = run_of(14)
        pages = list(range(14))
        engine = FakeEngine([{"text": "no answer"},
                             {"final": _run_reading(pages[7:])}])
        result = read_lab_sheet(doc, pages, engine, budget=1)
        assert [t.investigation_id for t in result.tests] == \
            [f"B-{p + 1}" for p in pages[7:]]
        (gap,) = [u for u in result.unresolved if u["what"] == "pages 0-6"]
        assert "no reading came back" in gap["why"]
        assert any("pages 0-6: not read" in w for w in result.warnings)

    def test_a_run_no_window_answers_is_an_error(self, run_of):
        doc = run_of(14)
        engine = FakeEngine([{"text": "no answer"}] * 2)
        with pytest.raises(RuntimeError, match="no structured reading"):
            read_lab_sheet(doc, list(range(14)), engine, budget=1)


# ---------------------------------------------------------------------------
# the specimen-label paragraph, behind a switch that is off (RI-07)
# ---------------------------------------------------------------------------

class TestTheSpecimenLabelSwitch:
    def test_it_is_off_and_the_prompt_unchanged_by_default(
            self, atterberg, monkeypatch):
        monkeypatch.delenv(LINK_FROM_LABEL_ENV, raising=False)
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        assert not link_from_label_enabled()
        assert engine.calls[0]["system"] == LAB_READER_SYSTEM
        assert result.link_from_label is False
        assert "link_from_label" not in result.to_dict()

    def test_on_it_adds_one_paragraph_after_the_link_rule(
            self, atterberg, monkeypatch):
        monkeypatch.setenv(LINK_FROM_LABEL_ENV, "1")
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        system = engine.calls[0]["system"]
        assert system == reader_system(True)
        assert system.replace("\n" + LINK_FROM_LABEL, "", 1) == \
            LAB_READER_SYSTEM
        assert system.index(LINK_FROM_LABEL) > \
            system.index("else will deal with it.")
        assert result.link_from_label is True
        assert result.to_dict()["link_from_label"] is True

    def test_the_block_outranks_the_environment(self, monkeypatch):
        monkeypatch.setenv(LINK_FROM_LABEL_ENV, "on")
        assert link_from_label_enabled()
        with use_link_from_label(False):
            assert not link_from_label_enabled()
            assert reader_system() == LAB_READER_SYSTEM
        assert link_from_label_enabled()

    def test_a_sample_number_alone_is_still_no_hole(self):
        assert "names no hole" in LINK_FROM_LABEL
        assert "exactly as printed" in LINK_FROM_LABEL


# ---------------------------------------------------------------------------
# the floor: the page's tables and title are the first voter
# ---------------------------------------------------------------------------

def _truth_of(gt, kind: str, sheet_id: str) -> dict:
    """The fixture's answers in the lab scorer's truth shape."""
    if gt.rows:
        rows = []
        for row in gt.rows:
            block = {"investigation_id": row["investigation_id"],
                     "sample_id": row["sample_id"],
                     "depth_top": row["depth_top"], "wc_pct": row["wc"],
                     "percent_finer": {"No. 200": row["passing_200"]}}
            for name in ("ll", "pl", "pi"):
                if row.get(name) is not None:
                    block[name] = row[name]
            rows.append(block)
        return {"id": sheet_id, "kind": kind, "depth_unit": gt.depth_unit,
                "rows": rows}
    block = {"investigation_id": gt.investigation_id,
             "depth_top": gt.depth_top}
    for name, value in gt.values.items():
        if name == "uscs":
            continue
        key = {"water_content": "wc_pct", "gravel_percent": "gravel_pct",
               "sand_percent": "sand_pct",
               "fines_percent": "fines_pct"}.get(name, name)
        block[key] = value
    if gt.passing:
        block["percent_finer"] = {sieve: pct for sieve, pct in gt.passing}
    return {"id": sheet_id, "kind": kind, "depth_unit": gt.depth_unit,
            "tests": [block]}


class TestTheFloor:

    def test_the_title_names_the_kind(self):
        assert kind_from_title(["ATTERBERG LIMITS RESULTS"])[0] == "atterberg"
        assert kind_from_title(["PARTICLE SIZE DISTRIBUTION"])[0] == \
            "gradation"
        assert kind_from_title(["SUMMARY OF LABORATORY TEST RESULTS"])[0] \
            == "summary_table"
        assert kind_from_title(["CERTIFICATE OF ANALYSIS"])[0] == "other"
        assert kind_from_title(["ANALYSE GRANULOMETRIQUE"])[0] == "gradation"
        assert kind_from_title(["Page 2 of 9"]) == ("", "")

    def test_the_floor_reads_the_tables_the_title_and_the_link(
            self, gradation):
        doc, gt = gradation
        floor = floor_from_tables(doc, [0], "RXX")
        assert floor.kind == "gradation"
        assert floor.link.investigation_id == "SB-11"
        assert floor.link.depth_top == 12.0 and floor.link.unit == "ft"
        assert [t.kind for t in floor.tests] == ["gradation", "atterberg"]
        grading, limits = floor.tests
        assert [p.percent_passing for p in grading.result.percent_passing] \
            == [100.0, 98.0, 87.0, 61.0, 43.0]
        assert grading.result.percent_passing[-1].size.value == 0.075
        assert grading.result.fines_percent == 43.0
        assert (limits.result.ll, limits.result.pl, limits.result.pi) == \
            (31.0, 19.0, 12.0)
        assert limits.investigation_id == "SB-11"
        assert limits.depth_top.value == 12.0
        assert all(p.method in ("tables", "text") for t in floor.tests
                   for p in t.prov)
        assert all(0.0 < p.confidence <= 1.0 for t in floor.tests
                   for p in t.prov)

    def test_a_summary_table_seeds_a_row_per_specimen(self, summary):
        doc, gt = summary
        floor = floor_from_tables(doc, [0], "RXX")
        (table,) = floor.tests
        assert table.kind == "summary_table"
        rows = table.result.rows
        assert [r.investigation_id for r in rows] == ["A-1", "A-1", "A-2",
                                                      "A-3"]
        assert rows[0].wc == 18.2 and rows[0].ll == 41.0
        assert rows[1].ll is None                  # the blank cell stays blank
        assert rows[3].percent_passing[0].percent_passing == 94.0
        assert rows[0].depth_top.unit == "m"

    def test_a_certificate_seeds_the_kind_and_no_results(self, certificate):
        doc, _gt = certificate
        floor = floor_from_tables(doc, [0], "RXX")
        (page,) = floor.tests
        assert page.kind == "other"
        assert isinstance(page.result, OtherResult)
        assert page.result.no_results is True

    def test_the_model_is_shown_the_starting_record(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        read_lab_sheet(doc, [0], engine)
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE STARTING RECORD" in brief
        assert "kind from the title: atterberg" in brief
        assert "boring 'B-4'" in brief and "depth 7.5 ft" in brief
        assert "ll 48, pl 22, pi 26" in brief
        assert "THE STARTING RECORD, AND THE THREE THINGS" in LAB_READER_SYSTEM
        assert "never DROP one" in LAB_READER_SYSTEM

    def test_a_reply_that_drops_the_values_scores_no_lower_than_the_tables(
            self, gradation):
        doc, gt = gradation
        truth = _truth_of(gt, "gradation", "gradation__RXX_p0")
        before = score_tables(truth, doc, [0])
        # The model names the kind and the link and returns not one number.
        empty = LabSheetReading(
            depth_unit="ft", pages_read=[0],
            tests=[ReadTest(kind="gradation", investigation_id="SB-11",
                            sample_id="3", depth_top=12.0, prov=_prov())])
        engine = FakeEngine([{"final": empty}])
        result = read_lab_sheet(doc, [0], engine, report_id="RXX")
        after = score_record(truth, result.tests)
        for metric in ("index", "series"):
            assert after.scores[metric].found >= before.scores[metric].found
        assert after.scores["kind"].found == 1
        assert after.scores["link"].found == 1
        alone = score_record(truth, result.model_tests)
        assert alone.scores["index"].found < before.scores["index"].found
        assert len(result.kept) >= 5

    def test_a_reply_that_contradicts_a_table_value_keeps_both(
            self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 58.0})}])
        result = read_lab_sheet(doc, [0], engine)
        limits = result.tests[0].result
        assert limits.ll == 48.0                        # the table's stands
        (alt,) = [a for a in result.tests[0].prov[0].alternatives
                  if a.field == "ll"]
        assert alt.value == "58" and alt.method == "model"
        (row,) = [d for d in result.disagreements
                  if d["what"].endswith(": ll")]
        assert row["floor"] == "48" and row["model"] == "58"
        assert row["kept"] == "floor"
        assert row["confidence"]["floor"] > 0

    def test_a_correction_with_evidence_overrules_the_table(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 58.0,
                  "prov": _prov(note="the box prints 58; the 4 is a "
                                     "smudge")})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.tests[0].result.ll == 58.0
        (alt,) = [a for a in result.tests[0].prov[0].alternatives
                  if a.field == "ll"]
        assert alt.value == "48" and alt.method == "tables"
        (row,) = [d for d in result.disagreements
                  if d["what"].endswith(": ll")]
        assert row["kept"] == "model"

    def test_a_value_within_tolerance_is_reconciled_not_disputed(
            self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading()}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.disagreements == []
        assert result.reconciled >= 3
        assert result.tests[0].result.ll == 48.0

    def test_a_kind_and_a_link_the_model_adds_are_added(self, atterberg):
        doc, _gt = atterberg
        # The model reads a chemical suite the tables did not label.
        reading = LabSheetReading(
            depth_unit="ft", pages_read=[0],
            tests=[ReadTest(kind="atterberg", investigation_id="B-4",
                            sample_id="S-2", depth_top=7.5,
                            ll=48.0, pl=22.0, pi=26.0, prov=_prov()),
                   ReadTest(kind="chemical", investigation_id="B-4",
                            depth_top=7.5,
                            pH=ReadReported(value=8.4), prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.kinds == ["atterberg", "chemical"]
        assert any(a["what"] == "chemical test" for a in result.added)

    def test_a_summary_row_the_model_left_out_is_kept(self, summary):
        doc, gt = summary
        rows = [ReadRow(investigation_id=r["investigation_id"],
                        sample_id=r["sample_id"], depth_top=r["depth_top"],
                        wc=r.get("wc"), ll=r.get("ll"),
                        pl=(ReadReported(value=r["pl"]) if r.get("pl")
                            else None), pi=r.get("pi"))
                for r in gt.rows[:2]]                 # two of the four
        reading = LabSheetReading(
            depth_unit="m", pages_read=[0],
            tests=[ReadTest(kind="summary_table", rows=rows, prov=_prov())])
        engine = FakeEngine([{"final": reading}])
        result = read_lab_sheet(doc, [0], engine)
        table = result.tests[0].result
        assert len(table.rows) == 4
        assert table.rows[2].investigation_id == "A-2"
        # ... and the No. 200 column the model's rows did not carry is in.
        assert table.rows[0].percent_passing[0].percent_passing == 72.0
        assert sum(1 for k in result.kept if k["what"].startswith("row")) \
            >= 2

    def test_the_result_serialises_both_voters_and_the_merge(self, gradation):
        import json
        doc, _gt = gradation
        engine = FakeEngine([{"final": _gradation_reading()}])
        blob = read_lab_sheet(doc, [0], engine).to_dict()
        json.dumps(blob)
        assert blob["floor_kind"] == "gradation"
        assert [t["kind"] for t in blob["floor_tests"]] == \
            ["gradation", "atterberg"]
        assert [t["kind"] for t in blob["model_tests"]] == \
            ["gradation", "atterberg"]
        assert blob["reconciled"] > 0


# ---------------------------------------------------------------------------
# RI-56: each page its own link; a compaction named by its parts
# ---------------------------------------------------------------------------

@pytest.fixture
def water_run(tmp_path):
    from report_ingest.tests.lab_fixtures import build_water_content_run
    doc = _open(build_water_content_run(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def density_run(tmp_path):
    from report_ingest.tests.lab_fixtures import build_density_run
    doc = _open(build_density_run(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def bare_run(tmp_path):
    from report_ingest.tests.lab_fixtures import build_bare_run
    doc = _open(build_bare_run(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def limits_trials(tmp_path):
    from report_ingest.tests.lab_fixtures import build_limits_trials
    doc = _open(build_limits_trials(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def bare_empty_run(tmp_path):
    from report_ingest.tests.lab_fixtures import build_bare_empty_run
    doc = _open(build_bare_empty_run(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def particle_density_page(tmp_path):
    from report_ingest.tests.lab_fixtures import build_particle_density_page
    doc = _open(build_particle_density_page(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def grading_legend(tmp_path):
    from report_ingest.tests.lab_fixtures import build_grading_legend
    doc = _open(build_grading_legend(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def field_density(tmp_path):
    from report_ingest.tests.lab_fixtures import build_field_density_sheet
    doc = _open(build_field_density_sheet(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def compaction_trials(tmp_path):
    from report_ingest.tests.lab_fixtures import build_compaction_trials
    doc = _open(build_compaction_trials(), tmp_path)
    yield doc
    doc.close()


@pytest.fixture
def no_per_page_switch(monkeypatch):
    monkeypatch.delenv(FLOOR_PER_PAGE_ENV, raising=False)


class TestEachPageItsOwnLink:

    def test_each_page_s_values_carry_that_page_s_link(self, water_run):
        floor = floor_from_tables(water_run, [0, 1, 2], "RXX")
        got = [(t.kind, t.investigation_id,
                t.depth_top.value if t.depth_top else None, t.result.wc)
               for t in floor.tests]
        assert got == [("moisture_content", "B-7", 3.0, 21.4),
                       ("moisture_content", "B-8", 6.0, 17.9),
                       ("moisture_content", "", None, 25.0)]

    def test_a_page_printing_no_link_is_left_unbound_and_named(
            self, water_run):
        floor = floor_from_tables(water_run, [0, 1, 2], "RXX")
        assert floor.unbound == [{"page": 2, "values": 2, "linked": [0, 1]}]
        unbound = floor.tests[-1]
        assert unbound.investigation_id == "" and unbound.depth_top is None

    def test_a_sheet_printing_no_link_anywhere_names_no_page(self, water_run):
        floor = floor_from_tables(water_run, [2], "RXX")
        assert floor.unbound == []

    def test_the_brief_shows_the_floor_as_before_with_the_switch_off(
            self, water_run, no_per_page_switch):
        # Switch off, the brief is as before RI-56: one test under the
        # first page's link, its values the first read.
        floor = floor_from_tables(water_run, [0, 1, 2], "RXX")
        (before,) = floor.as_before.tests
        assert (before.investigation_id, before.result.wc) == ("B-7", 21.4)
        engine = FakeEngine([{"final": LabSheetReading(tests=[])}])
        result = read_lab_sheet(water_run, [0, 1, 2], engine,
                                report_id="RXX")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        record = brief.split("THE STARTING RECORD")[1].split("THE PAGE")[0]
        assert "boring 'B-7'" in record and "wc 21.4" in record
        assert "17.9" not in record and "wc 25" not in record
        # ... while the merge, and the record, take each page's own.
        assert [t.investigation_id for t in result.tests] == \
            ["B-7", "B-8", ""]
        (row,) = [u for u in result.unresolved
                  if u["what"] == "lab_tests floor link"]
        assert row["page"] == 2 and "left unbound" in row["why"]
        assert "page(s) 0, 1" in row["why"]


class TestTheTitleKindIsCorrected:
    """The reader's kinds: a density or water content test reports that one
    measurement and nothing else; a compaction prints its own parts; a
    particle density is the specific gravity of solids (FHWA-NHI-06-088
    Vol. I, 5.3.6)."""

    @pytest.mark.parametrize("lines,kind", [
        (["Particle density by gas jar"], "specific_gravity"),
        (["Particle density of soil solids"], "specific_gravity"),
        (["BULK AND PARTICLE DENSITY"], "density"),
        (["DRY DENSITY", "Particle density assumed"], "density"),
        # a mould, the word compaction or a maximum dry density alone is
        # not enough; two different ones are
        (["BULK DENSITY", "Mould volume, cm3"],
         "density"),
        (["BULK DENSITY", "Mould mass, grams", "Compacted soil and mould"],
         "compaction"),
        # the review's six: field and index density tests quote a
        # compaction and are not one
        (["Field density, fill lift 3", "Proctor maximum 1.95 Mg/m3"],
         "density"),
        (["NUCLEAR GAUGE DENSITY", "Relative compaction (%)"], "density"),
        (["SAND CONE DENSITY", "Max. dry density (pcf)",
          "Relative compaction (%)"], "density"),
        (["MAXIMUM INDEX DENSITY", "Mold volume (cm3)",
          "Compacted on a vibrating table"], "density"),
        (["RELATIVE DENSITY", "Maximum dry density", "Minimum dry density"],
         "density"),
        (["UNIT WEIGHT AND MOISTURE CONTENT", "Mold no. 3"],
         "moisture_content"),
        (["In-place density of fill", "Optimum moisture content 12 %"],
         "density"),
        # round 2: in situ and drive cylinder tests, and a lab sheet giving
        # a relative compaction, quote a compaction's maximum dry density
        (["IN SITU DENSITY, LIFT 4", "Maximum dry density 1.98 Mg/m3",
          "Relative compaction 96 %"], "density"),
        (["Density by drive cylinder", "MDD (Mg/m3)",
          "Relative compaction (%)"], "density"),
        (["LAB DENSITY RESULTS, FILL", "Relative compaction (%)",
          "MDD (Mg/m3)"], "density"),
        # round 3: other field methods, and a compaction given as a percent
        (["DRIVE TUBE DENSITY", "Max. dry density (Mg/m3)",
          "Compaction (%)"], "density"),
        (["CORE CUTTER DENSITY", "MDD (Mg/m3)", "Degree of compaction (%)"],
         "density"),
        (["DENSITY - RUBBER BALLOON METHOD", "Maximum dry density (pcf)",
          "Compacted fill, layer 2"], "density"),
        (["LAB DENSITY RESULTS", "Percent compaction (%)",
          "Max. dry density (pcf)"], "density"),
        (["Dry density / water content relationship"], "compaction"),
        (["Water content", "Maximum dry density", "Optimum moisture"],
         "compaction"),
        (["BULK DENSITY"], "density"),
        (["MOISTURE CONTENT", "Natural water content (%)"],
         "moisture_content"),
        (["DRY DENSITY", "CBR", "Rammer: 2.5 kg"], "density"),
        (["PROCTOR TEST, STANDARD EFFORT"], "compaction"),
        (["ATTERBERG LIMITS", "Mould"], "atterberg"),
    ])
    def test_the_title_kind(self, lines, kind):
        assert kind_from_title(lines)[0] == kind

    def test_the_line_naming_the_part_is_the_title(self):
        assert kind_from_title(["BULK DENSITY", "Rammer: 2.5 kg"]) == \
            ("compaction", "Rammer: 2.5 kg")

    def test_the_rule_before_it_is_kept_for_the_brief(self):
        assert kind_from_title(["BULK DENSITY", "Mould volume, cm3"],
                               corrected=False) == ("density", "BULK DENSITY")
        assert kind_from_title(["PARTICLE DENSITY"], corrected=False) == \
            ("density", "PARTICLE DENSITY")


#: The starting record master (before RI-56) gives the model for
#: :func:`lab_fixtures.build_density_run`, byte for byte. With the per-page
#: switch off the brief must still be exactly this.
_MASTER_DENSITY_RECORD = (
    "THE STARTING RECORD (read off the sheet's title and its detected "
    "tables before you were called, with the page and box each came from "
    "and the reader's confidence -- build on it; add, correct with the box "
    "and a note, never drop)\n"
    "kind from the title: density ('BULK DENSITY')\n"
    "link printed on the sheet: boring 'TP-4', depth 1 m\n"
    "density (page 0 box 60,150,310,182, page 0 box 60,150,310,182, "
    "page 1 box 60,150,310,182, page 1 box 60,150,310,182)  "
    "[confidence 0.85]\n"
    "  wc 14.2, wet_density 2.05 Mg/m3\n")


def _brief_and_result(doc, pages, on=None):
    engine = FakeEngine([{"final": LabSheetReading(tests=[])}])
    if on is None:
        result = read_lab_sheet(doc, pages, engine, report_id="RXX")
    else:
        with use_floor_per_page(on):
            result = read_lab_sheet(doc, pages, engine, report_id="RXX")
    (call,) = engine.calls
    return call["messages"][0]["content"][0]["text"], call["system"], result


def _record_of(brief):
    start = brief.index("THE STARTING RECORD")
    return brief[start:brief.index("\nTHE PAGE\n", start)]


class TestThePerPageFloorSwitch:
    """RI-56: the merge always takes the per-page floor and the corrected
    kind; GEOTECH_INGEST_LAB_FLOOR_PER_PAGE also shows them to the model."""

    def test_off_by_default_the_brief_is_master_s_byte_for_byte(
            self, density_run, no_per_page_switch):
        brief, _system, result = _brief_and_result(density_run, [0, 1])
        assert not floor_per_page_enabled()
        assert _record_of(brief) == _MASTER_DENSITY_RECORD
        assert result.floor_per_page is False
        assert "floor_per_page" not in result.to_dict()

    @pytest.mark.parametrize("fixture,pages", [
        ("density_run", [0, 1]), ("water_run", [0, 1, 2])])
    def test_the_switch_changes_the_starting_record_and_nothing_else(
            self, request, fixture, pages, no_per_page_switch):
        doc = request.getfixturevalue(fixture)
        off, off_system, _ = _brief_and_result(doc, pages)
        on, on_system, _ = _brief_and_result(doc, pages, on=True)
        assert on != off and on_system == off_system
        assert on.replace(_record_of(on), _record_of(off)) == off

    def test_on_the_brief_shows_the_corrected_kind_and_each_page_s_link(
            self, density_run, no_per_page_switch):
        brief, _system, result = _brief_and_result(density_run, [0, 1],
                                                   on=True)
        record = _record_of(brief)
        assert "kind from the title: compaction ('Mould volume (cm3): " \
            "944')" in record
        assert "link printed on page 0: boring 'TP-4', depth 1 m" in record
        # the water content and density are the compaction's, not a
        # moisture test of their own
        assert record.count("link printed on its page: boring 'TP-5', "
                            "depth 2 m") == 1
        assert "moisture_content" not in record
        assert "also printed: wet_density 2.11 Mg/m3, wc 12.6" in record
        assert result.floor_per_page is True
        assert result.to_dict()["floor_per_page"] is True

    def test_a_sheet_whose_floor_is_the_same_reads_the_same_in_both_arms(
            self, atterberg, no_per_page_switch):
        doc, _gt = atterberg
        off, _system, _result = _brief_and_result(doc, [0])
        on, _system, result = _brief_and_result(doc, [0], on=True)
        assert on == off and result.floor_per_page is True

    def test_on_a_page_printing_no_link_is_said_to_be_unbound(
            self, water_run, no_per_page_switch):
        brief, _system, _result = _brief_and_result(water_run, [0, 1, 2],
                                                    on=True)
        assert "link printed on its page: none (its values are left " \
            "unbound)\n  wc 25," in _record_of(brief)

    def test_the_merge_is_the_same_in_both_arms(self, water_run,
                                                no_per_page_switch):
        _b, _s, off = _brief_and_result(water_run, [0, 1, 2], on=False)
        _b, _s, on = _brief_and_result(water_run, [0, 1, 2], on=True)
        assert [t.model_dump() for t in on.tests] == \
            [t.model_dump() for t in off.tests]
        assert on.floor_kind == off.floor_kind == "moisture_content"

    def test_the_block_outranks_the_environment(self, monkeypatch):
        monkeypatch.setenv(FLOOR_PER_PAGE_ENV, "on")
        assert floor_per_page_enabled()
        with use_floor_per_page(False):
            assert not floor_per_page_enabled()
        assert floor_per_page_enabled()


class TestTheMergeFollowsThePage:
    """RI-56: with each page's own link, a page that prints only a depth
    (or nothing) matches more than one of the model's specimens; the page
    the model put each one on decides."""

    @staticmethod
    def _test(kind, page, method, **over):
        from report_ingest.model import CBRResult, LabTest, Provenance
        base = dict(kind=kind, pages=[10, 11, 12],
                    prov=[Provenance(page=page, method=method,
                                     confidence=0.85)])
        result = over.pop("result")
        base.update(over)
        return LabTest(result=CBRResult(**result), **base)

    def test_a_continuation_page_joins_the_specimen_above_it(self):
        from report_ingest.model import Quantity

        def m(value):
            return Quantity(value=value, unit="m")

        floor = [
            self._test("cbr", 10, "tables", sample_id="S-5",
                       depth_top=m(3.0), result={"cbr_percent": 12.0}),
            # page 11 continues S-5 and prints no sample or depth
            self._test("cbr", 11, "tables", result={"wc": 8.0}),
            self._test("cbr", 12, "tables", sample_id="S-6",
                       depth_top=m(3.1), result={"cbr_percent": 30.0,
                                                 "wc": 9.0}),
        ]
        model = [
            self._test("cbr", 10, "model", investigation_id="B-1",
                       sample_id="S-5", depth_top=m(3.0),
                       result={"cbr_percent": 12.0, "wc": 8.0}),
            self._test("cbr", 12, "model", investigation_id="B-1",
                       sample_id="S-6", depth_top=m(3.1),
                       result={"cbr_percent": 30.0, "wc": 9.0}),
        ]
        merged, log = merge_lab_tests(floor, model, MergeLog())
        assert len(merged) == 2
        assert log.disagreements == []
        assert [t.result.cbr_percent for t in merged] == [12.0, 30.0]


def _moisture(page, method, wc, dry=None, **link):
    from report_ingest.model import (
        LabTest, MoistureDensityResult, Provenance, Quantity,
    )
    if "depth" in link:
        link["depth_top"] = Quantity(value=link.pop("depth"), unit="m")
    return LabTest(
        kind="moisture_content", pages=[10, 11, 12],
        prov=[Provenance(page=page, method=method, confidence=0.85)],
        result=MoistureDensityResult(
            kind="moisture_content", wc=wc,
            dry_density=(Quantity(value=dry, unit="pcf")
                         if dry is not None else None)), **link)


class TestALetterCIsACohesionOnlyWhereOneCanBe:
    """RI-71: the per-page floor reads a row headed by the one letter C as
    a cohesion only on a shear test's sheet or where the row prints a unit
    of stress; a grading chart's legend key is not one."""

    @pytest.mark.parametrize("row, kind, read", [
        (["C", "5", "Drum"], "gradation", False),
        (["C", "5"], "", False),
        (["C", "5"], "summary_table", False),
        (["C", "12"], "direct_shear", True),
        (["c'", "12"], "triaxial", True),
        (["C (kPa)", "12"], "gradation", True),
        (["C", "12", "kPa"], "", True),
        (["C", "0.4", "kg/cm2"], "atterberg", True),
        (["Cohesion", "12"], "gradation", True),
        (["Bath temp (°C)", "21"], "direct_shear", False),
        # RI-71 review: a temperature anywhere in the row; a unit of
        # stress only in the label, as a cell, in brackets or after a number
        (["C", "26", "°C"], "direct_shear", False),
        (["C", "6", "Bar 4"], "gradation", False),
        (["C", "0.6", "59 kPa"], "gradation", True),
        (["C", "6", "(kPa)"], "gradation", True)])
    def test_the_rows(self, row, kind, read):
        from report_ingest.lab_floor import (
            _field_of, _read_property_table, _Values,
        )
        values, before = _Values(), _Values()
        _read_property_table([row], values, 0.8, strict=True, kind=kind)
        _read_property_table([row], before, 0.8, kind=kind)

        assert ("c" in values.index) is read
        # the brief's floor as before reads every one
        assert ("c" in before.index) is (_field_of(row[0]) == "c")

    def test_a_column_headed_c_is_held_to_the_same_rule(self):
        from report_ingest.lab_floor import _Link, _read_header_table
        from report_ingest.model import Provenance
        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Boring", "Depth (m)", "LL", "C", "Bath temp (°C)"],
                ["QB-1", "1.0", "41", "Ct", "21"],
                ["QB-2", "2.5", "38", "7", "22"]]
        _n, summary = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                         prov, strict=True)
        assert [(r.ll, r.c) for r in summary] == [(41.0, None), (38.0, None)]
        # RI-71 review: their cells stay in the row's other, as printed
        assert [[tuple(o) for o in r.other] for r in summary] == [
            [("C", "Ct"), ("Bath temp (°C)", "21")],
            [("C", "7"), ("Bath temp (°C)", "22")]]
        rows[0][3], rows[1][3] = "C (kPa)", "5"
        _n, summary = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                         prov, strict=True)
        assert [r.c.value for r in summary] == [5.0, 7.0]
        assert [[tuple(o) for o in r.other] for r in summary] == [
            [("Bath temp (°C)", "21")], [("Bath temp (°C)", "22")]]

    def test_a_table_whose_only_value_is_such_a_column_seeds_nothing(self):
        from report_ingest.lab_floor import _Link, _read_header_table
        from report_ingest.model import Provenance
        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Boring", "Depth (m)", "C"],
                ["QB-1", "1.0", "5"], ["QB-2", "2.5", "7"]]
        assert _read_header_table(rows, {}, _Link(page=0), "", 0.8, prov,
                                  strict=True) == (0, [])
        _n, before = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                        prov)
        assert [r.c.value for r in before] == [5.0, 7.0]

    def test_a_grading_sheet_s_legend_makes_no_shear_test(
            self, grading_legend):
        floor = floor_from_tables(grading_legend, [0], "RXX")

        assert floor.kind == "gradation"
        assert [t.kind for t in floor.tests] == ["gradation"]
        assert "direct_shear" in [t.kind for t in floor.as_before.tests]


class TestNoValueThatIsNotTheSpecimens:
    """RI-56 review, round 2: the per-page floor reads no water content
    from a mass of water, and no specimen's value from a row of trials; and
    a page or a corrected kind with no value makes no empty test."""

    def test_a_mass_of_water_or_a_row_of_trials_is_no_water_content(
            self, limits_trials):
        floor = floor_from_tables(limits_trials, [0], "RXX")

        assert floor.kind == ""
        assert floor.tests
        assert all(getattr(t.result, "wc", None) is None
                   and getattr(t.result, "water_content", None) is None
                   for t in floor.tests)
        assert [t.result.ll for t in floor.tests
                if t.kind == "atterberg"] == [41.0]

    def test_the_rows_one_at_a_time(self):
        from report_ingest.lab_floor import _Values, _read_property_table
        rows = [["Moisture lost, mass (g)", "12.7"],
                ["Moisture content, trials (%)", "38.2", "41.5", "44.0"],
                ["Water content, natural (%)", "23.6"]]
        values = _Values()
        _read_property_table(rows, values, 0.8, strict=True)

        assert values.index["wc"][0] == 23.6

    @pytest.mark.parametrize("label", [
        "Moisture content (% by weight)", "Water content, % by wt.",
        "Moisture content, % dry weight", "Water content (% of dry mass)"])
    def test_a_percent_of_a_weight_is_a_water_content(self, label):
        from report_ingest.lab_floor import _Values, _read_property_table
        values = _Values()
        _read_property_table([[label, "17.4"]], values, 0.8, strict=True)

        assert values.index["wc"][0] == 17.4

    @pytest.mark.parametrize("label", [
        "Moisture (gm)", "Moisture, g", "Moisture (grams)",
        "Moisture lost, gm", "Moisture (Mw)", "Moisture (lb)",
        # RI-71: grams as "gr"; a unit of mass beside a percent; a mass
        # word in the floor's other languages
        "Moisture (gr)", "Humedad (gr.)", "Moisture lost (g) %",
        "Moisture, % [g]", "Moisture in g", "Peso de humedad",
        "Moisture content, Mw"])
    def test_a_mass_in_any_spelling_is_no_water_content(self, label):
        from report_ingest.lab_floor import _Values, _read_property_table
        values = _Values()
        _read_property_table([[label, "9.65"]], values, 0.8, strict=True)

        assert "wc" not in values.index

    @pytest.mark.parametrize("label, printed", [
        ("Water content, specimen G", "17.4"),
        ("Water content G-14", "17.4"),
        ("Moisture content MW-14", "17.4"),
        ("Moisture content, G-14", "17.4"),
        ("Water content (g/100 g)", "17.4"), ("Water content (g/g)", "0.174"),
        ("Water content (kg/kg)", "0.174"), ("Moisture (wt/wt)", "0.174"),
        ("Water content (g/kg)", "174"),
        ("Water content, percentage of dry weight", "17.4"),
        # RI-71 review: a capital G alone is no gram
        ("Water content, G", "17.4"), ("Water content (G)", "17.4")])
    def test_a_name_or_a_ratio_hides_no_water_content(self, label, printed):
        # RI-71: a specimen or a well named G-14 or MW-14 is no mass, nor is a
        # ratio of masses: a water content printed as a fraction is read as
        # its percent. The brief's floor as before reads what is printed.
        from report_ingest.lab_floor import _Values, _read_property_table
        values, before = _Values(), _Values()
        _read_property_table([[label, printed]], values, 0.8, strict=True)
        _read_property_table([[label, printed]], before, 0.8)

        assert values.index["wc"][0] == pytest.approx(17.4)
        assert before.index["wc"][0] == float(printed)

    @pytest.mark.parametrize("label", [
        "Moisture content (% wt/wt)", "Water content (%, g/g)",
        "Moisture, wt/wt %", "Water content, g/g x 100",
        "Water content (g/g)×100"])
    def test_a_ratio_printed_as_a_percent_is_read_as_printed(self, label):
        # RI-71 review: w = Ww/Ws x100 is "expressed as a percentage"
        # (FHWA-NHI-06-088 Eq. 2-4); a label that prints the percent or the
        # x 100 beside the ratio gives the percent already.
        from report_ingest.lab_floor import (
            _Link, _read_header_table, _read_property_table, _Values,
        )
        from report_ingest.model import Provenance
        values = _Values()
        _read_property_table([[label, "14.8"]], values, 0.8, strict=True)
        assert values.index["wc"][0] == 14.8

        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Boring", "Depth (m)", label],
                ["QB-1", "1.0", "14.8"], ["QB-2", "2.5", "23.5"]]
        _n, summary = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                         prov, strict=True)
        assert [r.wc for r in summary] == [14.8, 23.5]

    def test_a_fraction_column_is_read_as_its_percent(self):
        from report_ingest.lab_floor import _Link, _read_header_table
        from report_ingest.model import Provenance
        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Boring", "Depth (m)", "Water content (g/g)"],
                ["QB-1", "1.0", "0.264"], ["QB-2", "2.5", "0.159"]]
        _n, summary = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                         prov, strict=True)
        assert [r.wc for r in summary] == pytest.approx([26.4, 15.9])
        _n, before = _read_header_table(rows, {}, _Link(page=0), "", 0.8,
                                        prov)
        assert [r.wc for r in before] == [0.264, 0.159]

        one = {}
        _read_header_table([["Water content (kg/kg)"], ["0.264"]], one,
                           _Link(page=0), "", 0.8, prov, key="k",
                           strict=True)
        assert one["k"].index["wc"][:2] == (pytest.approx(26.4), "%")

    def test_a_run_of_bare_pages_with_no_value_makes_no_test(
            self, bare_empty_run):
        floor = floor_from_tables(bare_empty_run, [0, 1], "RXX")

        assert floor.kind == "atterberg"
        assert floor.tests == []
        # one sheet with no value still gives its kind alone
        assert [t.kind for t in floor_from_tables(
            bare_empty_run, [0], "RXX").tests] == ["atterberg"]

    def test_each_bare_page_of_a_run_is_its_own_specimen(self, bare_run):
        floor = floor_from_tables(bare_run, [0, 1, 2], "RXX")

        assert [(t.kind, t.result.wc, t.prov[1].page)
                for t in floor.tests] == [
            ("moisture_content", 23.6, 0), ("moisture_content", 19.2, 1)]
        # the brief's floor is as before: one specimen for the run
        assert len(floor.as_before.tests) == 1

    def test_a_corrected_kind_with_no_value_makes_no_test(
            self, particle_density_page):
        floor = floor_from_tables(particle_density_page, [0], "RXX")

        assert floor.kind == "specific_gravity"
        assert floor.tests == []


class TestAFloorValueThatPinsNoSpecimen:
    """RI-56 review: a floor test whose page prints no boring and no depth,
    or whose link matches more than one of the model's tests of its kind,
    may fill an empty slot but never wins a contradiction."""

    def test_a_bare_page_the_model_missed_overrules_nothing(
            self, water_run):
        # The model read the two linked pages and missed the bare page's
        # specimen, whose 25.0 % the floor would put on B-8.
        floor = floor_from_tables(water_run, [0, 1, 2], "RXX")
        model = [
            _moisture(0, "model", 21.4, 101.2, investigation_id="B-7"),
            _moisture(1, "model", 17.9, None, investigation_id="B-8"),
        ]
        model[0].depth_top = floor.tests[0].depth_top
        model[1].depth_top = floor.tests[1].depth_top
        merged, log = merge_lab_tests(floor.tests, model, MergeLog(),
                                      title_kind=floor.kind)

        b8 = next(t for t in merged if t.investigation_id == "B-8")
        assert b8.result.wc == 17.9
        assert b8.result.dry_density.value == 106.5     # B-8's own page
        assert not [d for d in log.disagreements if d["kept"] == "floor"]
        (row,) = [d for d in log.disagreements if d["floor"] == "25"]
        assert row["kept"] == "model" and "do not pin" in row["why"]
        assert any(a.value == "25" and "did not overrule" in a.note
                   for a in b8.prov[0].alternatives)

    def test_a_depth_matching_two_specimens_overrules_neither(self):
        floor = [_moisture(10, "tables", 33.7, depth=3.25)]
        model = [
            _moisture(10, "model", 8.6, investigation_id="TP-6", depth=3.25),
            _moisture(10, "model", 6.9, investigation_id="TP-7", depth=3.25),
        ]
        merged, log = merge_lab_tests(floor, model, MergeLog())

        assert [t.result.wc for t in merged] == [8.6, 6.9]
        (row,) = log.disagreements
        assert row["kept"] == "model" and row["floor"] == "33.7"

    def test_it_still_fills_an_empty_slot(self):
        floor = [_moisture(10, "tables", 33.7, 94.1, depth=3.25)]
        model = [
            _moisture(10, "model", 33.7, investigation_id="TP-6", depth=3.25),
            _moisture(10, "model", 6.9, investigation_id="TP-7", depth=3.25),
        ]
        merged, _log = merge_lab_tests(floor, model, MergeLog())

        assert merged[0].result.dry_density.value == 94.1

    def test_a_link_that_pins_one_specimen_still_wins(self):
        floor = [_moisture(10, "tables", 33.7, investigation_id="TP-6",
                           depth=3.25)]
        model = [_moisture(10, "model", 8.6, investigation_id="TP-6",
                           depth=3.25)]
        merged, log = merge_lab_tests(floor, model, MergeLog())

        assert merged[0].result.wc == 33.7
        assert log.disagreements[0]["kept"] == "floor"


class TestAValueNamedFloorTestJoinsTheModelsTest:
    """RI-56 review: a floor test named by its values, on a specimen the
    model reported as another kind, goes into that test's fields."""

    def _limits(self):
        from report_ingest.model import (
            AtterbergResult, LabTest, Provenance, Quantity,
        )
        return LabTest(
            kind="atterberg", investigation_id="TP-8", sample_id="S-6",
            depth_top=Quantity(value=3.75, unit="m"), pages=[10],
            prov=[Provenance(page=10, method="model", confidence=0.85)],
            result=AtterbergResult(ll=43.0, pl=21.0, pi=22.0))

    def test_a_water_content_beside_the_limits_is_the_limits_test_s(self):
        # Its slot in the Atterberg test where it has one; the density, which
        # has none there, in the test's fields.
        floor = [_moisture(10, "tables", 18.3, 95.6, sample_id="S-6",
                           depth=3.75)]
        merged, log = merge_lab_tests(floor, [self._limits()], MergeLog(),
                                      title_kind="")

        assert [t.kind for t in merged] == ["atterberg"]
        assert merged[0].result.water_content == 18.3
        assert merged[0].fields == {"dry_density": "95.6 pcf"}
        assert {k["what"] for k in log.kept} == {
            "moisture_content: water_content",
            "moisture_content: dry_density"}

    def test_a_titled_water_content_test_stays_its_own(self):
        floor = [_moisture(10, "tables", 18.3, sample_id="S-6", depth=3.75)]
        merged, _log = merge_lab_tests(floor, [self._limits()], MergeLog(),
                                       title_kind="moisture_content")

        assert [t.kind for t in merged] == ["atterberg", "moisture_content"]

    def test_a_compaction_sheet_s_water_content_is_its_compaction_s(
            self, density_run):
        floor = floor_from_tables(density_run, [0, 1], "RXX")
        assert floor.kind == "compaction"
        assert [t.kind for t in floor.tests] == ["compaction", "compaction"]
        assert floor.tests[1].fields == {"wet_density": "2.11 Mg/m3",
                                         "wc": "12.6"}
        # the brief's floor is as before: a density test, values and all
        assert [t.kind for t in floor.as_before.tests] == ["density"]

    def test_a_quoted_compaction_holds_none_of_a_field_test_s_values(
            self, field_density):
        # RI-71: a field density sheet quoting its compaction: the dry
        # density and water content are the field test's, not copied into
        # the quoted compaction's fields as well.
        floor = floor_from_tables(field_density, [0], "RXX")
        assert floor.kind == "density"
        assert [t.kind for t in floor.tests] == ["density", "compaction"]
        density, compaction = floor.tests
        assert (density.result.wc, density.result.dry_density.value) == (
            14.62, 1.794)
        assert compaction.result.max_dry_density.value == 1.951
        assert compaction.fields == {}


class TestTheFloorReadsTheSpecimensOwnValues:
    """RI-71: the per-page floor reads a value printed across a row once,
    and a value beside another label; no trial point as the specimen's; and
    no header text as a chemical result. The brief's floor as before reads
    as it did."""

    @pytest.mark.parametrize("row, read", [
        (["Water content (%)", "16.45", "Specific gravity", "2.71"], 16.45),
        (["Average water content (%)", "13.65", "13.65", "13.65"], 13.65),
        (["Water content, trials (%)", "31.64", "33.87", "36.21"], None),
        (["Water content (%)", "16.45", "Average", "17.45"], None),
        # a link word between them names no other value: two specimens
        (["Water content (%)", "16.45", "Specimen K", "17.45"], None),
        (["Water content (%)", "16.45", "Ring No. 4", "17.45"], None),
        # a CBR's swell and CBR are one a mould
        (["Swell, each mould (%)", "0.264", "0.715", "1.193"], None),
        (["Bearing ratio, each mould (%)", "4", "9", "15"], None)])
    def test_a_row_s_numbers(self, row, read):
        from report_ingest.lab_floor import (
            _field_of, _Values, _read_property_table,
        )
        values = _Values()
        _read_property_table([row], values, 0.8, strict=True)

        assert values.index.get(_field_of(row[0]), (None,))[0] == read

    def test_a_trial_table_s_first_row_is_no_specimen_s_value(
            self, compaction_trials):
        floor = floor_from_tables(compaction_trials, [0], "RXX")

        assert floor.kind == "compaction"
        (test,) = floor.tests
        assert (test.result.max_dry_density.value,
                test.result.optimum_wc) == (1.871, 11.46)
        assert test.fields == {}
        # the brief's floor as before reads the first point
        assert [(t.kind, t.result.wc, t.result.dry_density.value)
                for t in floor.as_before.tests
                if t.kind == "density"] == [("density", 8.72, 1.793)]

    def test_a_cbr_trial_table_seeds_no_value(self):
        from report_ingest.lab_floor import _Link, _read_header_table
        from report_ingest.model import Provenance
        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Blows per lift", "Dry density (Mg/m3)",
                 "Water content (%)", "Bearing ratio, each mould (%)"],
                ["11", "1.66", "12.3", "5"], ["26", "1.74", "12.1", "11"],
                ["57", "1.82", "11.8", "19"]]
        strict, before = {}, {}
        assert _read_header_table(rows, strict, _Link(page=0), "cbr", 0.8,
                                  prov, key="k", strict=True) == (0, [])
        _read_header_table(rows, before, _Link(page=0), "cbr", 0.8, prov,
                           key="k")
        assert before["k"].index["cbr_percent"][0] == 5.0
        # a one-row table of the specimen's values is still read
        one = {}
        _read_header_table([["LL", "PL", "PI"], ["57.4", "31.9", "25.5"]], one,
                           _Link(page=0), "atterberg", 0.8, prov, key="k",
                           strict=True)
        assert {k: v[0] for k, v in one["k"].index.items()} == {
            "ll": 57.4, "pl": 31.9, "pi": 25.5}

    def test_a_unit_column_is_no_chemical_result(self):
        from report_ingest.lab_floor import _Values, _read_property_table
        rows = [["Determination", "Units", "Result"],
                ["Resistivity", "ohm-cm", "8,300"], ["pH", "-", "8.35"],
                ["Redox", "mV", "-46"], ["Sulphide", "pos/neg", "Trace"]]
        values, before = _Values(), _Values()
        _read_property_table(rows, values, 0.8, strict=True)
        _read_property_table(rows, before, 0.8)

        assert values.index == {}
        assert before.index["sulfides"][0] == "pos/neg"
        # a result printed as words is still read
        words = _Values()
        _read_property_table([["Sulfate", "None detected"],
                              ["Sulphide", "Trace"],
                              ["Chloride", "<20 mg/kg"]], words, 0.8,
                             strict=True)
        assert {k: v[0] for k, v in words.index.items()} == {
            "sulfate": "None detected", "sulfides": "Trace",
            "chloride": "<20 mg/kg"}
        # a mark for no result is none
        marks = [["pH", "N/A"], ["Redox", "NT"], ["Sulfate", "n/a"]]
        none, before = _Values(), _Values()
        _read_property_table(marks, none, 0.8, strict=True)
        _read_property_table(marks, before, 0.8)
        assert none.index == {}
        assert before.index["pH"][0] == "N/A"

    def test_the_next_row_s_label_is_no_chemical_result(self):
        from report_ingest.lab_floor import (
            _Link, _read_header_table, _read_property_table, _Values,
        )
        from report_ingest.model import Provenance
        prov = Provenance(page=0, method="tables", confidence=0.8)
        rows = [["Resistivity", "2.9E+03", "ohm-m"],
                ["Chloride", "6", ""], ["Redox", "133", ""]]
        got = {}
        assert _read_header_table(rows, got, _Link(page=0), "chemical", 0.8,
                                  prov, key="k", strict=True) == (0, [])
        before = {}
        _read_header_table(rows, before, _Link(page=0), "chemical", 0.8,
                           prov, key="k")
        assert before["k"].index["resistivity"][0] == "Chloride"
        # read as the property table it is, the rows give their own values
        values = _Values()
        _read_property_table(rows, values, 0.8, strict=True)
        assert values.index["chloride"][0] == 6.0


# ---------------------------------------------------------------------------
# the builder's check mode (RI-87 piece 6, plan section 8.6)
# ---------------------------------------------------------------------------

def _ranked(rows):
    """The QA entries the reconciler makes of ``rows``, with their ranks."""
    from report_ingest.reconciler import _carry_unresolved
    qa = []
    _carry_unresolved(qa, rows)
    return qa


def _lab_flag(reading=None, pages=None):
    from report_ingest.lab_reader import _Builder
    from report_ingest.tests.checks_fixtures import LAB_PAGES, lab_reading
    builder = _Builder(reading or lab_reading(), pages or LAB_PAGES,
                       "R-made-up", [], checks="flag")
    return builder, builder.build()


def _rows_about(builder, what):
    return [u for u in builder.unresolved if u["what"].endswith(what)]


class TestLabChecksDefaultIsMaster:

    def test_every_lab_case_builds_as_on_master(self):
        from report_ingest.tests.checks_fixtures import (
            build_default, lab_cases, load_snapshot,
        )
        now, then = build_default(), load_snapshot()
        for case in lab_cases():
            assert now[f"lab/{case}"] == then[f"lab/{case}"], case

    def test_refuse_named_is_the_default(self):
        from report_ingest.lab_reader import _Builder
        from report_ingest.tests.checks_fixtures import LAB_PAGES, lab_reading
        a = _Builder(lab_reading(), LAB_PAGES, "R", [])
        b = _Builder(lab_reading(), LAB_PAGES, "R", [], checks="refuse")
        assert a.build() == b.build()
        assert a.unresolved == b.unresolved and a.changes == b.changes
        # every test carries the whole sheet's pages, as before
        assert {tuple(t.pages) for t in _Builder(
            lab_reading(), LAB_PAGES, "R", []).build()} == {(20, 21, 22)}

    def test_refuse_keeps_water_contents_and_limits_past_100(self):
        """RI-89: on the default path too, a water content or a limit is
        refused only below 0, and a fraction past 100 still is."""
        from report_ingest.lab_reader import _Builder
        from report_ingest.tests.checks_fixtures import (
            LAB_PAGES, lab_positions_reading, lab_reading,
        )
        builder = _Builder(lab_positions_reading(), LAB_PAGES, "R", [])
        tests = builder.build()
        assert tests[0].result.water_content == 150.0
        assert (tests[0].result.ll, tests[0].result.pl,
                tests[0].result.pi) == (120.0, 45.0, 75.0)
        assert tests[1].result.wc is None
        (row,) = [u for u in builder.unresolved if u.get("value") == -3.0]
        assert row["why"] == ("-3 is not a percentage; refused rather than "
                              "accepted")
        builder = _Builder(lab_reading(), LAB_PAGES, "R", [])
        tests = builder.build()
        assert tests[1].result.gravel_percent is None
        assert any(u["why"] == "105 is not a percentage; refused rather "
                               "than accepted" for u in builder.unresolved)


class TestLabChecksFlag:

    def test_a_liquid_limit_below_the_plastic_limit_is_kept_out(self):
        builder, tests = _lab_flag()
        atterberg = tests[0].result
        assert (atterberg.ll, atterberg.pl, atterberg.pi) == (None, None, None)
        (row,) = _rows_about(builder, "atterberg 1 (B-1): Atterberg limits")
        assert row["value"] == [25.0, 30.0, None]
        assert row["page"] == 21
        assert row["why"] == (
            "liquid limit 25 is below plastic limit 30, which no soil does; "
            "which of the three is wrong cannot be told, so all three go; "
            "no real sheet prints this, so it is kept out of the record and "
            "listed here as read")
        (entry,) = _ranked([row])
        assert entry.kind == "out_of_range" and entry.rank == "open"
        # the summary row's triple too, with its printed plastic limit
        (row,) = [u for u in builder.unresolved
                  if u["what"] == "summary row 2 (B-3): Atterberg limits"]
        assert row["value"] == [18.0, 26.0, 8.0]
        summary = tests[3].result.rows[1]
        assert (summary.ll, summary.pl, summary.pi) == (None, None, None)

    def test_a_rising_grading_is_kept_out_whole_with_the_series(self):
        import json
        builder, tests = _lab_flag()
        assert tests[1].result.percent_passing == []
        (row,) = _rows_about(builder, "grading series")
        assert row["value"] == [80.0, 92.0]
        assert row["page"] == 22
        assert "the series runs the wrong way, so the whole series goes; " \
               "no real sheet prints this" in row["why"]
        read = json.loads(row["read"])
        assert [(p["x"], p["y"]) for p in read["points"]] == [
            (19.0, 100.0), (4.75, 80.0), (0.075, 92.0)]
        (entry,) = _ranked([row])
        assert entry.rank == "open" and entry.values[-1] == row["read"]

    def test_percentages_and_depths_that_cannot_be_are_kept_out(self):
        builder, tests = _lab_flag()
        assert tests[1].result.gravel_percent is None
        assert tests[1].result.sand_percent == 60.0
        assert tests[1].depth_top is None
        assert tests[4].depth_top is None
        assert tests[3].result.rows[1].depth_top is None
        rows = {u["what"]: u for u in builder.unresolved}
        assert rows["gradation 2 (B-1): gravel"]["value"] == 105.0
        assert rows["gradation 2 (B-1): depth"]["value"] == 1500.0
        assert rows["summary row 2 (B-3): row depth"]["value"] == -4.0
        assert rows["moisture_content 5 (B-4): depth"]["value"] == 2500.0
        assert rows[
            "compaction 3 (B-2): compaction point water content "
            "(dry density 117 pcf)"]["value"] == -8.0
        impossible = [u for u in builder.unresolved
                      if u.get("refused_by") == "python"
                      and "provenance" not in u["what"]]
        assert impossible and all("listed here as read" in u["why"]
                                  for u in impossible)
        assert {e.rank for e in _ranked(impossible)} == {"open"}

    def test_an_entry_says_kept_out_once(self):
        """The refuse words ("refused rather than accepted", "the series
        was refused") are not repeated before the flag words."""
        builder, tests = _lab_flag()
        flagged = [u for u in builder.unresolved
                   if "listed here as read" in u["why"]]
        assert flagged
        assert not any("refused" in u["why"] for u in flagged)
        assert (_rows_about(builder, "gradation 2 (B-1): gravel")[0]["why"]
                == "105 is not a percentage; no real sheet prints this, so "
                   "it is kept out of the record and listed here as read")

    def test_an_entry_names_its_tests_page(self):
        builder, tests = _lab_flag()
        rows = {u["what"]: u for u in builder.unresolved}
        assert rows["gradation 2 (B-1): gravel"]["page"] == 22
        assert rows["gradation 2 (B-1): depth"]["page"] == 22
        assert rows["summary row 1 (B-1): percent passing No. 4"][
            "page"] == 20
        assert rows["moisture_content 5 (B-4): depth"]["page"] == 22
        # a test citing a page outside the sheet names none
        assert "page" not in rows[
            "compaction 3 (B-2): compaction point water content "
            "(dry density 117 pcf)"]

    def test_water_contents_and_limits_past_100_are_kept(self):
        """A water content can be greater than 100 % (FHWA-NHI-06-088 Vol. I
        p. 2-5); liquid limits run to 143 (DM 7.1 section 8-2.2.2). Only a
        negative one cannot be."""
        from report_ingest.tests.checks_fixtures import lab_positions_reading
        builder, tests = _lab_flag(lab_positions_reading())
        atterberg = tests[0].result
        assert (atterberg.ll, atterberg.pl, atterberg.pi) == (120.0, 45.0,
                                                              75.0)
        assert atterberg.water_content == 150.0
        assert tests[1].result.wc is None
        (row,) = [u for u in builder.unresolved if u.get("value") == -3.0]
        assert row["why"].startswith(
            "-3 is negative, which no water content or limit is; no real "
            "sheet prints this")
        assert row["page"] == 22
        assert _ranked([row])[0].rank == "open"

    def test_a_depth_with_no_unit_is_bounded_as_feet(self):
        from report_ingest.tests.checks_fixtures import lab_no_unit_reading
        builder, tests = _lab_flag(lab_no_unit_reading())
        assert tests[0].depth_top is None
        assert tests[0].depth_bottom.value == 4.0
        (row,) = builder.unresolved
        assert "even read as feet" in row["why"]
        assert row["page"] == 21
        assert _ranked([row])[0].rank == "open"

    def test_an_answer_citing_positions_is_read_as_positions(self):
        from report_ingest.tests.checks_fixtures import lab_positions_reading
        builder, tests = _lab_flag(lab_positions_reading())
        assert [t.prov[0].page for t in tests] == [20, 22]
        assert [t.pages for t in tests] == [[20], [22]]
        assert "a position in the 3 page(s)" in tests[0].prov[0].note
        assert [c["page"] for c in builder.changes] == [20, 22]
        (row,) = [u for u in builder.unresolved if u["what"] == "a stamp"]
        assert row["page"] == 21
        assert not any(u["what"].endswith("provenance")
                       for u in builder.unresolved)

    def test_a_mixed_answer_is_read_as_the_reports_pages(self):
        """The sheet's answer cites 20, 21, 22 and 57: the report's pages,
        so 57 is outside the sheet, as today, and nothing is moved."""
        builder, tests = _lab_flag()
        assert builder.changes == []
        assert [t.pages for t in tests] == [
            [21], [22], [20, 21, 22], [20], [22]]
        # cited page 57 is no page of the sheet: the whole window, as today,
        # on the first page at confidence 0.4
        assert tests[2].prov[0].page == 20
        assert tests[2].prov[0].confidence == 0.4

    def test_a_mixed_answer_gets_one_note(self):
        """RI-89 review: 20 to 22 are pages of the sheet and 57 is not; the
        answer is read as the report's pages, and says so once."""
        builder, tests = _lab_flag()
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "page_numbers"]
        assert note["why"] == (
            "the page numbers this answer cites mix this call's pages and "
            "others: 20, 21, 22 are among this call's pages and 57 is not; "
            "they were read as the report's pages")

    def test_a_series_the_kind_does_not_read_is_listed(self):
        """RI-89 review: an Atterberg test reads only a flow curve; a
        grading series given with it is listed as read under flag, and
        left out without a word under refuse, as today."""
        import json
        from report_ingest.lab_reader import _Builder
        reading = LabSheetReading.model_validate({
            "depth_unit": "ft",
            "tests": [{"kind": "atterberg", "investigation_id": "B-1",
                       "ll": 40.0, "pl": 20.0, "pi": 20.0,
                       "series": [
                           {"name": "flow_curve",
                            "points": [{"x": 25.0, "y": 41.0}]},
                           {"name": "percent_passing",
                            "points": [{"x": 4.75, "y": 90.0}]}],
                       "prov": {"page": 20}}]})
        builder = _Builder(reading, [20], "R", [], checks="flag")
        (test,) = builder.build()
        assert test.result.flow_curve == [(25, 41.0)]
        (row,) = builder.unresolved
        assert row["what"] == "atterberg 1 (B-1): series percent_passing"
        assert row["page"] == 20
        assert row["why"] == (
            "the atterberg test reads no series named 'percent_passing' (it "
            "reads flow_curve); kept out of the record and listed here as "
            "read")
        assert json.loads(row["read"])["points"] == [{"x": 4.75, "y": 90.0}]
        assert _ranked([row])[0].rank == "open"
        today = _Builder(reading, [20], "R", [])
        today.build()
        assert today.unresolved == []

    def test_a_second_series_of_a_name_already_read_is_listed(self):
        """A gradation holds one percent-passing curve: the first given is
        read, and a second of that name is listed as read under flag, and
        left out without a word under refuse, as today."""
        import json
        from report_ingest.lab_reader import _Builder
        reading = LabSheetReading.model_validate({
            "depth_unit": "ft",
            "tests": [{"kind": "gradation", "investigation_id": "B-1",
                       "series": [
                           {"name": "percent_passing", "x_unit": "mm",
                            "points": [{"x": 4.75, "y": 90.0},
                                       {"x": 0.075, "y": 30.0}]},
                           {"name": "Percent_Passing", "x_unit": "mm",
                            "points": [{"x": 4.75, "y": 70.0},
                                       {"x": 0.075, "y": 12.0}]}],
                       "prov": {"page": 20}}]})
        builder = _Builder(reading, [20], "R", [], checks="flag")
        (test,) = builder.build()
        assert [p.percent_passing for p in test.result.percent_passing] == [
            90.0, 30.0]
        (row,) = builder.unresolved
        assert row["what"] == "gradation 1 (B-1): series percent_passing"
        assert row["page"] == 20
        assert row["why"] == (
            "the gradation test keeps one series named 'percent_passing', "
            "the first one given; this is a second one; kept out of the "
            "record and listed here as read")
        assert [p["y"] for p in json.loads(row["read"])["points"]] == [
            70.0, 12.0]
        assert _ranked([row])[0].rank == "open"
        today = _Builder(reading, [20], "R", [])
        (same,) = today.build()
        assert today.unresolved == []
        assert same.result == test.result

    def test_a_summary_tables_own_entries_carry_its_label(self):
        """RI-89 review: the summary builder left its test's label empty,
        so the table's provenance and picture entries had no 'what'. A
        default-path fix too."""
        from report_ingest.lab_reader import _Builder
        reading = LabSheetReading.model_validate({
            "depth_unit": "ft",
            "tests": [{"kind": "summary_table", "investigation_id": "B-1",
                       "rows": [{"investigation_id": "B-1",
                                 "depth_top": 2.0, "wc": 12.0}],
                       "prov": {"page": 57, "from_image": True}}]})
        for checks in ("refuse", "flag"):
            builder = _Builder(reading, [20, 21], "R", [], checks=checks)
            builder.build()
            whats = [u["what"] for u in builder.unresolved]
            assert "summary_table 1 (B-1): provenance" in whats, checks
        positions = LabSheetReading.model_validate({
            "depth_unit": "ft",
            "tests": [{"kind": "summary_table", "investigation_id": "B-1",
                       "rows": [{"investigation_id": "B-1",
                                 "depth_top": 2.0}],
                       "prov": {"page": 1}}]})
        builder = _Builder(positions, [20, 21], "R", [], checks="flag")
        builder.build()
        (change,) = builder.changes
        assert change["what"] == "summary_table 1 (B-1)"
        assert change["page"] == 21
