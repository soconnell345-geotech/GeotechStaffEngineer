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

from report_ingest.lab_floor import floor_from_tables, kind_from_title
from report_ingest.lab_reader import (
    KIND_DEFINITIONS, LAB_READER_SYSTEM, LAB_TOOLS, LINK_FROM_LABEL,
    LINK_FROM_LABEL_ENV, MAX_LAB_PAGES, LabSheetReading, MAX_MODEL_CALLS,
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

    def test_a_percentage_past_a_hundred_is_refused(self, atterberg):
        doc, _gt = atterberg
        engine = FakeEngine([{"final": _atterberg_reading(
            test={"ll": 148.0, "pl": 22.0, "pi": 26.0})}])
        result = read_lab_sheet(doc, [0], engine)
        assert result.model_tests[0].result.ll is None
        assert result.model_tests[0].result.pl == 22.0
        assert any("not a percentage" in u["why"] for u in result.unresolved)
        # The table prints 48; the record carries it, from the floor.
        assert result.tests[0].result.ll == 48.0

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
