"""The calculation reader, offline: the floor, the merge and the gates.

Every test runs the real reader over a real synthetic printout with a fake
engine replaying a canned structured reply, so the brief that is asserted on
is the brief a model would actually be sent and the refusals are the ones
that would actually happen.

The pages come from ``report_ingest.tests.calc_fixtures``: a settlement
spreadsheet whose answer is the last filled row of a table, a program
printout with a banner and fixed-width labelled lines, and a slope section
whose factor of safety is printed on the drawing. Nothing in them is copied
from a real printout.
"""

from __future__ import annotations

import pytest

from report_ingest.calc_reader import (
    CALC_KIND_DEFINITIONS, CALC_READER_SYSTEM, CALC_TOOLS, CalcReading,
    MAX_CALC_CALLS, MAX_CALC_PAGES, ReadProv, ReadValue, Unsettled,
    _pair_in_line, _program_in, floor_from_pages, merge_calculation,
    page_window, printed_numbers, read_calculation, serialise_floor,
    serialise_page,
)
from report_ingest.model import (
    CALC_KINDS, Calculation, NamedQuantity, Provenance, Quantity,
)
from report_ingest.tests.fake_engine import FakeEngine, ScriptExhausted

fitz = pytest.importorskip("fitz", reason="the fixtures are drawn with PyMuPDF")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _open(pdf: bytes, tmp_path):
    from planlens.document import open_document
    path = tmp_path / "calc.pdf"
    path.write_bytes(pdf)
    return open_document(str(path))


@pytest.fixture
def spreadsheet(tmp_path):
    from report_ingest.tests.calc_fixtures import build_settlement_spreadsheet
    gt = build_settlement_spreadsheet()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def printout(tmp_path):
    from report_ingest.tests.calc_fixtures import build_program_printout
    gt = build_program_printout()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


@pytest.fixture
def plotted(tmp_path):
    from report_ingest.tests.calc_fixtures import build_plotted_result
    gt = build_plotted_result()
    doc = _open(gt.pdf, tmp_path)
    yield doc, gt
    doc.close()


def _prov(page: int = 0, bbox=None, note: str = "", from_image: bool = False):
    return ReadProv(page=page, bbox=bbox, note=note, from_image=from_image)


def _reading(**over) -> CalcReading:
    data = dict(kind="settlement", program="", program_version="",
                method="Schmertmann", subject="Column footing F-3",
                inputs=[], results=[], summary="A settlement calculation.",
                pages_read=[0])
    data.update(over)
    return CalcReading(**data)


def _value(name, value=None, unit="", text="", page=0, bbox=None, note="",
           from_image=False) -> ReadValue:
    return ReadValue(name=name, value=value, unit=unit, text=text,
                     prov=_prov(page, bbox, note, from_image))


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

class TestTheFloor:
    """What a pattern can read off a printout before a call is spent."""

    def test_a_label_and_its_value_on_one_printed_line_are_a_pair(
            self, spreadsheet):
        doc, gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")

        by_name = {v.name: v for v in floor.values}
        assert "Footing Bearing Pressure (ksf)" in by_name
        found = by_name["Footing Bearing Pressure (ksf)"]
        assert found.value == 4.0 and found.unit == "ksf"
        assert found.page == 0 and found.bbox is not None

    def test_the_unit_comes_off_the_column_heading(self, spreadsheet):
        """A spreadsheet puts the unit in the LABEL and the number in the
        cell beside it, so a floor that only looked at the cell would carry
        a width with no unit."""
        doc, _gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")

        width = next(v for v in floor.values
                     if v.name == "Footing Width B (ft)")
        assert (width.value, width.unit) == (8.5, "ft")

    def test_the_last_filled_row_of_a_table_is_read(self, spreadsheet):
        """The answer of a layer-by-layer settlement is the cumulative
        column's last FILLED row, not the last ruled line."""
        doc, _gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")

        row = next(v for v in floor.values
                   if v.name.startswith("Total Cumulative Settlement"))
        assert row.value == 0.52
        assert "last row" in row.name

    def test_a_program_printouts_labelled_lines_are_pairs(self, printout):
        doc, gt = printout
        floor = floor_from_pages(doc, [0], "SYN")

        by_name = {v.name: (v.value, v.unit) for v in floor.values}
        assert by_name["Maximum bending moment"] == (612.4, "kN-m")
        assert by_name["Pile-head deflection"] == (0.0184, "m")
        assert by_name["Applied lateral load"] == (240.0, "kN")

    def test_a_row_of_echoed_data_names_nothing_and_is_left_alone(
            self, printout):
        """A printout's depth-by-depth table has no labels in it. A floor
        that read one would emit a hundred values called nothing."""
        doc, _gt = printout
        floor = floor_from_pages(doc, [0], "SYN")

        assert floor.n_values < 15
        assert all(any(ch.isalpha() for ch in v.name) for v in floor.values)

    def test_a_unit_the_record_cannot_convert_is_still_a_unit(self, printout):
        """kN-m has no conversion in the record's table. Dropping it would
        leave a bending moment in the floor as a bare number."""
        doc, _gt = printout
        floor = floor_from_pages(doc, [0], "SYN")

        moment = next(v for v in floor.values
                      if v.name == "Maximum bending moment")
        assert moment.unit == "kN-m"
        assert Quantity(value=moment.value, unit=moment.unit).to_si() is None

    def test_a_banner_naming_a_known_program_is_read_and_one_that_does_not_is_not(
            self):
        assert _program_in("LPILE Plus Version 6.0.9") == ("LPILE", "6.0.9")
        assert _program_in("* * STABL6H * *") == ("STABL6H", "")
        assert _program_in("Settle3 v2023.1") == ("Settle3", "2023.1")
        # A bare number after a program name is not a version, and this
        # would rather say nothing than guess one.
        assert _program_in("Settle3 2023") == ("Settle3", "")
        assert _program_in("Spread Footing Settlement Calculation") is None

    def test_the_floor_of_a_spreadsheet_names_no_program(self, spreadsheet):
        """Which is the right answer: a spreadsheet names none, and the
        reader must not be handed an invented one as its starting record."""
        doc, _gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")

        assert floor.program is None

    def test_a_clock_time_in_a_footer_is_not_a_labelled_value(self):
        assert _pair_in_line("Thursday, June 29, 2006 11:53:13 AM") is None
        assert _pair_in_line("Date: 4/29/2024") is None
        assert _pair_in_line("Design ESALs .......... 137,774") == (
            "Design ESALs", 137774.0, "")

    def test_the_floor_becomes_a_calculation_of_kind_other(self,
                                                           spreadsheet):
        """A pattern cannot say what a printout works out, and the BEFORE
        column must not be credited with the kind."""
        doc, _gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")
        calc = floor.as_calculation("SYN")

        assert calc.kind == "other"
        assert calc.results == []
        assert len(calc.inputs) == floor.n_values

    def test_the_starting_record_is_in_the_brief(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading()}])
        read_calculation(doc, [0], engine, report_id="SYN")

        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE STARTING RECORD" in brief
        assert "Footing Bearing Pressure (ksf): 4 ksf" in brief
        assert "THE KIND VOCABULARY" in brief
        for kind in CALC_KINDS:
            assert kind in brief


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _floor_of(doc, pages=(0,)):
    return floor_from_pages(doc, list(pages), "SYN")


class TestTheMerge:
    """Add, correct with evidence, never drop -- and say where they split."""

    def test_a_value_the_reader_adds_is_kept_and_counted(self, spreadsheet):
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(
            kind="settlement",
            results=[NamedQuantity(
                name="Angular distortion",
                value=Quantity(value=0.0015, unit=""),
                prov=Provenance(page=0, method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        assert any(r.name == "Angular distortion" for r in merged.results)
        assert any(a["what"] == "Angular distortion" for a in log.added)

    def test_a_floor_value_the_reader_left_out_is_kept_not_dropped(
            self, spreadsheet):
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(kind="settlement")

        merged, log = merge_calculation(floor, model)

        names = {r.name for r in merged.inputs}
        assert "Footing Width B (ft)" in names
        assert len(log.kept) == floor.n_values

    def test_the_same_value_read_twice_is_reconciled(self, spreadsheet):
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(
            kind="settlement",
            inputs=[NamedQuantity(
                name="Footing Width B (ft)",
                value=Quantity(value=8.5, unit="ft"),
                prov=Provenance(page=0, method="model", confidence=0.9))])

        _merged, log = merge_calculation(floor, model)

        assert log.reconciled >= 1
        assert not log.disagreements

    def test_a_contradiction_without_evidence_keeps_the_pages_value(
            self, spreadsheet):
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(
            kind="settlement",
            inputs=[NamedQuantity(
                name="Footing Width B (ft)",
                value=Quantity(value=12.0, unit="ft"),
                prov=Provenance(page=0, method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        kept = next(r for r in merged.inputs
                    if r.name == "Footing Width B (ft)")
        assert kept.value.value == 8.5
        (split,) = [d for d in log.disagreements
                    if d["what"] == "Footing Width B (ft)"]
        assert split["kept"] == "floor"
        assert split["floor"] == "8.5 ft" and split["model"] == "12 ft"
        assert kept.prov.alternatives[0].value == "12 ft"

    def test_a_contradiction_with_a_box_and_a_note_replaces_it(self,
                                                               spreadsheet):
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(
            kind="settlement",
            inputs=[NamedQuantity(
                name="Footing Width B (ft)",
                value=Quantity(value=12.0, unit="ft"),
                prov=Provenance(page=0, bbox=(330.0, 120.0, 500.0, 135.0),
                                method="model", confidence=0.9,
                                note="the cell reads 12.0, not 8.5"))])

        merged, log = merge_calculation(floor, model)

        kept = next(r for r in merged.inputs
                    if r.name == "Footing Width B (ft)")
        assert kept.value.value == 12.0
        (split,) = [d for d in log.disagreements
                    if d["what"] == "Footing Width B (ft)"]
        assert split["kept"] == "model"
        assert kept.prov.alternatives[0].value == "8.5 ft"

    def test_the_floors_program_fills_a_reader_that_named_none(self, tmp_path):
        from report_ingest.tests.calc_fixtures import build_program_printout
        gt = build_program_printout()
        doc = _open(gt.pdf, tmp_path)
        try:
            floor = _floor_of(doc)
            floor.program = "LPILE 6.0.9"          # as a banner pattern would
            merged, log = merge_calculation(floor, Calculation(kind="other"))
        finally:
            doc.close()

        assert merged.program == "LPILE 6.0.9"
        assert any(k["what"] == "program" for k in log.kept)


# ---------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------

class TestTheGates:
    """What Python refuses after the model has answered."""

    def test_a_result_that_is_on_no_page_drops_to_confidence_0_3(
            self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(results=[
            _value("Total settlement", 9.99, "in")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.results
                   if r.name == "Total settlement")
        assert row.prov.confidence == 0.3
        assert "on none of these pages" in row.note
        (flag,) = [u for u in result.unresolved
                   if u["what"] == "result Total settlement"]
        assert flag["refused_by"] == "python"
        assert flag["value"] == "9.99 in"

    def test_a_result_that_is_on_the_page_keeps_its_confidence(self,
                                                               spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(results=[
            _value("Total Cumulative Settlement (inches)", 0.52, "in")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.results
                   if r.name.startswith("Total Cumulative"))
        assert row.prov.confidence == 0.9
        assert not [u for u in result.unresolved
                    if "on none of these pages" in u.get("why", "")]

    def test_the_gate_is_judged_at_the_precision_that_was_reported(
            self, spreadsheet):
        """The page prints 0.52. A reader that reports 0.521 has claimed a
        third digit the page does not carry, and the gate says so; one that
        reports 0.5 has rounded, and 0.52 is inside what 0.5 claims."""
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(results=[
            _value("Third digit", 0.521, "in")])}])
        result = read_calculation(doc, [0], engine, report_id="SYN")
        assert next(r for r in result.calculation.results
                    if r.name == "Third digit").prov.confidence == 0.3

        engine = FakeEngine([{"final": _reading(results=[
            _value("Rounded", 0.5, "in")])}])
        result = read_calculation(doc, [0], engine, report_id="SYN")
        assert next(r for r in result.calculation.results
                    if r.name == "Rounded").prov.confidence == 0.9

    def test_an_input_is_not_checked_against_the_pages(self, spreadsheet):
        """Only results are gated: an input a printout echoes from another
        sheet is a real input and is not on these pages."""
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Assumed modulus from the logs", 777.0, "tsf")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.inputs)
        assert row.prov.confidence == 0.9

    def test_a_kind_outside_the_list_becomes_other_with_a_note(self,
                                                               spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(kind="pile_driveability")}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert result.calculation.kind == "other"
        (note,) = [u for u in result.unresolved
                   if u["what"] == "calculation kind"]
        assert "not in the kind vocabulary" in note["why"]

    def test_a_kind_the_trade_spells_differently_is_understood(self,
                                                              spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(kind="bearing_capacity")}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert result.calculation.kind == "shallow_foundation_bearing"
        assert not [u for u in result.unresolved
                    if u["what"] == "calculation kind"]

    def test_a_unit_with_no_conversion_is_unsettled_and_the_value_is_kept(
            self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Subgrade reaction", 150.0, "pci")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.inputs
                   if r.name == "Subgrade reaction")
        assert row.value.value == 150.0 and row.value.unit == "pci"
        assert row.value.to_si() is None
        (flag,) = [u for u in result.unresolved
                   if u["what"] == "input Subgrade reaction"]
        assert "not one the record can convert" in flag["why"]

    def test_a_value_with_no_label_is_refused(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(results=[
            _value("   ", 0.52, "in")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert result.calculation.results == []
        assert any("unnamed" in u["what"] for u in result.unresolved)

    def test_a_page_outside_the_run_loses_its_box(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Footing Depth (ft)", 3.0, "ft", page=41,
                   bbox=(1.0, 2.0, 3.0, 4.0))])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.inputs
                   if r.name == "Footing Depth (ft)")
        assert row.prov.page == 0 and row.prov.bbox is None
        assert any("is not part of this printout" in u["why"]
                   for u in result.unresolved)

    def test_a_summary_past_sixty_words_is_cut_and_recorded(self,
                                                            spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(
            summary=" ".join(["word"] * 80))}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert len(result.calculation.summary.split()) == 61  # 60 + ellipsis
        assert any(u["what"] == "summary" for u in result.unresolved)

    def test_a_value_read_off_the_picture_is_recorded_as_a_change(
            self, plotted):
        doc, _gt = plotted
        engine = FakeEngine([{"final": _reading(
            kind="slope_stability",
            results=[_value("Factor of safety", 1.478, from_image=True,
                            note="read off the box on the section")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.results)
        assert row.prov.method == "model_from_picture"
        assert any(c["what"] == "result Factor of safety"
                   for c in result.changes)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

class TestTheLoop:

    def test_one_printout_costs_one_call(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading()}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert engine.n_calls == 1 and result.model_calls == 1
        assert result.calculation.pages == [0]

    def test_the_tool_is_offered_and_withdrawn_on_the_last_call(
            self, plotted):
        doc, gt = plotted
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 0, "bbox": list(gt.plot_bbox),
                                      "why": "read the factor of safety"})]},
            {"final": _reading(kind="slope_stability")},
        ])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert [c["tools"] for c in engine.calls] == [["zoom_plot"], []]
        assert result.tool_calls == 1
        assert any(c["what"] == "zoom_plot" for c in result.changes)

    def test_a_tool_call_on_a_page_outside_the_run_is_an_answer_not_a_stop(
            self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([
            {"tools": [("zoom_plot", {"page": 9, "bbox": [0, 0, 10, 10]})]},
            {"final": _reading()},
        ])

        read_calculation(doc, [0], engine, report_id="SYN")

        (block,) = engine.tool_results()
        assert block["is_error"] is True
        assert "not part of this printout" in str(block["content"])

    def test_the_second_call_reads_the_pages_the_first_was_not_shown(
            self, tmp_path):
        from report_ingest.tests.calc_fixtures import build_long_printout
        gt = build_long_printout(16)
        doc = _open(gt.pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(kind="lateral_pile", continues=True,
                               inputs=[_value("Pile diameter", 0.61, "m")])},
            {"final": _reading(kind="lateral_pile", results=[
                _value("Maximum bending moment", 612.4, "kN-m", page=15)])},
        ])
        pages = list(range(16))
        try:
            result = read_calculation(doc, pages, engine, report_id="SYN")
        finally:
            doc.close()

        assert result.model_calls == 2
        first = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "You are shown 12 of its 16 pages" in first
        assert "--- page 0:" in first and "--- page 15:" in first
        assert "--- page 12:" not in first
        second = engine.calls[1]["messages"][0]["content"][1]["text"]
        assert "--- page 11:" in second and "--- page 14:" in second
        # The two readings are folded into one calculation.
        assert any(r.name == "Pile diameter"
                   for r in result.calculation.inputs)
        assert any(r.name == "Maximum bending moment"
                   for r in result.calculation.results)

    def test_the_budget_is_never_exceeded(self, spreadsheet):
        doc, gt = spreadsheet
        engine = FakeEngine([
            {"final": _reading(unsettled=[
                Unsettled(what="the modulus column", page=0,
                          why="the column heading is cut off")])},
            {"final": _reading()},
            {"final": _reading()},
        ])

        read_calculation(doc, [0], engine, report_id="SYN")

        assert engine.n_calls == MAX_CALC_CALLS

    def test_a_reply_with_no_answer_is_asked_again_once(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"text": "I will look at the table first."},
                             {"final": _reading()}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert result.model_calls == 2
        assert result.calculation is not None

    def test_a_run_that_never_answers_raises(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"text": "thinking"}, {"text": "still thinking"}])

        with pytest.raises(RuntimeError, match="no structured reading"):
            read_calculation(doc, [0], engine, report_id="SYN")

    def test_no_pages_is_refused(self, spreadsheet):
        doc, _gt = spreadsheet
        with pytest.raises(ValueError, match="at least one page"):
            read_calculation(doc, [], FakeEngine([]), report_id="SYN")

    def test_the_first_page_travels_as_a_picture(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading()}])

        read_calculation(doc, [0], engine, report_id="SYN")

        assert engine.calls[0]["n_images"] == 1


class TestThePageWindow:
    """A long printout is shown its FIRST and its LAST pages, always."""

    def test_a_short_run_is_shown_whole(self):
        assert page_window([3, 4, 5]) == [3, 4, 5]

    def test_a_long_run_keeps_the_first_and_the_last(self):
        pages = list(range(100, 130))
        window = page_window(pages)

        assert len(window) == MAX_CALC_PAGES
        assert window[0] == 100 and window[-1] == 129
        assert window[1:-1] == pages[1:MAX_CALC_PAGES - 1]

    def test_the_second_window_continues_where_the_first_stopped(self):
        pages = list(range(100, 130))
        first = page_window(pages)
        second = page_window(pages, MAX_CALC_PAGES - 2)

        assert second[0] == 100 and second[-1] == 129
        assert second[1] == first[-2] + 1

    def test_a_run_that_fits_has_no_second_window(self):
        assert page_window([3, 4, 5], 10) == []

    def test_a_run_shown_whole_says_nothing_about_a_window(self,
                                                            spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading()}])
        read_calculation(doc, [0], engine, report_id="SYN")
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]

        assert "You are shown" not in brief      # the whole run was shown


# ---------------------------------------------------------------------------
# what the model is shown and what it may say
# ---------------------------------------------------------------------------

class TestTheBrief:

    def test_the_system_prompt_states_the_four_rules(self):
        for phrase in ("THE KIND IS WHAT THE CALCULATION WORKS OUT",
                       "THE PROGRAM IS WHAT THE BANNER PRINTS, OR NOTHING",
                       "AN INPUT IS WHAT IT WAS GIVEN",
                       "UNITS STAY AS PRINTED AND NOTHING IS COMPUTED",
                       "You may never DROP one"):
            assert phrase in CALC_READER_SYSTEM

    def test_every_kind_has_a_definition(self):
        assert set(CALC_KIND_DEFINITIONS) == set(CALC_KINDS)

    def test_the_page_serialisation_carries_the_boxes_and_the_tables(
            self, spreadsheet):
        doc, _gt = spreadsheet
        text = serialise_page(doc, 0)

        assert "x0,y0,x1,y1 | text" in text
        assert "TABLE" in text
        assert "Footing Bearing Pressure (ksf)" in text

    def test_the_only_tool_is_the_zoom(self):
        assert [t["name"] for t in CALC_TOOLS] == ["zoom_plot"]


class TestPrintedNumbers:

    def test_every_number_on_the_page_is_in_the_pool(self, spreadsheet):
        doc, _gt = spreadsheet
        pool = printed_numbers(doc, [0])

        for wanted in (8.5, 4.0, 19.2, 0.52, 0.26, 0.31):
            assert any(abs(v - wanted) < 1e-9 for v in pool), wanted

    def test_a_page_that_will_not_read_is_skipped_not_raised(self,
                                                             spreadsheet):
        doc, _gt = spreadsheet
        assert printed_numbers(doc, [99]) == []


# ---------------------------------------------------------------------------
# where one printout ends and the next begins
# ---------------------------------------------------------------------------

class TestSplittingACalculationRun:
    """The defect the hand truth found: an appendix of printouts run back to
    back came back as ONE item folding four different calculations."""

    @pytest.fixture
    def two_banners(self, tmp_path):
        from report_ingest.tests.sounding_fixtures import build_two_banner_run
        doc = _open(build_two_banner_run(), tmp_path)
        yield doc
        doc.close()

    def test_a_run_with_two_banners_becomes_two_items(self, two_banners):
        from report_ingest.calc_reader import split_calc_runs
        assert split_calc_runs(two_banners, [0, 1, 2, 3]) == [[0, 1], [2, 3]]

    def test_a_single_page_is_its_own_run(self, two_banners):
        from report_ingest.calc_reader import split_calc_runs
        assert split_calc_runs(two_banners, [2]) == [[2]]

    def test_a_run_of_one_printout_is_not_split(self, two_banners):
        from report_ingest.calc_reader import split_calc_runs
        assert split_calc_runs(two_banners, [0, 1]) == [[0, 1]]

    def test_a_run_is_capped_at_the_readers_own_ceiling(self, two_banners):
        from report_ingest.calc_reader import MAX_CALC_PAGES, split_calc_runs
        # Pages the document does not have read as nothing, which does not
        # split; the cap still bites.
        runs = split_calc_runs(two_banners, list(range(MAX_CALC_PAGES + 4)))
        assert all(len(run) <= MAX_CALC_PAGES for run in runs)
        assert len(runs) >= 2

    def test_the_items_keep_their_title_and_say_they_were_split(
            self, two_banners):
        from planlens.document.roles import Item
        from report_ingest.calc_reader import split_calc_items
        item = Item(id="item_3", kind="calculation", pages=[0, 1, 2, 3],
                    title="Appendix C", evidence={"first_page": 0})
        out = split_calc_items(two_banners, [item])
        assert [i.id for i in out] == ["item_3a", "item_3b"]
        assert [i.pages for i in out] == [[0, 1], [2, 3]]
        assert all(i.title == "Appendix C" for i in out)
        assert all(i.evidence["split_from"] == "item_3" for i in out)
        assert out[1].evidence["first_page"] == 2

    def test_an_item_of_another_kind_passes_through_untouched(
            self, two_banners):
        from planlens.document.roles import Item
        from report_ingest.calc_reader import split_calc_items
        item = Item(id="item_1", kind="boring_log", pages=[0, 1])
        assert split_calc_items(two_banners, [item]) == [item]

    def test_each_split_item_reads_as_its_own_calculation(self, two_banners):
        from report_ingest.calc_reader import split_calc_runs
        runs = split_calc_runs(two_banners, [0, 1, 2, 3])
        first = floor_from_pages(two_banners, runs[0], "RXX")
        second = floor_from_pages(two_banners, runs[1], "RXX")
        assert "lpile" in (first.program or "").lower()
        assert "stabl" in (second.program or "").lower()
