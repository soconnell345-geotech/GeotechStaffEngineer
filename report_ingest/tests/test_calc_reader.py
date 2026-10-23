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

pymupdf = pytest.importorskip("pymupdf", reason="the fixtures are drawn with PyMuPDF")


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
        """A unit with no conversion in the record's table is still a unit.
        Dropping it would leave a bending moment in the floor as a bare
        number. kN-m was such a unit until RI-20 gave moments a kind; it is
        kept as printed and now converts too."""
        from report_ingest.calc_reader import _unit_token
        doc, _gt = printout
        floor = floor_from_pages(doc, [0], "SYN")

        moment = next(v for v in floor.values
                      if v.name == "Maximum bending moment")
        assert moment.unit == "kN-m"
        si = Quantity(value=moment.value, unit=moment.unit).to_si()
        assert si is not None and si.unit == "kN.m"
        # what still has no conversion is kept as a unit all the same
        for unit in ("tons/ft", "ft-lb", "meq/L"):
            assert Quantity(value=1.0, unit=unit).to_si() is None
            assert _unit_token(unit) == unit

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

    # -- RI-17: one slot per printed place --

    def test_a_label_printed_once_per_block_is_three_values_not_one(
            self, tmp_path):
        """One page prints Nc three times, once per material. The reader
        returned two of them with a box and a note each; the third, which
        only the floor read, is a different printed value and stays in the
        record as its own row. Keyed by the label alone, it was 'corrected'
        to the reader's Nc from another block and lost."""
        from report_ingest.tests.calc_fixtures import (
            build_repeated_label_sheet,
        )
        gt = build_repeated_label_sheet()
        doc = _open(gt.pdf, tmp_path)
        try:
            floor = _floor_of(doc)
            box = {v.value: v.bbox for v in floor.values if v.name == "Nc"}
            model = Calculation(kind="shallow_foundation_bearing", results=[
                NamedQuantity(
                    name="Nc", value=Quantity(value=v, unit=""),
                    prov=Provenance(page=0, bbox=box[v], method="model",
                                    confidence=0.9, note=f"block {block}"))
                for v, block in ((50.59, "C"), (14.83, "B"))])
            merged, log = merge_calculation(floor, model)
        finally:
            doc.close()

        values = sorted(r.value.value for r in merged.inputs + merged.results
                        if r.name == "Nc")
        assert values == [14.83, 30.14, 50.59]
        assert not [d for d in log.disagreements if d["what"] == "Nc"]
        (kept,) = [k for k in log.kept if k["what"] == "Nc"]
        assert kept["value"] == "30.14"
        assert "somewhere else" in kept["why"]

    def test_a_label_on_every_page_keeps_each_pages_value(self, tmp_path):
        """A printout of plots prints the same legend label on every page.
        The reader's row read on the first page says nothing about the
        second page's value, which is kept."""
        from report_ingest.tests.calc_fixtures import build_repeated_label_run
        gt = build_repeated_label_run()
        doc = _open(gt.pdf, tmp_path)
        try:
            floor = _floor_of(doc, range(gt.n_pages))
            box = {v.page: v.bbox for v in floor.values}
            model = Calculation(kind="lateral_pile", inputs=[
                NamedQuantity(
                    name=name, value=Quantity(value=v, unit="kN"),
                    prov=Provenance(page=page, bbox=box[page], method="model",
                                    confidence=0.9, note="the legend"))
                for name, v, page in (("Head load", 12.5, 0),
                                      ("Head load, Pile P-3", 95.0, 2))])
            merged, log = merge_calculation(floor, model)
        finally:
            doc.close()

        values = sorted(r.value.value for r in merged.inputs
                        if r.name.startswith("Head load"))
        assert values == [12.5, 40.0, 95.0]
        assert not log.disagreements
        assert log.reconciled == 2

    @pytest.mark.parametrize("note", ["", "the line in block 2"])
    def test_the_order_the_patterns_met_the_values_in_decides_nothing(
            self, note):
        """Nc is printed 5 in block 1 and 7 in block 2, and the reader read 5
        with block 2's box. Whichever block the floor read first, the record
        ends the same; with no note to overrule block 2's 7, both printed
        values are in it. Settled in the floor's order, block 1's 5 was
        matched to the reader's row first and lost when block 2's 7 then won
        that row."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        block_1 = (300.0, 100.0, 360.0, 112.0)
        block_2 = (300.0, 400.0, 360.0, 412.0)

        def printed(v, box):
            return FloorValue(name="Nc", value=v, unit="", text="", page=0,
                              bbox=box, method="text", confidence=0.6)

        records = []
        for values in ([printed(5.0, block_1), printed(7.0, block_2)],
                       [printed(7.0, block_2), printed(5.0, block_1)]):
            floor = CalcFloor(program=None, program_page=0, title="",
                              values=values, pages=[0])
            model = Calculation(kind="shallow_foundation_bearing", results=[
                NamedQuantity(name="Nc", value=Quantity(value=5.0, unit=""),
                              prov=Provenance(page=0, bbox=block_2,
                                              method="model", confidence=0.9,
                                              note=note))])
            merged, _log = merge_calculation(floor, model)
            records.append(sorted(r.value.value for r in
                                  merged.inputs + merged.results
                                  if r.name == "Nc"))
        assert records[0] == records[1]
        if not note:
            assert records[0] == [5.0, 7.0]

    def test_a_reading_off_the_picture_is_settled_by_the_shared_rule(
            self, spreadsheet):
        """A value read off the picture names no box, so nothing is known
        about its place and the shared rule settles it: a note is what can
        be said about a picture."""
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(kind="settlement", inputs=[NamedQuantity(
            name="Footing Width B (ft)", value=Quantity(value=12.0, unit="ft"),
            prov=Provenance(page=0, method="model_from_picture",
                            confidence=0.8, note="the picture shows 12.0"))])

        merged, log = merge_calculation(floor, model)

        kept = next(r for r in merged.inputs
                    if r.name == "Footing Width B (ft)")
        assert kept.value.value == 12.0
        assert [d["kept"] for d in log.disagreements
                if d["what"] == "Footing Width B (ft)"] == ["model"]

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
        # "tons" is short or metric; nothing in the table guesses which
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Wall load", 3.0, "tons/ft")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        row = next(r for r in result.calculation.inputs
                   if r.name == "Wall load")
        assert row.value.value == 3.0 and row.value.unit == "tons/ft"
        assert row.value.to_si() is None
        (flag,) = [u for u in result.unresolved
                   if u["what"] == "input Wall load"]
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

    def test_an_answer_given_beside_a_look_is_taken(self, spreadsheet):
        # The app's chat model can answer and ask to look in one turn; the
        # answer is whole, so the look is not paid for (the review of RI-03).
        doc, _gt = spreadsheet
        engine = FakeEngine([{"tools": [("zoom_plot", {"page": 0})],
                              "final": _reading()}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert engine.n_calls == 1 and result.tool_calls == 0
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

    def test_the_rest_is_sent_though_the_reader_did_not_ask(self, tmp_path):
        """RI-17: a run the first window did not show whole gets its second
        call whatever ``continues`` says -- a results page or a scanned page
        there would otherwise reach the model only on its say-so."""
        from report_ingest.tests.calc_fixtures import build_long_printout
        doc = _open(build_long_printout(16).pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(kind="lateral_pile", continues=False)},
            {"final": _reading(kind="lateral_pile")},
        ])
        try:
            result = read_calculation(doc, list(range(16)), engine,
                                      report_id="SYN")
        finally:
            doc.close()

        assert result.model_calls == 2
        first = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "The others follow in your second call." in first
        second = engine.calls[1]["messages"][0]["content"][1]["text"]
        assert "--- page 11:" in second and "--- page 14:" in second

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

    def test_a_run_is_capped_at_what_two_calls_show(self, two_banners):
        """RI-17: one printout stays one item up to the reader's two windows,
        and a longer run is cut into balanced parts -- not a full item and an
        orphan."""
        from report_ingest.calc_reader import (
            MAX_CALC_PAGES, MAX_RUN_PAGES, page_window, split_calc_runs,
        )
        # Pages the document does not have read as nothing, which does not
        # split; the cap still bites.
        whole = list(range(2, 2 + MAX_RUN_PAGES))
        assert split_calc_runs(two_banners, whole) == [whole]
        seen = set(page_window(whole)) | set(
            page_window(whole, MAX_CALC_PAGES - 2))
        assert seen == set(whole)                    # two calls show it all
        over = list(range(2, 2 + MAX_RUN_PAGES + 1))
        assert [len(r) for r in split_calc_runs(two_banners, over)] == [
            (MAX_RUN_PAGES + 2) // 2, (MAX_RUN_PAGES + 1) // 2]
        long = list(range(2, 2 + 2 * MAX_RUN_PAGES + 1))
        sizes = [len(r) for r in split_calc_runs(two_banners, long)]
        assert len(sizes) == 3 and max(sizes) - min(sizes) <= 1

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

    # -- RI-17: what the first fix cut that is one printout ------------------

    @staticmethod
    def _runs(tmp_path, heads):
        from report_ingest.calc_reader import split_calc_runs
        from report_ingest.tests.calc_fixtures import build_titled_pages
        doc = _open(build_titled_pages(heads), tmp_path)
        try:
            return split_calc_runs(doc, list(range(len(heads))))
        finally:
            doc.close()

    def test_a_sub_heading_that_changes_page_to_page_does_not_split(
            self, tmp_path):
        heads = [(None, ["Footing Settlement Calculation", "Stratum A"]),
                 (None, ["Footing Settlement Calculation", "Layer Table"]),
                 (None, ["Footing Settlement Calculation",
                         "Strain in Each Layer"])]
        assert self._runs(tmp_path, heads) == [[0, 1, 2]]

    def test_a_cover_a_project_block_and_a_sheet_number_do_not_split(
            self, tmp_path):
        """None of them says what is worked out, so none of them is a title
        that could name a different calculation."""
        heads = [(None, ["Riverbend Logistics Park", "Job 24-0117"]),
                 (None, ["Calculation Output"]),
                 (None, ["Spread Footing Settlement Calculation"]),
                 (None, ["Sheet 4 of 5"]),
                 (None, ["Riverbend Logistics Park", "Job 24-0117"])]
        assert self._runs(tmp_path, heads) == [[0, 1, 2, 3, 4]]

    def test_titles_that_name_different_calculations_split(self, tmp_path):
        heads = [(None, ["Bearing Capacity by the Meyerhof Method"]),
                 (None, ["Bearing Capacity by the Meyerhof Method"]),
                 (None, ["Settlement of a Footing on Rock"])]
        assert self._runs(tmp_path, heads) == [[0, 1], [2]]

    def test_titles_that_differ_by_a_word_are_one_printout(self, tmp_path):
        heads = [(None, ["Lateral Analysis of Micropile MP-2"]),
                 (None, ["Lateral Analysis of Pile P-7"])]
        assert self._runs(tmp_path, heads) == [[0, 1]]

    def test_a_banner_on_every_page_is_one_printout(self, tmp_path):
        """The first fix split such a printout into pairs of pages."""
        heads = [("LPILE Version 2019.11.3", ["Lateral Pile Analysis"])] * 3
        assert self._runs(tmp_path, heads) == [[0, 1, 2]]

    def test_a_banner_back_after_pages_without_one_starts_a_run(
            self, tmp_path):
        heads = [("LPILE Version 2019.11.3", ["Lateral Pile Analysis"]),
                 (None, ["Lateral Pile Analysis"]),
                 ("LPILE Version 2019.11.3", ["Lateral Pile Analysis"]),
                 (None, ["Lateral Pile Analysis"])]
        assert self._runs(tmp_path, heads) == [[0, 1], [2, 3]]

    def test_a_word_a_program_pattern_matches_does_not_split_a_run(
            self, tmp_path):
        """'Strata' is an English word as well as a program's name. A page
        that seems to open with it, in a run that opened with no banner, is
        not the head of a new printout."""
        heads = [(None, ["Footing Settlement Calculation"]),
                 (None, ["Footing Settlement Calculation"]),
                 ("Strata summary", ["Footing Settlement Calculation"]),
                 (None, ["Footing Settlement Calculation"])]
        assert self._runs(tmp_path, heads) == [[0, 1, 2, 3]]

    def test_the_title_words_come_from_the_kind_vocabulary(self):
        from report_ingest.calc_reader import TITLE_WORDS, _KIND_ALIASES
        names: set = set()
        for key in list(CALC_KIND_DEFINITIONS) + list(_KIND_ALIASES):
            names |= set(key.split("_"))
        text = " ".join(CALC_KIND_DEFINITIONS.values()).lower()
        for word in TITLE_WORDS:
            assert word in names or word in text, word

    def test_title_words_are_matched_whole(self):
        from report_ingest.calc_reader import _names_calculation
        for title in ("Micropiles", "Retaining Walls", "Response Spectra",
                      "Footings on Clay", "Lateral Pile Analysis"):
            assert _names_calculation(title), title
        for title in ("Compiled by the checker", "Collateral notes",
                      "Uniaxial compression results", "Shafted"):
            assert not _names_calculation(title), title

    def test_a_word_inside_a_longer_word_does_not_split(self, tmp_path):
        heads = [(None, ["Footing Settlement Calculation"]),
                 (None, ["Compiled by the checker"]),
                 (None, ["Footing Settlement Calculation"]),
                 (None, ["Collateral notes"])]
        assert self._runs(tmp_path, heads) == [[0, 1, 2, 3]]

    def test_a_table_heading_on_a_continuation_page_does_not_split(
            self, tmp_path):
        """Column headings, a heading row of cells and a plot's axis title
        (a quantity with its unit in brackets) are not titles."""
        heads = [(None, ["Lateral Pile Analysis"]),
                 (None, ["Depth   Deflection   Moment   Soil Pressure"]),
                 (None, [("Overburden Pressure", "Depth", "Stress")]),
                 (None, ["Lateral Pile Deflection (meters)"]),
                 (None, ["Lateral Pile Analysis"])]
        assert self._runs(tmp_path, heads) == [[0, 1, 2, 3, 4]]

    def test_a_line_set_like_the_body_is_not_a_title(self, tmp_path):
        """A printout's sentence or table caption that opens the top band in
        the body's type does not say a new calculation starts."""
        heads = [(None, ["Lateral Pile Analysis"]),
                 (None, ["Section 1 is a drilled shaft with casing"], False),
                 (None, ["Summary of Pile-head Responses"], False)]
        assert self._runs(tmp_path, heads) == [[0, 1, 2]]

    def test_a_title_set_as_a_title_still_splits(self, tmp_path):
        heads = [(None, ["Lateral Pile Analysis"]),
                 (None, ["Lateral Pile Analysis"]),
                 (None, ["Retaining Wall Pressure"])]
        assert self._runs(tmp_path, heads) == [[0, 1], [2]]


# ---------------------------------------------------------------------------
# RI-17: which pages travel as pictures
# ---------------------------------------------------------------------------

class TestThePictures:
    """A run is shown whole as text; the first page is a picture, and with
    the switch on every page of the window is."""

    @pytest.fixture
    def two_pages(self, tmp_path):
        from report_ingest.tests.calc_fixtures import build_two_page_printout
        doc = _open(build_two_page_printout().pdf, tmp_path)
        yield doc
        doc.close()

    def test_only_the_first_page_is_a_picture_by_default(self, two_pages,
                                                         monkeypatch):
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        engine = FakeEngine([{"final": _reading(kind="lateral_pile")}])

        read_calculation(two_pages, [0, 1], engine, report_id="SYN")

        assert engine.calls[0]["n_images"] == 1
        assert engine.calls[0]["system"] == CALC_READER_SYSTEM
        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "--- page 1:" in brief            # the second page as text

    def test_with_the_switch_every_page_is_a_picture(self, two_pages):
        from report_ingest.calc_reader import use_page_images
        engine = FakeEngine([{"final": _reading(kind="lateral_pile")}])

        with use_page_images():
            read_calculation(two_pages, [0, 1], engine, report_id="SYN")

        assert engine.calls[0]["n_images"] == 2
        system = engine.calls[0]["system"]
        assert "every page you are shown as a picture" in system
        assert "the first page as a picture" not in system
        # Only a scanned page is re-encoded; a page with its own text stays
        # the PNG it renders to.
        from report_ingest.engine import image_media_type
        assert [image_media_type(b["png"]) for b in
                engine.calls[0]["image_blocks"]] == ["image/png"] * 2

    def test_the_switch_reads_the_environment(self, monkeypatch):
        from report_ingest.calc_reader import (
            PAGE_IMAGES_ENV, page_images_enabled, use_page_images,
        )
        monkeypatch.setenv(PAGE_IMAGES_ENV, "1")
        assert page_images_enabled()
        monkeypatch.setenv(PAGE_IMAGES_ENV, "off")
        assert not page_images_enabled()
        with use_page_images(True):
            assert page_images_enabled()
        assert not page_images_enabled()

    def test_the_second_window_brings_its_own_pictures(self, tmp_path):
        from report_ingest.calc_reader import use_page_images
        from report_ingest.tests.calc_fixtures import build_long_printout
        doc = _open(build_long_printout(16).pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(kind="lateral_pile", continues=True)},
            {"final": _reading(kind="lateral_pile")},
        ])
        try:
            with use_page_images():
                read_calculation(doc, list(range(16)), engine,
                                 report_id="SYN")
        finally:
            doc.close()

        assert engine.calls[0]["n_images"] == MAX_CALC_PAGES
        assert engine.calls[1]["n_images"] == len(
            page_window(list(range(16)), MAX_CALC_PAGES - 2))

    # -- a scanned page is a picture whatever the switch says ----------------

    def test_a_scanned_page_after_the_first_travels_as_a_picture(
            self, tmp_path, monkeypatch):
        """Switch off. The middle page has no text, so its picture is the
        only view of it; before RI-17 it was its own one-page item and went
        as that item's picture."""
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        from report_ingest.engine import image_media_type
        from report_ingest.tests.calc_fixtures import build_scanned_middle_page
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        doc = _open(build_scanned_middle_page(), tmp_path)
        engine = FakeEngine([{"final": _reading()}])
        try:
            read_calculation(doc, [0, 1, 2], engine, report_id="SYN")
        finally:
            doc.close()

        call = engine.calls[0]
        assert call["n_images"] == 2                 # page 0 and page 1
        first, scanned = (b["png"] for b in call["image_blocks"])
        assert image_media_type(first) == "image/png"    # as it was
        assert image_media_type(scanned) == "image/jpeg"  # added: JPEG
        texts = [b.get("text", "") for b in call["messages"][0]["content"]
                 if b.get("type") == "text"]
        assert "The picture of page 0:" in texts
        assert "The picture of page 1:" in texts
        assert "The picture of page 2:" not in texts
        assert "with a picture of every scanned page" in call["system"]

    def test_the_second_window_brings_its_scanned_pictures(
            self, tmp_path, monkeypatch):
        import report_ingest.calc_reader as calc_reader
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        from report_ingest.tests.calc_fixtures import build_long_printout
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        monkeypatch.setattr(calc_reader, "_scanned",
                            lambda _doc, page: page == 14)
        doc = _open(build_long_printout(16).pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(kind="lateral_pile", continues=True)},
            {"final": _reading(kind="lateral_pile")},
        ])
        try:
            read_calculation(doc, list(range(16)), engine, report_id="SYN")
        finally:
            doc.close()

        assert engine.calls[0]["n_images"] == 1
        assert engine.calls[1]["n_images"] == 1
        texts = [b.get("text", "") for b in
                 engine.calls[1]["messages"][0]["content"]
                 if b.get("type") == "text"]
        assert "The picture of page 14:" in texts

    @pytest.mark.parametrize("n_pages, scanned, adds", [
        (16, {0}, False),       # the second window repeats page 0
        (16, set(), False),
        (8, {0}, False),
        (16, {15}, True),       # the last page, in the first window
        (16, {12}, True),       # a page only the second window shows
    ])
    def test_the_prompt_says_scanned_pictures_only_when_it_adds_them(
            self, tmp_path, monkeypatch, n_pages, scanned, adds):
        import report_ingest.calc_reader as calc_reader
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        from report_ingest.tests.calc_fixtures import build_long_printout
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        monkeypatch.setattr(calc_reader, "_scanned",
                            lambda _doc, page: page in scanned)
        doc = _open(build_long_printout(n_pages).pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(kind="lateral_pile", continues=True)},
            {"final": _reading(kind="lateral_pile")},
        ])
        try:
            read_calculation(doc, list(range(n_pages)), engine,
                             report_id="SYN")
        finally:
            doc.close()

        assert (engine.calls[0]["system"] != CALC_READER_SYSTEM) is adds

    def test_the_default_picture_is_the_page_as_rendered(self, two_pages,
                                                         monkeypatch):
        from report_ingest.calc_reader import PAGE_DPI, PAGE_IMAGES_ENV
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        engine = FakeEngine([{"final": _reading(kind="lateral_pile")}])

        read_calculation(two_pages, [0, 1], engine, report_id="SYN")

        (block,) = engine.calls[0]["image_blocks"]
        assert block["png"] == two_pages.render(0, dpi=PAGE_DPI)[0]

    def test_what_counts_as_a_scanned_page(self):
        from planlens.document.model import SOURCE_AZURE_DI, SOURCE_OCR
        from report_ingest.calc_reader import OPTICAL_SOURCES, _scanned
        assert set(OPTICAL_SOURCES) == {SOURCE_OCR, SOURCE_AZURE_DI}

        class _Content:
            def __init__(self, lines, sources):
                self.lines, self.text_sources = lines, sources

        class _Summary:
            def __init__(self, reliable):
                self.text_reliable = reliable

        class _Doc:
            def __init__(self, lines=("a line",), sources=("pdf_text",),
                         reliable=True, broken=False):
                self.args = (list(lines), list(sources), reliable, broken)

            def page(self, _n):
                if self.args[3]:
                    raise ValueError("will not read")
                return _Content(self.args[0], self.args[1])

            def summary(self, _n):
                return _Summary(self.args[2])

        assert not _scanned(_Doc(), 0)                    # its own text
        assert _scanned(_Doc(lines=(), sources=()), 0)    # no text at all
        assert _scanned(_Doc(sources=(SOURCE_AZURE_DI,)), 0)
        assert _scanned(_Doc(sources=(SOURCE_OCR,)), 0)
        assert _scanned(_Doc(reliable=False), 0)          # unmapped fonts
        assert _scanned(_Doc(broken=True), 0)
