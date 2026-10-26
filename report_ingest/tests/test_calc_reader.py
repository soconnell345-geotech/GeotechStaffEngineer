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

    def test_a_program_named_by_a_word_needs_its_banner_to_say_so(self):
        """RI-17: "SHAFT" and "GROUP" are words as well as programs. Their
        banner is read where it says "for Windows" or gives a version; the
        word in prose is not a banner."""
        assert _program_in("SHAFT for Windows, Version 2019.4.2") == (
            "SHAFT", "2019.4.2")
        assert _program_in("GROUP for Windows") == ("GROUP", "")
        assert _program_in("Group v10") == ("Group", "10")
        assert _program_in("Section 1 is a drilled shaft with casing") is None
        assert _program_in("the pile group version of the check") is None

    def test_the_floor_of_a_spreadsheet_names_no_program(self, spreadsheet):
        """Which is the right answer: a spreadsheet names none, and the
        reader must not be handed an invented one as its starting record."""
        doc, _gt = spreadsheet
        floor = floor_from_pages(doc, [0], "SYN")

        assert floor.program is None

    def test_a_clock_time_in_a_footer_is_not_a_labelled_value(self):
        assert _pair_in_line("Thursday, June 29, 2006 11:53:13 AM") is None
        assert _pair_in_line("Date: 4/29/2024") is None
        assert _pair_in_line("Design ESALs .......... 245,318") == (
            "Design ESALs", 245318.0, "")

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

    def test_a_contradiction_with_no_place_keeps_both_values(
            self, spreadsheet):
        """The reader named no box, so nothing says its 12.0 and the
        pattern's 8.5 came off one place (RI-74 (2)). Neither replaces the
        other: both stand as rows, and the split is a disagreement."""
        doc, _gt = spreadsheet
        floor = _floor_of(doc)
        model = Calculation(
            kind="settlement",
            inputs=[NamedQuantity(
                name="Footing Width B (ft)",
                value=Quantity(value=12.0, unit="ft"),
                prov=Provenance(page=0, method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        values = sorted(r.value.value for r in merged.inputs
                        if r.name == "Footing Width B (ft)")
        assert values == [8.5, 12.0]
        (split,) = [d for d in log.disagreements
                    if d["what"] == "Footing Width B (ft)"]
        assert split["kept"] == "both"
        assert split["floor"] == "8.5 ft" and split["model"] == "12 ft"

    def test_a_reading_with_no_place_is_not_overwritten_in_turn(self):
        """A sheet prints a short label on several cells of a table, and the
        reader's one row under a longer label that contains it names no box
        (its page was refused and the box dropped). Settled against that
        row one after another, each pattern value replaced the last and the
        last stood. Now the reader's value stands and each printed value is
        its own row."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        cells = [FloorValue(name="Fs", value=v, unit="", page=0,
                            bbox=(100.0 + 60 * n, 200.0, 140.0 + 60 * n,
                                  210.0),
                            method="text", confidence=0.7)
                 for n, v in enumerate((0.45, 0.85, 1.15))]
        floor = CalcFloor(values=cells, pages=[0])
        model = Calculation(kind="site_response", inputs=[NamedQuantity(
            name="Site Class (Fs table)", text="B",
            prov=Provenance(page=0, method="model", confidence=0.4))])

        merged, log = merge_calculation(floor, model)

        reader = next(r for r in merged.inputs
                      if r.name == "Site Class (Fs table)")
        assert reader.text == "B" and reader.value is None
        assert not reader.prov.alternatives
        assert sorted(r.value.value for r in merged.inputs
                      if r.name == "Fs") == [0.45, 0.85, 1.15]
        assert [d["kept"] for d in log.disagreements] == ["both"] * 3

    def test_a_box_on_the_floors_own_line_overrules_it_without_a_note(self):
        """The pattern took the next number on the line, which was another
        column's; the reader boxed the same line and read its own column.
        The line is where the page settles it (RI-74 (2) (b)); the floor's
        number stays beside the reader's as the alternative."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        line = (40.0, 300.0, 520.0, 312.0)
        floor = CalcFloor(values=[FloorValue(
            name="Allowable bearing (kPa)", value=410.0, unit="kPa", page=0,
            bbox=line, method="tables", confidence=0.85)], pages=[0])
        model = Calculation(kind="shallow_foundation_bearing", results=[
            NamedQuantity(name="Allowable bearing (kPa), case A",
                          value=Quantity(value=137.0, unit="kPa"),
                          prov=Provenance(page=0,
                                          bbox=(300.0, 300.0, 340.0, 312.0),
                                          method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        (row,) = merged.results
        assert row.value.value == 137.0
        assert [a.value for a in row.prov.alternatives] == ["410 kPa"]
        assert "own printed line" in row.prov.alternatives[0].note
        assert [d["kept"] for d in log.disagreements] == ["model"]
        assert not [r for r in merged.inputs
                    if r.name.startswith("Allowable")]

    def test_a_floor_entry_with_no_number_never_wins_against_a_number(self):
        """The value is a symbol's, printed after "phi =", and the pattern
        read the label's own tail as the value. A pattern that read words
        cannot correct a number (RI-74 (2) (a)), boxed or not."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        floor = CalcFloor(values=[FloorValue(
            name="Backfill friction angle", text="phi =", page=0,
            bbox=(40.0, 120.0, 200.0, 132.0), method="text",
            confidence=0.7)], pages=[0])
        model = Calculation(kind="retaining_wall", inputs=[NamedQuantity(
            name="Backfill friction angle: phi",
            value=Quantity(value=31.0, unit="deg"),
            prov=Provenance(page=0, method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        (row,) = merged.inputs
        assert row.value.value == 31.0
        assert [a.value for a in row.prov.alternatives] == ["phi ="]
        assert [d["kept"] for d in log.disagreements] == ["model"]

    def test_two_units_of_different_kinds_are_two_readings(self):
        """A diameter of 700 mm and a reinforcement ratio of 0.7 % are both
        0.7 in SI. They are not one reading, so the floor's is not folded
        away as 'reconciled'."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        floor = CalcFloor(values=[FloorValue(
            name="Shaft, reinforcement ratio", value=0.7, unit="%", page=0,
            bbox=(40.0, 120.0, 200.0, 132.0), method="text",
            confidence=0.7)], pages=[0])
        model = Calculation(kind="lateral_pile", inputs=[NamedQuantity(
            name="Shaft, reinforcement ratio",
            value=Quantity(value=700.0, unit="mm"),
            prov=Provenance(page=0, bbox=(40.0, 120.0, 200.0, 132.0),
                            method="model", confidence=0.9))])

        _merged, log = merge_calculation(floor, model)

        assert log.reconciled == 0
        assert [d["floor"] for d in log.disagreements] == ["0.7 %"]

    # -- a place is not a label: what demotes the floor at its own line --

    def test_a_label_names_the_same_thing_only_with_all_its_words(self):
        from report_ingest.calc_reader import _names_same_thing
        assert _names_same_thing("Allowable bearing (kPa)",
                                 "Allowable bearing (kPa), case A")
        assert _names_same_thing("Settlement (in), last row of the table",
                                 "Settlement (in), case 3")
        assert not _names_same_thing("Q", "Qult")
        assert not _names_same_thing("Width (B), case 1",
                                     "Width (B), case 2")

    def test_another_label_at_the_floors_place_keeps_both(self):
        """A short label inside a longer one passes the merge's fuzzy label
        test, and the reader boxed the same line: but "Q" is not "Qult".
        The floor's number is not demoted on the strength of the place."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        floor = CalcFloor(values=[FloorValue(
            name="Q", value=150.0, unit="kN", page=0,
            bbox=(40.0, 300.0, 520.0, 312.0), method="text",
            confidence=0.7)], pages=[0])
        model = Calculation(kind="lateral_pile", results=[NamedQuantity(
            name="Qult", value=Quantity(value=450.0, unit="kN"),
            prov=Provenance(page=0, bbox=(300.0, 300.0, 340.0, 312.0),
                            method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model)

        assert [r.value.value for r in merged.results] == [450.0]
        assert not merged.results[0].prov.alternatives
        assert [(r.name, r.value.value) for r in merged.inputs] == [
            ("Q", 150.0)]
        (split,) = log.disagreements
        assert split["kept"] == "both" and "another label" in split["why"]

    @pytest.mark.parametrize("pool, kept", [
        (None, "model"), ([410.0, 137.0, 3.75], "model"),
        ([410.0, 3.75], "both")])
    def test_at_its_own_line_the_reader_wins_only_with_a_printed_number(
            self, pool, kept):
        """Given the pages' printed numbers, a reader's number on the
        floor's own line replaces the floor's only if the page prints it.
        With no numbers given -- a saved run merged again -- as before."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        floor = CalcFloor(values=[FloorValue(
            name="Allowable bearing (kPa)", value=410.0, unit="kPa", page=0,
            bbox=(40.0, 300.0, 520.0, 312.0), method="tables",
            confidence=0.85)], pages=[0])
        model = Calculation(kind="shallow_foundation_bearing", results=[
            NamedQuantity(name="Allowable bearing (kPa), case A",
                          value=Quantity(value=137.0, unit="kPa"),
                          prov=Provenance(page=0,
                                          bbox=(300.0, 300.0, 340.0, 312.0),
                                          method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model, pool=pool)

        assert [d["kept"] for d in log.disagreements] == [kept]
        assert [r.value.value for r in merged.results] == [137.0]
        floor_rows = [r.value.value for r in merged.inputs]
        assert floor_rows == ([410.0] if kept == "both" else [])

    def test_a_reading_in_words_is_not_held_to_the_printed_numbers(self):
        """The pool checks numbers. A contract number the pattern split
        into a value and a "unit" is read whole by the reader, in words, on
        the same line, and the reader's reading stands."""
        from report_ingest.calc_reader import CalcFloor, FloorValue
        floor = CalcFloor(values=[FloorValue(
            name="Contract No", value=47.0, unit="K1180", page=0,
            bbox=(40.0, 300.0, 520.0, 312.0), method="text",
            confidence=0.7)], pages=[0])
        model = Calculation(kind="site_response", inputs=[NamedQuantity(
            name="Contract No", text="47K1180",
            prov=Provenance(page=0, bbox=(300.0, 300.0, 340.0, 312.0),
                            method="model", confidence=0.9))])

        merged, log = merge_calculation(floor, model, pool=[47.0])

        (row,) = merged.inputs
        assert row.text == "47K1180"
        assert [d["kept"] for d in log.disagreements] == ["model"]

    def test_the_merge_is_given_the_pages_printed_numbers(self, tmp_path,
                                                         monkeypatch):
        """read_calculation passes the pool it already builds for the
        result gate; the merge does not read the pages itself."""
        import report_ingest.calc_reader as cr
        from report_ingest.tests.calc_fixtures import build_long_printout
        seen = {}
        real = cr.merge_calculation

        def spy(floor, model, log=None, pool=None):
            seen["pool"] = pool
            return real(floor, model, log, pool=pool)

        monkeypatch.setattr(cr, "merge_calculation", spy)
        doc = _open(build_long_printout(3).pdf, tmp_path)
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Footing Depth (ft)", 3.0, "ft", page=0)])}])
        try:
            read_calculation(doc, [0, 1, 2], engine, budget=1,
                             report_id="SYN")
            pool = printed_numbers(doc, [0, 1, 2])
        finally:
            doc.close()

        assert seen["pool"] == pool and pool

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
        ends the same, and both printed values are in it: the reader's in
        the slot, block 2's 7 beside it as the floor's alternative. Settled
        in the floor's order, block 1's 5 was matched to the reader's row
        first and lost when block 2's 7 then won that row."""
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
            records.append(sorted(
                (r.value.value, tuple(a.value for a in r.prov.alternatives))
                for r in merged.inputs + merged.results if r.name == "Nc"))
        assert records[0] == records[1]
        assert records[0] == [(5.0, ("7",))]

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

    def test_what_is_left_unsettled_is_ranked_by_what_happened(
            self, tmp_path):
        # BACKLOG RI-08: a value kept as printed is a note, a value the
        # page does not show is left open, a box dropped is a change. A box
        # is dropped only on a printout of more than one page (RI-74 (2)).
        from report_ingest.model import ReportRecord
        from report_ingest.reconciler import reconcile
        from report_ingest.tests.calc_fixtures import build_long_printout
        doc = _open(build_long_printout(3).pdf, tmp_path)
        engine = FakeEngine([{"final": _reading(
            inputs=[_value("Wall load", 3.0, "tons/ft"),
                    _value("Footing Depth (ft)", 3.0, "ft", page=57)],
            results=[_value("Total settlement", 987.654, "in")])}])
        try:
            result = read_calculation(doc, [0, 1, 2], engine, budget=1,
                                      report_id="SYN")
        finally:
            doc.close()
        record = reconcile(ReportRecord(), reader_unresolved=result.unresolved)

        rank = {e.where: e.rank for e in record.qa}
        assert rank["input Wall load"] == "note"
        assert rank["result Total settlement"] == "open"
        assert rank["input Footing Depth (ft)"] == "changed"

    def test_a_label_with_nothing_under_it_is_a_note(self, spreadsheet):
        # BACKLOG RI-08: the reader returned a label and no value: nothing
        # was read, so nothing was lost.
        from report_ingest.model import ReportRecord
        from report_ingest.reconciler import reconcile
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(
            inputs=[_value("Wall load")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")
        record = reconcile(ReportRecord(), reader_unresolved=result.unresolved)

        (entry,) = [e for e in record.qa if e.where == "input Wall load"]
        assert "neither a number nor any words" in entry.detail
        assert (entry.kind, entry.rank) == ("out_of_range", "note")

    def test_a_value_with_no_label_is_refused(self, spreadsheet):
        doc, _gt = spreadsheet
        engine = FakeEngine([{"final": _reading(results=[
            _value("   ", 0.52, "in")])}])

        result = read_calculation(doc, [0], engine, report_id="SYN")

        assert result.calculation.results == []
        assert any("unnamed" in u["what"] for u in result.unresolved)

    def test_a_page_outside_the_run_loses_its_box(self, tmp_path):
        from report_ingest.tests.calc_fixtures import build_long_printout
        gt = build_long_printout(3)
        doc = _open(gt.pdf, tmp_path)
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Footing Depth (ft)", 3.0, "ft", page=57,
                   bbox=(1.0, 2.0, 3.0, 4.0))])}])
        try:
            result = read_calculation(doc, [0, 1, 2], engine, budget=1,
                                      report_id="SYN")
        finally:
            doc.close()

        row = next(r for r in result.calculation.inputs
                   if r.name == "Footing Depth (ft)")
        assert row.prov.page == 0 and row.prov.bbox is None
        assert any("is not part of this printout" in u["why"]
                   for u in result.unresolved)

    @pytest.mark.parametrize("cited", [0, 1, 57])
    def test_a_one_page_printout_reads_any_cited_page_as_its_page(
            self, tmp_path, cited):
        """The reader was shown one page and no other, so whatever page it
        cites -- a position in the printout, a picture's index -- it read the
        value there, and the box it copied is that page's (RI-74 (2)). Not a
        refusal; the result's warnings say it was read so."""
        from report_ingest.tests.calc_fixtures import build_long_printout
        gt = build_long_printout(3)
        doc = _open(gt.pdf, tmp_path)
        engine = FakeEngine([{"final": _reading(inputs=[
            _value("Footing Depth (ft)", 3.0, "ft", page=cited,
                   bbox=(1.0, 2.0, 3.0, 4.0))])}])
        try:
            result = read_calculation(doc, [2], engine, budget=1,
                                      report_id="SYN")
        finally:
            doc.close()

        row = next(r for r in result.calculation.inputs
                   if r.name == "Footing Depth (ft)")
        assert row.prov.page == 2 and row.prov.bbox == (1.0, 2.0, 3.0, 4.0)
        assert not any("is not part of this printout" in u["why"]
                       for u in result.unresolved)
        assert any(f"cited page {cited} for 1 value" in w
                   for w in result.warnings)

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

    def test_the_continuation_asks_only_for_the_pages_it_shows(
            self, tmp_path):
        """RI-17 review: the continuation replaces the conversation, so the
        model cannot see its first answer. It is asked for the values on the
        pages it is shown, and told the earlier readings are kept for it --
        never to repeat what it reported, which would invite values from
        pages it is not shown."""
        from report_ingest.tests.calc_fixtures import build_long_printout
        doc = _open(build_long_printout(16).pdf, tmp_path)
        engine = FakeEngine([{"final": _reading(kind="lateral_pile")},
                             {"final": _reading(kind="lateral_pile")}])
        try:
            read_calculation(doc, list(range(16)), engine, report_id="SYN")
        finally:
            doc.close()

        asked = engine.calls[1]["messages"][0]["content"][0]["text"].lower()
        assert len(engine.calls[1]["messages"]) == 1
        assert "pages shown here" in asked
        assert "kept for you" in asked and "do not repeat" in asked
        for wording in ("already reported", "keeping everything",
                        "whole calculation again"):
            assert wording not in asked

    def test_the_second_window_adds_to_the_first_and_drops_nothing(
            self, tmp_path):
        """RI-17 review: every printout over one window now takes the second
        call, and its answer must not replace the first one's. A label read
        on another page is another value, and the first call's unsettled
        list, which the second call never saw, is kept."""
        from report_ingest.tests.calc_fixtures import build_long_printout
        doc = _open(build_long_printout(16).pdf, tmp_path)
        engine = FakeEngine([
            {"final": _reading(
                kind="lateral_pile",
                results=[_value("Head load", 10.0, "kN", page=0)],
                unsettled=[Unsettled(what="a value I could not read", page=3,
                                     why="smudged")])},
            {"final": _reading(
                kind="lateral_pile",
                results=[_value("Head load", 20.0, "kN", page=14)])},
        ])
        try:
            result = read_calculation(doc, list(range(16)), engine,
                                      report_id="SYN")
        finally:
            doc.close()

        assert result.model_calls == 2
        heads = sorted(r.value.value for r in result.calculation.results
                       if r.name == "Head load")
        assert heads == [10.0, 20.0]
        assert any(u.get("refused_by") == "reader"
                   and u.get("what") == "a value I could not read"
                   for u in result.unresolved)

    def test_two_answers_fold_by_label_and_place(self):
        """A later row replaces an earlier one only where both read one
        printed value: the label, on the same page, on one printed line."""
        from report_ingest.calc_reader import _fold_readings
        line = (72.0, 300.0, 400.0, 312.0)
        below = (72.0, 330.0, 400.0, 342.0)
        first = _reading(results=[
            _value("Maximum bending moment, Load Case 3", 41.0, "kN-m",
                   page=2),
            _value("Footing Width B, Footing F-1", 2.0, "m", page=1),
            _value("Head load", 10.0, "kN", page=11, bbox=line),
            _value("Shear, Load Case 3", 5.0, "kN", page=11),
            _value("Shear, Load Case 7", 6.0, "kN", page=11),
            _value("Axial load", 100.0, "kN", page=11, bbox=line),
        ])
        second = _reading(results=[
            _value("Maximum bending moment, Load Case 7", 57.0, "kN-m",
                   page=13),
            _value("Footing Width B, Footing F-9", 3.0, "m", page=12),
            _value("Head load", 12.0, "kN", page=11, bbox=line),
            _value("Shear, Load Case 7", 6.5, "kN", page=11),
            _value("Axial load", 120.0, "kN", page=11, bbox=below),
        ])

        out = _fold_readings(first, second, fresh=True)

        got = sorted((r.name, r.prov.page, r.value) for r in out.results)
        assert got == sorted([
            ("Maximum bending moment, Load Case 3", 2, 41.0),
            ("Maximum bending moment, Load Case 7", 13, 57.0),
            ("Footing Width B, Footing F-1", 1, 2.0),
            ("Footing Width B, Footing F-9", 12, 3.0),
            # The same line read in both windows: the later reading.
            ("Head load", 11, 12.0),
            # The closest label is the one replaced.
            ("Shear, Load Case 3", 11, 5.0),
            ("Shear, Load Case 7", 11, 6.5),
            # One label on two lines of one page is two values.
            ("Axial load", 11, 100.0),
            ("Axial load", 11, 120.0),
        ])

    def test_the_unsettled_lists_are_kept_together_only_after_the_rest(self):
        """After the rest of the pages, the second call never saw the first
        answer, so both lists stand (one entry per thing and page). After the
        unsettled follow-up, which did see it, the second list is what is
        still unsettled."""
        from report_ingest.calc_reader import _fold_readings
        first = _reading(unsettled=[
            Unsettled(what="the modulus column", page=3, why="cut off"),
            Unsettled(what="a smudged load", page=5, why="smudged")])
        second = _reading(unsettled=[
            Unsettled(what="The modulus column", page=3, why="cut off"),
            Unsettled(what="the last plot's axis", page=14, why="no scale")])

        after_rest = _fold_readings(first, second, fresh=True)
        assert [(u.what, u.page) for u in after_rest.unsettled] == [
            ("the modulus column", 3), ("a smudged load", 5),
            ("the last plot's axis", 14)]

        after_follow_up = _fold_readings(first, second)
        assert [(u.what, u.page) for u in after_follow_up.unsettled] == [
            ("The modulus column", 3), ("the last plot's axis", 14)]

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
        # The first page goes as rendered; every picture added after it goes
        # as a JPEG, with the switch on as for a scanned page.
        from report_ingest.engine import image_media_type
        assert [image_media_type(b["png"]) for b in
                engine.calls[0]["image_blocks"]] == ["image/png",
                                                     "image/jpeg"]

    def test_the_result_says_which_arm_it_ran_in(self, two_pages,
                                                 monkeypatch):
        """RI-67: the reading says which arm it ran in, so a run file
        does too, not only the name of the folder it was written to."""
        from report_ingest.calc_reader import PAGE_IMAGES_ENV, use_page_images
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)

        off = read_calculation(
            two_pages, [0, 1],
            FakeEngine([{"final": _reading(kind="lateral_pile")}]),
            report_id="SYN")
        with use_page_images():
            on = read_calculation(
                two_pages, [0, 1],
                FakeEngine([{"final": _reading(kind="lateral_pile")}]),
                report_id="SYN")

        assert off.page_images is False
        assert "page_images" not in off.to_dict()   # an item file as before
        assert on.page_images is True
        assert on.to_dict()["page_images"] is True

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


# ---------------------------------------------------------------------------
# the builder's check mode (RI-87 piece 6, plan section 8.6)
# ---------------------------------------------------------------------------

def _calc_flag(case="multi"):
    from report_ingest.calc_reader import _Builder
    from report_ingest.tests.checks_fixtures import CALC_POOL, calc_cases
    reading, pages = calc_cases()[case]
    builder = _Builder(reading, pages, "R-made-up", [], CALC_POOL,
                       checks="flag")
    return builder, builder.build()


class TestCalcChecksDefaultIsMaster:

    def test_every_calc_case_builds_as_on_master(self):
        from report_ingest.tests.checks_fixtures import (
            build_default, calc_cases, load_snapshot,
        )
        now, then = build_default(), load_snapshot()
        for case in calc_cases():
            assert now[f"calc/{case}"] == then[f"calc/{case}"], case

    def test_refuse_named_is_the_default(self):
        from report_ingest.calc_reader import _Builder
        from report_ingest.tests.checks_fixtures import (
            CALC_PAGES, CALC_POOL, calc_reading,
        )
        a = _Builder(calc_reading(), CALC_PAGES, "R", [], CALC_POOL)
        b = _Builder(calc_reading(), CALC_PAGES, "R", [], CALC_POOL,
                     checks="refuse")
        assert a.build() == b.build()
        assert a.unresolved == b.unresolved and a.warnings == b.warnings


class TestCalcChecksFlag:

    def test_an_answer_citing_positions_is_read_as_positions(self):
        builder, calc = _calc_flag("positions")       # cites 0-2 of 30-32
        load = calc.inputs[0]
        assert load.prov.page == 30 and load.prov.confidence == 0.9
        assert load.prov.bbox == (5.0, 5.0, 50.0, 15.0)
        assert "a position in the 3 page(s)" in load.prov.note
        assert [v.prov.page for v in calc.inputs + calc.results] == [
            30, 31, 32]
        assert [c["page"] for c in builder.changes] == [30, 31, 32]
        assert [u["page"] for u in builder.unresolved] == [31]

    def test_a_mixed_answer_is_read_as_the_reports_pages(self):
        """The printout's answer cites 30 to 32 and 99: the report's pages,
        so 99 is outside the printout, as today, and nothing is moved."""
        builder, calc = _calc_flag()
        assert builder.changes == []
        breadth = calc.inputs[0]
        assert breadth.prov.page == 30 and breadth.prov.confidence == 0.9
        weight = [v for v in calc.inputs if v.name == "Unit weight"][0]
        assert weight.prov.page == 30 and weight.prov.confidence == 0.4
        assert any("put this on page 99" in u["why"]
                   for u in builder.unresolved)

    def test_an_ambiguous_answer_gets_one_note(self):
        from report_ingest.calc_reader import _Builder
        reading = CalcReading.model_validate({
            "kind": "settlement",
            "inputs": [{"name": "Made-up load", "value": 5.0,
                        "unit": "kips", "prov": {"page": 0}},
                       {"name": "Made-up depth", "value": 3.0,
                        "unit": "ft", "prov": {"page": 2}}]})
        builder = _Builder(reading, [1, 2, 3], "R", [], [5.0, 3.0],
                           checks="flag")
        calc = builder.build()
        assert [v.prov.page for v in calc.inputs] == [1, 3]
        (note,) = [u for u in builder.unresolved
                   if u.get("flag") == "page_numbers"]
        assert "2 is among this call's pages and 0 is not" in note["why"]

    def test_a_one_page_printout_keeps_its_own_rule(self):
        builder, calc = _calc_flag("one_page")
        assert [v.prov.page for v in calc.inputs + calc.results] == [40, 40]
        assert len(builder.warnings) == 2 and builder.changes == []

    def test_a_result_is_not_checked_against_the_pool(self):
        """Under flag the printed-number check is placement's (plan 8.4
        step 4), for every stage: the builder keeps the result as read."""
        builder, calc = _calc_flag()
        safety = [r for r in calc.results if r.name == "Factor of safety"][0]
        assert safety.prov.confidence == 0.9 and safety.note == ""
        assert not any("on none of these pages" in u["why"]
                       for u in builder.unresolved)

    def test_the_other_gates_stand(self):
        builder, calc = _calc_flag()
        whys = " | ".join(u["why"] for u in builder.unresolved)
        assert "no printed label" in whys
        assert "'blorps' is not one the record can convert" in whys
