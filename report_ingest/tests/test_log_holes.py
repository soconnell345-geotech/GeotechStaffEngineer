"""The hole each log page's header names, and one log item per hole (RI-117).

Every name, number and page here is made up: the header layouts are the
kinds the corpus prints (a field and its value, a title, a name standing
alone at the head of a scanned log), written with invented hole names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from report_ingest.log_holes import (
    HoleName, drop_other_hole_rows, hole_parts, item_holes, lines_hole,
    missing_holes_qa, names_match, other_hole_pages, sort_rows_by_hole,
    split_log_items, take_rows)
from report_ingest.model import (
    Investigation, Layer, Provenance, QAEntry, Quantity, ReportRecord,
    Sample, WaterLevel)

HEIGHT = 792.0


@dataclass
class _Line:
    text: str
    bbox: Tuple[float, float, float, float]
    rotation: float = 0.0


def _at(text: str, x: float, y: float, width: float = 0.0,
        rotation: float = 0.0) -> _Line:
    w = width or 5.0 * len(text)
    return _Line(text, (x, y, x + w, y + 9.0), rotation)


def _hole(lines: Sequence[_Line]):
    found, _several = lines_hole(lines, HEIGHT, 0)
    return found


@dataclass
class _Page:
    lines: List[_Line]
    height: float = HEIGHT


class _Doc:
    """Pages of stub lines, as ``doc.page(n).lines`` reads them."""

    def __init__(self, pages: Dict[int, List[_Line]]) -> None:
        self.pages = pages

    def page(self, n: int) -> _Page:
        return _Page(self.pages[n])


@dataclass(frozen=True)
class _Item:
    id: str
    kind: str
    pages: List[int]
    title: Any = None
    evidence: Dict[str, Any] = field(default_factory=dict)


def _header(name: str, sheet: str = "Sheet 1 of 1") -> List[_Line]:
    """A log page's header: a form title, the hole field, a sheet number."""
    return [_at("SUBSURFACE LOG  FORM QZ-7", 42, 30, 200),
            _at("Boring No.:", 42, 52, 50), _at(name, 110, 52, 30),
            _at(sheet, 440, 52, 60),
            _at("N=14", 360, 400, 20)]


# ---------------------------------------------------------------------------
# what a header names
# ---------------------------------------------------------------------------

class TestAHeaderNamesItsHole:

    def test_a_field_and_its_value_on_one_line(self):
        found = _hole([_at("Boring No.: QX-12   Sheet 1 of 2", 40, 50, 300)])
        assert found == HoleName("QX-12", "QX12", 0, "label")

    def test_a_label_with_its_value_set_beside_it(self):
        # A value 25 pt right of its label, on its own text line.
        found = _hole([_at("Test Pit Number:", 400, 40, 64),
                       _at("TP-K", 489, 41, 22)])
        assert found is not None and found.text == "TP-K"
        assert found.how == "label"

    def test_a_label_with_its_value_set_under_it(self):
        found = _hole([_at("BOREHOLE ID:", 60, 40, 60),
                       _at("QX-31", 62, 52, 28)])
        assert found is not None and found.text == "QX-31"

    @pytest.mark.parametrize("title, name", [
        ("LOG OF BORING QX-7", "QX-7"),
        ("BORING NO. QX-08", "QX-08"),
        ("TEST PIT TP-M", "TP-M"),
        ("LOG OF TEST PIT 6", "6"),
        ("BORING NO. 14 (continued)", "14"),
    ])
    def test_a_title_that_names_its_hole(self, title, name):
        found = _hole([_at(title, 150, 40, 200)])
        assert found is not None and found.text == name

    def test_a_name_standing_alone_at_the_head_of_a_scan(self):
        # No label on the sheet: the name stands on a line of its own.
        found = _hole([_at("1", 12, 13, 8, rotation=270.0),
                       _at("QX- 3", 320, 118, 25, rotation=358.9),
                       _at("N- 18", 300, 205, 25)])
        assert found == HoleName("QX- 3", "QX3", 0, "alone")

    @pytest.mark.parametrize("text", [
        "SHEET 2", "FIG 3", "N- 18", "PLATE 4", "NO 2", "N 12"])
    def test_a_lone_page_part_or_count_is_not_a_hole(self, text):
        assert _hole([_at(text, 300, 100)]) is None

    def test_a_form_or_standard_number_is_not_a_hole(self):
        assert _hole([_at("SUBSURFACE LOG  FORM QZ-7", 40, 30, 200),
                       _at("Method: QK 31-207", 40, 60, 120),
                       _at("START 7", 300, 60, 40)]) is None

    def test_a_label_naming_two_holes_names_none(self):
        assert _hole([_at("Boring No.: QX-1", 40, 40, 90),
                       _at("Boring No.: QX-2", 300, 40, 90)]) is None

    def test_a_hole_named_below_the_header_is_not_the_page_s(self):
        assert _hole([_at("Boring No.: QX-4", 40, 400, 90)]) is None

    def test_a_footer_title_block_names_the_hole(self):
        assert _hole([_at("BORING NUMBER", 400, 720, 60),
                      _at("QX-21", 470, 720, 30)]).text == "QX-21"

    def test_a_turned_line_is_not_read(self):
        assert _hole([_at("Boring No.: QX-4", 40, 40, 90,
                          rotation=90.0)]) is None

    def test_a_lone_code_counts_only_with_no_labelled_name(self):
        found = _hole([_at("QX-9", 300, 30, 20),
                       _at("Boring No.: QX-5", 40, 50, 90)])
        assert found.text == "QX-5" and found.how == "label"


def _hash_header(name: str = "", sample: str = "S-5K") -> List[_Line]:
    """A form whose hole field prints "#" between series and number, its
    value set beside the label, and the first sample's number at the head
    of the log table, inside the header band (made up)."""
    lines = [_at("BORING NO:", 430, 90, 58),
             _at("SHEET:  1  OF  3", 421, 110, 77),
             _at(sample, 104, 178, 17)]
    if name:
        lines.append(_at(name, 517, 90, 47))
    return lines


def _title_header(name: str = "", logo: str = "QRS6TU") -> List[_Line]:
    """A form titled "Log of Boring" with the hole's name set beside the
    title on its baseline and no name word, under a firm's logo the text
    layer reads with a digit in it (made up)."""
    lines = [_at(logo, 40, 40, 180), _at("Log of Boring", 286, 64, 57),
             _at("Sheet", 459, 64, 24)]
    if name:
        lines.append(_at(name, 381, 63, 22))
    return lines


class TestTheHeadersTheReviewFound:
    """The layouts the corpus sweep of RI-117's first cut found misread,
    each written with made-up names."""

    def test_a_hash_between_series_and_number(self):
        found = _hole(_hash_header("QX # 07"))
        assert found == HoleName("QX # 07", "QX7", 0, "label")

    def test_no_between_series_and_number(self):
        found = _hole([_at("Boring No.: QX No. 4", 40, 50, 120)])
        assert found is not None and found.key == "QX4"

    def test_a_field_whose_value_is_not_read_names_no_hole(self):
        # The sample's number at the table head is not taken in its place.
        assert _hole(_hash_header()) is None

    def test_a_title_with_the_name_beside_it(self):
        found = _hole(_title_header("QX-14"))
        assert found is not None and found.text == "QX-14"
        assert found.how == "label"

    def test_the_logo_is_never_the_name(self):
        assert _hole(_title_header()) is None

    @pytest.mark.parametrize("title, value, name", [
        ("Log of Boring", "14", "14"),
        ("Test Pit Log", "TP-K", "TP-K"),
        ("Log of Test Pit", "QX-6", "QX-6"),
        ("Boring", "QX-05", "QX-05"),
        ("Boring Log", "3", None),          # a bare number only after "log of"
        ("Hole", "4 in", None),             # a size
    ])
    def test_titles_and_what_stands_beside_them(self, title, value, name):
        found = _hole([_at(title, 286, 64, 57), _at(value, 362, 64, 20)])
        assert (found.text if found else None) == name

    def test_a_value_centred_under_its_label(self):
        found = _hole([_Line("Sondage N°", (82.0, 52.0, 137.0, 61.5)),
                       _Line("QX-6", (99.0, 66.0, 121.0, 79.0))])
        assert found is not None and found.text == "QX-6"
        assert found.how == "label"

    def test_a_value_in_the_cell_under_its_label(self):
        # A header table: the label at its cell's left, the value centred
        # in the cell a row below, 10 pt under it.
        found = _hole([_Line("BORING #", (300.0, 30.0, 331.0, 37.0)),
                       _Line("SHEET", (390.0, 30.0, 411.0, 37.0)),
                       _Line("QX-3", (328.0, 47.0, 352.0, 58.0)),
                       _Line("1 OF 2", (386.0, 47.0, 419.0, 58.0))])
        assert found is not None and found.text == "QX-3"

    def test_a_name_in_three_parts(self):
        found = _hole([_at("Boring No.: QV-K-QX-04", 40, 50, 140)])
        assert found is not None and found.text == "QV-K-QX-04"

    @pytest.mark.parametrize("lines", [
        # under a blank field, the sheet number set below it
        [_at("BORING NO.", 40, 40, 50), _at("SHEET 1", 40, 51, 30)],
        # beside a blank field, the elevation
        [_at("BORING NO.", 40, 40, 50), _at("ELEV 47", 120, 40, 40)],
        # the inside diameter, on one line and set beside its label
        [_at("ID: 4 in", 40, 40, 40)],
        [_at("ID:", 40, 40, 15), _at("4 in", 58, 40, 20)],
        # the sheet count after a pit label
        [_at("Pit: 1 of 2", 40, 30, 60)],
    ], ids=["sheet-below", "elev-beside", "id-size", "id-size-beside",
            "pit-sheets"])
    def test_a_field_s_value_that_is_not_a_hole(self, lines):
        assert _hole(lines) is None

    @pytest.mark.parametrize("lines", [
        # the next field set under a blank one, its value beside it
        [_at("Boring No.:", 40, 40), _at("Elev.:", 40, 51),
         _at("38", 70, 51, 10)],
        # the next field beside a blank one, on its line
        [_at("BORING NO.:", 40, 40, 55), _at("ELEV.:", 110, 40, 30),
         _at("38", 145, 40, 10)],
        # the sheet count beside a blank field
        [_at("BORING NO.:", 40, 40, 55), _at("SHEET", 120, 40, 25),
         _at("2", 150, 40, 5), _at("OF 3", 160, 40, 20)],
    ], ids=["next-field-below", "next-field-beside", "sheet-count-beside"])
    def test_a_blank_field_does_not_reach_past_the_next_label(self, lines):
        # The nearest line beside or under a label is its value: a field
        # left blank takes nothing farther on.
        assert _hole(lines) is None

    def test_a_second_sheet_with_its_field_blank_is_not_cut(self):
        sheet = [_at("Boring No.:", 40, 40), _at("Elev.:", 40, 51),
                 _at("38", 70, 51, 10)]
        doc = _Doc({0: sheet + [_at("QX-7", 100, 40, 20)], 1: list(sheet)})
        names = item_holes(doc, [0, 1])
        assert names[0].text == "QX-7" and names[1] is None
        item = _Item("item_2", "boring_log", [0, 1])
        assert split_log_items(doc, [item]) == [item]

    @pytest.mark.parametrize("text", [
        "SS-3", "ST-2", "RUN-1", "U-2", "D-9K", "R-4", "C-11", "S-8K",
        "MAR-29", "QZ-5713", "QRS6TU", "QX12AB"])
    def test_what_never_stands_alone_as_a_hole(self, text):
        assert _hole([_at(text, 300, 150, 20)]) is None

    @pytest.mark.parametrize("text", ["QX-6A", "QX12", "QX-123"])
    def test_what_still_stands_alone_as_a_hole(self, text):
        assert _hole([_at(text, 300, 150, 20)]).text == text


class TestNamesMatch:

    @pytest.mark.parametrize("a, b, same", [
        ("QX-03", "QX-3", True),
        ("QX 3", "QX-3", True),
        ("3", "QX-3", True),
        ("QX-3", "QY-3", False),
        ("QX-3", "QX-13", False),
        ("", "QX-3", False),
        ("QX # 01", "QX-1", True),
        ("QX No. 2", "QX-2", True),
        # a site's or an area's prefix, on one side only
        ("QV-K-QX-04", "QX-4", True),
        ("QX-04", "QV-K-QX-4", True),
        ("QV-K-QX-04", "QW-K-QX-04", False),
        ("TP-M", "M", False),
    ])
    def test_leading_zeros_and_a_bare_number(self, a, b, same):
        assert names_match(a, b) is same


# ---------------------------------------------------------------------------
# the pages of one item
# ---------------------------------------------------------------------------

class TestTheItemsPages:

    def test_a_lone_code_of_another_series_is_not_a_new_hole(self):
        # A continuation sheet with a sample number at its head.
        doc = _Doc({0: _header("QX-4"),
                    1: [_at("S-4K", 300, 100, 15)],
                    2: _header("QX-5")})
        names = item_holes(doc, [0, 1, 2])
        assert names[0].text == "QX-4" and names[1] is None
        assert names[2].text == "QX-5"

    def test_with_no_labels_the_first_page_sets_the_series(self):
        doc = _Doc({0: [_at("QX-1", 300, 100, 20)],
                    1: [_at("S-2K", 300, 100, 15)],
                    2: [_at("QX-2", 300, 100, 20)]})
        names = item_holes(doc, [0, 1, 2])
        assert [n.text if n else None for n in names.values()] == \
            ["QX-1", None, "QX-2"]

    def test_unnamed_pages_stay_with_the_hole_before_them(self):
        names = {0: None, 1: HoleName("QX-1", "QX1", 1, "label"),
                 2: None, 3: HoleName("QX-01", "QX1", 3, "label"),
                 4: HoleName("QX-2", "QX2", 4, "label"), 5: None}
        parts = hole_parts([0, 1, 2, 3, 4, 5], names)
        assert [(run, h.text) for run, h in parts] == \
            [([0, 1, 2, 3], "QX-1"), ([4, 5], "QX-2")]

    def test_the_pages_of_another_hole(self):
        names = {7: HoleName("QX-1", "QX1", 7, "label"), 8: None,
                 9: HoleName("QX-2", "QX2", 9, "label")}
        own, other = other_hole_pages([7, 8, 9], names)
        assert own.text == "QX-1" and list(other) == [9]

    def test_a_name_written_two_ways_is_one_hole(self):
        # A bare number in the field, then the name with its series (or a
        # site's prefix) in the title: names_match takes them for one hole.
        names = {0: HoleName("1", "1", 0, "label"),
                 1: HoleName("QX-1", "QX1", 1, "label"),
                 2: HoleName("QV-QX-1", "QVQX1", 2, "label"),
                 3: HoleName("QX-2", "QX2", 3, "label")}
        parts = hole_parts([0, 1, 2, 3], names)
        assert [(run, h.text) for run, h in parts] == \
            [([0, 1, 2], "1"), ([3], "QX-2")]
        own, other = other_hole_pages([0, 1, 2, 3], names)
        assert own.text == "1" and list(other) == [3]

    def test_a_form_that_changes_on_its_second_sheet_is_not_cut(self):
        doc = _Doc({0: [_at("Boring No.: 1", 40, 50, 80)],
                    1: [_at("LOG OF BORING QX-1 (continued)", 150, 40, 200)]})
        names = item_holes(doc, [0, 1])
        assert names[0].text == "1" and names[1].text == "QX-1"
        item = _Item("item_4", "boring_log", [0, 1])
        assert split_log_items(doc, [item]) == [item]

    def test_names_going_up_one_a_page_are_plate_numbers(self):
        # Top right of each sheet of one hole's log, with no field label.
        doc = _Doc({0: [_at("Q-3", 520, 20, 15)],
                    1: [_at("Q-4", 520, 20, 15)],
                    2: [_at("Q-5", 520, 20, 15)]})
        assert list(item_holes(doc, [0, 1, 2]).values()) == [None] * 3
        item = _Item("item_2", "boring_log", [0, 1, 2])
        assert split_log_items(doc, [item]) == [item]

    def test_labelled_one_page_logs_are_never_plate_numbers(self):
        doc = _Doc({n: [_Line("Sondage N°", (82.0, 52.0, 137.0, 61.5)),
                        _Line(f"Q-{n + 6}", (99.0, 66.0, 121.0, 79.0))]
                    for n in range(3)})
        out = split_log_items(doc, [_Item("item_2", "boring_log", [0, 1, 2])])
        assert [i.title for i in out] == ["Q-6", "Q-7", "Q-8"]

    def test_a_one_letter_series_that_does_not_number_the_pages(self):
        doc = _Doc({0: [_at("Q-1", 300, 100, 15)],
                    1: [_at("Q-1", 300, 100, 15)],
                    2: [_at("Q-2", 300, 100, 15)]})
        assert [n.text for n in item_holes(doc, [0, 1, 2]).values()] == \
            ["Q-1", "Q-1", "Q-2"]


# ---------------------------------------------------------------------------
# splitting the items
# ---------------------------------------------------------------------------

class TestSplitLogItems:

    def test_one_item_per_hole(self):
        doc = _Doc({0: _header("QX-1", "Sheet 1 of 2"),
                    1: _header("QX-01", "Sheet 2 of 2"),
                    2: _header("QX-2"),
                    3: _header("QX-3", "Sheet 1 of 2"),
                    4: [_at("Sheet 2 of 2", 440, 52, 60)]})
        item = _Item("item_4", "boring_log", [0, 1, 2, 3, 4], "QZ-7",
                     {"why": "kept"})
        out = split_log_items(doc, [item])
        assert [(i.id, i.pages, i.title) for i in out] == [
            ("item_4a", [0, 1], "QX-1"), ("item_4b", [2], "QX-2"),
            ("item_4c", [3, 4], "QX-3")]
        assert all(i.kind == "boring_log" for i in out)
        assert out[1].evidence["split_from"] == "item_4"
        assert out[1].evidence["hole"] == "QX-2"
        assert out[1].evidence["why"] == "kept"

    def test_test_pits_split_too(self):
        doc = _Doc({0: [_at("Test Pit Number:", 400, 40, 64),
                        _at("TP-K", 489, 41, 22)],
                    1: [_at("Test Pit Number:", 400, 40, 64),
                        _at("TP-L", 489, 41, 22)]})
        out = split_log_items(doc, [_Item("item_9", "test_pit_log", [0, 1])])
        assert [(i.pages, i.title) for i in out] == [([0], "TP-K"),
                                                     ([1], "TP-L")]

    def test_items_of_one_hole_and_of_other_kinds_are_untouched(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-1"),
                    2: _header("QX-2"), 3: _header("QX-3")})
        one = _Item("item_1", "boring_log", [0, 1])
        lab = _Item("item_2", "lab_test", [2, 3])
        out = split_log_items(doc, [one, lab])
        assert out[0] is one and out[1] is lab

    def test_a_log_whose_pages_name_no_hole_is_untouched(self):
        doc = _Doc({0: [_at("LOG", 40, 40)], 1: [_at("LOG", 40, 40)]})
        item = _Item("item_1", "boring_log", [0, 1])
        assert split_log_items(doc, [item]) == [item]

    def test_a_hash_form_is_cut_only_where_the_hole_changes(self):
        # Each sheet's first sample number heads its log table; one
        # continuation sheet's field is blank.
        doc = _Doc({0: _hash_header("QX # 07", "S-5K"),
                    1: _hash_header("", "S-6K"),
                    2: _hash_header("QX # 07", "S-9K"),
                    3: _hash_header("QX # 08", "S-5K"),
                    4: _hash_header("QX # 08", "S-3K")})
        out = split_log_items(doc, [_Item("item_3", "boring_log",
                                          [0, 1, 2, 3, 4])])
        assert [(i.pages, i.title) for i in out] == [
            ([0, 1, 2], "QX # 07"), ([3, 4], "QX # 08")]

    def test_a_titled_form_under_a_logo_is_cut_by_the_name_beside_it(self):
        doc = _Doc({0: _title_header("QX-14"), 1: _title_header(),
                    2: _title_header("QX-15")})
        out = split_log_items(doc, [_Item("item_5", "boring_log",
                                          [0, 1, 2])])
        assert [(i.pages, i.title) for i in out] == [([0, 1], "QX-14"),
                                                     ([2], "QX-15")]

    def test_every_sheet_with_a_blank_field_stays_one_item(self):
        # "SHEET n" under the blank field is not a hole on each sheet.
        doc = _Doc({n: [_at("BORING NO.", 40, 40, 50),
                        _at(f"SHEET {n + 1}", 40, 51, 30)]
                    for n in range(3)})
        item = _Item("item_6", "boring_log", [0, 1, 2])
        assert split_log_items(doc, [item]) == [item]


# ---------------------------------------------------------------------------
# the QA entry for a hole the record lacks
# ---------------------------------------------------------------------------

class TestMissingHoles:

    def _record(self, *names: str) -> ReportRecord:
        record = ReportRecord()
        record.investigations = [
            Investigation(investigation_id=n, kind="boring") for n in names]
        return record

    def test_a_printed_hole_with_no_log_in_the_record(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2"),
                    2: _header("QX-3")})
        items = [_Item("item_1a", "boring_log", [0]),
                 _Item("item_1b", "boring_log", [1]),
                 _Item("item_1c", "boring_log", [2])]
        found = missing_holes_qa(doc, items, self._record("QX-1", "QX-03"))
        assert len(found) == 1
        entry = found[0]
        assert isinstance(entry, QAEntry) and entry.kind == "partial"
        assert entry.where == "items.boring_log.item_1b"
        assert "QX-2" in entry.detail and entry.pages == [1]
        assert entry.rank == "open"

    def test_nothing_when_every_printed_hole_is_in_the_record(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2")})
        items = [_Item("item_1", "boring_log", [0, 1]),
                 _Item("item_2", "lab_test", [1])]
        assert missing_holes_qa(doc, items,
                                self._record("QX-01", "QX-2")) == []

    def test_a_bare_number_under_a_label_matches_its_series(self):
        doc = _Doc({0: [_at("BORING NO. 4", 150, 40, 80)]})
        assert missing_holes_qa(doc, [_Item("item_1", "boring_log", [0])],
                                self._record("QX-4")) == []

    def test_a_name_standing_alone_is_not_listed(self):
        doc = _Doc({0: [_at("QX-9", 300, 100, 20)]})
        assert missing_holes_qa(doc, [_Item("item_1", "boring_log", [0])],
                                self._record("QX-1")) == []

    def test_a_logo_over_a_titled_form_is_never_listed(self):
        doc = _Doc({0: _title_header("QX-14"), 1: _title_header()})
        assert missing_holes_qa(doc, [_Item("item_1", "boring_log", [0, 1])],
                                self._record("QX-14")) == []

    def test_a_hash_name_matches_the_record_s_name(self):
        doc = _Doc({0: _hash_header("QX # 07")})
        assert missing_holes_qa(doc, [_Item("item_1", "boring_log", [0])],
                                self._record("QX-7")) == []

    def test_the_wording_counts_its_pages_and_names(self):
        doc = _Doc({4: _header("QX-2"), 5: _header("QX-2"),
                    6: _header("QX-3")})
        (entry,) = missing_holes_qa(
            doc, [_Item("item_1", "boring_log", [4, 5, 6])],
            self._record("QX-1"))
        assert entry.detail == (
            "pages 4-6 print the hole names QX-2, QX-3; no log in the record "
            "carries those names (the reader may have read them under other "
            "names, or not read them)")
        assert entry.values == ["QX-2: pages 4-5", "QX-3: page 6"]
        assert entry.pages == [4, 5, 6] and entry.rank == "open"

    def test_one_page_and_one_name(self):
        doc = _Doc({4: _header("QX-2")})
        (entry,) = missing_holes_qa(
            doc, [_Item("item_1", "boring_log", [4])], self._record("QX-1"))
        assert entry.detail == (
            "page 4 prints the hole name QX-2; no log in the record carries "
            "that name (the reader may have read it under another name, or "
            "not read it)")

    def test_a_model_first_group_is_named_by_its_saved_file(self):
        """RI-144: the model-first path hands in its log groups, whose ids
        are their keys; the entry names the file the group was saved
        under."""
        doc = _Doc({4: _header("QX-1"), 5: _header("QX-2")})
        (entry,) = missing_holes_qa(
            doc, [_Item("log-4", "boring_log", [4, 5])],
            self._record("QX-1"), prefix="mf-")
        assert entry.where == "items.boring_log.mf-log-4"
        assert entry.values == ["QX-2: page 5"] and entry.rank == "open"


# ---------------------------------------------------------------------------
# another hole's rows (RI-145)
# ---------------------------------------------------------------------------

def _m(value: float) -> Quantity:
    return Quantity(value=value, unit="m")


def _layer(page: int, top: float, bottom: float, words: str) -> Layer:
    return Layer(top=_m(top), bottom=_m(bottom), description=words,
                 prov=Provenance(page=page, method="model"))


def _sample(page: int, name: str, top: float) -> Sample:
    return Sample(sample_id=name, top=_m(top), kind="spt",
                  prov=Provenance(page=page, method="model"))


def _log(name: str, pages: List[int], layers=(), samples=(), water=(),
         unit: str = "m") -> Investigation:
    return Investigation(investigation_id=name, kind="boring",
                         depth_unit=unit, pages=list(pages),
                         layers=list(layers), samples=list(samples),
                         water=list(water))


class TestOtherHolePagesPrefer:

    def test_the_log_s_own_name_is_its_own_wherever_it_falls(self):
        names = {0: HoleName("QX-1", "QX1", 0, "label"),
                 1: HoleName("QX-2", "QX2", 1, "label")}
        own, foreign = other_hole_pages([0, 1], names, prefer="QX-2")
        assert own.text == "QX-2" and list(foreign) == [0]

    def test_a_name_no_page_prints_leaves_the_first_named(self):
        names = {0: None, 1: HoleName("QX-1", "QX1", 1, "label"),
                 2: HoleName("QX-2", "QX2", 2, "label")}
        own, foreign = other_hole_pages([0, 1, 2], names, prefer="QX-9")
        assert own.text == "QX-1" and list(foreign) == [2]


class TestTakeRows:

    def test_the_rows_of_the_pages_are_taken_by_page_and_list(self):
        log = _log("QX-1", [0, 1], layers=[_layer(0, 0.0, 2.0, "Made-up A"),
                                           _layer(1, 2.0, 5.0, "Made-up B")],
                   samples=[_sample(1, "Q-4", 2.5)],
                   water=[WaterLevel(depth=_m(1.5), when="while_drilling",
                                     prov=Provenance(page=0))])
        kept, taken = take_rows(log, {1})
        assert [r.description for r in kept.layers] == ["Made-up A"]
        assert kept.samples == [] and len(kept.water) == 1
        assert set(taken) == {1}
        assert [r.description for r in taken[1]["layers"]] == ["Made-up B"]
        assert [s.sample_id for s in taken[1]["samples"]] == ["Q-4"]
        # the log handed in is never changed
        assert len(log.layers) == 2 and len(log.samples) == 1

    def test_nothing_on_the_pages_returns_the_log_itself(self):
        log = _log("QX-1", [0, 1], layers=[_layer(0, 0.0, 2.0, "Made-up A")])
        kept, taken = take_rows(log, {1})
        assert kept is log and taken == {}

    def test_a_row_that_names_no_page_stays(self):
        log = _log("QX-1", [0, 1], layers=[
            Layer(top=_m(0.0), bottom=_m(1.0), description="Made-up C")])
        kept, taken = take_rows(log, {0, 1})
        assert kept is log and taken == {}


class TestDropOtherHoleRows:
    """Today's reader: a whole item's answer, before the build."""

    @staticmethod
    def _reading():
        from report_ingest.log_reader import LogReading, ReadLayer, ReadProv

        return LogReading(
            investigation_id="QX-1", kind="boring", depth_unit="m",
            layers=[ReadLayer(top=0.0, bottom=2.0, description="Made-up A",
                              prov=ReadProv(page=3)),
                    ReadLayer(top=0.0, bottom=4.0, description="Made-up D",
                              prov=ReadProv(page=4)),
                    ReadLayer(top=4.0, bottom=6.0, description="Made-up E",
                              prov=ReadProv(page=4))])

    def test_rows_off_another_hole_s_page_are_dropped_and_listed(self):
        own = HoleName("QX-1", "QX1", 3, "label")
        foreign = {4: HoleName("QX-2", "QX2", 4, "label")}
        kept, rows = drop_other_hole_rows(self._reading(), own, foreign)
        assert [r.description for r in kept.layers] == ["Made-up A"]
        (row,) = rows
        assert row["page"] == 4 and row["value"] == "QX-2"
        assert row["refused_by"] == "python"
        assert row["flag"] == "reader_other_hole"
        assert row["why"] == (
            "the header of page 4 names the hole QX-2, not this log's "
            "(QX-1): the 2 layers the model read off that page are that "
            "hole's and were dropped from this log")
        assert "Made-up D" in row["read"] and "Made-up E" in row["read"]

    def test_with_no_other_hole_the_answer_is_kept_whole(self):
        reading = self._reading()
        own = HoleName("QX-1", "QX1", 3, "label")
        assert drop_other_hole_rows(reading, own, {}) == (reading, [])
        assert drop_other_hole_rows(reading, None, {}) == (reading, [])


class TestSortRowsByHole:
    """Model first: a log group drawn round two holes' pages."""

    def test_rows_go_to_the_log_of_the_hole_the_page_names(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2"),
                    2: _header("QX-2")})
        first = _log("QX-1", [0, 1], layers=[
            _layer(0, 0.0, 2.0, "Made-up A"), _layer(1, 0.0, 1.0, "Made-up B")],
            samples=[_sample(1, "Q-4", 0.5)])
        second = _log("QX-2", [2], layers=[_layer(2, 1.0, 3.0, "Made-up C")])
        logs = [first, second]
        (entry,) = sort_rows_by_hole(doc, logs)
        assert [r.description for r in logs[0].layers] == ["Made-up A"]
        assert logs[0].samples == [] and logs[0].pages == [0]
        assert [r.description for r in logs[1].layers] == [
            "Made-up B", "Made-up C"]                  # in page order
        assert [s.sample_id for s in logs[1].samples] == ["Q-4"]
        assert logs[1].pages == [1, 2]
        assert entry.kind == "note" and entry.rank == "changed"
        assert entry.where == "investigations.QX-1" and entry.pages == [1]
        assert entry.detail == (
            "page 1 names the hole QX-2 in its header, not QX-1: the 1 layer "
            "and 1 sample the model read off it for log QX-1 are that "
            "hole's: moved to log QX-2, where that hole is logged")
        # the logs handed in are not changed: each changed one is a copy
        assert len(first.layers) == 2 and len(second.layers) == 1

    def test_with_no_log_of_that_hole_they_are_dropped_and_kept_as_read(
            self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2")})
        logs = [_log("QX-1", [0, 1], layers=[
            _layer(0, 0.0, 2.0, "Made-up A"),
            _layer(1, 0.0, 1.0, "Made-up B")])]
        (entry,) = sort_rows_by_hole(doc, logs)
        assert [r.description for r in logs[0].layers] == ["Made-up A"]
        assert logs[0].pages == [0]
        assert entry.kind == "out_of_range" and entry.rank == "changed"
        assert entry.pages == [1]
        assert "no other log in the record carries that name" in entry.detail
        assert "dropped from log QX-1" in entry.detail
        assert "Made-up B" in entry.values[0]

    def test_two_logs_carrying_the_name_or_another_unit_drop_them(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2"),
                    2: _header("QX-2"), 3: _header("QX-02")})
        rows = [_layer(0, 0.0, 2.0, "Made-up A"),
                _layer(1, 0.0, 1.0, "Made-up B")]
        two = [_log("QX-1", [0, 1], layers=rows), _log("QX-2", [2]),
               _log("QX-02", [3])]
        (entry,) = sort_rows_by_hole(doc, two)
        assert entry.kind == "out_of_range"
        assert "2 logs in the record carry names like it" in entry.detail
        assert two[1].layers == [] and two[2].layers == []
        feet = [_log("QX-1", [0, 1], layers=rows),
                _log("QX-2", [2], unit="ft")]
        (entry,) = sort_rows_by_hole(doc, feet)
        assert entry.kind == "out_of_range" and feet[1].layers == []
        assert "gives its depths in ft and log QX-1 in m" in entry.detail

    def test_the_log_s_own_name_decides_which_pages_are_another_s(self):
        """A group whose first page is another hole's: the log the model
        named QX-2 keeps its own page and gives page 0's rows to QX-1."""
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2"),
                    5: _header("QX-1")})
        logs = [_log("QX-1", [5], layers=[_layer(5, 3.0, 4.0, "Made-up F")]),
                _log("QX-2", [0, 1], layers=[
                    _layer(0, 0.0, 3.0, "Made-up G"),
                    _layer(1, 0.0, 2.0, "Made-up H")])]
        (entry,) = sort_rows_by_hole(doc, logs)
        assert [r.description for r in logs[1].layers] == ["Made-up H"]
        assert [r.description for r in logs[0].layers] == [
            "Made-up G", "Made-up F"]
        assert entry.pages == [0] and "moved to log QX-1" in entry.detail

    def test_a_page_naming_no_hole_and_logs_before_start_stay(self):
        doc = _Doc({0: _header("QX-1"), 1: [_at("N=14", 360, 400, 20)],
                    2: _header("QX-2")})
        earlier = _log("QX-2", [7])
        log = _log("QX-1", [0, 1, 2], layers=[
            _layer(1, 2.0, 4.0, "Made-up J"), _layer(2, 0.0, 1.0, "Made-up K")])
        logs = [earlier, log]
        (entry,) = sort_rows_by_hole(doc, logs, start=1)
        assert [r.description for r in logs[1].layers] == ["Made-up J"]
        assert logs[1].pages == [0, 1]
        assert logs[0] is earlier and earlier.layers == []
        assert entry.kind == "out_of_range"

    def test_a_log_of_one_hole_is_left_alone(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-01")})
        log = _log("QX-1", [0, 1], layers=[_layer(1, 0.0, 2.0, "Made-up A")])
        logs = [log]
        assert sort_rows_by_hole(doc, logs) == [] and logs[0] is log


# ---------------------------------------------------------------------------
# the synthetic report: three holes on five pages, one planlens item
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    from planlens.document import open_document
    from report_ingest.tests.multi_hole_fixtures import build_multi_hole_log
    gt = build_multi_hole_log(with_text=True)
    path = tmp_path_factory.mktemp("multi") / "multi.pdf"
    path.write_bytes(gt.pdf)
    doc = open_document(str(path))
    yield doc, gt
    doc.close()


class TestOnASyntheticReport:

    def test_planlens_makes_the_three_holes_one_item(self, multi):
        """The defect this module answers, as planlens' rules make it."""
        from planlens.document.roles import (
            assign, build_items, facts_from_document)
        doc, gt = multi
        facts = facts_from_document(doc)
        logs = [i for i in build_items(facts, assign(facts))
                if i.kind == "boring_log"]
        assert [list(i.pages) for i in logs] == [gt.log_pages]

    def test_split_gives_one_item_per_hole(self, multi):
        from planlens.document.roles import (
            assign, build_items, facts_from_document)
        doc, gt = multi
        facts = facts_from_document(doc)
        items = split_log_items(doc, build_items(facts, assign(facts)))
        logs = [i for i in items if i.kind == "boring_log"]
        assert [(list(i.pages), i.title) for i in logs] == [
            (pages, hole) for hole, pages in gt.holes.items()]
        # The continuation printing the name with a leading zero, and the
        # one printing no name, each stay with their hole.
        first = gt.first_log_page
        assert [first, first + 1] in [list(i.pages) for i in logs]
        assert [first + 3, first + 4] in [list(i.pages) for i in logs]


class TestATelephoneLineIsNotAHole:
    """RI-156: a letterhead's telephone or fax line is never a hole name."""

    @pytest.mark.parametrize("value", [
        "Tel 555-0100", "TEL. 555-0100", "Tel: 555 0100", "Telephone 555-0100",
        "Phone: 555-0100", "Ph. 555-0100", "Ph: 555-0100", "fax 555-0100",
        "Mobile 555-0100", "Cell: 555-0100", "CELL 555 0100",
        "(555) 010-0199", "+44 20 7946 0000", "+1 555 0100", "555.010.0199",
        "555-010-0199", "555 010 0199"])
    def test_a_field_value_that_is_a_phone_names_no_hole(self, value):
        assert _hole([_at(f"Boring No.: {value}", 40, 40, 200)]) is None

    @pytest.mark.parametrize("text", [
        "TEL-0199", "TEL0199", "FAX-0199", "FAX 0199", "Tel. 0199",
        "Tel: 0199", "PHONE-0199", "MOBILE-0199", "PH.0199", "CELL:0199"])
    def test_a_phone_standing_alone_is_not_a_hole(self, text):
        assert _hole([_at(text, 300, 100, 60)]) is None

    @pytest.mark.parametrize("value", [
        "BH-1", "B-12", "TP-3", "BH 01", "QX-01", "QX # 01", "14", "TP-M",
        "PH-1", "CELL-1", "B-1 (continued)"])
    def test_hole_names_still_pass_beside_the_refusals(self, value):
        found = _hole([_at(f"Boring No.: {value}", 40, 40, 200)])
        assert found is not None

    @pytest.mark.parametrize("text", ["BH-1", "B-12", "TP-3", "QX-6A", "QX12"])
    def test_names_standing_alone_still_pass(self, text):
        assert _hole([_at(text, 300, 150, 20)]).text == text

    def test_the_letterhead_phone_on_alternate_pages_does_not_cut(self):
        def page(n):
            head = [_at("QX-1", 300, 30, 20)]
            if n % 2:
                head = [_at("TEL-0199", 300, 12, 40)] + head
            return head
        doc = _Doc({n: page(n) for n in range(8)})
        item = _Item("item_1", "boring_log", list(range(8)), "QX-1")
        assert split_log_items(doc, [item]) == [item]
        assert all(h.text == "QX-1"
                   for h in item_holes(doc, range(8)).values())

    @pytest.mark.parametrize("title, phone", [
        ("TEL-55", True), ("TÉL-55", True), ("FAX-55", True),
        ("BH-1", False), ("PH-1", False), ("TP-3", False)])
    def test_a_title_that_is_a_phone_line(self, title, phone):
        from report_ingest.log_holes import is_phone_line
        assert is_phone_line(title) is phone

    def test_a_part_with_no_hole_does_not_keep_a_phone_title(self):
        doc = _Doc({0: _header("QX-1"), 1: _header("QX-2"),
                    2: [_at("Sheet 2 of 2", 440, 52, 60)]})
        item = _Item("item_9", "boring_log", [0, 1, 2], "TÉL-55")
        out = split_log_items(doc, [item])
        assert len(out) == 2 and all(i.title != "TÉL-55" for i in out)
