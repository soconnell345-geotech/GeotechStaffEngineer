"""Placement by text match (RI-87 piece 5), offline.

Every page here is drawn in the test with PyMuPDF and every value on it is
made up. The pages are opened through planlens as ingest opens a report, so
the words placement reads are the words a real text layer gives.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest

from report_ingest import locate
from report_ingest.checks import pdf_local
from report_ingest.locate import (
    MOVED, NO_PAGE, NO_TEXT, NOT_FOUND, NOT_FOUND_CONFIDENCE,
    OUTSIDE_CONFIDENCE, PLACED, SEVERAL, Candidate, PageWords, Placement,
    Placements, answer_pages, apply, label_score, numbers_in, one_line,
    page_for, place_calculation, place_citations, place_investigation,
    place_lab_tests, same_number,
)
from report_ingest.model import (
    Calculation, Citation, Investigation, LabTest, NamedQuantity, Provenance,
)

pymupdf = pytest.importorskip("pymupdf", reason="the pages are drawn with "
                                                "PyMuPDF")

Text = Tuple[float, float, str]          # x, baseline y, text


def _open(pages: Sequence[Sequence[Text]], tmp_path, name: str = "made.pdf"):
    """A PDF of letter pages with each text drawn at its place, opened
    through planlens; an empty page list draws a blank page."""
    from planlens.document import open_document

    pdf = pymupdf.open()
    for texts in pages:
        page = pdf.new_page(width=612, height=792)
        for x, y, text in texts:
            page.insert_text((x, y), text, fontsize=9, fontname="helv")
    path = tmp_path / name
    pdf.save(str(path))
    pdf.close()
    return open_document(str(path))


def _open_turned(texts: Sequence[Text], degrees: float, tmp_path,
                 name: str = "turned.pdf"):
    """One letter page as a scan tilted by ``degrees`` prints it: each text
    and its place turned together about the page's middle."""
    from planlens.document import open_document

    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    middle = pymupdf.Point(306, 396)
    for x, y, text in texts:
        at = (pymupdf.Point(x, y) - middle) * pymupdf.Matrix(-degrees) \
            + middle
        page.insert_text(at, text, fontsize=9, fontname="helv",
                         morph=(at, pymupdf.Matrix(degrees)))
    path = tmp_path / name
    pdf.save(str(path))
    pdf.close()
    return open_document(str(path))


@pytest.fixture
def opened(tmp_path):
    docs = []

    def make(pages, name="made.pdf"):
        doc = _open(pages, tmp_path, name)
        docs.append(doc)
        return doc, PageWords(doc)
    yield make
    for doc in docs:
        doc.close()


def _prov(page: int, confidence: float = 0.9) -> Provenance:
    return Provenance(page=page, method="model", confidence=confidence)


def _q(value: float, unit: str = "", page: int = 0) -> dict:
    return {"value": value, "unit": unit}


def _value(name: str, value=None, text: str = "", page: int = 0,
           unit: str = "") -> NamedQuantity:
    return NamedQuantity(name=name, value=_q(value, unit) if value is not None
                         else None, text=text, prov=_prov(page))


def _calc(inputs: List[NamedQuantity], results: List[NamedQuantity] = (),
          pages=(0,)) -> Calculation:
    return Calculation.model_validate({
        "kind": "other", "inputs": [v.model_dump() for v in inputs],
        "results": [v.model_dump() for v in results], "pages": list(pages)})


def _middle(box) -> float:
    return (box[1] + box[3]) / 2.0


# ---------------------------------------------------------------------------
# step 1: the page
# ---------------------------------------------------------------------------

def test_a_cited_page_of_the_call_stands():
    numbers = pdf_local([41, 42], [40, 41, 42])
    assert page_for(41, [40, 41, 42], numbers) == (41, "", None)


def test_an_answer_citing_positions_is_read_as_positions_throughout():
    # 0 is no page of the call, and every cite is a position in it: the
    # whole answer gave positions, 1 as well as 0
    numbers = pdf_local([0, 1], [40, 41, 42])
    page, note, confidence = page_for(1, [40, 41, 42], numbers)
    assert (page, confidence) == (41, None)
    assert "position" in note and "41" in note
    assert page_for(0, [40, 41, 42], numbers)[0] == 40


@pytest.mark.parametrize("cited", [99, None])
def test_any_other_page_goes_to_the_calls_first_page_at_low_confidence(cited):
    numbers = pdf_local([cited, 41], [40, 41, 42])
    page, note, confidence = page_for(cited, [40, 41, 42], numbers)
    assert (page, confidence) == (40, OUTSIDE_CONFIDENCE)
    assert "40-42" in note


def test_cites_that_mix_the_calls_pages_and_others_are_read_one_way(
        opened):
    # cites 0 and 2 on a call sent pages 1-3: 2 is a page of the call and 0
    # is not, so 0..2 could all be positions; one reading for all of them.
    # The note that they mix is the builder's, not repeated here.
    doc, words = opened([[], [(72, 100, "Ratio 0.42")], [], []])
    calc = _calc([_value("Ratio", 0.42, page=0), _value("Ratio", 0.42,
                                                        page=2)],
                 pages=(1, 2, 3))
    assert answer_pages([v.prov for v in calc.inputs], [1, 2, 3]).positions
    got = place_calculation(words, calc)
    assert "mix" not in got.qa_note()
    assert [p.cited for p in got.items] == [1, 3]       # read as positions


# ---------------------------------------------------------------------------
# reading printed numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("word, wanted", [
    ("5-9-12", [5.0, 9.0, 12.0]),       # a blow record: dashes, not minus
    ("6.0-7.5", [6.0, 7.5]),            # a range
    ("-2.5", [-2.5]),                   # a minus that opens the word
    ("m3", []),                         # a unit's digits
    ("kN/m3", []),
    ("S-3", []),                        # a label's number
    ("N60=14", [14.0]),
])
def test_numbers_in_a_word(word, wanted):
    assert [n for n, _ in numbers_in(word)] == wanted


def test_a_decimal_comma_is_read_too():
    assert 12.5 in [n for n, _ in numbers_in("12,5")]


def test_a_decimal_comma_is_read_in_printed_order():
    # the first number of a metric range is its top, not the integer before
    # the comma
    got = [n for n, _ in numbers_in("1,50-1,95")]
    assert got[0] == 1.5 and got.index(1.95) < got.index(95.0)
    assert [n for n, _ in numbers_in("1,500")] == [1500.0]


def test_numbers_are_held_to_the_stricter_of_the_two_precisions():
    (printed, tol), = numbers_in("2.45")
    assert not same_number(2.5, printed, tol)       # 2.45 is not 2.5
    (printed, tol), = numbers_in("2.50")
    assert same_number(2.5, printed, tol)
    (printed, tol), = numbers_in("20")
    assert same_number(20.0, printed, tol)
    assert same_number(2.5, 2.54)            # the reported value's own test
    assert not same_number(2.5, 2.54, 0.005)         # a page printing 2.54


def test_a_label_is_scored_by_its_weighted_words():
    assert label_score("Footing Width", "Footing Width B (ft) 31") == 100.0
    assert label_score("Footing Width", "Footing Length L (ft) 31") < 100.0
    # an abbreviation counts, less than the word itself
    assert 50.0 < label_score("coefficient", "coeff 0.3") < 100.0
    assert label_score("", "anything") == 0.0


def test_candidates_along_one_line_are_that_line():
    a = Candidate((10.0, 100.0, 20.0, 110.0))
    b = Candidate((200.0, 101.0, 220.0, 111.0))
    c = Candidate((10.0, 140.0, 20.0, 150.0))
    assert one_line([a, b]).bbox == (10.0, 100.0, 220.0, 111.0)
    assert one_line([a, c]) is None
    assert one_line([a]) is None


# ---------------------------------------------------------------------------
# writing a placement back
# ---------------------------------------------------------------------------

def test_a_placement_sets_the_box_and_keeps_the_method():
    prov = _prov(3)
    apply(prov, Placement("x", "input", 3, 4, MOVED, (1.0, 2.0, 3.0, 4.0),
                          "number", "pdf_text", 1, "moved there"))
    assert (prov.page, prov.bbox, prov.method, prov.confidence) == (
        4, (1.0, 2.0, 3.0, 4.0), "model", 0.9)
    assert "moved there" in prov.note


def test_not_found_on_a_page_with_text_lowers_confidence_and_keeps_page():
    prov = _prov(3)
    apply(prov, Placement("x", "input", 3, 3, NOT_FOUND, note="not found"))
    assert (prov.page, prov.bbox, prov.confidence) == (3, None,
                                                       NOT_FOUND_CONFIDENCE)


@pytest.mark.parametrize("status", [SEVERAL, NO_TEXT])
def test_several_places_or_no_text_keep_confidence(status):
    prov = _prov(3)
    apply(prov, Placement("x", "input", 3, 3, status, note="why"))
    assert (prov.page, prov.bbox, prov.confidence) == (3, None, 0.9)
    assert prov.note == "why"


def test_the_qa_note_counts_what_was_not_placed():
    got = Placements([
        Placement("a", "input", 0, 0, PLACED, (0, 0, 1, 1)),
        Placement("b", "input", 0, 0, NOT_FOUND),
        Placement("c", "input", 0, 0, SEVERAL),
    ])
    note = got.qa_note()
    assert "1 of 3 not found printed on the cited page" in note
    assert "1 of 3 printed in several places" in note
    assert Placements([got.items[0]]).qa_note() == ""


# ---------------------------------------------------------------------------
# a calculation printout
# ---------------------------------------------------------------------------

CALC_SHEET = [
    (72, 80, "Spread footing check"),
    (72, 110, "Footing Width B (ft)"), (300, 110, "6.5"),
    (72, 124, "Footing Length L (ft)"), (300, 124, "6.5"),
    (72, 138, "Allowable Bearing Pressure (ksf)"), (300, 138, "3.25"),
    (72, 152, "Unit weight"), (300, 152, "118 pcf"), (360, 152, "kN/m3"),
    (72, 166, "Site Class is D"),
    (72, 180, "Site Classes: B C D E"),
    (72, 740, "Page 2 of 7"), (300, 740, "Date: 3/14/2021"),
]


def test_a_calculation_value_lands_on_the_line_carrying_its_label(opened):
    doc, words = opened([CALC_SHEET])
    calc = _calc([_value("Footing Width B", 6.5),
                  _value("Footing Length L", 6.5),
                  _value("Allowable bearing pressure", 3.25),
                  _value("Seismic Site Class", text="D")])
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [PLACED] * 4
    width, length, bearing, site = got.items
    assert width.bbox != length.bbox
    assert abs(_middle(width.bbox) - 107) < 6
    assert abs(_middle(length.bbox) - 121) < 6
    assert abs(_middle(bearing.bbox) - 135) < 6
    assert abs(_middle(site.bbox) - 163) < 6         # not the list of classes
    assert calc.inputs[0].prov.bbox == width.bbox
    assert "box from the text layer" in calc.inputs[0].prov.note
    assert calc.inputs[0].prov.method == "model"


def test_a_value_not_printed_drops_to_the_not_found_confidence(opened):
    doc, words = opened([CALC_SHEET])
    calc = _calc([_value("Footing Width B", 6.5),
                  _value("Settlement (in)", 0.75)])
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [PLACED, NOT_FOUND]
    assert calc.inputs[1].prov.confidence == NOT_FOUND_CONFIDENCE
    assert calc.inputs[1].prov.page == 0
    assert "1 of 2 not found printed on the cited page" in got.qa_note()


def test_dates_page_numbers_and_units_print_no_values(opened):
    doc, words = opened([CALC_SHEET])
    calc = _calc([_value("Number of sheets", 7), _value("Day", 14),
                  _value("Exponent", 3)])
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [NOT_FOUND] * 3


TABLE_SHEET = [
    (72, 80, "Footing sizes"),
    (72, 100, "Case"), (140, 100, "Width"), (200, 100, "Length"),
    (260, 100, "Depth"),
    (78, 114, "1"), (142, 114, "2.0"), (202, 114, "2.0"), (262, 114, "1.5"),
    (78, 128, "2"), (142, 128, "1.5"), (202, 128, "4.0"), (262, 128, "1.5"),
    (72, 700, "Clay layer"),
]


def test_a_table_cell_is_found_by_its_column_head_and_its_row(opened):
    doc, words = opened([TABLE_SHEET])
    calc = _calc([_value("Width Case 2 (clay layer)", 1.5),
                  _value("Depth Case 1 (clay layer)", 1.5),
                  _value("Length Case 2 (clay layer)", 4.0)])
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [PLACED] * 3
    width2, depth1, length2 = (p.bbox for p in got.items)
    assert abs(_middle(width2) - 125) < 5 and width2[0] < 170
    assert abs(_middle(depth1) - 111) < 5 and depth1[0] > 250
    assert abs(_middle(length2) - 125) < 5


def test_a_row_of_evenly_stepping_numbers_is_read_as_values(opened):
    doc, words = opened([[
        (72, 80, "Pile capacity"),
        (72, 100, "Pile length (ft)"), (200, 100, "20"), (240, 100, "25"),
        (280, 100, "30"), (320, 100, "35"),
        (72, 114, "Capacity (kips)"), (200, 114, "88"), (240, 114, "104"),
        (280, 114, "121"), (320, 114, "139")]])
    calc = _calc([_value("Pile length", 25)], [_value("Capacity", 104)])
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [PLACED] * 2
    assert abs(_middle(got.items[0].bbox) - 97) < 6
    assert 235 < got.items[0].bbox[0] < 250


def test_a_text_value_whose_words_are_not_printed_is_not_found(opened):
    doc, words = opened([[(72, 100, "Soil profile type"),
                          (72, 120, "Footing Width B (ft) 6.5")]])
    calc = _calc([_value("Soil profile type", text="dense gravel")])
    got = place_calculation(words, calc)
    assert got.items[0].status == NOT_FOUND
    assert calc.inputs[0].prov.bbox is None
    assert calc.inputs[0].prov.confidence == NOT_FOUND_CONFIDENCE


SKEWED_SHEET = [
    (72, 80, "Spread footing check"),
    (72, 100, "Footing Width B (ft)"), (420, 100, "6.5"),
    (72, 110, "Footing Length L (ft)"), (420, 110, "6.5"),
    (72, 120, "Allowable Bearing Pressure (ksf)"), (420, 120, "3.25"),
    (72, 130, "Embedment Depth D (ft)"), (420, 130, "2.0"),
    (72, 140, "Unit weight (pcf)"), (420, 140, "118"),
]


def test_a_tilted_scan_keeps_each_value_on_its_own_line(tmp_path):
    doc = _open_turned(SKEWED_SHEET, 2.0, tmp_path)
    try:
        words = PageWords(doc)
        assert words(0).skew != 0.0
        assert len(words(0).rows) == len(SKEWED_SHEET) - 5
        calc = _calc([_value("Footing Width B", 6.5),
                      _value("Footing Length L", 6.5)])
        got = place_calculation(words, calc)
        assert [p.status for p in got.items] == [PLACED] * 2
        width, length = (p.bbox for p in got.items)
        assert _middle(width) < _middle(length)          # each its own line
    finally:
        doc.close()


def test_a_number_alone_on_another_page_does_not_move_a_value(opened):
    doc, words = opened([[(72, 100, "Cover")], [(300, 100, "0.42")]])
    calc = _calc([_value("Ratio", 0.42, page=0)], pages=(0, 1))
    got = place_calculation(words, calc)
    assert got.items[0].status == NOT_FOUND
    assert calc.inputs[0].prov.page == 0


def test_found_once_on_another_page_of_the_call_moves_there(opened):
    doc, words = opened([[(72, 100, "Cover")],
                         [(72, 100, "Total settlement (in) 1.35")]])
    calc = _calc([], [_value("Total settlement", 1.35, page=0)],
                 pages=(0, 1))
    got = place_calculation(words, calc)
    assert got.items[0].status == MOVED
    prov = calc.results[0].prov
    assert prov.page == 1 and prov.bbox is not None
    assert "found on page 1" in prov.note


def test_printed_twice_with_nothing_between_gives_the_page_only(opened):
    doc, words = opened([[(72, 100, "Ratio 0.42"), (72, 300, "Ratio 0.42")]])
    calc = _calc([_value("Ratio", 0.42)])
    got = place_calculation(words, calc)
    assert got.items[0].status == SEVERAL
    assert got.items[0].candidates == 2
    assert calc.inputs[0].prov.bbox is None
    assert calc.inputs[0].prov.confidence == 0.9


def test_a_page_with_no_text_lowers_nothing(opened):
    doc, words = opened([[]])
    calc = _calc([_value("Ratio", 0.42)])
    got = place_calculation(words, calc)
    assert got.items[0].status == NO_TEXT
    assert calc.inputs[0].prov.confidence == 0.9
    assert "no text" in got.qa_note()


def test_a_page_outside_the_call_is_checked_first(opened):
    doc, words = opened([[(72, 100, "Ratio 0.42")]])
    calc = _calc([_value("Ratio", 0.42, page=9)])
    got = place_calculation(words, calc)
    prov = calc.inputs[0].prov
    assert got.items[0].status == PLACED and prov.page == 0
    assert prov.confidence <= OUTSIDE_CONFIDENCE
    assert "not one of the pages" in prov.note


def test_without_write_the_record_is_left_alone(opened):
    doc, words = opened([CALC_SHEET])
    calc = _calc([_value("Footing Width B", 6.5, page=9)])
    got = place_calculation(words, calc, write=False)
    assert got.items[0].status == PLACED
    assert calc.inputs[0].prov.page == 9 and calc.inputs[0].prov.bbox is None


# ---------------------------------------------------------------------------
# a log
# ---------------------------------------------------------------------------

class _Ruler:
    """A depth ruler: 20 points a foot below y = 100."""

    unit = "ft"
    slope = 20.0
    ticks = ((100.0 + 20.0 * 5, 5.0), (100.0 + 20.0 * 10, 10.0))

    def y_at(self, depth: float) -> float:
        return 100.0 + 20.0 * float(depth)


def _y(depth: float) -> float:
    return 100.0 + 20.0 * depth + 3.0          # a baseline on the depth's row


LOG_PAGE = [
    (40, _y(5), "5"), (40, _y(10), "10"),                     # the ruler
    (80, _y(0), "Brown silty SAND, loose, moist"),
    (80, _y(6), "Gray lean CLAY, stiff"),
    (260, _y(2), "S-1"), (300, _y(2), "3-4-5"),
    (260, _y(7), "ST"), (290, _y(7.7), "2"),                  # a split label
    (220, _y(12), "12.0"), (300, _y(12), "4-6-8"),
    (420, _y(7), "see S-1 note"),                             # S-1 again
    (420, _y(3), "Groundwater at 3.5 ft"),
    (80, _y(3.5), "3.5"),                                     # a depth, again
]


def _investigation() -> Investigation:
    def prov():
        return {"page": 0, "method": "model", "confidence": 0.9}
    return Investigation.model_validate({
        "investigation_id": "TB-1", "depth_unit": "ft", "pages": [0],
        "layers": [
            {"top": _q(0.0), "bottom": _q(6.0), "prov": prov(),
             "description": "Brown silty SAND, loose, moist"},
            {"top": _q(6.0), "bottom": _q(15.0), "prov": prov(),
             "description": "Gray lean CLAY, stiff"}],
        "samples": [
            {"sample_id": "S-1", "top": _q(2.0), "bottom": _q(3.5),
             "prov": prov()},
            {"sample_id": "ST-2", "top": _q(7.0), "bottom": _q(9.0),
             "prov": prov()},
            {"sample_id": "", "top": _q(12.0), "bottom": _q(13.5),
             "prov": prov()}],
        "spt": [{"depth_top": _q(12.0), "blows": [4, 6, 8],
                 "prov": prov()}],
        "water": [{"depth": _q(3.5), "note": "groundwater",
                   "prov": prov()}],
    })


def test_a_logs_objects_land_on_their_rows(opened):
    doc, words = opened([LOG_PAGE])
    inv = _investigation()
    got = place_investigation(words, inv, rulers={0: _Ruler()})
    by_what = {p.what: p for p in got.items}
    assert all(p.status == PLACED for p in got.items), [
        (p.what, p.status) for p in got.items]

    def row_of(what):
        return _middle(by_what[what].bbox)

    assert abs(row_of("layer 1 at 0") - _Ruler().y_at(0)) < 8
    assert abs(row_of("layer 2 at 6") - _Ruler().y_at(6)) < 8
    # the label printed twice: the one on the sample's own row
    assert by_what["sample S-1 at 2"].bbox[0] < 300
    # a label printed as its letters, and its number on the next line: the
    # number, on the sample's own rows
    assert by_what["sample ST-2 at 7"].bbox[0] >= 285
    # no label: its printed depth is not the ruler's tick; the drive's blow
    # record is its own print
    assert by_what["drive 1 at 12"].anchor == "blow record"
    assert by_what["water 1 at 3.5"].bbox[0] > 400


def test_a_sample_not_found_takes_the_box_of_its_drive(opened):
    # neither the sample's label nor its depth is printed; the drive at its
    # top is, by its blow record, and the sample came out of that drive
    doc, words = opened([[(40, _y(5), "5"), (40, _y(10), "10"),
                          (300, _y(14), "7-8-9")]])
    prov = {"page": 0, "method": "model", "confidence": 0.9}
    inv = Investigation.model_validate({
        "investigation_id": "TB-3", "depth_unit": "ft", "pages": [0],
        "samples": [{"sample_id": "S-9", "top": _q(14.0), "bottom": _q(15.5),
                     "prov": prov}],
        "spt": [{"depth_top": _q(14.0), "blows": [7, 8, 9], "prov": prov}]})
    got = place_investigation(words, inv, rulers={0: _Ruler()})
    sample, drive = got.items
    assert drive.status == PLACED and drive.anchor == "blow record"
    assert (sample.status, sample.anchor, sample.bbox) == (
        PLACED, "its drive", drive.bbox)
    assert inv.samples[0].prov.bbox == drive.bbox
    assert inv.samples[0].prov.confidence == 0.9
    assert "came out of that drive" in inv.samples[0].prov.note


def test_a_samples_drive_comes_before_its_depth_alone(opened):
    # the sample's depth is printed on the line of the layer starting
    # there; the drive's blow record, its own print, is the line below
    doc, words = opened([[(40, _y(5), "5"), (40, _y(10), "10"),
                          (80, _y(14), "14.0 Brown lean CLAY"),
                          (300, _y(14.7), "7-8-9")]])
    prov = {"page": 0, "method": "model", "confidence": 0.9}
    inv = Investigation.model_validate({
        "investigation_id": "TB-4", "depth_unit": "ft", "pages": [0],
        "samples": [{"sample_id": "", "top": _q(14.0), "bottom": _q(15.5),
                     "prov": prov}],
        "spt": [{"depth_top": _q(14.0), "blows": [7, 8, 9], "prov": prov}]})
    got = place_investigation(words, inv, rulers={0: _Ruler()})
    sample, drive = got.items
    assert sample.anchor == "its drive" and sample.bbox == drive.bbox
    assert sample.bbox[0] >= 295


class _MetricRuler:
    """A depth ruler: 60 points a metre below y = 100."""

    unit = "m"
    slope = 60.0
    ticks = ((160.0, 1.0), (220.0, 2.0), (280.0, 3.0))

    def y_at(self, depth: float) -> float:
        return 100.0 + 60.0 * float(depth)


def test_a_metric_log_with_decimal_commas_places_its_samples_and_drives(
        opened):
    def y(depth):
        return 100.0 + 60.0 * depth + 3.0
    doc, words = opened([[
        (40, y(1), "1"), (40, y(2), "2"), (40, y(3), "3"),        # the ruler
        (80, y(1.5), "1,50-1,95"), (80, y(4.5), "4,50-4,95"),
        (300, y(2.1), "Water level 2,10")]])
    prov = {"page": 0, "method": "model", "confidence": 0.9}
    inv = Investigation.model_validate({
        "investigation_id": "P-1", "depth_unit": "m", "pages": [0],
        "samples": [{"sample_id": "", "top": _q(top, "m"),
                     "bottom": _q(bottom, "m"), "prov": prov}
                    for top, bottom in ((1.5, 1.95), (4.5, 4.95))],
        "spt": [{"depth_top": _q(top, "m"), "depth_bottom": _q(bottom, "m"),
                 "blows": [], "prov": prov}
                for top, bottom in ((1.5, 1.95), (4.5, 4.95))],
        "water": [{"depth": _q(2.1, "m"), "note": "water level",
                   "prov": prov}]})
    got = place_investigation(words, inv, rulers={0: _MetricRuler()})
    assert [p.status for p in got.items] == [PLACED] * 5, [
        (p.what, p.status) for p in got.items]
    first, second = inv.samples
    assert abs(_middle(first.prov.bbox) - _MetricRuler().y_at(1.5)) < 8
    assert abs(_middle(second.prov.bbox) - _MetricRuler().y_at(4.5)) < 8
    assert [d.prov.bbox for d in inv.spt] == [first.prov.bbox,
                                              second.prov.bbox]
    assert inv.water[0].prov.bbox[0] > 290


def test_the_planlens_log_fixture_places_on_its_own_ruler(tmp_path):
    from planlens.document import open_document
    from planlens.testing.loggrid_fixtures import build_imperial_log
    from report_ingest.visual_scales import log_grid_for

    gt = build_imperial_log()
    path = tmp_path / "fixture.pdf"
    path.write_bytes(gt.pdf)
    doc = open_document(str(path))
    try:
        grid, _ = log_grid_for(doc, [0])
        ruler = grid.rulers[0]
        prov = {"page": 0, "method": "model", "confidence": 0.9}
        inv = Investigation.model_validate({
            "investigation_id": "fixture", "depth_unit": gt.unit,
            "pages": [0],
            "layers": [{"top": _q(top), "bottom": _q(bottom) if bottom
                        else None, "description": words, "prov": prov}
                       for top, bottom, words in gt.layers],
            "spt": [{"depth_top": _q(depth),
                     "blows": [int(b) for b in record.split("-")],
                     "prov": prov}
                    for depth, record, _ in gt.samples]})
        got = place_investigation(PageWords(doc), inv, rulers=grid.rulers)
        assert [p.status for p in got.items] == [PLACED] * len(got.items)
        for (depth, _, _), drive in zip(gt.samples, inv.spt):
            assert abs(_middle(drive.prov.bbox) - ruler.y_at(depth)) < 12
        for (top, bottom, _), layer in zip(gt.layers, inv.layers):
            low = ruler.y_at(top) - 12
            high = ruler.y_at(bottom if bottom else top + 5) + 12
            assert low <= _middle(layer.prov.bbox) <= high
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# laboratory tests
# ---------------------------------------------------------------------------

LAB_SHEET = [
    (72, 60, "SUMMARY OF LABORATORY TESTS"),
    (72, 90, "Boring"), (140, 90, "Depth"), (200, 90, "LL"), (240, 90, "PL"),
    (280, 90, "PI"),
    (72, 104, "TB-4"), (140, 104, "2.5"), (200, 104, "41"), (240, 104, "19"),
    (280, 104, "22"),
    (72, 118, "TB-4"), (140, 118, "7.5"), (200, 118, "33"), (240, 118, "17"),
    (280, 118, "16"),
    (72, 132, "TB-9"), (140, 132, "5.0"), (200, 132, "NP"),
]


def _lab(boring: str, top: float, ll=None, pl=None, pi=None,
         sample: str = "") -> LabTest:
    return LabTest.model_validate({
        "kind": "atterberg", "investigation_id": boring, "sample_id": sample,
        "depth_top": _q(top), "pages": [0],
        "result": {"kind": "atterberg", "ll": ll, "pl": pl, "pi": pi},
        "prov": [{"page": 0, "method": "model", "confidence": 0.9}]})


def test_a_lab_test_lands_on_the_row_printing_its_results(opened):
    doc, words = opened([LAB_SHEET])
    tests = [_lab("TB-4", 7.5, 33, 17, 16), _lab("TB-4", 2.5, 41, 19, 22),
             _lab("TB-9", 5.0)]
    got = place_lab_tests(words, tests)
    assert [p.status for p in got.items] == [PLACED] * 3
    second, first, nonplastic = got.items
    assert second.anchor == "result values"
    assert abs(_middle(second.bbox) - 115) < 5
    assert abs(_middle(first.bbox) - 101) < 5
    # no results to match: its boring's label, once printed
    assert nonplastic.anchor == "boring label"
    assert abs(_middle(nonplastic.bbox) - 129) < 5
    assert not any(p.page_level for p in got.items)


def test_tests_found_only_by_the_sheets_title_are_page_level(opened):
    doc, words = opened([LAB_SHEET])
    got = place_lab_tests(words, [_lab("TB-77", 9.0), _lab("TB-78", 9.0),
                                  _lab("TB-4", 7.5, 33, 17, 16)])
    first, second, placed = got.items
    for title in (first, second):
        assert (title.status, title.anchor, title.page_level) == (
            PLACED, "test title", True)
    # a test placed by its own row still shares the page with them
    assert placed.anchor == "result values"
    assert "2 of 3 placed on the page only" in got.qa_note()


@pytest.mark.parametrize("kind", ["summary_table", "compaction",
                                  "atterberg"])
def test_the_answers_only_test_on_a_page_is_placed_at_its_title(opened,
                                                                 kind):
    doc, words = opened([LAB_SHEET, [(72, 60, "COMPACTION TEST")]])
    test = LabTest.model_validate({
        "kind": kind, "investigation_id": "TB-77", "pages": [0, 1],
        "result": {"kind": kind},
        "prov": [{"page": 0, "method": "model", "confidence": 0.9}]})
    other = LabTest.model_validate({
        "kind": "compaction", "pages": [0, 1],
        "result": {"kind": "compaction"},
        "prov": [{"page": 1, "method": "model", "confidence": 0.9}]})
    got = place_lab_tests(words, [test, other])
    title, _ = got.items
    assert (title.status, title.anchor, title.page_level) == (
        PLACED, "test title", False)
    assert abs(_middle(title.bbox) - 57) < 5
    assert "only test on page 0" in test.prov[0].note
    assert got.qa_note() == ""


MOISTURE_TITLE = (72, 60, "MOISTURE CONTENT TEST RESULTS")
#: boring, sample, depth and water content on every row
MOISTURE_ROWS = [
    MOISTURE_TITLE,
    (72, 90, "Boring"), (130, 90, "Sample"), (190, 90, "Depth (ft)"),
    (260, 90, "Water Content (%)"),
    (72, 104, "B-1"), (130, 104, "S-1"), (190, 104, "2.5"),
    (260, 104, "18.3"),
    (72, 118, "B-2"), (130, 118, "S-1"), (190, 118, "2.5"),
    (260, 118, "21.0"),
]
#: each boring a heading over its samples, whose labels repeat
MOISTURE_GROUPS = [
    MOISTURE_TITLE,
    (72, 90, "Boring B-1"),
    (130, 104, "S-1"), (190, 104, "2.5"), (260, 104, "18.3"),
    (130, 118, "S-2"), (190, 118, "5.0"), (260, 118, "20.1"),
    (72, 140, "Boring B-2"),
    (130, 154, "S-1"), (190, 154, "2.5"), (260, 154, "21.0"),
    (130, 168, "S-2"), (190, 168, "5.0"), (260, 168, "17.4"),
]
#: no sample column: boring and depth name the test
MOISTURE_NO_SAMPLES = [
    MOISTURE_TITLE,
    (72, 90, "Boring"), (190, 90, "Depth (ft)"),
    (260, 90, "Water Content (%)"),
    (72, 104, "B-1"), (190, 104, "2.5"), (260, 104, "18.3"),
    (72, 118, "B-1"), (190, 118, "5.0"), (260, 118, "20.1"),
    (72, 132, "B-2"), (190, 132, "2.5"), (260, 132, "21.0"),
]


def _moisture(boring: str, sample: str, top: float, wc: float) -> LabTest:
    return LabTest.model_validate({
        "kind": "moisture_content", "investigation_id": boring,
        "sample_id": sample, "depth_top": _q(top, "ft"), "pages": [0],
        "result": {"kind": "moisture_content", "wc": wc},
        "prov": [{"page": 0, "method": "model", "confidence": 0.9}]})


@pytest.mark.parametrize("sheet, tests, rows, anchor", [
    (MOISTURE_ROWS, [("B-1", "S-1", 2.5, 18.3), ("B-2", "S-1", 2.5, 21.0)],
     [101, 115], "sample label"),
    (MOISTURE_GROUPS, [("B-1", "S-1", 2.5, 18.3), ("B-2", "S-2", 5.0, 17.4)],
     [101, 165], "sample label"),
    (MOISTURE_NO_SAMPLES, [("B-1", "S-1", 2.5, 18.3),
                           ("B-2", "S-1", 2.5, 21.0)],
     [101, 129], "boring label"),
    (MOISTURE_ROWS, [("B-1", "SS-1", 2.5, 18.3)], [101], "boring label"),
], ids=["boring on each row", "boring as a heading", "no sample column",
        "label written another way"])
def test_a_repeated_label_is_narrowed_to_its_tests_row(opened, sheet, tests,
                                                       rows, anchor):
    doc, words = opened([sheet])
    got = place_lab_tests(words, [_moisture(*t) for t in tests])
    assert [(p.status, p.anchor) for p in got.items] == \
        [(PLACED, anchor)] * len(tests)
    for placement, row in zip(got.items, rows):
        assert abs(_middle(placement.bbox) - row) < 5


#: one boring's sheet: its label is a heading over the sample rows
ONE_BORING = [
    MOISTURE_TITLE,
    (72, 90, "Boring B-1"),
    (130, 104, "S-1"), (190, 104, "2.5"), (260, 104, "18.3"),
    (130, 118, "S-2"), (190, 118, "5.0"), (260, 118, "20.1"),
]


def test_a_boring_heading_is_not_a_tests_spot(opened):
    doc, words = opened([ONE_BORING])
    unprinted = _moisture("B-1", "S-3", 7.5, 25.0)
    got = place_lab_tests(words, [unprinted, _moisture("B-1", "S-1", 2.5,
                                                       18.3)])
    heading, placed = got.items
    assert (heading.status, heading.anchor, heading.page_level) == (
        PLACED, "test title", True)
    assert "only as a heading" in unprinted.prov[0].note
    assert locate.boring_heading(words, 0, unprinted)
    assert placed.anchor == "sample label"
    # a boring printed on its test's row is that test's line, not a heading
    doc, words = opened([MOISTURE_ROWS], "rows.pdf")
    assert not locate.boring_heading(words, 0, _moisture("B-2", "S-9", 2.5,
                                                          21.0))


def test_a_boring_label_does_not_move_a_test_to_another_page(opened):
    doc, words = opened([[(72, 60, "SUMMARY OF LABORATORY TESTS")],
                         [(72, 104, "TB-4")]])
    test = _lab("TB-4", 2.5)
    test.pages = [0, 1]
    got = place_lab_tests(words, [test])
    title, = got.items
    assert (title.status, title.page, title.anchor) == (
        PLACED, 0, "test title")


def test_objects_with_no_page_are_counted_in_every_stage(opened):
    doc, words = opened([LOG_PAGE])
    inv = _investigation()
    inv.pages = []
    got = place_investigation(words, inv)
    assert len(got.items) > 0
    assert got.counts()[NO_PAGE] == len(got.items)
    calc = _calc([_value("Unit weight", 120.0)], [_value("Factor", 1.5)],
                 pages=())
    got = place_calculation(words, calc)
    assert [p.status for p in got.items] == [NO_PAGE, NO_PAGE]
    assert "no page to place it on" in calc.inputs[0].prov.note
    got = place_citations(words, {"geology": [Citation(page=0, quote="a")]},
                          [])
    assert [p.status for p in got.items] == [NO_PAGE]
    assert "1 of 1 with no page" in got.qa_note()


def test_a_lab_test_with_no_page_or_no_provenance_is_counted(opened):
    doc, words = opened([LAB_SHEET])
    no_page = _lab("TB-4", 2.5, 41, 19, 22)
    no_page.pages = []
    no_page.prov = []
    got = place_lab_tests(words, [no_page])
    assert got.counts()[NO_PAGE] == 1
    assert "no page to place it on" in got.qa_note()
    no_prov = _lab("TB-4", 2.5, 41, 19, 22)
    no_prov.prov = []
    got = place_lab_tests(words, [no_prov])
    assert got.items[0].status == PLACED
    assert "no provenance" in got.items[0].note
    assert "1 of 1 test(s) have no provenance" in got.qa_note()


# ---------------------------------------------------------------------------
# a narrative's citations
# ---------------------------------------------------------------------------

def test_a_citation_is_checked_on_its_page_then_the_calls_others(opened):
    doc, words = opened([
        [(72, 100, "The site is underlain by stiff clay over sandstone.")],
        [(72, 100, "Groundwater was not encountered in any boring.")]])
    cites = {"geology": [Citation(page=0, quote="underlain by stiff clay "
                                                 "over sandstone")],
             "water": [Citation(page=0, quote="Groundwater was not "
                                               "encountered in any boring")],
             "made_up": [Citation(page=0, quote="a sentence no page prints "
                                                "anywhere at all")]}
    got = place_citations(words, cites, [0, 1])
    assert [p.status for p in got.items] == [PLACED, MOVED, NOT_FOUND]
    assert got.items[1].page == 1
    # a citation has no box to write: the record is left as it was
    assert cites["water"][0].page == 0


def test_a_quote_search_cannot_box_is_on_its_page_with_no_box(opened,
                                                               monkeypatch):
    doc, words = opened([
        [(72, 100, "The site is underlain by stiff clay over sandstone.")]])
    monkeypatch.setattr(words, "search", lambda *a, **k: [])
    cites = {"geology": [Citation(page=0, quote="underlain by stiff clay "
                                                 "over sandstone")]}
    got = place_citations(words, cites, [0])
    quote, = got.items
    assert (quote.status, quote.anchor, quote.bbox, quote.page_level) == (
        PLACED, "quote on its page", None, True)
