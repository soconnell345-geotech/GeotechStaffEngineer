"""BACKLOG RI-149 and RI-150: grading sheets scanned sideways.

RI-149: a grading sheet whose legend prints several specimens, each on its
own row (a label, a description, its percentages), scanned sideways onto a
portrait page, was read with each specimen's values attached to another
specimen. Listed top to bottom as displayed, a turned page's lines come in
the order each STARTS along the turned direction, which follows its length,
not its row; the picture went sideways; and the merge let a model's reading
off the wrong row overrule the floor.

RI-150: one hole's grading sheets were grouped with the summary table before
them, so no lab test held them.

Every sheet here is made up: the holes, depths and percentages are not any
report's (``lab_fixtures.TURNED_LEGEND``).
"""

from __future__ import annotations

import io
import re
from types import SimpleNamespace

import pytest

from report_ingest import lab_floor as LF
from report_ingest.floor import MergeLog
from report_ingest.lab_reader import (
    LabSheetReading, ReadProv, ReadTest, lab_parts, read_lab_sheet,
    serialise_page, split_lab_items,
)
from report_ingest.model import GradationResult, LabTest, Provenance, Quantity
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.lab_fixtures import (
    TURNED_HOLE, TURNED_LEGEND, build_portrait_form, build_turned_run,
    build_turned_sheets, build_upright_legend_sheet,
)
from report_ingest.turned import (
    TURN_MARGIN, TURNED_SHARE, Turn, displayed_box, page_turn, reading_order,
    rows, top_band, turn_of, turn_vote, upright_box, upright_png,
    upright_size,
)

pymupdf = pytest.importorskip("pymupdf",
                              reason="the fixtures are drawn with PyMuPDF")
Image = pytest.importorskip("PIL.Image", reason="pictures are turned by PIL")

#: The made-up run: page 0 a summary table, then one grading sheet for each
#: skew (a scan a degree off square either way, and one square).
SKEWS = (-1.0, 0.0, 1.5)
SHEETS = (1, 2, 3)


def _open(pdf: bytes, tmp_path, name: str = "sheet.pdf"):
    from planlens.document import open_document
    path = tmp_path / name
    path.write_bytes(pdf)
    return open_document(str(path))


@pytest.fixture
def turned(tmp_path):
    pdf = build_turned_run(SKEWS)
    doc = _open(pdf, tmp_path)
    yield doc, pdf
    doc.close()


@pytest.fixture
def upright(tmp_path):
    doc = _open(build_upright_legend_sheet(), tmp_path)
    yield doc
    doc.close()


#: The three quarter turns, one made-up sheet each, a little skewed.
QUARTERS = (90, 180, 270)


@pytest.fixture
def quarters(tmp_path):
    doc = _open(build_turned_sheets(QUARTERS, skew=-0.8), tmp_path,
                "quarters.pdf")
    yield doc
    doc.close()


def _expected():
    return [(label, values) for label, _words, _printed, values
            in TURNED_LEGEND]


def _label_of(text: str) -> str:
    return next((label for label, *_rest in TURNED_LEGEND
                 if text.strip() == label), "")


def _row_of_values(text: str) -> str:
    """The label of the legend row whose values a line prints."""
    return next((label for label, _w, printed, _v in TURNED_LEGEND
                 if text.strip() == printed), "")


def _size(png: bytes):
    return Image.open(io.BytesIO(png)).size


# ---------------------------------------------------------------------------
# the page's turn, its rows and its reading order
# ---------------------------------------------------------------------------

class TestTheTurn:
    def test_a_sheet_scanned_sideways_reads_a_quarter_turn_with_its_skew(
            self, turned):
        doc, _pdf = turned
        for page, skew in zip(SHEETS, SKEWS):
            turn = turn_of(doc, page)
            assert turn.quarter == 90
            assert turn.angle == pytest.approx(90.0 + skew, abs=0.4)

    def test_an_upright_sheet_is_upright(self, upright):
        assert turn_of(upright, 0) == Turn(0, 0.0, 612.0, 792.0)

    def test_tick_boxes_and_short_marks_do_not_outvote_the_words(self):
        line = SimpleNamespace
        words = [line(text="GRAIN SIZE DISTRIBUTION", rotation=89.0,
                      bbox=(50, 300, 70, 560))]
        marks = [line(text="[ ]", rotation=0.0, bbox=(10, 10 + 12 * n, 18,
                                                      18 + 12 * n))
                 for n in range(40)]
        assert page_turn(words + marks).quarter == 90
        # A page of upright words with one turned axis title stays upright.
        flat = [line(text="Liquid limit and plastic limit", rotation=0.0,
                     bbox=(60, 100 + 14 * n, 300, 110 + 14 * n))
                for n in range(5)]
        assert page_turn(flat + [line(text="WATER CONTENT", rotation=90.0,
                                      bbox=(20, 300, 30, 400))]).quarter == 0

    def test_a_line_with_no_direction_does_not_vote(self):
        line = SimpleNamespace(text="SAND =40%", rotation=None,
                               bbox=(0, 0, 10, 10))
        assert page_turn([line]).quarter == 0

    @pytest.mark.parametrize("page,quarter", list(enumerate(QUARTERS)))
    def test_every_quarter_turn_is_read_with_its_skew(self, quarters, page,
                                                      quarter):
        turn = turn_of(quarters, page)
        assert turn.quarter == quarter
        assert turn.angle == pytest.approx(quarter - 0.8, abs=0.4)

    def test_a_portrait_form_with_a_landscape_chart_stays_upright(
            self, tmp_path):
        """A form printing its header upright over a chart printed turned
        splits its letters about evenly: more than half run turned, but not
        clearly more than run upright, so the page is upright."""
        doc = _open(build_portrait_form(), tmp_path, "form.pdf")
        try:
            vote = turn_vote(doc.page(0).lines)
            assert vote.share >= TURNED_SHARE
            assert vote.margin < TURN_MARGIN
            assert turn_of(doc, 0).quarter == 0
        finally:
            doc.close()

    def test_a_sheet_scanned_sideways_beats_its_stray_upright_marks(self):
        line = SimpleNamespace
        words = [line(text="SAND =47%, Gravel =21%, Silt =22%", rotation=89.0,
                      bbox=(300 + 13 * n, 200, 310 + 13 * n, 420))
                 for n in range(4)]
        stamp = [line(text="RECEIVED MADE-UP LAB", rotation=0.0,
                      bbox=(40, 40, 200, 50))]
        vote = turn_vote(words + stamp)
        assert vote.margin >= TURN_MARGIN
        assert page_turn(words + stamp).quarter == 90
        # Turned letters only a little more than the upright ones: upright.
        even = [line(text="SAND =47%, Gravel =21%, Silt =22%", rotation=0.0,
                     bbox=(40, 60 + 12 * n, 300, 70 + 12 * n))
                for n in range(3)]
        assert page_turn(words + even).quarter == 0

    def test_an_upright_page_keeps_the_order_every_reader_used(self, upright):
        content = upright.page(0)
        old = sorted(content.lines, key=lambda ln: (round(ln.bbox[1], 1),
                                                    ln.bbox[0]))
        assert reading_order(content.lines, Turn()) == old


class TestTheRows:
    def test_displayed_order_scatters_the_legend(self, turned):
        """The fault itself: top to bottom as displayed, the values lines
        come in the order of their lengths, not of the labels."""
        doc, _pdf = turned
        content = doc.page(1)
        shown = sorted(content.lines, key=lambda ln: (round(ln.bbox[1], 1),
                                                      ln.bbox[0]))
        labels = [_label_of(ln.text) for ln in shown if _label_of(ln.text)]
        values = [_row_of_values(ln.text) for ln in shown
                  if _row_of_values(ln.text)]
        assert values != labels

    @pytest.mark.parametrize("page", SHEETS)
    def test_each_label_is_followed_by_its_own_values(self, turned, page):
        doc, _pdf = turned
        content = doc.page(page)
        order = reading_order(content.lines, turn_of(doc, page))
        texts = [ln.text.strip() for ln in order]
        for label, words, printed, _values in TURNED_LEGEND:
            at = texts.index(label)
            assert texts[at + 1:at + 3] == [words, printed]

    def test_the_rows_are_taken_across_the_texts_own_direction(self,
                                                               tmp_path):
        """A skew drifts a long row's far end across the rows: taken square
        to the page, a legend row two and a half degrees off square does not
        hold together; taken across the text's own direction, it does."""
        doc = _open(build_turned_run((-2.5,)), tmp_path, "skewed.pdf")
        content = doc.page(1)
        turn = turn_of(doc, 1)
        assert turn.angle == pytest.approx(87.5, abs=0.4)
        square = Turn(90, 90.0, turn.width, turn.height)

        def together(found):
            ok = 0
            for row in found:
                texts = {ln.text.strip() for ln in row}
                ok += sum(1 for label, words, printed, _v in TURNED_LEGEND
                          if {label, words, printed} <= texts)
            return ok

        try:
            assert together(rows(content.lines, turn)) == len(TURNED_LEGEND)
            assert together(rows(content.lines, square)) < len(TURNED_LEGEND)
            assert _read(LF.legend_specimens(doc, 1)) == _expected()
        finally:
            doc.close()

    def test_the_top_band_is_the_turned_sheets_top(self, turned):
        doc, _pdf = turned
        content = doc.page(1)
        band = [ln.text for ln in top_band(content.lines, turn_of(doc, 1),
                                          LF.TITLE_BAND)]
        assert "GRAIN SIZE DISTRIBUTION" in band
        assert not any(_label_of(t) or _row_of_values(t) for t in band)

    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_every_quarter_reads_row_by_row_with_its_title_on_top(
            self, quarters, page):
        content = quarters.page(page)
        turn = turn_of(quarters, page)
        texts = [ln.text.strip() for ln in reading_order(content.lines, turn)]
        for label, words, printed, _values in TURNED_LEGEND:
            at = texts.index(label)
            assert texts[at + 1:at + 3] == [words, printed]
        band = [ln.text for ln in top_band(content.lines, turn,
                                          LF.TITLE_BAND)]
        assert band[0] == "GRAIN SIZE DISTRIBUTION"
        assert not any(_label_of(t) or _row_of_values(t) for t in band)
        assert LF.page_title_kinds(quarters, page) == ("gradation",)


class TestThePicture:
    def test_a_turned_picture_is_set_upright(self):
        image = Image.new("RGB", (20, 10), "white")
        image.putpixel((0, 0), (255, 0, 0))
        out = io.BytesIO()
        image.save(out, format="PNG")
        png = out.getvalue()
        shown = Image.open(io.BytesIO(upright_png(png, Turn(90, 89.0))))
        # A quarter clockwise: 20x10 becomes 10x20, the top-left corner the
        # top-right.
        assert shown.size == (10, 20)
        assert shown.getpixel((9, 0)) == (255, 0, 0)

    def test_an_upright_picture_is_sent_as_it_is(self):
        assert upright_png(b"not a picture", Turn()) == b"not a picture"
        # One that will not open is sent as it is too.
        assert upright_png(b"not a picture", Turn(90, 90.0)) == \
            b"not a picture"

    def test_a_box_goes_where_the_picture_turns_it(self):
        turn = Turn(90, 90.0, 612.0, 792.0)
        # The displayed page's bottom-left corner is the upright one's
        # top-left.
        assert upright_box((0.0, 782.0, 10.0, 792.0), turn) == \
            (0.0, 0.0, 10.0, 10.0)

    @pytest.mark.parametrize("quarter,corner", [
        (90, (0.0, 782.0, 10.0, 792.0)), (180, (602.0, 782.0, 612.0, 792.0)),
        (270, (602.0, 0.0, 612.0, 10.0))])
    def test_each_quarter_takes_the_sheets_top_left_corner_there(
            self, quarter, corner):
        turn = Turn(quarter, quarter - 0.8, 612.0, 792.0)
        assert upright_box(corner, turn) == (0.0, 0.0, 10.0, 10.0)
        assert displayed_box((0.0, 0.0, 10.0, 10.0), turn) == corner

    @pytest.mark.parametrize("quarter", QUARTERS)
    def test_a_box_comes_back_to_the_displayed_frame(self, quarter):
        turn = Turn(quarter, quarter + 1.2, 612.0, 792.0)
        for box in ((40.0, 300.0, 140.0, 312.0), (0.0, 0.0, 612.0, 792.0),
                    (500.5, 20.25, 611.0, 791.0)):
            shown = upright_box(box, turn)
            width, height = upright_size(turn)
            assert 0.0 <= shown[0] <= shown[2] <= width
            assert 0.0 <= shown[1] <= shown[3] <= height
            assert displayed_box(shown, turn) == pytest.approx(box)
        assert upright_size(turn) == ((612.0, 792.0) if quarter == 180
                                      else (792.0, 612.0))

    def test_the_upright_picture_shows_a_box_where_upright_box_puts_it(
            self, tmp_path):
        """A red patch drawn on each turned sheet is, in the picture turned
        upright, where :func:`upright_box` says."""
        made = pymupdf.open(stream=build_turned_sheets(QUARTERS),
                            filetype="pdf")
        patch = (100.0, 300.0, 140.0, 330.0)
        for page in made:
            page.draw_rect(pymupdf.Rect(*patch), color=(1, 0, 0),
                           fill=(1, 0, 0))
        doc = _open(made.tobytes(), tmp_path, "patched.pdf")
        made.close()
        try:
            for page, quarter in enumerate(QUARTERS):
                turn = turn_of(doc, page)
                assert turn.quarter == quarter
                png, _info = doc.render(page, dpi=72)
                shown = Image.open(io.BytesIO(upright_png(png, turn)))
                shown = shown.convert("RGB")
                assert shown.size == tuple(int(round(v))
                                           for v in upright_size(turn))
                x0, y0, x1, y1 = upright_box(patch, turn)
                red = shown.getpixel((int((x0 + x1) / 2),
                                      int((y0 + y1) / 2)))
                assert red[0] > 200 and red[1] < 60 and red[2] < 60
        finally:
            doc.close()


# ---------------------------------------------------------------------------
# the floor reads the legend
# ---------------------------------------------------------------------------

def _read(rows_read):
    """What a legend reading says: each entry's label and its values."""
    return [(f"{r.link.depth_top:.1f}-{r.link.depth_bottom:.1f}",
             {k: v[0] for k, v in r.values.items()}) for r in rows_read]


class TestTheLegend:
    @pytest.mark.parametrize("page", SHEETS)
    def test_each_specimen_gets_its_own_rows_values(self, turned, page):
        doc, _pdf = turned
        got = LF.legend_specimens(doc, page)
        assert _read(got) == _expected()
        assert all(r.link.investigation_id == TURNED_HOLE for r in got)

    def test_an_upright_legend_reads_the_same(self, upright):
        assert _read(LF.legend_specimens(upright, 0)) == _expected()

    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_every_quarter_gives_each_specimen_its_own_rows_values(
            self, quarters, page):
        got = LF.legend_specimens(quarters, page)
        assert _read(got) == _expected()
        assert all(r.link.investigation_id == TURNED_HOLE for r in got)

    def _coded(self, tmp_path, codes, hole=TURNED_HOLE):
        from report_ingest.tests.lab_fixtures import build_turned_legend_page
        made = pymupdf.open()
        build_turned_legend_page(made, skew=-0.8, hole=hole,
                                 label_codes=codes)
        doc = _open(made.tobytes(), tmp_path, "coded.pdf")
        made.close()
        return doc

    def test_the_sheets_own_hole_before_a_label_is_no_sample(self, tmp_path):
        doc = self._coded(tmp_path, [TURNED_HOLE] * len(TURNED_LEGEND))
        try:
            got = LF.legend_specimens(doc, 0)
            assert _read(got) == _expected()
            assert all(r.link.investigation_id == TURNED_HOLE
                       and not r.link.sample_id for r in got)
        finally:
            doc.close()

    def test_another_hole_before_a_label_is_that_rows_hole(self, tmp_path):
        doc = self._coded(tmp_path, ["QX-5"] * len(TURNED_LEGEND))
        try:
            got = LF.legend_specimens(doc, 0)
            assert [r.link.investigation_id for r in got] == \
                ["QX-5"] * len(TURNED_LEGEND)
            assert not any(r.link.sample_id for r in got)
            # Merged with the model's reading of those specimens under
            # QX-5: one test each, not each twice.
            floor = LF.floor_from_tables(doc, [0], "RX")
            model = [_model_test(depth, dict(values), 0, None, hole="QX-5",
                                 note="")
                     for (_label, values), depth in zip(
                         _expected(), (2.0, 5.0, 8.0, 11.0))]
            merged, _log = LF.merge_lab_tests(floor.tests, model, MergeLog(),
                                              title_kind=floor.kind)
            grading = [t for t in merged if t.kind == "gradation"]
            assert len(grading) == len(TURNED_LEGEND)
            assert {t.investigation_id for t in grading} == {"QX-5"}
        finally:
            doc.close()

    def test_a_samples_code_before_a_label_is_that_rows_sample(self,
                                                               tmp_path):
        codes = [f"S-{n + 1}" for n in range(len(TURNED_LEGEND))]
        doc = self._coded(tmp_path, codes)
        try:
            got = LF.legend_specimens(doc, 0)
            assert [r.link.sample_id for r in got] == codes
            assert all(r.link.investigation_id == TURNED_HOLE for r in got)
        finally:
            doc.close()

    def test_the_values_box_is_the_line_that_prints_them(self, turned):
        doc, _pdf = turned
        content = doc.page(1)
        for row in LF.legend_specimens(doc, 1):
            inside = [ln.text.strip() for ln in content.lines
                      if ln.bbox[0] >= row.values_bbox[0] - 0.01
                      and ln.bbox[2] <= row.values_bbox[2] + 0.01
                      and ln.bbox[1] >= row.values_bbox[1] - 0.01
                      and ln.bbox[3] <= row.values_bbox[3] + 0.01]
            label = f"{row.link.depth_top:.1f}-{row.link.depth_bottom:.1f}"
            assert [_row_of_values(t) for t in inside
                    if _row_of_values(t)] == [label]

    def test_a_summary_rows_depths_without_percentages_are_no_legend(
            self, turned):
        doc, _pdf = turned
        assert LF.legend_specimens(doc, 0) == []

    def _page(self, tmp_path, entries, extra=()):
        """An upright made-up sheet: ``entries`` of (y, label) and
        (y, values) placed by hand, and ``extra`` (x, y, text)."""
        from report_ingest.tests.lab_fixtures import LETTER, _text
        doc = pymupdf.open()
        page = doc.new_page(width=LETTER[0], height=LETTER[1])
        _text(page, 150, 60, "GRAIN SIZE DISTRIBUTION", 14, bold=True)
        _text(page, 60, 100, f"Bore hole: {TURNED_HOLE}")
        for y, label, values in entries:
            if label:
                _text(page, 60, y, label, 8)
            if values:
                _text(page, 330, y, values, 8)
        for x, y, text in extra:
            _text(page, x, y, text, 8)
        pdf = doc.tobytes()
        doc.close()
        return _open(pdf, tmp_path, "hand.pdf")

    def test_one_specimen_is_not_a_legend(self, tmp_path):
        doc = self._page(tmp_path, [(600, "2.0-3.3", None),
                                    (600, None, "SAND =70%, Silt =30%")])
        assert LF.legend_specimens(doc, 0) == []

    def test_values_half_a_row_from_every_label_are_not_bound(self, tmp_path):
        doc = self._page(tmp_path, [
            (600, "2.0-3.3", None), (614, "5.0-6.3", None),
            (628, "8.0-9.3", None),
            (607, None, "SAND =70%, Silt =30%"),
            (621, None, "SAND =55%, Clay =45%")])
        assert LF.legend_specimens(doc, 0) == []

    def test_a_keys_symbols_read_as_a_range_are_not_a_label(self, tmp_path):
        """An optical reader reads a dashed key with its markers as a range
        ("-6-9-"); it stands in another column than the labels."""
        doc = self._page(
            tmp_path,
            [(600, "2.0-3.3", "Silt =30%, SAND =62%, Clay =8%"),
             (614, "5.0-6.3", "SAND =47%, Gravel =21%, Silt =22%, Clay =10%"),
             (628, "8.0-9.3", "Clay =35%, SILT =55%, Sand =10%")],
            extra=[(250, 614, "-6-9-")])
        got = LF.legend_specimens(doc, 0)
        assert [r.link.depth_top for r in got] == [2.0, 5.0, 8.0]

    def test_values_running_two_rows_together_refuse_the_page(self, tmp_path):
        doc = self._page(tmp_path, [
            (600, "2.0-3.3", "SAND =70%, Silt =30%"),
            (614, "5.0-6.3", "SAND =55%, Clay =45%, SAND =60%")])
        assert LF.legend_specimens(doc, 0) == []


class TestTheFloor:
    @pytest.mark.parametrize("page", SHEETS)
    def test_the_turned_sheets_title_and_specimens(self, turned, page):
        doc, _pdf = turned
        sheet = LF.floor_from_tables(doc, [page], "RX")
        assert sheet.kind == "gradation"
        legend = [t for t in sheet.tests
                  if any(p.note == LF.LEGEND_NOTE for p in t.prov)]
        assert [t.depth_top.value for t in legend] == [2.0, 5.0, 8.0, 11.0]
        for test, (_label, values) in zip(legend, _expected()):
            assert test.kind == "gradation"
            assert test.investigation_id == TURNED_HOLE
            assert {k: getattr(test.result, k) for k in values} == values
            # Nothing computed: no fines from silt and clay.
            assert test.result.fines_percent is None

    def test_the_summary_title_names_both_kinds(self, turned):
        doc, _pdf = turned
        assert set(LF.page_title_kinds(doc, 0)) == {"atterberg", "gradation"}
        assert LF.page_title_kinds(doc, 1) == ("gradation",)

    def test_the_serialised_floor_says_which_row_it_read(self, turned):
        doc, _pdf = turned
        text = LF.serialise_floor(LF.floor_from_tables(doc, [1], "RX"))
        assert text.count("read off one legend row") == len(TURNED_LEGEND)

    def test_a_hole_named_in_its_printed_field_is_that_hole(self):
        def test(hole):
            return LabTest(kind="gradation", investigation_id=hole,
                           depth_top=Quantity(value=5.0, unit="m"))
        assert LF._same_specimen(test(f"BORE HOLE NO. ({TURNED_HOLE})"),
                                 test(TURNED_HOLE))
        assert not LF._same_specimen(test("BORE HOLE NO. (QX-5)"),
                                     test(TURNED_HOLE))

    def test_a_label_printing_no_unit_meets_a_depth_in_feet(self, turned):
        doc, _pdf = turned
        legend = LF.floor_from_tables(doc, [1], "RX").tests
        five = next(t for t in legend if t.depth_top.value == 5.0)
        assert five.depth_top.unit == ""
        read = LabTest(kind="gradation", investigation_id=TURNED_HOLE,
                       depth_top=Quantity(value=5.0, unit="ft"))
        assert LF._same_specimen(five, read)
        other = read.model_copy(update={
            "depth_top": Quantity(value=8.0, unit="ft")})
        assert not LF._same_specimen(five, other)


# ---------------------------------------------------------------------------
# the merge: a legend row's value stands against a reading off another row
# ---------------------------------------------------------------------------

def _values_box(doc, page, label):
    for row in LF.legend_specimens(doc, page):
        if f"{row.link.depth_top:.1f}-{row.link.depth_bottom:.1f}" == label:
            return tuple(row.values_bbox)
    raise AssertionError(label)


def _label_box(doc, page, label):
    for row in LF.legend_specimens(doc, page):
        if f"{row.link.depth_top:.1f}-{row.link.depth_bottom:.1f}" == label:
            return tuple(row.label_bbox)
    raise AssertionError(label)


def _model_test(depth, values, page, bbox, note="read off the legend",
                hole=TURNED_HOLE, method="model"):
    return LabTest(
        kind="gradation", investigation_id=hole,
        depth_top=Quantity(value=depth, unit="m"), pages=[page],
        result=GradationResult(**values),
        prov=[Provenance(page=page, bbox=bbox, method=method,
                         confidence=0.9, note=note)])


def _models(merged):
    return [t for t in merged if t.prov and t.prov[0].method
            in ("model", "model_from_picture")]


class TestTheMerge:
    def test_a_reading_off_another_row_does_not_overrule_the_row(self,
                                                                  turned):
        """The model gave the 2.0 m specimen the 5.0 m row's values, boxed
        on the 5.0 m row: the 2.0 m row's own values stand, the model's
        beside them, and the gravel the 2.0 m row does not print is set
        aside."""
        doc, _pdf = turned
        floor = LF.floor_from_tables(doc, [1], "RX")
        wrong = dict(TURNED_LEGEND[1][3])
        model = [_model_test(2.0, wrong, 1, _values_box(doc, 1, "5.0-6.3"))]
        merged, log = LF.merge_lab_tests(floor.tests, model, MergeLog(),
                                         title_kind=floor.kind)
        two = next(t for t in merged if t.depth_top.value == 2.0
                   and t.prov and t.prov[0].method == "model")
        mine = TURNED_LEGEND[0][3]
        assert {k: getattr(two.result, k) for k in mine} == mine
        assert two.result.gravel_percent is None
        alts = {a.field: float(a.value) for a in two.prov[0].alternatives}
        assert alts == wrong
        whys = [d["why"] for d in log.disagreements]
        assert any(w == LF._SWAP_WHY for w in whys)
        assert any(w == LF._LEGEND_ASIDE_WHY for w in whys)

    def _merge(self, doc, model, page=1):
        floor = LF.floor_from_tables(doc, [page], "RX")
        merged, log = LF.merge_lab_tests(floor.tests, model, MergeLog(),
                                         title_kind=floor.kind)
        return _models(merged), log

    @pytest.mark.parametrize("on", ["values", "label"])
    def test_a_reading_boxed_on_the_rows_own_line_may_correct_it(
            self, turned, on):
        """Boxed on the line printing the specimen's values, or on the line
        printing its label (the box the reader is asked for), a reading
        with a note corrects the row, where no other row prints it."""
        doc, _pdf = turned
        box = (_values_box if on == "values" else _label_box)(doc, 1,
                                                               "2.0-3.3")
        own = dict(TURNED_LEGEND[0][3], sand_percent=61.0)
        (two,), log = self._merge(doc, [_model_test(
            2.0, own, 1, box, note="SAND =61% on the 2.0-3.3 row")])
        assert two.result.sand_percent == 61.0
        assert two.result.silt_percent == 30.0
        assert not any(d["why"] in (LF._LEGEND_WHY, LF._SWAP_WHY)
                       for d in log.disagreements)

    def test_a_reading_boxed_off_its_row_does_not_correct_it(self, turned):
        """Boxed on neither of its row's lines, the row's value stands; but
        nothing else is set aside where no other row prints the model's
        value: the swap is not shown."""
        doc, _pdf = turned
        own = dict(TURNED_LEGEND[0][3], sand_percent=61.0, gravel_percent=4.0)
        (two,), log = self._merge(doc, [_model_test(
            2.0, own, 1, (40.0, 40.0, 90.0, 52.0), note="SAND =61%")])
        assert two.result.sand_percent == 62.0
        assert two.result.gravel_percent == 4.0
        whys = [d["why"] for d in log.disagreements]
        assert LF._LEGEND_WHY in whys
        assert LF._LEGEND_ASIDE_WHY not in whys

    def test_another_rows_value_boxed_on_the_label_is_still_overruled(
            self, turned):
        """The swap signature: the model's sand is the one the 5.0 m row
        prints, so it was read off that row though it is boxed on the 2.0 m
        label; the row's own stands, and the gravel the 5.0 m row prints
        (and the 2.0 m row does not) is set aside."""
        doc, _pdf = turned
        wrong = dict(TURNED_LEGEND[0][3], sand_percent=47.0,
                     gravel_percent=21.0)
        (two,), log = self._merge(doc, [_model_test(
            2.0, wrong, 1, _label_box(doc, 1, "2.0-3.3"), note="the row")])
        assert two.result.sand_percent == 62.0
        assert two.result.gravel_percent is None
        whys = [d["why"] for d in log.disagreements]
        assert LF._SWAP_WHY in whys and LF._LEGEND_ASIDE_WHY in whys

    def test_another_rows_value_under_another_name_is_a_swap_too(self,
                                                                 turned):
        """Each row prints its shares in its own order, so a reader that
        took the 5.0 m row's values may have named them otherwise: the
        2.0 m specimen's sand given as the 5.0 m row's silt is that row's."""
        doc, _pdf = turned
        wrong = dict(TURNED_LEGEND[0][3], sand_percent=22.0)
        (two,), log = self._merge(doc, [_model_test(
            2.0, wrong, 1, _label_box(doc, 1, "2.0-3.3"), note="the row")])
        assert two.result.sand_percent == 62.0
        assert LF._SWAP_WHY in [d["why"] for d in log.disagreements]

    def test_a_value_the_sheet_prints_elsewhere_survives_a_swap(self,
                                                               turned):
        """A swapped share is overruled; a diameter no legend row prints (the
        sheet gives it off the curve, where the floor reads nothing) stays
        the model's."""
        doc, _pdf = turned
        wrong = dict(TURNED_LEGEND[0][3], sand_percent=47.0,
                     d60=Quantity(value=0.37, unit="mm"))
        (two,), _log = self._merge(doc, [_model_test(
            2.0, wrong, 1, _label_box(doc, 1, "2.0-3.3"), note="the row")])
        assert two.result.sand_percent == 62.0
        assert two.result.d60 is not None and two.result.d60.value == 0.37

    def test_the_model_wins_where_the_row_pins_no_one_specimen(self, turned):
        """Two of the model's tests at 2.0 m: the row pins neither, so the
        model's reading stands and nothing of it is set aside, though its
        sand is another row's."""
        doc, _pdf = turned
        box = _label_box(doc, 1, "2.0-3.3")
        first = dict(TURNED_LEGEND[0][3], sand_percent=47.0,
                     gravel_percent=21.0)
        second = dict(TURNED_LEGEND[0][3])
        tests, log = self._merge(doc, [
            _model_test(2.0, first, 1, box, note="the row"),
            _model_test(2.0, second, 1, box, note="the row")])
        assert tests[0].result.sand_percent == 47.0
        assert tests[0].result.gravel_percent == 21.0
        assert LF._LEGEND_ASIDE_WHY not in [d["why"]
                                            for d in log.disagreements]

    def test_a_picture_read_with_a_note_keeps_its_evidence(self, turned):
        """A value read off the picture (no box) with a note saying what it
        shows corrects the row, as it corrects any floor value."""
        doc, _pdf = turned
        own = dict(TURNED_LEGEND[0][3], sand_percent=61.0)
        (two,), _log = self._merge(doc, [_model_test(
            2.0, own, 1, None, note="the scan's 62 reads 61 on the picture",
            method="model_from_picture")])
        assert two.result.sand_percent == 61.0

    @pytest.mark.parametrize("on,kept", [("label", 3.2), ("values", 3.3)])
    def test_a_rows_label_is_corrected_only_on_its_labels_line(
            self, turned, on, kept):
        doc, _pdf = turned
        box = (_values_box if on == "values" else _label_box)(doc, 1,
                                                               "2.0-3.3")
        test = _model_test(2.0, dict(TURNED_LEGEND[0][3]), 1, box,
                           note="the label reads 2.0-3.2")
        test.depth_bottom = Quantity(value=3.2, unit="m")
        (two,), _log = self._merge(doc, [test])
        assert two.depth_bottom.value == kept

    def test_a_reading_that_agrees_sets_nothing_aside(self, turned):
        doc, _pdf = turned
        floor = LF.floor_from_tables(doc, [1], "RX")
        model = [_model_test(depth, dict(values), 1,
                             _values_box(doc, 1, label))
                 for (label, values), depth in zip(_expected(),
                                                   (2.0, 5.0, 8.0, 11.0))]
        merged, log = LF.merge_lab_tests(floor.tests, model, MergeLog(),
                                         title_kind=floor.kind)
        assert not log.disagreements
        model_tests = [t for t in merged if t.prov[0].method == "model"]
        assert len(model_tests) == len(TURNED_LEGEND)
        for test, (_label, values) in zip(model_tests, _expected()):
            assert {k: getattr(test.result, k) for k in values} == values

    def test_the_reader_end_to_end(self, turned):
        """The lab reader on the turned sheet: a scripted model that pairs
        the rows wrongly (each specimen given the next row's values) is
        overruled row by row, and the picture it was shown is upright."""
        doc, _pdf = turned
        turn = turn_of(doc, 1)
        tests = []
        for n, (label, _w, _p, _values) in enumerate(TURNED_LEGEND):
            top = float(label.split("-")[0])
            nxt = TURNED_LEGEND[(n + 1) % len(TURNED_LEGEND)]
            # The model copies the box off the listing, which gives a turned
            # page's boxes in its upright picture's frame.
            box = upright_box(_values_box(doc, 1, nxt[0]), turn)
            tests.append(ReadTest(
                kind="gradation", investigation_id=TURNED_HOLE,
                depth_top=top, prov=ReadProv(page=1, bbox=box,
                                             note="the legend row"),
                **nxt[3]))
        engine = FakeEngine([{"final": LabSheetReading(
            depth_unit="m", tests=tests, pages_read=[1])}])
        result = read_lab_sheet(doc, [1], engine, hint_kind="gradation")
        got = {t.depth_top.value: t for t in result.tests
               if t.kind == "gradation" and t.result is not None
               and t.depth_top is not None}
        for n, (label, _w, _p, values) in enumerate(TURNED_LEGEND):
            test = got[float(label.split("-")[0])]
            assert {k: getattr(test.result, k) for k in values} == values
            for name in ("gravel_percent", "sand_percent", "silt_percent",
                         "clay_percent"):
                if name not in values:
                    assert getattr(test.result, name) is None
            # Its box is back in the page's frame as displayed.
            nxt = TURNED_LEGEND[(n + 1) % len(TURNED_LEGEND)]
            model_prov = next(p for p in test.prov if p.method == "model")
            assert model_prov.bbox == pytest.approx(
                _values_box(doc, 1, nxt[0]))
        (blocks,) = [engine.calls[0]["image_blocks"]]
        width, height = _size(blocks[0]["png"])
        assert width > height

    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_every_quarter_end_to_end(self, quarters, page):
        """Each specimen given the next row's values, boxed on that row's
        values line in the listing's frame: overruled row by row at every
        quarter turn, its box back as displayed."""
        turn = turn_of(quarters, page)
        tests = []
        for n, (label, _w, _p, _values) in enumerate(TURNED_LEGEND):
            nxt = TURNED_LEGEND[(n + 1) % len(TURNED_LEGEND)]
            tests.append(ReadTest(
                kind="gradation", investigation_id=TURNED_HOLE,
                depth_top=float(label.split("-")[0]),
                prov=ReadProv(page=page, bbox=upright_box(
                    _values_box(quarters, page, nxt[0]), turn),
                    note="the legend row"), **nxt[3]))
        engine = FakeEngine([{"final": LabSheetReading(
            depth_unit="m", tests=tests, pages_read=[page])}])
        result = read_lab_sheet(quarters, [page], engine,
                                hint_kind="gradation")
        got = {t.depth_top.value: t for t in result.tests
               if t.kind == "gradation" and t.result is not None
               and t.depth_top is not None}
        for label, _w, _p, values in TURNED_LEGEND:
            test = got[float(label.split("-")[0])]
            assert {k: getattr(test.result, k) for k in values} == values
        width, height = _size(engine.calls[0]["image_blocks"][0]["png"])
        assert (width > height) == (turn.quarter in (90, 270))


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

class TestWhatTheModelIsShown:
    def test_the_turned_pages_lines_are_listed_row_by_row(self, turned):
        doc, _pdf = turned
        text = serialise_page(doc, 1)
        assert "THIS PAGE'S TEXT RUNS TURNED" in text
        listed = [line.split("| ", 1)[1].strip() for line in
                  text.splitlines() if "| " in line
                  and not line.strip().startswith("x0,y0")]
        for label, words, printed, _values in TURNED_LEGEND:
            at = listed.index(label)
            assert listed[at + 1:at + 3] == [words, printed]

    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_a_turned_pages_boxes_are_in_its_upright_pictures_frame(
            self, quarters, page):
        """Listed in the frame of the picture the model sees: a row's lines
        lie side by side along x, one under another down the rows, and each
        box is the displayed one turned (:func:`upright_box`)."""
        turn = turn_of(quarters, page)
        text = serialise_page(quarters, page)
        width, height = upright_size(turn)
        assert f"{width:.0f} wide and {height:.0f} high" in text
        boxes = {}
        for line in text.splitlines():
            if "| " not in line or line.strip().startswith("x0,y0"):
                continue
            head, words = line.split("| ", 1)
            boxes[words.strip()] = tuple(
                float(v) for v in head.split()[0].split(","))
            assert "[turned]" not in head
        for label, words, printed, _values in TURNED_LEGEND:
            assert boxes[label][2] <= boxes[words][0] + 1
            assert boxes[words][2] <= boxes[printed][0] + 1
        content = quarters.page(page)
        title = next(ln for ln in content.lines
                     if ln.text.strip() == "GRAIN SIZE DISTRIBUTION")
        assert boxes["GRAIN SIZE DISTRIBUTION"] == pytest.approx(
            upright_box(title.bbox, turn), abs=0.6)

    def test_the_floor_shows_a_turned_pages_boxes_in_the_same_frame(
            self, turned):
        doc, _pdf = turned
        floor = LF.floor_from_tables(doc, [1], "RX")
        turn = turn_of(doc, 1)
        plain = LF.serialise_floor(floor)
        shown = LF.serialise_floor(floor, turns={1: turn})
        box = upright_box(_values_box(doc, 1, "2.0-3.3"), turn)
        assert ",".join(f"{v:.0f}" for v in box) in shown
        assert ",".join(f"{v:.0f}" for v in box) not in plain
        # An upright page's floor reads as it always did.
        assert LF.serialise_floor(floor, turns={1: Turn()}) == plain

    def test_an_upright_page_is_listed_as_before(self, upright):
        text = serialise_page(upright, 0)
        assert "TURNED" not in text
        content = upright.page(0)
        old = sorted(content.lines, key=lambda ln: (round(ln.bbox[1], 1),
                                                    ln.bbox[0]))
        listed = [line.split("| ", 1)[1] for line in text.splitlines()
                  if "| " in line and not line.strip().startswith("x0,y0")]
        assert [t.strip() for t in listed] == [ln.text.strip() for ln in old]

    def test_a_cut_pdf_turns_its_turned_pages_upright(self, turned):
        from report_ingest.model_first_pages import cut_pdf
        _doc, pdf = turned
        data, n, _sanitized = cut_pdf(pdf, [0, 1], turns={1: 90})
        with pymupdf.open(stream=data, filetype="pdf") as cut:
            assert n == 2
            assert [p.rotation for p in cut] == [0, 90]
        data, _n, _s = cut_pdf(pdf, [1])
        with pymupdf.open(stream=data, filetype="pdf") as cut:
            assert cut[0].rotation == 0

    def test_the_model_first_pdf_call_sends_turned_pages_upright(self,
                                                                 turned):
        from report_ingest.model_first_read import _Caller
        doc, pdf = turned
        engine = FakeEngine([])
        engine.accepts_pdf = True
        caller = _Caller("lab", doc, engine, "RX", "pdf", pdf)
        blocks, problem = caller.blocks([0, 1])
        assert problem == ""
        with pymupdf.open(stream=blocks[0]["data"], filetype="pdf") as cut:
            assert [p.rotation for p in cut] == [90, 90]

    def test_the_pictures_route_shows_a_turned_page_upright(self, turned):
        from report_ingest.model_first_pages import picture_blocks
        doc, _pdf = turned
        blocks = picture_blocks(doc, [1])
        assert "turned upright" in blocks[0]["text"]
        width, height = _size(blocks[1]["png"])
        assert width > height
        words = blocks[2]["text"].splitlines()
        for label, _words, printed, _values in TURNED_LEGEND:
            assert words.index(printed) == words.index(label) + 2

    def test_a_zoom_on_a_turned_page_is_upright(self, turned):
        from report_ingest.visual_scales import zoom_plot
        doc, _pdf = turned
        blocks = zoom_plot(doc, [1], {"page": 1,
                                      "bbox": [470, 250, 560, 760]},
                           dpi=72, noun="sheet", instruction="Read it.",
                           zooms=[])
        assert "turned upright" in blocks[0]["text"]
        width, height = _size(blocks[-1]["png"])
        assert width > height
        # The size said is the upright crop's.
        assert f"({width}x{height} px)" in blocks[0]["text"]

    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_a_zoom_asked_in_the_upright_frame_crops_that_region(
            self, quarters, page):
        """The lab reader lists a turned page in its upright picture's
        frame, so the box it is asked to zoom is in that frame: cut from the
        page as displayed, said back in the frame asked in, kept as
        displayed."""
        from report_ingest.visual_scales import zoom_plot
        turn = turn_of(quarters, page)
        displayed = (100.0, 200.0, 300.0, 260.0)
        asked = list(upright_box(displayed, turn))
        zooms = []
        blocks = zoom_plot(quarters, [page], {"page": page, "bbox": asked},
                           dpi=72, noun="sheet", instruction="Read it.",
                           zooms=zooms, upright_frame=True)
        assert zooms[0]["bbox"] == pytest.approx(displayed)
        # The crop said back (the box with planlens' margin round it) is in
        # the frame the box was asked in.
        said = [float(v) for v in re.search(
            r"box \[([^\]]*)\]", blocks[0]["text"]).group(1).split(",")]
        assert said[0] <= asked[0] < asked[2] <= said[2]
        assert said[1] <= asked[1] < asked[3] <= said[3]
        assert "upright picture's frame" in blocks[0]["text"]
        width, height = _size(blocks[-1]["png"])
        assert f"({width}x{height} px)" in blocks[0]["text"]
        # Wider than high, as the region is in the upright frame.
        assert (width > height) == (asked[2] - asked[0] > asked[3] - asked[1])

    def test_the_lab_readers_zoom_is_asked_in_the_upright_frame(self,
                                                                 turned):
        from report_ingest.lab_reader import _Tools
        doc, _pdf = turned
        turn = turn_of(doc, 1)
        tools = _Tools(doc, [1])
        displayed = (100.0, 200.0, 300.0, 260.0)
        content, is_error = tools.run("zoom_plot", {
            "page": 1, "bbox": list(upright_box(displayed, turn))})
        assert not is_error
        assert tools.zooms[0]["bbox"] == pytest.approx(displayed)

    def test_a_zoom_on_an_upright_page_is_as_it_was(self, upright):
        from report_ingest.visual_scales import zoom_plot
        blocks = zoom_plot(upright, [0], {"page": 0,
                                          "bbox": [50, 580, 520, 650]},
                           dpi=72, noun="sheet", instruction="Read it.",
                           zooms=[])
        assert "turned" not in blocks[0]["text"]
        width, height = _size(blocks[-1]["png"])
        assert width > height


# ---------------------------------------------------------------------------
# placement by text (model-first): a turned sheet's rows are its rows
# ---------------------------------------------------------------------------

class TestPlacement:
    @pytest.mark.parametrize("page", range(len(QUARTERS)))
    def test_a_legend_rows_values_place_its_test_on_that_row(self, quarters,
                                                             page):
        from report_ingest.locate import PageWords, place_lab_tests
        label, _words, _printed, values = TURNED_LEGEND[1]
        test = LabTest(kind="gradation", depth_top=Quantity(value=5.0,
                                                            unit="m"),
                       pages=[page], result=GradationResult(**values),
                       prov=[Provenance(page=page, method="model")])
        words = PageWords(quarters)
        assert words(page).turn is not None
        got = place_lab_tests(words, [test], pages=[page])
        (placed,) = got.items
        assert placed.status == "placed"
        row = _values_box(quarters, page, label)
        box = test.prov[0].bbox
        # Displayed, and on the 5.0 m row's values line.
        assert box[0] >= row[0] - 1 and box[2] <= row[2] + 1
        assert box[1] >= row[1] - 1 and box[3] <= row[3] + 1

    def test_an_upright_page_is_searched_as_before(self, upright):
        from report_ingest.locate import PageWords
        words = PageWords(upright)
        assert words(0).turn is None
        assert words.displayed(0, (1.0, 2.0, 3.0, 4.0)) == (1.0, 2.0, 3.0,
                                                           4.0)


# ---------------------------------------------------------------------------
# RI-150: the summary and the sheets after it are two items
# ---------------------------------------------------------------------------

class TestOneItemPerSheet:
    def test_a_summary_and_the_grading_sheets_are_cut_apart(self, turned):
        doc, _pdf = turned
        parts = lab_parts(doc, [0, 1, 2, 3])
        assert [p for p, _k in parts] == [[0], [1, 2, 3]]
        assert set(parts[0][1]) == {"atterberg", "gradation"}
        assert parts[1][1] == ("gradation",)

    def test_the_items_are_named_and_say_why(self, turned):
        from planlens.document.roles import Item
        doc, _pdf = turned
        item = Item(id="item_5", kind="lab_test", pages=[0, 1, 2, 3],
                    title="Grain size", evidence={"first_page": 0})
        out = split_lab_items(doc, [item])
        assert [(i.id, i.pages) for i in out] == [("item_5a", [0]),
                                                  ("item_5b", [1, 2, 3])]
        assert all(i.evidence["split_from"] == "item_5" for i in out)
        assert out[1].evidence["first_page"] == 1
        assert out[1].evidence["title_kinds"] == ["gradation"]

    def test_other_items_and_one_sheet_items_pass_through(self, turned):
        from planlens.document.roles import Item
        doc, _pdf = turned
        log = Item(id="item_1", kind="boring_log", pages=[0, 1])
        one = Item(id="item_2", kind="lab_test", pages=[1, 2, 3])
        assert split_lab_items(doc, [log, one]) == [log, one]

    def test_a_page_naming_no_kind_continues_its_sheet(self, tmp_path):
        from report_ingest.tests.lab_fixtures import LETTER, _text
        made = pymupdf.open()
        for title in ("GRAIN SIZE DISTRIBUTION", "", "ATTERBERG LIMITS"):
            page = made.new_page(width=LETTER[0], height=LETTER[1])
            if title:
                _text(page, 150, 60, title, 14, bold=True)
            _text(page, 60, 400, "Sheet notes")
        doc = _open(made.tobytes(), tmp_path, "three.pdf")
        made.close()
        assert [p for p, _k in lab_parts(doc, [0, 1, 2])] == [[0, 1], [2]]

    def test_the_graph_cuts_lab_items(self):
        import report_ingest.graph as graph
        assert graph.split_lab_items is split_lab_items

    def _titled(self, tmp_path, titles, heads=()):
        """Upright made-up pages, one per title ("" for none), each with
        the header line in ``heads`` (or none)."""
        from report_ingest.tests.lab_fixtures import LETTER, _text
        made = pymupdf.open()
        for n, title in enumerate(titles):
            page = made.new_page(width=LETTER[0], height=LETTER[1])
            if title:
                _text(page, 150, 60, title, 14, bold=True)
            if n < len(heads) and heads[n]:
                _text(page, 60, 100, heads[n])
            _text(page, 60, 400, "Sheet notes")
        doc = _open(made.tobytes(), tmp_path, "titled.pdf")
        made.close()
        return doc

    @pytest.mark.parametrize("titles", [
        ("MOISTURE-DENSITY RELATION (PROCTOR)", "CALIFORNIA BEARING RATIO"),
        ("PARTICLE SIZE ANALYSIS", "HYDROMETER ANALYSIS",
         "SPECIFIC GRAVITY OF SOILS"),
        ("ATTERBERG LIMITS", "MOISTURE CONTENT"),
        ("SUMMARY OF LABORATORY TESTS",
         "SUMMARY OF LABORATORY TESTS - ATTERBERG LIMITS"),
        ("GRAIN SIZE DISTRIBUTION & ATTERBERG LIMITS TEST RESULTS",
         "GRAIN SIZE DISTRIBUTION, ATTERBERG LIMITS & MOISTURE CONTENT"),
    ])
    def test_companion_sheets_stay_one_item(self, tmp_path, titles):
        doc = self._titled(tmp_path, titles)
        try:
            parts = lab_parts(doc, list(range(len(titles))))
            assert [p for p, _k in parts] == [list(range(len(titles)))]
        finally:
            doc.close()

    @pytest.mark.parametrize("titles,cut", [
        (("SUMMARY OF LABORATORY TESTS", "ATTERBERG LIMITS"), [[0], [1]]),
        (("ATTERBERG LIMITS", "SUMMARY OF LABORATORY TESTS"), [[0], [1]]),
        (("ATTERBERG LIMITS", "CONSOLIDATION TEST"), [[0], [1]]),
        (("MOISTURE-DENSITY RELATION (PROCTOR)", "CALIFORNIA BEARING RATIO",
          "DIRECT SHEAR TEST"), [[0, 1], [2]]),
    ])
    def test_a_summary_or_an_unrelated_test_starts_a_new_item(
            self, tmp_path, titles, cut):
        doc = self._titled(tmp_path, titles)
        try:
            assert [p for p, _k in lab_parts(doc, list(range(len(titles))))] \
                == cut
        finally:
            doc.close()

    def test_a_cut_part_carries_the_hole_its_first_part_prints(self,
                                                               tmp_path):
        from planlens.document.roles import Item
        doc = self._titled(
            tmp_path, ("SUMMARY OF LABORATORY TESTS", "ATTERBERG LIMITS",
                       "ATTERBERG LIMITS"),
            heads=("Boring: QX-7", "", ""))
        try:
            item = Item(id="item_3", kind="lab_test", pages=[0, 1, 2])
            out = split_lab_items(doc, [item])
            assert [i.pages for i in out] == [[0], [1, 2]]
            assert "carried_link" not in out[0].evidence
            assert out[1].evidence["carried_link"] == {
                "investigation_id": "QX-7", "page": 0}
        finally:
            doc.close()

    def test_a_part_printing_its_own_hole_carries_none(self, tmp_path):
        from planlens.document.roles import Item
        doc = self._titled(
            tmp_path, ("SUMMARY OF LABORATORY TESTS", "ATTERBERG LIMITS"),
            heads=("Boring: QX-7", "Boring: QX-8"))
        try:
            out = split_lab_items(doc, [Item(id="item_3", kind="lab_test",
                                             pages=[0, 1])])
            assert "carried_link" not in out[1].evidence
        finally:
            doc.close()

    def test_the_reader_is_told_the_carried_hole(self, turned):
        doc, _pdf = turned
        engine = FakeEngine([{"final": LabSheetReading(
            depth_unit="m", tests=[], pages_read=[1])}])
        read_lab_sheet(doc, [1], engine, hint_kind="gradation",
                       carried_link={"investigation_id": "QX-7", "page": 0})
        brief = "\n".join(
            block.get("text", "") for block in
            engine.calls[0]["messages"][0]["content"]
            if isinstance(block, dict) and block.get("type") == "text")
        assert "boring 'QX-7'" in brief and "(page 0)" in brief
