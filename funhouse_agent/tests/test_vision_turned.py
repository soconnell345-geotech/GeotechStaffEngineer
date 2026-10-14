"""Turned lettering, sideways sheets and lettering cut by a zoom's edge
(BACKLOG VIS-01, VIS-04, VIS-05).

Foundry brief 5 (2026-10-08, module_work/review_eval_results/
2026-10-08_foundry_5.33.0rc1/TRACE_REVIEW_suite_rest.md "R" and
TRACE_REVIEW_suite_meck.md 4.3-4.4):

* Sol read turned lettering markedly worse than upright: 6 of 15 turned
  look-alike tags zoomed were misread, against 0 of 11 upright ones.
* One whole-page answer on a sideways sheet gave every box in the frame of
  the sheet turned upright to read: a mark placed from it would have landed
  about 300 pt off.
* A zoom whose edge cut a turned note's lines came back with the cut words
  made up (a 4 in pipe and 6 in spacing; 6 in and 3 in are printed).

Where the text layer says which way the lettering runs, the view now goes to
the model upright and its boxes come back turned; a zoom takes in whole the
lines its edges cut, within a limit, and is told of the rest. Where there is
no text layer (every sheet of the cases above), the agent may ask for a
``turn``, and behind ``GEOTECH_VISION_TURN_RETRY`` (OFF) a look that finds
its lettering turned is read again upright and a page found turned is sent
turned.

The stand-in engines SEE: they find a red mark in the image they are sent
and answer in its pixels, as the model is asked to. One of them also does
what Sol did: on lettering that runs up the image it reads the image turned
upright and gives its box in that turned frame.
"""

from __future__ import annotations

import json
import re

import pytest

fitz = pytest.importorskip("fitz")
np = pytest.importorskip("numpy")

from funhouse_agent import (  # noqa: E402
    review_flags, vision_probe, vision_tools, vision_view,
)
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_render_region, _vision_prompt,
)

#: The red mark, as drawn (PDF points on the unrotated page, top-left origin).
MARK = (400.0, 300.0, 430.0, 312.0)
#: One line of a note, about 240 pt long at 9 pt.
NOTE = "UNDERDRAIN PIPE SHALL BE 6 IN MIN AT 3 IN SPACING"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV,
                vision_view.POLICY_ENV, vision_view.PATCH_ALIGN_ENV,
                review_flags.VISION_TEXT_ENV,
                review_flags.VISION_STRUCTURED_ENV,
                review_flags.VISION_TURN_RETRY_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    vision_probe.clear_cache()
    vision_tools.clear_turn_memory()
    yield
    vision_probe.clear_cache()
    vision_tools.clear_turn_memory()


def _sheet(text_rotate=0, page_rotate=0, n_lines=4, mark=MARK, text=NOTE,
           strokes=False):
    """A letter page: ``n_lines`` note lines drawn turned ``text_rotate``
    degrees counter-clockwise (0 reads across, 90 from bottom to top), a red
    mark, and ``/Rotate`` ``page_rotate``. ``strokes`` draws the lettering's
    stand-in as lines instead (no text layer, like a CAD plot). Returns the
    PDF and the mark as the page displays it."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(n_lines):
        step = 14 * i
        origin = {0: (60, 450 + step), 90: (100 + step, 700),
                  180: (360, 450 + step), 270: (100 + step, 450)}[text_rotate]
        if strokes:
            x, y = origin
            end = {0: (x + 240, y), 90: (x, y - 240), 180: (x - 240, y),
                   270: (x, y + 240)}[text_rotate]
            page.draw_line(fitz.Point(x, y), fitz.Point(*end), color=(0, 0, 0),
                           width=2)
        elif text:
            page.insert_text(origin, text, fontsize=9, rotate=text_rotate)
    page.draw_rect(fitz.Rect(*mark), color=(1, 0, 0), fill=(1, 0, 0))
    if page_rotate:
        page.set_rotation(page_rotate)
    shown = tuple(fitz.Rect(*mark) * page.rotation_matrix)
    data = doc.tobytes()
    doc.close()
    return data, shown


def _pixels(image_bytes):
    pix = fitz.Pixmap(image_bytes)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return a[:, :, :3]


def _box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _red(a):
    return (a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)


def _black(a):
    return a.max(axis=2) < 100


class Seer:
    """Answers with the pixel box of the red mark in the image it is sent."""

    def __init__(self):
        self.prompts, self.images = [], []

    def analyze_image(self, image_bytes, prompt=""):
        self.prompts.append(prompt)
        self.images.append(image_bytes)
        a = self.frame(_pixels(image_bytes))
        if not _red(a).any():
            return "nothing marked here"
        b = _box(_red(a))
        return f"MARK px=[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"

    def frame(self, a):
        return a


class TurnsToRead(Seer):
    """What Sol did once in brief 5: where the lettering runs up the image,
    it reads the image turned a quarter turn clockwise, upright, and gives
    its box in THAT frame."""

    turned = 0

    def frame(self, a):
        ink = _black(a)
        if ink.any():
            x0, y0, x1, y1 = _box(ink)
            if (y1 - y0) > (x1 - x0):
                self.turned += 1
                return np.rot90(a, k=-1)
        return a


def _grid_box(text):
    m = re.search(r"\[(\d+), (\d+), (\d+), (\d+)\]", text)
    return [int(v) for v in m.groups()]


def _centre_error(view, grid_box, shown):
    """How far (pt) the centre of a 0-999 box on ``view`` is from the
    mark's."""
    b = vision_view.image_box_to_page(view, grid_box)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    tx, ty = (shown[0] + shown[2]) / 2, (shown[1] + shown[3]) / 2
    return ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5


def _page(pdf, engine, **args):
    return json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find the mark",
         "tiles": "off", **args}, engine, {"s": pdf}))


def _zoom(pdf, engine, **args):
    return json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "prompt": "read the note",
         **args}, engine, {"s": pdf}))


# -- the arithmetic of a quarter turn ---------------------------------------------

@pytest.mark.parametrize("turn", [0, 90, 180, 270])
def test_a_box_turned_and_turned_back_is_the_same_box(turn):
    size = (1583, 2048)                       # the view as the page shows it
    sent = vision_view.unturned_size(size, turn)
    for box in ([0, 0, 10, 20], [100.5, 200, 400, 260], [1500, 1900, 1583,
                                                         2048]):
        there = vision_view.turn_px_box(box, size, turn)
        assert all(0 <= v <= max(sent) for v in there)
        assert vision_view.unturn_px_box(there, sent, turn) == \
            pytest.approx(box)


def test_a_quarter_turn_clockwise_moves_the_top_left_to_the_top_right():
    # a 10 x 10 box in the top-left corner of a 200 x 100 image
    assert vision_view.turn_px_box([0, 0, 10, 10], (200, 100), 90) == \
        [90, 0, 100, 10]
    assert vision_view.turn_px_box([0, 0, 10, 10], (200, 100), 180) == \
        [190, 90, 200, 100]
    assert vision_view.turn_px_box([0, 0, 10, 10], (200, 100), 270) == \
        [0, 190, 10, 200]
    assert vision_view.turn_edge("top", 90) == "right"
    assert vision_view.turn_edge("left", 90) == "top"
    assert vision_view.turn_edge("top", 180) == "bottom"
    assert vision_view.turn_edge("top", 270) == "left"


def _layer(*lines, size=(612.0, 792.0)):
    return vision_view.TextLayer(
        [vision_view.LayerLine(t, b, r) for t, b, r in lines], *size)


def test_which_way_a_view_is_turned():
    col = (100.0, 450.0, 110.0, 700.0)          # a line running up the page
    row = (60.0, 443.0, 300.0, 452.0)           # a line running across
    whole = (0, 0, 612, 792)
    turned = _layer(*[(NOTE, col, 90.0)] * 4)
    assert vision_view.reading_turn(turned, whole) == 90
    assert vision_view.reading_turn(_layer((NOTE, col, 270.0)) , col) == 270
    assert vision_view.reading_turn(_layer((NOTE, row, 180.0)), row) == 180
    # half and half: neither way has a clear majority
    mixed = _layer((NOTE, col, 90.0), (NOTE, row, 0.0))
    assert vision_view.reading_turn(mixed, whole) == 0
    # a direction nobody knows (hidden CAD text) turns nothing
    assert vision_view.reading_turn(_layer((NOTE, col, None)), col) == 0
    # lettering along a curve counts against every turn
    assert vision_view.reading_turn(
        _layer((NOTE, col, 90.0), (NOTE * 2, row, 45.0)), whole) == 0
    # a few degrees off a quarter turn is that turn
    assert vision_view.reading_turn(_layer((NOTE, col, 93.0)), col) == 90
    # only what lies in the view counts
    assert vision_view.reading_turn(turned, (300, 0, 612, 400)) == 0
    assert vision_view.reading_turn(None, whole) == 0


def test_a_whole_sheet_is_not_turned_for_one_turned_stamp():
    """A sheet whose only text-layer lettering is a turned stamp (the rest
    drawn as strokes the text layer cannot see) is not turned whole; a zoom
    on the stamp is."""
    stamp = (580.0, 300.0, 590.0, 500.0)
    layer = _layer(("PRELIMINARY NOT FOR CONSTRUCTION", stamp, 90.0))
    assert vision_view.reading_turn(layer, (0, 0, 612, 792)) == 0
    assert vision_view.reading_turn(layer, (560, 280, 612, 520)) == 90


# -- VIS-01: turned lettering reaches the model upright ---------------------------

def _one_word(rotate):
    """A page with one line of lettering turned ``rotate`` degrees, and a
    240 pt square window centred on it (the same size whatever the turn, so
    every render draws the lettering at the same scale)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    origin = {0: (200, 400), 90: (300, 520), 180: (420, 400),
              270: (300, 280)}[rotate]
    page.insert_text(origin, "UPRIGHT 6 IN", fontsize=24, rotate=rotate)
    x0, y0, x1, y1 = page.get_text("dict")["blocks"][0]["lines"][0]["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    data = doc.tobytes()
    doc.close()
    return data, [cx - 120, cy - 120, cx + 120, cy + 120]


def _ink_iou(img_a, img_b):
    """How alike the lettering in two images is: the overlap of their ink,
    each cropped to its own ink and aligned at the top-left."""
    a = _black(_pixels(img_a))
    b = _black(_pixels(img_b))
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    a, b = a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1]
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), bool)
    pb = np.zeros((h, w), bool)
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    return (pa & pb).sum() / max(1, (pa | pb).sum())


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_turned_lettering_reaches_the_model_upright(rotate):
    pdf, window = _one_word(0)
    reference, rinfo = vision_view.render_view(pdf, page=0, bbox=window,
                                               pad_frac=0, upright=True)
    assert "turn" not in rinfo                  # upright already: as it was
    pdf, window = _one_word(rotate)
    before, binfo = vision_view.render_view(pdf, page=0, bbox=window,
                                            pad_frac=0)
    after, ainfo = vision_view.render_view(pdf, page=0, bbox=window,
                                           pad_frac=0, upright=True)
    assert "turn" not in binfo                  # nobody asked: as it was
    assert ainfo["turn"] == rotate
    assert ainfo["clip"] == binfo["clip"]       # the view is the same page rect
    if rotate != 180:
        assert (ainfo["width_px"], ainfo["height_px"]) == \
            (binfo["height_px"], binfo["width_px"])
    # the lettering as sent now matches the same words drawn upright
    assert _ink_iou(after, reference) > 0.8
    assert _ink_iou(before, reference) < 0.5


def test_a_page_read_whole_tells_the_agent_it_was_turned_and_why():
    pdf, shown = _sheet(text_rotate=90)
    engine = Seer()
    out = _page(pdf, engine)
    assert "turned 90 degrees clockwise" in out["turned"]
    assert "from bottom to top" in out["turned"]
    assert "image_box is on this view as the page shows it" in out["turned"]
    assert "turned back" in out["boxes"]
    assert out["view"] == [0.0, 0.0, 612.0, 792.0]
    assert out["view_px"][0] > out["view_px"][1]    # sent landscape
    assert f"This image is {out['view_px'][0]} x {out['view_px'][1]}" in \
        engine.prompts[0]


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("text_rotate", [0, 90, 180, 270])
def test_boxes_from_any_turned_view_land_on_the_thing(text_rotate, page_rotate):
    """VIS-05: every combination of a page's ``/Rotate`` and the way its
    lettering runs. The page answer's box, the zoom made on it and the
    zoom's own box all land on the mark."""
    pdf, shown = _sheet(text_rotate=text_rotate, page_rotate=page_rotate)
    engine = Seer()
    page = _page(pdf, engine)
    assert "error" not in page, page
    box = _grid_box(page["analysis"])
    assert _centre_error(page["view"], box, shown) < 1.0
    zoom = _zoom(pdf, engine, view=page["view"], image_box=box)
    assert "error" not in zoom, zoom
    vx0, vy0, vx1, vy1 = zoom["view"]
    assert vx0 < shown[0] and vy0 < shown[1] and vx1 > shown[2] \
        and vy1 > shown[3]
    assert _centre_error(zoom["view"], _grid_box(zoom["analysis"]),
                         shown) < 0.5


@pytest.mark.parametrize("page_rotate", [0, 90, 180, 270])
def test_a_mark_anchored_on_a_turned_zoom_lands_on_the_thing(page_rotate,
                                                            tmp_path):
    """VIS-05 to the end: a zoom sent turned, on a page with ``/Rotate``,
    whose box is passed to planlens as ``view`` + ``image_box`` (what
    annotate_document does) and read back off the written copy."""
    from planlens.document import Document
    from planlens.document.markup_writer import write_markups
    pdf, shown = _sheet(text_rotate=90, page_rotate=page_rotate,
                        mark=(120.0, 600.0, 150.0, 612.0))
    cx, cy = (shown[0] + shown[2]) / 2, (shown[1] + shown[3]) / 2
    zoom = _zoom(pdf, Seer(), bbox=[cx - 60, cy - 60, cx + 60, cy + 60],
                 pad_frac=0)
    assert "error" not in zoom, zoom
    if page_rotate in (0, 180):
        # the lettering runs up or down the page as displayed: sent turned
        assert "turned" in zoom
    src = tmp_path / "sheet.pdf"
    src.write_bytes(pdf)
    out = str(tmp_path / "marked.pdf")
    report = write_markups(str(src), out, [
        {"kind": "box", "page": 0, "comment": "the mark",
         "view": zoom["view"], "image_box": _grid_box(zoom["analysis"])}],
        author="test")
    assert report.n_written == 1, report.skipped
    doc = Document(filepath=out)
    try:
        (mk,) = doc.markups()
    finally:
        doc.close()
    assert all(abs(a - b) < 1.5 for a, b in zip(mk.bbox, shown)), \
        (mk.bbox, shown)


def test_a_sideways_sheet_is_read_in_upright_tiles_whose_boxes_come_back():
    pdf, shown = _sheet(text_rotate=90, mark=(120.0, 600.0, 150.0, 612.0))
    engine = Seer()
    out = _page(pdf, engine, tiles="2")
    assert len(out["tiles"]) == 4
    holding = [t for t in out["tiles"] if "MARK" in (t.get("analysis") or "")
               or "[" in (t.get("analysis") or "")]
    assert holding
    for t in holding:
        assert t["turned"] == 90
        assert _centre_error(t["view"], _grid_box(t["analysis"]), shown) < 0.5
    assert "were sent turned" in out["tiles_turned"]
    # the merged list carries the page box too
    assert any(abs((f["page_bbox"][0] + f["page_bbox"][2]) / 2 - 135) < 2
               for f in out.get("found", []))


def test_the_text_layer_and_located_items_use_the_turned_image(monkeypatch):
    monkeypatch.setenv(review_flags.VISION_TEXT_ENV, "1")
    monkeypatch.setenv(review_flags.VISION_STRUCTURED_ENV, "1")
    pdf, shown = _sheet(text_rotate=90)

    class Checks(Seer):
        boxes = []

        def analyze_image(self, image_bytes, prompt=""):
            a = _pixels(image_bytes)
            for m in re.finditer(r"px=\[(\d+), (\d+), (\d+), (\d+)\] UNDER",
                                 prompt):
                x0, y0, x1, y1 = (int(v) for v in m.groups())
                Checks.boxes.append((x1 - x0, y1 - y0,
                                     _black(a[y0:y1, x0:x1]).mean()))
            b = _box(_red(a))
            return ('The mark.\nLOCATED: [{"what": "mark", "px": '
                    f'[{b[0]}, {b[1]}, {b[2]}, {b[3]}]' + '}]')

    out = _page(pdf, Checks())
    # each text line given to the call is a row ACROSS the image it got, and
    # its box holds that line's ink
    assert len(Checks.boxes) == 4
    for w, h, ink in Checks.boxes:
        assert w > 5 * h and ink > 0.05
    (item,) = out["located"]
    b = item["page_bbox"]
    assert abs((b[0] + b[2]) / 2 - (shown[0] + shown[2]) / 2) < 1.0
    assert abs((b[1] + b[3]) / 2 - (shown[1] + shown[3]) / 2) < 1.0


def test_vis05_before_and_after_on_a_sideways_sheet(monkeypatch):
    """The brief-5 failure, measured on the fixture: a model that reads
    sideways lettering by turning the image gives its box in the turned
    frame. Sent as before, that box is a long way off; sent upright, there is
    nothing to turn, and the box lands on the mark."""
    pdf, shown = _sheet(text_rotate=90)
    with monkeypatch.context() as m:
        m.setattr(vision_view, "reading_turn",
                  lambda layer, clip, default=0: 0)
        engine = TurnsToRead()
        before = _page(pdf, engine)
        assert engine.turned == 1 and "turned" not in before
        err_before = _centre_error(before["view"],
                                   _grid_box(before["analysis"]), shown)
    engine = TurnsToRead()
    after = _page(pdf, engine)
    assert engine.turned == 0 and "turned" in after
    err_after = _centre_error(after["view"], _grid_box(after["analysis"]),
                              shown)
    assert err_before > 100.0, err_before
    assert err_after < 1.0, err_after


def test_a_sheet_with_no_text_layer_goes_exactly_as_before():
    """Lettering drawn as strokes (the Mecklenburg sheets, the tag sets):
    nothing says which way it runs, so the image is not touched."""
    pdf, shown = _sheet(text_rotate=90, strokes=True)
    engine = Seer()
    out = _page(pdf, engine)
    plain, _info = vision_view.render_view(pdf, page=0)
    assert engine.images[0] == plain
    assert "turned" not in out
    zoom = _zoom(pdf, Seer(), bbox=[80, 430, 200, 720])
    assert "turned" not in zoom and "window_widened" not in zoom
    assert "edge_note" not in zoom


def test_other_callers_get_the_page_as_it_is():
    """The markup check, the sweep and the chart read-offs call render_view
    without asking for upright: unchanged."""
    pdf, _shown = _sheet(text_rotate=90)
    _data, info = vision_view.render_view(pdf, page=0)
    assert "turn" not in info and info["height_px"] > info["width_px"]


def test_an_inline_look_is_not_turned():
    """Inline, the main model writes image_box itself on the image it is
    shown, so that image goes as the page shows it."""
    pdf, _shown = _sheet(text_rotate=90)
    from funhouse_agent import inline_store
    try:
        out = json.loads(_dispatch_analyze_pdf_page(
            {"attachment_key": "s", "page": 0, "_inline": True}, Seer(),
            {"s": pdf}))
        assert "turned" not in out and out["view_px"][1] > out["view_px"][0]
    finally:
        inline_store.clear()


def test_tiling_is_judged_the_same_for_a_turned_image():
    pdf, _shown = _sheet(text_rotate=90)
    _d, flat = vision_view.render_view(pdf, page=0)
    _d, turned = vision_view.render_view(pdf, page=0, upright=True)
    assert turned["turn"] == 90
    assert vision_view.tile_grid(turned) == vision_view.tile_grid(flat)


def test_a_turn_can_be_asked_for_by_hand():
    pdf, _shown = _sheet(text_rotate=0)
    _d, info = vision_view.render_view(pdf, page=0, turn=270)
    assert info["turn"] == 270 and info["turn_from"] == "asked"
    assert "as asked" in vision_view.view_payload(info)["turned"]


# -- VIS-05: every location request says boxes are in the image as sent -----------

def test_every_location_request_says_boxes_are_in_the_image_as_sent():
    sentence = vision_view.FRAME_INSTRUCTION
    assert "even where the drawing or its lettering is turned" in sentence
    assert sentence in vision_view.pixel_instruction((100, 100))
    assert sentence in vision_view.GRID_INSTRUCTION


# -- VIS-04: lettering cut by a zoom's edge ----------------------------------------

def test_every_look_asks_for_cut_lettering_to_be_marked_not_completed():
    text = vision_view.READING_INSTRUCTION
    assert "write (cut) where they stop" in text
    assert "never complete a cut word or line" in text
    assert text in vision_view.pixel_instruction((100, 100))


def test_a_zoom_that_cuts_lines_takes_them_in_whole():
    pdf, _shown = _sheet(text_rotate=0)        # lines from x 60 to about 300
    engine = Seer()
    out = _zoom(pdf, engine, bbox=[100, 430, 340, 520], pad_frac=0)
    assert "4 line(s)" in out["window_widened"]
    assert out["view"][0] < 60 and out["view"][2] == pytest.approx(340, abs=1)
    assert "edge_note" not in out
    assert "running past the edge" not in engine.prompts[0]
    # the zoom says nothing different when nothing is cut
    quiet = _zoom(pdf, Seer(), bbox=[40, 430, 340, 520], pad_frac=0)
    assert "window_widened" not in quiet and "edge_note" not in quiet


def test_lines_too_long_to_take_in_are_named_to_the_vision_call():
    pdf, _shown = _sheet(text_rotate=0)
    engine = Seer()
    out = _zoom(pdf, engine, bbox=[150, 430, 200, 520], pad_frac=0)
    assert "window_widened" not in out
    assert out["view"] == [150.0, 430.0, 200.0, 520.0]
    assert out["edge_note"].startswith("4 line(s) of lettering run past")
    prompt = engine.prompts[0]
    assert ("The page's text layer shows 4 line(s) of lettering running "
            "past the edge of this image") in prompt
    assert prompt.count("(cut by the right and left edge)") == 4
    assert "write (cut) where it stops" in prompt
    # each line's visible part, in pixels of the image as sent, holds ink
    a = _pixels(engine.images[0])
    for m in re.finditer(r"px=\[(\d+), (\d+), (\d+), (\d+)\] \(cut", prompt):
        x0, y0, x1, y1 = (int(v) for v in m.groups())
        assert _black(a[y0:y1, x0:x1]).mean() > 0.05


def test_a_turned_zoom_names_the_cut_edges_of_the_image_it_got():
    """Lines running up the page, cut by the window's top and bottom: sent
    turned upright, they run across the image and its RIGHT and LEFT edges
    cut them."""
    pdf, _shown = _sheet(text_rotate=90)        # lines x 90-150, y 457-700
    engine = Seer()
    out = _zoom(pdf, engine, bbox=[80, 520, 170, 620], pad_frac=0)
    assert "turned 90 degrees" in out["turned"]
    prompt = engine.prompts[0]
    assert prompt.count("(cut by the right and left edge)") == 4
    w, h = (int(v) for v in re.search(r"This image is (\d+) x (\d+)",
                                      prompt).groups())
    assert w > h
    for m in re.finditer(r"px=\[(\d+), (\d+), (\d+), (\d+)\] \(cut", prompt):
        x0, y0, x1, y1 = (int(v) for v in m.groups())
        assert x1 - x0 > 5 * (y1 - y0)          # a row across the image
        assert x0 == 0 and x1 == w              # it runs off both side edges


def test_tiles_are_told_which_lines_their_edges_cut():
    pdf, _shown = _sheet(text_rotate=0, n_lines=6)
    engine = Seer()
    _page(pdf, engine, tiles="2")
    told = [p for p in engine.prompts if "running past the edge" in p]
    assert told                                  # the tile on the right
    assert all("This image is tile" in p for p in told)


def test_the_cut_note_names_at_most_a_few_lines():
    layer = _layer(*[(NOTE, (0.0, 10.0 * i, 600.0, 10.0 * i + 8), 0.0)
                     for i in range(20)])
    cuts = vision_view.cut_lines((100, 0, 200, 300), layer)
    assert len(cuts) == 20
    note = vision_view.cut_note(cuts, (100, 0, 200, 300), (683, 2048))
    assert note.count("px=") == vision_view.CUT_LINES_LISTED
    assert "and 12 more" in note
    assert vision_view.cut_note([], (0, 0, 1, 1), (10, 10)) is None


def test_the_window_is_widened_within_its_limit_and_held_to_the_page():
    layer = _layer(("A LINE", (5.0, 100.0, 150.0, 108.0), 0.0),
                   ("FAR", (5.0, 200.0, 900.0, 208.0), 0.0),
                   size=(612.0, 792.0))
    clip, n = vision_view.widen_to_lines((20, 90, 220, 300), layer)
    assert n == 1                               # 15 pt left: inside 50 pt
    assert clip[0] == pytest.approx(3.0) and clip[2] == 220
    # a line wholly inside, or wholly outside, is not cut
    assert vision_view.cut_lines((0, 0, 612, 792), _layer(
        ("IN", (10.0, 10.0, 50.0, 18.0), 0.0))) == []
    assert vision_view.cut_lines((0, 0, 100, 100), _layer(
        ("OUT", (200.0, 10.0, 250.0, 18.0), 0.0))) == []
    # held to the page
    edge = _layer(("EDGE", (-4.0, 100.0, 40.0, 108.0), 0.0))
    clip, n = vision_view.widen_to_lines((10, 90, 200, 300), edge)
    assert n == 1 and clip[0] == 0.0


def test_a_window_narrow_enough_to_mark_from_stays_that_narrow():
    """A zoom of 300 pt or less is one a mark may be anchored on (planlens
    refuses a small mark read off a wider view): widening never takes it
    past, nearest lines first."""
    layer = _layer(("NEAR", (90.0, 100.0, 300.0, 108.0), 0.0),
                   ("FAR", (60.0, 120.0, 300.0, 128.0), 0.0))
    clip, n = vision_view.widen_to_lines((100, 90, 380, 200), layer)
    assert n == 1                               # NEAR (10 pt), not FAR (40)
    assert clip[2] - clip[0] <= vision_view.MARK_VIEW_PT
    # a window already wider than that may grow within its own limit
    clip, n = vision_view.widen_to_lines((100, 90, 420, 200), layer)
    assert n == 2 and clip[0] == pytest.approx(58.0)


def test_a_prompt_puts_the_cut_note_before_the_location_sentence():
    text = _vision_prompt("Q", [0, 0, 10, 10], None, (100, 100),
                          shape="SHAPE", cut="CUT NOTE")
    assert text.index("Q") < text.index("SHAPE") < text.index("CUT NOTE") \
        < text.index("This image is")


# -- (a) the agent's turn: lettering with no text layer ------------------------

@pytest.mark.parametrize("value,want", [
    (None, None), ("", None), ("auto", None), (90, 90), ("90", 90),
    ("270°", 270), (450, 90), (-90, 270), (0, 0), (180.0, 180)])
def test_turn_values_that_mean_something(value, want):
    assert vision_tools._parse_turn(value) == (want, None)


@pytest.mark.parametrize("value", [45, "sideways", 90.5, "ninety"])
def test_a_turn_that_is_not_a_quarter_turn_costs_nothing(value):
    pdf, _shown = _sheet(text_rotate=90, strokes=True)
    for tool in (_dispatch_render_region, _dispatch_analyze_pdf_page):
        engine = Seer()
        out = json.loads(tool({"attachment_key": "s", "page": 0,
                               "bbox": [60, 440, 200, 720], "turn": value},
                              engine, {"s": pdf}))
        assert "not a quarter turn" in out["error"]
        assert "90, 180 or 270" in out["accepted"]
        assert engine.prompts == []


def test_the_agent_can_turn_a_zoom_on_lettering_drawn_as_lines():
    """No text layer: nothing turns the view by itself; turn=90 does, and the
    box still lands on the thing."""
    pdf, shown = _sheet(text_rotate=90, strokes=True,
                        mark=(120.0, 600.0, 150.0, 612.0))
    plain = _zoom(pdf, Seer(), bbox=[60, 440, 200, 720], pad_frac=0)
    assert "turned" not in plain
    engine = Seer()
    out = _zoom(pdf, engine, bbox=[60, 440, 200, 720], pad_frac=0, turn=90)
    assert "turned 90 degrees clockwise, as asked" in out["turned"]
    w, h = out["view_px"]
    assert w > h                                # the tall window, sent across
    assert _centre_error(out["view"], _grid_box(out["analysis"]), shown) < 0.5


def test_the_agent_can_turn_a_page_and_its_tiles():
    pdf, shown = _sheet(text_rotate=90, strokes=True,
                        mark=(120.0, 600.0, 150.0, 612.0))
    engine = Seer()
    out = _page(pdf, engine, tiles="2", turn=270)
    assert "as asked" in out["turned"]
    assert all(t["turned"] == 270 and t["turned_why"] == "as asked"
               for t in out["tiles"])
    holding = [t for t in out["tiles"] if "[" in (t.get("analysis") or "")]
    assert holding
    for t in holding:
        assert _centre_error(t["view"], _grid_box(t["analysis"]), shown) < 0.5


def test_turn_0_sends_the_page_as_it_shows():
    pdf, _shown = _sheet(text_rotate=90)        # a text layer that is turned
    assert "turned" in _page(pdf, Seer())
    out = _page(pdf, Seer(), turn=0)
    assert "turned" not in out and out["view_px"][1] > out["view_px"][0]


def test_an_inline_look_says_it_did_not_turn():
    pdf, _shown = _sheet(text_rotate=90, strokes=True)
    from funhouse_agent import inline_store
    try:
        out = json.loads(_dispatch_render_region(
            {"attachment_key": "s", "page": 0, "bbox": [60, 440, 200, 720],
             "turn": 90, "_inline": True}, Seer(), {"s": pdf}))
        assert "turned" not in out and "not used" in out["turn_note"]
    finally:
        inline_store.clear()


def test_turn_is_offered_in_the_tools_own_descriptions_only():
    from funhouse_agent.deep import prompt as deep_prompt
    from funhouse_agent.deep.review_agent import REVIEW_DESCRIPTIONS
    from funhouse_agent.deep.tools import make_vision_tools
    from funhouse_agent.native_tools import OPENAI_TOOLS
    tools = {t.name: t for t in make_vision_tools(engine=None)}
    for name in ("analyze_pdf_page", "render_region"):
        assert "turn" in tools[name].args
        assert "turn = 90, 180 or 270" in tools[name].description
        assert "turn = 90, 180 or 270" in REVIEW_DESCRIPTIONS[name]
        (schema,) = [t["function"] for t in OPENAI_TOOLS
                     if t["function"]["name"] == name]
        assert schema["parameters"]["properties"]["turn"]["enum"] == \
            [0, 90, 180, 270]
    assert "`turn`" in vision_tools.VISION_TOOL_DESCRIPTIONS
    for text in (deep_prompt.DOCUMENT_REVIEW_PROMPT,
                 deep_prompt.DOCUMENT_REVIEW_PROMPT_LEAN,
                 deep_prompt.build_domain_prompt()):
        assert "turn=" not in text and "turn =" not in text
        assert "TURNED" not in text


# -- (b) GEOTECH_VISION_TURN_RETRY: found turned, read again upright ------------

class SaysTurned(Seer):
    """A model asked for the TURNED line: where the lettering (black lines)
    in the image runs up it, it says so, and boxes the red mark either way."""

    def analyze_image(self, image_bytes, prompt=""):
        answer = super().analyze_image(image_bytes, prompt)
        a = _pixels(image_bytes)
        ink = _black(a)
        if vision_view.TURN_TAG_INSTRUCTION in prompt and ink.any():
            x0, y0, x1, y1 = _box(ink)
            if (y1 - y0) > (x1 - x0):
                return answer + "\nTURNED: bottom-to-top"
        return answer


@pytest.fixture
def retry_on(monkeypatch):
    monkeypatch.setenv(review_flags.VISION_TURN_RETRY_ENV, "1")


def test_the_switch_is_off_by_default_and_asks_nothing():
    assert not review_flags.vision_turn_retry()
    assert review_flags.VISION_TURN_RETRY_ENV in review_flags.ALL_ENVS
    assert review_flags.ARMS["turn_retry"] == {
        review_flags.VISION_TURN_RETRY_ENV: "1"}
    pdf, _shown = _sheet(text_rotate=90, strokes=True)
    engine = SaysTurned()
    out = _page(pdf, engine)
    assert len(engine.prompts) == 1 and "turned" not in out
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[0]


def test_a_look_found_turned_is_read_again_upright(retry_on):
    pdf, shown = _sheet(text_rotate=90, strokes=True)
    engine = SaysTurned()
    out = _page(pdf, engine)
    assert len(engine.prompts) == 2
    assert vision_view.TURN_TAG_INSTRUCTION in engine.prompts[0]
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[1]
    first, second = (_pixels(i).shape for i in engine.images)
    assert first[0] > first[1] and second[1] > second[0]   # then sent across
    assert "TURNED" not in out["analysis"]
    assert "second reading" in out["turned"]
    assert _centre_error(out["view"], _grid_box(out["analysis"]), shown) < 1.0


def test_a_zoom_found_turned_is_read_again_upright(retry_on):
    pdf, shown = _sheet(text_rotate=90, strokes=True,
                        mark=(120.0, 600.0, 150.0, 612.0))
    engine = SaysTurned()
    out = _zoom(pdf, engine, bbox=[60, 440, 200, 720], pad_frac=0)
    assert len(engine.prompts) == 2 and "second reading" in out["turned"]
    assert _centre_error(out["view"], _grid_box(out["analysis"]), shown) < 0.5


def test_a_page_found_turned_is_sent_turned_later(retry_on):
    """Two looks found the page turned, the page look among them: the next
    zoom goes turned from the start, in one call."""
    pdf, shown = _sheet(text_rotate=90, strokes=True,
                        mark=(120.0, 600.0, 150.0, 612.0))
    _page(pdf, SaysTurned())
    assert vision_tools.remembered_turn(pdf, 0) == 0     # one look is not two
    _zoom(pdf, SaysTurned(), bbox=[60, 440, 200, 720], pad_frac=0)
    assert vision_tools.remembered_turn(pdf, 0) == 90
    engine = SaysTurned()
    out = _zoom(pdf, engine, bbox=[80, 500, 180, 650], pad_frac=0)
    assert len(engine.prompts) == 1
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[0]
    assert "earlier looks at this page" in out["turned"]
    assert _centre_error(out["view"], _grid_box(out["analysis"]), shown) < 0.5
    # another page of the same file is its own page
    assert vision_tools.remembered_turn(pdf, 1) == 0


def test_zooms_on_a_few_turned_tags_do_not_turn_an_upright_page(retry_on):
    """An upright sheet with one turned group: zooms on the group are read
    again upright, but the whole-page look outweighs them, so later looks
    at the page still go as it shows."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(4):
        page.draw_line(fitz.Point(60, 450 + 14 * i),
                       fitz.Point(300, 450 + 14 * i), color=(0, 0, 0), width=2)
        page.draw_line(fitz.Point(450 + 14 * i, 340),
                       fitz.Point(450 + 14 * i, 100), color=(0, 0, 0), width=2)
    page.draw_rect(fitz.Rect(*MARK), color=(1, 0, 0), fill=(1, 0, 0))
    pdf = doc.tobytes()
    doc.close()
    engine = SaysTurned()
    _page(pdf, engine)
    assert len(engine.prompts) == 1                       # read upright
    for _ in range(2):
        e = SaysTurned()
        out = _zoom(pdf, e, bbox=[430, 80, 520, 360], pad_frac=0)
        assert len(e.prompts) == 2 and "second reading" in out["turned"]
    assert vision_tools.remembered_turn(pdf, 0) == 0
    e = SaysTurned()
    _zoom(pdf, e, bbox=[40, 430, 320, 510], pad_frac=0)
    assert len(e.prompts) == 1
    assert vision_view.TURN_TAG_INSTRUCTION in e.prompts[0]


def test_tiles_found_turned_are_read_again(retry_on):
    pdf, shown = _sheet(text_rotate=90, strokes=True,
                        mark=(120.0, 600.0, 150.0, 612.0))
    engine = SaysTurned()
    out = _page(pdf, engine, tiles="2")
    again = [t for t in out["tiles"] if t.get("turned")]
    assert again and all(t["turned_why"].startswith("a first look")
                         for t in again)
    assert len(engine.prompts) == 5 + 1 + len(again)
    for t in again:
        assert "TURNED" not in t["analysis"]
        if "[" in t["analysis"]:
            assert _centre_error(t["view"], _grid_box(t["analysis"]),
                                 shown) < 0.5


def test_the_text_layer_decides_before_the_model_is_asked(retry_on):
    upright, _s = _sheet(text_rotate=0)
    engine = SaysTurned()
    _page(upright, engine)
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[0]
    turned, _s = _sheet(text_rotate=90)
    engine = SaysTurned()
    out = _page(turned, engine)
    assert len(engine.prompts) == 1 and "text layer" in out["turned"]
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[0]
    # nor when the agent asked for a turn itself
    strokes, _s = _sheet(text_rotate=90, strokes=True)
    engine = SaysTurned()
    _zoom(strokes, engine, bbox=[60, 440, 200, 720], pad_frac=0, turn=0)
    assert len(engine.prompts) == 1
    assert vision_view.TURN_TAG_INSTRUCTION not in engine.prompts[0]


def test_a_second_look_that_fails_leaves_the_first(retry_on):
    pdf, _shown = _sheet(text_rotate=90, strokes=True)

    class FailsSecond(SaysTurned):
        def analyze_image(self, image_bytes, prompt=""):
            if self.prompts:
                raise RuntimeError("the endpoint went away")
            return super().analyze_image(image_bytes, prompt)

    out = _zoom(pdf, FailsSecond(), bbox=[60, 440, 200, 720], pad_frac=0)
    assert "error" not in out
    assert "reading it again upright failed" in out["turn_retry_failed"]
    assert "TURNED" not in out["analysis"] and "turned" not in out


@pytest.mark.parametrize("line,turn", [
    ("TURNED: bottom-to-top", 90), ("**TURNED:** top to bottom.", 270),
    ("- TURNED: upside-down", 180), ("Turned: reads upward", 90),
    ("TURNED: left-to-right", None), ("TURNED: none", None)])
def test_the_turned_line_is_read_and_dropped(line, turn):
    text, got = vision_view.read_turn_tag(f"MARK [1, 2, 3, 4]\n{line}")
    assert got == turn and text == "MARK [1, 2, 3, 4]"
    assert vision_view.read_turn_tag("the sheet is turned") == \
        ("the sheet is turned", None)
