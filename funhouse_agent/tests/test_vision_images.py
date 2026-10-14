"""Zoom and tiles for image files: uploads and pasted screenshots (BACKLOG
VIS-09).

A PDF page could be looked at whole, tiled, and zoomed on a box; an image
file (a screenshot, a photo of a log, a scan saved as PNG) went to the model
once, whole - shrunk by the host when it was larger than the model reads -
with no zoom, no tiles and no boxes the agent could act on. Now:

* ``analyze_image`` gives the image's ``view`` in its own pixels and its
  boxes as 0-999 boxes on it, and an image too large for one look is ALSO
  read in tiles cut from the original;
* ``render_region`` takes the image as ``attachment_key``: ``bbox`` /
  ``view`` + ``image_box`` / ``turn`` / ``reuse`` mean what they mean on a
  page, in the image's pixels, and the window is cut from the ORIGINAL.

The stand-in engines SEE: they find a red mark and a blue-framed label in
the image they are sent and answer in its pixels. The label's "lettering"
is nine one-pixel bars a pixel apart: the engine reads it ("CODE-9") only
when every bar arrives as its own dark stroke, which a shrunk copy cannot
give (measured 2026-10-14: none of the nine survive at 0.53x, all nine at
1x and at 2x-4x).
"""

from __future__ import annotations

import io
import json
import re

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageOps  # noqa: E402

from funhouse_agent import (  # noqa: E402
    image_view, review_flags, vision_probe, vision_tools, vision_view,
)
from funhouse_agent.deep import vision_engine as _vision_engine  # noqa: E402,F401
from funhouse_agent.vision_tools import dispatch_extended_tool  # noqa: E402

#: The red mark on the screenshot (its own pixels).
MARK = (1000, 1500, 1040, 1520)
#: The blue frame round the fine print, top-right corner.
LABEL = (3600, 40, 3700, 80)


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
    monkeypatch.setenv(vision_tools.REPEAT_CACHE_ENV, "0")
    vision_probe.clear_cache()
    vision_tools.clear_turn_memory()
    yield
    vision_probe.clear_cache()
    vision_tools.clear_turn_memory()


def _screenshot(width=3840, height=2160, mark=MARK, label=LABEL,
                bar=1, gap=1, fmt="PNG", **save):
    """A white screenshot with a red mark and, in a blue frame, nine dark
    bars ``bar`` px wide, ``gap`` px apart: the fine print."""
    a = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = mark
    a[y0:y1, x0:x1] = (255, 0, 0)
    if label is not None:
        lx0, ly0, lx1, ly1 = label
        a[ly0:ly0 + 3, lx0:lx1] = a[ly1 - 3:ly1, lx0:lx1] = (0, 0, 255)
        a[ly0:ly1, lx0:lx0 + 3] = a[ly0:ly1, lx1 - 3:lx1] = (0, 0, 255)
        for i in range(9):
            x = lx0 + 10 + i * (bar + gap)
            a[ly0 + 10:ly1 - 10, x:x + bar] = 0
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format=fmt, **save)
    return buf.getvalue()


def _pixels(image_bytes):
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def _box(mask):
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _runs(row):
    dark = row.max(axis=1) < 90
    return int(np.sum(dark[1:] & ~dark[:-1]) + (1 if dark[0] else 0))


class Reader:
    """Answers in pixels of the image it is sent: the red mark, and the
    blue-framed label - read only when its bars arrive one by one."""

    accepts_jpeg = False

    def __init__(self):
        self.prompts, self.images = [], []

    def analyze_image(self, image_bytes, prompt=""):
        self.prompts.append(prompt)
        self.images.append(bytes(image_bytes))
        a = _pixels(image_bytes)
        lines = []
        red = (a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
        if red.any():
            b = _box(red)
            lines.append(f"MARK px=[{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
        blue = (a[:, :, 2] > 150) & (a[:, :, 0] < 110) & (a[:, :, 1] < 110)
        if blue.sum() > 20:
            b = _box(blue)
            inside = a[b[1]:b[3], b[0]:b[2]]
            mid = inside[inside.shape[0] // 2]
            edge = max(2, (b[2] - b[0]) // 20)
            n = _runs(mid[edge:-edge])
            said = (f"LABEL reads CODE-{n}" if n == 9 else
                    "LABEL lettering too small to read")
            lines.append(f"{said} px=[{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
        return "\n".join(lines) or "nothing here"


class JpegReader(Reader):
    accepts_jpeg = True


def _grid(text, word):
    """The 0-999 box on the line of ``text`` that names ``word``."""
    for line in str(text).splitlines():
        if word in line:
            m = re.search(r"\[(\d+), (\d+), (\d+), (\d+)\]", line)
            if m:
                return [int(v) for v in m.groups()]
    raise AssertionError(f"no {word} box in {text!r}")


def _err(view, grid, truth):
    """Worst corner error, in the image's pixels, of a 0-999 box on
    ``view`` against ``truth``."""
    b = vision_view.image_box_to_page(view, grid)
    return max(abs(b[i] - truth[i]) for i in range(4))


def _look(engine, data, name="shot.png", **args):
    return json.loads(dispatch_extended_tool(
        "analyze_image", {"attachment_key": name, "prompt": "What is here?",
                          **args}, engine, {name: data}))


def _zoom(engine, data, name="shot.png", **args):
    return json.loads(dispatch_extended_tool(
        "render_region", {"attachment_key": name, "prompt": "Read the label",
                          **args}, engine, {name: data}))


# -- the arithmetic ---------------------------------------------------------------

def test_a_crop_is_enlarged_by_whole_times_only():
    # openai-high: 2048 px, 2,500 patches of 32 px
    assert image_view.sent_size(828, 462, enlarge=True) == (1656, 924)
    assert image_view.sent_size(828, 462) == (828, 462)
    assert image_view.sent_size(20, 10, enlarge=True) == (80, 40)   # cap 4x
    # just under twice the budget holds: once, never 1.4 times
    assert image_view.sent_size(1400, 900, enlarge=True) == (1400, 900)
    w, h = image_view.sent_size(3840, 2160, enlarge=True)
    assert (w, h) == (2048, 1152)                       # shrunk to fit


@pytest.mark.parametrize("size, n", [((1920, 1080), 1), ((2560, 1440), 2),
                                     ((3840, 2160), 2), ((4032, 3024), 3),
                                     ((5120, 2880), 3), ((12000, 9000), 4)])
def test_tiles_per_side_follow_how_far_the_whole_image_shrinks(size, n):
    pic = image_view.Picture(None, size[0], size[1], "png", b"", None, True)
    assert image_view.tile_count(pic) == n


# -- (1) fine print in a corner: read only through a zoom ---------------------------

def test_fine_print_in_a_corner_is_read_only_through_a_zoom():
    shot, engine = _screenshot(), Reader()
    whole = _look(engine, shot, tiles="off")
    assert whole["view"] == [0, 0, 3840, 2160]
    assert whole["image_px"] == [3840, 2160]
    assert whole["view_px"] == [2048, 1152] and whole["sent_scale"] < 0.6
    assert "CODE-9" not in whole["analysis"]
    assert "too small" in whole["analysis"]
    assert "Zoom with render_region" in whole["shrunk_note"]
    # the agent zooms on the box the look gave, as the result says to
    box = _grid(whole["analysis"], "LABEL")
    zoom = _zoom(engine, shot, view=whole["view"], image_box=box)
    assert "error" not in zoom, zoom
    assert "CODE-9" in zoom["analysis"]
    vx0, vy0, vx1, vy1 = zoom["view"]
    assert vx0 <= LABEL[0] and vy0 <= LABEL[1] and vx1 >= LABEL[2] \
        and vy1 >= LABEL[3]
    assert zoom["sent_scale"] >= 2          # enlarged whole times, not shrunk
    assert zoom["window_padding_px"][0] >= 12
    # cut from the ORIGINAL: every bar is a stroke of whole sent pixels
    a = _pixels(engine.images[-1])
    k = int(zoom["sent_scale"])
    row = a[(LABEL[1] + 20 - vy0) * k + k // 2,
            (LABEL[0] + 5 - vx0) * k:(LABEL[2] - 5 - vx0) * k]
    assert _runs(row) == 9
    # and its box is the label's, in the image's own pixels
    assert _err(zoom["view"], _grid(zoom["analysis"], "LABEL"), LABEL) <= 2


def test_a_zoom_on_a_plain_bbox_in_the_images_pixels():
    shot, engine = _screenshot(), Reader()
    out = _zoom(engine, shot, bbox=list(LABEL))
    assert "CODE-9" in out["analysis"] and out["bbox"] == list(LABEL)
    assert "own pixels" in out["frame"]
    assert _err(out["view"], _grid(out["analysis"], "LABEL"), LABEL) <= 1


# -- (2) boxes round-trip to the original's pixels --------------------------------

@pytest.mark.parametrize("engine_cls, fmt", [(Reader, "PNG"),
                                             (JpegReader, "JPEG")])
def test_boxes_round_trip_to_the_original_pixels(engine_cls, fmt):
    shot, engine = _screenshot(fmt=fmt, label=None), engine_cls()
    whole = _look(engine, shot, tiles="off")
    box = _grid(whole["analysis"], "MARK")
    assert _err(whole["view"], box, MARK) <= 4      # one grid step is 3.8 px
    zoom = _zoom(engine, shot, view=whole["view"], image_box=box)
    assert _err(zoom["view"], _grid(zoom["analysis"], "MARK"), MARK) <= 1.5
    assert zoom["nearest_box_from_aim_px"] <= 4
    if fmt == "JPEG":                       # a photo's crop stays a JPEG
        assert vision_view.image_media_type(engine.images[-1]) == "image/jpeg"


def test_a_photo_turned_by_its_exif_tag_is_framed_as_a_viewer_shows_it():
    a = np.full((200, 300, 3), 255, np.uint8)
    a[10:30, 10:40] = (255, 0, 0)
    im = Image.fromarray(a)
    exif = im.getexif()
    exif[0x0112] = 6                        # shown turned 90 degrees clockwise
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)
    shown = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(
        buf.getvalue()))))
    truth = _box((shown[:, :, 0] > 200) & (shown[:, :, 1] < 80))
    engine = Reader()
    out = _look(engine, buf.getvalue(), name="photo.jpg")
    assert out["image_px"] == [200, 300] and out["view"] == [0, 0, 200, 300]
    assert "EXIF" in out["image_note"]
    assert _err(out["view"], _grid(out["analysis"], "MARK"), truth) <= 2


# -- (3) a turned zoom -------------------------------------------------------------

@pytest.mark.parametrize("turn", [90, 180, 270])
def test_a_turned_zoom_comes_back_in_the_images_own_frame(turn):
    shot, engine = _screenshot(label=None), Reader()
    window = [960, 1470, 1120, 1530]        # 160 x 60, the mark left of centre
    plain = _zoom(engine, shot, bbox=window, pad_frac=0)
    out = _zoom(engine, shot, bbox=window, pad_frac=0, turn=turn)
    assert out["view"] == plain["view"]
    sent = _pixels(engine.images[-1])
    w, h = plain["view_px"]
    if turn == 180:
        assert sent.shape[:2] == (h, w)
    else:
        assert sent.shape[:2] == (w, h)     # a quarter turn swaps the sides
    assert f"turned {turn} degrees clockwise" in out["turned"]
    assert "as the image shows it" in out["turned"]
    assert _err(out["view"], _grid(out["analysis"], "MARK"), MARK) <= 1
    # the image sent is the plain zoom turned clockwise, pixel for pixel
    unturned = _pixels(engine.images[-2])
    assert np.array_equal(np.rot90(unturned, k=-(turn // 90)), sent)


def test_a_bad_turn_is_refused_before_any_look():
    engine = Reader()
    for tool in (_look, _zoom):
        out = tool(engine, _screenshot(1200, 800, label=None), turn=45)
        assert "quarter turn" in out["error"]
    assert engine.images == []


# -- (4) tiles cover a large image -------------------------------------------------

def test_tiles_cover_a_large_image_and_read_what_the_whole_look_cannot():
    W, H = 5120, 2880
    label = (4900, 2700, 5000, 2740)        # 2-px bars, bottom-right corner
    shot = _screenshot(W, H, label=label, bar=2, gap=2)
    engine = Reader()
    out = _look(engine, shot)               # tiles="auto"
    assert out["sent_scale"] == pytest.approx(0.4, abs=0.01)
    assert "CODE-9" not in out["analysis"]
    tiles = out["tiles"]
    assert len(tiles) == 9 and "3x3" in out["tiling"]
    # every pixel of the image is in a tile, and the tiles overlap
    for axis, size in ((0, W), (1, H)):
        spans = sorted({(t["view"][axis], t["view"][axis + 2])
                        for t in tiles})
        assert spans[0][0] == 0 and spans[-1][1] == size
        assert all(a[1] > b[0] for a, b in zip(spans, spans[1:]))
    bud = vision_view.budget()
    for t in tiles:
        assert bud.fits(*t["view_px"])
        assert "sent_scale" not in t        # each tile at its own resolution
    # the corner tile reads the label the whole look saw but could not read;
    # the merged list keeps the tile's reading and its box
    corner = [t for t in tiles if "CODE-9" in t["analysis"]]
    assert [t["tile"] for t in corner] == ["r3c3"]
    found = [f for f in out["found"] if f["what"].startswith("LABEL reads")]
    assert found and found[0]["seen_in"] == ["image", "r3c3"]
    assert max(abs(found[0]["bbox"][i] - label[i]) for i in range(4)) <= 2
    assert "box in the image's own pixels" in out["found_note"]
    assert _err(corner[0]["view"], _grid(corner[0]["analysis"], "LABEL"),
                label) <= 1
    # the whole image was asked for the layout only
    assert "WHOLE picture" in engine.prompts[0]
    assert "LAYOUT" in engine.prompts[0]


def test_tiles_can_be_asked_for_or_switched_off():
    shot = _screenshot(1600, 900, mark=(100, 100, 140, 120), label=None)
    engine = Reader()
    out = _look(engine, shot)
    assert "tiles" not in out and len(engine.images) == 1   # it fits
    out = _look(engine, shot, tiles="2")
    assert len(out["tiles"]) == 4 and len(engine.images) == 1 + 5
    assert "error" in _look(engine, shot, tiles="6 by 6")
    big = _screenshot(label=None)
    out = _look(Reader(), big, tiles="off")
    assert "tiles" not in out and out["sent_scale"] < 1


def test_the_efficient_policy_does_not_tile(monkeypatch):
    monkeypatch.setenv(vision_view.POLICY_ENV, "efficient")
    out = _look(Reader(), _screenshot(label=None))
    assert "tiles" not in out and "shrunk_note" in out


# -- (5) the same reading rules as the page looks ----------------------------------

def test_every_image_look_carries_the_page_looks_reading_rules():
    engine = Reader()
    shot = _screenshot(label=None)
    _look(engine, shot)                     # whole + 4 tiles
    _zoom(engine, shot, bbox=list(MARK))
    assert len(engine.prompts) == 6
    for prompt, image in zip(engine.prompts, engine.images):
        h, w = _pixels(image).shape[:2]
        assert vision_view.READING_INSTRUCTION in prompt
        assert vision_view.FRAME_INSTRUCTION in prompt
        assert "(cut)" in prompt
        assert f"This image is {w} x {h} pixels" in prompt


# -- what stays as it was ------------------------------------------------------------

def test_an_image_that_fits_goes_as_it_is_with_its_frame():
    shot = _screenshot(800, 600, mark=(100, 100, 140, 120), label=None)
    engine = Reader()
    out = _look(engine, shot)
    assert engine.images == [shot]
    assert out["view"] == [0, 0, 800, 600] and out["view_px"] == [800, 600]
    assert "sent_scale" not in out and "shrunk_note" not in out
    assert "full resolution" in out["precision"]
    assert _err(out["view"], _grid(out["analysis"], "MARK"),
                (100, 100, 140, 120)) <= 1


def test_a_page_look_at_an_image_file_is_in_the_images_pixels_too():
    """PyMuPDF opens a PNG as a page in points at 96 dpi; a page look's view
    in points, handed to the image zoom (pixels), would land at 0.75 of the
    spot. Every look at an image is in its own pixels."""
    shot, engine = _screenshot(label=None), Reader()
    out = json.loads(dispatch_extended_tool(
        "analyze_pdf_page", {"attachment_key": "shot.png", "page": 0,
                             "prompt": "find the mark"},
        engine, {"shot.png": shot}))
    assert out["view"] == [0, 0, 3840, 2160] and "own pixels" in \
        out["image_file"]
    box = _grid(out["analysis"], "MARK")
    zoom = _zoom(engine, shot, view=out["view"], image_box=box)
    assert _err(zoom["view"], _grid(zoom["analysis"], "MARK"), MARK) <= 1.5
    from funhouse_agent import inline_store
    inline_store.clear()
    try:
        out = json.loads(dispatch_extended_tool(
            "analyze_pdf_page", {"attachment_key": "shot.png",
                                 "prompt": "look", "_inline": True},
            engine, {"shot.png": shot}))
        _data, meta = inline_store.get(out["image_id"])
        assert meta["view"] == [0, 0, 3840, 2160] == out["view"]
    finally:
        inline_store.clear()


def test_a_pdf_still_zooms_as_a_page():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(width=612, height=792).draw_rect(
        fitz.Rect(100, 100, 130, 112), color=(1, 0, 0), fill=(1, 0, 0))
    pdf = doc.tobytes()
    doc.close()
    out = json.loads(dispatch_extended_tool(
        "render_region", {"attachment_key": "s.pdf", "page": 0,
                          "bbox": [80, 90, 150, 120], "prompt": "what?"},
        Reader(), {"s.pdf": pdf}))
    assert out["page"] == 0 and "image_px" not in out and "frame" not in out


# -- the zoom's other options on an image --------------------------------------------

def test_page_dpi_and_a_box_off_the_image_are_said():
    shot, engine = _screenshot(1200, 800, mark=(100, 100, 140, 120),
                               label=None), Reader()
    out = _zoom(engine, shot, bbox=[100, 100, 140, 120], page=3, dpi=300)
    assert "not used" in out["page_note"] and "dpi" in out["dpi_note"]
    out = _zoom(engine, shot, bbox=[5000, 100, 5100, 120])
    assert "outside" in out["outside_note"]
    assert out["view"][0] == 0 and out["view"][2] == 1200
    out = _zoom(engine, shot, bbox="near the top")
    assert "error" in out


def test_a_zoom_on_one_character_still_shows_its_neighbours():
    shot = _screenshot(1200, 800, mark=(500, 400, 504, 404), label=None)
    out = _zoom(Reader(), shot, bbox=[500, 400, 504, 404])
    vx0, vy0, vx1, vy1 = out["view"]
    assert vx1 - vx0 >= image_view.MIN_WINDOW_PX
    assert vy1 - vy0 >= image_view.MIN_WINDOW_PX


def test_marks_are_drawn_where_they_are_in_the_images_pixels():
    shot = _screenshot(1200, 800, mark=(10, 10, 20, 20), label=None)
    engine = Reader()
    out = _zoom(engine, shot, bbox=[400, 300, 600, 400],
                marks=[[500, 350, "1"]])
    assert "error" not in out
    a = _pixels(engine.images[-1])
    red = (a[:, :, 0] > 200) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    x0, y0, x1, y1 = _box(red)
    k = out["sent_scale"]
    cx = (500 - out["view"][0]) * k
    cy = (350 - out["view"][1]) * k
    assert x0 <= cx <= x1 and y0 <= cy <= y1


def test_reuse_hands_back_a_kept_zoom_of_an_image(tmp_path):
    from funhouse_agent import _fileio
    work = tmp_path / "alice" / "files"
    work.mkdir(parents=True)
    shot, engine = _screenshot(label=None), Reader()
    with _fileio.working_dir_bound(str(work)):
        first = _zoom(engine, shot, bbox=list(MARK), reuse=True)
        assert "no earlier zoom" in first["reuse_note"]
        again = _zoom(engine, shot, bbox=list(MARK), reuse=True)
        # the kept zoom's own box, as the host's per-turn note offers it
        kept = vision_tools.readings_for_conversation()[-1]["region"]
        back = _zoom(engine, shot, bbox=kept, pdf_page=1, reuse=True)
    assert len(engine.images) == 1 and "earlier zoom" in again["reused"]
    assert "earlier zoom" in back["reused"]
    assert kept == first["view"]


def test_an_inline_zoom_on_an_image_is_shown_with_its_view():
    from funhouse_agent import inline_store
    inline_store.clear()
    try:
        engine = Reader()
        out = _zoom(engine, _screenshot(label=None), bbox=list(MARK),
                    _inline=True)
        assert engine.images == []
        data, meta = inline_store.get(out["image_id"])
        assert out["view"] == meta["view"] and "own pixels" in meta["label"]
        assert _pixels(data).shape[1] == out["view_px"][0]
    finally:
        inline_store.clear()


class Locates(Reader):
    """Ends with the structured LOCATED line when asked for it."""

    def analyze_image(self, image_bytes, prompt=""):
        text = super().analyze_image(image_bytes, prompt)
        if "LOCATED:" not in prompt:
            return text
        m = re.search(r"MARK px=\[(\d+), (\d+), (\d+), (\d+)\]", text)
        items = ([{"what": "mark", "px": [int(v) for v in m.groups()],
                   "sure": True}] if m else [])
        return text + "\nLOCATED: " + json.dumps(items)


def test_structured_locations_on_an_image_are_in_its_own_pixels(monkeypatch):
    monkeypatch.setenv(review_flags.VISION_STRUCTURED_ENV, "1")
    out = _look(Locates(), _screenshot(label=None), tiles="off")
    (item,) = out["located"]
    assert "page_bbox" not in item
    assert max(abs(item["bbox"][i] - MARK[i]) for i in range(4)) <= 3
    assert "zoom_bbox" in item and "own pixels" in out["located_note"]


class SaysTurned(Reader):
    """Asked for the TURNED line, says its lettering runs up the image
    while the image it is sent is wider than tall."""

    def analyze_image(self, image_bytes, prompt=""):
        text = super().analyze_image(image_bytes, prompt)
        h, w = _pixels(image_bytes).shape[:2]
        if vision_view.TURN_TAG_INSTRUCTION in prompt and w > h:
            return text + "\nTURNED: bottom-to-top"
        return text


def test_the_switched_turned_reread_works_on_an_image_zoom(monkeypatch):
    monkeypatch.setenv(review_flags.VISION_TURN_RETRY_ENV, "1")
    engine = SaysTurned()
    out = _zoom(engine, _screenshot(label=None), bbox=[960, 1470, 1120, 1530],
                pad_frac=0)
    assert len(engine.images) == 2
    first, second = (_pixels(i).shape[:2] for i in engine.images)
    assert first[1] > first[0] and second[0] > second[1]
    assert "second reading" in out["turned"]
    assert "TURNED" not in out["analysis"]
    assert _err(out["view"], _grid(out["analysis"], "MARK"), MARK) <= 1


def test_a_png_that_cannot_be_decoded_still_goes_whole_but_is_not_cut():
    broken = b"\x89PNG\r\n\x1a\n not really an image"
    engine = Reader.__new__(Reader)
    engine.prompts, engine.images = [], []
    engine.analyze_image = lambda img, prompt="": (
        engine.images.append(bytes(img)) or "seen")
    out = _look(engine, broken, name="odd.png")
    assert out["analysis"] == "seen" and engine.images == [broken]
    out = _zoom(engine, broken, name="odd.png", bbox=[0, 0, 10, 10])
    assert "cannot be zoomed" in out["error"] and len(engine.images) == 1


def test_a_16_bit_grey_scan_is_zoomed_with_its_ink_kept():
    a = np.full((600, 800), 60000, np.uint16)
    a[300:320, 400:440] = 2000                  # a dark stroke
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    engine = Reader()
    out = _zoom(engine, buf.getvalue(), name="scan.png",
                bbox=[380, 280, 460, 340])
    assert "error" not in out, out
    sent = _pixels(engine.images[-1])
    assert sent.min() < 40 and sent.max() > 220   # not clipped to white


def test_what_is_not_an_image_or_a_pdf_is_still_refused():
    engine = Reader()
    out = _look(engine, b"just some notes", name="notes.txt")
    assert "read_text_file" in out["error"] and engine.images == []


# -- described in the tools' own descriptions only ---------------------------------

def test_the_image_zoom_is_offered_in_the_tools_own_descriptions_only():
    from funhouse_agent.deep import prompt as deep_prompt
    from funhouse_agent.deep.agent import _review_page_descriptions
    from funhouse_agent.deep.review_agent import (
        INLINE_DESCRIPTIONS, REVIEW_DESCRIPTIONS)
    from funhouse_agent.deep.tools import make_vision_tools
    from funhouse_agent.native_tools import OPENAI_TOOLS
    zoom, look = vision_tools.IMAGE_ZOOM_NOTE, vision_tools.IMAGE_LOOK_NOTE
    tools = {t.name: t for t in make_vision_tools(engine=None)}
    # the geotech page and the review page's default build
    assert zoom in tools["render_region"].description
    assert zoom in _review_page_descriptions()["render_region"]
    assert look in tools["analyze_image"].description
    assert {"tiles", "turn"} <= set(tools["analyze_image"].args)
    # the lean build, with and without inline images
    assert zoom in REVIEW_DESCRIPTIONS["render_region"]
    assert zoom in INLINE_DESCRIPTIONS["render_region"]
    assert look in REVIEW_DESCRIPTIONS["analyze_image"]
    assert look in INLINE_DESCRIPTIONS["analyze_image"]
    # the native schemas
    schemas = {t["function"]["name"]: t["function"] for t in OPENAI_TOOLS}
    assert zoom in schemas["render_region"]["description"]
    assert {"tiles", "turn"} <= set(
        schemas["analyze_image"]["parameters"]["properties"])
    for text in (deep_prompt.DOCUMENT_REVIEW_PROMPT,
                 deep_prompt.DOCUMENT_REVIEW_PROMPT_LEAN,
                 deep_prompt.build_domain_prompt()):
        assert zoom not in text and look not in text
        assert "own pixels" not in text


def test_the_tools_take_the_new_arguments_end_to_end():
    from funhouse_agent.deep.tools import make_vision_tools
    engine = Reader()
    shot = _screenshot(label=None)
    tools = {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"shot.png": shot})}
    out = json.loads(tools["analyze_image"].invoke(
        {"attachment_key": "shot.png", "prompt": "what?", "tiles": "off",
         "turn": 90}))
    assert "tiles" not in out and "turned 90" in out["turned"]
    out = json.loads(tools["render_region"].invoke(
        {"attachment_key": "shot.png", "bbox": list(MARK), "prompt": "?"}))
    assert _err(out["view"], _grid(out["analysis"], "MARK"), MARK) <= 1
