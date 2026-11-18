"""BACKLOG VIS-33: on ``tiles='auto'`` a page of photographs takes one look,
behind ``GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK`` (off by default).

A page with almost no text layer is assumed to carry drawing lettering and
was read whole AND in tiles: five or more vision calls on a page of
photographs, whose readings ran several times longer than one look that
answered the same questions. With the switch on, a page of photographs
takes one look under ``auto`` and the result says tiles or a zoom can be
asked for. A page counts only when planlens names it a page of photographs
(page role ``photos``) AND the page shows it by itself
(``vision_tools._not_photographs``): the role alone is named from one phrase
and one picture, and drawings pass it. Every drawing below that the role
alone lets through is tiled with the switch on: a drawing that lost its
tiles would be the serious failure, a page of photographs still tiled only
costs calls. Off, every read is byte-for-byte what it was.

Made-up pages built here and a fake vision model: no model, no network.
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")
np = pytest.importorskip("numpy")
pytest.importorskip("planlens.document.roles")

from aec_agent_core import review_flags, vision_tools, vision_view  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402

ENV = review_flags.PHOTO_PAGE_ONE_LOOK_ENV
TILE = re.compile(r"tile row (\d) of \d, column (\d)")
#: A D-size drawing sheet, in points.
D_SHEET = (2592, 1728)
TABLOID = (1224, 792)
FILLER = ("made", "up", "note", "text", "for", "this", "sheet", "only")


# -- made-up pictures ------------------------------------------------------------

def _photo(seed: int, w: int = 320, h: int = 200, grey: bool = False):
    """A made-up photograph: smooth colour with noise, many tones."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // w), (yy * 255 // h),
                     ((xx + yy) * 255 // (w + h))], axis=2)
    a = np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)
    if grey:
        g = a.mean(axis=2).astype(np.uint8)
        return pymupdf.Pixmap(pymupdf.csGRAY, w, h, g.tobytes(), False)
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, a.tobytes(), False)


def _logo(w: int = 80, h: int = 40):
    """A made-up logo: two flat colours on white."""
    a = np.full((h, w, 3), 255, dtype=np.uint8)
    a[h // 10:h - h // 10, w // 20:w // 2 - w // 20] = (20, 60, 160)
    a[h // 3:h - h // 3, w // 2 + w // 20:w - w // 20] = (200, 30, 30)
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, a.tobytes(), False)


def _scan_array(w: int = 1700, h: int = 1100):
    """A made-up scanned drawing: black lines on white paper."""
    a = np.full((h, w), 255, dtype=np.uint8)
    for x in range(100, w - 100, 150):
        a[100:h - 100, x:x + 3] = 0
    for y in range(100, h - 100, 120):
        a[y:y + 3, 100:w - 100] = 0
    a[h - 220:h - 100, w - 600:w - 100] = 0        # a title block's frame
    a[h - 217:h - 103, w - 597:w - 103] = 255
    return a


def _grey(a):
    return pymupdf.Pixmap(pymupdf.csGRAY, a.shape[1], a.shape[0],
                          np.ascontiguousarray(a).tobytes(), False)


def _paper_toned_scan():
    """The scan on yellowed paper with brown ink and a little noise."""
    a = _scan_array(800, 520).astype(float) / 255.0
    paper, ink = np.array([235, 225, 190]), np.array([60, 50, 40])
    rgb = ink + (paper - ink) * a[:, :, None]
    rng = np.random.default_rng(7)
    rgb = np.clip(rgb + rng.integers(-2, 3, rgb.shape), 0, 255)
    rgb = rgb.astype(np.uint8)
    return pymupdf.Pixmap(pymupdf.csRGB, rgb.shape[1], rgb.shape[0],
                          rgb.tobytes(), False)


# -- made-up pages ---------------------------------------------------------------

def _pdf(build, size=(612, 792), rotate: int = 0) -> bytes:
    d = pymupdf.open()
    page = d.new_page(width=size[0], height=size[1])
    build(page)
    if rotate:
        page.set_rotation(rotate)
    data = d.tobytes()
    d.close()
    return data


def _lines(page, segments, color=(0, 0, 0)):
    """Each segment ``(x0, y0, x1, y1)`` drawn as a path of its own."""
    shape = page.new_shape()
    for x0, y0, x1, y1 in segments:
        shape.draw_line((x0, y0), (x1, y1))
        shape.finish(color=color, width=0.7)
    shape.commit()


def _filler(page, n, x, y, size=6.0, per_line=12, render_mode=0):
    """``n`` made-up words in lines from ``(x, y)``."""
    words = [FILLER[i % len(FILLER)] for i in range(n)]
    for k in range(0, n, per_line):
        page.insert_text((x, y), " ".join(words[k:k + per_line]),
                         fontsize=size, render_mode=render_mode)
        y += size * 1.6


def _title_block(page, *titles, logo=False):
    """A drawing sheet's frame and title block (bottom right, in the
    page's bottom band), its title lines at 14 pt; with ``logo`` a 40 x 20
    logo in its corner."""
    W, H = page.rect.width, page.rect.height
    page.draw_rect(pymupdf.Rect(36, 36, W - 36, H - 36), color=(0, 0, 0))
    block = pymupdf.Rect(W - 760, H - 260, W - 36, H - 36)
    page.draw_rect(block, color=(0, 0, 0))
    y = block.y0 + 40
    for t in titles:
        page.insert_text((block.x0 + 20, y), t, fontsize=14)
        y += 30
    if logo:
        page.insert_image(pymupdf.Rect(block.x1 - 60, block.y1 - 40,
                                       block.x1 - 20, block.y1 - 20),
                          pixmap=_logo())


# Pages of photographs: one look with the switch on.

def _photo_page_pdf(rotate: int = 0, label_over: bool = False) -> bytes:
    """A report page of two photographs, each with a short caption: almost
    no text layer, so ``auto`` would assume drawing lettering and tile it."""
    def build(page):
        page.insert_image(pymupdf.Rect(72, 60, 540, 350), pixmap=_photo(1))
        page.insert_text((72, 370), "Photo 1", fontsize=10)
        page.insert_image(pymupdf.Rect(72, 420, 540, 710), pixmap=_photo(2))
        page.insert_text((72, 730), "Photo 2", fontsize=10)
        if label_over:
            page.insert_text((250, 200), "CRACK", fontsize=10)
    return _pdf(build, rotate=rotate)


def _photo_and_sketch_pdf() -> bytes:
    """A photograph beside a dimensioned sketch of what it shows."""
    def build(page):
        page.insert_text((40, 50), "PHOTO 3 AND SKETCH OF EXPOSED FOOTING",
                         fontsize=14)
        page.insert_image(pymupdf.Rect(40, 90, 440, 390),
                          pixmap=_photo(3, 400, 300))
        page.insert_text((40, 410), "Photo 3: made-up footing, view looking "
                                    "east", fontsize=9)
        page.draw_rect(pymupdf.Rect(520, 200, 720, 300))        # footing
        page.draw_rect(pymupdf.Rect(600, 140, 640, 200))        # column
        _lines(page, [(520, 330, 720, 330), (520, 305, 520, 340),
                      (720, 305, 720, 340), (750, 200, 750, 300),
                      (725, 200, 760, 200), (725, 300, 760, 300),
                      (600, 120, 640, 120), (600, 115, 600, 138),
                      (640, 115, 640, 138)])
        page.insert_text((600, 344), "1.8 m", fontsize=4)
        page.insert_text((754, 255), "0.6 m", fontsize=4)
        page.insert_text((606, 112), "0.4 m", fontsize=4)
        page.insert_text((520, 380), "Sketch: plan of the footing (made up)",
                         fontsize=4)
    return _pdf(build, size=(792, 612))


# The seven drawings that planlens' role alone names pages of photographs.

def _plate_plan_pdf() -> bytes:
    """A location plan titled PLATE NO. 1 over an aerial base photograph,
    with boring symbols and notes: 155 words."""
    def build(page):
        page.insert_image(pymupdf.Rect(100, 100, 1900, 1380),
                          pixmap=_photo(11, 640, 456))
        _title_block(page, "BORING LOCATION PLAN", "PLATE NO. 1")
        for k in range(8):
            x, y = 300 + 200 * k, 300 + 110 * k
            page.draw_circle((x, y), 8, color=(1, 0, 0))
            page.insert_text((x + 12, y + 3), f"B-{k + 1}", fontsize=6)
        _filler(page, 155 - 14, 1950, 300)
    return _pdf(build, D_SHEET)


def _section_pdf() -> bytes:
    """A cross-section titled LOOKING NORTH, with a logo."""
    def build(page):
        _title_block(page, "CROSS SECTION A-A", "LOOKING NORTH", logo=True)
        ground = [(200 + 100 * k, 600 + 40 * ((k * 7) % 5))
                  for k in range(17)]
        _lines(page, [(*a, *b) for a, b in zip(ground, ground[1:])])
        _lines(page, [(x, y, x, y + 500) for x, y in ground[2::4]])
        _lines(page, [(150, y, 1950, y) for y in range(500, 1300, 100)],
               color=(0.6, 0.6, 0.6))
        for k, y in enumerate(range(500, 1300, 100)):
            page.insert_text((110, y + 2), f"EL {120 - 5 * k}", fontsize=6)
    return _pdf(build, D_SHEET)


def _stockpile_pdf() -> bytes:
    """A grading and stockpile plan (contours), with a logo."""
    def build(page):
        _title_block(page, "GRADING AND STOCKPILE PLAN", logo=True)
        segs = []
        for ring in range(10):
            r = 60 + 45 * ring
            pts = [(1000 + r * math.cos(t * math.pi / 12),
                    700 + 0.6 * r * math.sin(t * math.pi / 12))
                   for t in range(25)]
            segs += [(*a, *b) for a, b in zip(pts, pts[1:])]
        _lines(page, segs)
    return _pdf(build, D_SHEET)


def _foundation_pdf() -> bytes:
    """A foundation plan with a photograph inset, captioned PHOTO 1."""
    def build(page):
        _title_block(page, "FOUNDATION PLAN")
        cols, rows = range(300, 1900, 400), range(300, 1300, 300)
        _lines(page, [(x, 200, x, 1300) for x in cols]
               + [(200, y, 2000, y) for y in rows], color=(0.5, 0.5, 0.5))
        for x in cols:
            for y in rows:
                page.draw_rect(pymupdf.Rect(x - 30, y - 30, x + 30, y + 30))
                page.insert_text((x + 34, y - 34), "F2", fontsize=6)
        page.insert_image(pymupdf.Rect(2050, 1080, 2450, 1380),
                          pixmap=_photo(12, 400, 300))
        page.insert_text((2050, 1440), "PHOTO 1 EXISTING FOOTING",
                         fontsize=8)
    return _pdf(build, D_SHEET)


def _photo_location_pdf() -> bytes:
    """A photo location plan: arrows and numbers over an aerial image."""
    def build(page):
        page.insert_image(pymupdf.Rect(100, 100, 1800, 1500),
                          pixmap=_photo(13, 640, 527))
        _title_block(page, "PHOTO LOCATION PLAN")
        for k in range(6):
            x, y = 300 + 220 * k, 400 + 120 * k
            _lines(page, [(x, y, x + 60, y - 40)], color=(1, 1, 0))
            page.insert_text((x - 14, y + 4), str(k + 1), fontsize=6)
    return _pdf(build, D_SHEET)


def _profile_pdf() -> bytes:
    """A profile sheet whose top band carries the appendix's running header
    naming photographs, with a logo."""
    def build(page):
        page.insert_text((60, 70), "APPENDIX B SUBSURFACE PROFILES AND SITE "
                                   "PHOTOGRAPHS", fontsize=10)
        _title_block(page, "SUBSURFACE PROFILE 1", logo=True)
        _lines(page, [(200 + 150 * k, 500, 200 + 150 * k, 1200)
                      for k in range(10)])
        for k in range(10):
            x = 200 + 150 * k
            page.insert_text((x - 20, 490), f"B-{k + 1}", fontsize=6)
            for j in range(5):
                page.insert_text((x + 4, 560 + 120 * j), f"EL {110 - 3 * j}",
                                 fontsize=6)
    return _pdf(build, D_SHEET)


def _scan_with_ocr_pdf(phrase: str) -> bytes:
    """A scanned drawing (one page-sized picture) with an invisible OCR
    text layer of 44 words, ``phrase`` in its bottom band."""
    def build(page):
        page.insert_image(page.rect, pixmap=_grey(_scan_array()))
        page.insert_text((900, 770), phrase, fontsize=5, render_mode=3)
        _filler(page, 44 - len(phrase.split()), 120, 200, size=5,
                render_mode=3)
    return _pdf(build, TABLOID)


# Correct as built, and tiled with the switch on too.

def _scanned_drawing_pdf() -> bytes:
    """A made-up scanned drawing: one page-sized picture of black lines on
    white paper, no text layer at all."""
    def build(page):
        page.insert_image(page.rect, pixmap=_grey(_scan_array()))
    return _pdf(build, TABLOID)


def _scan_in_strips_pdf() -> bytes:
    """The scanned drawing stored as three strips, no text layer."""
    a = _scan_array()
    h = a.shape[0] // 3

    def build(page):
        W, H = page.rect.width, page.rect.height
        for k in range(3):
            page.insert_image(pymupdf.Rect(0, k * H / 3, W, (k + 1) * H / 3),
                              pixmap=_grey(a[k * h:(k + 1) * h]))
    return _pdf(build, TABLOID)


def _wordy_photo_sheet_pdf() -> bytes:
    """A sheet titled SITE PHOTOGRAPHS with a photograph and 400 words."""
    def build(page):
        page.insert_text((40, 40), "SITE PHOTOGRAPHS", fontsize=14)
        page.insert_image(pymupdf.Rect(40, 60, 600, 480),
                          pixmap=_photo(4, 400, 300))
        page.insert_text((40, 500), "Photo 1", fontsize=10)
        _filler(page, 400 - 4, 640, 80, size=5, per_line=20)
    return _pdf(build, TABLOID)


def _no_picture_pdf() -> bytes:
    """A photo location plan with no picture on it: line work only."""
    def build(page):
        _title_block(page, "PHOTO LOCATION PLAN")
        _lines(page, [(300 + 220 * k, 400, 360 + 220 * k, 360)
                      for k in range(6)])
    return _pdf(build, D_SHEET)


# -- the fake vision model and the read ----------------------------------------

class Reader:
    """Names what it was shown: a tile, or the whole page."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.labels = []
        self.prompts = {}

    def analyze_image(self, image, prompt=""):
        from aec_agent_core.deep.vision_engine import current_side_call
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "page"
        with self.lock:
            self.calls.append(who)
            # what the run record keeps for this vision call (``vision``)
            self.labels.append(dict(current_side_call() or {}))
            self.prompts[who] = prompt
        return "A photograph of a made-up slope px=[10, 10, 60, 40]"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for env in (ENV, review_flags.TILE_ANSWERS_ENV, vision_view.POLICY_ENV,
                vision_tools.VISION_CAP_ENV):
        monkeypatch.delenv(env, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _read_raw(pdf, engine, **args) -> str:
    raw = dispatch_extended_tool(
        "analyze_pdf_page",
        {"attachment_key": "made_up.pdf", "page": 0,
         "prompt": "What do the photographs show?", **args},
        engine=engine, attachments={"made_up.pdf": pdf})
    vision_tools.clear_repeat_reads()
    return raw


def _read(pdf, engine=None, **args):
    engine = engine or Reader()
    out = json.loads(_read_raw(pdf, engine, **args))
    return out, engine.calls


def _auto_would_tile(pdf) -> int:
    """Tiles per side ``auto`` asks for on this page by its lettering
    alone (what the read does with the switch off)."""
    _img, info = vision_view.render_view(pdf, page=0, engine=Reader())
    return vision_tools._tile_count("auto", info)


# -- the switch ------------------------------------------------------------------

def test_the_switch_is_off_unless_set(monkeypatch):
    assert ENV == "GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK"
    assert ENV in review_flags.VISION_ENVS
    assert ENV not in review_flags.ALL_ENVS
    assert not review_flags.photo_page_one_look()
    for off in ("", "0", "no", "off", "false"):
        monkeypatch.setenv(ENV, off)
        assert not review_flags.photo_page_one_look(), off
    monkeypatch.setenv(ENV, "1")
    assert review_flags.photo_page_one_look()
    assert vision_tools._photo_page_one_look()


def test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone(
        monkeypatch):
    assert ENV not in review_flags.describe()
    with review_flags.switches({ENV: "1"}):
        assert review_flags.photo_page_one_look()
        assert f"{ENV}=1" in review_flags.describe()
    monkeypatch.setenv(ENV, "1")
    with review_flags.switches(review_flags.ARMS["baseline"]):
        assert not review_flags.photo_page_one_look()
    assert review_flags.photo_page_one_look()           # restored after


@pytest.mark.parametrize("unset", [None, "", "0"])
@pytest.mark.parametrize("build", [_photo_page_pdf, _section_pdf],
                         ids=["photo page", "drawing"])
def test_off_every_read_is_as_before(monkeypatch, unset, build):
    """Off (unset, empty or 0), the page's kind is never even read, and the
    read's prompts and result are byte-for-byte those of a read whose photo
    rule can never fire: the code before VIS-33. A page of photographs is
    tiled as it was."""
    pdf = build()
    n = _auto_would_tile(pdf)
    assert n > 1
    if unset is not None:
        monkeypatch.setenv(ENV, unset)
    asked = []
    monkeypatch.setattr(vision_tools, "_photo_page",
                        lambda *a: asked.append(a) or True)
    now = Reader()
    raw_now = _read_raw(pdf, now)
    assert asked == []
    with monkeypatch.context() as m:
        m.setenv(ENV, "1")
        m.setattr(vision_tools, "_photo_page", lambda *a: False)
        before = Reader()
        raw_before = _read_raw(pdf, before)
    assert raw_now == raw_before
    assert now.prompts == before.prompts
    assert len(now.calls) == 1 + n * n
    assert vision_tools.PHOTO_PAGE_NOTE not in raw_now


# -- on: pages of photographs take one look ----------------------------------------

PHOTO_PAGES = {
    "two photographs with captions": _photo_page_pdf,
    "turned 90 degrees": lambda: _photo_page_pdf(rotate=90),
    "a photograph beside a dimensioned sketch": _photo_and_sketch_pdf,
}


@pytest.mark.parametrize("case", list(PHOTO_PAGES))
def test_on_a_page_of_photographs_takes_one_look(monkeypatch, case):
    pdf = PHOTO_PAGES[case]()
    assert vision_tools._not_photographs(pdf, 0) is None, case
    assert vision_tools._planlens_says_photos(pdf, 0), case
    assert _auto_would_tile(pdf) > 1, case        # tiled with the switch off
    monkeypatch.setenv(ENV, "1")
    out, calls = _read(pdf)
    assert "error" not in out, out
    assert calls == ["page"], case
    assert out["tiles_note"] == vision_tools.PHOTO_PAGE_NOTE


def test_on_a_photo_page_takes_one_look_and_says_tiles_can_be_asked(
        monkeypatch):
    monkeypatch.setenv(ENV, "1")
    pdf = _photo_page_pdf()
    assert vision_tools._photo_page(pdf, 0) is True
    engine = Reader()
    out, calls = _read(pdf, engine)         # tiles defaults to 'auto'
    assert "error" not in out, out
    assert calls == ["page"]                # one look, no tiles
    assert "tiles" not in out and "tiling" not in out
    assert out["tiles_note"] == vision_tools.PHOTO_PAGE_NOTE
    assert "tiles='2' to '4'" in out["tiles_note"]
    assert "render_region" in out["tiles_note"]
    assert out["analysis"].startswith("A photograph of a made-up slope")
    # the record: one vision call, a whole page read alone (not the layout
    # overview of a tiled read), and the tool result saying why
    assert [(lb.get("kind"), lb.get("page")) for lb in engine.labels] == [
        ("page", 0)]


def test_on_a_photo_page_is_tiled_when_tiles_are_asked_for(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    out, calls = _read(_photo_page_pdf(), tiles="2")
    assert sorted(calls) == ["page", "r1c1", "r1c2", "r2c1", "r2c2"]
    assert len(out["tiles"]) == 4
    assert "tiles_note" not in out


# -- on: the drawings planlens' role alone names pages of photographs --------------
#
# Each was read in one look when the role alone decided. ``stopped_by`` is
# the check of rule 2 (``vision_tools._page_evidence``) that tiles it.

SEVEN = {
    "plan titled PLATE NO. 1 over an aerial base, 155 words":
        (_plate_plan_pdf, "words in its text layer"),
    "cross-section LOOKING NORTH with a logo":
        (_section_pdf, "photographs cover 0% of the page"),
    "grading and stockpile plan with a logo":
        (_stockpile_pdf, "vector paths on the page"),
    "foundation plan with a photograph inset captioned PHOTO 1":
        (_foundation_pdf, "photographs cover 3% of the page"),
    "photo location plan over an aerial image":
        (_photo_location_pdf, "line work drawn over a picture"),
    "profile under an appendix header naming photographs":
        (_profile_pdf, "words in its text layer"),
    "scanned drawing with an OCR layer holding LOOKING NORTH":
        (lambda: _scan_with_ocr_pdf("LOOKING NORTH"), "pictures cover 100%"),
    "scanned drawing with an OCR layer holding PLATE NO. 4":
        (lambda: _scan_with_ocr_pdf("PLATE NO. 4"), "pictures cover 100%"),
}


@pytest.mark.parametrize("case", list(SEVEN))
def test_on_a_drawing_the_role_alone_would_pass_is_tiled(monkeypatch, case):
    build, stopped_by = SEVEN[case]
    pdf = build()
    # the premise: planlens' role alone names this drawing photographs
    assert vision_tools._planlens_says_photos(pdf, 0), case
    why = vision_tools._not_photographs(pdf, 0)
    assert why is not None and stopped_by in why, (case, why)
    assert vision_tools._photo_page(pdf, 0) is False
    n = _auto_would_tile(pdf)
    assert n > 1, case
    monkeypatch.setenv(ENV, "1")
    out, calls = _read(pdf)
    assert "error" not in out, out
    assert len(calls) == 1 + n * n, (case, calls)
    assert len(out["tiles"]) == n * n
    assert vision_tools.PHOTO_PAGE_NOTE not in json.dumps(out)


TILED = {
    "scanned drawing in three strips, no text layer":
        (_scan_in_strips_pdf, "pictures cover 100%"),
    "sheet of 400 words with a photograph":
        (_wordy_photo_sheet_pdf, "400 words in its text layer"),
    "sheet with no picture": (_no_picture_pdf, "no picture on the page"),
    "photographs with a word lettered over one":
        (lambda: _photo_page_pdf(label_over=True), "lettering over a picture"),
}


@pytest.mark.parametrize("case", list(TILED))
def test_on_other_pages_that_are_not_photographs_are_tiled(monkeypatch, case):
    build, stopped_by = TILED[case]
    pdf = build()
    why = vision_tools._not_photographs(pdf, 0)
    assert why is not None and stopped_by in why, (case, why)
    n = _auto_would_tile(pdf)
    assert n > 1, case
    monkeypatch.setenv(ENV, "1")
    out, calls = _read(pdf)
    assert len(calls) == 1 + n * n, (case, calls)
    assert vision_tools.PHOTO_PAGE_NOTE not in json.dumps(out)


def test_on_a_scanned_drawing_with_no_text_layer_is_still_tiled(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    pdf = _scanned_drawing_pdf()
    assert vision_tools._photo_page(pdf, 0) is False
    n = _auto_would_tile(pdf)
    assert n > 1
    out, calls = _read(pdf, prompt="Read the title block.")
    assert len(calls) == 1 + n * n
    assert len(out["tiles"]) == n * n
    assert vision_tools.PHOTO_PAGE_NOTE not in json.dumps(out)


# -- a photograph by its pixels ------------------------------------------------------

@pytest.mark.parametrize("picture, photographic", [
    (lambda: _photo(5, 400, 300), True),
    (lambda: _photo(5, 400, 300, grey=True), False),   # tiled: costs calls
    (lambda: _logo(400, 200), False),
    (lambda: _grey(_scan_array(800, 520)), False),
    (_paper_toned_scan, False),
], ids=["colour photograph", "greyscale photograph", "logo",
        "bilevel scan", "paper-toned scan"])
def test_a_picture_is_a_photograph_by_its_pixels(picture, photographic):
    rect = pymupdf.Rect(50, 50, 550, 425)

    def build(page):
        page.insert_image(rect, pixmap=picture())
    with pymupdf.open(stream=_pdf(build), filetype="pdf") as d:
        assert vision_tools._photographic(d[0], rect, np, pymupdf) is \
            photographic


# -- doubt tiles the page -------------------------------------------------------------

def test_a_page_whose_kind_cannot_be_told_is_tiled_as_before(monkeypatch):
    """A planlens without page roles (the core asks for 0.15 or later):
    the photo page is tiled, as it was."""
    monkeypatch.setenv(ENV, "1")
    pdf = _photo_page_pdf()
    n = _auto_would_tile(pdf)
    monkeypatch.setitem(sys.modules, "planlens.document.roles", None)
    assert vision_tools._photo_page(pdf, 0) is False
    out, calls = _read(pdf)
    assert len(calls) == 1 + n * n
    assert "tiles_note" not in out


def test_a_failing_role_read_tiles_as_before(monkeypatch):
    import planlens.document.roles as roles

    def broken(_facts):
        raise RuntimeError("made-up failure")

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(roles, "assign", broken)
    pdf = _photo_page_pdf()
    assert vision_tools._photo_page(pdf, 0) is False
    _out, calls = _read(pdf)
    assert len(calls) > 1


def test_a_failing_look_at_the_page_tiles_it(monkeypatch):
    def broken(*_a):
        raise RuntimeError("made-up failure")

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(vision_tools, "_page_evidence", broken)
    pdf = _photo_page_pdf()
    assert "could not tell" in vision_tools._not_photographs(pdf, 0)
    assert vision_tools._photo_page(pdf, 0) is False
    _out, calls = _read(pdf)
    assert len(calls) > 1


def test_the_efficient_policy_still_reads_one_look_without_the_note(
        monkeypatch):
    """``efficient`` turns ``auto`` off already: no role is read, no note."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(vision_view.POLICY_ENV, "efficient")
    out, calls = _read(_photo_page_pdf())
    assert calls == ["page"]
    assert "tiles_note" not in out
