"""BACKLOG VIS-34: tile reads spent their output on what is absent.

A tile holding nothing on the ask still answered at length, often listing
each thing asked about as not present. With ``GEOTECH_VISION_TILE_ANSWERS``
on, each tile of a tiled read gets the answer shape a page read whole and a
zoom have, and one line when it holds nothing on the ask. The switch is off
by default; off, every prompt and result is what it was.

Made-up pages and pictures built here and a fake vision model: no model, no
network.
"""

from __future__ import annotations

import json
import re
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import review_flags, vision_tools, vision_view  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402

ASK = "List every boring id on this sheet with its ground elevation."
TILE = re.compile(r"tile row (\d) of \d, column (\d)")
LOCATION = "If you give the location of anything in it"
#: A long answer a tile that holds something gives (made-up values).
FULL = ("B-7 ground elevation 101.3 m px=[12, 14, 80, 30]\n"
        "B-8 ground elevation 99.8 m px=[200, 14, 270, 30]\n" * 40)


def _sheet_pdf() -> bytes:
    """A made-up sheet with lettering in one corner only."""
    d = pymupdf.open()
    page = d.new_page(width=1224, height=792)
    page.insert_text((72, 100), "B-7  EL 101.3", fontsize=6)
    page.insert_text((72, 120), "B-8  EL 99.8", fontsize=6)
    page.draw_rect(pymupdf.Rect(600, 400, 1100, 700), color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


def _picture_png() -> bytes:
    """A made-up picture: lines on white."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1600, 1200),
                         False)
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(100, 100, 700, 110), (0, 0, 0))
    pix.set_rect(pymupdf.IRect(900, 800, 910, 1100), (0, 0, 0))
    return pix.tobytes("png")


class Reader:
    """Names what it was shown; the top-left tile holds the borings, every
    other tile holds nothing on the ask."""

    def __init__(self, empty_reply=None):
        self.lock = threading.Lock()
        self.prompts = {}
        self.seen = []
        self.empty_reply = empty_reply

    def analyze_image(self, image, prompt=""):
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "whole"
        with self.lock:
            self.prompts[who] = prompt
            self.seen.append(prompt)
        if who == "r1c1":
            return FULL
        if who != "whole" and self.empty_reply is not None:
            return self.empty_reply
        return f"Lines and frames ({who}); no boring ids px=[5, 5, 40, 40]"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for env in (review_flags.TILE_ANSWERS_ENV, vision_view.POLICY_ENV,
                vision_tools.VISION_CAP_ENV):
        monkeypatch.delenv(env, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _page(engine, **args):
    raw = dispatch_extended_tool(
        "analyze_pdf_page",
        {"attachment_key": "made_up.pdf", "page": 0, "prompt": ASK,
         "tiles": "2", **args},
        engine=engine, attachments={"made_up.pdf": _sheet_pdf()})
    vision_tools.clear_repeat_reads()
    return raw


def _image(engine):
    raw = dispatch_extended_tool(
        "analyze_image",
        {"attachment_key": "made_up.png", "prompt": ASK, "tiles": "2"},
        engine=engine, attachments={"made_up.png": _picture_png()})
    vision_tools.clear_repeat_reads()
    return raw


def _master_answer_shape(kind):
    """``_answer_shape`` as it was before VIS-34: tiles took none."""
    return {"page": vision_tools.PAGE_ANSWER_SHAPE,
            "region": vision_tools.ZOOM_ANSWER_SHAPE}.get(kind)


# -- the switch ----------------------------------------------------------------

def test_the_switch_is_off_unless_set(monkeypatch):
    assert review_flags.TILE_ANSWERS_ENV == "GEOTECH_VISION_TILE_ANSWERS"
    assert not review_flags.tile_answers()
    assert vision_tools._answer_shape("tile") is None
    for off in ("", "0", "no", "off", "false"):
        monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, off)
        assert not review_flags.tile_answers(), off
    monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, "1")
    assert review_flags.tile_answers()
    assert vision_tools._answer_shape("tile") == \
        vision_tools.TILE_ANSWER_SHAPE


def test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone(
        monkeypatch):
    """A turn's record carries ``describe()`` (``build.switches``): it says
    when the switch is on. An arm sets exactly what it names: the pair the
    suite takes turns it on; the baseline clears a value left in the
    notebook's environment."""
    assert review_flags.TILE_ANSWERS_ENV in review_flags.VISION_ENVS
    assert review_flags.TILE_ANSWERS_ENV not in review_flags.describe()
    with review_flags.switches({review_flags.TILE_ANSWERS_ENV: "1"}):
        assert review_flags.tile_answers()
        assert "GEOTECH_VISION_TILE_ANSWERS=1" in review_flags.describe()
    monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, "1")
    with review_flags.switches(review_flags.ARMS["baseline"]):
        assert not review_flags.tile_answers()
    assert review_flags.tile_answers()                  # restored after


# -- off: byte-for-byte as before -------------------------------------------------

@pytest.mark.parametrize("read", [_page, _image])
@pytest.mark.parametrize("unset", [None, "", "0"])
def test_off_every_prompt_and_result_is_as_before(monkeypatch, read, unset):
    """Off (unset, empty or 0), a tiled read's prompts and its result are
    byte-for-byte those of the code before VIS-34, whose tiles took no
    answer shape."""
    if unset is not None:
        monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, unset)
    now = Reader()
    raw_now = read(now)
    with monkeypatch.context() as m:
        m.setattr(vision_tools, "_answer_shape", _master_answer_shape)
        before = Reader()
        raw_before = read(before)
    assert raw_now == raw_before
    assert now.prompts == before.prompts
    assert len(now.prompts) == 5                        # whole + 2x2 tiles
    for who, prompt in now.prompts.items():
        assert "How to answer" not in prompt, who
        assert vision_tools.TILE_NOTHING_LINE not in prompt, who
        assert vision_tools.TILE_CUT_SENTENCE not in prompt, who


# -- on: the answer shape on every tile -----------------------------------------

@pytest.mark.parametrize("read", [_page, _image])
def test_on_each_tile_gets_the_answer_shape(monkeypatch, read):
    monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, "1")
    engine = Reader()
    out = json.loads(read(engine))
    assert "error" not in out, out
    tiles = {k: p for k, p in engine.prompts.items() if k != "whole"}
    assert sorted(tiles) == ["r1c1", "r1c2", "r2c1", "r2c2"]
    for who, prompt in tiles.items():
        assert prompt.startswith(ASK), who                 # the ask first
        assert vision_tools.TILE_ANSWER_SHAPE in prompt, who
        assert f'"{vision_tools.TILE_NOTHING_LINE}"' in prompt
        at = [prompt.index(p) for p in (
            ASK, "Report only what is IN this tile",
            vision_tools.TILE_ANSWER_SHAPE, LOCATION)]
        assert at == sorted(at), who
    # the whole-page layout call keeps its own prompt
    assert "How to answer" not in engine.prompts["whole"]


@pytest.mark.parametrize("read", [_page, _image])
def test_on_a_tile_quotes_its_part_of_what_its_edge_cuts(monkeypatch, read):
    """A schedule cut so its values sit in one tile and its row labels in
    the next: the value tile shows nothing that visibly bears on the ask,
    so the one-line exit would lose the values. Each tile is told to quote
    its part of a table, a dimension string or a note its edge cuts, and
    told so before it is offered the one line."""
    sentence = vision_tools.TILE_CUT_SENTENCE
    for words in ("edge cuts a table, a dimension string or a note",
                  "quote the part of it this tile holds",
                  "labels or what it refers to lie outside the tile"):
        assert words in sentence
    monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, "1")
    engine = Reader()
    read(engine)
    tiles = {k: p for k, p in engine.prompts.items() if k != "whole"}
    assert len(tiles) == 4
    for who, prompt in tiles.items():
        assert prompt.count(sentence) == 1, who
        assert prompt.index(sentence) < prompt.index(
            f'"{vision_tools.TILE_NOTHING_LINE}"'), who
    assert sentence not in engine.prompts["whole"]


def test_on_the_shape_asks_for_nothing_less_from_a_tile_that_holds_it():
    shape = vision_tools.TILE_ANSWER_SHAPE
    assert "quoting the lettering that bears on it exactly as printed, " \
           "and leave out none of it" in shape
    assert "do not list what is not there" in shape
    assert vision_tools._SHAPE_TAIL in shape


def test_on_page_and_zoom_reads_are_unchanged(monkeypatch):
    """Only tiles change: a page read whole and a zoom keep their shapes."""
    prompts = {}
    for value in (None, "1"):
        if value:
            monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, value)
        engine = Reader()
        _page(engine, tiles="off")
        dispatch_extended_tool(
            "render_region",
            {"attachment_key": "made_up.pdf", "page": 0, "prompt": ASK,
             "bbox": [60, 80, 300, 130]},
            engine=engine, attachments={"made_up.pdf": _sheet_pdf()})
        vision_tools.clear_repeat_reads()
        prompts[value] = engine.seen
    assert len(prompts[None]) == 2
    assert vision_tools.PAGE_ANSWER_SHAPE in prompts[None][0]
    assert vision_tools.ZOOM_ANSWER_SHAPE in prompts[None][1]
    assert prompts[None] == prompts["1"]


def test_on_nothing_is_cut_from_a_tile_that_holds_something(monkeypatch):
    """The one-line tiles come back as they were answered, and the tile
    that holds the borings comes back whole."""
    monkeypatch.setenv(review_flags.TILE_ANSWERS_ENV, "1")
    engine = Reader(empty_reply=vision_tools.TILE_NOTHING_LINE)
    out = json.loads(_page(engine))
    rows = {t["tile"]: t for t in out["tiles"]}
    for who in ("r1c2", "r2c1", "r2c2"):
        assert rows[who]["analysis"] == vision_tools.TILE_NOTHING_LINE
    full = rows["r1c1"]["analysis"]
    assert "shortened" not in full
    assert full.count("ground elevation") == FULL.count("ground elevation")
