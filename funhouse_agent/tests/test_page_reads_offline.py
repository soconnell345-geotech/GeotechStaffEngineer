"""Page reads: the whole page and its tiles at once, a failed tile named, an
identical read served again within one conversation only, and what was
read listed (live smoke wave 2a B5, B6, B7; wave 2b C5, C6, C7, C11).

* B5 -- the tiles waited for the whole-page answer they do not use: 10-30 s
  added to every drawing-sheet read.
* B6 -- 24 of 80 page reads re-read a page an earlier turn had read.
* B7 -- a tile that failed was a silent hole in the page read.
* C5 -- a side call cut off at its output limit came back as complete.
* C6 -- analyze_image sent any file labelled PNG; an .xlsx got a 400.
* C7 -- the per-turn note needs which pages were already read.
* C11 -- a blank page was read in tiles.

Fake engines, synthetic PDFs: no model, no network.
"""

import io
import json
import re
import threading
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio, vision_tools  # noqa: E402
from funhouse_agent.deep.vision_engine import VisionAnswer  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    REPEAT_NOTE, dispatch_extended_tool, reads_for_conversation)

TILE = re.compile(r"tile row (\d) of \d, column (\d)")


def _pdf(text="B-1  B-2  B-3", blank=False) -> bytes:
    d = fitz.open()
    page = d.new_page(width=612, height=792)
    if not blank:
        page.insert_text((72, 100), text, fontsize=12)
        page.draw_rect(fitz.Rect(300, 300, 500, 500), color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


class _Busy(Exception):
    """A busy-model error as the SDKs shape it."""

    def __init__(self):
        super().__init__("Error code: 529 overloaded_error")
        self.status_code = 529
        self.response = SimpleNamespace(status_code=529, headers={})


class Reader:
    """Answers which tile (or the whole page) it was shown; can wait at a
    barrier, fail chosen tiles, or cut chosen answers off."""

    def __init__(self, barrier=None, fail=(), cut=(), fail_page=False):
        self.barrier, self.fail, self.cut = barrier, set(fail), set(cut)
        self.fail_page = fail_page
        self.lock = threading.Lock()
        self.calls = []

    def analyze_image(self, image, prompt=""):
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "page"
        with self.lock:
            self.calls.append(who)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)       # all at once, or it breaks
        if who in self.fail or (who == "page" and self.fail_page):
            raise _Busy()
        text = f"I see {who}."
        if who in self.cut:
            text = VisionAnswer(text)
            text.cut_off = True
        return text


def _read(engine, pdf, **args):
    return json.loads(dispatch_extended_tool(
        "analyze_pdf_page", {"attachment_key": "sheet.pdf", "page": 0,
                             "prompt": "What is on this sheet?", **args},
        engine=engine, attachments={"sheet.pdf": pdf}))


@pytest.fixture(autouse=True)
def _fresh():
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


@pytest.fixture
def conv(tmp_path):
    """One conversation's working folder, bound as the app's turn worker
    binds it."""
    def bind(name="alice"):
        folder = tmp_path / name / "files"
        folder.mkdir(parents=True, exist_ok=True)
        return _fileio.working_dir_bound(str(folder))
    return bind


# -- B5: the whole page and its tiles at once, in order ----------------------

def test_the_whole_page_and_its_tiles_are_read_at_once():
    """Five calls must all be in flight together to pass the barrier: the
    tiles no longer wait for the whole-page answer."""
    engine = Reader(barrier=threading.Barrier(5))
    out = _read(engine, _pdf(), tiles="2")
    assert sorted(engine.calls) == ["page", "r1c1", "r1c2", "r2c1", "r2c2"]
    # Order and meaning kept: the overview is the page's, tiles in order.
    assert out["analysis"] == "I see page."
    assert [t["tile"] for t in out["tiles"]] == ["r1c1", "r1c2", "r2c1",
                                                 "r2c2"]
    assert [t["analysis"] for t in out["tiles"]] == [
        "I see r1c1.", "I see r1c2.", "I see r2c1.", "I see r2c2."]
    assert "tiles_not_read" not in out and "overview_not_read" not in out


def test_tiles_off_is_one_call():
    engine = Reader()
    out = _read(engine, _pdf(), tiles="off")
    assert engine.calls == ["page"] and "tiles" not in out


# -- B7: a failed tile is named; the rest still come back --------------------

def test_a_failed_tile_is_named_and_the_rest_were_read():
    engine = Reader(fail={"r2c2"})
    out = _read(engine, _pdf(), tiles="2")
    note = out["tiles_not_read"]
    assert note.startswith("tile r2c2 failed: the vision model was busy "
                           "(overloaded)")
    assert "the rest were read" in note and "render_region" in note
    row = next(t for t in out["tiles"] if t["tile"] == "r2c2")
    assert "busy" in row["error"] and "529" not in row["error"]
    assert out["analysis"] == "I see page."
    assert sum("analysis" in t for t in out["tiles"]) == 3


def test_a_failed_overview_with_tiles_read_is_said():
    out = _read(Reader(fail_page=True), _pdf(), tiles="2")
    assert "whole-page view was NOT read" in out["overview_not_read"]
    assert len([t for t in out["tiles"] if "analysis" in t]) == 4


def test_nothing_read_is_a_plain_error():
    out = _read(Reader(fail_page=True, fail={"r1c1", "r1c2", "r2c1",
                                             "r2c2"}), _pdf(), tiles="2")
    assert "busy" in out["error"] and "NOT read" in out["error"]
    out = _read(Reader(fail_page=True), _pdf(), tiles="off")
    assert "busy (overloaded)" in out["error"]


# -- C5: an answer cut off at the output limit is flagged --------------------

def test_cut_off_answers_are_flagged_in_the_result():
    out = _read(Reader(cut={"page", "r1c2"}), _pdf(), tiles="2")
    # With tiles the whole page is asked for the layout only (wave 2c, E1):
    # its cut is said apart, and the reading's cut names the tile.
    assert out["cut_off"].startswith("tile r1c2: ")
    assert "CUT OFF" in out["cut_off"]
    assert "layout answer stopped" in out["overview_cut"]
    assert next(t for t in out["tiles"] if t["tile"] == "r1c2")["cut_off"]
    # A page read whole, with no tiles, is the reading: its cut says so.
    out = _read(Reader(cut={"page"}), _pdf(), tiles="off")
    assert out["cut_off"].startswith("the whole-page view: ")
    assert "overview_cut" not in out


# -- C11: a blank page is not tiled -------------------------------------------

def test_a_blank_page_is_not_read_in_tiles():
    engine = Reader()
    out = _read(engine, _pdf(blank=True), tiles="3")
    assert engine.calls == ["page"] and "tiles" not in out
    assert "no ink" in out["tiles_note"]
    engine = Reader()
    _read(engine, _pdf(), tiles="3")
    assert len(engine.calls) == 10                 # a page with ink is


# -- B6: an identical read is served again, in this conversation only --------

def test_an_identical_read_in_the_same_conversation_is_a_repeat(conv):
    engine, pdf = Reader(), _pdf()
    with conv("alice"):
        first = _read(engine, pdf, tiles="2")
        assert len(engine.calls) == 5 and "repeat" not in first
        again = _read(engine, pdf, tiles="2")
    assert len(engine.calls) == 5                  # no new look
    assert again["repeat"] == REPEAT_NOTE
    assert {k: v for k, v in again.items() if k != "repeat"} == first


def test_a_different_prompt_is_a_new_read(conv):
    engine, pdf = Reader(), _pdf()
    with conv("alice"):
        _read(engine, pdf, tiles="off")
        out = _read(engine, pdf, tiles="off", prompt="Read the title block")
    assert engine.calls == ["page", "page"] and "repeat" not in out


def test_another_conversation_never_gets_the_reading(conv):
    engine, pdf = Reader(), _pdf()
    with conv("alice"):
        _read(engine, pdf, tiles="off")
    with conv("bob"):
        out = _read(engine, pdf, tiles="off")
    assert engine.calls == ["page", "page"] and "repeat" not in out


def test_no_conversation_no_repeat():
    engine, pdf = Reader(), _pdf()
    _read(engine, pdf, tiles="off")
    _read(engine, pdf, tiles="off")
    assert engine.calls == ["page", "page"]


def test_an_incomplete_read_is_asked_again(conv):
    engine, pdf = Reader(fail={"r1c1"}), _pdf()
    with conv("alice"):
        _read(engine, pdf, tiles="2")
        engine.fail.clear()
        out = _read(engine, pdf, tiles="2")
    assert len(engine.calls) == 10 and "repeat" not in out
    assert "tiles_not_read" not in out


def test_a_zoom_and_an_image_repeat_too(conv, tmp_path):
    engine, pdf = Reader(), _pdf()
    region = {"attachment_key": "sheet.pdf", "page": 0,
              "bbox": [60, 80, 260, 120], "prompt": "Read the tags"}
    png = fitz.open(stream=pdf, filetype="pdf")[0].get_pixmap().tobytes("png")
    with conv("alice"):
        for _ in range(2):
            r = json.loads(dispatch_extended_tool(
                "render_region", region, engine, {"sheet.pdf": pdf}))
            i = json.loads(dispatch_extended_tool(
                "analyze_image", {"attachment_key": "p.png"}, engine,
                {"p.png": png}))
    assert len(engine.calls) == 2
    assert r["repeat"] == REPEAT_NOTE and i["repeat"] == REPEAT_NOTE


def test_the_store_is_bounded(conv, monkeypatch):
    monkeypatch.setattr(vision_tools, "REPEAT_MAX_ENTRIES", 3)
    engine, pdf = Reader(), _pdf()
    with conv("alice"):
        for k in range(5):
            _read(engine, pdf, tiles="off", prompt=f"question {k}")
        assert vision_tools._REPEATS.size(
            vision_tools._conversation_for()) == 3
        _read(engine, pdf, tiles="off", prompt="question 0")   # dropped
    assert len(engine.calls) == 6


# -- C7: what was read in this conversation ----------------------------------

def test_reads_for_conversation_lists_pages_and_zooms(conv, tmp_path):
    engine, pdf = Reader(), _pdf()
    with conv("alice"):
        _read(engine, pdf, tiles="2")
        _read(engine, pdf, tiles="2")                # a repeat: not again
        json.loads(dispatch_extended_tool(
            "render_region", {"attachment_key": "sheet.pdf", "page": 0,
                              "bbox": [60, 80, 260, 120],
                              "prompt": "Read the tags"},
            engine, {"sheet.pdf": pdf}))
        reads = reads_for_conversation()
    assert [(r["document"], r["page"], r["pdf_page"], r["tool"])
            for r in reads] == [("sheet.pdf", 0, 1, "analyze_pdf_page"),
                                ("sheet.pdf", 0, 1, "render_region")]
    assert reads[0]["view"] == "page+tiles 2x2"
    assert isinstance(reads[1]["view"], list) and len(reads[1]["view"]) == 4
    assert reads[0]["prompt"] == "What is on this sheet?"
    assert reads[0]["when"] <= reads[1]["when"]
    # By folder, from outside the turn; another conversation sees none.
    assert reads_for_conversation(tmp_path / "alice" / "files") == reads
    assert reads_for_conversation(tmp_path / "bob" / "files") == []


# -- C6: only images go to analyze_image -------------------------------------

def _image(fmt, **kw):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 30, 30)).save(buf, format=fmt, **kw)
    return buf.getvalue()


class Seen:
    def __init__(self):
        self.images = []

    def analyze_image(self, image, prompt=""):
        self.images.append(bytes(image))
        return "seen"


@pytest.mark.parametrize("fmt", ["TIFF", "BMP", "GIF"])
def test_tiff_bmp_and_gif_are_converted_to_png(fmt):
    pytest.importorskip("PIL")
    engine = Seen()
    out = json.loads(dispatch_extended_tool(
        "analyze_image", {"attachment_key": f"scan.{fmt.lower()}"}, engine,
        {f"scan.{fmt.lower()}": _image(fmt)}))
    assert out["analysis"] == "seen"
    assert out["image_note"] == f"converted from {fmt} to PNG"
    assert engine.images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_png_and_jpeg_go_as_they_are():
    pytest.importorskip("PIL")
    engine = Seen()
    jpg = _image("JPEG")
    out = json.loads(dispatch_extended_tool(
        "analyze_image", {"attachment_key": "a.jpg"}, engine, {"a.jpg": jpg}))
    assert engine.images == [jpg] and "image_note" not in out


@pytest.mark.parametrize("name, data, tool", [
    ("log.xlsx", b"PK\x03\x04" + b"\x00" * 40, "read_text_file"),
    ("memo.docx", b"PK\x03\x04" + b"\x00" * 40, "read_text_file"),
    ("report.pdf", b"%PDF-1.7\n...", "analyze_pdf_page"),
    ("notes.txt", b"just some notes", "read_text_file"),
])
def test_what_is_not_an_image_is_refused_naming_the_right_tool(name, data,
                                                               tool):
    engine = Seen()
    out = json.loads(dispatch_extended_tool(
        "analyze_image", {"attachment_key": name}, engine, {name: data}))
    assert tool in out["error"] and engine.images == []
