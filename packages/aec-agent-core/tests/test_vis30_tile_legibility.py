"""VIS-30 (1): a tiled page read keeps its legibility note while a tile's
small lettering is still under the legible line, and names those tiles."""

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import vision_tools, vision_view  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402


class Reader:
    def analyze_image(self, image, prompt=""):
        return "ok"


def _sheet(width, height, fontsize):
    d = pymupdf.open()
    p = d.new_page(width=width, height=height)
    p.insert_text((100, 200), "NOTE 1 GENERAL NOTES AND KEYNOTES LISTED HERE "
                  "FOR THE WHOLE SHEET", fontsize=fontsize)
    p.draw_rect(pymupdf.Rect(300, 300, width - 300, height - 300),
                color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


@pytest.fixture(autouse=True)
def _fresh():
    vision_tools.clear_repeat_reads()


def _read(pdf, **a):
    return json.loads(dispatch_extended_tool(
        "analyze_pdf_page", {"attachment_key": "s.pdf", "page": 0,
                             "prompt": "x", **a},
        engine=Reader(), attachments={"s.pdf": pdf}))


def test_tiny_lettering_on_a_big_sheet_keeps_the_note_naming_tiles():
    out = _read(_sheet(2592, 1728, 4.3), tiles="2")
    assert "tiles" in out
    assert "legibility" in out
    assert "render_region" in out["legibility"]
    assert "r1c1" in out["legibility"] or "r1c2" in out["legibility"]
    assert all("_lettering_px" not in t for t in out["tiles"])


def test_note_dropped_when_every_tile_reads():
    out = _read(_sheet(612, 792, 12), tiles="2")
    px = [t.get("text_px") for t in out["tiles"]]
    assert all(p is None or p >= vision_view.LEGIBLE_TEXT_PX for p in px)
    assert "legibility" not in out
    assert all("_lettering_px" not in t for t in out["tiles"])


def test_assumed_lettering_when_not_in_text_layer():
    info = {"clip": [0, 0, 100, 100], "width_px": 600, "height_px": 600,
            "text_chars": 0}
    px = vision_tools._tile_lettering_px(info)
    assert px == pytest.approx(vision_view.ASSUMED_CAD_LETTERING_PT * 6)
    assert vision_tools._tile_lettering_px(
        {**info, "text_chars": 500}) is None
