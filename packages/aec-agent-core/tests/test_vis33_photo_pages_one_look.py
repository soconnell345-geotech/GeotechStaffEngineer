"""BACKLOG VIS-33: on ``tiles='auto'`` a page of photographs takes one look.

A page with almost no text layer is assumed to carry drawing lettering and
was read whole AND in tiles: five or more vision calls on a page of
photographs, whose readings ran several times longer than one look that
answered the same questions. Now a page that planlens names a page of
photographs (page role ``photos``) takes one look under ``auto``, and the
result says tiles or a zoom can be asked for. A scanned drawing (one
page-sized picture, no text layer) is still tiled; an explicit ``tiles`` is
honoured as asked; where the page's kind cannot be told it is tiled as
before.

Made-up pages built here and a fake vision model: no model, no network.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

pymupdf = pytest.importorskip("pymupdf")
np = pytest.importorskip("numpy")
pytest.importorskip("planlens.document.roles")

from aec_agent_core import vision_tools, vision_view  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402

TILE = re.compile(r"tile row (\d) of \d, column (\d)")


def _photo(seed: int, w: int = 320, h: int = 200):
    """A made-up photograph: smooth colour with noise, many tones."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // w), (yy * 255 // h),
                     ((xx + yy) * 255 // (w + h))], axis=2)
    a = np.clip(base + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, a.tobytes(), False)


def _photo_page_pdf() -> bytes:
    """A report page of two photographs, each with a short caption: almost
    no text layer, so ``auto`` would assume drawing lettering and tile it."""
    d = pymupdf.open()
    page = d.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(72, 60, 540, 350), pixmap=_photo(1))
    page.insert_text((72, 370), "Photo 1", fontsize=10)
    page.insert_image(pymupdf.Rect(72, 420, 540, 710), pixmap=_photo(2))
    page.insert_text((72, 730), "Photo 2", fontsize=10)
    data = d.tobytes()
    d.close()
    return data


def _scanned_drawing_pdf() -> bytes:
    """A made-up scanned drawing: one page-sized picture of black lines on
    white paper, no text layer at all."""
    w, h = 1700, 1100
    a = np.full((h, w), 255, dtype=np.uint8)
    for x in range(100, w - 100, 150):
        a[100:h - 100, x:x + 3] = 0
    for y in range(100, h - 100, 120):
        a[y:y + 3, 100:w - 100] = 0
    a[h - 220:h - 100, w - 600:w - 100] = 0        # a title block's frame
    a[h - 217:h - 103, w - 597:w - 103] = 255
    pix = pymupdf.Pixmap(pymupdf.csGRAY, w, h, a.tobytes(), False)
    d = pymupdf.open()
    page = d.new_page(width=1224, height=792)
    page.insert_image(page.rect, pixmap=pix)
    data = d.tobytes()
    d.close()
    return data


class Reader:
    """Names what it was shown: a tile, or the whole page."""

    def __init__(self):
        self.calls = []
        self.labels = []

    def analyze_image(self, image, prompt=""):
        from aec_agent_core.deep.vision_engine import current_side_call
        m = TILE.search(prompt)
        self.calls.append(f"r{m.group(1)}c{m.group(2)}" if m else "page")
        # what the run record keeps for this vision call (``vision``)
        self.labels.append(dict(current_side_call() or {}))
        return "A photograph of a made-up slope px=[10, 10, 60, 40]"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(vision_view.POLICY_ENV, raising=False)
    monkeypatch.delenv(vision_tools.VISION_CAP_ENV, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _read(pdf, engine=None, **args):
    engine = engine or Reader()
    out = json.loads(dispatch_extended_tool(
        "analyze_pdf_page",
        {"attachment_key": "made_up.pdf", "page": 0,
         "prompt": "What do the photographs show?", **args},
        engine=engine, attachments={"made_up.pdf": pdf}))
    return out, engine.calls


def _auto_would_tile(pdf) -> int:
    """Tiles per side ``auto`` asks for on this page by its lettering
    alone (what the read did before VIS-33)."""
    _img, info = vision_view.render_view(pdf, page=0, engine=Reader())
    return vision_tools._tile_count("auto", info)


def test_planlens_names_the_made_up_photo_page_photographs():
    assert vision_tools._photo_page(_photo_page_pdf(), 0) is True
    assert vision_tools._photo_page(_scanned_drawing_pdf(), 0) is False


def test_a_photo_page_takes_one_look_on_auto_and_says_tiles_can_be_asked():
    pdf = _photo_page_pdf()
    assert _auto_would_tile(pdf) > 1        # tiled before VIS-33
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


def test_a_photo_page_is_tiled_when_tiles_are_asked_for():
    out, calls = _read(_photo_page_pdf(), tiles="2")
    assert sorted(calls) == ["page", "r1c1", "r1c2", "r2c1", "r2c2"]
    assert len(out["tiles"]) == 4
    assert "tiles_note" not in out


def test_a_scanned_drawing_with_no_text_layer_is_still_tiled():
    pdf = _scanned_drawing_pdf()
    n = _auto_would_tile(pdf)
    assert n > 1
    out, calls = _read(pdf, prompt="Read the title block.")
    assert len(calls) == 1 + n * n
    assert len(out["tiles"]) == n * n
    assert vision_tools.PHOTO_PAGE_NOTE not in json.dumps(out)


def test_a_page_whose_kind_cannot_be_told_is_tiled_as_before(monkeypatch):
    """A planlens without page roles (the core asks for 0.15 or later):
    the photo page is tiled, as it was."""
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

    monkeypatch.setattr(roles, "assign", broken)
    pdf = _photo_page_pdf()
    assert vision_tools._photo_page(pdf, 0) is False
    _out, calls = _read(pdf)
    assert len(calls) > 1


def test_the_efficient_policy_still_reads_one_look_without_the_note(
        monkeypatch):
    """``efficient`` turns ``auto`` off already: no role is read, no note."""
    monkeypatch.setenv(vision_view.POLICY_ENV, "efficient")
    out, calls = _read(_photo_page_pdf())
    assert calls == ["page"]
    assert "tiles_note" not in out
