"""The vision tools after the Foundry brief 4 review (2026-10-07, TRACE_REVIEW
§7 items C, E, F, G and K).

* C — an agent's ``dpi`` quartered its zooms (45 of 106 zooms at a median
  4.2 px per point against 15-17 without); it is no longer used while an
  image budget is in force, and the tools stop offering it.
* E — the tag drawn turned 90 degrees was found only by a tile, nested under
  the whole-page answer, and never pursued; the result now lists every thing
  located, with where it was seen, and names what only the tiles found.
* F — a padded zoom window can hold a neighbour, and a look answered about
  it; the result now says how far the answer's nearest box is from the aim.
* G — the reading instruction asks for a best reading, with alternatives
  only where a character truly cannot be told, and a bracket left in an
  answer comes with "settle it with a closer zoom".
* K — images padded to whole 32 px patches, behind a switch, OFF by default.

The stand-in engines SEE: they find the ink in the image they are sent, or
answer from the fixture's truth through the view the tool used.
"""

from __future__ import annotations

import json
import re

import pytest

fitz = pytest.importorskip("fitz")
np = pytest.importorskip("numpy")

from funhouse_agent import vision_probe, vision_tools, vision_view  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_render_region,
)

#: The thing aimed at, and a neighbour 80 pt to its right (PDF points).
TARGET = (610.0, 404.0, 632.0, 413.0)
NEIGHBOUR = (712.0, 404.0, 734.0, 413.0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV,
                vision_view.POLICY_ENV, vision_view.PATCH_ALIGN_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    vision_probe.clear_cache()
    yield
    vision_probe.clear_cache()


def _sheet(*boxes) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)
    for b in boxes:
        page.draw_rect(fitz.Rect(*b), color=(0, 0, 0), fill=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


def _blobs(image_bytes):
    """``(size, [pixel boxes of the ink blobs, left to right])``."""
    pix = fitz.Pixmap(image_bytes)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, 0] < 100
    cols = a.any(axis=0)
    boxes, start = [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            ys = np.nonzero(a[:, start:x].any(axis=1))[0]
            boxes.append([start, int(ys.min()), x, int(ys.max()) + 1])
            start = None
    return (pix.width, pix.height), boxes


class Sees:
    """Answers with the pixel box of every ink blob it sees (``which`` =
    ``"all"``), or only the right-most one (``"right"``)."""

    def __init__(self, which="all", text="TAG"):
        self.which, self.text, self.prompts, self.sizes = which, text, [], []

    def analyze_image(self, image_bytes, prompt=""):
        self.prompts.append(prompt)
        size, boxes = _blobs(image_bytes)
        self.sizes.append(size)
        if self.which == "right":
            boxes = boxes[-1:]
        return "\n".join(f"{self.text} px=[{b[0]}, {b[1]}, {b[2]}, {b[3]}]"
                         for b in boxes)


def _grid(view, box):
    vx0, vy0, vx1, vy1 = view
    return [round((box[0] - vx0) / (vx1 - vx0) * 999),
            round((box[1] - vy0) / (vy1 - vy0) * 999),
            round((box[2] - vx0) / (vx1 - vx0) * 999),
            round((box[3] - vy0) / (vy1 - vy0) * 999)]


# -- C: an agent's dpi never shrinks a zoom ----------------------------------------

def test_an_agents_dpi_does_not_shrink_a_zoom(monkeypatch):
    """Brief 4: zooms with an agent-set dpi went at a median 4.2 px per pt,
    without one at 15-17. Under a budget the dpi is not used, and the result
    says so."""
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    pdf = _sheet(TARGET)
    window = [560, 370, 680, 450]
    plain, with_dpi = Sees(), Sees()
    a = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "bbox": window}, plain, {"s": pdf}))
    b = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "bbox": window, "dpi": 300},
        with_dpi, {"s": pdf}))
    assert "dpi_note" not in a and "dpi=300 was not used" in b["dpi_note"]
    assert with_dpi.sizes == plain.sizes
    assert b["view_px"] == a["view_px"] and max(b["view_px"]) == 2048


def test_the_zoom_tools_no_longer_offer_dpi():
    from funhouse_agent.deep.tools import make_vision_tools
    from funhouse_agent.native_tools import OPENAI_TOOLS
    rr = {t.name: t for t in make_vision_tools()}["render_region"]
    assert "dpi" not in rr.args
    native = [t for t in OPENAI_TOOLS
              if t["function"]["name"] == "render_region"]
    assert native and "dpi" not in native[0]["function"]["parameters"][
        "properties"]


# -- E: what only the tiles found is said -------------------------------------------

class TruthReader:
    """Answers from the tag fixture's truth through the view each call was
    given (the page, or the tile its prompt names): every GCE callout in the
    view, in pixels of the image sent — except, on the whole page, the
    callouts listed in ``page_misses`` (the turned tag of brief 4)."""

    def __init__(self, gt, page_misses=()):
        self.gt, self.misses, self.calls = gt, set(page_misses), 0
        self.callouts = [t for t in gt.tags if t.page == 0
                         and t.text == "GCE" and t.kind == "callout"]

    def analyze_image(self, image_bytes, prompt=""):
        self.calls += 1
        w, h = (int(v) for v in re.search(
            r"This image is (\d+) x (\d+) pixels", prompt).groups())
        page = (0.0, 0.0, 1224.0, 792.0)
        tile = re.search(r"tile row (\d+) of (\d+), column (\d+) of", prompt)
        if tile:
            r, n, c = (int(v) for v in tile.groups())
            view = vision_view.tile_boxes(page, n)[(r - 1) * n + (c - 1)]
        else:
            view = page
        vx0, vy0, vx1, vy1 = view
        lines = []
        for i, t in enumerate(self.callouts):
            if not tile and i in self.misses:
                continue
            cx, cy = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
            if not (vx0 <= cx <= vx1 and vy0 <= cy <= vy1):
                continue
            px = [round((t.bbox[0] - vx0) / (vx1 - vx0) * w),
                  round((t.bbox[1] - vy0) / (vy1 - vy0) * h),
                  round((t.bbox[2] - vx0) / (vx1 - vx0) * w),
                  round((t.bbox[3] - vy0) / (vy1 - vy0) * h)]
            lines.append(f"{len(lines) + 1}. GCE callout px={px}")
        return "\n".join(lines) or "No GCE callouts in this view."


def test_a_thing_only_the_tiles_found_is_named(monkeypatch):
    """Brief 4, GPT-5.4 baseline and r3: the whole page listed six callouts
    and missed T5 (drawn turned 90 degrees); tile r2c1 found it; the agent
    worked from the page's six. Now every thing located is listed once with
    where it was seen, and the one the page missed is named."""
    tags = pytest.importorskip("planlens.testing.tag_fixtures")
    gt = tags.build_synthetic_tag_set()
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    reader = TruthReader(gt)
    turned = next(i for i, t in enumerate(reader.callouts) if t.rotation)
    reader.misses = {turned}
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "t", "page": 0, "prompt": "find GCE callouts"},
        reader, {"t": gt.pdf}))
    assert len(out["tiles"]) == 9
    gce = [r for r in out["found"] if "GCE" in r["what"]]
    assert len(gce) == 7, out["found"]
    only = [r for r in out["found"] if r.get("tiles_only")]
    assert len(only) == 1
    t5 = reader.callouts[turned].bbox
    bx = only[0]["page_bbox"]
    assert abs((bx[0] + bx[2]) / 2 - (t5[0] + t5[2]) / 2) < 2
    assert abs((bx[1] + bx[3]) / 2 - (t5[1] + t5[3]) / 2) < 2
    assert "ONLY in the tiles" in out["found_note"]
    assert str(only[0]["page_bbox"]) in out["found_note"]
    # every other callout was seen on the page and in at least one tile
    for r in gce:
        if not r.get("tiles_only"):
            assert "page" in r["seen_in"] and len(r["seen_in"]) >= 2


def test_when_the_page_and_tiles_agree_the_note_says_so(monkeypatch):
    tags = pytest.importorskip("planlens.testing.tag_fixtures")
    gt = tags.build_synthetic_tag_set()
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "t", "page": 0, "prompt": "find GCE callouts"},
        TruthReader(gt), {"t": gt.pdf}))
    assert not any(r.get("tiles_only") for r in out["found"])
    assert "agree" in out["found_note"]


def test_no_tiles_no_merged_list():
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find", "tiles": "off"},
        Sees(), {"s": _sheet(TARGET)}))
    assert "found" not in out and "found_note" not in out


def test_answer_boxes_reads_labels_and_skips_regions():
    text = ("1. **GCE** callout [100, 200, 110, 205]\n"
            "| GCG | [300, 300, 310, 305] |\n"
            "The whole drawing area [0, 0, 999, 999]\n"
            "readings 5.3, 4.1 and nothing else")
    got = vision_view.answer_boxes(text, (0.0, 0.0, 999.0, 999.0))
    assert [g[0] for g in got] == ["GCE callout", "GCG"]
    assert got[0][1] == pytest.approx((100, 200, 110, 205))


# -- F: how far a zoom's answer is from its aim -----------------------------------

def test_a_zoom_answered_about_a_neighbour_says_so(monkeypatch):
    """Brief 4, Sol r3: a padded window aimed at one candidate also held a
    tag 98 pt away; the look answered about the tag, which was ringed twice.
    The result now says how far the answer's nearest box is from the aim."""
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    pdf = _sheet(TARGET, NEIGHBOUR)
    page = [0.0, 0.0, 1224.0, 792.0]
    args = {"attachment_key": "s", "page": 0, "view": page,
            "image_box": _grid(page, TARGET)}
    wrong = json.loads(_dispatch_render_region(dict(args), Sees("right"),
                                               {"s": pdf}))
    assert wrong["nearest_box_from_aim_pt"] > 90
    assert "different thing" in wrong["aim_note"]
    both = json.loads(_dispatch_render_region(dict(args), Sees("all"),
                                              {"s": pdf}))
    assert both["nearest_box_from_aim_pt"] < 3
    assert both["boxes_in_answer"] == 2 and "2 boxes" in both["aim_note"]
    alone = json.loads(_dispatch_render_region(
        dict(args), Sees("all"), {"s": _sheet(TARGET)}))
    assert alone["nearest_box_from_aim_pt"] < 3 and "aim_note" not in alone
    assert alone["aim"] == pytest.approx(
        [(TARGET[0] + TARGET[2]) / 2, (TARGET[1] + TARGET[3]) / 2], abs=1.0)


def test_a_bbox_zoom_has_no_aim():
    out = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "bbox": [560, 370, 680, 450]},
        Sees(), {"s": _sheet(TARGET)}))
    assert "aim" not in out and "aim_note" not in out


def test_the_aim_tolerance_is_a_fraction_of_the_source_view():
    assert vision_view.aim_tolerance([0, 0, 1224, 792]) == pytest.approx(30.6)
    assert vision_view.aim_tolerance([0, 0, 100, 60]) == 12.0


# -- G: a best reading, and a bracket settled by a closer zoom ----------------------

def test_the_reading_instruction_asks_for_a_best_reading():
    text = vision_view.READING_INSTRUCTION
    assert "best reading" in text
    assert "Only where a character truly cannot be told apart" in text
    assert "write plainly" in text
    assert "rather than picking one" not in text     # the old wording
    assert text in vision_view.pixel_instruction((100, 100))


def test_a_bracketed_reading_comes_with_a_closer_zoom_note():
    pdf = _sheet(TARGET)

    class Hedges:
        def __init__(self, answer):
            self.answer = answer

        def analyze_image(self, image_bytes, prompt=""):
            return self.answer

    args = {"attachment_key": "s", "page": 0, "bbox": [560, 370, 680, 450]}
    hedged = json.loads(_dispatch_render_region(
        dict(args), Hedges("[G/C]CE px=[10, 10, 40, 20]"), {"s": pdf}))
    assert "closer zoom" in hedged["reading_note"]
    plain = json.loads(_dispatch_render_region(
        dict(args), Hedges("GCE px=[10, 10, 40, 20]"), {"s": pdf}))
    assert "reading_note" not in plain
    assert vision_view.bracketed_note("G[C/O]E") is not None
    assert vision_view.bracketed_note("box [10, 20, 30, 40]") is None


# -- K: whole 32 px patches, behind a switch ----------------------------------------

def test_patch_alignment_is_off_by_default(monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    data, info = vision_view.render_view(_sheet(TARGET), page=0)
    assert "patch_aligned" not in info
    assert (info["width_px"], info["height_px"]) == (2048, 1325)


def test_patch_alignment_pads_to_whole_patches_and_keeps_boxes_exact(
        monkeypatch):
    """GPT-5.4's pixel boxes were stretched in y by the height rounded up to
    whole 32 px patches (1344 / 1325). With the switch on, the image already
    IS whole patches: white paper added at the right and bottom, the view
    widened by exactly what it shows — a box converted with the size sent
    lands where it did before."""
    monkeypatch.setenv(vision_view.PATCH_ALIGN_ENV, "1")
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    pdf = _sheet(TARGET)
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find", "tiles": "off"},
        Sees(), {"s": pdf}))
    assert out["view_px"] == [2048, 1344]
    assert out["view"][3] == pytest.approx(792 * 1344 / 1325, abs=0.1)
    boxes = [[int(v) for v in m] for m in re.findall(
        r"\[(\d+), (\d+), (\d+), (\d+)\]", out["analysis"])]
    page_box = vision_view.image_box_to_page(out["view"], boxes[0])
    assert all(abs(a - b) < 1.5 for a, b in zip(page_box, TARGET))
    # zooms are aligned too
    zoom = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "bbox": [560, 370, 680, 450]},
        Sees(), {"s": pdf}))
    assert all(v % 32 == 0 for v in zoom["view_px"])


def test_patch_alignment_leaves_an_aligned_or_over_cap_image_alone():
    info = {"width_px": 64, "height_px": 32, "clip": [0, 0, 10, 5],
            "format": "png"}
    assert vision_view.align_to_patches(b"x", info) == (b"x", info)
    info2 = {"width_px": 2000, "height_px": 1000, "clip": [0, 0, 10, 5],
             "format": "png"}
    assert vision_view.align_to_patches(b"x", info2, cap=2000) == (b"x", info2)
