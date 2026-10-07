"""Page locations the agent can rely on — the live check of 2026-10-07.

Measured on Funhouse (GPT-5.4; module_work/harness_theory/
locating_things_on_a_page.md §5.1): on one whole-sheet image the model's
0-999 boxes were 57-91 pt off with a scale that changed between identical
calls, while its PIXEL boxes, converted with the image's true size, were
within 1-6 pt; its own statement of the image's size was wrong; the host
shrank anything over 2,048 px, so the lettering the app thought was 14 px
arrived at 7 px and auto-tiling never fired; zooms on reported boxes came
back blank because the window was padded by 15 % of the box; and
``tiles="6x6"`` was silently read as no tiles.

A stand-in engine here SEES: it finds the ink in the image it is sent and
answers in pixels, the way the model was asked to — so these tests prove
the conversion, the cap, the padding and the tiles end to end, offline.
"""

from __future__ import annotations

import json
import re

import pytest

fitz = pytest.importorskip("fitz")
np = pytest.importorskip("numpy")

from funhouse_agent import vision_probe, vision_view  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_render_region, _parse_tiles,
)

#: A small inked mark on a tabloid sheet (PDF points, top-left origin).
MARK = (610.0, 404.0, 632.0, 413.0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV,
                vision_view.POLICY_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    vision_probe.clear_cache()
    yield
    vision_probe.clear_cache()


@pytest.fixture
def sheet():
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)
    page.draw_rect(fitz.Rect(*MARK), color=(0, 0, 0), fill=(0, 0, 0))
    data = doc.tobytes()
    doc.close()
    return data


def _ink_px(image_bytes):
    pix = fitz.Pixmap(image_bytes)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, 0]
    ys, xs = np.nonzero(a < 100)
    return (pix.width, pix.height), [int(xs.min()), int(ys.min()),
                                     int(xs.max()) + 1, int(ys.max()) + 1]


class SeeingEngine:
    """Answers with the pixel box of the ink it sees — and, like GPT-5.4 on
    2026-10-07, states a WRONG size for the image."""

    def __init__(self, style="px"):
        self.style, self.prompts, self.sizes = style, [], []

    def analyze_image(self, image_bytes, prompt=""):
        self.prompts.append(prompt)
        size, box = _ink_px(image_bytes)
        self.sizes.append(size)
        told = re.search(r"This image is (\d+) x (\d+) pixels", prompt)
        assert told and (int(told.group(1)), int(told.group(2))) == size
        wrong = f"SIZE={size[0]}x{round(size[1] * 1.03)}"
        if self.style == "px":
            return f"{wrong}\nTAG px=[{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
        if self.style == "suffix":
            return f"The tag is at [{box[0]}, {box[1]}, {box[2]}, {box[3]}] px."
        if self.style == "untagged":     # forgot the tag, but past 999
            return f"TAG [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
        raise AssertionError(self.style)


def _grid_boxes(text):
    return [[int(v) for v in m] for m in re.findall(
        r"\[(\d+), (\d+), (\d+), (\d+)\]", text)]


def _near(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# -- 1. pixel boxes in, exact 0-999 boxes out ----------------------------------

@pytest.mark.parametrize("style", ["px", "suffix", "untagged"])
def test_a_pixel_box_reaches_the_agent_as_an_exact_grid_box(sheet, style):
    engine = SeeingEngine(style)
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find the tag",
         "tiles": "off"}, engine, {"s": sheet}))
    assert "error" not in out, out
    assert "px=" not in out["analysis"]
    (box,) = _grid_boxes(out["analysis"])
    page_box = vision_view.image_box_to_page(out["view"], box)
    # within one grid step (1.2 pt across, 0.8 pt down) plus a pixel
    assert _near(page_box, MARK, 1.5), (page_box, MARK)
    assert "converted in code" in out["boxes"]
    assert f"{out['view_px'][0]} x {out['view_px'][1]} px" in out["boxes"]


def test_the_size_used_is_the_size_sent_not_the_size_the_model_states(sheet):
    engine = SeeingEngine("px")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find the tag",
         "tiles": "off"}, engine, {"s": sheet}))
    stated = re.search(r"SIZE=(\d+)x(\d+)", out["analysis"]).groups()
    assert int(stated[1]) != out["view_px"][1]           # the model is wrong
    (box,) = _grid_boxes(out["analysis"].split("\n", 1)[1])
    assert _near(vision_view.image_box_to_page(out["view"], box), MARK, 1.5)


def test_zooming_on_a_converted_box_finds_the_thing_and_reads_it_closely(sheet):
    engine = SeeingEngine("px")
    page = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "prompt": "find the tag",
         "tiles": "off"}, engine, {"s": sheet}))
    (box,) = _grid_boxes(page["analysis"])
    zoom = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "view": page["view"],
         "image_box": box, "prompt": "box the tag"}, engine, {"s": sheet}))
    assert "error" not in zoom, zoom
    vx0, vy0, vx1, vy1 = zoom["view"]
    assert vx0 < MARK[0] and vy0 < MARK[1] and vx1 > MARK[2] and vy1 > MARK[3]
    (zbox,) = _grid_boxes(zoom["analysis"])
    # from a zoom a few hundred points wide the grid step is a fraction of a
    # point: the box lands on the mark
    assert _near(vision_view.image_box_to_page(zoom["view"], zbox), MARK, 0.6)
    assert "may be anchored" in zoom["precision"]


def test_an_old_style_grid_answer_still_parses():
    text, counts = vision_view.boxes_to_grid(
        "the rectangle is at [490, 505, 572, 568]", (2048, 1326))
    assert text == "the rectangle is at [490, 505, 572, 568]"
    assert counts == {"converted": 0, "grid": 1}
    assert "without units" in vision_view.boxes_note(counts, (2048, 1326))


def test_box_conversion_rules():
    size = (2048, 1326)
    text, counts = vision_view.boxes_to_grid(
        "A px=[1024, 663, 1034, 673]; B px: [0, 0, 2048, 1326]; "
        "C [100, 50, 140, 60] px; D [100, 50, 140, 60]; "
        "readings [5.3, 4.1, 3.2, 2.5]; prose (1, 2, 3, 4)", size)
    assert "A [500, 500, 504, 507]" in text
    assert "B [0, 0, 999, 999]" in text
    assert "C [49, 38, 68, 45]" in text
    assert "D [49, 38, 68, 45]" in text          # untagged, in a pixel answer
    assert "readings [5.3, 4.1, 3.2, 2.5]" in text   # not a box: left alone
    assert "prose (1, 2, 3, 4)" in text
    assert counts == {"converted": 4, "grid": 1}
    # a value past 999 is pixels even with no tag anywhere
    text, counts = vision_view.boxes_to_grid("X [1500, 10, 1520, 20]", size)
    assert text == "X [732, 8, 741, 15]" and counts["converted"] == 1
    # no size: nothing is touched
    assert vision_view.boxes_to_grid("px=[1, 2, 3, 4]", None)[0] == \
        "px=[1, 2, 3, 4]"


def test_a_chart_read_off_converts_only_tagged_boxes(monkeypatch, tmp_path,
                                                     sheet):
    """A chart read-off's answer is about VALUES: a bracketed list of four
    readings must never be rewritten as a box, whatever its shape."""
    from funhouse_agent import vision_tools
    pdf = tmp_path / "ref.pdf"
    pdf.write_bytes(sheet)
    from geotech_references import _figures_db
    monkeypatch.setattr(_figures_db, "figure_get", lambda r, f: {
        "figure_number": f, "caption": "a chart", "page_estimated": False})
    monkeypatch.setattr(_figures_db, "resolve_pdf", lambda r, f: (pdf, 0))

    class Reader:
        def analyze_image(self, image_bytes, prompt=""):
            return ("Kp at 30, 35, 40, 45 deg: [500, 1200, 900, 1300]; the "
                    "curve label is at px=[1000, 500, 1100, 540]")

    out = json.loads(vision_tools._dispatch_read_reference_figure(
        {"reference": "dm7_1", "figure_number": "5-6", "prompt": "read Kp"},
        Reader()))
    assert "[500, 1200, 900, 1300]" in out["analysis"]
    assert "px=" not in out["analysis"]
    assert out["analysis"].count("[") == 2


def test_px_box_to_page_is_exact():
    view = [100.0, 200.0, 1100.0, 700.0]
    assert vision_view.px_box_to_page(view, [0, 0, 2000, 1000],
                                      (2000, 1000)) == tuple(view)
    assert vision_view.px_box_to_page(view, [500, 250, 1000, 500],
                                      (2000, 1000)) == (350.0, 325.0, 600.0,
                                                        450.0)
    with pytest.raises(ValueError):
        vision_view.px_box_to_page([0, 0, 0, 1], [0, 0, 1, 1], (10, 10))


def test_structured_locations_in_pixels_are_converted_exactly():
    text, items = vision_view.split_located(
        'Answer.\nLOCATED: [{"what": "tag", "px": [1000, 500, 1050, 520]}, '
        '{"what": "old", "box": [100, 100, 200, 200]}]',
        (0, 0, 1224, 792), (2048, 1326))
    assert text == "Answer."
    px_item, old = items
    assert px_item["page_bbox"] == pytest.approx(
        [597.7, 298.6, 627.5, 310.6], abs=0.1)
    assert px_item["image_box"] == [488, 377, 512, 392]
    # read off the whole sheet: a padded window to zoom on comes with it
    zx0, zy0, zx1, zy1 = px_item["zoom_bbox"]
    assert zx0 < 597.7 - 100 and zx1 > 627.5 + 100
    assert old["page_bbox"] == pytest.approx([122.5, 79.3, 245.1, 158.6],
                                             abs=0.1)
    # from a zoom there is no need
    _t, (zoomed,) = vision_view.split_located(
        'x\nLOCATED: [{"what": "tag", "px": [10, 10, 40, 30]}]',
        (500, 400, 700, 500), (2048, 1024))
    assert "zoom_bbox" not in zoomed


def test_the_text_layer_is_given_in_the_same_pixels():
    lines = [("GENERAL NOTE A", (100.0, 100.0, 200.0, 110.0))]
    block = vision_view.text_context(lines, (0, 0, 1224, 792), size=(2048, 1326))
    assert "px=[167, 167, 335, 184] GENERAL NOTE A" in block
    assert "in pixels of the image" in block


# -- 2. never send more than the host delivers ----------------------------------

def test_the_default_cap_and_its_override(monkeypatch):
    assert vision_view.max_px() == vision_view.DEFAULT_MAX_PX == 2048
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    bud = vision_view.budget()
    assert bud.name == "openai-original" and bud.max_edge == 2048
    assert bud.detail == "original"            # Funhouse needs it at 2048 too
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "none")
    assert vision_view.budget().max_edge == 6000
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "3000")
    assert vision_view.budget().max_edge == 3000
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "nonsense")
    assert vision_view.max_px() == 2048


def test_a_page_at_the_original_budget_is_sent_at_2048_px(sheet, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    data, info = vision_view.render_view(sheet, page=0)
    assert max(info["width_px"], info["height_px"]) == 2048
    assert (info["width_px"], info["height_px"]) == (2048, 1325)


def test_the_fixed_sizes_are_held_to_the_cap_too(sheet, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "none")
    data, info = vision_view.render_view(sheet, page=0, bbox=[0, 0, 1224, 792],
                                         pad_frac=0.0)    # 300 dpi: 5100 px
    assert max(info["width_px"], info["height_px"]) <= 2048


def test_small_lettering_is_tiled_at_the_size_actually_sent(monkeypatch):
    """The 2026-10-07 failure: a 3957 px render made 0.06 in lettering look
    14 px tall, so no tiles — but the host sent 2048 px, where it is 7 px.
    At the size actually sent the rule fires (3 x 3)."""
    tags = pytest.importorskip("planlens.testing.tag_fixtures")
    gt = tags.build_synthetic_tag_set()
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")

    class Quiet:
        def __init__(self):
            self.calls = 0

        def analyze_image(self, image_bytes, prompt=""):
            self.calls += 1
            return "nothing"

    eng = Quiet()
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "t", "page": 0, "prompt": "find GCE"}, eng,
        {"t": gt.pdf}))
    assert max(out["view_px"]) == 2048
    assert len(out["tiles"]) == 9 and eng.calls == 10
    assert "TILE's view" in out["tiling"]
    # uncapped (a host that delivers the full image), the rule does not fire
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "none")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "t", "page": 0, "prompt": "find GCE"}, Quiet(),
        {"t": gt.pdf}))
    assert max(out["view_px"]) > 3900 and "tiles" not in out


# -- 3. tiles --------------------------------------------------------------------

@pytest.mark.parametrize("arg,value,noted", [
    ("auto", "auto", False), (None, "auto", False), ("", "auto", False),
    ("off", "off", False), ("1", "off", False), (False, "off", False),
    (3, 3, False), ("3", 3, False), ("3x3", 3, False), ("3 X 3", 3, False),
    ("2×2", 2, False), ("6x6", 4, True), ("8", 4, True),
])
def test_tiles_values_that_mean_something(arg, value, noted):
    got, note, error = _parse_tiles(arg)
    assert error is None and got == value and bool(note) is noted


@pytest.mark.parametrize("arg", ["3x4", "banana", "4 by 4", "-2"])
def test_tiles_values_that_mean_nothing_fail_loudly(arg, sheet):
    engine = SeeingEngine("px")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "tiles": arg}, engine, {"s": sheet}))
    assert "error" in out and "NxN" in out["accepted"]
    assert engine.prompts == []                      # nothing was spent


def test_six_by_six_is_read_as_four_by_four_and_says_so(sheet):
    class Count:
        calls = 0

        def analyze_image(self, image_bytes, prompt=""):
            Count.calls += 1
            return "nothing"

    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "tiles": "6x6"}, Count(),
        {"s": sheet}))
    assert len(out["tiles"]) == 16 and Count.calls == 17
    assert "4x4 is the most made" in out["tiles_note"]


# -- 4. zoom windows sized by the source view's error ------------------------------

def test_a_zoom_from_a_zoom_is_padded_by_the_floor(sheet):
    view = [600.0, 380.0, 700.0, 440.0]                   # 100 x 60 pt
    box = [round((MARK[0] - 600) / 100 * 999), round((MARK[1] - 380) / 60 * 999),
           round((MARK[2] - 600) / 100 * 999), round((MARK[3] - 380) / 60 * 999)]
    out = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "view": view, "image_box": box},
        SeeingEngine("px"), {"s": sheet}))
    assert out["window_padding_pt"] == [12.0, 12.0]
    assert "look again rather than concluding it is absent" in out["window_note"]


def test_a_bbox_zoom_keeps_its_own_padding(sheet):
    out = json.loads(_dispatch_render_region(
        {"attachment_key": "s", "page": 0, "bbox": [600, 400, 700, 450]},
        SeeingEngine("px"), {"s": sheet}))
    assert "window_padding_pt" not in out
    x0, y0, x1, y1 = out["view"]
    assert 580 < x0 < 600 and 700 < x1 < 720         # 15 % of the box


def test_every_result_says_how_far_its_boxes_can_be_trusted(sheet):
    page = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "s", "page": 0, "tiles": "off"}, SeeingEngine("px"),
        {"s": sheet}))
    assert "NOT for placing a mark" in page["precision"]
    assert "~122 x 79 pt" in page["precision"]
    small = vision_view.precision_note([0, 0, 280, 200])
    assert "may be anchored on this view" in small


def test_the_location_error_has_a_floor():
    assert vision_view.location_error([0, 0, 1224, 792]) == \
        pytest.approx((122.4, 79.2))
    assert vision_view.location_error([0, 0, 50, 40]) == (12.0, 12.0)
    # a large box keeps planlens' own 15 % pad when that is more
    assert vision_view.zoom_pad([0, 0, 100, 100], [0, 0, 400, 100]) == \
        pytest.approx((60.0, 60.0))
