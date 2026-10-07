"""annotate_document checks where its marks landed (field report 2026-10-01).

A stand-in vision engine looks at the crop it is shown the way the check needs
a real one to: ink in the middle of the crop means the mark encloses
something, blank paper means it does not. So these tests prove the crop is
taken from the MARKED copy and centred on each mark, that only marks placed by
location are checked, and that the verdicts reach the tool result.
"""

import io
import json
import os

import pytest

fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL.Image")

from funhouse_agent import document_tools, markup_check  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402


def _needs_circle():
    try:
        from planlens.document.markup_writer import KINDS
    except ImportError:
        return True
    return "circle" not in KINDS


pytestmark = pytest.mark.skipif(
    not document_tools.has_tool("annotate_document") or _needs_circle(),
    reason="installed planlens predates annotate_document circles (0.11)")

#: Where the tag is drawn on the test sheet (PDF points, top-left origin).
TAG = (640.0, 515.0, 662.0, 535.0)


def _sheet() -> bytes:
    """A tabloid sheet with one small inked tag and a lot of blank paper."""
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)
    page.draw_rect(fitz.Rect(*TAG), color=(0, 0, 0), width=1.2)
    page.insert_text((TAG[0] + 3, TAG[3] - 6), "GCE", fontsize=8)
    page.insert_text((80, 80), "SHEET NOTES: penetrations shall be core "
                     "drilled.", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


class PixelEngine:
    """'Sees' ink in the middle of the crop; red (the mark itself) is not
    ink. Records every prompt."""

    def __init__(self):
        self.prompts = []

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        img = PIL.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        dark = 0
        for y in range(int(h * 0.3), int(h * 0.7), 2):
            for x in range(int(w * 0.3), int(w * 0.7), 2):
                r, g, b = img.getpixel((x, y))
                if r < 110 and g < 110 and b < 110:
                    dark += 1
        if dark > 3:
            return json.dumps({"encloses": True, "inside": "GCE",
                               "sure": True})
        return json.dumps({"encloses": False, "inside": "nothing",
                           "sure": True})


def _tools(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    tools = {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"sheet.pdf": _sheet()})}
    handle = json.loads(tools["open_document"].invoke(
        {"source": "sheet.pdf"}))["handle"]
    return tools, handle


def test_a_misplaced_mark_is_caught_and_a_right_one_confirmed(tmp_path,
                                                             monkeypatch):
    engine = PixelEngine()
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    # The tag as a look would report it: the render's view and the tag's
    # 0-999 box on it (view [600,480,700,560] -> TAG).
    view = [600.0, 480.0, 700.0, 560.0]
    ibox = [round((TAG[0] - 600) / 100 * 999), round((TAG[1] - 480) / 80 * 999),
            round((TAG[2] - 600) / 100 * 999), round((TAG[3] - 480) / 80 * 999)]
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "checked.pdf", "markups": [
            {"kind": "highlight", "page": 0, "comment": "skipped: no such "
             "words", "quote": "NOT ON THIS SHEET AT ALL"},
            {"kind": "circle", "page": 0, "comment": "penetration tag",
             "label": "GCE", "view": view, "image_box": ibox},
            {"kind": "circle", "page": 0, "comment": "penetration tag",
             "label": "GCE", "bbox": [430, 250, 500, 320]},     # empty paper
            {"kind": "note", "page": 0, "comment": "general note",
             "point": [80, 60]},
            {"kind": "highlight", "page": 0, "comment": "by quote",
             "quote": "core drilled"},
        ]}))
    assert out["n_written"] == 4 and out["n_skipped"] == 1
    check = out["check"]
    # only the two circles were placed by location; the note and the
    # quote-anchored highlight are not checked
    assert check["checked"] == 2 and check["confirmed"] == 1
    assert [m["index"] for m in check["misplaced"]] == [2]
    bad = check["misplaced"][0]
    assert bad["pdf_page"] == 1 and bad["names"] == "GCE"
    assert "append=false" in check["note"]
    # the look was asked about the label, and told the label is not the thing
    assert all("GCE" in p and "label" in p for p in engine.prompts)


def test_a_ring_round_the_right_thing_but_far_too_wide_is_not_confirmed(
        tmp_path, monkeypatch):
    """Foundry rc3 (2026-10-04): an agent widened misplaced rings until each
    took its tag in somewhere, and the check confirmed rings ten times the
    tag's size. Encloses-but-not-close is reported, not confirmed."""

    class WideEngine:
        """Sees the tag as a small box in the middle of the crop."""

        def __init__(self):
            self.prompts = []

        def analyze_image(self, image_bytes, prompt):
            self.prompts.append(prompt)
            return json.dumps({"encloses": True, "inside": "GCE",
                               "thing_box": [490, 495, 510, 505],
                               "sure": True})

    engine = WideEngine()
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    big = [TAG[0] - 60, TAG[1] - 40, TAG[2] + 60, TAG[3] + 40]
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "wide.pdf", "markups": [
            {"kind": "circle", "page": 0, "label": "GCE", "bbox": big}]}))
    check = out["check"]
    assert check["confirmed"] == 0 and len(check["misplaced"]) == 1
    assert "times the area" in check["misplaced"][0]["seen"]
    assert "zoom until you can box the thing itself" in check["note"]
    # The look is asked where the thing is in PIXELS of the crop it was
    # sent (this old-style 0-999 answer is still understood).
    assert "thing_px" in engine.prompts[0] and "pixel image" in engine.prompts[0]


def test_no_mark_from_a_wide_view_or_the_whole_view(tmp_path, monkeypatch):
    """Live check 2026-10-07: every ring placed from a whole-sheet look
    missed (20-90 pt off 10 pt tags), and a ring given the whole zoom window
    ([0, 0, 999, 999]) blanketed it. planlens refuses both, per mark, and
    says to anchor on a zoom; the check only looks at what went on."""
    from planlens.document import markup_writer
    if not hasattr(markup_writer, "VIEW_ANCHOR_MAX_PT"):
        pytest.skip("installed planlens predates the wide-view refusal")
    engine = PixelEngine()
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    sheet = [0.0, 0.0, 1224.0, 792.0]
    on_sheet = [round(TAG[0] / 1224 * 999), round(TAG[1] / 792 * 999),
                round(TAG[2] / 1224 * 999), round(TAG[3] / 792 * 999)]
    zoom = [600.0, 480.0, 700.0, 560.0]
    on_zoom = [round((TAG[0] - 600) / 100 * 999), round((TAG[1] - 480) / 80 * 999),
               round((TAG[2] - 600) / 100 * 999), round((TAG[3] - 480) / 80 * 999)]
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "anchors.pdf", "markups": [
            {"kind": "circle", "page": 0, "label": "GCE",
             "view": sheet, "image_box": on_sheet},
            {"kind": "circle", "page": 0, "label": "GCE",
             "view": zoom, "image_box": [0, 0, 999, 999]},
            {"kind": "circle", "page": 0, "label": "GCE",
             "view": zoom, "image_box": on_zoom}]}))
    assert out["n_written"] == 1 and out["n_skipped"] == 2
    reasons = {s["index"]: s["reason"] for s in out["skipped"]}
    assert "1224 x 792 pt view" in reasons[0] and "ZOOM's view" in reasons[0]
    assert "whole 100 x 80 pt view" in reasons[1]
    assert out["check"]["checked"] == 1 and out["check"]["confirmed"] == 1


def test_ring_size_is_measured_not_judged():
    """Funhouse GPT-5.4, 2026-10-07: asked whether rings were "drawn
    closely", the model rejected a 19 x 14 pt ring centred on a 10 x 4 pt tag.
    Size is now the mark's area over the thing's, from the look's box."""
    from funhouse_agent.markup_check import MAX_AREA_FACTOR, _too_wide
    view = [0.0, 0.0, 100.0, 100.0]                 # 1 grid unit ~ 0.1 pt
    tag = [450, 480, 550, 520]                      # a 10 x 4 pt tag
    # the snug rings of the good runs: 19 x 14 and 32 x 18 pt -> fine
    assert _too_wide([40.5, 43, 59.5, 57], tag, view) is None
    assert _too_wide([34, 41, 66, 59], tag, view) is None
    # the blanket rings of 2026-10-04: 73 x 43 and 125 x 66 pt -> too wide
    assert _too_wide([13, 28, 86, 71], tag, view) > MAX_AREA_FACTOR
    assert _too_wide([-12, 17, 113, 83], tag, view) > MAX_AREA_FACTOR
    # no usable box from the look -> size is not judged
    assert _too_wide([13, 28, 86, 71], None, view) is None
    assert _too_wide([13, 28, 86, 71], [1, 2, 3], view) is None


def test_the_thing_is_measured_in_pixels_of_the_crop(tmp_path, monkeypatch):
    """thing_px is converted with the crop's own size: the same blanket ring
    is caught from a pixel answer as from a grid one, and a snug ring
    passes."""
    import re as _re

    class PxEngine:
        def __init__(self, frac):
            self.frac, self.prompts = frac, []

        def analyze_image(self, image_bytes, prompt):
            self.prompts.append(prompt)
            w, h = (int(v) for v in _re.search(
                r"this (\d+) x (\d+) pixel image", prompt).groups())
            f = self.frac
            return json.dumps({"encloses": True, "inside": "GCE",
                               "thing_px": [w * (0.5 - f), h * (0.5 - f),
                                            w * (0.5 + f), h * (0.5 + f)],
                               "sure": True})

    big = [TAG[0] - 60, TAG[1] - 40, TAG[2] + 60, TAG[3] + 40]
    tools, handle = _tools(PxEngine(0.01), tmp_path, monkeypatch)
    wide = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "px_wide.pdf", "markups": [
            {"kind": "circle", "page": 0, "label": "GCE", "bbox": big}]}))
    assert wide["check"]["confirmed"] == 0
    assert "times the area" in wide["check"]["misplaced"][0]["seen"]
    tools, handle = _tools(PxEngine(0.16), tmp_path, monkeypatch)
    snug = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "px_snug.pdf", "markups": [
            {"kind": "circle", "page": 0, "label": "GCE", "bbox": list(TAG)}]}))
    assert snug["check"]["confirmed"] == 1


def test_too_wide_reads_pixels_with_the_crop_size():
    from funhouse_agent.markup_check import _too_wide
    view = [0.0, 0.0, 100.0, 100.0]
    # a 10 x 4 pt tag, as pixels of a 1000 x 1000 crop and as a 0-999 box
    px = [450, 480, 550, 520]
    assert _too_wide([13, 28, 86, 71], px, view, (1000, 1000)) > 30
    assert _too_wide([40.5, 43, 59.5, 57], px, view, (1000, 1000)) is None
    # the same numbers on a 2000 x 2000 crop are a thing half as wide each
    # way: a 40 x 25 pt ring is 25 x its area on the first, too wide on the
    # second — the size SENT decides
    assert _too_wide([30, 35, 70, 60], px, view, (1000, 1000)) is None
    assert _too_wide([30, 35, 70, 60], px, view, (2000, 2000)) > 30


def test_a_snug_ring_is_confirmed(tmp_path, monkeypatch):
    class SnugEngine:
        def analyze_image(self, image_bytes, prompt):
            # the tag fills about a third of the crop each way
            return json.dumps({"encloses": True, "inside": "GCE",
                               "thing_box": [340, 380, 660, 620],
                               "sure": True})

    tools, handle = _tools(SnugEngine(), tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "snug.pdf", "markups": [
            {"kind": "circle", "page": 0, "label": "GCE", "bbox": list(TAG)}]}))
    assert out["check"]["confirmed"] == 1 and not out["check"]["misplaced"]


def test_all_marks_right_says_so(tmp_path, monkeypatch):
    tools, handle = _tools(PixelEngine(), tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "ok.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": "tag", "bbox": list(TAG)}]}))
    assert out["check"]["confirmed"] == 1 and not out["check"]["misplaced"]
    assert "looked at" in out["check"]["note"]


def test_check_false_and_no_engine_leave_the_result_alone(tmp_path,
                                                         monkeypatch):
    engine = PixelEngine()
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "nocheck.pdf", "check": False,
        "markups": [{"kind": "box", "page": 0, "comment": "x",
                     "bbox": [430, 250, 500, 320]}]}))
    assert "check" not in out and engine.prompts == []
    tools2, handle2 = _tools(None, tmp_path, monkeypatch)
    out2 = json.loads(tools2["annotate_document"].invoke({
        "handle": handle2, "output_path": "noengine.pdf",
        "markups": [{"kind": "box", "page": 0, "comment": "x",
                     "bbox": [430, 250, 500, 320]}]}))
    assert "check" not in out2


def test_rows_pair_with_their_specs_past_skipped_ones():
    result = {"written": [{"kind": "box", "anchored_by": "bbox", "page": 0},
                          {"kind": "note", "anchored_by": "point", "page": 0},
                          {"kind": "circle", "anchored_by": "bbox",
                           "page": 2}],
              "skipped": [{"index": 0}, {"index": 2}]}
    specs = [{"comment": "a"}, {"comment": "b"}, {"comment": "c"},
             {"comment": "d"}, {"comment": "e", "label": "E"}]
    got = markup_check.marks_to_check(result, specs)
    assert [(g["index"], g["spec"].get("comment")) for g in got] == [
        (1, "b"), (4, "e")]


def test_a_failed_look_is_reported_not_raised(tmp_path, monkeypatch):
    class Broken:
        def analyze_image(self, image_bytes, prompt):
            raise RuntimeError("gateway closed the connection")
    tools, handle = _tools(Broken(), tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "broken.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": "x", "bbox": list(TAG)}]}))
    assert out["n_written"] == 1
    assert out["check"]["not_checked"][0]["reason"].startswith("RuntimeError")
    assert os.path.isfile(out["output_path"])
