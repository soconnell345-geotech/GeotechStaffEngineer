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
