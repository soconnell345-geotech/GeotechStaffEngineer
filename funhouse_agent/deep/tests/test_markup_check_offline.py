"""annotate_document checks where its marks landed (field report 2026-10-01).

A stand-in vision engine looks at the crop it is shown the way the check needs
a real one to: ink in the middle of the crop means the mark encloses
something, blank paper means it does not. So these tests prove the crop is
taken from the MARKED copy and centred on each mark, that only marks placed by
location are checked, and that the verdicts reach the tool result.
"""

import io
import json
import re
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
    # Every written mark is looked at — the note and the quote-anchored
    # highlight too (Foundry brief 4: unchecked anchors were the route round
    # a verdict). The note and the highlight sit on lettering, so this
    # stand-in sees ink there and confirms them.
    assert check["checked"] == 4 and check["confirmed"] == 3
    assert [m["index"] for m in check["misplaced"]] == [2]
    bad = check["misplaced"][0]
    # A label is display text, never the name (live smoke 2c, E3): with no
    # target, the mark is judged by what its comment is about.
    assert bad["pdf_page"] == 1 and bad["names"] == "penetration tag"
    assert "append=false" in check["note"]
    # the circles' looks were shown the label as display text and told it
    # is not the name of what is marked
    rings = [p for p in engine.prompts if "red ring" in p]
    assert len(rings) == 2
    assert all('The red label reads "GCE"' in p
               and "NOT the name of what is marked" in p for p in rings)
    assert not any("meant to enclose: GCE" in p for p in rings)


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
    # the note is checked too, since brief 4
    assert [(g["index"], g["spec"].get("comment")) for g in got] == [
        (1, "b"), (3, "d"), (4, "e")]


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
    # the model is handed the conversation's name for the copy, not a path
    assert out["output_path"] == "broken.pdf"
    assert os.path.isfile(os.path.join(str(tmp_path), out["output_path"]))


# ---------------------------------------------------------------------------
# Foundry brief 4 (2026-10-07): what a mark is compared with, every anchor
# checked, and enclosure measured
# ---------------------------------------------------------------------------

#: Two notes that both state a maximum slope; a review comment asks about the
#: second one's figure. Invented wording, the shape of the brief-4 sheet.
WRONG_LINE = "SLOPE UP TO 5% (6.2% MAX.)"
RIGHT_LINE = "SLOPE SHALL NOT EXCEED 6.25% MAX."
REQUEST = "DRAFT: Please confirm the 6.25% maximum applies over the full run."


def _notes_sheet() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)
    page.insert_text((60, 210), "3. CROSS SLOPE SHALL NOT EXCEED 2% MAX.",
                     fontsize=7)
    page.insert_text((60, 228), RIGHT_LINE, fontsize=7)
    page.insert_text((420, 430), WRONG_LINE, fontsize=7)
    data = doc.tobytes()
    doc.close()
    return data


def _tools_for(engine, tmp_path, monkeypatch, pdf, name="notes.pdf"):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    tools = {t.name: t for t in make_vision_tools(
        engine=engine, attachments={name: pdf})}
    handle = json.loads(tools["open_document"].invoke(
        {"source": name}))["handle"]
    return tools, handle


class ReadsThePrompt:
    """A stand-in that judges a mark the way the prompt asks: the thing named
    is what is there (an anchor on words lands on them), and the comment
    fits only if its figure is in that thing. Records every prompt."""

    def __init__(self, inside=None):
        self.prompts, self.inside = [], inside

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        named = re.search(r"meant to [a-z ]+: (.+?)   \(", prompt)
        target = named.group(1) if named else (self.inside or "")
        comment = re.search(r'review comment on this mark: "(.+?)"', prompt)
        figure = (re.search(r"\d+(?:\.\d+)?%", comment.group(1))
                  if comment else None)
        out = {"same_thing": True, "inside": target, "sure": True}
        if '"encloses"' in prompt:
            out["encloses"] = True
        if "comment_fits" in prompt:
            out["comment_fits"] = bool(figure and figure.group(0) in target)
        return json.dumps(out)


def test_a_comment_is_a_request_about_the_thing_not_its_name(tmp_path,
                                                            monkeypatch):
    """Brief 4: of 10 checks of comments, boxes and highlights placed on the
    right line, none was confirmed and 8 "misplaced" verdicts were wrong -
    the look was told the mark is meant to enclose "DRAFT: Please confirm
    ...", and answered that the line was not that. The comment is now shown
    as a request ABOUT the thing, and the look is asked whether the mark is
    on the thing the comment is about."""
    engine = ReadsThePrompt(inside=RIGHT_LINE)
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _notes_sheet())
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "request.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": REQUEST,
             "bbox": [58, 221, 200, 230]},
            {"kind": "callout", "page": 0, "comment": REQUEST,
             "points_at": [60, 226]}]}))
    check = out["check"]
    assert check["checked"] == 2 and check["confirmed"] == 2, check
    for p in engine.prompts:
        assert "meant to enclose: DRAFT" not in p
        assert "the thing that comment is about" in p
        assert "a remark or a request ABOUT the thing" in p
        assert '"same_thing"' in p
    # a callout's look reads the whole line at its tip, not one letter
    callout_p = [p for p in engine.prompts if "callout" in p][0]
    assert "WHOLE line" in callout_p and "tip of the arrow" in callout_p


def test_a_quoted_comment_on_the_wrong_line_is_caught(tmp_path, monkeypatch):
    """Brief 4, GPT-5.4: a callout on the wrong note was rejected twice,
    then re-placed BY QUOTE on that same note, where the check did not look -
    and that is the comment that reached the file. Quote-anchored marks and
    notes are now checked, and the question is whether the comment could be
    about the quoted words."""
    engine = ReadsThePrompt()
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _notes_sheet())
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "quoted.pdf", "markups": [
            {"kind": "callout", "page": 0, "comment": REQUEST,
             "quote": WRONG_LINE},
            {"kind": "note", "page": 0, "comment": REQUEST,
             "quote": RIGHT_LINE}]}))
    assert out["n_written"] == 2
    check = out["check"]
    assert check["checked"] == 2 and check["confirmed"] == 1
    (bad,) = check["misplaced"]
    assert bad["index"] == 0 and "something else" in bad["seen"]
    assert bad["names"] == WRONG_LINE
    assert "switching anchor does not settle a verdict" in check["note"]
    note_p = [p for p in engine.prompts if "sticky-note" in p][0]
    assert "printed words the mark was anchored on" in note_p
    assert "comment_fits" in note_p


def test_target_names_what_a_mark_is_on(tmp_path, monkeypatch):
    engine = ReadsThePrompt()
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _notes_sheet())
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "target.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": REQUEST,
             "target": RIGHT_LINE, "bbox": [58, 221, 200, 230]}]}))
    assert out["n_written"] == 1 and out["check"]["confirmed"] == 1
    (p,) = engine.prompts
    assert f"meant to enclose: {RIGHT_LINE}   (what the mark is on)" in p


class _AnswersWithBox:
    """Answers with a thing box placed in the crop by fractions of its size
    (pixels of the crop as sent), and the keys given."""

    def __init__(self, fx0, fy0, fx1, fy1, **keys):
        self.box, self.keys, self.prompts = (fx0, fy0, fx1, fy1), keys, []

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        w, h = (int(v) for v in re.search(
            r"this (\d+) x (\d+) pixel image", prompt).groups())
        fx0, fy0, fx1, fy1 = self.box
        return json.dumps({**self.keys, "thing_px": [
            w * fx0, h * fy0, w * fx1, h * fy1], "sure": True})


def _ring_check(engine, tmp_path, monkeypatch, name):
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    return json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": name, "markups": [
            {"kind": "circle", "page": 0, "comment": "penetration tag",
             "target": "GCE", "label": "GCE", "bbox": list(TAG)}]}))["check"]


def test_a_tight_ring_whose_look_counts_the_leader_end_is_confirmed(
        tmp_path, monkeypatch):
    """Brief 4, GPT-5.4 baseline: tight rings round T1 and T6 (1-5 pt from
    the tags, holding them whole) came back "encloses: false, inside - GCE"
    - the look took the end of the leader's shoulder for part of the tag and
    its box ran past the ring. Enclosure is measured now: the thing's centre
    is inside the ring, so the ring is confirmed, on the old answer and the
    new."""
    # the leader end + tag: wider than the ring, centred a little left
    leader_and_tag = (0.30, 0.47, 0.58, 0.53)
    old = _AnswersWithBox(*leader_and_tag, encloses=False, inside="- GCE")
    assert _ring_check(old, tmp_path, monkeypatch, "old.pdf")["confirmed"] == 1
    new = _AnswersWithBox(*leader_and_tag, same_thing=True, encloses=False,
                          inside="- GCE")
    assert _ring_check(new, tmp_path, monkeypatch, "new.pdf")["confirmed"] == 1


def test_a_labelled_ring_with_no_target_is_judged_by_its_comment(
        tmp_path, monkeypatch):
    """E3: a ring labelled "GCE" with no target is on the thing its comment
    is about; the look's box (leader end and tag) is measured as before."""
    leader_and_tag = (0.30, 0.47, 0.58, 0.53)
    eng = _AnswersWithBox(*leader_and_tag, same_thing=True, encloses=False,
                          inside="- GCE")
    tools, handle = _tools(eng, tmp_path, monkeypatch)
    check = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "label_only.pdf", "markups": [
            {"kind": "circle", "page": 0, "comment": "penetration tag",
             "label": "GCE", "bbox": list(TAG)}]}))["check"]
    assert check["confirmed"] == 1, check
    (p,) = eng.prompts
    assert "the thing that comment is about" in p
    assert "meant to enclose: GCE" not in p


#: A calc page like F38's: printed lines, each to be boxed with a reviewer's
#: VERDICT as its label (live smoke 2c, E3).
CALC_LINES = {
    "stab": ((60, 200), "M_stab = 460.1 kN-m/m"),
    "slide": ((60, 260), "FOS_sliding = 101.2/91.7 = 1.185"),
    "ecc": ((60, 320), "e = 3.00/2 - 1.056 = 0.444"),
}


def _calc_sheet() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for (x, y), text in CALC_LINES.values():
        page.insert_text((x, y), text, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def _line_box(key):
    (x, y), text = CALC_LINES[key]
    return [x - 2, y - 10, x + 5.2 * len(text), y + 3]


class ReadsThePrintedLine:
    """Knows which printed line each comment's box sits on, and answers as
    the look did in F38: what is there is the thing named only when the
    name is in the printed words; with nothing named, it is the thing the
    comment is about."""

    def __init__(self, printed_by_comment):
        self.printed, self.prompts = printed_by_comment, []

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        comment = re.search(r'review comment on this mark: "(.+?)"',
                            prompt).group(1)
        printed = self.printed[comment]
        named = re.search(r"meant to [a-z ]+: (.+?)   \(", prompt)
        fold = markup_check._fold
        same = (fold(named.group(1)) in fold(printed)
                or fold(printed) in fold(named.group(1))) if named else True
        out = {"same_thing": same, "inside": printed, "sure": True}
        if '"encloses"' in prompt:
            out["encloses"] = True
        if "comment_fits" in prompt:
            out["comment_fits"] = True
        return json.dumps(out)


def test_a_verdict_label_is_never_taken_for_the_things_name(tmp_path,
                                                            monkeypatch):
    """F38 t3: 18 boxes on the right numbers, labelled with the reviewer's
    verdicts ("460.1 vs 455.4", "= 1.10, not 1.185", "e = 0.444 ok"). The
    check named each thing by its label, confirmed 8 of 18, and the rebuild
    that passed dropped every label. Labels are display text now: the boxes
    are judged by their comments, confirmed with their labels on, and a
    misplaced mark's note says to keep the labels."""
    comments = {
        "stab": "DRAFT: the total moment does not follow from the table.",
        "slide": "DRAFT: 101.2 / 91.7 = 1.104, not 1.185.",
        "ecc": "DRAFT: eccentricity uses the uncorrected moment.",
        "wrong": "DRAFT: check the sliding factor of safety here.",
    }
    printed = {comments[k]: CALC_LINES[k][1] for k in CALC_LINES}
    printed[comments["wrong"]] = CALC_LINES["ecc"][1]     # boxed the e line
    engine = ReadsThePrintedLine(printed)
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _calc_sheet(),
                               name="calc.pdf")
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "calc_redline.pdf", "markups": [
            {"kind": "box", "page": 0, "bbox": _line_box("stab"),
             "label": "460.1 vs 455.4", "comment": comments["stab"]},
            {"kind": "box", "page": 0, "bbox": _line_box("slide"),
             "label": "= 1.10, not 1.185", "comment": comments["slide"]},
            {"kind": "box", "page": 0, "bbox": _line_box("ecc"),
             "label": "e = 0.444 ok", "comment": comments["ecc"]},
            # named, and on the wrong line: still caught
            {"kind": "box", "page": 0, "bbox": _line_box("ecc"),
             "target": "FOS_sliding", "label": "1.10?",
             "comment": comments["wrong"]}]}))
    assert out["n_written"] == 4
    check = out["check"]
    assert check["checked"] == 4 and check["confirmed"] == 3, check
    (bad,) = check["misplaced"]
    assert bad["index"] == 3 and bad["names"] == "FOS_sliding"
    assert "add target= and keep your labels" in check["note"]
    # every label was shown as display text, none as the thing's name
    for label in ("460.1 vs 455.4", "= 1.10, not 1.185", "e = 0.444 ok",
                  "1.10?"):
        assert any(f'The red label reads "{label}"' in p
                   for p in engine.prompts)
        assert not any(f"meant to enclose: {label}" in p
                       for p in engine.prompts)
    # the labels are on the delivered copy
    assert all(row.get("label_bbox") for row in out["written"])


def test_a_look_alike_inside_the_ring_is_still_misplaced(tmp_path,
                                                         monkeypatch):
    """The look's reading still says WHICH thing it is: a QCE in the ring
    (brief 4, Sol r2) is not a GCE however well the ring fits it."""
    centred = (0.45, 0.47, 0.55, 0.53)
    new = _AnswersWithBox(*centred, same_thing=False, encloses=True,
                          inside="QCE")
    check = _ring_check(new, tmp_path, monkeypatch, "qce_new.pdf")
    assert check["confirmed"] == 0 and check["misplaced"][0]["seen"] == "QCE"
    old = _AnswersWithBox(*centred, encloses=False, inside="QCE")
    assert _ring_check(old, tmp_path, monkeypatch,
                       "qce_old.pdf")["confirmed"] == 0


def test_the_thing_beside_the_ring_is_misplaced(tmp_path, monkeypatch):
    corner = (0.02, 0.02, 0.20, 0.12)
    eng = _AnswersWithBox(*corner, same_thing=True, encloses=True,
                          inside="GCE")
    check = _ring_check(eng, tmp_path, monkeypatch, "beside.pdf")
    assert check["confirmed"] == 0
    assert "beside the mark" in check["misplaced"][0]["seen"]


def test_crops_hold_the_whole_line_at_a_callout_tip_or_a_note():
    tip = markup_check._crop_box({"kind": "callout", "points_at": [60, 227],
                                  "bbox": [57, 153, 330, 229]})
    assert tip[2] - tip[0] >= 240 and tip[0] < 60 < tip[2]
    note = markup_check._crop_box({"kind": "note",
                                   "bbox": [143.0, 227.3, 159.0, 243.3]})
    assert note[2] - note[0] >= 240 and note[0] < 143 < note[2]
    assert note[1] < 227.3 < note[3]


def _needs_brief4_planlens():
    from planlens.document import markup_writer
    if not hasattr(markup_writer, "KIND_ALIASES"):
        pytest.skip("installed planlens predates the forgiven guesses")


def test_the_obvious_guesses_cost_no_round_trip(tmp_path, monkeypatch):
    """Brief 4: 8 of 14 marking runs lost a round trip to a refused
    ``color``, kind ``comment`` / ``text`` or a nested ``anchor``. They are
    read as what they mean, said in ``adjusted``, and checked like any
    other mark."""
    _needs_brief4_planlens()
    engine = ReadsThePrompt()
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _notes_sheet())
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "guesses.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": REQUEST, "color": "#FF0000",
             "target": RIGHT_LINE, "bbox": [58, 221, 200, 230]},
            {"kind": "comment", "page": 0, "comment": REQUEST,
             "quote": RIGHT_LINE},
            {"kind": "note", "page": 0, "comment": REQUEST,
             "anchor": {"quote": WRONG_LINE}}]}))
    assert "error" not in out, out
    assert out["n_written"] == 3
    assert [a["index"] for a in out["adjusted"]] == [0, 1, 2]
    check = out["check"]
    assert check["checked"] == 3 and check["confirmed"] == 2
    # the nested anchor was read as a quote, so the check knew what it was on
    assert [m["index"] for m in check["misplaced"]] == [2]
    assert check["misplaced"][0]["names"] == WRONG_LINE


def test_one_tag_ringed_twice_is_flagged(tmp_path, monkeypatch):
    """Brief 4, GPT-5.4 r2: one tag ringed twice (from two zooms), both
    rings confirmed, and the answer counted eight tags on a sheet of seven."""
    _needs_brief4_planlens()
    tools, handle = _tools(PixelEngine(), tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "twice.pdf", "markups": [
            {"kind": "circle", "page": 0, "comment": "tag", "label": "GCE",
             "bbox": list(TAG)},
            {"kind": "circle", "page": 0, "comment": "tag", "label": "GCE",
             "bbox": [TAG[0] + 1, TAG[1] + 0.5, TAG[2] + 1.2, TAG[3] + 0.4]},
        ]}))
    assert out["n_written"] == 2
    (dup,) = out["duplicates"]
    assert (dup["index"], dup["same_as"], dup["kind"], dup["label"]) == (
        1, 0, "circle", "GCE")
    assert dup["overlap"] > 0.5
    assert "count each thing once" in out["duplicates_note"]


def test_a_nested_anchor_is_read_the_way_planlens_reads_it():
    spec = markup_check._spec_view({"kind": "note", "comment": "c",
                                    "anchor": {"quote": "THE WORDS"}})
    assert spec["quote"] == "THE WORDS" and "anchor" not in spec
    assert markup_check._target(spec) == ("THE WORDS", "quote")
    assert markup_check._target({"label": "GCE", "quote": "x",
                                 "target": "the tag"}) == ("the tag", "target")
    # a label never names the thing (E3)
    assert markup_check._target({"label": "460.1?", "quote": "x"}) == (
        "x", "quote")
    assert markup_check._target({"label": "460.1?"}) == ("", "")


# ---------------------------------------------------------------------------
# Live smoke 2a (2026-10-09): a mark in the blank part of an area (B8), the
# labels (B1/B8), one frame on rotated pages (B1), no server path (B9)
# ---------------------------------------------------------------------------

#: A title strip down the left of a portrait sheet: lettering at its top,
#: blank paper below.
STRIP = (30.0, 100.0, 130.0, 700.0)


def _strip_sheet() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(*STRIP), color=(0, 0, 0), width=1.2)
    page.insert_text((40, 130), "COUNTY STANDARDS", fontsize=8)
    page.insert_text((40, 150), "STD. NO. 21.01", fontsize=8)
    page.insert_text((200, 300), "GENERAL NOTES: SEE SPECIFICATIONS.",
                     fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


class LooksAtAnArea:
    """Answers as a look at a mark in blank paper would: nothing in it; the
    named thing is an area whose box covers the crop fractions given; a
    stamp-like comment fits."""

    def __init__(self, area_frac, in_area=True):
        self.area, self.in_area, self.prompts = area_frac, in_area, []

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        w, h = (int(v) for v in re.search(
            r"this (\d+) x (\d+) pixel image", prompt).groups())
        fx0, fy0, fx1, fy1 = self.area
        out = {"same_thing": False, "inside": "nothing", "encloses": False,
               "thing_px": [w * fx0, h * fy0, w * fx1, h * fy1],
               "sure": True}
        if "in_area" in prompt:
            out["in_area"] = self.in_area
        if "comment_fits" in prompt:
            out["comment_fits"] = True
        return json.dumps(out)


def test_a_note_in_the_blank_part_of_the_area_it_names_is_on_it(
        tmp_path, monkeypatch):
    """F04: "put a note on the title block saying 'Checked - live smoke
    test'". The agent boxed the blank part of the title strip, named it
    ``target: title block``, and the look saw blank paper and said the
    comment was about something else, so the note never became visible. A
    mark inside the area it names is on that area; a stamp fits any mark."""
    engine = LooksAtAnArea((0.0, 0.0, 1.0, 1.0))
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _strip_sheet(),
                               name="strip.pdf")
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "stamped.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": "Checked - live smoke test",
             "target": "title block", "bbox": [40, 500, 120, 560]}]}))
    check = out["check"]
    assert check["confirmed"] == 1, check
    assert "inside the area it names" not in json.dumps(check["misplaced"])
    (p,) = engine.prompts
    assert "AREA of the sheet" in p and '"in_area"' in p
    assert "a stamp ('checked', 'reviewed', 'approved') fits any mark" in p


def test_a_mark_outside_the_area_it_names_is_still_misplaced(tmp_path,
                                                            monkeypatch):
    """Measured, not taken on trust: the look says the mark is in the area,
    but the area's box (the left fifth of the crop) does not hold the mark's
    centre, so it is not on it."""
    engine = LooksAtAnArea((0.0, 0.0, 0.2, 1.0))
    tools, handle = _tools_for(engine, tmp_path, monkeypatch, _strip_sheet(),
                               name="strip.pdf")
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "outside.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": "Checked",
             "target": "title block", "bbox": [300, 500, 380, 560]}]}))
    check = out["check"]
    assert check["confirmed"] == 0
    assert "outside the area it names" in check["misplaced"][0]["seen"]


def test_a_single_thing_in_blank_paper_is_still_misplaced(tmp_path,
                                                         monkeypatch):
    """The area rule is for areas: a ring meant for a tag, drawn in blank
    paper, is misplaced as before (the 2026-10-01 field report)."""
    engine = LooksAtAnArea((0.4, 0.4, 0.6, 0.6), in_area=False)
    tools, handle = _tools(engine, tmp_path, monkeypatch)
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": "blank.pdf", "markups": [
            {"kind": "circle", "page": 0, "comment": "penetration tag",
             "target": "GCE", "bbox": [430, 250, 500, 320]}]}))
    assert out["check"]["confirmed"] == 0
    assert len(out["check"]["misplaced"]) == 1


def _labels_sheet() -> bytes:
    """Lettering where a bad label would land, blank paper elsewhere, and a
    border line a good label may cross."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(4):
        page.insert_text((300, 200 + 9 * i), "BIORETENTION CROSS-SECTION",
                         fontsize=8)
    page.draw_line((100, 380), (100, 460), color=(0, 0, 0), width=1.2)
    data = doc.tobytes()
    doc.close()
    return data


def test_labels_are_measured_far_off_page_and_over_lettering():
    pdf = _labels_sheet()
    mark = [80.0, 420.0, 120.0, 440.0]
    result = {"written": [
        # beside its mark on blank paper, crossing one border line: fine
        {"page": 0, "kind": "box", "anchored_by": "bbox", "bbox": mark,
         "label": "Sheet 50.03", "label_bbox": [70.0, 401.0, 137.0, 418.0]},
        # the F16 failure: 120 pt from its mark
        {"page": 0, "kind": "box", "anchored_by": "bbox", "bbox": mark,
         "label": "Sheet 10.31A", "label_bbox": [240.0, 401.0, 313.0, 418.0]},
        # next to its mark but printed over the notes
        {"page": 0, "kind": "box", "anchored_by": "bbox",
         "bbox": [300.0, 230.0, 340.0, 250.0], "label": "Sheet 11.01",
         "label_bbox": [300.0, 196.0, 367.0, 213.0]},
        # off the page
        {"page": 0, "kind": "box", "anchored_by": "bbox",
         "bbox": [560.0, 20.0, 600.0, 40.0], "label": "Sheet 30.01",
         "label_bbox": [580.0, 2.0, 647.0, 19.0]},
        # no label: nothing to check
        {"page": 0, "kind": "box", "anchored_by": "bbox", "bbox": mark},
    ], "skipped": [{"index": 1}]}
    specs = [{}] * 6
    got = markup_check.check_labels(pdf, result, specs)
    by = {g["label"]: g for g in got}
    assert set(by) == {"Sheet 10.31A", "Sheet 11.01", "Sheet 30.01"}
    assert "pt from its mark" in by["Sheet 10.31A"]["problem"]
    assert "over the drawing's own lettering" in by["Sheet 11.01"]["problem"]
    assert "off the page" in by["Sheet 30.01"]["problem"]
    # indices are the caller's spec positions, past the skipped one
    assert by["Sheet 10.31A"]["index"] == 2
    assert by["Sheet 11.01"]["pdf_page"] == 1


class _RedCentre:
    """Records where the red mark sits in each crop it is shown."""

    def __init__(self):
        self.centres = []

    def analyze_image(self, image_bytes, prompt):
        img = PIL.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        xs, ys = [], []
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                r, g, b = img.getpixel((x, y))
                if r > 170 and g < 100 and b < 100:
                    xs.append(x)
                    ys.append(y)
        labelled = "red label beside it" in prompt
        self.centres.append((labelled, ((min(xs) + max(xs)) / 2.0 / w,
                                        (min(ys) + max(ys)) / 2.0 / h)
                             if xs else None))
        return json.dumps({"same_thing": True, "inside": "NOTE",
                           "encloses": True, "sure": True})


def _rotated_pdf(rot) -> bytes:
    """Shown 792 x 612 at /Rotate ``rot``, lettering reading across."""
    from planlens.document.frame import from_display_point
    doc = fitz.open()
    w, h = (792, 612) if rot in (0, 180) else (612, 792)
    page = doc.new_page(width=w, height=h)
    page.set_rotation(rot)
    page.insert_text(from_display_point(page, 405, 312), "NOTE", fontsize=8,
                     rotate=rot)
    page.insert_text(from_display_point(page, 300, 120), "GENERAL NOTES",
                     fontsize=8, rotate=rot)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.parametrize("rot", [90, 270])
def test_the_check_crops_the_mark_in_the_frame_planlens_wrote_it(
        tmp_path, monkeypatch, rot):
    """B1: one frame. The rows planlens returns, the crop the check renders
    and the label measurement are all in the displayed frame, so on a
    rotated sheet the crop is centred on the mark and the label is judged
    where the reader sees it."""
    engine = _RedCentre()
    tools, handle = _tools_for(engine, tmp_path, monkeypatch,
                               _rotated_pdf(rot), name=f"rot{rot}.pdf")
    out = json.loads(tools["annotate_document"].invoke({
        "handle": handle, "output_path": f"rot{rot}_marked.pdf", "markups": [
            {"kind": "box", "page": 0, "comment": "this note",
             "bbox": [400, 300, 440, 320]},
            {"kind": "box", "page": 0, "comment": "the general notes",
             "label": "N1", "bbox": [700, 540, 750, 575]}]}))
    assert out["n_written"] == 2
    (centre,) = [c for labelled, c in engine.centres if not labelled]
    assert centre is not None
    assert 0.4 < centre[0] < 0.6 and 0.4 < centre[1] < 0.6, centre
    row = out["written"][1]
    if not markup_check._label_reads_supported():
        return      # a planlens before the B1 fix places labels on 270 badly
    lb = row["label_bbox"]
    assert markup_check._gap(lb, row["bbox"]) <= 3.0
    assert "labels" not in out["check"], out["check"]


def test_the_model_is_never_handed_a_server_path(tmp_path, monkeypatch):
    """B9: the marked copy's absolute path stays inside the tool (the check
    opens it); the model gets the copy's name in the conversation, checked
    or not."""
    for check in (True, False):
        tools, handle = _tools(PixelEngine(), tmp_path, monkeypatch)
        out_raw = tools["annotate_document"].invoke({
            "handle": handle, "output_path": f"named_{check}.pdf",
            "check": check, "markups": [
                {"kind": "box", "page": 0, "comment": "tag",
                 "bbox": list(TAG)}]})
        out = json.loads(out_raw)
        assert out["output_path"] == f"named_{check}.pdf"
        assert str(tmp_path) not in out_raw
        assert str(tmp_path).replace("\\", "\\\\") not in out_raw
        assert ("check" in out) is check
        if check:
            assert out["check"]["confirmed"] == 1
        assert os.path.isfile(os.path.join(str(tmp_path),
                                           out["output_path"]))
