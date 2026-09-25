"""find_like in the app, and the tiled page read — the IZD "GCE" failure as a test.

On 2026-09-25 the agent read a 0.06 in stroke-lettered tag "GCE" as "QCE"
from a whole-sheet image and found none of 34 callouts. The fixture here is
that set in miniature (planlens.testing.tag_fixtures): lettering drawn as
lines, a legend on every sheet, GCE callouts with leaders among GCG / GPE /
QCE look-alikes. A fake vision engine answers the contact sheets from the
ground truth (it cannot see; it must only be ASKED the right thing), so the
test pins the tool's logic: every callout found, look-alikes rejected by the
reading, uncertain reads reported, the legend counted apart, the sheets saved.
"""

from __future__ import annotations

import json
import os

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("cv2")
tf = pytest.importorskip("planlens.testing.tag_fixtures")
pytest.importorskip("planlens.document.findlike")

from funhouse_agent import find_like as fl  # noqa: E402
from funhouse_agent import vision_probe, vision_view  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_find_like,
)


@pytest.fixture(scope="module")
def gt():
    return tf.build_synthetic_tag_set(n_pages=2)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for e in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
              vision_view.CHART_BUDGET_ENV, vision_view.POLICY_ENV):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    vision_probe.clear_cache()


def _candidates(gt, page, bbox, pages=None):
    """The candidates in the order the tool numbers them."""
    from planlens.document import Document
    d = Document(content=gt.pdf)
    try:
        res = d.find_like(page, bbox, pages)
        return [h for h in res["hits"] if h.context != "legend"]
    finally:
        d.close()


def _truth_of(gt, hit):
    cx, cy = hit.center
    for t in gt.tags:
        if t.page == hit.page:
            tx, ty = (t.bbox[0] + t.bbox[2]) / 2, (t.bbox[1] + t.bbox[3]) / 2
            if abs(cx - tx) < 6 and abs(cy - ty) < 6:
                return t
    return None


class TruthReader:
    """Answers every contact sheet from the ground truth — for ALL numbers;
    the tool keeps only the ones on the sheet it asked about."""
    accepts_jpeg = True

    def __init__(self, gt, cands, override=None):
        self.lines = []
        for i, h in enumerate(cands, start=1):
            t = _truth_of(gt, h)
            text = t.text if t else "-"
            kind = ("callout" if t and t.kind == "callout" else "other")
            if override and i in override:
                text = override[i]
            self.lines.append(f"#{i} | {text} | {kind}")
        self.calls = []

    def analyze_image(self, image_bytes, prompt=""):
        self.calls.append(prompt)
        return "\n".join(self.lines)


def test_every_gce_callout_is_found_and_look_alikes_are_rejected(gt, tmp_path):
    cands = _candidates(gt, 0, gt.example_bbox)
    engine = TruthReader(gt, cands)
    out = fl.find_like(gt.pdf, 0, gt.example_bbox, engine, text="GCE",
                       save_dir=str(tmp_path / "sheets"))
    on_plan = [t for t in gt.of("GCE") if t.kind != "legend"]
    callouts = gt.of("GCE", "callout")
    assert out["instances"] == len(on_plan)
    assert out["callouts"] == len(callouts)
    assert out["instances_by_page"] == {
        str(p): sum(1 for t in on_plan if t.page == p) for p in (0, 1)}
    assert out["legend_entries"] >= 2 and out["legend_pages"] == [0, 1]
    assert {"GCG", "GPE", "QCE"} <= set(out["rejected_as"])
    assert out["uncertain"] == []
    # Every callout carries where its leader points.
    got = [f for f in out["found"] if f["context"] == "callout"]
    assert got and all("points_to" in f for f in got)
    # The reviewer can check every call: the sheets are saved.
    assert out["contact_sheets"] and all(os.path.isfile(p)
                                         for p in out["contact_sheets"])
    assert "GCE" in engine.calls[0] and "[G/Q]CE" in engine.calls[0]


def test_a_bracketed_read_is_uncertain_not_counted(gt):
    cands = _candidates(gt, 0, gt.example_bbox)
    first_gce = next(i for i, h in enumerate(cands, start=1)
                     if (_truth_of(gt, h) or tf.TagTruth(0, "", (0,) * 4, 0, "")).text == "GCE")
    engine = TruthReader(gt, cands, override={first_gce: "[G/Q]CE"})
    out = fl.find_like(gt.pdf, 0, gt.example_bbox, engine, text="GCE")
    on_plan = [t for t in gt.of("GCE") if t.kind != "legend"]
    assert out["instances"] == len(on_plan) - 1
    assert [u["id"] for u in out["uncertain"]] == [first_gce]


def test_without_an_engine_the_candidates_are_marked_unverified(gt):
    out = fl.find_like(gt.pdf, 0, gt.example_bbox, None, text="GCE", pages="0")
    assert "UNVERIFIED" in out["note"]
    assert out["instances"] == 0 and out["unverified"]


def test_the_dispatcher_saves_to_the_working_folder(gt, tmp_path):
    cands = _candidates(gt, 0, gt.example_bbox)
    out = json.loads(_dispatch_find_like(
        {"attachment_key": "set", "page": 0, "bbox": list(gt.example_bbox),
         "text": "GCE"}, TruthReader(gt, cands), {"set": gt.pdf}))
    assert "error" not in out, out
    assert out["contact_sheets"][0].startswith(str(tmp_path / "work"))
    assert len(json.dumps(out)) <= fl.RESULT_BUDGET_CHARS


def test_the_dispatcher_explains_a_missing_example(gt):
    out = json.loads(_dispatch_find_like(
        {"attachment_key": "set", "page": 0}, TruthReader(gt, []),
        {"set": gt.pdf}))
    assert "ONE copy" in out["error"] and "render_region" in out["hint"]
    out = json.loads(_dispatch_find_like(
        {"attachment_key": "set", "page": 0, "bbox": [5, 5, 20, 12]},
        TruthReader(gt, []), {"set": gt.pdf}))
    assert "no ink" in out["error"]


def test_the_vision_tools_get_the_larger_cap_and_budget_under_it():
    from funhouse_agent.deep import tools as dt
    from funhouse_agent.vision_tools import TILED_RESULT_CHARS
    assert dt.DEFAULT_VISION_RESULT_CHARS > dt.DEFAULT_REFERENCE_RESULT_CHARS
    assert TILED_RESULT_CHARS < dt.DEFAULT_VISION_RESULT_CHARS
    assert fl.RESULT_BUDGET_CHARS < dt.DEFAULT_VISION_RESULT_CHARS


def test_parse_readings_and_verdicts():
    got = fl.parse_readings("noise\n#1 | GCE | callout\n#2|GCG|other\n"
                            "3: [G/Q]CE | callout\n`#4 | - | other`")
    assert got[1] == {"read": "GCE", "kind": "callout"}
    assert got[2]["read"] == "GCG" and got[3]["read"] == "[G/Q]CE"
    assert got[4]["read"] == "-"
    assert fl._verdict("GCE", "GCE") == "confirmed"
    assert fl._verdict("G C E", "gce") == "confirmed"
    assert fl._verdict("GCG", "GCE") == "rejected"
    assert fl._verdict("[G/Q]CE", "GCE") == "uncertain"
    assert fl._verdict("[C/O]CG", "GCE") == "rejected"
    assert fl._verdict("-", "GCE") == "no_lettering"


# -- the tiled page read ------------------------------------------------------------

class Eyes:
    accepts_jpeg = True

    def __init__(self):
        self.images = []

    def analyze_image(self, image_bytes, prompt=""):
        self.images.append(prompt)
        return "saw it"


def test_small_lettering_is_read_in_tiles(gt, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "gpt-4.1-high")   # 768 px
    eyes = Eyes()
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0, "prompt": "find GCE"},
        eyes, {"set": gt.pdf}))
    assert "error" not in out, out
    assert len(out["tiles"]) == 16 and len(eyes.images) == 17
    assert out["tiles"][0]["tile"] == "r1c1" and "view" in out["tiles"][0]
    assert "tile row 1 of 4" in "".join(eyes.images)
    assert "4x4" in out["tiling"]
    from funhouse_agent.vision_tools import TILED_RESULT_CHARS
    assert len(json.dumps(out)) <= TILED_RESULT_CHARS


def test_no_tiles_when_the_page_already_reads_or_when_asked(gt, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    eyes = Eyes()
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0}, eyes, {"set": gt.pdf}))
    assert "tiles" not in out and len(eyes.images) == 1
    monkeypatch.setenv(vision_view.BUDGET_ENV, "gpt-4.1-high")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0, "tiles": "off"}, Eyes(),
        {"set": gt.pdf}))
    assert "tiles" not in out
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0, "tiles": "2"}, Eyes(),
        {"set": gt.pdf}))
    assert len(out["tiles"]) == 4
    monkeypatch.setenv(vision_view.POLICY_ENV, "efficient")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0}, Eyes(), {"set": gt.pdf}))
    assert "tiles" not in out


def test_a_stroke_lettered_sheet_says_so(gt, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "set", "page": 0}, Eyes(), {"set": gt.pdf}))
    assert "not in its text layer" in out["legibility"]
    assert "find_like" in out["legibility"]
    assert out["budget"] == "openai-original"


def test_robust_policy_uses_the_honoured_original(monkeypatch):
    from funhouse_agent.vision_probe import VisionProfile

    class E:
        def vision_profile(self):
            return VisionProfile(answered_by="gpt-5.4-2026-03-05",
                                 general="gpt-4.1-high",
                                 detailed="openai-original", source="probe")

    assert vision_view.budget_name(E()) == "openai-original"
    monkeypatch.setenv(vision_view.POLICY_ENV, "efficient")
    assert vision_view.budget_name(E()) == "gpt-4.1-high"
    with vision_view.chart_reading():
        assert vision_view.budget_name(E()) == "openai-original"
