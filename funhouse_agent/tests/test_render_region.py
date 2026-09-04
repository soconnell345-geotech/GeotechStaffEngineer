"""Tests for the render_region vision primitive (B2) and the end-to-end
find-text-callout scenario (drawing_ir composition -> render_region -> vision).

The target string is always a runtime parameter — nothing in the pipeline is
specific to any label; "SAE" below is just one of the fixture's arbitrary
planted labels.

Uses a mock vision engine (records the rendered image + prompt, returns a
canned analysis) -- no API key required, same convention as
test_read_reference_figure.py.
"""

from __future__ import annotations

import json

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent.vision_tools import (
    EXTENDED_TOOLS, _dispatch_render_region, dispatch_extended_tool,
    render_region_to_file,
)
from drawing_ir import from_pdf_vector, queries
from drawing_ir.render import render_region
from drawing_ir.tests.leader_fixtures import PAGE_HEIGHT, build_synthetic_leader_pdf


class MockVisionEngine:
    def __init__(self, analysis="This is a leader pointing at a soil boring symbol."):
        self._analysis = analysis
        self.calls = []

    def analyze_image(self, image_bytes, prompt=""):
        self.calls.append({"image": image_bytes, "prompt": prompt})
        return self._analysis


def _ir_to_pdf(xy, page_height=PAGE_HEIGHT):
    """IR bottom_left point -> render_region's PDF (top-left) coordinate frame."""
    return (xy[0], page_height - xy[1])


class TestDispatchRenderRegionWired:
    """Phase 2/B6 wiring: render_region IS registered in the live catalog —
    in EXTENDED_TOOLS, routed by the public dispatch, and present on the
    deep-agent and native tool surfaces."""

    def test_in_extended_tools(self):
        assert "render_region" in EXTENDED_TOOLS

    def test_reachable_via_public_dispatch(self, tmp_path):
        path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                               include_decoys=False)
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        engine = MockVisionEngine()
        out = json.loads(dispatch_extended_tool(
            "render_region",
            {"attachment_key": "sheet", "page": 0,
             "bbox": [100, 100, 130, 130], "prompt": "what is here?"},
            engine, {"sheet": pdf_bytes}))
        assert "error" not in out
        assert out["analysis"] == engine._analysis

    def test_on_deep_agent_surface(self):
        from funhouse_agent.deep.tools import make_vision_tools
        names = {t.name for t in make_vision_tools(engine=None)}
        assert "render_region" in names

    def test_on_native_surface(self):
        from funhouse_agent.native_tools import (
            EXTENDED_TOOL_NAMES, OPENAI_TOOLS,
        )
        assert "render_region" in EXTENDED_TOOL_NAMES
        assert "render_region" in {t["function"]["name"]
                                   for t in OPENAI_TOOLS}

    def test_described_to_the_v1_agent(self):
        from funhouse_agent.vision_tools import VISION_TOOL_DESCRIPTIONS
        assert "render_region" in VISION_TOOL_DESCRIPTIONS

    def test_directly_callable(self, tmp_path):
        path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                               include_decoys=False)
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        engine = MockVisionEngine()
        out = json.loads(_dispatch_render_region(
            {"attachment_key": "sheet", "page": 0, "bbox": [100, 100, 130, 130],
             "prompt": "what is here?"},
            engine, {"sheet": pdf_bytes}))
        assert "error" not in out
        assert out["analysis"] == engine._analysis
        assert len(engine.calls) == 1
        img = engine.calls[0]["image"]
        assert isinstance(img, (bytes, bytearray)) and img[:8] == b"\x89PNG\r\n\x1a\n"

    def test_missing_pdf_source_errors(self):
        out = json.loads(_dispatch_render_region(
            {"attachment_key": "nope"}, MockVisionEngine(), {}))
        assert "error" in out

    def test_no_vision_engine_reports_gracefully(self, tmp_path):
        path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                               include_decoys=False)
        with open(path, "rb") as f:
            pdf_bytes = f.read()

        class NoVisionEngine:
            pass

        out = json.loads(_dispatch_render_region(
            {"attachment_key": "sheet", "bbox": [100, 100, 130, 130]},
            NoVisionEngine(), {"sheet": pdf_bytes}))
        assert "error" in out and "vision" in out["error"].lower()


class TestRenderRegionToFile:
    def test_saves_png_to_disk(self, tmp_path):
        pdf_path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                                    include_decoys=False)
        out_png = str(tmp_path / "crop.png")
        saved = render_region_to_file(out_png, filepath=pdf_path, page=0,
                                      bbox=[100, 100, 130, 130])
        import os
        assert os.path.isfile(saved)
        with open(saved, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"


class TestFindTextCalloutEndToEnd:
    """The design-memo acceptance scenario, for an ARBITRARY target string:
    find text X -> connected geometry -> the far endpoint it points at -> a
    rendered zoom of that endpoint, with marks -> handed to vision. Two
    routes into the same render_region call: the literal
    text_anchored_geometry composition, and the fuller find_leaders proposal.
    X = "SAE" here only because that is one of the fixture's planted labels."""

    TARGET = "SAE"  # arbitrary fixture label; the search string is runtime input

    def test_text_anchored_geometry_to_render_region(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=True)
        ir = from_pdf_vector(path)

        target_gt = next(L for L in gt["leaders"] if L.label == self.TARGET)

        hits = queries.text_anchored_geometry(ir, self.TARGET)
        assert len(hits) >= 1
        hit = hits[0]
        assert hit["points_at"] is not None
        tip_ir = tuple(hit["points_at"])
        # The far endpoint the geometry points at is the planted arrow tip.
        assert tip_ir == pytest.approx(target_gt.tip_xy, abs=0.6)

        # -> render_region needs PDF (top-left) coordinates, not IR's.
        tip_pdf = _ir_to_pdf(tip_ir)
        bbox = (tip_pdf[0] - 20, tip_pdf[1] - 20, tip_pdf[0] + 20, tip_pdf[1] + 20)
        png = render_region(filepath=path, page=0, bbox=bbox, dpi=300,
                            marks=[(tip_pdf[0], tip_pdf[1], "1")])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

        engine = MockVisionEngine("Mark 1 points at a soil sample callout.")
        analysis = engine.analyze_image(
            png, f"What does mark 1 (the callout at {hit['text']['content']}) "
                f"point at?")
        assert "points at" in analysis
        assert len(engine.calls) == 1
        assert engine.calls[0]["image"][:8] == b"\x89PNG\r\n\x1a\n"

    def test_find_leaders_to_render_region_via_dispatch(self, tmp_path):
        """The fuller pipeline through the dispatch-layer render_region (not
        just the pure function), proving the vision_tools glue hands a valid
        image + the right coordinates to the mock vision fn end to end."""
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=True)
        ir = from_pdf_vector(path)
        target_gt = next(L for L in gt["leaders"] if L.label == self.TARGET)

        proposals = [p for p in queries.find_leaders(ir)
                     if p["text"] == self.TARGET]
        assert proposals, "find_leaders did not recover the planted target leader"
        best = proposals[0]
        assert tuple(best["tip_xy"]) == pytest.approx(target_gt.tip_xy, abs=0.6)

        tip_pdf = _ir_to_pdf(best["tip_xy"])
        with open(path, "rb") as f:
            pdf_bytes = f.read()

        engine = MockVisionEngine("A leader pointing at the soil boring.")
        out = json.loads(_dispatch_render_region(
            {"attachment_key": "sheet", "page": 0,
             "bbox": [tip_pdf[0] - 20, tip_pdf[1] - 20,
                      tip_pdf[0] + 20, tip_pdf[1] + 20],
             "marks": [[tip_pdf[0], tip_pdf[1], self.TARGET]],
             "prompt": "What does the marked leader point at?"},
            engine, {"sheet": pdf_bytes}))
        assert "error" not in out
        assert out["analysis"] == engine._analysis
        assert engine.calls[0]["image"][:8] == b"\x89PNG\r\n\x1a\n"
