"""Tests for pdf_import vision grid overlay — C4 (v5.3)."""

import json

import pytest

fitz = pytest.importorskip("fitz")   # PyMuPDF required for rendering

from pdf_import.vision import (
    render_page_with_grid, extract_geometry_vision,
    GRID_VISION_PROMPT, VISION_PROMPT,
)


def _simple_pdf_bytes(w=300, h=200):
    doc = fitz.open()
    pg = doc.new_page(width=w, height=h)
    pg.draw_line((10, 150), (290, 120), color=(0, 0, 0), width=1)
    b = doc.tobytes()
    doc.close()
    return b


class TestRenderGrid:
    def test_returns_png(self):
        png = render_page_with_grid(content=_simple_pdf_bytes(), grid_spacing=50, dpi=150)
        assert png[:4] == b"\x89PNG"
        assert len(png) > 500

    def test_bad_spacing(self):
        with pytest.raises(ValueError):
            render_page_with_grid(content=_simple_pdf_bytes(), grid_spacing=0)

    def test_no_source(self):
        with pytest.raises(ValueError):
            render_page_with_grid(grid_spacing=50)

    def test_page_out_of_range(self):
        with pytest.raises(ValueError):
            render_page_with_grid(content=_simple_pdf_bytes(), page=5, grid_spacing=50)


class TestExtractWithGrid:
    def _mock(self, captured):
        def fn(img, prompt):
            captured["prompt"] = prompt
            captured["img_len"] = len(img)
            return json.dumps({"surface_points": [[0, 50], [300, 30]],
                               "boundary_profiles": {}, "gwt_points": None})
        return fn

    def test_grid_overlay_uses_grid_prompt_and_warns(self):
        cap = {}
        res = extract_geometry_vision(self._mock(cap), content=_simple_pdf_bytes(),
                                      grid_overlay=True, grid_spacing=50, scale=0.1)
        assert cap["prompt"] == GRID_VISION_PROMPT
        assert "COORDINATE GRID" in cap["prompt"]
        assert res.surface_points == [(0.0, 5.0), (30.0, 3.0)]     # scaled by 0.1
        assert any("grid overlay" in w for w in res.warnings)

    def test_default_path_unchanged(self):
        cap = {}
        res = extract_geometry_vision(self._mock(cap), content=_simple_pdf_bytes(),
                                      scale=1.0)
        assert cap["prompt"] == VISION_PROMPT
        assert not any("grid overlay" in w for w in res.warnings)

    def test_custom_prompt_overrides_grid_prompt(self):
        cap = {}
        extract_geometry_vision(self._mock(cap), content=_simple_pdf_bytes(),
                                grid_overlay=True, custom_prompt="MY PROMPT")
        assert cap["prompt"] == "MY PROMPT"
