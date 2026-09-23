"""How a page reaches the vision model: budget, detail, and the 0-999 grid.

Pinned: renders land inside the configured image budget (and fall back to the
old fixed sizes on a planlens without budgets); the engine labels the bytes
by their real type, sends the budget's ``detail`` and retries once without it
when a model rejects it; every vision prompt asks for 0-999 locations and
every vision result returns the ``view`` that turns one back into page points;
``render_region(view=, image_box=)`` zooms on exactly that box.
"""

from __future__ import annotations

import json

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import vision_view  # noqa: E402
from funhouse_agent.deep.vision_engine import LangChainVisionEngine  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_render_region,
)

JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"jpeg-body"
PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


@pytest.fixture
def pdf_bytes():
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)       # tabloid, landscape
    page.insert_text((100, 100), "BORING B-1", fontsize=10)
    page.draw_rect(fitz.Rect(600, 400, 700, 450))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def default_env(monkeypatch):
    monkeypatch.delenv(vision_view.BUDGET_ENV, raising=False)
    monkeypatch.delenv(vision_view.DETAIL_ENV, raising=False)


class Engine:
    accepts_jpeg = True

    def __init__(self):
        self.calls = []

    def analyze_image(self, image_bytes, prompt=""):
        self.calls.append({"image": image_bytes, "prompt": prompt})
        return "the rectangle is at [490, 505, 572, 568]"


# -- settings -------------------------------------------------------------------

def test_default_budget_and_detail():
    assert vision_view.budget().name == vision_view.DEFAULT_BUDGET == "openai-high"
    assert vision_view.detail() == "high"


def test_budget_and_detail_from_the_environment(monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    assert vision_view.budget().name == "openai-original"
    assert vision_view.detail() == "original"
    monkeypatch.setenv(vision_view.DETAIL_ENV, "none")
    assert vision_view.detail() is None
    monkeypatch.setenv(vision_view.BUDGET_ENV, "none")
    monkeypatch.delenv(vision_view.DETAIL_ENV)
    assert vision_view.budget() is None and vision_view.detail() is None
    monkeypatch.setenv(vision_view.BUDGET_ENV, "gpt-9")      # unknown: off
    assert vision_view.budget() is None


def test_media_type_is_read_from_the_bytes():
    assert vision_view.image_media_type(JPEG) == "image/jpeg"
    assert vision_view.image_media_type(PNG) == "image/png"


def test_image_box_to_page():
    view = [100.0, 200.0, 1099.0, 699.0]                     # 999 x 499 pt
    assert vision_view.image_box_to_page(view, [0, 0, 999, 999]) == tuple(view)
    assert vision_view.image_box_to_page(view, [100, 200, 300, 400]) == (
        pytest.approx(200.0), pytest.approx(299.9, abs=0.01),
        pytest.approx(400.0), pytest.approx(399.8, abs=0.01))
    with pytest.raises(ValueError):
        vision_view.image_box_to_page([0, 0, 0, 10], [0, 0, 9, 9])
    with pytest.raises(ValueError):
        vision_view.image_box_to_page(view, [0, 0, 9])


# -- rendering ------------------------------------------------------------------

def test_a_page_renders_inside_the_budget(pdf_bytes):
    data, info = vision_view.render_view(pdf_bytes, page=0)
    bud = vision_view.budget()
    assert bud.fits(info["width_px"], info["height_px"])
    assert max(info["width_px"], info["height_px"]) > 1900  # filled, not shrunk
    assert info["clip"] == [0.0, 0.0, 1224.0, 792.0]


def test_a_region_is_redrawn_to_fill_the_budget(pdf_bytes):
    data, info = vision_view.render_view(pdf_bytes, page=0,
                                         bbox=[600, 400, 700, 450])
    assert info["dpi"] > 300
    assert vision_view.budget().fits(info["width_px"], info["height_px"])


def test_without_a_budget_the_old_sizes_hold(pdf_bytes, monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "none")
    data, info = vision_view.render_view(pdf_bytes, page=0,
                                         bbox=[600, 400, 700, 450])
    assert info["dpi"] == 300.0 and data[:4] == b"\x89PNG"


def test_an_older_planlens_falls_back(pdf_bytes, monkeypatch):
    monkeypatch.setattr(vision_view, "_render_accepts_budget", lambda: False)
    data, info = vision_view.render_view(pdf_bytes, page=0,
                                         bbox=[600, 400, 700, 450])
    assert info["dpi"] == 300.0 and "budget" not in info


def test_png_unless_the_engine_labels_jpeg(pdf_bytes):
    # Line art is PNG either way; the flag only lets a scan go as JPEG.
    data, info = vision_view.render_view(pdf_bytes, page=0, allow_jpeg=False)
    assert info["format"] == "png"


# -- the engine -------------------------------------------------------------------

class FakeModel:
    def __init__(self, reject_detail=False, error=None):
        self.messages = []
        self.reject_detail = reject_detail
        self.error = error

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.messages.append(messages[0])
        block = messages[0].content[0]["image_url"]
        if self.error:
            raise self.error
        if self.reject_detail and "detail" in block:
            raise ValueError("Invalid value for 'detail': original")
        return AIMessage(content="ok")


def _image_url(message):
    return message.content[0]["image_url"]


def test_engine_labels_jpeg_and_sends_detail(monkeypatch):
    model = FakeModel()
    eng = LangChainVisionEngine(model)
    assert eng.analyze_image(JPEG, "look") == "ok"
    url = _image_url(model.messages[0])
    assert url["url"].startswith("data:image/jpeg;base64,")
    assert url["detail"] == "high"
    eng.analyze_image(PNG, "look")
    assert _image_url(model.messages[1])["url"].startswith("data:image/png;base64,")


def test_engine_retries_once_without_a_rejected_detail(monkeypatch):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-original")
    model = FakeModel(reject_detail=True)
    assert LangChainVisionEngine(model).analyze_image(PNG, "look") == "ok"
    assert [("detail" in _image_url(m)) for m in model.messages] == [True, False]
    assert _image_url(model.messages[0])["detail"] == "original"


def test_engine_does_not_swallow_other_errors():
    model = FakeModel(error=RuntimeError("rate limited"))
    with pytest.raises(RuntimeError, match="rate limited"):
        LangChainVisionEngine(model).analyze_image(PNG, "look")
    assert len(model.messages) == 1


def test_engine_without_detail(monkeypatch):
    monkeypatch.setenv(vision_view.DETAIL_ENV, "none")
    model = FakeModel()
    LangChainVisionEngine(model).analyze_image(PNG, "look")
    assert "detail" not in _image_url(model.messages[0])


# -- the tools: grid in, view out, zoom back in ------------------------------------

def test_analyze_pdf_page_asks_for_the_grid_and_returns_the_view(pdf_bytes):
    engine = Engine()
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "sheet", "page": 0, "prompt": "find the rectangle"},
        engine, {"sheet": pdf_bytes}))
    assert vision_view.GRID_INSTRUCTION in engine.calls[0]["prompt"]
    assert out["view"] == [0.0, 0.0, 1224.0, 792.0]
    assert len(out["view_px"]) == 2 and "render_region" in out["zoom_hint"]


def test_render_region_zooms_on_a_grid_box_from_a_view(pdf_bytes):
    engine = Engine()
    view = [0.0, 0.0, 1224.0, 792.0]
    # The rectangle (600,400)-(700,450) on the 0-999 grid of that view.
    box = [600 / 1224 * 999, 400 / 792 * 999, 700 / 1224 * 999, 450 / 792 * 999]
    out = json.loads(_dispatch_render_region(
        {"attachment_key": "sheet", "page": 0, "view": view, "image_box": box,
         "prompt": "what is this?"}, engine, {"sheet": pdf_bytes}))
    assert "error" not in out
    assert out["bbox"] == pytest.approx([600, 400, 700, 450], abs=0.01)
    assert vision_view.GRID_INSTRUCTION in engine.calls[0]["prompt"]
    # The zoom's own view is the padded crop, ready for the next zoom.
    x0, y0, x1, y1 = out["view"]
    assert x0 < 600 and y0 < 400 and x1 > 700 and y1 > 450


def test_render_region_view_mistakes(pdf_bytes):
    engine = Engine()
    both = json.loads(_dispatch_render_region(
        {"attachment_key": "sheet", "bbox": [0, 0, 9, 9],
         "view": [0, 0, 100, 100], "image_box": [0, 0, 9, 9]},
        engine, {"sheet": pdf_bytes}))
    assert "not both" in both["error"]
    bad = json.loads(_dispatch_render_region(
        {"attachment_key": "sheet", "view": [0, 0, 100, 100]},
        engine, {"sheet": pdf_bytes}))
    assert "view + image_box" in bad["error"]
    assert engine.calls == []


# -- chart read-offs get the larger budget -----------------------------------------

def test_chart_reading_raises_budget_and_detail_for_the_block():
    assert vision_view.budget().name == "openai-high"
    with vision_view.chart_reading():
        assert vision_view.budget().name == vision_view.DEFAULT_CHART_BUDGET
        assert vision_view.detail() == "original"
    assert vision_view.budget().name == "openai-high"


def test_chart_budget_from_the_environment(monkeypatch):
    monkeypatch.setenv(vision_view.CHART_BUDGET_ENV, "openai-high")
    with vision_view.chart_reading():
        assert vision_view.detail() == "high"


def test_read_reference_figure_runs_at_the_chart_budget(monkeypatch, tmp_path,
                                                        pdf_bytes):
    from funhouse_agent import vision_tools
    pdf = tmp_path / "ref.pdf"
    pdf.write_bytes(pdf_bytes)
    from geotech_references import _figures_db
    monkeypatch.setattr(_figures_db, "figure_get", lambda r, f: {
        "figure_number": f, "caption": "a chart", "page_estimated": False})
    monkeypatch.setattr(_figures_db, "resolve_pdf", lambda r, f: (pdf, 0))
    seen = {}

    class ChartEngine(Engine):
        def analyze_image(self, image_bytes, prompt=""):
            seen["detail"] = vision_view.detail()
            return super().analyze_image(image_bytes, prompt)

    out = json.loads(vision_tools._dispatch_read_reference_figure(
        {"reference": "dm7_1", "figure_number": "5-6", "prompt": "read mu0"},
        ChartEngine()))
    assert "error" not in out, out
    assert seen["detail"] == "original"
    w, h = out["view_px"]
    assert max(w, h) > 2100              # past the 2048 px of "high"
    assert out["source"] == str(pdf) and out["page"] == 0
    assert vision_view.detail() == "high"  # restored after the call
