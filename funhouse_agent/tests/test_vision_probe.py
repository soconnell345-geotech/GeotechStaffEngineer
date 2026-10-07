"""The vision model is measured, not assumed.

A fake chat model charges image tokens the way each real family does — a TILE
model (GPT-4o/4.1/5.1: everything cut to 768 px on the short side, 70 + 140
tokens a 512 px tile, ``original`` accepted and ignored, exactly what
``tinyapp-gpt-medium`` did on 2026-09-24) and a PATCH model (GPT-5.4: 32 px
patches, 2,500 at ``high``, 10,000 at ``original``, with a multiplier) — and
the probe must read the right budget off the numbers, feed it to every render
and every ``detail`` field, fall back to the model's name when the numbers are
missing, never break a vision call, run once per model, and give way to the
owner's env settings.
"""

from __future__ import annotations

import base64
import json
import math
import struct

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import vision_probe, vision_view  # noqa: E402
from funhouse_agent.deep.vision_engine import LangChainVisionEngine  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    _dispatch_analyze_pdf_page, _dispatch_read_reference_figure,
)


def _png_size(data_url: str):
    raw = base64.b64decode(data_url.split(",", 1)[1][:64])
    return struct.unpack(">II", raw[16:24])


class FakeVisionModel:
    """Answers like a deployment of one model family, with token counts."""

    def __init__(self, family: str, answered: str = "gpt-5.1-2025-11-13",
                 usage: bool = True, fail: bool = False):
        self.family, self.answered = family, answered
        self.usage, self.fail = usage, fail
        self.model_name = "tinyapp-gpt-medium"       # an alias, as configured
        self.calls = []

    def _image_tokens(self, w, h, detail):
        if self.family == "funhouse":
            # Funhouse GPT-5.4 (measured 2026-09-25 and 2026-10-07): "high"
            # cut down like a tile model (714 for both squares); "original"
            # honoured — but the HOST shrinks anything over 2,048 px first.
            if detail != "original":
                return 714
            s = min(1.0, 2048 / max(w, h))
            w, h = round(w * s), round(h * s)
            return int(math.ceil(w / 32) * math.ceil(h / 32) * 1.03) + 10
        if self.family == "tile":
            s = min(1.0, 2048 / max(w, h))
            w, h = w * s, h * s
            s = min(1.0, 768 / min(w, h))
            w, h = w * s, h * s
            return 70 + 140 * math.ceil(w / 512) * math.ceil(h / 512)
        cap = 10000 if detail == "original" else 2500
        patches = math.ceil(w / 32) * math.ceil(h / 32)
        if patches > cap:
            s = math.sqrt(cap / patches)
            patches = math.ceil(w * s / 32) * math.ceil(h * s / 32)
            patches = min(patches, cap)
        return int(patches * 1.2)                  # a multiplier ratios cancel

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        content = messages[0].content
        self.calls.append(content)
        if self.fail:
            raise RuntimeError("gateway down")
        tokens = 12
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    w, h = _png_size(block["image_url"]["url"])
                    tokens += self._image_tokens(
                        w, h, block["image_url"].get("detail", "auto"))
        msg = AIMessage(content="OK", response_metadata={
            "model_name": self.answered})
        if self.usage:
            msg.usage_metadata = {"input_tokens": tokens, "output_tokens": 1,
                                  "total_tokens": tokens + 1}
        return msg


@pytest.fixture(autouse=True)
def probing_on(monkeypatch):
    for env in (vision_view.BUDGET_ENV, vision_view.DETAIL_ENV,
                vision_view.CHART_BUDGET_ENV, vision_view.MAX_PX_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv(vision_probe.PROBE_ENV, "1")
    vision_probe.clear_cache()
    yield
    vision_probe.clear_cache()


@pytest.fixture
def sheet_pdf():
    """A half-size sheet whose notes are 5 pt tall, like BD-AB_01-26."""
    doc = fitz.open()
    page = doc.new_page(width=1224, height=792)
    for i in range(30):
        page.insert_text((60 + 280 * (i % 4), 80 + 22 * (i // 4)),
                         "#5 BARS @ 12 IN. MAX. SPACING", fontsize=5)
    data = doc.tobytes()
    doc.close()
    return data


def test_a_tile_model_is_recognised_from_its_token_counts():
    prof = vision_probe.probe(FakeVisionModel("tile"))
    assert prof.source == "probe"
    assert prof.answered_by == "gpt-5.1-2025-11-13"
    assert (prof.general, prof.detailed) == ("gpt-4.1-high", "gpt-4.1-high")
    assert prof.ratios["high"] == 1.0 and prof.ratios["original"] == 1.0


def test_a_patch_model_with_original_is_recognised():
    prof = vision_probe.probe(FakeVisionModel("patch", "gpt-5.4-2026-03-05"))
    assert (prof.general, prof.detailed) == ("openai-high", "openai-original")


def test_a_patch_model_is_recognised_whatever_its_name():
    # Measured, not looked up: a name nobody has heard of still works.
    prof = vision_probe.probe(FakeVisionModel("patch", "gpt-9-omni"))
    assert (prof.general, prof.detailed) == ("openai-high", "openai-original")


def test_no_token_counts_falls_back_to_the_model_name():
    prof = vision_probe.probe(FakeVisionModel("tile", usage=False))
    assert prof.source == "model name"
    assert prof.general == "gpt-4.1-high"


def test_a_failing_probe_never_raises_and_leaves_the_defaults():
    model = FakeVisionModel("tile", fail=True)
    prof = vision_probe.probe(model)
    assert prof.general is None and "gateway down" in prof.error
    engine = LangChainVisionEngine(model)
    assert vision_view.budget_name(engine) == vision_view.DEFAULT_BUDGET


def test_one_probe_per_model_per_process():
    model = FakeVisionModel("tile")
    a = LangChainVisionEngine(model)
    b = LangChainVisionEngine(model)
    a.vision_profile()
    n = len(model.calls)
    assert n == 4
    b.vision_profile()
    a.vision_profile()
    assert len(model.calls) == n


def test_the_probe_can_be_switched_off(monkeypatch):
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    model = FakeVisionModel("tile")
    assert LangChainVisionEngine(model).vision_profile() is None
    assert model.calls == []


def test_the_measured_budget_sizes_renders_and_detail(sheet_pdf):
    model = FakeVisionModel("tile")
    engine = LangChainVisionEngine(model)
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "sheet", "page": 0, "prompt": "list the bars",
         "tiles": "off"},
        engine, {"sheet": sheet_pdf}))
    assert "error" not in out, out
    # Rendered at exactly what a tile model looks at: 768 on the short side.
    assert out["view_px"] == [1187, 768]
    assert out["vision_model"] == "gpt-5.1-2025-11-13"
    sent = model.calls[-1][0]["image_url"]
    assert sent["detail"] == "high"
    # 5 pt notes at ~1 px/pt: the result says so and what to do instead.
    assert out["text_px"] < 6
    assert "read_document / search_document" in out["legibility"]
    assert "Do not report the page as unreadable" in out["legibility"]
    # And on the default (robust, tiles "auto") the page is ALSO read in
    # tiles large enough to carry that lettering.
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "sheet", "page": 0, "prompt": "list the bars"},
        engine, {"sheet": sheet_pdf}))
    assert len(out["tiles"]) == 16
    assert all(t.get("text_px", 0) >= 14 for t in out["tiles"])


def test_charts_follow_the_measurement_too(monkeypatch, tmp_path, sheet_pdf):
    """On a model that ignores detail="original" (GPT-5.1), a chart is not
    rendered at the 10,000-patch size the server would only shrink again."""
    pdf = tmp_path / "ref.pdf"
    pdf.write_bytes(sheet_pdf)
    from geotech_references import _figures_db
    monkeypatch.setattr(_figures_db, "figure_get", lambda r, f: {
        "figure_number": f, "caption": "a chart", "page_estimated": False})
    monkeypatch.setattr(_figures_db, "resolve_pdf", lambda r, f: (pdf, 0))
    import dataclasses
    from planlens.document.budget import BUDGETS, fit_size
    # A patch model honouring "original" gets that budget, held to the
    # 2,048 px a host delivers (vision_view.DEFAULT_MAX_PX).
    original_held = dataclasses.replace(BUDGETS["openai-original"],
                                        max_edge=vision_view.DEFAULT_MAX_PX)
    for family, answered, px in (
            ("tile", "gpt-5.1-2025-11-13",
             fit_size(1224, 792, BUDGETS["gpt-4.1-high"])[0]),
            ("patch", "gpt-5.4-2026-03-05",
             fit_size(1224, 792, original_held)[0])):
        vision_probe.clear_cache()
        model = FakeVisionModel(family, answered)
        out = json.loads(_dispatch_read_reference_figure(
            {"reference": "dm7_1", "figure_number": "5-6", "prompt": "read"},
            LangChainVisionEngine(model)))
        assert "error" not in out, out
        assert abs(out["view_px"][0] - px) <= 3, (family, out["view_px"])


def test_an_env_setting_still_wins(monkeypatch, sheet_pdf):
    monkeypatch.setenv(vision_view.BUDGET_ENV, "openai-high")
    model = FakeVisionModel("tile")
    engine = LangChainVisionEngine(model)
    assert vision_view.budget_name(engine) == "openai-high"
    with vision_view.chart_reading():
        # No chart override set: the measurement decides the chart budget.
        assert vision_view.budget_name(engine) == "gpt-4.1-high"
    monkeypatch.setenv(vision_view.CHART_BUDGET_ENV, "openai-original")
    with vision_view.chart_reading():
        assert vision_view.budget_name(engine) == "openai-original"
    monkeypatch.setenv(vision_view.BUDGET_ENV, "none")
    monkeypatch.delenv(vision_view.CHART_BUDGET_ENV)
    with vision_view.chart_reading():
        assert vision_view.budget_name(engine) == "none"


def test_summary_reads_plainly():
    prof = vision_probe.probe(FakeVisionModel("patch", "gpt-5.4-2026-03-05"))
    assert prof.summary() == ("gpt-5.4-2026-03-05: images at openai-high, "
                              "charts at openai-original (from probe)")
    assert json.dumps(prof.to_dict())


def test_connection_diagnostics_report_the_measured_model(monkeypatch):
    from webapp.diagnostics import _vision_check
    check = _vision_check(FakeVisionModel("tile"))
    assert check["status"] == "pass"
    assert check["detail"].startswith("gpt-5.1-2025-11-13: images at gpt-4.1-high")
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    assert _vision_check(FakeVisionModel("tile"))["status"] == "skip"


# -- the host's own cap (2026-10-07) ----------------------------------------------

def test_a_host_that_shrinks_large_images_is_measured():
    """Funhouse: "original" honoured up to 2,048 px, anything larger shrunk
    by the host. The four calls up to 2,048 px cannot see that; the fifth,
    a 3072 x 1024 image, can."""
    model = FakeVisionModel("funhouse", "gpt-5.4-2026-03-05")
    prof = vision_probe.probe(model)
    assert (prof.general, prof.detailed) == ("gpt-4.1-high", "openai-original")
    assert len(model.calls) == 5
    assert _png_size(model.calls[-1][0]["image_url"]["url"]) == \
        vision_probe.WIDE_PX
    assert prof.max_edge == 2048
    assert prof.ratios["wide"] < 0.4
    assert "the host delivers at most 2048 px" in prof.summary()
    assert prof.to_dict()["max_edge"] == 2048


def test_a_host_that_passes_large_images_whole_sets_no_cap():
    model = FakeVisionModel("patch", "gpt-5.6-2026-09-01")
    prof = vision_probe.probe(model)
    assert len(model.calls) == 5 and prof.max_edge is None
    assert "host delivers" not in prof.summary()


def test_no_fifth_call_where_nothing_past_2048_is_sent():
    model = FakeVisionModel("tile")
    prof = vision_probe.probe(model)
    assert len(model.calls) == 4 and prof.max_edge is None


def test_the_measured_host_edge_caps_renders_even_when_the_owner_lifts_the_cap(
        monkeypatch, sheet_pdf):
    engine = LangChainVisionEngine(FakeVisionModel("funhouse",
                                                   "gpt-5.4-2026-03-05"))
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "none")
    assert vision_view.max_px(engine) == 2048
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "4096")
    assert vision_view.max_px(engine) == 2048
    monkeypatch.setenv(vision_view.MAX_PX_ENV, "1600")
    assert vision_view.max_px(engine) == 1600         # a smaller cap stands
    monkeypatch.delenv(vision_view.MAX_PX_ENV)
    out = json.loads(_dispatch_analyze_pdf_page(
        {"attachment_key": "sheet", "page": 0, "prompt": "list the bars",
         "tiles": "off"}, engine, {"sheet": sheet_pdf}))
    assert max(out["view_px"]) == 2048 and out["detail"] == "original"


@pytest.mark.parametrize("square,wide,edge", [
    (4234, 1460, 2048),           # Funhouse: shrunk to 2048 x 683
    (4915, 3686, None),           # whole (3,072 patches), any multiplier
    (4096, 2160, 2544),           # a host that keeps ~2,560 px
    (None, 1000, None), (4096, None, None),
])
def test_host_edge_from_probe(square, wide, edge):
    assert vision_probe.host_edge_from_probe(square, wide) == edge
