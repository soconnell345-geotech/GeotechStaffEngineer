"""Tests for the read_reference_figure vision tool.

Uses a mock vision engine (records the rendered image + prompt and returns a
canned analysis) — no API key required. The render step needs the source PDF in
geotech-references/docs/ and PyMuPDF; those tests skip gracefully if absent.
"""

from __future__ import annotations

import json

import pytest

from funhouse_agent.vision_tools import dispatch_extended_tool, EXTENDED_TOOLS


class MockVisionEngine:
    def __init__(self, analysis="Following the theta=10 curve to phi'=35, Kp ~ 6.0."):
        self._analysis = analysis
        self.vision_calls = []

    def analyze_image(self, image_input, user_prompt=""):
        self.vision_calls.append({"image": image_input, "prompt": user_prompt})
        return self._analysis


class NoVisionEngine:
    def chat(self, user, system="", temperature=0):
        return "no vision here"


def _call(args, engine):
    return json.loads(dispatch_extended_tool(
        "read_reference_figure", args, engine, attachments={}
    ))


def _pdf_available() -> bool:
    try:
        import pymupdf  # noqa: F401
        from geotech_references import _figures_db
        _figures_db.resolve_pdf("dm7_2", "4-12")
        return True
    except Exception:
        return False


pdf_required = pytest.mark.skipif(
    not _pdf_available(),
    reason="DM7 source PDF / PyMuPDF not available for rendering",
)


def test_tool_is_registered():
    assert "read_reference_figure" in EXTENDED_TOOLS


def test_missing_args_error():
    out = _call({"reference": "dm7_2"}, MockVisionEngine())
    assert "error" in out
    out = _call({"figure_number": "4-12"}, MockVisionEngine())
    assert "error" in out


def test_unknown_figure_error():
    out = _call({"reference": "dm7_2", "figure_number": "99-99"}, MockVisionEngine())
    assert "error" in out
    assert "not found" in out["error"].lower()


@pdf_required
def test_reads_off_figure_4_12():
    engine = MockVisionEngine()
    out = _call(
        {"reference": "dm7_2", "figure_number": "4-12",
         "prompt": "Read Kp for phi'=35 deg, theta=10 deg, delta/phi=0.66"},
        engine,
    )
    assert "error" not in out
    assert out["figure_number"] == "4-12"
    assert out["pdf_page_index"] == 229
    assert "log spiral" in out["caption"].lower()
    assert "Kp" in out["analysis"]
    assert "estimate" in out["note"].lower()
    # the engine received a rendered PNG and a prompt carrying caption + request
    assert len(engine.vision_calls) == 1
    img = engine.vision_calls[0]["image"]
    assert isinstance(img, (bytes, bytearray)) and img[:8] == b"\x89PNG\r\n\x1a\n"
    prompt = engine.vision_calls[0]["prompt"]
    assert "4-12" in prompt
    assert "Log Spiral" in prompt
    assert "phi'=35" in prompt


def _library_takes_page() -> bool:
    from geotech_references import _figures_db
    from funhouse_agent.adapters._reference_common import takes
    return takes(_figures_db.figure_get, "pdf_page_index")


def test_the_page_asked_for_reaches_the_library(monkeypatch):
    """A figure number the catalog holds on several pages is read off the
    page the search hit named (2026-11-10 review: the library refuses
    such a number without its pdf_page_index)."""
    if not _library_takes_page():
        pytest.skip("library figure_get takes no pdf_page_index")
    from geotech_references import _figures_db
    asked = {}

    def figure_get(reference, figure_number, max_chars=None,
                   pdf_page_index=None):
        asked.update(reference=reference, figure_number=figure_number,
                     pdf_page_index=pdf_page_index)
        raise KeyError("made-up stop")

    monkeypatch.setattr(_figures_db, "figure_get", figure_get)
    out = _call({"reference": "made_up", "figure_number": "D-9",
                 "pdf_page_index": 36, "prompt": "x?"}, MockVisionEngine())
    assert "error" in out
    assert asked == {"reference": "made_up", "figure_number": "D-9",
                     "pdf_page_index": 36}
    asked.clear()
    _call({"reference": "made_up", "figure_number": "D-9", "prompt": "x?"},
          MockVisionEngine())
    assert asked["pdf_page_index"] is None


def test_a_figure_on_several_pages_without_a_page_says_which():
    if not _library_takes_page():
        pytest.skip("library figure_get takes no pdf_page_index")
    from geotech_references import _figures_db
    rows = [r for r in _figures_db.list_indexed_figures()
            if r.get("reference") == "gec_12"]
    if not rows:
        pytest.skip("no gec_12 figure catalog")
    out = _call({"reference": "gec_12", "figure_number": "D-1",
                 "prompt": "x?"}, MockVisionEngine())
    assert "error" in out
    assert "pdf_page_index" in out["error"]


@pdf_required
def test_no_vision_engine_reports_gracefully():
    out = _call(
        {"reference": "dm7_2", "figure_number": "4-12", "prompt": "Kp?"},
        NoVisionEngine(),
    )
    assert "error" in out
    assert "vision" in out["error"].lower()
