"""Review of BACKLOG EV-20 (the ``image_read`` event in the run report): an
ordinary look is not shown as a warning. A made-up image and reader; no
model is called.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.tools import tool

from aec_agent_core import run_report, vision_tools
from aec_agent_core.activity_log import ActivityLogger

pymupdf = pytest.importorskip("pymupdf")


class _Reader:
    def analyze_image(self, image, prompt=""):
        return "Two made-up lines on white."


def _png() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 48), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    return pix.tobytes("png")


@tool
def analyze_image(attachment_key: str) -> str:
    """Look at a made-up image."""
    return vision_tools.dispatch_extended_tool(
        "analyze_image", {"attachment_key": attachment_key,
                          "prompt": "What is drawn?", "tiles": "off"},
        engine=_Reader(), attachments={})


def test_a_look_that_went_well_shows_no_warning(tmp_path):
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    fig = tmp_path / "made_up_figure.png"
    fig.write_bytes(_png())
    conv = tmp_path / "conv"
    logger = ActivityLogger(str(conv), turn=1)
    logger.turn_start(prompt="made-up question")
    analyze_image.invoke({"attachment_key": str(fig)},
                         config={"callbacks": [logger]})
    logger.turn_end(turn_tokens=0, answer_chars=0)
    page = run_report.build_report(str(conv))
    # the call's own view names the image it read ...
    shown = re.findall(r"<div class='(\w+)'>([^<]*made_up_figure\.png[^<]*)"
                       r"</div>", page)
    assert shown and all("sha:" in text for _kind, text in shown)
    # ... as a note: nothing went wrong on this look
    assert {kind for kind, _text in shown} == {"note"}
    assert "<div class='warn'>" not in page


def test_a_look_given_up_is_still_a_warning():
    assert "side_call_timeout" not in run_report.PLAIN_EVENTS
    assert "vision_slot_timeout" not in run_report.PLAIN_EVENTS
