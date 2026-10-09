"""Live smoke wave 3 (module_work/live_smoke/runs/w3-confirm-sonnet/
REVIEW.md), F1: zooms and pages read whole were the long pole.

17 zoom / page-alone calls over 20 s took 533 s; about two-thirds of their
output tokens were the model's own reasoning and the visible rest was
verbose (a box for every leader segment, narration, closing summaries).
Both kinds now carry an answer shape -- the request and the lettering in
view, quoted exactly, one box per thing named, no commentary -- and the
page-alone cap is 8,000 (above every page answer on record). What a read
can SEE is unchanged: the same image, at the same size, and the same tiles.

Fakes and synthetic PDFs only: no model, no network.
"""

from __future__ import annotations

import json

import pytest

fitz = pytest.importorskip("fitz")

from funhouse_agent import vision_tools, vision_view  # noqa: E402
from funhouse_agent.vision_tools import dispatch_extended_tool  # noqa: E402

QUESTION = "Read every label and dimension in this region verbatim."


def _pdf() -> bytes:
    d = fitz.open()
    page = d.new_page(width=612, height=792)
    page.insert_text((72, 100), "Sheet 1  B-1  B-2", fontsize=12)
    page.draw_rect(fitz.Rect(300, 300, 500, 500), color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


class Reader:
    """Takes a cap, as the app's engine does; records image, prompt, cap."""

    accepts_output_cap = True

    def __init__(self):
        self.calls = []

    def analyze_image(self, image, prompt="", max_output_tokens=None):
        self.calls.append((bytes(image), prompt, max_output_tokens))
        return "B-1 px=[10, 10, 40, 30]"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(vision_tools.VISION_CAP_ENV, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _call(tool, engine, pdf, **args):
    base = {"attachment_key": "sheet.pdf", "page": 0, "prompt": QUESTION}
    return json.loads(dispatch_extended_tool(
        tool, {**base, **args}, engine=engine,
        attachments={"sheet.pdf": pdf}))


def _order(prompt, *parts):
    """Each part is in the prompt, in this order."""
    at = [prompt.index(p) for p in parts]
    return at == sorted(at)


def test_a_zoom_asks_for_the_request_and_the_lettering_in_view_briefly():
    engine = Reader()
    out = _call("render_region", engine, _pdf(), bbox=[60, 80, 300, 120])
    assert "error" not in out, out
    ((_img, prompt, cap),) = engine.calls
    assert prompt.startswith(QUESTION)                  # the agent's ask first
    assert vision_tools.ZOOM_ANSWER_SHAPE in prompt
    assert "then give the rest of the lettering in this view, quoted " \
           "exactly as printed" in prompt
    assert "ONE box per thing you name" in prompt
    assert "no boxes for parts of it, for the path of a leader" in prompt
    assert "no closing summary" in prompt
    # ... before the location sentence, which still asks for pixel boxes
    assert _order(prompt, QUESTION, vision_tools.ZOOM_ANSWER_SHAPE,
                  "If you give the location of anything in it")
    assert cap == vision_tools.VISION_OUTPUT_CAPS["region"] == 8000


def test_a_page_read_whole_asks_for_the_request_briefly_with_a_lower_cap():
    engine = Reader()
    out = _call("analyze_pdf_page", engine, _pdf(), tiles="off",
                prompt="What is the scale?")
    assert "error" not in out, out
    ((_img, prompt, cap),) = engine.calls
    assert prompt.startswith("What is the scale?")
    assert vision_tools.PAGE_ANSWER_SHAPE in prompt
    assert "quoting the page's own lettering that bears on it exactly as " \
           "printed" in prompt
    assert _order(prompt, "What is the scale?",
                  vision_tools.PAGE_ANSWER_SHAPE,
                  "If you give the location of anything in it")
    assert cap == vision_tools.VISION_OUTPUT_CAPS["page"] == 8000


def test_the_page_cap_is_above_every_page_answer_on_record():
    """The longest page-alone answers: 6,766 tokens in wave 3 (F41's index
    sheet), 5,844 in waves 2a-2c; zooms 6,321 and 6,841."""
    assert vision_tools.VISION_OUTPUT_CAPS["page"] > 6766
    assert vision_tools.VISION_OUTPUT_CAPS["region"] > 6841


def test_tiles_and_the_layout_overview_keep_their_prompts():
    engine = Reader()
    _call("analyze_pdf_page", engine, _pdf(), tiles="2")
    assert len(engine.calls) == 5
    for _img, prompt, _cap in engine.calls:
        assert vision_tools.PAGE_ANSWER_SHAPE not in prompt
        assert vision_tools.ZOOM_ANSWER_SHAPE not in prompt
    caps = sorted(c for _i, _p, c in engine.calls)
    assert caps == [vision_tools.VISION_OUTPUT_CAPS["overview"]] + \
        [vision_tools.VISION_OUTPUT_CAPS["tile"]] * 4


def test_what_a_read_sees_is_unchanged():
    """Robust first: the page and the zoom are sent as the same image the
    view renderer draws for them -- same size, same pixels."""
    pdf = _pdf()
    engine = Reader()
    out = _call("analyze_pdf_page", engine, pdf, tiles="off")
    image, _info = vision_view.render_view(pdf, page=0, engine=engine)
    assert engine.calls[0][0] == image
    assert f"This image is {out['view_px'][0]} x {out['view_px'][1]} " \
           "pixels" in engine.calls[0][1]
    engine = Reader()
    out = _call("render_region", engine, pdf, bbox=[60, 80, 300, 120])
    image, info = vision_view.render_view(pdf, page=0,
                                          bbox=[60, 80, 300, 120],
                                          pad_frac=0.15, engine=engine)
    assert engine.calls[0][0] == image
    assert out["view_px"] == [info["width_px"], info["height_px"]]


def test_other_vision_prompts_take_no_shape():
    assert vision_tools._answer_shape("overview") is None
    assert vision_tools._answer_shape("tile") is None
    assert vision_tools._answer_shape("check") is None
    assert vision_tools._vision_prompt("Q", [0, 0, 10, 10], None,
                                       (100, 100)).startswith("Q\n\nThis "
                                                              "image is")
