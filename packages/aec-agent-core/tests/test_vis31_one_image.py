"""VIS-31 (1): every reading that goes back to the main model is told it
sees one image, with or without a bound user question. Fake engine, no
model."""

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import turn_question, vision_tools  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402

PHRASE = "ONE image only"


class Recorder:
    def __init__(self):
        self.prompts = []

    def analyze_image(self, image, prompt=""):
        self.prompts.append(prompt)
        return "ok"


def _pdf():
    d = pymupdf.open()
    p = d.new_page(width=612, height=792)
    p.insert_text((72, 100), "B-1  B-2", fontsize=12)
    p.draw_rect(pymupdf.Rect(300, 300, 500, 500), color=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


@pytest.fixture(autouse=True)
def _fresh():
    vision_tools.clear_repeat_reads()


@pytest.mark.parametrize("kind", sorted(turn_question.QUESTION_KINDS))
def test_every_question_kind_is_told(kind):
    eng = Recorder()
    vision_tools.ask_vision(eng, b"x", "What changed?", kind)
    assert PHRASE in eng.prompts[0]
    assert "comparison needs the other image" in eng.prompts[0]


@pytest.mark.parametrize("kind", ["check", "find"])
def test_code_reads_are_not(kind):
    eng = Recorder()
    vision_tools.ask_vision(eng, b"x", "Is it there?", kind)
    assert eng.prompts[0] == "Is it there?"


def test_with_and_without_a_bound_question():
    eng = Recorder()
    vision_tools.ask_vision(eng, b"x", "p", "page")
    with turn_question.bound("Compare with the earlier issue"):
        vision_tools.ask_vision(eng, b"x", "p", "page")
    assert all(PHRASE in p for p in eng.prompts)
    assert "earlier issue" not in eng.prompts[0]
    assert "earlier issue" in eng.prompts[1]


def test_a_page_read_through_the_tool_is_told():
    eng = Recorder()
    json.loads(dispatch_extended_tool(
        "analyze_pdf_page", {"attachment_key": "s.pdf", "page": 0,
                             "prompt": "What changed?", "tiles": "off"},
        engine=eng, attachments={"s.pdf": _pdf()}))
    assert eng.prompts and all(PHRASE in p for p in eng.prompts)
