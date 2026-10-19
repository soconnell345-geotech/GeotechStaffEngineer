"""The calc helper looks at the pages of the PDF it builds (BACKLOG GM-175).

Live wave w13b, F55j: asked to look at its rebuilt PDF's pages, the calc
helper had no page tool. It called ``execute`` with no command, then
``analyze_image`` on the PDF, whose error told it to use
``analyze_pdf_page`` -- a tool it did not have. Now it has the tool, and
``analyze_image``'s "not an image" errors name only tools the caller has.
"""

from __future__ import annotations

import json

from funhouse_agent import vision_tools
from funhouse_agent.deep import agent as agent_mod
from funhouse_agent.deep.tools import make_vision_tools

PDF = b"%PDF-1.4\n%fake\n"


def _names(spec):
    return {getattr(t, "name", None) for t in spec["tools"]}


def test_the_calc_helper_has_the_page_tool():
    names = _names(agent_mod.build_calc_subagent())
    assert {"analyze_pdf_page", "analyze_image", "read_pdf_text"} <= names
    assert "open_document" not in names
    assert "render_region" not in names


def test_the_calc_prompt_names_no_new_rule():
    """No rule asks the helper to look at its pages, so its prompt does not
    name the tool (the read-back rule checks page_contents and the text)."""
    prompt = agent_mod.build_calc_subagent()["system_prompt"]
    assert "analyze_pdf_page" not in prompt


def test_the_pdf_error_names_only_the_callers_tools():
    msg = vision_tools._not_an_image_message
    # every tool, as before, when the caller's tools are not known
    assert msg(PDF, "calc.pdf") == (
        "'calc.pdf' is a PDF, not an image: look at a page with "
        "analyze_pdf_page, or open it with open_document.")
    calc = {"analyze_image", "analyze_pdf_page", "read_pdf_text"}
    assert msg(PDF, "calc.pdf", calc) == (
        "'calc.pdf' is a PDF, not an image: look at a page with "
        "analyze_pdf_page.")
    assert msg(PDF, "calc.pdf", {"analyze_image", "read_pdf_text"}) == (
        "'calc.pdf' is a PDF, not an image: read its text with "
        "read_pdf_text.")
    assert msg(PDF, "calc.pdf", {"analyze_image"}) == (
        "'calc.pdf' is a PDF, not an image: no tool here shows its pages.")
    assert msg(b"a,b\n1,2\n", "t.csv", {"analyze_image"}) == (
        "'t.csv' is a text file, not an image.")
    assert msg(b"a,b\n1,2\n", "t.csv") == (
        "'t.csv' is a text file, not an image: read it with read_text_file.")


def _analyze_image(include, attachments):
    tools = make_vision_tools(attachments=attachments, include=include)
    tool = next(t for t in tools if t.name == "analyze_image")
    return json.loads(tool.invoke({"attachment_key": "calc.pdf",
                                   "prompt": "Look at page 1."}))


def test_analyze_image_on_a_pdf_points_at_a_tool_the_agent_has():
    out = _analyze_image({"analyze_image", "read_pdf_text"},
                         {"calc.pdf": PDF})
    assert "read_pdf_text" in out["error"]
    assert "analyze_pdf_page" not in out["error"]
    assert "open_document" not in out["error"]
    out = _analyze_image({"analyze_image", "analyze_pdf_page",
                          "read_pdf_text"}, {"calc.pdf": PDF})
    assert "analyze_pdf_page" in out["error"]
    assert "open_document" not in out["error"]
