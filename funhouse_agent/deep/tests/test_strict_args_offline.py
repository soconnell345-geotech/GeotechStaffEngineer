"""An unknown tool argument is refused by name, never dropped (brief 5, N3).

Foundry brief 5 (2026-10-08): GPT-5.4 called ``analyze_pdf_page`` with
``pages: 12`` (and ``pages: "12-14"``). The tool takes ``page``; LangChain's
schema ignored the unknown key, ``page`` defaulted to 0, and the tool looked
at the report's cover three times. The agent then marked the scans skipped.
Every deep tool now answers such a call with a JSON error naming the bad key,
the keys it takes and the nearest one, and runs nothing.
"""

from __future__ import annotations

import json

import pytest

from funhouse_agent.deep.strict_args import (
    closest_argument, strict_tool, unknown_arguments_error,
)
from funhouse_agent.deep.tools import make_core_tools, make_vision_tools


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


class _Looks:
    """A vision engine that says which page image it was given."""

    def __init__(self):
        self.calls = 0

    def analyze_image(self, image, prompt=""):
        self.calls += 1
        return "a page"


def _pdf(n=15):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(n):
        doc.new_page(width=200, height=200).insert_text((20, 40),
                                                        f"page {i}")
    return doc.tobytes()


@pytest.fixture(autouse=True)
def _no_probe(monkeypatch):
    from funhouse_agent import vision_probe
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    vision_probe.clear_cache()


@pytest.mark.parametrize("pages", [12, "12-14"])
def test_pages_on_a_one_page_tool_is_refused_not_page_0(pages):
    """The brief-5 call shapes, replayed on the real tool."""
    engine = _Looks()
    tools = make_vision_tools(engine=engine, attachments={"r.pdf": _pdf()})
    out = json.loads(_tool(tools, "analyze_pdf_page").invoke(
        {"attachment_key": "r.pdf", "pages": pages}))
    assert engine.calls == 0                    # no page was looked at
    assert "page" not in out or out.get("page") != 0
    assert out["unknown_arguments"] == ["pages"]
    assert out["did_you_mean"] == {"pages": "page"}
    assert "page" in out["valid_arguments"]
    assert "NOT run" in out["error"]
    assert "once for each" in out["hint"]


def test_the_right_argument_still_works():
    engine = _Looks()
    tools = make_vision_tools(engine=engine, attachments={"r.pdf": _pdf()})
    out = json.loads(_tool(tools, "analyze_pdf_page").invoke(
        {"attachment_key": "r.pdf", "page": 12, "tiles": "off"}))
    assert "error" not in out and engine.calls >= 1


def _required(tool) -> dict:
    """A placeholder for each required argument, by its type."""
    fill = {"string": "x", "integer": 0, "number": 0.0, "boolean": False,
            "array": [], "object": {}}
    props = tool.args
    schema = tool.args_schema.model_json_schema()
    return {k: fill.get(props[k].get("type"), "x")
            for k in schema.get("required", [])}


def test_every_vision_tool_refuses_unknown_keys():
    tools = make_vision_tools(engine=_Looks(), attachments={})
    assert len(tools) > 10
    for t in tools:
        args = {**_required(t), "no_such_argument_xyz": 1}
        out = json.loads(t.invoke(args))
        assert out.get("unknown_arguments") == ["no_such_argument_xyz"], t.name


def test_the_model_sees_the_same_schema():
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    def look(attachment_key: str, page: int = 0) -> str:
        """Look at a page."""
        return json.dumps({"page": page})

    plain = StructuredTool.from_function(look, name="look", description="d")
    strict = strict_tool(plain)
    assert convert_to_openai_tool(strict) == convert_to_openai_tool(plain)
    assert strict_tool(strict) is strict                   # idempotent
    assert json.loads(strict.invoke({"attachment_key": "a", "page": 3})) \
        == {"page": 3}


def test_call_agent_keeps_its_flattened_parameters():
    """call_agent takes extra keys ON PURPOSE (flattened method parameters
    are nested for the model); it is not made strict."""
    call_agent = _tool(make_core_tools(), "call_agent")
    out = json.loads(call_agent.invoke({
        "agent_name": "bearing_capacity", "method": "no_such_method",
        "width": 2.0}))
    assert "unknown_arguments" not in out
    describe = _tool(make_core_tools(), "describe_method")
    out = json.loads(describe.invoke({"agent_name": "x", "method": "y",
                                      "module": "z"}))
    assert out["unknown_arguments"] == ["module"]


def test_closest_argument():
    valid = ["attachment_key", "page", "prompt", "tiles"]
    assert closest_argument("pages", valid) == "page"
    assert closest_argument("attachmentKey", valid) == "attachment_key"
    assert closest_argument("tile", valid) == "tiles"
    assert closest_argument("promt", valid) == "prompt"
    assert closest_argument("zzzz", valid) is None
    out = json.loads(unknown_arguments_error("t", ["zzzz"], valid))
    assert "did_you_mean" not in out and out["valid_arguments"] == valid


def test_the_primary_agent_s_tools_are_strict():
    """build_deep_agent applies it to the whole primary list, whatever
    module built a tool (host extras included)."""
    pytest.importorskip("deepagents")
    from langchain_core.tools import StructuredTool
    from funhouse_agent.deep import agent as deep_agent

    def host_tool(text: str) -> str:
        """A host's extra tool."""
        return "ran"

    captured = {}

    def fake_create(model=None, tools=(), **kw):
        captured["tools"] = list(tools)
        raise RuntimeError("stop after the tools are built")

    import unittest.mock as mock
    with mock.patch.object(deep_agent, "create_deep_agent", fake_create):
        with pytest.raises(RuntimeError, match="stop after"):
            deep_agent.build_deep_agent(
                model="fake", engine=_Looks(), reference_mode="off",
                extra_tools=[StructuredTool.from_function(
                    host_tool, name="host_tool", description="d")])
    host = next(t for t in captured["tools"] if t.name == "host_tool")
    assert json.loads(host.invoke({"text": "a", "txt": "b"}))[
        "did_you_mean"] == {"txt": "text"}
