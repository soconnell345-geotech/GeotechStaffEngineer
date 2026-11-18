"""The record says more of why (BACKLOG EV-20, the rest). Made-up tools,
helpers, images and replies only; no model is called.

Three additions to the run record, none to what a model or the user is
handed:

* each helper's own tools on the build record (``geotech_build``
  ``helper_tools``, which the turn's ``turn_start`` engine record and
  ``meta.json`` carry), so whether a helper had a tool is read from the
  record, not the code;
* an ``image_read`` event for every ``analyze_image`` call: which image, its
  size, the fingerprint of the bytes read (the ``sha`` the turn's
  ``turn_files`` gives a card's file) and a file's modification time, so a
  look can be matched to the version of the figure the user was shown;
* ``calls_not_started`` on ``turn_end``: the tool calls a model asked for
  that no tool started (the call a step cap cut off was only in
  ``messages.json``).
"""

from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import tool

from aec_agent_core import activity_log, run_report, vision_tools
from aec_agent_core.activity_log import ActivityLogger, file_facts
from aec_agent_core.deep import assembly

pymupdf = pytest.importorskip("pymupdf")


# -- helpers' own tools on the build record ------------------------------

@tool
def made_up_lookup(term: str) -> str:
    """Look up a made-up term."""
    return term


@tool
def made_up_plot(code: str) -> str:
    """Draw a made-up plot."""
    return code


def made_up_function(x: int) -> int:
    return x


class _Graph:
    """A compiled helper's graph with a tool node, as deepagents keeps it."""

    def __init__(self, names):
        self.nodes = {"tools": SimpleNamespace(bound=SimpleNamespace(
            tools_by_name={n: object() for n in names}))}


def _specs():
    return [
        {"name": "figure", "system_prompt": "P",
         "tools": [made_up_plot, made_up_lookup, made_up_function]},
        {"name": "general-purpose", "system_prompt": "P"},
        {"name": "reader", "runnable": _Graph(["read_b", "read_a"])},
        {"name": "opaque", "runnable": object()},
        {"name": "empty", "system_prompt": "P", "tools": []},
    ]


def test_each_helper_names_its_own_tools():
    out = assembly._helper_tool_names(_specs(), [made_up_lookup])
    assert out == {
        "figure": ["made_up_function", "made_up_lookup", "made_up_plot"],
        # no tools of its own: it inherits the primary's, as deepagents does
        "general-purpose": ["made_up_lookup"],
        "reader": ["read_a", "read_b"],
        "empty": [],
    }
    # a compiled helper with no tool node is left out, never guessed
    assert "opaque" not in out


def test_the_build_record_carries_them_and_nothing_else_moves():
    agent = SimpleNamespace()
    assembly.record_build(agent, attachments={}, model=None,
                          tools=[made_up_lookup], subagents=_specs(),
                          system_prompt="MAIN", helper_models={})
    build = agent.geotech_build
    assert build["helper_tools"]["figure"] == [
        "made_up_function", "made_up_lookup", "made_up_plot"]
    assert build["helpers"] == ["figure", "general-purpose", "reader",
                                "opaque", "empty"]
    assert build["tools"] == ["made_up_lookup"]
    assert build["prompt_chars"] == 4


def test_no_helpers_no_key():
    agent = SimpleNamespace()
    assembly.record_build(agent, attachments={}, model=None,
                          tools=[made_up_lookup], subagents=[],
                          system_prompt="MAIN", helper_models={})
    assert "helper_tools" not in agent.geotech_build


# -- the image a look read ------------------------------------------------

class _Reader:
    def __init__(self):
        self.n = 0

    def analyze_image(self, image, prompt=""):
        self.n += 1
        return "Two made-up lines on white."


def _png(shade: int) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 48), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(4, 4, 60, 8), (shade, 0, 0))
    return pix.tobytes("png")


@pytest.fixture(autouse=True)
def _fresh():
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _look_tool(engine, attachments=None):
    @tool
    def analyze_image(attachment_key: str) -> str:
        """Look at a made-up image."""
        return vision_tools.dispatch_extended_tool(
            "analyze_image",
            {"attachment_key": attachment_key, "prompt": "What is drawn?",
             "tiles": "off"},
            engine=engine, attachments=attachments or {})
    return analyze_image


def _events(folder, name):
    return [r for r in activity_log.load(str(folder))
            if r.get("event") == name]


def test_a_look_names_the_file_version_it_read(tmp_path):
    fig = tmp_path / "made_up_figure.png"
    fig.write_bytes(_png(0))
    logger = ActivityLogger(str(tmp_path / "conv"), turn=1)
    look = _look_tool(_Reader())
    look.invoke({"attachment_key": str(fig)}, config={"callbacks": [logger]})
    first = file_facts(str(fig))
    # the figure is drawn again; the next look reads the new version
    fig.write_bytes(_png(200))
    vision_tools.clear_repeat_reads()
    look.invoke({"attachment_key": str(fig)}, config={"callbacks": [logger]})
    second = file_facts(str(fig))
    assert first["sha"] != second["sha"]

    reads = _events(tmp_path / "conv", vision_tools.IMAGE_READ_EVENT)
    assert [r["sha"] for r in reads] == [first["sha"], second["sha"]]
    for r, facts in zip(reads, (first, second)):
        assert r["image"] == "made_up_figure.png"
        assert r["source"] == "path"
        assert r["bytes"] == facts["bytes"]
        assert r["tool"] == "analyze_image" and r["tool_run_id"]
    assert reads[1]["mtime"] == round(os.stat(fig).st_mtime, 3)


def test_an_upload_is_named_without_a_file_time(tmp_path):
    data = _png(0)
    logger = ActivityLogger(str(tmp_path), turn=1)
    _look_tool(_Reader(), {"made_up.png": data}).invoke(
        {"attachment_key": "made_up.png"}, config={"callbacks": [logger]})
    (read,) = _events(tmp_path, vision_tools.IMAGE_READ_EVENT)
    assert read["source"] == "attachment"
    assert read["sha"] == activity_log.fingerprint(data)
    assert "mtime" not in read


def test_a_missing_image_writes_no_event(tmp_path):
    logger = ActivityLogger(str(tmp_path), turn=1)
    out = _look_tool(_Reader()).invoke(
        {"attachment_key": str(tmp_path / "no_such.png")},
        config={"callbacks": [logger]})
    assert "error" in json.loads(out)
    assert _events(tmp_path, vision_tools.IMAGE_READ_EVENT) == []


def test_the_tool_result_is_the_same_with_and_without_a_record(tmp_path):
    fig = tmp_path / "made_up_figure.png"
    fig.write_bytes(_png(0))
    bare = vision_tools.dispatch_extended_tool(
        "analyze_image", {"attachment_key": str(fig),
                          "prompt": "What is drawn?", "tiles": "off"},
        engine=_Reader(), attachments={})
    vision_tools.clear_repeat_reads()
    logger = ActivityLogger(str(tmp_path / "conv"), turn=1)
    logged = _look_tool(_Reader()).invoke(
        {"attachment_key": str(fig)}, config={"callbacks": [logger]})
    assert logged == bare


def test_the_run_report_counts_the_reads(tmp_path):
    fig = tmp_path / "made_up_figure.png"
    fig.write_bytes(_png(0))
    conv = tmp_path / "conv"
    logger = ActivityLogger(str(conv), turn=1)
    logger.turn_start(prompt="made-up question")
    _look_tool(_Reader()).invoke({"attachment_key": str(fig)},
                                 config={"callbacks": [logger]})
    logger.turn_end(turn_tokens=0, answer_chars=0)
    page = run_report.build_report(str(conv))
    assert "image_read" in page and "1 time" in page


# -- the calls a model asked for that never started -----------------------

def _reply(calls=()) -> LLMResult:
    msg = AIMessage(content="", tool_calls=[
        {"name": n, "args": a, "id": i} for i, n, a in calls])
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _model(log, parent, calls):
    rid = uuid.uuid4()
    log.on_chat_model_start({"kwargs": {"model": "made-up-model"}},
                            [[HumanMessage(content="made-up")]],
                            run_id=rid, parent_run_id=parent)
    log.on_llm_end(_reply(calls), run_id=rid, parent_run_id=parent)


def _start(log, parent, name, args, call_id):
    rid = uuid.uuid4()
    log.on_tool_start({"name": name}, json.dumps(args), run_id=rid,
                      parent_run_id=parent, inputs=args,
                      tool_call_id=call_id)
    log.on_tool_end("done", run_id=rid, parent_run_id=parent)


def _turn_end(folder):
    (end,) = _events(folder, "turn_end")
    return end


def test_a_call_asked_for_and_never_started_is_on_turn_end(tmp_path):
    log = ActivityLogger(str(tmp_path), turn=1)
    top = uuid.uuid4()
    log.on_chain_start({}, {}, run_id=top, parent_run_id=None)
    _model(log, top, [("c1", "read_text", {"page": 2}),
                      ("c2", "analyze_image", {"attachment_key": "f.png"})])
    _start(log, top, "read_text", {"page": 2}, "c1")
    log.turn_end(turn_tokens=10, error="the step cap was reached")
    end = _turn_end(tmp_path)
    assert end["calls_not_started"] == [
        {"agent": "primary", "name": "analyze_image",
         "args": {"attachment_key": "f.png"}}]
    assert end["n_calls_not_started"] == 1


def test_a_turn_whose_calls_all_ran_has_no_key(tmp_path):
    log = ActivityLogger(str(tmp_path), turn=1)
    top = uuid.uuid4()
    log.on_chain_start({}, {}, run_id=top, parent_run_id=None)
    _model(log, top, [("c1", "read_text", {"page": 2}),
                      ("c2", "read_text", {"page": 3})])
    _start(log, top, "read_text", {"page": 2}, "c1")
    _start(log, top, "read_text", {"page": 3}, "c2")
    _model(log, top, [])
    log.turn_end(turn_tokens=10, answer_chars=5)
    end = _turn_end(tmp_path)
    assert "calls_not_started" not in end
    assert "n_calls_not_started" not in end


class _AlwaysAsks(FakeMessagesListChatModel):
    """Asks for one more made-up lookup at every step, never answers."""

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        n = sum(1 for m in messages if getattr(m, "type", "") == "ai")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"name": "made_up_lookup",
                                     "args": {"term": f"t{n}"},
                                     "id": f"call_{n}"}]))])


class _AnswersAfterOne(_AlwaysAsks):
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if any(getattr(m, "type", "") == "tool" for m in messages):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="A made-up answer."))])
        return super()._generate(messages, stop, run_manager, **kw)


def _agent(model):
    from langchain.agents import create_agent
    return create_agent(model, tools=[made_up_lookup])


def test_the_call_a_step_cap_cut_off_is_named(tmp_path):
    from langgraph.errors import GraphRecursionError
    log = ActivityLogger(str(tmp_path), turn=1)
    with pytest.raises(GraphRecursionError):
        _agent(_AlwaysAsks(responses=[])).invoke(
            {"messages": [HumanMessage(content="made-up")]},
            config={"callbacks": [log], "recursion_limit": 5})
    log.turn_end(turn_tokens=0, error="the step cap was reached")
    starts = _events(tmp_path, "tool_start")
    end = _turn_end(tmp_path)
    (cut,) = end["calls_not_started"]
    assert cut["name"] == "made_up_lookup"
    # the cut-off call is the one after the last that ran
    ran = [s["args"]["term"] for s in starts]
    assert cut["args"]["term"] not in ran
    assert cut["args"]["term"] == f"t{len(ran)}"


def test_a_turn_that_finishes_names_none(tmp_path):
    log = ActivityLogger(str(tmp_path), turn=1)
    out = _agent(_AnswersAfterOne(responses=[])).invoke(
        {"messages": [HumanMessage(content="made-up")]},
        config={"callbacks": [log], "recursion_limit": 25})
    assert out["messages"][-1].content == "A made-up answer."
    log.turn_end(turn_tokens=0, answer_chars=17)
    assert len(_events(tmp_path, "tool_start")) == 1
    assert "calls_not_started" not in _turn_end(tmp_path)
