"""The delivered answer is the run's FINAL AI message (live smoke wave 1, A3).

Before 2026-10-09 ``core.stream_turn`` delivered every token the model node
streamed in the turn: narration between tool calls was glued into the reply
mid-sentence ("...pages.The vision tool needs..."), and deepagents'
context-summarization handoff ("## SESSION INTENT ... ## NEXT STEPS", with
server paths) was pasted into one answer, because the summarizer's call runs
INSIDE the model node.

These tests drive ``stream_turn`` with a scripted ``(mode, chunk)`` stream
shaped like LangGraph's ``stream_mode=["updates", "messages"]`` output, and
once through a real compiled deep agent with a scripted chat model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from webapp import core

SUMMARY = ("## SESSION INTENT\nReview the tag set.\n## ARTIFACTS\nstaged at "
           "C:\\Users\\x\\files\\long_tag_set.pdf\n## NEXT STEPS\nPages 12-23 "
           "NOT YET ANALYZED")
NARRATION_1 = "I'll open the page map first."
NARRATION_2 = "The handle is doc_69410a8d21."
ANSWER = "Sheets 3 and 7 carry the GCE tag (PDF pages 4 and 8)."


def _tok(text, node="model", **meta):
    return ("messages", (AIMessageChunk(content=text),
                         {"langgraph_node": node, **meta}))


def _model_update(text="", calls=()):
    msg = AIMessage(content=text, tool_calls=[
        {"name": n, "args": {}, "id": f"c{i}"} for i, n in enumerate(calls)])
    return ("updates", {"model": {"messages": [msg]}})


def _tools_update(name="open_document", content='{"handle": "doc_1"}'):
    return ("updates", {"tools": {"messages": [
        ToolMessage(content=content, name=name, tool_call_id="c0")]}})


class _ScriptedAgent:
    """Replays one scripted stream per graph invocation."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.inputs = []

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(inp)
        script = self.passes.pop(0) if self.passes else []
        for item in script:
            yield item


def _narrated_turn_with_a_summary():
    return [
        _tok(NARRATION_1),
        _model_update(NARRATION_1, calls=["open_document"]),
        _tools_update(),
        _tok(NARRATION_2),
        _model_update(NARRATION_2, calls=["analyze_pdf_page"]),
        _tools_update("analyze_pdf_page", '{"text": "GCE"}'),
        # the context summarizer runs inside the model node, tagged by
        # langchain with lc_source=summarization ...
        _tok(SUMMARY, lc_source="summarization"),
        # ... and the real reply follows in the same model call
        _tok("Sheets 3 and 7 carry the GCE tag "),
        _tok("(PDF pages 4 and 8)."),
        _model_update(ANSWER),
    ]


def _run(agent, messages=None):
    return list(core.stream_turn(
        agent, messages or [{"role": "user", "content": "where is GCE?"}],
        "t-final"))


def test_the_answer_is_the_final_message_not_the_narration_or_the_summary():
    entries = _run(_ScriptedAgent(_narrated_turn_with_a_summary()))
    done = entries[-1]
    assert done["kind"] == "turn_done"
    assert done["answer"] == ANSWER
    live = "".join(e["text"] for e in entries if e["kind"] == "token")
    # the summarizer's handoff never reaches even the live view
    assert "SESSION INTENT" not in live and "NOT YET ANALYZED" not in live
    assert "long_tag_set.pdf" not in live
    # narration is shown live, each model call in its own paragraph
    assert NARRATION_1 in live and NARRATION_2 in live
    assert f"{NARRATION_1}\n\n{NARRATION_2}\n\nSheets 3" in live
    assert "first.The handle" not in live


def test_summarization_chunks_are_dropped_by_the_shared_formatter():
    from funhouse_agent.deep.notebook import _format_messages_chunk
    chunk = (AIMessageChunk(content=SUMMARY),
             {"langgraph_node": "model", "lc_source": "summarization"})
    assert _format_messages_chunk(chunk) == []
    plain = (AIMessageChunk(content="hi"), {"langgraph_node": "model"})
    assert _format_messages_chunk(plain) == [{"kind": "token", "text": "hi"}]


def test_a_stream_without_updates_still_delivers_its_text():
    """An agent that streams tokens alone (no updates) keeps the old
    behaviour: the streamed text is the answer."""
    entries = _run(_ScriptedAgent([_tok("Hello "), _tok("there.")]))
    assert entries[-1]["answer"] == "Hello there."


def test_a_history_rewrite_is_not_taken_for_the_reply():
    """A node that re-sends the whole history (deepagents' tool-call patcher
    uses Overwrite; langchain's summarizer opens with REMOVE_ALL) must not
    make an EARLIER turn's answer the reply."""
    from langchain_core.messages import RemoveMessage
    from langgraph.graph.message import REMOVE_ALL_MESSAGES
    from langgraph.types import Overwrite
    old = AIMessage(content="Last turn's answer.")
    script = [
        ("updates", {"PatchToolCallsMiddleware.before_agent": {
            "messages": Overwrite([old])}}),
        ("updates", {"SummarizationMiddleware.before_model": {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), old]}}),
        _tok(ANSWER),
        _model_update(ANSWER),
    ]
    entries = _run(_ScriptedAgent(script))
    assert entries[-1]["answer"] == ANSWER
    # and the formatter does not choke on an Overwrite value
    assert not any(e["kind"] == "tool_result" for e in entries)


def test_the_coverage_gate_still_holds_the_first_reply_back(monkeypatch):
    from langchain_core.messages import HumanMessage, RemoveMessage
    from funhouse_agent.coverage import GATE_PREFIX
    draft = "Draft: LL 32."
    whole = "LL 32, PL 20; laboratory sheets 3 of 14 read."
    script = [
        _tok("Reading the sheets."),
        _model_update("Reading the sheets.", calls=["read_document"]),
        _tools_update("read_document", "{}"),
        _tok(draft),
        _model_update(draft),
        ("updates", {"CoverageGate.after_model": {"messages": [
            RemoveMessage(id="draft-1"),
            HumanMessage(content=f"{GATE_PREFIX} write your whole answer")]}}),
        _tok(whole),
        _model_update(whole),
    ]
    entries = _run(_ScriptedAgent(script))
    assert entries[-1]["answer"] == whole
    assert any(e["kind"] == "tool_call" and e["text"] == core.GATE_STATUS
               for e in entries)


def test_the_gate_with_nothing_after_it_keeps_the_held_back_reply():
    from langchain_core.messages import HumanMessage
    from funhouse_agent.coverage import GATE_PREFIX
    draft = "Draft: LL 32."
    script = [
        _tok(draft), _model_update(draft),
        ("updates", {"CoverageGate.after_model": {"messages": [
            HumanMessage(content=f"{GATE_PREFIX} note")]}}),
    ]
    assert _run(_ScriptedAgent(script))[-1]["answer"] == draft


def test_auto_continue_still_fires_on_the_final_message():
    first = "Ka is next. Let me get that Ka."
    second = "Ka = 0.33 (Rankine, phi 30)."
    agent = _ScriptedAgent(
        [_tok("Computing."), _model_update("Computing.", calls=["call_agent"]),
         _tools_update("call_agent", "{}"), _tok(first), _model_update(first)],
        [_tok("Running."), _model_update("Running.", calls=["call_agent"]),
         _tools_update("call_agent", "{}"), _tok(second), _model_update(second)])
    entries = _run(agent)
    assert len(agent.inputs) == 2                     # one nudge
    nudged = agent.inputs[1]["messages"]
    # The next pass goes on from the run's OWN messages -- its tool call and
    # result as well as its reply (live smoke wave 2b, C1) -- not text alone.
    kinds =[getattr(m, "type", None) or m.get("role") for m in nudged]
    assert kinds[-4:] == ["ai", "tool", "ai", "user"]
    assert nudged[-2].content == first
    assert nudged[-4].tool_calls[0]["name"] == "call_agent"
    assert nudged[-1]["content"] == core.CONTINUE_NUDGE
    # One answer: the announced step is replaced by the reply that took it.
    assert entries[-1]["answer"] == f"Ka is next.\n\n{second}"
    assert "Computing." not in entries[-1]["answer"]
    assert "Running." not in entries[-1]["answer"]


# ---------------------------------------------------------------------------
# Through a real compiled deep agent
# ---------------------------------------------------------------------------

def _scripted_model(replies):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            text, calls = replies.pop(0)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{len(replies)}_{i}"}
                    for i, (n, a) in enumerate(calls)]))])

    return Scripted()


def test_a_real_deep_agent_delivers_only_its_final_message(tmp_path):
    from funhouse_agent.deep.agent import build_deep_agent
    model = _scripted_model([
        (NARRATION_1, [("write_todos", {"todos": [
            {"content": "look", "status": "in_progress"}]})]),
        (ANSWER, []),
    ])
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off")
    entries = list(core.stream_turn(
        agent, [{"role": "user", "content": "q"}], "t-real"))
    assert entries[-1]["answer"] == ANSWER
    assert any(e["kind"] == "tool_call" and "write_todos" in e["text"]
               for e in entries)


def test_the_activity_log_records_what_a_command_result_says(tmp_path):
    """A sub-agent's ``task`` result is a Command; the log keeps its
    ToolMessage text, not ``Command(update=...)`` (live smoke wave 1, A15e)."""
    from uuid import uuid4
    from langgraph.types import Command
    from webapp.activity_log import ActivityLogger, load
    log = ActivityLogger(str(tmp_path), turn=1)
    rid = uuid4()
    log.on_tool_start({"name": "task"}, "", run_id=rid,
                      inputs={"subagent_type": "calc"})
    log.on_tool_end(Command(update={"messages": [ToolMessage(
        content="q_ult = 1159 kPa (Vesic).", tool_call_id="t1")],
        "files": {}}), run_id=rid)
    end = [r for r in load(str(tmp_path)) if r["event"] == "tool_end"][0]
    assert end["result"] == "q_ult = 1159 kPa (Vesic)."
    assert "Command(" not in end["result"]


def test_the_activity_log_marks_a_summarization_call(tmp_path):
    from uuid import uuid4
    from webapp.activity_log import ActivityLogger, load
    log = ActivityLogger(str(tmp_path), turn=1)
    rid = uuid4()
    log.on_chat_model_start({"name": "m"}, [[]], run_id=rid,
                            metadata={"lc_source": "summarization"})

    class _Resp:
        generations = [[type("G", (), {"message": AIMessage(content=SUMMARY)})()]]
        llm_output = None

    log.on_llm_end(_Resp(), run_id=rid)
    recs = load(str(tmp_path))
    assert [r.get("source") for r in recs] == ["summarization"] * 2
