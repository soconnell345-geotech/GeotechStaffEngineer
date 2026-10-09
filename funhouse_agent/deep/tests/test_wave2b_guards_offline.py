"""Live smoke wave 2b: the guards on every agent's loop, and the engines.

* C3  a tool that raises is a tool error the model reads; the turn goes on.
* C5  a reply cut at the output-token limit: a cut tool call is not run (the
      model is told why), cut text is continued; each engine reports "cut".
* C12 the Document Review page shows no scratch-filesystem tools.
* C10 the vision probe holds no process-wide lock while it runs, and sends
      its calls at once.

Offline: scripted chat models through real compiled agents.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from funhouse_agent.deep import tool_guards
from funhouse_agent.deep.agent import build_deep_agent


def _scripted(replies, seen=None, tools_seen=None, system_seen=None):
    """A chat model that answers from ``replies`` (AIMessages or
    ``(text, [(name, args)])``) and records what it was sent."""

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            if tools_seen is not None:
                tools_seen.append([getattr(t, "name", None)
                                   or t.get("name") if isinstance(t, dict)
                                   else getattr(t, "name", None)
                                   for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if seen is not None:
                seen.append(list(messages))
            if system_seen is not None:
                system_seen.append(next(
                    (m.content for m in messages
                     if getattr(m, "type", "") == "system"), ""))
            reply = replies.pop(0)
            if not isinstance(reply, AIMessage):
                text, calls = reply
                reply = AIMessage(content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{len(replies)}_{i}"}
                    for i, (n, a) in enumerate(calls)])
            return ChatResult(generations=[ChatGeneration(message=reply)])

    return Scripted()


def _run(agent, text="go"):
    return agent.invoke({"messages": [{"role": "user", "content": text}]},
                        config={"recursion_limit": 60})


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------

def _raising_tool():
    def save_markup(path: str) -> str:
        """Save a marked copy."""
        raise RuntimeError(
            "FzErrorSystem: code=2: cannot remove file 'C:\\Users\\x\\data\\"
            "users\\livesmoke__tester\\files\\review_set_marked.pdf'")
    return StructuredTool.from_function(save_markup, name="save_markup")


def test_a_raising_tool_does_not_end_the_turn():
    """F31 t4: the turn ended on "(no answer text)" and the raw error with
    the tester's folder in it."""
    seen = []
    model = _scripted([("", [("save_markup", {"path": "x.pdf"})]),
                       ("The save failed; I will say so.", [])], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off",
                             extra_tools=[_raising_tool()])
    out = _run(agent)
    assert out["messages"][-1].content == "The save failed; I will say so."
    told = [m for m in seen[-1] if isinstance(m, ToolMessage)][-1]
    body = json.loads(told.content)
    assert body["error"].startswith("save_markup failed: RuntimeError")
    assert "review_set_marked.pdf" in body["error"]
    assert "livesmoke__tester" not in told.content
    assert told.status == "error"


def test_graph_control_flow_still_passes_through():
    from langgraph.errors import GraphInterrupt
    guard = tool_guards.ToolErrorGuard()

    class _Req:
        tool_call = {"name": "t", "id": "1"}

    def handler(_req):
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        guard.wrap_tool_call(_Req(), handler)


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------

def _cut(msg: AIMessage, how="finish_reason", value="length",
         tokens=8192) -> AIMessage:
    msg.response_metadata = {how: value}
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": tokens,
                          "total_tokens": tokens + 10}
    return msg


@pytest.mark.parametrize("meta,cut", [
    ({"finish_reason": "length"}, True),             # OpenAI chat
    ({"stop_reason": "max_tokens"}, True),           # Anthropic
    ({"status": "incomplete",
      "incomplete_details": {"reason": "max_output_tokens"}}, True),
    ({"finish_reason": "stop"}, False),
    ({"finish_reason": "tool_calls"}, False),
    ({}, False),
])
def test_a_cut_reply_is_read_off_the_providers_stop_reason(meta, cut):
    assert tool_guards.output_was_cut(AIMessage(content="x",
                                                response_metadata=meta)) is cut


def _saving_tool(ran):
    def save_file(path: str, content: str) -> str:
        """Save a file."""
        ran.append((path, len(content)))
        return json.dumps({"saved": path})
    return StructuredTool.from_function(save_file, name="save_file")


def test_a_cut_tool_call_is_not_run_and_the_model_is_told_why():
    """F40 t4: five identical cut save_file calls, each answered "content:
    Field required"."""
    ran, seen = [], []
    cut_call = _cut(AIMessage(content="", tool_calls=[
        {"name": "save_file", "args": {"path": "plan.dxf"}, "id": "s1"}]),
        how="stop_reason", value="max_tokens")
    model = _scripted([cut_call,
                       ("", [("save_file", {"path": "a.txt",
                                            "content": "part 1"})]),
                       ("Saved in parts.", [])], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off",
                             extra_tools=[_saving_tool(ran)])
    out = _run(agent)
    assert out["messages"][-1].content == "Saved in parts."
    assert ran == [("a.txt", 6)]                  # the cut call never ran
    told = [m for m in seen[1] if isinstance(m, ToolMessage)][-1]
    body = json.loads(told.content)
    assert "CUT OFF at the model's output limit" in body["error"]
    assert "8,192" in body["error"] and "only path arrived" in body["error"]
    assert "smaller" in body["hint"]


def test_the_complete_calls_before_the_cut_one_still_run():
    ran, seen = [], []
    cut = _cut(AIMessage(content="", tool_calls=[
        {"name": "save_file", "args": {"path": "a.txt", "content": "ok"},
         "id": "s1"},
        {"name": "save_file", "args": {"path": "b.txt"}, "id": "s2"}]))
    model = _scripted([cut, ("Done.", [])], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off",
                             extra_tools=[_saving_tool(ran)])
    _run(agent)
    assert ran == [("a.txt", 2)]


def test_a_call_too_broken_to_parse_is_answered_not_dropped():
    """ChatOpenAI puts unparseable arguments in invalid_tool_calls: the
    reply would end the turn with no text. It is answered instead."""
    seen = []
    broken = _cut(AIMessage(content="", invalid_tool_calls=[{
        "name": "save_file", "args": '{"path": "a.txt", "content": "abc',
        "id": "s9", "error": "bad json", "type": "invalid_tool_call"}]))
    model = _scripted([broken, ("I will send it in parts.", [])], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off",
                             extra_tools=[_saving_tool([])])
    out = _run(agent)
    assert out["messages"][-1].content == "I will send it in parts."
    told = [m for m in seen[1] if isinstance(m, ToolMessage)][-1]
    assert "CUT OFF" in told.content


def test_cut_text_is_continued_and_delivered_as_one_reply():
    """F42 t5: a whole-page transcription stopped mid-sentence, unflagged."""
    seen = []
    model = _scripted([
        _cut(AIMessage(content="The notes read: 1. All concrete 3600 psi. "
                               "2. This is")),
        AIMessage(content=" the second note, complete.",
                  response_metadata={"finish_reason": "stop"}),
    ], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off")
    out = _run(agent)
    assert out["messages"][-1].content == (
        "The notes read: 1. All concrete 3600 psi. 2. This is the second "
        "note, complete.")
    assert seen[1][-1].content == tool_guards.CONTINUE_CUT_TEXT


def test_the_palantir_responses_route_reports_a_cut_call_as_cut():
    pytest.importorskip("webapp.palantir_sdk_engine")
    import inspect
    from webapp import palantir_sdk_engine as P
    src = inspect.getsource(P.PalantirSdkChatModel._generate_responses)
    assert 'finish = ("length" if status.lower() == "incomplete"' in src


def test_the_prompter_model_lowers_a_cap_the_model_refuses():
    from funhouse_agent.deep.databricks_bridge import \
        _adjust_request_for_param_error
    req = {"model": "m", "max_completion_tokens": 32000}
    out = _adjust_request_for_param_error(
        req, "max_tokens is too large: 32000. This model supports at most "
             "16384 completion tokens, whereas you provided 32000.")
    assert out["max_completion_tokens"] == 16384


# ---------------------------------------------------------------------------
# C12: no scratch filesystem on the Document Review page
# ---------------------------------------------------------------------------

def test_the_review_page_shows_no_scratch_tools(monkeypatch):
    monkeypatch.delenv("GEOTECH_REVIEW_AGENT", raising=False)
    from funhouse_agent import review_flags
    if review_flags.lean_agent() or review_flags.minimal_agent():
        pytest.skip("a review-agent switch is on in this environment")
    from funhouse_agent.deep.prompt import build_document_review_prompt
    tools_seen, system_seen = [], []

    class Recording(BaseChatModel):
        @property
        def _llm_type(self):
            return "rec"

        def bind_tools(self, tools, **kw):
            tools_seen.append([t.get("name") if isinstance(t, dict)
                               else getattr(t, "name", None) for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            system_seen.append(" ".join(
                str(m.content) for m in messages
                if getattr(m, "type", "") == "system"))
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Hello."))])

    agent = build_deep_agent(Recording(), allowed_agents=(),
                             reference_mode="off", review_page=True,
                             system_prompt=build_document_review_prompt())
    _run(agent)
    names = set(tools_seen[-1])
    assert not names & tool_guards.SCRATCH_TOOLS
    assert "open_document" in names and "write_todos" in names
    assert "## Filesystem Tools" not in system_seen[-1]
    assert "Following Conventions" not in system_seen[-1]
    # the geotech page keeps them
    tools_seen.clear()
    plain = build_deep_agent(Recording(), allowed_agents=(),
                             reference_mode="off")
    _run(plain)
    assert "read_file" in set(tools_seen[-1])


def test_an_unknown_tool_error_does_not_list_the_hidden_tools(monkeypatch):
    """F31 t3: the model called a made-up read_docx_placeholder and the
    error listed ls, read_file, write_file, ... as the page's tools."""
    monkeypatch.delenv("GEOTECH_REVIEW_AGENT", raising=False)
    from funhouse_agent import review_flags
    if review_flags.lean_agent() or review_flags.minimal_agent():
        pytest.skip("a review-agent switch is on in this environment")
    seen = []
    model = _scripted([("", [("read_docx_placeholder", {"path": "a.docx"})]),
                       ("ok", [])], seen=seen)
    agent = build_deep_agent(model, allowed_agents=(), reference_mode="off",
                             review_page=True)
    _run(agent)
    told = [m for m in seen[-1] if isinstance(m, ToolMessage)][-1]
    assert "read_docx_placeholder" in told.content
    assert "open_document" in told.content
    for hidden in ("read_file", "write_file", "edit_file", "glob", "grep"):
        assert hidden not in told.content.split("try one of", 1)[-1], hidden


def test_the_lean_review_agent_carries_both_guards(monkeypatch):
    from funhouse_agent.deep import review_agent
    captured = {}
    import langchain.agents as la
    real = la.create_agent

    def spy(model, **kw):
        captured.setdefault("mw", kw.get("middleware"))
        return real(model, **kw)

    monkeypatch.setattr(la, "create_agent", spy)
    review_agent.build_review_agent(_scripted([("hi", [])]))
    kinds = [type(m).__name__ for m in captured["mw"]]
    assert "ToolErrorGuard" in kinds
    assert kinds[-1] == "OutputLimitGuard"


# ---------------------------------------------------------------------------
# C10: the vision probe
# ---------------------------------------------------------------------------

class _SlowModel:
    """Answers a probe call after ``delay`` seconds; counts calls in flight."""

    def __init__(self, name, delay=0.3):
        self.model_name = name
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def invoke(self, messages):
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.delay)
        with self._lock:
            self.in_flight -= 1
        msg = AIMessage(content="OK", response_metadata={
            "model_name": self.model_name})
        msg.usage_metadata = {"input_tokens": 12, "output_tokens": 1,
                              "total_tokens": 13}
        return msg


@pytest.fixture
def probing(monkeypatch):
    from funhouse_agent import vision_probe
    monkeypatch.setenv(vision_probe.PROBE_ENV, "1")
    vision_probe.clear_cache()
    yield vision_probe
    vision_probe.clear_cache()


def test_the_probe_sends_its_calls_at_once(probing):
    """F29 t1: four serial probe calls held the first vision call 23 s."""
    model = _SlowModel("m-a", delay=0.3)
    t0 = time.monotonic()
    probing.probe(model)
    took = time.monotonic() - t0
    assert model.max_in_flight == 4
    assert took < 4 * 0.3


def test_one_models_probe_does_not_hold_up_anothers(probing):
    slow, quick = _SlowModel("slow", delay=0.6), _SlowModel("quick", 0.01)
    started = threading.Event()

    def first():
        started.set()
        probing.profile_for(slow)

    t = threading.Thread(target=first)
    t.start()
    started.wait()
    time.sleep(0.05)
    t0 = time.monotonic()
    probing.profile_for(quick)
    assert time.monotonic() - t0 < 0.5
    t.join()


def test_the_same_model_is_probed_once_when_asked_at_once(probing):
    model = _SlowModel("same", delay=0.2)
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        probing.profile_for(model))) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert model.calls == 4                    # one probe: four calls
    assert len({id(r) for r in results}) == 1
