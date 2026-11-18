"""After a summary fails, it is not tried again in that turn (BACKLOG TA-24).

The library writes the older history to a file before it makes the summary
and asks the summary call several times with a back-off, so a failure that
lasts would repeat both at every model call of the turn. The guard
(:mod:`aec_agent_core.deep.summary_guard`) remembers the failure on the
turn's scope (:class:`aec_agent_core.deep.keep_work.TurnScope`): the later
model calls of that turn are handed the history as it is, with no summary
request and no new history file; the next turn tries again.

On the real library (deepagents' own summarizer, a langchain agent, and
``create_deep_agent`` for the helper) with a fake model; made-up messages
only, no model is called.
"""

import asyncio
import re
import threading
import uuid
from typing import Any, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (AIMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import keep_work as kw
from aec_agent_core.deep import summary_guard
from aec_agent_core.deep.summary_guard import EVENT


class Busy(Exception):
    """A made-up rate-limit refusal."""
    status_code = 429


#: A line of the summary prompt in both langchain generations.
_SUMMARY_PROMPT_MARK = "Context Extraction Assistant"

#: A made-up sheet whose conversation's summary is always refused.
_REFUSED_SHEET = "Q-901"


class FakeModel(BaseChatModel):
    """A made-up model that works in steps.

    A main call whose last message is a human message holding
    ``[steps=N]`` calls the ``step`` tool, and goes on calling it until the
    tool has answered ``step N of N``; one holding ``[delegate]`` calls
    ``task`` for the ``reader`` helper; any other main call answers
    "done". Its summary call writes ``summary``; it is refused as busy when
    ``summary`` is None or the messages to summarize name
    ``refuse_about``. Records what each main call was handed (the system prompt
    apart), and how many summary calls had been made by then."""

    summary: Optional[str] = None
    refuse_about: Optional[str] = None
    summary_calls: int = 0
    seen: list = Field(default_factory=list)
    asked_before: list = Field(default_factory=list)
    on_main_call: Any = None

    @property
    def _llm_type(self) -> str:
        return "fake-summary-turn"

    def bind_tools(self, tools, **kwargs):
        return self

    def _reply(self, messages) -> AIMessage:
        last = messages[-1]
        text = str(last.content)

        def call(name, **args):
            return AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "type": "tool_call",
                "id": f"call_{uuid.uuid4().hex[:12]}"}])
        if isinstance(last, HumanMessage):
            if "[delegate]" in text:
                return call("task", subagent_type="reader",
                            description="Read the made-up sheets. [steps=3]")
            steps = re.search(r"\[steps=(\d+)\]", text)
            if steps:
                return call("step", n=1, total=int(steps.group(1)))
        if isinstance(last, ToolMessage):
            at = re.fullmatch(r"step (\d+) of (\d+)", text)
            if at and int(at.group(1)) < int(at.group(2)):
                return call("step", n=int(at.group(1)) + 1,
                            total=int(at.group(2)))
        return AIMessage(content="done")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if (len(messages) == 1
                and _SUMMARY_PROMPT_MARK in str(messages[0].content)):
            self.summary_calls += 1
            if self.summary is None or (
                    self.refuse_about
                    and self.refuse_about in str(messages[0].content)):
                raise Busy("Error code: 429 - made-up rate limit")
            reply = AIMessage(content=self.summary)
        else:
            self.seen.append([str(m.content) for m in messages
                              if not isinstance(m, SystemMessage)])
            self.asked_before.append(self.summary_calls)
            if self.on_main_call is not None:
                self.on_main_call(messages)
            reply = self._reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _step_tool(hook=None):
    @tool
    def step(n: int, total: int) -> str:
        """One made-up step of work."""
        if hook is not None:
            hook(n, total)
        return f"step {n} of {total}"
    return step


def _history(sheet="X-101", steps=3):
    return [HumanMessage(f"Read sheet {sheet}."),
            AIMessage(f"Sheet {sheet} shows a made-up ground floor."),
            HumanMessage("Now the roof sheet."),
            AIMessage("It shows a made-up roof."),
            HumanMessage(f"Compare the two. [steps={steps}]")]


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(summary_guard, "_event",
                        lambda name, data: seen.append((name, data)))
    return seen


def _summarizer(model, tmp_path):
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    assert summary_guard.install()
    return SummarizationMiddleware(
        model, backend=FilesystemBackend(root_dir=str(tmp_path),
                                         virtual_mode=True),
        trigger=("messages", 4), keep=("messages", 2))


def _agent(model, tmp_path, hook=None):
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    return create_agent(model=model, tools=[_step_tool(hook)],
                        middleware=[_summarizer(model, tmp_path)],
                        checkpointer=InMemorySaver())


def _stored(agent, config):
    return agent.get_state(config).values.get("_summarization_event")


def _history_files(tmp_path):
    return [p for p in tmp_path.rglob("*") if p.is_file()]


def _copies(tmp_path, text):
    return sum(p.read_text(encoding="utf-8", errors="replace").count(text)
               for p in _history_files(tmp_path))


def _whole(seen, history):
    """Every call was handed the whole history: the opening messages, then
    one reply and one tool result more at each call."""
    texts = [str(m.content) for m in history]
    return all(call[:len(texts)] == texts
               and len(call) == len(texts) + 2 * i
               for i, call in enumerate(seen))


# ---------------------------------------------------------------------------
# One turn, and the next
# ---------------------------------------------------------------------------

def _one_turn_then_the_next(agent, model, tmp_path, events, run):
    config = {"configurable": {"thread_id": "t-turn"}}
    history = _history()
    with kw.turn_scope() as scope:
        run(agent, {"messages": history}, config)
    # four model calls of one turn, each over the threshold
    assert len(model.seen) == 4
    assert _whole(model.seen, history)
    # the summary model was asked at the first call only
    tries = model.asked_before[0]
    assert tries >= 1
    assert model.asked_before == [tries] * 4
    assert model.summary_calls == tries
    # at most one history file, holding the older part once
    assert len(_history_files(tmp_path)) <= 1
    assert _copies(tmp_path, "Read sheet X-101.") <= 1
    assert [n for n, _ in events] == [EVENT]
    assert _stored(agent, config) is None
    assert scope.summaries_skipped == 3

    # The next turn of the same conversation tries again...
    model.seen.clear()
    model.asked_before.clear()
    with kw.turn_scope() as scope:
        run(agent, {"messages": [HumanMessage("And the section?")]}, config)
    assert model.summary_calls == 2 * tries          # asked again, once
    assert [n for n, _ in events] == [EVENT, EVENT]
    assert _stored(agent, config) is None
    assert scope.summaries_skipped == 0              # a one-call turn

    # ...and keeps the summary once it is made.
    model.summary = "Made-up summary of the first exchanges."
    model.seen.clear()
    with kw.turn_scope():
        run(agent, {"messages": [HumanMessage("And the details?")]}, config)
    assert "Made-up summary of the first exchanges." in model.seen[0][0]
    assert "Made-up summary" in str(
        _stored(agent, config)["summary_message"].content)
    assert len(events) == 2


def test_failed_summary_is_not_tried_again_in_the_turn(tmp_path, events):
    model = FakeModel()
    _one_turn_then_the_next(
        _agent(model, tmp_path), model, tmp_path, events,
        lambda agent, inp, config: agent.invoke(inp, config=config))


def test_failed_summary_is_not_tried_again_in_the_turn_async(tmp_path,
                                                             events):
    model = FakeModel()
    _one_turn_then_the_next(
        _agent(model, tmp_path), model, tmp_path, events,
        lambda agent, inp, config: asyncio.run(
            agent.ainvoke(inp, config=config)))


def test_outside_a_turn_every_call_tries(tmp_path, events):
    """A library call opens no turn scope: there is no turn to remember a
    failure in, so each call tries, as before."""
    model = FakeModel()
    agent = _agent(model, tmp_path)
    agent.invoke({"messages": _history(steps=1)},
                 config={"configurable": {"thread_id": "t-none"}})
    assert len(model.seen) == 2
    assert model.asked_before[1] == 2 * model.asked_before[0]
    assert [n for n, _ in events] == [EVENT, EVENT]


# ---------------------------------------------------------------------------
# The main agent and a helper, apart
# ---------------------------------------------------------------------------

def _deep_agent(main, reader, tmp_path, monkeypatch):
    """``create_deep_agent`` with the library's own summarizers, built with
    a low trigger so that short made-up messages reach it."""
    import deepagents.graph as graph
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import InMemorySaver
    built = []

    def low_trigger(model, backend, **kwargs):
        mw = _summarizer(model, tmp_path / f"s{len(built)}")
        built.append(mw)
        return mw
    monkeypatch.setattr(graph, "create_summarization_middleware",
                        low_trigger)
    agent = create_deep_agent(
        model=main, tools=[], system_prompt="p",
        subagents=[{"name": "reader", "description": "Reads made-up sheets.",
                    "system_prompt": "p", "model": reader,
                    "tools": [_step_tool()]}],
        checkpointer=InMemorySaver())
    assert len(built) >= 2
    return agent


def _delegating():
    return [HumanMessage("Read sheet X-101."),
            AIMessage("Sheet X-101 shows a made-up ground floor."),
            HumanMessage("Now the roof sheet."),
            AIMessage("It shows a made-up roof."),
            HumanMessage("Have the reader compare the two. [delegate]")]


def test_a_helpers_failed_summary_does_not_stop_the_main_agents(
        tmp_path, events, monkeypatch):
    main = FakeModel(summary="Made-up summary by the main agent.")
    reader = FakeModel()                             # always refused
    agent = _deep_agent(main, reader, tmp_path, monkeypatch)
    config = {"configurable": {"thread_id": "t-helper-fails"}}
    with kw.turn_scope():
        agent.invoke({"messages": _delegating()}, config=config)
    # the helper: four calls, the summary asked at its third call only,
    # the fourth handed the whole of its own history
    assert len(reader.seen) == 4
    tries = reader.summary_calls
    assert tries >= 1 and reader.asked_before == [0, 0, tries, tries]
    assert len(reader.seen[3]) == 7
    assert [n for n, _ in events] == [EVENT]
    # the main agent's call AFTER the helper failed is summarized
    assert len(main.seen) == 2
    assert main.asked_before == [1, 2]
    assert all("Made-up summary by the main agent." in call[0]
               for call in main.seen)
    assert _stored(agent, config) is not None


def test_the_main_agents_failed_summary_does_not_stop_a_helpers(
        tmp_path, events, monkeypatch):
    main = FakeModel()                               # always refused
    reader = FakeModel(summary="Made-up summary by the reader.")
    agent = _deep_agent(main, reader, tmp_path, monkeypatch)
    config = {"configurable": {"thread_id": "t-main-fails"}}
    with kw.turn_scope() as scope:
        agent.invoke({"messages": _delegating()}, config=config)
    # the main agent: asked at its first call only, both calls whole
    assert len(main.seen) == 2
    tries = main.asked_before[0]
    assert tries >= 1 and main.asked_before == [tries, tries]
    assert main.seen[0][:5] == [str(m.content) for m in _delegating()]
    assert len(main.seen[1]) == 7
    assert [n for n, _ in events] == [EVENT]
    assert _stored(agent, config) is None
    assert scope.summaries_skipped == 1
    # the helper, after that failure in the same turn, is summarized
    assert len(reader.seen) == 4
    assert reader.summary_calls >= 1
    assert "Made-up summary by the reader." in reader.seen[2][0]


def test_two_runs_of_one_helper_are_apart(events):
    """One summarizer serves every run of a helper type; a failure in one
    run of it does not stop the summary in another run of the same turn."""

    class Shaped:
        def __init__(self):
            self.summaries = 0
            self.refuse = True

        def _should_summarize(self, messages, total_tokens):
            return len(messages) >= 4

        def _create_summary(self, messages):
            self.summaries += 1
            if self.refuse:
                raise Busy("Error code: 429 - made-up rate limit")
            return "made-up summary"

        async def _acreate_summary(self, messages):
            return self._create_summary(messages)

        def wrap_model_call(self, request, handler):
            msgs = request["messages"]
            if not self._should_summarize(msgs, 0):
                return handler(list(msgs))
            summary = self._create_summary(msgs[:-2])
            return {"response": handler([f"summary: {summary}"] + msgs[-2:]),
                    "update": {"_summarization_event": summary}}

        async def awrap_model_call(self, request, handler):
            return self.wrap_model_call(request, handler)

    assert summary_guard.install(Shaped)
    mw, msgs = Shaped(), ["one", "two", "three", "four", "five"]
    first = kw.HelperRecord("call_a", "reader", "made-up brief a")
    second = kw.HelperRecord("call_b", "reader", "made-up brief b")

    def call(record):
        handed = []
        token = kw._HELPER.set(record)
        try:
            mw.wrap_model_call({"messages": list(msgs)},
                               lambda m: handed.append(list(m)) or "r")
        finally:
            kw._HELPER.reset(token)
        return handed[0]

    with kw.turn_scope() as scope:
        assert call(first) == msgs and mw.summaries == 1
        assert call(first) == msgs and mw.summaries == 1    # not again
        mw.refuse = False
        assert call(first) == msgs and mw.summaries == 1    # still not
        assert call(second)[0] == "summary: made-up summary"
        assert mw.summaries == 2
        assert scope.summaries_skipped == 2
    assert [n for n, _ in events] == [EVENT]


# ---------------------------------------------------------------------------
# Two conversations at the same moment on one agent
# ---------------------------------------------------------------------------

def _refusing_and_fine(tmp_path, hook=None):
    model = FakeModel(summary="Made-up summary of the fine conversation.",
                      refuse_about=_REFUSED_SHEET)
    return model, _agent(model, tmp_path, hook)


def _check_two_conversations(agent, model, events, bad, good):
    mine = [c for c in model.seen if any("X-101" in t or
                                         "Made-up summary" in t for t in c)]
    theirs = [c for c in model.seen if any(_REFUSED_SHEET in t for t in c)]
    assert len(theirs) == 3 and _whole(theirs, _history(_REFUSED_SHEET, 2))
    assert _stored(agent, bad) is None
    assert len(mine) == 3
    assert all("Made-up summary of the fine conversation." in c[0]
               for c in mine)
    assert _stored(agent, good) is not None
    assert [n for n, _ in events] == [EVENT]


def test_two_conversations_at_once_threads(tmp_path, events):
    """One conversation's summary fails; while its turn is still running,
    another conversation on the same agent is summarized and stored."""
    import contextvars
    # the step tool runs on LangGraph's pool, in a copy of its
    # conversation's context: that tells the hook which one it serves
    who = contextvars.ContextVar("made_up_who", default="")
    failed, other_done = threading.Event(), threading.Event()

    def hook(n, total):
        # the failing conversation's first step waits for the other one
        if n == 1 and who.get() == "bad":
            failed.set()
            assert other_done.wait(60)
    model, agent = _refusing_and_fine(tmp_path, hook)
    bad = {"configurable": {"thread_id": "t-bad"}}
    good = {"configurable": {"thread_id": "t-good"}}
    errors = []

    def run_bad():
        who.set("bad")
        try:
            with kw.turn_scope():
                agent.invoke({"messages": _history(_REFUSED_SHEET, 2)},
                             config=bad)
        except BaseException as exc:  # noqa: BLE001 - shown by the test
            errors.append(exc)
        finally:
            failed.set()

    thread = threading.Thread(target=run_bad, name="bad")
    thread.start()
    try:
        assert failed.wait(60)
        with kw.turn_scope():
            agent.invoke({"messages": _history("X-101", 2)}, config=good)
    finally:
        other_done.set()
        thread.join(60)
    assert not errors
    _check_two_conversations(agent, model, events, bad, good)


def test_two_conversations_at_once_async(tmp_path, events):
    async def main():
        import contextvars
        who = contextvars.ContextVar("made_up_who_async", default="")
        loop = asyncio.get_running_loop()
        failed, other_done = asyncio.Event(), threading.Event()

        def hook(n, total):
            if n == 1 and who.get() == "bad":
                loop.call_soon_threadsafe(failed.set)
                assert other_done.wait(60)
        model, agent = _refusing_and_fine(tmp_path, hook)
        bad = {"configurable": {"thread_id": "t-bad"}}
        good = {"configurable": {"thread_id": "t-good"}}

        async def run_bad():
            who.set("bad")
            with kw.turn_scope():
                await agent.ainvoke(
                    {"messages": _history(_REFUSED_SHEET, 2)}, config=bad)

        async def run_good():
            await asyncio.wait_for(failed.wait(), 60)
            try:
                with kw.turn_scope():
                    await agent.ainvoke(
                        {"messages": _history("X-101", 2)}, config=good)
            finally:
                other_done.set()
        await asyncio.gather(run_bad(), run_good())
        return model, agent, bad, good
    model, agent, bad, good = asyncio.run(main())
    _check_two_conversations(agent, model, events, bad, good)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def _turn_end(tmp_path, skipped):
    log = ActivityLogger(str(tmp_path), turn=1)
    with kw.turn_scope() as scope:
        log.on_chat_model_start({}, [[]], run_id=uuid.uuid4())
        for _ in range(skipped):
            scope.note_summary_skipped()
    log.turn_end(turn_tokens=0)          # the scope has closed, as on a page
    (end,) = [r for r in activity_log.load(str(tmp_path))
              if r["event"] == "turn_end"]
    return end


def test_turn_end_counts_the_calls_not_summarized(tmp_path):
    assert _turn_end(tmp_path / "a", 3)["summaries_skipped"] == 3
    assert "summaries_skipped" not in _turn_end(tmp_path / "b", 0)
