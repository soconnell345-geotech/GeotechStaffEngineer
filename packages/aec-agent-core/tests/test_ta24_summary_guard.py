"""A summary that fails keeps the history (BACKLOG TA-24, the summarizer).

Two halves:

* the installed library, end to end: a langchain agent with deepagents' own
  summarizer and a fake model whose summary call is refused as busy. On
  langchain before 1.4 the library's summary comes back as its error line;
  on langchain 1.4 the library raises. Either way the model must be handed
  the history as it was, nothing may be stored in the state, and the next
  call tries again;
* the same flow in the shape both library generations share (read from
  deepagents 0.6.8 and 0.7.24), with each generation's way of failing, so
  that the generation not installed here is covered too.

Made-up messages only; no model is called.
"""

import asyncio
import re
from typing import Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from aec_agent_core import activity_log
from aec_agent_core.deep import summary_guard
from aec_agent_core.deep.summary_guard import (ERROR_PREFIX, EVENT,
                                               SummaryFailed)


class Busy(Exception):
    """A made-up rate-limit refusal."""
    status_code = 429


class Overflow(Exception):
    """A made-up "too long for the model"."""


#: A line of the summary prompt in both langchain generations.
_SUMMARY_PROMPT_MARK = "Context Extraction Assistant"


class FakeModel(BaseChatModel):
    """Answers "done"; its summary call writes ``summary``, or is refused as
    busy when ``summary`` is None. Records what each main call was handed."""

    summary: Optional[str] = None
    summary_calls: int = 0
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-summary-guard"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if (len(messages) == 1
                and _SUMMARY_PROMPT_MARK in str(messages[0].content)):
            self.summary_calls += 1
            if self.summary is None:
                raise Busy("Error code: 429 - made-up rate limit")
            text = self.summary
        else:
            self.seen.append([str(m.content) for m in messages])
            text = "done"
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=text))])


_HISTORY = [HumanMessage("Read sheet X-101."),
            AIMessage("Sheet X-101 shows a made-up ground floor."),
            HumanMessage("Now sheet X-102."),
            AIMessage("Sheet X-102 shows a made-up roof."),
            HumanMessage("Compare the two.")]
_TEXTS = [str(m.content) for m in _HISTORY]


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(summary_guard, "_event",
                        lambda name, data: seen.append((name, data)))
    return seen


def _agent(model, tmp_path):
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    assert summary_guard.install()
    summarizer = SummarizationMiddleware(
        model, backend=FilesystemBackend(root_dir=str(tmp_path),
                                         virtual_mode=True),
        trigger=("messages", 4), keep=("messages", 2))
    return create_agent(model=model, tools=[], middleware=[summarizer],
                        checkpointer=InMemorySaver())


def _event_of(agent, config):
    return agent.get_state(config).values.get("_summarization_event")


# ---------------------------------------------------------------------------
# The installed library, end to end
# ---------------------------------------------------------------------------

def test_failed_summary_hands_the_history_as_it_was(tmp_path, events):
    model = FakeModel()
    agent = _agent(model, tmp_path)
    config = {"configurable": {"thread_id": "t1"}}
    out = agent.invoke({"messages": list(_HISTORY)}, config=config)
    assert model.summary_calls >= 1                  # it was asked
    assert model.seen == [_TEXTS]                    # one call, whole history
    assert not any(ERROR_PREFIX in t for call in model.seen for t in call)
    assert _event_of(agent, config) is None          # nothing stored
    assert str(out["messages"][-1].content) == "done"
    assert [n for n, _ in events] == [EVENT]
    assert events[0][1]["history"] == "kept"
    assert "Error code" not in repr(events[0][1])    # no provider text

    # The next call tries again; this time the summary is made and kept.
    model.summary = "Made-up summary of the first exchanges."
    model.seen.clear()
    agent.invoke({"messages": [HumanMessage("And the section?")]},
                 config=config)
    assert len(model.seen) == 1
    assert "Made-up summary of the first exchanges." in model.seen[0][0]
    stored = _event_of(agent, config)
    assert stored is not None
    assert "Made-up summary" in str(stored["summary_message"].content)
    assert len(events) == 1


def test_failed_summary_async(tmp_path, events):
    model = FakeModel()
    agent = _agent(model, tmp_path)
    config = {"configurable": {"thread_id": "t2"}}
    asyncio.run(agent.ainvoke({"messages": list(_HISTORY)}, config=config))
    assert model.seen == [_TEXTS]
    assert _event_of(agent, config) is None
    assert [n for n, _ in events] == [EVENT]


def test_without_the_guard_the_library_loses_history_or_turn(tmp_path,
                                                              events):
    """The defect itself, on whichever generation is installed: the
    library's own wrap (``__wrapped__``) either hands the model and stores
    the error line in place of the history (langchain before 1.4) or ends
    the call on the refusal (langchain 1.4)."""
    from deepagents.middleware.summarization import SummarizationMiddleware
    guarded = SummarizationMiddleware.wrap_model_call
    SummarizationMiddleware.wrap_model_call = guarded.__wrapped__
    try:
        model = FakeModel()
        agent = _agent(model, tmp_path)
        config = {"configurable": {"thread_id": "t4"}}
        try:
            agent.invoke({"messages": list(_HISTORY)}, config=config)
        except Busy:
            assert model.seen == []                  # langchain 1.4
        else:                                        # langchain before 1.4
            assert len(model.seen) == 1
            assert ERROR_PREFIX in model.seen[0][0]
            assert _TEXTS[0] not in model.seen[0]
            stored = _event_of(agent, config)
            assert ERROR_PREFIX in str(stored["summary_message"].content)
    finally:
        SummarizationMiddleware.wrap_model_call = guarded
    assert events == []


def test_successful_summary_passes_through_untouched(tmp_path, events):
    """With the summary made, the guarded wrap hands the model exactly what
    the library's own wrap (``__wrapped__``) hands it, and stores the same
    summary."""
    from deepagents.middleware.summarization import SummarizationMiddleware
    seen = {}
    for label, wrapped in (("guarded", False), ("library", True)):
        model = FakeModel(summary="Made-up summary.")
        agent_tmp = tmp_path / label
        agent_tmp.mkdir()
        if wrapped:
            guarded = SummarizationMiddleware.wrap_model_call
            SummarizationMiddleware.wrap_model_call = guarded.__wrapped__
        try:
            agent = _agent(model, agent_tmp)
            config = {"configurable": {"thread_id": "t3"}}
            agent.invoke({"messages": list(_HISTORY)}, config=config)
        finally:
            if wrapped:
                SummarizationMiddleware.wrap_model_call = guarded
        stored = _event_of(agent, config)
        def norm(text, where=str(agent_tmp)):
            # deepagents 0.7 names the history file with a fresh id per run
            return re.sub(r"session_[0-9a-f]+", "session_<id>",
                          str(text).replace(where, "<tmp>"))
        seen[label] = ([[norm(t) for t in call] for call in model.seen],
                       norm(stored["summary_message"].content),
                       stored["cutoff_index"])
    assert seen["guarded"] == seen["library"]
    assert "Made-up summary." in seen["guarded"][0][0][0]
    assert len(seen["guarded"][0][0]) == 3           # summary + kept two
    assert events == []


def test_every_summarizer_deepagents_builds_is_guarded(monkeypatch, tmp_path):
    """The main agent's, the general-purpose helper's and each helper
    spec's summarizer come from one factory, all of the guarded class."""
    import deepagents.graph as graph
    from deepagents import create_deep_agent
    from deepagents.middleware.summarization import SummarizationMiddleware
    assert summary_guard.install()
    built = []
    factory = graph.create_summarization_middleware

    def recording(*args, **kwargs):
        mw = factory(*args, **kwargs)
        built.append(mw)
        return mw
    monkeypatch.setattr(graph, "create_summarization_middleware", recording)
    create_deep_agent(model=FakeModel(), tools=[], system_prompt="p",
                      subagents=[{"name": "reader", "description": "d",
                                  "system_prompt": "p"}])
    assert len(built) >= 3                           # main, general, reader
    assert all(isinstance(mw, SummarizationMiddleware) for mw in built)
    assert SummarizationMiddleware.__dict__.get("_aec_summary_guard")


def test_compile_agent_installs_the_guard(monkeypatch):
    from types import SimpleNamespace
    from aec_agent_core.deep import assembly
    calls = []
    monkeypatch.setattr(summary_guard, "install",
                        lambda cls=None: calls.append(cls) or True)
    assembly.compile_agent(lambda **kw: SimpleNamespace(nodes={}),
                           model=None, tools=[], system_prompt="p",
                           subagents=[], create_kwargs={})
    assert calls == [None]


def test_the_record_writes_the_event():
    assert EVENT in activity_log.APP_EVENTS


# ---------------------------------------------------------------------------
# Both generations' shape
# ---------------------------------------------------------------------------

def _shaped(failure: str):
    """A summarizer in the shape deepagents 0.6.8 and 0.7.24 share: the
    wrap asks ``_should_summarize`` (and, in 0.7, ``_over_budget``); when
    neither says so it hands the model the messages, falling to the summary
    when the model says they are too long; otherwise it makes the summary
    with ``_create_summary`` BEFORE the model call and returns the state
    update with the response. ``failure``: "raises" (langchain 1.4),
    "error_line" (langchain before 1.4) or "none"."""

    class Shaped:
        def __init__(self):
            self.summaries = 0

        def _should_summarize(self, messages, total_tokens):
            return len(messages) >= 4

        def _over_budget(self, request, total_tokens=None):
            return False

        def _create_summary(self, messages):
            self.summaries += 1
            if failure == "raises":
                raise Busy("Error code: 429 - made-up rate limit")
            if failure == "error_line":
                return f"{ERROR_PREFIX} Error code: 429 - made-up rate limit"
            return "made-up summary"

        async def _acreate_summary(self, messages):
            return self._create_summary(messages)

        def wrap_model_call(self, request, handler):
            msgs = request["messages"]
            if not (self._should_summarize(msgs, 0)
                    or self._over_budget(request, 0)):
                try:
                    return handler(list(msgs))
                except Overflow:
                    pass
            summary = self._create_summary(msgs[:-2])
            return {"response": handler([f"summary: {summary}"] + msgs[-2:]),
                    "update": {"_summarization_event": summary}}

        async def awrap_model_call(self, request, handler):
            msgs = request["messages"]
            if not (self._should_summarize(msgs, 0)
                    or self._over_budget(request, 0)):
                try:
                    return await handler(list(msgs))
                except Overflow:
                    pass
            summary = await self._acreate_summary(msgs[:-2])
            return {"response": await handler([f"summary: {summary}"]
                                              + msgs[-2:]),
                    "update": {"_summarization_event": summary}}

    assert summary_guard.install(Shaped)
    return Shaped()


_MSGS = ["one", "two", "three", "four", "five"]


def _handler(handed, overflow_from: Optional[int] = None):
    def handler(messages):
        if overflow_from is not None and len(messages) >= overflow_from:
            raise Overflow("made-up: too long")
        handed.append(list(messages))
        return "response"
    return handler


@pytest.mark.parametrize("failure", ["raises", "error_line"])
def test_shape_failed_summary_keeps_history(failure, events):
    mw, handed = _shaped(failure), []
    result = mw.wrap_model_call({"messages": list(_MSGS)}, _handler(handed))
    assert result == "response"                      # no state update
    assert handed == [_MSGS]                         # the history as it was
    assert mw.summaries == 1                         # not asked again now
    assert [n for n, _ in events] == [EVENT]
    want = {"error_type": "Busy"} if failure == "raises" else {}
    assert {k: v for k, v in events[0][1].items()
            if k == "error_type"} == want


@pytest.mark.parametrize("failure", ["raises", "error_line"])
def test_shape_failed_summary_keeps_history_async(failure, events):
    mw, handed = _shaped(failure), []

    async def handler(messages):
        handed.append(list(messages))
        return "response"
    result = asyncio.run(mw.awrap_model_call({"messages": list(_MSGS)},
                                             handler))
    assert result == "response"
    assert handed == [_MSGS]
    assert [n for n, _ in events] == [EVENT]


def test_shape_successful_summary_untouched(events):
    mw, handed = _shaped("none"), []
    result = mw.wrap_model_call({"messages": list(_MSGS)}, _handler(handed))
    assert result == {"response": "response",
                      "update": {"_summarization_event": "made-up summary"}}
    assert handed == [["summary: made-up summary", "four", "five"]]
    short = []
    assert mw.wrap_model_call({"messages": ["one"]}, _handler(short)) \
        == "response"
    assert short == [["one"]]
    assert events == []


@pytest.mark.parametrize("failure,error", [("raises", Busy),
                                           ("error_line", SummaryFailed)])
def test_shape_too_long_and_summary_fails_again_ends_on_it(failure, error,
                                                           events):
    """Asked again, the call is too long for the model, the library
    summarizes regardless, and that summary fails too: the call ends on the
    error, and nothing is stored."""
    mw, handed = _shaped(failure), []
    with pytest.raises(error):
        mw.wrap_model_call({"messages": list(_MSGS)},
                           _handler(handed, overflow_from=5))
    assert handed == []
    assert mw.summaries == 2
    assert [n for n, _ in events] == [EVENT]


def test_install_leaves_an_unknown_class_alone():
    class Other:
        def wrap_model_call(self, request, handler):
            return handler(request)
    assert summary_guard.install(Other) is False
    assert not hasattr(Other, "_aec_summary_guard")
    assert "__wrapped__" not in Other.__dict__["wrap_model_call"].__dict__


def test_install_wraps_once():
    mw = _shaped("none")
    cls = type(mw)
    first = cls.wrap_model_call
    assert summary_guard.install(cls)
    assert cls.wrap_model_call is first
    assert not hasattr(first.__wrapped__, "__wrapped__")
