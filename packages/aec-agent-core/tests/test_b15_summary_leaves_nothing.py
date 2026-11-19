"""A summary that fails leaves nothing in the history files (B15).

deepagents' summarizer writes the older history to a file and makes the
summary: one after the other in its sync wrap (the file first), side by
side in its async wrap. A summary that failed therefore left the file
behind, once per failing turn: a new file each turn where the library names
the file by an id it only stores with a summary (deepagents 0.7), a new
section of the conversation's one file where it names it by the thread
(deepagents 0.6.8). The guard (:mod:`aec_agent_core.deep.summary_guard`)
now has the summary made before the library's write starts, so a failure
writes nothing, and a summary that succeeds is made once and written as
before.

On the real library with a fake model; made-up messages only, no model is
called.
"""

import asyncio
from typing import Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from aec_agent_core.deep import keep_work as kw
from aec_agent_core.deep import summary_guard
from aec_agent_core.deep.summary_guard import EVENT


class Busy(Exception):
    """A made-up rate-limit refusal."""
    status_code = 429


#: A line of the summary prompt in both langchain generations.
_SUMMARY_PROMPT_MARK = "Context Extraction Assistant"

_OLDEST = "Read sheet X-101."
_SUMMARY = "Made-up summary of the first exchanges."


class FakeModel(BaseChatModel):
    """Answers "done"; its summary call writes ``summary``, or is refused as
    busy when ``summary`` is None. Records what each main call was handed."""

    summary: Optional[str] = None
    summary_calls: int = 0
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-summary-leaves-nothing"

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


def _history():
    return [HumanMessage(_OLDEST),
            AIMessage("Sheet X-101 shows a made-up ground floor."),
            HumanMessage("Now sheet X-102."),
            AIMessage("Sheet X-102 shows a made-up roof."),
            HumanMessage("Compare the two.")]


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


def _files(tmp_path):
    return sorted(p for p in tmp_path.rglob("*") if p.is_file())


def _size(tmp_path):
    return sum(p.stat().st_size for p in _files(tmp_path))


def _text(tmp_path):
    return "".join(p.read_text(encoding="utf-8") for p in _files(tmp_path))


def _sync(agent, messages, config):
    return agent.invoke({"messages": messages}, config=config)


def _async(agent, messages, config):
    return asyncio.run(agent.ainvoke({"messages": messages}, config=config))


_RUNS = pytest.mark.parametrize("run", [_sync, _async], ids=["sync", "async"])

_TURNS = (_history,
          lambda: [HumanMessage("And the section?")],
          lambda: [HumanMessage("And the details?")])


# ---------------------------------------------------------------------------
# A summary that fails
# ---------------------------------------------------------------------------

@_RUNS
def test_three_failing_turns_leave_no_history_file(tmp_path, events, run):
    model = FakeModel()                              # always refused
    agent = _agent(model, tmp_path)
    config = {"configurable": {"thread_id": "t-b15"}}
    after = []
    for n, messages in enumerate(_TURNS, start=1):
        with kw.turn_scope():
            out = run(agent, messages(), config)
        assert str(out["messages"][-1].content) == "done"
        assert model.seen[-1][0] == _OLDEST          # the whole history
        assert [name for name, _ in events] == [EVENT] * n
        after.append((len(_files(tmp_path)), _size(tmp_path)))
    # no file, no byte, after one, two and three failing turns
    assert after == [(0, 0), (0, 0), (0, 0)]
    assert agent.get_state(config).values.get(
        "_summarization_event") is None


@_RUNS
def test_a_failing_turn_after_a_summary_adds_nothing(tmp_path, events, run):
    """Once a summary is stored the library has a file to add to; a later
    summary that fails leaves that file as it was."""
    model = FakeModel(summary=_SUMMARY)
    agent = _agent(model, tmp_path)
    config = {"configurable": {"thread_id": "t-b15-later"}}
    with kw.turn_scope():
        run(agent, _history(), config)
    before = [(p, p.read_bytes()) for p in _files(tmp_path)]
    assert len(before) == 1 and not events
    stored = agent.get_state(config).values["_summarization_event"]

    model.summary = None                             # refused from here on
    asks = ("Now the section sheet.", "And the wall sheet.",
            "And the stair sheet.")
    for n, ask in enumerate(asks, start=1):
        with kw.turn_scope():
            run(agent, [HumanMessage(ask)], config)
        assert [name for name, _ in events] == [EVENT] * n
        assert [(p, p.read_bytes()) for p in _files(tmp_path)] == before
    assert agent.get_state(config).values["_summarization_event"] == stored


# ---------------------------------------------------------------------------
# A summary that succeeds: as before
# ---------------------------------------------------------------------------

@_RUNS
def test_a_summary_that_succeeds_is_made_once_and_stored_as_before(
        tmp_path, events, run):
    model = FakeModel(summary=_SUMMARY)
    agent = _agent(model, tmp_path)
    config = {"configurable": {"thread_id": "t-b15-fine"}}
    with kw.turn_scope() as scope:
        out = run(agent, _history(), config)
    assert model.summary_calls == 1                  # one summary call
    # the model: the summary, then the two messages kept
    assert len(model.seen) == 1 and len(model.seen[0]) == 3
    assert _SUMMARY in model.seen[0][0]
    assert model.seen[0][1:] == ["Sheet X-102 shows a made-up roof.",
                                 "Compare the two."]
    # one history file: the three older messages once each, nothing kept
    (path,) = _files(tmp_path)
    text = _text(tmp_path)
    for old in (_OLDEST, "Sheet X-101 shows a made-up ground floor.",
                "Now sheet X-102."):
        assert text.count(old) == 1
    assert "Compare the two." not in text and _SUMMARY not in text
    assert text.count("## Summarized at ") == 1
    # the state: the summary, the file it names, the cut after three
    stored = agent.get_state(config).values["_summarization_event"]
    assert sorted(stored) == ["cutoff_index", "file_path", "summary_message"]
    assert stored["cutoff_index"] == 3
    assert stored["file_path"].lstrip("/") == str(
        path.relative_to(tmp_path)).replace("\\", "/")
    assert _SUMMARY in str(stored["summary_message"].content)
    assert stored["file_path"] in str(stored["summary_message"].content)
    assert stored["file_path"] in model.seen[0][0]
    # the conversation's own messages are whole; nothing on the record
    assert [str(m.content) for m in out["messages"]][:5] == [
        str(m.content) for m in _history()]
    assert not events and scope.summaries_skipped == 0

    # the next summary adds a section to the same file
    with kw.turn_scope():
        run(agent, [HumanMessage("Now the section sheet.")], config)
    assert model.summary_calls == 2
    assert _files(tmp_path) == [path]
    assert _text(tmp_path).count("## Summarized at ") == 2
    assert _text(tmp_path).startswith(text)
    assert not events
