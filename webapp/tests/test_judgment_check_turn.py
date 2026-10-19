"""The turn's "Judgment calls" list, checked for calls with no source
(BACKLOG GM-174).

With ``GEOTECH_JUDGMENT_CALLS`` on and a build that asks for the list after
a consult (``agent.geotech_judgment_consult``), ``core.stream_turn`` sends
an answer back ONCE when its list has calls that name no source and no
``references`` task ran in the turn, and keeps the check in
``activity.jsonl``. Driven here with a scripted stream.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from funhouse_agent import judgment_check as jc
from funhouse_agent import review_flags
from webapp import core
from webapp.activity_log import ActivityLogger

QUESTION = "Size the strut for the 5 m cut."

UNSOURCED = """The strut force is 182 kN/m.

## Judgment calls

1. **Wall contact.** I used a smooth wall, the tool default.
2. **Drained analysis.** You gave c' and phi' and no water table.
3. **Strut preload.** None; preloading would reduce wall movement.
"""

SOURCED = """The strut force is 182 kN/m.

## Judgment calls

1. **Wall contact.** Smooth wall, per FHWA GEC-4 Section 5.4.
2. **Drained analysis.** Long-term case, as Duncan & Wright (2014) advise.
3. **Strut preload.** None. This rests on judgment, no reference found.
"""

NO_LIST = "The strut force is 182 kN/m, from the fem2d staged excavation."


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    yield


def _reply(text):
    return [("messages", (AIMessageChunk(content=text),
                          {"langgraph_node": "model"})),
            ("updates", {"model": {"messages": [AIMessage(content=text)]}})]


def _consult_then(text):
    """A pass that asks the references helper, then answers."""
    call = AIMessage(content="", tool_calls=[{
        "name": "task", "id": "t1", "type": "tool_call",
        "args": {"subagent_type": "references",
                 "description": "Basis for each judgment call."}}])
    return [("updates", {"model": {"messages": [call]}})] + _reply(text)


class _ScriptedAgent:
    """Replays one scripted pass per graph invocation."""

    def __init__(self, *passes, consult=True):
        self.passes = list(passes)
        self.inputs = []
        if consult:
            self.geotech_judgment_consult = True

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(inp)
        for item in (self.passes.pop(0) if self.passes else []):
            yield item


def _run(agent, tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    logger = ActivityLogger(str(conv), turn=1)
    items = list(core.stream_turn(
        agent, [{"role": "user", "content": QUESTION}], "thread-1",
        callbacks=[logger]))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    path = conv / "activity.jsonl"
    rows = ([json.loads(line) for line in
             path.read_text(encoding="utf-8").splitlines()]
            if path.exists() else [])
    return done["answer"], items, [r for r in rows
                                   if r.get("event") == jc.RECORD_EVENT]


def test_an_unsourced_list_is_sent_back_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(UNSOURCED), _reply(UNSOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2               # one retry, never a second
    told = agent.inputs[1]["messages"][-1]
    assert told["role"] == "user"
    assert told["content"].startswith(jc.NOTE_PREFIX)
    for head in ("Wall contact.", "Drained analysis.", "Strut preload."):
        assert f"- {head}" in told["content"]
    assert "`references` helper once, in one task" in told["content"]
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} in items
    assert answer == UNSOURCED                  # never changed by the check
    assert [e["action"] for e in events] == ["fed_back", "none"]
    first = events[0]
    assert first["agent"] == "primary" and first["mode"] == "retry"
    assert first["n_calls"] == 3 and first["n_unsourced"] == 3
    assert first["references_ran"] is False


def test_the_reply_after_the_note_is_the_answer(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(UNSOURCED), _consult_then(SOURCED))
    answer, _items, events = _run(agent, tmp_path)
    assert answer == SOURCED
    assert [e["action"] for e in events] == ["fed_back", "none"]
    assert events[1]["references_ran"] is True
    assert events[1]["unsourced"] == []


def test_a_sourced_list_is_not_sent_back(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(SOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert answer == SOURCED and len(agent.inputs) == 1
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} not in items
    assert [e["action"] for e in events] == ["none"]
    assert events[0]["n_calls"] == 3 and events[0]["unsourced"] == []


def test_an_answer_with_no_list_is_not_sent_back(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(NO_LIST))
    answer, _items, events = _run(agent, tmp_path)
    assert answer == NO_LIST and len(agent.inputs) == 1
    assert events == []                         # nothing to record


def test_a_turn_that_consulted_is_not_sent_back(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_consult_then(UNSOURCED))
    answer, _items, events = _run(agent, tmp_path)
    assert answer == UNSOURCED and len(agent.inputs) == 1
    assert [e["action"] for e in events] == ["none"]
    assert events[0]["references_ran"] is True
    assert events[0]["n_unsourced"] == 3       # kept for the wave to read


def test_switch_off_it_never_fires(tmp_path):
    agent = _ScriptedAgent(_reply(UNSOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert answer == UNSOURCED and len(agent.inputs) == 1
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} not in items
    assert [(e["mode"], e["action"]) for e in events] == [("off", "none")]


def test_a_build_with_no_consult_never_fires(tmp_path, monkeypatch):
    """The review page, or references off: the prompt asked for no consult,
    so the check has nothing to ask for."""
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(UNSOURCED), consult=False)
    answer, _items, events = _run(agent, tmp_path)
    assert answer == UNSOURCED and len(agent.inputs) == 1
    assert [e["mode"] for e in events] == ["off"]
