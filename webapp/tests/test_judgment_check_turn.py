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

#: Every call with a source of its own: no consult needed.
CITED = SOURCED.replace(
    "None. This rests on judgment, no reference found.",
    "None, as GEC-4 Section 5.6 allows for a temporary strut.")

#: w13 F51j's wording, word for word: "no reference" with nothing looked up.
F51J = """The strut force is 27 kN/m.

## Judgment calls

- **Smooth wall contact (default) instead of bonded.** The default is the Rankine, no wall friction assumption, and I found no reference for it. I ran the bonded alternative; the results are in the table above.
- **Strut preload.** None. This rests on judgment, no reference found.
"""

NO_LIST ="The strut force is 182 kN/m, from the fem2d staged excavation."


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
    logger.turn_start(prompt=QUESTION)          # as turn_jobs does
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
    agent = _ScriptedAgent(_reply(CITED))
    answer, items, events = _run(agent, tmp_path)
    assert answer == CITED and len(agent.inputs) == 1
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} not in items
    assert [e["action"] for e in events] == ["none"]
    assert events[0]["n_calls"] == 3 and events[0]["unsourced"] == []


def test_no_reference_found_with_no_consult_is_sent_back(tmp_path,
                                                          monkeypatch):
    """w13 F51j: "I found no reference for it" with no consult, and the
    exempt words with none: nothing was looked for, so both go back."""
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(F51J), _consult_then(F51J))
    answer, items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2               # sent back once
    told = agent.inputs[1]["messages"][-1]["content"]
    assert "- Smooth wall contact (default) instead of bonded." in told
    assert "- Strut preload." in told
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} in items
    assert answer == F51J
    assert [e["action"] for e in events] == ["fed_back", "none"]
    # after the consult the exempt words count; F51j's own wording never does
    assert events[1]["references_ran"] is True
    assert events[1]["n_unsourced"] == 1


def test_the_exempt_words_after_a_consult_are_not_sent_back(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_consult_then(SOURCED))
    answer, _items, events = _run(agent, tmp_path)
    assert answer == SOURCED and len(agent.inputs) == 1
    assert events[0]["unsourced"] == []


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


def test_with_the_citation_check_one_note_and_one_retry(tmp_path,
                                                       monkeypatch):
    """Both checks fire on the same answer: one message carries both notes,
    the turn is sent back once, and neither check sends it back again. (A
    check that did not fire on the first answer can still fire once on the
    rewritten one, as the page and citation checks do.)"""
    from funhouse_agent import cited_documents as cd
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    both = UNSOURCED.replace(
        "The strut force is 182 kN/m.",
        "The strut force is 182 kN/m (AASHTO LRFD 3.11.5).")
    agent = _ScriptedAgent(_reply(both), _reply(both))
    _answer, items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2               # one retry for both checks
    told = agent.inputs[1]["messages"][-1]["content"]
    assert told.count(jc.NOTE_PREFIX) == 1
    assert told.count(cd.NOTE_PREFIX) == 1
    statuses = [i["text"] for i in items if i["kind"] == "tool_call"]
    assert jc.RETRY_STATUS in statuses and cd.RETRY_STATUS in statuses
    assert [e["action"] for e in events] == ["fed_back", "none"]
