"""The turn's "Judgment calls" list, checked for calls with no source
(BACKLOG GM-174, GM-187).

With ``GEOTECH_JUDGMENT_CALLS`` on and a build that asks for the list after
a consult (``agent.geotech_judgment_consult``), ``core.stream_turn`` sends
an answer back ONCE when its list has calls that name no source: asking for
the consult when no ``references`` task ran in the turn, and for a source
or "no reference found" when one did. It keeps the check in
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


def test_unsourced_after_a_consult_is_sent_back_once(tmp_path, monkeypatch):
    """GM-187 (w14 F18j: 2 of 5 calls with no source after the consult):
    the answer goes back once, asking for no second consult, and the reply
    after the note goes out as written."""
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_consult_then(UNSOURCED), _reply(UNSOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2               # one retry, never a second
    told = agent.inputs[1]["messages"][-1]["content"]
    assert told.startswith(jc.NOTE_PREFIX)
    for head in ("Wall contact.", "Drained analysis.", "Strut preload."):
        assert f"- {head}" in told
    assert "Do not ask the helper again." in told
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS_AFTER_CONSULT} \
        in items
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} not in items
    assert answer == UNSOURCED
    assert [(e["action"], e["reason"]) for e in events] == [
        ("fed_back", jc.AFTER_CONSULT), ("none", jc.AFTER_CONSULT)]
    assert events[1]["n_unsourced"] == 3       # kept for the wave to read


def test_sent_back_for_no_consult_it_is_not_sent_back_again(tmp_path,
                                                            monkeypatch):
    """Once a turn: the reply after the no-consult note still has calls with
    no source after its consult, and it goes out as written."""
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    agent = _ScriptedAgent(_reply(UNSOURCED), _consult_then(UNSOURCED),
                           _reply(SOURCED))
    answer, _items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2
    assert answer == UNSOURCED
    assert [(e["action"], e["reason"]) for e in events] == [
        ("fed_back", jc.NO_CONSULT), ("none", jc.AFTER_CONSULT)]


def test_switch_off_it_never_fires(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "0")
    agent = _ScriptedAgent(_reply(UNSOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert answer == UNSOURCED and len(agent.inputs) == 1
    assert {"kind": "tool_call", "text": jc.RETRY_STATUS} not in items
    assert [(e["mode"], e["action"]) for e in events] == [("off", "none")]


def test_on_by_default_it_fires(tmp_path):
    """GM-226 (owner, OWN-65, 2026-10-20): the switch unset is on."""
    agent = _ScriptedAgent(_reply(UNSOURCED), _reply(SOURCED))
    answer, items, events = _run(agent, tmp_path)
    assert len(agent.inputs) == 2 and answer == SOURCED
    assert [e["mode"] for e in events][0] == "retry"


def _chat_build(monkeypatch, **behavior):
    """The geotech page's build from the chat's own Behavior settings
    (``core.behavior_build_kwargs``), the switch at its default; returns
    what create_deep_agent was handed and the agent."""
    pytest.importorskip("deepagents")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from funhouse_agent.deep import agent as agent_mod

    class _Quiet(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "quiet"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="done"))])

    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    kw = core.behavior_build_kwargs({**core.default_behavior(), **behavior})
    agent = agent_mod.build_deep_agent(model=_Quiet(), **kw)
    monkeypatch.setattr(agent_mod, "create_deep_agent", real)
    return captured, agent


def test_chat_references_off_stops_the_consult_and_the_check(
        tmp_path, monkeypatch):
    """The owner (OWN-65): the user can lower or stop reference calls from
    the chat. With "Consult references" off in Behavior, the judgment calls
    (on by default) build no references helper to consult, the prompt asks
    for no consult (each call names a `choices` basis or says it rests on
    judgment), and the turn-end check never sends an answer back."""
    from funhouse_agent.deep.prompt import judgment_calls_section
    captured, agent = _chat_build(monkeypatch, references="off")
    names = {s.get("name") for s in captured["subagents"]}
    assert not {"references", "reviewer"} & names
    prompt = captured["system_prompt"]
    assert judgment_calls_section(references=False) in prompt
    assert "`references` helper" not in prompt
    flat = " ".join(prompt.split())
    assert ("(a `choices` basis or a reference looked up); otherwise say "
            "plainly that it rests on judgment, with no reference cited;") \
        in flat
    assert agent.geotech_judgment_consult is False
    assert jc.mode_for(agent) == "off"
    # the turn: an unsourced list goes out as written, never sent back
    scripted = _ScriptedAgent(_reply(UNSOURCED), consult=False)
    scripted.geotech_judgment_consult = agent.geotech_judgment_consult
    answer, _items, events = _run(scripted, tmp_path)
    assert answer == UNSOURCED and len(scripted.inputs) == 1
    assert [e["mode"] for e in events] == ["off"]


def test_chat_references_on_asks_for_the_consult(monkeypatch):
    """The default chat (references on): the judgment calls ask for one
    consult and the turn-end check acts."""
    captured, agent = _chat_build(monkeypatch)
    assert "references" in {s.get("name") for s in captured["subagents"]}
    assert "`references` helper" in captured["system_prompt"]
    assert agent.geotech_judgment_consult is True
    assert jc.mode_for(agent) == "retry"


def test_chat_reference_budget_caps_the_judgment_consult(monkeypatch):
    """A lower "Reference consult budget" in the chat's Advanced caps is
    the consult's budget (the judgment consult never raises it)."""
    from funhouse_agent.deep.limits import (
        JUDGMENT_REFERENCES_MAX_MODEL_CALLS, ModelCallBudgetMiddleware)
    captured, _agent = _chat_build(monkeypatch, ref_max_calls=2)
    spec = next(s for s in captured["subagents"]
                if s.get("name") == "references")
    limits = [m.run_limit for m in spec.get("middleware") or []
              if isinstance(m, ModelCallBudgetMiddleware)]
    assert limits == [2] and 2 < JUDGMENT_REFERENCES_MAX_MODEL_CALLS
    assert "You have 2 model calls" in spec["system_prompt"]


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
