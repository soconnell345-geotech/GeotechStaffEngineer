"""The turn's answer, checked for documents nothing looked up (BACKLOG EV-21).

``core.stream_turn`` checks every published document the delivered answer
cites against the conversation's record (``funhouse_agent.cited_documents``)
and the user's own messages, keeps the result in ``activity.jsonl`` and, per
``GEOTECH_CITED_DOCUMENTS``, leaves the answer (off, the default), appends a
note (``note``) or holds the answer back once and tells the model
(``retry``). With EV-18's page check also telling the model, the two go in
ONE note. Driven here with a scripted stream and a synthetic record.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from funhouse_agent import cited_documents as cd
from funhouse_agent import cited_pages as cp
from funhouse_agent import review_flags
from webapp import core
from webapp.activity_log import ActivityLogger

QUESTION = "Use Rankine active pressure on our cantilever wall."
BAD = "Ka = 0.307 (Rankine 1857; GEC-11 / AASHTO LRFD 3.11.5)."
#: Rankine (1857): the user asked for Rankine and the module ran it, so the
#: author is in the record and only the year is the model's (not flagged).
GOOD = "Ka = 0.307 by Rankine (1857), from the retaining_walls module."
BOTH = BAD + " Boring 12 is logged on PDF pages 108-119."


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    cd.clear_cache()
    cp.clear_cache()
    yield
    cd.clear_cache()
    cp.clear_cache()


def _reply(text):
    return [("messages", (AIMessageChunk(content=text),
                          {"langgraph_node": "model"})),
            ("updates", {"model": {"messages": [AIMessage(content=text)]}})]


class _ScriptedAgent:
    """Replays one scripted pass per graph invocation, and notes the user's
    words a tool running in the turn would see."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.inputs = []
        self.words = []

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(inp)
        self.words.append(cd._USER_WORDS.get())
        for item in (self.passes.pop(0) if self.passes else []):
            yield item


def _conversation(tmp_path):
    """A conversation whose earlier turn ran the module and opened PDF pages
    1-10 of a report."""
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    with open(conv / "activity.jsonl", "w", encoding="utf-8") as fh:
        for rid, name, args, result in (
                ("r0", "call_agent",
                 {"agent_name": "retaining_walls",
                  "method": "earth_pressure_coefficient",
                  "parameters": {"phi_deg": 32, "theory": "rankine"}},
                 {"K": 0.3073, "theory": "rankine"}),
                ("r1", "open_document", {"source": "report.pdf"},
                 {"handle": "doc_abc", "name": "report.pdf", "n_pages": 160}),
                ("r2", "read_document", {"handle": "doc_abc"},
                 {"handle": "doc_abc", "pages_returned": "0-9"})):
            fh.write(json.dumps({"turn": 1, "event": "tool_start",
                                 "name": name, "run_id": rid,
                                 "args": args}) + "\n")
            fh.write(json.dumps({"turn": 1, "event": "tool_end",
                                 "name": name, "run_id": rid,
                                 "result": json.dumps(result)}) + "\n")
    return conv


def _run(agent, conv, question=QUESTION):
    logger = ActivityLogger(str(conv), turn=2)
    items = list(core.stream_turn(
        agent, [{"role": "user", "content": question}], "thread-1",
        callbacks=[logger]))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    events = [json.loads(line) for line in
              (conv / "activity.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    return done["answer"], items, [e for e in events
                                   if e.get("event") == cd.RECORD_EVENT]


def _flagged(event):
    return [r["cited"] for r in event.get("not_looked_up") or []]


def test_off_by_default_the_answer_is_left_and_the_check_recorded(tmp_path):
    conv = _conversation(tmp_path)
    answer, _items, events = _run(_ScriptedAgent(_reply(BAD)), conv)
    assert answer == BAD
    assert len(events) == 1
    ev = events[0]
    assert ev["mode"] == "off" and ev["action"] == "none"
    assert ev["where"] == "answer" and ev["agent"] == "primary"
    assert _flagged(ev) == ["GEC-11", "AASHTO LRFD 3.11.5"]
    assert ev["year_not_checked"][0]["cited"] == "Rankine (1857)"


def test_a_citation_the_user_named_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _conversation(tmp_path)
    answer, _items, events = _run(
        _ScriptedAgent(_reply(BAD)), conv,
        question=QUESTION + " Cite GEC-11 and AASHTO LRFD.")
    assert answer == BAD
    assert events[0]["action"] == "none" and not _flagged(events[0])


def test_note_appends_the_documents_nothing_looked_up(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    conv = _conversation(tmp_path)
    answer, _items, events = _run(_ScriptedAgent(_reply(BAD)), conv)
    assert answer.startswith(BAD)
    assert "Citation check" in answer
    assert "GEC-11, AASHTO LRFD 3.11.5" in answer
    assert events[-1]["action"] == "noted"


def test_retry_holds_the_answer_back_once_and_delivers_the_next(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BAD), _reply(GOOD))
    answer, items, events = _run(agent, conv)
    assert answer == GOOD
    assert len(agent.inputs) == 2
    told = agent.inputs[1]["messages"][-1]
    assert told["role"] == "user"
    assert told["content"].startswith(cd.NOTE_PREFIX)
    assert "- GEC-11" in told["content"]
    assert {"kind": "tool_call", "text": cd.RETRY_STATUS} in items
    assert [e["action"] for e in events] == ["fed_back", "none"]


def test_retry_notes_an_answer_that_still_cites_them(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BAD), _reply(BAD))
    answer, _items, events = _run(agent, conv)
    assert len(agent.inputs) == 2               # told once only
    assert answer.startswith(BAD) and "Citation check" in answer
    assert answer.count(BAD) == 1               # never the two glued
    assert [e["action"] for e in events] == ["fed_back", "noted"]


def test_both_checks_tell_the_model_in_one_note(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "retry")
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BOTH), _reply(GOOD))
    answer, items, _events = _run(agent, conv)
    assert answer == GOOD
    assert len(agent.inputs) == 2               # one retry, not two
    told = agent.inputs[1]["messages"][-1]["content"]
    assert told.startswith(cp.NOTE_PREFIX) and cd.NOTE_PREFIX in told
    assert {"kind": "tool_call", "text": cp.RETRY_STATUS} in items
    assert {"kind": "tool_call", "text": cd.RETRY_STATUS} in items


def test_both_notes_read_the_answer_as_written(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "note")
    conv = _conversation(tmp_path)
    answer, _items, _events = _run(_ScriptedAgent(_reply(BOTH)), conv)
    assert answer.startswith(BOTH)
    assert answer.count("Page check") == 1
    assert answer.count("Citation check") == 1


def test_the_turns_tools_see_the_users_words(tmp_path):
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(GOOD))
    _run(agent, conv)
    assert agent.words == [(QUESTION,)]
    assert cd._USER_WORDS.get() is None         # unbound after the turn


def test_a_turn_with_no_activity_logger_still_answers(monkeypatch):
    monkeypatch.setenv(review_flags.CITED_DOCUMENTS_ENV, "note")
    items = list(core.stream_turn(
        _ScriptedAgent(_reply(BAD)),
        [{"role": "user", "content": QUESTION}], "thread-2"))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    # no record of what was looked up here: nothing is claimed about it
    assert done["answer"] == BAD
