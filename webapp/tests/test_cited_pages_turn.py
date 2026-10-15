"""The turn's answer, checked for pages no tool opened (BACKLOG EV-18).

``core.stream_turn`` checks every page the delivered answer cites against
the conversation's record (``funhouse_agent.cited_pages``), keeps the result
in ``activity.jsonl`` and, per ``GEOTECH_CITED_PAGES``, leaves the answer
(off, the default), appends a note (``note``) or holds the answer back once
and tells the model (``retry``). Driven here with a scripted stream and a
synthetic record of the pages earlier turns opened.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from funhouse_agent import cited_pages as cp
from funhouse_agent import review_flags
from webapp import core
from webapp.activity_log import ActivityLogger

BAD = "Boring 12 is logged on PDF pages 108-119."
GOOD = "Boring 12 is logged on PDF page 10."


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


def _reply(text):
    return [("messages", (AIMessageChunk(content=text),
                          {"langgraph_node": "model"})),
            ("updates", {"model": {"messages": [AIMessage(content=text)]}})]


class _ScriptedAgent:
    """Replays one scripted pass per graph invocation, and notes the record
    folder a tool running in the turn would see."""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.inputs = []
        self.record_dirs = []

    def stream(self, inp, config=None, stream_mode=None):
        self.inputs.append(inp)
        self.record_dirs.append(cp.record_dir())
        for item in (self.passes.pop(0) if self.passes else []):
            yield item


def _conversation(tmp_path):
    """A conversation whose earlier turn opened PDF pages 1-10 of a report."""
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    with open(conv / "activity.jsonl", "w", encoding="utf-8") as fh:
        for rid, name, args, result in (
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


def _run(agent, conv):
    logger = ActivityLogger(str(conv), turn=2)
    items = list(core.stream_turn(
        agent, [{"role": "user", "content": "where is boring 12?"}],
        "thread-1", callbacks=[logger]))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    events = [json.loads(line) for line in
              (conv / "activity.jsonl").read_text(encoding="utf-8")
              .splitlines()]
    return done["answer"], items, [e for e in events
                                   if e.get("event") == cp.RECORD_EVENT]


def test_off_by_default_the_answer_is_left_and_the_check_recorded(tmp_path):
    conv = _conversation(tmp_path)
    answer, _items, events = _run(_ScriptedAgent(_reply(BAD)), conv)
    assert answer == BAD
    assert len(events) == 1
    ev = events[0]
    assert ev["mode"] == "off" and ev["action"] == "none"
    assert ev["agent"] == "primary" and ev["turn"] == 2
    assert [r["cited"] for r in ev["never_opened"]] == ["PDF pages 108-119"]


def test_an_answer_citing_opened_pages_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "note")
    conv = _conversation(tmp_path)
    answer, _items, events = _run(_ScriptedAgent(_reply(GOOD)), conv)
    assert answer == GOOD
    assert events[0]["action"] == "none" and "never_opened" not in events[0]


def test_note_appends_the_pages_nobody_opened(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "note")
    conv = _conversation(tmp_path)
    answer, _items, events = _run(_ScriptedAgent(_reply(BAD)), conv)
    assert answer.startswith(BAD)
    assert "Page check" in answer and "PDF pages 108-119" in answer
    assert events[-1]["action"] == "noted"


def test_retry_holds_the_answer_back_once_and_delivers_the_next(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BAD), _reply(GOOD))
    answer, items, events = _run(agent, conv)
    assert answer == GOOD                       # the reply after the note
    assert len(agent.inputs) == 2
    told = agent.inputs[1]["messages"][-1]
    assert told["role"] == "user"
    assert told["content"].startswith(cp.NOTE_PREFIX)
    assert "PDF pages 108-119" in told["content"]
    assert {"kind": "tool_call", "text": cp.RETRY_STATUS} in items
    assert [e["action"] for e in events] == ["fed_back", "none"]


def test_retry_notes_an_answer_that_still_cites_them(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(BAD), _reply(BAD))
    answer, _items, events = _run(agent, conv)
    assert len(agent.inputs) == 2               # told once only
    assert answer.startswith(BAD) and "Page check" in answer
    assert answer.count(BAD) == 1               # never the two glued
    assert [e["action"] for e in events] == ["fed_back", "noted"]


def test_retry_keeps_the_held_answer_when_the_next_says_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "retry")
    conv = _conversation(tmp_path)
    answer, _items, _events = _run(_ScriptedAgent(_reply(BAD), []), conv)
    assert answer.startswith(BAD) and "Page check" in answer


def test_the_turns_tools_see_the_conversations_record(tmp_path):
    conv = _conversation(tmp_path)
    agent = _ScriptedAgent(_reply(GOOD))
    _run(agent, conv)
    assert agent.record_dirs == [str(conv)]
    assert cp._RECORD_DIR.get() is None         # unbound after the turn


def test_a_turn_with_no_activity_logger_still_answers(tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.CITED_PAGES_ENV, "note")
    items = list(core.stream_turn(
        _ScriptedAgent(_reply(BAD)),
        [{"role": "user", "content": "where is boring 12?"}], "thread-2"))
    done = [i for i in items if i["kind"] == "turn_done"][-1]
    # no record of the pages read here: nothing is claimed about them
    assert done["answer"] == BAD
