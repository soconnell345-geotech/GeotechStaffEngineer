"""A retry made inside a model call is on the run record (BACKLOG INF-75).

Foundry brief 7 (2026-11-12): one helper's model call took 904 s for 324
tokens out, half of its run, and nothing in the record said why. The
Foundry engine asks a timed-out or dropped call again inside the call
(``palantir_sdk_engine.call_with_retries``) and wrote only a log line.

Now each retry is an ``engine_retry`` app event: the kind, the attempt, the
wait, the error's class and its HTTP status where it has one, written from
inside the model call under that call's own run id. The retry policy is
unchanged.

Everything here is made up: the model name, the SDK handle and its errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from aec_agent_core import activity_log
from aec_agent_core import palantir_sdk_engine as eng
from aec_agent_core.activity_log import ActivityLogger


class _Role:
    SYSTEM, USER, ASSISTANT, TOOL = "system", "user", "assistant", "tool"


class _ChatMessage:
    def __init__(self, role, content=None, tool_calls=None,
                 tool_call_id=None):
        self.role, self.content = role, content


class _Request:
    def __init__(self, messages, **kwargs):
        self.messages = messages


class _ReadTimeout(Exception):
    """Named as the HTTP client's read timeout is."""


_ReadTimeout.__name__ = "ReadTimeout"


def _busy_503():
    exc = Exception("made-up: service unavailable")
    exc.response = SimpleNamespace(status_code=503)
    return exc


class _Handle:
    """A made-up SDK model handle that fails the listed ways, then answers."""

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def create_chat_completion(self, request, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="the answer",
                                        tool_calls=None))],
            usage=None, model="MADE_UP_MODEL")


@pytest.fixture
def fake_sdk(monkeypatch):
    base = SimpleNamespace(ChatMessage=_ChatMessage, ChatMessageRole=_Role)
    v3 = SimpleNamespace(GptChatCompletionRequest=_Request)
    monkeypatch.setattr(eng, "_sdk", lambda: (None, base, v3))
    # no real waiting: every backoff draws 0 s
    monkeypatch.setattr(eng.random, "uniform", lambda a, b: 0.0)


def _model(failures):
    handle = _Handle(failures)
    model = eng.PalantirSdkChatModel.from_handles(
        "MADE_UP_MODEL", text_model=handle, vision_model=object(),
        route="chat", read_timeout_s=None)
    return model, handle


def _events(conv, name):
    return [r for r in activity_log.load(str(conv)) if r.get("event") == name]


def test_each_retry_inside_a_model_call_is_an_event(fake_sdk, tmp_path):
    model, handle = _model([_ReadTimeout("no answer"),
                            ConnectionResetError("reset"), _busy_503()])
    logger = ActivityLogger(str(tmp_path), turn=2)
    out = model.invoke([HumanMessage(content="hello")],
                       config={"callbacks": [logger]})
    assert out.content == "the answer" and handle.calls == 4
    events = _events(tmp_path, "engine_retry")
    assert [(e["kind"], e["attempt"], e["error"]) for e in events] == [
        ("timeout", 1, "ReadTimeout"),
        ("connection", 2, "ConnectionResetError"),
        ("http_503", 3, "Exception")]
    assert all(e["engine"] == "foundry" and e["turn"] == 2 for e in events)
    assert all("wait_s" in e and "waited_s" in e for e in events)
    assert "status" not in events[0] and events[2]["status"] == 503
    # never the error's own text
    assert all("no answer" not in str(e) and "reset" not in str(e)
               for e in events)


def test_the_event_names_the_model_call_it_happened_inside(fake_sdk,
                                                           tmp_path):
    """The 904 s call: its model_end can now be read beside its retries."""
    model, _handle = _model([_ReadTimeout("no answer")])
    logger = ActivityLogger(str(tmp_path), turn=1)
    model.invoke([HumanMessage(content="hello")],
                 config={"callbacks": [logger]})
    start = _events(tmp_path, "model_start")[0]
    end = _events(tmp_path, "model_end")[0]
    retry = _events(tmp_path, "engine_retry")[0]
    assert retry["run_id"] == start["run_id"] == end["run_id"]
    order = [r["event"] for r in activity_log.load(str(tmp_path))
             if r["event"] in ("model_start", "engine_retry", "model_end")]
    assert order == ["model_start", "engine_retry", "model_end"]


def test_the_event_arrives_from_inside_an_agents_step(fake_sdk, tmp_path):
    """As an agent's graph calls the model: from a step of a chain whose
    callbacks the turn was given."""
    model, _handle = _model([ConnectionResetError("reset")])
    logger = ActivityLogger(str(tmp_path), turn=1)
    step = RunnableLambda(lambda q: model.invoke([HumanMessage(content=q)]))
    assert step.invoke("hello", config={"callbacks": [logger]}).content == \
        "the answer"
    events = _events(tmp_path, "engine_retry")
    assert len(events) == 1 and events[0]["kind"] == "connection"
    assert events[0]["agent"] == "primary"


def test_only_a_retry_writes_an_event_and_the_policy_is_as_it_was(
        fake_sdk, tmp_path, monkeypatch):
    assert eng.RETRY_EVENT in activity_log.APP_EVENTS
    logger = ActivityLogger(str(tmp_path), turn=1)
    # a call that answers first time: no event
    model, handle = _model([])
    model.invoke([HumanMessage(content="hello")],
                 config={"callbacks": [logger]})
    assert handle.calls == 1
    assert _events(tmp_path, "engine_retry") == []
    # a failure that asking again would not fix: raised at once, no event
    model, handle = _model([ValueError("made-up: context too long")])
    with pytest.raises(ValueError):
        model.invoke([HumanMessage(content="hello")],
                     config={"callbacks": [logger]})
    assert handle.calls == 1
    assert _events(tmp_path, "engine_retry") == []
    # outside any run (a library caller): the retry happens, nothing is
    # written and nothing fails
    seen = []

    def once():
        seen.append(1)
        if len(seen) == 1:
            raise ConnectionResetError("reset")
        return "ok"

    assert eng.call_with_retries(once, sleep=lambda s: None) == "ok"
    # a failure that outlasts the wait budget is still raised
    waits, calls = [], []
    monkeypatch.setattr(eng.random, "uniform", lambda a, b: b)

    def always_drops():
        calls.append(1)
        raise ConnectionResetError("reset")

    with pytest.raises(ConnectionResetError):
        eng.call_with_retries(always_drops, total_s=5.0, sleep=waits.append)
    assert waits == [2.0] and len(calls) == 2     # 2 s, then 4 s is over 5
