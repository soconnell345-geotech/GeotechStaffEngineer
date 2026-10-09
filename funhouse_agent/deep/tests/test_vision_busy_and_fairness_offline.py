"""A busy model is asked again; testers share the vision slots fairly; an
answer cut off at the output limit is known (live smoke wave 2a B7, B13;
wave 2b C5).

* B7 -- F27's overloaded tile was asked again only because its error JSON
  held the word "details", which the ``detail`` fallback matched by accident;
  a 429 or a 5xx was not retried at all. Several testers share one Prompter
  key on Tiny Apps next week.
* B13 -- one 8-slot queue, first come first served: one tester's 30-call set
  read held every slot while another's one-page question waited behind all
  of it.
* C5 -- a side call stopped at its output limit came back as if complete.

Fakes only: no model, no network.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from funhouse_agent import _fileio
from funhouse_agent.deep import vision_engine
from funhouse_agent.deep.vision_engine import (
    FairSlots, LangChainVisionEngine, busy_kind, describe_error,
    refused_detail, was_cut_off,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class APIStatusError(Exception):
    """Shaped like the OpenAI / Anthropic SDK errors: a status code, a body
    and a response with headers."""

    def __init__(self, message, status_code, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(status_code=status_code,
                                        headers=headers or {})


class _Scripted:
    """Raises (or answers) from a script, one entry per ``invoke``; records
    whether each call carried an image ``detail``."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.detail_sent = []

    def invoke(self, messages):
        self.calls += 1
        self.detail_sent.append(
            "detail" in messages[0].content[0]["image_url"])
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step if isinstance(step, AIMessage) else AIMessage(content=step)


@pytest.fixture
def waits(monkeypatch):
    """The waits between tries, recorded instead of slept."""
    monkeypatch.delenv(vision_engine.BUSY_TRIES_ENV, raising=False)
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    seen = []
    monkeypatch.setattr(vision_engine, "_sleep", seen.append)
    return seen


# -- B7: busy is retried with backoff, honouring retry-after -----------------

def test_a_429_is_asked_again_after_the_retry_after(waits):
    limited = APIStatusError("Too Many Requests", 429,
                             headers={"retry-after": "1.5"})
    model = _Scripted([limited, limited, "ok"])
    out = LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert out == "ok" and model.calls == 3
    assert waits == [1.5, 1.5]


def test_without_retry_after_the_wait_backs_off(waits):
    busy = APIStatusError("Service Unavailable", 503)
    model = _Scripted([busy, busy, "ok"])
    assert LangChainVisionEngine(model, detail="").analyze_image(PNG, "x") \
        == "ok"
    assert len(waits) == 2 and 1.0 <= waits[0] < waits[1] <= 6.0


def test_a_busy_model_that_stays_busy_is_reported_in_plain_words(waits):
    overloaded = APIStatusError(
        "Error code: 529 - {'type': 'error', 'error': {'details': None, "
        "'type': 'overloaded_error', 'message': 'Overloaded'}}", 529,
        body={"type": "error", "error": {"details": None,
                                         "type": "overloaded_error"}})
    model = _Scripted([overloaded] * 5)
    with pytest.raises(APIStatusError) as caught:
        LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert model.calls == vision_engine.DEFAULT_BUSY_TRIES == 3
    text = describe_error(caught.value)
    assert "busy (overloaded)" in text and "after 3 tries" in text
    assert "NOT read" in text and "529" not in text


def test_f27s_overload_is_retried_as_busy_not_as_a_refused_detail(waits):
    """The overload's body holds 'details': it is asked again WITH the image
    detail, as a busy error, not stripped of it as a refused detail."""
    overloaded = APIStatusError(
        "{'details': None, 'type': 'overloaded_error'}", 529,
        body={"error": {"details": None, "type": "overloaded_error"}})
    model = _Scripted([overloaded, "ok"])
    out = LangChainVisionEngine(model, detail="original").analyze_image(
        PNG, "x")
    assert out == "ok" and model.detail_sent == [True, True]
    assert refused_detail(overloaded) is False


def test_the_detail_fallback_needs_a_request_error_that_names_detail(waits):
    refused = APIStatusError(
        "Invalid value: 'original'. param: messages[0].content[0]"
        ".image_url.detail", 400)
    model = _Scripted([refused, "ok"])
    assert LangChainVisionEngine(model, detail="original").analyze_image(
        PNG, "x") == "ok"
    assert model.detail_sent == [True, False] and waits == []
    # A server error that mentions "detail" is busy, not a refusal.
    assert refused_detail(APIStatusError("detail lost", 502)) is False


def test_an_error_asking_again_would_not_fix_is_not_retried(waits):
    model = _Scripted([ValueError("prompt too long")])
    with pytest.raises(ValueError):
        LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert model.calls == 1 and waits == []


def test_busy_kind_reads_type_status_and_body_never_the_message():
    assert busy_kind(APIStatusError("x", 429)) == "rate limit"
    assert busy_kind(APIStatusError("x", 529)) == "overloaded"
    assert busy_kind(APIStatusError("x", 500)) == "server error 500"
    assert busy_kind(type("RateLimitError", (Exception,), {})()) \
        == "rate limit"
    assert busy_kind(ConnectionResetError()) == "connection"
    assert busy_kind(ValueError("the server is overloaded, 429")) is None
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = APIStatusError("x", 503)
    assert busy_kind(wrapped) == "server error 503"


def test_the_waits_hold_no_slot(monkeypatch):
    """While one call waits to ask again, another call can use the slot."""
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "1")
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    other_done = threading.Event()

    def wait_for_other(_s):
        assert other_done.wait(5), "the other call never got the slot"

    monkeypatch.setattr(vision_engine, "_sleep", wait_for_other)
    model = _Scripted([APIStatusError("busy", 503), "late"])
    first = threading.Thread(target=lambda: LangChainVisionEngine(
        model, detail="").analyze_image(PNG, "x"))
    first.start()
    while model.calls < 1:
        time.sleep(0.01)
    other = LangChainVisionEngine(_Scripted(["other"]), detail="")
    assert other.analyze_image(PNG, "y") == "other"
    other_done.set()
    first.join(5)
    assert model.calls == 2


# -- B13: slots are shared fairly between conversations ----------------------

def test_a_freed_slot_goes_to_the_conversation_with_fewest_in_flight():
    slots = FairSlots(2)
    order, ready = [], []
    slots.acquire("alice")
    slots.acquire("alice")                     # alice holds both slots

    def wait(conv, tag):
        ev = threading.Event()
        ready.append(ev)

        def run():
            ev.set()
            slots.acquire(conv)
            order.append(tag)
        t = threading.Thread(target=run)
        t.start()
        ev.wait(5)
        time.sleep(0.05)                       # queued, in this order
        return t

    threads = [wait("alice", "alice-3"), wait("alice", "alice-4"),
               wait("bob", "bob-1")]
    slots.release("alice")                     # bob (0 in flight) is next
    time.sleep(0.1)
    assert order == ["bob-1"]
    slots.release("alice")                     # now alice 1, bob 1: oldest
    time.sleep(0.1)
    assert order == ["bob-1", "alice-3"]
    slots.release("bob")
    for t in threads:
        t.join(5)
    assert order == ["bob-1", "alice-3", "alice-4"]


def test_a_conversation_alone_uses_every_slot():
    slots = FairSlots(3)
    for _ in range(3):
        slots.acquire("alice")
    assert slots.in_flight("alice") == 3 == slots.in_flight()
    for _ in range(3):
        slots.release("alice")
    assert slots.in_flight() == 0


def test_a_per_conversation_cap_when_one_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "4")
    monkeypatch.setenv(vision_engine.PER_CONVERSATION_ENV, "1")
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    class _Counts:
        def invoke(self, messages):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.02)
            with lock:
                state["now"] -= 1
            return AIMessage(content="ok")

    engine = LangChainVisionEngine(_Counts(), detail="")

    def call():
        with _fileio.working_dir_bound(str(tmp_path / "one")):
            engine.analyze_image(PNG, "x")
    threads = [threading.Thread(target=call) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert state["peak"] == 1


def test_the_conversation_is_the_bound_working_folder(tmp_path):
    with _fileio.working_dir_bound(str(tmp_path / "a")):
        a = vision_engine.conversation_key()
    with _fileio.working_dir_bound(str(tmp_path / "b")):
        b = vision_engine.conversation_key()
    assert a and b and a != b


# -- C5: an answer cut off at the output limit -------------------------------

@pytest.mark.parametrize("meta", [
    {"finish_reason": "length"},                     # OpenAI chat
    {"stop_reason": "max_tokens"},                   # Anthropic
    {"status": "incomplete",
     "incomplete_details": {"reason": "max_output_tokens"}},  # Responses
])
def test_a_cut_off_answer_is_flagged(waits, meta):
    model = _Scripted([AIMessage(content="half a read", response_metadata=meta)])
    out = LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert out == "half a read" and out.cut_off is True


def test_a_complete_answer_is_not(waits):
    done = AIMessage(content="all", response_metadata={"finish_reason": "stop"})
    assert was_cut_off(done) is False
    out = LangChainVisionEngine(_Scripted([done]), detail="").analyze_image(
        PNG, "x")
    assert getattr(out, "cut_off", False) is False
