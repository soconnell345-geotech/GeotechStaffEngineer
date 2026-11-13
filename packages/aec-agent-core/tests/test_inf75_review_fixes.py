"""The review of BACKLOG TA-24 / INF-75 (2026-11-13).

1. A gateway's 408 or 504 counts as a timeout (asked again at most once in
   all, client and ladder together) only when the try itself lasted long
   (``vision_engine.SLOW_TRY_S``). A gateway that answers 408 or 504
   QUICKLY is refusing, not timing out, and keeps every try of the busy
   ladder, as before TA-24. Both ladders: the agent's and the vision side
   calls'.
2. A look made inside a helper that ends late is tied to its own tool call,
   not to the helper's ``task`` call that is still open.

Everything here is made up: the errors, the times, the tool names. The
clock is a fake one, moved by the fake model as it "works".
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import busy_retry, vision_engine
from aec_agent_core.deep.vision_engine import LangChainVisionEngine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

QUICK_S = 1.5                                   # a gateway that refuses
SLOW_S = 45.0                                   # a gateway that gave up


def _gateway(status=504):
    exc = Exception(f"made-up: gateway answered {status}")
    exc.status_code = status
    return exc


class _Clock:
    """A fake clock; the fake models move it by the time a try takes."""

    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(busy_retry, "_clock", c)
    monkeypatch.setattr(vision_engine, "_clock", c)
    return c


@pytest.fixture
def no_waits(monkeypatch):
    seen = []
    monkeypatch.setattr(busy_retry, "_sleep", seen.append)
    monkeypatch.setattr(vision_engine, "_sleep", seen.append)
    monkeypatch.delenv(busy_retry.TRIES_ENV, raising=False)
    monkeypatch.delenv(vision_engine.BUSY_TRIES_ENV, raising=False)
    monkeypatch.delenv("GEOTECH_TOKENS_PER_MINUTE", raising=False)
    return seen


def test_the_bound_is_a_named_constant_and_reads_the_try_time():
    assert vision_engine.SLOW_TRY_S == 20.0
    for status in (408, 504):
        exc = _gateway(status)
        assert not vision_engine.is_timeout(exc)               # time unknown
        assert not vision_engine.is_timeout(exc, QUICK_S)
        assert not vision_engine.is_timeout(
            exc, vision_engine.SLOW_TRY_S - 0.1)
        assert vision_engine.is_timeout(exc, vision_engine.SLOW_TRY_S)
        assert busy_retry.timed_out(exc, SLOW_S)
        assert not busy_retry.timed_out(exc, QUICK_S)
    # a client's own timeout is one however long it took
    assert vision_engine.is_timeout(TimeoutError(), QUICK_S)
    # other refusals are never timeouts, however slow
    assert not vision_engine.is_timeout(_gateway(503), SLOW_S)


# ---------------------------------------------------------------------------
# The agent's ladder
# ---------------------------------------------------------------------------

def _ladder_run(clock, model, errors, took_s):
    """One model call through the agent's ladder; each failing try takes
    ``took_s`` on the fake clock. Returns (answer or error, tries)."""
    asked = []

    def handler(request):
        asked.append(1)
        clock.now += took_s
        if len(asked) <= len(errors):
            raise errors[len(asked) - 1]
        return AIMessage(content="the answer")

    request = SimpleNamespace(model=model, messages=[], tools=[],
                              system_message=None)
    try:
        out = busy_retry.AgentBusyRetry().wrap_model_call(request, handler)
    except Exception as exc:  # noqa: BLE001 - the test reads it
        out = exc
    return out, len(asked)


@pytest.mark.parametrize("status", [504, 408])
def test_a_quick_gateway_refusal_gets_every_try_of_the_agent_ladder(
        clock, no_waits, status):
    model = SimpleNamespace(max_retries=1)       # the Prompter default
    out, tries = _ladder_run(clock, model, [_gateway(status)] * 4, QUICK_S)
    assert getattr(out, "content", None) == "the answer"
    assert tries == 5 and len(no_waits) == 4
    # all five refused: the ladder ran out, as on master
    no_waits.clear()
    out, tries = _ladder_run(clock, model, [_gateway(status)] * 5, QUICK_S)
    assert tries == 5 and out.geotech_busy["why"] == "the ladder ran out"


def test_a_slow_504_is_asked_again_at_most_once_in_all_by_the_agent_ladder(
        clock, no_waits):
    # the client asked again itself: the ladder adds no ask
    out, tries = _ladder_run(clock, SimpleNamespace(max_retries=1),
                             [_gateway()] * 5, SLOW_S)
    assert tries == 1 and no_waits == []
    assert "no answer in time" in out.geotech_busy["why"]
    # a client that does not: the ladder asks once more, then stops
    out, tries = _ladder_run(clock, SimpleNamespace(max_retries=0),
                             [_gateway()] * 5, SLOW_S)
    assert tries == 2 and len(no_waits) == 1


def test_the_async_ladder_reads_the_try_time_the_same_way(clock, no_waits,
                                                          monkeypatch):
    waits = []

    async def no_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(busy_retry, "_asleep", no_wait)

    def run(took_s):
        asked = []

        async def handler(request):
            asked.append(1)
            clock.now += took_s
            raise _gateway()

        request = SimpleNamespace(model=SimpleNamespace(max_retries=1),
                                  messages=[], tools=[], system_message=None)
        with pytest.raises(Exception):
            asyncio.run(busy_retry.AgentBusyRetry().awrap_model_call(
                request, handler))
        return len(asked)

    assert run(QUICK_S) == 5
    assert run(SLOW_S) == 1


# ---------------------------------------------------------------------------
# The vision side calls' ladder
# ---------------------------------------------------------------------------

class _Gateway:
    """A vision model answered by a gateway's error after ``took_s``."""

    def __init__(self, clock, took_s, max_retries=None, status=504):
        self.clock, self.took_s, self.status = clock, took_s, status
        self.calls = 0
        if max_retries is not None:
            self.max_retries = max_retries

    def invoke(self, messages):
        self.calls += 1
        self.clock.now += self.took_s
        raise _gateway(self.status)


@pytest.mark.parametrize("limit", [None, "0"])   # side-call limit on / off
def test_a_quick_504_gets_every_try_of_the_vision_ladder(clock, no_waits,
                                                         monkeypatch, limit):
    if limit is None:
        monkeypatch.delenv(vision_engine.TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(vision_engine.TIMEOUT_ENV, limit)
    model = _Gateway(clock, QUICK_S, max_retries=1)
    with pytest.raises(Exception) as caught:
        LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    assert model.calls == vision_engine.busy_tries() == 3
    assert caught.value.vision_tries == 3


@pytest.mark.parametrize("limit", [None, "0"])
def test_a_slow_504_is_asked_again_at_most_once_in_all_by_the_vision_ladder(
        clock, no_waits, monkeypatch, limit):
    if limit is None:
        monkeypatch.delenv(vision_engine.TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(vision_engine.TIMEOUT_ENV, limit)
    # 45 s is under the 90 s side-call limit: the gateway answered first
    asks_again = _Gateway(clock, SLOW_S, max_retries=1)
    with pytest.raises(Exception):
        LangChainVisionEngine(asks_again, detail="").analyze_image(PNG, "x")
    assert asks_again.calls == 1
    plain = _Gateway(clock, SLOW_S)
    with pytest.raises(Exception):
        LangChainVisionEngine(plain, detail="").analyze_image(PNG, "x")
    assert plain.calls == 2


# ---------------------------------------------------------------------------
# The late end of a helper's look
# ---------------------------------------------------------------------------

def test_a_helpers_late_end_is_tied_to_its_own_look(tmp_path):
    """The look's ask timed out while the look tool was open and ended
    after the look, with the helper's ``task`` call still open."""
    lg = ActivityLogger(str(tmp_path), turn=1)
    task, node, look = (uuid.uuid4() for _ in range(3))
    lg.on_tool_start({"name": "task"}, "", run_id=task, parent_run_id=None,
                     inputs={"subagent_type": "helper"})
    lg.on_chain_start({}, {}, run_id=node, parent_run_id=task)
    lg.on_tool_start({"name": "look_at_tile"}, "", run_id=look,
                     parent_run_id=node, inputs={"page": 4})
    lg.on_custom_event(vision_engine.TIMEOUT_EVENT,
                       {"ask_id": "made-up-ask", "kind": "tile",
                        "attempt": 1, "asked_again": True}, run_id=look)
    lg.on_tool_end("not read", run_id=look, parent_run_id=node)
    lg.on_custom_event(vision_engine.LATE_END_EVENT,
                       {"ask_id": "made-up-ask", "outcome": "answered"},
                       run_id=look)
    recs = {r["event"]: r for r in activity_log.load(str(tmp_path))}
    for name in (vision_engine.TIMEOUT_EVENT, vision_engine.LATE_END_EVENT):
        assert recs[name]["tool"] == "look_at_tile", name
        assert recs[name]["tool_run_id"] == str(look), name
