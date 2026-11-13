"""A vision side call that times out is on the run record (BACKLOG INF-75).

Foundry brief 7 (2026-11-12): a side call with no answer inside its limit
was given up and asked once more with only a log warning. The 84 records
held 44 ``model_start`` events with no ``model_end`` and nothing that said
why; a call given up kept its thread and its vision slot, unseen.

Now each ask given up is a ``side_call_timeout`` event (the limit, the
look's kind, page and tile, which ask, whether it was asked again), each
ask's ``model_start`` carries the same ``ask_id``, and a call that ends
after it was given up writes ``side_call_late_end`` with how long it held
its slot.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import vision_engine
from aec_agent_core.deep.vision_engine import (
    LangChainVisionEngine, VisionCallTimeout, side_call,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _StallingChat(BaseChatModel):
    """A chat model that stalls (until released) on the calls listed."""

    stall_on: tuple = (1,)
    fail_late: bool = False
    state: Any = None

    @property
    def _llm_type(self) -> str:
        return "stalling-fake"

    def _generate(self, messages, stop: Optional[List[str]] = None,
                  run_manager=None, **kwargs) -> ChatResult:
        with self.state["lock"]:
            self.state["calls"] += 1
            n = self.state["calls"]
        if n in self.stall_on:
            self.state["release"].wait(10)
            if self.fail_late:
                raise ConnectionError("dropped after the limit")
            text = f"late answer {n}"
        else:
            text = f"answer {n}"
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content=text))])


def _model(stall_on=(1,), fail_late=False) -> _StallingChat:
    return _StallingChat(stall_on=tuple(stall_on), fail_late=fail_late,
                         state={"calls": 0, "lock": threading.Lock(),
                                "release": threading.Event()})


@pytest.fixture
def short_limit(monkeypatch):
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "4")


def _look_tool(model):
    @tool
    def look_at_tile(page: int) -> str:
        """Look at one tile of a page."""
        with side_call(kind="tile", page=page, tile="r1c2"):
            try:
                return LangChainVisionEngine(model, detail="").analyze_image(
                    PNG, "what is here?")
            except VisionCallTimeout as exc:
                return f"not read: {exc}"
    return look_at_tile


def _events(conv, name=None):
    recs = activity_log.load(str(conv))
    return [r for r in recs if name is None or r.get("event") == name]


def _wait_for(conv, name, n=1, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        got = _events(conv, name)
        if len(got) >= n:
            return got
        time.sleep(0.02)
    return _events(conv, name)


def test_a_timeout_that_is_asked_again_is_an_event(short_limit, tmp_path):
    model = _model(stall_on=(1,))
    logger = ActivityLogger(str(tmp_path), turn=1)
    try:
        out = _look_tool(model).invoke({"page": 4},
                                       config={"callbacks": [logger]})
    finally:
        model.state["release"].set()
    assert out == "answer 2"
    events = _events(tmp_path, "side_call_timeout")
    assert len(events) == 1
    ev = events[0]
    assert ev["limit_s"] == 0.2
    assert (ev["kind"], ev["page"], ev["tile"]) == ("tile", 4, "r1c2")
    assert ev["ask"] == 1 and ev["attempt"] == 1
    assert ev["asked_again"] is True
    assert ev["tool"] == "look_at_tile" and ev["tool_run_id"]
    assert ev["turn"] == 1


def test_the_timeout_names_the_model_start_it_left_open(short_limit,
                                                        tmp_path):
    """The record held starts with no end and nothing to match them to."""
    model = _model(stall_on=(1,))
    logger = ActivityLogger(str(tmp_path), turn=1)
    try:
        _look_tool(model).invoke({"page": 7}, config={"callbacks": [logger]})
        ev = _events(tmp_path, "side_call_timeout")[0]
        starts = _events(tmp_path, "model_start")
        ends = _events(tmp_path, "model_end")
        assert [s["vision"]["ask"] for s in starts] == [1, 2]
        left_open = [s for s in starts
                     if s["vision"]["ask_id"] == ev["ask_id"]]
        assert len(left_open) == 1 and left_open[0]["vision"]["ask"] == 1
        # the abandoned ask has no end yet; the second ask has its own
        assert [e["vision"]["ask_id"] for e in ends] == \
            [starts[1]["vision"]["ask_id"]]
    finally:
        model.state["release"].set()


def test_two_timeouts_say_the_image_was_not_asked_again(short_limit,
                                                        tmp_path):
    model = _model(stall_on=(1, 2, 3))
    logger = ActivityLogger(str(tmp_path), turn=1)
    try:
        out = _look_tool(model).invoke({"page": 2},
                                       config={"callbacks": [logger]})
    finally:
        model.state["release"].set()
    assert out.startswith("not read") and "NOT read" in out
    events = _events(tmp_path, "side_call_timeout")
    assert [(e["ask"], e["attempt"], e["asked_again"]) for e in events] == \
        [(1, 1, True), (2, 2, False)]
    assert events[0]["ask_id"] != events[1]["ask_id"]


def test_an_abandoned_call_that_ends_late_is_recorded(short_limit, tmp_path):
    """A call given up keeps its slot until it ends: the record now says
    when it did, and for how long the slot was held."""
    model = _model(stall_on=(1,))
    logger = ActivityLogger(str(tmp_path), turn=1)
    _look_tool(model).invoke({"page": 4}, config={"callbacks": [logger]})
    assert _events(tmp_path, "side_call_late_end") == []
    time.sleep(0.15)
    model.state["release"].set()
    late = _wait_for(tmp_path, "side_call_late_end")
    assert len(late) == 1
    timeout = _events(tmp_path, "side_call_timeout")[0]
    assert late[0]["ask_id"] == timeout["ask_id"]
    assert late[0]["outcome"] == "answered"
    assert late[0]["held_s"] >= 0.3          # past the 0.2 s limit
    assert late[0]["limit_s"] == 0.2
    assert (late[0]["kind"], late[0]["page"]) == ("tile", 4)
    assert late[0]["tool"] == "look_at_tile"


def test_an_abandoned_call_that_fails_late_says_so(short_limit, tmp_path):
    model = _model(stall_on=(1,), fail_late=True)
    logger = ActivityLogger(str(tmp_path), turn=1)
    _look_tool(model).invoke({"page": 4}, config={"callbacks": [logger]})
    model.state["release"].set()
    late = _wait_for(tmp_path, "side_call_late_end")
    assert len(late) == 1
    assert late[0]["outcome"] == "error"
    assert late[0]["error"] == "ConnectionError"


def test_a_call_that_answers_in_time_writes_no_event(short_limit, tmp_path):
    model = _model(stall_on=())
    logger = ActivityLogger(str(tmp_path), turn=1)
    assert _look_tool(model).invoke(
        {"page": 4}, config={"callbacks": [logger]}) == "answer 1"
    time.sleep(0.1)
    assert _events(tmp_path, "side_call_timeout") == []
    assert _events(tmp_path, "side_call_late_end") == []
    start = _events(tmp_path, "model_start")[0]
    assert start["vision"]["ask"] == 1 and start["vision"]["ask_id"]


def test_outside_a_run_a_timeout_still_only_raises(short_limit):
    """A library caller has no record: the event is dropped, never an
    error of its own."""
    model = _model(stall_on=(1, 2))
    try:
        with pytest.raises(VisionCallTimeout, match="NOT read"):
            LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    finally:
        model.state["release"].set()


def test_the_events_are_app_events_of_the_activity_log():
    assert vision_engine.TIMEOUT_EVENT in activity_log.APP_EVENTS
    assert vision_engine.LATE_END_EVENT in activity_log.APP_EVENTS
