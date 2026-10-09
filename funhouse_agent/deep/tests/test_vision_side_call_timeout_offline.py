"""A vision side call has its own time limit and one more ask (brief 5, T).

Foundry brief 5 (2026-10-08): side vision calls (page looks, tiles, zooms,
contact-sheet reads) went through the same engine as the primary model and so
inherited its 900 s read timeout, set for long reasoning. One stalled tile
held a turn for 907 s, another for 343 s (903 s in brief 4), while the p99
side call took 74 s. Now a side call is given up after
``GEOTECH_VISION_CALL_TIMEOUT_S`` (default 180 s), asked once more, and if
that stalls too the image is reported as not read. The primary agent's own
calls never pass through the vision engine.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage

from funhouse_agent.deep import vision_engine
from funhouse_agent.deep.vision_engine import (
    LangChainVisionEngine, VisionCallTimeout,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _Stalls:
    """Stalls (until released) on the calls listed in ``stall_on``
    (1-based), answers at once otherwise."""

    def __init__(self, stall_on=(1,)):
        self.stall_on = set(stall_on)
        self.calls = 0
        self.release = threading.Event()
        self.lock = threading.Lock()

    def invoke(self, messages):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n in self.stall_on:
            self.release.wait(10)
            return AIMessage(content=f"late answer {n}")
        return AIMessage(content=f"answer {n}")


@pytest.fixture
def short_limit(monkeypatch):
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0.2")
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "4")


def test_the_default_limit_is_near_the_p99_not_the_engines_900s(monkeypatch):
    monkeypatch.delenv(vision_engine.TIMEOUT_ENV, raising=False)
    assert vision_engine.call_timeout_s() == vision_engine.DEFAULT_TIMEOUT_S
    assert 74 < vision_engine.DEFAULT_TIMEOUT_S < 900
    assert vision_engine.TIMEOUT_RETRIES == 1
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    assert vision_engine.call_timeout_s() is None
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "nonsense")
    assert vision_engine.call_timeout_s() == vision_engine.DEFAULT_TIMEOUT_S


def test_a_stalled_call_is_asked_once_more(short_limit):
    model = _Stalls(stall_on=(1,))
    t0 = time.monotonic()
    try:
        out = LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    finally:
        model.release.set()
    assert out == "answer 2"
    assert model.calls == 2
    assert time.monotonic() - t0 < 5


def test_two_stalls_give_up_and_say_the_image_was_not_read(short_limit):
    model = _Stalls(stall_on=(1, 2, 3))
    t0 = time.monotonic()
    try:
        with pytest.raises(VisionCallTimeout, match="NOT read"):
            LangChainVisionEngine(model, detail="").analyze_image(PNG, "x")
    finally:
        model.release.set()
    assert model.calls == 2                 # one retry, not three asks
    assert time.monotonic() - t0 < 5


def test_a_timeout_is_not_mistaken_for_a_refused_detail(short_limit):
    """The detail fallback asks again without ``detail`` on an error that
    names it; a timeout is not one, so it is not asked a third time."""
    model = _Stalls(stall_on=(1, 2, 3, 4))
    try:
        with pytest.raises(VisionCallTimeout):
            LangChainVisionEngine(model, detail="original").analyze_image(
                PNG, "x")
    finally:
        model.release.set()
    assert model.calls == 2


def test_the_call_runs_in_the_callers_context(short_limit):
    """The activity log and the token count reach the call through the
    caller's context; the worker thread runs in a copy of it."""
    var = contextvars.ContextVar("run", default="none")
    seen = []

    class _Reads:
        def invoke(self, messages):
            seen.append(var.get())
            return AIMessage(content="ok")

    var.set("this run")
    assert LangChainVisionEngine(_Reads(), detail="").analyze_image(
        PNG, "x") == "ok"
    assert seen == ["this run"]


def test_waiting_for_a_slot_is_not_counted(monkeypatch):
    """Queueing behind other calls is not a stall: with one slot and a call
    holding it longer than the limit, the next call still gets its answer."""
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0.3")
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "1")

    class _Slow:
        def invoke(self, messages):
            time.sleep(0.2)
            return AIMessage(content="ok")

    engine = LangChainVisionEngine(_Slow(), detail="")
    out = []
    threads = [threading.Thread(
        target=lambda: out.append(engine.analyze_image(PNG, "x")))
        for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert out == ["ok", "ok", "ok"]


def test_no_limit_calls_straight_through(monkeypatch):
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    seen = []

    class _Here:
        def invoke(self, messages):
            seen.append(threading.current_thread().name)
            return AIMessage(content="ok")

    LangChainVisionEngine(_Here(), detail="").analyze_image(PNG, "x")
    assert seen == [threading.current_thread().name]


def test_a_stalled_page_look_returns_an_error_not_a_hang(short_limit,
                                                         monkeypatch):
    """Through the tool: the stalled look comes back as a JSON error within
    the limit, so the turn goes on."""
    fitz = pytest.importorskip("fitz")
    from funhouse_agent import vision_probe
    from funhouse_agent.vision_tools import dispatch_extended_tool

    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")
    vision_probe.clear_cache()

    doc = fitz.open()
    doc.new_page(width=200, height=200).insert_text((20, 40), "B-1")
    pdf = doc.tobytes()
    model = _Stalls(stall_on=(1, 2, 3, 4, 5, 6, 7, 8))
    t0 = time.monotonic()
    try:
        raw = dispatch_extended_tool(
            "analyze_pdf_page",
            {"attachment_key": "a.pdf", "page": 0, "prompt": "x",
             "tiles": "off"},
            LangChainVisionEngine(model, detail=""), {"a.pdf": pdf})
    finally:
        model.release.set()
    out = json.loads(raw)
    assert "error" in out and "NOT read" in out["error"]
    assert time.monotonic() - t0 < 5
