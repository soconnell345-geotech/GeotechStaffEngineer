"""Connection diagnostics on a slow day (BACKLOG TA-41).

The battery runs in the page while the tester waits. Each live check now
waits a short time for the model, all of them together a bounded time, and
the battery stops at the first live check that fails, saying which one and
that the rest were not run. A run where everything passes in time shows
what it showed before. Fake models only.
"""

from __future__ import annotations

import contextvars
import threading
import time
from types import SimpleNamespace

import pytest

import aec_document_review.diagnostics as diag
import aec_agent_core.engine_config as engine_config
from aec_agent_core.engine_config import EngineResolution

LIVE = ("plain request (invoke)", "streaming request",
        "tool-calling request", "vision model")
_CALLER = contextvars.ContextVar("made_up_caller", default="nobody")


class _Reply:
    def __init__(self, content="OK", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {}
        self.usage_metadata = {}


class _Model:
    """A healthy fake model that counts its calls."""

    def __init__(self):
        self.calls = []

    def invoke(self, _msg):
        self.calls.append("invoke")
        return _Reply("OK")

    def stream(self, _msg):
        self.calls.append("stream")
        yield _Reply("O")
        yield _Reply("K")

    def bind_tools(self, _tools):
        self.calls.append("bind_tools")

        class _Bound:
            def invoke(self, _msg):
                return _Reply("", tool_calls=[{"name": "add_numbers",
                                               "args": {"a": 2, "b": 3}}])
        return _Bound()


class _Hangs(_Model):
    """A request the service never answers (until the test lets it go)."""

    def __init__(self, release):
        super().__init__()
        self.release = release

    def invoke(self, _msg):
        self.calls.append("invoke")
        self.release.wait(10)
        return _Reply("late")


class _Refuses(_Model):
    def invoke(self, _msg):
        self.calls.append("invoke")
        raise RuntimeError("made-up refusal from the gateway")


class _SaysCaller(_Model):
    def invoke(self, _msg):
        return _Reply(_CALLER.get())


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    from aec_agent_core import vision_probe
    monkeypatch.setenv(vision_probe.PROBE_ENV, "0")   # no probe calls here


@pytest.fixture
def release():
    ev = threading.Event()
    yield ev
    ev.set()                                          # let any worker end


def _patch_engine(monkeypatch, model):
    monkeypatch.setattr(
        engine_config, "resolve_engine",
        lambda model_id=None: EngineResolution(
            model, "foundry", "ri.fake", "Using the made-up proxy."))


def _by_name(checks, name):
    return next(c for c in checks if c["name"] == name)


def test_a_request_that_never_answers_holds_the_page_seconds_not_minutes(
        monkeypatch, release):
    monkeypatch.setattr(diag, "CALL_LIMIT_S", 0.3)
    model = _Hangs(release)
    _patch_engine(monkeypatch, model)
    t0 = time.monotonic()
    checks = diag.run_diagnostics("ri.fake")
    assert time.monotonic() - t0 < 5
    first = _by_name(checks, "plain request (invoke)")
    assert first["status"] == diag.FAIL
    assert first["detail"].startswith("no answer within 0.3 s")
    for name in LIVE[1:]:
        c = _by_name(checks, name)
        assert c["status"] == diag.SKIP
        assert c["detail"] == "not run: 'plain request (invoke)' failed first"
    assert model.calls == ["invoke"]                  # nothing else was asked
    report = diag.format_report(checks)
    assert "failed first" in report and "no answer within" in report


def test_the_first_failure_stops_the_rest_and_keeps_its_text(monkeypatch):
    model = _Refuses()
    _patch_engine(monkeypatch, model)
    checks = diag.run_diagnostics("ri.fake")
    first = _by_name(checks, "plain request (invoke)")
    assert first["status"] == diag.FAIL
    assert first["detail"] == "RuntimeError: made-up refusal from the gateway"
    assert [_by_name(checks, n)["status"] for n in LIVE[1:]] == [diag.SKIP] * 3
    assert model.calls == ["invoke"]


def test_a_passing_run_shows_what_it_showed_before(monkeypatch):
    model = _Model()
    _patch_engine(monkeypatch, model)
    checks = diag.run_diagnostics("ri.fake")
    before = [diag._invoke_check(model), diag._stream_check(model),
              diag._tool_check(model), diag._vision_check(model)]
    assert checks[-4:] == before
    assert [c["name"] for c in checks[-4:]] == list(LIVE)


def test_the_call_keeps_the_callers_context(monkeypatch):
    """On Tiny Apps the engine is the signed-in tester's: whatever the
    request reads from the caller's context reaches the worker thread."""
    _patch_engine(monkeypatch, _SaysCaller())
    token = _CALLER.set("made-up-tester")
    try:
        checks = diag.run_diagnostics("ri.fake")
    finally:
        _CALLER.reset(token)
    assert _by_name(checks, "plain request (invoke)")["detail"] == \
        "reply: made-up-tester"


def test_the_budget_bounds_the_whole_battery(release):
    def quick():
        return diag._check("first", diag.PASS, "ok")

    def slow():
        release.wait(10)
        return diag._check("second", diag.PASS, "late")

    steps = [("first", quick, 5.0, ""), ("second", slow, 5.0, ""),
             ("third", quick, 5.0, "")]
    t0 = time.monotonic()
    out = diag._live_checks(steps, 0.3)
    assert time.monotonic() - t0 < 3                  # the budget, not 5 s
    assert [c["status"] for c in out] == [diag.PASS, diag.FAIL, diag.SKIP]
    assert out[1]["detail"].startswith("no answer within")
    assert out[2]["detail"] == "not run: 'second' failed first"
    spent = diag._live_checks(steps, 0.0)
    assert [c["status"] for c in spent] == [diag.SKIP] * 3
    assert "were spent" in spent[0]["detail"]


def test_a_helper_model_that_never_answers_is_named(release):
    helper = SimpleNamespace(own_model=True, model=_Hangs(release),
                             name="reader", label="made-up-model",
                             route="made-up route")
    steps = diag._helper_steps([helper])
    assert [s[0] for s in steps] == [
        "plain request (invoke) (reader helper)",
        "tool-calling request (reader helper)"]
    out = diag._live_checks([(n, f, 0.3, note) for n, f, _l, note in steps],
                            5.0)
    assert out[0]["status"] == diag.FAIL
    assert out[0]["detail"].endswith(
        " | reader helper: made-up-model | made-up route")
    assert out[1]["status"] == diag.SKIP
