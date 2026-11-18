"""Connection diagnostics on a slow day (BACKLOG TA-41).

The battery runs in the page while the tester waits. Each live check now
waits a short time for the model, always its whole limit once started; no
check is started once the battery's time is used up (those read as not run,
never failed), and the battery stops at the first live check that fails,
saying which one and that the rest were not run. A run where everything
passes shows what it showed before. Fake models only; a slow day is played
on the battery's own clock.
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


def test_a_started_check_fails_only_by_its_own_whole_limit(release):
    def quick():
        return diag._check("first", diag.PASS, "ok")

    def slow():
        release.wait(10)
        return diag._check("second", diag.PASS, "late")

    steps = [("first", quick, 5.0, ""), ("second", slow, 0.3, ""),
             ("third", quick, 5.0, "")]
    # the battery's time (0.05 s) runs out while 'second' waits: it keeps
    # its own 0.3 s all the same, and fails by that
    out = diag._live_checks(steps, 0.05)
    assert [c["status"] for c in out] == [diag.PASS, diag.FAIL, diag.SKIP]
    assert out[1]["detail"].startswith("no answer within 0.3 s")
    assert out[2]["detail"] == "not run: 'second' failed first"
    spent = diag._live_checks(steps, 0.0)
    assert [c["status"] for c in spent] == [diag.SKIP] * 3
    assert all(c["detail"] == diag.TIME_USED_UP for c in spent)


class _Slow(_Model):
    """A healthy model on a slow day: each request takes ``s`` seconds of
    the battery's clock."""

    def __init__(self, clock, s):
        super().__init__()
        self.clock, self.s = clock, s

    def invoke(self, msg):
        self.clock[0] += self.s
        return super().invoke(msg)

    def stream(self, msg):
        self.clock[0] += self.s
        yield from super().stream(msg)

    def bind_tools(self, tools):
        bound, clock, s = super().bind_tools(tools), self.clock, self.s

        class _Bound:
            def invoke(self, msg):
                clock[0] += s
                return bound.invoke(msg)
        return _Bound()


def _slow_day(monkeypatch, request_s, probe_s=23.0, n_helpers=2):
    """The battery on a played clock: every request takes ``request_s``,
    the vision probe ``probe_s``, with ``n_helpers`` helper models."""
    clock = [0.0]
    monkeypatch.setattr(diag, "_clock", lambda: clock[0])
    _patch_engine(monkeypatch, _Slow(clock, request_s))

    def probe(_model):
        clock[0] += probe_s
        return diag._check("vision model", diag.PASS, "made-up profile")

    monkeypatch.setattr(diag, "_vision_check", probe)
    helpers = [SimpleNamespace(own_model=True, model=_Slow(clock, request_s),
                               name=name, label="made-up-model", route="")
               for name in ("reader", "writer")[:n_helpers]]
    monkeypatch.setattr(diag, "_helper_models",
                        lambda *a, **k: (None, helpers))
    return clock


HELPER_CHECKS = ("plain request (invoke) (reader helper)",
                 "tool-calling request (reader helper)",
                 "plain request (invoke) (writer helper)",
                 "tool-calling request (writer helper)")


def test_an_ordinary_slow_day_with_two_helper_models_passes(monkeypatch):
    """The reviewer's case: small calls of 6 s, a 23 s vision probe and two
    helper models (about 65 s in all) pass, every check."""
    clock = _slow_day(monkeypatch, 6.0)
    checks = diag.run_diagnostics("ri.fake")
    for name in LIVE + HELPER_CHECKS:
        assert _by_name(checks, name)["status"] == diag.PASS, name
    assert clock[0] == pytest.approx(65.0)


def test_checks_left_when_the_time_is_used_up_are_not_run_never_failed(
        monkeypatch):
    _slow_day(monkeypatch, 100.0)
    checks = diag.run_diagnostics("ri.fake")
    live = [_by_name(checks, n) for n in LIVE + HELPER_CHECKS]
    assert [c["status"] for c in live[:2]] == [diag.PASS] * 2
    assert [c["status"] for c in live[2:]] == [diag.SKIP] * 6
    assert all(c["detail"] == diag.TIME_USED_UP for c in live[2:])
    assert not any(c["status"] == diag.FAIL for c in live)
    assert "slow" not in diag.format_report(live)


def test_a_check_started_late_still_gets_its_whole_limit(monkeypatch):
    clock = _slow_day(monkeypatch, 59.0, n_helpers=0)
    given = []
    real = diag._limited

    def spy(name, fn, limit_s):
        given.append((name, limit_s, clock[0]))
        return real(name, fn, limit_s)

    monkeypatch.setattr(diag, "_limited", spy)
    checks = diag.run_diagnostics("ri.fake")
    assert [(n, s) for n, s, _t in given] == [
        ("plain request (invoke)", diag.CALL_LIMIT_S),
        ("streaming request", diag.CALL_LIMIT_S),
        ("tool-calling request", diag.CALL_LIMIT_S),
        ("vision model", diag.VISION_LIMIT_S)]
    assert given[-1][2] == pytest.approx(177.0)   # 3 s of the battery left
    assert all(_by_name(checks, n)["status"] == diag.PASS for n in LIVE)


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
