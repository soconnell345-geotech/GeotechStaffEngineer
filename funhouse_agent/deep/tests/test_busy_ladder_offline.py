"""The agents' own model calls ride a busy ladder (BACKLOG TA-20), paced per
key (TA-22), and general-purpose helpers have a budget (TA-22).

Field session 2026-10-14: 3 of 8 rate-limit refusals hit agent calls (two
helpers and the primary); each ended its agent 10-12 s after the call
started, while the 5 that hit vision side calls were asked again and read.

Offline: scripted chat models through real compiled deep agents; the waits
are recorded, not slept.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from funhouse_agent.deep import busy_retry, keep_work, pacing
from funhouse_agent.deep.agent import build_deep_agent


class _Headers(dict):
    def get(self, k, d=None):
        return super().get(k.lower(), d)


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = _Headers({k.lower(): v
                                 for k, v in (headers or {}).items()})


class RateLimitError(Exception):
    """Shaped like the OpenAI SDK's: a status, a response with headers, a
    body."""

    def __init__(self, msg="Too Many Requests", headers=None,
                 code="rate_limit_exceeded"):
        super().__init__(msg)
        self.status_code = 429
        self.response = _Resp(429, headers)
        self.body = {"error": {"code": code, "message": msg}}


@pytest.fixture
def waits(monkeypatch):
    seen = []
    monkeypatch.setattr(busy_retry, "_sleep", seen.append)
    monkeypatch.delenv(busy_retry.TRIES_ENV, raising=False)
    monkeypatch.delenv(pacing.TPM_ENV, raising=False)
    return seen


class Busy(BaseChatModel):
    """Refuses the call number(s) in ``refuse`` with ``error``; otherwise
    answers. Call numbers count across the primary and its helpers."""

    refuse: list = []
    error: object = None
    calls: list = []

    @property
    def _llm_type(self):
        return "busy"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls.append(len(messages))
        n = len(self.calls)
        if n in self.refuse:
            raise (self.error() if callable(self.error) else self.error)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=f"answer after {n} calls"))])


def _invoke(agent):
    return agent.invoke({"messages": [{"role": "user", "content": "q"}]},
                        config={"recursion_limit": 40})


# ---------------------------------------------------------------------------
# TA-20
# ---------------------------------------------------------------------------

def test_a_429_on_the_primary_is_waited_for_and_the_turn_completes(waits):
    """Before: the refusal ended the run. Now: 15 s, 30 s, then the
    answer."""
    model = Busy(refuse=[1, 2], error=RateLimitError, calls=[])
    out = _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert out["messages"][-1].content == "answer after 3 calls"
    assert len(waits) == 2
    assert 15 * 0.85 <= waits[0] <= 15 * 1.15
    assert 30 * 0.85 <= waits[1] <= 30 * 1.15


def test_without_the_ladder_the_same_refusal_ends_the_run(waits,
                                                          monkeypatch):
    monkeypatch.setenv(busy_retry.TRIES_ENV, "1")
    model = Busy(refuse=[1], error=RateLimitError, calls=[])
    with pytest.raises(RateLimitError):
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert waits == []


def test_retry_after_is_honoured_and_capped(waits):
    model = Busy(refuse=[1, 2], calls=[], error=iter([
        RateLimitError(headers={"retry-after": "7"}),
        RateLimitError(headers={"retry-after": "900"})]).__next__)
    _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert waits == [7.0, busy_retry.MAX_RETRY_AFTER_S]


def test_a_spent_budget_is_never_retried(waits):
    spent = RateLimitError("You exceeded your current quota",
                           code="insufficient_quota")
    model = Busy(refuse=[1], error=spent, calls=[])
    with pytest.raises(RateLimitError):
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert waits == [] and len(model.calls) == 1


def test_an_ordinary_error_is_not_retried(waits):
    model = Busy(refuse=[1], error=ValueError("bad request"), calls=[])
    with pytest.raises(ValueError):
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert waits == []


def test_the_ladder_runs_out_and_says_how_long_it_tried(waits):
    model = Busy(refuse=list(range(1, 20)), error=RateLimitError, calls=[])
    with pytest.raises(RateLimitError) as caught:
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert len(model.calls) == busy_retry.DEFAULT_TRIES == 5
    assert len(waits) == 4
    info = caught.value.geotech_busy
    assert info["tries"] == 5 and info["why"] == "the ladder ran out"
    from webapp.core import friendly_turn_error
    assert "The app asked 5 times over" in friendly_turn_error(caught.value)


def test_the_turn_bound_stops_a_dead_endpoint(waits):
    model = Busy(refuse=list(range(1, 20)), error=RateLimitError, calls=[])
    with keep_work.turn_scope(keep_work.TurnScope(busy_limit_s=30)):
        with pytest.raises(RateLimitError) as caught:
            _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert len(waits) == 1                     # 15 s fits, 15 + 30 does not
    assert caught.value.geotech_busy["why"] == \
        "the turn's time for a busy model ran out"


def test_a_helpers_own_call_rides_the_ladder(waits, tmp_path):
    """Before: H3's and H8's refusals ended them ("task error"). Now the
    helper waits and answers."""

    class Delegating(BaseChatModel):
        calls: list = []

        @property
        def _llm_type(self):
            return "delegating"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(1)
            first = next(m.content for m in messages
                         if isinstance(m, HumanMessage))
            if first == "q":                     # the primary
                if not any(isinstance(m, AIMessage) for m in messages):
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content="", tool_calls=[{
                            "name": "task", "id": "t1",
                            "args": {"description": "read logs",
                                     "subagent_type": "general-purpose"}}]))])
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="done"))])
            if len(self.calls) == 2:              # the helper's first call
                raise RateLimitError()
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="the helper's data"))])

    out = _invoke(build_deep_agent(model=Delegating(calls=[]),
                                   reference_mode="off"))
    task = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert task.content == "the helper's data"
    assert len(waits) == 1


def test_each_wait_is_on_the_record(waits, tmp_path):
    from webapp import activity_log as al
    model = Busy(refuse=[1], calls=[],
                 error=lambda: RateLimitError(headers={
                     "retry-after": "3", "x-ratelimit-remaining-tokens": "0"}))
    log = al.ActivityLogger(str(tmp_path), turn=1)
    build_deep_agent(model=model, reference_mode="off").invoke(
        {"messages": [{"role": "user", "content": "q"}]},
        config={"recursion_limit": 40, "callbacks": [log]})
    (rec,) = [r for r in al.load(str(tmp_path)) if r["event"] == "model_busy"]
    assert rec["kind"] == "rate limit" and rec["status"] == 429
    assert rec["retry_after_s"] == 3.0 and rec["wait_s"] == 3.0
    assert rec["attempt"] == 1 and rec["tries"] == 5
    assert rec["headers"]["x-ratelimit-remaining-tokens"] == "0"


# ---------------------------------------------------------------------------
# TA-22: the pacer and the helper budget
# ---------------------------------------------------------------------------

def _clocked(monkeypatch, limit):
    now = [0.0]
    monkeypatch.setattr(pacing, "_sleep",
                        lambda s: now.__setitem__(0, now[0] + s))
    return pacing.TokenPacer(limit, clock=lambda: now[0]), now


def test_the_pacer_holds_a_key_under_its_tokens_a_minute(monkeypatch):
    pacer, now = _clocked(monkeypatch, 1_000)
    _t1, w1 = pacer.acquire(600)
    _t2, w2 = pacer.acquire(600)              # waits for the first to age out
    assert w1 == 0 and 59.9 <= w2 <= 60.1
    assert pacer.used() == 600


def test_the_pacer_counts_what_the_provider_counted(monkeypatch):
    pacer, now = _clocked(monkeypatch, 1_000)
    t1, _ = pacer.acquire(900)                 # the estimate
    pacer.settle(t1, 200)                      # the provider's count
    _t2, w2 = pacer.acquire(700)
    assert w2 == 0 and pacer.used() == 900


def test_a_call_bigger_than_the_limit_goes_when_the_window_is_empty(
        monkeypatch):
    pacer, now = _clocked(monkeypatch, 1_000)
    assert pacer.acquire(5_000)[1] == 0


def test_pacing_is_off_by_default_and_per_key(monkeypatch):
    monkeypatch.delenv(pacing.TPM_ENV, raising=False)

    class M:
        openai_api_base = "https://prompter.test/api/v1"

        def __init__(self, key):
            self.openai_api_key = key

    assert pacing.pacer_for(M("a")) is None
    monkeypatch.setenv(pacing.TPM_ENV, "200000")
    assert pacing.pacer_for(M("a")) is pacing.pacer_for(M("a"))
    assert pacing.pacer_for(M("a")) is not pacing.pacer_for(M("b"))
    # the key is a fingerprint, never the key itself
    fp = pacing.model_key(M("secret-key")).split("|")[1]
    assert len(fp) == 12 and "secret" not in fp


def test_agent_calls_wait_on_the_pacer(waits, monkeypatch):
    """Two one-call turns on a key paced at 1,000 tokens a minute: every
    request here is far over that (the system prompt alone), so the second
    waits for the first to leave the window."""
    now = [0.0]
    slept = []

    def advance(s):
        slept.append(s)
        now[0] += s
    monkeypatch.setattr(pacing, "_now", lambda: now[0])
    monkeypatch.setattr(pacing, "_sleep", advance)
    monkeypatch.setenv(pacing.TPM_ENV, "1000")
    pacing._PACERS.clear()
    agent = build_deep_agent(model=Busy(refuse=[], calls=[]),
                             reference_mode="off")
    _invoke(agent)
    assert slept == []
    _invoke(agent)
    assert 59.0 <= sum(slept) <= 61.0
    pacing._PACERS.clear()


def test_estimate_counts_text_tools_and_images():
    msgs = [HumanMessage(content="x" * 4_000),
            HumanMessage(content=[{"type": "image_url",
                                   "image_url": {"url": "data:,"}},
                                  {"type": "text", "text": "y" * 400}])]
    assert pacing.estimate_tokens(msgs) == 1_000 + 100 + pacing.IMAGE_TOKENS


def test_general_purpose_helpers_have_a_budget(monkeypatch):
    from funhouse_agent.deep import limits
    captured = {}
    import funhouse_agent.deep.agent as agent_mod
    real = agent_mod.create_deep_agent

    def spy(**kw):
        captured.update(kw)
        return real(**kw)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    monkeypatch.delenv(limits.HELPER_MAX_MODEL_CALLS_ENV, raising=False)
    build_deep_agent(model=Busy(refuse=[], calls=[]), reference_mode="off")
    gp = next(s for s in captured["subagents"]
              if s["name"] == "general-purpose")
    budgets = [m for m in gp["middleware"]
               if isinstance(m, limits.ModelCallBudgetMiddleware)]
    assert len(budgets) == 1 and budgets[0].run_limit == 30
    assert budgets[0].final_turn_nudge == limits.HELPER_FINAL_NUDGE
    assert any(isinstance(m, busy_retry.AgentBusyRetry)
               for m in gp["middleware"])
    monkeypatch.setenv(limits.HELPER_MAX_MODEL_CALLS_ENV, "off")
    captured.clear()
    build_deep_agent(model=Busy(refuse=[], calls=[]), reference_mode="off")
    gp = next(s for s in captured["subagents"]
              if s["name"] == "general-purpose")
    assert not any(isinstance(m, limits.ModelCallBudgetMiddleware)
                   for m in gp["middleware"])
