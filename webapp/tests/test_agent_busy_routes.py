"""The busy ladder on the two real engines an agent's calls go through
(BACKLOG TA-20): the Prompter key route (``webapp.tinyapps_engine``: a real
``ChatOpenAI`` and OpenAI client over a faked HTTP transport) and the
Funhouse bridge (``PrompterChatModel`` over a faked Prompter client).

Field session 2026-10-14: the agents' own calls got only the client's quick
retries (about 10-12 s) and a refusal ended the agent; the primary's ended
a 51-minute turn.
"""

from __future__ import annotations

import pytest

from funhouse_agent.deep import busy_retry, pacing

pytest.importorskip("langchain_openai")
from webapp.engine_config import sdk_http_package  # noqa: E402

# the HTTP package openai's clients are built on (httpx; anthropic is on
# httpx2): transports and responses must come from the same one
httpx = sdk_http_package(pytest.importorskip("openai"))


@pytest.fixture
def waits(monkeypatch):
    seen = []
    monkeypatch.setattr(busy_retry, "_sleep", seen.append)
    monkeypatch.delenv(busy_retry.TRIES_ENV, raising=False)
    monkeypatch.delenv(pacing.TPM_ENV, raising=False)
    return seen


OK = {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
      "choices": [{"index": 0, "finish_reason": "stop",
                   "message": {"role": "assistant",
                               "content": "the answer"}}],
      "usage": {"prompt_tokens": 120, "completion_tokens": 8,
                "total_tokens": 128}}


def _rate_limited():
    return httpx.Response(429, json={"error": {
        "message": "Rate limit is exceeded. Try again later.",
        "type": "requests", "code": "rate_limit_exceeded"}})


def _spent():
    return httpx.Response(429, json={"error": {
        "message": "You exceeded your current quota, please check your plan "
                   "and billing details.",
        "type": "insufficient_quota", "code": "insufficient_quota"}})


def _prompter_model(monkeypatch, handler):
    """The app's own Prompter model, its HTTP answered by ``handler``."""
    import webapp.tinyapps_engine as te
    real_init = httpx.Client.__init__

    def mocked_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        real_init(self, *a, **kw)

    # The constructor, not the class: a package first imported while
    # ``httpx.Client`` is swapped (anthropic, by the agent build) subclasses
    # the swap for good, and every later test's Claude client got this
    # test's canned answers (the Foundry proxy tests, run after this file).
    monkeypatch.setattr(httpx.Client, "__init__", mocked_init)
    # the client's own quick retries off, so each refusal reaches the ladder
    monkeypatch.setenv(te.MAX_RETRIES_ENV, "0")
    return te.build_chat_model(prompter=te.PrompterSettings(
        url="https://prompter.test/api/v1/chat/completions", model="m",
        api_key="k"))


def _ask(model):
    from funhouse_agent.deep.agent import build_deep_agent
    agent = build_deep_agent(model=model, reference_mode="off")
    return agent.invoke({"messages": [{"role": "user", "content": "q"}]},
                        config={"recursion_limit": 30})


def test_prompter_route_a_429_is_retried_on_the_ladder(monkeypatch, waits):
    sent = []

    def handler(request):
        sent.append(request)
        return _rate_limited() if len(sent) <= 2 else httpx.Response(
            200, json=OK)

    out = _ask(_prompter_model(monkeypatch, handler))
    assert out["messages"][-1].content == "the answer"
    assert len(sent) == 3 and len(waits) == 2
    assert 12.75 <= waits[0] <= 17.25 and 25.5 <= waits[1] <= 34.5


def test_prompter_route_a_spent_budget_is_asked_once(monkeypatch, waits):
    sent = []

    def handler(request):
        sent.append(request)
        return _spent()

    import openai
    with pytest.raises(openai.RateLimitError):
        _ask(_prompter_model(monkeypatch, handler))
    assert len(sent) == 1 and waits == []


class _BusyCreate:
    """A Prompter client's ``chat.completions.create`` that refuses the
    first ``n`` calls with an SDK-shaped 429."""

    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __call__(self, **request):
        self.calls += 1
        if self.calls <= self.n:
            import openai
            resp = httpx.Response(
                429, request=httpx.Request("POST", "https://p/x"),
                headers={"retry-after": "4"},
                json={"error": {"code": "rate_limit_exceeded",
                                "message": "Too Many Requests"}})
            raise openai.RateLimitError("Too Many Requests", response=resp,
                                        body=resp.json())
        from types import SimpleNamespace as NS
        return NS(model="funhouse-gpt-high",
                  usage=NS(prompt_tokens=50, completion_tokens=5,
                           total_tokens=55),
                  choices=[NS(finish_reason="stop", message=NS(
                      content="bridged answer", tool_calls=[]))])


def test_funhouse_bridge_a_429_is_retried_on_the_ladder(waits):
    from types import SimpleNamespace as NS
    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    create = _BusyCreate(2)
    prompter = NS(client=NS(chat=NS(completions=NS(create=create))),
                  chat_model="funhouse-gpt-high")
    out = _ask(PrompterChatModel(prompter=prompter))
    assert out["messages"][-1].content == "bridged answer"
    assert create.calls == 3 and waits == [4.0, 4.0]   # retry-after honoured
