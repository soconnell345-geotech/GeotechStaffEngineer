"""A rate limit Foundry NAMES in a 400 is a rate limit on every route
(BACKLOG INF-62; the Funhouse-wiring audit, 2026-11-03, row 1).

Foundry's model service sends a rate limit as an HTTP 400 whose body names
it -- ``errorName: 'LanguageModelService:RateLimitsExceeded'``, or a QoS
exception -- and the AI FDE says its Anthropic proxy does the same. The
default Claude route read only a 429, a 5xx, a timeout or a known class,
so on Foundry a rate limit ended the turn with the raw 400.

Every error here is MADE UP, in the shapes the Anthropic and OpenAI SDKs
raise (``anthropic.BadRequestError`` carries the whole body; the OpenAI
SDK's carries the body's ``error`` object). No model, no network; the
ladder's waits are recorded, not slept.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from funhouse_agent.deep import busy_retry, pacing
from funhouse_agent.deep.vision_engine import busy_kind
from funhouse_agent.error_text import budget_exhausted, rate_limit_named

#: A made-up Foundry error body (no real ids, hosts or users).
FOUNDRY_BODY = {"errorCode": "CUSTOM_CLIENT",
                "errorName": "LanguageModelService:RateLimitsExceeded",
                "errorInstanceId": "made-up-0000", "parameters": {}}
QOS_BODY = dict(FOUNDRY_BODY, errorName="MadeUp:AzurePortalQosException")
OTHER_BODY = dict(FOUNDRY_BODY,
                  errorName="LanguageModelService:InvalidArgument")


def _response(status=400, body=None):
    import httpx
    request = httpx.Request("POST", "https://made-up.example/v1/messages")
    return httpx.Response(status, json=body, request=request)


def anthropic_400(body):
    """As the Anthropic SDK raises a JSON 400: the whole body, and the body
    in the message after the status."""
    anthropic = pytest.importorskip("anthropic")
    return anthropic.BadRequestError(f"Error code: 400 - {body}",
                                     response=_response(400, body), body=body)


def openai_400(body):
    """As the OpenAI SDK raises a JSON 400: ``body`` is the ``error`` object
    where there is one, the whole body where there is not."""
    openai = pytest.importorskip("openai")
    inner = body.get("error", body) if isinstance(body, dict) else body
    return openai.BadRequestError(f"Error code: 400 - {body}",
                                  response=_response(400, body), body=inner)


# ---------------------------------------------------------------------------
# The names, wherever the error carries them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: anthropic_400(FOUNDRY_BODY),
    lambda: anthropic_400(QOS_BODY),
    lambda: anthropic_400({"type": "error", "error": FOUNDRY_BODY}),
    lambda: openai_400(FOUNDRY_BODY),
    lambda: openai_400({"error": FOUNDRY_BODY}),
    lambda: openai_400({"error": QOS_BODY}),
], ids=["anthropic", "anthropic-qos", "anthropic-nested", "openai-top",
        "openai-nested", "openai-qos"])
def test_an_sdk_400_whose_body_names_a_rate_limit_is_one(make):
    error = make()
    assert error.status_code == 400
    assert rate_limit_named(error)
    assert busy_kind(error) == "rate limit"


def test_a_name_written_out_in_a_body_s_message_is_read():
    """A proxy that hands the service's body on as its own message."""
    body = {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": '{"errorCode":"CUSTOM_CLIENT","errorName":'
                   '"LanguageModelService:RateLimitsExceeded"}'}}
    assert busy_kind(anthropic_400(body)) == "rate limit"


def test_a_name_in_the_message_alone_is_read():
    """No body, no attribute: the Foundry SDK's "ErrorName: '...'"."""
    error = Exception(
        "400 Client Error: Bad Request for url: https://made-up.example/api/"
        "MADE_UP. ErrorCode: 'CUSTOM_CLIENT'. "
        "ErrorName: 'LanguageModelService:RateLimitsExceeded'.")
    assert busy_kind(error) == "rate limit"


def test_a_name_on_the_response_s_json_is_read():
    """An HTTP library's error: no body of its own, the JSON on its
    response."""
    error = Exception("400 Client Error: Bad Request (made up)")
    error.response = SimpleNamespace(status_code=400, headers={},
                                     json=lambda: FOUNDRY_BODY)
    assert busy_kind(error) == "rate limit"


def test_a_qos_exception_by_its_error_name_is_one():
    """The ``error_name`` read before named ``ratelimit`` only, so a QoS
    exception was not busy."""
    error = Exception("400 Client Error (made up)")
    error.error_name = "MadeUp:HubQosException"
    assert busy_kind(error) == "rate limit"


def test_a_wrapper_raised_from_one_is_one():
    try:
        try:
            raise anthropic_400(FOUNDRY_BODY)
        except Exception as inner:
            raise RuntimeError("the model call failed (made up)") from inner
    except RuntimeError as outer:
        assert busy_kind(outer) == "rate limit"


# ---------------------------------------------------------------------------
# Other 400s stay errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: anthropic_400(OTHER_BODY),
    lambda: openai_400({"error": OTHER_BODY}),
    lambda: anthropic_400({"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "messages.0: made-up schema problem"}}),
    # The word in someone's text, not as the error's name.
    lambda: anthropic_400({"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "field 'note': 'RateLimitsExceeded' is not allowed "
                   "(made up)"}}),
    lambda: openai_400({"error": {"message": "made-up bad image",
                                  "type": "invalid_request_error",
                                  "code": "invalid_image"}}),
], ids=["anthropic-invalid-argument", "openai-invalid-argument",
        "anthropic-schema", "anthropic-echo", "openai-invalid-image"])
def test_any_other_400_is_not_busy(make):
    error = make()
    assert not rate_limit_named(error)
    assert busy_kind(error) is None


def test_reading_an_odd_error_never_raises():
    class _Odd(Exception):
        @property
        def body(self):
            raise ValueError("made up")

    assert rate_limit_named(_Odd("Error code: 400")) is False


# ---------------------------------------------------------------------------
# A spent allowance is still one plain line, never retried
# ---------------------------------------------------------------------------

def test_a_named_rate_limit_that_says_the_allowance_is_spent_is_not_busy():
    error = anthropic_400({"type": "error", "error": {
        "type": "invalid_request_error",
        "errorName": FOUNDRY_BODY["errorName"],
        "message": "You have reached your specified API usage limits. You "
                   "will regain access on 2099-01-01 at 00:00 UTC."}})
    assert budget_exhausted(error) is not None
    assert busy_kind(error) is None
    from webapp.core import friendly_turn_error
    text = friendly_turn_error(error)
    assert text.startswith("Your AI budget is used up (until 2099-01-01")
    assert "rate limit" not in text and "Error code" not in text


# ---------------------------------------------------------------------------
# The ladder waits and asks again; the turn's text says rate limit
# ---------------------------------------------------------------------------

@pytest.fixture
def waits(monkeypatch):
    seen = []
    monkeypatch.setattr(busy_retry, "_sleep", seen.append)
    monkeypatch.delenv(busy_retry.TRIES_ENV, raising=False)
    monkeypatch.delenv(pacing.TPM_ENV, raising=False)
    return seen


def _busy_model(refuse, error):
    """A chat model that raises ``error()`` on the call numbers in
    ``refuse`` and otherwise answers "answer after N calls"."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Busy(BaseChatModel):
        calls: list = []

        @property
        def _llm_type(self):
            return "busy"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(len(messages))
            if len(self.calls) in refuse:
                raise error()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=f"answer after {len(self.calls)} calls"))])

    return Busy(calls=[])


def _invoke(agent):
    return agent.invoke({"messages": [{"role": "user", "content": "q"}]},
                        config={"recursion_limit": 40})


@pytest.mark.parametrize("make", [lambda: anthropic_400(FOUNDRY_BODY),
                                  lambda: openai_400({"error": QOS_BODY})],
                         ids=["anthropic", "openai-qos"])
def test_the_agent_waits_out_a_foundry_400_rate_limit(waits, make):
    """Before: the turn stopped on the raw 400. Now: 15 s, 30 s, then the
    answer, as for a 429."""
    from funhouse_agent.deep.agent import build_deep_agent
    model = _busy_model([1, 2], lambda: make())
    out = _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert out["messages"][-1].content == "answer after 3 calls"
    assert len(waits) == 2
    assert 15 * 0.85 <= waits[0] <= 15 * 1.15


def test_the_agent_raises_any_other_400_at_once(waits):
    from funhouse_agent.deep.agent import build_deep_agent
    anthropic = pytest.importorskip("anthropic")
    model = _busy_model([1], lambda: anthropic_400(OTHER_BODY))
    with pytest.raises(anthropic.BadRequestError):
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    assert waits == []


def test_a_foundry_400_the_ladder_could_not_outlast_reads_as_a_rate_limit(
        waits):
    from funhouse_agent.deep.agent import build_deep_agent
    from webapp.core import friendly_turn_error
    anthropic = pytest.importorskip("anthropic")
    model = _busy_model(list(range(1, 20)),
                        lambda: anthropic_400(FOUNDRY_BODY))
    with pytest.raises(anthropic.BadRequestError) as caught:
        _invoke(build_deep_agent(model=model, reference_mode="off"))
    text = friendly_turn_error(caught.value)
    assert text.startswith("The AI model's rate limit was reached")
    assert "The app asked 5 times over" in text


def test_the_turn_s_text_for_any_other_400_is_the_error():
    from webapp.core import friendly_turn_error
    text = friendly_turn_error(anthropic_400(OTHER_BODY))
    assert text.startswith("BadRequestError: Error code: 400")
    assert "rate limit" not in text
