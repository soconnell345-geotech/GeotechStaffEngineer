"""The Responses route asks for a reasoning summary and keeps it.

Foundry brief 4 (2026-10-07): in 49 agent runs the primary agent wrote no
text on any of its ~300 tool-calling steps and no reasoning summary was
requested, so the record could say WHAT each run did but never WHY. The
package's own Responses engine now asks for a summary where the SDK can say
so, puts it on the reply where the activity log reads it, and asks no more
if the service refuses it. Offline: the SDK is faked in ``sys.modules`` the
way ``test_palantir_sdk_engine`` fakes it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

from webapp.tests.test_palantir_sdk_engine import (  # noqa: E402
    _Handle, _install_fake_responses, _rec, _responses_reply,
)

RESP_MOD = ("language_model_service_api."
            "languagemodelservice_api_completion_v3_responses")


@pytest.fixture(autouse=True)
def _default_level(monkeypatch):
    import webapp.palantir_sdk_engine as eng
    monkeypatch.delenv(eng.REASONING_SUMMARY_ENV, raising=False)
    monkeypatch.setattr(eng.time, "sleep", lambda s: None)


def _with_reasoning_types(monkeypatch):
    import sys
    r = sys.modules[RESP_MOD]
    monkeypatch.setattr(r, "Reasoning", _rec("Reasoning"), raising=False)
    monkeypatch.setattr(r, "ReasoningSummary", SimpleNamespace(
        AUTO="AUTO", CONCISE="CONCISE", DETAILED="DETAILED"), raising=False)
    return r


def _reply_with_summary(text="OK", summary="I compared the tile with the "
                                            "whole page before zooming."):
    reply = _responses_reply(text)
    reply.open_ai_responses.output.insert(0, SimpleNamespace(
        output_message=None, function_tool_call=None,
        reasoning=SimpleNamespace(summary=[
            SimpleNamespace(text=summary),
            SimpleNamespace(summary_text=SimpleNamespace(
                text="Then I chose the tag."))])))
    return reply


def _model():
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    return PalantirSdkChatModel.from_handles("GPT_5_4", vision_model=_Handle())


def test_a_summary_is_asked_for_and_kept(monkeypatch):
    sent = _install_fake_responses(monkeypatch, [_reply_with_summary()])
    _with_reasoning_types(monkeypatch)
    out = _model().invoke([HumanMessage(content="hi")])
    request = sent[0].args[1].open_ai_responses
    assert request.reasoning.summary == "AUTO"
    assert out.content == "OK"
    assert out.additional_kwargs["reasoning"] == (
        "I compared the tile with the whole page before zooming.\n\n"
        "Then I chose the tag.")
    # ... and the activity log writes it on model_end
    from webapp.activity_log import _model_output
    text, reasoning = _model_output(LLMResult(
        generations=[[ChatGeneration(message=out)]]))
    assert text == "OK" and "chose the tag" in reasoning


def test_the_level_can_be_changed_or_switched_off(monkeypatch):
    import webapp.palantir_sdk_engine as eng
    sent = _install_fake_responses(monkeypatch, [_responses_reply(),
                                                 _responses_reply()])
    _with_reasoning_types(monkeypatch)
    monkeypatch.setenv(eng.REASONING_SUMMARY_ENV, "detailed")
    _model().invoke([HumanMessage(content="hi")])
    assert sent[0].args[1].open_ai_responses.reasoning.summary == "DETAILED"
    monkeypatch.setenv(eng.REASONING_SUMMARY_ENV, "off")
    _model().invoke([HumanMessage(content="hi")])
    assert not hasattr(sent[1].args[1].open_ai_responses, "reasoning")


def test_an_sdk_that_cannot_ask_sends_nothing(monkeypatch):
    sent = _install_fake_responses(monkeypatch, [_responses_reply()])
    out = _model().invoke([HumanMessage(content="hi")])   # no Reasoning type
    assert not hasattr(sent[0].args[1].open_ai_responses, "reasoning")
    assert "reasoning" not in out.additional_kwargs


def test_a_refused_setting_is_dropped_once_and_for_good(monkeypatch):
    bad = type("ConjureHTTPError", (Exception,), {})(
        "400 INVALID_ARGUMENT: unknown field reasoning")
    bad.error_name = "Default:InvalidArgument"
    sent = _install_fake_responses(monkeypatch, [bad, _responses_reply("a"),
                                                 _responses_reply("b")])
    _with_reasoning_types(monkeypatch)
    m = _model()
    assert m.invoke([HumanMessage(content="hi")]).content == "a"
    assert hasattr(sent[0].args[1].open_ai_responses, "reasoning")
    assert not hasattr(sent[1].args[1].open_ai_responses, "reasoning")
    assert m.invoke([HumanMessage(content="again")]).content == "b"
    assert not hasattr(sent[2].args[1].open_ai_responses, "reasoning")
    assert len(sent) == 3


def test_an_unrelated_bad_request_is_still_raised(monkeypatch):
    bad = type("ConjureHTTPError", (Exception,), {})("context too long")
    sent = _install_fake_responses(monkeypatch, [bad, _responses_reply()])
    _with_reasoning_types(monkeypatch)
    with pytest.raises(Exception, match="context too long"):
        _model().invoke([HumanMessage(content="hi")])
    assert len(sent) == 1
