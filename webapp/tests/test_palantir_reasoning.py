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


class _SummaryConfig:
    """The SDK's summary level enum (Foundry, setup check
    ``sdk_reasoning_types``, brief 5)."""
    AUTO, CONCISE, DETAILED, UNKNOWN = "AUTO", "CONCISE", "DETAILED", "UNKNOWN"


class _ReasoningEffort:
    MINIMAL, LOW, MEDIUM, HIGH = "MINIMAL", "LOW", "MEDIUM", "HIGH"


class _ReasoningConfig:
    """The SDK's request field: ``ReasoningConfig(effort, summary)``."""

    def __init__(self, effort=None, summary=None):
        self.effort = effort
        self.summary = summary


class _ReasoningSummary:
    """An OUTPUT class on Foundry (a summary's text): no AUTO level."""

    def __init__(self, text=None):
        self.text = text


#: The ten names Foundry's SDK carries, each in its real role. ``Reasoning``
#: is the result's reasoning ITEM and would happily be built with any
#: keyword, which is exactly what a first-name-that-exists lookup fell for.
_REAL_NAMES = {
    "Reasoning": _rec("Reasoning"),
    "ReasoningConfig": _ReasoningConfig,
    "ReasoningContent": _rec("ReasoningContent"),
    "ReasoningContentVisitor": _rec("ReasoningContentVisitor"),
    "ReasoningEffort": _ReasoningEffort,
    "ReasoningSummary": _ReasoningSummary,
    "ReasoningSummaryTextDeltaChunk": _rec("ReasoningSummaryTextDeltaChunk"),
    "ReasoningSummaryVisitor": _rec("ReasoningSummaryVisitor"),
    "ReasoningTextDeltaChunk": _rec("ReasoningTextDeltaChunk"),
    "SummaryConfig": _SummaryConfig,
}


def _with_reasoning_types(monkeypatch, names=None):
    """The fake Responses module carrying Foundry's REAL reasoning names."""
    import sys
    r = sys.modules[RESP_MOD]
    for name, obj in (names or _REAL_NAMES).items():
        monkeypatch.setattr(r, name, obj, raising=False)
    return r


def _with_old_guessed_names(monkeypatch):
    """The names the first version guessed (before brief 5): ``Reasoning``
    as the request class and ``ReasoningSummary`` as the level enum."""
    return _with_reasoning_types(monkeypatch, {
        "Reasoning": _rec("Reasoning"),
        "ReasoningSummary": SimpleNamespace(
            AUTO="AUTO", CONCISE="CONCISE", DETAILED="DETAILED")})


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
    # Foundry's real request class with its real level, not the output
    # classes that share the prefix (brief 5: nothing was ever requested).
    assert isinstance(request.reasoning, _ReasoningConfig)
    assert request.reasoning.summary is _SummaryConfig.AUTO
    assert out.response_metadata["reasoning_summary"] == "requested"
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


def test_the_first_existing_name_is_not_trusted(monkeypatch):
    """The brief-5 failure, replayed: with every real name present, the
    lookup that took the first name that EXISTS picked ``Reasoning`` and
    ``ReasoningSummary`` (no AUTO) and asked for nothing."""
    import sys
    from webapp import palantir_sdk_engine as eng
    _install_fake_responses(monkeypatch, [])
    r = _with_reasoning_types(monkeypatch)
    got = eng._reasoning_request(sys.modules[RESP_MOD], "auto")
    assert type(got) is _ReasoningConfig and got.summary == "AUTO"
    assert eng._reasoning_request(r, "detailed").summary == "DETAILED"
    assert eng._reasoning_request(r, None) is None


def test_the_old_guessed_names_still_work(monkeypatch):
    """An SDK that has only the names first guessed is still asked."""
    sent = _install_fake_responses(monkeypatch, [_responses_reply()])
    _with_old_guessed_names(monkeypatch)
    _model().invoke([HumanMessage(content="hi")])
    assert sent[0].args[1].open_ai_responses.reasoning.summary == "AUTO"


def test_a_level_no_request_class_takes_sends_nothing(monkeypatch):
    """``SummaryConfig`` exists but nothing can be built with it: no
    reasoning field is sent, and the call goes through."""
    class _Strict:
        def __init__(self, effort=None):
            self.effort = effort

    sent = _install_fake_responses(monkeypatch, [_responses_reply()])
    _with_reasoning_types(monkeypatch, {"SummaryConfig": _SummaryConfig,
                                        "ReasoningConfig": _Strict})
    out = _model().invoke([HumanMessage(content="hi")])
    assert not hasattr(sent[0].args[1].open_ai_responses, "reasoning")
    assert out.response_metadata["reasoning_summary"] == "not requested"


def test_a_requested_summary_that_is_not_read_is_logged_once(monkeypatch,
                                                             caplog):
    import logging
    sent = _install_fake_responses(monkeypatch, [_responses_reply("a"),
                                                 _responses_reply("b")])
    _with_reasoning_types(monkeypatch)
    m = _model()
    with caplog.at_level(logging.INFO, logger="webapp.palantir_sdk_engine"):
        m.invoke([HumanMessage(content="hi")])
        m.invoke([HumanMessage(content="again")])
    lines = [r.getMessage() for r in caplog.records
             if "none was read" in r.getMessage()]
    assert len(lines) == 1 and "output_message" in lines[0]
    assert len(sent) == 2


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
