"""Offline tests for GPT-5 / reasoning-model parameter compatibility in the
PrompterChatModel bridge (no API / no network)."""

from types import SimpleNamespace

import pytest

from funhouse_agent.deep.databricks_bridge import (
    PrompterChatModel,
    _adjust_request_for_param_error,
)
from langchain_core.messages import HumanMessage


# --- _adjust_request_for_param_error ---------------------------------------

def test_adjust_drops_temperature_on_temp_error():
    req = {"model": "funhouse-gpt-high", "messages": [], "temperature": 0.0}
    out = _adjust_request_for_param_error(
        req, "Unsupported value: 'temperature' does not support 0.0")
    assert out is not None
    assert "temperature" not in out
    assert out["model"] == "funhouse-gpt-high"


def test_adjust_renames_max_tokens():
    req = {"model": "m", "messages": [], "max_tokens": 1024}
    out = _adjust_request_for_param_error(
        req, "Use 'max_completion_tokens' instead of 'max_tokens'")
    assert out is not None
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 1024


def test_adjust_returns_none_for_unrelated_error():
    req = {"model": "m", "messages": [], "temperature": 0.0}
    assert _adjust_request_for_param_error(req, "rate limit exceeded") is None


# --- _build_request temperature omission ------------------------------------

def _model(temperature=0.0):
    fake = SimpleNamespace(client=SimpleNamespace(), chat_model="x")
    return PrompterChatModel(prompter=fake, model="funhouse-gpt-high",
                             temperature=temperature)


def test_build_request_omits_temperature_when_none():
    req = _model(temperature=None)._build_request([HumanMessage(content="hi")])
    assert "temperature" not in req
    assert req["model"] == "funhouse-gpt-high"


def test_build_request_includes_temperature_when_set():
    req = _model(temperature=0.0)._build_request([HumanMessage(content="hi")])
    assert req["temperature"] == 0.0


# --- _create_with_param_fallback retry --------------------------------------

class _Client:
    """Fake OpenAI client whose create() fails the first call then succeeds."""

    def __init__(self, fail_message=None, fail_times=1):
        self.calls = []
        self._fail_message = fail_message
        self._fail_times = fail_times
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_message and len(self.calls) <= self._fail_times:
            raise RuntimeError(self._fail_message)
        return SimpleNamespace(choices=[], usage=None)


def test_fallback_retries_after_temperature_error():
    client = _Client(fail_message="'temperature' is not supported", fail_times=1)
    m = PrompterChatModel(
        prompter=SimpleNamespace(client=client, chat_model="funhouse-gpt-high"),
        model="funhouse-gpt-high",
    )
    m._create_with_param_fallback({"model": "funhouse-gpt-high",
                                   "messages": [], "temperature": 0.0})
    assert len(client.calls) == 2                  # retried once
    assert "temperature" not in client.calls[1]    # retry dropped it


def test_fallback_propagates_unrelated_error():
    client = _Client(fail_message="rate limit", fail_times=1)
    m = PrompterChatModel(
        prompter=SimpleNamespace(client=client, chat_model="x"),
        model="x",
    )
    with pytest.raises(RuntimeError, match="rate limit"):
        m._create_with_param_fallback({"model": "x", "messages": [],
                                       "temperature": 0.0})
    assert len(client.calls) == 1                  # no retry
