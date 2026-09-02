"""Offline tests for PrompterChatModel token streaming (NO API key, NO network).

Covers the ``_stream`` design constraints:

  (a) plain text deltas — concatenated streamed text matches, and the summed
      AIMessageChunk carries the full text;
  (b) a tool call split across 3 delta chunks (id+name in the first, argument
      fragments after) merges into one correct tool_call;
  (c) the final usage-only chunk surfaces LangChain ``usage_metadata``;
  (d) the ``collect_usage`` landmine — the WRAPPED ``prompter.client`` create
      is never called with ``stream=True``; the clean/unwrapped client is used
      (both the ``__wrapped__`` fallback and the http_client/base_url
      clean-OpenAI construction path);
  (e) streaming disabled (the default) — ``stream()`` falls back to
      ``_generate`` and yields its result as a single chunk;
  (f) stream-setup exceptions fall back to ``_generate`` without raising;
  (g) ``prompter.logger.meter_log`` is called best-effort with token counts
      (and a raising meter_log never breaks the turn).

Driven by fake prompters whose stream is an iterator of SimpleNamespace
chunks shaped like OpenAI ``ChatCompletionChunk`` objects (mirrors the fake
conventions in ``test_deep_phase2_offline.py``).
"""

import json
from types import SimpleNamespace

import pytest

from langchain_core.messages import AIMessageChunk, HumanMessage

import funhouse_agent.deep.databricks_bridge as databricks_bridge
from funhouse_agent.deep.databricks_bridge import PrompterChatModel


# ===========================================================================
# Fakes
# ===========================================================================

def _delta_chunk(content=None, tool_calls=None, finish_reason=None,
                 usage=None, model="fake-gpt"):
    """One OpenAI-shaped streaming chunk (SimpleNamespace)."""
    if content is None and tool_calls is None and finish_reason is None:
        choices = []  # the usage-only final chunk has empty choices
    else:
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason,
                                   index=0)]
    return SimpleNamespace(choices=choices, usage=usage, model=model)


def _tc_delta(index=0, id=None, name=None, arguments=None):
    """One OpenAI tool-call delta fragment."""
    return SimpleNamespace(
        index=index, id=id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


_USAGE = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                         total_tokens=15)


def _make_openai_response(content="non-streamed answer"):
    """A canned NON-streaming response for the _generate fallback path."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=_USAGE, model="fake-gpt")


class _RecordingLogger:
    def __init__(self, raise_on_meter=False):
        self.meter_calls = []
        self.raise_on_meter = raise_on_meter

    def meter_log(self, **kwargs):
        if self.raise_on_meter:
            raise RuntimeError("metering backend down")
        self.meter_calls.append(kwargs)


class _FakeStreamingPrompter:
    """Fake PrompterAPI reproducing the SDK's wrapped-client landmine.

    ``client.chat.completions.create`` is the WRAPPED method: like
    ``wrap_openai_method`` it injects ``collect_usage=True`` when ``stream``
    is truthy, which (like the real OpenAI SDK) raises TypeError. Its
    ``__wrapped__`` attribute is the clean create, which returns the fake
    stream iterator. No ``http_client``/``base_url`` attrs → exercises the
    ``__wrapped__`` fallback path of ``_streaming_create``.
    """

    def __init__(self, stream_chunks=None, response=None,
                 chat_model="fake-gpt", stream_error=None, logger=None):
        self.chat_model = chat_model
        self.logger = logger if logger is not None else _RecordingLogger()
        self.wrapped_calls = []
        self.clean_calls = []
        self._stream_chunks = stream_chunks or []
        self._response = response
        self._stream_error = stream_error

        def clean_create(**kwargs):
            self.clean_calls.append(kwargs)
            if kwargs.get("stream"):
                if self._stream_error is not None:
                    raise self._stream_error
                return iter(list(self._stream_chunks))
            return self._response

        def wrapped_create(**kwargs):
            self.wrapped_calls.append(dict(kwargs))
            if kwargs.get("stream"):
                kwargs["collect_usage"] = True  # the SDK wrapper's injection
            if "collect_usage" in kwargs:
                raise TypeError(
                    "create() got an unexpected keyword argument "
                    "'collect_usage'")
            return clean_create(**kwargs)

        wrapped_create.__wrapped__ = clean_create
        self.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=wrapped_create)))


def _merge(chunks):
    """Sum a list of AIMessageChunk into one (LangChain chunk merging)."""
    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c
    return merged


TEXT_STREAM = [
    _delta_chunk(content="The "),
    _delta_chunk(content=""),                    # empty delta — skipped
    _delta_chunk(content="FoS "),
    _delta_chunk(content="is 3.0."),
    _delta_chunk(finish_reason="stop"),
    _delta_chunk(usage=_USAGE),                  # usage-only final chunk
]


# ===========================================================================
# (a) plain text deltas
# ===========================================================================

def test_streamed_text_concatenates():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)

    pieces = list(model.stream([HumanMessage(content="What is the FoS?")]))
    assert all(isinstance(p, AIMessageChunk) for p in pieces)
    assert "".join(p.content for p in pieces) == "The FoS is 3.0."

    merged = _merge(pieces)
    assert merged.content == "The FoS is 3.0."
    assert merged.response_metadata["finish_reason"] == "stop"
    assert merged.response_metadata["model_name"] == "fake-gpt"


def test_stream_sends_stream_options_and_request_shape():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    list(model.stream([HumanMessage(content="q")]))

    sent = prompter.clean_calls[0]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["model"] == "fake-gpt"
    assert sent["messages"][-1] == {"role": "user", "content": "q"}


# ===========================================================================
# (b) tool call split across 3 delta chunks
# ===========================================================================

def test_streamed_tool_call_fragments_merge():
    stream = [
        _delta_chunk(tool_calls=[
            _tc_delta(index=0, id="call_abc", name="call_agent",
                      arguments=""),
        ]),
        _delta_chunk(tool_calls=[
            _tc_delta(index=0, arguments='{"agent_name": "settle'),
        ]),
        _delta_chunk(tool_calls=[
            _tc_delta(index=0, arguments='ment"}'),
        ]),
        _delta_chunk(finish_reason="tool_calls"),
        _delta_chunk(usage=_USAGE),
    ]
    prompter = _FakeStreamingPrompter(stream_chunks=stream)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)

    merged = _merge(list(model.stream([HumanMessage(content="go")])))
    assert len(merged.tool_calls) == 1
    tc = merged.tool_calls[0]
    assert tc["name"] == "call_agent"
    assert tc["args"] == {"agent_name": "settlement"}
    assert tc["id"] == "call_abc"
    assert merged.response_metadata["finish_reason"] == "tool_calls"


# ===========================================================================
# (c) final usage chunk -> usage_metadata
# ===========================================================================

def test_streamed_usage_metadata_present():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)

    merged = _merge(list(model.stream([HumanMessage(content="q")])))
    assert merged.usage_metadata == {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }


# ===========================================================================
# (d) the collect_usage landmine — the wrapped client is never streamed
# ===========================================================================

def test_wrapped_client_never_called_with_stream():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    list(model.stream([HumanMessage(content="q")]))

    # The wrapped create was bypassed entirely (would raise TypeError on
    # stream=True via the collect_usage injection).
    assert all(not c.get("stream") for c in prompter.wrapped_calls)
    assert prompter.wrapped_calls == []
    # The clean (__wrapped__) create carried the stream — no collect_usage.
    assert prompter.clean_calls[0]["stream"] is True
    assert "collect_usage" not in prompter.clean_calls[0]


def test_clean_openai_client_built_from_prompter_attrs(monkeypatch):
    """With http_client/base_url present, a clean OpenAI client is built once
    mirroring PrompterAPI's own construction, and cached on the instance."""
    stream_calls = []

    class _FakeOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            type(self).instances.append(self)

            def create(**ckwargs):
                stream_calls.append(ckwargs)
                return iter(list(TEXT_STREAM))

            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=create))

    monkeypatch.setattr(databricks_bridge, "_OpenAI", _FakeOpenAI)

    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    sentinel_http = object()
    prompter.http_client = sentinel_http
    prompter.base_url = "https://dsaaiapi.example/api/v1"
    prompter.api_key = None
    prompter.max_retries = 2

    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    list(model.stream([HumanMessage(content="q")]))
    list(model.stream([HumanMessage(content="again")]))

    # Built ONCE (cached), with the PrompterAPI construction mirrored.
    assert len(_FakeOpenAI.instances) == 1
    kw = _FakeOpenAI.instances[0].init_kwargs
    assert kw["http_client"] is sentinel_http
    assert kw["base_url"] == "https://dsaaiapi.example/api/v1"
    assert kw["api_key"] == "PrompterOpenAI-Databricks"
    assert kw["max_retries"] == 2
    # Both streams went through the clean client; the wrapped client was
    # never called at all.
    assert len(stream_calls) == 2
    assert prompter.wrapped_calls == []
    assert prompter.clean_calls == []


# ===========================================================================
# (e) streaming disabled (default) -> _generate single-chunk fallback
# ===========================================================================

def test_streaming_disabled_by_default_falls_back_to_generate():
    # _stream is exercised directly: the public stream() harness appends its
    # own empty chunk_position="last" chunk on newer langchain, so the
    # single-chunk contract is asserted at the _stream level.
    prompter = _FakeStreamingPrompter(response=_make_openai_response())
    model = PrompterChatModel(prompter=prompter)

    pieces = list(model._stream([HumanMessage(content="q")]))
    assert len(pieces) == 1
    assert pieces[0].message.content == "non-streamed answer"
    assert pieces[0].message.usage_metadata == {
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
    }
    # Exactly one plain (non-stream) call through the normal wrapped client.
    assert len(prompter.wrapped_calls) == 1
    assert "stream" not in prompter.wrapped_calls[0]

    # And the public stream() surface delivers the same text end-to-end.
    merged = _merge(list(model.stream([HumanMessage(content="q")])))
    assert merged.content == "non-streamed answer"


def test_env_var_enables_streaming(monkeypatch):
    monkeypatch.setenv("GEOTECH_PROMPTER_STREAMING", "1")
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter)  # field stays default False

    pieces = list(model.stream([HumanMessage(content="q")]))
    assert "".join(p.content for p in pieces) == "The FoS is 3.0."
    assert prompter.clean_calls[0]["stream"] is True


def test_env_var_falsey_values_stay_disabled(monkeypatch):
    monkeypatch.setenv("GEOTECH_PROMPTER_STREAMING", "0")
    prompter = _FakeStreamingPrompter(response=_make_openai_response())
    model = PrompterChatModel(prompter=prompter)
    pieces = list(model._stream([HumanMessage(content="q")]))
    assert len(pieces) == 1
    assert pieces[0].message.content == "non-streamed answer"
    # No stream was ever attempted anywhere.
    assert all(not c.get("stream") for c in prompter.clean_calls)
    assert all(not c.get("stream") for c in prompter.wrapped_calls)


# ===========================================================================
# (f) stream-setup exception -> fallback, no raise
# ===========================================================================

def test_stream_setup_error_falls_back_without_raising():
    prompter = _FakeStreamingPrompter(
        response=_make_openai_response("fallback answer"),
        stream_error=RuntimeError("proxy refused the stream"),
    )
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)

    with pytest.warns(UserWarning, match="falling back"):
        pieces = list(model._stream([HumanMessage(content="q")]))
    assert len(pieces) == 1
    assert pieces[0].message.content == "fallback answer"
    # The clean client WAS asked to stream (and raised); the wrapped client
    # then served the plain _generate fallback.
    assert prompter.clean_calls[0]["stream"] is True
    assert any("stream" not in c for c in prompter.wrapped_calls)


def test_tool_calls_survive_fallback_single_chunk():
    """The disabled/fallback single chunk must still carry tool_calls."""
    message = SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(
            id="call_9", type="function",
            function=SimpleNamespace(
                name="call_agent",
                arguments=json.dumps({"agent_name": "settlement"}),
            ),
        )],
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice], usage=_USAGE,
                               model="fake-gpt")
    prompter = _FakeStreamingPrompter(response=response)
    model = PrompterChatModel(prompter=prompter)  # streaming off

    pieces = list(model._stream([HumanMessage(content="go")]))
    assert len(pieces) == 1
    assert len(pieces[0].message.tool_calls) == 1
    tc = pieces[0].message.tool_calls[0]
    assert tc["name"] == "call_agent"
    assert tc["args"] == {"agent_name": "settlement"}
    assert tc["id"] == "call_9"


# ===========================================================================
# (g) best-effort metering
# ===========================================================================

def test_meter_log_called_with_token_counts():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    list(model.stream([HumanMessage(content="q")]))

    calls = prompter.logger.meter_calls
    assert len(calls) == 1
    call = calls[0]
    assert call["service"] == "Prompter"
    assert call["metrics"]["prompt_tokens"] == 10
    assert call["metrics"]["completion_tokens"] == 5
    assert call["metrics"]["total_tokens"] == 15
    assert call["metrics"]["model"] == "fake-gpt"
    assert call["operation"] == call["metrics"]["object"]


def test_meter_log_failure_never_breaks_turn():
    prompter = _FakeStreamingPrompter(
        stream_chunks=TEXT_STREAM,
        logger=_RecordingLogger(raise_on_meter=True),
    )
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    pieces = list(model.stream([HumanMessage(content="q")]))
    assert "".join(p.content for p in pieces) == "The FoS is 3.0."


def test_missing_logger_is_fine():
    prompter = _FakeStreamingPrompter(stream_chunks=TEXT_STREAM)
    prompter.logger = None
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    pieces = list(model.stream([HumanMessage(content="q")]))
    assert "".join(p.content for p in pieces) == "The FoS is 3.0."


# ===========================================================================
# invoke() is unchanged by the streaming work
# ===========================================================================

def test_invoke_still_uses_wrapped_nonstreaming_path():
    prompter = _FakeStreamingPrompter(response=_make_openai_response())
    model = PrompterChatModel(prompter=prompter, streaming_enabled=True)
    result = model.invoke([HumanMessage(content="q")])
    assert result.content == "non-streamed answer"
    assert len(prompter.wrapped_calls) == 1
    assert "stream" not in prompter.wrapped_calls[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
