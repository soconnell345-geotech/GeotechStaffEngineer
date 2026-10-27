"""Offline tests for the Claude leg of the ``palantir_models`` engine
(BACKLOG INF-27).

The real SDK only exists on Foundry, so these tests install FAKE
``palantir_models`` / ``language_model_service_api`` modules in
``sys.modules``. The class and field names mirror what the AI FDE ran on
Foundry on 2026-10-20 (``AnthropicClaudeLanguageModel.get(api_name)``,
``create_chat_completion(ClaudeChatCompletionRequest, max_rate_limit_retries
=4)``, ``ClaudeChatMessage(role, content=[ClaudeChatMessageContent(text= |
image=)])``, ``ClaudeTool(custom=ClaudeCustomTool(name, description,
input_schema))``; replies with ``content[i].text.text`` /
``content[i].tool_use.(id, name, input)``, ``stop_reason``, ``model`` and
``usage``) and, for the tool-loop blocks the FDE did not send
(``ClaudeChatToolUseContent(id, name, input)``,
``ClaudeChatToolResultContent(content, tool_use_id, is_error)``), the
public ``langchain-palantir`` adapter. The thinking blocks and the cache
marks (the system block's and the last message's) follow Anthropic's own
field names; those are the parts not yet seen on Foundry.
"""

import importlib.abc
import importlib.util
import json
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from webapp import engine_config  # noqa: E402

OPUS = "AnthropicClaude_5_5_Opus"
SONNET = "AnthropicClaude_5_5_Sonnet"


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------

class _Role:
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    UNKNOWN = "UNKNOWN"


class _MediaType:
    IMAGE_PNG = "IMAGE_PNG"
    IMAGE_JPEG = "IMAGE_JPEG"
    IMAGE_GIF = "IMAGE_GIF"
    IMAGE_WEBP = "IMAGE_WEBP"


class _Text:
    def __init__(self, text, cache_control=None):
        self.text, self.cache_control = text, cache_control


class _Base64Source:
    def __init__(self, data, media_type):
        self.data, self.media_type = data, media_type


class _ImageSource:
    def __init__(self, base64=None, media_set_reference_v2=None):
        self.base64 = base64


class _Image:
    def __init__(self, source, cache_control=None):
        self.source, self.cache_control = source, cache_control


class _ToolUse:
    def __init__(self, id, input, name):
        self.id, self.input, self.name = id, input, name


class _ToolResult:
    """``is_error`` is required, as Foundry's class requires it: a result
    built without it failed every reader turn after a tool call on Foundry
    build e, and a default here is why no test caught it."""

    def __init__(self, content, tool_use_id, is_error, cache_control=None):
        self.content, self.tool_use_id = content, tool_use_id
        self.is_error, self.cache_control = is_error, cache_control


class _Thinking:
    def __init__(self, signature, thinking):
        self.signature, self.thinking = signature, thinking


class _RedactedThinking:
    def __init__(self, data):
        self.data = data


class _Content:
    def __init__(self, text=None, image=None, document=None, tool_use=None,
                 tool_result=None, thinking=None, redacted_thinking=None):
        self.text, self.image, self.tool_use = text, image, tool_use
        self.tool_result, self.thinking = tool_result, thinking
        self.redacted_thinking = redacted_thinking


class _Message:
    def __init__(self, content, role):
        self.content, self.role = content, role


class _CustomTool:
    def __init__(self, input_schema, name, description=None):
        self.input_schema, self.name = input_schema, name
        self.description = description


class _Tool:
    def __init__(self, custom=None):
        self.custom = custom


class _EphemeralCache:
    def __init__(self, ttl=None):
        self.ttl = ttl


class _CacheControl:
    def __init__(self, ephemeral=None):
        self.ephemeral = ephemeral


class _SystemBlock:
    def __init__(self, text, cache_control=None):
        self.text, self.cache_control = text, cache_control


class _Request:
    def __init__(self, messages, max_tokens, output_config=None,
                 stop_sequences=None, system=None, system_blocks=None,
                 temperature=None, thinking=None, tool_choice=None,
                 tools=None, top_k=None, top_p=None):
        self.messages, self.max_tokens = messages, max_tokens
        self.stop_sequences, self.system = stop_sequences, system
        self.system_blocks, self.temperature = system_blocks, temperature
        self.thinking, self.tool_choice, self.tools = (thinking, tool_choice,
                                                       tools)


class _FakeClaude:
    """``create_chat_completion`` as the SDK declares it; answers from the
    queue (an exception in it is raised)."""

    queue: list = []
    requests: list = []
    retries: list = []

    def create_chat_completion(self, completion_request,
                               max_rate_limit_retries=4):
        _FakeClaude.requests.append(completion_request)
        _FakeClaude.retries.append(max_rate_limit_retries)
        answer = _FakeClaude.queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _AnthropicClaudeLanguageModel:
    got_names: list = []

    @classmethod
    def get(cls, model_api_name):
        cls.got_names.append(model_api_name)
        return _FakeClaude()


def _install_fake_sdk(monkeypatch, cache_types=True, block_marks=True):
    pm = types.ModuleType("palantir_models")
    pm_models = types.ModuleType("palantir_models.models")
    pm_models.AnthropicClaudeLanguageModel = _AnthropicClaudeLanguageModel
    pm.models = pm_models
    lms = types.ModuleType("language_model_service_api")
    v3 = types.ModuleType(
        "language_model_service_api.languagemodelservice_api_completion_v3")
    for name, cls in {
            "ClaudeChatMessageRole": _Role,
            "ClaudeChatMediaType": _MediaType,
            "ClaudeChatTextContent": _Text,
            "ClaudeImageBase64Source": _Base64Source,
            "ClaudeImageSource": _ImageSource,
            "ClaudeChatImageContent": _Image,
            "ClaudeChatToolUseContent": _ToolUse,
            "ClaudeChatToolResultContent": _ToolResult,
            "ClaudeChatThinkingContent": _Thinking,
            "ClaudeChatRedactedThinkingContent": _RedactedThinking,
            "ClaudeChatMessageContent": _Content,
            "ClaudeChatMessage": _Message,
            "ClaudeCustomTool": _CustomTool,
            "ClaudeTool": _Tool,
            "ClaudeChatCompletionRequest": _Request}.items():
        setattr(v3, name, cls)
    if cache_types:
        v3.ClaudeChatEphemeralCacheControl = _EphemeralCache
        v3.ClaudeChatCacheControl = _CacheControl
        v3.ClaudeChatSystemBlock = _SystemBlock
    if not block_marks:        # message blocks that take no cache field
        v3.ClaudeChatTextContent = lambda text: _Text(text)
        v3.ClaudeChatImageContent = lambda source: _Image(source)
        v3.ClaudeChatToolResultContent = (
            lambda content, tool_use_id, is_error:
            _ToolResult(content, tool_use_id, is_error))
    lms.languagemodelservice_api_completion_v3 = v3
    monkeypatch.setitem(sys.modules, "palantir_models", pm)
    monkeypatch.setitem(sys.modules, "palantir_models.models", pm_models)
    monkeypatch.setitem(sys.modules, "language_model_service_api", lms)
    monkeypatch.setitem(
        sys.modules,
        "language_model_service_api.languagemodelservice_api_completion_v3",
        v3)
    _FakeClaude.queue, _FakeClaude.requests, _FakeClaude.retries = [], [], []
    _AnthropicClaudeLanguageModel.got_names = []
    from webapp import palantir_sdk_engine as P
    P._CACHE_REFUSED.clear()
    monkeypatch.delenv(P.PROMPT_CACHE_ENV, raising=False)
    # No real waiting between retries.
    monkeypatch.setattr(P.random, "uniform", lambda a, b: 0.0)
    return v3


def _block(**kw):
    """A reply content item: a union with one member set."""
    base = dict(text=None, tool_use=None, thinking=None,
                redacted_thinking=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _reply(*items, stop="end_turn", usage=(12, 4), model="claude-opus-5-5"):
    return SimpleNamespace(
        content=list(items), role="ASSISTANT", stop_reason=stop, model=model,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                              max_tokens=1_000_000))


def _text(t):
    return _block(text=SimpleNamespace(text=t))


def _use(name, args, id_):
    return _block(tool_use=SimpleNamespace(id=id_, name=name, input=args))


def _model(**kw):
    from webapp.palantir_sdk_engine import PalantirClaudeChatModel
    return PalantirClaudeChatModel(model_api_name=kw.pop("name", OPUS), **kw)


_TOOL = {"type": "function",
         "function": {"name": "get_ka",
                      "description": "Rankine Ka for phi in degrees.",
                      "parameters": {"type": "object", "properties": {
                          "phi_deg": {"type": "number"}},
                          "required": ["phi_deg"]}}}


# ---------------------------------------------------------------------------
# A text turn
# ---------------------------------------------------------------------------

def test_text_turn(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("OK"))]
    out = _model(max_tokens=1234).invoke([
        SystemMessage(content="You are a geotechnical engineer."),
        HumanMessage(content="Say OK")])
    assert out.content == "OK"
    assert _AnthropicClaudeLanguageModel.got_names == [OPUS]
    req = _FakeClaude.requests[0]
    assert req.max_tokens == 1234 and req.temperature is None
    assert req.tools is None and req.tool_choice is None
    # The system prompt as one block marked for caching (tools + system).
    assert req.system is None
    assert req.system_blocks[0].text == "You are a geotechnical engineer."
    assert isinstance(req.system_blocks[0].cache_control.ephemeral,
                      _EphemeralCache)
    [msg] = req.messages
    assert msg.role == "USER" and msg.content[0].text.text == "Say OK"
    # And the last message's last block (the conversation so far), as the
    # proxy route's caching middleware marks it.
    assert isinstance(msg.content[0].text.cache_control.ephemeral,
                      _EphemeralCache)
    # Usage and the model that answered, as the app's records read them.
    assert out.usage_metadata == {"input_tokens": 12, "output_tokens": 4,
                                  "total_tokens": 16}
    meta = out.response_metadata
    assert meta["model_name"] == "claude-opus-5-5"
    assert meta["finish_reason"] == "stop"
    assert meta["cache_marks"] == "system and last message"


def test_cache_reads_count_into_the_input(monkeypatch):
    _install_fake_sdk(monkeypatch)
    reply = _reply(_text("OK"))
    reply.usage.cache_read_input_tokens = 11703
    reply.usage.cache_creation_input_tokens = 0
    _FakeClaude.queue = [reply]
    out = _model().invoke([HumanMessage(content="hi")])
    assert out.usage_metadata["input_tokens"] == 12 + 11703
    assert out.usage_metadata["input_token_details"] == {
        "cache_read": 11703, "cache_creation": 0}


def test_no_cache_mark_when_switched_off_or_unbuildable(monkeypatch):
    from webapp import palantir_sdk_engine as P
    _install_fake_sdk(monkeypatch)
    monkeypatch.setenv(P.PROMPT_CACHE_ENV, "0")
    _FakeClaude.queue = [_reply(_text("a"))]
    _model().invoke([SystemMessage(content="sys"), HumanMessage(content="q")])
    req = _FakeClaude.requests[-1]
    assert req.system == "sys" and req.system_blocks is None
    assert req.messages[0].content[0].text.cache_control is None
    # An SDK without the cache types: the plain system text.
    _install_fake_sdk(monkeypatch, cache_types=False)
    _FakeClaude.queue = [_reply(_text("b"))]
    out = _model().invoke([SystemMessage(content="sys"),
                           HumanMessage(content="q")])
    req = _FakeClaude.requests[-1]
    assert req.system == "sys" and req.system_blocks is None
    assert req.messages[0].content[0].text.cache_control is None
    assert out.response_metadata["cache_marks"] == "none"


def test_message_blocks_without_a_cache_field_leave_the_system_mark(
        monkeypatch):
    """Where this SDK's message blocks take no cache field, only the tools
    and system prompt are cached."""
    _install_fake_sdk(monkeypatch, block_marks=False)
    _FakeClaude.queue = [_reply(_text("a"))]
    out = _model().invoke([SystemMessage(content="sys"),
                           HumanMessage(content="q")])
    req = _FakeClaude.requests[0]
    assert req.system_blocks[0].cache_control is not None
    assert req.messages[0].content[0].text.cache_control is None
    assert out.response_metadata["cache_marks"] == "system"


def test_the_last_tool_result_carries_the_mark(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("done"))]
    _model().bind_tools([_TOOL]).invoke([
        HumanMessage(content="Ka for 30?"),
        AIMessage(content="", tool_calls=[{"name": "get_ka", "id": "toolu_1",
                                           "args": {"phi_deg": 30}}]),
        ToolMessage(content="0.333", tool_call_id="toolu_1", status="error")])
    last = _FakeClaude.requests[0].messages[-1]
    result = last.content[-1].tool_result
    assert result.tool_use_id == "toolu_1" and result.content == "0.333"
    assert result.is_error is True and result.cache_control is not None
    # Earlier messages are left as they were.
    first = _FakeClaude.requests[0].messages[0]
    assert first.content[0].text.cache_control is None


def test_a_refused_cache_mark_is_asked_once_without_it(monkeypatch):
    """The proxy's refusal shape (a 400 naming the field): asked again with
    the plain system text, and not marked again in this process."""
    from webapp import palantir_sdk_engine as P
    _install_fake_sdk(monkeypatch)
    err = RuntimeError("400 Client Error: INVALID_ARGUMENT "
                       "{unrecognizedProperty=cache_control}")
    err.error_name = "LanguageModelService:InvalidRequest"
    _FakeClaude.queue = [err, _reply(_text("fine")), _reply(_text("again"))]
    model = _model()
    out = model.invoke([SystemMessage(content="sys"),
                        HumanMessage(content="q")])
    assert out.content == "fine"
    first, second = _FakeClaude.requests
    assert first.system_blocks is not None
    assert second.system == "sys" and second.system_blocks is None
    assert second.messages[0].content[0].text.cache_control is None
    assert out.response_metadata["cache_marks"] == "none"
    assert OPUS in P._CACHE_REFUSED
    # A copy (bind_tools makes one per call) does not ask with it again.
    model.bind_tools([_TOOL]).invoke([SystemMessage(content="sys"),
                                      HumanMessage(content="q")])
    assert _FakeClaude.requests[2].system_blocks is None


_SERVICE = "https://language-model-service-system.example/api/v2/completions"


class _ErrorResponse:
    """A requests Response is falsy for any 4xx or 5xx."""

    def __init__(self, status_code):
        self.status_code = status_code

    def __bool__(self):
        return False


def _http_error(status, text, cause=False):
    """A failed call as the SDK raises it, the status on ``response`` or on
    ``_cause.response``; the URL names the in-platform service's host."""
    reason = "Client" if status < 500 else "Server"
    err = RuntimeError(f"{status} {reason} Error: for url: {_SERVICE}. {text}")
    if cause:
        err._cause = SimpleNamespace(response=_ErrorResponse(status))
    else:
        err.response = _ErrorResponse(status)
    return err


@pytest.mark.parametrize("status", [500, 502, 504])
def test_a_5xx_with_system_in_the_url_is_not_resent(monkeypatch, status):
    """A one-off 5xx is raised as it is: not resent without the marks, and
    not blamed on them."""
    from webapp import palantir_sdk_engine as P
    _install_fake_sdk(monkeypatch)
    err = _http_error(status, "INTERNAL", cause=status == 502)
    assert not P._refused_cache(err)
    _FakeClaude.queue = [err, _reply(_text("not asked"))]
    with pytest.raises(RuntimeError, match=str(status)):
        _model().invoke([SystemMessage(content="sys"),
                         HumanMessage(content="q")])
    assert len(_FakeClaude.requests) == 1
    assert OPUS not in P._CACHE_REFUSED


@pytest.mark.parametrize("text", [
    "INVALID_ARGUMENT {unrecognizedProperty=cache_control}",
    "ErrorCode: 'INVALID_ARGUMENT'. Parameters: {'field': 'systemBlocks'}",
    "messages.0.content.0.text.cacheControl: Extra inputs are not permitted",
])
def test_a_400_naming_a_cache_field_is_a_refusal(text):
    from webapp import palantir_sdk_engine as P
    assert P._refused_cache(_http_error(400, text))
    assert P._refused_cache(_http_error(400, text, cause=True))
    # The same words on a 5xx are not.
    assert not P._refused_cache(_http_error(500, text))


@pytest.mark.parametrize("text", [
    "INVALID_ARGUMENT: prompt is too long: 212000 tokens > 200000 maximum",
    "INVALID_ARGUMENT: unexpected tool_use_id found in tool_result blocks: "
    "toolu_9. Each tool_result block must have a corresponding tool_use",
    "INVALID_ARGUMENT {unrecognizedProperty=output_config}",
    "messages: roles must alternate",
])
def test_an_unrelated_400_is_not_blamed_on_the_cache_marks(monkeypatch, text):
    """Sent once and raised: not resent without the marks on every turn."""
    from webapp import palantir_sdk_engine as P
    _install_fake_sdk(monkeypatch)
    bad = _http_error(400, text)
    bad.error_name = "LanguageModelService:InvalidArgument"
    assert not P._refused_cache(bad)
    _FakeClaude.queue = [bad, _reply(_text("not asked"))]
    with pytest.raises(RuntimeError, match="400"):
        _model().invoke([SystemMessage(content="sys"),
                         HumanMessage(content="q")])
    assert len(_FakeClaude.requests) == 1
    assert OPUS not in P._CACHE_REFUSED


# ---------------------------------------------------------------------------
# A tool-call turn
# ---------------------------------------------------------------------------

def test_tool_call_turn(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_use("get_ka", {"phi_deg": 30}, "toolu_1"),
                                stop="tool_use", usage=(404, 52))]
    out = _model().bind_tools([_TOOL]).invoke(
        [HumanMessage(content="Use the get_ka tool for phi = 30 degrees.")])
    req = _FakeClaude.requests[0]
    [tool] = req.tools
    assert tool.custom.name == "get_ka"
    assert tool.custom.description.startswith("Rankine")
    assert tool.custom.input_schema["required"] == ["phi_deg"]
    assert req.tool_choice is None             # the service default: auto
    assert out.tool_calls == [{"name": "get_ka", "args": {"phi_deg": 30},
                               "id": "toolu_1", "type": "tool_call"}]
    assert out.response_metadata["finish_reason"] == "tool_calls"


def test_tool_loop_round_trip(monkeypatch):
    """Two parallel calls with their text, then both results: one assistant
    turn of text + two tool_use blocks, one user turn of two tool_result
    blocks (an error flagged)."""
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("Ka = 0.333, Kp = 3.0."))]
    out = _model().bind_tools([_TOOL]).invoke([
        HumanMessage(content="Ka and Kp at phi 30"),
        AIMessage(content="Computing both.", tool_calls=[
            {"name": "get_ka", "args": {"phi_deg": 30}, "id": "toolu_1"},
            {"name": "get_kp", "args": {"phi_deg": 30}, "id": "toolu_2"}]),
        ToolMessage(content="0.333", tool_call_id="toolu_1"),
        ToolMessage(content="no such tool", tool_call_id="toolu_2",
                    status="error"),
    ])
    assert out.content == "Ka = 0.333, Kp = 3.0."
    sent = _FakeClaude.requests[0].messages
    assert [m.role for m in sent] == ["USER", "ASSISTANT", "USER"]
    ai = sent[1].content
    assert ai[0].text.text == "Computing both."
    assert [(c.tool_use.id, c.tool_use.name, c.tool_use.input)
            for c in ai[1:]] == [("toolu_1", "get_ka", {"phi_deg": 30}),
                                 ("toolu_2", "get_kp", {"phi_deg": 30})]
    results = [c.tool_result for c in sent[2].content]
    assert [(r.tool_use_id, r.content, r.is_error) for r in results] == [
        ("toolu_1", "0.333", False), ("toolu_2", "no such tool", True)]


@pytest.mark.parametrize("block_marks", [True, False])
def test_a_tool_round_trip_call_ok_result_next_call(monkeypatch,
                                                    block_marks):
    """A call, its ordinary result, then the next call: the turn after a
    tool call goes out with ``is_error=False`` on the result, whether the
    result is rebuilt with the cache mark or sent as it was built (Foundry
    build e: every such turn failed before it was sent)."""
    _install_fake_sdk(monkeypatch, block_marks=block_marks)
    _FakeClaude.queue = [
        _reply(_use("get_ka", {"phi_deg": 30}, "toolu_1"), stop="tool_use"),
        _reply(_use("get_ka", {"phi_deg": 34}, "toolu_2"), stop="tool_use")]
    model = _model().bind_tools([_TOOL])
    history = [HumanMessage(content="Ka at phi 30, then at 34?")]
    first = model.invoke(history)
    history += [first, ToolMessage(content="0.333", tool_call_id="toolu_1")]
    second = model.invoke(history)
    assert len(_FakeClaude.requests) == 2
    assert second.tool_calls[0]["id"] == "toolu_2"
    [result] = [c.tool_result for c in _FakeClaude.requests[1].messages[-1]
                .content if c.tool_result is not None]
    assert (result.tool_use_id, result.content) == ("toolu_1", "0.333")
    assert result.is_error is False
    assert (result.cache_control is not None) is block_marks


def test_a_pure_tool_call_turn_sends_no_empty_text(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("done"))]
    _model().invoke([
        HumanMessage(content="go"),
        AIMessage(content="", tool_calls=[
            {"name": "t", "args": {}, "id": "toolu_9"}]),
        ToolMessage(content="", tool_call_id="toolu_9")])
    sent = _FakeClaude.requests[0].messages
    assert len(sent[1].content) == 1 and sent[1].content[0].tool_use.id == \
        "toolu_9"
    assert sent[2].content[0].tool_result.content == "(no output)"


def test_tool_input_as_json_text_is_parsed(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_use("get_ka", '{"phi_deg": 32}', "toolu_3"),
                                stop="tool_use")]
    out = _model().invoke([HumanMessage(content="q")])
    assert out.tool_calls[0]["args"] == {"phi_deg": 32}


# ---------------------------------------------------------------------------
# An image turn
# ---------------------------------------------------------------------------

def test_image_turn(monkeypatch):
    """The vision tools' OpenAI-style data URI and LangChain's own image
    block both reach Claude as base64 images with their media type;
    ``detail`` means nothing to Claude and is dropped."""
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("Red"))]
    out = _model().invoke([HumanMessage(content=[
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgo=", "detail": "original"}},
        {"type": "image", "base64": "/9j/4AAQ", "mime_type": "image/jpeg"},
        {"type": "text", "text": "What single colour fills this image?"},
    ])])
    assert out.content == "Red"
    [msg] = _FakeClaude.requests[0].messages
    first, second, text = msg.content
    assert first.image.source.base64.data == "iVBORw0KGgo="
    assert first.image.source.base64.media_type == "IMAGE_PNG"
    assert second.image.source.base64.data == "/9j/4AAQ"
    assert second.image.source.base64.media_type == "IMAGE_JPEG"
    assert text.text.text == "What single colour fills this image?"


def test_image_after_tool_results_joins_their_turn(monkeypatch):
    """The inline-image shape: a tool's result, then the picture it drew as
    a user message: one user turn, the result first."""
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("The note says 3600 psi."))]
    _model().invoke([
        HumanMessage(content="what does the note say?"),
        AIMessage(content="", tool_calls=[
            {"name": "render_region", "args": {"page": 0}, "id": "toolu_5"}]),
        ToolMessage(content='{"view": [0, 0, 10, 10]}',
                    tool_call_id="toolu_5"),
        HumanMessage(content=[
            {"type": "text", "text": "the image you rendered"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,cQ=="}}]),
    ])
    sent = _FakeClaude.requests[0].messages
    assert [m.role for m in sent] == ["USER", "ASSISTANT", "USER"]
    last = sent[2].content
    assert last[0].tool_result.tool_use_id == "toolu_5"
    assert last[1].text.text == "the image you rendered"
    assert last[2].image.source.base64.data == "cQ=="


def test_an_image_by_web_address_is_said_not_sent(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("ok"))]
    _model().invoke([HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        {"type": "text", "text": "look"}])])
    [msg] = _FakeClaude.requests[0].messages
    assert "web address" in msg.content[0].text.text
    assert msg.content[0].image is None


# ---------------------------------------------------------------------------
# A block this route cannot send is refused, not dropped (BACKLOG RI-87)
# ---------------------------------------------------------------------------

_PDF_FILE = {"type": "file", "base64": "JVBERi0xLjcgbWFkZS11cA==",
             "mime_type": "application/pdf"}


@pytest.mark.parametrize("block, named", [
    (_PDF_FILE, "type 'file' (application/pdf)"),
    ({"type": "audio", "base64": "UklGRmFrZQ==", "mime_type": "audio/wav"},
     "type 'audio' (audio/wav)"),
    ({"type": "image", "mime_type": "image/png"},
     "type 'image' holds no image: no data and no web address"),
], ids=["pdf", "audio", "image_with_no_data"])
def test_a_block_this_route_cannot_send_is_refused_not_dropped(
        monkeypatch, block, named):
    """Until RI-87 such a block was left out without a word, and the model
    answered from the words alone."""
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("answered without it"))]
    with pytest.raises(ValueError) as refused:
        _model().invoke([HumanMessage(content=[
            block, {"type": "text", "text": "Extract it."}])])
    assert named in str(refused.value)
    if block["type"] == "image":     # this route does carry images
        assert "carries text and images only" not in str(refused.value)
    assert refused.value.nothing_sent is True
    if block.get("base64"):
        assert block["base64"] not in str(refused.value), "data never shown"
    assert _FakeClaude.requests == []


@pytest.mark.parametrize("content, said", [
    ([_PDF_FILE], "(a file block this route cannot send was left out)"),
    ([{"type": "text", "text": "Saved notes:"}, _PDF_FILE,
      {"type": "audio", "base64": "UklGRmFrZQ==", "mime_type": "audio/wav"}],
     "Saved notes:\n(a file block this route cannot send was left out)\n"
     "(an audio block this route cannot send was left out)"),
    ([{"type": "text", "text": "The chart:"},
      {"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
     "The chart:\n(an image given by web address was left out: this route "
     "sends images as data only)"),
    ([{"type": "image", "mime_type": "image/png"}],
     "(an image block with no data was left out)"),
], ids=["file_alone", "beside_words", "image_by_web_address",
        "image_with_no_data"])
def test_a_block_a_tool_result_cannot_send_is_said_in_its_text(
        monkeypatch, content, said):
    """deepagents' ``read_file`` answers a scratch file with a binary
    extension with a ``file`` block. Dropped, the model read only "(no
    output)"; refused, the agent's turn would end. It is said instead,
    as the web-address image is, and the turn goes on."""
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("The notes file is not readable."))]
    out = _model().invoke([
        HumanMessage(content="read my notes"),
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"file_path": "/notes.pdf"},
             "id": "toolu_9"}]),
        ToolMessage(content=content, tool_call_id="toolu_9")])
    assert out.content == "The notes file is not readable."
    [request] = _FakeClaude.requests
    [result] = request.messages[2].content      # nothing beside the result
    assert result.tool_result.tool_use_id == "toolu_9"
    assert result.tool_result.content == said


def test_the_ingest_engine_sends_no_pdf_on_this_route(monkeypatch):
    """report_ingest's engine knows this class takes no PDF; told it does,
    the class refuses the PDF itself. Either way nothing is sent."""
    from report_ingest.engine import (ChatModelEngine, pdf_block, text_block,
                                      user)
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("never"))]
    turn = [user(pdf_block(b"%PDF-1.7 made-up"), text_block("Extract it."))]
    engine = ChatModelEngine(_model())
    assert engine.accepts_pdf is False
    with pytest.raises(ValueError, match="not known to take a PDF"):
        engine.complete(turn)
    told = ChatModelEngine(_model(), accepts_pdf=True)
    with pytest.raises(ValueError, match="palantir_models"):
        told.complete(turn)
    assert _FakeClaude.requests == []
    # Nothing went out, so neither engine counts a call.
    for each in (engine, told):
        assert (each.meter.calls, each.meter.failed) == (0, 0)


def test_the_proxy_takes_a_pdf_only_with_its_switch(monkeypatch):
    """The Foundry proxy's ``ChatAnthropic``: no document block had been
    sent through it, so the ingest sends none until
    GEOTECH_FOUNDRY_CLAUDE_PDF is on (once the probe has read page 2)."""
    pytest.importorskip("langchain_anthropic")
    from report_ingest.engine import FOUNDRY_CLAUDE_PDF_ENV, ChatModelEngine
    proxy = engine_config.foundry_chat_anthropic_class()(
        model="claude-made-up", max_tokens=1000,
        anthropic_api_url="https://foundry.example.invalid/proxy/anthropic",
        anthropic_api_key="made-up")
    monkeypatch.delenv(FOUNDRY_CLAUDE_PDF_ENV, raising=False)
    assert ChatModelEngine(proxy).accepts_pdf is False
    monkeypatch.setenv(FOUNDRY_CLAUDE_PDF_ENV, "1")
    assert ChatModelEngine(proxy).accepts_pdf is True


def test_the_pdf_probe_on_this_route_reports_the_refusal(monkeypatch):
    pytest.importorskip("pymupdf")
    from report_ingest.model_first import probe_pdf_route
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("QUILLBANK-2468"))]
    out = probe_pdf_route(_model(), words=("AMBERWICK-1357",
                                           "QUILLBANK-2468"))
    assert out["use_pdf"] is False and out["accepts_pdf"] is False
    assert out["pdf"]["outcome"] == "refused"
    assert "type 'file' (application/pdf)" in out["pdf"]["error"]
    assert out["pictures"]["picture_read"] is True
    [request] = _FakeClaude.requests        # the pictures call alone
    pictures = [c for c in request.messages[0].content if c.image is not None]
    assert len(pictures) == 2
    assert "refused before it was sent" in out["says"]
    assert "send the pages as pictures" in out["says"]


def test_the_vision_probe_sizes_it_as_claude(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from funhouse_agent import vision_probe
    assert vision_probe._is_claude(_model(), None)


# ---------------------------------------------------------------------------
# Thinking blocks in the tool loop
# ---------------------------------------------------------------------------

def test_thinking_goes_back_within_the_tool_loop_only(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(
        _block(thinking=SimpleNamespace(thinking="Rankine: tan^2(45-phi/2).",
                                        signature="sig-1")),
        _use("get_ka", {"phi_deg": 30}, "toolu_1"), stop="tool_use")]
    model = _model().bind_tools([_TOOL])
    first = model.invoke([HumanMessage(content="Ka at 30?")])
    assert first.additional_kwargs["reasoning"] == "Rankine: tan^2(45-phi/2)."
    assert first.content == ""          # the thinking is not the answer
    _FakeClaude.queue = [_reply(_text("0.333"))]
    model.invoke([HumanMessage(content="Ka at 30?"), first,
                  ToolMessage(content="0.333", tool_call_id="toolu_1")])
    ai = _FakeClaude.requests[1].messages[1].content
    assert ai[0].thinking.signature == "sig-1"     # thinking first
    assert ai[0].thinking.thinking == "Rankine: tan^2(45-phi/2)."
    assert ai[1].tool_use.id == "toolu_1"
    # A later question: the earlier turn's thinking is not sent again.
    _FakeClaude.queue = [_reply(_text("1/3"))]
    model.invoke([HumanMessage(content="Ka at 30?"), first,
                  ToolMessage(content="0.333", tool_call_id="toolu_1"),
                  AIMessage(content="0.333"),
                  HumanMessage(content="As a fraction?")])
    ai = _FakeClaude.requests[2].messages[1].content
    assert ai[0].tool_use.id == "toolu_1" and ai[0].thinking is None


# ---------------------------------------------------------------------------
# Stop reasons, the 403 and the retries
# ---------------------------------------------------------------------------

def test_a_reply_cut_at_the_cap_reads_as_cut(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from funhouse_agent.deep.tool_guards import output_was_cut
    _FakeClaude.queue = [_reply(_text("partial"), stop="max_tokens")]
    out = _model().invoke([HumanMessage(content="write a lot")])
    assert out.response_metadata["finish_reason"] == "length"
    assert output_was_cut(out)


def _conjure_403():
    """Foundry's refusal as the FDE's probe of 2026-10-20 recorded it."""
    err = RuntimeError(
        "403 Client Error: Forbidden for url: https://host/language-model-"
        "service/api/llm/v3/completion/AnthropicClaude_5_5_Sonnet. ErrorCode: "
        "'PERMISSION_DENIED'. ErrorName: 'LanguageModelService:"
        "UseModelPermissionDenied'.")
    err.error_name = "LanguageModelService:UseModelPermissionDenied"
    return err


def test_model_not_imported_is_said_plainly(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import FoundryModelNotImported
    _FakeClaude.queue = [_conjure_403()]
    with pytest.raises(FoundryModelNotImported) as got:
        _model(name=SONNET).invoke([HumanMessage(content="hi")])
    text = str(got.value)
    assert "Models panel" in text and "Claude Sonnet 5.5" in text
    assert SONNET in text and "403 UseModelPermissionDenied" in text
    assert len(_FakeClaude.requests) == 1            # not retried
    # The plain words reach the turn's error text.
    from webapp import core
    assert "Models panel" in core.friendly_turn_error(got.value)


def test_rate_limit_retries_are_passed_through(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [_reply(_text("a")), _reply(_text("b"))]
    _model().invoke([HumanMessage(content="q")])
    _model(rate_limit_retries=7).invoke([HumanMessage(content="q")])
    assert _FakeClaude.retries == [4, 7]


def test_a_dropped_connection_is_retried(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _FakeClaude.queue = [ConnectionResetError("reset by peer"),
                         _reply(_text("back"))]
    out = _model().invoke([HumanMessage(content="q")])
    assert out.content == "back" and len(_FakeClaude.requests) == 2


# ---------------------------------------------------------------------------
# The switch: which route a Claude model takes on Foundry
# ---------------------------------------------------------------------------

_CLEAR = ("ANTHROPIC_API_KEY", "GEOTECH_FOUNDRY_MODELS", "GEOTECH_WEBAPP_MODEL",
          "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN", "GEOTECH_FOUNDRY_HOST",
          "FOUNDRY_HOSTNAME", "FOUNDRY_URL", "GEOTECH_DEPLOYMENT",
          "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "FOUNDRY_PROXY_URL",
          "MAESTRO_FOUNDRY_PROXY_URL", "GEOTECH_HELPER_MODELS",
          engine_config.FOUNDRY_CLAUDE_SDK_ENV)


@pytest.fixture
def foundry(monkeypatch, tmp_path):
    for env in _CLEAR:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok-123")
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.palantirfoundry.com")
    return monkeypatch


class _ImportWatch(importlib.abc.MetaPathFinder):
    """Records every attempt to import the Foundry SDK; finds nothing."""

    def __init__(self):
        self.seen = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("palantir_models",
                                      "language_model_service_api"):
            self.seen.append(fullname)
        return None


def test_switch_is_off_by_default(monkeypatch):
    env = engine_config.FOUNDRY_CLAUDE_SDK_ENV
    assert env == "GEOTECH_FOUNDRY_CLAUDE_SDK"
    monkeypatch.delenv(env, raising=False)
    assert not engine_config.foundry_claude_sdk_on()
    for value, on in (("1", True), ("on", True), ("true", True),
                      ("0", False), ("off", False), ("", False)):
        monkeypatch.setenv(env, value)
        assert engine_config.foundry_claude_sdk_on() is on


@pytest.mark.skipif(importlib.util.find_spec("langchain_anthropic") is None,
                    reason="langchain_anthropic not installed")
def test_switch_off_imports_nothing(foundry):
    """Off (the default): a Claude API name goes to the proxy, and the
    Foundry SDK is never imported (nor asked for a model)."""
    for name in list(sys.modules):
        if name.split(".")[0] in ("palantir_models",
                                  "language_model_service_api"):
            foundry.delitem(sys.modules, name)
    watch = _ImportWatch()
    foundry.setattr(sys, "meta_path", [watch] + sys.meta_path)
    res = engine_config.resolve_engine(model_id=OPUS)
    assert res.source == "foundry", res.message
    assert type(res.model).__name__ == "FoundryChatAnthropic"
    assert watch.seen == []
    assert "palantir_models" not in sys.modules


def test_switch_on_takes_the_sdk(foundry):
    _install_fake_sdk(foundry)
    foundry.setenv(engine_config.FOUNDRY_CLAUDE_SDK_ENV, "1")
    res = engine_config.resolve_engine(model_id=f"anthropic:{SONNET}")
    assert res.source == "foundry_sdk", res.message
    assert type(res.model).__name__ == "PalantirClaudeChatModel"
    assert res.model.model_api_name == SONNET
    assert res.model.max_tokens == engine_config._default_max_tokens()
    assert engine_config.FOUNDRY_CLAUDE_SDK_ENV in res.message
    # GPT names still take the GPT leg; nothing else moved.
    assert engine_config.model_family("GPT_5_1") == "openai"


def test_switch_on_without_the_claude_class_says_so(foundry):
    for name in list(sys.modules):
        if name.split(".")[0] in ("palantir_models",
                                  "language_model_service_api"):
            foundry.delitem(sys.modules, name)
    watch = _ImportWatch()
    foundry.setattr(sys, "meta_path", [watch] + sys.meta_path)
    foundry.setenv(engine_config.FOUNDRY_CLAUDE_SDK_ENV, "on")
    res = engine_config.resolve_engine(model_id=OPUS)
    assert res.source == "error"
    assert "AnthropicClaudeLanguageModel" in res.message
    assert "Libraries panel" in res.message
    # On, the SDK is asked for (so the watch in the test above would see it).
    assert "palantir_models" in watch.seen


@pytest.mark.skipif(importlib.util.find_spec("langchain_anthropic") is None,
                    reason="langchain_anthropic not installed")
def test_switch_on_a_rid_still_uses_the_proxy_and_says_so(foundry):
    foundry.setenv(engine_config.FOUNDRY_CLAUDE_SDK_ENV, "1")
    rid = "ri.language-model-service..language-model.claude-opus-5-5"
    res = engine_config.resolve_engine(model_id=rid)
    assert res.source == "foundry", res.message
    assert "a model RID always uses the proxy" in res.message


def test_the_switch_shows_in_connection_diagnostics(foundry):
    from webapp import diagnostics
    foundry.setenv(engine_config.FOUNDRY_CLAUDE_SDK_ENV, "1")
    assert "GEOTECH_FOUNDRY_CLAUDE_SDK=1" in diagnostics._env_check()["detail"]


def test_a_full_turn_through_the_switch(foundry):
    """resolve_engine -> bind_tools -> a tool call -> its result -> answer."""
    _install_fake_sdk(foundry)
    foundry.setenv(engine_config.FOUNDRY_CLAUDE_SDK_ENV, "1")
    model = engine_config.resolve_engine(model_id=OPUS).model
    bound = model.bind_tools([_TOOL])
    _FakeClaude.queue = [_reply(_use("get_ka", {"phi_deg": 30}, "toolu_1"),
                                stop="tool_use"),
                         _reply(_text("Ka = 0.333"))]
    history = [SystemMessage(content="sys"),
               HumanMessage(content="Ka at phi 30?")]
    first = bound.invoke(history)
    history += [first, ToolMessage(content=json.dumps(0.333),
                                   tool_call_id="toolu_1")]
    assert bound.invoke(history).content == "Ka = 0.333"
    assert [m.role for m in _FakeClaude.requests[1].messages] == [
        "USER", "ASSISTANT", "USER"]
