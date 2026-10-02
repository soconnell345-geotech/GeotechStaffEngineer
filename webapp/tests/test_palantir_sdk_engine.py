"""Offline tests for the ``palantir_models`` SDK engine (Foundry in-platform).

The real SDK only exists on Foundry, so these tests install FAKE
``palantir_models`` / ``language_model_service_api`` modules in ``sys.modules``
whose class signatures MIRROR the live enclave introspection (2026-07-21):

    ChatMessage(role, content=None, function_call=None, name=None,
                tool_call_id=None, tool_calls=None)
    GptTool(function=...); GptFunctionTool(name, parameters, description=None,
                                           strict=None)
    GptToolCall(id, tool_call); GptToolCallInfo(function=...)
    FunctionToolCallInfo(arguments, name)
    GptChatCompletionRequest(messages, ..., max_tokens=None, stop=None,
                             temperature=None, tools=None, ...)
    OpenAiGptChatLanguageModel.get(model_api_name)
"""

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


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------

class _ChatMessageRole:
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"
    FUNCTION = "FUNCTION"
    UNKNOWN = "UNKNOWN"


class _ChatMessage:
    def __init__(self, role, content=None, function_call=None, name=None,
                 tool_call_id=None, tool_calls=None):
        self.role, self.content = role, content
        self.function_call, self.name = function_call, name
        self.tool_call_id, self.tool_calls = tool_call_id, tool_calls


class _GptFunctionTool:
    def __init__(self, name, parameters, description=None, strict=None):
        self.name, self.parameters = name, parameters
        self.description, self.strict = description, strict


class _GptTool:
    def __init__(self, function=None, type_of_union=None):
        self.function = function


class _FunctionToolCallInfo:
    def __init__(self, arguments, name):
        self.arguments, self.name = arguments, name


class _GptToolCallInfo:
    def __init__(self, function=None, type_of_union=None):
        self.function = function


class _GptToolCall:
    def __init__(self, id, tool_call):
        self.id, self.tool_call = id, tool_call


class _GptChatCompletionRequest:
    def __init__(self, messages, frequency_penalty=None, logit_bias=None,
                 max_tokens=None, n=None, presence_penalty=None,
                 reasoning_effort=None, response_format=None, seed=None,
                 stop=None, temperature=None, tool_choice=None, tools=None,
                 top_p=None):
        self.messages, self.max_tokens, self.stop = messages, max_tokens, stop
        self.temperature, self.tools = temperature, tools
        self.tool_choice = tool_choice


class _FakeSdkModel:
    """Captures requests; returns the canned response set on the class."""

    next_response = None
    last_request = None

    def create_chat_completion(self, request):
        _FakeSdkModel.last_request = request
        return _FakeSdkModel.next_response


class _OpenAiGptChatLanguageModel:
    got_names = []

    @classmethod
    def get(cls, model_api_name):
        cls.got_names.append(model_api_name)
        return _FakeSdkModel()


# The vision door, as the owner's Foundry example (2026-09-30) and the
# public langchain-palantir adapter use it.

class _ImageDetail:
    AUTO = "AUTO"
    LOW = "LOW"
    HIGH = "HIGH"


class _Base64ImageContent:
    def __init__(self, image_url, detail=None):
        self.image_url, self.detail = image_url, detail


class _ChatMessageContent:
    def __init__(self, text=None, image=None):
        self.text, self.image = text, image


class _MultiContentChatMessage:
    def __init__(self, contents, role, tool_call_id=None, tool_calls=None):
        self.contents, self.role = contents, role
        self.tool_call_id, self.tool_calls = tool_call_id, tool_calls


class _GptChatWithVisionCompletionRequest(_GptChatCompletionRequest):
    pass


class _FakeVisionModel:
    """Takes the rate-limit keyword the vision examples pass."""

    last_request = None
    last_retries = None

    def create_chat_completion(self, completion_request,
                               max_rate_limit_retries=None):
        _FakeVisionModel.last_request = completion_request
        _FakeVisionModel.last_retries = max_rate_limit_retries
        return _FakeSdkModel.next_response


class _OpenAiGptChatWithVisionLanguageModel:
    got_names = []

    @classmethod
    def get(cls, model_api_name):
        cls.got_names.append(model_api_name)
        return _FakeVisionModel()


def _install_fake_sdk(monkeypatch):
    pm = types.ModuleType("palantir_models")
    pm_models = types.ModuleType("palantir_models.models")
    pm_models.OpenAiGptChatLanguageModel = _OpenAiGptChatLanguageModel
    pm_models.OpenAiGptChatWithVisionLanguageModel = \
        _OpenAiGptChatWithVisionLanguageModel
    pm.models = pm_models

    lms = types.ModuleType("language_model_service_api")
    base = types.ModuleType(
        "language_model_service_api.languagemodelservice_api")
    base.ChatMessage = _ChatMessage
    base.ChatMessageRole = _ChatMessageRole
    base.ImageDetail = _ImageDetail
    base.Base64ImageContent = _Base64ImageContent
    base.ChatMessageContent = _ChatMessageContent
    base.MultiContentChatMessage = _MultiContentChatMessage
    v3 = types.ModuleType(
        "language_model_service_api.languagemodelservice_api_completion_v3")
    v3.GptTool = _GptTool
    v3.GptFunctionTool = _GptFunctionTool
    v3.GptToolCall = _GptToolCall
    v3.GptToolCallInfo = _GptToolCallInfo
    v3.FunctionToolCallInfo = _FunctionToolCallInfo
    v3.GptChatCompletionRequest = _GptChatCompletionRequest
    v3.GptChatWithVisionCompletionRequest = _GptChatWithVisionCompletionRequest
    lms.languagemodelservice_api = base
    lms.languagemodelservice_api_completion_v3 = v3

    monkeypatch.setitem(sys.modules, "palantir_models", pm)
    monkeypatch.setitem(sys.modules, "palantir_models.models", pm_models)
    monkeypatch.setitem(sys.modules, "language_model_service_api", lms)
    monkeypatch.setitem(
        sys.modules,
        "language_model_service_api.languagemodelservice_api", base)
    monkeypatch.setitem(
        sys.modules,
        "language_model_service_api.languagemodelservice_api_completion_v3",
        v3)
    _FakeSdkModel.next_response = None
    _FakeSdkModel.last_request = None
    _OpenAiGptChatLanguageModel.got_names = []
    _FakeVisionModel.last_request = None
    _FakeVisionModel.last_retries = None
    _OpenAiGptChatWithVisionLanguageModel.got_names = []


def _text_response(text="OK", finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(content=text, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              total_tokens=15),
        model="GPT_5_1")


def _tool_call_response():
    fn = SimpleNamespace(name="bearing", arguments=json.dumps({"width_m": 2}))
    tc = SimpleNamespace(id="call_1", tool_call=SimpleNamespace(function=fn))
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=None, model="GPT_5_1")


# ---------------------------------------------------------------------------
# Chat model behaviour
# ---------------------------------------------------------------------------

def test_invoke_plain_text(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response("Hello!")
    m = PalantirSdkChatModel(model_api_name="GPT_5_1", max_tokens=1234)
    out = m.invoke([SystemMessage(content="be brief"),
                    HumanMessage(content="hi")])
    assert out.content == "Hello!"
    assert out.usage_metadata["total_tokens"] == 15
    assert _OpenAiGptChatLanguageModel.got_names == ["GPT_5_1"]
    req = _FakeSdkModel.last_request
    assert req.max_tokens == 1234 and req.temperature is None
    assert [m.role for m in req.messages] == ["SYSTEM", "USER"]
    assert req.messages[0].content == "be brief"


def test_tool_binding_and_tool_call_response(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _tool_call_response()
    tool = {"type": "function",
            "function": {"name": "bearing", "description": "compute",
                         "parameters": {"type": "object", "properties": {
                             "width_m": {"type": "number"}}}}}
    m = PalantirSdkChatModel(model_api_name="GPT_5_1").bind_tools([tool])
    out = m.invoke([HumanMessage(content="2 m footing")])
    # tools sent as GptTool(function=GptFunctionTool(...))
    req = _FakeSdkModel.last_request
    assert len(req.tools) == 1
    assert req.tools[0].function.name == "bearing"
    assert req.tools[0].function.parameters["properties"]["width_m"]
    assert req.tool_choice is None  # omitted -> service default (auto)
    # response parsed to LangChain tool_calls
    assert out.tool_calls == [{"name": "bearing", "args": {"width_m": 2},
                               "id": "call_1", "type": "tool_call"}]


def test_full_tool_loop_message_round_trip(monkeypatch):
    """Assistant tool_calls and TOOL results convert to the SDK shapes."""
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response("qult = 500 kPa")
    m = PalantirSdkChatModel(model_api_name="GPT_5_1")
    history = [
        HumanMessage(content="2 m footing"),
        AIMessage(content="", tool_calls=[
            {"name": "bearing", "args": {"width_m": 2}, "id": "call_1"}]),
        ToolMessage(content='{"q_ult": 500}', tool_call_id="call_1"),
    ]
    out = m.invoke(history)
    assert out.content == "qult = 500 kPa"
    sent = _FakeSdkModel.last_request.messages
    assert [s.role for s in sent] == ["USER", "ASSISTANT", "TOOL"]
    ai = sent[1]
    assert ai.content is None  # empty content omitted on tool-call messages
    assert ai.tool_calls[0].id == "call_1"
    assert ai.tool_calls[0].tool_call.function.name == "bearing"
    assert json.loads(ai.tool_calls[0].tool_call.function.arguments) == {
        "width_m": 2}
    tool_msg = sent[2]
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == '{"q_ult": 500}'


def test_image_call_goes_through_the_vision_door(monkeypatch):
    """An image block reaches the model as Base64ImageContent (it used to be
    flattened away, leaving every vision tool blind on Foundry)."""
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response("Red.")
    m = PalantirSdkChatModel(model_api_name="GPT_5_6_SOL")
    out = m.invoke([HumanMessage(content=[
        {"type": "text", "text": "read this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x",
                                            "detail": "high"}},
    ])])
    assert out.content == "Red."
    assert _OpenAiGptChatWithVisionLanguageModel.got_names == ["GPT_5_6_SOL"]
    assert _FakeSdkModel.last_request is None  # the text door was not used
    req = _FakeVisionModel.last_request
    assert isinstance(req, _GptChatWithVisionCompletionRequest)
    msg = req.messages[0]
    assert msg.role == "USER"
    assert msg.contents[0].text == "read this"
    image = msg.contents[1].image
    assert image.image_url == "data:image/png;base64,x"
    assert image.detail == "HIGH"
    assert _FakeVisionModel.last_retries == 5


def test_image_detail_mapping(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response()
    m = PalantirSdkChatModel(model_api_name="GPT_5_4")

    def detail_for(level):
        image_url = {"url": "data:image/jpeg;base64,y"}
        if level:
            image_url["detail"] = level
        m.invoke([HumanMessage(content=[{"type": "image_url",
                                         "image_url": image_url}])])
        return _FakeVisionModel.last_request.messages[0].contents[0].image.detail

    assert detail_for("low") == "LOW"
    # Not named by this SDK, and HIGH CAPS the image on Foundry (measured
    # 2026-10-02): full resolution is AUTO.
    assert detail_for("original") == "AUTO"
    assert detail_for("high") == "HIGH"       # asked for, so sent as asked
    assert detail_for(None) is None
    # LangChain's own image block shape is read too.
    m.invoke([HumanMessage(content=[{"type": "image", "base64": "zz",
                                     "mime_type": "image/jpeg"}])])
    img = _FakeVisionModel.last_request.messages[0].contents[0].image
    assert img.image_url == "data:image/jpeg;base64,zz"


def test_vision_call_keeps_tools_and_tool_history(monkeypatch):
    """The inline-image shape: an image AFTER an assistant tool call and its
    result, with tools bound."""
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _tool_call_response()
    tool = {"type": "function",
            "function": {"name": "render_region", "description": "zoom",
                         "parameters": {"type": "object", "properties": {}}}}
    m = PalantirSdkChatModel(model_api_name="GPT_5_6_SOL").bind_tools([tool])
    out = m.invoke([
        SystemMessage(content="review"),
        HumanMessage(content="what does the note say?"),
        AIMessage(content="", tool_calls=[
            {"name": "render_region", "args": {"page": 0}, "id": "c1"}]),
        ToolMessage(content='{"view": [0, 0, 10, 10]}', tool_call_id="c1"),
        HumanMessage(content=[
            {"type": "text", "text": "the image you rendered"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,q"}}]),
    ])
    assert out.tool_calls[0]["name"] == "bearing"
    req = _FakeVisionModel.last_request
    assert req.tools[0].function.name == "render_region"
    sent = req.messages
    assert [s.role for s in sent] == ["SYSTEM", "USER", "ASSISTANT", "TOOL",
                                      "USER"]
    ai = sent[2]
    assert ai.tool_calls[0].id == "c1"
    assert json.loads(ai.tool_calls[0].tool_call.function.arguments) == {
        "page": 0}
    assert sent[3].tool_call_id == "c1"
    assert sent[3].contents[0].text == '{"view": [0, 0, 10, 10]}'
    assert sent[4].contents[1].image.image_url == "data:image/png;base64,q"


def test_from_handles_vision_only_serves_every_call(monkeypatch):
    """A transform hands in model objects, not names: with only the vision
    handle, text and tool calls use it too, and bind_tools keeps it."""
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response("via handle")
    handle = _FakeVisionModel()
    m = PalantirSdkChatModel.from_handles("GPT_5_6_SOL", vision_model=handle)
    tool = {"type": "function",
            "function": {"name": "t", "description": "d",
                         "parameters": {"type": "object", "properties": {}}}}
    bound = m.bind_tools([tool])
    out = bound.invoke([HumanMessage(content="plain text")])
    assert out.content == "via handle"
    assert _OpenAiGptChatWithVisionLanguageModel.got_names == []  # no .get()
    assert _OpenAiGptChatLanguageModel.got_names == []
    req = _FakeVisionModel.last_request
    assert isinstance(req, _GptChatWithVisionCompletionRequest)
    assert req.messages[0].contents[0].text == "plain text"
    assert req.tools[0].function.name == "t"
    assert bound.model == "GPT_5_6_SOL"
    with pytest.raises(ValueError):
        PalantirSdkChatModel.from_handles("x")


def test_text_door_without_rate_limit_keyword(monkeypatch):
    """The text door's fake takes no max_rate_limit_retries; the call still
    goes through once without it."""
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response("fine")
    m = PalantirSdkChatModel(model_api_name="GPT_5_1")
    assert m.invoke([HumanMessage(content="hi")]).content == "fine"
    assert m.model == "GPT_5_1"
    assert _FakeVisionModel.last_request is None


# ---------------------------------------------------------------------------
# Engine resolution routing
# ---------------------------------------------------------------------------

_CLEAR = ("ANTHROPIC_API_KEY", "GEOTECH_FOUNDRY_MODELS",
          "GEOTECH_WEBAPP_MODEL", "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
          "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL")


def _foundry_env(monkeypatch):
    for e in _CLEAR:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")


def test_resolve_api_name_uses_sdk_on_foundry(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _foundry_env(monkeypatch)
    import webapp.engine_config as engine_config
    res = engine_config.resolve_engine("GPT_5_1")
    assert res.ok and res.source == "foundry_sdk"
    assert res.model.model_api_name == "GPT_5_1"
    assert "palantir_models" in res.message


def test_resolve_rid_still_uses_proxy_route(monkeypatch):
    _install_fake_sdk(monkeypatch)
    _foundry_env(monkeypatch)
    import webapp.engine_config as engine_config
    res = engine_config.resolve_engine(
        "ri.language-model-service..language-model.gpt-5-1")
    # No token/host set -> the proxy route reports its own config error, and
    # the RID is NOT hijacked by the SDK route.
    assert res.source == "error"
    assert "GEOTECH_FOUNDRY_TOKEN" in res.message or "token" in res.message


def test_resolve_api_name_without_sdk_gives_readable_error(monkeypatch):
    _foundry_env(monkeypatch)
    for name in list(sys.modules):
        if name.startswith(("palantir_models", "language_model_service_api")):
            monkeypatch.delitem(sys.modules, name, raising=False)
    import webapp.engine_config as engine_config
    res = engine_config.resolve_engine("GPT_5_1")
    assert res.source == "error"
    assert "palantir-models" in res.message
    # Foundry-mode wording rule: never mention the Anthropic key.
    assert "ANTHROPIC" not in res.message and "API key" not in res.message


def test_resolve_local_mode_unaffected(monkeypatch):
    for e in _CLEAR + ("GEOTECH_DEPLOYMENT",):
        monkeypatch.delenv(e, raising=False)
    import webapp.engine_config as engine_config
    res = engine_config.resolve_engine()
    assert res.source == "none"
    assert "ANTHROPIC_API_KEY" in res.message
