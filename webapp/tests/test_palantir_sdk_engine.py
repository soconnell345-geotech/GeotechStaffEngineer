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


def _install_fake_sdk(monkeypatch):
    pm = types.ModuleType("palantir_models")
    pm_models = types.ModuleType("palantir_models.models")
    pm_models.OpenAiGptChatLanguageModel = _OpenAiGptChatLanguageModel
    pm.models = pm_models

    lms = types.ModuleType("language_model_service_api")
    base = types.ModuleType(
        "language_model_service_api.languagemodelservice_api")
    base.ChatMessage = _ChatMessage
    base.ChatMessageRole = _ChatMessageRole
    v3 = types.ModuleType(
        "language_model_service_api.languagemodelservice_api_completion_v3")
    v3.GptTool = _GptTool
    v3.GptFunctionTool = _GptFunctionTool
    v3.GptToolCall = _GptToolCall
    v3.GptToolCallInfo = _GptToolCallInfo
    v3.FunctionToolCallInfo = _FunctionToolCallInfo
    v3.GptChatCompletionRequest = _GptChatCompletionRequest
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


def test_multimodal_content_flattened_to_text(monkeypatch):
    _install_fake_sdk(monkeypatch)
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    _FakeSdkModel.next_response = _text_response()
    m = PalantirSdkChatModel(model_api_name="GPT_5_1")
    m.invoke([HumanMessage(content=[
        {"type": "text", "text": "read this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ])])
    assert _FakeSdkModel.last_request.messages[0].content == "read this"


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
