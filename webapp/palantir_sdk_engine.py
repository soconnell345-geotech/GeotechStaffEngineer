"""``PalantirSdkChatModel`` — Foundry in-platform engine via ``palantir_models``.

The Foundry LLM **proxy** route (``engine_config._resolve_foundry``) needs a
host + bearer token and is subject to the enclave's proxy enrollment (the
observed 401). Code Workspaces expose a second, in-platform door that needs
NEITHER: the ``palantir_models`` SDK, whose auth is handled by the workspace
itself. This module wraps that SDK as a LangChain
:class:`~langchain_core.language_models.chat_models.BaseChatModel` so
``build_deep_agent`` can drive it with native tool calling — mirroring
:class:`funhouse_agent.deep.databricks_bridge.PrompterChatModel` (same
translate → call → translate-back shape).

SDK surface used (signatures verified live on the enclave, 2026-07-21)::

    from palantir_models.models import OpenAiGptChatLanguageModel
    m = OpenAiGptChatLanguageModel.get(model_api_name)   # e.g. "GPT_5_1" — the
                                                         # short API name, NOT
                                                         # the "ri...." RID
    m.create_chat_completion(GptChatCompletionRequest(
        messages,                # List[ChatMessage(role, content, tool_call_id,
                                 #                  tool_calls, ...)]
        max_tokens=..., temperature=..., stop=...,
        tools=[GptTool(function=GptFunctionTool(name, parameters, description))],
    ))

The response mirrors the OpenAI shape (``choices[0].message`` with optional
``tool_calls`` of ``GptToolCall(id, tool_call=GptToolCallInfo(
function=FunctionToolCallInfo(arguments=<json str>, name)))``, plus ``usage``).

**Images (the vision leg).** A call whose messages carry an image block goes
to the SDK's vision door instead — the class names come from the public
``langchain-palantir`` adapter (dragonejt, 2026-02), not yet from a live
introspection here::

    from palantir_models.models import OpenAiGptChatWithVisionLanguageModel
    m.create_chat_completion(completion_request=GptChatWithVisionCompletionRequest(
        messages=[MultiContentChatMessage(role, contents=[
            ChatMessageContent(text=...) |
            ChatMessageContent(image=Base64ImageContent(image_url="data:...")),
        ], tool_calls=..., tool_call_id=...)], tools=..., max_tokens=...),
        max_rate_limit_retries=N)

The image's ``detail`` goes as the SDK's ``ImageDetail`` — with
``original`` sent as AUTO, which is what reaches full resolution here (HIGH
caps the image; see :func:`_image_detail`). This chat route stops at about
the 2048 px level even at AUTO; Foundry's Responses route takes GPT-5.6 Sol
to at least 4096 px. Calls with no image keep the text door verified live on
2026-07-21. Before this leg every image was flattened away and the vision
tools were blind on Foundry.

The SDK is only installed on Foundry, so all SDK imports are lazy (call-time);
this module itself imports cleanly anywhere, and the offline tests fake the SDK
modules in ``sys.modules`` (``webapp/tests/test_palantir_sdk_engine.py``).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

from funhouse_agent.deep.databricks_bridge import (
    _FINISH_REASON_MAP,
    _text_content,
    _to_usage_metadata,
    _usage_to_dict,
)


def _sdk():
    """Import the SDK modules lazily. Raises ImportError off-platform."""
    from palantir_models.models import OpenAiGptChatLanguageModel
    import language_model_service_api.languagemodelservice_api as lms_base
    import language_model_service_api.languagemodelservice_api_completion_v3 \
        as lms_v3
    return OpenAiGptChatLanguageModel, lms_base, lms_v3


def _vision_sdk():
    """The vision door's model class. Raises ImportError where it is absent."""
    from palantir_models.models import OpenAiGptChatWithVisionLanguageModel
    return OpenAiGptChatWithVisionLanguageModel


def sdk_available() -> bool:
    """True when the ``palantir_models`` SDK is importable (i.e. on Foundry)."""
    try:
        _sdk()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LangChain messages  ->  SDK ChatMessage list
# ---------------------------------------------------------------------------

def _lc_messages_to_sdk(messages: Sequence[BaseMessage], lms_base, lms_v3):
    """Translate LangChain messages into SDK ``ChatMessage`` objects."""
    Role = lms_base.ChatMessageRole
    out = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append(lms_base.ChatMessage(Role.SYSTEM,
                                            _text_content(message.content)))
        elif isinstance(message, ToolMessage):
            out.append(lms_base.ChatMessage(
                Role.TOOL, _text_content(message.content),
                tool_call_id=message.tool_call_id))
        elif isinstance(message, AIMessage):
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    lms_v3.GptToolCall(
                        id=tc.get("id") or "",
                        tool_call=lms_v3.GptToolCallInfo(
                            function=lms_v3.FunctionToolCallInfo(
                                arguments=json.dumps(tc.get("args", {}) or {}),
                                name=tc.get("name", ""))))
                    for tc in message.tool_calls
                ]
            # Content: None (omitted) when empty on a pure tool-call message.
            text = _text_content(message.content)
            out.append(lms_base.ChatMessage(Role.ASSISTANT, text or None,
                                            tool_calls=tool_calls))
        else:  # HumanMessage and anything else — send as USER text.
            out.append(lms_base.ChatMessage(Role.USER,
                                            _text_content(message.content)))
    return out


def _image_url(block: dict) -> Optional[str]:
    """The data URI (or URL) of an image content block, else None.

    Takes OpenAI's ``{"type": "image_url", "image_url": {"url": ...}}`` (what
    the app's vision engine, inline images and probes send) and LangChain's
    own ``{"type": "image", "base64"/"data": ..., "mime_type": ...}``.
    """
    kind = block.get("type")
    if kind == "image_url":
        inner = block.get("image_url")
        return inner.get("url") if isinstance(inner, dict) else inner
    if kind == "image":
        if block.get("url"):
            return block["url"]
        data = block.get("base64") or block.get("data")
        if data:
            mime = block.get("mime_type") or "image/png"
            return f"data:{mime};base64,{data}"
    return None


def _has_image(messages: Sequence[BaseMessage]) -> bool:
    for message in messages:
        if isinstance(message.content, list) and any(
                isinstance(b, dict) and _image_url(b) for b in message.content):
            return True
    return False


def _image_detail(block: dict, lms_base) -> Any:
    """OpenAI's ``detail`` string as the SDK's ``ImageDetail``, else None.

    ``original`` — full resolution — has no ``ImageDetail`` member, and the
    backend refuses a raw "ORIGINAL" (400 INVALID_ARGUMENT). On this service
    it is AUTO (or no detail at all) that sends the image at full size, and
    HIGH that CAPS it: measured on Foundry 2026-10-02, GPT-5.6 Sol took 692
    image tokens for a 1024 px and a 2048 px square alike at HIGH (about
    768 px, a tile-style cap) and 1,229 / 4,401 at AUTO; an 11 px printed
    code was misread at HIGH and read exactly at AUTO. So ``original`` goes
    as AUTO, and the vision probe then measures full resolution honoured.
    Any other level the SDK does not name is left out.
    """
    inner = block.get("image_url")
    wanted = (inner.get("detail") if isinstance(inner, dict) else None) \
        or block.get("detail")
    Detail = getattr(lms_base, "ImageDetail", None)
    if not wanted or Detail is None:
        return None
    level = getattr(Detail, str(wanted).upper(), None)
    if level is None and str(wanted).lower() == "original":
        level = getattr(Detail, "AUTO", None)
    return level


def _sdk_contents(content: Any, lms_base) -> list:
    """LangChain content -> ``[ChatMessageContent]``, images kept."""
    Content = lms_base.ChatMessageContent
    if isinstance(content, str):
        return [Content(text=content)]
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(Content(text=block))
        elif isinstance(block, dict):
            if block.get("type") == "text":
                out.append(Content(text=block.get("text") or ""))
            else:
                url = _image_url(block)
                if url:
                    detail = _image_detail(block, lms_base)
                    image = (lms_base.Base64ImageContent(image_url=url,
                                                         detail=detail)
                             if detail is not None else
                             lms_base.Base64ImageContent(image_url=url))
                    out.append(Content(image=image))
    return out or [Content(text="")]


def _lc_messages_to_sdk_vision(messages: Sequence[BaseMessage], lms_base,
                               lms_v3):
    """Translate LangChain messages into ``MultiContentChatMessage`` objects."""
    Role = lms_base.ChatMessageRole
    Multi = lms_base.MultiContentChatMessage
    out = []
    for message in messages:
        contents = _sdk_contents(message.content, lms_base)
        if isinstance(message, SystemMessage):
            out.append(Multi(role=Role.SYSTEM, contents=contents))
        elif isinstance(message, ToolMessage):
            out.append(Multi(role=Role.TOOL, contents=contents,
                             tool_call_id=message.tool_call_id))
        elif isinstance(message, AIMessage):
            tool_calls = [
                lms_v3.GptToolCall(
                    id=tc.get("id") or "",
                    tool_call=lms_v3.GptToolCallInfo(
                        function=lms_v3.FunctionToolCallInfo(
                            arguments=json.dumps(tc.get("args", {}) or {}),
                            name=tc.get("name", ""))))
                for tc in message.tool_calls] or None
            out.append(Multi(role=Role.ASSISTANT, contents=contents,
                             tool_calls=tool_calls))
        else:
            out.append(Multi(role=Role.USER, contents=contents))
    return out


def _openai_tools_to_sdk(openai_tools: list, lms_v3):
    """OpenAI tool-schema dicts (from ``convert_to_openai_tool``) -> GptTool."""
    out = []
    for tool in openai_tools:
        fn = tool.get("function", tool)
        out.append(lms_v3.GptTool(function=lms_v3.GptFunctionTool(
            name=fn.get("name", ""),
            parameters=fn.get("parameters") or {},
            description=fn.get("description"))))
    return out


# ---------------------------------------------------------------------------
# SDK response  ->  LangChain AIMessage
# ---------------------------------------------------------------------------

def _sdk_message_to_ai_message(msg) -> AIMessage:
    """Translate the response ``choices[0].message`` into an AIMessage.

    Read defensively (``getattr``) — the choice-message class differs from the
    request ``ChatMessage`` but mirrors the same OpenAI field names.
    """
    content = getattr(msg, "content", "") or ""
    tool_calls = []
    for tc in getattr(msg, "tool_calls", None) or []:
        info = getattr(tc, "tool_call", None)
        fn = getattr(info, "function", None) if info is not None else None
        # Tolerate a flat OpenAI-style shape too (function directly on the call).
        if fn is None:
            fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        raw_args = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append({"name": name, "args": args,
                           "id": getattr(tc, "id", None), "type": "tool_call"})
    return AIMessage(content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------

class PalantirSdkChatModel(BaseChatModel):
    """LangChain chat model backed by ``palantir_models`` on Foundry.

    Parameters
    ----------
    model_api_name : str
        The Palantir model API name (e.g. ``"GPT_5_1"``) — the short name
        ``OpenAiGptChatLanguageModel.get`` accepts, NOT the ``ri....`` RID.
    max_tokens : int, optional
        Per-response output cap (``None`` = service default).
    temperature : float, optional
        ``None`` (default) OMITS the parameter — GPT-5/reasoning tiers reject
        non-default temperatures.
    rate_limit_retries : int, optional
        Passed as the SDK's ``max_rate_limit_retries`` (dropped if the SDK's
        ``create_chat_completion`` does not take it).

    vision_for_all : bool, optional
        Send EVERY call through the vision door, images or not (one model
        handle serves the whole agent).

    A call carrying an image goes through ``OpenAiGptChatWithVisionLanguageModel``
    (same API name); every other call through ``OpenAiGptChatLanguageModel``.
    Where models arrive as handles rather than names (a Python transform's
    ``OpenAiGptChatWithVisionLanguageModelInput``), use :meth:`from_handles`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_api_name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    rate_limit_retries: Optional[int] = 5
    vision_for_all: bool = False
    # OpenAI-schema tool dicts captured by bind_tools; replayed each call.
    openai_tools: Optional[list] = Field(default=None, exclude=True)

    # The SDK model handles, fetched once on first use (network-free construct).
    _sdk_model: Any = PrivateAttr(default=None)
    _vision_model: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "palantir-models-chat"

    @property
    def model(self) -> str:
        """The API name, for run records that label a model by ``.model``."""
        return self.model_api_name

    @classmethod
    def from_handles(cls, model_api_name: str, vision_model: Any = None,
                     text_model: Any = None, **kwargs: Any):
        """Wrap model objects already in hand (e.g. a transform's model
        inputs). With only ``vision_model``, every call uses it."""
        if vision_model is None and text_model is None:
            raise ValueError("pass vision_model and/or text_model")
        if text_model is None:
            kwargs.setdefault("vision_for_all", True)
        made = cls(model_api_name=model_api_name, **kwargs)
        made._vision_model = vision_model
        made._sdk_model = text_model
        return made

    def _model(self):
        if self._sdk_model is None:
            OpenAiGptChatLanguageModel, _, _ = _sdk()
            self._sdk_model = OpenAiGptChatLanguageModel.get(
                self.model_api_name)
        return self._sdk_model

    def _vision(self):
        if self._vision_model is None:
            self._vision_model = _vision_sdk().get(self.model_api_name)
        return self._vision_model

    def _complete(self, sdk_model, request):
        if self.rate_limit_retries is None:
            return sdk_model.create_chat_completion(request)
        try:
            return sdk_model.create_chat_completion(
                request, max_rate_limit_retries=self.rate_limit_retries)
        except TypeError as exc:
            if "max_rate_limit_retries" not in str(exc):
                raise
            return sdk_model.create_chat_completion(request)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        """Bind tools (standard LangChain pattern) — returns a copy carrying
        the OpenAI tool schemas; ``_generate`` replays them as ``GptTool``s."""
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        return self.model_copy(update={"openai_tools": openai_tools})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _, lms_base, lms_v3 = _sdk()

        request_kwargs: dict = {}
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if stop:
            request_kwargs["stop"] = list(stop)
        tools = kwargs.get("tools") or self.openai_tools
        if tools:
            # tool_choice is omitted -> service default ("auto"), matching the
            # OpenAI behaviour the deep agent expects.
            request_kwargs["tools"] = _openai_tools_to_sdk(tools, lms_v3)

        if self.vision_for_all or _has_image(messages):
            request = lms_v3.GptChatWithVisionCompletionRequest(
                _lc_messages_to_sdk_vision(messages, lms_base, lms_v3),
                **request_kwargs)
            response = self._complete(self._vision(), request)
        else:
            request = lms_v3.GptChatCompletionRequest(
                _lc_messages_to_sdk(messages, lms_base, lms_v3),
                **request_kwargs)
            response = self._complete(self._model(), request)

        choice = response.choices[0]
        ai_message = _sdk_message_to_ai_message(choice.message)

        finish_reason = getattr(choice, "finish_reason", None)
        generation_info = {
            "finish_reason": _FINISH_REASON_MAP.get(finish_reason,
                                                    finish_reason),
        }
        # model_name is REQUIRED alongside usage_metadata for LangChain's usage
        # aggregators to count this call (see PrompterChatModel._generate).
        model_name = getattr(response, "model", None) or self.model_api_name
        if model_name:
            generation_info["model_name"] = model_name

        usage = getattr(response, "usage", None)
        if usage is not None:
            generation_info["usage"] = _usage_to_dict(usage)
            ai_message.usage_metadata = _to_usage_metadata(usage)

        return ChatResult(generations=[
            ChatGeneration(message=ai_message,
                           generation_info=generation_info)])


# Resolve the postponed annotations here, so the class also works when this
# file is loaded by path or exec'd (Foundry glue) rather than imported.
PalantirSdkChatModel.model_rebuild()

__all__ = ["PalantirSdkChatModel", "sdk_available"]
