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

Known limitation: user-message content is flattened to TEXT — image blocks from
the vision tools are dropped on this route (the SDK has ``MultiContentChatMessage``
/ ``Base64ImageContent`` for a future vision leg).

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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_api_name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # OpenAI-schema tool dicts captured by bind_tools; replayed each call.
    openai_tools: Optional[list] = Field(default=None, exclude=True)

    # The SDK model handle, fetched once on first use (network-free construct).
    _sdk_model: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "palantir-models-chat"

    def _model(self):
        if self._sdk_model is None:
            OpenAiGptChatLanguageModel, _, _ = _sdk()
            self._sdk_model = OpenAiGptChatLanguageModel.get(
                self.model_api_name)
        return self._sdk_model

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

        request = lms_v3.GptChatCompletionRequest(
            _lc_messages_to_sdk(messages, lms_base, lms_v3), **request_kwargs)
        response = self._model().create_chat_completion(request)

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


__all__ = ["PalantirSdkChatModel", "sdk_available"]
