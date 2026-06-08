"""``PrompterChatModel`` — the Databricks spike for the v5.0 deepagents port.

deepagents / LangGraph drive a LangChain
:class:`~langchain_core.language_models.chat_models.BaseChatModel`. On
Databricks the only sanctioned way to reach an LLM is the Funhouse
``PrompterAPI`` and its OpenAI-compatible client
(``prompter.client.chat.completions.create``). This module bridges the two: a
``BaseChatModel`` whose ``_generate`` drives that client, mirroring the v1
:class:`funhouse_agent.engine.NativeToolEngine` and the native tool-calling loop
in ``funhouse_agent.agent._ask_native``.

Construct it in a Databricks notebook with an already-initialized PrompterAPI::

    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    from funhouse_agent.deep.agent import build_deep_agent

    model = PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")
    agent = build_deep_agent(model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

``build_deep_agent`` default-wraps this same ``model`` with
:class:`funhouse_agent.deep.vision_engine.LangChainVisionEngine`, so the vision
tools route through the OpenAI-compatible vision endpoint too.

Proxy hygiene (critical)
------------------------
The Databricks / Prompter proxy rejects assistant messages that carry null
``function_call`` / ``refusal`` / ``audio`` / ``annotations`` fields — i.e. a
raw pydantic dump of the SDK message object. We therefore translate every
LangChain message into an **explicit plain dict**, and build assistant
``tool_calls`` as explicit ``{id, type, function:{name, arguments}}`` dicts with
no extra fields. This is the exact precaution v1 takes in
``agent._ask_native`` (see its inline comment).

This class cannot be exercised against a live proxy from this dev box. It is
written to be importable and structurally correct, and unit-tested with a fake
``prompter`` whose ``.client.chat.completions.create`` returns a canned
OpenAI-shaped object (see ``tests/test_deep_phase2_offline.py``).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field


# ---------------------------------------------------------------------------
# LangChain message  ->  OpenAI message translation
# ---------------------------------------------------------------------------

def _lc_message_to_openai(message: BaseMessage) -> dict:
    """Translate one LangChain message into an OpenAI-shaped plain dict.

    Assistant (AIMessage) messages with tool calls are emitted with explicit
    ``tool_calls`` dicts and NO null pass-through fields, so they survive the
    Databricks/Prompter proxy (mirrors v1 ``agent._ask_native``).
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": _text_content(message.content)}

    if isinstance(message, HumanMessage):
        # Content may be a plain string or multimodal blocks (vision). The
        # OpenAI-compatible client accepts both; pass the list through verbatim
        # for multimodal, else a plain string.
        content = message.content
        if isinstance(content, str):
            return {"role": "user", "content": content}
        return {"role": "user", "content": content}

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text_content(message.content),
        }

    if isinstance(message, AIMessage):
        out: dict = {"role": "assistant", "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        # OpenAI expects arguments as a JSON STRING.
                        "arguments": json.dumps(tc.get("args", {}) or {}),
                    },
                }
                for tc in tool_calls
            ]
        return out

    # Fallback for any other/base message type — use its declared role.
    role = getattr(message, "type", "user")
    role = {"human": "user", "ai": "assistant"}.get(role, role)
    return {"role": role, "content": _text_content(message.content)}


def _text_content(content) -> str:
    """Coerce LangChain message content to a plain string for OpenAI roles
    that require string content (system/tool)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content is not None else ""


# ---------------------------------------------------------------------------
# OpenAI response  ->  LangChain AIMessage translation
# ---------------------------------------------------------------------------

def _openai_message_to_ai_message(msg) -> AIMessage:
    """Translate an OpenAI response ``message`` into a LangChain AIMessage.

    Populates ``AIMessage.tool_calls`` as a list of
    ``{name, args(dict), id}`` (LangChain's canonical tool-call shape); the
    ``args`` are parsed from the OpenAI ``function.arguments`` JSON string.
    """
    content = getattr(msg, "content", "") or ""
    raw_tool_calls = getattr(msg, "tool_calls", None) or []

    tool_calls = []
    for tc in raw_tool_calls:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        raw_args = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        tool_calls.append(
            {
                "name": name,
                "args": args,
                "id": getattr(tc, "id", None),
                "type": "tool_call",
            }
        )

    return AIMessage(content=content, tool_calls=tool_calls)


_FINISH_REASON_MAP = {
    "stop": "stop",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "length": "length",
    "content_filter": "content_filter",
}


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------

class PrompterChatModel(BaseChatModel):
    """LangChain chat model backed by a Funhouse PrompterAPI OpenAI client.

    Drives ``prompter.client.chat.completions.create(...)`` with native tool
    calling (``tools`` + ``tool_choice="auto"``), mirroring v1
    :class:`funhouse_agent.engine.NativeToolEngine`. Bound tools (from
    ``bind_tools``) are converted to OpenAI tool schemas and replayed on every
    generation.

    Parameters
    ----------
    prompter : PrompterAPI
        Initialized Funhouse PrompterAPI. Only ``.client`` and ``.chat_model``
        are used.
    model : str, optional
        Override model id. If ``None``, ``prompter.chat_model`` is read at each
        call (so a notebook model switch takes effect without rebuilding).
    max_tokens : int, optional
        Max response tokens per call. ``None`` = the API default.
    temperature : float
        Sampling temperature. Defaults to ``0.0``.
    """

    # BaseChatModel is a pydantic model; allow the arbitrary PrompterAPI object
    # and stored tool dicts as fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompter: Any = Field(default=None, exclude=True)
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    # Optional so ``temperature=None`` can OMIT the param (GPT-5 / reasoning
    # tiers that reject a fixed temperature). Defaults to 0.0 for determinism.
    temperature: Optional[float] = 0.0
    # OpenAI-schema tool dicts captured by bind_tools; replayed each call.
    openai_tools: Optional[list] = Field(default=None, exclude=True)
    tool_choice: Any = None

    @property
    def _llm_type(self) -> str:
        return "funhouse-prompter-chat"

    @property
    def _active_model(self) -> str:
        """The model id to send — override, else live from the PrompterAPI."""
        if self.model:
            return self.model
        return getattr(self.prompter, "chat_model", None)

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ):
        """Bind tools to the model (standard LangChain pattern).

        Converts each tool (StructuredTool / pydantic / dict / callable) to an
        OpenAI tool schema via
        :func:`langchain_core.utils.function_calling.convert_to_openai_tool`,
        stores them, and returns a COPY of this model carrying the schemas.
        ``_generate`` replays them as the OpenAI ``tools`` param with
        ``tool_choice="auto"``.
        """
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        # Return a copy so binding does not mutate the shared model instance,
        # matching how LangChain integrations implement bind_tools.
        return self.model_copy(
            update={
                "openai_tools": openai_tools,
                "tool_choice": tool_choice if tool_choice is not None
                else self.tool_choice,
            }
        )

    def _build_request(self, messages: list[BaseMessage], **kwargs) -> dict:
        """Assemble the OpenAI ``chat.completions.create`` kwargs."""
        openai_messages = [_lc_message_to_openai(m) for m in messages]
        request: dict = {
            "model": kwargs.get("model") or self._active_model,
            "messages": openai_messages,
        }
        # Some models (e.g. GPT-5 / reasoning tiers such as ``funhouse-gpt-high``)
        # reject a non-default ``temperature``. ``temperature=None`` omits it so
        # the model uses its own default; ``_create_with_param_fallback`` also
        # strips it automatically if the API still complains.
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            request["temperature"] = temperature

        tools = kwargs.get("tools") or self.openai_tools
        if tools:
            request["tools"] = tools
            # Default to "auto" (mirrors NativeToolEngine / _ask_native).
            request["tool_choice"] = (
                kwargs.get("tool_choice")
                or self.tool_choice
                or "auto"
            )

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        return request

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Translate -> call the OpenAI-compatible client -> translate back."""
        request = self._build_request(messages, **kwargs)
        if stop:
            request["stop"] = stop

        response = self._create_with_param_fallback(request)

        choice = response.choices[0]
        ai_message = _openai_message_to_ai_message(choice.message)

        finish_reason = getattr(choice, "finish_reason", None)
        generation_info = {
            "finish_reason": _FINISH_REASON_MAP.get(finish_reason, finish_reason),
        }
        # ``model_name`` flows into ``AIMessage.response_metadata`` and is REQUIRED
        # by ``UsageMetadataCallbackHandler`` — it only records a message's usage
        # when BOTH ``usage_metadata`` AND ``response_metadata["model_name"]`` are
        # present (see langchain_core.callbacks.usage). Without it the callback
        # aggregator silently drops this call's tokens.
        model_name = getattr(response, "model", None) or request.get("model")
        if model_name:
            generation_info["model_name"] = model_name

        # Surface token usage when the proxy returns it.
        usage = getattr(response, "usage", None)
        if usage is not None:
            generation_info["usage"] = _usage_to_dict(usage)
            # ALSO set the LangChain ``usage_metadata`` so the standard
            # aggregators (``_v2_usage`` and ``get_usage_metadata_callback``) see
            # this call's tokens — the proxy path previously set only the
            # generation_info ``usage`` blob, which those aggregators ignore.
            ai_message.usage_metadata = _to_usage_metadata(usage)

        generation = ChatGeneration(
            message=ai_message,
            generation_info=generation_info,
        )
        return ChatResult(generations=[generation])

    def _create_with_param_fallback(self, request: dict):
        """Call the OpenAI-compatible client, retrying ONCE if the model rejects
        a parameter.

        The first attempt sends the request as-is (the known-good path for
        OpenAI/Anthropic via the proxy). If it fails with a parameter error —
        e.g. a GPT-5 / reasoning model (``funhouse-gpt-high``) that rejects
        ``temperature`` or wants ``max_completion_tokens`` instead of
        ``max_tokens`` — the offending parameter is dropped/renamed and the call
        is retried once. Any other error (or a second failure) propagates.
        """
        create = self.prompter.client.chat.completions.create
        try:
            return create(**request)
        except Exception as exc:  # noqa: BLE001 - inspect + selectively retry
            adjusted = _adjust_request_for_param_error(request, str(exc))
            if adjusted is None:
                raise
            return create(**adjusted)


def _adjust_request_for_param_error(request: dict, message: str):
    """Return a retry request with the offending parameter removed/renamed, or
    ``None`` if the error does not look parameter-related.

    Handles the two common modern-model quirks: a rejected ``temperature``
    (dropped) and ``max_tokens`` that must be ``max_completion_tokens``
    (renamed). Detection is a substring match on the lower-cased error message.
    """
    low = (message or "").lower()
    new = dict(request)
    changed = False
    if "temperature" in low and "temperature" in new:
        new.pop("temperature", None)
        changed = True
    if (("max_completion_tokens" in low or "max_tokens" in low)
            and "max_tokens" in new):
        new["max_completion_tokens"] = new.pop("max_tokens")
        changed = True
    return new if changed else None


def _usage_to_dict(usage) -> dict:
    """Best-effort conversion of an OpenAI usage object to a plain dict."""
    if isinstance(usage, dict):
        return usage
    out = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = getattr(usage, key, None)
        if val is not None:
            out[key] = val
    return out


def _usage_field(usage, key: str):
    """Read ``key`` off an OpenAI-style usage object OR plain dict, else None."""
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _to_usage_metadata(usage) -> Optional[dict]:
    """Map an OpenAI-style usage object/dict to LangChain's ``UsageMetadata`` shape.

    LangChain's :class:`~langchain_core.messages.ai.UsageMetadata` is
    ``{"input_tokens", "output_tokens", "total_tokens"}``. OpenAI reports
    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``; the Funhouse
    proxy MAY report only a combined ``total_tokens`` with no in/out split.

    Parameters
    ----------
    usage : object or dict or None
        An OpenAI-style usage object (attributes) or dict, or ``None``.

    Returns
    -------
    dict or None
        ``{"input_tokens", "output_tokens", "total_tokens"}`` covering all the
        cases below, or ``None`` when no usage information is available at all:

        * ``prompt_tokens`` -> ``input_tokens``, ``completion_tokens`` ->
          ``output_tokens``, ``total_tokens`` -> ``total_tokens``.
        * If a split (prompt/completion) is present but no total, the total is
          computed as ``input + output``.
        * If ONLY a combined total is available (no prompt/completion), the
          owner cares about the TOTAL, so set ``input_tokens = total``,
          ``output_tokens = 0``, ``total_tokens = total`` — this way any
          aggregator that sums ``input + output`` still yields the correct
          TOTAL (the in/out split is best-effort, the total is authoritative).
    """
    if usage is None:
        return None

    prompt = _usage_field(usage, "prompt_tokens")
    completion = _usage_field(usage, "completion_tokens")
    total = _usage_field(usage, "total_tokens")

    has_split = prompt is not None or completion is not None
    if has_split:
        inp = int(prompt or 0)
        out = int(completion or 0)
        tot = int(total) if total is not None else inp + out
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}

    if total is not None:
        # Combined-total-only (the Funhouse proxy case): preserve the TOTAL while
        # keeping any input+output aggregator correct.
        tot = int(total)
        return {"input_tokens": tot, "output_tokens": 0, "total_tokens": tot}

    return None


__all__ = ["PrompterChatModel"]
