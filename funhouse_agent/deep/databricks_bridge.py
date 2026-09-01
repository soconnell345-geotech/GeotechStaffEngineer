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
import os
import warnings
from itertools import chain
from typing import Any, Iterator, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr

try:  # openai ships with the app (langchain-openai is a core dependency)
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover - openai is always present in practice
    _OpenAI = None


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
    streaming_enabled : bool
        Opt-in token streaming (default ``False``). The env var
        ``GEOTECH_PROMPTER_STREAMING`` ("1"/"true") also enables it at call
        time, so the Databricks launcher env passes through with no wiring.

    Token streaming (``_stream``) — design + the two landmines
    ----------------------------------------------------------
    Reasoning models (``funhouse-gpt-high``) can think for minutes; without
    streaming the Streamlit websocket sits silent through the Databricks
    driver proxy for the whole call. ``_stream`` drives
    ``chat.completions.create(stream=True)`` and yields
    :class:`~langchain_core.outputs.ChatGenerationChunk` deltas (text content
    plus ``tool_call_chunks`` carrying the OpenAI per-index fragments, which
    LangChain's chunk-merging reassembles into full ``tool_calls``).

    **Landmine 1 — the SDK's ``collect_usage`` injection.** Funhouse's
    ``wrap_all_openai_methods`` (prompter_api.py) wraps every method on
    ``prompter.client`` and injects ``kwargs["collect_usage"] = True``
    whenever ``stream`` is truthy; the OpenAI SDK rejects that kwarg with a
    ``TypeError``. Streaming therefore NEVER goes through the wrapped
    ``prompter.client``: a CLEAN unwrapped ``OpenAI`` client is built once
    (reusing ``prompter.http_client`` — so NTLM auth rides along — plus
    ``prompter.base_url``/``api_key``/``max_retries``, exactly mirroring the
    PrompterAPI's own construction) and cached on the instance. If those
    attributes are missing, ``create.__wrapped__`` (the pre-wrap original) is
    used instead.

    **Landmine 2 — the metering obligation.** Bypassing the wrapper skips the
    SDK's usage logging (a terms-of-use obligation). The stream request sends
    ``stream_options={"include_usage": True}``; the final usage-only chunk is
    surfaced as LangChain ``usage_metadata`` AND best-effort forwarded to
    ``prompter.logger.meter_log(service="Prompter", ...)`` with the same
    flattened-usage metrics the wrapper would log. Metering failures are
    swallowed — they must never break a turn.

    **Robustness.** Streaming is opt-in and default-OFF. When disabled, or
    when stream setup / the first-chunk fetch raises for ANY reason,
    ``_stream`` falls back to ``_generate`` and yields its result as a single
    chunk (with a ``warnings.warn`` note in the failure case) — a turn never
    breaks because streaming did. Mid-stream errors after the first chunk
    propagate (the turn is already partially delivered).
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
    # Opt-in token streaming (see class docstring). Default OFF; the env var
    # GEOTECH_PROMPTER_STREAMING ("1"/"true") also enables it at call time.
    streaming_enabled: bool = False
    # Cached CLEAN (unwrapped) create callable for streaming — built lazily
    # once per instance (see _streaming_create).
    _stream_create: Any = PrivateAttr(default=None)

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

    # ------------------------------------------------------------------
    # Token streaming (see class docstring for the design + landmines)
    # ------------------------------------------------------------------

    def _streaming_active(self) -> bool:
        """Streaming is on when the field says so OR the env var enables it
        (checked at call time so a launcher env passes through unwired)."""
        if self.streaming_enabled:
            return True
        env = os.environ.get("GEOTECH_PROMPTER_STREAMING", "")
        return env.strip().lower() in ("1", "true", "yes", "on")

    def _streaming_create(self):
        """Return (and cache) a CLEAN ``create`` callable for streaming.

        The Funhouse SDK's ``wrap_all_openai_methods`` injects
        ``collect_usage=True`` into any wrapped call with ``stream`` truthy,
        which the OpenAI SDK rejects (TypeError). So streaming uses an
        UNWRAPPED client, built once mirroring PrompterAPI's own construction
        (same http_client → same NTLM auth, base_url, api_key, max_retries).
        Falls back to ``create.__wrapped__`` when the prompter doesn't expose
        those attributes (e.g. fakes/other backends).
        """
        if self._stream_create is not None:
            return self._stream_create

        prompter = self.prompter
        http_client = getattr(prompter, "http_client", None)
        base_url = getattr(prompter, "base_url", None)
        create = None
        if _OpenAI is not None and http_client is not None and base_url:
            clean_client = _OpenAI(
                http_client=http_client,
                base_url=base_url,
                # NTLM mode: placeholder key the service never reads (the real
                # credentials ride the http_client's NTLM handshake). Key-auth
                # mode: reuse the configured key.
                api_key=getattr(prompter, "api_key", None)
                or "PrompterOpenAI-Databricks",
                max_retries=getattr(prompter, "max_retries", None) or 1,
            )
            create = clean_client.chat.completions.create
        else:
            wrapped = prompter.client.chat.completions.create
            create = getattr(wrapped, "__wrapped__", wrapped)

        self._stream_create = create
        return create

    def _generate_as_single_chunk(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        """Run the non-streaming ``_generate`` and repackage its result as one
        ChatGenerationChunk (the streaming-disabled / setup-failure fallback)."""
        result = self._generate(messages, stop=stop, run_manager=run_manager,
                                **kwargs)
        generation = result.generations[0]
        message = generation.message
        tool_call_chunks = [
            {
                "name": tc.get("name"),
                "args": json.dumps(tc.get("args", {}) or {}),
                "id": tc.get("id"),
                "index": i,
                "type": "tool_call_chunk",
            }
            for i, tc in enumerate(getattr(message, "tool_calls", None) or [])
        ]
        chunk_message = AIMessageChunk(
            content=message.content or "",
            tool_call_chunks=tool_call_chunks,
            usage_metadata=getattr(message, "usage_metadata", None),
            response_metadata=dict(message.response_metadata or {}),
        )
        return ChatGenerationChunk(
            message=chunk_message,
            generation_info=generation.generation_info,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream deltas from the OpenAI-compatible client.

        NEVER breaks a turn: when streaming is disabled (default) or stream
        setup / the first-chunk fetch fails, delegates to ``_generate`` and
        yields its result as a single chunk. Mid-stream errors after the first
        chunk propagate.
        """
        if not self._streaming_active():
            yield self._generate_as_single_chunk(
                messages, stop, run_manager, **kwargs)
            return

        request = self._build_request(messages, **kwargs)
        if stop:
            request["stop"] = stop
        request["stream"] = True
        # Terms obligation: ask the proxy for the final usage chunk so the
        # wrapper-bypassed call can still be metered (see _meter_streamed_usage).
        request["stream_options"] = {"include_usage": True}

        try:
            create = self._streaming_create()
            raw_stream = iter(create(**request))
            # Pull the first chunk inside the guard — connection/param errors
            # from lazy iterators surface here, while the fallback is still safe.
            try:
                pending = [next(raw_stream)]
            except StopIteration:
                pending = []
        except Exception as exc:  # noqa: BLE001 - any setup failure → fallback
            warnings.warn(
                "Prompter streaming failed during setup "
                f"({type(exc).__name__}: {exc}); falling back to the "
                "non-streaming call.",
                stacklevel=2,
            )
            yield self._generate_as_single_chunk(
                messages, stop, run_manager, **kwargs)
            return

        model_name: Optional[str] = None
        final_usage = None
        usage_obj_type = "ChatCompletionChunk"

        for raw in chain(pending, raw_stream):
            if model_name is None:
                model_name = getattr(raw, "model", None)
            usage = getattr(raw, "usage", None)
            if usage is not None:
                # The usage-only final chunk (empty choices) — or a proxy that
                # attaches usage to the last delta. Yielded after the loop.
                final_usage = usage
                usage_obj_type = type(raw).__name__
            choices = getattr(raw, "choices", None) or []
            if not choices:
                continue
            choice = choices[-1]
            delta = getattr(choice, "delta", None)
            finish_reason = getattr(choice, "finish_reason", None)
            text = getattr(delta, "content", None) if delta is not None else None
            raw_tool_calls = (
                getattr(delta, "tool_calls", None) if delta is not None else None
            )

            # OpenAI tool-call deltas: index + partial id/name/arguments
            # fragments. Pass them through as tool_call_chunks — LangChain's
            # AIMessageChunk merging accumulates them by index into full
            # tool_calls on the summed message.
            tool_call_chunks = []
            for tc in raw_tool_calls or []:
                fn = getattr(tc, "function", None)
                tool_call_chunks.append(
                    {
                        "name": getattr(fn, "name", None) if fn is not None
                        else None,
                        "args": getattr(fn, "arguments", None) if fn is not None
                        else None,
                        "id": getattr(tc, "id", None),
                        "index": getattr(tc, "index", None) or 0,
                        "type": "tool_call_chunk",
                    }
                )

            if not text and not tool_call_chunks and finish_reason is None:
                continue  # empty keep-alive delta

            generation_info = None
            response_metadata: dict = {}
            if finish_reason is not None:
                generation_info = {
                    "finish_reason": _FINISH_REASON_MAP.get(
                        finish_reason, finish_reason),
                }
                mn = model_name or request.get("model")
                if mn:
                    generation_info["model_name"] = mn
                response_metadata = dict(generation_info)

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=text or "",
                    tool_call_chunks=tool_call_chunks,
                    response_metadata=response_metadata,
                ),
                generation_info=generation_info,
            )
            if run_manager is not None and text:
                run_manager.on_llm_new_token(text, chunk=chunk)
            yield chunk

        if final_usage is not None:
            mn = model_name or request.get("model")
            usage_metadata = _to_usage_metadata(final_usage)
            # NOTE: no model_name here — it already rode the finish_reason
            # chunk, and LangChain's chunk merging CONCATENATES duplicate
            # response_metadata strings ("fake-gptfake-gpt").
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    usage_metadata=usage_metadata,
                ),
            )
            self._meter_streamed_usage(final_usage, mn, usage_obj_type)

    def _meter_streamed_usage(self, usage, model_name, obj_type) -> None:
        """Best-effort usage metering for the wrapper-bypassed streaming call.

        Mirrors what ``wrap_openai_method`` logs for non-streaming calls:
        ``logger.meter_log(service="Prompter", operation=<object type>,
        metrics=<flattened usage + model + object>)``. Any failure here is
        swallowed — metering must never break a turn.
        """
        try:
            logger = getattr(self.prompter, "logger", None)
            if logger is None or usage is None:
                return
            metrics = _flatten_usage_metrics(usage)
            if model_name:
                metrics["model"] = model_name
            if obj_type:
                metrics["object"] = obj_type
            logger.meter_log(service="Prompter", operation=obj_type,
                             metrics=metrics)
        except Exception:  # noqa: BLE001 - metering is strictly best-effort
            pass


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


def _flatten_usage_metrics(usage) -> dict:
    """Flatten an OpenAI-style usage object/dict into the metrics dict the
    Funhouse wrapper logs (mirrors its ``flatten_dict(dict(response.usage))``:
    nested dicts are joined with underscores)."""
    if isinstance(usage, dict):
        data = dict(usage)
    elif hasattr(usage, "model_dump"):
        data = usage.model_dump()
    else:
        try:
            data = dict(usage)
        except (TypeError, ValueError):
            data = {k: v for k, v in vars(usage).items()
                    if not k.startswith("_")}

    def _flatten(d: dict, parent: str = "") -> dict:
        items: dict = {}
        for key, val in d.items():
            new_key = f"{parent}_{key}" if parent else str(key)
            if isinstance(val, dict):
                items.update(_flatten(val, new_key))
            else:
                items[new_key] = val
        return items

    return _flatten(data)


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
