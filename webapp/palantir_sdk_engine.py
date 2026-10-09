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

**The Responses route** (``route="responses"``, the default where the SDK
offers it). On Foundry GPT-5.4 is served by a Bedrock backend that refuses
both chat doors (404 LanguageModelNotAvailable) and answers only the
language-model service's Responses request type
(``CompletionRequestV3.open_ai_responses``); that route also takes GPT-5.6
Sol's images to at least 4096 px where the chat door stops near 2048. Folded
in from the AI FDE's glue (``foundry_responses_model.py``, 2026-10-02), which
ran every Foundry suite run of 5.32. It reaches the service through the SDK's
own private helpers (``palantir_models.models._lms``), so where those are
missing the chat doors are used instead (``route="auto"``).

**Reasoning summaries** (since the Foundry brief 4 review, 2026-10-07). Each
Responses request asks for a summary of the model's reasoning where the SDK
can say so (``GEOTECH_FOUNDRY_REASONING_SUMMARY``: ``auto`` by default,
``off`` to stop), and the summary the result carries goes on the reply as
``additional_kwargs["reasoning"]``, which the activity log records on
``model_end``. A service that refuses the setting is asked once more without
it, and not asked again.

**Retries.** A Foundry call can drop its connection (Sol's keep-alive drops
came in pairs), time out on a long reasoning call or hit the project's
token-per-minute limit. Those — and only those (connection, timeout, rate
limit, HTTP 503) — are retried with jittered exponential backoff, at most
``retry_total_s`` of waiting per call; anything else (a bad request, a
content filter, a context too long) is raised at once.

The SDK is only installed on Foundry, so all SDK imports are lazy (call-time);
this module itself imports cleanly anywhere, and the offline tests fake the SDK
modules in ``sys.modules`` (``webapp/tests/test_palantir_sdk_engine.py``).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Callable, Optional, Sequence

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


log = logging.getLogger(__name__)

ROUTES = ("auto", "chat", "responses")


def _responses_sdk():
    """The Responses request types and the SDK helpers that send them.
    Raises ImportError (or AttributeError) where this SDK lacks them."""
    import language_model_service_api.languagemodelservice_api_completion_v3_responses as r  # noqa: E501
    from language_model_service_api.languagemodelservice_api_completion_v3 \
        import CompletionRequestV3, CreateCompletionRequest
    from palantir_models.models._lms import (_create_completion,
                                             _run_lms_request_with_retries)
    return (r, CompletionRequestV3, CreateCompletionRequest,
            _create_completion, _run_lms_request_with_retries)


def responses_available() -> bool:
    """True when the SDK can send Responses requests."""
    try:
        _responses_sdk()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Retries: infrastructure failures only
# ---------------------------------------------------------------------------

_RATE_LIMIT_ERRORS = ("RateLimitsExceeded", "AzurePortalQosException",
                      "HubQosException")
_CONNECTION_TYPES = ("ConnectionError", "RemoteDisconnected", "ProtocolError",
                     "ConnectionResetError", "ConnectionAbortedError",
                     "ChunkedEncodingError", "IncompleteRead")
_TIMEOUT_TYPES = ("Timeout", "ReadTimeout", "ReadTimeoutError",
                  "ConnectTimeout", "TimeoutError", "socket.timeout")


def transient_kind(exc: BaseException) -> Optional[str]:
    """``rate_limit`` / ``timeout`` / ``connection`` / ``http_503`` for a
    failure worth retrying, else ``None`` (raise it)."""
    for e in (exc, getattr(exc, "__cause__", None)):
        if e is None:
            continue
        error_name = str(getattr(e, "error_name", None)
                         or getattr(e, "_error_name", None) or "")
        if any(n in error_name for n in _RATE_LIMIT_ERRORS):
            return "rate_limit"
        if "LlmSocketTimeout" in error_name:
            return "timeout"
        response = (getattr(e, "response", None)
                    or getattr(getattr(e, "_cause", None), "response", None))
        status = getattr(response, "status_code", None)
        if status == 429:
            return "rate_limit"
        if status == 503:
            return "http_503"
        name = type(e).__name__
        if name in _TIMEOUT_TYPES or "ReadTimeout" in str(e)[:500]:
            return "timeout"
        if name in _CONNECTION_TYPES or isinstance(e, ConnectionError):
            return "connection"
    return None


def call_with_retries(fn: Callable[[], Any], *, total_s: float = 300.0,
                      max_sleep_s: float = 60.0,
                      sleep: Callable[[float], None] = time.sleep) -> Any:
    """``fn()``, retried on :func:`transient_kind` failures with full-jitter
    exponential backoff until ``total_s`` of waiting is spent."""
    attempt, waited = 0, 0.0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            kind = transient_kind(exc)
            if kind is None:
                raise
            attempt += 1
            wait = random.uniform(0.0, min(max_sleep_s, 2.0 ** attempt))
            if kind == "rate_limit":
                wait = max(wait, random.uniform(2.0, 6.0))
            if waited + wait > total_s:
                raise
            log.info("Foundry call failed (%s, attempt %d): %s; retrying in "
                     "%.1f s", kind, attempt, str(exc)[:200], wait)
            sleep(wait)
            waited += wait


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
# LangChain messages  ->  Responses input items (the AI FDE's glue, folded in)
# ---------------------------------------------------------------------------

def _responses_contents(content: Any, r) -> list:
    """LangChain content -> ``[InputMessageContent]``, images kept (with
    ``original`` sent as AUTO, as on the chat door)."""
    if isinstance(content, str):
        return [r.InputMessageContent(text=content)]
    out = []
    for block in content or []:
        if isinstance(block, str):
            out.append(r.InputMessageContent(text=block))
        elif isinstance(block, dict):
            if block.get("type") == "text":
                out.append(r.InputMessageContent(text=block.get("text") or ""))
            else:
                url = _image_url(block)
                if url:
                    detail = _image_detail(block, r)
                    image = (r.Base64ImageContent(image_url=url, detail=detail)
                             if detail is not None else
                             r.Base64ImageContent(image_url=url))
                    out.append(r.InputMessageContent(image=image))
    return out or [r.InputMessageContent(text="")]


def _responses_message(role, contents, r):
    return r.ResponsesInput(item=r.ResponsesItem(
        input_message=r.InputMessage(content=contents, role=role)))


def _lc_messages_to_responses(messages: Sequence[BaseMessage], r) -> list:
    """System / user messages as input messages, an assistant turn as its
    text plus one function-call item per tool call, a tool result as a
    function-call-output item."""
    Role = r.InputMessageRole
    out = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append(_responses_message(
                Role.SYSTEM,
                [r.InputMessageContent(text=_text_content(message.content))],
                r))
        elif isinstance(message, ToolMessage):
            out.append(r.ResponsesInput(item=r.ResponsesItem(
                function_tool_call_output=r.FunctionToolCallOutput(
                    call_id=message.tool_call_id or "",
                    output=_text_content(message.content)))))
        elif isinstance(message, AIMessage):
            text = _text_content(message.content)
            if text:
                out.append(r.ResponsesInput(input_message=r.EasyInputMessage(
                    content=r.EasyInputMessageContent(text=text),
                    role=Role.ASSISTANT)))
            for tc in message.tool_calls or []:
                out.append(r.ResponsesInput(item=r.ResponsesItem(
                    function_tool_call=r.FunctionToolCall(
                        arguments=json.dumps(tc.get("args", {}) or {}),
                        call_id=tc.get("id") or "",
                        name=tc.get("name", "")))))
        else:
            out.append(_responses_message(
                Role.USER, _responses_contents(message.content, r), r))
    return out


#: Ask the Responses route for a SUMMARY of the model's reasoning on every
#: call, and keep it (``additional_kwargs["reasoning"]``, which the activity
#: log writes on ``model_end``). Foundry brief 4 (2026-10-07): the primary
#: agent wrote no text on any of ~300 tool-calling steps in 49 runs and no
#: reasoning summary was requested, so why GPT-5.4 ignored the tile that
#: found a tag, or dropped two tags, could only be read from behaviour.
#: ``auto`` (default), ``concise`` or ``detailed``; ``off`` asks for none.
REASONING_SUMMARY_ENV = "GEOTECH_FOUNDRY_REASONING_SUMMARY"
DEFAULT_REASONING_SUMMARY = "auto"

#: The SDK's names for the request's reasoning settings and for the summary
#: level (conjure-generated from OpenAI's ``reasoning: {summary: ...}``).
#: Foundry's SDK (setup check ``sdk_reasoning_types``, brief 5, 2026-10-08)
#: names the request field ``ReasoningConfig(effort, summary)`` and the level
#: enum ``SummaryConfig`` (AUTO / CONCISE / DETAILED / UNKNOWN); there
#: ``Reasoning`` and ``ReasoningSummary`` are OUTPUT classes (the result's
#: reasoning item and its text), and ``ReasoningSummary`` has no AUTO. Taking
#: the first name that merely EXISTS picked those two, so no summary was ever
#: asked for (0 of 1,873 model calls). :func:`_reasoning_request` therefore
#: PROBES: the first summary type that has the level, then the first request
#: class that can be built with it. The older guesses stay behind the real
#: names for an SDK that has only them.
_REASONING_TYPES = ("ReasoningConfig", "Reasoning", "ResponsesReasoning",
                    "OpenAiResponsesReasoning", "ReasoningParams")
_SUMMARY_TYPES = ("SummaryConfig", "ReasoningSummary", "ReasoningSummaryType",
                  "ReasoningSummaryMode", "Summary")


def reasoning_summary_level() -> Optional[str]:
    """The summary level to ask for (:data:`REASONING_SUMMARY_ENV`), or
    ``None`` to ask for none."""
    raw = str(os.environ.get(REASONING_SUMMARY_ENV,
                             DEFAULT_REASONING_SUMMARY)).strip().lower()
    return None if raw in ("", "off", "none", "0", "false", "no") else raw


def _reasoning_request(r, level: Optional[str]) -> Any:
    """The SDK object asking for a reasoning summary at ``level``, or
    ``None`` where this SDK has no way to ask (then nothing is sent)."""
    if not level:
        return None
    want = level.upper()
    value = None
    for name in _SUMMARY_TYPES:
        kinds = getattr(r, name, None)
        got = getattr(kinds, want, None) if kinds is not None else None
        if got is not None:
            value = got
            break
    if value is None:
        return None
    for name in _REASONING_TYPES:
        cls = getattr(r, name, None)
        if cls is None:
            continue
        for kwargs in ({"summary": value}, {"effort": None, "summary": value}):
            try:
                return cls(**kwargs)
            except (TypeError, ValueError):
                continue
    return None


def _output_fields(resp) -> list:
    """What each output item of a Responses result carries (its set field
    names), for the one log line written when a summary was asked for and
    none could be read — so the first live call says where it is."""
    out = []
    for item in getattr(resp, "output", None) or []:
        try:
            fields = sorted(k.lstrip("_") for k, v in vars(item).items()
                            if v is not None)
        except TypeError:
            fields = []
        out.append(f"{type(item).__name__}({', '.join(fields)})")
    return out


def _reasoning_text(resp) -> str:
    """The reasoning summary a Responses result carries: every output item
    of the reasoning kind, its summary parts' text joined. Read defensively
    — the SDK's item is a union with one field set."""
    parts = []
    for item in getattr(resp, "output", None) or []:
        rs = getattr(item, "reasoning", None)
        if rs is None:
            continue
        for s in getattr(rs, "summary", None) or []:
            text = getattr(s, "text", None)
            if text is None:
                inner = getattr(s, "summary_text", None)
                text = getattr(inner, "text", None) if inner is not None \
                    else None
            text = getattr(text, "text", text)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


def _refused_reasoning(exc: BaseException) -> bool:
    """Whether a failed call reads as the service refusing the reasoning
    setting (a bad-request naming it, or an invalid-argument error)."""
    text = str(exc).lower()
    name = str(getattr(exc, "error_name", None)
               or getattr(exc, "_error_name", None) or "")
    return ("reasoning" in text or "summary" in text
            or "InvalidArgument" in name or "invalid_argument" in text)


def _openai_tools_to_responses(openai_tools: list, r) -> list:
    out = []
    for tool in openai_tools:
        fn = tool.get("function", tool)
        out.append(r.Tool(function=r.FunctionTool(
            name=fn.get("name", ""),
            parameters=fn.get("parameters") or {"type": "object",
                                                "properties": {}},
            strict=False, description=fn.get("description"))))
    return out


def _responses_to_ai_message(resp) -> AIMessage:
    texts, tool_calls = [], []
    for item in getattr(resp, "output", None) or []:
        message = getattr(item, "output_message", None)
        if message is not None:
            for c in getattr(message, "content", None) or []:
                text = getattr(c, "text", None)
                if text is not None:
                    texts.append(getattr(text, "text", text) or "")
        fc = getattr(item, "function_tool_call", None)
        if fc is not None:
            try:
                args = json.loads(fc.arguments) if fc.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"name": fc.name, "args": args,
                               "id": fc.call_id, "type": "tool_call"})
    return AIMessage(content="".join(texts), tool_calls=tool_calls)


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
    route : str, optional
        ``"responses"`` (the Responses request type: the only route GPT-5.4
        answers on Foundry, and full-size images for GPT-5.6 Sol),
        ``"chat"`` (the chat / chat-with-vision doors), or ``"auto"``
        (default: Responses where the SDK can send it, else chat).
    retry_total_s : float, optional
        Most seconds spent waiting between retries of one call (connection
        drops, timeouts, rate limits, 503 only).
    read_timeout_s : float, optional
        Raise the SDK client's HTTP read timeout to this (long reasoning
        calls); ``None`` leaves it.

    On the chat route a call carrying an image goes through
    ``OpenAiGptChatWithVisionLanguageModel`` (same API name); every other call
    through ``OpenAiGptChatLanguageModel``. Where models arrive as handles
    rather than names (a Python transform's
    ``OpenAiGptChatWithVisionLanguageModelInput``), use :meth:`from_handles`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_api_name: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    rate_limit_retries: Optional[int] = 5
    vision_for_all: bool = False
    route: str = "auto"
    retry_total_s: float = 300.0
    read_timeout_s: Optional[float] = 900.0
    # OpenAI-schema tool dicts captured by bind_tools; replayed each call.
    openai_tools: Optional[list] = Field(default=None, exclude=True)

    # The SDK model handles, fetched once on first use (network-free construct).
    _sdk_model: Any = PrivateAttr(default=None)
    _vision_model: Any = PrivateAttr(default=None)
    # Set once the service has refused the reasoning-summary setting, so it
    # is not asked for again (one refused call, not one per call).
    _reasoning_refused: bool = PrivateAttr(default=False)
    # Set once the "summary asked for, none read" line has been logged.
    _reasoning_fields_logged: bool = PrivateAttr(default=False)

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

    def _raise_read_timeout(self, handle) -> None:
        """Long reasoning calls outlast the SDK client's default read
        timeout; raise it where the handle exposes its client."""
        if not self.read_timeout_s:
            return
        service = getattr(handle, "_llm_service", None)
        old = getattr(service, "_read_timeout", None)
        if isinstance(old, (int, float)) and old < self.read_timeout_s:
            try:
                service._read_timeout = self.read_timeout_s
            except Exception:  # noqa: BLE001 - a nicety, never a failure
                pass

    def _complete(self, sdk_model, request):
        self._raise_read_timeout(sdk_model)

        def once():
            if self.rate_limit_retries is None:
                return sdk_model.create_chat_completion(request)
            try:
                return sdk_model.create_chat_completion(
                    request, max_rate_limit_retries=self.rate_limit_retries)
            except TypeError as exc:
                if "max_rate_limit_retries" not in str(exc):
                    raise
                return sdk_model.create_chat_completion(request)

        return call_with_retries(once, total_s=self.retry_total_s)

    def _use_responses(self) -> bool:
        route = str(self.route or "auto").strip().lower()
        if route == "responses":
            return True
        if route == "chat":
            return False
        return responses_available()

    def _call_responses(self, request):
        (_, CompletionRequestV3, CreateCompletionRequest, create,
         run_with_rate_limit_retries) = _responses_sdk()
        # Any language-model handle for this model carries the service, the
        # auth header and the attribution the request needs.
        handle = self._vision_model or self._sdk_model or self._vision()
        self._raise_read_timeout(handle)
        wrapped = CreateCompletionRequest(
            handle._attribution, CompletionRequestV3(open_ai_responses=request))

        def call():
            return create(handle._llm_service, handle._auth_header,
                          handle._model_api_name, wrapped,
                          is_registered_model=handle._is_registered_model)

        def once():
            if self.rate_limit_retries is None:
                return call()
            return run_with_rate_limit_retries(call, self.rate_limit_retries)

        response = call_with_retries(once, total_s=self.retry_total_s)
        got = getattr(response, "open_ai_responses", None)
        if got is None:
            raise RuntimeError("Expected an openAiResponses response, got "
                               f"{getattr(response, 'type', type(response))}")
        return got

    def _generate_responses(self, messages, max_tokens, temperature,
                            tools) -> ChatResult:
        r = _responses_sdk()[0]
        request_kwargs: dict = {}
        if max_tokens is not None:
            request_kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if tools:
            request_kwargs["tools"] = _openai_tools_to_responses(tools, r)
        items = _lc_messages_to_responses(messages, r)
        reasoning = (None if self._reasoning_refused else
                     _reasoning_request(r, reasoning_summary_level()))
        request = None
        if reasoning is not None:
            try:
                request = r.OpenAiResponsesRequest(
                    input=items, reasoning=reasoning, **request_kwargs)
            except TypeError:           # this SDK's request has no such field
                reasoning = None
        if request is None:
            request = r.OpenAiResponsesRequest(input=items, **request_kwargs)
        try:
            response = self._call_responses(request)
        except Exception as exc:  # noqa: BLE001 - classified below
            if (reasoning is None or transient_kind(exc) is not None
                    or not _refused_reasoning(exc)):
                raise
            # The service would not take the summary setting: ask once more
            # without it, and do not ask again on this model.
            log.warning("Foundry Responses refused the reasoning-summary "
                        "setting (%s); asking without it from now on",
                        str(exc)[:200])
            self._reasoning_refused = True
            reasoning = None
            response = self._call_responses(
                r.OpenAiResponsesRequest(input=items, **request_kwargs))

        ai_message = _responses_to_ai_message(response)
        summary = _reasoning_text(response)
        if summary:
            ai_message.additional_kwargs["reasoning"] = summary
        elif reasoning is not None and not self._reasoning_fields_logged:
            # Asked for and nothing read back: say once what the output
            # items carry, so the first live call shows where a summary is
            # (or that the model wrote none).
            self._reasoning_fields_logged = True
            log.info("Foundry Responses: a reasoning summary was requested "
                     "and none was read; output items: %s",
                     "; ".join(_output_fields(response)) or "(none)")
        status = str(getattr(response, "status", "")).rsplit(".", 1)[-1]
        finish = ("tool_calls" if ai_message.tool_calls else
                  "length" if status.lower() == "incomplete" else "stop")
        generation_info = {
            "finish_reason": finish,
            "model_name": getattr(response, "model", None)
            or self.model_api_name,
            "reasoning_summary": ("requested" if reasoning is not None
                                  else "not requested")}
        u = getattr(response, "usage", None)
        if u is not None:
            usage = {"prompt_tokens": getattr(u, "input_tokens", None),
                     "completion_tokens": getattr(u, "output_tokens", None),
                     "total_tokens": getattr(u, "total_tokens", None)}
            usage = {k: v for k, v in usage.items() if v is not None}
            generation_info["usage"] = _usage_to_dict(usage)
            ai_message.usage_metadata = _to_usage_metadata(usage)
        return ChatResult(generations=[
            ChatGeneration(message=ai_message,
                           generation_info=generation_info)])

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
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        tools = kwargs.get("tools") or self.openai_tools
        if self._use_responses():
            return self._generate_responses(messages, max_tokens, temperature,
                                            tools)

        _, lms_base, lms_v3 = _sdk()
        request_kwargs: dict = {}
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if stop:
            request_kwargs["stop"] = list(stop)
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

__all__ = ["PalantirSdkChatModel", "sdk_available", "responses_available",
           "call_with_retries", "transient_kind", "ROUTES",
           "REASONING_SUMMARY_ENV", "reasoning_summary_level"]
