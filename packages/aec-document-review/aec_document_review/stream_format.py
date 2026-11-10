"""The chat page's stream formatters (a copy of the first part of
GeotechStaffEngineer's ``funhouse_agent/deep/notebook.py`` at the app
split's step 13): a deep agent's streamed updates as the lines the
page shows, the final answer of an update, and the token line."""

from __future__ import annotations

import html as _html
import json
import re
from typing import Any, Iterable, Optional
from uuid import uuid4


# ---------------------------------------------------------------------------
# Stream-mode parsing constants (the contract documented in the module header)
# ---------------------------------------------------------------------------

#: The LangGraph node that emits the visible assistant answer tokens. Token
#: chunks from any other node (e.g. a sub-agent's inner model, or tool-argument
#: streaming) are NOT shown in the answer bubble.
_MODEL_NODE = "model"

#: Run-metadata key langchain sets on its own internal model calls, and the
#: values that mark a call whose text is NOT for the user (the context
#: summarizer). Its chunks still carry ``langgraph_node == "model"``.
_LC_SOURCE_KEY = "lc_source"
_INTERNAL_SOURCES = frozenset({"summarization"})

#: Tools whose call we render with bespoke one-liners; everything else falls back
#: to a generic ``name({args})`` summary.
_KNOWN_TOOLS = {
    "call_agent", "list_methods", "describe_method", "list_agents",
    "read_pdf_text", "analyze_image", "analyze_pdf_page",
    "read_reference_figure", "save_file",
    "write_todos", "task",
}


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

def _escape(text: Any) -> str:
    """HTML-escape ``text`` for safe rendering in the widget transcript."""
    return _html.escape(str(text))


def _format_answer(text: str) -> str:
    """Light markdown-to-HTML for agent answers (bold + line breaks)."""
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("\n", "<br>")
    return text


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis marker."""
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


def _content_to_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` (str OR block list) to text.

    Claude-style content is a list of blocks; only ``text`` blocks contribute to
    the visible answer (tool-use / image blocks are ignored here).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in (None, "text") and "text" in b:
                    parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return "" if content is None else str(content)


# ---------------------------------------------------------------------------
# Pure formatters over the (mode, chunk) stream — unit-testable, no model/UI
# ---------------------------------------------------------------------------

def _format_tool_call(name: str, args: Optional[dict]) -> str:
    """Render a single tool CALL as a compact one-line string.

    Parameters
    ----------
    name : str
        The tool name (``call_agent`` / ``write_todos`` / ``task`` / ...).
    args : dict, optional
        The tool-call arguments (the LangChain ``tool_call["args"]`` dict).

    Returns
    -------
    str
        A human-readable one-liner, e.g.
        ``"call_agent: bearing_capacity.bearing_capacity_analysis({width: 2.0})"``
        or ``"task: → delegating to references"``.
    """
    args = args or {}

    if name == "call_agent":
        agent_name = args.get("agent_name", "?")
        method = args.get("method", "?")
        params = args.get("parameters")
        if params is None:
            # Flattened-parameter quirk: non-call_agent keys are the params.
            params = {
                k: v for k, v in args.items()
                if k not in ("agent_name", "method", "parameters")
            }
        param_str = _format_params(params)
        return f"call_agent: {agent_name}.{method}({param_str})"

    if name == "task":
        sub = args.get("subagent_type") or args.get("subagent") or "?"
        return f"task: -> delegating to {sub}"

    if name == "write_todos":
        todos = args.get("todos") or []
        return f"write_todos: {len(todos)} item(s)"

    if name in ("describe_method", "list_methods"):
        agent_name = args.get("agent_name", "?")
        method = args.get("method")
        inner = f"{agent_name}.{method}" if method else agent_name
        return f"{name}: {inner}"

    if name == "read_reference_figure":
        ref = args.get("reference", "?")
        fig = args.get("figure_number", "?")
        return f"read_reference_figure: {ref} fig {fig}"

    if name == "read_pdf_text":
        src = args.get("source") or args.get("attachment_key") or "?"
        pages = args.get("pages")
        return f"read_pdf_text: {src}" + (f" pages {pages}" if pages not in (None, "") else "")

    if name in ("analyze_image", "analyze_pdf_page"):
        key = args.get("attachment_key", "?")
        return f"{name}: {key}"

    if name == "save_file":
        return f"save_file: {args.get('path', '?')}"

    # Generic fallback.
    return f"{name}({_format_params(args)})"


def _format_params(params: Any) -> str:
    """Render a params dict compactly as ``key: value, ...`` (single line)."""
    if not isinstance(params, dict) or not params:
        return ""
    return ", ".join(f"{k}: {v}" for k, v in params.items())


def _render_todos(todos: Iterable[dict]) -> str:
    """Render a to-do list as a checklist with ✓ / ▶ / ▢ markers.

    Parameters
    ----------
    todos : iterable of dict
        Each ``{"content": str, "status": "pending"|"in_progress"|"completed"}``
        (the deepagents ``Todo`` shape).

    Returns
    -------
    str
        A multi-line checklist, e.g.::

            To-dos:
              [x] Compute bearing capacity
              [>] Check against DM7
              [ ] Write up the result

        Empty input returns an empty string.
    """
    todos = list(todos or [])
    if not todos:
        return ""
    marker = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]"}
    lines = ["To-dos:"]
    for t in todos:
        status = (t.get("status") if isinstance(t, dict) else None) or "pending"
        content = (t.get("content") if isinstance(t, dict) else str(t)) or ""
        lines.append(f"  {marker.get(status, '[ ]')} {content}")
    return "\n".join(lines)


def _format_update(mode: str, chunk: Any, *, max_result_chars: int = 2000) -> list:
    """Render ONE ``(mode, chunk)`` stream item to transcript line(s).

    This is the pure heart of the renderer. It returns a list of dicts, each
    describing one transcript entry, so both the stdout fallback and the widget
    UI can format them however they like. Entry kinds:

    * ``{"kind": "token", "text": str}`` — a streamed assistant-answer token
      (from ``messages`` mode, model node only).
    * ``{"kind": "tool_call", "text": str}`` — a tool CALL one-liner.
    * ``{"kind": "todos", "text": str}`` — a rendered to-do checklist.
    * ``{"kind": "tool_result", "text": str}`` — a (truncated) tool RESULT.

    Parameters
    ----------
    mode : str
        ``"messages"`` or ``"updates"`` (other modes yield nothing).
    chunk : Any
        The mode-specific payload (see the module header for shapes).
    max_result_chars : int
        Truncation budget for tool-result text.

    Returns
    -------
    list of dict
        Zero or more transcript entries for this stream item.
    """
    if mode == "messages":
        return _format_messages_chunk(chunk)
    if mode == "updates":
        return _format_updates_chunk(chunk, max_result_chars=max_result_chars)
    return []


def _format_messages_chunk(chunk: Any) -> list:
    """Extract visible assistant token(s) from a ``messages``-mode chunk.

    The chunk is ``(message_chunk, metadata)``. We only surface text from the
    model node (``langgraph_node == "model"``) and only when the chunk is NOT
    streaming tool-call arguments (``tool_call_chunks``) — that keeps sub-agent
    internals and tool-arg streams out of the answer bubble.
    """
    if not isinstance(chunk, tuple) or len(chunk) != 2:
        return []
    message, metadata = chunk
    meta = metadata if isinstance(metadata, dict) else {}
    if meta.get("langgraph_node", _MODEL_NODE) != _MODEL_NODE:
        return []
    # A context-summarization call runs INSIDE the model node (deepagents'
    # SummarizationMiddleware wraps the model call) and tags its own run with
    # ``lc_source: summarization``. Its tokens are an internal handoff
    # ("SESSION INTENT ... NEXT STEPS", with server paths), never answer text
    # (live smoke wave 1, A3: one was pasted into a delivered answer).
    if meta.get(_LC_SOURCE_KEY) in _INTERNAL_SOURCES:
        return []
    # Skip pure tool-call-argument streaming chunks (no visible text).
    if getattr(message, "tool_call_chunks", None):
        return []
    text = _content_to_text(getattr(message, "content", ""))
    if not text:
        return []
    return [{"kind": "token", "text": text}]


def _format_updates_chunk(chunk: Any, *, max_result_chars: int = 2000) -> list:
    """Render an ``updates``-mode chunk: ``{node: {state_key: value}}``.

    Surfaces tool CALLS (from AIMessages carrying ``tool_calls``), the to-do
    checklist (from a ``todos`` state key), and tool RESULTS (from
    ``ToolMessage`` entries), in that order.
    """
    if not isinstance(chunk, dict):
        return []
    entries: list = []
    for _node, update in chunk.items():
        if not isinstance(update, dict):
            continue
        messages = _update_messages(update)
        # 1) Tool calls live on AIMessages in the node's "messages".
        for msg in messages:
            for tc in getattr(msg, "tool_calls", None) or []:
                if isinstance(tc, dict):
                    entries.append({
                        "kind": "tool_call",
                        "text": _format_tool_call(tc.get("name", "?"),
                                                  tc.get("args")),
                    })
        # 2) The current to-do list (write_todos result state).
        todos = update.get("todos")
        if todos:
            rendered = _render_todos(todos)
            if rendered:
                entries.append({"kind": "todos", "text": rendered})
        # 3) Tool RESULTS are ToolMessages in the node's "messages".
        for msg in messages:
            if _is_tool_message(msg):
                name = getattr(msg, "name", None) or "tool"
                result = _content_to_text(getattr(msg, "content", ""))
                entries.append({
                    "kind": "tool_result",
                    "text": f"{name} -> {_truncate(result, max_result_chars)}",
                })
    return entries


def _is_tool_message(msg: Any) -> bool:
    """True if ``msg`` is a LangChain ``ToolMessage`` (a tool result)."""
    return (
        getattr(msg, "type", None) == "tool"
        or type(msg).__name__ == "ToolMessage"
    )


def _update_messages(update: Any) -> list:
    """The NEW messages in one node's update, as a list.

    A node may instead REWRITE the whole history -- ``Overwrite([...])``
    (deepagents' tool-call patcher and large-result eviction) or a list that
    opens with ``RemoveMessage(REMOVE_ALL_MESSAGES)`` (langchain's summarizer)
    -- and that is not news: it re-sends every earlier message, earlier
    turns' answers included. Those return ``[]``."""
    if not isinstance(update, dict):
        return []
    msgs = update.get("messages")
    if not isinstance(msgs, (list, tuple)):
        return []
    for m in msgs:
        mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        kind = m.get("type") if isinstance(m, dict) else getattr(m, "type",
                                                                   None)
        if kind == "remove" and mid == "__remove_all__":
            return []
    return list(msgs)


def _is_ai_message(msg: Any) -> bool:
    if isinstance(msg, dict):
        return msg.get("role") in ("assistant", "ai") or msg.get("type") == "ai"
    return getattr(msg, "type", None) in ("ai", "AIMessageChunk")


def _final_answer_in_update(chunk: Any) -> tuple:
    """``(saw_ai, final_text)`` for one ``updates``-mode item.

    ``saw_ai`` -- whether a model call finished in it (any AI message);
    ``final_text`` -- the text of the LAST AI message in it that requested no
    tool, or ``None``. That message is the reply: an AI message that also
    calls tools is narration between steps ("I'll check the page map..."),
    and a context summary is never an AI message of the graph at all. The
    web app delivers this, not the concatenated stream (live smoke wave 1,
    A3: narration glued mid-sentence into 16 of 95 answers)."""
    saw_ai, final = False, None
    if not isinstance(chunk, dict):
        return saw_ai, final
    for _node, update in chunk.items():
        for msg in _update_messages(update):
            if not _is_ai_message(msg):
                continue
            saw_ai = True
            calls = (msg.get("tool_calls") if isinstance(msg, dict)
                     else getattr(msg, "tool_calls", None))
            if calls:
                continue
            content = (msg.get("content") if isinstance(msg, dict)
                       else getattr(msg, "content", ""))
            final = _content_to_text(content)
    return saw_ai, final


def _sum_callback_tokens(callback_usage: Any) -> int:
    """Sum ``total_tokens`` across a ``UsageMetadataCallbackHandler`` dict.

    The callback aggregates per model name
    (``{model_name: {"input_tokens", "output_tokens", "total_tokens"}}``) and —
    because sub-agent model calls propagate through the LangGraph run's
    callbacks — that dict already includes the references/reviewer sub-agent
    spend. Returns ``0`` when the callback recorded nothing (e.g. a backend that
    exposes no usage), so a turn with no usage simply adds nothing.
    """
    if not callback_usage:
        return 0
    total = 0
    for um in callback_usage.values():
        if isinstance(um, dict):
            t = um.get("total_tokens")
            if t:
                total += int(t)
            else:
                total += int(um.get("input_tokens", 0) or 0)
                total += int(um.get("output_tokens", 0) or 0)
    return total


def _format_token_line(turn_tokens: int, conversation_total: int) -> str:
    """Format the per-answer token line shown under each agent reply.

    Parameters
    ----------
    turn_tokens : int
        Tokens spent on THIS turn (all model calls, sub-agents included).
    conversation_total : int
        Running cumulative total for the whole conversation.

    Returns
    -------
    str
        ``"tokens this turn: 18,432 | conversation total: 142,907"``.
    """
    return (
        f"tokens this turn: {turn_tokens:,} | "
        f"conversation total: {conversation_total:,}"
    )
