"""ipywidgets chat interface for the v5.0 deepagents port (LangGraph streaming).

This is the Phase-4 sibling of the v1 :class:`funhouse_agent.notebook.NotebookChat`.
It mirrors the v1 chat UX — a scrollable transcript, a text input, a Send button,
a busy indicator, and a ``.display()`` — but drives it off the compiled
deepagents graph's **streaming** API instead of the blocking v1
``GeotechAgent.ask()`` + ``on_tool_call`` callback.

Two entry points share the same streaming/formatting core:

* :meth:`DeepNotebookChat.ask_and_print` — a non-widget fallback that streams the
  run to stdout (tool activity + the final answer) and returns the final answer
  text. Usable in any REPL/CI with no ipywidgets dependency.
* The widget *Send* button path — renders the same stream live into an
  ipywidgets transcript.

Streaming contract (LangGraph ``stream_mode=["updates", "messages"]``)
----------------------------------------------------------------------
Each yielded item is a ``(mode, chunk)`` tuple:

``("messages", (message_chunk, metadata))``
    ``message_chunk`` is an ``AIMessageChunk`` carrying a token of assistant text
    in ``.content`` (str, or a list of content blocks). ``metadata`` is a dict
    (``langgraph_node`` etc.). These drive the live assistant bubble. Only chunks
    whose ``langgraph_node`` is the model node ``"model"`` (and which carry no
    ``tool_call_chunks``) are treated as visible answer text, so a sub-agent's
    internal tokens and tool-argument streams do not leak into the bubble.

``("updates", {node_name: {state_key: value}})``
    A per-node state update. The ``model`` node yields ``{"messages": [AIMessage]}``
    where the ``AIMessage`` may carry ``.tool_calls`` (``call_agent`` /
    ``write_todos`` / ``task`` / ...). The ``tools`` node yields
    ``{"messages": [ToolMessage, ...], "todos": [...]}`` — tool RESULTS and the
    current to-do list. These drive the tool-activity display.

The formatters (:func:`_format_update`, :func:`_format_tool_call`,
:func:`_render_todos`) are PURE functions over those shapes, so they unit-test
without a live model or ipywidgets.

Usage::

    from funhouse_agent.deep.agent import build_deep_agent
    from funhouse_agent.deep.notebook import DeepNotebookChat

    agent = build_deep_agent(model)        # caller controls model/store/memory
    chat = DeepNotebookChat(agent)
    chat.display()

    # or without widgets:
    chat.ask_and_print("Bearing capacity of a 2 m strip footing, phi=30?")

    # or construct straight from a model:
    chat = DeepNotebookChat.from_model(model, enable_memory=True)

Requires ipywidgets ONLY for the widget UI (``pip install ipywidgets``); it is
lazy-imported inside the methods that need it, so ``import
funhouse_agent.deep.notebook`` and :meth:`ask_and_print` work without it.
"""

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
        # 1) Tool calls live on AIMessages in the node's "messages".
        for msg in update.get("messages", []) or []:
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
        for msg in update.get("messages", []) or []:
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


def _entry_to_plain(entry: dict) -> str:
    """Render a transcript entry dict to a plain-text line (stdout fallback)."""
    kind = entry.get("kind")
    text = entry.get("text", "")
    if kind == "token":
        return text  # tokens are concatenated, no prefix
    if kind == "tool_call":
        return f"  [tool] {text}"
    if kind == "todos":
        return text
    if kind == "tool_result":
        return f"  [result] {text}"
    if kind == "attachment":
        return f"  [attachment] {text}"
    return text


# ---------------------------------------------------------------------------
# DeepNotebookChat
# ---------------------------------------------------------------------------

class DeepNotebookChat:
    """ipywidgets chat over a compiled deepagents graph (LangGraph streaming).

    Mirrors the v1 :class:`funhouse_agent.notebook.NotebookChat` UX but streams
    the run from the compiled deep agent rather than calling a blocking
    ``ask()``. The streaming/formatting core is shared between the widget *Send*
    path and the non-widget :meth:`ask_and_print` fallback.

    Multi-turn continuity is maintained CLIENT-SIDE: a running ``messages`` list
    is replayed in full on every send, so conversation history is preserved with
    OR without a server-side checkpointer. When a ``thread_id`` is set it is also
    passed in ``config={"configurable": {"thread_id": ...}}`` so a
    checkpointer-backed agent additionally persists durable thread state. The two
    mechanisms are complementary, not exclusive (replaying the full list is
    idempotent for a stateless agent and harmless for a checkpointed one).

    Parameters
    ----------
    agent : CompiledStateGraph
        An ALREADY-BUILT compiled deepagents agent (from
        :func:`funhouse_agent.deep.agent.build_deep_agent`). The caller controls
        the model, store, checkpointer, and memory wiring — this UI only drives
        it.
    thread_id : str, optional
        Thread id for checkpointer/durable setups. A fresh ``uuid4().hex`` is
        generated when not given.
    store : BaseStore, optional
        Retained for symmetry / introspection (e.g. inspecting ``/memories/``).
        Not required to drive the agent — the agent already closes over its own
        store/checkpointer.
    max_result_chars : int
        Truncation budget for tool-result text in the transcript (default 2000).
    height : str
        CSS height for the scrollable widget transcript (default ``"500px"``).
    """

    def __init__(
        self,
        agent,
        *,
        thread_id: Optional[str] = None,
        store=None,
        attachments: Optional[dict] = None,
        max_result_chars: int = 2000,
        height: str = "500px",
    ):
        self._agent = agent
        self._store = store
        self._max_result_chars = max_result_chars
        self._height = height
        self._thread_id = thread_id or uuid4().hex

        # The attachments dict the FileUpload writes into MUST be the SAME object
        # the agent's vision tools read. Priority: an explicitly passed dict, else
        # the one build_deep_agent exposed on the agent, else a fresh dict (used
        # only if this agent was hand-built without one — uploads then may not
        # reach the tools; from_model always wires a shared dict).
        if attachments is not None:
            self._attachments = attachments
        else:
            self._attachments = getattr(agent, "geotech_attachments", None)
            if self._attachments is None:
                self._attachments = {}

        # Client-side conversation history (LangChain-style dict messages).
        self._messages: list[dict] = []
        # Rendered transcript entries (for the widget HTML render).
        self._transcript: list[dict] = []
        self._is_processing = False
        # Running cumulative token total for the WHOLE conversation (summed
        # across every turn, every model call — sub-agents included — via the
        # usage-metadata callback). The owner watches this to catch a runaway
        # conversation (one hit 800k tokens).
        self._total_tokens = 0
        self._last_turn_tokens = 0

        # Widgets are built lazily on .display() so importing this module and
        # using ask_and_print() never requires ipywidgets.
        self._container = None

    # -- alternate constructor ----------------------------------------------

    @classmethod
    def from_model(cls, model, *, thread_id=None, max_result_chars=2000,
                   height="500px", **build_kwargs):
        """Build a deep agent from ``model`` and wrap it in a chat UI.

        Convenience for the common case where the caller has a model but not yet
        a compiled agent. ``build_kwargs`` are forwarded verbatim to
        :func:`funhouse_agent.deep.agent.build_deep_agent` (e.g. ``store=``,
        ``checkpointer=``, ``enable_memory=True``, ``reference_mode=``).

        Parameters
        ----------
        model : str | BaseChatModel
            The chat model for ``build_deep_agent``.
        thread_id : str, optional
            See :class:`DeepNotebookChat`.
        max_result_chars : int
            See :class:`DeepNotebookChat`.
        height : str
            See :class:`DeepNotebookChat`.
        **build_kwargs
            Forwarded to ``build_deep_agent`` (``store``, ``checkpointer``,
            ``enable_memory``, ...).

        Returns
        -------
        DeepNotebookChat
        """
        from funhouse_agent.deep.agent import build_deep_agent

        store = build_kwargs.get("store")
        # One shared attachments dict handed to BOTH the agent build and the chat
        # so the FileUpload writes reach the tools.
        attachments = build_kwargs.pop("attachments", None)
        if attachments is None:
            attachments = {}
        agent = build_deep_agent(model, attachments=attachments, **build_kwargs)
        return cls(
            agent,
            thread_id=thread_id,
            store=store,
            attachments=attachments,
            max_result_chars=max_result_chars,
            height=height,
        )

    # -- public API ----------------------------------------------------------

    @property
    def thread_id(self) -> str:
        """The current LangGraph thread id."""
        return self._thread_id

    @property
    def messages(self) -> list:
        """The running client-side message history (live reference)."""
        return self._messages

    @property
    def total_tokens(self) -> int:
        """Running cumulative token total for the whole conversation.

        Summed across every turn and every model call (sub-agents included) via
        the usage-metadata callback. ``0`` before the first turn or after
        :meth:`reset`.
        """
        return self._total_tokens

    def reset(self) -> str:
        """Clear the conversation and start a fresh thread.

        Empties the client-side ``messages`` list and the rendered transcript,
        and rotates ``thread_id`` to a new ``uuid4().hex`` so a checkpointer-
        backed agent also starts a clean thread.

        Returns
        -------
        str
            The new thread id.
        """
        self._messages = []
        self._transcript = []
        self._thread_id = uuid4().hex
        self._total_tokens = 0
        self._last_turn_tokens = 0
        if self._container is not None:
            self._refresh_transcript()
            self._update_status("")
        return self._thread_id

    def display(self):
        """Build (lazily) and return the ipywidgets container for the notebook.

        Importing ipywidgets is deferred to here so headless imports and
        :meth:`ask_and_print` never need it.
        """
        if self._container is None:
            self._build_widgets()
        return self._container

    def _repr_mimebundle_(self, **kwargs):
        """Auto-display when this is the last expression in a notebook cell."""
        return self.display()._repr_mimebundle_(**kwargs)

    # -- streaming core (shared by widget + stdout paths) -------------------

    def _config(self) -> dict:
        """The LangGraph config carrying the thread id."""
        return {"configurable": {"thread_id": self._thread_id}}

    def _stream_items(self, question: str, *, config: Optional[dict] = None):
        """Yield ``(mode, chunk)`` items for one user turn.

        Appends the user message to the client-side history, then streams the
        agent over the FULL message list with
        ``stream_mode=["updates", "messages"]`` and the thread-id config. A
        ``config`` override (carrying the usage-metadata callback) is used when
        given so the stream's token usage is captured.
        """
        self._messages.append({"role": "user", "content": question})
        yield from self._agent.stream(
            {"messages": self._messages},
            config=config if config is not None else self._config(),
            stream_mode=["updates", "messages"],
        )

    def _run_stream(self, question: str, on_entry, on_token):
        """Drive one turn, dispatching rendered entries to callbacks.

        The whole stream runs under a
        :func:`~langchain_core.callbacks.get_usage_metadata_callback` so this
        turn's token total — aggregated across EVERY model call in the run,
        sub-agents included — is captured and folded into the running
        conversation total (:attr:`total_tokens`). Streaming is preserved: the
        callback aggregates regardless of streaming, so it is read AFTER the
        stream completes. If the callback is unavailable for any reason the turn
        still streams normally (token tracking simply adds nothing).

        Parameters
        ----------
        question : str
            The user's message.
        on_entry : callable(dict)
            Called for every NON-token transcript entry (tool_call / todos /
            tool_result) as it arrives.
        on_token : callable(str)
            Called for each assistant-answer token as it streams.

        Returns
        -------
        str
            The concatenated final assistant answer text.
        """
        from langchain_core.callbacks import get_usage_metadata_callback

        answer_parts: list[str] = []
        with get_usage_metadata_callback() as cb:
            config = dict(self._config())
            config["callbacks"] = [cb]
            for mode, chunk in self._stream_items(question, config=config):
                for entry in _format_update(
                    mode, chunk, max_result_chars=self._max_result_chars
                ):
                    if entry["kind"] == "token":
                        answer_parts.append(entry["text"])
                        on_token(entry["text"])
                    else:
                        on_entry(entry)
            # Read after the stream fully drains (callback aggregates regardless
            # of streaming). dict() snapshots the handler's per-model totals.
            turn_tokens = _sum_callback_tokens(dict(cb.usage_metadata))
        self._total_tokens += turn_tokens
        self._last_turn_tokens = turn_tokens
        final = "".join(answer_parts)
        # Record the assistant turn in the client-side history so the next send
        # replays it (continuity without a checkpointer).
        if final:
            self._messages.append({"role": "assistant", "content": final})
        return final

    # -- non-widget fallback ------------------------------------------------

    def ask_and_print(self, question: str, *, file=None) -> str:
        """Stream one turn to stdout and return the final answer text.

        A dependency-free path usable in any REPL/CI: prints tool activity (tool
        calls, to-dos, truncated results) and the streamed answer, then returns
        the full answer string. Shares the exact streaming/formatting core with
        the widget Send button.

        Parameters
        ----------
        question : str
            The user's message.
        file : file-like, optional
            Output stream (defaults to ``sys.stdout``).

        Returns
        -------
        str
            The final assistant answer text.
        """
        import sys

        out = file if file is not None else sys.stdout

        def _emit(text, end="\n"):
            # Tool output is arbitrary and may contain Unicode the console
            # encoding (e.g. Windows cp1252) cannot represent; degrade
            # gracefully instead of crashing the whole stream.
            try:
                print(text, end=end, file=out)
            except UnicodeEncodeError:
                enc = getattr(out, "encoding", None) or "ascii"
                print(text.encode(enc, "replace").decode(enc),
                      end=end, file=out)

        _emit(f"You: {question}")
        printed_answer_header = {"done": False}

        def on_entry(entry):
            _emit(_entry_to_plain(entry))

        def on_token(token):
            if not printed_answer_header["done"]:
                _emit("Agent: ", end="")
                printed_answer_header["done"] = True
            _emit(token, end="")

        final = self._run_stream(question, on_entry, on_token)
        if printed_answer_header["done"]:
            _emit("")  # newline after the streamed answer
        elif final:
            _emit(f"Agent: {final}")
        # A small running-token line under the answer (this turn + conversation).
        _emit(_format_token_line(self._last_turn_tokens, self._total_tokens))
        try:
            out.flush()
        except Exception:
            pass
        return final

    # -- widget construction -------------------------------------------------

    def _build_widgets(self):
        """Construct the ipywidgets UI (lazy import of ipywidgets)."""
        import ipywidgets as widgets

        self._widgets = widgets

        self._status_html = widgets.HTML(
            value="", layout=widgets.Layout(padding="4px 8px"),
        )
        self._chat_html = widgets.HTML(
            value="",
            layout=widgets.Layout(
                height=self._height,
                overflow_y="auto",
                border="1px solid #ddd",
                padding="8px",
            ),
        )
        # File upload — writes into the shared attachments dict so the user never
        # has to juggle paths (Databricks FileUpload caps around ~10 MB; for a
        # bigger PDF, add it to the attachments dict directly — see the guide).
        self._upload = widgets.FileUpload(
            accept=".pdf,.png,.jpg,.jpeg,.tif,.tiff,.xml,.diggs,.dxf,.csv,.txt",
            multiple=True,
            description="Attach",
            tooltip="Attach a file (PDF, image, DXF, DIGGS...) up to ~10 MB",
            layout=widgets.Layout(width="150px"),
        )
        self._attach_html = widgets.HTML(value="")
        upload_row = widgets.HBox(
            [self._upload, self._attach_html],
            layout=widgets.Layout(padding="4px 0"),
        )

        self._input = widgets.Text(
            placeholder="Ask a geotechnical question...",
            continuous_update=False,
            layout=widgets.Layout(flex="1"),
        )
        self._send_btn = widgets.Button(
            description="Send", button_style="primary",
            layout=widgets.Layout(width="80px"),
        )
        self._reset_btn = widgets.Button(
            description="Reset", button_style="warning",
            tooltip="Clear conversation and start a new thread",
            layout=widgets.Layout(width="80px"),
        )
        input_row = widgets.HBox(
            [self._input, self._send_btn, self._reset_btn],
            layout=widgets.Layout(padding="4px 0"),
        )
        self._container = widgets.VBox([
            self._status_html, self._chat_html, upload_row, input_row,
        ])

        self._send_btn.on_click(self._on_send)
        self._input.observe(self._on_input_change, names="value")
        self._reset_btn.on_click(lambda _=None: self.reset())
        self._upload.observe(self._on_upload_change, names="value")
        self._update_attachments_indicator()
        self._refresh_transcript()

    # -- widget event handlers ----------------------------------------------

    def _on_input_change(self, change):
        """Send on Enter (``continuous_update=False`` fires on commit)."""
        if change.get("new"):
            self._on_send()

    def _on_send(self, _=None):
        question = self._input.value.strip()
        if not question or self._is_processing:
            return
        self._is_processing = True
        self._send_btn.disabled = True
        self._input.value = ""

        self._transcript.append({"kind": "user", "text": question})
        # A live assistant bubble the streamed tokens accumulate into.
        bubble = {"kind": "assistant", "text": ""}
        self._transcript.append(bubble)
        self._update_status("Working...")
        self._refresh_transcript()

        def on_entry(entry):
            # Insert tool activity BEFORE the (still-streaming) assistant bubble.
            self._transcript.insert(len(self._transcript) - 1, entry)
            self._refresh_transcript()

        def on_token(token):
            bubble["text"] += token
            self._refresh_transcript()

        try:
            final = self._run_stream(question, on_entry, on_token)
            if not final:
                bubble["text"] = "(no answer text)"
            # A small running-token line under this answer.
            self._transcript.append({
                "kind": "tokens",
                "text": _format_token_line(self._last_turn_tokens,
                                           self._total_tokens),
            })
            self._refresh_transcript()
        except Exception as exc:  # surface errors in the transcript, don't raise
            self._transcript.append({
                "kind": "error",
                "text": f"{type(exc).__name__}: {exc}",
            })
            self._refresh_transcript()
        finally:
            self._update_status("")
            self._is_processing = False
            self._send_btn.disabled = False

    # -- attachments --------------------------------------------------------

    @property
    def attachments(self) -> dict:
        """The live attachments dict shared with the agent's vision tools."""
        return self._attachments

    def _ingest_uploads(self, value) -> list:
        """Write uploaded files into the shared attachments dict.

        Testable headless: updates ``self._attachments`` and the transcript, and
        refreshes the widgets ONLY when they have been built. Returns the list of
        attachment keys ingested (a sanitized basename; an existing key is
        overwritten with a visible note).
        """
        from funhouse_agent.vision_tools import (
            sanitize_upload_name, iter_upload_files,
        )
        names = []
        for raw_name, data in iter_upload_files(value):
            key = sanitize_upload_name(raw_name)
            overwrite = key in self._attachments
            self._attachments[key] = data
            names.append(key)
            note = (
                f"attached '{key}' ({len(data):,} bytes)"
                + (" — replaced the previous file of that name" if overwrite
                   else " — reference it by that name")
            )
            self._transcript.append({"kind": "attachment", "text": note})
        if names:
            self._update_attachments_indicator()
            self._refresh_transcript()
        return names

    def _on_upload_change(self, change):
        value = change.get("new")
        if not value:
            return
        self._ingest_uploads(value)
        # Clear the widget so re-uploading the same filename fires again.
        try:
            self._upload.value = {} if isinstance(self._upload.value, dict) else ()
        except Exception:
            pass

    def _update_attachments_indicator(self):
        if self._container is None:
            return
        keys = list(self._attachments.keys())
        if keys:
            self._attach_html.value = (
                '<span style="color:#555;font-family:monospace;font-size:12px;">'
                f"attachments: [{', '.join(_escape(k) for k in keys)}]</span>"
            )
        else:
            self._attach_html.value = (
                '<span style="color:#999;font-size:12px;">no attachments — use '
                "Attach, or reference a real path like /tmp/report.pdf</span>"
            )

    # -- widget rendering ----------------------------------------------------

    def _update_status(self, text: str):
        if self._container is None:
            return
        if text:
            self._status_html.value = (
                f'<span style="color:#888;font-style:italic;">{_escape(text)}'
                f"</span>"
            )
        else:
            self._status_html.value = (
                f'<span style="color:#666;font-family:monospace;font-size:12px;">'
                f"thread {self._thread_id[:8]} | {len(self._messages)} msgs</span>"
            )

    def _refresh_transcript(self):
        if self._container is None:
            return
        self._chat_html.value = self._render_transcript_html()
        self._update_status("Working..." if self._is_processing else "")

    def _render_transcript_html(self) -> str:
        parts = [_TRANSCRIPT_CSS]
        for entry in self._transcript:
            kind = entry.get("kind")
            text = entry.get("text", "")
            if kind == "user":
                parts.append(
                    f'<div class="dnb-msg dnb-user"><b>You:</b> '
                    f"{_escape(text)}</div>"
                )
            elif kind == "assistant":
                parts.append(
                    f'<div class="dnb-msg dnb-agent"><b>Agent:</b> '
                    f"{_format_answer(text)}</div>"
                )
            elif kind == "tool_call":
                parts.append(
                    f'<div class="dnb-tool">{_escape(text)}</div>'
                )
            elif kind == "todos":
                parts.append(
                    f'<div class="dnb-todos"><pre>{_escape(text)}</pre></div>'
                )
            elif kind == "tool_result":
                parts.append(
                    f'<div class="dnb-result">{_escape(text)}</div>'
                )
            elif kind == "tokens":
                parts.append(
                    f'<div class="dnb-tokens">{_escape(text)}</div>'
                )
            elif kind == "error":
                parts.append(
                    f'<div class="dnb-error">{_escape(text)}</div>'
                )
            elif kind == "attachment":
                parts.append(
                    f'<div class="dnb-attach">📎 {_escape(text)}</div>'
                )
        return "\n".join(parts)


# Inline CSS for the widget transcript — works in Databricks and local Jupyter.
_TRANSCRIPT_CSS = """<style>
.dnb-msg{margin:6px 0;padding:8px 12px;border-radius:8px;
         font-family:sans-serif;font-size:13px;line-height:1.5}
.dnb-user{background:#e3f2fd}
.dnb-agent{background:#f5f5f5}
.dnb-tool{background:#fff8e1;font-size:12px;margin:2px 0 2px 16px;
          padding:4px 8px;border-radius:6px;font-family:monospace}
.dnb-todos{background:#ede7f6;font-size:12px;margin:2px 0 2px 16px;
           padding:4px 8px;border-radius:6px}
.dnb-todos pre{margin:0;font-family:monospace;white-space:pre-wrap}
.dnb-result{background:#e8f5e9;font-size:12px;margin:2px 0 2px 16px;
            padding:4px 8px;border-radius:6px;font-family:monospace;
            white-space:pre-wrap}
.dnb-error{background:#ffebee;color:#b71c1c;font-size:12px;
           margin:6px 0;padding:6px 12px;border-radius:6px}
.dnb-tokens{color:#888;font-size:11px;font-family:monospace;
            margin:1px 0 6px 16px}
.dnb-attach{background:#fff8e1;color:#7a5c00;font-size:12px;
            margin:4px 0;padding:6px 12px;border-radius:6px}
</style>"""


__all__ = [
    "DeepNotebookChat",
    "_format_update",
    "_format_tool_call",
    "_render_todos",
    "_format_token_line",
    "_sum_callback_tokens",
]
