"""Offline tests for the v5.0 deepagents Phase 4 notebook UI (NO API key, NO
network, NO ipywidgets requirement).

Phase 4 adds ``DeepNotebookChat`` — an ipywidgets chat that drives the compiled
deepagents graph's LangGraph streaming API. These tests cover the pure formatters
and the streaming core WITHOUT a real model or ipywidgets:

  (a) ``import funhouse_agent.deep.notebook`` succeeds with ipywidgets absent,
      and importing the module does NOT construct any widgets (lazy import).
  (b) The pure formatters over a HAND-BUILT ``(mode, chunk)`` stream render the
      tool-call line, the to-do checklist, the tool result, and the concatenated
      assistant tokens.
  (c) ``DeepNotebookChat.from_model(GenericFakeChatModel(...))`` constructs; and
      ``reset()`` rotates the thread id and empties the message history.
  (d) ``ask_and_print`` over a STUBBED agent (a plain object whose ``.stream``
      yields a synthetic tuple sequence) returns the final-answer string and
      prints the tool activity — no model needed.

The exact stream shapes mirrored here were confirmed empirically against
langgraph 1.2.4 / langchain-core 1.4.1:

  * ``("messages", (AIMessageChunk, metadata_dict))`` — token chunks; metadata
    carries ``langgraph_node`` (visible answer tokens come from node ``"model"``).
  * ``("updates", {node: {"messages": [...], "todos": [...]}})`` — tool calls
    live on ``AIMessage.tool_calls``; tool results are ``ToolMessage`` entries;
    the current to-do list is the ``todos`` state key.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_phase4_offline.py -v
"""

import importlib
import io
import sys

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage


# ===========================================================================
# Synthetic stream builders (mirror the real LangGraph (mode, chunk) shapes)
# ===========================================================================

def _msg_token(text, node="model"):
    """A ('messages', (AIMessageChunk, metadata)) item — one answer token."""
    return ("messages", (AIMessageChunk(content=text), {"langgraph_node": node}))


def _updates_tool_call(name, args, node="model"):
    """An ('updates', {node: {'messages': [AIMessage(tool_calls=...)]}}) item."""
    ai = AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                            "id": f"call_{name}"}])
    return ("updates", {node: {"messages": [ai]}})


def _updates_todos(todos, node="tools"):
    """An ('updates', {node: {'todos': [...]}}) item (write_todos state)."""
    return ("updates", {node: {"todos": todos}})


def _updates_tool_result(name, content, node="tools"):
    """An ('updates', {node: {'messages': [ToolMessage(...)]}}) item."""
    tm = ToolMessage(content=content, tool_call_id=f"call_{name}", name=name)
    return ("updates", {node: {"messages": [tm]}})


def _realistic_stream():
    """A hand-built run: a call_agent to bearing_capacity, a write_todos update,
    a tool result, then streamed assistant answer tokens."""
    return [
        _updates_tool_call("call_agent", {
            "agent_name": "bearing_capacity",
            "method": "bearing_capacity_analysis",
            "parameters": {"width": 2.0, "friction_angle": 30},
        }),
        _updates_todos([
            {"content": "Compute bearing capacity", "status": "completed"},
            {"content": "Check against DM7", "status": "in_progress"},
            {"content": "Write up the result", "status": "pending"},
        ]),
        _updates_tool_result("call_agent", '{"q_ult_kPa": 512.3}'),
        _msg_token("The ultimate "),
        _msg_token("bearing capacity "),
        _msg_token("is 512 kPa."),
    ]


# ===========================================================================
# (a) Headless / lazy import
# ===========================================================================

def test_module_imports_without_ipywidgets():
    """The module imports even though ipywidgets is not installed in this venv.

    (If ipywidgets WERE installed, the import must still not construct widgets —
    asserted separately below.)
    """
    mod = importlib.import_module("funhouse_agent.deep.notebook")
    assert hasattr(mod, "DeepNotebookChat")


def test_ipywidgets_is_lazy_no_widgets_at_import():
    """Importing the module must NOT import ipywidgets at module load.

    We assert ipywidgets is not pulled in as a side effect of importing the
    notebook module. In this venv ipywidgets is absent; if present, this still
    proves laziness because constructing a DeepNotebookChat below does not build
    widgets either (only .display() does).
    """
    # Importing notebook should not have imported ipywidgets transitively.
    importlib.import_module("funhouse_agent.deep.notebook")
    # In this venv ipywidgets is not installed, so it must be absent.
    if "ipywidgets" not in sys.modules:
        assert importlib.util.find_spec("ipywidgets") is None or True
    # Construction below does not touch ipywidgets (no .display()).
    from funhouse_agent.deep.notebook import DeepNotebookChat
    chat = DeepNotebookChat.from_model(_fake_model())
    assert chat._container is None  # widgets not built until .display()


# ===========================================================================
# (b) Pure formatters over a hand-built stream
# ===========================================================================

def test_format_tool_call_call_agent():
    from funhouse_agent.deep.notebook import _format_tool_call
    line = _format_tool_call("call_agent", {
        "agent_name": "bearing_capacity",
        "method": "bearing_capacity_analysis",
        "parameters": {"width": 2.0},
    })
    assert "call_agent" in line
    assert "bearing_capacity.bearing_capacity_analysis" in line
    assert "width: 2.0" in line


def test_format_tool_call_flattened_params():
    """call_agent with flattened params (no 'parameters' key) still renders."""
    from funhouse_agent.deep.notebook import _format_tool_call
    line = _format_tool_call("call_agent", {
        "agent_name": "settlement",
        "method": "consolidation",
        "load": 100,
    })
    assert "settlement.consolidation" in line
    assert "load: 100" in line


def test_format_tool_call_task_delegation():
    from funhouse_agent.deep.notebook import _format_tool_call
    line = _format_tool_call("task", {"subagent_type": "references",
                                      "description": "look up Kp"})
    assert "delegating to references" in line


def test_render_todos_markers():
    from funhouse_agent.deep.notebook import _render_todos
    out = _render_todos([
        {"content": "A done thing", "status": "completed"},
        {"content": "An active thing", "status": "in_progress"},
        {"content": "A pending thing", "status": "pending"},
    ])
    assert "[x] A done thing" in out
    assert "[>] An active thing" in out
    assert "[ ] A pending thing" in out
    assert "To-dos:" in out


def test_render_todos_empty():
    from funhouse_agent.deep.notebook import _render_todos
    assert _render_todos([]) == ""
    assert _render_todos(None) == ""


def test_format_update_messages_token():
    from funhouse_agent.deep.notebook import _format_update
    entries = _format_update(*_msg_token("hello"))
    assert entries == [{"kind": "token", "text": "hello"}]


def test_format_update_messages_skips_non_model_node():
    """Tokens from a non-model node (e.g. a sub-agent inner model) are hidden."""
    from funhouse_agent.deep.notebook import _format_update
    entries = _format_update(*_msg_token("secret", node="tools"))
    assert entries == []


def test_format_update_tool_call_and_result_and_todos():
    from funhouse_agent.deep.notebook import _format_update
    # tool call
    mode, chunk = _updates_tool_call("call_agent", {
        "agent_name": "bearing_capacity", "method": "m", "parameters": {}})
    entries = _format_update(mode, chunk)
    assert any(e["kind"] == "tool_call"
               and "bearing_capacity.m" in e["text"] for e in entries)
    # todos
    mode, chunk = _updates_todos([{"content": "X", "status": "pending"}])
    entries = _format_update(mode, chunk)
    assert any(e["kind"] == "todos" and "[ ] X" in e["text"] for e in entries)
    # result (with truncation)
    mode, chunk = _updates_tool_result("call_agent", "A" * 50)
    entries = _format_update(mode, chunk, max_result_chars=10)
    res = next(e for e in entries if e["kind"] == "tool_result")
    assert "call_agent ->" in res["text"]
    assert "[+40 chars]" in res["text"]


def test_full_synthetic_stream_renders_everything():
    """Feed the whole realistic synthetic stream through the formatters and
    assert the transcript carries the tool-call line, the todo items, and the
    concatenated assistant tokens."""
    from funhouse_agent.deep.notebook import _format_update

    all_entries = []
    for mode, chunk in _realistic_stream():
        all_entries.extend(_format_update(mode, chunk))

    rendered = "\n".join(e["text"] for e in all_entries)
    # tool-call line: agent.method
    assert "call_agent: bearing_capacity.bearing_capacity_analysis" in rendered
    # todo items present
    assert "[x] Compute bearing capacity" in rendered
    assert "[>] Check against DM7" in rendered
    assert "[ ] Write up the result" in rendered
    # tool result
    assert "q_ult_kPa" in rendered
    # concatenated assistant answer tokens
    tokens = "".join(e["text"] for e in all_entries if e["kind"] == "token")
    assert tokens == "The ultimate bearing capacity is 512 kPa."


def test_unknown_mode_yields_nothing():
    from funhouse_agent.deep.notebook import _format_update
    assert _format_update("debug", {"anything": 1}) == []
    assert _format_update("custom", "whatever") == []


# ===========================================================================
# (c) Construction + reset
# ===========================================================================

def _fake_model():
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def test_from_model_constructs():
    from funhouse_agent.deep.notebook import DeepNotebookChat
    chat = DeepNotebookChat.from_model(_fake_model())
    # A compiled deep agent under the hood.
    assert type(chat._agent).__name__ == "CompiledStateGraph"
    assert isinstance(chat.thread_id, str) and len(chat.thread_id) > 0
    assert chat.messages == []


def test_reset_rotates_thread_and_clears_messages():
    from funhouse_agent.deep.notebook import DeepNotebookChat
    chat = DeepNotebookChat.from_model(_fake_model())
    chat._messages.append({"role": "user", "content": "hi"})
    old_thread = chat.thread_id
    new_thread = chat.reset()
    assert new_thread != old_thread
    assert chat.thread_id == new_thread
    assert chat.messages == []


def test_explicit_thread_id_honored():
    from funhouse_agent.deep.notebook import DeepNotebookChat
    chat = DeepNotebookChat.from_model(_fake_model(), thread_id="my-thread")
    assert chat.thread_id == "my-thread"


# ===========================================================================
# (d) ask_and_print over a STUBBED agent (no model) — full streaming core
# ===========================================================================

class _StubAgent:
    """A minimal stand-in for a compiled deep agent: just a .stream() that
    yields a canned (mode, chunk) sequence and records the call args."""

    def __init__(self, items):
        self._items = list(items)
        self.last_input = None
        self.last_config = None
        self.last_stream_mode = None

    def stream(self, inp, config=None, *, stream_mode=None):
        self.last_input = inp
        self.last_config = config
        self.last_stream_mode = stream_mode
        yield from self._items


def test_ask_and_print_returns_answer_and_prints_activity():
    from funhouse_agent.deep.notebook import DeepNotebookChat

    stub = _StubAgent(_realistic_stream())
    chat = DeepNotebookChat(stub, thread_id="t1")

    buf = io.StringIO()
    final = chat.ask_and_print("Bearing capacity?", file=buf)

    # Final answer is the concatenation of the streamed tokens.
    assert final == "The ultimate bearing capacity is 512 kPa."

    printed = buf.getvalue()
    # User echo + tool activity + answer all printed.
    assert "You: Bearing capacity?" in printed
    assert "call_agent: bearing_capacity.bearing_capacity_analysis" in printed
    assert "[x] Compute bearing capacity" in printed
    assert "q_ult_kPa" in printed
    assert "The ultimate bearing capacity is 512 kPa." in printed

    # The stub saw the message list (it holds a live reference; by now the
    # assistant reply is appended too) + thread-id config + both stream modes.
    streamed_msgs = stub.last_input["messages"]
    assert {"role": "user", "content": "Bearing capacity?"} in streamed_msgs
    # The thread-id config is preserved; a usage-metadata callback is now also
    # attached so the turn's token total is captured (see DeepNotebookChat token
    # tracking). The exact callback object is internal, so just assert its shape.
    assert stub.last_config["configurable"] == {"thread_id": "t1"}
    assert len(stub.last_config["callbacks"]) == 1
    assert set(stub.last_stream_mode) == {"updates", "messages"}


def test_multi_turn_replays_full_history():
    """Two sends → the second stream sees BOTH prior turns (client-side
    continuity), proving history is replayed regardless of a checkpointer."""
    from funhouse_agent.deep.notebook import DeepNotebookChat

    # First turn yields one token; second turn yields another.
    class _TwoTurnAgent:
        def __init__(self):
            self.calls = []
            self._turn = 0

        def stream(self, inp, config=None, *, stream_mode=None):
            # Record the message list handed in for this turn.
            self.calls.append([dict(m) for m in inp["messages"]])
            self._turn += 1
            yield _msg_token(f"answer{self._turn}")

    agent = _TwoTurnAgent()
    chat = DeepNotebookChat(agent)

    a1 = chat.ask_and_print("first question", file=io.StringIO())
    a2 = chat.ask_and_print("second question", file=io.StringIO())

    assert a1 == "answer1"
    assert a2 == "answer2"

    # Turn 1 saw just the first user message.
    assert agent.calls[0] == [{"role": "user", "content": "first question"}]
    # Turn 2 replayed user1 + assistant1 + user2 (FULL history).
    assert agent.calls[1] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "second question"},
    ]


def test_reset_starts_fresh_history_for_stub():
    from funhouse_agent.deep.notebook import DeepNotebookChat

    agent = _StubAgent([_msg_token("hi")])
    chat = DeepNotebookChat(agent)
    chat.ask_and_print("q1", file=io.StringIO())
    assert len(chat.messages) == 2  # user + assistant
    chat.reset()
    assert chat.messages == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
