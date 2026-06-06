"""Offline tests for the v5.0 deepagents Phase 3 capabilities (NO API key, NO network).

Phase 3 adds: planning, cross-session memory, summarization, and durable threads.
deepagents already ships planning (``write_todos``) and a scratch filesystem
(``ls`` / ``read_file`` / ``write_file`` / ``edit_file``) by default, so most of
this is *configuration*: a store-backed ``/memories/`` route, an explicit
summarizer, and forwarding ``store=`` / ``checkpointer=``.

Covers:
  (a) build_deep_agent(...) with store + checkpointer + enable_memory +
      enable_summarization constructs a CompiledStateGraph and exposes the
      planning + filesystem tools.
  (b) Cross-session memory at the BACKEND level (no LLM): a file written under
      ``/memories/`` through one CompositeBackend persists in the store and reads
      back through a NEW backend sharing the SAME store; a non-/memories path
      routes to the ephemeral StateBackend.
  (c) The custom SummarizationMiddleware is forwarded into the middleware stack
      when enable_summarization=True and absent when False.
  (d) Defaults: build_deep_agent(model=<fake>) with no Phase-3 args forwards no
      store / checkpointer / memory and still builds.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_phase3_offline.py -v
"""

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware import SummarizationMiddleware

import funhouse_agent.deep.agent as agent_mod
from funhouse_agent.deep.agent import (
    MEMORIES_AGENTS_FILE,
    MEMORIES_ROUTE,
    build_deep_agent,
    build_memory_backend,
    build_summarization_middleware,
)
from funhouse_agent.deep.prompt import build_domain_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_model():
    """A LangChain fake chat model — never calls a real LLM, no token profile."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _compiled_tool_names(agent):
    """Tool names bound into a compiled deep agent's ToolNode."""
    return set(agent.nodes["tools"].bound.tools_by_name.keys())


def _capture_create_kwargs(monkeypatch):
    """Patch ``create_deep_agent`` (as imported in agent.py) to record the kwargs
    ``build_deep_agent`` forwards, while still building a real agent."""
    captured = {}
    orig = agent_mod.create_deep_agent

    def spy(**kwargs):
        captured.update(kwargs)
        captured.setdefault("middleware", kwargs.get("middleware", []))
        return orig(**kwargs)

    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    return captured


# ===========================================================================
# (a) Full Phase-3 construction
# ===========================================================================

def test_full_phase3_agent_constructs():
    """store + checkpointer + enable_memory + enable_summarization builds OK."""
    agent = build_deep_agent(
        model=_fake_model(),
        store=InMemoryStore(),
        checkpointer=InMemorySaver(),
        enable_memory=True,
        enable_summarization=True,
    )
    # CompiledStateGraph from create_deep_agent.
    assert type(agent).__name__ == "CompiledStateGraph"


def test_planning_and_filesystem_tools_present():
    """Planning (write_todos) + scratch filesystem tools come from the default
    deepagents middleware and must be bound on the built agent."""
    agent = build_deep_agent(
        model=_fake_model(),
        store=InMemoryStore(),
        checkpointer=InMemorySaver(),
        enable_memory=True,
        enable_summarization=True,
    )
    names = _compiled_tool_names(agent)
    # Planning tool.
    assert "write_todos" in names, sorted(names)
    # Scratch filesystem tools.
    for fs_tool in ("ls", "read_file", "write_file", "edit_file"):
        assert fs_tool in names, f"{fs_tool} missing from {sorted(names)}"


def test_store_checkpointer_memory_forwarded(monkeypatch):
    """store=, checkpointer=, and the memory= source are forwarded to
    create_deep_agent, and the filesystem backend is a CompositeBackend."""
    captured = _capture_create_kwargs(monkeypatch)
    store = InMemoryStore()
    cp = InMemorySaver()
    build_deep_agent(
        model=_fake_model(),
        store=store,
        checkpointer=cp,
        enable_memory=True,
    )
    assert captured["store"] is store
    assert captured["checkpointer"] is cp
    # /memories/AGENTS.md is registered as a deepagents memory source.
    assert captured["memory"] == [MEMORIES_AGENTS_FILE]
    # The filesystem backend routes /memories/ to a store.
    backend = captured["backend"]
    assert isinstance(backend, CompositeBackend)
    assert MEMORIES_ROUTE in backend.routes
    assert isinstance(backend.routes[MEMORIES_ROUTE], StoreBackend)
    assert isinstance(backend.default, StateBackend)


def test_store_auto_enables_memory(monkeypatch):
    """Passing a store turns memory on even without enable_memory=True."""
    captured = _capture_create_kwargs(monkeypatch)
    build_deep_agent(model=_fake_model(), store=InMemoryStore())
    assert captured["memory"] == [MEMORIES_AGENTS_FILE]
    assert isinstance(captured["backend"], CompositeBackend)


# ===========================================================================
# (b) Cross-session memory at the BACKEND level (no LLM)
# ===========================================================================

def test_memories_persist_across_backend_instances():
    """A file written under /memories/ through one CompositeBackend persists in
    the store and reads back through a NEW backend sharing the SAME store.

    This is a REAL round-trip via the store, not a config-only assertion.
    """
    store = InMemoryStore()

    # Backend instance #1 (e.g. "session 1") writes a durable project note.
    backend1 = build_memory_backend(store=store)
    content = "Soil profile: medium dense sand, phi=33 deg, GWT at 3.5 m."
    write_res = backend1.write("/memories/profile.md", content)
    assert write_res.error is None
    assert write_res.path == "/memories/profile.md"

    # Backend instance #2 (e.g. "session 2") shares the SAME store and reads it.
    backend2 = build_memory_backend(store=store)
    read_res = backend2.read("/memories/profile.md")
    assert read_res.error is None
    assert read_res.file_data is not None
    assert read_res.file_data["content"] == content

    # The store actually holds the item (route prefix stripped to the key).
    items = list(store.search(("memories",)))
    assert any(content in (it.value.get("content") or "") for it in items)


def test_non_memories_path_uses_ephemeral_state_backend():
    """A non-/memories path routes to the ephemeral StateBackend.

    StateBackend requires a LangGraph graph execution context, so calling it
    directly (outside a graph) raises RuntimeError — proving the path is
    ephemeral/state-scoped, NOT store-backed.
    """
    store = InMemoryStore()
    backend = build_memory_backend(store=store)

    # Routing: /memories/ -> StoreBackend, everything else -> StateBackend.
    mem_backend, mem_key = backend._get_backend_and_key("/memories/x.md")
    assert isinstance(mem_backend, StoreBackend)
    scratch_backend, scratch_key = backend._get_backend_and_key("/scratch.txt")
    assert isinstance(scratch_backend, StateBackend)

    # The ephemeral backend cannot persist outside a graph context.
    with pytest.raises(RuntimeError):
        backend.write("/scratch.txt", "temporary scratch")

    # ...while the store-backed memories path persists fine outside a graph.
    assert backend.write("/memories/keep.md", "durable").error is None


# ===========================================================================
# (c) Summarization middleware presence / absence
# ===========================================================================

def test_summarization_middleware_present_when_enabled(monkeypatch):
    captured = _capture_create_kwargs(monkeypatch)
    build_deep_agent(model=_fake_model(), enable_summarization=True)
    mws = captured["middleware"]
    assert any(isinstance(m, SummarizationMiddleware) for m in mws), (
        f"no SummarizationMiddleware in {[type(m).__name__ for m in mws]}"
    )


def test_summarization_middleware_absent_when_disabled(monkeypatch):
    captured = _capture_create_kwargs(monkeypatch)
    build_deep_agent(model=_fake_model())  # enable_summarization defaults False
    mws = captured.get("middleware", [])
    assert not any(isinstance(m, SummarizationMiddleware) for m in mws)


def test_summarization_builder_uses_absolute_trigger_for_profileless_model():
    """A profile-less model (the fake) must get an absolute-token trigger, not a
    fractional one — a fractional trigger would raise at construction."""
    mw = build_summarization_middleware(_fake_model())
    assert isinstance(mw, SummarizationMiddleware)
    # Distinct name so it doesn't collide with the deepagents-default summarizer.
    assert mw.name != "SummarizationMiddleware"


def test_summarization_explicit_trigger_is_honored():
    mw = build_summarization_middleware(_fake_model(), trigger=("tokens", 12345))
    assert isinstance(mw, SummarizationMiddleware)


# ===========================================================================
# (d) Defaults: no Phase-3 args behaves as before
# ===========================================================================

def test_defaults_forward_no_store_or_checkpointer(monkeypatch):
    captured = _capture_create_kwargs(monkeypatch)
    agent = build_deep_agent(model=_fake_model())
    assert type(agent).__name__ == "CompiledStateGraph"
    # No durable / persistent wiring by default.
    assert captured.get("store") is None
    assert captured.get("checkpointer") is None
    assert captured.get("memory") is None
    assert captured.get("backend") is None
    # No custom summarizer appended.
    assert not any(
        isinstance(m, SummarizationMiddleware)
        for m in captured.get("middleware", [])
    )


def test_default_agent_still_builds_and_has_core_tools():
    """Sanity: the default agent (no Phase-3 args) still builds with our domain
    tools + planning + filesystem tools present."""
    agent = build_deep_agent(model=_fake_model())
    names = _compiled_tool_names(agent)
    for expected in ("call_agent", "write_todos", "read_file", "task"):
        assert expected in names, f"{expected} missing from {sorted(names)}"


# ===========================================================================
# Prompt nudges
# ===========================================================================

def test_prompt_nudges_planning_and_scratch():
    p = build_domain_prompt()
    assert "write_todos" in p
    assert "write_file" in p and "read_file" in p
    # The persistent /memories/ note is only added when memory is enabled.
    assert "/memories/" not in p


def test_prompt_adds_memories_note_when_enabled():
    p = build_domain_prompt(memory_enabled=True)
    assert "/memories/" in p
    assert "persist" in p.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
