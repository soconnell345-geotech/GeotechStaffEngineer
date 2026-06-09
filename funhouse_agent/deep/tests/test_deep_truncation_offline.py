"""Offline tests for the v5.0 deepagents tool-result truncation cap.

Guards against the token-bloat regression where large reference-text dumps
from the references sub-agent were re-sent on every round (one layered-clay
question hit ~472k INPUT tokens). The v1 agent capped tool results at 8000
chars (``GeotechAgent._max_result_chars``); the v2 port lost that cap and this
restores it.

These tests make NO API/network calls — they invoke the LangChain tools
directly and build the deep agent with a fake chat model.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_truncation_offline.py -v
"""

import json

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from funhouse_agent.dispatch import REFERENCE_MODULES
from funhouse_agent.deep.agent import (
    build_deep_agent,
    build_references_subagent,
)
from funhouse_agent.deep.tools import (
    DEFAULT_MAX_RESULT_CHARS,
    make_core_tools,
    make_vision_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRUNC_MARKER = "[truncated"


def _tool_by_name(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not in {[t.name for t in tools]}")


def _fake_model():
    """A LangChain fake chat model — never calls a real LLM."""
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


# ---------------------------------------------------------------------------
# make_core_tools — truncation behavior
# ---------------------------------------------------------------------------

def test_core_tool_truncates_large_result():
    """list_agents returns the full catalog; a tiny cap must truncate it."""
    full = _tool_by_name(make_core_tools(max_result_chars=0), "list_agents").invoke({})
    assert len(full) > 50  # the catalog is comfortably bigger than the cap

    capped = _tool_by_name(
        make_core_tools(max_result_chars=50), "list_agents"
    ).invoke({})
    # First 50 chars are preserved verbatim, then the marker is appended.
    assert capped.startswith(full[:50])
    assert _TRUNC_MARKER in capped
    # The kept content is exactly the cap; the only extra is the marker line.
    body, _, marker = capped.partition("\n...")
    assert len(body) == 50
    assert marker.startswith(_TRUNC_MARKER)
    # The marker reports the dropped-character count.
    assert str(len(full) - 50) in marker


def test_core_tool_zero_disables_truncation():
    """max_result_chars=0 returns the full result, no marker."""
    out = _tool_by_name(
        make_core_tools(max_result_chars=0), "list_agents"
    ).invoke({})
    assert _TRUNC_MARKER not in out
    # Round-trips as valid JSON (proves it was not cut mid-string).
    assert isinstance(json.loads(out), dict)


def test_core_tool_negative_disables_truncation():
    """A negative cap behaves like 0 (disabled)."""
    out = _tool_by_name(
        make_core_tools(max_result_chars=-1), "list_agents"
    ).invoke({})
    assert _TRUNC_MARKER not in out
    assert isinstance(json.loads(out), dict)


def test_default_cap_is_8000():
    """The default cap mirrors v1's GeotechAgent._max_result_chars."""
    assert DEFAULT_MAX_RESULT_CHARS == 8000


def test_default_cap_leaves_small_calc_result_intact():
    """A normal small result (a bearing-capacity calc) is NOT truncated under
    the default 8000-char cap."""
    call = _tool_by_name(make_core_tools(), "call_agent")
    out = call.invoke({
        "agent_name": "bearing_capacity",
        "method": "bearing_capacity_analysis",
        "parameters": {
            "width": 2.0,
            "length": 2.0,
            "depth": 1.5,
            "shape": "square",
            "friction_angle": 30.0,
            "cohesion": 5.0,
            "unit_weight": 18.0,
        },
    })
    assert _TRUNC_MARKER not in out
    result = json.loads(out)  # still valid JSON
    assert "error" not in result, result
    assert result["q_ultimate_kPa"] > 0


def test_default_cap_truncates_oversized_result():
    """A result longer than the default cap is truncated (catalog dump)."""
    # Build with the default cap but force a tiny override to prove the path,
    # then confirm the default-built tool would also cut a > 8000 char result.
    full = _tool_by_name(make_core_tools(max_result_chars=0), "list_agents").invoke({})
    if len(full) <= DEFAULT_MAX_RESULT_CHARS:
        pytest.skip("catalog smaller than the default cap; nothing to truncate")
    capped = _tool_by_name(make_core_tools(), "list_agents").invoke({})
    assert _TRUNC_MARKER in capped
    assert len(capped) < len(full)


# ---------------------------------------------------------------------------
# make_vision_tools — truncation behavior
# ---------------------------------------------------------------------------

def test_vision_tool_truncates_large_result():
    """A vision tool's result is capped too (offline error payload here)."""
    tools = make_vision_tools(engine=None, max_result_chars=10)
    out = _tool_by_name(tools, "read_reference_figure").invoke(
        {"reference": "x", "figure_number": "y"}
    )
    assert _TRUNC_MARKER in out


def test_vision_tool_zero_disables_truncation():
    tools = make_vision_tools(engine=None, max_result_chars=0)
    out = _tool_by_name(tools, "read_reference_figure").invoke(
        {"reference": "x", "figure_number": "y"}
    )
    assert _TRUNC_MARKER not in out
    assert "error" in json.loads(out)


# ---------------------------------------------------------------------------
# Cap threads through the agent builders (primary + sub-agents)
# ---------------------------------------------------------------------------

def test_build_deep_agent_accepts_cap_smoke():
    """build_deep_agent(max_result_chars=...) constructs without error."""
    agent = build_deep_agent(model=_fake_model(), max_result_chars=1234)
    assert type(agent).__name__ == "CompiledStateGraph"


def test_references_subagent_tools_truncate():
    """The references sub-agent's tools honor the cap (this is the key fix:
    the sub-agent is where the big reference text comes from)."""
    spec = build_references_subagent(max_result_chars=40)
    call = _tool_by_name(spec["tools"], "list_agents")
    out = call.invoke({})
    assert _TRUNC_MARKER in out
    # Reference modules are visible to this sub-agent (sanity: right scope).
    full = _tool_by_name(
        make_core_tools(allowed_agents=REFERENCE_MODULES, max_result_chars=0),
        "list_agents",
    ).invoke({})
    assert out.startswith(full[:40])


def test_references_subagent_prompt_has_concision_instruction():
    """The references sub-agent prompt appends a concision instruction so it
    returns only the value(s) + citation, not long chapter text."""
    from funhouse_agent.reviewer import CONSULTANT_FRAMING

    spec = build_references_subagent()
    prompt = spec["system_prompt"]
    assert prompt.startswith(CONSULTANT_FRAMING)
    assert "concise" in prompt.lower()
    assert "citation" in prompt.lower()
    # It must not paste long chapter text — that phrasing is the guard.
    assert "do not paste" in prompt.lower() or "don't paste" in prompt.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
