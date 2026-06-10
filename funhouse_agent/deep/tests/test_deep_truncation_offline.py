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
    DEFAULT_REFERENCE_RESULT_CHARS,
    SEARCH_NARROWER_NUDGE,
    _resolve_reference_cap,
    _result_cap_for_module,
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
    """A vision tool's result is capped too (offline error payload here).

    ``read_reference_figure`` is a REFERENCE read, so it honors the (larger)
    ``reference_result_chars`` budget rather than ``max_result_chars``.
    """
    tools = make_vision_tools(
        engine=None, max_result_chars=10, reference_result_chars=10
    )
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
# v5.1 smart per-tool truncation: reference reads get a LARGER budget, calc
# results are uncapped, and the marker nudges a NARROWER follow-up search.
# ---------------------------------------------------------------------------

def test_truncation_marker_includes_search_narrower_nudge():
    """The truncation marker tells the agent to follow up with a narrower
    search instead of re-requesting the same oversized item."""
    capped = _tool_by_name(
        make_core_tools(max_result_chars=50), "list_agents"
    ).invoke({})
    assert _TRUNC_MARKER in capped
    assert SEARCH_NARROWER_NUDGE in capped
    assert "NARROWER" in capped
    assert "Do NOT re-request" in capped


def test_reference_call_uses_reference_cap_not_general_cap():
    """call_agent on a REFERENCE module honors reference_result_chars, even
    when the general cap is much larger (here: huge general cap, tiny
    reference cap -> the reference result must still be cut)."""
    call = _tool_by_name(
        make_core_tools(max_result_chars=10_000, reference_result_chars=10),
        "call_agent",
    )
    # An unknown dm7 method returns a deterministic error payload > 10 chars
    # (offline, no reference DB needed) — it must be cut at the REFERENCE cap.
    out = call.invoke({
        "agent_name": "dm7", "method": "no_such_method", "parameters": {},
    })
    assert _TRUNC_MARKER in out
    body, _, _ = out.partition("\n...")
    assert len(body) == 10


def test_calc_call_is_uncapped_even_under_tiny_general_cap():
    """A numeric/calc call_agent result is NEVER truncated — capping it risks
    cutting a number for no token benefit."""
    call = _tool_by_name(make_core_tools(max_result_chars=10), "call_agent")
    out = call.invoke({
        "agent_name": "bearing_capacity",
        "method": "bearing_capacity_analysis",
        "parameters": {
            "width": 2.0, "length": 2.0, "depth": 1.5, "shape": "square",
            "friction_angle": 30.0, "cohesion": 5.0, "unit_weight": 18.0,
        },
    })
    assert len(out) > 10  # well past the general cap...
    assert _TRUNC_MARKER not in out  # ...yet untouched
    result = json.loads(out)  # intact JSON, valid numbers
    assert result["q_ultimate_kPa"] > 0


def test_reference_cap_defaults_and_floors():
    """Default reference cap is 16000 and never falls BELOW the general cap;
    explicit values are used as-is; global-disable disables it too."""
    assert DEFAULT_REFERENCE_RESULT_CHARS == 16_000
    # Default: 16000, floored at the general cap when that is raised higher.
    assert _resolve_reference_cap(8_000, None) == 16_000
    assert _resolve_reference_cap(20_000, None) == 20_000
    # Explicit value wins.
    assert _resolve_reference_cap(8_000, 12_345) == 12_345
    # Truncation disabled globally -> disabled for reference reads too.
    assert _resolve_reference_cap(0, 12_345) == 0
    assert _resolve_reference_cap(-1, None) == 0


def test_result_cap_for_module_routes_by_module_kind():
    """Reference modules -> reference cap; calc modules -> uncapped (0)."""
    for ref_mod in ("dm7", "gec10", "reference_db", "figure_db"):
        assert ref_mod in REFERENCE_MODULES
        assert _result_cap_for_module(ref_mod, 8_000, 16_000) == 16_000
    assert _result_cap_for_module("bearing_capacity", 8_000, 16_000) == 0
    assert _result_cap_for_module("slope_stability", 8_000, 16_000) == 0


def test_catalog_tools_keep_general_cap_when_reference_cap_raised():
    """list_agents/list_methods/describe_method stay on the GENERAL cap even
    when reference_result_chars is huge (only reference READS get the larger
    budget)."""
    capped = _tool_by_name(
        make_core_tools(max_result_chars=50, reference_result_chars=100_000),
        "list_agents",
    ).invoke({})
    assert _TRUNC_MARKER in capped
    body, _, _ = capped.partition("\n...")
    assert len(body) == 50


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
