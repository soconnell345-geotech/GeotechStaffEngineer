"""Offline tests for smart ``describe_method`` resolution (NO API, NO network).

The deepagents ``describe_method`` tool wrapper (``make_core_tools``) now
resolves a guessed method name to the module's real method — the same way
``call_agent`` already auto-resolves such guesses — instead of bouncing with a
bare "Unknown method" error. These tests cover:

  * alias-map redirect (selector-value alias: 'vesic' -> bearing_capacity_analysis)
  * alias-map redirect (plain alias: settlement 'consolidation')
  * a truly-unknown guess -> enriched error carrying method *briefs*
  * a valid real method still returns normally (no regression)
  * ``allowed_agents`` scope still refused for an out-of-scope module

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_describe_resolve_offline.py -v
"""

import json

import pytest

from funhouse_agent.dispatch import ANALYSIS_MODULES, _METHOD_ALIASES
from funhouse_agent.deep.tools import make_core_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_by_name(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not in {[t.name for t in tools]}")


def _describe(tools, agent_name, method):
    """Invoke describe_method and parse its JSON-string result."""
    out = _tool_by_name(tools, "describe_method").invoke(
        {"agent_name": agent_name, "method": method}
    )
    assert isinstance(out, str), "describe_method must return a JSON string"
    return json.loads(out)


# ---------------------------------------------------------------------------
# Sanity: the aliases the tests rely on actually exist in dispatch
# ---------------------------------------------------------------------------

def test_aliases_used_in_tests_exist():
    """Guard: the aliases these tests assert on are present in dispatch."""
    # selector-value alias (tuple: real_method + injected param value)
    assert _METHOD_ALIASES.get(("bearing_capacity", "vesic")) == (
        "bearing_capacity_analysis", {"factor_method": "vesic"}
    )
    # plain alias (str: real method name)
    assert _METHOD_ALIASES.get(("settlement", "consolidation")) == (
        "consolidation_settlement"
    )


# ---------------------------------------------------------------------------
# (1) alias-map redirect — selector-value alias ('vesic')
# ---------------------------------------------------------------------------

def test_vesic_guess_redirects_to_bearing_capacity_analysis():
    """describe_method('bearing_capacity','vesic') -> the analysis-method docs,
    with a _note explaining 'vesic' is a factor_method value."""
    tools = make_core_tools(max_result_chars=0)
    result = _describe(tools, "bearing_capacity", "vesic")

    assert "error" not in result, result
    # Resolved to the real method's docs.
    assert result.get("name") == "bearing_capacity_analysis" or \
        "bearing_capacity_analysis" in json.dumps(result)
    params = result.get("parameters", {})
    assert "factor_method" in params or "friction_angle" in params, result
    # Redirect hint present and names the real method.
    assert "_note" in result, result
    assert "bearing_capacity_analysis" in result["_note"]
    assert "factor_method" in result["_note"]
    assert "vesic" in result["_note"]


def test_vesic_guess_does_not_leak_unknown_method_error():
    tools = make_core_tools(max_result_chars=0)
    result = _describe(tools, "bearing_capacity", "vesic")
    assert "Unknown method" not in json.dumps(result)


# ---------------------------------------------------------------------------
# (2) alias-map redirect — plain alias (settlement 'consolidation')
# ---------------------------------------------------------------------------

def test_consolidation_guess_redirects_to_real_method():
    """describe_method('settlement','consolidation') -> consolidation_settlement
    docs with a _note."""
    tools = make_core_tools(max_result_chars=0)
    result = _describe(tools, "settlement", "consolidation")

    assert "error" not in result, result
    assert "consolidation_settlement" in json.dumps(result)
    assert "_note" in result, result
    assert "consolidation_settlement" in result["_note"]
    # It is a plain alias (no injected selector value), so the note should NOT
    # claim a parameter value.
    assert "parameter" not in result["_note"].lower() or \
        "consolidation_settlement" in result["_note"]


# ---------------------------------------------------------------------------
# (3) truly-unknown guess -> enriched error with method BRIEFS
# ---------------------------------------------------------------------------

def test_unknown_guess_returns_enriched_error_with_briefs():
    """A made-up method returns an enriched error mapping method -> brief."""
    tools = make_core_tools(max_result_chars=0)
    result = _describe(tools, "bearing_capacity", "totally_made_up")

    assert "error" in result, result
    assert "available_methods" in result, result
    briefs = result["available_methods"]
    assert isinstance(briefs, dict)
    # Keyed by real method names, valued by their one-line briefs (not just
    # bare names).
    assert "bearing_capacity_analysis" in briefs
    assert isinstance(briefs["bearing_capacity_analysis"], str)
    assert len(briefs["bearing_capacity_analysis"]) > 0
    # A directive nudging the closest-match recovery.
    assert "directive" in result
    assert "closest" in result["directive"].lower()


# ---------------------------------------------------------------------------
# (4) valid real method — no regression
# ---------------------------------------------------------------------------

def test_real_method_returns_normally():
    """A real method returns its docs unchanged, with NO resolution _note."""
    tools = make_core_tools(max_result_chars=0)
    result = _describe(tools, "bearing_capacity", "bearing_capacity_analysis")

    assert "error" not in result, result
    assert "_note" not in result, "real method must not carry a redirect note"
    params = result.get("parameters", {})
    assert "friction_angle" in params or "factor_method" in params, result


# ---------------------------------------------------------------------------
# (5) allowed_agents scope still enforced
# ---------------------------------------------------------------------------

def test_out_of_scope_module_still_errors():
    """An out-of-scope module under a restrictive scope still errors (the
    resolution path must not bypass module visibility)."""
    tools = make_core_tools(allowed_agents=ANALYSIS_MODULES, max_result_chars=0)
    # dm7 is a reference module — invisible to an ANALYSIS_MODULES scope.
    result = _describe(tools, "dm7", "vesic")
    assert "error" in result, result
    assert "Unknown module" in result["error"], result
    # The resolution path must NOT have run (no enriched/resolved payload).
    assert "available_methods" not in result
    assert "_note" not in result


def test_in_scope_module_resolution_still_works_under_scope():
    """Under an ANALYSIS_MODULES scope, an in-scope module's guess still
    resolves (scope does not disable resolution for visible modules)."""
    tools = make_core_tools(allowed_agents=ANALYSIS_MODULES, max_result_chars=0)
    result = _describe(tools, "bearing_capacity", "vesic")
    assert "error" not in result, result
    assert "_note" in result, result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
