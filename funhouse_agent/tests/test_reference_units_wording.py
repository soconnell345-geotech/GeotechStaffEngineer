"""The units wording is true for whoever reads it (2026-11-10 review,
item 2).

A reference lookup uses the units of the manual it comes from (many US
customary); the references helper was told "All units are SI". A tool
scope of reference modules only now says so, and the helper is told to
give each value with its unit and convert only with ``calculate``; the
calculation scopes keep the SI rule.
"""

import json

from funhouse_agent.dispatch import ANALYSIS_MODULES, REFERENCE_MODULES


def _call_agent_tool(scope):
    from funhouse_agent.deep.tools import make_core_tools
    return next(t for t in make_core_tools(allowed_agents=scope)
                if t.name == "call_agent")


def _parameters_description(tool):
    schema = tool.args_schema.model_json_schema()
    return schema["properties"]["parameters"]["description"]


def test_the_reference_scope_is_told_the_manuals_units():
    tool = _call_agent_tool(REFERENCE_MODULES)
    assert "All units are SI" not in tool.description
    assert "units of the manual" in tool.description
    assert "US customary" in tool.description
    desc = _parameters_description(tool)
    assert "SI" not in desc and "units of the manual" in desc


def test_the_calculation_scopes_keep_si():
    for scope in (ANALYSIS_MODULES, None,
                  set(REFERENCE_MODULES) | {"bearing_capacity"}):
        tool = _call_agent_tool(scope)
        assert "All units are SI" in tool.description
        assert "all SI units" in _parameters_description(tool)


def test_the_reference_call_still_nests_stray_keys():
    """The reference schema keeps the extra-keys behaviour: a flattened
    input reaches the method."""
    tool = _call_agent_tool(REFERENCE_MODULES)
    out = json.loads(tool.invoke({"agent_name": "gec12",
                                  "method": "list_chapters",
                                  "chapter": 7}))
    assert "error" not in out


def test_the_references_helper_states_units_and_converts_with_calculate():
    from funhouse_agent.deep.agent import build_references_subagent
    prompt = build_references_subagent()["system_prompt"]
    assert "units of the manual it comes from" in prompt
    assert "with its unit" in prompt
    assert "only with the `calculate` tool" in prompt


def test_the_v1_reference_prompt_rule_seven():
    from funhouse_agent.system_prompt import (
        REFERENCE_UNITS_RULE, SI_UNITS_RULE, build_system_prompt)
    ref = build_system_prompt(REFERENCE_MODULES)
    assert REFERENCE_UNITS_RULE in ref and SI_UNITS_RULE not in ref
    for scope in (None, ANALYSIS_MODULES):
        calc = build_system_prompt(scope)
        assert SI_UNITS_RULE in calc and REFERENCE_UNITS_RULE not in calc
