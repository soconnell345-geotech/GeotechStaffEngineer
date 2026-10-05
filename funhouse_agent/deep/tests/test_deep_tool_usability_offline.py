"""The deep agent's dispatch tools after the 2026-10 Foundry eval. Offline.

``list_methods`` called with no module was a pydantic "Field required" error;
``describe_method`` on a guessed name answered with every method's brief but
not the nearest one, and a method that lives on another module was not
pointed at.
"""

import json

from funhouse_agent.deep.tools import make_core_tools
from funhouse_agent.dispatch import ANALYSIS_MODULES


def _tools(**kw):
    return {t.name: t for t in make_core_tools(**kw)}


def test_list_methods_without_a_module_lists_the_modules():
    tool = _tools(allowed_agents=ANALYSIS_MODULES)["list_methods"]
    out = json.loads(tool.invoke({}))           # not a schema error
    assert "agent_name" in out["error"]
    assert "slope_stability" in out["modules"]
    assert "dm7" not in out["modules"]          # the scope still holds


def test_list_methods_schema_does_not_require_the_module():
    tool = _tools()["list_methods"]
    assert "agent_name" not in (tool.args_schema.model_json_schema()
                                .get("required") or [])


def test_list_methods_topic_category_through_the_tool():
    tool = _tools(allowed_agents=ANALYSIS_MODULES)["list_methods"]
    out = json.loads(tool.invoke({"agent_name": "fem2d",
                                  "category": "slope"}))
    assert "fem2d_slope_srm" in json.dumps(out["methods"])
    assert "note" in out


def test_describe_unknown_puts_the_closest_first():
    tool = _tools(allowed_agents=ANALYSIS_MODULES)["describe_method"]
    raw = tool.invoke({"agent_name": "lateral_pile",
                       "method": "composite_ei"})
    out = json.loads(raw)
    assert out["closest"][0] == "composite_section_ei"
    assert "available_methods" in out
    # the nearest names come before the long list, so a cut keeps them
    assert raw.index("closest") < raw.index("available_methods")


def test_describe_of_another_modules_method_shows_its_docs():
    tool = _tools(allowed_agents=ANALYSIS_MODULES)["describe_method"]
    out = json.loads(tool.invoke({"agent_name": "retaining_walls",
                                  "method": "apparent_earth_pressure"}))
    assert "parameters" in out
    assert "module 'soe' as 'apparent_pressure'" in out["_note"]


def test_describe_redirect_respects_scope():
    tool = _tools(allowed_agents=["retaining_walls"])["describe_method"]
    out = json.loads(tool.invoke({"agent_name": "retaining_walls",
                                  "method": "apparent_earth_pressure"}))
    assert "error" in out and "soe" not in json.dumps(out.get("closest"))
    assert "_note" not in out


def test_describe_obvious_synonym_resolves():
    tool = _tools(allowed_agents=ANALYSIS_MODULES)["describe_method"]
    out = json.loads(tool.invoke({"agent_name": "axial_pile",
                                  "method": "driven_pile_capacity"}))
    assert "axial_pile_capacity" in out["_note"]
    assert "parameters" in out
