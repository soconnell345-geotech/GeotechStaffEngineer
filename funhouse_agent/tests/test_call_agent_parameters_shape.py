"""GM-189: call_agent's ``parameters`` in a shape other than an object.

A calc helper sent ``parameters`` as a string that was not JSON (a JSON
object with code after it). The deep agent's schema refused it with
pydantic's "Input should be a valid dictionary", echoing the whole
argument back, and said nothing of the shape expected; the v1 dispatcher
ran the method with no inputs at all, and the native-tools route raised.
Now every route reads a string holding a JSON object as that object, and
refuses any other shape with the shape expected and, where the method is
known, its parameter names, without echoing the argument.
"""

import json

from langchain_core.utils.function_calling import convert_to_openai_tool

from funhouse_agent import dispatch as D
from funhouse_agent.deep.tools import make_core_tools
from funhouse_agent.native_tools import dispatch_native_tool

AGENT, METHOD = "bearing_capacity", "bearing_capacity_factors"
GOOD = {"friction_angle": 30}


def _call_agent_tool():
    return {t.name: t for t in make_core_tools()}["call_agent"]


def test_a_string_holding_a_json_object_runs_as_that_object():
    want = D.call_agent(AGENT, METHOD, GOOD)
    assert "error" not in want
    assert D.call_agent(AGENT, METHOD, json.dumps(GOOD)) == want


def test_none_and_a_blank_string_are_no_inputs():
    assert D.parameters_object(None) == {}
    assert D.parameters_object("  ") == {}
    assert D.parameters_object(GOOD) is GOOD


def test_a_string_that_is_not_json_is_refused_with_the_shape_expected():
    bad = json.dumps({"html": "<p>" + "x" * 9000 + "</p>"})[:-1] \
        + ".format(a=1)}"
    out = D.call_agent(AGENT, METHOD, bad)
    assert "NOT run" in out["error"]
    assert "not JSON" in out["error"]
    assert ".format(a=1)" in out["error"]          # where the JSON stops
    assert "object of named inputs" in out["expected"]
    assert "ONE string value" in out["expected"]
    assert "no code" in out["expected"]
    assert out["required_parameters"] == ["friction_angle"]
    assert len(json.dumps(out)) < 1500             # the argument not echoed


def test_other_shapes_are_refused_by_what_they_are():
    out = D.call_agent(AGENT, METHOD, "[30]")
    assert "a string holding a list" in out["error"]
    out = D.call_agent(AGENT, METHOD, [30])
    assert "came as a list" in out["error"]
    assert out["required_parameters"] == ["friction_angle"]


def test_a_method_not_known_gets_the_shape_without_names():
    out = D.call_agent(AGENT, "no_such_method", "not json")
    assert "NOT run" in out["error"] and "expected" in out
    assert "required_parameters" not in out


def test_v1_dispatch_tool_refuses_a_string_instead_of_running_empty():
    class Call:
        tool_name = "call_agent"
        arguments = {"agent_name": AGENT, "method": METHOD,
                     "parameters": "friction_angle=30"}
    out = json.loads(D.dispatch_tool(Call()))
    assert "NOT run" in out["error"]


def test_native_tools_route_refuses_a_string_and_reads_a_json_one():
    out = json.loads(dispatch_native_tool("call_agent", {
        "agent_name": AGENT, "method": METHOD, "parameters": "phi = 30"}))
    assert "NOT run" in out["error"]
    out = json.loads(dispatch_native_tool("call_agent", {
        "agent_name": AGENT, "method": METHOD,
        "parameters": json.dumps(GOOD)}))
    assert "error" not in out


def test_deep_tool_shows_an_object_and_refuses_a_string_itself():
    tool = _call_agent_tool()
    shown = convert_to_openai_tool(tool)["function"]["parameters"]
    assert shown["properties"]["parameters"]["anyOf"] == [
        {"additionalProperties": True, "type": "object"}, {"type": "null"}]
    out = json.loads(tool.invoke({"agent_name": AGENT, "method": METHOD,
                                  "parameters": '{"friction_angle": 30}x'}))
    assert "NOT run" in out["error"]
    assert out["required_parameters"] == ["friction_angle"]


def test_deep_tool_reads_a_json_string_and_keeps_flattened_keys():
    tool = _call_agent_tool()
    want = json.loads(tool.invoke({"agent_name": AGENT, "method": METHOD,
                                   "parameters": GOOD}))
    got = json.loads(tool.invoke({"agent_name": AGENT, "method": METHOD,
                                  "parameters": json.dumps(GOOD)}))
    assert got == want
    got = json.loads(tool.invoke({"agent_name": AGENT, "method": METHOD,
                                  "parameters": "{}", "friction_angle": 30}))
    assert got == want


def test_html_to_pdf_says_the_html_is_one_finished_string():
    doc = D.describe_method("calc_package", "html_to_pdf")
    html = doc["parameters"]["html"]["description"]
    assert "ONE string value" in html and "nothing in the call is run" in html
