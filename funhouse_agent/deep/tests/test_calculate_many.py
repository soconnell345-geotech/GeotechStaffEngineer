"""``calculate`` takes several expressions in one call (BACKLOG GM-169).

Live wave w13: F18's calc helper wrote ``"[1139.77/3, 1139.77/3*4, ...]"``
as one expression, was refused ("rejected: List"), then made 10 single
calls; F55j's made 17. Now a list, a ``{name: expression}`` map (each name a
variable of the expressions after it) or a string that is a list all come
back as ``results``, and the single form is unchanged.
"""

import json
import math

import pytest

from funhouse_agent.deep.calculate_tool import (CALCULATE_DESCRIPTION,
                                                MAX_EXPRESSIONS,
                                                calculate_any, evaluate,
                                                make_calculate_tool)


def test_the_single_form_is_unchanged():
    out = calculate_any("(qt - s) / Nkt", {"qt": 1250, "s": 95, "Nkt": 14})
    assert out == evaluate("(qt - s) / Nkt", {"qt": 1250, "s": 95, "Nkt": 14})
    assert out["value"] == pytest.approx(82.5)
    assert set(out) == {"value", "expression", "variables"}


def test_a_list_of_expressions():
    out = calculate_any(["qn/3 + 19", "qa*4", "1/0"],
                        {"qn": 1139.77, "qa": 386.26})
    vals = [r.get("value") for r in out["results"]]
    assert vals[0] == pytest.approx(1139.77 / 3 + 19)
    assert vals[1] == pytest.approx(386.26 * 4)
    assert out["results"][2]["error"] == "division by zero"   # its own alone
    assert [r["expression"] for r in out["results"]] == \
        ["qn/3 + 19", "qa*4", "1/0"]
    assert out["variables"] == {"qn": 1139.77, "qa": 386.26}
    assert all("variables" not in r for r in out["results"])


def test_the_list_written_inside_the_string_as_in_w13():
    out = calculate_any("[qn/3+19, qa*4, (qn/3+19)*4]",
                        {"qn": 931.24, "qa": 386.26})
    assert [r["value"] for r in out["results"]] == pytest.approx(
        [931.24 / 3 + 19, 386.26 * 4, (931.24 / 3 + 19) * 4])
    assert out["results"][0]["expression"] == "qn / 3 + 19"


def test_named_expressions_build_on_each_other():
    out = calculate_any(
        {"Ka": "tan(radians(45 - phi/2))**2", "Pa": "0.5*Ka*g*H**2"},
        {"phi": 30, "g": 18, "H": 4.5})
    ka = math.tan(math.radians(30)) ** 2
    assert out["results"]["Ka"]["value"] == pytest.approx(ka)
    assert out["results"]["Pa"]["value"] == pytest.approx(0.5 * ka * 18 * 4.5 ** 2)
    assert out["variables"] == {"phi": 30.0, "g": 18.0, "H": 4.5}


def test_a_name_that_is_already_a_variable_is_refused_alone():
    out = calculate_any({"H": "2*H", "B": "H + 1"}, {"H": 3})
    assert "already a variable" in out["results"]["H"]["error"]
    assert out["results"]["B"]["value"] == 4.0


def test_a_bad_variable_is_said_once():
    out = calculate_any(["a + 1", "a * 2"], {"a": "x"})
    assert "variable 'a'" in out["error"] and "results" not in out


def test_limits():
    assert "limit" in calculate_any(["1"] * (MAX_EXPRESSIONS + 1))["error"]
    assert "at least one" in calculate_any([])["error"]


def test_a_list_inside_a_function_is_still_one_expression():
    out = calculate_any("max(a, b)", {"a": 1, "b": 2})
    assert out["value"] == 2.0


def test_the_tool_takes_a_list_and_a_map():
    tool = make_calculate_tool()
    out = json.loads(tool.invoke({"expression": ["a+1", "a*3"],
                                  "variables": {"a": 2}}))
    assert [r["value"] for r in out["results"]] == [3.0, 6.0]
    out = json.loads(tool.invoke({"expression": {"x": "a+1", "y": "x*2"},
                                  "variables": {"a": 2}}))
    assert out["results"]["y"]["value"] == 6.0
    out = json.loads(tool.invoke({"expression": "a+1", "variables": {"a": 2}}))
    assert out["value"] == 3.0
    shown = tool.args["expression"]
    kinds = {s.get("type") for s in shown.get("anyOf", [shown])}
    assert {"string", "array", "object"} <= kinds


def test_the_description_says_so():
    assert "Several formulas go in ONE call" in CALCULATE_DESCRIPTION
    assert "results" in CALCULATE_DESCRIPTION
