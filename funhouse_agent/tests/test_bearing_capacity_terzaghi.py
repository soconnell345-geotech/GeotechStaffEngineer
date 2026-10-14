"""GM-30: an agent can ask the bearing_capacity tool for Terzaghi (1943).

Testers asked for Terzaghi's bearing capacity often and a memo shipped with
N-gamma blank because the module had no Terzaghi option. The method, its two
cited N-gamma sources and local shear are reachable by name; the result says
which N-gamma was used and gives the other.
"""

import json

import pytest

from funhouse_agent.dispatch import call_agent, describe_method

_SQUARE = {"width": 2.0, "shape": "square", "depth": 1.5,
           "friction_angle": 25, "cohesion": 20, "unit_weight": 16.5}


def test_analysis_by_factor_method():
    out = call_agent("bearing_capacity", "bearing_capacity_analysis",
                     {**_SQUARE, "factor_method": "terzaghi"})
    assert "error" not in out, out
    assert out["method"] == "Terzaghi (1943)"
    assert out["q_ultimate_kPa"] == pytest.approx(1078.3, rel=0.001)
    assert out["Ngamma"] == pytest.approx(8.34)
    assert out["Ngamma_alternatives"]["coduto"] == pytest.approx(9.18, abs=0.01)
    assert "Kumbhojkar (1993)" in out["notes"][0]
    json.dumps(out)


def test_method_named_terzaghi_runs_terzaghi():
    """Until 2026-10-14 the guess bearing_capacity/terzaghi routed to the
    full analysis with the default factor_method and returned a Vesic
    answer (1,374 kPa here) for a Terzaghi question."""
    out = call_agent("bearing_capacity", "terzaghi", dict(_SQUARE))
    assert out.get("factor_method") == "terzaghi", out
    assert out["q_ultimate_kPa"] == pytest.approx(1078.3, rel=0.001)


def test_coduto_and_local_shear_through_the_adapter():
    out = call_agent("bearing_capacity", "bearing_capacity_analysis",
                     {**_SQUARE, "factor_method": "terzaghi",
                      "ngamma_method": "coduto", "failure_mode": "local"})
    assert "error" not in out, out
    assert out["ngamma_method"] == "coduto"
    assert out["failure_mode"] == "local"
    assert out["c_used_kPa"] == pytest.approx(20 * 2 / 3, abs=1e-3)


def test_terzaghi_refuses_inclined_load():
    out = call_agent("bearing_capacity", "bearing_capacity_analysis",
                     {**_SQUARE, "factor_method": "terzaghi",
                      "load_inclination": 10})
    assert "error" in out and "Table 5-3" in out["error"]


def test_factor_lookup():
    out = call_agent("bearing_capacity", "bearing_capacity_factors",
                     {"friction_angle": 30, "method": "terzaghi"})
    assert (out["Nc"], out["Nq"], out["Ngamma"]) == (37.16, 22.46, 19.13)
    assert out["Ngamma_alternatives"] == {"coduto": 20.12}
    local = call_agent("bearing_capacity", "bearing_capacity_factors",
                       {"friction_angle": 35, "method": "terzaghi",
                        "failure_mode": "local"})
    assert local["phi_used_deg"] == pytest.approx(25.02, abs=0.01)
    vesic = call_agent("bearing_capacity", "bearing_capacity_factors",
                       {"friction_angle": 30})
    assert vesic["method"] == "vesic" and vesic["Nc"] == pytest.approx(30.14)


def test_described_for_the_agent():
    info = describe_method("bearing_capacity", "bearing_capacity_analysis")
    params = info["parameters"]
    assert "terzaghi" in params["factor_method"]["allowed_values"]
    assert {"kumbhojkar", "coduto"} <= set(
        params["ngamma_method"]["allowed_values"])
    assert params["failure_mode"]["allowed_values"] == ["general", "local"]
    f = describe_method("bearing_capacity", "bearing_capacity_factors")
    assert "terzaghi" in f["parameters"]["method"]["allowed_values"]


def test_calc_package_carries_terzaghi(tmp_path):
    pytest.importorskip("matplotlib")
    out = call_agent("calc_package", "bearing_capacity_package",
                     {**_SQUARE, "factor_method": "terzaghi",
                      "output_path": str(tmp_path / "terzaghi.html")})
    assert "error" not in out, out
    assert out["q_ultimate_kPa"] == pytest.approx(1078.3, abs=0.2)
    html = (tmp_path / "terzaghi.html").read_text(encoding="utf-8")
    assert "Terzaghi" in html and "Kumbhojkar" in html
