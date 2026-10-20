"""GM-220: two printed options the tools lacked.

Drilled shafts: decomposed or fractured rock below the top of rock "not
considered part of the socket for design purposes" (GEC-10 p. 8-15),
``fractured_rock_thickness``. Aggregate piers: the settlement ratio by
E_c / E_s off GEC-13 Fig. 5-21 (p. 5-36), ``settlement_method``
'weighted_modulus', 'balaam_rigid' or 'balaam_flexible', which take
``E_column``; the stress-ratio default still refuses it (GM-179).
"""

import json

import pytest

import funhouse_agent.adapters.calc_package as cp
from funhouse_agent import dispatch as D


def _call(agent, method, params):
    r = D.call_agent(agent, method, params)
    return json.loads(r) if isinstance(r, str) else r


_SHAFT = {"diameter": 1.2, "shaft_length": 14.0, "gwt_depth": 2.0,
          "layers": [{"thickness": 8.0, "soil_type": "cohesive",
                      "unit_weight": 18.5, "cu": 75.0},
                     {"thickness": 10.0, "soil_type": "rock",
                      "unit_weight": 23.0, "qu": 20000.0, "RQD": 80.0}]}


@pytest.mark.parametrize("method", ["drilled_shaft_capacity",
                                    "lrfd_capacity"])
def test_the_shaft_tools_take_the_rock_not_in_the_socket(method):
    base = {**_SHAFT, **({"tip_soil_type": "rock"}
                         if method == "lrfd_capacity" else {})}
    r0 = _call("drilled_shaft", method, base)
    r = _call("drilled_shaft", method,
              {**base, "fractured_rock_thickness": 1.5})
    assert r["Q_ultimate_kN"] < r0["Q_ultimate_kN"]
    assert r0["defaults_used"]["fractured_rock_thickness"] == 0.0
    if method == "drilled_shaft_capacity":
        assert r0["Q_ultimate_kN"] == pytest.approx(78489.5, abs=0.1)
        assert r["Q_ultimate_kN"] == pytest.approx(73257.0, abs=0.1)
        (c,) = [c for c in r["choices"] if "socket" in c["what"]]
        assert c["how"] == "given" and "p. 8-15" in c["basis"]


def test_the_shaft_curve_and_the_refusal_without_rock():
    r = _call("drilled_shaft", "capacity_vs_depth",
              {**_SHAFT, "fractured_rock_thickness": 1.5, "n_points": 5})
    assert r["capacity_vs_depth"][-1]["Q_skin_kN"] == pytest.approx(
        16708.3, abs=0.1)
    err = _call("drilled_shaft", "drilled_shaft_capacity",
                {**_SHAFT, "layers": _SHAFT["layers"][:1],
                 "shaft_length": 7.0, "fractured_rock_thickness": 1.0})
    assert "no rock layer" in err["error"]


def test_the_shaft_package_prints_the_zone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _call("calc_package", "drilled_shaft_package",
              {**_SHAFT, "fractured_rock_thickness": 1.5,
               "output_path": str(tmp_path / "ds.html")})
    assert r.get("status") == "success", r
    html = (tmp_path / "ds.html").read_text(encoding="utf-8")
    assert "Rock Not Considered Part of the Socket" in html
    assert "p. 8-15" in html


_PIERS = {"column_diameter": 0.9, "spacing": 2.1, "S_unreinforced": 150.0}


@pytest.mark.parametrize("method, settlement", [
    ("weighted_modulus", 60.0), ("balaam_rigid", 74.3),
    ("balaam_flexible", 83.8)])
def test_the_aggregate_piers_take_the_column_modulus_by_fig_5_21(
        method, settlement):
    r = _call("ground_improvement", "aggregate_piers",
              {**_PIERS, "settlement_method": method,
               "E_column": 50000.0, "E_soil": 5000.0})
    assert r["settlement_improved_mm"] == pytest.approx(settlement,
                                                        abs=0.05)
    assert r["modular_ratio"] == 10.0
    # n acts only on a bearing here, so it is no default used
    assert "n" not in r.get("defaults_used", {})
    alias = _call("ground_improvement", "aggregate_piers",
                  {**_PIERS, "settlement_method": method,
                   "column_modulus": 50000.0, "soil_modulus": 5000.0})
    assert alias["settlement_improved_mm"] == r["settlement_improved_mm"]


def test_the_default_method_is_unchanged_and_still_refuses_e_column():
    r = _call("ground_improvement", "aggregate_piers", dict(_PIERS))
    assert r["settlement_improved_mm"] == pytest.approx(90.0, abs=0.05)
    assert r["defaults_used"]["settlement_method"] == "stress_ratio"
    assert "settlement_ratio" not in r
    for name in ("E_column", "column_modulus"):
        err = _call("ground_improvement", "aggregate_piers",
                    {**_PIERS, name: 50000.0})["error"]
        assert "does not take" in err and name in err and "Give n" in err
        assert "balaam_rigid" in err


def test_the_fig_5_21_methods_need_both_moduli():
    err = _call("ground_improvement", "aggregate_piers",
                {**_PIERS, "settlement_method": "balaam_rigid",
                 "E_column": 50000.0})["error"]
    assert "E_soil" in err
    err = _call("ground_improvement", "aggregate_piers",
                {**_PIERS, "settlement_method": "balaam_flexible",
                 "E_column": 80000.0, "E_soil": 5000.0})["error"]
    assert "10 only" in err


def test_the_package_takes_e_pier_for_the_fig_5_21_methods(tmp_path,
                                                          monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = {"method": "aggregate_piers", **_PIERS,
         "settlement_method": "balaam_rigid", "E_pier": 37500.0,
         "E_soil": 5000.0, "output_path": str(tmp_path / "gi.html")}
    assert _call("calc_package", "ground_improvement_package",
                 p).get("status") == "success"
    html = (tmp_path / "gi.html").read_text(encoding="utf-8")
    assert "Settlement Ratio, Balaam" in html and "37500" in html
    with pytest.raises(ValueError, match="Give n"):
        cp.METHOD_REGISTRY["ground_improvement_package"](
            {"method": "aggregate_piers", **_PIERS, "E_pier": 37500.0})
