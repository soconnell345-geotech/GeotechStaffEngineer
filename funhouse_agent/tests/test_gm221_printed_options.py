"""GM-221: the tools against GEC-10 Example 10-4 and GEC-13's Priebe curve.

Drilled shafts: the base in rock is on the socket diameter, as GEC-10
Example 10-4 takes it (p. 10-28), and the rock side resistance's C is an
input, ``rock_side_C`` (GEC-10 Eq. 10-21; default 1.0 since GM-244, 0.65
before, Eq. 10-22 with alpha_E = 1.0). Example 10-4's statement that the
rock below the tip is competent is ``rock_base_conditions_met``. Aggregate
piers: ``priebe_improvement_factor`` method
'fig_5_21' reads GEC-13's printed Priebe curve, Fig. 5-21 curve (4); the
default is still Priebe's formula. The aggregate pier calc package cites
GEC-13's stress reduction method, not "Priebe (1995)", for its settlement.
"""

import json

import pytest

from funhouse_agent import dispatch as D

FT, KIP = 0.3048, 4.4482216

#: GEC-10 Example 10-4 (pp. 10-27, 10-28)
_EX_10_4 = {
    "diameter": 6.5 * FT, "socket_diameter": 6.0 * FT,
    "casing_depth": 34 * FT, "shaft_length": 44 * FT,
    "layers": [{"thickness": 34 * FT, "soil_type": "cohesionless",
                "unit_weight": 19.0, "phi": 32.0, "N60": 20},
               {"thickness": 20 * FT, "soil_type": "rock",
                "unit_weight": 23.0, "qu": 2500 * 6.894757, "RQD": 93}],
    # "Rock coring and local experience indicate the rock mass below the
    # socket tip is competent" (p. 10-28)
    "rock_base_conditions_met": True}


def _call(agent, method, params):
    r = D.call_agent(agent, method, params)
    return json.loads(r) if isinstance(r, str) else r


def test_example_10_4_through_the_tools():
    r = _call("drilled_shaft", "lrfd_capacity", dict(_EX_10_4))
    assert r["Q_side_rock_kN"] / KIP == pytest.approx(5202, abs=1)
    assert r["Q_tip_kN"] / KIP == pytest.approx(25447, abs=1)
    assert r["lrfd"]["phi_Qt_kN"] / KIP == pytest.approx(12723, abs=1)
    assert r["base_diameter_m"] == pytest.approx(6.0 * FT, abs=1e-4)
    c = [c for c in r["choices"] if c["what"].startswith("rock side")][0]
    assert c["how"] == "default"
    assert r["lrfd"]["phi_tip"] == 0.50     # the tip's own soil, rock
    err = _call("drilled_shaft", "lrfd_capacity",
                {**_EX_10_4, "tip_soil_type": "cohesive"})["error"]
    assert "not the soil the base was computed in" in err
    # GM-278: Table 8-4's 0.50 governs; 0.55 is never offered for rock
    assert r["lrfd"]["phi_Qs_rock_kN"] == pytest.approx(
        0.50 * r["Q_side_rock_kN"], abs=0.1)
    text = json.dumps(r["choices"])
    assert "Table 8-4" in text and "prints 0.50" in text
    assert "0.55" not in text
    assert not any("resistance factor for side resistance in rock"
                   in c["what"] for c in r["choices"])


def test_the_default_c_is_listed_and_the_curve_carries_c():
    r = _call("drilled_shaft", "drilled_shaft_capacity", dict(_EX_10_4))
    assert r["defaults_used"]["rock_side_C"] == 1.0
    assert r["Q_side_rock_kN"] / KIP == pytest.approx(5202, abs=1)
    old = _call("drilled_shaft", "drilled_shaft_capacity",
                {**_EX_10_4, "rock_side_C": 0.65})
    assert old["Q_side_rock_kN"] / KIP == pytest.approx(0.65 * 5202.75,
                                                       abs=1)
    curve = _call("drilled_shaft", "capacity_vs_depth",
                  {**_EX_10_4, "rock_side_C": 0.65, "depth_min": 40 * FT,
                   "n_points": 2})["capacity_vs_depth"]
    assert curve[-1]["Q_skin_kN"] / KIP == pytest.approx(0.65 * 5202.75,
                                                        abs=1)
    no_statement = {k: v for k, v in _EX_10_4.items()
                    if k != "rock_base_conditions_met"}
    err = _call("drilled_shaft", "drilled_shaft_capacity",
                no_statement)["error"]
    assert "RQD 93 % at" in err and "rock_base_conditions_met" in err
    err = _call("drilled_shaft", "drilled_shaft_capacity",
                {**_EX_10_4, "rock_side_C": -1.0})["error"]
    assert "rock_side_C" in err


def test_the_shaft_package_prints_the_socket_base_and_c(tmp_path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _call("calc_package", "drilled_shaft_package",
              {**_EX_10_4, "output_path": str(tmp_path / "ds.html")})
    assert r.get("status") == "success", r
    html = (tmp_path / "ds.html").read_text(encoding="utf-8")
    assert "D_sock" in html and "Example 10-4" in html
    assert "C = 1 (rock_side_C)" in html


def test_the_priebe_tool_reads_the_printed_curve():
    r = _call("ground_improvement", "priebe_improvement_factor",
              {"as_ratio": 0.277, "method": "fig_5_21"})
    assert r["n0"] == pytest.approx(2.55, abs=0.03)
    assert r["method"] == "fig_5_21" and "curve (4)" in r["note"]
    (c,) = r["choices"]
    assert c["how"] == "given" and "p. 5-36" in c["basis"]
    assert "defaults_used" not in r
    err = _call("ground_improvement", "priebe_improvement_factor",
                {"as_ratio": 0.277, "method": "fig_5_21",
                 "phi_column": 40.0})["error"]
    assert "does not take phi_column" in err and "37.5" in err


def test_the_priebe_default_is_still_the_formula_and_says_so():
    r = _call("ground_improvement", "priebe_improvement_factor",
              {"as_ratio": 0.277})
    assert r["n0"] == pytest.approx(3.06, abs=0.005)
    assert r["method"] == "formula"
    assert r["defaults_used"] == {"method": "formula", "phi_column": 42.5,
                                  "nu_soil": pytest.approx(1 / 3)}
    assert "no correction for the compressibility" in r["note"]
    assert "GEC-13 prints no formula" in r["note"]
    what = {c["what"]: c["how"] for c in r["choices"]}
    assert what == {"Priebe improvement factor": "default",
                    "column friction angle phi_c": "default"}
    # the default angle's basis is said plainly, with the printed curve's
    # angle and what the formula gives at it
    (phi,) = [c for c in r["choices"] if c["what"].startswith("column")]
    assert "GEC-13 prints no column friction angle" in phi["basis"]
    assert phi["alternatives"] == [
        "phi_column 37.5: the angle of GEC-13's printed curve, n0 = 2.58 "
        "here"]


def test_the_pier_package_cites_the_stress_reduction_method(tmp_path,
                                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = _call("calc_package", "ground_improvement_package",
              {"method": "aggregate_piers", "column_diameter": 0.9,
               "spacing": 2.1, "S_unreinforced": 150.0,
               "output_path": str(tmp_path / "gi.html")})
    assert r.get("status") == "success", r
    html = (tmp_path / "gi.html").read_text(encoding="utf-8")
    assert "Sec. 4.1.2.6, p. 5-36: the stress reduction method" in html
    assert "Priebe (1995); GEC-13 unit-cell model" not in html
