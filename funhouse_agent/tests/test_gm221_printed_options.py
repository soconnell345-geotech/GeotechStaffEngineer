"""GM-221: the tools against GEC-10 Example 10-4 and GEC-13's Priebe curve.

Drilled shafts: the base in rock is on the socket diameter, as GEC-10
Example 10-4 takes it (p. 10-28), and the rock side resistance's C is an
input, ``rock_side_C`` (GEC-10 Eq. 10-21; default 1.0 since GM-244, 0.65
before, Eq. 10-22 with alpha_E = 1.0). Example 10-4's statement that the
rock below the tip is competent is ``rock_base_conditions_met``. Aggregate
piers: ``priebe_improvement_factor`` reads GEC-13's printed Priebe curve,
Fig. 5-21 curve (4), and since GM-276 nothing else: the formula, its 42.5
deg default and its inputs are gone (owner, OWN-70). The aggregate pier
calc package cites GEC-13's stress reduction method, not "Priebe (1995)",
for its settlement.
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
    """GEC-13 Example 2 (pp. 5-48, 5-49): A/A_sc = (5.7 / 3.0)^2 = 3.61,
    where the printed curve (4) reads 2.55. The example's text prints 2.7,
    the value on its arrows, which are drawn at A/A_sc 3.53 (Fig. 5-27)."""
    r = _call("ground_improvement", "priebe_improvement_factor",
              {"as_ratio": 0.277})
    assert r["settlement_ratio"] == pytest.approx(2.55, abs=0.03)
    assert r["phi_column_deg"] == 37.5 and "curve (4)" in r["note"]
    assert "n0" not in r and "method" not in r
    (c,) = r["choices"]
    assert c["how"] == "module" and "p. 5-36" in c["basis"]
    assert "GEC-13 prints this curve and no formula" in c["basis"]
    assert "defaults_used" not in r
    # GM-276: the value first, then GEC-13's own words on it, with the page,
    # and the design value left to the engineer; in the note and the choice.
    assert r["note"].startswith('Settlement ratio "n" = 2.55 at A/A_c = 3.61')
    upper = ("p. 5-37: 'For preliminary estimates, the Priebe curve may be "
             "used to evaluate the upper bound effectiveness and cost at "
             "various spacings.'")
    for text in (r["note"], c["basis"]):
        assert upper in text and "the engineer's to judge" in text
    assert "upper bound" in c["chosen"]
    assert any("average or lower bound estimate" in a
               for a in c["alternatives"])


@pytest.mark.parametrize("name, value", [
    ("phi_column", 42.5), ("phi_c", 40.0), ("nu_soil", 0.3),
    ("method", "formula"), ("method", "fig_5_21")])
def test_the_formula_and_its_inputs_are_refused(name, value):
    """GM-276: the formula, its 42.5 deg default and its inputs are gone;
    each is refused with what the tool uses now."""
    err = _call("ground_improvement", "priebe_improvement_factor",
                {"as_ratio": 0.277, name: value})["error"]
    assert f"does not take ['{name}']" in err
    assert "Fig. 5-21 curve (4)" in err and "37.5 deg" in err
    assert "Did you mean" not in err


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
