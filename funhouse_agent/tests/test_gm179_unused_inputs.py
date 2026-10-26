"""GM-179: inputs that were accepted and changed nothing are refused with
their reason, or used as a printed method uses them.

Refused: axial E (the static capacity methods), drilled shaft
socket_length, the aggregate piers' column modulus (E_column,
column_modulus; the package's older E_pier), the feasibility screen's
water table, the CPT correlations' i_c_limit and s_g, a continuous
beam's area, an MSE wall reinforcement's yield strength (GM-269: the
rupture check takes the Tal given) and the Priebe tool's phi and Poisson's
ratio (GM-276: GEC-13's printed curve, 'PRIEBE phi = 37.5 deg', is its only
basis). Used: the LRFD MSE wall's gamma_foundation (GEC-11 Eq. 4-22;
mse_wall refuses it) and the feasibility screen's depth_to_top_m
(GEC-13 pp. 5-9, 4-65).
"""

import json

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import ABSENT_MODULES

_PROBES = json.load(open(
    __file__.rsplit("tests", 1)[0] + "tests/method_probes.json",
    encoding="utf-8"))


def _call(agent, method, params):
    r = D.call_agent(agent, method, params)
    return json.loads(r) if isinstance(r, str) else r


def _probe(key):
    return dict(_PROBES[key]["params"])


@pytest.mark.parametrize("agent, method, name, value, why", [
    ("axial_pile", "axial_pile_capacity", "E", 30e6, "make_pile_section"),
    ("axial_pile", "capacity_vs_depth", "E", 30e6, "elastic modulus"),
    ("drilled_shaft", "drilled_shaft_capacity", "socket_length", 3.0,
     "within the rock layers"),
    ("drilled_shaft", "capacity_vs_depth", "socket_length", 3.0,
     "within the rock layers"),
    ("drilled_shaft", "lrfd_capacity", "socket_length", 3.0,
     "within the rock layers"),
    ("ground_improvement", "aggregate_piers", "E_column", 90000.0, "Give n"),
    ("ground_improvement", "aggregate_piers", "column_modulus", 90000.0,
     "Give n"),
    ("ground_improvement", "feasibility", "gwt_depth_m", 2.0,
     "above and below the natural groundwater level"),
    ("ground_improvement", "priebe_improvement_factor", "phi_column", 42.5,
     "printed for 'PRIEBE phi = 37.5 deg' only"),
    ("ground_improvement", "priebe_improvement_factor", "nu_soil", 0.33,
     "with no Poisson's ratio"),
    ("liquepy", "field_correlations", "i_c_limit", 1.8,
     "cpt_liquefaction"),
    ("liquepy", "field_correlations", "s_g", 2.7, "void_ratio"),
    pytest.param("pynite", "continuous_beam", "A", 0.02, "frame_analysis",
                 marks=pytest.mark.skipif(
                     "pynite" in ABSENT_MODULES,
                     reason="PyNiteFEA not installed (INF-53)")),
    ("retaining_walls", "mse_wall", "gamma_foundation", 18.0,
     "mse_lrfd_external_stability"),
    ("retaining_walls", "mse_wall", "reinforcement_Fy", 448.0,
     "Eq. 3-11a (p. 3-22), Fy Ac / b: 0.55 Fy for strips and 0.48 Fy for "
     "bar mats (p. 4-77)"),
])
def test_an_input_that_changed_nothing_is_refused(agent, method, name,
                                                  value, why):
    params = _probe(f"{agent}|{method}")
    assert "error" not in _call(agent, method, dict(params))
    err = _call(agent, method, {**params, name: value}).get("error", "")
    assert "does not take" in err and name in err and why in err, err


@pytest.mark.parametrize("package, name", [
    ("axial_pile_package", "E"),
    ("drilled_shaft_package", "socket_length"),
])
def test_a_package_refuses_what_its_tool_refuses(package, name):
    params = _probe(f"calc_package|{package}")
    err = _call("calc_package", package, {**params, name: 3.0}).get(
        "error", "")
    assert "does not take" in err and name in err, err


def test_make_pile_section_still_takes_the_modulus():
    r = _call("axial_pile", "make_pile_section",
              {"pile_type": "h_pile", "designation": "HP12x74",
               "E": 210e6})
    assert r["E_kPa"] == pytest.approx(210e6)


def test_lrfd_mse_works_the_bearing_resistance_out():
    base = _probe("retaining_walls|mse_lrfd_external_stability")
    before = _call("retaining_walls", "mse_lrfd_external_stability", base)
    assert before["bearing"]["CDR_strength"] is None
    assert "bearing (Strength I)" in before["checks_not_run"]
    # the foundation soil's own phi, never the reinforced fill's
    no_phi = _call("retaining_walls", "mse_lrfd_external_stability",
                   {**base, "gamma_foundation": 18.0})
    assert "give phi_foundation" in no_phi.get("error", "")
    after = _call("retaining_walls", "mse_lrfd_external_stability",
                  {**base, "gamma_foundation": 18.0, "phi_foundation": 30.0})
    b = after["bearing"]
    assert b["qR_strength_max_kPa"] == pytest.approx(
        0.65 * b["qn_strength_max_kPa"], abs=0.02)
    assert b["CDR_strength"] == pytest.approx(
        b["qR_strength_max_kPa"] / b["sigma_v_strength_max_kPa"], abs=0.002)
    assert "bearing (Strength I)" not in after["checks_not_run"]
    assert "phi_bearing" in after["defaults_used"]
    assert "bearing_resistance_strength" not in after["defaults_used"]
    (choice,) = after["choices"]
    assert choice["how"] == "default" and "Table 4-5" in choice["basis"]
    both = _call("retaining_walls", "mse_lrfd_external_stability",
                 {**base, "gamma_foundation": 18.0, "phi_foundation": 30.0,
                  "bearing_resistance_strength": 500.0})
    assert "not both" in both.get("error", "")


def test_feasibility_holds_the_layer_depth_against_gec13():
    deep = _call("ground_improvement", "feasibility",
                 {"soil_type": "soft_clay", "thickness_m": 25.0,
                  "depth_to_top_m": 10.0})
    assert "Aggregate Piers" not in deep["applicable_methods"]
    shallow = _call("ground_improvement", "feasibility",
                    {"soil_type": "soft_clay", "thickness_m": 25.0})
    assert "Aggregate Piers" in shallow["applicable_methods"]
