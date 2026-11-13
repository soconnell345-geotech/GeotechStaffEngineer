"""GM-319: three things the pile_group tools said and did not do.

Made-up values throughout.
"""

import math

import pytest

from funhouse_agent import dispatch as D


def _call(method, params):
    return D.call_agent("pile_group", method, params)


_FOOTING = {"n_rows": 3, "n_cols": 3, "spacing": 0.9, "pile_diameter": 0.3,
            "pile_length": 12.0, "load_kN": 3000.0}


def test_equivalent_footing_settlement_is_the_printed_sum():
    """GEC-12 Eq. 7-55 over a uniform modulus has a closed form:
    S = (Q/M) [1/B - 1/(B + z)] for a square footing."""
    r = _call("equivalent_footing_settlement",
              {**_FOOTING, "constrained_modulus_kPa": 40000.0,
               "influence_depth_m": 5.0, "num_sublayers": 400})
    assert "error" not in r, r
    B = 2 * 0.9 + 0.3
    assert r["Bg_m"] == pytest.approx(B) and r["Lg_m"] == pytest.approx(B)
    assert r["raft_depth_m"] == pytest.approx(8.0)
    exact = 3000.0 / 40000.0 * (1 / B - 1 / (B + 5.0))
    assert r["settlement_m"] == pytest.approx(exact, rel=1e-4)
    assert r["max_stress_kPa"] == pytest.approx(3000.0 / B ** 2, abs=1e-3)
    assert "not_included" in r


def test_default_depth_is_ten_percent_of_the_bearing_stress():
    r = _call("equivalent_footing_settlement",
              {**_FOOTING, "constrained_modulus_kPa": 40000.0})
    z, B = r["influence_depth_m"], r["Bg_m"]
    assert 3000.0 / (B + z) ** 2 == pytest.approx(
        0.10 * r["max_stress_kPa"], rel=1e-3)
    depth = r["choices"][0]
    assert depth["how"] == "default" and "UFC 3-220-20" in depth["chosen"]


def test_youngs_modulus_by_ufc_eq_6_54():
    Es, nu = 30000.0, 0.3
    M = Es * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    a = _call("equivalent_footing_settlement",
              {**_FOOTING, "youngs_modulus_kPa": Es, "poissons_ratio": nu})
    b = _call("equivalent_footing_settlement",
              {**_FOOTING, "constrained_modulus_kPa": M})
    assert a["settlement_mm"] == pytest.approx(b["settlement_mm"], rel=1e-6)
    assert a["constrained_modulus_kPa"] == pytest.approx(M, abs=0.1)


@pytest.mark.parametrize("extra, word", [
    ({}, "constrained_modulus_kPa"),
    ({"youngs_modulus_kPa": 30000.0}, "poissons_ratio"),
    ({"youngs_modulus_kPa": 30000.0, "poissons_ratio": 0.5}, "below 0.5"),
    ({"youngs_modulus_kPa": 30000.0, "poissons_ratio": 0.3,
      "constrained_modulus_kPa": 40000.0}, "not both"),
])
def test_modulus_has_no_default(extra, word):
    r = _call("equivalent_footing_settlement", {**_FOOTING, **extra})
    assert word in r["error"]


@pytest.mark.parametrize("key, value", [
    ("batter_x", 5), ("batter_y", -3), ("axial_stiffness", 2e5),
    ("lateral_stiffness", 1e4)])
def test_simple_refuses_what_it_does_not_use(key, value):
    r = _call("pile_group_simple", {"Vz": 1000, "piles": [
        {"x": -1, "y": 0}, {"x": 1, "y": 0, key: value}]})
    assert "piles[1]" in r["error"] and key in r["error"]
    assert "pile_group_6dof" in r["error"]


def test_simple_reports_no_displacement_as_none():
    r = _call("pile_group_simple", {"Vz": 1000, "piles": [
        {"x": -1, "y": 0, "batter_x": 0}, {"x": 1, "y": 0}]})
    assert r["cap_displacements"] is None
    assert "not calculated" in r["notes"][0]


def test_6dof_battered_group_matches_statics():
    r = _call("pile_group_6dof", {"Vx": 100, "Vz": 1000, "My": 200, "piles": [
        {"x": -1, "y": 0, "axial_stiffness": 2e5, "label": "A"},
        {"x": 1, "y": 0, "axial_stiffness": 2e5, "label": "B"},
        {"x": 0, "y": 0, "batter_x": 15, "axial_stiffness": 2e5,
         "label": "C"}]})
    assert "error" not in r, r
    s, c = math.sin(math.radians(15)), math.cos(math.radians(15))
    Pc = 100 / s
    want = {"A": (1000 - Pc * c - 200) / 2, "B": (1000 - Pc * c + 200) / 2,
            "C": Pc}
    for pf in r["pile_forces"]:
        assert pf["axial_kN"] == pytest.approx(want[pf["label"]], abs=0.06)
    assert "pinned" in r["notes"][0]


def test_descriptions_say_what_the_code_does():
    from funhouse_agent.adapters.pile_group import METHOD_INFO
    assert "pinned" in METHOD_INFO["pile_group_6dof"]["brief"]
    assert "refused" in METHOD_INFO["pile_group_simple"]["parameters"][
        "piles"]["description"]
    assert "equivalent_footing_settlement" in METHOD_INFO


# --- the review's findings (GM-319 review) -------------------------------

_ROW = {"n_rows": 1, "n_cols": 3, "spacing": 1.4, "Vz": 960}


def test_simple_tool_refuses_a_moment_about_its_one_row():
    r = _call("pile_group_simple", {**_ROW, "Mx": 240})
    assert "Mx = 240 kN-m cannot be carried" in r["error"]
    assert "every pile is on y = 0" in r["error"]
    r = _call("pile_group_simple", {**_ROW, "n_rows": 3, "n_cols": 1,
                                    "My": 240})
    assert "My = 240 kN-m cannot be carried" in r["error"]


def test_simple_tool_one_row_without_the_moment_runs():
    r = _call("pile_group_simple", {**_ROW, "Mx": 0})
    assert "error" not in r, r
    assert [p["axial_kN"] for p in r["pile_forces"]] == pytest.approx(
        [320.0] * 3)


def test_6dof_tool_one_row_message_points_nowhere_it_cannot_go():
    r = _call("pile_group_6dof", {**_ROW, "Mx": 240,
                                  "axial_stiffness": 2e5,
                                  "lateral_stiffness": 6e3})
    assert "every pile is on y = 0" in r["error"]
    assert "pile_group_simple" not in r["error"]


def test_6dof_tool_refuses_batter_with_no_vertical_component():
    r = _call("pile_group_6dof", {"Vz": 600, "piles": [
        {"x": 0, "y": 0, "batter_x": 50, "batter_y": 50,
         "axial_stiffness": 2e5},
        {"x": 1, "y": 0, "axial_stiffness": 2e5},
        {"x": 0, "y": 1, "axial_stiffness": 2e5}]})
    assert "at or above 1" in r["error"]


_M40 = {**_FOOTING, "constrained_modulus_kPa": 40000.0}


@pytest.mark.parametrize("value", [2.7, 0, -3, True, "4"])
def test_num_sublayers_must_be_a_whole_number(value):
    r = _call("equivalent_footing_settlement", {**_M40,
                                                "num_sublayers": value})
    assert "num_sublayers must be a whole number" in r["error"]


def test_num_sublayers_whole_float_runs_as_the_int():
    a = _call("equivalent_footing_settlement", {**_M40, "num_sublayers": 4.0})
    b = _call("equivalent_footing_settlement", {**_M40, "num_sublayers": 4})
    assert a["settlement_mm"] == b["settlement_mm"]


def test_spacing_below_the_diameter_is_refused():
    r = _call("equivalent_footing_settlement", {**_M40, "spacing": 0.25})
    assert "less than the pile diameter" in r["error"]
    # touching piles (s = b) run
    assert "error" not in _call("equivalent_footing_settlement",
                                {**_M40, "spacing": 0.3})


@pytest.mark.parametrize("alias, value", [("soil_modulus_kPa", 40000.0),
                                          ("D", 12.0)])
def test_dropped_aliases_are_not_read(alias, value):
    params = {**_FOOTING, "constrained_modulus_kPa": 40000.0}
    if alias == "D":
        params.pop("pile_length")
    else:
        params.pop("constrained_modulus_kPa")
    params[alias] = value
    r = _call("equivalent_footing_settlement", params)
    assert "error" in r and alias in r["error"]


def test_poissons_ratio_above_coarse_grained_range_warns():
    """The review's case: Es 20,000 kPa at nu 0.49 makes M 17 Es and 2.8 mm,
    against about 36 mm at nu 0.3. The result warns that the equation is
    for coarse-grained soil and states M/Es."""
    hi = _call("equivalent_footing_settlement",
               {**_FOOTING, "youngs_modulus_kPa": 20000.0,
                "poissons_ratio": 0.49})
    assert "error" not in hi, hi
    assert hi["settlement_mm"] == pytest.approx(2.84, abs=0.01)
    (w,) = hi["warnings"]
    assert "GEC-5 Table 8-10" in w and "coarse-grained" in w
    assert "consolidation_settlement" in w
    m = next(c for c in hi["choices"] if c["what"] == "constrained modulus M")
    assert "= 17.11 Es" in m["chosen"]

    lo = _call("equivalent_footing_settlement",
               {**_FOOTING, "youngs_modulus_kPa": 20000.0,
                "poissons_ratio": 0.3})
    assert lo["settlement_mm"] == pytest.approx(36.09, abs=0.01)
    assert "warnings" not in lo
    m = next(c for c in lo["choices"] if c["what"] == "constrained modulus M")
    assert "= 1.35 Es" in m["chosen"]
    # at the top of the printed range: no warning
    assert "warnings" not in _call(
        "equivalent_footing_settlement",
        {**_FOOTING, "youngs_modulus_kPa": 20000.0, "poissons_ratio": 0.40})


def test_settlement_tool_says_its_soil():
    from funhouse_agent.adapters.pile_group import METHOD_INFO
    info = METHOD_INFO["equivalent_footing_settlement"]
    assert "coarse-grained" in info["brief"]
    assert "consolidation_settlement" in info["brief"]
    for p in ("constrained_modulus_kPa", "youngs_modulus_kPa",
              "poissons_ratio"):
        assert "coarse-grained" in info["parameters"][p]["description"]
    # GEC-12 Sec. 7.3.5.3 is the cohesive-soil method, not this equation's
    assert "7.3.5.3" not in info["brief"].split("cohesive soil")[0]
