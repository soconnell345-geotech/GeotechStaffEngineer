"""The MSE methods take a sloping backfill through dispatch (BACKLOG GM-112).

GEC-11 Example E3 (FHWA-NHI-10-025, a 2H:1V slope that continues) through
``mse_lrfd_external_stability``; refusals said with their reasons. The
module-level checks are in retaining_walls/tests/test_mse_sloping_backfill.py
and the published examples in
validation_examples/test_published_v056_v057_gec11_slope.py.
"""

import math

import pytest

from funhouse_agent import dispatch as D

FT, PCF, KSF = 0.3048, 0.157087, 47.88
BETA = round(math.degrees(math.atan(0.5)), 4)        # 26.5651

E3 = {"wall_height": 30 * FT, "reinforcement_length": 24 * FT,
      "gamma_backfill": 125 * PCF, "phi_backfill": 34,
      "phi_retained": 30, "phi_foundation": 30, "backfill_slope": BETA,
      "bearing_resistance_strength": 10.5 * KSF,
      "bearing_resistance_service": 7.5 * KSF}

MSE = {"wall_height": 6, "gamma_backfill": 19, "phi_backfill": 34,
       "reinforcement_name": "ribbed_steel_strip_75x4"}


def _call(method, params):
    return D.call_agent("retaining_walls", method, params)


def test_e3_through_dispatch():
    """Tables E3-6.1 to E3-6.3: sliding 1.07, eL 3.81 ft (critical),
    sigma_v 10.19 ksf (Str I max), Service I 7.16 ksf."""
    out = _call("mse_lrfd_external_stability", E3)
    assert "error" not in out, out
    assert out["sliding"]["CDR_critical"] == pytest.approx(1.07, abs=0.005)
    assert out["eccentricity"]["eL_critical_m"] / FT == pytest.approx(
        3.81, abs=0.01)
    assert out["bearing"]["sigma_v_strength_max_kPa"] / KSF == pytest.approx(
        10.19, abs=0.01)
    assert out["bearing"]["sigma_v_service_kPa"] / KSF == pytest.approx(
        7.16, abs=0.01)
    assert out["backslope"]["Kab"] == pytest.approx(0.537, abs=0.001)


def test_mse_wall_reports_the_slope_and_pullout_depth():
    out = _call("mse_wall", {**MSE, "backfill_slope": 20})
    assert "error" not in out, out
    b = out["backslope"]
    assert b["backfill_slope_deg"] == 20
    for key in ("Kab", "h_m", "FT_kN_per_m", "V2_kN_per_m", "S_eq_m",
                "sigma_2_kPa", "H1_m", "basis"):
        assert key in b, key
    assert all("Zp_m" in lv for lv in out["internal_results"])
    level = _call("mse_wall", MSE)
    assert out["FOS_sliding"] < level["FOS_sliding"]


def test_mse_wall_takes_the_retained_fill():
    """With a slope Kab follows the fill behind the zone (E3: 30 deg behind
    a 34 deg reinforced fill); mse_wall passes phi_retained through."""
    same = _call("mse_wall", {**MSE, "backfill_slope": 20})
    weaker = _call("mse_wall", {**MSE, "backfill_slope": 20,
                                "phi_retained": 30, "gamma_retained": 18})
    assert "error" not in weaker, weaker
    assert weaker["backslope"]["Kab"] > same["backslope"]["Kab"]
    assert weaker["FOS_sliding"] < same["FOS_sliding"]
    level = _call("mse_wall", MSE)
    given = _call("mse_wall", {**MSE, "phi_retained": 34,
                               "gamma_retained": 19})
    # left out, the two are listed as taken from the backfill (GM-165)
    used = level.pop("defaults_used")
    assert (used["phi_retained"], used["gamma_retained"]) == (34, 19)
    given.pop("defaults_used")
    assert given == level


def test_broken_backslope_with_traffic_runs():
    out = _call("mse_lrfd_external_stability",
                {"wall_height": 6, "gamma_backfill": 19, "phi_backfill": 34,
                 "backfill_slope": BETA, "broken_backslope_height": 1.5,
                 "surcharge": 12})
    assert "error" not in out, out
    assert out["backslope"]["broken_backslope_height_m"] == 1.5
    assert out["unfactored_kN_m"]["F2"] > 0


@pytest.mark.parametrize("method, extra", [
    ("mse_wall", {}), ("mse_lrfd_external_stability", {})])
def test_surcharge_on_a_slope_that_continues_refused(method, extra):
    params = {**MSE, "backfill_slope": 20, "surcharge": 10}
    if method != "mse_wall":
        params.pop("reinforcement_name")
    err = _call(method, params)["error"]
    assert "Eq. 4-16" in err and "broken_backslope_height" in err


def test_slope_steeper_than_phi_refused():
    err = _call("mse_wall", {**MSE, "backfill_slope": 35})["error"]
    assert "steeper than the friction angle" in err
    at_phi = _call("mse_wall", {**MSE, "backfill_slope": 34})
    assert "error" not in at_phi, at_phi            # Eq. 4-3: Gamma = 1


def test_traffic_vertical_resists_is_offered_and_named():
    """Example E1 counts the traffic thrust's vertical component; the
    default follows GEC-11 Vol I p. 4-22. The result says which."""
    params = {"wall_height": 6, "gamma_backfill": 19, "phi_backfill": 34,
              "backfill_slope": BETA, "broken_backslope_height": 3.0,
              "surcharge": 12}
    off = _call("mse_lrfd_external_stability", params)
    on = _call("mse_lrfd_external_stability",
               {**params, "traffic_vertical_resists": True})
    assert on["sliding"]["CDR_critical"] > off["sliding"]["CDR_critical"]
    assert "p. 4-22" in off["backslope"]["traffic_basis"]
    assert on["backslope"]["traffic_vertical_resists"] is True
    wall = _call("mse_wall", {**MSE, "backfill_slope": BETA,
                              "broken_backslope_height": 3.0,
                              "surcharge": 12,
                              "traffic_vertical_resists": True})
    assert wall["backslope"]["traffic_vertical_resists"] is True


def test_methods_describe_the_slope():
    for method in ("mse_wall", "mse_lrfd_external_stability"):
        info = D.describe_method("retaining_walls", method)
        text = str(info)
        assert "backfill_slope" in text and "broken_backslope_height" in text
