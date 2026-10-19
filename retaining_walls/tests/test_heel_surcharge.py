"""GM-157: a surcharge over the heel of a cantilever wall.

The references read (printed pages):

* FHWA NHI-06-089 Vol II, Sec. 10.5.5 (p. 10-43): "the live load
  surcharge is not considered as a stabilizing force over the heel of the
  wall when sliding resistance is being checked"; Sec. 10.5.3 (p. 10-41):
  for CIP semi-gravity walls the surcharge loads are "conservatively
  neglected within the width of the base slab since they contribute to
  overturning and sliding resistance".
* UFC 3-220-20 App. B-1.2 (p. 597): "For overturning and sliding checks,
  it is assumed that the traffic pressure does not act over the wall
  heel. For the check of bearing capacity, the traffic pressure is
  included"; B-1.6 (p. 602).
* GEC-11 Vol I p. 4-22: loads that increase sliding resistance are
  included "only if those loads are permanent"; p. 4-26: "eB =
  eccentricity for bearing calculation (not equal to eccentricity check
  e)".

So by default the surcharge's thrust drives every check, its weight over
the heel resists nothing, and it loads the base for bearing. These tests
work a simple wall by hand under each reading.
"""

import math

import pytest

from retaining_walls.cantilever import (
    analyze_cantilever_wall, check_bearing, check_overturning, check_sliding,
)
from retaining_walls.geometry import CantileverWallGeometry

# A simple wall: H = 4.0 m, B = 2.5 m, toe 0.5 m, stem 0.3 m (no taper),
# base 0.5 m; level backfill, Rankine; gamma 18, phi 30 (Ka = 1/3);
# concrete 24; q = 12 kPa.
GAMMA, PHI, Q = 18.0, 30.0, 12.0


def _wall(**kw):
    return CantileverWallGeometry(
        wall_height=4.0, base_width=2.5, toe_length=0.5,
        stem_thickness_top=0.3, stem_thickness_base=0.3,
        base_thickness=0.5, **kw)


# The hand check
KA = 1.0 / 3.0
PA_EARTH = 0.5 * KA * GAMMA * 4.0 ** 2        # 48.0 kN/m at H/3
PA_Q = KA * Q * 4.0                           # 16.0 kN/m at H/2
PH = PA_EARTH + PA_Q                          # 64.0 kN/m
M_OT = PA_EARTH * 4.0 / 3.0 + PA_Q * 2.0      # 96.0 kN.m/m
HEEL = 2.5 - 0.5 - 0.3                        # 1.7 m
W = [24.0 * 2.5 * 0.5,                        # base slab 30.0 at 1.25
     24.0 * 0.3 * 3.5,                        # stem 25.2 at 0.65
     GAMMA * HEEL * 3.5]                      # soil on heel 107.1 at 1.65
X = [1.25, 0.65, 0.5 + 0.3 + HEEL / 2.0]
V_WALL = sum(W)                               # 162.3
M_WALL = sum(w * x for w, x in zip(W, X))     # 230.595
W_Q = Q * HEEL                                # 20.4 at 1.65
M_Q = W_Q * X[2]                              # 33.66
TAN_DB = math.tan(math.radians(2.0 / 3.0 * PHI))


def test_the_hand_check_numbers():
    assert PH == pytest.approx(64.0)
    assert M_OT == pytest.approx(96.0)
    assert V_WALL == pytest.approx(162.3)
    assert M_WALL == pytest.approx(230.595)
    assert W_Q == pytest.approx(20.4)


def test_live_load_default_does_not_resist():
    r = analyze_cantilever_wall(_wall(surcharge=Q), GAMMA, PHI)
    # sliding: V without the surcharge, Ph with its thrust
    assert r.sliding_terms["V"] == pytest.approx(V_WALL)
    assert r.sliding_terms["V_left_out"] == pytest.approx(W_Q)
    assert r.FOS_sliding == pytest.approx(V_WALL * TAN_DB / PH, abs=5e-4)
    assert r.FOS_sliding == pytest.approx(0.923, abs=5e-4)
    # overturning: 230.595 / 96 = 2.402
    assert r.FOS_overturning == pytest.approx(M_WALL / M_OT, abs=5e-4)
    assert r.FOS_overturning == pytest.approx(2.402, abs=5e-4)
    # the eccentricity check on the same loads: x_R = 0.8293, e = 0.4207
    # > B/6 = 0.4167, so the resultant is outside the middle third
    e = 2.5 / 2.0 - (M_WALL - M_OT) / V_WALL
    assert r.eccentricity == pytest.approx(e, abs=5e-4)
    assert e == pytest.approx(0.4207, abs=1e-4)
    assert r.in_middle_third is False
    assert "eccentricity" in r.checks_failed
    # bearing counts the surcharge: V = 182.7, e_b = 0.3291
    V_b = V_WALL + W_Q
    e_b = 1.25 - (M_WALL + M_Q - M_OT) / V_b
    assert r.bearing_terms["V"] == pytest.approx(V_b)
    assert r.eccentricity_bearing == pytest.approx(e_b, abs=5e-4)
    assert r.q_toe == pytest.approx(V_b / 2.5 * (1 + 6 * e_b / 2.5),
                                    abs=0.05)
    assert r.q_heel == pytest.approx(V_b / 2.5 * (1 - 6 * e_b / 2.5),
                                     abs=0.05)
    assert r.to_dict()["eccentricity_bearing_m"] == round(e_b, 3)


def test_live_load_resisting_is_the_other_reading():
    r = analyze_cantilever_wall(_wall(surcharge=Q), GAMMA, PHI,
                                live_load_resists=True)
    V = V_WALL + W_Q                              # 182.7
    assert r.FOS_sliding == pytest.approx(V * TAN_DB / PH, abs=5e-4)
    assert r.FOS_sliding == pytest.approx(1.039, abs=5e-4)
    assert r.FOS_overturning == pytest.approx((M_WALL + M_Q) / M_OT,
                                              abs=5e-4)
    assert r.FOS_overturning == pytest.approx(2.753, abs=5e-4)
    assert r.in_middle_third is True
    # the same loads for bearing: one eccentricity
    assert r.eccentricity_bearing is None
    assert "eccentricity_bearing_m" not in r.to_dict()


def test_bearing_without_the_surcharge():
    r = analyze_cantilever_wall(_wall(surcharge=Q), GAMMA, PHI,
                                surcharge_in_bearing=False)
    b = r.bearing_terms
    assert b["V"] == pytest.approx(V_WALL)
    # e = 0.4207 > B/6: the triangle under the toe, width 3 x_R
    x_R = (M_WALL - M_OT) / V_WALL
    assert b["distribution"] == "triangular_toe"
    assert r.q_toe == pytest.approx(2 * V_WALL / (3 * x_R), abs=0.05)
    # sliding and overturning as by default
    assert r.FOS_sliding == pytest.approx(0.923, abs=5e-4)


def test_no_surcharge_is_unchanged_and_says_nothing():
    a = analyze_cantilever_wall(_wall(), GAMMA, PHI)
    b = analyze_cantilever_wall(_wall(), GAMMA, PHI, live_load_resists=True,
                                dead_load_resists=True,
                                surcharge_in_bearing=False)
    assert a.to_dict() == b.to_dict()
    # only the base friction (GM-173): nothing about a surcharge
    assert [c["what"] for c in a.choices] == [
        "base friction delta_b (sliding)"]


def test_dead_load_surcharge_by_its_own_choice():
    # the same 12 kPa as a dead load: its thrust drives every check, and
    # over the heel it is neglected by default (NHI-06-089 p. 10-41)
    d = analyze_cantilever_wall(_wall(dead_load_surcharge=Q), GAMMA, PHI)
    assert d.earth_pressure["Pa"] == pytest.approx(PH)
    assert d.FOS_sliding == pytest.approx(0.923, abs=5e-4)
    assert d.FOS_overturning == pytest.approx(2.402, abs=5e-4)
    # counted as a permanent load (GEC-11 p. 4-22)
    d = analyze_cantilever_wall(_wall(dead_load_surcharge=Q), GAMMA, PHI,
                                dead_load_resists=True)
    assert d.FOS_sliding == pytest.approx(1.039, abs=5e-4)
    assert d.FOS_overturning == pytest.approx(2.753, abs=5e-4)
    # live_load_resists has no say over a dead load
    d2 = analyze_cantilever_wall(_wall(dead_load_surcharge=Q), GAMMA, PHI,
                                 live_load_resists=True)
    assert d2.FOS_sliding == pytest.approx(0.923, abs=5e-4)


def test_live_and_dead_together():
    r = analyze_cantilever_wall(
        _wall(surcharge=Q, dead_load_surcharge=6.0), GAMMA, PHI,
        dead_load_resists=True)
    ph = PH + KA * 6.0 * 4.0
    v = V_WALL + 6.0 * HEEL
    assert r.earth_pressure["Pa"] == pytest.approx(ph)
    assert r.FOS_sliding == pytest.approx(v * TAN_DB / ph, abs=5e-4)
    assert r.bearing_terms["V"] == pytest.approx(V_WALL + W_Q + 6.0 * HEEL)
    whats = [c["what"] for c in r.choices]
    assert len(whats) == 4
    assert whats[0] == "base friction delta_b (sliding)"      # GM-173
    assert whats[1].startswith("live-load surcharge over the heel")
    assert whats[2].startswith("dead-load surcharge over the heel")
    assert whats[3] == "surcharge over the heel, in the bearing pressures"


def test_the_live_share_of_an_inclined_thrust_does_not_resist():
    # sloped backfill: the thrust is inclined at beta, and the live load's
    # share of its vertical component, Ka q H sin(beta), is a live load
    g = _wall(surcharge=Q, backfill_slope=10.0)
    r = analyze_cantilever_wall(g, GAMMA, PHI)
    ep = r.earth_pressure
    pv_live = ep["Ka"] * Q * ep["H_active"] * math.sin(math.radians(10.0))
    assert ep["Pv_live"] == pytest.approx(pv_live)
    s = r.sliding_terms
    assert s["Pv_counted"] == pytest.approx(ep["Pv"] - pv_live)
    assert s["V"] == pytest.approx(s["V_weights"] + ep["Pv"] - pv_live)
    assert s["V_left_out"] == pytest.approx(W_Q + pv_live)
    # bearing counts the whole thrust's vertical component
    assert r.bearing_terms["V"] == pytest.approx(
        s["V_weights"] + W_Q + ep["Pv"])
    counted = analyze_cantilever_wall(g, GAMMA, PHI, live_load_resists=True)
    assert counted.sliding_terms["V"] == pytest.approx(
        s["V_weights"] + W_Q + ep["Pv"])
    # the calc steps say so on the thrust components
    from retaining_walls.calc_steps import get_calc_steps
    analysis = {"geom": g, "gamma_backfill": GAMMA, "phi_backfill": PHI}
    note = (f"The live load's share, {pv_live:.1f} kN/m, is left out of "
            "sliding, overturning and the eccentricity check.")
    assert note in repr(get_calc_steps(r, analysis))
    assert "The live load's share" not in repr(get_calc_steps(
        counted, {**analysis, "live_load_resists": True}))


def test_the_public_checks_take_the_same_choices():
    g = _wall(surcharge=Q)
    assert check_sliding(g, GAMMA, PHI)["FOS_sliding"] \
        == pytest.approx(0.923, abs=5e-4)
    assert check_sliding(g, GAMMA, PHI, live_load_resists=True)[
        "FOS_sliding"] == pytest.approx(1.039, abs=5e-4)
    o = check_overturning(g, GAMMA, PHI)
    assert o["FOS_overturning"] == pytest.approx(2.402, abs=5e-4)
    assert o["in_middle_third"] is False
    assert check_overturning(g, GAMMA, PHI, live_load_resists=True)[
        "FOS_overturning"] == pytest.approx(2.753, abs=5e-4)
    b = check_bearing(g, GAMMA, PHI)
    assert b["in_middle_third"] is True        # e_b = 0.329 <= 0.417
    assert check_bearing(g, GAMMA, PHI, surcharge_in_bearing=False)[
        "in_middle_third"] is False


def test_the_f55_wall_before_and_after():
    # wave w12 F55: 1.17 / 3.06 with the surcharge over the heel resisting
    # (the module before GM-157), 1.07 / 2.78 without; bearing unchanged
    g = CantileverWallGeometry(
        wall_height=4.5, base_width=2.8, toe_length=0.7,
        stem_thickness_top=0.35, stem_thickness_base=0.35,
        base_thickness=0.5, surcharge=10.0)
    new = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=32.0)
    old = analyze_cantilever_wall(g, 19.0, 32.0, phi_foundation=32.0,
                                  live_load_resists=True)
    assert (new.FOS_sliding, new.FOS_overturning) == (
        pytest.approx(1.07, abs=0.005), pytest.approx(2.78, abs=0.005))
    assert (old.FOS_sliding, old.FOS_overturning) == (
        pytest.approx(1.17, abs=0.005), pytest.approx(3.06, abs=0.005))
    assert (new.q_toe, new.q_heel) == (old.q_toe, old.q_heel) \
        == (pytest.approx(122.4, abs=0.1), pytest.approx(33.1, abs=0.1))
    assert new.eccentricity == pytest.approx(0.338, abs=0.001)
    assert old.eccentricity == pytest.approx(0.268, abs=0.001)


def test_the_choices_say_what_was_used_and_why():
    r = analyze_cantilever_wall(_wall(surcharge=Q), GAMMA, PHI)
    friction, live, bearing = r.choices
    assert friction["what"] == "base friction delta_b (sliding)"
    assert live["how"] == "default"
    assert "not a resisting force" in live["chosen"]
    assert "NHI-06-089" in live["basis"] and "10-43" in live["basis"]
    assert "UFC 3-220-20" in live["basis"]
    assert live["alternatives"][0].startswith("counted as a resisting force")
    assert bearing["how"] == "default"
    assert bearing["chosen"].startswith("counted")
    assert "B-1.6" in bearing["basis"]
    given = analyze_cantilever_wall(_wall(surcharge=Q), GAMMA, PHI,
                                    live_load_resists=True)
    assert given.choices[1]["how"] == "given"
    assert "Choices:" in r.summary()
    assert r.to_dict()["choices"] == r.choices


def test_the_calc_steps_print_each_checks_loads():
    from retaining_walls.calc_steps import get_calc_steps
    g = _wall(surcharge=Q)
    r = analyze_cantilever_wall(g, GAMMA, PHI)
    text = repr(get_calc_steps(r, {"geom": g, "gamma_backfill": GAMMA,
                                   "phi_backfill": PHI}))
    assert "'TOTAL, sliding / overturning / eccentricity', '162.3'" in text
    assert "'TOTAL, bearing', '182.7'" in text
    assert "'Live-load surcharge on heel', '20.4'" in text
    assert "'bearing only'" in text
    assert "V leaves out 20.4 kN/m of surcharge over the heel" in text
    assert "R = 162.3 × tan(20.00°)" in text
    assert "x_R = (230.6 - 96.0) / 162.3" in text
    assert "Resultant for Bearing" in text
