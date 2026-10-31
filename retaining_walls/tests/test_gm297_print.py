"""GM-297: the MSE wall checked against GEC-11 (FHWA-NHI-10-024) as printed.

Made-up walls throughout. What the print gives:

* Vol I Eq. 4-36 (p. 4-51): Le >= TMAX / (phi F* alpha sigma_v C Rc)
  ">= 3 ft (1 m)"; Eq. 4-37, L = La + Le. The module floored Le at 1.0 m
  (so a level with less had its pullout worked on a length it does not
  have) and never checked the 3 ft.
* Vol I Eq. 3-8 (p. 3-21): a geosynthetic's F* = 2/3 tan phi (0.67 before).
* Vol I Eqs. 3-6, 3-7 (p. 3-21): a steel grid's F* = 20 t/St to 10 t/St;
  a grid given no t/St took the ribbed-strip curve and is now refused.
* Vol I Eqs. 4-12, 4-13 (p. 4-21) and mu (p. 4-22): sliding R = V mu with
  no cohesion term; Step 3(i) names cu for a cohesive foundation. The
  module added 2/3 c_foundation L, which nothing held prints.
* Vol I p. 4-23: emax = L/4 on soil and 3/8 L on rock.
"""

import math

import pytest

from retaining_walls import mse
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.reinforcement import Reinforcement

GEOGRID = Reinforcement(name="made-up geogrid", type="geogrid",
                        Tallowable=40.0)


# ---------------------------------------------------------------------------
# Le, the length beyond the failure surface (Eqs. 4-36, 4-37)
# ---------------------------------------------------------------------------

def test_le_is_l_less_la_with_no_floor():
    """At the top of a 4 m wall with L = 2.8 m the extensible failure
    surface (Eq. 4-38) takes La = 1.97 m, leaving Le = 0.83 m: pullout is
    worked on that, not on a 1.0 m floor."""
    g = MSEWallGeometry(wall_height=4.0, reinforcement_spacing=0.6)
    top = mse.check_internal_stability(g, 19.0, 34.0, GEOGRID)[0]
    L = g.reinforcement_length
    assert top["Le_m"] == pytest.approx(L - top["La_m"], abs=0.006)
    assert top["Le_m"] < 1.0


def test_le_under_3ft_fails_whatever_the_pullout_fos():
    """Eq. 4-36: Le at least 3 ft (1 m). The top level's FOS_pullout is
    well above 1.5, yet with Le 0.83 m it fails."""
    g = MSEWallGeometry(wall_height=4.0, reinforcement_spacing=0.6)
    levels = mse.check_internal_stability(g, 19.0, 34.0, GEOGRID)
    top, second = levels[0], levels[1]
    assert top["FOS_pullout"] > 1.5
    assert top["Le_at_least_3ft"] is False
    assert top["passes"] is False
    assert second["Le_m"] >= mse.LE_MIN_M
    assert second["Le_at_least_3ft"] is True
    assert mse.LE_MIN_M == pytest.approx(0.9144)


def test_le_check_reaches_the_verdict():
    g = MSEWallGeometry(wall_height=4.0, reinforcement_spacing=0.6)
    r = mse.analyze_mse_wall(g, gamma_backfill=19.0, phi_backfill=34.0,
                             reinforcement=GEOGRID, q_allowable=400.0)
    assert "internal" in r.checks_failed
    assert 0.3 in [round(z, 2) for z in r.failed_levels]
    assert "Le under 3 ft" in r.verdict


# ---------------------------------------------------------------------------
# F* (Eqs. 3-6 to 3-8)
# ---------------------------------------------------------------------------

def test_geosynthetic_F_star_is_two_thirds_tan_phi():
    g = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6,
                        reinforcement_length=5.0)
    for lv in mse.check_internal_stability(g, 19.0, 30.0, GEOGRID):
        assert lv["F_star"] == pytest.approx(
            2.0 / 3.0 * math.tan(math.radians(30.0)), abs=0.0006)


def test_grid_without_t_st_is_refused_by_the_wall_analysis():
    g = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6,
                        reinforcement_length=5.0)
    grid = Reinforcement(name="made-up grid", type="metallic_grid",
                         Tallowable=100.0, coverage_ratio=0.3)
    with pytest.raises(ValueError, match="t/St"):
        mse.analyze_mse_wall(g, gamma_backfill=18.0, phi_backfill=34.0,
                             reinforcement=grid)


# ---------------------------------------------------------------------------
# Sliding and the foundation cohesion (Eqs. 4-12, 4-13; p. 4-21 Step 3)
# ---------------------------------------------------------------------------

def _wall(slope=0.0):
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=4.2,
                           backfill_slope=slope)


def test_drained_rule_does_not_count_the_cohesion():
    """Eq. 4-12, R = V mu: a c' of 10 kPa adds nothing (it added
    2/3 x 10 x L before)."""
    with_c = mse.check_external_stability(_wall(), 19.0, 34.0, 18.0, 28.0,
                                          10.0)
    no_c = mse.check_external_stability(_wall(), 19.0, 34.0, 18.0, 28.0,
                                        0.0)
    assert with_c["FOS_sliding"] == pytest.approx(no_c["FOS_sliding"])
    t = mse._external_terms(_wall(), 19.0, 34.0, 28.0, 10.0)
    assert t["sliding_cohesion"] == "drained"
    assert t["delta_b"] == pytest.approx(28.0)
    assert t["resisting"] == pytest.approx(
        t["V"] * math.tan(math.radians(28.0)))


@pytest.mark.parametrize("cu, along", [(40.0, "the foundation soil"),
                                       (120.0, "the reinforced fill")])
def test_undrained_rule_takes_the_weaker_of_foundation_and_fill(cu, along):
    """phi_foundation 0: the smaller of cu L along the foundation and
    V tan phi_r along the reinforced fill (Step 3 (i), (ii))."""
    t = mse._external_terms(_wall(), 19.0, 34.0, 0.0, cu)
    assert t["sliding_cohesion"] == "undrained"
    assert t["delta_b"] is None
    assert t["R_foundation"] == pytest.approx(cu * 4.2)
    assert t["R_fill"] == pytest.approx(
        t["V"] * math.tan(math.radians(34.0)))
    assert t["resisting"] == pytest.approx(min(t["R_foundation"],
                                               t["R_fill"]))
    assert t["sliding_along"] == along


def test_sliding_rule_refusals():
    with pytest.raises(ValueError, match="friction_plus_adhesion"):
        mse.check_external_stability(_wall(), 19.0, 34.0, 18.0, 28.0, 10.0,
                                     sliding_cohesion="friction_plus_adhesion")
    with pytest.raises(ValueError, match="cu > 0"):
        mse.check_external_stability(_wall(), 19.0, 34.0, 18.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="cu > 0"):
        mse.check_external_stability(_wall(), 19.0, 34.0, 18.0, 28.0, 0.0,
                                     sliding_cohesion="undrained")


def test_sliding_choice_gives_the_other_rule():
    r = mse.analyze_mse_wall(_wall(), gamma_backfill=19.0, phi_backfill=34.0,
                             reinforcement=GEOGRID, phi_foundation=28.0,
                             c_foundation=10.0)
    [c] = [c for c in r.choices if "sliding_cohesion" in c["what"]]
    assert c["how"] == "default"
    assert c["chosen"].startswith("drained")
    undrained = mse.analyze_mse_wall(
        _wall(), gamma_backfill=19.0, phi_backfill=34.0,
        reinforcement=GEOGRID, phi_foundation=28.0, c_foundation=10.0,
        sliding_cohesion="undrained")
    assert any(f"FOS_sliding {undrained.FOS_sliding:.3f}" in a
               for a in c["alternatives"])
    # no cohesion, the drained rule by default: nothing to choose
    plain = mse.analyze_mse_wall(_wall(), gamma_backfill=19.0,
                                 phi_backfill=34.0, reinforcement=GEOGRID,
                                 phi_foundation=28.0)
    assert not [c for c in plain.choices if "sliding_cohesion" in c["what"]]


def test_lrfd_level_drained_ignores_cohesion_and_undrained_takes_cu_L():
    with_c = mse.check_external_stability_lrfd(_wall(), 19.0, 34.0, 28.0,
                                               c_foundation=10.0)
    no_c = mse.check_external_stability_lrfd(_wall(), 19.0, 34.0, 28.0)
    assert with_c["sliding"]["CDR_governing"] == \
        no_c["sliding"]["CDR_governing"]
    assert with_c["sliding"]["sliding_cohesion"] == "drained"
    r = mse.check_external_stability_lrfd(_wall(), 19.0, 34.0, 0.0,
                                          c_foundation=40.0)
    u = r["unfactored_kN_m"]
    R = min(40.0 * 4.2, 1.35 * u["V1"] * math.tan(math.radians(34.0)))
    assert r["sliding"]["sliding_cohesion"] == "undrained"
    assert r["sliding"]["CDR_strength_max"] == pytest.approx(
        R / (1.5 * u["F1"]), abs=0.001)
    assert [c for c in r["choices"] if "sliding_cohesion" in c["what"]]


def test_lrfd_sloping_drained_ignores_cohesion():
    with_c = mse.check_external_stability_lrfd(_wall(15.0), 19.0, 34.0,
                                               28.0, c_foundation=10.0)
    no_c = mse.check_external_stability_lrfd(_wall(15.0), 19.0, 34.0, 28.0)
    assert with_c["sliding"]["CDR_governing"] == \
        no_c["sliding"]["CDR_governing"]


def test_ecc_limit_ratio_text_says_three_eighths_on_rock():
    """p. 4-23: emax = 3/8 L on rock; the text said 1/6 before."""
    doc = mse.check_external_stability_lrfd.__doc__
    assert "emax = 3/8 L" in doc and "0.375 on rock" in doc
