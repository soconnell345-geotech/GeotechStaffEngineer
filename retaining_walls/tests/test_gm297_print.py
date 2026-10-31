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

From the review:

* Vol I p. 4-21 Step 3(iii), p. 4-22 and Eq. 3-9; Vol II E1 p. E1-7: for
  sheet reinforcement sliding is also checked along the lowest layer, on
  tan rho, tested or 2/3 tan phi_r.
* The undrained rule, min(cu L, V tan phi_r), cites UFC 3-220-20 p. 223
  beside GEC-6 Eq. 5-39; GEC-11 p. 4-19 defers to AASHTO (2007) Art.
  10.6.3.4, not held. EN 1997-1 Eq. 6.5, R <= 0.4 V where water or air
  can reach the base, is offered (interface_gap_possible).
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


# ---------------------------------------------------------------------------
# Review: sliding along the lowest sheet (p. 4-21 Step 3(iii); E1 p. E1-7)
# ---------------------------------------------------------------------------

STRIP = Reinforcement(name="made-up strip", type="metallic_strip",
                      Tallowable=30.0, width=0.05,
                      horizontal_spacing=0.75)


def _sheet(phi_f=28.0, c=0.0, slope=0.0, reinforcement=GEOGRID, **kw):
    """The 6 m wall, L 4.2 m, levels every 0.6 m, the lowest at z = 5.7 m."""
    g = MSEWallGeometry(wall_height=6.0, reinforcement_length=4.2,
                        reinforcement_spacing=0.6, backfill_slope=slope)
    return mse.analyze_mse_wall(g, 19.0, 34.0, reinforcement,
                                phi_foundation=phi_f, c_foundation=c, **kw)


def test_sheet_sliding_worked_by_hand():
    """The block above the lowest sheet, 5.7 m high: V = 19 x 5.7 x 4.2 =
    454.86 kN/m on tan rho = 2/3 tan 34 = 0.4497 against Pa = 0.5 Ka 19
    5.7^2 = 87.26 kN/m: 2.344, under the base's 2.633 (phi_f 28)."""
    r = _sheet()
    s = r.external_terms["sheet"]
    assert s["depth"] == pytest.approx(5.7)
    assert s["V"] == pytest.approx(19.0 * 5.7 * 4.2)
    assert s["tan_rho"] == pytest.approx(2 / 3 * math.tan(math.radians(34)))
    Ka = math.tan(math.radians(45 - 34 / 2)) ** 2
    assert s["Pa"] == pytest.approx(0.5 * Ka * 19.0 * 5.7 ** 2)
    assert r.FOS_sliding == pytest.approx(2.344, abs=0.0005)
    assert r.sliding_surfaces == {
        "governing": "lowest sheet", "FOS_base": 2.633,
        "FOS_lowest_sheet": 2.344, "lowest_sheet_depth_m": 5.7,
        "tan_rho": 0.4497}
    d = r.to_dict()
    assert d["sliding_surfaces"]["governing"] == "lowest sheet"


@pytest.mark.parametrize("kw, base, sheet, governs", [
    ({"phi_f": 34.0}, 3.340, 2.344, "lowest sheet"),
    ({"slope": 15.0}, 2.066, 1.812, "lowest sheet"),
    ({"phi_f": 0.0, "c": 40.0}, 1.738, 2.344, "base"),
])
def test_the_lesser_surface_governs(kw, base, sheet, governs):
    r = _sheet(**kw)
    s = r.sliding_surfaces
    assert s["FOS_base"] == pytest.approx(base, abs=0.0005)
    assert s["FOS_lowest_sheet"] == pytest.approx(sheet, abs=0.0005)
    assert s["governing"] == governs
    assert r.FOS_sliding == pytest.approx(min(base, sheet), abs=0.0005)


def test_lrfd_sheet_sliding_critical_combination():
    """The same wall on the LRFD path: the base's governing CDR 1.755,
    the sheet's 1.563, both at EV 1.0, EH 1.5."""
    r = _sheet(lrfd_external=True).external_lrfd["sliding"]
    assert r["surface"] == "lowest sheet"
    assert r["CDR_base_governing"] == pytest.approx(1.755, abs=0.0005)
    assert r["CDR_lowest_sheet_governing"] == pytest.approx(1.563,
                                                            abs=0.0005)
    assert r["CDR_governing"] == r["CDR_critical"] == pytest.approx(
        1.563, abs=0.0005)
    g = MSEWallGeometry(wall_height=6.0, reinforcement_length=4.2)
    assert "surface" not in mse.check_external_stability_lrfd(
        g, 19.0, 34.0, 28.0)["sliding"]


def test_tested_rho_and_the_choice():
    r = _sheet()
    [c] = [c for c in r.choices if "rho_tested" in c["what"]]
    assert c["how"] == "default" and "2/3 tan 34" in c["chosen"]
    assert "sheet governs" in c["chosen"] and "Step 3(iii)" in c["basis"]
    assert "E1-7" in c["basis"]
    tested = _sheet(rho_tested=30.0)
    assert tested.external_terms["sheet"]["tan_rho"] == pytest.approx(
        math.tan(math.radians(30.0)))
    [c] = [c for c in tested.choices if "rho_tested" in c["what"]]
    assert c["how"] == "given" and "2/3 tan phi'_r" in c["alternatives"][0]
    with pytest.raises(ValueError, match="rho_tested"):
        _sheet(rho_tested=95.0)


def test_steel_at_the_lowest_level_slides_on_the_base_only():
    r = _sheet(reinforcement=STRIP)
    assert r.external_terms["sheet"] is None and r.sliding_surfaces is None
    assert not [c for c in r.choices if "rho_tested" in c["what"]]
    with pytest.raises(ValueError, match="Leave rho_tested out"):
        _sheet(reinforcement=STRIP, rho_tested=25.0)
    with pytest.raises(ValueError, match="lowest_sheet_depth"):
        mse.check_external_stability(
            MSEWallGeometry(wall_height=6.0), 19.0, 34.0, 18.0, 28.0,
            lowest_sheet_depth=6.0)


def test_verdict_names_the_sheet_when_it_fails():
    """phi_r 25: the sheet's 2/3 tan 25 L / (Ka z / 2) is 1.13, under
    1.5, while the base passes."""
    g = MSEWallGeometry(wall_height=6.0, reinforcement_length=4.2,
                        reinforcement_spacing=0.6)
    r = mse.analyze_mse_wall(g, 19.0, 25.0, GEOGRID, phi_foundation=28.0)
    assert r.sliding_surfaces["FOS_base"] > 1.5 > r.FOS_sliding
    assert "along the lowest sheet, z = 5.7 m" in r.verdict


def test_calc_steps_print_the_sheet():
    from retaining_walls.calc_steps import get_calc_steps
    r = _sheet()
    analysis = {"wall_type": "mse", "geom": MSEWallGeometry(
        wall_height=6.0, reinforcement_length=4.2, reinforcement_spacing=0.6),
        "gamma_backfill": 19.0, "phi_backfill": 34.0,
        "phi_foundation": 28.0, "reinforcement": GEOGRID}
    ext = next(s for s in get_calc_steps(r, analysis)
               if s.title == "External Stability").items
    titles = [getattr(i, "title", "") for i in ext]
    sheet = ext[titles.index("Sliding Along the Lowest Sheet")]
    assert "2/3 × tan 34.0°" in sheet.substitution
    assert sheet.result_value == "2.344"
    fos = ext[titles.index("Factor of Safety Against Sliding")]
    assert "the lowest sheet governs" in fos.substitution
    assert fos.result_value == "2.344"


# ---------------------------------------------------------------------------
# Review: EN 1997-1's limit on the undrained resistance (Eq. 6.5)
# ---------------------------------------------------------------------------

def test_undrained_gap_holds_the_foundation_to_0_4_V():
    """cu 120 on a steel-reinforced wall: cu L = 504 > V tan phi_r, so the
    fill governs at 3.340; with interface_gap_possible the foundation's
    resistance is 0.4 V = 191.52 kN/m against Pa 96.69: 1.981. LRFD, the
    critical combination: 191.52 / (1.5 x 96.69) = 1.3205, 2.227
    without."""
    plain = _sheet(phi_f=0.0, c=120.0, reinforcement=STRIP)
    gap = _sheet(phi_f=0.0, c=120.0, reinforcement=STRIP,
                 interface_gap_possible=True)
    assert plain.FOS_sliding == pytest.approx(3.340, abs=0.0005)
    V = 19.0 * 6.0 * 4.2
    t = gap.external_terms
    assert t["gap_limited"] and t["R_foundation"] == pytest.approx(0.4 * V)
    assert gap.FOS_sliding == pytest.approx(0.4 * V / t["Pa"], abs=0.0005)
    assert gap.FOS_sliding == pytest.approx(1.981, abs=0.0005)
    [c] = [c for c in gap.choices if "interface_gap_possible" in c["what"]]
    assert c["how"] == "given" and "EN 1997-1" in c["basis"]
    assert "FOS_sliding 3.340" in c["alternatives"][0]
    [c] = [c for c in plain.choices if "interface_gap_possible" in c["what"]]
    assert c["how"] == "default" and "1.981" in c["alternatives"][0]
    g = MSEWallGeometry(wall_height=6.0, reinforcement_length=4.2)
    lrfd = mse.check_external_stability_lrfd(g, 19.0, 34.0, 0.0,
                                             c_foundation=120.0)
    held = mse.check_external_stability_lrfd(g, 19.0, 34.0, 0.0,
                                             c_foundation=120.0,
                                             interface_gap_possible=True)
    assert lrfd["sliding"]["CDR_governing"] == pytest.approx(2.227,
                                                             abs=0.0005)
    F1 = held["unfactored_kN_m"]["F1"]
    assert held["sliding"]["CDR_critical"] == pytest.approx(
        0.4 * V / (1.5 * F1), abs=0.0005)
    assert held["sliding"]["CDR_governing"] == pytest.approx(1.3205,
                                                             abs=0.0006)
    with pytest.raises(ValueError):
        mse.check_external_stability(g, 19.0, 34.0, 18.0, 28.0,
                                     interface_gap_possible=True)


def test_undrained_basis_cites_ufc_and_the_aashto_deferral():
    basis = mse.MSE_SLIDING_COHESION_BASIS["undrained"]
    assert "p. 223" in basis and "full undrained shear strength" in basis
    assert "Article 10.6.3.4 (AASHTO, 2007)" in basis
    assert "not held" in basis and "Eq. 5-39" in basis
