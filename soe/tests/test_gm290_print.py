"""GM-290: soe against the Caltrans Trenching and Shoring Manual (2025),
GEC-4 and UFC 3-220-20 as printed (module_work/gm290/CHECK.md).

Printed worked examples where the print gives the numbers; made-up inputs
elsewhere.
"""

import math

import pytest

from soe.beam_analysis import (FB_RATIO, _supports,
                               analyze_braced_excavation,
                               analyze_cantilever_excavation)
from soe.earth_pressure import (apparent_pressure_soft_clay,
                                apparent_pressure_stiff_clay, coulomb_Kp,
                                fhwa_apparent_pressure_anchored_wall,
                                get_pressure_at_depth,
                                select_apparent_pressure)
from soe.free_earth import apparent_pressure_profile
from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel
from soe.stability import (_interpolate_Nc_bjerrum, _interpolate_Nc_strip,
                           check_basal_heave_bjerrum_eide,
                           check_basal_heave_caltrans,
                           check_basal_heave_terzaghi)


def _sand(t=30.0, g=19.0, phi=32.0):
    return SOEWallLayer(thickness=t, unit_weight=g, friction_angle=phi,
                        soil_type="sand")


def _clay(cu, t=30.0, g=18.0, kind="soft_clay"):
    return SOEWallLayer(thickness=t, unit_weight=g, friction_angle=0.0,
                        cohesion=cu, soil_type=kind)


# ---------------------------------------------------------------------------
# Coulomb Kp: the Caltrans T&S Manual's Eq. 4-3-24 (p. 4-13)
# ---------------------------------------------------------------------------

def _eq_4_3_24(phi, delta, omega, beta):
    phi, delta, omega, beta = map(math.radians, (phi, delta, omega, beta))
    root = math.sqrt(math.sin(delta + phi) * math.sin(phi + beta)
                     / (math.cos(delta - omega) * math.cos(beta - omega)))
    return math.cos(phi + omega) ** 2 / (
        math.cos(omega) ** 2 * math.cos(delta - omega) * (1 - root) ** 2)


@pytest.mark.parametrize("phi, delta, omega, beta", [
    (30, 15, 10, 0), (30, 15, 10, 5), (35, 20, -8, 0), (32, 10, 0, 10)])
def test_coulomb_Kp_is_eq_4_3_24(phi, delta, omega, beta):
    """alpha = 90 deg - omega; the numerator was sin^2(alpha + phi), right
    for a vertical wall only."""
    assert coulomb_Kp(phi, delta, 90.0 - omega, beta) == pytest.approx(
        _eq_4_3_24(phi, delta, omega, beta), rel=1e-9)


# ---------------------------------------------------------------------------
# GEC-4 Appendix A, Design Example 1 (p. A-6): M1 and the embedment (GM-193)
# ---------------------------------------------------------------------------

_EX1 = dict(H=10.0, anchor_depths=[2.5, 6.25], gamma=18.0, phi=33.0,
            surcharge=11.0, spacing=2.5, inclination_deg=15.0)


def test_gec4_example_1_upper_moment():
    """M1 = 13/54 H1^2 pe + ps H1^2/2 = 65.6 + 10.0 = 75.6, 'use 76';
    M2,3 = (pe + ps) L^2/10 = 66."""
    r = fhwa_apparent_pressure_anchored_wall(**_EX1)
    pe, ps = r["pe_kPa"], r["ps_kPa"]
    assert pe == pytest.approx(43.6, abs=0.05)
    assert ps == pytest.approx(3.2, abs=0.05)
    assert r["max_moment_kN_m_per_m"] == pytest.approx(
        13 / 54 * 2.5 ** 2 * pe + ps * 2.5 ** 2 / 2, rel=1e-4)
    assert r["max_moment_kN_m_per_m"] == pytest.approx(75.6, abs=0.15)
    assert [a["TH_kN_per_m"] for a in r["anchors"]] == pytest.approx(
        [168.0, 172.0], abs=0.6)
    assert r["subgrade_reaction_kN_per_m"] == pytest.approx(37.0, abs=0.4)


def test_two_anchors_have_an_embedment():
    """GM-193: free earth support about the lowest anchor (Caltrans 8-4.02
    steps 3-4); FOS_embedment and Kp now read with two anchors."""
    r = fhwa_apparent_pressure_anchored_wall(**_EX1)
    assert r["converged"] is True
    assert 0 < r["embedment_D_FS1_m"] < r["embedment_D_m"]
    assert r["lowest_anchor_load_from_below_kN_per_m"] > 0
    assert r["free_earth_max_moment_kN_m_per_m"] > 0
    whats = [c["what"] for c in r["choices"]]
    assert whats[0] == "embedment factor of safety"
    more = fhwa_apparent_pressure_anchored_wall(**_EX1, FOS_embedment=1.5)
    assert more["embedment_D_m"] > r["embedment_D_m"]
    assert more["embedment_D_FS1_m"] == pytest.approx(r["embedment_D_FS1_m"])
    stronger = fhwa_apparent_pressure_anchored_wall(**_EX1, Kp=6.0)
    assert stronger["embedment_D_m"] < r["embedment_D_m"]
    # the tributary loads and moment do not depend on the embedment
    assert stronger["max_moment_kN_m_per_m"] == r["max_moment_kN_m_per_m"]


# ---------------------------------------------------------------------------
# Caltrans Example 8-2 (pp. 8-21 to 8-34): the hinge method's support loads
# ---------------------------------------------------------------------------

def _ex82_pressure(z):
    """1453 psf x 8 ft spacing on a 50 ft cut, ramped over 2/3 of the 10 ft
    end spans (P1 = P7 = 38,766, P2 = P6 = 38,708, P3-P5 = 116,240 lb)."""
    import numpy as np
    z = np.asarray(z, dtype=float)
    s, a = 1453.0 * 8.0, 20.0 / 3.0
    return np.where(z < a, s * z / a,
                    np.where(z <= 50.0 - a, s, s * (50.0 - z) / a))


def test_caltrans_example_8_2_hinge_loads():
    """T1U = P1 + P2 = 77,474; M1 = 279,729; T1L = P3/2 + M1/S = 86,093;
    T1 = 163,567; T2 = 30,147 + 58,120 = 88,267; T3 = 116,240; T4U =
    58,120, and T4L from free earth support (58,236.5 printed, here given)."""
    s = _supports(_ex82_pressure, [10.0, 20.0, 30.0, 40.0], 50.0,
                  [10.0, 20.0, 30.0, 40.0, 50.0], "hinge", 58236.5)
    up, lo = s["upper"], s["lower"]
    assert up[0] == pytest.approx(77474, rel=1e-3)
    assert lo[0] == pytest.approx(86093, rel=1e-3)
    assert up[0] + lo[0] == pytest.approx(163567, rel=1e-3)
    assert up[1] == pytest.approx(30147, rel=2e-3)
    assert up[1] + lo[1] == pytest.approx(88267, rel=1e-3)
    assert up[2] + lo[2] == pytest.approx(116240, rel=1e-3)
    assert up[3] == pytest.approx(58120, rel=1e-3)
    assert up[3] + lo[3] == pytest.approx(116356.5, rel=1e-3)
    # M1, the cantilever's moment at the first anchor
    assert s["M_max"] == pytest.approx(279729, rel=1e-3)
    assert s["z_max"] == pytest.approx(10.0)


def test_support_load_methods_balance():
    """Every method carries the whole load above the excavation: to the
    supports and, under 'hinge_subgrade', R; 'hinge' sends the span below
    the lowest support to free earth support instead."""
    import numpy as np
    from geotech_common.numpy_compat import trapezoid
    zz = np.linspace(0.0, 50.0, 20001)
    total = trapezoid(_ex82_pressure(zz), zz)
    d, b = [10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0, 50.0]
    sub = _supports(_ex82_pressure, d, 50.0, b, "hinge_subgrade", None)
    assert sum(sub["upper"]) + sum(sub["lower"]) + sub["subgrade"] == \
        pytest.approx(total, rel=1e-4)
    simple = _supports(_ex82_pressure, d, 50.0, b, "simple_spans", None)
    assert sum(simple["upper"]) + sum(simple["lower"]) == pytest.approx(
        total, rel=1e-4)
    assert simple["subgrade"] is None
    # UFC Fig. 4-54: the span below the lowest strut a cantilever on it
    assert simple["lower"][3] == pytest.approx(77474, rel=1e-3)
    hinge = _supports(_ex82_pressure, d, 50.0, b, "hinge", None)
    assert hinge["lower"][3] is None


def test_braced_choices_give_the_other_methods():
    g = ExcavationGeometry(excavation_depth=9.0, soil_layers=[_sand()],
                           support_levels=[SupportLevel(depth=2.0),
                                           SupportLevel(depth=5.5)],
                           surcharge=10.0)
    r = analyze_braced_excavation(g)
    (m,) = [c for c in r.choices if "support_load_method" in c["what"]]
    assert m["how"] == "default" and m["chosen"].startswith("hinge:")
    assert "Fig. 8-7" in m["basis"]
    assert [a.split(" ")[0] for a in m["alternatives"]] == [
        "hinge_subgrade", "simple_spans"]
    s = analyze_braced_excavation(g, support_load_method="hinge_subgrade")
    assert s.subgrade_reaction_kN_per_m > 0
    assert s.to_dict()["subgrade_reaction_kN_per_m"] == \
        s.subgrade_reaction_kN_per_m
    with pytest.raises(ValueError, match="support_load_method"):
        analyze_braced_excavation(g, support_load_method="tributary")


# ---------------------------------------------------------------------------
# Apparent pressure envelopes: UFC 3-220-20 Figs. 4-52 and 4-53, GEC-4
# Fig. 23, Caltrans Sec. 8-3
# ---------------------------------------------------------------------------

def test_soft_clay_envelope_ramps_over_the_top_quarter():
    H, p = 12.0, 50.0
    assert get_pressure_at_depth(0.0, H, "ramped_top", p) == 0.0
    assert get_pressure_at_depth(1.5, H, "ramped_top", p) == pytest.approx(25.0)
    assert get_pressure_at_depth(3.0, H, "ramped_top", p) == pytest.approx(p)
    assert get_pressure_at_depth(H, H, "ramped_top", p) == pytest.approx(p)
    assert get_pressure_at_depth(H, H, "trapezoidal", p) == pytest.approx(0.0)
    g = ExcavationGeometry(excavation_depth=H, soil_layers=[_clay(20.0)],
                           surcharge=0.0)
    ap, pz = apparent_pressure_profile(g)
    assert ap["shape"] == "ramped_top"
    assert float(pz(1.5)) == pytest.approx(ap["max_pressure_kPa"] / 2,
                                           rel=1e-3)
    assert float(pz(11.9)) == pytest.approx(ap["max_pressure_kPa"], rel=1e-3)


def test_soft_clay_Ka_and_its_floor():
    """Ka = 1 - 4 cu / (gamma H) (m = 1, GEC-4 Eq. 11), at least 0.25
    (Caltrans Eq. 8-3-4)."""
    shape, p = apparent_pressure_soft_clay(18.0, 12.0, 20.0)
    assert shape == "ramped_top"
    assert p == pytest.approx((1 - 80 / 216) * 216)
    _, p = apparent_pressure_soft_clay(18.0, 12.0, 45.0)
    assert p == pytest.approx(0.25 * 216)


def test_stiff_clay_coefficient():
    """UFC Fig. 4-53: 0.4 gamma H; 0.2 for minimal movements and a short
    construction period; Caltrans Eq. 8-3-2: 0.2 to 0.4."""
    _, p = apparent_pressure_stiff_clay(18.0, 6.0, 60.0)
    assert p == pytest.approx(0.4 * 108)
    _, p = apparent_pressure_stiff_clay(18.0, 6.0, 60.0, coefficient=0.2)
    assert p == pytest.approx(0.2 * 108)
    with pytest.raises(ValueError, match="0.2 to 0.4"):
        apparent_pressure_stiff_clay(18.0, 6.0, 60.0, coefficient=0.5)
    g = ExcavationGeometry(excavation_depth=6.0,
                           soil_layers=[_clay(60.0, kind="stiff_clay")],
                           support_levels=[SupportLevel(depth=1.5)])
    a = analyze_braced_excavation(g)
    b = analyze_braced_excavation(g, stiff_clay_coefficient=0.2)
    assert b.max_apparent_pressure_kPa == pytest.approx(
        a.max_apparent_pressure_kPa / 2)
    (c,) = [c for c in b.choices if "stiff_clay_coefficient" in c["what"]]
    assert c["how"] == "given" and "Fig. 4-53" in c["basis"]


def test_soft_clay_4_to_6_takes_the_larger():
    """Caltrans Sec. 8-3.02: for 4 < Ns < 6 the larger of Eqs. 8-3-2 and
    8-3-3."""
    layers = [_clay(36.0)]      # N = 18 x 10 / 36 = 5
    r = select_apparent_pressure(layers, 10.0)
    assert r["stability_number"] == pytest.approx(5.0)
    assert r["stiff_clay_ordinate_kPa"] == pytest.approx(72.0)
    assert r["Ka"] == pytest.approx(0.25)
    assert r["governed_by"] == "stiff_clay_ordinate"
    assert r["max_pressure_kPa"] == pytest.approx(72.0)
    assert r["shape"] == "ramped_top"
    low = select_apparent_pressure(layers, 10.0, stiff_clay_coefficient=0.2)
    assert low["governed_by"] == "soft_clay"
    assert low["max_pressure_kPa"] == pytest.approx(45.0)
    deep = select_apparent_pressure([_clay(25.0)], 10.0)     # N = 7.2
    assert "governed_by" not in deep


# ---------------------------------------------------------------------------
# Allowable bending stress
# ---------------------------------------------------------------------------

def test_allowable_bending_stress():
    g = ExcavationGeometry(excavation_depth=4.0, soil_layers=[_sand()],
                           surcharge=10.0)
    r = analyze_cantilever_excavation(g, Fy=345.0)
    assert FB_RATIO == 0.60
    assert r.required_Sx_cm3 == pytest.approx(
        r.max_moment_kNm_per_m / (0.60 * 345e3) * 1e6, rel=1e-3)
    s = analyze_cantilever_excavation(g, Fy=345.0, Fb_ratio=0.55)
    assert s.required_Sx_cm3 == pytest.approx(
        r.required_Sx_cm3 * 0.60 / 0.55, rel=1e-3)
    (c,) = [c for c in s.choices if "Fb_ratio" in c["what"]]
    assert c["how"] == "given"
    assert any("0.6 Fy" in a for a in c["alternatives"])
    with pytest.raises(ValueError, match="Fb_ratio"):
        analyze_cantilever_excavation(g, Fb_ratio=1.5)


# ---------------------------------------------------------------------------
# Basal heave: GEC-4 Eqs. 35 and 37, Fig. 51(c); Caltrans Sec. 10-3
# ---------------------------------------------------------------------------

def test_gec4_eq_35():
    H, cu, g, B, q = 6.0, 25.0, 18.0, 12.0, 10.0
    r = check_basal_heave_terzaghi(H=H, cu=cu, gamma=g, q_surcharge=q, B=B)
    Nc = 5.81       # Fig. 51(c), B/L = 0, H/B = 0.5
    assert r.parameters["Nc"] == pytest.approx(Nc)
    assert r.FOS == pytest.approx(
        Nc * cu / (g * H + q - math.sqrt(2) * cu * H / B), abs=1e-3)
    assert r.parameters["equation"] == "GEC-4 Eq. 35"
    assert _interpolate_Nc_strip(2.0) == pytest.approx(6.97)
    assert _interpolate_Nc_strip(9.0) == pytest.approx(7.44)


def test_gec4_eq_37_without_a_width():
    r = check_basal_heave_terzaghi(H=6.0, cu=25.0, gamma=18.0)
    assert r.parameters["Nc"] == pytest.approx(5.14)
    assert r.FOS == pytest.approx(5.14 * 25 / 108, abs=1e-3)
    assert r.parameters["equation"] == "GEC-4 Eq. 37"


def test_gec4_eq_35_no_answer_when_the_side_shear_carries_it():
    r = check_basal_heave_terzaghi(H=6.0, cu=60.0, gamma=18.0, B=3.0)
    assert r.FOS == float("inf") and r.passes is False
    assert "NOT demonstrated" in r.notes[0]


def test_ufc_fig_2_7b_beside_eq_35():
    """UFC 3-220-20 Fig. 2-7(b): FBH = (5.7 Su B1 + Su H) / ((gamma H + q)
    B1), B1 = 0.7B; it stays finite where Eq. 35's subtraction runs away
    (UFC p. 73 note 10)."""
    H, cu, g, B, q = 6.0, 50.0, 18.0, 4.0, 0.0
    r = check_basal_heave_terzaghi(H=H, cu=cu, gamma=g, q_surcharge=q, B=B)
    B1 = 0.7 * B
    ufc = (5.7 * cu * B1 + cu * H) / ((g * H + q) * B1)
    assert r.parameters["FOS_ufc_fig_2_7b"] == pytest.approx(ufc, abs=1e-3)
    assert ufc == pytest.approx(3.63, abs=0.01)
    assert r.FOS > 100                    # Eq. 35, side shear 106 of 108 kPa
    assert any("Fig. 2-7(b)" in n and f"{ufc:.2f}" in n for n in r.notes)
    wide = check_basal_heave_terzaghi(H=H, cu=cu, gamma=g)
    assert wide.parameters["FOS_ufc_fig_2_7b"] is None
    assert not any("Fig. 2-7(b)" in n for n in wide.notes)


def test_caltrans_example_10_2():
    """H 30 ft, B 15 ft, L 45 ft, q 300 psf, c 500 psf, gamma 120 pcf:
    Nc(square) 8.5 at H/B = 2, Nc 7.6 at L/B = 3; about 40 kip/ft against
    26.0, FS 1.54. In lb and ft."""
    assert _interpolate_Nc_bjerrum(2.0, 1.0) == pytest.approx(8.5, abs=0.03)
    r = check_basal_heave_caltrans(H=30.0, cu=500.0, gamma=120.0, B=15.0,
                                   L=45.0, q_surcharge=300.0)
    assert r.parameters["Nc"] == pytest.approx(7.6, abs=0.03)
    assert r.resistance / 1000 == pytest.approx(40.0, abs=0.3)
    assert r.demand / 1000 == pytest.approx(25.95, abs=1e-3)   # "26.0"
    assert r.FOS == pytest.approx(1.54, abs=0.01)


def test_bjerrum_eide_shape_factor():
    """Eq. 10-3-4: Nc = Nc(square) (0.84 + 0.16 B/L)."""
    sq = check_basal_heave_bjerrum_eide(H=8.0, cu=30.0, gamma=18.0, Be=8.0,
                                        Le=8.0)
    rect = check_basal_heave_bjerrum_eide(H=8.0, cu=30.0, gamma=18.0, Be=8.0,
                                          Le=32.0)
    assert sq.parameters["Nc"] == pytest.approx(7.68)
    assert rect.parameters["Nc"] == pytest.approx(7.68 * (0.84 + 0.04),
                                                  abs=1e-3)


# ---------------------------------------------------------------------------
# The calc package reads each support load method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["hinge", "hinge_subgrade", "simple_spans"])
def test_calc_steps_each_method(method):
    from soe.calc_steps import get_calc_steps, get_tables
    g = ExcavationGeometry(excavation_depth=9.0, soil_layers=[_sand()],
                           support_levels=[SupportLevel(depth=2.0),
                                           SupportLevel(depth=5.5)],
                           surcharge=10.0)
    r = analyze_braced_excavation(g, support_load_method=method)
    sections = get_calc_steps(r, {"geometry": g, "Fy": 345.0})
    titles = [s.title for s in sections]
    assert "Support Loads" in titles
    assert get_tables(r, {"geometry": g})


def test_calc_steps_without_a_lowest_load():
    from soe.calc_steps import get_calc_steps
    g = ExcavationGeometry(excavation_depth=8.0, soil_layers=[_clay(30.0)],
                           support_levels=[SupportLevel(depth=1.5)],
                           surcharge=10.0)
    r = analyze_braced_excavation(g)
    assert r.support_reactions[0]["load_kN_per_m"] is None
    sections = get_calc_steps(r, {"geometry": g, "Fy": 345.0})
    loads = next(s for s in sections if s.title == "Support Loads")
    assert any(getattr(i, "notes", "").startswith("Not found")
               for i in loads.items)
    assert "not found" in r.summary()
