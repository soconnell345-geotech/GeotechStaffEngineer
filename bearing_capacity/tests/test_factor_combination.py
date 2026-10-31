"""BACKLOG GM-281: how the correction factors combine, against the print.

UFC 3-220-20 Eq. 5-18 to 5-20 (p. 293) multiply the factors; Brinch
Hansen's phi = 0 factors are the exception, added in Eq. 5-21 (p. 295):
1 + s_c + d_c - i_c - b_c - g_c, with d_c = 0.4 k (Table 5-7). Vesic's
factors (GEC-6 Eq. 5-14, p. 63; EN 1997-1 D.1, p. 158) multiply and print
no d_c, so the module's phi = 0 d_c is Table 5-7's and is now added as
Eq. 5-21 adds it; it was multiplied, (1 + s_c)(1 + 0.4 k).

UFC 3-220-20 Example B-3, Footing 1 (pp. 618-623), sized for undrained
bearing: B = 6 ft, Df = 3 ft, s_u = 1000 psf, gamma = 115 pcf, Q = 150 kips,
F_BC = 2.5, q_net = Q/(B L) set equal to q_all,net = q_ult/F_BC - sigma_zD.
Brinch Hansen (Table B-17, p. 622): psi_c = [1 + 0.2(B/L)] + 0.4(Df/B),
L >= 9.97 ft (p. 623). Meyerhof (Table B-16, p. 621): psi_c =
[1 + 0.2(B/L)][1 + 0.2(Df/B)], L >= 10.84 ft (p. 622). (Both tables print
the depth term as "(3 ft)/L"; the lengths follow from Df/B = 0.5.)
"""

import math

import pytest

from bearing_capacity.calc_steps import get_calc_steps
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer

FT = 0.3048                       # m
PSF = 4.4482216152605 / 1000.0 / FT ** 2   # kPa per psf (lbf/ft2)
PCF = PSF / FT                    # kN/m3 per pcf (lbf/ft3)
KIP = 4.4482216152605             # kN


def _example_b3_footing_1(L_ft, factor_method):
    return BearingCapacityAnalysis(
        footing=Footing(width=6 * FT, length=L_ft * FT, depth=3 * FT,
                        shape="rectangular"),
        soil=BearingSoilProfile(layer1=SoilLayer(
            cohesion=1000 * PSF, friction_angle=0.0, unit_weight=115 * PCF)),
        factor_of_safety=2.5, factor_method=factor_method).compute()


def _required_length_ft(factor_method):
    """The L at which q_net = Q/(B L) meets q_all,net, by bisection."""
    def gap(L_ft):
        r = _example_b3_footing_1(L_ft, factor_method)
        return r.q_allowable_net - 150 * KIP / (6 * FT * L_ft * FT)
    lo, hi = 6.0, 30.0
    assert gap(lo) < 0 < gap(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_example_b3_brinch_hansen_length():
    # Vesic's phi = 0 shape and depth factors are Table 5-7's, combined as
    # Eq. 5-21: L >= 9.97 ft as printed (multiplied, they gave 9.75 ft)
    assert _required_length_ft("vesic") == pytest.approx(9.97, abs=0.01)


def test_example_b3_brinch_hansen_psi_c():
    r = _example_b3_footing_1(9.97, "vesic")
    psi_c = r.term_cohesion / (r.Nc * 1000 * PSF)
    assert psi_c == pytest.approx(1 + 0.2 * 6 / 9.97 + 0.4 * 0.5, rel=1e-12)
    assert r.c_term_adds_dc


def test_example_b3_meyerhof_length():
    # Meyerhof's factors multiply (Table B-16): L >= 10.84 ft as printed
    assert _required_length_ft("meyerhof") == pytest.approx(10.84, abs=0.01)
    r = _example_b3_footing_1(10.84, "meyerhof")
    psi_c = r.term_cohesion / (r.Nc * 1000 * PSF)
    assert psi_c == pytest.approx((1 + 0.2 * 6 / 10.84) * (1 + 0.2 * 0.5),
                                  rel=1e-12)
    assert not r.c_term_adds_dc


# ── the rule, on made-up footings ──────────────────────────────────────

def _clay(phi=0.0, c=37.0, **kw):
    footing = dict(width=1.6, length=2.4, depth=0.8, shape="rectangular")
    footing.update(kw.pop("footing", {}))
    return BearingCapacityAnalysis(
        footing=Footing(**footing),
        soil=BearingSoilProfile(layer1=SoilLayer(
            cohesion=c, friction_angle=phi, unit_weight=17.0)), **kw)


def test_phi_zero_vesic_adds_dc_to_the_other_factors():
    r = _clay(footing=dict(base_tilt=6.0), ground_slope=4.0).compute()
    assert r.dc > 1.0 and r.sc > 1.0 and r.bc < 1.0 and r.gc < 1.0
    assert r.term_cohesion == pytest.approx(
        37.0 * r.Nc * (r.sc * r.ic * r.bc * r.gc + r.dc - 1.0), rel=1e-12)
    assert r.term_cohesion < 37.0 * r.Nc * r.sc * r.dc * r.bc * r.gc


def test_without_a_depth_factor_phi_zero_still_multiplies():
    # GEC-6 Eq. 5-14 prints c Nc s_c b_c; no d_c, nothing is added
    for footing, kw in (({}, {"depth_factor_method": "none"}),
                        ({}, {"depth_factor_method": "gec6"}),
                        ({"depth": 0.0}, {})):
        r = _clay(footing={"base_tilt": 6.0, **footing}, **kw).compute()
        assert r.dc == 1.0
        assert r.term_cohesion == pytest.approx(
            37.0 * r.Nc * r.sc * r.bc, rel=1e-12)


def test_meyerhof_and_phi_above_zero_multiply():
    m = _clay(factor_method="meyerhof").compute()
    assert m.term_cohesion == pytest.approx(
        37.0 * m.Nc * m.sc * m.dc, rel=1e-12)
    v = _clay(phi=12.0).compute()
    assert not v.c_term_adds_dc
    assert v.term_cohesion == pytest.approx(
        37.0 * v.Nc * v.sc * v.dc, rel=1e-12)


def test_a_strip_is_unchanged():
    # s_c = 1: (1)(1 + 0.4 k) = 1 + 0.4 k either way (a strip's L is 1e6 m,
    # so s_c is 1 to 1e-6)
    r = _clay(footing=dict(shape="strip", length=None)).compute()
    k = 0.8 / 1.6
    assert r.term_cohesion == pytest.approx(37.0 * 5.14 * (1 + 0.4 * k),
                                            rel=1e-6)


def test_the_choice_table_and_calc_steps_say_so():
    a = _clay()
    r = a.compute()
    d = [c for c in r.choices if c["what"] == "depth factors"][0]
    assert "c N_c (s_c i_c b_c g_c + 0.4 k)" in d["basis"]
    assert "Eq. 5-21" in d["basis"] and "Table B-17" in d["basis"]
    assert "multiplies by" not in d["basis"]
    t = {t.name: t for t in r.get_tables()}["factors_and_terms"]
    assert "c Nc (sc ic bc gc + dc - 1)" in t.note
    steps = {getattr(s, "title", ""): s for sec in get_calc_steps(r, a)
             for s in getattr(sec, "items", [])}
    eq = steps["General Bearing Capacity Equation"]
    assert "(s_c × i_c × b_c × g_c + d_c - 1)" in eq.equation
    assert "Eq. 5-21" in eq.reference
    assert "Eq. 5-21" in steps["Depth Factors"].reference
    # phi > 0 keeps the product
    b = _clay(phi=12.0)
    s = {getattr(x, "title", ""): x for sec in get_calc_steps(b.compute(), b)
         for x in getattr(sec, "items", [])}
    assert "s_c × d_c × i_c" in s["General Bearing Capacity Equation"].equation
