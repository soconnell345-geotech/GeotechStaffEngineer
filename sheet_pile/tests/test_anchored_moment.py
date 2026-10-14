"""Anchored sheet pile wall: the shear and moment take the anchor reaction at
the anchor (GM-69, 2026-10-14).

Until 2026-10-14 the moment loop started the shear at -T at the top of the
wall, overstating every moment below the anchor by T x anchor_depth (+100 %
on the published example below). See sheet_pile/DESIGN.md, "Anchored wall
shear and moment".

Published pin: USS Steel Sheet Piling Design Manual (United States Steel,
1984 reprint by FHWA), Design Example No. 2, "Design of Anchor Sheet Pile
Wall - Cohesive Soil With Sand Backfill - Free Earth Support Method",
pp. 101-102.
"""

import math

import numpy as np
import pytest

from sheet_pile import WallSoilLayer, analyze_anchored

FT = 0.3048                 # m
PCF = 0.157087464           # kN/m3
PSF = 0.04788026            # kPa
LB_FT = 0.01459390          # kN/m per lb/ft
FTLB_FT = 0.00444822        # kN-m/m per ft-lb/ft


def _uss_example_2():
    """The example's wall in SI. Water stands at the low-water level on both
    sides, so the example works in effective unit weights with no net
    water; the layers carry those weights and no water table. The example's
    Ka = 0.33 for the backfill is entered as the friction angle whose
    Rankine Ka is 0.33. c = 500 psf below the dredge line is the example's
    factored strength (FS = 1.70), so the passive is not factored again."""
    phi = 90.0 - 2.0 * math.degrees(math.atan(math.sqrt(0.33)))
    layers = [
        WallSoilLayer(6.0 * FT, 115 * PCF, friction_angle=phi),
        WallSoilLayer(2.5 * FT, 60 * PCF, friction_angle=phi),
        WallSoilLayer(11.5 * FT, 65 * PCF, friction_angle=0.0,
                      cohesion=400 * PSF),
        WallSoilLayer(30.0 * FT, 65 * PCF, friction_angle=0.0,
                      cohesion=500 * PSF),
    ]
    return analyze_anchored(excavation_depth=20.0 * FT,
                            anchor_depth=5.0 * FT, soil_layers=layers,
                            FOS_passive=1.0)


def _hand_moment_from_published_diagram(T=3193.0):
    """M_max (ft-lb/ft) from the example's printed pressure diagram (p. 101:
    0 -> 228 psf over 6 ft, 228 -> 278 psf over 2.5 ft, then 40 + 65 y psf
    in the clay) and its printed T, by the zero-shear procedure of the
    manual's Example No. 1 (p. 98)."""
    p1, p2 = 684.0, 633.0        # published resultants, lb/ft
    # zero shear in the clay: p1 + p2 + 40 y + 32.5 y^2 = T
    y = (-40.0 + math.sqrt(40.0 ** 2 + 4 * 32.5 * (T - p1 - p2))) / 65.0
    z0 = 8.5 + y
    m = (T * (z0 - 5.0)
         - p1 * (z0 - 4.0)
         - 228.0 * 2.5 * (z0 - 7.25)
         - 0.5 * 50.0 * 2.5 * (z0 - (6.0 + 2.0 * 2.5 / 3.0))
         - 40.0 * y ** 2 / 2.0 - 65.0 * y ** 3 / 6.0)
    return z0, m


class TestUSSExampleNo2:
    def test_embedment_matches_the_published_equation(self):
        r = _uss_example_2()
        # published D^2 + 30 D = 257.4 -> D = 6.96 ft ("D = 7.0'")
        d_published = (-30.0 + math.sqrt(900.0 + 4 * 257.4)) / 2.0
        assert r.embedment_depth / FT == pytest.approx(d_published, rel=0.01)

    def test_anchor_force_matches_the_published_value(self):
        r = _uss_example_2()
        assert r.anchor_force / LB_FT == pytest.approx(3193.0, rel=0.01)

    def test_maximum_moment_matches_the_published_diagram(self):
        r = _uss_example_2()
        z0, m_hand = _hand_moment_from_published_diagram()
        assert z0 == pytest.approx(15.51, abs=0.01)
        assert m_hand == pytest.approx(15773.0, abs=5.0)
        assert r.max_moment / FTLB_FT == pytest.approx(m_hand, rel=0.02)
        assert r.max_moment_depth / FT == pytest.approx(z0, abs=0.2)

    def test_not_the_old_overstatement(self):
        r = _uss_example_2()
        old = r.max_moment + r.anchor_force * r.anchor_depth
        assert r.max_moment < 0.6 * old


class TestUniformSand:
    LAYERS = [WallSoilLayer(thickness=20.0, unit_weight=18.0,
                            friction_angle=30.0)]

    def test_hand_check(self):
        """Ka = 1/3: the shear is zero at z0 = sqrt(2T/(Ka gamma)) above the
        excavation level, and M = Ka gamma z0^3/6 - T (z0 - a)."""
        r = analyze_anchored(excavation_depth=6.0, anchor_depth=1.5,
                             soil_layers=self.LAYERS)
        ka, g, t, a = 1.0 / 3.0, 18.0, r.anchor_force, 1.5
        z0 = math.sqrt(2.0 * t / (ka * g))
        m = abs(ka * g * z0 ** 3 / 6.0 - t * (z0 - a))
        assert m == pytest.approx(139.6, abs=0.1)
        assert r.max_moment == pytest.approx(m, rel=0.01)
        assert r.max_moment_depth == pytest.approx(z0, abs=0.05)

    def test_agrees_with_the_soe_support_solver(self):
        from soe.free_earth import LayeredProfile, solve_about_support
        r = analyze_anchored(excavation_depth=6.0, anchor_depth=1.5,
                             soil_layers=self.LAYERS, FOS_passive=1.0)
        s = solve_about_support(LayeredProfile(self.LAYERS, H=6.0),
                                pivot=1.5, FS=1.0)
        assert r.embedment_depth == pytest.approx(s.D_prime, rel=0.01)
        assert r.anchor_force == pytest.approx(s.support_load, rel=0.01)
        assert r.max_moment == pytest.approx(s.max_moment, rel=0.02)

    def test_free_earth_support_closes_at_the_toe(self):
        r = analyze_anchored(excavation_depth=6.0, anchor_depth=1.5,
                             soil_layers=self.LAYERS)
        p = np.array(r.profile)
        assert abs(p[-1, 5]) < 1e-6 * r.anchor_force + 1e-9     # V(toe)
        assert abs(p[-1, 6]) < 0.03 * r.max_moment               # M(toe)
        # the reaction enters at the anchor: shear drops by T there
        z = p[:, 0]
        i = int(np.searchsorted(z, 1.5))
        dz = z[1] - z[0]
        assert p[i, 5] - p[i - 1, 5] == pytest.approx(
            -r.anchor_force + p[i, 4] * dz, rel=1e-9)
