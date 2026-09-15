"""Free-earth-support solvers (soe.free_earth) and the SOE wall analyses on them.

Field feedback 2026-09-15 (Nairobi SOE re-run, item N2): the old cantilever
used only the first soil layer and put the passive resultant at H + D/3 about
the wall base (a 6.3 m wall in uniform sand got 2.5 m of embedment against
6.9 m required); the braced embedment used d/3 and D/3 arms; the braced top
span was simply supported; surcharge and water were left out of braced loads.

Pins, in order of authority:
1. Caltrans Trenching and Shoring Manual (2025) Example 8-1 (published).
2. Closed-form Simplified Method for a uniform dry sand (independent algebra).
3. sheet_pile.analyze_cantilever (reconciled with Das; tests may cross modules).
"""

import math

import numpy as np
import pytest

from soe.beam_analysis import (
    analyze_braced_excavation,
    analyze_cantilever_excavation,
)
from soe.embedment import compute_embedment, embedment_detail
from soe.free_earth import (
    LayeredProfile,
    solve_about_support,
    solve_cantilever,
)
from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel

FT = 0.3048
PCF = 0.157087464     # pcf -> kN/m3
LBFT = 0.0145939029   # lb/ft -> kN/m
FTLB = 0.00444822162  # ft-lb/ft -> kN.m/m
GW = 9.81


def _sand(thickness=40.0, gamma=18.0, phi=30.0, c=0.0, soil_type="sand"):
    return SOEWallLayer(thickness=thickness, unit_weight=gamma,
                        friction_angle=phi, cohesion=c, soil_type=soil_type)


# ---------------------------------------------------------------------------
# 1. Caltrans T&S Manual Example 8-1 (published)
# ---------------------------------------------------------------------------

class TestCaltransExample8_1:
    """Single ground anchor sheet pile wall: H = 25 ft, anchor 10 ft below the
    top, gamma = 115 pcf, phi = 30 (Ka = 1/3), log-spiral Kp = 4.7, FHWA
    trapezoid above the dredge line, FS = 1.3 on embedment. Published:
    D = 6.09 ft, D' = 4.89 ft, T1 = 14,254 lb/ft, M = 22,494 ft-lb/ft at the
    anchor, zero shear 9.69 ft below T1 where |M| = 13,020 ft-lb/ft, and
    Vmax = T1L = 8,026 lb/ft (manual pp. 8-11 to 8-19)."""

    @pytest.fixture(scope="class")
    def sol(self):
        H, d, g = 25 * FT, 10 * FT, 115 * PCF
        prof = LayeredProfile([_sand(100.0, g, 30.0)], H, Kp=4.7)
        pe = 0.65 * prof.Ka[0] * g * H ** 2 / (H - d / 3 - (H - d) / 3)
        top, bot = 2 / 3 * d, H - 2 / 3 * (H - d)

        def trapezoid(z):
            z = np.asarray(z, float)
            return np.where(z < top, pe * z / top,
                            np.where(z <= bot, pe, pe * (H - z) / (H - bot)))

        return solve_about_support(prof, pivot=d, body_top=0.0, FS=1.3,
                                   above=trapezoid)

    def test_embedment_fs_1_3(self, sol):
        assert sol.D / FT == pytest.approx(6.09, rel=0.01)

    def test_embedment_fs_1(self, sol):
        assert sol.D_prime / FT == pytest.approx(4.89, rel=0.01)

    def test_anchor_load(self, sol):
        assert sol.support_load / LBFT == pytest.approx(14254, rel=0.01)

    def test_max_moment_at_anchor(self, sol):
        assert sol.max_moment / FTLB == pytest.approx(22494, rel=0.005)
        assert sol.max_moment_depth / FT == pytest.approx(10.0, abs=0.05)

    def test_zero_shear_point(self, sol):
        assert sol.zero_shear_depth / FT == pytest.approx(19.69, rel=0.01)
        # the manual rounds y' to 4.69 ft by hand; integration gives ~12,900
        assert sol.moment_at_zero_shear / FTLB == pytest.approx(13020, rel=0.015)

    def test_max_shear(self, sol):
        assert sol.max_shear / LBFT == pytest.approx(8026, rel=0.01)


# ---------------------------------------------------------------------------
# 2. Simplified Method, closed form (uniform dry sand, no surcharge)
# ---------------------------------------------------------------------------

class TestCantileverClosedForm:
    """Moments about O: Ka g (H+D0)^3 / 6 = (Kp/FS) g D0^3 / 6, so
    D0 = H / (r - 1) with r = (Kp / (FS Ka))^(1/3). Zero shear at y below the
    excavation where Ka (H+y)^2 = (Kp/FS) y^2."""

    H, G, PHI, FS = 5.0, 18.0, 30.0, 1.5

    def _geometry(self):
        return ExcavationGeometry(excavation_depth=self.H,
                                  soil_layers=[_sand(40.0, self.G, self.PHI)],
                                  surcharge=0.0)

    def test_embedment_and_moment(self):
        Ka = math.tan(math.radians(45 - self.PHI / 2)) ** 2
        Kp = 1.0 / Ka
        k = Kp / self.FS / Ka
        D0 = self.H / (k ** (1 / 3) - 1)
        y = self.H / (math.sqrt(k) - 1)
        M = (Ka * self.G * (self.H + y) ** 3 - Kp / self.FS * self.G * y ** 3) / 6
        r = analyze_cantilever_excavation(self._geometry(), FOS_passive=self.FS)
        assert r.embedment_converged_m == pytest.approx(D0, rel=0.002)
        assert r.required_embedment_m == pytest.approx(1.2 * D0, rel=0.005)
        assert r.max_moment_kNm_per_m == pytest.approx(M, rel=0.005)
        assert r.max_moment_depth_m == pytest.approx(self.H + y, rel=0.01)
        assert r.net_force_at_O_kN_per_m < 0  # resistance exceeds push: OK
        assert r.Ka == pytest.approx(Ka, abs=1e-4)
        assert r.Kp == pytest.approx(Kp, abs=1e-3)

    def test_embedment_increase_is_a_multiplier(self):
        a = analyze_cantilever_excavation(self._geometry(), embedment_increase=1.0)
        b = analyze_cantilever_excavation(self._geometry(), embedment_increase=1.3)
        assert b.embedment_converged_m == pytest.approx(a.embedment_converged_m)
        assert b.required_embedment_m == pytest.approx(
            1.3 * a.required_embedment_m, rel=0.01)


# ---------------------------------------------------------------------------
# 3. Parity with sheet_pile.analyze_cantilever (layered, wet, surcharge)
# ---------------------------------------------------------------------------

_NAIROBI = [  # (thickness, gamma, phi, c): the 2026-09-15 session's profile
    (1.5, 14.9, 0.0, 19.0), (2.5, 16.5, 33.0, 50.0), (2.0, 16.5, 33.0, 50.0),
    (0.3, 14.8, 26.0, 9.0), (2.7, 14.8, 26.0, 9.0), (1.0, 20.7, 38.0, 0.0),
    (20.0, 20.7, 38.0, 0.0),
]


@pytest.mark.parametrize("layers, gwt", [
    ([(30.0, 16.5, 33.0, 0.0)], None),
    ([(30.0, 16.5, 33.0, 0.0)], 4.0),
    (_NAIROBI, 4.0),
])
def test_cantilever_matches_sheet_pile(layers, gwt):
    from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
    prof = LayeredProfile([SOEWallLayer(t, g, p, c) for t, g, p, c in layers],
                          6.3, surcharge=7.2, gwt_retained=gwt)
    ours = solve_cantilever(prof, FS_passive=1.5, embedment_increase=1.0)
    theirs = analyze_cantilever(
        6.3, [WallSoilLayer(t, g, p, c) for t, g, p, c in layers],
        gwt_depth_active=gwt, surcharge=7.2, FOS_passive=1.5)
    assert ours.D0 == pytest.approx(theirs.embedment_converged, rel=0.005)
    assert ours.max_moment == pytest.approx(theirs.max_moment, rel=0.01)


# ---------------------------------------------------------------------------
# Regressions from the session
# ---------------------------------------------------------------------------

def _nairobi_geometry(phi_mid=33.0, c_mid=50.0, supports=()):
    """The session's profile; "mid" = Stratum C, the two phi = 33 layers."""
    layers = []
    for t, g, p, c in _NAIROBI:
        if p == 33.0:
            p, c = phi_mid, c_mid
        layers.append(SOEWallLayer(t, g, p, c,
                                   soil_type="soft_clay" if p == 0 else "sand"))
    return ExcavationGeometry(excavation_depth=6.3, soil_layers=layers,
                              support_levels=list(supports), surcharge=7.2,
                              gwt_depth=4.0)


def test_cantilever_uses_every_layer():
    """The session got Ka = Kp = 1.0 (the top clay layer) and 0.60 m of
    embedment, identical for every phi of the layers below."""
    base = analyze_cantilever_excavation(_nairobi_geometry(33.0, c_mid=0.0))
    weak = analyze_cantilever_excavation(_nairobi_geometry(26.0, c_mid=0.0))
    assert base.required_embedment_m > 5.0
    assert base.Ka == pytest.approx(0.3905, abs=1e-3)  # the phi = 26 layer at H
    assert weak.max_moment_kNm_per_m > base.max_moment_kNm_per_m
    assert weak.required_embedment_m > base.required_embedment_m
    assert any("5.5 m" in n for n in base.notes)  # H = 6.3 m is past typical


def test_cohesive_stratum_c_carries_no_active_pressure():
    """With c = 50 kPa, Ka*s'v - 2c*sqrt(Ka) < 0 through Stratum C at any
    phi from 26 to 33 deg, so phi there cannot change a Rankine cantilever --
    the same reason PYWall's per-layer active pressure for those layers is
    0.00 and only its sand envelope loads them (FINDINGS N17). Removing the
    cohesion does change the answer."""
    a = analyze_cantilever_excavation(_nairobi_geometry(33.0, c_mid=50.0))
    b = analyze_cantilever_excavation(_nairobi_geometry(26.0, c_mid=50.0))
    c0 = analyze_cantilever_excavation(_nairobi_geometry(33.0, c_mid=0.0))
    assert a.max_moment_kNm_per_m == pytest.approx(b.max_moment_kNm_per_m)
    assert c0.max_moment_kNm_per_m > a.max_moment_kNm_per_m


def test_braced_embedment_about_the_lowest_support():
    one = _nairobi_geometry(supports=[SupportLevel(depth=2.0)])
    d13 = compute_embedment(one, FOS_passive=1.3)
    d15 = compute_embedment(one, FOS_passive=1.5)
    assert 0 < d13 < d15
    # single support: the whole wall is the free body
    detail = embedment_detail(one, FOS_passive=1.3)
    assert detail.D == pytest.approx(d13)
    # two supports: the body starts at the lowest support (hinge method)
    two = _nairobi_geometry(supports=[SupportLevel(depth=1.5),
                                      SupportLevel(depth=4.0)])
    assert compute_embedment(two, FOS_passive=1.3) != pytest.approx(d13)


class TestBracedLoads:
    H, G, PHI = 6.0, 18.0, 30.0

    def _geo(self, q=0.0, gwt=None, depths=(3.0,)):
        return ExcavationGeometry(
            excavation_depth=self.H, soil_layers=[_sand(40.0, self.G, self.PHI)],
            support_levels=[SupportLevel(depth=d) for d in depths],
            surcharge=q, gwt_depth=gwt)

    def test_single_support_takes_envelope_plus_surcharge_plus_water(self):
        Ka = math.tan(math.radians(45 - self.PHI / 2)) ** 2
        q, gwt = 12.0, 4.0
        g_eff = (self.G * gwt + (self.G - GW) * (self.H - gwt)) / self.H
        expected = (0.65 * Ka * g_eff * self.H + Ka * q) * self.H \
            + 0.5 * GW * (self.H - gwt) ** 2
        r = analyze_braced_excavation(self._geo(q=q, gwt=gwt))
        assert r.support_reactions[0]["load_kN_per_m"] == pytest.approx(
            expected, rel=0.003)
        assert r.water_pressure_included is True
        assert r.surcharge_pressure_kPa == pytest.approx(Ka * q, abs=0.01)

    def test_top_span_is_a_cantilever(self):
        """Uniform envelope p, support at 3 m in a 6 m cut: the span above the
        support is a cantilever (p d^2 / 2 = 4.5 p), not simply supported
        (p d^2 / 8)."""
        r = analyze_braced_excavation(self._geo())
        p = r.max_apparent_pressure_kPa
        fe = r.free_earth["max_moment_kNm_per_m"]
        assert r.max_moment_kNm_per_m == pytest.approx(max(4.5 * p, fe), rel=0.005)
        assert r.max_moment_kNm_per_m >= 4.5 * p * 0.995

    def test_embedment_fs_reported(self):
        r = analyze_braced_excavation(self._geo(), FOS_passive=1.3)
        assert r.embedment_FS == 1.3
        assert r.free_earth["embedment_D_m"] == pytest.approx(
            r.required_embedment_m, abs=0.01)
        assert r.free_earth["embedment_D_FS1_m"] < r.required_embedment_m


# ---------------------------------------------------------------------------
# N3: FHWA apparent-pressure wall with ONE anchor level (Caltrans Example 8-1)
# ---------------------------------------------------------------------------

class TestFhwaSingleAnchor:
    """``fhwa_apparent_pressure_anchored_wall`` with one anchor used to return
    only the load above the anchor (TH_upper) and the moment above it, with
    TH = None and nothing saying the result was partial; the 2026-09-15
    session reported TH_upper as the anchor load (field feedback N3).
    Published Example 8-1: T1 = 14,254 lb/ft (T1U 6,228 + T1L 8,026),
    TH = 143.87 kips and T = 148.95 kips per anchor at 10 ft spacing and 15
    deg, D = 6.09 ft, Mmax = 22,494 ft-lb/ft at the anchor."""

    KIP = 4.44822162  # kN

    @pytest.fixture(scope="class")
    def r(self):
        from soe.earth_pressure import fhwa_apparent_pressure_anchored_wall
        return fhwa_apparent_pressure_anchored_wall(
            H=25 * FT, anchor_depths=[10 * FT], gamma=115 * PCF, phi=30.0,
            surcharge=0.0, spacing=10 * FT, inclination_deg=15.0, Kp=4.7,
            FOS_embedment=1.3)

    def test_anchor_load_is_total(self, r):
        a = r["anchors"][0]
        assert a["TH_upper_kN_per_m"] / LBFT == pytest.approx(6228, rel=0.01)
        assert a["TH_lower_kN_per_m"] / LBFT == pytest.approx(8026, rel=0.01)
        assert a["TH_kN_per_m"] / LBFT == pytest.approx(14254, rel=0.01)

    def test_design_load_per_anchor(self, r):
        """The manual's TH = 143.87 kips is not its own T1 x 10 ft (142.54
        kips); T = TH / cos 15 carries the same ~0.9 % (V-013 allows 2 %).
        Pin to the published numbers at 1.5 % and to the consistent
        T1 x s / cos 15 = 147.57 kips at 0.5 %."""
        a = r["anchors"][0]
        TH_kips = a["TH_kN_per_m"] * 10 * FT / self.KIP
        T_kips = a["design_load_kN"] / self.KIP
        assert TH_kips == pytest.approx(143.87, rel=0.015)
        assert T_kips == pytest.approx(148.95, rel=0.015)
        assert T_kips == pytest.approx(
            14254 * 10 / 1000 / math.cos(math.radians(15)), rel=0.005)

    def test_embedment_and_moment(self, r):
        assert r["embedment_D_m"] / FT == pytest.approx(6.09, rel=0.01)
        assert r["max_moment_kN_m_per_m"] / FTLB == pytest.approx(22494, rel=0.005)
        assert r["subgrade_reaction_kN_per_m"] is None
        assert "Example 8-1" in r["single_anchor_method"]

    def test_two_anchor_path_unchanged(self):
        """GEC-4 Design Example 1 (2 anchors) still goes through the tributary
        formulas: TH1 = 168, TH2 = 172, R = 37 kN/m (V-016)."""
        from soe.earth_pressure import fhwa_apparent_pressure_anchored_wall
        r = fhwa_apparent_pressure_anchored_wall(
            H=10.0, anchor_depths=[2.5, 6.25], gamma=18.9, phi=33.0,
            surcharge=11.0, spacing=2.4, inclination_deg=15.0)
        assert "single_anchor_method" not in r
        assert r["subgrade_reaction_kN_per_m"] is not None
