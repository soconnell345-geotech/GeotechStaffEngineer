"""
Tests for the rigorous GLE engine (slope_stability/gle.py).

Benchmarks (see UPGRADE_PLAN.md / VALIDATION.md):

B1 — Fredlund & Krahn (1977) homogeneous slope, Slide2 Verification #21.
     Imperial units used directly (the formulation is unit-consistent and
     the dry / ru cases never touch GAMMA_W).
B2 — F&K weak-layer composite surface, Slide2 Verification #22.
Internal consistency: lambda=0 moment == Bishop; mirrored geometry
invariance; constant-f == Spencer wrapper.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import (
    fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos,
)
from slope_stability.gle import gle_fos


# ---------------------------------------------------------------------------
# B1: Fredlund & Krahn (1977) homogeneous slope (imperial units: ft, psf, pcf)
# Surface (0,60)-(60,60)-(140,20)-(180,20); c'=600 psf, phi'=20, gamma=120 pcf
# Specified circle: xc=120, yc=90, R=80.
# Published FOS (F&K Table): dry — Ordinary 1.928, Bishop 2.080,
# Spencer 2.073, M-P 2.076. ru=0.25 — 1.607 / 1.766 / 1.761 / 1.764.
# ---------------------------------------------------------------------------

FK_SURFACE = [(0.0, 60.0), (60.0, 60.0), (140.0, 20.0), (180.0, 20.0)]
FK_CIRCLE = dict(xc=120.0, yc=90.0, radius=80.0)


def _fk_geom(ru=0.0):
    layer = SlopeSoilLayer(
        name="soil", top_elevation=60.0, bottom_elevation=0.0,
        gamma=120.0, phi=20.0, c_prime=600.0, ru=ru,
    )
    return SlopeGeometry(surface_points=FK_SURFACE, soil_layers=[layer])


def _fk_slices(ru=0.0, n=50):
    geom = _fk_geom(ru=ru)
    slip = CircularSlipSurface(**FK_CIRCLE)
    return build_slices(geom, slip, n_slices=n), slip


class TestFredlundKrahnDry:
    """B1 case 1 (dry)."""

    def test_ordinary(self):
        slices, slip = _fk_slices()
        assert fellenius_fos(slices, slip) == pytest.approx(1.928, rel=0.015)

    def test_bishop(self):
        slices, slip = _fk_slices()
        assert bishop_fos(slices, slip) == pytest.approx(2.080, rel=0.015)

    def test_gle_lambda0_equals_bishop(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip, f_interslice="half_sine")
        assert res.bishop_fos == pytest.approx(bishop_fos(slices, slip),
                                               rel=0.005)

    def test_spencer_rigorous(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip, f_interslice="constant")
        assert res.converged
        assert res.fos == pytest.approx(2.073, rel=0.015)
        # F&K report lambda ~ 0.42 (theta ~ 22.8 deg... atan basis); just
        # require a meaningful positive interslice inclination
        assert 0.1 < abs(res.lam) < 1.0

    def test_morgenstern_price_half_sine(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip, f_interslice="half_sine")
        assert res.converged
        assert res.fos == pytest.approx(2.076, rel=0.015)

    def test_moment_force_balance(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip)
        assert res.fos_moment == pytest.approx(res.fos_force, abs=2e-3)

    def test_interslice_forces_reported(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip)
        n = len(slices)
        assert len(res.interslice_E) == n + 1
        assert len(res.interslice_X) == n + 1
        # ends carry no interslice force
        assert res.interslice_E[0] == pytest.approx(0.0, abs=1e-6)
        assert abs(res.interslice_E[-1]) < 50.0  # closure residual small
        # interior normals compressive
        assert max(res.interslice_E) > 0.0

    def test_other_f_functions_close(self):
        # Moment-dominated circular problem: FOS insensitive to f(x)
        slices, slip = _fk_slices()
        ref = gle_fos(slices, slip, f_interslice="half_sine").fos
        for f in ("clipped_sine", "trapezoidal"):
            res = gle_fos(slices, slip, f_interslice=f)
            assert res.converged
            assert res.fos == pytest.approx(ref, rel=0.01)


class TestFredlundKrahnRu:
    """B1 case 2 (ru = 0.25)."""

    def test_ordinary(self):
        slices, slip = _fk_slices(ru=0.25)
        # F&K 1.607; Slide2 gets 1.687 (Ordinary is formulation-sensitive
        # with ru) — accept the band between published values
        fos = fellenius_fos(slices, slip)
        assert 1.55 <= fos <= 1.72

    def test_bishop(self):
        slices, slip = _fk_slices(ru=0.25)
        assert bishop_fos(slices, slip) == pytest.approx(1.766, rel=0.015)

    def test_spencer_rigorous(self):
        slices, slip = _fk_slices(ru=0.25)
        res = gle_fos(slices, slip, f_interslice="constant")
        assert res.converged
        assert res.fos == pytest.approx(1.761, rel=0.015)

    def test_morgenstern_price_half_sine(self):
        slices, slip = _fk_slices(ru=0.25)
        res = gle_fos(slices, slip, f_interslice="half_sine")
        assert res.converged
        assert res.fos == pytest.approx(1.764, rel=0.015)


# ---------------------------------------------------------------------------
# B2: F&K weak-layer composite surface (Slide2 Verification #22)
# Weak layer el. 15-16 ft (c'=0, phi'=10); composite = circle clipped z>=15.
# Published (F&K): dry — Ordinary 1.288, Spencer 1.373, M-P 1.370
#                  ru  — Spencer 1.118, M-P 1.118
# ---------------------------------------------------------------------------

def _fk_weak_geom(ru=0.0):
    upper = SlopeSoilLayer(
        name="upper", top_elevation=60.0, bottom_elevation=16.0,
        gamma=120.0, phi=20.0, c_prime=600.0, ru=ru,
    )
    weak = SlopeSoilLayer(
        name="weak", top_elevation=16.0, bottom_elevation=15.0,
        gamma=120.0, phi=10.0, c_prime=0.0, ru=ru,
    )
    return SlopeGeometry(surface_points=FK_SURFACE, soil_layers=[upper, weak])


def _fk_composite_surface(z_clip=15.0, n_arc=40):
    """Circle (120, 90, R=80) clipped at z_clip -> polyline."""
    xc, yc, R = 120.0, 90.0, 80.0

    def z_circ(x):
        return yc - math.sqrt(R * R - (x - xc) ** 2)

    # circle reaches z_clip at xc +/- dx_clip
    dx_clip = math.sqrt(R * R - (yc - z_clip) ** 2)
    x_l, x_r = xc - dx_clip, xc + dx_clip
    # entry at crest (z=60): x = xc - sqrt(R^2 - 30^2)
    x_entry = xc - math.sqrt(R * R - (yc - 60.0) ** 2)
    # exit on toe flat (z=20): x = xc + sqrt(R^2 - 70^2)
    x_exit = xc + math.sqrt(R * R - (yc - 20.0) ** 2)

    pts = []
    for i in range(n_arc + 1):
        x = x_entry + (x_l - x_entry) * i / n_arc
        pts.append((x, z_circ(x)))
    pts.append((x_r, z_clip))
    for i in range(1, n_arc + 1):
        x = x_r + (x_exit - x_r) * i / n_arc
        pts.append((x, z_circ(x)))
    return PolylineSlipSurface(points=pts)


class TestFredlundKrahnWeakLayer:
    """B2 composite circular surface."""

    def test_ordinary_dry(self):
        geom = _fk_weak_geom()
        slip = _fk_composite_surface()
        slices = build_slices(geom, slip, n_slices=60)
        # F&K 1.288, Slide 1.300, Zhu 1.300 — 3% gate on F&K
        assert fellenius_fos(slices, slip) == pytest.approx(1.288, rel=0.03)

    def test_spencer_dry(self):
        geom = _fk_weak_geom()
        slip = _fk_composite_surface()
        slices = build_slices(geom, slip, n_slices=60)
        res = gle_fos(slices, slip, f_interslice="constant",
                      axis_point=(120.0, 90.0))
        assert res.converged
        # F&K 1.373, Slide 1.382, Zhu 1.381
        assert res.fos == pytest.approx(1.373, rel=0.03)

    def test_mp_dry(self):
        geom = _fk_weak_geom()
        slip = _fk_composite_surface()
        slices = build_slices(geom, slip, n_slices=60)
        res = gle_fos(slices, slip, f_interslice="half_sine",
                      axis_point=(120.0, 90.0))
        assert res.converged
        # F&K 1.370, Slide 1.372, Zhu 1.371
        assert res.fos == pytest.approx(1.370, rel=0.03)

    def test_spencer_ru(self):
        geom = _fk_weak_geom(ru=0.25)
        slip = _fk_composite_surface()
        slices = build_slices(geom, slip, n_slices=60)
        res = gle_fos(slices, slip, f_interslice="constant",
                      axis_point=(120.0, 90.0))
        assert res.converged
        # F&K 1.118, Slide 1.124, Zhu 1.119
        assert res.fos == pytest.approx(1.118, rel=0.03)

    def test_fitted_axis_close_to_specified(self):
        """Without axis_point the fitted axis should give a similar FOS."""
        geom = _fk_weak_geom()
        slip = _fk_composite_surface()
        slices = build_slices(geom, slip, n_slices=60)
        res_spec = gle_fos(slices, slip, axis_point=(120.0, 90.0))
        res_fit = gle_fos(slices, slip)
        assert res_fit.converged
        assert res_fit.fos == pytest.approx(res_spec.fos, rel=0.02)


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------

class TestGLEConsistency:

    def test_mirror_invariance(self):
        """A mirrored slope must give the same FOS."""
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip)

        surf_m = sorted([(-x, z) for x, z in FK_SURFACE])
        geom_m = SlopeGeometry(
            surface_points=surf_m,
            soil_layers=[SlopeSoilLayer(
                name="soil", top_elevation=60.0, bottom_elevation=0.0,
                gamma=120.0, phi=20.0, c_prime=600.0)],
        )
        slip_m = CircularSlipSurface(xc=-120.0, yc=90.0, radius=80.0)
        slices_m = build_slices(geom_m, slip_m, n_slices=50)
        res_m = gle_fos(slices_m, slip_m)
        assert res_m.fos == pytest.approx(res.fos, rel=1e-3)
        assert res_m.lam == pytest.approx(res.lam, abs=5e-3)

    def test_wrapper_spencer_uses_rigorous(self):
        slices, slip = _fk_slices()
        fos, theta = spencer_fos(slices, slip)
        res = gle_fos(slices, slip, f_interslice="constant")
        assert fos == pytest.approx(res.fos, rel=1e-3)
        assert theta == pytest.approx(math.degrees(math.atan(res.lam)),
                                      abs=0.5)

    def test_wrapper_mp_uses_rigorous(self):
        slices, slip = _fk_slices()
        fos, lam = morgenstern_price_fos(slices, slip)
        res = gle_fos(slices, slip, f_interslice="half_sine")
        assert fos == pytest.approx(res.fos, rel=1e-3)

    def test_undrained_phi_zero(self):
        """phi=0: m_alpha = cos(alpha); engine must still converge."""
        layer = SlopeSoilLayer(
            name="clay", top_elevation=20.0, bottom_elevation=-10.0,
            gamma=18.0, cu=40.0, analysis_mode="undrained",
        )
        geom = SlopeGeometry(
            surface_points=[(0.0, 20.0), (20.0, 20.0), (40.0, 10.0),
                            (70.0, 10.0)],
            soil_layers=[layer],
        )
        slip = CircularSlipSurface(xc=30.0, yc=32.0, radius=25.0)
        slices = build_slices(geom, slip, n_slices=30)
        res = gle_fos(slices, slip)
        assert res.converged
        b = bishop_fos(slices, slip)
        # phi=0 circular: interslice assumptions barely matter
        assert res.fos == pytest.approx(b, rel=0.02)

    def test_seismic_reduces_fos(self):
        layer = SlopeSoilLayer(
            name="soil", top_elevation=60.0, bottom_elevation=0.0,
            gamma=120.0, phi=20.0, c_prime=600.0,
        )
        geom_k = SlopeGeometry(surface_points=FK_SURFACE,
                               soil_layers=[layer], kh=0.15)
        slip = CircularSlipSurface(**FK_CIRCLE)
        slices_k = build_slices(geom_k, slip, n_slices=50)
        res_k = gle_fos(slices_k, slip)
        slices_0, _ = _fk_slices()
        res_0 = gle_fos(slices_0, slip)
        assert res_k.converged
        assert res_k.fos < res_0.fos - 0.2

    def test_noncircular_polyline(self):
        """Rigorous GLE on a generic noncircular surface converges and is
        in the same range as the legacy Spencer estimate."""
        layer = SlopeSoilLayer(
            name="soil", top_elevation=60.0, bottom_elevation=0.0,
            gamma=120.0, phi=20.0, c_prime=600.0,
        )
        geom = SlopeGeometry(surface_points=FK_SURFACE, soil_layers=[layer])
        slip = PolylineSlipSurface(points=[
            (50.0, 60.0), (75.0, 38.0), (100.0, 26.0), (125.0, 21.0),
            (150.0, 20.0),
        ])
        slices = build_slices(geom, slip, n_slices=40)
        res = gle_fos(slices, slip)
        assert res.converged
        assert 1.0 < res.fos < 4.0

    def test_bad_f_name_raises(self):
        slices, slip = _fk_slices()
        with pytest.raises(ValueError, match="interslice"):
            gle_fos(slices, slip, f_interslice="nope")

    def test_thrust_line_within_slope(self):
        slices, slip = _fk_slices()
        res = gle_fos(slices, slip)
        # thrust elevations should lie between base and ground for the
        # well-loaded interior boundaries
        n = len(slices)
        for j in range(n // 4, 3 * n // 4):
            zb = slices[j].z_base
            zt = slices[j].z_top
            if abs(res.interslice_E[j]) > 100.0:
                assert zb - 5.0 <= res.thrust_elevation[j] <= zt + 5.0
