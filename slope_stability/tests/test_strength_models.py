"""
P6 tests: SHANSEP and Generalized Hoek-Brown per-layer strength models.

SHANSEP hand calc: su = S * OCR^m * sigma'_v with sigma'_v = gamma*h (dry)
at the slice base — checked slice by slice.

Hoek-Brown: for GSI=100, D=0 the GHB envelope reduces to the classic
Hoek-Brown sigma1 = sigma3 + sqrt(mi*sigci*sigma3 + sigci^2) (a=0.5, s=1);
the instantaneous (c, phi) must reproduce tau on the envelope at the
requested normal stress (Balmer consistency check).
"""

import math

import pytest

from slope_stability.geometry import (
    SlopeGeometry, SlopeSoilLayer, _hoek_brown_instantaneous,
)
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.analysis import analyze_slope


SURFACE = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
CIRCLE = dict(xc=30.0, yc=32.0, radius=26.0)


class TestSHANSEP:

    def _geom(self, S=0.25, m=0.8, ocr=1.0):
        layer = SlopeSoilLayer(
            name="clay", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=18.0, analysis_mode="undrained", cu=1.0,  # cu unused
            strength_model="shansep", shansep_S=S, shansep_m=m, ocr=ocr,
        )
        return SlopeGeometry(surface_points=SURFACE, soil_layers=[layer])

    def test_slice_su_hand_calc_dry(self):
        S, m, ocr = 0.25, 0.8, 2.0
        geom = self._geom(S=S, m=m, ocr=ocr)
        slip = CircularSlipSurface(**CIRCLE)
        slices = build_slices(geom, slip, 30)
        for s in slices:
            sigma_v = s.weight / s.width  # dry: total = effective
            su_hand = S * (ocr ** m) * sigma_v
            assert s.c == pytest.approx(su_hand, rel=1e-9)
            assert s.phi == 0.0

    def test_fos_monotonic_in_S_and_ocr(self):
        slip_kw = dict(method="bishop", n_slices=30, **CIRCLE)
        f1 = analyze_slope(self._geom(S=0.22, ocr=1.0), **slip_kw).FOS
        f2 = analyze_slope(self._geom(S=0.30, ocr=1.0), **slip_kw).FOS
        f3 = analyze_slope(self._geom(S=0.22, ocr=3.0), **slip_kw).FOS
        assert f2 > f1
        assert f3 > f1

    def test_ocr_scaling_exact(self):
        """FOS scales exactly by OCR^m for a fully SHANSEP slope (phi=0,
        FOS linear in su)."""
        m = 0.8
        f1 = analyze_slope(self._geom(S=0.25, m=m, ocr=1.0),
                           method="fellenius", n_slices=30, **CIRCLE).FOS
        f4 = analyze_slope(self._geom(S=0.25, m=m, ocr=4.0),
                           method="fellenius", n_slices=30, **CIRCLE).FOS
        assert f4 / f1 == pytest.approx(4.0 ** m, rel=1e-6)

    def test_gwt_reduces_su(self):
        """With water, sigma'_v drops and so does su/FOS."""
        geom_dry = self._geom()
        geom_wet = self._geom()
        geom_wet.gwt_points = [(0.0, 10.0), (70.0, 20.0)]
        f_dry = analyze_slope(geom_dry, method="fellenius", n_slices=30,
                              **CIRCLE).FOS
        f_wet = analyze_slope(geom_wet, method="fellenius", n_slices=30,
                              **CIRCLE).FOS
        assert f_wet < f_dry

    def test_validation(self):
        with pytest.raises(ValueError, match="OCR"):
            SlopeSoilLayer(name="x", top_elevation=1, bottom_elevation=0,
                           gamma=18, strength_model="shansep", ocr=0.5)
        with pytest.raises(ValueError, match="shansep_S"):
            SlopeSoilLayer(name="x", top_elevation=1, bottom_elevation=0,
                           gamma=18, strength_model="shansep", shansep_S=0)


class TestHoekBrown:

    @staticmethod
    def _tau_envelope(sigma_n, sigci, mi, gsi=100.0, D=0.0):
        """Independent Balmer evaluation: bisect sigma3 so the failure-
        plane normal stress equals sigma_n, return tau there."""
        mb = mi * math.exp((gsi - 100.0) / (28.0 - 14.0 * D))
        s = math.exp((gsi - 100.0) / (9.0 - 3.0 * D))
        a = 0.5 + (math.exp(-gsi / 15.0) - math.exp(-20.0 / 3.0)) / 6.0

        def sn_tau(s3):
            term = mb * s3 / sigci + s
            d1 = 1.0 + a * mb * term ** (a - 1.0)
            s1 = s3 + sigci * term ** a
            sn = s3 + (s1 - s3) / (d1 + 1.0)
            return sn, (sn - s3) * math.sqrt(d1)

        lo = -s * sigci / mb * 0.999
        hi = 2.0 * max(sigma_n, sigci)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            sn, tau = sn_tau(mid)
            if sn > sigma_n:
                hi = mid
            else:
                lo = mid
        return sn_tau(0.5 * (lo + hi))[1]

    def test_ghb_tangent_consistency_gsi100(self):
        """GSI=100, D=0 (a=0.5, s=1, mb=mi): the instantaneous (c, phi)
        must (1) reproduce tau on the envelope at sigma_n and (2) have
        tan(phi_i) equal to the envelope slope d tau/d sigma_n."""
        sigci, mi = 50000.0, 10.0  # 50 MPa in kPa
        sigma_n = 500.0
        c_i, phi_i = _hoek_brown_instantaneous(sigma_n, sigci, 100.0, mi, 0.0)
        assert c_i > 0
        assert 0 < phi_i < 90
        tau_tangent = c_i + sigma_n * math.tan(math.radians(phi_i))
        tau_env = self._tau_envelope(sigma_n, sigci, mi)
        assert tau_tangent == pytest.approx(tau_env, rel=1e-3)
        # slope check by central difference on the envelope
        d = 5.0
        slope = (self._tau_envelope(sigma_n + d, sigci, mi)
                 - self._tau_envelope(sigma_n - d, sigci, mi)) / (2 * d)
        assert math.tan(math.radians(phi_i)) == pytest.approx(slope,
                                                              rel=0.01)

    def test_tau_increases_with_sigma_n(self):
        taus = []
        for sn in (50.0, 200.0, 500.0, 1000.0):
            c_i, phi_i = _hoek_brown_instantaneous(sn, 20000.0, 60.0, 12.0,
                                                   0.0)
            taus.append(c_i + sn * math.tan(math.radians(phi_i)))
        assert all(t1 < t2 for t1, t2 in zip(taus, taus[1:]))

    def test_phi_decreases_with_stress(self):
        """Curved envelope: instantaneous phi drops as stress rises."""
        _, p1 = _hoek_brown_instantaneous(50.0, 20000.0, 60.0, 12.0, 0.0)
        _, p2 = _hoek_brown_instantaneous(2000.0, 20000.0, 60.0, 12.0, 0.0)
        assert p2 < p1

    def test_disturbance_weakens(self):
        c0, p0 = _hoek_brown_instantaneous(300.0, 20000.0, 60.0, 12.0, 0.0)
        c1, p1 = _hoek_brown_instantaneous(300.0, 20000.0, 60.0, 12.0, 1.0)
        t0 = c0 + 300.0 * math.tan(math.radians(p0))
        t1 = c1 + 300.0 * math.tan(math.radians(p1))
        assert t1 < t0

    def test_rock_slope_analysis(self):
        rock = SlopeSoilLayer(
            name="rock", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=26.0, strength_model="hoek_brown",
            hb_sigci=30000.0, hb_gsi=55.0, hb_mi=12.0, hb_D=0.0,
        )
        geom = SlopeGeometry(surface_points=SURFACE, soil_layers=[rock])
        res = analyze_slope(geom, method="bishop", n_slices=30, **CIRCLE)
        # decent rock mass: very stable on this gentle slope
        assert res.FOS > 3.0

        weak_rock = SlopeSoilLayer(
            name="rock", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=26.0, strength_model="hoek_brown",
            hb_sigci=1000.0, hb_gsi=20.0, hb_mi=5.0, hb_D=1.0,
        )
        geom_w = SlopeGeometry(surface_points=SURFACE,
                               soil_layers=[weak_rock])
        res_w = analyze_slope(geom_w, method="bishop", n_slices=30, **CIRCLE)
        assert res_w.FOS < res.FOS

    def test_validation(self):
        with pytest.raises(ValueError, match="hb_sigci"):
            SlopeSoilLayer(name="x", top_elevation=1, bottom_elevation=0,
                           gamma=20, strength_model="hoek_brown")
        with pytest.raises(ValueError, match="strength_model"):
            SlopeSoilLayer(name="x", top_elevation=1, bottom_elevation=0,
                           gamma=20, strength_model="nope")


class TestMixedProfiles:

    def test_shansep_layer_under_mc_fill(self):
        fill = SlopeSoilLayer(
            name="fill", top_elevation=20.0, bottom_elevation=8.0,
            gamma=20.0, phi=32.0, c_prime=2.0,
        )
        clay = SlopeSoilLayer(
            name="clay", top_elevation=8.0, bottom_elevation=-15.0,
            gamma=17.0, strength_model="shansep",
            shansep_S=0.22, shansep_m=0.8, ocr=1.5,
        )
        geom = SlopeGeometry(surface_points=SURFACE,
                             soil_layers=[fill, clay])
        res = analyze_slope(geom, method="gle", n_slices=40, **CIRCLE)
        assert 0.3 < res.FOS < 5.0
        slices = build_slices(geom, CircularSlipSurface(**CIRCLE), 40)
        # deep slices in the clay must be phi=0 with stress-dependent su
        deep = [s for s in slices if s.z_base < 7.5]
        assert deep
        assert all(s.phi == 0.0 for s in deep)
        assert all(s.c > 0 for s in deep)
