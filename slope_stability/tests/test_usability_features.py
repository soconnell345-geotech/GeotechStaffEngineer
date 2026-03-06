"""Tests for usability features: Morgenstern-Price, Ru, non-horizontal
boundaries, entry/exit visualization, click-to-inspect, multi-method comparison.
"""

import json
import math
import pytest
import numpy as np

from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, analyze_slope, search_critical_surface,
    morgenstern_price_fos, fellenius_fos, bishop_fos, spencer_fos,
    CircularSlipSurface, PolylineSlipSurface,
)
from slope_stability.slices import build_slices
from slope_stability.results import SlopeStabilityResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_geom(**kw):
    layer_keys = {'phi', 'c_prime', 'gamma', 'ru', 'cu', 'analysis_mode', 'gamma_sat'}
    layers_kw = {k: v for k, v in kw.items() if k in layer_keys}
    geom_keys = {'gwt_points', 'kh', 'tension_crack_depth', 'tension_crack_water_depth'}
    geom_kw = {k: v for k, v in kw.items() if k in geom_keys}
    layer = SlopeSoilLayer(
        'Fill', 10, -5,
        gamma=layers_kw.pop('gamma', 18),
        phi=layers_kw.pop('phi', 25),
        c_prime=layers_kw.pop('c_prime', 10),
        **layers_kw,
    )
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
        soil_layers=[layer],
        **geom_kw,
    )


SLIP = CircularSlipSurface(20, 15, 16)


# ===================================================================
# Morgenstern-Price
# ===================================================================

class TestMorgensternPrice:

    def test_constant_matches_spencer(self):
        slices = build_slices(_simple_geom(), SLIP, 30)
        fos_mp, _ = morgenstern_price_fos(slices, SLIP, f_interslice='constant')
        fos_sp, _ = spencer_fos(slices, SLIP)
        assert abs(fos_mp - fos_sp) < 0.01

    def test_all_functions_similar(self):
        slices = build_slices(_simple_geom(), SLIP, 30)
        fos_c, _ = morgenstern_price_fos(slices, SLIP, f_interslice='constant')
        fos_h, _ = morgenstern_price_fos(slices, SLIP, f_interslice='half_sine')
        fos_t, _ = morgenstern_price_fos(slices, SLIP, f_interslice='trapezoidal')
        avg = (fos_c + fos_h + fos_t) / 3
        for fos in [fos_c, fos_h, fos_t]:
            assert abs(fos - avg) / avg < 0.1

    def test_cohesionless(self):
        slices = build_slices(_simple_geom(c_prime=0, phi=35), SLIP, 30)
        fos, _ = morgenstern_price_fos(slices, SLIP)
        assert 0 < fos < 50

    def test_purely_cohesive(self):
        slices = build_slices(_simple_geom(c_prime=50, phi=0), SLIP, 30)
        fos, _ = morgenstern_price_fos(slices, SLIP)
        assert 0 < fos < 50

    def test_noncircular(self):
        poly = PolylineSlipSurface(points=[(5, 10), (12, 6), (20, 1), (30, 0), (40, 0)])
        slices = build_slices(_simple_geom(), poly, 25)
        fos_mp, _ = morgenstern_price_fos(slices, poly)
        fos_sp, _ = spencer_fos(slices, poly)
        assert 0 < fos_mp < 50
        assert abs(fos_mp - fos_sp) / fos_sp < 0.15

    def test_gwt_reduces_fos(self):
        slices_wet = build_slices(_simple_geom(gwt_points=[(0, 8), (50, -1)]), SLIP, 30)
        slices_dry = build_slices(_simple_geom(), SLIP, 30)
        fos_wet, _ = morgenstern_price_fos(slices_wet, SLIP)
        fos_dry, _ = morgenstern_price_fos(slices_dry, SLIP)
        assert fos_wet < fos_dry

    def test_seismic_reduces_fos(self):
        slices_seis = build_slices(_simple_geom(kh=0.15), SLIP, 30)
        slices_static = build_slices(_simple_geom(), SLIP, 30)
        fos_seis, _ = morgenstern_price_fos(slices_seis, SLIP)
        fos_static, _ = morgenstern_price_fos(slices_static, SLIP)
        assert fos_seis < fos_static

    def test_few_slices(self):
        slices = build_slices(_simple_geom(), SLIP, 5)
        fos, _ = morgenstern_price_fos(slices, SLIP)
        assert 0 < fos < 50

    def test_many_slices(self):
        slices = build_slices(_simple_geom(), SLIP, 80)
        fos, _ = morgenstern_price_fos(slices, SLIP)
        assert 0 < fos < 50

    def test_via_analyze_slope(self):
        r = analyze_slope(_simple_geom(), 20, 15, 16,
                          method='morgenstern_price', include_slice_data=True)
        assert r.method == 'Morgenstern-Price'
        assert r.FOS_morgenstern_price == r.FOS
        assert r.lambda_mp is not None
        assert r.slice_data is not None

    def test_noncircular_via_analyze(self):
        poly = PolylineSlipSurface(points=[(5, 10), (15, 5), (25, 0), (35, 0)])
        r = analyze_slope(_simple_geom(), slip_surface=poly, method='morgenstern_price')
        assert r.method == 'Morgenstern-Price'
        assert not r.is_circular

    def test_shallow_circle(self):
        slip = CircularSlipSurface(20, 30, 28)
        slices = build_slices(_simple_geom(), slip, 30)
        if len(slices) > 2:
            fos, _ = morgenstern_price_fos(slices, slip)
            assert fos > 0


# ===================================================================
# Pore Pressure Ratio (Ru)
# ===================================================================

class TestRu:

    def test_zero_no_effect(self):
        r1 = analyze_slope(_simple_geom(ru=0.0), 20, 15, 16)
        r2 = analyze_slope(_simple_geom(), 20, 15, 16)
        assert abs(r1.FOS - r2.FOS) < 0.001

    def test_reduces_fos(self):
        fos_dry = analyze_slope(_simple_geom(), 20, 15, 16).FOS
        for ru in [0.1, 0.2, 0.3, 0.5]:
            fos = analyze_slope(_simple_geom(ru=ru), 20, 15, 16).FOS
            assert fos < fos_dry, f"Ru={ru} should reduce FOS"

    def test_monotonic_decrease(self):
        fos_prev = float('inf')
        for ru in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            fos = analyze_slope(_simple_geom(ru=ru), 20, 15, 16).FOS
            assert fos <= fos_prev + 0.001
            fos_prev = fos

    def test_high_value_positive_fos(self):
        r = analyze_slope(_simple_geom(ru=0.5), 20, 15, 16, include_slice_data=True)
        assert r.FOS > 0
        assert any(s.pore_pressure > 0 for s in r.slice_data)

    def test_gwt_takes_precedence(self):
        """Where GWT provides pore pressure, Ru should not add on top."""
        geom_both = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[SlopeSoilLayer('Fill', 10, -5, 18, phi=25, c_prime=10, ru=0.3)],
            gwt_points=[(0, 8), (50, -1)],
        )
        geom_gwt = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[SlopeSoilLayer('Fill', 10, -5, 18, phi=25, c_prime=10)],
            gwt_points=[(0, 8), (50, -1)],
        )
        r_both = analyze_slope(geom_both, 20, 15, 16)
        r_gwt = analyze_slope(geom_gwt, 20, 15, 16)
        # With Ru as fallback, FOS <= GWT-only (Ru fills in dry slices)
        assert r_both.FOS <= r_gwt.FOS + 0.01

    def test_multi_layer_different_ru(self):
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[
                SlopeSoilLayer('Fill', 10, 2, 18, phi=30, c_prime=5, ru=0.1),
                SlopeSoilLayer('Clay', 2, -5, 19, phi=20, c_prime=15, ru=0.4),
            ],
        )
        r = analyze_slope(geom, 20, 15, 16, include_slice_data=True)
        assert r.FOS > 0

    def test_undrained_with_ru(self):
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[SlopeSoilLayer('Clay', 10, -5, 18, cu=50,
                                        analysis_mode='undrained', ru=0.2)],
        )
        r = analyze_slope(geom, 20, 15, 16)
        assert r.FOS > 0


# ===================================================================
# Non-horizontal layer boundaries
# ===================================================================

class TestNonHorizontalBoundary:

    def test_basic_analysis(self):
        layer = SlopeSoilLayer('Fill', 10, -5, 18, phi=25, c_prime=10,
                               bottom_boundary_points=[(0, -3), (25, -5), (50, -7)])
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[layer],
        )
        r = analyze_slope(geom, 20, 15, 16, include_slice_data=True)
        assert r.FOS > 0

    def test_steep_boundary(self):
        layer = SlopeSoilLayer('Fill', 10, -15, 18, phi=25, c_prime=10,
                               bottom_boundary_points=[(0, 0), (25, -10), (50, -15)])
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[layer],
        )
        r = analyze_slope(geom, 20, 15, 16)
        assert r.FOS > 0

    def test_with_ru(self):
        layer = SlopeSoilLayer('Fill', 10, -5, 18, phi=25, c_prime=10, ru=0.25,
                               bottom_boundary_points=[(0, -3), (25, -5), (50, -7)])
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[layer],
        )
        r = analyze_slope(geom, 20, 15, 16, include_slice_data=True)
        assert r.FOS > 0
        assert any(s.pore_pressure > 0 for s in r.slice_data)


# ===================================================================
# Multi-method comparison
# ===================================================================

class TestMultiMethodComparison:

    def test_compare_all_circular(self):
        r = analyze_slope(_simple_geom(), 20, 15, 16,
                          method='bishop', compare_methods=True)
        assert r.FOS_fellenius is not None
        assert r.FOS_bishop is not None
        assert r.FOS_spencer is not None
        assert r.theta_spencer is not None
        assert r.FOS_morgenstern_price is not None
        assert r.lambda_mp is not None

    def test_noncircular_no_bishop(self):
        poly = PolylineSlipSurface(points=[(5, 10), (15, 5), (25, 0), (35, 0)])
        r = analyze_slope(_simple_geom(), slip_surface=poly,
                          method='spencer', compare_methods=True)
        assert r.FOS_fellenius is not None
        assert r.FOS_bishop is None
        assert r.FOS_spencer is not None
        assert r.FOS_morgenstern_price is not None

    def test_to_dict_includes_all(self):
        r = analyze_slope(_simple_geom(), 20, 15, 16,
                          method='bishop', compare_methods=True)
        d = r.to_dict()
        assert 'FOS_spencer' in d
        assert 'FOS_morgenstern_price' in d
        assert 'lambda_mp' in d

    def test_summary_includes_comparison(self):
        r = analyze_slope(_simple_geom(), 20, 15, 16,
                          method='bishop', compare_methods=True)
        s = r.summary()
        assert 'Spencer' in s
        assert 'M-P' in s or 'lambda' in s

    def test_result_store_serializable(self):
        r = analyze_slope(_simple_geom(), 20, 15, 16, include_slice_data=True)
        store = {
            'FOS': r.FOS,
            'method': r.method,
            'slices': [s.to_dict() for s in r.slice_data],
        }
        json_str = json.dumps(store)
        parsed = json.loads(json_str)
        assert len(parsed['slices']) == len(r.slice_data)


# ===================================================================
# Combined features
# ===================================================================

class TestCombined:

    def test_ru_plus_mp(self):
        r = analyze_slope(_simple_geom(ru=0.3), 20, 15, 16,
                          method='morgenstern_price', include_slice_data=True,
                          compare_methods=True)
        assert r.FOS > 0
        assert r.FOS_morgenstern_price is not None

    def test_all_features(self):
        """Non-horiz boundary + Ru + seismic + M-P + comparison."""
        layer = SlopeSoilLayer('Fill', 10, -5, 18, phi=25, c_prime=10, ru=0.2,
                               bottom_boundary_points=[(0, -3), (25, -5), (50, -7)])
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=[layer],
            kh=0.1,
        )
        r = analyze_slope(geom, 20, 15, 16, method='morgenstern_price',
                          include_slice_data=True, compare_methods=True)
        assert r.FOS > 0
        assert r.method == 'Morgenstern-Price'
        assert r.FOS_fellenius is not None
        assert r.FOS_morgenstern_price is not None
        assert any(s.pore_pressure > 0 for s in r.slice_data)

    def test_10_layers(self):
        layers = [
            SlopeSoilLayer(f'L{i+1}', 10 - i * 1.5, 10 - (i + 1) * 1.5,
                           18, phi=25 + i, c_prime=5 + i * 2)
            for i in range(10)
        ]
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=layers,
        )
        r = analyze_slope(geom, 20, 15, 16)
        assert r.FOS > 0


# ===================================================================
# Tension Crack
# ===================================================================

class TestTensionCrack:

    def test_reduces_fos(self):
        """Tension crack should reduce FOS by removing resistance."""
        fos_dry = analyze_slope(_simple_geom(), 20, 15, 16).FOS
        fos_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16).FOS
        assert fos_crack < fos_dry

    def test_zero_depth_no_effect(self):
        """Zero crack depth = no change."""
        r1 = analyze_slope(_simple_geom(tension_crack_depth=0.0), 20, 15, 16)
        r2 = analyze_slope(_simple_geom(), 20, 15, 16)
        assert abs(r1.FOS - r2.FOS) < 0.001

    def test_water_further_reduces_fos(self):
        """Water-filled crack should reduce FOS more than dry crack."""
        fos_dry_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16).FOS
        fos_wet_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0,
                         tension_crack_water_depth=3.0), 20, 15, 16).FOS
        assert fos_wet_crack < fos_dry_crack

    def test_partial_water(self):
        """Partially filled crack: water < crack depth."""
        fos_half = analyze_slope(
            _simple_geom(tension_crack_depth=4.0,
                         tension_crack_water_depth=2.0), 20, 15, 16).FOS
        fos_full = analyze_slope(
            _simple_geom(tension_crack_depth=4.0,
                         tension_crack_water_depth=4.0), 20, 15, 16).FOS
        assert fos_full < fos_half

    def test_slices_marked_in_crack(self):
        """Slices in crack zone should have c=0, phi=0."""
        geom = _simple_geom(tension_crack_depth=3.0)
        r = analyze_slope(geom, 20, 15, 16, include_slice_data=True)
        cracked = [s for s in r.slice_data if s.in_tension_crack]
        assert len(cracked) > 0
        for s in cracked:
            assert s.c == 0.0
            assert s.phi == 0.0

    def test_positive_fos(self):
        """FOS should still be positive even with large crack."""
        r = analyze_slope(
            _simple_geom(tension_crack_depth=5.0,
                         tension_crack_water_depth=5.0), 20, 15, 16)
        assert r.FOS > 0

    def test_result_stores_crack_info(self):
        """Result should report crack depth and water depth."""
        r = analyze_slope(
            _simple_geom(tension_crack_depth=3.0,
                         tension_crack_water_depth=2.0), 20, 15, 16)
        assert r.tension_crack_depth == 3.0
        assert r.tension_crack_water_depth == 2.0

    def test_to_dict_includes_crack(self):
        """to_dict should include crack info."""
        r = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16)
        d = r.to_dict()
        assert d["tension_crack_depth_m"] == 3.0

    def test_summary_includes_crack(self):
        """summary() should mention tension crack."""
        r = analyze_slope(
            _simple_geom(tension_crack_depth=3.0,
                         tension_crack_water_depth=2.0), 20, 15, 16)
        s = r.summary()
        assert "Tension crack" in s
        assert "Crack water" in s

    def test_with_all_methods(self):
        """Crack should work with all 4 methods."""
        geom = _simple_geom(tension_crack_depth=3.0,
                            tension_crack_water_depth=2.0)
        for method in ['fellenius', 'bishop', 'spencer', 'morgenstern_price']:
            r = analyze_slope(geom, 20, 15, 16, method=method)
            assert r.FOS > 0, f"Method {method} failed with crack"

    def test_crack_with_seismic(self):
        """Tension crack + seismic loading together."""
        fos_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16).FOS
        fos_both = analyze_slope(
            _simple_geom(tension_crack_depth=3.0, kh=0.1), 20, 15, 16).FOS
        assert fos_both < fos_crack

    def test_crack_with_ru(self):
        """Tension crack + Ru together."""
        fos_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16).FOS
        fos_both = analyze_slope(
            _simple_geom(tension_crack_depth=3.0, ru=0.3), 20, 15, 16).FOS
        assert fos_both < fos_crack

    def test_crack_with_gwt(self):
        """Tension crack + GWT together."""
        fos_crack = analyze_slope(
            _simple_geom(tension_crack_depth=3.0), 20, 15, 16).FOS
        fos_both = analyze_slope(
            _simple_geom(tension_crack_depth=3.0,
                         gwt_points=[(0, 8), (50, -1)]), 20, 15, 16).FOS
        assert fos_both < fos_crack

    def test_water_clamped_to_crack_depth(self):
        """Water depth should be clamped to crack depth."""
        geom = _simple_geom(tension_crack_depth=3.0,
                            tension_crack_water_depth=10.0)
        assert geom.tension_crack_water_depth == 3.0

    def test_noncircular_with_crack(self):
        """Tension crack should work with noncircular surfaces."""
        poly = PolylineSlipSurface(points=[(5, 10), (15, 5), (25, 0), (35, 0)])
        geom = _simple_geom(tension_crack_depth=2.0)
        r = analyze_slope(geom, slip_surface=poly, method='spencer')
        assert r.FOS > 0

    def test_monotonic_with_depth(self):
        """Deeper crack should give lower FOS."""
        fos_prev = float('inf')
        for depth in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            fos = analyze_slope(
                _simple_geom(tension_crack_depth=depth), 20, 15, 16).FOS
            assert fos <= fos_prev + 0.01, \
                f"FOS should decrease with deeper crack: depth={depth}"
            fos_prev = fos

    def test_compare_methods_with_crack(self):
        """Multi-method comparison with tension crack."""
        r = analyze_slope(
            _simple_geom(tension_crack_depth=3.0,
                         tension_crack_water_depth=2.0),
            20, 15, 16, method='bishop', compare_methods=True)
        assert r.FOS_fellenius is not None
        assert r.FOS_bishop is not None
        assert r.FOS_morgenstern_price is not None

    def test_validation_negative_depth(self):
        """Negative crack depth should raise ValueError."""
        with pytest.raises(ValueError):
            _simple_geom(tension_crack_depth=-1.0)

    def test_validation_negative_water(self):
        """Negative water depth should raise ValueError."""
        with pytest.raises(ValueError):
            _simple_geom(tension_crack_water_depth=-1.0)
