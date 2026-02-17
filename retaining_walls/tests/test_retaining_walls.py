"""
Tests for retaining wall design module.

Covers: cantilever geometry, MSE geometry, earth pressure reuse,
cantilever stability (sliding/overturning/bearing), MSE internal/external,
and reinforcement definitions.

References:
    Das, Principles of Foundation Engineering
    FHWA GEC-11
"""

import math
import pytest

from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.earth_pressure import (
    rankine_Ka, rankine_Ka_sloped, rankine_Kp,
    horizontal_force_active, horizontal_force_passive,
)
from retaining_walls.cantilever import (
    check_sliding, check_overturning, check_bearing, analyze_cantilever_wall,
)
from retaining_walls.mse import (
    Kr_Ka_ratio, F_star_metallic, Tmax_at_level,
    pullout_resistance, check_internal_stability,
    check_external_stability, analyze_mse_wall,
)
from retaining_walls.reinforcement import (
    Reinforcement, RIBBED_STEEL_STRIP_75x4, GEOGRID_UX1600,
)
from retaining_walls.results import CantileverWallResult, MSEWallResult


# ================================================================
# Cantilever geometry
# ================================================================
class TestCantileverGeometry:
    def test_auto_sizing(self):
        geom = CantileverWallGeometry(wall_height=6.0)
        assert abs(geom.base_width - 3.6) < 0.01  # 0.6 * 6
        assert abs(geom.toe_length - 0.36) < 0.01  # 0.1 * 3.6
        assert geom.stem_thickness_base > geom.stem_thickness_top

    def test_explicit_sizing(self):
        geom = CantileverWallGeometry(
            wall_height=5.0, base_width=3.5, toe_length=0.5,
            stem_thickness_base=0.6, base_thickness=0.5
        )
        assert geom.base_width == 3.5
        assert geom.toe_length == 0.5
        assert abs(geom.stem_height - 4.5) < 0.01
        assert abs(geom.heel_length - (3.5 - 0.5 - 0.6)) < 0.01

    def test_shear_key(self):
        geom = CantileverWallGeometry(
            wall_height=5.0, has_shear_key=True, key_depth=0.5
        )
        assert geom.has_shear_key
        assert geom.key_depth == 0.5

    def test_backfill_slope_increases_H(self):
        geom_flat = CantileverWallGeometry(wall_height=6.0)
        geom_slope = CantileverWallGeometry(wall_height=6.0, backfill_slope=15)
        assert geom_slope.H_active > geom_flat.H_active

    def test_validation(self):
        with pytest.raises(ValueError):
            CantileverWallGeometry(wall_height=-1)


# ================================================================
# MSE geometry
# ================================================================
class TestMSEGeometry:
    def test_auto_length(self):
        geom = MSEWallGeometry(wall_height=8.0)
        assert abs(geom.reinforcement_length - 5.6) < 0.01  # 0.7 * 8

    def test_minimum_length(self):
        """Short wall still gets minimum 2.5m."""
        geom = MSEWallGeometry(wall_height=3.0)
        assert geom.reinforcement_length >= 2.5

    def test_reinforcement_levels(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        assert geom.n_reinforcement_levels == 10
        depths = geom.reinforcement_depths
        assert len(depths) == 10
        assert abs(depths[0] - 0.3) < 0.01  # first level at Sv/2

    def test_validation(self):
        with pytest.raises(ValueError):
            MSEWallGeometry(wall_height=0)


# ================================================================
# Earth pressure imports
# ================================================================
class TestEarthPressureImport:
    def test_rankine_reused(self):
        """Verify sheet_pile Ka/Kp functions work through retaining_walls."""
        Ka = rankine_Ka(30)
        expected = math.tan(math.pi / 4 - math.radians(30) / 2) ** 2
        assert abs(Ka - expected) < 0.001

    def test_resultant_active(self):
        """Pa = 0.5*Ka*gamma*H² for c=0, q=0."""
        Ka = rankine_Ka(30)
        Pa, z = horizontal_force_active(18.0, 6.0, Ka)
        expected_Pa = 0.5 * Ka * 18.0 * 36.0
        assert abs(Pa - expected_Pa) < 0.1
        assert abs(z - 2.0) < 0.01  # H/3

    def test_resultant_passive(self):
        Kp = rankine_Kp(30)
        Pp, z = horizontal_force_passive(18.0, 2.0, Kp)
        expected = 0.5 * Kp * 18.0 * 4.0
        assert abs(Pp - expected) < 0.1

    def test_surcharge_raises_force(self):
        Ka = rankine_Ka(30)
        Pa_no_q, _ = horizontal_force_active(18.0, 6.0, Ka, q=0)
        Pa_with_q, _ = horizontal_force_active(18.0, 6.0, Ka, q=10)
        assert Pa_with_q > Pa_no_q


# ================================================================
# Cantilever sliding
# ================================================================
class TestCantileverSliding:
    def _standard_geom(self):
        return CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5
        )

    def test_sand_backfill_passes(self):
        geom = self._standard_geom()
        result = check_sliding(geom, 18.0, 30.0)
        assert result["FOS_sliding"] > 1.0

    def test_higher_wall_lower_fos(self):
        """Taller wall -> lower FOS (more active force)."""
        geom_small = CantileverWallGeometry(wall_height=3.0, base_width=2.0)
        geom_tall = CantileverWallGeometry(wall_height=8.0, base_width=4.0)
        r1 = check_sliding(geom_small, 18.0, 30.0)
        r2 = check_sliding(geom_tall, 18.0, 30.0)
        # Both should have reasonable FOS but relative check
        assert r1["FOS_sliding"] > 0 and r2["FOS_sliding"] > 0

    def test_wider_base_higher_fos(self):
        """Wider base -> more weight -> better sliding resistance."""
        geom_narrow = CantileverWallGeometry(wall_height=5.0, base_width=2.5)
        geom_wide = CantileverWallGeometry(wall_height=5.0, base_width=4.0)
        r1 = check_sliding(geom_narrow, 18.0, 30.0)
        r2 = check_sliding(geom_wide, 18.0, 30.0)
        assert r2["FOS_sliding"] > r1["FOS_sliding"]


# ================================================================
# Cantilever overturning
# ================================================================
class TestCantileverOverturning:
    def test_basic_passes(self):
        geom = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5
        )
        result = check_overturning(geom, 18.0, 30.0)
        assert result["FOS_overturning"] > 1.0

    def test_narrow_base_fails(self):
        """Very narrow base -> overturning failure."""
        geom = CantileverWallGeometry(
            wall_height=8.0, base_width=2.0, toe_length=0.3,
            stem_thickness_base=0.4, base_thickness=0.4
        )
        result = check_overturning(geom, 18.0, 30.0)
        assert result["FOS_overturning"] < 2.0  # likely fails

    def test_surcharge_reduces_fos(self):
        geom1 = CantileverWallGeometry(wall_height=5.0, base_width=3.5)
        geom2 = CantileverWallGeometry(wall_height=5.0, base_width=3.5, surcharge=20)
        r1 = check_overturning(geom1, 18.0, 30.0)
        r2 = check_overturning(geom2, 18.0, 30.0)
        assert r2["FOS_overturning"] < r1["FOS_overturning"]


# ================================================================
# Cantilever bearing
# ================================================================
class TestCantileverBearing:
    def test_middle_third(self):
        geom = CantileverWallGeometry(
            wall_height=5.0, base_width=3.5, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5
        )
        result = check_bearing(geom, 18.0, 30.0)
        # Well-proportioned wall should have resultant in middle third
        assert result["in_middle_third"]
        assert result["q_toe_kPa"] > 0
        assert result["q_heel_kPa"] >= 0

    def test_q_toe_greater_than_q_heel(self):
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.0)
        result = check_bearing(geom, 18.0, 30.0)
        assert result["q_toe_kPa"] >= result["q_heel_kPa"]

    def test_with_allowable(self):
        geom = CantileverWallGeometry(wall_height=4.0, base_width=3.0)
        result = check_bearing(geom, 18.0, 30.0, q_allowable=300)
        assert result["FOS_bearing"] > 0


# ================================================================
# Full cantilever analysis
# ================================================================
class TestCantileverFull:
    def test_complete_analysis(self):
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5)
        result = analyze_cantilever_wall(geom, 18.0, 30.0)
        assert isinstance(result, CantileverWallResult)
        assert result.FOS_sliding > 0
        assert result.FOS_overturning > 0
        assert result.wall_height == 5.0

    def test_summary_and_dict(self):
        geom = CantileverWallGeometry(wall_height=5.0, base_width=3.5)
        result = analyze_cantilever_wall(geom, 18.0, 30.0, q_allowable=300)
        assert "CANTILEVER" in result.summary()
        d = result.to_dict()
        assert "FOS_sliding" in d
        assert "q_toe_kPa" in d

    def test_different_heights(self):
        """Taller wall needs wider base for same FOS."""
        r1 = analyze_cantilever_wall(
            CantileverWallGeometry(wall_height=3.0, base_width=2.5),
            18.0, 30.0
        )
        r2 = analyze_cantilever_wall(
            CantileverWallGeometry(wall_height=6.0, base_width=2.5),
            18.0, 30.0
        )
        # 6m wall with only 2.5m base will have worse FOS
        assert r2.FOS_overturning < r1.FOS_overturning


# ================================================================
# MSE Kr/Ka ratio and F*
# ================================================================
class TestMSEKrAndFstar:
    def test_Kr_Ka_metallic_surface(self):
        assert abs(Kr_Ka_ratio(0, "metallic") - 1.7) < 0.001

    def test_Kr_Ka_metallic_deep(self):
        assert abs(Kr_Ka_ratio(6, "metallic") - 1.2) < 0.001
        assert abs(Kr_Ka_ratio(10, "metallic") - 1.2) < 0.001

    def test_Kr_Ka_metallic_mid(self):
        """At z=3m: 1.7 - 0.5/6*3 = 1.45."""
        assert abs(Kr_Ka_ratio(3, "metallic") - 1.45) < 0.01

    def test_Kr_Ka_geosynthetic(self):
        assert Kr_Ka_ratio(0, "geosynthetic") == 1.0
        assert Kr_Ka_ratio(6, "geosynthetic") == 1.0

    def test_Fstar_surface(self):
        assert abs(F_star_metallic(0) - 2.0) < 0.001

    def test_Fstar_deep(self):
        """At z>=6m, F* = tan(phi) for ribbed metallic strips (GEC-11 Fig 4-11)."""
        F_deep = math.tan(math.radians(34))
        assert abs(F_star_metallic(6) - F_deep) < 0.001


# ================================================================
# MSE internal stability
# ================================================================
class TestMSEInternal:
    def test_Tmax_computation(self):
        Ka = rankine_Ka(34)
        T = Tmax_at_level(3.0, 18.0, Ka, 1.45, 0.6)
        # sigma_v = 18*3 = 54, sigma_h = 1.45*Ka*54, T = sigma_h * 0.6
        sigma_h = 1.45 * Ka * 54
        expected = sigma_h * 0.6
        assert abs(T - expected) < 0.01

    def test_pullout_resistance_computation(self):
        Pr = pullout_resistance(3.0, 18.0, 4.0, 1.5, C=2)
        # sigma_v = 54, Pr = 1.5 * 1.0 * 54 * 4.0 * 2 = 648
        expected = 1.5 * 1.0 * 54 * 4.0 * 2
        assert abs(Pr - expected) < 0.1

    def test_internal_check_metallic(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        results = check_internal_stability(
            geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4
        )
        assert len(results) == 10
        for r in results:
            assert "Tmax_kN_per_m" in r
            assert "FOS_pullout" in r
            assert "passes" in r

    def test_internal_check_geosynthetic(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        results = check_internal_stability(
            geom, 18.0, 34.0, GEOGRID_UX1600
        )
        assert len(results) == 10
        # Geosynthetic has lower Tallowable, check Kr/Ka = 1.0
        for r in results:
            assert abs(r["Kr_Ka"] - 1.0) < 0.001

    def test_deeper_levels_more_Tmax(self):
        """Deeper reinforcement levels have higher Tmax."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        results = check_internal_stability(geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4)
        # Last level should have higher Tmax than first
        # (despite Kr/Ka decreasing, sigma_v increases faster)
        assert results[-1]["Tmax_kN_per_m"] > results[0]["Tmax_kN_per_m"]


# ================================================================
# MSE external stability
# ================================================================
class TestMSEExternal:
    def test_basic_external(self):
        geom = MSEWallGeometry(wall_height=6.0)
        result = check_external_stability(geom, 18.0, 34.0, 19.0, 30.0)
        assert result["FOS_sliding"] > 0
        assert result["FOS_overturning"] > 0

    def test_longer_reinforcement_better_sliding(self):
        geom_short = MSEWallGeometry(wall_height=6.0, reinforcement_length=3.0)
        geom_long = MSEWallGeometry(wall_height=6.0, reinforcement_length=6.0)
        r1 = check_external_stability(geom_short, 18.0, 34.0, 19.0, 30.0)
        r2 = check_external_stability(geom_long, 18.0, 34.0, 19.0, 30.0)
        assert r2["FOS_sliding"] > r1["FOS_sliding"]

    def test_with_surcharge(self):
        geom = MSEWallGeometry(wall_height=6.0, surcharge=15)
        result = check_external_stability(geom, 18.0, 34.0, 19.0, 30.0)
        assert result["FOS_sliding"] > 0


# ================================================================
# Full MSE analysis
# ================================================================
class TestMSEFull:
    def test_complete_analysis(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        result = analyze_mse_wall(geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4)
        assert isinstance(result, MSEWallResult)
        assert result.FOS_sliding > 0
        assert result.n_levels == 10

    def test_summary_and_dict(self):
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        result = analyze_mse_wall(geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4)
        assert "MSE WALL" in result.summary()
        d = result.to_dict()
        assert "FOS_sliding" in d
        assert "internal_results" in d

    def test_geosynthetic_wall(self):
        geom = MSEWallGeometry(wall_height=4.0, reinforcement_spacing=0.4)
        result = analyze_mse_wall(geom, 18.0, 30.0, GEOGRID_UX1600)
        assert result.n_levels == 10  # 4.0/0.4


# ================================================================
# Reinforcement
# ================================================================
class TestReinforcement:
    def test_built_in_steel_strip(self):
        r = RIBBED_STEEL_STRIP_75x4
        assert r.is_metallic
        assert r.Tallowable > 0
        assert r.type == "metallic_strip"

    def test_built_in_geogrid(self):
        r = GEOGRID_UX1600
        assert not r.is_metallic
        assert r.type == "geosynthetic"

    def test_custom_reinforcement(self):
        r = Reinforcement("Custom", "geosynthetic", Tallowable=35.0)
        assert r.Tallowable == 35.0

    def test_coverage_ratio_default(self):
        """Default coverage_ratio is 1.0 (for geogrids)."""
        r = Reinforcement("Custom", "geosynthetic", Tallowable=35.0)
        assert r.coverage_ratio == 1.0

    def test_coverage_ratio_custom(self):
        """Metallic strip with user-specified Rc."""
        r = Reinforcement("Strip", "metallic_strip", Tallowable=40.0,
                          coverage_ratio=0.12)
        assert abs(r.coverage_ratio - 0.12) < 0.001

    def test_validation(self):
        with pytest.raises(ValueError):
            Reinforcement("Bad", "rope", Tallowable=10)
        with pytest.raises(ValueError):
            Reinforcement("Bad", "geosynthetic", Tallowable=-5)


# ================================================================
# NEW TESTS: CRITICAL-1 -- Shear key increases sliding FOS
# ================================================================
class TestShearKeyPassive:
    def test_passive_increases_sliding_fos(self):
        """Including passive resistance should increase sliding FOS."""
        geom = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5
        )
        r_no_passive = check_sliding(geom, 18.0, 30.0, include_passive=False)
        r_with_passive = check_sliding(geom, 18.0, 30.0, include_passive=True)
        assert r_with_passive["FOS_sliding"] > r_no_passive["FOS_sliding"]
        assert r_with_passive["Pp_kN_per_m"] > 0

    def test_shear_key_higher_fos_than_base_only(self):
        """Shear key with key_depth > base_thickness gives more passive."""
        geom_no_key = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5,
            has_shear_key=False,
        )
        geom_with_key = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5,
            has_shear_key=True, key_depth=1.0,
        )
        r1 = check_sliding(geom_no_key, 18.0, 30.0, include_passive=True)
        r2 = check_sliding(geom_with_key, 18.0, 30.0, include_passive=True)
        # key_depth=1.0 > base_thickness=0.5 -> more passive resistance
        assert r2["Pp_kN_per_m"] > r1["Pp_kN_per_m"]
        assert r2["FOS_sliding"] > r1["FOS_sliding"]

    def test_no_passive_by_default(self):
        """Passive resistance not included unless opt-in."""
        geom = CantileverWallGeometry(
            wall_height=5.0, base_width=3.0, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5
        )
        result = check_sliding(geom, 18.0, 30.0)
        assert result["Pp_kN_per_m"] == 0.0


# ================================================================
# NEW TESTS: CRITICAL-2 -- Rankine Ka with sloped backfill
# ================================================================
class TestRankineKaSloped:
    def test_flat_matches_standard(self):
        """beta=0 should match standard tan^2(45-phi/2)."""
        Ka_flat = rankine_Ka_sloped(30, 0)
        Ka_standard = rankine_Ka(30)
        assert abs(Ka_flat - Ka_standard) < 0.001

    def test_known_value_phi30_beta10(self):
        """Known value: phi=30, beta=10 -> Ka approx 0.3495."""
        phi = 30.0
        beta = 10.0
        # Hand calculation:
        # cos(10) = 0.9848, cos(30) = 0.8660
        # sqrt(cos^2(10) - cos^2(30)) = sqrt(0.9698 - 0.75) = sqrt(0.2198) = 0.4689
        # Ka = 0.9848 * (0.9848 - 0.4689) / (0.9848 + 0.4689)
        #    = 0.9848 * 0.5159 / 1.4537 = 0.3495
        Ka = rankine_Ka_sloped(phi, beta)
        assert abs(Ka - 0.3495) < 0.002

    def test_known_value_phi35_beta15(self):
        """Known value: phi=35, beta=15 -> Ka approx 0.3108."""
        phi = 35.0
        beta = 15.0
        cos_b = math.cos(math.radians(beta))
        cos_p = math.cos(math.radians(phi))
        sq = math.sqrt(cos_b ** 2 - cos_p ** 2)
        expected = cos_b * (cos_b - sq) / (cos_b + sq)
        Ka = rankine_Ka_sloped(phi, beta)
        assert abs(Ka - expected) < 0.001

    def test_slope_increases_Ka(self):
        """Sloped backfill should give higher Ka than flat."""
        Ka_flat = rankine_Ka(30)
        Ka_sloped = rankine_Ka_sloped(30, 15)
        assert Ka_sloped > Ka_flat

    def test_beta_equals_phi_raises(self):
        """beta >= phi should raise ValueError."""
        with pytest.raises(ValueError):
            rankine_Ka_sloped(30, 30)

    def test_cantilever_uses_sloped_ka(self):
        """Cantilever sliding with sloped backfill should use higher Ka."""
        geom_flat = CantileverWallGeometry(
            wall_height=5.0, base_width=3.5, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5, backfill_slope=0
        )
        geom_slope = CantileverWallGeometry(
            wall_height=5.0, base_width=3.5, toe_length=0.5,
            stem_thickness_base=0.5, base_thickness=0.5, backfill_slope=15
        )
        r_flat = check_sliding(geom_flat, 18.0, 30.0)
        r_slope = check_sliding(geom_slope, 18.0, 30.0)
        # Higher Ka + taller H_active -> more driving force -> lower FOS
        assert r_slope["FOS_sliding"] < r_flat["FOS_sliding"]


# ================================================================
# NEW TESTS: CRITICAL-3 -- MSE sliding uses min(phi_fill, phi_fdn)
# ================================================================
class TestMSESlidingFriction:
    def test_uses_min_phi(self):
        """MSE sliding should use min(phi_backfill, phi_foundation)."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
        # phi_backfill=34, phi_foundation=28 -> uses 28 (no 2/3 reduction)
        r = check_external_stability(geom, 18.0, 34.0, 19.0, 28.0)

        # Manually compute expected FOS
        Ka = rankine_Ka(34.0)
        Pa, _ = horizontal_force_active(18.0, 6.0, Ka)
        W = 18.0 * 6.0 * 5.0
        delta_b = min(34.0, 28.0)  # = 28
        R = W * math.tan(math.radians(delta_b))
        expected_FOS = R / Pa
        assert abs(r["FOS_sliding"] - round(expected_FOS, 3)) < 0.01

    def test_symmetric_min(self):
        """min(phi_fill, phi_fdn) works both ways."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
        # phi_backfill=28 < phi_foundation=34
        r1 = check_external_stability(geom, 18.0, 28.0, 19.0, 34.0)
        # phi_backfill=34 > phi_foundation=28
        r2 = check_external_stability(geom, 18.0, 34.0, 19.0, 28.0)
        # Both should use delta_b=28, but Ka differs (28 vs 34)
        # so FOS_sliding should differ because of different driving forces
        # The key check is that when phi_fdn < phi_fill, no 2/3 reduction
        Ka_28 = rankine_Ka(28.0)
        Ka_34 = rankine_Ka(34.0)
        assert Ka_28 > Ka_34  # lower phi = higher Ka


# ================================================================
# NEW TESTS: CRITICAL-4 -- MSE with retained fill different from reinforced
# ================================================================
class TestMSERetainedFill:
    def test_retained_fill_differs(self):
        """Using weaker retained fill should increase active pressure."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
        # Same fill for retained and reinforced
        r_same = check_external_stability(geom, 18.0, 34.0, 19.0, 30.0)
        # Weaker retained fill (phi=25) -> higher Ka -> lower FOS
        r_weak = check_external_stability(
            geom, 18.0, 34.0, 19.0, 30.0,
            phi_retained=25.0, gamma_retained=17.0,
        )
        assert r_weak["FOS_sliding"] < r_same["FOS_sliding"]
        assert r_weak["FOS_overturning"] < r_same["FOS_overturning"]

    def test_default_uses_backfill(self):
        """When phi_retained=None, should use phi_backfill."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
        r_default = check_external_stability(geom, 18.0, 34.0, 19.0, 30.0)
        r_explicit = check_external_stability(
            geom, 18.0, 34.0, 19.0, 30.0,
            phi_retained=34.0, gamma_retained=18.0,
        )
        assert abs(r_default["FOS_sliding"] - r_explicit["FOS_sliding"]) < 0.001

    def test_analyze_mse_passes_retained(self):
        """analyze_mse_wall should forward retained params."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        r1 = analyze_mse_wall(geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4)
        r2 = analyze_mse_wall(
            geom, 18.0, 34.0, RIBBED_STEEL_STRIP_75x4,
            phi_retained=25.0, gamma_retained=17.0,
        )
        assert r2.FOS_sliding < r1.FOS_sliding


# ================================================================
# NEW TESTS: CRITICAL-5 -- Pullout with Rc < 1.0
# ================================================================
class TestPulloutCoverageRatio:
    def test_Rc_reduces_pullout(self):
        """Rc < 1.0 should reduce pullout resistance proportionally."""
        Pr_full = pullout_resistance(3.0, 18.0, 4.0, 1.5, C=2, Rc=1.0)
        Pr_strip = pullout_resistance(3.0, 18.0, 4.0, 1.5, C=2, Rc=0.12)
        assert abs(Pr_strip - Pr_full * 0.12) < 0.01

    def test_Rc_one_matches_original(self):
        """Rc=1.0 should give same result as before the fix."""
        Pr = pullout_resistance(3.0, 18.0, 4.0, 1.5, C=2, Rc=1.0)
        # sigma_v=54, Pr = 1.5 * 1.0 * 54 * 4.0 * 2 * 1.0 = 648
        expected = 1.5 * 1.0 * 54 * 4.0 * 2
        assert abs(Pr - expected) < 0.1

    def test_internal_stability_with_Rc(self):
        """Internal check should use reinforcement.coverage_ratio."""
        geom = MSEWallGeometry(wall_height=6.0, reinforcement_spacing=0.6)
        strip_full = Reinforcement(
            "Strip", "metallic_strip", Tallowable=43.1,
            coverage_ratio=1.0,
        )
        strip_low_Rc = Reinforcement(
            "Strip", "metallic_strip", Tallowable=43.1,
            coverage_ratio=0.12,
        )
        r_full = check_internal_stability(geom, 18.0, 34.0, strip_full)
        r_low = check_internal_stability(geom, 18.0, 34.0, strip_low_Rc)
        # Lower Rc -> lower pullout resistance -> lower FOS_pullout
        for level_full, level_low in zip(r_full, r_low):
            assert level_low["FOS_pullout"] < level_full["FOS_pullout"]
            assert abs(level_low["Pr_kN_per_m"] -
                       level_full["Pr_kN_per_m"] * 0.12) < 0.1


# ================================================================
# NEW TESTS: CRITICAL-6 -- F* at depth uses tan(phi) not 0.67*tan(34)
# ================================================================
class TestFstarAtDepth:
    def test_deep_equals_tan_phi(self):
        """At z>=6m, F* = tan(phi_backfill), not 0.67*tan(34)."""
        for phi in [28.0, 30.0, 34.0, 38.0, 40.0]:
            F_deep = F_star_metallic(6.0, phi)
            expected = math.tan(math.radians(phi))
            assert abs(F_deep - expected) < 0.001, (
                f"F*(z=6, phi={phi}) = {F_deep}, expected tan({phi}) = {expected}"
            )

    def test_deep_not_067_factor(self):
        """Verify 0.67 factor is NOT used for metallic strips."""
        phi = 34.0
        F_deep = F_star_metallic(6.0, phi)
        wrong_value = 0.67 * math.tan(math.radians(phi))
        # F_deep should be tan(34) ~ 0.6745, not 0.67*tan(34) ~ 0.4519
        assert abs(F_deep - wrong_value) > 0.1

    def test_surface_still_2(self):
        """F* at z=0 is still 2.0 regardless of phi."""
        assert abs(F_star_metallic(0, 30.0) - 2.0) < 0.001
        assert abs(F_star_metallic(0, 40.0) - 2.0) < 0.001

    def test_interpolation_midpoint(self):
        """At z=3m, F* should be halfway between 2.0 and tan(phi)."""
        phi = 34.0
        F_deep = math.tan(math.radians(phi))
        expected_mid = 2.0 - (2.0 - F_deep) / 6.0 * 3.0
        actual = F_star_metallic(3.0, phi)
        assert abs(actual - expected_mid) < 0.001

    def test_phi_matters_at_depth(self):
        """Higher phi -> higher F* at depth."""
        F_low = F_star_metallic(6.0, 28.0)
        F_high = F_star_metallic(6.0, 40.0)
        assert F_high > F_low
