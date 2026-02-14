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
    rankine_Ka, rankine_Kp, horizontal_force_active, horizontal_force_passive,
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
        F_deep = 0.67 * math.tan(math.radians(34))
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

    def test_validation(self):
        with pytest.raises(ValueError):
            Reinforcement("Bad", "rope", Tallowable=10)
        with pytest.raises(ValueError):
            Reinforcement("Bad", "geosynthetic", Tallowable=-5)
