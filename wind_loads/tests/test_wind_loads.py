"""
Tests for the wind_loads module (ASCE 7-22).

Verification values from ASCE 7-22 Table 26.10-1 and hand calculations.
"""

import math
import pytest

from wind_loads import (
    compute_Kz,
    compute_Kzt,
    compute_Ke,
    compute_velocity_pressure,
    get_Cf_freestanding_wall,
    analyze_freestanding_wall_wind,
    analyze_fence_wind,
    VelocityPressureResult,
    FreestandingWallWindResult,
)


# ============================================================================
# Kz — Velocity Pressure Exposure Coefficient
# ============================================================================

class TestComputeKz:
    """Verify Kz against ASCE 7-22 Table 26.10-1."""

    def test_exposure_B_4_6m(self):
        """Exp B at z=4.6m: Table value 0.57."""
        assert compute_Kz(4.6, "B") == pytest.approx(0.57, abs=0.01)

    def test_exposure_B_9_1m(self):
        """Exp B at z=9.1m: Table value 0.70."""
        assert compute_Kz(9.1, "B") == pytest.approx(0.70, abs=0.01)

    def test_exposure_B_30_5m(self):
        """Exp B at z=30.5m: Table value 0.99."""
        assert compute_Kz(30.5, "B") == pytest.approx(0.99, abs=0.01)

    def test_exposure_C_4_6m(self):
        """Exp C at z=4.6m: Table value 0.85."""
        assert compute_Kz(4.6, "C") == pytest.approx(0.85, abs=0.01)

    def test_exposure_C_9_1m(self):
        """Exp C at z=9.1m: Table value 0.98."""
        assert compute_Kz(9.1, "C") == pytest.approx(0.98, abs=0.01)

    def test_exposure_C_30_5m(self):
        """Exp C at z=30.5m: Table value 1.26."""
        assert compute_Kz(30.5, "C") == pytest.approx(1.26, abs=0.01)

    def test_exposure_D_4_6m(self):
        """Exp D at z=4.6m: Table value 1.03."""
        assert compute_Kz(4.6, "D") == pytest.approx(1.03, abs=0.01)

    def test_exposure_D_9_1m(self):
        """Exp D at z=9.1m: Table value 1.16."""
        assert compute_Kz(9.1, "D") == pytest.approx(1.16, abs=0.01)

    def test_exposure_D_30_5m(self):
        """Exp D at z=30.5m: Table value 1.43."""
        assert compute_Kz(30.5, "D") == pytest.approx(1.43, abs=0.01)

    def test_below_minimum_clamps_to_4_6m(self):
        """Heights below 4.6m should give same Kz as z=4.6m."""
        Kz_at_min = compute_Kz(4.6, "C")
        assert compute_Kz(2.0, "C") == pytest.approx(Kz_at_min, abs=0.001)
        assert compute_Kz(1.0, "C") == pytest.approx(Kz_at_min, abs=0.001)

    def test_invalid_z_raises(self):
        """z <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="z must be > 0"):
            compute_Kz(0, "C")
        with pytest.raises(ValueError, match="z must be > 0"):
            compute_Kz(-5, "C")

    def test_invalid_exposure_raises(self):
        """Invalid exposure category should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid exposure_category"):
            compute_Kz(10.0, "A")
        with pytest.raises(ValueError, match="Invalid exposure_category"):
            compute_Kz(10.0, "E")


# ============================================================================
# Kzt — Topographic Factor
# ============================================================================

class TestComputeKzt:
    """Verify Kzt for flat and topographic conditions."""

    def test_flat_terrain(self):
        """Flat terrain: Kzt = 1.0."""
        assert compute_Kzt("none") == 1.0

    def test_flat_alias(self):
        """'flat' is alias for 'none'."""
        assert compute_Kzt("flat") == 1.0

    def test_ridge_at_crest(self):
        """2D ridge at crest: Kzt > 1.0."""
        Kzt = compute_Kzt("2d_ridge", H_hill=30, Lh=100, x_distance=0, z_height=0)
        assert Kzt > 1.0
        # K1 = 1.04 * 0.3 = 0.312, K2=1.0, K3=1.0
        # Kzt = (1 + 0.312)^2 = 1.721
        assert Kzt == pytest.approx(1.721, abs=0.01)

    def test_escarpment(self):
        """2D escarpment: Kzt > 1.0 but less than ridge."""
        Kzt = compute_Kzt("2d_escarpment", H_hill=30, Lh=100, x_distance=0, z_height=0)
        assert Kzt > 1.0
        # K1 = 0.75 * 0.3 = 0.225
        # Kzt = (1 + 0.225)^2 = 1.501
        assert Kzt == pytest.approx(1.501, abs=0.01)

    def test_3d_hill(self):
        """3D axisymmetric hill: Kzt > 1.0."""
        Kzt = compute_Kzt("3d_hill", H_hill=30, Lh=100, x_distance=0, z_height=0)
        assert Kzt > 1.0

    def test_invalid_shape_raises(self):
        """Invalid hill_shape should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hill_shape"):
            compute_Kzt("mountain", H_hill=10, Lh=50)


# ============================================================================
# Ke — Ground Elevation Factor
# ============================================================================

class TestComputeKe:
    """Verify Ke against ASCE 7-22 Table 26.9-1."""

    def test_sea_level(self):
        """At sea level: Ke = 1.0."""
        assert compute_Ke(0) == pytest.approx(1.0, abs=0.001)

    def test_1000m(self):
        """At 1000m: Ke = e^(-0.0362) ≈ 0.964."""
        expected = math.exp(-0.0000362 * 1000)
        assert compute_Ke(1000) == pytest.approx(expected, abs=0.001)
        assert compute_Ke(1000) == pytest.approx(0.964, abs=0.001)

    def test_2000m(self):
        """At 2000m: Ke ≈ 0.930."""
        expected = math.exp(-0.0000362 * 2000)
        assert compute_Ke(2000) == pytest.approx(expected, abs=0.001)

    def test_5000m(self):
        """At 5000m: Ke ≈ 0.834 (significant reduction at high elevation)."""
        expected = math.exp(-0.0000362 * 5000)
        assert compute_Ke(5000) == pytest.approx(expected, abs=0.001)
        assert compute_Ke(5000) < 0.85

    def test_below_sea_level(self):
        """Below sea level: Ke > 1.0 (denser air)."""
        assert compute_Ke(-100) > 1.0


# ============================================================================
# Velocity Pressure qz
# ============================================================================

class TestVelocityPressure:
    """Verify velocity pressure computation."""

    def test_basic_calculation(self):
        """V=50 m/s, z=10m, Exp C, flat, sea level.

        Kz ≈ 0.98 (from Table)
        qz = 0.613 * 0.98 * 1.0 * 0.85 * 1.0 * 50^2 = 1278 Pa
        """
        r = compute_velocity_pressure(50, 10, "C")
        assert isinstance(r, VelocityPressureResult)
        Kz_expected = compute_Kz(10, "C")
        qz_expected = 0.613 * Kz_expected * 1.0 * 0.85 * 1.0 * 50**2
        assert r.qz_Pa == pytest.approx(qz_expected, rel=0.001)
        assert r.qz_kPa == pytest.approx(qz_expected / 1000, rel=0.001)

    def test_exposure_effect(self):
        """Higher exposure increases velocity pressure."""
        r_B = compute_velocity_pressure(40, 10, "B")
        r_C = compute_velocity_pressure(40, 10, "C")
        r_D = compute_velocity_pressure(40, 10, "D")
        assert r_B.qz_Pa < r_C.qz_Pa < r_D.qz_Pa

    def test_Ke_reduces_pressure(self):
        """Ke < 1.0 at elevation reduces velocity pressure."""
        Ke = compute_Ke(2000)
        r_sea = compute_velocity_pressure(50, 10, "C", Ke=1.0)
        r_elev = compute_velocity_pressure(50, 10, "C", Ke=Ke)
        assert r_elev.qz_Pa < r_sea.qz_Pa

    def test_Kzt_increases_pressure(self):
        """Kzt > 1.0 near topography increases velocity pressure."""
        Kzt = compute_Kzt("2d_ridge", H_hill=30, Lh=100)
        r_flat = compute_velocity_pressure(50, 10, "C", Kzt=1.0)
        r_topo = compute_velocity_pressure(50, 10, "C", Kzt=Kzt)
        assert r_topo.qz_Pa > r_flat.qz_Pa

    def test_result_fields(self):
        """All result fields are populated."""
        r = compute_velocity_pressure(45, 6, "B")
        assert r.V_m_s == 45
        assert r.z_m == 6
        assert r.exposure_category == "B"
        assert r.Kd == 0.85
        assert r.Ke == 1.0
        assert r.Kzt == 1.0

    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        r = compute_velocity_pressure(50, 10, "C")
        d = r.to_dict()
        assert "qz_Pa" in d
        assert "qz_kPa" in d
        assert d["exposure_category"] == "C"

    def test_summary_string(self):
        """summary() returns non-empty string."""
        r = compute_velocity_pressure(50, 10, "C")
        s = r.summary()
        assert "VELOCITY PRESSURE" in s
        assert "50.0 m/s" in s

    def test_invalid_V_raises(self):
        """V <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Wind speed V must be > 0"):
            compute_velocity_pressure(0, 10, "C")


# ============================================================================
# Cf — Net Force Coefficient
# ============================================================================

class TestCfFreestandingWall:
    """Verify Cf against ASCE 7-22 Figure 29.3-1."""

    def test_case_a_bs_1(self):
        """Case A (on ground), B/s=1: Cf=1.30."""
        assert get_Cf_freestanding_wall(1.0) == pytest.approx(1.30, abs=0.01)

    def test_case_a_bs_2(self):
        """Case A, B/s=2: Cf=1.40."""
        assert get_Cf_freestanding_wall(2.0) == pytest.approx(1.40, abs=0.01)

    def test_case_a_bs_10(self):
        """Case A, B/s=10: Cf=1.70."""
        assert get_Cf_freestanding_wall(10.0) == pytest.approx(1.70, abs=0.01)

    def test_case_a_bs_40_plus(self):
        """Case A, B/s>=40: Cf clamped at 1.75."""
        assert get_Cf_freestanding_wall(40.0) == pytest.approx(1.75, abs=0.01)
        assert get_Cf_freestanding_wall(100.0) == pytest.approx(1.75, abs=0.01)

    def test_interpolation(self):
        """Intermediate B/s interpolates linearly."""
        Cf = get_Cf_freestanding_wall(3.5)
        # Between B/s=2 (1.40) and B/s=5 (1.55): 1.40 + (1.55-1.40)*(3.5-2)/(5-2) = 1.475
        assert Cf == pytest.approx(1.475, abs=0.01)

    def test_case_c_elevated(self):
        """Case C (elevated wall, h/s>=1.0): higher Cf than Case A."""
        Cf_a = get_Cf_freestanding_wall(2.0, clearance_ratio=0.0)
        Cf_c = get_Cf_freestanding_wall(2.0, clearance_ratio=1.0)
        assert Cf_c > Cf_a
        assert Cf_c == pytest.approx(1.85, abs=0.01)

    def test_partial_clearance(self):
        """Intermediate clearance ratio interpolates between A and C."""
        Cf_a = get_Cf_freestanding_wall(2.0, clearance_ratio=0.0)
        Cf_c = get_Cf_freestanding_wall(2.0, clearance_ratio=1.0)
        Cf_half = get_Cf_freestanding_wall(2.0, clearance_ratio=0.5)
        expected = (Cf_a + Cf_c) / 2.0
        assert Cf_half == pytest.approx(expected, abs=0.01)

    def test_invalid_bs_raises(self):
        """B/s <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="B_over_s must be > 0"):
            get_Cf_freestanding_wall(0)


# ============================================================================
# Freestanding Wall Analysis
# ============================================================================

class TestFreestandingWallWind:
    """Full freestanding wall wind analysis verification."""

    def test_hand_calc_exp_c(self):
        """V=50 m/s, s=3m, B=20m, Exp C, flat, sea level.

        z_top = 3.0 m (clamped to 4.6m for Kz)
        Kz ≈ 0.85 (Exp C at 4.6m)
        qh = 0.613 * 0.85 * 1.0 * 0.85 * 1.0 * 50^2 = 1107 Pa
        B/s = 20/3 = 6.67
        Cf ≈ interp between B/s=5 (1.55) and B/s=10 (1.70) ≈ 1.60
        p = 1107 * 0.85 * 1.60 = 1505 Pa
        f = 1505 * 3 / 1000 = 4.51 kN/m
        F = 4.51 * 20 = 90.3 kN
        M = 4.51 * (0 + 3/2) = 6.77 kN*m/m
        """
        r = analyze_freestanding_wall_wind(
            V=50, wall_height=3, wall_length=20, exposure_category="C"
        )
        assert isinstance(r, FreestandingWallWindResult)
        # Verify velocity pressure
        assert r.velocity_pressure_Pa == pytest.approx(1107, rel=0.02)
        # Verify Cf range
        assert 1.55 < r.Cf < 1.70
        # Verify wind pressure
        assert r.wind_pressure_Pa == pytest.approx(1505, rel=0.05)
        # Verify force per unit length
        assert r.force_per_unit_length_kN_m == pytest.approx(4.51, rel=0.05)
        # Verify total force
        assert r.total_force_kN == pytest.approx(90.3, rel=0.05)

    def test_clearance_increases_moment(self):
        """Wall with clearance has higher overturning moment (longer arm)."""
        r_ground = analyze_freestanding_wall_wind(50, 3, 10, "C")
        r_elevated = analyze_freestanding_wall_wind(50, 3, 10, "C", clearance_height=2.0)
        assert r_elevated.overturning_moment_kNm_per_m > r_ground.overturning_moment_kNm_per_m

    def test_clearance_increases_velocity_pressure(self):
        """Elevated wall has higher qh because z_top is higher."""
        r_ground = analyze_freestanding_wall_wind(50, 3, 10, "C")
        # With 10m clearance, z_top = 13m vs 3m (clamped to 4.6m)
        r_elevated = analyze_freestanding_wall_wind(50, 3, 10, "C", clearance_height=10.0)
        assert r_elevated.velocity_pressure_Pa > r_ground.velocity_pressure_Pa

    def test_short_wall_exposure_b(self):
        """Short wall in Exp B: conservative but typical residential."""
        r = analyze_freestanding_wall_wind(40, 2, 5, "B")
        assert r.total_force_kN > 0
        assert r.wind_pressure_Pa > 0
        assert r.Kz == pytest.approx(compute_Kz(4.6, "B"), abs=0.001)

    def test_long_wall(self):
        """Very long wall (B/s > 40): Cf clamped at max."""
        r = analyze_freestanding_wall_wind(50, 2, 100, "C")
        assert r.B_over_s == 50.0
        assert r.Cf == pytest.approx(1.75, abs=0.01)

    def test_result_to_dict(self):
        """to_dict contains all expected keys."""
        r = analyze_freestanding_wall_wind(50, 3, 10, "C")
        d = r.to_dict()
        required_keys = [
            "velocity_pressure_Pa", "wind_pressure_Pa",
            "force_per_unit_length_kN_m", "total_force_kN",
            "overturning_moment_kNm_per_m",
            "Kz", "Kzt", "Kd", "Ke", "G", "Cf", "B_over_s",
            "V_m_s", "wall_height_m", "wall_length_m",
            "exposure_category", "solidity_ratio",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_result_summary(self):
        """summary() produces readable output."""
        r = analyze_freestanding_wall_wind(50, 3, 10, "C")
        s = r.summary()
        assert "FREESTANDING WALL" in s
        assert "50.0 m/s" in s

    def test_solidity_ratio_is_1(self):
        """Solid wall has solidity_ratio = 1.0."""
        r = analyze_freestanding_wall_wind(50, 3, 10, "C")
        assert r.solidity_ratio == 1.0

    def test_invalid_wall_height_raises(self):
        """wall_height <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="wall_height must be > 0"):
            analyze_freestanding_wall_wind(50, 0, 10, "C")

    def test_invalid_wall_length_raises(self):
        """wall_length <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="wall_length must be > 0"):
            analyze_freestanding_wall_wind(50, 3, 0, "C")


# ============================================================================
# Fence Wind Analysis
# ============================================================================

class TestFenceWind:
    """Fence wind analysis with porosity reduction."""

    def test_solid_fence_equals_wall(self):
        """Solid fence (solidity=1.0) should equal solid wall analysis."""
        r_wall = analyze_freestanding_wall_wind(50, 3, 10, "C")
        r_fence = analyze_fence_wind(50, 3, 10, 1.0, "C")
        assert r_fence.total_force_kN == pytest.approx(r_wall.total_force_kN, rel=0.001)
        assert r_fence.wind_pressure_Pa == pytest.approx(r_wall.wind_pressure_Pa, rel=0.001)

    def test_porosity_reduces_force(self):
        """Porous fence has lower force than solid wall."""
        r_solid = analyze_fence_wind(50, 3, 10, 1.0, "C")
        r_porous = analyze_fence_wind(50, 3, 10, 0.5, "C")
        assert r_porous.total_force_kN < r_solid.total_force_kN
        # Force ratio should be approximately equal to solidity ratio
        ratio = r_porous.total_force_kN / r_solid.total_force_kN
        assert ratio == pytest.approx(0.5, rel=0.01)

    def test_chain_link_fence(self):
        """Typical chain-link fence: solidity ≈ 0.4-0.5."""
        r = analyze_fence_wind(45, 1.8, 30, 0.45, "C")
        assert r.total_force_kN > 0
        assert r.solidity_ratio == 0.45
        # Chain-link should have significantly less force than solid
        r_solid = analyze_fence_wind(45, 1.8, 30, 1.0, "C")
        assert r.total_force_kN < 0.5 * r_solid.total_force_kN

    def test_privacy_fence(self):
        """Privacy fence with slats: solidity ≈ 0.85-0.95."""
        r = analyze_fence_wind(50, 2, 20, 0.90, "B")
        assert r.total_force_kN > 0
        assert r.Cf > 0  # Cf_effective = Cf_solid * 0.90

    def test_fence_with_clearance(self):
        """Elevated fence (e.g., on wall base) uses clearance correctly."""
        r = analyze_fence_wind(50, 2, 10, 0.5, "C", clearance_height=1.0)
        assert r.clearance_height_m == 1.0
        assert r.overturning_moment_kNm_per_m > 0

    def test_fence_summary_shows_fence(self):
        """summary() should say 'FENCE' for solidity < 1.0."""
        r = analyze_fence_wind(50, 2, 10, 0.5, "C")
        assert "FENCE" in r.summary()

    def test_invalid_solidity_raises(self):
        """solidity_ratio must be in (0, 1.0]."""
        with pytest.raises(ValueError, match="solidity_ratio"):
            analyze_fence_wind(50, 2, 10, 0.0, "C")
        with pytest.raises(ValueError, match="solidity_ratio"):
            analyze_fence_wind(50, 2, 10, 1.5, "C")

    def test_fence_to_dict_has_solidity(self):
        """to_dict includes solidity_ratio for fences."""
        r = analyze_fence_wind(50, 2, 10, 0.6, "C")
        d = r.to_dict()
        assert d["solidity_ratio"] == pytest.approx(0.6, abs=0.001)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_short_wall(self):
        """1m wall in Exp B: z clamped to 4.6m, should still work."""
        r = analyze_freestanding_wall_wind(30, 1, 5, "B")
        assert r.total_force_kN > 0
        # Kz should be for z=4.6m (clamped)
        assert r.Kz == pytest.approx(compute_Kz(4.6, "B"), abs=0.001)

    def test_tall_wall(self):
        """20m tall wall: Kz evaluated at actual height."""
        r = analyze_freestanding_wall_wind(50, 20, 50, "D")
        assert r.total_force_kN > 0
        assert r.Kz > compute_Kz(4.6, "D")  # Higher than minimum

    def test_high_wind(self):
        """Extreme wind V=80 m/s (category 5 hurricane)."""
        r = analyze_freestanding_wall_wind(80, 3, 10, "D")
        # qz proportional to V^2: (80/50)^2 = 2.56x higher than V=50
        r_50 = analyze_freestanding_wall_wind(50, 3, 10, "D")
        assert r.velocity_pressure_Pa == pytest.approx(
            r_50.velocity_pressure_Pa * (80/50)**2, rel=0.001
        )

    def test_high_elevation(self):
        """High elevation (Denver, ~1600m) reduces air density."""
        Ke = compute_Ke(1600)
        r = analyze_freestanding_wall_wind(50, 3, 10, "C", Ke=Ke)
        r_sea = analyze_freestanding_wall_wind(50, 3, 10, "C", Ke=1.0)
        assert r.velocity_pressure_Pa < r_sea.velocity_pressure_Pa
        # ~5.6% reduction at 1600m
        ratio = r.velocity_pressure_Pa / r_sea.velocity_pressure_Pa
        assert ratio == pytest.approx(Ke, rel=0.001)

    def test_overturning_moment_increases_with_clearance(self):
        """Moment arm increases with clearance even if qh and Cf also change."""
        # Compare moment per unit force: M/f = clearance + s/2
        r0 = analyze_freestanding_wall_wind(50, 3, 10, "C", clearance_height=0)
        arm0 = r0.overturning_moment_kNm_per_m / r0.force_per_unit_length_kN_m
        assert arm0 == pytest.approx(1.5, rel=0.01)  # s/2 = 3/2 = 1.5

        r5 = analyze_freestanding_wall_wind(50, 3, 10, "C", clearance_height=5)
        arm5 = r5.overturning_moment_kNm_per_m / r5.force_per_unit_length_kN_m
        assert arm5 == pytest.approx(6.5, rel=0.01)  # 5 + 3/2 = 6.5
