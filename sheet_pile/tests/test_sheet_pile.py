"""
Validation tests for the sheet_pile module.

Tests cover earth pressure coefficients, cantilever wall analysis,
anchored wall analysis, and basic engineering checks.

References:
    [1] USACE EM 1110-2-2504
    [2] Das, "Principles of Foundation Engineering", Chapter 9
    [3] USS Steel Sheet Piling Design Manual
"""

import math
import pytest

from sheet_pile.earth_pressure import (
    rankine_Ka, rankine_Kp, coulomb_Ka, coulomb_Kp, caquot_kerisel_Kp, K0,
    active_pressure, passive_pressure, tension_crack_depth,
)
from sheet_pile.cantilever import (
    WallSoilLayer, CantileverWallResult, analyze_cantilever,
)
from sheet_pile.anchored import (
    AnchoredWallResult, analyze_anchored,
)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Earth Pressure Coefficients
# ═══════════════════════════════════════════════════════════════════════

class TestEarthPressureCoefficients:
    """Verify Ka, Kp, K0 against known values."""

    def test_Ka_phi_30(self):
        """Ka for phi=30: tan²(45-15)=tan²(30)=1/3."""
        Ka = rankine_Ka(30)
        assert Ka == pytest.approx(1 / 3, rel=1e-4)

    def test_Kp_phi_30(self):
        """Kp for phi=30: tan²(45+15)=tan²(60)=3."""
        Kp = rankine_Kp(30)
        assert Kp == pytest.approx(3.0, rel=1e-4)

    def test_Ka_times_Kp(self):
        """For Rankine: Ka * Kp = 1 (exact)."""
        for phi in [15, 20, 25, 30, 35, 40]:
            Ka = rankine_Ka(phi)
            Kp = rankine_Kp(phi)
            assert Ka * Kp == pytest.approx(1.0, rel=1e-6)

    def test_K0_phi_30(self):
        """K0 for phi=30: 1-sin(30)=0.5."""
        assert K0(30) == pytest.approx(0.5, rel=1e-6)

    def test_Ka_phi_0(self):
        """Ka for phi=0: Ka=1 (undrained, K=1)."""
        assert rankine_Ka(0) == pytest.approx(1.0, rel=1e-6)

    def test_coulomb_matches_rankine_no_friction(self):
        """Coulomb Ka with delta=0 should match Rankine Ka."""
        for phi in [20, 25, 30, 35]:
            Ka_r = rankine_Ka(phi)
            Ka_c = coulomb_Ka(phi, delta_deg=0)
            assert Ka_c == pytest.approx(Ka_r, rel=0.01)

    def test_coulomb_Ka_with_wall_friction(self):
        """Coulomb Ka with wall friction should be < Rankine Ka."""
        Ka_r = rankine_Ka(30)
        Ka_c = coulomb_Ka(30, delta_deg=15)
        assert Ka_c < Ka_r

    def test_coulomb_Kp_no_friction_vertical_equals_rankine(self):
        """Coulomb Kp (delta=0, vertical wall) must reduce to Rankine Kp."""
        for phi in [20, 25, 30, 35]:
            assert coulomb_Kp(phi, delta_deg=0, alpha_deg=90) == pytest.approx(
                rankine_Kp(phi), rel=0.01)

    def test_coulomb_Kp_inclined_wall_numerator_sign(self):
        """Passive numerator must be sin^2(alpha-phi), not sin^2(alpha+phi) (SP-1)."""
        phi, alpha, delta, beta = 30.0, 70.0, 0.0, 0.0
        Kp = coulomb_Kp(phi, delta_deg=delta, alpha_deg=alpha, beta_deg=beta)
        a, p, d, b = (math.radians(alpha), math.radians(phi),
                      math.radians(delta), math.radians(beta))
        denom = (math.sin(a) ** 2 * math.sin(a + d)
                 * (1 - math.sqrt(math.sin(p + d) * math.sin(p + b)
                                  / (math.sin(a + d) * math.sin(a + b)))) ** 2)
        correct = math.sin(a - p) ** 2 / denom
        wrong = math.sin(a + p) ** 2 / denom
        assert Kp == pytest.approx(correct, rel=1e-9)
        assert abs(correct - wrong) > 1e-3  # the sign genuinely matters here

    def test_Ka_decreases_with_phi(self):
        """Ka should decrease as phi increases."""
        prev = 2.0
        for phi in [0, 10, 20, 30, 40]:
            Ka = rankine_Ka(phi)
            assert Ka <= prev
            prev = Ka

    def test_Kp_increases_with_phi(self):
        """Kp should increase as phi increases."""
        prev = 0
        for phi in [0, 10, 20, 30, 40]:
            Kp = rankine_Kp(phi)
            assert Kp >= prev
            prev = Kp


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Active and Passive Pressure
# ═══════════════════════════════════════════════════════════════════════

class TestLateralPressure:
    """Test active and passive pressure calculations."""

    def test_active_pressure_sand(self):
        """Active pressure in sand at 5m depth.
        sigma_a = Ka * gamma * z = (1/3)*18*5 = 30 kPa."""
        Ka = rankine_Ka(30)
        pa = active_pressure(18.0, 5.0, Ka)
        assert pa == pytest.approx(30.0, rel=0.01)

    def test_passive_pressure_sand(self):
        """Passive pressure in sand at 3m below excavation.
        sigma_p = Kp * gamma * z = 3*18*3 = 162 kPa."""
        Kp = rankine_Kp(30)
        pp = passive_pressure(18.0, 3.0, Kp)
        assert pp == pytest.approx(162.0, rel=0.01)

    def test_active_pressure_with_cohesion(self):
        """Active pressure with cohesion: sigma_a = Ka*gamma*z - 2c*sqrt(Ka).
        At shallow depth, can be negative (tension zone)."""
        Ka = rankine_Ka(20)
        pa = active_pressure(17.0, 1.0, Ka, c=20.0)
        # Ka(20) ≈ 0.490, pa = 0.490*17*1 - 2*20*sqrt(0.490)
        # = 8.33 - 27.99 = -19.66 (tension)
        assert pa < 0

    def test_tension_crack_depth(self):
        """Tension crack depth for cohesive soil.
        z_crack = 2c/(gamma*sqrt(Ka))."""
        Ka = rankine_Ka(0)  # Ka=1 for phi=0
        z = tension_crack_depth(50, 18.0, Ka)
        # z = 2*50/(18*1) = 5.56 m
        assert z == pytest.approx(100 / 18, rel=0.01)

    def test_passive_greater_than_active(self):
        """At the same depth, passive > active for phi > 0."""
        Ka = rankine_Ka(30)
        Kp = rankine_Kp(30)
        pa = active_pressure(18.0, 5.0, Ka)
        pp = passive_pressure(18.0, 5.0, Kp)
        assert pp > pa


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Cantilever Wall
# ═══════════════════════════════════════════════════════════════════════

class TestCantileverWall:
    """Test cantilever sheet pile wall analysis."""

    def test_basic_cantilever_sand(self):
        """Basic cantilever in uniform sand: verify reasonable embedment."""
        result = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
            FOS_passive=1.5,
        )
        assert result.embedment_depth > 0
        assert result.total_wall_length > 5.0
        # Typical D/H ratio for cantilever in sand ≈ 0.7-1.5
        D_over_H = result.embedment_depth / 5.0
        assert 0.5 < D_over_H < 3.0

    def test_deeper_excavation_more_embedment(self):
        """Deeper excavation requires more embedment."""
        r5 = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(25.0, 18.0, friction_angle=30)],
        )
        r7 = analyze_cantilever(
            excavation_depth=7.0,
            soil_layers=[WallSoilLayer(25.0, 18.0, friction_angle=30)],
        )
        assert r7.embedment_depth > r5.embedment_depth

    def test_stronger_soil_less_embedment(self):
        """Higher phi should require less embedment."""
        r25 = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=25)],
        )
        r35 = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=35)],
        )
        assert r35.embedment_depth < r25.embedment_depth

    def test_positive_max_moment(self):
        """Maximum moment should be positive."""
        result = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
        )
        assert result.max_moment > 0

    def test_summary_and_dict(self):
        """summary() and to_dict() should work."""
        result = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
        )
        assert "CANTILEVER" in result.summary()
        d = result.to_dict()
        assert "embedment_depth_m" in d
        assert d["embedment_depth_m"] > 0

    def test_single_safety_basis_default(self):
        """SP-3: the default safety basis is FOS_passive alone — the design
        embedment equals the converged embedment (embedment_increase = 1.0),
        with no hidden 1.2x increase stacked on top of FOS_passive = 1.5."""
        result = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
            FOS_passive=1.5,
        )
        assert result.embedment_increase == 1.0
        assert result.embedment_depth == pytest.approx(
            result.embedment_converged)
        d = result.to_dict()
        assert d["embedment_increase"] == 1.0
        assert d["embedment_converged_m"] == d["embedment_depth_m"]

    def test_embedment_increase_basis(self):
        """SP-3: the depth-increase basis (FOS = 1.0, +20%) is available as
        an explicit, documented parameter and reproduces the pre-v5.1
        D_design = D_converged * 1.2 arithmetic."""
        result = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
            FOS_passive=1.0,
            embedment_increase=1.2,
        )
        assert result.embedment_depth == pytest.approx(
            result.embedment_converged * 1.2)
        assert result.total_wall_length == pytest.approx(
            5.0 + result.embedment_depth)

    def test_embedment_increase_below_one_raises(self):
        """SP-3: embedment_increase < 1.0 is rejected."""
        with pytest.raises(ValueError):
            analyze_cantilever(
                excavation_depth=5.0,
                soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
                embedment_increase=0.8,
            )

    def test_two_safety_bases_comparable_embedment(self):
        """SP-3 sanity: the two recognized safety bases (FS=1.5 alone vs
        FS=1.0 + 1.3x depth increase) give embedments of the same order
        for a standard sand case (neither basis double-counts)."""
        layers = [WallSoilLayer(20.0, 18.0, friction_angle=30)]
        r_fos = analyze_cantilever(5.0, layers, FOS_passive=1.5)
        r_inc = analyze_cantilever(5.0, layers, FOS_passive=1.0,
                                   embedment_increase=1.3)
        ratio = r_fos.embedment_depth / r_inc.embedment_depth
        assert 0.6 < ratio < 1.7


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Anchored Wall
# ═══════════════════════════════════════════════════════════════════════

class TestAnchoredWall:
    """Test anchored sheet pile wall analysis."""

    def test_basic_anchored(self):
        """Basic anchored wall: verify positive anchor force and embedment."""
        result = analyze_anchored(
            excavation_depth=7.0,
            anchor_depth=1.5,
            soil_layers=[WallSoilLayer(25.0, 18.0, friction_angle=30)],
            FOS_passive=1.5,
        )
        assert result.embedment_depth > 0
        assert result.anchor_force > 0
        assert result.total_wall_length > 7.0

    def test_anchored_less_embedment_than_cantilever(self):
        """Anchored wall should need less embedment than cantilever."""
        r_cant = analyze_cantilever(
            excavation_depth=5.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
        )
        r_anch = analyze_anchored(
            excavation_depth=5.0,
            anchor_depth=1.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
        )
        assert r_anch.embedment_depth < r_cant.embedment_depth

    def test_anchored_summary_dict(self):
        """summary() and to_dict() should work."""
        result = analyze_anchored(
            excavation_depth=6.0,
            anchor_depth=1.0,
            soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
        )
        assert "ANCHORED" in result.summary()
        d = result.to_dict()
        assert "anchor_force_kN_per_m" in d

    def test_invalid_anchor_depth(self):
        """Anchor below excavation should raise ValueError."""
        with pytest.raises(ValueError, match="Anchor depth"):
            analyze_anchored(
                excavation_depth=5.0,
                anchor_depth=6.0,
                soil_layers=[WallSoilLayer(20.0, 18.0, friction_angle=30)],
            )


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Soil Layer Validation
# ═══════════════════════════════════════════════════════════════════════

class TestSoilLayerValidation:
    """Test WallSoilLayer input validation."""

    def test_valid_layer(self):
        """Valid soil layer should construct without error."""
        layer = WallSoilLayer(5.0, 18.0, friction_angle=30)
        assert layer.thickness == 5.0

    def test_invalid_thickness(self):
        with pytest.raises(ValueError, match="positive"):
            WallSoilLayer(-1, 18.0, friction_angle=30)

    def test_zero_strength(self):
        with pytest.raises(ValueError, match="c > 0 or phi > 0"):
            WallSoilLayer(5, 18.0, friction_angle=0, cohesion=0)

    def test_wall_friction_above_phi_rejected(self):
        with pytest.raises(ValueError, match="Wall friction"):
            WallSoilLayer(5.0, 18.0, friction_angle=30, wall_friction_deg=40)


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Coulomb Wall Friction at the Analysis Level (SP-2)
# ═══════════════════════════════════════════════════════════════════════

class TestCoulombWallFriction:
    """The 'coulomb' pressure method must respond to wall friction instead of
    being a silent duplicate of Rankine (regression for SP-2)."""

    def test_coulomb_smooth_wall_equals_rankine(self):
        """delta=0 (smooth wall): coulomb result matches rankine."""
        layers = [WallSoilLayer(20.0, 18.0, friction_angle=32.0)]
        r_rank = analyze_cantilever(4.0, layers, pressure_method="rankine")
        r_coul = analyze_cantilever(4.0, layers, pressure_method="coulomb")
        assert r_coul.embedment_depth == pytest.approx(
            r_rank.embedment_depth, rel=1e-6)

    def test_wall_friction_reduces_embedment(self):
        """Nonzero wall friction lowers active / raises passive -> less embedment."""
        smooth = [WallSoilLayer(20.0, 18.0, friction_angle=32.0)]
        rough = [WallSoilLayer(20.0, 18.0, friction_angle=32.0,
                               wall_friction_deg=20.0)]
        r_rank = analyze_cantilever(4.0, smooth, pressure_method="rankine")
        r_coul = analyze_cantilever(4.0, rough, pressure_method="coulomb")
        assert r_coul.embedment_depth < r_rank.embedment_depth


class TestCaquotKeriselKp:
    """Log-spiral passive coefficient: a smooth wall (delta=0) must return the
    Rankine Kp, not clamp to the lowest tabulated column R(0.40) (which would
    over-predict passive resistance and under-predict embedment)."""

    @pytest.mark.parametrize("phi", [25.0, 30.0, 35.0])
    def test_smooth_wall_delta0_equals_rankine(self, phi):
        assert caquot_kerisel_Kp(phi, delta_deg=0.0) == pytest.approx(
            rankine_Kp(phi), rel=1e-9)

    def test_v013_case_unchanged(self):
        """phi=30, delta/phi=0.5 stays at the Caltrans Ex 8-1 value 4.70, and the
        delta=phi base stays 6.30 (the sub-0.40 anchor must not perturb these)."""
        assert caquot_kerisel_Kp(30.0, delta_deg=15.0) == pytest.approx(4.70, abs=0.02)
        assert caquot_kerisel_Kp(30.0) == pytest.approx(6.30, abs=0.02)

    @pytest.mark.parametrize("phi", [25.0, 30.0, 35.0])
    def test_R_monotonic_over_delta_ratio(self, phi):
        """Kp' rises monotonically with delta/phi from the Rankine value (delta=0)
        to the full-friction base Kp0 (delta=phi)."""
        ratios = [i / 20.0 for i in range(21)]
        kps = [caquot_kerisel_Kp(phi, delta_deg=r * phi) for r in ratios]
        assert all(b >= a - 1e-12 for a, b in zip(kps, kps[1:]))
        assert kps[0] == pytest.approx(rankine_Kp(phi), rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Coulomb calc-steps display matches the analysis (CS-2)
# ═══════════════════════════════════════════════════════════════════════

class TestCalcStepsCoulomb:
    """calc_steps must display the Coulomb Ka the analysis used (with wall
    friction), not Rankine (regression for CS-2)."""

    def test_coulomb_calc_steps_uses_wall_friction(self):
        from sheet_pile.calc_steps import get_calc_steps
        from calc_package.data_model import CalcStep

        phi, delta = 32.0, 20.0
        layers = [WallSoilLayer(20.0, 18.0, friction_angle=phi,
                                wall_friction_deg=delta)]
        result = analyze_cantilever(4.0, layers, pressure_method="coulomb")
        analysis = {
            "wall_type": "cantilever",
            "excavation_depth": 4.0,
            "soil_layers": layers,
            "pressure_method": "coulomb",
            "FOS_passive": 1.5,
        }
        sections = get_calc_steps(result, analysis)
        ka = [it for sec in sections for it in sec.items
              if isinstance(it, CalcStep) and it.result_name == "Ka"]
        assert ka, "no Ka step found in calc steps"
        shown = float(ka[0].result_value)
        # Matches the analysis Coulomb Ka (with delta), not Rankine
        assert shown == pytest.approx(coulomb_Ka(phi, delta_deg=delta), abs=1e-4)
        assert shown != pytest.approx(rankine_Ka(phi), abs=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
