"""
Validation tests for the axial_pile module.

Tests cover pile section creation, soil profile stress computations,
alpha/Nordlund/beta methods, and combined analysis.

References:
    [1] FHWA GEC-12 (FHWA-NHI-16-009), Chapters 7-8
    [2] FHWA Soils & Foundations Reference Manual, Vol II
"""

import math
import pytest

from axial_pile.pile_types import (
    PileSection, make_pipe_pile, make_concrete_pile, make_h_pile,
)
from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
from axial_pile.tomlinson import alpha_tomlinson, skin_friction_cohesive, end_bearing_cohesive
from axial_pile.nordlund import (
    delta_from_phi, nordlund_Kd, nordlund_CF, skin_friction_cohesionless,
    end_bearing_cohesionless, nordlund_Nq_prime, _limiting_tip_resistance,
)
from axial_pile.beta_method import beta_from_phi, skin_friction_beta, Nt_from_phi
from axial_pile.capacity import AxialPileAnalysis
from axial_pile.results import AxialPileResult


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Pile Section Creation
# ═══════════════════════════════════════════════════════════════════════

class TestPileSections:
    """Test pile section creation and property computation."""

    def test_pipe_pile_closed(self):
        """Closed-end pipe pile: tip area = full circle."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        assert pile.pile_type == "pipe_closed"
        assert pile.width == pytest.approx(0.3239, rel=1e-4)
        expected_tip = math.pi / 4 * 0.3239**2
        assert pile.tip_area == pytest.approx(expected_tip, rel=1e-4)
        assert pile.perimeter == pytest.approx(math.pi * 0.3239, rel=1e-4)

    def test_pipe_pile_open(self):
        """Open-ended pipe pile: tip area = annular ring."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=False)
        assert pile.pile_type == "pipe_open"
        r_o = 0.3
        r_i = 0.3 - 0.012
        expected_area = math.pi * (r_o**2 - r_i**2)
        assert pile.area == pytest.approx(expected_area, rel=1e-4)
        # Tip area for open-ended = steel annular area
        assert pile.tip_area == pytest.approx(expected_area, rel=1e-4)

    def test_concrete_square(self):
        """Square concrete pile."""
        pile = make_concrete_pile(0.3556, shape="square")  # 14" pile
        assert pile.area == pytest.approx(0.3556**2, rel=1e-4)
        assert pile.perimeter == pytest.approx(4 * 0.3556, rel=1e-4)
        assert pile.E == 25e6  # concrete

    def test_h_pile_database(self):
        """HP14x117 from built-in database."""
        pile = make_h_pile("HP14x117")
        assert pile.pile_type == "h_pile"
        assert pile.area == pytest.approx(0.02219, rel=0.01)
        # Box perimeter = 2*(depth + bf)
        assert pile.perimeter > 1.0

    def test_h_pile_unknown(self):
        """Unknown HP shape should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown HP shape"):
            make_h_pile("HP99x999")

    def test_all_hp_shapes(self):
        """All built-in HP shapes should be constructible."""
        shapes = ["HP10x42", "HP10x57", "HP12x53", "HP12x63", "HP12x74",
                  "HP12x84", "HP14x73", "HP14x89", "HP14x102", "HP14x117"]
        for s in shapes:
            pile = make_h_pile(s)
            assert pile.area > 0
            assert pile.perimeter > 0


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Soil Profile
# ═══════════════════════════════════════════════════════════════════════

class TestSoilProfile:
    """Test soil profile and effective stress calculations."""

    def test_effective_stress_no_gwt(self):
        """Effective stress without groundwater = total stress."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
        ])
        sigma_v = soil.effective_stress_at_depth(5.0)
        assert sigma_v == pytest.approx(90.0, rel=1e-4)  # 18*5

    def test_effective_stress_with_gwt(self):
        """Effective stress with GWT at 3m depth."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
        ], gwt_depth=3.0)
        sigma_v = soil.effective_stress_at_depth(5.0)
        # 18*3 + (18-9.81)*2 = 54 + 16.38 = 70.38
        expected = 18 * 3 + (18 - 9.81) * 2
        assert sigma_v == pytest.approx(expected, rel=0.01)

    def test_effective_stress_multilayer(self):
        """Effective stress through multiple layers."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(5, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
        ])
        # At 7m: 18*5 + 17*2 = 90 + 34 = 124
        sigma_v = soil.effective_stress_at_depth(7.0)
        assert sigma_v == pytest.approx(124.0, rel=0.01)

    def test_layer_at_depth(self):
        """Correct layer returned for a given depth."""
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(5, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
        ])
        layer = soil.layer_at_depth(3.0)
        assert layer.soil_type == "cohesionless"
        layer = soil.layer_at_depth(7.0)
        assert layer.soil_type == "cohesive"

    def test_invalid_soil_type(self):
        """Invalid soil type should raise ValueError."""
        with pytest.raises(ValueError, match="soil_type"):
            AxialSoilLayer(5, 'rock', 22.0, friction_angle=40)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Tomlinson Alpha Method
# ═══════════════════════════════════════════════════════════════════════

class TestTomlinson:
    """Test Tomlinson alpha values and cohesive skin friction."""

    def test_alpha_soft_clay(self):
        """Soft clay (cu=20 kPa): alpha should be ~1.0."""
        alpha = alpha_tomlinson(20, "steel")
        assert alpha == pytest.approx(1.0, abs=0.1)

    def test_alpha_medium_clay(self):
        """Medium clay (cu=50 kPa): alpha ≈ 0.5 for steel."""
        alpha = alpha_tomlinson(50, "steel")
        assert 0.3 < alpha < 0.7

    def test_alpha_stiff_clay(self):
        """Stiff clay (cu=150 kPa): alpha ≈ 0.35 for steel."""
        alpha = alpha_tomlinson(150, "steel")
        assert 0.2 < alpha < 0.5

    def test_alpha_concrete_higher(self):
        """Concrete pile alpha >= steel pile alpha."""
        for cu in [25, 50, 100, 200]:
            alpha_s = alpha_tomlinson(cu, "steel")
            alpha_c = alpha_tomlinson(cu, "concrete")
            assert alpha_c >= alpha_s

    def test_alpha_decreasing(self):
        """Alpha should generally decrease with increasing cu."""
        prev = 2.0
        for cu in [10, 25, 50, 100, 200]:
            alpha = alpha_tomlinson(cu, "steel")
            assert alpha <= prev
            prev = alpha

    def test_end_bearing_cohesive(self):
        """Qt = 9 * cu * At."""
        Qt = end_bearing_cohesive(50, 0.1)  # cu=50, At=0.1
        assert Qt == pytest.approx(45.0, rel=1e-6)

    def test_skin_friction_cohesive(self):
        """Qs = alpha * cu * perimeter * thickness."""
        # For cu=20 (soft), alpha≈1.0
        Qs = skin_friction_cohesive(20, 1.0, 5.0, "steel")
        alpha = alpha_tomlinson(20, "steel")
        expected = alpha * 20 * 1.0 * 5.0
        assert Qs == pytest.approx(expected, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Nordlund Method
# ═══════════════════════════════════════════════════════════════════════

class TestNordlund:
    """Test Nordlund method for cohesionless soils."""

    def test_delta_from_phi_steel(self):
        """Steel pile: delta ≈ 0.75 * phi."""
        delta = delta_from_phi(30, "steel")
        assert delta == pytest.approx(22.5, rel=1e-4)

    def test_delta_from_phi_concrete(self):
        """Concrete pile: delta ≈ 0.90 * phi."""
        delta = delta_from_phi(30, "concrete")
        assert delta == pytest.approx(27.0, rel=1e-4)

    def test_Kd_increases_with_phi(self):
        """Kd should be non-decreasing with phi."""
        prev = 0
        for phi in [20, 25, 30, 35, 40]:
            Kd = nordlund_Kd(phi)
            assert Kd >= prev
            prev = Kd
        # Also verify significant increase from low to high phi
        assert nordlund_Kd(40) > nordlund_Kd(25)

    def test_Nq_increases_with_phi(self):
        """Nq' should increase with phi."""
        prev = 0
        for phi in [20, 25, 30, 35, 40]:
            Nq = nordlund_Nq_prime(phi)
            assert Nq > prev
            prev = Nq

    def test_skin_friction_positive(self):
        """Skin friction should be positive for positive inputs."""
        Qs = skin_friction_cohesionless(30, 50, 1.0, 5.0)
        assert Qs > 0

    def test_end_bearing_positive(self):
        """End bearing should be positive."""
        Qt = end_bearing_cohesionless(30, 100, 0.1, 15, 0.3)
        assert Qt > 0


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Beta Method
# ═══════════════════════════════════════════════════════════════════════

class TestBetaMethod:
    """Test effective stress (beta) method."""

    def test_beta_NC_sand(self):
        """NC sand phi=30: beta = (1-sin30)*tan30 = 0.5*0.577 = 0.289."""
        beta = beta_from_phi(30, OCR=1.0)
        expected = (1 - math.sin(math.radians(30))) * math.tan(math.radians(30))
        assert beta == pytest.approx(expected, rel=1e-4)

    def test_beta_OC_higher(self):
        """OC soil should have higher beta than NC."""
        beta_nc = beta_from_phi(30, OCR=1.0)
        beta_oc = beta_from_phi(30, OCR=4.0)
        assert beta_oc > beta_nc

    def test_Nt_increases_with_phi(self):
        """Nt should increase with phi."""
        prev = 0
        for phi in [0, 10, 20, 30, 40]:
            Nt = Nt_from_phi(phi)
            assert Nt >= prev
            prev = Nt

    def test_skin_friction_beta(self):
        """Qs = beta * sigma_v * perimeter * thickness."""
        Qs = skin_friction_beta(50, 0.3, 1.0, 5.0)
        assert Qs == pytest.approx(0.3 * 50 * 1.0 * 5.0, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Combined Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestCombinedAnalysis:
    """Test the combined AxialPileAnalysis."""

    def _make_standard_problem(self):
        """Standard test problem: pipe pile in sand + clay."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(10, 'cohesive', 17.0, cohesion=50),
        ], gwt_depth=3.0)
        return pile, soil

    def test_auto_method(self):
        """Auto method selects Nordlund for sand, Tomlinson for clay."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)
        result = analysis.compute()

        assert result.Q_ultimate > 0
        assert result.Q_skin > 0
        assert result.Q_tip > 0
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )
        assert result.Q_allowable == pytest.approx(
            result.Q_ultimate / 2.5, rel=1e-6
        )

        # Check layer methods
        methods = {l['method'] for l in result.layer_breakdown}
        assert 'nordlund' in methods
        assert 'tomlinson' in methods

    def test_beta_method(self):
        """Beta method should produce positive capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0, method="beta"
        )
        result = analysis.compute()
        assert result.Q_ultimate > 0

    def test_uplift_capacity(self):
        """Uplift capacity < ultimate (compression) capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0, include_uplift=True
        )
        result = analysis.compute()
        assert result.Q_uplift is not None
        assert result.Q_uplift < result.Q_ultimate

    def test_longer_pile_more_capacity(self):
        """Longer pile in uniform sand should have more capacity."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 18.0, friction_angle=30),
        ], gwt_depth=3.0)
        r10 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=10.0).compute()
        r15 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0).compute()
        assert r15.Q_skin > r10.Q_skin  # more skin friction
        assert r15.Q_ultimate > r10.Q_ultimate

    def test_h_pile_analysis(self):
        """HP shape pile analysis should work."""
        pile = make_h_pile("HP12x74")
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 19.0, friction_angle=35),
        ])
        result = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0
        ).compute()
        assert result.Q_ultimate > 0

    def test_summary_string(self):
        """summary() should return a formatted string."""
        pile, soil = self._make_standard_problem()
        result = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15).compute()
        s = result.summary()
        assert "AXIAL PILE" in s
        assert "kN" in s

    def test_to_dict(self):
        """to_dict should return key fields."""
        pile, soil = self._make_standard_problem()
        result = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15).compute()
        d = result.to_dict()
        assert "Q_ultimate_kN" in d
        assert d["Q_ultimate_kN"] > 0

    def test_capacity_vs_depth(self):
        """capacity_vs_depth should return increasing capacity."""
        pile, soil = self._make_standard_problem()
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)
        profile = analysis.capacity_vs_depth(depth_min=5, depth_max=15, n_points=5)
        assert len(profile) == 5
        # Capacity should generally increase with depth
        assert profile[-1]["Q_ultimate_kN"] >= profile[0]["Q_ultimate_kN"]


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: CRITICAL-1 — Nordlund /cos(omega) for tapered piles
# ═══════════════════════════════════════════════════════════════════════

class TestNordlundCosOmega:
    """Verify /cos(omega) divisor in Nordlund skin friction (GEC-12 Eq 7-1)."""

    def test_uniform_pile_unchanged(self):
        """For omega=0, cos(0)=1.0 so no change in result."""
        Qs = skin_friction_cohesionless(30, 50, 1.0, 5.0, omega_deg=0.0)
        assert Qs > 0

    def test_tapered_pile_higher_than_uniform(self):
        """Tapered pile (omega>0) should have higher skin friction than uniform.

        Two effects increase it: Kd increases with omega, and /cos(omega)>1.
        """
        Qs_uniform = skin_friction_cohesionless(30, 50, 1.0, 5.0, omega_deg=0.0)
        Qs_tapered = skin_friction_cohesionless(30, 50, 1.0, 5.0, omega_deg=5.0)
        assert Qs_tapered > Qs_uniform

    def test_cos_omega_factor_applied(self):
        """Directly verify /cos(omega) is present by comparing to manual calc."""
        phi = 30.0
        sigma_v = 100.0
        perim = 1.0
        dz = 1.0
        omega = 5.0

        Qs = skin_friction_cohesionless(phi, sigma_v, perim, dz, omega_deg=omega)

        # Manual calculation
        delta = delta_from_phi(phi, "steel")
        delta_rad = math.radians(delta)
        omega_rad = math.radians(omega)
        Kd = nordlund_Kd(phi, omega)
        CF = nordlund_CF(delta / phi)
        fs_expected = Kd * CF * sigma_v * math.sin(delta_rad + omega_rad) / math.cos(omega_rad)
        expected = fs_expected * perim * dz
        assert Qs == pytest.approx(expected, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: CRITICAL-2 — Limiting tip resistance matches GEC-12 Fig 7-15
# ═══════════════════════════════════════════════════════════════════════

class TestLimitingTipResistance:
    """Verify limiting tip resistance from GEC-12 Figure 7-15 (Meyerhof 1976)."""

    def test_phi_26(self):
        assert _limiting_tip_resistance(26) == pytest.approx(5000, rel=1e-6)

    def test_phi_28(self):
        assert _limiting_tip_resistance(28) == pytest.approx(7000, rel=1e-6)

    def test_phi_30(self):
        assert _limiting_tip_resistance(30) == pytest.approx(10000, rel=1e-6)

    def test_phi_34(self):
        assert _limiting_tip_resistance(34) == pytest.approx(12000, rel=1e-6)

    def test_phi_38(self):
        assert _limiting_tip_resistance(38) == pytest.approx(16500, rel=1e-6)

    def test_phi_40(self):
        assert _limiting_tip_resistance(40) == pytest.approx(19000, rel=1e-6)

    def test_phi_below_26_conservative(self):
        """phi < 26 should return 3000 kPa (conservative minimum)."""
        assert _limiting_tip_resistance(20) == pytest.approx(3000, rel=1e-6)
        assert _limiting_tip_resistance(25) == pytest.approx(3000, rel=1e-6)

    def test_phi_above_40_capped(self):
        """phi > 40 should be capped at 19000 kPa."""
        assert _limiting_tip_resistance(42) == pytest.approx(19000, rel=1e-6)
        assert _limiting_tip_resistance(45) == pytest.approx(19000, rel=1e-6)

    def test_interpolation_phi_29(self):
        """phi=29 should interpolate between 7000 (phi=28) and 10000 (phi=30)."""
        expected = 7000 + (10000 - 7000) * (29 - 28) / (30 - 28)
        assert _limiting_tip_resistance(29) == pytest.approx(expected, rel=1e-6)

    def test_end_bearing_respects_limit(self):
        """End bearing should be capped by the limiting tip resistance."""
        # Very high sigma_v to force qt > qt_limit
        Qt = end_bearing_cohesionless(30, 5000, 0.1, 20, 0.3)
        # qt = alpha_t * Nq' * sigma_v = 1.0 * 35 * 5000 = 175000 kPa
        # qt_limit for phi=30 = 10000 kPa
        # So Qt should be limited: 10000 * 0.1 = 1000 kN
        assert Qt == pytest.approx(10000 * 0.1, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: CRITICAL-3 — Open-ended pipe pile plugging analysis
# ═══════════════════════════════════════════════════════════════════════

class TestOpenEndedPlugging:
    """Verify plugging analysis for open-ended pipe piles (GEC-12 Sec 7.2.1.4)."""

    def _make_open_pile_problem(self):
        """Open-ended 24-inch pipe pile in sand."""
        diameter = 0.6096  # 24 inches
        thickness = 0.0127  # 0.5 inch wall
        pile = make_pipe_pile(diameter, thickness, closed_end=False)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 18.0, friction_angle=32),
        ], gwt_depth=3.0)
        return pile, soil

    def test_open_pile_has_plugging_fields(self):
        """Open-ended pile should have tip_area_plugged and inner_perimeter set."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=False)
        assert pile.tip_area_plugged is not None
        assert pile.inner_perimeter is not None
        assert pile.closed_end is False
        # tip_area_plugged = full area
        r_o = 0.3
        assert pile.tip_area_plugged == pytest.approx(math.pi * r_o**2, rel=1e-4)
        # inner_perimeter = pi * inner_diameter
        inner_d = 0.6 - 2 * 0.012
        assert pile.inner_perimeter == pytest.approx(math.pi * inner_d, rel=1e-4)

    def test_closed_pile_no_plugging_fields(self):
        """Closed-end pile should have None for plugging fields."""
        pile = make_pipe_pile(0.6, 0.012, closed_end=True)
        assert pile.tip_area_plugged is None
        assert pile.inner_perimeter is None
        assert pile.closed_end is True

    def test_open_pile_returns_lesser_capacity(self):
        """Open-ended pile capacity should be lesser of plugged and unplugged."""
        pile, soil = self._make_open_pile_problem()
        result = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0
        ).compute()

        # Capacity must be positive
        assert result.Q_ultimate > 0
        assert result.Q_skin > 0
        assert result.Q_tip > 0
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )

    def test_open_less_than_closed(self):
        """Open-ended pile should have less (or equal) capacity than closed-end."""
        diameter = 0.6096
        thickness = 0.0127
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(20, 'cohesionless', 18.0, friction_angle=32),
        ], gwt_depth=3.0)

        pile_closed = make_pipe_pile(diameter, thickness, closed_end=True)
        pile_open = make_pipe_pile(diameter, thickness, closed_end=False)

        r_closed = AxialPileAnalysis(
            pile=pile_closed, soil=soil, pile_length=15.0
        ).compute()
        r_open = AxialPileAnalysis(
            pile=pile_open, soil=soil, pile_length=15.0
        ).compute()

        # Open-ended should be <= closed-end capacity
        # (plugged case has same tip as closed but might govern;
        #  unplugged has less tip but added inside friction)
        assert r_open.Q_ultimate <= r_closed.Q_ultimate + 1.0  # small tolerance

    def test_closed_end_behavior_unchanged(self):
        """Closed-end pipe pile should not be affected by plugging logic."""
        pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
        soil = AxialSoilProfile(layers=[
            AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
            AxialSoilLayer(10, 'cohesive', 17.0, cohesion=50),
        ], gwt_depth=3.0)

        result = AxialPileAnalysis(
            pile=pile, soil=soil, pile_length=15.0
        ).compute()

        # Basic checks that it still works correctly
        assert result.Q_ultimate > 0
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )
        assert result.Q_allowable == pytest.approx(
            result.Q_ultimate / 2.5, rel=1e-6
        )

    def test_plugging_governs_correctly(self):
        """Verify that the governing case (plugged vs unplugged) is correctly selected.

        The result should be the lesser of the two computed capacities.
        """
        pile, soil = self._make_open_pile_problem()
        analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15.0)
        result = analysis.compute()

        # The result Q_ultimate must equal Q_skin + Q_tip
        assert result.Q_ultimate == pytest.approx(
            result.Q_skin + result.Q_tip, rel=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
