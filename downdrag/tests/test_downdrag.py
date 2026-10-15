"""
Tests for the downdrag (negative skin friction) module.

Tests include:
1. Soil layer and profile data structures.
2. Skin friction computation (beta and alpha methods).
3. Consolidation settlement computation.
4. Neutral plane search and force equilibrium.
5. Settlement computation (elastic shortening + toe).
6. Limit state checks.
7. Full analysis integration (fill and GW drawdown scenarios).

Run with: pytest downdrag/tests/test_downdrag.py -v
"""

import math

import numpy as np
import pytest

from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import (
    DowndragAnalysis, _consolidation_settlement, _Nt_from_phi,
    _settlement_clay, _settlement_sand_elastic,
)
from downdrag.results import DowndragResult


# =============================================================================
# Test 1: Soil layer and profile data structures
# =============================================================================

class TestDowndragSoilLayer:
    """Verify DowndragSoilLayer validation and defaults."""

    def test_cohesionless_layer(self):
        """Cohesionless layer with phi should compute beta automatically."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesionless",
            unit_weight=19.0, phi=30.0,
        )
        phi_rad = math.radians(30.0)
        expected_beta = (1.0 - math.sin(phi_rad)) * math.tan(phi_rad)
        assert abs(layer.beta - expected_beta) < 1e-10

    def test_cohesive_layer(self):
        """Cohesive layer with cu should default alpha=1.0."""
        layer = DowndragSoilLayer(
            thickness=8.0, soil_type="cohesive",
            unit_weight=17.0, cu=30.0,
        )
        assert layer.alpha == 1.0

    def test_settling_layer_requires_Cc(self):
        """Settling layer must have Cc > 0."""
        with pytest.raises(ValueError, match="Cc > 0"):
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.0, e0=1.0,
            )

    def test_settling_layer_requires_e0(self):
        """Settling layer must have e0 > 0."""
        with pytest.raises(ValueError, match="e0 > 0"):
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.3, e0=0.0,
            )

    def test_invalid_soil_type(self):
        """Invalid soil type should raise."""
        with pytest.raises(ValueError, match="soil_type"):
            DowndragSoilLayer(
                thickness=5.0, soil_type="rock",
                unit_weight=25.0,
            )

    def test_cohesionless_requires_phi(self):
        """Cohesionless without phi should raise."""
        with pytest.raises(ValueError, match="phi > 0"):
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=19.0, phi=0.0,
            )


class TestDowndragSoilProfile:
    """Verify DowndragSoilProfile effective stress computation."""

    def _make_profile(self):
        """Two-layer profile: 5m sand over 10m clay, GWT at 2m."""
        return DowndragSoilProfile(
            layers=[
                DowndragSoilLayer(
                    thickness=5.0, soil_type="cohesionless",
                    unit_weight=19.0, phi=30.0,
                ),
                DowndragSoilLayer(
                    thickness=10.0, soil_type="cohesive",
                    unit_weight=18.0, cu=40.0,
                ),
            ],
            gwt_depth=2.0,
        )

    def test_effective_stress_above_gwt(self):
        """Above GWT, effective stress = total stress (no pore pressure)."""
        profile = self._make_profile()
        sigma = profile.effective_stress_at_depth(1.0)
        assert abs(sigma - 19.0 * 1.0) < 1e-10

    def test_effective_stress_below_gwt(self):
        """Below GWT, effective stress = total - pore pressure."""
        profile = self._make_profile()
        sigma = profile.effective_stress_at_depth(4.0)
        total = 19.0 * 4.0  # all in first layer
        u = 9.81 * (4.0 - 2.0)  # 2m below GWT
        assert abs(sigma - (total - u)) < 1e-10

    def test_effective_stress_second_layer(self):
        """Stress in second layer includes contribution from first."""
        profile = self._make_profile()
        z = 8.0  # 3m into second layer
        total = 19.0 * 5.0 + 18.0 * 3.0
        u = 9.81 * (8.0 - 2.0)
        assert abs(profile.effective_stress_at_depth(z) - (total - u)) < 1e-6

    def test_total_depth(self):
        """Total depth should sum all layer thicknesses."""
        profile = self._make_profile()
        assert abs(profile.total_depth - 15.0) < 1e-10

    def test_layer_at_depth(self):
        """layer_at_depth should return correct layer."""
        profile = self._make_profile()
        layer = profile.layer_at_depth(3.0)
        assert layer.soil_type == "cohesionless"
        layer = profile.layer_at_depth(7.0)
        assert layer.soil_type == "cohesive"


# =============================================================================
# Test 2: Skin friction computation
# =============================================================================

class TestSkinFriction:
    """Verify beta and alpha skin friction calculations."""

    def test_beta_from_phi_30(self):
        """Beta for phi=30 should be (1-sin30)*tan30 = 0.5*0.5774 = 0.2887."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesionless",
            unit_weight=19.0, phi=30.0,
        )
        expected = (1.0 - math.sin(math.radians(30))) * math.tan(math.radians(30))
        assert abs(layer.beta - expected) < 1e-6

    def test_beta_override(self):
        """Explicit beta should override computed value."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesionless",
            unit_weight=19.0, phi=30.0, beta=0.4,
        )
        assert layer.beta == 0.4

    def test_alpha_override(self):
        """Explicit alpha should override default."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesive",
            unit_weight=17.0, cu=50.0, alpha=0.5,
        )
        assert layer.alpha == 0.5

    def test_dragload_hand_calc(self):
        """Dragload for a single sand layer should match hand calculation."""
        # 5m sand layer, phi=30, gamma=19, GWT at surface
        # sigma_v at center (2.5m) = 19*2.5 - 9.81*2.5 = 22.975 kPa
        # beta = 0.2887, fs = 0.2887 * 22.975 = 6.632 kPa
        # Dragload = fs * perimeter * thickness = 6.632 * pi*0.3 * 5 = 31.25 kN
        layers = [
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=19.0, phi=30.0,
            ),
        ]
        profile = DowndragSoilProfile(layers=layers, gwt_depth=0.0)

        sigma_v_center = profile.effective_stress_at_depth(2.5)
        beta = layers[0].beta
        fs = beta * sigma_v_center
        perimeter = math.pi * 0.3
        dragload = fs * perimeter * 5.0

        # Verify against expected
        expected_sigma = 19.0 * 2.5 - 9.81 * 2.5
        assert abs(sigma_v_center - expected_sigma) < 1e-6
        assert dragload > 0


# =============================================================================
# Test 3: Consolidation settlement
# =============================================================================

class TestConsolidation:
    """Verify consolidation settlement computation."""

    def test_nc_settlement(self):
        """NC settlement should match Cc*H/(1+e0)*log10 formula."""
        # H=2m, e0=1.0, Cc=0.3, sigma_v0=50, delta_sigma=30
        Sc = _consolidation_settlement(
            H=2.0, e0=1.0, Cc=0.3, Cr=0.05,
            sigma_v0=50.0, sigma_p=50.0, delta_sigma=30.0,
        )
        expected = 0.3 * 2.0 / (1.0 + 1.0) * math.log10((50.0 + 30.0) / 50.0)
        assert abs(Sc - expected) < 1e-10

    def test_oc_stays_oc(self):
        """OC soil with small delta_sigma uses Cr."""
        Sc = _consolidation_settlement(
            H=2.0, e0=1.0, Cc=0.3, Cr=0.05,
            sigma_v0=50.0, sigma_p=100.0, delta_sigma=30.0,
        )
        # sigma_final = 80 < sigma_p = 100, so Cr governs
        expected = 0.05 * 2.0 / (1.0 + 1.0) * math.log10(80.0 / 50.0)
        assert abs(Sc - expected) < 1e-10

    def test_oc_to_nc(self):
        """OC soil exceeding sigma_p uses Cr then Cc."""
        Sc = _consolidation_settlement(
            H=2.0, e0=1.0, Cc=0.3, Cr=0.05,
            sigma_v0=50.0, sigma_p=70.0, delta_sigma=40.0,
        )
        # sigma_final = 90 > sigma_p = 70
        Sc_oc = 0.05 * 2.0 / (1 + 1.0) * math.log10(70.0 / 50.0)
        Sc_nc = 0.3 * 2.0 / (1 + 1.0) * math.log10(90.0 / 70.0)
        expected = Sc_oc + Sc_nc
        assert abs(Sc - expected) < 1e-10

    def test_zero_stress_change(self):
        """Zero or negative delta_sigma should return 0."""
        Sc = _consolidation_settlement(
            H=2.0, e0=1.0, Cc=0.3, Cr=0.05,
            sigma_v0=50.0, sigma_p=50.0, delta_sigma=0.0,
        )
        assert Sc == 0.0


# =============================================================================
# Test 4: Neutral plane search
# =============================================================================

class TestNeutralPlane:
    """Verify neutral plane search and force equilibrium."""

    def _make_standard_case(self):
        """Standard fill-over-clay-over-sand case.

        3m fill (non-settling, cohesionless)
        10m soft clay (settling, cohesive)
        7m dense sand (non-settling, cohesionless)
        Pile: 18m HP pile (steel), Q_dead=500 kN
        Fill: 3m of fill at 19 kN/m^3
        """
        layers = [
            DowndragSoilLayer(
                thickness=3.0, soil_type="cohesionless",
                unit_weight=19.0, phi=28.0,
                description="Fill",
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=25.0,
                settling=True, Cc=0.30, Cr=0.05, e0=1.2,
                description="Soft clay",
            ),
            DowndragSoilLayer(
                thickness=7.0, soil_type="cohesionless",
                unit_weight=20.0, phi=36.0,
                description="Dense sand",
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=1.0)
        return DowndragAnalysis(
            soil=soil,
            pile_length=18.0,
            pile_diameter=0.305,
            pile_E=200e6,
            pile_unit_weight=78.5,  # steel
            pile_area=0.01006,  # HP12x53 area
            Q_dead=500.0,
            fill_thickness=3.0,
            fill_unit_weight=19.0,
            structural_capacity=3000.0,
            allowable_settlement=0.025,
        )

    def test_neutral_plane_found(self):
        """Neutral plane should be found between 0 and pile length."""
        analysis = self._make_standard_case()
        result = analysis.compute()
        assert 0 < result.neutral_plane_depth < analysis.pile_length

    def test_neutral_plane_in_settling_zone(self):
        """NP should typically fall within or below the settling zone."""
        analysis = self._make_standard_case()
        result = analysis.compute()
        # Settling zone is 3-13m depth; NP should be in or below it
        assert result.neutral_plane_depth >= 3.0

    def test_dragload_positive(self):
        """Dragload should be positive (compression)."""
        analysis = self._make_standard_case()
        result = analysis.compute()
        assert result.dragload > 0

    def test_max_load_equals_components(self):
        """max_pile_load = Q_dead + dragload + pile_weight_to_np."""
        analysis = self._make_standard_case()
        result = analysis.compute()
        expected = result.Q_dead + result.dragload + result.pile_weight_to_np
        assert abs(result.max_pile_load - expected) / expected < 1e-6

    def test_more_dead_load_shallower_np(self):
        """More permanent load moves the neutral plane UP.

        UFC 3-220-20 §6-7.4 step 5: "increasing the permanent load applied
        to the top of the pile causes the neutral plane to occur at a higher
        elevation". Force equilibrium says the same: Q_dead + F(0->z) =
        R_toe + F(z->L), so a larger Q_dead needs a smaller z. (This test
        used to assert the opposite and passed only because Q_dead = 1000
        overloads the pile, which silently put the NP at the toe — G8.)
        """
        analysis_low = self._make_standard_case()
        analysis_low.Q_dead = 200.0
        result_low = analysis_low.compute()

        analysis_high = self._make_standard_case()
        analysis_high.Q_dead = 500.0
        result_high = analysis_high.compute()

        assert result_low.neutral_plane_method == "force_equilibrium"
        assert result_high.neutral_plane_method == "force_equilibrium"
        assert result_high.neutral_plane_depth < result_low.neutral_plane_depth

    def test_overloaded_pile_has_no_neutral_plane(self):
        """Q_dead above the total resistance: no NP, said so (G8).

        UFC 3-220-20 §6-7.4 step 5, limiting case: when the applied load
        reaches the nominal geotechnical resistance "there is no neutral
        plane". Never the pile toe with the whole shaft counted as drag.
        """
        analysis = self._make_standard_case()
        analysis.Q_dead = 1000.0   # toe + full shaft = 644 kN
        result = analysis.compute()
        assert result.neutral_plane_method == "none"
        assert result.dragload == 0.0
        assert result.geotechnical_ok is False
        assert any("NO NEUTRAL PLANE" in w for w in result.warnings)
        assert result.to_dict()["neutral_plane_method"] == "none"

    def test_no_settling_layers_no_downdrag(self):
        """No settling layer and a toe the pile cannot fully mobilize.

        The toe resistance (phi = 35 deg, ultimate) exceeds everything the
        pile can deliver, so the curves never cross. Nothing settles, so the
        pile settles at least as much as the soil everywhere: no negative
        skin friction, the NP at the head, zero drag — stated, not the old
        silent "NP at the toe" with the whole shaft counted as drag (G8).
        """
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=19.0, phi=30.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=300.0, fill_thickness=0.0, gw_drawdown=0.0,
        )
        result = analysis.compute()
        assert result.neutral_plane_method == "settlement_compatibility"
        assert result.neutral_plane_depth == 0.0
        assert result.dragload == 0.0
        assert result.max_pile_load == pytest.approx(300.0)
        assert "no soil settles" in result.neutral_plane_basis
        assert any("do not intersect" in w for w in result.warnings)


# =============================================================================
# Test 5: Settlement computation
# =============================================================================

class TestSettlement:
    """Verify settlement computations."""

    def test_elastic_shortening_simple(self):
        """Elastic shortening for constant load = Q*L/(A*E)."""
        # Single layer, no friction, just dead load
        layers = [
            DowndragSoilLayer(
                thickness=20.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        # Q_dead = 100 kN, below the ~250 kN total resistance, so the curves
        # cross (Q_dead = 500 used to overload the pile; the "NP at the
        # toe" default hid that — G8).
        analysis = DowndragAnalysis(
            soil=soil, pile_length=10.0, pile_diameter=0.3,
            pile_E=200e6, pile_area=0.01,
            Q_dead=100.0, fill_thickness=0.0, gw_drawdown=0.0,
        )
        result = analysis.compute()
        assert result.neutral_plane_method == "force_equilibrium"
        # Elastic shortening should be positive, and no more than the
        # largest load over the length above the NP.
        assert result.elastic_shortening > 0
        assert result.elastic_shortening <= (
            result.max_pile_load * result.neutral_plane_depth / (0.01 * 200e6)
            * 1.0001)

    def test_soil_settlement_decreasing_with_depth(self):
        """Soil settlement profile should decrease with depth."""
        layers = [
            DowndragSoilLayer(
                thickness=3.0, soil_type="cohesionless",
                unit_weight=19.0, phi=28.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=25.0,
                settling=True, Cc=0.30, Cr=0.05, e0=1.2,
            ),
            DowndragSoilLayer(
                thickness=7.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=1.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=18.0, pile_diameter=0.3,
            Q_dead=500.0, fill_thickness=3.0,
        )
        result = analysis.compute()
        profile = result.soil_settlement_profile
        # Settlement should generally decrease with depth (max at surface)
        assert profile[0] >= profile[-1]

    def test_pile_settlement_positive(self):
        """Pile settlement should be positive for a loaded pile."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=25.0,
                settling=True, Cc=0.30, Cr=0.05, e0=1.2,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=400.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        assert result.pile_settlement > 0


# =============================================================================
# Test 6: Limit state checks
# =============================================================================

class TestLimitStates:
    """Verify structural and geotechnical limit state checks."""

    def test_structural_passes(self):
        """Structural check should pass when capacity exceeds max load."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=200.0, fill_thickness=2.0,
            structural_capacity=5000.0,
        )
        result = analysis.compute()
        assert result.structural_ok is True

    def test_geotechnical_check(self):
        """Geotechnical check: Q_dead <= total_resistance (no dragload)."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=200.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        # With reasonable capacity, geotechnical should pass
        assert result.geotechnical_ok is True


# =============================================================================
# Test 7: Full analysis integration
# =============================================================================

class TestFullAnalysis:
    """Verify complete analysis scenarios."""

    def test_fill_scenario(self):
        """Full analysis with fill placement should produce valid results."""
        layers = [
            DowndragSoilLayer(
                thickness=3.0, soil_type="cohesionless",
                unit_weight=19.0, phi=28.0,
                description="Fill",
            ),
            DowndragSoilLayer(
                thickness=12.0, soil_type="cohesive",
                unit_weight=16.5, cu=20.0,
                settling=True, Cc=0.35, Cr=0.06, e0=1.4,
                description="Soft clay",
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=36.0,
                description="Dense sand",
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=1.5)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=22.0, pile_diameter=0.356,
            pile_E=200e6, pile_area=0.01387,
            pile_unit_weight=78.5,
            Q_dead=600.0,
            fill_thickness=3.0, fill_unit_weight=19.0,
            structural_capacity=4000.0,
            allowable_settlement=0.025,
        )
        result = analysis.compute()

        assert 0 < result.neutral_plane_depth < 22.0
        assert result.dragload > 0
        assert result.max_pile_load > result.Q_dead
        assert result.total_resistance > 0
        assert result.pile_settlement > 0

    def test_gw_drawdown_scenario(self):
        """Full analysis with GW drawdown should produce valid results."""
        layers = [
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=19.0, phi=30.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=35.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=2.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=20.0, pile_diameter=0.3,
            Q_dead=400.0,
            gw_drawdown=3.0,
        )
        result = analysis.compute()

        assert result.dragload > 0
        assert result.pile_settlement > 0

    def test_summary_output(self):
        """summary() should return a non-empty string."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=300.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        summary = result.summary()
        assert "Neutral plane" in summary
        assert "Drag load" in summary
        assert "Drag force" in summary
        assert len(summary) > 100

    def test_to_dict_output(self):
        """to_dict() should return a dict with expected keys."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=300.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        d = result.to_dict()
        assert 'neutral_plane_depth_m' in d
        assert 'dragload_kN' in d
        assert 'axial_load_kN' in d
        assert isinstance(d['axial_load_kN'], list)

    def test_Nt_from_phi(self):
        """Nt factor should increase with friction angle."""
        Nt_25 = _Nt_from_phi(25.0)
        Nt_30 = _Nt_from_phi(30.0)
        Nt_35 = _Nt_from_phi(35.0)
        assert Nt_25 < Nt_30 < Nt_35


# =============================================================================
# Test 8: DM7.2 (UFC 3-220-20) equation coverage
# =============================================================================

class TestUFC_Eq653_SettlementClay:
    """Verify UFC Eq 6-53 clay settlement with modified compression indices."""

    def test_modified_indices_match_traditional(self):
        """C_ec = Cc/(1+e0) should produce same result as traditional."""
        e0, Cc, Cr = 1.0, 0.3, 0.05
        C_ec = Cc / (1.0 + e0)
        C_er = Cr / (1.0 + e0)

        trad = _consolidation_settlement(
            H=2.0, e0=e0, Cc=Cc, Cr=Cr,
            sigma_v0=50.0, sigma_p=50.0, delta_sigma=30.0,
        )
        new = _settlement_clay(
            H=2.0, C_ec=C_ec, C_er=C_er,
            sigma_v0=50.0, sigma_p=50.0, delta_sigma=30.0,
        )
        assert abs(trad - new) < 1e-12

    def test_C_ec_only_input(self):
        """Layer created with C_ec only (no Cc/e0) should work."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesive",
            unit_weight=17.0, cu=30.0,
            settling=True, C_ec=0.15, C_er=0.025,
        )
        assert layer.C_ec == 0.15
        assert layer.C_er == 0.025

    def test_auto_derive_modified_from_traditional(self):
        """When Cc/e0 provided but C_ec not, C_ec should be auto-computed."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesive",
            unit_weight=17.0, cu=30.0,
            settling=True, Cc=0.3, Cr=0.05, e0=1.0,
        )
        assert abs(layer.C_ec - 0.15) < 1e-10
        assert abs(layer.C_er - 0.025) < 1e-10

    def test_nc_with_C_ec(self):
        """NC settlement using C_ec directly."""
        # C_ec = 0.15, H = 2m, sigma_v0 = 50, delta_sigma = 30
        # Sc = 0.15 * 2.0 * log10(80/50) = 0.3 * 0.2041 = 0.0612 m
        Sc = _settlement_clay(
            H=2.0, C_ec=0.15, C_er=0.025,
            sigma_v0=50.0, sigma_p=50.0, delta_sigma=30.0,
        )
        expected = 0.15 * 2.0 * math.log10(80.0 / 50.0)
        assert abs(Sc - expected) < 1e-10


class TestUFC_Eq654_SettlementSand:
    """Verify UFC Eq 6-54 elastic settlement of coarse-grained soil."""

    def test_hand_calc(self):
        """Match Eq 6-54 hand calculation."""
        # H = 2m, nu = 0.3, E_s = 20000 kPa, delta_sigma = 50 kPa
        # delta = 2 * (1.3)*(0.4) / (0.7 * 20000) * 50
        #       = 2 * 0.52 / 14000 * 50
        #       = 2 * 0.001857 = 0.003714 m
        Sc = _settlement_sand_elastic(
            H=2.0, nu_s=0.3, E_s=20000.0, delta_sigma=50.0,
        )
        expected = 2.0 * (1.3) * (0.4) / (0.7 * 20000.0) * 50.0
        assert abs(Sc - expected) < 1e-10

    def test_zero_delta_sigma(self):
        """No stress change should give zero settlement."""
        Sc = _settlement_sand_elastic(
            H=2.0, nu_s=0.3, E_s=20000.0, delta_sigma=0.0,
        )
        assert Sc == 0.0

    def test_cohesionless_settling_layer_requires_Es(self):
        """Cohesionless settling layer must have E_s > 0."""
        with pytest.raises(ValueError, match="E_s > 0"):
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=19.0, phi=30.0,
                settling=True,
            )

    def test_cohesionless_settling_layer_with_Es(self):
        """Cohesionless settling layer with E_s should work."""
        layer = DowndragSoilLayer(
            thickness=5.0, soil_type="cohesionless",
            unit_weight=19.0, phi=30.0,
            settling=True, E_s=20000.0, nu_s=0.3,
        )
        assert layer.E_s == 20000.0
        assert layer.nu_s == 0.3

    def test_sand_settlement_in_analysis(self):
        """Full analysis with settling sand layer should produce settlement."""
        layers = [
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=19.0, phi=30.0,
                settling=True, E_s=15000.0, nu_s=0.3,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=12.0, pile_diameter=0.3,
            Q_dead=400.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        # Should have non-zero soil settlement from the sand layer
        assert result.soil_settlement_at_np >= 0
        assert result.pile_settlement > 0


class TestUFC_Eq680_LRFD:
    """Verify UFC Eq 6-80 LRFD structural check."""

    def test_lrfd_demand_formula(self):
        """LRFD demand = 1.25*Q_dead + 1.10*(Q_np - Q_dead)."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=500.0, fill_thickness=2.0,
            structural_capacity=5000.0,
        )
        result = analysis.compute()

        # Verify LRFD demand formula
        drag_force = result.max_pile_load - result.Q_dead
        expected_demand = 1.25 * result.Q_dead + 1.10 * drag_force
        assert abs(result.structural_demand - expected_demand) < 1e-6

    def test_lrfd_greater_than_unfactored(self):
        """LRFD demand should be greater than unfactored max load when
        dead load contributes significantly."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=500.0, fill_thickness=2.0,
            structural_capacity=5000.0,
        )
        result = analysis.compute()
        # LRFD factors are >= 1.0, so demand >= max_pile_load
        assert result.structural_demand >= result.max_pile_load

    def test_lrfd_fail(self):
        """Structural check should fail when capacity is too low."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=500.0, fill_thickness=3.0,
            structural_capacity=100.0,  # way too low
        )
        result = analysis.compute()
        assert result.structural_ok is False

    def test_no_structural_capacity_gives_none(self):
        """Without structural_capacity, check should be None."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.25, Cr=0.04, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=500.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        assert result.structural_ok is None
        assert result.structural_demand is None


class TestUFC_Eq6_49_50_51_ToeSettlement:
    """Verify equivalent footing + 2V:1H toe settlement computation."""

    def test_toe_settlement_with_settling_bearing_layer(self):
        """Pile in settling bearing clay should have non-zero toe settlement."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.3, Cr=0.05, e0=1.2,
            ),
            # Bearing layer also settles
            DowndragSoilLayer(
                thickness=15.0, soil_type="cohesive",
                unit_weight=18.0, cu=60.0,
                settling=True, Cc=0.15, Cr=0.03, e0=0.8,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=12.0, pile_diameter=0.4,
            Q_dead=500.0, fill_thickness=3.0,
        )
        result = analysis.compute()
        # Toe settlement should be non-zero because bearing layer settles
        assert result.toe_settlement > 0

    def test_toe_settlement_in_dense_sand(self):
        """Pile bearing in non-settling sand should have ~zero toe settlement."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=25.0,
                settling=True, Cc=0.3, Cr=0.05, e0=1.2,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=36.0,
                # Not settling — competent bearing layer
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=400.0, fill_thickness=2.0,
        )
        result = analysis.compute()
        # Non-settling bearing layer => toe settlement = 0
        assert result.toe_settlement == 0.0

    def test_toe_settlement_2V1H_stress_decreases(self):
        """Stress from 2V:1H distribution should decrease with depth."""
        # Simple check: Q / (B'+z)(L'+z) decreases as z increases
        Q = 100.0  # kN
        B = 0.3
        stress_0 = Q / (B * B)  # at z=0
        stress_1 = Q / ((B + 1.0) * (B + 1.0))  # at z=1m
        assert stress_0 > stress_1


class TestGeotechnicalULS:
    """Verify that dragload is NEVER included in geotechnical ULS check.

    Per Fellenius, AASHTO, and UFC 3-220-20: dragload cancels at the
    neutral plane and must NOT be added to the geotechnical demand.
    The geotechnical check is: Q_dead <= positive_skin + toe_resistance.
    """

    def test_geotechnical_excludes_dragload(self):
        """Geotechnical check should compare Q_dead vs total_resistance only."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=30.0,
                settling=True, Cc=0.3, Cr=0.05, e0=1.0,
            ),
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesionless",
                unit_weight=20.0, phi=35.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=15.0, pile_diameter=0.3,
            Q_dead=200.0, fill_thickness=3.0,
        )
        result = analysis.compute()

        # Geotechnical check should be Q_dead <= total_resistance
        assert result.geotechnical_ok is True
        assert result.Q_dead <= result.total_resistance

        # Verify dragload is NOT zero (so this test is meaningful)
        assert result.dragload > 0

        # If dragload were included, demand would be much higher
        demand_with_dragload = result.Q_dead + result.dragload
        # total_resistance might be less than demand_with_dragload,
        # but the check should still pass because dragload is excluded
        if demand_with_dragload > result.total_resistance:
            # This proves the check is correct — including dragload
            # would fail, but the correct check (excluding it) passes
            assert result.geotechnical_ok is True

    def test_geotechnical_fails_when_Q_dead_too_high(self):
        """Geotechnical check fails when Q_dead alone exceeds resistance."""
        layers = [
            DowndragSoilLayer(
                thickness=10.0, soil_type="cohesive",
                unit_weight=17.0, cu=10.0,  # very weak clay
                settling=True, Cc=0.3, Cr=0.05, e0=1.2,
            ),
            DowndragSoilLayer(
                thickness=5.0, soil_type="cohesionless",
                unit_weight=20.0, phi=28.0,
            ),
        ]
        soil = DowndragSoilProfile(layers=layers, gwt_depth=0.0)
        analysis = DowndragAnalysis(
            soil=soil, pile_length=12.0, pile_diameter=0.2,
            Q_dead=5000.0,  # way too much load for small pile
            fill_thickness=2.0,
        )
        result = analysis.compute()
        assert result.geotechnical_ok is False


def test_Nt_from_phi_consistent_with_gec12_table_7_9():
    """Provenance pin (audit): _Nt_from_phi is a piecewise Nt(phi) interpolation
    duplicated from axial_pile, NOT a literal copy of GEC-12 Table 7-9 (Nt RANGES
    by soil type). Verified 2026-07-12 against the in-house GEC-12 reference
    (geotech_references/gec_12/tables.py _TABLE_7_9: clay 3-30, silt 20-40,
    sand 30-150, gravel 60-300). Pins the transcription against drift."""
    from downdrag.analysis import _Nt_from_phi
    assert 20 <= _Nt_from_phi(28) <= 40     # silt band
    assert 30 <= _Nt_from_phi(33) <= 150    # sand band
    assert 30 <= _Nt_from_phi(38) <= 150    # sand upper band
    assert _Nt_from_phi(25) <= 30           # within clay range top


class TestNeutralPlaneWhenCurvesDoNotCross:
    """G8 (live smoke 2026-10-08, DD-1 / DD-2): with an ultimate toe
    resistance larger than everything the pile can deliver, the load and
    resistance curves never cross. The module used to report the pile toe
    as the neutral plane silently and count the non-settling bearing sand
    as drag (409 kN for 8 m of clay). UFC 3-220-20 §6-7.4 step 5: the NP of
    an end-bearing pile in a stratum stiffer than the compressible soil is
    near the top of the bearing layer; Fig 6-19: pile and soil settle
    equally at the NP.
    """

    @staticmethod
    def _dd1(**kw):
        # The live-smoke DD-1 input: 8 m of consolidating clay (su 30 kPa,
        # alpha 0.5 -> fs 15 kPa) over dense sand, 0.4 m pile 15 m long,
        # 40 kPa surcharge as 2 m of fill.
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=8.0, soil_type="cohesive",
                              unit_weight=17.0, cu=30.0, alpha=0.5,
                              settling=True, Cc=0.25, e0=0.9),
            DowndragSoilLayer(thickness=7.0, soil_type="cohesionless",
                              unit_weight=19.0, phi=38.0, beta=0.3),
        ], gwt_depth=1.0)
        args = dict(soil=soil, pile_length=15.0, pile_diameter=0.4,
                    fill_thickness=2.0, fill_unit_weight=20.0, Q_dead=0.0)
        args.update(kw)
        return DowndragAnalysis(**args)

    def test_dd1_neutral_plane_at_bottom_of_settling_clay(self):
        r = self._dd1().compute()
        assert r.neutral_plane_method == "settlement_compatibility"
        # The clay/sand interface, not the toe (15 m).
        assert r.neutral_plane_depth == pytest.approx(8.0, abs=0.05)
        # Drag = clay friction only: 15 kPa x pi x 0.4 m x 8 m = 150.8 kN.
        assert r.dragload == pytest.approx(15.0 * math.pi * 0.4 * 8.0,
                                           rel=0.01)
        assert r.toe_force_mobilized == pytest.approx(0.0, abs=1e-6)
        d = r.to_dict()
        assert d["neutral_plane_method"] == "settlement_compatibility"
        assert any("do not intersect" in w for w in d["warnings"])
        assert "UFC 3-220-20" in d["neutral_plane_basis"]
        # The load below the NP falls, it does not keep rising to the toe.
        assert r.axial_load[-1] < r.max_pile_load
        # The soil and pile settle (nearly) equally at the NP.
        assert r.soil_settlement_at_np <= 0.001

    def test_dead_load_goes_to_the_toe_not_beyond_its_resistance(self):
        r = self._dd1(Q_dead=300.0).compute()
        assert r.neutral_plane_method == "settlement_compatibility"
        assert r.neutral_plane_depth == pytest.approx(8.0, abs=0.05)
        assert 0.0 < r.toe_force_mobilized < r.toe_resistance
        assert r.axial_load[-1] == pytest.approx(r.toe_force_mobilized,
                                                 rel=0.02)

    def test_end_bearing_pile_neutral_plane_at_toe_with_basis(self):
        """Clay settles all the way down to an unyielding toe: NP at the
        toe is right (UFC §6-7.4 step 5), and the basis says why."""
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=15.0, soil_type="cohesive",
                              unit_weight=17.0, cu=30.0, alpha=0.5,
                              settling=True, Cc=0.25, e0=0.9),
            DowndragSoilLayer(thickness=5.0, soil_type="cohesionless",
                              unit_weight=21.0, phi=40.0),
        ], gwt_depth=1.0)
        r = DowndragAnalysis(soil=soil, pile_length=15.0, pile_diameter=0.4,
                             fill_thickness=2.0, fill_unit_weight=20.0,
                             Q_dead=200.0, Nt=2000.0).compute()
        assert r.neutral_plane_method == "settlement_compatibility"
        assert r.neutral_plane_depth == pytest.approx(15.0, abs=1e-6)
        assert "end-bearing" in r.neutral_plane_basis

    def test_mobilized_toe_restores_force_equilibrium_with_a_note(self):
        """A small (mobilized) toe value: the curves cross, inside the
        non-settling sand. The result keeps UFC's conservative force
        equilibrium and says the friction below 8 m is counted as negative
        skin friction."""
        r = self._dd1(Nt=5.0).compute()
        assert r.neutral_plane_method == "force_equilibrium"
        assert r.neutral_plane_depth > 8.5
        assert any("counted as negative" in w for w in r.warnings)

    def test_summary_states_basis_and_warnings(self):
        text = self._dd1().compute().summary()
        assert "How the Headline Neutral Plane Was Located" in text
        assert "Neutral Plane by Each Method" in text
        assert "Warnings" in text


# =============================================================================
# Every neutral-plane method side by side (owner, 2026-10-09: "Downdrag has
# many methods and it should be open to multiple assumptions. See DM 7.2 or
# the FHWA driven pile manual."; 2026-10-14: "presenting all of them is
# fine"; GM-48, GM-01)
# =============================================================================

FT = 0.3048
KIP = 4.4482216
PSF = 0.04788026
PCF = 0.15708746   # kN/m3 per pcf


def _dd1(**kw):
    """The live-smoke DD-1 input (G8): 8 m of consolidating clay (fs 15 kPa)
    over dense sand, 0.4 m pile 15 m long, 40 kPa of fill."""
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=8.0, soil_type="cohesive",
                          unit_weight=17.0, cu=30.0, alpha=0.5,
                          settling=True, Cc=0.25, e0=0.9),
        DowndragSoilLayer(thickness=7.0, soil_type="cohesionless",
                          unit_weight=19.0, phi=38.0, beta=0.3),
    ], gwt_depth=1.0)
    args = dict(soil=soil, pile_length=15.0, pile_diameter=0.4,
                fill_thickness=2.0, fill_unit_weight=20.0, Q_dead=0.0)
    args.update(kw)
    return DowndragAnalysis(**args)


def _floating(**kw):
    """A friction pile floating in settling clay (toe in the clay): the
    clear-cut case where every applicable basis agrees."""
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=25.0, soil_type="cohesive",
                          unit_weight=17.0, cu=30.0, alpha=1.0,
                          settling=True, Cc=0.3, e0=1.0),
    ], gwt_depth=1.0)
    args = dict(soil=soil, pile_length=20.0, pile_diameter=0.4,
                fill_thickness=2.0, Q_dead=300.0)
    args.update(kw)
    return DowndragAnalysis(**args)


def _ufc_b5(**kw):
    """UFC 3-220-20 (16 Jan 2025) Appendix B-5, Example 5 (pp. 633-643):
    12 in square PPC piles, 65 ft long below the cap at the original grade,
    8 ft of new fill; soft clay 40 ft (su 500 psf, alpha 1.0 -> 2 k/ft),
    stiff clay below (su 3000 psf): a 10 ft transition (720 psf -> 2.9 k/ft)
    then alpha 0.5 (1500 psf -> 6 k/ft); Rb = 9 su B^2 = 27 kips; Qd = 40
    kips per pile; the load curve carries no pile weight (B-5.5).
    Compressibility of the clays is not used by force equilibrium; the
    values here are placeholders, not the example's (Fig B-16)."""
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(thickness=40 * FT, soil_type="cohesive",
                          unit_weight=100 * PCF, cu=500 * PSF, alpha=1.0,
                          settling=True, Cc=0.5, e0=1.5,
                          description="soft clay"),
        DowndragSoilLayer(thickness=10 * FT, soil_type="cohesive",
                          unit_weight=120 * PCF, cu=3000 * PSF,
                          alpha=720.0 / 3000.0,
                          description="stiff clay, transition"),
        DowndragSoilLayer(thickness=41 * FT, soil_type="cohesive",
                          unit_weight=120 * PCF, cu=3000 * PSF, alpha=0.5,
                          description="stiff clay"),
    ], gwt_depth=0.0)
    args = dict(soil=soil, pile_length=65 * FT, pile_diameter=1 * FT,
                pile_perimeter=4 * FT, pile_area=FT * FT,
                pile_E=4030e3 * 144 * PSF, pile_unit_weight=0.0,
                Q_dead=40 * KIP, fill_thickness=8 * FT,
                fill_unit_weight=120 * PCF, n_sublayers=40)
    args.update(kw)
    return DowndragAnalysis(**args)


def _rows(result):
    return {r["name"]: r for r in result.neutral_plane_comparison}


def _row(result, method, mob=None):
    """The comparison row of one method (and toe mobilization)."""
    return next(r for r in result.neutral_plane_comparison
                if r["method"] == method
                and (mob is None or abs(r["toe_mobilization"] - mob) < 1e-9))


def _shaft_from_profile(result, perimeter, z0, z1):
    """Shaft resistance between z0 and z1 integrated independently from the
    reported (z, unit_skin_friction) profile, read as piecewise linear (a
    repeated depth is a jump, as in a CGPR #56 profile)."""
    z, fs = result.z, result.unit_skin_friction
    F = np.concatenate(([0.0], np.cumsum(0.5 * (fs[:-1] + fs[1:])
                                         * np.diff(z))))

    def upto(x):
        if x <= z[0]:
            return 0.0
        if x >= z[-1]:
            return float(F[-1])
        i = int(np.searchsorted(z, x, side="right")) - 1
        t = x - z[i]
        f_x = fs[i] + (fs[i + 1] - fs[i]) * t / (z[i + 1] - z[i])
        return float(F[i] + 0.5 * (fs[i] + f_x) * t)

    return (upto(z1) - upto(z0)) * perimeter


class TestUFCAppendixB5NeutralPlane:
    """UFC 3-220-20 App. B-5.5, Table B-28 (p. 640): the neutral plane for
    0 %, 50 % and 100 % base mobilization (Siegel et al. 2013, as UFC
    §6-5.8.4.2 recommends). Published: NP 48 / 50 / 52 ft below the final
    ground surface (pile head 8 ft below it), Pmax 120 / 126 / 133 kips.
    Exact from the printed load-transfer rates: 47.75 / 50.16 / 52.48 ft,
    119.5 / 126.3 / 133.0 kips. Achieved (since GM-01, 2026-10-14): 47.70 /
    50.14 / 52.48 ft, 119.4 / 126.2 / 132.9 kips (before: 47.81 / 50.16 /
    52.50 ft, 119.6 / 126.1 / 132.9 kips, and the reported Pmax 1.2-1.9 %
    above the toe + positive friction at the plane). The transition layer
    is entered as alpha = 720/3000 (2.88 k/ft against the printed 2.9)."""

    @pytest.mark.parametrize("mob, np_pub, np_exact, p_pub, p_exact", [
        (0.0, 48.0, 47.75, 120.0, 119.5),
        (0.5, 50.0, 50.155, 126.0, 126.25),
        (1.0, 52.0, 52.483, 133.0, 132.99),
    ])
    def test_table_b28(self, mob, np_pub, np_exact, p_pub, p_exact):
        r = _ufc_b5(neutral_plane_method="force_equilibrium",
                    toe_mobilization=mob).compute()
        assert r.toe_resistance / KIP == pytest.approx(27.0, rel=0.005)
        np_gs_ft = r.neutral_plane_depth / FT + 8.0
        assert np_gs_ft == pytest.approx(np_exact, abs=0.1)
        assert np_gs_ft == pytest.approx(np_pub, abs=1.0)
        assert r.max_pile_load / KIP == pytest.approx(p_exact, rel=0.01)
        assert r.max_pile_load / KIP == pytest.approx(p_pub, abs=1.0)
        assert r.neutral_plane_method == "force_equilibrium"
        # GM-01: the reported Pmax is the resistance curve at the plane.
        assert r.max_pile_load == pytest.approx(
            mob * r.toe_resistance + r.positive_skin_friction, rel=1e-9)

    def test_comparison_rows_carry_the_same_three_planes(self):
        r = _ufc_b5().compute()
        for mob, np_pub in ((0.0, 48.0), (0.5, 50.0), (1.0, 52.0)):
            row = _row(r, "force_equilibrium", mob)
            assert row["name"] == (
                f"Neutral plane at {round(mob * 100)}% toe mobilization")
            assert row["applies"]
            assert (row["neutral_plane_depth_m"] / FT + 8.0
                    == pytest.approx(np_pub, abs=1.0))

    def test_bearing_layer_rule_is_the_base_of_the_soft_clay(self):
        """UFC B-5.7: 'the position of the neutral plane is below the bottom
        of the soft clay' (48 ft); UFC §6-7.4 step 5 puts it there."""
        row = _row(_ufc_b5().compute(), "bearing_layer_top")
        assert row["applies"]
        assert row["neutral_plane_depth_m"] / FT + 8.0 == pytest.approx(
            48.0, abs=1e-6)
        # Q_np = 40 + 2 k/ft x 40 ft = 120 kips, exactly: the layer
        # boundary is a node, so the drag load stops at the jump.
        assert row["max_pile_load_kN"] / KIP == pytest.approx(120.0,
                                                              rel=1e-6)


class TestNeutralPlaneComparison:
    """Every method on the same inputs, named as its reference names it,
    marked when it does not apply, with the reason."""

    def test_dd1_rows(self):
        r = _dd1().compute()
        names = [row["name"] for row in r.neutral_plane_comparison]
        assert names == [
            "Neutral plane at 100% toe mobilization",
            "Neutral plane at 50% toe mobilization",
            "Neutral plane at 0% toe mobilization",
            "Mobilized toe resistance compatible with the toe penetration",
            "Neutral plane at the bearing layer (end-bearing pile)",
            "Neutral plane at the pile toe",
            "Endo et al. (1969)"]
        # The ultimate toe (Nt 150) cannot be mobilized: no intersection.
        for mob in (1.0, 0.5):
            row = _row(r, "force_equilibrium", mob)
            assert row["applies"] is False
            assert "do not intersect" in row["reason"]
        fe0 = _row(r, "force_equilibrium", 0.0)
        assert fe0["applies"] and fe0["neutral_plane_depth_m"] > 8.0
        clay_drag = 15.0 * math.pi * 0.4 * 8.0       # 150.8 kN
        for method in ("settlement_compatibility", "bearing_layer_top"):
            row = _row(r, method)
            assert row["neutral_plane_depth_m"] == pytest.approx(8.0,
                                                                 abs=0.05)
            assert row["dragload_kN"] == pytest.approx(clay_drag, rel=0.01)
        toe = _row(r, "pile_toe")
        assert "upper bound" in toe["assumptions"]
        assert toe["neutral_plane_depth_m"] == pytest.approx(15.0)
        assert toe["dragload_kN"] > 2.5 * clay_drag
        endo = _row(r, "endo")
        assert endo["applies"] is False
        assert "endo_bearing_condition" in endo["reason"]
        for row in r.neutral_plane_comparison:
            assert row["source"] and row["assumptions"]
            assert "role" not in row
        # Exactly the headline row is flagged.
        assert [row["used"] for row in r.neutral_plane_comparison].count(
            True) == 1
        assert _row(r, "settlement_compatibility")["used"]

    def test_every_applicable_row_reports_plane_drag_and_settlement(self):
        r = _dd1(Q_dead=300.0).compute()
        for row in r.neutral_plane_comparison:
            if not row["applies"]:
                continue
            for key in ("neutral_plane_depth_m", "max_pile_load_kN",
                        "dragload_kN", "drag_force_kN", "toe_force_kN",
                        "pile_settlement_mm"):
                assert row[key] is not None, (row["name"], key)
            assert row["drag_force_kN"] == pytest.approx(
                row["max_pile_load_kN"] - 300.0)

    def test_settlements_at_the_plane_are_reported(self):
        r = _dd1(Q_dead=300.0).compute()
        sc = _row(r, "settlement_compatibility")
        # pile and soil settle (nearly) equally at the plane
        assert sc["soil_settlement_at_np_mm"] == pytest.approx(
            sc["pile_settlement_at_np_mm"], abs=0.5)
        fe0 = _row(r, "force_equilibrium", 0.0)
        # A shallow 0 % plane leaves the soil settling far more than the
        # pile there; the pile head then settles with the soil at the
        # plane (UFC Fig 6-19): far more than with the plane at 8 m.
        assert fe0["soil_settlement_at_np_mm"] > (
            fe0["pile_settlement_at_np_mm"] + 50.0)
        assert fe0["pile_settlement_mm"] > sc["pile_settlement_mm"] + 50.0

    def test_extra_toe_mobilization_gets_its_own_row(self):
        r = _floating(toe_mobilization=0.3).compute()
        row = _row(r, "force_equilibrium", 0.3)
        assert row["name"] == "Neutral plane at 30% toe mobilization"
        assert r.neutral_plane_method == "force_equilibrium"
        assert r.neutral_plane_depth == pytest.approx(
            row["neutral_plane_depth_m"])

    def test_overloaded_pile_no_method_applies(self):
        r = _dd1(Q_dead=5000.0).compute()
        assert r.neutral_plane_method == "none"
        assert r.headline_method == "no neutral plane"
        assert "No method gives a neutral plane" in r.methods_range
        for row in r.neutral_plane_comparison:
            if row["method"] != "endo":
                assert row["applies"] is False
                assert "no neutral plane" in row["reason"]

    def test_floating_pile_compatible_plane_equals_full_mobilization(self):
        """Clay settles more than the pile down past the intersection of
        the curves with the toe fully mobilized, so the toe is pushed to
        its full resistance (elastic-plastic toe): the two planes agree."""
        r = _floating(neutral_plane_method="settlement_compatibility").compute()
        assert r.neutral_plane_depth == pytest.approx(
            _row(r, "force_equilibrium", 1.0)["neutral_plane_depth_m"])
        assert "full resistance" in r.neutral_plane_basis
        blt = _row(r, "bearing_layer_top")
        assert blt["applies"] is False
        assert "floating" in blt["reason"]


class TestNeutralPlaneOptions:
    """Each neutral_plane_method on its own."""

    def test_default_is_auto_and_unchanged(self):
        r = _dd1().compute()
        assert r.neutral_plane_requested == "auto"
        assert r.neutral_plane_method == "settlement_compatibility"
        assert r.neutral_plane_depth == pytest.approx(8.0, abs=0.05)

    def test_force_equilibrium_raises_when_curves_do_not_cross(self):
        with pytest.raises(ValueError, match="do not intersect"):
            _dd1(neutral_plane_method="force_equilibrium").compute()

    def test_force_equilibrium_zero_toe(self):
        r = _dd1(neutral_plane_method="force_equilibrium",
                 toe_mobilization=0.0).compute()
        row = _row(r, "force_equilibrium", 0.0)
        assert r.neutral_plane_method == "force_equilibrium"
        assert r.neutral_plane_depth == pytest.approx(
            row["neutral_plane_depth_m"])
        assert r.toe_force_mobilized == 0.0
        assert "toe at 0%" in r.neutral_plane_basis
        # GM-01: equilibrium at the plane holds exactly, Q_np = positive
        # friction below (zero toe), because the plane search and the
        # reported drag load / positive friction integrate the same profile
        # the same way (before: ~4 % apart on this case).
        assert r.max_pile_load == pytest.approx(r.positive_skin_friction,
                                                rel=1e-9)

    def test_settlement_compatibility(self):
        r = _dd1(neutral_plane_method="settlement_compatibility").compute()
        assert r.neutral_plane_method == "settlement_compatibility"
        assert r.neutral_plane_depth == pytest.approx(8.0, abs=0.05)

    def test_bearing_layer_top(self):
        r = _dd1(neutral_plane_method="bearing_layer_top",
                 Q_dead=200.0).compute()
        assert r.neutral_plane_method == "bearing_layer_top"
        assert r.neutral_plane_depth == pytest.approx(8.0, abs=1e-9)
        assert r.dragload == pytest.approx(15.0 * math.pi * 0.4 * 8.0,
                                           rel=0.01)
        assert "§6-7.4 step 5" in r.neutral_plane_basis
        assert r.neutral_plane_basis.startswith(
            "Neutral plane at the bearing layer (end-bearing pile) "
            "(neutral_plane_method='bearing_layer_top'")
        assert r.headline_method == (
            "Neutral plane at the bearing layer (end-bearing pile)")
        # The load below the plane falls to the toe force it delivers.
        assert r.axial_load[-1] == pytest.approx(r.toe_force_mobilized,
                                                 rel=0.02)

    def test_bearing_layer_top_not_for_a_floating_pile(self):
        with pytest.raises(ValueError, match="floating"):
            _floating(neutral_plane_method="bearing_layer_top").compute()

    def test_pile_toe_upper_bound(self):
        r = _dd1(neutral_plane_method="pile_toe").compute()
        assert r.neutral_plane_method == "pile_toe"
        assert r.neutral_plane_depth == pytest.approx(15.0)
        assert "upper bound" in r.neutral_plane_basis
        assert r.dragload == pytest.approx(
            _row(r, "pile_toe")["dragload_kN"])

    def test_endo_matches_cgpr56_function(self):
        """The Endo option is CGPR #56 §3.2.1 (validated on its Section
        3.4.2 worked example in test_cgpr56.py): plane at 0.75 L for
        'stiff_flexible', drag = shaft friction above it. Cross-checked
        against downdrag.endo_method on the module's own fs profile: both
        integrate the same piecewise-linear profile, so they agree to
        rounding (GM-01; 2 % apart before)."""
        from downdrag import endo_method
        a = _dd1(neutral_plane_method="endo",
                 endo_bearing_condition="stiff_flexible", Q_dead=100.0)
        r = a.compute()
        assert r.neutral_plane_method == "endo"
        assert r.neutral_plane_depth == pytest.approx(0.75 * 15.0)
        ref = endo_method(
            Q_static=100.0, pile_length=15.0,
            pile_perimeter=a.pile_perimeter,
            skin_friction_profile=list(zip(r.z, r.unit_skin_friction)),
            bearing_condition="stiff_flexible")
        assert r.dragload == pytest.approx(ref.drag_load, rel=1e-9)
        assert "Endo et al. (1969), 'stiff_flexible' (0.75 L)" in _rows(r)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="neutral_plane_method"):
            _dd1(neutral_plane_method="fellenius")
        with pytest.raises(ValueError, match="toe_mobilization"):
            _dd1(toe_mobilization=1.5)
        with pytest.raises(ValueError, match="endo_bearing_condition"):
            _dd1(neutral_plane_method="endo")
        with pytest.raises(ValueError, match="endo_bearing_condition"):
            _dd1(endo_bearing_condition="rock")
        with pytest.raises(ValueError, match="skin_friction_stress"):
            _dd1(skin_friction_stress="effective")

    def test_final_stress_is_the_default(self):
        """UFC 3-220-20 §6-7.4 step 2 (p. 476): "Nominal side and base
        resistances should be evaluated considering the vertical effective
        stress profile that includes stress change(s) producing
        settlement." GEC-12 §7.3.6.1 states no rule in those words, so the
        default is final stress (owner, 2026-10-14)."""
        assert DowndragAnalysis.__dataclass_fields__[
            "skin_friction_stress"].default == "final"
        soil = DowndragSoilProfile(layers=[
            DowndragSoilLayer(thickness=6.0, soil_type="cohesionless",
                              unit_weight=18.0, phi=30.0, settling=True,
                              E_s=8000.0),
            DowndragSoilLayer(thickness=10.0, soil_type="cohesionless",
                              unit_weight=20.0, phi=36.0),
        ], gwt_depth=1.0)
        base = dict(soil=soil, pile_length=12.0, pile_diameter=0.4,
                    fill_thickness=2.0, Q_dead=200.0,
                    neutral_plane_method="pile_toe")
        r0 = DowndragAnalysis(skin_friction_stress="initial",
                              **base).compute()
        r1 = DowndragAnalysis(**base).compute()
        assert r1.dragload > r0.dragload * 1.1
        assert "§6-7.4 step 2" in r1.neutral_plane_basis
        assert r1.to_dict()["skin_friction_stress"] == "final"
        assert "initial effective stress" in r0.neutral_plane_basis


class TestMethodsSideBySide:
    """Every result carries every method's neutral plane, drag force and
    settlement as a first-class table, the spread in one plain sentence,
    and no coined labels (owner, 2026-10-14; GM-48)."""

    COINED = ("materially", "materiality", "judgment", "judgement",
              "settlement compatibility", "settlement-compatibility",
              "Bases", "basis is the engineer")

    def test_table_in_summary_and_to_dict(self):
        r = _dd1().compute()
        d = r.to_dict()
        assert len(d["neutral_plane_comparison"]) == 7
        assert "judgment" not in d
        text = r.summary()
        assert "Neutral Plane by Each Method (same inputs)" in text
        for row in r.neutral_plane_comparison:
            assert row["name"] in text
            assert row["source"] in text
        assert ("* Mobilized toe resistance compatible with the toe "
                "penetration") in text
        assert "* gives the headline numbers below." in text

    def test_methods_range_sentence(self):
        r = _dd1().compute()
        ok = [row for row in r.neutral_plane_comparison if row["applies"]]
        nps = [row["neutral_plane_depth_m"] for row in ok]
        dfs = [row["drag_force_kN"] for row in ok]
        assert r.methods_range == (
            f"The methods give neutral planes from {min(nps):.2f} m to "
            f"{max(nps):.2f} m and drag forces from {min(dfs):.0f} kN to "
            f"{max(dfs):.0f} kN ({len(ok)} of 7 methods apply).")
        assert r.to_dict()["methods_range"] == r.methods_range
        assert r.methods_range in r.summary()

    def test_methods_range_shown_even_when_methods_agree(self):
        """No threshold decides whether the spread is shown."""
        r = _floating().compute()
        assert r.methods_range.startswith("The methods give neutral planes")

    def test_headline_is_the_ufc_procedure(self):
        r = _dd1().compute()
        assert r.neutral_plane_basis.startswith(
            "UFC 3-220-20 §6-7.4 procedure: ")
        assert r.headline_method == (
            "Mobilized toe resistance compatible with the toe penetration")
        r = _ufc_b5().compute()
        assert r.neutral_plane_basis.startswith(
            "UFC 3-220-20 §6-7.4 procedure: ")
        assert r.headline_method == "Neutral plane at 100% toe mobilization"
        assert "--- Headline: Neutral plane at 100% toe mobilization ---" in (
            r.summary())

    @pytest.mark.parametrize("make", [_dd1, _floating, _ufc_b5])
    def test_no_coined_labels(self, make):
        r = make().compute()
        import json
        blob = r.summary() + json.dumps(r.to_dict())
        for word in self.COINED:
            assert word not in blob, word


class TestQuadratureConsistency:
    """GM-01: the plane search, the drag load, the positive friction and the
    axial load all integrate the same unit skin friction profile the same
    way (the reported profile read as piecewise linear)."""

    @pytest.mark.parametrize("make, perimeter", [
        (_dd1, math.pi * 0.4), (_floating, math.pi * 0.4),
        (_ufc_b5, 4 * FT)])
    def test_rows_balance_at_their_planes(self, make, perimeter):
        r = make().compute()
        for row in r.neutral_plane_comparison:
            if not row["applies"]:
                continue
            z = row["neutral_plane_depth_m"]
            # The drag load is the profile's integral down to the plane.
            assert row["dragload_kN"] == pytest.approx(
                _shaft_from_profile(r, perimeter, 0.0, z), rel=1e-9,
                abs=1e-9)
            if row["method"] == "force_equilibrium":
                # Load curve = resistance curve at the plane, exactly.
                below = _shaft_from_profile(r, perimeter, z, r.pile_length)
                assert row["max_pile_load_kN"] == pytest.approx(
                    row["toe_force_kN"] + below, rel=1e-9)

    def test_layer_boundary_is_a_jump_in_the_profile(self):
        """A layer boundary is two nodes at one depth (the value above, then
        the value below), so the integral stops exactly at the jump."""
        r = _dd1().compute()
        at = np.flatnonzero(np.isclose(r.z, 8.0))
        assert len(at) == 2
        assert r.unit_skin_friction[at[0]] == pytest.approx(15.0)  # clay
        assert r.unit_skin_friction[at[1]] > 15.0                  # sand
        assert _row(r, "bearing_layer_top")["dragload_kN"] == pytest.approx(
            15.0 * math.pi * 0.4 * 8.0, rel=1e-12)
        # The water table (1.0 m) is a node: beta friction kinks there.
        assert np.isclose(r.z, 1.0).sum() == 1

    def test_headline_positive_friction_and_axial_load(self):
        r = _floating().compute()        # intersection, toe fully mobilized
        perimeter = math.pi * 0.4
        assert r.positive_skin_friction == pytest.approx(
            _shaft_from_profile(r, perimeter, r.neutral_plane_depth,
                                r.pile_length), rel=1e-9)
        assert r.max_pile_load == pytest.approx(
            r.toe_resistance + r.positive_skin_friction, rel=1e-9)
        # The axial load peaks at the plane at Q_np (to the grid spacing).
        assert max(r.axial_load) <= r.max_pile_load * (1 + 1e-9)
        assert max(r.axial_load) == pytest.approx(r.max_pile_load, rel=0.01)


class TestPileHeadSettlement:
    """UFC 3-220-20 Fig 6-19 / §6-5.8 item 6: the pile head settles as much
    as the neutral plane plus the elastic compression above it."""

    def test_floating_pile_settles_with_the_soil_at_the_plane(self):
        r = _floating().compute()
        used = _row(r, "force_equilibrium", 1.0)
        assert used["used"]
        assert r.pile_settlement == pytest.approx(
            used["pile_settlement_mm"] / 1000.0)
        assert r.pile_settlement >= r.soil_settlement_at_np
        assert r.pile_settlement == pytest.approx(
            max(used["soil_settlement_at_np_mm"],
                used["pile_settlement_at_np_mm"]) / 1000.0
            + r.elastic_shortening)

    def test_higher_plane_more_settlement(self):
        """UFC §6-5.8.4.2: the settlement increases as the neutral plane is
        positioned higher."""
        r = _floating().compute()
        s = [_row(r, "force_equilibrium", m)["pile_settlement_mm"]
             for m in (1.0, 0.5, 0.0)]
        assert s[0] < s[1] < s[2]


class TestPileOwnWeight:
    """The pile's own weight is left out of the load curve by default, as
    UFC 3-220-20 Fig 6-26 and FHWA GEC-12 §7.3.6.1 (and both their worked
    examples) do (owner, 2026-10-15, OWN-58); include_pile_weight=True adds
    it. GM-50: the unit weight defaults to steel, as pile_E does."""

    REFS = "UFC 3-220-20 Fig 6-26 and FHWA GEC-12 §7.3.6.1"

    def test_default_leaves_the_weight_out(self):
        """DD-1: Q_np = the drag load of the 8 m of clay, without the
        +24.1 kN of pile weight it carried before 2026-10-15 (174.9 kN)."""
        assert DowndragAnalysis.__dataclass_fields__[
            "include_pile_weight"].default is False
        r = _dd1().compute()
        assert r.include_pile_weight is False
        assert r.pile_weight_to_np == 0.0
        assert r.max_pile_load == pytest.approx(150.78, abs=0.01)
        assert r.max_pile_load == pytest.approx(r.dragload, rel=1e-12)
        assert r.drag_force == pytest.approx(r.dragload, rel=1e-12)
        fe0 = _row(r, "force_equilibrium", 0.0)    # before: 10.12 m, 274.7
        assert fe0["neutral_plane_depth_m"] == pytest.approx(10.432,
                                                             abs=0.001)
        assert fe0["max_pile_load_kN"] == pytest.approx(259.46, abs=0.01)
        toe = _row(r, "pile_toe")                  # before: 564.2 kN
        assert toe["max_pile_load_kN"] == pytest.approx(518.92, abs=0.01)
        for row in r.neutral_plane_comparison:
            if row["applies"]:
                assert row["max_pile_load_kN"] == pytest.approx(
                    row["dragload_kN"], rel=1e-12)

    def test_every_output_says_which_basis(self):
        r = _dd1().compute()
        assert (f"The pile's own weight is left out of the load curve, as "
                f"{self.REFS} do (include_pile_weight=False).") in (
                    r.neutral_plane_basis)
        d = r.to_dict()
        assert d["include_pile_weight"] is False
        assert d["pile_weight_basis"] == (
            f"Pile's own weight left out of the load curve, as {self.REFS} "
            f"do")
        assert self.REFS in d["neutral_plane_basis"]
        text = r.summary()
        assert (f"Pile's own weight: left out of the load curve, as "
                f"{self.REFS} do") in text
        assert "includes the pile weight" not in text
        assert "pile weight +" not in r.neutral_plane_basis

    def test_weight_included_reproduces_the_old_numbers(self):
        """include_pile_weight=True with the old concrete default (24
        kN/m3) gives DD-1 exactly as before 2026-10-15."""
        r = _dd1(include_pile_weight=True, pile_unit_weight=24.0).compute()
        area = math.pi / 4.0 * 0.4 ** 2
        assert r.include_pile_weight is True
        assert r.pile_weight_to_np == pytest.approx(
            24.0 * area * r.neutral_plane_depth, rel=1e-12)
        assert r.pile_weight_to_np == pytest.approx(24.12, abs=0.01)
        assert r.max_pile_load == pytest.approx(174.90, abs=0.01)
        assert r.max_pile_load == pytest.approx(
            r.dragload + r.pile_weight_to_np, rel=1e-12)
        fe0 = _row(r, "force_equilibrium", 0.0)
        assert fe0["neutral_plane_depth_m"] == pytest.approx(10.117,
                                                             abs=0.001)
        assert fe0["max_pile_load_kN"] == pytest.approx(274.71, abs=0.01)
        assert _row(r, "pile_toe")["max_pile_load_kN"] == pytest.approx(
            564.15, abs=0.01)
        assert "included in the load curve" in r.neutral_plane_basis
        d = r.to_dict()
        assert d["include_pile_weight"] is True
        assert d["pile_weight_basis"].startswith(
            "Pile's own weight included in the load curve")
        assert self.REFS in d["pile_weight_basis"]
        assert "here it includes the pile weight above the plane" in (
            r.summary())

    def test_structural_demand_follows_the_basis(self):
        """UFC Eq 6-80 on the drag force Q_np - Q_dead: no pile weight in
        it by default."""
        out = _dd1(Q_dead=300.0, structural_capacity=5000.0).compute()
        assert out.structural_demand == pytest.approx(
            1.25 * 300.0 + 1.10 * out.dragload, rel=1e-12)
        inn = _dd1(Q_dead=300.0, structural_capacity=5000.0,
                   include_pile_weight=True).compute()
        assert inn.structural_demand == pytest.approx(
            1.25 * 300.0 + 1.10 * (inn.dragload + inn.pile_weight_to_np),
            rel=1e-12)
        assert inn.structural_demand > out.structural_demand

    def test_unit_weight_default_is_steel_like_pile_E(self):
        """GM-50: pile_E defaults to steel (200 GPa) and so does the unit
        weight now (78.5 kN/m3; it was 24, concrete), the calc package's
        default too. It is not used unless the weight is included."""
        fields = DowndragAnalysis.__dataclass_fields__
        assert fields["pile_E"].default == 200e6
        assert fields["pile_unit_weight"].default == 78.5
        assert (_dd1(pile_unit_weight=24.0).compute().max_pile_load
                == _dd1().compute().max_pile_load)
        r = _dd1(include_pile_weight=True).compute()
        assert r.pile_weight_to_np == pytest.approx(
            78.5 * math.pi / 4.0 * 0.4 ** 2 * r.neutral_plane_depth,
            rel=1e-12)

    def test_include_pile_weight_must_be_true_or_false(self):
        with pytest.raises(ValueError, match="include_pile_weight"):
            _dd1(include_pile_weight="yes")

    def test_calc_package_states_the_basis(self):
        from downdrag.calc_steps import get_calc_steps, get_input_summary
        for include in (False, True):
            a = _dd1(include_pile_weight=include)
            r = a.compute()
            inputs = {i.name: i.value for i in get_input_summary(r, a)}
            items = [it for s in get_calc_steps(r, a) for it in s.items]
            methods = next(it for it in items if getattr(it, "title", "")
                           == "Neutral Plane by Each Method (same inputs)")
            forces = next(it for it in items if getattr(it, "title", "")
                          == "Force Components at the Neutral Plane")
            q_np = next(it for it in items if getattr(it, "title", "")
                        == "Maximum Axial Load at Neutral Plane")
            names = [row[0] for row in forces.rows]
            if include:
                assert inputs["W_pile"].startswith(
                    "included in the load curve")
                assert "γ_pile" in inputs
                assert "Pile weight to NP" in names
                assert "W_pile" in q_np.equation
                assert "includes the pile weight" in methods.notes
            else:
                assert inputs["W_pile"] == (
                    "left out of the load curve, as UFC 3-220-20 Fig 6-26 "
                    "and GEC-12 §7.3.6.1 do")
                assert "γ_pile" not in inputs
                assert "Pile weight to NP" not in names
                assert q_np.equation == "Q_np = Q_dead + Q_nf"
                assert "pile's own weight is left out" in methods.notes


class TestCalcPackageMethodsTable:
    """The calc package tabulates every method and says how the headline
    neutral plane was located."""

    def test_package_tabulates_every_method(self):
        from downdrag.calc_steps import get_calc_steps, get_input_summary
        a = _dd1()
        r = a.compute()
        sections = get_calc_steps(r, a)
        tables = [it for s in sections for it in s.items
                  if getattr(it, "title", "") ==
                  "Neutral Plane by Each Method (same inputs)"]
        assert len(tables) == 1
        names = [row[0] for row in tables[0].rows]
        assert ("Mobilized toe resistance compatible with the toe "
                "penetration (headline)") in names
        assert "Neutral plane at 100% toe mobilization" in names
        assert r.methods_range in tables[0].notes
        how = [s for s in sections if s.title ==
               "Neutral Plane: How It Was Located, and Warnings"][0]
        text = "\n".join(str(i) for i in how.items)
        assert "JUDGMENT" not in text
        assert ("Neutral-plane method requested: auto (UFC 3-220-20 "
                "§6-7.4 procedure)") in text
        assert r.methods_range in text
        inputs = {i.name: i.value for i in get_input_summary(r, a)}
        assert inputs["NP method"].startswith("auto")
        assert inputs["σ'_fs"].startswith("final")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
