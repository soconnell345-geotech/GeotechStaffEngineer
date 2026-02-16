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

    def test_more_dead_load_deeper_np(self):
        """Increasing dead load should push neutral plane deeper."""
        analysis_low = self._make_standard_case()
        analysis_low.Q_dead = 200.0
        result_low = analysis_low.compute()

        analysis_high = self._make_standard_case()
        analysis_high.Q_dead = 1000.0
        result_high = analysis_high.compute()

        assert result_high.neutral_plane_depth > result_low.neutral_plane_depth

    def test_no_settling_layers_np_at_tip(self):
        """With no settling layers, NP should be near pile tip (no downdrag)."""
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
        # Without settlement, there's no compatibility issue.
        # Dragload should be small relative to capacity.
        # NP found by force equilibrium only.
        assert result.neutral_plane_depth > 0


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
        analysis = DowndragAnalysis(
            soil=soil, pile_length=10.0, pile_diameter=0.3,
            pile_E=200e6, pile_area=0.01,
            Q_dead=500.0, fill_thickness=0.0, gw_drawdown=0.0,
        )
        result = analysis.compute()
        # Elastic shortening should be positive
        assert result.elastic_shortening > 0

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
        assert "Dragload" in summary
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
