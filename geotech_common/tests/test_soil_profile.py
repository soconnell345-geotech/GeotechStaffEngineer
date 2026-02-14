"""
Tests for geotech_common.soil_profile module.

Test classes:
    TestSoilLayer          - Dataclass construction and validation
    TestGroundwater        - Groundwater condition handling
    TestSoilProfile        - Profile construction and layer lookup
    TestStressCalc         - Total, pore, and effective stress calculations
    TestFillCorrelations   - fill_missing_from_correlations estimation engine
    TestValidation         - Profile validation checks
    TestSoilProfileBuilder - Builder pattern (SPT, CPT, table)
    TestEstimationTracking - Measured vs estimated value tracking
    TestEdgeCases          - Boundary conditions and error handling
"""

import pytest
import warnings

from geotech_common.soil_profile import (
    SoilLayer, GroundwaterCondition, SoilProfile, SoilProfileBuilder,
    _infer_cohesive_from_uscs, _estimate_gamma, _estimate_Es, _estimate_eps50,
)
from geotech_common.water import GAMMA_W


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_simple_clay_profile(gwt=2.0):
    """3-layer profile: fill / soft clay / medium sand, GWT at gwt m."""
    layers = [
        SoilLayer(0, 2, "Sandy fill", gamma=18.0, gamma_sat=19.5,
                  phi=30, uscs="SM", is_cohesive=False),
        SoilLayer(2, 8, "Soft gray clay (CH)", gamma=16.5, gamma_sat=17.0,
                  cu=25, uscs="CH", e0=1.1, LL=55, PL=25),
        SoilLayer(8, 15, "Medium dense sand (SP)", gamma=19.0, gamma_sat=20.0,
                  phi=33, N_spt=20, uscs="SP", is_cohesive=False),
    ]
    gw = GroundwaterCondition(depth=gwt)
    return SoilProfile(layers=layers, groundwater=gw)


def _make_single_layer_profile(gamma=18.0, depth=10.0, gwt=5.0):
    """Uniform single-layer profile for simple hand checks."""
    layer = SoilLayer(0, depth, "Uniform soil", gamma=gamma, gamma_sat=gamma + 1.0)
    gw = GroundwaterCondition(depth=gwt)
    return SoilProfile(layers=[layer], groundwater=gw)


# ── TestSoilLayer ─────────────────────────────────────────────────────

class TestSoilLayer:

    def test_basic_creation(self):
        layer = SoilLayer(0, 5, "Soft clay", cu=25, gamma=17.0)
        assert layer.thickness == 5.0
        assert layer.mid_depth == 2.5
        assert layer.cu == 25
        assert layer.is_cohesive is None  # Not set, no USCS

    def test_uscs_infers_cohesive(self):
        layer = SoilLayer(0, 3, "Clay", uscs="CH")
        assert layer.is_cohesive is True

        layer2 = SoilLayer(0, 3, "Sand", uscs="SP")
        assert layer2.is_cohesive is False

    def test_invalid_depths(self):
        with pytest.raises(ValueError, match="bottom_depth"):
            SoilLayer(5, 3, "Bad layer")

    def test_equal_depths_rejected(self):
        with pytest.raises(ValueError, match="bottom_depth"):
            SoilLayer(5, 5, "Zero thickness")

    def test_negative_top_depth(self):
        with pytest.raises(ValueError, match="top_depth"):
            SoilLayer(-1, 5, "Below ground?")

    def test_gamma_sat_less_than_gamma_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SoilLayer(0, 5, "Bad weights", gamma=20.0, gamma_sat=18.0)
            assert any("gamma_sat" in str(x.message) for x in w)

    def test_all_optional_fields_none(self):
        layer = SoilLayer(0, 5, "Unknown soil")
        assert layer.cu is None
        assert layer.phi is None
        assert layer.N_spt is None
        assert layer.Cc is None
        assert layer.Es is None

    def test_estimation_tracking(self):
        layer = SoilLayer(0, 5, "Clay")
        assert not layer.is_estimated("cu")
        layer.mark_estimated("cu", "Terzaghi & Peck from N60=5")
        assert layer.is_estimated("cu")
        assert "Terzaghi" in layer.get_estimation_log()["cu"]

    def test_pi_property(self):
        layer = SoilLayer(0, 5, "Clay", LL=55, PL=25, PI=30)
        assert layer.PI == 30


# ── TestGroundwater ───────────────────────────────────────────────────

class TestGroundwater:

    def test_basic(self):
        gw = GroundwaterCondition(depth=3.0)
        assert gw.depth == 3.0
        assert gw.is_artesian is False

    def test_artesian_requires_head(self):
        with pytest.raises(ValueError, match="artesian_head"):
            GroundwaterCondition(depth=0.0, is_artesian=True)

    def test_artesian_with_head(self):
        gw = GroundwaterCondition(depth=0.0, is_artesian=True, artesian_head=3.0)
        assert gw.artesian_head == 3.0

    def test_negative_depth_rejected(self):
        with pytest.raises(ValueError):
            GroundwaterCondition(depth=-1.0)


# ── TestSoilProfile ──────────────────────────────────────────────────

class TestSoilProfile:

    def test_basic_creation(self):
        profile = _make_simple_clay_profile()
        assert len(profile.layers) == 3
        assert profile.total_depth == 15.0

    def test_layers_sorted_by_top_depth(self):
        """Layers given out of order should be sorted."""
        layers = [
            SoilLayer(5, 10, "Sand", gamma=19.0),
            SoilLayer(0, 5, "Clay", gamma=17.0),
        ]
        gw = GroundwaterCondition(depth=3.0)
        profile = SoilProfile(layers=layers, groundwater=gw)
        assert profile.layers[0].description == "Clay"
        assert profile.layers[1].description == "Sand"

    def test_gap_between_layers_rejected(self):
        layers = [
            SoilLayer(0, 5, "Clay", gamma=17.0),
            SoilLayer(6, 10, "Sand", gamma=19.0),  # Gap from 5 to 6
        ]
        gw = GroundwaterCondition(depth=3.0)
        with pytest.raises(ValueError, match="Gap or overlap"):
            SoilProfile(layers=layers, groundwater=gw)

    def test_overlap_rejected(self):
        layers = [
            SoilLayer(0, 6, "Clay", gamma=17.0),
            SoilLayer(5, 10, "Sand", gamma=19.0),  # Overlap at 5-6
        ]
        gw = GroundwaterCondition(depth=3.0)
        with pytest.raises(ValueError, match="Gap or overlap"):
            SoilProfile(layers=layers, groundwater=gw)

    def test_empty_layers_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            SoilProfile(layers=[], groundwater=GroundwaterCondition(depth=0))

    def test_layer_at_depth(self):
        profile = _make_simple_clay_profile()
        layer = profile.layer_at_depth(1.0)
        assert "fill" in layer.description.lower()

        layer = profile.layer_at_depth(5.0)
        assert "clay" in layer.description.lower()

        layer = profile.layer_at_depth(12.0)
        assert "sand" in layer.description.lower()

    def test_layer_at_depth_boundary(self):
        """Layer boundaries should be handled without error."""
        profile = _make_simple_clay_profile()
        layer = profile.layer_at_depth(2.0)
        # Should return fill or clay (both valid at boundary)
        assert layer is not None

        layer = profile.layer_at_depth(8.0)
        assert layer is not None

    def test_layer_at_depth_out_of_range(self):
        profile = _make_simple_clay_profile()
        with pytest.raises(ValueError, match="below the profile"):
            profile.layer_at_depth(20.0)

    def test_layers_in_range(self):
        profile = _make_simple_clay_profile()
        layers = profile.layers_in_range(1.0, 10.0)
        assert len(layers) == 3  # All three layers intersect 1-10

        layers = profile.layers_in_range(3.0, 7.0)
        assert len(layers) == 1  # Only soft clay
        assert "clay" in layers[0].description.lower()

    def test_summary_string(self):
        profile = _make_simple_clay_profile()
        profile.location_name = "Test Site"
        s = profile.summary()
        assert "Test Site" in s
        assert "Sandy fill" in s
        assert "clay" in s.lower()


# ── TestStressCalc ────────────────────────────────────────────────────

class TestStressCalc:

    def test_total_stress_surface(self):
        profile = _make_single_layer_profile(gamma=18.0, gwt=5.0)
        assert profile.total_stress_at_depth(0) == 0.0

    def test_total_stress_above_gwt(self):
        """sigma_v at 3m in uniform soil with gamma=18, GWT=5m: 18*3 = 54 kPa."""
        profile = _make_single_layer_profile(gamma=18.0, gwt=5.0)
        sigma = profile.total_stress_at_depth(3.0)
        assert abs(sigma - 54.0) < 0.01

    def test_total_stress_below_gwt(self):
        """sigma_v at 8m: 18*5 (above GWT) + 19*3 (below GWT) = 90 + 57 = 147 kPa."""
        profile = _make_single_layer_profile(gamma=18.0, gwt=5.0)
        sigma = profile.total_stress_at_depth(8.0)
        expected = 18.0 * 5.0 + 19.0 * 3.0  # gamma above + gamma_sat below
        assert abs(sigma - expected) < 0.01

    def test_pore_pressure_above_gwt(self):
        profile = _make_single_layer_profile(gwt=5.0)
        u = profile.pore_pressure_at_depth(3.0)
        assert u == 0.0

    def test_pore_pressure_at_gwt(self):
        profile = _make_single_layer_profile(gwt=5.0)
        u = profile.pore_pressure_at_depth(5.0)
        assert abs(u) < 0.01

    def test_pore_pressure_below_gwt(self):
        """u at 8m, GWT=5m: 9.81 * 3 = 29.43 kPa."""
        profile = _make_single_layer_profile(gwt=5.0)
        u = profile.pore_pressure_at_depth(8.0)
        assert abs(u - GAMMA_W * 3.0) < 0.01

    def test_effective_stress(self):
        """sigma_v' = sigma_v - u at 8m."""
        profile = _make_single_layer_profile(gamma=18.0, gwt=5.0)
        sigma_eff = profile.effective_stress_at_depth(8.0)
        sigma_v = 18.0 * 5.0 + 19.0 * 3.0
        u = GAMMA_W * 3.0
        assert abs(sigma_eff - (sigma_v - u)) < 0.01

    def test_effective_stress_above_gwt_equals_total(self):
        profile = _make_single_layer_profile(gamma=18.0, gwt=5.0)
        sigma_eff = profile.effective_stress_at_depth(3.0)
        sigma_tot = profile.total_stress_at_depth(3.0)
        assert abs(sigma_eff - sigma_tot) < 0.01

    def test_multilayer_stress(self):
        """Hand-check stress in 3-layer profile."""
        profile = _make_simple_clay_profile(gwt=2.0)
        # At 5m: 18.0*2 (fill above GWT) + 17.0*3 (clay below GWT) = 36+51=87 kPa total
        sigma_v = profile.total_stress_at_depth(5.0)
        # Fill: 0-2m above GWT -> gamma=18.0, thickness=2 -> 36.0
        # Clay: 2-5m below GWT -> gamma_sat=17.0, thickness=3 -> 51.0
        assert abs(sigma_v - 87.0) < 0.1

        # Pore pressure at 5m: (5-2)*9.81 = 29.43
        u = profile.pore_pressure_at_depth(5.0)
        assert abs(u - 3.0 * GAMMA_W) < 0.01

        # Effective stress
        sigma_eff = profile.effective_stress_at_depth(5.0)
        assert abs(sigma_eff - (87.0 - 3.0 * GAMMA_W)) < 0.1

    def test_effective_unit_weight_above_gwt(self):
        profile = _make_simple_clay_profile(gwt=2.0)
        gamma_eff = profile.effective_unit_weight_at_depth(1.0)
        assert gamma_eff == 18.0  # Total unit weight above GWT

    def test_effective_unit_weight_below_gwt(self):
        profile = _make_simple_clay_profile(gwt=2.0)
        gamma_eff = profile.effective_unit_weight_at_depth(5.0)
        # Soft clay gamma_sat=17.0, buoyant = 17.0 - 9.81 = 7.19
        assert abs(gamma_eff - (17.0 - GAMMA_W)) < 0.01

    def test_artesian_pore_pressure(self):
        """Artesian: piezometric surface 2m above ground."""
        layer = SoilLayer(0, 10, "Clay", gamma=18.0, gamma_sat=19.0)
        gw = GroundwaterCondition(depth=0.0, is_artesian=True, artesian_head=2.0)
        profile = SoilProfile(layers=[layer], groundwater=gw)
        # At 5m depth, depth below piezo = 5 + 2 = 7m
        u = profile.pore_pressure_at_depth(5.0)
        assert abs(u - 7.0 * GAMMA_W) < 0.01

    def test_stress_at_profile_bottom(self):
        profile = _make_single_layer_profile(gamma=18.0, depth=10.0, gwt=5.0)
        sigma = profile.total_stress_at_depth(10.0)
        expected = 18.0 * 5.0 + 19.0 * 5.0
        assert abs(sigma - expected) < 0.01


# ── TestFillCorrelations ─────────────────────────────────────────────

class TestFillCorrelations:

    def test_phi_from_n60_sand(self):
        """Sand layer with N60 should get phi estimated."""
        layers = [
            SoilLayer(0, 10, "Sand", N60=25, uscs="SP", gamma=19.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].phi is not None
        assert profile.layers[0].is_estimated("phi")
        assert any("phi" in entry for entry in log)

    def test_cu_from_n60_clay(self):
        """Clay layer with N60 should get cu estimated."""
        layers = [
            SoilLayer(0, 10, "Clay", N60=5, uscs="CH", gamma=17.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].cu is not None
        # Terzaghi & Peck: cu = 6.25 * 5 = 31.25 kPa
        assert abs(profile.layers[0].cu - 31.25) < 0.1
        assert profile.layers[0].is_estimated("cu")

    def test_n60_from_n_spt(self):
        """N60 estimated from N_spt when not provided."""
        layers = [
            SoilLayer(0, 10, "Sand", N_spt=20, uscs="SP", gamma=19.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].N60 == 20
        assert profile.layers[0].is_estimated("N60")

    def test_pi_from_ll_pl(self):
        layers = [
            SoilLayer(0, 10, "Clay", LL=55, PL=25, uscs="CH", gamma=17.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].PI == 30

    def test_cc_from_ll(self):
        """Cc ≈ 0.009*(LL-10) for cohesive soil."""
        layers = [
            SoilLayer(0, 10, "Clay", LL=55, uscs="CH", gamma=17.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        expected_Cc = 0.009 * (55 - 10)  # = 0.405
        assert abs(profile.layers[0].Cc - expected_Cc) < 0.001
        assert profile.layers[0].is_estimated("Cc")

    def test_cr_from_cc(self):
        """Cr estimated as Cc/6."""
        layers = [
            SoilLayer(0, 10, "Clay", Cc=0.4, uscs="CH", gamma=17.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert abs(profile.layers[0].Cr - 0.4 / 6.0) < 0.001

    def test_gamma_estimated_for_clay(self):
        layers = [
            SoilLayer(0, 10, "Soft clay", cu=25, uscs="CH")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].gamma is not None
        assert 14 < profile.layers[0].gamma < 24

    def test_gamma_estimated_for_sand(self):
        layers = [
            SoilLayer(0, 10, "Dense sand", N60=40, uscs="SP")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].gamma is not None
        assert profile.layers[0].gamma > 18  # Dense sand > 18 kN/m3

    def test_gamma_sat_estimated(self):
        layers = [
            SoilLayer(0, 10, "Clay", gamma=17.0, uscs="CH")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].gamma_sat is not None
        assert profile.layers[0].gamma_sat > profile.layers[0].gamma

    def test_gamma_sat_from_void_ratio(self):
        layers = [
            SoilLayer(0, 10, "Clay", gamma=17.0, e0=1.0, uscs="CH")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        # gamma_sat = (Gs + e) * gamma_w / (1+e) = (2.65+1.0)*9.81/2.0 = 17.90
        expected = (2.65 + 1.0) * GAMMA_W / 2.0
        assert abs(profile.layers[0].gamma_sat - expected) < 0.1

    def test_es_estimated_for_sand(self):
        layers = [
            SoilLayer(0, 10, "Sand", N60=25, uscs="SP", gamma=19.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].Es is not None
        assert profile.layers[0].Es > 0

    def test_es_estimated_for_clay(self):
        layers = [
            SoilLayer(0, 10, "Clay", cu=50, uscs="CH", gamma=18.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].Es is not None

    def test_eps50_estimated_for_clay(self):
        layers = [
            SoilLayer(0, 10, "Soft clay", cu=25, uscs="CH", gamma=16.5)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        log = profile.fill_missing_from_correlations()
        assert profile.layers[0].eps50 is not None
        # cu=25 -> eps50 should be around 0.010-0.020
        assert 0.005 <= profile.layers[0].eps50 <= 0.025

    def test_no_overwrite_of_measured_values(self):
        """fill_missing must NOT overwrite values that are already provided."""
        layers = [
            SoilLayer(0, 10, "Sand", phi=35, N60=10, uscs="SP", gamma=19.0)
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        profile.fill_missing_from_correlations()
        # phi was provided — should not be changed
        assert profile.layers[0].phi == 35
        assert not profile.layers[0].is_estimated("phi")

    def test_log_returns_entries(self):
        layers = [
            SoilLayer(0, 10, "Sand", N_spt=25, uscs="SP")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()
        assert len(log) > 0  # Should have at least N60, phi, gamma, gamma_sat, Es


# ── TestValidation ────────────────────────────────────────────────────

class TestValidation:

    def test_clean_profile_no_warnings(self):
        profile = _make_simple_clay_profile()
        warns = profile.validate()
        # Should have the cu+phi INFO at most (soft clay has cu, fill has phi)
        critical = [w for w in warns if w.startswith("CRITICAL")]
        assert len(critical) == 0

    def test_missing_unit_weight_warning(self):
        layers = [SoilLayer(0, 10, "Unknown soil")]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("no unit weight" in w for w in warns)

    def test_high_cu_warning(self):
        layers = [SoilLayer(0, 10, "Hard clay", cu=600, gamma=21.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("cu=600" in w for w in warns)

    def test_high_phi_warning(self):
        layers = [SoilLayer(0, 10, "Gravel?", phi=48, gamma=22.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("phi=48" in w for w in warns)

    def test_cr_gt_cc_critical(self):
        layers = [SoilLayer(0, 10, "Clay", Cc=0.3, Cr=0.5, gamma=17.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("CRITICAL" in w and "Cr" in w for w in warns)

    def test_high_cc_warning(self):
        layers = [SoilLayer(0, 10, "Peat", Cc=1.5, gamma=12.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("Cc=1.5" in w for w in warns)

    def test_gwt_below_profile_info(self):
        layers = [SoilLayer(0, 5, "Sand", gamma=19.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=10.0))
        warns = profile.validate()
        assert any("below profile" in w.lower() for w in warns)

    def test_low_unit_weight_warning(self):
        layers = [SoilLayer(0, 10, "Organic soil", gamma=12.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("gamma=12.0" in w for w in warns)

    def test_spt_refusal_info(self):
        layers = [SoilLayer(0, 10, "Rock", N_spt=150, gamma=24.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=10.0))
        warns = profile.validate()
        assert any("refusal" in w.lower() for w in warns)

    def test_high_void_ratio_warning(self):
        layers = [SoilLayer(0, 10, "Peat", e0=4.0, gamma=12.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("e0=4.0" in w for w in warns)

    def test_rqd_out_of_range(self):
        layers = [SoilLayer(0, 10, "Rock", RQD=110, gamma=25.0, is_rock=True)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=10.0))
        warns = profile.validate()
        assert any("RQD" in w and "CRITICAL" in w for w in warns)

    def test_profile_not_at_surface_warning(self):
        layers = [SoilLayer(1.0, 10, "Sand", gamma=19.0)]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        warns = profile.validate()
        assert any("not at ground surface" in w for w in warns)


# ── TestSoilProfileBuilder ───────────────────────────────────────────

class TestSoilProfileBuilder:

    def test_spt_builder(self):
        builder = SoilProfileBuilder(gwt_depth=2.0, location_name="Test Site")
        builder.add_spt_layer(0, 3, "Fill", N=8, uscs="SM", gamma=18.0)
        builder.add_spt_layer(3, 8, "Soft clay", N=3, uscs="CH", gamma=16.5)
        builder.add_spt_layer(8, 15, "Sand", N=22, uscs="SP", gamma=19.0)
        profile = builder.build()

        assert len(profile.layers) == 3
        assert profile.groundwater.depth == 2.0
        assert profile.location_name == "Test Site"
        assert profile.layers[0].N_spt == 8
        assert profile.layers[0].N60 == 8  # Builder assumes N60=N_spt

    def test_cpt_builder(self):
        builder = SoilProfileBuilder(gwt_depth=1.0)
        builder.add_cpt_layer(0, 5, "Clay", qc=500, fs=20, uscs="CH", gamma=17.0)
        builder.add_cpt_layer(5, 15, "Sand", qc=10000, fs=50, uscs="SP", gamma=20.0)
        profile = builder.build()

        assert len(profile.layers) == 2
        assert profile.layers[0].qc == 500
        assert profile.layers[1].fs == 50

    def test_from_table(self):
        profile = SoilProfileBuilder.from_table(
            gwt_depth=3.0,
            layers=[
                {"top": 0, "bottom": 5, "desc": "Stiff clay",
                 "cu": 75, "gamma": 18.5},
                {"top": 5, "bottom": 12, "desc": "Dense sand",
                 "phi": 36, "gamma": 19.5},
            ],
            location_name="Site A",
        )
        assert len(profile.layers) == 2
        assert profile.layers[0].cu == 75
        assert profile.layers[1].phi == 36
        assert profile.location_name == "Site A"

    def test_builder_with_fill_correlations(self):
        builder = SoilProfileBuilder(gwt_depth=5.0)
        builder.add_spt_layer(0, 10, "Sand", N=25, uscs="SP")
        profile = builder.build(fill_correlations=True)

        # Should have gamma and phi estimated
        assert profile.layers[0].gamma is not None
        assert profile.layers[0].phi is not None

    def test_chained_builder(self):
        """Builder methods return self for chaining."""
        profile = (
            SoilProfileBuilder(gwt_depth=3.0)
            .add_spt_layer(0, 5, "Fill", N=10, uscs="SM", gamma=18.0)
            .add_spt_layer(5, 10, "Sand", N=30, uscs="SP", gamma=20.0)
            .build()
        )
        assert len(profile.layers) == 2

    def test_from_table_with_correlations(self):
        profile = SoilProfileBuilder.from_table(
            gwt_depth=3.0,
            layers=[
                {"top": 0, "bottom": 10, "desc": "Clay", "N_spt": 5, "uscs": "CH"},
            ],
            fill_correlations=True,
        )
        assert profile.layers[0].cu is not None
        assert profile.layers[0].gamma is not None


# ── TestEstimationTracking ───────────────────────────────────────────

class TestEstimationTracking:

    def test_measured_not_flagged(self):
        """Directly provided values should never be marked estimated."""
        layers = [
            SoilLayer(0, 10, "Clay", cu=50, phi=28, gamma=18.0, uscs="CL")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=3.0))
        profile.fill_missing_from_correlations()
        assert not profile.layers[0].is_estimated("cu")
        assert not profile.layers[0].is_estimated("phi")
        assert not profile.layers[0].is_estimated("gamma")

    def test_estimated_values_logged(self):
        layers = [
            SoilLayer(0, 10, "Sand", N_spt=30, uscs="SP")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        log = profile.fill_missing_from_correlations()

        # Check estimation log on the layer
        est_log = profile.layers[0].get_estimation_log()
        assert "N60" in est_log
        assert "phi" in est_log
        assert "gamma" in est_log

    def test_summary_shows_asterisks(self):
        layers = [
            SoilLayer(0, 10, "Sand", N_spt=30, uscs="SP")
        ]
        profile = SoilProfile(layers=layers,
                              groundwater=GroundwaterCondition(depth=5.0))
        profile.fill_missing_from_correlations()
        s = profile.summary()
        assert "*" in s  # Asterisk for estimated values


# ── TestEdgeCases ─────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_layer_profile(self):
        profile = _make_single_layer_profile()
        assert len(profile.layers) == 1
        assert profile.total_depth == 10.0

    def test_zero_gwt(self):
        """GWT at ground surface — entire profile submerged."""
        layer = SoilLayer(0, 10, "Sand", gamma=18.0, gamma_sat=20.0)
        gw = GroundwaterCondition(depth=0.0)
        profile = SoilProfile(layers=[layer], groundwater=gw)

        # Pore pressure at 5m
        u = profile.pore_pressure_at_depth(5.0)
        assert abs(u - 5.0 * GAMMA_W) < 0.01

        # Total stress uses gamma_sat everywhere
        sigma = profile.total_stress_at_depth(5.0)
        assert abs(sigma - 20.0 * 5.0) < 0.01

    def test_very_deep_gwt(self):
        """GWT below profile — no pore pressure anywhere."""
        layer = SoilLayer(0, 10, "Sand", gamma=19.0, gamma_sat=20.0)
        gw = GroundwaterCondition(depth=20.0)
        profile = SoilProfile(layers=[layer], groundwater=gw)

        u = profile.pore_pressure_at_depth(5.0)
        assert u == 0.0

        # Total stress uses gamma (dry) everywhere
        sigma = profile.total_stress_at_depth(5.0)
        assert abs(sigma - 19.0 * 5.0) < 0.01

    def test_many_thin_layers(self):
        """Profile with many thin layers should work correctly."""
        layers = [
            SoilLayer(i * 0.5, (i + 1) * 0.5, f"Layer {i}", gamma=18.0)
            for i in range(20)
        ]
        gw = GroundwaterCondition(depth=5.0)
        profile = SoilProfile(layers=layers, groundwater=gw)

        assert profile.total_depth == 10.0
        sigma = profile.total_stress_at_depth(3.0)
        assert abs(sigma - 18.0 * 3.0) < 0.01

    def test_layer_with_no_gamma_raises_in_stress(self):
        """Stress calculation should raise if gamma is missing."""
        layer = SoilLayer(0, 10, "Unknown soil")
        gw = GroundwaterCondition(depth=5.0)
        profile = SoilProfile(layers=[layer], groundwater=gw)

        with pytest.raises(ValueError, match="no unit weight"):
            profile.total_stress_at_depth(5.0)

    def test_uscs_inference(self):
        assert _infer_cohesive_from_uscs("CH") is True
        assert _infer_cohesive_from_uscs("CL") is True
        assert _infer_cohesive_from_uscs("MH") is True
        assert _infer_cohesive_from_uscs("SP") is False
        assert _infer_cohesive_from_uscs("GW") is False
        assert _infer_cohesive_from_uscs("SM") is False
        assert _infer_cohesive_from_uscs("XY") is None  # Unknown

    def test_eps50_estimation_ranges(self):
        assert _estimate_eps50(10) == 0.020   # Very soft
        assert _estimate_eps50(30) == 0.010   # Soft
        assert _estimate_eps50(75) == 0.007   # Medium
        assert _estimate_eps50(150) == 0.005  # Stiff
        assert _estimate_eps50(300) == 0.004  # Very stiff

    def test_gamma_estimation_coverage(self):
        # Cohesive with cu
        layer = SoilLayer(0, 5, "Clay", cu=15, is_cohesive=True)
        assert _estimate_gamma(layer) == 16.0

        layer = SoilLayer(0, 5, "Clay", cu=120, is_cohesive=True)
        assert _estimate_gamma(layer) == 19.5

        # Granular with N60
        layer = SoilLayer(0, 5, "Sand", N60=5, is_cohesive=False)
        assert _estimate_gamma(layer) == 17.0

        layer = SoilLayer(0, 5, "Sand", N60=60, is_cohesive=False)
        assert _estimate_gamma(layer) == 21.0

        # Unknown soil, no data
        layer = SoilLayer(0, 5, "Unknown")
        assert _estimate_gamma(layer) is None

    def test_Es_estimation(self):
        # Sand with N60
        layer = SoilLayer(0, 5, "Sand", N60=20, is_cohesive=False)
        Es = _estimate_Es(layer)
        assert Es is not None
        assert Es > 0

        # Clay with cu
        layer = SoilLayer(0, 5, "Clay", cu=50, is_cohesive=True)
        Es = _estimate_Es(layer)
        assert Es is not None
        assert Es > 0

    def test_profile_start_not_at_zero(self):
        """Profile starting at depth > 0 is valid but flagged."""
        layer = SoilLayer(2.0, 10, "Sand", gamma=19.0)
        gw = GroundwaterCondition(depth=5.0)
        profile = SoilProfile(layers=[layer], groundwater=gw)
        warns = profile.validate()
        assert any("not at ground surface" in w for w in warns)
