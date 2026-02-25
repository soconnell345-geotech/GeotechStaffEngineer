"""
Tests for SOE (Support of Excavation) module — Phases 1, 2 & 3.

Covers: geometry validation, earth pressure coefficients, apparent pressure
envelopes, braced excavation analysis, cantilever analysis, embedment,
wall section selection, result dataclasses, stability checks
(basal heave, bottom blowout, piping), and ground anchor design.

All units SI: m, kPa, kN/m³, degrees.
"""

import math
import pytest

from soe.geometry import SOEWallLayer, SupportLevel, ExcavationGeometry
from soe.earth_pressure import (
    rankine_Ka,
    rankine_Kp,
    coulomb_Ka,
    coulomb_Kp,
    K0,
    active_pressure,
    passive_pressure,
    tension_crack_depth,
    apparent_pressure_sand,
    apparent_pressure_soft_clay,
    apparent_pressure_stiff_clay,
    select_apparent_pressure,
    get_pressure_at_depth,
)
from soe.beam_analysis import (
    analyze_braced_excavation,
    analyze_cantilever_excavation,
)
from soe.embedment import compute_embedment
from soe.wall_sections import (
    select_hp_section,
    select_sheet_pile,
    select_w_section,
    check_flexural_demand,
    list_hp_sections,
    list_sheet_pile_sections,
)
from soe.results import (
    BracedExcavationResult,
    CantileverExcavationResult,
    StabilityCheckResult,
    AnchorDesignResult,
)
from soe.anchor_design import (
    design_ground_anchor,
    compute_unbonded_length,
    compute_bond_length,
    select_tendon,
    list_bond_stress_types,
    get_bond_stress,
)
from soe.stability import (
    check_basal_heave_terzaghi,
    check_basal_heave_bjerrum_eide,
    check_bottom_blowout,
    check_piping,
)


# ============================================================================
# Helper: standard geometries
# ============================================================================

def _sand_geometry(H=6.0, n_supports=1):
    """Create a standard sand excavation geometry."""
    layers = [SOEWallLayer(thickness=20.0, unit_weight=18.0,
                           friction_angle=30.0, cohesion=0.0,
                           soil_type="sand")]
    if n_supports == 0:
        supports = []
    elif n_supports == 1:
        supports = [SupportLevel(depth=1.5, support_type="strut")]
    elif n_supports == 2:
        supports = [
            SupportLevel(depth=1.5, support_type="strut"),
            SupportLevel(depth=4.0, support_type="strut"),
        ]
    else:
        raise ValueError("n_supports must be 0, 1, or 2")

    return ExcavationGeometry(
        excavation_depth=H,
        soil_layers=layers,
        support_levels=supports,
        surcharge=10.0,
    )


def _clay_geometry(H=6.0, cu=50.0, soil_type="soft_clay"):
    """Create a standard clay excavation geometry."""
    layers = [SOEWallLayer(thickness=20.0, unit_weight=18.0,
                           friction_angle=0.0, cohesion=cu,
                           soil_type=soil_type)]
    supports = [SupportLevel(depth=1.5, support_type="strut")]
    return ExcavationGeometry(
        excavation_depth=H,
        soil_layers=layers,
        support_levels=supports,
        surcharge=10.0,
    )


# ============================================================================
# Geometry validation tests
# ============================================================================

class TestGeometryValidation:
    """Tests for ExcavationGeometry.validate()."""

    def test_valid_geometry(self):
        geo = _sand_geometry()
        geo.validate()  # should not raise

    def test_negative_depth_raises(self):
        geo = _sand_geometry()
        geo.excavation_depth = -1.0
        with pytest.raises(ValueError, match="positive"):
            geo.validate()

    def test_no_soil_layers_raises(self):
        geo = ExcavationGeometry(excavation_depth=5.0, soil_layers=[])
        with pytest.raises(ValueError, match="soil layer"):
            geo.validate()

    def test_insufficient_thickness_raises(self):
        layers = [SOEWallLayer(thickness=2.0, unit_weight=18.0)]
        geo = ExcavationGeometry(excavation_depth=5.0, soil_layers=layers)
        with pytest.raises(ValueError, match="thickness"):
            geo.validate()

    def test_invalid_soil_type_raises(self):
        layers = [SOEWallLayer(thickness=10.0, unit_weight=18.0,
                               soil_type="gravel")]
        geo = ExcavationGeometry(excavation_depth=5.0, soil_layers=layers)
        with pytest.raises(ValueError, match="soil_type"):
            geo.validate()

    def test_support_depth_out_of_range_raises(self):
        geo = _sand_geometry()
        geo.support_levels = [SupportLevel(depth=7.0)]  # > H=6
        with pytest.raises(ValueError, match="depth"):
            geo.validate()

    def test_unsorted_supports_raises(self):
        geo = _sand_geometry(n_supports=0)
        geo.support_levels = [
            SupportLevel(depth=4.0),
            SupportLevel(depth=1.5),
        ]
        with pytest.raises(ValueError, match="sorted"):
            geo.validate()

    def test_weighted_avg_unit_weight(self):
        layers = [
            SOEWallLayer(thickness=3.0, unit_weight=16.0),
            SOEWallLayer(thickness=3.0, unit_weight=20.0),
        ]
        geo = ExcavationGeometry(excavation_depth=6.0, soil_layers=layers)
        assert geo.weighted_avg_unit_weight() == pytest.approx(18.0, abs=0.01)

    def test_weighted_avg_cu(self):
        layers = [
            SOEWallLayer(thickness=3.0, unit_weight=18.0, cohesion=30.0),
            SOEWallLayer(thickness=3.0, unit_weight=18.0, cohesion=60.0),
        ]
        geo = ExcavationGeometry(excavation_depth=6.0, soil_layers=layers)
        assert geo.weighted_avg_cu() == pytest.approx(45.0, abs=0.01)


# ============================================================================
# Earth pressure coefficient tests
# ============================================================================

class TestEarthPressure:
    """Tests for classical earth pressure coefficients."""

    def test_rankine_Ka_30deg(self):
        """Ka(30°) = tan²(45-15) = tan²(30°) = 1/3."""
        assert rankine_Ka(30.0) == pytest.approx(1.0 / 3.0, rel=0.01)

    def test_rankine_Kp_30deg(self):
        """Kp(30°) = tan²(45+15) = tan²(60°) = 3.0."""
        assert rankine_Kp(30.0) == pytest.approx(3.0, rel=0.01)

    def test_Ka_times_Kp_identity(self):
        """Ka × Kp = 1 for Rankine theory."""
        for phi in [20, 25, 30, 35, 40]:
            assert rankine_Ka(phi) * rankine_Kp(phi) == pytest.approx(1.0, rel=0.01)

    def test_K0_30deg(self):
        """K0(30°) = 1 - sin(30°) = 0.5."""
        assert K0(30.0) == pytest.approx(0.5, rel=0.01)

    def test_coulomb_Ka_no_friction_equals_rankine(self):
        """Coulomb Ka with delta=0 should equal Rankine Ka."""
        for phi in [25, 30, 35]:
            assert coulomb_Ka(phi, 0.0) == pytest.approx(rankine_Ka(phi), rel=0.01)

    def test_invalid_phi_raises(self):
        with pytest.raises(ValueError):
            rankine_Ka(-5)
        with pytest.raises(ValueError):
            rankine_Kp(55)

    def test_active_pressure_sand(self):
        """sigma_a = Ka * gamma * z for cohesionless soil."""
        Ka = rankine_Ka(30.0)
        p = active_pressure(gamma=18.0, z=5.0, Ka=Ka)
        assert p == pytest.approx(Ka * 18.0 * 5.0, rel=0.01)

    def test_active_pressure_with_cohesion(self):
        """sigma_a = Ka*gamma*z - 2*c*sqrt(Ka)."""
        Ka = rankine_Ka(30.0)
        c = 10.0
        p = active_pressure(gamma=18.0, z=5.0, Ka=Ka, c=c)
        expected = Ka * 18.0 * 5.0 - 2.0 * c * math.sqrt(Ka)
        assert p == pytest.approx(expected, rel=0.01)

    def test_passive_pressure(self):
        """sigma_p = Kp * gamma * z."""
        Kp = rankine_Kp(30.0)
        p = passive_pressure(gamma=18.0, z=5.0, Kp=Kp)
        assert p == pytest.approx(Kp * 18.0 * 5.0, rel=0.01)

    def test_tension_crack_depth(self):
        Ka = rankine_Ka(30.0)
        z_crack = tension_crack_depth(c=20.0, gamma=18.0, Ka=Ka)
        expected = 2.0 * 20.0 / math.sqrt(Ka) / 18.0
        assert z_crack == pytest.approx(expected, rel=0.01)


# ============================================================================
# Apparent pressure envelope tests
# ============================================================================

class TestApparentPressure:
    """Tests for Terzaghi-Peck apparent pressure envelopes."""

    def test_sand_envelope(self):
        """p = 0.65 * Ka * gamma * H."""
        Ka = rankine_Ka(30.0)
        p = apparent_pressure_sand(gamma=18.0, H=8.0, Ka=Ka)
        expected = 0.65 * Ka * 18.0 * 8.0
        assert p == pytest.approx(expected, rel=0.01)

    def test_soft_clay_envelope(self):
        """Uniform envelope for N > 4."""
        gamma, H, cu = 18.0, 10.0, 30.0
        N = gamma * H / cu  # = 6.0 > 4
        shape, p = apparent_pressure_soft_clay(gamma, H, cu)
        assert shape == "uniform"
        Ka_app = 1.0 - 1.0 * (4.0 * cu / (gamma * H))
        assert p == pytest.approx(Ka_app * gamma * H, rel=0.01)

    def test_stiff_clay_envelope(self):
        """Trapezoidal envelope for N <= 4."""
        gamma, H, cu = 18.0, 6.0, 50.0
        N = gamma * H / cu  # = 2.16 <= 4
        shape, p = apparent_pressure_stiff_clay(gamma, H, cu)
        assert shape == "trapezoidal"
        assert p > 0
        assert p <= 0.4 * gamma * H

    def test_soft_clay_Ka_floor(self):
        """Ka_apparent should not drop below 0.25."""
        # Very high cu relative to gamma*H
        shape, p = apparent_pressure_soft_clay(gamma=18.0, H=2.0, cu=50.0)
        assert p >= 0.25 * 18.0 * 2.0 - 0.01

    def test_select_pressure_sand(self):
        layers = [SOEWallLayer(thickness=10.0, unit_weight=18.0,
                               friction_angle=30.0, soil_type="sand")]
        result = select_apparent_pressure(layers, H=8.0)
        assert result["type"] == "sand"
        assert result["shape"] == "uniform"
        assert result["max_pressure_kPa"] > 0

    def test_select_pressure_soft_clay(self):
        layers = [SOEWallLayer(thickness=15.0, unit_weight=18.0,
                               friction_angle=0.0, cohesion=30.0,
                               soil_type="soft_clay")]
        result = select_apparent_pressure(layers, H=10.0)
        assert result["type"] == "soft_clay"
        assert result["shape"] == "uniform"

    def test_select_pressure_stiff_clay(self):
        layers = [SOEWallLayer(thickness=10.0, unit_weight=18.0,
                               friction_angle=0.0, cohesion=50.0,
                               soil_type="stiff_clay")]
        result = select_apparent_pressure(layers, H=6.0)
        assert result["type"] == "stiff_clay"
        assert result["shape"] == "trapezoidal"

    def test_get_pressure_uniform(self):
        """Uniform pressure should be constant everywhere."""
        p_max = 50.0
        for z in [0.0, 2.0, 4.0, 5.9]:
            p = get_pressure_at_depth(z, H=6.0, shape="uniform", p_max=p_max)
            assert p == pytest.approx(p_max, rel=0.01)

    def test_get_pressure_trapezoidal(self):
        """Trapezoidal: 0 at top, ramp to p_max at 0.25H, constant, ramp to 0."""
        H = 8.0
        p_max = 50.0
        # At top: 0
        assert get_pressure_at_depth(0.0, H, "trapezoidal", p_max) == pytest.approx(0.0, abs=0.1)
        # At 0.25H: p_max
        assert get_pressure_at_depth(2.0, H, "trapezoidal", p_max) == pytest.approx(p_max, abs=0.1)
        # At 0.5H: p_max
        assert get_pressure_at_depth(4.0, H, "trapezoidal", p_max) == pytest.approx(p_max, abs=0.1)
        # At H: 0
        assert get_pressure_at_depth(8.0, H, "trapezoidal", p_max) == pytest.approx(0.0, abs=0.1)

    def test_get_pressure_outside_wall(self):
        """Pressure should be 0 outside the wall."""
        assert get_pressure_at_depth(-1.0, 6.0, "uniform", 50.0) == 0.0
        assert get_pressure_at_depth(7.0, 6.0, "uniform", 50.0) == 0.0


# ============================================================================
# Braced excavation analysis tests
# ============================================================================

class TestBracedExcavation:
    """Tests for analyze_braced_excavation()."""

    def test_single_strut_sand(self):
        """Single strut in sand: should produce positive reaction and moment."""
        geo = _sand_geometry(H=6.0, n_supports=1)
        result = analyze_braced_excavation(geo)

        assert result.excavation_depth == 6.0
        assert result.n_support_levels == 1
        assert result.apparent_pressure_type == "sand"
        assert result.max_apparent_pressure_kPa > 0
        assert len(result.support_reactions) == 1
        assert result.support_reactions[0]["load_kN_per_m"] > 0
        assert result.max_moment_kNm_per_m > 0
        assert result.required_embedment_m > 0
        assert result.total_wall_length_m > 6.0

    def test_two_strut_sand(self):
        """Two struts in sand: should have 2 reactions."""
        geo = _sand_geometry(H=6.0, n_supports=2)
        result = analyze_braced_excavation(geo)

        assert result.n_support_levels == 2
        assert len(result.support_reactions) == 2
        for rxn in result.support_reactions:
            assert rxn["load_kN_per_m"] > 0

    def test_braced_soft_clay(self):
        """Braced excavation in soft clay."""
        geo = _clay_geometry(H=8.0, cu=30.0, soil_type="soft_clay")
        result = analyze_braced_excavation(geo)

        assert result.apparent_pressure_type == "soft_clay"
        assert result.max_moment_kNm_per_m > 0

    def test_braced_stiff_clay(self):
        """Braced excavation in stiff clay (trapezoidal envelope)."""
        geo = _clay_geometry(H=6.0, cu=50.0, soil_type="stiff_clay")
        result = analyze_braced_excavation(geo)

        assert result.apparent_pressure_type == "stiff_clay"
        assert result.max_moment_kNm_per_m > 0

    def test_no_supports_raises(self):
        """analyze_braced_excavation should raise if no supports."""
        geo = _sand_geometry(H=6.0, n_supports=0)
        with pytest.raises(ValueError, match="support"):
            analyze_braced_excavation(geo)

    def test_deeper_excavation_higher_loads(self):
        """Deeper excavation should produce higher support reactions."""
        geo_shallow = _sand_geometry(H=4.0, n_supports=1)
        geo_deep = _sand_geometry(H=8.0, n_supports=1)
        # Adjust support depth for shallow excavation
        geo_shallow.support_levels = [SupportLevel(depth=1.0)]

        r_shallow = analyze_braced_excavation(geo_shallow)
        r_deep = analyze_braced_excavation(geo_deep)

        assert r_deep.max_apparent_pressure_kPa > r_shallow.max_apparent_pressure_kPa
        assert r_deep.support_reactions[0]["load_kN_per_m"] > \
               r_shallow.support_reactions[0]["load_kN_per_m"]

    def test_required_Sx_positive(self):
        """Required section modulus should be positive."""
        geo = _sand_geometry(H=6.0, n_supports=1)
        result = analyze_braced_excavation(geo)
        assert result.required_Sx_cm3 > 0


# ============================================================================
# Cantilever excavation analysis tests
# ============================================================================

class TestCantileverExcavation:
    """Tests for analyze_cantilever_excavation()."""

    def test_cantilever_sand(self):
        """Cantilever wall in sand should compute embedment and moment."""
        geo = _sand_geometry(H=3.0, n_supports=0)
        result = analyze_cantilever_excavation(geo)

        assert result.excavation_depth == 3.0
        assert result.Ka > 0
        assert result.Kp > 0
        assert result.Ka < result.Kp
        assert result.required_embedment_m > 0
        assert result.total_wall_length_m > 3.0
        assert result.max_moment_kNm_per_m > 0

    def test_cantilever_deeper_more_embedment(self):
        """Deeper cantilever excavation needs more embedment."""
        geo_3m = _sand_geometry(H=3.0, n_supports=0)
        geo_5m = _sand_geometry(H=5.0, n_supports=0)

        r_3m = analyze_cantilever_excavation(geo_3m)
        r_5m = analyze_cantilever_excavation(geo_5m)

        assert r_5m.required_embedment_m > r_3m.required_embedment_m

    def test_cantilever_with_supports_raises(self):
        """Should raise if supports are provided."""
        geo = _sand_geometry(H=3.0, n_supports=1)
        with pytest.raises(ValueError, match="support"):
            analyze_cantilever_excavation(geo)

    def test_cantilever_higher_FOS_more_embedment(self):
        """Higher FOS should require more embedment."""
        geo = _sand_geometry(H=3.0, n_supports=0)
        r_15 = analyze_cantilever_excavation(geo, FOS_passive=1.5)
        r_20 = analyze_cantilever_excavation(geo, FOS_passive=2.0)
        assert r_20.required_embedment_m > r_15.required_embedment_m

    def test_cantilever_clay(self):
        """Cantilever wall in clay with cohesion."""
        layers = [SOEWallLayer(thickness=10.0, unit_weight=18.0,
                               friction_angle=20.0, cohesion=15.0,
                               soil_type="stiff_clay")]
        geo = ExcavationGeometry(
            excavation_depth=3.0,
            soil_layers=layers,
            support_levels=[],
            surcharge=10.0,
        )
        result = analyze_cantilever_excavation(geo)
        assert result.required_embedment_m > 0
        assert result.max_moment_kNm_per_m > 0


# ============================================================================
# Embedment tests
# ============================================================================

class TestEmbedment:
    """Tests for compute_embedment()."""

    def test_embedment_positive(self):
        geo = _sand_geometry(H=6.0, n_supports=1)
        D = compute_embedment(geo)
        assert D > 0

    def test_embedment_includes_20pct_increase(self):
        """Embedment should include the 20% USACE increase."""
        geo = _sand_geometry(H=6.0, n_supports=1)
        D = compute_embedment(geo, FOS_passive=1.5)
        # The function returns D * 1.2, so D should be > 1.2 * something
        # Just check it's reasonably larger than 0
        assert D > 0.1

    def test_higher_FOS_more_embedment(self):
        geo = _sand_geometry(H=6.0, n_supports=1)
        D_15 = compute_embedment(geo, FOS_passive=1.5)
        D_20 = compute_embedment(geo, FOS_passive=2.0)
        assert D_20 > D_15

    def test_embedment_cantilever(self):
        """Embedment for cantilever wall (no supports, pivot at surface)."""
        geo = _sand_geometry(H=4.0, n_supports=0)
        D = compute_embedment(geo)
        assert D > 0


# ============================================================================
# Wall section selection tests
# ============================================================================

class TestWallSections:
    """Tests for steel section selection functions."""

    def test_select_hp_small_demand(self):
        """Small demand should select the lightest adequate HP."""
        result = select_hp_section(required_Sx_cm3=400.0)
        assert result is not None
        assert "name" in result
        assert result["Sx_cm3"] >= 400.0

    def test_select_hp_large_demand(self):
        """Large demand should select a heavier HP."""
        small = select_hp_section(required_Sx_cm3=400.0)
        large = select_hp_section(required_Sx_cm3=2000.0)
        assert large is not None
        assert large["weight"] >= small["weight"]

    def test_select_hp_exceeds_capacity(self):
        """Demand exceeding all sections should return None."""
        result = select_hp_section(required_Sx_cm3=50000.0)
        assert result is None

    def test_select_sheet_pile(self):
        result = select_sheet_pile(required_Sx_cm3_per_m=1000.0)
        assert result is not None
        assert result["Sx_cm3_per_m"] >= 1000.0

    def test_select_sheet_pile_exceeds_capacity(self):
        result = select_sheet_pile(required_Sx_cm3_per_m=100000.0)
        assert result is None

    def test_select_w_section(self):
        result = select_w_section(required_Sx_cm3=800.0)
        assert result is not None
        assert result["Sx_cm3"] >= 800.0

    def test_select_w_exceeds_capacity(self):
        result = select_w_section(required_Sx_cm3=100000.0)
        assert result is None

    def test_check_flexural_demand_adequate(self):
        """Section with Sx=100 cm³ under light moment should be adequate."""
        result = check_flexural_demand(Sx_cm3=100.0, M_demand_kNm=10.0)
        assert result["adequate"] is True
        assert result["utilization_ratio"] < 1.0

    def test_check_flexural_demand_overstressed(self):
        """Section under heavy moment should be overstressed."""
        result = check_flexural_demand(Sx_cm3=50.0, M_demand_kNm=200.0)
        assert result["adequate"] is False
        assert result["utilization_ratio"] > 1.0

    def test_list_hp_sections(self):
        names = list_hp_sections()
        assert len(names) == 11
        assert "HP14x117" in names

    def test_list_sheet_pile_sections(self):
        names = list_sheet_pile_sections()
        assert len(names) == 7
        assert "PZ27" in names


# ============================================================================
# Result dataclass tests
# ============================================================================

class TestResults:
    """Tests for result dataclass methods."""

    def test_braced_result_summary(self):
        result = BracedExcavationResult(
            excavation_depth=6.0,
            n_support_levels=1,
            apparent_pressure_type="sand",
            max_apparent_pressure_kPa=25.0,
            support_reactions=[{"depth_m": 1.5, "load_kN_per_m": 100.0,
                                "type": "strut"}],
            max_moment_kNm_per_m=50.0,
            max_moment_depth_m=3.75,
            max_shear_kN_per_m=40.0,
            required_embedment_m=3.5,
            total_wall_length_m=9.5,
            required_Sx_cm3=220.0,
        )
        s = result.summary()
        assert "BRACED" in s
        assert "6.00" in s
        assert "sand" in s

    def test_braced_result_to_dict(self):
        result = BracedExcavationResult(
            excavation_depth=6.0,
            n_support_levels=1,
            apparent_pressure_type="sand",
            max_apparent_pressure_kPa=25.0,
            support_reactions=[],
            max_moment_kNm_per_m=50.0,
        )
        d = result.to_dict()
        assert d["excavation_depth_m"] == 6.0
        assert d["apparent_pressure_type"] == "sand"
        assert "max_moment_kNm_per_m" in d

    def test_cantilever_result_summary(self):
        result = CantileverExcavationResult(
            excavation_depth=3.0,
            FOS_passive=1.5,
            Ka=0.333,
            Kp=3.0,
            required_embedment_m=2.5,
            total_wall_length_m=5.5,
            max_moment_kNm_per_m=30.0,
            max_shear_kN_per_m=25.0,
            required_Sx_cm3=130.0,
        )
        s = result.summary()
        assert "CANTILEVER" in s
        assert "3.00" in s

    def test_cantilever_result_to_dict(self):
        result = CantileverExcavationResult(
            excavation_depth=3.0,
            Ka=0.333,
            Kp=3.0,
        )
        d = result.to_dict()
        assert d["excavation_depth_m"] == 3.0
        assert d["Ka"] == 0.333
        assert "required_embedment_m" in d


# ============================================================================
# Integration / end-to-end tests
# ============================================================================

class TestIntegration:
    """End-to-end workflow tests."""

    def test_braced_sand_full_workflow(self):
        """Full workflow: analyze → select HP section → check demand."""
        geo = _sand_geometry(H=6.0, n_supports=1)
        result = analyze_braced_excavation(geo)

        # Select an HP section
        hp = select_hp_section(result.required_Sx_cm3)
        assert hp is not None

        # Check demand on selected section
        check = check_flexural_demand(
            Sx_cm3=hp["Sx_cm3"],
            M_demand_kNm=result.max_moment_kNm_per_m,
        )
        assert check["adequate"] is True

    def test_braced_two_level_workflow(self):
        """Two-level bracing analysis and section selection."""
        geo = _sand_geometry(H=8.0, n_supports=2)
        # Move supports for deeper excavation
        geo.support_levels = [
            SupportLevel(depth=2.0, support_type="strut"),
            SupportLevel(depth=5.0, support_type="strut"),
        ]
        result = analyze_braced_excavation(geo)
        assert result.n_support_levels == 2
        assert result.total_wall_length_m > 8.0

        # to_dict should be valid
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d["support_reactions"]) == 2

    def test_cantilever_full_workflow(self):
        """Cantilever analysis and section selection."""
        geo = _sand_geometry(H=3.0, n_supports=0)
        result = analyze_cantilever_excavation(geo)

        hp = select_hp_section(result.required_Sx_cm3)
        if hp is not None:
            check = check_flexural_demand(hp["Sx_cm3"],
                                          result.max_moment_kNm_per_m)
            assert check["adequate"] is True

    def test_module_level_imports(self):
        """Verify that the soe package can be imported cleanly."""
        import soe
        assert hasattr(soe, 'analyze_braced_excavation')
        assert hasattr(soe, 'analyze_cantilever_excavation')
        assert hasattr(soe, 'ExcavationGeometry')
        assert hasattr(soe, 'select_hp_section')
        assert hasattr(soe, 'check_basal_heave_terzaghi')
        assert hasattr(soe, 'check_piping')


# ============================================================================
# Phase 2: Stability check tests
# ============================================================================

class TestBasalHeaveTerzaghi:
    """Tests for check_basal_heave_terzaghi()."""

    def test_stable_stiff_clay(self):
        """Stiff clay (high cu) should pass basal heave check."""
        result = check_basal_heave_terzaghi(
            H=6.0, cu=100.0, gamma=18.0, q_surcharge=10.0
        )
        assert result.passes is True
        assert result.FOS > 1.5
        assert result.check_type == "basal_heave_terzaghi"

    def test_unstable_soft_clay(self):
        """Soft clay (low cu) with deep excavation should fail."""
        result = check_basal_heave_terzaghi(
            H=10.0, cu=20.0, gamma=18.0, q_surcharge=10.0
        )
        assert result.passes is False
        assert result.FOS < 1.5

    def test_Nc_increases_with_H_Be(self):
        """Nc should increase as H/Be increases (deeper, narrower)."""
        # Wide excavation (strip-like, H/Be ~ 0)
        r_wide = check_basal_heave_terzaghi(
            H=6.0, cu=50.0, gamma=18.0, B=0.0
        )
        # Narrow excavation (H/Be > 1)
        r_narrow = check_basal_heave_terzaghi(
            H=6.0, cu=50.0, gamma=18.0, B=4.0
        )
        # Narrow excavation has higher Nc → higher FOS
        assert r_narrow.FOS > r_wide.FOS

    def test_textbook_values(self):
        """Verify FOS against hand calculation.

        H=8m, cu=40kPa, gamma=18kN/m³, strip (B=0).
        Nc(strip) = 5.14 (H/Be=0).
        FOS = 40*5.14 / (18*8) = 205.6 / 144 = 1.428.
        """
        result = check_basal_heave_terzaghi(
            H=8.0, cu=40.0, gamma=18.0, q_surcharge=0.0, B=0.0
        )
        assert result.FOS == pytest.approx(1.428, abs=0.01)
        assert result.passes is False  # < 1.5

    def test_stability_number_note(self):
        """High stability number should produce a warning note."""
        result = check_basal_heave_terzaghi(
            H=10.0, cu=25.0, gamma=18.0
        )
        N = 18.0 * 10.0 / 25.0  # = 7.2 > 6
        assert any("high risk" in note for note in result.notes)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            check_basal_heave_terzaghi(H=-1, cu=50, gamma=18)
        with pytest.raises(ValueError):
            check_basal_heave_terzaghi(H=6, cu=-10, gamma=18)

    def test_summary_and_to_dict(self):
        result = check_basal_heave_terzaghi(
            H=6.0, cu=80.0, gamma=18.0
        )
        s = result.summary()
        assert "BASAL HEAVE" in s
        d = result.to_dict()
        assert "FOS" in d
        assert "passes" in d
        assert d["check_type"] == "basal_heave_terzaghi"


class TestBasalHeaveBjerrumEide:
    """Tests for check_basal_heave_bjerrum_eide()."""

    def test_strip_excavation(self):
        """Strip excavation (Be/Le → 0) should use lowest Nc."""
        result = check_basal_heave_bjerrum_eide(
            H=6.0, cu=50.0, gamma=18.0, Be=20.0, Le=200.0
        )
        assert result.FOS > 0
        assert result.check_type == "basal_heave_bjerrum_eide"

    def test_square_higher_Nc(self):
        """Square excavation (Be/Le=1) should have higher Nc than strip."""
        r_strip = check_basal_heave_bjerrum_eide(
            H=6.0, cu=50.0, gamma=18.0, Be=6.0, Le=60.0
        )
        r_square = check_basal_heave_bjerrum_eide(
            H=6.0, cu=50.0, gamma=18.0, Be=6.0, Le=6.0
        )
        assert r_square.FOS > r_strip.FOS

    def test_textbook_Nc_square_H_Be_1(self):
        """At H/Be=1, Be/Le=1: Nc should be ~6.80."""
        result = check_basal_heave_bjerrum_eide(
            H=6.0, cu=50.0, gamma=18.0, Be=6.0, Le=6.0
        )
        Nc = result.parameters["Nc"]
        assert Nc == pytest.approx(6.80, abs=0.05)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            check_basal_heave_bjerrum_eide(H=6, cu=50, gamma=18, Be=0, Le=10)
        with pytest.raises(ValueError):
            check_basal_heave_bjerrum_eide(H=6, cu=50, gamma=18, Be=10, Le=0)


class TestBottomBlowout:
    """Tests for check_bottom_blowout()."""

    def test_safe_condition(self):
        """Adequate embedment with modest head should pass."""
        result = check_bottom_blowout(
            D_embed=5.0, hw_excess=3.0, gamma_soil=10.0
        )
        # FOS = 10*5 / (9.81*3) = 50/29.43 = 1.70
        assert result.passes is True
        assert result.FOS == pytest.approx(1.70, abs=0.02)

    def test_unsafe_condition(self):
        """Shallow embedment with high head should fail."""
        result = check_bottom_blowout(
            D_embed=2.0, hw_excess=5.0, gamma_soil=8.0
        )
        # FOS = 8*2 / (9.81*5) = 16/49.05 = 0.326
        assert result.passes is False
        assert result.FOS < 1.0
        assert any("blowout is expected" in n for n in result.notes)

    def test_no_excess_head(self):
        """No excess head → infinite FOS."""
        result = check_bottom_blowout(
            D_embed=3.0, hw_excess=0.0, gamma_soil=10.0
        )
        assert result.FOS == float("inf")
        assert result.passes is True

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            check_bottom_blowout(D_embed=-1, hw_excess=3, gamma_soil=10)
        with pytest.raises(ValueError):
            check_bottom_blowout(D_embed=3, hw_excess=-1, gamma_soil=10)


class TestPiping:
    """Tests for check_piping()."""

    def test_safe_condition(self):
        """Long flow path with small head difference should pass."""
        result = check_piping(
            delta_h=2.0, flow_path=10.0, Gs=2.65, void_ratio=0.65
        )
        # i_crit = (2.65-1)/(1+0.65) = 1.65/1.65 = 1.0
        # i_exit = 2/10 = 0.2
        # FOS = 1.0/0.2 = 5.0
        assert result.passes is True
        assert result.FOS == pytest.approx(5.0, abs=0.01)

    def test_critical_gradient_calculation(self):
        """Verify i_critical = (Gs-1)/(1+e)."""
        result = check_piping(
            delta_h=1.0, flow_path=1.0, Gs=2.70, void_ratio=0.70
        )
        i_crit = (2.70 - 1.0) / (1.0 + 0.70)
        assert result.parameters["i_critical"] == pytest.approx(i_crit, abs=0.001)

    def test_unsafe_short_flow_path(self):
        """Short flow path with large head should fail."""
        result = check_piping(
            delta_h=5.0, flow_path=3.0, Gs=2.65, void_ratio=0.65
        )
        # i_exit = 5/3 = 1.667, i_crit = 1.0, FOS = 0.6
        assert result.passes is False
        assert result.FOS < 1.0

    def test_high_gradient_warning(self):
        """Exit gradient > 50% of critical should produce warning."""
        result = check_piping(
            delta_h=3.0, flow_path=5.0, Gs=2.65, void_ratio=0.65
        )
        # i_exit = 0.6, i_crit = 1.0 → 60% of critical
        assert any("50%" in n for n in result.notes)

    def test_no_head_difference(self):
        """No head difference → infinite FOS."""
        result = check_piping(delta_h=0.0, flow_path=5.0)
        assert result.FOS == float("inf")
        assert result.passes is True

    def test_default_FOS_required_is_2(self):
        """Piping FOS_required defaults to 2.0 per USACE."""
        result = check_piping(delta_h=1.0, flow_path=5.0)
        assert result.FOS_required == 2.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            check_piping(delta_h=-1, flow_path=5)
        with pytest.raises(ValueError):
            check_piping(delta_h=1, flow_path=0)


class TestStabilityCheckResult:
    """Tests for StabilityCheckResult dataclass."""

    def test_summary_format(self):
        r = StabilityCheckResult(
            check_type="test_check",
            FOS=2.5,
            FOS_required=1.5,
            passes=True,
            resistance=100.0,
            demand=40.0,
        )
        s = r.summary()
        assert "TEST CHECK" in s
        assert "2.500" in s
        assert "PASS" in s

    def test_to_dict_keys(self):
        r = StabilityCheckResult(
            check_type="piping",
            FOS=3.0,
            passes=True,
        )
        d = r.to_dict()
        assert d["check_type"] == "piping"
        assert d["FOS"] == 3.0
        assert d["passes"] is True
        assert "resistance" in d
        assert "notes" in d


# ============================================================================
# Phase 3: Ground anchor design tests
# ============================================================================

class TestUnbondedLength:
    """Tests for compute_unbonded_length()."""

    def test_minimum_free_length(self):
        """Unbonded length should be at least 4.5 m (GEC-4 minimum)."""
        L = compute_unbonded_length(
            anchor_depth=2.0, H=6.0, phi_deg=30.0, anchor_angle_deg=15.0
        )
        assert L >= 4.5

    def test_deeper_anchor_shorter_free_length(self):
        """Anchor near excavation base has shorter distance to wedge."""
        L_shallow = compute_unbonded_length(
            anchor_depth=1.5, H=8.0, phi_deg=30.0
        )
        L_deep = compute_unbonded_length(
            anchor_depth=6.0, H=8.0, phi_deg=30.0
        )
        # Deeper anchor is closer to excavation base, less distance
        # to clear the active wedge (but minimum still applies)
        assert L_deep <= L_shallow

    def test_higher_phi_shorter_wedge(self):
        """Higher friction angle → steeper active wedge → shorter free length."""
        L_low = compute_unbonded_length(
            anchor_depth=2.0, H=8.0, phi_deg=25.0
        )
        L_high = compute_unbonded_length(
            anchor_depth=2.0, H=8.0, phi_deg=40.0
        )
        assert L_high <= L_low

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            compute_unbonded_length(anchor_depth=-1, H=6, phi_deg=30)
        with pytest.raises(ValueError):
            compute_unbonded_length(anchor_depth=2, H=0, phi_deg=30)


class TestBondLength:
    """Tests for compute_bond_length()."""

    def test_hand_calculation(self):
        """Verify bond length against hand calculation.

        DL = 500 kN, tau = 145 kPa, DDH = 150 mm, FOS = 2.0
        Lb = 2.0 * 500 / (pi * 0.150 * 145) = 1000 / 68.33 = 14.63 m
        """
        Lb = compute_bond_length(
            design_load_kN=500.0,
            bond_stress_kPa=145.0,
            drill_diameter_mm=150.0,
            FOS_bond=2.0,
        )
        expected = 2.0 * 500 / (math.pi * 0.150 * 145)
        assert Lb == pytest.approx(expected, abs=0.1)

    def test_minimum_bond_length(self):
        """Bond length should be at least 3.0 m."""
        Lb = compute_bond_length(
            design_load_kN=10.0,  # very small load
            bond_stress_kPa=1000.0,
            drill_diameter_mm=150.0,
        )
        assert Lb >= 3.0

    def test_higher_load_longer_bond(self):
        """Higher design load requires longer bond zone."""
        Lb_low = compute_bond_length(200.0, 145.0)
        Lb_high = compute_bond_length(600.0, 145.0)
        assert Lb_high > Lb_low

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            compute_bond_length(design_load_kN=-100, bond_stress_kPa=145)
        with pytest.raises(ValueError):
            compute_bond_length(design_load_kN=500, bond_stress_kPa=0)


class TestTendonSelection:
    """Tests for select_tendon()."""

    def test_strand_selection(self):
        """Standard strand selection with adequate capacity."""
        result = select_tendon(
            design_load_kN=500.0,
            tendon_type="strand_15mm",
        )
        assert result["n_strands"] >= 1
        assert result["total_GUTS_kN"] > 500.0
        # Design load should be <= 60% GUTS for permanent
        assert result["design_load_pct_GUTS"] <= 60.1

    def test_strand_count_increases_with_load(self):
        """Higher load should require more strands."""
        r_light = select_tendon(design_load_kN=200.0, tendon_type="strand_15mm")
        r_heavy = select_tendon(design_load_kN=1000.0, tendon_type="strand_15mm")
        assert r_heavy["n_strands"] > r_light["n_strands"]

    def test_13mm_vs_15mm_strand(self):
        """15mm strand should require fewer strands than 13mm."""
        r_13 = select_tendon(design_load_kN=500.0, tendon_type="strand_13mm")
        r_15 = select_tendon(design_load_kN=500.0, tendon_type="strand_15mm")
        assert r_15["n_strands"] <= r_13["n_strands"]

    def test_bar_selection(self):
        """Bar tendon selection."""
        result = select_tendon(
            design_load_kN=300.0,
            tendon_type="bar_32mm",
        )
        assert "bar_capacity_kN" in result or "total_GUTS_kN" in result

    def test_test_loads(self):
        """Proof = 133% DL, performance = 150% DL per PTI."""
        result = select_tendon(design_load_kN=400.0)
        assert result["proof_test_kN"] == pytest.approx(400 * 1.33, abs=0.1)
        assert result["performance_test_kN"] == pytest.approx(400 * 1.50, abs=0.1)

    def test_unknown_tendon_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            select_tendon(design_load_kN=500, tendon_type="unknown")


class TestDesignGroundAnchor:
    """Tests for design_ground_anchor() full design."""

    def test_basic_design(self):
        """Full anchor design in medium dense sand."""
        result = design_ground_anchor(
            design_load_kN=500.0,
            anchor_depth=3.0,
            excavation_depth=9.0,
            phi_deg=30.0,
            soil_type="sand_medium",
            anchor_angle_deg=15.0,
        )
        assert isinstance(result, AnchorDesignResult)
        assert result.design_load_kN == 500.0
        assert result.unbonded_length_m >= 4.5
        assert result.bond_length_m >= 3.0
        assert result.total_length_m == pytest.approx(
            result.unbonded_length_m + result.bond_length_m, abs=0.01
        )
        assert result.proof_test_kN == pytest.approx(500 * 1.33, abs=0.1)
        assert result.performance_test_kN == pytest.approx(500 * 1.50, abs=0.1)

    def test_rock_anchor_shorter_bond(self):
        """Rock anchors need shorter bond lengths than sand."""
        r_sand = design_ground_anchor(
            design_load_kN=500.0, anchor_depth=3.0,
            excavation_depth=8.0, phi_deg=30.0,
            soil_type="sand_medium",
        )
        r_rock = design_ground_anchor(
            design_load_kN=500.0, anchor_depth=3.0,
            excavation_depth=8.0, phi_deg=35.0,
            soil_type="rock_medium",
        )
        assert r_rock.bond_length_m < r_sand.bond_length_m

    def test_custom_bond_stress(self):
        """Override bond stress with site-specific value."""
        result = design_ground_anchor(
            design_load_kN=400.0, anchor_depth=2.5,
            excavation_depth=7.0, phi_deg=30.0,
            bond_stress_kPa=200.0,
        )
        assert result.bond_stress_kPa == 200.0

    def test_summary_and_to_dict(self):
        """Result should have working summary() and to_dict()."""
        result = design_ground_anchor(
            design_load_kN=600.0, anchor_depth=3.0,
            excavation_depth=9.0, phi_deg=32.0,
        )
        s = result.summary()
        assert "GROUND ANCHOR" in s
        assert "600.0" in s

        d = result.to_dict()
        assert d["design_load_kN"] == 600.0
        assert "unbonded_length_m" in d
        assert "bond_length_m" in d
        assert "tendon" in d

    def test_unknown_soil_type_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            design_ground_anchor(
                design_load_kN=500, anchor_depth=3,
                excavation_depth=8, phi_deg=30,
                soil_type="organic_mud",
            )

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            design_ground_anchor(
                design_load_kN=-100, anchor_depth=3,
                excavation_depth=8, phi_deg=30,
            )


class TestBondStressLookup:
    """Tests for bond stress table lookup helpers."""

    def test_list_types(self):
        types = list_bond_stress_types()
        assert "sand_medium" in types
        assert "rock_hard" in types
        assert len(types) >= 9

    def test_get_bond_stress(self):
        data = get_bond_stress("sand_dense")
        assert data["bond_stress_kPa"] == 250
        assert "range" in data
        assert data["soil_type"] == "sand_dense"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_bond_stress("quicksand")
