"""
Tests for geotech_common.engineering_checks module.

Test classes:
    TestCheckBearingCapacity    - Bearing capacity result checks
    TestCheckSettlement         - Settlement result checks
    TestCheckPileCapacity       - Axial pile capacity checks
    TestCheckLateralPile        - Lateral pile result checks
    TestCheckSheetPile          - Sheet pile wall checks
    TestCheckWaveEquation       - Wave equation / driving stress checks
    TestCheckPileGroup          - Pile group checks
    TestCheckFoundationSelection - Cross-module foundation selection advice
    TestCheckParameterConsistency - Soil parameter consistency checks
    TestCheckSlopeStability       - Slope stability FOS checks
"""

import pytest

from geotech_common.engineering_checks import (
    check_bearing_capacity,
    check_settlement,
    check_pile_capacity,
    check_lateral_pile,
    check_sheet_pile,
    check_wave_equation,
    check_pile_group,
    check_foundation_selection,
    check_parameter_consistency,
    check_slope_stability,
)
from geotech_common.soil_profile import SoilLayer, GroundwaterCondition, SoilProfile


# ── Helper ────────────────────────────────────────────────────────────

def _has(warnings, severity=None, keyword=None):
    """Check if any warning matches the given severity and/or keyword."""
    for w in warnings:
        sev_match = severity is None or w.startswith(severity)
        kw_match = keyword is None or keyword.lower() in w.lower()
        if sev_match and kw_match:
            return True
    return False


# ── TestCheckBearingCapacity ──────────────────────────────────────────

class TestCheckBearingCapacity:

    def test_normal_result_no_warnings(self):
        w = check_bearing_capacity(qult_kPa=300, q_allowable_kPa=100,
                                   soil_type="stiff clay")
        assert len(w) == 0

    def test_low_qult_soft_clay(self):
        w = check_bearing_capacity(qult_kPa=40, q_allowable_kPa=13,
                                   soil_type="soft clay")
        assert _has(w, "WARNING", "below typical range")

    def test_very_low_qult(self):
        w = check_bearing_capacity(qult_kPa=50, q_allowable_kPa=17,
                                   soil_type="unknown")
        assert _has(w, "WARNING", "very low")

    def test_high_qult_for_soil(self):
        w = check_bearing_capacity(qult_kPa=2500, q_allowable_kPa=833,
                                   soil_type="dense sand")
        assert _has(w, "WARNING", "above typical range")

    def test_very_high_qult_on_soil(self):
        w = check_bearing_capacity(qult_kPa=3000, q_allowable_kPa=1000,
                                   soil_type="unknown")
        assert _has(w, "WARNING", "unusually high")

    def test_rock_high_qult_no_warning(self):
        """Rock can have very high qult — should not warn."""
        w = check_bearing_capacity(qult_kPa=5000, q_allowable_kPa=1667,
                                   soil_type="rock")
        assert not _has(w, keyword="unusually high")

    def test_demand_exceeds_capacity(self):
        w = check_bearing_capacity(qult_kPa=300, q_allowable_kPa=100,
                                   applied_stress_kPa=120)
        assert _has(w, "CRITICAL", "exceeds")

    def test_demand_near_capacity(self):
        w = check_bearing_capacity(qult_kPa=300, q_allowable_kPa=100,
                                   applied_stress_kPa=85)
        assert _has(w, "WARNING", "80%")

    def test_overdesigned(self):
        w = check_bearing_capacity(qult_kPa=1000, q_allowable_kPa=333,
                                   applied_stress_kPa=50)
        assert _has(w, "INFO", "overdesigned")

    def test_low_fos(self):
        w = check_bearing_capacity(qult_kPa=200, q_allowable_kPa=100,
                                   factor_of_safety=1.5)
        assert _has(w, "WARNING", "FOS")

    def test_deep_embedment_ratio(self):
        w = check_bearing_capacity(qult_kPa=300, q_allowable_kPa=100,
                                   footing_width_m=1.0, footing_depth_m=4.0)
        assert _has(w, "INFO", "depth/width")


# ── TestCheckSettlement ───────────────────────────────────────────────

class TestCheckSettlement:

    def test_normal_settlement_no_warnings(self):
        w = check_settlement(total_settlement_mm=20, structure_type="building")
        assert len(w) == 0

    def test_settlement_exceeds_limit_bridge(self):
        w = check_settlement(total_settlement_mm=60, structure_type="bridge")
        assert _has(w, "CRITICAL", "exceeds")

    def test_settlement_near_limit(self):
        w = check_settlement(total_settlement_mm=42, structure_type="building")
        assert _has(w, "WARNING", "80%")

    def test_industrial_higher_limit(self):
        w = check_settlement(total_settlement_mm=60, structure_type="industrial")
        assert not _has(w, "CRITICAL")  # 60 < 75 limit for industrial

    def test_angular_distortion_exceeds(self):
        w = check_settlement(total_settlement_mm=30, structure_type="bridge",
                             differential_settlement_mm=20, span_m=5.0)
        # 20mm / 5000mm = 1/250, bridge limit is 1/500
        assert _has(w, "CRITICAL", "angular distortion")

    def test_angular_distortion_near_limit(self):
        w = check_settlement(total_settlement_mm=30, structure_type="building_steel",
                             differential_settlement_mm=10, span_m=4.0)
        # 10/4000 = 1/400, steel building limit is 1/300
        # 1/400 < 1/300 so OK, but check if near limit
        # Actually 1/400 is 0.0025, limit 1/300 is 0.00333, 0.0025/0.00333 = 0.75, not > 0.8
        assert not _has(w, keyword="angular distortion")

    def test_long_consolidation_time(self):
        w = check_settlement(total_settlement_mm=30, consolidation_time_years=12)
        assert _has(w, "WARNING", "wick drains")

    def test_moderate_consolidation_time(self):
        w = check_settlement(total_settlement_mm=30, consolidation_time_years=7)
        assert _has(w, "INFO", "wick drains")

    def test_high_secondary_fraction(self):
        w = check_settlement(total_settlement_mm=30, secondary_fraction=0.3)
        assert _has(w, "WARNING", "secondary")

    def test_negative_settlement(self):
        w = check_settlement(total_settlement_mm=-5)
        assert _has(w, "CRITICAL", "negative")

    def test_very_large_settlement(self):
        w = check_settlement(total_settlement_mm=250)
        assert _has(w, "WARNING", "very large")


# ── TestCheckPileCapacity ─────────────────────────────────────────────

class TestCheckPileCapacity:

    def test_normal_capacity_no_warnings(self):
        w = check_pile_capacity(capacity_kN=800, pile_type="steel_pipe",
                                pile_diameter_m=0.3, pile_length_m=15)
        assert not _has(w, "CRITICAL")

    def test_low_capacity_for_type(self):
        w = check_pile_capacity(capacity_kN=200, pile_type="steel_pipe")
        assert _has(w, "INFO", "below typical range")

    def test_high_capacity_for_type(self):
        w = check_pile_capacity(capacity_kN=3000, pile_type="steel_pipe")
        assert _has(w, "WARNING", "above typical range")

    def test_high_ld_ratio(self):
        w = check_pile_capacity(capacity_kN=800, pile_diameter_m=0.3,
                                pile_length_m=20)
        assert _has(w, "WARNING", "L/D")

    def test_low_ld_ratio(self):
        w = check_pile_capacity(capacity_kN=500, pile_diameter_m=0.6,
                                pile_length_m=4.0)
        assert _has(w, "INFO", "short rigid")

    def test_tip_concentration(self):
        w = check_pile_capacity(capacity_kN=1000, Q_skin_kN=100, Q_tip_kN=900)
        assert _has(w, "INFO", "end bearing")

    def test_skin_concentration(self):
        w = check_pile_capacity(capacity_kN=1000, Q_skin_kN=900, Q_tip_kN=100)
        assert _has(w, "INFO", "skin friction")

    def test_overloaded(self):
        w = check_pile_capacity(capacity_kN=500, applied_load_kN=250,
                                factor_of_safety=2.5)
        # Allowable = 500/2.5 = 200 kN, load 250 > 200
        assert _has(w, "CRITICAL", "exceeds")

    def test_near_capacity(self):
        w = check_pile_capacity(capacity_kN=500, applied_load_kN=185,
                                factor_of_safety=2.5)
        # Allowable = 200, 185/200 = 0.925 > 0.9
        assert _has(w, "WARNING", "utilization")

    def test_low_fos(self):
        w = check_pile_capacity(capacity_kN=500, factor_of_safety=1.5)
        assert _has(w, "WARNING", "FOS")

    def test_very_low_capacity(self):
        w = check_pile_capacity(capacity_kN=80)
        assert _has(w, "WARNING", "very low")


# ── TestCheckLateralPile ──────────────────────────────────────────────

class TestCheckLateralPile:

    def test_normal_result_no_critical(self):
        w = check_lateral_pile(deflection_mm=10, max_moment_kNm=100,
                               pile_diameter_m=0.3, pile_length_m=15)
        assert not _has(w, "CRITICAL")

    def test_non_converged_critical(self):
        w = check_lateral_pile(deflection_mm=50, max_moment_kNm=200,
                               converged=False)
        assert _has(w, "CRITICAL", "did not converge")

    def test_service_deflection_exceeds_bridge(self):
        w = check_lateral_pile(deflection_mm=30, max_moment_kNm=100,
                               structure_type="bridge", service_or_ultimate="service")
        assert _has(w, "WARNING", "exceeds")

    def test_service_deflection_ok_sign(self):
        """Sign structures allow 50mm."""
        w = check_lateral_pile(deflection_mm=40, max_moment_kNm=100,
                               structure_type="sign_structure",
                               service_or_ultimate="service")
        assert not _has(w, keyword="exceeds")

    def test_deep_max_moment(self):
        w = check_lateral_pile(deflection_mm=10, max_moment_kNm=100,
                               pile_diameter_m=0.3, max_moment_depth_m=4.0)
        # 4.0 / 0.3 = 13.3 > 10
        assert _has(w, "INFO", "unusually deep")

    def test_short_pile_warning(self):
        w = check_lateral_pile(deflection_mm=10, max_moment_kNm=100,
                               pile_diameter_m=0.5, pile_length_m=5.0)
        # 15 * 0.5 = 7.5m, 5 < 7.5
        assert _has(w, "WARNING", "insufficient")

    def test_very_large_deflection(self):
        w = check_lateral_pile(deflection_mm=120, max_moment_kNm=200)
        assert _has(w, "WARNING", "very large")

    def test_zero_deflection_info(self):
        w = check_lateral_pile(deflection_mm=0.0, max_moment_kNm=0)
        assert _has(w, "INFO", "near-zero")

    def test_ultimate_no_deflection_limit(self):
        """Ultimate load check should not flag service deflection limit."""
        w = check_lateral_pile(deflection_mm=30, max_moment_kNm=200,
                               structure_type="bridge",
                               service_or_ultimate="ultimate")
        assert not _has(w, keyword="exceeds")


# ── TestCheckSheetPile ────────────────────────────────────────────────

class TestCheckSheetPile:

    def test_normal_cantilever(self):
        w = check_sheet_pile(embedment_m=6, retained_height_m=4,
                             max_moment_kNm_per_m=150,
                             wall_type="cantilever", soil_type="sand")
        assert not _has(w, "CRITICAL")

    def test_low_embedment_cantilever_sand(self):
        w = check_sheet_pile(embedment_m=3, retained_height_m=4,
                             max_moment_kNm_per_m=150,
                             wall_type="cantilever", soil_type="sand")
        # 3/4 = 0.75 < 1.2
        assert _has(w, "WARNING", "low")

    def test_over_embedded_cantilever(self):
        w = check_sheet_pile(embedment_m=15, retained_height_m=4,
                             max_moment_kNm_per_m=150,
                             wall_type="cantilever", soil_type="sand")
        # 15/4 = 3.75 > 3.0
        assert _has(w, "INFO", "overdesigned")

    def test_cantilever_too_tall(self):
        w = check_sheet_pile(embedment_m=10, retained_height_m=7,
                             max_moment_kNm_per_m=500,
                             wall_type="cantilever")
        assert _has(w, "WARNING", "rarely practical")

    def test_low_fos_passive(self):
        w = check_sheet_pile(embedment_m=5, retained_height_m=4,
                             max_moment_kNm_per_m=150, FOS_passive=1.3)
        assert _has(w, "WARNING", "FOS")

    def test_critical_fos_passive(self):
        w = check_sheet_pile(embedment_m=5, retained_height_m=4,
                             max_moment_kNm_per_m=150, FOS_passive=1.1)
        assert _has(w, "CRITICAL", "FOS")

    def test_anchored_low_embedment(self):
        w = check_sheet_pile(embedment_m=1, retained_height_m=6,
                             max_moment_kNm_per_m=200,
                             wall_type="anchored")
        # 1/6 = 0.17 < 0.3
        assert _has(w, "WARNING", "low")

    def test_high_moment_info(self):
        w = check_sheet_pile(embedment_m=8, retained_height_m=5,
                             max_moment_kNm_per_m=600)
        assert _has(w, "INFO", "section modulus")


# ── TestCheckWaveEquation ─────────────────────────────────────────────

class TestCheckWaveEquation:

    def test_normal_steel_no_critical(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=150_000,
                                max_tension_stress_kPa=80_000, pile_type="steel")
        assert not _has(w, "CRITICAL")

    def test_steel_compression_exceeds(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=230_000,
                                max_tension_stress_kPa=80_000, pile_type="steel")
        # 0.9 * 248000 = 223200, 230000 > 223200
        assert _has(w, "CRITICAL", "compression")

    def test_steel_compression_near_limit(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=185_000,
                                max_tension_stress_kPa=80_000, pile_type="steel")
        # 0.8 * 223200 = 178560, 185000 > 178560
        assert _has(w, "WARNING", "compression")

    def test_steel_tension_exceeds(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=150_000,
                                max_tension_stress_kPa=230_000, pile_type="steel")
        assert _has(w, "CRITICAL", "tension")

    def test_concrete_compression_exceeds(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=31_000,
                                max_tension_stress_kPa=2_000, pile_type="concrete",
                                fc_prime_kPa=35_000)
        # 0.85 * 35000 = 29750, 31000 > 29750
        assert _has(w, "CRITICAL", "compression")

    def test_concrete_tension_exceeds(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=20_000,
                                max_tension_stress_kPa=5_000, pile_type="concrete",
                                fc_prime_kPa=35_000)
        # 0.7 * sqrt(35) * 1000 = 0.7 * 5.916 * 1000 = 4141 kPa, 5000 > 4141
        assert _has(w, "CRITICAL", "tension")

    def test_practical_refusal(self):
        w = check_wave_equation(blow_count=300, max_comp_stress_kPa=200_000,
                                max_tension_stress_kPa=50_000, pile_type="steel")
        assert _has(w, "CRITICAL", "refusal")

    def test_hard_driving(self):
        w = check_wave_equation(blow_count=150, max_comp_stress_kPa=180_000,
                                max_tension_stress_kPa=50_000, pile_type="steel")
        assert _has(w, "WARNING", "hard driving")

    def test_easy_driving(self):
        w = check_wave_equation(blow_count=5, max_comp_stress_kPa=80_000,
                                max_tension_stress_kPa=30_000,
                                pile_type="steel", capacity_kN=500)
        assert _has(w, "INFO", "easy driving")

    def test_timber_compression(self):
        w = check_wave_equation(blow_count=50, max_comp_stress_kPa=35_000,
                                max_tension_stress_kPa=5_000, pile_type="timber")
        assert _has(w, "CRITICAL", "timber")


# ── TestCheckPileGroup ────────────────────────────────────────────────

class TestCheckPileGroup:

    def test_normal_group_no_warnings(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=0.7, n_piles=4,
                             spacing_m=1.0, pile_diameter_m=0.3)
        assert not _has(w, "CRITICAL")

    def test_unexpected_tension(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=50,
                             max_utilization=0.7, n_piles=4)
        assert _has(w, "WARNING", "tension")

    def test_tension_designed(self):
        """No warning when piles are designed for tension."""
        w = check_pile_group(max_compression_kN=500, max_tension_kN=50,
                             max_utilization=0.7, n_piles=4,
                             design_for_tension=True)
        assert not _has(w, keyword="tension")

    def test_overstressed(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=1.1, n_piles=4)
        assert _has(w, "CRITICAL", "exceeded")

    def test_near_capacity(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=0.95, n_piles=4)
        assert _has(w, "WARNING", "low margin")

    def test_uneven_distribution(self):
        forces = [
            {"label": "P1", "axial_kN": 100},
            {"label": "P2", "axial_kN": 100},
            {"label": "P3", "axial_kN": 500},
            {"label": "P4", "axial_kN": 500},
        ]
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=0.7, n_piles=4,
                             pile_forces=forces)
        assert _has(w, "WARNING", "uneven")

    def test_close_spacing(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=0.7, n_piles=4,
                             spacing_m=0.7, pile_diameter_m=0.3)
        # 0.7 / 0.3 = 2.33 < 3.0
        assert _has(w, "WARNING", "group effects")

    def test_wide_spacing(self):
        w = check_pile_group(max_compression_kN=500, max_tension_kN=0,
                             max_utilization=0.7, n_piles=4,
                             spacing_m=3.0, pile_diameter_m=0.3)
        # 3.0 / 0.3 = 10 > 8
        assert _has(w, "INFO", "wide spacing")


# ── TestCheckFoundationSelection ──────────────────────────────────────

class TestCheckFoundationSelection:

    def test_marginal_shallow(self):
        w = check_foundation_selection(shallow_settlement_mm=42,
                                       settlement_limit_mm=50)
        assert _has(w, "WARNING", "marginal")

    def test_overdesigned_shallow(self):
        w = check_foundation_selection(shallow_fos=6.0)
        assert _has(w, "INFO", "overdesigned")

    def test_short_pile_suggestion(self):
        w = check_foundation_selection(pile_length_m=4.0)
        assert _has(w, "INFO", "shallow foundation")

    def test_soft_layer_warning(self):
        w = check_foundation_selection(has_soft_layer_below_footing=True)
        assert _has(w, "WARNING", "punch-through")

    def test_liquefiable_critical(self):
        w = check_foundation_selection(has_liquefiable_layer=True)
        assert _has(w, "CRITICAL", "liquefiable")

    def test_no_issues(self):
        w = check_foundation_selection(shallow_settlement_mm=20,
                                       settlement_limit_mm=50,
                                       shallow_fos=3.0)
        assert not _has(w, "CRITICAL")
        assert not _has(w, "WARNING")


# ── TestCheckParameterConsistency ─────────────────────────────────────

class TestCheckParameterConsistency:

    def test_cu_and_phi_both_set(self):
        layers = [SoilLayer(0, 10, "Clay", cu=50, phi=25, gamma=18.0)]
        w = check_parameter_consistency(layers=layers)
        assert _has(w, "INFO", "both cu")

    def test_cu_vs_nspt_inconsistent_high(self):
        layers = [SoilLayer(0, 10, "Clay", cu=200, N_spt=5, gamma=18.0)]
        w = check_parameter_consistency(layers=layers)
        # Expected cu from N=5: ~31 kPa, actual 200 >> 3*31
        assert _has(w, "WARNING", "inconsistent")

    def test_cu_vs_nspt_inconsistent_low(self):
        layers = [SoilLayer(0, 10, "Clay", cu=5, N_spt=30, gamma=18.0)]
        w = check_parameter_consistency(layers=layers)
        # Expected cu from N=30: ~188 kPa, actual 5 << 188/3
        assert _has(w, "WARNING", "inconsistent")

    def test_phi_vs_nspt_inconsistent(self):
        layers = [SoilLayer(0, 10, "Sand", phi=45, N_spt=10, gamma=19.0)]
        w = check_parameter_consistency(layers=layers)
        assert _has(w, "WARNING", "high friction angle")

    def test_gamma_sat_lt_gamma(self):
        layers = [SoilLayer(0, 10, "Soil", gamma=20.0, gamma_sat=18.0)]
        w = check_parameter_consistency(layers=layers)
        assert _has(w, "CRITICAL", "impossible")

    def test_high_cc(self):
        layers = [SoilLayer(0, 10, "Organic", Cc=1.5, gamma=14.0)]
        w = check_parameter_consistency(layers=layers)
        assert _has(w, "WARNING", "Cc")

    def test_consistent_parameters_no_warnings(self):
        layers = [
            SoilLayer(0, 5, "Sand", phi=33, N_spt=20, gamma=19.0),
            SoilLayer(5, 10, "Clay", cu=50, gamma=18.0),
        ]
        w = check_parameter_consistency(layers=layers)
        assert not _has(w, "CRITICAL")
        assert not _has(w, "WARNING")

    def test_accepts_profile(self):
        """Can pass a SoilProfile object directly."""
        layers = [SoilLayer(0, 10, "Clay", cu=50, gamma=18.0)]
        gw = GroundwaterCondition(depth=3.0)
        profile = SoilProfile(layers=layers, groundwater=gw)
        w = check_parameter_consistency(profile=profile)
        # Should work without error, no CRITICAL warnings expected
        assert not _has(w, "CRITICAL")

    def test_empty_layers(self):
        w = check_parameter_consistency(layers=[])
        assert len(w) == 0

    def test_none_inputs(self):
        w = check_parameter_consistency()
        assert len(w) == 0


# ── TestCheckSlopeStability ───────────────────────────────────────

class TestCheckSlopeStability:

    def test_stable_no_warnings(self):
        """FOS well above required, bishop, >= 30 slices -> no warnings."""
        w = check_slope_stability(FOS=2.0, is_stable=True, FOS_required=1.5,
                                  method="bishop", n_slices=30)
        assert len(w) == 0

    def test_fos_below_one_critical(self):
        """FOS < 1.0 -> CRITICAL."""
        w = check_slope_stability(FOS=0.85, is_stable=False)
        assert _has(w, "CRITICAL", "failure")

    def test_fos_below_required_warning(self):
        """1.0 <= FOS < FOS_required -> WARNING marginal."""
        w = check_slope_stability(FOS=1.2, is_stable=False, FOS_required=1.5)
        assert _has(w, "WARNING", "marginal")

    def test_fos_way_above_required_info(self):
        """FOS > 3x FOS_required -> INFO overdesigned."""
        w = check_slope_stability(FOS=5.0, is_stable=True, FOS_required=1.5)
        assert _has(w, "INFO", "overdesigned")

    def test_seismic_low_fos_critical(self):
        """Seismic kh>0, FOS < 1.1 -> CRITICAL."""
        w = check_slope_stability(FOS=1.05, is_stable=True, FOS_required=1.0,
                                  has_seismic=True, kh=0.15)
        assert _has(w, "CRITICAL", "seismic")

    def test_low_n_slices_info(self):
        """n_slices < 20 -> INFO discretization."""
        w = check_slope_stability(FOS=1.8, is_stable=True, n_slices=10)
        assert _has(w, "INFO", "slices")

    def test_fellenius_only_info(self):
        """Fellenius alone with no FOS_bishop -> INFO recommendation."""
        w = check_slope_stability(FOS=1.8, is_stable=True,
                                  method="fellenius", FOS_bishop=None)
        assert _has(w, "INFO", "Fellenius")

    def test_fellenius_bishop_divergence_info(self):
        """Large difference between Fellenius and Bishop -> INFO."""
        w = check_slope_stability(FOS=1.5, is_stable=True,
                                  FOS_fellenius=1.2, FOS_bishop=1.5)
        # 0.3/1.5 = 20% > 15%
        assert _has(w, "INFO", "divergence")

    def test_clean_bishop_result(self):
        """Bishop, good FOS, 30 slices, both methods comparable -> no warnings."""
        w = check_slope_stability(FOS=1.8, is_stable=True, FOS_required=1.5,
                                  method="bishop", n_slices=30,
                                  FOS_fellenius=1.7, FOS_bishop=1.8)
        assert len(w) == 0
