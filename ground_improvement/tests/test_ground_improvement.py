"""
Tests for the ground_improvement module.

Covers aggregate piers, wick drains (GEC-13 Eq. 2-1 radial consolidation),
surcharge preloading, vibro-compaction feasibility, and the
feasibility decision-support engine.

40 tests across 8 test classes.

References:
    FHWA GEC-13 (FHWA-NHI-16-027) — radial consolidation (Eqs. 2-1 to 2-3),
    vibro-compaction (Ch. 4), aggregate columns (Ch. 5)
    UFC 3-220-10 (2026) — smear factor (Eq. 5-25)
    Barksdale & Bachus (1983) — aggregate piers
"""

import math
import pytest

from ground_improvement.aggregate_piers import (
    area_replacement_ratio, settlement_reduction_factor,
    composite_modulus, improved_bearing_capacity,
    improved_settlement, analyze_aggregate_piers,
)
from ground_improvement.wick_drains import (
    equivalent_drain_diameter, influence_diameter,
    drain_function_F, radial_time_factor,
    radial_degree_of_consolidation, combined_degree_of_consolidation,
    time_for_radial_consolidation, design_drain_spacing,
    analyze_wick_drains,
)
from ground_improvement.surcharge import (
    surcharge_settlement_at_time, surcharge_with_drains_settlement_at_time,
    required_surcharge_for_preconsolidation,
    time_to_target_consolidation, analyze_surcharge_preloading,
)
from ground_improvement.vibro import (
    vibro_feasibility, probe_spacing_fig_4_34, analyze_vibro_compaction,
)
from ground_improvement.feasibility import evaluate_feasibility
from ground_improvement.results import (
    AggregatePierResult, WickDrainResult, SurchargeResult,
    VibroResult, FeasibilityResult,
)


# ================================================================
# AGGREGATE PIERS
# ================================================================
class TestAggregatePiers:
    """Test aggregate pier calculations.

    Hand-calculated verification:
    dc=0.6m, s=2.0m, triangular:
        A_trib = sqrt(3)/2 * 4.0 = 3.4641 m²
        Ac = pi*0.36/4 = 0.28274 m²
        as = 0.28274/3.4641 = 0.08164
        SRF(n=5) = 1/(1 + 0.08164*4) = 1/1.3266 = 0.7538
    """

    def test_area_replacement_triangular(self):
        """as for triangular pattern: Ac / (sqrt(3)/2 * s^2)."""
        ar = area_replacement_ratio(0.6, 2.0, "triangular")
        Ac = math.pi / 4 * 0.6**2
        A_trib = math.sqrt(3) / 2 * 2.0**2
        assert ar == pytest.approx(Ac / A_trib, rel=1e-6)

    def test_area_replacement_square(self):
        """as for square pattern: Ac / s^2."""
        ar = area_replacement_ratio(0.6, 2.0, "square")
        Ac = math.pi / 4 * 0.6**2
        assert ar == pytest.approx(Ac / 4.0, rel=1e-6)

    def test_settlement_reduction_factor(self):
        """SRF = 1/(1 + as*(n-1)) with known values."""
        as_ratio = 0.08164
        srf = settlement_reduction_factor(as_ratio, 5.0)
        expected = 1.0 / (1.0 + 0.08164 * 4.0)
        assert srf == pytest.approx(expected, rel=1e-4)
        assert 0.7 < srf < 0.8  # sanity check

    def test_composite_modulus(self):
        """E_comp = Es * [1 + as*(n-1)], consistent with the equal-strain SRF."""
        # as=0.1, Es=5000, n=5  =>  E_comp = 5000*(1 + 0.1*4) = 7000
        E_comp = composite_modulus(0.1, 5000, 5.0)
        expected = 5000.0 * (1.0 + 0.1 * 4.0)
        assert E_comp == pytest.approx(expected, rel=1e-6)
        assert E_comp == pytest.approx(7000.0)

    def test_composite_modulus_consistency_with_srf(self):
        """E_comp = Es / SRF: composite modulus and SRF must be consistent."""
        as_ratio = 0.15
        n = 6.0
        Es = 4000.0

        srf = settlement_reduction_factor(as_ratio, n)
        E_comp = composite_modulus(as_ratio, Es, n)

        # E_comp should equal Es / SRF
        assert E_comp == pytest.approx(Es / srf, rel=1e-10)

    def test_composite_modulus_n_equals_1(self):
        """When n=1 (no improvement), E_comp = Es."""
        E_comp = composite_modulus(0.2, 5000, 1.0)
        assert E_comp == pytest.approx(5000.0, rel=1e-10)

    def test_composite_modulus_increases_with_n(self):
        """Higher n gives higher E_comp (more improvement)."""
        E_low = composite_modulus(0.1, 5000, 3.0)
        E_high = composite_modulus(0.1, 5000, 8.0)
        assert E_high > E_low > 5000.0

    def test_improved_bearing(self):
        """q_improved = q_unr * (1 + as*(n-1))."""
        q_imp = improved_bearing_capacity(100.0, 0.1, 5.0)
        expected = 100.0 * (1.0 + 0.1 * 4.0)
        assert q_imp == pytest.approx(expected, rel=1e-6)
        assert q_imp == pytest.approx(140.0)

    def test_improved_bearing_relabeled_not_priebe(self):
        """GI-2: the docstring must present 1+as(n-1) as a first-order
        stress-concentration estimate, not as the Priebe factor."""
        doc = improved_bearing_capacity.__doc__
        assert "first-order" in doc.lower()
        assert "not Priebe" in doc

    def test_full_analysis(self):
        """End-to-end analyze_aggregate_piers."""
        result = analyze_aggregate_piers(
            column_diameter=0.6,
            spacing=2.0,
            pattern="triangular",
            E_column=80000,
            E_soil=5000,
            n=5.0,
            q_unreinforced=120.0,
            S_unreinforced=80.0,
        )
        assert result.area_replacement_ratio == pytest.approx(0.0816, rel=0.01)
        assert result.settlement_reduction_factor == pytest.approx(0.754, rel=0.01)
        assert result.improved_bearing_kPa > 120.0
        assert result.settlement_improved_mm < 80.0
        assert result.settlement_improved_mm == pytest.approx(
            result.settlement_reduction_factor * 80.0, rel=1e-4
        )
        assert result.pattern == "triangular"
        # Composite modulus must be consistent with SRF: E_comp = Es / SRF
        assert result.composite_modulus_kPa == pytest.approx(
            5000.0 / result.settlement_reduction_factor, rel=1e-4
        )
        # summary and to_dict work
        assert "AGGREGATE PIER" in result.summary()
        assert "area_replacement_ratio" in result.to_dict()


# ================================================================
# WICK DRAIN THEORY
# ================================================================
class TestWickDrainTheory:
    """Test GEC-13's radial consolidation (Eqs. 2-1, 2-2; made-up inputs).

    Hand-calculated verification:
    s=1.5m triangular, dw=0.066m:
        de = 1.05*1.5 = 1.575 m
        n = 1.575/0.066 = 23.864
        F(n) = ln(23.864) - 0.75 = 3.1725 - 0.75 = 2.4225 (no smear)
    At ch=3.0 m²/yr, t=0.5yr:
        Tr = 3.0*0.5/1.575² = 1.5/2.4806 = 0.6047
        Ur = 1 - exp(-8*0.6047/2.4225) = 1 - exp(-1.9969) = 1 - 0.1358 = 86.42%
    """

    def test_equivalent_drain_diameter(self):
        """dw = 2(a+b)/pi, GEC-13 Eq. 2-3, for a 100 mm x 4 mm band."""
        dw = equivalent_drain_diameter(0.1, 0.004)
        assert dw == pytest.approx(2.0 * (0.1 + 0.004) / math.pi, rel=1e-6)
        assert dw == pytest.approx(0.0662, abs=1e-3)

    def test_influence_diameter_triangular(self):
        """de = 1.05*s for triangular."""
        de = influence_diameter(1.5, "triangular")
        assert de == pytest.approx(1.575, rel=1e-6)

    def test_influence_diameter_square(self):
        """de = 1.13*s for square."""
        de = influence_diameter(1.5, "square")
        assert de == pytest.approx(1.695, rel=1e-6)

    def test_drain_function_no_smear(self):
        """F(n) = ln(n) - 0.75 without smear."""
        n = 23.864
        F = drain_function_F(n)
        expected = math.log(n) - 0.75
        assert F == pytest.approx(expected, rel=1e-4)

    def test_drain_function_with_smear(self):
        """F(n) + F_s, F_s = (kh/ks) ln(s) as UFC 3-220-10 Eq. 5-25
        prints it (GM-291; until then Hansbo's ln(n/s) form)."""
        n = 23.864
        F_no_smear = drain_function_F(n)
        F_smear = drain_function_F(n, smear_ratio=2.5, kh_ks_ratio=3.0)
        # Smear increases F(n), which slows consolidation
        assert F_smear > F_no_smear
        expected = math.log(n) - 0.75 + 3.0 * math.log(2.5)
        assert F_smear == pytest.approx(expected, rel=1e-4)

    def test_radial_consolidation_basic(self):
        """Ur at known Tr and F(n)."""
        # From hand calc: Tr=0.6047, F_n=2.4225 -> Ur=86.4%
        Ur = radial_degree_of_consolidation(0.6047, 2.4225)
        assert Ur == pytest.approx(86.4, abs=0.5)

    def test_combined_consolidation(self):
        """U_total = 1 - (1-Uv)*(1-Ur)."""
        U_total = combined_degree_of_consolidation(40.0, 86.4)
        expected = (1.0 - 0.60 * 0.136) * 100.0  # = 91.84%
        assert U_total == pytest.approx(expected, abs=0.5)


# ================================================================
# WICK DRAIN DESIGN
# ================================================================
class TestWickDrainDesign:
    """Test wick drain spacing design and full analysis."""

    def test_time_for_radial_consolidation(self):
        """Solve for time given target Ur."""
        # Use Ur=86.4%, ch=3.0, de=1.575, F_n=2.4225
        t = time_for_radial_consolidation(86.4, ch=3.0, de=1.575, F_n=2.4225)
        assert t == pytest.approx(0.5, abs=0.02)

    def test_design_spacing_finds_solution(self):
        """Bisection finds a valid spacing for 90% U in 0.5 year."""
        result = design_drain_spacing(
            target_U=90.0, target_time=0.5,
            ch=3.0, cv=1.0, Hdr=5.0,
            dw=0.066, pattern="triangular",
        )
        assert result.U_total_percent >= 89.5  # close to target
        assert 1.0 <= result.drain_spacing_m <= 3.5

    def test_design_spacing_tight_time(self):
        """Tighter time constraint -> tighter (smaller) spacing."""
        result_loose = design_drain_spacing(
            target_U=90.0, target_time=1.0,
            ch=3.0, cv=1.0, Hdr=5.0,
        )
        result_tight = design_drain_spacing(
            target_U=90.0, target_time=0.25,
            ch=3.0, cv=1.0, Hdr=5.0,
        )
        assert result_tight.drain_spacing_m <= result_loose.drain_spacing_m

    def test_analyze_wick_drains_full(self):
        """Full analysis with time-settlement curve."""
        result = analyze_wick_drains(
            spacing=1.5, ch=3.0, cv=1.0, Hdr=5.0, time=0.5,
        )
        assert result.U_total_percent > result.Uv_percent
        assert result.U_total_percent > result.Ur_percent
        assert result.time_settlement_curve is not None
        assert len(result.time_settlement_curve) == 51  # 50 points + t=0
        # Curve should be monotonically increasing
        for i in range(1, len(result.time_settlement_curve)):
            assert result.time_settlement_curve[i][1] >= result.time_settlement_curve[i-1][1]

    def test_drains_faster_than_no_drains(self):
        """Combined U > Uv alone at the same time."""
        result = analyze_wick_drains(
            spacing=1.5, ch=3.0, cv=1.0, Hdr=5.0, time=0.5,
        )
        assert result.U_total_percent > result.Uv_percent


# ================================================================
# SURCHARGE PRELOADING
# ================================================================
class TestSurchargePreloading:
    """Test surcharge preloading analysis."""

    def test_surcharge_without_drains(self):
        """Settlement without drains matches settlement.time_rate."""
        from settlement.time_rate import settlement_at_time
        S_ref = settlement_at_time(0.5, cv=1.0, Hdr=5.0, t=1.0)
        S_test = surcharge_settlement_at_time(0.5, cv=1.0, Hdr=5.0, t=1.0)
        assert S_test == pytest.approx(S_ref, rel=1e-6)

    def test_surcharge_with_drains(self):
        """Settlement with drains > without drains at same time."""
        S_no_drain = surcharge_settlement_at_time(0.5, cv=1.0, Hdr=5.0, t=0.5)
        S_drain = surcharge_with_drains_settlement_at_time(
            0.5, cv=1.0, ch=3.0, Hdr=5.0,
            drain_spacing=1.5, dw=0.066, t=0.5,
        )
        assert S_drain > S_no_drain

    def test_required_surcharge_for_OC(self):
        """Compute surcharge for target sigma_p."""
        q = required_surcharge_for_preconsolidation(
            sigma_v0=50.0, sigma_p_target=100.0,
        )
        assert q == pytest.approx(50.0)

        # With influence factor
        q2 = required_surcharge_for_preconsolidation(
            sigma_v0=50.0, sigma_p_target=100.0, influence_factor=0.8,
        )
        assert q2 == pytest.approx(62.5)

    def test_time_to_target_no_drains(self):
        """Time without drains should match settlement.time_rate."""
        from settlement.time_rate import time_for_consolidation
        t_ref = time_for_consolidation(90.0, cv=1.0, Hdr=5.0)
        t_test = time_to_target_consolidation(90.0, cv=1.0, Hdr=5.0)
        assert t_test == pytest.approx(t_ref, rel=1e-6)

    def test_full_surcharge_analysis(self):
        """Full surcharge analysis with drains."""
        result = analyze_surcharge_preloading(
            S_ultimate=0.3,
            surcharge_kPa=60.0,
            cv=1.0,
            Hdr=5.0,
            target_U=90.0,
            ch=3.0,
            drain_spacing=1.5,
            sigma_v0=40.0,
        )
        assert result.uses_wick_drains is True
        assert result.wick_drain_result is not None
        assert result.time_to_target_years > 0
        assert result.settlement_at_target_mm == pytest.approx(
            0.9 * 300.0, rel=1e-4  # 90% of 300 mm
        )
        assert result.settlement_ultimate_mm == pytest.approx(300.0)
        assert result.equivalent_sigma_p_kPa == pytest.approx(100.0)
        assert result.time_settlement_curve is not None
        assert "SURCHARGE" in result.summary()


# ================================================================
# VIBRO-COMPACTION
# ================================================================
class TestVibroCompaction:
    """Test vibro-compaction feasibility assessment."""

    def test_feasible_clean_sand(self):
        """Fines below 10 percent: GEC-13 p. 4-67's zone B."""
        feasible, reasons = vibro_feasibility(5.0, initial_N_spt=8)
        assert feasible is True
        assert any("below 10 percent" in r for r in reasons)

    def test_infeasible_high_fines(self):
        """Above GEC-13's 12 to 15 percent (p. 4-65): not feasible."""
        feasible, reasons = vibro_feasibility(25.0, initial_N_spt=10)
        assert feasible is False
        assert any("12 to 15 percent" in r for r in reasons)

    def test_marginal_fines(self):
        """Within the printed 12 to 15 percent: feasible, marginal."""
        feasible, reasons = vibro_feasibility(13.0, initial_N_spt=10)
        assert feasible is True
        assert any("marginal" in r for r in reasons)

    def test_probe_spacing(self):
        """A higher target relative density -> a tighter spacing (Fig.
        4-34)."""
        assert probe_spacing_fig_4_34(80.0) < probe_spacing_fig_4_34(60.0)

    def test_full_vibro_analysis(self):
        """Full vibro analysis end-to-end."""
        result = analyze_vibro_compaction(
            fines_content=5.0, initial_N_spt=8,
            target_relative_density=70.0,
        )
        assert result.is_feasible is True
        assert result.recommended_spacing_m > 0
        assert result.probe_pattern == "triangular"
        assert "VIBRO" in result.summary()
        d = result.to_dict()
        assert d["is_feasible"] is True


# ================================================================
# FEASIBILITY DECISION SUPPORT
# ================================================================
class TestFeasibility:
    """Test ground improvement feasibility evaluation."""

    def test_soft_clay_recommends_drains(self):
        """Soft clay with settlement problem -> wick drains, surcharge, piers."""
        result = evaluate_feasibility(
            soil_type="soft_clay",
            cu_kPa=30.0,
            thickness_m=6.0,
            predicted_settlement_mm=120.0,
            allowable_settlement_mm=50.0,
            time_constraint_months=12,
            cv_m2_per_year=1.0,
            Hdr_m=3.0,
        )
        assert "Wick Drains (PVD)" in result.applicable_methods
        assert "Surcharge Preloading" in result.applicable_methods
        assert "Aggregate Piers" in result.applicable_methods

    def test_loose_sand_recommends_vibro(self):
        """Clean loose sand -> vibro-compaction."""
        result = evaluate_feasibility(
            soil_type="loose_sand",
            fines_content=5.0,
            N_spt=8,
            predicted_settlement_mm=40.0,
            allowable_settlement_mm=25.0,
        )
        assert "Vibro-Compaction" in result.applicable_methods
        # Drains and surcharge not applicable for sand
        excluded_methods = [d["method"] for d in result.not_applicable]
        assert "Wick Drains (PVD)" in excluded_methods
        assert "Surcharge Preloading" in excluded_methods

    def test_organic_takes_special_care(self):
        """Very soft organic soil: GEC-13 p. 5-9 asks for special care
        under 300 psf, it does not exclude aggregate columns (GM-291;
        until then excluded under 10 kPa, a limit printed nowhere)."""
        result = evaluate_feasibility(
            soil_type="organic",
            cu_kPa=8.0,
            thickness_m=5.0,
            predicted_settlement_mm=200.0,
            allowable_settlement_mm=50.0,
        )
        assert "Aggregate Piers" in result.applicable_methods
        assert any("Special care must be taken" in r
                   for r in result.recommendations)

    def test_stiff_clay_excludes_piers(self):
        """Above GEC-13's practical upper limit for stone columns, 1000 to
        2000 psf (p. 5-9): not applicable."""
        result = evaluate_feasibility(
            soil_type="soft_clay", cu_kPa=100.0, thickness_m=5.0,
            predicted_settlement_mm=200.0, allowable_settlement_mm=50.0,
        )
        excluded = {d["method"]: d["reason"] for d in result.not_applicable}
        assert "1000 to 2000 psf" in excluded["Aggregate Piers"]

    def test_time_constraint_favors_drains(self):
        """Tight time constraint -> recommendations mention drains."""
        result = evaluate_feasibility(
            soil_type="soft_clay",
            cu_kPa=25.0,
            thickness_m=8.0,
            predicted_settlement_mm=150.0,
            allowable_settlement_mm=50.0,
            time_constraint_months=6,
            cv_m2_per_year=0.5,
            Hdr_m=4.0,
        )
        # Should recommend combining surcharge with drains
        recs = " ".join(result.recommendations).lower()
        assert "drain" in recs

    def test_all_methods_checked(self):
        """All 4 methods should appear in applicable or not_applicable."""
        result = evaluate_feasibility(
            soil_type="mixed",
            fines_content=10.0,
            N_spt=12,
            cu_kPa=40.0,
            thickness_m=6.0,
            predicted_settlement_mm=80.0,
            allowable_settlement_mm=25.0,
            time_constraint_months=12,
            cv_m2_per_year=1.0,
            Hdr_m=3.0,
        )
        all_methods = set(result.applicable_methods)
        all_methods.update(d["method"] for d in result.not_applicable)
        assert len(all_methods) == 4
        assert "GROUND IMPROVEMENT FEASIBILITY" in result.summary()


class TestFeasibilityDepth:
    """GM-179: the treatment reaches depth_to_top_m + thickness_m, held
    against GEC-13's printed depths: stone columns 100 ft and rammed
    aggregate piers 35 ft (p. 5-9, item 7); vibro-compaction about 165 ft,
    about 100 ft generally recommended (p. 4-65)."""

    @staticmethod
    def _excluded(result):
        return {d["method"]: d["reason"] for d in result.not_applicable}

    def test_the_printed_depths(self):
        from ground_improvement.feasibility import (
            RAMMED_PIER_MAX_DEPTH_M, STONE_COLUMN_MAX_DEPTH_M,
            VIBRO_MAX_DEPTH_M, VIBRO_RECOMMENDED_DEPTH_M)
        assert STONE_COLUMN_MAX_DEPTH_M == pytest.approx(30.48)
        assert RAMMED_PIER_MAX_DEPTH_M == pytest.approx(10.668)
        assert VIBRO_MAX_DEPTH_M == pytest.approx(50.292)
        assert VIBRO_RECOMMENDED_DEPTH_M == pytest.approx(30.48)

    def test_within_35_ft_any_aggregate_pier(self):
        r = evaluate_feasibility(soil_type="soft_clay", thickness_m=8.0,
                                 depth_to_top_m=2.0)
        assert "Aggregate Piers" in r.applicable_methods
        assert "column_type" not in r.preliminary_sizing["aggregate_piers"]
        assert not any("p. 5-9" in x for x in r.recommendations)

    def test_beyond_35_ft_stone_columns(self):
        # the layer's depth counts: 10 m thick from 5 m reaches 15 m
        r = evaluate_feasibility(soil_type="soft_clay", thickness_m=10.0,
                                 depth_to_top_m=5.0)
        assert "Aggregate Piers" in r.applicable_methods
        assert r.preliminary_sizing["aggregate_piers"]["column_type"] == \
            "stone columns"
        assert any("15.0 m" in x and "GEC-13 p. 5-9" in x
                   for x in r.recommendations)
        # 15 m from the surface: excluded by the old 12 m limit, within
        # stone columns' 100 ft
        r = evaluate_feasibility(soil_type="soft_clay", thickness_m=15.0)
        assert "Aggregate Piers" in r.applicable_methods

    def test_beyond_100_ft_no_aggregate_pier(self):
        r = evaluate_feasibility(soil_type="soft_clay", thickness_m=25.0,
                                 depth_to_top_m=10.0)
        why = self._excluded(r)["Aggregate Piers"]
        assert "35.0 m" in why and "100 ft (30.5 m)" in why
        assert "GEC-13 p. 5-9" in why

    def test_vibro_beyond_the_recommended_depth(self):
        r = evaluate_feasibility(soil_type="loose_sand", fines_content=5.0,
                                 thickness_m=10.0, depth_to_top_m=25.0)
        assert "Vibro-Compaction" in r.applicable_methods
        assert any("generally recommended depth of about 100 ft" in x
                   for x in r.recommendations)

    def test_vibro_beyond_its_maximum(self):
        r = evaluate_feasibility(soil_type="loose_sand", fines_content=5.0,
                                 thickness_m=20.0, depth_to_top_m=35.0)
        why = self._excluded(r)["Vibro-Compaction"]
        assert "55.0 m" in why and "165 ft (50 m)" in why

    def test_the_water_table_changes_nothing(self):
        # p. 4-65: vibro-compaction works above and below it
        a = evaluate_feasibility(soil_type="loose_sand", fines_content=5.0,
                                 thickness_m=6.0).to_dict()
        b = evaluate_feasibility(soil_type="loose_sand", fines_content=5.0,
                                 thickness_m=6.0, gwt_depth_m=1.0).to_dict()
        assert a == b


# ================================================================
# RESULT CONTAINERS
# ================================================================
class TestResults:
    """Test result dataclass summary() and to_dict() methods."""

    def test_aggregate_pier_result_summary(self):
        """summary() returns readable formatted string."""
        r = AggregatePierResult(
            area_replacement_ratio=0.08,
            stress_concentration_ratio=5.0,
            composite_modulus_kPa=12500,
            settlement_reduction_factor=0.76,
            column_diameter_m=0.6,
            column_spacing_m=2.0,
            pattern="triangular",
        )
        s = r.summary()
        assert "0.0800" in s  # area replacement
        assert "triangular" in s

    def test_wick_drain_result_to_dict(self):
        """to_dict() returns correct keys and types."""
        r = WickDrainResult(
            drain_spacing_m=1.5,
            pattern="triangular",
            U_total_percent=91.8,
            time_years=0.5,
            time_settlement_curve=[(0.0, 0.0), (0.25, 60.0), (0.5, 91.8)],
        )
        d = r.to_dict()
        assert d["drain_spacing_m"] == 1.5
        assert d["U_total_percent"] == 91.8
        assert "time_settlement_curve" in d
        assert len(d["time_settlement_curve"]) == 3

    def test_feasibility_result_summary(self):
        """Feasibility summary formats recommendations."""
        r = FeasibilityResult(
            applicable_methods=["Wick Drains (PVD)", "Surcharge Preloading"],
            not_applicable=[{"method": "Vibro-Compaction", "reason": "Clay soil"}],
            recommendations=["Use wick drains with surcharge"],
            soil_description="soft_clay, cu=30 kPa",
            design_problem="settlement 120mm > 50mm",
        )
        s = r.summary()
        assert "Wick Drains" in s
        assert "Vibro-Compaction: Clay soil" in s
        assert "1." in s  # numbered recommendation


# ================================================================
# VALIDATION AND EDGE CASES
# ================================================================
class TestValidation:
    """Test input validation and error handling."""

    def test_negative_spacing_raises(self):
        """Negative spacing should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            area_replacement_ratio(0.6, -1.0)

    def test_zero_cv_raises(self):
        """Zero cv should raise ValueError for wick drains."""
        with pytest.raises(ValueError, match="positive"):
            analyze_wick_drains(spacing=1.5, ch=3.0, cv=0.0, Hdr=5.0, time=0.5)

    def test_overlapping_columns_raises(self):
        """Spacing < diameter should raise ValueError."""
        with pytest.raises(ValueError, match="exceed"):
            area_replacement_ratio(0.6, 0.5)

    def test_target_U_100_raises(self):
        """100% target consolidation should raise ValueError."""
        with pytest.raises(ValueError):
            time_for_radial_consolidation(100.0, ch=3.0, de=1.5, F_n=2.5)
