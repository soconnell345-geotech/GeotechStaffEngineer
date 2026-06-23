"""
Validation tests for the settlement module.

Test cases are drawn from textbook examples, FHWA guidance, and
hand calculations to verify stress distribution, immediate settlement,
consolidation settlement, and time-rate calculations.

References:
    [1] FHWA GEC-6 (FHWA-IF-02-054), Chapter 8
    [2] Das, B.M., "Principles of Geotechnical Engineering"
    [3] Holtz, Kovacs & Sheahan, "An Introduction to Geotechnical Engineering"
    [4] Coduto, D.P., "Foundation Design", 2nd ed.
"""

import math
import pytest

from settlement.stress_distribution import (
    approximate_2to1, boussinesq_center_rectangular, stress_at_depth,
)
from settlement.immediate import (
    elastic_settlement, schmertmann_settlement, SchmertmannLayer,
)
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
    total_consolidation_settlement,
)
from settlement.hough import (
    HoughLayer, HoughResult, hough_settlement, hough_settlement_layer,
)
from settlement.time_rate import (
    time_factor, degree_of_consolidation, time_for_consolidation,
    settlement_at_time,
)
from settlement.secondary import secondary_settlement
from settlement.analysis import SettlementAnalysis
from settlement.results import SettlementResult


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Stress Distribution
# ═══════════════════════════════════════════════════════════════════════

class TestStressDistribution:
    """Verify stress distribution methods."""

    def test_2to1_at_surface(self):
        """At z=0, stress should equal applied pressure."""
        ds = approximate_2to1(100, 2.0, 3.0, 0.0)
        assert ds == 100.0

    def test_2to1_at_depth_B(self):
        """2:1 method at depth z=B for a square footing B=L=2m.
        ds = q*B*L / ((B+z)*(L+z)) = 100*2*2 / (4*4) = 25 kPa."""
        ds = approximate_2to1(100, 2.0, 2.0, 2.0)
        assert ds == pytest.approx(25.0, rel=1e-6)

    def test_2to1_decays_with_depth(self):
        """Stress should decrease monotonically with depth."""
        prev = 1e10
        for z in [0, 0.5, 1, 2, 5, 10]:
            ds = approximate_2to1(100, 2.0, 3.0, z)
            assert ds <= prev
            prev = ds

    def test_boussinesq_at_depth(self):
        """Boussinesq at center of 2m x 2m footing at depth 2m.
        Should be approximately 33.5 kPa for 100 kPa applied
        (from standard Boussinesq charts, I_z ≈ 0.335)."""
        ds = boussinesq_center_rectangular(100, 2.0, 2.0, 2.0)
        # Center of 2x2 at z=2: m=n=0.5, I_corner from chart
        # 4 * I_corner(0.5, 0.5) ≈ 4*0.084 ≈ 0.336
        assert 30 < ds < 40  # reasonable range

    def test_boussinesq_vs_2to1(self):
        """2:1 and Boussinesq should give similar (same order) results."""
        ds_21 = stress_at_depth(100, 2.0, 2.0, 2.0, method="2:1")
        ds_b = stress_at_depth(100, 2.0, 2.0, 2.0, method="boussinesq")
        # Should be within factor of 2
        assert 0.5 < ds_21 / ds_b < 2.0

    def test_stress_interface(self):
        """stress_at_depth should dispatch correctly."""
        ds = stress_at_depth(100, 2.0, 3.0, 1.0, method="2:1")
        expected = 100 * 2 * 3 / (3 * 4)
        assert ds == pytest.approx(expected, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Elastic (Immediate) Settlement
# ═══════════════════════════════════════════════════════════════════════

class TestElasticSettlement:
    """Verify elastic settlement calculation."""

    def test_basic_elastic(self):
        """Se = q*B*(1-nu^2)/Es * Iw.
        q=100, B=2, Es=10000, nu=0.3, Iw=1
        Se = 100*2*(1-0.09)/10000 = 0.0182 m = 18.2 mm."""
        Se = elastic_settlement(100, 2.0, 10000, nu=0.3, Iw=1.0)
        assert Se == pytest.approx(0.0182, rel=1e-4)

    def test_elastic_proportional_to_q(self):
        """Settlement should be proportional to applied pressure."""
        Se1 = elastic_settlement(50, 2.0, 10000)
        Se2 = elastic_settlement(100, 2.0, 10000)
        assert Se2 == pytest.approx(2 * Se1, rel=1e-6)

    def test_elastic_proportional_to_B(self):
        """Settlement should be proportional to footing width."""
        Se1 = elastic_settlement(100, 1.0, 10000)
        Se2 = elastic_settlement(100, 2.0, 10000)
        assert Se2 == pytest.approx(2 * Se1, rel=1e-6)

    def test_elastic_inversely_proportional_to_Es(self):
        """Settlement should be inversely proportional to Es."""
        Se1 = elastic_settlement(100, 2.0, 10000)
        Se2 = elastic_settlement(100, 2.0, 20000)
        assert Se1 == pytest.approx(2 * Se2, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Schmertmann Settlement
# ═══════════════════════════════════════════════════════════════════════

class TestSchmertmannSettlement:
    """Verify Schmertmann's improved method (1978)."""

    def test_basic_schmertmann(self):
        """Schmertmann settlement for a simple case.
        Square footing B=2m, q_net=100 kPa, q0=20 kPa.
        Uniform soil Es=10000 kPa from 0 to 2B=4m."""
        layers = [
            SchmertmannLayer(depth_top=0, depth_bottom=1.0, Es=10000),
            SchmertmannLayer(depth_top=1.0, depth_bottom=2.0, Es=10000),
            SchmertmannLayer(depth_top=2.0, depth_bottom=3.0, Es=10000),
            SchmertmannLayer(depth_top=3.0, depth_bottom=4.0, Es=10000),
        ]
        Se = schmertmann_settlement(100, 20, 2.0, layers, footing_shape="square")
        # Should give a reasonable positive settlement
        assert Se > 0
        # Typical range: 5-30 mm for this loading
        assert 0.001 < Se < 0.05

    def test_schmertmann_C1_correction(self):
        """C1 increases with depth of embedment (lower q0/q_net)."""
        layers = [SchmertmannLayer(0, 4, 10000)]
        Se_shallow = schmertmann_settlement(100, 10, 2.0, layers)
        Se_deep = schmertmann_settlement(100, 80, 2.0, layers)
        # Higher q0/q_net → higher C1 correction → less settlement
        assert Se_deep < Se_shallow

    def test_schmertmann_zero_net_pressure(self):
        """Zero net pressure should give zero settlement."""
        layers = [SchmertmannLayer(0, 4, 10000)]
        Se = schmertmann_settlement(0, 20, 2.0, layers)
        assert Se == 0.0

    def test_schmertmann_creep(self):
        """Settlement with creep (C2) should exceed no-creep."""
        layers = [SchmertmannLayer(0, 4, 10000)]
        Se_0 = schmertmann_settlement(100, 20, 2.0, layers, time_years=0)
        Se_10 = schmertmann_settlement(100, 20, 2.0, layers, time_years=10)
        assert Se_10 > Se_0


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Primary Consolidation Settlement
# ═══════════════════════════════════════════════════════════════════════

class TestConsolidationSettlement:
    """Verify consolidation settlement calculations.

    Reference example: Das, "Principles of Geotechnical Engineering"
    Typical NC clay: H=3m, e0=1.1, Cc=0.36, sigma_v0=50 kPa,
    delta_sigma=40 kPa.
    Sc = 0.36*3/(1+1.1)*log10(90/50) = 0.514*0.2553 = 0.131 m ≈ 131 mm.
    """

    def test_NC_clay_single_layer(self):
        """NC clay: verify hand calculation."""
        layer = ConsolidationLayer(
            thickness=3.0, depth_to_center=1.5,
            e0=1.1, Cc=0.36, Cr=0.05, sigma_v0=50.0,
            sigma_p=50.0  # NC
        )
        Sc = consolidation_settlement_layer(layer, delta_sigma=40.0)
        # Sc = 0.36*3/(1+1.1)*log10(90/50) = 0.5143*0.2553 = 0.1313 m
        expected = 0.36 * 3 / (1 + 1.1) * math.log10(90 / 50)
        assert Sc == pytest.approx(expected, rel=1e-6)
        assert Sc == pytest.approx(0.1313, rel=0.01)

    def test_OC_clay_within_range(self):
        """OC clay where stress stays below preconsolidation.
        sigma_v0=50, sigma_p=100, delta_sigma=30 (final=80 < 100).
        Sc = Cr*H/(1+e0)*log10(80/50)."""
        layer = ConsolidationLayer(
            thickness=3.0, depth_to_center=1.5,
            e0=1.1, Cc=0.36, Cr=0.05, sigma_v0=50.0,
            sigma_p=100.0  # OCR=2
        )
        Sc = consolidation_settlement_layer(layer, delta_sigma=30.0)
        expected = 0.05 * 3 / (1 + 1.1) * math.log10(80 / 50)
        assert Sc == pytest.approx(expected, rel=1e-6)

    def test_OC_clay_exceeds_preconsolidation(self):
        """OC clay where stress exceeds preconsolidation.
        sigma_v0=50, sigma_p=80, delta_sigma=60 (final=110 > 80).
        Sc = Cr*H/(1+e0)*log10(80/50) + Cc*H/(1+e0)*log10(110/80)."""
        layer = ConsolidationLayer(
            thickness=3.0, depth_to_center=1.5,
            e0=1.1, Cc=0.36, Cr=0.05, sigma_v0=50.0,
            sigma_p=80.0
        )
        Sc = consolidation_settlement_layer(layer, delta_sigma=60.0)
        Sc_oc = 0.05 * 3 / (1 + 1.1) * math.log10(80 / 50)
        Sc_nc = 0.36 * 3 / (1 + 1.1) * math.log10(110 / 80)
        expected = Sc_oc + Sc_nc
        assert Sc == pytest.approx(expected, rel=1e-6)

    def test_OC_greater_than_NC(self):
        """NC settlement should exceed OC settlement for same delta_sigma."""
        nc_layer = ConsolidationLayer(
            thickness=3, depth_to_center=1.5,
            e0=1.0, Cc=0.3, Cr=0.05, sigma_v0=50, sigma_p=50
        )
        oc_layer = ConsolidationLayer(
            thickness=3, depth_to_center=1.5,
            e0=1.0, Cc=0.3, Cr=0.05, sigma_v0=50, sigma_p=100
        )
        Sc_nc = consolidation_settlement_layer(nc_layer, 40)
        Sc_oc = consolidation_settlement_layer(oc_layer, 40)
        assert Sc_nc > Sc_oc

    def test_zero_stress_increase(self):
        """Zero stress increase → zero settlement."""
        layer = ConsolidationLayer(
            thickness=3, depth_to_center=1.5,
            e0=1.0, Cc=0.3, Cr=0.05, sigma_v0=50
        )
        Sc = consolidation_settlement_layer(layer, 0.0)
        assert Sc == 0.0

    def test_layer_summation(self):
        """Layer summation should equal sum of individual layers."""
        layers = [
            ConsolidationLayer(thickness=2, depth_to_center=1,
                               e0=0.8, Cc=0.3, Cr=0.05, sigma_v0=30),
            ConsolidationLayer(thickness=2, depth_to_center=3,
                               e0=0.9, Cc=0.25, Cr=0.04, sigma_v0=50),
        ]
        total = total_consolidation_settlement(layers, 100, 2.0, 2.0, "2:1")
        # Verify it equals sum of individual calculations
        individual_sum = 0
        for layer in layers:
            ds = approximate_2to1(100, 2.0, 2.0, layer.depth_to_center)
            individual_sum += consolidation_settlement_layer(layer, ds)
        assert total == pytest.approx(individual_sum, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Time Rate of Consolidation
# ═══════════════════════════════════════════════════════════════════════

class TestTimeRate:
    """Verify Terzaghi time-rate relationships."""

    def test_U_at_Tv_0(self):
        """Tv=0 → U=0%."""
        assert degree_of_consolidation(0.0) == 0.0

    def test_U_at_small_Tv(self):
        """At Tv=0.05: U = 100*sqrt(4*0.05/pi) ≈ 25.2%."""
        U = degree_of_consolidation(0.05)
        expected = 100 * math.sqrt(4 * 0.05 / math.pi)
        assert U == pytest.approx(expected, rel=0.01)

    def test_U_at_Tv_0283(self):
        """At Tv≈0.283 (boundary): U ≈ 60%."""
        Tv_60 = math.pi / 4 * 0.36  # Tv for U=60%
        U = degree_of_consolidation(Tv_60)
        assert U == pytest.approx(60.0, abs=1.0)

    def test_U_at_Tv_1(self):
        """At Tv=1.0: U should be around 93%."""
        U = degree_of_consolidation(1.0)
        assert 90 < U < 97

    def test_U_monotonically_increasing(self):
        """U should increase with Tv."""
        prev = 0
        for Tv in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            U = degree_of_consolidation(Tv)
            assert U >= prev
            prev = U

    def test_time_for_50_percent(self):
        """Time for 50% consolidation.
        Tv = pi/4 * (0.5)^2 = 0.1963.
        t = Tv * Hdr^2 / cv. With cv=1, Hdr=2: t = 0.1963*4/1 = 0.785 years."""
        t = time_for_consolidation(50.0, cv=1.0, Hdr=2.0)
        Tv_50 = math.pi / 4 * 0.25
        expected = Tv_50 * 4.0 / 1.0
        assert t == pytest.approx(expected, rel=1e-4)

    def test_time_for_90_percent(self):
        """Time for 90% consolidation.
        Tv = -0.9332*log10(0.1) - 0.0851 = 0.9332 - 0.0851 = 0.848."""
        t = time_for_consolidation(90.0, cv=1.0, Hdr=1.0)
        Tv_90 = -0.9332 * math.log10(0.1) - 0.0851
        assert t == pytest.approx(Tv_90, rel=0.01)

    def test_settlement_at_time(self):
        """Settlement at time t = U(t) * S_ultimate."""
        S_ult = 0.1  # 100 mm
        # At very large time, settlement → S_ult
        S = settlement_at_time(S_ult, cv=1.0, Hdr=1.0, t=100.0)
        assert S == pytest.approx(S_ult, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Secondary Compression
# ═══════════════════════════════════════════════════════════════════════

class TestSecondaryCompression:
    """Verify secondary compression calculations."""

    def test_basic_secondary(self):
        """Ss = C_alpha/(1+e0) * H * log10(t2/t1).
        C_alpha=0.02, e0=1.0, H=3m, t1=1yr, t2=10yr.
        Ss = 0.02/2 * 3 * log10(10) = 0.01*3*1 = 0.03 m."""
        Ss = secondary_settlement(0.02, 3.0, 1.0, 10.0, e0=1.0)
        expected = 0.02 / 2 * 3 * math.log10(10)
        assert Ss == pytest.approx(expected, rel=1e-6)
        assert Ss == pytest.approx(0.03, rel=1e-6)

    def test_secondary_zero_calpha(self):
        """Zero C_alpha → zero secondary settlement."""
        assert secondary_settlement(0.0, 3.0, 1.0, 10.0) == 0.0

    def test_secondary_no_time_elapsed(self):
        """t2 <= t1 → zero secondary settlement."""
        assert secondary_settlement(0.02, 3.0, 5.0, 5.0) == 0.0
        assert secondary_settlement(0.02, 3.0, 5.0, 3.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Combined Settlement Analysis
# ═══════════════════════════════════════════════════════════════════════

class TestCombinedAnalysis:
    """Test the combined SettlementAnalysis interface."""

    def test_elastic_only(self):
        """Analysis with only elastic settlement."""
        analysis = SettlementAnalysis(
            q_applied=100, q_overburden=20, B=2.0, L=2.0,
            immediate_method="elastic", Es_immediate=10000, nu=0.3,
        )
        result = analysis.compute()
        assert result.immediate > 0
        assert result.consolidation == 0
        assert result.secondary == 0
        assert result.total == result.immediate

    def test_consolidation_only(self):
        """Analysis with only consolidation settlement."""
        analysis = SettlementAnalysis(
            q_applied=100, q_overburden=20, B=2.0, L=2.0,
            consolidation_layers=[
                ConsolidationLayer(thickness=3, depth_to_center=1.5,
                                   e0=1.0, Cc=0.3, Cr=0.05, sigma_v0=40)
            ]
        )
        result = analysis.compute()
        assert result.immediate == 0  # no Es provided
        assert result.consolidation > 0
        assert result.total == result.consolidation

    def test_full_analysis(self):
        """Full analysis: immediate + consolidation + secondary."""
        analysis = SettlementAnalysis(
            q_applied=150, q_overburden=20, B=3.0, L=3.0,
            immediate_method="elastic", Es_immediate=15000, nu=0.3,
            consolidation_layers=[
                ConsolidationLayer(thickness=2, depth_to_center=1,
                                   e0=0.8, Cc=0.3, Cr=0.05, sigma_v0=30),
            ],
            cv=1.5, drainage="double",
            C_alpha=0.01, e0_secondary=0.85, t_secondary=20.0,
        )
        result = analysis.compute()
        assert result.immediate > 0
        assert result.consolidation > 0
        assert result.secondary > 0
        total = result.immediate + result.consolidation + result.secondary
        assert result.total == pytest.approx(total, rel=1e-6)

    def test_time_curve_generated(self):
        """Time curve should be generated when cv is provided."""
        analysis = SettlementAnalysis(
            q_applied=100, q_overburden=20, B=2.0, L=2.0,
            consolidation_layers=[
                ConsolidationLayer(thickness=3, depth_to_center=1.5,
                                   e0=1.0, Cc=0.3, Cr=0.05, sigma_v0=40)
            ],
            cv=2.0, drainage="double",
        )
        result = analysis.compute()
        assert result.time_settlement_curve is not None
        assert len(result.time_settlement_curve) > 10
        # First point at t=0 should have only immediate settlement
        t0, s0 = result.time_settlement_curve[0]
        assert t0 == 0.0

    def test_to_dict(self):
        """to_dict should return all key fields."""
        analysis = SettlementAnalysis(
            q_applied=100, q_overburden=20, B=2.0, L=2.0,
            immediate_method="elastic", Es_immediate=10000,
        )
        result = analysis.compute()
        d = result.to_dict()
        assert "immediate_mm" in d
        assert "total_mm" in d
        assert d["total_mm"] > 0

    def test_summary_string(self):
        """summary() should return a formatted string."""
        analysis = SettlementAnalysis(
            q_applied=100, q_overburden=20, B=2.0, L=2.0,
            immediate_method="elastic", Es_immediate=10000,
        )
        result = analysis.compute()
        s = result.summary()
        assert "SETTLEMENT" in s
        assert "mm" in s


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Textbook Example — Das Consolidation Example
# ═══════════════════════════════════════════════════════════════════════

class TestDasConsolidation:
    """Textbook example: primary consolidation of NC clay.

    Problem (Das-style):
    A 3m thick NC clay layer with e0=1.1, Cc=0.36.
    Initial effective stress at center: sigma_v0 = 50 kPa.
    Stress increase at center: delta_sigma = 40 kPa.
    Sc = Cc*H/(1+e0)*log10((sigma_v0+delta_sigma)/sigma_v0)
       = 0.36*3/(1+1.1)*log10(90/50)
       = 0.5143*0.2553 = 0.131 m = 131 mm
    """

    def test_das_consolidation(self):
        """Verify Das-style NC clay consolidation example."""
        layer = ConsolidationLayer(
            thickness=3.0, depth_to_center=1.5,
            e0=1.1, Cc=0.36, Cr=0.05,
            sigma_v0=50.0, sigma_p=50.0
        )
        Sc = consolidation_settlement_layer(layer, 40.0)
        assert Sc * 1000 == pytest.approx(131.3, abs=1.0)  # 131 mm

    def test_das_time_50percent(self):
        """Time for 50% consolidation: cv=3.0 m²/yr, Hdr=1.5m.
        Tv50 = 0.197, t = 0.197*1.5²/3.0 = 0.148 years."""
        t = time_for_consolidation(50.0, cv=3.0, Hdr=1.5)
        Tv50 = math.pi / 4 * 0.25
        expected = Tv50 * 1.5**2 / 3.0
        assert t == pytest.approx(expected, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: QC fixes — Schmertmann Izp / C3 + elastic Iw (SET-1/2/3)
# ═══════════════════════════════════════════════════════════════════════

class TestSettlementQCFixes:
    """Regression tests for SET-1 (Izp at peak depth), SET-2 (no spurious C3
    shape factor), and SET-3 (shape-based elastic influence factor)."""

    def test_elastic_influence_factor_square(self):
        from settlement.immediate import elastic_influence_factor
        # Schleicher flexible-center square = 1.12
        assert elastic_influence_factor("square", 2.0, 2.0) == pytest.approx(
            1.122, abs=0.005)

    def test_elastic_influence_factor_circle_and_aspect(self):
        from settlement.immediate import elastic_influence_factor
        assert elastic_influence_factor("circular", 1.0, 1.0) == pytest.approx(1.0)
        Iw_sq = elastic_influence_factor("rectangular", 2.0, 2.0)
        Iw_long = elastic_influence_factor("rectangular", 20.0, 2.0)  # L/B = 10
        assert Iw_long > Iw_sq  # influence factor grows with L/B

    def test_analysis_applies_shape_influence_factor(self):
        """SET-3: the elastic method must apply a shape Iw, not a flat 1.0."""
        from settlement.analysis import SettlementAnalysis
        from settlement.immediate import (
            elastic_settlement, elastic_influence_factor,
        )
        a = SettlementAnalysis(
            q_applied=150, q_overburden=20, B=2.0, L=2.0,
            footing_shape="square",
            immediate_method="elastic", Es_immediate=10000, nu=0.3,
        )
        r = a.compute()
        Iw = elastic_influence_factor("square", 2.0, 2.0)
        expected = elastic_settlement(130, 2.0, 10000, nu=0.3, Iw=Iw)
        assert r.immediate == pytest.approx(expected, rel=1e-9)
        # ... and that is NOT the old Iw=1.0 result
        assert r.immediate != pytest.approx(
            elastic_settlement(130, 2.0, 10000, nu=0.3, Iw=1.0), rel=1e-3)

    def test_schmertmann_Izp_uses_peak_overburden(self):
        """SET-1: supplying gamma_soil raises sigma'_vp at the peak depth, which
        lowers Izp and hence the settlement."""
        layers = [SchmertmannLayer(0, 4, 10000)]
        Se_base = schmertmann_settlement(100, 20, 2.0, layers,
                                         footing_shape="square")
        Se_gamma = schmertmann_settlement(100, 20, 2.0, layers,
                                          footing_shape="square", gamma_soil=18.0)
        assert Se_gamma < Se_base

    def test_schmertmann_strip_not_reduced_by_C3(self):
        """SET-2: with the spurious C3 removed, a strip (plane-strain diagram)
        settles more than a square, not less."""
        layers = [SchmertmannLayer(0, 8, 10000)]
        Se_strip = schmertmann_settlement(100, 20, 2.0, layers,
                                          footing_shape="strip")
        Se_square = schmertmann_settlement(100, 20, 2.0, layers,
                                           footing_shape="square")
        assert Se_strip > Se_square


class TestStressAtSurface:
    """SET-4: center-of-footing stress at exactly z = 0 must return q,
    not 4q (corner function returns q, not q/4, at z <= 0)."""

    def test_boussinesq_center_z0_returns_q(self):
        q = 100.0
        assert stress_at_depth(q, 2.0, 3.0, 0.0,
                               method="boussinesq",
                               location="center") == pytest.approx(q)
        assert boussinesq_center_rectangular(q, 2.0, 3.0, 0.0) == pytest.approx(q)

    def test_westergaard_center_z0_returns_q(self):
        q = 100.0
        assert stress_at_depth(q, 2.0, 3.0, 0.0,
                               method="westergaard",
                               location="center") == pytest.approx(q)

    def test_z0_continuous_with_limit(self):
        """The z -> 0+ limit equals the z = 0 value (no jump)."""
        q = 100.0
        for method in ("boussinesq", "westergaard"):
            near = stress_at_depth(q, 2.0, 3.0, 1e-6,
                                   method=method, location="center")
            at0 = stress_at_depth(q, 2.0, 3.0, 0.0,
                                  method=method, location="center")
            assert near == pytest.approx(at0, rel=1e-3)

    def test_corner_z0_unchanged(self):
        """Corner location at z = 0 still returns q (existing behavior)."""
        q = 100.0
        assert stress_at_depth(q, 2.0, 3.0, 0.0,
                               method="boussinesq",
                               location="corner") == pytest.approx(q)

    def test_below_surface_unchanged(self):
        """Values at z > 0 are untouched by the guard."""
        q = 100.0
        v = stress_at_depth(q, 2.0, 2.0, 1.0,
                            method="boussinesq", location="center")
        expected = 4.0 * stress_at_depth(q, 1.0, 1.0, 1.0,
                                         method="boussinesq",
                                         location="corner")
        assert v == pytest.approx(expected, rel=1e-9)


class TestSecondaryAssumptionsDocumented:
    """SET-5: t1 = 1.0 yr default (no cv) and single-zone Hdr are documented
    assumptions; the calc package carries an explicit note."""

    def _analysis(self, cv=None):
        return SettlementAnalysis(
            q_applied=150, q_overburden=20, B=2.0, L=2.0,
            immediate_method="elastic", Es_immediate=10000,
            consolidation_layers=[
                ConsolidationLayer(thickness=2, depth_to_center=1,
                                   e0=0.8, Cc=0.3, Cr=0.05, sigma_v0=30)
            ],
            C_alpha=0.01, e0_secondary=0.8, t_secondary=20.0,
            cv=cv,
        )

    def test_t1_default_one_year_when_no_cv(self):
        """Without cv, secondary uses t1 = 1.0 yr (the documented default)."""
        a = self._analysis(cv=None)
        r = a.compute()
        expected = secondary_settlement(0.01, 2.0, 1.0, 21.0, 0.8)
        assert r.secondary == pytest.approx(expected, rel=1e-9)

    def test_hdr_single_zone_double_drainage(self):
        """_get_Hdr treats the whole zone as one doubly-drained layer."""
        a = self._analysis(cv=5.0)
        assert a._get_Hdr() == pytest.approx(1.0)  # 2.0 m zone / 2
        a.drainage = "single"
        assert a._get_Hdr() == pytest.approx(2.0)
        a.Hdr = 0.5  # explicit override wins
        assert a._get_Hdr() == pytest.approx(0.5)

    def test_calc_steps_note_flags_assumed_t1(self):
        from settlement.calc_steps import get_calc_steps
        from calc_package.data_model import CalcStep
        a = self._analysis(cv=None)
        r = a.compute()
        sections = get_calc_steps(r, a)
        notes = " ".join(
            it.notes for sec in sections for it in sec.items
            if isinstance(it, CalcStep) and getattr(it, "notes", None))
        assert "ASSUMED" in notes and "1.0 yr" in notes

    def test_calc_steps_note_flags_t95_when_cv_given(self):
        from settlement.calc_steps import get_calc_steps
        from calc_package.data_model import CalcStep
        a = self._analysis(cv=5.0)
        r = a.compute()
        sections = get_calc_steps(r, a)
        notes = " ".join(
            it.notes for sec in sections for it in sec.items
            if isinstance(it, CalcStep) and getattr(it, "notes", None))
        assert "one" in notes and "drained layer" in notes


# ═══════════════════════════════════════════════════════════════════════
# TEST 10: Hough granular (C'-index) settlement — GEC-6 Ex B-1 (V-022)
# ═══════════════════════════════════════════════════════════════════════

class TestHoughSettlement:
    """Verify the Hough (1959) granular C'-index method.

    Reference: FHWA GEC-6 (FHWA-SA-02-054), Appendix B, Example 1,
    Tables B1-2 / B1-3. Square footing Df=2.3 m, four granular layers below
    the base. Per-layer dH = H/C'*log10[(sigma'_vo + delta_sigma)/sigma'_vo],
    delta_sigma from the 2:1 method (square: q*B^2/(B+Z)^2).
    """

    # (thickness H [m], depth-to-mid Z [m], sigma'_vo [kPa], C' [-])
    LAYERS = [
        (2.1, 1.05, 65.7, 65),    # L2 silty sand
        (4.7, 4.45, 132.0, 120),  # L3a well-graded sand
        (3.0, 8.3, 193.0, 102),   # L3b well-graded sand (saturated)
        (3.0, 11.3, 222.0, 110),  # L4 clean uniform sand
    ]

    def _layers(self):
        return [
            HoughLayer(thickness=H, depth_to_center=Z, sigma_v0=svo, C_prime=Cp)
            for (H, Z, svo, Cp) in self.LAYERS
        ]

    def test_layer_formula(self):
        """Single-layer Hough form dH = H/C'*log10[(svo+dsig)/svo]."""
        lyr = HoughLayer(thickness=2.1, depth_to_center=1.05,
                         sigma_v0=65.7, C_prime=65)
        dH = hough_settlement_layer(lyr, delta_sigma=131.69)
        expected = 2.1 / 65 * math.log10((65.7 + 131.69) / 65.7)
        assert dH == pytest.approx(expected, rel=1e-9)

    def test_zero_stress_increase(self):
        """Zero stress increase → zero layer settlement."""
        lyr = HoughLayer(thickness=2.1, depth_to_center=1.05,
                         sigma_v0=65.7, C_prime=65)
        assert hough_settlement_layer(lyr, 0.0) == 0.0

    def test_worked_case_B3_q240(self):
        """GEC-6 Ex B-1 worked single case: B=3 m, q=240 kPa.
        Published per-layer 15/4/1/1 mm, total ~21 mm."""
        res = hough_settlement(self._layers(), q_net=240.0, B=3.0)
        per = [lyr["settlement_mm"] for lyr in res.layers]
        assert per[0] == pytest.approx(15.0, abs=1.0)   # pub 15
        assert per[1] == pytest.approx(4.0, abs=1.0)    # pub 4
        assert per[2] == pytest.approx(1.0, abs=1.0)    # pub 1
        assert per[3] == pytest.approx(1.0, abs=1.0)    # pub 1
        assert res.total_mm == pytest.approx(21.0, abs=1.5)  # pub 21 mm

    def test_other_width_B61_q380(self):
        """GEC-6 Ex B-1 Table B1-3: B=6.1 m, q=380 kPa → 41 mm published."""
        res = hough_settlement(self._layers(), q_net=380.0, B=6.1)
        assert res.total_mm == pytest.approx(41.0, rel=0.15)  # pub 41 mm

    def test_uses_2to1_stress_increase(self):
        """The per-layer delta_sigma equals the module's 2:1 stress (square)."""
        res = hough_settlement(self._layers(), q_net=240.0, B=3.0)
        for lyr, (H, Z, svo, Cp) in zip(res.layers, self.LAYERS):
            ds_expected = approximate_2to1(240.0, 3.0, 3.0, Z)
            assert lyr["delta_sigma_kPa"] == pytest.approx(ds_expected, abs=0.01)

    def test_rectangular_uses_BL(self):
        """Rectangular footing uses q*B*L/((B+z)(L+z)); L>B → less settlement
        than the same B with L=B is wrong — larger L spreads less, so a longer
        footing gives MORE stress at depth → more settlement."""
        sq = hough_settlement(self._layers(), q_net=240.0, B=3.0, L=3.0)
        rect = hough_settlement(self._layers(), q_net=240.0, B=3.0, L=6.0)
        assert rect.total_mm > sq.total_mm

    def test_zero_net_pressure(self):
        """Zero net pressure → zero total settlement."""
        res = hough_settlement(self._layers(), q_net=0.0, B=3.0)
        assert res.total_mm == 0.0
        assert res.layers == []

    def test_result_to_dict_and_summary(self):
        res = hough_settlement(self._layers(), q_net=240.0, B=3.0)
        d = res.to_dict()
        assert d["method"] == "hough"
        assert d["total_settlement_mm"] == pytest.approx(21.0, abs=1.5)
        assert len(d["layer_breakdown"]) == 4
        s = res.summary()
        assert "HOUGH" in s
        assert "mm" in s

    def test_invalid_layer_inputs(self):
        with pytest.raises(ValueError):
            HoughLayer(thickness=-1, depth_to_center=1, sigma_v0=50, C_prime=65)
        with pytest.raises(ValueError):
            HoughLayer(thickness=2, depth_to_center=1, sigma_v0=0, C_prime=65)
        with pytest.raises(ValueError):
            HoughLayer(thickness=2, depth_to_center=1, sigma_v0=50, C_prime=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
