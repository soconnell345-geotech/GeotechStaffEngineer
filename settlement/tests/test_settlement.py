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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
