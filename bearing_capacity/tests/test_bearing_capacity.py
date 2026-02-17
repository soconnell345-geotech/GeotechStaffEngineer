"""
Validation tests for the bearing_capacity module.

Test cases are drawn from published textbook examples, FHWA GEC-6,
and hand calculations to verify the general bearing capacity equation
implementation.

References:
    [1] FHWA GEC-6 (FHWA-IF-02-054), Chapter 6 — Example 6-1 and 6-2
    [2] Das, B.M., "Principles of Foundation Engineering", 9th ed.
    [3] Bowles, J.E., "Foundation Analysis and Design", 5th ed.
    [4] Coduto, D.P., "Foundation Design", 2nd ed.
"""

import math
import pytest

from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    bearing_capacity_Nc, bearing_capacity_Nq, bearing_capacity_Ngamma,
    all_N_factors, shape_factors, depth_factors,
)
from bearing_capacity.results import BearingCapacityResult


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Bearing Capacity Factors (Nc, Nq, Ngamma)
# ═══════════════════════════════════════════════════════════════════════

class TestBearingCapacityFactors:
    """Verify Nc, Nq, Ngamma against published tables.

    Reference values from FHWA GEC-6 Table 6-1 and Das Table 3.4.
    """

    def test_phi_0_factors(self):
        """phi=0: Nc=5.14 (Prandtl), Nq=1.0, Ng=0."""
        assert bearing_capacity_Nc(0) == 5.14
        assert bearing_capacity_Nq(0) == pytest.approx(1.0, rel=1e-6)
        assert bearing_capacity_Ngamma(0) == 0.0

    def test_phi_20_factors(self):
        """phi=20: Nc=14.83, Nq=6.40, Ng_vesic=5.39 (from FHWA GEC-6)."""
        Nc = bearing_capacity_Nc(20)
        Nq = bearing_capacity_Nq(20)
        Ng = bearing_capacity_Ngamma(20, "vesic")

        assert Nc == pytest.approx(14.83, rel=0.01)
        assert Nq == pytest.approx(6.40, rel=0.01)
        assert Ng == pytest.approx(5.39, rel=0.02)

    def test_phi_25_factors(self):
        """phi=25: Nc=20.72, Nq=10.66, Ng_vesic=10.88."""
        Nc = bearing_capacity_Nc(25)
        Nq = bearing_capacity_Nq(25)
        Ng = bearing_capacity_Ngamma(25, "vesic")

        assert Nc == pytest.approx(20.72, rel=0.01)
        assert Nq == pytest.approx(10.66, rel=0.01)
        assert Ng == pytest.approx(10.88, rel=0.02)

    def test_phi_30_factors(self):
        """phi=30: Nc=30.14, Nq=18.40, Ng_vesic=22.40 (FHWA GEC-6)."""
        Nc = bearing_capacity_Nc(30)
        Nq = bearing_capacity_Nq(30)
        Ng = bearing_capacity_Ngamma(30, "vesic")

        assert Nc == pytest.approx(30.14, rel=0.01)
        assert Nq == pytest.approx(18.40, rel=0.01)
        assert Ng == pytest.approx(22.40, rel=0.02)

    def test_phi_35_factors(self):
        """phi=35: Nc=46.12, Nq=33.30, Ng_vesic=48.03."""
        Nc = bearing_capacity_Nc(35)
        Nq = bearing_capacity_Nq(35)
        Ng = bearing_capacity_Ngamma(35, "vesic")

        assert Nc == pytest.approx(46.12, rel=0.01)
        assert Nq == pytest.approx(33.30, rel=0.01)
        assert Ng == pytest.approx(48.03, rel=0.02)

    def test_phi_40_factors(self):
        """phi=40: Nc=75.31, Nq=64.20, Ng_vesic=109.41."""
        Nc = bearing_capacity_Nc(40)
        Nq = bearing_capacity_Nq(40)
        Ng = bearing_capacity_Ngamma(40, "vesic")

        assert Nc == pytest.approx(75.31, rel=0.01)
        assert Nq == pytest.approx(64.20, rel=0.01)
        assert Ng == pytest.approx(109.41, rel=0.02)

    def test_ngamma_methods_comparison(self):
        """Vesic Ngamma >= Hansen Ngamma for phi > 0."""
        for phi in [15, 20, 25, 30, 35, 40]:
            ng_v = bearing_capacity_Ngamma(phi, "vesic")
            ng_h = bearing_capacity_Ngamma(phi, "hansen")
            assert ng_v >= ng_h, f"Vesic Ng should be >= Hansen Ng at phi={phi}"

    def test_nq_monotonically_increasing(self):
        """Nq should increase with increasing phi."""
        prev = 0
        for phi in range(0, 51, 5):
            Nq = bearing_capacity_Nq(phi)
            assert Nq >= prev, f"Nq should increase: Nq({phi})={Nq} < Nq({phi-5})={prev}"
            prev = Nq


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Shape and Depth Factors
# ═══════════════════════════════════════════════════════════════════════

class TestCorrectionFactors:
    """Verify shape and depth factor values."""

    def test_strip_shape_factors(self):
        """Strip footing (B/L -> 0): all shape factors -> 1."""
        sc, sq, sg = shape_factors(30, B=1.0, L=1e6)
        assert sc == pytest.approx(1.0, abs=0.001)
        assert sq == pytest.approx(1.0, abs=0.001)
        assert sg == pytest.approx(1.0, abs=0.01)

    def test_square_shape_factors_vesic(self):
        """Square footing (B/L=1): check Vesic shape factors for phi=30."""
        sc, sq, sg = shape_factors(30, B=1.0, L=1.0, method="vesic")
        # sc = 1 + (1)*(Nq/Nc) = 1 + 18.40/30.14 = 1.611
        assert sc == pytest.approx(1.611, rel=0.01)
        # sq = 1 + (1)*tan(30) = 1 + 0.5774 = 1.577
        assert sq == pytest.approx(1.577, rel=0.01)
        # sg = 1 - 0.4*(1) = 0.6
        assert sg == pytest.approx(0.6, rel=0.01)

    def test_depth_factors_surface_footing(self):
        """Surface footing (Df=0): all depth factors = 1."""
        dc, dq, dg = depth_factors(30, Df=0, B=1.0)
        assert dc == pytest.approx(1.0, abs=0.001)
        assert dq == pytest.approx(1.0, abs=0.001)
        assert dg == 1.0

    def test_depth_factors_embedded(self):
        """Embedded footing phi=30, Df/B=1: check Vesic depth factors."""
        dc, dq, dg = depth_factors(30, Df=2.0, B=2.0, method="vesic")
        # k = Df/B = 1.0 (since Df/B <= 1)
        # dq = 1 + 2*tan(30)*(1-sin(30))^2 * 1.0
        #    = 1 + 2*0.5774*(0.5)^2 = 1 + 2*0.5774*0.25 = 1.289
        assert dq == pytest.approx(1.289, rel=0.01)
        assert dg == 1.0

    def test_depth_factors_deep(self):
        """Deep footing Df/B > 1: uses arctan formula."""
        dc, dq, dg = depth_factors(30, Df=4.0, B=2.0, method="vesic")
        # k = arctan(2.0) = 1.107 rad
        # dq = 1 + 2*0.5774*0.25*1.107 = 1 + 0.320 = 1.320
        assert dq == pytest.approx(1.320, rel=0.01)

    def test_vesic_depth_factors_deep_phi25(self):
        """Vesic depth factors for phi=25, Df/B=3: verify arctan formula."""
        dc, dq, dg = depth_factors(25, Df=6.0, B=2.0, method="vesic")
        # k = arctan(3.0) = 1.2490 rad
        # 2*tan(25)*(1-sin(25))^2 = 2*0.4663*(1-0.4226)^2 = 2*0.4663*0.3334 = 0.3109
        # dq = 1 + 0.3109*1.2490 = 1.3883
        # Nc = 20.72
        # dc = dq - (1-dq)/(Nc*tan(25)) = 1.3883 - (-0.3883)/(20.72*0.4663)
        #    = 1.3883 + 0.3883/9.658 = 1.3883 + 0.0402 = 1.4285
        assert dq == pytest.approx(1.388, rel=0.01)
        assert dc == pytest.approx(1.428, rel=0.01)
        assert dg == 1.0

    def test_vesic_depth_factors_deep_phi0(self):
        """Vesic depth factors for phi=0, Df/B=2: arctan formula for dc."""
        dc, dq, dg = depth_factors(0, Df=4.0, B=2.0, method="vesic")
        # k = arctan(2.0) = 1.1071 rad
        # dc = 1 + 0.4*1.1071 = 1.4429
        assert dc == pytest.approx(1.443, rel=0.01)
        assert dq == 1.0
        assert dg == 1.0

    def test_vesic_depth_factors_deep_phi0_very_deep(self):
        """Vesic dc for phi=0, Df/B=10: arctan bounds the depth factor."""
        dc, dq, dg = depth_factors(0, Df=20.0, B=2.0, method="vesic")
        # k = arctan(10) = 1.4711 rad
        # dc = 1 + 0.4*1.4711 = 1.5884
        # Maximum possible dc (as Df/B -> inf): 1 + 0.4*pi/2 = 1.6283
        assert dc == pytest.approx(1.588, rel=0.01)
        assert dc < 1.0 + 0.4 * (math.pi / 2) + 0.001  # bounded by arctan limit

    def test_meyerhof_depth_factors_shallow(self):
        """Meyerhof depth factors for Df/B <= 1 (phi=30)."""
        dc, dq, dg = depth_factors(30, Df=1.0, B=2.0, method="meyerhof")
        # Df/B = 0.5, k = 0.5
        # Kp = tan(60)^2 = 3.0, sqrt(Kp) = 1.7321
        # dc = 1 + 0.2*1.7321*0.5 = 1.1732
        # dq = dg = 1 + 0.1*1.7321*0.5 = 1.0866
        assert dc == pytest.approx(1.173, rel=0.01)
        assert dq == pytest.approx(1.087, rel=0.01)
        assert dg == pytest.approx(dq, rel=1e-6)

    def test_meyerhof_depth_factors_deep_arctan(self):
        """Meyerhof depth factors for Df/B > 1: must use arctan(Df/B).

        This is the critical fix: without arctan, Meyerhof depth factors
        grow without bound for deep embedment, producing unrealistic results.
        """
        dc, dq, dg = depth_factors(30, Df=4.0, B=2.0, method="meyerhof")
        # Df/B = 2.0, k = arctan(2.0) = 1.1071 rad
        # Kp = 3.0, sqrt(Kp) = 1.7321
        # dc = 1 + 0.2*1.7321*1.1071 = 1 + 0.3835 = 1.3835
        # dq = dg = 1 + 0.1*1.7321*1.1071 = 1 + 0.1918 = 1.1918
        assert dc == pytest.approx(1.384, rel=0.01)
        assert dq == pytest.approx(1.192, rel=0.01)
        assert dg == pytest.approx(dq, rel=1e-6)

    def test_meyerhof_depth_factors_bounded(self):
        """Meyerhof depth factors must be bounded for very deep embedment.

        arctan(Df/B) -> pi/2 as Df/B -> inf, so:
          dc_max = 1 + 0.2*sqrt(Kp)*pi/2
          dq_max = 1 + 0.1*sqrt(Kp)*pi/2
        For phi=30: dc_max ~ 1.545, dq_max ~ 1.272
        Without the arctan fix, Df/B=10 would give dc=4.46 (unbounded!).
        """
        dc, dq, dg = depth_factors(30, Df=20.0, B=2.0, method="meyerhof")
        # Df/B = 10, k = arctan(10) = 1.4711 rad
        k_max = math.pi / 2  # arctan limit as Df/B -> inf
        sqrt_Kp = math.sqrt(3.0)  # phi=30
        dc_theoretical_max = 1.0 + 0.2 * sqrt_Kp * k_max
        dq_theoretical_max = 1.0 + 0.1 * sqrt_Kp * k_max
        assert dc < dc_theoretical_max + 0.001
        assert dq < dq_theoretical_max + 0.001
        # With arctan: dc ~ 1.51, dq ~ 1.25
        # Without arctan fix: dc would be 4.46, dq would be 2.73 (WRONG)
        assert dc < 2.0, "Meyerhof dc must be bounded (arctan transition)"
        assert dq < 2.0, "Meyerhof dq must be bounded (arctan transition)"

    def test_depth_factors_dgamma_always_one_vesic(self):
        """Vesic dgamma = 1.0 for all Df/B ratios."""
        for ratio in [0, 0.5, 1.0, 2.0, 5.0]:
            _, _, dg = depth_factors(30, Df=ratio * 2.0, B=2.0, method="vesic")
            assert dg == 1.0, f"Vesic dgamma must be 1.0 at Df/B={ratio}"

    def test_depth_factors_increase_with_embedment(self):
        """dq should increase (or stay constant) as Df/B increases."""
        prev_dq = 0
        for ratio in [0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
            _, dq, _ = depth_factors(30, Df=ratio * 2.0, B=2.0, method="vesic")
            assert dq >= prev_dq, (
                f"Vesic dq should not decrease: dq({ratio})={dq:.4f} < prev={prev_dq:.4f}"
            )
            prev_dq = dq


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Footing Data Structure
# ═══════════════════════════════════════════════════════════════════════

class TestFooting:
    """Test footing data structure and effective area computation."""

    def test_square_footing(self):
        """Square footing: L should equal B."""
        f = Footing(width=3.0, shape="square")
        assert f.length == 3.0
        assert f.A_eff == 9.0

    def test_circular_footing(self):
        """Circular footing area = pi/4 * D²."""
        f = Footing(width=2.0, shape="circular")
        assert f.A_eff == pytest.approx(math.pi, rel=1e-6)

    def test_eccentric_effective_area(self):
        """Effective area with eccentricity: B'=B-2eB, L'=L-2eL."""
        f = Footing(width=3.0, length=5.0, shape="rectangular",
                    eccentricity_B=0.3, eccentricity_L=0.5)
        assert f.B_eff == pytest.approx(2.4, rel=1e-6)
        assert f.L_eff == pytest.approx(4.0, rel=1e-6)
        assert f.A_eff == pytest.approx(9.6, rel=1e-6)

    def test_strip_footing(self):
        """Strip footing should have very large L."""
        f = Footing(width=1.5, shape="strip")
        assert f.length > 1e5

    def test_invalid_eccentricity(self):
        """Eccentricity exceeding B/2 should raise ValueError."""
        with pytest.raises(ValueError, match="outside the footing"):
            Footing(width=2.0, shape="square", eccentricity_B=1.5)

    def test_invalid_width(self):
        """Negative width should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Footing(width=-1.0, shape="square")

    def test_rectangular_swap(self):
        """Rectangular footing with L < B should swap dimensions."""
        f = Footing(width=5.0, length=3.0, shape="rectangular")
        assert f.width == 3.0  # shorter dimension
        assert f.length == 5.0  # longer dimension


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Soil Profile
# ═══════════════════════════════════════════════════════════════════════

class TestSoilProfile:
    """Test soil profile and overburden calculations."""

    def test_overburden_no_gwt(self):
        """Overburden with no groundwater: q = gamma * Df."""
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        q = soil.overburden_pressure(1.5)
        assert q == pytest.approx(27.0, rel=1e-6)

    def test_overburden_gwt_at_surface(self):
        """GWT at surface: q = gamma' * Df."""
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0),
            gwt_depth=0.0
        )
        q = soil.overburden_pressure(2.0)
        assert q == pytest.approx((18.0 - 9.81) * 2.0, rel=1e-4)

    def test_overburden_gwt_at_footing_base(self):
        """GWT exactly at footing base: q = gamma * Df (no buoyancy above)."""
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0),
            gwt_depth=1.5
        )
        q = soil.overburden_pressure(1.5)
        assert q == pytest.approx(27.0, rel=1e-4)

    def test_overburden_gwt_between(self):
        """GWT between surface and footing: split calculation."""
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0),
            gwt_depth=1.0
        )
        q = soil.overburden_pressure(2.0)
        # q = 18*1.0 + (18-9.81)*1.0 = 18 + 8.19 = 26.19
        expected = 18.0 * 1.0 + (18.0 - 9.81) * 1.0
        assert q == pytest.approx(expected, rel=1e-4)

    def test_invalid_soil(self):
        """Soil with c=0 and phi=0 should raise ValueError."""
        with pytest.raises(ValueError, match="cohesion > 0 or friction_angle > 0"):
            SoilLayer(cohesion=0, friction_angle=0, unit_weight=18.0)


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Published Example — Das Example 3.1 (similar to GEC-6)
#         Strip footing on cohesionless soil
# ═══════════════════════════════════════════════════════════════════════

class TestDasExample:
    """Strip footing on sand — hand-verifiable textbook example.

    Problem: Continuous (strip) footing, B=1.2m, Df=1.0m
    Sand: phi=25°, c=0, gamma=16.5 kN/m³
    No groundwater, vertical concentric load.

    Reference: Das, "Principles of Foundation Engineering", Example 3.1.
    """

    def test_strip_footing_sand(self):
        """Strip footing on sand: verify against hand calculation."""
        footing = Footing(width=1.2, depth=1.0, shape="strip")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=0.0, friction_angle=25.0, unit_weight=16.5)
        )
        analysis = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_of_safety=3.0,
            ngamma_method="vesic"
        )
        result = analysis.compute()

        # Verify bearing capacity factors
        assert result.Nc == pytest.approx(20.72, rel=0.01)
        assert result.Nq == pytest.approx(10.66, rel=0.01)

        # q = 16.5 * 1.0 = 16.5 kPa
        assert result.q_overburden == pytest.approx(16.5, rel=1e-4)

        # Strip footing: all shape factors ~ 1.0
        assert result.sc == pytest.approx(1.0, abs=0.01)

        # Cohesion term = 0 (c=0)
        assert result.term_cohesion == pytest.approx(0.0, abs=0.01)

        # Overburden term = q * Nq * sq * dq
        # For strip: sq ~ 1, dq calculated from Df/B=0.833
        # dq = 1 + 2*tan(25)*(1-sin(25))^2 * 0.833
        #    = 1 + 2*0.4663*0.3334*0.8333 = 1.259
        assert result.dq == pytest.approx(1.259, rel=0.01)

        # Self-weight term = 0.5 * 16.5 * 1.2 * Ng * sg * dg
        # sg ~ 1 for strip, dg = 1
        assert result.dgamma == 1.0

        # qult should be reasonable (roughly 300-400 kPa for this case)
        assert 200 < result.q_ultimate < 600


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Square Footing on Clay (phi=0 analysis)
# ═══════════════════════════════════════════════════════════════════════

class TestClayPhiZero:
    """Square footing on saturated clay, undrained (phi=0) analysis.

    Problem: Square footing 2m x 2m, Df=1m
    Clay: cu=50 kPa, phi=0, gamma=18 kN/m³

    For phi=0: qult = cu*Nc*sc*dc + q
    Nc = 5.14, sc = 1 + 0.2*(B/L) = 1.2, dc = 1 + 0.4*k
    """

    def test_clay_phi0(self):
        """Square footing on clay: verify phi=0 special case."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=50.0, friction_angle=0.0, unit_weight=18.0)
        )
        analysis = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_of_safety=3.0
        )
        result = analysis.compute()

        # Nc = 5.14 (exact Prandtl)
        assert result.Nc == 5.14

        # Nq = 1.0 for phi=0
        assert result.Nq == pytest.approx(1.0, rel=1e-6)

        # Ngamma = 0 for phi=0
        assert result.Ngamma == 0.0

        # Shape: sc = 1 + 0.2*(B/L) = 1 + 0.2*1 = 1.2
        assert result.sc == pytest.approx(1.2, rel=0.01)

        # Depth: dc = 1 + 0.4*k, k=Df/B=0.5 → dc = 1.2
        assert result.dc == pytest.approx(1.2, rel=0.01)

        # q = 18 * 1 = 18 kPa
        assert result.q_overburden == pytest.approx(18.0, rel=1e-4)

        # Cohesion term = cu * Nc * sc * dc = 50 * 5.14 * 1.2 * 1.2 = 370.08
        expected_coh = 50.0 * 5.14 * 1.2 * 1.2
        assert result.term_cohesion == pytest.approx(expected_coh, rel=0.01)

        # Overburden term = q * Nq * sq * dq = 18 * 1 * sq * dq
        # For phi=0: sq=1+0*tan(0)=1, dq=1.0
        assert result.term_overburden == pytest.approx(18.0, rel=0.01)

        # Self-weight term = 0 (Ng=0)
        assert result.term_selfweight == pytest.approx(0.0, abs=0.01)

        # Total: qult ≈ 370.08 + 18 = 388.08
        assert result.q_ultimate == pytest.approx(expected_coh + 18.0, rel=0.01)

    def test_clay_strip_phi0(self):
        """Strip footing on clay: classical Nc*cu + q for phi=0 strip."""
        footing = Footing(width=2.0, depth=1.0, shape="strip")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=100.0, friction_angle=0.0, unit_weight=19.0)
        )
        analysis = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_of_safety=2.5
        )
        result = analysis.compute()

        # For strip, phi=0:
        # sc ~ 1.0, dc = 1 + 0.4*(0.5) = 1.2
        # qult = 100*5.14*1*1.2 + 19*1.0*1*1 = 616.8 + 19 = 635.8
        expected = 100 * 5.14 * 1.0 * 1.2 + 19.0
        assert result.q_ultimate == pytest.approx(expected, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: FHWA GEC-6 Example (Verified Published Example)
# ═══════════════════════════════════════════════════════════════════════

class TestGEC6Example:
    """Bearing capacity from FHWA GEC-6 Chapter 6 worked example concepts.

    Square footing on c-phi soil:
    B=3m, Df=1.5m, c=10kPa, phi=28°, gamma=17 kN/m³
    """

    def test_c_phi_soil(self):
        """Square footing on c-phi soil."""
        footing = Footing(width=3.0, depth=1.5, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=10.0, friction_angle=28.0, unit_weight=17.0)
        )
        analysis = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_of_safety=3.0,
            ngamma_method="vesic"
        )
        result = analysis.compute()

        # Nc, Nq, Ng for phi=28
        Nc = bearing_capacity_Nc(28)
        Nq = bearing_capacity_Nq(28)
        Ng = bearing_capacity_Ngamma(28, "vesic")

        assert result.Nc == pytest.approx(Nc, rel=1e-6)
        assert result.Nq == pytest.approx(Nq, rel=1e-6)

        # q = 17 * 1.5 = 25.5 kPa
        assert result.q_overburden == pytest.approx(25.5, rel=1e-4)

        # All three terms should contribute
        assert result.term_cohesion > 0
        assert result.term_overburden > 0
        assert result.term_selfweight > 0

        # FS = 3
        assert result.q_allowable == pytest.approx(result.q_ultimate / 3.0, rel=1e-6)

        # Sanity: qult should be in a reasonable range
        assert 500 < result.q_ultimate < 3000


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Groundwater Effects
# ═══════════════════════════════════════════════════════════════════════

class TestGroundwaterEffects:
    """Verify that groundwater properly reduces bearing capacity."""

    def test_gwt_reduces_capacity(self):
        """Shallow GWT should give lower qult than deep GWT."""
        footing = Footing(width=2.0, depth=1.0, shape="square")

        # Case 1: No GWT
        soil_dry = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        result_dry = BearingCapacityAnalysis(
            footing=footing, soil=soil_dry
        ).compute()

        # Case 2: GWT at surface
        soil_wet = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0),
            gwt_depth=0.0
        )
        result_wet = BearingCapacityAnalysis(
            footing=footing, soil=soil_wet
        ).compute()

        assert result_wet.q_ultimate < result_dry.q_ultimate
        assert result_wet.q_overburden < result_dry.q_overburden


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: Eccentric Loading
# ═══════════════════════════════════════════════════════════════════════

class TestEccentricLoading:
    """Verify eccentric loading reduces bearing capacity."""

    def test_eccentricity_reduces_capacity(self):
        """Eccentric load should give lower qult than concentric."""
        # Concentric
        footing_c = Footing(width=3.0, length=5.0, depth=1.0, shape="rectangular")
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        result_c = BearingCapacityAnalysis(footing=footing_c, soil=soil).compute()

        # Eccentric
        footing_e = Footing(width=3.0, length=5.0, depth=1.0, shape="rectangular",
                            eccentricity_B=0.5, eccentricity_L=0.3)
        result_e = BearingCapacityAnalysis(footing=footing_e, soil=soil).compute()

        assert result_e.q_ultimate < result_c.q_ultimate
        assert result_e.B_eff < result_c.B_eff


# ═══════════════════════════════════════════════════════════════════════
# TEST 10: Two-Layer Soil
# ═══════════════════════════════════════════════════════════════════════

class TestTwoLayer:
    """Test two-layer bearing capacity analysis."""

    def test_strong_over_weak(self):
        """Strong sand over weak clay: capacity between the two."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=35, unit_weight=19.0, thickness=1.0),
            layer2=SoilLayer(cohesion=30.0, friction_angle=0, unit_weight=17.0),
        )
        result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()

        assert result.is_two_layer
        assert result.q_upper_layer is not None
        assert result.q_lower_layer is not None
        # Capacity should be between the two single-layer solutions
        q_min = min(result.q_upper_layer, result.q_lower_layer)
        q_max = max(result.q_upper_layer, result.q_lower_layer)
        assert q_min <= result.q_ultimate <= q_max

    def test_weak_over_strong(self):
        """Weak clay over strong sand: capacity increases toward bottom."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=25.0, friction_angle=0, unit_weight=17.0,
                             thickness=0.5),
            layer2=SoilLayer(friction_angle=35, unit_weight=19.0),
        )
        result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()
        assert result.is_two_layer


# ═══════════════════════════════════════════════════════════════════════
# TEST 11: Results Container
# ═══════════════════════════════════════════════════════════════════════

class TestResults:
    """Test result container functionality."""

    def test_summary_string(self):
        """summary() should return a non-empty formatted string."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()
        s = result.summary()
        assert "BEARING CAPACITY" in s
        assert "qult" in s

    def test_to_dict(self):
        """to_dict() should return a dict with all key fields."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()
        d = result.to_dict()
        assert "q_ultimate_kPa" in d
        assert "q_allowable_kPa" in d
        assert d["q_ultimate_kPa"] > 0
        assert d["factor_of_safety"] == 3.0

    def test_term_breakdown_sums(self):
        """Term breakdown should sum to qult."""
        footing = Footing(width=2.0, depth=1.5, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(cohesion=10.0, friction_angle=30.0, unit_weight=18.0)
        )
        result = BearingCapacityAnalysis(footing=footing, soil=soil).compute()
        total = result.term_cohesion + result.term_overburden + result.term_selfweight
        assert total == pytest.approx(result.q_ultimate, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# TEST 12: Meyerhof vs Vesic Factor Methods
# ═══════════════════════════════════════════════════════════════════════

class TestMethodComparison:
    """Compare Vesic vs Meyerhof factor methods."""

    def test_vesic_vs_meyerhof_same_order(self):
        """Both methods should give results in the same order of magnitude."""
        footing = Footing(width=2.0, depth=1.0, shape="square")
        soil = BearingSoilProfile(
            layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
        )
        result_v = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_method="vesic"
        ).compute()
        result_m = BearingCapacityAnalysis(
            footing=footing, soil=soil, factor_method="meyerhof"
        ).compute()

        # Both should be positive and reasonable
        assert result_v.q_ultimate > 0
        assert result_m.q_ultimate > 0
        # Should be within a factor of 2 of each other
        ratio = result_v.q_ultimate / result_m.q_ultimate
        assert 0.5 < ratio < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
