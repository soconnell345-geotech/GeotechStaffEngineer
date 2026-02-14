"""
Tests for drilled shaft axial capacity module.

Covers: shaft geometry, soil profile, alpha/beta/rock side resistance,
end bearing, full analysis, and LRFD factors.

References:
    FHWA GEC-10 (FHWA-NHI-10-016)
"""

import math
import pytest

from drilled_shaft.shaft import DrillShaft
from drilled_shaft.soil_profile import ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.side_resistance import (
    alpha_cohesive, side_resistance_cohesive,
    beta_cohesionless, side_resistance_cohesionless,
    side_resistance_rock, PA,
)
from drilled_shaft.end_bearing import (
    Nc_drilled_shaft, end_bearing_cohesive,
    end_bearing_cohesionless, end_bearing_rock,
)
from drilled_shaft.capacity import DrillShaftAnalysis
from drilled_shaft.results import DrillShaftResult
from drilled_shaft.lrfd import apply_lrfd, get_resistance_factor, RESISTANCE_FACTORS


# ================================================================
# Shaft geometry
# ================================================================
class TestDrillShaft:
    def test_basic_properties(self):
        s = DrillShaft(diameter=1.0, length=20.0)
        assert abs(s.area - math.pi / 4) < 1e-6
        assert abs(s.perimeter - math.pi) < 1e-6
        assert abs(s.tip_area - math.pi / 4) < 1e-6

    def test_bell_tip_area(self):
        s = DrillShaft(diameter=1.0, length=20.0, bell_diameter=2.0)
        assert abs(s.tip_area - math.pi) < 1e-6  # pi*2^2/4

    def test_socket_perimeter(self):
        s = DrillShaft(diameter=1.2, length=20.0, socket_diameter=1.0)
        assert abs(s.socket_perimeter - math.pi * 1.0) < 1e-6
        assert abs(s.perimeter - math.pi * 1.2) < 1e-6

    def test_validation(self):
        with pytest.raises(ValueError):
            DrillShaft(diameter=0, length=10)
        with pytest.raises(ValueError):
            DrillShaft(diameter=1.0, length=-5)
        with pytest.raises(ValueError):
            DrillShaft(diameter=1.0, length=10, bell_diameter=0.5)


# ================================================================
# Soil profile
# ================================================================
class TestSoilProfile:
    def test_effective_stress_no_gwt(self):
        profile = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10, "cohesive", 18.0, cu=50),
        ])
        assert abs(profile.effective_stress_at_depth(5) - 90.0) < 0.1

    def test_effective_stress_with_gwt(self):
        profile = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10, "cohesionless", 19.0, phi=30),
        ], gwt_depth=3.0)
        # 3m * 19 + 2m * (19-9.81) = 57 + 18.38 = 75.38
        assert abs(profile.effective_stress_at_depth(5) - 75.38) < 0.1

    def test_layer_at_depth(self):
        profile = ShaftSoilProfile(layers=[
            ShaftSoilLayer(5, "cohesionless", 18.0, phi=30),
            ShaftSoilLayer(10, "cohesive", 17.0, cu=50),
        ])
        layer = profile.layer_at_depth(3)
        assert layer.soil_type == "cohesionless"
        layer = profile.layer_at_depth(8)
        assert layer.soil_type == "cohesive"

    def test_total_thickness(self):
        profile = ShaftSoilProfile(layers=[
            ShaftSoilLayer(5, "cohesionless", 18.0, phi=30),
            ShaftSoilLayer(10, "cohesive", 17.0, cu=50),
        ])
        assert profile.total_thickness == 15.0

    def test_validation(self):
        with pytest.raises(ValueError):
            ShaftSoilLayer(5, "gravel", 18.0, phi=30)  # bad soil_type
        with pytest.raises(ValueError):
            ShaftSoilLayer(5, "cohesive", 18.0, cu=0)  # cu must be > 0
        with pytest.raises(ValueError):
            ShaftSoilProfile(layers=[])  # empty


# ================================================================
# Alpha method (cohesive)
# ================================================================
class TestAlphaMethod:
    def test_low_cu(self):
        """cu/pa <= 1.5 -> alpha = 0.55."""
        assert alpha_cohesive(50) == 0.55
        assert alpha_cohesive(100) == 0.55
        assert alpha_cohesive(PA * 1.5) == 0.55

    def test_medium_cu(self):
        """1.5 < cu/pa <= 2.5 -> linear decrease."""
        cu = PA * 2.0  # ratio = 2.0
        alpha = alpha_cohesive(cu)
        expected = 0.55 - 0.1 * (2.0 - 1.5)
        assert abs(alpha - expected) < 1e-6

    def test_high_cu(self):
        """cu/pa > 2.5 -> capped at 0.35 minimum."""
        cu = PA * 5.0  # ratio = 5.0
        alpha = alpha_cohesive(cu)
        assert alpha == 0.35

    def test_side_resistance_computation(self):
        """fs = alpha * cu, Qs = fs * perimeter * thickness."""
        cu = 75.0
        alpha = alpha_cohesive(cu)
        perimeter = math.pi * 1.0  # 1m diameter
        Qs = side_resistance_cohesive(cu, perimeter, 5.0)
        expected = alpha * cu * perimeter * 5.0
        assert abs(Qs - expected) < 0.1

    def test_custom_alpha(self):
        Qs = side_resistance_cohesive(100, 3.14, 2.0, alpha=0.3)
        assert abs(Qs - 0.3 * 100 * 3.14 * 2.0) < 0.1


# ================================================================
# Beta method (cohesionless)
# ================================================================
class TestBetaMethod:
    def test_shallow_depth(self):
        """At z=0, beta = 1.5, clamped to 1.2."""
        assert beta_cohesionless(0) == 1.2

    def test_moderate_depth(self):
        """At z=4m, beta = 1.5 - 0.245*2 = 1.01."""
        beta = beta_cohesionless(4.0)
        assert abs(beta - 1.01) < 0.01

    def test_deep(self):
        """At very deep, beta clamped to 0.25."""
        beta = beta_cohesionless(100)
        assert beta == 0.25

    def test_fs_cap_200kpa(self):
        """fs capped at 200 kPa."""
        # Large sigma_v * beta > 200
        sigma_v = 500.0
        beta = 1.0
        Qs = side_resistance_cohesionless(sigma_v, beta, 3.14, 1.0)
        expected = 200.0 * 3.14 * 1.0  # fs capped
        assert abs(Qs - expected) < 0.1

    def test_normal_case(self):
        sigma_v = 100.0
        beta = 0.8
        Qs = side_resistance_cohesionless(sigma_v, beta, 3.14, 2.0)
        expected = 80.0 * 3.14 * 2.0
        assert abs(Qs - expected) < 0.1


# ================================================================
# Rock socket
# ================================================================
class TestRockSocket:
    def test_rough_socket(self):
        """fs = C * sqrt(qu), C=1.0 for rough."""
        qu = 10000  # 10 MPa
        Qs = side_resistance_rock(qu, 3.14, 2.0, C=1.0)
        expected = math.sqrt(10000) * 3.14 * 2.0
        assert abs(Qs - expected) < 0.1

    def test_smooth_socket(self):
        """C = 0.65 for smooth socket."""
        qu = 10000
        Qs = side_resistance_rock(qu, 3.14, 2.0, C=0.65)
        expected = 0.65 * math.sqrt(10000) * 3.14 * 2.0
        assert abs(Qs - expected) < 0.1

    def test_alpha_E_reduction(self):
        """alpha_E reduces for jointed rock."""
        qu = 10000
        Qs_intact = side_resistance_rock(qu, 3.14, 2.0, alpha_E=1.0)
        Qs_jointed = side_resistance_rock(qu, 3.14, 2.0, alpha_E=0.5)
        assert abs(Qs_jointed / Qs_intact - 0.5) < 0.01

    def test_zero_length(self):
        Qs = side_resistance_rock(10000, 3.14, 0.0)
        assert Qs == 0.0


# ================================================================
# End bearing
# ================================================================
class TestEndBearing:
    def test_Nc_short_shaft(self):
        """Nc = 6 + L/D for L/D < 3."""
        assert Nc_drilled_shaft(1.0) == 7.0
        assert Nc_drilled_shaft(2.0) == 8.0

    def test_Nc_long_shaft(self):
        """Nc = 9 for L/D >= 3."""
        assert Nc_drilled_shaft(3.0) == 9.0
        assert Nc_drilled_shaft(10.0) == 9.0

    def test_end_bearing_clay(self):
        cu = 100.0
        area = math.pi * 1.0**2 / 4
        Qt = end_bearing_cohesive(cu, area, L_over_D=10)
        expected = 9.0 * 100 * area
        assert abs(Qt - expected) < 0.1

    def test_end_bearing_sand_N60(self):
        N60 = 30
        area = math.pi * 1.0**2 / 4
        Qt = end_bearing_cohesionless(N60, area)
        expected = 57.5 * 30 * area
        assert abs(Qt - expected) < 0.1

    def test_end_bearing_sand_N60_cap(self):
        """N60 capped at 50."""
        area = math.pi * 1.0**2 / 4
        Qt_50 = end_bearing_cohesionless(50, area)
        Qt_80 = end_bearing_cohesionless(80, area)
        assert abs(Qt_50 - Qt_80) < 0.1

    def test_end_bearing_sand_large_diameter(self):
        """D > 1.27m reduces end bearing."""
        area = math.pi * 2.0**2 / 4
        Qt = end_bearing_cohesionless(30, area, diameter=2.0)
        expected = 57.5 * 30 * (1.27 / 2.0) * area
        assert abs(Qt - expected) < 0.1

    def test_end_bearing_rock_intact(self):
        """RQD >= 70: qb = 2.5 * qu."""
        qu = 5000
        area = math.pi * 1.0**2 / 4
        Qt = end_bearing_rock(qu, area, RQD=80)
        expected = 2.5 * 5000 * area
        assert abs(Qt - expected) < 0.1

    def test_end_bearing_rock_fractured(self):
        """RQD < 70 reduces bearing."""
        area = math.pi * 1.0**2 / 4
        Qt_intact = end_bearing_rock(5000, area, RQD=90)
        Qt_fractured = end_bearing_rock(5000, area, RQD=40)
        assert Qt_fractured < Qt_intact


# ================================================================
# Full analysis
# ================================================================
class TestFullAnalysis:
    def _clay_profile(self):
        shaft = DrillShaft(diameter=1.0, length=15.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(15, "cohesive", 18.0, cu=75),
        ])
        return shaft, soil

    def _mixed_profile(self):
        shaft = DrillShaft(diameter=1.0, length=20.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(5, "cohesionless", 18.0, phi=32, N60=20),
            ShaftSoilLayer(10, "cohesive", 17.0, cu=80),
            ShaftSoilLayer(5, "rock", 22.0, qu=5000, RQD=85),
        ])
        return shaft, soil

    def test_clay_only(self):
        shaft, soil = self._clay_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        assert result.Q_ultimate > 0
        assert result.Q_skin > 0
        assert result.Q_tip > 0
        assert result.Q_side_clay > 0
        assert result.Q_side_sand == 0
        assert result.Q_side_rock == 0
        assert abs(result.Q_ultimate - result.Q_skin - result.Q_tip) < 0.1

    def test_mixed_profile(self):
        shaft, soil = self._mixed_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        assert result.Q_side_clay > 0
        assert result.Q_side_sand > 0
        assert result.Q_side_rock > 0
        assert abs(result.Q_skin - result.Q_side_clay -
                   result.Q_side_sand - result.Q_side_rock) < 0.1

    def test_exclusion_top_1_5m(self):
        """Top 1.5m excluded from side resistance."""
        shaft = DrillShaft(diameter=1.0, length=10.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10, "cohesive", 18.0, cu=75),
        ])
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        # Effective side area = perimeter * (10 - 1.5 - 1.0) = pi * 7.5
        # (1.5m top excl + 1.0m bottom excl for cohesive)
        expected_thickness = 10.0 - 1.5 - 1.0  # 7.5m
        alpha = alpha_cohesive(75)
        expected_Qs = alpha * 75 * math.pi * 1.0 * expected_thickness
        assert abs(result.Q_skin - expected_Qs) < 1.0

    def test_casing_exclusion(self):
        """Casing depth overrides top 1.5m exclusion."""
        shaft = DrillShaft(diameter=1.0, length=15.0, casing_depth=5.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(15, "cohesive", 18.0, cu=75),
        ])
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        # Top exclusion = max(1.5, 5.0) = 5.0m
        # Bottom exclusion = 1.0m (cohesive)
        expected_thickness = 15.0 - 5.0 - 1.0  # 9.0m
        alpha = alpha_cohesive(75)
        expected_Qs = alpha * 75 * math.pi * 1.0 * expected_thickness
        assert abs(result.Q_skin - expected_Qs) < 1.0

    def test_capacity_vs_depth(self):
        shaft, soil = self._mixed_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        cvd = analysis.capacity_vs_depth(depth_min=5, depth_max=20, n_points=5)
        assert len(cvd) == 5
        # Capacity should generally increase with depth
        assert cvd[-1]["Q_ultimate_kN"] > cvd[0]["Q_ultimate_kN"]

    def test_summary_and_dict(self):
        shaft, soil = self._mixed_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        summary = result.summary()
        assert "DRILLED SHAFT" in summary
        d = result.to_dict()
        assert "Q_ultimate_kN" in d
        assert "Q_side_clay_kN" in d

    def test_allowable_capacity(self):
        shaft, soil = self._clay_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil, factor_of_safety=3.0)
        result = analysis.compute()
        assert abs(result.Q_allowable - result.Q_ultimate / 3.0) < 0.1


# ================================================================
# LRFD
# ================================================================
class TestLRFD:
    def test_resistance_factors(self):
        assert get_resistance_factor("side_cohesive") == 0.45
        assert get_resistance_factor("tip_rock") == 0.50
        with pytest.raises(ValueError):
            get_resistance_factor("unknown")

    def test_apply_lrfd(self):
        result = DrillShaftResult(
            Q_side_clay=500, Q_side_sand=300, Q_side_rock=200,
            Q_tip=400, Q_skin=1000, Q_ultimate=1400,
        )
        lrfd = apply_lrfd(result, tip_soil_type="rock")
        assert abs(lrfd["phi_Qs_clay_kN"] - 500 * 0.45) < 0.1
        assert abs(lrfd["phi_Qs_sand_kN"] - 300 * 0.55) < 0.1
        assert abs(lrfd["phi_Qs_rock_kN"] - 200 * 0.55) < 0.1
        assert abs(lrfd["phi_Qt_kN"] - 400 * 0.50) < 0.1

    def test_total_factored(self):
        result = DrillShaftResult(
            Q_side_clay=1000, Q_side_sand=0, Q_side_rock=0,
            Q_tip=500, Q_skin=1000, Q_ultimate=1500,
        )
        lrfd = apply_lrfd(result, tip_soil_type="cohesive")
        expected = 1000 * 0.45 + 500 * 0.40
        assert abs(lrfd["phi_Qn_kN"] - expected) < 0.1
