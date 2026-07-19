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
    phi_prime_from_N1_60, preconsolidation_stress, k0_from_ocr,
    beta_cohesionless_rational, su_to_ciuc, alpha_cohesive_rational,
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
        """At z=4 m (13.12 ft): beta = 1.5 - 0.135*sqrt(13.12) = 1.5 -
        0.245*sqrt(4) = 1.011 (the pre-2026-07-19 code applied the metric
        0.245 coefficient to FEET — the unit-mixing defect found by the
        NHI-06-089 Ex 9-5 curation)."""
        beta = beta_cohesionless(4.0)
        expected = 1.5 - 0.135 * math.sqrt(4.0 * 3.28084)
        assert abs(beta - expected) < 0.002
        assert abs(beta - (1.5 - 0.245 * math.sqrt(4.0))) < 0.001

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

    def test_beta_at_5m(self):
        """At z=5 m (16.4 ft): beta = 1.5 - 0.135*sqrt(16.4) = 0.953."""
        beta = beta_cohesionless(5.0)
        expected = 1.5 - 0.135 * math.sqrt(5.0 * 3.28084)
        assert abs(beta - expected) < 0.002

    def test_beta_deep_clamped(self):
        """Raw beta reaches the 0.25 floor at z = (1.25/0.245)^2 = 26.0 m
        (85.4 ft — matching the feet-form floor at (1.25/0.135)^2 = 85.7 ft)."""
        assert beta_cohesionless(26.5) == 0.25
        assert beta_cohesionless(25.0) > 0.25

    def test_beta_matches_imperial(self):
        """The SI form (0.245, z in m) must equal the O'Neill & Reese
        feet form (0.135, z in ft) — the identity 0.135*sqrt(3.28084)=0.2445
        that the pre-fix code violated by double-converting."""
        for z_m in [1.0, 2.0, 3.0, 5.0, 7.5, 15.0]:
            z_ft = z_m * 3.28084
            beta_imperial = max(0.25, min(1.5 - 0.135 * math.sqrt(z_ft), 1.2))
            beta_si = beta_cohesionless(z_m)
            assert abs(beta_si - beta_imperial) < 2e-3, \
                f"Mismatch at z={z_m}m: SI={beta_si:.6f} vs imperial={beta_imperial:.6f}"


# ================================================================
# Rock socket
# ================================================================
class TestRockSocket:
    def test_explicit_C_1(self):
        """fs = C * alpha_E * sqrt(qu * pa), with explicit C=1.0."""
        qu = 10000  # 10 MPa
        Qs = side_resistance_rock(qu, 3.14, 2.0, C=1.0)
        expected = 1.0 * math.sqrt(10000 * PA) * 3.14 * 2.0
        assert abs(Qs - expected) < 0.1

    def test_explicit_C_065(self):
        """C = 0.65 with atmospheric pressure normalization."""
        qu = 10000
        Qs = side_resistance_rock(qu, 3.14, 2.0, C=0.65)
        expected = 0.65 * math.sqrt(10000 * PA) * 3.14 * 2.0
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

    def test_default_C_is_065(self):
        """Default C parameter should be 0.65 per GEC-10."""
        qu = 10000
        Qs_default = side_resistance_rock(qu, 3.14, 2.0)
        Qs_explicit = side_resistance_rock(qu, 3.14, 2.0, C=0.65)
        assert abs(Qs_default - Qs_explicit) < 0.001

    def test_rock_socket_uses_pa(self):
        """Formula must include atmospheric pressure normalization."""
        qu = 10000  # kPa
        Qs = side_resistance_rock(qu, 1.0, 1.0)  # unit perimeter, unit thickness
        expected_fs = 0.65 * math.sqrt(10000 * PA)
        assert abs(Qs - expected_fs) < 0.1

    def test_rock_socket_known_value(self):
        """Hand calc: qu=5000, C=0.65, alpha_E=1.0.
        fs = 0.65 * sqrt(5000 * 101.325) = 0.65 * 711.78 = 462.66 kPa.
        """
        qu = 5000
        Qs = side_resistance_rock(qu, 1.0, 1.0)
        expected_fs = 0.65 * math.sqrt(5000 * PA)
        assert abs(Qs - expected_fs) < 0.1

    def test_explicit_C_override(self):
        """User can still pass C=1.0 to override default."""
        qu = 10000
        Qs = side_resistance_rock(qu, 1.0, 1.0, C=1.0)
        expected_fs = 1.0 * math.sqrt(10000 * PA)
        assert abs(Qs - expected_fs) < 0.1


# ================================================================
# End bearing
# ================================================================
class TestEndBearing:
    def test_Nc_short_shaft(self):
        """O'Neill-Reese: Nc = 6*(1 + 0.2*L/D) for L/D < 2.5 (DS-2)."""
        assert Nc_drilled_shaft(0.0) == pytest.approx(6.0)
        assert Nc_drilled_shaft(1.0) == pytest.approx(7.2)
        assert Nc_drilled_shaft(2.0) == pytest.approx(8.4)

    def test_Nc_long_shaft(self):
        """Nc caps at 9 from L/D = 2.5 (O'Neill-Reese), not 3 (DS-2)."""
        assert Nc_drilled_shaft(2.5) == pytest.approx(9.0)
        assert Nc_drilled_shaft(2.75) == pytest.approx(9.0)  # 2.5-3 band
        assert Nc_drilled_shaft(3.0) == pytest.approx(9.0)
        assert Nc_drilled_shaft(10.0) == pytest.approx(9.0)

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

    def test_dict_exposes_side_resistance_by_layer(self):
        """to_dict() surfaces the per-layer side-resistance breakdown so the
        agent can quote the mobilized alpha/beta/fs instead of guessing."""
        shaft, soil = self._mixed_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
        result = analysis.compute()
        d = result.to_dict()
        assert "side_resistance_by_layer" in d
        assert "side_resistance_note" in d
        layers = d["side_resistance_by_layer"]
        assert layers is result.layer_breakdown
        # Every layer carries a unit side resistance.
        assert all("fs_kPa" in lb for lb in layers)
        # The cohesive layer's method string quotes the GEC-10 alpha.
        cohesive = [lb for lb in layers if lb["soil_type"] == "cohesive"]
        assert cohesive
        assert any(lb["method"].startswith("alpha=") for lb in cohesive)

    def test_dict_omits_layer_breakdown_when_absent(self):
        """No layer_breakdown -> the additive keys stay out of the dict."""
        result = DrillShaftResult(Q_ultimate=100.0, Q_skin=60.0, Q_tip=40.0)
        d = result.to_dict()
        assert "side_resistance_by_layer" not in d
        assert "side_resistance_note" not in d

    def test_allowable_capacity(self):
        shaft, soil = self._clay_profile()
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil, factor_of_safety=3.0)
        result = analysis.compute()
        assert abs(result.Q_allowable - result.Q_ultimate / 3.0) < 0.1


# ================================================================
# v5.1 QC regression tests (DS-1, DS-3, DS-4, DS-5)
# ================================================================
class TestCohesiveEndBearingCapAndLargeBase:
    """DS-1: clay end bearing capped at 80 ksf (3830 kPa) and reduced
    for base diameters > 1.90 m (O'Neill & Reese 1999 / AASHTO)."""

    def test_qb_cap_high_cu(self):
        """cu = 600 kPa, Nc = 9 -> Nc*cu = 5400 > 3830: qb is capped."""
        area = math.pi * 1.0**2 / 4
        Qt = end_bearing_cohesive(600.0, area, L_over_D=10)
        assert Qt == pytest.approx(3830.0 * area, rel=1e-9)

    def test_qb_below_cap_unchanged(self):
        """cu = 100 kPa -> Nc*cu = 900 < 3830: no cap, behavior unchanged."""
        area = math.pi * 1.0**2 / 4
        Qt = end_bearing_cohesive(100.0, area, L_over_D=10)
        assert Qt == pytest.approx(9.0 * 100.0 * area, rel=1e-9)

    def test_large_base_reduction_hand_calc(self):
        """Bb = 2.5 m > 1.90 m: Fr per O'Neill-Reese, hand-checked."""
        cu, L, Bb = 200.0, 20.0, 2.5
        area = math.pi * Bb**2 / 4
        # a = min(0.0071 + 0.0021*(20/2.5), 0.015) = 0.015
        a = min(0.0071 + 0.0021 * (L / Bb), 0.015)
        assert a == 0.015
        # b = 0.45*sqrt(cu_ksf), cu_ksf = 200/47.8803
        b = 0.45 * math.sqrt(200.0 / 47.8803)
        b = min(max(b, 0.5), 1.5)
        Fr = min(2.5 / (a * Bb * 39.3701 + 2.5 * b), 1.0)
        assert Fr < 1.0
        Qt = end_bearing_cohesive(cu, area, L_over_D=10,
                                  base_diameter=Bb, shaft_length=L)
        assert Qt == pytest.approx(9.0 * cu * Fr * area, rel=1e-6)

    def test_no_reduction_below_190(self):
        """Base diameter <= 1.90 m: no large-base reduction."""
        area = math.pi * 1.5**2 / 4
        Qt_base = end_bearing_cohesive(150.0, area, L_over_D=10,
                                       base_diameter=1.5, shaft_length=15.0)
        Qt_none = end_bearing_cohesive(150.0, area, L_over_D=10)
        assert Qt_base == pytest.approx(Qt_none, rel=1e-12)

    def test_full_analysis_high_cu_large_bell(self):
        """End-to-end: stiff clay (cu high enough to hit the cap) with a
        large bell -> tip equals capped qb x Fr x bell area."""
        Bb, L = 2.5, 15.0
        shaft = DrillShaft(diameter=1.2, length=L, bell_diameter=Bb)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(L, "cohesive", 19.0, cu=500.0),
        ])
        result = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()
        L_over_D = L / 1.2  # Nc uses shaft diameter; > 2.5 -> Nc = 9
        qb = min(9.0 * 500.0, 3830.0)
        a = min(0.0071 + 0.0021 * (L / Bb), 0.015)
        b = min(max(0.45 * math.sqrt(500.0 / 47.8803), 0.5), 1.5)
        Fr = min(2.5 / (a * Bb * 39.3701 + 2.5 * b), 1.0)
        expected_Qt = qb * Fr * shaft.tip_area
        assert result.Q_tip == pytest.approx(expected_Qt, rel=1e-6)


class TestBelledBaseDiameterCohesionless:
    """DS-3: the 1.27/D large-diameter tip reduction in sand uses the
    bell (base) diameter, not the shaft diameter."""

    def test_full_analysis_bell_governs_reduction(self):
        """Shaft D = 1.0 (no reduction alone), bell 2.0 m -> 1.27/2.0
        reduction applied to the bell-area tip."""
        shaft = DrillShaft(diameter=1.0, length=15.0, bell_diameter=2.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(15, "cohesionless", 18.5, phi=33, N60=30),
        ])
        result = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()
        expected_Qt = end_bearing_cohesionless(30, shaft.tip_area,
                                               diameter=2.0)
        assert result.Q_tip == pytest.approx(expected_Qt, rel=1e-9)
        # And the reduction actually engaged (bell > 1.27 m)
        unreduced = 57.5 * 30 * shaft.tip_area
        assert result.Q_tip == pytest.approx(unreduced * 1.27 / 2.0,
                                             rel=1e-9)

    def test_unbelled_shaft_unchanged(self):
        """No bell: shaft diameter still governs (regression)."""
        shaft = DrillShaft(diameter=1.5, length=15.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(15, "cohesionless", 18.5, phi=33, N60=30),
        ])
        result = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()
        expected_Qt = end_bearing_cohesionless(30, shaft.tip_area,
                                               diameter=1.5)
        assert result.Q_tip == pytest.approx(expected_Qt, rel=1e-9)


class TestBetaN60Reduction:
    """DS-4: O'Neill-Reese beta reduced by N60/15 for loose sand
    (N60 < 15); no reduction when N60 unmeasured or >= 15."""

    def test_beta_reduced_proportionally(self):
        z = 5.0
        base = beta_cohesionless(z)
        assert beta_cohesionless(z, N60=10) == pytest.approx(
            (10.0 / 15.0) * base, rel=1e-12)

    def test_no_reduction_at_or_above_15(self):
        z = 5.0
        base = beta_cohesionless(z)
        assert beta_cohesionless(z, N60=15) == pytest.approx(base)
        assert beta_cohesionless(z, N60=40) == pytest.approx(base)

    def test_none_means_no_reduction(self):
        z = 5.0
        assert beta_cohesionless(z, N60=None) == pytest.approx(
            beta_cohesionless(z))

    def test_full_analysis_loose_sand_side(self):
        """Loose sand (N60 = 9) gives 9/15 of the unmeasured-N60 side
        resistance; clay tip layer isolates the side-resistance effect."""
        def make(n60):
            shaft = DrillShaft(diameter=1.0, length=15.0)
            soil = ShaftSoilProfile(layers=[
                ShaftSoilLayer(10, "cohesionless", 18.5, phi=30, N60=n60),
                ShaftSoilLayer(5, "cohesive", 17.0, cu=80),
            ])
            return DrillShaftAnalysis(shaft=shaft, soil=soil).compute()

        r_loose = make(9)
        r_unmeasured = make(0)  # N60=0 -> treated as not measured
        assert r_loose.Q_side_sand == pytest.approx(
            (9.0 / 15.0) * r_unmeasured.Q_side_sand, rel=1e-9)
        # Tip (clay) unaffected
        assert r_loose.Q_tip == pytest.approx(r_unmeasured.Q_tip, rel=1e-12)


class TestCapacityVsDepthReentrant:
    """DS-5: capacity_vs_depth must not mutate the shared shaft.length
    and must return the same values as independent per-depth analyses."""

    def test_no_mutation_and_identical_results(self):
        shaft = DrillShaft(diameter=1.0, length=20.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(5, "cohesionless", 18.0, phi=32, N60=20),
            ShaftSoilLayer(15, "cohesive", 17.0, cu=80),
        ])
        analysis = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                      factor_of_safety=2.5)
        cvd = analysis.capacity_vs_depth(depth_min=5, depth_max=20,
                                         n_points=4)
        assert shaft.length == 20.0  # not mutated
        assert len(cvd) == 4
        for row in cvd:
            trial_shaft = DrillShaft(diameter=1.0, length=row["depth_m"])
            fresh = DrillShaftAnalysis(shaft=trial_shaft, soil=soil,
                                       factor_of_safety=2.5).compute()
            assert row["Q_ultimate_kN"] == pytest.approx(
                round(fresh.Q_ultimate, 1))
            assert row["Q_skin_kN"] == pytest.approx(
                round(fresh.Q_skin, 1))
            assert row["Q_tip_kN"] == pytest.approx(
                round(fresh.Q_tip, 1))


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


# ================================================================
# GEC-10 rational side-resistance chains (opt-in, default-preserving)
# ================================================================
class TestRationalSideResistance:
    def test_phi_from_N1_60(self):
        # (N1)60 = 21 -> phi' = 27.5 + 9.2*log10(21) ~ 39.66 deg (GEC-10 App A: 40)
        assert phi_prime_from_N1_60(21) == pytest.approx(39.66, abs=0.05)
        with pytest.raises(ValueError):
            phi_prime_from_N1_60(0)

    def test_preconsolidation_stress(self):
        # sigma'p = 0.47*pa*N60^0.6; N60=30, pa=101.325 -> 366.5 kPa
        assert preconsolidation_stress(30) == pytest.approx(366.5, rel=0.01)
        with pytest.raises(ValueError):
            preconsolidation_stress(-1)

    def test_k0_from_ocr(self):
        # Ko = (1-sin40)*1.65^sin40 ~ 0.49
        assert k0_from_ocr(40.0, 1.65) == pytest.approx(0.49, abs=0.01)
        # NC (OCR=1): Ko = 1 - sin phi'
        assert k0_from_ocr(30.0, 1.0) == pytest.approx(0.5, abs=1e-6)

    def test_beta_rational_matches_gec10(self):
        # phi'=40, OCR=1.65, delta=phi' -> beta ~ 0.41 (GEC-10 App A Layer 3)
        assert beta_cohesionless_rational(40.0, 1.65) == pytest.approx(0.41, abs=0.01)

    def test_beta_rational_delta_override(self):
        # delta < phi' lowers beta
        full = beta_cohesionless_rational(40.0, 1.65)
        reduced = beta_cohesionless_rational(40.0, 1.65, delta=30.0)
        assert reduced < full

    def test_su_to_ciuc_transforms(self):
        su_uu, sv0 = 1750 * 0.04788, 2114 * 0.04788  # kPa
        # UC pair (0.893/0.513): su(CIUC) ~ 2057 psf (GEC-10 App A Layer 4)
        assert su_to_ciuc(su_uu, sv0, "uc") / 0.04788 == pytest.approx(2057, rel=0.01)
        # UU pair differs slightly
        assert su_to_ciuc(su_uu, sv0, "uu") != su_to_ciuc(su_uu, sv0, "uc")
        # ciuc -> unchanged
        assert su_to_ciuc(su_uu, sv0, "ciuc") == su_uu
        with pytest.raises(ValueError):
            su_to_ciuc(su_uu, sv0, "bogus")

    def test_alpha_rational_matches_gec10(self):
        su_ciuc = 2057 * 0.04788
        assert alpha_cohesive_rational(su_ciuc) == pytest.approx(0.47, abs=0.01)
        with pytest.raises(ValueError):
            alpha_cohesive_rational(0)

    def test_high_level_beta_rational_vs_depth_default(self):
        """beta_method='rational' changes the sand side resistance; the default
        'depth' path is unchanged."""
        shaft = DrillShaft(diameter=8 * 0.3048, length=3.0 + 20 * 0.3048, casing_depth=3.0)
        layers = [
            ShaftSoilLayer(3.0, "cohesionless", 18.0, phi=30),
            ShaftSoilLayer(20 * 0.3048, "cohesionless", 17.9, phi=40, N60=30,
                           N1_60=21, sigma_v_ref=4645 * 0.04788, description="sand"),
        ]
        soil = ShaftSoilProfile(layers=layers)
        rational = DrillShaftAnalysis(shaft=shaft, soil=soil, beta_method="rational").compute()
        depth = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()  # default
        sand_r = next(lb for lb in rational.layer_breakdown if lb.get("description") == "sand")
        sand_d = next(lb for lb in depth.layer_breakdown if lb.get("description") == "sand")
        assert "rational" in sand_r["method"]
        assert "rational" not in sand_d["method"]
        # The two beta bases give genuinely different results (rational beta
        # 0.413 here vs the corrected depth-based O'Neill & Reese values ~0.9
        # at mid-layer). The pre-2026-07-19 assertion "rational > depth" only
        # held because the depth beta was unit-mixing-broken (floored at 0.25
        # everywhere below ~8 m); with the corrected formula the depth basis
        # exceeds rational for this profile, and both are positive/distinct.
        assert sand_r["side_resistance_kN"] > 0
        assert sand_d["side_resistance_kN"] > 0
        assert (abs(sand_r["side_resistance_kN"] - sand_d["side_resistance_kN"])
                / sand_d["side_resistance_kN"] > 0.05)

    def test_high_level_alpha_rational_vs_aashto_default(self):
        shaft = DrillShaft(diameter=8 * 0.3048, length=3.0 + 15 * 0.3048 + 3.0, casing_depth=3.0)
        layers = [
            ShaftSoilLayer(3.0, "cohesive", 18.0, cu=100.0),
            ShaftSoilLayer(15 * 0.3048, "cohesive", 20.3, cu=1750 * 0.04788, description="clay"),
            ShaftSoilLayer(3.0, "cohesionless", 19.0, phi=35),
        ]
        soil = ShaftSoilProfile(layers=layers)
        rational = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                      alpha_method="rational", su_test_type="uc").compute()
        aashto = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()  # default
        clay_r = next(lb for lb in rational.layer_breakdown if lb.get("description") == "clay")
        clay_a = next(lb for lb in aashto.layer_breakdown if lb.get("description") == "clay")
        assert float(clay_r["method"].split("=")[1].split()[0]) == pytest.approx(0.47, abs=0.02)
        assert float(clay_a["method"].split("=")[1].split()[0]) == pytest.approx(0.55, abs=1e-6)

    def test_invalid_method_selectors_raise(self):
        shaft = DrillShaft(diameter=1.0, length=10.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10.0, "cohesionless", 18.0, phi=32, N60=20)])
        with pytest.raises(ValueError):
            DrillShaftAnalysis(shaft=shaft, soil=soil, beta_method="bogus")
        with pytest.raises(ValueError):
            DrillShaftAnalysis(shaft=shaft, soil=soil, alpha_method="bogus")
        with pytest.raises(ValueError):
            DrillShaftAnalysis(shaft=shaft, soil=soil, su_test_type="bogus")

    def test_rational_beta_requires_N60_or_ocr(self):
        shaft = DrillShaft(diameter=1.0, length=10.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10.0, "cohesionless", 18.0, phi=32)])  # no N60, no OCR
        with pytest.raises(ValueError):
            DrillShaftAnalysis(shaft=shaft, soil=soil, beta_method="rational").compute()

    def test_rational_beta_direct_ocr_override(self):
        """A layer OCR override bypasses the sigma'p/N60 computation."""
        shaft = DrillShaft(diameter=1.0, length=10.0)
        soil = ShaftSoilProfile(layers=[
            ShaftSoilLayer(10.0, "cohesionless", 18.0, phi=40, OCR=1.65)])
        result = DrillShaftAnalysis(shaft=shaft, soil=soil, beta_method="rational").compute()
        sand = next(lb for lb in result.layer_breakdown if lb["soil_type"] == "cohesionless")
        assert float(sand["method"].split("=")[1].split()[0]) == pytest.approx(0.41, abs=0.02)
