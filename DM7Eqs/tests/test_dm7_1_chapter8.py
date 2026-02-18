"""Comprehensive tests for geotech.dm7_1.chapter8 -- Correlations for Soil and Rock.

Tests cover all 45 public functions (Equations 8-1 through 8-45) with:
  - Basic valid-input tests using hand-calculated expected values
  - Edge-case tests where applicable
  - ValueError checks for every validation branch

Reference: UFC 3-220-10, Chapter 8.
"""

import math

import pytest

from geotech.dm7_1.chapter8 import *


# ============================================================================
# Eq 8-1: spt_n_correction_for_pore_pressure
# ============================================================================


class TestSptNCorrectionForPorePressure:
    """Tests for Equation 8-1."""

    def test_basic_below_threshold(self):
        # N60 = 10 <= 15, returned unchanged
        assert spt_n_correction_for_pore_pressure(10.0) == pytest.approx(10.0, rel=1e-4)

    def test_basic_above_threshold(self):
        # N60 = 25: N' = 15 + 0.5*(25-15) = 15 + 5 = 20
        assert spt_n_correction_for_pore_pressure(25.0) == pytest.approx(20.0, rel=1e-4)

    def test_at_threshold(self):
        # N60 = 15: exactly at boundary, returned unchanged
        assert spt_n_correction_for_pore_pressure(15.0) == pytest.approx(15.0, rel=1e-4)

    def test_zero(self):
        assert spt_n_correction_for_pore_pressure(0.0) == pytest.approx(0.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n60 must be non-negative"):
            spt_n_correction_for_pore_pressure(-1.0)


# ============================================================================
# Eq 8-2: spt_friction_angle_roads
# ============================================================================


class TestSptFrictionAngleRoads:
    """Tests for Equation 8-2."""

    def test_basic(self):
        # N60 = 20: phi' = sqrt(15*20) + 15 = sqrt(300) + 15 = 17.3205 + 15 = 32.3205
        expected = math.sqrt(300.0) + 15.0
        assert spt_friction_angle_roads(20.0) == pytest.approx(expected, rel=1e-4)

    def test_zero(self):
        # N60 = 0: phi' = sqrt(0) + 15 = 15
        assert spt_friction_angle_roads(0.0) == pytest.approx(15.0, rel=1e-4)

    def test_large_value(self):
        # N60 = 60: phi' = sqrt(900) + 15 = 30 + 15 = 45
        assert spt_friction_angle_roads(60.0) == pytest.approx(45.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n60 must be non-negative"):
            spt_friction_angle_roads(-5.0)


# ============================================================================
# Eq 8-3: spt_friction_angle_buildings
# ============================================================================


class TestSptFrictionAngleBuildings:
    """Tests for Equation 8-3."""

    def test_basic(self):
        # N60 = 20: phi' = 0.3*20 + 27 = 6 + 27 = 33
        assert spt_friction_angle_buildings(20.0) == pytest.approx(33.0, rel=1e-4)

    def test_zero(self):
        # N60 = 0: phi' = 27
        assert spt_friction_angle_buildings(0.0) == pytest.approx(27.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n60 must be non-negative"):
            spt_friction_angle_buildings(-1.0)


# ============================================================================
# Eq 8-4: spt_friction_angle_kulhawy_mayne
# ============================================================================


class TestSptFrictionAngleKulhawyMayne:
    """Tests for Equation 8-4."""

    def test_basic(self):
        # N60=20, sigma_v_eff=100, pa=101.325
        # ratio = 20 / (12.2 + 20.3*(100/101.325))
        # = 20 / (12.2 + 20.3*0.98692) = 20 / (12.2 + 20.03447) = 20 / 32.23447
        # tan(phi) = 0.620455 ^ 0.34
        # 0.620455^0.34 = exp(0.34 * ln(0.620455)) = exp(0.34 * (-0.47836)) = exp(-0.16264) = 0.84987
        # phi = atan(0.84987) = 40.369 deg
        ratio = 20.0 / (12.2 + 20.3 * (100.0 / 101.325))
        tan_phi = ratio ** 0.34
        expected = math.degrees(math.atan(tan_phi))
        assert spt_friction_angle_kulhawy_mayne(20.0, 100.0, 101.325) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_n60(self):
        # N60=0 => ratio=0, tan(phi)=0^0.34=0, phi=0
        assert spt_friction_angle_kulhawy_mayne(0.0, 100.0, 101.325) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_negative_n60_raises(self):
        with pytest.raises(ValueError, match="n60 must be non-negative"):
            spt_friction_angle_kulhawy_mayne(-1.0, 100.0, 101.325)

    def test_negative_sigma_v_eff_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be non-negative"):
            spt_friction_angle_kulhawy_mayne(20.0, -1.0, 101.325)

    def test_zero_pa_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            spt_friction_angle_kulhawy_mayne(20.0, 100.0, 0.0)

    def test_negative_pa_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            spt_friction_angle_kulhawy_mayne(20.0, 100.0, -1.0)


# ============================================================================
# Eq 8-5: spt_friction_angle_wolff
# ============================================================================


class TestSptFrictionAngleWolff:
    """Tests for Equation 8-5."""

    def test_basic(self):
        # N1,60 = 20: phi' = 27.1 + 0.3*20 - 0.00054*20^2
        # = 27.1 + 6.0 - 0.00054*400 = 27.1 + 6.0 - 0.216 = 32.884
        assert spt_friction_angle_wolff(20.0) == pytest.approx(32.884, rel=1e-4)

    def test_zero(self):
        # N1,60 = 0: phi' = 27.1
        assert spt_friction_angle_wolff(0.0) == pytest.approx(27.1, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n1_60 must be non-negative"):
            spt_friction_angle_wolff(-1.0)


# ============================================================================
# Eq 8-6: spt_friction_angle_hatanaka_uchida
# ============================================================================


class TestSptFrictionAngleHatanakaUchida:
    """Tests for Equation 8-6."""

    def test_basic(self):
        # N1,60 = 25: phi' = sqrt(15.4*25) + 20 = sqrt(385) + 20 = 19.6214 + 20 = 39.6214
        expected = math.sqrt(15.4 * 25.0) + 20.0
        assert spt_friction_angle_hatanaka_uchida(25.0) == pytest.approx(expected, rel=1e-4)

    def test_zero(self):
        # N1,60 = 0: phi' = 0 + 20 = 20
        assert spt_friction_angle_hatanaka_uchida(0.0) == pytest.approx(20.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="n1_60 must be non-negative"):
            spt_friction_angle_hatanaka_uchida(-1.0)


# ============================================================================
# Eq 8-7: cpt_friction_angle_mayne
# ============================================================================


class TestCptFrictionAngleMayne:
    """Tests for Equation 8-7."""

    def test_basic(self):
        # qt=5000, sigma_v_eff=100, pa=101.325
        # phi' = 17.6 + 11 * log10((5000/101.325)/(100/101.325))
        # = 17.6 + 11 * log10(5000/100) = 17.6 + 11 * log10(50)
        # = 17.6 + 11 * 1.69897 = 17.6 + 18.6887 = 36.2887
        expected = 17.6 + 11.0 * math.log10(5000.0 / 100.0)
        assert cpt_friction_angle_mayne(5000.0, 100.0, 101.325) == pytest.approx(
            expected, rel=1e-4
        )

    def test_qt_not_positive_raises(self):
        with pytest.raises(ValueError, match="qt must be positive"):
            cpt_friction_angle_mayne(0.0, 100.0, 101.325)

    def test_sigma_v_eff_not_positive_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            cpt_friction_angle_mayne(5000.0, 0.0, 101.325)

    def test_pa_not_positive_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            cpt_friction_angle_mayne(5000.0, 100.0, 0.0)


# ============================================================================
# Eq 8-8: cpt_friction_angle_robertson_campanella
# ============================================================================


class TestCptFrictionAngleRobertsonCampanella:
    """Tests for Equation 8-8."""

    def test_basic(self):
        # qc=5000, sigma_v_eff=100
        # log10(5000/100) = log10(50) = 1.69897
        # denom = 2.68 * (1.69897 + 0.29) = 2.68 * 1.98897 = 5.33044
        # phi = degrees(atan(1/5.33044)) = degrees(atan(0.18760)) = 10.627 deg
        log_ratio = math.log10(5000.0 / 100.0)
        denom = 2.68 * (log_ratio + 0.29)
        expected = math.degrees(math.atan(1.0 / denom))
        assert cpt_friction_angle_robertson_campanella(5000.0, 100.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_qc_not_positive_raises(self):
        with pytest.raises(ValueError, match="qc must be positive"):
            cpt_friction_angle_robertson_campanella(0.0, 100.0)

    def test_sigma_v_eff_not_positive_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            cpt_friction_angle_robertson_campanella(5000.0, 0.0)

    def test_denominator_nonpositive_raises(self):
        # Need log10(qc/sigma_v_eff) + 0.29 <= 0
        # log10(qc/sigma_v_eff) <= -0.29 => qc/sigma_v_eff <= 10^(-0.29) = 0.5129
        # e.g. qc=50, sigma_v_eff=100 => ratio=0.5, log10(0.5)=-0.3010
        # denom = 2.68*(-0.3010 + 0.29) = 2.68*(-0.0110) = -0.02948 < 0
        with pytest.raises(ValueError, match="Computed denominator is non-positive"):
            cpt_friction_angle_robertson_campanella(50.0, 100.0)


# ============================================================================
# Eq 8-9: cpt_oc_nc_resistance_ratio
# ============================================================================


class TestCptOcNcResistanceRatio:
    """Tests for Equation 8-9."""

    def test_basic(self):
        # OCR=4, beta=0.5: R = 1 + 0.75*(4^0.5 - 1) = 1 + 0.75*(2-1) = 1 + 0.75 = 1.75
        assert cpt_oc_nc_resistance_ratio(4.0, 0.5) == pytest.approx(1.75, rel=1e-4)

    def test_nc_soil(self):
        # OCR=1 (NC): R = 1 + 0.75*(1^beta - 1) = 1 + 0 = 1 for any beta
        assert cpt_oc_nc_resistance_ratio(1.0, 0.8) == pytest.approx(1.0, rel=1e-4)

    def test_ocr_less_than_one_raises(self):
        with pytest.raises(ValueError, match="ocr must be >= 1.0"):
            cpt_oc_nc_resistance_ratio(0.5, 0.5)


# ============================================================================
# Eq 8-10: dmt_horizontal_stress_index
# ============================================================================


class TestDmtHorizontalStressIndex:
    """Tests for Equation 8-10."""

    def test_basic(self):
        # p0=300, u0=50, sigma_v_eff=100: KD = (300-50)/100 = 2.5
        assert dmt_horizontal_stress_index(300.0, 50.0, 100.0) == pytest.approx(
            2.5, rel=1e-4
        )

    def test_zero_pore_pressure(self):
        # p0=200, u0=0, sigma_v_eff=100: KD = 200/100 = 2.0
        assert dmt_horizontal_stress_index(200.0, 0.0, 100.0) == pytest.approx(
            2.0, rel=1e-4
        )

    def test_sigma_v_eff_not_positive_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            dmt_horizontal_stress_index(300.0, 50.0, 0.0)

    def test_sigma_v_eff_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            dmt_horizontal_stress_index(300.0, 50.0, -10.0)


# ============================================================================
# Eq 8-11: dmt_friction_angle_upper_bound
# ============================================================================


class TestDmtFrictionAngleUpperBound:
    """Tests for Equation 8-11."""

    def test_basic(self):
        # KD=5: phi' = 31 + 5/(0.236 + 0.066*5) = 31 + 5/(0.236+0.33) = 31 + 5/0.566 = 31 + 8.8339
        expected = 31.0 + 5.0 / (0.236 + 0.066 * 5.0)
        assert dmt_friction_angle_upper_bound(5.0) == pytest.approx(expected, rel=1e-4)

    def test_small_kd(self):
        # KD=1: phi' = 31 + 1/(0.236+0.066) = 31 + 1/0.302 = 31 + 3.3113
        expected = 31.0 + 1.0 / 0.302
        assert dmt_friction_angle_upper_bound(1.0) == pytest.approx(expected, rel=1e-4)

    def test_denominator_nonpositive_raises(self):
        # 0.236 + 0.066*kd <= 0 => kd <= -0.236/0.066 = -3.5758
        with pytest.raises(ValueError, match="Denominator.*non-positive"):
            dmt_friction_angle_upper_bound(-4.0)


# ============================================================================
# Eq 8-12: dmt_friction_angle_lower_bound
# ============================================================================


class TestDmtFrictionAngleLowerBound:
    """Tests for Equation 8-12."""

    def test_basic(self):
        # KD=5: log10(5) = 0.69897
        # phi' = 28 + 14.6*0.69897 - 2.1*0.69897^2
        # = 28 + 10.2050 - 2.1*0.48856 = 28 + 10.2050 - 1.0260 = 37.179
        log_kd = math.log10(5.0)
        expected = 28.0 + 14.6 * log_kd - 2.1 * log_kd ** 2
        assert dmt_friction_angle_lower_bound(5.0) == pytest.approx(expected, rel=1e-4)

    def test_kd_equals_one(self):
        # KD=1: log10(1)=0 => phi' = 28
        assert dmt_friction_angle_lower_bound(1.0) == pytest.approx(28.0, rel=1e-4)

    def test_kd_not_positive_raises(self):
        with pytest.raises(ValueError, match="kd must be positive"):
            dmt_friction_angle_lower_bound(0.0)

    def test_kd_negative_raises(self):
        with pytest.raises(ValueError, match="kd must be positive"):
            dmt_friction_angle_lower_bound(-1.0)


# ============================================================================
# Eq 8-13: fully_softened_friction_angle
# ============================================================================


class TestFullySoftenedFrictionAngle:
    """Tests for Equation 8-13."""

    def test_basic(self):
        # PI=30: phi'FS = -0.0058*30^1.73 + 0.32*30 + 36.2
        # 30^1.73 = exp(1.73*ln(30)) = exp(1.73*3.40120) = exp(5.88407) = 359.05
        # = -0.0058*359.05 + 9.6 + 36.2 = -2.0825 + 9.6 + 36.2 = 43.7175
        expected = -0.0058 * 30.0 ** 1.73 + 0.32 * 30.0 + 36.2
        assert fully_softened_friction_angle(30.0) == pytest.approx(expected, rel=1e-4)

    def test_zero_pi(self):
        # PI=0: phi'FS = 0 + 0 + 36.2 = 36.2
        assert fully_softened_friction_angle(0.0) == pytest.approx(36.2, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be non-negative"):
            fully_softened_friction_angle(-1.0)


# ============================================================================
# Eq 8-14: secant_friction_angle_shear_strength
# ============================================================================


class TestSecantFrictionAngleShearStrength:
    """Tests for Equation 8-14."""

    def test_basic(self):
        # sigma_ff_eff=100, phi_sec=30 deg
        # s = 100 * tan(30 deg) = 100 * 0.57735 = 57.735
        expected = 100.0 * math.tan(math.radians(30.0))
        assert secant_friction_angle_shear_strength(100.0, 30.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_stress(self):
        # sigma_ff_eff=0 => s=0
        assert secant_friction_angle_shear_strength(0.0, 30.0) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_zero_angle(self):
        # phi_sec=0 => s=0
        assert secant_friction_angle_shear_strength(100.0, 0.0) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_negative_stress_raises(self):
        with pytest.raises(ValueError, match="sigma_ff_eff must be non-negative"):
            secant_friction_angle_shear_strength(-1.0, 30.0)

    def test_angle_below_zero_raises(self):
        with pytest.raises(ValueError, match="phi_sec_deg must be in"):
            secant_friction_angle_shear_strength(100.0, -1.0)

    def test_angle_at_90_raises(self):
        with pytest.raises(ValueError, match="phi_sec_deg must be in"):
            secant_friction_angle_shear_strength(100.0, 90.0)

    def test_angle_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_sec_deg must be in"):
            secant_friction_angle_shear_strength(100.0, 91.0)


# ============================================================================
# Eq 8-15: power_function_shear_strength
# ============================================================================


class TestPowerFunctionShearStrength:
    """Tests for Equation 8-15."""

    def test_basic(self):
        # a=0.5, sigma_ff_eff=200, pa=101.325, b=0.8
        # s = 0.5 * 101.325 * (200/101.325)^0.8
        # (200/101.325) = 1.97384; 1.97384^0.8 = exp(0.8*ln(1.97384)) = exp(0.8*0.68009) = exp(0.54407) = 1.72333
        # s = 0.5 * 101.325 * 1.72333 = 87.307
        expected = 0.5 * 101.325 * (200.0 / 101.325) ** 0.8
        assert power_function_shear_strength(0.5, 200.0, 101.325, 0.8) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_stress(self):
        # sigma_ff_eff=0 => s=0 (since 0^b = 0 for b>0)
        assert power_function_shear_strength(0.5, 0.0, 101.325, 0.8) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_negative_stress_raises(self):
        with pytest.raises(ValueError, match="sigma_ff_eff must be non-negative"):
            power_function_shear_strength(0.5, -1.0, 101.325, 0.8)

    def test_pa_not_positive_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            power_function_shear_strength(0.5, 200.0, 0.0, 0.8)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            power_function_shear_strength(0.5, 200.0, -1.0, 0.8)


# ============================================================================
# Eq 8-16: residual_friction_angle_gibson
# ============================================================================


class TestResidualFrictionAngleGibson:
    """Tests for Equation 8-16."""

    def test_basic(self):
        # PI=40: phi'r = 0.084*40^1.4 - 0.75*40 + 31.9
        # 40^1.4 = exp(1.4*ln(40)) = exp(1.4*3.68888) = exp(5.16443) = 174.94
        # = 0.084*174.94 - 30.0 + 31.9 = 14.695 - 30.0 + 31.9 = 16.595
        expected = 0.084 * 40.0 ** 1.4 - 0.75 * 40.0 + 31.9
        assert residual_friction_angle_gibson(40.0) == pytest.approx(expected, rel=1e-4)

    def test_zero_pi(self):
        # PI=0: phi'r = 0 - 0 + 31.9 = 31.9
        assert residual_friction_angle_gibson(0.0) == pytest.approx(31.9, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be non-negative"):
            residual_friction_angle_gibson(-1.0)


# ============================================================================
# Eq 8-17: residual_friction_angle_stark_hussain
# ============================================================================


class TestResidualFrictionAngleStarkHussain:
    """Tests for Equation 8-17."""

    def test_basic(self):
        # LL=60, C0=46.6, C1=-0.3357, C2=-0.00171, C3=0.0000138
        # phi'r = 46.6 + (-0.3357)*60 + (-0.00171)*60^2 + 0.0000138*60^3
        # = 46.6 - 20.142 - 6.156 + 2.9808 = 23.2828
        expected = 46.6 + (-0.3357) * 60.0 + (-0.00171) * 3600.0 + 0.0000138 * 216000.0
        assert residual_friction_angle_stark_hussain(
            60.0, 46.6, -0.3357, -0.00171, 0.0000138
        ) == pytest.approx(expected, rel=1e-4)

    def test_zero_ll(self):
        # LL=0: phi'r = C0
        assert residual_friction_angle_stark_hussain(0.0, 50.0, -0.5, 0.001, -0.00001) == pytest.approx(
            50.0, rel=1e-4
        )

    def test_negative_ll_raises(self):
        with pytest.raises(ValueError, match="ll must be non-negative"):
            residual_friction_angle_stark_hussain(-1.0, 46.6, -0.3357, -0.00171, 0.0000138)


# ============================================================================
# Eq 8-18: undrained_strength_ratio_skempton
# ============================================================================


class TestUndrainedStrengthRatioSkempton:
    """Tests for Equation 8-18."""

    def test_basic(self):
        # PI=30: su/sv = 0.11 + 0.0037*30 = 0.11 + 0.111 = 0.221
        assert undrained_strength_ratio_skempton(30.0) == pytest.approx(0.221, rel=1e-4)

    def test_zero_pi(self):
        # PI=0: su/sv = 0.11
        assert undrained_strength_ratio_skempton(0.0) == pytest.approx(0.11, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be non-negative"):
            undrained_strength_ratio_skempton(-1.0)


# ============================================================================
# Eq 8-19: undrained_strength_ratio_hansbo
# ============================================================================


class TestUndrainedStrengthRatioHansbo:
    """Tests for Equation 8-19."""

    def test_basic(self):
        # LL=50: (su/sv)_NC = 0.0045*50 = 0.225
        assert undrained_strength_ratio_hansbo(50.0) == pytest.approx(0.225, rel=1e-4)

    def test_zero_ll(self):
        # LL=0: 0
        assert undrained_strength_ratio_hansbo(0.0) == pytest.approx(0.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="ll must be non-negative"):
            undrained_strength_ratio_hansbo(-1.0)


# ============================================================================
# Eq 8-20: undrained_strength_ratio_oc
# ============================================================================


class TestUndrainedStrengthRatioOc:
    """Tests for Equation 8-20."""

    def test_basic(self):
        # su_sigma_v_nc=0.25, OCR=4, m=0.8
        # (su/sv)_OC = 0.25 * 4^0.8 = 0.25 * 3.03143 = 0.75786
        expected = 0.25 * 4.0 ** 0.8
        assert undrained_strength_ratio_oc(0.25, 4.0, 0.8) == pytest.approx(
            expected, rel=1e-4
        )

    def test_nc_soil(self):
        # OCR=1: (su/sv)_OC = su_sigma_v_nc * 1^m = su_sigma_v_nc
        assert undrained_strength_ratio_oc(0.25, 1.0, 0.8) == pytest.approx(0.25, rel=1e-4)

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError, match="su_sigma_v_nc must be non-negative"):
            undrained_strength_ratio_oc(-0.1, 4.0, 0.8)

    def test_ocr_less_than_one_raises(self):
        with pytest.raises(ValueError, match="ocr must be >= 1.0"):
            undrained_strength_ratio_oc(0.25, 0.5, 0.8)


# ============================================================================
# Eq 8-21: undrained_strength_from_preconsolidation
# ============================================================================


class TestUndrainedStrengthFromPreconsolidation:
    """Tests for Equation 8-21."""

    def test_basic(self):
        # sigma_p_eff=200: su = 0.21*200 = 42
        assert undrained_strength_from_preconsolidation(200.0) == pytest.approx(42.0, rel=1e-4)

    def test_zero(self):
        assert undrained_strength_from_preconsolidation(0.0) == pytest.approx(0.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_p_eff must be non-negative"):
            undrained_strength_from_preconsolidation(-1.0)


# ============================================================================
# Eq 8-22: undrained_strength_ratio_acu_from_icu
# ============================================================================


class TestUndrainedStrengthRatioAcuFromIcu:
    """Tests for Equation 8-22."""

    def test_basic(self):
        # su_sigma_v_icu=0.5: (su/sv)_ACU = 0.15 + 0.49*0.5 = 0.15 + 0.245 = 0.395
        assert undrained_strength_ratio_acu_from_icu(0.5) == pytest.approx(0.395, rel=1e-4)

    def test_zero(self):
        # su_sigma_v_icu=0: 0.15 + 0 = 0.15
        assert undrained_strength_ratio_acu_from_icu(0.0) == pytest.approx(0.15, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="su_sigma_v_icu must be non-negative"):
            undrained_strength_ratio_acu_from_icu(-0.1)


# ============================================================================
# Eq 8-23: cpt_undrained_strength_nc
# ============================================================================


class TestCptUndrainedStrengthNc:
    """Tests for Equation 8-23."""

    def test_basic(self):
        # qc=2000, Nc=20: su = 2000/20 = 100
        assert cpt_undrained_strength_nc(2000.0, 20.0) == pytest.approx(100.0, rel=1e-4)

    def test_zero_qc(self):
        assert cpt_undrained_strength_nc(0.0, 20.0) == pytest.approx(0.0, rel=1e-4)

    def test_qc_negative_raises(self):
        with pytest.raises(ValueError, match="qc must be non-negative"):
            cpt_undrained_strength_nc(-1.0, 20.0)

    def test_nc_not_positive_raises(self):
        with pytest.raises(ValueError, match="nc must be positive"):
            cpt_undrained_strength_nc(2000.0, 0.0)

    def test_nc_negative_raises(self):
        with pytest.raises(ValueError, match="nc must be positive"):
            cpt_undrained_strength_nc(2000.0, -5.0)


# ============================================================================
# Eq 8-24: cpt_undrained_strength_nk
# ============================================================================


class TestCptUndrainedStrengthNk:
    """Tests for Equation 8-24."""

    def test_basic(self):
        # qc=2000, sigma_v=100, Nk=15: su = (2000-100)/15 = 1900/15 = 126.667
        expected = 1900.0 / 15.0
        assert cpt_undrained_strength_nk(2000.0, 100.0, 15.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_nk_not_positive_raises(self):
        with pytest.raises(ValueError, match="nk must be positive"):
            cpt_undrained_strength_nk(2000.0, 100.0, 0.0)

    def test_nk_negative_raises(self):
        with pytest.raises(ValueError, match="nk must be positive"):
            cpt_undrained_strength_nk(2000.0, 100.0, -1.0)


# ============================================================================
# Eq 8-25: cpt_undrained_strength_nkt
# ============================================================================


class TestCptUndrainedStrengthNkt:
    """Tests for Equation 8-25."""

    def test_basic(self):
        # qt=2500, sigma_v=120, Nkt=15: su = (2500-120)/15 = 2380/15 = 158.667
        expected = 2380.0 / 15.0
        assert cpt_undrained_strength_nkt(2500.0, 120.0, 15.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_nkt_not_positive_raises(self):
        with pytest.raises(ValueError, match="nkt must be positive"):
            cpt_undrained_strength_nkt(2500.0, 120.0, 0.0)

    def test_nkt_negative_raises(self):
        with pytest.raises(ValueError, match="nkt must be positive"):
            cpt_undrained_strength_nkt(2500.0, 120.0, -1.0)


# ============================================================================
# Eq 8-26: spt_undrained_strength_stroud_butler
# ============================================================================


class TestSptUndrainedStrengthStroudButler:
    """Tests for Equation 8-26."""

    def test_basic(self):
        # N=20, PI=30: su = 20 / (4.36 + 8910/30^3) = 20 / (4.36 + 8910/27000)
        # = 20 / (4.36 + 0.33) = 20 / 4.69 = 4.2644
        expected = 20.0 / (4.36 + 8910.0 / 27000.0)
        assert spt_undrained_strength_stroud_butler(20.0, 30.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_n(self):
        assert spt_undrained_strength_stroud_butler(0.0, 30.0) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_negative_n_raises(self):
        with pytest.raises(ValueError, match="n must be non-negative"):
            spt_undrained_strength_stroud_butler(-1.0, 30.0)

    def test_pi_not_positive_raises(self):
        with pytest.raises(ValueError, match="pi must be positive"):
            spt_undrained_strength_stroud_butler(20.0, 0.0)

    def test_pi_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be positive"):
            spt_undrained_strength_stroud_butler(20.0, -5.0)


# ============================================================================
# Eq 8-27: intrinsic_compression_index
# ============================================================================


class TestIntrinsicCompressionIndex:
    """Tests for Equation 8-27."""

    def test_basic(self):
        # e*100=1.2, e*1000=0.8: C*c = 1.2 - 0.8 = 0.4
        assert intrinsic_compression_index(1.2, 0.8) == pytest.approx(0.4, rel=1e-4)

    def test_equal_raises(self):
        with pytest.raises(ValueError, match="e_star_100 must be greater than e_star_1000"):
            intrinsic_compression_index(1.0, 1.0)

    def test_inverted_raises(self):
        with pytest.raises(ValueError, match="e_star_100 must be greater than e_star_1000"):
            intrinsic_compression_index(0.5, 1.0)


# ============================================================================
# Eq 8-28: void_index
# ============================================================================


class TestVoidIndex:
    """Tests for Equation 8-28."""

    def test_basic(self):
        # e=1.5, e*100=1.2, C*c=0.4: Iv = (1.5-1.2)/0.4 = 0.3/0.4 = 0.75
        assert void_index(1.5, 1.2, 0.4) == pytest.approx(0.75, rel=1e-4)

    def test_negative_result(self):
        # e=0.8, e*100=1.2, C*c=0.4: Iv = (0.8-1.2)/0.4 = -1.0
        assert void_index(0.8, 1.2, 0.4) == pytest.approx(-1.0, rel=1e-4)

    def test_c_star_c_not_positive_raises(self):
        with pytest.raises(ValueError, match="c_star_c must be positive"):
            void_index(1.5, 1.2, 0.0)

    def test_c_star_c_negative_raises(self):
        with pytest.raises(ValueError, match="c_star_c must be positive"):
            void_index(1.5, 1.2, -0.1)


# ============================================================================
# Eq 8-29: intrinsic_void_ratio_at_100kpa
# ============================================================================


class TestIntrinsicVoidRatioAt100kpa:
    """Tests for Equation 8-29."""

    def test_basic(self):
        # eL=1.5: e*100 = 0.109 + 0.679*1.5 - 0.089*1.5^2 + 0.016*1.5^3
        # = 0.109 + 1.0185 - 0.089*2.25 + 0.016*3.375
        # = 0.109 + 1.0185 - 0.20025 + 0.054 = 0.98125
        expected = 0.109 + 0.679 * 1.5 - 0.089 * 1.5 ** 2 + 0.016 * 1.5 ** 3
        assert intrinsic_void_ratio_at_100kpa(1.5) == pytest.approx(expected, rel=1e-4)

    def test_zero(self):
        # eL=0: e*100 = 0.109
        assert intrinsic_void_ratio_at_100kpa(0.0) == pytest.approx(0.109, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="e_l must be non-negative"):
            intrinsic_void_ratio_at_100kpa(-0.1)


# ============================================================================
# Eq 8-30: intrinsic_compression_index_from_el
# ============================================================================


class TestIntrinsicCompressionIndexFromEl:
    """Tests for Equation 8-30."""

    def test_basic(self):
        # eL=1.5: C*c = 0.256*1.5 - 0.04 = 0.384 - 0.04 = 0.344
        assert intrinsic_compression_index_from_el(1.5) == pytest.approx(0.344, rel=1e-4)

    def test_zero(self):
        # eL=0: C*c = -0.04
        assert intrinsic_compression_index_from_el(0.0) == pytest.approx(-0.04, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="e_l must be non-negative"):
            intrinsic_compression_index_from_el(-0.1)


# ============================================================================
# Eq 8-31: constrained_modulus_linear
# ============================================================================


class TestConstrainedModulusLinear:
    """Tests for Equation 8-31."""

    def test_basic(self):
        # m=10, sigma_v_eff=50: Mds = 10*50 = 500
        assert constrained_modulus_linear(10.0, 50.0) == pytest.approx(500.0, rel=1e-4)

    def test_zero_stress(self):
        assert constrained_modulus_linear(10.0, 0.0) == pytest.approx(0.0, rel=1e-4)

    def test_zero_modulus_number(self):
        assert constrained_modulus_linear(0.0, 50.0) == pytest.approx(0.0, rel=1e-4)

    def test_m_negative_raises(self):
        with pytest.raises(ValueError, match="m must be non-negative"):
            constrained_modulus_linear(-1.0, 50.0)

    def test_sigma_v_eff_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be non-negative"):
            constrained_modulus_linear(10.0, -1.0)


# ============================================================================
# Eq 8-32: constrained_modulus_nonlinear
# ============================================================================


class TestConstrainedModulusNonlinear:
    """Tests for Equation 8-32."""

    def test_basic(self):
        # m=10, sigma_v_eff=100, pa=101.325
        # Mds = 10 * 101.325 * (100/101.325)^0.5
        # = 1013.25 * (0.98692)^0.5 = 1013.25 * 0.99345 = 1006.61
        expected = 10.0 * 101.325 * (100.0 / 101.325) ** 0.5
        assert constrained_modulus_nonlinear(10.0, 100.0, 101.325) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_stress(self):
        # sigma_v_eff=0: (0/pa)^0.5 = 0 => Mds = 0
        assert constrained_modulus_nonlinear(10.0, 0.0, 101.325) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_m_negative_raises(self):
        with pytest.raises(ValueError, match="m must be non-negative"):
            constrained_modulus_nonlinear(-1.0, 100.0, 101.325)

    def test_sigma_v_eff_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be non-negative"):
            constrained_modulus_nonlinear(10.0, -1.0, 101.325)

    def test_pa_not_positive_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            constrained_modulus_nonlinear(10.0, 100.0, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            constrained_modulus_nonlinear(10.0, 100.0, -1.0)


# ============================================================================
# Eq 8-33: constrained_modulus_spt
# ============================================================================


class TestConstrainedModulusSpt:
    """Tests for Equation 8-33."""

    def test_basic(self):
        # f=5, N=20, pa=101.325: Mds = 5*20*101.325 = 10132.5
        assert constrained_modulus_spt(5.0, 20.0, 101.325) == pytest.approx(
            10132.5, rel=1e-4
        )

    def test_zero_n(self):
        assert constrained_modulus_spt(5.0, 0.0, 101.325) == pytest.approx(0.0, rel=1e-4)

    def test_f_negative_raises(self):
        with pytest.raises(ValueError, match="f must be non-negative"):
            constrained_modulus_spt(-1.0, 20.0, 101.325)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="n must be non-negative"):
            constrained_modulus_spt(5.0, -1.0, 101.325)

    def test_pa_not_positive_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            constrained_modulus_spt(5.0, 20.0, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="pa must be positive"):
            constrained_modulus_spt(5.0, 20.0, -1.0)


# ============================================================================
# Eq 8-34: constrained_modulus_cpt
# ============================================================================


class TestConstrainedModulusCpt:
    """Tests for Equation 8-34."""

    def test_basic(self):
        # alpha=2.5, qc=4000: Mds = 2.5*4000 = 10000
        assert constrained_modulus_cpt(2.5, 4000.0) == pytest.approx(10000.0, rel=1e-4)

    def test_zero_qc(self):
        assert constrained_modulus_cpt(2.5, 0.0) == pytest.approx(0.0, rel=1e-4)

    def test_zero_alpha(self):
        assert constrained_modulus_cpt(0.0, 4000.0) == pytest.approx(0.0, rel=1e-4)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            constrained_modulus_cpt(-1.0, 4000.0)

    def test_qc_negative_raises(self):
        with pytest.raises(ValueError, match="qc must be non-negative"):
            constrained_modulus_cpt(2.5, -1.0)


# ============================================================================
# Eq 8-35: secondary_compression_ratio
# ============================================================================


class TestSecondaryCompressionRatio:
    """Tests for Equation 8-35."""

    def test_basic(self):
        # wn=40: C_ea = 0.0001*40 = 0.004
        assert secondary_compression_ratio(40.0) == pytest.approx(0.004, rel=1e-4)

    def test_zero(self):
        assert secondary_compression_ratio(0.0) == pytest.approx(0.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="wn must be non-negative"):
            secondary_compression_ratio(-1.0)


# ============================================================================
# Eq 8-36: drained_youngs_modulus_from_undrained
# ============================================================================


class TestDrainedYoungsModulusFromUndrained:
    """Tests for Equation 8-36."""

    def test_basic(self):
        # Eu=3000, nu=0.3: E = 2*(1+0.3)/3 * 3000 = 2*1.3/3 * 3000 = 0.86667*3000 = 2600
        expected = 2.0 * (1.0 + 0.3) / 3.0 * 3000.0
        assert drained_youngs_modulus_from_undrained(3000.0, 0.3) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_eu(self):
        assert drained_youngs_modulus_from_undrained(0.0, 0.3) == pytest.approx(
            0.0, rel=1e-4
        )

    def test_nu_zero(self):
        # nu=0: E = 2*(1+0)/3 * 1000 = 2/3*1000 = 666.667
        expected = 2.0 / 3.0 * 1000.0
        assert drained_youngs_modulus_from_undrained(1000.0, 0.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_eu_negative_raises(self):
        with pytest.raises(ValueError, match="eu must be non-negative"):
            drained_youngs_modulus_from_undrained(-1.0, 0.3)

    def test_nu_at_minus_one_raises(self):
        with pytest.raises(ValueError, match="nu must be in the range"):
            drained_youngs_modulus_from_undrained(1000.0, -1.0)

    def test_nu_at_half_raises(self):
        with pytest.raises(ValueError, match="nu must be in the range"):
            drained_youngs_modulus_from_undrained(1000.0, 0.5)

    def test_nu_above_half_raises(self):
        with pytest.raises(ValueError, match="nu must be in the range"):
            drained_youngs_modulus_from_undrained(1000.0, 0.6)

    def test_nu_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="nu must be in the range"):
            drained_youngs_modulus_from_undrained(1000.0, -1.5)


# ============================================================================
# Eq 8-37: cbr_from_dcp_power
# ============================================================================


class TestCbrFromDcpPower:
    """Tests for Equation 8-37."""

    def test_basic(self):
        # A=292, DCP=10, x=-1.12: CBR = 292 * 10^(-1.12)
        # 10^(-1.12) = 1/10^1.12 = 1/13.1826 = 0.07586
        # CBR = 292 * 0.07586 = 22.15
        expected = 292.0 * 10.0 ** (-1.12)
        assert cbr_from_dcp_power(292.0, 10.0, -1.12) == pytest.approx(expected, rel=1e-4)

    def test_positive_exponent(self):
        # A=10, DCP=2, x=1.5: CBR = 10 * 2^1.5 = 10 * 2.8284 = 28.284
        expected = 10.0 * 2.0 ** 1.5
        assert cbr_from_dcp_power(10.0, 2.0, 1.5) == pytest.approx(expected, rel=1e-4)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a_coeff must be non-negative"):
            cbr_from_dcp_power(-1.0, 10.0, -1.12)

    def test_dcp_not_positive_raises(self):
        with pytest.raises(ValueError, match="dcp must be positive"):
            cbr_from_dcp_power(292.0, 0.0, -1.12)

    def test_dcp_negative_raises(self):
        with pytest.raises(ValueError, match="dcp must be positive"):
            cbr_from_dcp_power(292.0, -1.0, -1.12)


# ============================================================================
# Eq 8-38: cbr_from_dcp_nazzal
# ============================================================================


class TestCbrFromDcpNazzal:
    """Tests for Equation 8-38."""

    def test_basic(self):
        # DCP=20: CBR = 2559 / (20^1.84 + 7.35) - 1
        # 20^1.84 = exp(1.84*ln(20)) = exp(1.84*2.99573) = exp(5.51214) = 247.66
        # CBR = 2559 / (247.66 + 7.35) - 1 = 2559 / 255.01 - 1 = 10.035 - 1 = 9.035
        expected = 2559.0 / (20.0 ** 1.84 + 7.35) - 1.0
        assert cbr_from_dcp_nazzal(20.0) == pytest.approx(expected, rel=1e-4)

    def test_small_dcp(self):
        # DCP=6.3 (lower bound of valid range)
        expected = 2559.0 / (6.3 ** 1.84 + 7.35) - 1.0
        assert cbr_from_dcp_nazzal(6.3) == pytest.approx(expected, rel=1e-4)

    def test_dcp_not_positive_raises(self):
        with pytest.raises(ValueError, match="dcp must be positive"):
            cbr_from_dcp_nazzal(0.0)

    def test_dcp_negative_raises(self):
        with pytest.raises(ValueError, match="dcp must be positive"):
            cbr_from_dcp_nazzal(-5.0)


# ============================================================================
# Eq 8-39: cbr_from_spt
# ============================================================================


class TestCbrFromSpt:
    """Tests for Equation 8-39."""

    def test_basic(self):
        # N=15: log(CBR) = -5.13 + 6.55*log10(300/15^0.26)
        # 15^0.26 = exp(0.26*ln(15)) = exp(0.26*2.70805) = exp(0.70409) = 2.0221
        # 300/2.0221 = 148.36
        # log10(148.36) = 2.17130
        # log(CBR) = -5.13 + 6.55*2.17130 = -5.13 + 14.222 = 9.092
        # CBR = 10^9.092
        log_cbr = -5.13 + 6.55 * math.log10(300.0 / 15.0 ** 0.26)
        expected = 10.0 ** log_cbr
        assert cbr_from_spt(15.0) == pytest.approx(expected, rel=1e-4)

    def test_n_not_positive_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            cbr_from_spt(0.0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            cbr_from_spt(-5.0)


# ============================================================================
# Eq 8-40: hydraulic_conductivity_hazen
# ============================================================================


class TestHydraulicConductivityHazen:
    """Tests for Equation 8-40."""

    def test_basic(self):
        # D10=0.5, C=1.0: k = 1.0 * 0.5^2 = 0.25 cm/s
        assert hydraulic_conductivity_hazen(0.5) == pytest.approx(0.25, rel=1e-4)

    def test_with_custom_c(self):
        # D10=0.3, C=2.0: k = 2.0 * 0.3^2 = 2.0 * 0.09 = 0.18
        assert hydraulic_conductivity_hazen(0.3, c=2.0) == pytest.approx(0.18, rel=1e-4)

    def test_d10_not_positive_raises(self):
        with pytest.raises(ValueError, match="d10 must be positive"):
            hydraulic_conductivity_hazen(0.0)

    def test_d10_negative_raises(self):
        with pytest.raises(ValueError, match="d10 must be positive"):
            hydraulic_conductivity_hazen(-0.1)

    def test_c_not_positive_raises(self):
        with pytest.raises(ValueError, match="c must be positive"):
            hydraulic_conductivity_hazen(0.5, c=0.0)

    def test_c_negative_raises(self):
        with pytest.raises(ValueError, match="c must be positive"):
            hydraulic_conductivity_hazen(0.5, c=-1.0)


# ============================================================================
# Eq 8-41: hydraulic_conductivity_kozeny_carman
# ============================================================================


class TestHydraulicConductivityKozenyCarman:
    """Tests for Equation 8-41."""

    def test_basic(self):
        # Single fraction: fi=1.0, d_li=0.5, d_si=0.25, e=0.65, s=6.0
        # avg_d = 0.404*0.5 + 0.596*0.25 = 0.202 + 0.149 = 0.351
        # summation = 1.0/0.351 = 2.8490
        # k = 1.99e4 * (1/36) * (0.65^3/(1.65)) * (1/2.8490^2)
        # = 1.99e4 * 0.027778 * (0.274625/1.65) * (1/8.11683)
        # = 552.778 * 0.16644 * 0.12320
        # = 552.778 * 0.020505 = 11.334
        fractions = [(1.0, 0.5, 0.25)]
        e = 0.65
        s = 6.0
        avg_d = 0.404 * 0.5 + 0.596 * 0.25
        summation = 1.0 / avg_d
        expected = 1.99e4 * (1.0 / s ** 2) * (e ** 3 / (1.0 + e)) * (1.0 / summation ** 2)
        assert hydraulic_conductivity_kozeny_carman(fractions, e, s) == pytest.approx(
            expected, rel=1e-4
        )

    def test_two_fractions(self):
        # Two fractions
        fractions = [(0.6, 1.0, 0.5), (0.4, 0.5, 0.25)]
        e = 0.7
        s = 6.0
        avg_d1 = 0.404 * 1.0 + 0.596 * 0.5  # 0.702
        avg_d2 = 0.404 * 0.5 + 0.596 * 0.25  # 0.351
        summation = 0.6 / avg_d1 + 0.4 / avg_d2
        expected = 1.99e4 * (1.0 / 36.0) * (0.7 ** 3 / 1.7) * (1.0 / summation ** 2)
        assert hydraulic_conductivity_kozeny_carman(fractions, e, s) == pytest.approx(
            expected, rel=1e-4
        )

    def test_e_negative_raises(self):
        with pytest.raises(ValueError, match="e must be non-negative"):
            hydraulic_conductivity_kozeny_carman([(1.0, 0.5, 0.25)], -0.1, 6.0)

    def test_s_not_positive_raises(self):
        with pytest.raises(ValueError, match="s must be positive"):
            hydraulic_conductivity_kozeny_carman([(1.0, 0.5, 0.25)], 0.65, 0.0)

    def test_sieve_not_positive_raises(self):
        with pytest.raises(ValueError, match="All sieve sizes.*must be positive"):
            hydraulic_conductivity_kozeny_carman([(1.0, 0.5, 0.0)], 0.65, 6.0)

    def test_sieve_negative_raises(self):
        with pytest.raises(ValueError, match="All sieve sizes.*must be positive"):
            hydraulic_conductivity_kozeny_carman([(1.0, -0.1, 0.25)], 0.65, 6.0)

    def test_empty_fractions_zero_sum_raises(self):
        # All fractions with fi=0 result in summation=0
        with pytest.raises(ValueError, match="Sum of fi/D_avg is zero"):
            hydraulic_conductivity_kozeny_carman([(0.0, 0.5, 0.25)], 0.65, 6.0)


# ============================================================================
# Eq 8-42: hydraulic_conductivity_carrier_beckman
# ============================================================================


class TestHydraulicConductivityCarrierBeckman:
    """Tests for Equation 8-42."""

    def test_basic(self):
        # e=0.8, PL=25, PI=20
        # numerator_base = 0.8 - 0.027*(25 - 0.242*20) = 0.8 - 0.027*(25 - 4.84) = 0.8 - 0.027*20.16
        # = 0.8 - 0.54432 = 0.25568
        # numerator = 0.25568^4.29
        # denominator = (1+0.8)*20 = 1.8*20 = 36
        # k = 0.0174 * 0.25568^4.29 / 36
        num_base = 0.8 - 0.027 * (25.0 - 0.242 * 20.0)
        numerator = num_base ** 4.29
        denominator = (1.0 + 0.8) * 20.0
        expected = 0.0174 * numerator / denominator
        assert hydraulic_conductivity_carrier_beckman(0.8, 25.0, 20.0) == pytest.approx(
            expected, rel=1e-4
        )

    def test_e_negative_raises(self):
        with pytest.raises(ValueError, match="e must be non-negative"):
            hydraulic_conductivity_carrier_beckman(-0.1, 25.0, 20.0)

    def test_pi_not_positive_raises(self):
        with pytest.raises(ValueError, match="pi must be positive"):
            hydraulic_conductivity_carrier_beckman(0.8, 25.0, 0.0)

    def test_pi_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be positive"):
            hydraulic_conductivity_carrier_beckman(0.8, 25.0, -5.0)


# ============================================================================
# Eq 8-43: hydraulic_conductivity_benson_landfill
# ============================================================================


class TestHydraulicConductivityBensonLandfill:
    """Tests for Equation 8-43."""

    def test_basic(self):
        # Use inputs that keep the degree of saturation low enough to
        # avoid overflow in math.exp().  Pick a low water content and
        # a relatively high void ratio.
        # cf=50, gc=0, w_compactor=100, pi=30, w=0.02,
        # gamma_w=9.81, gamma_d=14.0, gs=2.70
        # e = 2.70*9.81/14.0 - 1 = 26.487/14 - 1 = 1.89193 - 1 = 0.89193
        # S = 0.02*2.70/0.89193 = 0.054/0.89193 = 0.060542
        # ln(k) = -18.35 + 0.08*30 + 2.87*0 - 0.32*50 - 0.02*100 + 894*0.060542
        #       = -18.35 + 2.4 + 0 - 16.0 - 2.0 + 54.124
        #       = 20.174
        # k = exp(20.174)
        e = 2.70 * 9.81 / 14.0 - 1.0
        s_degree = 0.02 * 2.70 / e
        ln_k = (-18.35 + 0.08 * 30.0 + 2.87 * 0.0 - 0.32 * 50.0
                - 0.02 * 100.0 + 894.0 * s_degree)
        expected = math.exp(ln_k)
        result = hydraulic_conductivity_benson_landfill(
            cf=50.0, gc=0.0, w_compactor=100.0, pi=30.0, w=0.02,
            gamma_w=9.81, gamma_d=14.0, gs=2.70
        )
        assert result == pytest.approx(expected, rel=1e-4)

    def test_gamma_d_not_positive_raises(self):
        with pytest.raises(ValueError, match="gamma_d must be positive"):
            hydraulic_conductivity_benson_landfill(
                cf=30.0, gc=0.05, w_compactor=300.0, pi=20.0, w=0.20,
                gamma_w=9.81, gamma_d=0.0, gs=2.70
            )

    def test_gamma_w_not_positive_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            hydraulic_conductivity_benson_landfill(
                cf=30.0, gc=0.05, w_compactor=300.0, pi=20.0, w=0.20,
                gamma_w=0.0, gamma_d=17.0, gs=2.70
            )

    def test_gs_not_positive_raises(self):
        with pytest.raises(ValueError, match="gs must be positive"):
            hydraulic_conductivity_benson_landfill(
                cf=30.0, gc=0.05, w_compactor=300.0, pi=20.0, w=0.20,
                gamma_w=9.81, gamma_d=17.0, gs=0.0
            )

    def test_void_ratio_nonpositive_raises(self):
        # e = gs*gamma_w/gamma_d - 1 <= 0 when gamma_d >= gs*gamma_w
        # gs=2.7, gamma_w=9.81 => gs*gamma_w=26.487; gamma_d=27 => e = 26.487/27 -1 = -0.019
        with pytest.raises(ValueError, match="Computed void ratio is non-positive"):
            hydraulic_conductivity_benson_landfill(
                cf=30.0, gc=0.05, w_compactor=300.0, pi=20.0, w=0.20,
                gamma_w=9.81, gamma_d=27.0, gs=2.70
            )


# ============================================================================
# Eq 8-44: hydraulic_conductivity_benson_trast
# ============================================================================


class TestHydraulicConductivityBensonTrast:
    """Tests for Equation 8-44."""

    def test_basic(self):
        # pi=25, cf=40, e_effort=0, w=0.22, gamma_w=9.81, gamma_d=16.5, gs=2.70
        # e = 2.70*9.81/16.5 - 1 = 26.487/16.5 - 1 = 1.60527 - 1 = 0.60527
        # S = 0.22*2.70/0.60527 = 0.594/0.60527 = 0.98128
        # ln(k) = -15 - 0.087*25 - 0.054*40 + 0.022*0 + 0.91*0.98128
        # = -15 - 2.175 - 2.16 + 0 + 0.89296 = -18.44204
        # k = exp(-18.44204)
        e = 2.70 * 9.81 / 16.5 - 1.0
        s_degree = 0.22 * 2.70 / e
        ln_k = -15.0 - 0.087 * 25.0 - 0.054 * 40.0 + 0.022 * 0.0 + 0.91 * s_degree
        expected = math.exp(ln_k)
        result = hydraulic_conductivity_benson_trast(
            pi=25.0, cf=40.0, e_effort=0.0, w=0.22,
            gamma_w=9.81, gamma_d=16.5, gs=2.70
        )
        assert result == pytest.approx(expected, rel=1e-4)

    def test_modified_proctor_effort(self):
        # e_effort=-1 (modified Proctor)
        e = 2.70 * 9.81 / 16.5 - 1.0
        s_degree = 0.22 * 2.70 / e
        ln_k = -15.0 - 0.087 * 25.0 - 0.054 * 40.0 + 0.022 * (-1.0) + 0.91 * s_degree
        expected = math.exp(ln_k)
        result = hydraulic_conductivity_benson_trast(
            pi=25.0, cf=40.0, e_effort=-1.0, w=0.22,
            gamma_w=9.81, gamma_d=16.5, gs=2.70
        )
        assert result == pytest.approx(expected, rel=1e-4)

    def test_gamma_d_not_positive_raises(self):
        with pytest.raises(ValueError, match="gamma_d must be positive"):
            hydraulic_conductivity_benson_trast(
                pi=25.0, cf=40.0, e_effort=0.0, w=0.22,
                gamma_w=9.81, gamma_d=0.0, gs=2.70
            )

    def test_gamma_w_not_positive_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            hydraulic_conductivity_benson_trast(
                pi=25.0, cf=40.0, e_effort=0.0, w=0.22,
                gamma_w=0.0, gamma_d=16.5, gs=2.70
            )

    def test_gs_not_positive_raises(self):
        with pytest.raises(ValueError, match="gs must be positive"):
            hydraulic_conductivity_benson_trast(
                pi=25.0, cf=40.0, e_effort=0.0, w=0.22,
                gamma_w=9.81, gamma_d=16.5, gs=0.0
            )

    def test_void_ratio_nonpositive_raises(self):
        with pytest.raises(ValueError, match="Computed void ratio is non-positive"):
            hydraulic_conductivity_benson_trast(
                pi=25.0, cf=40.0, e_effort=0.0, w=0.22,
                gamma_w=9.81, gamma_d=27.0, gs=2.70
            )


# ============================================================================
# Eq 8-45: shear_wave_velocity_spt
# ============================================================================


class TestShearWaveVelocitySpt:
    """Tests for Equation 8-45."""

    def test_basic(self):
        # B=80, N=20, x=0.33: Vs = 80 * 20^0.33 * 1^0 = 80 * 2.6390 = 211.12
        expected = 80.0 * 20.0 ** 0.33
        assert shear_wave_velocity_spt(80.0, 20.0, 0.33) == pytest.approx(
            expected, rel=1e-4
        )

    def test_with_depth(self):
        # B=80, N=20, x=0.33, z=10, y=0.5: Vs = 80 * 20^0.33 * 10^0.5
        # = 80 * 2.6390 * 3.1623 = 667.55
        expected = 80.0 * 20.0 ** 0.33 * 10.0 ** 0.5
        assert shear_wave_velocity_spt(80.0, 20.0, 0.33, z=10.0, y=0.5) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_n(self):
        # N=0: 0^x = 0 (for positive x) => Vs = 0
        assert shear_wave_velocity_spt(80.0, 0.0, 0.33) == pytest.approx(0.0, rel=1e-4)

    def test_default_depth(self):
        # z=1.0 (default), y=0.0 (default) => z^y = 1^0 = 1
        expected = 80.0 * 20.0 ** 0.33
        assert shear_wave_velocity_spt(80.0, 20.0, 0.33) == pytest.approx(
            expected, rel=1e-4
        )

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be non-negative"):
            shear_wave_velocity_spt(-1.0, 20.0, 0.33)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="n must be non-negative"):
            shear_wave_velocity_spt(80.0, -1.0, 0.33)

    def test_z_not_positive_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            shear_wave_velocity_spt(80.0, 20.0, 0.33, z=0.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            shear_wave_velocity_spt(80.0, 20.0, 0.33, z=-1.0)


# ============================================================================
# TABLE 8-6: table_8_6_stark_hussain
# ============================================================================


class TestTable86StarkHussain:
    """Tests for Table 8-6: Stark & Hussain coefficients."""

    def test_lt_20_50kpa(self):
        # lt_20, 50_kPa: c0=35.2
        result = table_8_6_stark_hussain("lt_20", "50_kPa")
        assert result["c0"] == pytest.approx(35.2, rel=1e-4)

    def test_gt_50_400kpa(self):
        # gt_50, 400_kPa: c0=44.1
        result = table_8_6_stark_hussain("gt_50", "400_kPa")
        assert result["c0"] == pytest.approx(44.1, rel=1e-4)

    def test_20_to_50_100kpa(self):
        # 20_to_50, 100_kPa: returns dict with 4 keys
        result = table_8_6_stark_hussain("20_to_50", "100_kPa")
        assert len(result) == 4
        assert "c0" in result
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result

    def test_returns_copy(self):
        # Modifying returned dict doesn't affect source
        result1 = table_8_6_stark_hussain("lt_20", "50_kPa")
        original_c0 = result1["c0"]
        result1["c0"] = 999.0
        result2 = table_8_6_stark_hussain("lt_20", "50_kPa")
        assert result2["c0"] == pytest.approx(original_c0, rel=1e-4)

    def test_unknown_cf_raises(self):
        with pytest.raises(ValueError, match="Unknown combination"):
            table_8_6_stark_hussain("lt_10", "50_kPa")

    def test_unknown_sigma_raises(self):
        with pytest.raises(ValueError, match="Unknown combination"):
            table_8_6_stark_hussain("lt_20", "200_kPa")


# ============================================================================
# TABLE 8-9: table_8_9_shansep_m
# ============================================================================


class TestTable89ShansepM:
    """Tests for Table 8-9: SHANSEP exponent m."""

    def test_dss(self):
        # DSS: m=0.80
        assert table_8_9_shansep_m("dss") == pytest.approx(0.80, rel=1e-4)

    def test_ciuc(self):
        # CIUC: m=0.85
        assert table_8_9_shansep_m("ciuc") == pytest.approx(0.85, rel=1e-4)

    def test_ck0ue(self):
        # CK0UE: m=0.70
        assert table_8_9_shansep_m("ck0ue") == pytest.approx(0.70, rel=1e-4)

    def test_case_insensitive(self):
        # "DSS" (uppercase) should work
        assert table_8_9_shansep_m("DSS") == pytest.approx(0.80, rel=1e-4)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown test_type"):
            table_8_9_shansep_m("unknown_test")


# ============================================================================
# TABLE 8-24: table_8_24_cbr_dcp
# ============================================================================


class TestTable824CbrDcp:
    """Tests for Table 8-24: CBR-DCP correlation coefficients."""

    def test_webster(self):
        # webster_1992: a=292.0, x=-1.12
        result = table_8_24_cbr_dcp("webster_1992")
        assert result["a"] == pytest.approx(292.0, rel=1e-4)
        assert result["x"] == pytest.approx(-1.12, rel=1e-4)

    def test_chua(self):
        # chua_1988: a=3370.0, x=-1.51
        result = table_8_24_cbr_dcp("chua_1988")
        assert result["a"] == pytest.approx(3370.0, rel=1e-4)
        assert result["x"] == pytest.approx(-1.51, rel=1e-4)

    def test_returns_copy(self):
        # Modifying returned dict doesn't affect source
        result1 = table_8_24_cbr_dcp("webster_1992")
        original_a = result1["a"]
        result1["a"] = 999.0
        result2 = table_8_24_cbr_dcp("webster_1992")
        assert result2["a"] == pytest.approx(original_a, rel=1e-4)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            table_8_24_cbr_dcp("unknown_source")


# ============================================================================
# FIGURE 8-46: figure_8_46_f
# ============================================================================


class TestFigure846F:
    """Tests for Figure 8-46: Stroud f-coefficient."""

    def test_at_10(self):
        # PI=10: f=500.0
        assert figure_8_46_f(10.0) == pytest.approx(500.0, rel=1e-4)

    def test_at_80(self):
        # PI=80: f=175.0
        assert figure_8_46_f(80.0) == pytest.approx(175.0, rel=1e-4)

    def test_interpolated_35(self):
        # PI=35: midpoint between 30→300 and 40→250 = 275
        # Linear interpolation: f = 300 + (35-30)/(40-30)*(250-300) = 300 - 25 = 275
        assert figure_8_46_f(35.0) == pytest.approx(275.0, rel=1e-2)

    def test_clamped_low(self):
        # PI=5 (below range): should clamp to f=500.0
        assert figure_8_46_f(5.0) == pytest.approx(500.0, rel=1e-4)

    def test_clamped_high(self):
        # PI=100 (above range): should clamp to f=175.0
        assert figure_8_46_f(100.0) == pytest.approx(175.0, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="pi must be non-negative"):
            figure_8_46_f(-1.0)


# ===========================================================================
# INTEGRATION & PUBLISHED REFERENCE VALUES
# ===========================================================================

class TestTable86Integration:
    """Chain table_8_6 → residual_friction_angle_stark_hussain."""

    def test_lt20_50kpa_ll60(self):
        # Clay fraction < 20%, sigma_n = 50 kPa, LL=60%
        coeffs = table_8_6_stark_hussain("lt_20", "50_kPa")
        phi_r = residual_friction_angle_stark_hussain(
            60.0, coeffs["c0"], coeffs["c1"], coeffs["c2"], coeffs["c3"]
        )
        # phi_r should be a reasonable residual angle (5-35 degrees)
        assert 5.0 < phi_r < 35.0

    def test_gt50_400kpa_ll80(self):
        # High clay fraction, high stress, LL=80
        coeffs = table_8_6_stark_hussain("gt_50", "400_kPa")
        phi_r = residual_friction_angle_stark_hussain(
            80.0, coeffs["c0"], coeffs["c1"], coeffs["c2"], coeffs["c3"]
        )
        # Higher clay fraction + higher LL = lower residual angle
        assert 3.0 < phi_r < 35.0

    def test_increasing_ll_decreases_phi_r(self):
        # For same clay fraction/stress, increasing LL should decrease phi_r
        coeffs = table_8_6_stark_hussain("20_to_50", "100_kPa")
        phi_30 = residual_friction_angle_stark_hussain(
            30.0, coeffs["c0"], coeffs["c1"], coeffs["c2"], coeffs["c3"]
        )
        phi_80 = residual_friction_angle_stark_hussain(
            80.0, coeffs["c0"], coeffs["c1"], coeffs["c2"], coeffs["c3"]
        )
        assert phi_80 < phi_30


class TestTable89Integration:
    """Chain table_8_9 → undrained_strength_ratio_oc."""

    def test_dss_ocr4(self):
        # DSS test, OCR=4: su/sv_oc = 0.23 * 4^0.8
        m = table_8_9_shansep_m("DSS")
        su_ratio = undrained_strength_ratio_oc(0.23, 4.0, m)
        # 0.23 * 4^0.8 = 0.23 * 3.031 = 0.697
        assert su_ratio == pytest.approx(0.23 * 4.0 ** 0.8, rel=1e-4)

    def test_ciuc_vs_dss(self):
        # CIUC (m=0.85) should give higher ratio than DSS (m=0.80)
        # for same su/sv_nc and OCR>1
        m_ciuc = table_8_9_shansep_m("CIUC")
        m_dss = table_8_9_shansep_m("DSS")
        su_ciuc = undrained_strength_ratio_oc(0.23, 5.0, m_ciuc)
        su_dss = undrained_strength_ratio_oc(0.23, 5.0, m_dss)
        assert su_ciuc > su_dss  # higher m → higher ratio for OCR > 1

    def test_published_m_values(self):
        # Ladd & DeGroot (2004) published values
        assert table_8_9_shansep_m("DSS") == pytest.approx(0.80, rel=1e-4)
        assert table_8_9_shansep_m("CIUC") == pytest.approx(0.85, rel=1e-4)
        assert table_8_9_shansep_m("CK0UE") == pytest.approx(0.70, rel=1e-4)


class TestTable824Integration:
    """Chain table_8_24 → cbr_from_dcp_power."""

    def test_webster_dcp10(self):
        # Webster (1992): CBR = 292 * DCP^(-1.12), DCP=10 mm/blow
        coeffs = table_8_24_cbr_dcp("webster_1992")
        cbr = cbr_from_dcp_power(coeffs["a"], 10.0, coeffs["x"])
        # 292 * 10^(-1.12) = 292 / 13.18 = 22.15
        expected = 292.0 * 10.0 ** (-1.12)
        assert cbr == pytest.approx(expected, rel=1e-4)

    def test_webster_dcp25(self):
        # DCP=25 mm/blow
        coeffs = table_8_24_cbr_dcp("webster_1992")
        cbr = cbr_from_dcp_power(coeffs["a"], 25.0, coeffs["x"])
        expected = 292.0 * 25.0 ** (-1.12)
        assert cbr == pytest.approx(expected, rel=1e-4)

    def test_published_webster_formula(self):
        # Published: CBR = 292 * DCP^(-1.12)
        coeffs = table_8_24_cbr_dcp("webster_1992")
        assert coeffs["a"] == pytest.approx(292.0, rel=1e-4)
        assert coeffs["x"] == pytest.approx(-1.12, rel=1e-4)

    def test_all_sources_give_positive_cbr(self):
        # For DCP=15 mm/blow, all sources should give positive CBR
        for source in ["webster_1992", "ese_2006", "livneh_2000",
                        "harrison_1987", "chua_1988", "pen_2002"]:
            coeffs = table_8_24_cbr_dcp(source)
            cbr = cbr_from_dcp_power(coeffs["a"], 15.0, coeffs["x"])
            assert cbr > 0.0, f"Negative CBR for {source}: {cbr}"


class TestFigure846Integration:
    """Chain figure_8_46_f → constrained_modulus_spt."""

    def test_pi20_n15(self):
        # PI=20, N=15, Pa=101.3 kPa
        f = figure_8_46_f(20.0)
        M_ds = constrained_modulus_spt(f, 15.0, 101.3)
        # M = 400 * 15 * 101.3 = 607,800 kPa
        assert M_ds == pytest.approx(f * 15.0 * 101.3, rel=1e-6)
        assert M_ds > 0.0

    def test_higher_pi_lower_modulus(self):
        # Higher PI → lower f → lower constrained modulus
        f_low_pi = figure_8_46_f(15.0)
        f_high_pi = figure_8_46_f(60.0)
        M_low = constrained_modulus_spt(f_low_pi, 20.0, 101.3)
        M_high = constrained_modulus_spt(f_high_pi, 20.0, 101.3)
        assert M_high < M_low


# ===========================================================================
# Plot function smoke tests
# ===========================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt


class TestPlotFigure846:
    """Smoke tests for plot_figure_8_46 (f vs PI curve)."""

    def test_no_query(self):
        ax = plot_figure_8_46(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        _plt.close("all")

    def test_query_point(self):
        ax = plot_figure_8_46(PI=30.0, show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        _plt.close("all")
