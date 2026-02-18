"""Comprehensive tests for geotech.dm7_2.chapter4 -- UFC 3-220-20 Chapter 4.

Tests cover all 33 public functions (Equations 4-1 through 4-31) for:
  - Basic valid inputs with hand-calculated expected results
  - Edge cases where applicable
  - ValueError checks for every validation branch
"""

import math

import pytest

from geotech.dm7_2.chapter4 import *


# ===========================================================================
# Equation 4-1: at_rest_earth_pressure_coefficient
# ===========================================================================

class TestAtRestEarthPressureCoefficient:

    def test_basic_valid(self):
        # K0 = sigma_h / sigma_z = 600 / 1200 = 0.5
        result = at_rest_earth_pressure_coefficient(600.0, 1200.0)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_equal_stresses(self):
        # K0 = 1000 / 1000 = 1.0
        result = at_rest_earth_pressure_coefficient(1000.0, 1000.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_zero_horizontal_stress(self):
        # K0 = 0 / 500 = 0.0
        result = at_rest_earth_pressure_coefficient(0.0, 500.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_zero_sigma_z(self):
        with pytest.raises(ValueError, match="sigma_z_eff must be positive"):
            at_rest_earth_pressure_coefficient(100.0, 0.0)

    def test_raises_negative_sigma_z(self):
        with pytest.raises(ValueError, match="sigma_z_eff must be positive"):
            at_rest_earth_pressure_coefficient(100.0, -50.0)


# ===========================================================================
# Equation 4-2: at_rest_coefficient_mayne_kulhawy
# ===========================================================================

class TestAtRestCoefficientMayneKulhawy:

    def test_normally_consolidated_phi30(self):
        # OCR=1: K0 = (1 - sin(30)) * 1^sin(30) = (1 - 0.5) * 1 = 0.5
        result = at_rest_coefficient_mayne_kulhawy(30.0, OCR=1.0)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_overconsolidated_phi30_ocr4(self):
        # K0 = (1 - sin(30)) * 4^sin(30) = 0.5 * 4^0.5 = 0.5 * 2.0 = 1.0
        result = at_rest_coefficient_mayne_kulhawy(30.0, OCR=4.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_default_ocr(self):
        # Default OCR=1.0; phi=45 => sin(45)=sqrt(2)/2 ~ 0.70711
        # K0 = (1 - 0.70711) * 1.0 = 0.29289
        result = at_rest_coefficient_mayne_kulhawy(45.0)
        expected = 1.0 - math.sin(math.radians(45.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            at_rest_coefficient_mayne_kulhawy(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            at_rest_coefficient_mayne_kulhawy(90.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            at_rest_coefficient_mayne_kulhawy(-5.0)

    def test_raises_ocr_less_than_one(self):
        with pytest.raises(ValueError, match="OCR must be >= 1.0"):
            at_rest_coefficient_mayne_kulhawy(30.0, OCR=0.5)


# ===========================================================================
# Equation 4-3: rankine_active_pressure_coefficient
# ===========================================================================

class TestRankineActivePressureCoefficient:

    def test_phi_30(self):
        # Ka = tan^2(45 - 15) = tan^2(30) = (1/sqrt(3))^2 = 1/3
        result = rankine_active_pressure_coefficient(30.0)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_phi_45(self):
        # Ka = tan^2(45 - 22.5) = tan^2(22.5)
        # tan(22.5) = sqrt(2) - 1 ~ 0.41421
        # Ka ~ 0.17157
        result = rankine_active_pressure_coefficient(45.0)
        expected = math.tan(math.radians(22.5)) ** 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_active_pressure_coefficient(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_active_pressure_coefficient(90.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_active_pressure_coefficient(-10.0)


# ===========================================================================
# Equation 4-4: rankine_passive_pressure_coefficient
# ===========================================================================

class TestRankinePassivePressureCoefficient:

    def test_phi_30(self):
        # Kp = tan^2(45 + 15) = tan^2(60) = (sqrt(3))^2 = 3.0
        result = rankine_passive_pressure_coefficient(30.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_phi_45(self):
        # Kp = tan^2(45 + 22.5) = tan^2(67.5)
        # tan(67.5) = 1 + sqrt(2) ~ 2.41421
        # Kp ~ 5.82843
        result = rankine_passive_pressure_coefficient(45.0)
        expected = math.tan(math.radians(67.5)) ** 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_passive_pressure_coefficient(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_passive_pressure_coefficient(90.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            rankine_passive_pressure_coefficient(-1.0)


# ===========================================================================
# Equation 4-5: rankine_Ka_Kp_reciprocal
# ===========================================================================

class TestRankineKaKpReciprocal:

    def test_basic_valid(self):
        # Kp = 1 / Ka = 1 / 0.333 = 3.003003...
        result = rankine_Ka_Kp_reciprocal(1.0 / 3.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_ka_one(self):
        # Kp = 1 / 1 = 1
        result = rankine_Ka_Kp_reciprocal(1.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_raises_ka_zero(self):
        with pytest.raises(ValueError, match="Ka must be positive"):
            rankine_Ka_Kp_reciprocal(0.0)

    def test_raises_ka_negative(self):
        with pytest.raises(ValueError, match="Ka must be positive"):
            rankine_Ka_Kp_reciprocal(-0.5)


# ===========================================================================
# Equation 4-6: rankine_active_horizontal_stress
# ===========================================================================

class TestRankineActiveHorizontalStress:

    def test_no_cohesion(self):
        # sigma_h = Ka * sigma_z - 2*c'*sqrt(Ka)
        # = 0.333 * 1000 - 0 = 333.0
        Ka = 1.0 / 3.0
        result = rankine_active_horizontal_stress(Ka, 1000.0)
        assert result == pytest.approx(1000.0 / 3.0, rel=1e-4)

    def test_with_cohesion(self):
        # Ka=0.25, sigma_z=2000, c'=100
        # sigma_h = 0.25*2000 - 2*100*sqrt(0.25) = 500 - 2*100*0.5 = 500 - 100 = 400
        result = rankine_active_horizontal_stress(0.25, 2000.0, c_prime=100.0)
        assert result == pytest.approx(400.0, rel=1e-4)

    def test_zero_vertical_stress(self):
        # Ka=0.333, sigma_z=0, c'=0 => sigma_h = 0
        result = rankine_active_horizontal_stress(1.0 / 3.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_ka_zero(self):
        with pytest.raises(ValueError, match="Ka must be positive"):
            rankine_active_horizontal_stress(0.0, 1000.0)

    def test_raises_ka_negative(self):
        with pytest.raises(ValueError, match="Ka must be positive"):
            rankine_active_horizontal_stress(-0.5, 1000.0)


# ===========================================================================
# Equation 4-7: rankine_passive_horizontal_stress
# ===========================================================================

class TestRankinePassiveHorizontalStress:

    def test_no_cohesion(self):
        # sigma_h = Kp * sigma_z + 2*c'*sqrt(Kp)
        # = 3.0 * 1000 + 0 = 3000
        result = rankine_passive_horizontal_stress(3.0, 1000.0)
        assert result == pytest.approx(3000.0, rel=1e-4)

    def test_with_cohesion(self):
        # Kp=4.0, sigma_z=500, c'=200
        # sigma_h = 4.0*500 + 2*200*sqrt(4) = 2000 + 2*200*2 = 2000 + 800 = 2800
        result = rankine_passive_horizontal_stress(4.0, 500.0, c_prime=200.0)
        assert result == pytest.approx(2800.0, rel=1e-4)

    def test_zero_vertical_stress_with_cohesion(self):
        # Kp=3.0, sigma_z=0, c'=100
        # sigma_h = 0 + 2*100*sqrt(3) = 200*1.7321 = 346.41
        result = rankine_passive_horizontal_stress(3.0, 0.0, c_prime=100.0)
        expected = 2.0 * 100.0 * math.sqrt(3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_kp_zero(self):
        with pytest.raises(ValueError, match="Kp must be positive"):
            rankine_passive_horizontal_stress(0.0, 1000.0)

    def test_raises_kp_negative(self):
        with pytest.raises(ValueError, match="Kp must be positive"):
            rankine_passive_horizontal_stress(-1.0, 1000.0)


# ===========================================================================
# Equation 4-8: active_earth_pressure_resultant
# ===========================================================================

class TestActiveEarthPressureResultant:

    def test_basic_valid(self):
        # Pa = 0.5 * Ka * gamma * H^2 = 0.5 * 0.333 * 120 * 10^2
        # = 0.5 * 0.333 * 120 * 100 = 1998
        Ka = 1.0 / 3.0
        result = active_earth_pressure_resultant(Ka, 120.0, 10.0)
        expected = 0.5 * (1.0 / 3.0) * 120.0 * 100.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_height(self):
        result = active_earth_pressure_resultant(0.333, 120.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_ka_negative(self):
        with pytest.raises(ValueError, match="Ka must be non-negative"):
            active_earth_pressure_resultant(-0.1, 120.0, 10.0)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            active_earth_pressure_resultant(0.333, -120.0, 10.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            active_earth_pressure_resultant(0.333, 120.0, -5.0)


# ===========================================================================
# Equation 4-9: passive_earth_pressure_resultant
# ===========================================================================

class TestPassiveEarthPressureResultant:

    def test_basic_valid(self):
        # Pp = 0.5 * Kp * gamma * H^2 = 0.5 * 3.0 * 120 * 10^2 = 18000
        result = passive_earth_pressure_resultant(3.0, 120.0, 10.0)
        assert result == pytest.approx(18000.0, rel=1e-4)

    def test_zero_height(self):
        result = passive_earth_pressure_resultant(3.0, 120.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_kp_negative(self):
        with pytest.raises(ValueError, match="Kp must be non-negative"):
            passive_earth_pressure_resultant(-0.1, 120.0, 10.0)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            passive_earth_pressure_resultant(3.0, -120.0, 10.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            passive_earth_pressure_resultant(3.0, 120.0, -5.0)


# ===========================================================================
# Equation 4-10: coulomb_active_earth_pressure
# ===========================================================================

class TestCoulombActiveEarthPressure:

    def test_vertical_wall_horizontal_backfill_no_friction(self):
        # With theta=0, delta=0, beta=0, Coulomb reduces to Rankine:
        # Ka = cos^2(phi) / [1 + sqrt(sin(phi)*sin(phi))]^2
        #    = cos^2(30) / [1 + sin(30)]^2
        #    = 0.75 / (1.5)^2 = 0.75 / 2.25 = 1/3
        # Pa = 0.5 * 120 * 100 * (1/3) = 2000
        result = coulomb_active_earth_pressure(120.0, 10.0, 30.0)
        expected = 0.5 * 120.0 * 100.0 * (1.0 / 3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_with_wall_friction(self):
        # phi=30, theta=0, delta=15, beta=0
        phi = math.radians(30.0)
        delta = math.radians(15.0)
        num = math.cos(phi) ** 2
        cos_delta = math.cos(delta)
        sin_phi_delta = math.sin(phi + delta)
        sin_phi = math.sin(phi)
        sqrt_term = math.sqrt((sin_phi_delta * sin_phi) / (cos_delta * 1.0))
        denom = 1.0 * cos_delta * (1.0 + sqrt_term) ** 2
        Ka = num / denom
        expected = 0.5 * 120.0 * 100.0 * Ka
        result = coulomb_active_earth_pressure(120.0, 10.0, 30.0, delta_deg=15.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            coulomb_active_earth_pressure(-120.0, 10.0, 30.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            coulomb_active_earth_pressure(120.0, -10.0, 30.0)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_active_earth_pressure(120.0, 10.0, 0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_active_earth_pressure(120.0, 10.0, 90.0)

    def test_raises_beta_exceeds_phi(self):
        # beta=40 > phi=30 => sin(phi - beta) < 0
        with pytest.raises(ValueError, match="phi_prime_deg must be >= beta_deg"):
            coulomb_active_earth_pressure(120.0, 10.0, 30.0, beta_deg=40.0)


# ===========================================================================
# Equation 4-10 (coefficient): coulomb_active_coefficient
# ===========================================================================

class TestCoulombActiveCoefficient:

    def test_vertical_wall_horizontal_backfill_no_friction(self):
        # Should match Rankine Ka for phi=30: Ka = 1/3
        result = coulomb_active_coefficient(30.0)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_with_delta(self):
        # phi=30, delta=20, theta=0, beta=0
        phi = math.radians(30.0)
        delta = math.radians(20.0)
        num = math.cos(phi) ** 2
        cos_delta = math.cos(delta)
        sqrt_t = math.sqrt(
            (math.sin(phi + delta) * math.sin(phi)) / (cos_delta * 1.0)
        )
        denom = 1.0 * cos_delta * (1.0 + sqrt_t) ** 2
        expected = num / denom
        result = coulomb_active_coefficient(30.0, delta_deg=20.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_active_coefficient(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_active_coefficient(90.0)

    def test_raises_beta_exceeds_phi(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be >= beta_deg"):
            coulomb_active_coefficient(20.0, beta_deg=30.0)


# ===========================================================================
# Equation 4-11: coulomb_passive_earth_pressure
# ===========================================================================

class TestCoulombPassiveEarthPressure:

    def test_vertical_wall_horizontal_backfill_no_friction(self):
        # With theta=0, delta=0, beta=0, Coulomb passive reduces to Rankine:
        # Kp = cos^2(phi) / [1 - sin(phi)]^2 = cos^2(30) / (0.5)^2
        #    = 0.75 / 0.25 = 3.0
        # Pp = 0.5 * 120 * 100 * 3 = 18000
        result = coulomb_passive_earth_pressure(120.0, 10.0, 30.0)
        assert result == pytest.approx(18000.0, rel=1e-4)

    def test_with_small_delta(self):
        # phi=30, delta=5, theta=0, beta=0 -- compute by hand
        phi = math.radians(30.0)
        delta = math.radians(5.0)
        num = math.cos(phi) ** 2
        cos_neg_delta = math.cos(-delta)  # cos(0 - delta) = cos(delta)
        sin_pd = math.sin(phi + delta)
        sin_pb = math.sin(phi)
        inner = (sin_pd * sin_pb) / (cos_neg_delta * 1.0)
        sqrt_term = math.sqrt(inner)
        bracket = 1.0 - sqrt_term
        denom = 1.0 * cos_neg_delta * bracket ** 2
        Kp = num / denom
        expected = 0.5 * 120.0 * 100.0 * Kp
        result = coulomb_passive_earth_pressure(120.0, 10.0, 30.0, delta_deg=5.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            coulomb_passive_earth_pressure(-120.0, 10.0, 30.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            coulomb_passive_earth_pressure(120.0, -10.0, 30.0)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_passive_earth_pressure(120.0, 10.0, 0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_passive_earth_pressure(120.0, 10.0, 90.0)

    def test_raises_bracket_nonpositive(self):
        # Large delta relative to phi can make bracket <= 0
        # delta=30, phi=30 => sin(60)*sin(30) / (cos(-30)*1) = 0.866*0.5/0.866 = 0.5
        # sqrt(0.5) ~ 0.707 => bracket = 1 - 0.707 = 0.293 (still positive)
        # delta=29, phi=30: sin(59)*sin(30)/cos(-29) = 0.857*0.5/0.875 = 0.4896
        # Try phi=10, delta=89 => cos(0-89)=cos(89)~0.0175
        # sin(10+89)=sin(99)=0.9877; sin(10+0)=sin(10)=0.1736
        # inner = 0.9877*0.1736/0.0175 = 9.80 => sqrt ~ 3.13 => bracket = -2.13
        with pytest.raises(ValueError, match="non-positive"):
            coulomb_passive_earth_pressure(120.0, 10.0, 10.0, delta_deg=89.0)


# ===========================================================================
# Equation 4-11 (coefficient): coulomb_passive_coefficient
# ===========================================================================

class TestCoulombPassiveCoefficient:

    def test_vertical_wall_no_friction(self):
        # Should match Rankine Kp for phi=30: Kp = 3.0
        result = coulomb_passive_coefficient(30.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_with_small_delta(self):
        phi = math.radians(30.0)
        delta = math.radians(10.0)
        num = math.cos(phi) ** 2
        cos_neg_delta = math.cos(-delta)
        inner = (math.sin(phi + delta) * math.sin(phi)) / (cos_neg_delta * 1.0)
        sqrt_term = math.sqrt(inner)
        bracket = 1.0 - sqrt_term
        denom = 1.0 * cos_neg_delta * bracket ** 2
        expected = num / denom
        result = coulomb_passive_coefficient(30.0, delta_deg=10.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_passive_coefficient(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            coulomb_passive_coefficient(90.0)

    def test_raises_bracket_nonpositive(self):
        with pytest.raises(ValueError, match="non-positive"):
            coulomb_passive_coefficient(10.0, delta_deg=89.0)


# ===========================================================================
# Equation 4-12: log_spiral_passive_coefficient
# ===========================================================================

class TestLogSpiralPassiveCoefficient:

    def test_delta_zero_reduces_to_rankine(self):
        # When delta=0, ln(Kp) = ln(ratio) + 0 = ln(ratio)
        # Kp = (1+sin(phi))/(1-sin(phi)) = Rankine Kp
        # For phi=30: Kp = (1+0.5)/(1-0.5) = 3.0
        result = log_spiral_passive_coefficient(30.0, delta_deg=0.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_phi30_delta15(self):
        # phi=30, delta=15
        sin_phi = math.sin(math.radians(30.0))  # 0.5
        ratio = (1.0 + 0.5) / (1.0 - 0.5)  # 3.0
        ln_ratio = math.log(3.0)  # 1.09861
        delta_phi_ratio = 15.0 / 30.0  # 0.5
        ln_Kp = ln_ratio + 1.443 * 0.5 * 0.5 * ln_ratio
        # = 1.09861 + 1.443 * 0.25 * 1.09861
        # = 1.09861 + 0.39629 = 1.49490
        expected = math.exp(ln_Kp)
        result = log_spiral_passive_coefficient(30.0, delta_deg=15.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_phi45_delta30(self):
        sin_phi = math.sin(math.radians(45.0))
        ratio = (1.0 + sin_phi) / (1.0 - sin_phi)
        ln_ratio = math.log(ratio)
        delta_phi_ratio = 30.0 / 45.0
        ln_Kp = ln_ratio + 1.443 * delta_phi_ratio * sin_phi * ln_ratio
        expected = math.exp(ln_Kp)
        result = log_spiral_passive_coefficient(45.0, delta_deg=30.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            log_spiral_passive_coefficient(0.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            log_spiral_passive_coefficient(90.0)

    def test_raises_delta_negative(self):
        with pytest.raises(ValueError, match="delta_deg must be non-negative"):
            log_spiral_passive_coefficient(30.0, delta_deg=-5.0)


# ===========================================================================
# Equation 4-13: equivalent_fluid_unit_weight
# ===========================================================================

class TestEquivalentFluidUnitWeight:

    def test_basic_valid(self):
        # gamma_eq = K * gamma = 0.333 * 120 = 39.96
        result = equivalent_fluid_unit_weight(120.0, 1.0 / 3.0)
        assert result == pytest.approx(40.0, rel=1e-4)

    def test_at_rest_coefficient(self):
        # gamma_eq = 0.5 * 120 = 60
        result = equivalent_fluid_unit_weight(120.0, 0.5)
        assert result == pytest.approx(60.0, rel=1e-4)

    def test_zero_gamma(self):
        result = equivalent_fluid_unit_weight(0.0, 0.5)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            equivalent_fluid_unit_weight(-120.0, 0.5)

    def test_raises_K_negative(self):
        with pytest.raises(ValueError, match="K must be non-negative"):
            equivalent_fluid_unit_weight(120.0, -0.1)


# ===========================================================================
# Equation 4-14: horizontal_earth_pressure_with_surcharge
# ===========================================================================

class TestHorizontalEarthPressureWithSurcharge:

    def test_basic_valid(self):
        # sigma_h = gamma_eq * z + K * q = 40 * 10 + 0.333 * 250
        # = 400 + 83.25 = 483.25
        result = horizontal_earth_pressure_with_surcharge(40.0, 10.0, 1.0 / 3.0, 250.0)
        expected = 40.0 * 10.0 + (1.0 / 3.0) * 250.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_no_surcharge(self):
        # sigma_h = 60 * 15 + 0.5 * 0 = 900
        result = horizontal_earth_pressure_with_surcharge(60.0, 15.0, 0.5, 0.0)
        assert result == pytest.approx(900.0, rel=1e-4)

    def test_zero_depth(self):
        # sigma_h = 60 * 0 + 0.5 * 200 = 100
        result = horizontal_earth_pressure_with_surcharge(60.0, 0.0, 0.5, 200.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_raises_z_negative(self):
        with pytest.raises(ValueError, match="z must be non-negative"):
            horizontal_earth_pressure_with_surcharge(40.0, -5.0, 0.5, 250.0)

    def test_raises_gamma_eq_negative(self):
        with pytest.raises(ValueError, match="gamma_eq must be non-negative"):
            horizontal_earth_pressure_with_surcharge(-40.0, 10.0, 0.5, 250.0)


# ===========================================================================
# Equation 4-15: mononobe_okabe_active_coefficient
# ===========================================================================

class TestMononobeOkabeActiveCoefficient:

    def test_no_seismic_reduces_to_coulomb(self):
        # kh=0, kv=0 => psi=0, reduces to Coulomb with defaults
        # For phi=30, theta=0, delta=0, beta=0 => Ka = 1/3
        result = mononobe_okabe_active_coefficient(30.0, kh=0.0)
        assert result == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_with_seismic(self):
        # phi=30, kh=0.2, kv=0, theta=0, delta=0, beta=0
        psi = math.atan(0.2 / 1.0)  # atan(0.2) ~ 0.19740 rad
        phi = math.radians(30.0)
        num = math.cos(phi - psi) ** 2
        cos_psi = math.cos(psi)
        cos_psi_term = math.cos(psi)  # cos(0+0+psi) = cos(psi)
        sin_phi_beta_psi = math.sin(phi - psi)  # sin(30 - psi)
        sin_phi = math.sin(phi)
        sqrt_t = math.sqrt((sin_phi * sin_phi_beta_psi) / (cos_psi * 1.0))
        denom = cos_psi * 1.0 * cos_psi * (1.0 + sqrt_t) ** 2
        expected = num / denom
        result = mononobe_okabe_active_coefficient(30.0, kh=0.2)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_with_kv(self):
        # phi=35, kh=0.15, kv=0.1
        psi = math.atan(0.15 / (1.0 - 0.1))  # atan(0.15/0.9) = atan(0.16667)
        phi = math.radians(35.0)
        num = math.cos(phi - psi) ** 2
        cos_psi = math.cos(psi)
        cos_psi2 = math.cos(psi)
        sin_pd = math.sin(phi)
        sin_pbp = math.sin(phi - psi)
        sqrt_t = math.sqrt((sin_pd * sin_pbp) / (cos_psi2 * 1.0))
        denom = cos_psi * cos_psi2 * (1.0 + sqrt_t) ** 2
        expected = num / denom
        result = mononobe_okabe_active_coefficient(35.0, kh=0.15, kv=0.1)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            mononobe_okabe_active_coefficient(0.0, kh=0.2)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            mononobe_okabe_active_coefficient(90.0, kh=0.2)

    def test_raises_kv_one(self):
        with pytest.raises(ValueError, match="kv must be less than 1.0"):
            mononobe_okabe_active_coefficient(30.0, kh=0.2, kv=1.0)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            mononobe_okabe_active_coefficient(30.0, kh=-0.1)

    def test_raises_phi_beta_psi_negative(self):
        # Large kh and beta so phi - beta - psi < 0
        # phi=20, beta=15, kh=0.5, kv=0 => psi=atan(0.5)=26.57deg
        # phi-beta-psi = 20-15-26.57 = -21.57 deg => sin < 0
        with pytest.raises(ValueError, match="phi' - beta - psi < 0"):
            mononobe_okabe_active_coefficient(20.0, kh=0.5, beta_deg=15.0)


# ===========================================================================
# Equation 4-16: mononobe_okabe_active_force
# ===========================================================================

class TestMononobeOkabeActiveForce:

    def test_basic_valid(self):
        # PAE = 0.5 * gamma * (1 - kv) * KAE * H^2
        # = 0.5 * 120 * (1 - 0) * 0.5 * 100 = 3000
        result = mononobe_okabe_active_force(120.0, 10.0, 0.5, kv=0.0)
        assert result == pytest.approx(3000.0, rel=1e-4)

    def test_with_kv(self):
        # PAE = 0.5 * 120 * (1 - 0.1) * 0.5 * 100 = 0.5 * 120 * 0.9 * 0.5 * 100
        # = 2700
        result = mononobe_okabe_active_force(120.0, 10.0, 0.5, kv=0.1)
        assert result == pytest.approx(2700.0, rel=1e-4)

    def test_zero_height(self):
        result = mononobe_okabe_active_force(120.0, 0.0, 0.5)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            mononobe_okabe_active_force(-120.0, 10.0, 0.5)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            mononobe_okabe_active_force(120.0, -10.0, 0.5)

    def test_raises_kv_one(self):
        with pytest.raises(ValueError, match="kv must be less than 1.0"):
            mononobe_okabe_active_force(120.0, 10.0, 0.5, kv=1.0)


# ===========================================================================
# Equation 4-17: seed_whitman_dynamic_increment
# ===========================================================================

class TestSeedWhitmanDynamicIncrement:

    def test_basic_valid(self):
        # delta_PAE = (3/8) * kh * gamma * H^2
        # = 0.375 * 0.2 * 120 * 100 = 900
        result = seed_whitman_dynamic_increment(120.0, 10.0, 0.2)
        assert result == pytest.approx(900.0, rel=1e-4)

    def test_zero_kh(self):
        result = seed_whitman_dynamic_increment(120.0, 10.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_height(self):
        result = seed_whitman_dynamic_increment(120.0, 0.0, 0.2)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            seed_whitman_dynamic_increment(-120.0, 10.0, 0.2)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            seed_whitman_dynamic_increment(120.0, -10.0, 0.2)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            seed_whitman_dynamic_increment(120.0, 10.0, -0.1)


# ===========================================================================
# Equation 4-18: wood_nonyielding_seismic_force
# ===========================================================================

class TestWoodNonyieldingSeismicForce:

    def test_basic_valid(self):
        # delta_PE = kh * gamma * H^2 = 0.2 * 120 * 100 = 2400
        result = wood_nonyielding_seismic_force(120.0, 10.0, 0.2)
        assert result == pytest.approx(2400.0, rel=1e-4)

    def test_zero_kh(self):
        result = wood_nonyielding_seismic_force(120.0, 10.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            wood_nonyielding_seismic_force(-120.0, 10.0, 0.2)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            wood_nonyielding_seismic_force(120.0, -10.0, 0.2)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            wood_nonyielding_seismic_force(120.0, 10.0, -0.1)


# ===========================================================================
# Equation 4-19: hydrodynamic_water_pressure
# ===========================================================================

class TestHydrodynamicWaterPressure:

    def test_basic_valid(self):
        # pw = 0.875 * kh * gamma_w * sqrt(H * z)
        # = 0.875 * 0.2 * 62.4 * sqrt(20 * 10)
        # = 0.875 * 0.2 * 62.4 * sqrt(200)
        # = 0.875 * 0.2 * 62.4 * 14.1421 = 154.34
        result = hydrodynamic_water_pressure(62.4, 0.2, 20.0, 10.0)
        expected = 0.875 * 0.2 * 62.4 * math.sqrt(200.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_at_surface_z_zero(self):
        # pw = 0.875 * 0.2 * 62.4 * sqrt(20 * 0) = 0
        result = hydrodynamic_water_pressure(62.4, 0.2, 20.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_z_equals_H(self):
        # pw = 0.875 * 0.2 * 62.4 * sqrt(20 * 20) = 0.875 * 0.2 * 62.4 * 20 = 218.4
        result = hydrodynamic_water_pressure(62.4, 0.2, 20.0, 20.0)
        expected = 0.875 * 0.2 * 62.4 * 20.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be non-negative"):
            hydrodynamic_water_pressure(-62.4, 0.2, 20.0, 10.0)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            hydrodynamic_water_pressure(62.4, -0.2, 20.0, 10.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            hydrodynamic_water_pressure(62.4, 0.2, -20.0, 10.0)

    def test_raises_z_negative(self):
        with pytest.raises(ValueError, match="z must be non-negative"):
            hydrodynamic_water_pressure(62.4, 0.2, 20.0, -5.0)

    def test_raises_z_exceeds_H(self):
        with pytest.raises(ValueError, match="z must not exceed H"):
            hydrodynamic_water_pressure(62.4, 0.2, 20.0, 25.0)


# ===========================================================================
# Equation 4-20: hydrodynamic_water_force
# ===========================================================================

class TestHydrodynamicWaterForce:

    def test_basic_valid(self):
        # Pw = (7/12) * kh * gamma_w * H^2
        # = 0.58333 * 0.2 * 62.4 * 400 = 2912.0
        result = hydrodynamic_water_force(62.4, 0.2, 20.0)
        expected = (7.0 / 12.0) * 0.2 * 62.4 * 400.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_kh(self):
        result = hydrodynamic_water_force(62.4, 0.0, 20.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_H(self):
        result = hydrodynamic_water_force(62.4, 0.2, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be non-negative"):
            hydrodynamic_water_force(-62.4, 0.2, 20.0)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            hydrodynamic_water_force(62.4, -0.2, 20.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be non-negative"):
            hydrodynamic_water_force(62.4, 0.2, -20.0)


# ===========================================================================
# Equation 4-21: seismic_psi_low_permeability
# ===========================================================================

class TestSeismicPsiLowPermeability:

    def test_basic_valid(self):
        # psi = atan(gamma_sat * kh / ((gamma_sat - gamma_w)*(1 - ru)*(1 - kv)))
        # gamma_sat=125, gamma_w=62.4, kh=0.2, kv=0, ru=0
        # = atan(125*0.2 / ((125-62.4)*1*1)) = atan(25/62.6) = atan(0.3994)
        result = seismic_psi_low_permeability(125.0, 62.4, 0.2)
        expected = math.atan(125.0 * 0.2 / (125.0 - 62.4))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_with_kv_and_ru(self):
        # gamma_sat=130, gamma_w=62.4, kh=0.15, kv=0.05, ru=0.3
        # num = 130 * 0.15 = 19.5
        # denom = (130 - 62.4) * (1 - 0.3) * (1 - 0.05) = 67.6 * 0.7 * 0.95 = 44.954
        # psi = atan(19.5 / 44.954) = atan(0.43379)
        result = seismic_psi_low_permeability(130.0, 62.4, 0.15, kv=0.05, ru=0.3)
        num = 130.0 * 0.15
        denom = (130.0 - 62.4) * 0.7 * 0.95
        expected = math.atan(num / denom)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_gamma_sat_le_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_sat must be greater than gamma_w"):
            seismic_psi_low_permeability(62.4, 62.4, 0.2)

    def test_raises_gamma_sat_less_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_sat must be greater than gamma_w"):
            seismic_psi_low_permeability(50.0, 62.4, 0.2)

    def test_raises_kv_one(self):
        with pytest.raises(ValueError, match="kv must be less than 1.0"):
            seismic_psi_low_permeability(125.0, 62.4, 0.2, kv=1.0)

    def test_raises_ru_one(self):
        with pytest.raises(ValueError, match="ru must be less than 1.0"):
            seismic_psi_low_permeability(125.0, 62.4, 0.2, ru=1.0)

    def test_raises_kh_negative(self):
        with pytest.raises(ValueError, match="kh must be non-negative"):
            seismic_psi_low_permeability(125.0, 62.4, -0.1)


# ===========================================================================
# Equation 4-22: seismic_adjusted_unit_weight
# ===========================================================================

class TestSeismicAdjustedUnitWeight:

    def test_basic_no_ru(self):
        # gamma = (gamma_sat - gamma_w) * (1 - ru) = (125 - 62.4) * 1 = 62.6
        result = seismic_adjusted_unit_weight(125.0, 62.4)
        assert result == pytest.approx(62.6, rel=1e-4)

    def test_with_ru(self):
        # gamma = (125 - 62.4) * (1 - 0.3) = 62.6 * 0.7 = 43.82
        result = seismic_adjusted_unit_weight(125.0, 62.4, ru=0.3)
        assert result == pytest.approx(43.82, rel=1e-4)

    def test_ru_zero_explicit(self):
        result = seismic_adjusted_unit_weight(130.0, 62.4, ru=0.0)
        assert result == pytest.approx(67.6, rel=1e-4)

    def test_raises_gamma_sat_le_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_sat must be greater than gamma_w"):
            seismic_adjusted_unit_weight(62.4, 62.4)

    def test_raises_gamma_sat_less_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_sat must be greater than gamma_w"):
            seismic_adjusted_unit_weight(50.0, 62.4)

    def test_raises_ru_one(self):
        with pytest.raises(ValueError, match="ru must be less than 1.0"):
            seismic_adjusted_unit_weight(125.0, 62.4, ru=1.0)


# ===========================================================================
# Equation 4-23: base_pressure_max_trapezoidal
# ===========================================================================

class TestBasePressureMaxTrapezoidal:

    def test_basic_valid(self):
        # qmax = R/B + 6*R*e/B^2
        # R=10000, e=1.0, B=8
        # = 10000/8 + 6*10000*1/64 = 1250 + 937.5 = 2187.5
        result = base_pressure_max_trapezoidal(10000.0, 1.0, 8.0)
        assert result == pytest.approx(2187.5, rel=1e-4)

    def test_zero_eccentricity(self):
        # qmax = R/B + 0 = 10000/8 = 1250
        result = base_pressure_max_trapezoidal(10000.0, 0.0, 8.0)
        assert result == pytest.approx(1250.0, rel=1e-4)

    def test_eccentricity_at_limit(self):
        # e = B/6 = 8/6 = 1.333 (exactly at limit)
        B = 8.0
        e = B / 6.0
        result = base_pressure_max_trapezoidal(10000.0, e, B)
        expected = 10000.0 / B + 6.0 * 10000.0 * e / B ** 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_B_zero(self):
        with pytest.raises(ValueError, match="B must be positive"):
            base_pressure_max_trapezoidal(10000.0, 0.5, 0.0)

    def test_raises_B_negative(self):
        with pytest.raises(ValueError, match="B must be positive"):
            base_pressure_max_trapezoidal(10000.0, 0.5, -5.0)

    def test_raises_eccentricity_exceeds_middle_third(self):
        # e > B/6: e=2.0, B=8, B/6=1.333
        with pytest.raises(ValueError, match="Eccentricity exceeds B/6"):
            base_pressure_max_trapezoidal(10000.0, 2.0, 8.0)


# ===========================================================================
# Equation 4-24: base_pressure_min_trapezoidal
# ===========================================================================

class TestBasePressureMinTrapezoidal:

    def test_basic_valid(self):
        # qmin = R/B - 6*R*e/B^2
        # R=10000, e=1.0, B=8
        # = 1250 - 937.5 = 312.5
        result = base_pressure_min_trapezoidal(10000.0, 1.0, 8.0)
        assert result == pytest.approx(312.5, rel=1e-4)

    def test_zero_eccentricity(self):
        # qmin = R/B = 10000/8 = 1250 (uniform pressure)
        result = base_pressure_min_trapezoidal(10000.0, 0.0, 8.0)
        assert result == pytest.approx(1250.0, rel=1e-4)

    def test_eccentricity_at_limit(self):
        # e = B/6 => qmin = R/B - R/B = 0 (triangular limit)
        B = 6.0
        e = B / 6.0  # e = 1.0
        result = base_pressure_min_trapezoidal(10000.0, e, B)
        # R/B - 6*R*e/B^2 = 10000/6 - 6*10000*1/36 = 1666.67 - 1666.67 = 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_raises_B_zero(self):
        with pytest.raises(ValueError, match="B must be positive"):
            base_pressure_min_trapezoidal(10000.0, 0.5, 0.0)

    def test_raises_B_negative(self):
        with pytest.raises(ValueError, match="B must be positive"):
            base_pressure_min_trapezoidal(10000.0, 0.5, -5.0)

    def test_raises_eccentricity_exceeds_middle_third(self):
        with pytest.raises(ValueError, match="Eccentricity exceeds B/6"):
            base_pressure_min_trapezoidal(10000.0, 2.0, 8.0)


# ===========================================================================
# Equation 4-25: base_pressure_max_triangular
# ===========================================================================

class TestBasePressureMaxTriangular:

    def test_basic_valid(self):
        # qmax = 2R / (3*x0) = 2*10000 / (3*2) = 20000/6 = 3333.33
        result = base_pressure_max_triangular(10000.0, 2.0)
        assert result == pytest.approx(3333.333333, rel=1e-4)

    def test_large_x0(self):
        # qmax = 2*5000 / (3*5) = 10000/15 = 666.667
        result = base_pressure_max_triangular(5000.0, 5.0)
        assert result == pytest.approx(666.6667, rel=1e-4)

    def test_raises_x0_zero(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            base_pressure_max_triangular(10000.0, 0.0)

    def test_raises_x0_negative(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            base_pressure_max_triangular(10000.0, -1.0)


# ===========================================================================
# Equation 4-26: effective_base_width_triangular
# ===========================================================================

class TestEffectiveBaseWidthTriangular:

    def test_basic_valid(self):
        # Be = 3 * x0 = 3 * 2.0 = 6.0
        result = effective_base_width_triangular(2.0)
        assert result == pytest.approx(6.0, rel=1e-4)

    def test_small_x0(self):
        # Be = 3 * 0.5 = 1.5
        result = effective_base_width_triangular(0.5)
        assert result == pytest.approx(1.5, rel=1e-4)

    def test_raises_x0_zero(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            effective_base_width_triangular(0.0)

    def test_raises_x0_negative(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            effective_base_width_triangular(-1.0)


# ===========================================================================
# Equation 4-27: base_pressure_uniform
# ===========================================================================

class TestBasePressureUniform:

    def test_basic_valid(self):
        # qmax = R / (2*x0) = 10000 / (2*2) = 2500
        result = base_pressure_uniform(10000.0, 2.0)
        assert result == pytest.approx(2500.0, rel=1e-4)

    def test_large_x0(self):
        # qmax = 6000 / (2*3) = 1000
        result = base_pressure_uniform(6000.0, 3.0)
        assert result == pytest.approx(1000.0, rel=1e-4)

    def test_raises_x0_zero(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            base_pressure_uniform(10000.0, 0.0)

    def test_raises_x0_negative(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            base_pressure_uniform(10000.0, -1.0)


# ===========================================================================
# Equation 4-28: effective_base_width_uniform
# ===========================================================================

class TestEffectiveBaseWidthUniform:

    def test_basic_valid(self):
        # Be = 2 * x0 = 2 * 2.0 = 4.0
        result = effective_base_width_uniform(2.0)
        assert result == pytest.approx(4.0, rel=1e-4)

    def test_small_x0(self):
        # Be = 2 * 1.5 = 3.0
        result = effective_base_width_uniform(1.5)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_raises_x0_zero(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            effective_base_width_uniform(0.0)

    def test_raises_x0_negative(self):
        with pytest.raises(ValueError, match="x0 must be positive"):
            effective_base_width_uniform(-1.0)


# ===========================================================================
# Equation 4-29: allowable_passive_resistance
# ===========================================================================

class TestAllowablePassiveResistance:

    def test_basic_valid(self):
        # Pp_allow = Pp / F = 18000 / 2.0 = 9000
        result = allowable_passive_resistance(18000.0, 2.0)
        assert result == pytest.approx(9000.0, rel=1e-4)

    def test_higher_fs(self):
        # Pp_allow = 18000 / 3.0 = 6000
        result = allowable_passive_resistance(18000.0, 3.0)
        assert result == pytest.approx(6000.0, rel=1e-4)

    def test_zero_pp(self):
        result = allowable_passive_resistance(0.0, 2.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_pp_negative(self):
        with pytest.raises(ValueError, match="Pp must be non-negative"):
            allowable_passive_resistance(-1000.0, 2.0)

    def test_raises_F_one(self):
        with pytest.raises(ValueError, match="Factor of safety F must be greater than 1.0"):
            allowable_passive_resistance(18000.0, 1.0)

    def test_raises_F_less_than_one(self):
        with pytest.raises(ValueError, match="Factor of safety F must be greater than 1.0"):
            allowable_passive_resistance(18000.0, 0.5)


# ===========================================================================
# Equation 4-30: allowable_friction_angle
# ===========================================================================

class TestAllowableFrictionAngle:

    def test_basic_valid(self):
        # phi_allow = atan(tan(30)/2.0)
        # tan(30) = 1/sqrt(3) ~ 0.57735
        # tan(30)/2 = 0.28868
        # atan(0.28868) ~ 16.10 deg
        result = allowable_friction_angle(30.0, 2.0)
        expected = math.degrees(math.atan(math.tan(math.radians(30.0)) / 2.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_phi_45_fs_1p5(self):
        # phi_allow = atan(tan(45)/1.5) = atan(1/1.5) = atan(0.6667) ~ 33.69 deg
        result = allowable_friction_angle(45.0, 1.5)
        expected = math.degrees(math.atan(1.0 / 1.5))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_result_less_than_input(self):
        result = allowable_friction_angle(35.0, 2.0)
        assert result < 35.0

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            allowable_friction_angle(0.0, 2.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            allowable_friction_angle(90.0, 2.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be between 0 and 90"):
            allowable_friction_angle(-10.0, 2.0)

    def test_raises_F_one(self):
        with pytest.raises(ValueError, match="Factor of safety F must be greater than 1.0"):
            allowable_friction_angle(30.0, 1.0)

    def test_raises_F_less_than_one(self):
        with pytest.raises(ValueError, match="Factor of safety F must be greater than 1.0"):
            allowable_friction_angle(30.0, 0.8)


# ===========================================================================
# Equation 4-31: relative_flexibility_anchored_bulkhead
# ===========================================================================

class TestRelativeFlexibilityAnchoredBulkhead:

    def test_basic_valid(self):
        # rho = (H + D)^4 / (E * I)
        # H=120, D=60, E=29e6, I=200
        # (120+60)^4 = 180^4 = 1,049,760,000
        # E*I = 29e6 * 200 = 5.8e9
        # rho = 1,049,760,000 / 5,800,000,000 = 0.18099
        H = 120.0
        D = 60.0
        E = 29.0e6
        I_val = 200.0
        result = relative_flexibility_anchored_bulkhead(H, D, E, I_val)
        expected = (180.0) ** 4 / (29.0e6 * 200.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_small_wall(self):
        # H=48, D=24, E=29e6, I=50
        # (48+24)^4 = 72^4 = 26,873,856
        # E*I = 29e6 * 50 = 1.45e9
        # rho = 26,873,856 / 1,450,000,000 = 0.01853
        result = relative_flexibility_anchored_bulkhead(48.0, 24.0, 29.0e6, 50.0)
        expected = 72.0 ** 4 / (29.0e6 * 50.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_H_zero(self):
        with pytest.raises(ValueError, match="H must be positive"):
            relative_flexibility_anchored_bulkhead(0.0, 60.0, 29e6, 200.0)

    def test_raises_H_negative(self):
        with pytest.raises(ValueError, match="H must be positive"):
            relative_flexibility_anchored_bulkhead(-10.0, 60.0, 29e6, 200.0)

    def test_raises_D_zero(self):
        with pytest.raises(ValueError, match="D must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, 0.0, 29e6, 200.0)

    def test_raises_D_negative(self):
        with pytest.raises(ValueError, match="D must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, -10.0, 29e6, 200.0)

    def test_raises_E_zero(self):
        with pytest.raises(ValueError, match="E must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, 60.0, 0.0, 200.0)

    def test_raises_E_negative(self):
        with pytest.raises(ValueError, match="E must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, 60.0, -29e6, 200.0)

    def test_raises_I_zero(self):
        with pytest.raises(ValueError, match="I must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, 60.0, 29e6, 0.0)

    def test_raises_I_negative(self):
        with pytest.raises(ValueError, match="I must be positive"):
            relative_flexibility_anchored_bulkhead(120.0, 60.0, 29e6, -200.0)


# ===========================================================================
# Cross-function consistency checks
# ===========================================================================

class TestCrossFunctionConsistency:

    def test_rankine_ka_kp_reciprocal_relationship(self):
        """Ka and Kp from Rankine should be reciprocals of each other."""
        Ka = rankine_active_pressure_coefficient(30.0)
        Kp = rankine_passive_pressure_coefficient(30.0)
        assert Ka * Kp == pytest.approx(1.0, rel=1e-4)

    def test_rankine_ka_kp_via_reciprocal_function(self):
        """rankine_Ka_Kp_reciprocal(Ka) should equal Kp from Rankine."""
        Ka = rankine_active_pressure_coefficient(30.0)
        Kp_from_reciprocal = rankine_Ka_Kp_reciprocal(Ka)
        Kp_from_rankine = rankine_passive_pressure_coefficient(30.0)
        assert Kp_from_reciprocal == pytest.approx(Kp_from_rankine, rel=1e-4)

    def test_coulomb_matches_rankine_simple_case(self):
        """Coulomb with defaults should equal Rankine for vertical wall,
        horizontal backfill, no friction."""
        Ka_rankine = rankine_active_pressure_coefficient(30.0)
        Ka_coulomb = coulomb_active_coefficient(30.0)
        assert Ka_coulomb == pytest.approx(Ka_rankine, rel=1e-4)

    def test_coulomb_passive_matches_rankine_simple_case(self):
        """Coulomb passive with defaults should equal Rankine passive."""
        Kp_rankine = rankine_passive_pressure_coefficient(30.0)
        Kp_coulomb = coulomb_passive_coefficient(30.0)
        assert Kp_coulomb == pytest.approx(Kp_rankine, rel=1e-4)

    def test_log_spiral_matches_rankine_when_delta_zero(self):
        """Log spiral with delta=0 should equal Rankine Kp."""
        Kp_rankine = rankine_passive_pressure_coefficient(35.0)
        Kp_log_spiral = log_spiral_passive_coefficient(35.0, delta_deg=0.0)
        assert Kp_log_spiral == pytest.approx(Kp_rankine, rel=1e-4)

    def test_mo_reduces_to_coulomb_no_seismic(self):
        """M-O with kh=0, kv=0 should equal Coulomb active coefficient."""
        Ka_coulomb = coulomb_active_coefficient(35.0, theta_deg=5.0, delta_deg=10.0, beta_deg=5.0)
        KAE = mononobe_okabe_active_coefficient(35.0, kh=0.0, kv=0.0, theta_deg=5.0, delta_deg=10.0, beta_deg=5.0)
        assert KAE == pytest.approx(Ka_coulomb, rel=1e-4)

    def test_active_passive_resultants_ratio(self):
        """Active and passive resultants should differ by Ka*Kp = 1."""
        Ka = rankine_active_pressure_coefficient(30.0)
        Kp = rankine_passive_pressure_coefficient(30.0)
        Pa = active_earth_pressure_resultant(Ka, 120.0, 10.0)
        Pp = passive_earth_pressure_resultant(Kp, 120.0, 10.0)
        # Pa/Pp = Ka/Kp = Ka^2 (since Kp = 1/Ka)
        assert (Pa / Pp) == pytest.approx(Ka ** 2, rel=1e-4)

    def test_equivalent_fluid_pressure_consistency(self):
        """Equivalent fluid unit weight * z should equal K*gamma*z."""
        gamma = 120.0
        K = 0.5
        z = 15.0
        gamma_eq = equivalent_fluid_unit_weight(gamma, K)
        sigma_h = horizontal_earth_pressure_with_surcharge(gamma_eq, z, K, 0.0)
        expected = K * gamma * z
        assert sigma_h == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Figure 4-36: figure_4_36_moment_reduction
# ===========================================================================

class TestFigure436MomentReduction:

    def test_dense_very_stiff(self):
        # log_rho = -3.5, dense_sand => Md/Mmax = 1.00
        result = figure_4_36_moment_reduction(-3.5, "dense_sand")
        assert result == pytest.approx(1.00, rel=1e-2)

    def test_dense_flexible(self):
        # log_rho = 0.0, dense_sand => Md/Mmax = 0.30
        result = figure_4_36_moment_reduction(0.0, "dense_sand")
        assert result == pytest.approx(0.30, rel=1e-2)

    def test_medium_very_stiff(self):
        # log_rho = -3.5, medium_sand => Md/Mmax = 1.00
        result = figure_4_36_moment_reduction(-3.5, "medium_sand")
        assert result == pytest.approx(1.00, rel=1e-2)

    def test_medium_flexible(self):
        # log_rho = 0.0, medium_sand => Md/Mmax = 0.38
        result = figure_4_36_moment_reduction(0.0, "medium_sand")
        assert result == pytest.approx(0.38, rel=1e-2)

    def test_loose_flexible(self):
        # log_rho = 0.0, loose_sand => Md/Mmax = 0.52
        result = figure_4_36_moment_reduction(0.0, "loose_sand")
        assert result == pytest.approx(0.52, rel=1e-2)

    def test_interpolated(self):
        # log_rho = -1.5, dense_sand => interpolate between 1.00 and 0.30
        # Expected value around 0.62
        result = figure_4_36_moment_reduction(-1.5, "dense_sand")
        assert result == pytest.approx(0.62, rel=1e-2)

    def test_clamped_low(self):
        # log_rho = -5.0 (below min -3.5), any soil type => clamped to 1.00
        result = figure_4_36_moment_reduction(-5.0, "medium_sand")
        assert result == pytest.approx(1.00, rel=1e-2)

    def test_clamped_high(self):
        # log_rho = 1.0 (above max 0.0), dense_sand => clamped to 0.30
        result = figure_4_36_moment_reduction(1.0, "dense_sand")
        assert result == pytest.approx(0.30, rel=1e-2)

    def test_unknown_soil_raises(self):
        with pytest.raises(ValueError, match="Unknown soil_type"):
            figure_4_36_moment_reduction(-1.0, "clay")


# ===========================================================================
# INTEGRATION: figure_4_36 + relative_flexibility_anchored_bulkhead
# ===========================================================================

class TestFigure436Integration:
    """Chain relative_flexibility_anchored_bulkhead → figure_4_36."""

    def test_steel_sheet_pile_dense_sand(self):
        # Typical: H=15ft, D=10ft, E=29e6 psi, I=38.6 in^4/ft
        rho = relative_flexibility_anchored_bulkhead(15.0, 10.0, 29e6, 38.6)
        log_rho = math.log10(rho)
        Md_ratio = figure_4_36_moment_reduction(log_rho, "dense_sand")
        # Md/Mmax should be between 0 and 1
        assert 0.0 < Md_ratio <= 1.0

    def test_flexible_wall_more_reduction(self):
        # More flexible wall (lower EI) → higher rho → more moment reduction
        rho_stiff = relative_flexibility_anchored_bulkhead(10.0, 8.0, 29e6, 100.0)
        rho_flex = relative_flexibility_anchored_bulkhead(10.0, 8.0, 29e6, 10.0)
        assert rho_flex > rho_stiff
        ratio_stiff = figure_4_36_moment_reduction(math.log10(rho_stiff), "medium_sand")
        ratio_flex = figure_4_36_moment_reduction(math.log10(rho_flex), "medium_sand")
        # More flexible → lower ratio (more reduction)
        assert ratio_flex <= ratio_stiff

    def test_dense_more_reduction_than_loose(self):
        # Same rho: dense sand allows more reduction than loose
        log_rho = -1.5
        dense = figure_4_36_moment_reduction(log_rho, "dense_sand")
        loose = figure_4_36_moment_reduction(log_rho, "loose_sand")
        assert dense <= loose


# ===========================================================================
# Plot function smoke tests
# ===========================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt


class TestPlotFigure436:
    """Smoke tests for plot_figure_4_36 (Rowe moment reduction curves)."""

    def test_no_query(self):
        ax = plot_figure_4_36(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        _plt.close("all")

    def test_query_point(self):
        ax = plot_figure_4_36(log_rho=-2.0, soil_type="medium_sand", show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        _plt.close("all")
