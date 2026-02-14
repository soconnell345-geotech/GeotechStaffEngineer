"""Comprehensive tests for geotech.dm7_2.prologue module.

Tests all 13 public functions from UFC 3-220-20, Prologue P:
  - mohr_coulomb_drained
  - mohr_coulomb_undrained_partial
  - mohr_coulomb_undrained_saturated
  - power_function_shear_strength
  - power_function_undrained_strength
  - three_parameter_power_function
  - equivalent_friction_angle
  - equivalent_cohesion
  - power_function_dimensional_to_normalized
  - power_function_normalized_to_dimensional
  - usr_icu_triaxial
  - usr_acu_triaxial
  - usr_dss
"""

import math

import pytest

from geotech.dm7_2.prologue import *


# ===========================================================================
# mohr_coulomb_drained  (Table P-1, Row 1)
#   s = c' + sigma' * tan(phi')
# ===========================================================================

class TestMohrCoulombDrained:

    def test_basic_case(self):
        # c'=200, sigma'=1000, phi'=30 deg
        # s = 200 + 1000*tan(30) = 200 + 577.350... = 777.350...
        result = mohr_coulomb_drained(200.0, 1000.0, 30.0)
        assert result == pytest.approx(777.3502691896257, rel=1e-4)

    def test_zero_cohesion(self):
        # c'=0, sigma'=500, phi'=45 deg
        # s = 0 + 500*tan(45) = 500.0
        result = mohr_coulomb_drained(0.0, 500.0, 45.0)
        assert result == pytest.approx(500.0, rel=1e-4)

    def test_zero_normal_stress(self):
        # c'=100, sigma'=0, phi'=30 deg
        # s = 100 + 0 = 100.0
        result = mohr_coulomb_drained(100.0, 0.0, 30.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_zero_friction_angle(self):
        # c'=100, sigma'=500, phi'=0 deg
        # s = 100 + 500*tan(0) = 100.0
        result = mohr_coulomb_drained(100.0, 500.0, 0.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_phi_negative_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            mohr_coulomb_drained(100.0, 500.0, -1.0)

    def test_phi_at_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            mohr_coulomb_drained(100.0, 500.0, 90.0)

    def test_phi_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            mohr_coulomb_drained(100.0, 500.0, 95.0)


# ===========================================================================
# mohr_coulomb_undrained_partial  (Table P-1, Row 2)
#   s = c + sigma * tan(phi)
# ===========================================================================

class TestMohrCoulombUndrainedPartial:

    def test_basic_case(self):
        # c=500, sigma=2000, phi=15 deg
        # s = 500 + 2000*tan(15) = 500 + 535.898... = 1035.898...
        result = mohr_coulomb_undrained_partial(500.0, 2000.0, 15.0)
        assert result == pytest.approx(1035.8983848622454, rel=1e-4)

    def test_zero_cohesion(self):
        # c=0, sigma=1000, phi=30 deg
        # s = 0 + 1000*tan(30) = 577.350...
        result = mohr_coulomb_undrained_partial(0.0, 1000.0, 30.0)
        assert result == pytest.approx(577.3502691896257, rel=1e-4)

    def test_zero_normal_stress(self):
        # c=300, sigma=0, phi=20 deg => s = 300.0
        result = mohr_coulomb_undrained_partial(300.0, 0.0, 20.0)
        assert result == pytest.approx(300.0, rel=1e-4)

    def test_phi_negative_raises(self):
        with pytest.raises(ValueError, match="phi_deg must be in the range"):
            mohr_coulomb_undrained_partial(100.0, 500.0, -0.5)

    def test_phi_at_90_raises(self):
        with pytest.raises(ValueError, match="phi_deg must be in the range"):
            mohr_coulomb_undrained_partial(100.0, 500.0, 90.0)

    def test_phi_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_deg must be in the range"):
            mohr_coulomb_undrained_partial(100.0, 500.0, 100.0)


# ===========================================================================
# mohr_coulomb_undrained_saturated  (Table P-1, Row 3)
#   s_u = c
# ===========================================================================

class TestMohrCoulombUndrainedSaturated:

    def test_basic_case(self):
        result = mohr_coulomb_undrained_saturated(250.0)
        assert result == pytest.approx(250.0, rel=1e-4)

    def test_zero_cohesion(self):
        result = mohr_coulomb_undrained_saturated(0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_large_value(self):
        result = mohr_coulomb_undrained_saturated(10000.0)
        assert result == pytest.approx(10000.0, rel=1e-4)

    def test_negative_c_raises(self):
        with pytest.raises(ValueError, match="c.*must be non-negative"):
            mohr_coulomb_undrained_saturated(-1.0)


# ===========================================================================
# power_function_shear_strength  (Equation P-1)
#   s = a * Pa * (sigma' / Pa)^b
# ===========================================================================

class TestPowerFunctionShearStrength:

    def test_basic_sigma_equals_pa(self):
        # a=0.8, b=0.9, sigma'=2116, pa=2116
        # s = 0.8*2116*(1)^0.9 = 1692.8
        result = power_function_shear_strength(0.8, 0.9, 2116.0, 2116.0)
        assert result == pytest.approx(1692.8, rel=1e-4)

    def test_sigma_double_pa(self):
        # a=1.2, b=0.75, sigma'=4232, pa=2116
        # s = 1.2*2116*(2)^0.75 = 1.2*2116*1.68179... = 4270.408...
        result = power_function_shear_strength(1.2, 0.75, 4232.0, 2116.0)
        assert result == pytest.approx(4270.408355224464, rel=1e-4)

    def test_linear_envelope_b_equals_one(self):
        # When b=1.0, s = a * sigma' (linear through origin)
        # a=0.5, b=1.0, sigma'=1000, pa=2116
        # s = 0.5*2116*(1000/2116) = 500.0
        result = power_function_shear_strength(0.5, 1.0, 1000.0, 2116.0)
        assert result == pytest.approx(500.0, rel=1e-4)

    def test_zero_sigma(self):
        # sigma'=0 with b=1.0 => s=0
        result = power_function_shear_strength(0.5, 1.0, 0.0, 2116.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            power_function_shear_strength(0.0, 0.8, 1000.0)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            power_function_shear_strength(-0.5, 0.8, 1000.0)

    def test_b_zero_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_shear_strength(0.8, 0.0, 1000.0)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_shear_strength(0.8, -0.1, 1000.0)

    def test_pa_zero_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_shear_strength(0.8, 0.9, 1000.0, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_shear_strength(0.8, 0.9, 1000.0, -100.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_prime must be non-negative"):
            power_function_shear_strength(0.8, 0.9, -100.0)


# ===========================================================================
# power_function_undrained_strength  (Equation P-2)
#   s_u = a_u * Pa * (sigma_1_con / Pa)^b_u
# ===========================================================================

class TestPowerFunctionUndrainedStrength:

    def test_basic_case(self):
        # a_u=0.3, b_u=0.85, sigma_1_con=4232, pa=2116
        # s_u = 0.3*2116*(4232/2116)^0.85 = 0.3*2116*2^0.85 = 1144.228...
        result = power_function_undrained_strength(0.3, 0.85, 4232.0, 2116.0)
        assert result == pytest.approx(1144.2275873307099, rel=1e-4)

    def test_sigma_equals_pa(self):
        # a_u=0.5, b_u=1.0, sigma_1_con=2116, pa=2116
        # s_u = 0.5*2116*1 = 1058.0
        result = power_function_undrained_strength(0.5, 1.0, 2116.0, 2116.0)
        assert result == pytest.approx(1058.0, rel=1e-4)

    def test_zero_sigma(self):
        # sigma_1_con=0, b_u=1.0 => s_u = 0
        result = power_function_undrained_strength(0.5, 1.0, 0.0, 2116.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_a_u_zero_raises(self):
        with pytest.raises(ValueError, match="a_u must be positive"):
            power_function_undrained_strength(0.0, 0.8, 1000.0)

    def test_a_u_negative_raises(self):
        with pytest.raises(ValueError, match="a_u must be positive"):
            power_function_undrained_strength(-0.3, 0.8, 1000.0)

    def test_b_u_zero_raises(self):
        with pytest.raises(ValueError, match="b_u must be positive"):
            power_function_undrained_strength(0.3, 0.0, 1000.0)

    def test_b_u_negative_raises(self):
        with pytest.raises(ValueError, match="b_u must be positive"):
            power_function_undrained_strength(0.3, -0.1, 1000.0)

    def test_pa_zero_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_undrained_strength(0.3, 0.85, 1000.0, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_undrained_strength(0.3, 0.85, 1000.0, -100.0)

    def test_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_1_con must be non-negative"):
            power_function_undrained_strength(0.3, 0.85, -500.0)


# ===========================================================================
# three_parameter_power_function  (Equation P-3)
#   s = a * Pa * (sigma'/Pa + t)^b
# ===========================================================================

class TestThreeParameterPowerFunction:

    def test_basic_case(self):
        # a=0.8, b=0.9, t=0.1, sigma'=2116, pa=2116
        # s = 0.8*2116*(2116/2116 + 0.1)^0.9 = 0.8*2116*(1.1)^0.9 = 1844.417...
        result = three_parameter_power_function(0.8, 0.9, 0.1, 2116.0, 2116.0)
        assert result == pytest.approx(1844.4167899360968, rel=1e-4)

    def test_t_zero_reduces_to_two_param(self):
        # When t=0, should equal power_function_shear_strength
        # a=1.0, b=0.8, t=0.0, sigma'=1000, pa=2116
        # s = 1.0*2116*(1000/2116)^0.8 = 1161.724...
        result = three_parameter_power_function(1.0, 0.8, 0.0, 1000.0, 2116.0)
        expected = power_function_shear_strength(1.0, 0.8, 1000.0, 2116.0)
        assert result == pytest.approx(expected, rel=1e-4)
        assert result == pytest.approx(1161.7244578318284, rel=1e-4)

    def test_zero_sigma_nonzero_t(self):
        # a=0.5, b=0.5, t=0.2, sigma'=0, pa=2116
        # s = 0.5*2116*(0 + 0.2)^0.5 = 0.5*2116*sqrt(0.2) = 473.152...
        result = three_parameter_power_function(0.5, 0.5, 0.2, 0.0, 2116.0)
        assert result == pytest.approx(473.1519840389555, rel=1e-4)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            three_parameter_power_function(0.0, 0.9, 0.1, 1000.0)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            three_parameter_power_function(-0.5, 0.9, 0.1, 1000.0)

    def test_b_zero_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            three_parameter_power_function(0.8, 0.0, 0.1, 1000.0)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            three_parameter_power_function(0.8, -0.1, 0.1, 1000.0)

    def test_t_negative_raises(self):
        with pytest.raises(ValueError, match="t must be non-negative"):
            three_parameter_power_function(0.8, 0.9, -0.1, 1000.0)

    def test_pa_zero_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            three_parameter_power_function(0.8, 0.9, 0.1, 1000.0, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            three_parameter_power_function(0.8, 0.9, 0.1, 1000.0, -100.0)

    def test_normalized_sum_negative_raises(self):
        # sigma'/Pa + t < 0 when sigma' is very negative relative to pa
        # t=0, sigma'=-500, pa=2116 => (-500/2116 + 0) = -0.2363 < 0
        # But sigma' < 0 won't hit this branch because a check on t < 0
        # would be caught first. Actually, there's no check on sigma_prime
        # being non-negative in three_parameter_power_function, so we can
        # trigger the normalized_sum check.
        with pytest.raises(ValueError, match="sigma_prime / p_a \\+ t.*must be non-negative"):
            three_parameter_power_function(0.8, 0.9, 0.0, -500.0, 2116.0)


# ===========================================================================
# equivalent_friction_angle  (Figure P-4)
#   phi'_EQ = arctan( (s_high - s_low) / (sigma_high - sigma_low) )
# ===========================================================================

class TestEquivalentFrictionAngle:

    def test_basic_case(self):
        # a=0.8, b=0.85, sigma_low=500, sigma_high=2000, pa=2116
        # s_high = 0.8*2116*(2000/2116)^0.85
        # s_low  = 0.8*2116*(500/2116)^0.85
        # slope  = (s_high - s_low)/(2000-500)
        # phi_eq = arctan(slope) = 36.6726 deg
        result = equivalent_friction_angle(0.8, 0.85, 500.0, 2000.0, 2116.0)
        assert result == pytest.approx(36.67256125262194, rel=1e-4)

    def test_linear_envelope_b_one(self):
        # When b=1.0: s = a*sigma', so slope = a, phi_eq = arctan(a)
        # a=0.7, b=1.0, sigma_low=100, sigma_high=1000, pa=2116
        # slope = 0.7, phi_eq = arctan(0.7) = 34.992 deg
        result = equivalent_friction_angle(0.7, 1.0, 100.0, 1000.0, 2116.0)
        expected_phi = math.degrees(math.atan(0.7))
        assert result == pytest.approx(expected_phi, rel=1e-4)

    def test_high_equals_low_raises(self):
        with pytest.raises(ValueError, match="sigma_prime_high must be greater"):
            equivalent_friction_angle(0.8, 0.85, 1000.0, 1000.0)

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError, match="sigma_prime_high must be greater"):
            equivalent_friction_angle(0.8, 0.85, 2000.0, 500.0)


# ===========================================================================
# equivalent_cohesion  (Figure P-4)
#   c'_EQ = s_low - sigma_low * slope
# ===========================================================================

class TestEquivalentCohesion:

    def test_basic_case(self):
        # Same parameters as equivalent_friction_angle basic test
        # c_eq = s_low - 500 * slope = 124.324...
        result = equivalent_cohesion(0.8, 0.85, 500.0, 2000.0, 2116.0)
        assert result == pytest.approx(124.32400132297943, rel=1e-4)

    def test_linear_envelope_b_one_cohesion_zero(self):
        # When b=1.0: s = a*sigma', a linear line through origin
        # c_eq = s_low - sigma_low * a = a*sigma_low - sigma_low*a = 0
        result = equivalent_cohesion(0.7, 1.0, 100.0, 1000.0, 2116.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_high_equals_low_raises(self):
        with pytest.raises(ValueError, match="sigma_prime_high must be greater"):
            equivalent_cohesion(0.8, 0.85, 1000.0, 1000.0)

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError, match="sigma_prime_high must be greater"):
            equivalent_cohesion(0.8, 0.85, 2000.0, 500.0)


# ===========================================================================
# power_function_dimensional_to_normalized  (Figure P-5)
#   a = A * Pa^(b-1)
# ===========================================================================

class TestPowerFunctionDimensionalToNormalized:

    def test_basic_case(self):
        # a_dim=2.5, b=0.8, pa=2116
        # a = 2.5 * 2116^(0.8-1) = 2.5 * 2116^(-0.2) = 0.5406...
        result = power_function_dimensional_to_normalized(2.5, 0.8, 2116.0)
        assert result == pytest.approx(0.5405512500351444, rel=1e-4)

    def test_b_equals_one(self):
        # When b=1.0: a = A * Pa^0 = A
        result = power_function_dimensional_to_normalized(3.0, 1.0, 2116.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_a_dim_zero_raises(self):
        with pytest.raises(ValueError, match="a_dim must be positive"):
            power_function_dimensional_to_normalized(0.0, 0.8)

    def test_a_dim_negative_raises(self):
        with pytest.raises(ValueError, match="a_dim must be positive"):
            power_function_dimensional_to_normalized(-1.0, 0.8)

    def test_b_zero_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_dimensional_to_normalized(2.5, 0.0)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_dimensional_to_normalized(2.5, -0.5)

    def test_pa_zero_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_dimensional_to_normalized(2.5, 0.8, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_dimensional_to_normalized(2.5, 0.8, -100.0)


# ===========================================================================
# power_function_normalized_to_dimensional  (Figure P-5)
#   A = a / Pa^(b-1)
# ===========================================================================

class TestPowerFunctionNormalizedToDimensional:

    def test_basic_case(self):
        # a=0.6, b=0.8, pa=2116
        # A = 0.6 / 2116^(0.8-1) = 0.6 / 2116^(-0.2) = 2.7749...
        result = power_function_normalized_to_dimensional(0.6, 0.8, 2116.0)
        assert result == pytest.approx(2.774945021221348, rel=1e-4)

    def test_b_equals_one(self):
        # When b=1.0: A = a / Pa^0 = a
        result = power_function_normalized_to_dimensional(0.6, 1.0, 2116.0)
        assert result == pytest.approx(0.6, rel=1e-4)

    def test_roundtrip_with_dim_to_norm(self):
        # dim -> norm -> dim should return original value
        a_dim_original = 2.5
        b = 0.8
        a_norm = power_function_dimensional_to_normalized(a_dim_original, b, 2116.0)
        a_dim_back = power_function_normalized_to_dimensional(a_norm, b, 2116.0)
        assert a_dim_back == pytest.approx(a_dim_original, rel=1e-4)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            power_function_normalized_to_dimensional(0.0, 0.8)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            power_function_normalized_to_dimensional(-0.5, 0.8)

    def test_b_zero_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_normalized_to_dimensional(0.6, 0.0)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            power_function_normalized_to_dimensional(0.6, -0.5)

    def test_pa_zero_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_normalized_to_dimensional(0.6, 0.8, 0.0)

    def test_pa_negative_raises(self):
        with pytest.raises(ValueError, match="p_a must be positive"):
            power_function_normalized_to_dimensional(0.6, 0.8, -100.0)


# ===========================================================================
# usr_icu_triaxial  (Table P-6)
#   USR = sin(phi') / (1 - sin(phi'))
# ===========================================================================

class TestUsrIcuTriaxial:

    def test_phi_30(self):
        # sin(30)=0.5, USR = 0.5/(1-0.5) = 1.0
        result = usr_icu_triaxial(30.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_phi_45(self):
        # sin(45) = sqrt(2)/2 = 0.7071...
        # USR = 0.7071/(1-0.7071) = 0.7071/0.2929 = 2.4142...
        result = usr_icu_triaxial(45.0)
        assert result == pytest.approx(2.414213562373096, rel=1e-4)

    def test_small_phi(self):
        # phi=5 deg, sin(5)=0.08716...
        # USR = 0.08716/(1-0.08716) = 0.08716/0.91284 = 0.09549...
        sin5 = math.sin(math.radians(5.0))
        expected = sin5 / (1.0 - sin5)
        result = usr_icu_triaxial(5.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_phi_zero_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_icu_triaxial(0.0)

    def test_phi_negative_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_icu_triaxial(-10.0)

    def test_phi_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_icu_triaxial(90.0)

    def test_phi_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_icu_triaxial(100.0)


# ===========================================================================
# usr_acu_triaxial  (Table P-6)
#   USR = sin(phi')
# ===========================================================================

class TestUsrAcuTriaxial:

    def test_phi_30(self):
        # sin(30) = 0.5
        result = usr_acu_triaxial(30.0)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_phi_45(self):
        # sin(45) = sqrt(2)/2 = 0.7071...
        result = usr_acu_triaxial(45.0)
        assert result == pytest.approx(0.7071067811865476, rel=1e-4)

    def test_phi_near_90(self):
        # sin(89) = 0.99985...
        result = usr_acu_triaxial(89.0)
        assert result == pytest.approx(math.sin(math.radians(89.0)), rel=1e-4)

    def test_phi_zero_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_acu_triaxial(0.0)

    def test_phi_negative_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_acu_triaxial(-5.0)

    def test_phi_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_acu_triaxial(90.0)

    def test_phi_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_acu_triaxial(91.0)


# ===========================================================================
# usr_dss  (Table P-6)
#   USR = sin(phi') - 0.5 * sin(2*phi')
# ===========================================================================

class TestUsrDss:

    def test_phi_30(self):
        # sin(30) - 0.5*sin(60) = 0.5 - 0.5*0.8660 = 0.5 - 0.4330 = 0.06699
        result = usr_dss(30.0)
        assert result == pytest.approx(0.06698729810778065, rel=1e-4)

    def test_phi_45(self):
        # sin(45) - 0.5*sin(90) = 0.7071 - 0.5*1.0 = 0.2071
        result = usr_dss(45.0)
        assert result == pytest.approx(0.20710678118654757, rel=1e-4)

    def test_phi_small(self):
        # phi=10 deg
        # sin(10) - 0.5*sin(20)
        phi_rad = math.radians(10.0)
        expected = math.sin(phi_rad) - 0.5 * math.sin(2.0 * phi_rad)
        result = usr_dss(10.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_phi_zero_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_dss(0.0)

    def test_phi_negative_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_dss(-15.0)

    def test_phi_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_dss(90.0)

    def test_phi_above_90_raises(self):
        with pytest.raises(ValueError, match="phi_prime_deg must be in the range"):
            usr_dss(120.0)
