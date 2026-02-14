"""Tests for geotech.dm7_1.chapter4 -- Distribution of Stresses.

Covers Equations 4-1 through 4-12 and the Boussinesq Table 4-2 functions.
Expected values are hand-computed from the equations in the source module.
"""

import math

import pytest

from geotech.dm7_1.chapter4 import *


# ===================================================================
# total_vertical_stress  (Equation 4-1)
# sigma_v = sum(z_i * gamma_t_i)
# ===================================================================

class TestTotalVerticalStress:

    def test_single_layer(self):
        # 10 ft of soil at 120 pcf => 10 * 120 = 1200 psf
        result = total_vertical_stress([(10.0, 120.0)])
        assert result == pytest.approx(1200.0, rel=1e-4)

    def test_two_layers(self):
        # Layer 1: 5 ft * 110 pcf = 550
        # Layer 2: 10 ft * 125 pcf = 1250
        # Total = 1800 psf
        result = total_vertical_stress([(5.0, 110.0), (10.0, 125.0)])
        assert result == pytest.approx(1800.0, rel=1e-4)

    def test_three_layers(self):
        # 3 * 62.4 + 5 * 110 + 8 * 130 = 187.2 + 550 + 1040 = 1777.2
        layers = [(3.0, 62.4), (5.0, 110.0), (8.0, 130.0)]
        result = total_vertical_stress(layers)
        assert result == pytest.approx(1777.2, rel=1e-4)

    def test_empty_layers_returns_zero(self):
        result = total_vertical_stress([])
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_thickness_layer(self):
        # A layer with zero thickness contributes nothing
        result = total_vertical_stress([(0.0, 120.0), (10.0, 100.0)])
        assert result == pytest.approx(1000.0, rel=1e-4)

    def test_zero_unit_weight_layer(self):
        result = total_vertical_stress([(5.0, 0.0)])
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="Layer thickness must be non-negative"):
            total_vertical_stress([(-1.0, 120.0)])

    def test_negative_unit_weight_raises(self):
        with pytest.raises(ValueError, match="Unit weight must be non-negative"):
            total_vertical_stress([(5.0, -120.0)])


# ===================================================================
# pore_water_pressure  (Equation 4-2)
# u = h_p * gamma_w
# ===================================================================

class TestPoreWaterPressure:

    def test_basic(self):
        # 10 ft depth below GWT, gamma_w = 62.4 pcf => u = 624.0 psf
        result = pore_water_pressure(10.0, 62.4)
        assert result == pytest.approx(624.0, rel=1e-4)

    def test_metric_units(self):
        # 5 m depth, gamma_w = 9.81 kN/m^3 => u = 49.05 kPa
        result = pore_water_pressure(5.0, 9.81)
        assert result == pytest.approx(49.05, rel=1e-4)

    def test_zero_head(self):
        # At the GWT, h_p = 0, so u = 0
        result = pore_water_pressure(0.0, 62.4)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_head(self):
        # Negative pressure head (capillary zone) is allowed
        result = pore_water_pressure(-3.0, 62.4)
        assert result == pytest.approx(-187.2, rel=1e-4)

    def test_zero_gamma_w_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            pore_water_pressure(10.0, 0.0)

    def test_negative_gamma_w_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            pore_water_pressure(10.0, -9.81)


# ===================================================================
# effective_vertical_stress  (Equation 4-3)
# sigma'_v = sigma_v - u
# ===================================================================

class TestEffectiveVerticalStress:

    def test_basic(self):
        # sigma_v = 1800, u = 624 => sigma'_v = 1176
        result = effective_vertical_stress(1800.0, 624.0)
        assert result == pytest.approx(1176.0, rel=1e-4)

    def test_zero_pore_pressure(self):
        result = effective_vertical_stress(500.0, 0.0)
        assert result == pytest.approx(500.0, rel=1e-4)

    def test_negative_result_allowed(self):
        # Artesian conditions: u > sigma_v
        result = effective_vertical_stress(100.0, 300.0)
        assert result == pytest.approx(-200.0, rel=1e-4)


# ===================================================================
# effective_horizontal_stress  (Equation 4-4)
# sigma'_h = K * sigma'_v
# ===================================================================

class TestEffectiveHorizontalStress:

    def test_basic_k0(self):
        # K0 = 0.5, sigma'_v = 1000 => sigma'_h = 500
        result = effective_horizontal_stress(0.5, 1000.0)
        assert result == pytest.approx(500.0, rel=1e-4)

    def test_k_equals_one(self):
        result = effective_horizontal_stress(1.0, 750.0)
        assert result == pytest.approx(750.0, rel=1e-4)

    def test_k_zero(self):
        result = effective_horizontal_stress(0.0, 1000.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError, match="K must be non-negative"):
            effective_horizontal_stress(-0.1, 500.0)


# ===================================================================
# total_horizontal_stress  (Equation 4-5)
# sigma_h = sigma'_h + u
# ===================================================================

class TestTotalHorizontalStress:

    def test_basic(self):
        # sigma'_h = 500, u = 624 => sigma_h = 1124
        result = total_horizontal_stress(500.0, 624.0)
        assert result == pytest.approx(1124.0, rel=1e-4)

    def test_zero_pore_pressure(self):
        result = total_horizontal_stress(400.0, 0.0)
        assert result == pytest.approx(400.0, rel=1e-4)

    def test_negative_effective_stress(self):
        result = total_horizontal_stress(-50.0, 200.0)
        assert result == pytest.approx(150.0, rel=1e-4)


# ===================================================================
# boussinesq_point_load  (Table 4-2)
# delta_sigma_z = 3*Q*z^3 / (2*pi*R^5),  R = sqrt(x^2 + y^2 + z^2)
# ===================================================================

class TestBoussinesqPointLoad:

    def test_directly_below_load(self):
        # Q=1000, x=0, y=0, z=10
        # R = 10, delta_sigma = 3*1000*1000 / (2*pi*100000) = 3000000/628318.53...
        # = 4.77465 psf
        Q, x, y, z = 1000.0, 0.0, 0.0, 10.0
        R = math.sqrt(x**2 + y**2 + z**2)  # 10
        expected = (3.0 * Q * z**3) / (2.0 * math.pi * R**5)
        result = boussinesq_point_load(Q, x, y, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_offset_point(self):
        # Q=500, x=3, y=4, z=5
        # R = sqrt(9+16+25) = sqrt(50) = 7.07107
        # delta_sigma = 3*500*125 / (2*pi*(sqrt(50))^5)
        # (sqrt(50))^5 = 50^2.5 = 17677.6695...
        # = 187500 / (2*pi*17677.6695) = 187500 / 111072.07 = 1.68824
        Q, x, y, z = 500.0, 3.0, 4.0, 5.0
        R = math.sqrt(9 + 16 + 25)
        expected = (3.0 * Q * z**3) / (2.0 * math.pi * R**5)
        result = boussinesq_point_load(Q, x, y, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_z_zero_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_point_load(1000.0, 0.0, 0.0, 0.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_point_load(1000.0, 0.0, 0.0, -5.0)

    def test_large_offset_small_stress(self):
        # When x, y are large relative to z, stress should be very small
        result = boussinesq_point_load(1000.0, 100.0, 100.0, 1.0)
        assert result < 0.001


# ===================================================================
# boussinesq_line_load  (Table 4-2)
# delta_sigma_z = 2*P*z^3 / (pi*R^4),  R = sqrt(x^2 + z^2)
# ===================================================================

class TestBoussinesqLineLoad:

    def test_directly_below(self):
        # P=500, x=0, z=10
        # R = 10, delta_sigma = 2*500*1000 / (pi*10000) = 1000000/31415.93 = 31.831
        P, x, z = 500.0, 0.0, 10.0
        R = math.sqrt(x**2 + z**2)
        expected = (2.0 * P * z**3) / (math.pi * R**4)
        result = boussinesq_line_load(P, x, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_offset_horizontally(self):
        # P=1000, x=5, z=5
        # R = sqrt(25+25) = sqrt(50) = 7.07107
        # R^4 = 2500, delta_sigma = 2*1000*125 / (pi*2500) = 250000/7853.98 = 31.831
        P, x, z = 1000.0, 5.0, 5.0
        R = math.sqrt(x**2 + z**2)
        expected = (2.0 * P * z**3) / (math.pi * R**4)
        result = boussinesq_line_load(P, x, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_z_zero_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_line_load(500.0, 5.0, 0.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_line_load(500.0, 5.0, -3.0)


# ===================================================================
# boussinesq_strip_load  (Table 4-2)
# delta_sigma_z = (q0/pi) * [alpha + sin(alpha)*cos(alpha + 2*gamma)]
# ===================================================================

class TestBoussinesqStripLoad:

    def test_basic(self):
        # q0=1000, alpha=pi/4, gamma=pi/6
        # sin(pi/4) = 0.707107, cos(pi/4 + pi/3) = cos(7*pi/12) = cos(105 deg)
        # cos(105 deg) = -0.258819
        # delta_sigma = (1000/pi)*(pi/4 + 0.707107*(-0.258819))
        # = (1000/pi)*(0.785398 + (-0.183013))
        # = (1000/pi)*(0.602386) = 318.31 * 0.602386 = 191.72
        q0, alpha, gamma_angle = 1000.0, math.pi / 4, math.pi / 6
        expected = (q0 / math.pi) * (
            alpha + math.sin(alpha) * math.cos(alpha + 2.0 * gamma_angle)
        )
        result = boussinesq_strip_load(q0, alpha, gamma_angle)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_alpha_zero(self):
        # alpha = 0 => sin(0) = 0, so result = 0
        result = boussinesq_strip_load(1000.0, 0.0, math.pi / 4)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_gamma_zero(self):
        # gamma = 0 => delta_sigma = (q0/pi)*(alpha + sin(alpha)*cos(alpha))
        q0, alpha = 500.0, math.pi / 3
        expected = (q0 / math.pi) * (alpha + math.sin(alpha) * math.cos(alpha))
        result = boussinesq_strip_load(q0, alpha, 0.0)
        assert result == pytest.approx(expected, rel=1e-4)


# ===================================================================
# boussinesq_rectangular_load  (Table 4-2)
# delta_sigma_z = (q0/(2*pi)) * [atan(xy/(z*R3)) + xyz/R3 * (1/R1^2 + 1/R2^2)]
# R1=sqrt(y^2+z^2), R2=sqrt(x^2+z^2), R3=sqrt(x^2+y^2+z^2)
# ===================================================================

class TestBoussinesqRectangularLoad:

    def test_square_load(self):
        # q0=100, x=5, y=5, z=5
        q0, x, y, z = 100.0, 5.0, 5.0, 5.0
        R1 = math.sqrt(y**2 + z**2)   # sqrt(50)
        R2 = math.sqrt(x**2 + z**2)   # sqrt(50)
        R3 = math.sqrt(x**2 + y**2 + z**2)  # sqrt(75)
        term1 = math.atan2(x * y, z * R3)
        term2 = (x * y * z / R3) * (1.0 / R1**2 + 1.0 / R2**2)
        expected = (q0 / (2.0 * math.pi)) * (term1 + term2)
        result = boussinesq_rectangular_load(q0, x, y, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_deep_point(self):
        # Large z relative to x,y => stress approaches zero
        result = boussinesq_rectangular_load(100.0, 1.0, 1.0, 1000.0)
        assert result < 0.01

    def test_x_zero_raises(self):
        with pytest.raises(ValueError, match="x must be positive"):
            boussinesq_rectangular_load(100.0, 0.0, 5.0, 5.0)

    def test_x_negative_raises(self):
        with pytest.raises(ValueError, match="x must be positive"):
            boussinesq_rectangular_load(100.0, -1.0, 5.0, 5.0)

    def test_y_zero_raises(self):
        with pytest.raises(ValueError, match="y must be positive"):
            boussinesq_rectangular_load(100.0, 5.0, 0.0, 5.0)

    def test_y_negative_raises(self):
        with pytest.raises(ValueError, match="y must be positive"):
            boussinesq_rectangular_load(100.0, 5.0, -1.0, 5.0)

    def test_z_zero_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_rectangular_load(100.0, 5.0, 5.0, 0.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_rectangular_load(100.0, 5.0, 5.0, -1.0)

    def test_asymmetric_rectangle(self):
        q0, x, y, z = 200.0, 3.0, 8.0, 4.0
        R1 = math.sqrt(64 + 16)   # sqrt(80)
        R2 = math.sqrt(9 + 16)    # sqrt(25) = 5
        R3 = math.sqrt(9 + 64 + 16)  # sqrt(89)
        term1 = math.atan2(x * y, z * R3)
        term2 = (x * y * z / R3) * (1.0 / R1**2 + 1.0 / R2**2)
        expected = (q0 / (2.0 * math.pi)) * (term1 + term2)
        result = boussinesq_rectangular_load(q0, x, y, z)
        assert result == pytest.approx(expected, rel=1e-4)


# ===================================================================
# boussinesq_circular_load  (Table 4-2)
# delta_sigma_z = q0 * [1 - 1 / (1 + (r/z)^2)^1.5]
# ===================================================================

class TestBoussinesqCircularLoad:

    def test_basic(self):
        # q0=1000, r=5, z=5 => r/z = 1
        # delta_sigma = 1000*(1 - 1/(1+1)^1.5) = 1000*(1 - 1/2.8284) = 1000*(1 - 0.35355)
        # = 646.45
        q0, r, z = 1000.0, 5.0, 5.0
        expected = q0 * (1.0 - 1.0 / (1.0 + (r / z)**2)**1.5)
        result = boussinesq_circular_load(q0, r, z)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_deep_point_small_stress(self):
        # r/z very small => stress approaches zero
        # r=1, z=100 => r/z=0.01
        # 1/(1+0.0001)^1.5 ~ 1 - 0.00015 => delta ~ q0*0.00015
        result = boussinesq_circular_load(1000.0, 1.0, 100.0)
        expected = 1000.0 * (1.0 - 1.0 / (1.0 + 0.01**2)**1.5)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_shallow_point_near_q0(self):
        # r/z very large => stress approaches q0
        # r=100, z=1 => r/z=100
        result = boussinesq_circular_load(500.0, 100.0, 1.0)
        assert result == pytest.approx(500.0, rel=1e-2)

    def test_r_zero_raises(self):
        with pytest.raises(ValueError, match="r must be positive"):
            boussinesq_circular_load(1000.0, 0.0, 5.0)

    def test_r_negative_raises(self):
        with pytest.raises(ValueError, match="r must be positive"):
            boussinesq_circular_load(1000.0, -1.0, 5.0)

    def test_z_zero_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_circular_load(1000.0, 5.0, 0.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            boussinesq_circular_load(1000.0, 5.0, -5.0)


# ===================================================================
# boussinesq_triangular_load  (Table 4-2)
# delta_sigma_z = (q0/pi) * [x*alpha/a + (a+b-x)/b * beta]
# ===================================================================

class TestBoussinesqTriangularLoad:

    def test_basic(self):
        # q0=500, x=3, a=4, b=6, alpha=pi/6, beta=pi/4
        # term1 = 3*(pi/6)/4 = 3*0.52360/4 = 0.39270
        # term2 = (4+6-3)/6 * (pi/4) = (7/6)*0.78540 = 0.91630
        # delta_sigma = (500/pi)*(0.39270 + 0.91630) = (500/pi)*1.30900
        # = 159.155 * 1.30900 = 208.37
        q0, x, a, b = 500.0, 3.0, 4.0, 6.0
        alpha, beta = math.pi / 6, math.pi / 4
        expected = (q0 / math.pi) * (x * alpha / a + (a + b - x) / b * beta)
        result = boussinesq_triangular_load(q0, x, a, b, alpha, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_x_zero(self):
        # x=0 => first term vanishes
        q0, a, b = 1000.0, 5.0, 10.0
        alpha, beta = math.pi / 3, math.pi / 6
        expected = (q0 / math.pi) * ((a + b) / b * beta)
        result = boussinesq_triangular_load(q0, 0.0, a, b, alpha, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_triangular_load(500.0, 3.0, 0.0, 6.0, 0.5, 0.5)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_triangular_load(500.0, 3.0, -1.0, 6.0, 0.5, 0.5)

    def test_b_zero_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            boussinesq_triangular_load(500.0, 3.0, 4.0, 0.0, 0.5, 0.5)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be positive"):
            boussinesq_triangular_load(500.0, 3.0, 4.0, -2.0, 0.5, 0.5)


# ===================================================================
# boussinesq_slope_load  (Table 4-2)
# delta_sigma_z = (q0/(pi*a)) * (x*beta + z)
# ===================================================================

class TestBoussinesqSlopeLoad:

    def test_basic(self):
        # q0=1000, x=5, z=10, a=8, beta=pi/4
        # delta_sigma = (1000/(pi*8))*(5*pi/4 + 10)
        # = (1000/25.1327)*(3.9270 + 10)
        # = 39.789 * 13.927 = 554.18
        q0, x, z, a, beta = 1000.0, 5.0, 10.0, 8.0, math.pi / 4
        expected = (q0 / (math.pi * a)) * (x * beta + z)
        result = boussinesq_slope_load(q0, x, z, a, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_x_zero(self):
        # x=0 => delta_sigma = (q0/(pi*a)) * z
        q0, z, a, beta = 500.0, 8.0, 4.0, math.pi / 3
        expected = (q0 / (math.pi * a)) * z
        result = boussinesq_slope_load(q0, 0.0, z, a, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_z_zero(self):
        q0, x, a, beta = 500.0, 3.0, 4.0, math.pi / 6
        expected = (q0 / (math.pi * a)) * (x * beta)
        result = boussinesq_slope_load(q0, x, 0.0, a, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_slope_load(1000.0, 5.0, 10.0, 0.0, 0.5)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_slope_load(1000.0, 5.0, 10.0, -2.0, 0.5)


# ===================================================================
# boussinesq_terrace_load  (Table 4-2)
# delta_sigma_z = (q0/(pi*a)) * (a*beta + x*alpha)
# ===================================================================

class TestBoussinesqTerraceLoad:

    def test_basic(self):
        # q0=800, x=6, a=10, alpha=pi/4, beta=pi/6
        # delta_sigma = (800/(pi*10))*(10*pi/6 + 6*pi/4)
        # = (800/31.4159)*(5.23599 + 4.71239)
        # = 25.4648 * 9.94838 = 25.4648*9.94838 = 253.32
        q0, x, a = 800.0, 6.0, 10.0
        alpha, beta = math.pi / 4, math.pi / 6
        expected = (q0 / (math.pi * a)) * (a * beta + x * alpha)
        result = boussinesq_terrace_load(q0, x, a, alpha, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_x_zero(self):
        # x=0 => delta_sigma = (q0/pi)*beta
        q0, a, alpha, beta = 500.0, 5.0, math.pi / 3, math.pi / 4
        expected = (q0 / (math.pi * a)) * (a * beta)
        result = boussinesq_terrace_load(q0, 0.0, a, alpha, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_a_zero_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_terrace_load(800.0, 6.0, 0.0, 0.5, 0.5)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be positive"):
            boussinesq_terrace_load(800.0, 6.0, -3.0, 0.5, 0.5)


# ===================================================================
# boussinesq_semi_infinite_load  (Table 4-2)
# delta_sigma_z = (q0/pi) * (beta + x*z/R^2)
# ===================================================================

class TestBoussinesqSemiInfiniteLoad:

    def test_basic(self):
        # q0=1000, x=3, z=4, R=5, beta=pi/4
        # delta_sigma = (1000/pi)*(pi/4 + 3*4/25)
        # = 318.310*(0.78540 + 0.48) = 318.310*1.26540 = 402.83
        q0, x, z, R, beta = 1000.0, 3.0, 4.0, 5.0, math.pi / 4
        expected = (q0 / math.pi) * (beta + (x * z) / R**2)
        result = boussinesq_semi_infinite_load(q0, x, z, R, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_x_zero(self):
        # x=0 => delta_sigma = (q0/pi)*beta
        q0, z, R, beta = 500.0, 10.0, 10.0, math.pi / 6
        expected = (q0 / math.pi) * beta
        result = boussinesq_semi_infinite_load(q0, 0.0, z, R, beta)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_R_zero_raises(self):
        with pytest.raises(ValueError, match="R must be positive"):
            boussinesq_semi_infinite_load(1000.0, 3.0, 4.0, 0.0, 0.5)

    def test_R_negative_raises(self):
        with pytest.raises(ValueError, match="R must be positive"):
            boussinesq_semi_infinite_load(1000.0, 3.0, 4.0, -5.0, 0.5)


# ===================================================================
# rigid_pipe_trench_load  (Equation 4-6)
# W_d = C_d * gamma_t * B_d^2
# ===================================================================

class TestRigidPipeTrenchLoad:

    def test_basic(self):
        # C_d=1.5, gamma_t=120, B_d=3 => W_d = 1.5*120*9 = 1620 lb/ft
        result = rigid_pipe_trench_load(1.5, 120.0, 3.0)
        assert result == pytest.approx(1620.0, rel=1e-4)

    def test_unit_values(self):
        result = rigid_pipe_trench_load(1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_gamma_t_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            rigid_pipe_trench_load(1.5, 0.0, 3.0)

    def test_gamma_t_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            rigid_pipe_trench_load(1.5, -120.0, 3.0)

    def test_B_d_zero_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            rigid_pipe_trench_load(1.5, 120.0, 0.0)

    def test_B_d_negative_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            rigid_pipe_trench_load(1.5, 120.0, -3.0)


# ===================================================================
# trench_load_coefficient  (Equation 4-7)
# C_d = (1 - exp(-2*K*mu'*H/B_d)) / (2*K*mu')
# ===================================================================

class TestTrenchLoadCoefficient:

    def test_basic(self):
        # H=10, B_d=3, K=0.33, mu'=0.55
        # exponent = -2*0.33*0.55*10/3 = -2*0.1815*10/3 = -3.63/3 = -1.21
        # C_d = (1 - exp(-1.21)) / (2*0.33*0.55) = (1 - 0.29820) / 0.3630
        # = 0.70180 / 0.3630 = 1.9333
        H, B_d, K, mu_prime = 10.0, 3.0, 0.33, 0.55
        exponent = -2.0 * K * mu_prime * H / B_d
        expected = (1.0 - math.exp(exponent)) / (2.0 * K * mu_prime)
        result = trench_load_coefficient(H, B_d, K, mu_prime)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_shallow_trench(self):
        # Very small H/B_d => C_d approaches H/B_d (limit of exponential)
        H, B_d, K, mu_prime = 0.01, 10.0, 0.5, 0.5
        # exponent ~ 0 => 1 - exp(~0) ~ 2*K*mu'*H/B_d
        # C_d ~ H/B_d = 0.001
        result = trench_load_coefficient(H, B_d, K, mu_prime)
        assert result == pytest.approx(H / B_d, rel=1e-2)

    def test_H_zero_raises(self):
        with pytest.raises(ValueError, match="H must be positive"):
            trench_load_coefficient(0.0, 3.0, 0.33, 0.55)

    def test_H_negative_raises(self):
        with pytest.raises(ValueError, match="H must be positive"):
            trench_load_coefficient(-1.0, 3.0, 0.33, 0.55)

    def test_B_d_zero_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            trench_load_coefficient(10.0, 0.0, 0.33, 0.55)

    def test_B_d_negative_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            trench_load_coefficient(10.0, -3.0, 0.33, 0.55)

    def test_K_zero_raises(self):
        with pytest.raises(ValueError, match="K must be positive"):
            trench_load_coefficient(10.0, 3.0, 0.0, 0.55)

    def test_K_negative_raises(self):
        with pytest.raises(ValueError, match="K must be positive"):
            trench_load_coefficient(10.0, 3.0, -0.33, 0.55)

    def test_mu_prime_zero_raises(self):
        with pytest.raises(ValueError, match="mu_prime must be positive"):
            trench_load_coefficient(10.0, 3.0, 0.33, 0.0)

    def test_mu_prime_negative_raises(self):
        with pytest.raises(ValueError, match="mu_prime must be positive"):
            trench_load_coefficient(10.0, 3.0, 0.33, -0.55)


# ===================================================================
# flexible_pipe_trench_load  (Equation 4-8)
# W_c = C_d * gamma_t * B_d * D
# ===================================================================

class TestFlexiblePipeTrenchLoad:

    def test_basic(self):
        # C_d=1.5, gamma_t=120, B_d=3, D=2 => W_c = 1.5*120*3*2 = 1080 lb/ft
        result = flexible_pipe_trench_load(1.5, 120.0, 3.0, 2.0)
        assert result == pytest.approx(1080.0, rel=1e-4)

    def test_unit_values(self):
        result = flexible_pipe_trench_load(1.0, 1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_gamma_t_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            flexible_pipe_trench_load(1.5, 0.0, 3.0, 2.0)

    def test_gamma_t_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            flexible_pipe_trench_load(1.5, -120.0, 3.0, 2.0)

    def test_B_d_zero_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            flexible_pipe_trench_load(1.5, 120.0, 0.0, 2.0)

    def test_B_d_negative_raises(self):
        with pytest.raises(ValueError, match="B_d must be positive"):
            flexible_pipe_trench_load(1.5, 120.0, -3.0, 2.0)

    def test_D_zero_raises(self):
        with pytest.raises(ValueError, match="D must be positive"):
            flexible_pipe_trench_load(1.5, 120.0, 3.0, 0.0)

    def test_D_negative_raises(self):
        with pytest.raises(ValueError, match="D must be positive"):
            flexible_pipe_trench_load(1.5, 120.0, 3.0, -2.0)


# ===================================================================
# flexible_pipe_prism_load  (Equation 4-9)
# W_p = gamma_t * H * D
# ===================================================================

class TestFlexiblePipePrismLoad:

    def test_basic(self):
        # gamma_t=120, H=10, D=2 => W_p = 120*10*2 = 2400 lb/ft
        result = flexible_pipe_prism_load(120.0, 10.0, 2.0)
        assert result == pytest.approx(2400.0, rel=1e-4)

    def test_metric(self):
        # gamma_t=18.0 kN/m^3, H=3 m, D=0.5 m => 18*3*0.5 = 27 kN/m
        result = flexible_pipe_prism_load(18.0, 3.0, 0.5)
        assert result == pytest.approx(27.0, rel=1e-4)

    def test_gamma_t_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            flexible_pipe_prism_load(0.0, 10.0, 2.0)

    def test_gamma_t_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            flexible_pipe_prism_load(-120.0, 10.0, 2.0)

    def test_H_zero_raises(self):
        with pytest.raises(ValueError, match="H must be positive"):
            flexible_pipe_prism_load(120.0, 0.0, 2.0)

    def test_H_negative_raises(self):
        with pytest.raises(ValueError, match="H must be positive"):
            flexible_pipe_prism_load(120.0, -10.0, 2.0)

    def test_D_zero_raises(self):
        with pytest.raises(ValueError, match="D must be positive"):
            flexible_pipe_prism_load(120.0, 10.0, 0.0)

    def test_D_negative_raises(self):
        with pytest.raises(ValueError, match="D must be positive"):
            flexible_pipe_prism_load(120.0, 10.0, -2.0)


# ===================================================================
# undrained_stability_factor  (Equation 4-10)
# N_crit = (sigma'_v - sigma_t) / s_u
# ===================================================================

class TestUndrainedStabilityFactor:

    def test_basic(self):
        # sigma'_v=2000, sigma_t=500, s_u=300
        # N_crit = (2000-500)/300 = 1500/300 = 5.0
        result = undrained_stability_factor(2000.0, 500.0, 300.0)
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_no_internal_pressure(self):
        # sigma_t = 0
        result = undrained_stability_factor(1500.0, 0.0, 250.0)
        assert result == pytest.approx(6.0, rel=1e-4)

    def test_negative_result(self):
        # sigma_t > sigma'_v => negative stability factor (very stable)
        result = undrained_stability_factor(500.0, 800.0, 100.0)
        assert result == pytest.approx(-3.0, rel=1e-4)

    def test_s_u_zero_raises(self):
        with pytest.raises(ValueError, match="s_u must be positive"):
            undrained_stability_factor(1000.0, 0.0, 0.0)

    def test_s_u_negative_raises(self):
        with pytest.raises(ValueError, match="s_u must be positive"):
            undrained_stability_factor(1000.0, 0.0, -100.0)


# ===================================================================
# shaft_critical_depth_clay  (Equation 4-11)
# z_crit = 2*s_u / gamma_t
# ===================================================================

class TestShaftCriticalDepthClay:

    def test_basic(self):
        # s_u=500 psf, gamma_t=120 pcf => z_crit = 1000/120 = 8.3333 ft
        result = shaft_critical_depth_clay(500.0, 120.0)
        assert result == pytest.approx(1000.0 / 120.0, rel=1e-4)

    def test_metric(self):
        # s_u=25 kPa, gamma_t=18 kN/m^3 => z_crit = 50/18 = 2.7778 m
        result = shaft_critical_depth_clay(25.0, 18.0)
        assert result == pytest.approx(50.0 / 18.0, rel=1e-4)

    def test_s_u_zero(self):
        # s_u=0 => z_crit = 0 (no unsupported depth allowed)
        result = shaft_critical_depth_clay(0.0, 120.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_s_u_negative_raises(self):
        with pytest.raises(ValueError, match="s_u must be non-negative"):
            shaft_critical_depth_clay(-100.0, 120.0)

    def test_gamma_t_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            shaft_critical_depth_clay(500.0, 0.0)

    def test_gamma_t_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            shaft_critical_depth_clay(500.0, -120.0)


# ===================================================================
# shaft_horizontal_pressure_clay  (Equation 4-12)
# sigma_h = gamma' * z - s_u
# ===================================================================

class TestShaftHorizontalPressureClay:

    def test_basic(self):
        # gamma_eff=60, z=20, s_u=200 => sigma_h = 1200 - 200 = 1000 psf
        result = shaft_horizontal_pressure_clay(60.0, 20.0, 200.0)
        assert result == pytest.approx(1000.0, rel=1e-4)

    def test_negative_result(self):
        # When s_u > gamma'*z, result is negative (no net pressure)
        # gamma_eff=50, z=2, s_u=500 => sigma_h = 100 - 500 = -400
        result = shaft_horizontal_pressure_clay(50.0, 2.0, 500.0)
        assert result == pytest.approx(-400.0, rel=1e-4)

    def test_s_u_zero(self):
        # s_u=0 => sigma_h = gamma'*z
        result = shaft_horizontal_pressure_clay(60.0, 15.0, 0.0)
        assert result == pytest.approx(900.0, rel=1e-4)

    def test_gamma_eff_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_eff must be positive"):
            shaft_horizontal_pressure_clay(0.0, 20.0, 200.0)

    def test_gamma_eff_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_eff must be positive"):
            shaft_horizontal_pressure_clay(-60.0, 20.0, 200.0)

    def test_z_zero_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            shaft_horizontal_pressure_clay(60.0, 0.0, 200.0)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be positive"):
            shaft_horizontal_pressure_clay(60.0, -20.0, 200.0)
