"""Comprehensive tests for geotech.dm7_1.chapter6 -- Seepage and Drainage.

Tests cover all 15 public functions (Equations 6-1 through 6-14 and 6-16).
Each function is tested for basic valid inputs with hand-calculated expected
values, edge cases, and all ValueError validation branches.
"""

import math

import pytest

from geotech.dm7_1.chapter6 import *


# ============================================================================
# total_hydraulic_head  (Equation 6-1)
#   h_t = u / gamma_w + z
# ============================================================================

class TestTotalHydraulicHead:

    def test_basic_valid(self):
        # u=624, gamma_w=62.4, z=5  =>  624/62.4 + 5 = 10 + 5 = 15
        result = total_hydraulic_head(624.0, 62.4, 5.0)
        assert result == pytest.approx(15.0, rel=1e-4)

    def test_zero_pressure(self):
        # u=0, gamma_w=9.81, z=3  =>  0/9.81 + 3 = 3
        result = total_hydraulic_head(0.0, 9.81, 3.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_negative_elevation(self):
        # u=62.4, gamma_w=62.4, z=-2  =>  1 + (-2) = -1
        result = total_hydraulic_head(62.4, 62.4, -2.0)
        assert result == pytest.approx(-1.0, rel=1e-4)

    def test_negative_pressure(self):
        # u=-312, gamma_w=62.4, z=10  =>  -5 + 10 = 5
        result = total_hydraulic_head(-312.0, 62.4, 10.0)
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_gamma_w_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            total_hydraulic_head(100.0, 0.0, 5.0)

    def test_gamma_w_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            total_hydraulic_head(100.0, -9.81, 5.0)


# ============================================================================
# hydraulic_gradient  (Equation 6-2)
#   i = h_L / L
# ============================================================================

class TestHydraulicGradient:

    def test_basic_valid(self):
        # h_L=10, L=50  =>  10/50 = 0.2
        result = hydraulic_gradient(10.0, 50.0)
        assert result == pytest.approx(0.2, rel=1e-4)

    def test_zero_head_loss(self):
        # h_L=0, L=10  =>  0/10 = 0
        result = hydraulic_gradient(0.0, 10.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_head_loss(self):
        # h_L=-5, L=25  =>  -5/25 = -0.2
        result = hydraulic_gradient(-5.0, 25.0)
        assert result == pytest.approx(-0.2, rel=1e-4)

    def test_L_zero_raises(self):
        with pytest.raises(ValueError, match="L must be positive"):
            hydraulic_gradient(10.0, 0.0)

    def test_L_negative_raises(self):
        with pytest.raises(ValueError, match="L must be positive"):
            hydraulic_gradient(10.0, -5.0)


# ============================================================================
# darcy_flow_rate  (Equation 6-3)
#   q = k * i * A
# ============================================================================

class TestDarcyFlowRate:

    def test_basic_valid(self):
        # k=0.01, i=0.5, A=10  =>  0.01*0.5*10 = 0.05
        result = darcy_flow_rate(0.01, 0.5, 10.0)
        assert result == pytest.approx(0.05, rel=1e-4)

    def test_zero_k(self):
        # k=0 => q=0
        result = darcy_flow_rate(0.0, 1.0, 10.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_area(self):
        # A=0 => q=0
        result = darcy_flow_rate(0.01, 1.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_gradient(self):
        # Negative i is allowed; k=0.01, i=-0.5, A=10 => -0.05
        result = darcy_flow_rate(0.01, -0.5, 10.0)
        assert result == pytest.approx(-0.05, rel=1e-4)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be non-negative"):
            darcy_flow_rate(-0.01, 0.5, 10.0)

    def test_A_negative_raises(self):
        with pytest.raises(ValueError, match="A must be non-negative"):
            darcy_flow_rate(0.01, 0.5, -10.0)


# ============================================================================
# darcy_flow_rate_per_unit_width  (Equation 6-4)
#   q = k * i * y
# ============================================================================

class TestDarcyFlowRatePerUnitWidth:

    def test_basic_valid(self):
        # k=0.02, i=0.3, y=5  =>  0.02*0.3*5 = 0.03
        result = darcy_flow_rate_per_unit_width(0.02, 0.3, 5.0)
        assert result == pytest.approx(0.03, rel=1e-4)

    def test_zero_k(self):
        result = darcy_flow_rate_per_unit_width(0.0, 1.0, 5.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_y(self):
        result = darcy_flow_rate_per_unit_width(0.01, 1.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_gradient_allowed(self):
        # k=0.01, i=-2.0, y=3.0 => -0.06
        result = darcy_flow_rate_per_unit_width(0.01, -2.0, 3.0)
        assert result == pytest.approx(-0.06, rel=1e-4)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be non-negative"):
            darcy_flow_rate_per_unit_width(-0.01, 0.5, 5.0)

    def test_y_negative_raises(self):
        with pytest.raises(ValueError, match="y must be non-negative"):
            darcy_flow_rate_per_unit_width(0.01, 0.5, -5.0)


# ============================================================================
# discharge_velocity  (Equation 6-5)
#   v_d = k * i
# ============================================================================

class TestDischargeVelocity:

    def test_basic_valid(self):
        # k=0.05, i=0.4  =>  0.05*0.4 = 0.02
        result = discharge_velocity(0.05, 0.4)
        assert result == pytest.approx(0.02, rel=1e-4)

    def test_zero_k(self):
        result = discharge_velocity(0.0, 1.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_gradient_allowed(self):
        # k=0.1, i=-0.5  =>  -0.05
        result = discharge_velocity(0.1, -0.5)
        assert result == pytest.approx(-0.05, rel=1e-4)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be non-negative"):
            discharge_velocity(-0.01, 0.5)


# ============================================================================
# seepage_velocity  (Equation 6-6)
#   v_s = v_d / n   OR   v_s = v_d * (1+e) / e
# ============================================================================

class TestSeepageVelocity:

    def test_basic_with_porosity(self):
        # v_d=0.5, n=0.4  =>  0.5/0.4 = 1.25
        result = seepage_velocity(0.5, n=0.4)
        assert result == pytest.approx(1.25, rel=1e-4)

    def test_basic_with_void_ratio(self):
        # v_d=0.5, e=0.6  =>  0.5*(1+0.6)/0.6 = 0.5*1.6/0.6 = 1.33333...
        result = seepage_velocity(0.5, e=0.6)
        assert result == pytest.approx(1.3333333333, rel=1e-4)

    def test_high_porosity(self):
        # v_d=1.0, n=0.99  =>  1.0/0.99 = 1.01010101...
        result = seepage_velocity(1.0, n=0.99)
        assert result == pytest.approx(1.01010101, rel=1e-4)

    def test_large_void_ratio(self):
        # v_d=1.0, e=5.0  =>  1.0*(1+5)/5 = 1.2
        result = seepage_velocity(1.0, e=5.0)
        assert result == pytest.approx(1.2, rel=1e-4)

    def test_both_n_and_e_raises(self):
        with pytest.raises(ValueError, match="Provide either n.*or e.*not both"):
            seepage_velocity(0.5, n=0.4, e=0.6)

    def test_neither_n_nor_e_raises(self):
        with pytest.raises(ValueError, match="Either n.*or e.*must be provided"):
            seepage_velocity(0.5)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="Porosity n must be between 0 and 1"):
            seepage_velocity(0.5, n=0.0)

    def test_n_one_raises(self):
        with pytest.raises(ValueError, match="Porosity n must be between 0 and 1"):
            seepage_velocity(0.5, n=1.0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="Porosity n must be between 0 and 1"):
            seepage_velocity(0.5, n=-0.1)

    def test_n_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="Porosity n must be between 0 and 1"):
            seepage_velocity(0.5, n=1.5)

    def test_e_zero_raises(self):
        with pytest.raises(ValueError, match="Void ratio e must be positive"):
            seepage_velocity(0.5, e=0.0)

    def test_e_negative_raises(self):
        with pytest.raises(ValueError, match="Void ratio e must be positive"):
            seepage_velocity(0.5, e=-0.5)


# ============================================================================
# laplace_check_2d  (Equation 6-7)
#   d2h/dx2 + d2h/dy2 = 0
# ============================================================================

class TestLaplaceCheck2D:

    def test_satisfied_exactly(self):
        # 5 + (-5) = 0
        assert laplace_check_2d(5.0, -5.0) is True

    def test_satisfied_within_tolerance(self):
        # 3.0 + (-3.0 + 1e-7) = 1e-7 which is < default 1e-6
        assert laplace_check_2d(3.0, -3.0 + 1e-7) is True

    def test_not_satisfied(self):
        # 3.0 + (-2.0) = 1.0, well above tolerance
        assert laplace_check_2d(3.0, -2.0) is False

    def test_at_boundary_of_tolerance(self):
        # sum = exactly 1e-6  =>  abs(1e-6) <= 1e-6 is True
        assert laplace_check_2d(1e-6, 0.0) is True

    def test_just_outside_tolerance(self):
        # sum = 1.1e-6  =>  abs(1.1e-6) <= 1e-6 is False
        assert laplace_check_2d(1.1e-6, 0.0) is False

    def test_custom_tolerance_satisfied(self):
        # sum = 0.005, tolerance = 0.01  =>  satisfied
        assert laplace_check_2d(1.005, -1.0, tolerance=0.01) is True

    def test_custom_tolerance_not_satisfied(self):
        # sum = 0.05, tolerance = 0.01  =>  not satisfied
        assert laplace_check_2d(1.05, -1.0, tolerance=0.01) is False

    def test_both_zero(self):
        assert laplace_check_2d(0.0, 0.0) is True


# ============================================================================
# anisotropic_transformation_factor  (Equation 6-8)
#   a = sqrt(k_max / k_min)
# ============================================================================

class TestAnisotropicTransformationFactor:

    def test_basic_valid(self):
        # k_max=4.0, k_min=1.0  =>  sqrt(4/1) = 2.0
        result = anisotropic_transformation_factor(4.0, 1.0)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_isotropic(self):
        # k_max = k_min = 3.0  =>  sqrt(1) = 1.0
        result = anisotropic_transformation_factor(3.0, 3.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_large_ratio(self):
        # k_max=100, k_min=1  =>  sqrt(100) = 10.0
        result = anisotropic_transformation_factor(100.0, 1.0)
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_k_max_zero_raises(self):
        with pytest.raises(ValueError, match="k_max must be positive"):
            anisotropic_transformation_factor(0.0, 1.0)

    def test_k_max_negative_raises(self):
        with pytest.raises(ValueError, match="k_max must be positive"):
            anisotropic_transformation_factor(-1.0, 1.0)

    def test_k_min_zero_raises(self):
        with pytest.raises(ValueError, match="k_min must be positive"):
            anisotropic_transformation_factor(4.0, 0.0)

    def test_k_min_negative_raises(self):
        with pytest.raises(ValueError, match="k_min must be positive"):
            anisotropic_transformation_factor(4.0, -1.0)

    def test_k_max_less_than_k_min_raises(self):
        with pytest.raises(ValueError, match="k_max must be greater than or equal to k_min"):
            anisotropic_transformation_factor(1.0, 4.0)


# ============================================================================
# flow_net_flow_rate  (Equation 6-9)
#   q = k * h_L * (N_f / N_d) * W
# ============================================================================

class TestFlowNetFlowRate:

    def test_basic_valid(self):
        # k=0.01, h_L=10, N_f=4, N_d=8, W=5
        # => 0.01 * 10 * (4/8) * 5 = 0.01 * 10 * 0.5 * 5 = 0.25
        result = flow_net_flow_rate(0.01, 10.0, 4.0, 8.0, 5.0)
        assert result == pytest.approx(0.25, rel=1e-4)

    def test_unit_width_default(self):
        # k=0.01, h_L=10, N_f=4, N_d=8, W=1.0 (default)
        # => 0.01 * 10 * 0.5 * 1 = 0.05
        result = flow_net_flow_rate(0.01, 10.0, 4.0, 8.0)
        assert result == pytest.approx(0.05, rel=1e-4)

    def test_negative_head_loss(self):
        # Negative h_L is allowed; k=0.01, h_L=-10, N_f=3, N_d=6, W=1
        # => 0.01 * (-10) * 0.5 * 1 = -0.05
        result = flow_net_flow_rate(0.01, -10.0, 3.0, 6.0, 1.0)
        assert result == pytest.approx(-0.05, rel=1e-4)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            flow_net_flow_rate(0.0, 10.0, 4.0, 8.0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            flow_net_flow_rate(-0.01, 10.0, 4.0, 8.0)

    def test_N_f_zero_raises(self):
        with pytest.raises(ValueError, match="N_f must be positive"):
            flow_net_flow_rate(0.01, 10.0, 0.0, 8.0)

    def test_N_f_negative_raises(self):
        with pytest.raises(ValueError, match="N_f must be positive"):
            flow_net_flow_rate(0.01, 10.0, -4.0, 8.0)

    def test_N_d_zero_raises(self):
        with pytest.raises(ValueError, match="N_d must be positive"):
            flow_net_flow_rate(0.01, 10.0, 4.0, 0.0)

    def test_N_d_negative_raises(self):
        with pytest.raises(ValueError, match="N_d must be positive"):
            flow_net_flow_rate(0.01, 10.0, 4.0, -8.0)

    def test_W_zero_raises(self):
        with pytest.raises(ValueError, match="W must be positive"):
            flow_net_flow_rate(0.01, 10.0, 4.0, 8.0, 0.0)

    def test_W_negative_raises(self):
        with pytest.raises(ValueError, match="W must be positive"):
            flow_net_flow_rate(0.01, 10.0, 4.0, 8.0, -5.0)


# ============================================================================
# head_loss_per_drop  (Equation 6-10)
#   delta_h_L = h_L / N_d
# ============================================================================

class TestHeadLossPerDrop:

    def test_basic_valid(self):
        # h_L=12, N_d=4  =>  12/4 = 3.0
        result = head_loss_per_drop(12.0, 4.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_zero_head_loss(self):
        # h_L=0, N_d=5  =>  0/5 = 0.0
        result = head_loss_per_drop(0.0, 5.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_fractional_drops(self):
        # N_d can be fractional; h_L=10, N_d=3  =>  10/3 = 3.33333...
        result = head_loss_per_drop(10.0, 3.0)
        assert result == pytest.approx(3.3333333333, rel=1e-4)

    def test_N_d_zero_raises(self):
        with pytest.raises(ValueError, match="N_d must be positive"):
            head_loss_per_drop(10.0, 0.0)

    def test_N_d_negative_raises(self):
        with pytest.raises(ValueError, match="N_d must be positive"):
            head_loss_per_drop(10.0, -4.0)


# ============================================================================
# pore_water_pressure  (Equation 6-11)
#   u = (h_t - h_z) * gamma_w
# ============================================================================

class TestPoreWaterPressure:

    def test_basic_valid(self):
        # h_t=15, h_z=5, gamma_w=62.4  =>  (15-5)*62.4 = 624.0
        result = pore_water_pressure(15.0, 5.0, 62.4)
        assert result == pytest.approx(624.0, rel=1e-4)

    def test_zero_pressure_head(self):
        # h_t = h_z = 10, gamma_w=9.81  =>  (10-10)*9.81 = 0
        result = pore_water_pressure(10.0, 10.0, 9.81)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_negative_pressure_head(self):
        # h_t=3, h_z=5, gamma_w=62.4  =>  (3-5)*62.4 = -124.8
        result = pore_water_pressure(3.0, 5.0, 62.4)
        assert result == pytest.approx(-124.8, rel=1e-4)

    def test_gamma_w_zero_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            pore_water_pressure(15.0, 5.0, 0.0)

    def test_gamma_w_negative_raises(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            pore_water_pressure(15.0, 5.0, -9.81)


# ============================================================================
# hydraulic_conductivity_effective_grain_size  (Equation 6-12)
#   k = beta_alpha * D_alpha^x
# ============================================================================

class TestHydraulicConductivityEffectiveGrainSize:

    def test_basic_valid_default_exponent(self):
        # beta=1.0, D=0.2, x=2.0 (default)  =>  1.0 * 0.2^2 = 0.04
        result = hydraulic_conductivity_effective_grain_size(1.0, 0.2)
        assert result == pytest.approx(0.04, rel=1e-4)

    def test_custom_exponent(self):
        # beta=0.5, D=0.3, x=2.5  =>  0.5 * 0.3^2.5 = 0.024647515...
        result = hydraulic_conductivity_effective_grain_size(0.5, 0.3, x=2.5)
        assert result == pytest.approx(0.024647515087732472, rel=1e-4)

    def test_zero_beta(self):
        # beta=0 => k=0 regardless
        result = hydraulic_conductivity_effective_grain_size(0.0, 0.5)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_D(self):
        # D=0 => k=0
        result = hydraulic_conductivity_effective_grain_size(1.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError, match="beta_alpha must be non-negative"):
            hydraulic_conductivity_effective_grain_size(-1.0, 0.2)

    def test_D_negative_raises(self):
        with pytest.raises(ValueError, match="D_alpha must be non-negative"):
            hydraulic_conductivity_effective_grain_size(1.0, -0.2)


# ============================================================================
# kozeny_carman_hydraulic_conductivity  (Equation 6-13)
#   k = 1.99e-4 * (1/S^2) * (e^3/(1+e)) * (1/sum(f_i / D_eff_i^2))
# ============================================================================

class TestKozenyCarmanHydraulicConductivity:

    def test_basic_single_fraction(self):
        # fractions = [(1.0, 2.0, 1.0)], S=6.0, e=0.5
        # D_eff = 2.0^0.404 * 1.0^0.596 = 1.32317...
        # summation = 1.0 / 1.32317^2 = 0.57117...
        # k = 1.99e-4 * (1/36) * (0.125/1.5) * (1/0.57117)
        # k = 8.06495e-07
        fractions = [(1.0, 2.0, 1.0)]
        result = kozeny_carman_hydraulic_conductivity(fractions, 6.0, 0.5)
        assert result == pytest.approx(8.064947898572156e-07, rel=1e-4)

    def test_two_fractions(self):
        # fractions: (0.6, 4.0, 2.0), (0.4, 2.0, 1.0), S=7.0, e=0.7
        # D_eff_1 = 4.0^0.404 * 2.0^0.596 = 1.7531 * 1.5117 = 2.6503
        # D_eff_2 = 2.0^0.404 * 1.0^0.596 = 1.3232 * 1.0 = 1.3232
        # summation = 0.6/2.6503^2 + 0.4/1.3232^2
        #           = 0.6/7.0241 + 0.4/1.7508
        #           = 0.08542 + 0.22849
        #           = 0.31391
        # k = 1.99e-4 * (1/49) * (0.343/1.7) * (1/0.31391)
        # Compute in Python for exact expected value
        import math
        D_eff_1 = 4.0**0.404 * 2.0**0.596
        D_eff_2 = 2.0**0.404 * 1.0**0.596
        summ = 0.6 / D_eff_1**2 + 0.4 / D_eff_2**2
        expected = 1.99e-4 * (1.0/49.0) * (0.7**3 / 1.7) * (1.0/summ)

        fractions = [(0.6, 4.0, 2.0), (0.4, 2.0, 1.0)]
        result = kozeny_carman_hydraulic_conductivity(fractions, 7.0, 0.7)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_spherical_particles(self):
        # S=6 (spherical), single fraction, e=1.0
        # fractions = [(1.0, 1.0, 0.5)]
        # D_eff = 1.0^0.404 * 0.5^0.596 = 1.0 * 0.6605 = 0.6605
        # summation = 1.0 / 0.6605^2 = 1.0 / 0.43626 = 2.2922
        # k = 1.99e-4 * (1/36) * (1.0/2.0) * (1/2.2922)
        D_eff = 1.0**0.404 * 0.5**0.596
        summ = 1.0 / D_eff**2
        expected = 1.99e-4 * (1.0/36.0) * (1.0**3 / 2.0) * (1.0/summ)

        fractions = [(1.0, 1.0, 0.5)]
        result = kozeny_carman_hydraulic_conductivity(fractions, 6.0, 1.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_S_zero_raises(self):
        with pytest.raises(ValueError, match="S must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, 1.0)], 0.0, 0.5)

    def test_S_negative_raises(self):
        with pytest.raises(ValueError, match="S must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, 1.0)], -6.0, 0.5)

    def test_e_zero_raises(self):
        with pytest.raises(ValueError, match="e must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, 1.0)], 6.0, 0.0)

    def test_e_negative_raises(self):
        with pytest.raises(ValueError, match="e must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, 1.0)], 6.0, -0.5)

    def test_D_li_zero_raises(self):
        with pytest.raises(ValueError, match="Sieve sizes D_li and D_si must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 0.0, 1.0)], 6.0, 0.5)

    def test_D_si_zero_raises(self):
        with pytest.raises(ValueError, match="Sieve sizes D_li and D_si must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, 0.0)], 6.0, 0.5)

    def test_D_li_negative_raises(self):
        with pytest.raises(ValueError, match="Sieve sizes D_li and D_si must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, -2.0, 1.0)], 6.0, 0.5)

    def test_D_si_negative_raises(self):
        with pytest.raises(ValueError, match="Sieve sizes D_li and D_si must be positive"):
            kozeny_carman_hydraulic_conductivity([(1.0, 2.0, -1.0)], 6.0, 0.5)

    def test_fraction_negative_raises(self):
        with pytest.raises(ValueError, match="Fraction f_i must be non-negative"):
            kozeny_carman_hydraulic_conductivity([(-0.1, 2.0, 1.0)], 6.0, 0.5)

    def test_all_fractions_zero_raises(self):
        with pytest.raises(ValueError, match="summation term is zero"):
            kozeny_carman_hydraulic_conductivity([(0.0, 2.0, 1.0)], 6.0, 0.5)


# ============================================================================
# kozeny_void_ratio_conductivity_ratio  (Equation 6-14)
#   k1/k2 = (e1^3 / e2^3) * ((1+e2) / (1+e1))
# ============================================================================

class TestKozenyVoidRatioConductivityRatio:

    def test_basic_valid(self):
        # e1=0.8, e2=0.5
        # => (0.8^3/0.5^3) * ((1+0.5)/(1+0.8))
        # => (0.512/0.125) * (1.5/1.8)
        # => 4.096 * 0.83333... = 3.41333...
        result = kozeny_void_ratio_conductivity_ratio(0.8, 0.5)
        assert result == pytest.approx(3.413333333333334, rel=1e-4)

    def test_equal_void_ratios(self):
        # e1=e2=0.7  =>  (0.7^3/0.7^3)*(1.7/1.7) = 1.0
        result = kozeny_void_ratio_conductivity_ratio(0.7, 0.7)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_e1_smaller_than_e2(self):
        # e1=0.5, e2=0.8  =>  (0.125/0.512)*(1.8/1.5) = 0.244140625*1.2 = 0.29297
        result = kozeny_void_ratio_conductivity_ratio(0.5, 0.8)
        expected = (0.5**3 / 0.8**3) * (1.8 / 1.5)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_large_void_ratios(self):
        # e1=3.0, e2=1.0  =>  (27/1)*(2/4) = 13.5
        result = kozeny_void_ratio_conductivity_ratio(3.0, 1.0)
        assert result == pytest.approx(13.5, rel=1e-4)

    def test_e1_zero_raises(self):
        with pytest.raises(ValueError, match="e1 must be positive"):
            kozeny_void_ratio_conductivity_ratio(0.0, 0.5)

    def test_e1_negative_raises(self):
        with pytest.raises(ValueError, match="e1 must be positive"):
            kozeny_void_ratio_conductivity_ratio(-0.5, 0.5)

    def test_e2_zero_raises(self):
        with pytest.raises(ValueError, match="e2 must be positive"):
            kozeny_void_ratio_conductivity_ratio(0.5, 0.0)

    def test_e2_negative_raises(self):
        with pytest.raises(ValueError, match="e2 must be positive"):
            kozeny_void_ratio_conductivity_ratio(0.5, -0.5)


# ============================================================================
# geotextile_permeability_factor_of_safety  (Equation 6-16)
#   FS_g = k_g / k_s  OR  (psi_g * t_g) / k_s
# ============================================================================

class TestGeotextilePermeabilityFactorOfSafety:

    def test_basic_with_k_g(self):
        # k_g=0.01, k_s=0.001  =>  0.01/0.001 = 10.0
        result = geotextile_permeability_factor_of_safety(k_g=0.01, k_s=0.001)
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_basic_with_psi_g_and_t_g(self):
        # psi_g=0.5, t_g=0.02, k_s=0.001  =>  (0.5*0.02)/0.001 = 0.01/0.001 = 10.0
        result = geotextile_permeability_factor_of_safety(
            k_s=0.001, psi_g=0.5, t_g=0.02
        )
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_k_g_takes_precedence_over_psi_t(self):
        # When k_g is provided, psi_g and t_g are ignored
        # k_g=0.05, k_s=0.01  =>  0.05/0.01 = 5.0
        result = geotextile_permeability_factor_of_safety(
            k_g=0.05, k_s=0.01, psi_g=100.0, t_g=100.0
        )
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_fs_less_than_one(self):
        # k_g=0.0005, k_s=0.001  =>  0.5
        result = geotextile_permeability_factor_of_safety(k_g=0.0005, k_s=0.001)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_k_s_none_raises(self):
        with pytest.raises(ValueError, match="k_s.*must be positive"):
            geotextile_permeability_factor_of_safety(k_g=0.01, k_s=None)

    def test_k_s_zero_raises(self):
        with pytest.raises(ValueError, match="k_s.*must be positive"):
            geotextile_permeability_factor_of_safety(k_g=0.01, k_s=0.0)

    def test_k_s_negative_raises(self):
        with pytest.raises(ValueError, match="k_s.*must be positive"):
            geotextile_permeability_factor_of_safety(k_g=0.01, k_s=-0.001)

    def test_k_g_zero_raises(self):
        with pytest.raises(ValueError, match="k_g must be positive"):
            geotextile_permeability_factor_of_safety(k_g=0.0, k_s=0.001)

    def test_k_g_negative_raises(self):
        with pytest.raises(ValueError, match="k_g must be positive"):
            geotextile_permeability_factor_of_safety(k_g=-0.01, k_s=0.001)

    def test_psi_g_zero_raises(self):
        with pytest.raises(ValueError, match="psi_g must be positive"):
            geotextile_permeability_factor_of_safety(
                k_s=0.001, psi_g=0.0, t_g=0.02
            )

    def test_psi_g_negative_raises(self):
        with pytest.raises(ValueError, match="psi_g must be positive"):
            geotextile_permeability_factor_of_safety(
                k_s=0.001, psi_g=-0.5, t_g=0.02
            )

    def test_t_g_zero_raises(self):
        with pytest.raises(ValueError, match="t_g must be positive"):
            geotextile_permeability_factor_of_safety(
                k_s=0.001, psi_g=0.5, t_g=0.0
            )

    def test_t_g_negative_raises(self):
        with pytest.raises(ValueError, match="t_g must be positive"):
            geotextile_permeability_factor_of_safety(
                k_s=0.001, psi_g=0.5, t_g=-0.02
            )

    def test_no_k_g_no_psi_t_raises(self):
        with pytest.raises(ValueError, match="Provide either k_g.*or both.*psi_g.*and t_g"):
            geotextile_permeability_factor_of_safety(k_s=0.001)

    def test_only_psi_g_no_t_g_raises(self):
        with pytest.raises(ValueError, match="Provide either k_g.*or both.*psi_g.*and t_g"):
            geotextile_permeability_factor_of_safety(k_s=0.001, psi_g=0.5)

    def test_only_t_g_no_psi_g_raises(self):
        with pytest.raises(ValueError, match="Provide either k_g.*or both.*psi_g.*and t_g"):
            geotextile_permeability_factor_of_safety(k_s=0.001, t_g=0.02)
