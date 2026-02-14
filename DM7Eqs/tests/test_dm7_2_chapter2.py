"""Tests for geotech.dm7_2.chapter2 -- UFC 3-220-20, Chapter 2: Excavations.

Covers all 10 public functions (Equations 2-1 through 2-10) with valid-input,
edge-case, and ValueError tests for every validation branch.
"""

import math

import pytest

from geotech.dm7_2.chapter2 import *


# ============================================================================
# 1. critical_depth_vertical_cut_clay  (Equation 2-1)
#    H_crit = 4 * s_u / gamma_t
# ============================================================================

class TestCriticalDepthVerticalCutClay:
    """Tests for critical_depth_vertical_cut_clay."""

    def test_basic_valid_input(self):
        # s_u = 500 psf, gamma_t = 120 pcf
        # H_crit = 4 * 500 / 120 = 2000 / 120 = 16.6667 ft
        result = critical_depth_vertical_cut_clay(500.0, 120.0)
        assert result == pytest.approx(2000.0 / 120.0, rel=1e-4)

    def test_another_valid_input(self):
        # s_u = 25 kPa, gamma_t = 18 kN/m^3
        # H_crit = 4 * 25 / 18 = 100 / 18 = 5.5556 m
        result = critical_depth_vertical_cut_clay(25.0, 18.0)
        assert result == pytest.approx(100.0 / 18.0, rel=1e-4)

    def test_s_u_zero(self):
        # s_u = 0 is allowed (non-negative), H_crit = 0
        result = critical_depth_vertical_cut_clay(0.0, 120.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_s_u(self):
        with pytest.raises(ValueError, match="s_u must be non-negative"):
            critical_depth_vertical_cut_clay(-1.0, 120.0)

    def test_raises_zero_gamma_t(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            critical_depth_vertical_cut_clay(500.0, 0.0)

    def test_raises_negative_gamma_t(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            critical_depth_vertical_cut_clay(500.0, -10.0)


# ============================================================================
# 2. normalized_wall_stiffness  (Equation 2-2)
#    K_wall = E * I / (gamma_t * h^4)
# ============================================================================

class TestNormalizedWallStiffness:
    """Tests for normalized_wall_stiffness."""

    def test_basic_valid_input(self):
        # E = 29e6 psi = 29e6 * 144 = 4.176e9 psf, I = 0.05 ft^3,
        # gamma_t = 120 pcf, h = 10 ft
        # K_wall = (4.176e9 * 0.05) / (120 * 10^4)
        #        = 2.088e8 / 1.2e6 = 174.0
        E = 4.176e9
        I = 0.05
        gamma_t = 120.0
        h = 10.0
        expected = (E * I) / (gamma_t * h ** 4)  # 2.088e8 / 1.2e6 = 174.0
        result = normalized_wall_stiffness(E, I, gamma_t, h)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_small_spacing(self):
        # h = 5 ft => h^4 = 625
        # E = 1e6, I = 1.0, gamma_t = 100, h = 5
        # K_wall = 1e6 * 1.0 / (100 * 625) = 1e6 / 62500 = 16.0
        result = normalized_wall_stiffness(1e6, 1.0, 100.0, 5.0)
        assert result == pytest.approx(16.0, rel=1e-4)

    def test_raises_zero_E(self):
        with pytest.raises(ValueError, match="E must be positive"):
            normalized_wall_stiffness(0.0, 1.0, 120.0, 10.0)

    def test_raises_negative_E(self):
        with pytest.raises(ValueError, match="E must be positive"):
            normalized_wall_stiffness(-1.0, 1.0, 120.0, 10.0)

    def test_raises_zero_I(self):
        with pytest.raises(ValueError, match="I must be positive"):
            normalized_wall_stiffness(1e6, 0.0, 120.0, 10.0)

    def test_raises_negative_I(self):
        with pytest.raises(ValueError, match="I must be positive"):
            normalized_wall_stiffness(1e6, -0.5, 120.0, 10.0)

    def test_raises_zero_gamma_t(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            normalized_wall_stiffness(1e6, 1.0, 0.0, 10.0)

    def test_raises_negative_gamma_t(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            normalized_wall_stiffness(1e6, 1.0, -120.0, 10.0)

    def test_raises_zero_h(self):
        with pytest.raises(ValueError, match="h must be positive"):
            normalized_wall_stiffness(1e6, 1.0, 120.0, 0.0)

    def test_raises_negative_h(self):
        with pytest.raises(ValueError, match="h must be positive"):
            normalized_wall_stiffness(1e6, 1.0, 120.0, -5.0)


# ============================================================================
# 3. angular_distortion  (Equation 2-3)
#    beta = (delta_Vi - delta_Vj) / d_b
# ============================================================================

class TestAngularDistortion:
    """Tests for angular_distortion."""

    def test_basic_valid_input(self):
        # delta_Vi = 1.2 in, delta_Vj = 0.4 in, d_b = 40 ft = 480 in
        # beta = (1.2 - 0.4) / 480 = 0.8 / 480 = 0.0016667
        result = angular_distortion(1.2, 0.4, 480.0)
        assert result == pytest.approx(0.8 / 480.0, rel=1e-4)

    def test_negative_distortion(self):
        # delta_Vi < delta_Vj is mathematically valid (negative beta)
        # delta_Vi = 0.2, delta_Vj = 0.8, d_b = 30
        # beta = (0.2 - 0.8) / 30 = -0.6 / 30 = -0.02
        result = angular_distortion(0.2, 0.8, 30.0)
        assert result == pytest.approx(-0.02, rel=1e-4)

    def test_zero_differential(self):
        # delta_Vi = delta_Vj => beta = 0
        result = angular_distortion(0.5, 0.5, 100.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_zero_d_b(self):
        with pytest.raises(ValueError, match="d_b must be positive"):
            angular_distortion(1.0, 0.5, 0.0)

    def test_raises_negative_d_b(self):
        with pytest.raises(ValueError, match="d_b must be positive"):
            angular_distortion(1.0, 0.5, -10.0)


# ============================================================================
# 4. horizontal_strain  (Equation 2-4)
#    epsilon_h = (delta_Hi - delta_Hj) / d_b
# ============================================================================

class TestHorizontalStrain:
    """Tests for horizontal_strain."""

    def test_basic_valid_input(self):
        # delta_Hi = 0.6 in, delta_Hj = 0.1 in, d_b = 50 ft = 600 in
        # epsilon_h = (0.6 - 0.1) / 600 = 0.5 / 600 = 8.3333e-4
        result = horizontal_strain(0.6, 0.1, 600.0)
        assert result == pytest.approx(0.5 / 600.0, rel=1e-4)

    def test_negative_strain(self):
        # delta_Hi < delta_Hj => compressive strain (negative)
        # delta_Hi = 0.1, delta_Hj = 0.5, d_b = 200
        # epsilon_h = (0.1 - 0.5) / 200 = -0.4 / 200 = -0.002
        result = horizontal_strain(0.1, 0.5, 200.0)
        assert result == pytest.approx(-0.002, rel=1e-4)

    def test_zero_differential(self):
        result = horizontal_strain(0.3, 0.3, 100.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_zero_d_b(self):
        with pytest.raises(ValueError, match="d_b must be positive"):
            horizontal_strain(0.5, 0.1, 0.0)

    def test_raises_negative_d_b(self):
        with pytest.raises(ValueError, match="d_b must be positive"):
            horizontal_strain(0.5, 0.1, -20.0)


# ============================================================================
# 5. movement_stiff_clay_sand  (Equation 2-5)
#    delta_i = delta_m * (d_0 - d_i) / d_0   for d_i < d_0, else 0
# ============================================================================

class TestMovementStiffClaySand:
    """Tests for movement_stiff_clay_sand."""

    def test_basic_valid_input(self):
        # delta_m = 0.5 in, d_i = 10 ft, d_0 = 30 ft (stiff clay, H=10 => d_0=3*10)
        # delta_i = 0.5 * (30 - 10) / 30 = 0.5 * 20/30 = 0.5 * 0.6667 = 0.3333
        result = movement_stiff_clay_sand(0.5, 10.0, 30.0)
        assert result == pytest.approx(0.5 * 20.0 / 30.0, rel=1e-4)

    def test_at_wall(self):
        # d_i = 0 => delta_i = delta_m * (d_0 - 0) / d_0 = delta_m
        result = movement_stiff_clay_sand(1.0, 0.0, 20.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_at_d_0_boundary(self):
        # d_i = d_0 => returns 0.0 (d_i >= d_0 branch)
        result = movement_stiff_clay_sand(1.0, 30.0, 30.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_beyond_d_0(self):
        # d_i > d_0 => returns 0.0
        result = movement_stiff_clay_sand(1.0, 50.0, 30.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_delta_m_zero(self):
        # delta_m = 0 => movement = 0 everywhere
        result = movement_stiff_clay_sand(0.0, 5.0, 30.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_delta_m(self):
        with pytest.raises(ValueError, match="delta_m must be non-negative"):
            movement_stiff_clay_sand(-0.1, 10.0, 30.0)

    def test_raises_negative_d_i(self):
        with pytest.raises(ValueError, match="d_i must be non-negative"):
            movement_stiff_clay_sand(0.5, -1.0, 30.0)

    def test_raises_zero_d_0(self):
        with pytest.raises(ValueError, match="d_0 must be positive"):
            movement_stiff_clay_sand(0.5, 10.0, 0.0)

    def test_raises_negative_d_0(self):
        with pytest.raises(ValueError, match="d_0 must be positive"):
            movement_stiff_clay_sand(0.5, 10.0, -5.0)


# ============================================================================
# 6. movement_soft_to_medium_clay  (Equation 2-6)
#    delta_i = delta_m                              for d_i <= 0.75*H
#    delta_i = delta_m * (1.5*H - d_i) / (0.75*H)  for 0.75*H < d_i <= 1.5*H
#    delta_i = 0                                    for d_i > 1.5*H
# ============================================================================

class TestMovementSoftToMediumClay:
    """Tests for movement_soft_to_medium_clay."""

    def test_within_plateau_zone(self):
        # H = 20 ft, 0.75*H = 15 ft, d_i = 10 ft < 15 ft => delta_i = delta_m
        result = movement_soft_to_medium_clay(1.5, 10.0, 20.0)
        assert result == pytest.approx(1.5, rel=1e-4)

    def test_at_plateau_boundary(self):
        # d_i = 0.75 * H = 15 ft => delta_i = delta_m (the <= branch)
        result = movement_soft_to_medium_clay(1.5, 15.0, 20.0)
        assert result == pytest.approx(1.5, rel=1e-4)

    def test_in_linear_decay_zone(self):
        # H = 20, 0.75*H = 15, 1.5*H = 30, d_i = 22.5 (midpoint of decay zone)
        # delta_i = 1.5 * (30 - 22.5) / 15 = 1.5 * 7.5 / 15 = 1.5 * 0.5 = 0.75
        result = movement_soft_to_medium_clay(1.5, 22.5, 20.0)
        assert result == pytest.approx(0.75, rel=1e-4)

    def test_at_far_boundary(self):
        # d_i = 1.5 * H = 30 ft
        # delta_i = 1.5 * (30 - 30) / 15 = 0
        result = movement_soft_to_medium_clay(1.5, 30.0, 20.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_beyond_far_boundary(self):
        # d_i = 35 > 1.5*20 = 30 => returns 0.0
        result = movement_soft_to_medium_clay(1.5, 35.0, 20.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_at_wall(self):
        # d_i = 0 => in plateau zone => delta_i = delta_m
        result = movement_soft_to_medium_clay(2.0, 0.0, 10.0)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_delta_m_zero(self):
        result = movement_soft_to_medium_clay(0.0, 5.0, 20.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_delta_m(self):
        with pytest.raises(ValueError, match="delta_m must be non-negative"):
            movement_soft_to_medium_clay(-0.1, 10.0, 20.0)

    def test_raises_negative_d_i(self):
        with pytest.raises(ValueError, match="d_i must be non-negative"):
            movement_soft_to_medium_clay(1.0, -1.0, 20.0)

    def test_raises_zero_H(self):
        with pytest.raises(ValueError, match="H must be positive"):
            movement_soft_to_medium_clay(1.0, 5.0, 0.0)

    def test_raises_negative_H(self):
        with pytest.raises(ValueError, match="H must be positive"):
            movement_soft_to_medium_clay(1.0, 5.0, -10.0)


# ============================================================================
# 7. scaled_distance  (Equation 2-7)
#    SD = D / W^beta
# ============================================================================

class TestScaledDistance:
    """Tests for scaled_distance."""

    def test_default_beta_half(self):
        # D = 100 ft, W = 25 lb, beta = 0.5 (default)
        # SD = 100 / 25^0.5 = 100 / 5 = 20.0
        result = scaled_distance(100.0, 25.0)
        assert result == pytest.approx(20.0, rel=1e-4)

    def test_beta_033_near_field(self):
        # D = 15 ft, W = 8 lb, beta = 0.33
        # SD = 15 / 8^0.33 = 15 / 8^0.33
        # 8^0.33 = exp(0.33 * ln(8)) = exp(0.33 * 2.07944) = exp(0.68622) = 1.98614
        # SD = 15 / 1.98614 = 7.5523
        expected = 15.0 / (8.0 ** 0.33)
        result = scaled_distance(15.0, 8.0, beta=0.33)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_D_zero(self):
        # D = 0 is allowed (non-negative), SD = 0
        result = scaled_distance(0.0, 10.0, 0.5)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_D(self):
        with pytest.raises(ValueError, match="D must be non-negative"):
            scaled_distance(-1.0, 10.0)

    def test_raises_zero_W(self):
        with pytest.raises(ValueError, match="W must be positive"):
            scaled_distance(100.0, 0.0)

    def test_raises_negative_W(self):
        with pytest.raises(ValueError, match="W must be positive"):
            scaled_distance(100.0, -5.0)

    def test_raises_invalid_beta(self):
        with pytest.raises(ValueError, match="beta must be 0.33"):
            scaled_distance(100.0, 25.0, beta=0.25)

    def test_raises_invalid_beta_one(self):
        with pytest.raises(ValueError, match="beta must be 0.33"):
            scaled_distance(100.0, 25.0, beta=1.0)


# ============================================================================
# 8. peak_particle_velocity  (Equation 2-8)
#    PPV = K * SD^(-1.6)
# ============================================================================

class TestPeakParticleVelocity:
    """Tests for peak_particle_velocity."""

    def test_basic_valid_input(self):
        # K = 150, SD = 20
        # PPV = 150 * 20^(-1.6)
        # 20^1.6 = exp(1.6 * ln(20)) = exp(1.6 * 2.99573) = exp(4.79317) = 120.903
        # PPV = 150 / 120.903 = 1.2407
        expected = 150.0 * (20.0 ** (-1.6))
        result = peak_particle_velocity(150.0, 20.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_lower_bound_K(self):
        # K = 20 (lower bound), SD = 10
        # PPV = 20 * 10^(-1.6) = 20 / 10^1.6 = 20 / 39.8107 = 0.50238
        expected = 20.0 * (10.0 ** (-1.6))
        result = peak_particle_velocity(20.0, 10.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_upper_bound_K(self):
        # K = 242, SD = 5
        # 5^1.6 = exp(1.6 * ln(5)) = exp(1.6 * 1.60944) = exp(2.57510) = 13.1601
        # PPV = 242 / 13.1601 = 18.389
        expected = 242.0 * (5.0 ** (-1.6))
        result = peak_particle_velocity(242.0, 5.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_zero_K(self):
        with pytest.raises(ValueError, match="K must be positive"):
            peak_particle_velocity(0.0, 10.0)

    def test_raises_negative_K(self):
        with pytest.raises(ValueError, match="K must be positive"):
            peak_particle_velocity(-10.0, 10.0)

    def test_raises_zero_SD(self):
        with pytest.raises(ValueError, match="SD must be positive"):
            peak_particle_velocity(150.0, 0.0)

    def test_raises_negative_SD(self):
        with pytest.raises(ValueError, match="SD must be positive"):
            peak_particle_velocity(150.0, -5.0)


# ============================================================================
# 9. confinement_factor_from_blast_data  (Equation 2-9)
#    K = PPV / SD^(-1.6)
# ============================================================================

class TestConfinementFactorFromBlastData:
    """Tests for confinement_factor_from_blast_data."""

    def test_basic_valid_input(self):
        # PPV = 2.0 in/sec, SD = 15
        # SD^(-1.6) = 15^(-1.6) = 1 / 15^1.6
        # 15^1.6 = exp(1.6 * ln(15)) = exp(1.6 * 2.70805) = exp(4.33288) = 76.1035
        # K = 2.0 / (1/76.1035) = 2.0 * 76.1035 = 152.207
        expected = 2.0 / (15.0 ** (-1.6))
        result = confinement_factor_from_blast_data(2.0, 15.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_round_trip_with_ppv(self):
        # Use K=150, SD=20, compute PPV, then back-calculate K
        K_original = 150.0
        SD = 20.0
        ppv = peak_particle_velocity(K_original, SD)
        K_back = confinement_factor_from_blast_data(ppv, SD)
        assert K_back == pytest.approx(K_original, rel=1e-4)

    def test_ppv_zero(self):
        # PPV = 0 is allowed (non-negative), K = 0
        result = confinement_factor_from_blast_data(0.0, 10.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_PPV(self):
        with pytest.raises(ValueError, match="PPV must be non-negative"):
            confinement_factor_from_blast_data(-1.0, 10.0)

    def test_raises_zero_SD(self):
        with pytest.raises(ValueError, match="SD must be positive"):
            confinement_factor_from_blast_data(2.0, 0.0)

    def test_raises_negative_SD(self):
        with pytest.raises(ValueError, match="SD must be positive"):
            confinement_factor_from_blast_data(2.0, -5.0)


# ============================================================================
# 10. inverse_specific_resistance_sheet_piling  (Equation 2-10)
#     rho = q * gamma_w / delta_p
# ============================================================================

class TestInverseSpecificResistanceSheetPiling:
    """Tests for inverse_specific_resistance_sheet_piling."""

    def test_basic_valid_input(self):
        # q = 0.001 cm^2/sec, gamma_w = 9.81 kN/m^3, delta_p = 50 kPa
        # rho = 0.001 * 9.81 / 50 = 0.00981 / 50 = 1.962e-4
        result = inverse_specific_resistance_sheet_piling(0.001, 9.81, 50.0)
        assert result == pytest.approx(0.001 * 9.81 / 50.0, rel=1e-4)

    def test_us_customary_units(self):
        # q = 0.005 ft^3/sec/ft, gamma_w = 62.4 pcf, delta_p = 200 psf
        # rho = 0.005 * 62.4 / 200 = 0.312 / 200 = 0.00156
        result = inverse_specific_resistance_sheet_piling(0.005, 62.4, 200.0)
        assert result == pytest.approx(0.312 / 200.0, rel=1e-4)

    def test_q_zero(self):
        # q = 0 is allowed (non-negative), rho = 0
        result = inverse_specific_resistance_sheet_piling(0.0, 62.4, 200.0)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_raises_negative_q(self):
        with pytest.raises(ValueError, match="q must be non-negative"):
            inverse_specific_resistance_sheet_piling(-0.001, 62.4, 200.0)

    def test_raises_zero_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            inverse_specific_resistance_sheet_piling(0.001, 0.0, 200.0)

    def test_raises_negative_gamma_w(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            inverse_specific_resistance_sheet_piling(0.001, -9.81, 200.0)

    def test_raises_zero_delta_p(self):
        with pytest.raises(ValueError, match="delta_p must be positive"):
            inverse_specific_resistance_sheet_piling(0.001, 62.4, 0.0)

    def test_raises_negative_delta_p(self):
        with pytest.raises(ValueError, match="delta_p must be positive"):
            inverse_specific_resistance_sheet_piling(0.001, 62.4, -100.0)
