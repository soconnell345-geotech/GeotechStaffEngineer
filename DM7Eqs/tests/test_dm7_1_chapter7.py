"""Tests for geotech.dm7_1.chapter7 — Slope Stability (Equations 7-1 through 7-6)."""

import math

import pytest

from geotech.dm7_1.chapter7 import *


# ---------------------------------------------------------------------------
# factor_of_safety  (Equation 7-1)
#   F = s / tau
# ---------------------------------------------------------------------------

class TestFactorOfSafety:
    """Tests for factor_of_safety(s, tau)."""

    def test_basic_valid(self):
        # F = 500 / 250 = 2.0
        assert factor_of_safety(500.0, 250.0) == pytest.approx(2.0, rel=1e-4)

    def test_stable_slope(self):
        # F = 1200 / 800 = 1.5
        assert factor_of_safety(1200.0, 800.0) == pytest.approx(1.5, rel=1e-4)

    def test_unity_factor(self):
        # Barely stable: F = 300 / 300 = 1.0
        assert factor_of_safety(300.0, 300.0) == pytest.approx(1.0, rel=1e-4)

    def test_unstable_slope(self):
        # F = 100 / 200 = 0.5 (unstable)
        assert factor_of_safety(100.0, 200.0) == pytest.approx(0.5, rel=1e-4)

    def test_zero_shear_strength(self):
        # Edge: s = 0 -> F = 0.0
        assert factor_of_safety(0.0, 100.0) == pytest.approx(0.0, rel=1e-4)

    def test_raises_tau_zero(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            factor_of_safety(500.0, 0.0)

    def test_raises_tau_negative(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            factor_of_safety(500.0, -10.0)

    def test_raises_s_negative(self):
        with pytest.raises(ValueError, match="s must be non-negative"):
            factor_of_safety(-1.0, 100.0)


# ---------------------------------------------------------------------------
# seepage_force  (Equation 7-2)
#   S = i * gamma_w
# ---------------------------------------------------------------------------

class TestSeepageForce:
    """Tests for seepage_force(i, gamma_w)."""

    def test_basic_valid_imperial(self):
        # S = 0.5 * 62.4 = 31.2 pcf
        assert seepage_force(0.5, 62.4) == pytest.approx(31.2, rel=1e-4)

    def test_basic_valid_metric(self):
        # S = 0.3 * 9.81 = 2.943 kN/m^3
        assert seepage_force(0.3, 9.81) == pytest.approx(2.943, rel=1e-4)

    def test_zero_gradient(self):
        # Edge: i = 0 -> S = 0.0
        assert seepage_force(0.0, 62.4) == pytest.approx(0.0, rel=1e-4)

    def test_unit_gradient(self):
        # i = 1.0 -> S = gamma_w = 9.81
        assert seepage_force(1.0, 9.81) == pytest.approx(9.81, rel=1e-4)

    def test_raises_gamma_w_zero(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            seepage_force(0.5, 0.0)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            seepage_force(0.5, -9.81)

    def test_raises_i_negative(self):
        with pytest.raises(ValueError, match="i must be non-negative"):
            seepage_force(-0.1, 62.4)


# ---------------------------------------------------------------------------
# long_term_geosynthetic_strength  (Equation 7-3)
#   T_al = t_ult / (rf_cr * rf_d * rf_id)
# ---------------------------------------------------------------------------

class TestLongTermGeosyntheticStrength:
    """Tests for long_term_geosynthetic_strength(t_ult, rf_cr, rf_d, rf_id)."""

    def test_basic_valid(self):
        # T_al = 4800 / (2.0 * 1.5 * 1.2) = 4800 / 3.6 = 1333.3333...
        result = long_term_geosynthetic_strength(4800.0, 2.0, 1.5, 1.2)
        assert result == pytest.approx(1333.3333333, rel=1e-4)

    def test_all_rf_unity(self):
        # If all reduction factors are 1.0, T_al = t_ult
        # T_al = 3000 / (1.0 * 1.0 * 1.0) = 3000
        result = long_term_geosynthetic_strength(3000.0, 1.0, 1.0, 1.0)
        assert result == pytest.approx(3000.0, rel=1e-4)

    def test_zero_ultimate_strength(self):
        # Edge: t_ult = 0 -> T_al = 0.0
        result = long_term_geosynthetic_strength(0.0, 1.5, 1.1, 1.1)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_large_reduction_factors(self):
        # T_al = 10000 / (3.0 * 2.0 * 2.5) = 10000 / 15.0 = 666.6667
        result = long_term_geosynthetic_strength(10000.0, 3.0, 2.0, 2.5)
        assert result == pytest.approx(666.66667, rel=1e-4)

    def test_raises_t_ult_negative(self):
        with pytest.raises(ValueError, match="t_ult must be non-negative"):
            long_term_geosynthetic_strength(-100.0, 2.0, 1.5, 1.2)

    def test_raises_rf_cr_zero(self):
        with pytest.raises(ValueError, match="rf_cr must be positive"):
            long_term_geosynthetic_strength(4800.0, 0.0, 1.5, 1.2)

    def test_raises_rf_cr_negative(self):
        with pytest.raises(ValueError, match="rf_cr must be positive"):
            long_term_geosynthetic_strength(4800.0, -1.0, 1.5, 1.2)

    def test_raises_rf_d_zero(self):
        with pytest.raises(ValueError, match="rf_d must be positive"):
            long_term_geosynthetic_strength(4800.0, 2.0, 0.0, 1.2)

    def test_raises_rf_d_negative(self):
        with pytest.raises(ValueError, match="rf_d must be positive"):
            long_term_geosynthetic_strength(4800.0, 2.0, -1.0, 1.2)

    def test_raises_rf_id_zero(self):
        with pytest.raises(ValueError, match="rf_id must be positive"):
            long_term_geosynthetic_strength(4800.0, 2.0, 1.5, 0.0)

    def test_raises_rf_id_negative(self):
        with pytest.raises(ValueError, match="rf_id must be positive"):
            long_term_geosynthetic_strength(4800.0, 2.0, 1.5, -1.0)


# ---------------------------------------------------------------------------
# geosynthetic_pullout_resistance  (Equation 7-4)
#   P_r = f_star * alpha * sigma_v_eff * l_e * c_surfaces
# ---------------------------------------------------------------------------

class TestGeosyntheticPulloutResistance:
    """Tests for geosynthetic_pullout_resistance(f_star, alpha, sigma_v_eff, l_e, c_surfaces)."""

    def test_basic_valid_default_c(self):
        # P_r = 0.8 * 0.6 * 500 * 5.0 * 2.0 = 2400.0
        result = geosynthetic_pullout_resistance(0.8, 0.6, 500.0, 5.0)
        assert result == pytest.approx(2400.0, rel=1e-4)

    def test_explicit_c_surfaces(self):
        # P_r = 0.8 * 0.6 * 500 * 5.0 * 1.0 = 1200.0
        result = geosynthetic_pullout_resistance(0.8, 0.6, 500.0, 5.0, c_surfaces=1.0)
        assert result == pytest.approx(1200.0, rel=1e-4)

    def test_different_values(self):
        # P_r = 1.2 * 0.8 * 1000 * 3.0 * 2.0 = 5760.0
        result = geosynthetic_pullout_resistance(1.2, 0.8, 1000.0, 3.0, 2.0)
        assert result == pytest.approx(5760.0, rel=1e-4)

    def test_zero_f_star(self):
        # Edge: f_star = 0 -> P_r = 0.0
        result = geosynthetic_pullout_resistance(0.0, 0.6, 500.0, 5.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_alpha(self):
        # Edge: alpha = 0 -> P_r = 0.0
        result = geosynthetic_pullout_resistance(0.8, 0.0, 500.0, 5.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_sigma_v_eff(self):
        # Edge: sigma_v_eff = 0 -> P_r = 0.0
        result = geosynthetic_pullout_resistance(0.8, 0.6, 0.0, 5.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_zero_l_e(self):
        # Edge: l_e = 0 -> P_r = 0.0
        result = geosynthetic_pullout_resistance(0.8, 0.6, 500.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_f_star_negative(self):
        with pytest.raises(ValueError, match="f_star must be non-negative"):
            geosynthetic_pullout_resistance(-0.1, 0.6, 500.0, 5.0)

    def test_raises_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            geosynthetic_pullout_resistance(0.8, -0.1, 500.0, 5.0)

    def test_raises_sigma_v_eff_negative(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be non-negative"):
            geosynthetic_pullout_resistance(0.8, 0.6, -100.0, 5.0)

    def test_raises_l_e_negative(self):
        with pytest.raises(ValueError, match="l_e must be non-negative"):
            geosynthetic_pullout_resistance(0.8, 0.6, 500.0, -1.0)

    def test_raises_c_surfaces_zero(self):
        with pytest.raises(ValueError, match="c_surfaces must be positive"):
            geosynthetic_pullout_resistance(0.8, 0.6, 500.0, 5.0, c_surfaces=0.0)

    def test_raises_c_surfaces_negative(self):
        with pytest.raises(ValueError, match="c_surfaces must be positive"):
            geosynthetic_pullout_resistance(0.8, 0.6, 500.0, 5.0, c_surfaces=-1.0)


# ---------------------------------------------------------------------------
# coefficient_of_interaction  (Equation 7-5)
#   C_i = tan(delta) / tan(phi')
# ---------------------------------------------------------------------------

class TestCoefficientOfInteraction:
    """Tests for coefficient_of_interaction(delta_deg, phi_eff_deg)."""

    def test_basic_valid(self):
        # C_i = tan(30) / tan(45) = 0.57735... / 1.0 = 0.57735...
        result = coefficient_of_interaction(30.0, 45.0)
        expected = math.tan(math.radians(30.0)) / math.tan(math.radians(45.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_equal_angles(self):
        # delta = phi -> C_i = 1.0
        result = coefficient_of_interaction(35.0, 35.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_zero_delta(self):
        # Edge: delta = 0 -> tan(0) = 0 -> C_i = 0.0
        result = coefficient_of_interaction(0.0, 30.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_small_angles(self):
        # C_i = tan(10) / tan(20)
        # tan(10) = 0.17633, tan(20) = 0.36397
        # C_i = 0.17633 / 0.36397 = 0.48442...
        result = coefficient_of_interaction(10.0, 20.0)
        expected = math.tan(math.radians(10.0)) / math.tan(math.radians(20.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_delta_greater_than_phi(self):
        # delta > phi is valid, C_i > 1
        # C_i = tan(40) / tan(25) = 0.83910 / 0.46631 = 1.7998...
        result = coefficient_of_interaction(40.0, 25.0)
        expected = math.tan(math.radians(40.0)) / math.tan(math.radians(25.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            coefficient_of_interaction(30.0, 0.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            coefficient_of_interaction(30.0, -10.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            coefficient_of_interaction(30.0, 90.0)

    def test_raises_phi_greater_than_90(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            coefficient_of_interaction(30.0, 100.0)

    def test_raises_delta_negative(self):
        with pytest.raises(ValueError, match="delta_deg must be non-negative and less than 90"):
            coefficient_of_interaction(-5.0, 30.0)

    def test_raises_delta_90(self):
        with pytest.raises(ValueError, match="delta_deg must be non-negative and less than 90"):
            coefficient_of_interaction(90.0, 30.0)

    def test_raises_delta_greater_than_90(self):
        with pytest.raises(ValueError, match="delta_deg must be non-negative and less than 90"):
            coefficient_of_interaction(95.0, 30.0)


# ---------------------------------------------------------------------------
# pullout_resistance_factor  (Equation 7-6)
#   F* = c_i * tan(phi')
# ---------------------------------------------------------------------------

class TestPulloutResistanceFactor:
    """Tests for pullout_resistance_factor(c_i, phi_eff_deg)."""

    def test_basic_valid(self):
        # F* = 0.8 * tan(30) = 0.8 * 0.57735 = 0.46188...
        result = pullout_resistance_factor(0.8, 30.0)
        expected = 0.8 * math.tan(math.radians(30.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_ci_unity(self):
        # F* = 1.0 * tan(45) = 1.0 * 1.0 = 1.0
        result = pullout_resistance_factor(1.0, 45.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_zero_ci(self):
        # Edge: c_i = 0 -> F* = 0.0
        result = pullout_resistance_factor(0.0, 30.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_high_friction_angle(self):
        # F* = 0.7 * tan(60) = 0.7 * 1.73205 = 1.21244...
        result = pullout_resistance_factor(0.7, 60.0)
        expected = 0.7 * math.tan(math.radians(60.0))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_roundtrip_with_coefficient_of_interaction(self):
        # Use eq 7-5 to get C_i, then eq 7-6 to get F*
        # F* = C_i * tan(phi) = [tan(delta)/tan(phi)] * tan(phi) = tan(delta)
        delta = 25.0
        phi = 35.0
        ci = coefficient_of_interaction(delta, phi)
        f_star = pullout_resistance_factor(ci, phi)
        assert f_star == pytest.approx(math.tan(math.radians(delta)), rel=1e-4)

    def test_raises_ci_negative(self):
        with pytest.raises(ValueError, match="c_i must be non-negative"):
            pullout_resistance_factor(-0.5, 30.0)

    def test_raises_phi_zero(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            pullout_resistance_factor(0.8, 0.0)

    def test_raises_phi_negative(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            pullout_resistance_factor(0.8, -10.0)

    def test_raises_phi_90(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            pullout_resistance_factor(0.8, 90.0)

    def test_raises_phi_greater_than_90(self):
        with pytest.raises(ValueError, match="phi_eff_deg must be greater than 0 and less than 90"):
            pullout_resistance_factor(0.8, 120.0)
