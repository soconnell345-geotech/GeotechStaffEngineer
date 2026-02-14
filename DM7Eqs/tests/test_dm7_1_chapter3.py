"""Comprehensive tests for geotech.dm7_1.chapter3 (UFC 3-220-10, Chapter 3).

Tests cover every public function with valid inputs (hand-calculated expected
results), edge cases, and all ValueError validation branches.
"""

import pytest

from geotech.dm7_1.chapter3 import *


# ---------------------------------------------------------------------------
# relative_compaction  (Equation 3-1)
# RC = (gamma_d / gamma_d_max) * 100
# ---------------------------------------------------------------------------

class TestRelativeCompaction:
    """Tests for relative_compaction()."""

    def test_basic_valid(self):
        # RC = (110 / 125) * 100 = 88.0
        result = relative_compaction(110.0, 125.0)
        assert result == pytest.approx(88.0, rel=1e-4)

    def test_full_compaction(self):
        # RC = (125 / 125) * 100 = 100.0
        result = relative_compaction(125.0, 125.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_gamma_d_zero(self):
        # RC = (0 / 125) * 100 = 0.0
        result = relative_compaction(0.0, 125.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_over_compaction(self):
        # gamma_d can exceed gamma_d_max
        # RC = (130 / 125) * 100 = 104.0
        result = relative_compaction(130.0, 125.0)
        assert result == pytest.approx(104.0, rel=1e-4)

    def test_small_values(self):
        # RC = (5 / 10) * 100 = 50.0
        result = relative_compaction(5.0, 10.0)
        assert result == pytest.approx(50.0, rel=1e-4)

    def test_raises_gamma_d_max_zero(self):
        with pytest.raises(ValueError, match="gamma_d_max must be positive"):
            relative_compaction(110.0, 0.0)

    def test_raises_gamma_d_max_negative(self):
        with pytest.raises(ValueError, match="gamma_d_max must be positive"):
            relative_compaction(110.0, -10.0)

    def test_raises_gamma_d_negative(self):
        with pytest.raises(ValueError, match="gamma_d must be non-negative"):
            relative_compaction(-5.0, 125.0)


# ---------------------------------------------------------------------------
# relative_density_from_void_ratio  (Equation 3-2, void-ratio form)
# Dr = ((e_max - e) / (e_max - e_min)) * 100
# ---------------------------------------------------------------------------

class TestRelativeDensityFromVoidRatio:
    """Tests for relative_density_from_void_ratio()."""

    def test_basic_valid(self):
        # Dr = (0.90 - 0.65) / (0.90 - 0.50) * 100 = 0.25/0.40*100 = 62.5
        result = relative_density_from_void_ratio(0.65, 0.90, 0.50)
        assert result == pytest.approx(62.5, rel=1e-4)

    def test_densest_state(self):
        # e = e_min => Dr = (0.90-0.50)/(0.90-0.50)*100 = 100.0
        result = relative_density_from_void_ratio(0.50, 0.90, 0.50)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_loosest_state(self):
        # e = e_max => Dr = (0.90-0.90)/(0.90-0.50)*100 = 0.0
        result = relative_density_from_void_ratio(0.90, 0.90, 0.50)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_e_beyond_e_max_gives_negative(self):
        # e > e_max is physically unusual but not blocked by validation
        # Dr = (0.90 - 1.00) / (0.90 - 0.50) * 100 = -0.10/0.40*100 = -25.0
        result = relative_density_from_void_ratio(1.00, 0.90, 0.50)
        assert result == pytest.approx(-25.0, rel=1e-4)

    def test_midpoint(self):
        # Dr = (0.80 - 0.60) / (0.80 - 0.40) * 100 = 0.20/0.40*100 = 50.0
        result = relative_density_from_void_ratio(0.60, 0.80, 0.40)
        assert result == pytest.approx(50.0, rel=1e-4)

    def test_raises_e_max_equals_e_min(self):
        with pytest.raises(ValueError, match="e_max must be greater than e_min"):
            relative_density_from_void_ratio(0.50, 0.50, 0.50)

    def test_raises_e_max_less_than_e_min(self):
        with pytest.raises(ValueError, match="e_max must be greater than e_min"):
            relative_density_from_void_ratio(0.50, 0.40, 0.60)


# ---------------------------------------------------------------------------
# relative_density_from_dry_density  (Equation 3-2, density form)
# Dr = (gamma_d_max / gamma_d) * ((gamma_d - gamma_d_min) /
#       (gamma_d_max - gamma_d_min)) * 100
# ---------------------------------------------------------------------------

class TestRelativeDensityFromDryDensity:
    """Tests for relative_density_from_dry_density()."""

    def test_basic_valid(self):
        # Dr = (120/105) * ((105-95)/(120-95)) * 100
        #    = 1.142857... * (10/25) * 100
        #    = 1.142857... * 0.4 * 100 = 45.71428...
        result = relative_density_from_dry_density(105.0, 120.0, 95.0)
        assert result == pytest.approx(45.71428571428571, rel=1e-4)

    def test_densest_state(self):
        # gamma_d = gamma_d_max => Dr = (120/120)*((120-95)/(120-95))*100
        #    = 1.0 * 1.0 * 100 = 100.0
        result = relative_density_from_dry_density(120.0, 120.0, 95.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_loosest_state(self):
        # gamma_d = gamma_d_min => Dr = (120/95)*((95-95)/(120-95))*100
        #    = (120/95) * 0.0 * 100 = 0.0
        result = relative_density_from_dry_density(95.0, 120.0, 95.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_another_valid(self):
        # Dr = (110/100) * ((100-90)/(110-90)) * 100
        #    = 1.1 * (10/20) * 100 = 1.1 * 0.5 * 100 = 55.0
        result = relative_density_from_dry_density(100.0, 110.0, 90.0)
        assert result == pytest.approx(55.0, rel=1e-4)

    def test_raises_gamma_d_zero(self):
        with pytest.raises(ValueError, match="gamma_d must be positive"):
            relative_density_from_dry_density(0.0, 120.0, 95.0)

    def test_raises_gamma_d_negative(self):
        with pytest.raises(ValueError, match="gamma_d must be positive"):
            relative_density_from_dry_density(-10.0, 120.0, 95.0)

    def test_raises_gamma_d_max_equals_gamma_d_min(self):
        with pytest.raises(ValueError, match="gamma_d_max must be greater than gamma_d_min"):
            relative_density_from_dry_density(100.0, 100.0, 100.0)

    def test_raises_gamma_d_max_less_than_gamma_d_min(self):
        with pytest.raises(ValueError, match="gamma_d_max must be greater than gamma_d_min"):
            relative_density_from_dry_density(100.0, 90.0, 110.0)


# ---------------------------------------------------------------------------
# cyclic_stress_ratio  (Equation 3-3)
# CSR = tau_cyc / sigma_v_eff
# ---------------------------------------------------------------------------

class TestCyclicStressRatio:
    """Tests for cyclic_stress_ratio()."""

    def test_basic_valid(self):
        # CSR = 500 / 2000 = 0.25
        result = cyclic_stress_ratio(500.0, 2000.0)
        assert result == pytest.approx(0.25, rel=1e-4)

    def test_zero_shear_stress(self):
        # CSR = 0 / 2000 = 0.0
        result = cyclic_stress_ratio(0.0, 2000.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_typical_liquefaction_value(self):
        # CSR = 150 / 1000 = 0.15
        result = cyclic_stress_ratio(150.0, 1000.0)
        assert result == pytest.approx(0.15, rel=1e-4)

    def test_high_csr(self):
        # CSR = 800 / 1000 = 0.8
        result = cyclic_stress_ratio(800.0, 1000.0)
        assert result == pytest.approx(0.8, rel=1e-4)

    def test_negative_tau_cyc_allowed(self):
        # Negative tau_cyc is not blocked by validation
        # CSR = -100 / 2000 = -0.05
        result = cyclic_stress_ratio(-100.0, 2000.0)
        assert result == pytest.approx(-0.05, rel=1e-4)

    def test_raises_sigma_v_eff_zero(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            cyclic_stress_ratio(500.0, 0.0)

    def test_raises_sigma_v_eff_negative(self):
        with pytest.raises(ValueError, match="sigma_v_eff must be positive"):
            cyclic_stress_ratio(500.0, -100.0)


# ---------------------------------------------------------------------------
# load_increment_ratio  (Equation 3-4)
# LIR = delta_sigma / sigma_0
# ---------------------------------------------------------------------------

class TestLoadIncrementRatio:
    """Tests for load_increment_ratio()."""

    def test_doubling_load(self):
        # LIR = 1000 / 1000 = 1.0 (load doubles)
        result = load_increment_ratio(1000.0, 1000.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_unloading(self):
        # LIR = -750 / 1000 = -0.75 (common unloading increment)
        result = load_increment_ratio(-750.0, 1000.0)
        assert result == pytest.approx(-0.75, rel=1e-4)

    def test_reloading(self):
        # LIR = 4000 / 1000 = 4.0 (common reloading increment)
        result = load_increment_ratio(4000.0, 1000.0)
        assert result == pytest.approx(4.0, rel=1e-4)

    def test_zero_increment(self):
        # LIR = 0 / 500 = 0.0 (no load change)
        result = load_increment_ratio(0.0, 500.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_fractional_loading(self):
        # LIR = 250 / 1000 = 0.25
        result = load_increment_ratio(250.0, 1000.0)
        assert result == pytest.approx(0.25, rel=1e-4)

    def test_raises_sigma_0_zero(self):
        with pytest.raises(ValueError, match="sigma_0 must be positive"):
            load_increment_ratio(1000.0, 0.0)

    def test_raises_sigma_0_negative(self):
        with pytest.raises(ValueError, match="sigma_0 must be positive"):
            load_increment_ratio(1000.0, -500.0)


# ---------------------------------------------------------------------------
# hydraulic_conductivity_from_consolidation  (Equation 3-5)
# k = c_v * m_v * gamma_w
# ---------------------------------------------------------------------------

class TestHydraulicConductivityFromConsolidation:
    """Tests for hydraulic_conductivity_from_consolidation()."""

    def test_basic_valid(self):
        # k = 0.01 * 0.0005 * 9.81 = 5.0e-6 * 9.81 = 4.905e-5
        result = hydraulic_conductivity_from_consolidation(0.01, 0.0005, 9.81)
        assert result == pytest.approx(4.905e-5, rel=1e-4)

    def test_typical_clay_values(self):
        # c_v = 0.001 cm^2/s, m_v = 0.001 1/(g/cm^2), gamma_w = 1.0 g/cm^3
        # k = 0.001 * 0.001 * 1.0 = 1.0e-6
        result = hydraulic_conductivity_from_consolidation(0.001, 0.001, 1.0)
        assert result == pytest.approx(1.0e-6, rel=1e-4)

    def test_imperial_units(self):
        # c_v = 0.5 ft^2/day, m_v = 0.0001 1/psf, gamma_w = 62.4 pcf
        # k = 0.5 * 0.0001 * 62.4 = 0.00312
        result = hydraulic_conductivity_from_consolidation(0.5, 0.0001, 62.4)
        assert result == pytest.approx(0.00312, rel=1e-4)

    def test_c_v_zero(self):
        # k = 0 * 0.0005 * 9.81 = 0.0
        result = hydraulic_conductivity_from_consolidation(0.0, 0.0005, 9.81)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_m_v_zero(self):
        # k = 0.01 * 0 * 9.81 = 0.0
        result = hydraulic_conductivity_from_consolidation(0.01, 0.0, 9.81)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_large_values(self):
        # k = 1.0 * 1.0 * 9.81 = 9.81
        result = hydraulic_conductivity_from_consolidation(1.0, 1.0, 9.81)
        assert result == pytest.approx(9.81, rel=1e-4)

    def test_raises_c_v_negative(self):
        with pytest.raises(ValueError, match="c_v must be non-negative"):
            hydraulic_conductivity_from_consolidation(-0.01, 0.0005, 9.81)

    def test_raises_m_v_negative(self):
        with pytest.raises(ValueError, match="m_v must be non-negative"):
            hydraulic_conductivity_from_consolidation(0.01, -0.0005, 9.81)

    def test_raises_gamma_w_zero(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            hydraulic_conductivity_from_consolidation(0.01, 0.0005, 0.0)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            hydraulic_conductivity_from_consolidation(0.01, 0.0005, -9.81)
