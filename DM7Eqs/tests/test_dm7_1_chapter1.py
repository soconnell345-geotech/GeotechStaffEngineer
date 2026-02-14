"""Tests for geotech.dm7_1.chapter1 — Identification and Classification of Soil and Rock.

Each public function is tested for:
  - Basic valid input with a hand-calculated expected result
  - Edge cases where applicable
  - ValueError checks covering every validation branch
"""

import math

import pytest

from geotech.dm7_1.chapter1 import *


# ---------------------------------------------------------------------------
# coefficient_of_uniformity  (Equation 1-1)
#   Cu = D60 / D10
# ---------------------------------------------------------------------------

def test_coefficient_of_uniformity_basic():
    # Cu = 6.0 / 1.5 = 4.0
    result = coefficient_of_uniformity(6.0, 1.5)
    assert result == pytest.approx(4.0, rel=1e-4)


def test_coefficient_of_uniformity_large_spread():
    # Cu = 20.0 / 0.5 = 40.0
    result = coefficient_of_uniformity(20.0, 0.5)
    assert result == pytest.approx(40.0, rel=1e-4)


def test_coefficient_of_uniformity_equal_diameters():
    # Cu = 2.0 / 2.0 = 1.0  (uniform soil)
    result = coefficient_of_uniformity(2.0, 2.0)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_coefficient_of_uniformity_d60_zero():
    # D60 = 0 is allowed (non-negative), Cu = 0.0 / 1.0 = 0.0
    result = coefficient_of_uniformity(0.0, 1.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_coefficient_of_uniformity_d10_zero_raises():
    with pytest.raises(ValueError, match="D10 must be positive"):
        coefficient_of_uniformity(6.0, 0.0)


def test_coefficient_of_uniformity_d10_negative_raises():
    with pytest.raises(ValueError, match="D10 must be positive"):
        coefficient_of_uniformity(6.0, -1.0)


def test_coefficient_of_uniformity_d60_negative_raises():
    with pytest.raises(ValueError, match="D60 must be non-negative"):
        coefficient_of_uniformity(-1.0, 1.0)


# ---------------------------------------------------------------------------
# coefficient_of_curvature  (Equation 1-2)
#   Cc = D30^2 / (D60 * D10)
# ---------------------------------------------------------------------------

def test_coefficient_of_curvature_basic():
    # Cc = 3.0^2 / (6.0 * 1.0) = 9.0 / 6.0 = 1.5
    result = coefficient_of_curvature(6.0, 3.0, 1.0)
    assert result == pytest.approx(1.5, rel=1e-4)


def test_coefficient_of_curvature_well_graded():
    # Cc = 4.0^2 / (10.0 * 1.0) = 16.0 / 10.0 = 1.6
    result = coefficient_of_curvature(10.0, 4.0, 1.0)
    assert result == pytest.approx(1.6, rel=1e-4)


def test_coefficient_of_curvature_d30_zero():
    # D30 = 0 is allowed, Cc = 0^2 / (6.0 * 1.0) = 0.0
    result = coefficient_of_curvature(6.0, 0.0, 1.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_coefficient_of_curvature_fractional():
    # Cc = 2.5^2 / (8.0 * 0.5) = 6.25 / 4.0 = 1.5625
    result = coefficient_of_curvature(8.0, 2.5, 0.5)
    assert result == pytest.approx(1.5625, rel=1e-4)


def test_coefficient_of_curvature_d60_zero_raises():
    with pytest.raises(ValueError, match="D60 must be positive"):
        coefficient_of_curvature(0.0, 3.0, 1.0)


def test_coefficient_of_curvature_d60_negative_raises():
    with pytest.raises(ValueError, match="D60 must be positive"):
        coefficient_of_curvature(-1.0, 3.0, 1.0)


def test_coefficient_of_curvature_d10_zero_raises():
    with pytest.raises(ValueError, match="D10 must be positive"):
        coefficient_of_curvature(6.0, 3.0, 0.0)


def test_coefficient_of_curvature_d10_negative_raises():
    with pytest.raises(ValueError, match="D10 must be positive"):
        coefficient_of_curvature(6.0, 3.0, -1.0)


def test_coefficient_of_curvature_d30_negative_raises():
    with pytest.raises(ValueError, match="D30 must be non-negative"):
        coefficient_of_curvature(6.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# aashto_group_index  (Equation 1-3 — full)
#   GI = (F - 35)(0.2 + 0.005(LL - 40)) + 0.01(F - 15)(PI - 10)
#   Returns max(GI, 0.0)
# ---------------------------------------------------------------------------

def test_aashto_group_index_basic():
    # term1 = (55 - 35)(0.2 + 0.005(50 - 40)) = 20 * 0.25 = 5.0
    # term2 = 0.01 * (55 - 15) * (25 - 10) = 0.01 * 40 * 15 = 6.0
    # GI = 5.0 + 6.0 = 11.0
    result = aashto_group_index(55.0, 50.0, 25.0)
    assert result == pytest.approx(11.0, rel=1e-4)


def test_aashto_group_index_second_example():
    # term1 = (40 - 35)(0.2 + 0.005(45 - 40)) = 5 * 0.225 = 1.125
    # term2 = 0.01 * (40 - 15) * (20 - 10) = 0.01 * 25 * 10 = 2.5
    # GI = 1.125 + 2.5 = 3.625
    result = aashto_group_index(40.0, 45.0, 20.0)
    assert result == pytest.approx(3.625, rel=1e-4)


def test_aashto_group_index_negative_clamped_to_zero():
    # With low F, low LL, and low PI the calculated GI will be negative.
    # term1 = (10 - 35)(0.2 + 0.005(20 - 40)) = (-25)(0.2 + (-0.1)) = (-25)(0.1) = -2.5
    # term2 = 0.01 * (10 - 15) * (5 - 10) = 0.01 * (-5) * (-5) = 0.25
    # GI = -2.5 + 0.25 = -2.25 -> clamped to 0.0
    result = aashto_group_index(10.0, 20.0, 5.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_aashto_group_index_zero_result():
    # F=35, LL=40, PI=10:
    # term1 = (35-35)(0.2+0.005*(40-40)) = 0 * 0.2 = 0.0
    # term2 = 0.01*(35-15)*(10-10) = 0.01*20*0 = 0.0
    # GI = 0.0
    result = aashto_group_index(35.0, 40.0, 10.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_aashto_group_index_high_values():
    # F=75, LL=80, PI=50
    # term1 = (75-35)(0.2+0.005*(80-40)) = 40*(0.2+0.2) = 40*0.4 = 16.0
    # term2 = 0.01*(75-15)*(50-10) = 0.01*60*40 = 24.0
    # GI = 16.0 + 24.0 = 40.0
    result = aashto_group_index(75.0, 80.0, 50.0)
    assert result == pytest.approx(40.0, rel=1e-4)


# ---------------------------------------------------------------------------
# aashto_group_index_a2  (Equation 1-3 — partial, for A-2-6 / A-2-7)
#   GI = 0.01 * (F - 15) * (PI - 10)
#   Returns max(GI, 0.0)
# ---------------------------------------------------------------------------

def test_aashto_group_index_a2_basic():
    # GI = 0.01 * (30 - 15) * (20 - 10) = 0.01 * 15 * 10 = 1.5
    result = aashto_group_index_a2(30.0, 20.0)
    assert result == pytest.approx(1.5, rel=1e-4)


def test_aashto_group_index_a2_second_example():
    # GI = 0.01 * (25 - 15) * (15 - 10) = 0.01 * 10 * 5 = 0.5
    result = aashto_group_index_a2(25.0, 15.0)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_aashto_group_index_a2_negative_clamped_to_zero():
    # GI = 0.01 * (10 - 15) * (20 - 10) = 0.01 * (-5) * 10 = -0.5 -> 0.0
    result = aashto_group_index_a2(10.0, 20.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_aashto_group_index_a2_pi_below_10_clamped():
    # GI = 0.01 * (30 - 15) * (5 - 10) = 0.01 * 15 * (-5) = -0.75 -> 0.0
    result = aashto_group_index_a2(30.0, 5.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_aashto_group_index_a2_at_boundary():
    # GI = 0.01 * (15 - 15) * (10 - 10) = 0.0
    result = aashto_group_index_a2(15.0, 10.0)
    assert result == pytest.approx(0.0, rel=1e-4)


# ---------------------------------------------------------------------------
# size_correction_factor  (Equation 1-5)
#   F = sqrt(D_e / 50)
# ---------------------------------------------------------------------------

def test_size_correction_factor_at_50mm():
    # F = sqrt(50 / 50) = sqrt(1) = 1.0
    result = size_correction_factor(50.0)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_size_correction_factor_at_200mm():
    # F = sqrt(200 / 50) = sqrt(4) = 2.0
    result = size_correction_factor(200.0)
    assert result == pytest.approx(2.0, rel=1e-4)


def test_size_correction_factor_at_12_5mm():
    # F = sqrt(12.5 / 50) = sqrt(0.25) = 0.5
    result = size_correction_factor(12.5)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_size_correction_factor_zero():
    # F = sqrt(0 / 50) = 0.0
    result = size_correction_factor(0.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_size_correction_factor_arbitrary():
    # F = sqrt(100 / 50) = sqrt(2) = 1.41421356...
    result = size_correction_factor(100.0)
    assert result == pytest.approx(math.sqrt(2.0), rel=1e-4)


def test_size_correction_factor_negative_raises():
    with pytest.raises(ValueError, match="D_e must be non-negative"):
        size_correction_factor(-1.0)


# ---------------------------------------------------------------------------
# point_load_strength_index  (Equations 1-4 and 1-5)
#   Is(50) = F * P / d^2    where F = sqrt(D_e / 50)
# ---------------------------------------------------------------------------

def test_point_load_strength_index_basic():
    # D_e=50 => F=1.0
    # Is(50) = 1.0 * 5.0 / 50.0^2 = 5.0 / 2500.0 = 0.002
    result = point_load_strength_index(5.0, 50.0, 50.0)
    assert result == pytest.approx(0.002, rel=1e-4)


def test_point_load_strength_index_large_diameter():
    # D_e=200 => F=sqrt(200/50)=2.0
    # Is(50) = 2.0 * 10.0 / 100.0^2 = 20.0 / 10000.0 = 0.002
    result = point_load_strength_index(10.0, 100.0, 200.0)
    assert result == pytest.approx(0.002, rel=1e-4)


def test_point_load_strength_index_small_diameter():
    # D_e=12.5 => F=sqrt(12.5/50)=sqrt(0.25)=0.5
    # Is(50) = 0.5 * 8.0 / 25.0^2 = 4.0 / 625.0 = 0.0064
    result = point_load_strength_index(8.0, 25.0, 12.5)
    assert result == pytest.approx(0.0064, rel=1e-4)


def test_point_load_strength_index_zero_force():
    # P=0 => Is(50) = F * 0 / d^2 = 0.0
    result = point_load_strength_index(0.0, 50.0, 50.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_point_load_strength_index_d_zero_raises():
    with pytest.raises(ValueError, match="d must be positive"):
        point_load_strength_index(5.0, 0.0, 50.0)


def test_point_load_strength_index_d_negative_raises():
    with pytest.raises(ValueError, match="d must be positive"):
        point_load_strength_index(5.0, -1.0, 50.0)


def test_point_load_strength_index_de_negative_raises():
    with pytest.raises(ValueError, match="D_e must be non-negative"):
        point_load_strength_index(5.0, 50.0, -1.0)


# ---------------------------------------------------------------------------
# expansion_index  (Equation 1-6)
#   EI = (delta_H / H_i) * 1000
# ---------------------------------------------------------------------------

def test_expansion_index_basic():
    # EI = (0.05 / 1.0) * 1000 = 50.0  (Low expansion)
    result = expansion_index(0.05, 1.0)
    assert result == pytest.approx(50.0, rel=1e-4)


def test_expansion_index_high_expansion():
    # EI = (0.12 / 1.0) * 1000 = 120.0  (High expansion)
    result = expansion_index(0.12, 1.0)
    assert result == pytest.approx(120.0, rel=1e-4)


def test_expansion_index_very_low():
    # EI = (0.005 / 0.5) * 1000 = 10.0  (Very low)
    result = expansion_index(0.005, 0.5)
    assert result == pytest.approx(10.0, rel=1e-4)


def test_expansion_index_zero_change():
    # EI = (0.0 / 1.0) * 1000 = 0.0
    result = expansion_index(0.0, 1.0)
    assert result == pytest.approx(0.0, rel=1e-4)


def test_expansion_index_fractional():
    # EI = (0.075 / 1.25) * 1000 = 0.06 * 1000 = 60.0
    result = expansion_index(0.075, 1.25)
    assert result == pytest.approx(60.0, rel=1e-4)


def test_expansion_index_hi_zero_raises():
    with pytest.raises(ValueError, match="H_i must be positive"):
        expansion_index(0.05, 0.0)


def test_expansion_index_hi_negative_raises():
    with pytest.raises(ValueError, match="H_i must be positive"):
        expansion_index(0.05, -1.0)


def test_expansion_index_delta_h_negative_raises():
    with pytest.raises(ValueError, match="delta_H must be non-negative"):
        expansion_index(-0.01, 1.0)
