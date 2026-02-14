"""Tests for geotech.dm7_1.chapter2 -- Equations 2-1 through 2-3.

Each public function is tested for:
  - basic valid input with a hand-calculated expected result
  - edge cases (zero torque, equal strengths, etc.)
  - every ValueError validation branch
"""

import math

import pytest

from geotech.dm7_1.chapter2 import *


# ---------------------------------------------------------------------------
# undrained_shear_strength_vane  (Equation 2-1)
#   s_u,fv = 6 * T_max / (7 * pi * D^3)
# ---------------------------------------------------------------------------


def test_undrained_shear_strength_vane_basic():
    """Hand-calc: T_max=100, D=0.5
    6*100 / (7*pi*0.5^3) = 600 / (7*pi*0.125)
                          = 600 / (0.875*pi)
                          = 600 / 2.748893571891069
                          = 218.268937...
    """
    expected = 6.0 * 100.0 / (7.0 * math.pi * 0.5 ** 3)
    result = undrained_shear_strength_vane(T_max=100.0, D=0.5)
    assert result == pytest.approx(expected, rel=1e-4)


def test_undrained_shear_strength_vane_unit_inputs():
    """Hand-calc: T_max=1, D=1
    6*1 / (7*pi*1) = 6 / (7*pi) = 0.27282...
    """
    expected = 6.0 / (7.0 * math.pi)
    result = undrained_shear_strength_vane(T_max=1.0, D=1.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_undrained_shear_strength_vane_large_diameter():
    """Hand-calc: T_max=50, D=2.0
    6*50 / (7*pi*8) = 300 / (56*pi) = 300 / 175.9291886... = 1.70484...
    """
    expected = 300.0 / (56.0 * math.pi)
    result = undrained_shear_strength_vane(T_max=50.0, D=2.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_undrained_shear_strength_vane_zero_torque():
    """Edge case: T_max=0 is allowed and should return 0."""
    result = undrained_shear_strength_vane(T_max=0.0, D=1.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_undrained_shear_strength_vane_small_diameter():
    """Hand-calc: T_max=0.01, D=0.065 (65 mm vane in metres).
    6*0.01 / (7*pi*0.065^3)
    = 0.06 / (7*pi*0.000274625)
    = 0.06 / 0.006040696...
    = 9.930...
    """
    expected = 0.06 / (7.0 * math.pi * 0.065 ** 3)
    result = undrained_shear_strength_vane(T_max=0.01, D=0.065)
    assert result == pytest.approx(expected, rel=1e-4)


def test_undrained_shear_strength_vane_raises_on_zero_diameter():
    """D = 0 must raise ValueError."""
    with pytest.raises(ValueError, match="D must be positive"):
        undrained_shear_strength_vane(T_max=100.0, D=0.0)


def test_undrained_shear_strength_vane_raises_on_negative_diameter():
    """D < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="D must be positive"):
        undrained_shear_strength_vane(T_max=100.0, D=-0.5)


def test_undrained_shear_strength_vane_raises_on_negative_torque():
    """T_max < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="T_max must be non-negative"):
        undrained_shear_strength_vane(T_max=-10.0, D=0.5)


def test_undrained_shear_strength_vane_negative_D_checked_before_T():
    """Both inputs invalid -- D <= 0 is checked first."""
    with pytest.raises(ValueError, match="D must be positive"):
        undrained_shear_strength_vane(T_max=-5.0, D=-1.0)


# ---------------------------------------------------------------------------
# sensitivity_vane  (Equation 2-2)
#   S_t,fv = s_u_fv / s_ur_fv
# ---------------------------------------------------------------------------


def test_sensitivity_vane_basic():
    """Hand-calc: 100 / 10 = 10.0"""
    result = sensitivity_vane(s_u_fv=100.0, s_ur_fv=10.0)
    assert result == pytest.approx(10.0, rel=1e-4)


def test_sensitivity_vane_insensitive_soil():
    """Sensitivity = 1 when peak equals remolded."""
    result = sensitivity_vane(s_u_fv=50.0, s_ur_fv=50.0)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_sensitivity_vane_high_sensitivity():
    """Quick-clay-like: 200 / 5 = 40.0"""
    result = sensitivity_vane(s_u_fv=200.0, s_ur_fv=5.0)
    assert result == pytest.approx(40.0, rel=1e-4)


def test_sensitivity_vane_fractional_result():
    """Hand-calc: 75 / 30 = 2.5"""
    result = sensitivity_vane(s_u_fv=75.0, s_ur_fv=30.0)
    assert result == pytest.approx(2.5, rel=1e-4)


def test_sensitivity_vane_zero_peak_strength():
    """Edge case: s_u_fv = 0 is allowed and returns 0."""
    result = sensitivity_vane(s_u_fv=0.0, s_ur_fv=10.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_sensitivity_vane_raises_on_zero_remolded():
    """s_ur_fv = 0 must raise ValueError (division by zero)."""
    with pytest.raises(ValueError, match="s_ur_fv must be positive"):
        sensitivity_vane(s_u_fv=100.0, s_ur_fv=0.0)


def test_sensitivity_vane_raises_on_negative_remolded():
    """s_ur_fv < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="s_ur_fv must be positive"):
        sensitivity_vane(s_u_fv=100.0, s_ur_fv=-5.0)


def test_sensitivity_vane_raises_on_negative_peak():
    """s_u_fv < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="s_u_fv must be non-negative"):
        sensitivity_vane(s_u_fv=-10.0, s_ur_fv=5.0)


def test_sensitivity_vane_negative_remolded_checked_before_peak():
    """Both inputs invalid -- s_ur_fv <= 0 is checked first."""
    with pytest.raises(ValueError, match="s_ur_fv must be positive"):
        sensitivity_vane(s_u_fv=-10.0, s_ur_fv=-5.0)


# ---------------------------------------------------------------------------
# corrected_undrained_shear_strength_vane  (Equation 2-3)
#   s_u,field = s_u_fv * mu_R
# ---------------------------------------------------------------------------


def test_corrected_undrained_shear_strength_vane_basic():
    """Hand-calc: 500 * 0.8 = 400.0"""
    result = corrected_undrained_shear_strength_vane(s_u_fv=500.0, mu_R=0.8)
    assert result == pytest.approx(400.0, rel=1e-4)


def test_corrected_undrained_shear_strength_vane_factor_one():
    """mu_R = 1.0 means no correction; result equals input strength."""
    result = corrected_undrained_shear_strength_vane(s_u_fv=250.0, mu_R=1.0)
    assert result == pytest.approx(250.0, rel=1e-4)


def test_corrected_undrained_shear_strength_vane_small_factor():
    """Hand-calc: 1000 * 0.5 = 500.0"""
    result = corrected_undrained_shear_strength_vane(s_u_fv=1000.0, mu_R=0.5)
    assert result == pytest.approx(500.0, rel=1e-4)


def test_corrected_undrained_shear_strength_vane_large_factor():
    """mu_R > 1.0 is technically allowed: 300 * 1.2 = 360.0"""
    result = corrected_undrained_shear_strength_vane(s_u_fv=300.0, mu_R=1.2)
    assert result == pytest.approx(360.0, rel=1e-4)


def test_corrected_undrained_shear_strength_vane_zero_strength():
    """Edge case: s_u_fv = 0 is allowed and returns 0."""
    result = corrected_undrained_shear_strength_vane(s_u_fv=0.0, mu_R=0.8)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_corrected_undrained_shear_strength_vane_raises_on_negative_strength():
    """s_u_fv < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="s_u_fv must be non-negative"):
        corrected_undrained_shear_strength_vane(s_u_fv=-100.0, mu_R=0.8)


def test_corrected_undrained_shear_strength_vane_raises_on_zero_factor():
    """mu_R = 0 must raise ValueError."""
    with pytest.raises(ValueError, match="mu_R must be positive"):
        corrected_undrained_shear_strength_vane(s_u_fv=500.0, mu_R=0.0)


def test_corrected_undrained_shear_strength_vane_raises_on_negative_factor():
    """mu_R < 0 must raise ValueError."""
    with pytest.raises(ValueError, match="mu_R must be positive"):
        corrected_undrained_shear_strength_vane(s_u_fv=500.0, mu_R=-0.5)


def test_corrected_undrained_shear_strength_vane_negative_strength_checked_first():
    """Both inputs invalid -- s_u_fv < 0 is checked first."""
    with pytest.raises(ValueError, match="s_u_fv must be non-negative"):
        corrected_undrained_shear_strength_vane(s_u_fv=-100.0, mu_R=-0.5)


# ---------------------------------------------------------------------------
# Integration-style tests: chain Equations 2-1 through 2-3
# ---------------------------------------------------------------------------


def test_chain_vane_strength_to_corrected():
    """Compute s_u,fv from Eq 2-1, then correct it with Eq 2-3.
    T_max=50, D=0.5, mu_R=0.85
    s_u,fv = 6*50 / (7*pi*0.125) = 300 / (0.875*pi) = 109.13446...
    s_u,field = 109.13446... * 0.85 = 92.76429...
    """
    s_u_fv = undrained_shear_strength_vane(T_max=50.0, D=0.5)
    expected_su = 300.0 / (0.875 * math.pi)
    assert s_u_fv == pytest.approx(expected_su, rel=1e-4)

    s_u_field = corrected_undrained_shear_strength_vane(s_u_fv, mu_R=0.85)
    expected_field = expected_su * 0.85
    assert s_u_field == pytest.approx(expected_field, rel=1e-4)


def test_chain_vane_strength_to_sensitivity():
    """Compute peak and remolded s_u from Eq 2-1, then sensitivity from Eq 2-2.
    T_max=80, T_res=10, D=0.5
    s_u,fv  = 6*80 / (7*pi*0.125) = 480 / (0.875*pi) = 174.6151...
    s_ur,fv = 6*10 / (7*pi*0.125) =  60 / (0.875*pi) =  21.8269...
    S_t     = 174.6151 / 21.8269  = 8.0
    """
    D = 0.5
    s_u_fv = undrained_shear_strength_vane(T_max=80.0, D=D)
    s_ur_fv = undrained_shear_strength_vane(T_max=10.0, D=D)
    S_t = sensitivity_vane(s_u_fv, s_ur_fv)
    # Sensitivity should equal T_max / T_res = 80 / 10 = 8.0
    assert S_t == pytest.approx(8.0, rel=1e-4)
