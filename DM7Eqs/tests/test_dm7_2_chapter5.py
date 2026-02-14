"""Tests for geotech.dm7_2.chapter5 -- Equations 5-1 through 5-42.

Each public function is tested for:
  - basic valid input with a hand-calculated expected result
  - edge cases where applicable
  - every ValueError validation branch
"""

import math

import pytest

from geotech.dm7_2.chapter5 import *


# ===========================================================================
# gross_bearing_pressure  (Equation 5-1)
#   q_gross = (Q_DL_LL + W_F + W_S) / A
# ===========================================================================


def test_gross_bearing_pressure_basic():
    """Hand-calc: (100 + 20 + 30) / 10 = 150 / 10 = 15.0"""
    result = gross_bearing_pressure(Q_DL_LL=100.0, W_F=20.0, W_S=30.0, A=10.0)
    assert result == pytest.approx(15.0, rel=1e-4)


def test_gross_bearing_pressure_zero_loads():
    """Edge: all loads zero -> 0.0"""
    result = gross_bearing_pressure(0.0, 0.0, 0.0, A=5.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_gross_bearing_pressure_raises_zero_area():
    with pytest.raises(ValueError, match="area A must be positive"):
        gross_bearing_pressure(100.0, 10.0, 10.0, A=0.0)


def test_gross_bearing_pressure_raises_negative_area():
    with pytest.raises(ValueError, match="area A must be positive"):
        gross_bearing_pressure(100.0, 10.0, 10.0, A=-5.0)


# ===========================================================================
# net_bearing_pressure_from_ultimate  (Equation 5-2)
#   q_net = q_ult - sigma_zD
# ===========================================================================


def test_net_bearing_pressure_from_ultimate_basic():
    """Hand-calc: 500 - 120 = 380"""
    result = net_bearing_pressure_from_ultimate(q_ult=500.0, sigma_zD=120.0)
    assert result == pytest.approx(380.0, rel=1e-4)


def test_net_bearing_pressure_from_ultimate_equal():
    """Edge: q_ult == sigma_zD -> 0"""
    result = net_bearing_pressure_from_ultimate(200.0, 200.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# net_bearing_pressure  (Equation 5-3)
#   q_net ~ Q_DL_LL / A
# ===========================================================================


def test_net_bearing_pressure_basic():
    """Hand-calc: 200 / 8 = 25.0"""
    result = net_bearing_pressure(Q_DL_LL=200.0, A=8.0)
    assert result == pytest.approx(25.0, rel=1e-4)


def test_net_bearing_pressure_zero_load():
    """Edge: zero load -> 0"""
    result = net_bearing_pressure(0.0, 4.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_net_bearing_pressure_raises_zero_area():
    with pytest.raises(ValueError, match="area A must be positive"):
        net_bearing_pressure(100.0, A=0.0)


def test_net_bearing_pressure_raises_negative_area():
    with pytest.raises(ValueError, match="area A must be positive"):
        net_bearing_pressure(100.0, A=-1.0)


# ===========================================================================
# eccentricity  (Equation 5-4)
#   e = M / Q
# ===========================================================================


def test_eccentricity_basic():
    """Hand-calc: 50 / 200 = 0.25"""
    result = eccentricity(M=50.0, Q=200.0)
    assert result == pytest.approx(0.25, rel=1e-4)


def test_eccentricity_zero_moment():
    """Edge: M=0 -> e=0"""
    result = eccentricity(M=0.0, Q=100.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_eccentricity_raises_zero_Q():
    with pytest.raises(ValueError, match="Q must be positive"):
        eccentricity(50.0, Q=0.0)


def test_eccentricity_raises_negative_Q():
    with pytest.raises(ValueError, match="Q must be positive"):
        eccentricity(50.0, Q=-10.0)


# ===========================================================================
# check_eccentricity_one_direction  (Equation 5-5)
#   e <= dimension / 6
# ===========================================================================


def test_check_eccentricity_one_direction_within():
    """Hand-calc: e=0.5, dim=6 -> 6/6=1.0 -> 0.5<=1.0 => True"""
    assert check_eccentricity_one_direction(e=0.5, dimension=6.0) is True


def test_check_eccentricity_one_direction_at_limit():
    """Edge: e exactly = dim/6 -> True"""
    assert check_eccentricity_one_direction(e=1.0, dimension=6.0) is True


def test_check_eccentricity_one_direction_outside():
    """Hand-calc: e=1.5, dim=6 -> 1.5 > 1.0 => False"""
    assert check_eccentricity_one_direction(e=1.5, dimension=6.0) is False


def test_check_eccentricity_one_direction_negative_e():
    """Edge: negative e uses abs -> abs(-0.5)=0.5 <= 1.0 => True"""
    assert check_eccentricity_one_direction(e=-0.5, dimension=6.0) is True


def test_check_eccentricity_one_direction_raises_nonpositive_dim():
    with pytest.raises(ValueError, match="dimension must be positive"):
        check_eccentricity_one_direction(0.5, dimension=0.0)


def test_check_eccentricity_one_direction_raises_negative_dim():
    with pytest.raises(ValueError, match="dimension must be positive"):
        check_eccentricity_one_direction(0.5, dimension=-3.0)


# ===========================================================================
# check_eccentricity_two_directions  (Equation 5-6)
#   6*|eB|/B + 6*|eL|/L <= 1
# ===========================================================================


def test_check_eccentricity_two_directions_within():
    """Hand-calc: 6*0.1/6 + 6*0.1/12 = 0.1 + 0.05 = 0.15 <= 1 => True"""
    assert check_eccentricity_two_directions(0.1, 0.1, B=6.0, L=12.0) is True


def test_check_eccentricity_two_directions_outside():
    """Hand-calc: 6*1.0/6 + 6*1.0/6 = 1.0 + 1.0 = 2.0 > 1 => False"""
    assert check_eccentricity_two_directions(1.0, 1.0, 6.0, 6.0) is False


def test_check_eccentricity_two_directions_at_limit():
    """Hand-calc: 6*eB/B + 6*eL/L = 1.0. Use eB=B/12, eL=L/12.
    6*(B/12)/B + 6*(L/12)/L = 0.5 + 0.5 = 1.0 => True"""
    assert check_eccentricity_two_directions(0.5, 0.5, 6.0, 6.0) is True


def test_check_eccentricity_two_directions_raises_zero_B():
    with pytest.raises(ValueError, match="B and L must be positive"):
        check_eccentricity_two_directions(0.1, 0.1, B=0.0, L=6.0)


def test_check_eccentricity_two_directions_raises_negative_L():
    with pytest.raises(ValueError, match="B and L must be positive"):
        check_eccentricity_two_directions(0.1, 0.1, B=6.0, L=-1.0)


# ===========================================================================
# corner_bearing_pressure  (Equation 5-7)
#   q_corner = q_gross * (1 +/- 6eB/B +/- 6eL/L)
# ===========================================================================


def test_corner_bearing_pressure_basic():
    """Hand-calc: q_gross=10, eB=0.5, eL=1.0, B=6, L=12
    term_B = 6*0.5/6 = 0.5, term_L = 6*1.0/12 = 0.5
    q1 = 10*(1+0.5+0.5) = 20
    q2 = 10*(1+0.5-0.5) = 10
    q3 = 10*(1-0.5+0.5) = 10
    q4 = 10*(1-0.5-0.5) = 0
    """
    q1, q2, q3, q4 = corner_bearing_pressure(10.0, 0.5, 1.0, 6.0, 12.0)
    assert q1 == pytest.approx(20.0, rel=1e-4)
    assert q2 == pytest.approx(10.0, rel=1e-4)
    assert q3 == pytest.approx(10.0, rel=1e-4)
    assert q4 == pytest.approx(0.0, abs=1e-12)


def test_corner_bearing_pressure_no_eccentricity():
    """Edge: eB=eL=0 -> all corners = q_gross"""
    q1, q2, q3, q4 = corner_bearing_pressure(15.0, 0.0, 0.0, 4.0, 8.0)
    assert q1 == pytest.approx(15.0, rel=1e-4)
    assert q4 == pytest.approx(15.0, rel=1e-4)


def test_corner_bearing_pressure_raises_zero_B():
    with pytest.raises(ValueError, match="B and L must be positive"):
        corner_bearing_pressure(10.0, 0.5, 0.5, B=0.0, L=6.0)


def test_corner_bearing_pressure_raises_negative_L():
    with pytest.raises(ValueError, match="B and L must be positive"):
        corner_bearing_pressure(10.0, 0.5, 0.5, B=6.0, L=-1.0)


# ===========================================================================
# effective_foundation_width  (Equation 5-8)
#   B' = B - 2*|eB|
# ===========================================================================


def test_effective_foundation_width_basic():
    """Hand-calc: 6 - 2*0.5 = 5.0"""
    result = effective_foundation_width(B=6.0, e_B=0.5)
    assert result == pytest.approx(5.0, rel=1e-4)


def test_effective_foundation_width_negative_ecc():
    """Edge: negative eccentricity -> abs used -> 6 - 2*1.0 = 4.0"""
    result = effective_foundation_width(B=6.0, e_B=-1.0)
    assert result == pytest.approx(4.0, rel=1e-4)


def test_effective_foundation_width_raises_too_large():
    """eB = B/2 -> B'=0 -> non-positive -> ValueError"""
    with pytest.raises(ValueError, match="non-positive"):
        effective_foundation_width(B=6.0, e_B=3.0)


def test_effective_foundation_width_raises_exceeds():
    """eB > B/2 -> B' negative -> ValueError"""
    with pytest.raises(ValueError, match="non-positive"):
        effective_foundation_width(B=6.0, e_B=4.0)


# ===========================================================================
# effective_foundation_length  (Equation 5-9)
#   L' = L - 2*|eL|
# ===========================================================================


def test_effective_foundation_length_basic():
    """Hand-calc: 12 - 2*1 = 10.0"""
    result = effective_foundation_length(L=12.0, e_L=1.0)
    assert result == pytest.approx(10.0, rel=1e-4)


def test_effective_foundation_length_negative_ecc():
    """Edge: negative eccentricity -> abs -> 12 - 2*2 = 8.0"""
    result = effective_foundation_length(L=12.0, e_L=-2.0)
    assert result == pytest.approx(8.0, rel=1e-4)


def test_effective_foundation_length_raises_too_large():
    with pytest.raises(ValueError, match="non-positive"):
        effective_foundation_length(L=10.0, e_L=5.0)


# ===========================================================================
# equivalent_uniform_bearing_pressure  (Equation 5-10)
#   q_unif = (Q_DL_LL + W_F + W_S) / (B' * L')
# ===========================================================================


def test_equivalent_uniform_bearing_pressure_basic():
    """Hand-calc: (100+20+30) / (5*10) = 150/50 = 3.0"""
    result = equivalent_uniform_bearing_pressure(100.0, 20.0, 30.0, 5.0, 10.0)
    assert result == pytest.approx(3.0, rel=1e-4)


def test_equivalent_uniform_bearing_pressure_raises_zero_B():
    with pytest.raises(ValueError, match="must be positive"):
        equivalent_uniform_bearing_pressure(100.0, 10.0, 10.0, B_prime=0.0, L_prime=5.0)


def test_equivalent_uniform_bearing_pressure_raises_negative_L():
    with pytest.raises(ValueError, match="must be positive"):
        equivalent_uniform_bearing_pressure(100.0, 10.0, 10.0, B_prime=5.0, L_prime=-1.0)


# ===========================================================================
# circular_effective_length  (Equation 5-11)
#   L' = 2 * sqrt( r^2 * acos(ex/r) - ex * sqrt(r^2 - ex^2) )
# ===========================================================================


def test_circular_effective_length_basic():
    """Hand-calc: r=5, e_x=1
    area = 25*acos(0.2) - 1*sqrt(25-1) = 25*1.369438... - 1*4.898979...
         = 34.23596... - 4.89898... = 29.33698...
    L' = 2*sqrt(29.33698...) = 2*5.41636... = 10.83272...
    """
    r, ex = 5.0, 1.0
    area = r**2 * math.acos(ex / r) - ex * math.sqrt(r**2 - ex**2)
    expected = 2.0 * math.sqrt(area)
    result = circular_effective_length(r=5.0, e_x=1.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_circular_effective_length_zero_eccentricity():
    """Edge: e_x=0 -> area = r^2 * acos(0) = r^2*pi/2
    L' = 2*sqrt(r^2*pi/2) = 2*r*sqrt(pi/2)
    For r=3: L' = 6*sqrt(pi/2) = 6*1.2533... = 7.5198...
    """
    r = 3.0
    expected = 2.0 * r * math.sqrt(math.pi / 2.0)
    result = circular_effective_length(r=3.0, e_x=0.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_circular_effective_length_raises_nonpositive_r():
    with pytest.raises(ValueError, match="Radius r must be positive"):
        circular_effective_length(r=0.0, e_x=0.0)


def test_circular_effective_length_raises_negative_r():
    with pytest.raises(ValueError, match="Radius r must be positive"):
        circular_effective_length(r=-1.0, e_x=0.0)


def test_circular_effective_length_raises_ex_ge_r():
    with pytest.raises(ValueError, match="less than radius"):
        circular_effective_length(r=5.0, e_x=5.0)


def test_circular_effective_length_raises_ex_exceeds_r():
    with pytest.raises(ValueError, match="less than radius"):
        circular_effective_length(r=5.0, e_x=6.0)


# ===========================================================================
# circular_effective_width  (Equation 5-12)
#   B' = area / L'
# ===========================================================================


def test_circular_effective_width_basic():
    """Hand-calc: r=5, ex=1, L'=circular_effective_length(5,1)
    area = 25*acos(0.2) - 1*sqrt(24)
    B' = area / L'
    """
    r, ex = 5.0, 1.0
    area = r**2 * math.acos(ex / r) - ex * math.sqrt(r**2 - ex**2)
    L_prime = 2.0 * math.sqrt(area)
    expected = area / L_prime
    result = circular_effective_width(r=5.0, e_x=1.0, L_prime=L_prime)
    assert result == pytest.approx(expected, rel=1e-4)


def test_circular_effective_width_raises_nonpositive_r():
    with pytest.raises(ValueError, match="Radius r must be positive"):
        circular_effective_width(r=0.0, e_x=0.0, L_prime=5.0)


def test_circular_effective_width_raises_ex_ge_r():
    with pytest.raises(ValueError, match="less than radius"):
        circular_effective_width(r=5.0, e_x=5.0, L_prime=5.0)


def test_circular_effective_width_raises_nonpositive_L():
    with pytest.raises(ValueError, match="L' must be positive"):
        circular_effective_width(r=5.0, e_x=1.0, L_prime=0.0)


# ===========================================================================
# gross_allowable_bearing_pressure  (Equation 5-13)
#   q_all,gross = q_ult / F_BC
# ===========================================================================


def test_gross_allowable_bearing_pressure_basic():
    """Hand-calc: 1000 / 3 = 333.333..."""
    result = gross_allowable_bearing_pressure(q_ult=1000.0, F_BC=3.0)
    assert result == pytest.approx(1000.0 / 3.0, rel=1e-4)


def test_gross_allowable_bearing_pressure_fs_one():
    """Edge: F_BC=1 -> q_all = q_ult"""
    result = gross_allowable_bearing_pressure(q_ult=500.0, F_BC=1.0)
    assert result == pytest.approx(500.0, rel=1e-4)


def test_gross_allowable_bearing_pressure_raises_zero_fs():
    with pytest.raises(ValueError, match="F_BC must be positive"):
        gross_allowable_bearing_pressure(1000.0, F_BC=0.0)


def test_gross_allowable_bearing_pressure_raises_negative_fs():
    with pytest.raises(ValueError, match="F_BC must be positive"):
        gross_allowable_bearing_pressure(1000.0, F_BC=-2.0)


# ===========================================================================
# net_allowable_bearing_pressure  (Equation 5-14)
#   q_all,net = q_ult / F_BC - sigma_zD
# ===========================================================================


def test_net_allowable_bearing_pressure_basic():
    """Hand-calc: 1000/3 - 50 = 333.33... - 50 = 283.33..."""
    result = net_allowable_bearing_pressure(1000.0, 3.0, 50.0)
    assert result == pytest.approx(1000.0 / 3.0 - 50.0, rel=1e-4)


def test_net_allowable_bearing_pressure_raises_zero_fs():
    with pytest.raises(ValueError, match="F_BC must be positive"):
        net_allowable_bearing_pressure(1000.0, F_BC=0.0, sigma_zD=50.0)


def test_net_allowable_bearing_pressure_raises_negative_fs():
    with pytest.raises(ValueError, match="F_BC must be positive"):
        net_allowable_bearing_pressure(1000.0, F_BC=-1.0, sigma_zD=50.0)


# ===========================================================================
# ultimate_bearing_capacity_drained  (Equation 5-15)
#   q_ult = c'*Nc*psi_c + sigma'_zD*Nq*psi_q + 0.5*gamma*B*Ngamma*psi_gamma
# ===========================================================================


def test_ultimate_bearing_capacity_drained_basic():
    """Hand-calc: 200*14.8*1 + 500*6.4*1 + 0.5*120*4*5.0*1
    = 2960 + 3200 + 1200 = 7360
    """
    result = ultimate_bearing_capacity_drained(
        c_prime=200.0, sigma_zD_prime=500.0, gamma=120.0, B=4.0,
        Nc=14.8, Nq=6.4, N_gamma=5.0
    )
    assert result == pytest.approx(7360.0, rel=1e-4)


def test_ultimate_bearing_capacity_drained_with_psi():
    """Hand-calc: 100*10*1.2 + 300*5*0.9 + 0.5*110*3*4*1.1
    = 1200 + 1350 + 726 = 3276
    """
    result = ultimate_bearing_capacity_drained(
        c_prime=100.0, sigma_zD_prime=300.0, gamma=110.0, B=3.0,
        Nc=10.0, Nq=5.0, N_gamma=4.0,
        psi_c=1.2, psi_q=0.9, psi_gamma=1.1
    )
    assert result == pytest.approx(3276.0, rel=1e-4)


def test_ultimate_bearing_capacity_drained_zero_cohesion():
    """Edge: c=0 -> only overburden and weight terms.
    0 + 500*6.4*1 + 0.5*120*4*5*1 = 3200 + 1200 = 4400"""
    result = ultimate_bearing_capacity_drained(
        0.0, 500.0, 120.0, 4.0, 14.8, 6.4, 5.0
    )
    assert result == pytest.approx(4400.0, rel=1e-4)


# ===========================================================================
# ultimate_bearing_capacity_undrained_unsaturated  (Equation 5-16)
#   q_ult = c*Nc*psi_c + sigma_zD*Nq*psi_q + 0.5*gamma*B*Ngamma*psi_gamma
# ===========================================================================


def test_ultimate_bearing_capacity_undrained_unsaturated_basic():
    """Hand-calc: 150*14.8*1 + 400*6.4*1 + 0.5*115*3*5*1
    = 2220 + 2560 + 862.5 = 5642.5
    """
    result = ultimate_bearing_capacity_undrained_unsaturated(
        c=150.0, sigma_zD=400.0, gamma=115.0, B=3.0,
        Nc=14.8, Nq=6.4, N_gamma=5.0
    )
    assert result == pytest.approx(5642.5, rel=1e-4)


def test_ultimate_bearing_capacity_undrained_unsaturated_with_psi():
    """Hand-calc: 100*10*1.3 + 200*5*1.0 + 0.5*120*4*3*0.8
    = 1300 + 1000 + 576 = 2876
    """
    result = ultimate_bearing_capacity_undrained_unsaturated(
        100.0, 200.0, 120.0, 4.0, 10.0, 5.0, 3.0,
        psi_c=1.3, psi_q=1.0, psi_gamma=0.8
    )
    assert result == pytest.approx(2876.0, rel=1e-4)


# ===========================================================================
# ultimate_bearing_capacity_undrained_saturated  (Equation 5-17)
#   q_ult = s_u * Nc * psi_c + sigma_zD * psi_q
# ===========================================================================


def test_ultimate_bearing_capacity_undrained_saturated_basic():
    """Hand-calc: 500*5.14*1 + 300*1 = 2570 + 300 = 2870"""
    result = ultimate_bearing_capacity_undrained_saturated(
        s_u=500.0, sigma_zD=300.0, Nc=5.14
    )
    assert result == pytest.approx(2870.0, rel=1e-4)


def test_ultimate_bearing_capacity_undrained_saturated_with_psi():
    """Hand-calc: 400*5.14*1.2 + 200*0.9 = 2467.2 + 180 = 2647.2"""
    result = ultimate_bearing_capacity_undrained_saturated(
        s_u=400.0, sigma_zD=200.0, Nc=5.14, psi_c=1.2, psi_q=0.9
    )
    assert result == pytest.approx(2647.2, rel=1e-4)


# ===========================================================================
# lumped_correction_factor_c  (Equation 5-18)
#   psi_c = sc * dc * ic * bc * gc
# ===========================================================================


def test_lumped_correction_factor_c_basic():
    """Hand-calc: 1.2*1.1*0.9*1.0*1.0 = 1.188"""
    result = lumped_correction_factor_c(sc=1.2, dc=1.1, ic=0.9)
    assert result == pytest.approx(1.188, rel=1e-4)


def test_lumped_correction_factor_c_defaults():
    """All defaults = 1.0 -> product = 1.0"""
    result = lumped_correction_factor_c()
    assert result == pytest.approx(1.0, rel=1e-4)


def test_lumped_correction_factor_c_all_factors():
    """Hand-calc: 1.1*1.2*0.95*0.98*0.97 = 1.1*1.2*0.95*0.98*0.97
    = 1.32 * 0.95 * 0.98 * 0.97
    = 1.254 * 0.98 * 0.97
    = 1.22892 * 0.97
    = 1.1920524
    """
    result = lumped_correction_factor_c(1.1, 1.2, 0.95, 0.98, 0.97)
    expected = 1.1 * 1.2 * 0.95 * 0.98 * 0.97
    assert result == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# lumped_correction_factor_q  (Equation 5-19)
#   psi_q = sq * dq * iq * bq * gq
# ===========================================================================


def test_lumped_correction_factor_q_basic():
    """Hand-calc: 1.3*1.05*0.8*1.0*1.0 = 1.092"""
    result = lumped_correction_factor_q(sq=1.3, dq=1.05, iq=0.8)
    assert result == pytest.approx(1.092, rel=1e-4)


def test_lumped_correction_factor_q_defaults():
    result = lumped_correction_factor_q()
    assert result == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# lumped_correction_factor_gamma  (Equation 5-20)
#   psi_gamma = s_gamma * d_gamma * i_gamma * b_gamma * g_gamma
# ===========================================================================


def test_lumped_correction_factor_gamma_basic():
    """Hand-calc: 0.8*1.0*0.7*1.0*1.0 = 0.56"""
    result = lumped_correction_factor_gamma(s_gamma=0.8, i_gamma=0.7)
    assert result == pytest.approx(0.56, rel=1e-4)


def test_lumped_correction_factor_gamma_defaults():
    result = lumped_correction_factor_gamma()
    assert result == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# hansen_correction_factor_c_phi0  (Equation 5-21)
#   psi_c = 1 + sc + dc - ic - bc - gc
# ===========================================================================


def test_hansen_correction_factor_c_phi0_basic():
    """Hand-calc: 1 + 0.2 + 0.4 - 0.1 - 0.0 - 0.0 = 1.5"""
    result = hansen_correction_factor_c_phi0(sc=0.2, dc=0.4, ic=0.1)
    assert result == pytest.approx(1.5, rel=1e-4)


def test_hansen_correction_factor_c_phi0_defaults():
    """All defaults=0 -> 1 + 0 + 0 - 0 - 0 - 0 = 1.0"""
    result = hansen_correction_factor_c_phi0()
    assert result == pytest.approx(1.0, rel=1e-4)


def test_hansen_correction_factor_c_phi0_all_terms():
    """Hand-calc: 1 + 0.15 + 0.3 - 0.05 - 0.02 - 0.01 = 1.37"""
    result = hansen_correction_factor_c_phi0(0.15, 0.3, 0.05, 0.02, 0.01)
    assert result == pytest.approx(1.37, rel=1e-4)


# ===========================================================================
# stability_number_undrained  (Equation 5-22)
#   Ns = gamma * H / s_u
# ===========================================================================


def test_stability_number_undrained_basic():
    """Hand-calc: 120*10 / 500 = 1200/500 = 2.4"""
    result = stability_number_undrained(gamma=120.0, H=10.0, s_u=500.0)
    assert result == pytest.approx(2.4, rel=1e-4)


def test_stability_number_undrained_zero_height():
    """Edge: H=0 -> Ns=0"""
    result = stability_number_undrained(120.0, 0.0, 500.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_stability_number_undrained_raises_zero_su():
    with pytest.raises(ValueError, match="s_u must be positive"):
        stability_number_undrained(120.0, 10.0, s_u=0.0)


def test_stability_number_undrained_raises_negative_su():
    with pytest.raises(ValueError, match="s_u must be positive"):
        stability_number_undrained(120.0, 10.0, s_u=-100.0)


# ===========================================================================
# stability_number_drained  (Equation 5-23)
#   Ns = gamma * H / c'
# ===========================================================================


def test_stability_number_drained_basic():
    """Hand-calc: 110*15 / 250 = 1650/250 = 6.6"""
    result = stability_number_drained(gamma=110.0, H=15.0, c_prime=250.0)
    assert result == pytest.approx(6.6, rel=1e-4)


def test_stability_number_drained_zero_height():
    result = stability_number_drained(110.0, 0.0, 250.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_stability_number_drained_raises_zero_c():
    with pytest.raises(ValueError, match="c' must be positive"):
        stability_number_drained(110.0, 15.0, c_prime=0.0)


def test_stability_number_drained_raises_negative_c():
    with pytest.raises(ValueError, match="c' must be positive"):
        stability_number_drained(110.0, 15.0, c_prime=-50.0)


# ===========================================================================
# bearing_capacity_factor_Nc_increasing_su  (Equation 5-24)
#   Nc = 5.14 + k*B / s_u0
# ===========================================================================


def test_bearing_capacity_factor_Nc_increasing_su_basic():
    """Hand-calc: 5.14 + 10*4/200 = 5.14 + 0.2 = 5.34"""
    result = bearing_capacity_factor_Nc_increasing_su(k=10.0, B=4.0, s_u0=200.0)
    assert result == pytest.approx(5.34, rel=1e-4)


def test_bearing_capacity_factor_Nc_increasing_su_zero_k():
    """Edge: k=0 -> Nc = 5.14"""
    result = bearing_capacity_factor_Nc_increasing_su(0.0, 4.0, 200.0)
    assert result == pytest.approx(5.14, rel=1e-4)


def test_bearing_capacity_factor_Nc_increasing_su_raises_zero_su0():
    with pytest.raises(ValueError, match="s_u0 must be positive"):
        bearing_capacity_factor_Nc_increasing_su(10.0, 4.0, s_u0=0.0)


def test_bearing_capacity_factor_Nc_increasing_su_raises_negative_su0():
    with pytest.raises(ValueError, match="s_u0 must be positive"):
        bearing_capacity_factor_Nc_increasing_su(10.0, 4.0, s_u0=-100.0)


# ===========================================================================
# modified_Nc_rectangular_layered_clay  (Equation 5-25)
#   Nc_m_r = Nc_m_c + (1 - B/L)*(Nc_m_s - Nc_m_c)
# ===========================================================================


def test_modified_Nc_rectangular_layered_clay_basic():
    """Hand-calc: Nc_m_s=5.0, Nc_m_c=6.2, B=3, L=6
    Nc_m_r = 6.2 + (1-3/6)*(5.0-6.2) = 6.2 + 0.5*(-1.2) = 6.2 - 0.6 = 5.6
    """
    result = modified_Nc_rectangular_layered_clay(5.0, 6.2, 3.0, 6.0)
    assert result == pytest.approx(5.6, rel=1e-4)


def test_modified_Nc_rectangular_layered_clay_square():
    """Edge: B=L -> Nc_m_r = Nc_m_c + (1-1)*(Nc_m_s - Nc_m_c) = Nc_m_c"""
    result = modified_Nc_rectangular_layered_clay(5.0, 6.2, 4.0, 4.0)
    assert result == pytest.approx(6.2, rel=1e-4)


def test_modified_Nc_rectangular_layered_clay_raises_zero_L():
    with pytest.raises(ValueError, match="L must be positive"):
        modified_Nc_rectangular_layered_clay(5.0, 6.2, 3.0, L=0.0)


def test_modified_Nc_rectangular_layered_clay_raises_zero_B():
    with pytest.raises(ValueError, match="B must be positive"):
        modified_Nc_rectangular_layered_clay(5.0, 6.2, B=0.0, L=6.0)


def test_modified_Nc_rectangular_layered_clay_raises_B_gt_L():
    with pytest.raises(ValueError, match="B must not exceed"):
        modified_Nc_rectangular_layered_clay(5.0, 6.2, B=8.0, L=6.0)


# ===========================================================================
# second_layer_thickness  (Equation 5-26)
#   H2 = (B/2 + H1*tan(phi1)) * c1 / (c2 * tan(phi2))
# ===========================================================================


def test_second_layer_thickness_basic():
    """Hand-calc: B=4, H1=2, c1=100, c2=200, phi1=30, phi2=20
    tan(30)=0.57735, tan(20)=0.36397
    num = (4/2 + 2*0.57735)*100 = (2 + 1.15470)*100 = 315.470
    den = 200*0.36397 = 72.7944
    H2 = 315.470 / 72.7944 = 4.33244...
    """
    expected = (4.0 / 2.0 + 2.0 * math.tan(math.radians(30.0))) * 100.0 / (
        200.0 * math.tan(math.radians(20.0))
    )
    result = second_layer_thickness(B=4.0, H1=2.0, c1=100.0, c2=200.0,
                                    phi1_deg=30.0, phi2_deg=20.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_second_layer_thickness_raises_zero_c2():
    with pytest.raises(ValueError, match="non-zero"):
        second_layer_thickness(4.0, 2.0, 100.0, c2=0.0, phi1_deg=30.0, phi2_deg=20.0)


def test_second_layer_thickness_raises_zero_tan_phi2():
    """phi2=0 -> tan(0)=0 -> ValueError"""
    with pytest.raises(ValueError, match="non-zero"):
        second_layer_thickness(4.0, 2.0, 100.0, 200.0, 30.0, phi2_deg=0.0)


# ===========================================================================
# average_cohesion_mixed_layers  (Equation 5-27)
#   c_ave = (H1*c1 + H2*c2) / (H1 + H2)
# ===========================================================================


def test_average_cohesion_mixed_layers_basic():
    """Hand-calc: (3*100 + 5*200)/(3+5) = (300+1000)/8 = 162.5"""
    result = average_cohesion_mixed_layers(H1=3.0, c1=100.0, H2=5.0, c2=200.0)
    assert result == pytest.approx(162.5, rel=1e-4)


def test_average_cohesion_mixed_layers_equal():
    """Edge: c1==c2 -> average = c1"""
    result = average_cohesion_mixed_layers(3.0, 150.0, 5.0, 150.0)
    assert result == pytest.approx(150.0, rel=1e-4)


def test_average_cohesion_mixed_layers_raises_zero_total():
    with pytest.raises(ValueError, match="must be positive"):
        average_cohesion_mixed_layers(0.0, 100.0, 0.0, 200.0)


# ===========================================================================
# average_friction_angle_mixed_layers  (Equation 5-28)
#   tan(phi_ave) = (H1*tan(phi1) + H2*tan(phi2)) / (H1+H2)
# ===========================================================================


def test_average_friction_angle_mixed_layers_basic():
    """Hand-calc: H1=3, phi1=30, H2=5, phi2=20
    tan_ave = (3*tan(30)+5*tan(20))/8 = (3*0.57735+5*0.36397)/8
            = (1.73205+1.81985)/8 = 3.55190/8 = 0.443988
    phi_ave = atan(0.443988) = 23.937... degrees
    """
    tan_ave = (3.0 * math.tan(math.radians(30.0)) +
               5.0 * math.tan(math.radians(20.0))) / 8.0
    expected = math.degrees(math.atan(tan_ave))
    result = average_friction_angle_mixed_layers(3.0, 30.0, 5.0, 20.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_average_friction_angle_mixed_layers_equal():
    """Edge: phi1==phi2 -> phi_ave = phi1"""
    result = average_friction_angle_mixed_layers(3.0, 25.0, 5.0, 25.0)
    assert result == pytest.approx(25.0, rel=1e-4)


def test_average_friction_angle_mixed_layers_raises_zero_total():
    with pytest.raises(ValueError, match="must be positive"):
        average_friction_angle_mixed_layers(0.0, 30.0, 0.0, 20.0)


# ===========================================================================
# rock_bearing_capacity_factor_Nc  (Equation 5-29)
#   Nc = 5 * tan^4(45 + phi_rf/2)
# ===========================================================================


def test_rock_bearing_capacity_factor_Nc_basic():
    """Hand-calc: phi=30 -> tan(60) = 1.73205
    tan^4(60) = (1.73205)^4 = 8.99999... ~ 9.0
    Nc = 5*9 = 45.0
    """
    angle = math.radians(45.0 + 30.0 / 2.0)
    expected = 5.0 * math.tan(angle) ** 4
    result = rock_bearing_capacity_factor_Nc(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_rock_bearing_capacity_factor_Nc_zero():
    """Edge: phi=0 -> tan(45)=1 -> Nc = 5*1 = 5.0"""
    result = rock_bearing_capacity_factor_Nc(0.0)
    assert result == pytest.approx(5.0, rel=1e-4)


# ===========================================================================
# rock_bearing_capacity_factor_Nq  (Equation 5-30)
#   Nq = tan^6(45 + phi_rf/2)
# ===========================================================================


def test_rock_bearing_capacity_factor_Nq_basic():
    """Hand-calc: phi=30 -> tan(60) = 1.73205
    tan^6(60) = (1.73205)^6 = 26.9999... ~ 27.0
    """
    angle = math.radians(60.0)
    expected = math.tan(angle) ** 6
    result = rock_bearing_capacity_factor_Nq(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_rock_bearing_capacity_factor_Nq_zero():
    """Edge: phi=0 -> tan(45)=1 -> Nq=1"""
    result = rock_bearing_capacity_factor_Nq(0.0)
    assert result == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# rock_bearing_capacity_factor_Ngamma  (Equation 5-31)
#   Ngamma = Nq + 1
# ===========================================================================


def test_rock_bearing_capacity_factor_Ngamma_basic():
    """Hand-calc: Nq=27 -> Ngamma = 28"""
    result = rock_bearing_capacity_factor_Ngamma(Nq=27.0)
    assert result == pytest.approx(28.0, rel=1e-4)


def test_rock_bearing_capacity_factor_Ngamma_one():
    """Edge: Nq=1 -> Ngamma = 2"""
    result = rock_bearing_capacity_factor_Ngamma(1.0)
    assert result == pytest.approx(2.0, rel=1e-4)


# ===========================================================================
# rock_reduced_ultimate_bearing_capacity  (Equation 5-32)
#   q'_ult = q_ult * RQD^2
# ===========================================================================


def test_rock_reduced_ultimate_bearing_capacity_basic():
    """Hand-calc: 1000 * 0.8^2 = 1000 * 0.64 = 640"""
    result = rock_reduced_ultimate_bearing_capacity(1000.0, RQD=0.8)
    assert result == pytest.approx(640.0, rel=1e-4)


def test_rock_reduced_ultimate_bearing_capacity_perfect():
    """Edge: RQD=1.0 -> q_ult unchanged"""
    result = rock_reduced_ultimate_bearing_capacity(1000.0, 1.0)
    assert result == pytest.approx(1000.0, rel=1e-4)


def test_rock_reduced_ultimate_bearing_capacity_zero_rqd():
    """Edge: RQD=0 -> 0"""
    result = rock_reduced_ultimate_bearing_capacity(1000.0, 0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_rock_reduced_ultimate_bearing_capacity_raises_rqd_gt_1():
    with pytest.raises(ValueError, match="RQD must be between"):
        rock_reduced_ultimate_bearing_capacity(1000.0, RQD=1.1)


def test_rock_reduced_ultimate_bearing_capacity_raises_rqd_negative():
    with pytest.raises(ValueError, match="RQD must be between"):
        rock_reduced_ultimate_bearing_capacity(1000.0, RQD=-0.1)


# ===========================================================================
# relative_stiffness_factor  (Equation 5-33)
#   Kr = E'Ib / (Es * B^3)
# ===========================================================================


def test_relative_stiffness_factor_basic():
    """Hand-calc: E'Ib=1e6, Es=500, B=10
    Kr = 1e6 / (500 * 1000) = 1e6/5e5 = 2.0
    """
    result = relative_stiffness_factor(1.0e6, Es=500.0, B=10.0)
    assert result == pytest.approx(2.0, rel=1e-4)


def test_relative_stiffness_factor_raises_zero_Es():
    with pytest.raises(ValueError, match="Es must be positive"):
        relative_stiffness_factor(1e6, Es=0.0, B=10.0)


def test_relative_stiffness_factor_raises_negative_Es():
    with pytest.raises(ValueError, match="Es must be positive"):
        relative_stiffness_factor(1e6, Es=-100.0, B=10.0)


def test_relative_stiffness_factor_raises_zero_B():
    with pytest.raises(ValueError, match="B must be positive"):
        relative_stiffness_factor(1e6, Es=500.0, B=0.0)


def test_relative_stiffness_factor_raises_negative_B():
    with pytest.raises(ValueError, match="B must be positive"):
        relative_stiffness_factor(1e6, Es=500.0, B=-5.0)


# ===========================================================================
# foundation_stiffness_factor  (Equation 5-34)
#   lambda = (ks / (4*Ec*I/B))^(1/4)
# ===========================================================================


def test_foundation_stiffness_factor_basic():
    """Hand-calc: ks=100, B=2, Ec=3e6, I=0.5
    4*Ec*I/B = 4*3e6*0.5/2 = 3e6
    lambda = (100/3e6)^0.25 = (3.333e-5)^0.25
    """
    denom = 4.0 * 3.0e6 * 0.5 / 2.0
    expected = (100.0 / denom) ** 0.25
    result = foundation_stiffness_factor(ks=100.0, B=2.0, Ec=3.0e6, I=0.5)
    assert result == pytest.approx(expected, rel=1e-4)


def test_foundation_stiffness_factor_raises_zero_ks():
    with pytest.raises(ValueError, match="ks must be positive"):
        foundation_stiffness_factor(0.0, 2.0, 3e6, 0.5)


def test_foundation_stiffness_factor_raises_zero_B():
    with pytest.raises(ValueError, match="B must be positive"):
        foundation_stiffness_factor(100.0, 0.0, 3e6, 0.5)


def test_foundation_stiffness_factor_raises_zero_Ec():
    with pytest.raises(ValueError, match="Ec must be positive"):
        foundation_stiffness_factor(100.0, 2.0, 0.0, 0.5)


def test_foundation_stiffness_factor_raises_zero_I():
    with pytest.raises(ValueError, match="I must be positive"):
        foundation_stiffness_factor(100.0, 2.0, 3e6, 0.0)


# ===========================================================================
# modulus_of_subgrade_reaction  (Equation 5-35)
#   ks = q / s
# ===========================================================================


def test_modulus_of_subgrade_reaction_basic():
    """Hand-calc: 2000 / 0.05 = 40000"""
    result = modulus_of_subgrade_reaction(q=2000.0, s=0.05)
    assert result == pytest.approx(40000.0, rel=1e-4)


def test_modulus_of_subgrade_reaction_zero_q():
    """Edge: q=0 -> ks=0"""
    result = modulus_of_subgrade_reaction(0.0, 0.05)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_modulus_of_subgrade_reaction_raises_zero_s():
    with pytest.raises(ValueError, match="s must be positive"):
        modulus_of_subgrade_reaction(2000.0, s=0.0)


def test_modulus_of_subgrade_reaction_raises_negative_s():
    with pytest.raises(ValueError, match="s must be positive"):
        modulus_of_subgrade_reaction(2000.0, s=-0.01)


# ===========================================================================
# subgrade_modulus_from_plate_load_test  (Equation 5-36)
#   ks = kp * (Bp/B)^n
# ===========================================================================


def test_subgrade_modulus_from_plate_load_test_basic():
    """Hand-calc: kp=500, Bp=1, B=4, n=0.5
    ks = 500 * (1/4)^0.5 = 500 * 0.5 = 250
    """
    result = subgrade_modulus_from_plate_load_test(kp=500.0, Bp=1.0, B=4.0, n=0.5)
    assert result == pytest.approx(250.0, rel=1e-4)


def test_subgrade_modulus_from_plate_load_test_default_n():
    """Default n=0.5: kp=1000, Bp=1, B=9 -> 1000*(1/9)^0.5 = 1000/3 = 333.33..."""
    result = subgrade_modulus_from_plate_load_test(1000.0, 1.0, 9.0)
    assert result == pytest.approx(1000.0 / 3.0, rel=1e-4)


def test_subgrade_modulus_from_plate_load_test_n_07():
    """kp=500, Bp=1, B=4, n=0.7 -> 500*(0.25)^0.7"""
    expected = 500.0 * (0.25) ** 0.7
    result = subgrade_modulus_from_plate_load_test(500.0, 1.0, 4.0, n=0.7)
    assert result == pytest.approx(expected, rel=1e-4)


def test_subgrade_modulus_from_plate_load_test_raises_zero_Bp():
    with pytest.raises(ValueError, match="Bp and foundation width B must be positive"):
        subgrade_modulus_from_plate_load_test(500.0, Bp=0.0, B=4.0)


def test_subgrade_modulus_from_plate_load_test_raises_zero_B():
    with pytest.raises(ValueError, match="Bp and foundation width B must be positive"):
        subgrade_modulus_from_plate_load_test(500.0, 1.0, B=0.0)


def test_subgrade_modulus_from_plate_load_test_raises_bad_n():
    with pytest.raises(ValueError, match="Exponent n"):
        subgrade_modulus_from_plate_load_test(500.0, 1.0, 4.0, n=0.0)


def test_subgrade_modulus_from_plate_load_test_raises_n_gt_1():
    with pytest.raises(ValueError, match="Exponent n"):
        subgrade_modulus_from_plate_load_test(500.0, 1.0, 4.0, n=1.5)


# ===========================================================================
# subgrade_modulus_from_elastic_parameters  (Equation 5-37)
#   ks = Es / (B * mu0 * mu1)
# ===========================================================================


def test_subgrade_modulus_from_elastic_parameters_basic():
    """Hand-calc: Es=5000, B=4, mu0=0.5, mu1=0.8
    ks = 5000 / (4*0.5*0.8) = 5000 / 1.6 = 3125
    """
    result = subgrade_modulus_from_elastic_parameters(4.0, 5000.0, 0.5, 0.8)
    assert result == pytest.approx(3125.0, rel=1e-4)


def test_subgrade_modulus_from_elastic_parameters_raises_zero_B():
    with pytest.raises(ValueError, match="B must be positive"):
        subgrade_modulus_from_elastic_parameters(0.0, 5000.0, 0.5, 0.8)


def test_subgrade_modulus_from_elastic_parameters_raises_zero_Es():
    with pytest.raises(ValueError, match="Es must be positive"):
        subgrade_modulus_from_elastic_parameters(4.0, 0.0, 0.5, 0.8)


def test_subgrade_modulus_from_elastic_parameters_raises_zero_mu0():
    with pytest.raises(ValueError, match="mu0 and mu1 must be positive"):
        subgrade_modulus_from_elastic_parameters(4.0, 5000.0, 0.0, 0.8)


def test_subgrade_modulus_from_elastic_parameters_raises_zero_mu1():
    with pytest.raises(ValueError, match="mu0 and mu1 must be positive"):
        subgrade_modulus_from_elastic_parameters(4.0, 5000.0, 0.5, 0.0)


# ===========================================================================
# poissons_ratio_from_friction_angle  (Equation 5-38)
#   nu = (1 - sin(phi')) / (2 - sin(phi'))
# ===========================================================================


def test_poissons_ratio_from_friction_angle_basic():
    """Hand-calc: phi=30 -> sin(30)=0.5
    nu = (1-0.5)/(2-0.5) = 0.5/1.5 = 0.33333...
    """
    result = poissons_ratio_from_friction_angle(30.0)
    assert result == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_poissons_ratio_from_friction_angle_high():
    """Hand-calc: phi=45 -> sin(45)=0.70711
    nu = (1-0.70711)/(2-0.70711) = 0.29289/1.29289 = 0.22654...
    """
    sin45 = math.sin(math.radians(45.0))
    expected = (1.0 - sin45) / (2.0 - sin45)
    result = poissons_ratio_from_friction_angle(45.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_poissons_ratio_from_friction_angle_raises_zero():
    with pytest.raises(ValueError, match="between 0 and 90"):
        poissons_ratio_from_friction_angle(0.0)


def test_poissons_ratio_from_friction_angle_raises_90():
    with pytest.raises(ValueError, match="between 0 and 90"):
        poissons_ratio_from_friction_angle(90.0)


def test_poissons_ratio_from_friction_angle_raises_negative():
    with pytest.raises(ValueError, match="between 0 and 90"):
        poissons_ratio_from_friction_angle(-10.0)


# ===========================================================================
# subgrade_modulus_time_dependent  (Equation 5-39)
#   ksc = s * ks / (s + sc)
# ===========================================================================


def test_subgrade_modulus_time_dependent_basic():
    """Hand-calc: s=0.02, sc=0.03, ks=40000
    ksc = 0.02*40000/(0.02+0.03) = 800/0.05 = 16000
    """
    result = subgrade_modulus_time_dependent(s=0.02, sc=0.03, ks=40000.0)
    assert result == pytest.approx(16000.0, rel=1e-4)


def test_subgrade_modulus_time_dependent_zero_consolidation():
    """Edge: sc=0 -> ksc = s*ks/s = ks"""
    result = subgrade_modulus_time_dependent(0.05, 0.0, 1000.0)
    assert result == pytest.approx(1000.0, rel=1e-4)


def test_subgrade_modulus_time_dependent_raises_zero_total():
    with pytest.raises(ValueError, match="must be positive"):
        subgrade_modulus_time_dependent(0.0, 0.0, 1000.0)


# ===========================================================================
# winkler_spring_stiffness  (Equation 5-40)
#   K = ks * A_cont
# ===========================================================================


def test_winkler_spring_stiffness_basic():
    """Hand-calc: 500 * 4 = 2000"""
    result = winkler_spring_stiffness(ks=500.0, A_cont=4.0)
    assert result == pytest.approx(2000.0, rel=1e-4)


def test_winkler_spring_stiffness_zero_area():
    """Edge: A_cont=0 -> K=0"""
    result = winkler_spring_stiffness(500.0, 0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_winkler_spring_stiffness_zero_ks():
    """Edge: ks=0 -> K=0"""
    result = winkler_spring_stiffness(0.0, 4.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_winkler_spring_stiffness_raises_negative_ks():
    with pytest.raises(ValueError, match="ks must be non-negative"):
        winkler_spring_stiffness(-10.0, 4.0)


def test_winkler_spring_stiffness_raises_negative_area():
    with pytest.raises(ValueError, match="A_cont must be non-negative"):
        winkler_spring_stiffness(500.0, -1.0)


# ===========================================================================
# subgrade_modulus_coupling  (Equation 5-41)
#   ks_i = ks_edge * (sigma_z_ave_edge / sigma_z_ave_i)
# ===========================================================================


def test_subgrade_modulus_coupling_basic():
    """Hand-calc: ks_edge=500, sigma_edge=100, sigma_i=200
    ks_i = 500 * (100/200) = 250
    """
    result = subgrade_modulus_coupling(500.0, 100.0, 200.0)
    assert result == pytest.approx(250.0, rel=1e-4)


def test_subgrade_modulus_coupling_equal_stress():
    """Edge: sigma_edge == sigma_i -> ks_i = ks_edge"""
    result = subgrade_modulus_coupling(500.0, 100.0, 100.0)
    assert result == pytest.approx(500.0, rel=1e-4)


def test_subgrade_modulus_coupling_raises_zero_sigma_i():
    with pytest.raises(ValueError, match="must be positive"):
        subgrade_modulus_coupling(500.0, 100.0, 0.0)


def test_subgrade_modulus_coupling_raises_negative_sigma_i():
    with pytest.raises(ValueError, match="must be positive"):
        subgrade_modulus_coupling(500.0, 100.0, -50.0)


# ===========================================================================
# floating_mat_net_pressure  (Equation 5-42)
#   q_net = (W_structure - W_excavated) / A_mat
# ===========================================================================


def test_floating_mat_net_pressure_basic():
    """Hand-calc: (5000 - 3000) / 100 = 20.0"""
    result = floating_mat_net_pressure(5000.0, 3000.0, 100.0)
    assert result == pytest.approx(20.0, rel=1e-4)


def test_floating_mat_net_pressure_balanced():
    """Edge: W_struct == W_excavated -> q_net = 0"""
    result = floating_mat_net_pressure(5000.0, 5000.0, 100.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_floating_mat_net_pressure_negative_net():
    """W_struct < W_excavated -> negative net pressure (heave)"""
    result = floating_mat_net_pressure(3000.0, 5000.0, 100.0)
    assert result == pytest.approx(-20.0, rel=1e-4)


def test_floating_mat_net_pressure_raises_zero_area():
    with pytest.raises(ValueError, match="A_mat must be positive"):
        floating_mat_net_pressure(5000.0, 3000.0, 0.0)


def test_floating_mat_net_pressure_raises_negative_area():
    with pytest.raises(ValueError, match="A_mat must be positive"):
        floating_mat_net_pressure(5000.0, 3000.0, -10.0)


# ===========================================================================
# meyerhof_hansen_Nq  (Table 5-2)
#   Nq = tan^2(pi/4 + phi/2) * exp(pi * tan(phi))
# ===========================================================================


def test_meyerhof_hansen_Nq_basic():
    """Hand-calc: phi=30 -> tan(60)=1.73205, tan(30)=0.57735
    Nq = (1.73205)^2 * exp(pi*0.57735) = 3.0 * exp(1.81380) = 3.0 * 6.11196
    = 18.33588...  (known tabulated value ~18.4)
    """
    phi = math.radians(30.0)
    expected = math.tan(math.pi / 4.0 + phi / 2.0) ** 2 * math.exp(
        math.pi * math.tan(phi)
    )
    result = meyerhof_hansen_Nq(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_meyerhof_hansen_Nq_zero():
    """Edge: phi=0 -> tan(45)^2 * exp(0) = 1*1 = 1.0"""
    result = meyerhof_hansen_Nq(0.0)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_meyerhof_hansen_Nq_phi20():
    """Hand-calc: phi=20"""
    phi = math.radians(20.0)
    expected = math.tan(math.pi / 4.0 + phi / 2.0) ** 2 * math.exp(
        math.pi * math.tan(phi)
    )
    result = meyerhof_hansen_Nq(20.0)
    assert result == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# meyerhof_hansen_Nc  (Table 5-2)
#   Nc = (Nq - 1) * cot(phi) for phi>0, 5.14 for phi=0
# ===========================================================================


def test_meyerhof_hansen_Nc_basic():
    """Hand-calc: phi=30
    Nq ~ 18.401, Nc = (18.401-1)/tan(30) = 17.401/0.57735 = 30.14
    """
    Nq = meyerhof_hansen_Nq(30.0)
    expected = (Nq - 1.0) / math.tan(math.radians(30.0))
    result = meyerhof_hansen_Nc(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_meyerhof_hansen_Nc_zero():
    """Edge: phi=0 -> Nc = 5.14"""
    result = meyerhof_hansen_Nc(0.0)
    assert result == pytest.approx(5.14, rel=1e-4)


# ===========================================================================
# meyerhof_Ngamma  (Table 5-2 / Table 5-6)
#   Ngamma = (Nq - 1) * tan(1.4*phi)
# ===========================================================================


def test_meyerhof_Ngamma_basic():
    """Hand-calc: phi=30
    Nq = meyerhof_hansen_Nq(30) ~ 18.401
    Ngamma = (18.401-1)*tan(42) = 17.401*0.90040 = 15.665
    """
    Nq = meyerhof_hansen_Nq(30.0)
    expected = (Nq - 1.0) * math.tan(math.radians(42.0))
    result = meyerhof_Ngamma(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_meyerhof_Ngamma_zero():
    """Edge: phi=0 -> Nq=1 -> (1-1)*tan(0) = 0"""
    result = meyerhof_Ngamma(0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# hansen_Ngamma  (Table 5-2 / Table 5-8)
#   Ngamma = 1.5 * (Nq - 1) * tan(phi)
# ===========================================================================


def test_hansen_Ngamma_basic():
    """Hand-calc: phi=30
    Nq ~ 18.401
    Ngamma = 1.5*(18.401-1)*tan(30) = 1.5*17.401*0.57735 = 15.07
    """
    Nq = meyerhof_hansen_Nq(30.0)
    expected = 1.5 * (Nq - 1.0) * math.tan(math.radians(30.0))
    result = hansen_Ngamma(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_hansen_Ngamma_zero():
    """Edge: phi=0 -> 1.5*(1-1)*tan(0) = 0"""
    result = hansen_Ngamma(0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# terzaghi_Nc  (Table 5-2)
# ===========================================================================


def test_terzaghi_Nc_basic():
    """phi=30: Nc = (Nq-1)/tan(30) where Nq = terzaghi_Nq(30)"""
    Nq = terzaghi_Nq(30.0)
    expected = (Nq - 1.0) / math.tan(math.radians(30.0))
    result = terzaghi_Nc(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_terzaghi_Nc_zero():
    """Edge: phi=0 -> Nc = 5.7"""
    result = terzaghi_Nc(0.0)
    assert result == pytest.approx(5.7, rel=1e-4)


# ===========================================================================
# terzaghi_Nq  (Table 5-2)
#   Nq = exp(2*(3pi/4 - phi/2)*tan(phi)) / (2*cos^2(45+phi/2))
# ===========================================================================


def test_terzaghi_Nq_basic():
    """Hand-calc: phi=30 deg
    exponent = 2*(3pi/4 - pi/12)*tan(30) = 2*(2.3562 - 0.2618)*0.57735
             = 2*2.0944*0.57735 = 2.4184
    cos_term = cos(60) = 0.5
    Nq = exp(2.4184) / (2*0.25) = 11.232 / 0.5 = 22.464
    """
    phi = math.radians(30.0)
    exponent = 2.0 * (3.0 * math.pi / 4.0 - phi / 2.0) * math.tan(phi)
    cos_term = math.cos(math.radians(60.0))
    expected = math.exp(exponent) / (2.0 * cos_term ** 2)
    result = terzaghi_Nq(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_terzaghi_Nq_zero():
    """Edge: phi=0 -> Nq = 1.0"""
    result = terzaghi_Nq(0.0)
    assert result == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# terzaghi_Ngamma  (Table 5-2)
#   Ngamma = 2*(Nq+1)*tan(phi) / (1 + 0.4*sin(4*phi))
# ===========================================================================


def test_terzaghi_Ngamma_basic():
    """phi=30"""
    Nq = terzaghi_Nq(30.0)
    phi = math.radians(30.0)
    expected = 2.0 * (Nq + 1.0) * math.tan(phi) / (
        1.0 + 0.4 * math.sin(4.0 * phi)
    )
    result = terzaghi_Ngamma(30.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_terzaghi_Ngamma_zero():
    """Edge: phi=0 -> Ngamma = 0"""
    result = terzaghi_Ngamma(0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# terzaghi_shape_factors  (Table 5-5)
# ===========================================================================


def test_terzaghi_shape_factors_continuous():
    sc, sq, sg = terzaghi_shape_factors("continuous")
    assert sc == pytest.approx(1.0, rel=1e-4)
    assert sq == pytest.approx(1.0, rel=1e-4)
    assert sg == pytest.approx(1.0, rel=1e-4)


def test_terzaghi_shape_factors_strip():
    sc, sq, sg = terzaghi_shape_factors("strip")
    assert sc == pytest.approx(1.0, rel=1e-4)
    assert sq == pytest.approx(1.0, rel=1e-4)
    assert sg == pytest.approx(1.0, rel=1e-4)


def test_terzaghi_shape_factors_square():
    sc, sq, sg = terzaghi_shape_factors("square")
    assert sc == pytest.approx(1.3, rel=1e-4)
    assert sq == pytest.approx(1.0, rel=1e-4)
    assert sg == pytest.approx(0.8, rel=1e-4)


def test_terzaghi_shape_factors_circular():
    sc, sq, sg = terzaghi_shape_factors("circular")
    assert sc == pytest.approx(1.3, rel=1e-4)
    assert sq == pytest.approx(1.0, rel=1e-4)
    assert sg == pytest.approx(0.6, rel=1e-4)


def test_terzaghi_shape_factors_circle_alias():
    sc, sq, sg = terzaghi_shape_factors("circle")
    assert sc == pytest.approx(1.3, rel=1e-4)
    assert sg == pytest.approx(0.6, rel=1e-4)


def test_terzaghi_shape_factors_case_insensitive():
    sc, sq, sg = terzaghi_shape_factors("SQUARE")
    assert sc == pytest.approx(1.3, rel=1e-4)


def test_terzaghi_shape_factors_raises_unknown():
    with pytest.raises(ValueError, match="Unknown shape"):
        terzaghi_shape_factors("triangle")


# ===========================================================================
# meyerhof_shape_factors  (Table 5-6)
# ===========================================================================


def test_meyerhof_shape_factors_basic():
    """phi=30: Kp=tan^2(60)=3.0, ratio=4/8=0.5
    sc = 1 + 0.2*0.5*3 = 1 + 0.3 = 1.3
    sq = 1 + 0.1*0.5*3 = 1 + 0.15 = 1.15
    s_gamma = 1 + 0.1*0.5*3 = 1.15
    """
    sc, sq, sg = meyerhof_shape_factors(B=4.0, L=8.0, phi_deg=30.0)
    assert sc == pytest.approx(1.3, rel=1e-4)
    assert sq == pytest.approx(1.15, rel=1e-4)
    assert sg == pytest.approx(1.15, rel=1e-4)


def test_meyerhof_shape_factors_phi0():
    """phi=0: Kp=tan^2(45)=1, ratio=3/6=0.5
    sc = 1 + 0.2*0.5 = 1.1
    sq = 1.0
    s_gamma = 1.0
    """
    sc, sq, sg = meyerhof_shape_factors(3.0, 6.0, 0.0)
    assert sc == pytest.approx(1.1, rel=1e-4)
    assert sq == pytest.approx(1.0, rel=1e-4)
    assert sg == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# meyerhof_depth_factors  (Table 5-6)
# ===========================================================================


def test_meyerhof_depth_factors_basic():
    """phi=30, Df=3, B=6 -> k=0.5, Kp=3, sqrt(Kp)=1.73205
    dc = 1 + 0.2*0.5*1.73205 = 1 + 0.17321 = 1.17321
    dq = 1 + 0.1*0.5*1.73205 = 1 + 0.08660 = 1.08660
    d_gamma = 1 + 0.1*0.5*1.73205 = 1.08660
    """
    dc, dq, dg = meyerhof_depth_factors(Df=3.0, B=6.0, phi_deg=30.0)
    sqrt_Kp = math.sqrt(3.0)
    assert dc == pytest.approx(1.0 + 0.2 * 0.5 * sqrt_Kp, rel=1e-4)
    assert dq == pytest.approx(1.0 + 0.1 * 0.5 * sqrt_Kp, rel=1e-4)
    assert dg == pytest.approx(1.0 + 0.1 * 0.5 * sqrt_Kp, rel=1e-4)


def test_meyerhof_depth_factors_phi0():
    """phi=0, Df=3, B=6 -> k=0.5, Kp=1, sqrt(Kp)=1
    dc = 1+0.2*0.5*1 = 1.1
    dq = 1.0, d_gamma = 1.0
    """
    dc, dq, dg = meyerhof_depth_factors(3.0, 6.0, 0.0)
    assert dc == pytest.approx(1.1, rel=1e-4)
    assert dq == pytest.approx(1.0, rel=1e-4)
    assert dg == pytest.approx(1.0, rel=1e-4)


def test_meyerhof_depth_factors_raises_zero_B():
    with pytest.raises(ValueError, match="B must be positive"):
        meyerhof_depth_factors(3.0, 0.0, 30.0)


# ===========================================================================
# meyerhof_inclination_factors  (Table 5-6)
# ===========================================================================


def test_meyerhof_inclination_factors_basic():
    """theta=10, phi=30
    theta_rad = pi/18
    ic = (1 - 2*(pi/18)/pi)^2 = (1-2/18)^2 = (1-0.11111)^2 = (0.88889)^2 = 0.79012
    iq = same = 0.79012
    i_gamma = (1 - (pi/18)/(pi/6))^2 = (1 - 1/3)^2 = (2/3)^2 = 0.44444
    """
    theta_rad = math.radians(10.0)
    expected_ic = (1.0 - 2.0 * theta_rad / math.pi) ** 2
    expected_ig = (1.0 - theta_rad / math.radians(30.0)) ** 2
    ic, iq, ig = meyerhof_inclination_factors(10.0, 30.0)
    assert ic == pytest.approx(expected_ic, rel=1e-4)
    assert iq == pytest.approx(expected_ic, rel=1e-4)
    assert ig == pytest.approx(expected_ig, rel=1e-4)


def test_meyerhof_inclination_factors_zero_theta():
    """theta=0 -> all factors = 1.0"""
    ic, iq, ig = meyerhof_inclination_factors(0.0, 30.0)
    assert ic == pytest.approx(1.0, rel=1e-4)
    assert iq == pytest.approx(1.0, rel=1e-4)
    assert ig == pytest.approx(1.0, rel=1e-4)


def test_meyerhof_inclination_factors_phi0():
    """phi=0 -> i_gamma = 1.0 (no frictional soil check)"""
    ic, iq, ig = meyerhof_inclination_factors(10.0, 0.0)
    assert ig == pytest.approx(1.0, rel=1e-4)


def test_meyerhof_inclination_factors_raises_theta_ge_phi():
    with pytest.raises(ValueError, match="theta must be less than phi"):
        meyerhof_inclination_factors(30.0, 30.0)


def test_meyerhof_inclination_factors_raises_theta_exceeds_phi():
    with pytest.raises(ValueError, match="theta must be less than phi"):
        meyerhof_inclination_factors(35.0, 30.0)


# ===========================================================================
# hansen_shape_factors_phi0  (Table 5-7)
#   sc = 0.2*(B/L), sq = 0
# ===========================================================================


def test_hansen_shape_factors_phi0_basic():
    """B=4, L=8: sc = 0.2*0.5 = 0.1, sq=0"""
    sc, sq = hansen_shape_factors_phi0(4.0, 8.0)
    assert sc == pytest.approx(0.1, rel=1e-4)
    assert sq == pytest.approx(0.0, abs=1e-12)


def test_hansen_shape_factors_phi0_square():
    """B=L=6: sc = 0.2*1 = 0.2"""
    sc, sq = hansen_shape_factors_phi0(6.0, 6.0)
    assert sc == pytest.approx(0.2, rel=1e-4)


def test_hansen_shape_factors_phi0_raises_zero_L():
    with pytest.raises(ValueError, match="L must be positive"):
        hansen_shape_factors_phi0(4.0, 0.0)


# ===========================================================================
# hansen_depth_factor_phi0  (Table 5-7)
#   dc = 0.4 * k
# ===========================================================================


def test_hansen_depth_factor_phi0_shallow():
    """Df=2, B=4 -> Df/B=0.5 < 1 -> k=0.5 -> dc=0.4*0.5=0.2"""
    result = hansen_depth_factor_phi0(Df=2.0, B=4.0)
    assert result == pytest.approx(0.2, rel=1e-4)


def test_hansen_depth_factor_phi0_deep():
    """Df=8, B=4 -> Df/B=2 >= 1 -> k=atan(2) -> dc=0.4*atan(2)
    atan(2) = 1.10715 rad -> dc = 0.44286
    """
    expected = 0.4 * math.atan(2.0)
    result = hansen_depth_factor_phi0(8.0, 4.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_hansen_depth_factor_phi0_at_boundary():
    """Df=B=4 -> Df/B=1.0 >= 1 -> k=atan(1)=pi/4
    dc = 0.4*pi/4 = pi/10 = 0.31416
    """
    expected = 0.4 * math.atan(1.0)
    result = hansen_depth_factor_phi0(4.0, 4.0)
    assert result == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# hansen_inclination_factor_phi0  (Table 5-7)
#   ic = 0.5 - 0.5*sqrt(1 - H/(A'*Ca))
# ===========================================================================


def test_hansen_inclination_factor_phi0_basic():
    """H=50, A'=100, Ca=2 -> ratio=50/200=0.25
    ic = 0.5 - 0.5*sqrt(1-0.25) = 0.5 - 0.5*sqrt(0.75)
       = 0.5 - 0.5*0.86603 = 0.5 - 0.43301 = 0.06699
    """
    expected = 0.5 - 0.5 * math.sqrt(0.75)
    result = hansen_inclination_factor_phi0(50.0, 100.0, 2.0)
    assert result == pytest.approx(expected, rel=1e-4)


def test_hansen_inclination_factor_phi0_zero_H():
    """H=0 -> ratio=0 -> ic = 0.5-0.5*sqrt(1) = 0.5-0.5 = 0.0"""
    result = hansen_inclination_factor_phi0(0.0, 100.0, 2.0)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_hansen_inclination_factor_phi0_max_H():
    """H = A'*Ca -> ratio=1 -> ic = 0.5-0.5*0 = 0.5"""
    result = hansen_inclination_factor_phi0(200.0, 100.0, 2.0)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_hansen_inclination_factor_phi0_raises_exceeds():
    with pytest.raises(ValueError, match="must not exceed"):
        hansen_inclination_factor_phi0(300.0, 100.0, 2.0)


def test_hansen_inclination_factor_phi0_raises_zero_A():
    with pytest.raises(ValueError, match="must be positive"):
        hansen_inclination_factor_phi0(50.0, 0.0, 2.0)


def test_hansen_inclination_factor_phi0_raises_zero_Ca():
    with pytest.raises(ValueError, match="must be positive"):
        hansen_inclination_factor_phi0(50.0, 100.0, 0.0)


# ===========================================================================
# hansen_base_factor_phi0  (Table 5-7)
#   bc = eta / (pi/2 + 1)
# ===========================================================================


def test_hansen_base_factor_phi0_basic():
    """eta=0.5 rad -> bc = 0.5 / (pi/2+1) = 0.5 / 2.5708 = 0.19449"""
    expected = 0.5 / (math.pi / 2.0 + 1.0)
    result = hansen_base_factor_phi0(0.5)
    assert result == pytest.approx(expected, rel=1e-4)


def test_hansen_base_factor_phi0_zero():
    """Edge: eta=0 -> bc=0"""
    result = hansen_base_factor_phi0(0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# hansen_ground_factor_phi0  (Table 5-7)
#   gc = beta / (pi/2 + 1)
# ===========================================================================


def test_hansen_ground_factor_phi0_basic():
    """beta=0.3 rad -> gc = 0.3 / (pi/2+1) = 0.3/2.5708 = 0.11671"""
    expected = 0.3 / (math.pi / 2.0 + 1.0)
    result = hansen_ground_factor_phi0(0.3)
    assert result == pytest.approx(expected, rel=1e-4)


def test_hansen_ground_factor_phi0_zero():
    """Edge: beta=0 -> gc=0"""
    result = hansen_ground_factor_phi0(0.0)
    assert result == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# hansen_shape_factors  (Table 5-8, phi > 0)
#   sc = 1 + (B/L)*(Nq/Nc)*cos(phi)
#   sq = 1 + sin(phi)*(B/L)
#   s_gamma = 1 - 0.4*(B/L)
# ===========================================================================


def test_hansen_shape_factors_basic():
    """phi=30, B=4, L=8: ratio=0.5
    Nq=meyerhof_hansen_Nq(30)~18.401, Nc=meyerhof_hansen_Nc(30)~30.14
    sc = 1 + 0.5*(18.401/30.14)*cos(30) = 1 + 0.5*0.6106*0.86603 = 1 + 0.2644 = 1.2644
    sq = 1 + sin(30)*0.5 = 1 + 0.5*0.5 = 1.25
    s_gamma = 1 - 0.4*0.5 = 0.8
    """
    Nq = meyerhof_hansen_Nq(30.0)
    Nc = meyerhof_hansen_Nc(30.0)
    phi = math.radians(30.0)
    ratio = 4.0 / 8.0
    exp_sc = 1.0 + ratio * (Nq / Nc) * math.cos(phi)
    exp_sq = 1.0 + math.sin(phi) * ratio
    exp_sg = 1.0 - 0.4 * ratio

    sc, sq, sg = hansen_shape_factors(4.0, 8.0, 30.0)
    assert sc == pytest.approx(exp_sc, rel=1e-4)
    assert sq == pytest.approx(exp_sq, rel=1e-4)
    assert sg == pytest.approx(exp_sg, rel=1e-4)


def test_hansen_shape_factors_raises_zero_L():
    with pytest.raises(ValueError, match="L must be positive"):
        hansen_shape_factors(4.0, 0.0, 30.0)


# ===========================================================================
# hansen_depth_factors  (Table 5-8, phi > 0)
#   dq = 1 + 2*k*tan(phi)*(1-sin(phi))^2
#   dc = 1 + k*(1-sin(phi))^2*(Nq/Nc)   (approx, see code)
#   d_gamma = 1.0
# ===========================================================================


def test_hansen_depth_factors_basic():
    """phi=30, Df=2, B=4 -> Df/B=0.5 < 1 -> k=0.5
    sin(30)=0.5, tan(30)=0.57735
    dq = 1 + 2*0.5*0.57735*(1-0.5)^2 = 1 + 0.57735*0.25 = 1.14434
    """
    phi = math.radians(30.0)
    k = 0.5
    sin_phi = math.sin(phi)
    tan_phi = math.tan(phi)
    Nq = meyerhof_hansen_Nq(30.0)
    Nc = meyerhof_hansen_Nc(30.0)
    exp_dq = 1.0 + 2.0 * k * tan_phi * (1.0 - sin_phi) ** 2
    exp_dc = 1.0 + k * (1.0 - sin_phi) ** 2 * (Nq / Nc)

    dc, dq, dg = hansen_depth_factors(2.0, 4.0, 30.0)
    assert dq == pytest.approx(exp_dq, rel=1e-4)
    assert dc == pytest.approx(exp_dc, rel=1e-4)
    assert dg == pytest.approx(1.0, rel=1e-4)


def test_hansen_depth_factors_deep():
    """Df=8, B=4 -> ratio=2 >=1 -> k=atan(2)"""
    k = math.atan(2.0)
    phi = math.radians(30.0)
    exp_dq = 1.0 + 2.0 * k * math.tan(phi) * (1.0 - math.sin(phi)) ** 2
    dc, dq, dg = hansen_depth_factors(8.0, 4.0, 30.0)
    assert dq == pytest.approx(exp_dq, rel=1e-4)
    assert dg == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# hansen_inclination_factors  (Table 5-8, phi > 0)
#   iq = (1 - 0.5*H/denom)^5
#   i_gamma = (1 - (0.7 - eta/(2.5*pi))*H/denom)^5
#   ic = (iq*Nq - 1)/(Nq - 1)
# ===========================================================================


def test_hansen_inclination_factors_basic():
    """H=100, V=500, A'=20, Ca=10, phi=30, eta=0
    cot(30)=1.73205, denom = 500 + 20*10*1.73205 = 500+346.41 = 846.41
    iq = (1 - 0.5*100/846.41)^5 = (1-0.05907)^5 = (0.94093)^5
    eta_factor = 0.7 - 0/(2.5*pi) = 0.7
    i_gamma = (1 - 0.7*100/846.41)^5 = (1-0.08269)^5 = (0.91731)^5
    """
    phi = math.radians(30.0)
    cot_phi = 1.0 / math.tan(phi)
    denom = 500.0 + 20.0 * 10.0 * cot_phi
    exp_iq = (1.0 - 0.5 * 100.0 / denom) ** 5
    exp_ig = (1.0 - 0.7 * 100.0 / denom) ** 5
    Nq = meyerhof_hansen_Nq(30.0)
    exp_ic = (exp_iq * Nq - 1.0) / (Nq - 1.0)

    ic, iq, ig = hansen_inclination_factors(100.0, 500.0, 20.0, 10.0, 30.0)
    assert iq == pytest.approx(exp_iq, rel=1e-4)
    assert ig == pytest.approx(exp_ig, rel=1e-4)
    assert ic == pytest.approx(exp_ic, rel=1e-4)


def test_hansen_inclination_factors_with_eta():
    """Test with eta > 0"""
    phi = math.radians(30.0)
    cot_phi = 1.0 / math.tan(phi)
    eta = 0.2
    denom = 500.0 + 20.0 * 10.0 * cot_phi
    eta_factor = 0.7 - eta / (2.5 * math.pi)
    exp_ig = (1.0 - eta_factor * 100.0 / denom) ** 5

    ic, iq, ig = hansen_inclination_factors(100.0, 500.0, 20.0, 10.0, 30.0, eta_rad=0.2)
    assert ig == pytest.approx(exp_ig, rel=1e-4)


def test_hansen_inclination_factors_raises_phi0():
    with pytest.raises(ValueError, match="phi = 0"):
        hansen_inclination_factors(100.0, 500.0, 20.0, 10.0, 0.0)


def test_hansen_inclination_factors_raises_negative_denom():
    """V very negative so denom <= 0"""
    with pytest.raises(ValueError, match="must be positive"):
        hansen_inclination_factors(100.0, -10000.0, 20.0, 10.0, 30.0)


# ===========================================================================
# hansen_base_factors  (Table 5-8, phi > 0)
#   bq = exp(-2 * eta * tan(phi))
#   b_gamma = exp(-2.7 * eta * tan(phi))
# ===========================================================================


def test_hansen_base_factors_basic():
    """eta=0.3 rad, phi=30 -> tan(30)=0.57735
    bq = exp(-2*0.3*0.57735) = exp(-0.34641) = 0.70711...
    b_gamma = exp(-2.7*0.3*0.57735) = exp(-0.46765) = 0.62664...
    """
    tan_phi = math.tan(math.radians(30.0))
    exp_bq = math.exp(-2.0 * 0.3 * tan_phi)
    exp_bg = math.exp(-2.7 * 0.3 * tan_phi)
    bq, bg = hansen_base_factors(0.3, 30.0)
    assert bq == pytest.approx(exp_bq, rel=1e-4)
    assert bg == pytest.approx(exp_bg, rel=1e-4)


def test_hansen_base_factors_zero_eta():
    """eta=0 -> bq=1, b_gamma=1"""
    bq, bg = hansen_base_factors(0.0, 30.0)
    assert bq == pytest.approx(1.0, rel=1e-4)
    assert bg == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# hansen_ground_factors  (Table 5-8, phi > 0)
#   gq = (1 - 0.5*tan(beta))^5
#   g_gamma = (1 - 0.5*tan(beta))^5
# ===========================================================================


def test_hansen_ground_factors_basic():
    """beta=10 deg -> tan(10)=0.17633
    gq = (1 - 0.5*0.17633)^5 = (1-0.08816)^5 = (0.91184)^5 = 0.63268...
    """
    tan_beta = math.tan(math.radians(10.0))
    expected = (1.0 - 0.5 * tan_beta) ** 5
    gq, gg = hansen_ground_factors(10.0)
    assert gq == pytest.approx(expected, rel=1e-4)
    assert gg == pytest.approx(expected, rel=1e-4)


def test_hansen_ground_factors_zero_beta():
    """beta=0 -> gq=g_gamma=1"""
    gq, gg = hansen_ground_factors(0.0)
    assert gq == pytest.approx(1.0, rel=1e-4)
    assert gg == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# local_shear_terzaghi  (Table 5-4)
#   c* = 0.67 * c'
#   phi* = atan(0.67 * tan(phi'))
# ===========================================================================


def test_local_shear_terzaghi_basic():
    """c=200, phi=30 -> c*=0.67*200=134
    phi* = atan(0.67*tan(30)) = atan(0.67*0.57735) = atan(0.38682) = 21.14 deg
    """
    exp_c = 0.67 * 200.0
    exp_phi = math.degrees(math.atan(0.67 * math.tan(math.radians(30.0))))
    c_star, phi_star = local_shear_terzaghi(200.0, 30.0)
    assert c_star == pytest.approx(exp_c, rel=1e-4)
    assert phi_star == pytest.approx(exp_phi, rel=1e-4)


def test_local_shear_terzaghi_zero_phi():
    """phi=0 -> phi*=atan(0)=0, c*=0.67*c"""
    c_star, phi_star = local_shear_terzaghi(100.0, 0.0)
    assert c_star == pytest.approx(67.0, rel=1e-4)
    assert phi_star == pytest.approx(0.0, abs=1e-12)


def test_local_shear_terzaghi_zero_c():
    """c=0 -> c*=0"""
    c_star, phi_star = local_shear_terzaghi(0.0, 30.0)
    assert c_star == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# local_shear_vesic  (Table 5-4)
#   Dr < 0.67: R = 0.67 + Dr - 0.75*Dr^2
#   Dr >= 0.67: R = 1
#   c* = R * c', phi* = atan(R*tan(phi'))
# ===========================================================================


def test_local_shear_vesic_low_density():
    """Dr=0.3, c=200, phi=30
    R = 0.67 + 0.3 - 0.75*0.09 = 0.67 + 0.3 - 0.0675 = 0.9025
    c* = 0.9025*200 = 180.5
    phi* = atan(0.9025*tan(30)) = atan(0.9025*0.57735) = atan(0.52103) = 27.55 deg
    """
    R = 0.67 + 0.3 - 0.75 * 0.3 ** 2
    exp_c = R * 200.0
    exp_phi = math.degrees(math.atan(R * math.tan(math.radians(30.0))))
    c_star, phi_star = local_shear_vesic(200.0, 30.0, Dr=0.3)
    assert c_star == pytest.approx(exp_c, rel=1e-4)
    assert phi_star == pytest.approx(exp_phi, rel=1e-4)


def test_local_shear_vesic_high_density():
    """Dr=0.8 >= 0.67 -> R=1 -> no reduction"""
    c_star, phi_star = local_shear_vesic(200.0, 30.0, Dr=0.8)
    assert c_star == pytest.approx(200.0, rel=1e-4)
    assert phi_star == pytest.approx(30.0, rel=1e-4)


def test_local_shear_vesic_at_boundary():
    """Dr=0.67 -> R=1 -> no reduction"""
    c_star, phi_star = local_shear_vesic(200.0, 30.0, Dr=0.67)
    assert c_star == pytest.approx(200.0, rel=1e-4)
    assert phi_star == pytest.approx(30.0, rel=1e-4)


def test_local_shear_vesic_zero_density():
    """Dr=0 -> R = 0.67 + 0 - 0 = 0.67
    c* = 0.67*200 = 134, phi* = atan(0.67*tan(30))
    """
    c_star, phi_star = local_shear_vesic(200.0, 30.0, Dr=0.0)
    assert c_star == pytest.approx(0.67 * 200.0, rel=1e-4)


def test_local_shear_vesic_raises_negative_Dr():
    with pytest.raises(ValueError, match="Dr must be between"):
        local_shear_vesic(200.0, 30.0, Dr=-0.1)


def test_local_shear_vesic_raises_Dr_gt_1():
    with pytest.raises(ValueError, match="Dr must be between"):
        local_shear_vesic(200.0, 30.0, Dr=1.1)


# ===========================================================================
# _Kp  (private helper - Rankine passive earth pressure coefficient)
#   Kp = tan^2(45 + phi/2)
# ===========================================================================


def test_Kp_basic():
    """phi=30 -> Kp = tan^2(60) = 3.0"""
    from geotech.dm7_2.chapter5 import _Kp
    result = _Kp(30.0)
    assert result == pytest.approx(3.0, rel=1e-4)


def test_Kp_zero():
    """phi=0 -> Kp = tan^2(45) = 1.0"""
    from geotech.dm7_2.chapter5 import _Kp
    result = _Kp(0.0)
    assert result == pytest.approx(1.0, rel=1e-4)


def test_Kp_phi45():
    """phi=45 -> Kp = tan^2(67.5)"""
    from geotech.dm7_2.chapter5 import _Kp
    expected = math.tan(math.radians(67.5)) ** 2
    result = _Kp(45.0)
    assert result == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# _hansen_k  (private helper - depth parameter k)
#   Df/B < 1 -> k = Df/B
#   Df/B >= 1 -> k = atan(Df/B)
# ===========================================================================


def test_hansen_k_shallow():
    """Df=2, B=4 -> ratio=0.5 < 1 -> k=0.5"""
    from geotech.dm7_2.chapter5 import _hansen_k
    result = _hansen_k(2.0, 4.0)
    assert result == pytest.approx(0.5, rel=1e-4)


def test_hansen_k_deep():
    """Df=8, B=4 -> ratio=2 >= 1 -> k=atan(2)=1.10715"""
    from geotech.dm7_2.chapter5 import _hansen_k
    result = _hansen_k(8.0, 4.0)
    assert result == pytest.approx(math.atan(2.0), rel=1e-4)


def test_hansen_k_at_boundary():
    """Df=B=4 -> ratio=1.0 >= 1 -> k=atan(1)=pi/4"""
    from geotech.dm7_2.chapter5 import _hansen_k
    result = _hansen_k(4.0, 4.0)
    assert result == pytest.approx(math.pi / 4.0, rel=1e-4)


def test_hansen_k_raises_zero_B():
    from geotech.dm7_2.chapter5 import _hansen_k
    with pytest.raises(ValueError, match="B must be positive"):
        _hansen_k(4.0, 0.0)
