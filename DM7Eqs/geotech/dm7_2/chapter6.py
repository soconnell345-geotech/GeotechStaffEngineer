"""
UFC 3-220-20, Chapter 6: Deep Foundations

Equations 6-1 through 6-80 covering scour critical velocity, pile impedance,
static axial capacity (Beta and Alpha methods), base resistance (semi-empirical
and Vesic), SPT/CPT-based methods, micropiles, rock sockets, group capacity,
uplift, settlement, lateral capacity (Broms method and Characteristic Load
Method), structural capacity, buckling, and dynamic methods.

Reference:
    UFC 3-220-20, Foundations and Earth Structures, 16 January 2025
"""

import math
from typing import List, Tuple


# ===========================================================================
# SCOUR
# ===========================================================================

def critical_scour_velocity(D50: float) -> float:
    """Critical velocity for onset of scour in coarse-grained soils (Equation 6-1).

    Correlates the critical velocity for scour onset to the median particle
    size for coarse-grained soils.  Should not be extrapolated below
    sand-sized particles.

    .. math::
        v_c = 0.35 \\cdot D_{50}^{0.45}

    Parameters
    ----------
    D50 : float
        Median particle size (mm).

    Returns
    -------
    float
        Critical velocity (m/s).

    Raises
    ------
    ValueError
        If *D50* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-1, p. 409.
    """
    if D50 <= 0.0:
        raise ValueError("D50 must be positive.")
    return 0.35 * D50 ** 0.45


# ===========================================================================
# PILE IMPEDANCE
# ===========================================================================

def pile_impedance(E: float, A: float, c: float) -> float:
    """Pile impedance (Equation 6-2).

    A key parameter to assess drivability.  Piles with higher impedance
    transmit more force during driving.

    .. math::
        I = \\frac{E \\cdot A}{c}

    Parameters
    ----------
    E : float
        Elastic modulus of the pile (force/length^2).
    A : float
        Cross-sectional area of the pile (length^2).
    c : float
        Material wave speed (length/time).

    Returns
    -------
    float
        Pile impedance (force * time / length).

    Raises
    ------
    ValueError
        If any input is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-2, p. 426.
    """
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if A <= 0.0:
        raise ValueError("A must be positive.")
    if c <= 0.0:
        raise ValueError("c must be positive.")
    return E * A / c


# ===========================================================================
# STATIC AXIAL CAPACITY – GENERAL
# ===========================================================================

def nominal_axial_resistance(R_s: float, R_b: float) -> float:
    """Nominal axial resistance of a deep foundation element (Equation 6-3).

    .. math::
        R = R_s + R_b

    Parameters
    ----------
    R_s : float
        Nominal shaft resistance (force).
    R_b : float
        Nominal base resistance (force).

    Returns
    -------
    float
        Total nominal resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-3, p. 444.
    """
    return R_s + R_b


def nominal_shaft_resistance(
    segments: List[Tuple[float, float]],
) -> float:
    """Nominal shaft resistance from discretized segments (Equation 6-4).

    .. math::
        R_s = \\sum f_{s,i} \\cdot A_{s,i}

    Parameters
    ----------
    segments : list of (float, float)
        Each element is ``(f_s_i, A_s_i)`` where *f_s_i* is the unit shaft
        resistance (stress) and *A_s_i* is the surface area (length^2) for
        segment *i*.

    Returns
    -------
    float
        Nominal shaft resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-4, p. 444.
    """
    return sum(f_s * A_s for f_s, A_s in segments)


def nominal_base_resistance(q_b: float, A_b: float) -> float:
    """Nominal base resistance (Equation 6-5).

    .. math::
        R_b = q_b \\cdot A_b

    Parameters
    ----------
    q_b : float
        Unit base resistance (stress).
    A_b : float
        Area of the base (length^2).

    Returns
    -------
    float
        Nominal base resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-5, p. 444.
    """
    return q_b * A_b


# ===========================================================================
# SHAFT RESISTANCE – BETA METHOD (Effective Stress / Drained)
# ===========================================================================

def beta_coefficient(K: float, delta_deg: float) -> float:
    """Beta coefficient for effective stress shaft resistance (Equation 6-6).

    .. math::
        \\beta = K \\cdot \\tan(\\delta)

    Parameters
    ----------
    K : float
        Earth pressure coefficient for the column-soil interface
        (dimensionless).
    delta_deg : float
        Effective interface friction angle (degrees).

    Returns
    -------
    float
        Beta coefficient (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-6, p. 449.
    """
    if K < 0.0:
        raise ValueError("K must be non-negative.")
    return K * math.tan(math.radians(delta_deg))


def beta_method_unit_shaft_resistance(
    beta: float, sigma_z_eff: float
) -> float:
    """Unit shaft resistance by Beta method (Equation 6-7).

    .. math::
        f_s = \\beta \\cdot \\sigma'_z

    Parameters
    ----------
    beta : float
        Beta coefficient (dimensionless).
    sigma_z_eff : float
        Average effective vertical stress over the shaft segment (stress).

    Returns
    -------
    float
        Unit shaft resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-7, p. 450.
    """
    return beta * sigma_z_eff


# ===========================================================================
# SHAFT RESISTANCE – ALPHA METHOD (Total Stress / Undrained)
# ===========================================================================

def alpha_method_unit_shaft_resistance(alpha: float, s_u: float) -> float:
    """Unit shaft resistance by Alpha method (Equation 6-8).

    .. math::
        f_s = \\alpha \\cdot s_u

    Parameters
    ----------
    alpha : float
        Adhesion factor (dimensionless, 0 to 1).
    s_u : float
        Undrained shear strength (stress).

    Returns
    -------
    float
        Unit shaft resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-8, p. 453.
    """
    return alpha * s_u


def alpha_tomlinson_transition(s_u: float, P_a: float) -> float:
    """Alpha for transition zone – driven pile through soft clay into stiff clay
    (Equation 6-9).

    Applies to upper 10b of the stiff clay layer when overlain by soft clay.

    .. math::
        \\alpha = 0.44 \\left(\\frac{s_u}{P_a}\\right)^{-0.28}

    Parameters
    ----------
    s_u : float
        Undrained shear strength of the stiff clay (stress).
    P_a : float
        Atmospheric pressure in same units as *s_u*.

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-9, p. 454.
    """
    if s_u <= 0.0:
        raise ValueError("s_u must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return 0.44 * (s_u / P_a) ** (-0.28)


def alpha_chen_drilled_shaft(s_u_ICU: float, P_a: float) -> float:
    """Alpha for drilled shafts – Chen et al. (2011) (Equation 6-10).

    .. math::
        \\alpha = 0.3 + \\frac{0.17}{s_{u,ICU}/P_a} \\leq 1

    Parameters
    ----------
    s_u_ICU : float
        Undrained strength from ICU triaxial test (stress).
    P_a : float
        Atmospheric pressure (same units).

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless), capped at 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-10, p. 456.
    """
    if s_u_ICU <= 0.0:
        raise ValueError("s_u_ICU must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return min(0.3 + 0.17 / (s_u_ICU / P_a), 1.0)


def su_ICU_from_UC(s_u_UC: float, OCR: float) -> float:
    """Convert UC undrained strength to ICU equivalent (Equation 6-11).

    .. math::
        s_{u,ICU} \\approx 1.74 \\cdot s_{u,UC} \\cdot OCR^{-0.25}

    Parameters
    ----------
    s_u_UC : float
        Undrained strength from unconfined compression test (stress).
    OCR : float
        Overconsolidation ratio (dimensionless).

    Returns
    -------
    float
        Equivalent ICU undrained strength (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-11, p. 456.
    """
    if OCR <= 0.0:
        raise ValueError("OCR must be positive.")
    return 1.74 * s_u_UC * OCR ** (-0.25)


def su_ICU_from_UU(s_u_UU: float, OCR: float) -> float:
    """Convert UU undrained strength to ICU equivalent (Equation 6-12).

    .. math::
        s_{u,ICU} \\approx 1.68 \\cdot s_{u,UU} \\cdot OCR^{-0.25}

    Parameters
    ----------
    s_u_UU : float
        Undrained strength from UU triaxial test (stress).
    OCR : float
        Overconsolidation ratio (dimensionless).

    Returns
    -------
    float
        Equivalent ICU undrained strength (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-12, p. 456.
    """
    if OCR <= 0.0:
        raise ValueError("OCR must be positive.")
    return 1.68 * s_u_UU * OCR ** (-0.25)


def su_ICU_from_DSS(s_u_DSS: float) -> float:
    """Convert DSS undrained strength to ICU equivalent (Equation 6-13).

    .. math::
        s_{u,ICU} \\approx 1.43 \\cdot s_{u,DSS}

    Parameters
    ----------
    s_u_DSS : float
        Undrained strength from direct simple shear test (stress).

    Returns
    -------
    float
        Equivalent ICU undrained strength (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-13, p. 456.
    """
    return 1.43 * s_u_DSS


def alpha_salgado_drilled_shaft(s_u: float, P_a: float) -> float:
    """Alpha for drilled shafts and CFAs – Salgado (2008) (Equation 6-14).

    For soils with clay fraction >= 50% and OCR between 3 and 5.

    .. math::
        \\alpha = 0.4 \\left[1 - 0.12 \\ln\\left(\\frac{s_u}{P_a}\\right)\\right]

    Parameters
    ----------
    s_u : float
        Undrained shear strength (stress).
    P_a : float
        Atmospheric pressure (same units).

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-14, p. 456.
    """
    if s_u <= 0.0:
        raise ValueError("s_u must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return 0.4 * (1.0 - 0.12 * math.log(s_u / P_a))


def alpha_coleman_CFA(s_u: float, P_a: float) -> float:
    """Alpha for CFAs – Coleman and Arcement (2002) (Equation 6-15).

    Valid for normalized strengths in the range 0.24 to 1.42.

    .. math::
        \\alpha = \\left(\\frac{s_u}{P_a}\\right)^{-0.53}

    Parameters
    ----------
    s_u : float
        Undrained shear strength (stress).
    P_a : float
        Atmospheric pressure (same units).

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-15, p. 457.
    """
    if s_u <= 0.0:
        raise ValueError("s_u must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return (s_u / P_a) ** (-0.53)


def alpha_API_P2A(s_u: float, sigma_z_eff: float) -> float:
    """Alpha by API P2A method – Randolph and Murphy (1985) (Equations 6-16, 6-17).

    .. math::
        \\alpha = \\left(\\frac{s_u}{\\sigma'_z}\\right)^{-0.5}
        \\quad \\text{for } s_u/\\sigma'_z \\leq 1

        \\alpha = \\left(\\frac{s_u}{\\sigma'_z}\\right)^{-0.25}
        \\quad \\text{for } s_u/\\sigma'_z > 1

    Parameters
    ----------
    s_u : float
        Undrained shear strength (stress).
    sigma_z_eff : float
        Effective vertical stress (same units).

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless), capped at 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equations 6-16 and 6-17, p. 457.
    """
    if sigma_z_eff <= 0.0:
        raise ValueError("sigma_z_eff must be positive.")
    ratio = s_u / sigma_z_eff
    if ratio <= 1.0:
        alpha = ratio ** (-0.5)
    else:
        alpha = ratio ** (-0.25)
    return min(alpha, 1.0)


def alpha_API_from_OCR(OCR: float) -> float:
    """Alpha by API method expressed in terms of OCR (Equations 6-18, 6-19).

    .. math::
        \\alpha = 1.07 \\cdot OCR^{-0.4} \\quad \\text{for } OCR \\leq 4.5

        \\alpha = 0.73 \\cdot OCR^{-0.2} \\quad \\text{for } OCR > 4.5

    Parameters
    ----------
    OCR : float
        Overconsolidation ratio (dimensionless).

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless), capped at 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equations 6-18 and 6-19, p. 458.
    """
    if OCR <= 0.0:
        raise ValueError("OCR must be positive.")
    if OCR <= 4.5:
        alpha = 1.07 * OCR ** (-0.4)
    else:
        alpha = 0.73 * OCR ** (-0.2)
    return min(alpha, 1.0)


# ===========================================================================
# BASE RESISTANCE – DRAINED (Semi-Empirical)
# ===========================================================================

def unit_base_resistance_drained(N_q: float, sigma_zD_eff: float) -> float:
    """Unit base resistance for drained conditions (Equation 6-20).

    .. math::
        q_b = N_q \\cdot \\sigma'_{zD}

    Parameters
    ----------
    N_q : float
        Bearing capacity factor (dimensionless, from Table 6-17).
    sigma_zD_eff : float
        Effective vertical stress at the base elevation (stress).

    Returns
    -------
    float
        Unit base resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-20, p. 459.
    """
    return N_q * sigma_zD_eff


def cheng_Nq(phi_deg: float, Z_over_b: float) -> float:
    """Bearing capacity factor Nq after Cheng (2004), from Table 6-17.

    .. math::
        N_q \\approx \\exp\\left[\\frac{\\phi' \\cdot \\tan\\phi'}{6.34}
        \\cdot \\left(\\frac{Z}{b}\\right)^{-0.0486}\\right]
        \\cdot \\left(\\frac{Z}{b}\\right)^{0.437}

    Parameters
    ----------
    phi_deg : float
        Effective friction angle (degrees).
    Z_over_b : float
        Ratio of embedment depth to pile width (dimensionless).

    Returns
    -------
    float
        Bearing capacity factor Nq (dimensionless), capped at 200.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-17, p. 459.
    """
    if phi_deg <= 0.0:
        raise ValueError("phi_deg must be positive.")
    if Z_over_b <= 0.0:
        raise ValueError("Z_over_b must be positive.")
    phi_rad = math.radians(phi_deg)
    N_q = (math.exp(phi_rad * math.tan(phi_rad) / 6.34
                     * Z_over_b ** (-0.0486))
           * Z_over_b ** 0.437)
    return min(N_q, 200.0)


# ===========================================================================
# BASE RESISTANCE – VESIC (Drained)
# ===========================================================================

def unit_base_resistance_vesic_drained(
    N_q_star: float, sigma_m_eff: float
) -> float:
    """Unit base resistance by Vesic method, drained (Equation 6-21).

    .. math::
        q_b = N^*_q \\cdot \\sigma'_m

    Parameters
    ----------
    N_q_star : float
        Modified bearing capacity factor (dimensionless).
    sigma_m_eff : float
        Mean effective stress at b/2 below pile base (stress).

    Returns
    -------
    float
        Unit base resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-21, p. 459.
    """
    return N_q_star * sigma_m_eff


def mean_effective_stress(
    sigma_zD_eff: float, phi_deg: float
) -> float:
    """Mean effective stress at b/2 below pile base (Equation 6-22).

    Assumes K0 conditions.

    .. math::
        \\sigma'_m = \\frac{2 - \\sin\\phi'}{3} \\cdot \\sigma'_{zD+b/2}

    Parameters
    ----------
    sigma_zD_eff : float
        Effective vertical stress at b/2 below the base (stress).
    phi_deg : float
        Peak effective friction angle (degrees).

    Returns
    -------
    float
        Mean effective stress (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-22, p. 460.
    """
    phi_rad = math.radians(phi_deg)
    return (2.0 - math.sin(phi_rad)) / 3.0 * sigma_zD_eff


def vesic_Nq_star(phi_deg: float, I_rr: float) -> float:
    """Modified bearing capacity factor N*q by Vesic (1977) (Equation 6-23).

    .. math::
        N^*_q = \\frac{3}{3 - \\sin\\phi'}
        \\cdot e^{\\left(\\frac{\\pi}{2} - \\phi'\\right)\\tan\\phi'}
        \\cdot I_{rr}^{\\frac{4\\sin\\phi'}{3(1+\\sin\\phi')}}

    Parameters
    ----------
    phi_deg : float
        Peak effective friction angle (degrees).
    I_rr : float
        Reduced rigidity index (dimensionless).

    Returns
    -------
    float
        Modified bearing capacity factor N*q (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-23, p. 460.
    """
    if I_rr <= 0.0:
        raise ValueError("I_rr must be positive.")
    phi = math.radians(phi_deg)
    sin_phi = math.sin(phi)
    term1 = 3.0 / (3.0 - sin_phi)
    term2 = math.exp((math.pi / 2.0 - phi) * math.tan(phi))
    exponent = 4.0 * sin_phi / (3.0 * (1.0 + sin_phi))
    term3 = I_rr ** exponent
    return term1 * term2 * term3


def rigidity_index(E: float, nu: float, sigma_m_eff: float,
                   phi_deg: float) -> float:
    """Rigidity index for coarse-grained soils – Vesic (1977) (Equation 6-24).

    .. math::
        I_r = \\frac{E}{(1 + 2\\nu) \\cdot \\sigma'_m \\cdot \\tan\\phi'}

    Parameters
    ----------
    E : float
        Young's Modulus of the soil (stress).
    nu : float
        Poisson's ratio (dimensionless).
    sigma_m_eff : float
        Mean effective stress (stress).
    phi_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    float
        Rigidity index (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-24, p. 460.
    """
    phi_rad = math.radians(phi_deg)
    denom = (1.0 + 2.0 * nu) * sigma_m_eff * math.tan(phi_rad)
    if denom <= 0.0:
        raise ValueError("Denominator must be positive.")
    return E / denom


def reduced_rigidity_index(I_r: float, epsilon_v: float) -> float:
    """Reduced rigidity index (Equation 6-25).

    .. math::
        I_{rr} = \\frac{I_r}{1 + I_r \\cdot \\varepsilon_v}

    Parameters
    ----------
    I_r : float
        Rigidity index (dimensionless).
    epsilon_v : float
        Volumetric strain from foundation loading (dimensionless).

    Returns
    -------
    float
        Reduced rigidity index (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-25, p. 460.
    """
    return I_r / (1.0 + I_r * epsilon_v)


def volumetric_strain(q_b_app: float, E_s: float, F_nu: float) -> float:
    """Volumetric strain estimate for Vesic method (Equation 6-26).

    .. math::
        \\varepsilon_v \\approx \\frac{q_{b,app} \\cdot F_\\nu}{E_s}

    Parameters
    ----------
    q_b_app : float
        Estimated applied bearing pressure at the base (stress).
    E_s : float
        Young's Modulus of the soil near the base (stress).
    F_nu : float
        Strain factor based on Boussinesq theory and Poisson's ratio
        (dimensionless, from Table 6-18).

    Returns
    -------
    float
        Volumetric strain (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-26, p. 461.
    """
    if E_s <= 0.0:
        raise ValueError("E_s must be positive.")
    return q_b_app * F_nu / E_s


# ===========================================================================
# BASE RESISTANCE – UNDRAINED
# ===========================================================================

def unit_base_resistance_undrained(N_c_star: float, s_u: float) -> float:
    """Unit base resistance for undrained conditions (Equation 6-27).

    .. math::
        q_b = N^*_c \\cdot s_u

    Parameters
    ----------
    N_c_star : float
        Modified bearing capacity factor (dimensionless).
    s_u : float
        Undrained strength near the base (stress).

    Returns
    -------
    float
        Unit base resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-27, p. 461.
    """
    return N_c_star * s_u


def vesic_Nc_star(I_rr: float) -> float:
    """Modified bearing capacity factor N*c by Vesic (1977) (Equation 6-28).

    .. math::
        N^*_c = \\frac{4}{3}(\\ln I_{rr} + 1) + \\frac{\\pi}{2} + 1

    Parameters
    ----------
    I_rr : float
        Reduced rigidity index (dimensionless).

    Returns
    -------
    float
        Modified bearing capacity factor N*c (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-28, p. 461.
    """
    if I_rr <= 0.0:
        raise ValueError("I_rr must be positive.")
    return (4.0 / 3.0) * (math.log(I_rr) + 1.0) + math.pi / 2.0 + 1.0


def undrained_rigidity_index(E_u: float, s_u: float) -> float:
    """Rigidity index for undrained conditions (Equation 6-29).

    .. math::
        I_{rr} = I_r = \\frac{E_u}{3 \\cdot s_u}

    Parameters
    ----------
    E_u : float
        Undrained Young's modulus (stress).
    s_u : float
        Undrained shear strength (stress).

    Returns
    -------
    float
        Rigidity index (dimensionless).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-29, p. 461.
    """
    if s_u <= 0.0:
        raise ValueError("s_u must be positive.")
    return E_u / (3.0 * s_u)


def Nc_star_FHWA_from_Ir(I_r: float) -> float:
    """N*c from FHWA (1999) as function of rigidity index (Equation 6-30).

    For formed-in-place columns with su < 2 ksf.

    .. math::
        N^*_c = \\frac{4}{3}(\\ln I_r + 1) + \\frac{\\pi}{2} + 1 \\leq 9

    Parameters
    ----------
    I_r : float
        Rigidity index (dimensionless).

    Returns
    -------
    float
        N*c (dimensionless), capped at 9.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-30, p. 462.
    """
    if I_r <= 0.0:
        raise ValueError("I_r must be positive.")
    val = (4.0 / 3.0) * (math.log(I_r) + 1.0) + math.pi / 2.0 + 1.0
    return min(val, 9.0)


def Nc_star_FHWA_from_su(s_u: float, P_a: float) -> float:
    """N*c from FHWA (1999) as function of undrained strength (Equation 6-31).

    .. math::
        N^*_c = 10.2 - \\frac{12.4}{0.1 + s_u/P_a} \\leq 9

    Parameters
    ----------
    s_u : float
        Undrained shear strength (stress).
    P_a : float
        Atmospheric pressure (same units).

    Returns
    -------
    float
        N*c (dimensionless), capped at 9.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-31, p. 462.
    """
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    ratio = s_u / P_a
    val = 10.2 - 12.4 / (0.1 + ratio)
    return min(val, 9.0)


# ===========================================================================
# SPT AND CPT-BASED METHODS (LCPC)
# ===========================================================================

def lcpc_unit_shaft_resistance(
    q_c: float, P_a: float, k_s: float, f_p: float
) -> float:
    """LCPC method unit side resistance from CPT (Equation 6-32).

    .. math::
        f_s = \\frac{q_c / P_a}{k_s} \\cdot P_a \\leq f_p

    Parameters
    ----------
    q_c : float
        Cone tip resistance (stress).
    P_a : float
        Atmospheric pressure (same units).
    k_s : float
        Side resistance factor (dimensionless, from Table 6-21).
    f_p : float
        Maximum unit side resistance (stress, from Table 6-22).

    Returns
    -------
    float
        Unit side resistance (stress), capped at *f_p*.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-32, p. 463.
    """
    if k_s <= 0.0:
        raise ValueError("k_s must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    f_s = (q_c / P_a) / k_s * P_a
    return min(f_s, f_p)


def lcpc_unit_base_resistance(
    q_c_avg: float, P_a: float, k_t: float
) -> float:
    """LCPC method unit base resistance from CPT (Equation 6-33).

    .. math::
        q_b = \\frac{q_{c,a} / P_a}{k_t} \\cdot P_a

    Parameters
    ----------
    q_c_avg : float
        Average cone tip resistance near the base (stress).
    P_a : float
        Atmospheric pressure (same units).
    k_t : float
        Base bearing factor (dimensionless, from Table 6-23).

    Returns
    -------
    float
        Unit base resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-33, p. 463.
    """
    if k_t <= 0.0:
        raise ValueError("k_t must be positive.")
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return (q_c_avg / P_a) / k_t * P_a


# ===========================================================================
# MICROPILES
# ===========================================================================

def micropile_shaft_resistance(
    alpha_bond: float, b: float, Z_b: float
) -> float:
    """Nominal side resistance of a micropile (Equation 6-34).

    .. math::
        R_s = \\alpha_{bond} \\cdot \\pi \\cdot b \\cdot Z_b

    Parameters
    ----------
    alpha_bond : float
        Nominal unit grout-to-ground bond strength (stress).
    b : float
        Diameter of the bond zone (length).
    Z_b : float
        Length of the bond zone (length).

    Returns
    -------
    float
        Nominal side resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-34, p. 466.
    """
    if b <= 0.0:
        raise ValueError("b must be positive.")
    if Z_b <= 0.0:
        raise ValueError("Z_b must be positive.")
    return alpha_bond * math.pi * b * Z_b


# ===========================================================================
# ROCK SOCKETS
# ===========================================================================

def rock_socket_unit_shaft_resistance(
    q_u: float, P_a: float, C: float = 1.0, n: float = 0.5
) -> float:
    """Unit side resistance of a rock socket (Equation 6-35).

    .. math::
        f_s = C \\cdot P_a \\left(\\frac{q_u}{P_a}\\right)^n

    Parameters
    ----------
    q_u : float
        Lesser of unconfined compressive strength of rock and concrete (stress).
    P_a : float
        Atmospheric pressure (same units).
    C : float, optional
        Fitting parameter (default 1.0 for normal rock).
    n : float, optional
        Fitting exponent (default 0.5).

    Returns
    -------
    float
        Unit side resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-35, p. 468.
    """
    if P_a <= 0.0:
        raise ValueError("P_a must be positive.")
    return C * P_a * (q_u / P_a) ** n


def rock_socket_unit_base_resistance(
    N_cr_star: float, q_u: float
) -> float:
    """Unit base resistance for rock-socketed drilled shafts (Equation 6-36).

    .. math::
        q_b = N^*_{cr} \\cdot q_u

    Parameters
    ----------
    N_cr_star : float
        Modified bearing capacity factor for rock (dimensionless,
        typically 2.5 for clean, level sockets).
    q_u : float
        Unconfined compressive strength of rock (stress).

    Returns
    -------
    float
        Unit base resistance (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-36, p. 469.
    """
    return N_cr_star * q_u


# ===========================================================================
# GROUP CAPACITY
# ===========================================================================

def group_axial_capacity(
    n: int, eta_g: float, R_r: float, R_r_gblock: float
) -> float:
    """Factored axial capacity of a pile group (Equation 6-37).

    .. math::
        R_{r,g} = \\min(n \\cdot \\eta_g \\cdot R_r, \\; R_{r,gblock})

    Parameters
    ----------
    n : int
        Number of columns.
    eta_g : float
        Group efficiency factor (dimensionless, typically 1.0).
    R_r : float
        Factored single column capacity (force).
    R_r_gblock : float
        Factored resistance to block failure (force).

    Returns
    -------
    float
        Factored axial capacity of the group (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-37, p. 469.
    """
    return min(n * eta_g * R_r, R_r_gblock)


def block_failure_resistance(
    Z: float, B: float, L: float, f_s1: float,
    s_u2: float, N_c: float
) -> float:
    """Nominal resistance to block failure (Equation 6-38).

    .. math::
        R_{n,gblock} = 2 Z (B + L) f_{s,1} + B \\cdot L \\cdot s_{u,2} \\cdot N_c

    Parameters
    ----------
    Z : float
        Depth of column embedment (length).
    B : float
        Short plan dimension of the group (length).
    L : float
        Long plan dimension of the group (length).
    f_s1 : float
        Weighted average unit shaft resistance (stress).
    s_u2 : float
        Average undrained strength from base to 2B-3B below base (stress).
    N_c : float
        Bearing capacity factor (dimensionless, from Equation 6-39).

    Returns
    -------
    float
        Nominal block failure resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-38, p. 471.
    """
    return 2.0 * Z * (B + L) * f_s1 + B * L * s_u2 * N_c


def block_failure_Nc(B: float, L: float, Z: float) -> float:
    """Bearing capacity factor Nc for block failure – Brinch-Hansen (1957)
    (Equation 6-39).

    .. math::
        N_c = 5 \\left(1 + \\frac{0.2 B}{L}\\right)
        \\left(1 + \\frac{0.2 Z}{B}\\right) \\leq 9

    Parameters
    ----------
    B : float
        Short plan dimension (length).
    L : float
        Long plan dimension (length).
    Z : float
        Embedment depth (length).

    Returns
    -------
    float
        Nc (dimensionless), capped at 9.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-39, p. 471.
    """
    if L <= 0.0:
        raise ValueError("L must be positive.")
    if B <= 0.0:
        raise ValueError("B must be positive.")
    val = 5.0 * (1.0 + 0.2 * B / L) * (1.0 + 0.2 * Z / B)
    return min(val, 9.0)


# ===========================================================================
# UPLIFT CAPACITY
# ===========================================================================

def group_uplift_capacity(
    n: int, R_r_s: float, W_r_e: float, W_r_e_cap: float,
    R_r_ublock: float
) -> float:
    """Combined factored uplift capacity of n columns (Equation 6-40).

    .. math::
        R_{r,gu} = \\min\\left(n \\cdot R_{r,s} + W_{r,e} + W_{r,e,cap},
        \\; R_{r,ublock}\\right)

    Parameters
    ----------
    n : int
        Number of columns.
    R_r_s : float
        Factored uplift capacity of a single column (force).
    W_r_e : float
        Factored effective weight of a single column (force).
    W_r_e_cap : float
        Factored effective weight of the pile cap (force).
    R_r_ublock : float
        Factored uplift capacity of block (force).

    Returns
    -------
    float
        Combined factored uplift capacity (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-40, p. 473.
    """
    individual = n * R_r_s + W_r_e + W_r_e_cap
    return min(individual, R_r_ublock)


def uplift_block_volume(B: float, L: float, Z: float) -> float:
    """Volume of the truncated uplift block prism for coarse-grained soils
    (Equation 6-41).

    .. math::
        V_{block} = \\frac{Z}{12}\\left(4 B L + Z^2 + 2 Z (B + L)\\right)

    Parameters
    ----------
    B : float
        Short plan dimension of the group (length).
    L : float
        Long plan dimension of the group (length).
    Z : float
        Embedment depth of the group (length).

    Returns
    -------
    float
        Volume of the block (length^3).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-41, p. 474.
    """
    return Z / 12.0 * (4.0 * B * L + Z ** 2 + 2.0 * Z * (B + L))


def uplift_block_weight_undrained(
    B: float, L: float, Z1: float, Z2: float,
    gamma_m: float, gamma_b: float, W_cap: float
) -> float:
    """Effective weight of uplift block for undrained conditions (Equation 6-42).

    .. math::
        W_{e,g} = B \\cdot L \\cdot (Z_1 \\gamma_m + Z_2 \\gamma_b) + W_{cap}

    Parameters
    ----------
    B : float
        Short plan dimension (length).
    L : float
        Long plan dimension (length).
    Z1 : float
        Depth above water table (length).
    Z2 : float
        Depth below water table (length).
    gamma_m : float
        Moist unit weight (force/length^3).
    gamma_b : float
        Buoyant unit weight (force/length^3).
    W_cap : float
        Weight of the pile cap (force).

    Returns
    -------
    float
        Effective weight of the block (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-42, p. 474.
    """
    return B * L * (Z1 * gamma_m + Z2 * gamma_b) + W_cap


def uplift_block_capacity_undrained(
    Z: float, B: float, L: float, s_u_avg: float, W_e_g: float
) -> float:
    """Uplift block capacity for undrained conditions (Equation 6-43).

    .. math::
        R_{n,block} = 2 Z (B + L) s_{u,avg} + W_{e,g}

    Parameters
    ----------
    Z : float
        Total embedment depth (length).
    B : float
        Short plan dimension (length).
    L : float
        Long plan dimension (length).
    s_u_avg : float
        Average undrained shear strength along block sides (stress).
    W_e_g : float
        Effective weight of the block (force).

    Returns
    -------
    float
        Nominal block uplift capacity (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-43, p. 474.
    """
    return 2.0 * Z * (B + L) * s_u_avg + W_e_g


# ===========================================================================
# SETTLEMENT
# ===========================================================================

def elastic_compression_pile(
    delta_Q: float, Z: float, A_p: float, E_p: float
) -> float:
    """Elastic compression of a foundation element (Equation 6-45).

    .. math::
        \\delta_e = \\frac{\\Delta Q \\cdot Z}{A_p \\cdot E_p}

    Parameters
    ----------
    delta_Q : float
        Average change in load over the element length (force).
    Z : float
        Length of the element (length).
    A_p : float
        Cross-sectional area of pile material (length^2).
    E_p : float
        Young's Modulus of pile material (stress).

    Returns
    -------
    float
        Elastic compression (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-45, p. 479.
    """
    if A_p <= 0.0:
        raise ValueError("A_p must be positive.")
    if E_p <= 0.0:
        raise ValueError("E_p must be positive.")
    return delta_Q * Z / (A_p * E_p)


def meyerhof_group_settlement(
    Q_d: float, B: float, L: float, N1_60: float, I_f: float
) -> float:
    """Settlement of a pile group in coarse-grained soil – Meyerhof (1976)
    (Equation 6-46).

    .. math::
        \\delta_s = \\frac{Q_d \\sqrt{B}}{B \\cdot L \\cdot \\bar{N}_{1,60}} \\cdot I_f

    Parameters
    ----------
    Q_d : float
        Unfactored group design load (kips).
    B : float
        Short dimension of pile group (ft).
    L : float
        Long dimension of pile group (ft).
    N1_60 : float
        Average overburden-corrected N value within B below base (blows/ft).
    I_f : float
        Influence factor (dimensionless, from Equation 6-47).

    Returns
    -------
    float
        Estimated settlement (inches).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-46, p. 480.
    """
    if N1_60 <= 0.0:
        raise ValueError("N1_60 must be positive.")
    if B <= 0.0 or L <= 0.0:
        raise ValueError("B and L must be positive.")
    return Q_d * math.sqrt(B) / (B * L * N1_60) * I_f


def meyerhof_influence_factor(Z: float, B: float) -> float:
    """Influence factor for Meyerhof group settlement (Equation 6-47).

    .. math::
        I_f = 1 - \\frac{1}{2} \\left(\\frac{Z}{B}\\right)^{-1}
        \\geq 0.5

    Parameters
    ----------
    Z : float
        Length of the pile group (same units as B).
    B : float
        Short dimension of the pile group.

    Returns
    -------
    float
        Influence factor (dimensionless), minimum 0.5.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-47, p. 480.
    """
    if B <= 0.0:
        raise ValueError("B must be positive.")
    if Z <= 0.0:
        raise ValueError("Z must be positive.")
    return max(1.0 - 0.5 * B / Z, 0.5)


def drilled_shaft_base_resistance_at_4pct(
    R_b: float, soil_type: str
) -> float:
    """Nominal base resistance at 4% displacement for drilled shafts
    (Equation 6-48).

    .. math::
        R'_b = \\begin{cases}
        0.71 \\cdot R_b & \\text{fine-grained (cohesive)} \\\\
        R_b & \\text{coarse-grained (cohesionless)}
        \\end{cases}

    Parameters
    ----------
    R_b : float
        Nominal base resistance from bearing capacity theory (force).
    soil_type : str
        Either ``'fine'`` or ``'coarse'``.

    Returns
    -------
    float
        Base resistance at 4% displacement (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-48, p. 481.
    """
    if soil_type == "fine":
        return 0.71 * R_b
    elif soil_type == "coarse":
        return R_b
    else:
        raise ValueError("soil_type must be 'fine' or 'coarse'.")


def equivalent_footing_width(B: float, z2: float) -> float:
    """Width of equivalent footing for pile group settlement (Equation 6-49).

    .. math::
        B' = B + z_2

    Parameters
    ----------
    B : float
        Short dimension of the pile group (length).
    z2 : float
        Depth interval of load spreading at 4V:1H (length).

    Returns
    -------
    float
        Equivalent footing width (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-49, p. 483.
    """
    return B + z2


def equivalent_footing_length(L: float, z2: float) -> float:
    """Length of equivalent footing for pile group settlement (Equation 6-50).

    .. math::
        L' = L + z_2

    Parameters
    ----------
    L : float
        Long dimension of the pile group (length).
    z2 : float
        Depth interval of load spreading at 4V:1H (length).

    Returns
    -------
    float
        Equivalent footing length (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-50, p. 483.
    """
    return L + z2


def stress_change_2V1H(
    Q: float, B_prime: float, L_prime: float, z_prime: float
) -> float:
    """Change in total vertical stress by 2V:1H load spreading (Equation 6-51).

    .. math::
        \\Delta\\sigma_z = \\frac{Q}{(B' + z')(L' + z')}

    Parameters
    ----------
    Q : float
        Applied load (force).
    B_prime : float
        Equivalent footing width (length).
    L_prime : float
        Equivalent footing length (length).
    z_prime : float
        Depth below the equivalent footing (length).

    Returns
    -------
    float
        Change in total vertical stress (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-51, p. 485.
    """
    denom = (B_prime + z_prime) * (L_prime + z_prime)
    if denom <= 0.0:
        raise ValueError("Denominator must be positive.")
    return Q / denom


def stress_change_neutral_plane(
    Q: float, B_prime: float, L_prime: float, z_prime: float,
    delta_sigma_other: float = 0.0
) -> float:
    """Stress change at depth using neutral plane method (Equation 6-52).

    .. math::
        \\Delta\\sigma_z = \\frac{Q}{(B' + z')(L' + z')} + \\Delta\\sigma_{z,other}

    Parameters
    ----------
    Q : float
        Applied load (force).
    B_prime : float
        Equivalent footing width (length).
    L_prime : float
        Equivalent footing length (length).
    z_prime : float
        Depth below equivalent footing (length).
    delta_sigma_other : float, optional
        Additional stress change from other sources (stress), default 0.

    Returns
    -------
    float
        Total stress change (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-52, p. 486.
    """
    return stress_change_2V1H(Q, B_prime, L_prime, z_prime) + delta_sigma_other


def settlement_clay(
    H0: float, C_er: float, C_ec: float,
    sigma_z0_eff: float, sigma_p: float, delta_sigma_z: float
) -> float:
    """Settlement of clay using compression indices (Equation 6-53).

    .. math::
        \\delta_s = C_{\\varepsilon r} H_0 \\log\\frac{\\min(\\sigma'_{z0}+\\Delta\\sigma'_z, \\sigma'_p)}{\\sigma'_{z0}}
        + C_{\\varepsilon c} H_0 \\log\\frac{\\max(\\sigma'_{z0}+\\Delta\\sigma'_z, \\sigma'_p)}{\\sigma'_p}

    Parameters
    ----------
    H0 : float
        Initial thickness of consolidating layer (length).
    C_er : float
        Modified recompression index (dimensionless).
    C_ec : float
        Modified compression index (dimensionless).
    sigma_z0_eff : float
        Initial effective vertical stress (stress).
    sigma_p : float
        Preconsolidation stress (stress).
    delta_sigma_z : float
        Change in effective vertical stress (stress).

    Returns
    -------
    float
        Settlement (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-53, p. 486.
    """
    if sigma_z0_eff <= 0.0:
        raise ValueError("sigma_z0_eff must be positive.")
    if sigma_p <= 0.0:
        raise ValueError("sigma_p must be positive.")
    sigma_final = sigma_z0_eff + delta_sigma_z
    if sigma_final <= 0.0:
        raise ValueError("Final effective stress must be positive.")

    settlement = 0.0
    if sigma_final <= sigma_p:
        # Entirely in recompression range
        settlement = C_er * H0 * math.log10(sigma_final / sigma_z0_eff)
    elif sigma_z0_eff >= sigma_p:
        # Entirely in virgin compression range
        settlement = C_ec * H0 * math.log10(sigma_final / sigma_z0_eff)
    else:
        # Spans both ranges
        settlement = (C_er * H0 * math.log10(sigma_p / sigma_z0_eff)
                      + C_ec * H0 * math.log10(sigma_final / sigma_p))
    return settlement


def settlement_sand_elastic(
    H0: float, nu_s: float, E_s: float, delta_sigma_z: float
) -> float:
    """Elastic settlement of coarse-grained soil (Equation 6-54).

    .. math::
        \\delta_s = \\frac{H_0 (1+\\nu_s)(1-2\\nu_s)}{(1-\\nu_s) E_s} \\Delta\\sigma'_z

    Parameters
    ----------
    H0 : float
        Thickness of the sand layer (length).
    nu_s : float
        Poisson's ratio (dimensionless).
    E_s : float
        Young's Modulus of the soil (stress).
    delta_sigma_z : float
        Average change in effective vertical stress (stress).

    Returns
    -------
    float
        Settlement (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-54, p. 487.
    """
    if E_s <= 0.0:
        raise ValueError("E_s must be positive.")
    return (H0 * (1.0 + nu_s) * (1.0 - 2.0 * nu_s)
            / ((1.0 - nu_s) * E_s) * delta_sigma_z)


# ===========================================================================
# LATERAL CAPACITY – BROMS METHOD
# ===========================================================================

def broms_factored_load(P_dead: float, P_live: float) -> float:
    """Factored ultimate lateral load for Broms analysis (Equation 6-55).

    .. math::
        P_{t,ult} = 1.5 P_{t,dead} + 2.0 P_{t,live}

    Parameters
    ----------
    P_dead : float
        Applied lateral load from dead load (force).
    P_live : float
        Applied lateral load from live load (force).

    Returns
    -------
    float
        Factored ultimate lateral load (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-55, p. 494.
    """
    return 1.5 * P_dead + 2.0 * P_live


def broms_factored_moment(M_dead: float, M_live: float) -> float:
    """Factored ultimate moment for Broms analysis (Equation 6-56).

    .. math::
        M_{t,ult} = 1.5 M_{t,dead} + 2.0 M_{t,live}

    Parameters
    ----------
    M_dead : float
        Applied moment from dead load (force*length).
    M_live : float
        Applied moment from live load (force*length).

    Returns
    -------
    float
        Factored ultimate moment (force*length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-56, p. 494.
    """
    return 1.5 * M_dead + 2.0 * M_live


def broms_factored_su(s_u: float) -> float:
    """Factored undrained shear strength for Broms analysis (Equation 6-57).

    .. math::
        s^*_u = 0.75 \\cdot s_u

    Parameters
    ----------
    s_u : float
        Undrained shear strength (stress).

    Returns
    -------
    float
        Factored undrained shear strength (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-57, p. 494.
    """
    return 0.75 * s_u


def broms_factored_phi(phi_deg: float) -> float:
    """Factored friction angle for Broms analysis (Equation 6-58).

    .. math::
        \\phi'^* = \\arctan(0.75 \\tan\\phi')

    Parameters
    ----------
    phi_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    float
        Factored friction angle (degrees).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-58, p. 494.
    """
    return math.degrees(math.atan(0.75 * math.tan(math.radians(phi_deg))))


def broms_undrained_f(P_t_ult: float, s_u_star: float, b: float) -> float:
    """Length f for Broms method in undrained soil (Equation 6-59).

    Length of pile required to resist the applied lateral load.

    .. math::
        f = \\frac{P_{t,ult}}{9 \\cdot s^*_u \\cdot b}

    Parameters
    ----------
    P_t_ult : float
        Factored ultimate lateral load (force).
    s_u_star : float
        Factored undrained shear strength (stress).
    b : float
        Width or diameter of the pile (length).

    Returns
    -------
    float
        Length f (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-59, p. 494.
    """
    if s_u_star <= 0.0:
        raise ValueError("s_u_star must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    return P_t_ult / (9.0 * s_u_star * b)


def broms_undrained_g(
    M_t_ult: float, P_t_ult: float, b: float, f: float,
    s_u_star: float
) -> float:
    """Length g for Broms method in undrained soil (Equation 6-60).

    .. math::
        g = \\sqrt{\\frac{M_{t,ult} + P_{t,ult}(1.5b + 0.5f)}{2.25 \\cdot b \\cdot s^*_u}}

    Parameters
    ----------
    M_t_ult : float
        Factored ultimate moment (force*length).
    P_t_ult : float
        Factored ultimate lateral load (force).
    b : float
        Pile width or diameter (length).
    f : float
        Length f from Equation 6-59 (length).
    s_u_star : float
        Factored undrained shear strength (stress).

    Returns
    -------
    float
        Length g (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-60, p. 495.
    """
    numerator = M_t_ult + P_t_ult * (1.5 * b + 0.5 * f)
    denominator = 2.25 * b * s_u_star
    if denominator <= 0.0:
        raise ValueError("Denominator must be positive.")
    return math.sqrt(numerator / denominator)


def broms_undrained_Zmin(b: float, f: float, g: float) -> float:
    """Minimum pile length for Broms method, undrained (Equation 6-61).

    .. math::
        Z_{min} = 1.5 b + f + g

    Parameters
    ----------
    b : float
        Pile width or diameter (length).
    f : float
        Length f (length).
    g : float
        Length g (length).

    Returns
    -------
    float
        Minimum pile length (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-61, p. 495.
    """
    return 1.5 * b + f + g


def broms_drained_passive(
    gamma_eff: float, b: float, K_P: float, Z_min: float
) -> float:
    """Passive resistance for Broms method, drained soil (Equation 6-62).

    .. math::
        P_P = \\frac{3}{2} \\gamma' b K_P Z_{min}^2

    Parameters
    ----------
    gamma_eff : float
        Effective unit weight (force/length^3).
    b : float
        Pile width or diameter (length).
    K_P : float
        Rankine passive earth pressure coefficient (dimensionless).
    Z_min : float
        Minimum pile length (length).

    Returns
    -------
    float
        Passive resistance (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-62, p. 495.
    """
    return 1.5 * gamma_eff * b * K_P * Z_min ** 2


def broms_drained_Zmin(
    P_t_ult: float, M_t_ult: float, gamma_eff: float,
    b: float, K_P: float, Z_min_guess: float = 10.0,
    tol: float = 1e-6, max_iter: int = 100
) -> float:
    """Minimum pile length for Broms method, drained (Equation 6-63).

    Iteratively solves:

    .. math::
        Z_{min} = \\left(\\frac{2(P_{t,ult} Z_{min} + M_{t,ult})}
        {\\gamma' b K_P}\\right)^{1/3}

    Parameters
    ----------
    P_t_ult : float
        Factored ultimate lateral load (force).
    M_t_ult : float
        Factored ultimate moment (force*length).
    gamma_eff : float
        Effective unit weight (force/length^3).
    b : float
        Pile width or diameter (length).
    K_P : float
        Rankine passive earth pressure coefficient (dimensionless).
    Z_min_guess : float, optional
        Initial guess for Z_min (length), default 10.
    tol : float, optional
        Convergence tolerance, default 1e-6.
    max_iter : int, optional
        Maximum iterations, default 100.

    Returns
    -------
    float
        Minimum pile length (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-63, p. 496.
    """
    denom = gamma_eff * b * K_P
    if denom <= 0.0:
        raise ValueError("gamma_eff, b, and K_P must all be positive.")
    Z = Z_min_guess
    for _ in range(max_iter):
        Z_new = (2.0 * (P_t_ult * Z + M_t_ult) / denom) ** (1.0 / 3.0)
        if abs(Z_new - Z) < tol:
            return Z_new
        Z = Z_new
    return Z


def broms_drained_zero_shear_depth(
    P_t_ult: float, gamma_eff: float, b: float, K_P: float
) -> float:
    """Depth of zero shear for Broms method, drained (Equation 6-64).

    .. math::
        f = \\sqrt{\\frac{P_{t,ult}}{3 \\gamma' b K_P / 2}}

    Parameters
    ----------
    P_t_ult : float
        Factored ultimate lateral load (force).
    gamma_eff : float
        Effective unit weight (force/length^3).
    b : float
        Pile width or diameter (length).
    K_P : float
        Rankine passive earth pressure coefficient (dimensionless).

    Returns
    -------
    float
        Depth of zero shear f (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-64, p. 496.
    """
    denom = 1.5 * gamma_eff * b * K_P
    if denom <= 0.0:
        raise ValueError("gamma_eff, b, and K_P must all be positive.")
    return math.sqrt(P_t_ult / denom)


def broms_drained_max_moment(
    M_t_ult: float, P_t_ult: float, f: float,
    gamma_eff: float, b: float, K_P: float
) -> float:
    """Maximum moment for Broms method, drained (Equation 6-65).

    .. math::
        M_{max,ult} = M_{t,ult} + P_{t,ult} f
        - \\frac{\\gamma' b K_P f^3}{2}

    Parameters
    ----------
    M_t_ult : float
        Factored ultimate moment (force*length).
    P_t_ult : float
        Factored ultimate lateral load (force).
    f : float
        Depth of zero shear (length).
    gamma_eff : float
        Effective unit weight (force/length^3).
    b : float
        Pile width or diameter (length).
    K_P : float
        Rankine passive earth pressure coefficient (dimensionless).

    Returns
    -------
    float
        Maximum moment (force*length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-65, p. 497.
    """
    return M_t_ult + P_t_ult * f - gamma_eff * b * K_P * f ** 3 / 2.0


# ===========================================================================
# LATERAL CAPACITY – CHARACTERISTIC LOAD METHOD (CLM)
# ===========================================================================

def clm_deflection_from_load(
    P_t: float, P_c: float, b: float, a: float, n: float
) -> float:
    """Lateral deflection from applied load – CLM (Equation 6-66).

    .. math::
        \\frac{y_t}{b} = a \\left(\\frac{P_t}{P_c}\\right)^n

    Parameters
    ----------
    P_t : float
        Applied ground line load (force).
    P_c : float
        Characteristic load (force).
    b : float
        Pile width or diameter (length).
    a : float
        Constant from Table 6-36.
    n : float
        Exponent from Table 6-36.

    Returns
    -------
    float
        Lateral deflection at pile top (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-66, p. 500.
    """
    if P_c <= 0.0:
        raise ValueError("P_c must be positive.")
    return b * a * (P_t / P_c) ** n


def clm_load_from_deflection(
    y_t: float, b: float, P_c: float, a: float, n: float
) -> float:
    """Load from deflection – CLM (Equation 6-67).

    .. math::
        P_t = P_c \\left(\\frac{y_t}{a \\cdot b}\\right)^{1/n}

    Parameters
    ----------
    y_t : float
        Lateral deflection at pile top (length).
    b : float
        Pile width or diameter (length).
    P_c : float
        Characteristic load (force).
    a : float
        Constant from Table 6-36.
    n : float
        Exponent from Table 6-36.

    Returns
    -------
    float
        Applied load (force).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-67, p. 500.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive.")
    return P_c * (y_t / (a * b)) ** (1.0 / n)


def clm_deflection_from_moment(
    M_t: float, M_c: float, b: float, a: float, n: float
) -> float:
    """Lateral deflection from applied moment – CLM (Equation 6-68).

    .. math::
        \\frac{y_t}{b} = a \\left(\\frac{M_t}{M_c}\\right)^n

    Parameters
    ----------
    M_t : float
        Applied moment at ground line (force*length).
    M_c : float
        Characteristic moment (force*length).
    b : float
        Pile width or diameter (length).
    a : float
        Constant from Table 6-36.
    n : float
        Exponent from Table 6-36.

    Returns
    -------
    float
        Lateral deflection at pile top (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-68, p. 500.
    """
    if M_c <= 0.0:
        raise ValueError("M_c must be positive.")
    return b * a * (M_t / M_c) ** n


def clm_moment_from_deflection(
    y_t: float, b: float, M_c: float, a: float, n: float
) -> float:
    """Moment from deflection – CLM (Equation 6-69).

    .. math::
        M_t = M_c \\left(\\frac{y_t}{a \\cdot b}\\right)^{1/n}

    Parameters
    ----------
    y_t : float
        Lateral deflection at pile top (length).
    b : float
        Pile width or diameter (length).
    M_c : float
        Characteristic moment (force*length).
    a : float
        Constant from Table 6-36.
    n : float
        Exponent from Table 6-36.

    Returns
    -------
    float
        Applied moment (force*length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-69, p. 500.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive.")
    return M_c * (y_t / (a * b)) ** (1.0 / n)


def clm_max_moment(
    P_t: float, P_c: float, M_c: float, a: float, n: float
) -> float:
    """Maximum moment from applied load – CLM (Equation 6-70).

    .. math::
        \\frac{M_{max}}{M_c} \\approx a \\left(\\frac{P_t}{P_c}\\right)^n

    Parameters
    ----------
    P_t : float
        Applied ground line load (force).
    P_c : float
        Characteristic load (force).
    M_c : float
        Characteristic moment (force*length).
    a : float
        Constant from Table 6-37.
    n : float
        Exponent from Table 6-37.

    Returns
    -------
    float
        Maximum moment (force*length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-70, p. 501.
    """
    if P_c <= 0.0:
        raise ValueError("P_c must be positive.")
    return M_c * a * (P_t / P_c) ** n


def mobilized_passive_resistance(
    P_P_ult: float, y: float, H_cap: float
) -> float:
    """Mobilized passive resistance on pile cap (Equation 6-71).

    Hyperbolic load-displacement relationship from Mokwa (1999).

    .. math::
        P_{P,mob} = \\frac{y}{\\frac{0.006}{P_{P,ult}} + \\frac{0.85 y}{H_{cap} \\cdot P_{P,ult}}}

    Simplified form:

    .. math::
        P_{P,mob} = \\frac{P_{P,ult} \\cdot y}{0.006 H_{cap} + 0.85 y} \\leq P_{P,ult}

    Parameters
    ----------
    P_P_ult : float
        Fully mobilized passive resistance (force).
    y : float
        Average lateral movement of the pile cap (length).
    H_cap : float
        Height of the pile cap (same units as y).

    Returns
    -------
    float
        Mobilized passive resistance (force), capped at P_P_ult.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-71, p. 505.
    """
    if P_P_ult <= 0.0:
        raise ValueError("P_P_ult must be positive.")
    if H_cap <= 0.0:
        raise ValueError("H_cap must be positive.")
    if y <= 0.0:
        return 0.0
    # From the equation: P_mob = y / (0.006/P_ult + 0.85*y/(H_cap*P_ult))
    # Simplify: = P_ult * y * H_cap / (0.006*H_cap + 0.85*y)
    # But checking original: P_mob/P_ult = y / (0.006*H_cap + 0.85*y)
    P_mob = P_P_ult * y / (0.006 * H_cap + 0.85 * y)
    return min(P_mob, P_P_ult)


# ===========================================================================
# STRUCTURAL CAPACITY – DRIVING STRESSES
# ===========================================================================

def driving_stress_limit_steel(phi_da: float, f_y: float) -> float:
    """Maximum driving stress for steel piles (Equation 6-72).

    .. math::
        \\sigma_{dr} \\leq \\phi_{da} \\cdot 0.9 f_y

    Parameters
    ----------
    phi_da : float
        Resistance factor for pile driving (typically 1.0).
    f_y : float
        Yield strength of steel (stress).

    Returns
    -------
    float
        Maximum allowable driving stress (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-72, p. 507.
    """
    return phi_da * 0.9 * f_y


def driving_stress_limit_concrete_compression(
    phi_da: float, f_c_prime: float, f_pe: float
) -> float:
    """Maximum compressive driving stress for concrete piles (Equation 6-73).

    .. math::
        \\sigma_{dr,comp} \\leq \\phi_{da}(0.85 f'_c - f_{pe})

    Parameters
    ----------
    phi_da : float
        Resistance factor for pile driving (typically 1.0).
    f_c_prime : float
        Design compressive strength of concrete (stress).
    f_pe : float
        Effective prestress (stress).

    Returns
    -------
    float
        Maximum allowable compressive driving stress (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-73, p. 508.
    """
    return phi_da * (0.85 * f_c_prime - f_pe)


def driving_stress_limit_concrete_tension(
    phi_da: float, f_c_prime: float, f_pe: float
) -> float:
    """Maximum tensile driving stress for concrete piles (Equation 6-74).

    .. math::
        \\sigma_{dr,tensile} \\leq \\phi_{da}(0.095 \\sqrt{f'_c} + f_{pe})

    Parameters
    ----------
    phi_da : float
        Resistance factor for pile driving (typically 1.0).
    f_c_prime : float
        Design compressive strength of concrete (ksi).
    f_pe : float
        Effective prestress (ksi).

    Returns
    -------
    float
        Maximum allowable tensile driving stress (ksi).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-74, p. 508.
    """
    return phi_da * (0.095 * math.sqrt(f_c_prime) + f_pe)


def driving_stress_limit_timber(phi_da: float, f_cto: float) -> float:
    """Maximum driving stress for timber piles (Equation 6-75).

    .. math::
        \\sigma_{dr} \\leq \\phi_{da} \\cdot 2.6 \\cdot f_{cto}

    Parameters
    ----------
    phi_da : float
        Resistance factor for pile driving (typically 1.15).
    f_cto : float
        Reference compressive strength parallel to grain (stress).

    Returns
    -------
    float
        Maximum allowable driving stress (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-75, p. 508.
    """
    return phi_da * 2.6 * f_cto


# ===========================================================================
# BUCKLING – DEPTH TO FIXITY
# ===========================================================================

def depth_to_fixity_undrained(
    E_p: float, I_p: float, b: float, k_h: float
) -> float:
    """Depth to fixity for undrained soil conditions (Equation 6-76).

    .. math::
        Z_f = 2 \\left(\\frac{E_p I_p}{b \\cdot k_h}\\right)^{0.25}

    Parameters
    ----------
    E_p : float
        Elastic modulus of the pile (force/length^2).
    I_p : float
        Moment of inertia of the pile (length^4).
    b : float
        Pile width (length).
    k_h : float
        Coefficient of horizontal subgrade reaction (force/length^3).

    Returns
    -------
    float
        Depth to fixity (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-76, p. 512.
    """
    if b <= 0.0 or k_h <= 0.0:
        raise ValueError("b and k_h must be positive.")
    return 2.0 * (E_p * I_p / (b * k_h)) ** 0.25


def undrained_subgrade_modulus(C: float, s_u: float) -> float:
    """Soil modulus for undrained buckling analysis (Equation 6-77).

    .. math::
        E_s = b \\cdot k_h = C \\cdot s_u

    Parameters
    ----------
    C : float
        Constant (dimensionless, Davisson 1970 recommends 67).
    s_u : float
        Undrained shear strength (stress).

    Returns
    -------
    float
        Soil modulus E_s = b*k_h (stress).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-77, p. 512.
    """
    return C * s_u


def depth_to_fixity_drained(
    E_p: float, I_p: float, n_h: float
) -> float:
    """Depth to fixity for drained (coarse-grained) soil (Equation 6-78).

    .. math::
        Z_f = 1.8 \\left(\\frac{E_p I_p}{n_h}\\right)^{0.2}

    Parameters
    ----------
    E_p : float
        Elastic modulus of the pile (force/length^2).
    I_p : float
        Moment of inertia of the pile (length^4).
    n_h : float
        Rate of increase of subgrade modulus with depth (force/length^3).

    Returns
    -------
    float
        Depth to fixity (length).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-78, p. 513.
    """
    if n_h <= 0.0:
        raise ValueError("n_h must be positive.")
    return 1.8 * (E_p * I_p / n_h) ** 0.2


# ===========================================================================
# DRAG FORCE / DOWNDRAG
# ===========================================================================

def drag_force_check(
    Q_d: float, Q_np: float, Q_d_drag: float, P_r: float
) -> bool:
    """Check structural strength limit state with drag force (Equation 6-80).

    .. math::
        1.25 Q_d + 1.10 (Q_{np} - Q_d) \\leq P_r

    Parameters
    ----------
    Q_d : float
        Unfactored dead load (force).
    Q_np : float
        Maximum load at neutral plane (force).
    Q_d_drag : float
        Not used directly; Q_np - Q_d is the drag force component.
    P_r : float
        Factored structural resistance (force).

    Returns
    -------
    bool
        True if the check passes (demand <= capacity).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-80, p. 516.
    """
    demand = 1.25 * Q_d + 1.10 * (Q_np - Q_d)
    return demand <= P_r


# ===========================================================================
# PRIVATE HELPER
# ===========================================================================

def _linterp(x: float, xp: list, fp: list) -> float:
    """Pure-Python piecewise linear interpolation."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            t = (x - xp[i]) / (xp[i + 1] - xp[i])
            return fp[i] + t * (fp[i + 1] - fp[i])
    return fp[-1]


# ===========================================================================
# TABLE 6-18: Vesic Strain Factor F_nu
# ===========================================================================

_TABLE_6_18_NU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
_TABLE_6_18_FNU = [1.00, 0.95, 0.90, 0.82, 0.69, 0.50]


def table_6_18_Fnu(poisson_ratio: float) -> float:
    """Vesic strain factor F_nu from Poisson's ratio (Table 6-18).

    Returns the strain factor used in ``volumetric_strain``
    (Equation 6-26) to estimate volumetric strain near the pile base.

    Parameters
    ----------
    poisson_ratio : float
        Poisson's ratio of the soil (0 to 0.5).

    Returns
    -------
    float
        Strain factor F_nu (dimensionless).

    Raises
    ------
    ValueError
        If *poisson_ratio* is outside [0, 0.5].

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-18, p. 461.
    """
    if poisson_ratio < 0.0 or poisson_ratio > 0.5:
        raise ValueError("poisson_ratio must be in [0, 0.5].")
    return _linterp(poisson_ratio, _TABLE_6_18_NU, _TABLE_6_18_FNU)


# ===========================================================================
# TABLES 6-21 / 6-22 / 6-23: LCPC Method Factors
# ===========================================================================

_TABLE_6_21_KS = {
    ("soft_clay", "driven_concrete"): 50,
    ("soft_clay", "driven_steel_closed"): 50,
    ("soft_clay", "driven_steel_open"): 50,
    ("soft_clay", "bored"): 75,
    ("stiff_clay", "driven_concrete"): 40,
    ("stiff_clay", "driven_steel_closed"): 40,
    ("stiff_clay", "driven_steel_open"): 80,
    ("stiff_clay", "bored"): 60,
    ("loose_sand", "driven_concrete"): 60,
    ("loose_sand", "driven_steel_closed"): 60,
    ("loose_sand", "driven_steel_open"): 120,
    ("loose_sand", "bored"): 100,
    ("medium_sand", "driven_concrete"): 100,
    ("medium_sand", "driven_steel_closed"): 100,
    ("medium_sand", "driven_steel_open"): 200,
    ("medium_sand", "bored"): 150,
    ("dense_sand", "driven_concrete"): 150,
    ("dense_sand", "driven_steel_closed"): 150,
    ("dense_sand", "driven_steel_open"): 200,
    ("dense_sand", "bored"): 200,
    ("chalk", "driven_concrete"): 100,
    ("chalk", "driven_steel_closed"): 100,
    ("chalk", "driven_steel_open"): 120,
    ("chalk", "bored"): 60,
    ("marl_limestone", "driven_concrete"): 60,
    ("marl_limestone", "driven_steel_closed"): 60,
    ("marl_limestone", "driven_steel_open"): 80,
    ("marl_limestone", "bored"): 60,
}


def table_6_21_ks(soil_type: str, pile_type: str) -> float:
    """LCPC side resistance factor k_s (Table 6-21).

    Returns the side resistance factor for use with
    ``lcpc_unit_shaft_resistance`` (Equation 6-32).

    Parameters
    ----------
    soil_type : str
        One of ``"soft_clay"``, ``"stiff_clay"``, ``"loose_sand"``,
        ``"medium_sand"``, ``"dense_sand"``, ``"chalk"``,
        ``"marl_limestone"``.
    pile_type : str
        One of ``"driven_concrete"``, ``"driven_steel_closed"``,
        ``"driven_steel_open"``, ``"bored"``.

    Returns
    -------
    float
        Side resistance factor k_s (dimensionless).

    Raises
    ------
    ValueError
        If the combination is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-21, p. 464.
    """
    key = (soil_type.lower().strip(), pile_type.lower().strip())
    if key not in _TABLE_6_21_KS:
        raise ValueError(
            f"Unknown combination soil_type='{soil_type}', "
            f"pile_type='{pile_type}'."
        )
    return float(_TABLE_6_21_KS[key])


_TABLE_6_22_FP = {
    ("soft_clay", "driven_concrete"): 15.0,
    ("soft_clay", "driven_steel_closed"): 15.0,
    ("soft_clay", "driven_steel_open"): 15.0,
    ("soft_clay", "bored"): 15.0,
    ("stiff_clay", "driven_concrete"): 40.0,
    ("stiff_clay", "driven_steel_closed"): 40.0,
    ("stiff_clay", "driven_steel_open"): 40.0,
    ("stiff_clay", "bored"): 40.0,
    ("loose_sand", "driven_concrete"): 35.0,
    ("loose_sand", "driven_steel_closed"): 35.0,
    ("loose_sand", "driven_steel_open"): 35.0,
    ("loose_sand", "bored"): 35.0,
    ("medium_sand", "driven_concrete"): 80.0,
    ("medium_sand", "driven_steel_closed"): 80.0,
    ("medium_sand", "driven_steel_open"): 80.0,
    ("medium_sand", "bored"): 80.0,
    ("dense_sand", "driven_concrete"): 120.0,
    ("dense_sand", "driven_steel_closed"): 120.0,
    ("dense_sand", "driven_steel_open"): 120.0,
    ("dense_sand", "bored"): 120.0,
    ("chalk", "driven_concrete"): 40.0,
    ("chalk", "driven_steel_closed"): 40.0,
    ("chalk", "driven_steel_open"): 40.0,
    ("chalk", "bored"): 40.0,
    ("marl_limestone", "driven_concrete"): 40.0,
    ("marl_limestone", "driven_steel_closed"): 40.0,
    ("marl_limestone", "driven_steel_open"): 40.0,
    ("marl_limestone", "bored"): 40.0,
}


def table_6_22_fp(soil_type: str, pile_type: str) -> float:
    """LCPC maximum unit side resistance f_p (Table 6-22).

    Returns the limiting unit side resistance (kPa) for use with
    ``lcpc_unit_shaft_resistance`` (Equation 6-32).

    Parameters
    ----------
    soil_type : str
        Same keys as ``table_6_21_ks``.
    pile_type : str
        Same keys as ``table_6_21_ks``.

    Returns
    -------
    float
        Maximum unit side resistance f_p (kPa).

    Raises
    ------
    ValueError
        If the combination is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-22, p. 464.
    """
    key = (soil_type.lower().strip(), pile_type.lower().strip())
    if key not in _TABLE_6_22_FP:
        raise ValueError(
            f"Unknown combination soil_type='{soil_type}', "
            f"pile_type='{pile_type}'."
        )
    return _TABLE_6_22_FP[key]


_TABLE_6_23_KT = {
    ("soft_clay", "driven"): 10,
    ("soft_clay", "bored"): 10,
    ("stiff_clay", "driven"): 15,
    ("stiff_clay", "bored"): 15,
    ("loose_sand", "driven"): 40,
    ("loose_sand", "bored"): 20,
    ("medium_sand", "driven"): 80,
    ("medium_sand", "bored"): 40,
    ("dense_sand", "driven"): 120,
    ("dense_sand", "bored"): 80,
    ("chalk", "driven"): 20,
    ("chalk", "bored"): 20,
    ("marl_limestone", "driven"): 40,
    ("marl_limestone", "bored"): 40,
}


def table_6_23_kt(soil_type: str, pile_type: str) -> float:
    """LCPC base bearing factor k_t (Table 6-23).

    Returns the base bearing factor for use with
    ``lcpc_unit_base_resistance`` (Equation 6-33).

    Parameters
    ----------
    soil_type : str
        Same keys as ``table_6_21_ks``.
    pile_type : str
        ``"driven"`` or ``"bored"``.

    Returns
    -------
    float
        Base bearing factor k_t (dimensionless).

    Raises
    ------
    ValueError
        If the combination is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-23, p. 464.
    """
    key = (soil_type.lower().strip(), pile_type.lower().strip())
    if key not in _TABLE_6_23_KT:
        raise ValueError(
            f"Unknown combination soil_type='{soil_type}', "
            f"pile_type='{pile_type}'."
        )
    return float(_TABLE_6_23_KT[key])


# ===========================================================================
# TABLE 6-36: CLM Deflection Constants (a, n)
# ===========================================================================

_TABLE_6_36_CLM = {
    ("clay", "free", "deflection_from_load"): {"a": 0.0075, "n": 1.85},
    ("clay", "fixed", "deflection_from_load"): {"a": 0.0052, "n": 1.82},
    ("clay", "free", "deflection_from_moment"): {"a": 0.031, "n": 1.88},
    ("clay", "fixed", "deflection_from_moment"): {"a": 0.0090, "n": 1.46},
    ("sand", "free", "deflection_from_load"): {"a": 0.0038, "n": 2.10},
    ("sand", "fixed", "deflection_from_load"): {"a": 0.0021, "n": 2.15},
    ("sand", "free", "deflection_from_moment"): {"a": 0.015, "n": 2.10},
    ("sand", "fixed", "deflection_from_moment"): {"a": 0.0042, "n": 1.69},
}


def table_6_36_clm(
    soil_type: str, head_condition: str, quantity: str
) -> dict:
    """CLM deflection constants a and n (Table 6-36).

    Returns the empirical constants for Equations 6-66 through 6-69.

    Parameters
    ----------
    soil_type : str
        ``"clay"`` or ``"sand"``.
    head_condition : str
        ``"free"`` or ``"fixed"``.
    quantity : str
        ``"deflection_from_load"`` or ``"deflection_from_moment"``.

    Returns
    -------
    dict
        ``{"a": float, "n": float}``.

    Raises
    ------
    ValueError
        If the combination is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-36, p. 500.
    """
    key = (
        soil_type.lower().strip(),
        head_condition.lower().strip(),
        quantity.lower().strip(),
    )
    if key not in _TABLE_6_36_CLM:
        raise ValueError(
            f"Unknown combination soil_type='{soil_type}', "
            f"head_condition='{head_condition}', quantity='{quantity}'."
        )
    return dict(_TABLE_6_36_CLM[key])


# ===========================================================================
# TABLE 6-37: CLM Maximum Moment Constants (a, n)
# ===========================================================================

_TABLE_6_37_CLM = {
    ("clay", "free"): {"a": 0.55, "n": 0.72},
    ("clay", "fixed"): {"a": 0.47, "n": 0.71},
    ("sand", "free"): {"a": 0.57, "n": 0.82},
    ("sand", "fixed"): {"a": 0.40, "n": 0.82},
}


def table_6_37_clm(soil_type: str, head_condition: str) -> dict:
    """CLM maximum moment constants a and n (Table 6-37).

    Returns the empirical constants for Equation 6-70.

    Parameters
    ----------
    soil_type : str
        ``"clay"`` or ``"sand"``.
    head_condition : str
        ``"free"`` or ``"fixed"``.

    Returns
    -------
    dict
        ``{"a": float, "n": float}``.

    Raises
    ------
    ValueError
        If the combination is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Table 6-37, p. 501.
    """
    key = (soil_type.lower().strip(), head_condition.lower().strip())
    if key not in _TABLE_6_37_CLM:
        raise ValueError(
            f"Unknown combination soil_type='{soil_type}', "
            f"head_condition='{head_condition}'."
        )
    return dict(_TABLE_6_37_CLM[key])
