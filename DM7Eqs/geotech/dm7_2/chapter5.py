"""
UFC 3-220-20, Chapter 5: Shallow Foundations

Equations 5-1 through 5-42 covering gross and net bearing pressure,
eccentricity (effective foundation dimensions), allowable bearing pressure
and factor of safety, bearing capacity theory (Terzaghi, Meyerhof, Hansen,
Vesic bearing capacity factors Nc, Nq, Ngamma), shape/depth/inclination
correction factors, groundwater correction, foundations near slopes,
layered soil bearing capacity, bearing capacity of rock, modulus of
subgrade reaction, and mat/raft foundation design.

Reference:
    UFC 3-220-20, Foundations, 16 January 2025
"""

import math
from typing import Tuple


# ===========================================================================
# SECTION 5-2: Gross and Net Bearing Pressure (Equations 5-1 through 5-3)
# ===========================================================================


def gross_bearing_pressure(
    Q_DL_LL: float, W_F: float, W_S: float, A: float
) -> float:
    """Gross bearing pressure on a shallow foundation (Equation 5-1).

    The gross bearing pressure applied to the soil by a shallow foundation
    is the total load (structural load + foundation weight + overburden soil
    weight) divided by the bearing surface area.

    .. math::
        q_{gross} = \\frac{Q_{DL+LL} + W_F + W_S}{A}

    Parameters
    ----------
    Q_DL_LL : float
        Structural load: dead load plus live load (lb, kN, or any
        consistent force unit).
    W_F : float
        Weight of the foundation (same force unit as *Q_DL_LL*).
    W_S : float
        Weight of overlying soil (same force unit as *Q_DL_LL*).
    A : float
        Area of the bearing surface (ft^2, m^2, or consistent area unit).

    Returns
    -------
    float
        Gross bearing pressure (psf, kPa, or consistent stress unit).

    Raises
    ------
    ValueError
        If *A* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-1, p. 275.
    """
    if A <= 0.0:
        raise ValueError("Bearing surface area A must be positive.")
    return (Q_DL_LL + W_F + W_S) / A


def net_bearing_pressure_from_ultimate(
    q_ult: float, sigma_zD: float
) -> float:
    """Net bearing pressure from ultimate bearing capacity (Equation 5-2).

    The net bearing pressure is the ultimate bearing capacity minus the
    existing vertical overburden pressure at the foundation bearing
    elevation.

    .. math::
        q_{net} = q_{ult} - \\sigma_{zD}

    Parameters
    ----------
    q_ult : float
        Ultimate bearing capacity (psf, kPa, or consistent stress unit).
    sigma_zD : float
        Vertical overburden pressure at the bearing elevation (same unit).

    Returns
    -------
    float
        Net bearing pressure (same stress unit).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-2, p. 275.
    """
    return q_ult - sigma_zD


def net_bearing_pressure(Q_DL_LL: float, A: float) -> float:
    """Simplified net bearing pressure (Equation 5-3).

    When the unit weights of the foundation and soil backfill are assumed
    to be the same as the existing soil, the net bearing pressure simplifies
    to the structural load divided by the foundation area.

    .. math::
        q_{net} \\approx \\frac{Q_{DL+LL}}{A}

    Parameters
    ----------
    Q_DL_LL : float
        Structural load: dead load plus live load (lb, kN, or consistent
        force unit).
    A : float
        Area of the bearing surface (ft^2, m^2, or consistent area unit).

    Returns
    -------
    float
        Net bearing pressure (psf, kPa, or consistent stress unit).

    Raises
    ------
    ValueError
        If *A* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-3, p. 276.
    """
    if A <= 0.0:
        raise ValueError("Bearing surface area A must be positive.")
    return Q_DL_LL / A


# ===========================================================================
# SECTION 5-2.3: Eccentricity (Equations 5-4 through 5-12)
# ===========================================================================


def eccentricity(M: float, Q: float) -> float:
    """Eccentricity of foundation loading (Equation 5-4).

    The eccentricity is the horizontal distance between the resultant
    force on the bearing surface and the centerline of the foundation.

    .. math::
        e = \\frac{M}{Q}

    Parameters
    ----------
    M : float
        Applied moment (lb-ft, kN-m, or consistent moment unit).
    Q : float
        Gross vertical load on the footing (lb, kN, or consistent force
        unit).

    Returns
    -------
    float
        Eccentricity (ft, m, or consistent length unit).

    Raises
    ------
    ValueError
        If *Q* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-4, p. 276.
    """
    if Q <= 0.0:
        raise ValueError("Gross vertical load Q must be positive.")
    return M / Q


def check_eccentricity_one_direction(e: float, dimension: float) -> bool:
    """Check one-direction eccentricity limit (Equation 5-5).

    For eccentricity in one direction, the resultant must be in the middle
    one-third of the foundation.

    .. math::
        e_B \\leq \\frac{B}{6} \\quad \\text{or} \\quad e_L \\leq \\frac{L}{6}

    Parameters
    ----------
    e : float
        Eccentricity in B or L direction (ft, m).
    dimension : float
        Corresponding foundation dimension B or L (ft, m).

    Returns
    -------
    bool
        True if eccentricity is within the middle-third limit.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-5, p. 276.
    """
    if dimension <= 0.0:
        raise ValueError("Foundation dimension must be positive.")
    return abs(e) <= dimension / 6.0


def check_eccentricity_two_directions(
    e_B: float, e_L: float, B: float, L: float
) -> bool:
    """Check two-direction eccentricity kern limit (Equation 5-6).

    For cases with eccentricity in two directions, the resultant must fall
    within the diamond-shaped kern area in the middle of the rectangular
    foundation.

    .. math::
        \\frac{6 \\, e_B}{B} + \\frac{6 \\, e_L}{L} \\leq 1

    Parameters
    ----------
    e_B : float
        Eccentricity in B direction (ft, m).
    e_L : float
        Eccentricity in L direction (ft, m).
    B : float
        Foundation width -- shorter dimension (ft, m).
    L : float
        Foundation length -- longer dimension (ft, m).

    Returns
    -------
    bool
        True if eccentricity is within the kern limit.

    Raises
    ------
    ValueError
        If *B* or *L* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-6, p. 276.
    """
    if B <= 0.0 or L <= 0.0:
        raise ValueError("Foundation dimensions B and L must be positive.")
    return (6.0 * abs(e_B) / B + 6.0 * abs(e_L) / L) <= 1.0


def corner_bearing_pressure(
    q_gross: float, e_B: float, e_L: float, B: float, L: float
) -> Tuple[float, float, float, float]:
    """Corner bearing pressures for eccentric loading (Equation 5-7).

    Returns the bearing pressure at the four corners of a rectangular
    foundation with two-way eccentricity within the kern.

    .. math::
        q_{corner} = q_{gross} \\left(1 \\pm \\frac{6 e_B}{B}
                      \\pm \\frac{6 e_L}{L}\\right)

    Parameters
    ----------
    q_gross : float
        Gross bearing pressure (psf, kPa, or consistent stress unit).
    e_B : float
        Eccentricity in B direction (ft, m).
    e_L : float
        Eccentricity in L direction (ft, m).
    B : float
        Foundation width -- shorter dimension (ft, m).
    L : float
        Foundation length -- longer dimension (ft, m).

    Returns
    -------
    tuple of float
        (q1, q2, q3, q4) bearing pressures at four corners:
        q1 = +eB, +eL corner (maximum);
        q2 = +eB, -eL corner;
        q3 = -eB, +eL corner;
        q4 = -eB, -eL corner (minimum).

    Raises
    ------
    ValueError
        If *B* or *L* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-7, p. 277.
    """
    if B <= 0.0 or L <= 0.0:
        raise ValueError("Foundation dimensions B and L must be positive.")
    term_B = 6.0 * e_B / B
    term_L = 6.0 * e_L / L
    q1 = q_gross * (1.0 + term_B + term_L)
    q2 = q_gross * (1.0 + term_B - term_L)
    q3 = q_gross * (1.0 - term_B + term_L)
    q4 = q_gross * (1.0 - term_B - term_L)
    return (q1, q2, q3, q4)


def effective_foundation_width(B: float, e_B: float) -> float:
    """Effective (equivalent) foundation width (Equation 5-8).

    The equivalent width is the width of the bearing area on which the
    resultant bearing force is centered (Meyerhof 1953).

    .. math::
        B' = B - 2 \\, e_B

    Parameters
    ----------
    B : float
        Actual foundation width (ft, m).
    e_B : float
        Eccentricity in B direction (ft, m).

    Returns
    -------
    float
        Effective foundation width B' (ft, m).

    Raises
    ------
    ValueError
        If resulting B' is non-positive.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-8, p. 277.
    """
    B_prime = B - 2.0 * abs(e_B)
    if B_prime <= 0.0:
        raise ValueError(
            "Effective width B' is non-positive; eccentricity is too large."
        )
    return B_prime


def effective_foundation_length(L: float, e_L: float) -> float:
    """Effective (equivalent) foundation length (Equation 5-9).

    The equivalent length is the length of the bearing area on which the
    resultant bearing force is centered (Meyerhof 1953).

    .. math::
        L' = L - 2 \\, e_L

    Parameters
    ----------
    L : float
        Actual foundation length (ft, m).
    e_L : float
        Eccentricity in L direction (ft, m).

    Returns
    -------
    float
        Effective foundation length L' (ft, m).

    Raises
    ------
    ValueError
        If resulting L' is non-positive.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-9, p. 277.
    """
    L_prime = L - 2.0 * abs(e_L)
    if L_prime <= 0.0:
        raise ValueError(
            "Effective length L' is non-positive; eccentricity is too large."
        )
    return L_prime


def equivalent_uniform_bearing_pressure(
    Q_DL_LL: float, W_F: float, W_S: float, B_prime: float, L_prime: float
) -> float:
    """Equivalent uniform bearing pressure for eccentric loading (Equation 5-10).

    The equivalent uniform bearing pressure for a rectangular foundation
    using the equivalent (effective) dimensions.

    .. math::
        q_{unif} = \\frac{Q_{DL+LL} + W_F + W_S}{B' \\cdot L'}

    Parameters
    ----------
    Q_DL_LL : float
        Structural load: dead load plus live load (lb, kN).
    W_F : float
        Weight of the foundation (same force unit).
    W_S : float
        Weight of overlying soil (same force unit).
    B_prime : float
        Effective foundation width (ft, m).
    L_prime : float
        Effective foundation length (ft, m).

    Returns
    -------
    float
        Equivalent uniform bearing pressure (psf, kPa).

    Raises
    ------
    ValueError
        If effective dimensions are non-positive.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-10, p. 277.
    """
    if B_prime <= 0.0 or L_prime <= 0.0:
        raise ValueError("Effective dimensions B' and L' must be positive.")
    return (Q_DL_LL + W_F + W_S) / (B_prime * L_prime)


def circular_effective_length(r: float, e_x: float) -> float:
    """Effective length for a circular footing (Equation 5-11).

    The equivalent rectangular length for a circular footing with
    eccentricity, based on the aspect ratio of the circumscribed
    lens-shaped area.

    .. math::
        L' = 2 \\left[ r^2 \\cos^{-1}\\!\\left(\\frac{e_x}{r}\\right)
             - e_x \\sqrt{r^2 - e_x^2}\\,\\right]^{0.5}

    Parameters
    ----------
    r : float
        Radius of the circular foundation (ft, m).
    e_x : float
        Eccentricity (ft, m).  Must be less than *r*.

    Returns
    -------
    float
        Effective length L' (ft, m).

    Raises
    ------
    ValueError
        If *e_x* >= *r* or if *r* is non-positive.

    Notes
    -----
    The inverse cosine term must be expressed in radians.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-11, p. 277.
    """
    if r <= 0.0:
        raise ValueError("Radius r must be positive.")
    e_x = abs(e_x)
    if e_x >= r:
        raise ValueError("Eccentricity e_x must be less than radius r.")
    area = r * r * math.acos(e_x / r) - e_x * math.sqrt(r * r - e_x * e_x)
    return 2.0 * math.sqrt(area)


def circular_effective_width(r: float, e_x: float, L_prime: float) -> float:
    """Effective width for a circular footing (Equation 5-12).

    The equivalent rectangular width for a circular footing with
    eccentricity, derived from the aspect ratio of the circumscribed
    lens-shaped area.

    .. math::
        B' = \\frac{r^2 - e_x^2}{L'} \\cdot \\frac{\\text{(area ratio)}}{1}

    More precisely:

    .. math::
        B' = \\frac{\\sqrt{r^2 - e_x^2}}{L'} \\cdot \\text{(lens area / aspect)}

    Parameters
    ----------
    r : float
        Radius of the circular foundation (ft, m).
    e_x : float
        Eccentricity (ft, m).
    L_prime : float
        Effective length from Equation 5-11 (ft, m).

    Returns
    -------
    float
        Effective width B' (ft, m).

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-12, p. 278.
    """
    if r <= 0.0:
        raise ValueError("Radius r must be positive.")
    e_x = abs(e_x)
    if e_x >= r:
        raise ValueError("Eccentricity e_x must be less than radius r.")
    if L_prime <= 0.0:
        raise ValueError("Effective length L' must be positive.")
    # The lens-shaped area = r^2 * acos(ex/r) - ex * sqrt(r^2 - ex^2)
    # B' = area / L' (preserving the aspect ratio)
    area = r * r * math.acos(e_x / r) - e_x * math.sqrt(r * r - e_x * e_x)
    return area / L_prime


# ===========================================================================
# SECTION 5-2.4: Allowable Bearing Pressure (Equations 5-13 and 5-14)
# ===========================================================================


def gross_allowable_bearing_pressure(q_ult: float, F_BC: float) -> float:
    """Gross allowable bearing pressure (Equation 5-13).

    The gross allowable bearing pressure is the ultimate bearing capacity
    divided by the bearing capacity factor of safety.

    .. math::
        q_{all,gross} = \\frac{q_{ult}}{F_{BC}}

    Parameters
    ----------
    q_ult : float
        Ultimate bearing capacity (psf, kPa, or consistent stress unit).
    F_BC : float
        Bearing capacity factor of safety (dimensionless, typically 2 to 4).

    Returns
    -------
    float
        Gross allowable bearing pressure (same stress unit as *q_ult*).

    Raises
    ------
    ValueError
        If *F_BC* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-13, p. 279.
    """
    if F_BC <= 0.0:
        raise ValueError("Factor of safety F_BC must be positive.")
    return q_ult / F_BC


def net_allowable_bearing_pressure(
    q_ult: float, F_BC: float, sigma_zD: float
) -> float:
    """Net allowable bearing pressure (Equation 5-14).

    The net allowable bearing pressure accounts for the overburden
    pressure at the bearing elevation (Peck et al. 1974).

    .. math::
        q_{all,net} \\leq \\frac{q_{ult}}{F_{BC}} - \\sigma_{zD}

    Parameters
    ----------
    q_ult : float
        Ultimate bearing capacity (psf, kPa, or consistent stress unit).
    F_BC : float
        Bearing capacity factor of safety (dimensionless).
    sigma_zD : float
        Vertical overburden pressure at bearing elevation (same stress
        unit).

    Returns
    -------
    float
        Net allowable bearing pressure (same stress unit).

    Raises
    ------
    ValueError
        If *F_BC* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-14, p. 279.
    """
    if F_BC <= 0.0:
        raise ValueError("Factor of safety F_BC must be positive.")
    return q_ult / F_BC - sigma_zD


# ===========================================================================
# SECTION 5-3.1.2: Ultimate Bearing Capacity (Equations 5-15 through 5-17)
# ===========================================================================


def ultimate_bearing_capacity_drained(
    c_prime: float,
    sigma_zD_prime: float,
    gamma: float,
    B: float,
    Nc: float,
    Nq: float,
    N_gamma: float,
    psi_c: float = 1.0,
    psi_q: float = 1.0,
    psi_gamma: float = 1.0,
) -> float:
    """Ultimate bearing capacity -- effective stress (drained) analysis (Equation 5-15).

    For effective stress or drained analysis, the ultimate bearing capacity
    of the soil uses effective stress parameters and bearing capacity
    factors.

    .. math::
        q_{ult} = c' N_c \\Psi_c + \\sigma'_{zD} N_q \\Psi_q
                  + 0.5 \\gamma B N_\\gamma \\Psi_\\gamma

    Parameters
    ----------
    c_prime : float
        Effective stress cohesion (psf, kPa).
    sigma_zD_prime : float
        Effective vertical stress at bearing elevation (psf, kPa).
    gamma : float
        Average effective unit weight of soil between Df and Df + B
        (pcf, kN/m^3).
    B : float
        Foundation width (ft, m).  Use B' for eccentric loading.
    Nc : float
        Bearing capacity factor for cohesion term (dimensionless).
    Nq : float
        Bearing capacity factor for overburden term (dimensionless).
    N_gamma : float
        Bearing capacity factor for unit weight term (dimensionless).
    psi_c : float, optional
        Lumped correction factor for Nc term (default 1.0).
    psi_q : float, optional
        Lumped correction factor for Nq term (default 1.0).
    psi_gamma : float, optional
        Lumped correction factor for N_gamma term (default 1.0).

    Returns
    -------
    float
        Ultimate bearing capacity (psf, kPa).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-15, p. 288.
    """
    return (
        c_prime * Nc * psi_c
        + sigma_zD_prime * Nq * psi_q
        + 0.5 * gamma * B * N_gamma * psi_gamma
    )


def ultimate_bearing_capacity_undrained_unsaturated(
    c: float,
    sigma_zD: float,
    gamma: float,
    B: float,
    Nc: float,
    Nq: float,
    N_gamma: float,
    psi_c: float = 1.0,
    psi_q: float = 1.0,
    psi_gamma: float = 1.0,
) -> float:
    """Ultimate bearing capacity -- undrained, unsaturated soils (Equation 5-16).

    For undrained conditions in unsaturated soils, the ultimate bearing
    capacity is calculated using total stress parameters (c, phi).

    .. math::
        q_{ult} = c \\cdot N_c \\Psi_c + \\sigma_{zD} N_q \\Psi_q
                  + 0.5 \\gamma B N_\\gamma \\Psi_\\gamma

    Parameters
    ----------
    c : float
        Undrained cohesion (psf, kPa).
    sigma_zD : float
        Total vertical stress at bearing elevation (psf, kPa).
    gamma : float
        Average total unit weight of soil between Df and Df + B
        (pcf, kN/m^3).
    B : float
        Foundation width (ft, m).
    Nc : float
        Bearing capacity factor for cohesion term (dimensionless).
    Nq : float
        Bearing capacity factor for overburden term (dimensionless).
    N_gamma : float
        Bearing capacity factor for unit weight term (dimensionless).
    psi_c : float, optional
        Lumped correction factor for Nc term (default 1.0).
    psi_q : float, optional
        Lumped correction factor for Nq term (default 1.0).
    psi_gamma : float, optional
        Lumped correction factor for N_gamma term (default 1.0).

    Returns
    -------
    float
        Ultimate bearing capacity (psf, kPa).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-16, p. 288.
    """
    return (
        c * Nc * psi_c
        + sigma_zD * Nq * psi_q
        + 0.5 * gamma * B * N_gamma * psi_gamma
    )


def ultimate_bearing_capacity_undrained_saturated(
    s_u: float,
    sigma_zD: float,
    Nc: float,
    psi_c: float = 1.0,
    psi_q: float = 1.0,
) -> float:
    """Ultimate bearing capacity -- undrained, saturated soils (Equation 5-17).

    For undrained conditions in saturated soils (phi = 0), the bearing
    capacity simplifies to the cohesion and overburden terms only (Nq = 1,
    N_gamma = 0 for phi = 0).

    .. math::
        q_{ult} = s_u N_c \\Psi_c + \\sigma_{zD} \\Psi_q

    Parameters
    ----------
    s_u : float
        Undrained shear strength (psf, kPa).
    sigma_zD : float
        Total vertical stress at bearing elevation (psf, kPa).
    Nc : float
        Bearing capacity factor for phi = 0 (5.14 for Meyerhof/Hansen,
        5.7 for Terzaghi).
    psi_c : float, optional
        Lumped correction factor for Nc term (default 1.0).
    psi_q : float, optional
        Lumped correction factor for Nq term (default 1.0).  Nq = 1
        for phi = 0.

    Returns
    -------
    float
        Ultimate bearing capacity (psf, kPa).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-17, p. 288.
    """
    return s_u * Nc * psi_c + sigma_zD * psi_q


# ===========================================================================
# SECTION 5-3.3: Lumped Correction Factors (Equations 5-18 through 5-20)
# ===========================================================================


def lumped_correction_factor_c(
    sc: float = 1.0,
    dc: float = 1.0,
    ic: float = 1.0,
    bc: float = 1.0,
    gc: float = 1.0,
) -> float:
    """Lumped correction factor for Nc term (Equation 5-18).

    Combines individual correction factors for shape, depth, load
    inclination, base inclination, and ground inclination.

    .. math::
        \\Psi_c = s_c \\cdot d_c \\cdot i_c \\cdot b_c \\cdot g_c

    Parameters
    ----------
    sc : float, optional
        Shape factor for Nc (default 1.0).
    dc : float, optional
        Depth factor for Nc (default 1.0).
    ic : float, optional
        Load inclination factor for Nc (default 1.0).
    bc : float, optional
        Sloping base factor for Nc (default 1.0).
    gc : float, optional
        Sloping ground factor for Nc (default 1.0).

    Returns
    -------
    float
        Lumped correction factor psi_c (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-18, p. 293.
    """
    return sc * dc * ic * bc * gc


def lumped_correction_factor_q(
    sq: float = 1.0,
    dq: float = 1.0,
    iq: float = 1.0,
    bq: float = 1.0,
    gq: float = 1.0,
) -> float:
    """Lumped correction factor for Nq term (Equation 5-19).

    Combines individual correction factors for shape, depth, load
    inclination, base inclination, and ground inclination.

    .. math::
        \\Psi_q = s_q \\cdot d_q \\cdot i_q \\cdot b_q \\cdot g_q

    Parameters
    ----------
    sq : float, optional
        Shape factor for Nq (default 1.0).
    dq : float, optional
        Depth factor for Nq (default 1.0).
    iq : float, optional
        Load inclination factor for Nq (default 1.0).
    bq : float, optional
        Sloping base factor for Nq (default 1.0).
    gq : float, optional
        Sloping ground factor for Nq (default 1.0).

    Returns
    -------
    float
        Lumped correction factor psi_q (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-19, p. 293.
    """
    return sq * dq * iq * bq * gq


def lumped_correction_factor_gamma(
    s_gamma: float = 1.0,
    d_gamma: float = 1.0,
    i_gamma: float = 1.0,
    b_gamma: float = 1.0,
    g_gamma: float = 1.0,
) -> float:
    """Lumped correction factor for N_gamma term (Equation 5-20).

    Combines individual correction factors for shape, depth, load
    inclination, base inclination, and ground inclination.

    .. math::
        \\Psi_\\gamma = s_\\gamma \\cdot d_\\gamma \\cdot i_\\gamma
                       \\cdot b_\\gamma \\cdot g_\\gamma

    Parameters
    ----------
    s_gamma : float, optional
        Shape factor for N_gamma (default 1.0).
    d_gamma : float, optional
        Depth factor for N_gamma (default 1.0).
    i_gamma : float, optional
        Load inclination factor for N_gamma (default 1.0).
    b_gamma : float, optional
        Sloping base factor for N_gamma (default 1.0).
    g_gamma : float, optional
        Sloping ground factor for N_gamma (default 1.0).

    Returns
    -------
    float
        Lumped correction factor psi_gamma (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-20, p. 293.
    """
    return s_gamma * d_gamma * i_gamma * b_gamma * g_gamma


# ===========================================================================
# SECTION 5-3.3.3: Brinch Hansen phi=0 Correction (Equation 5-21)
# ===========================================================================


def hansen_correction_factor_c_phi0(
    sc: float = 0.0,
    dc: float = 0.0,
    ic: float = 0.0,
    bc: float = 0.0,
    gc: float = 0.0,
) -> float:
    """Brinch Hansen lumped correction factor for Nc term, phi = 0 (Equation 5-21).

    For the Brinch Hansen method with undrained shear strength (phi = 0),
    the lumped correction factor is additive rather than multiplicative.

    .. math::
        \\Psi_{c,BH,\\phi=0} = 1 + s_c + d_c - i_c - b_c - g_c

    Parameters
    ----------
    sc : float, optional
        Shape factor for Nc, phi = 0 (default 0.0).
        For vertical load: sc = 0.2 * (B/L).
    dc : float, optional
        Depth factor for Nc, phi = 0 (default 0.0).
        dc = 0.4 * k where k = Df/B for Df/B < 1,
        k = atan(Df/B) in radians for Df/B >= 1.
    ic : float, optional
        Inclination factor for Nc, phi = 0 (default 0.0).
        ic = 0.5 - 0.5 * sqrt(1 - H / (A' * Ca)).
    bc : float, optional
        Sloping base factor for Nc, phi = 0 (default 0.0).
        bc = eta / (pi/2 + 1).
    gc : float, optional
        Sloping ground factor for Nc, phi = 0 (default 0.0).
        gc = beta / (pi/2 + 1).

    Returns
    -------
    float
        Brinch Hansen correction factor psi_c for phi = 0 (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-21, p. 295.
    """
    return 1.0 + sc + dc - ic - bc - gc


# ===========================================================================
# SECTION 5-3.4: Foundations Near Slopes (Equations 5-22 and 5-23)
# ===========================================================================


def stability_number_undrained(gamma: float, H: float, s_u: float) -> float:
    """Stability number for undrained slope conditions (Equation 5-22).

    The stability number incorporates the influence of slope stability on
    the bearing capacity for saturated undrained conditions (su, phi = 0).

    .. math::
        N_S = \\frac{\\gamma \\cdot H}{s_u}

    Parameters
    ----------
    gamma : float
        Total unit weight of the soil (pcf, kN/m^3).
    H : float
        Height of the slope (ft, m).
    s_u : float
        Undrained shear strength of the soil (psf, kPa).

    Returns
    -------
    float
        Stability number N_S (dimensionless).

    Raises
    ------
    ValueError
        If *s_u* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-22, p. 297.
    """
    if s_u <= 0.0:
        raise ValueError("Undrained shear strength s_u must be positive.")
    return gamma * H / s_u


def stability_number_drained(gamma: float, H: float, c_prime: float) -> float:
    """Stability number for drained slope conditions (Equation 5-23).

    The stability number for soils modeled using an effective stress
    cohesion intercept to incorporate slope influence on bearing capacity.

    .. math::
        N_S = \\frac{\\gamma \\cdot H}{c'}

    Parameters
    ----------
    gamma : float
        Unit weight of the soil considering groundwater effects
        (pcf, kN/m^3).
    H : float
        Height of the slope (ft, m).
    c_prime : float
        Effective stress cohesion intercept (psf, kPa).

    Returns
    -------
    float
        Stability number N_S (dimensionless).

    Raises
    ------
    ValueError
        If *c_prime* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-23, p. 299.
    """
    if c_prime <= 0.0:
        raise ValueError("Effective cohesion c' must be positive.")
    return gamma * H / c_prime


# ===========================================================================
# SECTION 5-3.6.1: Layered Soils -- Increasing su with Depth (Equation 5-24)
# ===========================================================================


def bearing_capacity_factor_Nc_increasing_su(
    k: float, B: float, s_u0: float
) -> float:
    """Modified Nc for clay with increasing su with depth (Equation 5-24).

    Based on an upper bound plasticity solution (Chi and Lin 2020), the
    bearing capacity factor Nc for a perfectly smooth strip footing on
    clay with increasing undrained strength with depth.

    .. math::
        N_c = 5.14 + \\frac{k \\cdot B}{s_{u0}}

    Parameters
    ----------
    k : float
        Rate of increase in undrained shear strength with depth
        (psf/ft, kPa/m).
    B : float
        Footing width (ft, m).
    s_u0 : float
        Undrained shear strength at the surface of the clay layer
        (psf, kPa).

    Returns
    -------
    float
        Modified bearing capacity factor Nc (dimensionless).

    Raises
    ------
    ValueError
        If *s_u0* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-24, p. 307.
    """
    if s_u0 <= 0.0:
        raise ValueError("Surface undrained shear strength s_u0 must be positive.")
    return 5.14 + k * B / s_u0


# ===========================================================================
# SECTION 5-3.6.2: Layered Clay -- Modified Nc (Equation 5-25)
# ===========================================================================


def modified_Nc_rectangular_layered_clay(
    Nc_m_s: float, Nc_m_c: float, B: float, L: float
) -> float:
    """Modified Nc for rectangular footing on layered clay (Equation 5-25).

    For a layered clay soil profile, the modified bearing capacity factor
    for a rectangular foundation (Nc,m,r) can be estimated from the
    circular and strip factors.

    .. math::
        N_{c,m,r} = N_{c,m,c} + \\left(1 - \\frac{B}{L}\\right)
                    \\left(N_{c,m,s} - N_{c,m,c}\\right)

    Equivalently:

    .. math::
        N_{c,m,r} = N_{c,m,s} + \\frac{B}{L}
                    \\left(N_{c,m,c} - N_{c,m,s}\\right)

    Parameters
    ----------
    Nc_m_s : float
        Modified bearing capacity factor for strip footing from
        Figure 5-18(a) (dimensionless).
    Nc_m_c : float
        Modified bearing capacity factor for circular footing from
        Figure 5-18(b) (dimensionless).
    B : float
        Rectangular foundation width (ft, m).
    L : float
        Rectangular foundation length (ft, m).

    Returns
    -------
    float
        Modified bearing capacity factor Nc,m,r for rectangular footing
        (dimensionless).

    Raises
    ------
    ValueError
        If *L* is zero or negative, or if *B* > *L*.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-25, p. 309.
    """
    if L <= 0.0:
        raise ValueError("Foundation length L must be positive.")
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    if B > L:
        raise ValueError("Width B must not exceed length L.")
    return Nc_m_c + (1.0 - B / L) * (Nc_m_s - Nc_m_c)


# ===========================================================================
# SECTION 5-3.6.3: Mixed Soil Layers (Equations 5-26 through 5-28)
# ===========================================================================


def second_layer_thickness(
    B: float,
    H1: float,
    c1: float,
    c2: float,
    phi1_deg: float,
    phi2_deg: float,
) -> float:
    """Thickness of the second layer for mixed soil profiles (Equation 5-26).

    For mixed soil profiles of sand and clay evaluated using the
    Satyanarayana and Garg (1980) method (unsaturated, undrained).

    .. math::
        H_2 = \\frac{(B/2 + H_1 \\tan\\phi_1) \\cdot c_1}{c_2 \\cdot \\tan\\phi_2}

    Parameters
    ----------
    B : float
        Width of strip footing (ft, m).
    H1 : float
        Thickness of top layer below bearing elevation (ft, m).
    c1 : float
        Undrained cohesion of top layer (psf, kPa).
    c2 : float
        Undrained cohesion of bottom layer (psf, kPa).
    phi1_deg : float
        Undrained friction angle of top layer (degrees).
    phi2_deg : float
        Undrained friction angle of bottom layer (degrees).

    Returns
    -------
    float
        Thickness of the second layer H2 (ft, m).

    Raises
    ------
    ValueError
        If *c2* is zero or *phi2_deg* yields zero tangent.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-26, p. 311.
    """
    phi1 = math.radians(phi1_deg)
    phi2 = math.radians(phi2_deg)
    tan_phi2 = math.tan(phi2)
    if c2 == 0.0 or tan_phi2 == 0.0:
        raise ValueError(
            "c2 and tan(phi2) must be non-zero for this formulation."
        )
    numerator = (B / 2.0 + H1 * math.tan(phi1)) * c1
    denominator = c2 * tan_phi2
    return numerator / denominator


def average_cohesion_mixed_layers(
    H1: float, c1: float, H2: float, c2: float
) -> float:
    """Average cohesion for mixed soil layers (Equation 5-27).

    Weighted average shear strength cohesion for the Satyanarayana and
    Garg (1980) method.

    .. math::
        c_{ave} = \\frac{H_1 c_1 + H_2 c_2}{H_1 + H_2}

    Parameters
    ----------
    H1 : float
        Thickness of top layer (ft, m).
    c1 : float
        Undrained cohesion of top layer (psf, kPa).
    H2 : float
        Thickness of second layer (ft, m).
    c2 : float
        Undrained cohesion of bottom layer (psf, kPa).

    Returns
    -------
    float
        Average cohesion c_ave (psf, kPa).

    Raises
    ------
    ValueError
        If *H1 + H2* is zero.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-27, p. 312.
    """
    total_H = H1 + H2
    if total_H <= 0.0:
        raise ValueError("Total layer thickness must be positive.")
    return (H1 * c1 + H2 * c2) / total_H


def average_friction_angle_mixed_layers(
    H1: float, phi1_deg: float, H2: float, phi2_deg: float
) -> float:
    """Average friction angle for mixed soil layers (Equation 5-28).

    Weighted average friction angle for the Satyanarayana and Garg (1980)
    method.  The averaging is performed on the tangent of the friction
    angles.

    .. math::
        \\tan\\phi_{ave} = \\frac{H_1 \\tan\\phi_1 + H_2 \\tan\\phi_2}
                          {H_1 + H_2}

    Parameters
    ----------
    H1 : float
        Thickness of top layer (ft, m).
    phi1_deg : float
        Undrained friction angle of top layer (degrees).
    H2 : float
        Thickness of second layer (ft, m).
    phi2_deg : float
        Undrained friction angle of bottom layer (degrees).

    Returns
    -------
    float
        Average friction angle phi_ave (degrees).

    Raises
    ------
    ValueError
        If *H1 + H2* is zero.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-28, p. 312.
    """
    total_H = H1 + H2
    if total_H <= 0.0:
        raise ValueError("Total layer thickness must be positive.")
    phi1 = math.radians(phi1_deg)
    phi2 = math.radians(phi2_deg)
    tan_ave = (H1 * math.tan(phi1) + H2 * math.tan(phi2)) / total_H
    return math.degrees(math.atan(tan_ave))


# ===========================================================================
# SECTION 5-3.7: Bearing Capacity of Rock (Equations 5-29 through 5-32)
# ===========================================================================


def rock_bearing_capacity_factor_Nc(phi_rf_deg: float) -> float:
    """Bearing capacity factor Nc for rock (Equation 5-29).

    Rock bearing capacity factor per Stagg and Zienkiewicz (1968) based
    on the rock fracture friction angle.

    .. math::
        N_c = 5 \\tan^4\\!\\left(45^\\circ + \\frac{\\phi'_{rf}}{2}\\right)

    Parameters
    ----------
    phi_rf_deg : float
        Rock fracture friction angle including the effect of asperities
        (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nc for rock (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-29, p. 317.
    """
    angle = math.radians(45.0 + phi_rf_deg / 2.0)
    return 5.0 * math.tan(angle) ** 4


def rock_bearing_capacity_factor_Nq(phi_rf_deg: float) -> float:
    """Bearing capacity factor Nq for rock (Equation 5-30).

    Rock bearing capacity factor per Stagg and Zienkiewicz (1968).

    .. math::
        N_q = \\tan^6\\!\\left(45^\\circ + \\frac{\\phi'_{rf}}{2}\\right)

    Parameters
    ----------
    phi_rf_deg : float
        Rock fracture friction angle including the effect of asperities
        (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nq for rock (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-30, p. 317.
    """
    angle = math.radians(45.0 + phi_rf_deg / 2.0)
    return math.tan(angle) ** 6


def rock_bearing_capacity_factor_Ngamma(Nq: float) -> float:
    """Bearing capacity factor N_gamma for rock (Equation 5-31).

    Rock bearing capacity factor per Stagg and Zienkiewicz (1968).

    .. math::
        N_\\gamma = N_q + 1

    Parameters
    ----------
    Nq : float
        Bearing capacity factor Nq for rock (dimensionless).

    Returns
    -------
    float
        Bearing capacity factor N_gamma for rock (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-31, p. 317.
    """
    return Nq + 1.0


def rock_reduced_ultimate_bearing_capacity(
    q_ult: float, RQD: float
) -> float:
    """Reduced ultimate bearing capacity for rock using RQD (Equation 5-32).

    Incorporates Rock Quality Designation as a reduction to the ultimate
    bearing capacity (Bowles 1996).

    .. math::
        q'_{ult} = q_{ult} \\cdot RQD^2

    Parameters
    ----------
    q_ult : float
        Ultimate bearing capacity of intact rock (psf, kPa).
    RQD : float
        Rock Quality Designation expressed as a decimal (0 to 1),
        evaluated to a depth of B below the footing.

    Returns
    -------
    float
        Reduced ultimate bearing capacity (psf, kPa).

    Raises
    ------
    ValueError
        If *RQD* is outside the range [0, 1].

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-32, p. 317.
    """
    if RQD < 0.0 or RQD > 1.0:
        raise ValueError("RQD must be between 0 and 1 (decimal, not percent).")
    return q_ult * RQD ** 2


# ===========================================================================
# SECTION 5-4.2.1: Rigid Foundation Criteria (Equations 5-33 and 5-34)
# ===========================================================================


def relative_stiffness_factor(
    E_prime_Ib: float, Es: float, B: float
) -> float:
    """Relative stiffness factor Kr for rigid foundation check (Equation 5-33).

    Determines whether a combined footing or mat foundation may be
    designed as a rigid structure (Meyerhof 1953).  When Kr >= 0.5, the
    foundation may be considered rigid.

    .. math::
        K_r = \\frac{E' I_b}{E_s \\cdot B^3}

    Parameters
    ----------
    E_prime_Ib : float
        Flexural stiffness of the structure, E' * Ib (force-length^2,
        e.g., lb-ft^2 or kN-m^2).
    Es : float
        Soil modulus (psf, kPa, or consistent stress unit).
    B : float
        Width of foundation (ft, m).

    Returns
    -------
    float
        Relative stiffness factor Kr (dimensionless).

    Raises
    ------
    ValueError
        If *Es* or *B* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-33, p. 320.
    """
    if Es <= 0.0:
        raise ValueError("Soil modulus Es must be positive.")
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    return E_prime_Ib / (Es * B ** 3)


def foundation_stiffness_factor(
    ks: float, B: float, Ec: float, I: float
) -> float:
    """Foundation stiffness factor lambda for rigidity assessment (Equation 5-34).

    Used to determine if continuous foundations may be considered rigid.
    A foundation is rigid if the average spacing of two adjacent column
    spans is less than 1.75 / lambda.

    .. math::
        \\lambda = \\sqrt[4]{\\frac{k_s}{4 E_c I / B}}

    which can be rewritten as:

    .. math::
        \\lambda = \\left(\\frac{k_s}{4 E_c I / B}\\right)^{1/4}

    Parameters
    ----------
    ks : float
        Modulus of subgrade reaction (pci, kN/m^3, or consistent unit,
        force per cubic length).
    B : float
        Width of footing (ft, m, or consistent length unit).
    Ec : float
        Modulus of concrete (psi, kPa, or consistent stress unit).
    I : float
        Moment of inertia of the footing cross-section (in^4, m^4, or
        consistent unit).

    Returns
    -------
    float
        Stiffness factor lambda (1/length).

    Raises
    ------
    ValueError
        If any input is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-34, p. 321.
    """
    if ks <= 0.0:
        raise ValueError("Subgrade modulus ks must be positive.")
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    if Ec <= 0.0:
        raise ValueError("Concrete modulus Ec must be positive.")
    if I <= 0.0:
        raise ValueError("Moment of inertia I must be positive.")
    return (ks / (4.0 * Ec * I / B)) ** 0.25


# ===========================================================================
# SECTION 5-4.5: Modulus of Subgrade Reaction (Equations 5-35 through 5-37)
# ===========================================================================


def modulus_of_subgrade_reaction(q: float, s: float) -> float:
    """Modulus of subgrade reaction ks (Equation 5-35).

    The ratio of the contact pressure to the corresponding settlement.

    .. math::
        k_s = \\frac{q}{s}

    Parameters
    ----------
    q : float
        Contact pressure acting perpendicular to the contact area
        (psf, kPa).
    s : float
        Soil settlement (ft, m, or consistent length unit).

    Returns
    -------
    float
        Modulus of subgrade reaction ks (force/length^3).

    Raises
    ------
    ValueError
        If *s* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-35, p. 323.
    """
    if s <= 0.0:
        raise ValueError("Settlement s must be positive.")
    return q / s


def subgrade_modulus_from_plate_load_test(
    kp: float, Bp: float, B: float, n: float = 0.5
) -> float:
    """Subgrade modulus scaled from plate load test (Equation 5-36).

    Approximates the subgrade modulus for a footing from a plate load
    test value, valid for footings with width <= 5 ft with uniform soil
    conditions within the depth of influence (Sowers 1977).

    .. math::
        k_s = k_p \\left(\\frac{B_p}{B}\\right)^n

    Parameters
    ----------
    kp : float
        Modulus of subgrade reaction from the plate load test
        (force/length^3).
    Bp : float
        Width of the test plate (ft, m).
    B : float
        Width of the foundation (ft, m).
    n : float, optional
        Exponent, typically 0.5 to 0.7 (default 0.5).

    Returns
    -------
    float
        Estimated modulus of subgrade reaction ks (force/length^3).

    Raises
    ------
    ValueError
        If *Bp* or *B* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-36, p. 324.
    """
    if Bp <= 0.0 or B <= 0.0:
        raise ValueError("Plate width Bp and foundation width B must be positive.")
    if not (0.0 < n <= 1.0):
        raise ValueError("Exponent n should be between 0 and 1.")
    return kp * (Bp / B) ** n


def subgrade_modulus_from_elastic_parameters(
    B: float, Es: float, mu0: float, mu1: float
) -> float:
    """Subgrade modulus estimated from elastic parameters (Equation 5-37).

    Using the definition of ks and the elastic settlement equation, the
    subgrade modulus can be estimated from the foundation size and soil
    elastic properties.

    .. math::
        k_s = \\frac{1}{B \\, \\mu_0 \\, \\mu_1} \\cdot E_s

    Parameters
    ----------
    B : float
        Width of the foundation (ft, m).
    Es : float
        Elastic modulus of the soil within the zone of influence
        (psf, kPa).
    mu0 : float
        Influence factor related to embedment of the load and Poisson's
        ratio (dimensionless).
    mu1 : float
        Influence factor related to problem geometry and Poisson's ratio
        (dimensionless).

    Returns
    -------
    float
        Estimated modulus of subgrade reaction ks (force/length^3).

    Raises
    ------
    ValueError
        If any input is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-37, p. 325.
    """
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    if Es <= 0.0:
        raise ValueError("Soil elastic modulus Es must be positive.")
    if mu0 <= 0.0 or mu1 <= 0.0:
        raise ValueError("Influence factors mu0 and mu1 must be positive.")
    return Es / (B * mu0 * mu1)


# ===========================================================================
# SECTION 5-4.5.3: Poisson's Ratio from Friction Angle (Equation 5-38)
# ===========================================================================


def poissons_ratio_from_friction_angle(phi_prime_deg: float) -> float:
    """Drained Poisson's ratio estimated from friction angle (Equation 5-38).

    For drained conditions, the Poisson's ratio can be related to the
    effective friction angle.  Valid for phi' approximately 20 to 55
    degrees (nu ranges from 0.4 to 0.15).

    .. math::
        \\nu = \\frac{1 - \\sin\\phi'}{2 - \\sin\\phi'}

    Parameters
    ----------
    phi_prime_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    float
        Drained Poisson's ratio (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is non-positive or >= 90 degrees.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-38, p. 329.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError(
            "Friction angle must be between 0 and 90 degrees (exclusive)."
        )
    sin_phi = math.sin(math.radians(phi_prime_deg))
    return (1.0 - sin_phi) / (2.0 - sin_phi)


# ===========================================================================
# SECTION 5-4.5.4: Time-Dependent Subgrade Modulus (Equation 5-39)
# ===========================================================================


def subgrade_modulus_time_dependent(
    s: float, sc: float, ks: float
) -> float:
    """Reduced subgrade modulus for time-dependent settlement (Equation 5-39).

    When the structure imposes stresses beyond the preconsolidation stress,
    or when recompression/heave occurs due to excavation, consolidation
    settlement must be added to the elastic settlement.

    .. math::
        k_{sc} = \\frac{s \\cdot k_s}{s + s_c}

    where s is the elastic settlement and s_c is the consolidation
    settlement.

    Parameters
    ----------
    s : float
        Elastic settlement (ft, m, or consistent length unit).
    sc : float
        Consolidation settlement (ft, m, or consistent length unit).
    ks : float
        Modulus of subgrade reaction from elastic analysis
        (force/length^3).

    Returns
    -------
    float
        Reduced subgrade modulus ksc (force/length^3).

    Raises
    ------
    ValueError
        If total settlement (s + sc) is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-39, p. 330.
    """
    total = s + sc
    if total <= 0.0:
        raise ValueError("Total settlement (s + sc) must be positive.")
    return s * ks / total


# ===========================================================================
# SECTION 5-4.6: Node Coupling of Soil Effects (Equation 5-40)
# ===========================================================================


def winkler_spring_stiffness(ks: float, A_cont: float) -> float:
    """Winkler spring stiffness at a node (Equation 5-40).

    At the interface between a mat and the soil, the soil response is
    concentrated at nodes.  The spring stiffness K for each node is the
    subgrade modulus multiplied by the contributing area.

    .. math::
        K = k_s \\cdot A_{cont}

    Parameters
    ----------
    ks : float
        Modulus of subgrade reaction (force/length^3).
    A_cont : float
        Contributing area to the node (ft^2, m^2).

    Returns
    -------
    float
        Winkler spring stiffness K (force/length).

    Raises
    ------
    ValueError
        If *ks* or *A_cont* is negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-40, p. 330.
    """
    if ks < 0.0:
        raise ValueError("Subgrade modulus ks must be non-negative.")
    if A_cont < 0.0:
        raise ValueError("Contributing area A_cont must be non-negative.")
    return ks * A_cont


# ===========================================================================
# SECTION 5-4.7: Indirect Method for Coupling (Equation 5-41)
# ===========================================================================


def subgrade_modulus_coupling(
    ks_edge: float,
    sigma_z_ave_edge: float,
    sigma_z_ave_i: float,
) -> float:
    """Subgrade modulus at interior point using coupling method (Equation 5-41).

    Bowles (1996) indirect method for considering coupling in a mat
    foundation.  The ks at an interior point is scaled from the edge
    ks value using the ratio of average vertical stresses.

    .. math::
        k_{s,i} = k_s \\left(\\frac{\\sigma_{z,ave}(1)}
                  {\\sigma_{z,ave}(i)}\\right)

    Parameters
    ----------
    ks_edge : float
        Modulus of subgrade reaction at the edge point (Point 1)
        (force/length^3).
    sigma_z_ave_edge : float
        Average vertical stress at the edge point (psf, kPa).
    sigma_z_ave_i : float
        Average vertical stress at interior point i (psf, kPa).

    Returns
    -------
    float
        Modulus of subgrade reaction at interior point i (force/length^3).

    Raises
    ------
    ValueError
        If *sigma_z_ave_i* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-41, p. 332.
    """
    if sigma_z_ave_i <= 0.0:
        raise ValueError(
            "Average vertical stress at interior point must be positive."
        )
    return ks_edge * (sigma_z_ave_edge / sigma_z_ave_i)


# ===========================================================================
# SECTION 5-4.8: Floating Mat Foundation (Equation 5-42)
# ===========================================================================


def floating_mat_net_pressure(
    W_structure: float, W_excavated: float, A_mat: float
) -> float:
    """Net pressure for a floating mat foundation (Equation 5-42).

    The net pressure that results in settlement for a floating mat
    foundation.  If the structure weight equals the excavated soil weight,
    the net pressure is zero and settlement relates only to recompression
    or heave.

    .. math::
        q_{net} = \\frac{W_{structure} - W_{excavated}}{A_{mat}}

    Parameters
    ----------
    W_structure : float
        Total weight of the structure (lb, kN, or consistent force unit).
    W_excavated : float
        Total weight of excavated soil (same force unit).
    A_mat : float
        Area of the mat foundation (ft^2, m^2).

    Returns
    -------
    float
        Net bearing pressure for settlement analysis (psf, kPa).

    Raises
    ------
    ValueError
        If *A_mat* is zero or negative.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Equation 5-42, p. 334.
    """
    if A_mat <= 0.0:
        raise ValueError("Mat area A_mat must be positive.")
    return (W_structure - W_excavated) / A_mat


# ===========================================================================
# BEARING CAPACITY FACTOR HELPER FUNCTIONS
# (Supporting equations referenced in Tables 5-2, 5-5, 5-6, 5-8
#  and Figure 5-5 -- Terzaghi, Meyerhof, and Brinch Hansen methods)
# ===========================================================================


def _Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient.

    .. math::
        K_P = \\tan^2(\\pi/4 + \\phi'/2)

    Used internally by Meyerhof bearing capacity and correction factors.
    """
    return math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2


# ---------------------------------------------------------------------------
# Meyerhof / Brinch Hansen bearing capacity factors
# ---------------------------------------------------------------------------


def meyerhof_hansen_Nq(phi_deg: float) -> float:
    """Bearing capacity factor Nq -- Meyerhof and Brinch Hansen (Table 5-2).

    .. math::
        N_q = \\tan^2\\!\\left(\\frac{\\pi}{4} + \\frac{\\phi'}{2}\\right)
              e^{\\pi \\tan\\phi'}

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nq (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-2 / Table 5-6 /
    Table 5-8, pp. 289-296.
    """
    phi = math.radians(phi_deg)
    return math.tan(math.pi / 4.0 + phi / 2.0) ** 2 * math.exp(
        math.pi * math.tan(phi)
    )


def meyerhof_hansen_Nc(phi_deg: float) -> float:
    """Bearing capacity factor Nc -- Meyerhof and Brinch Hansen (Table 5-2).

    For phi > 0:

    .. math::
        N_c = (N_q - 1) \\cot\\phi'

    For phi = 0, Nc = 5.14.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nc (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-2 / Table 5-6 /
    Table 5-8, pp. 289-296.
    """
    if phi_deg == 0.0:
        return 5.14
    Nq = meyerhof_hansen_Nq(phi_deg)
    phi = math.radians(phi_deg)
    return (Nq - 1.0) / math.tan(phi)


def meyerhof_Ngamma(phi_deg: float) -> float:
    """Bearing capacity factor N_gamma -- Meyerhof (Table 5-2 / Table 5-6).

    .. math::
        N_\\gamma = (N_q - 1) \\tan(1.4\\phi')

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor N_gamma (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-6, p. 294.
    """
    Nq = meyerhof_hansen_Nq(phi_deg)
    return (Nq - 1.0) * math.tan(math.radians(1.4 * phi_deg))


def hansen_Ngamma(phi_deg: float) -> float:
    """Bearing capacity factor N_gamma -- Brinch Hansen (Table 5-2 / Table 5-8).

    .. math::
        N_\\gamma = 1.5 (N_q - 1) \\tan\\phi'

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor N_gamma (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    Nq = meyerhof_hansen_Nq(phi_deg)
    return 1.5 * (Nq - 1.0) * math.tan(math.radians(phi_deg))


# ---------------------------------------------------------------------------
# Terzaghi bearing capacity factors (Table 5-2, approximations per
# Coduto et al. 2016)
# ---------------------------------------------------------------------------


def terzaghi_Nc(phi_deg: float) -> float:
    """Bearing capacity factor Nc -- Terzaghi (Table 5-2).

    For phi = 0, Nc = 5.7. For phi > 0, uses the Coduto et al. (2016)
    approximation tabulated in the UFC.

    .. math::
        N_c = \\cot\\phi' \\left[\\frac{e^{2(3\\pi/4 - \\phi'/2)\\tan\\phi'}}
              {2\\cos^2(45+\\phi'/2)} - 1\\right]

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees), 0 to 50.

    Returns
    -------
    float
        Bearing capacity factor Nc (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-2, p. 289.
    """
    if phi_deg == 0.0:
        return 5.7
    phi = math.radians(phi_deg)
    # Terzaghi's Nq
    Nq = terzaghi_Nq(phi_deg)
    return (Nq - 1.0) / math.tan(phi)


def terzaghi_Nq(phi_deg: float) -> float:
    """Bearing capacity factor Nq -- Terzaghi (Table 5-2).

    Uses the exact Terzaghi formulation:

    .. math::
        N_q = \\frac{e^{2(3\\pi/4 - \\phi'/2)\\tan\\phi'}}
              {2\\cos^2(45^\\circ + \\phi'/2)}

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nq (dimensionless).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-2, p. 289.
    """
    if phi_deg == 0.0:
        return 1.0
    phi = math.radians(phi_deg)
    exponent = 2.0 * (3.0 * math.pi / 4.0 - phi / 2.0) * math.tan(phi)
    cos_term = math.cos(math.radians(45.0 + phi_deg / 2.0))
    return math.exp(exponent) / (2.0 * cos_term ** 2)


def terzaghi_Ngamma(phi_deg: float) -> float:
    """Bearing capacity factor N_gamma -- Terzaghi (Table 5-2).

    Uses the approximation by Coduto et al. (2016) that matches the
    tabulated values in the UFC.

    .. math::
        N_\\gamma \\approx 2(N_q + 1)\\tan\\phi' \\cdot \\frac{1}{1 + 0.4\\sin(4\\phi')}

    This is an approximation; for phi = 0, N_gamma = 0.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor N_gamma (dimensionless).

    Notes
    -----
    The approximation gives values consistent with Table 5-2 per the
    UFC note that Coduto et al. (2016) approximation is used.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-2, p. 289.
    """
    if phi_deg == 0.0:
        return 0.0
    phi = math.radians(phi_deg)
    Nq = terzaghi_Nq(phi_deg)
    return 2.0 * (Nq + 1.0) * math.tan(phi) / (1.0 + 0.4 * math.sin(4.0 * phi))


# ===========================================================================
# TERZAGHI SHAPE FACTORS (Table 5-5)
# ===========================================================================


def terzaghi_shape_factors(
    shape: str,
) -> Tuple[float, float, float]:
    """Terzaghi shape factors (Table 5-5).

    Returns the shape factors (sc, sq, s_gamma) for the Terzaghi method.

    Parameters
    ----------
    shape : str
        Foundation shape: 'continuous' (strip), 'square', or 'circular'.

    Returns
    -------
    tuple of float
        (sc, sq, s_gamma) shape factors.

    Raises
    ------
    ValueError
        If *shape* is not recognized.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-5, p. 294.
    """
    shape_lower = shape.lower().strip()
    if shape_lower in ("continuous", "strip"):
        return (1.0, 1.0, 1.0)
    elif shape_lower == "square":
        return (1.3, 1.0, 0.8)
    elif shape_lower in ("circular", "circle"):
        return (1.3, 1.0, 0.6)
    else:
        raise ValueError(
            f"Unknown shape '{shape}'. Use 'continuous', 'square', or 'circular'."
        )


# ===========================================================================
# MEYERHOF CORRECTION FACTORS (Table 5-6)
# ===========================================================================


def meyerhof_shape_factors(
    B: float, L: float, phi_deg: float
) -> Tuple[float, float, float]:
    """Meyerhof (1963) shape factors (Table 5-6).

    Parameters
    ----------
    B : float
        Foundation width (ft, m).  Use B/L = 1 for circle.
    L : float
        Foundation length (ft, m).
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    tuple of float
        (sc, sq, s_gamma) shape factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-6, p. 294.
    """
    Kp = _Kp(phi_deg)
    ratio = B / L if L > 0.0 else 0.0
    if phi_deg == 0.0:
        sc = 1.0 + 0.2 * ratio
        sq = 1.0
        s_gamma = 1.0
    else:
        sc = 1.0 + 0.2 * ratio * Kp
        sq = 1.0 + 0.1 * ratio * Kp
        s_gamma = 1.0 + 0.1 * ratio * Kp
    return (sc, sq, s_gamma)


def meyerhof_depth_factors(
    Df: float, B: float, phi_deg: float
) -> Tuple[float, float, float]:
    """Meyerhof (1963) depth factors (Table 5-6).

    Parameters
    ----------
    Df : float
        Foundation depth (ft, m).
    B : float
        Foundation width (ft, m).
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    tuple of float
        (dc, dq, d_gamma) depth factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-6, p. 294.
    """
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    k = Df / B
    Kp = _Kp(phi_deg)
    if phi_deg == 0.0:
        dc = 1.0 + 0.2 * k * math.sqrt(Kp)
        dq = 1.0
        d_gamma = 1.0
    else:
        dc = 1.0 + 0.2 * k * math.sqrt(Kp)
        dq = 1.0 + 0.1 * k * math.sqrt(Kp)
        d_gamma = 1.0 + 0.1 * k * math.sqrt(Kp)
    return (dc, dq, d_gamma)


def meyerhof_inclination_factors(
    theta_deg: float, phi_deg: float
) -> Tuple[float, float, float]:
    """Meyerhof (1963) load inclination factors (Table 5-6).

    Parameters
    ----------
    theta_deg : float
        Angle of load inclination from vertical (degrees).
        theta = atan(H/V) where H = horizontal load, V = vertical load.
        Must be less than phi' for i_gamma.
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    tuple of float
        (ic, iq, i_gamma) inclination factors.

    Raises
    ------
    ValueError
        If *theta_deg* >= *phi_deg* for frictional soils.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-6, p. 294.
    """
    theta = math.radians(theta_deg)
    ic = (1.0 - 2.0 * theta / math.pi) ** 2
    iq = (1.0 - 2.0 * theta / math.pi) ** 2
    if phi_deg > 0.0:
        if theta_deg >= phi_deg:
            raise ValueError(
                "Load inclination angle theta must be less than phi'."
            )
        i_gamma = (1.0 - theta / math.radians(phi_deg)) ** 2
    else:
        i_gamma = 1.0
    return (ic, iq, i_gamma)


# ===========================================================================
# BRINCH HANSEN CORRECTION FACTORS (Tables 5-7 and 5-8)
# ===========================================================================


def _hansen_k(Df: float, B: float) -> float:
    """Depth parameter k for Brinch Hansen factors (Table 5-7/5-8 note).

    For Df/B < 1: k = Df/B.
    For Df/B >= 1: k = atan(Df/B) in radians.
    """
    if B <= 0.0:
        raise ValueError("Foundation width B must be positive.")
    ratio = Df / B
    if ratio < 1.0:
        return ratio
    return math.atan(ratio)


def hansen_shape_factors_phi0(
    B: float, L: float
) -> Tuple[float, float]:
    """Brinch Hansen shape factors for phi = 0, vertical load (Table 5-7).

    Parameters
    ----------
    B : float
        Foundation width (ft, m).
    L : float
        Foundation length (ft, m).

    Returns
    -------
    tuple of float
        (sc, sq) where sq = 0 (no correction for q factor in phi=0).
        sc = 0.2 * (B/L).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-7, p. 295.
    """
    if L <= 0.0:
        raise ValueError("Foundation length L must be positive.")
    sc = 0.2 * (B / L)
    return (sc, 0.0)


def hansen_depth_factor_phi0(Df: float, B: float) -> float:
    """Brinch Hansen depth factor for phi = 0 (Table 5-7).

    .. math::
        d_c = 0.4 k

    Parameters
    ----------
    Df : float
        Foundation depth (ft, m).
    B : float
        Foundation width (ft, m).

    Returns
    -------
    float
        Depth factor dc for phi = 0.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-7, p. 295.
    """
    k = _hansen_k(Df, B)
    return 0.4 * k


def hansen_inclination_factor_phi0(
    H: float, A_prime: float, Ca: float
) -> float:
    """Brinch Hansen inclination factor for phi = 0 (Table 5-7).

    .. math::
        i_c = 0.5 - 0.5\\sqrt{1 - \\frac{H}{A' \\cdot C_a}}

    Parameters
    ----------
    H : float
        Horizontal component of the load (lb, kN).
    A_prime : float
        Equivalent bearing area (ft^2, m^2).
    Ca : float
        Base adhesion (psf, kPa).

    Returns
    -------
    float
        Inclination factor ic for phi = 0.

    Raises
    ------
    ValueError
        If H / (A' * Ca) > 1.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-7, p. 295.
    """
    if A_prime <= 0.0 or Ca <= 0.0:
        raise ValueError("A_prime and Ca must be positive.")
    ratio = H / (A_prime * Ca)
    if ratio > 1.0:
        raise ValueError("H / (A' * Ca) must not exceed 1.0.")
    return 0.5 - 0.5 * math.sqrt(1.0 - ratio)


def hansen_base_factor_phi0(eta_rad: float) -> float:
    """Brinch Hansen sloping base factor for phi = 0 (Table 5-7).

    .. math::
        b_c = \\frac{\\eta}{\\pi/2 + 1}

    Parameters
    ----------
    eta_rad : float
        Base inclination angle (radians).

    Returns
    -------
    float
        Base inclination factor bc for phi = 0.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-7, p. 295.
    """
    return eta_rad / (math.pi / 2.0 + 1.0)


def hansen_ground_factor_phi0(beta_rad: float) -> float:
    """Brinch Hansen sloping ground factor for phi = 0 (Table 5-7).

    .. math::
        g_c = \\frac{\\beta}{\\pi/2 + 1}

    Parameters
    ----------
    beta_rad : float
        Ground inclination angle (radians).

    Returns
    -------
    float
        Ground inclination factor gc for phi = 0.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-7, p. 295.
    """
    return beta_rad / (math.pi / 2.0 + 1.0)


# ---------------------------------------------------------------------------
# Brinch Hansen factors for phi > 0 (Table 5-8)
# ---------------------------------------------------------------------------


def hansen_shape_factors(
    B: float, L: float, phi_deg: float
) -> Tuple[float, float, float]:
    """Brinch Hansen shape factors for phi > 0, vertical load (Table 5-8).

    Parameters
    ----------
    B : float
        Foundation width (ft, m).  Use B/L = 1 for circular.
    L : float
        Foundation length (ft, m).
    phi_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    tuple of float
        (sc, sq, s_gamma) shape factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    if L <= 0.0:
        raise ValueError("Foundation length L must be positive.")
    phi = math.radians(phi_deg)
    Nq = meyerhof_hansen_Nq(phi_deg)
    Nc = meyerhof_hansen_Nc(phi_deg)
    ratio = B / L
    sc = 1.0 + ratio * (Nq / Nc) * math.cos(phi) if Nc != 0.0 else 1.0
    sq = 1.0 + math.sin(phi) * ratio
    s_gamma = 1.0 - 0.4 * ratio
    return (sc, sq, s_gamma)


def hansen_depth_factors(
    Df: float, B: float, phi_deg: float
) -> Tuple[float, float, float]:
    """Brinch Hansen depth factors for phi > 0 (Table 5-8).

    Parameters
    ----------
    Df : float
        Foundation depth (ft, m).
    B : float
        Foundation width (ft, m).
    phi_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    tuple of float
        (dc, dq, d_gamma) depth factors.  d_gamma is always 1.0.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    k = _hansen_k(Df, B)
    phi = math.radians(phi_deg)
    sin_phi = math.sin(phi)
    tan_phi = math.tan(phi)
    Nq = meyerhof_hansen_Nq(phi_deg)
    Nc = meyerhof_hansen_Nc(phi_deg)
    dq = 1.0 + 2.0 * k * tan_phi * (1.0 - sin_phi) ** 2
    dc = 1.0 + k * (1.0 - sin_phi) ** 2 * (Nq / Nc) if Nc != 0.0 else 1.0
    d_gamma = 1.0
    return (dc, dq, d_gamma)


def hansen_inclination_factors(
    H: float,
    V: float,
    A_prime: float,
    Ca: float,
    phi_deg: float,
    eta_rad: float = 0.0,
) -> Tuple[float, float, float]:
    """Brinch Hansen load inclination factors for phi > 0 (Table 5-8).

    .. math::
        i_q = \\left(1 - \\frac{0.5 H}{V + A' C_a \\cot\\phi'}\\right)^5

    .. math::
        i_\\gamma = \\left(1 - \\frac{(0.7 - \\eta/450^\\circ) H}
                   {V + A' C_a \\cot\\phi'}\\right)^5

    .. math::
        i_c = \\frac{i_q N_q - 1}{N_q - 1}

    Parameters
    ----------
    H : float
        Horizontal component of load (lb, kN).
    V : float
        Vertical component of load (lb, kN).
    A_prime : float
        Equivalent bearing area (ft^2, m^2).
    Ca : float
        Base adhesion (psf, kPa).  Use 0 if no adhesion.
    phi_deg : float
        Effective friction angle (degrees).
    eta_rad : float, optional
        Base inclination angle (radians, default 0.0).

    Returns
    -------
    tuple of float
        (ic, iq, i_gamma) inclination factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    if phi_deg <= 0.0:
        raise ValueError("Use hansen_inclination_factor_phi0 for phi = 0.")
    phi = math.radians(phi_deg)
    cot_phi = 1.0 / math.tan(phi)
    denom = V + A_prime * Ca * cot_phi
    if denom <= 0.0:
        raise ValueError(
            "Denominator (V + A' Ca cot(phi)) must be positive."
        )
    iq = (1.0 - 0.5 * H / denom) ** 5
    # eta factor: (0.7 - eta/450deg) where 450deg = 2.5*pi radians
    eta_factor = 0.7 - eta_rad / (2.5 * math.pi)
    i_gamma = (1.0 - eta_factor * H / denom) ** 5
    Nq = meyerhof_hansen_Nq(phi_deg)
    ic = (iq * Nq - 1.0) / (Nq - 1.0) if Nq != 1.0 else 1.0
    return (ic, iq, i_gamma)


def hansen_base_factors(
    eta_rad: float, phi_deg: float
) -> Tuple[float, float]:
    """Brinch Hansen sloping base factors for phi > 0 (Table 5-8).

    .. math::
        b_q = e^{-2 \\eta \\tan\\phi'}

    .. math::
        b_\\gamma = e^{-2.7 \\eta \\tan\\phi'}

    Note: c factors for inclined base are not provided by Brinch Hansen
    for phi > 0.

    Parameters
    ----------
    eta_rad : float
        Base inclination angle (radians).
    phi_deg : float
        Effective friction angle (degrees).

    Returns
    -------
    tuple of float
        (bq, b_gamma) base inclination factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    tan_phi = math.tan(math.radians(phi_deg))
    bq = math.exp(-2.0 * eta_rad * tan_phi)
    b_gamma = math.exp(-2.7 * eta_rad * tan_phi)
    return (bq, b_gamma)


def hansen_ground_factors(beta_deg: float) -> Tuple[float, float]:
    """Brinch Hansen sloping ground factors for phi > 0 (Table 5-8).

    .. math::
        g_q = (1 - 0.5 \\tan\\beta)^5

    .. math::
        g_\\gamma = (1 - 0.5 \\tan\\beta)^5

    Note: c factors for sloping ground are not provided by Brinch Hansen
    for phi > 0.

    Parameters
    ----------
    beta_deg : float
        Ground inclination angle (degrees).

    Returns
    -------
    tuple of float
        (gq, g_gamma) ground inclination factors.

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-8, p. 296.
    """
    tan_beta = math.tan(math.radians(beta_deg))
    gq = (1.0 - 0.5 * tan_beta) ** 5
    g_gamma = (1.0 - 0.5 * tan_beta) ** 5
    return (gq, g_gamma)


# ===========================================================================
# LOCAL SHEAR CORRECTIONS (Table 5-4)
# ===========================================================================


def local_shear_terzaghi(
    c_prime: float, phi_prime_deg: float
) -> Tuple[float, float]:
    """Local shear reduced strength parameters -- Terzaghi (Table 5-4).

    For local or punching shear failure modes, the shear strength
    parameters are reduced.

    .. math::
        c^* = 0.67 \\, c'

    .. math::
        \\phi^* = \\tan^{-1}(0.67 \\tan\\phi')

    Parameters
    ----------
    c_prime : float
        Effective stress cohesion (psf, kPa).
    phi_prime_deg : float
        Effective stress friction angle (degrees).

    Returns
    -------
    tuple of float
        (c_star, phi_star_deg) reduced cohesion and friction angle (degrees).

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-4, p. 291.
    """
    c_star = 0.67 * c_prime
    phi_star = math.degrees(
        math.atan(0.67 * math.tan(math.radians(phi_prime_deg)))
    )
    return (c_star, phi_star)


def local_shear_vesic(
    c_prime: float, phi_prime_deg: float, Dr: float
) -> Tuple[float, float]:
    """Local shear reduced strength parameters -- Vesic (Table 5-4).

    Variable reduction based on relative density (Dr).

    For Dr < 0.67: R = 0.67 + Dr - 0.75 * Dr^2
    For Dr >= 0.67: R = 1

    .. math::
        c^* = R \\cdot c'

    .. math::
        \\phi^* = \\tan^{-1}(R \\cdot \\tan\\phi')

    Parameters
    ----------
    c_prime : float
        Effective stress cohesion (psf, kPa).
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    Dr : float
        Relative density (decimal, 0 to 1).

    Returns
    -------
    tuple of float
        (c_star, phi_star_deg) reduced cohesion and friction angle (degrees).

    Raises
    ------
    ValueError
        If *Dr* is outside [0, 1].

    References
    ----------
    UFC 3-220-20, Foundations, 16 Jan 2025, Chapter 5, Table 5-4, p. 291.
    """
    if Dr < 0.0 or Dr > 1.0:
        raise ValueError("Relative density Dr must be between 0 and 1.")
    if Dr < 0.67:
        R = 0.67 + Dr - 0.75 * Dr ** 2
    else:
        R = 1.0
    c_star = R * c_prime
    phi_star = math.degrees(
        math.atan(R * math.tan(math.radians(phi_prime_deg)))
    )
    return (c_star, phi_star)
