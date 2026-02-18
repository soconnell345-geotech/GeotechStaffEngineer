"""
UFC 3-220-20, Chapter 4: Analysis of Walls and Retaining Structures

Equations 4-1 through 4-31 covering at-rest earth pressure, Rankine active
and passive earth pressures, earth pressure distributions and loads, Coulomb
wedge method, log spiral method, equivalent fluid pressures, surface loads
behind retaining structures, seismic earth pressures (Mononobe-Okabe),
base pressure distributions for rigid gravity walls, allowable passive
resistance, and flexible retaining structure analysis.

Reference:
    UFC 3-220-20, Foundations and Earth Structures,
    16 January 2025
"""

import math


# ===========================================================================
# Equations 4-1 and 4-2: At-Rest Earth Pressure
# ===========================================================================

def at_rest_earth_pressure_coefficient(
    sigma_h_eff: float,
    sigma_z_eff: float,
) -> float:
    """At-rest earth pressure coefficient (Equation 4-1).

    Computes the at-rest earth pressure coefficient as the ratio of the
    horizontal effective stress to the vertical effective stress.  Valid
    when the soil mass is in a state of zero lateral strain (e.g., a rigid
    basement wall or a level ground surface with no surface loads of limited
    areal extent).

    .. math::
        K_0 = \\frac{\\sigma'_h}{\\sigma'_z}

    Parameters
    ----------
    sigma_h_eff : float
        Horizontal effective stress (psf or kPa).
    sigma_z_eff : float
        Vertical effective stress (psf or kPa).

    Returns
    -------
    float
        At-rest earth pressure coefficient, K0 (dimensionless).

    Raises
    ------
    ValueError
        If *sigma_z_eff* is zero or negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-1, p. 176.
    """
    if sigma_z_eff <= 0.0:
        raise ValueError("sigma_z_eff must be positive.")
    return sigma_h_eff / sigma_z_eff


def at_rest_coefficient_mayne_kulhawy(
    phi_prime_deg: float,
    OCR: float = 1.0,
) -> float:
    """At-rest earth pressure coefficient from Mayne and Kulhawy (Equation 4-2).

    Computes K0 using the correlation by Mayne and Kulhawy (1982) which
    accounts for overconsolidation.  For normally consolidated soils
    (OCR = 1), this reduces to K0 = 1 - sin(phi').

    .. math::
        K_0 = (1 - \\sin \\phi') \\cdot OCR^{\\sin \\phi'}

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle for normally consolidated
        conditions (degrees).
    OCR : float, optional
        Overconsolidation ratio (dimensionless, >= 1.0).  Default is 1.0.

    Returns
    -------
    float
        At-rest earth pressure coefficient, K0 (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is not in (0, 90) or *OCR* is less than 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-2, p. 177.
    Mayne, P.W. and Kulhawy, F.H. (1982).
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    if OCR < 1.0:
        raise ValueError("OCR must be >= 1.0.")
    sin_phi = math.sin(math.radians(phi_prime_deg))
    return (1.0 - sin_phi) * OCR ** sin_phi


# ===========================================================================
# Equations 4-3 through 4-5: Rankine Active and Passive Earth Pressures
# ===========================================================================

def rankine_active_pressure_coefficient(phi_prime_deg: float) -> float:
    """Rankine active earth pressure coefficient (Equation 4-3).

    Computes the coefficient of active earth pressure from Rankine theory
    for a horizontal backfill with no wall friction.

    .. math::
        K_A = \\frac{1 - \\sin \\phi'}{1 + \\sin \\phi'}
            = \\tan^2\\left(45 - \\frac{\\phi'}{2}\\right)

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).

    Returns
    -------
    float
        Rankine active earth pressure coefficient, KA (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is not in (0, 90).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-3, p. 178.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    return math.tan(math.radians(45.0 - phi_prime_deg / 2.0)) ** 2


def rankine_passive_pressure_coefficient(phi_prime_deg: float) -> float:
    """Rankine passive earth pressure coefficient (Equation 4-4).

    Computes the coefficient of passive earth pressure from Rankine theory
    for a horizontal backfill with no wall friction.

    .. math::
        K_P = \\frac{1 + \\sin \\phi'}{1 - \\sin \\phi'}
            = \\tan^2\\left(45 + \\frac{\\phi'}{2}\\right)

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).

    Returns
    -------
    float
        Rankine passive earth pressure coefficient, KP (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is not in (0, 90).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-4, p. 178.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    return math.tan(math.radians(45.0 + phi_prime_deg / 2.0)) ** 2


def rankine_Ka_Kp_reciprocal(Ka: float) -> float:
    """Reciprocal relationship between KA and KP (Equation 4-5).

    Demonstrates that the Rankine passive earth pressure coefficient is
    the reciprocal of the active earth pressure coefficient.

    .. math::
        K_P = \\frac{1}{K_A}

    Parameters
    ----------
    Ka : float
        Active earth pressure coefficient (dimensionless).

    Returns
    -------
    float
        Passive earth pressure coefficient, KP (dimensionless).

    Raises
    ------
    ValueError
        If *Ka* is zero or negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-5, p. 178.
    """
    if Ka <= 0.0:
        raise ValueError("Ka must be positive.")
    return 1.0 / Ka


# ===========================================================================
# Equations 4-6 and 4-7: Rankine Horizontal Effective Stresses
# ===========================================================================

def rankine_active_horizontal_stress(
    Ka: float,
    sigma_z_eff: float,
    c_prime: float = 0.0,
) -> float:
    """Rankine active horizontal effective stress (Equation 4-6).

    Computes the horizontal effective stress for the active earth pressure
    condition using Rankine theory.  The cohesion term theoretically
    decreases the active earth pressure.  It is usually neglected for
    fine-grained soils due to creep, shrinkage, and swelling effects.

    .. math::
        \\sigma'_h = K_A \\cdot \\sigma'_z - 2 c' \\sqrt{K_A}

    Parameters
    ----------
    Ka : float
        Active earth pressure coefficient (dimensionless).
    sigma_z_eff : float
        Vertical effective stress at the depth of interest (psf or kPa).
    c_prime : float, optional
        Effective stress cohesion intercept (psf or kPa).  Default is 0.0.

    Returns
    -------
    float
        Horizontal effective stress for active condition (psf or kPa).

    Raises
    ------
    ValueError
        If *Ka* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-6, p. 178.
    """
    if Ka <= 0.0:
        raise ValueError("Ka must be positive.")
    return Ka * sigma_z_eff - 2.0 * c_prime * math.sqrt(Ka)


def rankine_passive_horizontal_stress(
    Kp: float,
    sigma_z_eff: float,
    c_prime: float = 0.0,
) -> float:
    """Rankine passive horizontal effective stress (Equation 4-7).

    Computes the horizontal effective stress for the passive earth pressure
    condition using Rankine theory.  The cohesion term increases the
    passive earth pressure.

    .. math::
        \\sigma'_h = K_P \\cdot \\sigma'_z + 2 c' \\sqrt{K_P}

    Parameters
    ----------
    Kp : float
        Passive earth pressure coefficient (dimensionless).
    sigma_z_eff : float
        Vertical effective stress at the depth of interest (psf or kPa).
    c_prime : float, optional
        Effective stress cohesion intercept (psf or kPa).  Default is 0.0.

    Returns
    -------
    float
        Horizontal effective stress for passive condition (psf or kPa).

    Raises
    ------
    ValueError
        If *Kp* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-7, p. 178.
    """
    if Kp <= 0.0:
        raise ValueError("Kp must be positive.")
    return Kp * sigma_z_eff + 2.0 * c_prime * math.sqrt(Kp)


# ===========================================================================
# Equations 4-8 and 4-9: Earth Pressure Resultant Forces
# ===========================================================================

def active_earth_pressure_resultant(
    Ka: float,
    gamma: float,
    H: float,
) -> float:
    """Active earth pressure resultant force (Equation 4-8).

    Computes the resultant force from the triangular active earth pressure
    distribution on a wall of height H with horizontal backfill, no
    cohesion, and no wall friction.  The resultant acts at H/3 above
    the base of the wall.

    .. math::
        P_A = \\frac{1}{2} K_A \\gamma H^2

    Parameters
    ----------
    Ka : float
        Active earth pressure coefficient (dimensionless).
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Height of the wall (ft or m).

    Returns
    -------
    float
        Active earth pressure resultant force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If any input is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-8, p. 181.
    """
    if Ka < 0.0:
        raise ValueError("Ka must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    return 0.5 * Ka * gamma * H ** 2


def passive_earth_pressure_resultant(
    Kp: float,
    gamma: float,
    H: float,
) -> float:
    """Passive earth pressure resultant force (Equation 4-9).

    Computes the resultant force from the triangular passive earth pressure
    distribution on a wall of height H with horizontal backfill, no
    cohesion, and no wall friction.  The resultant acts at H/3 above
    the base of the wall.

    .. math::
        P_P = \\frac{1}{2} K_P \\gamma H^2

    Parameters
    ----------
    Kp : float
        Passive earth pressure coefficient (dimensionless).
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Height of the wall (ft or m).

    Returns
    -------
    float
        Passive earth pressure resultant force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If any input is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-9, p. 181.
    """
    if Kp < 0.0:
        raise ValueError("Kp must be non-negative.")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    return 0.5 * Kp * gamma * H ** 2


# ===========================================================================
# Equations 4-10 and 4-11: Coulomb Wedge Method
# ===========================================================================

def coulomb_active_earth_pressure(
    gamma: float,
    H: float,
    phi_prime_deg: float,
    theta_deg: float = 0.0,
    delta_deg: float = 0.0,
    beta_deg: float = 0.0,
) -> float:
    """Coulomb active earth pressure force (Equation 4-10).

    Computes the active earth pressure force for a wall with a sloping
    face (theta), sloping backfill (beta), and interface friction angle
    (delta) using the Coulomb limit equilibrium method with a linear
    failure surface.

    .. math::
        P_A = \\frac{1}{2} \\gamma H^2
              \\frac{\\cos^2(\\phi' - \\theta)}
              {\\cos^2\\theta \\cdot \\cos(\\theta + \\delta)
              \\left[1 + \\sqrt{\\frac{\\sin(\\phi' + \\delta)\\sin(\\phi' - \\beta)}
              {\\cos(\\theta + \\delta)\\cos(\\theta - \\beta)}}\\right]^2}

    Parameters
    ----------
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Wall height measured along the back of the wall (ft or m).
    phi_prime_deg : float
        Effective stress (drained) friction angle (degrees).
    theta_deg : float, optional
        Slope angle of the wall face from vertical, positive when wall
        leans into the soil (degrees).  Default is 0.0 (vertical wall).
    delta_deg : float, optional
        Interface friction angle between wall and soil (degrees).
        Default is 0.0.
    beta_deg : float, optional
        Slope angle of the backfill surface from horizontal (degrees).
        Default is 0.0.

    Returns
    -------
    float
        Coulomb active earth pressure force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma* or *H* is negative, or if angle combinations are
        physically invalid (phi' - beta < 0 or denominator terms <= 0).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-10, p. 186.
    Coulomb, C.A. (1776).
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")

    phi = math.radians(phi_prime_deg)
    theta = math.radians(theta_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)

    numerator = math.cos(phi - theta) ** 2

    cos_theta = math.cos(theta)
    cos_theta_delta = math.cos(theta + delta)
    cos_theta_beta = math.cos(theta - beta)

    sin_phi_delta = math.sin(phi + delta)
    sin_phi_beta = math.sin(phi - beta)

    if sin_phi_beta < 0.0:
        raise ValueError(
            "phi_prime_deg must be >= beta_deg for active conditions."
        )
    if cos_theta_delta <= 0.0 or cos_theta_beta <= 0.0:
        raise ValueError("Invalid angle combination: denominator term <= 0.")

    sqrt_term = math.sqrt(
        (sin_phi_delta * sin_phi_beta) / (cos_theta_delta * cos_theta_beta)
    )
    denominator = (
        cos_theta ** 2 * cos_theta_delta * (1.0 + sqrt_term) ** 2
    )

    Ka_coulomb = numerator / denominator
    return 0.5 * gamma * H ** 2 * Ka_coulomb


def coulomb_active_coefficient(
    phi_prime_deg: float,
    theta_deg: float = 0.0,
    delta_deg: float = 0.0,
    beta_deg: float = 0.0,
) -> float:
    """Coulomb active earth pressure coefficient (from Equation 4-10).

    Extracts the active earth pressure coefficient KA from the Coulomb
    equation.  This is the coefficient that multiplies 0.5 * gamma * H^2
    in Equation 4-10.

    .. math::
        K_A = \\frac{\\cos^2(\\phi' - \\theta)}
              {\\cos^2\\theta \\cdot \\cos(\\theta + \\delta)
              \\left[1 + \\sqrt{\\frac{\\sin(\\phi' + \\delta)\\sin(\\phi' - \\beta)}
              {\\cos(\\theta + \\delta)\\cos(\\theta - \\beta)}}\\right]^2}

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    theta_deg : float, optional
        Slope angle of wall face from vertical (degrees).  Default is 0.0.
    delta_deg : float, optional
        Interface friction angle between wall and soil (degrees).
        Default is 0.0.
    beta_deg : float, optional
        Slope of backfill surface from horizontal (degrees).  Default is 0.0.

    Returns
    -------
    float
        Coulomb active earth pressure coefficient, KA (dimensionless).

    Raises
    ------
    ValueError
        If angle combinations are physically invalid.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-10, p. 186.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")

    phi = math.radians(phi_prime_deg)
    theta = math.radians(theta_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)

    numerator = math.cos(phi - theta) ** 2

    cos_theta = math.cos(theta)
    cos_theta_delta = math.cos(theta + delta)
    cos_theta_beta = math.cos(theta - beta)

    sin_phi_delta = math.sin(phi + delta)
    sin_phi_beta = math.sin(phi - beta)

    if sin_phi_beta < 0.0:
        raise ValueError(
            "phi_prime_deg must be >= beta_deg for active conditions."
        )
    if cos_theta_delta <= 0.0 or cos_theta_beta <= 0.0:
        raise ValueError("Invalid angle combination: denominator term <= 0.")

    sqrt_term = math.sqrt(
        (sin_phi_delta * sin_phi_beta) / (cos_theta_delta * cos_theta_beta)
    )
    denominator = (
        cos_theta ** 2 * cos_theta_delta * (1.0 + sqrt_term) ** 2
    )

    return numerator / denominator


def coulomb_passive_earth_pressure(
    gamma: float,
    H: float,
    phi_prime_deg: float,
    theta_deg: float = 0.0,
    delta_deg: float = 0.0,
    beta_deg: float = 0.0,
) -> float:
    """Coulomb passive earth pressure force (Equation 4-11).

    Computes the passive earth pressure force for a wall with a sloping
    face (theta), sloping backfill (beta), and interface friction angle
    (delta) using the Coulomb method.

    Note: The Coulomb method can significantly overestimate passive
    pressures, especially when delta/phi' > 0.4.  The log spiral method
    (Equation 4-12) should be used instead for passive pressure
    calculations involving wall friction.

    .. math::
        P_P = \\frac{1}{2} \\gamma H^2
              \\frac{\\cos^2(\\phi' + \\theta)}
              {\\cos^2\\theta \\cdot \\cos(\\theta - \\delta)
              \\left[1 - \\sqrt{\\frac{\\sin(\\phi' + \\delta)\\sin(\\phi' + \\beta)}
              {\\cos(\\theta - \\delta)\\cos(\\theta - \\beta)}}\\right]^2}

    Parameters
    ----------
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Wall height (ft or m).
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    theta_deg : float, optional
        Slope angle of wall face from vertical (degrees).  Default is 0.0.
    delta_deg : float, optional
        Interface friction angle between wall and soil (degrees).
        Default is 0.0.
    beta_deg : float, optional
        Slope of backfill surface from horizontal (degrees).  Default is 0.0.

    Returns
    -------
    float
        Coulomb passive earth pressure force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma* or *H* is negative, or if the denominator sqrt term
        exceeds 1.0 (physically invalid).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-11, p. 190.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")

    phi = math.radians(phi_prime_deg)
    theta = math.radians(theta_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)

    numerator = math.cos(phi + theta) ** 2

    cos_theta = math.cos(theta)
    cos_theta_delta = math.cos(theta - delta)
    cos_theta_beta = math.cos(theta - beta)

    sin_phi_delta = math.sin(phi + delta)
    sin_phi_beta = math.sin(phi + beta)

    if cos_theta_delta <= 0.0 or cos_theta_beta <= 0.0:
        raise ValueError("Invalid angle combination: denominator term <= 0.")

    inner = (sin_phi_delta * sin_phi_beta) / (cos_theta_delta * cos_theta_beta)
    sqrt_term = math.sqrt(inner)

    bracket = 1.0 - sqrt_term
    if bracket <= 0.0:
        raise ValueError(
            "Denominator bracket term is non-positive. "
            "Coulomb passive solution is invalid for these angles. "
            "Use the log spiral method (Equation 4-12) instead."
        )

    denominator = cos_theta ** 2 * cos_theta_delta * bracket ** 2

    Kp_coulomb = numerator / denominator
    return 0.5 * gamma * H ** 2 * Kp_coulomb


def coulomb_passive_coefficient(
    phi_prime_deg: float,
    theta_deg: float = 0.0,
    delta_deg: float = 0.0,
    beta_deg: float = 0.0,
) -> float:
    """Coulomb passive earth pressure coefficient (from Equation 4-11).

    Extracts the passive earth pressure coefficient KP from the Coulomb
    passive equation.

    Note: Coulomb passive KP is unconservative when delta/phi' > 0.4.
    Use log_spiral_passive_coefficient() instead.

    .. math::
        K_P = \\frac{\\cos^2(\\phi' + \\theta)}
              {\\cos^2\\theta \\cdot \\cos(\\theta - \\delta)
              \\left[1 - \\sqrt{\\frac{\\sin(\\phi' + \\delta)\\sin(\\phi' + \\beta)}
              {\\cos(\\theta - \\delta)\\cos(\\theta - \\beta)}}\\right]^2}

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    theta_deg : float, optional
        Slope angle of wall face from vertical (degrees).  Default is 0.0.
    delta_deg : float, optional
        Interface friction angle between wall and soil (degrees).
        Default is 0.0.
    beta_deg : float, optional
        Slope of backfill surface from horizontal (degrees).  Default is 0.0.

    Returns
    -------
    float
        Coulomb passive earth pressure coefficient, KP (dimensionless).

    Raises
    ------
    ValueError
        If angle combinations produce an invalid solution.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-11, p. 190.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")

    phi = math.radians(phi_prime_deg)
    theta = math.radians(theta_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)

    numerator = math.cos(phi + theta) ** 2

    cos_theta = math.cos(theta)
    cos_theta_delta = math.cos(theta - delta)
    cos_theta_beta = math.cos(theta - beta)

    sin_phi_delta = math.sin(phi + delta)
    sin_phi_beta = math.sin(phi + beta)

    if cos_theta_delta <= 0.0 or cos_theta_beta <= 0.0:
        raise ValueError("Invalid angle combination: denominator term <= 0.")

    inner = (sin_phi_delta * sin_phi_beta) / (cos_theta_delta * cos_theta_beta)
    sqrt_term = math.sqrt(inner)

    bracket = 1.0 - sqrt_term
    if bracket <= 0.0:
        raise ValueError(
            "Denominator bracket is non-positive. Coulomb passive solution "
            "is invalid for these angles. Use log spiral method instead."
        )

    denominator = cos_theta ** 2 * cos_theta_delta * bracket ** 2
    return numerator / denominator


# ===========================================================================
# Equation 4-12: Log Spiral Method
# ===========================================================================

def log_spiral_passive_coefficient(
    phi_prime_deg: float,
    delta_deg: float = 0.0,
) -> float:
    """Log spiral passive earth pressure coefficient (Equation 4-12).

    Approximation for the log spiral passive earth pressure coefficient
    for vertical walls (theta = 0) with horizontal backfill (beta = 0).
    The log spiral method is preferred over Coulomb for passive pressure
    calculations, particularly when delta/phi' > 0.4.

    When delta = 0, this equation reduces to the Rankine passive KP.

    .. math::
        \\ln(K_P) = \\ln\\left(\\frac{1 + \\sin \\phi'}{1 - \\sin \\phi'}\\right)
                   + 1.443 \\cdot \\frac{\\delta}{\\phi'} \\cdot \\sin \\phi'
                   \\cdot \\ln\\left(\\frac{1 + \\sin \\phi'}{1 - \\sin \\phi'}\\right)

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    delta_deg : float, optional
        Wall-soil interface friction angle (degrees).  Default is 0.0.

    Returns
    -------
    float
        Log spiral passive earth pressure coefficient, KP (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is not in (0, 90) or *delta_deg* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-12, p. 191.
    Caquot, A. and Kerisel, J. (1948).
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    if delta_deg < 0.0:
        raise ValueError("delta_deg must be non-negative.")

    phi_rad = math.radians(phi_prime_deg)
    sin_phi = math.sin(phi_rad)

    ratio = (1.0 + sin_phi) / (1.0 - sin_phi)
    ln_ratio = math.log(ratio)

    # delta/phi' ratio (using degrees for the ratio since it cancels)
    if phi_prime_deg > 0.0:
        delta_phi_ratio = delta_deg / phi_prime_deg
    else:
        delta_phi_ratio = 0.0

    ln_Kp = ln_ratio + 1.443 * delta_phi_ratio * sin_phi * ln_ratio

    return math.exp(ln_Kp)


# ===========================================================================
# Equations 4-13 and 4-14: Equivalent Fluid Pressures
# ===========================================================================

def equivalent_fluid_unit_weight(
    gamma: float,
    K: float,
) -> float:
    """Equivalent fluid unit weight (Equation 4-13).

    Computes the equivalent fluid unit weight for calculating earth
    pressures using hydrostatic fluid analogy.  The equivalent fluid
    unit weight combines the soil unit weight and earth pressure
    coefficient into a single parameter for simplified calculations.

    Limited to walls less than 20 feet (6 m) tall.

    .. math::
        \\gamma_{eq} = K \\cdot \\gamma

    Parameters
    ----------
    gamma : float
        Unit weight of the backfill soil (pcf or kN/m^3).
    K : float
        Appropriate earth pressure coefficient (at-rest or active,
        dimensionless).

    Returns
    -------
    float
        Equivalent fluid unit weight (pcf or kN/m^3).

    Raises
    ------
    ValueError
        If *gamma* or *K* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-13, p. 195.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if K < 0.0:
        raise ValueError("K must be non-negative.")
    return K * gamma


def horizontal_earth_pressure_with_surcharge(
    gamma_eq: float,
    z: float,
    K: float,
    q: float,
) -> float:
    """Horizontal earth pressure at depth with uniform surcharge (Equation 4-14).

    Computes the horizontal earth pressure at the bottom of a wall
    considering both the equivalent fluid pressure from the soil and a
    uniform surcharge applied to the ground surface.

    .. math::
        \\sigma_h = \\gamma_{eq} \\cdot z + K \\cdot q

    Parameters
    ----------
    gamma_eq : float
        Equivalent fluid unit weight (pcf or kN/m^3).
    z : float
        Depth below ground surface (ft or m).
    K : float
        Horizontal earth pressure coefficient (dimensionless).
    q : float
        Uniform surcharge pressure at the ground surface (psf or kPa).

    Returns
    -------
    float
        Horizontal earth pressure at depth z (psf or kPa).

    Raises
    ------
    ValueError
        If *z* is negative or *gamma_eq* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-14, p. 196.
    """
    if z < 0.0:
        raise ValueError("z must be non-negative.")
    if gamma_eq < 0.0:
        raise ValueError("gamma_eq must be non-negative.")
    return gamma_eq * z + K * q


# ===========================================================================
# Equations 4-15 through 4-17: Seismic Earth Pressures -- Yielding Walls
# (Mononobe-Okabe Method)
# ===========================================================================

def mononobe_okabe_active_coefficient(
    phi_prime_deg: float,
    kh: float,
    kv: float = 0.0,
    theta_deg: float = 0.0,
    delta_deg: float = 0.0,
    beta_deg: float = 0.0,
) -> float:
    """Mononobe-Okabe seismic active earth pressure coefficient (Equation 4-15).

    Computes the seismic active earth pressure coefficient for yielding
    walls that can rotate or translate sufficiently to develop active
    pressures during an earthquake.  Assumes cohesionless backfill and
    the phreatic surface is below the wall base.

    .. math::
        K_{AE} = \\frac{\\cos^2(\\phi' - \\theta - \\psi)}
                 {\\cos\\psi \\cdot \\cos^2\\theta \\cdot \\cos(\\theta + \\delta + \\psi)
                 \\left[1 + \\sqrt{\\frac{\\sin(\\phi' + \\delta)\\sin(\\phi' - \\beta - \\psi)}
                 {\\cos(\\theta + \\delta + \\psi)\\cos(\\theta - \\beta)}}\\right]^2}

    where :math:`\\psi = \\tan^{-1}\\left(\\frac{k_h}{1 - k_v}\\right)`

    Parameters
    ----------
    phi_prime_deg : float
        Backfill soil friction angle (degrees).
    kh : float
        Horizontal ground acceleration coefficient (dimensionless, in g).
    kv : float, optional
        Vertical ground acceleration coefficient (dimensionless, in g).
        Default is 0.0.
    theta_deg : float, optional
        Slope of wall back from vertical (degrees).  Default is 0.0.
    delta_deg : float, optional
        Interface friction angle between soil backfill and wall (degrees).
        Default is 0.0.
    beta_deg : float, optional
        Slope of backfill from horizontal (degrees).  Default is 0.0.

    Returns
    -------
    float
        Seismic active earth pressure coefficient, KAE (dimensionless).

    Raises
    ------
    ValueError
        If kv >= 1.0, if phi_prime_deg is out of range, or if the
        argument under the square root is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-15, p. 214.
    Mononobe, N. and Matsuo, H. (1929); Okabe, S. (1924).
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    if kv >= 1.0:
        raise ValueError("kv must be less than 1.0.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")

    psi = math.atan(kh / (1.0 - kv))

    phi = math.radians(phi_prime_deg)
    theta = math.radians(theta_deg)
    delta = math.radians(delta_deg)
    beta = math.radians(beta_deg)

    numerator = math.cos(phi - theta - psi) ** 2

    cos_psi = math.cos(psi)
    cos_theta = math.cos(theta)
    cos_td_psi = math.cos(theta + delta + psi)
    cos_t_beta = math.cos(theta - beta)

    sin_phi_delta = math.sin(phi + delta)
    sin_phi_beta_psi = math.sin(phi - beta - psi)

    if sin_phi_beta_psi < 0.0:
        raise ValueError(
            "phi' - beta - psi < 0: The M-O method is not valid for these "
            "conditions.  Consider numerical methods."
        )

    if cos_td_psi <= 0.0 or cos_t_beta <= 0.0:
        raise ValueError("Invalid angle combination: denominator term <= 0.")

    sqrt_term = math.sqrt(
        (sin_phi_delta * sin_phi_beta_psi) / (cos_td_psi * cos_t_beta)
    )

    denominator = (
        cos_psi * cos_theta ** 2 * cos_td_psi * (1.0 + sqrt_term) ** 2
    )

    return numerator / denominator


def mononobe_okabe_active_force(
    gamma: float,
    H: float,
    KAE: float,
    kv: float = 0.0,
) -> float:
    """Mononobe-Okabe seismic active earth pressure force (Equation 4-16).

    Computes the total seismic active earth pressure force including both
    static and dynamic components for a yielding wall.

    .. math::
        P_{AE} = \\frac{1}{2} \\gamma (1 - k_v) K_{AE} H^2

    Parameters
    ----------
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Wall height (ft or m).
    KAE : float
        Seismic active earth pressure coefficient from Equation 4-15
        (dimensionless).
    kv : float, optional
        Vertical ground acceleration coefficient (dimensionless, in g).
        Default is 0.0.

    Returns
    -------
    float
        Total seismic active earth pressure force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma* or *H* is negative, or *kv* >= 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-16, p. 215.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if kv >= 1.0:
        raise ValueError("kv must be less than 1.0.")
    return 0.5 * gamma * (1.0 - kv) * KAE * H ** 2


def seed_whitman_dynamic_increment(
    gamma: float,
    H: float,
    kh: float,
) -> float:
    """Seed-Whitman simplified dynamic earth pressure increment (Equation 4-17).

    Computes the dynamic component of the seismic earth pressure force
    using the simplified Seed and Whitman (1970) approach.  This is
    limited to cases with horizontal backfill, vertical wall face, and
    no wall friction, with wall height less than 20 feet.

    The dynamic component acts at 0.6H above the wall base.

    .. math::
        \\Delta P_{AE} = \\frac{3}{8} k_h \\gamma H^2

    Parameters
    ----------
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Wall height (ft or m).
    kh : float
        Horizontal ground acceleration coefficient (dimensionless, in g),
        assumed equal to the maximum ground acceleration.

    Returns
    -------
    float
        Dynamic earth pressure force increment per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma*, *H*, or *kh* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-17, p. 215.
    Seed, H.B. and Whitman, R.V. (1970).
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")
    return (3.0 / 8.0) * kh * gamma * H ** 2


# ===========================================================================
# Equation 4-18: Seismic Earth Pressures -- Nonyielding Walls
# ===========================================================================

def wood_nonyielding_seismic_force(
    gamma: float,
    H: float,
    kh: float,
) -> float:
    """Seismic earth pressure force increment for nonyielding walls (Equation 4-18).

    Computes the increase in earth pressure force for nonyielding walls
    (e.g., basement walls restrained top and bottom) during seismic
    events using Wood (1973).  This force is added to the existing static
    earth pressure.

    The point of application is assumed at 0.6H above the wall base.

    .. math::
        \\Delta P_E = k_h \\gamma H^2

    Parameters
    ----------
    gamma : float
        Unit weight of backfill soil (pcf or kN/m^3).
    H : float
        Wall height (ft or m).
    kh : float
        Horizontal ground acceleration coefficient (dimensionless, in g).

    Returns
    -------
    float
        Seismic earth pressure force increment per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma*, *H*, or *kh* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-18, p. 217.
    Wood, J.H. (1973).
    """
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")
    return kh * gamma * H ** 2


# ===========================================================================
# Equations 4-19 and 4-20: Dynamic Water Pressure (High Hydraulic
# Conductivity Soils)
# ===========================================================================

def hydrodynamic_water_pressure(
    gamma_w: float,
    kh: float,
    H: float,
    z: float,
) -> float:
    """Hydrodynamic water pressure for high-k soils (Equation 4-19).

    Computes the hydrodynamic water pressure at a depth z below the
    phreatic surface for soils with hydraulic conductivity greater than
    about 10^-3 cm/s, where water acts independently of the soil skeleton.

    Can also estimate hydrodynamic pressure from an impounded reservoir.

    .. math::
        p_w = 0.875 \\cdot k_h \\cdot \\gamma_w \\cdot \\sqrt{H \\cdot z}

    Parameters
    ----------
    gamma_w : float
        Unit weight of water (62.4 pcf or 9.81 kN/m^3).
    kh : float
        Horizontal ground acceleration coefficient (dimensionless, in g).
    H : float
        Total depth of water behind the wall (ft or m).
    z : float
        Depth below the phreatic surface at the point of interest (ft or m).

    Returns
    -------
    float
        Hydrodynamic water pressure at depth z (psf or kPa).

    Raises
    ------
    ValueError
        If *gamma_w*, *H*, or *z* is negative, or if z > H.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-19, p. 217.
    """
    if gamma_w < 0.0:
        raise ValueError("gamma_w must be non-negative.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    if z < 0.0:
        raise ValueError("z must be non-negative.")
    if z > H:
        raise ValueError("z must not exceed H.")
    return 0.875 * kh * gamma_w * math.sqrt(H * z)


def hydrodynamic_water_force(
    gamma_w: float,
    kh: float,
    H: float,
) -> float:
    """Resultant hydrodynamic water pressure force (Equation 4-20).

    Computes the resultant force from the hydrodynamic water pressure
    distribution for high hydraulic conductivity soils.

    When using this method, the seismic earth force (Equation 4-16)
    should be calculated using the buoyant unit weight.

    .. math::
        P_w = \\frac{7}{12} k_h \\gamma_w H^2

    Parameters
    ----------
    gamma_w : float
        Unit weight of water (62.4 pcf or 9.81 kN/m^3).
    kh : float
        Horizontal ground acceleration coefficient (dimensionless, in g).
    H : float
        Total depth of water behind the wall (ft or m).

    Returns
    -------
    float
        Resultant hydrodynamic water force per unit length of wall
        (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma_w*, *kh*, or *H* is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-20, p. 218.
    Kramer, S.L. (1996).
    """
    if gamma_w < 0.0:
        raise ValueError("gamma_w must be non-negative.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")
    if H < 0.0:
        raise ValueError("H must be non-negative.")
    return (7.0 / 12.0) * kh * gamma_w * H ** 2


# ===========================================================================
# Equations 4-21 and 4-22: Seismic Earth Pressures for Low Hydraulic
# Conductivity Soils
# ===========================================================================

def seismic_psi_low_permeability(
    gamma_sat: float,
    gamma_w: float,
    kh: float,
    kv: float = 0.0,
    ru: float = 0.0,
) -> float:
    """Seismic angle psi for low hydraulic conductivity soils (Equation 4-21).

    Computes the modified seismic inertia angle psi for use in the
    Mononobe-Okabe equation (Equation 4-15) when the backfill has low
    hydraulic conductivity (k < 10^-3 cm/s) and water moves with the
    soil skeleton.

    .. math::
        \\psi = \\tan^{-1}\\left(
            \\frac{\\gamma_{sat} \\cdot k_h}
            {(\\gamma_{sat} - \\gamma_w)(1 - r_u)(1 - k_v)}
        \\right)

    Parameters
    ----------
    gamma_sat : float
        Saturated total unit weight of soil (pcf or kN/m^3).
    gamma_w : float
        Unit weight of water (62.4 pcf or 9.81 kN/m^3).
    kh : float
        Horizontal ground acceleration (g).
    kv : float, optional
        Vertical ground acceleration (g).  Default is 0.0.
    ru : float, optional
        Pore pressure coefficient (dimensionless, 0 to 1).
        Default is 0.0.

    Returns
    -------
    float
        Modified seismic inertia angle psi (radians).

    Raises
    ------
    ValueError
        If gamma_sat <= gamma_w, kv >= 1.0, or ru >= 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-21, p. 218.
    Kramer, S.L. (1996).
    """
    if gamma_sat <= gamma_w:
        raise ValueError("gamma_sat must be greater than gamma_w.")
    if kv >= 1.0:
        raise ValueError("kv must be less than 1.0.")
    if ru >= 1.0:
        raise ValueError("ru must be less than 1.0.")
    if kh < 0.0:
        raise ValueError("kh must be non-negative.")

    numerator = gamma_sat * kh
    denominator = (gamma_sat - gamma_w) * (1.0 - ru) * (1.0 - kv)

    return math.atan(numerator / denominator)


def seismic_adjusted_unit_weight(
    gamma_sat: float,
    gamma_w: float,
    ru: float = 0.0,
) -> float:
    """Adjusted unit weight for seismic analysis with excess pore pressure (Equation 4-22).

    Computes the adjusted unit weight for use with Equation 4-16 when
    excess pore pressures develop during seismic events in low hydraulic
    conductivity soils.

    .. math::
        \\gamma = (\\gamma_{sat} - \\gamma_w)(1 - r_u)

    Parameters
    ----------
    gamma_sat : float
        Saturated total unit weight of soil (pcf or kN/m^3).
    gamma_w : float
        Unit weight of water (62.4 pcf or 9.81 kN/m^3).
    ru : float, optional
        Pore pressure coefficient (dimensionless, 0 to 1).
        Default is 0.0.

    Returns
    -------
    float
        Adjusted unit weight for seismic earth pressure calculation
        (pcf or kN/m^3).

    Raises
    ------
    ValueError
        If gamma_sat <= gamma_w or ru >= 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-22, p. 218.
    """
    if gamma_sat <= gamma_w:
        raise ValueError("gamma_sat must be greater than gamma_w.")
    if ru >= 1.0:
        raise ValueError("ru must be less than 1.0.")
    return (gamma_sat - gamma_w) * (1.0 - ru)


# ===========================================================================
# Equations 4-23 through 4-28: Base Pressure Distribution for Rigid
# Retaining Walls
# ===========================================================================

def base_pressure_max_trapezoidal(
    R: float,
    e: float,
    B: float,
) -> float:
    """Maximum base pressure for trapezoidal distribution (Equation 4-23).

    Computes the maximum bearing pressure at the toe of a rigid retaining
    wall when the resultant R acts within the middle third of the base,
    producing a trapezoidal pressure distribution.

    .. math::
        q_{max} = \\frac{R}{B} + \\frac{6 R e}{B^2}

    Parameters
    ----------
    R : float
        Resultant normal force on the base of the wall (lb/ft or kN/m).
    e : float
        Eccentricity -- distance from the centroid of the base to the
        point where R acts (ft or m).  Positive toward the toe.
    B : float
        Width of the wall base (ft or m).

    Returns
    -------
    float
        Maximum base pressure at the toe (psf or kPa).

    Raises
    ------
    ValueError
        If *B* is not positive, or if *e* > B/6 (resultant outside
        middle third -- use Equation 4-25 instead).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-23, p. 221.
    """
    if B <= 0.0:
        raise ValueError("B must be positive.")
    if abs(e) > B / 6.0:
        raise ValueError(
            "Eccentricity exceeds B/6.  Resultant is outside the middle "
            "third.  Use base_pressure_max_triangular() (Eq. 4-25) instead."
        )
    return R / B + 6.0 * R * e / B ** 2


def base_pressure_min_trapezoidal(
    R: float,
    e: float,
    B: float,
) -> float:
    """Minimum base pressure for trapezoidal distribution (Equation 4-24).

    Computes the minimum bearing pressure at the heel of a rigid retaining
    wall when the resultant R acts within the middle third of the base.

    .. math::
        q_{min} = \\frac{R}{B} - \\frac{6 R e}{B^2}

    Parameters
    ----------
    R : float
        Resultant normal force on the base (lb/ft or kN/m).
    e : float
        Eccentricity of R from the centroid of the base (ft or m).
        Positive toward the toe.
    B : float
        Width of the wall base (ft or m).

    Returns
    -------
    float
        Minimum base pressure at the heel (psf or kPa).

    Raises
    ------
    ValueError
        If *B* is not positive, or if *e* > B/6.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-24, p. 221.
    """
    if B <= 0.0:
        raise ValueError("B must be positive.")
    if abs(e) > B / 6.0:
        raise ValueError(
            "Eccentricity exceeds B/6.  Resultant is outside the middle "
            "third.  Use base_pressure_max_triangular() (Eq. 4-25) instead."
        )
    return R / B - 6.0 * R * e / B ** 2


def base_pressure_max_triangular(
    R: float,
    x0: float,
) -> float:
    """Maximum base pressure when resultant is outside middle third (Equation 4-25).

    Computes the maximum bearing pressure at the toe when the resultant
    is located outside the middle third of the base.  Only a portion of
    the base is under compression.

    .. math::
        q_{max} = \\frac{2R}{3 x_0}

    Parameters
    ----------
    R : float
        Resultant normal force on the base (lb/ft or kN/m).
    x0 : float
        Horizontal distance from the toe to the resultant, calculated as
        x0 = B/2 - e (ft or m).

    Returns
    -------
    float
        Maximum base pressure at the toe (psf or kPa).

    Raises
    ------
    ValueError
        If *x0* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-25, p. 221.
    """
    if x0 <= 0.0:
        raise ValueError("x0 must be positive (x0 = B/2 - e).")
    return 2.0 * R / (3.0 * x0)


def effective_base_width_triangular(x0: float) -> float:
    """Effective base width under compression -- triangular case (Equation 4-26).

    Computes the portion of the wall base that is under compression when
    the resultant force is outside the middle third of the base,
    producing a triangular pressure distribution.

    .. math::
        B_e = 3 x_0

    Parameters
    ----------
    x0 : float
        Horizontal distance from the toe to the resultant (ft or m),
        x0 = B/2 - e.

    Returns
    -------
    float
        Width of base under compression (ft or m).

    Raises
    ------
    ValueError
        If *x0* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-26, p. 221.
    """
    if x0 <= 0.0:
        raise ValueError("x0 must be positive.")
    return 3.0 * x0


def base_pressure_uniform(R: float, x0: float) -> float:
    """Simplified uniform base pressure distribution (Equation 4-27).

    Computes a uniform base pressure over a width of 2*x0, used as a
    simplified alternative to the triangular distribution when the
    resultant falls outside the middle third.

    .. math::
        q_{max} = \\frac{R}{2 x_0}

    Parameters
    ----------
    R : float
        Resultant normal force on the base (lb/ft or kN/m).
    x0 : float
        Horizontal distance from the toe to the resultant (ft or m),
        x0 = B/2 - e.

    Returns
    -------
    float
        Uniform base pressure (psf or kPa).

    Raises
    ------
    ValueError
        If *x0* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-27, p. 221.
    """
    if x0 <= 0.0:
        raise ValueError("x0 must be positive.")
    return R / (2.0 * x0)


def effective_base_width_uniform(x0: float) -> float:
    """Effective base width under compression -- uniform case (Equation 4-28).

    Computes the portion of the wall base under compression when a
    simplified uniform pressure distribution is assumed.

    .. math::
        B_e = 2 x_0

    Parameters
    ----------
    x0 : float
        Horizontal distance from the toe to the resultant (ft or m),
        x0 = B/2 - e.

    Returns
    -------
    float
        Width of base under compression (ft or m).

    Raises
    ------
    ValueError
        If *x0* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-28, p. 221.
    """
    if x0 <= 0.0:
        raise ValueError("x0 must be positive.")
    return 2.0 * x0


# ===========================================================================
# Equations 4-29 and 4-30: Allowable Passive Resistance for Flexible
# Retaining Structures
# ===========================================================================

def allowable_passive_resistance(
    Pp: float,
    F: float,
) -> float:
    """Allowable passive earth pressure force (Equation 4-29).

    Computes the allowable passive resistance by applying a factor of
    safety to the calculated passive earth pressure force.  The factor
    of safety is applied to the load (similar to bearing capacity).

    Recommended F values:
    - F = 2.0 to 3.0 for coarse-grained soils (effective stress design)
    - F = 1.5 to 2.0 for undrained analysis in fine-grained soils

    .. math::
        P_{P,allow} = \\frac{P_P}{F}

    Parameters
    ----------
    Pp : float
        Passive earth pressure force calculated using shear strength
        parameters (lb/ft or kN/m).
    F : float
        Factor of safety (dimensionless, typically 1.5 to 3.0).

    Returns
    -------
    float
        Allowable passive earth pressure force (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *Pp* is negative or *F* is not greater than 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-29, p. 232.
    """
    if Pp < 0.0:
        raise ValueError("Pp must be non-negative.")
    if F <= 1.0:
        raise ValueError("Factor of safety F must be greater than 1.0.")
    return Pp / F


def allowable_friction_angle(
    phi_prime_deg: float,
    F: float,
) -> float:
    """Allowable effective stress friction angle for passive resistance (Equation 4-30).

    Computes a reduced (allowable) friction angle by applying a factor
    of safety to the tangent of the friction angle.  This provides a
    factor of safety on shear strength, similar to slope stability
    analysis.  A factor of safety of 1.5 to 2.0 is appropriate for all
    soil types and analysis conditions.

    .. math::
        \\phi'_{allow} = \\tan^{-1}\\left(\\frac{\\tan \\phi'}{F}\\right)

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress friction angle (degrees).
    F : float
        Factor of safety (dimensionless, typically 1.5 to 2.0).

    Returns
    -------
    float
        Allowable friction angle (degrees).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is out of range or *F* is not greater than 1.0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-30, p. 232.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be between 0 and 90 degrees.")
    if F <= 1.0:
        raise ValueError("Factor of safety F must be greater than 1.0.")
    tan_phi = math.tan(math.radians(phi_prime_deg))
    return math.degrees(math.atan(tan_phi / F))


# ===========================================================================
# Equation 4-31: Relative Flexibility of Anchored Bulkheads
# ===========================================================================

def relative_flexibility_anchored_bulkhead(
    H: float,
    D: float,
    E: float,
    I: float,
) -> float:
    """Relative flexibility of an anchored bulkhead (Equation 4-31).

    Computes the relative flexibility of an anchored bulkhead wall, which
    is used with Rowe (1952) moment reduction charts (Figure 4-36) to
    determine the ratio of design moment to theoretical moment.

    Moment reduction applies only for walls penetrating into medium dense
    or better sand.  No moment reduction for penetration into fine-grained
    soils or loose sands.

    .. math::
        \\rho = \\frac{(H + D)^4}{E I}

    Parameters
    ----------
    H : float
        Exposed height of the wall above the dredge line (inches).
    D : float
        Depth of penetration below the dredge line (inches).
    E : float
        Young's modulus of the wall material (psi).
    I : float
        Moment of inertia of the wall section (in^4/ft).

    Returns
    -------
    float
        Relative flexibility, rho (in^3/lb, dimensionless ratio when
        consistent units are used).

    Raises
    ------
    ValueError
        If *H*, *D*, *E*, or *I* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Equation 4-31, p. 235.
    Rowe, P.W. (1952).
    """
    if H <= 0.0:
        raise ValueError("H must be positive.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if I <= 0.0:
        raise ValueError("I must be positive.")
    return (H + D) ** 4 / (E * I)


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
# FIGURE 4-36: Rowe (1952) Moment Reduction for Anchored Bulkheads
# ===========================================================================

# Digitised from Figure 4-36.  x-axis = log10(rho), y-axis = Md/Mmax.
# Three curves for different soil densities below dredge line.

_FIG_4_36_LOG_RHO = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]

_FIG_4_36_DENSE_SAND = [1.00, 1.00, 0.95, 0.82, 0.62, 0.46, 0.35, 0.30]
_FIG_4_36_MEDIUM_SAND = [1.00, 1.00, 0.97, 0.88, 0.72, 0.56, 0.45, 0.38]
_FIG_4_36_LOOSE_SAND = [1.00, 1.00, 1.00, 0.95, 0.85, 0.72, 0.60, 0.52]

_FIG_4_36_CURVES = {
    "dense_sand": _FIG_4_36_DENSE_SAND,
    "medium_sand": _FIG_4_36_MEDIUM_SAND,
    "loose_sand": _FIG_4_36_LOOSE_SAND,
}


def figure_4_36_moment_reduction(log_rho: float, soil_type: str) -> float:
    """Rowe (1952) moment reduction factor for anchored bulkheads
    (Figure 4-36).

    Returns the ratio of design moment to theoretical free-earth-support
    moment (Md / Mmax) as a function of log10(rho), where rho is the
    relative flexibility from ``relative_flexibility_anchored_bulkhead``
    (Equation 4-31).

    Moment reduction applies ONLY for walls penetrating into sands of
    medium density or better.  Do NOT apply for fine-grained soils or
    loose sands (use loose_sand conservatively if needed).

    Parameters
    ----------
    log_rho : float
        Logarithm (base 10) of the relative flexibility rho.
        Typical range: -3.5 to 0.0.
    soil_type : str
        Soil below the dredge line: ``"dense_sand"``,
        ``"medium_sand"``, or ``"loose_sand"``.

    Returns
    -------
    float
        Moment reduction ratio Md/Mmax (dimensionless, 0 to 1).

    Raises
    ------
    ValueError
        If *soil_type* is not recognised.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 4, Figure 4-36, p. 236.
    Rowe, P.W. (1952).
    """
    key = soil_type.lower().strip()
    if key not in _FIG_4_36_CURVES:
        raise ValueError(
            f"Unknown soil_type '{soil_type}'. "
            f"Choose from: {list(_FIG_4_36_CURVES.keys())}"
        )
    return _linterp(log_rho, _FIG_4_36_LOG_RHO, _FIG_4_36_CURVES[key])


def plot_figure_4_36(log_rho=None, soil_type=None, ax=None, show=True,
                      **kwargs):
    """Reproduce UFC Figure 4-36: Rowe Moment Reduction Charts.

    Plots the three Rowe (1952) moment reduction curves for dense,
    medium, and loose sand.

    Parameters
    ----------
    log_rho : float, optional
        log10(rho) query point.
    soil_type : str, optional
        Which curve to mark: "dense_sand", "medium_sand", "loose_sand".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    show : bool, optional
        If True, calls plt.show().

    Returns
    -------
    matplotlib.axes.Axes
    """
    from geotech_common.plotting import get_pyplot, setup_engineering_plot
    plt = get_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Smooth interpolated curves
    lr_smooth = [i * 0.05 - 4.0 for i in range(0, 101)]  # -4.0 to 1.0
    colors = {'dense_sand': '#1f77b4', 'medium_sand': '#ff7f0e',
              'loose_sand': '#2ca02c'}
    labels = {'dense_sand': 'Dense Sand', 'medium_sand': 'Medium Sand',
              'loose_sand': 'Loose Sand'}

    for stype, curve_data in _FIG_4_36_CURVES.items():
        vals = [figure_4_36_moment_reduction(lr, stype) for lr in lr_smooth]
        ax.plot(lr_smooth, vals, '-', color=colors[stype], linewidth=1.5,
                label=labels[stype])
        ax.plot(_FIG_4_36_LOG_RHO, curve_data, 'o', color=colors[stype],
                markersize=5)

    # Query point
    if log_rho is not None and soil_type is not None:
        key = soil_type.lower().strip()
        ratio_q = figure_4_36_moment_reduction(log_rho, key)
        ax.plot(log_rho, ratio_q, 's', color='red', markersize=10, zorder=5,
                label=f'log(ρ)={log_rho:.2f} → Md/Mmax={ratio_q:.3f}')
        ax.axhline(ratio_q, color='red', linestyle=':', alpha=0.4)
        ax.axvline(log_rho, color='red', linestyle=':', alpha=0.4)

    ax.legend(fontsize=8)
    setup_engineering_plot(
        ax, 'Figure 4-36: Rowe Moment Reduction',
        'log₁₀(ρ)', 'Md / Mmax')
    if show:
        plt.tight_layout()
        plt.show()
    return ax
