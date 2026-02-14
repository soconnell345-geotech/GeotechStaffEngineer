"""
UFC 3-220-20, Prologue P: Shear Strength for Geotechnical Design

Equations from Tables P-1 and P-6, and Equations P-1, P-2, and P-3 covering
Mohr-Coulomb failure criteria, two-parameter and three-parameter power
functions for nonlinear strength envelopes, undrained shear strength power
functions, equivalent Mohr-Coulomb parameter conversion, and undrained
strength ratio (USR) correlations for low-plasticity silts.

Reference:
    UFC 3-220-20, Foundations and Earth Structures,
    16 January 2025
"""

import math


# ---------------------------------------------------------------------------
# Table P-1 -- Mohr-Coulomb failure criteria
# ---------------------------------------------------------------------------

def mohr_coulomb_drained(c_prime: float, sigma_prime: float,
                         phi_prime_deg: float) -> float:
    """Drained (effective stress) shear strength on the failure plane
    (Table P-1, Row 1).

    Calculates the shear strength of a soil using the Mohr-Coulomb
    failure criterion expressed in terms of effective stress parameters.
    Applicable for any degree of saturation (0 <= S <= 100 %).

    .. math::
        s = c' + \\sigma' \\tan \\phi'

    Parameters
    ----------
    c_prime : float
        Effective stress (drained) cohesion intercept (psf, kPa, or any
        consistent stress unit).
    sigma_prime : float
        Effective normal stress on the failure plane (same stress unit as
        *c_prime*).
    phi_prime_deg : float
        Effective stress (drained) friction angle (degrees).

    Returns
    -------
    float
        Shear strength *s* on the failure plane (same stress unit as inputs).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is outside the range [0, 90).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-1, p. 40.
    """
    if phi_prime_deg < 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be in the range [0, 90).")
    phi_rad = math.radians(phi_prime_deg)
    return c_prime + sigma_prime * math.tan(phi_rad)


def mohr_coulomb_undrained_partial(c: float, sigma: float,
                                   phi_deg: float) -> float:
    """Undrained (total stress) shear strength for partially saturated soil
    (Table P-1, Row 2).

    Calculates the shear strength using total stress parameters for soils
    with degree of saturation S < 100 %.  Both cohesion and friction angle
    contribute to strength.

    .. math::
        s = c + \\sigma \\tan \\phi

    Parameters
    ----------
    c : float
        Total stress (undrained) cohesion intercept (psf, kPa, or any
        consistent stress unit).
    sigma : float
        Total normal stress on the failure plane (same stress unit as *c*).
    phi_deg : float
        Total stress (undrained) friction angle (degrees).

    Returns
    -------
    float
        Shear strength *s* on the failure plane (same stress unit as inputs).

    Raises
    ------
    ValueError
        If *phi_deg* is outside the range [0, 90).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-1, p. 40.
    """
    if phi_deg < 0.0 or phi_deg >= 90.0:
        raise ValueError("phi_deg must be in the range [0, 90).")
    phi_rad = math.radians(phi_deg)
    return c + sigma * math.tan(phi_rad)


def mohr_coulomb_undrained_saturated(c: float) -> float:
    """Undrained shear strength for fully saturated soil, phi = 0 case
    (Table P-1, Row 3).

    For fully saturated soils (S = 100 %), the undrained shear strength
    is independent of the normal stress on the failure plane.  The friction
    angle is zero, and the shear strength equals the cohesion intercept.

    .. math::
        s_u = c

    Parameters
    ----------
    c : float
        Total stress cohesion intercept, equal to the undrained shear
        strength *s_u* (psf, kPa, or any consistent stress unit).

    Returns
    -------
    float
        Undrained shear strength *s_u* (same stress unit as *c*).

    Raises
    ------
    ValueError
        If *c* is negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-1, p. 40.
    """
    if c < 0.0:
        raise ValueError("c (undrained shear strength) must be non-negative.")
    return c


# ---------------------------------------------------------------------------
# Equation P-1 -- Two-parameter power function (drained envelope)
# ---------------------------------------------------------------------------

def power_function_shear_strength(a: float, b: float,
                                  sigma_prime: float,
                                  p_a: float = 2116.0) -> float:
    """Shear strength from two-parameter power function (Equation P-1).

    Models a nonlinear (curved) drained failure envelope using a power
    function with normalized stress.  Instead of the usual phi' and c',
    the alternative parameters *a* and *b* define the envelope curvature
    and steepness.

    .. math::
        s = a \\, P_a \\left( \\frac{\\sigma'}{P_a} \\right)^b

    Parameters
    ----------
    a : float
        Power function strength parameter defining the steepness of the
        curve (dimensionless).  Must be positive.
    b : float
        Power function strength parameter defining the amount of
        curvature (dimensionless).  Must be positive.  A value of 1.0
        produces a linear envelope passing through the origin.
    sigma_prime : float
        Effective normal stress on the failure plane (psf, kPa, or any
        consistent stress unit matching *p_a*).
    p_a : float, optional
        Atmospheric pressure in the same stress unit as *sigma_prime*.
        Default is 2116 psf (approximately 1 atm).

    Returns
    -------
    float
        Shear strength *s* (same stress unit as *sigma_prime*).

    Raises
    ------
    ValueError
        If *a*, *b*, *p_a*, or *sigma_prime* is non-positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Equation P-1, p. 42.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    if p_a <= 0.0:
        raise ValueError("p_a must be positive.")
    if sigma_prime < 0.0:
        raise ValueError("sigma_prime must be non-negative.")
    return a * p_a * (sigma_prime / p_a) ** b


# ---------------------------------------------------------------------------
# Equation P-2 -- Power function for undrained shear strength
# ---------------------------------------------------------------------------

def power_function_undrained_strength(a_u: float, b_u: float,
                                      sigma_1_con: float,
                                      p_a: float = 2116.0) -> float:
    """Undrained shear strength from power function (Equation P-2).

    Relates undrained shear strength to the effective consolidation
    stress using a two-parameter power function.  This form is used
    to capture the nonlinear variation of undrained strength with
    consolidation stress.

    .. math::
        s_u = a_u \\, P_a \\left( \\frac{\\sigma'_{1,con}}{P_a} \\right)^{b_u}

    Parameters
    ----------
    a_u : float
        Power function strength parameter for undrained strength
        (dimensionless).  Must be positive.
    b_u : float
        Power function strength parameter for undrained strength
        (dimensionless).  Must be positive.
    sigma_1_con : float
        Major effective consolidation stress (psf, kPa, or any
        consistent stress unit matching *p_a*).
    p_a : float, optional
        Atmospheric pressure in the same stress unit as
        *sigma_1_con*.  Default is 2116 psf.

    Returns
    -------
    float
        Undrained shear strength *s_u* (same stress unit as
        *sigma_1_con*).

    Raises
    ------
    ValueError
        If *a_u*, *b_u*, *p_a*, or *sigma_1_con* is non-positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Equation P-2, p. 42.
    """
    if a_u <= 0.0:
        raise ValueError("a_u must be positive.")
    if b_u <= 0.0:
        raise ValueError("b_u must be positive.")
    if p_a <= 0.0:
        raise ValueError("p_a must be positive.")
    if sigma_1_con < 0.0:
        raise ValueError("sigma_1_con must be non-negative.")
    return a_u * p_a * (sigma_1_con / p_a) ** b_u


# ---------------------------------------------------------------------------
# Equation P-3 -- Three-parameter power function
# ---------------------------------------------------------------------------

def three_parameter_power_function(a: float, b: float, t: float,
                                   sigma_prime: float,
                                   p_a: float = 2116.0) -> float:
    """Shear strength from three-parameter power function (Equation P-3).

    Extends the two-parameter power function (Equation P-1) by adding a
    tensile intercept parameter *t*, allowing the envelope to model a
    cohesion intercept.  Suggested by Jiang et al. (2003).

    .. math::
        s = a \\, P_a \\left( \\frac{\\sigma'}{P_a} + t \\right)^b

    Parameters
    ----------
    a : float
        Power function strength parameter defining the steepness of the
        curve (dimensionless).  Must be positive.
    b : float
        Power function strength parameter defining the amount of
        curvature (dimensionless).  Must be positive.
    t : float
        Tensile intercept (*T*) normalized by atmospheric pressure
        (dimensionless).  Represents the tensile strength (attraction)
        of the soil in normalized form.  Must be non-negative.
    sigma_prime : float
        Effective normal stress on the failure plane (psf, kPa, or any
        consistent stress unit matching *p_a*).
    p_a : float, optional
        Atmospheric pressure in the same stress unit as *sigma_prime*.
        Default is 2116 psf.

    Returns
    -------
    float
        Shear strength *s* (same stress unit as *sigma_prime*).

    Raises
    ------
    ValueError
        If *a*, *b*, or *p_a* is non-positive, or if *t* is negative,
        or if the term (sigma'/Pa + t) is negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Equation P-3, p. 42.
    Jiang, J.-C., Baker, R., and Yamagami, T. (2003).
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    if t < 0.0:
        raise ValueError("t must be non-negative.")
    if p_a <= 0.0:
        raise ValueError("p_a must be positive.")
    normalized_sum = (sigma_prime / p_a) + t
    if normalized_sum < 0.0:
        raise ValueError(
            "The term (sigma_prime / p_a + t) must be non-negative."
        )
    return a * p_a * normalized_sum ** b


# ---------------------------------------------------------------------------
# Figure P-4 / P-5 -- Equivalent Mohr-Coulomb from power function
# ---------------------------------------------------------------------------

def equivalent_friction_angle(a: float, b: float,
                              sigma_prime_low: float,
                              sigma_prime_high: float,
                              p_a: float = 2116.0) -> float:
    """Equivalent Mohr-Coulomb friction angle from power function parameters
    (Figure P-4).

    For a specified range of effective normal stresses, converts the
    nonlinear power function envelope into an equivalent linear
    Mohr-Coulomb envelope.  The equivalent friction angle is determined
    by the slope of the line connecting the shear strengths at the two
    bounding stresses.

    .. math::
        \\phi'_{EQ} = \\arctan \\left(
            \\frac{s(\\sigma'_{high}) - s(\\sigma'_{low})}
                 {\\sigma'_{high} - \\sigma'_{low}}
        \\right)

    Parameters
    ----------
    a : float
        Power function strength parameter (dimensionless).
    b : float
        Power function strength parameter (dimensionless).
    sigma_prime_low : float
        Lower bound of the effective normal stress range (psf, kPa, or
        any consistent stress unit matching *p_a*).
    sigma_prime_high : float
        Upper bound of the effective normal stress range (same stress
        unit as *sigma_prime_low*).  Must be greater than
        *sigma_prime_low*.
    p_a : float, optional
        Atmospheric pressure (same stress unit).  Default is 2116 psf.

    Returns
    -------
    float
        Equivalent friction angle *phi'_EQ* (degrees).

    Raises
    ------
    ValueError
        If *sigma_prime_high* <= *sigma_prime_low* or if power function
        parameter requirements are violated.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Figure P-4, p. 44.
    """
    if sigma_prime_high <= sigma_prime_low:
        raise ValueError(
            "sigma_prime_high must be greater than sigma_prime_low."
        )
    s_high = power_function_shear_strength(a, b, sigma_prime_high, p_a)
    s_low = power_function_shear_strength(a, b, sigma_prime_low, p_a)
    slope = (s_high - s_low) / (sigma_prime_high - sigma_prime_low)
    return math.degrees(math.atan(slope))


def equivalent_cohesion(a: float, b: float,
                        sigma_prime_low: float,
                        sigma_prime_high: float,
                        p_a: float = 2116.0) -> float:
    """Equivalent Mohr-Coulomb cohesion intercept from power function
    parameters (Figure P-4).

    For a specified range of effective normal stresses, converts the
    nonlinear power function envelope into an equivalent linear
    Mohr-Coulomb envelope.  The equivalent cohesion intercept is
    determined by projecting the secant line back to the shear stress
    axis.

    .. math::
        c'_{EQ} = s(\\sigma'_{low}) - \\sigma'_{low}
                  \\cdot \\frac{s(\\sigma'_{high}) - s(\\sigma'_{low})}
                               {\\sigma'_{high} - \\sigma'_{low}}

    Parameters
    ----------
    a : float
        Power function strength parameter (dimensionless).
    b : float
        Power function strength parameter (dimensionless).
    sigma_prime_low : float
        Lower bound of the effective normal stress range (psf, kPa, or
        any consistent stress unit matching *p_a*).
    sigma_prime_high : float
        Upper bound of the effective normal stress range (same stress
        unit as *sigma_prime_low*).  Must be greater than
        *sigma_prime_low*.
    p_a : float, optional
        Atmospheric pressure (same stress unit).  Default is 2116 psf.

    Returns
    -------
    float
        Equivalent cohesion intercept *c'_EQ* (same stress unit as inputs).

    Raises
    ------
    ValueError
        If *sigma_prime_high* <= *sigma_prime_low* or if power function
        parameter requirements are violated.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Figure P-4, p. 44.
    """
    if sigma_prime_high <= sigma_prime_low:
        raise ValueError(
            "sigma_prime_high must be greater than sigma_prime_low."
        )
    s_high = power_function_shear_strength(a, b, sigma_prime_high, p_a)
    s_low = power_function_shear_strength(a, b, sigma_prime_low, p_a)
    slope = (s_high - s_low) / (sigma_prime_high - sigma_prime_low)
    return s_low - sigma_prime_low * slope


# ---------------------------------------------------------------------------
# Figure P-5 -- Conversion between normalized and dimensional parameters
# ---------------------------------------------------------------------------

def power_function_dimensional_to_normalized(
    a_dim: float, b: float, p_a: float = 2116.0
) -> float:
    """Convert a dimensional power function *A* parameter to the
    normalized *a* parameter (Figure P-5).

    Some software packages express the power function using dimensional
    parameters (s = A * sigma'^b) rather than normalized parameters
    (s = a * Pa * (sigma'/Pa)^b).  Setting the two forms equal and
    solving yields:

    .. math::
        a = \\frac{A \\cdot P_a^{\\,b}}{P_a} = A \\cdot P_a^{\\,b - 1}

    Parameters
    ----------
    a_dim : float
        Dimensional power function parameter *A* with units such that
        s = A * sigma'^b yields stress units.
    b : float
        Power function curvature parameter (dimensionless).  Same value
        in both dimensional and normalized forms.
    p_a : float, optional
        Atmospheric pressure (same stress unit as used in the
        dimensional equation).  Default is 2116 psf.

    Returns
    -------
    float
        Normalized (dimensionless) power function parameter *a*.

    Raises
    ------
    ValueError
        If *a_dim* is non-positive, *b* is non-positive, or *p_a* is
        non-positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Figure P-5, p. 45.
    """
    if a_dim <= 0.0:
        raise ValueError("a_dim must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    if p_a <= 0.0:
        raise ValueError("p_a must be positive.")
    return a_dim * p_a ** (b - 1.0)


def power_function_normalized_to_dimensional(
    a: float, b: float, p_a: float = 2116.0
) -> float:
    """Convert a normalized power function *a* parameter to the
    dimensional *A* parameter (Figure P-5).

    Inverse of :func:`power_function_dimensional_to_normalized`.
    Solves for the dimensional parameter *A* so that the equation
    s = A * sigma'^b produces the same result as the normalized form.

    .. math::
        A = \\frac{a}{P_a^{\\,b - 1}}

    Parameters
    ----------
    a : float
        Normalized (dimensionless) power function parameter.
    b : float
        Power function curvature parameter (dimensionless).
    p_a : float, optional
        Atmospheric pressure (stress unit).  Default is 2116 psf.

    Returns
    -------
    float
        Dimensional power function parameter *A*.

    Raises
    ------
    ValueError
        If *a* is non-positive, *b* is non-positive, or *p_a* is
        non-positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Figure P-5, p. 45.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    if p_a <= 0.0:
        raise ValueError("p_a must be positive.")
    return a / p_a ** (b - 1.0)


# ---------------------------------------------------------------------------
# Table P-6 -- Undrained strength ratio (USR) for low-plasticity silt
# ---------------------------------------------------------------------------

def usr_icu_triaxial(phi_prime_deg: float) -> float:
    """Undrained strength ratio from isotropically consolidated undrained
    (ICU) triaxial compression (Table P-6).

    Estimates the undrained strength ratio (USR = su / sigma'_vc) for
    nonplastic and low-plasticity silts, assuming zero excess pore
    pressure at failure (Af = 0) and c' = 0.

    .. math::
        USR = \\frac{\\sin \\phi'}{1 - \\sin \\phi'}

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress (drained) friction angle (degrees).  Must be
        in the range (0, 90).

    Returns
    -------
    float
        Undrained strength ratio (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is outside the range (0, 90).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-6, p. 54.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be in the range (0, 90).")
    sin_phi = math.sin(math.radians(phi_prime_deg))
    return sin_phi / (1.0 - sin_phi)


def usr_acu_triaxial(phi_prime_deg: float) -> float:
    """Undrained strength ratio from anisotropically consolidated undrained
    (ACU) triaxial compression (Table P-6).

    Estimates the undrained strength ratio (USR = su / sigma'_vc) for
    nonplastic and low-plasticity silts, assuming zero excess pore
    pressure at failure (Af = 0) and c' = 0.

    .. math::
        USR = \\sin \\phi'

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress (drained) friction angle (degrees).  Must be
        in the range (0, 90).

    Returns
    -------
    float
        Undrained strength ratio (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is outside the range (0, 90).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-6, p. 54.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be in the range (0, 90).")
    return math.sin(math.radians(phi_prime_deg))


def usr_dss(phi_prime_deg: float) -> float:
    """Undrained strength ratio from direct simple shear (DSS) test
    (Table P-6).

    Estimates the undrained strength ratio (USR = su / sigma'_vc) for
    nonplastic and low-plasticity silts, assuming zero excess pore
    pressure at failure (Af = 0) and c' = 0.

    .. math::
        USR = \\sin \\phi' - 0.5 \\sin 2\\phi'

    Parameters
    ----------
    phi_prime_deg : float
        Effective stress (drained) friction angle (degrees).  Must be
        in the range (0, 90).

    Returns
    -------
    float
        Undrained strength ratio (dimensionless).

    Raises
    ------
    ValueError
        If *phi_prime_deg* is outside the range (0, 90).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Prologue P, Table P-6, p. 54.
    """
    if phi_prime_deg <= 0.0 or phi_prime_deg >= 90.0:
        raise ValueError("phi_prime_deg must be in the range (0, 90).")
    phi_rad = math.radians(phi_prime_deg)
    return math.sin(phi_rad) - 0.5 * math.sin(2.0 * phi_rad)
