"""
UFC 3-220-10, Chapter 2: Field Exploration, Testing, and Instrumentation

Equations 2-1 through 2-3 covering undrained shear strength and sensitivity
from the field vane shear test (VST), and the corrected field vane shear
strength.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""

import math


def undrained_shear_strength_vane(T_max: float, D: float) -> float:
    """Undrained shear strength from the vane shear test (Equation 2-1).

    Computes the undrained shear strength from a field vane shear test
    (VST) using the maximum net torque measured during rotation of a
    "standard" rectangular vane with a height-to-diameter ratio (H/D)
    of 2.  At failure the vane cuts a cylindrical surface of soil and
    the peak torque is recorded.

    .. math::
        s_{u,fv} = \\frac{6 \\, T_{max}}{7 \\, \\pi \\, D^3}

    Parameters
    ----------
    T_max : float
        Maximum net torque measured during rotation of the vane
        (force * length, e.g., lb-ft, N-m, or kN-m).
    D : float
        Diameter of the vane (length, e.g., ft, m, or mm).  Must use
        units consistent with *T_max* so that the result has dimensions
        of stress.

    Returns
    -------
    float
        Undrained shear strength from the vane shear test, *s_u,fv*
        (stress, e.g., psf, Pa, or kPa).  Units depend on the
        consistent unit system used for the inputs.

    Raises
    ------
    ValueError
        If *D* is zero or negative, or if *T_max* is negative.

    Notes
    -----
    This equation is valid only for a standard rectangular vane with
    H/D = 2.  The remolded undrained shear strength (*s_ur,fv*) can be
    obtained by substituting the residual torque (*T_res*) in place of
    *T_max*.  *T_res* is the torque measured after five to ten rapid
    revolutions of the vane.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 2, Section 2-9.1.2, Equation 2-1, p. 100.
    """
    if D <= 0.0:
        raise ValueError("D must be positive.")
    if T_max < 0.0:
        raise ValueError("T_max must be non-negative.")
    return (6.0 * T_max) / (7.0 * math.pi * D ** 3)


def sensitivity_vane(s_u_fv: float, s_ur_fv: float) -> float:
    """Sensitivity of soil from the vane shear test (Equation 2-2).

    Computes the sensitivity of the soil as the ratio of the peak
    (intact) undrained shear strength to the remolded undrained shear
    strength, both obtained from the field vane shear test.  The peak
    strength is determined from the maximum torque, and the remolded
    strength is determined from the residual torque after five to ten
    rapid turns of the vane.

    .. math::
        S_{t,fv} = \\frac{s_{u,fv}}{s_{ur,fv}}

    Parameters
    ----------
    s_u_fv : float
        Undrained shear strength (peak/intact) from the vane shear
        test (stress, e.g., psf, Pa, or kPa).
    s_ur_fv : float
        Remolded undrained shear strength from the vane shear test
        (same stress unit as *s_u_fv*).

    Returns
    -------
    float
        Sensitivity from the vane shear test, *S_t,fv* (dimensionless).

    Raises
    ------
    ValueError
        If *s_ur_fv* is zero or negative, or if *s_u_fv* is negative.

    Notes
    -----
    Sensitivity values typically range from about 1 (insensitive) to
    greater than 16 (quick clays).  The remolded shear strength
    *s_ur_fv* is calculated using Equation 2-1 with the residual
    torque *T_res* in place of *T_max*.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 2, Section 2-9.1.2, Equation 2-2, p. 100.
    """
    if s_ur_fv <= 0.0:
        raise ValueError("s_ur_fv must be positive.")
    if s_u_fv < 0.0:
        raise ValueError("s_u_fv must be non-negative.")
    return s_u_fv / s_ur_fv


def corrected_undrained_shear_strength_vane(
    s_u_fv: float, mu_R: float
) -> float:
    """Corrected undrained shear strength from the vane shear test (Equation 2-3).

    The undrained shear strength obtained from the vane shear test tends
    to overpredict the shear strength mobilized in failures of
    embankments, shallow footings, and slopes constructed on soft clay.
    This function applies a vane correction factor to adjust the measured
    vane shear strength for use in design.  The correction factor is a
    function of the plasticity index (PI) of the soil tested; three
    different vane correction methods are given in the ASTM specification.

    .. math::
        s_{u,field} = s_{u,fv} \\times \\mu_R

    Parameters
    ----------
    s_u_fv : float
        Undrained shear strength from the vane shear test (stress,
        e.g., psf, Pa, or kPa), as computed from Equation 2-1.
    mu_R : float
        Vane correction factor (dimensionless).  This factor is a
        function of the plasticity index (PI) of the soil tested.
        Three methods for determining *mu_R* are provided in ASTM
        D2573.  Typical values are less than or equal to 1.0.

    Returns
    -------
    float
        Corrected undrained shear strength for field conditions,
        *s_u,field* (same stress unit as *s_u_fv*).

    Raises
    ------
    ValueError
        If *s_u_fv* is negative, or if *mu_R* is zero or negative.

    Notes
    -----
    The VST is most reliable for measuring the in situ strength of soft
    to medium clays (*s_u* < 2000 psf).  The correction is necessary
    because the VST overpredicts the mobilized shear strength in the
    field for embankments, footings, and slopes on soft clay.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 2, Section 2-9.1.2, Equation 2-3, p. 101.
    """
    if s_u_fv < 0.0:
        raise ValueError("s_u_fv must be non-negative.")
    if mu_R <= 0.0:
        raise ValueError("mu_R must be positive.")
    return s_u_fv * mu_R
