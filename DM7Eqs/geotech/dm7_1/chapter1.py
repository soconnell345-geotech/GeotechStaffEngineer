"""
UFC 3-220-10, Chapter 1: Identification and Classification of Soil and Rock

Equations 1-1 through 1-6 covering grain-size distribution coefficients for
coarse-grained soil classification, the AASHTO group index, point load
strength index for rock classification, and the expansion index for
expansive soils.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""

import math


def coefficient_of_uniformity(D60: float, D10: float) -> float:
    """Coefficient of uniformity for coarse-grained soils (Equation 1-1).

    Quantifies the spread of particle sizes in a coarse-grained soil.
    A higher value indicates a wider range of particle sizes.  Used together
    with the coefficient of curvature (Equation 1-2) to classify a soil as
    well-graded or poorly-graded per ASTM D2487 (USCS).

    Gravels are well-graded when Cu >= 4; sands when Cu >= 6.

    .. math::
        C_u = \\frac{D_{60}}{D_{10}}

    Parameters
    ----------
    D60 : float
        Particle-size diameter corresponding to 60% passing on the
        cumulative particle-size distribution curve (mm).
    D10 : float
        Particle-size diameter corresponding to 10% passing on the
        cumulative particle-size distribution curve (mm).

    Returns
    -------
    float
        Coefficient of uniformity, Cu (dimensionless).

    Raises
    ------
    ValueError
        If *D10* is zero or negative, or if *D60* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-1, p. 49.
    """
    if D10 <= 0.0:
        raise ValueError("D10 must be positive.")
    if D60 < 0.0:
        raise ValueError("D60 must be non-negative.")
    return D60 / D10


def coefficient_of_curvature(D60: float, D30: float, D10: float) -> float:
    """Coefficient of curvature for coarse-grained soils (Equation 1-2).

    Characterizes the shape of the grain-size distribution curve.  Used
    together with the coefficient of uniformity (Equation 1-1) to determine
    whether a coarse-grained soil is well-graded or poorly-graded per
    ASTM D2487 (USCS).

    A soil is well-graded when Cc is between 1.0 and 3.0, inclusive.

    .. math::
        C_c = \\frac{D_{30}^{2}}{D_{60} \\times D_{10}}

    Parameters
    ----------
    D60 : float
        Particle-size diameter corresponding to 60% passing on the
        cumulative particle-size distribution curve (mm).
    D30 : float
        Particle-size diameter corresponding to 30% passing on the
        cumulative particle-size distribution curve (mm).
    D10 : float
        Particle-size diameter corresponding to 10% passing on the
        cumulative particle-size distribution curve (mm).

    Returns
    -------
    float
        Coefficient of curvature, Cc (dimensionless).

    Raises
    ------
    ValueError
        If *D60* or *D10* is zero or negative, or if *D30* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-2, p. 50.
    """
    if D60 <= 0.0:
        raise ValueError("D60 must be positive.")
    if D10 <= 0.0:
        raise ValueError("D10 must be positive.")
    if D30 < 0.0:
        raise ValueError("D30 must be non-negative.")
    return (D30 ** 2) / (D60 * D10)


def aashto_group_index(F: float, LL: float, PI: float) -> float:
    """AASHTO group index for highway soil classification (Equation 1-3).

    Computes the group index used to modify the AASHTO soil classification
    (ASTM D3282).  The group index is appended in parentheses after the
    group symbol.  A higher group index generally indicates poorer
    subgrade performance.

    The result is reported as zero when the calculated value is negative,
    when the soil is non-plastic, or when the liquid limit cannot be
    determined.  For soils in the A-2-6 and A-2-7 subgroups, only the
    second term of the equation should be used; call this function with
    the *a2_subgroup* parameter set appropriately or compute the second
    term directly.

    .. math::
        GI = (F - 35)[0.2 + 0.005(LL - 40)] + [0.01(F - 15)(PI - 10)]

    Parameters
    ----------
    F : float
        Percentage passing the No. 200 (75 um) sieve, considering only
        particles passing a 3-inch sieve (%).
    LL : float
        Liquid limit of the soil (%).
    PI : float
        Plasticity index of the soil (%).

    Returns
    -------
    float
        Group index, GI (dimensionless).  Returned as 0.0 if the
        calculated value is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-3, p. 51.
    """
    term1 = (F - 35.0) * (0.2 + 0.005 * (LL - 40.0))
    term2 = 0.01 * (F - 15.0) * (PI - 10.0)
    gi = term1 + term2
    return max(gi, 0.0)


def aashto_group_index_a2(F: float, PI: float) -> float:
    """AASHTO group index for A-2-6 and A-2-7 subgroups (Equation 1-3, partial).

    For soils in the A-2-6 and A-2-7 subgroups, the group index is
    calculated using only the second term of Equation 1-3 (the term
    containing the plasticity index).

    .. math::
        GI = 0.01 (F - 15)(PI - 10)

    Parameters
    ----------
    F : float
        Percentage passing the No. 200 (75 um) sieve, considering only
        particles passing a 3-inch sieve (%).
    PI : float
        Plasticity index of the soil (%).

    Returns
    -------
    float
        Group index, GI (dimensionless).  Returned as 0.0 if the
        calculated value is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-3, p. 51-52.
    """
    gi = 0.01 * (F - 15.0) * (PI - 10.0)
    return max(gi, 0.0)


def point_load_strength_index(
    P: float, d: float, D_e: float
) -> float:
    """Point load strength index, Is(50) (Equations 1-4 and 1-5).

    Computes the size-corrected point load strength index for rock
    classification per ASTM D5731.  The size correction factor *F* is
    calculated internally from the equivalent core diameter using
    Equation 1-5, and then applied in Equation 1-4.

    .. math::
        I_{s(50)} = F \\cdot \\frac{P}{d^{2}}

    where

    .. math::
        F = \\sqrt{\\frac{D_e}{50}}

    Parameters
    ----------
    P : float
        Applied force at failure (kN or lbf).
    d : float
        Distance between the loaded points (mm or in).
    D_e : float
        Equivalent core diameter (mm or in).  Must use the same length
        unit as *d*.

    Returns
    -------
    float
        Point load strength index, Is(50) (stress units, e.g., MPa if
        *P* is in kN and *d* is in mm: kN/mm^2 = GPa; typically reported
        after unit conversion).

    Raises
    ------
    ValueError
        If *d* is zero or negative, or if *D_e* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equations 1-4 and 1-5, p. 66.
    """
    if d <= 0.0:
        raise ValueError("d must be positive.")
    if D_e < 0.0:
        raise ValueError("D_e must be non-negative.")
    F = size_correction_factor(D_e)
    return F * P / (d ** 2)


def size_correction_factor(D_e: float) -> float:
    """Size correction factor for point load strength (Equation 1-5).

    Corrects the point load strength measurement to the standard
    equivalent core diameter of 50 mm per ASTM D5731.

    .. math::
        F = \\sqrt{\\frac{D_e}{50}}

    Parameters
    ----------
    D_e : float
        Equivalent core diameter (mm).

    Returns
    -------
    float
        Size correction factor, F (dimensionless).

    Raises
    ------
    ValueError
        If *D_e* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-5, p. 66.
    """
    if D_e < 0.0:
        raise ValueError("D_e must be non-negative.")
    return math.sqrt(D_e / 50.0)


def expansion_index(delta_H: float, H_i: float) -> float:
    """Expansion index for characterizing expansive soils (Equation 1-6).

    Computes the expansion index from results of the Expansion Index test
    (ASTM D4829).  The soil specimen is compacted to 50% +/- 2% saturation,
    loaded to 1 psi confining pressure, submerged in distilled water, and
    the deformation is recorded.

    Classification of potential expansion based on EI (ASTM D4829):
        0-20    Very low
        21-50   Low
        51-90   Medium
        91-130  High
        >130    Very high

    .. math::
        EI = \\frac{\\Delta H}{H_i} \\times 1000

    Parameters
    ----------
    delta_H : float
        Change in height of the specimen during the test (in. or mm).
    H_i : float
        Initial height of the test specimen (in. or mm).  Must use the
        same length unit as *delta_H*.

    Returns
    -------
    float
        Expansion index, EI (dimensionless).

    Raises
    ------
    ValueError
        If *H_i* is zero or negative, or if *delta_H* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 1, Equation 1-6, p. 74.
    """
    if H_i <= 0.0:
        raise ValueError("H_i must be positive.")
    if delta_H < 0.0:
        raise ValueError("delta_H must be non-negative.")
    return (delta_H / H_i) * 1000.0
