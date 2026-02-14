"""
UFC 3-220-10, Chapter 4: Distribution of Stresses

Equations 4-1 through 4-12 covering total vertical stress, pore water pressure,
effective stress, horizontal stress, Boussinesq point load, line load, strip
load, rectangular loaded area, circular loaded area, pipe trench loads
(Marston theory), tunnel stability, and shaft pressures in clay.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""

import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Equations 4-1 through 4-5: Stress conditions at a point
# ---------------------------------------------------------------------------

def total_vertical_stress(
    layers: List[Tuple[float, float]],
) -> float:
    """Total vertical stress from overlying soil layers (Equation 4-1).

    Computes the total vertical (overburden) stress at a depth equal to
    the sum of all layer thicknesses.  Each layer is described by its
    thickness and total unit weight.  Ponded water can be included as a
    layer with unit weight equal to the unit weight of water.

    .. math::
        \\sigma_v = \\sum_{i=1}^{n} z_i \\cdot \\gamma_{t,i}

    Parameters
    ----------
    layers : list of (float, float)
        Each element is a tuple ``(z_i, gamma_t_i)`` where *z_i* is the
        thickness of layer *i* (ft or m) and *gamma_t_i* is the total
        unit weight of layer *i* (pcf or kN/m^3).  Units must be
        consistent across all layers.

    Returns
    -------
    float
        Total vertical stress at the base of the lowest layer
        (psf or kPa, depending on input units).

    Raises
    ------
    ValueError
        If any layer thickness or unit weight is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-1, p. 178.
    """
    sigma_v = 0.0
    for z_i, gamma_t_i in layers:
        if z_i < 0.0:
            raise ValueError("Layer thickness must be non-negative.")
        if gamma_t_i < 0.0:
            raise ValueError("Unit weight must be non-negative.")
        sigma_v += z_i * gamma_t_i
    return sigma_v


def pore_water_pressure(h_p: float, gamma_w: float) -> float:
    """Pore water pressure from pressure head (Equation 4-2).

    Computes the pore water pressure at a point from the pressure head.
    For hydrostatic conditions, the pressure head equals the depth below
    the groundwater table.  For flowing water conditions, the pressure
    head must first be determined from a seepage analysis.

    .. math::
        u = h_p \\cdot \\gamma_w

    Parameters
    ----------
    h_p : float
        Pressure head at the point of interest (ft or m).  For hydrostatic
        conditions this equals the depth below the groundwater table.
    gamma_w : float
        Unit weight of water (62.4 pcf or 9.81 kN/m^3).

    Returns
    -------
    float
        Pore water pressure (psf or kPa, depending on input units).

    Raises
    ------
    ValueError
        If *gamma_w* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-2, p. 178.
    """
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    return h_p * gamma_w


def effective_vertical_stress(sigma_v: float, u: float) -> float:
    """Effective vertical stress (Equation 4-3).

    Computes the effective vertical stress by subtracting the pore water
    pressure from the total vertical stress.

    .. math::
        \\sigma'_v = \\sigma_v - u

    Parameters
    ----------
    sigma_v : float
        Total vertical stress (psf or kPa).
    u : float
        Pore water pressure at the same point (psf or kPa, same units
        as *sigma_v*).

    Returns
    -------
    float
        Effective vertical stress (psf or kPa).

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-3, p. 180.
    """
    return sigma_v - u


def effective_horizontal_stress(K: float, sigma_v_eff: float) -> float:
    """Effective horizontal stress (Equation 4-4).

    Computes the effective horizontal stress as a proportion of the
    effective vertical stress using a lateral earth pressure coefficient.

    .. math::
        \\sigma'_h = K \\cdot \\sigma'_v

    Parameters
    ----------
    K : float
        Lateral earth pressure coefficient (dimensionless).  Common values
        include at-rest (K0), active (KA), and passive (KP) coefficients.
    sigma_v_eff : float
        Effective vertical stress (psf or kPa).

    Returns
    -------
    float
        Effective horizontal stress (psf or kPa).

    Raises
    ------
    ValueError
        If *K* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-4, p. 180.
    """
    if K < 0.0:
        raise ValueError("K must be non-negative.")
    return K * sigma_v_eff


def total_horizontal_stress(sigma_h_eff: float, u: float) -> float:
    """Total horizontal stress (Equation 4-5).

    Computes the total horizontal stress by adding the pore water pressure
    to the effective horizontal stress.

    .. math::
        \\sigma_h = \\sigma'_h + u

    Parameters
    ----------
    sigma_h_eff : float
        Effective horizontal stress (psf or kPa).
    u : float
        Pore water pressure at the same point (psf or kPa, same units as
        *sigma_h_eff*).

    Returns
    -------
    float
        Total horizontal stress (psf or kPa).

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-5, p. 180.
    """
    return sigma_h_eff + u


# ---------------------------------------------------------------------------
# Table 4-2: Elastic solutions for change in vertical stress
# (These are unnumbered equations in Table 4-2, presented between
#  Equations 4-5 and 4-6 on pages 184-185.)
# ---------------------------------------------------------------------------

def boussinesq_point_load(Q: float, x: float, y: float, z: float) -> float:
    """Change in vertical stress from a point load -- Boussinesq (Table 4-2).

    Computes the change in vertical stress at a point (x, y, z) within a
    semi-infinite, homogeneous, isotropic, elastic half-space caused by a
    vertical point load Q applied at the surface origin.

    .. math::
        \\Delta\\sigma_z = \\frac{3 Q z^3}{2 \\pi R^5}

    where :math:`R = \\sqrt{x^2 + y^2 + z^2}`.

    Parameters
    ----------
    Q : float
        Applied point load (lb or kN).
    x : float
        Horizontal distance in the x-direction from the point of load
        application (ft or m).
    y : float
        Horizontal distance in the y-direction from the point of load
        application (ft or m).
    z : float
        Depth below the surface at the point of interest (ft or m).
        Must be positive.

    Returns
    -------
    float
        Change in vertical stress (psf or kPa, depending on input units).

    Raises
    ------
    ValueError
        If *z* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 184.
    """
    if z <= 0.0:
        raise ValueError("z must be positive.")
    R = math.sqrt(x**2 + y**2 + z**2)
    if R == 0.0:
        raise ValueError("R must be non-zero.")
    return (3.0 * Q * z**3) / (2.0 * math.pi * R**5)


def boussinesq_line_load(P: float, x: float, z: float) -> float:
    """Change in vertical stress from a uniform line load -- Boussinesq
    (Table 4-2).

    Computes the change in vertical stress at a point (x, z) below a
    uniform line load of infinite length with intensity P (force per unit
    length) applied at the surface.  The line load is oriented along the
    y-axis; x is the horizontal offset perpendicular to the line load.

    .. math::
        \\Delta\\sigma_z = \\frac{2 P z^3}{\\pi R^4}

    where :math:`R = \\sqrt{x^2 + z^2}`.

    Parameters
    ----------
    P : float
        Line load intensity (lb/ft or kN/m).
    x : float
        Horizontal distance perpendicular to the line load (ft or m).
    z : float
        Depth below the surface (ft or m).  Must be positive.

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Raises
    ------
    ValueError
        If *z* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 184.
    """
    if z <= 0.0:
        raise ValueError("z must be positive.")
    R = math.sqrt(x**2 + z**2)
    if R == 0.0:
        raise ValueError("R must be non-zero.")
    return (2.0 * P * z**3) / (math.pi * R**4)


def boussinesq_strip_load(
    q0: float, alpha: float, gamma_angle: float
) -> float:
    """Change in vertical stress beneath a uniform strip load -- Boussinesq
    (Table 4-2).

    Computes the change in vertical stress at a point beneath a strip load
    of infinite length using angle parameters that describe the geometry
    of the point relative to the loaded strip.

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{\\pi}
        \\bigl[\\alpha + \\sin\\alpha \\cdot \\cos(\\alpha + 2\\gamma)\\bigr]

    Parameters
    ----------
    q0 : float
        Uniform applied stress on the strip (psf or kPa).
    alpha : float
        Angle subtended at the point of interest by the loaded strip
        width (radians).
    gamma_angle : float
        Angle from the vertical at the point of interest to the nearest
        edge of the strip (radians).

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Notes
    -----
    Both *alpha* and *gamma_angle* must be in radians.  See Figure 4-4 in
    UFC 3-220-10 for the geometric definitions of these angles.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 184.
    """
    return (q0 / math.pi) * (
        alpha + math.sin(alpha) * math.cos(alpha + 2.0 * gamma_angle)
    )


def boussinesq_rectangular_load(
    q0: float, x: float, y: float, z: float
) -> float:
    """Change in vertical stress beneath a corner of a uniformly loaded
    rectangular area -- Boussinesq (Table 4-2).

    Computes the change in vertical stress at depth *z* directly beneath
    a corner of a rectangular loaded area with dimensions *x* by *y*.
    To find the stress at a point not under a corner, superimpose
    contributions from multiple rectangles.

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{2\\pi}
        \\left[
            \\arctan\\!\\left(\\frac{x \\cdot y}{z \\cdot R_3}\\right)
            + \\frac{x \\cdot y \\cdot z}{R_3}
              \\left(\\frac{1}{R_1^2} + \\frac{1}{R_2^2}\\right)
        \\right]

    where :math:`R_1 = \\sqrt{y^2 + z^2}`,
    :math:`R_2 = \\sqrt{x^2 + z^2}`,
    :math:`R_3 = \\sqrt{x^2 + y^2 + z^2}`.

    Parameters
    ----------
    q0 : float
        Uniform applied stress on the rectangular area (psf or kPa).
    x : float
        Length of the rectangle in the x-direction, measured from the
        corner where the stress is computed (ft or m).  Must be positive.
    y : float
        Length of the rectangle in the y-direction, measured from the
        corner where the stress is computed (ft or m).  Must be positive.
    z : float
        Depth below the surface (ft or m).  Must be positive.

    Returns
    -------
    float
        Change in vertical stress beneath the corner (psf or kPa).

    Raises
    ------
    ValueError
        If *x*, *y*, or *z* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 184.
    """
    if x <= 0.0:
        raise ValueError("x must be positive.")
    if y <= 0.0:
        raise ValueError("y must be positive.")
    if z <= 0.0:
        raise ValueError("z must be positive.")
    R1 = math.sqrt(y**2 + z**2)
    R2 = math.sqrt(x**2 + z**2)
    R3 = math.sqrt(x**2 + y**2 + z**2)
    term1 = math.atan2(x * y, z * R3)
    term2 = (x * y * z / R3) * (1.0 / R1**2 + 1.0 / R2**2)
    return (q0 / (2.0 * math.pi)) * (term1 + term2)


def boussinesq_circular_load(q0: float, r: float, z: float) -> float:
    """Change in vertical stress beneath the center of a uniformly loaded
    circular area -- Boussinesq (Table 4-2).

    Computes the change in vertical stress at depth *z* directly beneath
    the center of a circular loaded area of radius *r*.

    .. math::
        \\Delta\\sigma_z = q_0
        \\left[1 - \\frac{1}{\\bigl(1 + (r/z)^2\\bigr)^{3/2}}\\right]

    Parameters
    ----------
    q0 : float
        Uniform applied stress on the circular area (psf or kPa).
    r : float
        Radius of the circular loaded area (ft or m).  Must be positive.
    z : float
        Depth below the center of the loaded area (ft or m).
        Must be positive.

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Raises
    ------
    ValueError
        If *r* or *z* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 185.
    """
    if r <= 0.0:
        raise ValueError("r must be positive.")
    if z <= 0.0:
        raise ValueError("z must be positive.")
    return q0 * (1.0 - 1.0 / (1.0 + (r / z) ** 2) ** 1.5)


def boussinesq_triangular_load(
    q0: float, x: float, a: float, b: float, alpha: float, beta: float
) -> float:
    """Change in vertical stress beneath a triangular strip load -- Boussinesq
    (Table 4-2).

    Computes the change in vertical stress at a point beneath a triangular
    (linearly varying) strip load.  The load varies linearly from zero at
    one edge to *q0* at the other edge over width *b*.  The parameter *a*
    is the horizontal distance from the point of interest to the far edge
    of the loaded strip.

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{\\pi}
        \\left[\\frac{x \\cdot \\alpha}{a}
              + \\frac{(a + b - x)}{b} \\cdot \\beta\\right]

    Parameters
    ----------
    q0 : float
        Maximum intensity of the triangular load (psf or kPa).
    x : float
        Horizontal distance from the point where load is zero to the
        point of interest, measured at the surface (ft or m).
    a : float
        Horizontal distance from the point of interest to the edge
        where the full load q0 acts (ft or m).  Must be positive.
    b : float
        Width of the triangular load (ft or m).  Must be positive.
    alpha : float
        Angle subtended at the point of interest by the distance *a*
        (radians).
    beta : float
        Angle subtended at the point of interest by the loaded width *b*
        (radians).

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Notes
    -----
    Both *alpha* and *beta* must be in radians.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 185.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if b <= 0.0:
        raise ValueError("b must be positive.")
    return (q0 / math.pi) * (
        (x * alpha / a) + ((a + b - x) / b) * beta
    )


def boussinesq_slope_load(
    q0: float, x: float, z: float, a: float, beta: float
) -> float:
    """Change in vertical stress beneath a slope (embankment ramp) load --
    Boussinesq (Table 4-2).

    Computes the change in vertical stress beneath a load that increases
    linearly from zero to *q0* over a horizontal distance *a* (a ramp or
    slope load).

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{\\pi \\cdot a}
        \\left(x \\cdot \\beta + z\\right)

    Parameters
    ----------
    q0 : float
        Maximum load intensity at the top of the slope (psf or kPa).
    x : float
        Horizontal distance from the toe (zero-load edge) to the point
        of interest, measured at the surface (ft or m).
    z : float
        Depth below the surface at the point of interest (ft or m).
    a : float
        Horizontal distance over which the load increases from zero to
        *q0* (ft or m).  Must be positive.
    beta : float
        Angle at the point of interest subtended by the slope load
        geometry (radians).

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Notes
    -----
    *beta* must be in radians.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 185.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    return (q0 / (math.pi * a)) * (x * beta + z)


def boussinesq_terrace_load(
    q0: float, x: float, a: float, alpha: float, beta: float
) -> float:
    """Change in vertical stress beneath a terrace (trapezoidal) load --
    Boussinesq (Table 4-2).

    Computes the change in vertical stress beneath a terrace load
    consisting of a slope portion (width *a*) and a level portion.  The
    load rises linearly from zero over the slope width *a* to intensity
    *q0*, then continues at *q0*.

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{\\pi \\cdot a}
        \\left(a \\cdot \\beta + x \\cdot \\alpha\\right)

    Parameters
    ----------
    q0 : float
        Uniform load intensity on the level portion (psf or kPa).
    x : float
        Horizontal distance from the toe of the slope to the point
        of interest, measured at the surface (ft or m).
    a : float
        Horizontal distance of the sloping portion of the load (ft or m).
        Must be positive.
    alpha : float
        Angle at the point of interest subtended by the level load
        portion (radians).
    beta : float
        Angle at the point of interest subtended by the sloping load
        portion (radians).

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    Notes
    -----
    Both *alpha* and *beta* must be in radians.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 185.
    """
    if a <= 0.0:
        raise ValueError("a must be positive.")
    return (q0 / (math.pi * a)) * (a * beta + x * alpha)


def boussinesq_semi_infinite_load(
    q0: float, x: float, z: float, R: float, beta: float
) -> float:
    """Change in vertical stress beneath a semi-infinite uniform load --
    Boussinesq (Table 4-2).

    Computes the change in vertical stress at a point (x, z) due to a
    uniform load of intensity *q0* extending from x = 0 to x = +infinity
    (a semi-infinite uniform surcharge).

    .. math::
        \\Delta\\sigma_z = \\frac{q_0}{\\pi}
        \\left(\\beta + \\frac{x z}{R^2}\\right)

    Parameters
    ----------
    q0 : float
        Uniform load intensity (psf or kPa).
    x : float
        Horizontal distance from the edge of the load to the point of
        interest (ft or m).
    z : float
        Depth below the surface (ft or m).
    R : float
        Radial distance from the edge of the load to the point of
        interest, :math:`R = \\sqrt{x^2 + z^2}` (ft or m).  Must be
        positive.
    beta : float
        Angle at the point of interest measured from the vertical to the
        line connecting the edge of the load and the point (radians).

    Returns
    -------
    float
        Change in vertical stress (psf or kPa).

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Table 4-2, p. 185.
    """
    if R <= 0.0:
        raise ValueError("R must be positive.")
    return (q0 / math.pi) * (beta + (x * z) / R**2)


# ---------------------------------------------------------------------------
# Equations 4-6 through 4-9: Shallow pipes and conduits
# ---------------------------------------------------------------------------

def rigid_pipe_trench_load(
    C_d: float, gamma_t: float, B_d: float
) -> float:
    """Dead load per unit length on a rigid pipe in a trench -- Marston
    theory (Equation 4-6).

    Estimates the load per unit length of rigid pipe (precast concrete,
    cast-in-place concrete, or cast iron) in a trench using the approach
    suggested by Marston and Anderson (1913) and Spangler (1948).

    .. math::
        W_d = C_d \\cdot \\gamma_t \\cdot B_d^2

    Parameters
    ----------
    C_d : float
        Trench load coefficient (dimensionless).  Calculated using
        :func:`trench_load_coefficient` (Equation 4-7).
    gamma_t : float
        Total unit weight of the trench backfill soil (pcf or kN/m^3).
    B_d : float
        Width of the trench (ft or m).  Must be positive.

    Returns
    -------
    float
        Dead load per unit length of pipe (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *B_d* or *gamma_t* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-6, p. 194.
    """
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    if B_d <= 0.0:
        raise ValueError("B_d must be positive.")
    return C_d * gamma_t * B_d**2


def trench_load_coefficient(
    H: float, B_d: float, K: float, mu_prime: float
) -> float:
    """Trench load coefficient for rigid pipe (Equation 4-7).

    Computes the Marston load coefficient *C_d* used in the calculation
    of dead load on a rigid pipe in a trench (Equation 4-6).

    .. math::
        C_d = \\frac{1 - e^{-2 K \\mu' H / B_d}}{2 K \\mu'}

    Parameters
    ----------
    H : float
        Depth of the trench above the top of the pipe (ft or m).
        Must be positive.
    B_d : float
        Width of the trench (ft or m).  Must be positive.
    K : float
        Lateral earth pressure coefficient for the trench fill
        (dimensionless).  See Table 4-3 in UFC 3-220-10 for
        recommended values.
    mu_prime : float
        Coefficient of friction between the trench fill and the
        trench walls (dimensionless).  See Table 4-3 in UFC 3-220-10
        for recommended values.

    Returns
    -------
    float
        Trench load coefficient *C_d* (dimensionless).

    Raises
    ------
    ValueError
        If *H*, *B_d*, *K*, or *mu_prime* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-7, p. 194.
    """
    if H <= 0.0:
        raise ValueError("H must be positive.")
    if B_d <= 0.0:
        raise ValueError("B_d must be positive.")
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if mu_prime <= 0.0:
        raise ValueError("mu_prime must be positive.")
    exponent = -2.0 * K * mu_prime * H / B_d
    return (1.0 - math.exp(exponent)) / (2.0 * K * mu_prime)


def flexible_pipe_trench_load(
    C_d: float, gamma_t: float, B_d: float, D: float
) -> float:
    """Dead load per unit length on a flexible pipe in a trench -- Marston
    theory (Equation 4-8).

    Estimates the load per unit length on a very flexible pipe (corrugated
    metal, plastic, or thin-wall smooth steel) in a trench.  This method
    assumes that the pipe stiffness and soil stiffness are equal.

    .. math::
        W_c = C_d \\cdot \\gamma_t \\cdot B_d \\cdot D

    Parameters
    ----------
    C_d : float
        Trench load coefficient for rigid pipe (dimensionless), computed
        from Equation 4-7.
    gamma_t : float
        Total unit weight of the trench backfill (pcf or kN/m^3).
    B_d : float
        Width of the trench (ft or m).  Must be positive.
    D : float
        Outer diameter of the pipe (ft or m).  Must be positive.

    Returns
    -------
    float
        Dead load per unit length of flexible pipe (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma_t*, *B_d*, or *D* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-8, p. 196.
    """
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    if B_d <= 0.0:
        raise ValueError("B_d must be positive.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    return C_d * gamma_t * B_d * D


def flexible_pipe_prism_load(
    gamma_t: float, H: float, D: float
) -> float:
    """Prism load on a buried flexible pipe (Equation 4-9).

    Computes the prism load, which is the total weight of soil directly
    above the pipe.  This is a more conservative approach than Equation
    4-8 for determining the dead load on a buried flexible pipe.

    .. math::
        W_p = \\gamma_t \\cdot H \\cdot D

    Parameters
    ----------
    gamma_t : float
        Total unit weight of the backfill soil (pcf or kN/m^3).
    H : float
        Depth of soil cover above the pipe (ft or m).  Must be positive.
    D : float
        Outer diameter of the pipe (ft or m).  Must be positive.

    Returns
    -------
    float
        Prism load per unit length of pipe (lb/ft or kN/m).

    Raises
    ------
    ValueError
        If *gamma_t*, *H*, or *D* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-9, p. 197.
    """
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    if H <= 0.0:
        raise ValueError("H must be positive.")
    if D <= 0.0:
        raise ValueError("D must be positive.")
    return gamma_t * H * D


# ---------------------------------------------------------------------------
# Equations 4-10 through 4-12: Tunnels and shafts
# ---------------------------------------------------------------------------

def undrained_stability_factor(
    sigma_v_eff: float, sigma_t: float, s_u: float
) -> float:
    """Undrained stability factor for tunnels in fine-grained soils
    (Equation 4-10).

    Computes the undrained stability factor (N_crit) used to evaluate
    ground behavior for tunnels in fine-grained soils and silty sands
    above the water table.  Higher values indicate less stable ground
    conditions (see Table 4-9 in UFC 3-220-10).

    .. math::
        N_{crit} = \\frac{\\sigma'_v - \\sigma_t}{s_u}

    Parameters
    ----------
    sigma_v_eff : float
        Effective overburden pressure at the tunnel centerline
        (psf or kPa).
    sigma_t : float
        Interior applied pressure from compressed air or breasting
        (psf or kPa, same units as *sigma_v_eff*).  Use 0 if no
        internal pressure is applied.
    s_u : float
        Undrained shear strength of the soil (psf or kPa).
        Must be positive.

    Returns
    -------
    float
        Undrained stability factor N_crit (dimensionless).

    Raises
    ------
    ValueError
        If *s_u* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-10, p. 203.
    """
    if s_u <= 0.0:
        raise ValueError("s_u must be positive.")
    return (sigma_v_eff - sigma_t) / s_u


def shaft_critical_depth_clay(s_u: float, gamma_t: float) -> float:
    """Critical unsupported depth for a vertical shaft in clay
    (Equation 4-11).

    Computes the maximum depth from the ground surface to which no
    lateral support is needed for a vertical shaft in clay.  Below this
    depth, lateral support must be provided.

    .. math::
        z_{crit} = \\frac{2 s_u}{\\gamma_t}

    Parameters
    ----------
    s_u : float
        Undrained shear strength of the clay (psf or kPa).
    gamma_t : float
        Total unit weight of the clay soil (pcf or kN/m^3).
        Must be positive.

    Returns
    -------
    float
        Critical unsupported depth (ft or m).

    Raises
    ------
    ValueError
        If *s_u* is negative or *gamma_t* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-11, p. 209.
    """
    if s_u < 0.0:
        raise ValueError("s_u must be non-negative.")
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    return (2.0 * s_u) / gamma_t


def shaft_horizontal_pressure_clay(
    gamma_eff: float, z: float, s_u: float
) -> float:
    """Ultimate horizontal pressure on a shaft lining in soft clay
    (Equation 4-12).

    Estimates the ultimate horizontal pressure on a vertical shaft lining
    in soft clay at depths greater than the critical depth (Equation
    4-11).  This pressure is expected to develop after several months of
    unsupported excavation.

    .. math::
        \\sigma_h = \\gamma' \\cdot z - s_u

    Parameters
    ----------
    gamma_eff : float
        Effective (buoyant) unit weight of the soil (pcf or kN/m^3).
        Must be positive.
    z : float
        Depth below the ground surface (ft or m).  Must be positive.
    s_u : float
        Undrained shear strength of the soft clay (psf or kPa).

    Returns
    -------
    float
        Ultimate horizontal pressure on the shaft lining (psf or kPa).
        May be negative if the shear strength exceeds the effective
        overburden, indicating no net pressure on the lining.

    Raises
    ------
    ValueError
        If *gamma_eff* or *z* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 4, Equation 4-12, p. 210.
    """
    if gamma_eff <= 0.0:
        raise ValueError("gamma_eff must be positive.")
    if z <= 0.0:
        raise ValueError("z must be positive.")
    return gamma_eff * z - s_u
