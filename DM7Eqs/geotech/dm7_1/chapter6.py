"""
UFC 3-220-10, Chapter 6: Seepage and Drainage

Equations 6-1 through 6-14 and 6-16 covering total hydraulic head, hydraulic
gradient, Darcy's Law (volumetric flow rate and flow per unit width), discharge
velocity, seepage velocity, the LaPlace equation (2-D steady-state flow),
anisotropic transformation factor, flow net calculations (volumetric flow,
head drop per equipotential, pore water pressure), hydraulic conductivity
correlations (effective grain size, Kozeny-Carman, void-ratio adjustment),
and geotextile permeability factor of safety.

Note: Equation 6-15 does not exist in this chapter; the label 6-15 is used
for a figure and a table, not an equation.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""

import math
from typing import List, Tuple


def total_hydraulic_head(u: float, gamma_w: float, z: float) -> float:
    """Total hydraulic head at a point (Equation 6-1).

    The total hydraulic head is composed of the pressure head and the
    elevation head.  The velocity head is neglected as is standard practice
    in geologic media.

    .. math::
        h_t = \\frac{u}{\\gamma_w} + z

    Parameters
    ----------
    u : float
        Water pressure at the point of interest (force/length^2, e.g.,
        psf or kPa).
    gamma_w : float
        Unit weight of water (force/length^3, e.g., 62.4 pcf or
        9.81 kN/m^3).
    z : float
        Elevation of the point of interest above the elevation datum
        (length, e.g., ft or m).

    Returns
    -------
    float
        Total hydraulic head, h_t (length, e.g., ft or m).

    Raises
    ------
    ValueError
        If *gamma_w* is zero or negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-1, p. 284.
    """
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    return u / gamma_w + z


def hydraulic_gradient(h_L: float, L: float) -> float:
    """Hydraulic gradient across a flow region (Equation 6-2).

    The hydraulic gradient is the ratio of the differential total head (head
    loss) to the length over which the head loss occurs.

    .. math::
        i = \\frac{h_L}{L}

    Parameters
    ----------
    h_L : float
        Differential total head or head loss across the flow region
        (length, e.g., ft or m).
    L : float
        Length of the flow path over which the head loss occurs
        (length, e.g., ft or m).

    Returns
    -------
    float
        Hydraulic gradient, i (dimensionless).

    Raises
    ------
    ValueError
        If *L* is zero or negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-2, p. 285.
    """
    if L <= 0.0:
        raise ValueError("L must be positive.")
    return h_L / L


def darcy_flow_rate(k: float, i: float, A: float) -> float:
    """Volumetric flow rate through soil via Darcy's Law (Equation 6-3).

    Darcy's Law governs one-dimensional flow through soil under laminar
    conditions.

    .. math::
        q = k \\cdot i \\cdot A

    Parameters
    ----------
    k : float
        Hydraulic conductivity of the soil (length/time, e.g., cm/s or
        ft/day).
    i : float
        Hydraulic gradient across the flow region (dimensionless).
    A : float
        Cross-sectional area of the flow region perpendicular to the flow
        direction (length^2, e.g., ft^2 or m^2).

    Returns
    -------
    float
        Volumetric flow rate, q (length^3/time, e.g., ft^3/s or m^3/s).

    Raises
    ------
    ValueError
        If *k* or *A* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-3, p. 286.
    """
    if k < 0.0:
        raise ValueError("k must be non-negative.")
    if A < 0.0:
        raise ValueError("A must be non-negative.")
    return k * i * A


def darcy_flow_rate_per_unit_width(k: float, i: float, y: float) -> float:
    """Flow rate per unit width through a constant-height region (Equation 6-4).

    When the flow region has a constant height and an extended width
    perpendicular to the cross-section, the flow area can be defined by the
    height times a unit width.  This equation gives the flow rate per unit
    length of the model.

    .. math::
        q = k \\cdot i \\cdot y

    Parameters
    ----------
    k : float
        Hydraulic conductivity of the soil (length/time, e.g., cm/s or
        ft/day).
    i : float
        Hydraulic gradient across the flow region (dimensionless).
    y : float
        Height of the flow region (length, e.g., ft or m).

    Returns
    -------
    float
        Flow rate per unit length of the model (length^2/time, e.g.,
        ft^2/s or m^2/s).

    Raises
    ------
    ValueError
        If *k* or *y* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-4, p. 286.
    """
    if k < 0.0:
        raise ValueError("k must be non-negative.")
    if y < 0.0:
        raise ValueError("y must be non-negative.")
    return k * i * y


def discharge_velocity(k: float, i: float) -> float:
    """Discharge velocity (Darcy velocity) through soil (Equation 6-5).

    The discharge velocity is the volumetric flow rate divided by the total
    cross-sectional area.  It is not the true particle velocity because it
    is based on the total area, not just the pore space.

    .. math::
        v_d = k \\cdot i

    Parameters
    ----------
    k : float
        Hydraulic conductivity of the soil (length/time, e.g., cm/s or
        ft/day).
    i : float
        Hydraulic gradient across the flow region (dimensionless).

    Returns
    -------
    float
        Discharge velocity, v_d (length/time, e.g., cm/s or ft/day).

    Raises
    ------
    ValueError
        If *k* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-5, p. 286.
    """
    if k < 0.0:
        raise ValueError("k must be non-negative.")
    return k * i


def seepage_velocity(v_d: float, n: float = None, e: float = None) -> float:
    """Seepage (true particle) velocity through soil pores (Equation 6-6).

    The seepage velocity is the actual velocity at which a water particle
    moves through the soil, accounting for the fact that flow only occurs
    through the pore space.  Either porosity or void ratio must be provided.

    .. math::
        v_s = \\frac{v_d}{n} = v_d \\cdot \\frac{1 + e}{e}

    Parameters
    ----------
    v_d : float
        Discharge velocity (length/time, e.g., cm/s or ft/day).
    n : float, optional
        Porosity of the soil (dimensionless, 0 < n < 1).  Provide either
        *n* or *e*, not both.
    e : float, optional
        Void ratio of the soil (dimensionless, e > 0).  Provide either
        *n* or *e*, not both.

    Returns
    -------
    float
        Seepage velocity, v_s (length/time, e.g., cm/s or ft/day).

    Raises
    ------
    ValueError
        If neither *n* nor *e* is provided, if both are provided, or if
        values are outside physically meaningful ranges.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-6, p. 286.
    """
    if n is not None and e is not None:
        raise ValueError("Provide either n (porosity) or e (void ratio), not both.")
    if n is not None:
        if n <= 0.0 or n >= 1.0:
            raise ValueError("Porosity n must be between 0 and 1 (exclusive).")
        return v_d / n
    if e is not None:
        if e <= 0.0:
            raise ValueError("Void ratio e must be positive.")
        return v_d * (1.0 + e) / e
    raise ValueError("Either n (porosity) or e (void ratio) must be provided.")


def laplace_check_2d(
    d2h_dx2: float, d2h_dy2: float, tolerance: float = 1.0e-6
) -> bool:
    """Verify that the 2-D LaPlace equation is satisfied (Equation 6-7).

    The governing equation for steady-state, two-dimensional seepage flow
    in an isotropic soil is the LaPlace equation.  This function checks
    whether the sum of the second partial derivatives of total head with
    respect to x and y is approximately zero.

    .. math::
        \\frac{\\partial^2 h}{\\partial x^2}
        + \\frac{\\partial^2 h}{\\partial y^2} = 0

    Parameters
    ----------
    d2h_dx2 : float
        Second partial derivative of total hydraulic head with respect to
        x (1/length, e.g., 1/ft or 1/m).
    d2h_dy2 : float
        Second partial derivative of total hydraulic head with respect to
        y (1/length, e.g., 1/ft or 1/m).
    tolerance : float, optional
        Acceptable absolute residual for the sum of the second derivatives
        (default 1.0e-6).

    Returns
    -------
    bool
        True if the LaPlace equation is satisfied within the specified
        tolerance; False otherwise.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-7, p. 287.
    """
    return abs(d2h_dx2 + d2h_dy2) <= tolerance


def anisotropic_transformation_factor(k_max: float, k_min: float) -> float:
    """Isotropic transformation factor for anisotropic flow nets (Equation 6-8).

    Transforms an anisotropic flow region into an equivalent isotropic
    region.  Dimensions parallel to the direction of maximum hydraulic
    conductivity are divided by this factor.

    .. math::
        a = \\sqrt{\\frac{k_{max}}{k_{min}}}

    Parameters
    ----------
    k_max : float
        Maximum hydraulic conductivity in the anisotropic soil
        (length/time, e.g., cm/s).
    k_min : float
        Minimum hydraulic conductivity in the anisotropic soil
        (length/time, e.g., cm/s).

    Returns
    -------
    float
        Isotropic transformation factor, a (dimensionless).

    Raises
    ------
    ValueError
        If *k_max* or *k_min* is not positive, or if *k_max* < *k_min*.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-8, p. 289.
    """
    if k_max <= 0.0:
        raise ValueError("k_max must be positive.")
    if k_min <= 0.0:
        raise ValueError("k_min must be positive.")
    if k_max < k_min:
        raise ValueError("k_max must be greater than or equal to k_min.")
    return math.sqrt(k_max / k_min)


def flow_net_flow_rate(
    k: float, h_L: float, N_f: float, N_d: float, W: float = 1.0
) -> float:
    """Volumetric flow rate through a flow net section (Equation 6-9).

    Calculates the total seepage flow through a flow net.  For transformed
    (anisotropic) flow nets, use k = sqrt(k_max * k_min).

    .. math::
        q = k \\cdot h_L \\cdot \\frac{N_f}{N_d} \\cdot W

    Parameters
    ----------
    k : float
        Isotropic hydraulic conductivity of the soil (length/time, e.g.,
        cm/s or ft/day).  For transformed flow nets, use
        sqrt(k_max * k_min).
    h_L : float
        Total differential head or head loss across the flow net
        (length, e.g., ft or m).
    N_f : float
        Number of flow channels in the flow net (dimensionless).
    N_d : float
        Total number of equipotential (head) drops in the flow net
        (dimensionless).
    W : float, optional
        Width of the system perpendicular to the cross-section (length,
        e.g., ft or m).  Default is 1.0 for unit width analysis.

    Returns
    -------
    float
        Volumetric flow rate, q (length^3/time, e.g., ft^3/s or m^3/s).

    Raises
    ------
    ValueError
        If *k*, *N_f*, *N_d*, or *W* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-9, p. 290.
    """
    if k <= 0.0:
        raise ValueError("k must be positive.")
    if N_f <= 0.0:
        raise ValueError("N_f must be positive.")
    if N_d <= 0.0:
        raise ValueError("N_d must be positive.")
    if W <= 0.0:
        raise ValueError("W must be positive.")
    return k * h_L * (N_f / N_d) * W


def head_loss_per_drop(h_L: float, N_d: float) -> float:
    """Head loss associated with one equipotential drop (Equation 6-10).

    The flow net divides the total head loss into N_d equal head drops.
    This function calculates the head loss per single drop.

    .. math::
        \\Delta h_L = \\frac{h_L}{N_d}

    Parameters
    ----------
    h_L : float
        Total differential head or head loss across the flow net
        (length, e.g., ft or m).
    N_d : float
        Total number of equipotential (head) drops in the flow net
        (dimensionless).

    Returns
    -------
    float
        Head loss per equipotential drop, delta_h_L (length, e.g.,
        ft or m).

    Raises
    ------
    ValueError
        If *N_d* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-10, p. 291.
    """
    if N_d <= 0.0:
        raise ValueError("N_d must be positive.")
    return h_L / N_d


def pore_water_pressure(h_t: float, h_z: float, gamma_w: float) -> float:
    """Pore water pressure from flow net data (Equation 6-11).

    Calculates the pore water pressure at any point within a flow net
    from the total head and elevation at that point.

    .. math::
        u = (h_t - h_z) \\cdot \\gamma_w = h_p \\cdot \\gamma_w

    Parameters
    ----------
    h_t : float
        Total hydraulic head at the point of interest (length, e.g.,
        ft or m).
    h_z : float
        Elevation head at the point of interest (length, e.g., ft or m).
    gamma_w : float
        Unit weight of water (force/length^3, e.g., 62.4 pcf or
        9.81 kN/m^3).

    Returns
    -------
    float
        Pore water pressure, u (force/length^2, e.g., psf or kPa).

    Raises
    ------
    ValueError
        If *gamma_w* is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-11, p. 291.
    """
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    return (h_t - h_z) * gamma_w


def hydraulic_conductivity_effective_grain_size(
    beta_alpha: float, D_alpha: float, x: float = 2.0
) -> float:
    """Hydraulic conductivity from effective grain size (Equation 6-12).

    Empirical correlation relating hydraulic conductivity to an effective
    grain size.  This general form encompasses the Hazen, Kenney, Slichter,
    Chapuis, Carrier, and Sherard correlations summarized in Table 6-5.

    .. math::
        k = \\beta_{\\alpha} \\cdot D_{\\alpha}^{x}

    Parameters
    ----------
    beta_alpha : float
        Empirical or semi-empirical coefficient (cm/sec/mm^x).  Value
        depends on the correlation used; see Table 6-5 of UFC 3-220-10.
        Common values include 1.0 for Hazen's simplified form.
    D_alpha : float
        Effective grain size (mm).  The subscript alpha refers to the
        percent passing on the grain-size distribution (typically 5, 10,
        15, or 20).
    x : float, optional
        Exponent on the effective grain size (dimensionless).
        Theoretically equal to 2; empirically slightly above 2.
        Default is 2.0.

    Returns
    -------
    float
        Hydraulic conductivity, k (cm/sec).

    Raises
    ------
    ValueError
        If *beta_alpha* or *D_alpha* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-12, p. 299.
    Kenney, T. C. et al. (1984).
    """
    if beta_alpha < 0.0:
        raise ValueError("beta_alpha must be non-negative.")
    if D_alpha < 0.0:
        raise ValueError("D_alpha must be non-negative.")
    return beta_alpha * (D_alpha ** x)


def kozeny_carman_hydraulic_conductivity(
    fractions: List[Tuple[float, float, float]],
    S: float,
    e: float,
) -> float:
    """Hydraulic conductivity from the Kozeny-Carman equation (Equation 6-13).

    Accounts for the effect of the entire grain-size distribution and
    particle shape on the hydraulic conductivity of a soil.

    .. math::
        k = 1.99 \\times 10^{-4}
            \\cdot \\frac{1}{S^2}
            \\cdot \\frac{e^3}{1 + e}
            \\cdot \\left[
                \\sum_i \\frac{f_i}{\\left(
                    D_{li}^{0.404} \\cdot D_{si}^{0.596}
                \\right)^2}
            \\right]^{-1}

    The constant 1.99 x 10^-4 incorporates unit conversions when particle
    sizes are in mm and k is returned in cm/sec.

    Parameters
    ----------
    fractions : list of (f_i, D_li, D_si) tuples
        Each tuple contains:
        - f_i : float -- fraction of particles by mass between two
          adjacent sieve sizes (as a decimal, e.g., 0.25 for 25%).
          All f_i values should sum to 1.0 (100%).
        - D_li : float -- particle size of the coarser sieve (mm).
        - D_si : float -- particle size of the finer sieve (mm).
    S : float
        Surface area factor (dimensionless).  Ranges from 6 for spherical
        particles to 8.5 for angular particles.
    e : float
        Void ratio (dimensionless).

    Returns
    -------
    float
        Hydraulic conductivity, k (cm/sec).

    Raises
    ------
    ValueError
        If *S* is not positive, if *e* is not positive, or if any sieve
        size is not positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-13, p. 300.
    Carrier, W. D. (2003).
    """
    if S <= 0.0:
        raise ValueError("S must be positive.")
    if e <= 0.0:
        raise ValueError("e must be positive.")

    summation = 0.0
    for f_i, D_li, D_si in fractions:
        if D_li <= 0.0 or D_si <= 0.0:
            raise ValueError(
                "Sieve sizes D_li and D_si must be positive."
            )
        if f_i < 0.0:
            raise ValueError("Fraction f_i must be non-negative.")
        D_eff = (D_li ** 0.404) * (D_si ** 0.596)
        summation += f_i / (D_eff ** 2)

    if summation <= 0.0:
        raise ValueError(
            "The summation term is zero; at least one fraction must be "
            "non-zero with valid sieve sizes."
        )

    k = (1.99e-4) * (1.0 / (S ** 2)) * (e ** 3 / (1.0 + e)) * (1.0 / summation)
    return k


def kozeny_void_ratio_conductivity_ratio(
    e1: float, e2: float
) -> float:
    """Ratio of hydraulic conductivities for two void ratios (Equation 6-14).

    For a given soil, the hydraulic conductivity changes with void ratio.
    This equation provides the ratio k1/k2 corresponding to void ratios
    e1 and e2 respectively.

    .. math::
        \\frac{k_1}{k_2} = \\frac{e_1^3}{e_2^3}
        \\cdot \\frac{1 + e_2}{1 + e_1}

    Parameters
    ----------
    e1 : float
        Void ratio corresponding to hydraulic conductivity k1
        (dimensionless).
    e2 : float
        Void ratio corresponding to hydraulic conductivity k2
        (dimensionless).

    Returns
    -------
    float
        Ratio k1/k2 (dimensionless).

    Raises
    ------
    ValueError
        If *e1* or *e2* is not positive.

    Notes
    -----
    To obtain a specific hydraulic conductivity, multiply a known k value
    by this ratio.  For example, if k2 is known at void ratio e2, then
    k1 = k2 * kozeny_void_ratio_conductivity_ratio(e1, e2).

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-14, p. 300.
    Kozeny, J. (1927).
    """
    if e1 <= 0.0:
        raise ValueError("e1 must be positive.")
    if e2 <= 0.0:
        raise ValueError("e2 must be positive.")
    return (e1 ** 3 / e2 ** 3) * ((1.0 + e2) / (1.0 + e1))


def geotextile_permeability_factor_of_safety(
    k_g: float = None,
    k_s: float = None,
    psi_g: float = None,
    t_g: float = None,
) -> float:
    """Factor of safety for geotextile permeability (Equation 6-16).

    Evaluates whether a geotextile filter has adequate hydraulic
    conductivity relative to the base soil.  The factor of safety can be
    computed from either the geotextile and soil hydraulic conductivities
    directly, or from the geotextile permittivity and thickness.

    .. math::
        FS_g = \\frac{k_g}{k_s}
             = \\frac{\\psi_g \\cdot t_g}{k_s}

    At least *k_g* and *k_s*, or *psi_g*, *t_g*, and *k_s* must be
    provided.

    Parameters
    ----------
    k_g : float, optional
        Hydraulic conductivity of the geotextile across the plane of
        the fabric (length/time, e.g., cm/s).  If not provided, it is
        computed as psi_g * t_g.
    k_s : float, optional
        Hydraulic conductivity of the base soil (length/time, e.g.,
        cm/s).  Required.
    psi_g : float, optional
        Permittivity of the geotextile (1/time, e.g., 1/s).  Provided
        by manufacturers or from testing per ASTM D4491.
    t_g : float, optional
        Geotextile thickness (length, e.g., cm or mm).

    Returns
    -------
    float
        Factor of safety for geotextile permeability, FS_g
        (dimensionless).

    Raises
    ------
    ValueError
        If insufficient parameters are provided or if values are not
        positive.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 6, Equation 6-16, p. 326.
    """
    if k_s is None or k_s <= 0.0:
        raise ValueError("k_s (base soil hydraulic conductivity) must be positive.")

    if k_g is not None:
        if k_g <= 0.0:
            raise ValueError("k_g must be positive.")
        return k_g / k_s

    if psi_g is not None and t_g is not None:
        if psi_g <= 0.0:
            raise ValueError("psi_g must be positive.")
        if t_g <= 0.0:
            raise ValueError("t_g must be positive.")
        return (psi_g * t_g) / k_s

    raise ValueError(
        "Provide either k_g (geotextile hydraulic conductivity) or both "
        "psi_g (permittivity) and t_g (thickness)."
    )
