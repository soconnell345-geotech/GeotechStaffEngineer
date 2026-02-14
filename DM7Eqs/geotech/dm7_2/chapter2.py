"""
UFC 3-220-20, Chapter 2: Excavations

Equations 2-1 through 2-10 covering critical height of vertical cuts in clay,
normalized wall stiffness for deep excavations, angular distortion and
horizontal strain for damage prediction, ground movement estimation for
stiff and soft clays, scaled distance and peak particle velocity for
blasting design, and permeability of sheet piling.

Reference:
    UFC 3-220-20, Foundations and Earth Structures,
    16 January 2025
"""

import math


def critical_depth_vertical_cut_clay(s_u: float, gamma_t: float) -> float:
    """Critical depth of a vertical cut in clay (Equation 2-1).

    Computes the maximum depth to which a vertical excavation in clay
    can stand unsupported before failure occurs.  This is a theoretical
    upper bound based on undrained (short-term) conditions; progressive
    deterioration in stability with time and stress release means that
    actual safe heights are lower.  OSHA requires all excavations be
    sloped or supported if greater than 5 ft deep.

    .. math::
        H_{crit} = \\frac{4 \\, s_u}{\\gamma_t}

    Parameters
    ----------
    s_u : float
        Undrained shear strength of the clay (stress, e.g., psf, Pa,
        or kPa).
    gamma_t : float
        Total (moist) unit weight of the clay (force/volume, e.g.,
        pcf, N/m^3, or kN/m^3).  Must use units consistent with
        *s_u* so that the result has dimensions of length.

    Returns
    -------
    float
        Critical depth *H_crit* of the vertical cut (length, e.g.,
        ft or m).  Units depend on the consistent unit system used
        for the inputs.

    Raises
    ------
    ValueError
        If *s_u* is negative or *gamma_t* is zero or negative.

    Notes
    -----
    This equation applies to cohesive (clay) soils under undrained
    conditions.  Changes in shear strength with time and stress release
    from the excavation can lead to progressive deterioration in
    stability.  The critical depth should not be used as a design depth
    without appropriate factors of safety.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-2.2.1, Equation 2-1, p. 94.
    """
    if s_u < 0.0:
        raise ValueError("s_u must be non-negative.")
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    return (4.0 * s_u) / gamma_t


def normalized_wall_stiffness(
    E: float, I: float, gamma_t: float, h: float
) -> float:
    """Normalized wall stiffness for deep excavation systems (Equation 2-2).

    Computes the normalized wall stiffness as defined by Mana and Clough
    (1981).  This parameter is used to classify excavation support walls
    as flexible or stiff for basal heave and ground movement analyses.
    Flexible walls (e.g., sheet piling) typically have K_wall = 10 to 50;
    stiff concrete diaphragm walls often have K_wall > 100.

    The normalized wall stiffness is greatly influenced by the vertical
    spacing of the support system braces or anchors, because this variable
    is raised to the fourth power.

    .. math::
        K_{wall} = \\frac{E \\, I}{\\gamma_t \\, h^4}

    Parameters
    ----------
    E : float
        Young's modulus of the wall material (stress, e.g., psf, Pa,
        or kPa).
    I : float
        Second moment of area (moment of inertia) of the wall section
        per unit length of wall (length^4 / length = length^3, e.g.,
        ft^3, m^3).  For a wall of thickness *t*, I = t^3 / 12.
    gamma_t : float
        Total unit weight of the retained soil (force/volume, e.g.,
        pcf, N/m^3, or kN/m^3).
    h : float
        Vertical spacing of the support system braces or anchors
        (length, e.g., ft or m).

    Returns
    -------
    float
        Normalized wall stiffness *K_wall* (dimensionless).

    Raises
    ------
    ValueError
        If *E*, *I*, *gamma_t*, or *h* is zero or negative.

    Notes
    -----
    Ground anchors and internal bracing can be prestressed to reduce
    mobilization movement.  Soil nails, in contrast, require movement
    to develop support forces.  Soldier piles and wood lagging walls are
    stiffer than sheet piling walls but are likely to deflect similarly.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-4.3, Equation 2-2, p. 111.
    Mana, A.I. and Clough, G.W. (1981).
    """
    if E <= 0.0:
        raise ValueError("E must be positive.")
    if I <= 0.0:
        raise ValueError("I must be positive.")
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    return (E * I) / (gamma_t * h ** 4)


def angular_distortion(
    delta_Vi: float, delta_Vj: float, d_b: float
) -> float:
    """Angular distortion between two points on a building (Equation 2-3).

    Computes the angular distortion, which is the differential vertical
    movement (settlement) between two points on a building divided by
    the distance separating them.  This parameter is used with horizontal
    strain to assess damage potential to structures adjacent to deep
    excavations using the Boscardin and Cording (1989) damage categories.

    .. math::
        \\beta = \\frac{\\delta_{Vi} - \\delta_{Vj}}{d_b}

    Parameters
    ----------
    delta_Vi : float
        Estimated settlement at point *i* on the building (length,
        e.g., in, ft, mm, or m).  Typically the point closest to the
        excavation (front of building).
    delta_Vj : float
        Estimated settlement at point *j* on the building (same length
        unit as *delta_Vi*).  Typically the point farthest from the
        excavation (back of building).
    d_b : float
        Distance separating the two points, typically the building
        width (same length unit as *delta_Vi*).

    Returns
    -------
    float
        Angular distortion *beta* (dimensionless, radians).

    Raises
    ------
    ValueError
        If *d_b* is zero or negative.

    Notes
    -----
    In most cases, beta is measured across the whole building width.
    The movements at the front and back of the building are estimated
    from delta_Vm and the movement profile behind the wall, using the
    methods in Sections 2-4.4.1 and 2-4.4.2.  Damage categories from
    Boscardin and Cording (1989) range from negligible to severe to
    very severe.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-4.4.3, Equation 2-3, p. 122.
    """
    if d_b <= 0.0:
        raise ValueError("d_b must be positive.")
    return (delta_Vi - delta_Vj) / d_b


def horizontal_strain(
    delta_Hi: float, delta_Hj: float, d_b: float
) -> float:
    """Horizontal strain between two points on a building (Equation 2-4).

    Computes the horizontal strain across a building, which is the
    differential horizontal movement between two points divided by
    the distance separating them.  This parameter is used together with
    angular distortion to assess damage potential to structures adjacent
    to deep excavations using the Boscardin and Cording (1989) damage
    categories mapped in Figure 2-14 of the UFC.

    .. math::
        \\varepsilon_h = \\frac{\\delta_{Hi} - \\delta_{Hj}}{d_b}

    Parameters
    ----------
    delta_Hi : float
        Estimated horizontal movement at point *i* on the building
        (length, e.g., in, ft, mm, or m).  Typically the point closest
        to the excavation.
    delta_Hj : float
        Estimated horizontal movement at point *j* on the building
        (same length unit as *delta_Hi*).  Typically the point farthest
        from the excavation.
    d_b : float
        Distance separating the two points, typically the building
        width (same length unit as *delta_Hi*).

    Returns
    -------
    float
        Horizontal strain *epsilon_h* (dimensionless).

    Raises
    ------
    ValueError
        If *d_b* is zero or negative.

    Notes
    -----
    In most cases, epsilon_h is measured across the whole building
    width.  The movements at the front and back of the building are
    estimated from delta_Hm and the movement profile behind the wall.
    When beta is approximately 0, horizontal tensile strains equal
    critical tensile strains.  When epsilon_h is approximately 0, the
    boundaries are inclined at about 45 degrees and represent diagonal
    tensile strains.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-4.4.3, Equation 2-4, p. 122.
    """
    if d_b <= 0.0:
        raise ValueError("d_b must be positive.")
    return (delta_Hi - delta_Hj) / d_b


def movement_stiff_clay_sand(
    delta_m: float, d_i: float, d_0: float
) -> float:
    """Ground movement at a point behind the wall in stiff clays or sands (Equation 2-5).

    Estimates the horizontal or vertical ground movement at a given
    distance from the excavation support wall for stiff to hard clays
    and sands.  The movement profile is assumed to decrease linearly
    from the maximum at the wall to zero at a distance *d_0* from the
    wall.

    .. math::
        \\delta_i = \\delta_m \\left( \\frac{d_0 - d_i}{d_0} \\right)

    Parameters
    ----------
    delta_m : float
        Maximum horizontal (delta_Hm) or vertical (delta_Vm) movement
        at the wall (length, e.g., in, ft, mm, or m).
    d_i : float
        Distance from the wall to the point of interest (same length
        unit as *delta_m*).
    d_0 : float
        Distance from the wall at which movement becomes negligible
        (same length unit as *delta_m*).  Use d_0 = 3*H for stiff to
        hard clays and d_0 = 2*H for sand, where H is the excavation
        depth.

    Returns
    -------
    float
        Estimated horizontal or vertical movement *delta_i* at the
        point of interest (same length unit as *delta_m*).  Returns
        0.0 if *d_i* >= *d_0*.

    Raises
    ------
    ValueError
        If *delta_m* is negative, *d_i* is negative, or *d_0* is
        zero or negative.

    Notes
    -----
    For stiff to hard clays, d_0 = 3H where H is the excavation
    depth.  For sands, d_0 = 2H.  This equation applies to both
    horizontal and vertical movements.  The movement profile is
    triangular (linear decrease from wall to d_0).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-4.4.3, Equation 2-5, p. 122.
    """
    if delta_m < 0.0:
        raise ValueError("delta_m must be non-negative.")
    if d_i < 0.0:
        raise ValueError("d_i must be non-negative.")
    if d_0 <= 0.0:
        raise ValueError("d_0 must be positive.")
    if d_i >= d_0:
        return 0.0
    return delta_m * (d_0 - d_i) / d_0


def movement_soft_to_medium_clay(
    delta_m: float, d_i: float, H: float
) -> float:
    """Ground movement at a point behind the wall in soft to medium clays (Equation 2-6).

    Estimates the horizontal or vertical ground movement at a given
    distance from the excavation support wall for soft to medium clays.
    The movement profile is trapezoidal: constant at the maximum value
    from the wall to 0.75*H, then decreasing linearly to zero at 1.5*H.

    .. math::
        \\delta_i = \\begin{cases}
            \\delta_m & \\text{for } 0 \\le d_i \\le 0.75H \\\\
            \\delta_m \\left( \\frac{1.5H - d_i}{0.75H} \\right)
                & \\text{for } 0.75H < d_i \\le 1.5H
        \\end{cases}

    Parameters
    ----------
    delta_m : float
        Maximum horizontal (delta_Hm) or vertical (delta_Vm) movement
        at the wall (length, e.g., in, ft, mm, or m).
    d_i : float
        Distance from the wall to the point of interest (same length
        unit as *delta_m*).
    H : float
        Depth of excavation (same length unit as *delta_m*).

    Returns
    -------
    float
        Estimated horizontal or vertical movement *delta_i* at the
        point of interest (same length unit as *delta_m*).  Returns
        0.0 if *d_i* > 1.5*H.

    Raises
    ------
    ValueError
        If *delta_m* is negative, *d_i* is negative, or *H* is
        zero or negative.

    Notes
    -----
    The trapezoidal profile for soft to medium clays contrasts with
    the triangular (linear) profile used for stiff to hard clays and
    sands (Equation 2-5).  When reasonable care is used during
    construction and the factor of safety against basal heave is about
    2, delta_Vm = 1% * H may be assumed.  When walls are flexible and
    FBH < 1.5, delta_Vm = 2% * H is a reasonable assumption.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-4.4.3, Equation 2-6, p. 123.
    """
    if delta_m < 0.0:
        raise ValueError("delta_m must be non-negative.")
    if d_i < 0.0:
        raise ValueError("d_i must be non-negative.")
    if H <= 0.0:
        raise ValueError("H must be positive.")
    if d_i <= 0.75 * H:
        return delta_m
    elif d_i <= 1.5 * H:
        return delta_m * (1.5 * H - d_i) / (0.75 * H)
    else:
        return 0.0


def scaled_distance(D: float, W: float, beta: float = 0.5) -> float:
    """Scaled distance for evaluating blast effects on structures (Equation 2-7).

    Computes the scaled distance, which is the true distance from a
    blast charge to a structure corrected by the weight of the charge.
    The scaled distance is used with empirical relationships to predict
    peak particle velocity and evaluate the potential for structural
    damage induced by blasting vibration.

    .. math::
        SD = \\frac{D}{W^{\\beta}}

    Parameters
    ----------
    D : float
        True distance from the charge to the structure (ft).
    W : float
        Weight of the charge (lb).
    beta : float, optional
        Distance exponent (dimensionless).  Use 0.33 for near-field
        structures (i.e., less than 20 ft from the charge) or 0.5 for
        structures farther from the charge.  Default is 0.5.

    Returns
    -------
    float
        Scaled distance *SD* (ft/lb^beta).  The scaled distance is not
        dimensionally correct and requires use of the indicated units
        (ft and lb).

    Raises
    ------
    ValueError
        If *D* is negative, *W* is zero or negative, or *beta* is
        not 0.33 or 0.5.

    Notes
    -----
    This equation is empirical and requires specific units: distance in
    feet and charge weight in pounds.  The scaled distance is not
    dimensionally consistent.  Use beta = 0.33 (cube-root scaling) for
    near-field structures within 20 ft of the charge, and beta = 0.5
    (square-root scaling) for structures farther away.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-5.3.1, Equation 2-7, p. 132.
    USBM (1971), Oriard (1987).
    """
    if D < 0.0:
        raise ValueError("D must be non-negative.")
    if W <= 0.0:
        raise ValueError("W must be positive.")
    if beta not in (0.33, 0.5):
        raise ValueError("beta must be 0.33 (near field) or 0.5 (far field).")
    return D / (W ** beta)


def peak_particle_velocity(K: float, SD: float) -> float:
    """Peak particle velocity from blasting (Equation 2-8).

    Computes the peak (maximum) particle velocity (PPV), which is the
    longitudinal velocity of a particle in the direction of the wave
    generated by blasting.  PPV is an accepted criterion for evaluating
    the potential for structural damage induced by blasting vibration.

    .. math::
        PPV = K \\cdot SD^{-1.6}

    Parameters
    ----------
    K : float
        Confinement factor (dimensionless, empirical).  Typical values:
        lower bound = 20, upper bound = 242, average = 150.  The values
        are empirical and require use of the indicated units (ft, lb,
        in/sec).
    SD : float
        Scaled distance (ft/lb^beta) as computed from Equation 2-7.

    Returns
    -------
    float
        Peak particle velocity *PPV* (in/sec when using the standard
        empirical units).

    Raises
    ------
    ValueError
        If *K* is zero or negative, or *SD* is zero or negative.

    Notes
    -----
    The values of K are empirical and require use of the indicated
    units (distance in ft, charge weight in lb, velocity in in/sec).
    K may be calculated from measured blast data using Equation 2-9.
    The critical level of PPV depends on rock properties, the nature
    of the overburden, frequency characteristics of the structure, and
    the capability of the structure to withstand dynamic stresses.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-5.3.1, Equation 2-8, p. 132.
    """
    if K <= 0.0:
        raise ValueError("K must be positive.")
    if SD <= 0.0:
        raise ValueError("SD must be positive.")
    return K * SD ** (-1.6)


def confinement_factor_from_blast_data(PPV: float, SD: float) -> float:
    """Confinement factor calculated from blast monitoring data (Equation 2-9).

    Back-calculates the confinement factor *K* from measured peak
    particle velocity and the corresponding scaled distance.  This
    allows site-specific calibration of the blasting attenuation
    relationship (Equation 2-8) using field measurements.

    .. math::
        K = \\frac{PPV}{SD^{-1.6}}

    Parameters
    ----------
    PPV : float
        Measured peak particle velocity (in/sec when using the
        standard empirical units).
    SD : float
        Scaled distance (ft/lb^beta) as computed from Equation 2-7.

    Returns
    -------
    float
        Confinement factor *K* (dimensionless, empirical).  Typical
        range: lower bound = 20, upper bound = 242, average = 150.

    Raises
    ------
    ValueError
        If *PPV* is negative or *SD* is zero or negative.

    Notes
    -----
    This is simply a rearrangement of Equation 2-8 to solve for K.
    The values are empirical and require use of the indicated units
    (distance in ft, charge weight in lb, velocity in in/sec).  Once
    K is determined from site-specific blast monitoring data, it can
    be used in Equation 2-8 to predict PPV for different charge
    weights and distances.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-5.3.1, Equation 2-9, p. 132.
    """
    if PPV < 0.0:
        raise ValueError("PPV must be non-negative.")
    if SD <= 0.0:
        raise ValueError("SD must be positive.")
    return PPV / (SD ** (-1.6))


def inverse_specific_resistance_sheet_piling(
    q: float, gamma_w: float, delta_p: float
) -> float:
    """Inverse specific resistance of sheet pile wall interlocks (Equation 2-10).

    Computes the inverse specific resistance of a sheet pile wall,
    which characterizes the permeability through the interlocks.  This
    parameter is defined in European Standard EN 12063 (1999) and is
    used to estimate seepage flow through sheet piling for groundwater
    control during excavations.

    .. math::
        \\rho = \\frac{q \\, \\gamma_w}{\\Delta p}

    Parameters
    ----------
    q : float
        Discharge or flow rate per unit height along the interlock
        (volume / time / length, e.g., ft^3/sec/ft, m^3/sec/m, or
        cm^2/sec).
    gamma_w : float
        Unit weight of water (force/volume, e.g., 62.4 pcf,
        9.81 kN/m^3).
    delta_p : float
        Differential pressure across the sheet pile wall
        (stress, e.g., psf, Pa, or kPa).

    Returns
    -------
    float
        Inverse specific resistance *rho* (length/time, e.g.,
        cm/sec).  Typical values range from about 1e-3 cm/sec for
        unsealed joints to about 1e-9 cm/sec for sealed joints.

    Raises
    ------
    ValueError
        If *q* is negative, *gamma_w* is zero or negative, or
        *delta_p* is zero or negative.

    Notes
    -----
    Seepage through sheet piling occurs only through the interlocks.
    Seepage can be reduced by maintaining tension in the interlocks
    and/or by sealing the joints with bitumen or swelling fillers.
    Typical rho values:
      - Unsealed joints: ~1e-3 cm/sec
      - Sealed joints (no tension): ~1e-9 cm/sec
      - Vinyl sheet piling unsealed (tension): ~1e-5 cm/sec
      - Vinyl sheet piling sealed: ~1e-10 cm/sec

    If rho is known or assumed, this equation can be rearranged to
    calculate the flow rate per unit length of interlock:
    q = rho * delta_p / gamma_w.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 2, Section 2-6.2, Equation 2-10, p. 136.
    European Standard EN 12063 (1999).
    """
    if q < 0.0:
        raise ValueError("q must be non-negative.")
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    if delta_p <= 0.0:
        raise ValueError("delta_p must be positive.")
    return (q * gamma_w) / delta_p
