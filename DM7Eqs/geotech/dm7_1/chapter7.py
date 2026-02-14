"""
UFC 3-220-10, Chapter 7: Slope Stability

Equations 7-1 through 7-6 covering factor of safety, seepage forces,
geosynthetic reinforcement long-term design strength, pullout resistance,
coefficient of interaction, and pullout resistance factor.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""

import math


def factor_of_safety(s: float, tau: float) -> float:
    """Factor of safety for slope stability (Equation 7-1).

    The stability of slopes is characterized by the factor of safety, F.
    A factor of safety equal to unity indicates barely stable equilibrium
    (at the point of failure).  Values greater than unity indicate increasing
    stability; values less than unity indicate an unstable slope.

    The shear strength *s* depends on the strength model used to
    characterize the soil (e.g., effective stress or total stress models).
    The shear stress *tau* is calculated from statics along with
    assumptions regarding the conditions for equilibrium.

    .. math::
        F = \\frac{s}{\\tau}

    Parameters
    ----------
    s : float
        Shear strength of the soil along the failure surface (psf, kPa,
        or any consistent stress unit).
    tau : float
        Shear stress required for equilibrium along the failure surface
        (same stress unit as *s*).

    Returns
    -------
    float
        Factor of safety (dimensionless).

    Raises
    ------
    ValueError
        If *tau* is zero or negative, or if *s* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-1, p. 384.
    """
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    if s < 0.0:
        raise ValueError("s must be non-negative.")
    return s / tau


def seepage_force(i: float, gamma_w: float) -> float:
    """Seepage force per unit volume (Equation 7-2).

    As water flows through a soil, a seepage force is imparted to the soil
    from the viscous resistance to the flow of water.  The seepage force
    provides a force per unit volume for the volume where the hydraulic
    gradient (head loss) occurs.

    For slope stability analyses, the effect of flowing water can be
    handled by calculating the seepage forces for each slice or free body
    and using the buoyant unit weight of the soil below the phreatic
    surface.

    .. math::
        S = i \\cdot \\gamma_w

    Parameters
    ----------
    i : float
        Hydraulic gradient (dimensionless).
    gamma_w : float
        Unit weight of water (force / length^3, e.g., 62.4 pcf or
        9.81 kN/m^3).

    Returns
    -------
    float
        Seepage force per unit volume (force / length^3, same unit system
        as *gamma_w*).

    Raises
    ------
    ValueError
        If *gamma_w* is zero or negative, or if *i* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-2, p. 392.
    """
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    if i < 0.0:
        raise ValueError("i must be non-negative.")
    return i * gamma_w


def long_term_geosynthetic_strength(
    t_ult: float, rf_cr: float, rf_d: float, rf_id: float
) -> float:
    """Long-term design strength of geosynthetic reinforcement (Equation 7-3).

    Computes the long-term tensile strength (Tal) of a geosynthetic for
    use in the design of a mechanically stabilized earth (MSE) slope,
    following FHWA procedures (FHWA 2009b).  The ultimate tensile strength
    (MARV) is reduced by three independent reduction factors that account
    for creep, environmental degradation, and installation damage.

    .. math::
        T_{al} = \\frac{T_{ULT}}{RF_{CR} \\times RF_D \\times RF_{ID}}

    Parameters
    ----------
    t_ult : float
        Ultimate tensile strength of the geosynthetic based on the
        minimum average roll value, MARV (force / length, e.g., lb/ft
        or kN/m).
    rf_cr : float
        Reduction factor for creep under sustained tensile loading
        (dimensionless, > 1.0).
    rf_d : float
        Reduction factor for degradation due to environmental conditions
        (dimensionless, > 1.0).
    rf_id : float
        Reduction factor for damage during installation
        (dimensionless, > 1.0).

    Returns
    -------
    float
        Long-term tensile strength, Tal (force / length, same unit system
        as *t_ult*).

    Raises
    ------
    ValueError
        If *t_ult* is negative, or if any reduction factor is less than
        or equal to zero.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-3, p. 405.
    """
    if t_ult < 0.0:
        raise ValueError("t_ult must be non-negative.")
    if rf_cr <= 0.0:
        raise ValueError("rf_cr must be positive.")
    if rf_d <= 0.0:
        raise ValueError("rf_d must be positive.")
    if rf_id <= 0.0:
        raise ValueError("rf_id must be positive.")
    return t_ult / (rf_cr * rf_d * rf_id)


def geosynthetic_pullout_resistance(
    f_star: float,
    alpha: float,
    sigma_v_eff: float,
    l_e: float,
    c_surfaces: float = 2.0,
) -> float:
    """Pullout resistance of geosynthetic reinforcement (Equation 7-4).

    Computes the resistance of a geosynthetic reinforcement to pullout
    from between layers of confining soil.  This resistance is used to
    determine the required reinforcement length in the design of an MSE
    slope, following FHWA procedures (FHWA 2009b).

    The embedment length *l_e* is measured behind the trial failure
    surface and should be no less than 3 feet (0.9 m) to ensure adequate
    pullout resistance.

    .. math::
        P_r = F^{*} \\cdot \\alpha \\cdot \\sigma'_v \\cdot L_e \\cdot C

    Parameters
    ----------
    f_star : float
        Pullout resistance factor (dimensionless).
    alpha : float
        Scale correction factor to account for nonlinear stress reduction
        over the embedded length of the reinforcement (dimensionless,
        typically 0.6 to 1.0).
    sigma_v_eff : float
        Effective vertical stress at the soil-reinforcement interface
        (psf, kPa, or any consistent stress unit).
    l_e : float
        Length of reinforcement embedded behind the trial failure surface
        (ft or m).  Must be >= 3 ft (0.9 m) per FHWA guidelines.
    c_surfaces : float, optional
        Number of surfaces on which pullout resistance is mobilized
        (dimensionless).  Default is 2 for geosynthetics (top and bottom
        surfaces).

    Returns
    -------
    float
        Pullout resistance, Pr (force / length of wall, e.g., lb/ft or
        kN/m).  Units depend on the consistent unit system used for the
        inputs.

    Raises
    ------
    ValueError
        If *f_star*, *alpha*, *sigma_v_eff*, *l_e*, or *c_surfaces* is
        negative, or if *c_surfaces* is zero.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-4, p. 407.
    """
    if f_star < 0.0:
        raise ValueError("f_star must be non-negative.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if sigma_v_eff < 0.0:
        raise ValueError("sigma_v_eff must be non-negative.")
    if l_e < 0.0:
        raise ValueError("l_e must be non-negative.")
    if c_surfaces <= 0.0:
        raise ValueError("c_surfaces must be positive.")
    return f_star * alpha * sigma_v_eff * l_e * c_surfaces


def coefficient_of_interaction(
    delta_deg: float, phi_eff_deg: float
) -> float:
    """Coefficient of interaction for soil-geosynthetic interface (Equation 7-5).

    Characterizes the soil-geosynthetic interaction based on pullout
    tests.  The coefficient of interaction (Ci) relates the effective
    soil-geosynthetic interface friction angle to the effective stress
    internal angle of friction of the soil.

    .. math::
        C_i = \\frac{\\tan(\\delta)}{\\tan(\\phi')}

    Parameters
    ----------
    delta_deg : float
        Effective soil-geosynthetic interface friction angle (degrees).
    phi_eff_deg : float
        Effective stress internal angle of friction of the soil (degrees).
        Must be greater than 0 and less than 90 degrees.

    Returns
    -------
    float
        Coefficient of interaction, Ci (dimensionless).

    Raises
    ------
    ValueError
        If *phi_eff_deg* is zero or negative, or if either angle is
        outside the range [0, 90) degrees.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-5, p. 407.
    """
    if phi_eff_deg <= 0.0 or phi_eff_deg >= 90.0:
        raise ValueError(
            "phi_eff_deg must be greater than 0 and less than 90 degrees."
        )
    if delta_deg < 0.0 or delta_deg >= 90.0:
        raise ValueError(
            "delta_deg must be non-negative and less than 90 degrees."
        )
    delta_rad = math.radians(delta_deg)
    phi_rad = math.radians(phi_eff_deg)
    return math.tan(delta_rad) / math.tan(phi_rad)


def pullout_resistance_factor(
    c_i: float, phi_eff_deg: float
) -> float:
    """Pullout resistance factor from coefficient of interaction (Equation 7-6).

    Relates the pullout resistance factor (F*) used in Equation 7-4 to the
    coefficient of interaction (Ci) determined from pullout tests
    (Equation 7-5) and the effective stress friction angle of the soil.

    .. math::
        F^{*} = C_i \\cdot \\tan(\\phi')

    Parameters
    ----------
    c_i : float
        Coefficient of interaction from pullout testing (dimensionless).
        See :func:`coefficient_of_interaction` (Equation 7-5).
    phi_eff_deg : float
        Effective stress internal angle of friction of the soil (degrees).
        Must be greater than 0 and less than 90 degrees.

    Returns
    -------
    float
        Pullout resistance factor, F* (dimensionless).

    Raises
    ------
    ValueError
        If *c_i* is negative, or if *phi_eff_deg* is outside the range
        (0, 90) degrees.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 7, Equation 7-6, p. 407.
    """
    if c_i < 0.0:
        raise ValueError("c_i must be non-negative.")
    if phi_eff_deg <= 0.0 or phi_eff_deg >= 90.0:
        raise ValueError(
            "phi_eff_deg must be greater than 0 and less than 90 degrees."
        )
    phi_rad = math.radians(phi_eff_deg)
    return c_i * math.tan(phi_rad)
