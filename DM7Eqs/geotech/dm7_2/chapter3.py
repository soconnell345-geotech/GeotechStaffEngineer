"""
UFC 3-220-20, Chapter 3: Earthwork, Hydraulic, and Underwater Fills

Equations 3-1 through 3-10 covering compaction characterization (dry unit
weight, degree of saturation, relative compaction, relative water content,
relative density), oversize particle corrections, and borrow-fill volume
calculations.

Reference:
    UFC 3-220-20, Foundations and Earth Structures, 16 January 2025
"""


def dry_unit_weight(gamma_t: float, w: float) -> float:
    """Dry unit weight from total unit weight and water content (Equation 3-1).

    Compaction focuses on changing the dry unit weight of soil.  This equation
    converts the total (moist) unit weight to the dry unit weight by removing
    the contribution of pore water.

    .. math::
        \\gamma_d = \\frac{\\gamma_t}{1 + \\frac{w}{100}}

    Parameters
    ----------
    gamma_t : float
        Total (moist) unit weight of the soil (pcf or kN/m^3).
    w : float
        Water content expressed as a percentage (%).  For example, a water
        content of 15% is entered as 15.

    Returns
    -------
    float
        Dry unit weight (pcf or kN/m^3), in the same units as *gamma_t*.

    Raises
    ------
    ValueError
        If *gamma_t* is not positive or if *w* is negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-1, p. 147.
    """
    if gamma_t <= 0.0:
        raise ValueError("gamma_t must be positive.")
    if w < 0.0:
        raise ValueError("w must be non-negative.")
    return gamma_t / (1.0 + w / 100.0)


def dry_unit_weight_from_saturation(S: float, w: float,
                                    G_s: float,
                                    gamma_w: float = 62.4) -> float:
    """Dry unit weight from degree of saturation (Equation 3-2).

    Relates the dry unit weight to the degree of saturation, water content,
    and specific gravity of solids.  This relationship defines the family of
    constant-saturation curves on the compaction plane.

    .. math::
        \\gamma_d = \\frac{S \\, \\gamma_w}{\\frac{w}{100} \\left(S + \\frac{100}{G_s}\\right)}

    Rearranging the standard phase relationship:

    .. math::
        \\gamma_d = \\frac{\\gamma_w}{\\frac{w}{S} + \\frac{1}{G_s}}

    Parameters
    ----------
    S : float
        Degree of saturation as a percentage (%).  Must be in the range
        (0, 100].  For example, 85% saturation is entered as 85.
    w : float
        Water content as a percentage (%).  Must be positive.
    G_s : float
        Specific gravity of solids (dimensionless).  Typically 2.60 to 2.80
        for most soils.
    gamma_w : float, optional
        Unit weight of water (pcf or kN/m^3).  Defaults to 62.4 pcf.
        Use 9.81 kN/m^3 for SI units.

    Returns
    -------
    float
        Dry unit weight (pcf or kN/m^3), in the same units as *gamma_w*.

    Raises
    ------
    ValueError
        If *S* is not in (0, 100], *w* is not positive, *G_s* is not
        positive, or *gamma_w* is not positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-2, p. 147.
    """
    if S <= 0.0 or S > 100.0:
        raise ValueError("S must be in the range (0, 100].")
    if w <= 0.0:
        raise ValueError("w must be positive.")
    if G_s <= 0.0:
        raise ValueError("G_s must be positive.")
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    w_dec = w / 100.0
    return gamma_w / (w_dec / (S / 100.0) + 1.0 / G_s)


def relative_compaction(gamma_d_field: float, gamma_d_max: float) -> float:
    """Relative compaction of compacted fill (Equation 3-3).

    Compares the dry unit weight achieved in the field to the maximum dry
    unit weight from a laboratory compaction test at a specified compactive
    effort.  Used for soils with appreciable fines (greater than about 5%
    to 15% passing the #200 sieve).

    .. math::
        R.C. = \\frac{\\gamma_{d,field}}{\\gamma_{d,max}} \\times 100\\%

    Parameters
    ----------
    gamma_d_field : float
        Dry unit weight of the compacted fill measured in the field
        (pcf or kN/m^3).
    gamma_d_max : float
        Maximum dry unit weight for a specified compactive effort from a
        laboratory compaction test such as ASTM D698 or ASTM D1557
        (pcf or kN/m^3).  Must use the same units as *gamma_d_field*.

    Returns
    -------
    float
        Relative compaction as a percentage (%).

    Raises
    ------
    ValueError
        If *gamma_d_max* is zero or negative, or if *gamma_d_field* is
        negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-3, p. 149.
    """
    if gamma_d_max <= 0.0:
        raise ValueError("gamma_d_max must be positive.")
    if gamma_d_field < 0.0:
        raise ValueError("gamma_d_field must be non-negative.")
    return (gamma_d_field / gamma_d_max) * 100.0


def relative_water_content(w_field: float, w_opt: float) -> float:
    """Relative water content of compacted fill (Equation 3-4).

    Expresses the water content of the compacted fill relative to the
    optimum water content from the reference compaction curve.  Positive
    values indicate wet of optimum; negative values indicate dry of
    optimum.

    .. math::
        \\Delta w = w_{field} - w_{opt}

    Parameters
    ----------
    w_field : float
        Water content of the compacted fill as a percentage (%).
    w_opt : float
        Optimum water content from the reference compaction curve as a
        percentage (%).

    Returns
    -------
    float
        Relative water content as a percentage (%).  Positive means wet of
        optimum; negative means dry of optimum.

    Raises
    ------
    ValueError
        If *w_field* or *w_opt* is negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-4, p. 150.
    """
    if w_field < 0.0:
        raise ValueError("w_field must be non-negative.")
    if w_opt < 0.0:
        raise ValueError("w_opt must be non-negative.")
    return w_field - w_opt


def relative_density_from_void_ratio(
    e: float, e_max: float, e_min: float
) -> float:
    """Relative density from void ratios (Equation 3-5, void-ratio form).

    Characterizes the density state of coarse-grained soils without
    appreciable fines (less than about 5% to 15% fines).  A relative
    density of 0% corresponds to the loosest state (e = e_max) and 100%
    corresponds to the densest state (e = e_min).

    .. math::
        D_r = \\frac{e_{max} - e}{e_{max} - e_{min}} \\times 100\\%

    Parameters
    ----------
    e : float
        Void ratio of the compacted soil (dimensionless).
    e_max : float
        Maximum void ratio corresponding to the loosest state, determined
        by ASTM D4254 (dimensionless).
    e_min : float
        Minimum void ratio corresponding to the densest state, determined
        by ASTM D4253 (dimensionless).

    Returns
    -------
    float
        Relative density as a percentage (%).

    Raises
    ------
    ValueError
        If *e_max* is less than or equal to *e_min*.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-5, p. 151.
    """
    if e_max <= e_min:
        raise ValueError("e_max must be greater than e_min.")
    return ((e_max - e) / (e_max - e_min)) * 100.0


def relative_density_from_dry_unit_weight(
    gamma_d_field: float, gamma_d_max: float, gamma_d_min: float
) -> float:
    """Relative density from dry unit weights (Equation 3-5, density form).

    Equivalent formulation of relative density using dry unit weights instead
    of void ratios.  Applicable to coarse-grained soils without appreciable
    fines (less than about 5% to 15% passing the #200 sieve).

    .. math::
        D_r = \\frac{\\gamma_{d,max}}{\\gamma_{d,field}}
              \\cdot \\frac{\\gamma_{d,field} - \\gamma_{d,min}}
                          {\\gamma_{d,max} - \\gamma_{d,min}}
              \\times 100\\%

    Parameters
    ----------
    gamma_d_field : float
        Dry unit weight of the compacted soil (pcf or kN/m^3).
    gamma_d_max : float
        Maximum index dry unit weight corresponding to *e_min* (the densest
        state), determined by ASTM D4253 (pcf or kN/m^3).  Same units as
        *gamma_d_field*.
    gamma_d_min : float
        Minimum index dry unit weight corresponding to *e_max* (the loosest
        state), determined by ASTM D4254 (pcf or kN/m^3).  Same units as
        *gamma_d_field*.

    Returns
    -------
    float
        Relative density as a percentage (%).

    Raises
    ------
    ValueError
        If *gamma_d_field* is zero or negative, or if *gamma_d_max* is less
        than or equal to *gamma_d_min*.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-5, p. 151.
    """
    if gamma_d_field <= 0.0:
        raise ValueError("gamma_d_field must be positive.")
    if gamma_d_max <= gamma_d_min:
        raise ValueError("gamma_d_max must be greater than gamma_d_min.")
    return (
        (gamma_d_max / gamma_d_field)
        * ((gamma_d_field - gamma_d_min) / (gamma_d_max - gamma_d_min))
        * 100.0
    )


def oversize_corrected_water_content(
    P_C: float, w_C: float, P_F: float, w_F: float
) -> float:
    """Combined water content with oversize correction (Equation 3-6).

    Corrects the water content measured on the finer fraction to account for
    the oversize particles that were excluded from the laboratory compaction
    mold.  Applicable when more than 5% oversize particles are present.
    Corrections are typically limited to 40% oversize for 4.75 mm particles
    and 30% oversize for 3/4-inch particles.

    .. math::
        w_T = P_C \\, w_C + P_F \\, w_F

    Parameters
    ----------
    P_C : float
        Percent of the oversize (coarse) fraction as a decimal.  For example,
        30% oversize is entered as 0.30.
    w_C : float
        Water content of the oversize fraction as a decimal.
    P_F : float
        Percent of the finer fraction as a decimal.  Must satisfy
        P_C + P_F = 1.0.
    w_F : float
        Water content of the finer fraction as a decimal.

    Returns
    -------
    float
        Combined water content of the finer and oversize fractions as a
        decimal.

    Raises
    ------
    ValueError
        If *P_C* or *P_F* is outside [0, 1], if *w_C* or *w_F* is
        negative, or if P_C + P_F does not equal 1.0 (within tolerance).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-6, p. 161.
    """
    if P_C < 0.0 or P_C > 1.0:
        raise ValueError("P_C must be in the range [0, 1].")
    if P_F < 0.0 or P_F > 1.0:
        raise ValueError("P_F must be in the range [0, 1].")
    if abs((P_C + P_F) - 1.0) > 1e-6:
        raise ValueError("P_C + P_F must equal 1.0.")
    if w_C < 0.0:
        raise ValueError("w_C must be non-negative.")
    if w_F < 0.0:
        raise ValueError("w_F must be non-negative.")
    return P_C * w_C + P_F * w_F


def oversize_corrected_dry_unit_weight(
    gamma_dF: float, P_C: float, G_sC: float,
    P_F: float, gamma_w: float = 62.4
) -> float:
    """Combined dry unit weight with oversize correction (Equation 3-7).

    Corrects the dry unit weight measured on the finer fraction to account
    for the oversize particles excluded from the laboratory compaction mold.
    The oversize fraction is treated as solid rock with a known specific
    gravity.

    .. math::
        \\gamma_{dT} = \\frac{\\gamma_{dF} \\, G_{sC} \\, \\gamma_w}
                            {\\gamma_{dF} \\, P_C + G_{sC} \\, \\gamma_w \\, P_F}

    Parameters
    ----------
    gamma_dF : float
        Dry unit weight of the finer fraction from the laboratory compaction
        test (pcf or kN/m^3).
    P_C : float
        Percent of the oversize (coarse) fraction as a decimal.
    G_sC : float
        Specific gravity of solids of the oversize fraction (dimensionless).
    P_F : float
        Percent of the finer fraction as a decimal.  Must satisfy
        P_C + P_F = 1.0.
    gamma_w : float, optional
        Unit weight of water (pcf or kN/m^3).  Defaults to 62.4 pcf.
        Use 9.81 kN/m^3 for SI units.

    Returns
    -------
    float
        Combined dry unit weight of the finer and oversize fractions
        (pcf or kN/m^3), in the same units as *gamma_dF* and *gamma_w*.

    Raises
    ------
    ValueError
        If any input is non-positive where required, or if P_C + P_F does
        not equal 1.0 (within tolerance).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-7, p. 162.
    """
    if gamma_dF <= 0.0:
        raise ValueError("gamma_dF must be positive.")
    if P_C < 0.0 or P_C > 1.0:
        raise ValueError("P_C must be in the range [0, 1].")
    if P_F < 0.0 or P_F > 1.0:
        raise ValueError("P_F must be in the range [0, 1].")
    if abs((P_C + P_F) - 1.0) > 1e-6:
        raise ValueError("P_C + P_F must equal 1.0.")
    if G_sC <= 0.0:
        raise ValueError("G_sC must be positive.")
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    numerator = gamma_dF * G_sC * gamma_w
    denominator = gamma_dF * P_C + G_sC * gamma_w * P_F
    if denominator <= 0.0:
        raise ValueError("Denominator is non-positive; check inputs.")
    return numerator / denominator


def borrow_volume_from_waste_weight(
    V_F: float, gamma_d_F: float, gamma_d_B: float, W_L: float
) -> float:
    """Total borrow volume from fill volume and waste weight (Equation 3-8).

    Calculates the total volume of borrow required to produce a specified
    fill volume, accounting for the difference in dry unit weights between
    borrow and fill and for material lost as waste.  The total weight of
    solids is conserved.

    .. math::
        V_B = V_F \\frac{\\gamma_{d,F}}{\\gamma_{d,B}}
              + \\frac{W_L}{\\gamma_{d,B}}

    Parameters
    ----------
    V_F : float
        Required fill volume (cubic yards, m^3, or other volume unit).
    gamma_d_F : float
        Average dry unit weight of the compacted fill (pcf or kN/m^3).
    gamma_d_B : float
        Average dry unit weight of the borrow material in its natural state
        (pcf or kN/m^3).  Must use the same units as *gamma_d_F*.
    W_L : float
        Total weight of waste material that will not be used in the fill
        (lb, kN, or other force unit consistent with *gamma_d_B* and the
        volume unit).

    Returns
    -------
    float
        Total borrow volume required (same volume unit as *V_F*).

    Raises
    ------
    ValueError
        If *gamma_d_B* is zero or negative, *V_F* is negative, or
        *gamma_d_F* is not positive.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-8, p. 173.
    """
    if V_F < 0.0:
        raise ValueError("V_F must be non-negative.")
    if gamma_d_F <= 0.0:
        raise ValueError("gamma_d_F must be positive.")
    if gamma_d_B <= 0.0:
        raise ValueError("gamma_d_B must be positive.")
    if W_L < 0.0:
        raise ValueError("W_L must be non-negative.")
    return V_F * (gamma_d_F / gamma_d_B) + W_L / gamma_d_B


def borrow_volume_from_waste_fraction(
    V_F: float, gamma_d_F: float, gamma_d_B: float, X_L: float
) -> float:
    """Total borrow volume from fill volume and waste fraction (Equation 3-9).

    Calculates the total volume of borrow required to produce a specified
    fill volume when the waste is expressed as a loss percentage rather than
    a total weight.

    .. math::
        V_B = V_F \\frac{\\gamma_{d,F}}{\\gamma_{d,B} (1 - X_L)}

    Parameters
    ----------
    V_F : float
        Required fill volume (cubic yards, m^3, or other volume unit).
    gamma_d_F : float
        Average dry unit weight of the compacted fill (pcf or kN/m^3).
    gamma_d_B : float
        Average dry unit weight of the borrow material in its natural state
        (pcf or kN/m^3).  Must use the same units as *gamma_d_F*.
    X_L : float
        Loss percentage expressed as a decimal fraction.  For example, a 10%
        loss is entered as 0.10.  Must be in the range [0, 1).

    Returns
    -------
    float
        Total borrow volume required (same volume unit as *V_F*).

    Raises
    ------
    ValueError
        If *gamma_d_B* is zero or negative, *X_L* is outside [0, 1),
        or *V_F* is negative.

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-9, p. 173.
    """
    if V_F < 0.0:
        raise ValueError("V_F must be non-negative.")
    if gamma_d_F <= 0.0:
        raise ValueError("gamma_d_F must be positive.")
    if gamma_d_B <= 0.0:
        raise ValueError("gamma_d_B must be positive.")
    if X_L < 0.0 or X_L >= 1.0:
        raise ValueError("X_L must be in the range [0, 1).")
    return V_F * gamma_d_F / (gamma_d_B * (1.0 - X_L))


def shrinkage_factor(
    gamma_d_F: float, gamma_d_B: float, X_L: float
) -> float:
    """Overall shrinkage factor for borrow-fill calculations (Equation 3-10).

    Quantifies the fractional change in total volume from borrow to fill,
    accounting for both the change in dry unit weight and the loss of
    material as waste.  Positive values indicate shrinkage (borrow volume
    exceeds fill volume); negative values indicate bulking.  If detailed
    unit weight information is not available, a shrinkage factor of 10% to
    15% of V_F can be used for estimating purposes.

    .. math::
        \\frac{\\Delta V}{V_F} = \\frac{\\gamma_{d,F}}{\\gamma_{d,B} (1 - X_L)} - 1

    where :math:`\\Delta V = V_B - V_F`.

    Parameters
    ----------
    gamma_d_F : float
        Average dry unit weight of the compacted fill (pcf or kN/m^3).
    gamma_d_B : float
        Average dry unit weight of the borrow material in its natural state
        (pcf or kN/m^3).  Must use the same units as *gamma_d_F*.
    X_L : float
        Loss percentage expressed as a decimal fraction.  For example, a 10%
        loss is entered as 0.10.  Must be in the range [0, 1).

    Returns
    -------
    float
        Shrinkage factor (dimensionless), defined as (V_B - V_F) / V_F.
        Positive means shrinkage; negative means bulking.

    Raises
    ------
    ValueError
        If *gamma_d_B* is zero or negative, or *X_L* is outside [0, 1).

    References
    ----------
    UFC 3-220-20, Foundations and Earth Structures, 16 Jan 2025,
    Chapter 3, Equation 3-10, p. 173.
    """
    if gamma_d_F <= 0.0:
        raise ValueError("gamma_d_F must be positive.")
    if gamma_d_B <= 0.0:
        raise ValueError("gamma_d_B must be positive.")
    if X_L < 0.0 or X_L >= 1.0:
        raise ValueError("X_L must be in the range [0, 1).")
    return gamma_d_F / (gamma_d_B * (1.0 - X_L)) - 1.0
