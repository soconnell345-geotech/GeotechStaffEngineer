"""
UFC 3-220-10, Chapter 3: Laboratory Testing

Equations 3-1 through 3-5 covering compaction characterization, cyclic stress
ratios, consolidation load increments, and hydraulic conductivity from
consolidation test data.

Reference:
    UFC 3-220-10, Soil Mechanics, 1 February 2022, Change 1, 11 March 2025
"""


def relative_compaction(gamma_d: float, gamma_d_max: float) -> float:
    """Relative compaction of a soil (Equation 3-1).

    Characterizes the as-compacted density of a soil relative to the maximum
    dry density obtained from an impact compaction test (ASTM D698 or
    ASTM D1557).  Applicable to soils having greater than 15% fines.

    .. math::
        RC = \\frac{\\gamma_d}{\\gamma_{d-max}} \\times 100\\%

    Parameters
    ----------
    gamma_d : float
        Dry density of the soil to be characterized (pcf or kN/m^3).
    gamma_d_max : float
        Maximum dry density from the compaction curve for a particular
        compactive effort (pcf or kN/m^3).  Must use the same units as
        *gamma_d*.

    Returns
    -------
    float
        Relative compaction as a percentage (%).

    Raises
    ------
    ValueError
        If *gamma_d_max* is zero or negative, or if *gamma_d* is negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-1, p. 140.
    """
    if gamma_d_max <= 0.0:
        raise ValueError("gamma_d_max must be positive.")
    if gamma_d < 0.0:
        raise ValueError("gamma_d must be non-negative.")
    return (gamma_d / gamma_d_max) * 100.0


def relative_density_from_void_ratio(
    e: float, e_max: float, e_min: float
) -> float:
    """Relative density from void ratios (Equation 3-2, void-ratio form).

    Characterizes the density state of a coarse-grained soil (less than 15%
    fines) using void ratios.  A relative density of 0% corresponds to the
    loosest state and 100% corresponds to the densest state.

    .. math::
        D_r = \\frac{e_{max} - e}{e_{max} - e_{min}} \\times 100\\%

    Parameters
    ----------
    e : float
        Void ratio of the soil to be characterized (dimensionless).
    e_max : float
        Maximum index void ratio corresponding to the loosest state
        (dimensionless).
    e_min : float
        Minimum index void ratio corresponding to the densest state
        (dimensionless).

    Returns
    -------
    float
        Relative density as a percentage (%).

    Raises
    ------
    ValueError
        If *e_max* equals *e_min* (division by zero) or if *e_max* < *e_min*.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-2, p. 141.
    """
    if e_max <= e_min:
        raise ValueError("e_max must be greater than e_min.")
    return ((e_max - e) / (e_max - e_min)) * 100.0


def relative_density_from_dry_density(
    gamma_d: float, gamma_d_max: float, gamma_d_min: float
) -> float:
    """Relative density from dry densities (Equation 3-2, density form).

    Equivalent formulation of relative density using dry densities instead
    of void ratios.  Applicable to coarse-grained soils with less than 15%
    fines.

    .. math::
        D_r = \\frac{\\gamma_{d-max}}{\\gamma_d}
              \\cdot \\frac{\\gamma_d - \\gamma_{d-min}}
                          {\\gamma_{d-max} - \\gamma_{d-min}}
              \\times 100\\%

    Parameters
    ----------
    gamma_d : float
        Dry density of the soil to be characterized (pcf or kN/m^3).
    gamma_d_max : float
        Maximum index dry density corresponding to *e_min* -- the densest
        state (pcf or kN/m^3).  Same units as *gamma_d*.
    gamma_d_min : float
        Minimum index dry density corresponding to *e_max* -- the loosest
        state (pcf or kN/m^3).  Same units as *gamma_d*.

    Returns
    -------
    float
        Relative density as a percentage (%).

    Raises
    ------
    ValueError
        If *gamma_d_max* equals *gamma_d_min*, or if *gamma_d* is zero
        or negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-2, p. 141.
    """
    if gamma_d <= 0.0:
        raise ValueError("gamma_d must be positive.")
    if gamma_d_max <= gamma_d_min:
        raise ValueError("gamma_d_max must be greater than gamma_d_min.")
    return (
        (gamma_d_max / gamma_d)
        * ((gamma_d - gamma_d_min) / (gamma_d_max - gamma_d_min))
        * 100.0
    )


def cyclic_stress_ratio(tau_cyc: float, sigma_v_eff: float) -> float:
    """Cyclic stress ratio for cyclic direct simple shear tests (Equation 3-3).

    The cyclic stress ratio (CSR) is the ratio of the applied peak cyclic
    shear stress to the vertical effective consolidation stress.  It is used
    to specify loading in cyclic direct simple shear (CYCDSS) tests and is
    a key parameter in liquefaction triggering analyses.

    .. math::
        CSR = \\frac{\\tau_{cyc}}{\\sigma'_v}

    Parameters
    ----------
    tau_cyc : float
        Applied peak cyclic shear stress (psf, kPa, or any consistent
        stress unit).
    sigma_v_eff : float
        Vertical effective consolidation stress (same stress unit as
        *tau_cyc*).

    Returns
    -------
    float
        Cyclic stress ratio (dimensionless).

    Raises
    ------
    ValueError
        If *sigma_v_eff* is zero or negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-3, p. 160.
    """
    if sigma_v_eff <= 0.0:
        raise ValueError("sigma_v_eff must be positive.")
    return tau_cyc / sigma_v_eff


def load_increment_ratio(delta_sigma: float, sigma_0: float) -> float:
    """Load Increment Ratio for consolidation testing (Equation 3-4).

    Quantifies the change in load applied to a consolidation test specimen
    relative to the initial total stress.  An LIR of 1 corresponds to
    doubling the load.  For unloading, an LIR of -0.75 is often used
    (skipping the immediately previous load).  For reloading, an LIR of 4
    is commonly used.

    .. math::
        LIR = \\frac{\\Delta\\sigma}{\\sigma_0}

    Parameters
    ----------
    delta_sigma : float
        Change in applied stress (psf, kPa, or any consistent stress unit).
        Positive for loading, negative for unloading.
    sigma_0 : float
        Initial total stress on the specimen before the load increment
        (same stress unit as *delta_sigma*).

    Returns
    -------
    float
        Load increment ratio (dimensionless).

    Raises
    ------
    ValueError
        If *sigma_0* is zero or negative.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-4, p. 164.
    """
    if sigma_0 <= 0.0:
        raise ValueError("sigma_0 must be positive.")
    return delta_sigma / sigma_0


def hydraulic_conductivity_from_consolidation(
    c_v: float, m_v: float, gamma_w: float
) -> float:
    """Hydraulic conductivity from coefficient of consolidation (Equation 3-5).

    Computes the hydraulic conductivity (permeability) of a soil from data
    obtained during a one-dimensional consolidation test.  The coefficient
    of consolidation *c_v* is determined from time-rate data, and the
    coefficient of volumetric compressibility *m_v* is determined from the
    strain-versus-arithmetic-effective-stress curve at the stress where *c_v*
    is calculated.

    .. math::
        k = c_v \\cdot m_v \\cdot \\gamma_w

    Parameters
    ----------
    c_v : float
        Coefficient of consolidation (length^2 / time, e.g., cm^2/s or
        ft^2/day).
    m_v : float
        Coefficient of volumetric compressibility (1 / stress, e.g.,
        1/kPa or 1/psf).  Determined as the slope of the
        strain-versus-arithmetic-effective-stress plot.
    gamma_w : float
        Unit weight of water (force / length^3, e.g., 9.81 kN/m^3 or
        62.4 pcf).  Units must be consistent with *c_v* and *m_v* so that
        the result has dimensions of length / time.

    Returns
    -------
    float
        Hydraulic conductivity *k* (length / time, e.g., cm/s or ft/day).
        Units depend on the consistent unit system used for the inputs.

    Raises
    ------
    ValueError
        If any input is negative.

    Notes
    -----
    Ensure dimensional consistency among all three inputs.  For example,
    if *c_v* is in cm^2/s and *m_v* is in 1/(g/cm^2), then *gamma_w*
    should be in g/cm^3 so that *k* is returned in cm/s.

    References
    ----------
    UFC 3-220-10, Soil Mechanics, 1 Feb 2022, Change 1, 11 Mar 2025,
    Chapter 3, Equation 3-5, p. 169.
    """
    if c_v < 0.0:
        raise ValueError("c_v must be non-negative.")
    if m_v < 0.0:
        raise ValueError("m_v must be non-negative.")
    if gamma_w <= 0.0:
        raise ValueError("gamma_w must be positive.")
    return c_v * m_v * gamma_w
