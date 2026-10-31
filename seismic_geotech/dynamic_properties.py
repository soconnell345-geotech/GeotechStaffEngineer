"""
Small-strain dynamic soil properties: Gmax correlations and the
Ishibashi & Zhang (1993) modulus-reduction / damping curves.

Native implementations of the published correlations (no third-party
correlation library). Each function cites its source; all use SI units
(kPa, m/s, kN/m3) unless noted. CPT-based forms take qc in MPa (the
customary CPT reporting unit) and convert internally.

References:
    - Rix & Stokoe (1991) — Gmax from CPT, uncemented silica sand
    - Mayne & Rix (1993) — Gmax from CPT and void ratio, clays
    - Hardin (1978) — Gmax from void ratio, OCR and mean stress,
      fine-grained soils
    - Andersen (2015) — Gmax for clays from PI, OCR, overburden
    - Ishibashi & Zhang (1993) — G/Gmax and damping vs strain, PI, stress

Checked against the print held (GM-293, module_work/gm293/CHECK.md),
FHWA GEC-5 (FHWA-NHI-16-072) Sec. 8.6: Rix & Stokoe (1991) and Mayne &
Rix (1993) as its Table 8-7 (p. 8-21) prints them; Hardin (1978) as its
Eq. 8.14 with Eq. 8.8 and Table 8-4 (pp. 8-16 to 8-18). Andersen (2015)
and Ishibashi & Zhang (1993) are printed on no held page (GEC-5 p. 8-23
names Ishibashi & Zhang only) and are not checked.
"""

import math

__all__ = [
    "gmax_from_vs",
    "gmax_cpt_sand_rix_stokoe",
    "gmax_cpt_clay_mayne_rix",
    "gmax_hardin_black",
    "gmax_clay_andersen",
    "modulus_reduction_ishibashi_zhang",
]

_G = 9.81          # m/s2
_PA = 100.0        # atmospheric pressure, kPa


def gmax_from_vs(Vs: float, unit_weight: float, g: float = _G) -> dict:
    """Small-strain shear modulus from shear wave velocity: G = rho * Vs^2
    (FHWA GEC-5 Eq. 8.1).

    Parameters
    ----------
    Vs : float
        Shear wave velocity (m/s).
    unit_weight : float
        Bulk unit weight (kN/m3).
    g : float, optional
        Gravitational acceleration (m/s2), default 9.81.

    Returns
    -------
    dict
        {"rho_kg_m3", "Gmax_kPa"}.
    """
    if Vs < 0:
        raise ValueError(f"Vs must be non-negative, got {Vs}")
    if not 10.0 <= unit_weight <= 25.0:
        raise ValueError(f"unit_weight outside 10-25 kN/m3: {unit_weight}")
    rho = unit_weight / g * 1000.0            # kg/m3
    return {"rho_kg_m3": rho, "Gmax_kPa": rho * Vs ** 2 / 1000.0}


def gmax_cpt_sand_rix_stokoe(qc_MPa: float, sigma_vo_eff_kPa: float) -> dict:
    """Gmax for uncemented silica sand from CPT (Rix & Stokoe 1991).

    Gmax = 1634 * qc^0.25 * sigma_v'^0.375   (Gmax, qc and sigma_v' in
    kPa; quartz sand), FHWA GEC-5 Table 8-7 (p. 8-21)

    Parameters
    ----------
    qc_MPa : float
        Cone tip resistance (MPa).
    sigma_vo_eff_kPa : float
        Vertical effective stress (kPa).

    Returns
    -------
    dict
        {"Gmax_kPa"}.
    """
    if qc_MPa < 0 or sigma_vo_eff_kPa < 0:
        raise ValueError("qc_MPa and sigma_vo_eff_kPa must be non-negative")
    qc_kPa = qc_MPa * 1000.0
    return {"Gmax_kPa": 1634.0 * qc_kPa ** 0.25 * sigma_vo_eff_kPa ** 0.375}


def gmax_cpt_clay_mayne_rix(qt_MPa: float, void_ratio: float) -> dict:
    """Gmax for clay from CPT tip resistance and void ratio (Mayne & Rix
    1993), as FHWA GEC-5 Table 8-7 (p. 8-21) prints it:

    Gmax = 49.4 * qt^0.695 * e^-1.130   (Gmax and qt in MPa; clay)

    Until GM-293 this was Gmax = 2.78 qc^1.335 (kPa), a form printed on no
    held page, with no void ratio.

    Parameters
    ----------
    qt_MPa : float
        Corrected cone tip resistance qt (MPa).
    void_ratio : float
        Void ratio e.

    Returns
    -------
    dict
        {"Gmax_kPa"}.
    """
    if qt_MPa < 0:
        raise ValueError(f"qt_MPa must be non-negative, got {qt_MPa}")
    if void_ratio <= 0:
        raise ValueError(f"void_ratio must be positive, got {void_ratio}")
    return {"Gmax_kPa":
            49.4 * qt_MPa ** 0.695 * void_ratio ** -1.130 * 1000.0}


#: FHWA GEC-5 Table 8-4 (p. 8-18; Hardin 1978): the OCR exponent k by PI
#: (percent); "> 100" takes 0.50.
_HARDIN_K = ((0.0, 0.0), (20.0, 0.18), (40.0, 0.30), (60.0, 0.41),
             (80.0, 0.48), (100.0, 0.50))


def hardin_k(PI_pct: float) -> float:
    """The exponent k of OCR in Hardin (1978), FHWA GEC-5 Table 8-4
    (p. 8-18), by straight lines between its rows; 0.50 above PI 100."""
    if PI_pct < 0:
        raise ValueError(f"PI_pct must be non-negative, got {PI_pct}")
    if PI_pct >= 100.0:
        return 0.50
    for (p0, k0), (p1, k1) in zip(_HARDIN_K, _HARDIN_K[1:]):
        if p0 <= PI_pct <= p1:
            return k0 + (PI_pct - p0) / (p1 - p0) * (k1 - k0)
    return 0.50


def gmax_hardin_black(
    sigma_m_eff_kPa: float,
    void_ratio: float,
    coefficient_B: float = 625.0,
    p_ref_kPa: float = _PA,
    OCR: float = 1.0,
    PI_pct: float = None,
) -> dict:
    """Gmax of a fine-grained soil from void ratio, OCR and mean effective
    stress, Hardin (1978), FHWA GEC-5 Eq. 8.14 (p. 8-18):

    Gmax = A F(e) OCR^k pa^(1 - n) sigma_m'^n,  F(e) = 1 / (0.3 + 0.7 e^2)
    (Eq. 8.8), n = 0.5

    "Hardin (1978) found A = 625 and n = 0.5, and the void ratio function
    from Equation 8.8 for fine-grained soils" (GEC-5 p. 8-18); k by PI
    from Table 8-4. The name keeps the tool's "hardin_black": GEC-5 gives
    Hardin and Black (1968) the other void ratio functions, Eqs. 8.9 and
    8.10 (Table 8-2, coarse-grained soils), which this function does not
    use.

    Until GM-293 A defaulted to 875, the value Taborda et al. (2019) fitted
    to a dense marine sand (printed on no held page), and the function was
    labelled Hardin & Black (1968) with no OCR.

    Parameters
    ----------
    sigma_m_eff_kPa : float
        Mean effective stress p' (kPa).
    void_ratio : float
        Initial void ratio e0.
    coefficient_B : float, optional
        The coefficient A; default 625 (Hardin 1978, fine-grained soils).
    p_ref_kPa : float, optional
        Atmospheric pressure pa (kPa), default 100.
    OCR : float, optional
        Overconsolidation ratio, default 1.
    PI_pct : float, optional
        Plasticity index (percent), for k (Table 8-4); needed when OCR is
        not 1.

    Returns
    -------
    dict
        {"Gmax_kPa", "k"}.
    """
    if sigma_m_eff_kPa < 0:
        raise ValueError(f"sigma_m_eff_kPa must be non-negative: {sigma_m_eff_kPa}")
    if not 0.1 <= void_ratio <= 3.0:
        raise ValueError(f"void_ratio outside 0.1-3.0: {void_ratio}")
    if OCR < 1.0:
        raise ValueError(f"OCR must be at least 1, got {OCR}")
    if OCR != 1.0 and PI_pct is None:
        raise ValueError("OCR is not 1: give PI_pct, for the exponent k of "
                         "OCR (FHWA GEC-5 Table 8-4)")
    k = hardin_k(PI_pct) if PI_pct is not None else 0.0
    return {"Gmax_kPa": coefficient_B * p_ref_kPa / (0.3 + 0.7 * void_ratio ** 2)
            * OCR ** k * math.sqrt(sigma_m_eff_kPa / p_ref_kPa),
            "k": k}


def gmax_clay_andersen(
    PI_pct: float,
    OCR: float,
    sigma_vo_eff_kPa: float,
    atmospheric_pressure_kPa: float = _PA,
) -> dict:
    """Gmax for cohesive soils from PI, OCR and overburden (Andersen 2015).

    Gmax / sigma_ref' = (30 + 75 / (Ip/100 + 0.03)) * OCR^0.5
    sigma_ref' = Pa * (sigma_v0' / Pa)^0.9

    Parameters
    ----------
    PI_pct : float
        Plasticity index (percent), 0-160.
    OCR : float
        Overconsolidation ratio, 1-40.
    sigma_vo_eff_kPa : float
        Vertical effective stress (kPa).

    Returns
    -------
    dict
        {"sigma_ref_kPa", "Gmax_kPa"}.

    References
    ----------
    Andersen, K.H. (2015). "Cyclic soil parameters for offshore foundation
    design." Frontiers in Offshore Geotechnics III.
    """
    if not 0.0 <= PI_pct <= 160.0:
        raise ValueError(f"PI_pct outside 0-160: {PI_pct}")
    if not 1.0 <= OCR <= 40.0:
        raise ValueError(f"OCR outside 1-40: {OCR}")
    if sigma_vo_eff_kPa <= 0:
        raise ValueError(f"sigma_vo_eff_kPa must be positive: {sigma_vo_eff_kPa}")
    sigma_ref = atmospheric_pressure_kPa * \
        (sigma_vo_eff_kPa / atmospheric_pressure_kPa) ** 0.9
    ratio = (30.0 + 75.0 / (PI_pct / 100.0 + 0.03)) * math.sqrt(OCR)
    return {"sigma_ref_kPa": sigma_ref, "Gmax_kPa": ratio * sigma_ref}


def modulus_reduction_ishibashi_zhang(
    strain_pct: float,
    PI_pct: float,
    sigma_m_eff_kPa: float,
) -> dict:
    """G/Gmax and damping vs cyclic strain (Ishibashi & Zhang 1993).

    G/Gmax = K(gamma, PI) * (sigma_m')^(m(gamma, PI) - m0)

    with the published K, m and n(PI) piecewise expressions, and the
    companion damping fit

    D = 0.333 * (1 + exp(-0.0145 PI^1.3)) / 2
        * (0.586 (G/Gmax)^2 - 1.547 (G/Gmax) + 1)

    Use PI = 0 for cohesionless soils.

    Parameters
    ----------
    strain_pct : float
        Cyclic shear strain amplitude in PERCENT (e.g. 0.01 for 1e-4
        strain). Valid roughly 1e-5 to 1 percent.
    PI_pct : float
        Plasticity index (percent).
    sigma_m_eff_kPa : float
        Mean effective confining stress (kPa).

    Returns
    -------
    dict
        {"G_over_Gmax", "K", "m", "n", "damping_pct"}.
        G/Gmax is clipped to at most 1.0.

    References
    ----------
    Ishibashi, I. & Zhang, X. (1993). "Unified dynamic shear moduli and
    damping ratios of sand and clay." Soils and Foundations, 33(1).
    """
    if strain_pct <= 0:
        raise ValueError(f"strain_pct must be positive, got {strain_pct}")
    if PI_pct < 0:
        raise ValueError(f"PI_pct must be non-negative, got {PI_pct}")
    if sigma_m_eff_kPa <= 0:
        raise ValueError(f"sigma_m_eff_kPa must be positive: {sigma_m_eff_kPa}")
    gamma = strain_pct / 100.0                 # decimal strain
    if PI_pct == 0.0:
        n = 0.0
    elif PI_pct <= 15.0:
        n = 3.37e-6 * PI_pct ** 1.404
    elif PI_pct <= 70.0:
        n = 7.0e-7 * PI_pct ** 1.976
    else:
        n = 2.7e-5 * PI_pct ** 1.115
    K = 0.5 * (1.0 + math.tanh(math.log(((0.000102 + n) / gamma) ** 0.492)))
    m = 0.272 * (1.0 - math.tanh(math.log((0.000556 / gamma) ** 0.4))) \
        * math.exp(-0.0145 * PI_pct ** 1.3)
    g_ratio = min(K * sigma_m_eff_kPa ** m, 1.0)
    damping = 0.333 * (1.0 + math.exp(-0.0145 * PI_pct ** 1.3)) / 2.0 \
        * (0.586 * g_ratio ** 2 - 1.547 * g_ratio + 1.0)
    return {"G_over_Gmax": g_ratio, "K": K, "m": m, "n": n,
            "damping_pct": damping * 100.0}
