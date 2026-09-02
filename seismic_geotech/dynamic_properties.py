"""
Small-strain dynamic soil properties: Gmax correlations and the
Ishibashi & Zhang (1993) modulus-reduction / damping curves.

Native implementations of the published correlations (no third-party
correlation library). Each function cites its source; all use SI units
(kPa, m/s, kN/m3) unless noted. CPT-based forms take qc in MPa (the
customary CPT reporting unit) and convert internally.

References:
    - Rix & Stokoe (1991) — Gmax from CPT, uncemented silica sand
    - Mayne & Rix (1993) — Gmax from CPT, clays (481-record regression)
    - Hardin & Black (1968) — Gmax from void ratio + mean stress
      (coefficient B per Taborda et al. 2019, PISA dense marine sand)
    - Andersen (2015) — Gmax for clays from PI, OCR, overburden
    - Ishibashi & Zhang (1993) — G/Gmax and damping vs strain, PI, stress
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
    """Small-strain shear modulus from shear wave velocity: G = rho * Vs^2.

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

    Gmax = 1634 * qc^0.25 * sigma_v'^0.375   (qc and sigma_v' in kPa)

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


def gmax_cpt_clay_mayne_rix(qc_MPa: float) -> dict:
    """Gmax for clay from CPT tip resistance (Mayne & Rix 1993).

    Gmax = 2.78 * qc^1.335   (both in kPa; 481-record worldwide regression,
    Gmax ranging ~0.7-800 MPa)

    Parameters
    ----------
    qc_MPa : float
        Cone tip resistance (MPa).

    Returns
    -------
    dict
        {"Gmax_kPa"}.
    """
    if qc_MPa < 0:
        raise ValueError(f"qc_MPa must be non-negative, got {qc_MPa}")
    return {"Gmax_kPa": 2.78 * (qc_MPa * 1000.0) ** 1.335}


def gmax_hardin_black(
    sigma_m_eff_kPa: float,
    void_ratio: float,
    coefficient_B: float = 875.0,
    p_ref_kPa: float = _PA,
) -> dict:
    """Gmax from void ratio and mean effective stress (Hardin & Black 1968).

    Gmax = B * p_ref / (0.3 + 0.7 e^2) * sqrt(p' / p_ref)

    Parameters
    ----------
    sigma_m_eff_kPa : float
        Mean effective stress p' (kPa).
    void_ratio : float
        Initial void ratio e0.
    coefficient_B : float, optional
        Calibration coefficient; default 875 per Taborda et al. (2019),
        calibrated to dense marine sand for the PISA monopile project.
        Recalibrate for site-specific work.
    p_ref_kPa : float, optional
        Reference stress (kPa), default 100.

    Returns
    -------
    dict
        {"Gmax_kPa"}.
    """
    if sigma_m_eff_kPa < 0:
        raise ValueError(f"sigma_m_eff_kPa must be non-negative: {sigma_m_eff_kPa}")
    if not 0.1 <= void_ratio <= 3.0:
        raise ValueError(f"void_ratio outside 0.1-3.0: {void_ratio}")
    return {"Gmax_kPa": coefficient_B * p_ref_kPa / (0.3 + 0.7 * void_ratio ** 2)
            * math.sqrt(sigma_m_eff_kPa / p_ref_kPa)}


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
