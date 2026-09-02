"""
Common soil property correlations.

Provides empirical correlations between field test results (primarily SPT N-values)
and engineering soil parameters. These are approximate relationships for preliminary
design and should be verified with laboratory testing for final design.

References:
    - Peck, Hanson & Thornburn (1974) — SPT to phi
    - Meyerhof (1956) — SPT to phi
    - Terzaghi & Peck (1967) — SPT to cu
    - Kulhawy & Mayne (1990) — SPT to relative density
"""

import math
import warnings


def spt_to_phi(N60: float, method: str = "peck") -> float:
    """Estimate drained friction angle from corrected SPT blow count.

    Parameters
    ----------
    N60 : float
        SPT blow count corrected for 60% energy ratio.
        Typical range: 0–100.
    method : str, optional
        Correlation method. Options:
        - "peck" (default): Peck, Hanson & Thornburn (1974)
        - "meyerhof": Meyerhof (1956) — phi = 25 + 0.3*N60 (capped)

    Returns
    -------
    float
        Estimated friction angle phi (degrees).

    Raises
    ------
    ValueError
        If N60 < 0 or method is unknown.

    References
    ----------
    Peck, Hanson & Thornburn (1974), "Foundation Engineering", Table 10-3.
    Meyerhof, G.G. (1956), "Penetration Tests and Bearing Capacity of
    Cohesionless Soils", JSMFE, ASCE, Vol. 82, No. SM1.
    """
    if N60 < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N60}")
    if N60 > 100:
        warnings.warn(f"SPT N60={N60} is unusually high; correlation may not be reliable")

    method = method.lower()
    if method == "peck":
        # Peck, Hanson & Thornburn (1974) — piecewise linear approximation
        # of their chart relating N to phi.
        # Source basis: HAND-DIGITIZATION of the PHT N-vs-phi chart. The
        # breakpoints (26/28/31/38/42 deg) trace the chart's zones, but the
        # intermediate SLOPES are author-chosen to interpolate smoothly and were
        # NOT verified per-value against PHT 1974 Table 10-3 / the original
        # figure in hand. Intended for preliminary design; candidate for wiki
        # verification against Peck, Hanson & Thornburn (1974) Table 10-3.
        if N60 <= 0:
            return 26.0
        elif N60 <= 4:
            return 26.0 + N60 * 0.5  # 26-28 for very loose
        elif N60 <= 10:
            return 28.0 + (N60 - 4) * 0.5  # 28-31 for loose
        elif N60 <= 30:
            return 31.0 + (N60 - 10) * 0.35  # 31-38 for medium
        elif N60 <= 50:
            return 38.0 + (N60 - 30) * 0.2  # 38-42 for dense
        else:
            return min(42.0 + (N60 - 50) * 0.1, 50.0)  # cap at 50°

    elif method == "meyerhof":
        # Meyerhof (1956): phi ≈ 25 + 0.3*N60 (approximately)
        phi = 25.0 + 0.3 * N60
        return min(phi, 50.0)  # cap at 50°

    else:
        raise ValueError(f"Unknown method '{method}'. Options: 'peck', 'meyerhof'")


def spt_to_cu(N60: float, method: str = "terzaghi_peck") -> float:
    """Estimate undrained shear strength from SPT blow count.

    Parameters
    ----------
    N60 : float
        SPT blow count corrected for 60% energy ratio.
    method : str, optional
        Correlation method. Options:
        - "terzaghi_peck" (default): cu = 6.25 * N60 (kPa)
        - "hara": cu = 29 * N60^0.72 (kPa) — Hara et al. (1971)

    Returns
    -------
    float
        Estimated undrained shear strength cu (kPa).

    References
    ----------
    Terzaghi & Peck (1967), cu/pa ≈ 0.0625*N (pa=100 kPa).
    Hara, A. et al. (1971), Soils and Foundations, Vol. 11, No. 3.
    """
    if N60 < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N60}")

    method = method.lower()
    if method == "terzaghi_peck":
        # cu ≈ 6.25 * N60 kPa (equivalent to cu/pa = 0.0625*N with pa=100 kPa)
        return 6.25 * N60

    elif method == "hara":
        # Hara et al. (1971): cu = 29 * N60^0.72 kPa
        if N60 == 0:
            return 0.0
        return 29.0 * N60**0.72

    else:
        raise ValueError(f"Unknown method '{method}'. Options: 'terzaghi_peck', 'hara'")


def spt_to_relative_density(N60: float, sigma_v: float = 100.0) -> float:
    """Estimate relative density from SPT blow count.

    Uses Kulhawy & Mayne (1990) correlation:
        Dr (%) = 100 * sqrt(N60 / (Cp * Ca * Cocr))
    Simplified as Dr = 100 * sqrt(N60 / (60 + 25*log10(sigma_v'/100)))

    For this simplified version:
        Dr = sqrt(N60 / 46) * 100  (approximate, for sigma_v' ~ 100 kPa)

    Parameters
    ----------
    N60 : float
        SPT blow count corrected for 60% energy ratio.
    sigma_v : float, optional
        Vertical effective stress at test depth (kPa). Default 100 kPa.

    Returns
    -------
    float
        Estimated relative density Dr (percent, 0–100).

    References
    ----------
    Kulhawy, F.H. & Mayne, P.W. (1990), "Manual on Estimating Soil Properties
    for Foundation Design", EPRI EL-6800.
    """
    if N60 < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N60}")
    if sigma_v <= 0:
        raise ValueError(f"Effective stress must be positive, got {sigma_v}")

    if N60 == 0:
        return 0.0

    # Kulhawy & Mayne simplified: Dr² = N60 / (Cd * sigma_v'/pa)
    # Cd ≈ 0.46 for OC=1, pa = 100 kPa
    pa = 100.0  # atmospheric pressure in kPa
    Dr = math.sqrt(N60 / (0.46 * sigma_v / pa)) * 100.0
    return min(Dr, 100.0)


def phi_to_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient.

    Ka = tan²(45° - phi/2)

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4.0 - phi_rad / 2.0) ** 2


def phi_to_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient.

    Kp = tan²(45° + phi/2)

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4.0 + phi_rad / 2.0) ** 2


def phi_to_K0(phi_deg: float) -> float:
    """At-rest earth pressure coefficient (Jaky's formula).

    K0 = 1 - sin(phi)

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).

    Returns
    -------
    float
        At-rest earth pressure coefficient K0.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return 1.0 - math.sin(phi_rad)


# ---------------------------------------------------------------------------
# SPT field corrections (raw field N  ->  N60  ->  (N1)60)
# ---------------------------------------------------------------------------

#: Rod-length correction C_R per Youd et al. (2001), Table 2.
_ROD_LENGTH_BANDS = (
    (3.0, 0.75),    # rod length < 3 m
    (4.0, 0.80),    # 3-4 m
    (6.0, 0.85),    # 4-6 m
    (10.0, 0.95),   # 6-10 m
    (float("inf"), 1.00),  # 10-30 m
)


def spt_energy_correction(
    N_field: float,
    energy_ratio_pct: float = 60.0,
    rod_length_m: float = 10.0,
    borehole_diameter_mm: float = 100.0,
    sampler_correction: float = 1.0,
) -> dict:
    """Correct a raw field SPT blow count to the standardized N60.

    N60 = N_field * CE * CB * CR * CS  (Youd et al. 2001, Table 2)

    Parameters
    ----------
    N_field : float
        Raw field SPT blow count N (blows/300 mm).
    energy_ratio_pct : float, optional
        Measured hammer energy ratio ER (percent of theoretical free-fall
        energy). CE = ER/60. Default 60 (already-standardized energy).
        Typical: donut hammer 45-60, safety hammer 55-72, automatic-trip
        80-100.
    rod_length_m : float, optional
        Rod length from hammer to sampler (m); sets CR per Youd et al.
        (2001) Table 2 bands (0.75 below 3 m rising to 1.0 beyond 10 m).
        Default 10.0 (CR = 1.0).
    borehole_diameter_mm : float, optional
        Borehole diameter; CB = 1.0 for 65-115 mm (default), 1.05 for
        150 mm, 1.15 for 200 mm.
    sampler_correction : float, optional
        CS: 1.0 for a standard sampler (default); 1.1-1.3 for a sampler
        used without liners (Youd et al. 2001).

    Returns
    -------
    dict
        Keys "CE", "CB", "CR", "CS", "N60".

    References
    ----------
    Youd et al. (2001). "Liquefaction resistance of soils: summary report
    from the 1996 NCEER and 1998 NCEER/NSF workshops." J. Geotech.
    Geoenviron. Eng., 127(10), Table 2.
    """
    if N_field < 0:
        raise ValueError(f"N_field must be non-negative, got {N_field}")
    if not 30.0 <= energy_ratio_pct <= 120.0:
        raise ValueError(
            f"energy_ratio_pct outside credible 30-120 range: {energy_ratio_pct}")
    if rod_length_m < 0:
        raise ValueError(f"rod_length_m must be non-negative, got {rod_length_m}")
    if not 1.0 <= sampler_correction <= 1.3:
        raise ValueError(
            f"sampler_correction (CS) must be 1.0-1.3, got {sampler_correction}")
    CE = energy_ratio_pct / 60.0
    if borehole_diameter_mm <= 115.0:
        CB = 1.0
    elif borehole_diameter_mm <= 150.0:
        CB = 1.05
    elif borehole_diameter_mm <= 200.0:
        CB = 1.15
    else:
        raise ValueError(
            f"borehole_diameter_mm above 200 mm not covered: {borehole_diameter_mm}")
    CR = next(cr for upper, cr in _ROD_LENGTH_BANDS if rod_length_m < upper)
    CS = sampler_correction
    return {"CE": CE, "CB": CB, "CR": CR, "CS": CS,
            "N60": N_field * CE * CB * CR * CS}


def spt_overburden_correction(
    N: float,
    sigma_vo_eff_kPa: float,
    method: str = "liao_whitman",
    CN_cap: float | None = None,
) -> dict:
    """Overburden correction C_N for granular soils: N1 = C_N * N.

    Pass the raw field N to get N1, or an energy-corrected N60 to get
    (N1)60 -- the correction is multiplicative either way.

    Parameters
    ----------
    N : float
        SPT blow count (field N or N60).
    sigma_vo_eff_kPa : float
        Vertical effective stress at the test depth (kPa).
    method : str, optional
        - "liao_whitman" (default): C_N = (Pa / sigma_v')^0.5 with
          Pa = 100 kPa (Liao & Whitman 1986); capped at 1.7 per
          Youd et al. (2001).
        - "iso": C_N = sqrt(98 / sigma_v') per EN ISO 22476-3 guidance;
          capped at 2.0 (values above ~1.5 used with caution).
    CN_cap : float, optional
        Override the default cap (1.7 liao_whitman / 2.0 iso).

    Returns
    -------
    dict
        Keys "CN", "N1".

    References
    ----------
    Liao, S. & Whitman, R. (1986). "Overburden correction factors for SPT
    in sand." J. Geotech. Eng., 112(3). Youd et al. (2001) Table 2 (cap).
    EN ISO 22476-3 (SPT).
    """
    if N < 0:
        raise ValueError(f"N must be non-negative, got {N}")
    if sigma_vo_eff_kPa <= 0:
        raise ValueError(
            f"sigma_vo_eff_kPa must be positive, got {sigma_vo_eff_kPa}")
    if method == "liao_whitman":
        CN = math.sqrt(100.0 / sigma_vo_eff_kPa)
        cap = 1.7 if CN_cap is None else CN_cap
    elif method == "iso":
        CN = math.sqrt(98.0 / sigma_vo_eff_kPa)
        cap = 2.0 if CN_cap is None else CN_cap
    else:
        raise ValueError(f"Unknown method '{method}' (liao_whitman | iso)")
    CN = min(CN, cap)
    return {"CN": CN, "N1": CN * N}


def spt_n1_60(
    N_field: float,
    sigma_vo_eff_kPa: float,
    energy_ratio_pct: float = 60.0,
    rod_length_m: float = 10.0,
    borehole_diameter_mm: float = 100.0,
    sampler_correction: float = 1.0,
    overburden_method: str = "liao_whitman",
) -> dict:
    """Full field-N -> (N1)60 correction chain (energy then overburden).

    Convenience wrapper: spt_energy_correction then
    spt_overburden_correction. Returns every intermediate factor so the
    calculation can be reported step by step.

    Returns
    -------
    dict
        Keys "CE", "CB", "CR", "CS", "N60", "CN", "N1_60".
    """
    e = spt_energy_correction(N_field, energy_ratio_pct, rod_length_m,
                              borehole_diameter_mm, sampler_correction)
    o = spt_overburden_correction(e["N60"], sigma_vo_eff_kPa,
                                  method=overburden_method)
    return {**e, "CN": o["CN"], "N1_60": o["N1"]}


# ---------------------------------------------------------------------------
# Bolton (1986) stress-dilatancy
# ---------------------------------------------------------------------------

def stress_dilatancy_bolton(
    relative_density: float,
    p_eff_kPa: float,
    Q: float = 10.0,
    R: float = 1.0,
    stress_condition: str = "triaxial",
) -> dict:
    """Bolton (1986) relative dilatancy index and peak-friction excess.

    I_R = D_r * (Q - ln p') - R

    phi_max - phi_crit = 3 * I_R   (triaxial)
    phi_max - phi_crit = 5 * I_R   (plane strain; = 0.8 * psi_max)
    (-d eps_v / d eps_1)_max = 0.3 * I_R

    Parameters
    ----------
    relative_density : float
        Relative density D_r as a FRACTION (0-1).
    p_eff_kPa : float
        Mean effective stress p' (kPa).
    Q, R : float, optional
        Bolton's mineralogy fitting constants (defaults 10 and 1 for
        silica sand).
    stress_condition : str, optional
        "triaxial" (default) or "plane_strain".

    Returns
    -------
    dict
        Keys "IR", "phi_excess_deg", "dilation_angle_deg",
        "max_dilation_rate". I_R is clipped at 0 (no dilation predicted
        for loose states / high stress); Bolton recommends 0 <= I_R <= 4.

    References
    ----------
    Bolton, M.D. (1986). "The strength and dilatancy of sands."
    Geotechnique, 36(1), 65-78.
    """
    if not 0.0 <= relative_density <= 1.0:
        raise ValueError(
            f"relative_density must be a 0-1 fraction, got {relative_density}")
    if p_eff_kPa <= 0:
        raise ValueError(f"p_eff_kPa must be positive, got {p_eff_kPa}")
    IR = relative_density * (Q - math.log(p_eff_kPa)) - R
    IR = max(IR, 0.0)
    if stress_condition == "triaxial":
        phi_excess = 3.0 * IR
        dilation_angle = phi_excess       # psi ~= dphi for triaxial
    elif stress_condition == "plane_strain":
        phi_excess = 5.0 * IR
        dilation_angle = phi_excess / 0.8  # dphi = 0.8 * psi_max
    else:
        raise ValueError(
            f"Unknown stress_condition '{stress_condition}' "
            "(triaxial | plane_strain)")
    return {"IR": IR, "phi_excess_deg": phi_excess,
            "dilation_angle_deg": dilation_angle,
            "max_dilation_rate": 0.3 * IR}
