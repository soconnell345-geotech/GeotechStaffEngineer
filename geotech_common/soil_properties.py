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
        # of their chart relating N to phi
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
