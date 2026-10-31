"""
Time rate of consolidation — Terzaghi 1-D theory.

Computes the degree of consolidation U(t) and time to reach a given
degree of consolidation, using the Terzaghi time factor Tv.

All units are SI: kPa, meters, seconds (or years where noted).

References:
    Terzaghi, K. (1925) — Erdbaumechanik
    UFS 3-220-10 (2026), Eq. 5-15 and Fig. 5-16 (p. 245)
    FHWA NHI-06-088, Eq. 7-8 and Table 7-4 (pp. 7-31, 7-32)
    FHWA GEC-6 (FHWA-SA-02-054), Sec. 5.3.3.1 (no procedure; it refers to Cheney & Chassie 2000)
"""

import math
from typing import Optional


def time_factor(cv: float, t: float, Hdr: float) -> float:
    """Compute Terzaghi time factor Tv.

    Tv = cv * t / Hdr²

    Parameters
    ----------
    cv : float
        Coefficient of consolidation (m²/year).
    t : float
        Time (years).
    Hdr : float
        Longest drainage path (m). For double drainage, Hdr = H/2.
        For single drainage, Hdr = H.

    Returns
    -------
    float
        Time factor Tv (dimensionless).
    """
    if Hdr <= 0:
        raise ValueError(f"Drainage path must be positive, got {Hdr}")
    return cv * t / Hdr**2


#: The time factor where UFS 3-220-10 Fig. 5-16 changes from one of its
#: approximations to the other.
T_SWITCH = 0.2


def degree_of_consolidation(Tv: float) -> float:
    """Compute average degree of consolidation U from time factor Tv,
    by the approximations UFS 3-220-10 Fig. 5-16 (p. 245) prints:

        For T <= 0.2:  U = (2 / sqrt(pi)) sqrt(T)
        For T > 0.2:   U = 1 - (8 / pi^2) exp(-pi^2 T / 4)

    (FHWA NHI-06-088 Table 7-4, p. 7-32, tabulates the same relation.)

    Parameters
    ----------
    Tv : float
        Time factor (dimensionless).

    Returns
    -------
    float
        Average degree of consolidation U (percent, 0–100).

    References
    ----------
    UFS 3-220-10 Eq. 5-15 and Fig. 5-16 (p. 245).
    """
    if Tv <= 0:
        return 0.0
    if Tv <= T_SWITCH:
        U = 2.0 / math.sqrt(math.pi) * math.sqrt(Tv)
    else:
        U = 1.0 - 8.0 / math.pi**2 * math.exp(-math.pi**2 * Tv / 4.0)
    return min(100.0 * U, 100.0)


def time_for_consolidation(U_target: float, cv: float, Hdr: float) -> float:
    """Compute time to reach a target degree of consolidation.

    Parameters
    ----------
    U_target : float
        Target degree of consolidation (percent, 0-100).
    cv : float
        Coefficient of consolidation (m²/year).
    Hdr : float
        Longest drainage path (m).

    Returns
    -------
    float
        Time to reach U_target (years).

    Notes
    -----
    The inverse of :func:`degree_of_consolidation`: T = (pi/4) U^2 up to
    T = 0.2, then T = -(4/pi^2) ln[(pi^2/8)(1 - U)] (UFS 3-220-10
    Fig. 5-16). The two printed approximations differ by 0.0005 in U at
    T = 0.2; a U between them is reached at T = 0.2.
    """
    if U_target <= 0:
        return 0.0
    if U_target >= 100:
        raise ValueError("Cannot compute time for 100% consolidation (takes infinite time)")
    if cv <= 0:
        raise ValueError(f"cv must be positive, got {cv}")

    U_frac = U_target / 100.0
    Tv = math.pi / 4.0 * U_frac**2
    if Tv > T_SWITCH:
        Tv = max(T_SWITCH, -4.0 / math.pi**2
                 * math.log(math.pi**2 / 8.0 * (1.0 - U_frac)))

    return Tv * Hdr**2 / cv


def settlement_at_time(S_ultimate: float, cv: float, Hdr: float,
                       t: float) -> float:
    """Compute settlement at time t.

    S(t) = U(t) * S_ultimate

    Parameters
    ----------
    S_ultimate : float
        Ultimate (final) consolidation settlement (m).
    cv : float
        Coefficient of consolidation (m²/year).
    Hdr : float
        Longest drainage path (m).
    t : float
        Time since load application (years).

    Returns
    -------
    float
        Settlement at time t (m).
    """
    Tv = time_factor(cv, t, Hdr)
    U = degree_of_consolidation(Tv) / 100.0
    return U * S_ultimate
