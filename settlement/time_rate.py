"""
Time rate of consolidation — Terzaghi 1-D theory.

Computes the degree of consolidation U(t) and time to reach a given
degree of consolidation, using the Terzaghi time factor Tv.

All units are SI: kPa, meters, seconds (or years where noted).

References:
    Terzaghi, K. (1925) — Erdbaumechanik
    USACE EM 1110-1-1904, Chapter 6
    FHWA GEC-6 (FHWA-IF-02-054), Section 8.4.3
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


def degree_of_consolidation(Tv: float) -> float:
    """Compute average degree of consolidation U from time factor Tv.

    Uses the approximate relationships:
        For U < 60%: Tv = (pi/4) * (U/100)²  →  U = 100*sqrt(4*Tv/pi)
        For U >= 60%: Tv = -0.9332*log10(1-U/100) - 0.0851

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
    FHWA GEC-6, Eq 8-15 and 8-16.
    """
    if Tv <= 0:
        return 0.0

    # Check if in the U < 60% range
    # At U=60%, Tv = (pi/4)*(0.6)^2 = 0.2827
    Tv_60 = math.pi / 4.0 * 0.6**2  # ≈ 0.2827

    if Tv <= Tv_60:
        U = 100.0 * math.sqrt(4.0 * Tv / math.pi)
    else:
        # Solve Tv = -0.9332*log10(1 - U/100) - 0.0851
        # 1 - U/100 = 10^(-(Tv + 0.0851)/0.9332)
        exponent = -(Tv + 0.0851) / 0.9332
        one_minus_U = 10.0**exponent
        U = (1.0 - one_minus_U) * 100.0

    return min(U, 100.0)


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
    """
    if U_target <= 0:
        return 0.0
    if U_target >= 100:
        raise ValueError("Cannot compute time for 100% consolidation (takes infinite time)")
    if cv <= 0:
        raise ValueError(f"cv must be positive, got {cv}")

    U_frac = U_target / 100.0

    if U_target < 60:
        Tv = math.pi / 4.0 * U_frac**2
    else:
        Tv = -0.9332 * math.log10(1.0 - U_frac) - 0.0851

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
