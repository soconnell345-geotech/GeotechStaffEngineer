"""
Secondary compression (creep) settlement.

Computes the time-dependent settlement that occurs after primary
consolidation is essentially complete.

All units are SI: kPa, meters.

References:
    Mesri, G. (1973) — C_alpha / Cc concept
    USACE EM 1110-1-1904, Chapter 5
    FHWA GEC-6, Section 8.4.4
"""

import math


def secondary_settlement(C_alpha: float, H: float,
                         t1: float, t2: float,
                         e0: float = 1.0) -> float:
    """Compute secondary compression settlement.

    Ss = C_alpha_epsilon * H * log10(t2 / t1)

    where C_alpha_epsilon = C_alpha / (1 + e0) is the modified
    secondary compression index.

    Parameters
    ----------
    C_alpha : float
        Secondary compression index (dimensionless). This is the
        slope of e vs log10(t) from the oedometer test.
        Typical values: 0.003–0.03 for inorganic clays,
        0.03–0.1 for organic/highly plastic clays.
    H : float
        Layer thickness at end of primary consolidation (m).
    t1 : float
        Time at end of primary consolidation (years).
    t2 : float
        Time at which secondary settlement is desired (years).
    e0 : float, optional
        Void ratio at end of primary consolidation. Default 1.0.
        Used to convert C_alpha to C_alpha_epsilon.

    Returns
    -------
    float
        Secondary compression settlement (m).

    References
    ----------
    FHWA GEC-6, Eq 8-20.
    Mesri, G. (1973), "Coefficient of Secondary Compression",
    JSMFE, ASCE, Vol. 99, No. SM1.
    """
    if t2 <= t1:
        return 0.0
    if C_alpha <= 0:
        return 0.0
    if t1 <= 0:
        raise ValueError(f"t1 must be positive, got {t1}")

    C_alpha_eps = C_alpha / (1.0 + e0)
    return C_alpha_eps * H * math.log10(t2 / t1)
