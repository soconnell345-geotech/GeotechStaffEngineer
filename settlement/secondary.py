"""
Secondary compression (creep) settlement.

Computes the time-dependent settlement that occurs after primary
consolidation is essentially complete.

All units are SI: kPa, meters.

References:
    FHWA NHI-06-088, Eq. 7-10 (p. 7-39)
    UFS 3-220-10 (2026), Eq. 5-18 (p. 254) and Fig. 5-24 (p. 255)
"""

import math


def secondary_settlement(C_alpha: float, H: float,
                         t1: float, t2: float,
                         e0: float) -> float:
    """Compute secondary compression settlement.

    Ss = C_alpha / (1 + e0) * H * log10(t2 / t1)
       = C_alpha_epsilon * H * log10(t2 / t1)

    as FHWA NHI-06-088 Eq. 7-10 and UFS 3-220-10 Eq. 5-18 print it, where
    C_alpha_epsilon = C_alpha / (1 + e0) is the modified secondary
    compression index.

    Parameters
    ----------
    C_alpha : float
        Secondary compression index (dimensionless). This is the
        slope of e vs log10(t) from the oedometer test.
        Typical values: 0.003–0.03 for inorganic clays,
        0.03–0.1 for organic/highly plastic clays.
    H : float
        Layer thickness (m): the initial thickness H0 (UFS Eq. 5-18).
    t1 : float
        Time at end of primary consolidation (years): NHI-06-088 takes
        the time when about 90 % of primary consolidation has occurred
        (Eq. 7-10); UFS the time required to finish it (t_p, Eq. 5-18).
    t2 : float
        Time at which secondary settlement is desired (years), from the
        load (UFS: time after loading; NHI-06-088: the service life or
        any other time of interest).
    e0 : float
        Initial void ratio (UFS Eq. 5-18), which converts C_alpha to
        C_alpha_epsilon. No default (GM-285: it was 1.0, printed
        nowhere).

    Returns
    -------
    float
        Secondary compression settlement (m).

    References
    ----------
    FHWA NHI-06-088 Eq. 7-10 (p. 7-39); UFS 3-220-10 Eq. 5-18 (p. 254),
    worked in its Fig. 5-24 (p. 255). FHWA GEC-6 Sec. 5.3.3.2 (p. 85)
    discusses secondary compression and gives no equation.
    """
    if t2 <= t1:
        return 0.0
    if C_alpha <= 0:
        return 0.0
    if t1 <= 0:
        raise ValueError(f"t1 must be positive, got {t1}")

    C_alpha_eps = C_alpha / (1.0 + e0)
    return C_alpha_eps * H * math.log10(t2 / t1)
