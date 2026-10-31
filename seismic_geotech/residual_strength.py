"""
Post-liquefaction residual strength estimation.

Implements:
- Seed & Harder (1990) lower-bound relationship
- Idriss & Boulanger (2007) chart, as GEC-12 Vol I Fig. 7-78 prints it
  (p. 383): both of its curves, read off the page

All units are SI: kPa.

References:
    Seed & Harder (1990), H. Bolton Seed Memorial Symposium
    Idriss & Boulanger (2007), as printed in FHWA GEC-12 Vol I
    (FHWA-NHI-16-009), Fig. 7-78 (p. 383), Table 7-31 and Eq. 7-80 (p. 384)
"""


def Sr_seed_harder(N160cs: float) -> float:
    """Residual strength per Seed & Harder (1990) lower bound.

    Piecewise linear approximation of the lower-bound curve.

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count (N1)60cs.

    Returns
    -------
    float
        Residual undrained strength Sr (kPa).

    References
    ----------
    Seed & Harder (1990), Fig. 1 lower bound
    """
    # Piecewise linear approximation of lower-bound curve
    # (N160cs, Sr_kPa) control points
    _points = [
        (0, 0),
        (4, 2.5),
        (8, 5.0),
        (12, 10.0),
        (16, 15.0),
        (20, 25.0),
        (24, 35.0),
    ]

    if N160cs <= 0:
        return 0.0
    if N160cs >= _points[-1][0]:
        return _points[-1][1]

    for i in range(len(_points) - 1):
        x0, y0 = _points[i]
        x1, y1 = _points[i + 1]
        if x0 <= N160cs <= x1:
            frac = (N160cs - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)

    return _points[-1][1]


# GEC-12 Vol I Fig. 7-78 (p. 383; Idriss and Boulanger 2007): s_r / sigma'_vo
# against (N1)60-e, read off the figure's image by
# module_work/gm293/trace_fig7_78.py (GM-293; 1 px = 0.046 blows and
# 0.0009 in the ratio). Until GM-293 the module used
# exp(N/16 + (N/21.2)^3 - 3.0) capped at 0.6, which no held page prints and
# which matches neither curve (0.050 at N 0, 0.208 at N 16, 0.40 at N 20).
#
# "Recommended curve for conditions where void redistribution effects could
# be significant": solid, then dashed. Its label hides it from N 18.9 to
# 24.4; the straight line between those points stands in there (up to about
# 0.01 above the convex curve). It leaves the chart at 0.4 at N 29.1.
_IB_SIGNIFICANT = (
    (0.00, 0.0308),
    (1.00, 0.0370),
    (2.00, 0.0429),
    (3.00, 0.0487),
    (4.00, 0.0542),
    (5.00, 0.0598),
    (6.00, 0.0648),
    (7.00, 0.0716),
    (8.00, 0.0776),
    (9.00, 0.0845),
    (10.00, 0.0907),
    (11.00, 0.0979),
    (12.00, 0.1055),
    (13.00, 0.1132),
    (14.00, 0.1218),
    (15.00, 0.1306),
    (16.00, 0.1401),
    (17.00, 0.1503),
    (18.00, 0.1612),
    (18.90, 0.1713),
    (24.39, 0.2596),
    (25.21, 0.2779),
    (25.97, 0.2971),
    (26.69, 0.3172),
    (27.34, 0.3373),
    (27.99, 0.3588),
    (28.57, 0.3804),
    (29.00, 0.3972),
    (29.07, 0.4000),
)
# "Recommended curve for conditions where void redistribution effects are
# expected to be negligible": leaves the solid line at N 9, the same curve
# below it. Its label hides it from 0.265 to 0.342 (a straight line
# there). It leaves the chart at 0.4 at N 17.1.
_IB_NEGLIGIBLE = _IB_SIGNIFICANT[:10] + (
    (9.90, 0.0968),
    (11.08, 0.1100),
    (11.66, 0.1200),
    (12.24, 0.1300),
    (12.73, 0.1400),
    (13.18, 0.1500),
    (13.54, 0.1600),
    (13.86, 0.1700),
    (14.14, 0.1800),
    (14.41, 0.1900),
    (14.63, 0.2000),
    (14.84, 0.2100),
    (15.04, 0.2200),
    (15.19, 0.2300),
    (15.40, 0.2400),
    (15.53, 0.2500),
    (15.67, 0.2600),
    (16.62, 0.3457),
    (16.81, 0.3673),
    (16.99, 0.3888),
    (17.08, 0.4000),
)

#: The curves of Fig. 7-78, by the void redistribution they are drawn for.
VOID_REDISTRIBUTION = {"significant": _IB_SIGNIFICANT,
                       "negligible": _IB_NEGLIGIBLE}


def Sr_ratio_idriss_boulanger(N160cs: float,
                              void_redistribution: str = "significant"
                              ) -> float:
    """s_r / sigma'_vo read off GEC-12 Vol I Fig. 7-78 (p. 383; Idriss and
    Boulanger 2007), by straight lines between the points read.

    Parameters
    ----------
    N160cs : float
        Equivalent clean sand blow count (N1)60-e: (N1)60 + Ncorr, GEC-12
        Eq. 7-80, with Ncorr from Table 7-31 by the percent passing the
        No. 200 sieve (0-9 %: 0; 10-24: 1; 25-49: 2; 50-74: 4; > 75: 5).
    void_redistribution : str
        Which of the two recommended curves: "significant" (void
        redistribution effects could be significant; the lower curve) or
        "negligible" (expected to be negligible).

    Returns
    -------
    float
        s_r / sigma'_vo.

    Raises
    ------
    ValueError
        Where the curve has left the chart (s_r / sigma'_vo above 0.4: N
        beyond 29.1 on the "significant" curve, 17.1 on the "negligible"
        one), for a negative N, or an unknown curve.
    """
    pts = VOID_REDISTRIBUTION.get(void_redistribution)
    if pts is None:
        raise ValueError(
            f"void_redistribution must be 'significant' or 'negligible' "
            f"(the two curves of GEC-12 Fig. 7-78), got "
            f"'{void_redistribution}'")
    if N160cs < 0:
        raise ValueError(f"(N1)60-e must not be negative, got {N160cs}")
    if N160cs > pts[-1][0]:
        raise ValueError(
            f"(N1)60-e = {N160cs:g} is beyond GEC-12 Fig. 7-78: the curve "
            f"for void redistribution {void_redistribution} leaves the chart "
            f"at s_r / sigma'_vo = 0.4 at (N1)60-e = {pts[-1][0]:g}, so no "
            f"value is read above it")
    for (n0, r0), (n1, r1) in zip(pts, pts[1:]):
        if n0 <= N160cs <= n1:
            return r0 + (N160cs - n0) / (n1 - n0) * (r1 - r0)
    return pts[-1][1]


def Sr_idriss_boulanger(N160cs: float, sigma_v_eff: float,
                        void_redistribution: str = "significant") -> float:
    """Residual strength per Idriss & Boulanger (2007), GEC-12 Vol I
    Fig. 7-78 (p. 383): s_r = (s_r / sigma'_vo) sigma'_vo, the ratio read
    off the chart (:func:`Sr_ratio_idriss_boulanger`).

    Parameters
    ----------
    N160cs : float
        Equivalent clean sand blow count (N1)60-e (GEC-12 Eq. 7-80).
    sigma_v_eff : float
        Pre-earthquake vertical effective stress sigma'_vo (kPa).
    void_redistribution : str
        "significant" (default; the lower curve) or "negligible".

    Returns
    -------
    float
        Residual undrained strength Sr (kPa).
    """
    if sigma_v_eff <= 0:
        return 0.0
    return Sr_ratio_idriss_boulanger(N160cs, void_redistribution) * sigma_v_eff


def post_liquefaction_strength(N160cs: float,
                               sigma_v_eff: float = None,
                               method: str = "seed_harder",
                               void_redistribution: str = "significant"
                               ) -> float:
    """Compute post-liquefaction residual strength.

    Parameters
    ----------
    N160cs : float
        Clean-sand corrected SPT blow count.
    sigma_v_eff : float, optional
        Effective vertical stress (kPa). Required for Idriss-Boulanger.
    method : str, optional
        "seed_harder" (default) or "idriss_boulanger".
    void_redistribution : str, optional
        For "idriss_boulanger": the curve of GEC-12 Fig. 7-78,
        "significant" (default) or "negligible".

    Returns
    -------
    float
        Residual strength Sr (kPa).
    """
    if method == "seed_harder":
        return Sr_seed_harder(N160cs)
    elif method == "idriss_boulanger":
        if sigma_v_eff is None:
            raise ValueError(
                "sigma_v_eff is required for Idriss-Boulanger method"
            )
        return Sr_idriss_boulanger(N160cs, sigma_v_eff, void_redistribution)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'seed_harder' or 'idriss_boulanger'")
