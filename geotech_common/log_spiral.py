"""Log-spiral passive earth-pressure coefficient Kp, as the references print it.

Three printed sources, offered side by side (BACKLOG GM-21, GM-183):

``caquot_kerisel_1948`` (the default)
    The Caquot and Kerisel (1948) chart as printed in the Caltrans Trenching
    and Shoring Manual (2025), Figure 4-20, p. 4-30 (the same chart: FHWA
    NHI-06-089 Figure 10-10, p. 10-22): Kp at delta = phi for a vertical
    wall, one curve per ground slope beta/phi from -0.9 to +1, times the
    reduction factor R(delta/phi) printed on the same figure. The manual's
    example on p. 4-29 reads it for sloping ground: phi = 32, beta = 27
    (beta/phi = 0.84), "Kp ~= 19" at delta = phi, R = 0.679 at delta/phi =
    0.44, Kp' = 0.679 x 19 = 13.
``dm7_2_fig_4_13``
    UFC 3-220-20 (DM 7.2, 2025) Figure 4-13, p. 193, after Kerisel and Absi
    (1990): a vertical wall with sloping ground, beta/phi' from -0.6 to +1,
    drawn for delta/phi' = 0.66 and theta = 0 only; the chart stops at
    Kp = 20. Footnote 21 (p. 191): the 1982 DM 7.2 charts "used delta/phi'
    = 1 along with reduction ratios", which "were averages that introduced
    substantial inaccuracy for some cases"; the 1948 chart takes that form.
``dm7_2_eq_4_12``
    UFC 3-220-20 Eq. 4-12, p. 191, "for conditions with theta = 0 deg and
    beta = 0 deg": ln Kp = ln[(1 + sin phi')/(1 - sin phi')] x
    [1.443 (delta/phi') sin phi' + 1], for any wall friction ("while
    delta/phi' = 0.66 is a common assumption, this type of design may require
    Kp for a different value of wall friction").

Both charts are printed only as graphs. Their values here were read off the
rendered pages (2026-10-19): each curve traced pixel by pixel on the scanned
figure, calibrated on its printed grid lines, and the reads drawn back over
the figure to check they sit on the curves. Where a curve's label covers it,
the curve is fitted between the traced part and where it meets the top of
the chart. Values between the tabulated nodes are interpolated with log Kp
linear in phi (and in beta/phi), as the charts are drawn on a log scale:
linear interpolation in Kp between 5 degree nodes overstated the chart by
up to 6 % (level ground) and 17 % (beta = phi).

UFC 3-220-20 Table 4-2 (p. 194) is not used (GM-212, :data:`TABLE_4_2_NOTE`):
its row headed delta/phi' = 0.5 prints Fig. 4-13's values for delta/phi' =
0.66 (5.3, 8.0, 12, 20 at phi' = 30, 35, 40, 45), with Coulomb's Kp at
delta = 0.66 phi' beside them (6.1, 9.8, 18, 44). Its delta/phi' = 1 row
agrees with Eq. 4-12 (6.5, 10.5, 18, 35 against 6.63, 10.87, 18.94, 35.21).
"""

from __future__ import annotations

import math

SOURCES = ("caquot_kerisel_1948", "dm7_2_fig_4_13", "dm7_2_eq_4_12")
DEFAULT_SOURCE = "caquot_kerisel_1948"

#: Why UFC 3-220-20 Table 4-2 is not one of the sources (GM-212), for a
#: result's choices and the tools' descriptions. Read on the rendered page
#: (p. 194) beside Fig. 4-13 (p. 193) and Eq. 4-12 (p. 191).
TABLE_4_2_NOTE = (
    "UFC 3-220-20 Table 4-2 (p. 194) is not used: the log-spiral Kp in its "
    "row headed delta/phi' = 0.5, 5.3, 8.0, 12 and 20 at phi' = 30, 35, 40 "
    "and 45, are Fig. 4-13's values, drawn for delta/phi' = 0.66 (5.30, "
    "8.00, 12.0, 20), and the Coulomb Kp printed beside them, 6.1, 9.8, 18 "
    "and 44, are Coulomb's at delta = 0.66 phi' (at 0.5 it gives 5.0, 7.4, "
    "11.8 and 21.4). Eq. 4-12 at delta/phi' = 0.5 gives 4.46, 6.33, 9.33 "
    "and 14.3. The tool reads Fig. 4-13 at the 0.66 it is drawn for and "
    "Eq. 4-12 at the wall friction given")

#: What each source is, in the references' words, for a result's choices.
SOURCE_BASIS = {
    "caquot_kerisel_1948": (
        "Caquot and Kerisel (1948) log-spiral chart, Kp at delta = phi times "
        "the reduction factor R(delta/phi), as printed in the Caltrans "
        "Trenching and Shoring Manual Fig. 4-20 (p. 4-30) and FHWA "
        "NHI-06-089 Fig. 10-10"),
    "dm7_2_fig_4_13": (
        "UFC 3-220-20 (DM 7.2) Fig. 4-13 (p. 193), log spiral after Kerisel "
        "and Absi (1990), vertical wall with sloping ground, drawn for "
        "delta/phi' = 0.66"),
    "dm7_2_eq_4_12": (
        "UFC 3-220-20 (DM 7.2) Eq. 4-12 (p. 191), the log-spiral Kp for "
        "theta = 0 and beta = 0 with any wall friction"),
}

# ---------------------------------------------------------------------------
# Caquot and Kerisel (1948): Caltrans T&S Fig. 4-20, p. 4-30
# ---------------------------------------------------------------------------

# Kp at delta = phi, by beta/phi, at phi nodes (deg). The level row (beta =
# 0) keeps the GM-21 reads at 0 and every 5 deg (phi = 30 keeps 6.30, the
# value the manual's Example 8-1 reads, p. 8-12) and adds the 2.5 deg reads;
# above 45 deg it is clamped at 33.5, as before. The sloping rows are read
# from 10 deg (below it the curves crowd into their labels) to the top of the
# chart (Kp = 100); a row stops where its curve leaves the chart.
_CK_ROWS = {
    1.0: {10: 1.94, 12.5: 2.36, 15: 2.90, 17.5: 3.66, 20: 4.68, 22.5: 6.14,
          25: 8.20, 27.5: 11.2, 30: 15.7, 32.5: 22.8, 35: 34.6, 37.5: 54.9,
          40: 93.5},
    0.8: {10: 1.89, 12.5: 2.29, 15: 2.82, 17.5: 3.55, 20: 4.53, 22.5: 5.87,
          25: 7.77, 27.5: 10.4, 30: 14.3, 32.5: 20.1, 35: 29.4, 37.5: 44.2,
          40: 70.1},
    0.6: {10: 1.85, 12.5: 2.21, 15: 2.67, 17.5: 3.29, 20: 4.13, 22.5: 5.23,
          25: 6.77, 27.5: 8.89, 30: 12.0, 32.5: 16.5, 35: 23.5, 37.5: 34.3,
          40: 52.4, 42.5: 81.1},
    0.4: {10: 1.79, 12.5: 2.11, 15: 2.51, 17.5: 3.03, 20: 3.72, 22.5: 4.61,
          25: 5.78, 27.5: 7.41, 30: 9.63, 32.5: 12.8, 35: 17.7, 37.5: 25.0,
          40: 36.7, 42.5: 55.3, 45: 86.0},
    0.2: {10: 1.74, 12.5: 2.03, 15: 2.37, 17.5: 2.80, 20: 3.35, 22.5: 4.07,
          25: 4.94, 27.5: 6.12, 30: 7.75, 32.5: 9.99, 35: 13.4, 37.5: 18.4,
          40: 26.2, 42.5: 37.6, 45: 54.2},
    0.0: {0: 1.00, 10: 1.65, 12.5: 1.92, 15: 2.20, 17.5: 2.57, 20: 3.00,
          22.5: 3.59, 25: 4.30, 27.5: 5.17, 30: 6.30, 32.5: 7.93, 35: 10.2,
          37.5: 13.3, 40: 17.7, 42.5: 24.3, 45: 33.5},
    -0.2: {10: 1.59, 12.5: 1.79, 15: 2.03, 17.5: 2.31, 20: 2.66, 22.5: 3.08,
           25: 3.60, 27.5: 4.24, 30: 5.01, 32.5: 6.05, 35: 7.48, 37.5: 9.33,
           40: 12.0, 42.5: 15.5, 45: 20.2},
    -0.4: {10: 1.49, 12.5: 1.66, 15: 1.85, 17.5: 2.06, 20: 2.31, 22.5: 2.60,
           25: 2.94, 27.5: 3.36, 30: 3.86, 32.5: 4.46, 35: 5.24, 37.5: 6.26,
           40: 7.66, 42.5: 9.37, 45: 11.6},
    -0.6: {10: 1.37, 12.5: 1.49, 15: 1.62, 17.5: 1.77, 20: 1.93, 22.5: 2.12,
           25: 2.32, 27.5: 2.55, 30: 2.81, 32.5: 3.13, 35: 3.52, 37.5: 4.00,
           40: 4.57, 42.5: 5.24, 45: 6.05},
    -0.8: {10: 1.23, 12.5: 1.30, 15: 1.36, 17.5: 1.44, 20: 1.52, 22.5: 1.60,
           25: 1.69, 27.5: 1.78, 30: 1.88, 32.5: 1.99, 35: 2.11, 37.5: 2.24,
           40: 2.39, 42.5: 2.55, 45: 2.75},
    -0.9: {10: 1.16, 12.5: 1.20, 15: 1.24, 17.5: 1.28, 20: 1.32, 22.5: 1.36,
           25: 1.40, 27.5: 1.44, 30: 1.47, 32.5: 1.51, 35: 1.55, 37.5: 1.59,
           40: 1.63, 42.5: 1.67, 45: 1.71},
}

# Wall-friction reduction factor R = Kp(delta)/Kp(delta = phi), the table
# printed on Caltrans T&S Fig. 4-20 (p. 4-30) and, identically, FHWA
# NHI-06-089 Fig. 10-10 (p. 10-22): phi 10-45 deg, delta/phi 0 to 0.7
# (Matrix 4-1, p. 4-29, quotes its phi 30/32/35 rows). One table serves every
# ground slope (the manual's p. 4-29 example applies it at beta/phi = 0.84).
# R = 1.0 at delta = phi by definition; between 0.7 and 1.0 it is
# interpolated linearly.
_CK_R_PHI = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
_CK_R_RATIO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
_CK_R = {
    10.0: {0.0: 0.864, 0.1: 0.881, 0.2: 0.898, 0.3: 0.912, 0.4: 0.929, 0.5: 0.946, 0.6: 0.962, 0.7: 0.978, 1.0: 1.000},
    15.0: {0.0: 0.775, 0.1: 0.803, 0.2: 0.830, 0.3: 0.854, 0.4: 0.881, 0.5: 0.907, 0.6: 0.934, 0.7: 0.961, 1.0: 1.000},
    20.0: {0.0: 0.678, 0.1: 0.716, 0.2: 0.752, 0.3: 0.787, 0.4: 0.824, 0.5: 0.862, 0.6: 0.901, 0.7: 0.939, 1.0: 1.000},
    25.0: {0.0: 0.574, 0.1: 0.620, 0.2: 0.666, 0.3: 0.711, 0.4: 0.759, 0.5: 0.808, 0.6: 0.860, 0.7: 0.912, 1.0: 1.000},
    30.0: {0.0: 0.467, 0.1: 0.520, 0.2: 0.574, 0.3: 0.627, 0.4: 0.686, 0.5: 0.746, 0.6: 0.811, 0.7: 0.878, 1.0: 1.000},
    35.0: {0.0: 0.362, 0.1: 0.417, 0.2: 0.475, 0.3: 0.536, 0.4: 0.603, 0.5: 0.674, 0.6: 0.752, 0.7: 0.836, 1.0: 1.000},
    40.0: {0.0: 0.262, 0.1: 0.316, 0.2: 0.375, 0.3: 0.439, 0.4: 0.512, 0.5: 0.592, 0.6: 0.682, 0.7: 0.783, 1.0: 1.000},
    45.0: {0.0: 0.174, 0.1: 0.221, 0.2: 0.276, 0.3: 0.339, 0.4: 0.414, 0.5: 0.500, 0.6: 0.600, 0.7: 0.718, 1.0: 1.000},
}

# ---------------------------------------------------------------------------
# Kerisel and Absi (1990): UFC 3-220-20 Fig. 4-13, p. 193
# ---------------------------------------------------------------------------

# Kp at delta/phi' = 0.66, theta = 0, by beta/phi', at phi' nodes (deg). Every
# curve starts at Kp = 1 at phi' = 0; at phi' = 5 the beta = phi' and 0.8 phi'
# curves run together and are left out. A row whose curve leaves the top of
# the chart (Kp = 20) ends there, at the phi' where it does. Cross-read: the
# beta = 0 row agrees within 0.3 % with the GM-21 read of the same chart
# (2.75, 3.70, 5.30, 8.0, 12.0 at 20 to 40 deg).
_DM_TOP = 20.0
_DM_ROWS = {
    1.0: {0: 1.0, 7.5: 1.62, 10: 1.91, 12.5: 2.29, 15: 2.80, 17.5: 3.46,
          20: 4.30, 22.5: 5.45, 25: 7.01, 27.5: 9.27, 30: 12.5, 32.5: 17.4,
          33.46: _DM_TOP},
    0.8: {0: 1.0, 7.5: 1.59, 10: 1.86, 12.5: 2.20, 15: 2.65, 17.5: 3.25,
          20: 4.00, 22.5: 4.99, 25: 6.31, 27.5: 8.23, 30: 10.9, 32.5: 14.8,
          34.81: _DM_TOP},
    0.6: {0: 1.0, 5: 1.34, 7.5: 1.54, 10: 1.79, 12.5: 2.10, 15: 2.50,
          17.5: 3.03, 20: 3.70, 22.5: 4.58, 25: 5.71, 27.5: 7.24, 30: 9.31,
          32.5: 12.3, 35: 16.7, 36.39: _DM_TOP},
    0.4: {0: 1.0, 5: 1.31, 7.5: 1.51, 10: 1.73, 12.5: 2.00, 15: 2.35,
          17.5: 2.82, 20: 3.40, 22.5: 4.12, 25: 5.00, 27.5: 6.21, 30: 7.81,
          32.5: 9.98, 35: 13.0, 37.5: 17.5, 38.51: _DM_TOP},
    0.2: {0: 1.0, 5: 1.29, 7.5: 1.46, 10: 1.66, 12.5: 1.90, 15: 2.20,
          17.5: 2.62, 20: 3.10, 22.5: 3.64, 25: 4.30, 27.5: 5.26, 30: 6.50,
          32.5: 8.01, 35: 10.0, 37.5: 13.0, 40: 17.5, 41.05: _DM_TOP},
    0.0: {0: 1.0, 5: 1.26, 7.5: 1.41, 10: 1.59, 12.5: 1.80, 15: 2.05,
          17.5: 2.37, 20: 2.75, 22.5: 3.18, 25: 3.70, 27.5: 4.39, 30: 5.30,
          32.5: 6.52, 35: 8.00, 37.5: 9.74, 40: 12.0, 42.5: 15.2,
          45: _DM_TOP},
    -0.2: {0: 1.0, 5: 1.23, 7.5: 1.37, 10: 1.52, 12.5: 1.70, 15: 1.90,
           17.5: 2.14, 20: 2.40, 22.5: 2.72, 25: 3.10, 27.5: 3.60, 30: 4.20,
           32.5: 4.88, 35: 5.70, 37.5: 6.78, 40: 8.20, 42.5: 10.1, 45: 12.5},
    -0.4: {0: 1.0, 5: 1.19, 7.5: 1.30, 10: 1.42, 12.5: 1.56, 15: 1.72,
           17.5: 1.90, 20: 2.10, 22.5: 2.31, 25: 2.55, 27.5: 2.85, 30: 3.20,
           32.5: 3.61, 35: 4.10, 37.5: 4.70, 40: 5.40, 42.5: 6.21, 45: 7.10},
    -0.6: {0: 1.0, 5: 1.15, 7.5: 1.24, 10: 1.33, 12.5: 1.42, 15: 1.52,
           17.5: 1.63, 20: 1.76, 22.5: 1.90, 25: 2.05, 27.5: 2.22, 30: 2.40,
           32.5: 2.58, 35: 2.80, 37.5: 3.08, 40: 3.40, 42.5: 3.74, 45: 4.06},
}
DM_DELTA_RATIO = 0.66      # the one wall friction Fig. 4-13 is drawn for
_DM_DELTA_TOL = 0.01


def _rankine_Kp(phi_deg: float) -> float:
    return math.tan(math.radians(45.0 + phi_deg / 2.0)) ** 2


def _lininterp(x, xs, ys):
    """Clamped piecewise-linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _row_at(row: dict, phi: float, clamp: bool = False):
    """log Kp linear in phi between a row's nodes; None outside them (off
    the chart), unless ``clamp``."""
    xs = sorted(row)
    if phi < xs[0] - 1e-9 or phi > xs[-1] + 1e-9:
        if not clamp:
            return None
        phi = min(max(phi, xs[0]), xs[-1])
    logs = [math.log(row[x]) for x in xs]
    return math.exp(_lininterp(phi, xs, logs))


def _across_rows(rows: dict, phi: float, ratio: float, what: str):
    """Kp between the two rows that bracket beta/phi = ``ratio``, log Kp
    linear in beta/phi."""
    keys = sorted(rows)
    for k in keys:
        if abs(ratio - k) < 1e-9:
            value = _row_at(rows[k], phi)
            if value is None:
                raise ValueError(f"{what}: phi {phi:g} deg with beta/phi "
                                 f"{ratio:.3g} is off the chart")
            return value
    lo = max(k for k in keys if k < ratio)
    hi = min(k for k in keys if k > ratio)
    a, b = _row_at(rows[lo], phi), _row_at(rows[hi], phi)
    if a is None or b is None:
        raise ValueError(f"{what}: phi {phi:g} deg with beta/phi "
                         f"{ratio:.3g} is off the chart")
    t = (ratio - lo) / (hi - lo)
    return math.exp(math.log(a) + t * (math.log(b) - math.log(a)))


def caquot_kerisel_1948_Kp0(phi_deg: float, beta_deg: float = 0.0) -> float:
    """Kp at delta = phi off the Caquot-Kerisel chart (Caltrans T&S Fig.
    4-20): the level curve for beta = 0 (clamped at 45 deg), else between
    the beta/phi curves, -0.9 to +1, phi 10 to 45 deg."""
    if not beta_deg:
        return _row_at(_CK_ROWS[0.0], phi_deg, clamp=True)
    what = "Caquot-Kerisel (Caltrans T&S Fig. 4-20)"
    if phi_deg < 10.0 or phi_deg > 45.0:
        raise ValueError(f"{what}: the sloping-ground curves are read for "
                         f"phi 10 to 45 deg, got {phi_deg:g}")
    ratio = beta_deg / phi_deg
    if ratio < -0.9 - 1e-9 or ratio > 1.0 + 1e-9:
        raise ValueError(f"{what}: the chart's curves run from beta/phi = "
                         f"-0.9 to +1, got beta {beta_deg:g} deg on phi "
                         f"{phi_deg:g} deg ({ratio:.3g})")
    return _across_rows(_CK_ROWS, phi_deg, min(max(ratio, -0.9), 1.0), what)


def _ck_R(phi_deg: float, ratio: float) -> float:
    """R off the printed table, bilinear in (phi, delta/phi), clamped in
    phi; ``ratio`` from 0 to 1."""
    phi_c = min(max(phi_deg, _CK_R_PHI[0]), _CK_R_PHI[-1])
    row = [_lininterp(ratio, _CK_R_RATIO, [_CK_R[p][k] for k in _CK_R_RATIO])
           for p in _CK_R_PHI]
    return _lininterp(phi_c, _CK_R_PHI, row)


def caquot_kerisel_1948_Kp(phi_deg: float, delta_deg: float = None,
                           beta_deg: float = 0.0,
                           Kp_initial: float = None) -> float:
    """Caquot-Kerisel (1948) log-spiral Kp' = R x Kp0 (Caltrans T&S Sec. 4-6,
    Fig. 4-20). Kp0 is the chart's Kp at delta = phi for the ground slope
    beta (or ``Kp_initial``); R the printed reduction factor for delta/phi.
    ``delta_deg`` None means delta = phi (R = 1). On level ground, between
    delta = 0 and the 0.1 column R runs to the Rankine anchor R(0) =
    Kp_rankine/Kp0, so a smooth wall returns the Rankine Kp (the printed R(0)
    times the chart Kp0 gives it within 2 %); on sloping ground the printed
    R(0) column is used."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    Kp0 = (Kp_initial if Kp_initial is not None
           else caquot_kerisel_1948_Kp0(phi_deg, beta_deg))
    if delta_deg is None or phi_deg <= 0:
        return Kp0
    ratio = min(max(delta_deg / phi_deg, 0.0), 1.0)
    if ratio >= 1.0:
        return Kp0
    lowest = _CK_R_RATIO[1]  # 0.1
    if ratio >= lowest or beta_deg:
        return Kp0 * _ck_R(phi_deg, ratio)
    R0 = _rankine_Kp(phi_deg) / Kp0
    return Kp0 * (R0 + (ratio / lowest) * (_ck_R(phi_deg, lowest) - R0))


def dm7_2_eq_4_12_Kp(phi_deg: float, delta_deg: float) -> float:
    """UFC 3-220-20 Eq. 4-12 (p. 191), theta = 0 and beta = 0:
    ln Kp = ln[(1 + sin phi')/(1 - sin phi')] [1.443 (delta/phi') sin phi'
    + 1]."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    if delta_deg is None:
        raise ValueError("UFC 3-220-20 Eq. 4-12 needs the wall friction "
                         "angle delta given (delta_deg); p. 191: 'while "
                         "delta/phi' = 0.66 is a common assumption, this "
                         "type of design may require Kp for a different "
                         "value of wall friction'")
    if phi_deg == 0:
        return 1.0
    if delta_deg < 0 or delta_deg > phi_deg + 1e-9:
        raise ValueError(f"UFC 3-220-20 Eq. 4-12: delta must be 0 to phi, "
                         f"got {delta_deg:g} on phi {phi_deg:g}")
    s = math.sin(math.radians(phi_deg))
    return math.exp(math.log((1 + s) / (1 - s))
                    * (1.443 * (delta_deg / phi_deg) * s + 1.0))


def dm7_2_fig_4_13_Kp(phi_deg: float, beta_deg: float = 0.0,
                      delta_deg: float = None) -> float:
    """Kp off UFC 3-220-20 Fig. 4-13 (p. 193; Kerisel and Absi 1990): theta
    = 0, delta/phi' = 0.66 (``delta_deg`` None means 0.66 phi'; another
    value is refused), beta/phi' from -0.6 to +1, Kp up to 20."""
    what = "UFC 3-220-20 Fig. 4-13"
    if phi_deg < 0 or phi_deg > 45:
        raise ValueError(f"{what}: phi' runs 0 to 45 deg, got {phi_deg:g}")
    if phi_deg == 0:
        if beta_deg:
            raise ValueError(f"{what}: a slope on phi' = 0 is off the chart")
        return 1.0
    if delta_deg is not None and abs(delta_deg / phi_deg
                                     - DM_DELTA_RATIO) > _DM_DELTA_TOL:
        raise ValueError(
            f"{what} is drawn for delta/phi' = 0.66 only, got delta "
            f"{delta_deg:g} on phi' {phi_deg:g} ({delta_deg / phi_deg:.3g}); "
            "UFC 3-220-20 Eq. 4-12 takes another wall friction on level "
            "ground, and the Caquot-Kerisel chart any wall friction")
    ratio = (beta_deg or 0.0) / phi_deg
    if ratio < -0.6 - 1e-9 or ratio > 1.0 + 1e-9:
        raise ValueError(f"{what}: the chart's curves run from beta/phi' = "
                         f"-0.6 to +1, got beta {beta_deg:g} deg on phi' "
                         f"{phi_deg:g} deg ({ratio:.3g})")
    value = _across_rows(_DM_ROWS, phi_deg, min(max(ratio, -0.6), 1.0), what)
    if value > _DM_TOP * (1 + 1e-9):
        raise ValueError(f"{what}: Kp above 20 is off the chart")
    return value


def log_spiral_Kp(phi_deg: float, delta_deg: float = None,
                  beta_deg: float = 0.0,
                  source: str = DEFAULT_SOURCE) -> float:
    """The log-spiral passive Kp from ``source`` (see :data:`SOURCES`)."""
    if source == "caquot_kerisel_1948":
        return caquot_kerisel_1948_Kp(phi_deg, delta_deg, beta_deg=beta_deg)
    if source == "dm7_2_fig_4_13":
        return dm7_2_fig_4_13_Kp(phi_deg, beta_deg, delta_deg)
    if source == "dm7_2_eq_4_12":
        if beta_deg:
            raise ValueError(
                "UFC 3-220-20 Eq. 4-12 is for level ground (beta = 0); its "
                "Fig. 4-13 or the Caquot-Kerisel chart takes a slope")
        return dm7_2_eq_4_12_Kp(phi_deg, delta_deg)
    raise ValueError(f"Unknown log-spiral source {source!r}; one of "
                     f"{list(SOURCES)}")
