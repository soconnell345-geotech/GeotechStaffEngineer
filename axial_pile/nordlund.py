"""
Nordlund Method for driven piles in cohesionless soils.

Side resistance by Nordlund (1963, 1979) and toe resistance by Thurman's
alpha_t N'q with Meyerhof's limiting unit toe resistance, both read from the
GEC-12 design charts as the step-by-step procedure reads them (GEC-12,
FHWA-NHI-16-009, Sec. 7.2.1.3.1, Steps 1-11, printed pp. 242-245):

* Step 2: delta/phi from Figure 7-9 by pile type and displaced volume V;
* Step 3: K_delta from Tables 7-6 and 7-7 (uniform piles, omega = 0),
  linear in phi and log-linear in V (Steps 3c-3e);
* Step 4: C_F from Figure 7-14 by phi and delta/phi;
* Step 7: alpha_t from Figure 7-16(a) by phi and D/b, N'q from 7-16(b);
* Step 8: sigma'p at the toe limited to 3 ksf ("Note: The limiting value
  of sigma'p is 3 ksf", p. 244);
* Step 9: Rp = alpha_t N'q Ap sigma'p, at most q_L Ap from Figure 7-15.

The charts were measured off the printed figures (GEC 12 Vol 1, pdf pages
280, 283 and 284) on 2026-10-18 by tracing the curve strokes between the
fitted gridlines. Read against GEC-12's worked example (Vol 3, Table D-6,
HP12x74 by DrivenPiles) they give the published unit side resistance to
about 2 % and alpha_t N'q to about 5 %.

Outside a chart's range a value is held at the chart's edge and a note
says so (``notes``). Units are SI (kPa, m, kN, degrees); V is in ft3/ft, as
the charts give it.

References:
    Nordlund, R.L. (1963, 1979)
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Sec. 7.2.1.3.1
    Meyerhof, G.G. (1976) -- Bearing capacity and settlement of pile
    foundations
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

#: 1 ft3/ft in m3/m (= 1 ft2 in m2).
FT3_PER_FT = 0.09290304
#: 1 ksf in kPa.
KSF = 47.880259
#: Step 8 (p. 244): the limiting value of sigma'p is 3 ksf.
SIGMA_P_LIMIT = 3.0 * KSF


def _note(notes: Optional[list], text: str) -> None:
    """Add ``text`` to ``notes`` once; with no list, warn."""
    if notes is None:
        warnings.warn(text, stacklevel=3)
    elif text not in notes:
        notes.append(text)


def _held(x: float, xs, ys) -> Tuple[float, bool]:
    """Linear interpolation, held at the ends; and whether it was held."""
    return float(np.interp(x, xs, ys)), (x < xs[0] or x > xs[-1])


def displaced_volume(area_m2: float) -> float:
    """Displaced volume V (ft3/ft) of a pile whose displacing area is
    ``area_m2`` (m2): the volume per unit length GEC-12 Step 2a computes."""
    return area_m2 / FT3_PER_FT


# ---------------------------------------------------------------------------
# Figure 7-9: delta/phi against displaced volume V, by pile type.
# ---------------------------------------------------------------------------

_FIG_7_9_V_ABC = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                  1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
_FIG_7_9 = {
    # a. pipe piles and non-tapered portion of monotube piles
    "a": (_FIG_7_9_V_ABC,
          [0.344, 0.411, 0.464, 0.514, 0.560, 0.594, 0.627, 0.655, 0.678,
           0.700, 0.717, 0.735, 0.749, 0.763, 0.777, 0.788, 0.798]),
    # b. timber piles
    "b": (_FIG_7_9_V_ABC,
          [0.406, 0.467, 0.517, 0.561, 0.603, 0.636, 0.666, 0.694, 0.714,
           0.734, 0.751, 0.767, 0.781, 0.793, 0.805, 0.816, 0.823]),
    # c. precast concrete piles
    "c": (_FIG_7_9_V_ABC,
          [0.467, 0.525, 0.579, 0.629, 0.672, 0.706, 0.738, 0.767, 0.789,
           0.813, 0.832, 0.849, 0.864, 0.878, 0.892, 0.904, 0.913]),
    # f. H-piles and augercast piles
    "f": ([0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.52],
          [0.731, 0.747, 0.791, 0.823, 0.858, 0.887, 0.933, 0.971, 0.979]),
}

#: The Figure 7-9 curve for each pile type, and its legend.
FIG_7_9_CURVE = {
    "pipe_closed": ("a", "pipe piles"),
    "pipe_open": ("a", "pipe piles"),
    "timber": ("b", "timber piles"),
    "concrete": ("c", "precast concrete piles"),
    "h_pile": ("f", "H-piles"),
}


def delta_phi_ratio_fig_7_9(pile_type: str, V: float,
                            notes: Optional[list] = None) -> float:
    """delta/phi from GEC-12 Figure 7-9 for ``pile_type`` at displaced
    volume ``V`` (ft3/ft); held at the curve's end outside it (with a
    note)."""
    key, label = FIG_7_9_CURVE[pile_type]
    vs, ratios = _FIG_7_9[key]
    ratio, held = _held(V, vs, ratios)
    if held:
        _note(notes, (
            f"Displaced volume V = {V:.3g} ft3/ft is outside GEC-12 Figure "
            f"7-9's curve {key} ({label}, V = {vs[0]:g}-{vs[-1]:g} ft3/ft); "
            f"delta/phi was held at the curve's end value, {ratio:.2f}."))
    return ratio


# ---------------------------------------------------------------------------
# Tables 7-6 and 7-7: K_delta for uniform piles (omega = 0), at the chart
# volumes V = 0.1, 1.0 and 10 ft3/ft. The tables' columns between those
# volumes are the log-linear interpolation Step 3d describes.
# ---------------------------------------------------------------------------

_KD_PHI = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
_KD_V = [0.1, 1.0, 10.0]
_KD = [  # K_delta at V = 0.1, 1.0, 10 ft3/ft
    (0.70, 0.85, 1.00), (0.73, 0.91, 1.09), (0.76, 0.97, 1.18),
    (0.79, 1.03, 1.27), (0.82, 1.09, 1.36), (0.85, 1.15, 1.45),
    (0.91, 1.27, 1.63), (0.97, 1.39, 1.81), (1.03, 1.51, 1.99),
    (1.09, 1.63, 2.17), (1.15, 1.75, 2.35), (1.26, 2.00, 2.74),
    (1.37, 2.25, 3.13), (1.48, 2.50, 3.52), (1.59, 2.75, 3.91),
    (1.70, 3.00, 4.30),
]


def k_delta(phi_deg: float, V: float, notes: Optional[list] = None) -> float:
    """K_delta (GEC-12 Step 3, Tables 7-6 and 7-7, omega = 0): linear in
    phi at the chart volumes, then log-linear in V (Step 3e). Outside
    phi 25-40 deg or V 0.1-10 ft3/ft the table's edge is held (a note)."""
    at_v = []
    for j in range(3):
        k, held_phi = _held(phi_deg, _KD_PHI, [row[j] for row in _KD])
        at_v.append(k)
    if held_phi:
        _note(notes, (
            f"phi = {phi_deg:g} deg is outside GEC-12 Tables 7-6/7-7 "
            f"(25-40 deg); K_delta was held at the table's "
            f"{min(max(phi_deg, 25), 40):g} deg row."))
    v = min(max(V, _KD_V[0]), _KD_V[-1])
    if v != V:
        _note(notes, (
            f"Displaced volume V = {V:.3g} ft3/ft is outside GEC-12 Tables "
            f"7-6/7-7 (0.1-10 ft3/ft); K_delta was held at V = {v:g}."))
    return float(np.interp(math.log10(v), [-1.0, 0.0, 1.0], at_v))


# ---------------------------------------------------------------------------
# Figure 7-14: correction factor C_F against phi, one curve per delta/phi.
# ---------------------------------------------------------------------------

_FIG_7_14_PHI = [15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5,
                 45]
_FIG_7_14_RATIO = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
_FIG_7_14_CF = [
    [0.815, 0.778, 0.740, 0.701, 0.661, 0.621, 0.580, 0.538, 0.488, 0.432,
     0.370, 0.306, 0.241],
    [0.888, 0.857, 0.825, 0.793, 0.759, 0.725, 0.689, 0.654, 0.612, 0.570,
     0.525, 0.479, 0.431],
    [0.942, 0.925, 0.905, 0.886, 0.864, 0.842, 0.817, 0.787, 0.757, 0.728,
     0.691, 0.654, 0.620],
    [0.988, 0.982, 0.976, 0.973, 0.966, 0.956, 0.945, 0.932, 0.917, 0.902,
     0.886, 0.865, 0.841],
    [1.0] * 13,
    [1.005, 1.008, 1.012, 1.015, 1.018, 1.024, 1.030, 1.038, 1.050, 1.059,
     1.074, 1.095, 1.120],
    [1.012, 1.017, 1.024, 1.033, 1.038, 1.047, 1.061, 1.074, 1.086, 1.098,
     1.115, 1.136, 1.160],
]


def correction_factor(phi_deg: float, delta_phi_ratio: float,
                      notes: Optional[list] = None) -> float:
    """C_F (GEC-12 Step 4, Figure 7-14) for ``phi_deg`` and
    ``delta_phi_ratio``: linear along each curve, then between curves.
    Outside phi 15-45 deg or delta/phi 0.2-1.4 the edge is held (a
    note)."""
    per_curve = []
    for row in _FIG_7_14_CF:
        cf, held_phi = _held(phi_deg, _FIG_7_14_PHI, row)
        per_curve.append(cf)
    if held_phi:
        _note(notes, (
            f"phi = {phi_deg:g} deg is outside GEC-12 Figure 7-14 (15-45 "
            f"deg); C_F was held at the chart's edge."))
    cf, held_ratio = _held(delta_phi_ratio, _FIG_7_14_RATIO, per_curve)
    if held_ratio:
        _note(notes, (
            f"delta/phi = {delta_phi_ratio:.2f} is outside GEC-12 Figure "
            f"7-14's curves (0.2-1.4); C_F was held at the nearest curve."))
    return cf


def nordlund_side_factors(phi_deg: float, delta_phi_ratio: float, V: float,
                          notes: Optional[list] = None) -> Dict[str, float]:
    """The Nordlund factors of one layer (GEC-12 Steps 2-4):
    ``delta_phi_ratio``, ``delta_deg``, ``K_delta``, ``C_F`` and
    ``fs_ratio`` = K_delta C_F sin(delta), the unit side resistance over
    sigma'd (Eq. 7-8, uniform cross section)."""
    delta = phi_deg * delta_phi_ratio
    kd = k_delta(phi_deg, V, notes)
    cf = correction_factor(phi_deg, delta_phi_ratio, notes)
    return {"delta_phi_ratio": delta_phi_ratio, "delta_deg": delta,
            "K_delta": kd, "C_F": cf,
            "fs_ratio": kd * cf * math.sin(math.radians(delta))}


def skin_friction_cohesionless(phi_deg: float, sigma_v: float,
                               pile_perimeter: float,
                               layer_thickness: float,
                               delta_phi_ratio: float,
                               V: float,
                               notes: Optional[list] = None) -> float:
    """Side resistance (kN) of one cohesionless segment, GEC-12 Eq. 7-8:

    Rs = K_delta C_F sigma'd sin(delta) Cd D

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    sigma_v : float
        Effective vertical stress at the segment's mid-depth (kPa); no
        limit is applied to it (Step 5: "A limiting value is not applied
        to sigma'd").
    pile_perimeter : float
        Pile perimeter Cd (m); the "box" perimeter for H-piles (Step 6).
    layer_thickness : float
        Segment length (m).
    delta_phi_ratio : float
        delta/phi (Figure 7-9, or given).
    V : float
        Displaced volume (ft3/ft) for K_delta.
    notes : list, optional
        Collects the out-of-chart notes; without it they are warned.
    """
    f = nordlund_side_factors(phi_deg, delta_phi_ratio, V, notes)
    return f["fs_ratio"] * sigma_v * pile_perimeter * layer_thickness


# ---------------------------------------------------------------------------
# Figure 7-16: (a) alpha_t against phi for D/b = 20, 30 and 45; (b) N'q.
# ---------------------------------------------------------------------------

_FIG_7_16A_PHI = list(range(20, 46))
_FIG_7_16A_DB = [20, 30, 45]
_FIG_7_16A_AT = [  # rows: phi 20..45; columns: D/b 20, 30, 45
    (0.365, 0.257, 0.178), (0.386, 0.280, 0.202), (0.406, 0.306, 0.229),
    (0.426, 0.333, 0.257), (0.448, 0.361, 0.288), (0.470, 0.390, 0.320),
    (0.491, 0.418, 0.351), (0.513, 0.446, 0.384), (0.536, 0.475, 0.420),
    (0.559, 0.502, 0.455), (0.580, 0.531, 0.491), (0.602, 0.559, 0.526),
    (0.625, 0.591, 0.563), (0.649, 0.621, 0.595), (0.665, 0.645, 0.625),
    (0.682, 0.669, 0.657), (0.695, 0.690, 0.686), (0.707, 0.707, 0.707),
    (0.721, 0.721, 0.721), (0.736, 0.736, 0.736), (0.750, 0.750, 0.750),
    (0.762, 0.762, 0.762), (0.774, 0.774, 0.774), (0.786, 0.786, 0.786),
    (0.791, 0.791, 0.791), (0.800, 0.800, 0.800),
]

# N'q measured again (GM-282, 2026-10-30) on the figure's own 150 dpi image
# (GEC 12 Vol 1, pdf page index 284), the centre of the drawn line at each
# degree against the decade gridlines: the 2026-10-18 nodes read 1-3.5 %
# low from 29 deg up (155.3 at 40 deg, the figure 160.4).
_FIG_7_16B_PHI = list(range(15, 46))
_FIG_7_16B_NQ = [4.83, 5.32, 5.90, 6.56, 7.31, 8.18, 9.19, 10.34, 11.70,
                 13.24, 15.00, 17.1, 19.6, 22.6, 26.0, 30.1, 34.6, 40.0,
                 46.5, 54.3, 63.8, 75.5, 89.8, 107.9, 130.9, 160.4, 194.8,
                 238.7, 296.5, 374.0, 475.0]


def alpha_t(phi_deg: float, D_over_b: float,
            notes: Optional[list] = None) -> float:
    """alpha_t (GEC-12 Step 7a, Figure 7-16(a)) for the toe phi and the
    embedded length over the pile width, D/b. Linear in phi along each
    curve, then linear in D/b between the curves; D/b below 20 or above 45
    takes the nearest curve. phi outside 20-45 deg is held (a note)."""
    per_curve = []
    for j in range(3):
        at, held = _held(phi_deg, _FIG_7_16A_PHI,
                         [row[j] for row in _FIG_7_16A_AT])
        per_curve.append(at)
    if held:
        _note(notes, (
            f"Toe phi = {phi_deg:g} deg is outside GEC-12 Figure 7-16(a) "
            f"(20-45 deg); alpha_t was held at the chart's edge."))
    return float(np.interp(D_over_b, _FIG_7_16A_DB, per_curve))


def n_q_prime(phi_deg: float, notes: Optional[list] = None) -> float:
    """N'q (GEC-12 Step 7b, Figure 7-16(b)), linear in log N'q between the
    1-degree nodes; held outside 15-45 deg (a note)."""
    log_nq, held = _held(phi_deg, _FIG_7_16B_PHI,
                         [math.log10(n) for n in _FIG_7_16B_NQ])
    if held:
        _note(notes, (
            f"Toe phi = {phi_deg:g} deg is outside GEC-12 Figure 7-16(b) "
            f"(15-45 deg); N'q was held at the chart's edge."))
    return 10.0 ** log_nq


# GEC-12 Figure 7-15 (after Meyerhof 1976), measured off the printed figure
# (GEC 12 Vol 1, pdf page index 283) on 2026-10-08. The same nodes are
# geotech_references.gec_12.figure_7_15_limiting_toe_resistance, in tsf.
# The axis starts at 30 deg and the curve ends at 43.75 deg.
TSF_TO_KPA = 95.76
_FIG_7_15_PHI = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 43.75]
_FIG_7_15_QL_TSF = [7.1, 10.1, 16.0, 24.5, 35.9, 53.7, 75.2, 102.3, 133.6,
                    168.3, 208.5, 251.6, 296.0, 339.2, 368.0]
_FIG_7_15_QL_KPA = [q * TSF_TO_KPA for q in _FIG_7_15_QL_TSF]
TOE_LIMIT_PHI_MIN = _FIG_7_15_PHI[0]
TOE_LIMIT_PHI_MAX = _FIG_7_15_PHI[-1]


def _limiting_tip_resistance(phi_deg: float) -> float:
    """Limiting unit tip resistance q_L from GEC-12 Figure 7-15 (Meyerhof, 1976).

    Piecewise linear interpolation of the chart's 1-degree nodes; outside
    the chart (below 30 deg, above 43.75 deg) the curve's end value is used
    and :func:`toe_limit_chart_note` says so.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).

    Returns
    -------
    float
        Limiting unit tip resistance q_L (kPa).

    References
    ----------
    FHWA GEC-12, Figure 7-15 (after Meyerhof, 1976).

    Provenance.
    - 2026-07-19: replaced a table about 10x unconservative at phi = 30.
    - 2026-10-08: re-measured off the figure (gridline-fitted; a second
      pixel read and two vision models agree within about 1 tsf).
      - The 2026-07-19 table was still +41 % at 30 deg, +25 % at 32 deg and
        +11 % at 34 deg, and 3-6 % low from 38 to 43 deg.
      - It also carried nodes at 26, 28, 44 and 45 deg that are not on the
        chart.
      - GEC-12's worked example (FHWA-NHI-16-064 Table D-6, the 428.1 kip
        toe plateau at a toe phi of 40) now agrees to +0.3 % (429.3 kips) instead of -4.6 %.
    """
    phi = min(max(phi_deg, TOE_LIMIT_PHI_MIN), TOE_LIMIT_PHI_MAX)
    return float(np.interp(phi, _FIG_7_15_PHI, _FIG_7_15_QL_KPA))


def toe_limit_chart_note(phi_deg: float, governed: Optional[bool] = None
                         ) -> Optional[str]:
    """Say when the toe friction angle is outside GEC-12 Figure 7-15.

    Returns None inside the chart (30-43.75 deg). Outside it, returns the
    warning to show the user: which end value was used, and, below 30 deg,
    that the true limit is lower. ``governed`` (whether the limit capped the
    toe resistance) adds one clause when known.
    """
    if TOE_LIMIT_PHI_MIN <= phi_deg <= TOE_LIMIT_PHI_MAX:
        return None
    if phi_deg < TOE_LIMIT_PHI_MIN:
        end, where = _FIG_7_15_QL_TSF[0], "below"
        tail = (" The chart's curve falls toward zero below 30 deg, so the "
                "true limit is LOWER than this and the toe resistance may be "
                "overstated; check it by another method.")
    else:
        end, where = _FIG_7_15_QL_TSF[-1], "above"
        tail = (" The curve stops at 43.75 deg, so holding its end value is "
                "on the low side of any extrapolation.")
    note = (f"Toe friction angle {phi_deg:g} deg is {where} GEC-12 Figure "
            f"7-15 (it spans 30-43.75 deg). The limiting toe resistance was "
            f"held at the chart's end value, {end:g} tsf "
            f"({end * TSF_TO_KPA:,.0f} kPa).{tail}")
    if governed is True:
        note += " The limit governed the toe resistance here."
    elif governed is False and phi_deg < TOE_LIMIT_PHI_MIN:
        note += (" It did not govern here, but a lower true limit might "
                 "have.")
    return note


def end_bearing_cohesionless(phi_deg: float, sigma_v_tip: float,
                             tip_area: float,
                             pile_depth: float,
                             pile_width: float,
                             notes: Optional[list] = None,
                             sigma_p_limit: Optional[float] = SIGMA_P_LIMIT,
                             q_limit: Optional[float] = None,
                             details: Optional[dict] = None) -> float:
    """Toe resistance (kN) in cohesionless soil, GEC-12 Steps 7-9:

    Rp = alpha_t N'q Ap sigma'p  (Eq. 7-9), at most q_L Ap (Figure 7-15)

    Parameters
    ----------
    phi_deg : float
        Friction angle near the pile toe (degrees).
    sigma_v_tip : float
        Effective vertical stress at the pile toe (kPa).
    tip_area : float
        Pile toe area Ap (m2).
    pile_depth : float
        Embedded pile length D (m), for D/b in Figure 7-16(a).
    pile_width : float
        Pile diameter or width b (m).
    notes : list, optional
        Collects the out-of-chart notes; without it they are warned.
    sigma_p_limit : float or None, optional
        The limit on sigma'p (kPa). Default 3 ksf (Step 8); None applies
        no limit.
    q_limit : float or None, optional
        A further limit on the unit toe resistance (kPa), e.g. 105 ksf for
        an open-end pipe pile (GEC-12 Sec. 7.10.7). None for none.
    details : dict, optional
        Filled with the toe's factors: alpha_t, N'q, D/b, sigma'p, q_L,
        the unit toe resistance and which limit governed.
    """
    Db = pile_depth / pile_width if pile_width > 0 else 0.0
    at = alpha_t(phi_deg, Db, notes)
    Nq = n_q_prime(phi_deg, notes)
    sigma_p = (sigma_v_tip if sigma_p_limit is None
               else min(sigma_v_tip, sigma_p_limit))
    q_eq = at * Nq * sigma_p

    q_L = _limiting_tip_resistance(phi_deg)
    note = toe_limit_chart_note(phi_deg, governed=q_eq >= q_L)
    if note:
        _note(notes, note)

    q_p, governs = q_eq, "Eq. 7-9"
    if q_L < q_p:
        q_p, governs = q_L, "q_L (Fig. 7-15)"
    if q_limit is not None and q_limit < q_p:
        q_p, governs = q_limit, "open-end pipe limit"
    if details is not None:
        details.update({
            "alpha_t": at, "N_q_prime": Nq, "D_over_b": Db,
            "sigma_v_tip_kPa": sigma_v_tip, "sigma_p_kPa": sigma_p,
            "q_L_kPa": q_L, "q_p_kPa": q_p, "governs": governs,
            "toe_area_m2": tip_area,
        })
    return q_p * tip_area
