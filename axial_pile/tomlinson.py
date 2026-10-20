"""
The alpha method for driven piles in cohesive soils (total stress), GEC-12
Sec. 7.2.1.3.2 (printed pp. 245-250).

Unit side resistance fs = Ca = alpha su (Eq. 7-10). Step 1 (p. 250) reads
the pile adhesion Ca off Figure 7-17 (after Tomlinson 1979, p. 248) by the
undrained shear strength su, the pile type and D/b:

* solid curves: concrete, timber and corrugated steel piles; an H-pile
  takes the corrugated pile curve times its "box" area (p. 247);
* dashed curves: smooth steel piles;
* curves for D = 10b and D = 40b, where D runs from the ground surface to
  the bottom of the clay layer, or to the pile toe if that is less (p. 246
  and the figure's legend).

Ca is read as drawn: the alpha method's steps take Ca from the figure
with no limit, and the D = 40b concrete, timber and corrugated curve passes
above Ca = su below about 1.5 ksf (alpha about 1.1 at su = 1.0 ksf).

Where su exceeds the figure's curves, p. 247 and Step 1 recommend the
adhesion factor of API RP 2A (Sec. 7.2.1.3.3, Eqs. 7-13 to 7-15, p. 251)
for highly overconsolidated clays. :func:`adhesion` takes API alpha there
(``beyond_chart="api"``) or holds Ca at the curve's end
(``"chart_end"``, as GEC-12's worked example, Vol 3 Table D-100, did);
which clays count as highly overconsolidated is the engineer's call, made
in :class:`~axial_pile.capacity.AxialPileAnalysis`. API alpha is at most
1.0, API RP 2A's own limit, so it applies on that route only.

Toe resistance qp = 9 su (Eq. 7-11). Figure 7-17's curves were measured off
the printed figure on 2026-10-18. Units are SI (kPa, m, kN).

References:
    Tomlinson, M.J. (1979) -- Pile Design and Construction Practice
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Sec. 7.2.1.3.2 and 7.2.1.3.3
    API RP 2A (1993)
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from axial_pile.nordlund import _note

#: 1 ksf in kPa.
KSF = 47.880259

#: The Figure 7-17 curve families: GEC-12's legend for each.
SOLID = "concrete_timber_corrugated"
DASHED = "smooth_steel"
CURVES = {SOLID: "concrete, timber, corrugated steel piles",
          DASHED: "smooth steel piles"}

_SU = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
       1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
       3.0]
# Ca (ksf) at _SU (ksf). Solid D = 10b runs on to 3.45 ksf; the solid D = 40b
# curve starts at su = 1.0 ksf (below it the chart has the one solid curve);
# the dashed curves end at about 3.05 ksf.
_SOLID_10 = ([0.26, 0.324, 0.413, 0.503, 0.598, 0.691, 0.784, 0.875, 0.948,
              1.013, 1.082, 1.14, 1.188, 1.249, 1.296, 1.333, 1.359, 1.367,
              1.369, 1.354, 1.328, 1.289, 1.222, 1.164, 1.104, 1.052, 1.004,
              0.961, 0.922, 0.892, 0.862, 0.844, 0.832, 0.827])
_SOLID_40 = ([1.106, 1.181, 1.263, 1.346, 1.419, 1.497, 1.566, 1.626, 1.695,
              1.749, 1.799, 1.819, 1.819, 1.814, 1.797, 1.767, 1.724, 1.685,
              1.633, 1.586, 1.514, 1.465, 1.429, 1.402, 1.376, 1.367])
_DASHED_10 = ([0.23, 0.257, 0.317, 0.403, 0.49, 0.575, 0.66, 0.737, 0.801,
               0.859, 0.92, 0.965, 1.009, 1.063, 1.108, 1.14, 1.164, 1.168,
               1.17, 1.152, 1.127, 1.084, 1.045, 1.0, 0.946, 0.896, 0.858,
               0.821, 0.786, 0.78])
_DASHED_40 = ([0.26, 0.324, 0.413, 0.503, 0.598, 0.691, 0.784, 0.875, 0.948,
               1.013, 1.082, 1.14, 1.243, 1.295, 1.344, 1.406, 1.456, 1.497,
               1.516, 1.555, 1.581, 1.578, 1.555, 1.524, 1.486, 1.464, 1.408,
               1.352, 1.294, 1.279])
_SU_SOLID = _SU + [3.1, 3.2, 3.3, 3.4, 3.45]
_SU_SOLID_40 = _SU[8:] + [3.1, 3.2, 3.3, 3.4, 3.45]
_SU_DASHED = _SU + [3.05]

#: Each curve family: (su, Ca at D = 10b, su, Ca at D = 40b) in ksf.
_FIG_7_17 = {
    SOLID: (_SU_SOLID, _SOLID_10, _SU_SOLID_40, _SOLID_40),
    DASHED: (_SU_DASHED, _DASHED_10, _SU_DASHED, _DASHED_40),
}

#: Where each family's curves end (ksf). Past them GEC-12 recommends API
#: alpha for highly overconsolidated clays (alpha method, p. 247 and Step
#: 1); :func:`adhesion` takes it or holds Ca at the curve's end.
CHART_END_KSF = {SOLID: 3.45, DASHED: 3.05}


def curve_for_pile(pile_type: str) -> str:
    """The Figure 7-17 curve family GEC-12 names for a pile type: smooth
    steel for pipe piles, the corrugated curve for H-piles (p. 247), and
    the concrete and timber curve for those piles."""
    return DASHED if pile_type.startswith("pipe") else SOLID


def _one_curve(su_ksf: float, su_nodes, ca_nodes, lower=None) -> float:
    """Ca on one curve; a D = 40b curve that starts above the chart's start
    takes the shared lower curve (``lower``) below its first node."""
    if lower is not None and su_ksf < su_nodes[0]:
        return lower
    return float(np.interp(su_ksf, su_nodes, ca_nodes))


def adhesion_fig_7_17(su: float, D_over_b: float, curve: str = SOLID,
                      notes: Optional[list] = None) -> float:
    """Pile adhesion Ca (kPa) from GEC-12 Figure 7-17 (alpha method, Step
    1), as drawn.

    Linear along each curve in su, then linear in D/b between the D = 10b
    and D = 40b curves; D/b below 10 takes the 10b curve and above 40 the
    40b curve. Below the curves' start (0.25 ksf) Ca = su times the
    curve's alpha there. Above the curve's end the value is held there;
    :func:`adhesion` decides what to do past it.
    """
    if curve not in _FIG_7_17:
        raise ValueError(f"curve must be one of {list(_FIG_7_17)}, got "
                         f"'{curve}'")
    su_ksf = su / KSF
    s10, c10, s40, c40 = _FIG_7_17[curve]
    if su_ksf < s10[0]:
        a10 = c10[0] / s10[0]
        a40 = (c40[0] if s40[0] == s10[0] else c10[0]) / s10[0]
        ca10, ca40 = a10 * su_ksf, a40 * su_ksf
    else:
        ca10 = _one_curve(su_ksf, s10, c10)
        ca40 = _one_curve(su_ksf, s40, c40, lower=ca10)
    w = min(max((D_over_b - 10.0) / 30.0, 0.0), 1.0)
    return (ca10 + w * (ca40 - ca10)) * KSF


def api_alpha(su: float, sigma_v: float) -> float:
    """Adhesion factor alpha of the API method (GEC-12 Sec. 7.2.1.3.3,
    Eqs. 7-13 to 7-15, p. 251): 0.5 psi^-0.5 for psi <= 1.0 and
    0.5 psi^-0.25 for psi > 1.0, psi = su / sigma'vo. At most 1.0, API RP
    2A's limit (GEC-12 p. 252: "An alpha value of 1.0 is recommended for
    unconsolidated clays")."""
    if sigma_v <= 0:
        return 1.0
    psi = su / sigma_v
    a = 0.5 * psi ** (-0.5) if psi <= 1.0 else 0.5 * psi ** (-0.25)
    return min(a, 1.0)


def adhesion(su: float, D_over_b: float, curve: str = SOLID,
             sigma_v: Optional[float] = None,
             beyond_chart: str = "chart_end",
             notes: Optional[list] = None) -> Tuple[float, str]:
    """(Ca in kPa, its source) for one clay segment by the alpha method,
    Step 1: Figure 7-17 as drawn within its curves (Ca may exceed su).
    Past the curve's end, ``beyond_chart`` = "chart_end" (default) holds Ca
    at the curve's end value and "api" takes API alpha with ``sigma_v``
    (GEC-12 p. 247 and Step 1 recommend it for highly overconsolidated
    clays)."""
    if beyond_chart not in ("api", "chart_end"):
        raise ValueError("beyond_chart must be 'api' or 'chart_end', got "
                         f"'{beyond_chart}'")
    end = CHART_END_KSF[curve] * KSF
    if su <= end:
        return adhesion_fig_7_17(su, D_over_b, curve, notes), "Fig. 7-17"
    if beyond_chart == "api" and sigma_v is not None:
        return api_alpha(su, sigma_v) * su, "API alpha"
    ca = adhesion_fig_7_17(end, D_over_b, curve, notes)
    msg = (f"alpha method: su = {su / KSF:.2f} ksf is past the end of GEC-12 "
           f"Figure 7-17's curves for {CURVES[curve]} ({end / KSF:.2f} ksf); "
           f"Ca was held at the curve's end value.")
    if notes is None:
        warnings.warn(msg, stacklevel=2)
    elif msg not in notes:
        notes.append(msg)
    return ca, "Fig. 7-17 end"


def alpha_tomlinson(cu: float, pile_type: str = "steel",
                    D_over_b: float = 10.0,
                    sigma_v: Optional[float] = None,
                    beyond_chart: str = "chart_end") -> float:
    """Adhesion factor alpha = Ca / su from GEC-12 Figure 7-17 (alpha
    method, Step 1), as drawn: up to about 1.1 where a curve passes above
    Ca = su.

    Parameters
    ----------
    cu : float
        Undrained shear strength su (kPa).
    pile_type : str, optional
        "steel" or "pipe" (smooth steel, the dashed curves), or "h_pile",
        "concrete", "timber", "corrugated" (the solid curves). Default
        "steel".
    D_over_b : float, optional
        D/b of Figure 7-17. Default 10.
    sigma_v : float, optional
        sigma'vo (kPa), for API alpha beyond the chart.
    beyond_chart : str, optional
        "chart_end" (default) or "api"; see :func:`adhesion`.
    """
    if cu <= 0:
        raise ValueError(f"Undrained shear strength must be positive, got {cu}")
    key = pile_type.lower()
    if key in ("steel", "pipe", "pipe_open", "pipe_closed", DASHED):
        curve = DASHED
    elif key in ("h_pile", "concrete", "timber", "corrugated", SOLID):
        curve = SOLID
    else:
        raise ValueError(
            f"Unknown pile_type '{pile_type}'. Options: 'steel', 'h_pile', "
            "'concrete', 'timber'")
    ca, _ = adhesion(cu, D_over_b, curve, sigma_v, beyond_chart, notes=[])
    return ca / cu


def skin_friction_cohesive(cu: float, pile_perimeter: float,
                           layer_thickness: float, D_over_b: float = 10.0,
                           curve: str = SOLID,
                           sigma_v: Optional[float] = None,
                           beyond_chart: str = "chart_end",
                           notes: Optional[list] = None) -> float:
    """Side resistance (kN) of one clay segment: Ca x perimeter x length
    (GEC-12 Eq. 7-10 and Step 3); Ca from :func:`adhesion`."""
    ca, _ = adhesion(cu, D_over_b, curve, sigma_v, beyond_chart, notes)
    return ca * pile_perimeter * layer_thickness


# ---------------------------------------------------------------------------
# Figure 7-18 (Tomlinson 1980): the adhesion factor alpha against su for
# three soil stratigraphy cases (GEC-12 pp. 246-247, the figure p. 249).
# Measured off the printed figure (GEC 12 Vol 1, pdf page 294; the embedded
# raster at its own 150 dpi) on 2026-10-19 by tracing each curve's stroke
# between the fitted gridlines (su every 0.2 ksf, alpha every 0.05), and
# overlaid on the figure to check. D is the pile's penetration into the
# stiff clay: in the top and middle graphs from the top of the stiff clay,
# in the bottom graph (clay from the ground surface) from the ground
# surface; b is the pile width.
# ---------------------------------------------------------------------------

#: The three cases, and the graph of Figure 7-18 each is read from.
SAND_OVER_STIFF_CLAY = "sand_over_stiff_clay"
SOFT_OVER_STIFF_CLAY = "soft_clay_over_stiff_clay"
STIFF_CLAY = "stiff_clay"
FIG_7_18_CASES = {
    SAND_OVER_STIFF_CLAY: "sands or sandy gravels over stiff clay (top graph)",
    SOFT_OVER_STIFF_CLAY: "soft clay over stiff clay (middle graph)",
    STIFF_CLAY: "stiff clay without any different overlying strata "
                "(bottom graph)",
}


def _tenths(a: float, b: float) -> list:
    return [round(a + 0.1 * i, 1) for i in range(int(round((b - a) * 10)) + 1)]


#: Each case's curves: (D/b, su in ksf, alpha), shallowest curve first.
#: Outside the curves' D/b the nearest curve is held ("D < 10b", "D > 20b",
#: "D > 40b" in the legends).
_FIG_7_18 = {
    SAND_OVER_STIFF_CLAY: [
        # "D < 10b": alpha = 1.0 over the whole chart
        (10.0, [0.0, 5.0], [1.0, 1.0]),
        (20.0, [0.0, 1.67, 1.8] + _tenths(1.9, 3.0) + [5.0],
         [1.0, 1.0, 0.982, 0.969, 0.955, 0.938, 0.918, 0.897, 0.866, 0.839,
          0.813, 0.790, 0.770, 0.757, 0.754, 0.752]),
        (40.0, [0.0, 0.57, 0.7] + _tenths(0.8, 3.4) + [5.0],
         [1.0, 1.0, 0.982, 0.969, 0.956, 0.939, 0.922, 0.901, 0.881, 0.856,
          0.830, 0.801, 0.771, 0.744, 0.713, 0.680, 0.647, 0.616, 0.589,
          0.562, 0.532, 0.500, 0.469, 0.444, 0.428, 0.415, 0.403, 0.394,
          0.392, 0.390, 0.390]),
    ],
    SOFT_OVER_STIFF_CLAY: [
        (10.0, [0.45] + _tenths(0.5, 4.5) + [4.56],
         [0.530, 0.519, 0.493, 0.467, 0.439, 0.412, 0.396, 0.380, 0.368,
          0.354, 0.342, 0.333, 0.324, 0.316, 0.307, 0.300, 0.296, 0.295,
          0.291, 0.282, 0.276, 0.272, 0.270, 0.266, 0.258, 0.249, 0.245,
          0.241, 0.237, 0.233, 0.229, 0.225, 0.221, 0.217, 0.213, 0.209,
          0.207, 0.201, 0.199, 0.195, 0.191, 0.185, 0.183]),
        (20.0, [0.52] + _tenths(0.6, 4.7) + [4.76],
         [0.833, 0.814, 0.791, 0.771, 0.756, 0.743, 0.732, 0.720, 0.709,
          0.700, 0.692, 0.685, 0.680, 0.679, 0.676, 0.674, 0.672, 0.670,
          0.666, 0.660, 0.654, 0.650, 0.644, 0.640, 0.632, 0.624, 0.617,
          0.610, 0.606, 0.601, 0.594, 0.589, 0.584, 0.577, 0.571, 0.567,
          0.561, 0.555, 0.551, 0.545, 0.539, 0.533, 0.527, 0.524]),
    ],
    STIFF_CLAY: [
        (10.0, [0.5] + _tenths(0.6, 3.6) + [5.0],
         [0.985, 0.972, 0.959, 0.951, 0.940, 0.924, 0.907, 0.893, 0.876,
          0.854, 0.830, 0.805, 0.782, 0.757, 0.726, 0.691, 0.646, 0.592,
          0.536, 0.487, 0.442, 0.403, 0.372, 0.348, 0.321, 0.295, 0.281,
          0.267, 0.255, 0.247, 0.243, 0.239, 0.238]),
        (40.0, [0.5, 1.55] + _tenths(1.7, 4.2) + [5.0],
         [1.0, 1.0, 0.980, 0.962, 0.936, 0.902, 0.861, 0.819, 0.781, 0.741,
          0.703, 0.662, 0.617, 0.572, 0.535, 0.500, 0.472, 0.446, 0.425,
          0.406, 0.388, 0.373, 0.355, 0.339, 0.326, 0.313, 0.305, 0.299,
          0.300]),
    ],
}


def _fig_7_18_curves(case: str, D_over_b: float):
    """The curves Figure 7-18's ``case`` reads at ``D_over_b``: a list of
    (weight, su nodes, alpha nodes), linear in D/b between the bracketing
    curves and the nearest curve held outside them."""
    if case not in _FIG_7_18:
        raise ValueError(f"Figure 7-18 case must be one of "
                         f"{list(_FIG_7_18)}, got '{case}'")
    curves = _FIG_7_18[case]
    if D_over_b <= curves[0][0]:
        return [(1.0, curves[0][1], curves[0][2])]
    for (d0, s0, a0), (d1, s1, a1) in zip(curves[:-1], curves[1:]):
        if D_over_b <= d1:
            w = (D_over_b - d0) / (d1 - d0)
            return [(1.0 - w, s0, a0), (w, s1, a1)]
    return [(1.0, curves[-1][1], curves[-1][2])]


def fig_7_18_range_ksf(case: str, D_over_b: float) -> Tuple[float, float]:
    """(start, end) of su (ksf) the curves read at ``D_over_b`` cover."""
    used = [c for c in _fig_7_18_curves(case, D_over_b) if c[0] > 0]
    return (max(s[0] for _, s, _ in used), min(s[-1] for _, s, _ in used))


def alpha_fig_7_18(su: float, D_over_b: float, case: str,
                   notes: Optional[list] = None) -> float:
    """Adhesion factor alpha from GEC-12 Figure 7-18 for ``case`` (one of
    :data:`FIG_7_18_CASES`) at su (kPa) and D/b, D the penetration into
    the stiff clay. Linear along each curve in su, then linear in D/b
    between the curves; outside a curve's su range its end value is held
    (a note below the start; :func:`adhesion_fig_7_18` decides past the
    end)."""
    su_ksf = su / KSF
    start, _ = fig_7_18_range_ksf(case, D_over_b)
    if su_ksf < start:
        _note(notes, (
            f"su = {su_ksf:.2f} ksf is below GEC-12 Figure 7-18's "
            f"{FIG_7_18_CASES[case]} curves (they start near {start:g} "
            f"ksf); alpha was held at the curves' starting value."))
    lowest = _FIG_7_18[case][0][0]
    if case != SAND_OVER_STIFF_CLAY and D_over_b < lowest:
        _note(notes, (
            f"The penetration into the stiff clay, D = {D_over_b:.1f}b, is "
            f"less than the shallowest curve of GEC-12 Figure 7-18's "
            f"{FIG_7_18_CASES[case]} (D = {lowest:g}b); alpha was read off "
            f"that curve. GEC-12 p. 247 says this alpha rises with "
            f"penetration, so it is overstated for a shallower one."))
    return sum(w * float(np.interp(su_ksf, s, a))
               for w, s, a in _fig_7_18_curves(case, D_over_b))


def adhesion_fig_7_18(su: float, D_over_b: float, case: str,
                      sigma_v: Optional[float] = None,
                      beyond_chart: str = "chart_end",
                      notes: Optional[list] = None) -> Tuple[float, str]:
    """(Ca in kPa, its source) from GEC-12 Figure 7-18 (alpha method, Step
    1): alpha su within the chart. Past the curves' end ``beyond_chart`` =
    "chart_end" (default) holds Ca at the chart's end value, as
    :func:`adhesion` does for Figure 7-17, and "api" takes API alpha with
    ``sigma_v`` (p. 247 recommends it for highly overconsolidated
    clays)."""
    if beyond_chart not in ("api", "chart_end"):
        raise ValueError("beyond_chart must be 'api' or 'chart_end', got "
                         f"'{beyond_chart}'")
    end = fig_7_18_range_ksf(case, D_over_b)[1] * KSF
    if su <= end:
        return alpha_fig_7_18(su, D_over_b, case, notes) * su, "Fig. 7-18"
    if beyond_chart == "api" and sigma_v is not None:
        return api_alpha(su, sigma_v) * su, "API alpha"
    ca = alpha_fig_7_18(end, D_over_b, case, notes) * end
    _note(notes, (
        f"alpha method: su = {su / KSF:.2f} ksf is past the end of GEC-12 "
        f"Figure 7-18's {FIG_7_18_CASES[case]} curves ({end / KSF:.2f} "
        f"ksf); Ca was held at the chart's end value."))
    return ca, "Fig. 7-18 end"


#: The adhesion sources that mean su was past a chart's curves.
BEYOND_SOURCES = ("Fig. 7-17 end", "Fig. 7-18 end", "API alpha")


def end_bearing_cohesive(cu_tip: float, tip_area: float) -> float:
    """Compute end bearing in cohesive soil.

    Qt = 9 * cu_tip * At

    Parameters
    ----------
    cu_tip : float
        Undrained shear strength at pile tip (kPa).
    tip_area : float
        Pile tip area (m²).

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    FHWA GEC-12, Eq. 7-11 (Nc = 9 for deep foundations).
    """
    return 9.0 * cu_tip * tip_area
