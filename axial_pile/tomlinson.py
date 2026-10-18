"""
The alpha-method for driven piles in cohesive soils (total stress).

Unit side resistance fs = Ca = alpha su (GEC-12 Eq. 7-10), with the pile
adhesion Ca read from GEC-12 Figure 7-17 (after Tomlinson 1979) by the
undrained shear strength su, the pile surface and D/b (GEC-12 Sec.
7.2.1.3.2, Step 1, printed pp. 246-250):

* solid curves: concrete, timber and corrugated steel piles; H-piles use
  the corrugated curve on the "box" area (p. 247);
* dashed curves: smooth steel piles;
* D/b = 10 and D/b = 40, where D is the distance from the ground surface to
  the bottom of the clay layer, or to the pile toe if less (p. 246 and the
  figure's legend);
* where su exceeds the chart (highly overconsolidated clays) GEC-12
  recommends alpha from API RP 2A, Eqs. 7-13 to 7-15 (pp. 247, 250-252);
  :func:`adhesion` holds Ca at the curve's end by default, as GEC-12's
  worked example does (Vol 3, Table D-100), or takes API alpha;
* Ca is at most su (alpha <= 1): API RP 2A's alpha is at most 1.0 (p. 252)
  and p. 247 takes su on an H-pile's soil-to-soil faces, while Figure 7-17
  as drawn passes above Ca = su at low su.

Toe resistance qp = 9 su (Eq. 7-11). Figure 7-17 was measured off the
printed figure (GEC 12 Vol 1, pdf page 293) on 2026-10-18 by tracing each
curve between the fitted gridlines. Units are SI (kPa, m, kN).

References:
    Tomlinson, M.J. (1979) -- Pile Design and Construction Practice
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Sec. 7.2.1.3.2 and 7.2.1.3.3
    API RP 2A (1993)
"""

import warnings
from typing import Optional, Tuple

import numpy as np

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

#: Where each family's curves end (ksf): beyond, GEC-12 p. 247 recommends
#: API alpha; :func:`adhesion` holds Ca at the end by default.
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
    """Pile adhesion Ca (kPa) from GEC-12 Figure 7-17.

    Linear along each curve in su, then linear in D/b between the D = 10b
    and D = 40b curves; D/b below 10 takes the 10b curve and above 40 the
    40b curve. Below the chart's start (0.25 ksf) Ca = su times the
    curve's starting alpha, at most 1. Above the curve's end the value is
    held there. The chart as drawn: :func:`adhesion` limits Ca to su.
    """
    if curve not in _FIG_7_17:
        raise ValueError(f"curve must be one of {list(_FIG_7_17)}, got "
                         f"'{curve}'")
    su_ksf = su / KSF
    s10, c10, s40, c40 = _FIG_7_17[curve]
    if su_ksf < s10[0]:
        a10 = min(c10[0] / s10[0], 1.0)
        a40 = min((c40[0] if s40[0] == s10[0] else c10[0]) / s10[0], 1.0)
        ca10, ca40 = a10 * su_ksf, a40 * su_ksf
    else:
        ca10 = _one_curve(su_ksf, s10, c10)
        ca40 = _one_curve(su_ksf, s40, c40, lower=ca10)
    w = min(max((D_over_b - 10.0) / 30.0, 0.0), 1.0)
    return (ca10 + w * (ca40 - ca10)) * KSF


def api_alpha(su: float, sigma_v: float) -> float:
    """Adhesion factor alpha of API RP 2A (GEC-12 Eqs. 7-13 to 7-15):
    0.5 psi^-0.5 for psi <= 1, 0.5 psi^-0.25 for psi > 1, psi = su /
    sigma'vo; at most 1.0 (GEC-12 p. 252 recommends 1.0 for
    unconsolidated clays)."""
    if sigma_v <= 0:
        return 1.0
    psi = su / sigma_v
    a = 0.5 * psi ** (-0.5) if psi <= 1.0 else 0.5 * psi ** (-0.25)
    return min(a, 1.0)


#: The source :func:`adhesion` gives where Figure 7-17 passes above su.
AT_MOST_SU = "Fig. 7-17, at most su"


def adhesion(su: float, D_over_b: float, curve: str = SOLID,
             sigma_v: Optional[float] = None,
             beyond_chart: str = "chart_end",
             notes: Optional[list] = None) -> Tuple[float, str]:
    """(Ca in kPa, its source) for one clay segment: Figure 7-17 within the
    chart, at most su (alpha <= 1); beyond the curve's end, ``beyond_chart``
    = "chart_end" (default) holds Ca at the curve's end value, and "api"
    takes API alpha (GEC-12 p. 247, Step 1) with ``sigma_v``."""
    if beyond_chart not in ("api", "chart_end"):
        raise ValueError("beyond_chart must be 'api' or 'chart_end', got "
                         f"'{beyond_chart}'")
    end = CHART_END_KSF[curve] * KSF
    if su <= end:
        ca = adhesion_fig_7_17(su, D_over_b, curve, notes)
        if ca > su:
            return su, AT_MOST_SU
        return ca, "Fig. 7-17"
    if beyond_chart == "api" and sigma_v is not None:
        return api_alpha(su, sigma_v) * su, "API alpha"
    ca = adhesion_fig_7_17(end, D_over_b, curve, notes)
    msg = (f"su = {su / KSF:.2f} ksf is beyond GEC-12 Figure 7-17's "
           f"{CURVES[curve]} curves (they end near {end / KSF:.2f} ksf); "
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
    """Adhesion factor alpha = Ca / su from GEC-12 Figure 7-17.

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
