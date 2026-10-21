"""The alpha method's charts for driven piles in clay, as the references
print them, shared by the analysis modules (they do not import each other).

Unit side resistance fs = alpha su (UFC 3-220-20 Eq 6-8; FHWA GEC-12 Eq.
7-10, fs = Ca = alpha su). Two printed sources, both after Tomlinson:

UFC 3-220-20 (16 Jan 2025) §6-5.4.2, "Shaft Resistance for Saturated,
Undrained Soils - Alpha Method" (printed pp. 414-417):

* Table 6-16 (p. 416, Tomlinson 1994) by the clay's consistency: soft clay,
  su below 800 psf (su/Pa below 0.38), takes alpha = 1 (fs = su), "or su ~ 0
  -> fs = 0" where the designer ignores it; firm (800-1500 psf), stiff
  (1500-3000) and very stiff (3000-6000) clay take curves A and B of Figure
  6-14;
* Figure 6-14 (p. 417): alpha against su/Pa, curve A for driven piles with
  Z of 40b or more, curve B for Z = 10b (Tomlinson 1994). Z is "the distance
  from the ground surface to the bottom of the clay layer, sublayer, or
  pile, whichever comes first", b the pile width or diameter (p. 415).
  alpha is "limited to values between zero and 1.0" (p. 415). The chart has
  no pile-type curves: Tomlinson (1970) found "pile texture has no
  discernable effect" (p. 415). The curves were read off the figure's own
  vector paths (the PDF draws them as Bezier curves on axes ticked every
  0.1 su/Pa and 0.02 alpha) on 2026-10-20; curve A passes alpha 0.50 at
  su/Pa 1.42, where UFC's drag example reads 0.5 for its stiff clay (App.
  B-5.4, Fig. B-17, pp. 636-637);
* the transition zone (pp. 416-417): the top 10b of a stiff clay under soft
  clay takes Eq 6-9, alpha = 0.28 (su/Pa)^-0.44 (with "the unit shaft
  resistance from the soft clay layer" where that is higher); under sand or
  gravel alpha = 1.0 there (Tomlinson 1980). Below the zone Figure 6-14 as
  usual. Tomlinson gives no guidance for firm clay under soft clay or sand:
  "judgment should be applied" (p. 417).

FHWA GEC-12 (FHWA-NHI-16-009) Vol I §7.2.1.3.2, the alpha method, Step 1
(p. 250): Ca read off Figure 7-17 (after Tomlinson 1979, p. 248) by su, the
pile type and D/b, as drawn, with no limit on Ca (the D = 40b concrete,
timber and corrugated curve passes above Ca = su). Solid curves: concrete,
timber and corrugated steel piles (an H-pile takes the corrugated curve, p.
247); dashed: smooth steel piles. D runs from the ground surface to the
bottom of the clay layer, or to the pile toe if that is less (p. 246).
Figure 7-17's curves were measured off the printed figure on 2026-10-18
(``axial_pile`` GM-121 / GM-229; they moved here for ``downdrag``, GM-255).

Units are SI (kPa) unless a name says ksf.
"""

from typing import List, Optional, Tuple

import numpy as np

#: 1 ksf and 1 psf in kPa.
KSF = 47.880259
PSF = KSF / 1000.0

#: Atmospheric pressure Pa (kPa): UFC 3-220-20 takes su/Pa "in same units
#: as su" (Table 6-16: 800 psf is su/Pa 0.38, Pa about 2116 psf).
PA_KPA = 101.325


def _note(notes: Optional[list], text: str) -> None:
    """Add ``text`` to ``notes`` once."""
    if notes is not None and text not in notes:
        notes.append(text)


# ---------------------------------------------------------------------------
# UFC 3-220-20 Table 6-16 and Figure 6-14 (Tomlinson 1994)
# ---------------------------------------------------------------------------

#: Table 6-16's consistencies by su (psf, as the table prints them): soft
#: below 800, firm 800-1500, stiff 1500-3000, very stiff 3000-6000.
UFC_SOFT_BELOW_KPA = 800 * PSF
UFC_STIFF_FROM_KPA = 1500 * PSF
UFC_VERY_STIFF_FROM_KPA = 3000 * PSF
UFC_VERY_STIFF_TO_KPA = 6000 * PSF

#: Figure 6-14, curve A (driven piles, Z >= 40b) and curve B (Z = 10b):
#: (su/Pa, alpha), from the figure's vector paths. Curve A runs along the
#: top of the chart (alpha 1.0) until su/Pa 0.724; both curves end, flat,
#: at su/Pa 2.357.
UFC_CURVE_A = (
    [0.724, 0.773, 0.823, 0.873, 0.924, 0.98, 1.036, 1.086, 1.134, 1.184,
     1.233, 1.281, 1.331, 1.388, 1.45, 1.519, 1.588, 1.648, 1.706, 1.765,
     1.824, 1.887, 1.952, 2.013, 2.09, 2.216, 2.357],
    [1.0, 0.988, 0.973, 0.95, 0.918, 0.872, 0.822, 0.78, 0.74, 0.7, 0.658,
     0.611, 0.564, 0.52, 0.479, 0.442, 0.411, 0.389, 0.37, 0.349, 0.329,
     0.313, 0.301, 0.298, 0.299, 0.299, 0.299])
UFC_CURVE_B = (
    [0.233, 0.28, 0.327, 0.375, 0.423, 0.47, 0.518, 0.567, 0.617, 0.669,
     0.721, 0.776, 0.83, 0.878, 0.924, 0.966, 1.005, 1.04, 1.074, 1.113,
     1.153, 1.196, 1.242, 1.296, 1.351, 1.398, 1.449, 1.515, 1.587, 1.644,
     1.735, 2.016, 2.357],
    [0.984, 0.971, 0.959, 0.95, 0.94, 0.925, 0.907, 0.892, 0.874, 0.85,
     0.822, 0.795, 0.767, 0.738, 0.707, 0.67, 0.63, 0.589, 0.548, 0.507,
     0.466, 0.428, 0.392, 0.36, 0.332, 0.309, 0.288, 0.267, 0.25, 0.242,
     0.238, 0.237, 0.238])

#: Where Figure 6-14's curves A and B end (su/Pa).
UFC_CURVE_END = 2.357

#: Where each UFC reading comes from, as a result names it.
UFC_TABLE_SOFT = "Table 6-16, soft clay"
UFC_FIG = "Fig 6-14"
UFC_FIG_END = "Fig 6-14, held at the curves' end"
UFC_EQ_6_9 = "Eq 6-9, transition zone under soft clay"
UFC_SAND_ZONE = "transition zone under sand, alpha 1 (Tomlinson 1980)"


def ufc_consistency(su: float) -> str:
    """The clay's consistency by UFC 3-220-20 Table 6-16 (su in kPa):
    "soft", "firm", "stiff", "very stiff", or "harder than very stiff"
    (above 6000 psf, which the table does not cover)."""
    if su < UFC_SOFT_BELOW_KPA:
        return "soft"
    if su < UFC_STIFF_FROM_KPA:
        return "firm"
    if su < UFC_VERY_STIFF_FROM_KPA:
        return "stiff"
    if su <= UFC_VERY_STIFF_TO_KPA:
        return "very stiff"
    return "harder than very stiff"


def alpha_ufc(su: float, Z_over_b: float,
              notes: Optional[List[str]] = None) -> Tuple[float, str]:
    """(alpha, its source) for a driven pile in clay by UFC 3-220-20
    §6-5.4.2: alpha = 1 in soft clay (Table 6-16), else Figure 6-14, curve
    A at Z/b of 40 or more, curve B at Z/b = 10, linear in Z/b between.

    Z/b below 10 reads curve B and says so in ``notes`` (UFC p. 415,
    Tomlinson 1994: alpha is "reduced more at lower ratios of normalized
    embedment", so curve B overstates it there). su/Pa past the curves' end
    (2.357, where both are flat) holds alpha at the end and says so.
    """
    if su <= 0:
        raise ValueError(f"su must be positive, got {su}")
    if su < UFC_SOFT_BELOW_KPA:
        return 1.0, UFC_TABLE_SOFT
    x = su / PA_KPA
    source = UFC_FIG
    if x > UFC_CURVE_END:
        _note(notes, (
            f"su/Pa = {x:.2f} is past the end of UFC 3-220-20 Fig 6-14's "
            f"curves A and B ({UFC_CURVE_END:g}, where both are flat); alpha "
            f"was held at the curves' end."))
        x, source = UFC_CURVE_END, UFC_FIG_END
    if Z_over_b < 10.0:
        _note(notes, (
            f"Z/b = {Z_over_b:.1f} is less than curve B's 10 (UFC 3-220-20 "
            f"Fig 6-14); alpha was read off curve B, which overstates it for "
            f"a shorter embedment (p. 415: Tomlinson reduces alpha more at "
            f"lower Z/b)."))
    a_curve = float(np.interp(x, *UFC_CURVE_A, left=1.0))
    b_curve = float(np.interp(x, *UFC_CURVE_B, left=UFC_CURVE_B[1][0]))
    w = min(max((Z_over_b - 10.0) / 30.0, 0.0), 1.0)
    return min(b_curve + w * (a_curve - b_curve), 1.0), source


def alpha_ufc_eq_6_9(su: float) -> float:
    """alpha in the transition zone, the top 10b of a stiff clay under soft
    clay: UFC 3-220-20 Eq 6-9 (p. 416), 0.28 (su/Pa)^-0.44, su that of the
    stiff clay; at most 1.0."""
    if su <= 0:
        raise ValueError(f"su must be positive, got {su}")
    return min(0.28 * (su / PA_KPA) ** -0.44, 1.0)


# ---------------------------------------------------------------------------
# FHWA GEC-12 Figure 7-17 (Tomlinson 1979)
# ---------------------------------------------------------------------------

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
FIG_7_17 = {
    SOLID: (_SU_SOLID, _SOLID_10, _SU_SOLID_40, _SOLID_40),
    DASHED: (_SU_DASHED, _DASHED_10, _SU_DASHED, _DASHED_40),
}

#: Where each family's curves end (ksf). Past them GEC-12 recommends API
#: alpha for highly overconsolidated clays (alpha method, p. 247 and Step
#: 1).
CHART_END_KSF = {SOLID: 3.45, DASHED: 3.05}


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
    the caller decides what to do past it (:data:`CHART_END_KSF`).
    """
    if curve not in FIG_7_17:
        raise ValueError(f"curve must be one of {list(FIG_7_17)}, got "
                         f"'{curve}'")
    su_ksf = su / KSF
    s10, c10, s40, c40 = FIG_7_17[curve]
    if su_ksf < s10[0]:
        a10 = c10[0] / s10[0]
        a40 = (c40[0] if s40[0] == s10[0] else c10[0]) / s10[0]
        ca10, ca40 = a10 * su_ksf, a40 * su_ksf
    else:
        ca10 = _one_curve(su_ksf, s10, c10)
        ca40 = _one_curve(su_ksf, s40, c40, lower=ca10)
    w = min(max((D_over_b - 10.0) / 30.0, 0.0), 1.0)
    return (ca10 + w * (ca40 - ca10)) * KSF
