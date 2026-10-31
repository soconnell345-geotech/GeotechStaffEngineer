"""
Prefabricated vertical drain (wick drain / PVD) analysis.

Radial consolidation as GEC-13 (FHWA-NHI-16-027) Vol. I, Ch. 2, Sec. 4.2
prints it (Rixner et al. 1986a, after Barron 1948), combined with Terzaghi
vertical consolidation from the settlement module (GM-291):

- Cylinder of influence:  d_c = 1.05 s (triangular) or 1.13 s (square)
  (p. 2-33)
- Eq. 2-1:  t = (d_c^2 / 8 c_h) [F(n) + F_s + F_r] ln[1 / (1 - U_h)], so
  U_h = 1 - exp(-8 T_r / F) with T_r = c_h t / d_c^2 and F the bracket
- Eq. 2-2:  F(n) = ln(d_c / d_w) - 0.75 ("simplified")
- Eq. 2-3:  d_w = 2 (a + b) / pi, a band drain's equivalent diameter
- Smear: GEC-13 prints no formula for F_s, only that smear is "often
  ignored for typical projects" and that "a sample disturbance factor
  F_s ~ 2 may be considered" in highly plastic, sensitive soils whose c_h
  is accurately determined (p. 2-35). The tool takes UFC 3-220-10 (2026)
  Eq. 5-25 as printed, F_s ~ (k_h / k_s) ln(s) (p. 267), and ignores smear
  by default (F_s = 0), as both manuals do.
- Well resistance: not modelled (F_r = 0). GEC-13 p. 2-36: "Well
  resistance is rarely significant, and, therefore, ignored except for
  extremely deep drains or where a combination of high loads and very
  permeable soils is present."
- Combined vertical and radial: U = 1 - (1 - U_v)(1 - U_r), Carrillo
  (1942) as UFC 3-220-10 Eq. 5-27 prints it (p. 268). GEC-13's Eq. 2-1 is
  radial only; p. 2-34: "for a conservative approach, no beneficial
  effects may be considered for vertical drainage in preliminary
  feasibility evaluations". Each result gives U_r beside the combined U.

All units SI: meters, years, m²/year.

References:
    FHWA GEC-13 (FHWA-NHI-16-027), Ground Modification Methods Reference
    Manual, Vol. I, Ch. 2, pp. 2-30 to 2-38
    UFC 3-220-10, Soil Mechanics (2026), Sec. 5-7.4, pp. 266 to 268
"""

import math
from typing import List, Optional, Tuple

from settlement.time_rate import time_factor, degree_of_consolidation
from ground_improvement.results import WickDrainResult

#: Equivalent drain diameter by default: 2.5 in (0.0635 m). GEC-13 p. 2-35:
#: "Various diameters ranging from 1.5 inch to 5.5 inch have been used, but
#: the most common is to use 2.5 inch. In fact, there is little difference
#: in design if 2.0 inch to 4.0 inch is used."
DW_DEFAULT_M = 2.5 * 0.0254


def equivalent_drain_diameter(width: float, thickness: float) -> float:
    """Compute equivalent circular diameter for a band-shaped PVD.

    GEC-13 Eq. 2-3 (p. 2-35, after Hansbo 1979): a cylinder of the same
    circumference as the band drain,
        dw = 2 * (a + b) / pi
    with a the PVD's thickness and b its width.

    Parameters
    ----------
    width : float
        Drain width (m), typically ~0.1 m.
    thickness : float
        Drain thickness (m), typically ~0.003-0.005 m.

    Returns
    -------
    float
        Equivalent drain diameter dw (m).
    """
    if width <= 0 or thickness <= 0:
        raise ValueError(f"Drain dimensions must be positive, got width={width}, thickness={thickness}")
    return 2.0 * (width + thickness) / math.pi


def influence_diameter(spacing: float, pattern: str = "triangular") -> float:
    """Compute influence zone diameter de: GEC-13 p. 2-33, d_c is 1.05
    times the spacing in an equilateral triangular pattern and 1.13 times
    it in a square pattern.

    Parameters
    ----------
    spacing : float
        Center-to-center drain spacing (m).
    pattern : str
        'triangular' or 'square'.

    Returns
    -------
    float
        Influence diameter de (m).
    """
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")

    if pattern == "triangular":
        return 1.05 * spacing
    elif pattern == "square":
        return 1.13 * spacing
    else:
        raise ValueError(f"Pattern must be 'triangular' or 'square', got '{pattern}'")


def drain_function_F(n_ratio: float, smear_ratio: float = 1.0,
                     kh_ks_ratio: float = 1.0) -> float:
    """The bracket of GEC-13 Eq. 2-1 without well resistance: F(n) + F_s.

    F(n) = ln(n) - 0.75, GEC-13 Eq. 2-2 ("simplified", p. 2-32), n = d_c /
    d_w. F_s = (k_h / k_s) ln(s), UFC 3-220-10 (2026) Eq. 5-25 as printed
    (p. 267; GEC-13 prints no formula for F_s), with s the ratio of the
    disturbed zone's diameter to the drain's. smear_ratio 1 is no smear,
    F_s = 0 (GM-291; until then the smear term was Hansbo's ln(n/s) +
    (k_h/k_s) ln(s) - 0.75, printed in neither manual).

    Parameters
    ----------
    n_ratio : float
        Spacing ratio n = de/dw.
    smear_ratio : float
        Smear zone ratio s = ds/dw. Default 1.0 (no smear).
    kh_ks_ratio : float
        Horizontal permeability ratio kh/ks. Default 1.0. Used only with
        smear_ratio above 1.

    Returns
    -------
    float
        F(n) + F_s.
    """
    if n_ratio <= 1.0:
        raise ValueError(f"Spacing ratio n must be > 1, got {n_ratio}")
    if smear_ratio < 1.0:
        raise ValueError(f"Smear ratio must be >= 1, got {smear_ratio}")
    if kh_ks_ratio <= 0.0:
        raise ValueError(f"kh/ks must be positive, got {kh_ks_ratio}")

    return math.log(n_ratio) - 0.75 + smear_factor(smear_ratio, kh_ks_ratio)


def smear_factor(smear_ratio: float, kh_ks_ratio: float) -> float:
    """F_s = (k_h / k_s) ln(s), UFC 3-220-10 (2026) Eq. 5-25 as printed
    (p. 267): "In order to account for smear, the soil disturbance factor
    (F_s) can be calculated as" F_s ~ (k_h / k_s) ln(s); 0 with no smear
    (s = 1). GEC-13 p. 2-35 prints no formula: "a sample disturbance factor
    F_s ~ 2 may be considered" in highly plastic, sensitive soils whose c_h
    is accurately determined."""
    if smear_ratio <= 1.0:
        return 0.0
    return kh_ks_ratio * math.log(smear_ratio)


def radial_time_factor(ch: float, t: float, de: float) -> float:
    """Compute radial time factor Tr.

    Tr = ch * t / de²

    Parameters
    ----------
    ch : float
        Horizontal coefficient of consolidation (m²/year).
    t : float
        Time (years).
    de : float
        Influence zone diameter (m).

    Returns
    -------
    float
        Radial time factor Tr (dimensionless).
    """
    if de <= 0:
        raise ValueError(f"Influence diameter must be positive, got {de}")
    return ch * t / de**2


def radial_degree_of_consolidation(Tr: float, F_n: float) -> float:
    """Compute radial degree of consolidation Ur.

    Ur = 1 - exp(-8 * Tr / F(n))

    Parameters
    ----------
    Tr : float
        Radial time factor.
    F_n : float
        Drain function F(n).

    Returns
    -------
    float
        Radial degree of consolidation (percent, 0-100).
    """
    if F_n <= 0:
        raise ValueError(f"Drain function F(n) must be positive, got {F_n}")
    if Tr <= 0:
        return 0.0
    Ur = 1.0 - math.exp(-8.0 * Tr / F_n)
    return min(Ur * 100.0, 100.0)


def combined_degree_of_consolidation(Uv_pct: float, Ur_pct: float) -> float:
    """Compute combined vertical + radial degree of consolidation.

    U_total = 1 - (1 - Uv/100) * (1 - Ur/100)

    Parameters
    ----------
    Uv_pct : float
        Vertical degree of consolidation (percent).
    Ur_pct : float
        Radial degree of consolidation (percent).

    Returns
    -------
    float
        Combined degree of consolidation (percent, 0-100).
    """
    Uv = min(Uv_pct / 100.0, 1.0)
    Ur = min(Ur_pct / 100.0, 1.0)
    return (1.0 - (1.0 - Uv) * (1.0 - Ur)) * 100.0


def time_for_radial_consolidation(Ur_target: float, ch: float,
                                   de: float, F_n: float) -> float:
    """Solve for time to reach a target radial degree of consolidation.

    From Ur = 1 - exp(-8*Tr/F_n):
        Tr = -F_n/8 * ln(1 - Ur/100)
        t = Tr * de² / ch

    Parameters
    ----------
    Ur_target : float
        Target radial consolidation (percent, 0-100).
    ch : float
        Horizontal coefficient of consolidation (m²/year).
    de : float
        Influence zone diameter (m).
    F_n : float
        Drain function F(n).

    Returns
    -------
    float
        Time to reach target Ur (years).
    """
    if Ur_target <= 0:
        return 0.0
    if Ur_target >= 100:
        raise ValueError("Cannot compute time for 100% radial consolidation")
    if ch <= 0:
        raise ValueError(f"ch must be positive, got {ch}")

    Ur_frac = Ur_target / 100.0
    Tr = -F_n / 8.0 * math.log(1.0 - Ur_frac)
    return Tr * de**2 / ch


def design_drain_spacing(
    target_U: float,
    target_time: float,
    ch: float,
    cv: float,
    Hdr: float,
    dw: float = DW_DEFAULT_M,
    pattern: str = "triangular",
    smear_ratio: float = 1.0,
    kh_ks_ratio: float = 1.0,
    spacing_range: Tuple[float, float] = (1.0, 3.5),
) -> WickDrainResult:
    """Find drain spacing to achieve target consolidation in target time.

    Uses bisection to find spacing s such that U_total(t=target_time) >= target_U.

    Parameters
    ----------
    target_U : float
        Target combined degree of consolidation (percent).
    target_time : float
        Time to achieve target (years).
    ch : float
        Horizontal coefficient of consolidation (m²/year).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    dw : float
        Equivalent drain diameter (m). Default 0.0635 (2.5 in, "the most
        common", GEC-13 p. 2-35).
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    smear_ratio : float
        Smear zone ratio ds/dw. Default 1.0: smear ignored, as GEC-13 p.
        2-35 says it often is for typical projects.
    kh_ks_ratio : float
        Permeability ratio kh/ks, with smear_ratio above 1. Default 1.0.
    spacing_range : tuple of (float, float)
        Search range for spacing (m). Default (1.0, 3.5).

    Returns
    -------
    WickDrainResult
        Result with the designed spacing and consolidation details.
    """
    if target_U >= 100:
        raise ValueError("Target U must be < 100%")
    if target_U <= 0:
        raise ValueError("Target U must be > 0%")
    if target_time <= 0:
        raise ValueError(f"Target time must be positive, got {target_time}")
    if ch <= 0:
        raise ValueError(f"ch must be positive, got {ch}")
    if cv <= 0:
        raise ValueError(f"cv must be positive, got {cv}")

    # Compute vertical consolidation at target time (independent of spacing)
    Tv = time_factor(cv, target_time, Hdr)
    Uv = degree_of_consolidation(Tv)

    def _combined_U_at_spacing(s):
        de = influence_diameter(s, pattern)
        n_ratio = de / dw
        F_n = drain_function_F(n_ratio, smear_ratio, kh_ks_ratio)
        Tr = radial_time_factor(ch, target_time, de)
        Ur = radial_degree_of_consolidation(Tr, F_n)
        return combined_degree_of_consolidation(Uv, Ur)

    s_lo, s_hi = spacing_range

    # Check if tightest spacing achieves target
    U_tight = _combined_U_at_spacing(s_lo)
    if U_tight < target_U:
        # Even tightest spacing isn't enough — return tightest with warning
        s_found = s_lo
    else:
        # Check if widest spacing already achieves target
        U_wide = _combined_U_at_spacing(s_hi)
        if U_wide >= target_U:
            s_found = s_hi
        else:
            # Bisection: find largest spacing that achieves target
            for _ in range(50):
                s_mid = (s_lo + s_hi) / 2.0
                U_mid = _combined_U_at_spacing(s_mid)
                if U_mid >= target_U:
                    s_lo = s_mid  # can go wider
                else:
                    s_hi = s_mid  # need tighter
                if s_hi - s_lo < 0.01:
                    break
            s_found = s_lo  # conservative (tighter side)

    # Build result at found spacing
    return analyze_wick_drains(
        spacing=s_found,
        ch=ch,
        cv=cv,
        Hdr=Hdr,
        time=target_time,
        dw=dw,
        pattern=pattern,
        smear_ratio=smear_ratio,
        kh_ks_ratio=kh_ks_ratio,
    )


def analyze_wick_drains(
    spacing: float,
    ch: float,
    cv: float,
    Hdr: float,
    time: float,
    dw: float = DW_DEFAULT_M,
    pattern: str = "triangular",
    smear_ratio: float = 1.0,
    kh_ks_ratio: float = 1.0,
    n_time_points: int = 50,
) -> WickDrainResult:
    """Analyze a specific wick drain layout.

    Computes consolidation at the given time and generates a
    time-settlement curve.

    Parameters
    ----------
    spacing : float
        Center-to-center drain spacing (m).
    ch : float
        Horizontal coefficient of consolidation (m²/year).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    time : float
        Analysis time (years).
    dw : float
        Equivalent drain diameter (m). Default 0.0635 (2.5 in, "the most
        common", GEC-13 p. 2-35).
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    smear_ratio : float
        Smear zone ratio ds/dw. Default 1.0: smear ignored, as GEC-13 p.
        2-35 says it often is for typical projects.
    kh_ks_ratio : float
        Permeability ratio kh/ks, with smear_ratio above 1. Default 1.0.
    n_time_points : int
        Number of points in time-settlement curve. Default 50.

    Returns
    -------
    WickDrainResult
        Complete analysis results.
    """
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if ch <= 0:
        raise ValueError(f"ch must be positive, got {ch}")
    if cv <= 0:
        raise ValueError(f"cv must be positive, got {cv}")
    if time <= 0:
        raise ValueError(f"Time must be positive, got {time}")

    de = influence_diameter(spacing, pattern)
    n_ratio = de / dw
    F_n = drain_function_F(n_ratio, smear_ratio, kh_ks_ratio)

    # Consolidation at analysis time
    Tr = radial_time_factor(ch, time, de)
    Ur = radial_degree_of_consolidation(Tr, F_n)

    Tv = time_factor(cv, time, Hdr)
    Uv = degree_of_consolidation(Tv)

    U_total = combined_degree_of_consolidation(Uv, Ur)

    # Time-settlement curve
    curve = []
    for i in range(n_time_points + 1):
        t_i = time * i / n_time_points
        if t_i == 0:
            curve.append((0.0, 0.0))
            continue
        Tv_i = time_factor(cv, t_i, Hdr)
        Uv_i = degree_of_consolidation(Tv_i)
        Tr_i = radial_time_factor(ch, t_i, de)
        Ur_i = radial_degree_of_consolidation(Tr_i, F_n)
        U_i = combined_degree_of_consolidation(Uv_i, Ur_i)
        curve.append((t_i, U_i))

    return WickDrainResult(
        drain_spacing_m=spacing,
        pattern=pattern,
        equivalent_drain_diameter_m=dw,
        influence_diameter_m=de,
        spacing_ratio_n=n_ratio,
        F_n=F_n,
        Ur_percent=Ur,
        Uv_percent=Uv,
        U_total_percent=U_total,
        time_years=time,
        ch_m2_per_year=ch,
        cv_m2_per_year=cv,
        time_settlement_curve=curve,
    )
