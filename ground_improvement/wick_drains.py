"""
Prefabricated vertical drain (wick drain / PVD) analysis.

Implements Barron (1948) / Hansbo (1981) radial consolidation theory
combined with Terzaghi vertical consolidation from the settlement module.

Key equations:
- Influence diameter:  de = 1.05*s (triangular) or 1.13*s (square)
- Drain function:      F(n) = ln(n/s) + (kh/ks)*ln(s) - 0.75
- Radial consolidation: Ur = 1 - exp(-8*Tr/F(n))
- Combined:            U_total = 1 - (1 - Uv)*(1 - Ur)

All units SI: meters, years, m²/year.

References:
    Barron, R.A. (1948) — Consolidation of fine-grained soils by drain wells
    Hansbo, S. (1981) — Consolidation of fine-grained soils by PVDs
    FHWA NHI-06-019: Ground Improvement Methods, Vol. I
"""

import math
from typing import List, Optional, Tuple

from settlement.time_rate import time_factor, degree_of_consolidation
from ground_improvement.results import WickDrainResult


def equivalent_drain_diameter(width: float, thickness: float) -> float:
    """Compute equivalent circular diameter for a band-shaped PVD.

    dw = (a + b) / pi

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
    return (width + thickness) / math.pi


def influence_diameter(spacing: float, pattern: str = "triangular") -> float:
    """Compute influence zone diameter de.

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
    """Compute Barron/Hansbo drain function F(n).

    Without smear (smear_ratio=1, kh_ks_ratio=1):
        F(n) = ln(n) - 0.75

    With smear zone:
        F(n) = ln(n/s) + (kh/ks)*ln(s) - 0.75

    where s = smear_ratio = ds/dw, kh/ks = permeability ratio.

    Parameters
    ----------
    n_ratio : float
        Spacing ratio n = de/dw.
    smear_ratio : float
        Smear zone ratio ds/dw (typically 2-3). Default 1.0 (no smear).
    kh_ks_ratio : float
        Horizontal permeability ratio kh/ks (typically 2-5). Default 1.0.

    Returns
    -------
    float
        Drain function F(n).
    """
    if n_ratio <= 1.0:
        raise ValueError(f"Spacing ratio n must be > 1, got {n_ratio}")
    if smear_ratio < 1.0:
        raise ValueError(f"Smear ratio must be >= 1, got {smear_ratio}")

    if smear_ratio <= 1.0:
        return math.log(n_ratio) - 0.75
    else:
        return (math.log(n_ratio / smear_ratio)
                + kh_ks_ratio * math.log(smear_ratio) - 0.75)


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
    dw: float = 0.066,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
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
        Equivalent drain diameter (m). Default 0.066.
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    smear_ratio : float
        Smear zone ratio ds/dw. Default 2.0.
    kh_ks_ratio : float
        Permeability ratio kh/ks. Default 2.0.
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
    dw: float = 0.066,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
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
        Equivalent drain diameter (m). Default 0.066.
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    smear_ratio : float
        Smear zone ratio ds/dw. Default 2.0.
    kh_ks_ratio : float
        Permeability ratio kh/ks. Default 2.0.
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
