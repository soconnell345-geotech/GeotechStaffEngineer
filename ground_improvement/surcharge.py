"""
Surcharge preloading analysis.

Evaluates surcharge preloading for accelerating consolidation settlement,
both with and without prefabricated vertical drains (PVDs).

All units SI: kPa, meters, years.

References:
    Terzaghi (1925) — 1-D consolidation theory
    FHWA NHI-06-019: Ground Improvement Methods, Vol. I
    FHWA GEC-13: Ground Modification Methods Reference Manual
"""

import math
from typing import List, Optional, Tuple

from settlement.time_rate import (
    time_factor, degree_of_consolidation, settlement_at_time,
    time_for_consolidation,
)
from ground_improvement.wick_drains import (
    influence_diameter, drain_function_F,
    radial_time_factor, radial_degree_of_consolidation,
    combined_degree_of_consolidation,
)
from ground_improvement.results import SurchargeResult, WickDrainResult


def surcharge_settlement_at_time(
    S_ultimate: float,
    cv: float,
    Hdr: float,
    t: float,
) -> float:
    """Settlement at time t under surcharge (no drains).

    Parameters
    ----------
    S_ultimate : float
        Ultimate consolidation settlement under surcharge (m).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    t : float
        Time since surcharge application (years).

    Returns
    -------
    float
        Settlement at time t (m).
    """
    return settlement_at_time(S_ultimate, cv, Hdr, t)


def surcharge_with_drains_settlement_at_time(
    S_ultimate: float,
    cv: float,
    ch: float,
    Hdr: float,
    drain_spacing: float,
    dw: float,
    t: float,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
) -> float:
    """Settlement at time t under surcharge with wick drains.

    Parameters
    ----------
    S_ultimate : float
        Ultimate consolidation settlement (m).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    ch : float
        Horizontal coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    drain_spacing : float
        Center-to-center drain spacing (m).
    dw : float
        Equivalent drain diameter (m).
    t : float
        Time since surcharge application (years).
    pattern : str
        'triangular' or 'square'.
    smear_ratio : float
        Smear zone ratio ds/dw.
    kh_ks_ratio : float
        Permeability ratio kh/ks.

    Returns
    -------
    float
        Settlement at time t (m).
    """
    if t <= 0:
        return 0.0

    # Vertical component
    Tv = time_factor(cv, t, Hdr)
    Uv = degree_of_consolidation(Tv)

    # Radial component
    de = influence_diameter(drain_spacing, pattern)
    n_ratio = de / dw
    F_n = drain_function_F(n_ratio, smear_ratio, kh_ks_ratio)
    Tr = radial_time_factor(ch, t, de)
    Ur = radial_degree_of_consolidation(Tr, F_n)

    U_total = combined_degree_of_consolidation(Uv, Ur) / 100.0
    return U_total * S_ultimate


def required_surcharge_for_preconsolidation(
    sigma_v0: float,
    sigma_p_target: float,
    influence_factor: float = 1.0,
) -> float:
    """Compute surcharge needed to achieve target preconsolidation pressure.

    q_surcharge = (sigma_p_target - sigma_v0) / influence_factor

    Parameters
    ----------
    sigma_v0 : float
        Current effective vertical stress at layer center (kPa).
    sigma_p_target : float
        Desired preconsolidation pressure (kPa).
    influence_factor : float
        Stress influence factor at depth of interest. Default 1.0.

    Returns
    -------
    float
        Required surcharge pressure (kPa).
    """
    if sigma_p_target <= sigma_v0:
        return 0.0
    if influence_factor <= 0:
        raise ValueError(f"Influence factor must be positive, got {influence_factor}")
    return (sigma_p_target - sigma_v0) / influence_factor


def _combined_U_at_time(
    t: float,
    cv: float,
    Hdr: float,
    ch: float = None,
    drain_spacing: float = None,
    dw: float = 0.066,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
) -> float:
    """Compute combined degree of consolidation at time t.

    If ch and drain_spacing are provided, uses combined vertical + radial.
    Otherwise, uses vertical only.

    Returns percent (0-100).
    """
    if t <= 0:
        return 0.0

    Tv = time_factor(cv, t, Hdr)
    Uv = degree_of_consolidation(Tv)

    if ch is not None and drain_spacing is not None:
        de = influence_diameter(drain_spacing, pattern)
        n_ratio = de / dw
        F_n = drain_function_F(n_ratio, smear_ratio, kh_ks_ratio)
        Tr = radial_time_factor(ch, t, de)
        Ur = radial_degree_of_consolidation(Tr, F_n)
        return combined_degree_of_consolidation(Uv, Ur)
    else:
        return Uv


def time_to_target_consolidation(
    target_U: float,
    cv: float,
    Hdr: float,
    ch: float = None,
    drain_spacing: float = None,
    dw: float = 0.066,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
) -> float:
    """Compute time to reach target degree of consolidation.

    Without drains (ch=None or drain_spacing=None): uses Terzaghi
    vertical theory via settlement.time_rate.

    With drains: uses bisection on combined U formula.

    Parameters
    ----------
    target_U : float
        Target degree of consolidation (percent, 0-100).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    ch : float, optional
        Horizontal coefficient of consolidation (m²/year).
    drain_spacing : float, optional
        Center-to-center drain spacing (m).
    dw : float
        Equivalent drain diameter (m). Default 0.066.
    pattern : str
        'triangular' or 'square'.
    smear_ratio : float
        Smear zone ratio ds/dw.
    kh_ks_ratio : float
        Permeability ratio kh/ks.

    Returns
    -------
    float
        Time to reach target U (years).
    """
    if target_U <= 0:
        return 0.0
    if target_U >= 100:
        raise ValueError("Cannot compute time for 100% consolidation")

    # No drains: use analytical solution
    if ch is None or drain_spacing is None:
        return time_for_consolidation(target_U, cv, Hdr)

    # With drains: bisection on combined U
    t_lo = 0.001  # small positive start
    t_hi = 100.0  # years

    # Expand upper bound if needed
    while _combined_U_at_time(t_hi, cv, Hdr, ch, drain_spacing, dw,
                              pattern, smear_ratio, kh_ks_ratio) < target_U:
        t_hi *= 2.0
        if t_hi > 10000:
            break

    for _ in range(100):
        t_mid = (t_lo + t_hi) / 2.0
        U_mid = _combined_U_at_time(t_mid, cv, Hdr, ch, drain_spacing, dw,
                                    pattern, smear_ratio, kh_ks_ratio)
        if U_mid < target_U:
            t_lo = t_mid
        else:
            t_hi = t_mid
        if t_hi - t_lo < 0.0001:
            break

    return t_hi  # conservative (slightly longer)


def analyze_surcharge_preloading(
    S_ultimate: float,
    surcharge_kPa: float,
    cv: float,
    Hdr: float,
    target_U: float = 90.0,
    ch: float = None,
    drain_spacing: float = None,
    dw: float = 0.066,
    pattern: str = "triangular",
    smear_ratio: float = 2.0,
    kh_ks_ratio: float = 2.0,
    sigma_v0: float = 0.0,
    n_time_points: int = 50,
) -> SurchargeResult:
    """Full surcharge preloading analysis.

    Parameters
    ----------
    S_ultimate : float
        Ultimate consolidation settlement under surcharge (m).
    surcharge_kPa : float
        Applied surcharge pressure (kPa).
    cv : float
        Vertical coefficient of consolidation (m²/year).
    Hdr : float
        Vertical drainage path (m).
    target_U : float
        Target degree of consolidation (percent). Default 90%.
    ch : float, optional
        Horizontal coefficient of consolidation (m²/year).
    drain_spacing : float, optional
        Center-to-center drain spacing (m).
    dw : float
        Equivalent drain diameter (m). Default 0.066.
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    smear_ratio : float
        Smear zone ratio ds/dw. Default 2.0.
    kh_ks_ratio : float
        Permeability ratio kh/ks. Default 2.0.
    sigma_v0 : float
        Current effective stress at layer center (kPa). Default 0.
    n_time_points : int
        Points in time-settlement curve. Default 50.

    Returns
    -------
    SurchargeResult
        Complete surcharge analysis results.
    """
    if S_ultimate <= 0:
        raise ValueError(f"S_ultimate must be positive, got {S_ultimate}")
    if cv <= 0:
        raise ValueError(f"cv must be positive, got {cv}")

    uses_drains = ch is not None and drain_spacing is not None

    # Time to target consolidation
    t_target = time_to_target_consolidation(
        target_U, cv, Hdr, ch, drain_spacing, dw,
        pattern, smear_ratio, kh_ks_ratio,
    )

    # Settlement at target
    settlement_at_target = target_U / 100.0 * S_ultimate

    # Equivalent preconsolidation pressure
    sigma_p = 0.0
    if sigma_v0 > 0 and surcharge_kPa > 0:
        sigma_p = sigma_v0 + surcharge_kPa

    # Build wick drain sub-result if drains used
    wick_result = None
    if uses_drains:
        from ground_improvement.wick_drains import analyze_wick_drains
        wick_result = analyze_wick_drains(
            spacing=drain_spacing,
            ch=ch,
            cv=cv,
            Hdr=Hdr,
            time=t_target,
            dw=dw,
            pattern=pattern,
            smear_ratio=smear_ratio,
            kh_ks_ratio=kh_ks_ratio,
        )

    # Time-settlement curve
    # Extend curve to 1.5x target time to show behavior after target
    t_max = t_target * 1.5 if t_target > 0 else 1.0
    curve = []
    for i in range(n_time_points + 1):
        t_i = t_max * i / n_time_points
        if t_i == 0:
            curve.append((0.0, 0.0))
            continue

        U_i = _combined_U_at_time(t_i, cv, Hdr, ch, drain_spacing, dw,
                                  pattern, smear_ratio, kh_ks_ratio)
        S_i = U_i / 100.0 * S_ultimate * 1000.0  # convert m to mm
        curve.append((t_i, S_i))

    return SurchargeResult(
        surcharge_kPa=surcharge_kPa,
        target_U_percent=target_U,
        time_to_target_years=t_target,
        settlement_at_target_mm=settlement_at_target * 1000.0,
        settlement_ultimate_mm=S_ultimate * 1000.0,
        uses_wick_drains=uses_drains,
        wick_drain_result=wick_result,
        equivalent_sigma_p_kPa=sigma_p,
        time_settlement_curve=curve,
    )
