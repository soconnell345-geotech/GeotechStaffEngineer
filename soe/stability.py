"""
Stability checks for braced excavations.

Implements three critical stability failure modes:
1. Basal heave — bearing capacity failure at excavation base (clay)
2. Bottom blowout — upward seepage pressure lifting the soil plug
3. Piping — internal erosion from excessive hydraulic gradient

All units SI: m, kPa, kN/m³.

References:
    Terzaghi, K. (1943) Theoretical Soil Mechanics, Ch. 9
    Bjerrum, L. & Eide, O. (1956) Stability of Strutted Excavations in Clay
    FHWA-IF-99-015, GEC-4, Sections 5.7-5.8
    USACE EM 1110-2-2504, Chapter 6
    Terzaghi, Peck, & Mesri (1996) Soil Mechanics in Engineering Practice
"""

import math
from typing import Optional

from soe.results import StabilityCheckResult


# ============================================================================
# Bearing capacity factor Nc for basal heave (Terzaghi)
# ============================================================================

# Nc values for strip (infinite length) excavation as function of H/Be
# From Terzaghi (1943) and FHWA GEC-4 Table 5-1
# H/Be:  0     1     2     3     4     >=5
# Nc:   5.14  5.70  6.30  7.00  7.70  8.60 (strip)
_NC_STRIP = {
    0.0: 5.14,
    1.0: 5.70,
    2.0: 6.30,
    3.0: 7.00,
    4.0: 7.70,
    5.0: 8.60,
}

# Correction for finite excavation length (Be/Le ratio)
# Nc_rect = Nc_strip * (1 + 0.2 * Be/Le) per Skempton (1951)
# For square (Be/Le = 1.0): multiply by 1.2


def _interpolate_Nc_strip(H_Be: float) -> float:
    """Interpolate Nc for strip excavation from H/Be ratio."""
    keys = sorted(_NC_STRIP.keys())
    if H_Be <= keys[0]:
        return _NC_STRIP[keys[0]]
    if H_Be >= keys[-1]:
        return _NC_STRIP[keys[-1]]

    for i in range(len(keys) - 1):
        if keys[i] <= H_Be <= keys[i + 1]:
            x0, x1 = keys[i], keys[i + 1]
            y0, y1 = _NC_STRIP[x0], _NC_STRIP[x1]
            return y0 + (y1 - y0) * (H_Be - x0) / (x1 - x0)

    return _NC_STRIP[keys[-1]]


# ============================================================================
# Basal heave — Terzaghi method
# ============================================================================

def check_basal_heave_terzaghi(
    H: float,
    cu: float,
    gamma: float,
    q_surcharge: float = 0.0,
    B: float = 0.0,
    FOS_required: float = 1.5,
) -> StabilityCheckResult:
    """Check for basal heave failure using Terzaghi (1943) method.

    Basal heave occurs when the weight of soil behind the wall plus
    surcharge exceeds the bearing capacity of the soil at the
    excavation base. This is primarily a concern in soft clay.

    FOS = cu * Nc / (gamma * H + q)

    Parameters
    ----------
    H : float
        Excavation depth (m).
    cu : float
        Undrained shear strength at excavation base (kPa).
    gamma : float
        Average unit weight of soil above base (kN/m³).
    q_surcharge : float
        Surface surcharge (kPa). Default 0.
    B : float
        Excavation width (m). 0 = strip (infinite length).
        Used to compute H/Be for Nc.
    FOS_required : float
        Minimum required factor of safety. Default 1.5.

    Returns
    -------
    StabilityCheckResult
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")
    if gamma <= 0:
        raise ValueError("Unit weight gamma must be positive")

    # Effective width for Nc lookup
    # Be = min(B, sqrt(2)*H) per Terzaghi; if B=0 use strip
    if B > 0:
        Be = min(B, math.sqrt(2) * H)
        H_Be = H / Be
    else:
        Be = float("inf")
        H_Be = 0.0  # strip → lowest Nc

    Nc = _interpolate_Nc_strip(H_Be)

    # For finite length, apply Skempton correction: Nc *= (1 + 0.2*Be/Le)
    # For strip excavation (B=0 or Le>>Be), no correction needed
    # We assume strip (Le >> Be) unless B is given as a square
    # (Le = B); user should pass Le explicitly if needed

    resistance = cu * Nc
    demand = gamma * H + q_surcharge
    FOS = resistance / demand if demand > 0 else float("inf")

    notes = []
    N_stability = gamma * H / cu if cu > 0 else float("inf")
    if N_stability > 6:
        notes.append(f"Stability number N = {N_stability:.1f} > 6: "
                     "high risk of basal heave")
    elif N_stability > 4:
        notes.append(f"Stability number N = {N_stability:.1f} > 4: "
                     "moderate risk of basal heave")

    return StabilityCheckResult(
        check_type="basal_heave_terzaghi",
        FOS=round(FOS, 3),
        FOS_required=FOS_required,
        passes=FOS >= FOS_required,
        resistance=round(resistance, 2),
        demand=round(demand, 2),
        parameters={
            "H_m": H,
            "cu_kPa": cu,
            "gamma_kNm3": gamma,
            "q_surcharge_kPa": q_surcharge,
            "B_m": B,
            "Be_m": round(Be, 3) if Be != float("inf") else "inf",
            "H_Be": round(H_Be, 3),
            "Nc": round(Nc, 3),
            "stability_number_N": round(N_stability, 2),
        },
        notes=notes,
    )


# ============================================================================
# Basal heave — Bjerrum & Eide (1956) method
# ============================================================================

# Nc values for Bjerrum-Eide, which accounts for both H/Be and Be/Le
# From Bjerrum & Eide (1956), also in FHWA GEC-4 Figure 5-6
# This method treats the excavation base as an inverted footing
_NC_BJERRUM_EIDE = {
    # (H_Be, Be_Le): Nc
    # Be/Le = 0 (strip)
    (0.0, 0.0): 5.14, (0.5, 0.0): 5.14, (1.0, 0.0): 5.53,
    (2.0, 0.0): 6.14, (3.0, 0.0): 6.78, (4.0, 0.0): 7.44,
    (5.0, 0.0): 7.94,
    # Be/Le = 0.5
    (0.0, 0.5): 5.14, (0.5, 0.5): 5.73, (1.0, 0.5): 6.33,
    (2.0, 0.5): 7.00, (3.0, 0.5): 7.54, (4.0, 0.5): 8.06,
    (5.0, 0.5): 8.50,
    # Be/Le = 1.0 (square)
    (0.0, 1.0): 5.14, (0.5, 1.0): 6.05, (1.0, 1.0): 6.80,
    (2.0, 1.0): 7.65, (3.0, 1.0): 8.30, (4.0, 1.0): 8.70,
    (5.0, 1.0): 9.00,
}

# Available Be/Le ratios in the table
_BE_LE_VALUES = [0.0, 0.5, 1.0]
# Available H/Be ratios in the table
_H_BE_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


def _interpolate_Nc_bjerrum(H_Be: float, Be_Le: float) -> float:
    """Bilinear interpolation of Nc from Bjerrum-Eide table."""
    # Clamp inputs to table range
    H_Be = max(0.0, min(H_Be, 5.0))
    Be_Le = max(0.0, min(Be_Le, 1.0))

    # Find bounding H/Be values
    h_lo = _H_BE_VALUES[0]
    h_hi = _H_BE_VALUES[-1]
    for i in range(len(_H_BE_VALUES) - 1):
        if _H_BE_VALUES[i] <= H_Be <= _H_BE_VALUES[i + 1]:
            h_lo = _H_BE_VALUES[i]
            h_hi = _H_BE_VALUES[i + 1]
            break

    # Find bounding Be/Le values
    b_lo = _BE_LE_VALUES[0]
    b_hi = _BE_LE_VALUES[-1]
    for i in range(len(_BE_LE_VALUES) - 1):
        if _BE_LE_VALUES[i] <= Be_Le <= _BE_LE_VALUES[i + 1]:
            b_lo = _BE_LE_VALUES[i]
            b_hi = _BE_LE_VALUES[i + 1]
            break

    # Get corner values
    Q11 = _NC_BJERRUM_EIDE[(h_lo, b_lo)]
    Q21 = _NC_BJERRUM_EIDE[(h_hi, b_lo)]
    Q12 = _NC_BJERRUM_EIDE[(h_lo, b_hi)]
    Q22 = _NC_BJERRUM_EIDE[(h_hi, b_hi)]

    # Bilinear interpolation
    dh = h_hi - h_lo
    db = b_hi - b_lo

    if dh == 0 and db == 0:
        return Q11

    if dh == 0:
        t = (Be_Le - b_lo) / db
        return Q11 + (Q12 - Q11) * t

    if db == 0:
        t = (H_Be - h_lo) / dh
        return Q11 + (Q21 - Q11) * t

    th = (H_Be - h_lo) / dh
    tb = (Be_Le - b_lo) / db

    Nc = (Q11 * (1 - th) * (1 - tb)
          + Q21 * th * (1 - tb)
          + Q12 * (1 - th) * tb
          + Q22 * th * tb)
    return Nc


def check_basal_heave_bjerrum_eide(
    H: float,
    cu: float,
    gamma: float,
    Be: float,
    Le: float,
    q_surcharge: float = 0.0,
    FOS_required: float = 1.5,
) -> StabilityCheckResult:
    """Check for basal heave using Bjerrum & Eide (1956) method.

    Treats the excavation base as an inverted footing problem.
    Nc depends on both H/Be (depth ratio) and Be/Le (shape ratio).

    FOS = cu * Nc / (gamma * H + q)

    Parameters
    ----------
    H : float
        Excavation depth (m).
    cu : float
        Undrained shear strength at excavation base (kPa).
    gamma : float
        Average unit weight of soil above base (kN/m³).
    Be : float
        Excavation width (m). Must be > 0.
    Le : float
        Excavation length (m). Must be >= Be.
    q_surcharge : float
        Surface surcharge (kPa). Default 0.
    FOS_required : float
        Minimum required factor of safety. Default 1.5.

    Returns
    -------
    StabilityCheckResult
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")
    if gamma <= 0:
        raise ValueError("Unit weight gamma must be positive")
    if Be <= 0:
        raise ValueError("Excavation width Be must be positive")
    if Le <= 0:
        raise ValueError("Excavation length Le must be positive")

    H_Be = H / Be
    Be_Le = Be / Le if Le > 0 else 0.0

    Nc = _interpolate_Nc_bjerrum(H_Be, Be_Le)

    resistance = cu * Nc
    demand = gamma * H + q_surcharge
    FOS = resistance / demand if demand > 0 else float("inf")

    return StabilityCheckResult(
        check_type="basal_heave_bjerrum_eide",
        FOS=round(FOS, 3),
        FOS_required=FOS_required,
        passes=FOS >= FOS_required,
        resistance=round(resistance, 2),
        demand=round(demand, 2),
        parameters={
            "H_m": H,
            "cu_kPa": cu,
            "gamma_kNm3": gamma,
            "Be_m": Be,
            "Le_m": Le,
            "q_surcharge_kPa": q_surcharge,
            "H_Be": round(H_Be, 3),
            "Be_Le": round(Be_Le, 3),
            "Nc": round(Nc, 3),
        },
    )


# ============================================================================
# Bottom blowout (uplift)
# ============================================================================

def check_bottom_blowout(
    D_embed: float,
    hw_excess: float,
    gamma_soil: float,
    gamma_w: float = 9.81,
    FOS_required: float = 1.5,
) -> StabilityCheckResult:
    """Check for bottom blowout (hydraulic uplift) below excavation.

    Bottom blowout occurs when the upward water pressure at the toe
    of the embedded wall exceeds the downward weight of the soil plug.
    This is a concern when the wall penetrates into a confined aquifer
    or when there is significant head difference across the wall.

    FOS = gamma_soil * D / (gamma_w * hw)

    Parameters
    ----------
    D_embed : float
        Embedment depth below excavation base (m).
    hw_excess : float
        Excess piezometric head above excavation base (m).
        This is the height of the piezometric surface above
        the excavation level.
    gamma_soil : float
        Submerged (buoyant) unit weight of the soil plug (kN/m³).
    gamma_w : float
        Unit weight of water (kN/m³). Default 9.81.
    FOS_required : float
        Minimum required factor of safety. Default 1.5.

    Returns
    -------
    StabilityCheckResult
    """
    if D_embed <= 0:
        raise ValueError("Embedment depth D must be positive")
    if hw_excess < 0:
        raise ValueError("Excess head hw must be >= 0")
    if gamma_soil <= 0:
        raise ValueError("Soil unit weight must be positive")

    resistance = gamma_soil * D_embed
    demand = gamma_w * hw_excess

    if demand <= 0:
        FOS = float("inf")
    else:
        FOS = resistance / demand

    notes = []
    if FOS < 1.0:
        notes.append("FOS < 1.0: bottom blowout is expected — "
                      "increase embedment or dewater")

    return StabilityCheckResult(
        check_type="bottom_blowout",
        FOS=round(FOS, 3),
        FOS_required=FOS_required,
        passes=FOS >= FOS_required,
        resistance=round(resistance, 2),
        demand=round(demand, 2),
        parameters={
            "D_embed_m": D_embed,
            "hw_excess_m": hw_excess,
            "gamma_soil_kNm3": gamma_soil,
            "gamma_w_kNm3": gamma_w,
        },
        notes=notes,
    )


# ============================================================================
# Piping (internal erosion)
# ============================================================================

def check_piping(
    delta_h: float,
    flow_path: float,
    Gs: float = 2.65,
    void_ratio: float = 0.65,
    FOS_required: float = 2.0,
) -> StabilityCheckResult:
    """Check for piping (internal erosion) due to upward seepage.

    Piping occurs when the exit hydraulic gradient exceeds the
    critical gradient. The critical gradient is the gradient at
    which the upward seepage force equals the submerged weight
    of the soil.

    i_critical = (Gs - 1) / (1 + e)
    i_exit = delta_h / flow_path
    FOS = i_critical / i_exit

    Parameters
    ----------
    delta_h : float
        Total head difference driving seepage (m).
    flow_path : float
        Length of the shortest seepage path (m).
        Typically = 2 * D for flow around the wall toe,
        where D is the embedment depth.
    Gs : float
        Specific gravity of soil solids. Default 2.65.
    void_ratio : float
        Void ratio of the soil. Default 0.65.
    FOS_required : float
        Minimum required factor of safety. Default 2.0
        (higher than other checks per USACE guidance).

    Returns
    -------
    StabilityCheckResult
    """
    if delta_h < 0:
        raise ValueError("Head difference delta_h must be >= 0")
    if flow_path <= 0:
        raise ValueError("Flow path length must be positive")
    if Gs <= 1.0:
        raise ValueError("Specific gravity Gs must be > 1.0")
    if void_ratio <= 0:
        raise ValueError("Void ratio must be positive")

    i_critical = (Gs - 1.0) / (1.0 + void_ratio)
    i_exit = delta_h / flow_path

    if i_exit <= 0:
        FOS = float("inf")
    else:
        FOS = i_critical / i_exit

    notes = []
    if i_exit > 0.5 * i_critical:
        notes.append(f"Exit gradient i = {i_exit:.3f} exceeds 50% of "
                     f"critical gradient ({i_critical:.3f})")

    return StabilityCheckResult(
        check_type="piping",
        FOS=round(FOS, 3),
        FOS_required=FOS_required,
        passes=FOS >= FOS_required,
        resistance=round(i_critical, 4),
        demand=round(i_exit, 4),
        parameters={
            "delta_h_m": delta_h,
            "flow_path_m": flow_path,
            "Gs": Gs,
            "void_ratio": void_ratio,
            "i_critical": round(i_critical, 4),
            "i_exit": round(i_exit, 4),
        },
        notes=notes,
    )
