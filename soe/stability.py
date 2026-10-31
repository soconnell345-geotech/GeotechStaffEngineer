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
# Bearing capacity factors Nc for basal heave, read from the printed charts
# ============================================================================

#: GEC-4 Fig. 51(c) (p. 106), the curve for B/L = 0 (a cut of infinite
#: length), read from the figure's vector paths (GM-290): H/B -> Nc. The
#: curve is drawn from 5.18 at H/B = 0; it is taken from 5.14 there (the
#: strip's value, GEC-4 Eq. 12's 5.14), so a cut with no width given keeps
#: Eq. 37's Nc. Until GM-290 the module read an unprinted table (5.14 to
#: 8.6) at H / min(B, sqrt(2) H).
_NC_GEC4_STRIP_HB = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                     4.5, 5.0)
_NC_GEC4_STRIP = (5.14, 5.50, 5.81, 6.07, 6.31, 6.69, 6.97, 7.18, 7.32, 7.42,
                  7.46, 7.47, 7.44)

#: The Caltrans Trenching and Shoring Manual Fig. 10-15 (p. 10-18, after
#: Bjerrum and Eide 1956), the curve for L/B = 1 (square), read from the
#: printed chart to about +/- 0.03 (GM-290): H/B -> Nc(square). Its curve
#: for L/B = infinity agrees with 0.84 x this within about 1 %, as Eq.
#: 10-3-4 has it.
_NC_CALTRANS_SQUARE_HB = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                          2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0)
_NC_CALTRANS_SQUARE = (6.23, 6.66, 7.05, 7.39, 7.68, 7.94, 8.16, 8.34, 8.48,
                       8.59, 8.70, 8.78, 8.86, 8.95, 8.99, 9.00, 9.00)


def _interpolate_Nc_strip(H_B: float) -> float:
    """Nc for a cut of infinite length, GEC-4 Fig. 51(c)'s B/L = 0 curve
    (H/B clamped to the chart, 0 to 5)."""
    from soe.earth_pressure import _lininterp
    return _lininterp(H_B, list(_NC_GEC4_STRIP_HB), list(_NC_GEC4_STRIP))


def _interpolate_Nc_bjerrum(H_B: float, B_L: float) -> float:
    """Bjerrum and Eide's Nc as the Caltrans T&S Manual gives it: Fig.
    10-15's square curve at H/B (clamped to the chart, 0 to 5) times Eq.
    10-3-4's 0.84 + 0.16 B/L (B/L clamped to 0 to 1)."""
    from soe.earth_pressure import _lininterp
    square = _lininterp(H_B, list(_NC_CALTRANS_SQUARE_HB),
                        list(_NC_CALTRANS_SQUARE))
    return square * (0.84 + 0.16 * max(0.0, min(B_L, 1.0)))


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
    """Basal heave by Terzaghi's mechanism as GEC-4 prints it (Sec. 5.8.2.2).

    A cut of infinite length B wide (GEC-4 p. 107: "the design should be
    based on the assumption that the contractor will remove the entire
    length of each lift of excavation"):

        B > 0:  FS = Nc Su / (gamma H + q - sqrt(2) Su H / B)   (GEC-4 Eq. 35)
        B = 0:  FS = Nc Su / (gamma H + q)                      (GEC-4 Eq. 37)

    Eq. 35 is Eq. 34 with B' at its largest, B/sqrt(2) (Fig. 51(a)): the
    block's weight less the shear on its side over the strip B' it bears
    on. Eq. 37 is the limit for a very wide cut ("the contribution of the
    shearing resistance along the exterior of the failure block is
    negligible"); a width left out (B = 0) takes it. Nc is GEC-4 Fig.
    51(c)'s curve for B/L = 0 at H/B (5.14 at B = 0). GEC-4's equations
    have no surcharge; the surcharge q adds to the block's pressure, as the
    Caltrans T&S Manual's Eq. 10-3-2 adds it. Eq. 36 (a stiff layer at D
    below the base, B' = D) is not offered. GEC-4 (p. 107) takes a minimum
    FS of 1.5 for support of excavation and 2.5 for permanent facilities.

    With a width, a note and ``parameters["FOS_ufc_fig_2_7b"]`` give UFC
    3-220-20 Fig. 2-7(b) (p. 75, a flexible wall), which adds the side
    strength to the resistance instead: FBH = (5.7 Su B1 + Su H) / ((gamma
    H + q) B1), B1 = 0.7 B, Su the same above and below the base and the
    surcharge over B1. UFC p. 73, note 10: Terzaghi's subtraction "can lead
    to unreasonably high factors of safety for narrow excavations".

    Until GM-290 this was cu Nc / (gamma H + q) with Nc from an unprinted
    table at H / min(B, sqrt(2) H) and no side shear.

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
        Excavation width (m). 0 = not given: Eq. 37, a very wide cut.
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
    if B < 0:
        raise ValueError("Excavation width B must not be negative")

    H_B = H / B if B > 0 else 0.0
    Nc = _interpolate_Nc_strip(H_B)
    side = math.sqrt(2.0) * cu * H / B if B > 0 else 0.0

    resistance = cu * Nc
    demand = gamma * H + q_surcharge - side
    notes = []
    if demand > 0:
        FOS = resistance / demand
        passes = FOS >= FOS_required
    else:
        FOS = float("inf")
        passes = False
        notes.append(
            f"The side shear sqrt(2) Su H / B ({side:.1f} kPa) is at least "
            "the block's pressure gamma H + q: GEC-4 Eq. 35 gives no factor "
            "of safety here, and basal heave adequacy is NOT demonstrated by "
            "it. GEC-4 p. 107: below the target factors of safety, use more "
            "rigorous procedures such as limit equilibrium methods or "
            "Henkel's method.")

    FOS_ufc = None
    if B > 0:
        B1 = 0.7 * B
        FOS_ufc = (5.7 * cu * B1 + cu * H) / ((gamma * H + q_surcharge) * B1)
        notes.append(
            "UFC 3-220-20 Fig. 2-7(b) (p. 75, a flexible wall) adds the side "
            "strength to the resistance instead of taking it off the load: "
            "FBH = (5.7 Su B1 + Su H) / ((gamma H + q) B1), B1 = 0.7B, = "
            f"{FOS_ufc:.2f} here (Su the same above and below the base, the "
            "surcharge over B1). UFC p. 73, note 10: the two give similar "
            "results below 1.5, and Terzaghi's subtraction 'can lead to "
            "unreasonably high factors of safety for narrow excavations'.")

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
        passes=passes,
        resistance=round(resistance, 2),
        demand=round(demand, 2),
        parameters={
            "H_m": H,
            "cu_kPa": cu,
            "gamma_kNm3": gamma,
            "q_surcharge_kPa": q_surcharge,
            "B_m": B,
            "H_B": round(H_B, 3),
            "Nc": round(Nc, 3),
            "side_shear_kPa": round(side, 2),
            "equation": "GEC-4 Eq. 35" if B > 0 else "GEC-4 Eq. 37",
            "FOS_ufc_fig_2_7b": (None if FOS_ufc is None
                                 else round(FOS_ufc, 3)),
            "stability_number_N": round(N_stability, 2),
        },
        notes=notes,
    )


# ============================================================================
# Basal heave — Bjerrum & Eide (1956) method
# ============================================================================

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

    Treats the excavation base as an inverted footing problem, with no side
    shear (GEC-4 Eq. 37's form):

        FOS = cu * Nc / (gamma * H + q)

    Nc from the Caltrans T&S Manual's Fig. 10-15 (after Bjerrum and Eide
    1956), its square curve at H/Be, times Eq. 10-3-4's 0.84 + 0.16 Be/Le
    (GM-290; until then an unprinted table).

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
    Be_Le = min(Be / Le, 1.0)

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


def check_basal_heave_caltrans(
    H: float,
    cu: float,
    gamma: float,
    B: float,
    L: Optional[float] = None,
    q_surcharge: float = 0.0,
    Nc: Optional[float] = None,
    width_factor: float = 0.7,
    FOS_required: float = 1.5,
) -> StabilityCheckResult:
    """Basal-heave FS by the Caltrans (T&S Manual Ex 10-2) force-balance method.

    Unlike ``check_basal_heave_bjerrum_eide`` (an inverted-footing bearing ratio
    FOS = cu*Nc/(gamma*H+q) with no side shear), this reproduces the Caltrans
    limiting-equilibrium block balance, which INCLUDES the sidewall-shear
    resistance S = cu*H on the vertical failure plane:

        resisting  F_RS = qu * (width_factor * B),  qu = cu * Nc
        driving    F_dr = W + (width_factor*B)*q - S
                        = (width_factor*B)*H*gamma + (width_factor*B)*q - cu*H
        FS = F_RS / F_dr

    The failure block width is ``width_factor * B`` (Terzaghi's 0.7B; Eqs.
    10-3-2 and 10-3-3). With a hard layer within 0.7B of the base, D below
    it, Sec. 10-3.01 steps 1-6 (p. 10-19) change the width to D: give
    ``width_factor`` = D / B. Nc is Bjerrum and Eide's (Fig. 10-15's square
    curve at H/B times Eq. 10-3-4's 0.84 + 0.16 B/L), or an explicit
    chart-read ``Nc``. Example 10-2 (p. 10-20) reads 8.5 and 7.6 at H/B = 2,
    L/B = 3; the module reads 8.48 and 7.575 (FS 1.533 against the
    printed 1.54). Until GM-290 Nc was Skempton's formula, 5.14 (1 + 0.2
    B/L)(1 + 0.2 H/B).

    Parameters
    ----------
    H : float
        Excavation depth (m).
    cu : float
        Undrained shear strength (kPa) (clay, phi=0).
    gamma : float
        Total unit weight of the retained soil (kN/m3).
    B : float
        Excavation width (m).
    L : float, optional
        Excavation length (m) for the rectangular Nc correction. Default None
        (square, B/L = 1).
    q_surcharge : float, optional
        Surface surcharge (kPa). Default 0.
    Nc : float, optional
        Bearing-capacity factor. If None, Bjerrum and Eide's from the
        Caltrans Fig. 10-15 and Eq. 10-3-4.
    width_factor : float, optional
        Failure-block width as a fraction of B. Default 0.7 (Terzaghi).
    FOS_required : float, optional
        Minimum required FS. Default 1.5.

    Returns
    -------
    StabilityCheckResult

    References
    ----------
    Caltrans Trenching & Shoring Manual, Sec. 10-3, Eqs. 10-3-2 to 10-3-5,
    Fig. 10-15, Example 10-2; Terzaghi (1943); Bjerrum & Eide (1956).
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")
    if gamma <= 0:
        raise ValueError("Unit weight gamma must be positive")
    if B <= 0:
        raise ValueError("Excavation width B must be positive")

    B_over_L = (B / L) if (L and L > 0) else 1.0
    if Nc is None:
        Nc = _interpolate_Nc_bjerrum(H / B, B_over_L)

    width = width_factor * B
    qu = cu * Nc
    F_RS = qu * width
    W = width * H * gamma
    surcharge_force = width * q_surcharge
    side_shear = cu * H
    F_dr = W + surcharge_force - side_shear

    notes = []
    if F_dr > 0:
        FOS = F_RS / F_dr
        passes = FOS >= FOS_required
    else:
        # Non-positive driving force: the sidewall shear cu*H exceeds the block
        # weight + surcharge, so the Caltrans force-balance mechanism is outside
        # its applicability range (the block cannot drive heave here). FOS is
        # mathematically +inf, but a hard PASS on an out-of-range method is the
        # defect this branch fixes -- for narrow/deep geometry basal heave is
        # governed by the inverted-footing bearing check, which can still fail.
        FOS = float("inf")
        passes = False
        notes.append(
            "Driving force non-positive: sidewall shear cu*H exceeds block "
            "weight + surcharge -- the Caltrans force-balance mechanism is "
            "outside its applicability range here; basal-heave adequacy NOT "
            "demonstrated. Verify with check_basal_heave_bjerrum_eide (governs "
            "for narrow/deep geometry).")

    return StabilityCheckResult(
        check_type="basal_heave_caltrans",
        FOS=round(FOS, 3),
        FOS_required=FOS_required,
        passes=passes,
        resistance=round(F_RS, 2),
        demand=round(F_dr, 2),
        parameters={
            "H_m": H,
            "B_m": B,
            "L_m": L,
            "cu_kPa": cu,
            "gamma_kNm3": gamma,
            "q_surcharge_kPa": q_surcharge,
            "Nc": round(Nc, 3),
            "qu_kPa": round(qu, 2),
            "block_width_m": round(width, 3),
            "W_kN_per_m": round(W, 2),
            "surcharge_force_kN_per_m": round(surcharge_force, 2),
            "side_shear_S_kN_per_m": round(side_shear, 2),
            "width_factor": width_factor,
        },
        notes=notes,
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
