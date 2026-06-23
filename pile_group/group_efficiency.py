"""
Group efficiency factors for pile groups.

Implements Converse-Labarre formula, block failure check, equivalent
raft group settlement, and the Meyerhof (1976) SPT group-settlement
method per FHWA GEC-12.

All units are SI: meters, kPa, kN, degrees.

References:
    FHWA GEC-12, Chapter 9 (FHWA-NHI-16-009)
    AASHTO LRFD Section 10.7.2.3
    USACE EM 1110-2-2906, Chapter 5
    Meyerhof, G.G. (1976), "Bearing capacity and settlement of pile
        foundations," J. Geotech. Eng. Div., ASCE, 102(GT3), 197-228.
"""

import math
from typing import Optional, List, Tuple


def converse_labarre(n_rows: int, n_cols: int,
                     pile_diameter: float,
                     spacing: float) -> float:
    """Converse-Labarre group efficiency formula.

    Eg = 1 - theta_deg / (90 * m * n) * [n*(m-1) + m*(n-1)]

    where theta = arctan(d/s) in degrees, d = pile diameter,
    s = center-to-center spacing.

    The formula is symmetric in m and n so the result is identical
    regardless of which dimension is called "rows" vs "columns".

    Per AASHTO LRFD Section 10.7.2.3, the group efficiency is capped
    at 1.0 for friction piles.

    Parameters
    ----------
    n_rows : int
        Number of rows in the group (m >= 1).
    n_cols : int
        Number of columns / piles per row (n >= 1).
    pile_diameter : float
        Pile diameter or width, d (m).
    spacing : float
        Center-to-center pile spacing, s (m).

    Returns
    -------
    float
        Group efficiency Eg, clamped to [0, 1].

    References
    ----------
    FHWA GEC-12, Eq 9-1; AASHTO LRFD 10.7.2.3.
    """
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if pile_diameter <= 0:
        raise ValueError(f"Pile diameter must be positive, got {pile_diameter}")

    m = n_rows
    n = n_cols
    theta_deg = math.degrees(math.atan(pile_diameter / spacing))

    Eg = 1.0 - theta_deg / (90.0 * m * n) * (n * (m - 1) + m * (n - 1))
    return min(max(Eg, 0.0), 1.0)


def block_failure_capacity(n_rows: int, n_cols: int,
                           spacing_x: float, spacing_y: float,
                           pile_length: float,
                           cu: float,
                           pile_diameter: float) -> float:
    """Block failure capacity for a pile group in cohesive soil.

    The block failure mode treats the entire pile group as a single
    large pier. The capacity is:
        Q_block = cu_base * Nc * B_block * L_block + 2*(B_block + L_block) * L_pile * cu_avg

    Simplified: Q_block = 9*cu*B*L + 2*(B+L)*L_pile*cu

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    spacing_x : float
        Spacing in X (m).
    spacing_y : float
        Spacing in Y (m).
    pile_length : float
        Pile embedment length (m).
    cu : float
        Average undrained shear strength (kPa).
    pile_diameter : float
        Pile diameter (m).

    Returns
    -------
    float
        Block failure capacity (kN).

    References
    ----------
    FHWA GEC-12, Section 9.4.1.
    """
    B_block = (n_cols - 1) * spacing_x + pile_diameter
    L_block = (n_rows - 1) * spacing_y + pile_diameter

    # End bearing: Nc=9 for deep block
    Q_base = 9.0 * cu * B_block * L_block
    # Side friction
    Q_side = 2.0 * (B_block + L_block) * pile_length * cu

    return Q_base + Q_side


def p_multiplier(row_position: int, spacing_diameter_ratio: float) -> float:
    """Lateral p-multiplier for piles in a group.

    Accounts for group effects on lateral resistance (shadowing).

    Parameters
    ----------
    row_position : int
        Row position: 1 = leading row, 2 = second row, 3+ = trailing rows.
    spacing_diameter_ratio : float
        Center-to-center spacing / pile diameter (s/D).

    Returns
    -------
    float
        p-multiplier (0 to 1).

    References
    ----------
    Brown et al. (1988); FHWA GEC-12, Table 9-2.
    """
    sd = spacing_diameter_ratio
    if sd >= 5:
        return 1.0  # No reduction for wide spacing

    if row_position == 1:
        # Leading row
        if sd <= 3:
            return 0.8
        return 0.8 + (sd - 3) * 0.1  # 0.8 to 1.0
    elif row_position == 2:
        # Second row
        if sd <= 3:
            return 0.4
        return 0.4 + (sd - 3) * 0.3  # 0.4 to 1.0
    else:
        # Third and subsequent rows
        if sd <= 3:
            return 0.3
        return 0.3 + (sd - 3) * 0.35  # 0.3 to 1.0


def group_settlement_equivalent_raft(
    n_rows: int,
    n_cols: int,
    spacing: float,
    pile_diameter: float,
    pile_length: float,
    load_kN: float,
    soil_modulus_kPa: float,
    num_sublayers: int = 10,
) -> dict:
    """Group settlement by the equivalent raft method (FHWA GEC-12, Section 9.8).

    The pile group is replaced by an equivalent footing placed at a depth of
    2/3 * L below the ground surface (L = pile embedment length).  The
    equivalent raft dimensions are:

        Bg = (n_cols - 1) * s + d      (width, along columns)
        Lg = (n_rows - 1) * s + d      (length, along rows)

    Stress is distributed below the equivalent raft using the 2-Vertical :
    1-Horizontal (2V:1H) approximation.  At depth z below the raft:

        delta_sigma(z) = Q / [(Bg + z) * (Lg + z)]

    Settlement is computed by summing elastic compression of sublayers:

        S = SUM( delta_sigma_i * dz_i / Es_i )

    where Es_i is the constrained (or elastic) modulus of sublayer i.
    The stress influence zone extends to a depth of 5*Bg below the raft
    (or to the point where the stress increment < 10 % of overburden,
    whichever controls — this simplified version uses 5*Bg).

    Parameters
    ----------
    n_rows : int
        Number of rows in the group (>= 1).
    n_cols : int
        Number of columns (piles per row) in the group (>= 1).
    spacing : float
        Center-to-center pile spacing, s (m).  Assumed equal in both
        directions.
    pile_diameter : float
        Pile diameter or width, d (m).
    pile_length : float
        Pile embedment length, L (m).
    load_kN : float
        Total vertical group load, Q (kN).
    soil_modulus_kPa : float
        Representative constrained / elastic modulus of the soil below
        the equivalent raft, Es (kPa).  For layered soils, provide a
        weighted average.
    num_sublayers : int
        Number of sublayers used to discretize the influence zone.
        Default 10.  More sublayers improve accuracy at negligible cost.

    Returns
    -------
    dict
        Keys:
            - ``settlement_m`` : float — total settlement (m)
            - ``settlement_mm`` : float — total settlement (mm)
            - ``Bg_m`` : float — equivalent raft width (m)
            - ``Lg_m`` : float — equivalent raft length (m)
            - ``raft_depth_m`` : float — depth of equivalent raft (m)
            - ``influence_depth_m`` : float — depth of stress influence zone (m)
            - ``max_stress_kPa`` : float — stress at top of influence zone (kPa)

    Raises
    ------
    ValueError
        For non-positive spacing, diameter, length, load, or modulus.

    References
    ----------
    FHWA GEC-12 (FHWA-NHI-16-009), Section 9.8.1.
    AASHTO LRFD 10.7.2.3.
    """
    # --- Input validation ---
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if pile_diameter <= 0:
        raise ValueError(f"Pile diameter must be positive, got {pile_diameter}")
    if pile_length <= 0:
        raise ValueError(f"Pile length must be positive, got {pile_length}")
    if load_kN <= 0:
        raise ValueError(f"Load must be positive, got {load_kN}")
    if soil_modulus_kPa <= 0:
        raise ValueError(f"Soil modulus must be positive, got {soil_modulus_kPa}")
    if num_sublayers < 1:
        raise ValueError(f"num_sublayers must be >= 1, got {num_sublayers}")

    # --- Equivalent raft dimensions ---
    Bg = (n_cols - 1) * spacing + pile_diameter
    Lg = (n_rows - 1) * spacing + pile_diameter

    # --- Equivalent raft depth (2/3 of pile length) ---
    raft_depth = (2.0 / 3.0) * pile_length

    # --- Influence zone = 5 * Bg below raft ---
    influence_depth = 5.0 * Bg

    # --- Discretize and integrate settlement ---
    dz = influence_depth / num_sublayers
    settlement = 0.0

    for i in range(num_sublayers):
        z_mid = (i + 0.5) * dz  # mid-depth of sublayer below raft

        # 2V:1H stress distribution
        delta_sigma = load_kN / ((Bg + z_mid) * (Lg + z_mid))

        settlement += delta_sigma * dz / soil_modulus_kPa

    # Stress at top of influence zone (z = 0)
    max_stress = load_kN / (Bg * Lg)

    return {
        "settlement_m": round(settlement, 6),
        "settlement_mm": round(settlement * 1000.0, 3),
        "Bg_m": round(Bg, 4),
        "Lg_m": round(Lg, 4),
        "raft_depth_m": round(raft_depth, 4),
        "influence_depth_m": round(influence_depth, 4),
        "max_stress_kPa": round(max_stress, 3),
    }


# --- Unit-conversion constants for the Meyerhof (1976) SPT method ------------
# The Meyerhof formula's leading coefficient "4" is calibrated for US customary
# units (settlement in INCHES, net pressure pf in ksf, group width B in FEET).
_KSF_PER_KPA = 1.0 / 47.88   # 1 kPa = 1/47.88 ksf  (1 ksf = 47.88 kPa)
_FT_PER_M = 1.0 / 0.3048     # 1 m = 1/0.3048 ft
_MM_PER_IN = 25.4            # 1 in = 25.4 mm


def meyerhof_group_settlement(
    group_width: float,
    group_length: float,
    N160: float,
    load_kN: Optional[float] = None,
    pf_kPa: Optional[float] = None,
    embedment_DB: float = 0.0,
) -> dict:
    """Meyerhof (1976) SPT pile-group settlement (FHWA GEC-12 Vol 3 App D).

    Empirical group settlement of a pile group founded in/over a granular
    (SPT-characterized) bearing stratum.  In the original US-customary
    calibrated form (settlement in INCHES):

        S[in] = 4 * pf[ksf] * If * sqrt(B[ft]) / N160

    where

        pf = net foundation pressure = Q / (group plan area)   [the group's
             unfactored permanent load over its plan footprint]
        If = embedment/influence factor
           = 1 - (2/3 * DB) / (8 * B),  floored at 0.5
        DB = embedment depth of the group into the bearing stratum
        B  = group width (the SMALLER plan dimension)
        N160 = average corrected SPT N within ~B below the pile-group toe.

    Unit handling (SI public API)
    -----------------------------
    This module is SI, but the leading coefficient "4" is calibrated for US
    customary units (S in inches, pf in ksf, B in feet).  This function
    therefore takes **SI inputs** (widths in m, load in kN / pf in kPa) and
    handles the conversion explicitly: it converts the SI inputs to US units,
    evaluates the published US-customary formula, then converts the resulting
    settlement (inches) back to **mm** for the return value.  Concretely:

        pf[ksf] = pf[kPa] / 47.88
        B[ft]   = B[m] / 0.3048      (likewise DB)
        S[mm]   = 25.4 * S[in]

    Substituting gives the equivalent SI-consistent closed form

        S[mm] = C * pf[kPa] * If * sqrt(B[m]) / N160,
        C     = 25.4 * 4 / 47.88 / sqrt(0.3048) = 3.8435...,

    which reproduces the published inches->mm value (see the validation note
    below); the explicit-conversion path above is used internally so the
    US-form formula is applied verbatim.

    The net pressure can be supplied either as a total permanent load
    ``load_kN`` (Q, kN) — divided here by the group plan area
    ``group_width * group_length`` — or directly as ``pf_kPa``.

    Parameters
    ----------
    group_width : float
        Group plan width, B (m).  By definition B is the **smaller** plan
        dimension; if ``group_length`` is given smaller, the two are
        swapped so B is always the smaller of the two.
    group_length : float
        Group plan length, Z (m), the other (larger) plan dimension.  Used
        with ``group_width`` to form the plan area when ``load_kN`` is given.
    N160 : float
        Average corrected SPT blow count (N1)60 within ~B below the
        pile-group toe (dimensionless).
    load_kN : float, optional
        Total unfactored permanent (working) load on the group, Q (kN).
        Required if ``pf_kPa`` is not given; the net pressure is then
        ``Q / (B * Z)`` (kPa).
    pf_kPa : float, optional
        Net foundation pressure pf (kPa), supplied directly.  Takes
        precedence over ``load_kN`` if both are given.
    embedment_DB : float, optional
        Embedment depth of the group into the bearing stratum, DB (m).
        Default 0.0 (no embedment -> If = 1.0).

    Returns
    -------
    dict
        Keys:
            - ``settlement_mm`` : float — group settlement (mm)
            - ``settlement_in`` : float — group settlement (inches)
            - ``pf_kPa`` : float — net foundation pressure used (kPa)
            - ``pf_ksf`` : float — net foundation pressure (ksf)
            - ``influence_factor`` : float — embedment factor If (-)
            - ``B_m`` : float — group width B used (smaller dimension, m)
            - ``N160`` : float — SPT N used

    Raises
    ------
    ValueError
        For non-positive dimensions or N160, or if neither ``load_kN`` nor
        ``pf_kPa`` is provided (or the supplied value is non-positive).

    References
    ----------
    Meyerhof, G.G. (1976), J. Geotech. Eng. Div., ASCE, 102(GT3), 197-228.
    FHWA GEC-12 Vol 3 (FHWA-NHI-16-064), Appendix D, Table D-23.
    """
    # --- Input validation ---
    if group_width <= 0:
        raise ValueError(f"group_width must be positive, got {group_width}")
    if group_length <= 0:
        raise ValueError(f"group_length must be positive, got {group_length}")
    if N160 <= 0:
        raise ValueError(f"N160 must be positive, got {N160}")
    if embedment_DB < 0:
        raise ValueError(f"embedment_DB must be >= 0, got {embedment_DB}")

    # B is the smaller plan dimension by definition; Z the larger.
    B_m = min(group_width, group_length)
    Z_m = max(group_width, group_length)

    # --- Net foundation pressure pf (kPa) ---
    if pf_kPa is not None:
        if pf_kPa <= 0:
            raise ValueError(f"pf_kPa must be positive, got {pf_kPa}")
        pf = float(pf_kPa)
    elif load_kN is not None:
        if load_kN <= 0:
            raise ValueError(f"load_kN must be positive, got {load_kN}")
        pf = load_kN / (B_m * Z_m)
    else:
        raise ValueError(
            "Provide either load_kN (total permanent load, kN) or "
            "pf_kPa (net foundation pressure, kPa)."
        )

    # --- Embedment / influence factor (floored at 0.5) ---
    If = 1.0 - (2.0 / 3.0 * embedment_DB) / (8.0 * B_m)
    If = max(If, 0.5)

    # --- SI -> US, apply the published US-customary formula, US -> SI ---
    pf_ksf = pf * _KSF_PER_KPA
    B_ft = B_m * _FT_PER_M
    S_in = 4.0 * pf_ksf * If * math.sqrt(B_ft) / N160
    S_mm = S_in * _MM_PER_IN

    return {
        "settlement_mm": round(S_mm, 3),
        "settlement_in": round(S_in, 4),
        "pf_kPa": round(pf, 3),
        "pf_ksf": round(pf_ksf, 4),
        "influence_factor": round(If, 4),
        "B_m": round(B_m, 4),
        "N160": N160,
    }
