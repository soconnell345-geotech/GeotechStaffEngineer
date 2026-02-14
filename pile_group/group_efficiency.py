"""
Group efficiency factors for pile groups.

Implements Converse-Labarre formula and block failure check.

All units are SI: meters, degrees.

References:
    FHWA GEC-12, Chapter 9 (FHWA-NHI-16-009)
    USACE EM 1110-2-2906, Chapter 5
"""

import math
from typing import Optional


def converse_labarre(n_rows: int, n_cols: int,
                     pile_diameter: float,
                     spacing: float) -> float:
    """Converse-Labarre group efficiency formula.

    Eg = 1 - theta/(90*m*n) * [n*(m-1) + m*(n-1)]

    where theta = arctan(b/s), b = pile diameter, s = center-to-center spacing.

    Parameters
    ----------
    n_rows : int
        Number of rows (m).
    n_cols : int
        Number of columns (n).
    pile_diameter : float
        Pile diameter or width (m).
    spacing : float
        Center-to-center pile spacing (m).

    Returns
    -------
    float
        Group efficiency Eg (0 to 1).

    References
    ----------
    FHWA GEC-12, Eq 9-1.
    """
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if pile_diameter <= 0:
        raise ValueError(f"Pile diameter must be positive, got {pile_diameter}")

    m = n_rows
    n = n_cols
    theta = math.degrees(math.atan(pile_diameter / spacing))

    Eg = 1.0 - theta / (90.0 * m * n) * (n * (m - 1) + m * (n - 1))
    return max(Eg, 0.0)


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
