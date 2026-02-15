"""
Aggregate pier / rammed aggregate pier analysis.

Computes area replacement ratio, composite modulus, settlement reduction
factor, and bearing capacity improvement for stone column / aggregate
pier ground improvement.

All units SI: kPa, meters.

References:
    Barksdale & Bachus (1983) — Design and Construction of Stone Columns
    FHWA GEC-13: Ground Modification Methods Reference Manual
    Priebe (1995) — Design of vibro replacement
"""

import math
import warnings

from ground_improvement.results import AggregatePierResult


def area_replacement_ratio(column_diameter: float, spacing: float,
                           pattern: str = "triangular") -> float:
    """Compute area replacement ratio as = Ac / A_tributary.

    Parameters
    ----------
    column_diameter : float
        Pier column diameter (m).
    spacing : float
        Center-to-center spacing (m).
    pattern : str
        'triangular' or 'square'.

    Returns
    -------
    float
        Area replacement ratio (dimensionless, 0 to 1).
    """
    if column_diameter <= 0:
        raise ValueError(f"Column diameter must be positive, got {column_diameter}")
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if spacing <= column_diameter:
        raise ValueError(
            f"Spacing ({spacing} m) must exceed column diameter ({column_diameter} m)"
        )

    Ac = math.pi / 4.0 * column_diameter**2

    if pattern == "triangular":
        A_trib = math.sqrt(3) / 2.0 * spacing**2
    elif pattern == "square":
        A_trib = spacing**2
    else:
        raise ValueError(f"Pattern must be 'triangular' or 'square', got '{pattern}'")

    return Ac / A_trib


def settlement_reduction_factor(as_ratio: float, n: float) -> float:
    """Compute settlement reduction factor.

    SRF = 1 / (1 + as * (n - 1))

    Parameters
    ----------
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio (typically 3-8).

    Returns
    -------
    float
        Settlement reduction factor (0 to 1).
    """
    if n < 1.0:
        raise ValueError(f"Stress concentration ratio must be >= 1, got {n}")
    return 1.0 / (1.0 + as_ratio * (n - 1.0))


def composite_modulus(as_ratio: float, E_column: float,
                      E_soil: float) -> float:
    """Compute composite modulus of improved ground.

    E_comp = as * Ec + (1 - as) * Es

    Parameters
    ----------
    as_ratio : float
        Area replacement ratio.
    E_column : float
        Modulus of column material (kPa).
    E_soil : float
        Modulus of surrounding soil (kPa).

    Returns
    -------
    float
        Composite modulus (kPa).
    """
    return as_ratio * E_column + (1.0 - as_ratio) * E_soil


def improved_bearing_capacity(q_unreinforced: float, as_ratio: float,
                              n: float) -> float:
    """Compute improved bearing capacity with aggregate piers.

    q_improved = q_unreinforced * (1 + as * (n - 1))

    This is the Priebe improvement factor approach.

    Parameters
    ----------
    q_unreinforced : float
        Unreinforced bearing capacity (kPa).
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio.

    Returns
    -------
    float
        Improved bearing capacity (kPa).
    """
    return q_unreinforced * (1.0 + as_ratio * (n - 1.0))


def improved_settlement(S_unreinforced: float, as_ratio: float,
                        n: float) -> float:
    """Compute improved (reduced) settlement.

    S_improved = SRF * S_unreinforced

    Parameters
    ----------
    S_unreinforced : float
        Unreinforced settlement (mm).
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio.

    Returns
    -------
    float
        Improved settlement (mm).
    """
    srf = settlement_reduction_factor(as_ratio, n)
    return srf * S_unreinforced


def analyze_aggregate_piers(
    column_diameter: float,
    spacing: float,
    pattern: str = "triangular",
    E_column: float = 80000.0,
    E_soil: float = 5000.0,
    n: float = 5.0,
    q_unreinforced: float = 0.0,
    S_unreinforced: float = 0.0,
) -> AggregatePierResult:
    """Run a complete aggregate pier analysis.

    Parameters
    ----------
    column_diameter : float
        Pier column diameter (m).
    spacing : float
        Center-to-center spacing (m).
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    E_column : float
        Modulus of aggregate column (kPa). Default 80,000 kPa.
    E_soil : float
        Modulus of surrounding soil (kPa). Default 5,000 kPa.
    n : float
        Stress concentration ratio. Default 5.0.
    q_unreinforced : float
        Unreinforced bearing capacity (kPa). Default 0 (not computed).
    S_unreinforced : float
        Unreinforced settlement (mm). Default 0 (not computed).

    Returns
    -------
    AggregatePierResult
        Complete analysis results.
    """
    as_ratio = area_replacement_ratio(column_diameter, spacing, pattern)

    if as_ratio > 0.4:
        warnings.warn(
            f"Area replacement ratio {as_ratio:.3f} is unusually high (> 0.4). "
            "Verify column spacing and diameter."
        )
    if E_column < E_soil:
        warnings.warn(
            f"Column modulus ({E_column} kPa) is less than soil modulus "
            f"({E_soil} kPa). This is unusual for aggregate piers."
        )

    srf = settlement_reduction_factor(as_ratio, n)
    E_comp = composite_modulus(as_ratio, E_column, E_soil)

    q_improved = 0.0
    if q_unreinforced > 0:
        q_improved = improved_bearing_capacity(q_unreinforced, as_ratio, n)

    S_improved = 0.0
    if S_unreinforced > 0:
        S_improved = improved_settlement(S_unreinforced, as_ratio, n)

    return AggregatePierResult(
        area_replacement_ratio=as_ratio,
        stress_concentration_ratio=n,
        composite_modulus_kPa=E_comp,
        settlement_reduction_factor=srf,
        improved_bearing_kPa=q_improved,
        unreinforced_bearing_kPa=q_unreinforced,
        settlement_improved_mm=S_improved,
        settlement_unreinforced_mm=S_unreinforced,
        column_diameter_m=column_diameter,
        column_spacing_m=spacing,
        pattern=pattern,
    )
