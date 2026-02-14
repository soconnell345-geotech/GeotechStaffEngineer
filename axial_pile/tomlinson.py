"""
Tomlinson Alpha Method for skin friction in cohesive soils.

Computes unit skin friction fs = alpha * cu, where alpha is the
adhesion factor that depends on undrained shear strength and pile type.

All units are SI: kPa, meters, kN.

References:
    Tomlinson, M.J. (1980, 1985) — Adhesion factors for piles in clay
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Section 7.2.2.1
    FHWA Soils & Foundations Reference Manual, Vol II
"""

import warnings
from typing import Optional


def alpha_tomlinson(cu: float, pile_type: str = "steel") -> float:
    """Adhesion factor alpha for cohesive soil (Tomlinson method).

    Interpolates from Tomlinson's alpha vs cu relationship per
    FHWA GEC-12 Figure 7-17.

    Parameters
    ----------
    cu : float
        Undrained shear strength (kPa).
    pile_type : str, optional
        Pile material: "steel", "concrete", or "timber". Default "steel".
        Concrete and timber typically have higher adhesion.

    Returns
    -------
    float
        Adhesion factor alpha (dimensionless, 0 to 1).

    References
    ----------
    FHWA GEC-12, Figure 7-17 (after Tomlinson, 1980).
    API RP 2A-WSD (2000) also provides similar curves.
    """
    if cu <= 0:
        raise ValueError(f"Undrained shear strength must be positive, got {cu}")

    pile_type = pile_type.lower()

    # Tomlinson alpha vs cu relationship (FHWA GEC-12, simplified)
    # Higher alpha for lower cu; approaches 0.25-0.5 for stiff clays
    if pile_type in ("steel", "h_pile", "pipe"):
        # Steel piles — lower adhesion
        if cu <= 25:
            alpha = 1.0
        elif cu <= 50:
            alpha = 1.0 - 0.5 * (cu - 25) / 25  # 1.0 → 0.5
        elif cu <= 100:
            alpha = 0.5 - 0.1 * (cu - 50) / 50  # 0.5 → 0.4
        elif cu <= 200:
            alpha = 0.4 - 0.1 * (cu - 100) / 100  # 0.4 → 0.3
        else:
            alpha = 0.3
            if cu > 400:
                warnings.warn(f"cu={cu} kPa is very high; alpha may be lower than 0.3")

    elif pile_type in ("concrete", "timber"):
        # Concrete/timber — higher adhesion
        if cu <= 25:
            alpha = 1.0
        elif cu <= 50:
            alpha = 1.0 - 0.35 * (cu - 25) / 25  # 1.0 → 0.65
        elif cu <= 100:
            alpha = 0.65 - 0.15 * (cu - 50) / 50  # 0.65 → 0.50
        elif cu <= 200:
            alpha = 0.50 - 0.10 * (cu - 100) / 100  # 0.50 → 0.40
        else:
            alpha = 0.40

    else:
        raise ValueError(
            f"Unknown pile_type '{pile_type}'. Options: 'steel', 'concrete', 'timber'"
        )

    return alpha


def skin_friction_cohesive(cu: float, pile_perimeter: float,
                           layer_thickness: float,
                           pile_type: str = "steel") -> float:
    """Compute skin friction in a cohesive soil layer.

    Qs_layer = alpha * cu * perimeter * thickness

    Parameters
    ----------
    cu : float
        Undrained shear strength (kPa).
    pile_perimeter : float
        Pile perimeter (m).
    layer_thickness : float
        Layer thickness (m) over which this computation applies.
    pile_type : str, optional
        Pile material type. Default "steel".

    Returns
    -------
    float
        Skin friction from this layer (kN).
    """
    alpha = alpha_tomlinson(cu, pile_type)
    return alpha * cu * pile_perimeter * layer_thickness


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
    FHWA GEC-12, Eq 7-4. Nc=9 for deep foundations (Skempton, 1951).
    """
    return 9.0 * cu_tip * tip_area
