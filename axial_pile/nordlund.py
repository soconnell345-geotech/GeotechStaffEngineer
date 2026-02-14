"""
Nordlund Method for skin friction in cohesionless soils.

Computes skin friction using Nordlund's (1963, 1979) method and
end bearing using Meyerhof's method for driven piles in sand/gravel.

All units are SI: kPa, meters, kN, degrees.

References:
    Nordlund, R.L. (1963, 1979)
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Sections 7.2.1
    FHWA Soils & Foundations Reference Manual, Vol II
    Meyerhof, G.G. (1976) — Bearing capacity and settlement of pile foundations
"""

import math
import warnings
from typing import Optional

import numpy as np


def delta_from_phi(phi_deg: float, pile_material: str = "steel") -> float:
    """Estimate pile-soil friction angle delta.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    pile_material : str, optional
        "steel" (delta/phi ≈ 0.67-0.83), "concrete" (≈ 0.80-1.0),
        "timber" (≈ 0.80-1.0). Default "steel".

    Returns
    -------
    float
        Pile-soil friction angle delta (degrees).

    References
    ----------
    FHWA GEC-12, Table 7-1.
    """
    ratios = {
        "steel": 0.75,
        "concrete": 0.90,
        "timber": 0.90,
    }
    pile_material = pile_material.lower()
    ratio = ratios.get(pile_material, 0.75)
    return phi_deg * ratio


def nordlund_Kd(phi_deg: float, omega_deg: float = 0.0) -> float:
    """Coefficient of lateral earth pressure Kd for Nordlund method.

    Simplified from Nordlund's charts. For uniform (non-tapered) piles
    (omega=0), Kd depends primarily on phi.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    omega_deg : float, optional
        Pile taper angle (degrees). Default 0 (uniform pile).

    Returns
    -------
    float
        Coefficient of lateral earth pressure Kd.

    References
    ----------
    FHWA GEC-12, Figures 7-3 through 7-7 (Nordlund Kd charts).
    Simplified curve fit for omega=0.
    """
    # Simplified Kd for omega=0 (uniform piles)
    # From FHWA GEC-12 Figure 7-5 (V/V0 = 0.1 to 1.0 curves)
    # Using V/V0 ≈ 0.5 to 1.0 for typical displacement piles
    if omega_deg > 0:
        # Tapered piles have higher Kd; simplified increase
        kd_factor = 1.0 + 0.5 * omega_deg / 10.0  # rough approximation
    else:
        kd_factor = 1.0

    # Kd vs phi (omega=0, displacement pile, from FHWA charts)
    if phi_deg <= 25:
        Kd = 0.7
    elif phi_deg <= 30:
        Kd = 0.7 + (phi_deg - 25) * (1.0 - 0.7) / 5
    elif phi_deg <= 35:
        Kd = 1.0 + (phi_deg - 30) * (1.5 - 1.0) / 5
    elif phi_deg <= 40:
        Kd = 1.5 + (phi_deg - 35) * (2.5 - 1.5) / 5
    else:
        Kd = 2.5 + (phi_deg - 40) * 0.2

    return Kd * kd_factor


def nordlund_CF(delta_phi_ratio: float) -> float:
    """Correction factor CF for Kd when delta/phi differs from chart value.

    Parameters
    ----------
    delta_phi_ratio : float
        Actual delta/phi ratio.

    Returns
    -------
    float
        Correction factor CF. Approximately 1.0 when delta/phi matches
        the value used to develop the Kd charts.

    References
    ----------
    FHWA GEC-12, Figure 7-8.
    """
    # CF ≈ 1.0 for typical values; simplified linear adjustment
    # CF chart shows CF varies from ~0.4 to 1.5 based on delta/phi
    # For delta/phi = 0.75 (typical steel), CF ≈ 1.0
    return max(0.5, min(delta_phi_ratio / 0.75, 1.5))


def skin_friction_cohesionless(phi_deg: float, sigma_v: float,
                               pile_perimeter: float,
                               layer_thickness: float,
                               pile_material: str = "steel",
                               delta_phi_ratio: Optional[float] = None,
                               omega_deg: float = 0.0) -> float:
    """Compute skin friction in a cohesionless soil layer (Nordlund).

    Qs = Kd * CF * sigma_v' * sin(delta) * perimeter * dz

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    sigma_v : float
        Effective overburden pressure at the center of the layer (kPa).
    pile_perimeter : float
        Pile perimeter (m).
    layer_thickness : float
        Layer thickness (m).
    pile_material : str, optional
        Pile material. Default "steel".
    delta_phi_ratio : float, optional
        delta/phi ratio. If None, determined from pile_material.
    omega_deg : float, optional
        Pile taper angle (degrees). Default 0.

    Returns
    -------
    float
        Skin friction from this layer (kN).

    References
    ----------
    FHWA GEC-12, Eq 7-1.
    """
    if delta_phi_ratio is None:
        delta_deg = delta_from_phi(phi_deg, pile_material)
    else:
        delta_deg = phi_deg * delta_phi_ratio

    delta_rad = math.radians(delta_deg)
    omega_rad = math.radians(omega_deg)

    Kd = nordlund_Kd(phi_deg, omega_deg)
    CF = nordlund_CF(delta_deg / phi_deg if phi_deg > 0 else 0.75)

    fs = Kd * CF * sigma_v * math.sin(delta_rad + omega_rad)
    return fs * pile_perimeter * layer_thickness


def nordlund_Nq_prime(phi_deg: float) -> float:
    """Bearing capacity factor Nq' for pile tip (Meyerhof, 1976).

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).

    Returns
    -------
    float
        Nq' factor.

    References
    ----------
    FHWA GEC-12, Figure 7-14 (after Meyerhof, 1976).
    """
    # Meyerhof Nq' for driven piles (from FHWA charts)
    # Interpolated from Figure 7-14
    phi = phi_deg
    if phi <= 20:
        return 8.0
    elif phi <= 25:
        return 8.0 + (phi - 20) * (12 - 8) / 5
    elif phi <= 28:
        return 12 + (phi - 25) * (20 - 12) / 3
    elif phi <= 30:
        return 20 + (phi - 28) * (35 - 20) / 2
    elif phi <= 32:
        return 35 + (phi - 30) * (55 - 35) / 2
    elif phi <= 34:
        return 55 + (phi - 32) * (90 - 55) / 2
    elif phi <= 36:
        return 90 + (phi - 34) * (130 - 90) / 2
    elif phi <= 38:
        return 130 + (phi - 36) * (200 - 130) / 2
    elif phi <= 40:
        return 200 + (phi - 38) * (300 - 200) / 2
    elif phi <= 42:
        return 300 + (phi - 40) * (400 - 300) / 2
    else:
        return 400 + (phi - 42) * 50


def alpha_t_factor(Db_ratio: float) -> float:
    """Dimensionless factor alpha_t for Nordlund end bearing.

    Parameters
    ----------
    Db_ratio : float
        Pile depth / pile diameter ratio (D/b).

    Returns
    -------
    float
        alpha_t factor (0 to 1). Approaches 1.0 for deep piles.

    References
    ----------
    FHWA GEC-12, Figure 7-13.
    """
    # alpha_t approaches 1.0 quickly; simplified curve
    if Db_ratio <= 0:
        return 0.0
    elif Db_ratio <= 5:
        return 0.5 + 0.5 * Db_ratio / 5
    else:
        return 1.0


def end_bearing_cohesionless(phi_deg: float, sigma_v_tip: float,
                              tip_area: float,
                              pile_depth: float,
                              pile_width: float) -> float:
    """Compute end bearing in cohesionless soil (Nordlund/Meyerhof).

    Qt = alpha_t * Nq' * sigma_v' * At

    With limiting value: qt_limit from Meyerhof (1976).

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).
    sigma_v_tip : float
        Effective overburden at pile tip (kPa).
    tip_area : float
        Pile tip area (m²).
    pile_depth : float
        Pile embedment depth (m).
    pile_width : float
        Pile width or diameter (m).

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    FHWA GEC-12, Eq 7-3 and limiting qt from Figure 7-15.
    """
    Db = pile_depth / pile_width if pile_width > 0 else 0
    at = alpha_t_factor(Db)
    Nq = nordlund_Nq_prime(phi_deg)

    qt = at * Nq * sigma_v_tip

    # Limiting tip resistance (Meyerhof, 1976)
    # qt_limit ≈ Nq' * tan(phi) * 1000 kPa (approximate from charts)
    qt_limit = Nq * math.tan(math.radians(phi_deg)) * 50  # simplified limit
    qt_limit = max(qt_limit, 2000)  # minimum 2000 kPa

    qt = min(qt, qt_limit)
    return qt * tip_area
