"""
Beta (Effective Stress) Method for axial pile capacity.

The beta method computes skin friction as fs = beta * sigma_v' and
end bearing as qt = Nt * sigma_v', where beta = Ks * tan(delta).

Applicable to any soil type (sand, clay, silt).

All units are SI: kPa, meters, kN.

References:
    Fellenius, B.H. (1991) — "Pile foundations"
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Section 7.2.3
    Burland, J.B. (1973) — Shaft friction of piles in clay
"""

import math
import warnings


def beta_from_phi(phi_deg: float, OCR: float = 1.0,
                  method: str = "fellenius") -> float:
    """Estimate beta coefficient from soil friction angle.

    beta = Ks * tan(delta)

    Parameters
    ----------
    phi_deg : float
        Effective friction angle (degrees).
    OCR : float, optional
        Overconsolidation ratio. Default 1.0 (NC soil).
    method : str, optional
        "fellenius" (default): beta = (1-sin(phi))*tan(phi). Normally-
            consolidated closed form; OCR is NOT used (a warning is issued if
            OCR != 1 — use "burland" for overconsolidated soils).
        "burland": beta = (1-sin(phi))*sqrt(OCR)*tan(phi). OCR-enhanced K0
            (Burland 1973; Mayne & Kulhawy 1982).

    Returns
    -------
    float
        Beta coefficient (dimensionless).

    References
    ----------
    Fellenius (1991), FHWA GEC-12 Table 7-9.
    Burland (1973), Geotechnique, Vol. 23, No. 2.
    """
    if phi_deg <= 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")

    phi_rad = math.radians(phi_deg)
    method = method.lower()

    if method == "fellenius":
        # Fellenius (1991) normally-consolidated closed form:
        #   beta = (1 - sin(phi)) * tan(phi)   [K0 = 1 - sin(phi), delta = phi]
        # This NC form does not use OCR; use method="burland" for an
        # OCR-enhanced beta in overconsolidated soils.
        if OCR != 1.0:
            warnings.warn(
                f"The Fellenius NC beta form ignores OCR (got OCR={OCR}). "
                "Use method='burland' for an OCR-enhanced beta in "
                "overconsolidated soils.",
                stacklevel=2,
            )
        beta = (1.0 - math.sin(phi_rad)) * math.tan(phi_rad)
    elif method == "burland":
        # Burland (1973) with the Mayne & Kulhawy (1982) OCR enhancement:
        #   K0 = (1 - sin(phi)) * sqrt(OCR);  beta = K0 * tan(phi)
        K0 = (1.0 - math.sin(phi_rad)) * OCR**0.5
        beta = K0 * math.tan(phi_rad)
    else:
        raise ValueError(f"Unknown method '{method}'. Options: 'fellenius', 'burland'")

    return beta


def skin_friction_beta(sigma_v: float, beta: float,
                       pile_perimeter: float,
                       layer_thickness: float) -> float:
    """Compute skin friction using the beta method.

    Qs_layer = beta * sigma_v' * perimeter * thickness

    Parameters
    ----------
    sigma_v : float
        Effective vertical stress at center of layer (kPa).
    beta : float
        Beta coefficient.
    pile_perimeter : float
        Pile perimeter (m).
    layer_thickness : float
        Layer thickness (m).

    Returns
    -------
    float
        Skin friction from this layer (kN).
    """
    fs = beta * sigma_v
    return fs * pile_perimeter * layer_thickness


def end_bearing_beta(sigma_v_tip: float, Nt: float,
                     tip_area: float) -> float:
    """Compute end bearing using the effective stress method.

    Qt = Nt * sigma_v' * At

    Parameters
    ----------
    sigma_v_tip : float
        Effective vertical stress at pile tip (kPa).
    Nt : float
        Bearing capacity factor. Typical values:
        Sand: Nt = 30-150 (function of phi)
        Clay: Nt = 3-9
    tip_area : float
        Pile tip area (m²).

    Returns
    -------
    float
        End bearing capacity (kN).
    """
    return Nt * sigma_v_tip * tip_area


def Nt_from_phi(phi_deg: float) -> float:
    """Estimate Nt factor from soil friction angle.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nt.

    References
    ----------
    Fellenius (1991), FHWA GEC-12 Table 7-9.
    """
    if phi_deg <= 0:
        return 3.0  # pure clay, Nt ≈ 3-9
    elif phi_deg <= 20:
        return 3.0 + (phi_deg / 20) * 7  # 3 to 10
    elif phi_deg <= 28:
        return 10 + (phi_deg - 20) * 2.5  # 10 to 30
    elif phi_deg <= 33:
        return 30 + (phi_deg - 28) * 8  # 30 to 70
    elif phi_deg <= 38:
        return 70 + (phi_deg - 33) * 16  # 70 to 150
    else:
        return 150 + (phi_deg - 38) * 20
