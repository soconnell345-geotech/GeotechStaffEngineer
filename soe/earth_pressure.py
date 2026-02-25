"""
Earth pressure calculations for support of excavation design.

Implements classical Rankine/Coulomb coefficients (copied from sheet_pile module
per no-cross-module-import convention) plus Terzaghi-Peck apparent earth
pressure envelopes for braced/anchored excavations.

All units SI: kPa, kN/m³, degrees, meters.

References:
    Rankine (1857), Coulomb (1776)
    Terzaghi, K. & Peck, R.B. (1967) Soil Mechanics in Engineering Practice
    FHWA-IF-99-015, GEC-4: Ground Anchors and Anchored Systems
    Peck, R.B. (1969) Deep Excavation and Tunneling in Soft Ground, SOA Report
"""

import math
import warnings
from typing import List, Optional, Tuple


# ============================================================================
# Classical earth pressure coefficients (from sheet_pile/earth_pressure.py)
# ============================================================================

def rankine_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient Ka = tan²(45° - phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi_rad / 2) ** 2


def rankine_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient Kp = tan²(45° + phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def coulomb_Ka(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient."""
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    num = math.sin(alpha + phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha - delta)
    term2 = math.sin(phi + delta) * math.sin(phi - beta)
    term3 = math.sin(alpha - delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Ka calculation")

    denom = term1 * (1 + math.sqrt(term2 / term3)) ** 2
    return num / denom


def coulomb_Kp(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb passive earth pressure coefficient."""
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    num = math.sin(alpha + phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha + delta)
    term2 = math.sin(phi + delta) * math.sin(phi + beta)
    term3 = math.sin(alpha + delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Kp calculation")

    denom = term1 * (1 - math.sqrt(term2 / term3)) ** 2
    if denom <= 0:
        warnings.warn("Coulomb Kp computation failed; using Rankine Kp")
        return rankine_Kp(phi_deg)
    return num / denom


def K0(phi_deg: float) -> float:
    """At-rest earth pressure coefficient (Jaky). K0 = 1 - sin(phi)."""
    return 1.0 - math.sin(math.radians(phi_deg))


def active_pressure(gamma: float, z: float, Ka: float,
                    c: float = 0.0, q_surcharge: float = 0.0) -> float:
    """Active lateral pressure: sigma_a = Ka*(gamma*z + q) - 2*c*sqrt(Ka)."""
    return Ka * (gamma * z + q_surcharge) - 2.0 * c * math.sqrt(Ka)


def passive_pressure(gamma: float, z: float, Kp: float,
                     c: float = 0.0) -> float:
    """Passive lateral pressure: sigma_p = Kp*gamma*z + 2*c*sqrt(Kp)."""
    return Kp * gamma * z + 2.0 * c * math.sqrt(Kp)


def tension_crack_depth(c: float, gamma: float, Ka: float,
                        q_surcharge: float = 0.0) -> float:
    """Depth of tension crack in cohesive soil behind active wall."""
    if c <= 0 or Ka <= 0 or gamma <= 0:
        return 0.0
    z_crack = (2.0 * c / math.sqrt(Ka) - q_surcharge) / gamma
    return max(z_crack, 0.0)


# ============================================================================
# Apparent earth pressure envelopes (Terzaghi & Peck 1967, FHWA GEC-4)
# ============================================================================

def apparent_pressure_sand(gamma: float, H: float, Ka: float) -> float:
    """Apparent earth pressure for braced excavations in sand.

    Returns the uniform (rectangular) envelope ordinate:
        p = 0.65 * Ka * gamma * H

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    Ka : float
        Active earth pressure coefficient.

    Returns
    -------
    float
        Uniform apparent pressure (kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 3.
    FHWA-IF-99-015, Section 5.2.1.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    return 0.65 * Ka * gamma * H


def apparent_pressure_soft_clay(gamma: float, H: float, cu: float,
                                m: float = 1.0) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in soft to medium clay.

    Applies when stability number N = gamma*H/cu > 4.

    Returns a uniform (rectangular) envelope:
        Ka_apparent = 1 - m * (4*cu / (gamma*H))
        p = Ka_apparent * gamma * H

    where m = 1.0 for most cases (Peck 1969), m = 0.4 if movements are
    critical and excavation has no significant soft layer below the base.

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).
    m : float
        Empirical coefficient. Default 1.0.

    Returns
    -------
    tuple of (str, float)
        ("uniform", pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 4.
    FHWA-IF-99-015, Section 5.2.2.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")

    N = gamma * H / cu  # stability number
    Ka_apparent = 1.0 - m * (4.0 * cu / (gamma * H))
    # Ka_apparent should not be less than 0.25 per FHWA guidance
    Ka_apparent = max(Ka_apparent, 0.25)
    p = Ka_apparent * gamma * H
    return ("uniform", p)


def apparent_pressure_stiff_clay(gamma: float, H: float,
                                 cu: float) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in stiff clay.

    Applies when stability number N = gamma*H/cu <= 4.

    Returns a trapezoidal envelope with maximum ordinate between
    0.2*gamma*H and 0.4*gamma*H. Uses 0.2 + 0.4*(N/4) interpolation
    based on stability number.

    The trapezoidal shape is:
    - Zero at ground surface
    - Increases linearly to max over top 0.25*H
    - Constant max from 0.25*H to 0.75*H
    - Decreases linearly to zero at H (or to a reduced value)

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).

    Returns
    -------
    tuple of (str, float)
        ("trapezoidal", max_pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 5.
    FHWA-IF-99-015, Section 5.2.3.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")

    N = gamma * H / cu
    # Interpolate coefficient between 0.2 and 0.4 based on stability number
    coeff = min(0.2 + 0.2 * (N / 4.0), 0.4)
    p_max = coeff * gamma * H
    return ("trapezoidal", p_max)


def select_apparent_pressure(soil_layers, H: float,
                             surcharge: float = 0.0) -> dict:
    """Auto-select apparent pressure diagram from soil profile.

    Determines the controlling soil type over the excavation depth and
    returns the appropriate apparent pressure envelope.

    Parameters
    ----------
    soil_layers : list of SOEWallLayer
        Soil layers from ground surface downward.
    H : float
        Excavation depth (m).
    surcharge : float
        Surface surcharge (kPa). Default 0.

    Returns
    -------
    dict
        Keys: "type" ("sand", "soft_clay", "stiff_clay"),
              "shape" ("uniform" or "trapezoidal"),
              "max_pressure_kPa" (peak ordinate),
              "stability_number" (gamma*H/cu for clay, None for sand).
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if not soil_layers:
        raise ValueError("At least one soil layer is required")

    # Compute weighted averages over excavation depth
    total_gamma_h = 0.0
    total_cu_h = 0.0
    total_phi_h = 0.0
    sand_thickness = 0.0
    clay_thickness = 0.0
    remaining = H

    for layer in soil_layers:
        h = min(layer.thickness, remaining)
        total_gamma_h += layer.unit_weight * h
        total_cu_h += layer.cohesion * h
        total_phi_h += layer.friction_angle * h
        if layer.soil_type == "sand":
            sand_thickness += h
        else:
            clay_thickness += h
        remaining -= h
        if remaining <= 0:
            break

    gamma_avg = total_gamma_h / H
    cu_avg = total_cu_h / H
    phi_avg = total_phi_h / H

    # Determine controlling soil type
    if sand_thickness >= clay_thickness:
        # Predominantly sand
        Ka = rankine_Ka(phi_avg)
        p_max = apparent_pressure_sand(gamma_avg, H, Ka)
        return {
            "type": "sand",
            "shape": "uniform",
            "max_pressure_kPa": round(p_max, 2),
            "stability_number": None,
            "Ka": round(Ka, 4),
            "gamma_avg": round(gamma_avg, 2),
        }
    else:
        # Predominantly clay
        N = gamma_avg * H / cu_avg if cu_avg > 0 else float("inf")

        if N > 4:
            # Soft to medium clay
            shape, p_max = apparent_pressure_soft_clay(gamma_avg, H, cu_avg)
            return {
                "type": "soft_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_max, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
            }
        else:
            # Stiff clay
            shape, p_max = apparent_pressure_stiff_clay(
                gamma_avg, H, cu_avg
            )
            return {
                "type": "stiff_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_max, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
            }


def get_pressure_at_depth(z: float, H: float, shape: str,
                          p_max: float) -> float:
    """Return apparent pressure ordinate at depth z for given envelope shape.

    Parameters
    ----------
    z : float
        Depth from top of wall (m).
    H : float
        Total excavation depth (m).
    shape : str
        "uniform" or "trapezoidal".
    p_max : float
        Maximum pressure ordinate (kPa).

    Returns
    -------
    float
        Apparent pressure at depth z (kPa).
    """
    if z < 0 or z > H:
        return 0.0

    if shape == "uniform":
        return p_max

    elif shape == "trapezoidal":
        # Trapezoidal: ramps from 0 to p_max over top 0.25H,
        # constant from 0.25H to 0.75H, ramps to 0 at H.
        if z <= 0.25 * H:
            return p_max * (z / (0.25 * H)) if H > 0 else 0.0
        elif z <= 0.75 * H:
            return p_max
        else:
            return p_max * ((H - z) / (0.25 * H)) if H > 0 else 0.0

    return p_max  # fallback
