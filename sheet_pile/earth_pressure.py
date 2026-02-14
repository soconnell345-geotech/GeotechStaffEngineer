"""
Earth pressure calculations for sheet pile wall design.

Implements Rankine and Coulomb active/passive earth pressure coefficients
and lateral pressure distributions for layered soils with water.

All units are SI: kPa, kN/m³, degrees, meters.

References:
    Rankine, W.J.M. (1857) — On the stability of loose earth
    Coulomb, C.A. (1776) — Essai sur une application des regles de maximis
    USACE EM 1110-2-2504, Chapter 3
    Das, B.M., "Principles of Foundation Engineering", Ch 7
"""

import math
import warnings
from typing import Optional


def rankine_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient.

    Ka = tan²(45° - phi/2)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.

    References
    ----------
    Rankine (1857); USACE EM 1110-2-2504, Eq 3-1.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi_rad / 2) ** 2


def rankine_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient.

    Kp = tan²(45° + phi/2)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.

    References
    ----------
    Rankine (1857); USACE EM 1110-2-2504, Eq 3-2.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def coulomb_Ka(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default 0 (smooth wall = Rankine).
    alpha_deg : float, optional
        Wall inclination from horizontal (degrees). Default 90 (vertical).
    beta_deg : float, optional
        Backfill slope angle (degrees). Default 0 (horizontal).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.

    References
    ----------
    Coulomb (1776); USACE EM 1110-2-2504, Eq 3-3.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    sin_alpha = math.sin(alpha)
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
    """Coulomb passive earth pressure coefficient.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default 0.
    alpha_deg : float, optional
        Wall inclination from horizontal (degrees). Default 90.
    beta_deg : float, optional
        Surface slope angle (degrees). Default 0.

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.

    References
    ----------
    Coulomb (1776); USACE EM 1110-2-2504, Eq 3-4.

    Notes
    -----
    Coulomb Kp can be unconservative for high delta/phi ratios.
    Use log-spiral method for delta/phi > 0.5.
    """
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    num = math.sin(alpha - phi) ** 2
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
    """At-rest earth pressure coefficient (Jaky's formula).

    K0 = 1 - sin(phi)

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).

    Returns
    -------
    float
        At-rest earth pressure coefficient K0.
    """
    return 1.0 - math.sin(math.radians(phi_deg))


def active_pressure(gamma: float, z: float, Ka: float,
                    c: float = 0.0, q_surcharge: float = 0.0) -> float:
    """Active lateral earth pressure at depth z.

    sigma_a = Ka * (gamma * z + q_surcharge) - 2*c*sqrt(Ka)

    Parameters
    ----------
    gamma : float
        Unit weight of soil (kN/m³). Use effective weight below GWT.
    z : float
        Depth below the top of the soil layer (m).
    Ka : float
        Active earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.
    q_surcharge : float, optional
        Uniform surcharge at ground surface (kPa). Default 0.

    Returns
    -------
    float
        Active lateral pressure (kPa). Can be negative near surface
        in cohesive soils (tension crack zone).
    """
    return Ka * (gamma * z + q_surcharge) - 2.0 * c * math.sqrt(Ka)


def passive_pressure(gamma: float, z: float, Kp: float,
                     c: float = 0.0) -> float:
    """Passive lateral earth pressure at depth z.

    sigma_p = Kp * gamma * z + 2*c*sqrt(Kp)

    Parameters
    ----------
    gamma : float
        Unit weight of soil (kN/m³). Use effective weight below GWT.
    z : float
        Depth below the excavation level (m).
    Kp : float
        Passive earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.

    Returns
    -------
    float
        Passive lateral pressure (kPa).
    """
    return Kp * gamma * z + 2.0 * c * math.sqrt(Kp)


def tension_crack_depth(c: float, gamma: float, Ka: float,
                        q_surcharge: float = 0.0) -> float:
    """Depth of tension crack in cohesive soil behind an active wall.

    The active pressure is zero when Ka*(gamma*z + q) = 2*c*sqrt(Ka),
    so z_crack = (2*c/sqrt(Ka) - Ka*q) / (Ka*gamma).

    Simplified for q=0: z_crack = 2*c / (gamma * sqrt(Ka))

    Parameters
    ----------
    c : float
        Cohesion (kPa).
    gamma : float
        Unit weight (kN/m³).
    Ka : float
        Active earth pressure coefficient.
    q_surcharge : float, optional
        Surcharge (kPa). Default 0.

    Returns
    -------
    float
        Tension crack depth (m). Returns 0 if no tension crack.
    """
    if c <= 0 or Ka <= 0 or gamma <= 0:
        return 0.0
    z_crack = (2.0 * c / math.sqrt(Ka) - q_surcharge) / gamma
    return max(z_crack, 0.0)
