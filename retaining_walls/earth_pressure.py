"""
Earth pressure utilities for retaining wall design.

Reuses Rankine/Coulomb functions from the sheet_pile module and adds
resultant force computations for retaining wall stability checks.

All units are SI: kN, kPa, kN/m³, degrees, meters.
"""

import math
from typing import Tuple

# Reuse existing earth pressure functions from sheet_pile module
from sheet_pile.earth_pressure import (
    rankine_Ka,
    rankine_Kp,
    coulomb_Ka,
    coulomb_Kp,
    K0,
    active_pressure,
    passive_pressure,
    tension_crack_depth,
)


def rankine_Ka_sloped(phi_deg: float, beta_deg: float) -> float:
    """Rankine active earth pressure coefficient for sloped backfill.

    Ka = cos(beta) * [cos(beta) - sqrt(cos²(beta) - cos²(phi))]
                    / [cos(beta) + sqrt(cos²(beta) - cos²(phi))]

    Reduces to tan²(45 - phi/2) when beta = 0.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    beta_deg : float
        Backfill slope angle (degrees). Must be < phi_deg.

    Returns
    -------
    float
        Active earth pressure coefficient Ka.

    References
    ----------
    Das, B.M., Principles of Foundation Engineering, Eq 7.18
    """
    if beta_deg <= 0:
        return rankine_Ka(phi_deg)
    if beta_deg >= phi_deg:
        raise ValueError(
            f"Backfill slope ({beta_deg}°) must be less than phi ({phi_deg}°)"
        )
    phi = math.radians(phi_deg)
    beta = math.radians(beta_deg)
    cos_beta = math.cos(beta)
    cos_phi = math.cos(phi)
    sqrt_term = math.sqrt(cos_beta ** 2 - cos_phi ** 2)
    Ka = cos_beta * (cos_beta - sqrt_term) / (cos_beta + sqrt_term)
    return Ka


def horizontal_force_active(gamma: float, H: float, Ka: float,
                            c: float = 0.0, q: float = 0.0) -> Tuple[float, float]:
    """Resultant active thrust magnitude and its location.

    For a wall of height H with uniform backfill:
    Pa = 0.5 * Ka * gamma * H² + Ka * q * H - 2*c*sqrt(Ka)*H
    Applied at H/3 from base (triangular) or adjusted for surcharge. Where
    cohesion takes the pressure below zero near the top it is taken as zero
    there (the tension zone, GM-288: :func:`active_force_components`).

    Note
    ----
    ``Pa`` is the magnitude of the thrust along its line of action. It is
    horizontal only for Rankine theory with a vertical wall and level
    backfill. For Coulomb (wall friction ``delta``) or Rankine with sloped
    backfill (``beta``) the thrust is inclined at ``delta`` (resp. ``beta``)
    from horizontal, and the caller must decompose it into
    ``Ph = Pa*cos(delta)`` and ``Pv = Pa*sin(delta)`` (see
    ``retaining_walls.cantilever._active_thrust``).

    Parameters
    ----------
    gamma : float
        Unit weight of backfill (kN/m³).
    H : float
        Height over which pressure acts (m).
    Ka : float
        Active earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.
    q : float, optional
        Uniform surcharge (kPa). Default 0.

    Returns
    -------
    Pa : float
        Total active thrust per unit width (kN/m), along its line of action.
    z_Pa : float
        Height of resultant above base (m).
    """
    parts = active_force_components(gamma, H, Ka, c, q)
    return parts["Pa"], parts["z_Pa"]


def active_force_components(gamma: float, H: float, Ka: float,
                            c: float = 0.0, q: float = 0.0) -> dict:
    """The parts of :func:`horizontal_force_active` (GM-143):
    ``Pa_earth`` (0.5 Ka gamma H^2, at H/3), ``Pa_surcharge`` (Ka q H, at
    H/2), ``Pa_cohesion`` (-2 c sqrt(Ka) H, at H/2), their total ``Pa``
    and its height ``z_Pa`` above the base (m), as that function works
    them out (it returns this ``Pa`` and ``z_Pa``).

    The pressure at depth z is Ka (q + gamma z) - 2 c sqrt(Ka) (UFC
    3-220-20 Eq. 4-6, p. 178; FHWA NHI-06-089 Eq. 10-10, p. 10-14), and
    where cohesion takes it below zero it is zero, never a suction pulling
    on the wall (GM-288): UFC 3-220-20 Fig. 4-5 (p. 183), "In no case
    should negative pressure be considered"; NHI-06-089 p. 10-15, "The
    active earth pressure acting against the wall within the depth of the
    tension crack is assumed to be zero". ``z_tension`` is the depth of
    that zone below the top of the pressure plane (0 without one) and
    ``Pa_tension_zone`` the negative thrust over it that ``Pa`` leaves out
    (>= 0), so Pa = Pa_earth + Pa_surcharge + Pa_cohesion +
    Pa_tension_zone."""
    # Triangular earth pressure component
    Pa_earth = 0.5 * Ka * gamma * H ** 2

    # Surcharge component (rectangular)
    Pa_surcharge = Ka * q * H

    # Cohesion component (reduces active, rectangular)
    Pa_cohesion = -2.0 * c * math.sqrt(Ka) * H

    # The pressure at the top of the plane, sigma_0, and its gradient
    sigma_0 = Ka * q - 2.0 * c * math.sqrt(Ka)
    grad = Ka * gamma
    if sigma_0 >= 0:
        # Pressure everywhere: the trapezoid, as the three parts sum
        z_tension = 0.0
        Pa_total = Pa_earth + Pa_surcharge + Pa_cohesion
        z_Pa = ((Pa_earth * H / 3.0 + (Pa_surcharge + Pa_cohesion) * H / 2.0)
                / Pa_total if Pa_total > 0 else H / 3.0)
    elif grad > 0 and -sigma_0 / grad < H:
        # Zero over the tension zone, a triangle below it
        z_tension = -sigma_0 / grad
        Pa_total = 0.5 * grad * (H - z_tension) ** 2
        z_Pa = (H - z_tension) / 3.0
    else:
        # Cohesion holds the whole height: no thrust
        z_tension = H
        Pa_total = 0.0
        z_Pa = H / 3.0
    Pa_tension_zone = Pa_total - (Pa_earth + Pa_surcharge + Pa_cohesion)

    return {"Pa_earth": Pa_earth, "Pa_surcharge": Pa_surcharge,
            "Pa_cohesion": Pa_cohesion, "Pa_tension_zone": Pa_tension_zone,
            "z_tension": z_tension, "Pa": Pa_total, "z_Pa": z_Pa}


def active_force_components_water(gamma: float, gamma_b: float, H: float,
                                  z_w: float, Ka: float, c: float = 0.0,
                                  q: float = 0.0) -> dict:
    """:func:`active_force_components` with a water table ``z_w`` below the
    top of the pressure plane (0 <= z_w < H; GM-292): the effective
    vertical stress is q + gamma z above it and q + gamma z_w + gamma_b
    (z - z_w) below it, gamma_b the buoyant unit weight, and the active
    pressure Ka sigma'_v - 2 c sqrt(Ka), zero where that is negative, as
    UFC 3-220-20 Figs. 4-5 and 4-16(a) (pp. 183, 202) and FHWA NHI-06-089
    Fig. 10-12 (p. 10-27) draw it: "Instead of the total unit weight
    (gamma), the buoyant or effective unit weight (gamma_b) is used in the
    earth pressure equations for soils below the water table" (UFC
    p. 200). The water's own pressure is not in it.

    The same keys: ``Pa_earth`` is the soil weight's part, Ka [gamma z_w^2
    / 2 + gamma z_w (H - z_w) + gamma_b (H - z_w)^2 / 2], ``Pa_surcharge``
    Ka q H, ``Pa_cohesion`` -2 c sqrt(Ka) H, ``Pa_tension_zone`` what the
    zero pressure in the tension zone adds back, ``z_tension`` that
    zone's depth below the top, and ``Pa`` with its height ``z_Pa`` above
    the base, the integral of the two straight-line pieces."""
    root = math.sqrt(Ka)

    def p(z):
        sigma = q + gamma * min(z, z_w) + gamma_b * max(z - z_w, 0.0)
        return Ka * sigma - 2.0 * c * root

    Pa_earth = Ka * (0.5 * gamma * z_w ** 2 + gamma * z_w * (H - z_w)
                     + 0.5 * gamma_b * (H - z_w) ** 2)
    Pa_surcharge = Ka * q * H
    Pa_cohesion = -2.0 * c * root * H

    # The pressure rises with depth on both pieces (gamma, gamma_b >= 0),
    # so the zero zone runs from the top down to z_tension
    total = moment = 0.0
    z_tension = H
    for z0, z1 in ((0.0, z_w), (z_w, H)):
        if z1 <= z0:
            continue
        p0, p1 = p(z0), p(z1)
        if p1 <= 0.0:
            continue
        if p0 < 0.0:
            z0 = z0 + (z1 - z0) * (-p0) / (p1 - p0)
            p0 = 0.0
        z_tension = min(z_tension, z0)
        L = z1 - z0
        F = 0.5 * (p0 + p1) * L
        depth = z0 + L * (p0 + 2.0 * p1) / (3.0 * (p0 + p1))
        total += F
        moment += F * (H - depth)
    z_Pa = moment / total if total > 0 else H / 3.0
    return {"Pa_earth": Pa_earth, "Pa_surcharge": Pa_surcharge,
            "Pa_cohesion": Pa_cohesion,
            "Pa_tension_zone": total - (Pa_earth + Pa_surcharge
                                        + Pa_cohesion),
            "z_tension": z_tension,
            "Pa": total, "z_Pa": z_Pa}


def horizontal_force_passive(gamma: float, D: float, Kp: float,
                             c: float = 0.0) -> Tuple[float, float]:
    """Resultant passive horizontal force and its location.

    Parameters
    ----------
    gamma : float
        Unit weight of soil in front of wall (kN/m³).
    D : float
        Depth of soil in front of wall (m).
    Kp : float
        Passive earth pressure coefficient.
    c : float, optional
        Cohesion (kPa). Default 0.

    Returns
    -------
    Pp : float
        Total passive horizontal force per unit width (kN/m).
    z_Pp : float
        Height of resultant above base of embedment (m).
    """
    Pp_earth = 0.5 * Kp * gamma * D ** 2
    z_earth = D / 3.0

    Pp_cohesion = 2.0 * c * math.sqrt(Kp) * D
    z_cohesion = D / 2.0

    Pp_total = Pp_earth + Pp_cohesion

    if Pp_total > 0:
        z_Pp = (Pp_earth * z_earth + Pp_cohesion * z_cohesion) / Pp_total
    else:
        z_Pp = D / 3.0

    return Pp_total, z_Pp
