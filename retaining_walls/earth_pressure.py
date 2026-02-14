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


def horizontal_force_active(gamma: float, H: float, Ka: float,
                            c: float = 0.0, q: float = 0.0) -> Tuple[float, float]:
    """Resultant active horizontal force and its location.

    For a wall of height H with uniform backfill:
    Pa = 0.5 * Ka * gamma * H² + Ka * q * H - 2*c*sqrt(Ka)*H
    Applied at H/3 from base (triangular) or adjusted for surcharge.

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
        Total active horizontal force per unit width (kN/m).
    z_Pa : float
        Height of resultant above base (m).
    """
    # Triangular earth pressure component
    Pa_earth = 0.5 * Ka * gamma * H ** 2
    z_earth = H / 3.0

    # Surcharge component (rectangular)
    Pa_surcharge = Ka * q * H
    z_surcharge = H / 2.0

    # Cohesion component (reduces active, rectangular)
    Pa_cohesion = -2.0 * c * math.sqrt(Ka) * H
    z_cohesion = H / 2.0

    Pa_total = Pa_earth + Pa_surcharge + Pa_cohesion
    Pa_total = max(Pa_total, 0.0)

    if Pa_total > 0:
        z_Pa = (Pa_earth * z_earth + Pa_surcharge * z_surcharge +
                Pa_cohesion * z_cohesion) / Pa_total
        z_Pa = max(z_Pa, 0.0)
    else:
        z_Pa = H / 3.0

    return Pa_total, z_Pa


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
