"""
Embedment depth calculations for SOE walls.

Below the lowest support level, the wall acts as a cantilever resisting
passive earth pressure. This module computes the required embedment
using moment balance about the lowest support.

All units SI: m, kPa, kN/m³, degrees.

References:
    USACE EM 1110-2-2504, Chapter 5
    FHWA-IF-99-015, GEC-4, Section 5.4
    California Trenching and Shoring Manual (2011), Section 5
"""

import math
from typing import Optional

from soe.geometry import ExcavationGeometry, SOEWallLayer
from soe.earth_pressure import rankine_Ka, rankine_Kp


def compute_embedment(geometry: ExcavationGeometry,
                      FOS_passive: float = 1.5) -> float:
    """Compute required wall embedment below excavation level.

    For braced walls, embedment provides passive resistance below the
    lowest support. Uses moment balance about the lowest support level.

    For cantilever walls (no supports), computes embedment using
    moment balance about the excavation base.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Complete excavation geometry.
    FOS_passive : float
        Factor of safety on passive resistance. Default 1.5.

    Returns
    -------
    float
        Required embedment depth below excavation level (m).
        Includes 20% increase per USACE guidance.
    """
    H = geometry.excavation_depth

    # Get soil properties at/below excavation level
    # Use the deepest layer that extends below excavation
    depth_accumulated = 0.0
    embed_layer = geometry.soil_layers[-1]  # default to bottom layer
    for layer in geometry.soil_layers:
        depth_accumulated += layer.thickness
        if depth_accumulated >= H:
            embed_layer = layer
            break

    gamma = embed_layer.unit_weight
    phi = embed_layer.friction_angle
    c = embed_layer.cohesion

    Ka = rankine_Ka(phi)
    Kp = rankine_Kp(phi)

    if geometry.support_levels:
        lowest_support = max(geometry.support_levels, key=lambda s: s.depth)
        pivot_depth = lowest_support.depth
    else:
        pivot_depth = 0.0

    # Distance from pivot to excavation base
    h_below_pivot = H - pivot_depth

    # Iterate to find D where passive moment about pivot >= active moment
    D_min = 0.1
    D_max = 3.0 * H
    n_steps = 500
    D_solution = D_max

    for i in range(n_steps):
        D = D_min + (D_max - D_min) * i / n_steps

        # Active forces on the wall from pivot down to base of wall
        depth_active = h_below_pivot + D
        Pa = 0.5 * Ka * gamma * depth_active ** 2
        Ma = Pa * depth_active / 3.0  # moment about pivot

        # Passive resistance below excavation (over embedment D)
        Pp = 0.5 * Kp * gamma * D ** 2 / FOS_passive
        Pp_cohesion = 2.0 * c * math.sqrt(Kp) * D / FOS_passive
        # Moment arm: passive resultant is at H + D/3 below ground surface,
        # so arm from pivot = (H - pivot_depth) + D/3
        arm_Pp = h_below_pivot + D / 3.0
        arm_Pp_c = h_below_pivot + D / 2.0
        Mp = Pp * arm_Pp + Pp_cohesion * arm_Pp_c

        if Mp >= Ma:
            D_solution = D
            break

    # Apply 20% increase per USACE
    return D_solution * 1.2
