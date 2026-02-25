"""
Beam analysis for braced and cantilever excavation walls.

Uses the tributary area method (California Trenching and Shoring Manual)
to compute support reactions and wall demands from apparent earth pressure
envelopes. For cantilever walls, uses classical limit equilibrium.

All units SI: kPa, kN/m, kN·m/m, meters.

References:
    California Dept. of Transportation, Trenching and Shoring Manual (2011)
    FHWA-IF-99-015, GEC-4, Sections 5.2-5.4
    Terzaghi & Peck (1967) Soil Mechanics in Engineering Practice
"""

import math
from typing import List, Optional, Tuple

from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel
from soe.earth_pressure import (
    rankine_Ka,
    rankine_Kp,
    active_pressure,
    passive_pressure,
    select_apparent_pressure,
    get_pressure_at_depth,
)
from soe.results import BracedExcavationResult, CantileverExcavationResult


# ============================================================================
# Tributary area method for multi-level braced excavations
# ============================================================================

def _integrate_pressure(z_top: float, z_bot: float, H: float,
                        shape: str, p_max: float,
                        n_points: int = 50) -> Tuple[float, float]:
    """Integrate apparent pressure over a span to get total force and centroid.

    Parameters
    ----------
    z_top, z_bot : float
        Depth bounds of the span (m).
    H : float
        Total excavation depth (m).
    shape : str
        "uniform" or "trapezoidal".
    p_max : float
        Maximum pressure ordinate (kPa).
    n_points : int
        Integration points.

    Returns
    -------
    tuple of (float, float)
        (total_force kN/m, centroid_depth m from z_top)
    """
    dz = (z_bot - z_top) / n_points
    total_force = 0.0
    moment_about_top = 0.0

    for i in range(n_points):
        z_mid = z_top + (i + 0.5) * dz
        p = get_pressure_at_depth(z_mid, H, shape, p_max)
        f = p * dz
        total_force += f
        moment_about_top += f * (z_mid - z_top)

    if total_force > 0:
        centroid = moment_about_top / total_force
    else:
        centroid = (z_bot - z_top) / 2.0

    return total_force, centroid


def _compute_simply_supported_max_moment(L: float, z_top: float,
                                         H: float, shape: str,
                                         p_max: float,
                                         n_points: int = 100) -> float:
    """Max bending moment in a simply-supported beam span under pressure load.

    Uses numerical integration: compute shear at discrete points, find where
    shear crosses zero, then integrate moment up to that point.

    Parameters
    ----------
    L : float
        Span length (m).
    z_top : float
        Depth of the top of this span (m).
    H : float
        Total excavation depth (m).
    shape : str
        Pressure envelope shape.
    p_max : float
        Max pressure ordinate (kPa).

    Returns
    -------
    float
        Maximum bending moment in the span (kN·m/m), positive.
    """
    if L <= 0:
        return 0.0

    # Get reactions at supports (top and bottom of span)
    total_force, centroid = _integrate_pressure(z_top, z_top + L, H,
                                                shape, p_max, n_points)
    # Reaction at top support (moment about bottom = 0)
    R_top = total_force * (L - centroid) / L if L > 0 else 0.0

    # Walk along the span computing shear and tracking max moment
    dz = L / n_points
    shear = R_top
    moment = 0.0
    max_moment = 0.0

    for i in range(n_points):
        z_mid = z_top + (i + 0.5) * dz
        p = get_pressure_at_depth(z_mid, H, shape, p_max)
        moment += shear * dz - p * dz * dz / 2.0
        shear -= p * dz
        max_moment = max(max_moment, abs(moment))

    return max_moment


def analyze_braced_excavation(geometry: ExcavationGeometry,
                              Fy: float = 345.0) -> BracedExcavationResult:
    """Analyze a multi-level braced excavation using the tributary area method.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Complete excavation geometry with soil layers and support levels.
    Fy : float
        Steel yield strength (MPa). Default 345 (Grade 50). Used to
        compute required section modulus.

    Returns
    -------
    BracedExcavationResult
        Analysis results including support reactions, moments, and demands.
    """
    geometry.validate()
    H = geometry.excavation_depth

    if not geometry.support_levels:
        raise ValueError(
            "Braced excavation requires at least one support level. "
            "Use analyze_cantilever_excavation() for unsupported walls."
        )

    # Step 1: Select apparent pressure envelope
    ap = select_apparent_pressure(geometry.soil_layers, H, geometry.surcharge)
    shape = ap["shape"]
    p_max = ap["max_pressure_kPa"]
    pressure_type = ap["type"]

    # Step 2: Define span boundaries
    # Spans: [0, sup1], [sup1, sup2], ..., [supN, H]
    support_depths = [s.depth for s in geometry.support_levels]
    boundaries = [0.0] + support_depths + [H]
    n_spans = len(boundaries) - 1

    # Step 3: Compute force on each span (tributary area)
    span_forces = []
    span_centroids = []
    for i in range(n_spans):
        z_top = boundaries[i]
        z_bot = boundaries[i + 1]
        force, centroid = _integrate_pressure(z_top, z_bot, H, shape, p_max)
        span_forces.append(force)
        span_centroids.append(centroid)

    # Step 4: Compute support reactions using tributary area method
    # Each interior support takes half the load from the span above and
    # half from the span below. End supports take their full tributary share.
    n_supports = len(support_depths)
    reactions = [0.0] * n_supports

    for i in range(n_spans):
        z_top = boundaries[i]
        z_bot = boundaries[i + 1]
        L = z_bot - z_top
        force = span_forces[i]
        centroid = span_centroids[i]

        if n_spans == 1:
            # Single span — all load to the one support
            reactions[0] += force
        elif i == 0:
            # Top span: reaction goes to first support
            reactions[0] += force
        elif i == n_spans - 1:
            # Bottom span: reaction goes to last support
            reactions[-1] += force
        else:
            # Interior span: distribute by lever arm to adjacent supports
            L_span = z_bot - z_top
            if L_span > 0:
                # Distance from top support of this span to centroid
                frac_bot = centroid / L_span
                frac_top = 1.0 - frac_bot
                reactions[i - 1] += force * frac_top
                reactions[i] += force * frac_bot
            else:
                reactions[i] += force

    # Step 5: Compute max moment in each span
    max_moment = 0.0
    max_moment_depth = 0.0
    max_shear = 0.0

    for i in range(n_spans):
        z_top = boundaries[i]
        z_bot = boundaries[i + 1]
        L = z_bot - z_top

        M = _compute_simply_supported_max_moment(L, z_top, H, shape, p_max)
        if M > max_moment:
            max_moment = M
            max_moment_depth = (z_top + z_bot) / 2.0  # approximate

        # Max shear is approximately the larger reaction at span ends
        V, _ = _integrate_pressure(z_top, z_bot, H, shape, p_max)
        max_shear = max(max_shear, V)

    # Step 6: Compute required section modulus
    # Sx_required = M / (0.66 * Fy) for allowable stress design
    # Fy in MPa, M in kN·m/m, Sx in cm³/m
    Fy_kPa = Fy * 1000.0  # MPa to kPa
    Fb = 0.66 * Fy_kPa  # allowable bending stress (kPa)
    required_Sx = max_moment / Fb * 1e6  # m³ to cm³

    # Step 7: Build support reaction list
    support_reactions = []
    for i, sup in enumerate(geometry.support_levels):
        support_reactions.append({
            "depth_m": round(sup.depth, 3),
            "load_kN_per_m": round(reactions[i], 2),
            "type": sup.support_type,
        })

    # Step 8: Compute required embedment
    from soe.embedment import compute_embedment
    embedment = compute_embedment(geometry)

    return BracedExcavationResult(
        excavation_depth=H,
        n_support_levels=n_supports,
        apparent_pressure_type=pressure_type,
        max_apparent_pressure_kPa=round(p_max, 2),
        support_reactions=support_reactions,
        max_moment_kNm_per_m=round(max_moment, 2),
        max_moment_depth_m=round(max_moment_depth, 2),
        max_shear_kN_per_m=round(max_shear, 2),
        required_embedment_m=round(embedment, 2),
        total_wall_length_m=round(H + embedment, 2),
        required_Sx_cm3=round(required_Sx, 1),
    )


def analyze_cantilever_excavation(
    geometry: ExcavationGeometry,
    FOS_passive: float = 1.5,
    Fy: float = 345.0,
) -> CantileverExcavationResult:
    """Analyze a cantilever (unbraced) excavation wall.

    Uses classical Rankine earth pressure (not apparent pressure) with
    limit equilibrium to find required embedment. Appropriate for
    short walls (typically H <= 4-5 m in sand, H <= 3 m in clay).

    Parameters
    ----------
    geometry : ExcavationGeometry
        Excavation geometry. support_levels should be empty.
    FOS_passive : float
        Factor of safety on passive resistance. Default 1.5.
    Fy : float
        Steel yield strength (MPa). Default 345.

    Returns
    -------
    CantileverExcavationResult
    """
    geometry.validate()
    H = geometry.excavation_depth

    if geometry.support_levels:
        raise ValueError(
            "Cantilever analysis requires no support levels. "
            "Use analyze_braced_excavation() for braced walls."
        )

    # Use first layer properties (simplified single-layer analysis)
    layer = geometry.soil_layers[0]
    gamma = layer.unit_weight
    phi = layer.friction_angle
    c = layer.cohesion
    q = geometry.surcharge

    Ka = rankine_Ka(phi)
    Kp = rankine_Kp(phi)

    # Iterate to find embedment depth D where moment balance is satisfied
    D_min = 0.5
    D_max = 4.0 * H
    n_steps = 500
    D_solution = D_max

    for i in range(n_steps):
        D = D_min + (D_max - D_min) * i / n_steps

        # Active forces: triangular pressure over (H + D) plus surcharge
        # Moment about base of wall
        z_total = H + D

        # Active force
        Pa_soil = 0.5 * Ka * gamma * z_total ** 2
        Pa_surcharge = Ka * q * z_total
        Pa_cohesion_reduction = 2.0 * c * math.sqrt(Ka) * z_total

        Ma_soil = Pa_soil * z_total / 3.0
        Ma_surcharge = Pa_surcharge * z_total / 2.0
        Ma_cohesion = Pa_cohesion_reduction * z_total / 2.0

        M_active = Ma_soil + Ma_surcharge - Ma_cohesion

        # Passive force below excavation (acts on embedded portion only)
        Pp = 0.5 * Kp * gamma * D ** 2 / FOS_passive
        Pp_cohesion = 2.0 * c * math.sqrt(Kp) * D / FOS_passive
        Mp_soil = Pp * (H + D / 3.0)
        Mp_cohesion = Pp_cohesion * (H + D / 2.0)

        M_passive = Mp_soil + Mp_cohesion

        if M_passive >= M_active:
            D_solution = D
            break

    # Apply 20% increase per USACE guidance
    D_design = D_solution * 1.2

    # Max moment (approximate: at depth where shear = 0)
    # For cantilever, max moment is typically near excavation level
    Pa_at_H = 0.5 * Ka * gamma * H ** 2 + Ka * q * H - 2.0 * c * math.sqrt(Ka) * H
    max_moment = abs(Pa_at_H * H / 3.0)

    # Max shear
    max_shear = abs(0.5 * Ka * gamma * H ** 2 + Ka * q * H
                    - 2.0 * c * math.sqrt(Ka) * H)

    # Required section modulus
    Fy_kPa = Fy * 1000.0
    Fb = 0.66 * Fy_kPa
    required_Sx = max_moment / Fb * 1e6 if Fb > 0 else 0.0

    return CantileverExcavationResult(
        excavation_depth=H,
        FOS_passive=FOS_passive,
        Ka=round(Ka, 4),
        Kp=round(Kp, 4),
        required_embedment_m=round(D_design, 2),
        total_wall_length_m=round(H + D_design, 2),
        max_moment_kNm_per_m=round(max_moment, 2),
        max_shear_kN_per_m=round(max_shear, 2),
        required_Sx_cm3=round(required_Sx, 1),
    )
