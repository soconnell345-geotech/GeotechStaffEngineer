"""
Cantilever retaining wall external stability analysis.

Checks sliding, overturning, and bearing capacity for a cantilever
(or gravity) retaining wall with Rankine or Coulomb earth pressures.

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    Das, B.M., Principles of Foundation Engineering, Ch 13
    AASHTO LRFD Section 11.6
"""

import math
from typing import Dict, Any, Optional

from retaining_walls.geometry import CantileverWallGeometry
from retaining_walls.earth_pressure import (
    rankine_Ka, rankine_Ka_sloped, rankine_Kp, coulomb_Ka,
    horizontal_force_active, horizontal_force_passive,
)
from retaining_walls.results import CantileverWallResult


def _compute_wall_weights(geom: CantileverWallGeometry,
                          gamma_backfill: float,
                          gamma_concrete: float = 24.0):
    """Compute wall component weights and moment arms about toe.

    Returns list of (weight, arm) tuples for moment calculation.
    """
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_base = geom.stem_thickness_base
    t_stem_top = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length

    weights = []

    # Base slab
    W_base = gamma_concrete * B * t_base
    x_base = B / 2.0
    weights.append((W_base, x_base))

    # Stem (trapezoidal: wider at base)
    # Rectangular part (stem_top thickness)
    W_stem_rect = gamma_concrete * t_stem_top * h_stem
    x_stem_rect = t_toe + t_stem_top / 2.0
    weights.append((W_stem_rect, x_stem_rect))

    # Triangular part of stem taper
    if t_stem_base > t_stem_top:
        t_taper = t_stem_base - t_stem_top
        W_stem_tri = gamma_concrete * 0.5 * t_taper * h_stem
        # Centroid of triangle at 1/3 from wider base (back face)
        x_stem_tri = t_toe + t_stem_top + t_taper / 3.0
        weights.append((W_stem_tri, x_stem_tri))

    # Soil on heel (backfill above heel, behind stem)
    if heel > 0:
        W_soil_heel = gamma_backfill * heel * h_stem
        x_soil_heel = t_toe + t_stem_base + heel / 2.0
        weights.append((W_soil_heel, x_soil_heel))

    # Soil on toe (if any, above toe in front of stem)
    # Typically minimal for cantilever walls, skip for simplicity

    return weights


def check_sliding(geom: CantileverWallGeometry,
                  gamma_backfill: float, phi_backfill: float,
                  c_backfill: float = 0.0,
                  phi_foundation: float = None,
                  c_foundation: float = 0.0,
                  gamma_concrete: float = 24.0,
                  FOS_required: float = 1.5,
                  pressure_method: str = "rankine",
                  include_passive: bool = False,
                  gamma_foundation: float = None) -> Dict[str, Any]:
    """Check sliding stability.

    FOS_sliding = (V * tan(delta_b) + ca * B + Pp) / Pa_horizontal

    Parameters
    ----------
    geom : CantileverWallGeometry
        Wall geometry.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    c_backfill : float, optional
        Backfill cohesion (kPa). Default 0.
    phi_foundation : float, optional
        Foundation soil friction angle (degrees). If None, uses phi_backfill.
    c_foundation : float, optional
        Foundation soil cohesion (kPa). Default 0.
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 24.
    FOS_required : float, optional
        Required FOS. Default 1.5.
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    include_passive : bool, optional
        If True, include passive resistance Pp in front of wall.
        If has_shear_key, passive is computed on key_depth;
        otherwise on base_thickness. Default False.
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m³). If None, uses gamma_backfill.

    Returns
    -------
    dict
        FOS_sliding, driving_force, resisting_force, Pp_kN_per_m, passes.
    """
    if phi_foundation is None:
        phi_foundation = phi_backfill
    if gamma_foundation is None:
        gamma_foundation = gamma_backfill

    H = geom.H_active
    B = geom.base_width

    # Active force
    if pressure_method == "coulomb":
        delta_wall = 2.0 / 3.0 * phi_backfill
        Ka = coulomb_Ka(phi_backfill, delta_wall,
                        beta_deg=geom.backfill_slope)
    else:
        # Use sloped Rankine Ka when backfill is sloped
        if geom.backfill_slope > 0:
            Ka = rankine_Ka_sloped(phi_backfill, geom.backfill_slope)
        else:
            Ka = rankine_Ka(phi_backfill)

    Pa, _ = horizontal_force_active(gamma_backfill, H, Ka,
                                    c_backfill, geom.surcharge)

    # Vertical force (sum of weights)
    weights = _compute_wall_weights(geom, gamma_backfill, gamma_concrete)
    V = sum(w for w, _ in weights)

    # Add vertical component of surcharge on heel
    V_surcharge = geom.surcharge * geom.heel_length
    V += V_surcharge

    # Base friction: delta_b = 2/3 * phi_foundation
    delta_b = 2.0 / 3.0 * phi_foundation
    ca = 2.0 / 3.0 * c_foundation

    resisting = V * math.tan(math.radians(delta_b)) + ca * B

    # Passive resistance (optional)
    Pp = 0.0
    if include_passive:
        Kp = rankine_Kp(phi_foundation)
        if geom.has_shear_key and geom.key_depth > 0:
            # Passive resistance on shear key depth
            D = geom.key_depth
        else:
            # Passive resistance on embedment (base_thickness)
            D = geom.base_thickness
        Pp = 0.5 * Kp * gamma_foundation * D ** 2
        resisting += Pp

    driving = Pa

    FOS = resisting / driving if driving > 0 else 99.9

    return {
        "FOS_sliding": round(FOS, 3),
        "driving_force_kN_per_m": round(driving, 1),
        "resisting_force_kN_per_m": round(resisting, 1),
        "Pp_kN_per_m": round(Pp, 1),
        "passes": FOS >= FOS_required,
    }


def check_overturning(geom: CantileverWallGeometry,
                      gamma_backfill: float, phi_backfill: float,
                      c_backfill: float = 0.0,
                      gamma_concrete: float = 24.0,
                      FOS_required: float = 2.0,
                      pressure_method: str = "rankine") -> Dict[str, Any]:
    """Check overturning stability about the toe.

    FOS_OT = M_stabilizing / M_overturning

    Parameters
    ----------
    geom : CantileverWallGeometry
        Wall geometry.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    c_backfill : float, optional
        Backfill cohesion (kPa). Default 0.
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 24.
    FOS_required : float, optional
        Required FOS. Default 2.0.
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".

    Returns
    -------
    dict
        FOS_overturning, stabilizing_moment, overturning_moment, passes.
    """
    H = geom.H_active

    if pressure_method == "coulomb":
        delta_wall = 2.0 / 3.0 * phi_backfill
        Ka = coulomb_Ka(phi_backfill, delta_wall,
                        beta_deg=geom.backfill_slope)
    else:
        if geom.backfill_slope > 0:
            Ka = rankine_Ka_sloped(phi_backfill, geom.backfill_slope)
        else:
            Ka = rankine_Ka(phi_backfill)

    Pa, z_Pa = horizontal_force_active(gamma_backfill, H, Ka,
                                       c_backfill, geom.surcharge)

    # Overturning moment about toe
    M_overturning = Pa * z_Pa

    # Stabilizing moments about toe
    weights = _compute_wall_weights(geom, gamma_backfill, gamma_concrete)
    M_stabilizing = sum(w * x for w, x in weights)

    # Surcharge on heel
    heel = geom.heel_length
    if heel > 0 and geom.surcharge > 0:
        W_q = geom.surcharge * heel
        x_q = geom.toe_length + geom.stem_thickness_base + heel / 2.0
        M_stabilizing += W_q * x_q

    FOS = M_stabilizing / M_overturning if M_overturning > 0 else 99.9

    return {
        "FOS_overturning": round(FOS, 3),
        "stabilizing_moment_kNm_per_m": round(M_stabilizing, 1),
        "overturning_moment_kNm_per_m": round(M_overturning, 1),
        "passes": FOS >= FOS_required,
    }


def check_bearing(geom: CantileverWallGeometry,
                  gamma_backfill: float, phi_backfill: float,
                  c_backfill: float = 0.0,
                  q_allowable: float = None,
                  gamma_concrete: float = 24.0,
                  pressure_method: str = "rankine") -> Dict[str, Any]:
    """Check bearing pressure and eccentricity.

    Parameters
    ----------
    geom : CantileverWallGeometry
        Wall geometry.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    c_backfill : float, optional
        Backfill cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing pressure (kPa). If None, only eccentricity checked.
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 24.
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".

    Returns
    -------
    dict
        q_toe, q_heel, eccentricity, in_middle_third, FOS_bearing.
    """
    H = geom.H_active
    B = geom.base_width

    if pressure_method == "coulomb":
        delta_wall = 2.0 / 3.0 * phi_backfill
        Ka = coulomb_Ka(phi_backfill, delta_wall,
                        beta_deg=geom.backfill_slope)
    else:
        if geom.backfill_slope > 0:
            Ka = rankine_Ka_sloped(phi_backfill, geom.backfill_slope)
        else:
            Ka = rankine_Ka(phi_backfill)

    Pa, z_Pa = horizontal_force_active(gamma_backfill, H, Ka,
                                       c_backfill, geom.surcharge)

    # Weights and moments about toe
    weights = _compute_wall_weights(geom, gamma_backfill, gamma_concrete)
    V = sum(w for w, _ in weights)
    M_stab = sum(w * x for w, x in weights)

    # Surcharge on heel
    heel = geom.heel_length
    if heel > 0 and geom.surcharge > 0:
        W_q = geom.surcharge * heel
        x_q = geom.toe_length + geom.stem_thickness_base + heel / 2.0
        V += W_q
        M_stab += W_q * x_q

    M_over = Pa * z_Pa

    # Location of resultant from toe
    x_resultant = (M_stab - M_over) / V if V > 0 else B / 2.0
    eccentricity = B / 2.0 - x_resultant

    in_middle_third = abs(eccentricity) <= B / 6.0

    # Bearing pressures (Meyerhof: trapezoidal if in middle third)
    if in_middle_third:
        q_toe = V / B * (1.0 + 6.0 * eccentricity / B)
        q_heel = V / B * (1.0 - 6.0 * eccentricity / B)
    else:
        # Resultant outside middle third: triangular distribution
        # Effective width = 3 * x_resultant (toe side only)
        B_eff = 3.0 * x_resultant
        q_toe = 2.0 * V / B_eff if B_eff > 0 else 0.0
        q_heel = 0.0

    FOS_bearing = q_allowable / q_toe if (q_allowable and q_toe > 0) else 99.9

    return {
        "q_toe_kPa": round(q_toe, 1),
        "q_heel_kPa": round(q_heel, 1),
        "eccentricity_m": round(eccentricity, 3),
        "in_middle_third": in_middle_third,
        "FOS_bearing": round(FOS_bearing, 3),
    }


def analyze_cantilever_wall(geom: CantileverWallGeometry,
                            gamma_backfill: float,
                            phi_backfill: float,
                            c_backfill: float = 0.0,
                            phi_foundation: float = None,
                            c_foundation: float = 0.0,
                            q_allowable: float = None,
                            gamma_concrete: float = 24.0,
                            FOS_sliding: float = 1.5,
                            FOS_overturning: float = 2.0,
                            pressure_method: str = "rankine",
                            include_passive: bool = False,
                            gamma_foundation: float = None) -> CantileverWallResult:
    """Run complete cantilever wall stability analysis.

    Parameters
    ----------
    geom : CantileverWallGeometry
        Wall geometry.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    c_backfill : float, optional
        Backfill cohesion (kPa). Default 0.
    phi_foundation : float, optional
        Foundation soil friction angle. If None, uses phi_backfill.
    c_foundation : float, optional
        Foundation cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing pressure (kPa).
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 24.
    FOS_sliding : float, optional
        Required FOS for sliding. Default 1.5.
    FOS_overturning : float, optional
        Required FOS for overturning. Default 2.0.
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    include_passive : bool, optional
        If True, include passive resistance in sliding check. Default False.
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m³). If None, uses gamma_backfill.

    Returns
    -------
    CantileverWallResult
    """
    sliding = check_sliding(
        geom, gamma_backfill, phi_backfill, c_backfill,
        phi_foundation, c_foundation, gamma_concrete,
        FOS_sliding, pressure_method, include_passive, gamma_foundation
    )

    overturning = check_overturning(
        geom, gamma_backfill, phi_backfill, c_backfill,
        gamma_concrete, FOS_overturning, pressure_method
    )

    bearing = check_bearing(
        geom, gamma_backfill, phi_backfill, c_backfill,
        q_allowable, gamma_concrete, pressure_method
    )

    return CantileverWallResult(
        FOS_sliding=sliding["FOS_sliding"],
        FOS_overturning=overturning["FOS_overturning"],
        FOS_bearing=bearing["FOS_bearing"],
        passes_sliding=sliding["passes"],
        passes_overturning=overturning["passes"],
        passes_bearing=bearing.get("FOS_bearing", 0) >= 1.0 if q_allowable else True,
        q_toe=bearing["q_toe_kPa"],
        q_heel=bearing["q_heel_kPa"],
        eccentricity=bearing["eccentricity_m"],
        in_middle_third=bearing["in_middle_third"],
        wall_height=geom.wall_height,
        base_width=geom.base_width,
    )
