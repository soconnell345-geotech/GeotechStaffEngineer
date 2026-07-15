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


def _active_thrust(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   pressure_method: str = "rankine"):
    """Active thrust magnitude, location, inclination, and components.

    The total active thrust ``Pa = 0.5*Ka*gamma*H^2 + Ka*q*H - 2*c*sqrt(Ka)*H``
    acts at an inclination ``delta`` from horizontal that depends on the
    earth-pressure method (RW-1):

    - Rankine, level backfill: horizontal (``delta = 0``) — the classical case
      where treating Pa as purely horizontal is exact.
    - Rankine, sloped backfill: parallel to the backfill slope
      (``delta = beta``), per Rankine theory for a sloping surface (Das Ch. 13).
    - Coulomb: at the wall-friction angle ``delta = (2/3)*phi`` from the normal
      to the (vertical) pressure plane, i.e. (2/3)*phi above horizontal —
      matching the ``delta_wall`` used to compute the Coulomb Ka.

    The horizontal component ``Ph = Pa*cos(delta)`` drives sliding/overturning;
    the vertical component ``Pv = Pa*sin(delta)`` acts downward on the vertical
    pressure plane at the back of the heel (x = B from the toe) and is
    stabilizing — it adds to the weight resultant and to the resisting moment.

    Returns
    -------
    tuple
        (Ka, Pa, z_Pa, delta_deg, Ph, Pv)
    """
    H = geom.H_active

    if pressure_method == "coulomb":
        delta_deg = 2.0 / 3.0 * phi_backfill
        Ka = coulomb_Ka(phi_backfill, delta_deg,
                        beta_deg=geom.backfill_slope)
    elif geom.backfill_slope > 0:
        Ka = rankine_Ka_sloped(phi_backfill, geom.backfill_slope)
        delta_deg = geom.backfill_slope
    else:
        Ka = rankine_Ka(phi_backfill)
        delta_deg = 0.0

    Pa, z_Pa = horizontal_force_active(gamma_backfill, H, Ka,
                                       c_backfill, geom.surcharge)
    delta_rad = math.radians(delta_deg)
    Ph = Pa * math.cos(delta_rad)
    Pv = Pa * math.sin(delta_rad)
    return Ka, Pa, z_Pa, delta_deg, Ph, Pv


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

        # Sloped backfill: triangular soil wedge above the heel (RW-4).
        # The surface rises from the stem back face to heel*tan(beta) at the
        # back of the heel; its centroid is at 2/3 of the heel from the stem.
        if geom.backfill_slope > 0:
            rise = heel * math.tan(math.radians(geom.backfill_slope))
            W_soil_wedge = 0.5 * gamma_backfill * heel * rise
            x_soil_wedge = t_toe + t_stem_base + 2.0 * heel / 3.0
            weights.append((W_soil_wedge, x_soil_wedge))

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
                  gamma_foundation: float = None,
                  delta_base: float = None,
                  base_adhesion: float = None) -> Dict[str, Any]:
    """Check sliding stability.

    FOS_sliding = (V * tan(delta_b) + ca * B + Pp) / (Pa * cos(delta))

    By DEFAULT the base interface parameters are derived from the foundation
    soil: ``delta_b = (2/3)*phi_foundation`` and ``ca = (2/3)*c_foundation``
    (Das Ch. 13 / GEC-11 practice for a footing cast against soil with a
    reduced interface). Pass the FULL foundation soil strengths — do NOT
    pre-reduce them, or the 2/3 factor is applied twice (owner wall session
    2026-07-14: phi_foundation=22 intended as "delta_b = 22 deg" produced
    delta_b = 14.7 deg). To set the interface directly (e.g. delta_b = phi
    for cast-in-place concrete failing soil-on-soil, or a measured adhesion),
    use ``delta_base`` / ``base_adhesion``, which BYPASS the 2/3 factors.

    The active thrust is decomposed per the chosen earth-pressure method
    (RW-1): only the horizontal component Pa*cos(delta) drives sliding, and
    the vertical component Pa*sin(delta) (Coulomb wall friction or sloped
    Rankine) is added to V, increasing the frictional resistance.

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
    delta_base : float, optional
        Base interface friction angle (degrees), used DIRECTLY (no 2/3
        factor). If None (default), delta_b = (2/3)*phi_foundation.
    base_adhesion : float, optional
        Base adhesion ca (kPa), used DIRECTLY (no 2/3 factor). If None
        (default), ca = (2/3)*c_foundation.

    Returns
    -------
    dict
        FOS_sliding, driving_force, resisting_force, Pp_kN_per_m, passes,
        plus the interface values actually used (delta_base_deg,
        base_adhesion_kPa).
    """
    if phi_foundation is None:
        phi_foundation = phi_backfill
    if gamma_foundation is None:
        gamma_foundation = gamma_backfill

    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    Ka, Pa, _, delta_deg, Ph, Pv = _active_thrust(
        geom, gamma_backfill, phi_backfill, c_backfill, pressure_method
    )

    # Vertical force (sum of weights)
    weights = _compute_wall_weights(geom, gamma_backfill, gamma_concrete)
    V = sum(w for w, _ in weights)

    # Add vertical component of surcharge on heel
    V_surcharge = geom.surcharge * geom.heel_length
    V += V_surcharge

    # Vertical component of the active thrust (stabilizing)
    V += Pv

    # Base interface: default delta_b = 2/3 * phi_foundation and
    # ca = 2/3 * c_foundation; explicit overrides bypass the 2/3 factors.
    delta_b = delta_base if delta_base is not None else 2.0 / 3.0 * phi_foundation
    ca = base_adhesion if base_adhesion is not None else 2.0 / 3.0 * c_foundation

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

    driving = Ph

    FOS = resisting / driving if driving > 0 else 99.9

    return {
        "FOS_sliding": round(FOS, 3),
        "driving_force_kN_per_m": round(driving, 1),
        "resisting_force_kN_per_m": round(resisting, 1),
        "Pp_kN_per_m": round(Pp, 1),
        "Pa_kN_per_m": round(Pa, 1),
        "Pa_vertical_kN_per_m": round(Pv, 1),
        "thrust_inclination_deg": round(delta_deg, 2),
        "delta_base_deg": round(delta_b, 2),
        "base_adhesion_kPa": round(ca, 2),
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
    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    Ka, Pa, z_Pa, delta_deg, Ph, Pv = _active_thrust(
        geom, gamma_backfill, phi_backfill, c_backfill, pressure_method
    )

    # Overturning moment about toe: horizontal component only
    M_overturning = Ph * z_Pa

    # Stabilizing moments about toe
    weights = _compute_wall_weights(geom, gamma_backfill, gamma_concrete)
    M_stabilizing = sum(w * x for w, x in weights)

    # Surcharge on heel
    heel = geom.heel_length
    if heel > 0 and geom.surcharge > 0:
        W_q = geom.surcharge * heel
        x_q = geom.toe_length + geom.stem_thickness_base + heel / 2.0
        M_stabilizing += W_q * x_q

    # Vertical thrust component acts down at the pressure plane (back of
    # heel, x = B from toe) -> stabilizing moment
    M_stabilizing += Pv * B

    FOS = M_stabilizing / M_overturning if M_overturning > 0 else 99.9

    return {
        "FOS_overturning": round(FOS, 3),
        "stabilizing_moment_kNm_per_m": round(M_stabilizing, 1),
        "overturning_moment_kNm_per_m": round(M_overturning, 1),
        "Pa_kN_per_m": round(Pa, 1),
        "Pa_vertical_kN_per_m": round(Pv, 1),
        "thrust_inclination_deg": round(delta_deg, 2),
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
    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    Ka, Pa, z_Pa, delta_deg, Ph, Pv = _active_thrust(
        geom, gamma_backfill, phi_backfill, c_backfill, pressure_method
    )

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

    # Vertical thrust component (acts at the pressure plane, x = B)
    V += Pv
    M_stab += Pv * B

    M_over = Ph * z_Pa

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
        "Pa_kN_per_m": round(Pa, 1),
        "Pa_vertical_kN_per_m": round(Pv, 1),
        "thrust_inclination_deg": round(delta_deg, 2),
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
                            gamma_foundation: float = None,
                            delta_base: float = None,
                            base_adhesion: float = None) -> CantileverWallResult:
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
    delta_base : float, optional
        Base interface friction angle (degrees), used DIRECTLY in the sliding
        check (no 2/3 factor). Default None → (2/3)*phi_foundation.
    base_adhesion : float, optional
        Base adhesion ca (kPa), used DIRECTLY (no 2/3 factor). Default
        None → (2/3)*c_foundation.

    Returns
    -------
    CantileverWallResult
    """
    sliding = check_sliding(
        geom, gamma_backfill, phi_backfill, c_backfill,
        phi_foundation, c_foundation, gamma_concrete,
        FOS_sliding, pressure_method, include_passive, gamma_foundation,
        delta_base=delta_base, base_adhesion=base_adhesion,
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
