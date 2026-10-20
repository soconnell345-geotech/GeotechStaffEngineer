"""
Cantilever retaining wall external stability analysis.

Checks sliding, overturning, eccentricity (middle third) and bearing for a
cantilever (or gravity) retaining wall with Rankine or Coulomb earth
pressures.

Wall height H (``CantileverWallGeometry.wall_height``) is the TOTAL height,
measured from the underside of the base slab (footing) to the top of the
stem: H = base_thickness + stem height. The backfill surface meets the stem
at its top, and the active thrust acts on the vertical plane through the back
of the heel over H (plus the rise of a sloping backfill over the heel). A
wall that retains 4.0 m above the top of a 0.5 m footing is H = 4.5 m.

The bearing check needs an allowable bearing pressure ``q_allowable``.
Without one it is NOT RUN: ``FOS_bearing`` and ``passes_bearing`` are None
and the result says "not checked"; the bearing pressures q_toe / q_heel and
the middle-third check are still reported.

A surcharge over the heel (GM-157). The geometry's ``surcharge`` is a live
load (traffic) and ``dead_load_surcharge`` a permanent one. Both thrusts
drive every check. Over the heel, by default neither is a resisting force
for sliding, overturning or the eccentricity (middle-third) check, and both
load the base in the bearing pressures; the live load's share of the
thrust's vertical component is left out of the resisting side with it.
``live_load_resists``, ``dead_load_resists`` and ``surcharge_in_bearing``
choose the other readings, and the result's ``choices`` say which was used
and why (:data:`LIVE_LOAD_BASIS` and the others). The eccentricity check
and the bearing pressures can therefore use two resultants (GEC-11 Vol I
p. 4-26: "eB = eccentricity for bearing calculation (not equal to
eccentricity check e)").

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    Das, B.M., Principles of Foundation Engineering, Ch 13
    AASHTO LRFD Section 11.6
    FHWA NHI-06-089, Soils and Foundations Reference Manual Vol II (2006),
        Sec. 10.5.3 (p. 10-41) and 10.5.5 (p. 10-43)
    UFC 3-220-20 (2025), App. B-1 (pp. 597 and 602)
    FHWA GEC-11 Vol I, FHWA-NHI-10-024 (2009), pp. 4-19, 4-22, 4-26
"""

import math
from typing import Dict, Any, List, Optional

from retaining_walls.geometry import CantileverWallGeometry
from retaining_walls.earth_pressure import (
    rankine_Ka, rankine_Ka_sloped, rankine_Kp, coulomb_Ka,
    active_force_components, horizontal_force_active,
    horizontal_force_passive,
)
from retaining_walls.results import CantileverWallResult

#: Whether the live-load surcharge resists sliding and overturning (GM-157),
#: by ``live_load_resists``: the basis of each reading.
LIVE_LOAD_BASIS = {
    False: ("FHWA NHI-06-089 Sec. 10.5.5 (p. 10-43): 'the live load "
            "surcharge is not considered as a stabilizing force over the "
            "heel of the wall when sliding resistance is being checked'; "
            "Sec. 10.5.3 (p. 10-41): for CIP semi-gravity walls the "
            "surcharge loads are 'conservatively neglected within the width "
            "of the base slab since they contribute to overturning and "
            "sliding resistance'; UFC 3-220-20 App. B-1.2 (p. 597): 'For "
            "overturning and sliding checks, it is assumed that the traffic "
            "pressure does not act over the wall heel'; GEC-11 Vol I p. 4-22 "
            "(AASHTO): external loads that increase sliding resistance are "
            "included 'only if those loads are permanent', so the vertical "
            "component of the live load's thrust is left out too"),
    True: ("counted: the free body with the live load over the heel "
           "whenever it acts behind the wall (the module's reading before "
           "GM-157); none of the references above counts a live load so"),
}

#: Whether a dead-load (permanent) surcharge over the heel resists sliding
#: and overturning (GM-157), by ``dead_load_resists``.
DEAD_LOAD_BASIS = {
    False: ("FHWA NHI-06-089 Sec. 10.5.3 (p. 10-41): for CIP semi-gravity "
            "walls the surcharge loads are 'generally assumed to be located "
            "behind the heel of the wall, and conservatively neglected "
            "within the width of the base slab since they contribute to "
            "overturning and sliding resistance'"),
    True: ("counted as a permanent load: GEC-11 Vol I p. 4-22 (AASHTO), "
           "'External loads that increase sliding resistance should only be "
           "included if those loads are permanent'; p. 4-4, the minimum load "
           "factor of a permanent load that increases stability"),
}

#: Whether the surcharge over the heel loads the base in the bearing
#: pressures (GM-157), by ``surcharge_in_bearing``.
BEARING_BASIS = {
    True: ("UFC 3-220-20 App. B-1.6 (p. 602): 'Unlike the checks of "
           "overturning and sliding, it is conservative to include the "
           "surcharge over the heel of the wall, if this is possible to "
           "occur, in the bearing capacity check'; GEC-11 Vol I p. 4-26: "
           "'the bearing check applies live load above both the reinforced "
           "zone and the retained backfill'"),
    False: ("left out of the base pressures, the surcharge taken as "
            "located behind the heel (FHWA NHI-06-089 Sec. 10.5.3, "
            "p. 10-41)"),
}

#: The sliding check's base friction delta_b (GM-173): the module's
#: default, 2/3 phi_f, and the reading for concrete cast against the
#: ground, delta_b = phi_f.
BASE_FRICTION_BASIS = {
    "two_thirds": (
        "a base not cast against the ground: EN 1997-1 Sec. 6.5.3(10) "
        "(p. 64) takes 2/3 phi'cv;d for smooth precast foundations; UFC "
        "3-220-20 Fig. 4-27 (p. 220) takes tan delta, the friction factor "
        "between soil and base, from Table 4-1 (p. 184), e.g. mass concrete "
        "on clean fine to medium sand 24-29 deg"),
    "phi_f": (
        "concrete cast against the ground: GEC-6 Sec. 5.4 (p. 99): 'For "
        "concrete cast against cohesionless or granular material, the "
        "coefficient of friction, tan delta, will be equal to the tangent "
        "of the friction angle (tan phi) for the soil supporting the "
        "footing'; EN 1997-1 Sec. 6.5.3(10) (p. 64): phi'cv;d for "
        "cast-in-situ concrete foundations"),
}


def _active_thrust_terms(geom: CantileverWallGeometry,
                         gamma_backfill: float, phi_backfill: float,
                         c_backfill: float = 0.0,
                         pressure_method: str = "rankine") -> Dict[str, Any]:
    """The active thrust and every term of it (GM-143), as
    :func:`_active_thrust` uses them: ``method``, ``Ka``, ``H_active``,
    the components ``Pa_earth``, ``Pa_surcharge``, ``Pa_cohesion`` and
    their total ``Pa`` (kN/m), its height ``z_Pa`` above the base (m),
    its inclination ``delta_deg`` and its components ``Ph``, ``Pv``.

    The surcharge component is K q H with q the live and the dead-load
    surcharges together (``q_live``, ``q_dead``, kPa). ``Pa_live`` is the
    live load's share of Pa and ``Pv_live`` its vertical component, which
    the sliding and overturning checks leave out by default (GM-157)."""
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

    q_live = geom.surcharge
    q_dead = geom.dead_load_surcharge
    parts = active_force_components(gamma_backfill, H, Ka, c_backfill,
                                    q_live + q_dead)
    Pa, z_Pa = parts["Pa"], parts["z_Pa"]
    delta_rad = math.radians(delta_deg)
    Ph = Pa * math.cos(delta_rad)
    Pv = Pa * math.sin(delta_rad)
    # The live load's share, never more than the thrust (cohesion may
    # take the total below it)
    Pa_live = min(Ka * q_live * H, Pa) if q_live > 0 else 0.0
    return {"method": pressure_method, "Ka": Ka, "H_active": H,
            "q_live": q_live, "q_dead": q_dead,
            "Pa_earth": parts["Pa_earth"],
            "Pa_surcharge": parts["Pa_surcharge"],
            "Pa_cohesion": parts["Pa_cohesion"],
            "Pa": Pa, "z_Pa": z_Pa, "delta_deg": delta_deg,
            "Ph": Ph, "Pv": Pv,
            "Pa_live": Pa_live, "Pv_live": Pa_live * math.sin(delta_rad)}


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
    t = _active_thrust_terms(geom, gamma_backfill, phi_backfill, c_backfill,
                             pressure_method)
    return t["Ka"], t["Pa"], t["z_Pa"], t["delta_deg"], t["Ph"], t["Pv"]


def _wall_weight_items(geom: CantileverWallGeometry,
                       gamma_backfill: float,
                       gamma_concrete: float = 24.0) -> List[Dict[str, Any]]:
    """The wall's weights as ``{"component", "W", "x"}`` (kN/m, and the
    arm from the toe in m), in the order :func:`_compute_wall_weights`
    sums them (GM-143)."""
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_base = geom.stem_thickness_base
    t_stem_top = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length

    items = []

    # Base slab
    W_base = gamma_concrete * B * t_base
    x_base = B / 2.0
    items.append({"component": "Base slab", "W": W_base, "x": x_base})

    # Stem (trapezoidal: wider at base)
    # Rectangular part (stem_top thickness)
    W_stem_rect = gamma_concrete * t_stem_top * h_stem
    x_stem_rect = t_toe + t_stem_top / 2.0
    items.append({"component": "Stem (rect)", "W": W_stem_rect,
                  "x": x_stem_rect})

    # Triangular part of stem taper
    if t_stem_base > t_stem_top:
        t_taper = t_stem_base - t_stem_top
        W_stem_tri = gamma_concrete * 0.5 * t_taper * h_stem
        # Centroid of triangle at 1/3 from wider base (back face)
        x_stem_tri = t_toe + t_stem_top + t_taper / 3.0
        items.append({"component": "Stem (taper)", "W": W_stem_tri,
                      "x": x_stem_tri})

    # Soil on heel (backfill above heel, behind stem)
    if heel > 0:
        W_soil_heel = gamma_backfill * heel * h_stem
        x_soil_heel = t_toe + t_stem_base + heel / 2.0
        items.append({"component": "Soil on heel", "W": W_soil_heel,
                      "x": x_soil_heel})

        # Sloped backfill: triangular soil wedge above the heel (RW-4).
        # The surface rises from the stem back face to heel*tan(beta) at the
        # back of the heel; its centroid is at 2/3 of the heel from the stem.
        if geom.backfill_slope > 0:
            rise = heel * math.tan(math.radians(geom.backfill_slope))
            W_soil_wedge = 0.5 * gamma_backfill * heel * rise
            x_soil_wedge = t_toe + t_stem_base + 2.0 * heel / 3.0
            items.append({"component": "Backfill wedge (sloped)",
                          "W": W_soil_wedge, "x": x_soil_wedge})

    # Soil on toe (if any, above toe in front of stem)
    # Typically minimal for cantilever walls, skip for simplicity

    return items


def _compute_wall_weights(geom: CantileverWallGeometry,
                          gamma_backfill: float,
                          gamma_concrete: float = 24.0):
    """Compute wall component weights and moment arms about toe.

    Returns list of (weight, arm) tuples for moment calculation.
    """
    return [(i["W"], i["x"]) for i in
            _wall_weight_items(geom, gamma_backfill, gamma_concrete)]


def _vertical_force_items(geom: CantileverWallGeometry,
                          gamma_backfill: float, gamma_concrete: float,
                          thrust: Dict[str, Any],
                          live_load_resists: bool = False,
                          dead_load_resists: bool = False,
                          surcharge_in_bearing: bool = True
                          ) -> List[Dict[str, Any]]:
    """Every vertical force on the base, in the order the checks sum them,
    as ``{"component", "kind", "W", "x", "M", "resists", "bearing"}``
    (GM-143): the weights, the surcharges on the heel and the vertical
    component of the active thrust at the pressure plane (x = B). ``kind``
    is "weight", "surcharge" or "thrust". ``resists``: counted in the
    sliding, overturning and eccentricity checks; ``bearing``: counted in
    the bearing pressures (GM-157, the choices in the module docstring)."""
    items = [{**i, "kind": "weight", "resists": True, "bearing": True}
             for i in _wall_weight_items(geom, gamma_backfill,
                                         gamma_concrete)]
    heel = geom.heel_length
    x_q = geom.toe_length + geom.stem_thickness_base + heel / 2.0
    for q, name, resists in (
            (geom.surcharge, "Live-load surcharge on heel",
             live_load_resists),
            (geom.dead_load_surcharge, "Dead-load surcharge on heel",
             dead_load_resists)):
        if heel > 0 and q > 0:
            items.append({"component": name, "kind": "surcharge",
                          "W": q * heel, "x": x_q, "resists": resists,
                          "bearing": surcharge_in_bearing})
    Pv, Pv_live = thrust["Pv"], thrust.get("Pv_live", 0.0)
    if Pv_live != 0.0:
        # the live load's share of P_v is a live load too (GEC-11 p. 4-22)
        if Pv - Pv_live != 0.0:
            items.append({"component": "Active thrust vert. comp. (P_v), "
                                       "less live load",
                          "kind": "thrust", "W": Pv - Pv_live,
                          "x": geom.base_width, "resists": True,
                          "bearing": True})
        items.append({"component": "Active thrust vert. comp. (P_v), "
                                   "live load",
                      "kind": "thrust", "W": Pv_live, "x": geom.base_width,
                      "resists": live_load_resists, "bearing": True})
    elif Pv != 0.0:
        items.append({"component": "Active thrust vert. comp. (P_v)",
                      "kind": "thrust", "W": Pv, "x": geom.base_width,
                      "resists": True, "bearing": True})
    for i in items:
        i["M"] = i["W"] * i["x"]
    return items


def _sum(items: List[Dict[str, Any]], key: str, flag: str) -> float:
    """The sum of ``key`` over the items a check counts (``flag``)."""
    return sum(i[key] for i in items if i[flag])


def _sliding_terms(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   phi_foundation: float = None,
                   c_foundation: float = 0.0,
                   gamma_concrete: float = 24.0,
                   pressure_method: str = "rankine",
                   include_passive: bool = False,
                   gamma_foundation: float = None,
                   delta_base: float = None,
                   base_adhesion: float = None,
                   live_load_resists: bool = False,
                   dead_load_resists: bool = False) -> Dict[str, Any]:
    """The sliding check's terms, unrounded (GM-143): ``V`` and its parts
    (``V_weights``, ``V_surcharge`` the surcharge over the heel counted,
    ``Pv_counted`` the part of the thrust's vertical component ``Pv``
    counted), ``V_left_out`` (the surcharge over the heel and the live
    load's share of Pv not counted, GM-157), the base interface
    ``delta_b`` (deg) and ``ca`` (kPa) with whether each was given,
    ``Kp``, ``D_passive`` and ``Pp`` (None / 0 without passive; with a
    shear key Pp acts over the depth of the key only, from
    ``D_passive_top`` = the base thickness down to ``D_passive``, GM-210),
    ``resisting``, ``driving`` (= Ph) and ``FOS`` (None with no driving
    force)."""
    if phi_foundation is None:
        phi_foundation = phi_backfill
    if gamma_foundation is None:
        gamma_foundation = gamma_backfill

    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method)
    Pa, delta_deg, Ph = thrust["Pa"], thrust["delta_deg"], thrust["Ph"]

    # The vertical forces the check counts: the weights, the surcharges
    # over the heel it counts (GM-157) and the thrust's vertical component
    # (stabilizing), the live load's share of it only when counted
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust, live_load_resists,
                                  dead_load_resists)
    V = _sum(items, "W", "resists")
    V_weights = sum(i["W"] for i in items if i["kind"] == "weight")
    V_surcharge = sum(i["W"] for i in items
                      if i["kind"] == "surcharge" and i["resists"])
    Pv_counted = sum(i["W"] for i in items
                     if i["kind"] == "thrust" and i["resists"])
    V_left_out = sum(i["W"] for i in items if not i["resists"])

    # Base interface: default delta_b = 2/3 * phi_foundation and
    # ca = 2/3 * c_foundation; explicit overrides bypass the 2/3 factors.
    delta_b = delta_base if delta_base is not None else 2.0 / 3.0 * phi_foundation
    ca = base_adhesion if base_adhesion is not None else 2.0 / 3.0 * c_foundation

    resisting = V * math.tan(math.radians(delta_b)) + ca * B

    # Passive resistance (optional), with the soil in front taken up to the
    # top of the base
    Pp = 0.0
    Kp = D = D_top = None
    if include_passive:
        Kp = rankine_Kp(phi_foundation)
        if geom.has_shear_key and geom.key_depth > 0:
            # Over the depth of the key only (GM-210): UFC 3-220-20 Sec.
            # 4-5.1.3 and Fig. 4-29 (pp. 223-224), "the passive pressure is
            # applied only along the depth of the key and not along the
            # entire burial depth of the wall"; GEC-6 Fig. 5-22 likewise.
            # Both figures draw the trapezoid on the vertical plane below
            # the toe (a-a1, the front of the soil the key carries), from
            # the underside of the base (depth t) to the level of the bottom
            # of the key (t + key_depth).
            D_top = geom.base_thickness
            D = D_top + geom.key_depth
        else:
            # On the embedment of the base (base_thickness)
            D_top, D = 0.0, geom.base_thickness
        Pp = 0.5 * Kp * gamma_foundation * (D ** 2 - D_top ** 2)
        resisting += Pp

    driving = Ph

    # No driving force: nothing pushes the wall, so it cannot slide. Report
    # that, not a placeholder FOS (GM-100).
    no_driving_force = driving <= 0
    FOS = None if no_driving_force else resisting / driving
    return {"V_weights": V_weights, "V_surcharge": V_surcharge,
            "Pv": thrust["Pv"], "Pv_counted": Pv_counted,
            "V": V, "V_left_out": V_left_out,
            "live_load_resists": live_load_resists,
            "dead_load_resists": dead_load_resists,
            "phi_foundation": phi_foundation,
            "delta_b": delta_b, "delta_b_given": delta_base is not None,
            "ca": ca, "ca_given": base_adhesion is not None,
            "Kp": Kp, "D_passive": D, "D_passive_top": D_top,
            "gamma_foundation": gamma_foundation,
            "Pp": Pp, "resisting": resisting, "driving": driving,
            "Pa": Pa, "delta_deg": delta_deg,
            "no_driving_force": no_driving_force, "FOS": FOS}


def _sliding_dict(t: Dict[str, Any], FOS_required: float) -> Dict[str, Any]:
    """:func:`check_sliding`'s dict from its terms."""
    FOS = t["FOS"]
    return {
        "FOS_sliding": None if FOS is None else round(FOS, 3),
        "no_driving_force": t["no_driving_force"],
        "driving_force_kN_per_m": round(t["driving"], 1),
        "resisting_force_kN_per_m": round(t["resisting"], 1),
        "Pp_kN_per_m": round(t["Pp"], 1),
        "Pa_kN_per_m": round(t["Pa"], 1),
        "Pa_vertical_kN_per_m": round(t["Pv"], 1),
        "thrust_inclination_deg": round(t["delta_deg"], 2),
        "delta_base_deg": round(t["delta_b"], 2),
        "base_adhesion_kPa": round(t["ca"], 2),
        "passes": True if FOS is None else FOS >= FOS_required,
    }


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
                  base_adhesion: float = None,
                  live_load_resists: bool = False,
                  dead_load_resists: bool = False) -> Dict[str, Any]:
    """Check sliding stability.

    FOS_sliding = (V * tan(delta_b) + ca * B + Pp) / (Pa * cos(delta))

    V counts the surcharge over the heel only as ``live_load_resists`` and
    ``dead_load_resists`` say (default: neither, GM-157; the module
    docstring gives the basis); both surcharges' thrust drives it.

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
    live_load_resists : bool, optional
        Count the live-load surcharge (``geom.surcharge``) over the heel,
        and its share of the thrust's vertical component, as resisting
        weight. Default False (:data:`LIVE_LOAD_BASIS`).
    dead_load_resists : bool, optional
        Count the dead-load surcharge (``geom.dead_load_surcharge``) over
        the heel as resisting weight. Default False
        (:data:`DEAD_LOAD_BASIS`).

    Returns
    -------
    dict
        FOS_sliding, driving_force, resisting_force, Pp_kN_per_m, passes,
        plus the interface values actually used (delta_base_deg,
        base_adhesion_kPa). With no driving force (the active thrust is
        zero, e.g. cancelled by backfill cohesion) nothing pushes the wall:
        ``passes`` is True, ``no_driving_force`` is True and ``FOS_sliding``
        is None rather than a placeholder number.
    """
    return _sliding_dict(_sliding_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, phi_foundation,
        c_foundation, gamma_concrete, pressure_method, include_passive,
        gamma_foundation, delta_base, base_adhesion, live_load_resists,
        dead_load_resists), FOS_required)


def _overturning_terms(geom: CantileverWallGeometry,
                       gamma_backfill: float, phi_backfill: float,
                       c_backfill: float = 0.0,
                       gamma_concrete: float = 24.0,
                       pressure_method: str = "rankine",
                       live_load_resists: bool = False,
                       dead_load_resists: bool = False) -> Dict[str, Any]:
    """The overturning check's terms, unrounded (GM-143):
    ``M_stabilizing``, ``M_overturning`` (= Ph z_Pa), ``Ph``, ``z_Pa``
    and ``FOS`` (None with no overturning moment); and the eccentricity
    check on the same loads (GM-157): ``V``, ``x_resultant``,
    ``eccentricity`` (B/2 - x_R) and ``in_middle_third``."""
    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method)
    Ph, z_Pa = thrust["Ph"], thrust["z_Pa"]

    # Overturning moment about toe: horizontal component only
    M_overturning = Ph * z_Pa

    # Stabilizing moments about the toe: the weights, the surcharges over
    # the heel the check counts (GM-157) and the thrust's vertical
    # component, down at the pressure plane (back of heel, x = B)
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust, live_load_resists,
                                  dead_load_resists)
    M_stabilizing = _sum(items, "M", "resists")

    # The eccentricity check (middle third), on the loads the overturning
    # check counts: UFC 3-220-20 B-1.4, NHI-06-089 Sec. 10.5.5 (the
    # eccentricity criterion stands for overturning), GEC-11 p. 4-22
    V = _sum(items, "W", "resists")
    x_resultant = (M_stabilizing - M_overturning) / V if V > 0 else B / 2.0
    eccentricity = B / 2.0 - x_resultant

    # No overturning moment: nothing tips the wall (GM-100, as for sliding).
    no_driving_force = M_overturning <= 0
    FOS = None if no_driving_force else M_stabilizing / M_overturning
    return {"M_stabilizing": M_stabilizing, "M_overturning": M_overturning,
            "Ph": Ph, "z_Pa": z_Pa, "Pa": thrust["Pa"], "Pv": thrust["Pv"],
            "delta_deg": thrust["delta_deg"],
            "no_driving_force": no_driving_force, "FOS": FOS,
            "V": V, "x_resultant": x_resultant,
            "eccentricity": eccentricity,
            "in_middle_third": abs(eccentricity) <= B / 6.0,
            "live_load_resists": live_load_resists,
            "dead_load_resists": dead_load_resists}


def check_overturning(geom: CantileverWallGeometry,
                      gamma_backfill: float, phi_backfill: float,
                      c_backfill: float = 0.0,
                      gamma_concrete: float = 24.0,
                      FOS_required: float = 2.0,
                      pressure_method: str = "rankine",
                      live_load_resists: bool = False,
                      dead_load_resists: bool = False) -> Dict[str, Any]:
    """Check overturning stability about the toe, and the eccentricity
    (middle-third) check on the same loads.

    FOS_OT = M_stabilizing / M_overturning

    M_stabilizing counts the surcharge over the heel only as
    ``live_load_resists`` and ``dead_load_resists`` say (default: neither,
    GM-157; see :func:`check_sliding`); both surcharges' thrust drives it.

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
    live_load_resists, dead_load_resists : bool, optional
        As in :func:`check_sliding`. Default False.

    Returns
    -------
    dict
        FOS_overturning, stabilizing_moment, overturning_moment, passes,
        and the eccentricity check: eccentricity_m, in_middle_third.
        With no overturning moment (zero active thrust) ``passes`` is True,
        ``no_driving_force`` is True and ``FOS_overturning`` is None.
    """
    return _overturning_dict(_overturning_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, gamma_concrete,
        pressure_method, live_load_resists, dead_load_resists),
        FOS_required)


def _overturning_dict(t: Dict[str, Any],
                      FOS_required: float) -> Dict[str, Any]:
    """:func:`check_overturning`'s dict from its terms."""
    FOS = t["FOS"]
    return {
        "FOS_overturning": None if FOS is None else round(FOS, 3),
        "no_driving_force": t["no_driving_force"],
        "stabilizing_moment_kNm_per_m": round(t["M_stabilizing"], 1),
        "overturning_moment_kNm_per_m": round(t["M_overturning"], 1),
        "Pa_kN_per_m": round(t["Pa"], 1),
        "Pa_vertical_kN_per_m": round(t["Pv"], 1),
        "thrust_inclination_deg": round(t["delta_deg"], 2),
        "passes": True if FOS is None else FOS >= FOS_required,
        "eccentricity_m": round(t["eccentricity"], 3),
        "in_middle_third": t["in_middle_third"],
    }


def _bearing_terms(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   q_allowable: float = None,
                   gamma_concrete: float = 24.0,
                   pressure_method: str = "rankine",
                   surcharge_in_bearing: bool = True) -> Dict[str, Any]:
    """The bearing check's terms, unrounded (GM-143): ``V``, ``M_stab``,
    ``M_over``, ``x_resultant``, ``eccentricity`` (the eccentricity for
    bearing, GM-157), ``in_middle_third``, ``distribution``
    ("trapezoidal", or "triangular_toe" / "triangular_heel" outside the
    middle third), ``B_eff`` (the triangle's width, None for a
    trapezoid), ``q_toe``, ``q_heel``, ``q_max``, ``FOS`` and
    ``passes``.

    The loads are the bearing check's (GM-157): the surcharges over the
    heel count as ``surcharge_in_bearing`` says (default: they do,
    :data:`BEARING_BASIS`), and the whole of the thrust's vertical
    component counts. So the eccentricity here can differ from the
    eccentricity check's (:func:`_overturning_terms`): GEC-11 Vol I
    p. 4-26, "eB = eccentricity for bearing calculation (not equal to
    eccentricity check e)"."""
    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method)
    Pa, z_Pa, delta_deg = thrust["Pa"], thrust["z_Pa"], thrust["delta_deg"]
    Ph, Pv = thrust["Ph"], thrust["Pv"]

    # Weights, the surcharges over the heel the check counts and the
    # vertical thrust component (at the pressure plane, x = B), with their
    # moments about the toe
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust,
                                  surcharge_in_bearing=surcharge_in_bearing)
    V = _sum(items, "W", "bearing")
    M_stab = _sum(items, "M", "bearing")

    M_over = Ph * z_Pa

    # Location of resultant from toe
    x_resultant = (M_stab - M_over) / V if V > 0 else B / 2.0
    eccentricity = B / 2.0 - x_resultant

    in_middle_third = abs(eccentricity) <= B / 6.0

    # Bearing pressures (Meyerhof: trapezoidal if in middle third)
    B_eff = None
    if in_middle_third:
        distribution = "trapezoidal"
        q_toe = V / B * (1.0 + 6.0 * eccentricity / B)
        q_heel = V / B * (1.0 - 6.0 * eccentricity / B)
    elif eccentricity > 0:
        # Resultant outside the middle third on the toe side: triangular
        # distribution under the toe, width 3 x (distance from the toe)
        distribution = "triangular_toe"
        B_eff = 3.0 * x_resultant
        q_toe = 2.0 * V / B_eff if B_eff > 0 else 0.0
        q_heel = 0.0
    else:
        # ... on the heel side: the triangle sits under the heel (GM-103;
        # it was drawn from the toe), width 3 x (distance from the heel)
        distribution = "triangular_heel"
        B_eff = 3.0 * (B - x_resultant)
        q_heel = 2.0 * V / B_eff if B_eff > 0 else 0.0
        q_toe = 0.0

    # Bearing check (GM-100). Without q_allowable it is NOT RUN: no FOS and
    # no verdict, never a placeholder reported as a pass. The demand is the
    # larger edge pressure (a resultant on the heel side of the centre puts
    # it under the heel).
    q_max = max(q_toe, q_heel)
    if q_allowable is None:
        FOS_bearing = None
        passes = None
    elif q_max > 0:
        FOS_bearing = q_allowable / q_max
        passes = FOS_bearing >= 1.0
    else:
        # The resultant falls outside the base (x_resultant <= 0): no
        # bearing pressure balances it and the wall tips over the toe.
        FOS_bearing = 0.0
        passes = False
    return {"V": V, "M_stab": M_stab, "M_over": M_over,
            "x_resultant": x_resultant, "eccentricity": eccentricity,
            "in_middle_third": in_middle_third,
            "distribution": distribution, "B_eff": B_eff,
            "q_toe": q_toe, "q_heel": q_heel, "q_max": q_max,
            "q_allowable": q_allowable, "FOS": FOS_bearing,
            "passes": passes, "Pa": Pa, "Pv": Pv, "delta_deg": delta_deg,
            "surcharge_in_bearing": surcharge_in_bearing}


def _bearing_dict(t: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`check_bearing`'s dict from its terms."""
    FOS_bearing = t["FOS"]
    return {
        "q_toe_kPa": round(t["q_toe"], 1),
        "q_heel_kPa": round(t["q_heel"], 1),
        "eccentricity_m": round(t["eccentricity"], 3),
        "in_middle_third": t["in_middle_third"],
        "FOS_bearing": None if FOS_bearing is None else round(FOS_bearing, 3),
        "bearing_checked": t["q_allowable"] is not None,
        "passes": t["passes"],
        "Pa_kN_per_m": round(t["Pa"], 1),
        "Pa_vertical_kN_per_m": round(t["Pv"], 1),
        "thrust_inclination_deg": round(t["delta_deg"], 2),
    }


def check_bearing(geom: CantileverWallGeometry,
                  gamma_backfill: float, phi_backfill: float,
                  c_backfill: float = 0.0,
                  q_allowable: float = None,
                  gamma_concrete: float = 24.0,
                  pressure_method: str = "rankine",
                  surcharge_in_bearing: bool = True) -> Dict[str, Any]:
    """Check bearing pressure, with the eccentricity for bearing.

    The surcharges over the heel load the base as ``surcharge_in_bearing``
    says (default: they do, GM-157). The eccentricity here is the one for
    bearing; the eccentricity (middle-third) check, on the loads the
    overturning check counts, is :func:`check_overturning`'s.

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
        Allowable bearing pressure (kPa). If None, the bearing check is NOT
        RUN (FOS_bearing and passes are None, bearing_checked is False); the
        bearing pressures and the middle-third check are still computed.
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 24.
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    surcharge_in_bearing : bool, optional
        Count the surcharges over the heel in the bearing pressures.
        Default True (:data:`BEARING_BASIS`).

    Returns
    -------
    dict
        q_toe, q_heel, eccentricity, in_middle_third, FOS_bearing
        (= q_allowable / the larger of q_toe and q_heel), bearing_checked,
        passes (None when not checked).
    """
    return _bearing_dict(_bearing_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, q_allowable,
        gamma_concrete, pressure_method, surcharge_in_bearing))


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
                            base_adhesion: float = None,
                            live_load_resists: bool = False,
                            dead_load_resists: bool = False,
                            surcharge_in_bearing: bool = True
                            ) -> CantileverWallResult:
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
        Allowable bearing pressure (kPa). Without it the bearing check is
        NOT RUN: the result's FOS_bearing and passes_bearing are None, its
        ``checks_not_run`` lists "bearing" and its summary says "not
        checked". q_toe / q_heel and the middle-third check are reported
        either way.
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
    live_load_resists : bool, optional
        Count the live-load surcharge (``geom.surcharge``) over the heel,
        and its share of the thrust's vertical component, as a resisting
        force in the sliding, overturning and eccentricity checks. Default
        False (:data:`LIVE_LOAD_BASIS`, GM-157).
    dead_load_resists : bool, optional
        The same for the dead-load surcharge (``geom.dead_load_surcharge``).
        Default False (:data:`DEAD_LOAD_BASIS`).
    surcharge_in_bearing : bool, optional
        Count the surcharges over the heel in the bearing pressures
        (q_toe, q_heel, FOS_bearing). Default True (:data:`BEARING_BASIS`).

    Returns
    -------
    CantileverWallResult
        With the terms each check used (GM-143): ``earth_pressure``,
        ``vertical_forces``, ``sliding_terms``, ``overturning_terms`` and
        ``bearing_terms``; ``eccentricity`` / ``in_middle_third`` are the
        eccentricity check's (on the overturning check's loads), q_toe /
        q_heel the bearing check's; and ``choices``: the base friction
        delta_b (GM-173) and how the surcharges over the heel were counted
        when there are any (GM-157).
    """
    s_terms = _sliding_terms(
        geom, gamma_backfill, phi_backfill, c_backfill,
        phi_foundation, c_foundation, gamma_concrete,
        pressure_method, include_passive, gamma_foundation,
        delta_base=delta_base, base_adhesion=base_adhesion,
        live_load_resists=live_load_resists,
        dead_load_resists=dead_load_resists,
    )
    sliding = _sliding_dict(s_terms, FOS_sliding)

    o_terms = _overturning_terms(
        geom, gamma_backfill, phi_backfill, c_backfill,
        gamma_concrete, pressure_method,
        live_load_resists=live_load_resists,
        dead_load_resists=dead_load_resists,
    )
    overturning = _overturning_dict(o_terms, FOS_overturning)

    b_terms = _bearing_terms(
        geom, gamma_backfill, phi_backfill, c_backfill,
        q_allowable, gamma_concrete, pressure_method,
        surcharge_in_bearing=surcharge_in_bearing,
    )
    bearing = _bearing_dict(b_terms)

    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method)

    return CantileverWallResult(
        FOS_sliding=sliding["FOS_sliding"],
        FOS_overturning=overturning["FOS_overturning"],
        FOS_bearing=bearing["FOS_bearing"],
        passes_sliding=sliding["passes"],
        passes_overturning=overturning["passes"],
        passes_bearing=bearing["passes"],
        q_toe=bearing["q_toe_kPa"],
        q_heel=bearing["q_heel_kPa"],
        eccentricity=overturning["eccentricity_m"],
        in_middle_third=overturning["in_middle_third"],
        wall_height=geom.wall_height,
        base_width=geom.base_width,
        FOS_sliding_required=FOS_sliding,
        FOS_overturning_required=FOS_overturning,
        q_allowable=q_allowable,
        earth_pressure=thrust,
        vertical_forces=_vertical_force_items(
            geom, gamma_backfill, gamma_concrete, thrust,
            live_load_resists, dead_load_resists, surcharge_in_bearing),
        sliding_terms=s_terms,
        overturning_terms=o_terms,
        bearing_terms=b_terms,
        choices=[base_friction_choice(s_terms)]
        + heel_surcharge_choices(geom, live_load_resists,
                                 dead_load_resists, surcharge_in_bearing),
    )


def base_friction_choice(terms: Dict[str, Any]) -> dict:
    """The sliding check's base friction delta_b (GM-173), as an entry of
    a result's ``choices`` (``geotech_common.choices``): 2/3 phi_f by
    default or ``delta_base`` as input; the other reading is the
    alternative. phi_f is ``phi_foundation``; that it is the backfill's
    when not given is the tool's declared absent case (GM-165, in
    ``defaults_used``), not repeated here. ``terms`` are the sliding
    check's (:func:`_sliding_terms`)."""
    from geotech_common.choices import DEFAULT, choice, how_given
    phi_f, delta_b = terms["phi_foundation"], terms["delta_b"]
    two_thirds = 2.0 / 3.0 * phi_f
    cast = (f"phi_f = {phi_f:g} deg (delta_base = {phi_f:g}): "
            + BASE_FRICTION_BASIS["phi_f"])
    if terms["delta_b_given"]:
        return choice(
            "base friction delta_b (sliding)",
            f"{delta_b:g} deg (delta_base)",
            how_given(delta_b, None, name="delta_base"),
            "the base interface friction angle as input, with no 2/3 "
            "factor",
            [f"2/3 phi_f = {two_thirds:.1f} deg, the module's default: "
             + BASE_FRICTION_BASIS["two_thirds"], cast])
    return choice(
        "base friction delta_b (sliding)",
        f"{delta_b:.1f} deg = 2/3 phi_f, phi_f = {phi_f:g} deg, the "
        "foundation soil's friction angle (phi_foundation)",
        DEFAULT, BASE_FRICTION_BASIS["two_thirds"], [cast])


def heel_surcharge_choices(geom: CantileverWallGeometry,
                           live_load_resists: bool = False,
                           dead_load_resists: bool = False,
                           surcharge_in_bearing: bool = True) -> list:
    """How :func:`analyze_cantilever_wall` counted the surcharges over the
    heel (GM-157), as a result's ``choices`` (``geotech_common.choices``):
    one entry per surcharge the wall carries, and one for bearing; none
    without a surcharge."""
    from geotech_common.choices import choice, how_given
    out = []
    counted = "counted as a resisting force"
    left_out = "not a resisting force"
    for q, what, flag, name, basis in (
            (geom.surcharge, "live-load surcharge over the heel, in "
             "sliding, overturning and the eccentricity check",
             live_load_resists, "live_load_resists", LIVE_LOAD_BASIS),
            (geom.dead_load_surcharge, "dead-load surcharge over the heel, "
             "in sliding, overturning and the eccentricity check",
             dead_load_resists, "dead_load_resists", DEAD_LOAD_BASIS)):
        if q > 0:
            chosen = counted if flag else left_out
            other = left_out if flag else counted
            out.append(choice(
                what, f"{chosen} ({name}={flag})",
                how_given(flag, False, name=name), basis[flag],
                [f"{other} ({name}={not flag}): {basis[not flag]}"]))
    if out:
        flag = surcharge_in_bearing
        out.append(choice(
            "surcharge over the heel, in the bearing pressures",
            f"{'counted' if flag else 'left out'} "
            f"(surcharge_in_bearing={flag})",
            how_given(flag, True, name="surcharge_in_bearing"),
            BEARING_BASIS[flag],
            [f"{'left out' if flag else 'counted'} "
             f"(surcharge_in_bearing={not flag}): {BEARING_BASIS[not flag]}"]))
    return out
