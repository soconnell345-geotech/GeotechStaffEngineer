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

#: How the sliding check counts the foundation soil's cohesion (GM-212),
#: by ``sliding_cohesion``: the rules the references print. GEC-6 and
#: EN 1997-1 keep the drained (friction) and undrained (adhesion)
#: resistances apart; UFC 3-220-20 and FHWA NHI-06-089 (after NAVFAC
#: 1986b) add them.
SLIDING_COHESION = ("drained", "undrained", "friction_plus_adhesion")
SLIDING_COHESION_BASIS = {
    "drained": (
        "friction alone, R = V tan delta_b, the cohesion not counted: GEC-6 "
        "Sec. 5.4 (p. 100) Eq. 5-38, FR = (W + Pv) tan delta 'for footings "
        "with planar bearing surfaces on frictional (drained) materials', "
        "and 'Because drained and undrained conditions will not occur "
        "simultaneously, Equations 5-38 and 5-39 should not be combined "
        "for use at the same time'; EN 1997-1 Sec. 6.5.3(8) and (10) "
        "(p. 64), drained Rd = V'd tan delta_d (Eq. 6.3a): 'Any effective "
        "cohesion c' should be neglected'"),
    "undrained": (
        "adhesion alone, R = ca B, no friction, ca the undrained shear "
        "strength cu (c_foundation): GEC-6 Sec. 5.4 (p. 100) Eq. 5-39, "
        "FR = ca Bf 'for footings with planar bearing surfaces on cohesive "
        "(undrained) materials', 'In practice, ca is typically taken as "
        "equal to the undrained shear strength (cohesion, cu) of the "
        "clay', and not combined with Eq. 5-38; EN 1997-1 Sec. 6.5.3(11) "
        "(p. 64), Rd = A cu;d (Eq. 6.4a), and (12)P (p. 65): 'If it is "
        "possible for water or air to reach the interface between a "
        "foundation and an undrained clay subgrade', Rd <= 0.4 Vd "
        "(Eq. 6.5), which this check does not apply"),
    "friction_plus_adhesion": (
        "friction and adhesion added, R = V tan delta_b + Ca Be: UFC "
        "3-220-20 Fig. 4-27 (p. 220), T = (W + Pv) tan delta + Ca Be, and "
        "Sec. 4-5.1.3 (p. 223): 'The resisting force is a combination of "
        "the frictional resistance at the base of the wall and the "
        "adhesion between the base of the wall and the soil foundation. "
        "For sliding resistance on fine-grained soils, the adhesion should "
        "be multiplied by the effective width of the base (Be) found using "
        "the uniform bearing pressure method' (Be = B - 2e), and 'Minimum "
        "FSL may also need to be increased when base adhesion is included "
        "in the resistance'; Ca from Table 4-1 (p. 184) by the undrained "
        "shear strength (c_foundation). FHWA NHI-06-089 Fig. 10-17 "
        "(p. 10-39, after NAVFAC 1986b) prints the same sum with the full "
        "base width B and ca from its Table 10-1 (p. 10-18)"),
}

#: UFC 3-220-20 Table 4-1 (p. 184), "Interface with Clayey Soils
#: (Undrained shear strength)": (undrained shear strength, adhesion Ca),
#: psf, the ends of each printed row's ranges (very soft 0-250 -> 0-250,
#: soft 250-500 -> 250-500, medium stiff 500-1000 -> 500-750, stiff
#: 1000-2000 -> 750-950, very stiff 2000-4000 -> 950-1300). FHWA
#: NHI-06-089 Table 10-1 (p. 10-18) prints the same rows in kPa.
UFC_TABLE_4_1_ADHESION_PSF = ((0.0, 0.0), (250.0, 250.0), (500.0, 500.0),
                              (1000.0, 750.0), (2000.0, 950.0),
                              (4000.0, 1300.0))
_KPA_PER_PSF = 0.04788026


def adhesion_ufc_table_4_1(su_kPa: float) -> float:
    """The base adhesion Ca (kPa) UFC 3-220-20 Table 4-1 (p. 184) gives for
    an undrained shear strength ``su_kPa``, read linearly within the row
    the strength falls in (:data:`UFC_TABLE_4_1_ADHESION_PSF`). The table
    stops at 4000 psf (191.5 kPa); above it there is no printed value and
    this raises."""
    su = su_kPa / _KPA_PER_PSF
    rows = UFC_TABLE_4_1_ADHESION_PSF
    if su < 0:
        raise ValueError(f"undrained shear strength {su_kPa} kPa < 0")
    if su > rows[-1][0] + 1e-6:
        raise ValueError(
            f"c_foundation {su_kPa:g} kPa is above UFC 3-220-20 Table 4-1 "
            f"(p. 184), which stops at very stiff fine-grained soil, "
            f"{rows[-1][0]:.0f} psf ({rows[-1][0] * _KPA_PER_PSF:.1f} kPa, "
            f"adhesion {rows[-1][1]:.0f} psf); give base_adhesion")
    for (s0, c0), (s1, c1) in zip(rows, rows[1:]):
        if su <= s1:
            return (c0 + (c1 - c0) * (su - s0) / (s1 - s0)) * _KPA_PER_PSF
    return rows[-1][1] * _KPA_PER_PSF


def default_sliding_cohesion(phi_foundation: Optional[float]) -> str:
    """The sliding check's rule for the foundation cohesion when
    ``sliding_cohesion`` is not given (GM-212): ``"undrained"`` for a
    foundation soil given with phi_foundation = 0 (an undrained strength,
    c_foundation = cu: GEC-6 Eq. 5-39 is for 'cohesive (undrained)
    materials'), else ``"drained"`` (GEC-6 Eq. 5-38, EN 1997-1
    6.5.3(10)). A phi_foundation not given takes the drained rule."""
    if phi_foundation is not None and phi_foundation == 0:
        return "undrained"
    return "drained"


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


def _refuse_unused_interface(rule: str, c_foundation: float,
                             delta_base: Optional[float],
                             base_adhesion: Optional[float]) -> None:
    """An interface input the chosen rule for the foundation cohesion does
    not count is refused with the rules that do (GM-212), never taken and
    left out of the resistance; and the undrained rule needs an undrained
    strength."""
    if rule == "drained" and base_adhesion:
        raise ValueError(
            f"base_adhesion {base_adhesion:g} kPa: the drained sliding "
            "resistance counts no adhesion (GEC-6 Eq. 5-38; EN 1997-1 "
            "6.5.3(10): 'Any effective cohesion c' should be neglected'). "
            "To count it give sliding_cohesion 'friction_plus_adhesion' "
            "(UFC 3-220-20 Fig. 4-27: V tan delta + Ca Be) or 'undrained' "
            "(GEC-6 Eq. 5-39: ca B alone, not combined with friction).")
    if rule == "undrained" and delta_base is not None:
        raise ValueError(
            f"delta_base {delta_base:g} deg: the undrained sliding "
            "resistance is the adhesion ca B alone (GEC-6 Eq. 5-39; "
            "'Equations 5-38 and 5-39 should not be combined'), with no "
            "base friction. Leave delta_base out, or give sliding_cohesion "
            "'friction_plus_adhesion' (UFC 3-220-20 Fig. 4-27) or "
            "'drained'.")
    if rule == "undrained" and base_adhesion is None \
            and not (c_foundation and c_foundation > 0):
        raise ValueError(
            "the undrained sliding resistance (sliding_cohesion "
            "'undrained', GEC-6 Eq. 5-39: ca B, ca typically the undrained "
            "shear strength cu) needs c_foundation = cu > 0, or "
            "base_adhesion. It is the rule taken when phi_foundation is 0; "
            "for a drained strength give phi_foundation > 0.")


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
                   dead_load_resists: bool = False,
                   sliding_cohesion: str = None) -> Dict[str, Any]:
    """The sliding check's terms, unrounded (GM-143): ``V`` and its parts
    (``V_weights``, ``V_surcharge`` the surcharge over the heel counted,
    ``Pv_counted`` the part of the thrust's vertical component ``Pv``
    counted), ``V_left_out`` (the surcharge over the heel and the live
    load's share of Pv not counted, GM-157), the base interface
    ``delta_b`` (deg, None when no friction is counted) and ``ca`` (kPa)
    with whether each was given, the rule for the foundation cohesion
    ``sliding_cohesion`` (GM-212, :data:`SLIDING_COHESION_BASIS`) with
    ``c_foundation``, where ca came from (``ca_from``), the width it acts
    on (``adhesion_width``: B, or Be = B - 2e for
    "friction_plus_adhesion", with ``eccentricity``) and the two parts
    ``R_friction`` and ``R_adhesion``; ``Kp``, ``D_passive`` and ``Pp``
    (None / 0 without passive; with a shear key Pp acts over the depth of
    the key only, from ``D_passive_top`` = the base thickness down to
    ``D_passive``, GM-210), ``resisting``, ``driving`` (= Ph) and ``FOS``
    (None with no driving force)."""
    if sliding_cohesion is None:
        sliding_cohesion = default_sliding_cohesion(phi_foundation)
    if sliding_cohesion not in SLIDING_COHESION:
        raise ValueError(f"sliding_cohesion must be one of "
                         f"{list(SLIDING_COHESION)}, got "
                         f"{sliding_cohesion!r}")
    _refuse_unused_interface(sliding_cohesion, c_foundation, delta_base,
                             base_adhesion)
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

    # Base interface (GM-212): the friction and the adhesion the printed
    # rule for the foundation cohesion counts. delta_b = 2/3 phi_foundation
    # by default (GM-173), delta_base used directly. The adhesion: none
    # (drained, GEC-6 Eq. 5-38; EN 1997-1 6.5.3(10)), cu on B alone
    # (undrained, GEC-6 Eq. 5-39), or UFC Table 4-1's Ca on Be added to the
    # friction (UFC 3-220-20 Fig. 4-27); base_adhesion used directly.
    delta_b = None
    R_friction = 0.0
    if sliding_cohesion != "undrained":
        delta_b = (delta_base if delta_base is not None
                   else 2.0 / 3.0 * phi_foundation)
        R_friction = V * math.tan(math.radians(delta_b))
    ca, ca_from, width, e = 0.0, None, None, None
    if sliding_cohesion == "undrained":
        ca, ca_from = ((base_adhesion, "base_adhesion")
                       if base_adhesion is not None
                       else (c_foundation, "c_foundation"))
        width = B
    elif sliding_cohesion == "friction_plus_adhesion":
        ca, ca_from = ((base_adhesion, "base_adhesion")
                       if base_adhesion is not None
                       else (adhesion_ufc_table_4_1(c_foundation),
                             "ufc_table_4_1"))
        # Be by the uniform bearing pressure method, on the loads this
        # check counts (Pp left out, as UFC Fig. 4-27's resultant is)
        M = (_sum(items, "M", "resists")
             - thrust["Ph"] * thrust["z_Pa"])
        e = B / 2.0 - (M / V if V > 0 else B / 2.0)
        width = max(B - 2.0 * abs(e), 0.0)
    R_adhesion = ca * width if width is not None else 0.0

    resisting = R_friction + R_adhesion

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
            "sliding_cohesion": sliding_cohesion,
            "c_foundation": c_foundation, "ca_from": ca_from,
            "adhesion_width": width, "eccentricity": e,
            "R_friction": R_friction, "R_adhesion": R_adhesion,
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
        "sliding_cohesion": t["sliding_cohesion"],
        "delta_base_deg": (None if t["delta_b"] is None
                           else round(t["delta_b"], 2)),
        "base_adhesion_kPa": round(t["ca"], 2),
        "adhesion_width_m": (None if t["adhesion_width"] is None
                             else round(t["adhesion_width"], 3)),
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
                  dead_load_resists: bool = False,
                  sliding_cohesion: str = None) -> Dict[str, Any]:
    """Check sliding stability.

    FOS_sliding = (R_friction + R_adhesion + Pp) / (Pa * cos(delta))

    V counts the surcharge over the heel only as ``live_load_resists`` and
    ``dead_load_resists`` say (default: neither, GM-157; the module
    docstring gives the basis); both surcharges' thrust drives it.

    The foundation soil's cohesion enters by the printed rule
    ``sliding_cohesion`` names (GM-212, :data:`SLIDING_COHESION_BASIS`):

    * ``"drained"``: R = V tan(delta_b), the cohesion not counted (GEC-6
      Eq. 5-38; EN 1997-1 6.5.3(10), "Any effective cohesion c' should be
      neglected"). The default, unless phi_foundation is 0.
    * ``"undrained"``: R = ca B alone, ca = c_foundation taken as the
      undrained shear strength cu (GEC-6 Eq. 5-39; "Equations 5-38 and
      5-39 should not be combined"). The default when phi_foundation is 0.
    * ``"friction_plus_adhesion"``: R = V tan(delta_b) + Ca Be, Ca from UFC
      3-220-20 Table 4-1 at c_foundation taken as the undrained shear
      strength, Be = B - 2e (UFC 3-220-20 Fig. 4-27, Sec. 4-5.1.3).

    The base friction is ``delta_b = (2/3)*phi_foundation`` by default
    (GM-173: :data:`BASE_FRICTION_BASIS`). Pass the FULL foundation soil
    strengths: do NOT pre-reduce them, or the 2/3 factor is applied twice
    (owner wall session 2026-07-14: phi_foundation=22 intended as "delta_b
    = 22 deg" produced delta_b = 14.7 deg). To set the interface directly
    (e.g. delta_b = phi for concrete cast against the ground, or a measured
    adhesion), use ``delta_base`` / ``base_adhesion``. An input the rule
    does not count is refused (base_adhesion with the drained rule,
    delta_base with the undrained one).

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
        Base adhesion ca (kPa), used DIRECTLY, with the "undrained" or
        "friction_plus_adhesion" rule. If None (default), cu
        (c_foundation) or UFC 3-220-20 Table 4-1's Ca at c_foundation.
    live_load_resists : bool, optional
        Count the live-load surcharge (``geom.surcharge``) over the heel,
        and its share of the thrust's vertical component, as resisting
        weight. Default False (:data:`LIVE_LOAD_BASIS`).
    dead_load_resists : bool, optional
        Count the dead-load surcharge (``geom.dead_load_surcharge``) over
        the heel as resisting weight. Default False
        (:data:`DEAD_LOAD_BASIS`).
    sliding_cohesion : str, optional
        "drained", "undrained" or "friction_plus_adhesion" (above). None
        (default): :func:`default_sliding_cohesion`.

    Returns
    -------
    dict
        FOS_sliding, driving_force, resisting_force, Pp_kN_per_m, passes,
        plus the interface values actually used (sliding_cohesion,
        delta_base_deg (None when no friction is counted),
        base_adhesion_kPa, adhesion_width_m). With no driving force (the active thrust is
        zero, e.g. cancelled by backfill cohesion) nothing pushes the wall:
        ``passes`` is True, ``no_driving_force`` is True and ``FOS_sliding``
        is None rather than a placeholder number.
    """
    return _sliding_dict(_sliding_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, phi_foundation,
        c_foundation, gamma_concrete, pressure_method, include_passive,
        gamma_foundation, delta_base, base_adhesion, live_load_resists,
        dead_load_resists, sliding_cohesion), FOS_required)


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
                            surcharge_in_bearing: bool = True,
                            sliding_cohesion: str = None
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
        Foundation cohesion (kPa). Default 0. The sliding check counts it
        by ``sliding_cohesion``: not at all (drained), or as the undrained
        shear strength cu.
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
        check (no 2/3 factor). Default None → (2/3)*phi_foundation. Refused
        with the undrained rule, which counts no friction.
    base_adhesion : float, optional
        Base adhesion ca (kPa), used DIRECTLY with the "undrained" or
        "friction_plus_adhesion" rule. Default None → cu (c_foundation),
        or UFC 3-220-20 Table 4-1's Ca at c_foundation. Refused with the
        drained rule, which counts no adhesion.
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
    sliding_cohesion : str, optional
        How the sliding check counts the foundation cohesion (GM-212,
        :func:`check_sliding`): "drained", "undrained" or
        "friction_plus_adhesion". None (default):
        :func:`default_sliding_cohesion` ("undrained" when phi_foundation
        is 0, else "drained").

    Returns
    -------
    CantileverWallResult
        With the terms each check used (GM-143): ``earth_pressure``,
        ``vertical_forces``, ``sliding_terms``, ``overturning_terms`` and
        ``bearing_terms``; ``eccentricity`` / ``in_middle_third`` are the
        eccentricity check's (on the overturning check's loads), q_toe /
        q_heel the bearing check's; and ``choices``: the rule for the
        foundation cohesion in sliding when there is any (GM-212), the base
        friction delta_b when friction is counted (GM-173) and how the
        surcharges over the heel were counted when there are any (GM-157).
    """
    rule_default = default_sliding_cohesion(phi_foundation)

    def slide(rule, delta_base=delta_base, base_adhesion=base_adhesion):
        return _sliding_terms(
            geom, gamma_backfill, phi_backfill, c_backfill,
            phi_foundation, c_foundation, gamma_concrete,
            pressure_method, include_passive, gamma_foundation,
            delta_base=delta_base, base_adhesion=base_adhesion,
            live_load_resists=live_load_resists,
            dead_load_resists=dead_load_resists, sliding_cohesion=rule)

    s_terms = slide(sliding_cohesion)
    sliding = _sliding_dict(s_terms, FOS_sliding)

    # The other printed rules on the same wall (GM-212), each without the
    # input it does not count, for the choice
    others = {}
    for rule in SLIDING_COHESION:
        if rule == s_terms["sliding_cohesion"]:
            continue
        if rule == "drained" and phi_foundation == 0 and delta_base is None:
            others[rule] = (
                "phi_foundation 0 is an undrained strength, so there is no "
                "base friction to count: GEC-6 Eq. 5-38 needs the "
                "foundation soil's drained friction angle (phi_foundation) "
                "or delta_base")
            continue
        try:
            t = slide(rule,
                      delta_base=None if rule == "undrained" else delta_base,
                      base_adhesion=(None if rule == "drained"
                                     else base_adhesion))
            others[rule] = t["FOS"]
        except ValueError as exc:
            others[rule] = str(exc)

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
        choices=sliding_cohesion_choice(s_terms, rule_default, others)
        + ([base_friction_choice(s_terms)]
           if s_terms["delta_b"] is not None else [])
        + heel_surcharge_choices(geom, live_load_resists,
                                 dead_load_resists, surcharge_in_bearing),
    )


#: Each rule for the foundation cohesion in sliding in a line (GM-212),
#: for the choice's alternatives; :data:`SLIDING_COHESION_BASIS` quotes them.
SLIDING_COHESION_SHORT = {
    "drained": ("R = V tan delta_b, the cohesion not counted (GEC-6 "
                "Eq. 5-38; EN 1997-1 6.5.3(10))"),
    "undrained": ("R = ca B alone, ca = c_foundation read as cu, no "
                  "friction (GEC-6 Eq. 5-39, not combined with Eq. 5-38)"),
    "friction_plus_adhesion": (
        "R = V tan delta_b + Ca Be (UFC 3-220-20 Fig. 4-27, Ca from its "
        "Table 4-1 at c_foundation as the undrained shear strength)"),
}


def sliding_cohesion_choice(terms: Dict[str, Any], rule_default: str,
                            others: Dict[str, Any]) -> list:
    """How the sliding check counted the foundation cohesion (GM-212), as
    a list of at most one entry of a result's ``choices``: none when there
    is no cohesion or adhesion to count and the drained rule was the one
    in force by default. ``terms`` are the sliding check's
    (:func:`_sliding_terms`), ``rule_default`` the rule taken when none is
    given (:func:`default_sliding_cohesion`), ``others`` the sliding FOS
    each other rule gives on the same wall (None: no driving force), or
    why it did not run. The basis of a default rule starts with its reason:
    the phi_foundation in force (phi_backfill when none is given)."""
    from geotech_common.choices import choice, how_given
    rule, c_f, ca = (terms["sliding_cohesion"], terms["c_foundation"],
                     terms["ca"])
    if rule == "drained" and not c_f and not terms["ca_given"] \
            and rule == rule_default:
        return []
    if rule == "drained":
        chosen = "drained: R = V tan delta_b" + (
            f", c_foundation = {c_f:g} kPa not counted" if c_f else "")
    else:
        source = {"base_adhesion": "base_adhesion",
                  "c_foundation": "c_foundation, as cu",
                  "ufc_table_4_1": (f"UFC 3-220-20 Table 4-1 at c_foundation "
                                    f"= {c_f:g} kPa")}[terms["ca_from"]]
        width = terms["adhesion_width"]
        chosen = (f"undrained: R = ca B = {ca:.1f} kPa x {width:.2f} m, ca "
                  f"from {source}, no friction" if rule == "undrained" else
                  f"friction_plus_adhesion: R = V tan delta_b + Ca Be, Ca = "
                  f"{ca:.1f} kPa from {source}, Be = B - 2e = {width:.2f} m")
    basis = SLIDING_COHESION_BASIS[rule]
    if rule == "undrained" and rule == rule_default:
        basis = ("phi_foundation 0 is an undrained strength, so "
                 "c_foundation is cu: " + basis)
    elif rule == "drained" and rule == rule_default:
        basis = (f"phi_foundation {terms['phi_foundation']:g} deg is taken "
                 "as a drained strength, so c_foundation is c': " + basis)

    def other(name):
        fos = others.get(name)
        if isinstance(fos, str):
            return (f"{name}: not run here ({fos}). "
                    + SLIDING_COHESION_SHORT[name])
        said = ("no driving force" if fos is None
                else f"FOS_sliding {fos:.3f}")
        return f"{name} ({said}): {SLIDING_COHESION_SHORT[name]}"

    return [choice(
        "foundation cohesion in the sliding resistance (sliding_cohesion)",
        chosen, how_given(rule, rule_default, name="sliding_cohesion"),
        basis, [other(name) for name in SLIDING_COHESION if name != rule])]


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
