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
the middle-third check are still reported. On soil it compares the
equivalent uniform pressure q_eq = V / (B - 2e) with q_allowable, so
q_allowable is the allowable bearing pressure for the effective width B' =
B - 2e (GEC-6 Sec. 6.4.2, p. 119: "Bf in Equation 5-14 may be replaced by
B'f"); on rock the larger edge pressure (``bearing_pressure``, GM-288).

Checked against GEC-6 Ch. 5-7, UFC 3-220-20 Sec. 4-5.1 and App. B-1, and
FHWA NHI-06-089 Sec. 10.5 as printed (GM-288, module_work/gm288/CHECK.md):
the wall's weight counts the soil over a battered back face and over the
toe (``embedment_depth``); no negative active pressure in a cohesive
backfill (the tension zone); Coulomb's wall friction on the pressure plane
through the heel is the backfill slope; the passive resistance in front
acts over the embedment and counts cu under the undrained rule (GEC-6
Eq. 5-41 with a key); EN 1997-1's 0.4 V limit on request; FOS_sliding 2.0
with the passive resistance counted; the eccentricity limit B/4 and FOS
1.5 on rock. Without a water table the backfill is taken as drained
(NHI-06-089 Sec. 10.5.3; UFC 3-220-20 App. B-1.1).

A water table behind the wall (GM-292, module_work/gm292/CHECK.md).
``CantileverWallGeometry.water_table_depth`` puts a level water table that
depth below the top of the backfill at the stem. Below it the earth
pressure is on the buoyant unit weight gamma_b = gamma_sat - gamma_w and
the water pressure gamma_w z is added, horizontal, with a coefficient of 1
(UFC 3-220-20 Sec. 4-4.1 and Fig. 4-16(a), pp. 199-202; FHWA NHI-06-089
Fig. 10-12, p. 10-27); the backfill below it weighs gamma_sat
(``gamma_sat_backfill``, by default gamma_backfill) and the water lifts
the base. The water pressure under the base varies in a straight line
between the heads at the heel and the toe, as EM 1110-2-2104 para. E-12
(p. 115) prints a wall's uplift: gamma_w h_w at the heel to none at the
toe, with no water in front of the wall (``base_uplift`` "linear"), or
gamma_w h_w under the whole base, the same head at both ("uniform", as
UFC Fig. 5-40, p. 346, takes U = gamma_w h L). The uplift reduces V in
sliding; its moment about the toe is an overturning moment (the module's
assumption); the bearing pressures leave it out by default
(``uplift_in_bearing``; UFC Sec. 5-3.2, p. 292: "Its effect on the
bearing pressure can be conservatively ignored unless sliding or uplift
is a concern", on the safe side while e <= B/6). The choices state each,
with the other readings' numbers (:func:`water_choices`).

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
        Sec. 10.2 (pp. 10-14 to 10-20) and 10.5 (pp. 10-37 to 10-56)
    UFC 3-220-20 (2025), Sec. 4-2 (pp. 178-190), Sec. 4-5.1 (pp. 219-224)
        and App. B-1 (pp. 596-606)
    FHWA GEC-6, FHWA-SA-02-054 (2002), Sec. 5.4 (pp. 99-101) and 6.4
        (pp. 117-122)
    EN 1997-1:2004, Sec. 6.5.3 (pp. 64-65)
    FHWA GEC-11 Vol I, FHWA-NHI-10-024 (2009), pp. 4-19, 4-22, 4-26
"""

import math
from typing import Dict, Any, List, Optional

from retaining_walls.geometry import CantileverWallGeometry
from retaining_walls.earth_pressure import (
    rankine_Ka, rankine_Ka_sloped, rankine_Kp, coulomb_Ka,
    active_force_components, active_force_components_water,
    horizontal_force_active, horizontal_force_passive,
)
from retaining_walls.results import CantileverWallResult
from geotech_common.water import GAMMA_W

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


#: The Coulomb wall friction on the cantilever wall's pressure plane, the
#: vertical plane through the back of the heel (GM-288), for the choice.
COULOMB_DELTA_BASIS = (
    "FHWA NHI-06-089 p. 10-20: for cantilever walls the interface where "
    "the earth pressures are computed 'is within the retained soils along a "
    "vertical plane passing through the heel of the base slab. In such "
    "cases, there is soil-to-soil contact and the resultant may be oriented "
    "at the angle of mobilized friction ... it is generally conservative to "
    "assume that the earth pressure is parallel to the slope of the "
    "backfill', delta = beta, as its Example 10-3 (p. 10-53) takes it for "
    "Coulomb's Ka (Sec. 10.5.3, p. 10-37: Coulomb on that plane)")

#: How the water table behind the wall loads it (GM-292), for the choice.
WATER_PRESSURE_BASIS = (
    "UFC 3-220-20 Sec. 4-4.1 (pp. 199-200): 'Water applies a triangular "
    "pressure distribution equivalent to a soil with a unit weight of 62.4 "
    "pcf and an earth pressure coefficient of unity', and 'Instead of the "
    "total unit weight (gamma), the buoyant or effective unit weight "
    "(gamma_b) is used in the earth pressure equations for soils below the "
    "water table. The second effect is that an additional load, supplied "
    "by the water pressure, is applied to the wall' (Fig. 4-16(a), "
    "p. 202); FHWA NHI-06-089 Fig. 10-12 (p. 10-27), pa = Ka [gamma hw + "
    "gamma' zw] and pw = gamma_w zw, 'lateral water pressure coefficient = "
    "1.0', and Sec. 10.5.3 (p. 10-41): 'The wall must be designed for the "
    "full hydrostatic pressure when it is necessary to maintain the "
    "groundwater level behind the wall'")

#: The drained backfill, the reading without a water table (GM-292).
DRAINED_BASIS = (
    "FHWA NHI-06-089 Sec. 10.5.3 (p. 10-41): 'If adequate drainage measures "
    "are provided, the hydrostatic pressure due to groundwater behind the "
    "wall generally need not be considered. However, hydrostatic pressure "
    "must be considered for portions of the wall below the level of the "
    "weep holes unless a deeper drainage system is provided behind the base "
    "of the wall'")

#: The saturated unit weight below the water table when none is given
#: (GM-292).
GAMMA_SAT_BASIS = (
    "the backfill's unit weight (gamma_backfill) taken below the water "
    "table too, as UFC 3-220-20 Fig. 4-16(a) (p. 202) takes gamma = 125 "
    "pcf above it and gamma_b = 62.5 pcf below it, about gamma - gamma_w. "
    "Against a larger gamma_sat it errs on the safe side for sliding, "
    "overturning and the eccentricity where the heel carries soil below "
    "the water table (the heel's extra weight resists more than the extra "
    "thrust drives), and on the unsafe side for the bearing pressures (a "
    "few per cent) and for a wall with little or no heel; give "
    "gamma_sat_backfill for the saturated unit weight")

#: The water pressure under the base (GM-292), by ``base_uplift``.
BASE_UPLIFT = ("linear", "uniform")
BASE_UPLIFT_BASIS = {
    "linear": (
        "EM 1110-2-2104 (2023) para. E-12 (p. 115), for a wall: "
        "'Uplift across base varying uniformly from tailwater at heel to a "
        "value midway between tailwater and jump profile at toe', a "
        "straight line between the heads at the heel and the toe. Here the "
        "head behind the wall, gamma_w h_w at the heel, to none at the toe, "
        "with no water in front of the wall above the underside of the "
        "base. UFC 3-220-20 p. 203: 'Water pressures can add an uplift "
        "force on the base of retaining walls which can be detrimental to "
        "the wall stability'"),
    "uniform": (
        "the full head gamma_w h_w under the whole base, the same head at "
        "the heel and the toe (EM 1110-2-2104 para. E-12, p. 116: 'Uplift "
        "equal to tailwater'; UFC 3-220-20 Fig. 5-40, p. 346, U = gamma_w h "
        "L for a structure below the water table): the larger uplift"),
}

#: Where the uplift's moment about the toe goes in the overturning check
#: (GM-292), for the choice.
UPLIFT_MOMENT_BASIS = (
    "the module's assumption: no reference held prints it (UFC 3-220-20 "
    "Fig. 4-27, p. 220, FOT = W a / (PH b - PV f), has no uplift). Added to "
    "the overturning moment it gives the smaller FOS; the resultant and "
    "the eccentricity are the same either way")

#: Whether the uplift counts in the bearing pressures (GM-292), by
#: ``uplift_in_bearing``.
UPLIFT_BEARING_BASIS = {
    False: ("UFC 3-220-20 Sec. 5-3.2 (p. 292): 'Uplift is present on the "
            "base of the foundation when the groundwater table is above the "
            "bearing elevation. Its effect on the bearing pressure can be "
            "conservatively ignored unless sliding or uplift is a concern' "
            "(sliding, overturning and the eccentricity check count it "
            "here). On the safe side while the resultant is in the middle "
            "third (e <= B/6): counted, the linear uplift takes nothing off "
            "q_toe and lowers q_eq; past B/6, where the eccentricity check "
            "fails, counting it can raise them"),
    True: ("counted: the effective pressure under the base, the weights "
           "less the uplift, as EN 1997-1 Sec. 6.5.2.1(3)P (p. 63): 'Water "
           "pressures not caused by the foundation load shall be included "
           "as actions'"),
}


def _active_thrust_terms(geom: CantileverWallGeometry,
                         gamma_backfill: float, phi_backfill: float,
                         c_backfill: float = 0.0,
                         pressure_method: str = "rankine",
                         wall_friction_deg: Optional[float] = None,
                         gamma_sat_backfill: Optional[float] = None
                         ) -> Dict[str, Any]:
    """The active thrust and every term of it (GM-143), as
    :func:`_active_thrust` uses them: ``method``, ``Ka``, ``H_active``,
    the components ``Pa_earth``, ``Pa_surcharge``, ``Pa_cohesion`` and
    ``Pa_tension_zone`` (the negative pressure cohesion would give near the
    top, left out, GM-288) with ``z_tension`` its depth, their total ``Pa``
    (kN/m), its height ``z_Pa`` above the base (m), its inclination
    ``delta_deg`` (and ``delta_given``) and its components ``Ph``, ``Pv``.

    The surcharge component is K q H with q the live and the dead-load
    surcharges together (``q_live``, ``q_dead``, kPa). ``Pa_live`` is the
    live load's share of Pa and ``Pv_live`` its vertical component, which
    the sliding and overturning checks leave out by default (GM-157).

    Coulomb's Ka (UFC 3-220-20 Eq. 4-10, p. 186, theta = 0) takes the wall
    friction on the pressure plane ``wall_friction_deg``, by default the
    backfill slope beta (:data:`COULOMB_DELTA_BASIS`, GM-288; it was
    2/3 phi_backfill, which no reference read prints for soil on soil).

    With a water table above the underside of the base
    (``geom.water_table_depth``, GM-292) Pa is the effective earth
    pressure's, on the buoyant unit weight ``gamma_b`` = ``gamma_sat`` -
    gamma_w below the water table (``gamma_sat_backfill``, default
    gamma_backfill; :func:`active_force_components_water`), and the water
    pressure's thrust ``Pw`` = gamma_w h_w^2 / 2 acts horizontally at
    ``z_Pw`` = h_w / 3, ``h_w`` the water's height above the underside of
    the base and ``z_w`` the water table's depth below the top of the
    pressure plane; ``u_heel`` = gamma_w h_w is the water pressure at the
    back of the heel. Without one ``Pw`` is 0 and the others None."""
    H = geom.H_active
    if wall_friction_deg is not None:
        if pressure_method != "coulomb":
            raise ValueError(
                f"wall_friction_deg {wall_friction_deg:g}: the wall friction "
                "on the pressure plane enters Coulomb's Ka (pressure_method "
                "'coulomb'); Rankine's thrust is horizontal on level ground "
                "and parallel to a sloping one. Give pressure_method "
                "'coulomb', or leave it out.")
        if not 0.0 <= wall_friction_deg <= phi_backfill:
            raise ValueError(
                f"wall_friction_deg must be from 0 to phi_backfill "
                f"{phi_backfill:g} deg, got {wall_friction_deg:g}")

    if pressure_method == "coulomb":
        delta_deg = (geom.backfill_slope if wall_friction_deg is None
                     else wall_friction_deg)
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
    if gamma_sat_backfill is not None and geom.water_table_depth is None:
        raise ValueError(
            f"gamma_sat_backfill {gamma_sat_backfill:g} kN/m3 is the "
            "backfill's unit weight below a water table behind the wall; "
            "with no water_table_depth there is none. Give "
            "water_table_depth, or leave gamma_sat_backfill out.")
    h_w = geom.water_height
    gamma_sat = gamma_b = z_w = None
    if h_w > 0:
        # GM-292: the effective earth pressure on gamma_b below the water
        # table (UFC 3-220-20 p. 200 and Fig. 4-16(a); NHI Fig. 10-12)
        gamma_sat = (gamma_backfill if gamma_sat_backfill is None
                     else gamma_sat_backfill)
        gamma_b = gamma_sat - GAMMA_W
        if gamma_b <= 0:
            raise ValueError(
                f"the saturated unit weight {gamma_sat:g} kN/m3 "
                f"(gamma_sat_backfill, or gamma_backfill when it is not "
                f"given) must exceed the unit weight of water, "
                f"{GAMMA_W:g} kN/m3: the buoyant unit weight below the "
                f"water table is their difference")
        z_w = H - h_w
        parts = active_force_components_water(
            gamma_backfill, gamma_b, H, z_w, Ka, c_backfill,
            q_live + q_dead)
    else:
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
            "backfill_slope": geom.backfill_slope,
            "q_live": q_live, "q_dead": q_dead,
            "Pa_earth": parts["Pa_earth"],
            "Pa_surcharge": parts["Pa_surcharge"],
            "Pa_cohesion": parts["Pa_cohesion"],
            "Pa_tension_zone": parts["Pa_tension_zone"],
            "z_tension": parts["z_tension"],
            "Pa": Pa, "z_Pa": z_Pa, "delta_deg": delta_deg,
            "delta_given": (pressure_method == "coulomb"
                            and wall_friction_deg is not None),
            "Ph": Ph, "Pv": Pv,
            "Pa_live": Pa_live, "Pv_live": Pa_live * math.sin(delta_rad),
            # GM-292: the water table behind the wall
            "water_table_depth": geom.water_table_depth, "h_w": h_w,
            "z_w": z_w, "gamma_sat": gamma_sat, "gamma_b": gamma_b,
            "gamma_w": GAMMA_W, "Pw": 0.5 * GAMMA_W * h_w ** 2,
            "z_Pw": h_w / 3.0, "u_heel": GAMMA_W * h_w}


def _active_thrust(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   pressure_method: str = "rankine",
                   wall_friction_deg: Optional[float] = None):
    """Active thrust magnitude, location, inclination, and components.

    The total active thrust ``Pa = 0.5*Ka*gamma*H^2 + Ka*q*H - 2*c*sqrt(Ka)*H``
    acts at an inclination ``delta`` from horizontal that depends on the
    earth-pressure method (RW-1):

    - Rankine, level backfill: horizontal (``delta = 0``) — the classical case
      where treating Pa as purely horizontal is exact.
    - Rankine, sloped backfill: parallel to the backfill slope
      (``delta = beta``), per Rankine theory for a sloping surface (Das Ch. 13).
    - Coulomb: at the wall friction ``delta`` on the (vertical) pressure
      plane, by default the backfill slope (GM-288,
      :data:`COULOMB_DELTA_BASIS`) — the ``delta`` its Ka is computed with.

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
                             pressure_method, wall_friction_deg)
    return t["Ka"], t["Pa"], t["z_Pa"], t["delta_deg"], t["Ph"], t["Pv"]


#: The unit weight of concrete the checks take by default (kN/m3, GM-288):
#: 150 pcf, as UFC 3-220-20 App. B-1.3 (p. 599) and FHWA NHI-06-089
#: Example 10-3 (p. 10-53) assume it (GEC-6's examples take 23.5 kN/m3,
#: e.g. App. B Example 3). It was 24, which no reference read prints.
GAMMA_CONCRETE = 23.6


def _wall_weight_items(geom: CantileverWallGeometry,
                       gamma_backfill: float,
                       gamma_concrete: float = GAMMA_CONCRETE,
                       gamma_front: Optional[float] = None
                       ) -> List[Dict[str, Any]]:
    """The wall's weights as ``{"component", "W", "x"}`` (kN/m, and the
    arm from the toe in m), in the order :func:`_compute_wall_weights`
    sums them (GM-143).

    The weight is the wall's and the soil's above the base (UFC 3-220-20
    Fig. 4-27, p. 220: "includes weight of soil above footing for
    cantilever and counterfort walls"; FHWA NHI-06-089 Fig. 10-17,
    p. 10-39): the backfill over the heel, over a battered back face of the
    stem (GM-288; it was left out) and, with the base below the ground in
    front (``geom.Df`` above the base thickness), the soil over the toe at
    ``gamma_front`` (the foundation soil's, default ``gamma_backfill``; UFC
    App. B-1.3, p. 599, W_ff)."""
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_base = geom.stem_thickness_base
    t_stem_top = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length
    t_taper = max(t_stem_base - t_stem_top, 0.0)
    front = geom.stem_batter == "front"

    items = []

    # Base slab
    W_base = gamma_concrete * B * t_base
    x_base = B / 2.0
    items.append({"component": "Base slab", "W": W_base, "x": x_base})

    # Stem (trapezoidal: wider at base), its rectangular part (the top
    # thickness) against the vertical face: the front one with the back
    # battered, the back one with the front battered (UFC App. B-1.3,
    # Table B-1: W_s1 at bt + (t1 - t2) + t2/2, W_s2 at bt + 2 (t1 - t2)/3)
    W_stem_rect = gamma_concrete * t_stem_top * h_stem
    x_stem_rect = t_toe + (t_taper if front else 0.0) + t_stem_top / 2.0
    items.append({"component": "Stem (rect)", "W": W_stem_rect,
                  "x": x_stem_rect})

    # Triangular part of stem taper, its centroid a third of the taper
    # from its wide side (next to the rectangular part)
    if t_taper > 0:
        W_stem_tri = gamma_concrete * 0.5 * t_taper * h_stem
        x_stem_tri = (t_toe + 2.0 * t_taper / 3.0 if front
                      else t_toe + t_stem_top + t_taper / 3.0)
        items.append({"component": "Stem (taper)", "W": W_stem_tri,
                      "x": x_stem_tri})
        if not front:
            # The backfill over the battered back (GM-288, GM-235): the
            # triangle between the back face and the vertical through the
            # back of the stem's base, its centroid 2/3 of the taper from
            # the stem's top back corner
            items.append({"component": "Soil over battered stem back",
                          "W": gamma_backfill * 0.5 * t_taper * h_stem,
                          "x": t_toe + t_stem_top + 2.0 * t_taper / 3.0})

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

    # Soil over the toe, with the base below the ground in front (GM-288):
    # (Df - t) bt gamma, at bt/2 (UFC App. B-1.3, p. 599, W_ff; the sliver
    # in front of a battered face is left out, as there)
    over_toe = geom.Df - t_base
    if over_toe > 0 and t_toe > 0:
        g_front = gamma_backfill if gamma_front is None else gamma_front
        items.append({"component": "Soil over toe",
                      "W": g_front * over_toe * t_toe, "x": t_toe / 2.0})

    return items


def _compute_wall_weights(geom: CantileverWallGeometry,
                          gamma_backfill: float,
                          gamma_concrete: float = GAMMA_CONCRETE):
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
                          surcharge_in_bearing: bool = True,
                          gamma_front: Optional[float] = None,
                          base_uplift: Optional[str] = None,
                          uplift_in_bearing: bool = False
                          ) -> List[Dict[str, Any]]:
    """Every vertical force on the base, in the order the checks sum them,
    as ``{"component", "kind", "W", "x", "M", "resists", "bearing"}``
    (GM-143): the weights, the surcharges on the heel and the vertical
    component of the active thrust at the pressure plane (x = B). ``kind``
    is "weight", "surcharge", "thrust" or "uplift". ``resists``: counted in
    the sliding, overturning and eccentricity checks; ``bearing``: counted
    in the bearing pressures (GM-157, the choices in the module docstring).
    ``gamma_front`` weighs the soil over the toe (:func:`_wall_weight_items`).

    With a water table above the underside of the base (``thrust["h_w"]``,
    GM-292) the backfill below it weighs gamma_sat: the excess over
    gamma_backfill is a weight of its own where it is not zero, and the
    water pressure under the base, by ``base_uplift`` (None: "linear",
    :data:`BASE_UPLIFT_BASIS`), is an "uplift" item with W = -U, counted
    in the bearing pressures as ``uplift_in_bearing`` says."""
    items = [{**i, "kind": "weight", "resists": True, "bearing": True}
             for i in _wall_weight_items(geom, gamma_backfill,
                                         gamma_concrete, gamma_front)]
    h_w = thrust.get("h_w", 0.0) or 0.0
    if h_w > 0:
        items += _saturated_items(geom, thrust["gamma_sat"] - gamma_backfill,
                                  h_w)
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
    if h_w > 0:
        items.append(_uplift_item(geom, thrust["u_heel"], base_uplift,
                                  uplift_in_bearing))
    for i in items:
        i["M"] = i["W"] * i["x"]
    return items


def _saturated_items(geom: CantileverWallGeometry, excess: float,
                     h_w: float) -> List[Dict[str, Any]]:
    """The backfill's extra weight below the water table (GM-292),
    (gamma_sat - gamma_backfill) = ``excess`` over the part of the soil on
    the heel and over a battered back below it, ``h_w`` above the underside
    of the base; none where the excess or that part is zero. The sloped
    wedge over the heel is above the top of the stem, so above the water
    table; in front of the wall the module takes no water."""
    s_w = min(h_w, geom.wall_height) - geom.base_thickness
    if excess == 0 or s_w <= 0:
        return []
    x_back = geom.toe_length + geom.stem_thickness_base
    items = []
    t_taper = max(geom.stem_thickness_base - geom.stem_thickness_top, 0.0)
    h_stem = geom.stem_height
    if t_taper > 0 and geom.stem_batter != "front" and h_stem > 0:
        # the battered back's soil triangle widens upward from the back of
        # the stem's base; below the water table, a triangle s_w high
        w = t_taper * s_w / h_stem
        items.append({"component": "Soil over battered stem back, below "
                                   "the water table (gamma_sat - gamma)",
                      "W": excess * 0.5 * w * s_w, "x": x_back - w / 3.0})
    heel = geom.heel_length
    if heel > 0:
        items.append({"component": "Soil on heel, below the water table "
                                   "(gamma_sat - gamma)",
                      "W": excess * heel * s_w, "x": x_back + heel / 2.0})
    return [{**i, "kind": "weight", "resists": True, "bearing": True}
            for i in items]


def _uplift_item(geom: CantileverWallGeometry, u_heel: float,
                 base_uplift: Optional[str],
                 uplift_in_bearing: bool) -> Dict[str, Any]:
    """The water pressure under the base as a vertical force (GM-292): W =
    -U, by ``base_uplift`` (:data:`BASE_UPLIFT_BASIS`): "linear" (None),
    from ``u_heel`` at the heel to 0 at the toe, U = u_heel B / 2 at 2B/3
    from the toe; "uniform", U = u_heel B at B/2."""
    rule = "linear" if base_uplift is None else base_uplift
    if rule not in BASE_UPLIFT:
        raise ValueError(f"base_uplift must be one of {list(BASE_UPLIFT)}, "
                         f"got {base_uplift!r}")
    B = geom.base_width
    if rule == "linear":
        U, x, name = 0.5 * u_heel * B, 2.0 * B / 3.0, (
            "Uplift on the base (linear, heel to toe)")
    else:
        U, x, name = u_heel * B, B / 2.0, "Uplift on the base (uniform)"
    return {"component": name, "kind": "uplift", "W": -U, "x": x,
            "base_uplift": rule, "resists": True,
            "bearing": bool(uplift_in_bearing)}


def _uplift(items: List[Dict[str, Any]]) -> tuple:
    """(U, its moment about the toe), both >= 0, of the uplift items."""
    up = [i for i in items if i["kind"] == "uplift"]
    return -sum(i["W"] for i in up), -sum(i["M"] for i in up)


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


#: The theories of the passive resistance in front of the wall (GM-242):
#: Rankine's Kp, no wall friction (the default), or the log-spiral Kp of
#: ``geotech_common.log_spiral`` with wall friction.
PASSIVE_THEORIES = ("rankine", "log_spiral")

#: Why the passive resistance in front of the wall takes Rankine's Kp by
#: default (GM-242), for the choice.
TOE_PASSIVE_BASIS = (
    "UFC 3-220-20 Sec. 4-5.1 (p. 219): 'the passive pressure on the front "
    "of the wall is typically ignored when assessing the stability', and "
    "with it considered Fig. 4-27 (p. 220) and Sec. 4-5.1.3 (p. 223) ask "
    "FSL >= 2.0; with a shear key the passive pressure acts over the key's "
    "depth only (Fig. 4-29, p. 224). Neither names the theory of its Kp. "
    "By default the module takes Rankine's, with no wall friction, the "
    "smaller; passive_theory 'log_spiral' takes the log-spiral Kp with "
    "wall friction. GEC-6 Sec. 5.4 (pp. 99-101): 'the design of most "
    "shallow foundations conservatively ignores passive resistance', and "
    "'keys constructed in granular soils are not very efficient in "
    "developing passive pressures'")

#: Whether the passive resistance in front of the wall counts the
#: foundation soil's cohesion (GM-288, GM-235), by the rule the sliding
#: check reads c_foundation with.
TOE_PASSIVE_COHESION_BASIS = {
    True: ("the undrained rule reads c_foundation as the undrained shear "
           "strength cu, so the passive pressure is Kp sigma_v + 2 cu "
           "sqrt(Kp): UFC 3-220-20 Eq. 4-7 (p. 178) and p. 181, 'The same "
           "equations can be expressed for undrained or total stress "
           "analyses ... total stress strength parameters (c, phi, or su) "
           "are used'; FHWA NHI-06-089 Eq. 10-11 (p. 10-14); GEC-6 "
           "Eq. 5-41 (p. 101) counts Pp in front of a key in cohesive soil"),
    False: ("c_foundation not counted in the passive pressure: UFC 3-220-20 "
            "p. 179, 'the changes to the earth pressures caused by c' are "
            "usually neglected for fine-grained soils' (the drained rule "
            "neglects it at the base too, EN 1997-1 6.5.3(10))"),
}

#: The required factor of safety against sliding when none is given
#: (GM-288), without and with the passive resistance in front of the wall.
FOS_SLIDING = {False: 1.5, True: 2.0}
FOS_SLIDING_BASIS = {
    False: ("GEC-6 Eq. 5-40 (p. 100) and Sec. 6.4.3 (p. 119); UFC 3-220-20 "
            "Fig. 4-27 (p. 220) and Sec. 4-5.1.3 (p. 223); FHWA NHI-06-089 "
            "Fig. 10-17 (p. 10-39): FS >= 1.5"),
    True: ("UFC 3-220-20 Fig. 4-27 (p. 220), 'If passive pressure is "
           "considered: FSL = (T + PP)/PH >= 2.0', and Sec. 4-5.1.3 "
           "(p. 223): 'If passive resistance is included, the minimum FSL "
           "should be increased to 2.0'"),
}

#: The required factor of safety against overturning when none is given
#: (GM-288), for a wall on soil and on rock.
FOS_OVERTURNING = {False: 2.0, True: 1.5}
FOS_OVERTURNING_BASIS = {
    False: ("FHWA NHI-06-089 Sec. 10.5.5 (p. 10-44): the eccentricity "
            "criterion 'essentially requires that the safety factor of the "
            "wall against overturning is approximately of 2.0 for soils'; "
            "GEC-6 Sec. 7.3 (p. 124): 'greater than about 2.0 for permanent "
            "dead loads'. UFC 3-220-20 Fig. 4-27 (p. 220) and Sec. 4-5.1.2 "
            "(p. 223) ask FOT >= 1.5"),
    True: ("FHWA NHI-06-089 Sec. 10.5.5 (p. 10-44): 'approximately ... 1.5 "
           "for rocks'; UFC 3-220-20 Sec. 4-5.1.2 (p. 223): FOT >= 1.5"),
}

#: The eccentricity limit, as a fraction of the base width, for a wall on
#: soil and on rock (GM-288).
ECCENTRICITY_LIMIT = {False: 1.0 / 6.0, True: 1.0 / 4.0}
ECCENTRICITY_BASIS = {
    False: ("e <= B/6, the resultant in the middle third: GEC-6 Sec. 6.4.1 "
            "(p. 119); UFC 3-220-20 Sec. 4-5.1.2 (p. 223); FHWA NHI-06-089 "
            "Fig. 10-17 (p. 10-39)"),
    True: ("e <= B/4, the middle half, for a wall founded on rock: GEC-6 "
           "Sec. 6.4.1 (p. 119); UFC 3-220-20 Sec. 4-5.1.2 (p. 223); FHWA "
           "NHI-06-089 Fig. 10-17 (p. 10-39)"),
}

#: The bearing pressure compared with q_allowable (GM-288).
BEARING_PRESSURES = ("uniform", "linear")
BEARING_PRESSURE_BASIS = {
    "uniform": (
        "the equivalent uniform pressure V / B' over the effective width "
        "B' = B - 2e: GEC-6 Eq. 6-6 (p. 118) and Sec. 6.4.2 (p. 119), "
        "'sizing the footing such that the applied bearing stress over the "
        "effective footing area, A', assuming a rectangular pressure "
        "distribution, is less than the allowable bearing pressure'; FHWA "
        "NHI-06-089 Fig. 10-17 (p. 10-39), 'Use uniform stress, qeq, for "
        "soils' and Sec. 10.5.4.1 (p. 10-41); UFC 3-220-20 Eq. 4-27 "
        "(p. 221), Sec. 4-5.1.4 (p. 224), 'A uniform pressure distribution "
        "can be used to determine qmax if the soil below the wall is "
        "relatively ductile', and App. B-1.6 (p. 602), q = R/(2 x0)"),
    "linear": (
        "the larger edge pressure of the trapezoid or triangle (q_toe, "
        "q_heel): UFC 3-220-20 Eqs. 4-23 and 4-25 (p. 221) and Sec. 4-5.1.4 "
        "(p. 224), 'For more brittle materials, such as rock and sensitive "
        "clays, the triangular or trapezoidal pressure distribution should "
        "be used to find qmax'; FHWA NHI-06-089 Fig. 10-17 (p. 10-39), "
        "'use trapezoidal distribution with qmax and qmin for rocks and "
        "structural analysis', as its Example 10-3 (p. 10-55) divides qult "
        "by qmax; GEC-6 Eqs. 6-9 to 6-13 (pp. 121-122) give it for "
        "structural design"),
}

#: EN 1997-1's limit on the undrained sliding resistance (GM-288, GM-235).
UNDRAINED_GAP_BASIS = (
    "EN 1997-1 Sec. 6.5.3(12)P (p. 65): 'If it is possible for water or air "
    "to reach the interface between a foundation and an undrained clay "
    "subgrade, the following check shall be made: Rd <= 0,4 Vd' (Eq. 6.5); "
    "(13): it 'may only be disregarded if the formation of a gap between "
    "the foundation and the ground will be prevented by suction in areas "
    "where there is no positive bearing pressure'. GEC-6 and UFC 3-220-20 "
    "print no such limit")


def default_FOS_sliding(include_passive: bool) -> float:
    """The required factor of safety against sliding when none is given
    (GM-288, :data:`FOS_SLIDING_BASIS`): 1.5, or 2.0 with the passive
    resistance in front of the wall counted."""
    return FOS_SLIDING[bool(include_passive)]


def default_FOS_overturning(founded_on_rock: bool = False) -> float:
    """The required factor of safety against overturning when none is
    given (GM-288, :data:`FOS_OVERTURNING_BASIS`): 2.0 on soil, 1.5 on
    rock."""
    return FOS_OVERTURNING[bool(founded_on_rock)]


def default_bearing_pressure(founded_on_rock: bool = False) -> str:
    """The bearing pressure compared with q_allowable when none is named
    (GM-288, :data:`BEARING_PRESSURE_BASIS`): "uniform" on soil, "linear"
    on rock."""
    return "linear" if founded_on_rock else "uniform"


def toe_passive_Kp(phi_f: float, passive_theory: str = "rankine",
                   log_spiral_source: Optional[str] = None,
                   passive_wall_friction_deg: Optional[float] = None
                   ) -> Dict[str, Any]:
    """The passive coefficient in front of the wall (GM-242): ``Kp``,
    ``delta`` (deg), ``delta_given`` and ``route``. "rankine": Rankine's Kp,
    no wall friction. "log_spiral": the log-spiral Kp of
    ``log_spiral_source`` (default UFC 3-220-20's for a vertical face and
    level ground, Eq. 4-12, p. 191) at ``passive_wall_friction_deg``; left
    out, the source's own (``geotech_common.log_spiral.level_front_Kp``)."""
    from geotech_common import log_spiral as _ls
    if passive_theory == "rankine":
        return {"Kp": rankine_Kp(phi_f), "delta": 0.0,
                "delta_given": False, "route": None}
    r = _ls.level_front_Kp(phi_f, passive_wall_friction_deg,
                           log_spiral_source or _ls.DEFAULT_SOURCE)
    return {"Kp": r["Kp"], "delta": r["delta_deg"],
            "delta_given": r["delta_given"], "route": r["route"]}


def _refuse_unused_passive(include_passive: bool, passive_theory: str,
                           log_spiral_source: Optional[str],
                           passive_wall_friction_deg: Optional[float],
                           phi_f: float) -> None:
    """The passive theory's inputs checked, and refused where the sliding
    check does not use them (GM-242)."""
    from geotech_common import log_spiral as _ls
    if passive_theory not in PASSIVE_THEORIES:
        raise ValueError(f"passive_theory must be one of "
                         f"{list(PASSIVE_THEORIES)}, got {passive_theory!r}")
    if log_spiral_source is not None \
            and log_spiral_source not in _ls.SOURCES:
        raise ValueError(f"log_spiral_source must be one of "
                         f"{list(_ls.SOURCES)}, got {log_spiral_source!r}")
    if passive_theory == "log_spiral" and not include_passive:
        raise ValueError(
            "passive_theory 'log_spiral' is the theory of the passive "
            "resistance in front of the wall, which the sliding check counts "
            "only with include_passive. Give include_passive true, or leave "
            "passive_theory out.")
    if passive_theory != "log_spiral":
        unused = [n for n, v in (
            ("log_spiral_source", log_spiral_source),
            ("passive_wall_friction_deg", passive_wall_friction_deg))
            if v is not None]
        if unused:
            raise ValueError(
                f"{' and '.join(unused)}: for the log-spiral passive Kp in "
                "front of the wall (passive_theory 'log_spiral'); Rankine's "
                "Kp takes no wall friction. Give passive_theory "
                "'log_spiral', or leave "
                f"{'it' if len(unused) == 1 else 'them'} out.")
        return
    if passive_wall_friction_deg is not None and not (
            0.0 <= passive_wall_friction_deg <= max(phi_f, 0.0)):
        raise ValueError(
            f"passive_wall_friction_deg must be in [0, phi_foundation = "
            f"{phi_f:g}] deg, got {passive_wall_friction_deg:g}")
    if passive_wall_friction_deg is None and phi_f > 0 \
            and log_spiral_source in _ls.LEVEL_FRONT_NEEDS_DELTA:
        raise ValueError(_ls.LEVEL_FRONT_NEEDS_DELTA[
            log_spiral_source].format(name="passive_wall_friction_deg"))


def _sliding_terms(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   phi_foundation: float = None,
                   c_foundation: float = 0.0,
                   gamma_concrete: float = GAMMA_CONCRETE,
                   pressure_method: str = "rankine",
                   include_passive: bool = False,
                   gamma_foundation: float = None,
                   delta_base: float = None,
                   base_adhesion: float = None,
                   live_load_resists: bool = False,
                   dead_load_resists: bool = False,
                   sliding_cohesion: str = None,
                   passive_theory: str = "rankine",
                   log_spiral_source: str = None,
                   passive_wall_friction_deg: float = None,
                   wall_friction_deg: float = None,
                   interface_gap_possible: bool = False,
                   gamma_sat_backfill: float = None,
                   base_uplift: str = None
                   ) -> Dict[str, Any]:
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
    the key only, from ``D_passive_top`` = the embedment Df down to
    ``D_passive``, GM-210; how Kp was taken, ``passive_theory``,
    ``passive_delta``, ``passive_delta_given``, ``log_spiral_source`` and
    ``log_spiral_route``, GM-242), ``resisting``, ``driving`` (= Ph) and
    ``FOS`` (None with no driving force).

    GM-288: the passive resistance acts over the embedment ``geom.Df``
    (from the ground in front to the underside of the base; with a key,
    from there to the bottom of the key) and counts the cohesion
    ``Pp_cohesion`` = 2 cu sqrt(Kp) h under the undrained rule
    (:data:`TOE_PASSIVE_COHESION_BASIS`); with a key under the undrained
    rule GEC-6 Eq. 5-41 is ca (B - a1b) + cu a1b + Pp, a1b from the toe to
    the key's front face, taken here as 0 (ca B + Pp: the same when ca =
    cu, the default, and smaller when base_adhesion is below cu); with
    ``interface_gap_possible`` the undrained base resistance is held to
    0.4 V (EN 1997-1 Eq. 6.5, ``gap_limited``).

    GM-292: with a water table the water's thrust ``Pw`` drives the wall
    with Ph (``driving`` = Ph + Pw) and the uplift ``U`` is taken off V."""
    if sliding_cohesion is None:
        sliding_cohesion = default_sliding_cohesion(phi_foundation)
    if sliding_cohesion not in SLIDING_COHESION:
        raise ValueError(f"sliding_cohesion must be one of "
                         f"{list(SLIDING_COHESION)}, got "
                         f"{sliding_cohesion!r}")
    _refuse_unused_interface(sliding_cohesion, c_foundation, delta_base,
                             base_adhesion)
    if interface_gap_possible and sliding_cohesion != "undrained":
        raise ValueError(
            "interface_gap_possible is EN 1997-1 6.5.3(12)P's limit on the "
            "undrained sliding resistance of a clay (Rd <= 0.4 Vd, "
            f"Eq. 6.5); the rule in force is {sliding_cohesion!r}. Give "
            "sliding_cohesion 'undrained' (or phi_foundation 0), or leave "
            "it out.")
    if phi_foundation is None:
        phi_foundation = phi_backfill
    if gamma_foundation is None:
        gamma_foundation = gamma_backfill
    _refuse_unused_passive(include_passive, passive_theory,
                           log_spiral_source, passive_wall_friction_deg,
                           phi_foundation)

    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method,
                                  wall_friction_deg, gamma_sat_backfill)
    Pa, delta_deg, Ph = thrust["Pa"], thrust["delta_deg"], thrust["Ph"]

    # The vertical forces the check counts: the weights, the surcharges
    # over the heel it counts (GM-157) and the thrust's vertical component
    # (stabilizing), the live load's share of it only when counted; less
    # the uplift under a water table (GM-292)
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust, live_load_resists,
                                  dead_load_resists,
                                  gamma_front=gamma_foundation,
                                  base_uplift=base_uplift)
    U = _uplift(items)[0]
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
        # (V > 0 but for an uplift that floats the wall, GM-292)
        R_friction = max(V, 0.0) * math.tan(math.radians(delta_b))
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
             - thrust["Ph"] * thrust["z_Pa"]
             - thrust["Pw"] * thrust["z_Pw"])
        e = B / 2.0 - (M / V if V > 0 else B / 2.0)
        width = max(B - 2.0 * abs(e), 0.0)
    R_adhesion = ca * width if width is not None else 0.0

    # EN 1997-1 6.5.3(12)P (GM-288): water or air at the interface of an
    # undrained clay holds the base resistance to 0.4 V (Eq. 6.5)
    gap_limited = False
    if interface_gap_possible and sliding_cohesion == "undrained" \
            and R_adhesion > 0.4 * V:
        R_adhesion, gap_limited = 0.4 * V, True

    resisting = R_friction + R_adhesion

    # Passive resistance (optional), from the ground in front of the wall,
    # Df above the underside of the base (GM-288; the top of the base when
    # embedment_depth is not given)
    Pp = Pp_cohesion = 0.0
    Kp = D = D_top = None
    passive_cohesion = False
    toe = {"delta": None, "delta_given": False, "route": None}
    if include_passive:
        # Rankine's Kp by default, or the log-spiral Kp (GM-242)
        toe = toe_passive_Kp(phi_foundation, passive_theory,
                             log_spiral_source, passive_wall_friction_deg)
        Kp = toe["Kp"]
        if geom.has_shear_key and geom.key_depth > 0:
            # Over the depth of the key only (GM-210): UFC 3-220-20 Sec.
            # 4-5.1.3 and Fig. 4-29 (pp. 223-224), "the passive pressure is
            # applied only along the depth of the key and not along the
            # entire burial depth of the wall"; GEC-6 Fig. 5-22 likewise.
            # Both figures draw the trapezoid on the vertical plane below
            # the toe (a-a1, the front of the soil the key carries), from
            # the underside of the base (depth Df) to the level of the
            # bottom of the key (Df + key_depth).
            D_top = geom.Df
            D = D_top + geom.key_depth
        else:
            # On the embedment of the base, Df
            D_top, D = 0.0, geom.Df
        Pp = 0.5 * Kp * gamma_foundation * (D ** 2 - D_top ** 2)
        # The cohesion, under the undrained rule (c_foundation = cu): the
        # 2 c sqrt(Kp) of UFC 3-220-20 Eq. 4-7 over the height (GM-288,
        # GM-235; :data:`TOE_PASSIVE_COHESION_BASIS`)
        passive_cohesion = sliding_cohesion == "undrained"
        if passive_cohesion and c_foundation:
            Pp_cohesion = 2.0 * c_foundation * math.sqrt(Kp) * (D - D_top)
            Pp += Pp_cohesion
        resisting += Pp

    # with the water's thrust under a water table (GM-292)
    driving = Ph + thrust["Pw"]

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
            "interface_gap_possible": interface_gap_possible,
            "gap_limited": gap_limited,
            "R_friction": R_friction, "R_adhesion": R_adhesion,
            "Kp": Kp, "D_passive": D, "D_passive_top": D_top,
            "Pp_cohesion": Pp_cohesion, "passive_cohesion": passive_cohesion,
            "passive_theory": passive_theory if include_passive else None,
            "passive_delta": toe["delta"],
            "passive_delta_given": toe["delta_given"],
            "log_spiral_source": (
                (log_spiral_source or "ufc_3_220_20")
                if include_passive and passive_theory == "log_spiral"
                else None),
            "log_spiral_route": toe["route"],
            "gamma_foundation": gamma_foundation,
            "Pp": Pp, "resisting": resisting, "driving": driving,
            "Pa": Pa, "delta_deg": delta_deg,
            "Pw": thrust["Pw"], "U": U, "h_w": thrust["h_w"],
            "no_driving_force": no_driving_force, "FOS": FOS}


def _sliding_dict(t: Dict[str, Any], FOS_required: float) -> Dict[str, Any]:
    """:func:`check_sliding`'s dict from its terms."""
    FOS = t["FOS"]
    out = {
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
    if t.get("h_w"):
        # GM-292: the water's thrust in the driving force, and the uplift
        out["water_thrust_kN_per_m"] = round(t["Pw"], 1)
        out["uplift_kN_per_m"] = round(t["U"], 1)
    return out


def check_sliding(geom: CantileverWallGeometry,
                  gamma_backfill: float, phi_backfill: float,
                  c_backfill: float = 0.0,
                  phi_foundation: float = None,
                  c_foundation: float = 0.0,
                  gamma_concrete: float = GAMMA_CONCRETE,
                  FOS_required: float = None,
                  pressure_method: str = "rankine",
                  include_passive: bool = False,
                  gamma_foundation: float = None,
                  delta_base: float = None,
                  base_adhesion: float = None,
                  live_load_resists: bool = False,
                  dead_load_resists: bool = False,
                  sliding_cohesion: str = None,
                  passive_theory: str = "rankine",
                  log_spiral_source: str = None,
                  passive_wall_friction_deg: float = None,
                  wall_friction_deg: float = None,
                  interface_gap_possible: bool = False,
                  gamma_sat_backfill: float = None,
                  base_uplift: str = None
                  ) -> Dict[str, Any]:
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
        Concrete unit weight (kN/m³). Default 23.6 (:data:`GAMMA_CONCRETE`).
    FOS_required : float, optional
        Required FOS. Default None: 1.5, or 2.0 with include_passive
        (:func:`default_FOS_sliding`, GM-288).
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    include_passive : bool, optional
        If True, include passive resistance Pp in front of wall, over the
        embedment ``geom.Df`` (with a shear key, over the key's depth
        below it), with the cohesion under the undrained rule (GM-288).
        Default False.
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m³). If None, uses gamma_backfill.
    passive_theory : str, optional
        With include_passive, the passive coefficient (GM-242): "rankine"
        (default, no wall friction; :data:`TOE_PASSIVE_BASIS`) or
        "log_spiral" (:func:`toe_passive_Kp`).
    log_spiral_source : str, optional
        With passive_theory "log_spiral", the printed source of the Kp
        (``geotech_common.log_spiral.SOURCES``). None: UFC 3-220-20's for a
        vertical face and level ground, Eq. 4-12 (p. 191).
    passive_wall_friction_deg : float, optional
        With passive_theory "log_spiral", the wall friction delta in front
        of the wall (deg), 0 to phi_foundation. None: the source's own,
        delta/phi' = 0.66 (UFC 3-220-20 p. 191), or phi' with the
        Caquot-Kerisel (1948) chart.
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
    wall_friction_deg : float, optional
        With pressure_method "coulomb", the wall friction on the pressure
        plane (deg). None: the backfill slope (GM-288,
        :data:`COULOMB_DELTA_BASIS`).
    interface_gap_possible : bool, optional
        Under the undrained rule, hold the base resistance to 0.4 V where
        water or air can reach the interface (EN 1997-1 Eq. 6.5,
        :data:`UNDRAINED_GAP_BASIS`). Default False.
    gamma_sat_backfill : float, optional
        With ``geom.water_table_depth``, the backfill's unit weight below
        the water table (kN/m3). None: gamma_backfill (GM-292,
        :data:`GAMMA_SAT_BASIS`).
    base_uplift : str, optional
        With ``geom.water_table_depth``, the water pressure under the base:
        "linear" (None) or "uniform" (:data:`BASE_UPLIFT_BASIS`).

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
        dead_load_resists, sliding_cohesion, passive_theory,
        log_spiral_source, passive_wall_friction_deg, wall_friction_deg,
        interface_gap_possible, gamma_sat_backfill, base_uplift),
        default_FOS_sliding(include_passive) if FOS_required is None
        else FOS_required)


def _overturning_terms(geom: CantileverWallGeometry,
                       gamma_backfill: float, phi_backfill: float,
                       c_backfill: float = 0.0,
                       gamma_concrete: float = GAMMA_CONCRETE,
                       pressure_method: str = "rankine",
                       live_load_resists: bool = False,
                       dead_load_resists: bool = False,
                       wall_friction_deg: float = None,
                       gamma_foundation: float = None,
                       founded_on_rock: bool = False,
                       gamma_sat_backfill: float = None,
                       base_uplift: str = None) -> Dict[str, Any]:
    """The overturning check's terms, unrounded (GM-143):
    ``M_stabilizing``, ``M_overturning`` (= Ph z_Pa), ``Ph``, ``z_Pa``
    and ``FOS`` (None with no overturning moment); and the eccentricity
    check on the same loads (GM-157): ``V``, ``x_resultant``,
    ``eccentricity`` (B/2 - x_R), ``in_middle_third``, and (GM-288)
    ``eccentricity_limit`` (B/6 on soil, B/4 on rock,
    :data:`ECCENTRICITY_BASIS`) with ``within_eccentricity_limit``.
    ``gamma_foundation`` weighs the soil over the toe.

    GM-292: with a water table the overturning moment adds the water's
    thrust, Pw z_Pw, and the uplift's moment about the toe ``M_uplift``
    (U x_U, :data:`UPLIFT_MOMENT_BASIS`); V is net of the uplift ``U``."""
    B = geom.base_width

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method,
                                  wall_friction_deg, gamma_sat_backfill)
    Ph, z_Pa = thrust["Ph"], thrust["z_Pa"]

    # Stabilizing moments about the toe: the weights, the surcharges over
    # the heel the check counts (GM-157) and the thrust's vertical
    # component, down at the pressure plane (back of heel, x = B)
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust, live_load_resists,
                                  dead_load_resists,
                                  gamma_front=gamma_foundation,
                                  base_uplift=base_uplift)
    M_stabilizing = sum(i["M"] for i in items
                        if i["resists"] and i["kind"] != "uplift")
    U, M_uplift = _uplift(items)

    # Overturning moment about toe: horizontal component only; under a
    # water table the water's thrust and the uplift's moment too (GM-292)
    M_overturning = Ph * z_Pa + thrust["Pw"] * thrust["z_Pw"] + M_uplift

    # The eccentricity check (middle third), on the loads the overturning
    # check counts: UFC 3-220-20 B-1.4, NHI-06-089 Sec. 10.5.5 (the
    # eccentricity criterion stands for overturning), GEC-11 p. 4-22
    V = _sum(items, "W", "resists")
    x_resultant = (M_stabilizing - M_overturning) / V if V > 0 else B / 2.0
    eccentricity = B / 2.0 - x_resultant

    # No overturning moment: nothing tips the wall (GM-100, as for sliding).
    no_driving_force = M_overturning <= 0
    FOS = None if no_driving_force else M_stabilizing / M_overturning
    limit = ECCENTRICITY_LIMIT[bool(founded_on_rock)] * B
    return {"M_stabilizing": M_stabilizing, "M_overturning": M_overturning,
            "Ph": Ph, "z_Pa": z_Pa, "Pa": thrust["Pa"], "Pv": thrust["Pv"],
            "delta_deg": thrust["delta_deg"],
            "no_driving_force": no_driving_force, "FOS": FOS,
            "V": V, "x_resultant": x_resultant,
            "eccentricity": eccentricity,
            "in_middle_third": abs(eccentricity) <= B / 6.0,
            "founded_on_rock": bool(founded_on_rock),
            "eccentricity_limit": limit,
            "within_eccentricity_limit": abs(eccentricity) <= limit + 1e-12,
            "live_load_resists": live_load_resists,
            "dead_load_resists": dead_load_resists,
            "Pw": thrust["Pw"], "z_Pw": thrust["z_Pw"], "U": U,
            "M_uplift": M_uplift, "h_w": thrust["h_w"]}


def check_overturning(geom: CantileverWallGeometry,
                      gamma_backfill: float, phi_backfill: float,
                      c_backfill: float = 0.0,
                      gamma_concrete: float = GAMMA_CONCRETE,
                      FOS_required: float = None,
                      pressure_method: str = "rankine",
                      live_load_resists: bool = False,
                      dead_load_resists: bool = False,
                      wall_friction_deg: float = None,
                      gamma_foundation: float = None,
                      founded_on_rock: bool = False,
                      gamma_sat_backfill: float = None,
                      base_uplift: str = None) -> Dict[str, Any]:
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
        Concrete unit weight (kN/m³). Default 23.6 (:data:`GAMMA_CONCRETE`).
    FOS_required : float, optional
        Required FOS. Default None: 2.0 on soil, 1.5 on rock
        (:func:`default_FOS_overturning`, GM-288).
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    live_load_resists, dead_load_resists : bool, optional
        As in :func:`check_sliding`. Default False.
    wall_friction_deg, gamma_foundation : float, optional
        As in :func:`check_sliding`.
    founded_on_rock : bool, optional
        The eccentricity limit is B/4, not B/6 (:data:`ECCENTRICITY_BASIS`),
        and the default FOS_required 1.5. Default False.
    gamma_sat_backfill, base_uplift : optional
        With ``geom.water_table_depth``, as in :func:`check_sliding`
        (GM-292).

    Returns
    -------
    dict
        FOS_overturning, stabilizing_moment, overturning_moment, passes,
        and the eccentricity check: eccentricity_m, in_middle_third,
        eccentricity_limit_m, within_eccentricity_limit.
        With no overturning moment (zero active thrust) ``passes`` is True,
        ``no_driving_force`` is True and ``FOS_overturning`` is None.
    """
    return _overturning_dict(_overturning_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, gamma_concrete,
        pressure_method, live_load_resists, dead_load_resists,
        wall_friction_deg, gamma_foundation, founded_on_rock,
        gamma_sat_backfill, base_uplift),
        default_FOS_overturning(founded_on_rock) if FOS_required is None
        else FOS_required)


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
        "eccentricity_limit_m": round(t["eccentricity_limit"], 3),
        "within_eccentricity_limit": t["within_eccentricity_limit"],
    }


def _bearing_terms(geom: CantileverWallGeometry,
                   gamma_backfill: float, phi_backfill: float,
                   c_backfill: float = 0.0,
                   q_allowable: float = None,
                   gamma_concrete: float = GAMMA_CONCRETE,
                   pressure_method: str = "rankine",
                   surcharge_in_bearing: bool = True,
                   wall_friction_deg: float = None,
                   gamma_foundation: float = None,
                   bearing_pressure: str = None,
                   founded_on_rock: bool = False,
                   gamma_sat_backfill: float = None,
                   base_uplift: str = None,
                   uplift_in_bearing: bool = False) -> Dict[str, Any]:
    """The bearing check's terms, unrounded (GM-143): ``V``, ``M_stab``,
    ``M_over``, ``x_resultant``, ``eccentricity`` (the eccentricity for
    bearing, GM-157), ``in_middle_third``, ``distribution``
    ("trapezoidal", or "triangular_toe" / "triangular_heel" outside the
    middle third), ``B_eff`` (the triangle's width, None for a
    trapezoid), ``q_toe``, ``q_heel``, ``q_max`` (the larger edge
    pressure), ``FOS`` and ``passes``.

    GM-288: the pressure compared with q_allowable, ``q_bearing``, is by
    ``bearing_pressure`` (:data:`BEARING_PRESSURE_BASIS`; None:
    :func:`default_bearing_pressure`) the equivalent uniform pressure
    ``q_eq`` = V / ``B_prime``, B' = B - 2|e| ("uniform"), or ``q_max``
    ("linear"); FOS = q_allowable / q_bearing.

    The loads are the bearing check's (GM-157): the surcharges over the
    heel count as ``surcharge_in_bearing`` says (default: they do,
    :data:`BEARING_BASIS`), and the whole of the thrust's vertical
    component counts. So the eccentricity here can differ from the
    eccentricity check's (:func:`_overturning_terms`): GEC-11 Vol I
    p. 4-26, "eB = eccentricity for bearing calculation (not equal to
    eccentricity check e)".

    GM-292: with a water table ``M_over`` adds the water's thrust, Pw
    z_Pw; the uplift counts as ``uplift_in_bearing`` says (default: it does
    not, :data:`UPLIFT_BEARING_BASIS`), taken off V with its moment in
    ``M_over``, as the overturning check counts it."""
    B = geom.base_width

    bearing_pressure_given = bearing_pressure is not None
    if bearing_pressure is None:
        bearing_pressure = default_bearing_pressure(founded_on_rock)
    if bearing_pressure not in BEARING_PRESSURES:
        raise ValueError(f"bearing_pressure must be one of "
                         f"{list(BEARING_PRESSURES)}, got {bearing_pressure!r}")

    # Active thrust, decomposed per the chosen method (RW-1)
    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method,
                                  wall_friction_deg, gamma_sat_backfill)
    Pa, z_Pa, delta_deg = thrust["Pa"], thrust["z_Pa"], thrust["delta_deg"]
    Ph, Pv = thrust["Ph"], thrust["Pv"]

    # Weights, the surcharges over the heel the check counts and the
    # vertical thrust component (at the pressure plane, x = B), with their
    # moments about the toe; the uplift as uplift_in_bearing says (GM-292)
    items = _vertical_force_items(geom, gamma_backfill, gamma_concrete,
                                  thrust,
                                  surcharge_in_bearing=surcharge_in_bearing,
                                  gamma_front=gamma_foundation,
                                  base_uplift=base_uplift,
                                  uplift_in_bearing=uplift_in_bearing)
    V = _sum(items, "W", "bearing")
    M_stab = sum(i["M"] for i in items
                 if i["bearing"] and i["kind"] != "uplift")
    U, M_uplift = _uplift([i for i in items if i["bearing"]])

    M_over = Ph * z_Pa + thrust["Pw"] * thrust["z_Pw"] + M_uplift

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

    # The equivalent uniform pressure over B' = B - 2|e| (GEC-6 Eq. 6-6 and
    # Sec. 6.4.2; NHI-06-089 Fig. 10-17; UFC 3-220-20 Eq. 4-27), GM-288
    B_prime = B - 2.0 * abs(eccentricity)
    q_eq = V / B_prime if B_prime > 0 and V > 0 else 0.0

    # Bearing check (GM-100). Without q_allowable it is NOT RUN: no FOS and
    # no verdict, never a placeholder reported as a pass. The demand is
    # q_eq ("uniform") or the larger edge pressure ("linear"; a resultant on
    # the heel side of the centre puts it under the heel).
    q_max = max(q_toe, q_heel)
    q_bearing = q_eq if bearing_pressure == "uniform" else q_max
    if q_allowable is None:
        FOS_bearing = None
        passes = None
    elif q_bearing > 0:
        FOS_bearing = q_allowable / q_bearing
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
            "B_prime": B_prime, "q_eq": q_eq,
            "bearing_pressure": bearing_pressure,
            "bearing_pressure_given": bearing_pressure_given,
            "founded_on_rock": bool(founded_on_rock),
            "q_bearing": q_bearing,
            "q_allowable": q_allowable, "FOS": FOS_bearing,
            "passes": passes, "Pa": Pa, "Pv": Pv, "delta_deg": delta_deg,
            "surcharge_in_bearing": surcharge_in_bearing,
            "Pw": thrust["Pw"], "U": U, "M_uplift": M_uplift,
            "uplift_in_bearing": bool(uplift_in_bearing),
            "h_w": thrust["h_w"]}


def _bearing_dict(t: Dict[str, Any]) -> Dict[str, Any]:
    """:func:`check_bearing`'s dict from its terms."""
    FOS_bearing = t["FOS"]
    return {
        "q_toe_kPa": round(t["q_toe"], 1),
        "q_heel_kPa": round(t["q_heel"], 1),
        "q_eq_kPa": round(t["q_eq"], 1),
        "B_prime_m": round(t["B_prime"], 3),
        "bearing_pressure": t["bearing_pressure"],
        "q_bearing_kPa": round(t["q_bearing"], 1),
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
                  gamma_concrete: float = GAMMA_CONCRETE,
                  pressure_method: str = "rankine",
                  surcharge_in_bearing: bool = True,
                  wall_friction_deg: float = None,
                  gamma_foundation: float = None,
                  bearing_pressure: str = None,
                  founded_on_rock: bool = False,
                  gamma_sat_backfill: float = None,
                  base_uplift: str = None,
                  uplift_in_bearing: bool = False) -> Dict[str, Any]:
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
        Concrete unit weight (kN/m³). Default 23.6 (:data:`GAMMA_CONCRETE`).
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    surcharge_in_bearing : bool, optional
        Count the surcharges over the heel in the bearing pressures.
        Default True (:data:`BEARING_BASIS`).
    wall_friction_deg, gamma_foundation : float, optional
        As in :func:`check_sliding`.
    bearing_pressure : str, optional
        The pressure compared with q_allowable: "uniform", q_eq = V / B'
        over B' = B - 2e, or "linear", the larger edge pressure
        (:data:`BEARING_PRESSURE_BASIS`). None: "uniform" on soil, "linear"
        with founded_on_rock (GM-288).
    founded_on_rock : bool, optional
        Default False.
    gamma_sat_backfill, base_uplift : optional
        With ``geom.water_table_depth``, as in :func:`check_sliding`
        (GM-292).
    uplift_in_bearing : bool, optional
        With ``geom.water_table_depth``, count the uplift in the bearing
        pressures. Default False (:data:`UPLIFT_BEARING_BASIS`).

    Returns
    -------
    dict
        q_toe, q_heel, q_eq, B_prime, bearing_pressure, q_bearing,
        eccentricity, in_middle_third, FOS_bearing (= q_allowable /
        q_bearing), bearing_checked, passes (None when not checked).
    """
    return _bearing_dict(_bearing_terms(
        geom, gamma_backfill, phi_backfill, c_backfill, q_allowable,
        gamma_concrete, pressure_method, surcharge_in_bearing,
        wall_friction_deg, gamma_foundation, bearing_pressure,
        founded_on_rock, gamma_sat_backfill, base_uplift,
        uplift_in_bearing))


def analyze_cantilever_wall(geom: CantileverWallGeometry,
                            gamma_backfill: float,
                            phi_backfill: float,
                            c_backfill: float = 0.0,
                            phi_foundation: float = None,
                            c_foundation: float = 0.0,
                            q_allowable: float = None,
                            gamma_concrete: float = GAMMA_CONCRETE,
                            FOS_sliding: float = None,
                            FOS_overturning: float = None,
                            pressure_method: str = "rankine",
                            include_passive: bool = False,
                            gamma_foundation: float = None,
                            delta_base: float = None,
                            base_adhesion: float = None,
                            live_load_resists: bool = False,
                            dead_load_resists: bool = False,
                            surcharge_in_bearing: bool = True,
                            sliding_cohesion: str = None,
                            passive_theory: str = "rankine",
                            log_spiral_source: str = None,
                            passive_wall_friction_deg: float = None,
                            wall_friction_deg: float = None,
                            interface_gap_possible: bool = False,
                            founded_on_rock: bool = False,
                            bearing_pressure: str = None,
                            gamma_sat_backfill: float = None,
                            base_uplift: str = None,
                            uplift_in_bearing: bool = False
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
        Allowable bearing pressure (kPa); on soil ("uniform"
        ``bearing_pressure``, the default) the allowable pressure for the
        effective width B' = B - 2e (GEC-6 Sec. 6.4.2, p. 119). Without it
        the bearing check is NOT RUN: the result's FOS_bearing and
        passes_bearing are None, its
        ``checks_not_run`` lists "bearing" and its summary says "not
        checked". q_toe / q_heel and the middle-third check are reported
        either way.
    gamma_concrete : float, optional
        Concrete unit weight (kN/m³). Default 23.6, 150 pcf
        (:data:`GAMMA_CONCRETE`, GM-288).
    FOS_sliding : float, optional
        Required FOS for sliding. Default None: 1.5, or 2.0 with
        include_passive (:func:`default_FOS_sliding`, GM-288).
    FOS_overturning : float, optional
        Required FOS for overturning. Default None: 2.0, or 1.5 with
        founded_on_rock (:func:`default_FOS_overturning`, GM-288).
    pressure_method : str, optional
        "rankine" or "coulomb". Default "rankine".
    include_passive : bool, optional
        If True, include passive resistance in sliding check. Default False.
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m³). If None, uses gamma_backfill.
    passive_theory, log_spiral_source, passive_wall_friction_deg : optional
        With include_passive, the passive coefficient in front of the wall
        as :func:`check_sliding` takes them (GM-242): Rankine's by
        default.
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
    wall_friction_deg : float, optional
        With pressure_method "coulomb", the wall friction on the pressure
        plane through the heel (deg). None: the backfill slope
        (:data:`COULOMB_DELTA_BASIS`, GM-288).
    interface_gap_possible : bool, optional
        Under the undrained rule, hold the base resistance to 0.4 V where
        water or air can reach the interface (EN 1997-1 Eq. 6.5,
        :data:`UNDRAINED_GAP_BASIS`, GM-288). Default False.
    founded_on_rock : bool, optional
        The wall is founded on rock: the eccentricity limit is B/4
        (:data:`ECCENTRICITY_BASIS`), FOS_overturning defaults to 1.5 and
        bearing_pressure to "linear" (GM-288). Default False.
    bearing_pressure : str, optional
        "uniform" (q_eq = V / B', B' = B - 2e) or "linear" (the larger edge
        pressure), compared with q_allowable (:data:`BEARING_PRESSURE_BASIS`,
        GM-288). None: "uniform" on soil, "linear" on rock.
    gamma_sat_backfill : float, optional
        With a water table (``geom.water_table_depth``), the backfill's
        unit weight below it (kN/m3), more than gamma_w. None:
        gamma_backfill (:data:`GAMMA_SAT_BASIS`, GM-292).
    base_uplift : str, optional
        With a water table, the water pressure under the base: "linear"
        (None; gamma_w h_w at the heel to 0 at the toe) or "uniform"
        (gamma_w h_w under the whole base); :data:`BASE_UPLIFT_BASIS`.
    uplift_in_bearing : bool, optional
        With a water table, count the uplift in the bearing pressures.
        Default False (:data:`UPLIFT_BEARING_BASIS`).

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
        surcharges over the heel were counted when there are any (GM-157),
        and with include_passive the passive coefficient in front of the
        wall (GM-242, :func:`toe_passive_choices`); with a water table how
        it loads the wall (GM-292, :func:`water_choices`).
    """
    if geom.water_table_depth is None and (base_uplift is not None
                                           or uplift_in_bearing):
        raise ValueError(
            "base_uplift and uplift_in_bearing are the water pressure under "
            "the base below a water table behind the wall; with no "
            "water_table_depth there is none. Give water_table_depth, or "
            "leave them out.")
    rule_default = default_sliding_cohesion(phi_foundation)
    FOS_sliding_given = FOS_sliding is not None
    FOS_overturning_given = FOS_overturning is not None
    if FOS_sliding is None:
        FOS_sliding = default_FOS_sliding(include_passive)
    if FOS_overturning is None:
        FOS_overturning = default_FOS_overturning(founded_on_rock)

    def slide(rule, delta_base=delta_base, base_adhesion=base_adhesion,
              gap=interface_gap_possible, g=geom, gamma_sat=gamma_sat_backfill,
              uplift=base_uplift):
        return _sliding_terms(
            g, gamma_backfill, phi_backfill, c_backfill,
            phi_foundation, c_foundation, gamma_concrete,
            pressure_method, include_passive, gamma_foundation,
            delta_base=delta_base, base_adhesion=base_adhesion,
            live_load_resists=live_load_resists,
            dead_load_resists=dead_load_resists, sliding_cohesion=rule,
            passive_theory=passive_theory,
            log_spiral_source=log_spiral_source,
            passive_wall_friction_deg=passive_wall_friction_deg,
            wall_friction_deg=wall_friction_deg,
            interface_gap_possible=gap, gamma_sat_backfill=gamma_sat,
            base_uplift=uplift)

    s_terms = slide(sliding_cohesion)
    sliding = _sliding_dict(s_terms, FOS_sliding)
    # The undrained resistance with the other reading of EN 1997-1's gap
    # limit, for its choice
    s_gap_other = (slide(sliding_cohesion, gap=not interface_gap_possible)
                   if s_terms["sliding_cohesion"] == "undrained" else None)

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
                                     else base_adhesion),
                      gap=interface_gap_possible and rule == "undrained")
            others[rule] = t["FOS"]
        except ValueError as exc:
            others[rule] = str(exc)

    def overturn(g=geom, gamma_sat=gamma_sat_backfill, uplift=base_uplift):
        return _overturning_terms(
            g, gamma_backfill, phi_backfill, c_backfill,
            gamma_concrete, pressure_method,
            live_load_resists=live_load_resists,
            dead_load_resists=dead_load_resists,
            wall_friction_deg=wall_friction_deg,
            gamma_foundation=gamma_foundation,
            founded_on_rock=founded_on_rock,
            gamma_sat_backfill=gamma_sat, base_uplift=uplift)

    o_terms = overturn()
    overturning = _overturning_dict(o_terms, FOS_overturning)

    def bear(pressure, in_bearing=uplift_in_bearing):
        return _bearing_terms(
            geom, gamma_backfill, phi_backfill, c_backfill,
            q_allowable, gamma_concrete, pressure_method,
            surcharge_in_bearing=surcharge_in_bearing,
            wall_friction_deg=wall_friction_deg,
            gamma_foundation=gamma_foundation,
            bearing_pressure=pressure, founded_on_rock=founded_on_rock,
            gamma_sat_backfill=gamma_sat_backfill, base_uplift=base_uplift,
            uplift_in_bearing=in_bearing)

    b_terms = bear(bearing_pressure)
    bearing = _bearing_dict(b_terms)

    thrust = _active_thrust_terms(geom, gamma_backfill, phi_backfill,
                                  c_backfill, pressure_method,
                                  wall_friction_deg, gamma_sat_backfill)

    # GM-292: the water table's choices, with the other readings' numbers:
    # the drained backfill, the other uplift and the uplift in bearing
    water = []
    if geom.water_table_depth is not None:
        others_w = {}
        if thrust["h_w"] > 0:
            import dataclasses
            dry = dataclasses.replace(geom, water_table_depth=None)
            others_w["dry"] = (slide(sliding_cohesion, g=dry, gamma_sat=None,
                                     uplift=None),
                               overturn(g=dry, gamma_sat=None, uplift=None))
            other = ("uniform" if (base_uplift or "linear") == "linear"
                     else "linear")
            others_w["uplift"] = (other, slide(sliding_cohesion, uplift=other),
                                  overturn(uplift=other))
            others_w["bearing"] = bear(bearing_pressure,
                                       in_bearing=not uplift_in_bearing)
        water = water_choices(geom, thrust, s_terms, o_terms, b_terms,
                              others_w, gamma_sat_backfill=gamma_sat_backfill,
                              base_uplift=base_uplift,
                              gamma_backfill=gamma_backfill)

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
        founded_on_rock=bool(founded_on_rock),
        bearing_pressure=b_terms["bearing_pressure"],
        q_bearing=round(b_terms["q_bearing"], 1),
        earth_pressure=thrust,
        vertical_forces=_vertical_force_items(
            geom, gamma_backfill, gamma_concrete, thrust,
            live_load_resists, dead_load_resists, surcharge_in_bearing,
            gamma_front=gamma_foundation, base_uplift=base_uplift,
            uplift_in_bearing=uplift_in_bearing),
        sliding_terms=s_terms,
        overturning_terms=o_terms,
        bearing_terms=b_terms,
        choices=sliding_cohesion_choice(s_terms, rule_default, others)
        + ([base_friction_choice(s_terms)]
           if s_terms["delta_b"] is not None else [])
        + heel_surcharge_choices(geom, live_load_resists,
                                 dead_load_resists, surcharge_in_bearing)
        + toe_passive_choices(s_terms)
        + wall_basis_choices(
            thrust, s_terms, b_terms,
            other_bearing=(bear("linear" if b_terms["bearing_pressure"]
                                == "uniform" else "uniform")
                           if q_allowable is not None else None),
            s_gap_other=s_gap_other, phi_backfill=phi_backfill,
            gamma_backfill=gamma_backfill, c_backfill=c_backfill,
            FOS_sliding=FOS_sliding, FOS_sliding_given=FOS_sliding_given,
            FOS_overturning=FOS_overturning,
            FOS_overturning_given=FOS_overturning_given,
            founded_on_rock=founded_on_rock)
        + water,
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
        # A pavement or slab is modelled as retained soil up to the ground
        # surface; any other surcharge is transient live load (OWN-64).
        out.append(choice(
            "pavement or slab over the backfill",
            "modelled as retained soil up to the ground surface; a "
            "surcharge is transient live load (its thrust counts in every "
            "check, its weight over the heel does not resist sliding or "
            "overturning)",
            "module", "the usual practice for pavements and slabs; "
            "dead_load_resists=true counts a dead-load surcharge over the "
            "heel as resisting weight in sliding and overturning",
            []))
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


def toe_passive_choices(terms: Dict[str, Any]) -> list:
    """How the sliding check took the passive resistance in front of the
    wall (GM-242), as a result's ``choices`` (``geotech_common.choices``):
    none without passive; else the theory of Kp with the other theory's and
    no passive's FOS as alternatives, and with the log-spiral Kp its source
    and the wall friction. ``terms`` are the sliding check's
    (:func:`_sliding_terms`)."""
    from geotech_common import log_spiral as _ls
    from geotech_common.choices import choice, how_given
    if terms["passive_theory"] is None:
        return []
    phi_f = terms["phi_foundation"]
    D, D_top = terms["D_passive"], terms["D_passive_top"]
    g, driving = terms["gamma_foundation"], terms["driving"]
    without = terms["resisting"] - terms["Pp"]

    def fos(R):
        return ("no FOS (no driving force)" if driving <= 0
                else f"FOS_sliding {R / driving:.3g}")

    def with_Kp(Kp):
        Pp = 0.5 * Kp * g * (D ** 2 - D_top ** 2)
        if terms["passive_cohesion"] and terms["c_foundation"]:
            # with its cohesion, as the check counts it (GM-288)
            Pp += 2.0 * terms["c_foundation"] * math.sqrt(Kp) * (D - D_top)
        return f"Kp {Kp:.3g}, Pp {Pp:.1f} kN/m, {fos(without + Pp)}"

    theory, Kp = terms["passive_theory"], terms["Kp"]
    left_out = (f"passive resistance left out (include_passive false), as "
                f"UFC 3-220-20 p. 219 says is typical: {fos(without)}")
    log_spiral = terms["log_spiral_source"]
    if theory == "rankine":
        default = _ls.level_front_Kp(phi_f)
        others = [f"log_spiral (passive_theory 'log_spiral'), UFC 3-220-20 "
                  f"Eq. 4-12 at delta/phi' 0.66: {with_Kp(default['Kp'])}",
                  left_out]
        chosen = f"rankine: {with_Kp(Kp)}, no wall friction"
    else:
        others = [f"rankine (passive_theory 'rankine'), no wall friction: "
                  f"{with_Kp(rankine_Kp(phi_f))}", left_out]
        chosen = (f"log_spiral: {with_Kp(Kp)}, wall friction "
                  f"{terms['passive_delta']:.3g} deg")
    out = [choice(
        "passive resistance in front of the wall, in sliding "
        "(passive_theory)", chosen,
        how_given(theory, "rankine", name="passive_theory"),
        TOE_PASSIVE_BASIS, others)]
    if log_spiral is None:
        return out
    delta, route = terms["passive_delta"], terms["log_spiral_route"]
    basis = _ls.SOURCE_BASIS[log_spiral]
    if route == "dm7_2_eq_4_12":
        basis += (". The front of the wall is a vertical face with level "
                  "ground, so it is Eq. 4-12 (p. 191). "
                  + _ls.EQ_VS_CHARTS_NOTE)
    if "dm7_2_eq_4_12" in (log_spiral, route) \
            and _ls.above_the_charts(phi_f, delta):
        basis += ". " + _ls.HIGH_FRICTION_NOTE
    out.append(choice(
        "Log-spiral passive Kp in front of the wall (log_spiral_source)",
        log_spiral,
        how_given(log_spiral, _ls.DEFAULT_SOURCE, name="log_spiral_source"),
        basis, _ls.other_sources(phi_f, delta, used=(log_spiral, route))))
    out.append(choice(
        "Wall friction delta in front of the wall "
        "(passive_wall_friction_deg)",
        f"{delta:.3g} deg" + (f" (delta/phi' {delta / phi_f:.2f})"
                              if phi_f else ""),
        how_given(delta if terms["passive_delta_given"] else None, None,
                  name="passive_wall_friction_deg", took=round(delta, 4)),
        _ls.wall_friction_note(
            log_spiral, left_out=_ls.level_front_left_out(log_spiral))))
    return out


#: NHI-06-089's minimum active pressure in a cohesive backfill (kPa per m
#: of depth), for the tension-zone choice's alternative (GM-288).
NHI_MIN_ACTIVE_PRESSURE_PER_M = 5.0


def _thrust_with_floor(Ka: float, gamma: float, H: float, c: float,
                       q: float, floor_per_m: float,
                       n: int = 2000, gamma_b: float = None,
                       z_w: float = None) -> float:
    """The active thrust (kN/m) with the pressure at no depth below
    ``floor_per_m`` z (kPa): FHWA NHI-06-089 p. 10-15's "the effective
    pressure on the wall at any depth should not be less than ... 5z
    kN/m2", by the midpoint rule over ``n`` slices; below a water table
    ``z_w`` deep on ``gamma_b`` (GM-292)."""
    dz = H / n
    root = math.sqrt(Ka)
    total = 0.0
    for i in range(n):
        z = (i + 0.5) * dz
        sigma = (q + gamma * z if z_w is None else
                 q + gamma * min(z, z_w) + gamma_b * max(z - z_w, 0.0))
        total += max(Ka * sigma - 2.0 * c * root,
                     floor_per_m * z) * dz
    return total


def wall_basis_choices(thrust: Dict[str, Any], s_terms: Dict[str, Any],
                       b_terms: Dict[str, Any],
                       other_bearing: Optional[Dict[str, Any]] = None,
                       s_gap_other: Optional[Dict[str, Any]] = None,
                       phi_backfill: float = None,
                       gamma_backfill: float = None,
                       c_backfill: float = 0.0,
                       FOS_sliding: float = None,
                       FOS_sliding_given: bool = False,
                       FOS_overturning: float = None,
                       FOS_overturning_given: bool = False,
                       founded_on_rock: bool = False) -> list:
    """The readings GM-288 made stated choices, as a result's ``choices``
    (``geotech_common.choices``), each only where it applies: Coulomb's
    wall friction on the pressure plane; the tension zone of a cohesive
    backfill; the foundation cohesion in the passive resistance; the
    required FOS for sliding with the passive resistance counted, and for
    overturning on rock; the bearing pressure compared with q_allowable;
    EN 1997-1's limit on the undrained sliding resistance. ``thrust``,
    ``s_terms`` and ``b_terms`` are the checks' terms; ``other_bearing``
    the bearing terms under the other pressure and ``s_gap_other`` the
    sliding terms under the other reading of the gap limit, for the
    alternatives."""
    from geotech_common.choices import choice, how_given
    out = []

    def fos(R, driving):
        return ("no driving force" if driving <= 0
                else f"FOS_sliding {R / driving:.3g}")

    # Coulomb's wall friction on the pressure plane
    if thrust["method"] == "coulomb":
        delta, beta = thrust["delta_deg"], thrust["backfill_slope"]
        given = thrust["delta_given"]
        chosen = (f"delta = {delta:g} deg (wall_friction_deg), Ka "
                  f"{thrust['Ka']:.3f}" if given else
                  f"delta = the backfill slope = {delta:g} deg, Ka "
                  f"{thrust['Ka']:.3f}")
        if given:
            Ka_b = coulomb_Ka(phi_backfill, beta, beta_deg=beta)
            alts = [f"delta = the backfill slope = {beta:g} deg, the "
                    f"module's default: Ka {Ka_b:.3f}. "
                    + COULOMB_DELTA_BASIS]
        else:
            Ka_phi = coulomb_Ka(phi_backfill, phi_backfill, beta_deg=beta)
            alts = [f"delta = phi_backfill = {phi_backfill:g} deg "
                    f"(wall_friction_deg), the full soil-on-soil friction "
                    f"the resultant may take on that plane (NHI-06-089 "
                    f"p. 10-20): Ka {Ka_phi:.3f}"]
        out.append(choice(
            "wall friction on the pressure plane through the heel, in "
            "Coulomb's Ka (wall_friction_deg)", chosen,
            how_given(delta if given else None, None,
                      name="wall_friction_deg", took=round(delta, 4)),
            COULOMB_DELTA_BASIS, alts))

    # The tension zone of a cohesive backfill
    z_t = thrust.get("z_tension", 0.0) or 0.0
    if z_t > 0:
        Ka, H = thrust["Ka"], thrust["H_active"]
        q = thrust["q_live"] + thrust["q_dead"]
        if thrust.get("h_w"):
            # on the effective stress below a water table (GM-292)
            g_b, z_w = thrust["gamma_b"], thrust["z_w"]
            Pa_c0 = active_force_components_water(
                gamma_backfill, g_b, H, z_w, Ka, 0.0, q)["Pa"]
            Pa_5z = _thrust_with_floor(Ka, gamma_backfill, H, c_backfill, q,
                                       NHI_MIN_ACTIVE_PRESSURE_PER_M,
                                       gamma_b=g_b, z_w=z_w)
        else:
            Pa_c0 = active_force_components(gamma_backfill, H, Ka, 0.0,
                                            q)["Pa"]
            Pa_5z = _thrust_with_floor(Ka, gamma_backfill, H, c_backfill, q,
                                       NHI_MIN_ACTIVE_PRESSURE_PER_M)
        out.append(choice(
            "active pressure where the backfill cohesion takes it below "
            "zero (tension zone)",
            f"zero over the top {z_t:.2f} m; Pa = {thrust['Pa']:.1f} kN/m "
            f"(the negative pressure, {thrust['Pa_tension_zone']:.1f} kN/m "
            f"of pull, not counted)",
            "module",
            "UFC 3-220-20 Fig. 4-5 (p. 183), 'In no case should negative "
            "pressure be considered'; FHWA NHI-06-089 p. 10-15, 'The active "
            "earth pressure acting against the wall within the depth of the "
            "tension crack is assumed to be zero. Unless positive drainage "
            "measures are provided, water infiltration into the tension "
            "crack may result in hydrostatic pressure on the retaining "
            "structure' (not modelled)",
            [f"the cohesion neglected (c_backfill 0), as UFC 3-220-20 "
             f"Fig. 4-5 allows ('Cohesion term may be neglected'): Pa = "
             f"{Pa_c0:.1f} kN/m",
             f"a minimum pressure of 5z kN/m2 at depth z (NHI-06-089 "
             f"p. 10-15, for c' above about 5 kPa; not an input here): "
             f"Pa = {Pa_5z:.1f} kN/m"]))

    # The foundation cohesion in the passive resistance
    c_f = s_terms["c_foundation"]
    if s_terms["passive_theory"] is not None and c_f:
        Kp, D, D_top = (s_terms["Kp"], s_terms["D_passive"],
                        s_terms["D_passive_top"])
        h = D - D_top
        Ppc = 2.0 * c_f * math.sqrt(Kp) * h
        driving = s_terms["driving"]
        counted = s_terms["passive_cohesion"]
        R = s_terms["resisting"]
        if counted:
            chosen = (f"counted (the undrained rule): 2 cu sqrt(Kp) h = 2 x "
                      f"{c_f:g} x sqrt({Kp:.3g}) x {h:.2f} = {Ppc:.1f} kN/m, "
                      f"{fos(R, driving)}")
            alt = (f"not counted: {fos(R - Ppc, driving)}. "
                   + TOE_PASSIVE_COHESION_BASIS[False])
        else:
            chosen = (f"not counted (the {s_terms['sliding_cohesion']} "
                      f"rule), {fos(R, driving)}")
            alt = (f"2 c sqrt(Kp) h = {Ppc:.1f} kN/m counted: "
                   f"{fos(R + Ppc, driving)}, with c_foundation as cu "
                   "(sliding_cohesion 'undrained')")
        out.append(choice(
            "foundation cohesion in the passive resistance in front of the "
            "wall", chosen, "module", TOE_PASSIVE_COHESION_BASIS[counted],
            [alt]))

    # The required factors of safety the readings set; the words depend on
    # the value only, given or not ("how" says which)
    if s_terms["passive_theory"] is not None:
        printed = FOS_sliding == FOS_SLIDING[True]
        alts = ([] if printed else
                [f"{FOS_SLIDING[True]:g}, the default with the passive "
                 f"resistance counted: " + FOS_SLIDING_BASIS[True]])
        alts.append(f"{FOS_SLIDING[False]:g} without the passive "
                    "resistance: " + FOS_SLIDING_BASIS[False])
        out.append(choice(
            "required factor of safety against sliding "
            "(FOS_sliding_required)",
            f"{FOS_sliding:g}, the passive resistance counted",
            how_given(FOS_sliding if FOS_sliding_given else None, None,
                      name="FOS_sliding_required", took=FOS_sliding),
            FOS_SLIDING_BASIS[True] if printed else "the value given",
            alts))
    if founded_on_rock:
        printed = FOS_overturning == FOS_OVERTURNING[True]
        alts = ([] if printed else
                [f"{FOS_OVERTURNING[True]:g}, the default on rock: "
                 + FOS_OVERTURNING_BASIS[True]])
        alts.append(f"{FOS_OVERTURNING[False]:g} on soil: "
                    + FOS_OVERTURNING_BASIS[False])
        out.append(choice(
            "required factor of safety against overturning "
            "(FOS_overturning_required)",
            f"{FOS_overturning:g}, on rock",
            how_given(FOS_overturning if FOS_overturning_given else None,
                      None, name="FOS_overturning_required",
                      took=FOS_overturning),
            FOS_OVERTURNING_BASIS[True] if printed else "the value given",
            alts))

    # The bearing pressure compared with q_allowable
    if b_terms["q_allowable"] is not None:
        def bearing_words(t):
            fs = ("the resultant outside the base" if not t["q_bearing"] > 0
                  else f"FOS_bearing {t['q_allowable'] / t['q_bearing']:.3g}")
            if t["bearing_pressure"] == "uniform":
                return (f"uniform: q_eq = V / B' = {t['V']:.1f} / "
                        f"{max(t['B_prime'], 0.0):.2f} = {t['q_eq']:.1f} kPa, "
                        f"B' = B - 2e, {fs}")
            return (f"linear: the larger edge pressure q_max = "
                    f"{t['q_max']:.1f} kPa ({t['distribution']}), {fs}")
        bp = b_terms["bearing_pressure"]
        basis = BEARING_PRESSURE_BASIS[bp]
        if founded_on_rock and not b_terms["bearing_pressure_given"]:
            basis = "founded on rock: " + basis
        alts = []
        if other_bearing is not None:
            alts.append(bearing_words(other_bearing) + ". "
                        + BEARING_PRESSURE_BASIS[
                            other_bearing["bearing_pressure"]])
        out.append(choice(
            "bearing pressure compared with q_allowable (bearing_pressure)",
            bearing_words(b_terms),
            how_given(bp if b_terms["bearing_pressure_given"] else None,
                      None, name="bearing_pressure", took=bp),
            basis, alts))

    # EN 1997-1's limit on the undrained sliding resistance
    if s_terms["sliding_cohesion"] == "undrained":
        flag = s_terms["interface_gap_possible"]
        driving = s_terms["driving"]
        if flag:
            chosen = ("applied (interface_gap_possible true): R_base held "
                      f"to 0.4 V = {0.4 * s_terms['V']:.1f} kN/m"
                      if s_terms["gap_limited"] else
                      "applied (interface_gap_possible true), not governing: "
                      f"R_base {s_terms['R_adhesion']:.1f} kN/m <= 0.4 V = "
                      f"{0.4 * s_terms['V']:.1f} kN/m")
        else:
            chosen = ("not applied (interface_gap_possible false): no water "
                      "or air taken to reach the base")
        alts = []
        if s_gap_other is not None:
            o = s_gap_other
            alts.append(
                ("applied (interface_gap_possible true): "
                 if not flag else "not applied: ")
                + fos(o["resisting"], driving))
        out.append(choice(
            "EN 1997-1 limit on the undrained sliding resistance, Rd <= "
            "0.4 Vd (interface_gap_possible)", chosen,
            how_given(flag, False, name="interface_gap_possible"),
            UNDRAINED_GAP_BASIS, alts))
    return out


def water_choices(geom: CantileverWallGeometry, thrust: Dict[str, Any],
                  s_terms: Dict[str, Any], o_terms: Dict[str, Any],
                  b_terms: Dict[str, Any], others: Dict[str, Any],
                  gamma_sat_backfill: float = None,
                  base_uplift: str = None,
                  gamma_backfill: float = None) -> list:
    """How the water table behind the wall loads it (GM-292), as a
    result's ``choices`` (``geotech_common.choices``): the water table,
    with the drained backfill's FOS beside it; and with the water above
    the underside of the base the unit weight below it, the water pressure
    under the base (the other distribution's numbers beside it), where the
    uplift's moment goes in the overturning check, and whether the bearing
    pressures count the uplift. ``others``: "dry", the (sliding,
    overturning) terms with no water table; "uplift", (the other
    distribution, its sliding and overturning terms); "bearing", the
    bearing terms with ``uplift_in_bearing`` the other way. A
    ``gamma_sat_backfill`` given equal to ``gamma_backfill`` keeps the
    default's basis."""
    from geotech_common.choices import choice, how_given

    def fos_s(t):
        return ("no driving force" if t["FOS"] is None
                else f"FOS_sliding {t['FOS']:.3g}")

    def fos_o(t):
        return ("no overturning moment" if t["FOS"] is None
                else f"FOS_overturning {t['FOS']:.3g}")

    d_w, h_w = geom.water_table_depth, thrust["h_w"]
    what = "water table behind the wall (water_table_depth)"
    how = how_given(d_w, None, name="water_table_depth")
    if not h_w > 0:
        return [choice(
            what, f"{d_w:g} m below the top of the backfill, at or below the "
            f"underside of the base (wall_height {geom.wall_height:g} m): no "
            "water pressure on the wall and no uplift", how, DRAINED_BASIS)]
    dry_s, dry_o = others["dry"]
    basis = WATER_PRESSURE_BASIS
    if b_terms["q_allowable"] is not None:
        basis += (". q_allowable is the input's: give it for the water "
                  "table at this level (GEC-6 Sec. 5.2.3.2, p. 56; UFC "
                  "3-220-20 Sec. 5-3.2, p. 291)")
    out = [choice(
        what,
        f"{d_w:g} m below the top of the backfill, h_w = {h_w:.2f} m above "
        f"the underside of the base: the earth pressure below it on "
        f"gamma_b = {thrust['gamma_b']:.2f} kN/m3 (Pa = {thrust['Pa']:.1f} "
        f"kN/m), and the water pressure added, Pw = gamma_w h_w^2 / 2 = "
        f"{thrust['Pw']:.1f} kN/m at h_w / 3 = {thrust['z_Pw']:.2f} m: "
        f"{fos_s(s_terms)}, {fos_o(o_terms)}", how, basis,
        [f"the backfill drained, no water table (water_table_depth left "
         f"out): {fos_s(dry_s)}, {fos_o(dry_o)}. " + DRAINED_BASIS])]

    g_sat, g_b = thrust["gamma_sat"], thrust["gamma_b"]
    out.append(choice(
        "backfill unit weight below the water table (gamma_sat_backfill)",
        f"gamma_sat = {g_sat:g} kN/m3, gamma_b = gamma_sat - gamma_w = "
        f"{g_sat:g} - {GAMMA_W:g} = {g_b:.2f} kN/m3",
        how_given(gamma_sat_backfill, None, name="gamma_sat_backfill",
                  took=g_sat),
        GAMMA_SAT_BASIS if gamma_sat_backfill in (None, gamma_backfill)
        else "the value given"))

    rule = "linear" if base_uplift is None else base_uplift
    B, u = geom.base_width, thrust["u_heel"]

    def uplift_words(name, U):
        if name == "linear":
            return (f"linear: gamma_w h_w = {u:.1f} kPa at the heel to 0 at "
                    f"the toe, U = gamma_w h_w B / 2 = {U:.1f} kN/m at 2B/3 "
                    f"= {2.0 * B / 3.0:.2f} m from the toe")
        return (f"uniform: gamma_w h_w = {u:.1f} kPa under the whole base, "
                f"U = gamma_w h_w B = {U:.1f} kN/m at B/2 = {B / 2.0:.2f} m")
    other, s_o, o_o = others["uplift"]
    out.append(choice(
        "water pressure under the base, uplift (base_uplift)",
        uplift_words(rule, s_terms["U"]) + f": {fos_s(s_terms)}, "
        f"{fos_o(o_terms)}",
        how_given(rule, "linear", name="base_uplift"),
        BASE_UPLIFT_BASIS[rule],
        [f"{uplift_words(other, s_o['U'])} (base_uplift '{other}'): "
         f"{fos_s(s_o)}, {fos_o(o_o)}, e = {o_o['eccentricity']:.3f} m. "
         + BASE_UPLIFT_BASIS[other]]))

    M_s, M_U = o_terms["M_stabilizing"], o_terms["M_uplift"]
    M_ot = o_terms["M_overturning"] - M_U
    if M_ot > 0 and M_U > 0:
        out.append(choice(
            "uplift moment about the toe, in the overturning check",
            f"an overturning moment: FOS_overturning = M_stab / (M_ot + U "
            f"x_U) = {M_s:.1f} / ({M_ot:.1f} + {M_U:.1f}) = "
            f"{M_s / (M_ot + M_U):.3g}", "module", UPLIFT_MOMENT_BASIS,
            [f"taken off the stabilizing moment: FOS_overturning = (M_stab "
             f"- U x_U) / M_ot = ({M_s:.1f} - {M_U:.1f}) / {M_ot:.1f} = "
             f"{(M_s - M_U) / M_ot:.3g}"]))

    flag = b_terms["uplift_in_bearing"]

    def bearing_words(t):
        words = (f"q_toe {t['q_toe']:.1f} kPa, q_heel {t['q_heel']:.1f} kPa, "
                 f"q_eq {t['q_eq']:.1f} kPa")
        if t["FOS"] is not None:
            words += f", FOS_bearing {t['FOS']:.3g}"
        return words
    out.append(choice(
        "uplift on the base, in the bearing pressures (uplift_in_bearing)",
        f"{'counted' if flag else 'left out'} (uplift_in_bearing={flag}): "
        + bearing_words(b_terms),
        how_given(flag, False, name="uplift_in_bearing"),
        UPLIFT_BEARING_BASIS[flag],
        [f"{'left out' if flag else 'counted'} (uplift_in_bearing="
         f"{not flag}): " + bearing_words(others["bearing"]) + ". "
         + UPLIFT_BEARING_BASIS[not flag]]))
    return out
