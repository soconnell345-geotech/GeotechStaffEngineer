"""
Earth pressure calculations for support of excavation design.

Implements classical Rankine/Coulomb coefficients (copied from sheet_pile module
per no-cross-module-import convention) plus Terzaghi-Peck apparent earth
pressure envelopes for braced/anchored excavations.

All units SI: kPa, kN/m³, degrees, meters.

References:
    Rankine (1857), Coulomb (1776)
    Terzaghi, K. & Peck, R.B. (1967) Soil Mechanics in Engineering Practice
    FHWA-IF-99-015, GEC-4: Ground Anchors and Anchored Systems
    Peck, R.B. (1969) Deep Excavation and Tunneling in Soft Ground, SOA Report
"""

import math
import warnings
from typing import List, Optional, Tuple

from geotech_common import log_spiral as _log_spiral


# ============================================================================
# Classical earth pressure coefficients (from sheet_pile/earth_pressure.py)
# ============================================================================

def rankine_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient Ka = tan²(45° - phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi_rad / 2) ** 2


def rankine_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient Kp = tan²(45° + phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def coulomb_Ka(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient."""
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

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

    The Caltrans Trenching and Shoring Manual's Eq. 4-3-24 (p. 4-13) with
    alpha = 90 deg - omega (omega the wall's slope from the vertical):
    Kp = cos^2(phi + omega) / (cos^2 omega cos(delta - omega) [1 -
    sqrt(sin(delta + phi) sin(phi + beta) / (cos(delta - omega) cos(beta -
    omega)))]^2). Until GM-290 the numerator was sin^2(alpha + phi), which
    is the same for a vertical wall only (sheet_pile's copy had it right).
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


# ----------------------------------------------------------------------------
# Caquot-Kerisel (1948) log-spiral passive coefficient
# (copied from sheet_pile/earth_pressure.py per no-cross-module-import convention)
# ----------------------------------------------------------------------------

# The chart's tables live in geotech_common.log_spiral (GM-183), shared with
# the sheet_pile copy; _lininterp stays for soe.stability.

def _lininterp(x, xs, ys):
    """Clamped piecewise-linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def caquot_kerisel_Kp(phi_deg: float, delta_deg: float = None,
                      Kp_initial: float = None,
                      beta_deg: float = 0.0) -> float:
    """Caquot-Kerisel (1948) log-spiral passive coefficient Kp' = R * Kp0.

    Kp0 is the passive coefficient at delta = phi, read off the Caquot-Kerisel
    chart as printed in the Caltrans Trenching and Shoring Manual (Fig. 4-20,
    p. 4-30; FHWA NHI-06-089 Fig. 10-10) for the ground slope beta in front
    of the wall (beta/phi -0.9 to +1); R = Kp(delta)/Kp(delta=phi) <= 1, the
    table printed on the same figure, reduces it for the wall friction.
    Unlike Coulomb (planar wedge), the log-spiral surface does not
    over-predict Kp at high delta/phi. For phi = 30 deg, delta/phi = 0.5,
    Kp' = 6.30 * 0.746 = 4.70 (Caltrans Ex 8-1); for phi = 32, beta = 27,
    delta = 14 deg, Kp0 ~= 19 and Kp' = 13 (the manual's p. 4-29 example).
    On level ground, below delta/phi = 0.1 R runs to the Rankine anchor
    R(0) = Kp_rankine(phi)/Kp0, so a smooth wall (delta = 0) returns the
    Rankine Kp. The tables and the other printed sources (UFC 3-220-20
    Fig. 4-13 and Eq. 4-12) are in ``geotech_common.log_spiral``.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default None -> delta = phi (R = 1.0).
    Kp_initial : float, optional
        Chart-read base Kp at delta = phi. If None, from the digitized chart.
    beta_deg : float, optional
        Ground slope in front of the wall (degrees, + rising away from the
        wall). Default 0, level ground.

    References
    ----------
    Caquot & Kerisel (1948); Caltrans T&S Manual Sec 4-6 / Fig 4-20 / Matrix
    4-1; FHWA NHI-06-089 Fig 10-10.
    """
    return _log_spiral.caquot_kerisel_1948_Kp(
        phi_deg, delta_deg, beta_deg=beta_deg, Kp_initial=Kp_initial)


def K0(phi_deg: float) -> float:
    """At-rest earth pressure coefficient (Jaky). K0 = 1 - sin(phi)."""
    return 1.0 - math.sin(math.radians(phi_deg))


def active_pressure(gamma: float, z: float, Ka: float,
                    c: float = 0.0, q_surcharge: float = 0.0) -> float:
    """Active lateral pressure: sigma_a = Ka*(gamma*z + q) - 2*c*sqrt(Ka)."""
    return Ka * (gamma * z + q_surcharge) - 2.0 * c * math.sqrt(Ka)


def passive_pressure(gamma: float, z: float, Kp: float,
                     c: float = 0.0) -> float:
    """Passive lateral pressure: sigma_p = Kp*gamma*z + 2*c*sqrt(Kp)."""
    return Kp * gamma * z + 2.0 * c * math.sqrt(Kp)


def tension_crack_depth(c: float, gamma: float, Ka: float,
                        q_surcharge: float = 0.0) -> float:
    """Depth of tension crack in cohesive soil behind active wall."""
    if c <= 0 or Ka <= 0 or gamma <= 0:
        return 0.0
    z_crack = (2.0 * c / math.sqrt(Ka) - q_surcharge) / gamma
    return max(z_crack, 0.0)


# ============================================================================
# Apparent earth pressure envelopes (Terzaghi & Peck 1967, FHWA GEC-4)
# ============================================================================

def apparent_pressure_sand(gamma: float, H: float, Ka: float) -> float:
    """Apparent earth pressure for braced excavations in sand.

    Returns the uniform (rectangular) envelope ordinate:
        p = 0.65 * Ka * gamma * H

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    Ka : float
        Active earth pressure coefficient.

    Returns
    -------
    float
        Uniform apparent pressure (kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 3.
    FHWA-IF-99-015, Section 5.2.1.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    return 0.65 * Ka * gamma * H


#: The stiff clay envelope's ordinate as a multiple of gamma H (GM-290).
#: UFC 3-220-20 Fig. 4-53 (p. 256), internally supported walls: "For N0 <=
#: 4, use: sigma_H2 = 0.4 gamma H. When movements are minimal and the
#: construction period is short, use: sigma_H1 = 0.2 gamma H"; the Caltrans
#: Trenching and Shoring Manual Eq. 8-3-2 (p. 8-5): 0.2 to 0.4 gamma H,
#: "best engineering judgment is needed to use the applicable factor between
#: 0.2 and 0.4"; GEC-4 Sec. 5.2.5 (Fig. 27, p. 56) the same range. Until
#: GM-290 the module interpolated 0.2 + 0.2 N/4, which none of them prints.
STIFF_CLAY_COEFFICIENT = 0.4
#: The printed range of that coefficient.
STIFF_CLAY_COEFFICIENT_RANGE = (0.2, 0.4)


def apparent_pressure_soft_clay(gamma: float, H: float, cu: float,
                                m: float = 1.0) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in soft to medium clay.

    Applies when stability number N = gamma*H/cu > 4.

    Terzaghi and Peck's envelope for soft to medium clay (GEC-4 Fig. 23c,
    p. 50; UFC 3-220-20 Fig. 4-52, p. 255, internally supported walls):
    zero at the top, rising over the top 0.25H to

        p = Ka_apparent * gamma * H,  Ka_apparent = 1 - m * (4*cu / (gamma*H))

    and constant over the lower 0.75H (shape "ramped_top"). m = 1.0 (Peck
    1969; GEC-4 Eq. 11, p. 57); GEC-4 sets m = 0.4 "when the excavation is
    underlain by deep soft clay and Ns exceeds 6" and says that this "may
    result in an underestimation of loads on the wall" (p. 57); UFC takes
    m = 0.4 F_BH there. Ka_apparent is at least 0.25: the Caltrans Trenching
    and Shoring Manual's Eq. 8-3-4 (p. 8-6), "not to be less than 0.25",
    which with no softer layer below the base (D = 0) is this equation with
    m = 1. Until GM-290 the envelope was uniform over the whole height and
    the floor was cited to "FHWA guidance".

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).
    m : float
        Empirical coefficient. Default 1.0.

    Returns
    -------
    tuple of (str, float)
        ("ramped_top", maximum pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969).
    FHWA-IF-99-015, Section 5.2.6, Eq. 11 and Fig. 23c; UFC 3-220-20 Fig.
    4-52; Caltrans T&S Manual Eq. 8-3-4.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")

    Ka_apparent = 1.0 - m * (4.0 * cu / (gamma * H))
    # Caltrans T&S Eq. 8-3-4 (p. 8-6): "not to be less than 0.25"
    Ka_apparent = max(Ka_apparent, 0.25)
    p = Ka_apparent * gamma * H
    return ("ramped_top", p)


def apparent_pressure_stiff_clay(gamma: float, H: float,
                                 cu: float,
                                 coefficient: float = STIFF_CLAY_COEFFICIENT
                                 ) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in stiff clay.

    Applies when stability number N = gamma*H/cu <= 4.

    Terzaghi and Peck's trapezoid for stiff fissured clay (UFC 3-220-20
    Fig. 4-53, p. 256, internally supported walls; GEC-4 Fig. 23b, p. 50):
    zero at the top, rising over the top 0.25H to

        p = coefficient * gamma * H,

    constant from 0.25H to 0.75H and falling to zero at H.
    ``coefficient`` is 0.4 by default and 0.2 to 0.4
    (:data:`STIFF_CLAY_COEFFICIENT`): UFC uses 0.4, and 0.2 "when movements
    are minimal and the construction period is short"; the Caltrans T&S
    Manual (Eq. 8-3-2, p. 8-5) leaves the factor between 0.2 and 0.4 to
    judgment.

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).
    coefficient : float
        The ordinate as a multiple of gamma H, 0.2 to 0.4. Default 0.4.

    Returns
    -------
    tuple of (str, float)
        ("trapezoidal", max_pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969).
    UFC 3-220-20 Fig. 4-53; FHWA-IF-99-015 Section 5.2.5; Caltrans T&S
    Manual Sec. 8-3.01.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")
    _check_stiff_coefficient(coefficient)
    p_max = coefficient * gamma * H
    return ("trapezoidal", p_max)


def _check_stiff_coefficient(coefficient: float) -> None:
    lo, hi = STIFF_CLAY_COEFFICIENT_RANGE
    if not lo <= coefficient <= hi:
        raise ValueError(
            f"stiff_clay_coefficient must be {lo:g} to {hi:g} (the stiff "
            "clay envelope's ordinate, 0.2 to 0.4 gamma H: UFC 3-220-20 Fig. "
            "4-53, Caltrans T&S Eq. 8-3-2), got " f"{coefficient:g}")


def select_apparent_pressure(soil_layers, H: float,
                             surcharge: float = 0.0,
                             gwt_depth: Optional[float] = None,
                             stiff_clay_coefficient: float =
                             STIFF_CLAY_COEFFICIENT) -> dict:
    """Auto-select apparent pressure diagram from soil profile.

    Determines the controlling soil type over the excavation depth and
    returns the appropriate apparent pressure envelope. The envelope itself
    carries no surcharge or water pressure; GEC-4 Sec 5.2.4 adds those
    explicitly (``soe.free_earth.apparent_pressure_profile`` does, for the
    braced-wall analysis).

    Clay: stiff fissured clay (N <= 4) takes the trapezoid of
    :func:`apparent_pressure_stiff_clay` at ``stiff_clay_coefficient``; soft
    to medium clay (N > 4) the envelope of :func:`apparent_pressure_soft_clay`,
    and for 4 < N < 6 the larger of its ordinate and the stiff clay
    ordinate: the Caltrans T&S Manual Sec. 8-3.02 (p. 8-6), "For soils with
    4 < Ns < 6, use the larger sigma_a from Equations 8-3-2 and 8-3-3" (GM-290;
    the shape stays the soft clay one). GEC-4 (p. 59) and UFC 3-220-20 Fig.
    4-52 (externally supported walls) take Ka = 0.22 for 4 < N <= 5.14
    instead, and Henkel's equation above it, which needs the depth to a firm
    layer below the base; the module does not offer them.

    Parameters
    ----------
    soil_layers : list of SOEWallLayer
        Soil layers from ground surface downward.
    H : float
        Excavation depth (m).
    surcharge : float
        Surface surcharge (kPa). Default 0. Not part of the envelope.
    gwt_depth : float, optional
        Water table depth (m). For the (effective-stress) sand envelope the
        average unit weight uses gamma - gamma_w below it; clay envelopes are
        total-stress and ignore it.
    stiff_clay_coefficient : float
        The stiff clay ordinate as a multiple of gamma H, 0.2 to 0.4.
        Default 0.4 (:data:`STIFF_CLAY_COEFFICIENT`).

    Returns
    -------
    dict
        Keys: "type" ("sand", "soft_clay", "stiff_clay"),
              "shape" ("uniform" for sand, "ramped_top" for soft clay,
              "trapezoidal" for stiff clay),
              "max_pressure_kPa" (peak ordinate),
              "stability_number" (gamma*H/cu for clay, None for sand);
              clay also "stiff_clay_coefficient", soft clay "Ka" (the
              apparent Ka) and, for 4 < N < 6, "stiff_clay_ordinate_kPa"
              and "governed_by" ("soft_clay" or "stiff_clay_ordinate").
    """
    _check_stiff_coefficient(stiff_clay_coefficient)
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if not soil_layers:
        raise ValueError("At least one soil layer is required")

    from geotech_common.water import GAMMA_W

    # Compute weighted averages over excavation depth
    total_gamma_h = 0.0
    total_gamma_eff_h = 0.0
    total_cu_h = 0.0
    total_phi_h = 0.0
    sand_thickness = 0.0
    clay_thickness = 0.0
    z_top = 0.0

    for layer in soil_layers:
        h = min(layer.thickness, H - z_top)
        if h <= 0:
            break
        total_gamma_h += layer.unit_weight * h
        wet = 0.0
        if gwt_depth is not None:
            wet = min(max(z_top + h - gwt_depth, 0.0), h)
        total_gamma_eff_h += layer.unit_weight * h - GAMMA_W * wet
        total_cu_h += layer.cohesion * h
        total_phi_h += layer.friction_angle * h
        if layer.soil_type == "sand":
            sand_thickness += h
        else:
            clay_thickness += h
        z_top += h

    gamma_avg = total_gamma_h / H
    cu_avg = total_cu_h / H
    phi_avg = total_phi_h / H

    # Determine controlling soil type
    if sand_thickness >= clay_thickness:
        # Predominantly sand: effective-stress envelope
        gamma_eff = total_gamma_eff_h / H
        Ka = rankine_Ka(phi_avg)
        p_max = apparent_pressure_sand(gamma_eff, H, Ka)
        return {
            "type": "sand",
            "shape": "uniform",
            "max_pressure_kPa": round(p_max, 2),
            "stability_number": None,
            "Ka": round(Ka, 4),
            "gamma_avg": round(gamma_eff, 2),
            "effective_stress": gwt_depth is not None and gwt_depth < H,
        }
    else:
        # Predominantly clay
        N = gamma_avg * H / cu_avg if cu_avg > 0 else float("inf")

        if N > 4:
            # Soft to medium clay
            shape, p_soft = apparent_pressure_soft_clay(gamma_avg, H, cu_avg)
            out = {
                "type": "soft_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_soft, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
                "Ka": round(p_soft / (gamma_avg * H), 4),
                "stiff_clay_coefficient": stiff_clay_coefficient,
            }
            if N < 6:
                # Caltrans T&S 8-3.02 (p. 8-6): for 4 < Ns < 6 the larger of
                # the stiff (Eq. 8-3-2) and soft (Eq. 8-3-3) ordinates
                p_stiff = stiff_clay_coefficient * gamma_avg * H
                out["stiff_clay_ordinate_kPa"] = round(p_stiff, 2)
                out["governed_by"] = ("stiff_clay_ordinate"
                                      if p_stiff > p_soft else "soft_clay")
                out["max_pressure_kPa"] = round(max(p_soft, p_stiff), 2)
            return out
        else:
            # Stiff clay
            shape, p_max = apparent_pressure_stiff_clay(
                gamma_avg, H, cu_avg, stiff_clay_coefficient
            )
            return {
                "type": "stiff_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_max, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
                "stiff_clay_coefficient": stiff_clay_coefficient,
            }


def _embedment_choices(FOS_embedment: float, Kp_given, Kp_used: float):
    """What the embedment below the (lowest) anchor took (GM-182; both
    routes since GM-290), as a result's ``choices``
    (``geotech_common.choices``)."""
    from geotech_common.choices import choice, how_given
    return [
        choice("embedment factor of safety", FOS_embedment,
               how_given(FOS_embedment, 1.3, name="FOS_embedment"),
               "embedment D from MR = FS x MD about the anchor (the lowest "
               "anchor with two or more), free earth support as in the "
               "Caltrans Trenching and Shoring Manual Sec. 8-4, which takes "
               "FS = 1.3 (Secs. 8-4.01 step 3 and 8-4.02 step 4; Example "
               "8-1); its Example 8-2 takes 1.5 (p. 8-31)"),
        choice("passive coefficient below the excavation",
               f"{Kp_used:.4f}" + ("" if Kp_given is not None
                                   else " (Rankine, from phi)"),
               how_given(Kp_given, None, name="Kp"),
               "Rankine Kp = tan^2(45 + phi/2), no wall friction, unless "
               "Kp is given",
               ["a log-spiral Kp with wall friction (give Kp)"]),
    ]


def fhwa_apparent_pressure_anchored_wall(
    H: float,
    anchor_depths: List[float],
    gamma: float,
    phi: float,
    surcharge: float = 0.0,
    spacing: float = 1.0,
    inclination_deg: float = 0.0,
    Ka: float = None,
    Kp: float = None,
    FOS_embedment: float = 1.3,
) -> dict:
    """FHWA/GEC-4 apparent-pressure anchored-wall design (tributary method).

    Builds the FHWA apparent earth-pressure envelope for a multi-level anchored
    wall in sand and distributes it to the anchors by the tributary-area (hinge)
    method — the GEC-4 (FHWA-IF-99-015) Appendix A procedure.

    The maximum apparent-pressure ordinate is

        pe = 0.65 * Ka * gamma * H^2 / (H - H1/3 - Hn+1/3)

    where H1 = depth to the topmost anchor and Hn+1 = height from the lowest
    anchor to the base. (The total apparent earth load is
    PT = 0.65*Ka*gamma*H^2 = 1.3 * (1/2 Ka gamma H^2), i.e. 1.3x the triangular
    Rankine total.) A uniform surcharge adds ps = Ka*q over the full height.

    Tributary horizontal anchor loads (per unit width of wall):
      - top anchor:     (2/3 H1 + H2/2) pe + (H1 + H2/2) ps
      - interior anchor: (Hi/2 + Hi+1/2) (pe + ps)
      - bottom anchor:  (Hn/2 + 23/48 Hn+1) pe + (Hn/2 + Hn+1/2) ps
      - subgrade reaction R = 3/16 Hn+1 pe + Hn+1/2 ps
    Moments (GEC-4 Fig. 39(b), p. 80, with the surcharge as App. A Example
    1 adds it, p. A-6): at the upper anchor M1 = (13/54) H1^2 pe + ps H1^2/2
    (the surcharge's uniform pressure on the cantilever); below it
    (1/10) L^2 (pe + ps) for each span L. Until GM-290 M1 was
    (13/54) H1^2 (pe + ps), 70.4 kN-m/m in the example against the printed
    76. Anchor design loads DL = TH * spacing / cos(incl).

    Validated vs GEC-4 Design Example 1 (2-anchor, SI): pe=43.6, ps=3.2,
    TH1=168, TH2=172, R=37, M1=76 (M2,3=66) kN-m/m, DL1=435, DL2=445 kN.

    The embedment below the lowest anchor (two or more anchors, GM-193,
    GM-290): free earth support about the lowest anchor, the Caltrans
    Trenching and Shoring Manual Sec. 8-4.02 hinge method, steps 3 and 4
    (p. 8-9): the trapezoid (+ ps) from the lowest anchor to the excavation
    and the Rankine active pressure below it against the passive pressure
    ``Kp``, moments about the lowest anchor; D from MR = FOS_embedment x MD,
    D' from MR = MD, and that body's load on the lowest anchor and moment.
    The tributary area method itself carries the load below the lowest
    anchor as R and works out no embedment; its anchor loads and moments
    are kept, and the free earth body's moment is reported beside them.

    SINGLE anchor (n=1): the tributary formulas do not apply, so the wall is
    solved by free earth support about the anchor, as in the Caltrans
    Trenching and Shoring Manual Sec 8-4 / Example 8-1 (``soe.free_earth``):
    the trapezoid (+ ps) above the excavation, Rankine active pressure below
    it against passive pressure ``Kp`` (Rankine by default; pass a log-spiral
    value), embedment D from MR = FOS_embedment x MD, the anchor load from
    MR = MD and horizontal equilibrium, and the wall moment from that body.
    Reproduces Example 8-1 (D = 6.09 ft, T1 = 14,254 lb/ft, M = 22,494
    ft-lb/ft). Until 2026-09-15 the n=1 result carried only the upper
    tributary load and the moment above the anchor, unlabelled as partial
    (field feedback N3). The soil below the excavation is taken as the same
    gamma/phi as the retained soil, dry.

    Parameters
    ----------
    H : float
        Total wall height (m).
    anchor_depths : list of float
        Depths of the anchor levels below the top of wall (m).
    gamma : float
        Retained-soil unit weight (kN/m3).
    phi : float
        Retained-soil friction angle (deg).
    surcharge : float, optional
        Uniform surface surcharge q (kPa). Default 0.
    spacing : float, optional
        Horizontal anchor spacing (m) for the design loads. Default 1.0.
    inclination_deg : float, optional
        Anchor inclination from horizontal (deg). Default 0.
    Ka : float, optional
        Active coefficient. If None, Rankine Ka(phi).
    Kp : float, optional
        Passive coefficient below the excavation, for the embedment. If
        None, Rankine Kp(phi).
    FOS_embedment : float, optional
        MR = FOS x MD for the embedment about the (lowest) anchor. Default
        1.3 (Caltrans T&S Manual).

    Returns
    -------
    dict
        {Ka, pe_kPa, ps_kPa, PT_total_kN_per_m, anchors[], subgrade_reaction_kN_per_m,
         max_moment_kN_m_per_m, ...}. Each anchors[] entry: {depth_m,
         TH_kN_per_m (horizontal load), design_load_kN}. For n=1 also
         TH_upper/TH_lower, max_moment_depth_m and
         subgrade_reaction_kN_per_m = None (the passive pressure below the
         excavation takes the base load). Both routes: embedment_D_m,
         embedment_D_FS1_m, converged, notes, choices (the embedment FS and
         Kp, given or default; GM-182); two or more anchors also
         lowest_anchor_load_from_below_kN_per_m and
         free_earth_max_moment_kN_m_per_m (with its depth). With no moment
         balance within 6H (``converged`` False) the depths that have no
         root, and (single anchor) the anchor load, design load and moment
         when MR = MD has none, are None and the first note says why.

    References
    ----------
    FHWA-IF-99-015 (GEC-4), Ground Anchors and Anchored Systems, App. A;
    Caltrans T&S Manual Ch. 8.
    """
    if H <= 0:
        raise ValueError("Wall height H must be positive")
    if not anchor_depths:
        raise ValueError("At least one anchor depth is required")
    d = sorted(float(x) for x in anchor_depths)
    if d[0] <= 0 or d[-1] >= H:
        raise ValueError("Anchor depths must be between 0 and H")
    n = len(d)
    if Ka is None:
        Ka = rankine_Ka(phi)

    H1 = d[0]
    Hn1 = H - d[-1]
    denom = H - H1 / 3.0 - Hn1 / 3.0
    pe = 0.65 * Ka * gamma * H ** 2 / denom
    ps = Ka * surcharge
    PT_total = 0.65 * Ka * gamma * H ** 2

    # Spans between successive supports: span[i] = d[i] - d[i-1] (i>=1).
    cos_incl = math.cos(math.radians(inclination_deg))
    anchors = []
    moments = []

    # The free earth body below the excavation (both routes): the trapezoid
    # (+ ps) above the excavation, Rankine active pressure below it against
    # the passive pressure Kp, soil below as the retained soil, dry.
    import numpy as np
    from soe.free_earth import LayeredProfile, solve_about_support
    from soe.geometry import SOEWallLayer

    if FOS_embedment <= 0:
        raise ValueError("FOS_embedment must be positive")
    Kp_used = rankine_Kp(phi) if Kp is None else float(Kp)
    prof = LayeredProfile(
        [SOEWallLayer(thickness=max(10.0 * H, 50.0), unit_weight=gamma,
                      friction_angle=phi)],
        H, surcharge=surcharge, Ka=Ka, Kp=Kp_used)
    top_len = (2.0 / 3.0) * H1
    bot_start = H - (2.0 / 3.0) * Hn1

    def above(z):
        z = np.asarray(z, dtype=float)
        env = np.where(z < top_len, pe * z / top_len,
                       np.where(z <= bot_start, pe,
                                pe * (H - z) / (H - bot_start)))
        return np.where((z >= 0.0) & (z <= H), env + ps, 0.0)

    def _r(x, nd):
        return None if x is None else round(x, nd)

    single = {}
    if n == 1:
        # Single anchor: free earth support about the anchor (Caltrans T&S 8-4,
        # Example 8-1) -- the tributary formulas below need two or more levels.
        sol = solve_about_support(prof, pivot=d[0], body_top=0.0,
                                  FS=FOS_embedment, above=above)

        TH_upper = (2.0 / 3.0) * H1 * pe + H1 * ps
        TH = sol.support_load  # None with no moment balance at FS = 1
        anchors.append({
            "depth_m": round(d[0], 3),
            "TH_upper_kN_per_m": round(TH_upper, 3),
            "TH_lower_kN_per_m": None if TH is None else round(TH - TH_upper, 3),
            "TH_kN_per_m": _r(TH, 3),
            "design_load_kN": (None if TH is None
                               else round(TH * spacing / cos_incl, 2)),
        })
        moments.append(sol.max_moment)
        single = {
            "single_anchor_method": "free earth support about the anchor "
                                    "(Caltrans T&S Manual 8-4, Example 8-1)",
            "Kp": round(Kp_used, 4),
            "FOS_embedment": FOS_embedment,
            "embedment_D_m": _r(sol.D, 3),
            "embedment_D_FS1_m": _r(sol.D_prime, 3),
            "max_moment_depth_m": _r(sol.max_moment_depth, 3),
            "zero_shear_depth_m": _r(sol.zero_shear_depth, 3),
            "converged": sol.converged,
            "notes": list(sol.notes),
            "choices": _embedment_choices(FOS_embedment, Kp, Kp_used),
        }
    else:
        for i in range(n):
            if i == 0:
                H2 = d[1] - d[0]
                TH = (2.0 / 3.0 * H1 + H2 / 2.0) * pe + (H1 + H2 / 2.0) * ps
            elif i == n - 1:
                Hn = d[i] - d[i - 1]
                TH = (Hn / 2.0 + 23.0 / 48.0 * Hn1) * pe + (Hn / 2.0 + Hn1 / 2.0) * ps
            else:
                Hi = d[i] - d[i - 1]
                Hi1 = d[i + 1] - d[i]
                TH = (Hi / 2.0 + Hi1 / 2.0) * (pe + ps)
            anchors.append({
                "depth_m": round(d[i], 3),
                "TH_kN_per_m": round(TH, 3),
                "design_load_kN": round(TH * spacing / cos_incl, 2),
            })
        # Moments, GEC-4 Fig. 39(b) with App. A Example 1's surcharge term
        # (p. A-6): M1 at the upper anchor, pL^2/10 below it.
        moments.append((13.0 / 54.0) * H1 ** 2 * pe + ps * H1 ** 2 / 2.0)
        for i in range(1, n):
            span = d[i] - d[i - 1]
            moments.append((1.0 / 10.0) * span ** 2 * (pe + ps))
        moments.append((1.0 / 10.0) * Hn1 ** 2 * (pe + ps))

        # The embedment below the lowest anchor (GM-193): free earth support
        # about it, Caltrans T&S 8-4.02 hinge method steps 3-4 (p. 8-9).
        sol = solve_about_support(prof, pivot=d[-1], body_top=d[-1],
                                  FS=FOS_embedment, above=above)
        notes = list(sol.notes)
        if sol.max_moment is not None and sol.max_moment > max(moments):
            embedded = sol.max_moment_depth > H
            notes.append(
                f"The free earth body below the lowest anchor (FS = 1, "
                f"Caltrans T&S 8-4.02) reaches {sol.max_moment:.1f} kN-m/m at "
                f"z = {sol.max_moment_depth:.2f} m ("
                + ("in the embedded portion" if embedded else
                   "above the excavation base")
                + f"), more than the tributary area method's "
                f"{max(moments):.1f} kN-m/m (GEC-4 Fig. 39, whose span below "
                "the lowest anchor bears on the subgrade reaction R); "
                "max_moment_kN_m_per_m is the tributary one."
                + (" GEC-4 Sec. 5.4.1 (p. 79): for walls that penetrate deep "
                   "deposits of weak material, the maximum bending moment "
                   "may occur in the embedded portion." if embedded else ""))
        single = {
            "embedment_method": "free earth support about the lowest anchor "
                                "(Caltrans T&S Manual 8-4.02, hinge method "
                                "steps 3-4)",
            "Kp": round(Kp_used, 4),
            "FOS_embedment": FOS_embedment,
            "embedment_D_m": _r(sol.D, 3),
            "embedment_D_FS1_m": _r(sol.D_prime, 3),
            "lowest_anchor_load_from_below_kN_per_m": _r(sol.support_load, 3),
            "free_earth_max_moment_kN_m_per_m": _r(sol.max_moment, 3),
            "free_earth_max_moment_depth_m": _r(sol.max_moment_depth, 3),
            "converged": sol.converged,
            "notes": notes,
            "choices": _embedment_choices(FOS_embedment, Kp, Kp_used),
        }

    R = None if n == 1 else (3.0 / 16.0) * Hn1 * pe + (Hn1 / 2.0) * ps

    out = {
        "Ka": round(Ka, 4),
        "pe_kPa": round(pe, 3),
        "ps_kPa": round(ps, 3),
        "PT_total_kN_per_m": round(PT_total, 2),
        "H1_m": round(H1, 3),
        "Hn1_m": round(Hn1, 3),
        "n_anchors": n,
        "anchors": anchors,
        "subgrade_reaction_kN_per_m": None if R is None else round(R, 3),
        # None for a single anchor with no moment balance (see notes)
        "max_moment_kN_m_per_m": (None if None in moments
                                  else round(max(moments), 3)),
        "spacing_m": spacing,
        "inclination_deg": inclination_deg,
    }
    out.update(single)
    return out


def get_pressure_at_depth(z: float, H: float, shape: str,
                          p_max: float) -> float:
    """Return apparent pressure ordinate at depth z for given envelope shape.

    Parameters
    ----------
    z : float
        Depth from top of wall (m).
    H : float
        Total excavation depth (m).
    shape : str
        "uniform", "trapezoidal" (stiff clay: 0.25H ramps at both ends) or
        "ramped_top" (soft clay: a 0.25H ramp at the top, GM-290).
    p_max : float
        Maximum pressure ordinate (kPa).

    Returns
    -------
    float
        Apparent pressure at depth z (kPa).
    """
    if z < 0 or z > H:
        return 0.0

    if shape == "uniform":
        return p_max

    elif shape == "trapezoidal":
        # Trapezoidal: ramps from 0 to p_max over top 0.25H,
        # constant from 0.25H to 0.75H, ramps to 0 at H.
        if z <= 0.25 * H:
            return p_max * (z / (0.25 * H)) if H > 0 else 0.0
        elif z <= 0.75 * H:
            return p_max
        else:
            return p_max * ((H - z) / (0.25 * H)) if H > 0 else 0.0

    elif shape == "ramped_top":
        # Soft to medium clay: ramps from 0 to p_max over the top 0.25H,
        # constant below (GEC-4 Fig. 23c; UFC 3-220-20 Fig. 4-52).
        if z <= 0.25 * H:
            return p_max * (z / (0.25 * H)) if H > 0 else 0.0
        return p_max

    return p_max  # fallback
