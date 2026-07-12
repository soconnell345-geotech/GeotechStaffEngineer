"""
MSE (Mechanically Stabilized Earth) wall analysis.

Implements external stability (sliding, overturning, bearing) and
internal stability (tensile rupture, pullout) per FHWA GEC-11.

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-11 (Berg, Christopher & Samtani 2009), a TWO-VOLUME publication:
      - Vol I  = FHWA-NHI-10-024 — design method, Chapters 4-5 (in the in-house
                 geotech-references `gec_11` module, ch 1-11).
      - Vol II = FHWA-NHI-10-025 — appendices incl. Appendix E worked example
                 E4 (the external-stability validation anchor). Vol II is NOT in
                 the in-house reference module.
    Both citations below are correct — they name the two volumes, not a typo.
    AASHTO LRFD Section 11.10

    Source basis (Kr/K0 ratio and F* pullout curves): DIGITIZED from GEC-11
    Figs 4-10/4-11/E4-5 (also AASHTO Fig. 11.10.6.2.1-3). Authoring-time in-hand
    status not recorded, but the curves are anchored by the GEC-11 Example E4
    reproduction (CDRs and F* within ~3%; see tests). The reinforcement PRODUCT
    constants (W11 grid geometry, 75x4 mm strip T_allowable) are representative
    values, NOT a specific manufacturer catalog — see DESIGN.md.
"""

import math
from typing import Dict, Any, List

from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.earth_pressure import rankine_Ka, horizontal_force_active
from retaining_walls.reinforcement import Reinforcement
from retaining_walls.results import MSEWallResult


# 20 ft = 6.096 m: the GEC-11 / AASHTO depth below which Kr/Ka and F* become
# constant for inextensible (metallic) reinforcement.
_Z20_FT_M = 20.0 * 0.3048  # 6.096 m


def Kr_Ka_ratio(z: float, reinforcement_type: str = "metallic") -> float:
    """Ratio Kr/Ka as a function of depth per GEC-11 / AASHTO.

    Two inextensible (metallic) lateral-stress-ratio curves, plus geosynthetic:

    - ribbed metallic STRIP ("metallic" / "metallic_strip", DEFAULT):
      1.7 at z=0 -> 1.2 at z>=6 m, constant below (GEC-11 Fig. 4-10).
    - steel bar-mat / welded-grid ("bar_mat" / "welded_grid" /
      "metallic_grid"): 2.5 at z=0 -> 1.2 at z>=20 ft (6.096 m), constant below
      (GEC-11 Fig. E4-5 / AASHTO Fig. 11.10.6.2.1-3).
    - geosynthetic / extensible ("geosynthetic"): constant 1.0.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    reinforcement_type : str, optional
        "metallic"/"metallic_strip" (ribbed strip, DEFAULT),
        "bar_mat"/"welded_grid"/"metallic_grid" (steel grid),
        or "geosynthetic".

    Returns
    -------
    float
        Kr/Ka ratio.

    References
    ----------
    FHWA GEC-11, Figures 4-10 and E4-5; AASHTO LRFD Fig. 11.10.6.2.1-3
    """
    if reinforcement_type == "geosynthetic":
        return 1.0

    if reinforcement_type in ("bar_mat", "welded_grid", "metallic_grid"):
        # Steel bar mat / welded grid: 2.5 at z=0 -> 1.2 at z=20 ft (6.096 m)
        if z <= 0:
            return 2.5
        elif z >= _Z20_FT_M:
            return 1.2
        else:
            return 2.5 - (2.5 - 1.2) / _Z20_FT_M * z

    # Ribbed metallic strip (default): linear 1.7 at z=0 to 1.2 at z=6m
    if z <= 0:
        return 1.7
    elif z >= 6.0:
        return 1.2
    else:
        return 1.7 - (0.5 / 6.0) * z


def F_star_metallic(z: float, phi_backfill: float = 34.0,
                    reinforcement_type: str = "metallic",
                    t_over_St: float = None) -> float:
    """Pullout friction-bearing factor F* for inextensible reinforcement.

    Two curves per GEC-11 Figure 4-11 / E4-5:

    - ribbed metallic STRIP ("metallic" / "metallic_strip", DEFAULT):
      F* = 2.0 at z=0 -> tan(phi) at z>=6 m (linear interpolation).
    - steel bar-mat / welded-grid ("bar_mat" / "welded_grid" /
      "metallic_grid"): F* = 20·(t/St) at z=0 -> 10·(t/St) at z>=20 ft
      (6.096 m), where t = transverse-bar diameter and St = transverse-bar
      (grid) spacing. ``t_over_St`` is required for the grid curve.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    phi_backfill : float, optional
        Backfill friction angle (degrees). Default 34. (Strip curve only.)
    reinforcement_type : str, optional
        "metallic"/"metallic_strip" (ribbed strip, DEFAULT) or
        "bar_mat"/"welded_grid"/"metallic_grid" (steel grid).
    t_over_St : float, optional
        Transverse-bar diameter / spacing ratio (unit-free) for the steel-grid
        curve. Required when ``reinforcement_type`` is a grid type.

    Returns
    -------
    float
        Pullout resistance factor F*.

    References
    ----------
    FHWA GEC-11, Figure 4-11 (ribbed steel strips) and Figure E4-5 (bar mats)
    """
    if reinforcement_type in ("bar_mat", "welded_grid", "metallic_grid"):
        if t_over_St is None:
            raise ValueError(
                "F_star_metallic: a steel-grid (bar_mat) F* curve requires "
                "t_over_St = transverse-bar diameter / spacing."
            )
        F_surface = 20.0 * t_over_St
        F_deep = 10.0 * t_over_St
        if z <= 0:
            return F_surface
        elif z >= _Z20_FT_M:
            return F_deep
        else:
            return F_deep + (_Z20_FT_M - z) * (F_surface - F_deep) / _Z20_FT_M

    F_star_surface = 2.0
    # At depth >= 6m, F* = tan(phi) for ribbed metallic strips (GEC-11 Fig 4-11)
    F_star_deep = math.tan(math.radians(phi_backfill))

    if z <= 0:
        return F_star_surface
    elif z >= 6.0:
        return F_star_deep
    else:
        return F_star_surface - (F_star_surface - F_star_deep) / 6.0 * z


def Tmax_at_level(z: float, gamma_backfill: float, Ka: float,
                  Kr_Ka: float, Sv: float,
                  q_surcharge: float = 0.0) -> float:
    """Maximum tensile force in reinforcement at depth z.

    sigma_h = Kr * sigma_v = (Kr/Ka) * Ka * (gamma*z + q)
    Tmax = sigma_h * Sv

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    Ka : float
        Active earth pressure coefficient of backfill.
    Kr_Ka : float
        Kr/Ka ratio at this depth.
    Sv : float
        Vertical spacing of reinforcement (m).
    q_surcharge : float, optional
        Surcharge (kPa). Default 0.

    Returns
    -------
    float
        Maximum tensile force Tmax (kN/m width).
    """
    sigma_v = gamma_backfill * z + q_surcharge
    sigma_h = Kr_Ka * Ka * sigma_v
    return sigma_h * Sv


def pullout_resistance(z: float, gamma_backfill: float,
                       Le: float, F_star: float,
                       alpha_pullout: float = 1.0,
                       C: float = 2.0,
                       q_surcharge: float = 0.0,
                       Rc: float = 1.0) -> float:
    """Pullout resistance of reinforcement at depth z.

    Pr = F* * alpha * sigma_v' * Le * C * Rc

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    Le : float
        Effective length beyond failure surface (m).
    F_star : float
        Pullout resistance factor.
    alpha_pullout : float, optional
        Scale effect correction factor. Default 1.0.
    C : float, optional
        Overall reinforcement surface area geometry factor.
        C=2 for strips/grids (top and bottom surface). Default 2.
    q_surcharge : float, optional
        Surcharge (kPa). Default 0.
    Rc : float, optional
        Coverage ratio (b/Sh for strips, 1.0 for continuous grids).
        Default 1.0.

    Returns
    -------
    float
        Pullout resistance Pr (kN/m width).
    """
    sigma_v = gamma_backfill * z + q_surcharge
    return F_star * alpha_pullout * sigma_v * Le * C * Rc


def check_internal_stability(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    reinforcement: Reinforcement,
    FOS_pullout: float = 1.5,
    FOS_rupture: float = 1.0,
) -> List[Dict[str, Any]]:
    """Check internal stability at each reinforcement level.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry with reinforcement layout.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    reinforcement : Reinforcement
        Reinforcement properties.
    FOS_pullout : float, optional
        Required FOS for pullout. Default 1.5.
    FOS_rupture : float, optional
        Required FOS for rupture. Default 1.0.

    Returns
    -------
    list of dict
        Per-level results.
    """
    Ka = rankine_Ka(phi_backfill)
    H = geom.wall_height
    L = geom.reinforcement_length
    Sv = geom.reinforcement_spacing
    # Select the depth-variation curve family by reinforcement type:
    #  - steel bar mat / welded grid -> bar-mat curves (Kr/Ka 2.5->1.2,
    #    F* 20(t/St)->10(t/St))
    #  - ribbed metallic strip       -> strip curves (Kr/Ka 1.7->1.2,
    #    F* 2.0->tan-phi)  [unchanged default]
    #  - geosynthetic                -> Kr/Ka 1.0, F* 0.67*tan-phi
    if reinforcement.is_grid:
        r_type = "metallic_grid"
    elif reinforcement.is_metallic:
        r_type = "metallic"
    else:
        r_type = "geosynthetic"
    t_over_St = reinforcement.t_over_St if reinforcement.is_grid else None

    results = []
    for z in geom.reinforcement_depths:
        # Kr/Ka ratio
        kr_ka = Kr_Ka_ratio(z, r_type)

        # Tmax at this level
        T = Tmax_at_level(z, gamma_backfill, Ka, kr_ka, Sv, geom.surcharge)

        # Effective length Le = total length - active zone length La.
        # GEC-11 / AASHTO Fig. 11.10.6.3.1-1: the failure surface depends
        # on reinforcement extensibility (RW-2):
        #  - Inextensible (metallic): bilinear coherent-gravity surface —
        #    vertical at 0.3H from the face over the upper half of the
        #    wall, then tapering linearly to the toe:
        #       La = 0.3*H        for z <= H/2
        #       La = 0.6*(H - z)  for z >  H/2
        #  - Extensible (geosynthetic): Rankine wedge at 45+phi/2:
        #       La = (H - z)*tan(45 - phi/2)
        if reinforcement.is_metallic:
            La = 0.3 * H if z <= H / 2.0 else 0.6 * (H - z)
        else:
            La = (H - z) * math.tan(math.radians(45 - phi_backfill / 2))
        Le = max(L - La, 1.0)  # minimum 1m effective length

        # Pullout resistance
        if reinforcement.is_grid and t_over_St is not None:
            # Steel bar-mat / welded-grid F* = 20(t/St) -> 10(t/St)
            F_star = F_star_metallic(z, phi_backfill, "metallic_grid", t_over_St)
        elif reinforcement.is_metallic:
            # Ribbed strip F* = 2.0 -> tan(phi) (also the fallback for a grid
            # whose transverse geometry t/St was not supplied)
            F_star = F_star_metallic(z, phi_backfill)
        else:
            F_star = 0.67 * math.tan(math.radians(phi_backfill))

        Pr = pullout_resistance(
            z, gamma_backfill, Le, F_star,
            q_surcharge=geom.surcharge,
            Rc=reinforcement.coverage_ratio,
        )

        # FOS checks
        FOS_po = Pr / T if T > 0 else 99.9
        FOS_ru = reinforcement.Tallowable / T if T > 0 else 99.9

        passes = (FOS_po >= FOS_pullout) and (FOS_ru >= FOS_rupture)

        results.append({
            "depth_m": round(z, 2),
            "Tmax_kN_per_m": round(T, 2),
            "Pr_kN_per_m": round(Pr, 2),
            "La_m": round(La, 2),
            "Le_m": round(Le, 2),
            "Kr_Ka": round(kr_ka, 3),
            "F_star": round(F_star, 3),
            "FOS_pullout": round(FOS_po, 3),
            "FOS_rupture": round(FOS_ru, 3),
            "Tallowable_kN_per_m": reinforcement.Tallowable,
            "passes": passes,
        })

    return results


def check_external_stability(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    gamma_foundation: float,
    phi_foundation: float,
    c_foundation: float = 0.0,
    q_allowable: float = None,
    FOS_sliding_req: float = 1.5,
    FOS_overturning_req: float = 2.0,
    phi_retained: float = None,
    gamma_retained: float = None,
) -> Dict[str, Any]:
    """Check external stability of MSE wall as rigid block.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry.
    gamma_backfill : float
        Reinforced fill unit weight (kN/m³).
    phi_backfill : float
        Reinforced fill friction angle (degrees).
    gamma_foundation : float
        Foundation soil unit weight (kN/m³).
    phi_foundation : float
        Foundation friction angle (degrees).
    c_foundation : float, optional
        Foundation cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing (kPa). Default None.
    FOS_sliding_req : float, optional
        Required sliding FOS. Default 1.5.
    FOS_overturning_req : float, optional
        Required overturning FOS. Default 2.0.
    phi_retained : float, optional
        Retained soil friction angle (degrees). If None, uses phi_backfill.
    gamma_retained : float, optional
        Retained soil unit weight (kN/m³). If None, uses gamma_backfill.

    Returns
    -------
    dict
    """
    H = geom.wall_height
    L = geom.reinforcement_length

    # Use retained fill properties for external active pressure if provided
    phi_ext = phi_retained if phi_retained is not None else phi_backfill
    gamma_ext = gamma_retained if gamma_retained is not None else gamma_backfill

    Ka = rankine_Ka(phi_ext)

    # Active force from retained soil behind reinforced zone
    Pa, z_Pa = horizontal_force_active(gamma_ext, H, Ka, q=geom.surcharge)

    # Weight of reinforced block (uses reinforced fill properties)
    W = gamma_backfill * H * L + geom.surcharge * L

    # Sliding: soil-on-soil interface, use min(phi_backfill, phi_foundation)
    # per GEC-11 Section 4.3 (no 2/3 reduction for soil-on-soil)
    delta_b = min(phi_backfill, phi_foundation)
    ca = 2.0 / 3.0 * c_foundation
    R_sliding = W * math.tan(math.radians(delta_b)) + ca * L
    FOS_sliding = R_sliding / Pa if Pa > 0 else 99.9

    # Overturning about toe
    M_stab = W * L / 2.0
    M_over = Pa * z_Pa
    FOS_overturning = M_stab / M_over if M_over > 0 else 99.9

    # Bearing — Meyerhof uniform pressure over the effective width
    # (AASHTO 11.10.5.4 / GEC-11 Eq. 4-13): sigma_v = W / (L - 2e).
    # This replaces the trapezoidal q_toe = (W/L)(1 + 6e/L) convention,
    # which is not the AASHTO/GEC-11 MSE convention (RW-3).
    x_R = (M_stab - M_over) / W if W > 0 else L / 2.0
    e = L / 2.0 - x_R
    L_eff = L - 2.0 * abs(e)
    if L_eff > 0:
        sigma_v = W / L_eff
    else:
        sigma_v = 1.0e6  # resultant off the base — bearing effectively fails

    FOS_bearing = q_allowable / sigma_v if (q_allowable and sigma_v > 0) else 99.9

    passes = (
        FOS_sliding >= FOS_sliding_req
        and FOS_overturning >= FOS_overturning_req
        and (FOS_bearing >= 1.0 if q_allowable else True)
    )

    return {
        "FOS_sliding": round(FOS_sliding, 3),
        "FOS_overturning": round(FOS_overturning, 3),
        "FOS_bearing": round(FOS_bearing, 3),
        # Meyerhof uniform bearing pressure over effective width L - 2e
        # (key name kept for backward compatibility)
        "q_toe_kPa": round(sigma_v, 1),
        "sigma_v_kPa": round(sigma_v, 1),
        "L_eff_m": round(max(L_eff, 0.0), 3),
        "eccentricity_m": round(e, 3),
        "passes": passes,
    }


def check_external_stability_lrfd(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    phi_foundation: float,
    *,
    phi_retained: float = None,
    gamma_retained: float = None,
    live_load: float = None,
    bearing_resistance_strength: float = None,
    bearing_resistance_service: float = None,
    c_foundation: float = 0.0,
    gamma_EV_max: float = 1.35,
    gamma_EV_min: float = 1.00,
    gamma_EH_max: float = 1.50,
    gamma_EH_min: float = 0.90,
    gamma_LL: float = 1.75,
    phi_sliding: float = 1.0,
    ecc_limit_ratio: float = 0.25,
) -> Dict[str, Any]:
    """AASHTO/GEC-11 LRFD external stability for an MSE wall (rigid block).

    Computes the full capacity:demand ratio (CDR) set — sliding, limiting
    eccentricity (overturning), and bearing — under the AASHTO Strength I
    (max and min load-factor) and Service I load combinations, with the GEC-11
    load-side bookkeeping built in:

    * The **live-load surcharge is EXCLUDED from the resisting side** for
      sliding and eccentricity (a vertical LL over the mass is destabilizing
      only through its horizontal thrust, so counting its weight would be
      unconservative) and **INCLUDED for bearing** (it raises the bearing
      stress).
    * Each mode is checked for the governing load combination. The **critical**
      ("min-vertical") combination pairs the minimum vertical load factor
      (least resistance / largest eccentricity, ``gamma_EV_min`` on all vertical
      gravity loads) with the maximum horizontal/driving factors
      (``gamma_EH_max``, ``gamma_LL``).

    This is the LRFD counterpart to the ASD ``check_external_stability``; it does
    NOT change that function. It reproduces the GEC-11 Example E4 (FHWA-NHI-10-025,
    Appendix E) external-stability results (sliding CDR 1.85/2.08/1.37,
    eccentricity eL 2.87/3.87 ft vs L/4, bearing sigma_v 6.70 ksf / CDR 1.57,
    Service sigma_v 4.66 ksf).

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry. ``geom.surcharge`` is the live-load surcharge unless
        ``live_load`` overrides it.
    gamma_backfill, phi_backfill : float
        Reinforced fill unit weight (kN/m3) and friction angle (deg).
    phi_foundation : float
        Foundation friction angle (deg); the sliding interface uses
        min(phi_backfill, phi_foundation).
    phi_retained, gamma_retained : float, optional
        Retained (behind-the-mass) fill properties for the external active
        thrust. Default to the reinforced-fill values.
    live_load : float, optional
        Live-load surcharge q (kPa). Default: ``geom.surcharge``.
    bearing_resistance_strength : float, optional
        Factored bearing resistance qR at the strength limit (kPa) = phi_b*qn.
        If given, the strength/critical bearing CDR = qR / sigma_v is reported.
    bearing_resistance_service : float, optional
        Service bearing resistance (kPa). If given, the Service I bearing CDR is
        reported.
    c_foundation : float, optional
        Foundation cohesion (kPa) for sliding adhesion (2/3 c). Default 0.
    gamma_EV_max, gamma_EV_min : float, optional
        Vertical earth (EV) load factors. Defaults 1.35 / 1.00 (AASHTO Str I).
    gamma_EH_max, gamma_EH_min : float, optional
        Horizontal earth (EH) load factors. Defaults 1.50 / 0.90.
    gamma_LL : float, optional
        Live-load (LL) factor. Default 1.75.
    phi_sliding : float, optional
        Sliding resistance factor. Default 1.0 (AASHTO 11.10.5.3).
    ecc_limit_ratio : float, optional
        Limiting-eccentricity ratio e_max/L. Default 0.25 (L/4, MSE on soil;
        use 1/6 for foundations on rock).

    Returns
    -------
    dict
        Nested {sliding, eccentricity, bearing} CDR/eL/sigma_v results (SI:
        kN/m, kN-m/m, kPa, m), plus the unfactored load components and the load
        factors used. All ratios are capacity:demand (>=1.0 passes).
    """
    H = geom.wall_height
    L = geom.reinforcement_length
    phi_ext = phi_retained if phi_retained is not None else phi_backfill
    gamma_ext = gamma_retained if gamma_retained is not None else gamma_backfill
    q = geom.surcharge if live_load is None else live_load
    Ka = rankine_Ka(phi_ext)

    # Unfactored load components about the toe (Point A). Vertical loads V*
    # resist overturning (arm L/2); horizontal loads F* drive it.
    V1 = gamma_backfill * H * L          # EV: reinforced mass weight
    Vs = q * L                           # LL: vertical surcharge over the mass
    F1 = 0.5 * Ka * gamma_ext * H ** 2   # EH: active earth thrust (arm H/3)
    F2 = Ka * q * H                      # LL: surcharge thrust (arm H/2)
    MV1 = V1 * (L / 2.0)
    MVs = Vs * (L / 2.0)
    MF1 = F1 * (H / 3.0)
    MF2 = F2 * (H / 2.0)

    delta_b = min(phi_backfill, phi_foundation)
    tan_d = math.tan(math.radians(delta_b))
    ca = (2.0 / 3.0) * c_foundation

    # --- Sliding (LL excluded from resistance) -----------------------------
    def _sliding_cdr(ev, eh):
        demand = eh * F1 + gamma_LL * F2
        resist = phi_sliding * (ev * V1 * tan_d + ca * L)
        return (resist / demand) if demand > 0 else 99.9

    cdr_slide_max = _sliding_cdr(gamma_EV_max, gamma_EH_max)   # all-max
    cdr_slide_min = _sliding_cdr(gamma_EV_min, gamma_EH_min)   # all-min
    cdr_slide_crit = _sliding_cdr(gamma_EV_min, gamma_EH_max)  # min V, max H
    cdr_slide_gov = min(cdr_slide_max, cdr_slide_min, cdr_slide_crit)

    # --- Limiting eccentricity (LL excluded from resistance) ---------------
    MOA = gamma_EH_max * MF1 + gamma_LL * MF2   # max overturning about toe

    def _eL(ev):
        Vr = ev * V1
        a = (ev * MV1 - MOA) / Vr
        return L / 2.0 - a

    eL_max = _eL(gamma_EV_max)     # Strength I max (EV=1.35)
    eL_crit = _eL(gamma_EV_min)    # min-vertical (EV=1.00): largest eccentricity
    e_limit = ecc_limit_ratio * L
    eL_gov = max(eL_max, eL_crit)

    # --- Bearing (LL included; min-vertical combo uses gamma_EV_min on all
    #     vertical gravity loads) ---------------------------------------------
    def _bearing(ev_soil, ev_ll):
        SV = ev_soil * V1 + ev_ll * Vs
        MR = ev_soil * MV1 + ev_ll * MVs
        a = (MR - MOA) / SV
        eL = L / 2.0 - a
        Bp = L - 2.0 * eL
        sigma = (SV / Bp) if Bp > 0 else 1.0e6
        return sigma, eL, Bp

    sig_str, eL_bear, Bp_str = _bearing(gamma_EV_max, gamma_LL)   # Str I max
    sig_crit, _, _ = _bearing(gamma_EV_min, gamma_EV_min)         # min-vertical

    # Service I: all factors 1.0, LL included, overturning without EH/LL factors.
    SV_s = V1 + Vs
    MR_s = MV1 + MVs
    MO_s = MF1 + MF2
    a_s = (MR_s - MO_s) / SV_s
    eL_s = L / 2.0 - a_s
    Bp_s = L - 2.0 * eL_s
    sig_svc = (SV_s / Bp_s) if Bp_s > 0 else 1.0e6

    def _cdr(resistance, demand):
        if resistance is None or demand <= 0:
            return None
        return resistance / demand

    cdr_bear_str = _cdr(bearing_resistance_strength, sig_str)
    cdr_bear_crit = _cdr(bearing_resistance_strength, sig_crit)
    cdr_bear_svc = _cdr(bearing_resistance_service, sig_svc)
    bear_cdrs = [c for c in (cdr_bear_str, cdr_bear_crit) if c is not None]
    cdr_bear_gov = min(bear_cdrs) if bear_cdrs else None

    passes = (
        cdr_slide_gov >= 1.0
        and eL_gov <= e_limit
        and (cdr_bear_gov is None or cdr_bear_gov >= 1.0)
    )

    return {
        "sliding": {
            "CDR_strength_max": round(cdr_slide_max, 3),
            "CDR_strength_min": round(cdr_slide_min, 3),
            "CDR_critical": round(cdr_slide_crit, 3),
            "CDR_governing": round(cdr_slide_gov, 3),
            "passes": cdr_slide_gov >= 1.0,
        },
        "eccentricity": {
            "eL_strength_max_m": round(eL_max, 4),
            "eL_critical_m": round(eL_crit, 4),
            "eL_governing_m": round(eL_gov, 4),
            "e_limit_m": round(e_limit, 4),
            "passes": eL_gov <= e_limit,
        },
        "bearing": {
            "sigma_v_strength_max_kPa": round(sig_str, 2),
            "eL_strength_max_m": round(eL_bear, 4),
            "B_eff_strength_max_m": round(Bp_str, 4),
            "sigma_v_critical_kPa": round(sig_crit, 2),
            "sigma_v_service_kPa": round(sig_svc, 2),
            "CDR_strength": round(cdr_bear_str, 3) if cdr_bear_str else None,
            "CDR_critical": round(cdr_bear_crit, 3) if cdr_bear_crit else None,
            "CDR_service": round(cdr_bear_svc, 3) if cdr_bear_svc else None,
            "CDR_governing": round(cdr_bear_gov, 3) if cdr_bear_gov else None,
            "passes": (cdr_bear_gov is None or cdr_bear_gov >= 1.0),
        },
        "passes": passes,
        "unfactored_kN_m": {
            "V1": round(V1, 3), "Vs": round(Vs, 3),
            "F1": round(F1, 3), "F2": round(F2, 3),
            "Ka_retained": round(Ka, 4),
        },
        "load_factors": {
            "EV_max": gamma_EV_max, "EV_min": gamma_EV_min,
            "EH_max": gamma_EH_max, "EH_min": gamma_EH_min,
            "LL": gamma_LL, "phi_sliding": phi_sliding,
            "ecc_limit_ratio": ecc_limit_ratio,
        },
    }


def analyze_mse_wall(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    reinforcement: Reinforcement,
    gamma_foundation: float = None,
    phi_foundation: float = None,
    c_foundation: float = 0.0,
    q_allowable: float = None,
    phi_retained: float = None,
    gamma_retained: float = None,
    lrfd_external: bool = False,
    bearing_resistance_strength: float = None,
    bearing_resistance_service: float = None,
    live_load: float = None,
) -> MSEWallResult:
    """Run complete MSE wall analysis.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry.
    gamma_backfill : float
        Reinforced fill unit weight (kN/m³).
    phi_backfill : float
        Reinforced fill friction angle (degrees).
    reinforcement : Reinforcement
        Reinforcement properties.
    gamma_foundation : float, optional
        Foundation unit weight. If None, uses gamma_backfill.
    phi_foundation : float, optional
        Foundation friction angle. If None, uses phi_backfill.
    c_foundation : float, optional
        Foundation cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing pressure (kPa). Default None.
    phi_retained : float, optional
        Retained soil friction angle (degrees). If None, uses phi_backfill.
    gamma_retained : float, optional
        Retained soil unit weight (kN/m³). If None, uses gamma_backfill.
    lrfd_external : bool, optional
        If True, ALSO run the AASHTO/GEC-11 LRFD external-stability path
        (``check_external_stability_lrfd``) and attach its CDR set to
        ``result.external_lrfd``. Default False (ASD FOS path only — unchanged).
    bearing_resistance_strength, bearing_resistance_service : float, optional
        Factored strength / service bearing resistances (kPa) for the LRFD
        bearing CDRs (only used when lrfd_external=True).
    live_load : float, optional
        Live-load surcharge (kPa) for the LRFD path. Default: geom.surcharge.

    Returns
    -------
    MSEWallResult
    """
    if gamma_foundation is None:
        gamma_foundation = gamma_backfill
    if phi_foundation is None:
        phi_foundation = phi_backfill

    external = check_external_stability(
        geom, gamma_backfill, phi_backfill,
        gamma_foundation, phi_foundation, c_foundation,
        q_allowable, phi_retained=phi_retained,
        gamma_retained=gamma_retained,
    )

    internal = check_internal_stability(
        geom, gamma_backfill, phi_backfill, reinforcement,
    )

    all_pass_internal = all(r["passes"] for r in internal)

    external_lrfd = None
    if lrfd_external:
        external_lrfd = check_external_stability_lrfd(
            geom, gamma_backfill, phi_backfill, phi_foundation,
            phi_retained=phi_retained, gamma_retained=gamma_retained,
            live_load=live_load, c_foundation=c_foundation,
            bearing_resistance_strength=bearing_resistance_strength,
            bearing_resistance_service=bearing_resistance_service,
        )

    return MSEWallResult(
        FOS_sliding=external["FOS_sliding"],
        FOS_overturning=external["FOS_overturning"],
        FOS_bearing=external["FOS_bearing"],
        passes_external=external["passes"],
        internal_results=internal,
        all_pass_internal=all_pass_internal,
        wall_height=geom.wall_height,
        reinforcement_length=geom.reinforcement_length,
        external_lrfd=external_lrfd,
    )
