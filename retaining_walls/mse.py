"""
MSE (Mechanically Stabilized Earth) wall analysis.

Implements external stability (sliding, overturning, bearing) and
internal stability (tensile rupture, pullout) per FHWA GEC-11.

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-11 (FHWA-NHI-10-024), Chapters 4-5
    AASHTO LRFD Section 11.10
"""

import math
from typing import Dict, Any, List

from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.earth_pressure import rankine_Ka, horizontal_force_active
from retaining_walls.reinforcement import Reinforcement
from retaining_walls.results import MSEWallResult


def Kr_Ka_ratio(z: float, reinforcement_type: str = "metallic") -> float:
    """Ratio Kr/Ka as a function of depth per GEC-11.

    For metallic reinforcements: varies from 1.7 at z=0 to 1.2 at z>=6m.
    For geosynthetic: constant 1.0.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    reinforcement_type : str, optional
        "metallic" or "geosynthetic". Default "metallic".

    Returns
    -------
    float
        Kr/Ka ratio.

    References
    ----------
    FHWA GEC-11, Figure 4-10
    """
    if reinforcement_type == "geosynthetic":
        return 1.0

    # Metallic: linear from 1.7 at z=0 to 1.2 at z=6m, constant below
    if z <= 0:
        return 1.7
    elif z >= 6.0:
        return 1.2
    else:
        return 1.7 - (0.5 / 6.0) * z


def F_star_metallic(z: float, phi_backfill: float = 34.0) -> float:
    """Pullout resistance factor F* for ribbed metallic strips.

    F* varies linearly from 2.0 at z=0 to tan(phi) at z>=6m
    per GEC-11 Figure 4-11.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    phi_backfill : float, optional
        Backfill friction angle (degrees). Default 34.

    Returns
    -------
    float
        Pullout resistance factor F*.

    References
    ----------
    FHWA GEC-11, Figure 4-11 (for ribbed steel strips)
    """
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
    r_type = "metallic" if reinforcement.is_metallic else "geosynthetic"

    results = []
    for z in geom.reinforcement_depths:
        # Kr/Ka ratio
        kr_ka = Kr_Ka_ratio(z, r_type)

        # Tmax at this level
        T = Tmax_at_level(z, gamma_backfill, Ka, kr_ka, Sv, geom.surcharge)

        # Effective length Le = total length - active zone length
        # Active zone extends from face at 45+phi/2 from horizontal
        La = (H - z) * math.tan(math.radians(45 - phi_backfill / 2))
        Le = max(L - La, 1.0)  # minimum 1m effective length

        # Pullout resistance
        if reinforcement.is_metallic:
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

    # Bearing
    x_R = (M_stab - M_over) / W if W > 0 else L / 2.0
    e = L / 2.0 - x_R
    if abs(e) <= L / 6.0:
        q_toe = W / L * (1.0 + 6.0 * e / L)
    else:
        B_eff = 3.0 * x_R
        q_toe = 2.0 * W / B_eff if B_eff > 0 else 0.0

    FOS_bearing = q_allowable / q_toe if (q_allowable and q_toe > 0) else 99.9

    passes = (
        FOS_sliding >= FOS_sliding_req
        and FOS_overturning >= FOS_overturning_req
        and (FOS_bearing >= 1.0 if q_allowable else True)
    )

    return {
        "FOS_sliding": round(FOS_sliding, 3),
        "FOS_overturning": round(FOS_overturning, 3),
        "FOS_bearing": round(FOS_bearing, 3),
        "q_toe_kPa": round(q_toe, 1),
        "eccentricity_m": round(e, 3),
        "passes": passes,
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

    return MSEWallResult(
        FOS_sliding=external["FOS_sliding"],
        FOS_overturning=external["FOS_overturning"],
        FOS_bearing=external["FOS_bearing"],
        passes_external=external["passes"],
        internal_results=internal,
        all_pass_internal=all_pass_internal,
        wall_height=geom.wall_height,
        reinforcement_length=geom.reinforcement_length,
    )
