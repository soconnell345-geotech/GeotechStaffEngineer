"""
Beam analysis for braced and cantilever excavation walls.

Braced walls use the tributary area method (California Trenching and Shoring
Manual; FHWA GEC-4 Sec 5.4). The design pressure above the excavation is the
apparent-pressure envelope plus the surcharge and water pressures GEC-4 adds
explicitly (``soe.free_earth.apparent_pressure_profile``). The span above the
first support is a cantilever; the spans below it are simply supported (hinge
method). Embedment, and the moment in the embedded portion, come from free
earth support about the lowest support (``soe.embedment``).

Cantilever walls use the Caltrans Simplified Method on layered Rankine
pressures with water (``soe.free_earth.solve_cantilever``).

History (2026-09-15, field feedback N2): the cantilever used to take every
property from the first soil layer and put the passive resultant at H + D/3
about the wall base, so embedment came out about a third of the correct value;
the braced top span was treated as simply supported (pL^2/8 instead of the
cantilever pL^2/2); and surcharge and water were left out of the braced loads.
See DESIGN.md.

All units SI: kPa, kN/m, kN·m/m, meters.

References:
    California Dept. of Transportation, Trenching and Shoring Manual (2025),
        Sec 7-5.02 and 8-4
    FHWA-IF-99-015, GEC-4, Sections 5.2-5.4
    Terzaghi & Peck (1967) Soil Mechanics in Engineering Practice
"""

from typing import List

import numpy as np

from soe.geometry import ExcavationGeometry
from soe.free_earth import (
    _grid,
    apparent_pressure_profile,
    profile_for,
    solve_cantilever,
)
from soe.results import BracedExcavationResult, CantileverExcavationResult

#: Caltrans T&S Manual 7-1: cantilever sheet pile walls are mainly used for
#: temporary excavations not greater than about 18 ft.
_CANTILEVER_TYPICAL_MAX_H = 5.5


def _section_modulus_cm3(moment_kNm: float, Fy_MPa: float) -> float:
    """ASD required Sx (cm³/m): M / (0.66 Fy)."""
    Fb = 0.66 * Fy_MPa * 1000.0  # kPa
    return moment_kNm / Fb * 1e6 if Fb > 0 else 0.0


# ============================================================================
# Tributary area method for multi-level braced excavations
# ============================================================================

def _span_load(p, z_top: float, z_bot: float, breaks) -> tuple:
    """(force kN/m, centroid below z_top m) of pressure p over a span."""
    if z_bot <= z_top:
        return 0.0, 0.0
    segs = _grid(z_top, z_bot, breaks)
    F = float(sum(np.trapezoid(p(z), z) for z in segs))
    Mt = float(sum(np.trapezoid(p(z) * (z - z_top), z) for z in segs))
    return F, (Mt / F if F > 0 else 0.5 * (z_bot - z_top))


def _span_diagram(p, z_top: float, z_bot: float, breaks, top_shear: float):
    """Shear and moment along a span whose top carries ``top_shear``
    (0 for the free top of the cantilever span) and zero moment."""
    z = np.concatenate(_grid(z_top, z_bot, breaks))
    q = p(z)
    V = top_shear - np.concatenate(
        [[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(z))])
    M = np.concatenate([[0.0], np.cumsum(0.5 * (V[1:] + V[:-1]) * np.diff(z))])
    return z, V, M


def analyze_braced_excavation(geometry: ExcavationGeometry,
                              Fy: float = 345.0,
                              FOS_passive: float = 1.5,
                              embedment_increase: float = 1.0,
                              ) -> BracedExcavationResult:
    """Analyze a multi-level braced excavation using the tributary area method.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Complete excavation geometry with soil layers and support levels.
    Fy : float
        Steel yield strength (MPa). Default 345 (Grade 50). Used to
        compute required section modulus.
    FOS_passive : float
        Embedment factor of safety, MR = FS x MD about the lowest support.
        Default 1.5; the Caltrans T&S Manual uses 1.3 for temporary shoring.
    embedment_increase : float
        Multiplier on the free-earth-support embedment. Default 1.0 (the FS
        carries the safety, as in Caltrans Example 8-1).

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

    # Step 1: design pressure above the excavation
    ap, p = apparent_pressure_profile(geometry)
    depths = [s.depth for s in geometry.support_levels]
    breaks = depths + [H] + ([geometry.gwt_depth]
                             if geometry.gwt_depth is not None else [])

    # Step 2: spans [0, s1], [s1, s2], ..., [sN, H]
    bounds = [0.0] + depths + [H]
    n_spans = len(bounds) - 1
    reactions = [0.0] * len(depths)
    max_moment = max_moment_depth = max_shear = 0.0

    for i in range(n_spans):
        z_top, z_bot = bounds[i], bounds[i + 1]
        L = z_bot - z_top
        force, centroid = _span_load(p, z_top, z_bot, breaks)
        if i == 0:
            # Cantilever above the first support: all its load to that support.
            reactions[0] += force
            top_shear = 0.0
        elif i == n_spans - 1:
            # Bottom span: all load to the lowest support (none to the soil
            # below the excavation) -- conservative for that support. Its
            # moment is the simply-supported one.
            reactions[-1] += force
            top_shear = force * (L - centroid) / L if L > 0 else 0.0
        else:
            frac_bot = centroid / L if L > 0 else 1.0
            reactions[i - 1] += force * (1.0 - frac_bot)
            reactions[i] += force * frac_bot
            top_shear = force * (1.0 - frac_bot)
        z, V, M = _span_diagram(p, z_top, z_bot, breaks, top_shear)
        k = int(np.argmax(np.abs(M)))
        if abs(M[k]) > max_moment:
            max_moment, max_moment_depth = float(abs(M[k])), float(z[k])
        max_shear = max(max_shear, float(np.max(np.abs(V))))

    # Step 3: embedment by free earth support about the lowest support
    from soe.embedment import embedment_detail
    if embedment_increase < 1.0:
        raise ValueError("embedment_increase must be >= 1.0")
    fe = embedment_detail(geometry, FOS_passive=FOS_passive)
    embedment = fe.D * embedment_increase
    if fe.max_moment > max_moment:
        max_moment, max_moment_depth = fe.max_moment, fe.max_moment_depth
        max_shear = max(max_shear, fe.max_shear)

    notes: List[str] = [
        f"Pressure above the excavation: {ap['type']} apparent envelope "
        f"(max {ap['max_pressure_kPa']:.2f} kPa) + surcharge "
        f"{ap['surcharge_pressure_kPa']:.2f} kPa"
        + (" + water pressure below the water table" if ap["water_pressure_added"]
           else "")
        + " (GEC-4 5.2.4 adds surcharge and water explicitly).",
        "The bottom span's load is assigned wholly to the lowest support; "
        "free earth support gives that support "
        f"{fe.support_load:.1f} kN/m from the wall below it.",
        f"Embedment: free earth support about the lowest support, MR = FS x MD "
        f"with FS = {FOS_passive:g} (Caltrans T&S 8-4 uses 1.3).",
    ] + list(fe.notes)

    support_reactions = [
        {"depth_m": round(sup.depth, 3),
         "load_kN_per_m": round(reactions[i], 2),
         "type": sup.support_type}
        for i, sup in enumerate(geometry.support_levels)
    ]

    return BracedExcavationResult(
        excavation_depth=H,
        n_support_levels=len(depths),
        apparent_pressure_type=ap["type"],
        max_apparent_pressure_kPa=round(float(ap["max_pressure_kPa"]), 2),
        support_reactions=support_reactions,
        max_moment_kNm_per_m=round(max_moment, 2),
        max_moment_depth_m=round(max_moment_depth, 2),
        max_shear_kN_per_m=round(max_shear, 2),
        required_embedment_m=round(embedment, 2),
        total_wall_length_m=round(H + embedment, 2),
        required_Sx_cm3=round(_section_modulus_cm3(max_moment, Fy), 1),
        embedment_FS=FOS_passive,
        surcharge_pressure_kPa=round(float(ap["surcharge_pressure_kPa"]), 2),
        water_pressure_included=bool(ap["water_pressure_added"]),
        free_earth={
            "embedment_D_m": round(fe.D, 3),
            "embedment_D_FS1_m": round(fe.D_prime, 3),
            "lowest_support_load_from_below_kN_per_m": round(fe.support_load, 2),
            "max_moment_kNm_per_m": round(fe.max_moment, 2),
            "max_moment_depth_m": round(fe.max_moment_depth, 2),
        },
        notes=notes,
    )


# ============================================================================
# Cantilever excavation analysis
# ============================================================================

def analyze_cantilever_excavation(
    geometry: ExcavationGeometry,
    FOS_passive: float = 1.5,
    Fy: float = 345.0,
    embedment_increase: float = 1.2,
) -> CantileverExcavationResult:
    """Analyze a cantilever (unbraced) excavation wall.

    Caltrans Trenching and Shoring Manual Simplified Method (Sec 7-5.02) on
    layered Rankine active and passive pressures, with water on both sides:
    moments about the rotation point O at depth D0 below the excavation give
    D0, and D = ``embedment_increase`` x D0 (1.2 per AASHTO 3.11.5.6 -- it
    accounts for rotation below O and is not a factor of safety). Passive
    pressure is divided by ``FOS_passive``. The maximum moment and shear come
    from the same pressure diagram. Each depth uses its own layer's phi and c.

    Parameters
    ----------
    geometry : ExcavationGeometry
        Excavation geometry. support_levels should be empty.
    FOS_passive : float
        Factor of safety on passive resistance. Default 1.5.
    Fy : float
        Steel yield strength (MPa). Default 345.
    embedment_increase : float
        D / D0. Default 1.2; pass 1.0 to carry all safety on FOS_passive.

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

    prof = profile_for(geometry)
    sol = solve_cantilever(prof, FS_passive=FOS_passive,
                           embedment_increase=embedment_increase)
    i = int(prof._layer(H + 1e-6))  # the layer the wall is embedded in

    notes = list(sol.notes)
    if H > _CANTILEVER_TYPICAL_MAX_H:
        notes.append(
            f"H = {H:.1f} m: cantilever walls are mainly used for excavations up "
            f"to about {_CANTILEVER_TYPICAL_MAX_H:.1f} m (18 ft, Caltrans T&S "
            "7-1); check deflection.")

    return CantileverExcavationResult(
        excavation_depth=H,
        FOS_passive=FOS_passive,
        Ka=round(float(prof.Ka[i]), 4),
        Kp=round(float(prof.Kp[i]), 4),
        required_embedment_m=round(sol.D, 2),
        total_wall_length_m=round(H + sol.D, 2),
        max_moment_kNm_per_m=round(sol.max_moment, 2),
        max_shear_kN_per_m=round(sol.max_shear, 2),
        required_Sx_cm3=round(_section_modulus_cm3(sol.max_moment, Fy), 1),
        embedment_converged_m=round(sol.D0, 3),
        embedment_increase=embedment_increase,
        max_moment_depth_m=round(sol.max_moment_depth, 2),
        net_force_at_O_kN_per_m=round(sol.net_force_at_O, 2),
        notes=notes,
    )
