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
    profile_for, passive_by_layer, PASSIVE_BASIS,
    PASSIVE_FRICTION_LEFT_OUT,
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
                              passive_theory: str = "rankine",
                              log_spiral_source: str = None,
                              passive_wall_friction_deg: float = None,
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
    passive_theory : str
        The passive coefficient below the excavation (GM-242): "rankine"
        (default, no wall friction; ``soe.free_earth.PASSIVE_BASIS``) or
        "log_spiral" (``soe.free_earth.passive_by_layer``).
    log_spiral_source : str, optional
        With "log_spiral", the printed source of Kp
        (``geotech_common.log_spiral.SOURCES``); None: UFC 3-220-20's,
        Eq. 4-12 for the vertical face and level ground in front.
    passive_wall_friction_deg : float, optional
        With "log_spiral", the wall friction (deg) in every layer below the
        excavation, at most its phi'; None: the source's own in a layer
        without cohesion, none in one with cohesion.

    Returns
    -------
    BracedExcavationResult
        Analysis results including support reactions, moments, and demands.
    """
    geometry.validate()
    H = geometry.excavation_depth
    passive = _with_passive(passive_theory, log_spiral_source,
                            passive_wall_friction_deg)
    passive_by_layer(geometry.soil_layers, H, **passive)  # checked first

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
    spans, dz, dp, dV, dM = [], [], [], [], []

    for i in range(n_spans):
        z_top, z_bot = bounds[i], bounds[i + 1]
        L = z_bot - z_top
        force, centroid = _span_load(p, z_top, z_bot, breaks)
        if i == 0:
            # Cantilever above the first support: all its load to that support.
            reactions[0] += force
            top_shear = 0.0
            to_upper, to_lower = 0.0, force
        elif i == n_spans - 1:
            # Bottom span: all load to the lowest support (none to the soil
            # below the excavation) -- conservative for that support. Its
            # moment is the simply-supported one.
            reactions[-1] += force
            top_shear = force * (L - centroid) / L if L > 0 else 0.0
            to_upper, to_lower = force, 0.0
        else:
            frac_bot = centroid / L if L > 0 else 1.0
            reactions[i - 1] += force * (1.0 - frac_bot)
            reactions[i] += force * frac_bot
            top_shear = force * (1.0 - frac_bot)
            to_upper, to_lower = force * (1.0 - frac_bot), force * frac_bot
        z, V, M = _span_diagram(p, z_top, z_bot, breaks, top_shear)
        spans.append({"z_top_m": round(z_top, 3), "z_bot_m": round(z_bot, 3),
                      "load_kN_per_m": round(force, 2),
                      "centroid_below_top_m": round(centroid, 3),
                      "to_upper_kN_per_m": round(to_upper, 2),
                      "to_lower_kN_per_m": round(to_lower, 2)})
        # Recorded in the free earth body's sign (push toward the excavation
        # positive, supports resisting it), so the two diagrams the calc
        # package draws on one set of axes agree: _span_diagram's V and M
        # are the negatives of that.
        dz.extend(map(float, z)); dp.extend(map(float, p(z)))
        dV.extend(-float(v) for v in V); dM.extend(-float(m) for m in M)
        k = int(np.argmax(np.abs(M)))
        if abs(M[k]) > max_moment:
            max_moment, max_moment_depth = float(abs(M[k])), float(z[k])
        max_shear = max(max_shear, float(np.max(np.abs(V))))

    # Step 3: embedment by free earth support about the lowest support
    from soe.embedment import embedment_detail
    if embedment_increase < 1.0:
        raise ValueError("embedment_increase must be >= 1.0")
    fe = embedment_detail(geometry, FOS_passive=FOS_passive, **passive)
    embedment = None if fe.D is None else fe.D * embedment_increase

    def embedment_with(other):
        D = embedment_detail(geometry, FOS_passive=FOS_passive, **other).D
        return None if D is None else D * embedment_increase
    span_moment = max_moment
    if fe.max_moment is None:
        # No FS = 1 body: the moment below the excavation is unknown, so the
        # wall's maximum moment (and the section it sizes) is too.
        max_moment = max_moment_depth = max_shear = None
    elif fe.max_moment > max_moment:
        max_moment, max_moment_depth = fe.max_moment, fe.max_moment_depth
        max_shear = max(max_shear, fe.max_shear)

    # A no-balance warning goes first, ahead of the method notes.
    notes: List[str] = list(fe.notes) + [
        f"Pressure above the excavation: {ap['type']} apparent envelope "
        f"(max {ap['max_pressure_kPa']:.2f} kPa) + surcharge "
        f"{ap['surcharge_pressure_kPa']:.2f} kPa"
        + (" + water pressure below the water table" if ap["water_pressure_added"]
           else "")
        + " (GEC-4 5.2.4 adds surcharge and water explicitly).",
        "The bottom span's load is assigned wholly to the lowest support; "
        + (f"free earth support gives that support {fe.support_load:.1f} kN/m "
           "from the wall below it." if fe.support_load is not None else
           "free earth support gives no load from below (no moment balance)."),
        f"Embedment: free earth support about the lowest support, MR = FS x MD "
        f"with FS = {FOS_passive:g} (Caltrans T&S 8-4 uses 1.3).",
    ]
    if max_moment is None:
        notes.append(
            f"The spans above the excavation reach {span_moment:.1f} kN·m/m; "
            "the moment below the excavation, and so the wall's maximum "
            "moment and required section, are undefined without a moment "
            "balance.")

    def _r(x, nd):
        return None if x is None else round(x, nd)

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
        max_moment_kNm_per_m=_r(max_moment, 2),
        max_moment_depth_m=_r(max_moment_depth, 2),
        max_shear_kN_per_m=_r(max_shear, 2),
        required_embedment_m=_r(embedment, 2),
        total_wall_length_m=None if embedment is None else round(H + embedment, 2),
        required_Sx_cm3=(None if max_moment is None
                         else round(_section_modulus_cm3(max_moment, Fy), 1)),
        embedment_FS=FOS_passive,
        surcharge_pressure_kPa=round(float(ap["surcharge_pressure_kPa"]), 2),
        water_pressure_included=bool(ap["water_pressure_added"]),
        free_earth={
            "embedment_D_m": _r(fe.D, 3),
            "embedment_D_FS1_m": _r(fe.D_prime, 3),
            "lowest_support_load_from_below_kN_per_m": _r(fe.support_load, 2),
            "max_moment_kNm_per_m": _r(fe.max_moment, 2),
            "max_moment_depth_m": _r(fe.max_moment_depth, 2),
            "converged": fe.converged,
        },
        converged=fe.converged,
        notes=notes,
        apparent_pressure=dict(ap),
        layers_used=_layers_used(profile_for(geometry, **passive),
                                 geometry, passive),
        spans=spans,
        diagram={"z": dz, "p": dp, "V": dV, "M": dM},
        free_earth_diagram=getattr(fe, "diagram", None),
        choices=passive_choices(geometry, passive, embedment,
                                embedment_with),
    )


# ============================================================================
# Cantilever excavation analysis
# ============================================================================

def _layers_used(prof, geometry, passive=None) -> list:
    """Each layer of a profile as the pressures used it: top_m, bottom_m,
    unit_weight, friction_angle, cohesion, Ka, Kp; with the log-spiral Kp
    (``passive``, GM-242) also the wall friction on the passive side,
    passive_wall_friction_deg."""
    tops = np.concatenate([[0.0], prof.bottoms[:-1]])
    rows = (passive_by_layer(geometry.soil_layers, geometry.excavation_depth,
                             **passive)
            if passive and passive["passive_theory"] == "log_spiral"
            else None)
    out = [
        {"top_m": round(float(t), 3), "bottom_m": round(float(b), 3),
         "unit_weight": float(prof.gamma[k]),
         "friction_angle": float(geometry.soil_layers[k].friction_angle),
         "cohesion": float(prof.c[k]),
         "Ka": round(float(prof.Ka[k]), 4), "Kp": round(float(prof.Kp[k]), 4)}
        for k, (t, b) in enumerate(zip(tops, prof.bottoms))]
    for row, r in zip(out, rows or ()):
        row["passive_wall_friction_deg"] = round(r["delta"], 4)
    return out


def _pressure_points(prof, FS: float, O: float) -> list:
    """The pressures of a cantilever profile at its break depths down to O
    (just above and just below a depth where a layer changes)."""
    eps = 1e-6
    depths = sorted({0.0, O, *[b for b in prof.breaks() if 0.0 < b < O]})
    rows = []
    for z in depths:
        for zz in ((z,) if z in (0.0, O) else (z - eps, z + eps)):
            act = float(prof.active(zz))
            water = float(prof.u_retained(zz) - prof.u_excavation(zz))
            pas = float(prof.passive(zz)) / FS
            rows.append({
                "z_m": round(z, 3),
                "side": ("" if zz == z else ("above" if zz < z else "below")),
                "sv_retained_kPa": round(float(prof.sv_retained(zz)), 2),
                "active_kPa": round(act, 2),
                "water_net_kPa": round(water, 2),
                "sv_excavation_kPa": round(float(prof.sv_excavation(zz)), 2),
                "passive_over_FS_kPa": round(pas, 2),
                "net_kPa": round(act + water - pas, 2)})
    # one row where nothing changes across the depth
    out = []
    for r in rows:
        if out and out[-1]["z_m"] == r["z_m"] and all(
                out[-1][k] == r[k] for k in r if k not in ("side",)):
            out[-1]["side"] = ""
            continue
        out.append(r)
    return out


def analyze_cantilever_excavation(
    geometry: ExcavationGeometry,
    FOS_passive: float = 1.5,
    Fy: float = 345.0,
    embedment_increase: float = 1.2,
    passive_theory: str = "rankine",
    log_spiral_source: str = None,
    passive_wall_friction_deg: float = None,
) -> CantileverExcavationResult:
    """Analyze a cantilever (unbraced) excavation wall.

    Caltrans Trenching and Shoring Manual Simplified Method (Sec 7-5.02) on
    layered Rankine active and passive pressures (or the log-spiral passive
    Kp, ``passive_theory``, GM-242), with water on both sides:
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
    passive_theory, log_spiral_source, passive_wall_friction_deg : optional
        The passive coefficient below the excavation, as
        :func:`analyze_braced_excavation` takes them (GM-242).

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

    passive = _with_passive(passive_theory, log_spiral_source,
                            passive_wall_friction_deg)
    prof = profile_for(geometry, **passive)
    sol = solve_cantilever(prof, FS_passive=FOS_passive,
                           embedment_increase=embedment_increase)

    def embedment_with(other):
        return solve_cantilever(profile_for(geometry, **other),
                                FS_passive=FOS_passive,
                                embedment_increase=embedment_increase).D

    choices = passive_choices(geometry, passive,
                              sol.D if sol.converged else None,
                              embedment_with)
    i = int(prof._layer(H + 1e-6))  # the layer the wall is embedded in

    notes = list(sol.notes)
    if H > _CANTILEVER_TYPICAL_MAX_H:
        notes.append(
            f"H = {H:.1f} m: cantilever walls are mainly used for excavations up "
            f"to about {_CANTILEVER_TYPICAL_MAX_H:.1f} m (18 ft, Caltrans T&S "
            "7-1); check deflection.")

    layers_used = _layers_used(prof, geometry, passive)

    if not sol.converged:
        # No D0 within 6H: nothing below is reported (see the first note).
        return CantileverExcavationResult(
            layers_used=layers_used,
            excavation_depth=H, FOS_passive=FOS_passive,
            Ka=round(float(prof.Ka[i]), 4), Kp=round(float(prof.Kp[i]), 4),
            required_embedment_m=None, total_wall_length_m=None,
            max_moment_kNm_per_m=None, max_shear_kN_per_m=None,
            required_Sx_cm3=None, embedment_converged_m=None,
            embedment_increase=embedment_increase, max_moment_depth_m=None,
            net_force_at_O_kN_per_m=None, converged=False, notes=notes,
            choices=choices)

    return CantileverExcavationResult(
        layers_used=layers_used, diagram=sol.diagram,
        pressure_points=_pressure_points(prof, FOS_passive, H + sol.D0),
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
        choices=choices,
    )


# ============================================================================
# The passive coefficient below the excavation (GM-242)
# ============================================================================
def _with_passive(theory, source, delta):
    """The keyword arguments of a passive theory."""
    return {"passive_theory": theory, "log_spiral_source": source,
            "passive_wall_friction_deg": delta}


def _layers_Kp(rows) -> str:
    """Each layer below the excavation's Kp and wall friction, in a line."""
    return "; ".join(
        f"layer {k}: Kp {r['Kp']:.3g}" + (f" at delta {r['delta']:.3g} deg"
                                         if r["delta"] else "")
        for k, r in enumerate(rows, 1) if r["below"])


#: How each layer's wall friction came, for the choice.
_CAME = {"given": "given, at most phi'",
         "source": "delta/phi' 0.66, taken when left out",
         "cohesion": "none with cohesion, taken when left out",
         "phi_0": "none, phi' 0"}


def passive_choices(geometry: ExcavationGeometry, passive: dict,
                    embedment, embedment_with) -> list:
    """How the analysis took the passive pressure below the excavation
    (GM-242), as a result's ``choices`` (``geotech_common.choices``): the
    theory with the other theory's Kp and embedment as the alternative
    (``embedment_with(passive)`` gives it, None with no balance), and with
    the log-spiral Kp its source and each layer's wall friction.
    ``embedment`` is the analysis's own (m, or None)."""
    from geotech_common import log_spiral as _ls
    from geotech_common.choices import DEFAULT, choice, how_given
    layers, H = geometry.soil_layers, geometry.excavation_depth
    theory = passive["passive_theory"]
    rows = passive_by_layer(layers, H, **passive)
    if theory == "rankine":
        other = _with_passive("log_spiral", None, None)
        other_name = ("log_spiral (passive_theory 'log_spiral'), UFC "
                      "3-220-20 Eq. 4-12, wall friction left out")
    else:
        other = _with_passive("rankine", None, None)
        other_name = "rankine (passive_theory 'rankine'), no wall friction"

    def depth(D):
        return ("no embedment balances the moments" if D is None
                else f"embedment {D:.2f} m")

    try:
        rows_other = passive_by_layer(layers, H, **other)
        alt = (f"{other_name}: {_layers_Kp(rows_other)}; "
               f"{depth(embedment_with(other))}")
    except ValueError as exc:
        alt = f"{other_name}: none here, {exc}"
    chosen = (f"{theory}: {_layers_Kp(rows)}; {depth(embedment)}")
    out = [choice(
        "passive pressure below the excavation (passive_theory)", chosen,
        how_given(theory, "rankine", name="passive_theory"),
        PASSIVE_BASIS, [alt])]
    if theory == "rankine":
        return out
    source = passive["log_spiral_source"] or _ls.DEFAULT_SOURCE
    below = [(k, layer, r) for k, (layer, r) in
             enumerate(zip(layers, rows), 1) if r["below"]]
    basis = _ls.SOURCE_BASIS[source]
    if source == _ls.DEFAULT_SOURCE:
        basis += (". In front of the wall the face is vertical and the "
                  "ground level, so it is Eq. 4-12 (p. 191). "
                  + _ls.EQ_VS_CHARTS_NOTE)
    if source in (_ls.DEFAULT_SOURCE, "dm7_2_eq_4_12") and any(
            _ls.above_the_charts(layer.friction_angle, r["delta"])
            for _, layer, r in below):
        basis += ". " + _ls.HIGH_FRICTION_NOTE
    # the other sources, at the layer the wall is embedded in
    k, layer, r = next(((k, l, r) for k, l, r in below
                        if r["delta"] > 0), below[0])
    out.append(choice(
        "Log-spiral passive Kp below the excavation (log_spiral_source)",
        source, how_given(source, _ls.DEFAULT_SOURCE,
                          name="log_spiral_source"), basis,
        [f"layer {k}, " + a for a in _ls.other_sources(
            layer.friction_angle, r["delta"],
            used=(source, "dm7_2_eq_4_12"
                  if source == _ls.DEFAULT_SOURCE else source))]))
    given = passive["passive_wall_friction_deg"]
    left_out = any(r["came"] in ("source", "cohesion") for _, _, r in below)
    out.append(choice(
        "Wall friction delta below the excavation "
        "(passive_wall_friction_deg)",
        "; ".join(f"layer {k}: {r['delta']:.3g} deg ({_CAME[r['came']]})"
                  for k, _, r in below),
        DEFAULT if left_out else how_given(
            given, None, name="passive_wall_friction_deg"),
        PASSIVE_FRICTION_LEFT_OUT + ". " + _ls.wall_friction_note(
            source, left_out="")))
    return out
