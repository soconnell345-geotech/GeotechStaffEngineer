"""Retaining walls adapter — cantilever + MSE wall analysis + earth pressure coefficients."""

from funhouse_agent.adapters import (
    apply_aliases, declare_aliases, reject_unknown_params, require_params,
    without_unused,
)
from retaining_walls import earth_pressure as _ep
from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.mse import analyze_mse_wall, check_external_stability_lrfd
from retaining_walls.reinforcement import (
    Reinforcement, RIBBED_STEEL_STRIP_75x4, WELDED_WIRE_GRID_W11,
    GEOGRID_UX1600, GEOGRID_UX1700,
)

# Names the agent commonly reaches for, mapped to the adapter's canonical names.
_WALL_ALIASES = {"phi": "phi_backfill", "gamma": "gamma_backfill",
                 "friction_angle": "phi_backfill", "unit_weight": "gamma_backfill",
                 "c": "c_backfill", "cohesion": "c_backfill",
                 "height": "wall_height", "H": "wall_height",
                 "delta_b": "delta_base", "base_friction_angle": "delta_base",
                 "adhesion": "base_adhesion", "ca": "base_adhesion"}

# MSE reinforced fill is select granular — there is no cohesion input, so do
# not alias c/cohesion to c_backfill (the reject message would name a key the
# caller never sent).
_MSE_ALIASES = {k: v for k, v in _WALL_ALIASES.items() if v != "c_backfill"}

_REINFORCEMENT_DB = {
    "ribbed_steel_strip_75x4": RIBBED_STEEL_STRIP_75x4,
    "welded_wire_grid_w11": WELDED_WIRE_GRID_W11,
    "geogrid_ux1600": GEOGRID_UX1600,
    "geogrid_ux1700": GEOGRID_UX1700,
}


# Reinforcement-type aliases. The bar-mat / welded-grid Kr/Ka (2.5->1.2) and
# F* (20 t/St -> 10 t/St) curves are keyed off the dataclass type
# "metallic_grid"; expose the friendlier "bar_mat"/"welded_grid" names too.
_REINFORCEMENT_TYPE_ALIASES = {
    "bar_mat": "metallic_grid",
    "barmat": "metallic_grid",
    "welded_grid": "metallic_grid",
    "welded_wire_grid": "metallic_grid",
    "steel_grid": "metallic_grid",
    "grid": "metallic_grid",
    "strip": "metallic_strip",
    "ribbed_strip": "metallic_strip",
    "metallic_strip": "metallic_strip",
    "metallic_grid": "metallic_grid",
    "geosynthetic": "geosynthetic",
}


def _build_reinforcement(params):
    name = params.get("reinforcement_name", "").lower()
    if name in _REINFORCEMENT_DB:
        return _REINFORCEMENT_DB[name]
    if params.get("reinforcement_Tallowable") is None:
        given = params.get("reinforcement_name")
        not_built_in = (f"reinforcement_name '{given}' is not a built-in. "
                        if given else "")
        raise ValueError(
            f"mse_wall: {not_built_in}provide a built-in reinforcement_name "
            f"({sorted(_REINFORCEMENT_DB)}; see list_reinforcement) or a custom "
            "reinforcement_Tallowable (kN/m) with optional reinforcement_type/"
            "reinforcement_width/reinforcement_Fy/reinforcement_thickness "
            "(and, for a bar mat / welded grid: reinforcement_transverse_spacing "
            "= St with reinforcement_thickness = transverse-bar diameter t)."
        )
    r_type_in = params.get("reinforcement_type", "geosynthetic")
    r_type = _REINFORCEMENT_TYPE_ALIASES.get(r_type_in, r_type_in)
    return Reinforcement(
        name=params.get("reinforcement_name", "Custom"),
        type=r_type,
        Tallowable=params["reinforcement_Tallowable"],
        width=params.get("reinforcement_width", 0.05),
        Fy=params.get("reinforcement_Fy", 0.0),
        thickness=params.get("reinforcement_thickness", 0.0),
        transverse_spacing=params.get("reinforcement_transverse_spacing", 0.0),
    )


def _run_cantilever_wall(params):
    params = apply_aliases(params, _WALL_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "base_width", "toe_length", "stem_thickness_top",
         "stem_thickness_base", "base_thickness", "has_shear_key", "key_depth",
         "backfill_slope", "surcharge", "gamma_backfill", "phi_backfill",
         "c_backfill", "phi_foundation", "c_foundation", "q_allowable",
         "gamma_concrete", "FOS_sliding_required", "FOS_overturning_required",
         "pressure_method", "include_passive", "gamma_foundation",
         "delta_base", "base_adhesion"),
        method="cantilever_wall")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="cantilever_wall")
    geom = CantileverWallGeometry(
        wall_height=params["wall_height"],
        base_width=params.get("base_width"), toe_length=params.get("toe_length"),
        stem_thickness_top=params.get("stem_thickness_top", 0.30),
        stem_thickness_base=params.get("stem_thickness_base"),
        base_thickness=params.get("base_thickness", 0.60),
        has_shear_key=params.get("has_shear_key", False),
        key_depth=params.get("key_depth", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
    )
    result = analyze_cantilever_wall(
        geom, gamma_backfill=params["gamma_backfill"], phi_backfill=params["phi_backfill"],
        c_backfill=params.get("c_backfill", 0.0), phi_foundation=params.get("phi_foundation"),
        c_foundation=params.get("c_foundation", 0.0), q_allowable=params.get("q_allowable"),
        gamma_concrete=params.get("gamma_concrete", 24.0),
        FOS_sliding=params.get("FOS_sliding_required", 1.5),
        FOS_overturning=params.get("FOS_overturning_required", 2.0),
        pressure_method=params.get("pressure_method", "rankine"),
        include_passive=params.get("include_passive", False),
        gamma_foundation=params.get("gamma_foundation"),
        delta_base=params.get("delta_base"),
        base_adhesion=params.get("base_adhesion"),
    )
    output = result.to_dict()
    # The geometry actually analysed, auto-sized parts included, with the
    # height's datum named (GM-100: a model read H as the retained height).
    output["geometry"] = {
        "wall_height_m": geom.wall_height,
        "wall_height_basis": "underside of the base slab to the top of the stem",
        "stem_height_m": round(geom.stem_height, 3),
        "base_thickness_m": geom.base_thickness,
        "base_width_m": geom.base_width,
        "toe_length_m": geom.toe_length,
        "heel_length_m": round(geom.heel_length, 3),
        "stem_thickness_top_m": geom.stem_thickness_top,
        "stem_thickness_base_m": geom.stem_thickness_base,
    }
    # Surface the sliding-interface values actually used, so the caller can
    # verify the basis instead of guessing (owner wall session 2026-07-14).
    from retaining_walls.cantilever import check_sliding as _cs
    sl = _cs(geom, params["gamma_backfill"], params["phi_backfill"],
             params.get("c_backfill", 0.0), params.get("phi_foundation"),
             params.get("c_foundation", 0.0), params.get("gamma_concrete", 24.0),
             params.get("FOS_sliding_required", 1.5),
             params.get("pressure_method", "rankine"),
             params.get("include_passive", False), params.get("gamma_foundation"),
             delta_base=params.get("delta_base"),
             base_adhesion=params.get("base_adhesion"))
    output["sliding_basis"] = {
        "delta_base_deg": sl["delta_base_deg"],
        "base_adhesion_kPa": sl["base_adhesion_kPa"],
        "driving_force_kN_per_m": sl["driving_force_kN_per_m"],
        "resisting_force_kN_per_m": sl["resisting_force_kN_per_m"],
        "Pa_kN_per_m": sl["Pa_kN_per_m"],
        "Pa_vertical_kN_per_m": sl["Pa_vertical_kN_per_m"],
        "thrust_inclination_deg": sl["thrust_inclination_deg"],
    }
    return output


def _run_mse_wall(params):
    params = apply_aliases(params, _MSE_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "reinforcement_spacing",
         "surcharge", "backfill_slope", "broken_backslope_height",
         "gamma_backfill", "phi_backfill",
         "reinforcement_name", "reinforcement_type",
         "reinforcement_Tallowable", "reinforcement_width",
         "reinforcement_Fy", "reinforcement_thickness",
         "reinforcement_transverse_spacing", "gamma_foundation",
         "phi_foundation", "c_foundation", "q_allowable",
         "phi_retained", "gamma_retained", "traffic_vertical_resists"),
        method="mse_wall")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="mse_wall")
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        reinforcement_spacing=params.get("reinforcement_spacing", 0.60),
        surcharge=params.get("surcharge", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        broken_backslope_height=params.get("broken_backslope_height"),
    )
    reinforcement = _build_reinforcement(params)
    result = analyze_mse_wall(
        geom, gamma_backfill=params["gamma_backfill"], phi_backfill=params["phi_backfill"],
        reinforcement=reinforcement, gamma_foundation=params.get("gamma_foundation"),
        phi_foundation=params.get("phi_foundation"), c_foundation=params.get("c_foundation", 0.0),
        q_allowable=params.get("q_allowable"),
        phi_retained=params.get("phi_retained"),
        gamma_retained=params.get("gamma_retained"),
        traffic_vertical_resists=bool(
            params.get("traffic_vertical_resists", False)),
    )
    output = result.to_dict()
    output["reinforcement_name"] = reinforcement.name
    return output


def _run_mse_lrfd_external_stability(params):
    """AASHTO/GEC-11 LRFD external-stability CDR set (sliding / eccentricity /
    bearing) under Strength I (max + min) and Service I combinations."""
    params = without_unused(
        apply_aliases(params, _MSE_ALIASES),
        METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"])
    # No reinforcement_spacing: external stability does not read it, so it
    # is refused, not dropped (GM-110).
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "surcharge",
         "backfill_slope", "broken_backslope_height",
         "gamma_backfill", "phi_backfill",
         "phi_foundation", "phi_retained", "gamma_retained", "live_load",
         "bearing_resistance_strength", "bearing_resistance_service",
         "c_foundation", "gamma_EV_max", "gamma_EV_min", "gamma_EH_max",
         "gamma_EH_min", "gamma_LL", "phi_sliding", "ecc_limit_ratio",
         "traffic_vertical_resists"),
        method="mse_lrfd_external_stability")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="mse_lrfd_external_stability")
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        surcharge=params.get("surcharge", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        broken_backslope_height=params.get("broken_backslope_height"),
    )
    # Only forward load/resistance factors the caller actually set (so the
    # GEC-11 defaults live in one place — the module function).
    factor_keys = ("gamma_EV_max", "gamma_EV_min", "gamma_EH_max",
                   "gamma_EH_min", "gamma_LL", "phi_sliding", "ecc_limit_ratio")
    factors = {k: params[k] for k in factor_keys if k in params}
    result = check_external_stability_lrfd(
        geom, gamma_backfill=params["gamma_backfill"],
        phi_backfill=params["phi_backfill"],
        phi_foundation=params.get("phi_foundation", params["phi_backfill"]),
        phi_retained=params.get("phi_retained"),
        gamma_retained=params.get("gamma_retained"),
        live_load=params.get("live_load"),
        c_foundation=params.get("c_foundation", 0.0),
        bearing_resistance_strength=params.get("bearing_resistance_strength"),
        bearing_resistance_service=params.get("bearing_resistance_service"),
        traffic_vertical_resists=bool(
            params.get("traffic_vertical_resists", False)),
        **factors,
    )
    result["reinforcement_length_m"] = geom.reinforcement_length
    return result


_K_ALIASES = {"phi": "phi_deg", "friction_angle": "phi_deg",
              "phi_backfill": "phi_deg",
              "delta": "delta_deg", "wall_friction": "delta_deg",
              "beta": "beta_deg", "backfill_slope": "beta_deg",
              "slope_angle": "beta_deg",
              "alpha": "alpha_deg", "wall_angle": "alpha_deg"}


def _run_earth_pressure_coefficient(params):
    """Thin wrapper over retaining_walls.earth_pressure Ka/Kp/K0 functions."""
    p = apply_aliases(params, _K_ALIASES)
    reject_unknown_params(
        p, ("phi_deg", "theory", "state", "delta_deg", "beta_deg", "alpha_deg"),
        method="earth_pressure_coefficient")
    require_params(p, ["phi_deg"], method="earth_pressure_coefficient",
                   valid=["phi_deg", "theory", "state", "delta_deg", "beta_deg", "alpha_deg"])
    theory = p.get("theory", "rankine")
    state = p.get("state", "active")
    phi = p["phi_deg"]
    beta = p.get("beta_deg", 0.0)
    if theory not in ("rankine", "coulomb", "caquot_kerisel"):
        raise ValueError(f"Unknown theory '{theory}'. Allowed: "
                         "['rankine', 'coulomb', 'caquot_kerisel'].")
    if state not in ("active", "passive", "at_rest"):
        raise ValueError(f"Unknown state '{state}'. Allowed: ['active', 'passive', 'at_rest'].")

    if theory == "caquot_kerisel":
        # Caquot-Kerisel (1948) log-spiral PASSIVE coefficient Kp (v5.3). Unlike
        # Coulomb it does not over-predict Kp at high wall friction delta/phi.
        # Passive-only; delta_deg defaults to phi (R=1) when omitted.
        if state != "passive":
            raise ValueError(
                "theory 'caquot_kerisel' is a passive (Kp) log-spiral coefficient; "
                "use state='passive' (it has no active/at-rest form).")
        from soe.earth_pressure import caquot_kerisel_Kp
        K, symbol = caquot_kerisel_Kp(phi, delta_deg=p.get("delta_deg")), "Kp"
    elif state == "at_rest":
        K, symbol = _ep.K0(phi), "K0"
    elif theory == "rankine":
        if state == "active":
            K = _ep.rankine_Ka_sloped(phi, beta) if beta else _ep.rankine_Ka(phi)
            symbol = "Ka"
        else:
            K, symbol = _ep.rankine_Kp(phi), "Kp"
    else:  # coulomb
        delta = p.get("delta_deg", 0.0)
        alpha = p.get("alpha_deg", 90.0)
        if state == "active":
            K, symbol = _ep.coulomb_Ka(phi, delta, alpha, beta), "Ka"
        else:
            K, symbol = _ep.coulomb_Kp(phi, delta, alpha, beta), "Kp"
    return {
        "K": round(K, 4),
        "coefficient": symbol,
        "theory": "jaky_at_rest" if state == "at_rest" else theory,
        "state": state,
        "phi_deg": phi,
        "delta_deg": p.get("delta_deg", 0.0),
        "beta_deg": beta,
        "alpha_deg": p.get("alpha_deg", 90.0),
    }


def _run_list_reinforcement(params):
    result = {}
    for key, r in _REINFORCEMENT_DB.items():
        result[key] = {"name": r.name, "type": r.type, "Tallowable_kN_per_m": r.Tallowable}
    return {"reinforcement_types": result}


METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "mse_wall": _run_mse_wall,
    "mse_lrfd_external_stability": _run_mse_lrfd_external_stability,
    "list_reinforcement": _run_list_reinforcement,
    "earth_pressure_coefficient": _run_earth_pressure_coefficient,
}

METHOD_INFO = {
    "cantilever_wall": {
        "category": "Cantilever Wall",
        "brief": "Full cantilever retaining wall stability analysis (sliding, overturning, eccentricity, bearing).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "TOTAL wall height H (m), measured from the UNDERSIDE of the base slab (footing) to the top of the stem: H = base_thickness + stem height. The active thrust acts over H on the vertical plane through the back of the heel. If you know the height retained above the top of the footing, add base_thickness (4.0 m retained on a 0.5 m footing is wall_height 4.5)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "base_width": {"type": "float", "required": False, "description": "Base width (m). Auto-sized if omitted."},
            "toe_length": {"type": "float", "required": False, "description": "Toe length (m). Auto-sized if omitted."},
            "stem_thickness_top": {"type": "float", "required": False, "default": 0.30, "description": "Stem thickness at top (m)."},
            "stem_thickness_base": {"type": "float", "required": False, "description": "Stem thickness at base (m)."},
            "base_thickness": {"type": "float", "required": False, "default": 0.60, "description": "Base slab thickness (m)."},
            "has_shear_key": {"type": "bool", "required": False, "default": False, "description": "Include a shear key (with key_depth)."},
            "key_depth": {"type": "float", "required": False, "default": 0.0, "description": "Shear key depth (m)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surcharge (kPa)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees)."},
            "c_backfill": {"type": "float", "required": False, "default": 0.0, "description": "Backfill cohesion (kPa)."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg), FULL value — the sliding check applies delta_b = (2/3)*phi_foundation INTERNALLY. Do NOT pre-reduce (passing 2/3*phi here double-counts the reduction). To set the base friction angle directly, use delta_base instead. Defaults to phi_backfill."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa), FULL value — the sliding check applies adhesion ca = (2/3)*c_foundation INTERNALLY. Do NOT pre-reduce. To set the adhesion directly, use base_adhesion instead."},
            "delta_base": {"type": "float", "required": False, "description": "Base interface friction angle delta_b (deg), used DIRECTLY in the sliding check with NO 2/3 factor (e.g. delta_b = phi for cast-in-place concrete failing soil-on-soil, or a chosen 2/3*phi). Overrides the (2/3)*phi_foundation default."},
            "base_adhesion": {"type": "float", "required": False, "description": "Base adhesion ca (kPa), used DIRECTLY with NO 2/3 factor (0 = no adhesion). Overrides the (2/3)*c_foundation default."},
            "include_passive": {"type": "bool", "required": False, "default": False, "description": "Include passive resistance in front of the wall (on key_depth if has_shear_key, else on base_thickness) in the sliding check."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation soil unit weight (kN/m3) for passive resistance. Defaults to gamma_backfill."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa) for the bearing check. Without it the bearing check is NOT RUN: FOS_bearing and passes_bearing are null and checks_not_run lists 'bearing' (q_toe / q_heel are still reported)."},
            "gamma_concrete": {"type": "float", "required": False, "default": 24.0, "description": "Concrete unit weight (kN/m3)."},
            "FOS_sliding_required": {"type": "float", "required": False, "default": 1.5, "description": "Required sliding FOS."},
            "FOS_overturning_required": {"type": "float", "required": False, "default": 2.0, "description": "Required overturning FOS."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
        },
        "returns": {"verdict": "The result in plain words: the FAILED checks first with their numbers, then those that pass, then those not checked. Report it to the user, failures first.", "checks_failed": "Checks that were run and failed: sliding, overturning, bearing, eccentricity (resultant outside the middle third). Never leave one out of the answer.", "checks_not_run": "Checks not run ('bearing' when no q_allowable was given). Say so; never call the wall adequate in bearing.", "all_checks_pass": "true only when every check ran and passed; false when one failed; null when none failed but a check was not run.", "FOS_sliding": "Factor of safety against sliding (null when there is no driving force: the check passes).", "FOS_overturning": "Factor of safety against overturning (null when there is no driving force).", "FOS_bearing": "q_allowable / the larger of q_toe and q_heel; null when the bearing check was not run.", "q_toe_kPa": "Bearing pressure at the toe.", "q_heel_kPa": "Bearing pressure at the heel.", "geometry": "The geometry analysed, auto-sized parts included; wall_height is from the underside of the base to the top of the stem.", "sliding_basis": "The sliding-check basis actually used: delta_base_deg, base_adhesion_kPa, driving/resisting forces, Pa and its vertical component + inclination — verify these match your intended assumptions."},
    },
    "earth_pressure_coefficient": {
        "category": "Earth Pressure",
        "brief": "Lateral earth pressure coefficient Ka/Kp/K0 (Rankine, Coulomb, or Caquot-Kerisel log-spiral passive Kp).",
        "parameters": {
            "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees). Alias: phi."},
            "theory": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "caquot_kerisel"], "description": "Earth pressure theory. 'caquot_kerisel' = Caquot-Kerisel (1948) log-spiral PASSIVE coefficient Kp (state must be 'passive'; does not over-predict Kp at high wall friction the way Coulomb does). Theory is ignored for at_rest, which uses Jaky K0 = 1 - sin(phi)."},
            "state": {"type": "str", "required": False, "default": "active", "allowed_values": ["active", "passive", "at_rest"], "description": "Pressure state: Ka (active), Kp (passive), or K0 (at rest). 'caquot_kerisel' requires 'passive'."},
            "delta_deg": {"type": "float", "required": False, "default": 0.0, "description": "Wall friction angle (degrees). Coulomb and caquot_kerisel. Alias: delta. For caquot_kerisel, omitting it means delta = phi (full wall friction, R = 1)."},
            "beta_deg": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees). Aliases: beta, backfill_slope."},
            "alpha_deg": {"type": "float", "required": False, "default": 90.0, "description": "Wall back face angle from horizontal (degrees, 90 = vertical). Coulomb only. Alias: alpha."},
        },
        "returns": {"K": "Earth pressure coefficient value.", "coefficient": "Ka, Kp, or K0.", "theory": "Theory used."},
    },
    "mse_wall": {
        "category": "MSE Wall",
        "brief": "MSE wall external + internal stability analysis (GEC-11), level or sloping backfill (incl. a broken backslope).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Design wall height H (m): the height of the reinforced soil mass, incl. embedment (GEC-11)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "reinforcement_name": {"type": "str", "required": False, "description": "Built-in reinforcement name (use list_reinforcement to see options), or omit and define a custom reinforcement via reinforcement_type + reinforcement_Tallowable."},
            "reinforcement_type": {"type": "str", "required": False, "default": "geosynthetic", "allowed_values": ["metallic_strip", "metallic_grid", "bar_mat", "welded_grid", "geosynthetic"], "description": "Custom reinforcement type. Drives the internal-stability lateral-stress-ratio Kr/Ka and pullout F* curves: ribbed STRIP (metallic_strip: Kr/Ka 1.7->1.2, F* 2.0->tan-phi), steel bar mat / welded grid (metallic_grid / bar_mat / welded_grid: Kr/Ka 2.5->1.2, F* 20(t/St)->10(t/St) — supply reinforcement_thickness=t and reinforcement_transverse_spacing=St), or geosynthetic (Kr/Ka 1.0)."},
            "reinforcement_Tallowable": {"type": "float", "required": False, "description": "Allowable tensile strength (kN/m) for a custom reinforcement. Required when reinforcement_name is not a built-in."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length (m). Auto-sized (0.7H minimum) if omitted."},
            "reinforcement_spacing": {"type": "float", "required": False, "default": 0.6, "description": "Vertical reinforcement spacing (m)."},
            "reinforcement_width": {"type": "float", "required": False, "default": 0.05, "description": "Custom reinforcement width (m), pullout."},
            "reinforcement_Fy": {"type": "float", "required": False, "default": 0.0, "description": "Custom steel yield strength (MPa)."},
            "reinforcement_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Strip thickness (m) for a strip; transverse-bar diameter t (m) for a steel bar mat / welded grid (the t in F* = 20 t/St)."},
            "reinforcement_transverse_spacing": {"type": "float", "required": False, "default": 0.0, "description": "Transverse-bar (grid) spacing St (m) for a steel bar mat / welded grid (the St in F* = 20 t/St). Used only for bar_mat/welded_grid/metallic_grid."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Live-load (traffic) surcharge (kPa), e.g. heq*gamma. Internally it is in Tmax and not in the pullout resistance, which GEC-11 takes 'neglecting traffic loads' (p. 4-49). With a sloping backfill it is taken only with broken_backslope_height, as traffic on the level ground behind the crest; refused on a slope that continues."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Slope of the backfill surface above the wall, beta (degrees from horizontal), rising away from the face; 0 = level. GEC-11 Section 4.4.4: Coulomb Kab with the interface angle delta = beta (Eqs. 4-2/4-3); thrust F_T = 1/2*Kab*gamma_retained*h^2 on h = H + L*tan(beta), inclined at beta (Eq. 4-7); the slope fill over the reinforced zone is added to the weight; internally the slope adds S_eq = 1/2*(0.7H)*tan(beta) of fill (Fig. 4-11). Must not be steeper than phi_retained."},
            "phi_retained": {"type": "float", "required": False, "description": "Friction angle (deg) of the retained fill behind the reinforced zone, for the external thrust. Defaults to phi_backfill."},
            "gamma_retained": {"type": "float", "required": False, "description": "Unit weight (kN/m3) of the retained fill behind the reinforced zone. Defaults to gamma_backfill."},
            "broken_backslope_height": {"type": "float", "required": False, "description": "For a broken backslope (GEC-11 Fig. 4-4): the height S (m) of the slope crest above the top of the wall, with level ground behind it. Omit for a slope that continues. Kab and the thrust inclination then use I, the angle from the top of the wall to the ground 2H behind the face, and h = H + min(L*tan(beta), S)."},
            "traffic_vertical_resists": {"type": "bool", "required": False, "default": False, "description": "Broken backslope with traffic (surcharge) behind the crest only: count the traffic thrust's vertical component F2*sin(I) as sliding and eccentricity resistance, as GEC-11 Example E1 does (Vol II pp. E1-7 to E1-9). Default false: the GEC-11 text counts loads that increase sliding resistance 'only if those loads are permanent' (Vol I p. 4-22). Bearing counts it either way (p. 4-26). The result's backslope.traffic_basis says which was used."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation soil unit weight (kN/m3)."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg)."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa). Without it the external bearing check is NOT RUN (FOS_bearing null, checks_not_run lists 'bearing'); sigma_v is still reported."},
        },
        "returns": {"verdict": "The result in plain words: failed checks first, then those not checked. Report it to the user.", "checks_failed": "Checks run and failed: sliding, overturning, bearing, internal (pullout or rupture at some level).", "checks_not_run": "Checks not run ('bearing' when no q_allowable).", "all_checks_pass": "true only when every check ran and passed; false when one failed; null when none failed but a check was not run.", "FOS_sliding": "Sliding FOS.", "FOS_overturning": "Overturning FOS.", "FOS_bearing": "Bearing FOS q_allowable / sigma_v; null when not checked.", "passes_external": "true / false, or null when nothing failed but bearing was not checked.", "backslope": "With a sloping backfill only: the GEC-11 basis used (Kab, interface angle, h, F_T and its components, the slope fill weight V2; S_eq, sigma_2 and H1 for internal stability). Each internal level then also gives Zp_m, the depth below the slope used for pullout."},
    },
    "mse_lrfd_external_stability": {
        "category": "MSE Wall",
        "brief": "MSE wall external stability by AASHTO/GEC-11 LRFD: sliding / eccentricity / bearing capacity:demand ratios (CDRs) under Strength I (max+min) and Service I, level or sloping backfill. Distinct from mse_wall, which reports ASD factors of safety.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Design wall height H (m), incl. embedment. Alias: height."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Reinforced fill unit weight (kN/m3). Alias: gamma."},
            "phi_backfill": {"type": "float", "required": True, "description": "Reinforced fill friction angle (deg). Alias: phi."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length L (m). Auto-sized (0.7H) if omitted."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Live-load surcharge q (kPa) over the mass (e.g. heq*gamma). Used as the LL unless live_load overrides. With a broken backslope it is traffic on the level ground behind the crest (thrust Kab*q*h, inclined at I; GEC-11 Example E1); refused on a slope that continues."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Slope of the backfill surface above the wall, beta (degrees from horizontal), rising away from the face; 0 = level. GEC-11 Section 4.4.6 (Examples E3, E1): Coulomb Kab with delta = beta; F_T = 1/2*Kab*gamma_retained*h^2 on h = H + L*tan(beta) (Eq. 4-7), inclined at beta; F_T*cos(beta) drives and F_T*sin(beta) resists, both EH-factored (Eqs. 4-11, 4-13, 4-16, 4-21); the slope fill over the zone (V2) is an EV load. Must not be steeper than phi_retained."},
            "broken_backslope_height": {"type": "float", "required": False, "description": "For a broken backslope (GEC-11 Fig. 4-4): the height S (m) of the slope crest above the top of the wall, with level ground behind it. Omit for a slope that continues. Kab and the thrust inclination then use I, the angle from the top of the wall to the ground 2H behind the face, and h = H + min(L*tan(beta), S)."},
            "traffic_vertical_resists": {"type": "bool", "required": False, "default": False, "description": "Broken backslope with traffic (surcharge) behind the crest only: count the traffic thrust's vertical component F2*sin(I) as sliding and eccentricity resistance, as GEC-11 Example E1 does (Vol II pp. E1-7 to E1-9). Default false: the GEC-11 text counts loads that increase sliding resistance 'only if those loads are permanent' (Vol I p. 4-22). Bearing counts it either way (p. 4-26). The result's backslope.traffic_basis says which was used."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation friction angle (deg); the sliding interface uses min(phi_backfill, phi_foundation). Defaults to phi_backfill."},
            "phi_retained": {"type": "float", "required": False, "description": "Retained (behind-mass) fill friction angle (deg) for the external active thrust. Defaults to phi_backfill."},
            "gamma_retained": {"type": "float", "required": False, "description": "Retained fill unit weight (kN/m3). Defaults to gamma_backfill."},
            "live_load": {"type": "float", "required": False, "description": "Live-load surcharge q (kPa) if different from `surcharge`."},
            "bearing_resistance_strength": {"type": "float", "required": False, "description": "Factored bearing resistance qR at the strength limit (kPa) = phi_b*qn; enables the strength/critical bearing CDR = qR/sigma_v."},
            "bearing_resistance_service": {"type": "float", "required": False, "description": "Service bearing resistance (kPa); enables the Service I bearing CDR."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation cohesion (kPa) for sliding adhesion (2/3 c)."},
            "gamma_EV_max": {"type": "float", "required": False, "default": 1.35, "description": "Max vertical-earth (EV) load factor (AASHTO Strength I)."},
            "gamma_EV_min": {"type": "float", "required": False, "default": 1.00, "description": "Min vertical-earth (EV) load factor."},
            "gamma_EH_max": {"type": "float", "required": False, "default": 1.50, "description": "Max horizontal-earth (EH) load factor."},
            "gamma_EH_min": {"type": "float", "required": False, "default": 0.90, "description": "Min horizontal-earth (EH) load factor."},
            "gamma_LL": {"type": "float", "required": False, "default": 1.75, "description": "Live-load (LL) factor."},
            "phi_sliding": {"type": "float", "required": False, "default": 1.0, "description": "Sliding resistance factor (AASHTO 11.10.5.3)."},
            "ecc_limit_ratio": {"type": "float", "required": False, "default": 0.25, "description": "Limiting-eccentricity ratio e_max/L (0.25 = L/4 for MSE on soil; use 1/6 on rock)."},
        },
        "returns": {"sliding": "Sliding CDRs (strength max/min, critical, governing).", "eccentricity": "eL (strength max, critical) vs e_limit.", "bearing": "sigma_v (strength/critical/service) + bearing CDRs; checked false and passes null when no bearing resistance was given.", "passes": "true only when every check ran and passed; false when one failed; null when none failed but one was not run.", "checks_failed": "Checks that failed. Report every one.", "checks_not_run": "Checks not run for want of a bearing resistance: 'bearing (Strength I)', 'bearing (Service I)'. Say so.", "backslope": "With a sloping backfill only: the GEC-11 basis used (Kab, interface angle, h, F_T and its components, the slope fill weight V2)."},
    },
    "list_reinforcement": {
        "category": "MSE Wall",
        "brief": "List built-in reinforcement types with properties.",
        "parameters": {},
        "returns": {"reinforcement_types": "Dict of available reinforcement types."},
    },
}

declare_aliases(METHOD_INFO, {
    "cantilever_wall": _WALL_ALIASES,
    "mse_wall": _MSE_ALIASES,
    "mse_lrfd_external_stability": _MSE_ALIASES,
    "earth_pressure_coefficient": _K_ALIASES,
})

# Names the MSE methods do not take, with the reason dispatch gives (GM-110).
# A sloping backfill is taken since GM-112; what it does not cover (a slope
# steeper than phi, a surcharge on a slope that continues) is refused by
# the module with its reason.
METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"] = {
    "reinforcement_spacing": (
        "external stability does not depend on it; mse_wall's internal "
        "checks use it")}
