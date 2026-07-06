"""Retaining walls adapter — cantilever + MSE wall analysis + earth pressure coefficients."""

from funhouse_agent.adapters import apply_aliases, reject_unknown_params, require_params
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
                 "height": "wall_height", "H": "wall_height"}

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
        raise ValueError(
            "mse_wall: provide a built-in reinforcement_name "
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
         "pressure_method"),
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
    )
    output = result.to_dict()
    output["geometry"] = {"wall_height_m": geom.wall_height, "base_width_m": geom.base_width}
    return output


def _run_mse_wall(params):
    params = apply_aliases(params, _MSE_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "reinforcement_spacing",
         "backfill_slope", "surcharge", "gamma_backfill", "phi_backfill",
         "reinforcement_name", "reinforcement_type",
         "reinforcement_Tallowable", "reinforcement_width",
         "reinforcement_Fy", "reinforcement_thickness",
         "reinforcement_transverse_spacing", "gamma_foundation",
         "phi_foundation", "c_foundation", "q_allowable"),
        method="mse_wall")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="mse_wall")
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        reinforcement_spacing=params.get("reinforcement_spacing", 0.60),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
    )
    reinforcement = _build_reinforcement(params)
    result = analyze_mse_wall(
        geom, gamma_backfill=params["gamma_backfill"], phi_backfill=params["phi_backfill"],
        reinforcement=reinforcement, gamma_foundation=params.get("gamma_foundation"),
        phi_foundation=params.get("phi_foundation"), c_foundation=params.get("c_foundation", 0.0),
        q_allowable=params.get("q_allowable"),
    )
    output = result.to_dict()
    output["reinforcement_name"] = reinforcement.name
    return output


def _run_mse_lrfd_external_stability(params):
    """AASHTO/GEC-11 LRFD external-stability CDR set (sliding / eccentricity /
    bearing) under Strength I (max + min) and Service I combinations."""
    params = apply_aliases(params, _MSE_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "reinforcement_spacing",
         "backfill_slope", "surcharge", "gamma_backfill", "phi_backfill",
         "phi_foundation", "phi_retained", "gamma_retained", "live_load",
         "bearing_resistance_strength", "bearing_resistance_service",
         "c_foundation", "gamma_EV_max", "gamma_EV_min", "gamma_EH_max",
         "gamma_EH_min", "gamma_LL", "phi_sliding", "ecc_limit_ratio"),
        method="mse_lrfd_external_stability")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="mse_lrfd_external_stability")
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        reinforcement_spacing=params.get("reinforcement_spacing", 0.60),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
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
        **factors,
    )
    result["reinforcement_length_m"] = geom.reinforcement_length
    return result


def _run_earth_pressure_coefficient(params):
    """Thin wrapper over retaining_walls.earth_pressure Ka/Kp/K0 functions."""
    p = apply_aliases(params, {"phi": "phi_deg", "friction_angle": "phi_deg",
                               "phi_backfill": "phi_deg",
                               "delta": "delta_deg", "wall_friction": "delta_deg",
                               "beta": "beta_deg", "backfill_slope": "beta_deg",
                               "slope_angle": "beta_deg",
                               "alpha": "alpha_deg", "wall_angle": "alpha_deg"})
    reject_unknown_params(
        p, ("phi_deg", "theory", "state", "delta_deg", "beta_deg", "alpha_deg"),
        method="earth_pressure_coefficient")
    require_params(p, ["phi_deg"], method="earth_pressure_coefficient",
                   valid=["phi_deg", "theory", "state", "delta_deg", "beta_deg", "alpha_deg"])
    theory = p.get("theory", "rankine")
    state = p.get("state", "active")
    phi = p["phi_deg"]
    beta = p.get("beta_deg", 0.0)
    if theory not in ("rankine", "coulomb"):
        raise ValueError(f"Unknown theory '{theory}'. Allowed: ['rankine', 'coulomb'].")
    if state not in ("active", "passive", "at_rest"):
        raise ValueError(f"Unknown state '{state}'. Allowed: ['active', 'passive', 'at_rest'].")

    if state == "at_rest":
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
        "brief": "Full cantilever retaining wall stability analysis (sliding, overturning, bearing).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Wall height (m)."},
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
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg). Defaults to phi_backfill."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa) for the bearing check."},
            "gamma_concrete": {"type": "float", "required": False, "default": 24.0, "description": "Concrete unit weight (kN/m3)."},
            "FOS_sliding_required": {"type": "float", "required": False, "default": 1.5, "description": "Required sliding FOS."},
            "FOS_overturning_required": {"type": "float", "required": False, "default": 2.0, "description": "Required overturning FOS."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
        },
        "returns": {"FOS_sliding": "Factor of safety against sliding.", "FOS_overturning": "Factor of safety against overturning."},
    },
    "earth_pressure_coefficient": {
        "category": "Earth Pressure",
        "brief": "Lateral earth pressure coefficient Ka/Kp/K0 (Rankine or Coulomb).",
        "parameters": {
            "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees). Alias: phi."},
            "theory": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory (ignored for at_rest, which uses Jaky K0 = 1 - sin(phi))."},
            "state": {"type": "str", "required": False, "default": "active", "allowed_values": ["active", "passive", "at_rest"], "description": "Pressure state: Ka (active), Kp (passive), or K0 (at rest)."},
            "delta_deg": {"type": "float", "required": False, "default": 0.0, "description": "Wall friction angle (degrees). Coulomb only. Alias: delta."},
            "beta_deg": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees). Aliases: beta, backfill_slope."},
            "alpha_deg": {"type": "float", "required": False, "default": 90.0, "description": "Wall back face angle from horizontal (degrees, 90 = vertical). Coulomb only. Alias: alpha."},
        },
        "returns": {"K": "Earth pressure coefficient value.", "coefficient": "Ka, Kp, or K0.", "theory": "Theory used."},
    },
    "mse_wall": {
        "category": "MSE Wall",
        "brief": "MSE wall external + internal stability analysis (GEC-11).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Wall height (m)."},
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
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surcharge (kPa)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees)."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation soil unit weight (kN/m3)."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg)."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa)."},
        },
        "returns": {"FOS_sliding": "Sliding FOS.", "FOS_overturning": "Overturning FOS.", "FOS_bearing": "Bearing FOS."},
    },
    "mse_lrfd_external_stability": {
        "category": "MSE Wall",
        "brief": "MSE wall external stability by AASHTO/GEC-11 LRFD: sliding / eccentricity / bearing capacity:demand ratios (CDRs) under Strength I (max+min) and Service I. Distinct from mse_wall, which reports ASD factors of safety.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Design wall height H (m), incl. embedment. Alias: height."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Reinforced fill unit weight (kN/m3). Alias: gamma."},
            "phi_backfill": {"type": "float", "required": True, "description": "Reinforced fill friction angle (deg). Alias: phi."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length L (m). Auto-sized (0.7H) if omitted."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Live-load surcharge q (kPa) over the mass (e.g. heq*gamma). Used as the LL unless live_load overrides."},
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
        "returns": {"sliding": "Sliding CDRs (strength max/min, critical, governing).", "eccentricity": "eL (strength max, critical) vs e_limit.", "bearing": "sigma_v (strength/critical/service) + bearing CDRs."},
    },
    "list_reinforcement": {
        "category": "MSE Wall",
        "brief": "List built-in reinforcement types with properties.",
        "parameters": {},
        "returns": {"reinforcement_types": "Dict of available reinforcement types."},
    },
}
