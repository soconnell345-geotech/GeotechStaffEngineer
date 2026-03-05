"""Retaining walls adapter — cantilever + MSE wall analysis."""

from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.analysis import analyze_cantilever_wall, analyze_mse_wall
from retaining_walls.reinforcement import (
    Reinforcement, RIBBED_STEEL_STRIP_75x4, WELDED_WIRE_GRID_W11,
    GEOGRID_UX1600, GEOGRID_UX1700,
)

_REINFORCEMENT_DB = {
    "ribbed_steel_strip_75x4": RIBBED_STEEL_STRIP_75x4,
    "welded_wire_grid_w11": WELDED_WIRE_GRID_W11,
    "geogrid_ux1600": GEOGRID_UX1600,
    "geogrid_ux1700": GEOGRID_UX1700,
}


def _build_reinforcement(params):
    name = params.get("reinforcement_name", "").lower()
    if name in _REINFORCEMENT_DB:
        return _REINFORCEMENT_DB[name]
    return Reinforcement(
        name=params.get("reinforcement_name", "Custom"),
        type=params.get("reinforcement_type", "geosynthetic"),
        Tallowable=params["reinforcement_Tallowable"],
        width=params.get("reinforcement_width", 0.05),
        Fy=params.get("reinforcement_Fy", 0.0),
        thickness=params.get("reinforcement_thickness", 0.0),
    )


def _run_cantilever_wall(params):
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


def _run_list_reinforcement(params):
    result = {}
    for key, r in _REINFORCEMENT_DB.items():
        result[key] = {"name": r.name, "type": r.type, "Tallowable_kN_per_m": r.Tallowable}
    return {"reinforcement_types": result}


METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "mse_wall": _run_mse_wall,
    "list_reinforcement": _run_list_reinforcement,
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
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surcharge (kPa)."},
        },
        "returns": {"FOS_sliding": "Factor of safety against sliding.", "FOS_overturning": "Factor of safety against overturning."},
    },
    "mse_wall": {
        "category": "MSE Wall",
        "brief": "MSE wall external + internal stability analysis (GEC-11).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Wall height (m)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "reinforcement_name": {"type": "str", "required": False, "description": "Built-in name or custom. Use list_reinforcement to see options."},
        },
        "returns": {"FOS_sliding": "Sliding FOS.", "FOS_overturning": "Overturning FOS.", "FOS_bearing": "Bearing FOS."},
    },
    "list_reinforcement": {
        "category": "MSE Wall",
        "brief": "List built-in reinforcement types with properties.",
        "parameters": {},
        "returns": {"reinforcement_types": "Dict of available reinforcement types."},
    },
}
