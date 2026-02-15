"""
Retaining Walls Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. retaining_walls_agent        - Run a retaining wall analysis
  2. retaining_walls_list_methods - Browse available methods
  3. retaining_walls_describe_method - Get detailed parameter docs

Covers cantilever retaining walls and MSE (Mechanically Stabilized Earth) walls.
"""

import json
import math
from functions.api import function

from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.mse import analyze_mse_wall
from retaining_walls.reinforcement import (
    Reinforcement,
    RIBBED_STEEL_STRIP_75x4, WELDED_WIRE_GRID_W11,
    GEOGRID_UX1600, GEOGRID_UX1700,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _clean_result(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [_clean_result(item) if isinstance(item, dict) else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


# Built-in reinforcement lookup
_REINFORCEMENT_DB = {
    "ribbed_steel_strip_75x4": RIBBED_STEEL_STRIP_75x4,
    "welded_wire_grid_w11": WELDED_WIRE_GRID_W11,
    "geogrid_ux1600": GEOGRID_UX1600,
    "geogrid_ux1700": GEOGRID_UX1700,
}


def _build_reinforcement(params: dict) -> Reinforcement:
    """Build Reinforcement from params or lookup built-in."""
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


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_cantilever_wall(params: dict) -> dict:
    """Full cantilever retaining wall stability analysis."""
    geom = CantileverWallGeometry(
        wall_height=params["wall_height"],
        base_width=params.get("base_width"),
        toe_length=params.get("toe_length"),
        stem_thickness_top=params.get("stem_thickness_top", 0.30),
        stem_thickness_base=params.get("stem_thickness_base"),
        base_thickness=params.get("base_thickness", 0.60),
        has_shear_key=params.get("has_shear_key", False),
        key_depth=params.get("key_depth", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
    )

    result = analyze_cantilever_wall(
        geom,
        gamma_backfill=params["gamma_backfill"],
        phi_backfill=params["phi_backfill"],
        c_backfill=params.get("c_backfill", 0.0),
        phi_foundation=params.get("phi_foundation"),
        c_foundation=params.get("c_foundation", 0.0),
        q_allowable=params.get("q_allowable"),
        gamma_concrete=params.get("gamma_concrete", 24.0),
        FOS_sliding=params.get("FOS_sliding_required", 1.5),
        FOS_overturning=params.get("FOS_overturning_required", 2.0),
        pressure_method=params.get("pressure_method", "rankine"),
    )

    output = result.to_dict()
    output["geometry"] = {
        "wall_height_m": geom.wall_height,
        "base_width_m": geom.base_width,
        "toe_length_m": geom.toe_length,
        "heel_length_m": round(geom.heel_length, 3),
        "stem_thickness_top_m": geom.stem_thickness_top,
        "stem_thickness_base_m": geom.stem_thickness_base,
        "base_thickness_m": geom.base_thickness,
    }
    return output


def _run_mse_wall(params: dict) -> dict:
    """Full MSE wall analysis (external + internal stability)."""
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        reinforcement_spacing=params.get("reinforcement_spacing", 0.60),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
    )

    reinforcement = _build_reinforcement(params)

    result = analyze_mse_wall(
        geom,
        gamma_backfill=params["gamma_backfill"],
        phi_backfill=params["phi_backfill"],
        reinforcement=reinforcement,
        gamma_foundation=params.get("gamma_foundation"),
        phi_foundation=params.get("phi_foundation"),
        c_foundation=params.get("c_foundation", 0.0),
        q_allowable=params.get("q_allowable"),
    )

    output = result.to_dict()
    output["geometry"] = {
        "wall_height_m": geom.wall_height,
        "reinforcement_length_m": geom.reinforcement_length,
        "reinforcement_spacing_m": geom.reinforcement_spacing,
        "n_reinforcement_levels": geom.n_reinforcement_levels,
    }
    output["reinforcement_name"] = reinforcement.name
    return output


def _run_list_reinforcement(params: dict) -> dict:
    """List built-in reinforcement types."""
    result = {}
    for key, r in _REINFORCEMENT_DB.items():
        result[key] = {
            "name": r.name,
            "type": r.type,
            "Tallowable_kN_per_m": r.Tallowable,
            "is_metallic": r.is_metallic,
        }
    return {"reinforcement_types": result}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "mse_wall": _run_mse_wall,
    "list_reinforcement": _run_list_reinforcement,
}

METHOD_INFO = {
    "cantilever_wall": {
        "category": "Cantilever Wall",
        "brief": "Full cantilever retaining wall external stability analysis (sliding, overturning, bearing).",
        "description": (
            "Checks external stability of a cantilever (or gravity) retaining wall. "
            "Includes sliding (FOS_sliding), overturning about the toe (FOS_overturning), "
            "and bearing capacity/eccentricity checks. Supports Rankine or Coulomb "
            "earth pressure methods, sloping backfill, surcharge, and shear key. "
            "Auto-sizes geometry if not fully specified."
        ),
        "reference": "AASHTO LRFD Section 11.6; Das, Principles of Foundation Engineering, Ch 13",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Retained height H (m)."},
            "base_width": {"type": "float", "required": False, "description": "Base slab width B (m). Auto-sized to ~0.6*H if omitted."},
            "toe_length": {"type": "float", "required": False, "description": "Toe projection (m). Auto-sized to ~0.1*B if omitted."},
            "stem_thickness_top": {"type": "float", "required": False, "default": 0.30, "description": "Stem thickness at top (m)."},
            "stem_thickness_base": {"type": "float", "required": False, "description": "Stem thickness at base (m). Auto-sized if omitted."},
            "base_thickness": {"type": "float", "required": False, "default": 0.60, "description": "Base slab thickness (m)."},
            "has_shear_key": {"type": "bool", "required": False, "default": False, "description": "Whether wall has a shear key."},
            "key_depth": {"type": "float", "required": False, "default": 0.0, "description": "Shear key depth below base (m)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surcharge on backfill (kPa)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "c_backfill": {"type": "float", "required": False, "default": 0.0, "description": "Backfill cohesion (kPa)."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (degrees). Default = phi_backfill."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa). Omit to skip bearing FOS check."},
            "gamma_concrete": {"type": "float", "required": False, "default": 24.0, "description": "Concrete unit weight (kN/m3)."},
            "FOS_sliding_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS for sliding."},
            "FOS_overturning_required": {"type": "float", "required": False, "default": 2.0, "description": "Required FOS for overturning."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "description": "'rankine' or 'coulomb'."},
        },
        "returns": {
            "FOS_sliding": "Factor of safety against sliding.",
            "FOS_overturning": "Factor of safety against overturning.",
            "FOS_bearing": "Factor of safety against bearing failure.",
            "passes_sliding": "True if FOS_sliding >= required.",
            "passes_overturning": "True if FOS_overturning >= required.",
            "passes_bearing": "True if bearing is adequate.",
            "q_toe_kPa": "Maximum bearing pressure at toe (kPa).",
            "q_heel_kPa": "Bearing pressure at heel (kPa).",
            "eccentricity_m": "Eccentricity of resultant (m).",
            "in_middle_third": "True if resultant is within middle third.",
            "geometry": "Auto-sized wall geometry details.",
        },
    },
    "mse_wall": {
        "category": "MSE Wall",
        "brief": "Full MSE wall analysis: external stability + internal stability per GEC-11.",
        "description": (
            "Checks external stability (sliding, overturning, bearing) treating the "
            "reinforced zone as a rigid block, and internal stability (tensile rupture "
            "and pullout) at each reinforcement level. Supports metallic strips/grids "
            "and geosynthetic reinforcement. Implements Kr/Ka ratio, F*, and effective "
            "length calculations per FHWA GEC-11."
        ),
        "reference": "FHWA GEC-11 (FHWA-NHI-10-024), Chapters 4-5; AASHTO LRFD Section 11.10",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Total wall height H (m)."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length L (m). Auto-sized to max(0.7*H, 2.5) if omitted."},
            "reinforcement_spacing": {"type": "float", "required": False, "default": 0.60, "description": "Vertical spacing Sv (m)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Slope above wall (degrees)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform surcharge (kPa)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees). Min 34° for select backfill."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation unit weight (kN/m3). Default = gamma_backfill."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation friction angle (degrees). Default = phi_backfill."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa)."},
            "reinforcement_name": {"type": "str", "required": False, "description": (
                "Built-in name: 'ribbed_steel_strip_75x4', 'welded_wire_grid_w11', "
                "'geogrid_ux1600', 'geogrid_ux1700'. Or provide custom params below."
            )},
            "reinforcement_type": {"type": "str", "required": False, "description": "'metallic_strip', 'metallic_grid', or 'geosynthetic'. For custom reinforcement."},
            "reinforcement_Tallowable": {"type": "float", "required": False, "description": "Allowable tensile strength per unit width (kN/m). Required for custom."},
        },
        "returns": {
            "FOS_sliding": "External FOS against sliding.",
            "FOS_overturning": "External FOS against overturning.",
            "FOS_bearing": "External FOS against bearing.",
            "passes_external": "True if all external checks pass.",
            "all_pass_internal": "True if all reinforcement levels pass.",
            "n_levels": "Number of reinforcement levels.",
            "internal_results": "Per-level: depth, Tmax, Pr, FOS_pullout, FOS_rupture, passes.",
            "geometry": "Wall geometry details.",
        },
    },
    "list_reinforcement": {
        "category": "Reinforcement",
        "brief": "List built-in MSE wall reinforcement types and properties.",
        "description": "Returns available pre-defined reinforcement products (steel strips, wire grids, geogrids).",
        "reference": "FHWA GEC-11, Chapter 3",
        "parameters": {},
        "returns": {
            "reinforcement_types": "Dict of reinforcement name -> properties (type, Tallowable, is_metallic).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def retaining_walls_agent(method: str, parameters_json: str) -> str:
    """
    Retaining wall design and stability checker.

    Analyzes cantilever retaining walls and MSE (Mechanically Stabilized
    Earth) walls for external stability (sliding, overturning, bearing)
    and internal stability (reinforcement pullout and rupture for MSE).

    Parameters:
        method: The calculation method name. Use retaining_walls_list_methods() to see options.
        parameters_json: JSON string of parameters. Use retaining_walls_describe_method() for details.

    Returns:
        JSON string with calculation results or an error message.
    """
    try:
        parameters = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    try:
        result = METHOD_REGISTRY[method](parameters)
        return json.dumps(_clean_result(result), default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def retaining_walls_list_methods(category: str = "") -> str:
    """
    Lists available retaining wall methods.

    Parameters:
        category: Optional filter by category (e.g. 'Cantilever Wall', 'MSE Wall').

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return json.dumps(result)


@function
def retaining_walls_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a retaining wall method.

    Parameters:
        method: The method name (e.g. 'cantilever_wall', 'mse_wall').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
