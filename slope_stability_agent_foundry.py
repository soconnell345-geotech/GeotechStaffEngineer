"""
Slope Stability Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. slope_stability_agent           - Run slope stability analysis or search
  2. slope_stability_list_methods    - Browse available methods
  3. slope_stability_describe_method - Get detailed parameter docs

Implements limit equilibrium methods (Fellenius/Bishop/Spencer) with
circular slip surface analysis and critical surface grid search.
"""

import json
import math
import numpy as np
from functions.api import function

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import analyze_slope, search_critical_surface


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _clean_result(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [_clean_result(item) if isinstance(item, dict)
                          else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


def _build_geometry(params: dict) -> SlopeGeometry:
    """Build SlopeGeometry from flat JSON params."""
    soil_layers = []
    for d in params["soil_layers"]:
        soil_layers.append(SlopeSoilLayer(
            name=d["name"],
            top_elevation=d["top_elevation"],
            bottom_elevation=d["bottom_elevation"],
            gamma=d["gamma"],
            gamma_sat=d.get("gamma_sat"),
            phi=d.get("phi", 0.0),
            c_prime=d.get("c_prime", 0.0),
            cu=d.get("cu", 0.0),
            analysis_mode=d.get("analysis_mode", "drained"),
        ))

    surface_points = [tuple(pt) for pt in params["surface_points"]]

    gwt_points = None
    if params.get("gwt_points") is not None:
        gwt_points = [tuple(pt) for pt in params["gwt_points"]]

    surcharge_x_range = None
    if params.get("surcharge_x_range") is not None:
        surcharge_x_range = tuple(params["surcharge_x_range"])

    return SlopeGeometry(
        surface_points=surface_points,
        soil_layers=soil_layers,
        gwt_points=gwt_points,
        surcharge=params.get("surcharge", 0.0),
        surcharge_x_range=surcharge_x_range,
        reinforcement_force=params.get("reinforcement_force", 0.0),
        reinforcement_elevation=params.get("reinforcement_elevation"),
        kh=params.get("kh", 0.0),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_analyze_slope(params: dict) -> dict:
    geom = _build_geometry(params)
    result = analyze_slope(
        geom=geom,
        xc=params["xc"],
        yc=params["yc"],
        radius=params["radius"],
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30),
        FOS_required=params.get("FOS_required", 1.5),
        include_slice_data=params.get("include_slice_data", False),
        compare_methods=params.get("compare_methods", False),
    )
    return result.to_dict()


def _run_search_critical_surface(params: dict) -> dict:
    geom = _build_geometry(params)

    x_range = None
    if params.get("x_range") is not None:
        x_range = tuple(params["x_range"])

    y_range = None
    if params.get("y_range") is not None:
        y_range = tuple(params["y_range"])

    result = search_critical_surface(
        geom=geom,
        x_range=x_range,
        y_range=y_range,
        nx=params.get("nx", 10),
        ny=params.get("ny", 10),
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30),
        FOS_required=params.get("FOS_required", 1.5),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "analyze_slope": _run_analyze_slope,
    "search_critical_surface": _run_search_critical_surface,
}

METHOD_INFO = {
    "analyze_slope": {
        "category": "Slope Stability",
        "brief": "Single slip surface analysis using Fellenius, Bishop, or Spencer method.",
        "description": (
            "Analyzes a circular slip surface at a specified center (xc, yc) and radius R "
            "using the method of slices. Returns factor of safety (FOS), entry/exit points, "
            "and optionally per-slice data and comparison across all three methods."
        ),
        "reference": (
            "Fellenius (1927); Bishop (1955) Geotechnique Vol.5; "
            "Spencer (1967) Geotechnique Vol.17; Duncan, Wright & Brandon (2014)"
        ),
        "parameters": {
            "xc": {"type": "float", "required": True,
                   "description": "Circle center x-coordinate (m)."},
            "yc": {"type": "float", "required": True,
                   "description": "Circle center elevation (m)."},
            "radius": {"type": "float", "required": True,
                       "description": "Circle radius (m)."},
            "method": {"type": "str", "required": False, "default": "bishop",
                       "description": "'fellenius', 'bishop', or 'spencer'."},
            "n_slices": {"type": "int", "required": False, "default": 30,
                         "description": "Number of slices."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5,
                             "description": "Minimum required FOS for pass/fail."},
            "include_slice_data": {"type": "bool", "required": False, "default": False,
                                   "description": "Include per-slice breakdown."},
            "compare_methods": {"type": "bool", "required": False, "default": False,
                                "description": "Compute FOS for all three methods."},
            "surface_points": {
                "type": "array", "required": True,
                "description": "Ground surface as [[x, z], ...] sorted left-to-right."},
            "soil_layers": {
                "type": "array", "required": True,
                "description": (
                    "Array of soil layers. Each: name (str), top_elevation (float), "
                    "bottom_elevation (float), gamma (float, kN/m3), "
                    "gamma_sat (float, optional), phi (float, degrees, default 0), "
                    "c_prime (float, kPa, default 0), cu (float, kPa, default 0), "
                    "analysis_mode ('drained' or 'undrained', default 'drained')."
                ),
            },
            "gwt_points": {
                "type": "array", "required": False,
                "description": "Water table as [[x, z_gwt], ...]. None = no water."},
            "surcharge": {"type": "float", "required": False, "default": 0.0,
                          "description": "Uniform surcharge on surface (kPa)."},
            "surcharge_x_range": {"type": "array", "required": False,
                                  "description": "[x_start, x_end] for surcharge."},
            "kh": {"type": "float", "required": False, "default": 0.0,
                   "description": "Horizontal seismic coefficient."},
            "reinforcement_force": {"type": "float", "required": False, "default": 0.0,
                                    "description": "Horizontal reinforcement force (kN/m)."},
            "reinforcement_elevation": {"type": "float", "required": False,
                                        "description": "Elevation of reinforcement (m)."},
        },
        "returns": {
            "FOS": "Factor of safety.",
            "method": "Method name used.",
            "is_stable": "True if FOS >= FOS_required.",
            "xc_m": "Circle center x (m).",
            "yc_m": "Circle center y (m).",
            "radius_m": "Circle radius (m).",
            "x_entry_m": "Slip surface entry x (m).",
            "x_exit_m": "Slip surface exit x (m).",
            "n_slices": "Number of slices used.",
            "has_seismic": "Whether seismic load was applied.",
            "kh": "Seismic coefficient.",
            "FOS_fellenius": "Fellenius FOS (if compare_methods=true).",
            "FOS_bishop": "Bishop FOS (if compare_methods=true).",
            "theta_spencer_deg": "Spencer interslice angle (if Spencer used).",
            "slice_data": "Per-slice data (if include_slice_data=true).",
        },
    },
    "search_critical_surface": {
        "category": "Slope Stability",
        "brief": "Grid search for the critical slip surface (minimum FOS).",
        "description": (
            "Searches a grid of circle centers (xc, yc) and optimizes the radius "
            "at each center to find the minimum factor of safety."
        ),
        "reference": "Duncan, Wright & Brandon (2014) Chapter 14",
        "parameters": {
            "x_range": {"type": "array", "required": False,
                        "description": "[x_min, x_max] for center search. Auto if omitted."},
            "y_range": {"type": "array", "required": False,
                        "description": "[y_min, y_max] for center search. Auto if omitted."},
            "nx": {"type": "int", "required": False, "default": 10,
                   "description": "Number of x grid points."},
            "ny": {"type": "int", "required": False, "default": 10,
                   "description": "Number of y grid points."},
            "method": {"type": "str", "required": False, "default": "bishop",
                       "description": "'fellenius', 'bishop', or 'spencer'."},
            "n_slices": {"type": "int", "required": False, "default": 30,
                         "description": "Number of slices per analysis."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5,
                             "description": "Required FOS for pass/fail."},
            "surface_points": {"type": "array", "required": True,
                               "description": "Same as analyze_slope."},
            "soil_layers": {"type": "array", "required": True,
                            "description": "Same as analyze_slope."},
            "gwt_points": {"type": "array", "required": False,
                           "description": "Same as analyze_slope."},
            "surcharge": {"type": "float", "required": False, "default": 0.0,
                          "description": "Same as analyze_slope."},
            "kh": {"type": "float", "required": False, "default": 0.0,
                   "description": "Same as analyze_slope."},
        },
        "returns": {
            "n_surfaces_evaluated": "Total number of trial surfaces.",
            "critical": "Dict with FOS, method, circle geometry for the minimum FOS surface.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry agent functions
# ---------------------------------------------------------------------------

@function
def slope_stability_agent(method: str, parameters_json: str) -> str:
    """
    Slope stability analysis calculator.

    Analyzes slope stability using limit equilibrium methods (Fellenius,
    Bishop, Spencer) with circular slip surfaces.

    Parameters:
        method: The calculation method name. Use slope_stability_list_methods() to see options.
        parameters_json: JSON string of parameters. Use slope_stability_describe_method() for details.

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
def slope_stability_list_methods(category: str = "") -> str:
    """
    Lists available slope stability calculation methods.

    Parameters:
        category: Optional filter by category.

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
def slope_stability_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a slope stability method.

    Parameters:
        method: The method name (e.g. 'analyze_slope', 'search_critical_surface').

    Returns:
        JSON string with parameters, types, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
