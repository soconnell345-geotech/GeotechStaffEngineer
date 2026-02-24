"""
Sheet Pile Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. sheet_pile_agent        - Run a sheet pile wall analysis
  2. sheet_pile_list_methods - Browse available methods
  3. sheet_pile_describe_method - Get detailed parameter docs

Covers cantilever and anchored sheet pile wall design using
free earth support method with Rankine or Coulomb pressures.

FOUNDRY SETUP:
  - pip install geotech-staff-engineer (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import math
import numpy as np
try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from sheet_pile.earth_pressure import (
    rankine_Ka, rankine_Kp, coulomb_Ka, coulomb_Kp, K0,
    active_pressure, passive_pressure, tension_crack_depth,
)
from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
from sheet_pile.anchored import analyze_anchored


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


def _build_soil_layers(params: dict) -> list:
    """Build WallSoilLayer list from JSON array."""
    layers = []
    for lay in params["layers"]:
        layers.append(WallSoilLayer(
            thickness=lay["thickness"],
            unit_weight=lay["unit_weight"],
            friction_angle=lay.get("friction_angle", 30.0),
            cohesion=lay.get("cohesion", 0.0),
            description=lay.get("description", ""),
        ))
    return layers


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_cantilever_wall(params: dict) -> dict:
    """Cantilever sheet pile wall analysis."""
    layers = _build_soil_layers(params)
    result = analyze_cantilever(
        excavation_depth=params["excavation_depth"],
        soil_layers=layers,
        gwt_depth_active=params.get("gwt_depth_active"),
        gwt_depth_passive=params.get("gwt_depth_passive"),
        surcharge=params.get("surcharge", 0.0),
        FOS_passive=params.get("FOS_passive", 1.5),
        gamma_w=params.get("gamma_w", 9.81),
        pressure_method=params.get("pressure_method", "rankine"),
    )
    return result.to_dict()


def _run_anchored_wall(params: dict) -> dict:
    """Anchored (tied-back) sheet pile wall analysis."""
    layers = _build_soil_layers(params)
    result = analyze_anchored(
        excavation_depth=params["excavation_depth"],
        anchor_depth=params["anchor_depth"],
        soil_layers=layers,
        gwt_depth_active=params.get("gwt_depth_active"),
        gwt_depth_passive=params.get("gwt_depth_passive"),
        surcharge=params.get("surcharge", 0.0),
        FOS_passive=params.get("FOS_passive", 1.5),
        gamma_w=params.get("gamma_w", 9.81),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "anchored_wall": _run_anchored_wall,
}

METHOD_INFO = {
    "cantilever_wall": {
        "category": "Sheet Pile Walls",
        "brief": "Cantilever sheet pile wall design using free earth support method.",
        "description": (
            "Determines required embedment depth, total wall length, and maximum "
            "bending moment for a cantilever sheet pile wall. Uses iterative moment "
            "balance about the wall base with Rankine or Coulomb earth pressure "
            "coefficients. Applies 1.2x design factor to computed embedment per USACE."
        ),
        "reference": "USACE EM 1110-2-2504; USS Steel Sheet Piling Design Manual",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth H (m)."},
            "layers": {"type": "array", "required": True, "description": (
                "Array of soil layers. Each: thickness (m), unit_weight (kN/m3), "
                "friction_angle (deg, default 30), cohesion (kPa, default 0)."
            )},
            "gwt_depth_active": {"type": "float", "required": False, "description": "GWT depth on active side (m from top)."},
            "gwt_depth_passive": {"type": "float", "required": False, "description": "GWT depth on passive side (m from excavation)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge on active side (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "Factor of safety on passive resistance."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "description": "'rankine' or 'coulomb'."},
        },
        "returns": {
            "embedment_depth_m": "Required embedment below excavation (m).",
            "total_wall_length_m": "Total wall length (m).",
            "max_moment_kNm_per_m": "Maximum bending moment (kN-m/m of wall).",
            "max_moment_depth_m": "Depth of max moment from wall top (m).",
        },
        "related": {
            "anchored_wall": "Alternative: anchored sheet pile for deeper excavations.",
            "retaining_walls_agent.cantilever_wall": "Alternative: concrete cantilever wall.",
        },
        "common_mistakes": [
            "FOS_passive is applied to passive resistance only — typical value is 1.5-2.0.",
            "Layers must extend below the excavation depth for passive resistance.",
        ],
    },
    "anchored_wall": {
        "category": "Sheet Pile Walls",
        "brief": "Anchored sheet pile wall design using free earth support method.",
        "description": (
            "Determines required embedment depth, anchor force, and maximum bending "
            "moment for an anchored sheet pile wall. Takes moments about the anchor "
            "point for embedment, then horizontal equilibrium for anchor force."
        ),
        "reference": "USACE EM 1110-2-2504; USS Steel Sheet Piling Design Manual",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth H (m)."},
            "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth from wall top (m). Must be < excavation_depth."},
            "layers": {"type": "array", "required": True, "description": (
                "Array of soil layers. Each: thickness (m), unit_weight (kN/m3), "
                "friction_angle (deg, default 30), cohesion (kPa, default 0)."
            )},
            "gwt_depth_active": {"type": "float", "required": False, "description": "GWT depth on active side (m from top)."},
            "gwt_depth_passive": {"type": "float", "required": False, "description": "GWT depth on passive side (m from excavation)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge on active side (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "Factor of safety on passive resistance."},
        },
        "returns": {
            "embedment_depth_m": "Required embedment below excavation (m).",
            "total_wall_length_m": "Total wall length (m).",
            "anchor_force_kN_per_m": "Required anchor force (kN per meter of wall).",
            "max_moment_kNm_per_m": "Maximum bending moment (kN-m/m of wall).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def sheet_pile_agent(method: str, parameters_json: str) -> str:
    """
    Sheet pile wall design calculator.

    Designs cantilever and anchored sheet pile walls using the free earth
    support method. Computes required embedment, anchor forces, and
    maximum bending moments.

    Parameters:
        method: The calculation method name. Use sheet_pile_list_methods() to see options.
        parameters_json: JSON string of parameters. Use sheet_pile_describe_method() for details.

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
def sheet_pile_list_methods(category: str = "") -> str:
    """
    Lists available sheet pile wall methods.

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return json.dumps(result)


@function
def sheet_pile_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a sheet pile method.

    Parameters:
        method: The method name (e.g. 'cantilever_wall', 'anchored_wall').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
