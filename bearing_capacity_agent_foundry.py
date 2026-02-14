"""
Bearing Capacity Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. bearing_capacity_agent        - Run a bearing capacity calculation
  2. bearing_capacity_list_methods - Browse available methods
  3. bearing_capacity_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - No external dependencies beyond numpy (no groundhog needed)
"""

import json
import math
import numpy as np
from functions.api import function

from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    bearing_capacity_Nc, bearing_capacity_Nq, bearing_capacity_Ngamma,
)


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
    return {k: _clean_value(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_bearing_capacity_analysis(params: dict) -> dict:
    """Full bearing capacity analysis from flat JSON parameters."""
    # Build footing
    footing = Footing(
        width=params["width"],
        length=params.get("length"),
        depth=params.get("depth", 0.0),
        shape=params.get("shape", "strip"),
        base_tilt=params.get("base_tilt", 0.0),
        eccentricity_B=params.get("eccentricity_B", 0.0),
        eccentricity_L=params.get("eccentricity_L", 0.0),
    )

    # Build soil layer 1
    layer1 = SoilLayer(
        cohesion=params.get("cohesion", 0.0),
        friction_angle=params.get("friction_angle", 0.0),
        unit_weight=params["unit_weight"],
        thickness=params.get("layer1_thickness"),
        description=params.get("layer1_description", ""),
    )

    # Optional layer 2
    layer2 = None
    if "layer2_unit_weight" in params:
        layer2 = SoilLayer(
            cohesion=params.get("layer2_cohesion", 0.0),
            friction_angle=params.get("layer2_friction_angle", 0.0),
            unit_weight=params["layer2_unit_weight"],
            description=params.get("layer2_description", ""),
        )

    soil = BearingSoilProfile(
        layer1=layer1,
        layer2=layer2,
        gwt_depth=params.get("gwt_depth"),
    )

    analysis = BearingCapacityAnalysis(
        footing=footing,
        soil=soil,
        load_inclination=params.get("load_inclination", 0.0),
        ground_slope=params.get("ground_slope", 0.0),
        vertical_load=params.get("vertical_load", 0.0),
        factor_of_safety=params.get("factor_of_safety", 3.0),
        ngamma_method=params.get("ngamma_method", "vesic"),
        factor_method=params.get("factor_method", "vesic"),
    )

    result = analysis.compute()
    return result.to_dict()


def _run_bearing_capacity_factors(params: dict) -> dict:
    """Quick lookup of Nc, Nq, Ngamma for a given friction angle."""
    phi = params["friction_angle"]
    method = params.get("method", "vesic")

    Nc = bearing_capacity_Nc(phi)
    Nq = bearing_capacity_Nq(phi)
    Ngamma = bearing_capacity_Ngamma(phi, method)

    return {
        "Nc": round(Nc, 2),
        "Nq": round(Nq, 2),
        "Ngamma": round(Ngamma, 2),
        "method": method,
        "friction_angle_deg": phi,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "bearing_capacity_analysis": _run_bearing_capacity_analysis,
    "bearing_capacity_factors": _run_bearing_capacity_factors,
}

METHOD_INFO = {
    "bearing_capacity_analysis": {
        "category": "Bearing Capacity",
        "brief": "Full bearing capacity analysis for a shallow foundation using the general bearing capacity equation.",
        "description": (
            "Computes ultimate and allowable bearing capacity using Vesic, Meyerhof, or Hansen "
            "methods. Includes shape, depth, inclination, base tilt, and ground slope factors. "
            "Supports two-layer systems (Meyerhof & Hanna), groundwater effects, and eccentric loading."
        ),
        "reference": "FHWA GEC-6; FHWA-SA-94-034 (CBEAR); Meyerhof & Hanna (1978)",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Footing width B (m)."},
            "length": {"type": "float", "required": False, "description": "Footing length L (m). Omit for strip or provide for rectangular/square."},
            "depth": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth Df (m)."},
            "shape": {"type": "str", "required": False, "default": "strip", "description": "'strip', 'rectangular', 'square', or 'circular'."},
            "cohesion": {"type": "float", "required": False, "default": 0.0, "description": "Soil cohesion c or cu (kPa). Use cu for undrained."},
            "friction_angle": {"type": "float", "required": False, "default": 0.0, "description": "Soil friction angle phi (degrees). Use 0 for undrained clay."},
            "unit_weight": {"type": "float", "required": True, "description": "Soil total unit weight gamma (kN/m3)."},
            "layer1_thickness": {"type": "float", "required": False, "description": "Thickness of layer 1 (m). Omit if semi-infinite."},
            "layer2_cohesion": {"type": "float", "required": False, "description": "Layer 2 cohesion (kPa). Include to enable two-layer analysis."},
            "layer2_friction_angle": {"type": "float", "required": False, "description": "Layer 2 friction angle (degrees)."},
            "layer2_unit_weight": {"type": "float", "required": False, "description": "Layer 2 unit weight (kN/m3). Must be provided to enable two-layer."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater table depth from surface (m). Omit if no GWT."},
            "eccentricity_B": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity in B direction (m)."},
            "eccentricity_L": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity in L direction (m)."},
            "load_inclination": {"type": "float", "required": False, "default": 0.0, "description": "Load angle from vertical (degrees)."},
            "ground_slope": {"type": "float", "required": False, "default": 0.0, "description": "Ground slope angle (degrees)."},
            "base_tilt": {"type": "float", "required": False, "default": 0.0, "description": "Footing base tilt angle (degrees)."},
            "vertical_load": {"type": "float", "required": False, "default": 0.0, "description": "Total vertical load (kN). Needed for Vesic inclination factors."},
            "factor_of_safety": {"type": "float", "required": False, "default": 3.0, "description": "Factor of safety for allowable capacity."},
            "ngamma_method": {"type": "str", "required": False, "default": "vesic", "description": "'vesic', 'meyerhof', or 'hansen'."},
            "factor_method": {"type": "str", "required": False, "default": "vesic", "description": "'vesic' or 'meyerhof' for correction factors."},
        },
        "returns": {
            "q_ultimate_kPa": "Ultimate bearing capacity (kPa).",
            "q_allowable_kPa": "Allowable bearing capacity (kPa).",
            "q_net_kPa": "Net ultimate capacity (kPa).",
            "Nc, Nq, Ngamma": "Bearing capacity factors.",
            "term_cohesion, term_overburden, term_selfweight": "Individual equation terms (kPa).",
        },
    },
    "bearing_capacity_factors": {
        "category": "Bearing Capacity",
        "brief": "Quick lookup of Nc, Nq, Ngamma bearing capacity factors for a given friction angle.",
        "description": "Returns the three classical bearing capacity factors without running a full analysis.",
        "reference": "Vesic (1973); Meyerhof (1963); Hansen (1970)",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "0 to 50", "description": "Soil friction angle phi (degrees)."},
            "method": {"type": "str", "required": False, "default": "vesic", "description": "'vesic', 'meyerhof', or 'hansen' for Ngamma."},
        },
        "returns": {
            "Nc": "Cohesion factor.",
            "Nq": "Overburden factor.",
            "Ngamma": "Self-weight factor.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def bearing_capacity_agent(method: str, parameters_json: str) -> str:
    """
    Shallow foundation bearing capacity calculator.

    Computes ultimate and allowable bearing capacity using the general
    bearing capacity equation (Vesic/Meyerhof/Hansen). Supports shape,
    depth, inclination, base tilt, ground slope, groundwater, eccentric
    loading, and two-layer soil systems.

    Parameters:
        method: The calculation method name. Use bearing_capacity_list_methods() to see options.
        parameters_json: JSON string of parameters. Use bearing_capacity_describe_method() for details.

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
def bearing_capacity_list_methods(category: str = "") -> str:
    """
    Lists available bearing capacity calculation methods.

    Parameters:
        category: Optional filter (not used — all methods are Bearing Capacity).

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
def bearing_capacity_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a bearing capacity method.

    Parameters:
        method: The method name (e.g. 'bearing_capacity_analysis').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
