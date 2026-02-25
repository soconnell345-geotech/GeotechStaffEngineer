"""
Drilled Shaft Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. drilled_shaft_agent        - Run a drilled shaft capacity calculation
  2. drilled_shaft_list_methods - Browse available methods
  3. drilled_shaft_describe_method - Get detailed parameter docs

Covers GEC-10 alpha/beta/rock socket methods for drilled shafts.

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

from drilled_shaft.shaft import DrillShaft
from drilled_shaft.soil_profile import ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.capacity import DrillShaftAnalysis
from drilled_shaft.lrfd import apply_lrfd, RESISTANCE_FACTORS


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
            cleaned[k] = [_clean_result(item) if isinstance(item, dict) else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


def _build_shaft(params: dict) -> DrillShaft:
    """Build DrillShaft from flat params."""
    return DrillShaft(
        diameter=params["diameter"],
        length=params["shaft_length"],
        socket_diameter=params.get("socket_diameter"),
        socket_length=params.get("socket_length", 0.0),
        bell_diameter=params.get("bell_diameter"),
        casing_depth=params.get("casing_depth", 0.0),
        concrete_fc=params.get("concrete_fc", 28000.0),
    )


def _build_soil_profile(params: dict) -> ShaftSoilProfile:
    """Build soil profile from layers array."""
    layers = []
    for lay in params["layers"]:
        layers.append(ShaftSoilLayer(
            thickness=lay["thickness"],
            soil_type=lay["soil_type"],
            unit_weight=lay["unit_weight"],
            cu=lay.get("cu", 0.0),
            phi=lay.get("phi", 0.0),
            N60=lay.get("N60", 0.0),
            qu=lay.get("qu", 0.0),
            RQD=lay.get("RQD", 100.0),
            description=lay.get("description", ""),
        ))
    return ShaftSoilProfile(
        layers=layers,
        gwt_depth=params.get("gwt_depth"),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_drilled_shaft_capacity(params: dict) -> dict:
    """Full drilled shaft capacity analysis."""
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)

    analysis = DrillShaftAnalysis(
        shaft=shaft,
        soil=soil,
        factor_of_safety=params.get("factor_of_safety", 2.5),
    )
    result = analysis.compute()
    return result.to_dict()


def _run_capacity_vs_depth(params: dict) -> dict:
    """Capacity vs depth curve for shaft length optimization."""
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)

    analysis = DrillShaftAnalysis(
        shaft=shaft,
        soil=soil,
        factor_of_safety=params.get("factor_of_safety", 2.5),
    )
    curve = analysis.capacity_vs_depth(
        depth_min=params.get("depth_min", 3.0),
        depth_max=params.get("depth_max"),
        n_points=params.get("n_points", 20),
    )
    return {"capacity_vs_depth": curve}


def _run_lrfd_capacity(params: dict) -> dict:
    """Drilled shaft capacity with LRFD resistance factors."""
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)

    analysis = DrillShaftAnalysis(
        shaft=shaft,
        soil=soil,
        factor_of_safety=1.0,  # LRFD uses phi factors, not FOS
    )
    result = analysis.compute()

    # Determine tip soil type
    tip_soil_type = params.get("tip_soil_type", "cohesive")
    lrfd = apply_lrfd(result, tip_soil_type)

    output = result.to_dict()
    output["lrfd"] = lrfd
    return output


def _run_get_resistance_factors(params: dict) -> dict:
    """Return AASHTO LRFD resistance factors for drilled shafts."""
    return {"resistance_factors": RESISTANCE_FACTORS}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "drilled_shaft_capacity": _run_drilled_shaft_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
    "lrfd_capacity": _run_lrfd_capacity,
    "get_resistance_factors": _run_get_resistance_factors,
}

METHOD_INFO = {
    "drilled_shaft_capacity": {
        "category": "Drilled Shaft Capacity",
        "brief": "Full drilled shaft axial capacity analysis using GEC-10 methods (alpha/beta/rock socket).",
        "description": (
            "Computes ultimate and allowable axial capacity of a drilled shaft. "
            "Auto-selects alpha method for cohesive soils, beta method for cohesionless "
            "soils, and rock socket method for rock layers. Applies GEC-10 exclusion "
            "zones (top 1.5m, bottom 1D for clay, casing zone). Returns total capacity, "
            "skin/tip breakdown, and per-layer details."
        ),
        "reference": "FHWA GEC-10 (FHWA-NHI-10-016); O'Neill & Reese (1999)",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "shaft_length": {"type": "float", "required": True, "description": "Total shaft embedment length (m)."},
            "socket_diameter": {"type": "float", "required": False, "description": "Rock socket diameter (m). Default same as shaft."},
            "socket_length": {"type": "float", "required": False, "default": 0.0, "description": "Rock socket length (m)."},
            "bell_diameter": {"type": "float", "required": False, "description": "Bell diameter at base (m). Omit if no bell."},
            "casing_depth": {"type": "float", "required": False, "default": 0.0, "description": "Permanent casing depth (m). Excluded from side resistance."},
            "concrete_fc": {"type": "float", "required": False, "default": 28000, "description": "Concrete f'c (kPa). Default 28 MPa."},
            "layers": {"type": "array", "required": True, "description": (
                "Array of soil layers from top to bottom. Each layer: "
                "thickness (m), soil_type ('cohesive', 'cohesionless', or 'rock'), "
                "unit_weight (kN/m3), cu (kPa, for clay), phi (degrees, for sand), "
                "N60 (SPT, for sand tip), qu (kPa, for rock UCS), RQD (%, for rock)."
            )},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
        },
        "returns": {
            "Q_ultimate_kN": "Ultimate axial capacity (kN).",
            "Q_skin_kN": "Total skin friction (kN).",
            "Q_tip_kN": "End bearing (kN).",
            "Q_allowable_kN": "Allowable capacity = Q_ult / FOS (kN).",
            "Q_side_clay_kN": "Side resistance from cohesive layers (kN).",
            "Q_side_sand_kN": "Side resistance from cohesionless layers (kN).",
            "Q_side_rock_kN": "Side resistance from rock layers (kN).",
            "layer_breakdown": "Per-layer details: depth, soil type, method, fs, side_resistance.",
        },
        "related": {
            "lrfd_capacity": "Apply LRFD resistance factors to this capacity.",
            "capacity_vs_depth": "Optimize shaft length with capacity-depth curve.",
            "axial_pile_agent.axial_pile_capacity": "Alternative: driven pile capacity.",
        },
        "typical_workflow": (
            "1. Select shaft diameter and length\n"
            "2. Compute capacity (this method)\n"
            "3. Apply LRFD factors (lrfd_capacity)\n"
            "4. Check settlement if needed (settlement_agent)"
        ),
        "common_mistakes": [
            "Using cohesion instead of cu for clay layers — drilled shaft uses 'cu' parameter.",
            "Forgetting gwt_depth — affects effective stress for beta method in sand.",
            "Sand layers need both phi and N60 for the beta method.",
        ],
    },
    "capacity_vs_depth": {
        "category": "Drilled Shaft Capacity",
        "brief": "Capacity vs depth curve for shaft length optimization.",
        "description": "Runs capacity analysis at multiple shaft lengths to find optimal depth.",
        "reference": "FHWA GEC-10, Chapter 14",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "shaft_length": {"type": "float", "required": True, "description": "Maximum shaft length (m)."},
            "layers": {"type": "array", "required": True, "description": "Soil layers (same format as drilled_shaft_capacity)."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Minimum depth (m)."},
            "depth_max": {"type": "float", "required": False, "description": "Maximum depth (m). Default = shaft_length."},
            "n_points": {"type": "int", "required": False, "default": 20, "description": "Number of evaluation points."},
        },
        "returns": {
            "capacity_vs_depth": "Array of {depth_m, Q_ultimate_kN, Q_skin_kN, Q_tip_kN}.",
        },
    },
    "lrfd_capacity": {
        "category": "Drilled Shaft Capacity",
        "brief": "Drilled shaft capacity with AASHTO LRFD resistance factors applied.",
        "description": (
            "Runs full capacity analysis then applies AASHTO resistance factors "
            "(phi = 0.45 for clay side, 0.55 for sand/rock side, 0.40-0.50 for tip). "
            "Returns both unfactored and factored resistances."
        ),
        "reference": "AASHTO LRFD Table 10.5.5.2.4-1; FHWA GEC-10 Table 13-1",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "shaft_length": {"type": "float", "required": True, "description": "Total shaft length (m)."},
            "layers": {"type": "array", "required": True, "description": "Soil layers (same format as drilled_shaft_capacity)."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "tip_soil_type": {"type": "str", "required": False, "default": "cohesive",
                              "description": "'cohesive', 'cohesionless', or 'rock' at shaft tip."},
        },
        "returns": {
            "Q_ultimate_kN": "Unfactored ultimate capacity (kN).",
            "lrfd.phi_Qn_kN": "Total factored resistance (kN).",
            "lrfd.phi_Qs_clay_kN": "Factored clay side resistance (kN).",
            "lrfd.phi_Qs_sand_kN": "Factored sand side resistance (kN).",
            "lrfd.phi_Qs_rock_kN": "Factored rock side resistance (kN).",
            "lrfd.phi_Qt_kN": "Factored tip resistance (kN).",
        },
    },
    "get_resistance_factors": {
        "category": "LRFD",
        "brief": "Returns AASHTO LRFD resistance factors for drilled shaft design.",
        "description": "Lookup table of phi-factors for side and tip resistance by soil type.",
        "reference": "AASHTO LRFD Table 10.5.5.2.4-1",
        "parameters": {},
        "returns": {
            "resistance_factors": "Dict of component -> phi factor (e.g. side_cohesive: 0.45).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def drilled_shaft_agent(method: str, parameters_json: str) -> str:
    """
    Drilled shaft (bored pile) axial capacity calculator.

    Computes ultimate and allowable axial capacity using FHWA GEC-10
    methods: alpha (clay), beta (sand), and rock socket. Supports
    belled shafts, permanent casing, LRFD resistance factors, and
    capacity vs depth optimization.

    Parameters:
        method: The calculation method name. Use drilled_shaft_list_methods() to see options.
        parameters_json: JSON string of parameters. Use drilled_shaft_describe_method() for details.

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
def drilled_shaft_list_methods(category: str = "") -> str:
    """
    Lists available drilled shaft calculation methods.

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
def drilled_shaft_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a drilled shaft method.

    Parameters:
        method: The method name (e.g. 'drilled_shaft_capacity', 'lrfd_capacity').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
