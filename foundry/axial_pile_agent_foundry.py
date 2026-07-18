"""
Axial Pile Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. axial_pile_agent        - Run an axial pile capacity calculation
  2. axial_pile_list_methods - Browse available methods
  3. axial_pile_describe_method - Get detailed parameter docs

Covers Nordlund, Tomlinson alpha, and Beta methods for driven piles.

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

from axial_pile.pile_types import (
    PileSection, make_pipe_pile, make_concrete_pile, make_h_pile,
)
from axial_pile.soil_profile import AxialSoilLayer, AxialSoilProfile
from axial_pile.capacity import AxialPileAnalysis


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


def _build_pile(params: dict) -> PileSection:
    """Build a PileSection from flat params."""
    pile_type = params.get("pile_type", "pipe_closed")

    if pile_type in ("pipe_closed", "pipe_open"):
        return make_pipe_pile(
            diameter=params["diameter"],
            thickness=params["wall_thickness"],
            closed_end=(pile_type == "pipe_closed"),
            E=params.get("E", 200e6),
        )
    elif pile_type in ("concrete_square", "concrete_circular"):
        shape = "square" if "square" in pile_type else "circular"
        return make_concrete_pile(
            width=params["width"],
            shape=shape,
            E=params.get("E", 25e6),
        )
    elif pile_type == "h_pile":
        return make_h_pile(
            designation=params["designation"],
            E=params.get("E", 200e6),
        )
    else:
        raise ValueError(f"Unknown pile_type '{pile_type}'. Use: pipe_closed, pipe_open, concrete_square, concrete_circular, h_pile")


def _build_soil_profile(params: dict) -> AxialSoilProfile:
    """Build soil profile from layers array."""
    layers = []
    for lay in params["layers"]:
        layers.append(AxialSoilLayer(
            thickness=lay["thickness"],
            soil_type=lay.get("soil_type", "cohesionless"),
            unit_weight=lay["unit_weight"],
            friction_angle=lay.get("friction_angle"),
            cohesion=lay.get("cohesion"),
            delta_phi_ratio=lay.get("delta_phi_ratio", 0.75),
            description=lay.get("description", ""),
        ))
    return AxialSoilProfile(
        layers=layers,
        gwt_depth=params.get("gwt_depth"),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_axial_pile_capacity(params: dict) -> dict:
    """Full axial pile capacity analysis."""
    pile = _build_pile(params)
    soil = _build_soil_profile(params)

    analysis = AxialPileAnalysis(
        pile=pile,
        soil=soil,
        pile_length=params["pile_length"],
        method=params.get("method", "auto"),
        factor_of_safety=params.get("factor_of_safety", 2.5),
        include_uplift=params.get("include_uplift", False),
    )
    result = analysis.compute()
    return result.to_dict()


def _run_capacity_vs_depth(params: dict) -> dict:
    """Capacity vs depth curve for pile length optimization."""
    pile = _build_pile(params)
    soil = _build_soil_profile(params)

    analysis = AxialPileAnalysis(
        pile=pile,
        soil=soil,
        pile_length=params.get("pile_length", soil.total_thickness),
        method=params.get("method", "auto"),
    )
    curve = analysis.capacity_vs_depth(
        depth_min=params.get("depth_min", 3.0),
        depth_max=params.get("depth_max"),
        n_points=params.get("n_points", 20),
    )
    return {"capacity_vs_depth": curve}


def _run_make_pile_section(params: dict) -> dict:
    """Create a pile section and return its properties."""
    pile = _build_pile(params)
    return {
        "name": pile.name,
        "pile_type": pile.pile_type,
        "area_m2": round(pile.area, 6),
        "perimeter_m": round(pile.perimeter, 4),
        "tip_area_m2": round(pile.tip_area, 6),
        "width_m": round(pile.width, 4),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "axial_pile_capacity": _run_axial_pile_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
    "make_pile_section": _run_make_pile_section,
}

METHOD_INFO = {
    "axial_pile_capacity": {
        "category": "Axial Capacity",
        "brief": "Full driven pile axial capacity analysis (Nordlund/Tomlinson/Beta).",
        "description": (
            "Computes ultimate axial capacity of a driven pile. Auto-selects Nordlund "
            "for cohesionless soils and Tomlinson alpha for cohesive soils. Beta (effective "
            "stress) method also available. Returns skin friction, end bearing, and "
            "per-layer breakdown."
        ),
        "reference": "FHWA GEC-12; Nordlund (1963); Tomlinson (1980); Meyerhof (1976)",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "description": "'pipe_closed', 'pipe_open', 'concrete_square', 'concrete_circular', or 'h_pile'."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m). For pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Wall thickness (m). For pipe piles."},
            "width": {"type": "float", "required": False, "description": "Width or side length (m). For concrete piles."},
            "designation": {"type": "str", "required": False, "description": "HP section (e.g. 'HP14x89'). For H-piles."},
            "E": {"type": "float", "required": False, "default": 200e6, "description": "Elastic modulus (kPa). Steel=200e6, concrete=25e6."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "layers": {"type": "array", "required": True, "description": (
                "Array of soil layers. Each: thickness (m), soil_type ('cohesionless' or 'cohesive'), "
                "unit_weight (kN/m3), friction_angle (deg, for sand), cohesion (kPa, for clay), "
                "delta_phi_ratio (default 0.75)."
            )},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "method": {"type": "str", "required": False, "default": "auto", "description": "'auto' (Nordlund+Tomlinson), or 'beta'."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "include_uplift": {"type": "bool", "required": False, "default": False, "description": "Compute tension (uplift) capacity."},
        },
        "returns": {
            "Q_ultimate_kN": "Ultimate compression capacity (kN).",
            "Q_skin_kN": "Total skin friction (kN).",
            "Q_tip_kN": "End bearing (kN).",
            "Q_allowable_kN": "Allowable capacity (kN).",
            "layer_breakdown": "Per-layer skin friction details.",
        },
        "related": {
            "pile_group_agent.group_efficiency": "Compute group reduction for pile groups.",
            "pile_group_agent.pile_group_simple": "Distribute loads across a pile group.",
            "downdrag_agent.downdrag_analysis": "Check downdrag if settling soils are present.",
            "wave_equation_agent.bearing_graph": "Generate blow count vs capacity for driving.",
            "drilled_shaft_agent.drilled_shaft_capacity": "Alternative: drilled shaft (not driven).",
        },
        "typical_workflow": (
            "1. Select pile type and section (make_pile_section)\n"
            "2. Compute single pile capacity (this method)\n"
            "3. Check group efficiency (pile_group_agent.group_efficiency)\n"
            "4. If settling soils: check downdrag (downdrag_agent.downdrag_analysis)\n"
            "5. Generate bearing graph for driving (wave_equation_agent.bearing_graph)"
        ),
        "common_mistakes": [
            "Forgetting gwt_depth — effective stress is critical for skin friction in sand.",
            "Using cohesion for sand layers or friction_angle for clay — match soil_type to parameters.",
            "H-pile requires 'designation' (e.g. 'HP14x89'), not diameter/width.",
        ],
    },
    "capacity_vs_depth": {
        "category": "Axial Capacity",
        "brief": "Capacity vs depth curve for pile length optimization.",
        "description": "Runs axial capacity at multiple depths to find optimal pile length.",
        "reference": "FHWA GEC-12, Chapter 8",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "description": "Same as axial_pile_capacity."},
            "diameter": {"type": "float", "required": False, "description": "For pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "For pipe piles."},
            "width": {"type": "float", "required": False, "description": "For concrete piles."},
            "designation": {"type": "str", "required": False, "description": "For H-piles."},
            "layers": {"type": "array", "required": True, "description": "Soil layers (same format as axial_pile_capacity)."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "method": {"type": "str", "required": False, "default": "auto"},
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Minimum depth (m)."},
            "depth_max": {"type": "float", "required": False, "description": "Maximum depth (m). Default = total soil thickness."},
            "n_points": {"type": "int", "required": False, "default": 20, "description": "Number of depth points."},
        },
        "returns": {
            "capacity_vs_depth": "Array of {depth_m, Q_ultimate_kN, Q_skin_kN, Q_tip_kN}.",
        },
    },
    "make_pile_section": {
        "category": "Pile Properties",
        "brief": "Create a pile section and return geometric properties.",
        "description": "Returns area, perimeter, tip area, and width for a pile section.",
        "reference": "AISC steel shapes; standard pile geometries",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "description": "'pipe_closed', 'pipe_open', 'concrete_square', 'concrete_circular', or 'h_pile'."},
            "diameter": {"type": "float", "required": False, "description": "For pipe piles (m)."},
            "wall_thickness": {"type": "float", "required": False, "description": "For pipe piles (m)."},
            "width": {"type": "float", "required": False, "description": "For concrete piles (m)."},
            "designation": {"type": "str", "required": False, "description": "For H-piles (e.g. 'HP14x89')."},
        },
        "returns": {
            "name": "Section name.",
            "area_m2": "Cross-sectional area (m2).",
            "perimeter_m": "Perimeter (m).",
            "tip_area_m2": "Tip area (m2).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def axial_pile_agent(method: str, parameters_json: str) -> str:
    """
    Driven pile axial capacity calculator.

    Computes ultimate and allowable axial capacity using Nordlund (sand),
    Tomlinson alpha (clay), or Beta (effective stress) methods. Supports
    pipe piles, concrete piles, and H-piles in multi-layer soil profiles.

    Parameters:
        method: The calculation method name. Use axial_pile_list_methods() to see options.
        parameters_json: JSON string of parameters. Use axial_pile_describe_method() for details.

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
def axial_pile_list_methods(category: str = "") -> str:
    """
    Lists available axial pile calculation methods.

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
def axial_pile_describe_method(method: str) -> str:
    """
    Returns detailed documentation for an axial pile method.

    Parameters:
        method: The method name (e.g. 'axial_pile_capacity', 'capacity_vs_depth').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
