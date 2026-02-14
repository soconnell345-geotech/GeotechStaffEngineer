"""
Wave Equation Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. wave_equation_agent        - Run a wave equation analysis
  2. wave_equation_list_methods - Browse available methods
  3. wave_equation_describe_method - Get detailed parameter docs

Covers Smith 1-D wave equation: single blow, bearing graph,
drivability study, and hammer database lookup.
"""

import json
import math
import numpy as np
from functions.api import function

from wave_equation.hammer import Hammer, get_hammer, list_hammers
from wave_equation.cushion import Cushion, make_cushion_from_properties
from wave_equation.pile_model import discretize_pile
from wave_equation.soil_model import SoilSetup
from wave_equation.time_integration import simulate_blow
from wave_equation.bearing_graph import generate_bearing_graph
from wave_equation.drivability import drivability_study


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


def _build_hammer(params: dict) -> Hammer:
    """Build hammer from name or custom parameters."""
    if "hammer_name" in params:
        return get_hammer(params["hammer_name"])
    return Hammer(
        name=params.get("hammer_custom_name", "Custom"),
        ram_weight=params["ram_weight"],
        stroke=params["stroke"],
        efficiency=params.get("efficiency", 0.67),
        hammer_type=params.get("hammer_type", "single_acting"),
        rated_energy=params.get("rated_energy"),
    )


def _build_cushion(params: dict) -> Cushion:
    """Build cushion from stiffness or material properties."""
    if "cushion_stiffness" in params:
        return Cushion(
            stiffness=params["cushion_stiffness"],
            cor=params.get("cushion_cor", 0.80),
        )
    elif "cushion_area" in params:
        return make_cushion_from_properties(
            area=params["cushion_area"],
            thickness=params["cushion_thickness"],
            elastic_modulus=params["cushion_E"],
            cor=params.get("cushion_cor", 0.80),
        )
    else:
        # Default: reasonable cushion for typical steel pile
        return Cushion(stiffness=500000, cor=0.80)


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_single_blow(params: dict) -> dict:
    """Simulate a single hammer blow."""
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)

    pile = discretize_pile(
        length=params["pile_length"],
        area=params["pile_area"],
        elastic_modulus=params.get("pile_E", 200e6),
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", 78.5),
    )

    soil = SoilSetup(
        R_ultimate=params["R_ultimate"],
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.50),
    )

    result = simulate_blow(
        hammer, cushion, pile, soil,
        helmet_weight=params.get("helmet_weight", 5.0),
        max_time=params.get("max_time", 0.10),
    )

    blow_count = 1.0 / result.permanent_set if result.permanent_set > 1e-6 else 1e6

    return {
        "permanent_set_m": round(result.permanent_set, 6),
        "permanent_set_mm": round(result.permanent_set * 1000, 2),
        "blow_count_per_m": round(blow_count, 0),
        "max_compression_stress_kPa": round(result.max_compression_stress, 0),
        "max_compression_stress_MPa": round(result.max_compression_stress / 1000, 1),
        "max_tension_stress_kPa": round(result.max_tension_stress, 0),
        "max_pile_force_kN": round(result.max_pile_force, 1),
        "n_time_steps": result.n_steps,
        "hammer_name": hammer.name,
        "hammer_energy_kNm": round(hammer.energy, 1),
        "R_ultimate_kN": params["R_ultimate"],
    }


def _run_bearing_graph(params: dict) -> dict:
    """Generate a bearing graph."""
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)

    pile = discretize_pile(
        length=params["pile_length"],
        area=params["pile_area"],
        elastic_modulus=params.get("pile_E", 200e6),
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", 78.5),
    )

    bg = generate_bearing_graph(
        hammer, cushion, pile,
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.50),
        R_min=params.get("R_min", 200.0),
        R_max=params.get("R_max", 2000.0),
        R_step=params.get("R_step", 200.0),
        helmet_weight=params.get("helmet_weight", 5.0),
        max_time=params.get("max_time", 0.10),
    )

    return bg.to_dict()


def _run_drivability(params: dict) -> dict:
    """Run drivability study."""
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)

    result = drivability_study(
        hammer, cushion,
        pile_area=params["pile_area"],
        pile_E=params.get("pile_E", 200e6),
        pile_unit_weight=params.get("pile_unit_weight", 78.5),
        depths=params["depths"],
        R_at_depth=params["R_at_depth"],
        skin_fractions=params.get("skin_fractions"),
        segment_length=params.get("segment_length", 1.0),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.50),
        helmet_weight=params.get("helmet_weight", 5.0),
        refusal_blow_count=params.get("refusal_blow_count", 3000.0),
    )

    return result.to_dict()


def _run_list_hammers(params: dict) -> dict:
    """List all available hammers with properties."""
    hammers = {}
    for name in list_hammers():
        h = get_hammer(name)
        hammers[name] = {
            "ram_weight_kN": h.ram_weight,
            "stroke_m": h.stroke,
            "rated_energy_kNm": round(h.energy, 1),
            "efficiency": h.efficiency,
            "hammer_type": h.hammer_type,
            "impact_velocity_m_s": round(h.impact_velocity, 2),
        }
    return {"hammers": hammers}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "single_blow": _run_single_blow,
    "bearing_graph": _run_bearing_graph,
    "drivability": _run_drivability,
    "list_available_hammers": _run_list_hammers,
}

METHOD_INFO = {
    "single_blow": {
        "category": "Wave Equation",
        "brief": "Simulate a single hammer blow on a pile using Smith's 1-D wave equation.",
        "description": (
            "Performs explicit time-stepping simulation of a hammer blow on a pile "
            "embedded in soil. Returns permanent set, blow count, and maximum driving "
            "stresses. Uses Smith (1960) soil model with quake and damping parameters."
        ),
        "reference": "Smith (1960); FHWA GEC-12, Chapter 12; WEAP87 Manual",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Built-in hammer name (e.g. 'Vulcan 010'). Use list_available_hammers to see options."},
            "ram_weight": {"type": "float", "required": False, "description": "Custom ram weight (kN). Use if not using hammer_name."},
            "stroke": {"type": "float", "required": False, "description": "Custom stroke height (m)."},
            "efficiency": {"type": "float", "required": False, "default": 0.67, "description": "Hammer efficiency (0-1)."},
            "cushion_stiffness": {"type": "float", "required": False, "description": "Pile cushion stiffness (kN/m). Default 500,000."},
            "cushion_cor": {"type": "float", "required": False, "default": 0.80, "description": "Cushion coefficient of restitution (0-1)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-sectional area (m2)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Steel=200e6."},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5, "description": "Pile material unit weight (kN/m3). Steel=78.5."},
            "segment_length": {"type": "float", "required": False, "default": 1.0, "description": "Pile segment length for discretization (m)."},
            "R_ultimate": {"type": "float", "required": True, "description": "Total ultimate soil resistance (kN)."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Fraction of Rult as skin friction (0-1)."},
            "quake_side": {"type": "float", "required": False, "default": 0.0025, "description": "Side quake (m). Typical 2.5mm."},
            "quake_toe": {"type": "float", "required": False, "default": 0.0025, "description": "Toe quake (m). Typical 2.5mm."},
            "damping_side": {"type": "float", "required": False, "default": 0.16, "description": "Side Smith damping (s/m). Sand=0.16, clay=0.65."},
            "damping_toe": {"type": "float", "required": False, "default": 0.50, "description": "Toe Smith damping (s/m). Typical 0.50."},
            "helmet_weight": {"type": "float", "required": False, "default": 5.0, "description": "Helmet/drive cap weight (kN)."},
        },
        "returns": {
            "permanent_set_mm": "Permanent set per blow (mm).",
            "blow_count_per_m": "Blow count (blows/m).",
            "max_compression_stress_MPa": "Max compression stress (MPa).",
            "max_tension_stress_kPa": "Max tension stress (kPa).",
            "max_pile_force_kN": "Max compressive force in pile (kN).",
        },
    },
    "bearing_graph": {
        "category": "Wave Equation",
        "brief": "Generate bearing graph (ultimate capacity vs blow count) for a hammer-pile-soil system.",
        "description": (
            "Runs wave equation simulations at multiple ultimate resistance values. "
            "Produces the classic bearing graph showing Rult vs blow count, plus "
            "maximum driving stresses at each resistance level."
        ),
        "reference": "FHWA GEC-12, Section 12.5",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Built-in hammer name."},
            "ram_weight": {"type": "float", "required": False, "description": "Custom ram weight (kN)."},
            "stroke": {"type": "float", "required": False, "description": "Custom stroke (m)."},
            "efficiency": {"type": "float", "required": False, "default": 0.67},
            "cushion_stiffness": {"type": "float", "required": False, "description": "Cushion stiffness (kN/m)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5},
            "quake_side": {"type": "float", "required": False, "default": 0.0025},
            "quake_toe": {"type": "float", "required": False, "default": 0.0025},
            "damping_side": {"type": "float", "required": False, "default": 0.16},
            "damping_toe": {"type": "float", "required": False, "default": 0.50},
            "R_min": {"type": "float", "required": False, "default": 200, "description": "Minimum Rult (kN)."},
            "R_max": {"type": "float", "required": False, "default": 2000, "description": "Maximum Rult (kN)."},
            "R_step": {"type": "float", "required": False, "default": 200, "description": "Rult step size (kN)."},
        },
        "returns": {
            "R_values_kN": "Ultimate resistance values analyzed.",
            "blow_counts_per_m": "Blow count at each Rult.",
            "permanent_sets_m": "Set per blow at each Rult.",
            "max_comp_stresses_kPa": "Max compression stress at each Rult.",
            "max_tens_stresses_kPa": "Max tension stress at each Rult.",
        },
    },
    "drivability": {
        "category": "Wave Equation",
        "brief": "Drivability study: blow count and stresses at multiple pile penetration depths.",
        "description": (
            "Assesses whether a given hammer can drive a pile to the required depth. "
            "Runs wave equation at multiple depths with corresponding resistance values. "
            "Reports blow count, driving stresses, and whether refusal is reached."
        ),
        "reference": "FHWA GEC-12, Section 12.6",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Built-in hammer name."},
            "ram_weight": {"type": "float", "required": False, "description": "Custom ram weight (kN)."},
            "stroke": {"type": "float", "required": False, "description": "Custom stroke (m)."},
            "cushion_stiffness": {"type": "float", "required": False},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5},
            "depths": {"type": "array", "required": True, "description": "Penetration depths to analyze (m)."},
            "R_at_depth": {"type": "array", "required": True, "description": "Ultimate resistance at each depth (kN)."},
            "skin_fractions": {"type": "array", "required": False, "description": "Skin fraction at each depth. Default 0.5 for all."},
            "refusal_blow_count": {"type": "float", "required": False, "default": 3000, "description": "Blow count considered as refusal (blows/m)."},
        },
        "returns": {
            "can_drive": "True if pile can be driven to all depths.",
            "refusal_depth_m": "Depth at which refusal occurs (0 if none).",
            "points": "Per-depth results: depth, Rult, set, blow_count, stresses.",
        },
    },
    "list_available_hammers": {
        "category": "Hammer Database",
        "brief": "List all built-in hammers with their properties.",
        "description": "Returns a database of common pile driving hammers (Vulcan, Delmag, ICE).",
        "reference": "Manufacturer specifications",
        "parameters": {},
        "returns": {
            "hammers": "Dictionary of hammer names to properties (ram weight, stroke, energy, type).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def wave_equation_agent(method: str, parameters_json: str) -> str:
    """
    Wave equation analysis for pile driving.

    Simulates pile driving using Smith's 1-D wave equation. Predicts
    blow count, driving stresses, bearing graphs, and drivability
    for hammer-pile-soil systems.

    Parameters:
        method: The calculation method name. Use wave_equation_list_methods() to see options.
        parameters_json: JSON string of parameters. Use wave_equation_describe_method() for details.

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
def wave_equation_list_methods(category: str = "") -> str:
    """
    Lists available wave equation methods.

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
def wave_equation_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a wave equation method.

    Parameters:
        method: The method name (e.g. 'single_blow', 'bearing_graph').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
