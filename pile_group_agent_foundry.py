"""
Pile Group Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. pile_group_agent        - Run a pile group analysis
  2. pile_group_list_methods - Browse available methods
  3. pile_group_describe_method - Get detailed parameter docs

Covers rigid cap pile group analysis (simplified and 6-DOF),
group efficiency (Converse-Labarre), and block failure.
"""

import json
import math
import numpy as np
from functions.api import function

from pile_group.pile_layout import GroupPile, create_rectangular_layout
from pile_group.group_efficiency import (
    converse_labarre, block_failure_capacity, p_multiplier,
)
from pile_group.rigid_cap import (
    GroupLoad, analyze_vertical_group_simple, analyze_group_6dof,
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
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [_clean_result(item) if isinstance(item, dict) else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


def _build_piles(params: dict) -> list:
    """Build pile list from params. Supports rectangular layout or custom piles."""
    if "n_rows" in params and "n_cols" in params:
        # Rectangular layout
        piles = create_rectangular_layout(
            n_rows=params["n_rows"],
            n_cols=params["n_cols"],
            spacing_x=params["spacing_x"],
            spacing_y=params["spacing_y"],
            axial_stiffness=params.get("axial_stiffness"),
            lateral_stiffness=params.get("lateral_stiffness"),
        )
        # Apply capacities if given
        comp_cap = params.get("axial_capacity_compression")
        tens_cap = params.get("axial_capacity_tension")
        if comp_cap:
            for p in piles:
                p.axial_capacity_compression = comp_cap
        if tens_cap:
            for p in piles:
                p.axial_capacity_tension = tens_cap
        return piles
    elif "piles" in params:
        # Custom pile list
        piles = []
        for pd in params["piles"]:
            piles.append(GroupPile(
                x=pd["x"],
                y=pd["y"],
                batter_x=pd.get("batter_x", 0.0),
                batter_y=pd.get("batter_y", 0.0),
                axial_stiffness=pd.get("axial_stiffness"),
                lateral_stiffness=pd.get("lateral_stiffness"),
                axial_capacity_compression=pd.get("axial_capacity_compression"),
                axial_capacity_tension=pd.get("axial_capacity_tension"),
                label=pd.get("label", ""),
            ))
        return piles
    else:
        raise ValueError("Provide either n_rows/n_cols for rectangular layout, or 'piles' array for custom.")


def _build_load(params: dict) -> GroupLoad:
    """Build GroupLoad from params."""
    return GroupLoad(
        Vx=params.get("Vx", 0.0),
        Vy=params.get("Vy", 0.0),
        Vz=params.get("Vz", 0.0),
        Mx=params.get("Mx", 0.0),
        My=params.get("My", 0.0),
        Mz=params.get("Mz", 0.0),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_pile_group_simple(params: dict) -> dict:
    """Simplified elastic vertical analysis."""
    piles = _build_piles(params)
    load = _build_load(params)
    result = analyze_vertical_group_simple(piles, load)
    return result.to_dict()


def _run_pile_group_6dof(params: dict) -> dict:
    """General 6-DOF rigid cap analysis."""
    piles = _build_piles(params)
    load = _build_load(params)
    result = analyze_group_6dof(piles, load)
    return result.to_dict()


def _run_group_efficiency(params: dict) -> dict:
    """Group efficiency calculations."""
    results = {}

    # Converse-Labarre
    if "n_rows" in params and "n_cols" in params:
        Eg = converse_labarre(
            params["n_rows"], params["n_cols"],
            params["pile_diameter"], params["spacing"],
        )
        results["converse_labarre_Eg"] = round(Eg, 4)

    # Block failure
    if "pile_length" in params and "cu" in params:
        Qb = block_failure_capacity(
            params["n_rows"], params["n_cols"],
            params.get("spacing_x", params.get("spacing", 0)),
            params.get("spacing_y", params.get("spacing", 0)),
            params["pile_length"],
            params["cu"],
            params["pile_diameter"],
        )
        results["block_failure_kN"] = round(Qb, 1)

    # P-multiplier
    if "row_position" in params:
        sd = params.get("spacing_diameter_ratio", params.get("spacing", 3.0) / params.get("pile_diameter", 0.3))
        pm = p_multiplier(params["row_position"], sd)
        results["p_multiplier"] = round(pm, 3)

    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "pile_group_simple": _run_pile_group_simple,
    "pile_group_6dof": _run_pile_group_6dof,
    "group_efficiency": _run_group_efficiency,
}

METHOD_INFO = {
    "pile_group_simple": {
        "category": "Pile Group Analysis",
        "brief": "Simplified elastic method for vertical pile groups: Pi = V/n +/- M*xi/SUM(xi^2).",
        "description": (
            "Distributes vertical load and overturning moments to individual piles "
            "using the standard pile group equation. For vertical piles only. "
            "Ignores lateral loads and torsion."
        ),
        "reference": "USACE EM 1110-2-2906; FHWA GEC-12, Chapter 9",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Number of rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Number of columns for rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X-direction spacing (m)."},
            "spacing_y": {"type": "float", "required": False, "description": "Y-direction spacing (m)."},
            "axial_stiffness": {"type": "float", "required": False, "description": "Pile axial stiffness (kN/m)."},
            "axial_capacity_compression": {"type": "float", "required": False, "description": "Compression capacity per pile (kN)."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "Tension capacity per pile (kN)."},
            "piles": {"type": "array", "required": False, "description": "Custom pile array. Each: x (m), y (m), axial_stiffness, axial_capacity_compression, label."},
            "Vx": {"type": "float", "required": False, "default": 0, "description": "Horizontal force X (kN)."},
            "Vy": {"type": "float", "required": False, "default": 0, "description": "Horizontal force Y (kN)."},
            "Vz": {"type": "float", "required": False, "default": 0, "description": "Vertical force (kN, positive down)."},
            "Mx": {"type": "float", "required": False, "default": 0, "description": "Moment about X (kN-m)."},
            "My": {"type": "float", "required": False, "default": 0, "description": "Moment about Y (kN-m)."},
            "Mz": {"type": "float", "required": False, "default": 0, "description": "Torsion about Z (kN-m)."},
        },
        "returns": {
            "n_piles": "Number of piles.",
            "pile_forces": "Per-pile axial forces and utilization ratios.",
            "max_compression_kN": "Maximum compression in any pile (kN).",
            "max_tension_kN": "Maximum tension in any pile (kN).",
            "max_utilization": "Maximum utilization ratio.",
        },
    },
    "pile_group_6dof": {
        "category": "Pile Group Analysis",
        "brief": "General 6-DOF rigid cap analysis for vertical and battered piles.",
        "description": (
            "Assembles a 6x6 group stiffness matrix from individual pile stiffnesses, "
            "solves for cap displacements, and back-calculates pile forces. "
            "Supports battered piles. Requires axial_stiffness on all piles."
        ),
        "reference": "CPGA User's Guide (ITL-89-4); USACE EM 1110-2-2906, Chapter 4",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Number of rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Number of columns for rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X-direction spacing (m)."},
            "spacing_y": {"type": "float", "required": False, "description": "Y-direction spacing (m)."},
            "axial_stiffness": {"type": "float", "required": True, "description": "Pile axial stiffness (kN/m). Required for 6-DOF."},
            "lateral_stiffness": {"type": "float", "required": False, "description": "Pile lateral stiffness (kN/m)."},
            "piles": {"type": "array", "required": False, "description": "Custom pile array. Each: x, y, batter_x, batter_y, axial_stiffness, lateral_stiffness."},
            "Vx": {"type": "float", "required": False, "default": 0}, "Vy": {"type": "float", "required": False, "default": 0},
            "Vz": {"type": "float", "required": False, "default": 0}, "Mx": {"type": "float", "required": False, "default": 0},
            "My": {"type": "float", "required": False, "default": 0}, "Mz": {"type": "float", "required": False, "default": 0},
        },
        "returns": {
            "cap_displacements": "Cap displacements {dx, dy, dz, rx, ry, rz} in m and rad.",
            "pile_forces": "Per-pile forces and utilization.",
            "max_compression_kN": "Max compression (kN).",
            "max_utilization": "Max utilization ratio.",
        },
    },
    "group_efficiency": {
        "category": "Group Efficiency",
        "brief": "Converse-Labarre efficiency, block failure capacity, and lateral p-multiplier.",
        "description": (
            "Calculates pile group efficiency factor using the Converse-Labarre formula, "
            "block failure capacity for cohesive soils, and lateral p-multiplier for "
            "group effects on lateral resistance."
        ),
        "reference": "FHWA GEC-12, Eq 9-1; Brown et al. (1988)",
        "parameters": {
            "n_rows": {"type": "int", "required": True, "description": "Number of rows."},
            "n_cols": {"type": "int", "required": True, "description": "Number of columns."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter or width (m)."},
            "spacing": {"type": "float", "required": True, "description": "Center-to-center spacing (m)."},
            "pile_length": {"type": "float", "required": False, "description": "Pile length (m). For block failure."},
            "cu": {"type": "float", "required": False, "description": "Undrained shear strength (kPa). For block failure."},
            "row_position": {"type": "int", "required": False, "description": "Row position (1=leading). For p-multiplier."},
            "spacing_diameter_ratio": {"type": "float", "required": False, "description": "s/D ratio. For p-multiplier."},
        },
        "returns": {
            "converse_labarre_Eg": "Group efficiency (0-1).",
            "block_failure_kN": "Block failure capacity (kN).",
            "p_multiplier": "Lateral p-multiplier (0-1).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def pile_group_agent(method: str, parameters_json: str) -> str:
    """
    Pile group analysis calculator.

    Analyzes pile groups with rigid caps under combined axial, lateral,
    and moment loading. Distributes loads to individual piles and checks
    utilization against capacity.

    Parameters:
        method: The calculation method name. Use pile_group_list_methods() to see options.
        parameters_json: JSON string of parameters. Use pile_group_describe_method() for details.

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
def pile_group_list_methods(category: str = "") -> str:
    """
    Lists available pile group calculation methods.

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
def pile_group_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a pile group method.

    Parameters:
        method: The method name (e.g. 'pile_group_simple', 'group_efficiency').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
