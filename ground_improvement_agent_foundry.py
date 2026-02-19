"""
Ground Improvement Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. ground_improvement_agent           - Run a ground improvement analysis
  2. ground_improvement_list_methods    - Browse available methods
  3. ground_improvement_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - No external dependencies beyond numpy
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

from ground_improvement import (
    analyze_aggregate_piers,
    analyze_wick_drains,
    design_drain_spacing,
    analyze_surcharge_preloading,
    analyze_vibro_compaction,
    evaluate_feasibility,
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
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _clean_result(result):
    if isinstance(result, dict):
        return {k: _clean_value(v) if not isinstance(v, dict) else _clean_result(v)
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_aggregate_piers(params):
    result = analyze_aggregate_piers(**params)
    return _clean_result(result.to_dict())


def _run_wick_drains(params):
    result = analyze_wick_drains(**params)
    return _clean_result(result.to_dict())


def _run_design_drain_spacing(params):
    result = design_drain_spacing(**params)
    return _clean_result(result.to_dict())


def _run_surcharge_preloading(params):
    result = analyze_surcharge_preloading(**params)
    return _clean_result(result.to_dict())


def _run_vibro_compaction(params):
    result = analyze_vibro_compaction(**params)
    return _clean_result(result.to_dict())


def _run_feasibility(params):
    result = evaluate_feasibility(**params)
    return _clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "aggregate_piers": _run_aggregate_piers,
    "wick_drains": _run_wick_drains,
    "design_drain_spacing": _run_design_drain_spacing,
    "surcharge_preloading": _run_surcharge_preloading,
    "vibro_compaction": _run_vibro_compaction,
    "feasibility": _run_feasibility,
}

METHOD_INFO = {
    "aggregate_piers": {
        "category": "Column Reinforcement",
        "brief": "Aggregate pier (stone column) analysis per GEC-13.",
        "description": (
            "Computes area replacement ratio, stress concentration ratio, "
            "composite modulus, settlement reduction factor, and improved "
            "bearing capacity for aggregate pier reinforcement."
        ),
        "reference": "FHWA GEC-13: Ground Modification Methods",
        "parameters": {
            "column_diameter": {"type": "float", "required": True, "description": "Column diameter (m)."},
            "spacing": {"type": "float", "required": True, "description": "Center-to-center spacing (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "'triangular' or 'square'."},
            "E_column": {"type": "float", "required": False, "default": 80000, "description": "Column modulus (kPa)."},
            "E_soil": {"type": "float", "required": False, "default": 5000, "description": "Soil modulus (kPa)."},
            "n": {"type": "float", "required": False, "default": 5.0, "description": "Stress concentration ratio."},
            "q_unreinforced": {"type": "float", "required": False, "default": 0, "description": "Unreinforced bearing capacity (kPa)."},
            "S_unreinforced": {"type": "float", "required": False, "default": 0, "description": "Unreinforced settlement (mm)."},
        },
        "returns": {
            "area_replacement_ratio": "Ratio of column area to tributary area.",
            "stress_concentration_ratio": "Stress in column / stress in soil.",
            "composite_modulus_kPa": "Weighted modulus of reinforced zone (kPa).",
            "settlement_reduction_factor": "Settlement reduction ratio.",
            "improved_bearing_kPa": "Improved bearing capacity (kPa).",
            "settlement_improved_mm": "Improved settlement (mm).",
        },
    },
    "wick_drains": {
        "category": "Drainage",
        "brief": "Wick drain consolidation analysis (Barron/Hansbo).",
        "description": (
            "Computes radial and vertical consolidation with PVDs using "
            "Barron/Hansbo solution. Includes smear zone and well resistance effects."
        ),
        "reference": "FHWA GEC-13; Hansbo (1979); Barron (1948)",
        "parameters": {
            "spacing": {"type": "float", "required": True, "description": "Center-to-center drain spacing (m)."},
            "ch": {"type": "float", "required": True, "description": "Horizontal coefficient of consolidation (m2/year)."},
            "cv": {"type": "float", "required": True, "description": "Vertical coefficient of consolidation (m2/year)."},
            "Hdr": {"type": "float", "required": True, "description": "Maximum drainage path length (m)."},
            "time": {"type": "float", "required": True, "description": "Time for consolidation (years)."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "'triangular' or 'square'."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone diameter / drain diameter."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Undisturbed / smear permeability ratio."},
            "n_time_points": {"type": "int", "required": False, "default": 50, "description": "Points in time-settlement curve."},
        },
        "returns": {
            "Ur_percent": "Radial consolidation (%).",
            "Uv_percent": "Vertical consolidation (%).",
            "U_total_percent": "Combined consolidation (%).",
            "spacing_ratio_n": "n = de/dw spacing ratio.",
            "F_n": "Drain spacing function.",
        },
    },
    "design_drain_spacing": {
        "category": "Drainage",
        "brief": "Inverse design: find wick drain spacing for target consolidation and time.",
        "description": (
            "Uses bisection to find the drain spacing that achieves a target "
            "degree of consolidation within a specified time frame."
        ),
        "reference": "FHWA GEC-13",
        "parameters": {
            "target_U": {"type": "float", "required": True, "description": "Target consolidation (%)."},
            "target_time": {"type": "float", "required": True, "description": "Time available (years)."},
            "ch": {"type": "float", "required": True, "description": "Horizontal cv (m2/year)."},
            "cv": {"type": "float", "required": True, "description": "Vertical cv (m2/year)."},
            "Hdr": {"type": "float", "required": True, "description": "Maximum drainage path (m)."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "'triangular' or 'square'."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio."},
            "spacing_range": {"type": "list", "required": False, "default": [1.0, 3.5], "description": "Search range [min, max] (m)."},
        },
        "returns": {
            "spacing": "Optimized drain spacing (m).",
            "U_total_percent": "Achieved consolidation (%).",
        },
    },
    "surcharge_preloading": {
        "category": "Preloading",
        "brief": "Surcharge preloading analysis with optional wick drains.",
        "description": (
            "Computes time-settlement behavior under surcharge loading, "
            "optionally combined with wick drains. Determines time to "
            "achieve target degree of consolidation."
        ),
        "reference": "FHWA GEC-13; Terzaghi 1D Consolidation",
        "parameters": {
            "S_ultimate": {"type": "float", "required": True, "description": "Ultimate primary settlement (mm)."},
            "surcharge_kPa": {"type": "float", "required": True, "description": "Surcharge pressure (kPa)."},
            "cv": {"type": "float", "required": True, "description": "Vertical cv (m2/year)."},
            "Hdr": {"type": "float", "required": True, "description": "Maximum drainage path (m)."},
            "target_U": {"type": "float", "required": False, "default": 90.0, "description": "Target consolidation (%)."},
            "ch": {"type": "float", "required": False, "description": "Horizontal cv (m2/year). Needed if drains used."},
            "drain_spacing": {"type": "float", "required": False, "description": "Wick drain spacing (m). Omit for no drains."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Drain diameter (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "'triangular' or 'square'."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio."},
            "sigma_v0": {"type": "float", "required": False, "default": 0.0, "description": "Initial effective stress (kPa)."},
            "n_time_points": {"type": "int", "required": False, "default": 50, "description": "Points in time-settlement curve."},
        },
        "returns": {
            "settlement_at_target_mm": "Settlement when target U reached (mm).",
            "settlement_ultimate_mm": "Ultimate settlement (mm).",
            "time_to_target_years": "Time to reach target U (years).",
            "U_total_percent": "Degree of consolidation at target time (%).",
        },
    },
    "vibro_compaction": {
        "category": "Densification",
        "brief": "Vibro-compaction feasibility assessment.",
        "description": (
            "Evaluates feasibility of vibro-compaction based on fines content "
            "and grain size. Recommends probe spacing to achieve target SPT N-value."
        ),
        "reference": "FHWA GEC-13; Slocombe (1993)",
        "parameters": {
            "fines_content": {"type": "float", "required": True, "description": "Fines content (%)."},
            "initial_N_spt": {"type": "float", "required": True, "description": "Initial SPT blow count."},
            "target_N_spt": {"type": "float", "required": False, "default": 25.0, "description": "Target SPT blow count."},
            "D50": {"type": "float", "required": False, "description": "Mean grain size D50 (mm)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "'triangular' or 'square'."},
        },
        "returns": {
            "is_feasible": "Whether vibro-compaction is feasible (bool or str).",
            "recommended_spacing_m": "Recommended probe spacing (m).",
            "initial_N_spt": "Input SPT value.",
            "target_N_spt": "Target SPT value.",
        },
    },
    "feasibility": {
        "category": "Feasibility",
        "brief": "Ground improvement method selection and feasibility evaluation.",
        "description": (
            "Evaluates which ground improvement methods are applicable for "
            "given soil conditions, performance requirements, and constraints. "
            "Returns ranked methods with preliminary sizing recommendations."
        ),
        "reference": "FHWA GEC-13",
        "parameters": {
            "soil_type": {"type": "str", "required": True, "description": "Soil type: 'sand', 'silt', 'clay', 'gravel', 'organic', or 'peat'."},
            "fines_content": {"type": "float", "required": False, "description": "Fines content (%)."},
            "N_spt": {"type": "float", "required": False, "description": "SPT blow count."},
            "cu_kPa": {"type": "float", "required": False, "description": "Undrained shear strength (kPa)."},
            "thickness_m": {"type": "float", "required": False, "default": 0, "description": "Treatment zone thickness (m)."},
            "depth_to_top_m": {"type": "float", "required": False, "default": 0, "description": "Depth to top of treatment zone (m)."},
            "required_bearing_kPa": {"type": "float", "required": False, "default": 0, "description": "Required bearing capacity (kPa)."},
            "current_bearing_kPa": {"type": "float", "required": False, "default": 0, "description": "Current bearing capacity (kPa)."},
            "predicted_settlement_mm": {"type": "float", "required": False, "default": 0, "description": "Predicted settlement (mm)."},
            "allowable_settlement_mm": {"type": "float", "required": False, "default": 50, "description": "Allowable settlement (mm)."},
            "time_constraint_months": {"type": "float", "required": False, "default": 0, "description": "Available time (months)."},
            "cv_m2_per_year": {"type": "float", "required": False, "description": "Coefficient of consolidation (m2/year)."},
            "Hdr_m": {"type": "float", "required": False, "description": "Max drainage path (m)."},
            "gwt_depth_m": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
        },
        "returns": {
            "applicable_methods": "List of applicable ground improvement methods.",
            "not_applicable": "List of methods with reasons why not applicable.",
            "recommendations": "Ranked recommendations.",
            "preliminary_sizing": "Preliminary design parameters.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def ground_improvement_agent(method: str, parameters_json: str) -> str:
    """
    Ground improvement analysis agent.

    Provides aggregate pier, wick drain, surcharge preloading, vibro-compaction,
    and feasibility evaluation analyses per FHWA GEC-13.

    Call ground_improvement_list_methods() first to see available analyses,
    then ground_improvement_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "aggregate_piers").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with analysis results or an error message.
    """
    try:
        params = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def ground_improvement_list_methods(category: str = "") -> str:
    """
    Lists available ground improvement analysis methods.

    Parameters:
        category: Optional filter (e.g. "Drainage"). Leave empty for all.

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

    if not result:
        cats = sorted(set(i["category"] for i in METHOD_INFO.values()))
        return json.dumps({
            "error": f"No methods found for category '{category}'. "
                     f"Available: {', '.join(cats)}"
        })
    return json.dumps(result)


@function
def ground_improvement_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a ground improvement method.

    Parameters:
        method: The method name (e.g. "aggregate_piers").

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
