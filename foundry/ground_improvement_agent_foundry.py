"""
Ground Improvement Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. ground_improvement_agent           - Run a ground improvement analysis
  2. ground_improvement_list_methods    - Browse available methods
  3. ground_improvement_describe_method - Get detailed parameter docs

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
            "bearing capacity for aggregate pier reinforcement. Spacing must "
            "be greater than column diameter."
        ),
        "reference": "FHWA GEC-13: Ground Modification Methods",
        "parameters": {
            "column_diameter": {"type": "float", "required": True, "range": "> 0",
                                "description": "Column diameter (m). Typical 0.3-1.0m."},
            "spacing": {"type": "float", "required": True, "range": "> column_diameter",
                        "description": "Center-to-center spacing (m). Must exceed column diameter. Typical 1.5-3.0m."},
            "pattern": {"type": "str", "required": False, "default": "triangular",
                        "choices": ["triangular", "square"],
                        "description": "Column layout pattern. Triangular gives higher area replacement for same spacing."},
            "E_column": {"type": "float", "required": False, "default": 80000, "range": "> 0",
                         "description": "Column modulus (kPa). Typical 50000-150000 for aggregate piers."},
            "E_soil": {"type": "float", "required": False, "default": 5000, "range": "> 0",
                       "description": "Surrounding soil modulus (kPa). Typical 2000-20000."},
            "n": {"type": "float", "required": False, "default": 5.0, "range": "> 1",
                  "description": "Stress concentration ratio (stress in column / stress in soil). Typical 3-8."},
            "q_unreinforced": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                               "description": "Unreinforced bearing capacity (kPa). If 0, improved bearing is not computed."},
            "S_unreinforced": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                               "description": "Unreinforced settlement (mm). If 0, improved settlement is not computed."},
        },
        "returns": {
            "area_replacement_ratio": "Ratio of column area to tributary area (0-1). Higher = more reinforcement.",
            "stress_concentration_ratio": "Stress in column / stress in soil.",
            "composite_modulus_kPa": "Weighted average modulus of the reinforced zone (kPa).",
            "settlement_reduction_factor": "Ratio of improved to unreinforced settlement (0-1). Lower = more improvement.",
            "improved_bearing_kPa": "Improved bearing capacity (kPa). 0 if q_unreinforced not provided.",
            "settlement_improved_mm": "Improved settlement (mm). 0 if S_unreinforced not provided.",
        },
    },
    "wick_drains": {
        "category": "Drainage",
        "brief": "Wick drain consolidation analysis (Barron/Hansbo).",
        "description": (
            "Computes radial and vertical consolidation with prefabricated "
            "vertical drains (PVDs/wick drains) using the Barron/Hansbo "
            "solution. Includes smear zone and well resistance effects. "
            "Returns degree of consolidation at the specified time."
        ),
        "reference": "FHWA GEC-13; Hansbo (1979); Barron (1948)",
        "parameters": {
            "spacing": {"type": "float", "required": True, "range": "> 0",
                        "description": "Center-to-center drain spacing (m). Typical 1.0-3.0m."},
            "ch": {"type": "float", "required": True, "range": "> 0",
                   "description": "Horizontal coefficient of consolidation (m2/year). Typically 1-10x cv."},
            "cv": {"type": "float", "required": True, "range": "> 0",
                   "description": "Vertical coefficient of consolidation (m2/year). Lab or field test value."},
            "Hdr": {"type": "float", "required": True, "range": "> 0",
                    "description": "Maximum vertical drainage path length (m). Half the layer thickness for two-way drainage, full thickness for one-way."},
            "time": {"type": "float", "required": True, "range": "> 0",
                     "description": "Time for consolidation analysis (years)."},
            "dw": {"type": "float", "required": False, "default": 0.066, "range": "> 0",
                   "description": "Equivalent drain diameter (m). 0.066m is standard for 100x4mm wick drains."},
            "pattern": {"type": "str", "required": False, "default": "triangular",
                        "choices": ["triangular", "square"],
                        "description": "Drain layout pattern."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "range": "> 1",
                            "description": "Smear zone diameter / drain diameter. Typical 2-5. Higher = more disturbance."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "range": "> 1",
                            "description": "Undisturbed / smear zone permeability ratio. Typical 2-5."},
            "n_time_points": {"type": "int", "required": False, "default": 50, "range": "> 0",
                              "description": "Number of points in time-settlement output curve."},
        },
        "returns": {
            "Ur_percent": "Radial consolidation due to drains (%).",
            "Uv_percent": "Vertical consolidation (%).",
            "U_total_percent": "Combined total consolidation (%). = 1 - (1-Ur)(1-Uv).",
            "spacing_ratio_n": "n = de/dw (equivalent diameter of drain zone / drain diameter).",
            "F_n": "Drain spacing function F(n) used in Barron/Hansbo equation.",
        },
        "related": {
            "settlement_agent.consolidation_settlement": "Compute how much settlement to accelerate.",
            "design_drain_spacing": "Optimize drain spacing for target consolidation.",
            "surcharge_preloading": "Combine with surcharge for faster consolidation.",
        },
        "typical_workflow": (
            "1. Compute consolidation settlement (settlement_agent.consolidation_settlement)\n"
            "2. If settlement too large or too slow, design wick drains (this method)\n"
            "3. Optimize spacing (design_drain_spacing)\n"
            "4. Consider surcharge preloading to reduce post-construction settlement"
        ),
        "common_mistakes": [
            "Using cv (vertical) instead of ch (horizontal) for drain consolidation — ch is typically 2-5x cv.",
            "Hdr is the drainage path, not the full clay thickness — for double drainage, Hdr = thickness/2.",
        ],
    },
    "design_drain_spacing": {
        "category": "Drainage",
        "brief": "Inverse design: find wick drain spacing for target consolidation and time.",
        "description": (
            "Uses bisection to find the drain spacing that achieves a target "
            "degree of consolidation within a specified time frame. Useful for "
            "design optimization — provide the required U% and available time, "
            "and this method finds the required spacing."
        ),
        "reference": "FHWA GEC-13",
        "parameters": {
            "target_U": {"type": "float", "required": True, "range": "50 to 99",
                         "description": "Target degree of consolidation (%). Typical 80-95%."},
            "target_time": {"type": "float", "required": True, "range": "> 0",
                            "description": "Available time for consolidation (years)."},
            "ch": {"type": "float", "required": True, "range": "> 0",
                   "description": "Horizontal coefficient of consolidation (m2/year)."},
            "cv": {"type": "float", "required": True, "range": "> 0",
                   "description": "Vertical coefficient of consolidation (m2/year)."},
            "Hdr": {"type": "float", "required": True, "range": "> 0",
                    "description": "Maximum vertical drainage path (m)."},
            "dw": {"type": "float", "required": False, "default": 0.066,
                   "description": "Equivalent drain diameter (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular",
                        "choices": ["triangular", "square"],
                        "description": "Drain layout pattern."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0,
                            "description": "Smear zone diameter / drain diameter."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0,
                            "description": "Undisturbed / smear permeability ratio."},
            "spacing_range": {"type": "list[float]", "required": False, "default": [1.0, 3.5],
                              "description": "Bisection search range [min_spacing, max_spacing] in meters. Widen if solution not found."},
        },
        "returns": {
            "spacing": "Required drain spacing to achieve target_U within target_time (m).",
            "U_total_percent": "Actual consolidation achieved at the computed spacing (%).",
        },
    },
    "surcharge_preloading": {
        "category": "Preloading",
        "brief": "Surcharge preloading analysis with optional wick drains.",
        "description": (
            "Computes time-settlement behavior under surcharge loading, "
            "optionally combined with wick drains. Determines time to "
            "achieve target degree of consolidation. To include wick drains, "
            "provide both ch and drain_spacing parameters."
        ),
        "reference": "FHWA GEC-13; Terzaghi 1D Consolidation",
        "parameters": {
            "S_ultimate": {"type": "float", "required": True, "range": "> 0",
                           "description": "Ultimate primary consolidation settlement (mm)."},
            "surcharge_kPa": {"type": "float", "required": True, "range": "> 0",
                              "description": "Applied surcharge pressure (kPa)."},
            "cv": {"type": "float", "required": True, "range": "> 0",
                   "description": "Vertical coefficient of consolidation (m2/year)."},
            "Hdr": {"type": "float", "required": True, "range": "> 0",
                    "description": "Maximum vertical drainage path (m)."},
            "target_U": {"type": "float", "required": False, "default": 90.0, "range": "50 to 99",
                         "description": "Target degree of consolidation (%)."},
            "ch": {"type": "float", "required": False, "range": "> 0",
                   "description": "Horizontal cv (m2/year). Required if drain_spacing is provided."},
            "drain_spacing": {"type": "float", "required": False, "range": "> 0",
                              "description": "Wick drain spacing (m). Omit for surcharge-only (no drains). Requires ch."},
            "dw": {"type": "float", "required": False, "default": 0.066,
                   "description": "Equivalent drain diameter (m)."},
            "pattern": {"type": "str", "required": False, "default": "triangular",
                        "choices": ["triangular", "square"],
                        "description": "Drain layout pattern."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0,
                            "description": "Smear zone diameter / drain diameter."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0,
                            "description": "Undisturbed / smear permeability ratio."},
            "sigma_v0": {"type": "float", "required": False, "default": 0.0,
                         "description": "Initial effective vertical stress at the center of the compressible layer (kPa). 0 if unknown."},
            "n_time_points": {"type": "int", "required": False, "default": 50,
                              "description": "Number of points in the output time-settlement curve."},
        },
        "returns": {
            "settlement_at_target_mm": "Settlement when target U is reached (mm).",
            "settlement_ultimate_mm": "Ultimate primary settlement (mm). Same as S_ultimate input.",
            "time_to_target_years": "Time to reach target U (years). Key design output.",
            "U_total_percent": "Degree of consolidation achieved (%).",
        },
    },
    "vibro_compaction": {
        "category": "Densification",
        "brief": "Vibro-compaction feasibility assessment.",
        "description": (
            "Evaluates feasibility of vibro-compaction based on fines content "
            "and grain size. Recommends probe spacing to achieve target SPT "
            "N-value. Vibro-compaction is generally feasible only for granular "
            "soils with fines content < 15-20%."
        ),
        "reference": "FHWA GEC-13; Slocombe (1993)",
        "parameters": {
            "fines_content": {"type": "float", "required": True, "range": "0 to 100",
                              "description": "Fines content (% passing #200 sieve). >20% generally not feasible."},
            "initial_N_spt": {"type": "float", "required": True, "range": "> 0",
                              "description": "Initial (pre-treatment) SPT blow count (N-value)."},
            "target_N_spt": {"type": "float", "required": False, "default": 25.0, "range": "> initial_N_spt",
                             "description": "Target (post-treatment) SPT blow count. Typical 20-30."},
            "D50": {"type": "float", "required": False, "range": "> 0",
                    "description": "Mean grain size D50 (mm). Improves feasibility assessment if provided."},
            "pattern": {"type": "str", "required": False, "default": "triangular",
                        "choices": ["triangular", "square"],
                        "description": "Vibro-probe layout pattern."},
        },
        "returns": {
            "is_feasible": "Feasibility assessment: true, false, or 'marginal'. Based on fines content and D50.",
            "recommended_spacing_m": "Recommended probe spacing (m). Based on empirical Slocombe (1993) chart.",
            "initial_N_spt": "Input initial SPT blow count.",
            "target_N_spt": "Target SPT blow count.",
        },
    },
    "feasibility": {
        "category": "Feasibility",
        "brief": "Ground improvement method selection and feasibility evaluation.",
        "description": (
            "Evaluates which ground improvement methods are applicable for "
            "given soil conditions, performance requirements, and project "
            "constraints. Returns ranked recommendations with preliminary "
            "sizing. Provide as many optional parameters as available — the "
            "more information given, the better the recommendations. At minimum, "
            "soil_type is required."
        ),
        "reference": "FHWA GEC-13",
        "parameters": {
            "soil_type": {"type": "str", "required": True,
                          "choices": ["soft_clay", "loose_sand", "mixed", "organic"],
                          "description": "Primary soil classification in the treatment zone. 'soft_clay' for cohesive soils, 'loose_sand' for granular soils, 'mixed' for interbedded or transitional, 'organic' for organic/peat soils."},
            "fines_content": {"type": "float", "required": False, "range": "0 to 100",
                              "description": "Fines content (%). Affects method applicability (e.g., vibro-compaction needs <20%)."},
            "N_spt": {"type": "float", "required": False, "range": "> 0",
                      "description": "Representative SPT blow count in treatment zone. Indicates soil density/strength."},
            "cu_kPa": {"type": "float", "required": False, "range": "> 0",
                       "description": "Undrained shear strength (kPa). For cohesive soils."},
            "thickness_m": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                            "description": "Treatment zone thickness (m). 0 if unknown. Affects column length estimates."},
            "depth_to_top_m": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                               "description": "Depth to top of treatment zone (m below ground surface)."},
            "required_bearing_kPa": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                                     "description": "Required bearing capacity (kPa). 0 if not a design criterion."},
            "current_bearing_kPa": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                                    "description": "Current (untreated) bearing capacity (kPa). 0 if unknown."},
            "predicted_settlement_mm": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                                        "description": "Predicted settlement without treatment (mm). 0 if unknown."},
            "allowable_settlement_mm": {"type": "float", "required": False, "default": 50, "range": "> 0",
                                        "description": "Allowable settlement criterion (mm). Default 50mm (typical for shallow foundations)."},
            "time_constraint_months": {"type": "float", "required": False, "default": 0, "range": ">= 0",
                                       "description": "Available construction time (months). 0 if no time constraint. Affects preloading/drain feasibility."},
            "cv_m2_per_year": {"type": "float", "required": False, "range": "> 0",
                               "description": "Coefficient of consolidation (m2/year). Needed for drainage time estimates."},
            "Hdr_m": {"type": "float", "required": False, "range": "> 0",
                      "description": "Maximum drainage path (m). Needed for drainage time estimates."},
            "gwt_depth_m": {"type": "float", "required": False, "range": ">= 0",
                            "description": "Groundwater table depth (m below surface). Affects vibro-compaction and dewatering needs."},
        },
        "returns": {
            "applicable_methods": "List of strings naming feasible methods (e.g., ['Aggregate Piers', 'Wick Drains (PVD)']).",
            "not_applicable": "List of dicts, each with 'method' (str) and 'reason' (str) explaining why that method was excluded.",
            "recommendations": "List of prioritized recommendation strings with preliminary guidance. If empty, no standard methods are feasible.",
            "preliminary_sizing": "Dict keyed by method name (e.g., 'aggregate_piers', 'wick_drains') with sub-dicts of typical spacing, diameters, and SRF estimates.",
            "soil_description": "Human-readable summary of the input soil conditions.",
            "design_problem": "Human-readable summary of the design problem (settlement, bearing, time constraint).",
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
