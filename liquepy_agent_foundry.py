"""
liquepy Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. liquepy_agent           - Run a CPT-based liquefaction analysis
  2. liquepy_list_methods    - Browse available methods
  3. liquepy_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - Requires liquepy (pip install liquepy) for analysis execution
  - Metadata functions work without liquepy installed
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

from liquepy_agent.liquepy_utils import has_liquepy


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

def _run_cpt_liquefaction(params):
    from liquepy_agent import analyze_cpt_liquefaction
    # Convert list inputs to arrays
    for key in ("depth", "q_c", "f_s", "u_2"):
        if key in params and isinstance(params[key], list):
            params[key] = np.asarray(params[key], dtype=float)
    result = analyze_cpt_liquefaction(**params)
    return _clean_result(result.to_dict())


def _run_field_correlations(params):
    from liquepy_agent import analyze_field_correlations
    for key in ("depth", "q_c", "f_s", "u_2"):
        if key in params and isinstance(params[key], list):
            params[key] = np.asarray(params[key], dtype=float)
    result = analyze_field_correlations(**params)
    return _clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "cpt_liquefaction": _run_cpt_liquefaction,
    "field_correlations": _run_field_correlations,
}

METHOD_INFO = {
    "cpt_liquefaction": {
        "category": "Triggering",
        "brief": "CPT-based liquefaction triggering analysis (Boulanger & Idriss 2014).",
        "description": (
            "Performs CPT-based simplified liquefaction triggering using the "
            "Boulanger & Idriss (2014) procedure. Computes factor of safety, "
            "CSR, CRR, Ic, fines content, post-triggering strains, LPI, LSN, "
            "and LDI from cone penetration test data."
        ),
        "reference": "Boulanger & Idriss (2014); Zhang et al. (2002, 2004)",
        "parameters": {
            "depth": {"type": "list[float]", "required": True, "description": "Depth from surface (m). Monotonically increasing."},
            "q_c": {"type": "list[float]", "required": True, "description": "Cone tip resistance (kPa)."},
            "f_s": {"type": "list[float]", "required": True, "description": "Sleeve friction (kPa)."},
            "u_2": {"type": "list[float]", "required": False, "default": "zeros", "description": "Pore pressure behind cone tip (kPa)."},
            "gwl": {"type": "float", "required": False, "default": 1.0, "description": "Groundwater level depth (m below surface)."},
            "pga": {"type": "float", "required": False, "default": 0.25, "description": "Peak ground acceleration (g)."},
            "m_w": {"type": "float", "required": False, "default": 7.5, "description": "Moment magnitude."},
            "a_ratio": {"type": "float", "required": False, "default": 0.8, "description": "Cone area ratio."},
            "i_c_limit": {"type": "float", "required": False, "default": 2.6, "description": "Ic limit for liquefiable material."},
            "cfc": {"type": "float", "required": False, "default": 0.0, "description": "Fines content correction factor."},
            "unit_wt_method": {"type": "str", "required": False, "default": "robertson2009", "description": "'robertson2009' or 'void_ratio'."},
            "gamma_predrill": {"type": "float", "required": False, "default": 17.0, "description": "Pre-drill unit weight (kN/m3)."},
            "s_g": {"type": "float", "required": False, "default": 2.65, "description": "Specific gravity of solids."},
            "p_a": {"type": "float", "required": False, "default": 101.0, "description": "Atmospheric pressure (kPa)."},
        },
        "returns": {
            "n_points": "Number of CPT data points.",
            "gwl_m": "Groundwater level (m).",
            "pga_g": "Peak ground acceleration (g).",
            "m_w": "Moment magnitude.",
            "lpi": "Liquefaction Potential Index (>15 = high).",
            "lsn": "Liquefaction Severity Number (>30 = severe).",
            "ldi_m": "Lateral Displacement Index (m).",
            "min_fos": "Minimum factor of safety in liquefiable zone.",
            "max_settlement_mm": "Maximum 1D settlement (mm).",
        },
    },
    "field_correlations": {
        "category": "Correlations",
        "brief": "CPT-based field correlations (Vs, Dr, su/σv', permeability).",
        "description": (
            "Computes shear wave velocity, relative density, undrained strength "
            "ratio, and permeability from CPT data using published correlations. "
            "Multiple Vs methods available: McGann 2015, Robertson 2009, Andrus 2007."
        ),
        "reference": "McGann et al. (2015); Robertson (2009); Andrus et al. (2007); Boulanger et al. (2014)",
        "parameters": {
            "depth": {"type": "list[float]", "required": True, "description": "Depth from surface (m)."},
            "q_c": {"type": "list[float]", "required": True, "description": "Cone tip resistance (kPa)."},
            "f_s": {"type": "list[float]", "required": True, "description": "Sleeve friction (kPa)."},
            "u_2": {"type": "list[float]", "required": False, "default": "zeros", "description": "Pore pressure behind cone tip (kPa)."},
            "gwl": {"type": "float", "required": False, "default": 1.0, "description": "Groundwater level depth (m below surface)."},
            "a_ratio": {"type": "float", "required": False, "default": 0.8, "description": "Cone area ratio."},
            "vs_method": {"type": "str", "required": False, "default": "mcgann2015", "description": "'mcgann2015', 'robertson2009', or 'andrus2007'."},
            "i_c_limit": {"type": "float", "required": False, "default": 2.6, "description": "Ic limit for classification."},
            "p_a": {"type": "float", "required": False, "default": 101.0, "description": "Atmospheric pressure (kPa)."},
            "s_g": {"type": "float", "required": False, "default": 2.65, "description": "Specific gravity of solids."},
            "gamma_predrill": {"type": "float", "required": False, "default": 17.0, "description": "Pre-drill unit weight (kN/m3)."},
        },
        "returns": {
            "n_points": "Number of CPT data points.",
            "gwl_m": "Groundwater level (m).",
            "vs_method": "Vs correlation method used.",
            "vs_min_m_per_s": "Minimum Vs (m/s).",
            "vs_max_m_per_s": "Maximum Vs (m/s).",
            "vs_avg_m_per_s": "Average Vs (m/s).",
            "dr_min": "Minimum relative density.",
            "dr_max": "Maximum relative density.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def liquepy_agent(method: str, parameters_json: str) -> str:
    """
    CPT-based liquefaction analysis agent.

    Provides Boulanger & Idriss (2014) CPT-based liquefaction triggering,
    post-triggering strains and indices (LPI, LSN, LDI), and field
    correlations (Vs, Dr, su/σv', permeability).

    Call liquepy_list_methods() first to see available analyses,
    then liquepy_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "cpt_liquefaction").
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

    if not has_liquepy():
        return json.dumps({
            "error": "liquepy is not installed. Install with: pip install liquepy"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def liquepy_list_methods(category: str = "") -> str:
    """
    Lists available liquepy analysis methods.

    Parameters:
        category: Optional filter (e.g. "Triggering"). Leave empty for all.

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
def liquepy_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a liquepy analysis method.

    Parameters:
        method: The method name (e.g. "cpt_liquefaction").

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
