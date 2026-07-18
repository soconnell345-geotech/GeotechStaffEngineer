"""
liquepy Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. liquepy_agent           - Run a CPT-based liquefaction analysis
  2. liquepy_list_methods    - Browse available methods
  3. liquepy_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[liquepy] (PyPI)
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
            "CSR, CRR, Ic, fines content, post-triggering strains (Zhang et al. "
            "2002/2004), and liquefaction indices (LPI, LSN, LDI) from CPT data. "
            "All input arrays (depth, q_c, f_s, u_2) must have the same length."
        ),
        "reference": "Boulanger & Idriss (2014); Zhang et al. (2002, 2004); Iwasaki et al. (1982)",
        "parameters": {
            "depth": {"type": "list[float]", "required": True,
                      "description": "Depth from surface (m). Monotonically increasing. Same length as q_c and f_s."},
            "q_c": {"type": "list[float]", "required": True,
                    "description": "Cone tip resistance (kPa). Same length as depth."},
            "f_s": {"type": "list[float]", "required": True,
                    "description": "Sleeve friction (kPa). Same length as depth."},
            "u_2": {"type": "list[float]", "required": False,
                    "description": "Pore pressure behind cone tip (kPa). Same length as depth. If omitted, defaults to all zeros."},
            "gwl": {"type": "float", "required": False, "default": 1.0, "range": ">= 0",
                    "description": "Groundwater level depth (m below surface). 0 = at surface."},
            "pga": {"type": "float", "required": False, "default": 0.25, "range": "> 0",
                    "description": "Peak ground acceleration (g). Typical range 0.05-0.6."},
            "m_w": {"type": "float", "required": False, "default": 7.5, "range": "> 0",
                    "description": "Moment magnitude. Typical range 5.5-9.0."},
            "a_ratio": {"type": "float", "required": False, "default": 0.8, "range": "0 to 1",
                        "description": "Cone area ratio (net area ratio for pore pressure correction). Typical 0.6-0.85."},
            "i_c_limit": {"type": "float", "required": False, "default": 2.6, "range": "1.5 to 3.5",
                          "description": "Ic limit for liquefiable material. Soils with Ic > limit are considered non-liquefiable (claylike). Standard value 2.6."},
            "cfc": {"type": "float", "required": False, "default": 0.0,
                    "description": "Fines content correction factor (Eq 2.29 of B&I 2014). 0 for no correction."},
            "unit_wt_method": {"type": "str", "required": False, "default": "robertson2009",
                               "choices": ["robertson2009", "void_ratio"],
                               "description": "Unit weight estimation method. 'robertson2009' uses CPT-based correlation."},
            "gamma_predrill": {"type": "float", "required": False, "default": 17.0, "range": "14 to 22",
                               "description": "Assumed unit weight above pre-drill depth (kN/m3)."},
            "s_g": {"type": "float", "required": False, "default": 2.65, "range": "2.5 to 2.8",
                    "description": "Specific gravity of soil solids. 2.65 typical for quartz sand."},
            "p_a": {"type": "float", "required": False, "default": 101.0,
                    "description": "Atmospheric pressure (kPa). Standard: 101.325."},
        },
        "returns": {
            "n_points": "Number of CPT data points.",
            "gwl_m": "Groundwater level used in analysis (m below surface).",
            "pga_g": "Peak ground acceleration used (g).",
            "m_w": "Moment magnitude used.",
            "i_c_limit": "Ic limit used for liquefiable/non-liquefiable boundary.",
            "lpi": "Liquefaction Potential Index (Iwasaki). 0 = none, 0-5 = low, 5-15 = moderate, >15 = high.",
            "lsn": "Liquefaction Severity Number. 0-10 = minor, 10-30 = moderate, >30 = severe damage expected.",
            "ldi_m": "Lateral Displacement Index (m). Integral of post-liquefaction shear strain over depth.",
            "min_fos": "Minimum factor of safety in liquefiable zone (Ic < limit, below GWL). FoS < 1.0 = liquefaction expected.",
            "max_settlement_mm": "Maximum estimated 1D post-liquefaction settlement (mm). Based on volumetric strain integration.",
        },
    },
    "field_correlations": {
        "category": "Correlations",
        "brief": "CPT-based field correlations (Vs, Dr, su/σv', permeability).",
        "description": (
            "Computes shear wave velocity, relative density, undrained strength "
            "ratio, and permeability from CPT data using published correlations. "
            "All input arrays (depth, q_c, f_s, u_2) must have the same length. "
            "Internally runs a B&I 2014 analysis to obtain Ic, q_c1n, and σ'v "
            "needed by the correlations."
        ),
        "reference": "McGann et al. (2015); Robertson (2009); Andrus et al. (2007); Boulanger et al. (2014)",
        "parameters": {
            "depth": {"type": "list[float]", "required": True,
                      "description": "Depth from surface (m). Same length as q_c and f_s."},
            "q_c": {"type": "list[float]", "required": True,
                    "description": "Cone tip resistance (kPa). Same length as depth."},
            "f_s": {"type": "list[float]", "required": True,
                    "description": "Sleeve friction (kPa). Same length as depth."},
            "u_2": {"type": "list[float]", "required": False,
                    "description": "Pore pressure behind cone tip (kPa). Same length as depth. If omitted, defaults to all zeros."},
            "gwl": {"type": "float", "required": False, "default": 1.0, "range": ">= 0",
                    "description": "Groundwater level depth (m below surface)."},
            "a_ratio": {"type": "float", "required": False, "default": 0.8, "range": "0 to 1",
                        "description": "Cone area ratio."},
            "vs_method": {"type": "str", "required": False, "default": "mcgann2015",
                          "choices": ["mcgann2015", "robertson2009", "andrus2007"],
                          "description": "Shear wave velocity correlation method. 'mcgann2015' = McGann et al. (2015), 'robertson2009' = Robertson (2009), 'andrus2007' = Andrus et al. (2007)."},
            "i_c_limit": {"type": "float", "required": False, "default": 2.6,
                          "description": "Ic limit for soil classification. Standard: 2.6."},
            "p_a": {"type": "float", "required": False, "default": 101.0,
                    "description": "Atmospheric pressure (kPa)."},
            "s_g": {"type": "float", "required": False, "default": 2.65,
                    "description": "Specific gravity of soil solids."},
            "gamma_predrill": {"type": "float", "required": False, "default": 17.0,
                               "description": "Assumed unit weight above pre-drill depth (kN/m3)."},
        },
        "returns": {
            "n_points": "Number of CPT data points.",
            "gwl_m": "Groundwater level used (m below surface).",
            "vs_method": "Vs correlation method used.",
            "vs_min_m_per_s": "Minimum shear wave velocity in profile (m/s).",
            "vs_max_m_per_s": "Maximum shear wave velocity in profile (m/s).",
            "vs_avg_m_per_s": "Average shear wave velocity in profile (m/s). Useful for Vs30-based site classification.",
            "dr_min": "Minimum relative density in profile (decimal, 0-1).",
            "dr_max": "Maximum relative density in profile (decimal, 0-1).",
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
