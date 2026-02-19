"""
pygef Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. pygef_agent           - Parse a CPT or borehole file
  2. pygef_list_methods    - Browse available methods
  3. pygef_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - Requires pygef (pip install pygef) for file parsing
  - Metadata functions work without pygef installed
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

from pygef_agent.pygef_utils import has_pygef


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
        cleaned = {}
        for k, v in result.items():
            if isinstance(v, dict):
                cleaned[k] = _clean_result(v)
            elif isinstance(v, list):
                cleaned[k] = [_clean_result(item) if isinstance(item, dict)
                              else _clean_value(item) for item in v]
            else:
                cleaned[k] = _clean_value(v)
        return cleaned
    return result


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_parse_cpt(params):
    from pygef_agent import parse_cpt_file
    result = parse_cpt_file(**params)
    return _clean_result(result.to_dict())


def _run_parse_bore(params):
    from pygef_agent import parse_bore_file
    result = parse_bore_file(**params)
    return _clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "parse_cpt": _run_parse_cpt,
    "parse_bore": _run_parse_bore,
}

METHOD_INFO = {
    "parse_cpt": {
        "category": "File Parsing",
        "brief": "Parse a CPT file (GEF or BRO-XML) into standardized arrays.",
        "description": (
            "Reads a CPT file in GEF (Dutch Geotechnical Exchange Format) or "
            "BRO-XML format. Extracts cone resistance, sleeve friction, pore "
            "pressure, and friction ratio. Converts all pressures from MPa to kPa."
        ),
        "reference": "GEF format spec; BRO-XML schema",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to CPT file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "description": "'auto', 'gef', or 'xml'."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_points": "Number of data points.",
            "alias": "Test ID or filename.",
            "final_depth_m": "Final penetration depth (m).",
            "gwl_m": "Groundwater level (m below surface).",
            "depth_m": "Depth array (m).",
            "q_c_kPa": "Cone tip resistance array (kPa).",
            "f_s_kPa": "Sleeve friction array (kPa).",
            "u_2_kPa": "Pore pressure u2 array (kPa).",
            "Rf_pct": "Friction ratio array (%).",
        },
    },
    "parse_bore": {
        "category": "File Parsing",
        "brief": "Parse a borehole file (GEF or BRO-XML) into layer descriptions.",
        "description": (
            "Reads a borehole file in GEF or BRO-XML format. Extracts layer "
            "boundaries, geotechnical soil names, and soil codes."
        ),
        "reference": "GEF format spec; BRO-XML schema",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to borehole file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "description": "'auto', 'gef', or 'xml'."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_layers": "Number of soil layers.",
            "alias": "Borehole ID or filename.",
            "final_depth_m": "Total bore depth (m).",
            "gwl_m": "Groundwater level (m below surface).",
            "layers": "List of layers with top_m, bottom_m, soil_name, soil_code.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def pygef_agent(method: str, parameters_json: str) -> str:
    """
    CPT and borehole file parsing agent.

    Parses GEF and BRO-XML files for cone penetration test and borehole
    data. Converts all pressures to kPa and depths to meters.

    Call pygef_list_methods() first to see available methods,
    then pygef_describe_method() for parameter details.

    Parameters:
        method: Method name (e.g. "parse_cpt").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with parsed data or an error message.
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

    if not has_pygef():
        return json.dumps({
            "error": "pygef is not installed. Install with: pip install pygef"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def pygef_list_methods(category: str = "") -> str:
    """
    Lists available pygef parsing methods.

    Parameters:
        category: Optional filter (e.g. "File Parsing"). Leave empty for all.

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
def pygef_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a pygef parsing method.

    Parameters:
        method: The method name (e.g. "parse_cpt").

    Returns:
        JSON string with parameters, types, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
