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
            "BRO-XML format. Extracts cone resistance (qc), sleeve friction (fs), "
            "pore pressure (u2), and friction ratio (Rf). All pressures are "
            "converted from the source file's MPa to kPa. Depths are in meters "
            "below ground surface. Only GEF and BRO-XML formats are supported "
            "(not CSV or other text formats). The output arrays can be passed "
            "directly to liquepy_agent for liquefaction triggering analysis."
        ),
        "reference": "GEF format spec (Dutch); BRO-XML schema (Netherlands PDOK)",
        "parameters": {
            "file_path": {"type": "str", "required": True,
                          "description": "Local filesystem path to CPT file (.gef or .xml). Absolute or relative path."},
            "engine": {"type": "str", "required": False, "default": "auto",
                       "choices": ["auto", "gef", "xml"],
                       "description": "Parser engine. 'auto' detects format from file header. Use 'gef' for .gef files, 'xml' for BRO-XML."},
            "index": {"type": "int", "required": False, "default": 0, "range": ">= 0",
                      "description": "Zero-based record index. Only needed for BRO-XML files that contain multiple CPT records. Use 0 for single-record files and all GEF files."},
        },
        "returns": {
            "n_points": "Number of CPT data points (readings).",
            "alias": "Test ID from file header (e.g. 'CPT-01'), or filename if not specified.",
            "final_depth_m": "Maximum penetration depth (m below ground surface).",
            "predrilled_depth_m": "Pre-excavated depth before CPT start (m). 0 if not pre-drilled.",
            "gwl_m": "Groundwater level (m below surface). null if not recorded in file.",
            "x": "X coordinate (easting) from file header. null if not recorded.",
            "y": "Y coordinate (northing) from file header. null if not recorded.",
            "srs_name": "Spatial reference system (e.g. 'urn:ogc:def:crs:EPSG::28992' for Dutch RD New).",
            "available_columns": "List of column names present in the source file (e.g. ['penetrationLength', 'coneResistance', ...]).",
            "depth_m": "Array of depths below ground surface (m). Always present.",
            "q_c_kPa": "Array of cone tip resistance values (kPa). Always present. Converted from MPa.",
            "f_s_kPa": "Array of sleeve friction values (kPa). Present if file has localFriction column. Converted from MPa.",
            "u_2_kPa": "Array of pore pressure u2 values (kPa). Present if file has porePressureU2 column. Converted from MPa.",
            "Rf_pct": "Array of friction ratio values (%). Present if file has frictionRatio column.",
        },
    },
    "parse_bore": {
        "category": "File Parsing",
        "brief": "Parse a borehole file (GEF or BRO-XML) into layer descriptions.",
        "description": (
            "Reads a borehole file in GEF or BRO-XML format. Extracts layer "
            "boundaries, geotechnical soil names (Dutch classification), and "
            "soil codes. Only GEF and BRO-XML formats are supported. GEF bore "
            "files use Dutch soil classification codes (e.g. 'Zs1' = slightly "
            "silty sand, 'Kz' = sandy clay) which are translated to full Dutch "
            "soil names by pygef."
        ),
        "reference": "GEF format spec (Dutch); BRO-XML schema; NEN 5104 classification",
        "parameters": {
            "file_path": {"type": "str", "required": True,
                          "description": "Local filesystem path to borehole file (.gef or .xml). Absolute or relative path."},
            "engine": {"type": "str", "required": False, "default": "auto",
                       "choices": ["auto", "gef", "xml"],
                       "description": "Parser engine. 'auto' detects format from file header."},
            "index": {"type": "int", "required": False, "default": 0, "range": ">= 0",
                      "description": "Zero-based record index for multi-record XML files."},
        },
        "returns": {
            "n_layers": "Number of soil layers in the borehole log.",
            "alias": "Borehole ID from file header (e.g. 'BH-01'), or filename if not specified.",
            "final_depth_m": "Total bore depth (m below ground surface).",
            "gwl_m": "Groundwater level (m below surface). null if not recorded.",
            "x": "X coordinate (easting). null if not recorded.",
            "y": "Y coordinate (northing). null if not recorded.",
            "srs_name": "Spatial reference system.",
            "layers": "List of layer objects, each with: top_m (upper boundary depth), bottom_m (lower boundary depth), soil_name (Dutch geotechnical name), soil_code (NEN 5104 code if available).",
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
