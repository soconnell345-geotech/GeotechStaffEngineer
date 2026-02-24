"""
AGS4 Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. ags4_agent           - Read or validate an AGS4 file
  2. ags4_list_methods    - Browse available methods
  3. ags4_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[ags4] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from ags4_agent.ags4_utils import has_ags4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_result(result):
    """Ensure all values are JSON-serializable."""
    if isinstance(result, dict):
        return {k: _clean_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_clean_result(v) for v in result]
    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_read_ags4(params):
    from ags4_agent import read_ags4
    result = read_ags4(**params)
    return _clean_result(result.to_dict())


def _run_validate_ags4(params):
    from ags4_agent import validate_ags4
    result = validate_ags4(**params)
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "read_ags4": _run_read_ags4,
    "validate_ags4": _run_validate_ags4,
}

METHOD_INFO = {
    "read_ags4": {
        "category": "Data Import",
        "brief": "Read and parse an AGS4 file.",
        "description": (
            "Reads an AGS4 geotechnical data exchange file (or string) and "
            "returns structured data. AGS4 is the UK standard for exchanging "
            "borehole, lab test, and field test data. Returns group (table) "
            "names, row counts, and optionally the full table data as JSON."
        ),
        "reference": "AGS4 Data Format v4 (www.ags.org.uk)",
        "parameters": {
            "filepath": {"type": "str", "required": False, "default": "null",
                         "description": "Path to an AGS4 file. Provide either filepath or content, not both."},
            "content": {"type": "str", "required": False, "default": "null",
                        "description": "Raw AGS4 data as a string. Use this when AGS4 data is embedded in an API response."},
            "encoding": {"type": "str", "required": False, "default": "utf-8",
                         "description": "File encoding. Try 'cp1252' for older Windows-generated files."},
            "include_data": {"type": "bool", "required": False, "default": True,
                             "description": "If true, include full table data in the result. Set false to only get group names and row counts."},
            "convert_numeric": {"type": "bool", "required": False, "default": True,
                                "description": "If true, convert numeric columns (DP, SF, SCI, MC types) from text to numbers."},
        },
        "returns": {
            "filepath": "Source file path or '<string>' for string input.",
            "n_groups": "Number of AGS4 groups (tables) found.",
            "group_names": "List of group names (e.g. PROJ, HOLE, ISPT, GEOL).",
            "group_row_counts": "Dict mapping group name to data row count.",
            "tables": "Dict of group_name -> list of row dicts (only if include_data=true).",
        },
    },
    "validate_ags4": {
        "category": "Validation",
        "brief": "Validate an AGS4 file against AGS4 rules.",
        "description": (
            "Checks an AGS4 file (or string) against AGS4 format rules and "
            "returns a validation report. Identifies errors (format violations), "
            "warnings, and FYI messages. A file is valid if it has zero errors."
        ),
        "reference": "AGS4 Data Format v4 (www.ags.org.uk)",
        "parameters": {
            "filepath": {"type": "str", "required": False, "default": "null",
                         "description": "Path to an AGS4 file. Provide either filepath or content, not both."},
            "content": {"type": "str", "required": False, "default": "null",
                        "description": "Raw AGS4 data as a string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8",
                         "description": "File encoding."},
        },
        "returns": {
            "filepath": "Source file path.",
            "n_errors": "Number of errors (format violations).",
            "n_warnings": "Number of warnings.",
            "n_fyi": "Number of FYI messages.",
            "is_valid": "True if no errors found. Warnings/FYI are acceptable.",
            "errors": "Dict of rule_number -> list of error messages.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def ags4_agent(method: str, parameters_json: str) -> str:
    """
    AGS4 geotechnical data format agent.

    Reads and validates AGS4 format files — the UK standard for
    exchanging borehole logs, lab tests, and field test data.

    Call ags4_list_methods() first to see available operations,
    then ags4_describe_method() for parameter details.

    Parameters:
        method: Method name (e.g. "read_ags4", "validate_ags4").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with results or an error message.
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

    if not has_ags4():
        return json.dumps({
            "error": "python-ags4 is not installed. Install with: pip install python-ags4"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def ags4_list_methods(category: str) -> str:
    """
    List available AGS4 operations.

    Parameters:
        category: Filter by category (e.g. "Data Import", "Validation") or "" for all.

    Returns:
        JSON string mapping categories to method lists.
    """
    grouped = {}
    for name, info in METHOD_INFO.items():
        cat = info["category"]
        grouped.setdefault(cat, []).append(name)

    if not category or category.strip() == "":
        return json.dumps(grouped)

    if category in grouped:
        return json.dumps({category: grouped[category]})

    return json.dumps({"error": f"Unknown category '{category}'. Available: {sorted(grouped.keys())}"})


@function
def ags4_describe_method(method: str) -> str:
    """
    Describe an AGS4 method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "read_ags4", "validate_ags4").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
