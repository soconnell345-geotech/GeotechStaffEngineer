"""
Foundry agent wrapper for pydiggs_agent module.

Provides LLM-friendly JSON interface to DIGGS XML validation functions.

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[pydiggs] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
from typing import Optional

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from pydiggs_agent import (
    has_pydiggs,
    validate_diggs_schema,
    validate_diggs_dictionary,
)


# Method registry
METHODS = {
    "validate_schema": {
        "name": "validate_schema",
        "category": "Validation",
        "description": "Validate DIGGS XML file or content against XSD schema",
        "parameters": {
            "filepath": {
                "type": "string",
                "description": "Path to DIGGS XML file (mutually exclusive with content)",
                "required": False
            },
            "content": {
                "type": "string",
                "description": "DIGGS XML as string (mutually exclusive with filepath)",
                "required": False
            },
            "schema_version": {
                "type": "string",
                "description": "DIGGS schema version ('2.6' or '2.5.a')",
                "default": "2.6",
                "required": False
            }
        },
        "returns": {
            "type": "DiggValidationResult",
            "description": "Validation result with is_valid, n_errors, errors list"
        },
        "example": {
            "content": "<?xml version='1.0'?><Diggs>...</Diggs>",
            "schema_version": "2.6"
        }
    },
    "validate_dictionary": {
        "name": "validate_dictionary",
        "category": "Validation",
        "description": "Validate DIGGS propertyClass values against DIGGS dictionary",
        "parameters": {
            "filepath": {
                "type": "string",
                "description": "Path to DIGGS XML file (mutually exclusive with content)",
                "required": False
            },
            "content": {
                "type": "string",
                "description": "DIGGS XML as string (mutually exclusive with filepath)",
                "required": False
            }
        },
        "returns": {
            "type": "DiggValidationResult",
            "description": "Validation result with undefined properties in errors list"
        },
        "example": {
            "filepath": "borehole_data.xml"
        }
    }
}


@function
def pydiggs_list_methods(category: Optional[str] = None) -> str:
    """
    List available pydiggs agent methods.

    Args:
        category: Filter by category (e.g., "Validation"), or None for all

    Returns:
        JSON string with list of method metadata dictionaries
    """
    methods = list(METHODS.values())

    if category is not None and category != "":
        methods = [m for m in methods if m["category"] == category]

    return json.dumps(methods)


@function
def pydiggs_describe_method(method: str) -> str:
    """
    Get detailed documentation for a specific method.

    Args:
        method: Method name (e.g., "validate_schema")

    Returns:
        JSON string with method metadata, or error if method not found
    """
    if method not in METHODS:
        return json.dumps({"error": f"Unknown method: {method}"})

    return json.dumps(METHODS[method])


@function
def pydiggs_agent(method: str, params_json: str) -> str:
    """
    Execute a pydiggs agent method.

    Args:
        method: Method name (e.g., "validate_schema")
        params_json: JSON string with method parameters

    Returns:
        JSON string with result or error

    Raises:
        json.JSONDecodeError: If params_json is invalid JSON
    """
    # Validate JSON first (before checking has_pydiggs)
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise  # Re-raise to caller

    # Check method exists
    if method not in METHODS:
        return json.dumps({"error": f"Unknown method: {method}"})

    # Check pydiggs availability
    if not has_pydiggs():
        return json.dumps({
            "error": "pydiggs is not installed. Install with: pip install pydiggs"
        })

    # Execute method
    try:
        if method == "validate_schema":
            result = validate_diggs_schema(
                filepath=params.get("filepath"),
                content=params.get("content"),
                schema_version=params.get("schema_version", "2.6")
            )
        elif method == "validate_dictionary":
            result = validate_diggs_dictionary(
                filepath=params.get("filepath"),
                content=params.get("content")
            )
        else:
            return json.dumps({"error": f"Method not implemented: {method}"})

        return json.dumps(result.to_dict())

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "error_type": type(e).__name__
        })
