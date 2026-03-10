"""pydiggs adapter — DIGGS XML schema and dictionary validation."""

from funhouse_agent.adapters import clean_result


def _run_validate_diggs_schema(params):
    from pydiggs_agent import has_pydiggs, validate_diggs_schema

    if not has_pydiggs():
        return {"error": "pydiggs is not installed. Install with: pip install pydiggs"}

    filepath = params.get("file_path")
    content = params.get("content")
    schema_version = params.get("schema_version", "2.6")

    result = validate_diggs_schema(
        filepath=filepath,
        content=content,
        schema_version=schema_version,
    )
    return clean_result(result.to_dict())


def _run_validate_diggs_dictionary(params):
    from pydiggs_agent import has_pydiggs, validate_diggs_dictionary

    if not has_pydiggs():
        return {"error": "pydiggs is not installed. Install with: pip install pydiggs"}

    filepath = params.get("file_path")
    content = params.get("content")

    result = validate_diggs_dictionary(
        filepath=filepath,
        content=content,
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "validate_diggs_schema": _run_validate_diggs_schema,
    "validate_diggs_dictionary": _run_validate_diggs_dictionary,
}

METHOD_INFO = {
    "validate_diggs_schema": {
        "category": "File Validation",
        "brief": "Validate DIGGS XML against XSD schema (v2.6 or v2.5.a).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
            "schema_version": {"type": "str", "required": False, "default": "2.6", "description": "Schema version: '2.6' or '2.5.a'."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'schema'.",
            "schema_version": "Schema version validated against.",
            "is_valid": "Whether validation passed.",
            "n_errors": "Number of validation errors.",
            "errors": "List of error messages.",
        },
    },
    "validate_diggs_dictionary": {
        "category": "File Validation",
        "brief": "Validate DIGGS propertyClass values against DIGGS dictionary.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'dictionary'.",
            "is_valid": "Whether all propertyClass values are valid.",
            "n_errors": "Number of undefined properties.",
            "errors": "List of undefined property messages.",
        },
    },
}
