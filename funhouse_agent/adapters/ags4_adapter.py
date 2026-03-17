"""AGS4 adapter — read and validate AGS4 geotechnical data files."""

from funhouse_agent.adapters import clean_result


def _run_read_ags4(params):
    from ags4_agent import has_ags4, read_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    filepath = params.get("file_path")
    content = params.get("content")
    encoding = params.get("encoding", "utf-8")
    include_data = params.get("include_data", True)
    convert_numeric = params.get("convert_numeric", True)

    result = read_ags4(
        filepath=filepath,
        content=content,
        encoding=encoding,
        include_data=include_data,
        convert_numeric=convert_numeric,
    )
    return clean_result(result.to_dict())


def _run_validate_ags4(params):
    from ags4_agent import has_ags4, validate_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    filepath = params.get("file_path")
    content = params.get("content")
    encoding = params.get("encoding", "utf-8")

    result = validate_ags4(
        filepath=filepath,
        content=content,
        encoding=encoding,
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "read_ags4": _run_read_ags4,
    "validate_ags4": _run_validate_ags4,
}

METHOD_INFO = {
    "read_ags4": {
        "category": "File Import",
        "brief": "Read and parse an AGS4 geotechnical data file into structured tables.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
            "include_data": {"type": "bool", "required": False, "default": True, "description": "Include all table data in result."},
            "convert_numeric": {"type": "bool", "required": False, "default": True, "description": "Convert numeric columns from text."},
        },
        "returns": {
            "filepath": "Source file path or '<string>'.",
            "n_groups": "Number of AGS4 groups (tables) found.",
            "group_names": "Names of all groups.",
            "group_row_counts": "Row counts per group.",
            "tables": "Dict of group_name to list of row dicts (if include_data=True).",
        },
    },
    "validate_ags4": {
        "category": "File Validation",
        "brief": "Validate an AGS4 file against AGS4 rules.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
        },
        "returns": {
            "filepath": "Source file path.",
            "n_errors": "Number of errors found.",
            "n_warnings": "Number of warnings found.",
            "n_fyi": "Number of FYI messages.",
            "is_valid": "True if no errors (warnings/FYI acceptable).",
            "errors": "Error details grouped by rule number.",
        },
    },
}
