"""
AGS4 format adapter (python-ags4-backed).

Wraps the optional ``python-ags4`` library for reading, parsing, and validating
AGS4 format geotechnical data exchange files.

``python-ags4`` is an OPTIONAL dependency — call ``has_ags4()`` to check
availability at runtime; the functions raise a clear ImportError if it is missing.

Public API
----------
read_ags4 : Read and parse an AGS4 file or string.
validate_ags4 : Validate an AGS4 file against AGS4 rules.
AGS4ReadResult, AGS4ValidationResult : Result dataclasses.
has_ags4 : Check if python-ags4 is installed.
"""

import io

from subsurface_characterization.formats.ags4_results import (
    AGS4ReadResult,
    AGS4ValidationResult,
)


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

def has_ags4():
    """Check if python-ags4 is installed and importable."""
    try:
        import python_ags4  # noqa: F401
        return True
    except ImportError:
        return False


def import_ags4():
    """Import and return the AGS4 class from python-ags4."""
    try:
        from python_ags4 import AGS4
        return AGS4
    except ImportError:
        raise ImportError(
            "python-ags4 is not installed. Install with: pip install python-ags4"
        )


# ---------------------------------------------------------------------------
# Read / validate
# ---------------------------------------------------------------------------

def _validate_read_inputs(filepath_or_content, is_content):
    """Validate inputs for read_ags4."""
    if is_content:
        if not isinstance(filepath_or_content, str) or len(filepath_or_content.strip()) == 0:
            raise ValueError("content must be a non-empty string of AGS4 data")
    else:
        if not isinstance(filepath_or_content, str) or len(filepath_or_content.strip()) == 0:
            raise ValueError("filepath must be a non-empty string")


def read_ags4(
    filepath=None,
    content=None,
    encoding="utf-8",
    include_data=True,
    convert_numeric=True,
) -> AGS4ReadResult:
    """Read an AGS4 file and return structured data.

    Parameters
    ----------
    filepath : str or None
        Path to an AGS4 file. Provide either filepath or content.
    content : str or None
        Raw AGS4 content as a string.
    encoding : str
        File encoding. Default 'utf-8'.
    include_data : bool
        If True, include all table data in result. Default True.
    convert_numeric : bool
        If True, convert numeric columns from text. Default True.

    Returns
    -------
    AGS4ReadResult
        Parsed AGS4 data with group info and optional table data.
    """
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Provide either filepath or content, not both")

    AGS4 = import_ags4()

    if content is not None:
        _validate_read_inputs(content, is_content=True)
        source = io.StringIO(content)
        source_name = "<string>"
    else:
        _validate_read_inputs(filepath, is_content=False)
        source = filepath
        source_name = filepath

    tables, headings = AGS4.AGS4_to_dataframe(source, encoding=encoding)

    group_names = list(tables.keys())
    group_row_counts = {}
    result_tables = {}

    for name, df in tables.items():
        data_df = df
        if convert_numeric:
            # convert_to_numeric strips UNIT/TYPE rows, leaving only DATA rows
            data_df = AGS4.convert_to_numeric(df)
            n_data = len(data_df)
        else:
            # Raw dataframe has UNIT + TYPE rows at positions 0-1
            n_data = max(len(data_df) - 2, 0)
        group_row_counts[name] = n_data
        if include_data:
            # Convert to list of dicts for JSON serialization
            result_tables[name] = data_df.to_dict(orient="records")

    return AGS4ReadResult(
        filepath=source_name,
        n_groups=len(group_names),
        group_names=group_names,
        group_row_counts=group_row_counts,
        tables=result_tables if include_data else None,
    )


def validate_ags4(
    filepath=None,
    content=None,
    encoding="utf-8",
) -> AGS4ValidationResult:
    """Validate an AGS4 file against AGS4 rules.

    Parameters
    ----------
    filepath : str or None
        Path to an AGS4 file.
    content : str or None
        Raw AGS4 content as a string.
    encoding : str
        File encoding. Default 'utf-8'.

    Returns
    -------
    AGS4ValidationResult
        Validation results with error/warning counts.
    """
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Provide either filepath or content, not both")

    AGS4 = import_ags4()

    if content is not None:
        source = io.StringIO(content)
        source_name = "<string>"
    else:
        source = filepath
        source_name = filepath

    ags_errors = AGS4.check_file(source, encoding=encoding)

    # Count errors by type
    error_counts = AGS4.count_errors(ags_errors)

    # Convert error dict for JSON — keys are rule numbers
    errors_clean = {}
    for rule, messages in ags_errors.items():
        if isinstance(messages, list):
            errors_clean[str(rule)] = [str(m) for m in messages]
        else:
            errors_clean[str(rule)] = str(messages)

    n_errors = error_counts.get("Errors", 0) if isinstance(error_counts, dict) else 0
    n_warnings = error_counts.get("Warnings", 0) if isinstance(error_counts, dict) else 0
    n_fyi = error_counts.get("FYI", 0) if isinstance(error_counts, dict) else 0

    return AGS4ValidationResult(
        filepath=source_name,
        n_errors=n_errors,
        n_warnings=n_warnings,
        n_fyi=n_fyi,
        is_valid=(n_errors == 0),
        errors=errors_clean,
    )
