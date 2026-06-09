"""
DIGGS XML validation format adapter (pydiggs-backed).

Provides validation functions for DIGGS (Data Interchange for Geotechnical and
Geoenvironmental Specialists) XML files. Wraps the optional ``pydiggs`` library for
schema and dictionary validation.

``pydiggs`` is an OPTIONAL dependency. Use ``has_pydiggs()`` to check availability
at runtime; the validation functions raise a clear ImportError if it is missing.

Public API
----------
validate_diggs_schema() - Validate DIGGS XML against XSD schema
validate_diggs_dictionary() - Validate DIGGS propertyClass values against dictionary
DiggValidationResult - Result dataclass with summary() and to_dict()
has_pydiggs() - Check if pydiggs is installed

Note: this is the *validation* adapter (pydiggs). For DIGGS *data extraction* into a
SiteModel, use ``subsurface_characterization.parse_diggs`` (native parser, no
external dependency).

Example
-------
>>> from subsurface_characterization.formats import validate_diggs_schema, has_pydiggs
>>> if has_pydiggs():
...     result = validate_diggs_schema(filepath="report.xml")
...     print(result.summary())
"""

import os
import tempfile
from typing import Optional

from subsurface_characterization.formats.diggs_validation_results import (
    DiggValidationResult,
)


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

def has_pydiggs() -> bool:
    """
    Check if pydiggs is installed.

    Returns:
        bool: True if pydiggs is available, False otherwise
    """
    try:
        import pydiggs  # noqa: F401
        return True
    except ImportError:
        return False


def get_schema_path(version: str = "2.6") -> str:
    """
    Get the path to the bundled DIGGS schema file.

    Args:
        version: Schema version ("2.6" or "2.5.a")

    Returns:
        Absolute path to the schema XSD file

    Raises:
        ImportError: If pydiggs is not installed
        ValueError: If schema version is not supported
    """
    if not has_pydiggs():
        raise ImportError("pydiggs is not installed")

    import pydiggs

    pydiggs_dir = os.path.dirname(pydiggs.__file__)

    if version == "2.6":
        schema_path = os.path.join(pydiggs_dir, "schemas", "diggs-schema-2.6", "Diggs.xsd")
    elif version == "2.5.a":
        schema_path = os.path.join(pydiggs_dir, "schemas", "diggs-schema-2.5.a", "Complete.xsd")
    else:
        raise ValueError(f"Unsupported schema version: {version}. Use '2.6' or '2.5.a'")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    return schema_path


def get_dictionary_path() -> str:
    """
    Get the path to the bundled DIGGS dictionary file.

    Returns:
        Absolute path to the properties.xml dictionary file

    Raises:
        ImportError: If pydiggs is not installed
    """
    if not has_pydiggs():
        raise ImportError("pydiggs is not installed")

    import pydiggs

    pydiggs_dir = os.path.dirname(pydiggs.__file__)
    dict_path = os.path.join(pydiggs_dir, "dictionaries", "properties.xml")

    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    return dict_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_diggs_schema(
    filepath: Optional[str] = None,
    content: Optional[str] = None,
    schema_version: str = "2.6"
) -> DiggValidationResult:
    """
    Validate DIGGS XML against XSD schema.

    Args:
        filepath: Path to DIGGS XML file (mutually exclusive with content)
        content: DIGGS XML as string (mutually exclusive with filepath)
        schema_version: Schema version to validate against ("2.6" or "2.5.a")

    Returns:
        DiggValidationResult with validation outcome

    Raises:
        ImportError: If pydiggs is not installed
        ValueError: If neither or both filepath and content are provided,
                   or if schema_version is invalid
    """
    if not has_pydiggs():
        raise ImportError(
            "pydiggs is not installed. Install with: pip install pydiggs"
        )

    # Validate inputs
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Only one of filepath or content should be provided")
    if schema_version not in ("2.6", "2.5.a"):
        raise ValueError(f"Invalid schema version: {schema_version}. Use '2.6' or '2.5.a'")

    from pydiggs import validator

    temp_file = None
    try:
        # Handle content input - write to temp file
        if content is not None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.xml',
                delete=False,
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            filepath = temp_file.name
            source = "content"
        else:
            source = os.path.basename(filepath)

        # Get schema path
        schema_path = get_schema_path(schema_version)

        # Create validator and run schema check
        v = validator(
            instance_path=filepath,
            schema_path=schema_path,
            output_log=False  # Don't write .log files
        )

        # Run schema validation
        v.schema_check()

        # Check for syntax errors
        if v.syntax_error_log is not None:
            errors = [str(v.syntax_error_log)]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=1,
                errors=errors
            )

        # Check for schema parse errors
        if v.schema_error_log is not None:
            errors = [str(v.schema_error_log)]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=1,
                errors=errors
            )

        # Check validation results
        if v.schema_validation_log is None:
            # None means valid
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=True,
                n_errors=0,
                errors=[]
            )
        else:
            # Has errors
            errors = [str(e) for e in v.schema_validation_log]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=len(errors),
                errors=errors
            )

    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Best effort cleanup


def validate_diggs_dictionary(
    filepath: Optional[str] = None,
    content: Optional[str] = None
) -> DiggValidationResult:
    """
    Validate DIGGS propertyClass values against DIGGS dictionary.

    Args:
        filepath: Path to DIGGS XML file (mutually exclusive with content)
        content: DIGGS XML as string (mutually exclusive with filepath)

    Returns:
        DiggValidationResult with validation outcome

    Raises:
        ImportError: If pydiggs is not installed
        ValueError: If neither or both filepath and content are provided
    """
    if not has_pydiggs():
        raise ImportError(
            "pydiggs is not installed. Install with: pip install pydiggs"
        )

    # Validate inputs
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Only one of filepath or content should be provided")

    from pydiggs import validator

    temp_file = None
    try:
        # Handle content input - write to temp file
        if content is not None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.xml',
                delete=False,
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            filepath = temp_file.name
            source = "content"
        else:
            source = os.path.basename(filepath)

        # Get dictionary path
        dictionary_path = get_dictionary_path()

        # Create validator and run dictionary check
        v = validator(
            instance_path=filepath,
            dictionary_path=dictionary_path,
            output_log=False  # Don't write .log files
        )

        # Run dictionary validation
        v.dictionary_check()

        # Check validation results — attribute only exists when errors found
        if getattr(v, 'dictionary_validation_log', None) is None:
            # None means valid
            return DiggValidationResult(
                source=source,
                check_type="dictionary",
                is_valid=True,
                n_errors=0,
                errors=[]
            )
        else:
            # Has undefined properties
            errors = v.dictionary_validation_log
            return DiggValidationResult(
                source=source,
                check_type="dictionary",
                is_valid=False,
                n_errors=len(errors),
                errors=errors
            )

    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Best effort cleanup
