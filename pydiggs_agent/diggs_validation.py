"""
DIGGS XML validation functions.

These functions wrap the pydiggs validator class to provide validation of DIGGS XML
files against XSD schemas and DIGGS dictionaries.
"""

import os
import tempfile
from typing import Optional

from pydiggs_agent.pydiggs_utils import has_pydiggs, get_schema_path, get_dictionary_path
from pydiggs_agent.results import DiggValidationResult


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
