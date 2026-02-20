# pydiggs_agent Design Document

## Purpose

Wrapper module for pydiggs library to validate DIGGS (Data Interchange for Geotechnical and Geoenvironmental Specialists) XML files. Provides schema validation, dictionary validation, and integration with the GeotechStaffEngineer Foundry agent framework.

## Background

DIGGS is an international XML standard for exchanging geotechnical and geoenvironmental data. The pydiggs library validates DIGGS XML files but has a file-only interface and attribute-based result extraction. This wrapper provides:

1. String content support (via temp files)
2. Dataclass-based results for LLM agents
3. Consistent error handling
4. Foundry agent integration

## pydiggs Library Overview

pydiggs v0.1.5 is **validation-only**. It does NOT parse or extract data from DIGGS XML.

### validator Class

```python
from pydiggs import validator

v = validator(
    instance_path="file.xml",       # DIGGS XML file path (required)
    schema_path=None,               # Custom XSD (optional, uses bundled 2.6)
    dictionary_path=None,           # Custom dictionary (optional)
    schematron_path=None,           # Schematron rules (optional)
    output_log=True                 # Write .log files to CWD
)

v.schema_check()       # Validate against XSD
v.dictionary_check()   # Validate propertyClass values
v.schematron_check()   # Validate schematron rules
```

### Result Attributes

After calling check methods, read these attributes:

- `v.syntax_error_log` - XMLSyntaxError or None
- `v.schema_validation_log` - Error log or None (None = valid)
- `v.schema_error_log` - XMLSchemaParseError or None
- `v.dictionary_validation_log` - list[str] of undefined properties or None
- `v.schematron_validation_log` - Error log or None
- `v.schematron_error_log` - SchematronParseError or None

### Key Constraints

1. **File-only interface**: No XML string support - must write to temp file
2. **Console printing**: Methods use rich.print() - output can't be suppressed
3. **No return values**: Methods return None - must read attributes
4. **Log file pollution**: `output_log=True` writes .log files to CWD

## Module Architecture

### Files

```
pydiggs_agent/
  __init__.py                    # Public API exports
  pydiggs_utils.py               # has_pydiggs(), get_schema_path(), get_dictionary_path()
  diggs_validation.py            # validate_diggs_schema(), validate_diggs_dictionary()
  results.py                     # DiggValidationResult dataclass
  tests/
    __init__.py
    test_pydiggs_agent.py        # ~30 tests (Tier 1 + Tier 2)
  DESIGN.md                      # This file
```

### Public API

```python
from pydiggs_agent import (
    has_pydiggs,                  # Runtime availability check
    validate_diggs_schema,        # Schema validation
    validate_diggs_dictionary,    # Dictionary validation
    DiggValidationResult,         # Result dataclass
)
```

## Validation Functions

### validate_diggs_schema()

```python
def validate_diggs_schema(
    filepath: Optional[str] = None,
    content: Optional[str] = None,
    schema_version: str = "2.6"
) -> DiggValidationResult:
    """
    Validate DIGGS XML against XSD schema.

    Args:
        filepath: Path to XML file (mutually exclusive with content)
        content: XML string (mutually exclusive with filepath)
        schema_version: "2.6" (default) or "2.5.a"

    Returns:
        DiggValidationResult

    Raises:
        ImportError: If pydiggs not installed
        ValueError: If input validation fails
    """
```

**Implementation notes**:
- Exactly one of filepath or content must be provided
- For content: write to `tempfile.NamedTemporaryFile(suffix='.xml', delete=False)`
- Always use `output_log=False` to prevent .log file pollution
- Clean up temp file in finally block
- Check `v.syntax_error_log`, `v.schema_error_log`, `v.schema_validation_log` in order
- `None` validation log means valid; presence means errors

### validate_diggs_dictionary()

```python
def validate_diggs_dictionary(
    filepath: Optional[str] = None,
    content: Optional[str] = None
) -> DiggValidationResult:
    """
    Validate DIGGS propertyClass values against dictionary.

    Args:
        filepath: Path to XML file (mutually exclusive with content)
        content: XML string (mutually exclusive with filepath)

    Returns:
        DiggValidationResult

    Raises:
        ImportError: If pydiggs not installed
        ValueError: If input validation fails
    """
```

**Implementation notes**:
- Same input handling as schema validation
- `v.dictionary_validation_log` is list of undefined property names or None
- None means valid; list means invalid with n_errors = len(list)

## Result Dataclass

### DiggValidationResult

```python
@dataclass
class DiggValidationResult:
    source: str                    # Filename or "content"
    check_type: str                # "schema", "dictionary", "schematron"
    schema_version: Optional[str]  # "2.6", "2.5.a", or None
    is_valid: bool                 # Validation passed
    n_errors: int                  # Count of errors
    errors: list[str]              # Error messages

    def summary() -> str:          # Text report
    def to_dict() -> dict:         # JSON serialization
```

**Field logic**:
- `source`: os.path.basename(filepath) or "content"
- `check_type`: "schema" for schema validation, "dictionary" for dictionary
- `schema_version`: Only set for schema validation
- `is_valid`: False if any errors found
- `n_errors`: len(errors)
- `errors`: List of error strings (converted from lxml objects)

## Bundled Resources

pydiggs bundles schemas and dictionaries:

```
{pydiggs_install}/
  schemas/
    diggs-schema-2.6/
      Diggs.xsd              # Default schema
    diggs-schema-2.5.a/
      Complete.xsd           # Legacy schema
  dictionaries/
    properties.xml           # DIGGS property dictionary
```

Use `get_schema_path(version)` and `get_dictionary_path()` to locate these.

## Error Handling

### Input Validation Errors

Raise `ValueError` for:
- Neither filepath nor content provided
- Both filepath and content provided
- Invalid schema_version (not "2.6" or "2.5.a")

### Missing Dependency

Raise `ImportError` if pydiggs not installed:
```python
if not has_pydiggs():
    raise ImportError("pydiggs is not installed. Install with: pip install pydiggs")
```

### XML Validation Errors

Validation errors are captured in the result, not raised:
- Syntax errors → `errors` list with 1 entry
- Schema errors → `errors` list with parsed error messages
- Dictionary errors → `errors` list with undefined property names

## Testing Strategy

### Tier 1: No pydiggs required (~18 tests)

Test functionality that doesn't need the library:
- DiggValidationResult creation, defaults, summary, to_dict
- JSON serialization of result
- Input validation (missing/duplicate filepath+content, bad schema version)
- has_pydiggs() returns bool
- Foundry metadata (list_methods, describe_method)

### Tier 2: Requires pydiggs (~12 tests)

Test actual validation (marked with `@pytest.mark.skipif(not has_pydiggs())`):
- Schema validation: valid XML, invalid XML, syntax error
- Schema validation: 2.5.a version
- Schema validation from content string
- Dictionary validation: valid and invalid
- Dictionary validation from content string
- Temp file cleanup after content validation
- Foundry integration calls
- Result summary with real errors

### Test XML Samples

See `test_pydiggs_agent.py` for minimal valid/invalid DIGGS XML constants.

## Foundry Agent Integration

File: `pydiggs_agent_foundry.py` in project root

### Methods

1. **validate_schema** (category: "Validation")
   - Params: `filepath`, `content`, `schema_version`
   - Returns: DiggValidationResult dict

2. **validate_dictionary** (category: "Validation")
   - Params: `filepath`, `content`
   - Returns: DiggValidationResult dict

### Functions

```python
def pydiggs_agent(method: str, params_json: str) -> str:
    """Execute pydiggs method, return JSON result"""

def pydiggs_list_methods(category: Optional[str] = None) -> list[dict]:
    """List available methods with metadata"""

def pydiggs_describe_method(method: str) -> dict:
    """Get detailed method documentation"""
```

## Usage Examples

### Schema Validation from File

```python
from pydiggs_agent import validate_diggs_schema

result = validate_diggs_schema(filepath="borehole_data.xml")
print(result.summary())

if not result.is_valid:
    print(f"Found {result.n_errors} errors")
    for error in result.errors:
        print(f"  - {error}")
```

### Schema Validation from String

```python
xml_content = """<?xml version="1.0"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6">
  ...
</Diggs>"""

result = validate_diggs_schema(content=xml_content, schema_version="2.6")
```

### Dictionary Validation

```python
from pydiggs_agent import validate_diggs_dictionary

result = validate_diggs_dictionary(filepath="lab_data.xml")
if result.is_valid:
    print("All propertyClass values are valid")
else:
    print("Undefined properties:", result.errors)
```

### Via Foundry Agent

```python
from pydiggs_agent_foundry import pydiggs_agent
import json

params = {"filepath": "test.xml", "schema_version": "2.6"}
result_json = pydiggs_agent("validate_schema", json.dumps(params))
result = json.loads(result_json)
```

## Dependencies

- **pydiggs** (optional): ~0.1.5
  - Requires: lxml, rich
- **Standard library**: os, tempfile, dataclasses, typing

Install with:
```bash
pip install pydiggs
```

## Known Limitations

1. **Validation only**: pydiggs doesn't parse/extract data - use separate XML parser
2. **Console output**: rich.print() output from pydiggs can't be suppressed
3. **File performance**: Content input requires temp file write/cleanup
4. **Schema coverage**: Only supports DIGGS 2.6 and 2.5.a (bundled schemas)
5. **No schematron**: This wrapper doesn't expose schematron validation (rarely used)

## Future Enhancements

Potential additions if needed:
1. Schematron validation wrapper
2. Custom schema/dictionary path support
3. Batch validation of multiple files
4. Integration with DIGGS data extraction (separate module)
5. Validation report export (HTML/PDF)

## References

- pydiggs: https://pypi.org/project/pydiggs/
- DIGGS standard: https://diggsml.org/
- DIGGS 2.6 spec: https://diggsml.org/schemas/2.6/
