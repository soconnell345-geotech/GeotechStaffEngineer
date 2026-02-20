"""
pydiggs_agent - DIGGS XML validation wrapper

This module provides validation functions for DIGGS (Data Interchange for Geotechnical and
Geoenvironmental Specialists) XML files. It wraps the pydiggs library for schema and
dictionary validation.

pydiggs is an optional dependency. Use has_pydiggs() to check availability at runtime.

Public API:
-----------
validate_diggs_schema() - Validate DIGGS XML against XSD schema
validate_diggs_dictionary() - Validate DIGGS propertyClass values against dictionary
DiggValidationResult - Result dataclass with summary() and to_dict()
has_pydiggs() - Check if pydiggs is installed

Example:
--------
>>> from pydiggs_agent import validate_diggs_schema, has_pydiggs
>>> if has_pydiggs():
...     result = validate_diggs_schema(filepath="report.xml")
...     print(result.summary())
"""

from pydiggs_agent.pydiggs_utils import has_pydiggs
from pydiggs_agent.results import DiggValidationResult
from pydiggs_agent.diggs_validation import (
    validate_diggs_schema,
    validate_diggs_dictionary,
)

__all__ = [
    "has_pydiggs",
    "validate_diggs_schema",
    "validate_diggs_dictionary",
    "DiggValidationResult",
]
