"""
AGS4 agent — AGS4 geotechnical data format reader/validator.

Wraps the python-ags4 library for reading, parsing, and validating
AGS4 format geotechnical data exchange files.

Public API
----------
read_ags4 : Read and parse an AGS4 file or string.
validate_ags4 : Validate an AGS4 file against AGS4 rules.
AGS4ReadResult, AGS4ValidationResult : Result dataclasses.
has_ags4 : Check if python-ags4 is installed.
"""

from ags4_agent.ags4_reader import read_ags4, validate_ags4
from ags4_agent.results import AGS4ReadResult, AGS4ValidationResult
from ags4_agent.ags4_utils import has_ags4

__all__ = [
    "read_ags4",
    "validate_ags4",
    "AGS4ReadResult",
    "AGS4ValidationResult",
    "has_ags4",
]
