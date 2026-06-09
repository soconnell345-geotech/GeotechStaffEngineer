"""
subsurface_characterization.formats — optional, dependency-backed format adapters.

These submodules fold in the former standalone parser/validator modules
(`pygef_agent`, `ags4_agent`, `pydiggs_agent`) as optional format adapters of the
native subsurface_characterization data-I/O home. Each is backed by a third-party
library that is an *optional* dependency — use the ``has_*`` availability check
before calling, and the parsing functions raise a clear error if the library is
missing.

Format adapters
---------------
gef (pygef)              : GEF / BRO-XML CPT & borehole parsing
    parse_cpt_file, parse_bore_file, CPTParseResult, BoreParseResult, has_pygef
ags4 (python-ags4)       : AGS4 read / validate
    read_ags4, validate_ags4, AGS4ReadResult, AGS4ValidationResult, has_ags4
diggs_validation (pydiggs): DIGGS 2.6 schema / dictionary validation
    validate_diggs_schema, validate_diggs_dictionary, DiggValidationResult, has_pydiggs
"""

from subsurface_characterization.formats.gef import (
    parse_cpt_file,
    parse_bore_file,
    has_pygef,
    CPTParseResult,
    BoreParseResult,
)
from subsurface_characterization.formats.ags4 import (
    read_ags4,
    validate_ags4,
    has_ags4,
    AGS4ReadResult,
    AGS4ValidationResult,
)
from subsurface_characterization.formats.diggs_validation import (
    validate_diggs_schema,
    validate_diggs_dictionary,
    has_pydiggs,
    DiggValidationResult,
)

__all__ = [
    # GEF / BRO-XML (pygef)
    "parse_cpt_file",
    "parse_bore_file",
    "has_pygef",
    "CPTParseResult",
    "BoreParseResult",
    # AGS4 (python-ags4)
    "read_ags4",
    "validate_ags4",
    "has_ags4",
    "AGS4ReadResult",
    "AGS4ValidationResult",
    # DIGGS validation (pydiggs)
    "validate_diggs_schema",
    "validate_diggs_dictionary",
    "has_pydiggs",
    "DiggValidationResult",
]
