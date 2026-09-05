"""Passthrough shim — the canonical implementation moved to planlens.dxf.units
in the 2026-09-04 planlens split. Geotech-side imports keep working unchanged.
"""

from planlens.dxf.units import (  # noqa: F401
    UNIT_FACTORS,
    convert_coords,
    detect_units_from_header,
)
