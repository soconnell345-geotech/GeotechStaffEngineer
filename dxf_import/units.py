"""
Unit conversion utilities for DXF geometry import.

Converts coordinates from DXF drawing units (mm, cm, ft, in) to SI meters.
Also detects units from the $INSUNITS DXF header variable.

$INSUNITS values (per DXF spec):
    0 = Unitless, 1 = Inches, 2 = Feet, 3 = Miles, 4 = Millimeters,
    5 = Centimeters, 6 = Meters, 7 = Kilometers, ...
"""

from typing import List, Optional, Tuple

# Conversion factors to meters
UNIT_FACTORS = {
    "m": 1.0,
    "mm": 0.001,
    "cm": 0.01,
    "ft": 0.3048,
    "in": 0.0254,
}

# DXF $INSUNITS header variable mapping
_INSUNITS_MAP = {
    1: "in",
    2: "ft",
    4: "mm",
    5: "cm",
    6: "m",
}


def convert_coords(
    points: List[Tuple[float, float]],
    from_units: str,
    to_units: str = "m",
) -> List[Tuple[float, float]]:
    """Convert a list of (x, y) coordinate pairs between unit systems.

    Parameters
    ----------
    points : list of (float, float)
        Input coordinates.
    from_units : str
        Source units ('m', 'mm', 'cm', 'ft', 'in').
    to_units : str
        Target units. Default 'm'.

    Returns
    -------
    list of (float, float)
        Converted coordinates.

    Raises
    ------
    ValueError
        If either unit string is not recognized.
    """
    if from_units not in UNIT_FACTORS:
        raise ValueError(
            f"Unknown source unit '{from_units}'. "
            f"Supported: {sorted(UNIT_FACTORS.keys())}"
        )
    if to_units not in UNIT_FACTORS:
        raise ValueError(
            f"Unknown target unit '{to_units}'. "
            f"Supported: {sorted(UNIT_FACTORS.keys())}"
        )
    if from_units == to_units:
        return list(points)
    factor = UNIT_FACTORS[from_units] / UNIT_FACTORS[to_units]
    return [(x * factor, y * factor) for x, y in points]


def detect_units_from_header(doc) -> Optional[str]:
    """Read $INSUNITS from a DXF document header and return unit string.

    Parameters
    ----------
    doc : ezdxf.document.Drawing
        An ezdxf document object.

    Returns
    -------
    str or None
        Unit string ('m', 'ft', 'mm', etc.) or None if not set / unknown.
    """
    try:
        insunits = doc.header.get("$INSUNITS", 0)
    except Exception:
        return None
    return _INSUNITS_MAP.get(insunits, None)
