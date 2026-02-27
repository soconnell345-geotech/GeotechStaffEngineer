"""
Wind Loads Module (ASCE 7-22)

Computes wind loads on freestanding walls and fences per ASCE 7-22 Chapter 29.3.

Supports:
- Velocity pressure with exposure categories B, C, D
- Topographic factor Kzt for ridges, escarpments, and hills
- Ground elevation factor Ke
- Net force coefficient Cf for freestanding walls (Figure 29.3-1)
- Porosity reduction for fences
- Clearance (elevated walls)

All units SI: m, m/s, Pa, kPa, kN, kN/m, degrees.

References:
    ASCE 7-22 Minimum Design Loads and Associated Criteria for
    Buildings and Other Structures, Chapters 26 and 29.
"""

from wind_loads.wind_pressure import (
    compute_Kz,
    compute_Kzt,
    compute_Ke,
    compute_velocity_pressure,
)
from wind_loads.freestanding_wall import (
    get_Cf_freestanding_wall,
    analyze_freestanding_wall_wind,
    analyze_fence_wind,
)
from wind_loads.results import (
    VelocityPressureResult,
    FreestandingWallWindResult,
)

__all__ = [
    "compute_Kz",
    "compute_Kzt",
    "compute_Ke",
    "compute_velocity_pressure",
    "get_Cf_freestanding_wall",
    "analyze_freestanding_wall_wind",
    "analyze_fence_wind",
    "VelocityPressureResult",
    "FreestandingWallWindResult",
]
