"""
Bearing Capacity Module

Computes ultimate and allowable bearing capacity of shallow foundations
using Meyerhof's and Vesic's general bearing capacity equation.

Supports:
- Strip, rectangular, square, and circular footings
- One- and two-layer soil systems
- Eccentric loading (effective area method)
- Inclined loading, tilted base, sloping ground
- Groundwater at any elevation

References:
    FHWA-SA-94-034 (CBEAR User's Guide)
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6
    FHWA Soils & Foundations Reference Manual, Volume II, Chapter 8
"""

from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.results import BearingCapacityResult

__all__ = [
    'Footing',
    'SoilLayer',
    'BearingSoilProfile',
    'BearingCapacityAnalysis',
    'BearingCapacityResult',
]
