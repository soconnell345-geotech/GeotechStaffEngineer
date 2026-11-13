"""
Retaining Wall Design Module

Implements external stability checks for:
- Cantilever retaining walls (sliding, overturning, bearing)
- MSE walls (external + internal stability per FHWA GEC-11)

and the structural design of a cantilever wall's concrete (stem, toe and
heel) on USACE EM 1110-2-2104 (2023), ``structural`` (GM-312).

References:
    AASHTO LRFD Bridge Design Specifications, Section 11
    FHWA GEC-11 (FHWA-NHI-10-024), MSE Walls and Reinforced Slopes
    Das, B.M., Principles of Foundation Engineering
"""

from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.mse import analyze_mse_wall, check_external_stability_lrfd
from retaining_walls.reinforcement import Reinforcement
from retaining_walls.results import CantileverWallResult, MSEWallResult
from retaining_walls.structural import (
    CantileverWallStructuralResult, analyze_cantilever_wall_structure,
)

__all__ = [
    'CantileverWallGeometry', 'MSEWallGeometry',
    'analyze_cantilever_wall', 'analyze_mse_wall',
    'check_external_stability_lrfd',
    'Reinforcement', 'CantileverWallResult', 'MSEWallResult',
    'analyze_cantilever_wall_structure', 'CantileverWallStructuralResult',
]
