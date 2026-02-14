"""
Wave Equation Analysis Module

Performs one-dimensional wave equation analysis of pile driving using
Smith's (1960) model. Predicts blow count vs capacity (bearing graph),
driving stresses, and drivability for hammer-pile-soil systems.

Methods:
- Smith 1-D wave equation with explicit time integration
- Bearing graph generation (Rult vs blow count)
- Drivability study (blow count vs depth)
- Built-in hammer database (Vulcan, Delmag, ICE)

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    FHWA GEC-12, Chapter 12
    WEAP87 Manual (FHWA, Goble & Rausche)
"""

from wave_equation.hammer import Hammer, get_hammer, list_hammers
from wave_equation.cushion import Cushion, make_cushion_from_properties
from wave_equation.pile_model import PileSegment, PileModel, discretize_pile
from wave_equation.soil_model import SmithSoilModel, SoilSetup
from wave_equation.time_integration import BlowResult, simulate_blow
from wave_equation.bearing_graph import BearingGraphResult, generate_bearing_graph
from wave_equation.drivability import (
    DrivabilityPoint, DrivabilityResult, drivability_study,
)

__all__ = [
    'Hammer', 'get_hammer', 'list_hammers',
    'Cushion', 'make_cushion_from_properties',
    'PileSegment', 'PileModel', 'discretize_pile',
    'SmithSoilModel', 'SoilSetup',
    'BlowResult', 'simulate_blow',
    'BearingGraphResult', 'generate_bearing_graph',
    'DrivabilityPoint', 'DrivabilityResult', 'drivability_study',
]
