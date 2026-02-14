"""
Drilled Shaft Axial Capacity Module

Estimates ultimate axial capacity of drilled shafts (bored piles)
using FHWA GEC-10 methods:
- Alpha method for side resistance in cohesive soils
- Beta method for side resistance in cohesionless soils
- Rock socket methods for side resistance in rock
- End bearing for clay, sand, and rock

References:
    FHWA GEC-10 (FHWA-NHI-10-016), Brown et al. (2010/2018)
    O'Neill & Reese (1999), FHWA-RD-99-049
"""

from drilled_shaft.shaft import DrillShaft
from drilled_shaft.soil_profile import ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.capacity import DrillShaftAnalysis
from drilled_shaft.results import DrillShaftResult

__all__ = [
    'DrillShaft', 'ShaftSoilLayer', 'ShaftSoilProfile',
    'DrillShaftAnalysis', 'DrillShaftResult',
]
