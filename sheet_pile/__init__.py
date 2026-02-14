"""
Sheet Pile Wall Analysis Module

Designs and analyzes cantilever and anchored sheet pile retaining walls
using classical earth pressure methods (Rankine, Coulomb).

Determines:
- Required embedment depth
- Maximum bending moment
- Anchor force (anchored walls)

References:
    USACE EM 1110-2-2504 (Design of Sheet Pile Walls)
    USS Steel Sheet Piling Design Manual
    Dawkins (1991), CWALSHT methodology
"""

from sheet_pile.earth_pressure import (
    rankine_Ka, rankine_Kp, coulomb_Ka, coulomb_Kp, K0,
    active_pressure, passive_pressure, tension_crack_depth,
)
from sheet_pile.cantilever import (
    WallSoilLayer, CantileverWallResult, analyze_cantilever,
)
from sheet_pile.anchored import (
    AnchoredWallResult, analyze_anchored,
)

__all__ = [
    'rankine_Ka', 'rankine_Kp', 'coulomb_Ka', 'coulomb_Kp', 'K0',
    'active_pressure', 'passive_pressure', 'tension_crack_depth',
    'WallSoilLayer', 'CantileverWallResult', 'analyze_cantilever',
    'AnchoredWallResult', 'analyze_anchored',
]
