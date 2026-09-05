"""
Downdrag (Negative Skin Friction) Analysis Module

Implements the Fellenius unified method for pile downdrag analysis.
Finds the neutral plane depth, computes dragload and settlement, and
checks structural and geotechnical limit states.

Supports fill placement and groundwater drawdown as settlement triggers.

Also implements the CGPR #56 method family (Greenfield & Filz 2009,
"Downdrag and Drag Load on Piles", Virginia Tech CGPR #56): Endo, Poulos
hand approximation, the report's Fellenius formulation, the PILENEG
procedure, pile-group rigid block / drag load reduction methods, and a
multi-method comparison runner. See downdrag/cgpr56.py.

References:
    - Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
    - Fellenius, B.H. (2004). ASCE GSP 125.
    - AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
    - UFC 3-220-20, Chapter 6.
    - Greenfield, M.L. and Filz, G.M. (2009). Virginia Tech CGPR #56.
"""

from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import DowndragAnalysis
from downdrag.results import DowndragResult
from downdrag.cgpr56 import (
    endo_method, poulos_method, fellenius_method_cgpr56, pileneg_procedure,
    rigid_block_method, drag_load_reduction_factor, drag_load_reduction_method,
    downdrag_method_comparison, consolidation_settlement_profile,
    ENDO_NEUTRAL_PLANE_RATIOS,
    EndoResult, PoulosResult, FelleniusCgpr56Result, PilenegResult,
    RigidBlockResult, GroupReductionResult, MethodComparisonResult,
)

__all__ = [
    'DowndragSoilLayer', 'DowndragSoilProfile',
    'DowndragAnalysis', 'DowndragResult',
    'endo_method', 'poulos_method', 'fellenius_method_cgpr56',
    'pileneg_procedure', 'rigid_block_method', 'drag_load_reduction_factor',
    'drag_load_reduction_method', 'downdrag_method_comparison',
    'consolidation_settlement_profile', 'ENDO_NEUTRAL_PLANE_RATIOS',
    'EndoResult', 'PoulosResult', 'FelleniusCgpr56Result', 'PilenegResult',
    'RigidBlockResult', 'GroupReductionResult', 'MethodComparisonResult',
]
