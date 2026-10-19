"""
Downdrag (Negative Skin Friction) Analysis Module

Implements the neutral plane method for pile downdrag analysis (UFC
3-220-20 §6-7.4 and §6-5.8.4, FHWA GEC-12 §7.3.6.1, Fellenius 2004).
Finds the neutral plane by every method the references give, side by side,
computes drag load, drag force and settlement, and checks structural and
geotechnical limit states.

Supports fill placement and groundwater drawdown as settlement triggers,
and post-liquefaction downdrag (FHWA GEC-10 §10.7): the same analysis before
the earthquake and after reconsolidation of the liquefied layers, with the
cited side-resistance options for those layers side by side. See
downdrag/liquefaction.py.

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
    - FHWA GEC-10 (FHWA-NHI-18-024), §10.7 Post-Liquefaction Downdrag.
"""

from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import DowndragAnalysis, NEUTRAL_PLANE_METHODS
from downdrag.results import DowndragResult
from downdrag.cgpr56 import (
    endo_method, poulos_method, fellenius_method_cgpr56, pileneg_procedure,
    rigid_block_method, drag_load_reduction_factor, drag_load_reduction_method,
    downdrag_method_comparison, consolidation_settlement_profile,
    ENDO_NEUTRAL_PLANE_RATIOS,
    EndoResult, PoulosResult, FelleniusCgpr56Result, PilenegResult,
    RigidBlockResult, GroupReductionResult, MethodComparisonResult,
)
from downdrag.liquefaction import (
    post_liquefaction_downdrag, PostLiquefactionResult, LiquefiedLayer,
    liquefied_layers_from_result, residual_strength_kramer_wang,
    LIQUEFIED_SIDE_RESISTANCE,
)

__all__ = [
    'DowndragSoilLayer', 'DowndragSoilProfile',
    'DowndragAnalysis', 'DowndragResult', 'NEUTRAL_PLANE_METHODS',
    'endo_method', 'poulos_method', 'fellenius_method_cgpr56',
    'pileneg_procedure', 'rigid_block_method', 'drag_load_reduction_factor',
    'drag_load_reduction_method', 'downdrag_method_comparison',
    'consolidation_settlement_profile', 'ENDO_NEUTRAL_PLANE_RATIOS',
    'EndoResult', 'PoulosResult', 'FelleniusCgpr56Result', 'PilenegResult',
    'RigidBlockResult', 'GroupReductionResult', 'MethodComparisonResult',
    'post_liquefaction_downdrag', 'PostLiquefactionResult', 'LiquefiedLayer',
    'liquefied_layers_from_result', 'residual_strength_kramer_wang',
    'LIQUEFIED_SIDE_RESISTANCE',
]
