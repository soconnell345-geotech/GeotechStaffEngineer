"""
Lateral Pile Analysis Module

A Python implementation of laterally loaded pile analysis using p-y curve methods,
based on the public-domain methods from FHWA's COM624P program and modern
advancements including Jeanjean (2009) soft clay.

References:
    - COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
    - FHWA GEC-13: FHWA-HIF-18-031
    - Jeanjean (2009): OTC-20158-MS
"""

from lateral_pile.pile import Pile
from lateral_pile.soil import SoilLayer
from lateral_pile.analysis import LateralPileAnalysis

__all__ = ['Pile', 'SoilLayer', 'LateralPileAnalysis']
