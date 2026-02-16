"""
Downdrag (Negative Skin Friction) Analysis Module

Implements the Fellenius unified method for pile downdrag analysis.
Finds the neutral plane depth, computes dragload and settlement, and
checks structural and geotechnical limit states.

Supports fill placement and groundwater drawdown as settlement triggers.

References:
    - Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
    - Fellenius, B.H. (2004). ASCE GSP 125.
    - AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
    - UFC 3-220-20, Chapter 6.
"""

from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import DowndragAnalysis
from downdrag.results import DowndragResult

__all__ = [
    'DowndragSoilLayer', 'DowndragSoilProfile',
    'DowndragAnalysis', 'DowndragResult',
]
