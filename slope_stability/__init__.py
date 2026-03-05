"""
Slope Stability Analysis Module

Implements limit equilibrium methods for slope stability:
- Ordinary Method of Slices (Fellenius, 1927)
- Bishop's Simplified Method (Bishop, 1955)
- Spencer's Method (Spencer, 1967)

Includes critical slip surface search via grid-center optimization.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
    Abramson et al. (2002) — Slope Stability and Stabilization Methods
    Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
"""

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import Slice, build_slices
from slope_stability.methods import (
    fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos,
)
from slope_stability.nails import SoilNail
from slope_stability.search import (
    grid_search, optimize_radius, search_noncircular,
    search_pso, search_weak_layer_biased, search_entry_exit,
)
from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.results import (
    SlopeStabilityResult, SliceData, SearchResult,
)

__all__ = [
    'SlopeGeometry', 'SlopeSoilLayer', 'SoilNail',
    'CircularSlipSurface', 'PolylineSlipSurface',
    'Slice', 'build_slices',
    'fellenius_fos', 'bishop_fos', 'spencer_fos', 'morgenstern_price_fos',
    'grid_search', 'optimize_radius', 'search_noncircular',
    'search_pso', 'search_weak_layer_biased', 'search_entry_exit',
    'analyze_slope', 'search_critical_surface',
    'SlopeStabilityResult', 'SliceData', 'SearchResult',
]
