"""
hvsrpy agent — HVSR site characterization wrapper.

Wraps the hvsrpy library for computing Horizontal-to-Vertical Spectral
Ratios from 3-component seismograms to identify site resonant frequency
(f0), peak amplification (A0), and site period (T0).

Public API
----------
analyze_hvsr : Compute HVSR from 3-component arrays.
HvsrResult : Result dataclass with f0, A0, T0, SESAME criteria, curves.
has_hvsrpy : Check if hvsrpy is installed.
"""

from hvsrpy_agent.hvsr_analysis import analyze_hvsr
from hvsrpy_agent.results import HvsrResult
from hvsrpy_agent.hvsrpy_utils import has_hvsrpy

__all__ = [
    "analyze_hvsr",
    "HvsrResult",
    "has_hvsrpy",
]
