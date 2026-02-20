"""
swprocess agent — MASW surface wave dispersion analysis.

Wraps the swprocess library for multichannel analysis of surface
waves (MASW) to extract dispersion curves from seismic array data.

Public API
----------
analyze_masw : Run MASW dispersion analysis.
DispersionResult : Result dataclass.
has_swprocess : Check if swprocess is installed.
"""

from swprocess_agent.masw_analysis import analyze_masw
from swprocess_agent.results import DispersionResult
from swprocess_agent.swprocess_utils import has_swprocess

__all__ = [
    "analyze_masw",
    "DispersionResult",
    "has_swprocess",
]
