"""
GSTools agent — geostatistical analysis wrapper.

Wraps the GSTools library for kriging interpolation, variogram analysis,
and spatial random field generation for geotechnical soil properties.

Public API
----------
analyze_kriging : Krige soil property values onto a regular grid.
analyze_variogram : Estimate and fit empirical variograms.
generate_random_field : Generate 2D spatial random fields.
KrigingResult, VariogramResult, RandomFieldResult : Result dataclasses.
has_gstools : Check if gstools is installed.
"""

from gstools_agent.kriging import analyze_kriging
from gstools_agent.variogram import analyze_variogram
from gstools_agent.random_field import generate_random_field
from gstools_agent.results import KrigingResult, VariogramResult, RandomFieldResult
from gstools_agent.gstools_utils import has_gstools

__all__ = [
    "analyze_kriging",
    "analyze_variogram",
    "generate_random_field",
    "KrigingResult",
    "VariogramResult",
    "RandomFieldResult",
    "has_gstools",
]
