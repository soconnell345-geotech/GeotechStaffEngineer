"""
GSTools agent — geostatistical analysis wrapper.

Wraps the GSTools library for kriging interpolation, variogram analysis,
spatial random field generation and conditional simulation of soil and rock
properties.

Public API
----------
analyze_kriging : Krige values onto a regular grid (one smoothed best
    estimate: the mean surface), with optional region totals on it.
analyze_variogram : Estimate and fit empirical variograms.
generate_random_field : Generate 2D UNconditioned spatial random fields.
analyze_conditional_simulation : Realizations that honour the data, with
    totals (volume, area) over plan regions and their probabilities.
KrigingResult, VariogramResult, RandomFieldResult,
ConditionalSimulationResult : Result dataclasses.
has_gstools : Check if gstools is installed.
"""

from gstools_agent.kriging import analyze_kriging
from gstools_agent.variogram import analyze_variogram
from gstools_agent.random_field import generate_random_field
from gstools_agent.conditional import analyze_conditional_simulation
from gstools_agent.results import (
    ConditionalSimulationResult, KrigingResult, RandomFieldResult,
    VariogramResult,
)
from gstools_agent.gstools_utils import has_gstools, gstools_import_error

__all__ = [
    "analyze_kriging",
    "analyze_variogram",
    "generate_random_field",
    "analyze_conditional_simulation",
    "KrigingResult",
    "VariogramResult",
    "RandomFieldResult",
    "ConditionalSimulationResult",
    "has_gstools",
    "gstools_import_error",
]
