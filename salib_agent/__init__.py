"""
SALib agent — sensitivity analysis wrapper.

Wraps the SALib library for Sobol variance-based sensitivity analysis
and Morris elementary effects screening for geotechnical parameter studies.

Public API
----------
sobol_sample : Generate Sobol quasi-random sample matrix.
sobol_analyze : Compute Sobol first/total-order sensitivity indices.
morris_sample : Generate Morris OAT sample matrix.
morris_analyze : Compute Morris elementary effects (mu*, sigma).
sobol_analysis, morris_analysis : One call — sample, evaluate a model (a
    callable or an arithmetic expression of the variables), analyze; returns
    the indices only.
SobolResult, MorrisResult : Result dataclasses.
has_salib : Check if SALib is installed.
"""

from salib_agent.sensitivity import (
    sobol_sample,
    sobol_analyze,
    morris_sample,
    morris_analyze,
    sobol_analysis,
    morris_analysis,
)
from salib_agent.expression import compile_expression
from salib_agent.results import SobolResult, MorrisResult
from salib_agent.salib_utils import has_salib

__all__ = [
    "sobol_analysis",
    "morris_analysis",
    "compile_expression",
    "sobol_sample",
    "sobol_analyze",
    "morris_sample",
    "morris_analyze",
    "SobolResult",
    "MorrisResult",
    "has_salib",
]
