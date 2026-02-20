"""
pystra_agent — Structural reliability analysis using pystra

This module provides a wrapper around the pystra library for structural reliability
analysis in geotechnical engineering. It implements FORM, SORM, and Monte Carlo
methods for calculating reliability indices and probabilities of failure.

Public API:
    analyze_form() → FormResult
    analyze_sorm() → SormResult
    analyze_monte_carlo() → MonteCarloResult

Result classes:
    FormResult: First Order Reliability Method results
    SormResult: Second Order Reliability Method results
    MonteCarloResult: Monte Carlo simulation results

Utilities:
    has_pystra() → bool: Check if pystra is available

All units are SI (meters, kPa, kN, degrees) to match project conventions.

Optional dependency: pystra
Install via: pip install pystra
"""

from pystra_agent.reliability import analyze_form, analyze_sorm, analyze_monte_carlo
from pystra_agent.results import FormResult, SormResult, MonteCarloResult
from pystra_agent.pystra_utils import has_pystra

__all__ = [
    "analyze_form",
    "analyze_sorm",
    "analyze_monte_carlo",
    "FormResult",
    "SormResult",
    "MonteCarloResult",
    "has_pystra",
]
