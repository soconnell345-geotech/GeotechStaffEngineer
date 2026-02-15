"""
Ground Improvement Evaluation Module

Evaluates feasibility and preliminary design of ground improvement
methods as alternatives to deep foundations:
- Aggregate piers / rammed aggregate piers
- Prefabricated vertical drains (wick drains)
- Surcharge preloading (with and without drains)
- Vibro-compaction

References:
    FHWA NHI-06-019/020: Ground Improvement Methods
    FHWA GEC-13: Ground Modification Methods Reference Manual
    Barron (1948), Hansbo (1981) — radial consolidation
    Barksdale & Bachus (1983) — aggregate piers
"""

from ground_improvement.aggregate_piers import analyze_aggregate_piers
from ground_improvement.wick_drains import (
    analyze_wick_drains, design_drain_spacing,
)
from ground_improvement.surcharge import analyze_surcharge_preloading
from ground_improvement.vibro import analyze_vibro_compaction
from ground_improvement.feasibility import evaluate_feasibility
from ground_improvement.results import (
    AggregatePierResult, WickDrainResult, SurchargeResult,
    VibroResult, FeasibilityResult,
)

__all__ = [
    'analyze_aggregate_piers',
    'analyze_wick_drains', 'design_drain_spacing',
    'analyze_surcharge_preloading',
    'analyze_vibro_compaction',
    'evaluate_feasibility',
    'AggregatePierResult', 'WickDrainResult', 'SurchargeResult',
    'VibroResult', 'FeasibilityResult',
]
