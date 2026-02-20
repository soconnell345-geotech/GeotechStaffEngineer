"""
geolysis_agent — Soil classification, SPT corrections, and bearing capacity.

Wraps the geolysis library (v0.24.1) for USCS/AASHTO classification,
SPT N-value corrections (energy, overburden, dilatancy), and SPT-based
or ultimate bearing capacity analyses.

Public API
----------
classify_uscs : USCS soil classification from index properties
classify_aashto : AASHTO soil classification
correct_spt : Full SPT N-value correction (energy + overburden + dilatancy)
design_n_value : Design N-value from corrected values (weighted/minimum/average)
allowable_bc_spt : Allowable bearing capacity from SPT
ultimate_bc : Ultimate bearing capacity (Vesic/Terzaghi)
has_geolysis : Check if geolysis is installed
"""

from geolysis_agent.geolysis_utils import has_geolysis
from geolysis_agent.classification import classify_uscs, classify_aashto
from geolysis_agent.spt_corrections import correct_spt, design_n_value
from geolysis_agent.bearing import allowable_bc_spt, ultimate_bc
from geolysis_agent.results import (
    ClassificationResult,
    SPTCorrectionResult,
    BearingCapacityResult,
)

__all__ = [
    "classify_uscs",
    "classify_aashto",
    "correct_spt",
    "design_n_value",
    "allowable_bc_spt",
    "ultimate_bc",
    "has_geolysis",
    "ClassificationResult",
    "SPTCorrectionResult",
    "BearingCapacityResult",
]
