"""
liquepy_agent — Boulanger & Idriss (2014) liquefaction triggering analysis.

Wraps the liquepy library for B&I-2014 liquefaction triggering:
- CPT triggering + post-triggering (LPI/LSN/LDI, strain, settlement) via
  liquepy's packaged ``run_bi2014``.
- SPT triggering (per-layer FoS) assembled from liquepy's tested B&I-2014
  building blocks (CRR from (N1)60cs, rd, K_sigma) — liquepy ships no packaged
  SPT triggering object, only field correlations, so this module composes them.
- CPT field correlations (Vs, Dr, su/σv', permeability).

Public API
----------
analyze_cpt_liquefaction : Full CPT-based triggering + post-triggering (B&I 2014)
analyze_spt_liquefaction : SPT-based triggering, per-layer FoS (B&I 2014)
analyze_field_correlations : Vs, Dr, su/σv', permeability from CPT
has_liquepy : Check if liquepy is installed
"""

from liquepy_agent.liquepy_utils import has_liquepy
from liquepy_agent.cpt_liquefaction import analyze_cpt_liquefaction
from liquepy_agent.spt_liquefaction import (
    analyze_spt_liquefaction,
    bi2014_spt_fines_correction,
    bi2014_spt_msf,
)
from liquepy_agent.field_correlations import analyze_field_correlations
from liquepy_agent.results import (
    CPTLiquefactionResult,
    SPTLiquefactionResult,
    FieldCorrelationsResult,
)

__all__ = [
    "analyze_cpt_liquefaction",
    "analyze_spt_liquefaction",
    "bi2014_spt_fines_correction",
    "bi2014_spt_msf",
    "analyze_field_correlations",
    "has_liquepy",
    "CPTLiquefactionResult",
    "SPTLiquefactionResult",
    "FieldCorrelationsResult",
]
