"""
OpenSees Agent Module

High-level workflow wrappers for OpenSees finite element analyses
targeting geotechnical earthquake engineering problems.

Current analyses:
  - PM4Sand undrained cyclic DSS (liquefaction triggering)
  - BNWF laterally-loaded pile (PySimple1/TzSimple1/QzSimple1)
  - 1D effective-stress site response (PDMY02/PIMY + Lysmer dashpot)

References:
    - OpenSeesPy: https://openseespydoc.readthedocs.io
    - PM4Sand: Boulanger & Ziotopoulou (2017), UCD/CGM-17/01
    - BNWF: API RP2A-WSD (2000); Matlock (1970); Reese et al. (1974, 1975)
"""

from opensees_agent.results import (
    PM4SandDSSResult,
    BNWFPileResult,
    SiteResponseResult,
)
from opensees_agent.pm4sand_dss import analyze_pm4sand_dss
from opensees_agent.bnwf_pile import analyze_bnwf_pile
from opensees_agent.site_response import analyze_site_response
from opensees_agent.ground_motions import get_motion, list_motions
from opensees_agent.opensees_utils import has_opensees

__all__ = [
    'analyze_pm4sand_dss',
    'analyze_bnwf_pile',
    'analyze_site_response',
    'PM4SandDSSResult',
    'BNWFPileResult',
    'SiteResponseResult',
    'get_motion',
    'list_motions',
    'has_opensees',
]
