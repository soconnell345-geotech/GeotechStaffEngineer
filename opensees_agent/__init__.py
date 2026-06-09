"""
OpenSees Agent Module

High-level workflow wrappers for OpenSees finite element analyses
targeting geotechnical earthquake engineering problems.

Current analyses:
  - PM4Sand undrained cyclic DSS (liquefaction triggering)
  - 1D effective-stress site response (PDMY02/PIMY + Lysmer dashpot)

References:
    - OpenSeesPy: https://openseespydoc.readthedocs.io
    - PM4Sand: Boulanger & Ziotopoulou (2017), UCD/CGM-17/01
"""

from opensees_agent.results import (
    PM4SandDSSResult,
    SiteResponseResult,
)
from opensees_agent.pm4sand_dss import analyze_pm4sand_dss
from opensees_agent.site_response import analyze_site_response
from opensees_agent.ground_motions import get_motion, list_motions
from opensees_agent.opensees_utils import has_opensees

__all__ = [
    'analyze_pm4sand_dss',
    'analyze_site_response',
    'PM4SandDSSResult',
    'SiteResponseResult',
    'get_motion',
    'list_motions',
    'has_opensees',
]
