"""
pyStrata Agent Module

Equivalent-linear and linear elastic 1D site response analysis
using the pystrata library (SHAKE-type frequency-domain approach).

Current analyses:
  - Equivalent-linear site response (Darendeli, Menq, custom curves)
  - Linear elastic site response (constant properties)

References:
    - pystrata: https://github.com/arkottke/pystrata
    - Schnabel, Lysmer & Seed (1972). "SHAKE." EERC 72-12, UC Berkeley.
    - Darendeli (2001). PhD Dissertation, UT Austin.
"""

from pystrata_agent.results import EQLSiteResponseResult
from pystrata_agent.eql_site_response import (
    analyze_eql_site_response,
    analyze_linear_site_response,
)
from pystrata_agent.pystrata_utils import has_pystrata

__all__ = [
    'analyze_eql_site_response',
    'analyze_linear_site_response',
    'EQLSiteResponseResult',
    'has_pystrata',
]
