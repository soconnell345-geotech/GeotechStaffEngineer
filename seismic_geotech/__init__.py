"""
Seismic Geotechnical Analysis Module

Implements common seismic geotechnical evaluations:
- AASHTO/NEHRP site classification (Vs30, N-bar, su-bar)
- Mononobe-Okabe seismic earth pressures
- Simplified SPT liquefaction triggering — NCEER / Youd et al. (2001)
  (the updated Seed-Idriss simplified procedure). This is NOT Boulanger &
  Idriss (2014); for B&I-2014 triggering use ``liquepy_agent``.
- Post-liquefaction residual strength

References:
    AASHTO LRFD Bridge Design Specifications, Section 3 and 11
    FHWA GEC-3 (FHWA-NHI-11-032)
    Youd et al. (2001), ASCE JGGE (NCEER/NSF workshop)
    Seed & Harder (1990); Idriss & Boulanger (2008) — residual strength
"""

from seismic_geotech.site_class import classify_site, site_coefficients
from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure,
)
from seismic_geotech.liquefaction import evaluate_liquefaction
from seismic_geotech.residual_strength import post_liquefaction_strength
from seismic_geotech.results import (
    SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
)

__all__ = [
    'classify_site', 'site_coefficients',
    'mononobe_okabe_KAE', 'mononobe_okabe_KPE', 'seismic_earth_pressure',
    'evaluate_liquefaction', 'post_liquefaction_strength',
    'SiteClassResult', 'SeismicEarthPressureResult', 'LiquefactionResult',
]
