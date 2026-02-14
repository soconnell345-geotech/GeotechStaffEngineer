"""
Seismic Geotechnical Analysis Module

Implements common seismic geotechnical evaluations:
- AASHTO/NEHRP site classification (Vs30, N-bar, su-bar)
- Mononobe-Okabe seismic earth pressures
- Simplified liquefaction triggering (Seed & Idriss / Youd et al.)
- Post-liquefaction residual strength

References:
    AASHTO LRFD Bridge Design Specifications, Section 3 and 11
    FHWA GEC-3 (FHWA-NHI-11-032)
    Youd et al. (2001), ASCE JGGE
    Boulanger & Idriss (2014)
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
