"""
Seismic Geotechnical Analysis Module

Implements common seismic geotechnical evaluations:
- AASHTO site classification (Vs30, N-bar, su-bar), as FHWA GEC-12 Vol I
  Tables 7-26 to 7-29 print it
- Mononobe-Okabe seismic earth pressures
- Simplified SPT liquefaction triggering — NCEER / Youd et al. (2001)
  (the updated Seed-Idriss simplified procedure). This is NOT Boulanger &
  Idriss (2014); for B&I-2014 triggering use ``liquepy_agent``.
- Post-liquefaction residual strength

Checked against the print held in GM-293 (module_work/gm293/CHECK.md).

References:
    FHWA GEC-12 Vol I (FHWA-NHI-16-009) Sec. 7.4.2 — site class, site
    factors, spectrum, residual strength (Fig. 7-78)
    UFC 3-220-20 (16 January 2025) Sec. 4-4.4.1; FHWA GEC-4 Eqs. 40 to 46
    FHWA GEC-13 Vol I (FHWA-NHI-16-027) Eq. 5-9 and its r_d
    FHWA GEC-5 (FHWA-NHI-16-072) Sec. 8.6 — Gmax
    AASHTO LRFD Bridge Design Specifications, Section 3 and 11
    Youd et al. (2001), ASCE JGGE (NCEER/NSF workshop)
    Seed & Harder (1990); Idriss & Boulanger (2007) — residual strength
"""

from seismic_geotech.site_class import classify_site, site_coefficients
from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure,
)
from seismic_geotech.liquefaction import evaluate_liquefaction
from seismic_geotech.residual_strength import post_liquefaction_strength
from seismic_geotech.dynamic_properties import (
    gmax_from_vs, gmax_cpt_sand_rix_stokoe, gmax_cpt_clay_mayne_rix,
    gmax_hardin_black, gmax_clay_andersen, modulus_reduction_ishibashi_zhang,
)
from seismic_geotech.results import (
    SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
)

__all__ = [
    'classify_site', 'site_coefficients',
    'mononobe_okabe_KAE', 'mononobe_okabe_KPE', 'seismic_earth_pressure',
    'evaluate_liquefaction', 'post_liquefaction_strength',
    'SiteClassResult', 'SeismicEarthPressureResult', 'LiquefactionResult',
]
