"""
Spatial averaging — Vanmarcke variance reduction.

A soil property averaged over a length L (e.g. along a failure surface or a
pile shaft) fluctuates less than the point property. Vanmarcke's variance
function Gamma^2(L) gives the reduction:

    Var[mean over L] = Gamma^2(L) * Var[point]

Models
------
``exponential`` (default) — exact variance function for the exponential
(Markov) autocorrelation rho(tau) = exp(-2|tau|/delta), where delta is the
scale of fluctuation:

    Gamma^2(L) = 2 (delta/(2L))^2 [ 2L/delta - 1 + exp(-2L/delta) ]

``simple`` — Vanmarcke's wide-band approximation:

    Gamma^2(L) = 1            for L <= delta
               = delta / L    for L >  delta

Both satisfy Gamma^2 -> 1 as L -> 0 and Gamma^2 -> delta/L as L -> inf.

How it plugs into a RandomVariable: reduce only the INHERENT (spatially
variable) part of the COV —

    cov_avg = combined_cov(cov_inherent, cov_measurement,
                           cov_transformation,
                           variance_reduction=Gamma^2)

then build the RandomVariable with the reduced cov. Systematic components
(measurement bias, transformation model error) do NOT average out.

For full 2-D/3-D random-field simulation (kriging, conditional fields,
random-field FEM) use the ``gstools_agent`` module — out of scope here.

References
----------
Vanmarcke, E.H. (1977). "Probabilistic modeling of soil profiles."
    J. Geotech. Eng. Div. ASCE, 103(GT11), 1227-1246.
Vanmarcke, E.H. (1983). Random Fields: Analysis and Synthesis. MIT Press.
Cami, B., Javankhoshdel, S., Phoon, K.K. & Ching, J. (2020). "Scale of
    fluctuation for spatially varying soils: estimation methods and values."
    ASCE-ASME J. Risk Uncertainty Eng. Syst. A, 6(4), 03120002 (values
    reproduced in ISSMGE-TC304 2021, Table 3.1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

_MODELS = ("exponential", "simple")


def variance_reduction(L: float, delta: float,
                       model: str = "exponential") -> float:
    """Vanmarcke variance reduction factor Gamma^2(L/delta).

    Parameters
    ----------
    L : float
        Averaging length (same units as delta). L=0 returns 1 (point value).
    delta : float
        Scale of fluctuation (autocorrelation length measure). Must be > 0.
    model : str
        "exponential" (exact, Markov ACF — default) or "simple"
        (Vanmarcke's delta/L approximation).

    Returns
    -------
    float
        Gamma^2 in (0, 1].
    """
    if delta <= 0:
        raise ValueError("delta (scale of fluctuation) must be positive.")
    if L < 0:
        raise ValueError("L must be non-negative.")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got '{model}'.")
    if L == 0:
        return 1.0
    if model == "simple":
        return min(1.0, delta / L)
    # exponential ACF, a = delta/2
    x = 2.0 * L / delta
    if x < 1e-6:
        return 1.0  # series limit
    return 2.0 / (x * x) * (x - 1.0 + math.exp(-x))


def averaged_std(std: float, L: float, delta: float,
                 model: str = "exponential") -> float:
    """Standard deviation of the spatial average over L: std * Gamma."""
    if std < 0:
        raise ValueError("std must be non-negative.")
    return std * math.sqrt(variance_reduction(L, delta, model))


def averaged_cov(cov_inherent: float, L: float, delta: float,
                 cov_measurement: float = 0.0,
                 cov_transformation: float = 0.0,
                 model: str = "exponential") -> float:
    """Total COV of a parameter averaged over length L.

    Applies Gamma^2 to the inherent component only, then combines with the
    systematic components (UFC 3-220-20 Eq. 7-5 with spatial averaging):

        cov_avg = sqrt( Gamma^2 cov_w^2 + cov_e^2 + cov_t^2 )
    """
    from reliability.stats import combined_cov
    g2 = variance_reduction(L, delta, model)
    return combined_cov(cov_inherent, cov_measurement, cov_transformation,
                        variance_reduction=g2)


# ---------------------------------------------------------------------------
# Published scale-of-fluctuation guidance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FluctuationScaleEntry:
    """Published scale-of-fluctuation range (metres)."""
    soil_type: str
    delta_v_min: Optional[float]
    delta_v_max: Optional[float]
    delta_v_avg: Optional[float]
    delta_h_min: Optional[float]
    delta_h_max: Optional[float]
    delta_h_avg: Optional[float]
    source: str

    def to_dict(self) -> Dict:
        return asdict(self)


_CAMI = ("Cami et al. (2020), reproduced in ISSMGE-TC304 (2021) "
         "Table 3.1")

SCALE_OF_FLUCTUATION: List[FluctuationScaleEntry] = [
    FluctuationScaleEntry("alluvial", 0.07, 2.53, 0.66, 1.1, 49.0, 14.8,
                          _CAMI),
    FluctuationScaleEntry("clay", 0.06, 12.7, 2.47, 0.14, 92.4, 24.43,
                          _CAMI),
    FluctuationScaleEntry("clay_sand_silt_mix", 0.07, 21.0, 1.65, 1.0,
                          1546.0, 152.38, _CAMI),
    FluctuationScaleEntry("marine_clay", 0.11, 6.0, 1.85, 2.0, 60.0, 31.3,
                          _CAMI),
    FluctuationScaleEntry("marine_sand", 0.08, 7.2, 1.77, 55.0, 55.0, 55.0,
                          _CAMI),
    FluctuationScaleEntry("offshore_soil", 0.05, 9.1, 2.37, 14.0, 67.0,
                          34.71, _CAMI),
    FluctuationScaleEntry("sand", 0.1, 4.0, 1.14, 1.7, 75.0, 11.29, _CAMI),
    FluctuationScaleEntry("sensitive_clay", 2.0, 4.0, 3.0, 30.0, 46.0,
                          38.0, _CAMI),
    FluctuationScaleEntry("silt", 0.14, 7.19, 2.08, 12.7, 45.5, 33.22,
                          _CAMI),
    FluctuationScaleEntry("silty_clay", 0.095, 6.47, 1.58, 5.0, 45.4,
                          30.26, _CAMI),
    FluctuationScaleEntry("soft_clay", 0.14, 6.0, 1.76, 22.1, 80.0, 41.1,
                          _CAMI),
]

#: Typical anisotropy of the scale of fluctuation (ISSMGE-TC304 2021,
#: sec. 3.3): delta_h/delta_v ranges ~3-500; most typical ~10-20.
ANISOTROPY_TYPICAL = (10.0, 20.0)
ANISOTROPY_RANGE = (3.0, 500.0)


def scale_of_fluctuation_guidance(
        soil_type: Optional[str] = None) -> List[FluctuationScaleEntry]:
    """Published vertical/horizontal scale-of-fluctuation ranges (m).

    Parameters
    ----------
    soil_type : str, optional
        Filter (substring match), e.g. 'clay', 'sand', 'silt'. None
        returns all rows.
    """
    if soil_type is None:
        return list(SCALE_OF_FLUCTUATION)
    s = soil_type.lower().replace(" ", "_")
    out = [e for e in SCALE_OF_FLUCTUATION if s in e.soil_type]
    if not out:
        known = [e.soil_type for e in SCALE_OF_FLUCTUATION]
        raise ValueError(
            f"No scale-of-fluctuation guidance for '{soil_type}'. "
            f"Known soil types: {known}.")
    return out
