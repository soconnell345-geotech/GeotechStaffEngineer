"""
Descriptive statistics + reliability-index utilities.

Modernized resurrection of the retired DM7Eqs chapter-7 toolkit
(UFC 3-220-20, Foundations and Earth Structures, ch. 7, pp. 546-595) —
sample statistics, combined COV, the N-sigma range rule, normal/lognormal
reliability indices, beta <-> pf conversions, and rates of exceedance.

References
----------
UFC 3-220-20 (16 Jan 2025), ch. 7 (equation numbers cited per function).
Duncan, J.M. (2000). "Factors of safety and reliability in geotechnical
    engineering." J. Geotech. Geoenviron. Eng., 126(4), 307-316.
Phoon, K.K. & Kulhawy, F.H. (1999). Can. Geotech. J., 36(4), 612-624.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Sample statistics (UFC 3-220-20 Table 7-1)
# ---------------------------------------------------------------------------

def sample_mean(x: Sequence[float]) -> float:
    """Arithmetic mean (UFC 3-220-20 Table 7-1)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must contain at least one value.")
    return float(x.mean())


def sample_variance(x: Sequence[float]) -> float:
    """Unbiased (n-1) sample variance (UFC 3-220-20 Table 7-1)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("x must contain at least two values.")
    return float(x.var(ddof=1))


def sample_std(x: Sequence[float]) -> float:
    """Sample standard deviation (UFC 3-220-20 Table 7-1)."""
    return math.sqrt(sample_variance(x))


def sample_cov(x: Sequence[float]) -> float:
    """Coefficient of variation from data, std/mean (Table 7-1)."""
    m = sample_mean(x)
    if m == 0.0:
        raise ValueError("Sample mean is zero; COV is undefined.")
    return sample_std(x) / m


def cov_from_params(std: float, mean: float) -> float:
    """COV from known std and mean (UFC 3-220-20 Table 7-1)."""
    if std < 0.0:
        raise ValueError("std must be non-negative.")
    if mean == 0.0:
        raise ValueError("mean must be non-zero; COV is undefined.")
    return std / mean


# ---------------------------------------------------------------------------
# Judgment-based dispersion estimates
# ---------------------------------------------------------------------------

def std_from_range(hcv: float, lcv: float, n_sigma: float = 6.0) -> float:
    """Std dev from highest/lowest conceivable values (UFC Eq. 7-6).

    sigma = (HCV - LCV) / N. N=6 is the UFC recommendation and Duncan (2000)
    Eq. 5 ("three-sigma rule"). Duncan's paper notes people underestimate the
    conceivable range (often by about a factor of two); the smaller divisors
    (N < 6) that compensate come from the Christian & Baecher discussion and
    Duncan's closure, not the 2000 paper itself (verified in-hand 2026-07-18,
    module_work/wiki_verification/duncan_2000_cov.md).
    """
    if hcv <= lcv:
        raise ValueError("hcv must be greater than lcv.")
    if n_sigma <= 0.0:
        raise ValueError("n_sigma must be positive.")
    return (hcv - lcv) / n_sigma


def combined_cov(cov_inherent: float,
                 cov_measurement: float = 0.0,
                 cov_transformation: float = 0.0,
                 variance_reduction: float = 1.0) -> float:
    """Total parameter COV from component uncertainties (UFC Eq. 7-5;
    Phoon & Kulhawy 1999).

        COV_total^2 = Gamma^2 * COV_w^2 + COV_e^2 + COV_t^2

    where Gamma^2 (``variance_reduction``, default 1 = point property) is
    Vanmarcke's spatial-averaging variance reduction applied to the inherent
    (spatially variable) component only — measurement error and
    transformation uncertainty are systematic and do not average out.
    """
    for nm, v in (("cov_inherent", cov_inherent),
                  ("cov_measurement", cov_measurement),
                  ("cov_transformation", cov_transformation)):
        if v < 0.0:
            raise ValueError(f"{nm} must be non-negative.")
    if not 0.0 < variance_reduction <= 1.0:
        raise ValueError("variance_reduction must be in (0, 1].")
    return math.sqrt(variance_reduction * cov_inherent ** 2
                     + cov_measurement ** 2 + cov_transformation ** 2)


# ---------------------------------------------------------------------------
# Reliability indices and probability of failure
# ---------------------------------------------------------------------------

def beta_normal(mu: float, sigma: float, threshold: float = 0.0) -> float:
    """Normal reliability index (UFC Eq. 7-7; this form is not printed in
    Duncan 2000, which works in the lognormal index — see beta_lognormal).

    beta = (mu - threshold) / sigma. Use threshold=0 for a margin
    g = R - S, threshold=1 for a factor of safety.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    return (mu - threshold) / sigma


def beta_lognormal(mu: float, cov: float) -> float:
    """Lognormal reliability index for a FACTOR OF SAFETY (Duncan 2000;
    UFC Eq. 7-12).

        beta_LN = ln( F / sqrt(1 + COV_F^2) ) / sqrt( ln(1 + COV_F^2) )

    Anchor: F=1.5, COV_F=0.17 -> beta_LN = 2.32, pf ~ 1% (Duncan 2000).
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive for the lognormal index.")
    if cov <= 0.0:
        raise ValueError("cov must be positive.")
    ln_term = math.log(1.0 + cov ** 2)
    return (math.log(mu / math.sqrt(1.0 + cov ** 2))) / math.sqrt(ln_term)


def pf_from_beta(beta: float) -> float:
    """Probability of failure pf = PHI(-beta) (UFC sec. 7-4.2)."""
    return float(sps.norm.cdf(-beta))


def beta_from_pf(pf: float) -> float:
    """Reliability index from probability of failure, beta = -PHI^-1(pf)."""
    if not 0.0 < pf < 1.0:
        raise ValueError("pf must be in (0, 1).")
    return float(-sps.norm.ppf(pf))


# ---------------------------------------------------------------------------
# Hazard rates (UFC Eqs. 7-15 / 7-16)
# ---------------------------------------------------------------------------

def rate_of_exceedance(return_period: float) -> float:
    """Annual rate of exceedance lambda = 1/R (UFC Eq. 7-15)."""
    if return_period <= 0.0:
        raise ValueError("return_period must be positive.")
    return 1.0 / return_period


def rate_of_exceedance_from_probability(prob_exceedance: float,
                                        exposure_period: float) -> float:
    """lambda = -ln(1 - P) / T_R (UFC Eq. 7-16).

    Example: 10% in 50 yr -> lambda = 0.002107 (return period ~475 yr).
    """
    if not 0.0 < prob_exceedance < 1.0:
        raise ValueError("prob_exceedance must be in the range (0, 1).")
    if exposure_period <= 0.0:
        raise ValueError("exposure_period must be positive.")
    return -math.log(1.0 - prob_exceedance) / exposure_period
