"""
reliability — statistical variability of subsurface properties +
probabilistic geotechnical analyses.

Engines (FOSM, Rosenblueth PEM, Monte Carlo, native FORM) all drive a
user-supplied callable g(values_dict) -> scalar (FOS or margin).
"""

from reliability.variables import (
    RandomVariable, build_correlation, variables_from_spec,
)
from reliability.stats import (
    sample_mean, sample_variance, sample_std, sample_cov, cov_from_params,
    std_from_range, combined_cov, beta_normal, beta_lognormal,
    pf_from_beta, beta_from_pf,
    rate_of_exceedance, rate_of_exceedance_from_probability,
)
from reliability.results import (
    FOSMResult, PEMResult, MonteCarloResult, FORMResult,
)

__all__ = [
    "RandomVariable", "build_correlation", "variables_from_spec",
    "sample_mean", "sample_variance", "sample_std", "sample_cov",
    "cov_from_params", "std_from_range", "combined_cov",
    "beta_normal", "beta_lognormal", "pf_from_beta", "beta_from_pf",
    "rate_of_exceedance", "rate_of_exceedance_from_probability",
    "FOSMResult", "PEMResult", "MonteCarloResult", "FORMResult",
]
