"""
First-order second-moment (FOSM / Taylor series) reliability engine.

Duncan (2000) / USACE central-difference procedure: evaluate the
performance function g at the mean values and at +/- one standard deviation
per variable, then

    E[g]      ~ g(mu)
    Var[g]    ~ sum_i (Delta g_i / 2)^2
                + 2 sum_{i<j} rho_ij (Delta g_i / 2)(Delta g_j / 2)

The (Delta g_i/2)^2 terms divided by the uncorrelated variance total give
the per-variable variance contributions — Duncan's "which variable matters"
table.

References
----------
Duncan, J.M. (2000). J. Geotech. Geoenviron. Eng., 126(4), 307-316.
UFC 3-220-20 (2025), ch. 7, Eqs. 7-8, 7-10, 7-11 (pp. 571-573).
USACE ETL 1110-2-547 (1995). Introduction to Probability and Reliability
    Methods for Use in Geotechnical Engineering.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

from reliability.results import FOSMResult
from reliability.stats import beta_lognormal, pf_from_beta
from reliability.variables import build_correlation, variables_from_spec

_CONVENTIONS = ("fos", "margin")


def _threshold(convention: str) -> float:
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_CONVENTIONS}, got '{convention}'.")
    return 1.0 if convention == "fos" else 0.0


def fosm(g: Callable[[Dict[str, float]], float],
         variables,
         correlation=None,
         convention: str = "fos") -> FOSMResult:
    """FOSM (Taylor-series) reliability analysis of ``g``.

    Parameters
    ----------
    g : callable
        Performance function ``g(values: dict[name, float]) -> float``.
        A factor of safety (``convention="fos"``, failure at g < 1) or a
        margin R - S (``convention="margin"``, failure at g < 0).
    variables : dict or list of RandomVariable
        ``{"phi": {"mean": 33, "cov": 0.1, "dist": "lognormal"}, ...}``.
        FOSM uses only the first two moments — the distribution shape
        affects only the choice of reported index (normal vs lognormal).
    correlation : optional
        Pairwise dict ``{("a","b"): rho}`` or full matrix.
    convention : str
        "fos" (default) or "margin".

    Returns
    -------
    FOSMResult
        Moments of g, beta (normal + lognormal), pf, per-variable variance
        contributions.

    Notes
    -----
    2n+1 evaluations of g. The +/-1-sigma central difference (rather than a
    small numerical step) is intentional and standard geotechnical practice
    (Duncan 2000): it captures secant behavior over the plausible range of
    each variable.
    """
    thr = _threshold(convention)
    rvs = variables_from_spec(variables)
    R = build_correlation(rvs, correlation)
    correlated = correlation is not None

    means = {v.name: v.mean for v in rvs}
    g_mu = float(g(dict(means)))
    n_calls = 1

    half_deltas = []
    deltas: Dict[str, float] = {}
    for v in rvs:
        up = dict(means)
        dn = dict(means)
        up[v.name] = v.mean + v.std
        dn[v.name] = v.mean - v.std
        g_up = float(g(up))
        g_dn = float(g(dn))
        n_calls += 2
        dg = g_up - g_dn
        deltas[v.name] = dg
        half_deltas.append(dg / 2.0)

    var_diag = sum(h * h for h in half_deltas)
    var_g = var_diag
    if correlated:
        k = len(rvs)
        for i in range(k):
            for j in range(i + 1, k):
                var_g += 2.0 * R[i, j] * half_deltas[i] * half_deltas[j]
    if var_g < 0:
        raise ValueError(
            "FOSM variance is negative — the correlation structure is "
            "inconsistent with the variable sensitivities. Check signs of "
            "the correlation coefficients.")
    sigma_g = math.sqrt(var_g)

    contributions = {}
    for v, h in zip(rvs, half_deltas):
        contributions[v.name] = (100.0 * h * h / var_diag
                                 if var_diag > 0 else 0.0)

    if sigma_g > 0:
        b_n = (g_mu - thr) / sigma_g
        cov_g = sigma_g / g_mu if g_mu != 0 else float("inf")
    else:
        b_n = math.inf if g_mu > thr else -math.inf
        cov_g = 0.0

    b_ln = pf_ln = None
    if convention == "fos" and g_mu > 0 and sigma_g > 0:
        b_ln = beta_lognormal(g_mu, sigma_g / g_mu)
        pf_ln = pf_from_beta(b_ln)

    return FOSMResult(
        g_mean=g_mu, g_std=sigma_g, g_cov=cov_g,
        beta_normal=b_n, beta_lognormal=b_ln,
        pf_normal=pf_from_beta(b_n) if math.isfinite(b_n)
        else (0.0 if b_n > 0 else 1.0),
        pf_lognormal=pf_ln,
        convention=convention,
        variable_deltas=deltas,
        variance_contributions_pct=contributions,
        correlated=correlated,
        n_g_calls=n_calls,
    )
