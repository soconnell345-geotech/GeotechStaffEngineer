"""
Monte Carlo simulation engine.

- numpy ``Generator`` sampling with reproducible ``seed``
- optional Latin Hypercube sampling (``scipy.stats.qmc``) for variance
  reduction
- correlation via Cholesky factorization of the correlation matrix in
  standard-normal space, then marginal transform x_i = F_i^-1(PHI(z_i))
  ("Nataf-lite": exact for normal marginals; for non-normal marginals the
  realized product-moment correlation differs slightly from the target —
  adequate for the moderate |rho| typical of soil properties)
- empirical pf with a 95% binomial confidence interval and a cumulative
  convergence trace

References
----------
UFC 3-220-20 (2025), ch. 7, sec. 7-4.2.4 (pp. 578-579).
Baecher, G.B. & Christian, J.T. (2003). Reliability and Statistics in
    Geotechnical Engineering. Wiley.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import stats as sps

from reliability.fosm import _threshold
from reliability.results import MonteCarloResult
from reliability.variables import build_correlation, variables_from_spec

_SAMPLING = ("random", "lhs")


def monte_carlo(g: Callable[[Dict[str, float]], float],
                variables,
                n: int = 10_000,
                seed: Optional[int] = None,
                sampling: str = "random",
                correlation=None,
                convention: str = "fos",
                n_bins: int = 30,
                keep_samples: bool = False,
                percentiles: Sequence[float] = (1, 5, 10, 50, 90, 95, 99),
                ) -> MonteCarloResult:
    """Monte Carlo reliability analysis of ``g``.

    Parameters
    ----------
    g, variables, correlation, convention
        As for :func:`reliability.fosm.fosm`.
    n : int
        Number of realizations (default 10,000). For small pf, aim for
        n >= 10/pf (the CI in the result tells you if you are short).
    seed : int, optional
        Reproducible RNG seed.
    sampling : str
        "random" (default) or "lhs" (Latin Hypercube via scipy.stats.qmc).
    n_bins : int
        Histogram bins for the g-distribution. Default 30.
    keep_samples : bool
        Store all g samples on the result. Default False.

    Returns
    -------
    MonteCarloResult
        Empirical pf + 95% CI, g-distribution statistics, histogram,
        convergence trace, moment and lognormal-fit reliability indices.
    """
    thr = _threshold(convention)
    if sampling not in _SAMPLING:
        raise ValueError(
            f"sampling must be one of {_SAMPLING}, got '{sampling}'.")
    if n < 2:
        raise ValueError("n must be at least 2.")
    rvs = variables_from_spec(variables)
    R = build_correlation(rvs, correlation)
    correlated = correlation is not None
    k = len(rvs)
    L = np.linalg.cholesky(R)

    rng = np.random.default_rng(seed)
    if sampling == "lhs":
        sampler = sps.qmc.LatinHypercube(d=k, seed=rng)
        u01 = sampler.random(n)
        z = sps.norm.ppf(np.clip(u01, 1e-12, 1.0 - 1e-12))
    else:
        z = rng.standard_normal((n, k))
    zc = z @ L.T
    u = sps.norm.cdf(zc)
    x = np.empty((n, k))
    for j, v in enumerate(rvs):
        x[:, j] = v.ppf(u[:, j])

    names = [v.name for v in rvs]
    gv = np.empty(n)
    for i in range(n):
        gv[i] = float(g({nm: float(x[i, j])
                         for j, nm in enumerate(names)}))

    finite = np.isfinite(gv)
    n_dropped = int((~finite).sum())
    gv = gv[finite]
    n_ok = gv.size
    if n_ok < 2:
        raise ValueError(
            f"Monte Carlo failed: only {n_ok} finite g evaluations out of "
            f"{n} ({n_dropped} dropped).")

    failed = gv < thr
    n_failed = int(failed.sum())
    pf = n_failed / n_ok
    half = 1.96 * math.sqrt(max(pf * (1.0 - pf), 0.0) / n_ok)
    ci = (max(0.0, pf - half), min(1.0, pf + half))

    # cumulative convergence trace at ~10 checkpoints
    convergence = []
    for frac in np.linspace(0.1, 1.0, 10):
        m = max(1, int(round(frac * n_ok)))
        convergence.append((m, float(failed[:m].sum()) / m))

    mean_g = float(gv.mean())
    std_g = float(gv.std(ddof=1))
    cov_g = std_g / mean_g if mean_g != 0 else float("inf")
    b_n = ((mean_g - thr) / std_g) if std_g > 0 else math.inf

    b_ln = pf_ln = None
    if convention == "fos" and (gv > 0).all():
        ln_g = np.log(gv)
        s_ln = float(ln_g.std(ddof=1))
        if s_ln > 0:
            b_ln = float(ln_g.mean()) / s_ln  # P(F<1) = P(ln F < 0)
            pf_ln = float(sps.norm.cdf(-b_ln))

    counts, edges = np.histogram(gv, bins=n_bins)
    pct = {f"p{p:g}": float(np.percentile(gv, p)) for p in percentiles}

    return MonteCarloResult(
        n=n_ok, n_failed=n_failed, pf=pf, pf_ci95=ci,
        g_mean=mean_g, g_std=std_g, g_cov=cov_g,
        g_min=float(gv.min()), g_max=float(gv.max()),
        g_median=float(np.median(gv)),
        percentiles=pct,
        beta_normal=b_n, beta_lognormal=b_ln, pf_lognormal=pf_ln,
        convention=convention, sampling=sampling, seed=seed,
        correlated=correlated,
        convergence=convergence,
        histogram_bins=[float(e) for e in edges],
        histogram_counts=[int(c) for c in counts],
        samples=[float(v) for v in gv] if keep_samples else None,
    )
