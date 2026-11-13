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
- realizations whose g is not finite (NaN or infinite) are counted, never
  dropped in silence: the result carries ``n_requested``, ``n_dropped``,
  ``dropped_fraction`` and the bounds on pf if every dropped realization
  had failed or none had; above ``MAX_DROPPED_FRACTION`` no probability is
  reported at all (GM-322)

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

#: The largest share of realizations with a non-finite g for which a
#: probability is still reported. Each dropped realization is one whose
#: outcome is unknown, so n_dropped/n is the width of the band pf can lie
#: in (all of them failures, or none). Up to 5 % that band is reported with
#: the result (``pf_bounds_dropped``) and a warning; above it the samples
#: kept are no longer a sample of the variables as specified but of the
#: part of their range where g happens to evaluate, the band is wider than
#: the 95 % confidence interval the result quotes at any n the tool is used
#: with (for n = 10,000 the half-width is at most 0.01), and the run is
#: refused with the count and the first offending point. The value is this
#: module's choice: no held reference prints a limit.
MAX_DROPPED_FRACTION = 0.05


def _g_or_nan(g, values) -> float:
    """g at one realization; NaN where g has no real value there.

    A division by zero, an overflow, a ``math`` domain error (the log or
    square root of a negative number) and a complex result (a negative
    number to a fractional power) are the arithmetic's way of saying "not
    finite": they count as a dropped realization like a NaN, instead of
    ending the whole run on the first one. Any other exception is a fault
    in g and is raised.
    """
    try:
        value = g(values)
    except ArithmeticError:
        return math.nan
    except ValueError as exc:
        if "math domain error" in str(exc) or "expected a" in str(exc):
            return math.nan
        raise
    if isinstance(value, complex):
        return math.nan
    return float(value)


def _pf_interval(n_failed: int, n: int):
    """95 % interval on pf and the name of the formula used.

    The normal-approximation (Wald) interval pf +/- 1.96 sqrt(pf(1-pf)/n)
    in general. With no failure, or nothing but failures, that interval
    collapses to a single point (0 to 0 reads as certainty), so the
    Clopper-Pearson (binomial) bound is given there instead:
    0 to 1 - 0.025^(1/n) for no failure in n, about 3.7/n.
    """
    pf = n_failed / n
    if n_failed == 0:
        return (0.0, 1.0 - 0.025 ** (1.0 / n)), "clopper_pearson_no_failures"
    if n_failed == n:
        return (0.025 ** (1.0 / n), 1.0), "clopper_pearson_all_failures"
    half = 1.96 * math.sqrt(max(pf * (1.0 - pf), 0.0) / n)
    return (max(0.0, pf - half), min(1.0, pf + half)), "normal_approximation"


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
        A realization whose g is NaN or infinite is counted in
        ``n_dropped`` and left out of the statistics; the run is refused
        when more than ``MAX_DROPPED_FRACTION`` (5 %) are.
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
        ``n`` is the number of realizations KEPT (finite g); ``n_requested``
        the number asked for and ``n_dropped`` the difference.
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
        gv[i] = _g_or_nan(g, {nm: float(x[i, j])
                              for j, nm in enumerate(names)})

    finite = np.isfinite(gv)
    n_dropped = int((~finite).sum())
    dropped_fraction = n_dropped / n
    if n_dropped:
        j = int(np.argmax(~finite))
        first_bad = {nm: float(x[j, c]) for c, nm in enumerate(names)}
        if n - n_dropped < 2 or dropped_fraction > MAX_DROPPED_FRACTION:
            raise ValueError(
                f"Monte Carlo: g was not finite (NaN or infinite) at "
                f"{n_dropped} of {n} realizations "
                f"({100.0 * dropped_fraction:.1f} %), above the "
                f"{100.0 * MAX_DROPPED_FRACTION:.0f} % limit, so no "
                f"probability is reported: the outcome of those "
                f"realizations is unknown and pf could lie anywhere in a "
                f"band that wide. First such point: {first_bad}. Guard the "
                f"performance function (a division by zero, a log or "
                f"square root of a negative value) or bound the variables "
                f"(lower/upper) so that g evaluates over their whole "
                f"range.")
    gv = gv[finite]
    n_ok = gv.size

    failed = gv < thr
    n_failed = int(failed.sum())
    pf = n_failed / n_ok
    ci, ci_method = _pf_interval(n_failed, n_ok)

    run_warnings = []
    pf_bounds = None
    if n_dropped:
        pf_bounds = (n_failed / n, (n_failed + n_dropped) / n)
        run_warnings.append(
            f"g was not finite at {n_dropped} of {n} realizations "
            f"({100.0 * dropped_fraction:.2f} %); they are left out, and "
            f"pf, the interval and the statistics of g are over the "
            f"{n_ok} kept. If none of the dropped had failed pf would be "
            f"{pf_bounds[0]:.4g}; if all had, {pf_bounds[1]:.4g}. First "
            f"such point: {first_bad}.")
    if ci_method != "normal_approximation":
        run_warnings.append(
            ("No realization failed" if n_failed == 0
             else "Every realization failed")
            + f" in {n_ok}: pf = {pf:g} is the count, not a certainty. "
            f"The 95 % interval given is the Clopper-Pearson bound "
            f"({ci[0]:.3g} to {ci[1]:.3g}); the normal approximation "
            f"used otherwise collapses to a point here.")

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
        pf_ci95_method=ci_method,
        n_requested=n, n_dropped=n_dropped,
        dropped_fraction=dropped_fraction,
        pf_bounds_dropped=pf_bounds,
        warnings=run_warnings,
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
