"""
Rosenblueth point-estimate method (PEM).

Full scheme: evaluate g at all 2^n combinations of mu_i +/- sigma_i with
weights

    P(s1..sn) = 2^-n * (1 + sum_{i<j} s_i s_j rho_ij),   s_i = +/-1

(Rosenblueth 1975 correlated form; reduces to equal weights 2^-n when
uncorrelated — UFC 3-220-20 Eqs. 7-13/7-14). Then

    E[g]   = sum P_k g_k
    Var[g] = sum P_k g_k^2 - E[g]^2

Reduced scheme ("multiplicative", uncorrelated variables only): Rosenblueth's
2n+1-point alternative for functions that behave multiplicatively,

    E[g]       ~ g0 * prod_i ( gbar_i / g0 )
    1 + V_g^2  = prod_i ( 1 + V_i^2 )

with g0 = g(mu), gbar_i = (g_i+ + g_i-)/2 and V_i = (g_i+ - g_i-)/(g_i+ + g_i-).
Use it when n is large enough that 2^n evaluations are impractical.

References
----------
Rosenblueth, E. (1975). "Point estimates for probability moments."
    Proc. Nat. Acad. Sci. USA, 72(10), 3812-3814.
UFC 3-220-20 (2025), ch. 7, Eqs. 7-13/7-14 (p. 574).
Christian, J.T. & Baecher, G.B. (1999). "Point-estimate method as numerical
    quadrature." J. Geotech. Geoenviron. Eng., 125(9), 779-786.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict

from reliability.fosm import _threshold
from reliability.results import PEMResult
from reliability.stats import beta_lognormal, pf_from_beta
from reliability.variables import build_correlation, variables_from_spec

_MAX_FULL_VARS = 20  # 2^20 ~ 1e6 evaluations — hard stop above this


def pem(g: Callable[[Dict[str, float]], float],
        variables,
        correlation=None,
        convention: str = "fos",
        scheme: str = "full") -> PEMResult:
    """Rosenblueth point-estimate reliability analysis of ``g``.

    Parameters
    ----------
    g, variables, correlation, convention
        As for :func:`reliability.fosm.fosm`.
    scheme : str
        "full" (2^n points, supports correlation) or "multiplicative"
        (Rosenblueth's 2n+1-point reduced alternative; uncorrelated only,
        intended for many-variable problems and g > 0).

    Returns
    -------
    PEMResult
    """
    thr = _threshold(convention)
    if scheme not in ("full", "multiplicative"):
        raise ValueError(
            f"scheme must be 'full' or 'multiplicative', got '{scheme}'.")
    rvs = variables_from_spec(variables)
    R = build_correlation(rvs, correlation)
    correlated = correlation is not None
    n = len(rvs)

    if scheme == "multiplicative":
        if correlated:
            raise ValueError(
                "The multiplicative (2n+1) PEM scheme assumes uncorrelated "
                "variables; use scheme='full' for correlated problems.")
        mu_g, var_g, n_points = _pem_multiplicative(g, rvs)
    else:
        if n > _MAX_FULL_VARS:
            raise ValueError(
                f"Full PEM needs 2^{n} g-evaluations for {n} variables. "
                f"Use scheme='multiplicative' (2n+1 points) or Monte Carlo.")
        mu_g, var_g, n_points = _pem_full(g, rvs, R, correlated)

    sigma_g = math.sqrt(max(var_g, 0.0))
    if sigma_g > 0:
        b_n = (mu_g - thr) / sigma_g
        cov_g = sigma_g / mu_g if mu_g != 0 else float("inf")
    else:
        b_n = math.inf if mu_g > thr else -math.inf
        cov_g = 0.0

    b_ln = pf_ln = None
    if convention == "fos" and mu_g > 0 and sigma_g > 0:
        b_ln = beta_lognormal(mu_g, sigma_g / mu_g)
        pf_ln = pf_from_beta(b_ln)

    return PEMResult(
        g_mean=mu_g, g_std=sigma_g, g_cov=cov_g,
        beta_normal=b_n, beta_lognormal=b_ln,
        pf_normal=pf_from_beta(b_n) if math.isfinite(b_n)
        else (0.0 if b_n > 0 else 1.0),
        pf_lognormal=pf_ln,
        convention=convention,
        scheme="full_2n" if scheme == "full" else "multiplicative_2n_plus_1",
        n_points=n_points,
        correlated=correlated,
    )


def _pem_full(g, rvs, R, correlated):
    n = len(rvs)
    base_w = 2.0 ** (-n)
    sum_g = 0.0
    sum_g2 = 0.0
    n_points = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        w = base_w
        if correlated:
            corr_term = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    corr_term += signs[i] * signs[j] * R[i, j]
            w = base_w * (1.0 + corr_term)
            if w < -1e-12:
                raise ValueError(
                    "Rosenblueth weights became negative for this "
                    "correlation matrix — the 2^n PEM weight formula is "
                    "outside its validity range. Reduce |rho| or use "
                    "Monte Carlo / FORM.")
            w = max(w, 0.0)
        values = {v.name: v.mean + s * v.std for v, s in zip(rvs, signs)}
        gv = float(g(values))
        n_points += 1
        sum_g += w * gv
        sum_g2 += w * gv * gv
    return sum_g, sum_g2 - sum_g ** 2, n_points


def _pem_multiplicative(g, rvs):
    means = {v.name: v.mean for v in rvs}
    g0 = float(g(dict(means)))
    n_points = 1
    if g0 == 0.0:
        raise ValueError(
            "Multiplicative PEM requires g(means) != 0. Use scheme='full'.")
    mean_ratio = 1.0
    one_plus_v2 = 1.0
    for v in rvs:
        up = dict(means)
        dn = dict(means)
        up[v.name] = v.mean + v.std
        dn[v.name] = v.mean - v.std
        gp = float(g(up))
        gm = float(g(dn))
        n_points += 2
        s = gp + gm
        if s == 0.0:
            raise ValueError(
                f"Multiplicative PEM breakdown at variable '{v.name}' "
                f"(g+ + g- = 0). Use scheme='full'.")
        gbar = 0.5 * s
        vi = (gp - gm) / s
        mean_ratio *= gbar / g0
        one_plus_v2 *= (1.0 + vi * vi)
    mu_g = g0 * mean_ratio
    var_g = (mu_g ** 2) * (one_plus_v2 - 1.0)
    return mu_g, var_g, n_points
