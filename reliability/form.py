"""
Native FORM — Hasofer-Lind with Rackwitz-Fiessler non-normal transformation.

Improved HL-RF iteration in independent standard-normal (U) space:

1. Map u -> z = L u (Cholesky of the correlation matrix), then
   x_i = F_i^-1(PHI(z_i)) (exact marginal transform).
2. Linearize the transform via Rackwitz-Fiessler equivalent normals:
   dx_i/dz_i = sigma_eq,i = phi(z_i)/f_i(x_i), so
   grad_u G = L^T (sigma_eq ∘ grad_x G), with grad_x G by central finite
   differences.
3. HL-RF step  u+ = [ (grad·u - G/|grad|·|grad|) ... ] — implemented in the
   standard normalized form with simple step-halving on a merit function
   (|G| + distance penalty) for robustness.

beta = |u*| signed by g(mean): positive when the mean point is on the safe
side. pf = PHI(-beta). alpha = -grad_u G/|grad_u G| at the design point
(alpha_i^2 = fraction of total uncertainty; negative alpha_i means failure is
driven by the variable DECREASING — a resistance variable).

References
----------
Hasofer, A.M. & Lind, N.C. (1974). "Exact and invariant second-moment code
    format." J. Eng. Mech. Div. ASCE, 100(1), 111-121.
Rackwitz, R. & Fiessler, B. (1978). Computers & Structures, 9(5), 489-494.
Baecher & Christian (2003), ch. 15.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np
from scipy import stats as sps

from reliability.fosm import _threshold
from reliability.results import FORMResult
from reliability.variables import build_correlation, variables_from_spec


def form(g: Callable[[Dict[str, float]], float],
         variables,
         correlation=None,
         convention: str = "fos",
         max_iterations: int = 100,
         tol: float = 1e-6,
         fd_step: float = 0.01) -> FORMResult:
    """First-order reliability method (HL-RF) for ``g``.

    Parameters
    ----------
    g, variables, correlation, convention
        As for :func:`reliability.fosm.fosm`. Failure at g < 1 ("fos") or
        g < 0 ("margin").
    max_iterations : int
        HL-RF iteration cap. Default 100.
    tol : float
        Convergence tolerance on |u_k+1 - u_k| and on |G|/scale.
    fd_step : float
        Central-difference step for grad_x G, as a fraction of each
        variable's std. Default 0.01.

    Returns
    -------
    FORMResult
        beta, pf, design point (x* and u*), alpha sensitivity vector.
    """
    thr = _threshold(convention)
    rvs = variables_from_spec(variables)
    R = build_correlation(rvs, correlation)
    correlated = correlation is not None
    k = len(rvs)
    L = np.linalg.cholesky(R)
    names = [v.name for v in rvs]

    calls = [0]

    def G(x_vec: np.ndarray) -> float:
        calls[0] += 1
        return float(g({nm: float(x) for nm, x in zip(names, x_vec)})) - thr

    def x_from_u(u: np.ndarray) -> np.ndarray:
        z = L @ u
        Fz = sps.norm.cdf(z)
        return np.array([float(v.ppf(Fz[i])) for i, v in enumerate(rvs)])

    def sigma_eq_at(x: np.ndarray) -> np.ndarray:
        return np.array([v.equivalent_normal(float(x[i]))[1]
                         for i, v in enumerate(rvs)])

    def grad_x(x: np.ndarray) -> np.ndarray:
        gr = np.empty(k)
        for i, v in enumerate(rvs):
            h = fd_step * v.std
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            gr[i] = (G(xp) - G(xm)) / (2.0 * h)
        return gr

    g_mean_side = G(np.array([v.mean for v in rvs]))
    scale = max(abs(g_mean_side), 1e-8)

    u = np.zeros(k)
    converged = False
    n_iter = 0
    g_u = G(x_from_u(u))

    for n_iter in range(1, max_iterations + 1):
        x = x_from_u(u)
        gx = grad_x(x)
        s_eq = sigma_eq_at(x)
        grad_u = L.T @ (s_eq * gx)
        norm_grad = float(np.linalg.norm(grad_u))
        if norm_grad < 1e-14:
            raise ValueError(
                "FORM: zero gradient of g — the performance function does "
                "not depend on the random variables at the current point.")
        # HL-RF step
        u_new = ((grad_u @ u - g_u) / norm_grad ** 2) * grad_u
        # merit-based step halving for robustness
        lam = 1.0
        m0 = abs(g_u) / scale + 0.0
        for _ in range(8):
            u_try = u + lam * (u_new - u)
            g_try = G(x_from_u(u_try))
            if abs(g_try) / scale <= m0 + 1e-12 or lam <= 1.0 / 128.0:
                break
            lam *= 0.5
        step = float(np.linalg.norm(u_try - u))
        u, g_u = u_try, g_try
        if step < tol and abs(g_u) / scale < max(tol, 1e-5):
            converged = True
            break

    x_star = x_from_u(u)
    gx = grad_x(x_star)
    s_eq = sigma_eq_at(x_star)
    grad_u = L.T @ (s_eq * gx)
    norm_grad = float(np.linalg.norm(grad_u))
    alpha = (-grad_u / norm_grad) if norm_grad > 0 else np.zeros(k)

    beta = float(np.linalg.norm(u))
    if g_mean_side < 0:
        beta = -beta  # mean point already in the failure domain
    pf = float(sps.norm.cdf(-beta))

    return FORMResult(
        beta=beta, pf=pf,
        design_point={nm: float(x) for nm, x in zip(names, x_star)},
        design_point_u=[float(ui) for ui in u],
        alphas={nm: float(a) for nm, a in zip(names, alpha)},
        convention=convention,
        n_iterations=n_iter,
        converged=converged,
        n_g_calls=calls[0],
        g_at_design_point=g_u + thr,
        correlated=correlated,
    )
