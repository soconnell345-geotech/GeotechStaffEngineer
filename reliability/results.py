"""
Result dataclasses for the reliability engines.

House pattern (cf. slope_stability/probabilistic.py): plain dataclasses with
``summary()`` (formatted text block) and ``to_dict()`` (JSON-safe, rounded).

Conventions
-----------
``convention="fos"``    : g is a factor of safety; failure when g < 1.
                          beta_normal = (mu-1)/sigma; beta_lognormal is the
                          Duncan (2000) lognormal index.
``convention="margin"`` : g is a safety margin (R - S); failure when g < 0.
                          beta_normal = mu/sigma; the lognormal index is not
                          defined (a margin can be negative) and is None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _r(v, nd=6):
    return None if v is None else round(float(v), nd)


def _fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


@dataclass
class FOSMResult:
    """First-order second-moment (Taylor series) result."""
    g_mean: float = 0.0
    g_std: float = 0.0
    g_cov: float = 0.0
    beta_normal: float = 0.0
    beta_lognormal: Optional[float] = None
    pf_normal: float = 0.0
    pf_lognormal: Optional[float] = None
    convention: str = "fos"
    variable_deltas: Dict[str, float] = field(default_factory=dict)
    variance_contributions_pct: Dict[str, float] = field(default_factory=dict)
    correlated: bool = False
    n_g_calls: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  FOSM (TAYLOR SERIES) RELIABILITY",
            "=" * 62,
            f"  Convention:        {self.convention} "
            f"(failure at g < {1 if self.convention == 'fos' else 0})",
            f"  E[g] (mean values): {self.g_mean:.4f}",
            f"  sigma_g:           {self.g_std:.4f}   (COV {self.g_cov:.3f})",
            f"  beta (normal):     {self.beta_normal:.3f}  "
            f"(pf = {self.pf_normal:.3e})",
            f"  beta (lognormal):  {_fmt(self.beta_lognormal)}  "
            f"(pf = {_fmt(self.pf_lognormal, '.3e')})",
            f"  g() evaluations:   {self.n_g_calls}"
            + ("   [correlated]" if self.correlated else ""),
            "  Variance contributions (Duncan 2000 'which variable matters'):",
        ]
        for k, pct in sorted(self.variance_contributions_pct.items(),
                             key=lambda kv: -kv[1]):
            lines.append(f"    {k:<18s} {pct:6.1f}%")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": "fosm",
            "convention": self.convention,
            "g_mean": _r(self.g_mean, 4),
            "g_std": _r(self.g_std, 4),
            "g_cov": _r(self.g_cov, 4),
            "beta_normal": _r(self.beta_normal, 3),
            "beta_lognormal": _r(self.beta_lognormal, 3),
            "pf_normal": _r(self.pf_normal, 8),
            "pf_lognormal": _r(self.pf_lognormal, 8),
            "variable_deltas": {k: _r(v, 4)
                                for k, v in self.variable_deltas.items()},
            "variance_contributions_pct": {
                k: _r(v, 1)
                for k, v in self.variance_contributions_pct.items()},
            "correlated": self.correlated,
            "n_g_calls": self.n_g_calls,
        }


@dataclass
class PEMResult:
    """Rosenblueth point-estimate method result."""
    g_mean: float = 0.0
    g_std: float = 0.0
    g_cov: float = 0.0
    beta_normal: float = 0.0
    beta_lognormal: Optional[float] = None
    pf_normal: float = 0.0
    pf_lognormal: Optional[float] = None
    convention: str = "fos"
    scheme: str = "full_2n"
    n_points: int = 0
    correlated: bool = False

    def summary(self) -> str:
        return "\n".join([
            "=" * 62,
            "  ROSENBLUETH POINT-ESTIMATE METHOD",
            "=" * 62,
            f"  Convention:        {self.convention}",
            f"  Scheme:            {self.scheme}  "
            f"({self.n_points} evaluation points)"
            + ("   [correlated]" if self.correlated else ""),
            f"  E[g]:              {self.g_mean:.4f}",
            f"  sigma_g:           {self.g_std:.4f}   (COV {self.g_cov:.3f})",
            f"  beta (normal):     {self.beta_normal:.3f}  "
            f"(pf = {self.pf_normal:.3e})",
            f"  beta (lognormal):  {_fmt(self.beta_lognormal)}  "
            f"(pf = {_fmt(self.pf_lognormal, '.3e')})",
            "=" * 62,
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": "pem",
            "convention": self.convention,
            "scheme": self.scheme,
            "n_points": self.n_points,
            "g_mean": _r(self.g_mean, 4),
            "g_std": _r(self.g_std, 4),
            "g_cov": _r(self.g_cov, 4),
            "beta_normal": _r(self.beta_normal, 3),
            "beta_lognormal": _r(self.beta_lognormal, 3),
            "pf_normal": _r(self.pf_normal, 8),
            "pf_lognormal": _r(self.pf_lognormal, 8),
            "correlated": self.correlated,
        }


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    n: int = 0
    n_failed: int = 0
    pf: float = 0.0
    pf_ci95: Tuple[float, float] = (0.0, 0.0)
    g_mean: float = 0.0
    g_std: float = 0.0
    g_cov: float = 0.0
    g_min: float = 0.0
    g_max: float = 0.0
    g_median: float = 0.0
    percentiles: Dict[str, float] = field(default_factory=dict)
    beta_normal: float = 0.0
    beta_lognormal: Optional[float] = None
    pf_lognormal: Optional[float] = None
    convention: str = "fos"
    sampling: str = "random"
    seed: Optional[int] = None
    correlated: bool = False
    convergence: List[Tuple[int, float]] = field(default_factory=list)
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[int] = field(default_factory=list)
    samples: Optional[List[float]] = None

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  MONTE CARLO SIMULATION",
            "=" * 62,
            f"  Convention:       {self.convention}",
            f"  Realizations:     {self.n}  (sampling: {self.sampling}"
            f"{', seed ' + str(self.seed) if self.seed is not None else ''})"
            + ("   [correlated]" if self.correlated else ""),
            f"  E[g] / median:    {self.g_mean:.4f} / {self.g_median:.4f}",
            f"  sigma_g (COV):    {self.g_std:.4f} ({self.g_cov:.3f})",
            f"  g range:          {self.g_min:.4f} - {self.g_max:.4f}",
            f"  pf (empirical):   {self.pf:.4e}  "
            f"[95% CI {self.pf_ci95[0]:.3e} - {self.pf_ci95[1]:.3e}]  "
            f"({self.n_failed}/{self.n})",
            f"  beta (moments):   {self.beta_normal:.3f}",
            f"  beta_LN (fit):    {_fmt(self.beta_lognormal)}  "
            f"(pf = {_fmt(self.pf_lognormal, '.3e')})",
            "=" * 62,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": "monte_carlo",
            "convention": self.convention,
            "n": self.n,
            "n_failed": self.n_failed,
            "pf": _r(self.pf, 8),
            "pf_ci95": [_r(self.pf_ci95[0], 8), _r(self.pf_ci95[1], 8)],
            "g_mean": _r(self.g_mean, 4),
            "g_std": _r(self.g_std, 4),
            "g_cov": _r(self.g_cov, 4),
            "g_min": _r(self.g_min, 4),
            "g_max": _r(self.g_max, 4),
            "g_median": _r(self.g_median, 4),
            "percentiles": {k: _r(v, 4) for k, v in self.percentiles.items()},
            "beta_normal": _r(self.beta_normal, 3),
            "beta_lognormal": _r(self.beta_lognormal, 3),
            "pf_lognormal": _r(self.pf_lognormal, 8),
            "sampling": self.sampling,
            "seed": self.seed,
            "correlated": self.correlated,
            "convergence": [[int(n), _r(p, 8)] for n, p in self.convergence],
            "histogram_bins": [_r(b, 4) for b in self.histogram_bins],
            "histogram_counts": [int(c) for c in self.histogram_counts],
        }


@dataclass
class FORMResult:
    """First-order reliability method (HL-RF) result."""
    beta: float = 0.0
    pf: float = 0.0
    design_point: Dict[str, float] = field(default_factory=dict)
    design_point_u: List[float] = field(default_factory=list)
    alphas: Dict[str, float] = field(default_factory=dict)
    convention: str = "fos"
    n_iterations: int = 0
    converged: bool = False
    n_g_calls: int = 0
    g_at_design_point: float = 0.0
    correlated: bool = False

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  FORM (HASOFER-LIND / RACKWITZ-FIESSLER)",
            "=" * 62,
            f"  Convention:       {self.convention}",
            f"  beta:             {self.beta:.4f}",
            f"  pf = PHI(-beta):  {self.pf:.4e}",
            f"  Converged:        {self.converged} "
            f"({self.n_iterations} iterations, {self.n_g_calls} g() calls)"
            + ("   [correlated]" if self.correlated else ""),
            f"  g at x*:          {self.g_at_design_point:.2e} "
            f"(should be ~ {1.0 if self.convention == 'fos' else 0.0})",
            "  Design point x* and sensitivities alpha:",
        ]
        for k in self.design_point:
            lines.append(f"    {k:<18s} x* = {self.design_point[k]:>12.5g}"
                         f"   alpha = {self.alphas.get(k, 0.0):+.4f}")
        lines.append("  (alpha^2 = relative contribution; negative alpha ->")
        lines.append("   failure when the variable DECREASES, e.g. resistance)")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": "form",
            "convention": self.convention,
            "beta": _r(self.beta, 4),
            "pf": _r(self.pf, 10),
            "design_point": {k: _r(v, 6)
                             for k, v in self.design_point.items()},
            "design_point_u": [_r(u, 6) for u in self.design_point_u],
            "alphas": {k: _r(v, 4) for k, v in self.alphas.items()},
            "alpha_squared_pct": {
                k: _r(100.0 * v * v, 1) for k, v in self.alphas.items()},
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "n_g_calls": self.n_g_calls,
            "g_at_design_point": _r(self.g_at_design_point, 6),
            "correlated": self.correlated,
        }
