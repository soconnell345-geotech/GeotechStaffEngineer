"""
Probabilistic slope stability: FOSM (Taylor series) and Monte Carlo.

Treats soil parameters (phi, c_prime, cu, gamma — per layer or global)
as random variables and propagates them through any of the module's LE
methods on a critical surface.

Reliability measures follow Duncan (2000) / USACE practice:

    beta_normal    = (F_MLV - 1) / sigma_F
    beta_lognormal = ln( F_MLV / sqrt(1 + COV_F^2) ) / sqrt( ln(1 + COV_F^2) )
    pf             = Phi(-beta)

The published anchor used for validation: F_MLV = 1.5 with COV_F = 0.17
gives beta_LN = 2.32 and pf ~ 1% (Duncan 2000).

Variable spec format (the agent-facing dict):

    variables = {
        "phi":        {"mean": 30.0, "cov": 0.10, "dist": "normal"},
        "cu:soft":    {"mean": 35.0, "cov": 0.25, "dist": "lognormal"},
    }

Keys are parameter names ('phi', 'c_prime', 'cu', 'gamma'), optionally
scoped to one layer with ':layer_name'. 'std' may be given instead of
'cov'. If 'mean' is omitted the current layer value is used.

References
----------
Duncan, J.M. (2000). "Factors of safety and reliability in geotechnical
    engineering." J. Geotech. Geoenviron. Eng., 126(4), 307-316.
USACE (1999). ETL 1110-2-556, Risk-based analysis in geotechnical
    engineering for support of planning studies.
Baecher & Christian (2003). Reliability and Statistics in Geotechnical
    Engineering.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface

_VALID_PARAMS = ("phi", "c_prime", "cu", "gamma")


# ---------------------------------------------------------------------------
# Reliability index helpers (Duncan 2000)
# ---------------------------------------------------------------------------

def lognormal_beta(fos_mlv: float, cov_f: float) -> float:
    """Lognormal reliability index beta_LN (Duncan 2000, Eq. 8)."""
    if fos_mlv <= 0 or cov_f <= 0:
        return float("inf") if fos_mlv > 1 else float("-inf")
    return (math.log(fos_mlv / math.sqrt(1.0 + cov_f ** 2))
            / math.sqrt(math.log(1.0 + cov_f ** 2)))


def normal_beta(fos_mlv: float, sigma_f: float) -> float:
    """Normal reliability index beta = (F - 1)/sigma_F."""
    if sigma_f <= 0:
        return float("inf") if fos_mlv > 1 else float("-inf")
    return (fos_mlv - 1.0) / sigma_f


def _phi_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Variable parsing / application
# ---------------------------------------------------------------------------

def _parse_variables(geom: SlopeGeometry, variables: Dict[str, Dict]) -> list:
    """Normalize the variable spec into
    [(key, param, layer_names, mean, std, dist), ...]."""
    out = []
    layer_names = [L.name for L in geom.soil_layers]
    for key, spec in variables.items():
        if ":" in key:
            param, lname = key.split(":", 1)
            if lname not in layer_names:
                raise ValueError(
                    f"Unknown layer '{lname}' in variable '{key}'. "
                    f"Layers: {layer_names}")
            targets = [lname]
        else:
            param = key
            targets = None  # all layers carrying the parameter
        if param not in _VALID_PARAMS:
            raise ValueError(
                f"Unknown parameter '{param}' in variable '{key}'. "
                f"Valid: {_VALID_PARAMS}")
        dist = spec.get("dist", "normal")
        if dist not in ("normal", "lognormal"):
            raise ValueError(
                f"dist must be 'normal' or 'lognormal', got '{dist}'")
        mean = spec.get("mean")
        if mean is None:
            # take from the first targeted layer
            for L in geom.soil_layers:
                if targets is None or L.name in targets:
                    mean = getattr(L, param)
                    break
        if mean is None or mean <= 0:
            raise ValueError(
                f"Variable '{key}': positive 'mean' required "
                f"(got {mean}); set it explicitly or on the layer.")
        if "std" in spec:
            std = float(spec["std"])
        elif "cov" in spec:
            std = float(spec["cov"]) * mean
        else:
            raise ValueError(f"Variable '{key}': provide 'cov' or 'std'.")
        if std <= 0:
            raise ValueError(f"Variable '{key}': std must be positive.")
        out.append((key, param, targets, float(mean), std, dist))
    if not out:
        raise ValueError("variables dict is empty")
    return out


def _apply_values(geom: SlopeGeometry, parsed: list,
                  values: List[float]) -> SlopeGeometry:
    """Return a deep-copied geometry with variable values applied."""
    g = copy.deepcopy(geom)
    for (key, param, targets, mean, std, dist), v in zip(parsed, values):
        v = max(v, 1e-6)  # physical floor
        for L in g.soil_layers:
            if targets is None or L.name in targets:
                setattr(L, param, v)
                if param == "gamma" and L.gamma_sat is not None:
                    # keep gamma_sat consistent offset
                    pass
    return g


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FOSMResult:
    """First-order second-moment (Taylor series) reliability result."""
    fos_mlv: float = 0.0
    sigma_f: float = 0.0
    cov_f: float = 0.0
    beta_normal: float = 0.0
    beta_lognormal: float = 0.0
    pf_normal: float = 0.0
    pf_lognormal: float = 0.0
    method: str = ""
    variable_deltas: Dict[str, float] = field(default_factory=dict)
    variable_variance_pct: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  FOSM (TAYLOR SERIES) RELIABILITY",
            "=" * 60,
            f"  Method:            {self.method}",
            f"  FOS (mean values): {self.fos_mlv:.3f}",
            f"  sigma_F:           {self.sigma_f:.4f}",
            f"  COV_F:             {self.cov_f:.3f}",
            f"  beta (normal):     {self.beta_normal:.2f}  "
            f"(pf = {self.pf_normal:.2%})",
            f"  beta (lognormal):  {self.beta_lognormal:.2f}  "
            f"(pf = {self.pf_lognormal:.2%})",
            "  Variance contributions:",
        ]
        for k, pct in self.variable_variance_pct.items():
            lines.append(f"    {k:<16s} {pct:5.1f}%")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "FOS_mean_values": round(self.fos_mlv, 4),
            "sigma_F": round(self.sigma_f, 4),
            "COV_F": round(self.cov_f, 4),
            "beta_normal": round(self.beta_normal, 3),
            "beta_lognormal": round(self.beta_lognormal, 3),
            "pf_normal": round(self.pf_normal, 6),
            "pf_lognormal": round(self.pf_lognormal, 6),
            "method": self.method,
            "variable_deltaF": {k: round(v, 4)
                                for k, v in self.variable_deltas.items()},
            "variable_variance_pct": {
                k: round(v, 1)
                for k, v in self.variable_variance_pct.items()},
        }


@dataclass
class MonteCarloResult:
    """Monte Carlo FOS distribution result."""
    n: int = 0
    n_failed: int = 0
    pf: float = 0.0
    fos_mean: float = 0.0
    fos_std: float = 0.0
    fos_cov: float = 0.0
    fos_min: float = 0.0
    fos_max: float = 0.0
    fos_median: float = 0.0
    beta_lognormal: float = 0.0
    pf_lognormal: float = 0.0
    method: str = ""
    seed: Optional[int] = None
    research_surface: bool = False
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[int] = field(default_factory=list)
    samples: Optional[List[float]] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  MONTE CARLO FOS DISTRIBUTION",
            "=" * 60,
            f"  Method:           {self.method}",
            f"  Realizations:     {self.n}"
            + (f" (surface re-searched)" if self.research_surface else
               " (fixed critical surface)"),
            f"  FOS mean/median:  {self.fos_mean:.3f} / {self.fos_median:.3f}",
            f"  FOS std (COV):    {self.fos_std:.3f} ({self.fos_cov:.3f})",
            f"  FOS range:        {self.fos_min:.3f} - {self.fos_max:.3f}",
            f"  P(FOS < 1):       {self.pf:.2%}  ({self.n_failed}/{self.n})",
            f"  beta_LN (fit):    {self.beta_lognormal:.2f}  "
            f"(pf = {self.pf_lognormal:.2%})",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_realizations": self.n,
            "n_failed": self.n_failed,
            "pf": round(self.pf, 6),
            "FOS_mean": round(self.fos_mean, 4),
            "FOS_std": round(self.fos_std, 4),
            "FOS_cov": round(self.fos_cov, 4),
            "FOS_min": round(self.fos_min, 4),
            "FOS_max": round(self.fos_max, 4),
            "FOS_median": round(self.fos_median, 4),
            "beta_lognormal": round(self.beta_lognormal, 3),
            "pf_lognormal": round(self.pf_lognormal, 6),
            "method": self.method,
            "seed": self.seed,
            "research_surface": self.research_surface,
            "histogram_bins": [round(b, 4) for b in self.histogram_bins],
            "histogram_counts": list(self.histogram_counts),
        }
        return d


# ---------------------------------------------------------------------------
# Deterministic FOS evaluation helper
# ---------------------------------------------------------------------------

def _eval_fos(geom: SlopeGeometry, slip, method: str, n_slices: int,
              tol: float) -> float:
    from slope_stability.search import _compute_fos
    return _compute_fos(geom, slip, method, n_slices, tol=tol)


def _resolve_surface(geom, xc, yc, radius, slip_surface):
    if slip_surface is not None:
        return slip_surface
    if xc is None or yc is None or radius is None:
        raise ValueError("Provide slip_surface or all of xc, yc, radius "
                         "(run search_critical_surface first)")
    return CircularSlipSurface(xc, yc, radius)


# ---------------------------------------------------------------------------
# FOSM
# ---------------------------------------------------------------------------

def fosm_fos(geom: SlopeGeometry,
             variables: Dict[str, Dict],
             xc: float = None, yc: float = None, radius: float = None,
             slip_surface=None,
             method: str = "bishop",
             n_slices: int = 30,
             tol: float = 1e-4) -> FOSMResult:
    """FOSM / Taylor-series reliability of the FOS on a fixed surface.

    Central finite differences at +/- one standard deviation per variable
    (Duncan 2000's procedure):

        sigma_F^2 = sum( (DeltaF_i / 2)^2 )

    Returns both beta_normal and beta_lognormal (+ pf for each).
    """
    parsed = _parse_variables(geom, variables)
    slip = _resolve_surface(geom, xc, yc, radius, slip_surface)

    means = [p[3] for p in parsed]
    f_mlv = _eval_fos(_apply_values(geom, parsed, means), slip, method,
                      n_slices, tol)
    if f_mlv >= 900:
        raise ValueError("FOS evaluation failed at mean values "
                         "(invalid surface?)")

    deltas = {}
    var_sum = 0.0
    for i, p in enumerate(parsed):
        key, _, _, mean, std, _ = p
        up = list(means)
        dn = list(means)
        up[i] = mean + std
        dn[i] = max(mean - std, 1e-6)
        f_up = _eval_fos(_apply_values(geom, parsed, up), slip, method,
                         n_slices, tol)
        f_dn = _eval_fos(_apply_values(geom, parsed, dn), slip, method,
                         n_slices, tol)
        dF = (f_up - f_dn)
        deltas[key] = dF
        var_sum += (dF / 2.0) ** 2

    sigma_f = math.sqrt(var_sum)
    cov_f = sigma_f / f_mlv if f_mlv > 0 else float("inf")
    b_n = normal_beta(f_mlv, sigma_f)
    b_ln = lognormal_beta(f_mlv, cov_f)

    pct = {}
    for k, dF in deltas.items():
        pct[k] = 100.0 * ((dF / 2.0) ** 2) / var_sum if var_sum > 0 else 0.0

    return FOSMResult(
        fos_mlv=f_mlv, sigma_f=sigma_f, cov_f=cov_f,
        beta_normal=b_n, beta_lognormal=b_ln,
        pf_normal=_phi_cdf(-b_n), pf_lognormal=_phi_cdf(-b_ln),
        method=method, variable_deltas=deltas, variable_variance_pct=pct,
    )


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def monte_carlo_fos(geom: SlopeGeometry,
                    variables: Dict[str, Dict],
                    xc: float = None, yc: float = None, radius: float = None,
                    slip_surface=None,
                    method: str = "bishop",
                    n: int = 1000,
                    seed: Optional[int] = None,
                    n_slices: int = 30,
                    tol: float = 1e-4,
                    research_surface: bool = False,
                    search_kwargs: Optional[Dict] = None,
                    n_bins: int = 25,
                    keep_samples: bool = True) -> MonteCarloResult:
    """Monte Carlo FOS distribution.

    By default the critical surface is FIXED (fast, standard practice —
    the surface from a deterministic search). With
    ``research_surface=True`` each realization re-runs
    search_critical_surface (expensive; use small n) so the critical
    surface can move with the realized parameters.

    Parameters
    ----------
    variables : dict
        See module docstring for the spec format.
    n : int
        Number of realizations. Default 1000.
    seed : int, optional
        RNG seed for reproducibility.
    research_surface : bool
        Re-search the critical surface per realization. Default False.
    search_kwargs : dict, optional
        Passed to search_critical_surface when re-searching.
    n_bins : int
        Histogram bins. Default 25.
    keep_samples : bool
        Store the FOS samples on the result. Default True.

    Returns
    -------
    MonteCarloResult
        pf (= fraction of FOS < 1), distribution stats, lognormal-fit
        beta, histogram data.
    """
    import numpy as np

    parsed = _parse_variables(geom, variables)
    if not research_surface:
        slip = _resolve_surface(geom, xc, yc, radius, slip_surface)
    else:
        slip = None

    rng = np.random.default_rng(seed)
    k = len(parsed)
    samples_in = np.empty((n, k))
    for j, (key, _, _, mean, std, dist) in enumerate(parsed):
        if dist == "lognormal":
            sln = math.sqrt(math.log(1.0 + (std / mean) ** 2))
            mln = math.log(mean) - 0.5 * sln ** 2
            samples_in[:, j] = rng.lognormal(mln, sln, size=n)
        else:
            samples_in[:, j] = rng.normal(mean, std, size=n)
    samples_in = np.maximum(samples_in, 1e-6)

    fos_vals = []
    n_eval_failed = 0
    for i in range(n):
        g = _apply_values(geom, parsed, list(samples_in[i]))
        if research_surface:
            from slope_stability.analysis import search_critical_surface
            kw = dict(search_kwargs or {})
            kw.setdefault("method", method)
            kw.setdefault("n_slices", n_slices)
            res = search_critical_surface(g, **kw)
            f = res.critical.FOS if res.critical else float("nan")
        else:
            f = _eval_fos(g, slip, method, n_slices, tol)
        if f is None or f >= 900 or math.isnan(f):
            n_eval_failed += 1
            continue
        fos_vals.append(f)

    if not fos_vals:
        raise ValueError("All Monte Carlo realizations failed to evaluate")

    arr = np.asarray(fos_vals)
    n_ok = len(arr)
    n_failed = int(np.sum(arr < 1.0))
    pf = n_failed / n_ok
    mean_f = float(arr.mean())
    std_f = float(arr.std(ddof=1)) if n_ok > 1 else 0.0
    cov_f = std_f / mean_f if mean_f > 0 else float("inf")

    # Lognormal fit of the FOS samples
    ln_arr = np.log(np.maximum(arr, 1e-9))
    mu_ln = float(ln_arr.mean())
    s_ln = float(ln_arr.std(ddof=1)) if n_ok > 1 else 1e-9
    beta_ln = mu_ln / s_ln if s_ln > 0 else float("inf")
    pf_ln = _phi_cdf(-beta_ln)

    counts, edges = np.histogram(arr, bins=n_bins)

    return MonteCarloResult(
        n=n_ok, n_failed=n_failed, pf=pf,
        fos_mean=mean_f, fos_std=std_f, fos_cov=cov_f,
        fos_min=float(arr.min()), fos_max=float(arr.max()),
        fos_median=float(np.median(arr)),
        beta_lognormal=beta_ln, pf_lognormal=pf_ln,
        method=method, seed=seed, research_surface=research_surface,
        histogram_bins=[float(e) for e in edges],
        histogram_counts=[int(c) for c in counts],
        samples=[float(f) for f in arr] if keep_samples else None,
    )
