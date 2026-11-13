"""
Random-variable model for geotechnical reliability analyses.

A :class:`RandomVariable` is defined by its first two moments (mean and
std/cov) plus a marginal distribution shape. Supported distributions:

- ``normal``
- ``lognormal``  (moments are the ARITHMETIC mean/std of X, not of ln X)
- ``uniform``    (from ``lower``/``upper`` bounds, or symmetric from mean/std)
- ``triangular`` (from ``lower``/``mode``/``upper``, or symmetric from mean/std)

Optional truncation (``lower``/``upper`` for normal/lognormal) renormalizes
the marginal on [lower, upper] — the standard device for keeping sampled
strengths/unit weights physical (UFC 3-220-20 ch. 7 discussion).

References
----------
UFC 3-220-20 (2025). Foundations and Earth Structures, ch. 7.
Phoon, K.K. & Kulhawy, F.H. (1999). "Characterization of geotechnical
    variability." Can. Geotech. J., 36(4), 612-624.
Rackwitz, R. & Fiessler, B. (1978). "Structural reliability under combined
    random load sequences." Computers & Structures, 9(5), 489-494.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats as sps

VALID_DISTS = ("normal", "lognormal", "uniform", "triangular")

_Z_CLIP = 8.0  # clip standard-normal scores in equivalent-normal transform


@dataclass
class RandomVariable:
    """A scalar random variable defined by moments + distribution shape.

    Parameters
    ----------
    name : str
        Variable name (the key used in ``g(values_dict)``).
    mean : float, optional
        Arithmetic mean. Required for normal/lognormal; derived from bounds
        for uniform/triangular when bounds are given.
    std : float, optional
        Standard deviation. Give ``std`` or ``cov`` (not required when
        uniform/triangular bounds are supplied).
    cov : float, optional
        Coefficient of variation (std/mean). Alternative to ``std``.
    dist : str
        One of ``normal | lognormal | uniform | triangular``.
    lower, upper : float, optional
        For uniform/triangular: the distribution support (preferred input).
        For normal/lognormal: optional truncation bounds.
    mode : float, optional
        Triangular only. Defaults to symmetric (mode = mean).

    Notes
    -----
    For truncated normal/lognormal the ``mean``/``std`` attributes remain the
    parameters of the UNDERLYING (untruncated) distribution; truncation only
    renormalizes pdf/cdf/ppf/sampling. Monte Carlo and FORM read the
    truncated distribution. The moment methods (FOSM, PEM) evaluate g at
    mean +/- std and have no step that takes a truncation, so they refuse a
    truncated variable (:func:`refuse_truncated`, GM-322) instead of
    answering for the untruncated one. Bounds that cut off no probability
    (infinite bounds, a lognormal with ``lower=0``), or at most
    :data:`TRUNCATION_IGNORED_MASS`, are not a truncation to them.
    """

    name: str
    mean: Optional[float] = None
    std: Optional[float] = None
    cov: Optional[float] = None
    dist: str = "normal"
    lower: Optional[float] = None
    upper: Optional[float] = None
    mode: Optional[float] = None

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("RandomVariable requires a non-empty string name.")
        if self.dist not in VALID_DISTS:
            raise ValueError(
                f"Variable '{self.name}': dist must be one of {VALID_DISTS}, "
                f"got '{self.dist}'.")

        if self.dist in ("uniform", "triangular") and self.lower is not None \
                and self.upper is not None:
            self._init_from_bounds()
        else:
            self._init_from_moments()

        if self.std is None or self.std <= 0:
            raise ValueError(
                f"Variable '{self.name}': positive std required "
                f"(got {self.std}). Provide 'std' or 'cov' (or bounds for "
                f"uniform/triangular).")
        if self.cov is None:
            self.cov = self.std / self.mean if self.mean != 0 else float("inf")

        if self.dist == "lognormal":
            if self.mean is None or self.mean <= 0:
                raise ValueError(
                    f"Variable '{self.name}': lognormal requires mean > 0.")
            v2 = (self.std / self.mean) ** 2
            self.sigma_ln = math.sqrt(math.log(1.0 + v2))
            self.mu_ln = math.log(self.mean) - 0.5 * self.sigma_ln ** 2
        else:
            self.sigma_ln = None
            self.mu_ln = None

        if self.lower is not None and self.upper is not None \
                and self.lower >= self.upper:
            raise ValueError(
                f"Variable '{self.name}': lower ({self.lower}) must be "
                f"< upper ({self.upper}).")
        if self.dist == "lognormal" and self.lower is not None \
                and self.lower < 0:
            raise ValueError(
                f"Variable '{self.name}': lognormal truncation lower bound "
                f"must be >= 0.")

        self._base = self._frozen_base()
        # truncation renormalization constants
        if self._is_truncated():
            a = self.lower if self.lower is not None else -math.inf
            b = self.upper if self.upper is not None else math.inf
            self._Fa = float(self._base.cdf(a)) if a != -math.inf else 0.0
            self._Fb = float(self._base.cdf(b)) if b != math.inf else 1.0
            if self._Fb - self._Fa < 1e-12:
                raise ValueError(
                    f"Variable '{self.name}': truncation interval "
                    f"[{self.lower}, {self.upper}] has ~zero probability "
                    f"mass under the {self.dist} distribution.")
        else:
            self._Fa, self._Fb = 0.0, 1.0

    # -- construction helpers ------------------------------------------------

    def _init_from_bounds(self):
        a, b = float(self.lower), float(self.upper)
        if a >= b:
            raise ValueError(
                f"Variable '{self.name}': lower must be < upper.")
        if self.dist == "uniform":
            mean = 0.5 * (a + b)
            std = (b - a) / math.sqrt(12.0)
        else:  # triangular
            m = float(self.mode) if self.mode is not None else 0.5 * (a + b)
            if not (a <= m <= b):
                raise ValueError(
                    f"Variable '{self.name}': mode must lie within "
                    f"[lower, upper].")
            self.mode = m
            mean = (a + m + b) / 3.0
            std = math.sqrt(
                (a * a + m * m + b * b - a * b - a * m - b * m) / 18.0)
        if self.mean is not None and not math.isclose(
                self.mean, mean, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(
                f"Variable '{self.name}': given mean ({self.mean}) is "
                f"inconsistent with bounds (implied mean {mean:.6g}). "
                f"Give bounds OR mean/std, not conflicting values.")
        if self.std is not None and not math.isclose(
                self.std, std, rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError(
                f"Variable '{self.name}': given std ({self.std}) is "
                f"inconsistent with bounds (implied std {std:.6g}).")
        self.mean, self.std = mean, std

    def _init_from_moments(self):
        if self.mean is None:
            raise ValueError(
                f"Variable '{self.name}': mean is required (or lower/upper "
                f"bounds for uniform/triangular).")
        if self.std is None and self.cov is not None:
            if self.cov <= 0:
                raise ValueError(
                    f"Variable '{self.name}': cov must be positive.")
            self.std = float(self.cov) * abs(float(self.mean))
        if self.std is not None and self.cov is not None and self.mean != 0:
            implied = self.std / abs(self.mean)
            if not math.isclose(implied, self.cov, rel_tol=1e-6):
                raise ValueError(
                    f"Variable '{self.name}': std ({self.std}) and cov "
                    f"({self.cov}) are inconsistent.")
        if self.dist == "uniform":
            half = math.sqrt(3.0) * self.std
            self.lower = self.mean - half
            self.upper = self.mean + half
        elif self.dist == "triangular":
            half = math.sqrt(6.0) * self.std
            self.lower = self.mean - half
            self.upper = self.mean + half
            self.mode = self.mean

    def _frozen_base(self):
        if self.dist == "normal":
            return sps.norm(loc=self.mean, scale=self.std)
        if self.dist == "lognormal":
            return sps.lognorm(s=self.sigma_ln, scale=math.exp(self.mu_ln))
        if self.dist == "uniform":
            return sps.uniform(loc=self.lower, scale=self.upper - self.lower)
        # triangular
        c = (self.mode - self.lower) / (self.upper - self.lower)
        return sps.triang(c=c, loc=self.lower,
                          scale=self.upper - self.lower)

    def _is_truncated(self) -> bool:
        return (self.dist in ("normal", "lognormal")
                and (self.lower is not None or self.upper is not None))

    def _cut_mass(self) -> float:
        """Probability the bounds of a normal/lognormal variable cut off,
        both tails together (0 for no bounds, infinite bounds, or a
        lognormal with lower=0). The upper tail is read from the survival
        function, so a far-tail bound is not lost to rounding."""
        if not self._is_truncated():
            return 0.0
        upper_tail = (float(self._base.sf(self.upper))
                      if self.upper is not None and self.upper != math.inf
                      else 0.0)
        return self._Fa + upper_tail

    # -- distribution interface (truncation-aware) ---------------------------

    def pdf(self, x):
        p = self._base.pdf(x) / (self._Fb - self._Fa)
        if self._is_truncated():
            x_arr = np.asarray(x, dtype=float)
            mask = np.ones_like(x_arr, dtype=bool)
            if self.lower is not None:
                mask &= x_arr >= self.lower
            if self.upper is not None:
                mask &= x_arr <= self.upper
            p = np.where(mask, p, 0.0)
            if np.ndim(x) == 0:
                p = float(p)
        return p

    def cdf(self, x):
        F = (self._base.cdf(x) - self._Fa) / (self._Fb - self._Fa)
        return np.clip(F, 0.0, 1.0)

    def ppf(self, q):
        q = np.clip(q, 0.0, 1.0)
        return self._base.ppf(self._Fa + q * (self._Fb - self._Fa))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` independent samples (truncation-aware)."""
        return np.asarray(self.ppf(rng.uniform(size=n)), dtype=float)

    def equivalent_normal(self, x: float) -> Tuple[float, float]:
        """Rackwitz-Fiessler equivalent normal (mu_eq, sigma_eq) at x.

        Matches the non-normal marginal's CDF and PDF at ``x``:

            sigma_eq = phi(z) / f(x),   z = PHI^-1(F(x))
            mu_eq    = x - sigma_eq * z

        For an (untruncated) normal variable this returns (mean, std).
        """
        if self.dist == "normal" and not self._is_truncated():
            return self.mean, self.std
        F = float(self.cdf(x))
        eps = 1e-12
        F = min(max(F, eps), 1.0 - eps)
        z = float(sps.norm.ppf(F))
        z = max(min(z, _Z_CLIP), -_Z_CLIP)
        f = float(self.pdf(x))
        if f <= 0:
            # fall back to the underlying moments
            return self.mean, self.std
        sigma_eq = float(sps.norm.pdf(z)) / f
        mu_eq = x - sigma_eq * z
        return mu_eq, sigma_eq

    def to_dict(self) -> Dict:
        d = {"name": self.name, "mean": self.mean, "std": self.std,
             "cov": self.cov, "dist": self.dist}
        if self.lower is not None:
            d["lower"] = self.lower
        if self.upper is not None:
            d["upper"] = self.upper
        if self.dist == "triangular":
            d["mode"] = self.mode
        return d


# ---------------------------------------------------------------------------
# Spec parsing + correlation assembly
# ---------------------------------------------------------------------------

#: Probability that lower/upper on one normal/lognormal variable may cut off
#: (both tails together) and still be read by FOSM and PEM as no truncation.
#: The lead's call in the GM-322 review round (module_work/gm322/CHECK.md,
#: "Calls made"): their answer moves by far less than its own error there.
TRUNCATION_IGNORED_MASS = 1e-6


def refuse_truncated(variables: Sequence["RandomVariable"],
                     engine: str) -> List[str]:
    """Refuse truncated normal/lognormal variables in a moment engine.

    FOSM (Duncan 2000; UFC 3-220-20 Eqs. 7-8 to 7-11) and Rosenblueth's
    point-estimate method (UFC 3-220-20 Eqs. 7-13/7-14) evaluate g at
    ``mean +/- std`` only. Neither has a step that reads a truncation, and
    ``mean``/``std`` here are the moments of the UNTRUNCATED distribution,
    so running them would answer for a different variable than the one
    Monte Carlo and FORM read (GM-322). Uniform and triangular bounds are
    the distribution itself (their mean and std follow from the bounds) and
    are accepted.

    Only bounds that cut off probability are refused. Bounds that cut off
    none (infinite bounds, a lognormal with ``lower=0``) are no truncation
    and run silently. Bounds that cut off at most
    :data:`TRUNCATION_IGNORED_MASS` run on the untruncated variable, and
    the returned notes (one per such variable) say how much they cut off;
    the engine puts them in its result's ``warnings``.
    """
    cut = []
    notes: List[str] = []
    for v in variables:
        mass = v._cut_mass()
        if mass > TRUNCATION_IGNORED_MASS:
            cut.append((v, mass))
        elif mass > 0.0:
            notes.append(
                f"'{v.name}': lower/upper cut off {mass:.2g} of the "
                f"{v.dist} distribution's probability, at most "
                f"{TRUNCATION_IGNORED_MASS:g}, which {engine} reads as no "
                f"truncation: the bounds were not applied, and {engine} ran "
                f"on the untruncated variable.")
    if not cut:
        return notes
    listed = ", ".join(
        f"'{v.name}' ({v.dist}, lower={v.lower}, upper={v.upper}, cutting "
        f"off {mass:.2g} of the probability)" for v, mass in cut)
    raise ValueError(
        f"{engine} takes no truncation: it evaluates g at mean +/- std of "
        f"each variable and has no step that reads lower/upper on a normal "
        f"or lognormal variable. Truncated variable(s): {listed}. Use "
        f"monte_carlo or form, which read the truncated distribution, or "
        f"remove lower/upper to run {engine} on the untruncated variable "
        f"(bounds that cut off at most {TRUNCATION_IGNORED_MASS:g} of the "
        f"probability run, untruncated; uniform and triangular bounds are "
        f"accepted: they define the distribution).")


def variables_from_spec(spec: Union[Dict[str, Dict], Sequence[RandomVariable]]
                        ) -> List[RandomVariable]:
    """Normalize a variable spec into a list of RandomVariable.

    Accepts either a list of RandomVariable or the agent-facing dict form::

        {"phi": {"mean": 33, "cov": 0.10, "dist": "lognormal"},
         "gamma": {"mean": 19, "std": 1.0}}
    """
    if isinstance(spec, dict):
        out = []
        for name, d in spec.items():
            if not isinstance(d, dict):
                raise ValueError(
                    f"Variable '{name}': spec must be a dict like "
                    f"{{'mean': 30, 'cov': 0.1, 'dist': 'normal'}}.")
            allowed = {"mean", "std", "cov", "dist", "lower", "upper", "mode"}
            unknown = set(d) - allowed
            if unknown:
                raise ValueError(
                    f"Variable '{name}': unknown spec key(s) "
                    f"{sorted(unknown)}. Allowed: {sorted(allowed)}.")
            out.append(RandomVariable(name=name, **d))
    else:
        out = list(spec)
        for v in out:
            if not isinstance(v, RandomVariable):
                raise ValueError(
                    "variables must be RandomVariable instances or a "
                    "{name: spec} dict.")
    if not out:
        raise ValueError("At least one random variable is required.")
    names = [v.name for v in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate variable names: {names}")
    return out


def build_correlation(variables: Sequence[RandomVariable],
                      correlation=None) -> np.ndarray:
    """Assemble + validate a correlation matrix for ``variables``.

    ``correlation`` may be:

    - None -> identity (independent variables)
    - a full matrix (list of lists / ndarray) in variable order
    - a dict of pairwise entries ``{("phi", "c"): -0.5, ...}`` (order-free;
      string keys "phi,c" also accepted for JSON friendliness)

    The matrix must be symmetric with unit diagonal and positive definite
    (Cholesky check).
    """
    n = len(variables)
    names = [v.name for v in variables]
    if correlation is None:
        return np.eye(n)
    if isinstance(correlation, dict):
        R = np.eye(n)
        idx = {nm: i for i, nm in enumerate(names)}
        for key, rho in correlation.items():
            if isinstance(key, str):
                parts = [p.strip() for p in key.split(",")]
                if len(parts) != 2:
                    raise ValueError(
                        f"Correlation key '{key}' must name two variables, "
                        f"e.g. 'phi,c'.")
                a, b = parts
            else:
                a, b = key
            for nm in (a, b):
                if nm not in idx:
                    raise ValueError(
                        f"Correlation references unknown variable '{nm}'. "
                        f"Variables: {names}")
            if a == b:
                raise ValueError(
                    f"Correlation key ({a},{b}) must reference two "
                    f"DIFFERENT variables.")
            if not -1.0 < float(rho) < 1.0:
                raise ValueError(
                    f"Correlation rho({a},{b})={rho} must be in (-1, 1).")
            R[idx[a], idx[b]] = R[idx[b], idx[a]] = float(rho)
    else:
        R = np.asarray(correlation, dtype=float)
        if R.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {R.shape} does not match the "
                f"{n} variables {names}.")
        if not np.allclose(R, R.T, atol=1e-10):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(R), 1.0, atol=1e-10):
            raise ValueError("Correlation matrix diagonal must be 1.0.")
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Correlation matrix is not positive definite. Check that the "
            "pairwise correlations are jointly consistent.")
    return R
