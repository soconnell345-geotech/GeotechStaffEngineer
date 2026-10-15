"""
Conditional simulation: realizations of a spatial field that honour the
observations, integrated over plan regions.

The question this answers is "what is the probability that a TOTAL over an
area exceeds a value" -- rock above an excavation level under a footprint, fill
below a grade, the area where a property exceeds a limit. Kriging gives one
smoothed best estimate (the mean surface); it understates the spread and
cannot give the probability of a total over an area. An unconditioned random
field does not honour the data. A conditioned random field does both: each
realization is a plausible surface that passes through (or, with a nugget or
measurement error, near) the observations, with the spatial correlation of the
covariance model, and the total is computed on every realization.

Two ways to condition, offered with their basis (``simulation_method``):

``kriging_residual`` (default) -- conditioning by kriging (Journel &
    Huijbregts 1978, *Mining Geostatistics*, ch. V; Chiles & Delfiner 2012,
    *Geostatistics*, s. 7.3.1). An unconditional realization ``S`` is drawn
    with the covariance model, its values at the observations are kriged, and
    the realization is ``S + K(z - S_obs)`` -- the kriged data plus the
    realization's own kriging error. For a Gaussian field this draws the
    conditional distribution the chosen model implies: the right mean (the
    kriged surface), the right pointwise variance (the kriging variance) AND
    the right covariance between points, which is what decides the spread of
    a total. "Right" is relative to the assumed model (Gaussian, its
    covariance, the observations used), never a claim about the ground;
    ``tests/test_conditioning_by_definition.py`` checks it against keeping
    only the unconditioned fields that pass through the observations. The kriging weights come from GSTools' own ``Krige`` (the same
    kriging type); the unconditional field is drawn by factoring the model
    covariance over the cells (Cholesky; Davis 1987), exact for any GSTools
    model and fast for the cell counts allowed here.

``condsrf`` -- GSTools' ``CondSRF`` exactly as GSTools builds it (Mueller,
    Schueler, Zech & Hesse 2022, GMD 15, 3161): the kriged field plus an
    unconditional randomization-method field scaled by the local kriging
    standard deviation (plus the nugget as white noise per cell). It honours
    the data and the pointwise kriging variance, but the correlation between
    two points is that of the UNconditioned field: across an observation the
    realizations stay correlated where the exact conditional field is not
    (for an exponential model, two points either side of a datum: 0.23 against
    0.0 in a 4,000-realization check), so the spread of a total over an area
    can be overstated. Offered to reproduce a GSTools CondSRF calculation.
"""

import math
import time

import numpy as np

from gstools_agent.gstools_utils import import_gstools
from gstools_agent.regions import (
    QUANTITIES, auto_cell_size, build_lattice, integrate, parse_regions,
    pointwise_gaussian, region_geometry, region_masks,
)
from gstools_agent.results import ConditionalSimulationResult

_VALID_MODELS = {
    "Gaussian", "Exponential", "Matern", "Spherical", "Linear",
    "Stable", "Rational", "Cubic", "HyperSpherical",
}
_VALID_KRIGE_TYPES = ("ordinary", "simple", "universal")
SIMULATION_METHODS = ("kriging_residual", "condsrf")
BOUNDS_MODES = ("report", "reject")

#: Most cells (centres inside any region) one run simulates. The covariance
#: factor is n x n doubles: 5,000 cells = 200 MB.
MAX_SIM_CELLS = 5000
#: Most realizations one run keeps.
MAX_REALIZATIONS = 20000
#: The kriged mean / variance file covers the whole lattice up to this size,
#: else only the simulated cells.
MAX_LATTICE_CELLS = 250000
#: ``bounds_mode='reject'`` draws at most this many times n_realizations.
REJECT_ATTEMPT_FACTOR = 10
#: ``condsrf`` refuses a run projected to take longer than this (s).
MAX_PROJECTED_SECONDS = 600.0
#: A non-finite realization is redrawn with seed + this + k (k = 0, 1, ...),
#: clear of every seed + i a run can use.
REDRAW_SEED_OFFSET = 1_000_000_000
MAX_REDRAW_TRIES = 5
#: Seeds are passed to numpy and GSTools (which need 0 <= seed < 2**32).
MAX_SEED = 2**31 - 1

PERCENTILES = (5, 10, 25, 50, 75, 80, 90, 95)
_CHECKPOINTS = (100, 250, 500, 1000, 2000, 5000, 10000, 20000)


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _as_float_array(v, what):
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a list of numbers") from None
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what} holds a non-finite value")
    return a


def _parse_bounds(bounds):
    if bounds is None:
        return []
    if isinstance(bounds, dict):
        bounds = [bounds]
    if not isinstance(bounds, (list, tuple)):
        raise ValueError("bounds must be a list of {x, y, max} and/or "
                         "{x, y, min} dicts")
    out = []
    for i, b in enumerate(bounds):
        if not isinstance(b, dict):
            raise ValueError(f"bounds[{i}] must be a dict with x, y and max "
                             "and/or min")
        unknown = sorted(set(b) - {"x", "y", "max", "min", "name"})
        if unknown:
            raise ValueError(f"bounds[{i}] has unknown key(s) {unknown}; "
                             "valid keys are x, y, max, min, name")
        if b.get("x") is None or b.get("y") is None:
            raise ValueError(f"bounds[{i}] needs x and y")
        if b.get("max") is None and b.get("min") is None:
            raise ValueError(f"bounds[{i}] needs max and/or min (e.g. a "
                             "boring that ended above rock: the rock surface "
                             "is below its bottom elevation -> max)")
        rec = {"name": str(b.get("name") or f"bound_{i + 1}"),
               "x": float(b["x"]), "y": float(b["y"]),
               "max": None if b.get("max") is None else float(b["max"]),
               "min": None if b.get("min") is None else float(b["min"])}
        if (rec["max"] is not None and rec["min"] is not None
                and rec["min"] > rec["max"]):
            raise ValueError(f"bounds[{i}]: min {rec['min']} is above max "
                             f"{rec['max']}")
        for k in ("x", "y", "max", "min"):
            if rec[k] is not None and not math.isfinite(rec[k]):
                raise ValueError(f"bounds[{i}] {k} must be finite")
        out.append(rec)
    return out


def _measurement_error(me, n):
    if me is None:
        return None
    if isinstance(me, (list, tuple, np.ndarray)):
        a = _as_float_array(me, "measurement_error")
        if a.size == 1:
            a = np.full(n, float(a[0]))
        if a.size != n:
            raise ValueError(f"measurement_error has {a.size} values for {n} "
                             "observations: give one variance, or one per "
                             "observation")
    else:
        try:
            a = np.full(n, float(me))
        except (TypeError, ValueError):
            raise ValueError("measurement_error must be a variance (one "
                             "number or one per observation)") from None
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("measurement_error is a variance and must be >= 0")
    return a


# ---------------------------------------------------------------------------
# Covariance model and kriging
# ---------------------------------------------------------------------------

def _make_krige(gs, model, x, y, vals, kriging_type, cond_err, sk_mean):
    kw = {"exact": False, "cond_err": cond_err}
    if kriging_type == "simple":
        return gs.krige.Simple(model, [x, y], vals, mean=sk_mean, **kw)
    if kriging_type == "universal":
        return gs.krige.Universal(model, [x, y], vals,
                                  drift_functions="linear", **kw)
    return gs.krige.Ordinary(model, [x, y], vals, **kw)


def _kriging_weights(gs, model, x, y, kriging_type, cond_err, px, py):
    """Kriging weights (n_points x n_obs) from GSTools' own Krige: kriging
    is linear in the data, so kriging the unit vectors gives the weights."""
    n = len(x)
    W = np.empty((len(px), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        kr = _make_krige(gs, model, x, y, e, kriging_type, cond_err, 0.0)
        W[:, j] = kr([px, py], mesh_type="unstructured", return_var=False,
                     store=False)
    return W


def _covariance_factor(model_c, px, py):
    """Lower factor L with L L^T = C (the continuous-part covariance over the
    points), with the smallest diagonal jitter that factors (a covariance
    with a very smooth model, e.g. Gaussian, on close points is numerically
    singular). Returns (L, jitter_fraction_of_variance)."""
    import scipy.linalg as sla
    from scipy.spatial.distance import cdist

    iso = np.asarray(model_c.isometrize(np.vstack([px, py])), dtype=float)
    pts = iso.T
    n = pts.shape[0]
    C = np.empty((n, n))
    step = 1000
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        C[i0:i1] = model_c.covariance(cdist(pts[i0:i1], pts))
    var = float(model_c.var)
    diag = np.arange(n)
    base = C[diag, diag].copy()
    for frac in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3):
        C[diag, diag] = base + frac * var
        try:
            L = sla.cholesky(C, lower=True, check_finite=False)
            return L, frac
        except np.linalg.LinAlgError:
            continue
    # Last resort: a symmetric eigen-factor with negative eigenvalues cut.
    C[diag, diag] = base
    w, V = np.linalg.eigh(C)
    return V * np.sqrt(np.clip(w, 0.0, None)), float("nan")


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------

def _nice_step(v):
    if v <= 0:
        return 0.0
    k = math.floor(math.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        s = m * 10.0 ** k
        if s >= v * (1 - 1e-12):
            return s
    return 10.0 ** (k + 1)


def _auto_report_values(vals):
    hi = float(np.percentile(vals, 99)) if len(vals) else 0.0
    if hi <= 0:
        return [0.0]
    step = _nice_step(hi / 8.0)
    out = [0.0]
    while out[-1] < hi and len(out) < 12:
        out.append(round(out[-1] + step, 10))
    return out


def _stats(v):
    v = np.asarray(v, dtype=float)
    d = {"mean": float(v.mean()), "std": float(v.std(ddof=1))
         if v.size > 1 else 0.0}
    pct = np.percentile(v, PERCENTILES)
    for p, q in zip(PERCENTILES, pct):
        d[f"P{p}"] = float(q)
    return d


def _exceedance(v, values):
    n = len(v)
    out = []
    for t in values:
        p = float(np.mean(v > t))
        out.append({"value": float(t), "p_exceed": p,
                    "std_error": float(math.sqrt(p * (1 - p) / n))})
    return out


def _convergence(v):
    n = len(v)
    cps = [c for c in _CHECKPOINTS if c < n] + [n]
    return [{"n": int(c), "mean": float(np.mean(v[:c])),
             "P90": float(np.percentile(v[:c], 90))} for c in cps]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_conditional_simulation(
    x,
    y,
    values,
    regions,
    model_type="Gaussian",
    variance=None,
    len_scale=None,
    nugget=None,
    fit_variogram=False,
    kriging_type="ordinary",
    measurement_error=None,
    anis=1.0,
    angle_deg=0.0,
    cell_size=None,
    margin=None,
    quantity="above",
    n_realizations=1000,
    seed=42,
    simulation_method="kriging_residual",
    bounds=None,
    bounds_mode="report",
    report_values=None,
    names=None,
) -> ConditionalSimulationResult:
    """Conditional simulation of a 2D field from point observations, with
    totals (volume, area) over plan regions on every realization.

    Parameters
    ----------
    x, y, values : array-like
        Observation coordinates (m) and values (e.g. top-of-rock elevation,
        m). At least 3.
    regions : list of dict
        ``{name, polygon: [[x, y], ...], level}``; ``level`` one number or a
        list (a sweep). Every region and level is evaluated on the SAME
        realizations (common random numbers).
    model_type : str
        GSTools covariance model. Default 'Gaussian'.
    variance, len_scale : float or None
        Sill variance (value units squared) and correlation length (m) of the
        continuous part. None = the sample variance / one third of the data
        extent, flagged as an assumption, unless ``fit_variogram``.
    nugget : float or None
        Nugget variance. None = 0, or the largest ``measurement_error``.
    fit_variogram : bool
        Fit variance, len_scale (and the nugget when not given) to the
        empirical variogram. Default False: with a handful of observations a
        fitted variogram is unstable.
    kriging_type : str
        'ordinary' (unknown constant mean, default), 'simple' (known mean =
        the sample mean) or 'universal' (linear drift in x and y).
    measurement_error : float, list or None
        Error variance at the observations (GSTools ``cond_err``), one number
        or one per observation; must not exceed the nugget (GSTools). None =
        the nugget is the error (GSTools' default); with nugget 0 too the
        field passes exactly through the data.
    anis, angle_deg : float
        Anisotropy ratio (minor / main correlation length) and the main
        axis' direction, degrees counter-clockwise from +x (GSTools
        ``angles`` in radians).
    cell_size : float or None
        Cell size (m); None = the smallest round size giving at most
        ~3,000 cells in the regions.
    margin : float or None
        Margin (m) of the lattice around the regions (kriged-mean file and
        map). None = max(2 cells, 5 % of the extent).
    quantity : str
        'above' (volume/area of the field above the level, default) or
        'below' (below the level).
    n_realizations : int
        Realizations kept (default 1000, max 20,000).
    seed : int
        Realization i is drawn with seed + i.
    simulation_method : str
        'kriging_residual' (default, exact conditioning by kriging) or
        'condsrf' (GSTools CondSRF as GSTools builds it).
    bounds : list of dict or None
        Inequality data ``{x, y, max}`` / ``{x, y, min}``.
    bounds_mode : str
        'report' (default): fraction of realizations violating each bound.
        'reject': keep only realizations that satisfy every bound (rejection
        sampling, not inequality kriging).
    report_values : list of float or None
        Totals for the exceedance table; None = round values chosen per
        region and level.
    names : list of str or None
        Observation names for the figures and files.
    """
    t_start = time.perf_counter()
    gs = import_gstools()

    xo = _as_float_array(x, "x")
    yo = _as_float_array(y, "y")
    zo = _as_float_array(values, "values")
    n = len(xo)
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}")
    if len(yo) != n or len(zo) != n:
        raise ValueError(f"x, y, values must have the same length: x={n}, "
                         f"y={len(yo)}, values={len(zo)}")
    if model_type not in _VALID_MODELS:
        raise ValueError(f"model_type must be one of {sorted(_VALID_MODELS)}"
                         f", got '{model_type}'")
    if kriging_type not in _VALID_KRIGE_TYPES:
        raise ValueError(f"kriging_type must be one of {_VALID_KRIGE_TYPES}"
                         f", got '{kriging_type}'")
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got "
                         f"'{quantity}'")
    if simulation_method not in SIMULATION_METHODS:
        raise ValueError(f"simulation_method must be one of "
                         f"{SIMULATION_METHODS}, got '{simulation_method}'")
    if bounds_mode not in BOUNDS_MODES:
        raise ValueError(f"bounds_mode must be one of {BOUNDS_MODES}, got "
                         f"'{bounds_mode}'")
    if isinstance(n_realizations, bool):
        raise ValueError("n_realizations must be a whole number")
    try:
        n_real = int(n_realizations)
    except (TypeError, ValueError):
        raise ValueError("n_realizations must be a whole number") from None
    if n_real != float(n_realizations) or not 1 <= n_real <= MAX_REALIZATIONS:
        raise ValueError(f"n_realizations must be a whole number from 1 to "
                         f"{MAX_REALIZATIONS}, got {n_realizations}")
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise ValueError("seed must be an integer") from None
    if not 0 <= seed <= MAX_SEED - REDRAW_SEED_OFFSET:
        raise ValueError(f"seed must be from 0 to "
                         f"{MAX_SEED - REDRAW_SEED_OFFSET:,}")
    anis = float(anis)
    if not anis > 0:
        raise ValueError("anis (minor / main correlation length) must be > 0")
    angle_deg = float(angle_deg)
    if names is not None:
        if isinstance(names, str) or len(names) != n:
            raise ValueError(f"names must be a list with one name per "
                             f"observation ({n})")
        obs_names = [str(s) for s in names]
    else:
        obs_names = [f"obs{i + 1}" for i in range(n)]

    regs = parse_regions(regions)
    blist = _parse_bounds(bounds)
    me = _measurement_error(measurement_error, n)

    warnings, notes, assumptions = [], [], []

    # -- grid ---------------------------------------------------------------
    if cell_size is None:
        cell = auto_cell_size(regs)
        notes.append(f"cell_size not given: {cell:g} m chosen (a round size "
                     f"giving at most about 3,000 cells in the regions).")
    else:
        cell = float(cell_size)
        if not cell > 0:
            raise ValueError("cell_size must be > 0 (m)")
    if margin is not None and float(margin) < 0:
        raise ValueError("margin must be >= 0 (m)")
    lat = build_lattice(regs, cell, margin)
    n_lat = lat["nx"] * lat["ny"]
    # cheap guard before testing points against polygons
    if n_lat > 50 * MAX_LATTICE_CELLS:
        raise ValueError(
            f"cell_size {cell:g} m makes a {lat['nx']} x {lat['ny']} lattice "
            f"({n_lat:,} cells) over the regions; use a larger cell_size.")
    lat_masks = region_masks(regs, lat["px"], lat["py"], cell)
    union = np.zeros(n_lat, dtype=bool)
    for m in lat_masks:
        union |= m
    n_cells = int(union.sum())
    for reg, m in zip(regs, lat_masks):
        if not m.any():
            raise ValueError(
                f"region '{reg['name']}' holds no cell centre at cell_size "
                f"{cell:g} m; use a smaller cell_size.")
    if n_cells > MAX_SIM_CELLS:
        total = sum(float(m.sum()) for m in lat_masks) * cell * cell
        suggest = _nice_step(math.sqrt(n_cells / MAX_SIM_CELLS) * cell)
        raise ValueError(
            f"{n_cells:,} cells of {cell:g} m lie inside the regions; the "
            f"limit is {MAX_SIM_CELLS:,} (the covariance of every cell pair "
            f"is held in memory). Use cell_size >= {suggest:g} m (about "
            f"{int(total / suggest ** 2):,} cells).")
    cell_area = cell * cell
    cpx, cpy = lat["px"][union], lat["py"][union]
    cell_masks = [m[union] for m in lat_masks]
    geom = region_geometry(regs, cell_masks, cell_area, xo, yo)

    # -- covariance model ----------------------------------------------------
    # Every default that depends on the data is computed HERE, from the
    # observations of this call, and echoed in ``defaults_from_data`` (a
    # field session kept a variance worked out before a boring was added).
    extent = float(max(xo.max() - xo.min(), yo.max() - yo.min()))
    d_max = float(np.max(np.hypot(xo[:, None] - xo[None, :],
                                  yo[:, None] - yo[None, :])))
    if extent <= 0:
        raise ValueError("All observations coincide; nothing to krige from.")
    data_defaults = {"observations": {
        "n": n, "min": float(zo.min()), "max": float(zo.max()),
        "mean": float(zo.mean()), "variance": float(np.var(zo)),
        "extent": extent}}
    if variance is None:
        variance_v = float(np.var(zo))
        data_defaults["variance"] = {
            "value": variance_v,
            "basis": (f"sample variance of the {n} observations"
                      + (" (starting value; then fitted)" if fit_variogram
                         else ""))}
        if not fit_variogram:
            assumptions.append(
                f"variance not given: the sample variance of the {n} "
                f"observations ({variance_v:.4g}) was used.")
    else:
        variance_v = float(variance)
    if len_scale is None:
        len_v = extent / 3.0
        data_defaults["len_scale"] = {
            "value": len_v,
            "basis": (f"one third of the data extent ({extent:.4g} m)"
                      + (" (starting value; then fitted)" if fit_variogram
                         else ""))}
        if not fit_variogram:
            assumptions.append(
                f"len_scale not given: one third of the data extent "
                f"({len_v:.4g} m) was used.")
    else:
        len_v = float(len_scale)
    if kriging_type == "simple":
        data_defaults["simple_kriging_mean"] = {
            "value": float(np.mean(zo)),
            "basis": f"mean of the {n} observations"}
    if not variance_v > 0:
        raise ValueError("variance must be > 0 (the observations have no "
                         "spread: give variance explicitly)")
    if not len_v > 0:
        raise ValueError("len_scale must be > 0 (m)")
    nugget_given = nugget is not None
    if nugget_given:
        nug = float(nugget)
        if nug < 0:
            raise ValueError("nugget must be >= 0")
    else:
        nug = float(me.max()) if me is not None else 0.0
        if me is not None and nug > 0:
            notes.append(f"nugget not given: set to the largest measurement "
                         f"error ({nug:.4g}); GSTools treats the measurement "
                         f"error as part of the nugget.")
    if me is not None and nugget_given and float(me.max()) > nug + 1e-12:
        raise ValueError(
            f"measurement_error ({float(me.max()):.4g}) exceeds the nugget "
            f"({nug:.4g}). GSTools treats the measurement error as part of "
            f"the nugget (cond_err <= nugget): give nugget >= "
            f"{float(me.max()):.4g}, or leave nugget out.")
    model_cls = getattr(gs, model_type)
    model = model_cls(dim=2, var=variance_v, len_scale=len_v, nugget=nug,
                      anis=anis, angles=math.radians(angle_deg))
    fitted = False
    if fit_variogram:
        from gstools_agent.variogram import empirical_variogram, fit_warnings
        try:
            bc, gam, _cnt, vw = empirical_variogram(gs, xo, yo, zo)
        except ValueError as exc:
            raise ValueError(
                f"{exc} For conditional_simulation pass fit_variogram=false "
                f"with variance and len_scale.") from None
        model.fit_variogram(bc, gam, nugget=not nugget_given)
        fitted = True
        warnings += vw
        warnings += fit_warnings(model, zo, d_max)
        if me is not None and model.nugget < float(me.max()):
            model.nugget = float(me.max())
            notes.append("The fitted nugget was below the measurement error; "
                         "raised to the largest measurement error.")
        if n < 10:
            warnings.append(
                f"fit_variogram with {n} observations: the fitted sill and "
                f"range are not determined by so few points and the volume "
                f"percentiles follow them directly. Check them against "
                f"judgement or published values.")
    cond_err = me if me is not None else "nugget"
    cond_err_vec = me if me is not None else np.full(n, float(model.nugget))
    exact_data = bool(np.all(cond_err_vec == 0))
    model_c = model_cls(dim=2, var=float(model.var),
                        len_scale=float(model.len_scale), nugget=0.0,
                        anis=anis, angles=math.radians(angle_deg))
    sk_mean = float(np.mean(zo))

    # -- kriged mean on the lattice (deterministic answer and file) ---------
    krig = _make_krige(gs, model, xo, yo, zo, kriging_type, cond_err, sk_mean)
    if n_lat <= MAX_LATTICE_CELLS:
        kx, ky = lat["px"], lat["py"]
        kmean, kvar = krig([kx, ky], mesh_type="unstructured", store=False)
        k_union = np.asarray(kmean)[union]
        k_scope = "lattice"
    else:
        kx, ky = cpx, cpy
        kmean, kvar = krig([kx, ky], mesh_type="unstructured", store=False)
        k_union = np.asarray(kmean)
        k_scope = "cells"
        notes.append(f"The lattice holds {n_lat:,} cells; the kriged-mean "
                     f"file covers the {n_cells:,} cells inside the regions.")
    kmean = np.asarray(kmean, dtype=float)
    # GSTools' kriging variance is of the value AT a point (sill - ...),
    # nugget included; the continuous part's is that less the nugget.
    kvar_c = np.clip(np.asarray(kvar, dtype=float) - float(model.nugget),
                     0.0, None)
    if not (np.all(np.isfinite(kmean)) and np.all(np.isfinite(kvar_c))):
        raise ValueError("Kriging returned non-finite values (a degenerate "
                         "kriging system: check duplicate observations, "
                         "variance and len_scale).")
    ksd_union = np.sqrt(kvar_c[union] if k_scope == "lattice" else kvar_c)

    # -- points the realizations are evaluated at ----------------------------
    bx = np.array([b["x"] for b in blist], dtype=float)
    by = np.array([b["y"] for b in blist], dtype=float)
    Px = np.concatenate([cpx, xo, bx])
    Py = np.concatenate([cpy, yo, by])
    nP = len(Px)
    rows_c = slice(0, n_cells)
    rows_d = slice(n_cells, n_cells + n)
    rows_b = slice(n_cells + n, nP)
    if blist:
        kb, kbv = krig([bx, by], mesh_type="unstructured", store=False)
        kb = np.atleast_1d(np.asarray(kb, dtype=float))
        kbv = np.clip(np.atleast_1d(np.asarray(kbv, dtype=float))
                      - float(model.nugget), 0.0, None)

    # -- realization batches -------------------------------------------------
    jitter = 0.0
    if simulation_method == "kriging_residual":
        W = _kriging_weights(gs, model, xo, yo, kriging_type, cond_err,
                             Px, Py)
        L, jitter = _covariance_factor(model_c, Px, Py)
        offset = sk_mean if kriging_type == "simple" else 0.0
        sd_err = np.sqrt(cond_err_vec)

        def draw(offsets):
            """Realizations drawn with seeds ``seed + offsets``."""
            count = len(offsets)
            Z = np.empty((nP, count))
            E = np.empty((n, count))
            for k, off in enumerate(offsets):
                rng = np.random.default_rng(seed + int(off))
                Z[:, k] = rng.standard_normal(nP)
                E[:, k] = rng.standard_normal(n)
            S = L @ Z
            resid = (zo - offset)[:, None] - (S[rows_d] + sd_err[:, None] * E)
            return S + W @ resid + offset
        batch = max(1, min(256, int(2.5e7 // max(nP, 1))))
    else:
        # Positions set ONCE; with the raw kriging field stored the kriging
        # is solved on the first call and reused by every later one.
        cs = gs.CondSRF(krig)
        cs.set_pos([Px, Py], "unstructured")

        def draw(offsets):
            out = np.empty((nP, len(offsets)))
            for k, off in enumerate(offsets):
                out[:, k] = cs(seed=seed + int(off),
                               store=[False, False, True])
            return out
        t0 = time.perf_counter()
        probe = draw(np.arange(2))
        per = (time.perf_counter() - t0) / 2.0
        planned = n_real * (REJECT_ATTEMPT_FACTOR if (
            blist and bounds_mode == "reject") else 1)
        if per * planned > MAX_PROJECTED_SECONDS:
            raise ValueError(
                f"simulation_method='condsrf' takes about {per:.2f} s per "
                f"realization here ({nP:,} points, {model_type} model), so "
                f"{planned:,} would take about {per * planned / 60:.0f} min. "
                f"Use fewer realizations, a larger cell_size, a Gaussian or "
                f"Exponential model (GSTools samples their spectra directly), "
                f"or simulation_method='kriging_residual'.")
        batch = 64
        notes.append("condsrf: GSTools CondSRF as GSTools builds it -- the "
                     "kriged field plus an unconditional randomization-method "
                     "field scaled by the local kriging standard deviation, "
                     "with the nugget added as white noise per cell. Its "
                     "correlation between points is the unconditioned one, "
                     "so the spread of a total can be overstated; "
                     "kriging_residual draws the conditional field the "
                     "model implies.")

    # -- accumulate -------------------------------------------------------------
    lev_all = sorted({lv for r in regs for lv in r["levels"]})
    lev_index = {lv: i for i, lv in enumerate(lev_all)}
    cnt = np.zeros((len(lev_all), n_cells), dtype=np.int64)
    zsum = np.zeros(n_cells)
    kept_cols = []                 # per kept realization: [vol, area per rl]
    kept_ids = []
    viol = np.zeros(len(blist), dtype=np.int64)
    dev_max = 0.0
    dev_sum = 0.0
    attempts = 0
    rejecting = bool(blist) and bounds_mode == "reject"
    max_attempts = REJECT_ATTEMPT_FACTOR * n_real if rejecting else n_real
    rl_pairs = [(ri, lv) for ri, r in enumerate(regs) for lv in r["levels"]]
    redrawn = []
    n_redraw_seeds = 0
    pending = None
    if simulation_method == "condsrf":
        pending = probe
    while len(kept_ids) < n_real and attempts < max_attempts:
        need = n_real - len(kept_ids)
        count = min(batch, max_attempts - attempts)
        if not rejecting:
            count = min(count, need)
        if pending is not None:
            Zc = pending[:, :min(count, pending.shape[1])]
            pending = None
            count = Zc.shape[1]
        else:
            Zc = draw(np.arange(attempts, attempts + count))
        ids = np.arange(attempts, attempts + count)
        attempts += count
        # A realization with any non-finite value (GSTools has returned an
        # all-NaN field for one seed in a thousand) is redrawn with a fresh
        # seed, so NaN never reaches a statistic.
        bad_cols = np.flatnonzero(~np.all(np.isfinite(Zc), axis=0))
        for k in bad_cols:
            for _try in range(MAX_REDRAW_TRIES):
                off = REDRAW_SEED_OFFSET + n_redraw_seeds
                n_redraw_seeds += 1
                col_new = draw(np.array([off]))[:, 0]
                if np.all(np.isfinite(col_new)):
                    redrawn.append({"replaced_seed": int(seed + ids[k]),
                                    "redrawn_seed": int(seed + off)})
                    Zc[:, k] = col_new
                    ids[k] = off
                    break
            else:
                raise ValueError(
                    f"Realization seed {seed + int(ids[k])} came out "
                    f"non-finite (NaN) and {MAX_REDRAW_TRIES} fresh seeds "
                    f"did too; the covariance model or kriging system is "
                    f"degenerate (check variance, len_scale and duplicate "
                    f"observations).")
        ok = np.ones(count, dtype=bool)
        if blist:
            zb = Zc[rows_b]
            for bi, b in enumerate(blist):
                bad = np.zeros(count, dtype=bool)
                if b["max"] is not None:
                    bad |= zb[bi] > b["max"]
                if b["min"] is not None:
                    bad |= zb[bi] < b["min"]
                viol[bi] += int(bad.sum())
                if rejecting:
                    ok &= ~bad
        sel = np.flatnonzero(ok)[:need]
        if sel.size == 0:
            continue
        Zk = Zc[:, sel]
        zc = Zk[rows_c]
        dev = np.abs(Zk[rows_d] - zo[:, None])
        dev_max = max(dev_max, float(dev.max()))
        dev_sum += float(dev.sum())
        zsum += zc.sum(axis=1)
        for lv, li in lev_index.items():
            if quantity == "above":
                cnt[li] += (zc > lv).sum(axis=1)
            else:
                cnt[li] += (zc < lv).sum(axis=1)
        block = np.empty((sel.size, 2 * len(rl_pairs)))
        col = 0
        for ri, reg in enumerate(regs):
            vol, area = integrate(zc[cell_masks[ri]], reg["levels"],
                                  cell_area, quantity)
            for li in range(len(reg["levels"])):
                block[:, col] = vol[li]
                block[:, col + 1] = area[li]
                col += 2
        kept_cols.append(block)
        kept_ids.extend(ids[sel].tolist())

    n_kept = len(kept_ids)
    if n_kept == 0:
        raise ValueError(
            f"bounds_mode='reject': none of {attempts:,} realizations "
            f"satisfied every bound (the attempt cap, "
            f"{REJECT_ATTEMPT_FACTOR} x n_realizations, was reached). The "
            f"bounds contradict the kriged surface under this covariance "
            f"model; check them, or use bounds_mode='report' to see how "
            f"often each is violated.")
    table = np.vstack(kept_cols)
    if not np.all(np.isfinite(table)):              # belt and braces
        raise ValueError("A non-finite total survived the realization "
                         "guard; no statistics are reported.")
    zmean = zsum / n_kept
    accept_rate = n_kept / attempts if attempts else 1.0
    if redrawn:
        warnings.append(
            f"{len(redrawn)} realization(s) came out non-finite (NaN) and "
            f"were redrawn with fresh seeds (seed + "
            f"{REDRAW_SEED_OFFSET:,} + k); see non_finite_redrawn.")

    # -- per region / level results -------------------------------------------
    results = []
    col = 0
    for ri, reg in enumerate(regs):
        g = geom[ri]
        kv, ka = integrate(k_union[cell_masks[ri]], reg["levels"], cell_area,
                           quantity)
        for li, lv in enumerate(reg["levels"]):
            vol = table[:, col]
            area = table[:, col + 1]
            col += 2
            pw_p, pw_e = pointwise_gaussian(k_union[cell_masks[ri]],
                                            ksd_union[cell_masks[ri]], lv,
                                            quantity)
            rv = (list(report_values) if report_values is not None
                  else _auto_report_values(vol))
            entry = {
                "region": reg["name"],
                "level": float(lv),
                "polygon_area": g["polygon_area"],
                "cell_area_sum": g["cell_area_sum"],
                "n_cells": g["n_cells"],
                f"volume_{quantity}": _stats(vol),
                f"area_{quantity}": _stats(area),
                "p_any": float(np.mean(vol > 0)),
                "exceedance": _exceedance(vol, rv),
                "convergence": _convergence(vol),
                "kriged_surface": {
                    f"volume_{quantity}": float(kv[li]),
                    f"area_{quantity}": float(ka[li]),
                },
                "pointwise_expectation": {
                    f"volume_{quantity}": float(pw_e.sum() * cell_area),
                    f"area_{quantity}": float(pw_p.sum() * cell_area),
                },
                "nearest_observation_to_centroid": {
                    "name": obs_names[g["nearest_observation_index"]],
                    "distance": g["nearest_observation_distance"],
                },
            }
            results.append(entry)
            conv = entry["convergence"]
            if len(conv) >= 2 and conv[-1]["P90"] > 0:
                ch = abs(conv[-1]["P90"] - conv[-2]["P90"]) / conv[-1]["P90"]
                if ch > 0.05:
                    warnings.append(
                        f"{reg['name']} @ {lv:g}: the P90 moved "
                        f"{100 * ch:.0f}% between {conv[-2]['n']} and "
                        f"{conv[-1]['n']} realizations; use more "
                        f"realizations for a stable P90.")

    # -- warnings ---------------------------------------------------------------
    if n < 10:
        warnings.append(
            f"Only {n} observations: the covariance model (variance "
            f"{float(model.var):.4g}, len_scale {float(model.len_scale):.4g} "
            f"m, nugget {float(model.nugget):.4g}) cannot be checked against "
            f"these data and the percentiles depend on it directly; vary "
            f"variance and len_scale to see how much.")
    for reg, g in zip(regs, geom):
        if abs(g["discretisation_ratio"] - 1.0) > 0.05:
            warnings.append(
                f"{reg['name']}: the cells sum to {g['cell_area_sum']:.4g} "
                f"m2 against a polygon area of {g['polygon_area']:.4g} m2 "
                f"({100 * (g['discretisation_ratio'] - 1):+.1f}%); use a "
                f"smaller cell_size.")
        if g["n_cells"] < 25:
            warnings.append(f"{reg['name']}: only {g['n_cells']} cells; the "
                            f"totals are coarse. Use a smaller cell_size.")
        if g["nearest_observation_distance"] > float(model.len_scale):
            warnings.append(
                f"{reg['name']}: its centroid is "
                f"{g['nearest_observation_distance']:.1f} m from the nearest "
                f"observation ({obs_names[g['nearest_observation_index']]}), "
                f"more than len_scale ({float(model.len_scale):.4g} m); the "
                f"field there is close to the unconditioned mean with its "
                f"full variance.")
        for lv in reg["levels"]:
            if lv > zo.max() or lv < zo.min():
                warnings.append(
                    f"{reg['name']}: level {lv:g} is outside the observed "
                    f"range [{zo.min():g}, {zo.max():g}]; the result there "
                    f"comes from the simulated variation and the covariance "
                    f"model, not from data near that level.")
    if cell > float(model.len_scale) / 3.0:
        warnings.append(
            f"cell_size {cell:g} m is more than a third of len_scale "
            f"({float(model.len_scale):.4g} m); the field varies within a "
            f"cell. Use a smaller cell_size.")
    if len(np.unique(np.round(np.column_stack([xo, yo]), 6), axis=0)) < n:
        warnings.append("Two or more observations share a location; GSTools "
                        "averages them (pseudo-inverse).")
    if jitter and jitter >= 1e-6:
        notes.append(f"The covariance factor needed a diagonal jitter of "
                     f"{jitter:g} x variance (a smooth model on close cells "
                     f"is numerically singular); its effect is negligible.")
    elif jitter != jitter:                         # NaN: eigen fallback
        notes.append("The covariance did not factor by Cholesky; an "
                     "eigen-factor with negative eigenvalues cut was used.")
    bounds_out = []
    if blist:
        for bi, b in enumerate(blist):
            rec = {"name": b["name"], "x": b["x"], "y": b["y"]}
            if b["max"] is not None:
                rec["max"] = b["max"]
            if b["min"] is not None:
                rec["min"] = b["min"]
            rec["kriged_value"] = float(kb[bi])
            rec["kriging_std"] = float(math.sqrt(kbv[bi]))
            rec["violated_fraction"] = float(viol[bi] / attempts)
            bounds_out.append(rec)
        if bounds_mode == "reject":
            if n_kept < n_real:
                warnings.append(
                    f"bounds_mode='reject' reached its attempt cap "
                    f"({attempts:,} = {REJECT_ATTEMPT_FACTOR} x "
                    f"n_realizations) with only {n_kept:,} of {n_real:,} "
                    f"realizations satisfying every bound; the statistics "
                    f"use those {n_kept:,}.")
            if accept_rate < 0.1:
                warnings.append(
                    f"Only {100 * accept_rate:.1f}% of realizations satisfied "
                    f"every bound: the bounds sit far in the tail of the "
                    f"model. Rejection keeps the few that fit, which is "
                    f"slow and can be unrepresentative; check the bounds and "
                    f"the covariance model.")
        else:
            bad = [b for b in bounds_out if b["violated_fraction"] > 0.05]
            if bad:
                warnings.append(
                    "bounds_mode='report': "
                    + "; ".join(f"{b['name']} violated in "
                                f"{100 * b['violated_fraction']:.0f}%"
                                for b in bad)
                    + " of realizations. The statistics INCLUDE those "
                    "realizations; bounds_mode='reject' keeps only the ones "
                    "that satisfy every bound.")

    assumptions += [
        "The field is Gaussian and stationary: one covariance model "
        "everywhere, about a constant mean (ordinary kriging), a known mean "
        "(simple) or a linear drift (universal). A transform is not applied.",
        "Realizations are the continuous part of the field; the nugget is "
        "not added cell by cell (a total over cells would then depend on the "
        "cell size). It enters through the observations as their error "
        "variance (GSTools cond_err)." if simulation_method ==
        "kriging_residual" else
        "condsrf adds the nugget as white noise at every cell (GSTools' "
        "behaviour); with a nugget the totals depend somewhat on cell size.",
        "Every region and level is evaluated on the same realizations "
        "(common random numbers), so differences between footprints and "
        "levels are not Monte Carlo noise.",
    ]

    if simulation_method == "kriging_residual":
        method_text = (
            "Conditioning by kriging (Journel & Huijbregts 1978, ch. V; "
            "Chiles & Delfiner 2012, s. 7.3.1): realization = unconditional "
            "Gaussian field S + kriging of (observations - S at the "
            "observations), with the kriging weights from GSTools' "
            f"{kriging_type} Krige and S drawn by Cholesky factoring of the "
            f"{model_type} covariance over the cells. It draws the "
            "distribution the chosen model implies: mean = kriged surface, "
            "pointwise variance = kriging variance, and the conditional "
            "correlation between cells (checked in the tests against "
            "keeping only the unconditioned fields that pass through the "
            "observations). The model itself (a Gaussian field, its "
            "covariance, which observations count) is an assumption and "
            "usually moves the answer more than the method does: run "
            "cases.")
    else:
        method_text = (
            "GSTools CondSRF (Mueller et al. 2022, GMD 15, 3161): kriged "
            "field + unconditional randomization-method field scaled by the "
            "local kriging standard deviation.")

    return ConditionalSimulationResult(
        simulation_method=simulation_method,
        method_basis=method_text,
        n_data=n,
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        nugget=float(model.nugget),
        anis=anis,
        angle_deg=angle_deg,
        fitted=fitted,
        kriging_type=kriging_type,
        measurement_error=(None if me is None else
                           (float(me[0]) if np.all(me == me[0])
                            else me.tolist())),
        data_exact=exact_data,
        quantity=quantity,
        cell_size=cell,
        cell_area=cell_area,
        n_cells=n_cells,
        lattice={k: lat[k] for k in ("nx", "ny", "x_min", "x_max", "y_min",
                                     "y_max", "margin")},
        n_realizations=n_real,
        n_kept=n_kept,
        attempts=attempts,
        acceptance_rate=accept_rate,
        seed=seed,
        bounds_mode=bounds_mode if blist else None,
        bounds=bounds_out,
        results=results,
        data_check={
            "max_abs_deviation": dev_max,
            "mean_abs_deviation": dev_sum / (n_kept * n),
        },
        defaults_from_data=data_defaults,
        redrawn=redrawn,
        warnings=warnings,
        notes=notes,
        assumptions=assumptions,
        runtime_s=time.perf_counter() - t_start,
        # arrays and plotting inputs (not in to_dict)
        obs_x=xo, obs_y=yo, obs_values=zo, obs_names=obs_names,
        regions_geom=[{"name": r["name"], "polygon": r["polygon"],
                       "levels": list(r["levels"])} for r in regs],
        cell_x=cpx, cell_y=cpy, cell_masks=cell_masks,
        levels_all=lev_all, exceed_counts=cnt,
        realization_mean=zmean, kriged_cells=k_union,
        realization_table=table,
        realization_ids=np.asarray(kept_ids, dtype=np.int64) + seed,
        rl_pairs=[(regs[ri]["name"], lv) for ri, lv in rl_pairs],
        kriging_x=np.asarray(kx), kriging_y=np.asarray(ky),
        kriging_mean=kmean, kriging_var=kvar_c, kriging_scope=k_scope,
        lattice_full=lat if k_scope == "lattice" else None,
        lattice_union=union,
        bounds_input=blist,
    )
