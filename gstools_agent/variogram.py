"""
Variogram estimation and fitting using GSTools.
"""

import numpy as np

from gstools_agent.gstools_utils import import_gstools
from gstools_agent.results import VariogramResult


_VALID_MODELS = {
    "Gaussian", "Exponential", "Matern", "Spherical", "Linear",
    "Stable", "Rational", "Cubic", "HyperSpherical",
}


def _validate_variogram_inputs(x, y, values, model_type):
    if len(x) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x)}")
    if len(x) != len(y) or len(x) != len(values):
        raise ValueError(
            f"x, y, values must have same length: "
            f"x={len(x)}, y={len(y)}, values={len(values)}"
        )
    if model_type not in _VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(_VALID_MODELS)}, got '{model_type}'"
        )


# A common rule for a usable experimental variogram asks for 30-50 pairs
# per lag class (Journel & Huijbregts 1978, Mining Geostatistics).
_MIN_PAIRS_ADVISED = 30
_MIN_LAG_CLASSES = 3      # a sill + range (+ nugget) model needs >= 3 points


def empirical_variogram(gs, x, y, values, n_bins=10):
    """Experimental variogram with bins built from the data's own pair
    distances; empty classes are dropped.

    gstools' automatic bins span only a fraction of the field diameter, so
    sparse data can have no pair in ANY bin and every gamma reads 0 (live
    smoke GS-1: four points 10 m apart, bins at 0.6-4.1 m). Here the classes
    run from 0 to the largest pair distance, and only classes holding pairs
    are kept.

    Returns
    -------
    bin_center, gamma, counts : numpy arrays (populated classes only)
    warnings : list of str

    Raises
    ------
    ValueError
        When fewer than three lag classes hold pairs: no variogram model
        (sill, range, nugget) is determined by that.
    """
    pts = np.column_stack([x, y])
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))[np.triu_indices(len(pts), k=1)]
    d_max = float(dist.max()) if dist.size else 0.0
    if d_max <= 0.0:
        raise ValueError("All data points coincide; no variogram can be "
                         "estimated.")
    distinct = np.unique(np.round(dist, 9))
    edges = np.linspace(0.0, d_max * (1.0 + 1e-9), int(n_bins) + 1)
    bin_center, gamma, counts = gs.vario_estimate(
        [x, y], values, bin_edges=edges, return_counts=True)
    keep = np.asarray(counts) > 0
    bin_center = np.asarray(bin_center)[keep]
    gamma = np.asarray(gamma)[keep]
    counts = np.asarray(counts)[keep]
    if len(bin_center) < _MIN_LAG_CLASSES:
        raise ValueError(
            f"Only {len(bin_center)} lag class(es) hold any pairs "
            f"({len(dist)} pairs at {len(distinct)} distinct separations: "
            f"{[round(float(v), 3) for v in distinct[:6]]}). A variogram "
            f"model (sill, range, nugget) cannot be fitted to fewer than "
            f"{_MIN_LAG_CLASSES} points, so it is not fitted. Use more data, "
            f"or set the sill (variance) and range (len_scale) from "
            f"judgment / a published COV and correlation length (for "
            f"kriging: fit_variogram=false with variance and len_scale).")
    warnings = []
    thin = int((counts < _MIN_PAIRS_ADVISED).sum())
    if thin:
        warnings.append(
            f"{thin} of {len(counts)} lag classes hold fewer than "
            f"{_MIN_PAIRS_ADVISED} pairs (counts {counts.tolist()}); a common "
            f"rule asks for 30-50 per class (Journel & Huijbregts 1978), so "
            f"the fitted parameters are poorly constrained.")
    return bin_center, gamma, counts, warnings


def fit_warnings(model, values, d_max):
    """Flags for a fitted model the data do not determine."""
    out = []
    sample_var = float(np.var(values))
    if model.len_scale > 10.0 * d_max:
        out.append(
            f"DEGENERATE FIT: the fitted correlation length "
            f"({model.len_scale:.4g}) is more than 10x the largest data "
            f"separation ({d_max:.4g}); the range is not determined by the "
            f"data.")
    if sample_var > 0 and (model.var + model.nugget) < 1e-3 * sample_var:
        out.append(
            f"DEGENERATE FIT: the fitted sill ({model.var + model.nugget:.4g})"
            f" is negligible against the sample variance "
            f"({sample_var:.4g}).")
    return out


def analyze_variogram(
    x,
    y,
    values,
    model_type="Gaussian",
    n_bins=10,
    nugget=0.0,
) -> VariogramResult:
    """Estimate and fit an empirical variogram.

    Parameters
    ----------
    x, y : array-like
        Coordinates of measurement points.
    values : array-like
        Measured values at each point.
    model_type : str
        Covariance model to fit. Default 'Gaussian'.
    n_bins : int
        Number of lag classes between 0 and the largest pair separation;
        classes with no pairs are dropped. Default 10.
    nugget : float
        Nugget variance. Default 0.

    Returns
    -------
    VariogramResult
        Empirical variogram (populated classes, with pair counts) and the
        fitted model, with ``warnings`` when the fit is poorly constrained
        or degenerate.

    Raises
    ------
    ValueError
        When fewer than three lag classes hold pairs (see
        ``empirical_variogram``).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    val_arr = np.asarray(values, dtype=float)

    _validate_variogram_inputs(x_arr, y_arr, val_arr, model_type)

    gs = import_gstools()

    bin_center, gamma, counts, warnings = empirical_variogram(
        gs, x_arr, y_arr, val_arr, n_bins=n_bins)

    model_cls = getattr(gs, model_type)
    model = model_cls(dim=2)
    model.fit_variogram(bin_center, gamma, nugget=nugget >= 0)
    warnings += fit_warnings(model, val_arr, float(
        np.max(np.hypot(x_arr[:, None] - x_arr[None, :],
                        y_arr[:, None] - y_arr[None, :]))))

    return VariogramResult(
        n_data=len(x_arr),
        n_bins=len(bin_center),
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        nugget=float(model.nugget),
        bin_center=bin_center,
        gamma=gamma,
        pair_counts=counts,
        warnings=warnings,
    )
