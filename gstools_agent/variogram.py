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
        Number of lag bins. Default 10.
    nugget : float
        Nugget variance. Default 0.

    Returns
    -------
    VariogramResult
        Empirical variogram and fitted model parameters.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    val_arr = np.asarray(values, dtype=float)

    _validate_variogram_inputs(x_arr, y_arr, val_arr, model_type)

    gs = import_gstools()

    bin_center, gamma = gs.vario_estimate(
        [x_arr, y_arr], val_arr, bin_no=n_bins,
    )

    model_cls = getattr(gs, model_type)
    model = model_cls(dim=2)
    model.fit_variogram(bin_center, gamma, nugget=nugget >= 0)

    return VariogramResult(
        n_data=len(x_arr),
        n_bins=len(bin_center),
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        nugget=float(model.nugget),
        bin_center=bin_center,
        gamma=gamma,
    )
