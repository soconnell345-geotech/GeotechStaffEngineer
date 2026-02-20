"""
Kriging interpolation wrapper using GSTools.

Spatially interpolates soil properties (SPT N, Vs, etc.) from point
measurements onto a regular grid.
"""

import numpy as np

from gstools_agent.gstools_utils import import_gstools
from gstools_agent.results import KrigingResult


_VALID_MODELS = {
    "Gaussian", "Exponential", "Matern", "Spherical", "Linear",
    "Stable", "Rational", "Cubic", "HyperSpherical",
}

_VALID_KRIGE_TYPES = {"ordinary", "simple", "universal"}


def _validate_kriging_inputs(x, y, values, model_type, kriging_type,
                             grid_x_min, grid_x_max, grid_y_min, grid_y_max,
                             n_grid_x, n_grid_y):
    """Validate kriging inputs."""
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
    if kriging_type not in _VALID_KRIGE_TYPES:
        raise ValueError(
            f"kriging_type must be one of {sorted(_VALID_KRIGE_TYPES)}, got '{kriging_type}'"
        )
    if grid_x_max <= grid_x_min:
        raise ValueError(f"grid_x_max must be > grid_x_min")
    if grid_y_max <= grid_y_min:
        raise ValueError(f"grid_y_max must be > grid_y_min")
    if n_grid_x < 2 or n_grid_y < 2:
        raise ValueError(f"n_grid_x and n_grid_y must be >= 2")


def analyze_kriging(
    x,
    y,
    values,
    model_type="Gaussian",
    kriging_type="ordinary",
    grid_x_min=None,
    grid_x_max=None,
    grid_y_min=None,
    grid_y_max=None,
    n_grid_x=50,
    n_grid_y=50,
    variance=None,
    len_scale=None,
    nugget=0.0,
    fit_variogram=True,
) -> KrigingResult:
    """Krige soil property values onto a regular grid.

    Parameters
    ----------
    x, y : array-like
        Coordinates of measurement points.
    values : array-like
        Measured values at each point (e.g. SPT N-values).
    model_type : str
        Covariance model. Default 'Gaussian'.
    kriging_type : str
        Kriging type: 'ordinary', 'simple', or 'universal'. Default 'ordinary'.
    grid_x_min, grid_x_max : float or None
        Grid extent in X. None = auto from data bounds with 10% buffer.
    grid_y_min, grid_y_max : float or None
        Grid extent in Y.
    n_grid_x, n_grid_y : int
        Number of grid points. Default 50 each.
    variance : float or None
        Sill variance. None = estimate from data.
    len_scale : float or None
        Correlation length. None = estimate from data.
    nugget : float
        Nugget variance. Default 0.
    fit_variogram : bool
        If True, fit variogram model to data. Default True.

    Returns
    -------
    KrigingResult
        Kriged field, variance, and model parameters.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    val_arr = np.asarray(values, dtype=float)

    # Auto grid bounds
    buf = 0.1
    if grid_x_min is None:
        rng = x_arr.max() - x_arr.min()
        grid_x_min = x_arr.min() - buf * max(rng, 1.0)
    if grid_x_max is None:
        rng = x_arr.max() - x_arr.min()
        grid_x_max = x_arr.max() + buf * max(rng, 1.0)
    if grid_y_min is None:
        rng = y_arr.max() - y_arr.min()
        grid_y_min = y_arr.min() - buf * max(rng, 1.0)
    if grid_y_max is None:
        rng = y_arr.max() - y_arr.min()
        grid_y_max = y_arr.max() + buf * max(rng, 1.0)

    _validate_kriging_inputs(
        x_arr, y_arr, val_arr, model_type, kriging_type,
        grid_x_min, grid_x_max, grid_y_min, grid_y_max,
        n_grid_x, n_grid_y,
    )

    gs = import_gstools()

    # Build covariance model
    model_cls = getattr(gs, model_type)
    if variance is None:
        variance = float(np.var(val_arr))
    if len_scale is None:
        # Initial guess: 1/3 of data extent
        extent = max(x_arr.max() - x_arr.min(), y_arr.max() - y_arr.min())
        len_scale = extent / 3.0

    model = model_cls(dim=2, var=variance, len_scale=len_scale, nugget=nugget)

    # Fit variogram if requested
    if fit_variogram:
        bin_center, gamma = gs.vario_estimate([x_arr, y_arr], val_arr)
        model.fit_variogram(bin_center, gamma, nugget=nugget >= 0)

    # Run kriging
    krige_cls_map = {
        "ordinary": gs.krige.Ordinary,
        "simple": gs.krige.Simple,
        "universal": gs.krige.Universal,
    }
    krige_cls = krige_cls_map[kriging_type]
    krig = krige_cls(model, [x_arr, y_arr], val_arr)

    grid_x = np.linspace(grid_x_min, grid_x_max, n_grid_x)
    grid_y = np.linspace(grid_y_min, grid_y_max, n_grid_y)
    field, krige_var = krig.structured([grid_x, grid_y])

    return KrigingResult(
        n_data=len(x_arr),
        n_grid_x=n_grid_x,
        n_grid_y=n_grid_y,
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        nugget=float(model.nugget),
        kriging_type=kriging_type,
        field=field,
        krige_variance=krige_var,
        grid_x=grid_x,
        grid_y=grid_y,
    )
