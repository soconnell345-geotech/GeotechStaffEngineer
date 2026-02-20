"""
Spatial random field generation using GSTools.
"""

import numpy as np

from gstools_agent.gstools_utils import import_gstools
from gstools_agent.results import RandomFieldResult


_VALID_MODELS = {
    "Gaussian", "Exponential", "Matern", "Spherical", "Linear",
    "Stable", "Rational", "Cubic", "HyperSpherical",
}


def _validate_rf_inputs(model_type, variance, len_scale,
                        x_min, x_max, y_min, y_max, n_x, n_y):
    if model_type not in _VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(_VALID_MODELS)}, got '{model_type}'"
        )
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}")
    if len_scale <= 0:
        raise ValueError(f"len_scale must be positive, got {len_scale}")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("max must be > min for both x and y")
    if n_x < 2 or n_y < 2:
        raise ValueError("n_x and n_y must be >= 2")


def generate_random_field(
    model_type="Gaussian",
    variance=1.0,
    len_scale=10.0,
    nugget=0.0,
    mean=0.0,
    x_min=0.0,
    x_max=100.0,
    y_min=0.0,
    y_max=100.0,
    n_x=50,
    n_y=50,
    seed=42,
) -> RandomFieldResult:
    """Generate a 2D spatial random field.

    Useful for probabilistic geotechnical analysis — modeling spatial
    variability of soil properties like SPT N, friction angle, etc.

    Parameters
    ----------
    model_type : str
        Covariance model. Default 'Gaussian'.
    variance : float
        Field variance. Default 1.0.
    len_scale : float
        Correlation length. Default 10.0.
    nugget : float
        Nugget variance. Default 0.
    mean : float
        Mean value. Default 0. The field = mean + random component.
    x_min, x_max : float
        X-axis extent.
    y_min, y_max : float
        Y-axis extent.
    n_x, n_y : int
        Number of grid points. Default 50.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    RandomFieldResult
        Generated field and model parameters.
    """
    _validate_rf_inputs(
        model_type, variance, len_scale,
        x_min, x_max, y_min, y_max, n_x, n_y,
    )

    gs = import_gstools()

    model_cls = getattr(gs, model_type)
    model = model_cls(dim=2, var=variance, len_scale=len_scale, nugget=nugget)

    srf = gs.SRF(model, mean=mean, seed=seed)
    grid_x = np.linspace(x_min, x_max, n_x)
    grid_y = np.linspace(y_min, y_max, n_y)
    field = srf.structured([grid_x, grid_y])

    return RandomFieldResult(
        n_grid_x=n_x,
        n_grid_y=n_y,
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        mean=mean,
        seed=seed,
        field=field,
        grid_x=grid_x,
        grid_y=grid_y,
    )
