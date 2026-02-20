"""
Sensitivity analysis wrappers using SALib.
"""

import numpy as np

from salib_agent.salib_utils import (
    import_salib_sobol_sample,
    import_salib_sobol_analyze,
    import_salib_morris_sample,
    import_salib_morris_analyze,
)
from salib_agent.results import SobolResult, MorrisResult


def _validate_problem(var_names, bounds):
    """Validate problem definition."""
    if len(var_names) < 2:
        raise ValueError(f"Need at least 2 variables, got {len(var_names)}")
    if len(var_names) != len(bounds):
        raise ValueError(
            f"var_names and bounds must have same length: "
            f"var_names={len(var_names)}, bounds={len(bounds)}"
        )
    for i, b in enumerate(bounds):
        if len(b) != 2:
            raise ValueError(f"bounds[{i}] must have 2 values [min, max], got {len(b)}")
        if b[0] >= b[1]:
            raise ValueError(f"bounds[{i}]: min ({b[0]}) must be < max ({b[1]})")


def _validate_model_output(Y, n_samples):
    """Validate model output array."""
    if len(Y) != n_samples:
        raise ValueError(
            f"Y length ({len(Y)}) must equal number of samples ({n_samples})"
        )


def sobol_analyze(
    var_names,
    bounds,
    Y,
    n_samples=1024,
    calc_second_order=True,
    seed=42,
) -> SobolResult:
    """Run Sobol variance-based sensitivity analysis.

    The caller must evaluate the model at the Sobol sample points and
    provide the output Y. Use sobol_sample() to generate the sample
    matrix first, evaluate your model, then pass Y here.

    Parameters
    ----------
    var_names : list of str
        Variable names.
    bounds : list of [min, max]
        Bounds for each variable.
    Y : array-like
        Model output for each sample point.
    n_samples : int
        Base number of samples used for Sobol sampling. Default 1024.
    calc_second_order : bool
        Whether second-order indices were computed. Default True.
    seed : int
        Random seed for resampling. Default 42.

    Returns
    -------
    SobolResult
        Sobol sensitivity indices.
    """
    _validate_problem(var_names, bounds)
    Y_arr = np.asarray(Y, dtype=float)

    problem = {
        'num_vars': len(var_names),
        'names': list(var_names),
        'bounds': [list(b) for b in bounds],
    }

    sobol_mod = import_salib_sobol_analyze()
    Si = sobol_mod.analyze(
        problem, Y_arr,
        calc_second_order=calc_second_order,
        seed=seed,
    )

    s2 = None
    if calc_second_order and 'S2' in Si:
        s2 = Si['S2']

    return SobolResult(
        n_samples=len(Y_arr),
        n_vars=len(var_names),
        var_names=list(var_names),
        S1=list(Si['S1']),
        S1_conf=list(Si['S1_conf']),
        ST=list(Si['ST']),
        ST_conf=list(Si['ST_conf']),
        S2=s2,
    )


def sobol_sample(
    var_names,
    bounds,
    n_samples=1024,
    calc_second_order=True,
    seed=42,
) -> np.ndarray:
    """Generate Sobol sample matrix.

    Parameters
    ----------
    var_names : list of str
        Variable names.
    bounds : list of [min, max]
        Bounds for each variable.
    n_samples : int
        Base number of samples. Total = N*(2D+2) for second-order.
    calc_second_order : bool
        Include second-order indices. Default True.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    np.ndarray
        Sample matrix of shape (total_samples, n_vars).
    """
    _validate_problem(var_names, bounds)

    problem = {
        'num_vars': len(var_names),
        'names': list(var_names),
        'bounds': [list(b) for b in bounds],
    }

    sobol_mod = import_salib_sobol_sample()
    return sobol_mod.sample(
        problem, n_samples,
        calc_second_order=calc_second_order,
        seed=seed,
    )


def morris_analyze(
    var_names,
    bounds,
    X,
    Y,
    n_trajectories=20,
    num_levels=4,
    seed=42,
) -> MorrisResult:
    """Run Morris elementary effects screening.

    Parameters
    ----------
    var_names : list of str
        Variable names.
    bounds : list of [min, max]
        Bounds for each variable.
    X : array-like
        Sample matrix from morris_sample().
    Y : array-like
        Model output for each sample point.
    n_trajectories : int
        Number of trajectories used in sampling.
    num_levels : int
        Number of grid levels used in sampling.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    MorrisResult
        Morris elementary effects (mu*, sigma).
    """
    _validate_problem(var_names, bounds)
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)

    problem = {
        'num_vars': len(var_names),
        'names': list(var_names),
        'bounds': [list(b) for b in bounds],
    }

    morris_mod = import_salib_morris_analyze()
    Si = morris_mod.analyze(
        problem, X_arr, Y_arr,
        num_levels=num_levels,
        seed=seed,
    )

    return MorrisResult(
        n_trajectories=n_trajectories,
        n_vars=len(var_names),
        var_names=list(var_names),
        mu_star=list(Si['mu_star']),
        sigma=list(Si['sigma']),
        mu_star_conf=list(Si['mu_star_conf']),
    )


def morris_sample(
    var_names,
    bounds,
    n_trajectories=20,
    num_levels=4,
    seed=42,
) -> np.ndarray:
    """Generate Morris sample matrix.

    Parameters
    ----------
    var_names : list of str
        Variable names.
    bounds : list of [min, max]
        Bounds for each variable.
    n_trajectories : int
        Number of trajectories. Default 20.
    num_levels : int
        Number of grid levels. Default 4.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    np.ndarray
        Sample matrix of shape (n_trajectories * (n_vars + 1), n_vars).
    """
    _validate_problem(var_names, bounds)

    problem = {
        'num_vars': len(var_names),
        'names': list(var_names),
        'bounds': [list(b) for b in bounds],
    }

    morris_mod = import_salib_morris_sample()
    return morris_mod.sample(
        problem, n_trajectories,
        num_levels=num_levels,
        seed=seed,
    )
