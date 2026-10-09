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


# ---------------------------------------------------------------------------
# One-call analyses: sample, evaluate the model, analyze (live smoke G9).
# The two-step functions above hand a sample matrix back to the caller to
# evaluate; an agent cannot loop over 1,280 rows, so these do it here.
# ---------------------------------------------------------------------------

DEFAULT_MAX_EVALUATIONS = 20000


def _resolve_model(model, expression, var_names):
    if (model is None) == (expression is None):
        raise ValueError(
            "Give exactly one of: model (a callable taking a dict of the "
            "variables) or expression (an arithmetic expression of them, "
            "e.g. '(c + 18*z*tan(radians(phi)))/40').")
    if expression is not None:
        from salib_agent.expression import compile_expression
        return compile_expression(expression, var_names)
    if not callable(model):
        raise ValueError("model must be callable: f(values: dict) -> float")
    return model


def _evaluate(model, X, var_names, max_evaluations):
    n = len(X)
    if n > max_evaluations:
        raise ValueError(
            f"The design needs {n} model evaluations, above the limit of "
            f"{max_evaluations}. Use fewer samples (Sobol: N*(2D+2) rows "
            f"with second-order indices, N*(D+2) without; Morris: "
            f"trajectories*(D+1)), or screen with Morris first.")
    Y = np.empty(n)
    for i, row in enumerate(X):
        point = {name: float(v) for name, v in zip(var_names, row)}
        try:
            Y[i] = float(model(point))
        except Exception as exc:
            raise ValueError(
                f"The model failed at sample {i + 1} of {n}, {point}: "
                f"{type(exc).__name__}: {exc}") from exc
    bad = ~np.isfinite(Y)
    if bad.any():
        j = int(np.argmax(bad))
        raise ValueError(
            f"The model returned a non-finite value at {int(bad.sum())} of "
            f"{n} sample points (first: "
            f"{dict(zip(var_names, map(float, X[j])))}). Sensitivity "
            f"indices need a finite output everywhere inside the bounds: "
            f"narrow the bounds or guard the model.")
    return Y


def sobol_analysis(
    var_names,
    bounds,
    model=None,
    expression=None,
    n_samples=256,
    calc_second_order=False,
    seed=42,
    max_evaluations=DEFAULT_MAX_EVALUATIONS,
) -> SobolResult:
    """Sobol indices in one call: sample, evaluate the model, analyze.

    Parameters
    ----------
    var_names : list of str
    bounds : list of [min, max]
    model : callable, optional
        ``f(values: dict) -> float`` evaluated at every sample point.
    expression : str, optional
        Instead of ``model``: an arithmetic expression of the variables
        (see ``salib_agent.expression``).
    n_samples : int
        Base sample size N (a power of 2 keeps the Sobol sequence balanced).
        Evaluations = N*(2D+2) with second-order indices, N*(D+2) without.
        Default 256.
    calc_second_order : bool
        Default False (halves the evaluations; S1 and ST do not need it).
    seed : int
    max_evaluations : int
        Refuse a design larger than this. Default 20,000.

    Returns
    -------
    SobolResult
        Indices only, plus the output mean and standard deviation; never
        the sample matrix.
    """
    f = _resolve_model(model, expression, var_names)
    X = sobol_sample(var_names, bounds, n_samples=n_samples,
                     calc_second_order=calc_second_order, seed=seed)
    Y = _evaluate(f, X, list(var_names), max_evaluations)
    result = sobol_analyze(var_names, bounds, Y, n_samples=n_samples,
                           calc_second_order=calc_second_order, seed=seed)
    result.output_mean = float(np.mean(Y))
    result.output_std = float(np.std(Y))
    return result


def morris_analysis(
    var_names,
    bounds,
    model=None,
    expression=None,
    n_trajectories=20,
    num_levels=4,
    seed=42,
    max_evaluations=DEFAULT_MAX_EVALUATIONS,
) -> MorrisResult:
    """Morris screening in one call: sample, evaluate the model, analyze.

    Evaluations = n_trajectories * (D + 1). See ``sobol_analysis`` for
    ``model`` / ``expression``.
    """
    f = _resolve_model(model, expression, var_names)
    X = morris_sample(var_names, bounds, n_trajectories=n_trajectories,
                      num_levels=num_levels, seed=seed)
    Y = _evaluate(f, X, list(var_names), max_evaluations)
    result = morris_analyze(var_names, bounds, X, Y,
                            n_trajectories=n_trajectories,
                            num_levels=num_levels, seed=seed)
    result.output_mean = float(np.mean(Y))
    result.output_std = float(np.std(Y))
    return result


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
