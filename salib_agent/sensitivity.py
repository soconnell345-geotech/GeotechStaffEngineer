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


def sobol_rows_per_base_sample(n_vars, calc_second_order):
    """Rows of a Saltelli/Sobol design per base sample N: 2D+2 with
    second-order indices, D+2 without."""
    return 2 * n_vars + 2 if calc_second_order else n_vars + 2


def _sobol_base_samples(n_outputs, n_vars, calc_second_order, n_samples):
    """The base sample size N the outputs imply, checked against the
    design (GM-323).

    A Sobol design for D variables has N*(2D+2) rows with second-order
    indices and N*(D+2) without. An output count that is not a whole
    multiple was not made by that design (most often ``calc_second_order``
    differs between the sampling and the analysis), and an ``n_samples``
    that disagrees with the count is refused instead of being ignored.
    """
    per = sobol_rows_per_base_sample(n_vars, calc_second_order)
    kind = (f"N*(2D+2) = N*{per} with second-order indices"
            if calc_second_order else f"N*(D+2) = N*{per} without "
            f"second-order indices")
    if n_outputs == 0 or n_outputs % per:
        other = sobol_rows_per_base_sample(n_vars, not calc_second_order)
        hint = (f" It is a whole multiple of {other}: was the sample made "
                f"with calc_second_order={not calc_second_order}?"
                if n_outputs and n_outputs % other == 0 else "")
        raise ValueError(
            f"Y has {n_outputs} value(s), which is not a Sobol design for "
            f"{n_vars} variables: sobol_sample gives {kind} rows, one "
            f"output per row.{hint}")
    n_base = n_outputs // per
    if n_samples is not None and int(n_samples) != n_base:
        raise ValueError(
            f"n_samples={n_samples} does not match Y: a design with "
            f"N={n_samples} for {n_vars} variables has "
            f"{int(n_samples) * per} rows ({kind}), and Y has {n_outputs} "
            f"values (N={n_base}). Give the n_samples and "
            f"calc_second_order used in sobol_sample, or leave n_samples "
            f"out to take N from Y.")
    return n_base


def _sobol_design_ambiguity(n_outputs, n_vars, calc_second_order,
                            n_samples):
    """With ``n_samples`` left out, a Y whose length fits both designs (a
    whole multiple of both 2D+2 and D+2; for D = 1, 3, 7 and 15 every
    power-of-two N does) cannot say which design made it. The analysis
    takes the one ``calc_second_order`` names; this says so, with the
    other reading (GM-323 review). Returns (warning, choice) or None."""
    if n_samples is not None or n_outputs == 0:
        return None
    per = sobol_rows_per_base_sample(n_vars, calc_second_order)
    other = sobol_rows_per_base_sample(n_vars, not calc_second_order)
    if n_outputs % per or n_outputs % other:
        return None
    with_2nd = "with second-order indices, N*(2D+2)"
    without = "without second-order indices, N*(D+2)"
    taken, not_taken = ((with_2nd, without) if calc_second_order
                        else (without, with_2nd))
    n_taken, n_other = n_outputs // per, n_outputs // other
    warning = (
        f"Y's {n_outputs} values fit two Sobol designs for {n_vars} "
        f"variables: N={n_taken} {taken}, taken because "
        f"calc_second_order={calc_second_order}, and N={n_other} "
        f"{not_taken}. If the sample was made with "
        f"calc_second_order={not calc_second_order}, these indices come "
        f"from the wrong design. Give n_samples (or check "
        f"calc_second_order against sobol_sample) to settle it.")
    choice = {
        "what": "The Sobol design Y came from",
        "chosen": f"N={n_taken} {taken}",
        "how": "module",
        "basis": (f"calc_second_order={calc_second_order}; n_samples was "
                  f"not given, and Y's length fits both designs"),
        "alternatives": [f"N={n_other} {not_taken} "
                         f"(calc_second_order={not calc_second_order})"],
    }
    return warning, choice


def _morris_trajectories(X, Y, n_vars, n_trajectories):
    """The trajectories the sample matrix holds, checked against the
    design (GM-323): morris_sample gives T*(D+1) rows of D columns, one
    output per row."""
    if X.ndim != 2 or X.shape[1] != n_vars:
        raise ValueError(
            f"X must be the morris_sample matrix, one row per sample point "
            f"and one column per variable ({n_vars}); got shape "
            f"{tuple(X.shape)}.")
    _validate_model_output(Y, len(X))
    per = n_vars + 1
    if len(X) == 0 or len(X) % per:
        raise ValueError(
            f"X has {len(X)} row(s), which is not a Morris design for "
            f"{n_vars} variables: morris_sample gives trajectories*(D+1) = "
            f"trajectories*{per} rows.")
    counted = len(X) // per
    if n_trajectories is not None and int(n_trajectories) != counted:
        raise ValueError(
            f"n_trajectories={n_trajectories} does not match X: "
            f"{n_trajectories} trajectories of {n_vars} variables are "
            f"{int(n_trajectories) * per} rows, and X has {len(X)} "
            f"({counted} trajectories). Give the n_trajectories used in "
            f"morris_sample, or leave it out to count them from X.")
    return counted


def sobol_analyze(
    var_names,
    bounds,
    Y,
    n_samples=None,
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
    n_samples : int, optional
        The base sample size N given to sobol_sample. When given, Y must
        have N*(2D+2) values (N*(D+2) without second-order indices) or the
        call is refused. Left out, N is taken from the length of Y.
    calc_second_order : bool
        Whether the sample was made for second-order indices. Default True.
        It must match the sampling: it sets the design's row count.
    seed : int
        Random seed for resampling. Default 42.

    Returns
    -------
    SobolResult
        Sobol sensitivity indices. ``n_samples`` is the number of outputs
        analysed (the length of Y); ``n_base_samples`` is N.
    """
    _validate_problem(var_names, bounds)
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim != 1:
        raise ValueError(
            f"Y must be a flat list, one output per sample row; got shape "
            f"{tuple(Y_arr.shape)}.")
    n_base = _sobol_base_samples(len(Y_arr), len(var_names),
                                 calc_second_order, n_samples)
    ambiguity = _sobol_design_ambiguity(len(Y_arr), len(var_names),
                                        calc_second_order, n_samples)

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
        n_base_samples=n_base,
        n_vars=len(var_names),
        var_names=list(var_names),
        S1=list(Si['S1']),
        S1_conf=list(Si['S1_conf']),
        ST=list(Si['ST']),
        ST_conf=list(Si['ST_conf']),
        S2=s2,
        warnings=[ambiguity[0]] if ambiguity else [],
        choices=[ambiguity[1]] if ambiguity else [],
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
    n_trajectories=None,
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
    n_trajectories : int, optional
        The number of trajectories given to morris_sample. When given, X
        must have n_trajectories*(D+1) rows or the call is refused. Left
        out, the trajectories are counted from X.
    num_levels : int
        Number of grid levels used in sampling.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    MorrisResult
        Morris elementary effects (mu*, sigma). ``n_trajectories`` is the
        number counted from X: rows / (D+1).
    """
    _validate_problem(var_names, bounds)
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    n_counted = _morris_trajectories(X_arr, Y_arr, len(var_names),
                                     n_trajectories)

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
        n_trajectories=n_counted,
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
