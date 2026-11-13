"""
Reliability analysis functions using pystra.

This module provides wrappers for FORM, SORM, and Monte Carlo reliability
analysis methods from the pystra library.
"""

import numpy as np
from typing import Optional

from pystra_agent.results import FormResult, SormResult, MonteCarloResult
from pystra_agent.pystra_utils import (
    pystra_import_error,
    _apply_numpy2_patches,
    _compile_limit_state,
    _build_stochastic_model,
    run_form,
    run_monte_carlo,
)


def _named(values, names, what, notes) -> dict:
    """pystra's array ``values`` keyed by variable name.

    pystra gives no array for a run that did not converge, and a length
    that does not match the random variables would pair values with the
    wrong names. Either is said in ``notes`` and nothing is returned for
    that quantity; no exception is swallowed (GM-322).
    """
    if values is None:
        notes.append(f"pystra returned no {what} for this run.")
        return {}
    flat = np.ravel(np.asarray(values, dtype=float))
    if flat.size != len(names):
        notes.append(
            f"pystra returned {flat.size} value(s) of {what} for "
            f"{len(names)} random variable(s) {list(names)}; they cannot "
            f"be matched to the variables and are not reported.")
        return {}
    return {name: float(v) for name, v in zip(names, flat)}


def _function_calls(run, n_variables, notes):
    """(count, counted?) of limit-state evaluations.

    pystra 2.x counts them; 1.x does not, and the figure is then the old
    estimate iterations x variables, flagged as such.
    """
    if run.n_evaluations is not None:
        return int(run.n_evaluations), True
    n = run.iterations * n_variables if run.iterations > 0 else 0
    notes.append(
        "n_function_calls is an estimate (iterations x number of "
        "variables): this pystra version does not count limit-state "
        "evaluations.")
    return n, False


def analyze_form(
    variables: list,
    limit_state: str,
    correlation: Optional[list] = None,
) -> FormResult:
    """
    Perform First Order Reliability Method (FORM) analysis.

    FORM finds the design point (most probable failure point) and computes the
    reliability index as the shortest distance from the origin to the failure
    surface in standard normal space.

    Args:
        variables: List of dicts, each with keys:
            - name: str (variable name)
            - dist: str (distribution type: normal, lognormal, gumbel, uniform,
                constant, weibull, gamma_dist, beta)
            - mean: float (for most distributions)
            - stdv: float (for most distributions)
            - value: float (for constant distribution)
            Additional distribution-specific parameters as needed.
        limit_state: String expression for limit state function. Should return
            positive for safe states, negative for failure. Example: "R - S"
            for resistance R and load S.
        correlation: Optional correlation matrix as list of lists. Must be
            symmetric positive definite with 1.0 on diagonal.

    Returns:
        FormResult with reliability index, failure probability, sensitivity
            factors, and design point.

    Raises:
        ImportError: If pystra not available
        ValueError: If inputs invalid or analysis fails to converge

    Example:
        >>> variables = [
        ...     {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
        ...     {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ... ]
        >>> result = analyze_form(variables, "R - S")
        >>> print(f"Beta = {result.beta:.2f}, Pf = {result.pf:.2e}")
    """
    error = pystra_import_error()
    if error:
        raise ImportError(f"pystra is required for FORM analysis. {error}")

    # Apply numpy 2.x compatibility patches
    _apply_numpy2_patches()

    import pystra

    # Validate inputs
    if not variables:
        raise ValueError("Variables list cannot be empty")
    if not limit_state or not limit_state.strip():
        raise ValueError("Limit state expression cannot be empty")

    # Build stochastic model
    sm, pystra_vars, var_names = _build_stochastic_model(variables, correlation)

    # Compile limit state function
    lsf_func = _compile_limit_state(limit_state, var_names)
    lsf = pystra.LimitState(lsf_func)

    # Run FORM
    try:
        run = run_form(sm, lsf)
    except Exception as e:
        raise ValueError(f"FORM analysis failed: {str(e)}")

    notes = []
    beta = run.beta
    pf = run.pf
    converged = bool(run.converged)
    if not converged:
        notes.append(
            f"FORM did not converge ({run.message}; "
            f"{run.iterations} iterations)."
            + (" pystra reports no reliability index, probability or "
               "design point for an unconverged run."
               if beta is None else
               " beta, pf, alpha and the design point are the last "
               "iterate's, not a solution."))

    # Identify random (non-constant) variable names — pystra excludes constants
    # from alpha and design point arrays
    random_names = [
        v.get("name") for v in variables
        if v.get("dist", "").lower() != "constant"
    ]

    alpha = _named(run.alpha, random_names, "sensitivity factors (alpha)",
                   notes)
    design_point_x = _named(run.x, random_names,
                            "design point (physical space)", notes)
    if design_point_x:
        # Add constants at their fixed values
        for v in variables:
            if v.get("dist", "").lower() == "constant":
                design_point_x[v["name"]] = float(v["value"])
    design_point_u = _named(run.u, random_names,
                            "design point (standard normal space)", notes)

    n_iterations = run.iterations
    n_function_calls, counted = _function_calls(run, len(var_names), notes)

    return FormResult(
        beta=beta,
        pf=pf,
        alpha=alpha,
        design_point_x=design_point_x,
        design_point_u=design_point_u,
        n_iterations=n_iterations,
        n_function_calls=n_function_calls,
        n_function_calls_counted=counted,
        converged=converged,
        converged_source=run.converged_source,
        message=run.message,
        warnings=notes,
        limit_state_expr=limit_state,
        n_variables=len(var_names),
    )


def analyze_sorm(
    variables: list,
    limit_state: str,
    correlation: Optional[list] = None,
) -> SormResult:
    """
    Perform Second Order Reliability Method (SORM) analysis.

    SORM improves upon FORM by including second-order curvature information
    of the failure surface. This provides better probability estimates for
    nonlinear limit state functions.

    Args:
        variables: List of variable specification dicts (same as analyze_form)
        limit_state: Limit state function expression string
        correlation: Optional correlation matrix

    Returns:
        SormResult with FORM and SORM reliability indices, curvatures, and
            other analysis results.

    Raises:
        ImportError: If pystra not available
        ValueError: If inputs invalid or analysis fails

    Example:
        >>> variables = [
        ...     {"name": "R", "dist": "lognormal", "mean": 200, "stdv": 40},
        ...     {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ... ]
        >>> result = analyze_sorm(variables, "R**2 - S**2")
        >>> print(f"FORM beta = {result.beta_form:.2f}")
        >>> print(f"SORM beta = {result.beta_breitung:.2f}")
    """
    error = pystra_import_error()
    if error:
        raise ImportError(f"pystra is required for SORM analysis. {error}")

    # Apply numpy 2.x compatibility patches
    _apply_numpy2_patches()

    import pystra

    # Validate inputs
    if not variables:
        raise ValueError("Variables list cannot be empty")
    if not limit_state or not limit_state.strip():
        raise ValueError("Limit state expression cannot be empty")

    # Build stochastic model
    sm, pystra_vars, var_names = _build_stochastic_model(variables, correlation)

    # Compile limit state function
    lsf_func = _compile_limit_state(limit_state, var_names)
    lsf = pystra.LimitState(lsf_func)

    # Run FORM, then SORM on its design point
    try:
        run = run_form(sm, lsf, with_sorm=True)
    except Exception as e:
        raise ValueError(f"SORM analysis failed: {str(e)}")

    notes = []
    beta_form = run.beta
    pf_form = run.pf
    form_converged = bool(run.converged)
    if not form_converged:
        notes.append(
            f"FORM did not converge ({run.message}; "
            f"{run.iterations} iterations)."
            + (" pystra reports no reliability index, probability or "
               "design point for an unconverged run."
               if beta_form is None else
               " The FORM values are the last iterate's, not a solution."))

    # pystra leaves constants out of alpha and the design point arrays
    random_names = [
        v.get("name") for v in variables
        if v.get("dist", "").lower() != "constant"
    ]

    alpha = _named(run.alpha, random_names, "sensitivity factors (alpha)",
                   notes)
    design_point_x = _named(run.x, random_names,
                            "design point (physical space)", notes)
    design_point_u = _named(run.u, random_names,
                            "design point (standard normal space)", notes)

    # SORM (Breitung). A failed second-order stage returns NO second-order
    # numbers: the FORM values are never reported under the SORM names.
    sorm_converged = bool(run.sorm_converged)
    beta_breitung = run.beta_breitung if sorm_converged else None
    pf_breitung = run.pf_breitung if sorm_converged else None
    if beta_breitung is None or pf_breitung is None:
        sorm_converged = False
        beta_breitung = pf_breitung = None
    if not sorm_converged:
        notes.append(
            f"SORM FAILED: {run.sorm_message or 'pystra gave no value'}. "
            f"beta_breitung and pf_breitung are not reported (null); "
            f"beta_form and pf_form are FORM's first-order values only.")
    if run.kappa is None:
        kappa = []
        if sorm_converged:
            notes.append("pystra returned no principal curvatures.")
    else:
        kappa = [float(k) for k in np.ravel(run.kappa)]

    n_iterations = run.iterations
    n_function_calls, counted = _function_calls(run, len(var_names), notes)

    return SormResult(
        beta_form=beta_form,
        beta_breitung=beta_breitung,
        pf_breitung=pf_breitung,
        kappa=kappa,
        pf_form=pf_form,
        alpha=alpha,
        design_point_x=design_point_x,
        design_point_u=design_point_u,
        n_iterations=n_iterations,
        n_function_calls=n_function_calls,
        n_function_calls_counted=counted,
        converged=form_converged and sorm_converged,
        form_converged=form_converged,
        sorm_converged=sorm_converged,
        sorm_failed=not sorm_converged,
        converged_source=run.converged_source,
        message=run.message,
        sorm_message=run.sorm_message,
        warnings=notes,
        limit_state_expr=limit_state,
        n_variables=len(var_names),
    )


def analyze_monte_carlo(
    variables: list,
    limit_state: str,
    n_samples: int = 100000,
    correlation: Optional[list] = None,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Perform Crude Monte Carlo reliability analysis.

    Monte Carlo estimates failure probability by sampling the joint distribution
    and counting failures. Provides exact (asymptotic) results but requires many
    samples for rare events (small failure probabilities).

    Args:
        variables: List of variable specification dicts (same as analyze_form)
        limit_state: Limit state function expression string
        n_samples: Number of Monte Carlo samples (default 100,000). All
            of them are run (pystra's own early stop at a 5 % coefficient
            of variation of pf is switched off).
        correlation: Optional correlation matrix
        seed: Optional integer seed, local to this call (numpy's global
            generator is neither read nor changed). None: fresh entropy,
            so the result differs from run to run.

    Returns:
        MonteCarloResult with reliability index, failure probability, and
            sampling statistics.

    Raises:
        ImportError: If pystra not available
        ValueError: If inputs invalid or n_samples <= 0

    Example:
        >>> variables = [
        ...     {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
        ...     {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ... ]
        >>> result = analyze_monte_carlo(variables, "R - S", n_samples=1000000)
        >>> print(f"Beta = {result.beta:.2f}, Pf = {result.pf:.2e}")
        >>> print(f"COV = {result.cov_pf:.3f}")
    """
    error = pystra_import_error()
    if error:
        raise ImportError(f"pystra is required for Monte Carlo analysis. {error}")

    # Apply numpy 2.x compatibility patches
    _apply_numpy2_patches()

    import pystra
    from scipy.stats import norm

    # Validate inputs
    if not variables:
        raise ValueError("Variables list cannot be empty")
    if not limit_state or not limit_state.strip():
        raise ValueError("Limit state expression cannot be empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    # Build stochastic model
    sm, pystra_vars, var_names = _build_stochastic_model(variables, correlation)

    # Compile limit state function
    lsf_func = _compile_limit_state(limit_state, var_names)
    lsf = pystra.LimitState(lsf_func)

    # Run Monte Carlo
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) \
                or seed < 0:
            raise ValueError(
                f"seed must be a non-negative integer, got {seed!r}")
        seed = int(seed)
    try:
        run = run_monte_carlo(sm, lsf, n_samples, seed=seed)
    except Exception as e:
        raise ValueError(f"Monte Carlo analysis failed: {str(e)}")

    notes = []
    pf = run.pf
    n_run = run.n_samples if run.n_samples is not None else n_samples
    if n_run != n_samples:
        notes.append(
            f"pystra ran {n_run} samples of the {n_samples} asked for; "
            f"n_samples, n_failures and cov_pf are for the {n_run} run.")

    # pf is the count over the samples run, so the count is pf x n
    n_failures = int(round(pf * n_run))

    # COV of the pf estimate, sqrt((1-pf)/(n*pf)) for binomial sampling.
    # With no failure it is undefined (not 0), and so is beta: pf = 0 is
    # the count; the exact two-sided 95 % binomial interval for it runs up
    # to 1 - 0.025^(1/n), about 3.7/n.
    if n_failures > 0 and pf > 0:
        cov_pf = float(np.sqrt((1.0 - pf) / (n_run * pf)))
        beta = run.beta if np.isfinite(run.beta) else None
        pf_upper95 = None
    else:
        cov_pf = None
        beta = None
        pf_upper95 = float(1.0 - 0.025 ** (1.0 / n_run))
        notes.append(
            f"No failure in {n_run} samples: pf = 0 is the count, not a "
            f"certainty. beta and cov_pf are undefined and not reported; "
            f"the exact 95 % binomial interval on pf is 0 to "
            f"{pf_upper95:.3g} (1 - 0.025^(1/n)). Run more samples, or use "
            f"FORM/SORM for a small probability.")
    if beta is None and n_failures > 0:
        notes.append(
            "Every sample failed (pf = 1): beta is not finite and is not "
            "reported.")
    if seed is None:
        notes.append(
            "No seed was given: the samples, and so pf, differ from run "
            "to run. Give seed to repeat a run.")

    return MonteCarloResult(
        beta=beta,
        pf=pf,
        n_samples=n_run,
        n_samples_requested=n_samples,
        n_failures=n_failures,
        cov_pf=cov_pf,
        pf_upper95_no_failures=pf_upper95,
        seed=seed,
        warnings=notes,
        limit_state_expr=limit_state,
        n_variables=len(var_names),
    )
