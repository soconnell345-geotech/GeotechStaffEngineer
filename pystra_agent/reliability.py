"""
Reliability analysis functions using pystra.

This module provides wrappers for FORM, SORM, and Monte Carlo reliability
analysis methods from the pystra library.
"""

import numpy as np
from typing import Optional

from pystra_agent.results import FormResult, SormResult, MonteCarloResult
from pystra_agent.pystra_utils import (
    has_pystra,
    _apply_numpy2_patches,
    _compile_limit_state,
    _build_stochastic_model,
)


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
    if not has_pystra():
        raise ImportError(
            "pystra is required for FORM analysis. Install via: pip install pystra"
        )

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

    # Configure analysis options
    options = pystra.AnalysisOptions()
    options.setPrintOutput(False)

    # Run FORM
    form = pystra.Form(sm, lsf, options)
    try:
        form.run()
    except Exception as e:
        raise ValueError(f"FORM analysis failed: {str(e)}")

    # Extract results
    beta = float(form.getBeta())

    # Extract Pf (returns ndarray, need scalar)
    pf_array = form.getFailure()
    pf = float(pf_array.flat[0])

    # Identify random (non-constant) variable names — pystra excludes constants
    # from alpha and design point arrays
    random_names = [
        v.get("name") for v in variables
        if v.get("dist", "").lower() != "constant"
    ]

    # Extract sensitivity factors
    try:
        alpha_raw = form.getAlpha()
        if isinstance(alpha_raw, dict):
            alpha = {k: float(v) for k, v in alpha_raw.items()}
        else:
            alpha = {
                name: float(np.asarray(alpha_raw[i]).flat[0])
                for i, name in enumerate(random_names)
            }
    except Exception:
        alpha = {}

    # Extract design point in physical space
    try:
        x_array = form.getDesignPoint(uspace=False)
        design_point_x = {
            name: float(np.asarray(x_array[i]).flat[0])
            for i, name in enumerate(random_names)
        }
        # Add constants at their fixed values
        for v in variables:
            if v.get("dist", "").lower() == "constant":
                design_point_x[v["name"]] = float(v["value"])
    except Exception:
        design_point_x = {}

    # Extract design point in standard normal space
    try:
        u_array = form.getDesignPoint(uspace=True)
        design_point_u = {
            name: float(np.asarray(u_array[i]).flat[0])
            for i, name in enumerate(random_names)
        }
    except Exception:
        design_point_u = {}

    # Extract iteration info
    try:
        n_iterations = int(form.i)
    except Exception:
        n_iterations = 0

    # Function calls not directly available in pystra, estimate conservatively
    n_function_calls = n_iterations * len(var_names) if n_iterations > 0 else 0

    # Check convergence (beta should be positive and finite)
    converged = np.isfinite(beta) and beta > 0 and pf > 0

    return FormResult(
        beta=beta,
        pf=pf,
        alpha=alpha,
        design_point_x=design_point_x,
        design_point_u=design_point_u,
        n_iterations=n_iterations,
        n_function_calls=n_function_calls,
        converged=converged,
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
    if not has_pystra():
        raise ImportError(
            "pystra is required for SORM analysis. Install via: pip install pystra"
        )

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

    # Configure analysis options
    options = pystra.AnalysisOptions()
    options.setPrintOutput(False)

    # Run FORM first (required for SORM)
    form = pystra.Form(sm, lsf, options)
    try:
        form.run()
    except Exception as e:
        raise ValueError(f"FORM analysis failed: {str(e)}")

    # Run SORM using FORM result
    sorm = pystra.Sorm(sm, lsf, options, form)
    try:
        sorm.run()
    except Exception as e:
        raise ValueError(f"SORM analysis failed: {str(e)}")

    # Extract FORM results
    beta_form = float(form.getBeta())
    pf_form_array = form.getFailure()
    pf_form = float(pf_form_array.flat[0])

    # Extract sensitivity factors from FORM
    try:
        alpha_raw = form.getAlpha()
        if isinstance(alpha_raw, dict):
            alpha = {k: float(v) for k, v in alpha_raw.items()}
        else:
            alpha = {name: float(alpha_raw[i]) for i, name in enumerate(var_names)}
    except Exception:
        alpha = {}

    # Extract design point
    try:
        x_array = form.getX()
        design_point_x = {name: float(x_array[i]) for i, name in enumerate(var_names)}
    except Exception:
        design_point_x = {}

    try:
        u_array = form.getU()
        design_point_u = {name: float(u_array[i]) for i, name in enumerate(var_names)}
    except Exception:
        design_point_u = {}

    # Extract SORM results (Breitung approximation)
    try:
        beta_breitung = float(sorm.betag_breitung)
    except Exception:
        beta_breitung = beta_form  # Fallback to FORM

    try:
        pf_breitung = float(sorm.pf2_breitung)
    except Exception:
        pf_breitung = pf_form  # Fallback to FORM

    # Extract principal curvatures
    try:
        kappa_raw = sorm.kappa
        if hasattr(kappa_raw, '__iter__'):
            kappa = [float(k) for k in kappa_raw]
        else:
            kappa = [float(kappa_raw)]
    except Exception:
        kappa = []

    # Extract iteration info
    try:
        n_iterations = int(form.i)
    except Exception:
        n_iterations = 0

    n_function_calls = n_iterations * len(var_names) if n_iterations > 0 else 0

    # Check convergence
    converged = (
        np.isfinite(beta_form) and beta_form > 0 and
        np.isfinite(beta_breitung) and beta_breitung > 0
    )

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
        converged=converged,
        limit_state_expr=limit_state,
        n_variables=len(var_names),
    )


def analyze_monte_carlo(
    variables: list,
    limit_state: str,
    n_samples: int = 100000,
    correlation: Optional[list] = None,
) -> MonteCarloResult:
    """
    Perform Crude Monte Carlo reliability analysis.

    Monte Carlo estimates failure probability by sampling the joint distribution
    and counting failures. Provides exact (asymptotic) results but requires many
    samples for rare events (small failure probabilities).

    Args:
        variables: List of variable specification dicts (same as analyze_form)
        limit_state: Limit state function expression string
        n_samples: Number of Monte Carlo samples (default 100,000)
        correlation: Optional correlation matrix

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
    if not has_pystra():
        raise ImportError(
            "pystra is required for Monte Carlo analysis. Install via: pip install pystra"
        )

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

    # Configure analysis options
    options = pystra.AnalysisOptions()
    options.setPrintOutput(False)
    options.setSamples(n_samples)

    # Run Monte Carlo
    mc = pystra.CrudeMonteCarlo(options, lsf, sm)
    try:
        mc.run()
    except Exception as e:
        raise ValueError(f"Monte Carlo analysis failed: {str(e)}")

    # Extract results
    beta = float(mc.getBeta())

    # Extract Pf (returns ndarray, need scalar)
    pf_array = mc.getFailure()
    pf = float(pf_array.flat[0])

    # Compute number of failures
    n_failures = int(round(pf * n_samples))

    # Compute coefficient of variation of Pf estimate
    # COV_pf = sqrt((1-pf)/(n*pf)) for binomial sampling
    if pf > 0 and n_samples > 0:
        cov_pf = np.sqrt((1.0 - pf) / (n_samples * pf))
    else:
        cov_pf = 0.0

    return MonteCarloResult(
        beta=beta,
        pf=pf,
        n_samples=n_samples,
        n_failures=n_failures,
        cov_pf=cov_pf,
        limit_state_expr=limit_state,
        n_variables=len(var_names),
    )
