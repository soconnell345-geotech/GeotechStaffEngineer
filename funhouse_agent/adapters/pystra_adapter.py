"""Pystra adapter — FORM/SORM/Monte Carlo structural reliability analysis."""

from funhouse_agent.adapters import clean_result


def _run_form_reliability(params: dict) -> dict:
    from pystra_agent import analyze_form, has_pystra

    if not has_pystra():
        return {"error": "pystra is not installed. Install via: pip install pystra"}

    result = analyze_form(
        variables=params["variables"],
        limit_state=params["limit_state"],
        correlation=params.get("correlation"),
    )
    return clean_result(result.to_dict())


def _run_sorm_reliability(params: dict) -> dict:
    from pystra_agent import analyze_sorm, has_pystra

    if not has_pystra():
        return {"error": "pystra is not installed. Install via: pip install pystra"}

    result = analyze_sorm(
        variables=params["variables"],
        limit_state=params["limit_state"],
        correlation=params.get("correlation"),
    )
    return clean_result(result.to_dict())


def _run_monte_carlo_reliability(params: dict) -> dict:
    from pystra_agent import analyze_monte_carlo, has_pystra

    if not has_pystra():
        return {"error": "pystra is not installed. Install via: pip install pystra"}

    result = analyze_monte_carlo(
        variables=params["variables"],
        limit_state=params["limit_state"],
        n_samples=params.get("n_samples", 100000),
        correlation=params.get("correlation"),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "form_reliability": _run_form_reliability,
    "sorm_reliability": _run_sorm_reliability,
    "monte_carlo_reliability": _run_monte_carlo_reliability,
}

METHOD_INFO = {
    "form_reliability": {
        "category": "Reliability",
        "brief": "First Order Reliability Method (FORM) — find design point and reliability index.",
        "parameters": {
            "variables": {
                "type": "array",
                "brief": "List of dicts with keys: name (str), dist (normal/lognormal/gumbel/uniform/constant/weibull/gamma_dist/beta), mean (float), stdv (float), value (float, for constant).",
            },
            "limit_state": {
                "type": "str",
                "brief": "Limit state expression string. Positive = safe, negative = failure. Example: 'R - S'.",
            },
            "correlation": {
                "type": "array",
                "brief": "Optional correlation matrix as list of lists (symmetric, positive definite, 1.0 on diagonal).",
                "default": None,
            },
        },
        "returns": {
            "beta": "Reliability index (dimensionless).",
            "pf": "Probability of failure.",
            "alpha": "Sensitivity factors per variable.",
            "design_point_x": "Design point in physical space.",
            "design_point_u": "Design point in standard normal space.",
            "n_iterations": "Number of iterations to convergence.",
            "n_function_calls": "Number of limit state evaluations.",
            "converged": "Whether analysis converged.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
    "sorm_reliability": {
        "category": "Reliability",
        "brief": "Second Order Reliability Method (SORM) — improved probability with curvature correction.",
        "parameters": {
            "variables": {
                "type": "array",
                "brief": "List of dicts with keys: name, dist, mean, stdv (same as FORM).",
            },
            "limit_state": {
                "type": "str",
                "brief": "Limit state expression string. Positive = safe, negative = failure.",
            },
            "correlation": {
                "type": "array",
                "brief": "Optional correlation matrix as list of lists.",
                "default": None,
            },
        },
        "returns": {
            "beta_form": "FORM reliability index.",
            "beta_breitung": "SORM reliability index (Breitung approximation).",
            "pf_breitung": "SORM probability of failure.",
            "pf_form": "FORM probability of failure.",
            "kappa": "Principal curvatures at design point.",
            "alpha": "Sensitivity factors per variable.",
            "design_point_x": "Design point in physical space.",
            "design_point_u": "Design point in standard normal space.",
            "n_iterations": "Number of iterations.",
            "n_function_calls": "Number of limit state evaluations.",
            "converged": "Whether analysis converged.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
    "monte_carlo_reliability": {
        "category": "Reliability",
        "brief": "Crude Monte Carlo reliability analysis — estimate failure probability by random sampling.",
        "parameters": {
            "variables": {
                "type": "array",
                "brief": "List of dicts with keys: name, dist, mean, stdv (same as FORM).",
            },
            "limit_state": {
                "type": "str",
                "brief": "Limit state expression string. Positive = safe, negative = failure.",
            },
            "n_samples": {
                "type": "int",
                "brief": "Number of Monte Carlo samples.",
                "default": 100000,
            },
            "correlation": {
                "type": "array",
                "brief": "Optional correlation matrix as list of lists.",
                "default": None,
            },
        },
        "returns": {
            "beta": "Reliability index.",
            "pf": "Probability of failure.",
            "n_samples": "Total Monte Carlo samples.",
            "n_failures": "Number of failure samples.",
            "cov_pf": "Coefficient of variation of Pf estimate.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
}
