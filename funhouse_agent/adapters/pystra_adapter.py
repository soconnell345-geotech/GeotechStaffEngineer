"""Pystra adapter — FORM/SORM/Monte Carlo structural reliability analysis."""

from funhouse_agent.adapters import clean_result, require_params

#: What a call says when pystra is missing; a pystra that is installed but
#: fails to import says its import's own error instead (BACKLOG INF-54).
_PYSTRA_MISSING = "pystra is not installed. Install via: pip install pystra"


def _run_form_reliability(params: dict) -> dict:
    from pystra_agent import analyze_form, has_pystra, pystra_import_error

    if not has_pystra():
        return {"error": pystra_import_error() or _PYSTRA_MISSING}

    require_params(params, ["variables", "limit_state"], method="form_analysis")
    result = analyze_form(
        variables=params["variables"],
        limit_state=params["limit_state"],
        correlation=params.get("correlation"),
    )
    return clean_result(result.to_dict())


def _run_sorm_reliability(params: dict) -> dict:
    from pystra_agent import analyze_sorm, has_pystra, pystra_import_error

    if not has_pystra():
        return {"error": pystra_import_error() or _PYSTRA_MISSING}

    require_params(params, ["variables", "limit_state"], method="sorm_analysis")
    result = analyze_sorm(
        variables=params["variables"],
        limit_state=params["limit_state"],
        correlation=params.get("correlation"),
    )
    return clean_result(result.to_dict())


def _run_monte_carlo_reliability(params: dict) -> dict:
    from pystra_agent import analyze_monte_carlo, has_pystra, pystra_import_error

    if not has_pystra():
        return {"error": pystra_import_error() or _PYSTRA_MISSING}

    require_params(params, ["variables", "limit_state"],
                   method="monte_carlo_analysis")
    result = analyze_monte_carlo(
        variables=params["variables"],
        limit_state=params["limit_state"],
        n_samples=params.get("n_samples", 100000),
        correlation=params.get("correlation"),
        seed=params.get("seed"),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "form_analysis": _run_form_reliability,
    "sorm_analysis": _run_sorm_reliability,
    "monte_carlo_analysis": _run_monte_carlo_reliability,
}

METHOD_INFO = {
    "form_analysis": {
        "category": "Reliability",
        "brief": "First Order Reliability Method (FORM) — find design point and reliability index.",
        "parameters": {
            "variables": {
                "type": "array",
                "required": True,
                "description": "List of dicts with keys: name (str), dist (normal/lognormal/gumbel/uniform/constant/weibull/gamma_dist/beta), mean (float), stdv (float), value (float, for constant). uniform takes a (lower) and b (upper) instead of mean/stdv.",
            },
            "limit_state": {
                "type": "str",
                "required": True,
                "description": "Limit state expression string. Positive = safe, negative = failure. Example: 'R - S'.",
            },
            "correlation": {
                "type": "array",
                "required": False,
                "description": "Optional correlation matrix as list of lists (symmetric, positive definite, 1.0 on diagonal).",
                "default": None,
            },
        },
        "returns": {
            "beta": "Reliability index (dimensionless).",
            "pf": "Probability of failure.",
            "alpha": "Sensitivity factors per variable.",
            "design_point_x": "Design point in physical space.",
            "design_point_u": "Design point in standard normal space.",
            "n_iterations": "Number of iterations run.",
            "n_function_calls": "Limit state evaluations: pystra's own count (2.x), or on 1.x an estimate (iterations x number of variables); n_function_calls_counted says which.",
            "n_function_calls_counted": "True: n_function_calls is pystra's count. False: the estimate.",
            "converged": "pystra's own convergence flag (pystra 2.x); on 1.x, which keeps none, whether it stopped before its iteration limit. Not inferred from beta: a converged run can have beta < 0 (mean point in the failure domain). When false on 2.x, beta, pf, alpha and the design point are null/empty (pystra gives none).",
            "converged_source": "Which of the two the converged flag is.",
            "message": "pystra's status message.",
            "warnings": "What the run could not give (not converged, a quantity pystra did not return, an estimated count). Empty for a clean run.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
    "sorm_analysis": {
        "category": "Reliability",
        "brief": "Second Order Reliability Method (SORM) — improved probability with curvature correction.",
        "parameters": {
            "variables": {
                "type": "array",
                "required": True,
                "description": "List of dicts with keys: name, dist, mean, stdv (same as FORM); uniform takes a and b, constant takes value.",
            },
            "limit_state": {
                "type": "str",
                "required": True,
                "description": "Limit state expression string. Positive = safe, negative = failure.",
            },
            "correlation": {
                "type": "array",
                "required": False,
                "description": "Optional correlation matrix as list of lists.",
                "default": None,
            },
        },
        "returns": {
            "beta_form": "FORM reliability index.",
            "beta_breitung": "SORM reliability index (Breitung approximation). Null when the second-order stage failed (sorm_failed): never FORM's value.",
            "pf_breitung": "SORM probability of failure. Null when sorm_failed.",
            "sorm_failed": "True when there is no second-order result (Breitung's formula undefined for the fitted curvatures, e.g. beta < 0, or FORM did not converge); sorm_message says why.",
            "sorm_message": "pystra's message for the second-order stage.",
            "form_converged": "FORM's convergence flag (as form_analysis's converged).",
            "sorm_converged": "Whether the second-order fit gave a value.",
            "pf_form": "FORM probability of failure.",
            "kappa": "Principal curvatures at design point.",
            "alpha": "Sensitivity factors per variable.",
            "design_point_x": "Design point in physical space.",
            "design_point_u": "Design point in standard normal space.",
            "n_iterations": "Number of FORM iterations.",
            "n_function_calls": "Limit state evaluations: pystra's own count (2.x), or on 1.x an estimate (iterations x number of variables); n_function_calls_counted says which. FORM and SORM together.",
            "n_function_calls_counted": "True: n_function_calls is pystra's count. False: the estimate.",
            "converged": "form_converged and sorm_converged, both pystra's own flags (not inferred from beta).",
            "converged_source": "Where form_converged comes from (pystra 2.x's flag, or 1.x's stopping rule).",
            "message": "pystra's FORM status message.",
            "warnings": "What the run could not give (a failed SORM, FORM not converged, an estimated count). Empty for a clean run.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
    "monte_carlo_analysis": {
        "category": "Reliability",
        "brief": "Crude Monte Carlo reliability analysis — estimate failure probability by random sampling.",
        "parameters": {
            "variables": {
                "type": "array",
                "required": True,
                "description": "List of dicts with keys: name, dist, mean, stdv (same as FORM); uniform takes a and b, constant takes value.",
            },
            "limit_state": {
                "type": "str",
                "required": True,
                "description": "Limit state expression string. Positive = safe, negative = failure.",
            },
            "n_samples": {
                "type": "int",
                "required": False,
                "description": "Number of Monte Carlo samples. All of them are run (pystra's own early stop at a 5% COV of pf is off).",
                "default": 100000,
            },
            "seed": {
                "type": "int",
                "required": False,
                "description": "Seed for the sampling (a non-negative integer), local to this call. Give it to repeat a run.",
            },
            "correlation": {
                "type": "array",
                "required": False,
                "description": "Optional correlation matrix as list of lists.",
                "default": None,
            },
        },
        "returns": {
            "beta": "Reliability index, -PHI^-1(pf). Null when no sample (or every sample) failed.",
            "pf": "Probability of failure: n_failures / n_samples.",
            "n_samples": "Monte Carlo samples run.",
            "n_samples_requested": "Samples asked for.",
            "n_failures": "Number of failure samples.",
            "cov_pf": "Coefficient of variation of the pf estimate, sqrt((1-pf)/(n pf)). Null when no sample failed (undefined, not zero).",
            "pf_upper95_no_failures": "With no failure: the upper end of the two-sided 95% Clopper-Pearson interval on pf, 1 - 0.025^(1/n) (about 3.7/n). Null otherwise.",
            "seed": "The seed used; null = none given (the run is not repeatable).",
            "warnings": "No failure in the samples; no seed given.",
            "limit_state_expr": "Limit state expression used.",
            "n_variables": "Number of random variables.",
        },
    },
}


# What a left-out seed stands for (GM-322; as reliability.monte_carlo's).
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
declare_when_absent(METHOD_INFO, {
    "monte_carlo_analysis": {
        "seed": ("a fresh random seed",
                 "the samples, and so the result, differ from run to run; "
                 "give seed to repeat a run")},
})
