"""
Foundry wrapper for pystra_agent.

Exposes three functions for LLM tool discovery:
- pystra_agent(method, params_json) — Execute a reliability analysis
- pystra_list_methods(category) — List available methods
- pystra_describe_method(method) — Get method metadata
"""

import json
from typing import Optional


def pystra_list_methods(category: Optional[str] = None) -> list:
    """
    List all available pystra agent methods.

    Args:
        category: Optional category filter (e.g., "Reliability")

    Returns:
        List of dicts with method metadata:
        [{"name": str, "category": str, "description": str}, ...]
    """
    methods = [
        {
            "name": "form_analysis",
            "category": "Reliability",
            "description": (
                "First Order Reliability Method (FORM). Computes reliability index "
                "and failure probability using linear approximation of limit state "
                "surface. Fast and provides sensitivity factors."
            ),
        },
        {
            "name": "sorm_analysis",
            "category": "Reliability",
            "description": (
                "Second Order Reliability Method (SORM). Improves upon FORM by "
                "including curvature information. More accurate for nonlinear "
                "limit state functions."
            ),
        },
        {
            "name": "monte_carlo_analysis",
            "category": "Reliability",
            "description": (
                "Crude Monte Carlo simulation. Provides exact (asymptotic) failure "
                "probability by direct sampling. Computationally expensive but "
                "works for any limit state function."
            ),
        },
    ]

    if category:
        methods = [m for m in methods if m["category"].lower() == category.lower()]

    return methods


def pystra_describe_method(method: str) -> dict:
    """
    Get detailed metadata for a specific method.

    Args:
        method: Method name (e.g., "form_analysis")

    Returns:
        Dict with:
        {
            "name": str,
            "category": str,
            "description": str,
            "parameters": {param_name: param_description, ...},
            "returns": str,
            "example": str,
        }

    Raises:
        ValueError: If method not found
    """
    descriptions = {
        "form_analysis": {
            "name": "form_analysis",
            "category": "Reliability",
            "description": (
                "First Order Reliability Method (FORM) for structural reliability analysis. "
                "FORM finds the design point (most probable failure point) and computes the "
                "reliability index beta as the shortest distance from the origin to the "
                "failure surface in standard normal space. Provides sensitivity factors "
                "indicating which variables most affect reliability."
            ),
            "parameters": {
                "variables": (
                    "List of dicts defining random variables. Each dict must have: "
                    "'name' (str), 'dist' (distribution type), and distribution-specific params. "
                    "Supported distributions: 'normal' (mean, stdv), 'lognormal' (mean, stdv), "
                    "'gumbel' (mean, stdv), 'uniform' (a, b), 'constant' (value), 'weibull' (mean, stdv), "
                    "'gamma_dist' (mean, stdv), 'beta' (mean, stdv)."
                ),
                "limit_state": (
                    "String expression for limit state function. Must evaluate to positive "
                    "for safe states, negative for failure. Example: 'R - S' for resistance R "
                    "and load S. Can use math functions: sqrt, log, exp, sin, cos, tan, etc."
                ),
                "correlation": (
                    "Optional correlation matrix as list of lists. Must be square (n_vars x n_vars), "
                    "symmetric, with 1.0 on diagonal. Example: [[1.0, 0.3], [0.3, 1.0]]."
                ),
            },
            "returns": (
                "FormResult with: beta (reliability index), pf (failure probability), "
                "alpha (sensitivity factors dict), design_point_x (physical space), "
                "design_point_u (standard normal space), convergence info, and methods "
                "summary() and to_dict()."
            ),
            "example": (
                '{"variables": [{"name": "R", "dist": "normal", "mean": 200, "stdv": 20}, '
                '{"name": "S", "dist": "normal", "mean": 100, "stdv": 30}], '
                '"limit_state": "R - S"}'
            ),
        },
        "sorm_analysis": {
            "name": "sorm_analysis",
            "category": "Reliability",
            "description": (
                "Second Order Reliability Method (SORM) for structural reliability analysis. "
                "SORM improves upon FORM by including second-order curvature information of the "
                "failure surface at the design point. This provides more accurate probability "
                "estimates for nonlinear limit state functions. Returns both FORM and SORM results."
            ),
            "parameters": {
                "variables": (
                    "List of variable dicts (same format as form_analysis). See form_analysis "
                    "for details on supported distributions and required fields."
                ),
                "limit_state": (
                    "Limit state function expression (same as form_analysis). Nonlinear functions "
                    "like 'R**2 - S**2' will show the benefit of SORM over FORM."
                ),
                "correlation": "Optional correlation matrix (same as form_analysis).",
            },
            "returns": (
                "SormResult with: beta_form (FORM reliability index), beta_breitung (SORM index), "
                "pf_form, pf_breitung (failure probabilities), kappa (principal curvatures), "
                "alpha (sensitivity factors), design points, and methods summary() and to_dict()."
            ),
            "example": (
                '{"variables": [{"name": "R", "dist": "lognormal", "mean": 200, "stdv": 40}, '
                '{"name": "S", "dist": "normal", "mean": 100, "stdv": 30}], '
                '"limit_state": "R**2 - S**2"}'
            ),
        },
        "monte_carlo_analysis": {
            "name": "monte_carlo_analysis",
            "category": "Reliability",
            "description": (
                "Crude Monte Carlo simulation for reliability analysis. Directly samples the "
                "joint distribution of random variables and counts failures to estimate failure "
                "probability. Provides exact (asymptotic) results but requires many samples for "
                "rare events. Use n_samples >= 100,000 for Pf ~ 0.001."
            ),
            "parameters": {
                "variables": "List of variable dicts (same format as form_analysis).",
                "limit_state": "Limit state function expression (same as form_analysis).",
                "n_samples": (
                    "Number of Monte Carlo samples (integer). Default 100,000. "
                    "Rule of thumb: need ~100 failures for 10% coefficient of variation. "
                    "For Pf ~ 1e-3, use n >= 100,000. For Pf ~ 1e-4, use n >= 1,000,000."
                ),
                "correlation": "Optional correlation matrix (same as form_analysis).",
            },
            "returns": (
                "MonteCarloResult with: beta (reliability index), pf (failure probability), "
                "n_samples, n_failures, cov_pf (coefficient of variation of Pf estimate), "
                "and methods summary() and to_dict()."
            ),
            "example": (
                '{"variables": [{"name": "R", "dist": "normal", "mean": 200, "stdv": 20}, '
                '{"name": "S", "dist": "normal", "mean": 100, "stdv": 30}], '
                '"limit_state": "R - S", "n_samples": 100000}'
            ),
        },
    }

    if method not in descriptions:
        available = ", ".join(descriptions.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")

    return descriptions[method]


def pystra_agent(method: str, params_json: str) -> dict:
    """
    Execute a pystra reliability analysis method.

    Args:
        method: Method name ("form_analysis", "sorm_analysis", "monte_carlo_analysis")
        params_json: JSON string with method parameters

    Returns:
        Dict with analysis results (from result.to_dict()), or {"error": str} if failed

    Example:
        >>> params = json.dumps({
        ...     "variables": [
        ...         {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
        ...         {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ...     ],
        ...     "limit_state": "R - S",
        ... })
        >>> result = pystra_agent("form_analysis", params)
        >>> print(result["beta"], result["pf"])
    """
    # Validate JSON first (before checking pystra availability for better error messages)
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}

    # Validate method name
    valid_methods = {"form_analysis", "sorm_analysis", "monte_carlo_analysis"}
    if method not in valid_methods:
        return {"error": f"Unknown method '{method}'. Valid: {', '.join(valid_methods)}"}

    # Check if pystra is available
    from pystra_agent import has_pystra
    if not has_pystra():
        return {
            "error": (
                "pystra is not installed. Install via: pip install pystra. "
                "Note: pystra requires numpy and scipy."
            )
        }

    # Import analysis functions (only after confirming pystra is available)
    from pystra_agent import analyze_form, analyze_sorm, analyze_monte_carlo

    # Extract common parameters
    variables = params.get("variables")
    limit_state = params.get("limit_state")
    correlation = params.get("correlation")

    if not variables:
        return {"error": "Missing required parameter 'variables'"}
    if not limit_state:
        return {"error": "Missing required parameter 'limit_state'"}

    # Execute requested method
    try:
        if method == "form_analysis":
            result = analyze_form(
                variables=variables,
                limit_state=limit_state,
                correlation=correlation,
            )

        elif method == "sorm_analysis":
            result = analyze_sorm(
                variables=variables,
                limit_state=limit_state,
                correlation=correlation,
            )

        elif method == "monte_carlo_analysis":
            n_samples = params.get("n_samples", 100000)
            result = analyze_monte_carlo(
                variables=variables,
                limit_state=limit_state,
                n_samples=n_samples,
                correlation=correlation,
            )

        # Convert result to dict and return
        return result.to_dict()

    except Exception as e:
        return {"error": f"{method} failed: {str(e)}"}


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # List all methods
    print("Available methods:")
    for m in pystra_list_methods():
        print(f"  - {m['name']}: {m['description'][:60]}...")
    print()

    # Describe a method
    desc = pystra_describe_method("form_analysis")
    print(f"Method: {desc['name']}")
    print(f"Description: {desc['description'][:80]}...")
    print(f"Example: {desc['example']}")
    print()

    # Run FORM analysis
    from pystra_agent import has_pystra
    if has_pystra():
        params = json.dumps({
            "variables": [
                {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
                {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
            ],
            "limit_state": "R - S",
        })
        result = pystra_agent("form_analysis", params)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"FORM Results:")
            print(f"  Beta = {result['beta']:.3f}")
            print(f"  Pf = {result['pf']:.6e}")
            print(f"  Converged = {result['converged']}")
    else:
        print("pystra not available — install via: pip install pystra")
