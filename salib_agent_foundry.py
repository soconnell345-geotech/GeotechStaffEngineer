"""
SALib Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. salib_agent           - Run a sensitivity analysis
  2. salib_list_methods    - Browse available methods
  3. salib_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - Requires SALib (pip install SALib)
"""

import json
import numpy as np

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from salib_agent.salib_utils import has_salib


# ---------------------------------------------------------------------------
# Built-in test functions for LLM self-testing
# ---------------------------------------------------------------------------

def _ishigami(X, a=7.0, b=0.1):
    """Ishigami function — standard SA benchmark."""
    return (np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2 +
            b * X[:, 2]**4 * np.sin(X[:, 0]))


def _linear(X):
    """Simple linear function: Y = sum(i * x_i)."""
    coeffs = np.arange(1, X.shape[1] + 1, dtype=float)
    return X @ coeffs


_TEST_FUNCTIONS = {
    "ishigami": _ishigami,
    "linear": _linear,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _clean_result(result):
    if isinstance(result, dict):
        return {k: _clean_value(v) if not isinstance(v, (dict, list)) else
                (_clean_result(v) if isinstance(v, dict) else
                 [_clean_value(x) for x in v])
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_sobol(params):
    from salib_agent import sobol_sample, sobol_analyze
    test_fn_name = params.pop("test_function", None)
    Y = params.pop("Y", None)
    n_samples = params.get("n_samples", 1024)
    seed = params.get("seed", 42)
    var_names = params["var_names"]
    bounds = params["bounds"]

    X = sobol_sample(var_names, bounds, n_samples=n_samples, seed=seed)

    if Y is not None:
        Y_arr = np.asarray(Y, dtype=float)
    elif test_fn_name in _TEST_FUNCTIONS:
        Y_arr = _TEST_FUNCTIONS[test_fn_name](X)
    else:
        return {
            "error": (
                "Must provide 'Y' (model output array) or 'test_function' "
                f"(one of {list(_TEST_FUNCTIONS.keys())}). "
                "Typical workflow: 1) call with just var_names/bounds to get "
                "sample matrix, 2) evaluate your model, 3) call again with Y."
            ),
            "sample_matrix": X.tolist(),
            "n_samples_total": len(X),
        }

    result = sobol_analyze(var_names, bounds, Y_arr, n_samples=n_samples, seed=seed)
    return _clean_result(result.to_dict())


def _run_morris(params):
    from salib_agent import morris_sample, morris_analyze
    test_fn_name = params.pop("test_function", None)
    Y = params.pop("Y", None)
    n_trajectories = params.get("n_trajectories", 20)
    num_levels = params.get("num_levels", 4)
    seed = params.get("seed", 42)
    var_names = params["var_names"]
    bounds = params["bounds"]

    X = morris_sample(
        var_names, bounds,
        n_trajectories=n_trajectories,
        num_levels=num_levels,
        seed=seed,
    )

    if Y is not None:
        Y_arr = np.asarray(Y, dtype=float)
    elif test_fn_name in _TEST_FUNCTIONS:
        Y_arr = _TEST_FUNCTIONS[test_fn_name](X)
    else:
        return {
            "error": (
                "Must provide 'Y' (model output array) or 'test_function' "
                f"(one of {list(_TEST_FUNCTIONS.keys())}). "
            ),
            "sample_matrix": X.tolist(),
            "n_samples_total": len(X),
        }

    result = morris_analyze(
        var_names, bounds, X, Y_arr,
        n_trajectories=n_trajectories,
        num_levels=num_levels,
        seed=seed,
    )
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "sobol_analyze": _run_sobol,
    "morris_analyze": _run_morris,
}

METHOD_INFO = {
    "sobol_analyze": {
        "category": "Sobol",
        "brief": "Sobol variance-based global sensitivity analysis.",
        "description": (
            "Decomposes output variance into contributions from each input "
            "variable and their interactions. Quantifies first-order (S1) "
            "and total-order (ST) sensitivity indices. Requires many model "
            "evaluations: N*(2D+2) where N=n_samples and D=n_vars."
        ),
        "reference": "Sobol (1993); Saltelli (2002); SALib (Herman & Usher)",
        "parameters": {
            "var_names": {"type": "list", "required": True,
                          "description": "List of variable names (strings). E.g., ['phi', 'c', 'gamma']."},
            "bounds": {"type": "list", "required": True,
                       "description": "List of [min, max] bounds for each variable. E.g., [[25, 40], [0, 50], [16, 20]]."},
            "n_samples": {"type": "int", "required": False, "default": 1024, "range": ">= 64",
                          "description": "Base sample count. Total evaluations = N*(2D+2). Higher = more accurate. 1024+ recommended."},
            "seed": {"type": "int", "required": False, "default": 42,
                     "description": "Random seed for reproducibility."},
            "test_function": {"type": "str", "required": False, "default": "null",
                              "choices": ["ishigami", "linear"],
                              "description": "Built-in test function for demo. Use 'ishigami' (3 vars, bounds [-pi,pi]) or 'linear'. If null, must provide 'Y'."},
            "Y": {"type": "array", "required": False, "default": "null",
                  "description": "Model output array. Length must equal N*(2D+2). Provide either Y or test_function."},
        },
        "returns": {
            "n_samples": "Total number of model evaluations.",
            "n_vars": "Number of input variables.",
            "var_names": "Variable names.",
            "S1": "First-order Sobol indices (main effect of each variable). Sum ≈ 1 if no interactions.",
            "S1_conf": "95% confidence intervals for S1.",
            "ST": "Total-order indices (including interactions). ST >= S1 always.",
            "ST_conf": "95% confidence intervals for ST.",
            "S2": "Second-order interaction indices (matrix, if computed).",
        },
    },
    "morris_analyze": {
        "category": "Morris",
        "brief": "Morris elementary effects screening.",
        "description": (
            "Fast screening method to identify important variables from "
            "a large set. Uses one-at-a-time (OAT) trajectories to compute "
            "mu* (importance) and sigma (nonlinearity/interactions). Much "
            "cheaper than Sobol: N*(D+1) evaluations."
        ),
        "reference": "Morris (1991); Campolongo et al. (2007); SALib (Herman & Usher)",
        "parameters": {
            "var_names": {"type": "list", "required": True,
                          "description": "List of variable names (strings)."},
            "bounds": {"type": "list", "required": True,
                       "description": "List of [min, max] bounds for each variable."},
            "n_trajectories": {"type": "int", "required": False, "default": 20, "range": ">= 4",
                               "description": "Number of OAT trajectories. 10-50 typical. Total evals = n_trajectories * (n_vars + 1)."},
            "num_levels": {"type": "int", "required": False, "default": 4, "range": ">= 2",
                           "description": "Number of grid levels for sampling."},
            "seed": {"type": "int", "required": False, "default": 42,
                     "description": "Random seed for reproducibility."},
            "test_function": {"type": "str", "required": False, "default": "null",
                              "choices": ["ishigami", "linear"],
                              "description": "Built-in test function for demo. If null, must provide 'Y'."},
            "Y": {"type": "array", "required": False, "default": "null",
                  "description": "Model output array. Length must equal n_trajectories * (n_vars + 1)."},
        },
        "returns": {
            "n_trajectories": "Number of trajectories used.",
            "n_vars": "Number of input variables.",
            "var_names": "Variable names.",
            "mu_star": "Mean of absolute elementary effects. Higher = more important.",
            "sigma": "Std of elementary effects. High sigma = nonlinear or interacting.",
            "mu_star_conf": "95% confidence intervals for mu*.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def salib_agent(method: str, parameters_json: str) -> str:
    """
    Sensitivity analysis agent (SALib).

    Runs global sensitivity analysis on geotechnical models to rank
    which input parameters most influence the output.

    Call salib_list_methods() first to see available analyses,
    then salib_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "sobol_analyze", "morris_analyze").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with sensitivity indices or an error message.
    """
    try:
        params = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })

    if not has_salib():
        return json.dumps({
            "error": "SALib is not installed. Install with: pip install SALib"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def salib_list_methods(category: str) -> str:
    """
    List available sensitivity analysis methods.

    Parameters:
        category: Filter by category (e.g. "Sobol", "Morris") or "" for all.

    Returns:
        JSON string mapping categories to method lists.
    """
    grouped = {}
    for name, info in METHOD_INFO.items():
        cat = info["category"]
        grouped.setdefault(cat, []).append(name)

    if not category or category.strip() == "":
        return json.dumps(grouped)

    if category in grouped:
        return json.dumps({category: grouped[category]})

    return json.dumps({"error": f"Unknown category '{category}'. Available: {sorted(grouped.keys())}"})


@function
def salib_describe_method(method: str) -> str:
    """
    Describe a sensitivity analysis method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "sobol_analyze", "morris_analyze").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
