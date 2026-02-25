"""
GSTools Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. gstools_agent           - Run a geostatistical analysis
  2. gstools_list_methods    - Browse available methods
  3. gstools_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[gstools] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import math
import numpy as np

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from gstools_agent.gstools_utils import has_gstools


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
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
                (_clean_result(v) if isinstance(v, dict) else v)
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_kriging(params):
    from gstools_agent import analyze_kriging
    for key in ("x", "y", "values"):
        if key in params and isinstance(params[key], list):
            params[key] = np.array(params[key], dtype=float)
    result = analyze_kriging(**params)
    return _clean_result(result.to_dict())


def _run_variogram(params):
    from gstools_agent import analyze_variogram
    for key in ("x", "y", "values"):
        if key in params and isinstance(params[key], list):
            params[key] = np.array(params[key], dtype=float)
    result = analyze_variogram(**params)
    return _clean_result(result.to_dict())


def _run_random_field(params):
    from gstools_agent import generate_random_field
    result = generate_random_field(**params)
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "kriging": _run_kriging,
    "variogram": _run_variogram,
    "random_field": _run_random_field,
}

METHOD_INFO = {
    "kriging": {
        "category": "Kriging",
        "brief": "Krige soil properties onto a regular grid.",
        "description": (
            "Spatially interpolates point measurements (SPT N, Vs, friction "
            "angle, etc.) from borehole or CPT locations onto a 2D regular "
            "grid using kriging. Automatically fits a variogram model to the "
            "data for optimal interpolation. Returns the interpolated field "
            "and kriging variance (uncertainty map)."
        ),
        "reference": "Matheron (1963); GSTools (Mueller et al., 2022)",
        "parameters": {
            "x": {"type": "array", "required": True,
                   "description": "X-coordinates of measurement points (m). Array of floats."},
            "y": {"type": "array", "required": True,
                   "description": "Y-coordinates of measurement points (m). Array of floats."},
            "values": {"type": "array", "required": True,
                       "description": "Measured values at each point (e.g. SPT N-values, Vs in m/s). Array of floats."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian",
                           "choices": ["Gaussian", "Exponential", "Matern", "Spherical",
                                       "Linear", "Stable", "Rational", "Cubic", "HyperSpherical"],
                           "description": "Covariance model for spatial correlation. Gaussian = smooth fields, Exponential = rough fields."},
            "kriging_type": {"type": "str", "required": False, "default": "ordinary",
                             "choices": ["ordinary", "simple", "universal"],
                             "description": "Kriging type. Ordinary = unknown constant mean, Simple = known mean, Universal = trend + residual."},
            "grid_x_min": {"type": "float", "required": False, "default": "auto",
                           "description": "Grid minimum X (m). null = auto from data bounds + 10% buffer."},
            "grid_x_max": {"type": "float", "required": False, "default": "auto",
                           "description": "Grid maximum X (m). null = auto."},
            "grid_y_min": {"type": "float", "required": False, "default": "auto",
                           "description": "Grid minimum Y (m). null = auto."},
            "grid_y_max": {"type": "float", "required": False, "default": "auto",
                           "description": "Grid maximum Y (m). null = auto."},
            "n_grid_x": {"type": "int", "required": False, "default": 50, "range": ">= 2",
                          "description": "Number of grid points in X."},
            "n_grid_y": {"type": "int", "required": False, "default": 50, "range": ">= 2",
                          "description": "Number of grid points in Y."},
            "variance": {"type": "float", "required": False, "default": "auto", "range": "> 0",
                          "description": "Sill variance. null = estimate from data variance."},
            "len_scale": {"type": "float", "required": False, "default": "auto", "range": "> 0",
                           "description": "Correlation length (m). null = estimate as 1/3 of data extent."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "range": ">= 0",
                        "description": "Nugget variance (measurement noise). 0 = exact interpolation."},
            "fit_variogram": {"type": "bool", "required": False, "default": True,
                               "description": "If true, fit variogram model to data before kriging."},
        },
        "returns": {
            "n_data": "Number of input measurement points.",
            "n_grid_x": "Grid points in X.",
            "n_grid_y": "Grid points in Y.",
            "model_type": "Fitted covariance model name.",
            "variance": "Fitted sill variance.",
            "len_scale": "Fitted correlation length (m).",
            "nugget": "Fitted nugget variance.",
            "kriging_type": "Kriging method used.",
            "field": "2D interpolated field (n_grid_x x n_grid_y array).",
            "field_min": "Minimum interpolated value.",
            "field_max": "Maximum interpolated value.",
            "field_mean": "Mean interpolated value.",
            "krige_variance": "2D kriging variance field — higher values indicate more uncertainty.",
            "grid_x": "X coordinates of grid points.",
            "grid_y": "Y coordinates of grid points.",
        },
    },
    "variogram": {
        "category": "Variogram",
        "brief": "Estimate and fit empirical variogram.",
        "description": (
            "Estimates the empirical variogram from point measurements and "
            "fits a theoretical covariance model. The variogram describes "
            "how spatial correlation decays with distance — essential for "
            "understanding spatial variability of soil properties and "
            "parameterizing kriging or random field generation."
        ),
        "reference": "Matheron (1963); GSTools (Mueller et al., 2022)",
        "parameters": {
            "x": {"type": "array", "required": True,
                   "description": "X-coordinates of measurement points (m). Array of floats."},
            "y": {"type": "array", "required": True,
                   "description": "Y-coordinates of measurement points (m). Array of floats."},
            "values": {"type": "array", "required": True,
                       "description": "Measured values at each point. Array of floats."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian",
                           "choices": ["Gaussian", "Exponential", "Matern", "Spherical",
                                       "Linear", "Stable", "Rational", "Cubic", "HyperSpherical"],
                           "description": "Theoretical covariance model to fit."},
            "n_bins": {"type": "int", "required": False, "default": 10, "range": "> 0",
                        "description": "Number of lag distance bins for the empirical variogram."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "range": ">= 0",
                        "description": "Nugget variance. 0 = no measurement noise."},
        },
        "returns": {
            "n_data": "Number of input points.",
            "n_bins": "Number of variogram bins.",
            "model_type": "Fitted covariance model name.",
            "variance": "Fitted sill variance.",
            "len_scale": "Fitted correlation length (m). Distance at which correlation drops significantly.",
            "nugget": "Fitted nugget variance.",
            "bin_center": "Lag distances at bin centers (m).",
            "gamma": "Empirical semivariance at each lag.",
        },
    },
    "random_field": {
        "category": "Random Field",
        "brief": "Generate 2D spatially correlated random field.",
        "description": (
            "Generates a 2D spatially correlated random field using the "
            "Spectral Random Field (SRF) method. Useful for probabilistic "
            "geotechnical analysis — modeling spatial variability of soil "
            "properties (friction angle, undrained strength, SPT N) for "
            "Monte Carlo or reliability analysis."
        ),
        "reference": "GSTools (Mueller et al., 2022)",
        "parameters": {
            "model_type": {"type": "str", "required": False, "default": "Gaussian",
                           "choices": ["Gaussian", "Exponential", "Matern", "Spherical",
                                       "Linear", "Stable", "Rational", "Cubic", "HyperSpherical"],
                           "description": "Covariance model for spatial correlation."},
            "variance": {"type": "float", "required": False, "default": 1.0, "range": "> 0",
                          "description": "Field variance. Larger = more variable."},
            "len_scale": {"type": "float", "required": False, "default": 10.0, "range": "> 0",
                           "description": "Correlation length (m). Larger = smoother spatial patterns."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "range": ">= 0",
                        "description": "Nugget variance (uncorrelated noise)."},
            "mean": {"type": "float", "required": False, "default": 0.0,
                      "description": "Mean value of the field. E.g., 30 for SPT N = 30."},
            "x_min": {"type": "float", "required": False, "default": 0.0,
                       "description": "X-axis minimum (m)."},
            "x_max": {"type": "float", "required": False, "default": 100.0,
                       "description": "X-axis maximum (m)."},
            "y_min": {"type": "float", "required": False, "default": 0.0,
                       "description": "Y-axis minimum (m)."},
            "y_max": {"type": "float", "required": False, "default": 100.0,
                       "description": "Y-axis maximum (m)."},
            "n_x": {"type": "int", "required": False, "default": 50, "range": ">= 2",
                     "description": "Number of grid points in X."},
            "n_y": {"type": "int", "required": False, "default": 50, "range": ">= 2",
                     "description": "Number of grid points in Y."},
            "seed": {"type": "int", "required": False, "default": 42,
                      "description": "Random seed for reproducibility."},
        },
        "returns": {
            "n_grid_x": "Grid points in X.",
            "n_grid_y": "Grid points in Y.",
            "model_type": "Covariance model used.",
            "variance": "Field variance.",
            "len_scale": "Correlation length (m).",
            "mean": "Mean value.",
            "seed": "Random seed used.",
            "field": "2D random field (n_grid_x x n_grid_y array).",
            "field_min": "Minimum field value.",
            "field_max": "Maximum field value.",
            "field_mean": "Actual field mean (close to specified mean for large grids).",
            "field_std": "Actual field standard deviation.",
            "grid_x": "X coordinates of grid points.",
            "grid_y": "Y coordinates of grid points.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def gstools_agent(method: str, parameters_json: str) -> str:
    """
    Geostatistical analysis agent (GSTools).

    Provides kriging interpolation, variogram analysis, and spatial random
    field generation for geotechnical soil properties.

    Call gstools_list_methods() first to see available analyses,
    then gstools_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "kriging", "variogram", "random_field").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with analysis results or an error message.
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

    if not has_gstools():
        return json.dumps({
            "error": "gstools is not installed. Install with: pip install gstools"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def gstools_list_methods(category: str) -> str:
    """
    List available geostatistical analysis methods.

    Parameters:
        category: Filter by category (e.g. "Kriging", "Variogram") or "" for all.

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
def gstools_describe_method(method: str) -> str:
    """
    Describe a geostatistical method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "kriging", "variogram", "random_field").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
