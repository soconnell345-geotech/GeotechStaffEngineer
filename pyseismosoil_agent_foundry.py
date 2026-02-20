"""
PySeismoSoil Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. pyseismosoil_agent           - Generate soil curves or analyze Vs profile
  2. pyseismosoil_list_methods    - Browse available methods
  3. pyseismosoil_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - Requires PySeismoSoil (pip install PySeismoSoil)
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

from pyseismosoil_agent.pyseismosoil_utils import has_pyseismosoil


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
                (_clean_result(v) if isinstance(v, dict) else
                 [_clean_value(x) for x in v])
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_generate_curves(params):
    from pyseismosoil_agent import generate_curves
    result = generate_curves(**params)
    return _clean_result(result.to_dict())


def _run_analyze_vs_profile(params):
    from pyseismosoil_agent import analyze_vs_profile
    result = analyze_vs_profile(**params)
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "generate_curves": _run_generate_curves,
    "analyze_vs_profile": _run_analyze_vs_profile,
}

METHOD_INFO = {
    "generate_curves": {
        "category": "Soil Curves",
        "brief": "Generate G/Gmax and damping curves from MKZ or HH model.",
        "description": (
            "Generates modulus reduction (G/Gmax) and damping ratio curves "
            "from constitutive models. MKZ (Modified Kodner-Zelasko) is "
            "simpler (4 parameters). HH (Hybrid Hyperbolic) is more "
            "flexible (9 parameters, better large-strain behavior). "
            "Output curves can be used directly in site response analysis."
        ),
        "reference": "Matasovic & Vucetic (1993); Shi & Asimaki (2017); PySeismoSoil",
        "parameters": {
            "model": {"type": "str", "required": False, "default": "MKZ",
                      "choices": ["MKZ", "HH"],
                      "description": "Constitutive model. MKZ = Modified Kodner-Zelasko, HH = Hybrid Hyperbolic."},
            "params": {"type": "dict", "required": True,
                       "description": (
                           "Model parameters as a dict. "
                           "MKZ requires: gamma_ref (reference strain %), beta, s, Gmax (kPa). "
                           "HH requires: gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d."
                       )},
            "strain_min": {"type": "float", "required": False, "default": 0.0001, "range": "> 0",
                           "description": "Minimum shear strain in percent."},
            "strain_max": {"type": "float", "required": False, "default": 10.0, "range": "> strain_min",
                           "description": "Maximum shear strain in percent."},
            "n_points": {"type": "int", "required": False, "default": 50, "range": ">= 2",
                         "description": "Number of log-spaced strain points."},
        },
        "returns": {
            "model": "Model name (MKZ or HH).",
            "params": "Parameters used.",
            "n_points": "Number of strain points.",
            "strain_pct": "Shear strain values (%).",
            "G_Gmax": "Modulus reduction values (0 to 1).",
            "damping_pct": "Damping ratio values (%).",
        },
    },
    "analyze_vs_profile": {
        "category": "Vs Profile",
        "brief": "Compute Vs30, f0, z1 from shear wave velocity profile.",
        "description": (
            "Analyzes a layered Vs profile to compute site characterization "
            "parameters: Vs30 (NEHRP), fundamental frequency (two methods), "
            "and basin depth proxy (z1). Input is layer thicknesses and "
            "Vs values; last layer must have thickness=0 (halfspace)."
        ),
        "reference": "Borcherdt (1994); Roesset (1970); PySeismoSoil",
        "parameters": {
            "thicknesses": {"type": "list", "required": True,
                            "description": "Layer thicknesses in meters. Last must be 0 (halfspace). E.g., [5, 5, 10, 0]."},
            "vs_values": {"type": "list", "required": True,
                          "description": "Shear wave velocity for each layer in m/s. E.g., [150, 200, 300, 400]."},
        },
        "returns": {
            "n_layers": "Number of soil layers (excluding halfspace).",
            "vs30": "Time-averaged Vs in top 30m (m/s). Used for NEHRP site classification.",
            "f0_bh": "Fundamental frequency from Borcherdt-Hartzell method (Hz). f0 = Vs/(4H).",
            "f0_ro": "Fundamental frequency from Roesset transfer function method (Hz).",
            "z1": "Depth to Vs >= 1000 m/s (m). Basin depth proxy for GMPEs.",
            "z_max": "Total profile depth (m).",
            "thicknesses": "Layer thicknesses (m).",
            "vs_values": "Layer Vs values (m/s).",
            "depth_array": "Interface depths from surface (m).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def pyseismosoil_agent(method: str, parameters_json: str) -> str:
    """
    PySeismoSoil agent — soil curves and Vs profile analysis.

    Generates nonlinear G/Gmax and damping curves, and computes
    site characterization parameters from Vs profiles.

    Call pyseismosoil_list_methods() first to see available methods,
    then pyseismosoil_describe_method() for parameter details.

    Parameters:
        method: Method name (e.g. "generate_curves", "analyze_vs_profile").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with results or an error message.
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

    if not has_pyseismosoil():
        return json.dumps({
            "error": "PySeismoSoil is not installed. Install with: pip install PySeismoSoil"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def pyseismosoil_list_methods(category: str) -> str:
    """
    List available PySeismoSoil methods.

    Parameters:
        category: Filter by category (e.g. "Soil Curves", "Vs Profile") or "" for all.

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
def pyseismosoil_describe_method(method: str) -> str:
    """
    Describe a PySeismoSoil method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "generate_curves", "analyze_vs_profile").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
