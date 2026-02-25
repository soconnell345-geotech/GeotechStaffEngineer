"""
hvsrpy Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. hvsrpy_agent           - Run an HVSR analysis
  2. hvsrpy_list_methods    - Browse available methods
  3. hvsrpy_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[hvsrpy] (PyPI)
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

from hvsrpy_agent.hvsrpy_utils import has_hvsrpy


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
        return {k: _clean_value(v) if not isinstance(v, dict) else _clean_result(v)
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_analyze_hvsr(params):
    from hvsrpy_agent import analyze_hvsr
    # Convert list inputs to numpy arrays
    for key in ("ns", "ew", "vt"):
        if key in params and isinstance(params[key], list):
            params[key] = np.array(params[key], dtype=float)
    result = analyze_hvsr(**params)
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "analyze_hvsr": _run_analyze_hvsr,
}

METHOD_INFO = {
    "analyze_hvsr": {
        "category": "HVSR Analysis",
        "brief": "Compute HVSR from 3-component seismogram.",
        "description": (
            "Computes Horizontal-to-Vertical Spectral Ratio from 3-component "
            "microtremor or earthquake recordings. Identifies site resonant "
            "frequency (f0), peak amplification (A0), and site period (T0). "
            "Evaluates SESAME (2004) reliability and clarity criteria. "
            "All three input arrays must have the same length."
        ),
        "reference": "Nakamura (1989); SESAME (2004)",
        "parameters": {
            "ns": {"type": "array", "required": True,
                   "description": "North-south component amplitudes (any consistent unit, e.g. m/s, counts). Array of floats."},
            "ew": {"type": "array", "required": True,
                   "description": "East-west component amplitudes (same unit as ns). Array of floats."},
            "vt": {"type": "array", "required": True,
                   "description": "Vertical component amplitudes (same unit as ns). Array of floats."},
            "dt": {"type": "float", "required": True, "range": "> 0",
                   "description": "Sampling interval (seconds). E.g., 0.01 for 100 Hz."},
            "window_length_s": {"type": "float", "required": False, "default": 60.0, "range": "> 0",
                                "description": "Time window length for spectral averaging (seconds). Must be less than total recording duration. Typical 30-120s."},
            "filter_hz": {"type": "list", "required": False, "default": "null",
                          "description": "Butterworth bandpass corners [f_low, f_high] in Hz. E.g., [0.5, 25.0]. null = no filter."},
            "smoothing_operator": {"type": "str", "required": False, "default": "konno_and_ohmachi",
                                   "choices": ["konno_and_ohmachi", "parzen", "savitzky_and_golay"],
                                   "description": "Spectral smoothing method. Konno-Ohmachi is standard for HVSR."},
            "smoothing_bandwidth": {"type": "float", "required": False, "default": 40, "range": "> 0",
                                    "description": "Smoothing bandwidth. 40 for Konno-Ohmachi, 0.5 Hz for Parzen, 9 points for Savitzky-Golay."},
            "freq_min": {"type": "float", "required": False, "default": 0.2, "range": "> 0",
                         "description": "Minimum output frequency (Hz)."},
            "freq_max": {"type": "float", "required": False, "default": 50.0, "range": "> freq_min",
                         "description": "Maximum output frequency (Hz)."},
            "n_freq": {"type": "int", "required": False, "default": 200, "range": "> 0",
                       "description": "Number of log-spaced frequency points."},
            "horizontal_method": {"type": "str", "required": False, "default": "geometric_mean",
                                  "choices": ["geometric_mean", "arithmetic_mean", "quadratic_mean",
                                              "total_horizontal_energy", "vector_summation", "maximum_horizontal_value"],
                                  "description": "Method to combine horizontal components. Geometric mean is standard."},
            "distribution": {"type": "str", "required": False, "default": "lognormal",
                             "choices": ["lognormal", "normal"],
                             "description": "Statistical distribution for averaging. Lognormal is recommended for HVSR amplitudes."},
            "rejection_n_std": {"type": "float", "required": False, "default": 2.0, "range": ">= 0",
                                "description": "Standard deviations for frequency-domain window rejection. 0 = no rejection."},
            "rejection_max_iterations": {"type": "int", "required": False, "default": 50, "range": "> 0",
                                         "description": "Maximum iterations for rejection convergence."},
            "degrees_from_north": {"type": "float", "required": False, "default": 0.0,
                                   "description": "Sensor orientation in degrees clockwise from north. 0 = north-aligned."},
        },
        "returns": {
            "f0_hz": "Resonant frequency (Hz). The fundamental frequency of the soil column.",
            "A0": "Peak HVSR amplitude at f0. Values > 2 suggest clear impedance contrast.",
            "T0_s": "Site period (s) = 1/f0. Relates to soil thickness and Vs: T0 = 4H/Vs.",
            "f0_std_hz": "Standard deviation of f0 across time windows.",
            "A0_std": "Standard deviation of A0 across time windows.",
            "n_windows": "Total number of time windows.",
            "n_valid_windows": "Windows passing rejection criteria.",
            "sesame_reliability": "List of 3 bools (0/1). All 3 must pass for reliable result.",
            "sesame_clarity": "List of 6 bools (0/1). At least 5 of 6 must pass for clear peak.",
            "sesame_reliability_pass": "Count of passing reliability criteria (out of 3).",
            "sesame_clarity_pass": "Count of passing clarity criteria (out of 6).",
            "frequency_hz": "Frequency vector (Hz). Log-spaced.",
            "mean_curve": "Mean HVSR curve (amplitude vs frequency).",
            "std_curve": "Standard deviation of HVSR curve.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def hvsrpy_agent(method: str, parameters_json: str) -> str:
    """
    HVSR site characterization agent.

    Computes Horizontal-to-Vertical Spectral Ratios from 3-component
    seismograms to identify site resonant frequency (f0), peak
    amplification (A0), and site period (T0).

    Call hvsrpy_list_methods() first to see available analyses,
    then hvsrpy_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "analyze_hvsr").
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

    if not has_hvsrpy():
        return json.dumps({
            "error": "hvsrpy is not installed. Install with: pip install hvsrpy"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def hvsrpy_list_methods(category: str) -> str:
    """
    List available HVSR analysis methods.

    Parameters:
        category: Filter by category (e.g. "HVSR Analysis") or "" for all.

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
def hvsrpy_describe_method(method: str) -> str:
    """
    Describe an HVSR analysis method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "analyze_hvsr").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
