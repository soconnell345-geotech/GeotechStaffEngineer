"""
swprocess Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. swprocess_agent           - Run MASW dispersion analysis
  2. swprocess_list_methods    - Browse available methods
  3. swprocess_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[swprocess] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import numpy as np

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from swprocess_agent.swprocess_utils import has_swprocess


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

def _run_analyze_masw(params):
    from swprocess_agent import analyze_masw
    # Convert list inputs to numpy arrays
    if "traces" in params and isinstance(params["traces"], list):
        params["traces"] = [np.array(t, dtype=float) for t in params["traces"]]
    result = analyze_masw(**params)
    return _clean_result(result.to_dict())


METHOD_REGISTRY = {
    "analyze_masw": _run_analyze_masw,
}

METHOD_INFO = {
    "analyze_masw": {
        "category": "MASW",
        "brief": "Extract dispersion curve from multi-channel seismic data.",
        "description": (
            "Runs Multichannel Analysis of Surface Waves (MASW) on seismic "
            "array data. Applies a wavefield transform (phase-shift, FK, or "
            "FDBF) to convert time-distance records into a frequency-velocity "
            "dispersion image, then extracts the dispersion curve (peak "
            "velocity at each frequency). Dispersion curves are inverted "
            "to obtain Vs profiles for site characterization."
        ),
        "reference": "Park et al. (1999); Foti et al. (2018); swprocess (Vantassel & Cox)",
        "parameters": {
            "traces": {"type": "list of arrays", "required": True,
                       "description": "Seismograms for each sensor channel. List of 1D arrays (same length). Minimum 3 channels."},
            "offsets": {"type": "list", "required": True,
                        "description": "Source-to-receiver offset for each channel (m). Positive values. E.g., [2, 4, 6, 8, 10, 12]."},
            "dt": {"type": "float", "required": True, "range": "> 0",
                   "description": "Sampling interval in seconds. E.g., 0.001 for 1 kHz."},
            "transform": {"type": "str", "required": False, "default": "phase_shift",
                           "choices": ["phase_shift", "fk", "fdbf"],
                           "description": "Wavefield transform. phase_shift is most common for active MASW."},
            "fmin": {"type": "float", "required": False, "default": 5.0, "range": "> 0",
                     "description": "Minimum frequency for dispersion image (Hz)."},
            "fmax": {"type": "float", "required": False, "default": 100.0, "range": "> fmin",
                     "description": "Maximum frequency (Hz). Typically Nyquist/2 or less."},
            "vmin": {"type": "float", "required": False, "default": 50.0, "range": "> 0",
                     "description": "Minimum phase velocity (m/s)."},
            "vmax": {"type": "float", "required": False, "default": 1000.0, "range": "> vmin",
                     "description": "Maximum phase velocity (m/s)."},
            "nvel": {"type": "int", "required": False, "default": 200, "range": ">= 2",
                     "description": "Number of velocity bins in dispersion image."},
        },
        "returns": {
            "n_channels": "Number of sensor channels.",
            "spacing_m": "Average sensor spacing (m).",
            "transform": "Wavefield transform used.",
            "n_freq": "Number of frequency bins.",
            "n_vel": "Number of velocity bins.",
            "disp_freq_hz": "Dispersion curve frequencies (Hz).",
            "disp_vel_mps": "Dispersion curve phase velocities (m/s).",
            "freq_min": "Minimum frequency in image (Hz).",
            "freq_max": "Maximum frequency in image (Hz).",
            "vel_min": "Minimum velocity in image (m/s).",
            "vel_max": "Maximum velocity in image (m/s).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def swprocess_agent(method: str, parameters_json: str) -> str:
    """
    MASW surface wave processing agent.

    Extracts dispersion curves from multi-channel seismic array data
    for Vs profiling and site characterization.

    Call swprocess_list_methods() first to see available analyses,
    then swprocess_describe_method() for parameter details.

    Parameters:
        method: Method name (e.g. "analyze_masw").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with dispersion results or an error message.
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

    if not has_swprocess():
        return json.dumps({
            "error": "swprocess is not installed. Install with: pip install swprocess"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def swprocess_list_methods(category: str) -> str:
    """
    List available MASW methods.

    Parameters:
        category: Filter by category (e.g. "MASW") or "" for all.

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
def swprocess_describe_method(method: str) -> str:
    """
    Describe an MASW method with full parameter documentation.

    Parameters:
        method: Method name (e.g. "analyze_masw").

    Returns:
        JSON string with description, parameters, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    return json.dumps(METHOD_INFO[method])
