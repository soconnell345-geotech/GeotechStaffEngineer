"""
Seismic Signals Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. seismic_signals_agent          - Run a seismic signal analysis
  2. seismic_signals_list_methods   - Browse available analysis methods
  3. seismic_signals_describe_method - Get detailed docs for a specific method

FOUNDRY SETUP:
  - Add 'eqsig' and/or 'pyrotd' to your conda_recipe/meta.yaml dependencies
  - These functions accept and return JSON strings for maximum LLM compatibility
  - The LLM should call seismic_signals_list_methods first, then
    seismic_signals_describe_method for parameter details, then
    seismic_signals_agent to run the analysis
"""

import json

try:
    from functions.api import function
except ImportError:
    # Outside Palantir Foundry: provide a no-op decorator
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from opensees_agent.opensees_utils import clean_numpy


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_response_spectrum(params):
    from seismic_signals_agent.response_spectrum import analyze_response_spectrum
    result = analyze_response_spectrum(**params)
    return result.to_dict()


def _run_intensity_measures(params):
    from seismic_signals_agent.intensity_measures import analyze_intensity_measures
    result = analyze_intensity_measures(**params)
    return result.to_dict()


def _run_rotd_spectrum(params):
    from seismic_signals_agent.rotd_spectrum import analyze_rotd_spectrum
    result = analyze_rotd_spectrum(**params)
    return result.to_dict()


def _run_signal_processing(params):
    from seismic_signals_agent.signal_processing import analyze_signal_processing
    result = analyze_signal_processing(**params)
    return result.to_dict()


METHOD_REGISTRY = {
    "response_spectrum": _run_response_spectrum,
    "intensity_measures": _run_intensity_measures,
    "rotd_spectrum": _run_rotd_spectrum,
    "signal_processing": _run_signal_processing,
}

# Methods grouped by required dependency
_EQSIG_METHODS = {"response_spectrum", "intensity_measures", "signal_processing"}
_PYROTD_METHODS = {"rotd_spectrum"}


# ---------------------------------------------------------------------------
# Method metadata
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "response_spectrum": {
        "category": "Response Spectrum",
        "brief": "Response spectrum via Nigam-Jennings algorithm (eqsig).",
        "description": (
            "Computes the pseudo-spectral acceleration response spectrum using "
            "the Nigam & Jennings (1969) algorithm via eqsig. This is the exact "
            "piecewise-linear solution, more accurate than Newmark-beta for a "
            "given time step. Also returns PGA, PGV, and PGD."
        ),
        "reference": (
            'Nigam, N.C. & Jennings, P.C. (1969). "Calculation of Response '
            'Spectra from Strong-Motion Earthquake Records." BSSA 59(2), 909-922.'
        ),
        "parameters": {
            "motion": {
                "type": "str", "required": False,
                "description": (
                    "Built-in motion name (e.g. 'synthetic_pulse'). "
                    "Provide either 'motion' or 'accel_history'+'dt'."
                ),
            },
            "accel_history": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration time history (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step for custom motion (s).",
            },
            "periods": {
                "type": "list of float", "required": False,
                "default": "logspace(-2, 1, 200)",
                "description": "Spectral periods (s). Default: 200 periods from 0.01 to 10 s.",
            },
            "damping": {
                "type": "float", "required": False,
                "default": 0.05, "range": "0 to 1 (exclusive)",
                "description": "Damping ratio (decimal). Default: 0.05 (5%).",
            },
        },
        "returns": {
            "motion_name": "Input motion name.",
            "n_points": "Number of points in the record.",
            "duration_s": "Total record duration (s).",
            "dt_s": "Time step (s).",
            "pga_g": "Peak ground acceleration (g).",
            "pgv_m_per_s": "Peak ground velocity (m/s).",
            "pgd_m": "Peak ground displacement (m).",
            "damping": "Damping ratio used.",
        },
    },
    "intensity_measures": {
        "category": "Intensity Measures",
        "brief": "Earthquake intensity measures (Arias, CAV, duration).",
        "description": (
            "Computes earthquake intensity measures from an acceleration time "
            "history using eqsig: Arias intensity, significant duration (D5-95), "
            "Cumulative Absolute Velocity (CAV), bracketed duration, PGA, PGV, PGD."
        ),
        "reference": (
            'Arias, A. (1970). "A Measure of Earthquake Intensity." In Seismic '
            'Design for Nuclear Power Plants, MIT Press. '
            'EPRI (1988). "A Criterion for Determining Exceedance of the '
            'Operating Basis Earthquake." EPRI NP-5930.'
        ),
        "parameters": {
            "motion": {
                "type": "str", "required": False,
                "description": (
                    "Built-in motion name. "
                    "Provide either 'motion' or 'accel_history'+'dt'."
                ),
            },
            "accel_history": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration time history (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step for custom motion (s).",
            },
            "sig_dur_start": {
                "type": "float", "required": False,
                "default": 0.05, "range": "0 to 1 (exclusive)",
                "description": "Husid start fraction for significant duration.",
            },
            "sig_dur_end": {
                "type": "float", "required": False,
                "default": 0.95, "range": "0 to 1 (exclusive)",
                "description": "Husid end fraction for significant duration.",
            },
        },
        "returns": {
            "motion_name": "Input motion name.",
            "pga_g": "Peak ground acceleration (g).",
            "pgv_m_per_s": "Peak ground velocity (m/s).",
            "pgd_m": "Peak ground displacement (m).",
            "arias_intensity_m_per_s": "Total Arias intensity (m/s).",
            "significant_duration_s": "Significant duration (s).",
            "cav_m_per_s": "Cumulative Absolute Velocity (m/s).",
            "bracketed_duration_s": "Bracketed duration (s).",
        },
    },
    "rotd_spectrum": {
        "category": "Response Spectrum",
        "brief": "Rotated spectral acceleration (RotD50/RotD100) via pyrotd.",
        "description": (
            "Computes orientation-independent spectral acceleration from two "
            "orthogonal horizontal components using pyrotd. RotD50 is the "
            "median-orientation PSA (Boore 2010), now the standard for "
            "NGA-West2 ground motion models."
        ),
        "reference": (
            'Boore, D.M. (2010). "Orientation-Independent, Nongeometric-Mean '
            'Measures of Seismic Intensity from Two Horizontal Components of '
            'Motion." BSSA 100(4), 1830-1835.'
        ),
        "parameters": {
            "motion_a": {
                "type": "str", "required": False,
                "description": "Built-in motion name for component A.",
            },
            "accel_history_a": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration for component A (g).",
            },
            "motion_b": {
                "type": "str", "required": False,
                "description": "Built-in motion name for component B.",
            },
            "accel_history_b": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration for component B (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step (s). Required for custom motions.",
            },
            "periods": {
                "type": "list of float", "required": False,
                "default": "logspace(-2, 1, 200)",
                "description": "Spectral periods (s).",
            },
            "damping": {
                "type": "float", "required": False,
                "default": 0.05,
                "description": "Damping ratio (decimal).",
            },
            "percentiles": {
                "type": "list of int", "required": False,
                "default": "[0, 50, 100]",
                "description": "Percentiles to compute (e.g. [0, 50, 100] for RotD0/50/100).",
            },
        },
        "returns": {
            "motion_a_name": "Component A motion name.",
            "motion_b_name": "Component B motion name.",
            "n_points": "Number of points in the records.",
            "dt_s": "Time step (s).",
            "pga_a_g": "PGA of component A (g).",
            "pga_b_g": "PGA of component B (g).",
            "damping": "Damping ratio used.",
            "peak_rotd50_g": "Peak RotD50 spectral acceleration (g).",
            "peak_rotd100_g": "Peak RotD100 spectral acceleration (g).",
        },
    },
    "signal_processing": {
        "category": "Signal Processing",
        "brief": "Butterworth filtering and baseline correction (eqsig).",
        "description": (
            "Processes an acceleration time history using eqsig. Applies "
            "Butterworth bandpass filtering and/or polynomial baseline correction. "
            "Returns original and processed acceleration, velocity, and displacement. "
            "Bandpass is applied before baseline correction (standard practice)."
        ),
        "reference": (
            'Boore, D.M. & Bommer, J.J. (2005). "Processing of strong-motion '
            'accelerograms: needs, options and consequences." Soil Dynamics and '
            'Earthquake Engineering 25, 93-115.'
        ),
        "parameters": {
            "motion": {
                "type": "str", "required": False,
                "description": (
                    "Built-in motion name. "
                    "Provide either 'motion' or 'accel_history'+'dt'."
                ),
            },
            "accel_history": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration time history (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step for custom motion (s).",
            },
            "bandpass": {
                "type": "list of float", "required": False,
                "description": "Bandpass frequencies [f_low, f_high] in Hz.",
            },
            "baseline_order": {
                "type": "int", "required": False,
                "description": (
                    "Polynomial order for baseline correction "
                    "(0=mean, 1=linear, 2=quadratic)."
                ),
            },
        },
        "returns": {
            "motion_name": "Input motion name.",
            "n_points": "Number of points in the record.",
            "dt_s": "Time step (s).",
            "bandpass_hz": "Bandpass frequencies applied (Hz), or null.",
            "baseline_order": "Baseline correction order applied, or null.",
            "pga_original_g": "PGA of original record (g).",
            "pga_processed_g": "PGA of processed record (g).",
            "pgv_processed_m_per_s": "PGV of processed record (m/s).",
            "pgd_processed_m": "PGD of processed record (m).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry agent functions
# ---------------------------------------------------------------------------

@function
def seismic_signals_agent(method: str, parameters_json: str) -> str:
    """
    Seismic signals analysis agent (eqsig + pyrotd).

    Provides response spectra, intensity measures, rotated spectral
    acceleration (RotD50/RotD100), and signal processing.

    Call seismic_signals_list_methods() first to see available analyses,
    then seismic_signals_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "response_spectrum").
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

    # Per-method dependency check
    from seismic_signals_agent.signal_utils import has_eqsig, has_pyrotd
    if method in _EQSIG_METHODS and not has_eqsig():
        return json.dumps({
            "error": "eqsig is not installed. Install with: pip install eqsig"
        })
    if method in _PYROTD_METHODS and not has_pyrotd():
        return json.dumps({
            "error": "pyrotd is not installed. Install with: pip install pyrotd"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(clean_numpy(result), default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def seismic_signals_list_methods(category: str = "") -> str:
    """
    Lists available seismic signal analysis methods.

    Use this to discover what analyses are available before calling
    seismic_signals_agent.

    Parameters:
        category: Optional filter (e.g. "Response Spectrum"). Leave empty for all.

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]

    if not result:
        cats = sorted(set(i["category"] for i in METHOD_INFO.values()))
        return json.dumps({
            "error": f"No methods found for category '{category}'. "
                     f"Available: {', '.join(cats)}"
        })
    return json.dumps(result)


@function
def seismic_signals_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a seismic signal analysis method.

    Use this to understand what parameters a method needs before calling
    seismic_signals_agent.

    Parameters:
        method: The method name (e.g. "response_spectrum").

    Returns:
        JSON string with: category, description, parameters (types, ranges,
        defaults), and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
