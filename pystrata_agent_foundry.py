"""
pyStrata Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. pystrata_agent          - Run a pystrata site response analysis
  2. pystrata_list_methods   - Browse available analysis methods
  3. pystrata_describe_method - Get detailed docs for a specific method

FOUNDRY SETUP:
  - Add 'pystrata' to your conda_recipe/meta.yaml dependencies
  - These functions accept and return JSON strings for maximum LLM compatibility
  - The LLM should call pystrata_list_methods first, then pystrata_describe_method
    for parameter details, then pystrata_agent to run the analysis
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

def _run_eql_site_response(params):
    """Wrapper for analyze_eql_site_response."""
    from pystrata_agent.eql_site_response import analyze_eql_site_response
    result = analyze_eql_site_response(**params)
    return result.to_dict()


def _run_linear_site_response(params):
    """Wrapper for analyze_linear_site_response."""
    from pystrata_agent.eql_site_response import analyze_linear_site_response
    result = analyze_linear_site_response(**params)
    return result.to_dict()


METHOD_REGISTRY = {
    "eql_site_response": _run_eql_site_response,
    "linear_site_response": _run_linear_site_response,
}


# ---------------------------------------------------------------------------
# Method metadata
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "eql_site_response": {
        "category": "Site Response",
        "brief": "1D equivalent-linear site response analysis (SHAKE-type).",
        "description": (
            "Performs 1D equivalent-linear site response analysis using pystrata. "
            "Iteratively adjusts shear modulus and damping to be strain-compatible "
            "using the frequency-domain transfer function approach (Schnabel et al. "
            "1972). Supports Darendeli (2001), Menq (2003), linear, and custom "
            "G/Gmax and damping curves. Returns surface acceleration, response "
            "spectra, and depth profiles of maximum strain, acceleration, and "
            "shear wave velocity."
        ),
        "reference": (
            'Schnabel, P.B., Lysmer, J. & Seed, H.B. (1972). "SHAKE: A Computer '
            'Program for Earthquake Response Analysis of Horizontally Layered '
            'Sites." Report EERC 72-12, UC Berkeley. '
            'Darendeli, M.B. (2001). "Development of a New Family of Normalized '
            'Modulus Reduction and Material Damping Curves." PhD Dissertation, '
            'UT Austin.'
        ),
        "parameters": {
            "layers": {
                "type": "list of dict", "required": True,
                "description": (
                    "Soil layers from surface to bedrock. Each dict requires: "
                    "thickness (m, 0 for bedrock half-space), Vs (m/s), "
                    "unit_wt (kN/m3), soil_model ('darendeli', 'menq', 'linear', "
                    "'custom'). Darendeli requires: plas_index. Optional: ocr "
                    "(default 1), stress_mean (auto-calculated). Menq optional: "
                    "uniformity_coeff (default 10), diam_mean (default 5). "
                    "Linear requires: damping (decimal). Custom requires: "
                    "strains, mod_reduc, damping_values (all arrays). "
                    "Last layer must be bedrock half-space (thickness=0)."
                ),
            },
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
            "strain_ratio": {
                "type": "float", "required": False,
                "default": 0.65, "range": "0.5 to 1.0",
                "description": (
                    "Ratio of effective to maximum strain for EQL iteration."
                ),
            },
            "tolerance": {
                "type": "float", "required": False,
                "default": 0.01,
                "description": "Convergence tolerance for EQL iterations.",
            },
            "max_iterations": {
                "type": "int", "required": False,
                "default": 15, "range": "1 to 50",
                "description": "Maximum number of EQL iterations.",
            },
            "max_freq_hz": {
                "type": "float", "required": False,
                "default": 25.0,
                "description": (
                    "Max frequency for profile auto-discretization (Hz)."
                ),
            },
            "wave_frac": {
                "type": "float", "required": False,
                "default": 0.2,
                "description": (
                    "Wavelength fraction for auto-discretization (1/5)."
                ),
            },
        },
        "returns": {
            "analysis_type": "Analysis type (equivalent_linear).",
            "total_depth_m": "Total profile depth (m).",
            "n_layers": "Number of soil layers.",
            "motion_name": "Input motion name.",
            "pga_input_g": "Peak input acceleration (g).",
            "pga_surface_g": "Peak surface acceleration (g).",
            "amplification_factor": "PGA amplification (surface/input).",
            "n_iterations": "Number of EQL iterations.",
            "converged": "Whether iterations converged.",
            "max_shear_strain_pct": "Maximum shear strain in profile (%).",
        },
    },
    "linear_site_response": {
        "category": "Site Response",
        "brief": "1D linear elastic site response analysis.",
        "description": (
            "Performs 1D linear elastic site response analysis using pystrata. "
            "No iteration — uses initial (small-strain) properties throughout. "
            "Useful as a baseline comparison or for low-strain scenarios. "
            "Same profile/motion input format as eql_site_response."
        ),
        "reference": (
            'Kramer, S.L. (1996). "Geotechnical Earthquake Engineering." '
            'Prentice Hall, Chapter 7.'
        ),
        "parameters": {
            "layers": {
                "type": "list of dict", "required": True,
                "description": "Same format as eql_site_response layers.",
            },
            "motion": {
                "type": "str", "required": False,
                "description": "Built-in motion name.",
            },
            "accel_history": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration time history (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step for custom motion (s).",
            },
            "max_freq_hz": {
                "type": "float", "required": False,
                "default": 25.0,
                "description": (
                    "Max frequency for auto-discretization (Hz)."
                ),
            },
            "wave_frac": {
                "type": "float", "required": False,
                "default": 0.2,
                "description": (
                    "Wavelength fraction for auto-discretization."
                ),
            },
        },
        "returns": {
            "analysis_type": "Analysis type (linear_elastic).",
            "total_depth_m": "Total profile depth (m).",
            "n_layers": "Number of soil layers.",
            "motion_name": "Input motion name.",
            "pga_input_g": "Peak input acceleration (g).",
            "pga_surface_g": "Peak surface acceleration (g).",
            "amplification_factor": "PGA amplification (surface/input).",
            "max_shear_strain_pct": "Maximum shear strain in profile (%).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry agent functions
# ---------------------------------------------------------------------------

@function
def pystrata_agent(method: str, parameters_json: str) -> str:
    """
    pyStrata equivalent-linear site response analysis agent.

    Wraps the pystrata library for 1D site response analysis using
    frequency-domain equivalent-linear or linear elastic methods.

    Call pystrata_list_methods() first to see available analyses, then
    pystrata_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "eql_site_response").
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

    from pystrata_agent.pystrata_utils import has_pystrata
    if not has_pystrata():
        return json.dumps({
            "error": "pystrata is not installed. "
                     "Install with: pip install pystrata"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(clean_numpy(result), default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def pystrata_list_methods(category: str = "") -> str:
    """
    Lists available pystrata analysis methods.

    Use this to discover what analyses are available before calling
    pystrata_agent.

    Parameters:
        category: Optional filter (e.g. "Site Response"). Leave empty for all.

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
def pystrata_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a pystrata analysis method.

    Use this to understand what parameters a method needs before calling
    pystrata_agent.

    Parameters:
        method: The method name (e.g. "eql_site_response").

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
