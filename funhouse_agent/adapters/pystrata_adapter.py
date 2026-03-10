"""pyStrata adapter — equivalent-linear and linear 1D site response."""

from funhouse_agent.adapters import clean_result


def _check_pystrata():
    """Raise ValueError if pystrata is not installed."""
    from pystrata_agent import has_pystrata
    if not has_pystrata():
        raise ValueError(
            "pystrata is not installed. Install with: pip install pystrata"
        )


def _run_eql_site_response(params: dict) -> dict:
    _check_pystrata()
    from pystrata_agent import analyze_eql_site_response

    result = analyze_eql_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        strain_ratio=params.get("strain_ratio", 0.65),
        tolerance=params.get("tolerance", 0.01),
        max_iterations=params.get("max_iterations", 15),
        max_freq_hz=params.get("max_freq_hz", 25.0),
        wave_frac=params.get("wave_frac", 0.2),
    )
    return clean_result(result.to_dict())


def _run_linear_site_response(params: dict) -> dict:
    _check_pystrata()
    from pystrata_agent import analyze_linear_site_response

    result = analyze_linear_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        max_freq_hz=params.get("max_freq_hz", 25.0),
        wave_frac=params.get("wave_frac", 0.2),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "eql_site_response": _run_eql_site_response,
    "linear_site_response": _run_linear_site_response,
}

METHOD_INFO = {
    "eql_site_response": {
        "category": "Site Response",
        "brief": "1D equivalent-linear site response analysis (SHAKE-type, Darendeli/Menq/custom curves).",
        "parameters": {
            "layers": {"type": "array", "brief": "Soil layers from surface to bedrock. Each dict: thickness, Vs, unit_wt, soil_model (darendeli/menq/linear/custom). Last layer is bedrock (thickness=0)."},
            "motion": {"type": "str", "brief": "Built-in motion name (e.g. 'synthetic_pulse').", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "strain_ratio": {"type": "float", "brief": "Effective-to-max shear strain ratio.", "default": 0.65},
            "tolerance": {"type": "float", "brief": "Convergence tolerance.", "default": 0.01},
            "max_iterations": {"type": "int", "brief": "Maximum EQL iterations.", "default": 15},
            "max_freq_hz": {"type": "float", "brief": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "brief": "Wavelength fraction for auto-discretization.", "default": 0.2},
        },
        "returns": {
            "analysis_type": "Analysis type (equivalent_linear).",
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g).",
            "amplification_ratio": "PGA surface / PGA input.",
            "n_iterations": "Number of EQL iterations to converge.",
        },
    },
    "linear_site_response": {
        "category": "Site Response",
        "brief": "1D linear elastic site response analysis (constant small-strain properties).",
        "parameters": {
            "layers": {"type": "array", "brief": "Soil layers from surface to bedrock. Same format as eql_site_response."},
            "motion": {"type": "str", "brief": "Built-in motion name.", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "max_freq_hz": {"type": "float", "brief": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "brief": "Wavelength fraction for auto-discretization.", "default": 0.2},
        },
        "returns": {
            "analysis_type": "Analysis type (linear_elastic).",
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g).",
            "amplification_ratio": "PGA surface / PGA input.",
        },
    },
}
