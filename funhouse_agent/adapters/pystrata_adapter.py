"""pyStrata adapter — equivalent-linear and linear 1D site response."""

from funhouse_agent.adapters import clean_result, require_params


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

    require_params(params, ["layers"], method="eql_site_response")
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

    require_params(params, ["layers"], method="linear_site_response")
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
            "layers": {"type": "array", "required": True, "description": "Soil layers from surface to bedrock (required). Each dict: thickness (m; the LAST layer is the bedrock half-space, thickness 0), Vs (m/s), unit_wt (kN/m3) and soil_model, plus that model's own keys: 'darendeli' needs plas_index (PI, %; optional ocr, default 1, and stress_mean, kPa, default from depth); 'menq' optional uniformity_coeff (default 10), diam_mean (mm, default 5), stress_mean; 'linear' needs damping (ratio 0-1, e.g. 0.02); 'custom' needs strains, mod_reduc and damping_values (equal-length lists). At least 2 layers (1 soil + bedrock)."},
            "motion": {"type": "str", "required": False, "description": "Built-in motion name (e.g. 'synthetic_pulse'). Give motion OR accel_history + dt.", "default": None},
            "accel_history": {"type": "array", "required": False, "description": "Custom acceleration time history (g); with dt, instead of motion.", "default": None},
            "dt": {"type": "float", "required": False, "description": "Time step of accel_history (s); required with it.", "default": None},
            "strain_ratio": {"type": "float", "required": False, "description": "Effective-to-max shear strain ratio (0.5-1.0).", "default": 0.65},
            "tolerance": {"type": "float", "required": False, "description": "Convergence tolerance.", "default": 0.01},
            "max_iterations": {"type": "int", "required": False, "description": "Maximum EQL iterations.", "default": 15},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
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
            "layers": {"type": "array", "required": True, "description": "Soil layers from surface to bedrock (required). Each dict: thickness (m; the LAST layer is the bedrock half-space, thickness 0), Vs (m/s), unit_wt (kN/m3) and soil_model, plus that model's own keys: 'darendeli' needs plas_index (PI, %; optional ocr, default 1, and stress_mean, kPa, default from depth); 'menq' optional uniformity_coeff (default 10), diam_mean (mm, default 5), stress_mean; 'linear' needs damping (ratio 0-1, e.g. 0.02); 'custom' needs strains, mod_reduc and damping_values (equal-length lists). At least 2 layers (1 soil + bedrock)."},
            "motion": {"type": "str", "required": False, "description": "Built-in motion name (e.g. 'synthetic_pulse'). Give motion OR accel_history + dt.", "default": None},
            "accel_history": {"type": "array", "required": False, "description": "Custom acceleration time history (g); with dt, instead of motion.", "default": None},
            "dt": {"type": "float", "required": False, "description": "Time step of accel_history (s); required with it.", "default": None},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
        },
        "returns": {
            "analysis_type": "Analysis type (linear_elastic).",
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g).",
            "amplification_ratio": "PGA surface / PGA input.",
        },
    },
}
