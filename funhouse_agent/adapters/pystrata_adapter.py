"""pyStrata adapter — equivalent-linear and linear 1D site response."""

from funhouse_agent.adapters import (
    check_nested_keys, clean_result, require_params,
)

# Keys every layer dict may carry, and each soil_model's own keys: what
# pystrata_agent reads. Anything else was read with .get and a default and
# silently left out, e.g. a Darendeli layer's 'OCR' ran at ocr = 1 and a
# 'damping' on it was never used (GM-58).
_LAYER_COMMON_KEYS = ("thickness", "Vs", "unit_wt", "soil_model")
_MODEL_KEYS = {
    "darendeli": ("plas_index", "ocr", "stress_mean"),
    "menq": ("uniformity_coeff", "diam_mean", "stress_mean"),
    "linear": ("damping",),
    "custom": ("strains", "mod_reduc", "damping_values"),
}
# Words the near-miss matcher cannot place; only the keys the layer's own
# model reads are offered. Suggestions only, never renamed.
_LAYER_KEY_HINTS = {
    "pi": ("plas_index",), "plasticityindex": ("plas_index",),
    "gamma": ("unit_wt",), "unitweight": ("unit_wt",),
    "model": ("soil_model",), "cu": ("uniformity_coeff",),
    "confiningpressure": ("stress_mean",),
    "meaneffectivestress": ("stress_mean",),
    "d50": ("diam_mean",), "xi": ("damping",), "dampingratio": ("damping",),
    "h": ("thickness",),
}


def _check_layers(layers, *, method):
    """Refuse a key a layer's soil_model does not read (GM-58). A layer
    with no soil_model, or one the module does not know, is left to its
    own check, which names the models."""
    for i, layer in enumerate(layers if isinstance(layers, list) else []):
        model = layer.get("soil_model") if isinstance(layer, dict) else None
        own = _MODEL_KEYS.get(str(model).lower().strip())
        if own is None:
            continue
        check_nested_keys(layer, _LAYER_COMMON_KEYS + own, method=method,
                          label=f"layers[{i}] ({model})",
                          hints=_LAYER_KEY_HINTS)


def _check_pystrata():
    """Raise ValueError if pystrata is not installed or fails to import."""
    from pystrata_agent import has_pystrata, pystrata_import_error
    if not has_pystrata():
        raise ValueError(
            pystrata_import_error()
            or "pystrata is not installed. Install with: pip install pystrata"
        )


def _run_eql_site_response(params: dict) -> dict:
    _check_pystrata()
    from pystrata_agent import analyze_eql_site_response

    require_params(params, ["layers"], method="eql_site_response")
    _check_layers(params["layers"], method="eql_site_response")
    result = analyze_eql_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
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
    _check_layers(params["layers"], method="linear_site_response")
    result = analyze_linear_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
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
            "target_pga_g": {"type": "float", "required": False, "description": "Scale the input motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "strain_ratio": {"type": "float", "required": False, "description": "Effective-to-max shear strain ratio (0.5-1.0).", "default": 0.65},
            "tolerance": {"type": "float", "required": False, "description": "Convergence tolerance as a fraction: stop when no layer's shear modulus or damping changes by more than this between iterations (0.01 = 1 %).", "default": 0.01},
            "max_iterations": {"type": "int", "required": False, "description": "Maximum EQL iterations.", "default": 15},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
        },
        "returns": {
            "analysis_type": "Analysis type (equivalent_linear).",
            "total_depth_m": "Profile depth to the half-space (m).",
            "n_layers": "Number of layers.",
            "motion_name": "Input motion name (notes any PGA scaling).",
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g; after any target_pga_g scaling).",
            "amplification_factor": "PGA surface / PGA input.",
            "n_iterations": "EQL iterations run (wave solves counted), or null when they could not be counted.",
            "converged": "Whether the EQL iterations converged within tolerance.",
            "max_shear_strain_pct": "Peak shear strain in the profile (%).",
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
            "target_pga_g": {"type": "float", "required": False, "description": "Scale the input motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
        },
        "returns": {
            "analysis_type": "Analysis type (linear_elastic).",
            "total_depth_m": "Profile depth to the half-space (m).",
            "n_layers": "Number of layers.",
            "motion_name": "Input motion name (notes any PGA scaling).",
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g; after any target_pga_g scaling).",
            "amplification_factor": "PGA surface / PGA input.",
            "n_iterations": "0 for a linear analysis.",
            "converged": "Always true for a linear analysis.",
            "max_shear_strain_pct": "Peak shear strain in the profile (%).",
        },
    },
}


# What a left-out input with no default stands for (BACKLOG GM-165):
# an unscaled motion.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_UNSCALED = ("the motion as given, not scaled",
             "the input motion's own peak ground acceleration")
declare_when_absent(METHOD_INFO, {
    "eql_site_response": {"target_pga_g": _UNSCALED},
    "linear_site_response": {"target_pga_g": _UNSCALED},
})
