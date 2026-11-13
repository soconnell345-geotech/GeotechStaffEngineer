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
        water_table_depth=params.get("water_table_depth"),
        k0=params.get("k0", 0.5),
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
        water_table_depth=params.get("water_table_depth"),
        k0=params.get("k0", 0.5),
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
            "layers": {"type": "array", "required": True, "description": "Soil layers from surface to bedrock (required). Each dict: thickness (m; the LAST layer is the bedrock half-space, thickness 0), Vs (m/s), unit_wt (kN/m3) and soil_model, plus that model's own keys: 'darendeli' needs plas_index (PI, %; optional ocr, default 1, and stress_mean, the mean effective stress its curves are built for, kPa; left out, it is computed from water_table_depth and k0); 'menq' optional uniformity_coeff (default 10), diam_mean (mm, default 5), stress_mean (as for darendeli); 'linear' needs damping (ratio 0-1, e.g. 0.02); 'custom' needs strains, mod_reduc and damping_values (equal-length lists). At least 2 layers (1 soil + bedrock)."},
            "motion": {"type": "str", "required": False, "description": "Built-in motion name (e.g. 'synthetic_pulse'). Give motion OR accel_history + dt. Either is taken as a rock outcrop motion at the top of the half-space.", "default": None},
            "accel_history": {"type": "array", "required": False, "description": "Custom acceleration time history (g); with dt, instead of motion.", "default": None},
            "dt": {"type": "float", "required": False, "description": "Time step of accel_history (s); required with it.", "default": None},
            "target_pga_g": {"type": "float", "required": False, "description": "Scale the input motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "strain_ratio": {"type": "float", "required": False, "description": "Effective-to-max shear strain ratio (0.5-1.0).", "default": 0.65},
            "tolerance": {"type": "float", "required": False, "description": "Convergence tolerance as a fraction: stop when no layer's shear modulus or damping changes by more than this between iterations (0.01 = 1 %).", "default": 0.01},
            "max_iterations": {"type": "int", "required": False, "description": "Maximum EQL iterations.", "default": 15},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
            "water_table_depth": {"type": "float", "required": False, "description": "Depth to the water table (m below the ground surface). Used only for the mean effective stress behind the Darendeli and Menq curves of a layer with no stress_mean of its own: hydrostatic pore pressure below it. Left out, the profile is taken to have NO water table (effective stress = total stress) and the result's warnings say so; with the water table in the profile that overstates the confining stress and understates the modulus reduction.", "default": None},
            "k0": {"type": "float", "required": False, "description": "At-rest earth pressure coefficient K0 in that mean effective stress, sigma'm = sigma'v (1 + 2 K0) / 3 (NAVFAC DM7.2 Eq. 6-22; 0.5 is Jaky's 1 - sin(phi') at 30 deg).", "default": 0.5},
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
            "max_shear_strain_pct": "Peak shear strain in the profile (%): the largest max_strain_pct of depth_profile.",
            "periods_s": "Periods (s) of Sa_surface_g and Sa_input_g: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 3 and 5 s.",
            "Sa_surface_g": "Surface (outcrop) spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "Sa_input_g": "Input (rock outcrop) spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "Sa_surface_max_g": "Largest surface spectral acceleration over 0.01 to 10 s (g).",
            "period_at_Sa_surface_max_s": "The period of that peak (s).",
            "depth_profile": "The response with depth: rows of top_m, bottom_m, max_strain_pct (peak shear strain at the sub-layer's middle; null where pystrata gave none), max_accel_g (peak acceleration at its top), initial_Vs_m_per_s and compatible_Vs_m_per_s (strain-compatible; equal to the initial value in a linear analysis). One row per sub-layer of the discretised profile, up to 20; more are merged (largest strain and acceleration, travel-time average Vs) and depth_profile_note says so.",
            "water_table_depth_m": "The water table used for the Darendeli/Menq confining stress (null: none taken in the profile, or not used, and stress_inputs_note then says why).",
            "k0": "The K0 used for it; null when not used (stress_inputs_note then says why).",
            "stress_inputs_note": "Present when no Darendeli or Menq layer had its confining stress computed (the profile has none, or each gave stress_mean): water_table_depth and k0 were not used, and why.",
            "depth_profile_note": "Present when sub-layers were merged (how), or when depth_profile is null (why: pystrata's profile arrays were not in the form the module reads).",
            "layer_stresses": "Present when the profile has a Darendeli or Menq layer. One row per such layer (layer counts from 1 at the surface): mid_depth_m, sigma_v_kPa, u_kPa, sigma_v_eff_kPa, stress_mean_kPa (what its curves were built for) and stress_mean_from ('given' or 'computed').",
            "warnings": "Present when there is something to say, e.g. a confining stress computed with no water table given.",
            "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives.",
        },
    },
    "linear_site_response": {
        "category": "Site Response",
        "brief": "1D linear elastic site response analysis (constant small-strain properties).",
        "parameters": {
            "layers": {"type": "array", "required": True, "description": "Soil layers from surface to bedrock (required). Each dict: thickness (m; the LAST layer is the bedrock half-space, thickness 0), Vs (m/s), unit_wt (kN/m3) and soil_model, plus that model's own keys: 'darendeli' needs plas_index (PI, %; optional ocr, default 1, and stress_mean, the mean effective stress its curves are built for, kPa; left out, it is computed from water_table_depth and k0); 'menq' optional uniformity_coeff (default 10), diam_mean (mm, default 5), stress_mean (as for darendeli); 'linear' needs damping (ratio 0-1, e.g. 0.02); 'custom' needs strains, mod_reduc and damping_values (equal-length lists). At least 2 layers (1 soil + bedrock)."},
            "motion": {"type": "str", "required": False, "description": "Built-in motion name (e.g. 'synthetic_pulse'). Give motion OR accel_history + dt. Either is taken as a rock outcrop motion at the top of the half-space.", "default": None},
            "accel_history": {"type": "array", "required": False, "description": "Custom acceleration time history (g); with dt, instead of motion.", "default": None},
            "dt": {"type": "float", "required": False, "description": "Time step of accel_history (s); required with it.", "default": None},
            "target_pga_g": {"type": "float", "required": False, "description": "Scale the input motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "max_freq_hz": {"type": "float", "required": False, "description": "Max frequency for auto-discretization (Hz).", "default": 25.0},
            "wave_frac": {"type": "float", "required": False, "description": "Wavelength fraction for auto-discretization.", "default": 0.2},
            "water_table_depth": {"type": "float", "required": False, "description": "Depth to the water table (m below the ground surface). Used only for the mean effective stress behind the Darendeli and Menq curves of a layer with no stress_mean of its own: hydrostatic pore pressure below it. Left out, the profile is taken to have NO water table (effective stress = total stress) and the result's warnings say so; with the water table in the profile that overstates the confining stress and understates the modulus reduction.", "default": None},
            "k0": {"type": "float", "required": False, "description": "At-rest earth pressure coefficient K0 in that mean effective stress, sigma'm = sigma'v (1 + 2 K0) / 3 (NAVFAC DM7.2 Eq. 6-22; 0.5 is Jaky's 1 - sin(phi') at 30 deg).", "default": 0.5},
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
            "max_shear_strain_pct": "Peak shear strain in the profile (%): the largest max_strain_pct of depth_profile. pystrata's linear calculator computes no strain; the module takes each sub-layer's from its strain history (pystrata's StrainTSOutput at the sub-layer's middle), as the equivalent-linear calculator computes its own.",
            "periods_s": "Periods (s) of Sa_surface_g and Sa_input_g: 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 3 and 5 s.",
            "Sa_surface_g": "Surface (outcrop) spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "Sa_input_g": "Input (rock outcrop) spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "Sa_surface_max_g": "Largest surface spectral acceleration over 0.01 to 10 s (g).",
            "period_at_Sa_surface_max_s": "The period of that peak (s).",
            "depth_profile": "The response with depth: rows of top_m, bottom_m, max_strain_pct (peak shear strain at the sub-layer's middle; null where pystrata gave none), max_accel_g (peak acceleration at its top), initial_Vs_m_per_s and compatible_Vs_m_per_s (strain-compatible; equal to the initial value in a linear analysis). One row per sub-layer of the discretised profile, up to 20; more are merged (largest strain and acceleration, travel-time average Vs) and depth_profile_note says so.",
            "water_table_depth_m": "The water table used for the Darendeli/Menq confining stress (null: none taken in the profile, or not used, and stress_inputs_note then says why).",
            "k0": "The K0 used for it; null when not used (stress_inputs_note then says why).",
            "stress_inputs_note": "Present when no Darendeli or Menq layer had its confining stress computed (the profile has none, or each gave stress_mean): water_table_depth and k0 were not used, and why.",
            "depth_profile_note": "Present when sub-layers were merged (how), or when depth_profile is null (why: pystrata's profile arrays were not in the form the module reads).",
            "layer_stresses": "Present when the profile has a Darendeli or Menq layer. One row per such layer (layer counts from 1 at the surface): mid_depth_m, sigma_v_kPa, u_kPa, sigma_v_eff_kPa, stress_mean_kPa (what its curves were built for) and stress_mean_from ('given' or 'computed').",
            "warnings": "Present when there is something to say, e.g. a confining stress computed with no water table given.",
            "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives.",
        },
    },
}


# What a left-out input with no default stands for (BACKLOG GM-165):
# an unscaled motion.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_UNSCALED = ("the motion as given, not scaled",
             "the input motion's own peak ground acceleration")
_NO_WATER = ("no water table in the profile",
             "effective stress = total stress in the Darendeli/Menq "
             "confining stress; the result's warnings say so")
declare_when_absent(METHOD_INFO, {
    "eql_site_response": {"target_pga_g": _UNSCALED,
                          "water_table_depth": _NO_WATER},
    "linear_site_response": {"target_pga_g": _UNSCALED,
                             "water_table_depth": _NO_WATER},
})

# The keys each item of a list parameter carries, so a key given at the
# top level is refused with a hint naming its list (BACKLOG GM-113).
from funhouse_agent.adapters import item_keys_for, merge_item_keys  # noqa: E402,F401
ITEM_KEYS = item_keys_for(
    METHOD_INFO, "layers", _LAYER_COMMON_KEYS + tuple(
        k for own in _MODEL_KEYS.values() for k in own))
