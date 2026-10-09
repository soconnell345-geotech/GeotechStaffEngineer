"""Seismic signals adapter — response spectra, intensity measures, RotD, signal processing."""

from funhouse_agent.adapters import mark_required, clean_result


def _check_eqsig():
    """Raise ValueError if eqsig is not installed."""
    from seismic_signals_agent import has_eqsig
    if not has_eqsig():
        raise ValueError(
            "eqsig is not installed. Install with: pip install eqsig"
        )


def _require_motion(params, *, method, a_key="motion", b_key="accel_history"):
    """Raise a clear ValueError if neither a built-in motion nor a custom
    history was supplied (a custom history also needs dt)."""
    if params.get(a_key) is None and params.get(b_key) is None:
        raise ValueError(
            f"{method}: provide either '{a_key}' (built-in motion name) or "
            f"'{b_key}' (acceleration array, g) with 'dt' (s)."
        )
    if params.get(b_key) is not None and params.get("dt") is None:
        raise ValueError(f"{method}: '{b_key}' requires 'dt' (time step, s).")


def _run_response_spectrum(params: dict) -> dict:
    _check_eqsig()
    from seismic_signals_agent import analyze_response_spectrum

    _require_motion(params, method="response_spectrum")
    result = analyze_response_spectrum(
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
        periods=params.get("periods"),
        damping=params.get("damping", 0.05),
    )
    return clean_result(result.to_dict())


def _run_intensity_measures(params: dict) -> dict:
    _check_eqsig()
    from seismic_signals_agent import analyze_intensity_measures

    _require_motion(params, method="intensity_measures")
    result = analyze_intensity_measures(
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
        sig_dur_start=params.get("sig_dur_start", 0.05),
        sig_dur_end=params.get("sig_dur_end", 0.95),
    )
    return clean_result(result.to_dict())


def _run_rotd_spectrum(params: dict) -> dict:
    # No pyrotd check: analyze_rotd_spectrum uses pyrotd when it imports and
    # the numpy implementation of the same procedure (rotd_native) when it
    # does not, and says which in 'engine' (live smoke G10).
    from seismic_signals_agent import analyze_rotd_spectrum

    _require_motion(params, method="rotd_spectrum",
                    a_key="motion_a", b_key="accel_history_a")
    _require_motion(params, method="rotd_spectrum",
                    a_key="motion_b", b_key="accel_history_b")
    result = analyze_rotd_spectrum(
        motion_a=params.get("motion_a"),
        accel_history_a=params.get("accel_history_a"),
        motion_b=params.get("motion_b"),
        accel_history_b=params.get("accel_history_b"),
        dt=params.get("dt"),
        periods=params.get("periods"),
        damping=params.get("damping", 0.05),
        percentiles=params.get("percentiles"),
    )
    return clean_result(result.to_dict())


def _run_signal_processing(params: dict) -> dict:
    _check_eqsig()
    from seismic_signals_agent import analyze_signal_processing

    _require_motion(params, method="signal_processing")
    if params.get("bandpass") is None and params.get("baseline_order") is None:
        raise ValueError(
            "signal_processing: say what to do to the record — "
            "bandpass=[f_low, f_high] in Hz (e.g. [0.1, 25]) to filter, "
            "baseline_order=<int> (e.g. 1 for a linear trend) to correct the "
            "baseline, or both. Neither was given.")
    result = analyze_signal_processing(
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
        bandpass=params.get("bandpass"),
        baseline_order=params.get("baseline_order"),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "response_spectrum": _run_response_spectrum,
    "intensity_measures": _run_intensity_measures,
    "rotd_spectrum": _run_rotd_spectrum,
    "signal_processing": _run_signal_processing,
}

METHOD_INFO = {
    "response_spectrum": {
        "category": "Seismic Signals",
        "brief": "Compute response spectrum using Nigam-Jennings algorithm (eqsig).",
        "parameters": {
            "motion": {"type": "str", "brief": "Built-in motion name (e.g. 'synthetic_pulse').", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "target_pga_g": {"type": "float", "brief": "Scale the motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "periods": {"type": "array", "brief": "Spectral periods (s). Default: logspace(-2, 1, 200).", "default": None},
            "damping": {"type": "float", "brief": "Damping ratio (decimal).", "default": 0.05},
        },
        "returns": {
            "motion_name": "Name of input motion (notes any PGA scaling).",
            "n_points": "Number of time steps.",
            "duration_s": "Record duration (s).",
            "dt_s": "Time step (s).",
            "pga_g": "Peak ground acceleration (g).",
            "pgv_m_per_s": "Peak ground velocity (m/s).",
            "pgd_m": "Peak ground displacement (m).",
            "damping": "Damping ratio used.",
            "Sa_max_g": "Peak spectral acceleration (g).",
            "T_peak_s": "Period at peak Sa (s).",
            "periods_s": "Spectral period array (s).",
            "Sa_g": "Spectral acceleration array (g), one per period in periods_s.",
        },
    },
    "intensity_measures": {
        "category": "Seismic Signals",
        "brief": "Compute earthquake intensity measures (Arias intensity, CAV, significant duration).",
        "parameters": {
            "motion": {"type": "str", "brief": "Built-in motion name.", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "target_pga_g": {"type": "float", "brief": "Scale the motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "sig_dur_start": {"type": "float", "brief": "Husid start fraction for significant duration.", "default": 0.05},
            "sig_dur_end": {"type": "float", "brief": "Husid end fraction for significant duration.", "default": 0.95},
        },
        "returns": {
            "motion_name": "Name of input motion (notes any PGA scaling).",
            "pga_g": "Peak ground acceleration (g).",
            "pgv_m_per_s": "Peak ground velocity (m/s).",
            "pgd_m": "Peak ground displacement (m).",
            "arias_intensity_m_per_s": "Arias intensity (m/s).",
            "cav_m_per_s": "Cumulative absolute velocity (m/s).",
            "significant_duration_s": "Significant duration (s).",
            "bracketed_duration_s": "Bracketed duration (s).",
        },
    },
    "rotd_spectrum": {
        "category": "Seismic Signals",
        "brief": "Compute rotated spectral acceleration (RotD0/RotD50/RotD100 or any percentile) from two horizontal components. Uses pyrotd when it imports, otherwise a numpy implementation of the same procedure ('engine' says which).",
        "parameters": {
            "motion_a": {"type": "str", "brief": "Built-in motion name for component A.", "default": None},
            "accel_history_a": {"type": "array", "brief": "Custom acceleration for component A (g).", "default": None},
            "motion_b": {"type": "str", "brief": "Built-in motion name for component B.", "default": None},
            "accel_history_b": {"type": "array", "brief": "Custom acceleration for component B (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step (s). Required for custom motions.", "default": None},
            "periods": {"type": "array", "brief": "Spectral periods (s). Default: logspace(-2, 1, 200).", "default": None},
            "damping": {"type": "float", "brief": "Damping ratio (decimal).", "default": 0.05},
            "percentiles": {"type": "array", "brief": "Percentiles to compute (0-100). Default: [0, 50, 100].", "default": None},
        },
        "returns": {
            "motion_a_name": "Component A motion name.",
            "motion_b_name": "Component B motion name.",
            "n_points": "Number of time steps.",
            "dt_s": "Time step (s).",
            "pga_a_g": "PGA of component A (g).",
            "pga_b_g": "PGA of component B (g).",
            "damping": "Damping ratio used.",
            "peak_rotd50_g": "Peak RotD50 spectral acceleration (g), when 50 was computed.",
            "peak_rotd100_g": "Peak RotD100 spectral acceleration (g), when 100 was computed.",
            "periods_s": "Spectral period array (s).",
            "percentiles": "Percentile labels computed, e.g. ['0', '50', '100'].",
            "spectra": "Dict mapping each percentile label to its Sa array (g), one value per period in periods_s; spectra['50'] is RotD50.",
            "engine": "'pyrotd', or 'numpy' when pyrotd could not be imported.",
            "engine_note": "Why the numpy engine ran (the pyrotd import error); only when engine is 'numpy'.",
        },
    },
    "signal_processing": {
        "category": "Seismic Signals",
        "brief": "Bandpass filtering and/or baseline correction of acceleration time history.",
        "parameters": {
            "motion": {"type": "str", "brief": "Built-in motion name.", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "target_pga_g": {"type": "float", "brief": "Scale the motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Frequency content and duration are kept.", "default": None},
            "bandpass": {"type": "array", "brief": "Bandpass frequencies [f_low, f_high] in Hz. Give this and/or baseline_order (at least one).", "default": None},
            "baseline_order": {"type": "int", "brief": "Polynomial order for baseline correction (0, 1, 2, ...). Give this and/or bandpass (at least one).", "default": None},
        },
        "returns": {
            "motion_name": "Name of input motion (notes any PGA scaling).",
            "n_points": "Number of time steps.",
            "dt_s": "Time step (s).",
            "bandpass_hz": "Bandpass corners [f_low, f_high] (Hz), or null.",
            "baseline_order": "Baseline polynomial order, or null.",
            "filter_applied": "Whether bandpass filter was applied.",
            "baseline_corrected": "Whether baseline correction was applied.",
            "pga_original_g": "Original PGA (g; after any target_pga_g scaling).",
            "pga_processed_g": "Processed PGA (g).",
            "pgv_processed_m_per_s": "Processed PGV (m/s).",
            "pgd_processed_m": "Processed PGD (m).",
        },
    },
}


def _motion_alternatives(method_info: dict) -> dict:
    """Say on each motion parameter that a built-in name OR a custom history
    (+ dt) is needed: neither alone is marked required, one of them is."""
    for info in method_info.values():
        params = info.get("parameters") or {}
        for name, spec in params.items():
            if not isinstance(spec, dict):
                continue
            suffix = name[len("motion"):] if name.startswith("motion") else (
                name[len("accel_history"):] if name.startswith("accel_history")
                else None)
            if suffix is None:
                continue
            spec["description"] = (
                spec.get("description", "").rstrip(". ")
                + f". Give motion{suffix} (a built-in name) OR "
                  f"accel_history{suffix} + dt: one of the two is required.")
    return method_info


# One METHOD_INFO style everywhere: explicit required flags matching the code
# above (live smoke wave 1, G7). Every method needs a motion -- a built-in
# name or a custom history -- which no single parameter carries alone.
mark_required(METHOD_INFO, {})
_motion_alternatives(METHOD_INFO)
