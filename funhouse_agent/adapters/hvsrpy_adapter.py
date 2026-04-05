"""HVSR adapter — flat dict -> hvsrpy_agent API -> dict."""

from funhouse_agent.adapters import clean_result


def _run_hvsr_analysis(params: dict) -> dict:
    from hvsrpy_agent import analyze_hvsr, has_hvsrpy
    if not has_hvsrpy():
        return {"error": "hvsrpy is not installed. Install with: pip install hvsrpy"}

    result = analyze_hvsr(
        ns=params["ns"],
        ew=params["ew"],
        vt=params["vt"],
        dt=params["dt"],
        window_length_s=params.get("window_length_s", 60.0),
        filter_hz=params.get("filter_hz"),
        smoothing_operator=params.get("smoothing_operator", "konno_and_ohmachi"),
        smoothing_bandwidth=params.get("smoothing_bandwidth", 40),
        freq_min=params.get("freq_min", 0.2),
        freq_max=params.get("freq_max", 50.0),
        n_freq=params.get("n_freq", 200),
        horizontal_method=params.get("horizontal_method", "geometric_mean"),
        distribution=params.get("distribution", "lognormal"),
        rejection_n_std=params.get("rejection_n_std", 2.0),
        rejection_max_iterations=params.get("rejection_max_iterations", 50),
        degrees_from_north=params.get("degrees_from_north", 0.0),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "analyze_hvsr": _run_hvsr_analysis,
}

METHOD_INFO = {
    "analyze_hvsr": {
        "category": "Site Characterization",
        "brief": "Compute HVSR from 3-component seismograms to identify site resonant frequency and amplification.",
        "parameters": {
            "ns": {"type": "array", "required": True, "description": "North-south component amplitudes."},
            "ew": {"type": "array", "required": True, "description": "East-west component amplitudes."},
            "vt": {"type": "array", "required": True, "description": "Vertical component amplitudes."},
            "dt": {"type": "float", "required": True, "description": "Sampling interval (seconds)."},
            "window_length_s": {"type": "float", "required": False, "default": 60.0, "description": "Window length for spectral estimation (s)."},
            "filter_hz": {"type": "array", "required": False, "description": "Bandpass filter [fmin, fmax] in Hz. None = no filter."},
            "smoothing_operator": {"type": "str", "required": False, "default": "konno_and_ohmachi", "description": "Spectral smoothing operator."},
            "smoothing_bandwidth": {"type": "int", "required": False, "default": 40, "description": "Smoothing bandwidth parameter."},
            "freq_min": {"type": "float", "required": False, "default": 0.2, "description": "Minimum frequency (Hz)."},
            "freq_max": {"type": "float", "required": False, "default": 50.0, "description": "Maximum frequency (Hz)."},
            "n_freq": {"type": "int", "required": False, "default": 200, "description": "Number of frequency points."},
            "horizontal_method": {"type": "str", "required": False, "default": "geometric_mean", "description": "Horizontal combination method."},
            "distribution": {"type": "str", "required": False, "default": "lognormal", "description": "Statistical distribution: lognormal or normal."},
            "rejection_n_std": {"type": "float", "required": False, "default": 2.0, "description": "Number of std deviations for window rejection."},
            "rejection_max_iterations": {"type": "int", "required": False, "default": 50, "description": "Maximum rejection iterations."},
            "degrees_from_north": {"type": "float", "required": False, "default": 0.0, "description": "Rotation angle from north (degrees)."},
        },
        "returns": {
            "f0_hz": "Resonant frequency.",
            "A0": "Peak HVSR amplification.",
            "T0_s": "Site period (1/f0).",
            "n_windows": "Number of time windows used.",
        },
    },
}
