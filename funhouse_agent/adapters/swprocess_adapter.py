"""swprocess adapter — flat dict -> swprocess_agent API -> dict."""

from funhouse_agent.adapters import clean_result


def _run_masw_dispersion(params: dict) -> dict:
    from swprocess_agent import analyze_masw, has_swprocess
    if not has_swprocess():
        return {"error": "swprocess is not installed. Install with: pip install swprocess"}

    result = analyze_masw(
        traces=params["traces"],
        offsets=params["offsets"],
        dt=params["dt"],
        transform=params.get("transform", "phase_shift"),
        fmin=params.get("fmin", 5.0),
        fmax=params.get("fmax", 100.0),
        vmin=params.get("vmin", 50.0),
        vmax=params.get("vmax", 1000.0),
        nvel=params.get("nvel", 200),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "analyze_masw": _run_masw_dispersion,
}

METHOD_INFO = {
    "analyze_masw": {
        "category": "Site Characterization",
        "brief": "MASW surface wave dispersion analysis from multi-channel seismic data.",
        "parameters": {
            "traces": {"type": "array", "required": True, "description": "Seismograms for each sensor channel (list of 1D arrays)."},
            "offsets": {"type": "array", "required": True, "description": "Source-receiver offset for each channel (m)."},
            "dt": {"type": "float", "required": True, "description": "Sampling interval (seconds)."},
            "transform": {"type": "str", "required": False, "default": "phase_shift", "description": "Wavefield transform: phase_shift/fk/fdbf."},
            "fmin": {"type": "float", "required": False, "default": 5.0, "description": "Minimum frequency (Hz)."},
            "fmax": {"type": "float", "required": False, "default": 100.0, "description": "Maximum frequency (Hz)."},
            "vmin": {"type": "float", "required": False, "default": 50.0, "description": "Minimum phase velocity (m/s)."},
            "vmax": {"type": "float", "required": False, "default": 1000.0, "description": "Maximum phase velocity (m/s)."},
            "nvel": {"type": "int", "required": False, "default": 200, "description": "Number of velocity bins."},
        },
        "returns": {
            "n_channels": "Number of receiver channels.",
            "n_frequencies": "Number of frequency bins in dispersion image.",
            "dispersion_frequencies": "Extracted dispersion curve frequencies.",
            "dispersion_velocities": "Extracted dispersion curve phase velocities.",
        },
    },
}
