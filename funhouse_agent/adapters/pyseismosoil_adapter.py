"""PySeismoSoil adapter — nonlinear soil curves and Vs profile analysis."""

from funhouse_agent.adapters import clean_result


def _run_generate_curves(params: dict) -> dict:
    from pyseismosoil_agent import generate_curves, has_pyseismosoil

    if not has_pyseismosoil():
        return {"error": "PySeismoSoil is not installed. Install via: pip install PySeismoSoil"}

    result = generate_curves(
        model=params.get("model", "MKZ"),
        params=params["params"],
        strain_min=params.get("strain_min", 1e-4),
        strain_max=params.get("strain_max", 10.0),
        n_points=params.get("n_points", 50),
    )
    return clean_result(result.to_dict())


def _run_analyze_vs_profile(params: dict) -> dict:
    from pyseismosoil_agent import analyze_vs_profile, has_pyseismosoil

    if not has_pyseismosoil():
        return {"error": "PySeismoSoil is not installed. Install via: pip install PySeismoSoil"}

    result = analyze_vs_profile(
        thicknesses=params["thicknesses"],
        vs_values=params["vs_values"],
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "generate_curves": _run_generate_curves,
    "analyze_vs_profile": _run_analyze_vs_profile,
}

METHOD_INFO = {
    "generate_curves": {
        "category": "PySeismoSoil",
        "brief": "Generate G/Gmax and damping curves from MKZ or HH constitutive model.",
        "parameters": {
            "model": {
                "type": "str",
                "brief": "Model type: 'MKZ' (Modified Kodner-Zelasko) or 'HH' (Hybrid Hyperbolic).",
                "default": "MKZ",
            },
            "params": {
                "type": "dict",
                "brief": "Model parameters. MKZ requires: gamma_ref, beta, s, Gmax. HH requires: gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d.",
            },
            "strain_min": {
                "type": "float",
                "brief": "Minimum shear strain in percent.",
                "default": 1e-4,
            },
            "strain_max": {
                "type": "float",
                "brief": "Maximum shear strain in percent.",
                "default": 10.0,
            },
            "n_points": {
                "type": "int",
                "brief": "Number of log-spaced strain points.",
                "default": 50,
            },
        },
        "returns": {
            "model": "Model name used (MKZ or HH).",
            "params": "Model parameters used.",
            "n_points": "Number of strain points.",
            "strain_pct": "Shear strain values in percent.",
            "G_Gmax": "Modulus reduction ratio G/Gmax.",
            "damping_pct": "Damping ratio in percent.",
        },
    },
    "analyze_vs_profile": {
        "category": "PySeismoSoil",
        "brief": "Compute Vs30, fundamental frequency, and basin depth from a Vs profile.",
        "parameters": {
            "thicknesses": {
                "type": "array",
                "brief": "Layer thicknesses in meters. Last layer must be 0 (halfspace).",
            },
            "vs_values": {
                "type": "array",
                "brief": "Shear wave velocity for each layer in m/s.",
            },
        },
        "returns": {
            "n_layers": "Number of soil layers.",
            "vs30": "Time-averaged Vs in top 30 m (m/s).",
            "f0_bh": "Fundamental frequency, Borcherdt-Hartzell method (Hz).",
            "f0_ro": "Fundamental frequency, Roesset method (Hz).",
            "z1": "Depth to Vs >= 1000 m/s (m).",
            "z_max": "Maximum profile depth (m).",
            "thicknesses": "Layer thicknesses (m).",
            "vs_values": "Layer Vs values (m/s).",
            "depth_array": "Interface depths (m).",
        },
    },
}
