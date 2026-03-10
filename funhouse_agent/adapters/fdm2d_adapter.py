"""FDM2D adapter — flat dict -> fdm2d high-level API -> dict."""

from funhouse_agent.adapters import clean_result


def _run_analyze_gravity(params: dict) -> dict:
    from fdm2d import analyze_gravity
    result = analyze_gravity(
        width=params["width"],
        depth=params["depth"],
        gamma=params["gamma"],
        E=params["E"],
        nu=params["nu"],
        nx=params.get("nx", 10),
        ny=params.get("ny", 10),
        t=params.get("t", 1.0),
        c=params.get("c", 0.0),
        phi=params.get("phi", 0.0),
        psi=params.get("psi", 0.0),
        max_steps=params.get("max_steps", 100000),
        tol=params.get("tol", 1e-5),
        damping=params.get("damping", 0.8),
    )
    return clean_result(result.to_dict())


def _run_analyze_foundation(params: dict) -> dict:
    from fdm2d import analyze_foundation
    result = analyze_foundation(
        B=params["B"],
        q=params["q"],
        depth=params["depth"],
        E=params["E"],
        nu=params["nu"],
        gamma=params.get("gamma", 0.0),
        nx=params.get("nx", 20),
        ny=params.get("ny", 10),
        t=params.get("t", 1.0),
        c=params.get("c", 0.0),
        phi=params.get("phi", 0.0),
        psi=params.get("psi", 0.0),
        max_steps=params.get("max_steps", 100000),
        tol=params.get("tol", 1e-5),
        damping=params.get("damping", 0.8),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "fdm2d_gravity": _run_analyze_gravity,
    "fdm2d_foundation": _run_analyze_foundation,
}

METHOD_INFO = {
    "fdm2d_gravity": {
        "category": "FDM 2D",
        "brief": "Elastic/MC gravity analysis of a soil column (FLAC-style explicit FDM).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Domain width (m)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "gamma": {"type": "float", "required": True, "description": "Unit weight (kN/m3)."},
            "E": {"type": "float", "required": True, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": True, "description": "Poisson's ratio."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in y."},
            "t": {"type": "float", "required": False, "default": 1.0, "description": "Out-of-plane thickness (m)."},
            "c": {"type": "float", "required": False, "default": 0.0, "description": "Cohesion (kPa). 0 = elastic."},
            "phi": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (degrees). 0 = elastic."},
            "psi": {"type": "float", "required": False, "default": 0.0, "description": "Dilation angle (degrees)."},
            "max_steps": {"type": "int", "required": False, "default": 100000, "description": "Maximum timesteps."},
            "tol": {"type": "float", "required": False, "default": 1e-5, "description": "Convergence tolerance."},
            "damping": {"type": "float", "required": False, "default": 0.8, "description": "Local damping coefficient."},
        },
        "returns": {
            "max_displacement_m": "Maximum displacement magnitude.",
            "max_sigma_yy_kPa": "Maximum vertical stress.",
            "converged": "Whether solution converged.",
            "n_timesteps": "Number of timesteps to convergence.",
        },
    },
    "fdm2d_foundation": {
        "category": "FDM 2D",
        "brief": "Strip foundation analysis on elastic half-space (FLAC-style explicit FDM).",
        "parameters": {
            "B": {"type": "float", "required": True, "description": "Foundation width (m)."},
            "q": {"type": "float", "required": True, "description": "Applied pressure (kPa, positive downward)."},
            "depth": {"type": "float", "required": True, "description": "Domain depth (m)."},
            "E": {"type": "float", "required": True, "description": "Young's modulus (kPa)."},
            "nu": {"type": "float", "required": True, "description": "Poisson's ratio."},
            "gamma": {"type": "float", "required": False, "default": 0.0, "description": "Soil unit weight (kN/m3)."},
            "nx": {"type": "int", "required": False, "default": 20, "description": "Mesh divisions in x."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Mesh divisions in y."},
            "c": {"type": "float", "required": False, "default": 0.0, "description": "Cohesion (kPa)."},
            "phi": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (degrees)."},
            "psi": {"type": "float", "required": False, "default": 0.0, "description": "Dilation angle (degrees)."},
            "max_steps": {"type": "int", "required": False, "default": 100000, "description": "Maximum timesteps."},
            "tol": {"type": "float", "required": False, "default": 1e-5, "description": "Convergence tolerance."},
            "damping": {"type": "float", "required": False, "default": 0.8, "description": "Local damping coefficient."},
        },
        "returns": {
            "max_displacement_m": "Maximum displacement magnitude.",
            "max_sigma_yy_kPa": "Maximum vertical stress.",
            "converged": "Whether solution converged.",
        },
    },
}
