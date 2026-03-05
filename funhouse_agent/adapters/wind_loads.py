"""Wind loads adapter — ASCE 7-22 freestanding walls and fences."""


def _run_velocity_pressure(params):
    from wind_loads import compute_velocity_pressure, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        Ke = compute_Ke(params.pop("elevation_m", 0))
    params["Ke"] = Ke
    return compute_velocity_pressure(**params).to_dict()


def _run_freestanding_wall(params):
    from wind_loads import analyze_freestanding_wall_wind, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        Ke = compute_Ke(params.pop("elevation_m", 0))
    params["Ke"] = Ke
    return analyze_freestanding_wall_wind(**params).to_dict()


def _run_fence_wind(params):
    from wind_loads import analyze_fence_wind, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        Ke = compute_Ke(params.pop("elevation_m", 0))
    params["Ke"] = Ke
    return analyze_fence_wind(**params).to_dict()


METHOD_REGISTRY = {
    "velocity_pressure": _run_velocity_pressure,
    "freestanding_wall": _run_freestanding_wall,
    "fence_wind": _run_fence_wind,
}

METHOD_INFO = {
    "velocity_pressure": {
        "category": "Wind Pressure",
        "brief": "ASCE 7-22 velocity pressure qz at height z.",
        "parameters": {
            "V": {"type": "float", "required": True, "description": "Basic wind speed (m/s)."},
            "z": {"type": "float", "required": True, "description": "Height above ground (m)."},
            "exposure_category": {"type": "str", "required": True, "description": "B, C, or D."},
            "Kzt": {"type": "float", "required": False, "default": 1.0, "description": "Topographic factor."},
        },
        "returns": {"qz_Pa": "Velocity pressure (Pa).", "Kz": "Exposure coefficient."},
    },
    "freestanding_wall": {
        "category": "Freestanding Wall",
        "brief": "Wind forces on freestanding solid wall (ASCE 7-22 Ch 29.3).",
        "parameters": {
            "V": {"type": "float", "required": True, "description": "Basic wind speed (m/s)."},
            "z": {"type": "float", "required": True, "description": "Wall height (m)."},
            "B": {"type": "float", "required": True, "description": "Wall length (m)."},
            "s": {"type": "float", "required": True, "description": "Wall height above ground (m)."},
            "exposure_category": {"type": "str", "required": True, "description": "B, C, or D."},
        },
        "returns": {"F_kN_per_m": "Wind force per unit length.", "M_kNm_per_m": "Overturning moment per unit length."},
    },
    "fence_wind": {
        "category": "Fence",
        "brief": "Wind forces on open (porous) fence.",
        "parameters": {
            "V": {"type": "float", "required": True, "description": "Basic wind speed (m/s)."},
            "z": {"type": "float", "required": True, "description": "Fence height (m)."},
            "B": {"type": "float", "required": True, "description": "Fence length (m)."},
            "s": {"type": "float", "required": True, "description": "Fence height above ground (m)."},
            "solidity_ratio": {"type": "float", "required": True, "description": "Solid fraction (0-1)."},
            "exposure_category": {"type": "str", "required": True, "description": "B, C, or D."},
        },
        "returns": {"F_kN_per_m": "Wind force per unit length."},
    },
}
