"""
Wind Loads Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. wind_loads_agent           - Run a wind load calculation
  2. wind_loads_list_methods    - Browse available methods
  3. wind_loads_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_velocity_pressure(params):
    from wind_loads import compute_velocity_pressure, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        elev = params.pop("elevation_m", 0)
        Ke = compute_Ke(elev)
    params["Ke"] = Ke
    result = compute_velocity_pressure(**params)
    return result.to_dict()


def _run_freestanding_wall(params):
    from wind_loads import analyze_freestanding_wall_wind, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        elev = params.pop("elevation_m", 0)
        Ke = compute_Ke(elev)
    params["Ke"] = Ke
    result = analyze_freestanding_wall_wind(**params)
    return result.to_dict()


def _run_fence_wind(params):
    from wind_loads import analyze_fence_wind, compute_Ke
    Ke = params.pop("Ke", None)
    if Ke is None:
        elev = params.pop("elevation_m", 0)
        Ke = compute_Ke(elev)
    params["Ke"] = Ke
    result = analyze_fence_wind(**params)
    return result.to_dict()


def _run_compute_Kz(params):
    from wind_loads import compute_Kz
    Kz = compute_Kz(**params)
    return {"Kz": round(Kz, 4), "z_m": params["z"],
            "exposure_category": params["exposure_category"]}


def _run_compute_Kzt(params):
    from wind_loads import compute_Kzt
    Kzt = compute_Kzt(**params)
    return {"Kzt": round(Kzt, 4), "hill_shape": params.get("hill_shape", "none")}


def _run_compute_Ke(params):
    from wind_loads import compute_Ke
    elev = params.get("elevation_m", 0)
    Ke = compute_Ke(elev)
    return {"Ke": round(Ke, 4), "elevation_m": elev}


def _run_get_Cf(params):
    from wind_loads import get_Cf_freestanding_wall
    Cf = get_Cf_freestanding_wall(**params)
    return {"Cf": round(Cf, 4), "B_over_s": params["B_over_s"],
            "clearance_ratio": params.get("clearance_ratio", 0.0)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "velocity_pressure": _run_velocity_pressure,
    "freestanding_wall": _run_freestanding_wall,
    "fence_wind": _run_fence_wind,
    "compute_Kz": _run_compute_Kz,
    "compute_Kzt": _run_compute_Kzt,
    "compute_Ke": _run_compute_Ke,
    "get_Cf": _run_get_Cf,
}

METHOD_INFO = {
    "velocity_pressure": {
        "category": "Velocity Pressure",
        "brief": "Velocity pressure qz at height z (ASCE 7-22 Eq. 26.10-1).",
        "description": (
            "Computes velocity pressure qz = 0.613 * Kz * Kzt * Kd * Ke * V² (Pa) "
            "per ASCE 7-22 Section 26.10. V is basic wind speed in m/s. Kz is computed "
            "from exposure category (B/C/D) and height. Returns qz in Pa and kPa."
        ),
        "reference": "ASCE 7-22, Section 26.10, Eq. 26.10-1",
        "parameters": {
            "V": {
                "type": "float", "required": True,
                "description": "Basic wind speed (m/s). > 0.",
            },
            "z": {
                "type": "float", "required": True,
                "description": "Height above ground (m). > 0. Values below 4.6m use Kz at 4.6m.",
            },
            "exposure_category": {
                "type": "str", "required": True,
                "choices": ["B", "C", "D"],
                "description": "Exposure category. B=suburban, C=open, D=coastal.",
            },
            "Kzt": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Topographic factor. >= 1.0. Use compute_Kzt for hills.",
            },
            "Kd": {
                "type": "float", "required": False, "default": 0.85,
                "description": "Wind directionality factor. 0.85 for walls (Table 26.6-1).",
            },
            "Ke": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Ground elevation factor. Use compute_Ke or pass elevation_m.",
            },
            "elevation_m": {
                "type": "float", "required": False, "default": 0,
                "description": "Ground elevation (m) — used to auto-compute Ke if Ke not given.",
            },
        },
        "returns": {
            "qz_Pa": "Velocity pressure (Pa).",
            "qz_kPa": "Velocity pressure (kPa).",
            "Kz": "Velocity pressure exposure coefficient.",
            "Kzt": "Topographic factor used.",
            "Kd": "Directionality factor used.",
            "Ke": "Ground elevation factor used.",
        },
        "related": {
            "freestanding_wall": "Full wall wind analysis using this velocity pressure.",
            "compute_Kz": "Compute Kz separately for a given height.",
        },
        "typical_workflow": (
            "1. Determine exposure category (B/C/D) from site conditions\n"
            "2. Compute velocity pressure (this method)\n"
            "3. Use freestanding_wall or fence_wind for design forces"
        ),
        "common_mistakes": [
            "Using mph instead of m/s for wind speed — convert first.",
            "Using Exposure A (removed in ASCE 7-22) — use B, C, or D.",
        ],
    },
    "freestanding_wall": {
        "category": "Wall/Fence",
        "brief": "Wind loads on a solid freestanding wall (ASCE 7-22 Ch 29.3).",
        "description": (
            "Full wind analysis for a solid freestanding wall: velocity pressure qh, "
            "net force coefficient Cf from Figure 29.3-1, design pressure p = qh*G*Cf, "
            "force per unit length, total force, and overturning moment. Supports "
            "clearance (elevated walls)."
        ),
        "reference": "ASCE 7-22, Chapter 29.3, Figure 29.3-1",
        "parameters": {
            "V": {
                "type": "float", "required": True,
                "description": "Basic wind speed (m/s). > 0.",
            },
            "wall_height": {
                "type": "float", "required": True,
                "description": "Wall height s (m). > 0.",
            },
            "wall_length": {
                "type": "float", "required": True,
                "description": "Wall length B (m). > 0.",
            },
            "exposure_category": {
                "type": "str", "required": True,
                "choices": ["B", "C", "D"],
                "description": "Exposure category.",
            },
            "Kzt": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Topographic factor. >= 1.0.",
            },
            "Kd": {
                "type": "float", "required": False, "default": 0.85,
                "description": "Wind directionality factor.",
            },
            "Ke": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Ground elevation factor (or pass elevation_m).",
            },
            "G": {
                "type": "float", "required": False, "default": 0.85,
                "description": "Gust-effect factor. 0.85 for rigid structures.",
            },
            "clearance_height": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Clearance from ground to bottom of wall (m). >= 0.",
            },
            "elevation_m": {
                "type": "float", "required": False, "default": 0,
                "description": "Ground elevation (m) — auto-computes Ke if Ke not given.",
            },
        },
        "returns": {
            "velocity_pressure_Pa": "Velocity pressure at wall top (Pa).",
            "wind_pressure_Pa": "Design wind pressure p = qh*G*Cf (Pa).",
            "force_per_unit_length_kN_m": "Horizontal force per unit wall length (kN/m).",
            "total_force_kN": "Total horizontal force on wall (kN).",
            "overturning_moment_kNm_per_m": "Overturning moment per unit length about base (kN*m/m).",
            "Cf": "Net force coefficient from Figure 29.3-1.",
            "B_over_s": "Wall length-to-height ratio.",
        },
        "related": {
            "fence_wind": "Fence analysis with porosity reduction.",
            "retaining_walls_agent.cantilever_wall": "Combine with retaining wall for above-grade wind.",
        },
        "typical_workflow": (
            "1. Determine wind speed V for site\n"
            "2. Analyze wall wind (this method)\n"
            "3. Combine wind force with earth pressure for retaining wall design"
        ),
        "common_mistakes": [
            "Confusing wall_height (s, exposed height) with total embedded height.",
            "Forgetting to include clearance_height for elevated walls.",
        ],
    },
    "fence_wind": {
        "category": "Wall/Fence",
        "brief": "Wind loads on a porous fence (ASCE 7-22 Ch 29.3, Note 4).",
        "description": (
            "Wind analysis for fences with porosity reduction. The effective force "
            "coefficient Cf_effective = Cf_solid * solidity_ratio. Solidity ratio is "
            "the fraction of solid area to gross area (1.0 = solid, ~0.45 = chain-link)."
        ),
        "reference": "ASCE 7-22, Chapter 29.3, Figure 29.3-1 Note 4",
        "parameters": {
            "V": {
                "type": "float", "required": True,
                "description": "Basic wind speed (m/s). > 0.",
            },
            "fence_height": {
                "type": "float", "required": True,
                "description": "Fence height s (m). > 0.",
            },
            "fence_length": {
                "type": "float", "required": True,
                "description": "Fence length B (m). > 0.",
            },
            "solidity_ratio": {
                "type": "float", "required": True,
                "description": "Solid area / gross area. (0, 1.0]. 1.0=solid, ~0.45=chain-link.",
            },
            "exposure_category": {
                "type": "str", "required": True,
                "choices": ["B", "C", "D"],
                "description": "Exposure category.",
            },
            "Kzt": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Topographic factor.",
            },
            "Kd": {
                "type": "float", "required": False, "default": 0.85,
                "description": "Wind directionality factor.",
            },
            "Ke": {
                "type": "float", "required": False, "default": 1.0,
                "description": "Ground elevation factor.",
            },
            "G": {
                "type": "float", "required": False, "default": 0.85,
                "description": "Gust-effect factor.",
            },
            "clearance_height": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Clearance from ground to bottom of fence (m).",
            },
            "elevation_m": {
                "type": "float", "required": False, "default": 0,
                "description": "Ground elevation (m) — auto-computes Ke if Ke not given.",
            },
        },
        "returns": {
            "velocity_pressure_Pa": "Velocity pressure at fence top (Pa).",
            "wind_pressure_Pa": "Design wind pressure with porosity reduction (Pa).",
            "force_per_unit_length_kN_m": "Horizontal force per unit length (kN/m).",
            "total_force_kN": "Total horizontal force on fence (kN).",
            "overturning_moment_kNm_per_m": "Overturning moment per unit length (kN*m/m).",
            "Cf": "Effective force coefficient (Cf_solid * solidity_ratio).",
            "solidity_ratio": "Solidity ratio used.",
        },
    },
    "compute_Kz": {
        "category": "Coefficients",
        "brief": "Velocity pressure exposure coefficient Kz (Table 26.10-1).",
        "description": (
            "Computes Kz = 2.01 * (z/zg)^(2/alpha) per ASCE 7-22 Table 26.10-1. "
            "Heights below 4.6m (15 ft) are evaluated at 4.6m."
        ),
        "reference": "ASCE 7-22, Table 26.10-1",
        "parameters": {
            "z": {
                "type": "float", "required": True,
                "description": "Height above ground (m). > 0.",
            },
            "exposure_category": {
                "type": "str", "required": True,
                "choices": ["B", "C", "D"],
                "description": "Exposure category.",
            },
        },
        "returns": {
            "Kz": "Velocity pressure exposure coefficient.",
        },
    },
    "compute_Kzt": {
        "category": "Coefficients",
        "brief": "Topographic factor Kzt (Section 26.8, Eq. 26.8-1).",
        "description": (
            "Computes Kzt = (1 + K1*K2*K3)² for hills, ridges, and escarpments. "
            "Returns 1.0 for flat terrain."
        ),
        "reference": "ASCE 7-22, Section 26.8",
        "parameters": {
            "hill_shape": {
                "type": "str", "required": False, "default": "none",
                "choices": ["none", "flat", "2d_ridge", "2d_escarpment", "3d_hill"],
                "description": "Topographic feature type.",
            },
            "H_hill": {
                "type": "float", "required": False, "default": 0,
                "description": "Height of hill/escarpment (m). >= 0.",
            },
            "Lh": {
                "type": "float", "required": False, "default": 1,
                "description": "Distance from crest to half-height on upwind side (m). > 0.",
            },
            "x_distance": {
                "type": "float", "required": False, "default": 0,
                "description": "Horizontal distance from crest (m).",
            },
            "z_height": {
                "type": "float", "required": False, "default": 0,
                "description": "Height above local ground (m). >= 0.",
            },
        },
        "returns": {
            "Kzt": "Topographic factor (>= 1.0).",
        },
    },
    "compute_Ke": {
        "category": "Coefficients",
        "brief": "Ground elevation factor Ke (Table 26.9-1).",
        "description": (
            "Computes Ke = e^(-0.0000362 * ze) where ze is ground elevation "
            "above sea level in meters. Returns 1.0 at sea level."
        ),
        "reference": "ASCE 7-22, Table 26.9-1",
        "parameters": {
            "elevation_m": {
                "type": "float", "required": False, "default": 0,
                "description": "Ground elevation above sea level (m).",
            },
        },
        "returns": {
            "Ke": "Ground elevation factor.",
        },
    },
    "get_Cf": {
        "category": "Coefficients",
        "brief": "Net force coefficient Cf for freestanding walls (Figure 29.3-1).",
        "description": (
            "Looks up Cf from Figure 29.3-1 based on B/s ratio and clearance ratio. "
            "Interpolates between Case A (on ground) and Case C (elevated)."
        ),
        "reference": "ASCE 7-22, Figure 29.3-1",
        "parameters": {
            "B_over_s": {
                "type": "float", "required": True,
                "description": "Wall length B divided by wall height s. > 0.",
            },
            "clearance_ratio": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Clearance h divided by wall height s. >= 0.",
            },
        },
        "returns": {
            "Cf": "Net force coefficient.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def wind_loads_agent(method: str, parameters_json: str) -> str:
    """
    ASCE 7-22 wind loads on freestanding walls and fences.

    Computes velocity pressure (Kz, Kzt, Ke), net force coefficients (Cf),
    and design forces/moments for solid walls and porous fences per
    ASCE 7-22 Chapters 26 and 29.3.

    Call wind_loads_list_methods() first to see available analyses,
    then wind_loads_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "freestanding_wall").
        parameters_json: JSON string of parameters.

    Returns:
        JSON string with analysis results or an error message.
    """
    try:
        params = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def wind_loads_list_methods(category: str = "") -> str:
    """
    Lists available wind loads analysis methods.

    Parameters:
        category: Optional filter (e.g. "Wall/Fence", "Coefficients",
                  "Velocity Pressure"). Leave empty for all.

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]

    if not result:
        cats = sorted(set(i["category"] for i in METHOD_INFO.values()))
        return json.dumps({
            "error": f"No methods found for category '{category}'. "
                     f"Available: {', '.join(cats)}"
        })
    return json.dumps(result)


@function
def wind_loads_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a wind loads analysis method.

    Parameters:
        method: The method name (e.g. "freestanding_wall").

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
