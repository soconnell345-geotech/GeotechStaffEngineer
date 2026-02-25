"""
OpenSees Geotechnical Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. opensees_agent        - Run an OpenSees FE analysis
  2. opensees_list_methods - Browse available analysis methods
  3. opensees_describe_method - Get detailed docs for a specific method

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[opensees] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json

try:
    from functions.api import function
except ImportError:
    # Outside Palantir Foundry: provide a no-op decorator
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from opensees_agent.opensees_utils import has_opensees, clean_numpy


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

def _run_pm4sand_dss(params):
    """Wrapper for analyze_pm4sand_dss."""
    from opensees_agent.pm4sand_dss import analyze_pm4sand_dss
    result = analyze_pm4sand_dss(**params)
    return result.to_dict()


def _run_bnwf_pile(params):
    """Wrapper for analyze_bnwf_pile."""
    from opensees_agent.bnwf_pile import analyze_bnwf_pile
    result = analyze_bnwf_pile(**params)
    return result.to_dict()


def _run_site_response(params):
    """Wrapper for analyze_site_response."""
    from opensees_agent.site_response import analyze_site_response
    result = analyze_site_response(**params)
    return result.to_dict()


METHOD_REGISTRY = {
    "pm4sand_cyclic_dss": _run_pm4sand_dss,
    "bnwf_lateral_pile": _run_bnwf_pile,
    "site_response_1d": _run_site_response,
}


# ---------------------------------------------------------------------------
# Method metadata
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "bnwf_lateral_pile": {
        "category": "Lateral Pile",
        "brief": "BNWF lateral pile analysis with PySimple1/TzSimple1 springs.",
        "description": (
            "Builds a Beam on Nonlinear Winkler Foundation model in OpenSees "
            "for lateral pile analysis. The pile is modeled with elastic beam "
            "elements and soil is modeled with PySimple1 (lateral), TzSimple1 "
            "(shaft friction), and QzSimple1 (tip bearing) springs. Reuses "
            "the 7 p-y curve models from the lateral_pile module: Matlock soft "
            "clay, Jeanjean soft clay, Reese stiff clay below/above WT, "
            "API sand, Reese sand, and weak rock."
        ),
        "reference": (
            "API RP2A-WSD (2000); Matlock (1970) OTC 1204; "
            "Reese, Cox & Koop (1974) OTC 2080; "
            "OpenSeesPy BNWF Example: "
            "https://openseespydoc.readthedocs.io/en/latest/src/pile.html"
        ),
        "parameters": {
            "pile_length": {
                "type": "float", "required": True,
                "description": "Embedded pile length (m).",
            },
            "pile_diameter": {
                "type": "float", "required": True,
                "description": "Outer diameter (m).",
            },
            "wall_thickness": {
                "type": "float", "required": True,
                "description": "Pipe pile wall thickness (m). Use 0 for solid.",
            },
            "E_pile": {
                "type": "float", "required": True,
                "description": "Young's modulus of pile (kPa). "
                               "Steel: 200e6; Concrete: 25e6-35e6.",
            },
            "layers": {
                "type": "list of dict", "required": True,
                "description": (
                    "Soil layers. Each dict must have: top (m), bottom (m), "
                    "py_model (str: 'matlock', 'jeanjean', 'stiff_clay_below_wt', "
                    "'stiff_clay_above_wt', 'api_sand', 'reese_sand', 'weak_rock', "
                    "'liquefied_sand'), plus model-specific params (phi, gamma, k, "
                    "su/c, eps50, etc.). liquefied_sand requires only 'diameter'."
                ),
            },
            "lateral_load": {
                "type": "float", "required": False,
                "default": 0.0,
                "description": "Lateral force at pile head (kN).",
            },
            "moment": {
                "type": "float", "required": False,
                "default": 0.0,
                "description": "Moment at pile head (kN-m).",
            },
            "axial_load": {
                "type": "float", "required": False,
                "default": 0.0,
                "description": "Axial force (kN, compression positive).",
            },
            "head_condition": {
                "type": "str", "required": False,
                "default": "free",
                "description": "'free' or 'fixed' head condition.",
            },
            "pile_above_ground": {
                "type": "float", "required": False,
                "default": 0.0,
                "description": "Free length above ground surface (m).",
            },
            "n_elem_per_meter": {
                "type": "float", "required": False,
                "default": 5,
                "description": "Mesh density (elements per meter of pile).",
            },
        },
        "returns": {
            "y_top_mm": "Pile head lateral deflection (mm).",
            "rotation_top_mrad": "Pile head rotation (mrad).",
            "max_deflection_mm": "Maximum lateral deflection (mm).",
            "max_moment_kNm": "Maximum bending moment (kN-m).",
            "max_moment_depth_m": "Depth of maximum moment (m).",
            "converged": "Whether the analysis converged.",
        },
    },
    "site_response_1d": {
        "category": "Site Response",
        "brief": "1D effective-stress site response analysis with pore pressure.",
        "description": (
            "Builds a 1D soil column in OpenSees using SSPquadUP elements with "
            "PressureDependMultiYield02 (sand) and PressureIndependMultiYield "
            "(clay) constitutive models. Applies earthquake ground motion "
            "through a Lysmer-Kuhlemeyer viscous dashpot at the base. Returns "
            "surface acceleration time history, response spectra, and depth "
            "profiles of maximum acceleration, shear strain, and excess pore "
            "pressure ratio."
        ),
        "reference": (
            'Lysmer, J. & Kuhlemeyer, R.L. (1969). "Finite Dynamic Model '
            'for Infinite Media." J. Eng. Mech. Div., ASCE, 95(4), 859-877. '
            'Yang, Z., Elgamal, A., & Parra, E. (2003). "Computational Model '
            'for Cyclic Mobility and Associated Shear Deformation." J. Geotech. '
            'Geoenviron. Eng., 129(12), 1119-1127.'
        ),
        "parameters": {
            "layers": {
                "type": "list of dict", "required": True,
                "description": (
                    "Soil layers (top to bottom). Each dict: "
                    "thickness (m), Vs (m/s), density (Mg/m3), "
                    "material_type ('sand' or 'clay'). "
                    "Sand requires: phi (degrees). Optional: K0. "
                    "Clay requires: su (kPa). "
                    "Optional for all: n_surf (default 20), e_init."
                ),
            },
            "motion": {
                "type": "str", "required": False,
                "description": (
                    "Built-in motion name (e.g. 'synthetic_pulse'). "
                    "Provide either 'motion' or 'accel_history'+'dt'."
                ),
            },
            "accel_history": {
                "type": "list of float", "required": False,
                "description": "Custom acceleration time history (g).",
            },
            "dt": {
                "type": "float", "required": False,
                "description": "Time step for custom motion (s).",
            },
            "gwt_depth": {
                "type": "float", "required": False,
                "default": 0.0,
                "description": "Groundwater table depth from surface (m).",
            },
            "bedrock_Vs": {
                "type": "float", "required": False,
                "default": 760.0, "range": "300 to 3000",
                "description": "Bedrock shear wave velocity (m/s).",
            },
            "bedrock_density": {
                "type": "float", "required": False,
                "default": 2.4, "range": "2.0 to 3.0",
                "description": "Bedrock mass density (Mg/m3).",
            },
            "damping": {
                "type": "float", "required": False,
                "default": 0.02, "range": "0 to 0.1",
                "description": "Target Rayleigh damping ratio.",
            },
            "scale_factor": {
                "type": "float", "required": False,
                "default": 1.0,
                "description": "Scale factor applied to input acceleration.",
            },
            "n_elem_per_layer": {
                "type": "int", "required": False,
                "default": 4, "range": "1 to 20",
                "description": "Number of elements per soil layer.",
            },
        },
        "returns": {
            "total_depth_m": "Total profile depth (m).",
            "n_layers": "Number of soil layers.",
            "motion_name": "Input motion name.",
            "pga_input_g": "Peak input acceleration (g).",
            "pga_surface_g": "Peak surface acceleration (g).",
            "amplification_factor": "PGA amplification (surface/input).",
            "max_shear_strain_pct": "Maximum shear strain in profile (%).",
            "max_ru": "Maximum excess pore pressure ratio in profile.",
        },
    },
    "pm4sand_cyclic_dss": {
        "category": "Cyclic Element Tests",
        "brief": "PM4Sand undrained cyclic direct simple shear analysis.",
        "description": (
            "Runs a single-element undrained cyclic DSS test using the PM4Sand "
            "constitutive model (Boulanger & Ziotopoulou 2017). Builds an "
            "SSPquadUP element, consolidates under vertical stress, then applies "
            "stress-controlled cyclic shear loading. Reports number of cycles "
            "to liquefaction (ru threshold), max pore pressure ratio, and "
            "stress-strain history."
        ),
        "reference": (
            'Boulanger, R.W. & Ziotopoulou, K. (2017). "PM4Sand (Version 3.1): '
            'A Sand Plasticity Model for Earthquake Engineering Applications." '
            "Report No. UCD/CGM-17/01. UC Davis."
        ),
        "parameters": {
            "Dr": {
                "type": "float", "required": True,
                "range": "0.1 to 1.0",
                "description": "Relative density (fraction, e.g. 0.55 for 55%).",
            },
            "G0": {
                "type": "float", "required": True,
                "range": "100 to 1500",
                "description": "Shear modulus coefficient (dimensionless). "
                               "Typical: 400-900. Higher for denser sands.",
            },
            "hpo": {
                "type": "float", "required": True,
                "range": "0.01 to 5.0",
                "description": "Contraction rate parameter. Primary calibration "
                               "parameter controlling liquefaction triggering.",
            },
            "Den": {
                "type": "float", "required": True,
                "range": "1.0 to 2.5",
                "description": "Mass density (Mg/m3).",
            },
            "sigma_v": {
                "type": "float", "required": False,
                "default": 100.0, "range": "1 to 1000",
                "description": "Initial vertical effective stress (kPa).",
            },
            "CSR": {
                "type": "float", "required": False,
                "default": 0.15, "range": "0.01 to 0.5",
                "description": "Cyclic stress ratio to apply.",
            },
            "K0": {
                "type": "float", "required": False,
                "default": 0.5, "range": "0.3 to 1.5",
                "description": "Coefficient of lateral earth pressure.",
            },
            "P_atm": {
                "type": "float", "required": False,
                "default": 101.325,
                "description": "Atmospheric pressure (kPa).",
            },
            "e_max": {
                "type": "float", "required": False,
                "default": 0.8,
                "description": "Maximum void ratio.",
            },
            "e_min": {
                "type": "float", "required": False,
                "default": 0.5,
                "description": "Minimum void ratio.",
            },
            "phi_cv": {
                "type": "float", "required": False,
                "default": 33.0, "range": "25 to 40",
                "description": "Critical state friction angle (degrees).",
            },
            "n_cycles": {
                "type": "int", "required": False,
                "default": 30, "range": "1 to 200",
                "description": "Maximum number of loading cycles.",
            },
            "ru_threshold": {
                "type": "float", "required": False,
                "default": 0.95, "range": "0.5 to 1.0",
                "description": "Pore pressure ratio defining liquefaction.",
            },
        },
        "returns": {
            "Dr": "Relative density used.",
            "sigma_v_kPa": "Vertical effective stress (kPa).",
            "CSR_applied": "Applied cyclic stress ratio.",
            "liquefied": "True if liquefaction was triggered.",
            "n_cycles_to_liq": "Cycles to liquefaction (null if no liq).",
            "max_ru": "Peak excess pore pressure ratio.",
            "max_shear_strain_pct": "Peak shear strain (%).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry agent functions
# ---------------------------------------------------------------------------

@function
def opensees_agent(method: str, parameters_json: str) -> str:
    """
    OpenSees finite element analysis agent for geotechnical engineering.

    Builds, runs, and post-processes OpenSees FE models for geotechnical
    problems including liquefaction triggering, lateral pile analysis, and
    site response.

    Call opensees_list_methods() first to see available analyses, then
    opensees_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "pm4sand_cyclic_dss").
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

    if not has_opensees():
        return json.dumps({
            "error": "openseespy is not installed. "
                     "Install with: pip install openseespy"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(clean_numpy(result), default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def opensees_list_methods(category: str = "") -> str:
    """
    Lists available OpenSees analysis methods.

    Use this to discover what analyses are available before calling
    opensees_agent.

    Parameters:
        category: Optional filter (e.g. "Cyclic Element Tests",
            "Lateral Pile", "Site Response"). Leave empty for all.

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
def opensees_describe_method(method: str) -> str:
    """
    Returns detailed documentation for an OpenSees analysis method.

    Use this to understand what parameters a method needs before calling
    opensees_agent.

    Parameters:
        method: The method name (e.g. "pm4sand_cyclic_dss").

    Returns:
        JSON string with: category, description, parameters (types, ranges,
        defaults), and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
