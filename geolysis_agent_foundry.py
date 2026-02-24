"""
geolysis Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. geolysis_agent           - Run a soil classification, SPT, or bearing capacity analysis
  2. geolysis_list_methods    - Browse available methods
  3. geolysis_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - pip install geotech-staff-engineer[geolysis] (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json

try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from geolysis_agent.geolysis_utils import has_geolysis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    """Clean a value for JSON serialization."""
    if v is None:
        return None
    return v


def _clean_result(result):
    """Clean result dict for JSON serialization."""
    if isinstance(result, dict):
        return {k: _clean_value(v) if not isinstance(v, dict) else _clean_result(v)
                for k, v in result.items()}
    return result


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_classify_uscs(params):
    from geolysis_agent import classify_uscs
    result = classify_uscs(**params)
    return _clean_result(result.to_dict())


def _run_classify_aashto(params):
    from geolysis_agent import classify_aashto
    result = classify_aashto(**params)
    return _clean_result(result.to_dict())


def _run_correct_spt(params):
    from geolysis_agent import correct_spt
    result = correct_spt(**params)
    return _clean_result(result.to_dict())


def _run_allowable_bc_spt(params):
    from geolysis_agent import allowable_bc_spt
    result = allowable_bc_spt(**params)
    return _clean_result(result.to_dict())


def _run_ultimate_bc(params):
    from geolysis_agent import ultimate_bc
    result = ultimate_bc(**params)
    return _clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "classify_uscs": _run_classify_uscs,
    "classify_aashto": _run_classify_aashto,
    "correct_spt": _run_correct_spt,
    "allowable_bc_spt": _run_allowable_bc_spt,
    "ultimate_bc": _run_ultimate_bc,
}

METHOD_INFO = {
    "classify_uscs": {
        "category": "Classification",
        "brief": "USCS soil classification from index properties.",
        "description": (
            "Classifies soil using the Unified Soil Classification System (USCS) "
            "based on liquid limit, plastic limit, fines content, sand content, "
            "and gradation (d10, d30, d60). Returns USCS symbol (e.g., 'SW-SC', "
            "'CL', 'Pt') and description. May return dual symbols for borderline "
            "cases (e.g., 'SW-SC,SP-SC')."
        ),
        "reference": "ASTM D2487",
        "parameters": {
            "liquid_limit": {
                "type": "float", "required": False,
                "description": "Liquid limit (%). 0-200. If None, assumes non-plastic or granular.",
            },
            "plastic_limit": {
                "type": "float", "required": False,
                "description": "Plastic limit (%). 0-200. Must be ≤ liquid_limit. If None, assumes non-plastic.",
            },
            "fines": {
                "type": "float", "required": False,
                "description": "Fines content (% passing #200 sieve). 0-100. If None, assumes 0.",
            },
            "sand": {
                "type": "float", "required": False,
                "description": "Sand content (% passing #4, retained on #200). 0-100. If None, assumes 0.",
            },
            "d_10": {
                "type": "float", "required": False,
                "description": "Effective size (mm). ≥ 0 or None. If None, gradation-based classification unavailable.",
            },
            "d_30": {
                "type": "float", "required": False,
                "description": "Particle size at 30% passing (mm). ≥ 0 or None.",
            },
            "d_60": {
                "type": "float", "required": False,
                "description": "Particle size at 60% passing (mm). ≥ 0 or None.",
            },
            "organic": {
                "type": "bool", "required": False, "default": False,
                "description": "True if soil is organic (peat, Pt).",
            },
        },
        "returns": {
            "system": "Classification system ('uscs').",
            "symbol": "USCS symbol (e.g., 'SW-SC', 'CL', 'Pt').",
            "description": "Verbal description of soil type.",
            "liquid_limit": "Liquid limit (%).",
            "plastic_limit": "Plastic limit (%).",
            "plasticity_index": "Plasticity index (%).",
            "fines": "Fines content (%).",
            "sand": "Sand content (%).",
        },
        "related": {
            "classify_aashto": "Also classify using AASHTO system.",
            "correct_spt": "Correct SPT N-values for the classified soil.",
            "bearing_capacity_agent.bearing_capacity_analysis": "Use classification to select appropriate strength parameters.",
        },
        "typical_workflow": (
            "1. Classify soil (this method)\n"
            "2. Also classify AASHTO if needed for highway projects (classify_aashto)\n"
            "3. Correct SPT N-values (correct_spt)\n"
            "4. Use corrected N-value for bearing capacity or settlement"
        ),
        "common_mistakes": [
            "Omitting fines content — required for proper classification.",
            "Not providing d_10/d_30/d_60 for coarse-grained soils — needed for Cu/Cc gradation check.",
            "Setting liquid_limit=0 and plastic_limit=0 for non-plastic soils — this is correct behavior.",
        ],
    },
    "classify_aashto": {
        "category": "Classification",
        "brief": "AASHTO soil classification from index properties.",
        "description": (
            "Classifies soil using the AASHTO (American Association of State Highway "
            "and Transportation Officials) system based on liquid limit, plastic limit, "
            "and fines content. Returns AASHTO symbol with group index in parentheses "
            "(e.g., 'A-7-6(20)'). Group index indicates soil quality (higher = worse subgrade)."
        ),
        "reference": "AASHTO M145",
        "parameters": {
            "liquid_limit": {
                "type": "float", "required": False,
                "description": "Liquid limit (%). 0-200. If None, assumes non-plastic.",
            },
            "plastic_limit": {
                "type": "float", "required": False,
                "description": "Plastic limit (%). 0-200. Must be ≤ liquid_limit. If None, assumes non-plastic.",
            },
            "fines": {
                "type": "float", "required": False,
                "description": "Fines content (% passing #200 sieve). 0-100. If None, assumes 0.",
            },
        },
        "returns": {
            "system": "Classification system ('aashto').",
            "symbol": "AASHTO symbol (e.g., 'A-7-6(20)').",
            "description": "Verbal description of soil type.",
            "group_index": "AASHTO group index (string).",
            "liquid_limit": "Liquid limit (%).",
            "plastic_limit": "Plastic limit (%).",
            "plasticity_index": "Plasticity index (%).",
            "fines": "Fines content (%).",
        },
    },
    "correct_spt": {
        "category": "SPT",
        "brief": "Full SPT N-value correction (energy + overburden + dilatancy).",
        "description": (
            "Corrects Standard Penetration Test (SPT) N-value for energy, overburden "
            "pressure, and optionally dilatancy. Returns N60 (energy-corrected), N1_60 "
            "(overburden-corrected), and final corrected N-value. Supports multiple "
            "correction methods: gibbs, bazaraa, peck, liao (for high overburden), skempton."
        ),
        "reference": "Gibbs & Holtz (1957); Bowles (1996)",
        "parameters": {
            "recorded_spt_n_value": {
                "type": "int", "required": True,
                "description": "Field-recorded SPT N-value (blows per 300 mm). ≥ 0.",
            },
            "eop": {
                "type": "float", "required": True,
                "description": "Effective overburden pressure (kPa). > 0.",
            },
            "energy_percentage": {
                "type": "float", "required": False, "default": 0.6,
                "description": "Energy ratio (decimal, 0-1). 0.6 = 60% energy. Typical range 0.45-0.8.",
            },
            "borehole_diameter": {
                "type": "float", "required": False, "default": 65.0,
                "description": "Borehole diameter (mm). Standard: 65-115 mm.",
            },
            "rod_length": {
                "type": "float", "required": False, "default": 10.0,
                "description": "Rod length (m). > 0.",
            },
            "hammer_type": {
                "type": "str", "required": False, "default": "safety",
                "choices": ["automatic", "donut_1", "donut_2", "safety", "drop", "pin"],
                "description": "Hammer type. 'safety' is most common.",
            },
            "sampler_type": {
                "type": "str", "required": False, "default": "standard",
                "choices": ["standard", "non_standard", "liner_4_dense_sand_and_clay", "liner_4_loose_sand"],
                "description": "Sampler type. 'standard' is most common.",
            },
            "opc_method": {
                "type": "str", "required": False, "default": "gibbs",
                "choices": ["gibbs", "bazaraa", "peck", "liao", "skempton"],
                "description": "Overburden pressure correction method. 'liao' recommended for σ'v > 300 kPa.",
            },
            "dilatancy_corr_method": {
                "type": "str", "required": False,
                "description": "Dilatancy correction method. If None, no dilatancy correction applied. Only for fine sands.",
            },
        },
        "returns": {
            "recorded_n": "Field-recorded SPT N-value.",
            "n60": "Energy-corrected N-value.",
            "n1_60": "Overburden-corrected N-value.",
            "n_corrected": "Final corrected N-value (after dilatancy if applied).",
            "energy_percentage": "Energy ratio used (decimal).",
            "hammer_type": "Hammer type used.",
            "sampler_type": "Sampler type used.",
            "opc_method": "Overburden correction method used.",
            "eop_kpa": "Effective overburden pressure (kPa).",
            "dilatancy_applied": "True if dilatancy correction was applied.",
        },
    },
    "allowable_bc_spt": {
        "category": "Bearing Capacity",
        "brief": "Allowable bearing capacity from SPT (Bowles/Meyerhof/Terzaghi).",
        "description": (
            "Computes allowable bearing capacity for cohesionless soils using empirical "
            "SPT-based correlations. Supports Bowles, Meyerhof, and Terzaghi methods. "
            "Returns allowable bearing capacity (kPa) and allowable load (kN) based on "
            "corrected SPT N-value and tolerable settlement."
        ),
        "reference": "Bowles (1996); Meyerhof (1956); Terzaghi & Peck (1967)",
        "parameters": {
            "corrected_spt_n_value": {
                "type": "float", "required": True,
                "description": "Corrected SPT N-value (N1_60 or fully corrected). ≥ 0.",
            },
            "tol_settlement": {
                "type": "float", "required": False, "default": 25.0,
                "description": "Tolerable settlement (mm). > 0. Standard: 25 mm.",
            },
            "depth": {
                "type": "float", "required": False, "default": 1.5,
                "description": "Foundation depth below ground surface (m). ≥ 0.",
            },
            "width": {
                "type": "float", "required": False, "default": 2.0,
                "description": "Foundation width (m). > 0. For rectangular, use smaller dimension.",
            },
            "shape": {
                "type": "str", "required": False, "default": "square",
                "choices": ["square", "rectangle", "circle", "strip"],
                "description": "Foundation shape.",
            },
            "foundation_type": {
                "type": "str", "required": False, "default": "pad",
                "choices": ["pad", "raft"],
                "description": "Foundation type.",
            },
            "abc_method": {
                "type": "str", "required": False, "default": "bowles",
                "choices": ["bowles", "meyerhof", "terzaghi"],
                "description": "Method for allowable bearing capacity.",
            },
        },
        "returns": {
            "method": "Analysis method used.",
            "bc_type": "'allowable_spt'.",
            "bearing_capacity_kpa": "Allowable bearing capacity (kPa).",
            "allowable_load_kn": "Allowable load (kN).",
            "depth_m": "Foundation depth (m).",
            "width_m": "Foundation width (m).",
            "shape": "Foundation shape.",
            "corrected_spt_n": "Corrected SPT N-value used.",
            "settlement_mm": "Tolerable settlement (mm).",
        },
    },
    "ultimate_bc": {
        "category": "Bearing Capacity",
        "brief": "Ultimate bearing capacity (Vesic/Terzaghi).",
        "description": (
            "Computes ultimate bearing capacity for all soil types (cohesive, cohesionless, "
            "mixed) using Vesic or Terzaghi method. Returns ultimate and allowable bearing "
            "capacity (kPa) and bearing capacity factors (Nc, Nq, Nγ). Vesic method is "
            "recommended for general use."
        ),
        "reference": "Vesic (1973); Terzaghi & Peck (1967)",
        "parameters": {
            "friction_angle": {
                "type": "float", "required": True,
                "description": "Soil friction angle (degrees). 0-50. 0 for pure cohesion (undrained clay).",
            },
            "cohesion": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Soil cohesion (kPa). ≥ 0. 0 for pure friction (clean sand).",
            },
            "moist_unit_wgt": {
                "type": "float", "required": False, "default": 18.0,
                "description": "Moist unit weight of soil (kN/m³). > 0. Typical 16-22.",
            },
            "depth": {
                "type": "float", "required": False, "default": 1.5,
                "description": "Foundation depth below ground surface (m). ≥ 0.",
            },
            "width": {
                "type": "float", "required": False, "default": 2.0,
                "description": "Foundation width (m). > 0.",
            },
            "factor_of_safety": {
                "type": "float", "required": False, "default": 3.0,
                "description": "Factor of safety. > 1.0. Typical 2.5-3.5.",
            },
            "shape": {
                "type": "str", "required": False, "default": "square",
                "choices": ["square", "rectangle", "circle", "strip"],
                "description": "Foundation shape.",
            },
            "ubc_method": {
                "type": "str", "required": False, "default": "vesic",
                "choices": ["vesic", "terzaghi"],
                "description": "Method for ultimate bearing capacity. 'vesic' recommended.",
            },
        },
        "returns": {
            "method": "Analysis method used.",
            "bc_type": "'ultimate'.",
            "bearing_capacity_kpa": "Ultimate bearing capacity (kPa).",
            "allowable_bearing_capacity_kpa": "Allowable bearing capacity (kPa) = ultimate / FoS.",
            "depth_m": "Foundation depth (m).",
            "width_m": "Foundation width (m).",
            "shape": "Foundation shape.",
            "n_c": "Bearing capacity factor Nc.",
            "n_q": "Bearing capacity factor Nq.",
            "n_gamma": "Bearing capacity factor Nγ.",
            "factor_of_safety": "Factor of safety used.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def geolysis_agent(method: str, parameters_json: str) -> str:
    """
    Soil classification, SPT correction, and bearing capacity agent.

    Provides USCS/AASHTO classification, SPT N-value corrections
    (energy, overburden, dilatancy), and bearing capacity analyses
    (allowable from SPT, ultimate from soil properties).

    Call geolysis_list_methods() first to see available analyses,
    then geolysis_describe_method() for parameter details.

    Parameters:
        method: Analysis method name (e.g. "classify_uscs").
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

    if not has_geolysis():
        return json.dumps({
            "error": "geolysis is not installed. Install with: pip install geolysis"
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": f"ValueError: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def geolysis_list_methods(category: str = "") -> str:
    """
    Lists available geolysis analysis methods.

    Parameters:
        category: Optional filter (e.g. "Classification", "SPT", "Bearing Capacity").
                  Leave empty for all.

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
def geolysis_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a geolysis analysis method.

    Parameters:
        method: The method name (e.g. "classify_uscs").

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({
            "error": f"Unknown method '{method}'. Available: {available}"
        })
    return json.dumps(METHOD_INFO[method], default=str)
