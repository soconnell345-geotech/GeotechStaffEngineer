"""
SOE (Support of Excavation) Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. soe_agent           - Run an SOE analysis
  2. soe_list_methods    - Browse available methods
  3. soe_describe_method - Get detailed parameter docs

Covers multi-level braced excavations, cantilever walls, stability checks,
ground anchor design, and steel section selection.

FOUNDRY SETUP:
  - pip install geotech-staff-engineer (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import math
try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from soe.geometry import ExcavationGeometry, SOEWallLayer, SupportLevel
from soe.beam_analysis import analyze_braced_excavation, analyze_cantilever_excavation
from soe.earth_pressure import select_apparent_pressure
from soe.wall_sections import (
    select_hp_section, select_sheet_pile, select_w_section,
    check_flexural_demand, list_hp_sections, list_sheet_pile_sections,
)
from soe.stability import (
    check_basal_heave_terzaghi, check_basal_heave_bjerrum_eide,
    check_bottom_blowout, check_piping,
)
from soe.anchor_design import (
    design_ground_anchor, compute_unbonded_length, compute_bond_length,
    select_tendon, list_bond_stress_types, get_bond_stress,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _clean_result(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [_clean_result(item) if isinstance(item, dict) else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


def _build_geometry(params: dict) -> ExcavationGeometry:
    """Build ExcavationGeometry from JSON params."""
    layers = []
    for lay in params["layers"]:
        layers.append(SOEWallLayer(
            thickness=lay["thickness"],
            unit_weight=lay["unit_weight"],
            friction_angle=lay.get("friction_angle", 30.0),
            cohesion=lay.get("cohesion", 0.0),
            soil_type=lay.get("soil_type", "sand"),
            description=lay.get("description", ""),
        ))

    supports = []
    for sup in params.get("supports", []):
        supports.append(SupportLevel(
            depth=sup["depth"],
            support_type=sup.get("support_type", "strut"),
            spacing=sup.get("spacing", 3.0),
            angle_deg=sup.get("angle_deg", 0.0),
            preload_kN_per_m=sup.get("preload_kN_per_m", 0.0),
        ))

    return ExcavationGeometry(
        excavation_depth=params["excavation_depth"],
        soil_layers=layers,
        support_levels=supports,
        surcharge=params.get("surcharge", 10.0),
        gwt_depth=params.get("gwt_depth"),
        excavation_width=params.get("excavation_width", 0.0),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_braced_excavation(params: dict) -> dict:
    """Multi-level braced excavation analysis."""
    geo = _build_geometry(params)
    Fy = params.get("Fy", 345.0)
    result = analyze_braced_excavation(geo, Fy=Fy)
    return result.to_dict()


def _run_cantilever_excavation(params: dict) -> dict:
    """Cantilever (unbraced) excavation wall analysis."""
    geo = _build_geometry(params)
    result = analyze_cantilever_excavation(
        geo,
        FOS_passive=params.get("FOS_passive", 1.5),
        Fy=params.get("Fy", 345.0),
    )
    return result.to_dict()


def _run_apparent_pressure(params: dict) -> dict:
    """Compute apparent earth pressure envelope for given soil profile."""
    layers = []
    for lay in params["layers"]:
        layers.append(SOEWallLayer(
            thickness=lay["thickness"],
            unit_weight=lay["unit_weight"],
            friction_angle=lay.get("friction_angle", 30.0),
            cohesion=lay.get("cohesion", 0.0),
            soil_type=lay.get("soil_type", "sand"),
        ))
    H = params["excavation_depth"]
    surcharge = params.get("surcharge", 0.0)
    return select_apparent_pressure(layers, H, surcharge)


def _run_select_wall_section(params: dict) -> dict:
    """Select lightest adequate steel section for given demand."""
    section_type = params.get("section_type", "hp")
    required_Sx = params["required_Sx_cm3"]

    if section_type == "hp":
        result = select_hp_section(required_Sx)
    elif section_type == "sheet_pile":
        result = select_sheet_pile(required_Sx)
    elif section_type == "w":
        result = select_w_section(required_Sx)
    else:
        return {"error": f"Unknown section_type '{section_type}'. Use 'hp', 'sheet_pile', or 'w'."}

    if result is None:
        return {"error": f"No {section_type} section adequate for Sx = {required_Sx} cm³"}
    return result


def _run_check_basal_heave(params: dict) -> dict:
    """Check basal heave stability (Terzaghi or Bjerrum-Eide)."""
    method = params.get("method", "terzaghi")
    if method == "terzaghi":
        result = check_basal_heave_terzaghi(
            H=params["H"],
            cu=params["cu"],
            gamma=params["gamma"],
            q_surcharge=params.get("q_surcharge", 0.0),
            B=params.get("B", 0.0),
            FOS_required=params.get("FOS_required", 1.5),
        )
    elif method == "bjerrum_eide":
        result = check_basal_heave_bjerrum_eide(
            H=params["H"],
            cu=params["cu"],
            gamma=params["gamma"],
            Be=params["Be"],
            Le=params["Le"],
            q_surcharge=params.get("q_surcharge", 0.0),
            FOS_required=params.get("FOS_required", 1.5),
        )
    else:
        return {"error": f"Unknown method '{method}'. Use 'terzaghi' or 'bjerrum_eide'."}
    return result.to_dict()


def _run_check_bottom_blowout(params: dict) -> dict:
    """Check bottom blowout (hydraulic uplift)."""
    result = check_bottom_blowout(
        D_embed=params["D_embed"],
        hw_excess=params["hw_excess"],
        gamma_soil=params["gamma_soil"],
        gamma_w=params.get("gamma_w", 9.81),
        FOS_required=params.get("FOS_required", 1.5),
    )
    return result.to_dict()


def _run_check_piping(params: dict) -> dict:
    """Check piping (internal erosion)."""
    result = check_piping(
        delta_h=params["delta_h"],
        flow_path=params["flow_path"],
        Gs=params.get("Gs", 2.65),
        void_ratio=params.get("void_ratio", 0.65),
        FOS_required=params.get("FOS_required", 2.0),
    )
    return result.to_dict()


def _run_design_ground_anchor(params: dict) -> dict:
    """Design a ground anchor per GEC-4/PTI."""
    result = design_ground_anchor(
        design_load_kN=params["design_load_kN"],
        anchor_depth=params["anchor_depth"],
        excavation_depth=params["excavation_depth"],
        phi_deg=params["phi_deg"],
        soil_type=params.get("soil_type", "sand_medium"),
        anchor_angle_deg=params.get("anchor_angle_deg", 15.0),
        drill_diameter_mm=params.get("drill_diameter_mm", 150.0),
        tendon_type=params.get("tendon_type", "strand_15mm"),
        FOS_bond=params.get("FOS_bond", 2.0),
        lock_off_pct=params.get("lock_off_pct", 0.80),
        max_load_pct=params.get("max_load_pct", 0.60),
        bond_stress_kPa=params.get("bond_stress_kPa"),
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "braced_excavation": _run_braced_excavation,
    "cantilever_excavation": _run_cantilever_excavation,
    "apparent_pressure": _run_apparent_pressure,
    "select_wall_section": _run_select_wall_section,
    "check_basal_heave": _run_check_basal_heave,
    "check_bottom_blowout": _run_check_bottom_blowout,
    "check_piping": _run_check_piping,
    "design_ground_anchor": _run_design_ground_anchor,
}

METHOD_INFO = {
    "braced_excavation": {
        "category": "Excavation Analysis",
        "brief": "Multi-level braced excavation analysis using apparent pressure envelopes.",
        "description": (
            "Analyzes a multi-level braced excavation using Terzaghi-Peck apparent "
            "earth pressure envelopes and the tributary area method. Computes support "
            "reactions, maximum wall moment, required section modulus, and embedment depth."
        ),
        "reference": "Terzaghi & Peck (1967); FHWA GEC-4; California Trenching Manual",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Total excavation depth H (m)."},
            "layers": {"type": "array", "required": True, "description": (
                "Array of soil layers. Each: thickness (m), unit_weight (kN/m3), "
                "friction_angle (deg, default 30), cohesion (kPa, default 0), "
                "soil_type ('sand', 'soft_clay', 'stiff_clay', default 'sand')."
            )},
            "supports": {"type": "array", "required": True, "description": (
                "Array of support levels. Each: depth (m from top), "
                "support_type ('strut'/'anchor'/'raker', default 'strut'), "
                "spacing (m, default 3.0)."
            )},
            "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa)."},
            "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa)."},
        },
        "returns": {
            "excavation_depth_m": "Excavation depth (m).",
            "n_support_levels": "Number of support levels.",
            "apparent_pressure_type": "'sand', 'soft_clay', or 'stiff_clay'.",
            "max_apparent_pressure_kPa": "Peak apparent pressure (kPa).",
            "support_reactions": "List of {depth_m, load_kN_per_m, type}.",
            "max_moment_kNm_per_m": "Max wall bending moment (kN-m/m).",
            "required_Sx_cm3": "Required section modulus (cm³).",
            "required_embedment_m": "Required embedment below excavation (m).",
            "total_wall_length_m": "Total wall length (m).",
        },
        "related": {
            "cantilever_excavation": "For unbraced walls (H <= 4-5m in sand).",
            "sheet_pile_agent.cantilever_wall": "Classical sheet pile with single anchor.",
            "select_wall_section": "Select HP/sheet pile section from required Sx.",
        },
        "typical_workflow": [
            "1. braced_excavation → get support reactions and required Sx",
            "2. select_wall_section → pick HP or sheet pile section",
            "3. check_basal_heave → verify basal stability (clay only)",
            "4. design_ground_anchor → design anchors if using tiebacks",
        ],
        "common_mistakes": [
            "soil_type must be 'sand', 'soft_clay', or 'stiff_clay' — controls pressure diagram.",
            "Supports must be at depths between 0 and H (exclusive).",
            "Layers must extend below excavation depth for embedment analysis.",
        ],
    },
    "cantilever_excavation": {
        "category": "Excavation Analysis",
        "brief": "Cantilever (unbraced) excavation wall analysis.",
        "description": (
            "Analyzes a cantilever excavation wall using classical Rankine earth "
            "pressure with limit equilibrium. Appropriate for short walls "
            "(typically H <= 4-5m in sand, H <= 3m in clay)."
        ),
        "reference": "USACE EM 1110-2-2504",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth H (m)."},
            "layers": {"type": "array", "required": True, "description": "Same as braced_excavation."},
            "surcharge": {"type": "float", "required": False, "default": 10.0, "description": "Surface surcharge (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS on passive resistance."},
            "Fy": {"type": "float", "required": False, "default": 345.0, "description": "Steel yield strength (MPa)."},
        },
        "returns": {
            "excavation_depth_m": "Excavation depth (m).",
            "Ka": "Active pressure coefficient.",
            "Kp": "Passive pressure coefficient.",
            "required_embedment_m": "Required embedment (m).",
            "total_wall_length_m": "Total wall length (m).",
            "max_moment_kNm_per_m": "Max moment (kN-m/m).",
            "required_Sx_cm3": "Required section modulus (cm³).",
        },
    },
    "apparent_pressure": {
        "category": "Earth Pressure",
        "brief": "Compute apparent earth pressure envelope for given soil profile.",
        "description": (
            "Auto-selects the appropriate Terzaghi-Peck apparent pressure diagram "
            "(sand, soft clay, or stiff clay) from the soil profile. Returns the "
            "envelope type, shape, and maximum pressure ordinate."
        ),
        "reference": "Terzaghi & Peck (1967); Peck (1969)",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth H (m)."},
            "layers": {"type": "array", "required": True, "description": "Same as braced_excavation."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
        },
        "returns": {
            "type": "'sand', 'soft_clay', or 'stiff_clay'.",
            "shape": "'uniform' or 'trapezoidal'.",
            "max_pressure_kPa": "Peak apparent pressure ordinate (kPa).",
        },
    },
    "select_wall_section": {
        "category": "Structural",
        "brief": "Select lightest adequate steel section for given demand.",
        "description": (
            "Selects the lightest HP section, sheet pile, or W section that provides "
            "adequate section modulus for the computed wall demand."
        ),
        "reference": "AISC 16th Ed; Nucor Skyline; ArcelorMittal",
        "parameters": {
            "required_Sx_cm3": {"type": "float", "required": True, "description": "Required section modulus (cm³)."},
            "section_type": {"type": "str", "required": False, "default": "hp", "description": "'hp', 'sheet_pile', or 'w'."},
        },
        "returns": {
            "name": "Section designation (e.g., 'HP14x89').",
            "Sx": "Section modulus (in³).",
            "Sx_cm3": "Section modulus (cm³).",
            "weight": "Weight (lb/ft for HP/W, kg/m² for sheet pile).",
        },
    },
    "check_basal_heave": {
        "category": "Stability",
        "brief": "Check basal heave stability (Terzaghi or Bjerrum-Eide method).",
        "description": (
            "Checks bearing capacity failure at the excavation base in clay. "
            "Terzaghi method: FOS = cu*Nc / (gamma*H + q). "
            "Bjerrum-Eide method: Nc depends on H/Be and Be/Le ratios."
        ),
        "reference": "Terzaghi (1943); Bjerrum & Eide (1956); FHWA GEC-4 §5.7",
        "parameters": {
            "H": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "cu": {"type": "float", "required": True, "description": "Undrained shear strength at base (kPa)."},
            "gamma": {"type": "float", "required": True, "description": "Average unit weight (kN/m³)."},
            "method": {"type": "str", "required": False, "default": "terzaghi", "description": "'terzaghi' or 'bjerrum_eide'."},
            "B": {"type": "float", "required": False, "default": 0.0, "description": "Excavation width (m). For Terzaghi only."},
            "Be": {"type": "float", "required": False, "description": "Excavation width (m). For Bjerrum-Eide."},
            "Le": {"type": "float", "required": False, "description": "Excavation length (m). For Bjerrum-Eide."},
            "q_surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS."},
        },
        "returns": {
            "FOS": "Factor of safety.",
            "passes": "True if FOS >= FOS_required.",
            "resistance": "Resisting force (kPa).",
            "demand": "Driving force (kPa).",
        },
    },
    "check_bottom_blowout": {
        "category": "Stability",
        "brief": "Check bottom blowout (hydraulic uplift) below excavation.",
        "description": (
            "Checks if upward water pressure at the wall toe exceeds the weight "
            "of the soil plug. FOS = gamma_soil * D / (gamma_w * hw)."
        ),
        "reference": "USACE EM 1110-2-2504 §6",
        "parameters": {
            "D_embed": {"type": "float", "required": True, "description": "Embedment depth below excavation (m)."},
            "hw_excess": {"type": "float", "required": True, "description": "Excess piezometric head above excavation (m)."},
            "gamma_soil": {"type": "float", "required": True, "description": "Buoyant unit weight of soil plug (kN/m³)."},
            "gamma_w": {"type": "float", "required": False, "default": 9.81, "description": "Unit weight of water (kN/m³)."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required FOS."},
        },
        "returns": {
            "FOS": "Factor of safety.",
            "passes": "True if FOS >= FOS_required.",
        },
    },
    "check_piping": {
        "category": "Stability",
        "brief": "Check piping (internal erosion) from hydraulic gradient.",
        "description": (
            "Checks if exit hydraulic gradient exceeds critical gradient. "
            "FOS = i_critical / i_exit, where i_critical = (Gs-1)/(1+e)."
        ),
        "reference": "USACE EM 1110-2-2504 §6; Terzaghi (1943)",
        "parameters": {
            "delta_h": {"type": "float", "required": True, "description": "Total head difference (m)."},
            "flow_path": {"type": "float", "required": True, "description": "Shortest seepage path length (m). Typically 2*D."},
            "Gs": {"type": "float", "required": False, "default": 2.65, "description": "Specific gravity of solids."},
            "void_ratio": {"type": "float", "required": False, "default": 0.65, "description": "Void ratio."},
            "FOS_required": {"type": "float", "required": False, "default": 2.0, "description": "Required FOS (higher for piping)."},
        },
        "returns": {
            "FOS": "Factor of safety.",
            "passes": "True if FOS >= FOS_required.",
            "i_critical": "Critical hydraulic gradient.",
            "i_exit": "Exit hydraulic gradient.",
        },
    },
    "design_ground_anchor": {
        "category": "Anchor Design",
        "brief": "Design a ground anchor per GEC-4 and PTI.",
        "description": (
            "Complete ground anchor design: unbonded length (past active wedge), "
            "bond length (grout-ground interface), tendon selection (strand count "
            "or bar size), and test loads (proof = 133% DL, performance = 150% DL)."
        ),
        "reference": "FHWA-IF-99-015 (GEC-4); PTI DC35.1",
        "parameters": {
            "design_load_kN": {"type": "float", "required": True, "description": "Design anchor load per anchor (kN)."},
            "anchor_depth": {"type": "float", "required": True, "description": "Depth of anchor head from surface (m)."},
            "excavation_depth": {"type": "float", "required": True, "description": "Total excavation depth H (m)."},
            "phi_deg": {"type": "float", "required": True, "description": "Friction angle of retained soil (deg)."},
            "soil_type": {"type": "str", "required": False, "default": "sand_medium", "description": (
                "Soil/rock type for bond stress. Options: sand_loose, sand_medium, "
                "sand_dense, gravel, clay_stiff, clay_hard, rock_soft, rock_medium, rock_hard."
            )},
            "anchor_angle_deg": {"type": "float", "required": False, "default": 15.0, "description": "Inclination below horizontal (deg)."},
            "drill_diameter_mm": {"type": "float", "required": False, "default": 150.0, "description": "Drill hole diameter (mm)."},
            "tendon_type": {"type": "str", "required": False, "default": "strand_15mm", "description": "'strand_13mm', 'strand_15mm', or bar types."},
            "FOS_bond": {"type": "float", "required": False, "default": 2.0, "description": "FOS on bond capacity."},
            "bond_stress_kPa": {"type": "float", "required": False, "description": "Override bond stress (kPa). If omitted, uses soil_type lookup."},
        },
        "returns": {
            "unbonded_length_m": "Required free length (m).",
            "bond_length_m": "Required bond length (m).",
            "total_length_m": "Total anchor length (m).",
            "tendon": "Tendon details (type, strand count, capacity).",
            "proof_test_kN": "Proof test load (kN).",
            "performance_test_kN": "Performance test load (kN).",
        },
        "typical_workflow": [
            "1. braced_excavation → get support reactions at each level",
            "2. Convert reaction per m to per-anchor load: load * spacing",
            "3. design_ground_anchor → get anchor geometry and tendon",
        ],
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def soe_agent(method: str, parameters_json: str) -> str:
    """
    Support of excavation design calculator.

    Designs multi-level braced and cantilever excavation walls, checks
    stability (basal heave, blowout, piping), designs ground anchors,
    and selects steel sections.

    Parameters:
        method: The calculation method name. Use soe_list_methods() to see options.
        parameters_json: JSON string of parameters. Use soe_describe_method() for details.

    Returns:
        JSON string with calculation results or an error message.
    """
    try:
        parameters = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    try:
        result = METHOD_REGISTRY[method](parameters)
        return json.dumps(_clean_result(result), default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def soe_list_methods(category: str = "") -> str:
    """
    Lists available SOE analysis methods.

    Parameters:
        category: Optional filter (e.g., 'excavation', 'stability', 'anchor').

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    cat_filter = category.lower() if category else ""
    for method_name, info in METHOD_INFO.items():
        cat = info["category"]
        if cat_filter and cat_filter not in cat.lower() and cat_filter not in method_name:
            continue
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return json.dumps(result)


@function
def soe_describe_method(method: str) -> str:
    """
    Returns detailed documentation for an SOE method.

    Parameters:
        method: The method name (e.g. 'braced_excavation', 'design_ground_anchor').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
