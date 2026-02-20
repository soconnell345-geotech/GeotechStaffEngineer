"""
Downdrag Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. downdrag_agent          - Run a pile downdrag (negative skin friction) analysis
  2. downdrag_list_methods   - Browse available methods
  3. downdrag_describe_method - Get detailed parameter docs

Implements the Fellenius unified neutral plane method with UFC 3-220-20 coverage.
"""

import json
import math
import numpy as np
try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from downdrag.soil import DowndragSoilLayer, DowndragSoilProfile
from downdrag.analysis import DowndragAnalysis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
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


def _build_soil_profile(params: dict) -> DowndragSoilProfile:
    """Build a DowndragSoilProfile from flat JSON params."""
    layers = []
    for lay in params["layers"]:
        layers.append(DowndragSoilLayer(
            thickness=lay["thickness"],
            soil_type=lay["soil_type"],
            unit_weight=lay["unit_weight"],
            phi=lay.get("phi", 0.0),
            cu=lay.get("cu", 0.0),
            beta=lay.get("beta"),
            alpha=lay.get("alpha"),
            Cc=lay.get("Cc", 0.0),
            Cr=lay.get("Cr", 0.0),
            e0=lay.get("e0", 0.0),
            C_ec=lay.get("C_ec"),
            C_er=lay.get("C_er"),
            sigma_p=lay.get("sigma_p"),
            E_s=lay.get("E_s"),
            nu_s=lay.get("nu_s", 0.3),
            settling=lay.get("settling", False),
            description=lay.get("description", ""),
        ))
    return DowndragSoilProfile(
        layers=layers,
        gwt_depth=params.get("gwt_depth", 0.0),
        gamma_w=params.get("gamma_w", 9.81),
    )


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_downdrag_analysis(params: dict) -> dict:
    """Full downdrag (negative skin friction) analysis."""
    soil = _build_soil_profile(params)

    analysis = DowndragAnalysis(
        soil=soil,
        pile_length=params["pile_length"],
        pile_diameter=params["pile_diameter"],
        pile_perimeter=params.get("pile_perimeter"),
        pile_area=params.get("pile_area"),
        pile_E=params.get("pile_E", 200e6),
        pile_unit_weight=params.get("pile_unit_weight", 24.0),
        Q_dead=params.get("Q_dead", 0.0),
        structural_capacity=params.get("structural_capacity"),
        allowable_settlement=params.get("allowable_settlement"),
        fill_thickness=params.get("fill_thickness", 0.0),
        fill_unit_weight=params.get("fill_unit_weight", 19.0),
        gw_drawdown=params.get("gw_drawdown", 0.0),
        Nt=params.get("Nt"),
        n_sublayers=params.get("n_sublayers", 10),
    )
    result = analysis.compute()
    return result.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "downdrag_analysis": _run_downdrag_analysis,
}

METHOD_INFO = {
    "downdrag_analysis": {
        "category": "Downdrag",
        "brief": "Pile downdrag analysis using the Fellenius unified neutral plane method.",
        "description": (
            "Computes the neutral plane depth, dragload, maximum pile load, and "
            "settlement for a pile subject to negative skin friction from consolidating "
            "soil. Uses Fellenius unified method (2004/2006) with force equilibrium "
            "(drag-from-top = resistance-from-tip). Checks structural limit state "
            "(UFC Eq 6-80 LRFD), geotechnical limit state (dragload correctly excluded "
            "per Fellenius/AASHTO/UFC), and settlement serviceability. Supports both "
            "fill placement and groundwater drawdown as consolidation triggers."
        ),
        "reference": (
            "UFC 3-220-20 (DM7.2, 2025) Eqs 6-45, 6-49–6-54, 6-80; "
            "Fellenius (2004, 2006) unified method; AASHTO LRFD Bridge Design"
        ),
        "parameters": {
            "pile_length": {
                "type": "float", "required": True,
                "description": "Embedded pile length (m).",
            },
            "pile_diameter": {
                "type": "float", "required": True,
                "description": "Pile diameter or equivalent diameter (m).",
            },
            "pile_perimeter": {
                "type": "float", "required": False,
                "description": "Pile perimeter (m). Default: pi * diameter.",
            },
            "pile_area": {
                "type": "float", "required": False,
                "description": "Pile cross-sectional area (m2). Default: pi/4 * diameter^2.",
            },
            "pile_E": {
                "type": "float", "required": False, "default": 200e6,
                "description": "Pile Young's modulus (kPa). Default 200e6 (steel).",
            },
            "pile_unit_weight": {
                "type": "float", "required": False, "default": 24.0,
                "description": "Pile material unit weight (kN/m3). Default 24.0 (concrete). Steel hollow piles: use effective value.",
            },
            "Q_dead": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Dead load at pile head (kN). Only dead load contributes to downdrag.",
            },
            "structural_capacity": {
                "type": "float", "required": False,
                "description": "Factored structural resistance P_r (kN). Required for UFC Eq 6-80 LRFD check.",
            },
            "allowable_settlement": {
                "type": "float", "required": False,
                "description": "Allowable pile settlement (m). Required for serviceability check.",
            },
            "fill_thickness": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Thickness of new fill placed at surface (m). Triggers consolidation settlement.",
            },
            "fill_unit_weight": {
                "type": "float", "required": False, "default": 19.0,
                "description": "Fill unit weight (kN/m3). Default 19.0.",
            },
            "gw_drawdown": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Groundwater drawdown from original GWT (m). Triggers consolidation settlement.",
            },
            "Nt": {
                "type": "float", "required": False,
                "description": "Toe bearing capacity factor. If omitted, estimated from tip layer friction angle.",
            },
            "n_sublayers": {
                "type": "int", "required": False, "default": 10,
                "description": "Number of sublayers per soil layer for discretization.",
            },
            "layers": {
                "type": "array", "required": True,
                "description": (
                    "Array of soil layers from surface down. Each layer object has: "
                    "thickness (m, required), soil_type ('cohesionless' or 'cohesive', required), "
                    "unit_weight (kN/m3, required), phi (deg, for cohesionless), cu (kPa, for cohesive), "
                    "beta (optional override), alpha (optional override), settling (bool, true if "
                    "layer consolidates). Settling cohesive layers need: Cc, Cr, e0 OR C_ec, C_er. "
                    "Settling cohesionless layers need: E_s (kPa). Optional: sigma_p (kPa, "
                    "preconsolidation pressure), nu_s (Poisson's ratio, default 0.3), description."
                ),
            },
            "gwt_depth": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Groundwater table depth from surface (m). Default 0.0 (at surface).",
            },
            "gamma_w": {
                "type": "float", "required": False, "default": 9.81,
                "description": "Unit weight of water (kN/m3). Default 9.81.",
            },
        },
        "returns": {
            "neutral_plane_depth_m": "Depth of neutral plane from pile head (m).",
            "dragload_kN": "Negative skin friction above neutral plane (kN).",
            "max_pile_load_kN": "Maximum axial load at neutral plane (kN). Equals Q_dead + dragload + pile_weight_to_np.",
            "Q_dead_kN": "Applied dead load (kN).",
            "pile_weight_to_np_kN": "Pile self-weight from head to neutral plane (kN).",
            "positive_skin_friction_kN": "Positive shaft resistance below neutral plane (kN).",
            "toe_resistance_kN": "Toe bearing resistance (kN).",
            "total_resistance_kN": "Total resistance = positive_skin + toe (kN).",
            "pile_settlement_m": "Pile settlement at neutral plane (m).",
            "elastic_shortening_m": "Elastic compression above neutral plane (m).",
            "toe_settlement_m": "Settlement of bearing stratum below pile tip (m).",
            "soil_settlement_at_np_m": "Soil settlement at neutral plane depth (m).",
            "pile_length_m": "Pile length (m).",
            "pile_diameter_m": "Pile diameter (m).",
            "z_m": "Depth array along pile (m).",
            "axial_load_kN": "Load distribution Q(z) along pile (kN).",
            "soil_settlement_mm": "Soil settlement profile along pile (mm).",
            "unit_skin_friction_kPa": "Unit skin friction fs(z) along pile (kPa).",
            "structural_ok": "True if LRFD structural check passes (UFC Eq 6-80). Present only if structural_capacity provided.",
            "structural_demand_kN": "LRFD factored demand: 1.25*Q_dead + 1.10*(Q_np - Q_dead). Present only if structural_capacity provided.",
            "geotechnical_ok": "True if Q_dead <= total_resistance. Dragload correctly excluded per Fellenius/AASHTO.",
            "settlement_ok": "True if pile_settlement <= allowable. Present only if allowable_settlement provided.",
        },
        "related": {
            "axial_pile_agent.axial_pile_capacity": "Compute pile capacity before checking downdrag.",
            "settlement_agent.consolidation_settlement": "Estimate settlement causing downdrag.",
        },
        "typical_workflow": (
            "1. Compute pile capacity (axial_pile_agent.axial_pile_capacity)\n"
            "2. Run downdrag analysis (this method)\n"
            "3. Check design_load + dragload < Q_geotechnical at neutral plane\n"
            "4. Check settlement at neutral plane is acceptable"
        ),
        "common_mistakes": [
            "Settling cohesive layers MUST have Cc/e0 or C_ec parameters for neutral plane calculation.",
            "Mark layers as settling=True only if they will consolidate (e.g., under new fill).",
            "Q_dead is the dead load only — do not include live load for downdrag check.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def downdrag_agent(method: str, parameters_json: str) -> str:
    """
    Pile downdrag (negative skin friction) calculator.

    Computes the neutral plane depth, dragload, maximum pile load, and
    settlement using the Fellenius unified method. Checks structural
    (UFC Eq 6-80 LRFD), geotechnical, and settlement limit states.

    Parameters:
        method: The calculation method name. Use downdrag_list_methods() to see options.
        parameters_json: JSON string of parameters. Use downdrag_describe_method() for details.

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
def downdrag_list_methods(category: str = "") -> str:
    """
    Lists available downdrag calculation methods.

    Parameters:
        category: Optional filter by category.

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
    return json.dumps(result)


@function
def downdrag_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a downdrag method.

    Parameters:
        method: The method name (e.g. 'downdrag_analysis').

    Returns:
        JSON string with parameters, types, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
