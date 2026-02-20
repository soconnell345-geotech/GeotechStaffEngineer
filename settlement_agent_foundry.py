"""
Settlement Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. settlement_agent        - Run a settlement calculation
  2. settlement_list_methods - Browse available methods
  3. settlement_describe_method - Get detailed parameter docs

Covers elastic, Schmertmann, consolidation, secondary compression,
and combined settlement analyses with time-rate curves.
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

from settlement.immediate import elastic_settlement, SchmertmannLayer, schmertmann_settlement
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer, total_consolidation_settlement,
)
from settlement.stress_distribution import stress_at_depth
from settlement.time_rate import time_factor, degree_of_consolidation, time_for_consolidation
from settlement.secondary import secondary_settlement
from settlement.analysis import SettlementAnalysis


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


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_elastic_settlement(params: dict) -> dict:
    """Elastic immediate settlement: Se = q*B*(1-nu^2)/Es * Iw."""
    Se = elastic_settlement(
        q=params["q_net"],
        B=params["B"],
        Es=params["Es"],
        nu=params.get("nu", 0.3),
        Iw=params.get("Iw", 1.0),
        shape=params.get("shape", "square"),
    )
    return {
        "immediate_settlement_m": round(Se, 6),
        "immediate_settlement_mm": round(Se * 1000, 2),
        "method": "elastic",
    }


def _run_schmertmann_settlement(params: dict) -> dict:
    """Schmertmann (1978) strain influence factor method."""
    layers = [
        SchmertmannLayer(
            depth_top=lay["depth_top"],
            depth_bottom=lay["depth_bottom"],
            Es=lay["Es"],
        )
        for lay in params["layers"]
    ]
    Se = schmertmann_settlement(
        q_net=params["q_net"],
        q0=params.get("q_overburden", 0.0),
        B=params["B"],
        L=params.get("L", params["B"]),
        layers=layers,
        footing_shape=params.get("shape", "square"),
        time_years=params.get("time_years", 0.0),
    )
    return {
        "schmertmann_settlement_m": round(Se, 6),
        "schmertmann_settlement_mm": round(Se * 1000, 2),
        "method": "schmertmann",
    }


def _run_consolidation_settlement(params: dict) -> dict:
    """Cc/Cr e-log(p) consolidation settlement with layer summation."""
    layers = [
        ConsolidationLayer(
            thickness=lay["thickness"],
            depth_to_center=lay["depth_to_center"],
            e0=lay["e0"],
            Cc=lay["Cc"],
            Cr=lay["Cr"],
            sigma_v0=lay["sigma_v0"],
            sigma_p=lay.get("sigma_p"),
            description=lay.get("description", ""),
        )
        for lay in params["layers"]
    ]

    # Two modes: explicit delta_sigma per layer, or compute from q_net/B/L
    delta_sigmas = params.get("delta_sigmas")
    if delta_sigmas is None and "delta_sigma" in params:
        delta_sigmas = [params["delta_sigma"]] * len(layers)

    layer_results = []
    S_total = 0.0

    if delta_sigmas is not None:
        # Explicit stress increases provided
        for layer, ds in zip(layers, delta_sigmas):
            Sc = consolidation_settlement_layer(layer, ds)
            S_total += Sc
            layer_results.append({
                "depth_m": layer.depth_to_center,
                "thickness_m": layer.thickness,
                "settlement_mm": round(Sc * 1000, 3),
                "delta_sigma_kPa": round(ds, 2),
                "OCR": round(layer.OCR, 2),
            })
    else:
        # Compute stress at each layer depth from q_net, B, L
        q_net = params.get("q_net", 0)
        B = params.get("B", 1.0)
        L = params.get("L", B)
        stress_method = params.get("stress_method", "2:1")

        for layer in layers:
            ds = stress_at_depth(q_net, B, L, layer.depth_to_center,
                                 method=stress_method)
            Sc = consolidation_settlement_layer(layer, ds)
            S_total += Sc
            layer_results.append({
                "depth_m": layer.depth_to_center,
                "thickness_m": layer.thickness,
                "settlement_mm": round(Sc * 1000, 3),
                "delta_sigma_kPa": round(ds, 2),
                "OCR": round(layer.OCR, 2),
            })

    return {
        "consolidation_settlement_m": round(S_total, 6),
        "consolidation_settlement_mm": round(S_total * 1000, 2),
        "layer_breakdown": layer_results,
    }


def _run_combined_settlement(params: dict) -> dict:
    """Full combined analysis: immediate + consolidation + secondary + time curve."""
    # Build consolidation layers if provided
    consol_layers = None
    if "consolidation_layers" in params:
        consol_layers = [
            ConsolidationLayer(
                thickness=lay["thickness"],
                depth_to_center=lay["depth_to_center"],
                e0=lay["e0"],
                Cc=lay["Cc"],
                Cr=lay["Cr"],
                sigma_v0=lay["sigma_v0"],
                sigma_p=lay.get("sigma_p"),
            )
            for lay in params["consolidation_layers"]
        ]

    schmertmann_layers = None
    if "schmertmann_layers" in params:
        schmertmann_layers = [
            SchmertmannLayer(
                depth_top=lay["depth_top"],
                depth_bottom=lay["depth_bottom"],
                Es=lay["Es"],
            )
            for lay in params["schmertmann_layers"]
        ]

    analysis = SettlementAnalysis(
        q_applied=params["q_applied"],
        q_overburden=params.get("q_overburden", 0.0),
        B=params["B"],
        L=params.get("L", params["B"]),
        footing_shape=params.get("shape", "square"),
        immediate_method=params.get("immediate_method", "elastic"),
        Es_immediate=params.get("Es"),
        nu=params.get("nu", 0.3),
        schmertmann_layers=schmertmann_layers,
        consolidation_layers=consol_layers,
        cv=params.get("cv"),
        Hdr=params.get("Hdr"),
        drainage=params.get("drainage", "double"),
        C_alpha=params.get("C_alpha"),
        e0_secondary=params.get("e0_secondary", 1.0),
        t_secondary=params.get("t_secondary", 0.0),
        stress_method=params.get("stress_method", "2:1"),
        time_years_schmertmann=params.get("time_years_schmertmann", 0.0),
    )
    result = analysis.compute()
    return result.to_dict()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "elastic_settlement": _run_elastic_settlement,
    "schmertmann_settlement": _run_schmertmann_settlement,
    "consolidation_settlement": _run_consolidation_settlement,
    "combined_settlement_analysis": _run_combined_settlement,
}

METHOD_INFO = {
    "elastic_settlement": {
        "category": "Immediate Settlement",
        "brief": "Elastic settlement using Se = q*B*(1-nu^2)/Es * Iw.",
        "description": "Simple elastic immediate settlement for a footing on elastic half-space.",
        "reference": "Bowles (1996); Das, Principles of Foundation Engineering",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "Es": {"type": "float", "required": True, "description": "Soil elastic modulus (kPa)."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio."},
            "Iw": {"type": "float", "required": False, "default": 1.0, "description": "Influence factor. Rigid circular=0.79, rigid square=0.82, flexible=1.0."},
            "shape": {"type": "str", "required": False, "default": "square", "description": "'square', 'circular', 'rectangular', or 'strip'. For reference."},
        },
        "returns": {
            "immediate_settlement_m": "Settlement in meters.",
            "immediate_settlement_mm": "Settlement in millimeters.",
        },
        "related": {
            "bearing_capacity_agent.bearing_capacity_analysis": "Get q_allowable first, then use as q_net here.",
            "schmertmann_settlement": "Better method for granular soils.",
            "combined_settlement_analysis": "Full analysis with both immediate + consolidation.",
        },
        "typical_workflow": (
            "1. Compute q_allowable (bearing_capacity_agent.bearing_capacity_analysis)\n"
            "2. Use q_allowable as q_net for elastic settlement\n"
            "3. Check settlement < 25 mm (typical limit for structures)"
        ),
        "common_mistakes": [
            "Using Es in MPa instead of kPa — Es should be in kPa (e.g., 10000 kPa, not 10 MPa).",
            "Confusing q_net with total applied pressure — q_net = q_applied - q_overburden.",
        ],
    },
    "schmertmann_settlement": {
        "category": "Immediate Settlement",
        "brief": "Schmertmann (1978) strain influence factor method for granular soils.",
        "description": (
            "Computes settlement using Schmertmann's modified method with strain influence "
            "factor diagram, depth correction C1, and creep correction C2."
        ),
        "reference": "Schmertmann et al. (1978); FHWA GEC-6",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure (kPa)."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure at footing base (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Default = B."},
            "shape": {"type": "str", "required": False, "default": "square", "description": "'square' or 'strip'."},
            "layers": {"type": "array", "required": True, "description": "Array of soil layers, each with: depth_top (m), depth_bottom (m), Es (kPa)."},
            "time_years": {"type": "float", "required": False, "default": 0.0, "description": "Time for creep correction (years). 0 = no creep."},
        },
        "returns": {
            "schmertmann_settlement_m": "Settlement in meters.",
            "schmertmann_settlement_mm": "Settlement in millimeters.",
        },
        "related": {
            "elastic_settlement": "Simpler elastic method for preliminary estimates.",
            "combined_settlement_analysis": "Full analysis including consolidation.",
        },
        "common_mistakes": [
            "Layers must have depth_top and depth_bottom (not thickness).",
            "Es per layer is in kPa, not MPa.",
            "q_overburden is the overburden at the footing base level, not at the layer depth.",
        ],
    },
    "consolidation_settlement": {
        "category": "Consolidation",
        "brief": "Cc/Cr e-log(p) consolidation settlement for clay layers.",
        "description": (
            "Computes consolidation settlement using the classical e-log(p) method. "
            "Handles NC clay, OC within preconsolidation range, and OC exceeding "
            "preconsolidation. Supports multi-layer summation."
        ),
        "reference": "Terzaghi et al.; Das, Advanced Soil Mechanics",
        "parameters": {
            "layers": {"type": "array", "required": True, "description": (
                "Array of consolidation layers. Each: thickness (m), depth_to_center (m), "
                "e0 (void ratio), Cc (compression index), Cr (recompression index), "
                "sigma_v0 (initial effective stress, kPa), sigma_p (preconsolidation pressure, kPa, omit for NC)."
            )},
            "delta_sigma": {"type": "float", "required": False, "description": "Uniform stress increase for all layers (kPa). OR use delta_sigmas array. If neither is provided, stress is computed from q_net/B/L."},
            "delta_sigmas": {"type": "array", "required": False, "description": "Stress increase for each layer (kPa). Must match number of layers."},
            "q_net": {"type": "float", "required": False, "description": "Net bearing pressure (kPa). Used to auto-compute stress at each layer depth if delta_sigma/delta_sigmas not provided."},
            "B": {"type": "float", "required": False, "description": "Footing width (m). Required if using q_net."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Default = B."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "description": "'2:1', 'boussinesq', or 'westergaard'. Used with q_net."},
        },
        "returns": {
            "consolidation_settlement_m": "Total consolidation settlement (m).",
            "consolidation_settlement_mm": "Total consolidation settlement (mm).",
            "layer_breakdown": "Per-layer settlement details.",
        },
        "related": {
            "ground_improvement_agent.wick_drains": "Accelerate consolidation with wick drains.",
            "ground_improvement_agent.surcharge_preloading": "Pre-load to reduce post-construction settlement.",
            "combined_settlement_analysis": "Add immediate + secondary components.",
        },
        "typical_workflow": (
            "1. Define clay layer properties (e0, Cc, Cr, sigma_v0)\n"
            "2. Compute delta_sigma from bearing_capacity or fill loading\n"
            "3. Run consolidation_settlement\n"
            "4. If settlement too large, consider ground_improvement_agent.wick_drains"
        ),
        "common_mistakes": [
            "Omitting sigma_p for OC clay — if sigma_p is omitted, clay is treated as normally consolidated.",
            "Using delta_sigma = total applied pressure instead of net (subtract overburden if footing is embedded).",
            "Each layer needs thickness, depth_to_center, e0, Cc, Cr, and sigma_v0.",
        ],
    },
    "combined_settlement_analysis": {
        "category": "Combined Analysis",
        "brief": "Full settlement analysis: immediate + consolidation + secondary + time curve.",
        "description": (
            "Comprehensive settlement analysis combining immediate (elastic or Schmertmann), "
            "consolidation (Cc/Cr method), and secondary compression. Generates time-settlement "
            "curve using Terzaghi 1-D consolidation theory."
        ),
        "reference": "FHWA GEC-6; Terzaghi et al.; Schmertmann (1978)",
        "parameters": {
            "q_applied": {"type": "float", "required": True, "description": "Applied bearing pressure at footing base (kPa)."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure before construction (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Default = B."},
            "shape": {"type": "str", "required": False, "default": "square", "description": "'square', 'circular', 'rectangular', or 'strip'."},
            "immediate_method": {"type": "str", "required": False, "default": "elastic", "description": "'elastic' or 'schmertmann'."},
            "Es": {"type": "float", "required": False, "description": "Elastic modulus (kPa). Required for elastic method."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio for elastic method."},
            "schmertmann_layers": {"type": "array", "required": False, "description": "Layers for Schmertmann: depth_top, depth_bottom, Es."},
            "consolidation_layers": {"type": "array", "required": False, "description": "Consolidation layers: thickness, depth_to_center, e0, Cc, Cr, sigma_v0, sigma_p."},
            "cv": {"type": "float", "required": False, "description": "Coefficient of consolidation (m2/year)."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path length (m)."},
            "drainage": {"type": "str", "required": False, "default": "double", "description": "'double' or 'single' drainage."},
            "C_alpha": {"type": "float", "required": False, "description": "Secondary compression index."},
            "e0_secondary": {"type": "float", "required": False, "default": 1.0, "description": "Void ratio for secondary compression."},
            "t_secondary": {"type": "float", "required": False, "default": 0.0, "description": "Time for secondary settlement (years)."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "description": "'2:1', 'boussinesq', or 'westergaard'."},
        },
        "returns": {
            "immediate_mm": "Immediate settlement (mm).",
            "consolidation_mm": "Consolidation settlement (mm).",
            "secondary_mm": "Secondary settlement (mm).",
            "total_mm": "Total settlement (mm).",
            "time_settlement_curve": "Time-settlement data points.",
        },
        "related": {
            "elastic_settlement": "Run immediate component only.",
            "consolidation_settlement": "Run consolidation component only.",
            "calc_package_agent.settlement_package": "Generate PDF calculation package.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def settlement_agent(method: str, parameters_json: str) -> str:
    """
    Settlement calculator for shallow foundations.

    Computes immediate (elastic/Schmertmann), consolidation (Cc/Cr),
    secondary compression, and combined time-dependent settlement.

    Parameters:
        method: The calculation method name. Use settlement_list_methods() to see options.
        parameters_json: JSON string of parameters. Use settlement_describe_method() for details.

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
def settlement_list_methods(category: str = "") -> str:
    """
    Lists available settlement calculation methods.

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
def settlement_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a settlement method.

    Parameters:
        method: The method name (e.g. 'elastic_settlement', 'combined_settlement_analysis').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
