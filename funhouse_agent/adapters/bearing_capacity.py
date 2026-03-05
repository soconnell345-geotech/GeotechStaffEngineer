"""Bearing capacity adapter — flat dict → BearingCapacityAnalysis → dict."""

from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    bearing_capacity_Nc, bearing_capacity_Nq, bearing_capacity_Ngamma,
)


def _run_bearing_capacity_analysis(params: dict) -> dict:
    footing = Footing(
        width=params["width"],
        length=params.get("length"),
        depth=params.get("depth", 0.0),
        shape=params.get("shape", "strip"),
        base_tilt=params.get("base_tilt", 0.0),
        eccentricity_B=params.get("eccentricity_B", 0.0),
        eccentricity_L=params.get("eccentricity_L", 0.0),
    )
    layer1 = SoilLayer(
        cohesion=params.get("cohesion", 0.0),
        friction_angle=params.get("friction_angle", 0.0),
        unit_weight=params["unit_weight"],
        thickness=params.get("layer1_thickness"),
        description=params.get("layer1_description", ""),
    )
    layer2 = None
    if "layer2_unit_weight" in params:
        layer2 = SoilLayer(
            cohesion=params.get("layer2_cohesion", 0.0),
            friction_angle=params.get("layer2_friction_angle", 0.0),
            unit_weight=params["layer2_unit_weight"],
            description=params.get("layer2_description", ""),
        )
    soil = BearingSoilProfile(layer1=layer1, layer2=layer2, gwt_depth=params.get("gwt_depth"))
    analysis = BearingCapacityAnalysis(
        footing=footing, soil=soil,
        load_inclination=params.get("load_inclination", 0.0),
        ground_slope=params.get("ground_slope", 0.0),
        vertical_load=params.get("vertical_load", 0.0),
        factor_of_safety=params.get("factor_of_safety", 3.0),
        ngamma_method=params.get("ngamma_method", "vesic"),
        factor_method=params.get("factor_method", "vesic"),
    )
    return analysis.compute().to_dict()


def _run_bearing_capacity_factors(params: dict) -> dict:
    phi = params["friction_angle"]
    method = params.get("method", "vesic")
    return {
        "Nc": round(bearing_capacity_Nc(phi), 2),
        "Nq": round(bearing_capacity_Nq(phi), 2),
        "Ngamma": round(bearing_capacity_Ngamma(phi, method), 2),
        "method": method,
        "friction_angle_deg": phi,
    }


METHOD_REGISTRY = {
    "bearing_capacity_analysis": _run_bearing_capacity_analysis,
    "bearing_capacity_factors": _run_bearing_capacity_factors,
}

METHOD_INFO = {
    "bearing_capacity_analysis": {
        "category": "Bearing Capacity",
        "brief": "Full bearing capacity analysis for a shallow foundation (Vesic/Meyerhof/Hansen).",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Footing width B (m)."},
            "length": {"type": "float", "required": False, "description": "Footing length L (m). Omit for strip."},
            "depth": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth Df (m)."},
            "shape": {"type": "str", "required": False, "default": "strip", "description": "strip/square/rectangular/circular."},
            "cohesion": {"type": "float", "required": False, "default": 0.0, "description": "Soil cohesion c (kPa)."},
            "friction_angle": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle phi (degrees)."},
            "unit_weight": {"type": "float", "required": True, "description": "Soil unit weight gamma (kN/m3)."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth below surface (m)."},
            "layer2_cohesion": {"type": "float", "required": False, "description": "Second layer cohesion (kPa)."},
            "layer2_friction_angle": {"type": "float", "required": False, "description": "Second layer phi (degrees)."},
            "layer2_unit_weight": {"type": "float", "required": False, "description": "Second layer gamma (kN/m3). Triggers 2-layer analysis."},
            "layer1_thickness": {"type": "float", "required": False, "description": "First layer thickness (m) for 2-layer."},
            "factor_of_safety": {"type": "float", "required": False, "default": 3.0, "description": "Factor of safety."},
            "factor_method": {"type": "str", "required": False, "default": "vesic", "description": "vesic/meyerhof/hansen."},
        },
        "returns": {
            "q_ultimate_kPa": "Ultimate bearing capacity.",
            "q_allowable_kPa": "Allowable bearing capacity (q_ult / FOS).",
            "Nc": "Cohesion factor.", "Nq": "Overburden factor.", "Ngamma": "Unit weight factor.",
        },
    },
    "bearing_capacity_factors": {
        "category": "Bearing Capacity",
        "brief": "Quick lookup of Nc, Nq, Ngamma for a given friction angle.",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "description": "Friction angle phi (degrees, 0-50)."},
            "method": {"type": "str", "required": False, "default": "vesic", "description": "vesic/meyerhof/hansen."},
        },
        "returns": {"Nc": "Cohesion factor.", "Nq": "Overburden factor.", "Ngamma": "Unit weight factor."},
    },
}
