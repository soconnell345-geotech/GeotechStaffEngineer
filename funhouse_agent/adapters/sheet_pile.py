"""Sheet pile adapter — cantilever and anchored wall analysis."""

from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
from sheet_pile.anchored import analyze_anchored


def _build_soil_layers(params):
    return [WallSoilLayer(
        thickness=l["thickness"], unit_weight=l["unit_weight"],
        friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
        description=l.get("description", ""),
    ) for l in params["layers"]]


def _run_cantilever_wall(params):
    layers = _build_soil_layers(params)
    result = analyze_cantilever(
        excavation_depth=params["excavation_depth"], soil_layers=layers,
        gwt_depth_active=params.get("gwt_depth_active"), gwt_depth_passive=params.get("gwt_depth_passive"),
        surcharge=params.get("surcharge", 0.0), FOS_passive=params.get("FOS_passive", 1.5),
        pressure_method=params.get("pressure_method", "rankine"),
    )
    return result.to_dict()


def _run_anchored_wall(params):
    layers = _build_soil_layers(params)
    result = analyze_anchored(
        excavation_depth=params["excavation_depth"], anchor_depth=params["anchor_depth"],
        soil_layers=layers, gwt_depth_active=params.get("gwt_depth_active"),
        gwt_depth_passive=params.get("gwt_depth_passive"), surcharge=params.get("surcharge", 0.0),
        FOS_passive=params.get("FOS_passive", 1.5),
    )
    return result.to_dict()


METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "anchored_wall": _run_anchored_wall,
}

METHOD_INFO = {
    "cantilever_wall": {
        "category": "Sheet Pile",
        "brief": "Cantilever sheet pile wall: embedment depth and maximum moment.",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, unit_weight, friction_angle, cohesion} dicts."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS applied to passive resistance."},
        },
        "returns": {"embedment_depth_m": "Required embedment.", "max_moment_kNm_per_m": "Maximum bending moment."},
    },
    "anchored_wall": {
        "category": "Sheet Pile",
        "brief": "Anchored sheet pile wall: embedment, anchor force, and moment.",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth below surface (m)."},
            "layers": {"type": "array", "required": True, "description": "Soil layers."},
        },
        "returns": {"embedment_depth_m": "Required embedment.", "anchor_force_kN_per_m": "Anchor force."},
    },
}
