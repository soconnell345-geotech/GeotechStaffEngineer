"""Sheet pile adapter — cantilever and anchored wall analysis."""

from funhouse_agent.adapters import apply_aliases, require_keys, require_params
from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
from sheet_pile.anchored import analyze_anchored

# Layer-dict key aliases the agent commonly uses.
_LAYER_ALIASES = {"gamma": "unit_weight", "phi": "friction_angle",
                  "c": "cohesion", "cu": "cohesion"}


def _build_soil_layers(params, *, method):
    require_params(params, ["layers"], method=method)
    layers = []
    for l in params["layers"]:
        l = apply_aliases(l, _LAYER_ALIASES)
        require_keys(l, ["thickness", "unit_weight"], method=method)
        layers.append(WallSoilLayer(
            thickness=l["thickness"], unit_weight=l["unit_weight"],
            friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
            description=l.get("description", ""),
        ))
    return layers


def _run_cantilever_wall(params):
    layers = _build_soil_layers(params, method="cantilever_wall")
    require_params(params, ["excavation_depth"], method="cantilever_wall")
    result = analyze_cantilever(
        excavation_depth=params["excavation_depth"], soil_layers=layers,
        gwt_depth_active=params.get("gwt_depth_active"), gwt_depth_passive=params.get("gwt_depth_passive"),
        surcharge=params.get("surcharge", 0.0), FOS_passive=params.get("FOS_passive", 1.5),
        pressure_method=params.get("pressure_method", "rankine"),
        embedment_increase=params.get("embedment_increase", 1.0),
    )
    return result.to_dict()


def _run_anchored_wall(params):
    layers = _build_soil_layers(params, method="anchored_wall")
    require_params(params, ["excavation_depth", "anchor_depth"], method="anchored_wall")
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
            "layers": {"type": "array", "required": True, "description": "Array of {thickness (m, required), unit_weight (kN/m3, required; alias gamma), friction_angle (deg, default 30; alias phi), cohesion (kPa, default 0)} dicts."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS applied to passive resistance."},
            "gwt_depth_active": {"type": "float", "required": False, "description": "Groundwater depth on active (retained) side (m)."},
            "gwt_depth_passive": {"type": "float", "required": False, "description": "Groundwater depth on passive (excavated) side (m)."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
            "embedment_increase": {"type": "float", "required": False, "default": 1.0, "description": "Multiplier on the computed embedment (e.g. 1.2 for the traditional 20% rule). Use 1.0 (default) when FOS_passive already provides the safety basis — do not double-count."},
        },
        "returns": {"embedment_depth_m": "Required embedment.", "max_moment_kNm_per_m": "Maximum bending moment."},
    },
    "anchored_wall": {
        "category": "Sheet Pile",
        "brief": "Anchored sheet pile wall: embedment, anchor force, and moment.",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "anchor_depth": {"type": "float", "required": True, "description": "Anchor depth below surface (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness (m, required), unit_weight (kN/m3, required; alias gamma), friction_angle (deg, default 30; alias phi), cohesion (kPa, default 0)} dicts."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "gwt_depth_active": {"type": "float", "required": False, "description": "Groundwater depth on active side (m)."},
            "gwt_depth_passive": {"type": "float", "required": False, "description": "Groundwater depth on passive side (m)."},
        },
        "returns": {"embedment_depth_m": "Required embedment.", "anchor_force_kN_per_m": "Anchor force."},
    },
}
