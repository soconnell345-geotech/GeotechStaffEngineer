"""Sheet pile adapter — cantilever and anchored wall analysis."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, field_keys, reject_unknown_params,
    require_keys, require_params,
)
from sheet_pile.cantilever import WallSoilLayer, analyze_cantilever
from sheet_pile.anchored import analyze_anchored

# Layer-dict key aliases the agent commonly uses.
_LAYER_ALIASES = {"gamma": "unit_weight", "phi": "friction_angle",
                  "c": "cohesion", "cu": "cohesion",
                  "delta": "wall_friction_deg", "wall_friction": "wall_friction_deg"}
# Keys a layer dict may carry: WallSoilLayer's fields and the aliases above.
# Anything else was read with .get and silently left out, e.g. 'phi_deg' ran
# the layer at the 30 degree default (GM-58).
_LAYER_KEYS = field_keys(WallSoilLayer) + tuple(_LAYER_ALIASES)


def _build_soil_layers(params, *, method, key="layers"):
    """The WallSoilLayers of ``params[key]``: unknown keys refused, the
    aliases above accepted. Shared with the sheet-pile calc package (its
    ``soil_layers``), so both read a layer the same way."""
    require_params(params, [key], method=method)
    check_nested_list(params[key], _LAYER_KEYS, method=method, label=key)
    layers = []
    for l in params[key]:
        l = apply_aliases(l, _LAYER_ALIASES)
        require_keys(l, ["thickness", "unit_weight"], method=method,
                     item_label=f"{key}[]")
        layers.append(WallSoilLayer(
            thickness=l["thickness"], unit_weight=l["unit_weight"],
            friction_angle=l.get("friction_angle", 30.0), cohesion=l.get("cohesion", 0.0),
            wall_friction_deg=l.get("wall_friction_deg", 0.0),
            description=l.get("description", ""),
        ))
    return layers


# The wall methods as (result, inputs in force): their own code path, also
# run by ``calc_package.sheet_pile_package`` (GM-141). The inputs in force
# are the soil layers built (``layers``) and each input given or defaulted,
# by the tool's names. ``method`` names the caller in error messages.

def run_cantilever_wall(params, *, method="cantilever_wall"):
    reject_unknown_params(
        params,
        ("excavation_depth", "layers", "gwt_depth_active", "gwt_depth_passive",
         "surcharge", "FOS_passive", "pressure_method", "embedment_increase"),
        method=method)
    layers = _build_soil_layers(params, method=method)
    require_params(params, ["excavation_depth"], method=method)
    used = {
        "excavation_depth": params["excavation_depth"], "layers": layers,
        "gwt_depth_active": params.get("gwt_depth_active"),
        "gwt_depth_passive": params.get("gwt_depth_passive"),
        "surcharge": params.get("surcharge", 0.0),
        "FOS_passive": params.get("FOS_passive", 1.5),
        "pressure_method": params.get("pressure_method", "rankine"),
        "embedment_increase": params.get("embedment_increase", 1.0),
    }
    result = analyze_cantilever(
        excavation_depth=used["excavation_depth"], soil_layers=layers,
        gwt_depth_active=used["gwt_depth_active"],
        gwt_depth_passive=used["gwt_depth_passive"],
        surcharge=used["surcharge"], FOS_passive=used["FOS_passive"],
        pressure_method=used["pressure_method"],
        embedment_increase=used["embedment_increase"],
    )
    return result, used


def run_anchored_wall(params, *, method="anchored_wall"):
    reject_unknown_params(
        params,
        ("excavation_depth", "anchor_depth", "layers", "gwt_depth_active",
         "gwt_depth_passive", "surcharge", "FOS_passive", "pressure_method"),
        method=method)
    layers = _build_soil_layers(params, method=method)
    require_params(params, ["excavation_depth", "anchor_depth"], method=method)
    used = {
        "excavation_depth": params["excavation_depth"],
        "anchor_depth": params["anchor_depth"], "layers": layers,
        "gwt_depth_active": params.get("gwt_depth_active"),
        "gwt_depth_passive": params.get("gwt_depth_passive"),
        "surcharge": params.get("surcharge", 0.0),
        "FOS_passive": params.get("FOS_passive", 1.5),
        "pressure_method": params.get("pressure_method", "rankine"),
    }
    result = analyze_anchored(
        excavation_depth=used["excavation_depth"],
        anchor_depth=used["anchor_depth"], soil_layers=layers,
        gwt_depth_active=used["gwt_depth_active"],
        gwt_depth_passive=used["gwt_depth_passive"],
        surcharge=used["surcharge"], FOS_passive=used["FOS_passive"],
        pressure_method=used["pressure_method"],
    )
    return result, used


def _run_cantilever_wall(params):
    return run_cantilever_wall(params)[0].to_dict()


def _run_anchored_wall(params):
    return run_anchored_wall(params)[0].to_dict()


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
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "log_spiral"], "description": "Earth pressure theory. 'log_spiral' (Caquot-Kerisel) uses Coulomb active + the log-spiral PASSIVE coefficient (avoids Coulomb's over-prediction of Kp at high delta/phi); set the layer wall_friction_deg (delta)."},
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
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS applied to passive resistance."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "log_spiral"], "description": "Earth pressure theory. 'log_spiral' (Caquot-Kerisel) uses Coulomb active + the log-spiral PASSIVE coefficient (avoids Coulomb's over-prediction of Kp at high delta/phi); set the layer wall_friction_deg (delta)."},
        },
        "returns": {"embedment_depth_m": "Required embedment.", "anchor_force_kN_per_m": "Anchor force."},
    },
}


# What a left-out input with no default stands for (BACKLOG GM-165):
# the water table on each side.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_NO_WATER = ("no water table",
             "no water pressure on either side of the wall; total unit "
             "weight throughout")
declare_when_absent(METHOD_INFO, {
    method: {"gwt_depth_active": _NO_WATER,
             "gwt_depth_passive": "gwt_depth_active"}
    for method in ("cantilever_wall", "anchored_wall")})
