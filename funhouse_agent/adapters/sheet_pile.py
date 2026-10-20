"""Sheet pile adapter — cantilever and anchored wall analysis."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_applies_when, field_keys,
    reject_unknown_params, require_keys, require_params,
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


def _log_spiral_source(params, layers, *, method):
    """The printed source of the log-spiral passive Kp (GM-183); UFC
    3-220-20 Fig. 4-13 is drawn for one wall friction, delta/phi' = 0.66,
    so each friction layer must give it."""
    source = params.get("log_spiral_source", "caquot_kerisel_1948")
    if source == "dm7_2_fig_4_13" and params.get("pressure_method")             == "log_spiral":
        off = [i for i, l in enumerate(layers) if l.friction_angle > 0
               and abs(l.wall_friction_deg / l.friction_angle - 0.66) > 0.01]
        if off:
            raise ValueError(
                f"{method}: log_spiral_source 'dm7_2_fig_4_13' (UFC "
                "3-220-20 Fig. 4-13) is drawn for delta/phi' = 0.66 only; "
                f"layers {off} give another wall_friction_deg. Set each "
                "layer's wall_friction_deg to 0.66 x its friction_angle, or "
                "use 'dm7_2_eq_4_12' (UFC 3-220-20 Eq. 4-12, any wall "
                "friction) or 'caquot_kerisel_1948'.")
    return source


# The wall methods as (result, inputs in force): their own code path, also
# run by ``calc_package.sheet_pile_package`` (GM-141). The inputs in force
# are the soil layers built (``layers``) and each input given or defaulted,
# by the tool's names. ``method`` names the caller in error messages.

def run_cantilever_wall(params, *, method="cantilever_wall"):
    reject_unknown_params(
        params,
        ("excavation_depth", "layers", "gwt_depth_active", "gwt_depth_passive",
         "surcharge", "FOS_passive", "pressure_method", "embedment_increase",
         "log_spiral_source"),
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
        "log_spiral_source": _log_spiral_source(params, layers,
                                                method=method),
    }
    result = analyze_cantilever(
        excavation_depth=used["excavation_depth"], soil_layers=layers,
        gwt_depth_active=used["gwt_depth_active"],
        gwt_depth_passive=used["gwt_depth_passive"],
        surcharge=used["surcharge"], FOS_passive=used["FOS_passive"],
        pressure_method=used["pressure_method"],
        embedment_increase=used["embedment_increase"],
        log_spiral_source=used["log_spiral_source"],
    )
    return result, used


def run_anchored_wall(params, *, method="anchored_wall"):
    reject_unknown_params(
        params,
        ("excavation_depth", "anchor_depth", "layers", "gwt_depth_active",
         "gwt_depth_passive", "surcharge", "FOS_passive", "pressure_method",
         "log_spiral_source"),
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
        "log_spiral_source": _log_spiral_source(params, layers,
                                                method=method),
    }
    result = analyze_anchored(
        excavation_depth=used["excavation_depth"],
        anchor_depth=used["anchor_depth"], soil_layers=layers,
        gwt_depth_active=used["gwt_depth_active"],
        gwt_depth_passive=used["gwt_depth_passive"],
        surcharge=used["surcharge"], FOS_passive=used["FOS_passive"],
        pressure_method=used["pressure_method"],
        log_spiral_source=used["log_spiral_source"],
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

# The printed sources of the log-spiral passive Kp (GM-183), shared with
# retaining_walls.earth_pressure_coefficient.
_LOG_SPIRAL_SOURCE = {
    "type": "str", "required": False, "default": "caquot_kerisel_1948",
    "allowed_values": ["caquot_kerisel_1948", "dm7_2_fig_4_13",
                       "dm7_2_eq_4_12"],
    "description": (
        "With pressure_method 'log_spiral', the printed source of the "
        "passive Kp. 'caquot_kerisel_1948' (default, as before): the Caquot "
        "and Kerisel (1948) chart, Kp at delta = phi times the printed "
        "reduction factor R(delta/phi), Caltrans Trenching and Shoring "
        "Manual Fig. 4-20 (p. 4-30), the method its Sec. 4-6 uses for "
        "passive pressure. 'dm7_2_fig_4_13': UFC 3-220-20 (DM 7.2) Fig. "
        "4-13 (p. 193), Kerisel and Absi (1990), drawn for delta/phi' = "
        "0.66 only, so every layer's wall_friction_deg must be 0.66 x its "
        "friction_angle. 'dm7_2_eq_4_12': UFC 3-220-20 Eq. 4-12 (p. 191) "
        "for level ground and any wall friction. They differ: at phi 30, "
        "delta/phi 0.66 they give 5.4, 5.3 and 5.1. UFC 3-220-20 footnote "
        "21 (p. 191): the 1982 DM 7.2 charts 'used delta/phi' = 1 along "
        "with reduction ratios', which 'were averages that introduced "
        "substantial inaccuracy for some cases'; the 1948 chart takes that "
        "form (Kp at delta = phi times R)."),
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
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "log_spiral"], "description": "Earth pressure theory. 'log_spiral' uses Coulomb active + the log-spiral PASSIVE coefficient from log_spiral_source (avoids Coulomb's over-prediction of Kp at high delta/phi); set the layer wall_friction_deg (delta)."},
            "log_spiral_source": _LOG_SPIRAL_SOURCE,
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
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "log_spiral"], "description": "Earth pressure theory. 'log_spiral' uses Coulomb active + the log-spiral PASSIVE coefficient from log_spiral_source (avoids Coulomb's over-prediction of Kp at high delta/phi); set the layer wall_friction_deg (delta)."},
            "log_spiral_source": _LOG_SPIRAL_SOURCE,
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

# When each default applies (GM-119): the log-spiral source to the
# log-spiral passive Kp (GM-183).
declare_applies_when(METHOD_INFO, {
    method: {"log_spiral_source": {"pressure_method": ["log_spiral"]}}
    for method in ("cantilever_wall", "anchored_wall")})
