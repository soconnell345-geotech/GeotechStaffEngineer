"""Wave equation adapter — Smith 1-D (bearing graph, drivability)."""

from funhouse_agent.adapters import (declare_applies_when,
                                     reject_unknown_params, require_params)
from wave_equation import (
    Hammer, get_hammer, list_hammers, Cushion, make_cushion_from_properties,
    discretize_pile, SoilSetup, simulate_blow, generate_bearing_graph, drivability_study,
)

# Hammer + cushion params shared by every method's valid set.
_HAMMER_PARAMS = ("hammer_name", "hammer_custom_name", "ram_weight", "stroke",
                  "efficiency", "hammer_type")
_CUSHION_PARAMS = ("cushion_stiffness", "cushion_area", "cushion_thickness",
                   "cushion_E", "cushion_cor")


def _build_hammer(params):
    if "hammer_name" in params:
        return get_hammer(params["hammer_name"])
    require_params(params, ["ram_weight", "stroke"],
                   method="custom hammer (no hammer_name)",
                   valid=_HAMMER_PARAMS)
    return Hammer(name=params.get("hammer_custom_name", "Custom"), ram_weight=params["ram_weight"],
                   stroke=params["stroke"], efficiency=params.get("efficiency", 0.67),
                   hammer_type=params.get("hammer_type", "single_acting"))


def _build_cushion(params):
    if "cushion_stiffness" in params:
        return Cushion(stiffness=params["cushion_stiffness"], cor=params.get("cushion_cor", 0.80))
    elif "cushion_area" in params:
        require_params(params, ["cushion_area", "cushion_thickness", "cushion_E"],
                       method="cushion from properties",
                       valid=_CUSHION_PARAMS)
        return make_cushion_from_properties(area=params["cushion_area"], thickness=params["cushion_thickness"],
                                              elastic_modulus=params["cushion_E"], cor=params.get("cushion_cor", 0.80))
    return Cushion(stiffness=500000, cor=0.80)


def run_bearing_graph(params, *, method="bearing_graph"):
    """(result, inputs in force) of ``bearing_graph``: the tool's own code
    path, also run by ``calc_package.wave_equation_package`` (GM-141). The
    inputs in force are the hammer, cushion and pile models and each
    soil and range input given or defaulted, by the tool's names.
    ``method`` names the caller in error messages."""
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_length", "pile_area", "pile_E", "segment_length",
            "pile_unit_weight", "skin_fraction", "quake_side", "quake_toe",
            "damping_side", "damping_toe", "R_min", "R_max", "R_step",
            "helmet_weight", "damping_model"),
        method=method)
    require_params(params, ["pile_length", "pile_area"], method=method)
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)
    pile_E = params.get("pile_E", 200e6)
    segment_length = params.get("segment_length", 1.0)
    pile_unit_weight = params.get("pile_unit_weight", 78.5)
    pile = discretize_pile(length=params["pile_length"], area=params["pile_area"],
                            elastic_modulus=pile_E,
                            segment_length=segment_length,
                            unit_weight_material=pile_unit_weight)
    used = {
        "hammer": hammer, "cushion": cushion, "pile": pile,
        "pile_E": pile_E, "segment_length": segment_length,
        "pile_unit_weight": pile_unit_weight,
        "skin_fraction": params.get("skin_fraction", 0.5),
        "quake_side": params.get("quake_side", 0.0025),
        "quake_toe": params.get("quake_toe", 0.0025),
        "damping_side": params.get("damping_side", 0.16),
        "damping_toe": params.get("damping_toe", 0.50),
        "R_min": params.get("R_min", 200.0),
        "R_max": params.get("R_max", 2000.0),
        "R_step": params.get("R_step", 200.0),
        "helmet_weight": params.get("helmet_weight", 5.0),
        "damping_model": params.get("damping_model", "smith"),
    }
    bg = generate_bearing_graph(
        hammer, cushion, pile, skin_fraction=used["skin_fraction"],
        quake_side=used["quake_side"], quake_toe=used["quake_toe"],
        damping_side=used["damping_side"], damping_toe=used["damping_toe"],
        R_min=used["R_min"], R_max=used["R_max"], R_step=used["R_step"],
        helmet_weight=used["helmet_weight"],
        damping_model=used["damping_model"],
    )
    return bg, used


def _run_bearing_graph(params):
    return run_bearing_graph(params)[0].to_dict()


def _run_drivability(params):
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_area", "pile_E", "pile_unit_weight", "depths", "R_at_depth",
            "skin_fractions", "segment_length", "helmet_weight",
            "refusal_blow_count", "damping_model"),
        method="drivability")
    require_params(params, ["pile_area", "depths", "R_at_depth"],
                   method="drivability")
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)
    result = drivability_study(
        hammer, cushion, pile_area=params["pile_area"], pile_E=params.get("pile_E", 200e6),
        pile_unit_weight=params.get("pile_unit_weight", 78.5), depths=params["depths"],
        R_at_depth=params["R_at_depth"], skin_fractions=params.get("skin_fractions"),
        segment_length=params.get("segment_length", 1.0),
        helmet_weight=params.get("helmet_weight", 5.0),
        refusal_blow_count=params.get("refusal_blow_count", 3000.0),
        damping_model=params.get("damping_model", "smith"),
    )
    return result.to_dict()


def _run_list_hammers(params):
    hammers = {}
    for name in list_hammers():
        h = get_hammer(name)
        hammers[name] = {"ram_weight_kN": h.ram_weight, "stroke_m": h.stroke,
                          "rated_energy_kNm": round(h.energy, 1), "efficiency": h.efficiency}
    return {"hammers": hammers}


def _run_single_blow(params):
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_length", "pile_area", "pile_E", "segment_length",
            "pile_unit_weight", "R_total", "R_ultimate", "skin_fraction",
            "quake_side", "quake_toe", "damping_side", "damping_toe",
            "damping_model", "helmet_weight"),
        method="single_blow")
    require_params(params, ["pile_length", "pile_area"], method="single_blow")
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)
    pile = discretize_pile(
        length=params["pile_length"], area=params["pile_area"],
        elastic_modulus=params.get("pile_E", 200e6),
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", 78.5),
    )
    # Module keyword is R_ultimate; METHOD_INFO advertises R_total — accept both.
    R = params.get("R_total", params.get("R_ultimate"))
    if R is None:
        raise ValueError("single_blow: missing required parameter 'R_total' "
                         "(total ultimate soil resistance, kN).")
    soil = SoilSetup(
        R_ultimate=R,
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.50),
        damping_model=params.get("damping_model", "smith"),
    )
    result = simulate_blow(
        hammer, cushion, pile, soil,
        helmet_weight=params.get("helmet_weight", 5.0),
    )
    perm_set = result.permanent_set
    blow_count = round(1.0 / perm_set, 1) if perm_set > 0 else float("inf")
    return {
        "permanent_set_mm": round(perm_set * 1000, 3),
        "blow_count_per_m": blow_count,
        "max_pile_stress_MPa": round(result.max_compression_stress / 1000, 2),
        "max_tension_stress_MPa": round(result.max_tension_stress / 1000, 2),
        "max_pile_force_kN": round(result.max_pile_force, 1),
    }


METHOD_REGISTRY = {
    "single_blow": _run_single_blow,
    "bearing_graph": _run_bearing_graph,
    "drivability": _run_drivability,
    "list_available_hammers": _run_list_hammers,
}

# Shared hammer/cushion doc fragment (custom hammer in place of hammer_name;
# cushion by stiffness or by properties; default cushion if omitted).
_HAMMER_CUSHION_DOC = {
    "ram_weight": {"type": "float", "required": False, "description": "Custom hammer ram weight (kN) — used with stroke when hammer_name is omitted."},
    "stroke": {"type": "float", "required": False, "description": "Custom hammer stroke (m)."},
    "efficiency": {"type": "float", "required": False, "default": 0.67, "description": "Custom hammer efficiency."},
    "hammer_custom_name": {"type": "str", "required": False, "default": "Custom", "description": "Custom hammer label."},
    "hammer_type": {"type": "str", "required": False, "default": "single_acting", "allowed_values": ["single_acting", "diesel", "hydraulic"], "description": "Custom hammer type."},
    "segment_length": {"type": "float", "required": False, "default": 1.0, "description": "Pile segment length (m)."},
    "cushion_stiffness": {"type": "float", "required": False, "default": 500000.0, "description": "Cushion stiffness (kN/m). Or give cushion_area + cushion_thickness + cushion_E. Default 500,000 kN/m when neither is given."},
    "cushion_area": {"type": "float", "required": False, "description": "Cushion area (m2) — with cushion_thickness and cushion_E."},
    "cushion_thickness": {"type": "float", "required": False, "description": "Cushion thickness (m)."},
    "cushion_E": {"type": "float", "required": False, "description": "Cushion elastic modulus (kPa)."},
    "cushion_cor": {"type": "float", "required": False, "default": 0.80, "description": "Cushion coefficient of restitution."},
    "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Default is steel."},
    "pile_unit_weight": {"type": "float", "required": False, "default": 78.5, "description": "Pile material unit weight (kN/m3)."},
    "helmet_weight": {"type": "float", "required": False, "default": 5.0, "description": "Helmet weight (kN)."},
}

# Smith soil constants (single blow and bearing graph).
_SOIL_DOC = {
    "quake_side": {"type": "float", "required": False, "default": 0.0025, "description": "Side quake (m)."},
    "quake_toe": {"type": "float", "required": False, "default": 0.0025, "description": "Toe quake (m)."},
    "damping_side": {"type": "float", "required": False, "default": 0.16, "description": "Side Smith damping (s/m)."},
    "damping_toe": {"type": "float", "required": False, "default": 0.50, "description": "Toe Smith damping (s/m)."},
}

METHOD_INFO = {
    "single_blow": {
        "category": "Wave Equation",
        "brief": "Simulate a single hammer blow — set, stress, blow count.",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name (use list_available_hammers)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "R_total": {"type": "float", "required": True, "description": "Total soil resistance (kN). Alias: R_ultimate."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Skin friction fraction."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": "Smith damping form: 'smith' (R_static-proportional) or 'smith_viscous' (R_ultimate-proportional)."},
            **_SOIL_DOC,
            **_HAMMER_CUSHION_DOC,
        },
        "parameter_aliases": {"R_ultimate": "R_total"},
        "returns": {
            "permanent_set_mm": "Pile penetration per blow (mm).",
            "blow_count_per_m": "Blows per meter.",
            "max_pile_stress_MPa": "Max compressive stress (MPa).",
            "max_tension_stress_MPa": "Max tensile stress (MPa).",
        },
    },
    "bearing_graph": {
        "category": "Wave Equation",
        "brief": "Generate bearing capacity vs blow count graph.",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name (use list_available_hammers to see options)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-sectional area (m2)."},
            "R_min": {"type": "float", "required": False, "default": 200.0, "description": "Min resistance (kN)."},
            "R_max": {"type": "float", "required": False, "default": 2000.0, "description": "Max resistance (kN)."},
            "R_step": {"type": "float", "required": False, "default": 200.0, "description": "Resistance increment (kN)."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Skin friction fraction."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": "Smith damping form: 'smith' (R_static-proportional) or 'smith_viscous' (R_ultimate-proportional)."},
            **_SOIL_DOC,
            **_HAMMER_CUSHION_DOC,
        },
        "returns": {"blow_counts_per_m": "Blow counts.", "max_pile_force_kN": "Max driving stress."},
    },
    "drivability": {
        "category": "Wave Equation",
        "brief": "Drivability study: can the hammer drive the pile to target depth?",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name."},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "depths": {"type": "array", "required": True, "description": "Depth stations (m)."},
            "R_at_depth": {"type": "array", "required": True, "description": "SRD at each depth (kN)."},
            "skin_fractions": {"type": "array", "required": False, "description": "Skin friction fraction per depth station."},
            "refusal_blow_count": {"type": "float", "required": False, "default": 3000.0, "description": "Blows/m treated as refusal."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": "Smith damping form: 'smith' (R_static-proportional) or 'smith_viscous' (R_ultimate-proportional)."},
            **_HAMMER_CUSHION_DOC,
        },
        "returns": {"can_drive": "Whether pile can be driven.", "points": "Per-depth drivability data."},
    },
    "list_available_hammers": {
        "category": "Wave Equation",
        "brief": "List all available pile driving hammers.",
        "parameters": {},
        "returns": {"hammers": "Dict of hammer properties."},
    },
}

# When each default applies (GM-119): a custom hammer's efficiency and type
# only when no hammer_name picks a hammer from the list.
declare_applies_when(METHOD_INFO, {
    m: {name: {"hammer_name": "unset"}
        for name in ("efficiency", "hammer_type")}
    for m in ("bearing_graph", "drivability", "single_blow")})

# The default cushion, when no cushion is described (GM-165).
for _m in ("single_blow", "bearing_graph", "drivability"):
    METHOD_INFO[_m]["parameters"]["cushion_stiffness"]["applies_when"] = {
        "cushion_area": "unset"}
