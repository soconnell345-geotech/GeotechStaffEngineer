"""Wave equation adapter — Smith 1-D (bearing graph, drivability)."""

from wave_equation import (
    Hammer, get_hammer, list_hammers, Cushion, make_cushion_from_properties,
    discretize_pile, SoilSetup, simulate_blow, generate_bearing_graph, drivability_study,
)


def _build_hammer(params):
    if "hammer_name" in params:
        return get_hammer(params["hammer_name"])
    return Hammer(name=params.get("hammer_custom_name", "Custom"), ram_weight=params["ram_weight"],
                   stroke=params["stroke"], efficiency=params.get("efficiency", 0.67),
                   hammer_type=params.get("hammer_type", "single_acting"))


def _build_cushion(params):
    if "cushion_stiffness" in params:
        return Cushion(stiffness=params["cushion_stiffness"], cor=params.get("cushion_cor", 0.80))
    elif "cushion_area" in params:
        return make_cushion_from_properties(area=params["cushion_area"], thickness=params["cushion_thickness"],
                                              elastic_modulus=params["cushion_E"], cor=params.get("cushion_cor", 0.80))
    return Cushion(stiffness=500000, cor=0.80)


def _run_bearing_graph(params):
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)
    pile = discretize_pile(length=params["pile_length"], area=params["pile_area"],
                            elastic_modulus=params.get("pile_E", 200e6),
                            segment_length=params.get("segment_length", 1.0),
                            unit_weight_material=params.get("pile_unit_weight", 78.5))
    bg = generate_bearing_graph(
        hammer, cushion, pile, skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025), quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16), damping_toe=params.get("damping_toe", 0.50),
        R_min=params.get("R_min", 200.0), R_max=params.get("R_max", 2000.0), R_step=params.get("R_step", 200.0),
        helmet_weight=params.get("helmet_weight", 5.0),
    )
    return bg.to_dict()


def _run_drivability(params):
    hammer = _build_hammer(params)
    cushion = _build_cushion(params)
    result = drivability_study(
        hammer, cushion, pile_area=params["pile_area"], pile_E=params.get("pile_E", 200e6),
        pile_unit_weight=params.get("pile_unit_weight", 78.5), depths=params["depths"],
        R_at_depth=params["R_at_depth"], skin_fractions=params.get("skin_fractions"),
        segment_length=params.get("segment_length", 1.0),
        helmet_weight=params.get("helmet_weight", 5.0),
        refusal_blow_count=params.get("refusal_blow_count", 3000.0),
    )
    return result.to_dict()


def _run_list_hammers(params):
    hammers = {}
    for name in list_hammers():
        h = get_hammer(name)
        hammers[name] = {"ram_weight_kN": h.ram_weight, "stroke_m": h.stroke,
                          "rated_energy_kNm": round(h.energy, 1), "efficiency": h.efficiency}
    return {"hammers": hammers}


METHOD_REGISTRY = {
    "bearing_graph": _run_bearing_graph,
    "drivability": _run_drivability,
    "list_available_hammers": _run_list_hammers,
}

METHOD_INFO = {
    "bearing_graph": {
        "category": "Wave Equation",
        "brief": "Generate bearing capacity vs blow count graph.",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name (use list_available_hammers to see options)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-sectional area (m2)."},
            "R_min": {"type": "float", "required": False, "default": 200.0, "description": "Min resistance (kN)."},
            "R_max": {"type": "float", "required": False, "default": 2000.0, "description": "Max resistance (kN)."},
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
