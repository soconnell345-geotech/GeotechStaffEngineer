"""Axial pile adapter — driven pile capacity (Nordlund/Tomlinson/Beta)."""

from axial_pile import PileSection, make_pipe_pile, make_concrete_pile, make_h_pile
from axial_pile import AxialSoilLayer, AxialSoilProfile, AxialPileAnalysis


def _build_pile(params):
    pile_type = params.get("pile_type", "pipe_closed")
    if pile_type in ("pipe_closed", "pipe_open"):
        return make_pipe_pile(diameter=params["diameter"], thickness=params["wall_thickness"],
                              closed_end=(pile_type == "pipe_closed"), E=params.get("E", 200e6))
    elif pile_type in ("concrete_square", "concrete_circular"):
        shape = "square" if "square" in pile_type else "circular"
        return make_concrete_pile(width=params["width"], shape=shape, E=params.get("E", 25e6))
    elif pile_type == "h_pile":
        return make_h_pile(designation=params["designation"], E=params.get("E", 200e6))
    else:
        raise ValueError(f"Unknown pile_type '{pile_type}'.")


def _build_soil_profile(params):
    layers = [AxialSoilLayer(
        thickness=l["thickness"], soil_type=l.get("soil_type", "cohesionless"),
        unit_weight=l["unit_weight"], friction_angle=l.get("friction_angle"),
        cohesion=l.get("cohesion"), delta_phi_ratio=l.get("delta_phi_ratio", 0.75),
        description=l.get("description", ""),
    ) for l in params["layers"]]
    return AxialSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def _run_axial_pile_capacity(params):
    pile = _build_pile(params)
    soil = _build_soil_profile(params)
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params["pile_length"],
                                  method=params.get("method", "auto"),
                                  factor_of_safety=params.get("factor_of_safety", 2.5),
                                  include_uplift=params.get("include_uplift", False))
    return analysis.compute().to_dict()


def _run_capacity_vs_depth(params):
    pile = _build_pile(params)
    soil = _build_soil_profile(params)
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params.get("pile_length", soil.total_thickness),
                                  method=params.get("method", "auto"))
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=params.get("n_points", 20))
    return {"capacity_vs_depth": curve}


METHOD_REGISTRY = {
    "axial_pile_capacity": _run_axial_pile_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
}

METHOD_INFO = {
    "axial_pile_capacity": {
        "category": "Axial Pile",
        "brief": "Full driven pile axial capacity analysis (Nordlund/Tomlinson/Beta).",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "description": "pipe_closed/pipe_open/concrete_square/concrete_circular/h_pile."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, friction_angle, cohesion} dicts."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
        },
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Skin friction.", "Q_tip_kN": "Tip resistance.", "Q_allowable_kN": "Allowable capacity."},
    },
    "capacity_vs_depth": {
        "category": "Axial Pile",
        "brief": "Capacity vs depth curve for pile length optimization.",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "description": "Pile type."},
            "layers": {"type": "array", "required": True, "description": "Soil layers."},
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Min depth (m)."},
            "depth_max": {"type": "float", "required": False, "description": "Max depth (m)."},
        },
        "returns": {"capacity_vs_depth": "List of {depth, Q_ult, Q_skin, Q_tip} dicts."},
    },
}
