"""Axial pile adapter — driven pile capacity (Nordlund/Tomlinson/Beta)."""

from axial_pile import PileSection, make_pipe_pile, make_concrete_pile, make_h_pile
from axial_pile import AxialSoilLayer, AxialSoilProfile, AxialPileAnalysis
from funhouse_agent.adapters import apply_aliases, require_keys, require_params

_PILE_TYPES = ("pipe_closed", "pipe_open", "concrete_square", "concrete_circular", "h_pile")


def _build_pile(params):
    pile_type = params.get("pile_type", "pipe_closed")
    if pile_type in ("pipe_closed", "pipe_open"):
        # 'width' is a common guess for the pipe diameter; 'thickness' for wall_thickness.
        p = apply_aliases(params, {"width": "diameter", "thickness": "wall_thickness"})
        require_params(p, ["diameter", "wall_thickness"],
                       method=f"pile_type '{pile_type}'",
                       valid=["diameter", "wall_thickness", "E"])
        return make_pipe_pile(diameter=p["diameter"], thickness=p["wall_thickness"],
                              closed_end=(pile_type == "pipe_closed"), E=p.get("E", 200e6))
    elif pile_type in ("concrete_square", "concrete_circular"):
        # 'diameter' is the natural name for a circular pile's width.
        p = apply_aliases(params, {"diameter": "width"})
        require_params(p, ["width"], method=f"pile_type '{pile_type}'",
                       valid=["width", "E"])
        shape = "square" if "square" in pile_type else "circular"
        return make_concrete_pile(width=p["width"], shape=shape, E=p.get("E", 25e6))
    elif pile_type == "h_pile":
        require_params(params, ["designation"], method="pile_type 'h_pile'",
                       valid=["designation", "E"])
        return make_h_pile(designation=params["designation"], E=params.get("E", 200e6))
    else:
        raise ValueError(f"Unknown pile_type '{pile_type}'. Allowed: {list(_PILE_TYPES)}.")


def _build_soil_profile(params, *, method):
    require_params(params, ["layers"], method=method)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "unit_weight"], method=method)
        layers.append(AxialSoilLayer(
            thickness=l["thickness"], soil_type=l.get("soil_type", "cohesionless"),
            unit_weight=l["unit_weight"], friction_angle=l.get("friction_angle"),
            cohesion=l.get("cohesion"), delta_phi_ratio=l.get("delta_phi_ratio", 0.75),
            description=l.get("description", ""),
        ))
    return AxialSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def _run_axial_pile_capacity(params):
    pile = _build_pile(params)
    soil = _build_soil_profile(params, method="axial_pile_capacity")
    require_params(params, ["pile_length"], method="axial_pile_capacity")
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params["pile_length"],
                                  method=params.get("method", "auto"),
                                  factor_of_safety=params.get("factor_of_safety", 2.5),
                                  include_uplift=params.get("include_uplift", False),
                                  cohesive_phi=params.get("cohesive_phi", 25.0),
                                  uplift_skin_fraction=params.get("uplift_skin_fraction", 0.75),
                                  pile_weight=params.get("pile_weight"))
    return analysis.compute().to_dict()


def _run_capacity_vs_depth(params):
    pile = _build_pile(params)
    soil = _build_soil_profile(params, method="capacity_vs_depth")
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=params.get("pile_length", soil.total_thickness),
                                  method=params.get("method", "auto"))
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=params.get("n_points", 20))
    return {"capacity_vs_depth": curve}


def _run_make_pile_section(params):
    pile = _build_pile(params)
    return {
        "name": pile.name,
        "pile_type": pile.pile_type,
        "area_m2": round(pile.area, 6),
        "perimeter_m": round(pile.perimeter, 4),
        "tip_area_m2": round(pile.tip_area, 6),
        "width_m": round(pile.width, 4),
    }


METHOD_REGISTRY = {
    "axial_pile_capacity": _run_axial_pile_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
    "make_pile_section": _run_make_pile_section,
}

METHOD_INFO = {
    "axial_pile_capacity": {
        "category": "Axial Pile",
        "brief": "Full driven pile axial capacity analysis (Nordlund/Tomlinson/Beta).",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": ["pipe_closed", "pipe_open", "concrete_square", "concrete_circular", "h_pile"], "description": "Pile type. pipe_closed/pipe_open require diameter + wall_thickness; concrete_square/concrete_circular require width (side dimension OR diameter); h_pile requires designation."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of soil-layer dicts: {thickness (m), soil_type, unit_weight (kN/m3), friction_angle (deg, for cohesionless), cohesion (kPa, for cohesive)}. soil_type must be 'cohesionless' or 'cohesive' (NOT 'sand'/'clay')."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "beta"], "description": "auto = Nordlund (sand) / Tomlinson (clay); beta = effective-stress method for all layers."},
            "include_uplift": {"type": "bool", "required": False, "default": False, "description": "Also report uplift (tension) capacity."},
            "cohesive_phi": {"type": "float", "required": False, "default": 25.0, "description": "Friction angle (deg) assumed for cohesive layers in Nordlund tip term."},
            "uplift_skin_fraction": {"type": "float", "required": False, "default": 0.75, "description": "Fraction of outside skin friction credited in uplift (with include_uplift)."},
            "pile_weight": {"type": "float", "required": False, "description": "Pile self-weight (kN) added to uplift capacity (with include_uplift)."},
        },
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Skin friction.", "Q_tip_kN": "Tip resistance.", "Q_allowable_kN": "Allowable capacity."},
    },
    "capacity_vs_depth": {
        "category": "Axial Pile",
        "brief": "Capacity vs depth curve for pile length optimization.",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": ["pipe_closed", "pipe_open", "concrete_square", "concrete_circular", "h_pile"], "description": "Pile type (see axial_pile_capacity for geometry params per type)."},
            "layers": {"type": "array", "required": True, "description": "Soil layers; each soil_type must be 'cohesionless' or 'cohesive'."},
            "depth_min": {"type": "float", "required": False, "default": 3.0, "description": "Min depth (m)."},
            "depth_max": {"type": "float", "required": False, "description": "Max depth (m)."},
        },
        "returns": {"capacity_vs_depth": "List of {depth, Q_ult, Q_skin, Q_tip} dicts."},
    },
    "make_pile_section": {
        "category": "Axial Pile",
        "brief": "Create a pile section and return its geometric properties.",
        "parameters": {
            "pile_type": {"type": "str", "required": True, "allowed_values": ["pipe_closed", "pipe_open", "concrete_square", "concrete_circular", "h_pile"], "description": "Pile type. pipe_closed/pipe_open require diameter + wall_thickness; concrete_square/concrete_circular require width (side dimension OR diameter); h_pile requires designation."},
            "diameter": {"type": "float", "required": False, "description": "Pile diameter (m) for pipe piles."},
            "wall_thickness": {"type": "float", "required": False, "description": "Pipe wall thickness (m)."},
            "width": {"type": "float", "required": False, "description": "Side dimension (m) for concrete piles."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g., 'HP14x117')."},
        },
        "returns": {"name": "Pile section name.", "area_m2": "Cross-sectional area.", "perimeter_m": "Perimeter.", "tip_area_m2": "Tip area.", "width_m": "Width."},
    },
}
