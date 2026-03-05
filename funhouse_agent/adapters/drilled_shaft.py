"""Drilled shaft adapter — GEC-10 alpha/beta/rock socket capacity."""

from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis


def _build_shaft(params):
    return DrillShaft(diameter=params["diameter"], length=params["shaft_length"],
                       socket_diameter=params.get("socket_diameter"),
                       socket_length=params.get("socket_length", 0.0),
                       bell_diameter=params.get("bell_diameter"),
                       casing_depth=params.get("casing_depth", 0.0),
                       concrete_fc=params.get("concrete_fc", 28000.0))


def _build_soil_profile(params):
    layers = [ShaftSoilLayer(
        thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
        cu=l.get("cu", 0.0), phi=l.get("phi", 0.0), N60=l.get("N60", 0.0),
        qu=l.get("qu", 0.0), RQD=l.get("RQD", 100.0), description=l.get("description", ""),
    ) for l in params["layers"]]
    return ShaftSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))


def _run_drilled_shaft_capacity(params):
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil, factor_of_safety=params.get("factor_of_safety", 2.5))
    return analysis.compute().to_dict()


def _run_capacity_vs_depth(params):
    shaft = _build_shaft(params)
    soil = _build_soil_profile(params)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil, factor_of_safety=params.get("factor_of_safety", 2.5))
    curve = analysis.capacity_vs_depth(depth_min=params.get("depth_min", 3.0),
                                        depth_max=params.get("depth_max"), n_points=params.get("n_points", 20))
    return {"capacity_vs_depth": curve}


METHOD_REGISTRY = {
    "drilled_shaft_capacity": _run_drilled_shaft_capacity,
    "capacity_vs_depth": _run_capacity_vs_depth,
}

METHOD_INFO = {
    "drilled_shaft_capacity": {
        "category": "Drilled Shaft",
        "brief": "Full drilled shaft capacity (GEC-10 alpha/beta/rock socket).",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "shaft_length": {"type": "float", "required": True, "description": "Shaft length (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, cu, phi, N60, qu, RQD} dicts. soil_type: cohesive/cohesionless/rock."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
        },
        "returns": {"Q_ultimate_kN": "Ultimate capacity.", "Q_skin_kN": "Side resistance.", "Q_tip_kN": "Tip resistance."},
    },
    "capacity_vs_depth": {
        "category": "Drilled Shaft",
        "brief": "Capacity vs depth curve for length optimization.",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "shaft_length": {"type": "float", "required": True, "description": "Max shaft length (m)."},
            "layers": {"type": "array", "required": True, "description": "Soil layers."},
        },
        "returns": {"capacity_vs_depth": "List of {depth, Q_ult, Q_skin, Q_tip} dicts."},
    },
}
