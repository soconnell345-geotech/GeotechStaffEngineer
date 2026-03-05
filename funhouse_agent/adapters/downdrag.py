"""Downdrag adapter — Fellenius neutral plane, UFC 3-220-20."""

from downdrag import DowndragSoilLayer, DowndragSoilProfile, DowndragAnalysis


def _run_downdrag_analysis(params):
    layers = [DowndragSoilLayer(
        thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
        phi=l.get("phi", 0.0), cu=l.get("cu", 0.0), beta=l.get("beta"), alpha=l.get("alpha"),
        Cc=l.get("Cc", 0.0), Cr=l.get("Cr", 0.0), e0=l.get("e0", 0.0),
        C_ec=l.get("C_ec"), C_er=l.get("C_er"), sigma_p=l.get("sigma_p"),
        E_s=l.get("E_s"), nu_s=l.get("nu_s", 0.3),
        settling=l.get("settling", False), description=l.get("description", ""),
    ) for l in params["layers"]]
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))
    analysis = DowndragAnalysis(
        soil=soil, pile_length=params["pile_length"], pile_diameter=params["pile_diameter"],
        pile_perimeter=params.get("pile_perimeter"), pile_area=params.get("pile_area"),
        pile_E=params.get("pile_E", 200e6), pile_unit_weight=params.get("pile_unit_weight", 24.0),
        Q_dead=params.get("Q_dead", 0.0), structural_capacity=params.get("structural_capacity"),
        fill_thickness=params.get("fill_thickness", 0.0), fill_unit_weight=params.get("fill_unit_weight", 19.0),
        gw_drawdown=params.get("gw_drawdown", 0.0), Nt=params.get("Nt"), n_sublayers=params.get("n_sublayers", 10),
    )
    return analysis.compute().to_dict()


METHOD_REGISTRY = {"downdrag_analysis": _run_downdrag_analysis}

METHOD_INFO = {
    "downdrag_analysis": {
        "category": "Downdrag",
        "brief": "Full downdrag (negative skin friction) analysis via Fellenius neutral plane.",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, phi, cu, beta, Cc, e0, settling} dicts. Mark settling layers with settling=True."},
            "Q_dead": {"type": "float", "required": False, "default": 0.0, "description": "Dead load at pile top (kN)."},
            "fill_thickness": {"type": "float", "required": False, "description": "Fill thickness causing downdrag (m)."},
            "gw_drawdown": {"type": "float", "required": False, "description": "Groundwater drawdown (m)."},
        },
        "returns": {"neutral_plane_depth_m": "Neutral plane depth.", "dragload_kN": "Downdrag force on pile."},
    },
}
