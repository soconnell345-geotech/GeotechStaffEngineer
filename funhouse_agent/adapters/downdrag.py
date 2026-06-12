"""Downdrag adapter — Fellenius neutral plane, UFC 3-220-20."""

from downdrag import DowndragSoilLayer, DowndragSoilProfile, DowndragAnalysis
from funhouse_agent.adapters import reject_unknown_params, require_keys, require_params

_SOIL_TYPES = ("cohesionless", "cohesive")

# Every top-level parameter _run_downdrag_analysis consumes.
_VALID_PARAMS = (
    "layers", "gwt_depth", "pile_length", "pile_diameter", "pile_perimeter",
    "pile_area", "pile_E", "pile_unit_weight", "Q_dead", "structural_capacity",
    "fill_thickness", "fill_unit_weight", "gw_drawdown", "Nt", "n_sublayers",
)


def _run_downdrag_analysis(params):
    reject_unknown_params(params, _VALID_PARAMS, method="downdrag_analysis")
    require_params(params, ["layers", "pile_length", "pile_diameter"],
                   method="downdrag_analysis", valid=_VALID_PARAMS)
    layers = []
    for l in params["layers"]:
        require_keys(l, ["thickness", "soil_type", "unit_weight"], method="downdrag_analysis")
        if l["soil_type"] not in _SOIL_TYPES:
            raise ValueError(
                f"downdrag_analysis: layer soil_type must be one of {list(_SOIL_TYPES)} "
                f"(got '{l['soil_type']}'). Mark settling layers with settling=True."
            )
        layers.append(DowndragSoilLayer(
            thickness=l["thickness"], soil_type=l["soil_type"], unit_weight=l["unit_weight"],
            phi=l.get("phi", 0.0), cu=l.get("cu", 0.0), beta=l.get("beta"), alpha=l.get("alpha"),
            Cc=l.get("Cc", 0.0), Cr=l.get("Cr", 0.0), e0=l.get("e0", 0.0),
            C_ec=l.get("C_ec"), C_er=l.get("C_er"), sigma_p=l.get("sigma_p"),
            E_s=l.get("E_s"), nu_s=l.get("nu_s", 0.3),
            settling=l.get("settling", False), description=l.get("description", ""),
        ))
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
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, soil_type, unit_weight, phi, cu, beta, Cc, e0, settling} dicts. soil_type must be 'cohesionless' or 'cohesive' (NOT 'sand'/'clay'/'settling_fill'). Mark settling layers with settling=True."},
            "Q_dead": {"type": "float", "required": False, "default": 0.0, "description": "Dead load at pile top (kN)."},
            "fill_thickness": {"type": "float", "required": False, "description": "Fill thickness causing downdrag (m)."},
            "fill_unit_weight": {"type": "float", "required": False, "default": 19.0, "description": "Fill unit weight (kN/m3)."},
            "gw_drawdown": {"type": "float", "required": False, "description": "Groundwater drawdown (m)."},
            "gwt_depth": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater depth (m)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Default is steel."},
            "pile_perimeter": {"type": "float", "required": False, "description": "Pile perimeter (m). Computed from diameter if omitted."},
            "pile_area": {"type": "float", "required": False, "description": "Pile cross-section area (m2). Computed from diameter if omitted."},
            "structural_capacity": {"type": "float", "required": False, "description": "Pile structural capacity (kN) for the max-load check."},
            "Nt": {"type": "float", "required": False, "description": "Toe bearing capacity coefficient override."},
        },
        "returns": {"neutral_plane_depth_m": "Neutral plane depth.", "dragload_kN": "Downdrag force on pile."},
    },
}
