"""Lateral pile adapter — COM624P, 8 p-y models, FD solver."""

from lateral_pile import Pile, ReinforcedConcreteSection, rebar_diameter, SoilLayer, LateralPileAnalysis
from lateral_pile.py_curves import (
    SoftClayMatlock, StiffClayBelowWT, StiffClayAboveWT, SoftClayJeanjean,
    SandReese, SandAPI, WeakRock,
)

_PY_MODELS = {
    "SoftClayMatlock": SoftClayMatlock, "StiffClayBelowWT": StiffClayBelowWT,
    "StiffClayAboveWT": StiffClayAboveWT, "SoftClayJeanjean": SoftClayJeanjean,
    "SandReese": SandReese, "SandAPI": SandAPI, "WeakRock": WeakRock,
}


def _build_py_model(d):
    model_name = d.get("model")
    if model_name not in _PY_MODELS:
        raise ValueError(f"Unknown p-y model '{model_name}'. Available: {sorted(_PY_MODELS)}")
    cls = _PY_MODELS[model_name]
    loading = d.get("loading", "static")
    if model_name == "SoftClayMatlock":
        return cls(c=d["c"], gamma=d["gamma"], eps50=d["eps50"], J=d.get("J", 0.5), loading=loading)
    elif model_name == "StiffClayBelowWT":
        return cls(c=d["c"], gamma=d["gamma"], eps50=d["eps50"], ks=d["ks"], loading=loading)
    elif model_name == "StiffClayAboveWT":
        return cls(c=d["c"], gamma=d["gamma"], eps50=d["eps50"], J=d.get("J", 0.5), loading=loading)
    elif model_name == "SoftClayJeanjean":
        return cls(su=d["su"], gamma=d["gamma"], Gmax=d["Gmax"], J=d.get("J", 0.5), loading=loading)
    elif model_name in ("SandReese", "SandAPI"):
        return cls(phi=d["phi"], gamma=d["gamma"], k=d["k"], loading=loading)
    elif model_name == "WeakRock":
        return cls(qu=d["qu"], Er=d["Er"], gamma_r=d.get("gamma_r", 22.0), RQD=d.get("RQD", 100.0), loading=loading)


def _build_pile(params):
    pile_type = params.get("pile_type", "pipe")
    if pile_type == "h_pile":
        return Pile.from_h_pile(designation=params["designation"], length=params["pile_length"],
                                 axis=params.get("axis", "strong"), E=params.get("pile_E", 200e6))
    elif pile_type == "filled_pipe":
        return Pile.from_filled_pipe(length=params["pile_length"], diameter=params["pile_diameter"],
                                      thickness=params["pile_thickness"], E_steel=params.get("pile_E", 200e6),
                                      fc=params.get("fc", 28000.0))
    else:
        return Pile(length=params["pile_length"], diameter=params["pile_diameter"],
                    E=params.get("pile_E", 200e6), thickness=params.get("pile_thickness"),
                    moment_of_inertia=params.get("moment_of_inertia"))


def _run_lateral_pile_analysis(params):
    pile = _build_pile(params)
    layers = [SoilLayer(top=l["top"], bottom=l["bottom"], py_model=_build_py_model(l),
                         description=l.get("description")) for l in params["layers"]]
    analysis = LateralPileAnalysis(pile, layers)
    result = analysis.solve(
        Vt=params.get("Vt", 0.0), Mt=params.get("Mt", 0.0), Q=params.get("Q", 0.0),
        head_condition=params.get("head_condition", "free"),
        n_elements=params.get("n_elements", 100), tolerance=params.get("tolerance", 1e-5),
        max_iterations=params.get("max_iterations", 100),
    )
    return result.to_dict()


def _run_get_py_curve(params):
    py_model = _build_py_model(params)
    z = params.get("z", 5.0)
    b = params.get("pile_diameter", 0.6)
    n_points = params.get("n_points", 50)
    y_arr, p_arr = py_model.get_py_curve(z=z, b=b, n_points=n_points)
    return {
        "model": params["model"],
        "z_m": z,
        "pile_diameter_m": b,
        "y_m": y_arr.tolist(),
        "p_kN_per_m": p_arr.tolist(),
    }


METHOD_REGISTRY = {
    "lateral_pile_analysis": _run_lateral_pile_analysis,
    "get_py_curve": _run_get_py_curve,
}

METHOD_INFO = {
    "lateral_pile_analysis": {
        "category": "Lateral Pile",
        "brief": "Full lateral pile analysis with p-y curves and FD solver.",
        "parameters": {
            "pile_type": {"type": "str", "required": False, "default": "pipe", "description": "pipe/h_pile/filled_pipe."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {top, bottom, model, ...model params}. model: SoftClayMatlock/StiffClayBelowWT/SandAPI/SandReese/WeakRock etc."},
            "Vt": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load at pile top (kN)."},
            "Mt": {"type": "float", "required": False, "default": 0.0, "description": "Moment at pile top (kN-m)."},
            "head_condition": {"type": "str", "required": False, "default": "free", "description": "free or fixed."},
        },
        "returns": {"max_deflection_mm": "Maximum deflection.", "max_moment_kNm": "Maximum bending moment.", "max_shear_kN": "Maximum shear."},
    },
    "get_py_curve": {
        "category": "Lateral Pile",
        "brief": "Extract a single p-y curve at a given depth for any soil model.",
        "parameters": {
            "model": {"type": "str", "required": True, "description": "P-y model: SoftClayMatlock/StiffClayBelowWT/StiffClayAboveWT/SoftClayJeanjean/SandReese/SandAPI/WeakRock."},
            "z": {"type": "float", "required": False, "default": 5.0, "description": "Depth (m)."},
            "pile_diameter": {"type": "float", "required": False, "default": 0.6, "description": "Pile diameter (m)."},
            "n_points": {"type": "int", "required": False, "default": 50, "description": "Points on curve."},
            "c": {"type": "float", "required": False, "description": "Undrained shear strength (kPa). For clay models."},
            "gamma": {"type": "float", "required": False, "description": "Effective unit weight (kN/m3)."},
            "eps50": {"type": "float", "required": False, "description": "Strain at 50% deviator stress. For clay."},
            "phi": {"type": "float", "required": False, "description": "Friction angle (deg). For sand."},
            "k": {"type": "float", "required": False, "description": "Initial subgrade modulus (kN/m3). For sand."},
            "qu": {"type": "float", "required": False, "description": "UCS (kPa). For WeakRock."},
            "Er": {"type": "float", "required": False, "description": "Rock mass modulus (kPa). For WeakRock."},
        },
        "returns": {"y_m": "Deflection array (m).", "p_kN_per_m": "Soil resistance array (kN/m)."},
    },
}
