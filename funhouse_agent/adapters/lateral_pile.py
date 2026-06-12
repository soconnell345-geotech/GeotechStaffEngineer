"""Lateral pile adapter — COM624P, 8 p-y models, FD solver."""

from funhouse_agent.adapters import require_keys, require_params, reject_unknown_params
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

# Required parameter keys per p-y model (beyond optional loading/J/etc.).
_PY_REQUIRED = {
    "SoftClayMatlock": ("c", "gamma", "eps50"),
    "StiffClayBelowWT": ("c", "gamma", "eps50", "ks"),
    "StiffClayAboveWT": ("c", "gamma", "eps50"),
    "SoftClayJeanjean": ("su", "gamma", "Gmax"),
    "SandReese": ("phi", "gamma", "k"),
    "SandAPI": ("phi", "gamma", "k"),
    "WeakRock": ("qu", "Er"),
}


def _build_py_model(d):
    model_name = d.get("model")
    if model_name not in _PY_MODELS:
        raise ValueError(f"Unknown p-y model '{model_name}'. Available: {sorted(_PY_MODELS)}")
    missing = [k for k in _PY_REQUIRED[model_name] if d.get(k) is None]
    if missing:
        raise ValueError(
            f"p-y model '{model_name}' requires {list(_PY_REQUIRED[model_name])}; "
            f"missing {missing}. (k = initial subgrade modulus in kN/m3 for sand models.)"
        )
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
        require_params(params, ["designation", "pile_length"], method="pile_type 'h_pile'",
                       valid=["designation", "pile_length", "axis", "pile_E"])
        return Pile.from_h_pile(designation=params["designation"], length=params["pile_length"],
                                 axis=params.get("axis", "strong"), E=params.get("pile_E", 200e6))
    elif pile_type == "filled_pipe":
        require_params(params, ["pile_length", "pile_diameter", "pile_thickness"],
                       method="pile_type 'filled_pipe'",
                       valid=["pile_length", "pile_diameter", "pile_thickness", "pile_E", "fc"])
        return Pile.from_filled_pipe(length=params["pile_length"], diameter=params["pile_diameter"],
                                      thickness=params["pile_thickness"], E_steel=params.get("pile_E", 200e6),
                                      fc=params.get("fc", 28000.0))
    else:
        require_params(params, ["pile_length", "pile_diameter"], method=f"pile_type '{pile_type}'",
                       valid=["pile_length", "pile_diameter", "pile_E", "pile_thickness",
                              "moment_of_inertia"])
        return Pile(length=params["pile_length"], diameter=params["pile_diameter"],
                    E=params.get("pile_E", 200e6), thickness=params.get("pile_thickness"),
                    moment_of_inertia=params.get("moment_of_inertia"))


# Every top-level parameter _run_lateral_pile_analysis consumes. Anything else
# is rejected loudly: a silently dropped parameter (e.g. an invented stiffness
# name like 'E_GPa') would otherwise run the analysis with the steel default
# and return a confidently wrong answer.
_ANALYSIS_VALID_PARAMS = (
    "pile_type", "pile_diameter", "pile_length", "pile_E", "pile_thickness",
    "moment_of_inertia", "designation", "axis", "fc", "layers",
    "Vt", "Mt", "Q", "head_condition", "n_elements", "tolerance",
    "max_iterations", "stickup", "free_length",
)


def _run_lateral_pile_analysis(params):
    reject_unknown_params(params, _ANALYSIS_VALID_PARAMS,
                          method="lateral_pile_analysis")
    pile = _build_pile(params)
    require_params(params, ["layers"], method="lateral_pile_analysis")
    for l in params["layers"]:
        require_keys(l, ["top", "bottom", "model"], method="lateral_pile_analysis")
    layers = [SoilLayer(top=l["top"], bottom=l["bottom"], py_model=_build_py_model(l),
                         description=l.get("description")) for l in params["layers"]]
    analysis = LateralPileAnalysis(pile, layers)
    # 'free_length' is a common name for the above-ground stickup.
    stickup = params.get("stickup", params.get("free_length", 0.0))
    result = analysis.solve(
        Vt=params.get("Vt", 0.0), Mt=params.get("Mt", 0.0), Q=params.get("Q", 0.0),
        head_condition=params.get("head_condition", "free"),
        n_elements=params.get("n_elements", 100), tolerance=params.get("tolerance", 1e-5),
        max_iterations=params.get("max_iterations", 100),
        stickup=stickup,
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
            "pile_type": {"type": "str", "required": False, "default": "pipe", "allowed_values": ["pipe", "h_pile", "filled_pipe"], "description": "Pile type. h_pile requires designation; filled_pipe also requires pile_thickness. For a solid pile/drilled shaft, use 'pipe' with no pile_thickness and set pile_E."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile Young's modulus (kPa). DEFAULT IS STEEL (200e6) — for concrete drilled shafts set this explicitly, e.g. ~25e6 gross or a reduced value (e.g. 0.3x) for cracked sections."},
            "pile_thickness": {"type": "float", "required": False, "description": "Wall thickness (m) for hollow pipe. Omit for a solid section."},
            "moment_of_inertia": {"type": "float", "required": False, "description": "Moment of inertia (m4). Overrides the value computed from diameter/thickness."},
            "designation": {"type": "str", "required": False, "description": "H-pile designation (e.g. 'HP360x132') — required for pile_type 'h_pile'."},
            "axis": {"type": "str", "required": False, "default": "strong", "allowed_values": ["strong", "weak"], "description": "Bending axis — pile_type 'h_pile' only."},
            "fc": {"type": "float", "required": False, "default": 28000.0, "description": "Concrete f'c (kPa) — pile_type 'filled_pipe' only."},
            "layers": {"type": "array", "required": True, "description": "Array of {top, bottom, model, ...model params}. Required model params: SoftClayMatlock/StiffClayAboveWT {c, gamma, eps50}; StiffClayBelowWT {c, gamma, eps50, ks}; SoftClayJeanjean {su, gamma, Gmax}; SandReese/SandAPI {phi, gamma, k} where k = initial subgrade modulus (kN/m3); WeakRock {qu, Er}."},
            "Vt": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load at pile top (kN)."},
            "Mt": {"type": "float", "required": False, "default": 0.0, "description": "Moment at pile top (kN-m)."},
            "head_condition": {"type": "str", "required": False, "default": "free", "allowed_values": ["free", "fixed"], "description": "Pile head condition."},
            "Q": {"type": "float", "required": False, "default": 0.0, "description": "Axial load at pile top (kN), for P-delta."},
            "stickup": {"type": "float", "required": False, "default": 0.0, "description": "Above-ground free length (m): head loads/BCs act at the top of the stickup; no soil resistance over it. pile_length stays the EMBEDDED length. Alias: free_length."},
        },
        "returns": {"max_deflection_m": "Maximum deflection (m).", "y_top_m": "Deflection at loaded head (m).", "max_moment_kNm": "Maximum bending moment.", "max_moment_depth_m": "Depth of max moment (m)."},
    },
    "get_py_curve": {
        "category": "Lateral Pile",
        "brief": "Extract a single p-y curve at a given depth for any soil model.",
        "parameters": {
            "model": {"type": "str", "required": True, "allowed_values": ["SoftClayMatlock", "StiffClayBelowWT", "StiffClayAboveWT", "SoftClayJeanjean", "SandReese", "SandAPI", "WeakRock"], "description": "P-y model. Required params: clay models need {c, gamma, eps50} (+ks for StiffClayBelowWT); sand models need {phi, gamma, k}; WeakRock needs {qu, Er}."},
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
