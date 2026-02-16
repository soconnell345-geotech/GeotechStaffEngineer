"""
Lateral Pile Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. lateral_pile_agent          - Run a lateral pile analysis or extract p-y curves
  2. lateral_pile_list_methods   - Browse available methods
  3. lateral_pile_describe_method - Get detailed parameter docs

Implements COM624P p-y curve methods with 7 soil models and finite-difference solver.
"""

import json
import math
import numpy as np
from functions.api import function

from lateral_pile.pile import Pile, PileSection, ReinforcedConcreteSection, rebar_diameter
from lateral_pile.soil import SoilLayer
from lateral_pile.analysis import LateralPileAnalysis
from lateral_pile.py_curves import (
    SoftClayMatlock, StiffClayBelowWT, StiffClayAboveWT,
    SoftClayJeanjean, SandReese, SandAPI, WeakRock,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _clean_result(result: dict) -> dict:
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [_clean_result(item) if isinstance(item, dict) else _clean_value(item) for item in v]
        elif isinstance(v, dict):
            cleaned[k] = _clean_result(v)
        else:
            cleaned[k] = _clean_value(v)
    return cleaned


# Map of model name strings to classes
_PY_MODELS = {
    "SoftClayMatlock": SoftClayMatlock,
    "StiffClayBelowWT": StiffClayBelowWT,
    "StiffClayAboveWT": StiffClayAboveWT,
    "SoftClayJeanjean": SoftClayJeanjean,
    "SandReese": SandReese,
    "SandAPI": SandAPI,
    "WeakRock": WeakRock,
}

# Required parameters per model (everything else has defaults)
_PY_MODEL_PARAMS = {
    "SoftClayMatlock": {"c": "c", "gamma": "gamma", "eps50": "eps50"},
    "StiffClayBelowWT": {"c": "c", "gamma": "gamma", "eps50": "eps50", "ks": "ks"},
    "StiffClayAboveWT": {"c": "c", "gamma": "gamma", "eps50": "eps50"},
    "SoftClayJeanjean": {"su": "su", "gamma": "gamma", "Gmax": "Gmax"},
    "SandReese": {"phi": "phi", "gamma": "gamma", "k": "k"},
    "SandAPI": {"phi": "phi", "gamma": "gamma", "k": "k"},
    "WeakRock": {"qu": "qu", "Er": "Er"},
}


def _build_py_model(layer_dict: dict):
    """Build a p-y model instance from a flat layer dict.

    The layer dict must have a 'model' key naming the p-y model class,
    plus the model's required parameters as sibling keys.

    Example: {"model": "SandAPI", "phi": 35, "gamma": 10, "k": 16000}
    """
    model_name = layer_dict.get("model")
    if model_name not in _PY_MODELS:
        available = ", ".join(sorted(_PY_MODELS.keys()))
        raise ValueError(
            f"Unknown p-y model '{model_name}'. Available: {available}"
        )

    cls = _PY_MODELS[model_name]
    kwargs = {}

    # Common optional params with defaults
    loading = layer_dict.get("loading", "static")

    if model_name == "SoftClayMatlock":
        kwargs = dict(c=layer_dict["c"], gamma=layer_dict["gamma"],
                      eps50=layer_dict["eps50"],
                      J=layer_dict.get("J", 0.5), loading=loading)
    elif model_name == "StiffClayBelowWT":
        kwargs = dict(c=layer_dict["c"], gamma=layer_dict["gamma"],
                      eps50=layer_dict["eps50"], ks=layer_dict["ks"],
                      loading=loading)
    elif model_name == "StiffClayAboveWT":
        kwargs = dict(c=layer_dict["c"], gamma=layer_dict["gamma"],
                      eps50=layer_dict["eps50"],
                      J=layer_dict.get("J", 0.5), loading=loading)
    elif model_name == "SoftClayJeanjean":
        kwargs = dict(su=layer_dict["su"], gamma=layer_dict["gamma"],
                      Gmax=layer_dict["Gmax"],
                      J=layer_dict.get("J", 0.5), loading=loading)
    elif model_name == "SandReese":
        kwargs = dict(phi=layer_dict["phi"], gamma=layer_dict["gamma"],
                      k=layer_dict["k"], loading=loading)
    elif model_name == "SandAPI":
        kwargs = dict(phi=layer_dict["phi"], gamma=layer_dict["gamma"],
                      k=layer_dict["k"], loading=loading)
    elif model_name == "WeakRock":
        kwargs = dict(qu=layer_dict["qu"], Er=layer_dict["Er"],
                      gamma_r=layer_dict.get("gamma_r", 22.0),
                      RQD=layer_dict.get("RQD", 100.0),
                      loading=loading)

    return cls(**kwargs)


def _build_pile(params: dict) -> Pile:
    """Build a Pile from flat JSON params."""
    pile_type = params.get("pile_type", "pipe")

    if pile_type == "h_pile":
        return Pile.from_h_pile(
            designation=params["designation"],
            length=params["pile_length"],
            axis=params.get("axis", "strong"),
            E=params.get("pile_E", 200e6),
        )
    elif pile_type == "filled_pipe":
        return Pile.from_filled_pipe(
            length=params["pile_length"],
            diameter=params["pile_diameter"],
            thickness=params["pile_thickness"],
            E_steel=params.get("pile_E", 200e6),
            fc=params.get("fc", 28000.0),
            E_concrete=params.get("E_concrete"),
        )
    elif pile_type == "rc":
        bar_diam = params.get("bar_diameter", 0.0254)
        if isinstance(bar_diam, str):
            bar_diam = rebar_diameter(bar_diam)
        rc = ReinforcedConcreteSection(
            diameter=params["pile_diameter"],
            fc=params["fc"],
            fy=params.get("fy", 420000.0),
            n_bars=params.get("n_bars", 12),
            bar_diameter=bar_diam,
            cover=params.get("cover", 0.075),
            E_steel=params.get("E_steel_rebar", 200e6),
        )
        return Pile.from_rc_section(
            length=params["pile_length"],
            rc_section=rc,
        )
    else:
        # "pipe" or "solid" — standard Pile constructor
        return Pile(
            length=params["pile_length"],
            diameter=params["pile_diameter"],
            E=params.get("pile_E", 200e6),
            thickness=params.get("pile_thickness"),
            moment_of_inertia=params.get("moment_of_inertia"),
        )


def _build_layers(params: dict) -> list:
    """Build list of SoilLayer from layers array in params."""
    layers = []
    for lay in params["layers"]:
        py_model = _build_py_model(lay)
        layers.append(SoilLayer(
            top=lay["top"],
            bottom=lay["bottom"],
            py_model=py_model,
            description=lay.get("description"),
        ))
    return layers


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_lateral_pile_analysis(params: dict) -> dict:
    """Full lateral pile analysis with p-y curves and FD solver."""
    pile = _build_pile(params)
    layers = _build_layers(params)

    analysis = LateralPileAnalysis(pile, layers)
    result = analysis.solve(
        Vt=params.get("Vt", 0.0),
        Mt=params.get("Mt", 0.0),
        Q=params.get("Q", 0.0),
        head_condition=params.get("head_condition", "free"),
        rotational_stiffness=params.get("rotational_stiffness", 0.0),
        n_elements=params.get("n_elements", 100),
        tolerance=params.get("tolerance", 1e-5),
        max_iterations=params.get("max_iterations", 100),
    )
    return result.to_dict()


def _run_get_py_curve(params: dict) -> dict:
    """Extract a single p-y curve at a given depth for a given model."""
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "lateral_pile_analysis": _run_lateral_pile_analysis,
    "get_py_curve": _run_get_py_curve,
}

METHOD_INFO = {
    "lateral_pile_analysis": {
        "category": "Lateral Analysis",
        "brief": "Lateral pile analysis using p-y curves and finite-difference solver.",
        "description": (
            "Computes the lateral response of a single pile using nonlinear p-y curves "
            "and a finite-difference beam-column solver. Supports 7 p-y models "
            "(Matlock soft clay, Reese stiff clay below/above WT, Jeanjean soft clay, "
            "Reese sand, API sand, weak rock), multiple pile types (pipe, solid, H-pile, "
            "concrete-filled pipe, reinforced concrete with cracked-EI), and free/fixed/"
            "partial head conditions. Returns deflection, moment, shear, and soil "
            "reaction profiles along the pile."
        ),
        "reference": (
            "COM624P (FHWA-SA-91-048); Matlock (1970); Reese, Cox & Koop (1974); "
            "Reese et al. (1975); Welch & Reese (1972); Jeanjean (2009, OTC-20158); "
            "API RP2A; Reese (1997) weak rock; FHWA GEC-13"
        ),
        "parameters": {
            "pile_type": {
                "type": "str", "required": False, "default": "pipe",
                "description": (
                    "'pipe' (hollow steel), 'solid' (solid circular), "
                    "'h_pile' (AISC HP shape), 'filled_pipe' (concrete-filled steel), "
                    "'rc' (reinforced concrete with cracked-EI iteration)."
                ),
            },
            "pile_length": {
                "type": "float", "required": True,
                "description": "Embedded pile length (m).",
            },
            "pile_diameter": {
                "type": "float", "required": True,
                "description": "Outer diameter (m). Not needed for h_pile (uses flange width).",
            },
            "pile_thickness": {
                "type": "float", "required": False,
                "description": "Wall thickness (m). For pipe and filled_pipe types.",
            },
            "pile_E": {
                "type": "float", "required": False, "default": 200e6,
                "description": "Young's modulus (kPa). Steel=200e6, concrete=25e6.",
            },
            "moment_of_inertia": {
                "type": "float", "required": False,
                "description": "Override moment of inertia (m4). Computed from geometry if omitted.",
            },
            "designation": {
                "type": "str", "required": False,
                "description": "AISC HP shape for h_pile type (e.g. 'HP14x117'). Available: HP10x42, HP10x57, HP12x53, HP12x63, HP12x74, HP12x84, HP14x73, HP14x89, HP14x102, HP14x117.",
            },
            "axis": {
                "type": "str", "required": False, "default": "strong",
                "description": "Bending axis for h_pile: 'strong' (Ixx) or 'weak' (Iyy).",
            },
            "fc": {
                "type": "float", "required": False, "default": 28000.0,
                "description": "Concrete f'c (kPa). For filled_pipe and rc types.",
            },
            "E_concrete": {
                "type": "float", "required": False,
                "description": "Concrete modulus (kPa). For filled_pipe. If omitted, computed from f'c per ACI 318.",
            },
            "fy": {
                "type": "float", "required": False, "default": 420000.0,
                "description": "Rebar yield strength (kPa). For rc type. Default 420 MPa (Grade 60).",
            },
            "n_bars": {
                "type": "int", "required": False, "default": 12,
                "description": "Number of longitudinal bars. For rc type.",
            },
            "bar_diameter": {
                "type": "float or str", "required": False, "default": 0.0254,
                "description": "Rebar diameter (m) or US designation string (e.g. '#8'). For rc type.",
            },
            "cover": {
                "type": "float", "required": False, "default": 0.075,
                "description": "Clear cover to bar center (m). For rc type.",
            },
            "Vt": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Lateral load at pile head (kN).",
            },
            "Mt": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Moment at pile head (kN-m).",
            },
            "Q": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Axial load (kN, positive = compression). Affects P-delta.",
            },
            "head_condition": {
                "type": "str", "required": False, "default": "free",
                "description": "'free' (specified V and M), 'fixed' (specified V, zero rotation), 'partial' (specified V and rotational stiffness).",
            },
            "rotational_stiffness": {
                "type": "float", "required": False, "default": 0.0,
                "description": "Rotational stiffness at pile head (kN-m/rad). Only used with head_condition='partial'.",
            },
            "n_elements": {
                "type": "int", "required": False, "default": 100,
                "description": "Number of pile segments for FD mesh.",
            },
            "tolerance": {
                "type": "float", "required": False, "default": 1e-5,
                "description": "Convergence tolerance on deflection.",
            },
            "max_iterations": {
                "type": "int", "required": False, "default": 100,
                "description": "Maximum solver iterations.",
            },
            "layers": {
                "type": "array", "required": True,
                "description": (
                    "Array of soil layers. Each layer: top (m), bottom (m), "
                    "model (p-y model name), plus model-specific parameters. "
                    "Models: 'SoftClayMatlock' (c, gamma, eps50; opt: J, loading), "
                    "'StiffClayBelowWT' (c, gamma, eps50, ks; opt: loading), "
                    "'StiffClayAboveWT' (c, gamma, eps50; opt: J, loading), "
                    "'SoftClayJeanjean' (su, gamma, Gmax; opt: J, loading), "
                    "'SandReese' (phi, gamma, k; opt: loading), "
                    "'SandAPI' (phi, gamma, k; opt: loading), "
                    "'WeakRock' (qu, Er; opt: gamma_r, RQD, loading). "
                    "loading: 'static' (default) or 'cyclic'. Optional: description."
                ),
            },
        },
        "returns": {
            "z": "Array of depths along pile (m).",
            "deflection_m": "Array of lateral deflection (m).",
            "slope_rad": "Array of rotation (radians).",
            "moment_kNm": "Array of bending moment (kN-m).",
            "shear_kN": "Array of shear force (kN).",
            "soil_reaction_kN_per_m": "Array of soil reaction p (kN/m).",
            "y_top_m": "Pile head deflection (m).",
            "rotation_top_rad": "Pile head rotation (radians).",
            "max_moment_kNm": "Maximum bending moment magnitude (kN-m).",
            "max_moment_depth_m": "Depth of maximum moment (m).",
            "max_deflection_m": "Maximum deflection magnitude (m).",
            "iterations": "Number of solver iterations.",
            "converged": "Whether the solver converged (bool).",
            "EI_profile_kNm2": "Array of EI at each node (kN-m2). Only for RC piles.",
            "ei_iterations": "Number of EI iterations. Only for RC piles.",
        },
    },
    "get_py_curve": {
        "category": "P-Y Curves",
        "brief": "Extract a single p-y curve at a given depth for any soil model.",
        "description": (
            "Generates a p-y curve (soil resistance vs lateral deflection) for a "
            "specified p-y model at a given depth and pile diameter. Useful for "
            "inspecting soil response, comparing models, or plotting."
        ),
        "reference": "Same as lateral_pile_analysis.",
        "parameters": {
            "model": {
                "type": "str", "required": True,
                "description": (
                    "P-y model name: 'SoftClayMatlock', 'StiffClayBelowWT', "
                    "'StiffClayAboveWT', 'SoftClayJeanjean', 'SandReese', "
                    "'SandAPI', or 'WeakRock'."
                ),
            },
            "z": {
                "type": "float", "required": False, "default": 5.0,
                "description": "Depth at which to compute the p-y curve (m).",
            },
            "pile_diameter": {
                "type": "float", "required": False, "default": 0.6,
                "description": "Pile diameter (m).",
            },
            "n_points": {
                "type": "int", "required": False, "default": 50,
                "description": "Number of points on the curve.",
            },
            "c": {"type": "float", "required": False, "description": "Undrained shear strength (kPa). For clay models."},
            "su": {"type": "float", "required": False, "description": "Undrained shear strength (kPa). For SoftClayJeanjean."},
            "gamma": {"type": "float", "required": False, "description": "Effective unit weight (kN/m3)."},
            "eps50": {"type": "float", "required": False, "description": "Strain at 50% max deviator stress. For clay models."},
            "ks": {"type": "float", "required": False, "description": "Initial modulus of subgrade reaction (kN/m3). For StiffClayBelowWT."},
            "Gmax": {"type": "float", "required": False, "description": "Small-strain shear modulus (kPa). For SoftClayJeanjean."},
            "J": {"type": "float", "required": False, "default": 0.5, "description": "Empirical constant. For Matlock/Jeanjean/StiffClayAboveWT."},
            "phi": {"type": "float", "required": False, "description": "Friction angle (degrees). For sand models."},
            "k": {"type": "float", "required": False, "description": "Initial modulus of subgrade reaction (kN/m3). For sand models."},
            "qu": {"type": "float", "required": False, "description": "Unconfined compressive strength (kPa). For WeakRock."},
            "Er": {"type": "float", "required": False, "description": "Rock mass modulus (kPa). For WeakRock."},
            "gamma_r": {"type": "float", "required": False, "default": 22.0, "description": "Rock unit weight (kN/m3). For WeakRock."},
            "RQD": {"type": "float", "required": False, "default": 100.0, "description": "Rock Quality Designation (%). For WeakRock."},
            "loading": {"type": "str", "required": False, "default": "static", "description": "'static' or 'cyclic'."},
        },
        "returns": {
            "model": "P-y model name used.",
            "z_m": "Depth (m).",
            "pile_diameter_m": "Pile diameter (m).",
            "y_m": "Array of lateral deflection values (m).",
            "p_kN_per_m": "Array of soil resistance values (kN/m).",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def lateral_pile_agent(method: str, parameters_json: str) -> str:
    """
    Lateral pile analysis calculator.

    Computes lateral deflection, moment, shear, and soil reaction profiles
    for a single pile using p-y curve methods (COM624P) and a finite-difference
    beam-column solver. Supports 7 p-y models and multiple pile types.

    Parameters:
        method: The calculation method name. Use lateral_pile_list_methods() to see options.
        parameters_json: JSON string of parameters. Use lateral_pile_describe_method() for details.

    Returns:
        JSON string with calculation results or an error message.
    """
    try:
        parameters = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})

    try:
        result = METHOD_REGISTRY[method](parameters)
        return json.dumps(_clean_result(result), default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def lateral_pile_list_methods(category: str = "") -> str:
    """
    Lists available lateral pile calculation methods.

    Parameters:
        category: Optional filter by category.

    Returns:
        JSON string with method names and brief descriptions.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    return json.dumps(result)


@function
def lateral_pile_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a lateral pile method.

    Parameters:
        method: The method name (e.g. 'lateral_pile_analysis', 'get_py_curve').

    Returns:
        JSON string with parameters, types, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
