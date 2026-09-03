"""Concrete properties adapter — flat dict -> concrete_props_agent -> dict."""

from funhouse_agent.adapters import (
    clean_result, reject_unknown_params, require_params,
)

_ERR = ("concreteproperties is not installed (requires Python >= 3.12). "
        "Install with: pip install concreteproperties")


def _run_rc_section(params: dict) -> dict:
    from concrete_props_agent import (
        analyze_rc_rectangle, has_concreteproperties)
    if not has_concreteproperties():
        return {"error": _ERR}
    reject_unknown_params(
        params,
        ("b", "h", "fc", "fy", "n_bot", "dia_bot", "cover", "n_top",
         "dia_top", "ec", "es", "include_interaction",
         "n_interaction_points"),
        method="rc_rectangular_section")
    require_params(params, ["b", "h", "fc", "fy", "n_bot", "dia_bot"],
                   method="rc_rectangular_section")
    result = analyze_rc_rectangle(
        b_mm=params["b"], h_mm=params["h"],
        fc_MPa=params["fc"], fy_MPa=params["fy"],
        n_bot=int(params["n_bot"]), dia_bot_mm=params["dia_bot"],
        cover_mm=params.get("cover", 40.0),
        n_top=int(params.get("n_top", 0)),
        dia_top_mm=params.get("dia_top", 0.0),
        ec_MPa=params.get("ec"),
        es_MPa=params.get("es", 200e3),
        include_interaction=params.get("include_interaction", False),
        n_interaction_points=params.get("n_interaction_points", 24),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "rc_rectangular_section": _run_rc_section,
}

METHOD_INFO = {
    "rc_rectangular_section": {
        "category": "RC Section Analysis",
        "brief": "Rectangular RC section: gross/cracked Ixx, cracking moment, nominal Mn (sag/hog), optional N-M interaction. mm/MPa in, kN*m out.",
        "parameters": {
            "b": {"type": "float", "required": True, "description": "Width (mm)."},
            "h": {"type": "float", "required": True, "description": "Overall depth (mm)."},
            "fc": {"type": "float", "required": True, "description": "Concrete f'c (MPa)."},
            "fy": {"type": "float", "required": True, "description": "Bar yield strength (MPa)."},
            "n_bot": {"type": "int", "required": True, "description": "Bottom bar count."},
            "dia_bot": {"type": "float", "required": True, "description": "Bottom bar diameter (mm)."},
            "cover": {"type": "float", "required": False, "default": 40.0, "description": "Clear cover to bar surface (mm), both faces."},
            "n_top": {"type": "int", "required": False, "default": 0, "description": "Top bar count (0 = none)."},
            "dia_top": {"type": "float", "required": False, "description": "Top bar diameter (mm)."},
            "ec": {"type": "float", "required": False, "description": "Concrete modulus (MPa). Default ACI 4700*sqrt(f'c)."},
            "es": {"type": "float", "required": False, "default": 200000.0, "description": "Steel modulus (MPa)."},
            "include_interaction": {"type": "bool", "required": False, "default": False, "description": "Also compute the N-M interaction diagram."},
            "n_interaction_points": {"type": "int", "required": False, "default": 24, "description": "Interaction diagram resolution."},
        },
        "returns": {
            "mn_pos_kNm": "Nominal sagging capacity (no phi).",
            "mn_neg_kNm": "Nominal hogging capacity (if top steel).",
            "m_cr_kNm": "Cracking moment.",
            "ixx_cracked_mm4": "Cracked transformed Ixx.",
        },
    },
}
