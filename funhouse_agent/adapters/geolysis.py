"""Geolysis adapter — soil classification (USCS/AASHTO) + SPT corrections."""

from funhouse_agent.adapters import clean_result


def _run_classify_uscs(p):
    from geolysis_agent import classify_uscs
    return clean_result(classify_uscs(**p).to_dict())


def _run_classify_aashto(p):
    from geolysis_agent import classify_aashto
    return clean_result(classify_aashto(**p).to_dict())


def _run_correct_spt(p):
    from geolysis_agent import correct_spt
    return clean_result(correct_spt(**p).to_dict())


def _run_allowable_bc_spt(p):
    from geolysis_agent import allowable_bc_spt
    return clean_result(allowable_bc_spt(**p).to_dict())


def _run_ultimate_bc(p):
    from geolysis_agent import ultimate_bc
    return clean_result(ultimate_bc(**p).to_dict())


METHOD_REGISTRY = {
    "classify_uscs": _run_classify_uscs,
    "classify_aashto": _run_classify_aashto,
    "correct_spt": _run_correct_spt,
    "allowable_bc_spt": _run_allowable_bc_spt,
    "ultimate_bc": _run_ultimate_bc,
}

METHOD_INFO = {
    "classify_uscs": {"category": "Classification", "brief": "USCS soil classification from grain size + Atterberg limits.",
        "parameters": {"liquid_limit": {"type": "float", "required": True, "description": "Liquid limit (%)."}, "plasticity_index": {"type": "float", "required": True, "description": "Plasticity index (%)."}, "fines": {"type": "float", "required": True, "description": "Fines content (%)."}},
        "returns": {"symbol": "USCS symbol.", "description": "Soil description."}},
    "classify_aashto": {"category": "Classification", "brief": "AASHTO soil classification.",
        "parameters": {"liquid_limit": {"type": "float", "required": True, "description": "Liquid limit."}, "plasticity_index": {"type": "float", "required": True, "description": "PI."}, "fines": {"type": "float", "required": True, "description": "Fines content (%)."}},
        "returns": {"group": "AASHTO group.", "description": "Soil description."}},
    "correct_spt": {"category": "SPT", "brief": "SPT N-value corrections (energy, overburden, rod length, etc.).",
        "parameters": {"n_field": {"type": "int", "required": True, "description": "Field SPT N-value."}, "energy_ratio": {"type": "float", "required": False, "default": 0.6, "description": "Hammer energy ratio."}, "sigma_v_eff": {"type": "float", "required": True, "description": "Effective overburden (kPa)."}, "rod_length": {"type": "float", "required": False, "description": "Rod length (m)."}},
        "returns": {"n60": "Energy-corrected N.", "n1_60": "Overburden-corrected N."}},
    "allowable_bc_spt": {"category": "Bearing Capacity", "brief": "Allowable bearing capacity from SPT (Meyerhof method).",
        "parameters": {"n60": {"type": "float", "required": True, "description": "Corrected SPT N60."}, "B": {"type": "float", "required": True, "description": "Footing width (m)."}, "D": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth (m)."}},
        "returns": {"bearing_capacity_kpa": "Allowable bearing capacity (kPa)."}},
    "ultimate_bc": {"category": "Bearing Capacity", "brief": "Ultimate bearing capacity from soil parameters.",
        "parameters": {"c": {"type": "float", "required": True, "description": "Cohesion (kPa)."}, "phi": {"type": "float", "required": True, "description": "Friction angle (deg)."}, "gamma": {"type": "float", "required": True, "description": "Unit weight (kN/m3)."}, "B": {"type": "float", "required": True, "description": "Footing width (m)."}, "D": {"type": "float", "required": False, "default": 0.0, "description": "Depth (m)."}},
        "returns": {"q_ult_kpa": "Ultimate bearing capacity (kPa)."}},
}
