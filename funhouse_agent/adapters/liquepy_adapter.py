"""liquepy adapter — CPT-based liquefaction triggering and field correlations."""

from funhouse_agent.adapters import clean_result


def _check_liquepy():
    """Raise ValueError if liquepy is not installed."""
    from liquepy_agent import has_liquepy
    if not has_liquepy():
        raise ValueError(
            "liquepy is not installed. Install with: pip install liquepy"
        )


def _run_cpt_liquefaction(params: dict) -> dict:
    _check_liquepy()
    from liquepy_agent import analyze_cpt_liquefaction

    result = analyze_cpt_liquefaction(
        depth=params["depth"],
        q_c=params["q_c"],
        f_s=params["f_s"],
        u_2=params.get("u_2"),
        gwl=params.get("gwl", 1.0),
        pga=params.get("pga", 0.25),
        m_w=params.get("m_w", 7.5),
        a_ratio=params.get("a_ratio", 0.8),
        i_c_limit=params.get("i_c_limit", 2.6),
        cfc=params.get("cfc", 0.0),
        unit_wt_method=params.get("unit_wt_method", "robertson2009"),
        gamma_predrill=params.get("gamma_predrill", 17.0),
        s_g=params.get("s_g", 2.65),
        p_a=params.get("p_a", 101.0),
    )
    return clean_result(result.to_dict())


def _run_field_correlations(params: dict) -> dict:
    _check_liquepy()
    from liquepy_agent import analyze_field_correlations

    result = analyze_field_correlations(
        depth=params["depth"],
        q_c=params["q_c"],
        f_s=params["f_s"],
        u_2=params.get("u_2"),
        gwl=params.get("gwl", 1.0),
        a_ratio=params.get("a_ratio", 0.8),
        vs_method=params.get("vs_method", "mcgann2015"),
        i_c_limit=params.get("i_c_limit", 2.6),
        p_a=params.get("p_a", 101.0),
        s_g=params.get("s_g", 2.65),
        gamma_predrill=params.get("gamma_predrill", 17.0),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "cpt_liquefaction": _run_cpt_liquefaction,
    "field_correlations": _run_field_correlations,
}

METHOD_INFO = {
    "cpt_liquefaction": {
        "category": "Liquefaction",
        "brief": "CPT-based liquefaction triggering analysis (Boulanger & Idriss 2014) with LPI, LSN, LDI.",
        "parameters": {
            "depth": {"type": "array", "brief": "Depth from surface (m), monotonically increasing."},
            "q_c": {"type": "array", "brief": "Cone tip resistance (kPa)."},
            "f_s": {"type": "array", "brief": "Sleeve friction (kPa)."},
            "u_2": {"type": "array", "brief": "Pore pressure behind cone tip (kPa). Default: zeros.", "default": None},
            "gwl": {"type": "float", "brief": "Groundwater level depth (m below surface).", "default": 1.0},
            "pga": {"type": "float", "brief": "Peak ground acceleration (g).", "default": 0.25},
            "m_w": {"type": "float", "brief": "Moment magnitude.", "default": 7.5},
            "a_ratio": {"type": "float", "brief": "Cone area ratio.", "default": 0.8},
            "i_c_limit": {"type": "float", "brief": "Ic limit for liquefiable material.", "default": 2.6},
            "cfc": {"type": "float", "brief": "Fines content correction factor.", "default": 0.0},
            "unit_wt_method": {"type": "str", "brief": "Unit weight method: 'robertson2009' or 'void_ratio'.", "default": "robertson2009"},
            "gamma_predrill": {"type": "float", "brief": "Unit weight above pre-drill depth (kN/m3).", "default": 17.0},
            "s_g": {"type": "float", "brief": "Specific gravity of solids.", "default": 2.65},
            "p_a": {"type": "float", "brief": "Atmospheric pressure (kPa).", "default": 101.0},
        },
        "returns": {
            "LPI": "Liquefaction Potential Index.",
            "LSN": "Liquefaction Severity Number.",
            "LDI_m": "Lateral Displacement Index (m).",
            "n_liquefiable": "Number of liquefiable points.",
            "min_FOS": "Minimum factor of safety against liquefaction.",
        },
    },
    "field_correlations": {
        "category": "CPT Correlations",
        "brief": "Compute Vs, Dr, su/sigma_v', and permeability from CPT data.",
        "parameters": {
            "depth": {"type": "array", "brief": "Depth from surface (m)."},
            "q_c": {"type": "array", "brief": "Cone tip resistance (kPa)."},
            "f_s": {"type": "array", "brief": "Sleeve friction (kPa)."},
            "u_2": {"type": "array", "brief": "Pore pressure behind cone tip (kPa). Default: zeros.", "default": None},
            "gwl": {"type": "float", "brief": "Groundwater level depth (m below surface).", "default": 1.0},
            "a_ratio": {"type": "float", "brief": "Cone area ratio.", "default": 0.8},
            "vs_method": {"type": "str", "brief": "Vs correlation method: 'mcgann2015', 'robertson2009', 'andrus2007'.", "default": "mcgann2015"},
            "i_c_limit": {"type": "float", "brief": "Ic limit for sand vs clay classification.", "default": 2.6},
            "p_a": {"type": "float", "brief": "Atmospheric pressure (kPa).", "default": 101.0},
            "s_g": {"type": "float", "brief": "Specific gravity of solids.", "default": 2.65},
            "gamma_predrill": {"type": "float", "brief": "Unit weight above pre-drill depth (kN/m3).", "default": 17.0},
        },
        "returns": {
            "n_points": "Number of data points.",
            "Vs_m_per_s": "Shear wave velocity profile (m/s).",
            "Dr_pct": "Relative density profile (%).",
            "su_ratio": "Undrained strength ratio profile.",
            "I_c": "Soil behavior type index profile.",
        },
    },
}
