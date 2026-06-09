"""OpenSees adapter — PM4Sand DSS, 1D site response."""

from funhouse_agent.adapters import clean_result


def _check_opensees():
    """Raise ValueError if OpenSeesPy is not installed."""
    from opensees_agent import has_opensees
    if not has_opensees():
        raise ValueError(
            "OpenSeesPy is not installed. Install with: pip install openseespy"
        )


def _run_pm4sand_cyclic_dss(params: dict) -> dict:
    _check_opensees()
    from opensees_agent import analyze_pm4sand_dss

    result = analyze_pm4sand_dss(
        Dr=params["Dr"],
        G0=params["G0"],
        hpo=params["hpo"],
        Den=params["Den"],
        sigma_v=params.get("sigma_v", 100.0),
        CSR=params.get("CSR", 0.15),
        K0=params.get("K0", 0.5),
        P_atm=params.get("P_atm", 101.325),
        h0=params.get("h0", -1.0),
        e_max=params.get("e_max", 0.8),
        e_min=params.get("e_min", 0.5),
        nb=params.get("nb", 0.5),
        nd=params.get("nd", 0.1),
        Ado=params.get("Ado", -1.0),
        z_max=params.get("z_max", -1.0),
        c_z=params.get("c_z", 250.0),
        c_e=params.get("c_e", -1.0),
        phi_cv=params.get("phi_cv", 33.0),
        nu=params.get("nu"),
        g_degr=params.get("g_degr", 2.0),
        c_dr=params.get("c_dr", -1.0),
        c_kaf=params.get("c_kaf", -1.0),
        Q=params.get("Q", 10.0),
        R=params.get("R", 1.5),
        m_par=params.get("m_par", 0.01),
        F_sed=params.get("F_sed", -1.0),
        p_sed=params.get("p_sed", -1.0),
        n_cycles=params.get("n_cycles", 30),
        ru_threshold=params.get("ru_threshold", 0.95),
        strain_increment=params.get("strain_increment", 5.0e-6),
        dev_disp_limit=params.get("dev_disp_limit", 0.03),
    )
    return clean_result(result.to_dict())


def _run_site_response_1d(params: dict) -> dict:
    _check_opensees()
    from opensees_agent import analyze_site_response

    result = analyze_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        gwt_depth=params.get("gwt_depth", 0.0),
        bedrock_Vs=params.get("bedrock_Vs", 760.0),
        bedrock_density=params.get("bedrock_density", 2.4),
        damping=params.get("damping", 0.02),
        scale_factor=params.get("scale_factor", 1.0),
        n_elem_per_layer=params.get("n_elem_per_layer", 4),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "pm4sand_cyclic_dss": _run_pm4sand_cyclic_dss,
    "site_response_1d": _run_site_response_1d,
}

METHOD_INFO = {
    "pm4sand_cyclic_dss": {
        "category": "OpenSees",
        "brief": "PM4Sand undrained cyclic DSS analysis for liquefaction triggering.",
        "parameters": {
            "Dr": {"type": "float", "brief": "Relative density (0 to 1, e.g. 0.55)."},
            "G0": {"type": "float", "brief": "Shear modulus coefficient (dimensionless, typically 400-900)."},
            "hpo": {"type": "float", "brief": "Contraction rate parameter (typically 0.05-2.0)."},
            "Den": {"type": "float", "brief": "Mass density (Mg/m3, e.g. 1.7)."},
            "sigma_v": {"type": "float", "brief": "Initial vertical effective stress (kPa).", "default": 100.0},
            "CSR": {"type": "float", "brief": "Cyclic stress ratio to apply.", "default": 0.15},
            "K0": {"type": "float", "brief": "Coefficient of lateral earth pressure.", "default": 0.5},
            "P_atm": {"type": "float", "brief": "Atmospheric pressure (kPa).", "default": 101.325},
            "n_cycles": {"type": "int", "brief": "Maximum number of loading cycles.", "default": 30},
            "ru_threshold": {"type": "float", "brief": "Pore pressure ratio threshold for liquefaction.", "default": 0.95},
            "phi_cv": {"type": "float", "brief": "Critical-state friction angle (degrees).", "default": 33.0},
        },
        "returns": {
            "Dr": "Relative density used.",
            "sigma_v_kPa": "Initial vertical effective stress (kPa).",
            "CSR_applied": "Applied cyclic stress ratio.",
            "liquefied": "Whether liquefaction occurred.",
            "n_cycles_to_liq": "Number of cycles to liquefaction (null if none).",
            "max_ru": "Peak excess pore pressure ratio.",
            "max_shear_strain_pct": "Peak shear strain (%).",
        },
    },
    "site_response_1d": {
        "category": "OpenSees",
        "brief": "1D effective-stress site response analysis (PDMY02/PIMY + Lysmer dashpot).",
        "parameters": {
            "layers": {"type": "array", "brief": "Soil layers: list of dicts with thickness, Vs, density, material_type (sand/clay), phi or su."},
            "motion": {"type": "str", "brief": "Built-in motion name (e.g. 'synthetic_pulse').", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "gwt_depth": {"type": "float", "brief": "Groundwater table depth (m).", "default": 0.0},
            "bedrock_Vs": {"type": "float", "brief": "Bedrock shear wave velocity (m/s).", "default": 760.0},
            "bedrock_density": {"type": "float", "brief": "Bedrock mass density (Mg/m3).", "default": 2.4},
            "damping": {"type": "float", "brief": "Target Rayleigh damping ratio.", "default": 0.02},
            "scale_factor": {"type": "float", "brief": "Scale factor for input acceleration.", "default": 1.0},
            "n_elem_per_layer": {"type": "int", "brief": "Elements per soil layer.", "default": 4},
        },
        "returns": {
            "pga_surface_g": "Peak ground acceleration at surface (g).",
            "pga_input_g": "Peak input acceleration (g).",
            "amplification_ratio": "PGA surface / PGA input.",
            "max_ru": "Peak excess pore pressure ratio in profile.",
        },
    },
}
