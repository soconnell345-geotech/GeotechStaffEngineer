"""OpenSees adapter — PM4Sand DSS, 1D site response."""

from funhouse_agent.adapters import (
    check_nested_keys, mark_required, clean_result, require_params,
)

# Keys every site-response layer may carry, and each material_type's own
# keys: what opensees_agent reads. Anything else was read with .get and a
# default and silently left out, e.g. a sand layer's 'k0' ran at Jaky's
# 1 - sin(phi) (GM-58).
_LAYER_COMMON_KEYS = ("thickness", "Vs", "density", "material_type",
                      "n_surf", "e_init", "n_elem", "nu", "permeability",
                      "stabilization_alpha",
                      # the stress-strain curve and its keys (GM-80)
                      "curve", "plas_index", "ocr", "freq_hz", "n_cycles",
                      "uniformity_coeff", "diam_mean", "G_Gmax_curve",
                      "damping_curve")
_MATERIAL_KEYS = {
    "sand": ("phi", "K0", "PTAng", "contrac1", "contrac2", "contrac3",
             "dilat1", "dilat2", "dilat3", "liquefac1", "liquefac2"),
    "clay": ("su",),
}
# Words the near-miss matcher cannot place; only the keys the layer's own
# material reads are offered. Suggestions only, never renamed. Keys are
# the checker's normalised form: lower case, letters and digits only.
_LAYER_KEY_HINTS = {
    "rho": ("density",), "massdensity": ("density",),
    "cohesion": ("su",), "c": ("su",),
    "frictionangle": ("phi",), "phaseangle": ("PTAng",),
    "phasetransformationangle": ("PTAng",), "e": ("e_init",),
    "pi": ("plas_index",), "plasticityindex": ("plas_index",),
    "cu": ("su", "uniformity_coeff"), "d50": ("diam_mean",),
    "poisson": ("nu",), "poissonsratio": ("nu",),
    "nelements": ("n_elem",), "model": ("curve",),
    "soilmodel": ("curve",),
    "e0": ("e_init",), "voidratio": ("e_init",), "type": ("material_type",),
    "soiltype": ("material_type",), "h": ("thickness",),
    # 'k' is hydraulic conductivity, or a sand layer's K0 written short
    "k": ("permeability", "K0"), "perm": ("permeability",),
    "hydraulicconductivity": ("permeability",),
    "conductivity": ("permeability",), "ksat": ("permeability",),
    # the u-p element's pressure stabilization (GM-260)
    "alpha": ("stabilization_alpha",),
    "stabilization": ("stabilization_alpha",),
    "pressurestabilization": ("stabilization_alpha",),
}


def _check_layers(layers):
    """Refuse a key a site-response layer's material does not read
    (GM-58). A layer with no material_type, or one the module does not
    know, is left to the module's own check, which names the materials."""
    for i, layer in enumerate(layers if isinstance(layers, list) else []):
        mat = layer.get("material_type") if isinstance(layer, dict) else None
        own = _MATERIAL_KEYS.get(str(mat).lower().strip())
        if own is None:
            continue
        check_nested_keys(layer, _LAYER_COMMON_KEYS + own,
                          method="site_response_1d",
                          label=f"layers[{i}] ({mat})",
                          hints=_LAYER_KEY_HINTS)


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

    require_params(params, ["Dr", "G0", "hpo", "Den"],
                   method="pm4sand_cyclic_dss")
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

    require_params(params, ["layers"], method="site_response_1d")
    _check_layers(params["layers"])
    result = analyze_site_response(
        layers=params["layers"],
        motion=params.get("motion"),
        accel_history=params.get("accel_history"),
        dt=params.get("dt"),
        target_pga_g=params.get("target_pga_g"),
        gwt_depth=params.get("gwt_depth", 0.0),
        bedrock_Vs=params.get("bedrock_Vs", 760.0),
        bedrock_density=params.get("bedrock_density", 2.4),
        damping=params.get("damping", 0.02),
        scale_factor=params.get("scale_factor", 1.0),
        n_elem_per_layer=params.get("n_elem_per_layer", 4),
        linear_elastic=bool(params.get("linear_elastic", False)),
        rayleigh_freqs_hz=params.get("rayleigh_freqs_hz"),
        pga_lowpass_hz=params.get("pga_lowpass_hz", 10.0),
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
            # PM4Sand secondary parameters (Boulanger & Ziotopoulou); -1 = the
            # model computes its own default.
            "h0": {"type": "float", "brief": "PM4Sand h0, plastic-to-elastic modulus ratio (-1 = model default).", "default": -1.0},
            "e_max": {"type": "float", "brief": "Maximum void ratio.", "default": 0.8},
            "e_min": {"type": "float", "brief": "Minimum void ratio.", "default": 0.5},
            "nb": {"type": "float", "brief": "Bounding surface parameter.", "default": 0.5},
            "nd": {"type": "float", "brief": "Dilatancy surface parameter.", "default": 0.1},
            "Ado": {"type": "float", "brief": "Dilatancy parameter (-1 = model default).", "default": -1.0},
            "z_max": {"type": "float", "brief": "Maximum fabric-dilatancy tensor (-1 = model default).", "default": -1.0},
            "c_z": {"type": "float", "brief": "Fabric growth parameter.", "default": 250.0},
            "c_e": {"type": "float", "brief": "Strain accumulation factor (-1 = model default).", "default": -1.0},
            "nu": {"type": "float", "brief": "Poisson's ratio (omit for the model default).", "default": None},
            "g_degr": {"type": "float", "brief": "Elastic modulus degradation with fabric.", "default": 2.0},
            "c_dr": {"type": "float", "brief": "Rotated dilatancy surface parameter (-1 = model default).", "default": -1.0},
            "c_kaf": {"type": "float", "brief": "Sustained static shear effect on plastic modulus (-1 = model default).", "default": -1.0},
            "Q": {"type": "float", "brief": "Critical state line parameter Q.", "default": 10.0},
            "R": {"type": "float", "brief": "Critical state line parameter R.", "default": 1.5},
            "m_par": {"type": "float", "brief": "Yield surface size (stress ratio).", "default": 0.01},
            "F_sed": {"type": "float", "brief": "Minimum elastic modulus reduction during reconsolidation (-1 = model default).", "default": -1.0},
            "p_sed": {"type": "float", "brief": "Mean effective stress up to which reconsolidation strains are enhanced (-1 = model default).", "default": -1.0},
            "strain_increment": {"type": "float", "brief": "Displacement increment per analysis step (m).", "default": 5.0e-6},
            "dev_disp_limit": {"type": "float", "brief": "Largest deviatoric displacement before a half-cycle stops (m).", "default": 0.03},
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
            "layers": {"type": "array", "brief": "Soil layers (required): list of dicts, each with thickness (m), Vs (m/s), density (Mg/m3) and material_type: 'sand' needs phi (degrees; optional K0 (default Jaky 1 - sin phi; it sets only the reference pressure where G = density Vs^2, while the column's own horizontal stress after gravity follows nu), PTAng and the PressureDependMultiYield02 constants contrac1, contrac2, contrac3, dilat1, dilat2, dilat3, liquefac1, liquefac2, default the OpenSees wiki's Dr = 40 % column), 'clay' needs su (kPa, undrained strength in simple shear). Optional for any layer: n_elem (its element count, default n_elem_per_layer; a layer the water table falls inside is meshed in two parts with a node at the water table), nu (Poisson's ratio, default 0.3 sand, 0.495 clay), permeability (hydraulic conductivity k, m/s, default 1e-4 sand, 1e-7 clay), stabilization_alpha (the u-p element's pressure stabilization, default h^2 / (4 (K + 4/3 G)) per element, the element documentation's formula), e_init (void ratio, default 0.77 sand, 0.85 clay), n_surf (yield surfaces, default 20; not with a fitted curve), and curve, the modulus-reduction curve the material's backbone follows to the layer's strength: 'opensees' (default, the material's own hyperbolic backbone, much stiffer than laboratory curves in sand), 'darendeli' (Darendeli 2001; optional plas_index 0, ocr 1, freq_hz 1, n_cycles 10), 'menq' (Menq 2003; needs uniformity_coeff Cu and diam_mean D50 in mm) or 'user' (needs G_Gmax_curve {strain_pct: [...], G_Gmax: [...]}; optional damping_curve {strain_pct: [...], damping_pct: [...]}, reported, not imposed)."},
            "motion": {"type": "str", "brief": "Built-in motion name (e.g. 'synthetic_pulse'), taken as a rock outcrop motion.", "default": None},
            "accel_history": {"type": "array", "brief": "Custom acceleration time history (g).", "default": None},
            "dt": {"type": "float", "brief": "Time step for custom motion (s).", "default": None},
            "target_pga_g": {"type": "float", "brief": "Scale the input motion linearly so its PGA equals this value (g) before the analysis (the built-in motions have fixed PGAs: synthetic_pulse 0.30 g, synthetic_long 0.15 g). Applied before scale_factor; give one or the other.", "default": None},
            "gwt_depth": {"type": "float", "brief": "Groundwater table depth (m).", "default": 0.0},
            "bedrock_Vs": {"type": "float", "brief": "Bedrock shear wave velocity (m/s).", "default": 760.0},
            "bedrock_density": {"type": "float", "brief": "Bedrock mass density (Mg/m3).", "default": 2.4},
            "damping": {"type": "float", "brief": "Target Rayleigh damping ratio.", "default": 0.02},
            "scale_factor": {"type": "float", "brief": "Scale factor for input acceleration.", "default": 1.0},
            "n_elem_per_layer": {"type": "int", "brief": "Elements per soil layer (a layer's own n_elem overrides it).", "default": 4},
            "rayleigh_freqs_hz": {"type": "array", "brief": "The two frequencies [f1, f2] (Hz) at which the Rayleigh damping equals damping.", "default": None},
            "pga_lowpass_hz": {"type": "float", "brief": "Corner (Hz) of the low-pass (fourth-order Butterworth, run forward and backward) behind pga_surface_lowpass_g and pga_input_lowpass_g, and of the spike test on the raw PGA; the tool's choice, no reference prints one. pga_surface_g is not filtered.", "default": 10.0},
            "linear_elastic": {"type": "bool", "brief": "Keep the soil linear elastic through the shaking (Gmax = rho*Vs^2, Rayleigh damping only): a check of the column, not a design analysis.", "default": False},
        },
        "returns": {
            "analysis_type": "'nonlinear effective stress' or 'linear elastic'.",
            "total_depth_m": "Profile depth (m).",
            "n_layers": "Number of layers.",
            "site_period_s": "Site period 4H/Vs (s).",
            "motion_name": "Input motion name (notes any PGA scaling).",
            "pga_surface_g": "Peak ground acceleration at surface (g), the raw peak of the computed record. In a sand that dilates it can be a spike a few hundredths of a second long whose height depends on the element size; pga_surface_lowpass_g and Sa_surface_g at 0.2 s and longer depend on it much less.",
            "pga_input_g": "Peak input acceleration (g).",
            "amplification_factor": "PGA surface / PGA input.",
            "pga_time_s": "Time of the surface PGA (s).",
            "pga_lowpass_hz": "Corner (Hz) of the low-pass behind pga_surface_lowpass_g and pga_input_lowpass_g.",
            "pga_surface_lowpass_g": "Peak of the surface record low-passed at pga_lowpass_hz (g): an alternative to the raw PGA that a spike above the corner does not set; it also removes real motion above the corner.",
            "pga_input_lowpass_g": "Peak of the input record low-passed the same way (g), for an amplification on the same basis.",
            "pga_share_above_lowpass": "At the raw PGA's instant, the share of it that is motion above pga_lowpass_hz (the record less the low-passed record).",
            "pga_set_by_spike": "True when pga_share_above_lowpass is over one half: the raw PGA is set by a spike.",
            "pga_note": "Only when pga_set_by_spike: the spike, its time and the alternatives, in words.",
            "Sa_surface_0_01s_g": "Surface spectral acceleration at 0.01 s (g, 5 % damping), which ASCE 7-22 puts in place of SDS/2.5 for a site-specific spectrum (Tables 19.3-1 and 19.3-2, note c); it carries a spike as the raw PGA does.",
            "periods_s": "Periods (s) of Sa_surface_g and Sa_input_g: 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2 and 3 s.",
            "Sa_surface_g": "Surface spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "Sa_input_g": "Input (rock outcrop) spectral acceleration (g, 5 % damping), one per period in periods_s.",
            "max_shear_strain_pct": "Peak shear strain in the profile (%).",
            "max_ru": "Peak excess pore pressure ratio in the profile: excess pore pressure over the initial vertical effective stress (not capped at 1).",
            "solver_steps": "Dynamic steps by algorithm, the steps that needed a fallback algorithm and those taken in substeps (substeps can move a peak).",
            "layer_curves": "Each layer's curve: G/Gmax and the curve's damping at 0.001-1 % strain, and the Masing damping its fitted backbone gives.",
            "rayleigh_freqs_hz": "The two Rayleigh damping frequencies used (Hz).",
            "choices": "What the column was built with (water table, curve, Poisson's ratio, element count, sand constants, K0 and reference pressure, phase transformation angle, void ratio, yield surfaces, permeability, pressure stabilization, Rayleigh frequencies) and the surface PGA reported with its alternatives and the low-pass, each with its basis.",
        },
    },
}


def _motion_alternatives(method_info: dict) -> dict:
    """Say on each motion parameter that a built-in name OR a custom history
    (+ dt) is needed: neither alone is marked required, one of them is."""
    for info in method_info.values():
        params = info.get("parameters") or {}
        for name, spec in params.items():
            if not isinstance(spec, dict):
                continue
            suffix = name[len("motion"):] if name.startswith("motion") else (
                name[len("accel_history"):] if name.startswith("accel_history")
                else None)
            if suffix is None:
                continue
            spec["description"] = (
                spec.get("description", "").rstrip(". ")
                + f". Give motion{suffix} (a built-in name) OR "
                  f"accel_history{suffix} + dt: one of the two is required.")
    return method_info


# One METHOD_INFO style everywhere: explicit required flags matching the code
# above (live smoke wave 1, G7).
mark_required(METHOD_INFO, {
    "pm4sand_cyclic_dss": ["Dr", "G0", "hpo", "Den"],
    "site_response_1d": ["layers"],
})
_motion_alternatives(METHOD_INFO)


# What a left-out input with no default stands for (BACKLOG GM-165):
# Poisson's ratio from K0 and an unscaled motion.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
_UNSCALED = ("the motion as given, not scaled",
             "the input motion's own peak ground acceleration")
declare_when_absent(METHOD_INFO, {
    "pm4sand_cyclic_dss": {
        "nu": ("K0 / (1 + K0)", "Poisson's ratio from the at-rest "
               "coefficient K0 given")},
    "site_response_1d": {
        "target_pga_g": _UNSCALED,
        "rayleigh_freqs_hz": (
            "the site frequency Vs/(4H) and five times it",
            "the column's first mode and a higher frequency, so damping "
            "stays near the target over the band that matters")},
})
