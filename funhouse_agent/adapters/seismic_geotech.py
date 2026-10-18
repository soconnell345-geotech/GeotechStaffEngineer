"""Seismic geotechnical adapter — site class, M-O pressure, liquefaction."""

from funhouse_agent.adapters import (declare_applies_when,
                                     reject_unknown_params, require_params)
from seismic_geotech.site_class import compute_vs30, compute_n_bar, compute_su_bar, classify_site, site_coefficients
from seismic_geotech.mononobe_okabe import mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure, earth_pressure_choices, mo_no_solution_warning
from seismic_geotech.liquefaction import evaluate_liquefaction, CRR_from_N160cs, compute_CSR, stress_reduction_rd, magnitude_scaling_factor, fines_correction, liquefaction_choices
from seismic_geotech.residual_strength import post_liquefaction_strength
from seismic_geotech.dynamic_properties import (
    gmax_from_vs, gmax_cpt_sand_rix_stokoe, gmax_cpt_clay_mayne_rix,
    gmax_hardin_black, gmax_clay_andersen, modulus_reduction_ishibashi_zhang,
)
from sheet_pile.earth_pressure import rankine_Ka


def run_site_classification(params: dict, *,
                            method: str = "site_classification"):
    """(site class, vs30, n_bar, su_bar, coefficients) of
    ``site_classification``: the tool's own code path, also run by
    ``calc_package.seismic_package`` (GM-141). The averages are those
    worked out from layers or given (None when neither); the coefficients
    are a SiteClassResult when Ss and S1 are given, else None. ``method``
    names the caller in error messages."""
    reject_unknown_params(
        params,
        ("vs30", "layer_thicknesses", "layer_vs", "layer_N", "n_bar",
         "su_layer_thicknesses", "layer_su", "su_bar", "Ss", "S1", "pga"),
        method=method)
    if all(params.get(k) is None for k in
           ("vs30", "n_bar", "su_bar", "layer_vs", "layer_N", "layer_su")):
        raise ValueError(
            f"{method}: provide one of 'vs30' (m/s), 'n_bar' "
            "(avg SPT N), 'su_bar' (kPa), or layer arrays "
            "(layer_thicknesses + layer_vs / layer_N / layer_su)."
        )
    vs30 = None
    n_bar = None
    su_bar = None
    if "layer_thicknesses" in params and "layer_vs" in params:
        vs30 = compute_vs30(params["layer_thicknesses"], params["layer_vs"])
    elif "vs30" in params:
        vs30 = params["vs30"]
    if "layer_thicknesses" in params and "layer_N" in params:
        n_bar = compute_n_bar(params["layer_thicknesses"], params["layer_N"])
    elif "n_bar" in params:
        n_bar = params["n_bar"]
    if "su_layer_thicknesses" in params and "layer_su" in params:
        su_bar = compute_su_bar(params["su_layer_thicknesses"], params["layer_su"])
    elif "su_bar" in params:
        su_bar = params["su_bar"]
    site_class = classify_site(vs30=vs30, n_bar=n_bar, su_bar=su_bar)
    Ss = params.get("Ss")
    S1 = params.get("S1")
    coefficients = None
    if Ss is not None and S1 is not None:
        coefficients = site_coefficients(site_class, Ss, S1,
                                         pga=params.get("pga"))
    return site_class, vs30, n_bar, su_bar, coefficients


def _run_site_classification(params: dict) -> dict:
    site_class, vs30, n_bar, _, coefficients = run_site_classification(params)
    result = {"site_class": site_class, "vs30_m_per_s": round(vs30, 1) if vs30 else None, "n_bar": round(n_bar, 1) if n_bar else None}
    if coefficients is not None:
        result.update(coefficients.to_dict())
    return result


_MO_VALID = ("phi", "delta", "kh", "kv", "beta", "i", "include_passive",
             "gamma", "H", "KA")


def run_seismic_earth_pressure(params: dict, *,
                               method: str = "seismic_earth_pressure"):
    """(result, forces, inputs in force) of ``seismic_earth_pressure``: the
    tool's own code path, also run by ``calc_package.seismic_package``
    (GM-141). ``result`` is a SeismicEarthPressureResult; ``forces`` the
    resultants as ``seismic_earth_pressure`` gives them (empty without
    gamma and H); the inputs in force carry the static KA behind PA (as
    given, else Rankine Ka from phi). ``method`` names the caller in error
    messages."""
    from seismic_geotech.results import SeismicEarthPressureResult
    reject_unknown_params(params, _MO_VALID, method=method)
    require_params(params, ["phi", "kh"], method=method, valid=_MO_VALID)
    phi = params["phi"]
    delta = params.get("delta", 2.0 / 3.0 * phi)
    kh = params["kh"]
    kv = params.get("kv", 0.0)
    beta = params.get("beta", 0.0)
    i = params.get("i", 0.0)
    include_passive = params.get("include_passive", False)
    KAE = mononobe_okabe_KAE(phi, delta, kh, kv, beta, i)
    # KA not given: Rankine Ka from phi (vertical wall, level backfill, no
    # wall friction; GM-120)
    ka_given = params.get("KA") is not None
    KA = float(params["KA"]) if ka_given else rankine_Ka(phi)
    KPE = (mononobe_okabe_KPE(phi, delta, kh, kv, beta, i)
           if include_passive else 0.0)
    gamma = params.get("gamma")
    H = params.get("H")
    forces = (seismic_earth_pressure(gamma, H, KAE, KA)
              if gamma is not None and H is not None else {})
    note = mo_no_solution_warning(phi, kh, kv, i)
    result = SeismicEarthPressureResult(
        KAE=KAE, KPE=KPE,
        PAE_total=forces.get("PAE_total_kN_per_m", 0.0),
        PA_static=forces.get("PA_static_kN_per_m", 0.0),
        delta_PAE=forces.get("delta_PAE_kN_per_m", 0.0),
        height_of_application=forces.get("height_of_application_m", 0.0),
        phi=phi, delta=delta, kh=kh, kv=kv,
        choices=earth_pressure_choices(
            phi, kh, kv=params.get("kv"), delta_deg=params.get("delta"),
            i_deg=i, ka=KA if ka_given else "rankine", ka_how="given",
            forces=bool(forces), ka_can_be_given=True, beta_deg=beta),
        warnings=[note] if note else [],
    )
    used = {"phi": phi, "delta": delta, "kh": kh, "kv": kv, "beta": beta,
            "i": i, "include_passive": include_passive, "gamma": gamma,
            "H": H, "KA": KA, "KA_given": ka_given}
    return result, forces, used


def _run_seismic_earth_pressure(params: dict) -> dict:
    r, forces, used = run_seismic_earth_pressure(params)
    result = {"KAE": round(r.KAE, 4), "KA_static": round(used["KA"], 4)}
    if used["include_passive"]:
        result["KPE"] = round(r.KPE, 4)
    result.update(forces)
    result.update({"phi_deg": r.phi, "delta_deg": round(r.delta, 1),
                   "kh": r.kh, "kv": r.kv})
    result["choices"] = r.choices
    if r.warnings:
        result["warnings"] = list(r.warnings)
    return result


_LIQ_VALID = ("depths", "N160", "FC", "gamma", "amax_g", "gwt_depth",
              "magnitude")


def run_liquefaction_evaluation(params: dict, *,
                                method: str = "liquefaction_evaluation"):
    """The LiquefactionResult of ``liquefaction_evaluation``: the tool's own
    code path, also run by ``calc_package.seismic_package`` (GM-141).
    ``method`` names the caller in error messages."""
    from seismic_geotech.results import LiquefactionResult
    reject_unknown_params(params, _LIQ_VALID, method=method)
    require_params(params, ["depths", "N160", "FC", "gamma", "amax_g",
                            "gwt_depth"],
                   method=method, valid=_LIQ_VALID)
    magnitude = params.get("magnitude", 7.5)
    results = evaluate_liquefaction(
        layer_depths=params["depths"], layer_N160=params["N160"],
        layer_FC=params["FC"], layer_gamma=params["gamma"],
        amax_g=params["amax_g"], gwt_depth=params["gwt_depth"],
        M=magnitude,
    )
    return LiquefactionResult(
        layer_results=results, amax_g=params["amax_g"], magnitude=magnitude,
        gwt_depth=params["gwt_depth"],
        choices=liquefaction_choices(results, magnitude))


def _run_liquefaction_evaluation(params: dict) -> dict:
    r = run_liquefaction_evaluation(params)
    results = r.layer_results
    n_liq = sum(1 for layer in results if layer.get("liquefiable", False))
    min_fos = min((layer["FOS_liq"] for layer in results), default=99.9)
    return {"layer_results": results, "n_liquefiable": n_liq, "min_FOS_liq": round(min_fos, 3), "magnitude": r.magnitude,
            "choices": r.choices}


def _run_residual_strength(params: dict) -> dict:
    _valid = ("N160cs", "sigma_v_eff", "method")
    reject_unknown_params(params, _valid, method="residual_strength")
    require_params(params, ["N160cs"], method="residual_strength", valid=_valid)
    Sr = post_liquefaction_strength(params["N160cs"], params.get("sigma_v_eff"), params.get("method", "seed_harder"))
    result = {"Sr_kPa": round(Sr, 1), "N160cs": params["N160cs"], "method": params.get("method", "seed_harder")}
    if params.get("sigma_v_eff") and params["sigma_v_eff"] > 0:
        result["Sr_ratio"] = round(Sr / params["sigma_v_eff"], 4)
    return result


def _run_csr_crr_check(params: dict) -> dict:
    _valid = ("depth", "N160", "FC", "amax_g", "sigma_v", "sigma_v_eff",
              "magnitude")
    reject_unknown_params(params, _valid, method="csr_crr_check")
    require_params(params, ["depth", "N160", "amax_g", "sigma_v",
                            "sigma_v_eff"],
                   method="csr_crr_check", valid=_valid)
    z = params["depth"]
    N160 = params["N160"]
    FC = params.get("FC", 5.0)
    amax = params["amax_g"]
    sigma_v = params["sigma_v"]
    sigma_v_eff = params["sigma_v_eff"]
    M = params.get("magnitude", 7.5)
    rd = stress_reduction_rd(z)
    # compute_CSR already applies rd (from z) and divides by MSF, so the
    # returned CSR is on the M7.5 basis and pairs directly with CRR_from_N160cs.
    CSR = compute_CSR(amax, sigma_v, sigma_v_eff, z, M)
    MSF = magnitude_scaling_factor(M)
    N160cs = fines_correction(N160, FC)
    CRR = CRR_from_N160cs(N160cs)
    FOS = CRR / CSR if CSR > 0 else 99.9
    return {"CSR": round(CSR, 4), "CRR": round(CRR, 4), "FOS_liq": round(FOS, 3), "N160cs": round(N160cs, 1), "rd": round(rd, 4), "MSF": round(MSF, 3), "liquefiable": FOS < 1.0}


def _run_gmax(params: dict) -> dict:
    _valid = ("correlation", "Vs", "unit_weight", "qc_MPa", "sigma_vo_eff",
              "sigma_m_eff", "void_ratio", "coefficient_B", "PI", "OCR")
    reject_unknown_params(params, _valid, method="gmax")
    require_params(params, ["correlation"], method="gmax", valid=_valid)
    corr = params["correlation"]
    if corr == "vs":
        require_params(params, ["Vs", "unit_weight"], method="gmax", valid=_valid)
        r = gmax_from_vs(params["Vs"], params["unit_weight"])
    elif corr == "cpt_sand":
        require_params(params, ["qc_MPa", "sigma_vo_eff"], method="gmax",
                       valid=_valid)
        r = gmax_cpt_sand_rix_stokoe(params["qc_MPa"], params["sigma_vo_eff"])
    elif corr == "cpt_clay":
        require_params(params, ["qc_MPa"], method="gmax", valid=_valid)
        r = gmax_cpt_clay_mayne_rix(params["qc_MPa"])
    elif corr == "hardin_black":
        require_params(params, ["sigma_m_eff", "void_ratio"], method="gmax",
                       valid=_valid)
        r = gmax_hardin_black(params["sigma_m_eff"], params["void_ratio"],
                              params.get("coefficient_B", 875.0))
    elif corr == "andersen":
        require_params(params, ["PI", "OCR", "sigma_vo_eff"], method="gmax",
                       valid=_valid)
        r = gmax_clay_andersen(params["PI"], params["OCR"],
                               params["sigma_vo_eff"])
    else:
        raise ValueError(
            f"gmax: unknown correlation '{corr}' "
            "(vs | cpt_sand | cpt_clay | hardin_black | andersen)")
    out = {"correlation": corr,
           "Gmax_kPa": round(r["Gmax_kPa"], 1),
           "Gmax_MPa": round(r["Gmax_kPa"] / 1000.0, 2)}
    for k, v in r.items():
        if k != "Gmax_kPa":
            out[k] = round(v, 2)
    return out


def _run_modulus_reduction(params: dict) -> dict:
    _valid = ("strain_pct", "PI", "sigma_m_eff")
    reject_unknown_params(params, _valid, method="modulus_reduction")
    require_params(params, ["strain_pct", "sigma_m_eff"],
                   method="modulus_reduction", valid=_valid)
    r = modulus_reduction_ishibashi_zhang(
        params["strain_pct"], params.get("PI", 0.0), params["sigma_m_eff"])
    return {"G_over_Gmax": round(r["G_over_Gmax"], 4),
            "damping_pct": round(r["damping_pct"], 3),
            "strain_pct": params["strain_pct"],
            "PI": params.get("PI", 0.0)}


METHOD_REGISTRY = {
    "site_classification": _run_site_classification,
    "gmax": _run_gmax,
    "modulus_reduction": _run_modulus_reduction,
    "seismic_earth_pressure": _run_seismic_earth_pressure,
    "liquefaction_evaluation": _run_liquefaction_evaluation,
    "residual_strength": _run_residual_strength,
    "csr_crr_check": _run_csr_crr_check,
}

METHOD_INFO = {
    "gmax": {
        "category": "Dynamic Properties",
        "brief": "Small-strain shear modulus Gmax from Vs, CPT, void ratio, or PI/OCR.",
        "parameters": {
            "correlation": {"type": "str", "required": True,
                            "allowed_values": ["vs", "cpt_sand", "cpt_clay", "hardin_black", "andersen"],
                            "description": "vs: G=rho*Vs^2; cpt_sand: Rix & Stokoe (1991); cpt_clay: Mayne & Rix (1993); hardin_black: Hardin & Black (1968) from void ratio; andersen: Andersen (2015) clay from PI/OCR."},
            "Vs": {"type": "float", "required": False, "description": "Shear wave velocity (m/s) — for 'vs'."},
            "unit_weight": {"type": "float", "required": False, "description": "Bulk unit weight (kN/m3) — for 'vs'."},
            "qc_MPa": {"type": "float", "required": False, "description": "CPT tip resistance (MPa) — for 'cpt_sand'/'cpt_clay'."},
            "sigma_vo_eff": {"type": "float", "required": False, "description": "Vertical effective stress (kPa) — for 'cpt_sand'/'andersen'."},
            "sigma_m_eff": {"type": "float", "required": False, "description": "Mean effective stress (kPa) — for 'hardin_black'."},
            "void_ratio": {"type": "float", "required": False, "description": "Void ratio e0 — for 'hardin_black'."},
            "coefficient_B": {"type": "float", "required": False, "default": 875.0, "description": "Hardin-Black calibration coefficient (default: PISA dense marine sand)."},
            "PI": {"type": "float", "required": False, "description": "Plasticity index (%) — for 'andersen'."},
            "OCR": {"type": "float", "required": False, "description": "Overconsolidation ratio — for 'andersen'."},
        },
        "returns": {"Gmax_kPa": "Small-strain shear modulus (kPa).", "Gmax_MPa": "Same in MPa."},
    },
    "modulus_reduction": {
        "category": "Dynamic Properties",
        "brief": "Ishibashi & Zhang (1993) G/Gmax and damping vs cyclic strain.",
        "parameters": {
            "strain_pct": {"type": "float", "required": True, "description": "Cyclic shear strain in PERCENT (e.g. 0.01 = 1e-4 strain)."},
            "PI": {"type": "float", "required": False, "default": 0.0, "description": "Plasticity index (%); 0 for sands."},
            "sigma_m_eff": {"type": "float", "required": True, "description": "Mean effective confining stress (kPa)."},
        },
        "returns": {"G_over_Gmax": "Modulus reduction ratio.", "damping_pct": "Damping ratio (%)."},
    },
    "site_classification": {
        "category": "Site Classification",
        "brief": "ASCE 7 site classification and coefficients from Vs30, N-bar, or Su-bar.",
        "parameters": {
            "vs30": {"type": "float", "required": False, "description": "Time-averaged shear wave velocity (m/s)."},
            "layer_thicknesses": {"type": "array", "required": False, "description": "Layer thicknesses for Vs30/N-bar calc (m)."},
            "layer_vs": {"type": "array", "required": False, "description": "Layer shear wave velocities (m/s)."},
            "layer_N": {"type": "array", "required": False, "description": "Layer SPT N values for N-bar calc (with layer_thicknesses)."},
            "n_bar": {"type": "float", "required": False, "description": "Average SPT blow count (alternative to vs30)."},
            "su_layer_thicknesses": {"type": "array", "required": False, "description": "Cohesive layer thicknesses for su-bar calc (m)."},
            "layer_su": {"type": "array", "required": False, "description": "Layer undrained strengths (kPa) for su-bar calc."},
            "su_bar": {"type": "float", "required": False, "description": "Average undrained strength (kPa) (alternative to vs30)."},
            "Ss": {"type": "float", "required": False, "description": "Short-period spectral acceleration. If provided with S1, computes Fa/Fv."},
            "S1": {"type": "float", "required": False, "description": "1-second spectral acceleration."},
            "pga": {"type": "float", "required": False, "description": "Mapped peak ground acceleration (g). If provided with Ss/S1, also returns Fpga and the site-adjusted PGA."},
        },
        "returns": {"site_class": "ASCE 7 site class (A-F).", "Fa": "Short-period coefficient.", "Fv": "Long-period coefficient."},
    },
    "seismic_earth_pressure": {
        "category": "Seismic Earth Pressure",
        "brief": "Mononobe-Okabe active/passive seismic earth pressure coefficients.",
        "parameters": {
            "phi": {"type": "float", "required": True, "description": "Friction angle (degrees)."},
            "kh": {"type": "float", "required": True, "description": "Horizontal seismic coefficient."},
            "kv": {"type": "float", "required": False, "default": 0.0, "description": "Vertical seismic coefficient."},
            "delta": {"type": "float", "required": False, "description": "Wall friction angle (deg). Defaults to 2/3 phi."},
            "beta": {"type": "float", "required": False, "default": 0.0, "description": "Wall back face batter from vertical (degrees)."},
            "i": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees)."},
            "include_passive": {"type": "bool", "required": False, "default": False, "description": "Also return KPE (passive seismic coefficient)."},
            "gamma": {"type": "float", "required": False, "description": "Soil unit weight (kN/m3) for force calculation."},
            "H": {"type": "float", "required": False, "description": "Wall height (m) for force calculation."},
            "KA": {"type": "float", "required": False, "description": "Static active coefficient behind PA and delta-PAE. Not given: Rankine Ka from phi (vertical wall, level backfill, no wall friction)."},
        },
        "returns": {"KAE": "Active seismic coefficient.", "KA_static": "Static active coefficient.", "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives."},
    },
    "liquefaction_evaluation": {
        "category": "Liquefaction",
        "brief": "Full liquefaction triggering evaluation (SPT-based, simplified method).",
        "parameters": {
            "depths": {"type": "array", "required": True, "description": "Layer depths (m)."},
            "N160": {"type": "array", "required": True, "description": "Corrected SPT blow counts."},
            "FC": {"type": "array", "required": True, "description": "Fines content (%)."},
            "gamma": {"type": "array", "required": True, "description": "Unit weights (kN/m3)."},
            "amax_g": {"type": "float", "required": True, "description": "Peak ground acceleration (g)."},
            "gwt_depth": {"type": "float", "required": True, "description": "Groundwater depth (m)."},
            "magnitude": {"type": "float", "required": False, "default": 7.5, "description": "Earthquake magnitude."},
        },
        "returns": {"min_FOS_liq": "Minimum FOS against liquefaction.", "n_liquefiable": "Number of liquefiable layers.", "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives."},
    },
    "residual_strength": {
        "category": "Liquefaction",
        "brief": "Post-liquefaction residual strength from corrected SPT.",
        "parameters": {
            "N160cs": {"type": "float", "required": True, "description": "Clean-sand equivalent SPT blow count."},
            "sigma_v_eff": {"type": "float", "required": False, "description": "Effective overburden stress (kPa)."},
            "method": {"type": "str", "required": False, "default": "seed_harder", "allowed_values": ["seed_harder", "olson_stark"], "description": "Residual strength correlation."},
        },
        "returns": {"Sr_kPa": "Residual strength (kPa)."},
    },
    "csr_crr_check": {
        "category": "Liquefaction",
        "brief": "Quick CSR/CRR check at a single depth.",
        "parameters": {
            "depth": {"type": "float", "required": True, "description": "Depth (m)."},
            "N160": {"type": "float", "required": True, "description": "Corrected SPT blow count."},
            "FC": {"type": "float", "required": False, "default": 5.0, "description": "Fines content (%)."},
            "amax_g": {"type": "float", "required": True, "description": "PGA (g)."},
            "sigma_v": {"type": "float", "required": True, "description": "Total overburden (kPa)."},
            "sigma_v_eff": {"type": "float", "required": True, "description": "Effective overburden (kPa)."},
            "magnitude": {"type": "float", "required": False, "default": 7.5, "description": "Earthquake magnitude."},
        },
        "returns": {"CSR": "Cyclic stress ratio.", "CRR": "Cyclic resistance ratio.", "FOS_liq": "Factor of safety."},
    },
}

# When each default applies (GM-119): B to Hardin & Black only.
declare_applies_when(METHOD_INFO, {
    "gmax": {"coefficient_B": {"correlation": ["hardin_black"]}}})
