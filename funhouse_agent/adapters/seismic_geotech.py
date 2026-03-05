"""Seismic geotechnical adapter — site class, M-O pressure, liquefaction."""

from seismic_geotech.site_class import compute_vs30, compute_n_bar, compute_su_bar, classify_site, site_coefficients
from seismic_geotech.mononobe_okabe import mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure
from seismic_geotech.liquefaction import evaluate_liquefaction, CRR_from_N160cs, compute_CSR, stress_reduction_rd, magnitude_scaling_factor, fines_correction
from seismic_geotech.residual_strength import post_liquefaction_strength
from sheet_pile.earth_pressure import rankine_Ka


def _run_site_classification(params: dict) -> dict:
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
    result = {"site_class": site_class, "vs30_m_per_s": round(vs30, 1) if vs30 else None, "n_bar": round(n_bar, 1) if n_bar else None}
    Ss = params.get("Ss")
    S1 = params.get("S1")
    if Ss is not None and S1 is not None:
        sc_result = site_coefficients(site_class, Ss, S1)
        result.update(sc_result.to_dict())
    return result


def _run_seismic_earth_pressure(params: dict) -> dict:
    phi = params["phi"]
    delta = params.get("delta", 2.0 / 3.0 * phi)
    kh = params["kh"]
    kv = params.get("kv", 0.0)
    beta = params.get("beta", 0.0)
    i = params.get("i", 0.0)
    KAE = mononobe_okabe_KAE(phi, delta, kh, kv, beta, i)
    KA = rankine_Ka(phi)
    result = {"KAE": round(KAE, 4), "KA_static": round(KA, 4)}
    if params.get("include_passive", False):
        result["KPE"] = round(mononobe_okabe_KPE(phi, delta, kh, kv, beta, i), 4)
    gamma = params.get("gamma")
    H = params.get("H")
    if gamma is not None and H is not None:
        result.update(seismic_earth_pressure(gamma, H, KAE, KA))
    result.update({"phi_deg": phi, "delta_deg": round(delta, 1), "kh": kh, "kv": kv})
    return result


def _run_liquefaction_evaluation(params: dict) -> dict:
    results = evaluate_liquefaction(
        layer_depths=params["depths"], layer_N160=params["N160"],
        layer_FC=params["FC"], layer_gamma=params["gamma"],
        amax_g=params["amax_g"], gwt_depth=params["gwt_depth"],
        M=params.get("magnitude", 7.5),
    )
    n_liq = sum(1 for r in results if r.get("liquefiable", False))
    min_fos = min((r["FOS_liq"] for r in results), default=99.9)
    return {"layer_results": results, "n_liquefiable": n_liq, "min_FOS_liq": round(min_fos, 3), "magnitude": params.get("magnitude", 7.5)}


def _run_residual_strength(params: dict) -> dict:
    Sr = post_liquefaction_strength(params["N160cs"], params.get("sigma_v_eff"), params.get("method", "seed_harder"))
    result = {"Sr_kPa": round(Sr, 1), "N160cs": params["N160cs"], "method": params.get("method", "seed_harder")}
    if params.get("sigma_v_eff") and params["sigma_v_eff"] > 0:
        result["Sr_ratio"] = round(Sr / params["sigma_v_eff"], 4)
    return result


def _run_csr_crr_check(params: dict) -> dict:
    z = params["depth"]
    N160 = params["N160"]
    FC = params.get("FC", 5.0)
    amax = params["amax_g"]
    sigma_v = params["sigma_v"]
    sigma_v_eff = params["sigma_v_eff"]
    M = params.get("magnitude", 7.5)
    rd = stress_reduction_rd(z, M)
    CSR = compute_CSR(amax, sigma_v, sigma_v_eff, rd)
    MSF = magnitude_scaling_factor(M)
    N160cs = N160 + fines_correction(FC)
    CRR75 = CRR_from_N160cs(N160cs)
    CRR = CRR75 * MSF
    FOS = CRR / CSR if CSR > 0 else 99.9
    return {"CSR": round(CSR, 4), "CRR": round(CRR, 4), "FOS_liq": round(FOS, 3), "N160cs": round(N160cs, 1), "rd": round(rd, 4), "MSF": round(MSF, 3), "liquefiable": FOS < 1.0}


METHOD_REGISTRY = {
    "site_classification": _run_site_classification,
    "seismic_earth_pressure": _run_seismic_earth_pressure,
    "liquefaction_evaluation": _run_liquefaction_evaluation,
    "residual_strength": _run_residual_strength,
    "csr_crr_check": _run_csr_crr_check,
}

METHOD_INFO = {
    "site_classification": {
        "category": "Site Classification",
        "brief": "ASCE 7 site classification and coefficients from Vs30, N-bar, or Su-bar.",
        "parameters": {
            "vs30": {"type": "float", "required": False, "description": "Time-averaged shear wave velocity (m/s)."},
            "layer_thicknesses": {"type": "array", "required": False, "description": "Layer thicknesses for Vs30 calc (m)."},
            "layer_vs": {"type": "array", "required": False, "description": "Layer shear wave velocities (m/s)."},
            "Ss": {"type": "float", "required": False, "description": "Short-period spectral acceleration. If provided with S1, computes Fa/Fv."},
            "S1": {"type": "float", "required": False, "description": "1-second spectral acceleration."},
        },
        "returns": {"site_class": "ASCE 7 site class (A-F).", "Fa": "Short-period coefficient.", "Fv": "Long-period coefficient."},
    },
    "seismic_earth_pressure": {
        "category": "Seismic Earth Pressure",
        "brief": "Mononobe-Okabe active/passive seismic earth pressure coefficients.",
        "parameters": {
            "phi": {"type": "float", "required": True, "description": "Friction angle (degrees)."},
            "kh": {"type": "float", "required": True, "description": "Horizontal seismic coefficient."},
            "gamma": {"type": "float", "required": False, "description": "Soil unit weight (kN/m3) for force calculation."},
            "H": {"type": "float", "required": False, "description": "Wall height (m) for force calculation."},
        },
        "returns": {"KAE": "Active seismic coefficient.", "KA_static": "Static active coefficient."},
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
        "returns": {"min_FOS_liq": "Minimum FOS against liquefaction.", "n_liquefiable": "Number of liquefiable layers."},
    },
    "residual_strength": {
        "category": "Liquefaction",
        "brief": "Post-liquefaction residual strength from corrected SPT.",
        "parameters": {
            "N160cs": {"type": "float", "required": True, "description": "Clean-sand equivalent SPT blow count."},
            "sigma_v_eff": {"type": "float", "required": False, "description": "Effective overburden stress (kPa)."},
            "method": {"type": "str", "required": False, "default": "seed_harder", "description": "seed_harder or olson_stark."},
        },
        "returns": {"Sr_kPa": "Residual strength (kPa)."},
    },
    "csr_crr_check": {
        "category": "Liquefaction",
        "brief": "Quick CSR/CRR check at a single depth.",
        "parameters": {
            "depth": {"type": "float", "required": True, "description": "Depth (m)."},
            "N160": {"type": "float", "required": True, "description": "Corrected SPT blow count."},
            "amax_g": {"type": "float", "required": True, "description": "PGA (g)."},
            "sigma_v": {"type": "float", "required": True, "description": "Total overburden (kPa)."},
            "sigma_v_eff": {"type": "float", "required": True, "description": "Effective overburden (kPa)."},
        },
        "returns": {"CSR": "Cyclic stress ratio.", "CRR": "Cyclic resistance ratio.", "FOS_liq": "Factor of safety."},
    },
}
