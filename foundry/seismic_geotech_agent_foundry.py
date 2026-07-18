"""
Seismic Geotechnical Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. seismic_geotech_agent        - Run a seismic geotechnical analysis
  2. seismic_geotech_list_methods - Browse available methods
  3. seismic_geotech_describe_method - Get detailed parameter docs

Covers site classification, Mononobe-Okabe seismic pressures,
liquefaction triggering, and post-liquefaction residual strength.

FOUNDRY SETUP:
  - pip install geotech-staff-engineer (PyPI)
  - These functions accept and return JSON strings for LLM compatibility
"""

import json
import math
try:
    from functions.api import function
except ImportError:
    def function(fn):
        fn.__wrapped__ = fn
        return fn

from seismic_geotech.site_class import (
    compute_vs30, compute_n_bar, compute_su_bar,
    classify_site, site_coefficients,
)
from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure,
)
from seismic_geotech.liquefaction import (
    evaluate_liquefaction, fines_correction, CRR_from_N160cs,
    compute_CSR, stress_reduction_rd, magnitude_scaling_factor,
)
from seismic_geotech.residual_strength import post_liquefaction_strength
from sheet_pile.earth_pressure import rankine_Ka


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
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


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def _run_site_classification(params: dict) -> dict:
    """Classify site and compute coefficients."""
    vs30 = None
    n_bar = None
    su_bar = None

    # Compute from layer data if provided
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

    result = {
        "site_class": site_class,
        "vs30_m_per_s": round(vs30, 1) if vs30 is not None else None,
        "n_bar": round(n_bar, 1) if n_bar is not None else None,
        "su_bar_kPa": round(su_bar, 1) if su_bar is not None else None,
    }

    # Compute site coefficients if spectral values provided
    Ss = params.get("Ss")
    S1 = params.get("S1")
    if Ss is not None and S1 is not None:
        sc_result = site_coefficients(site_class, Ss, S1)
        result.update(sc_result.to_dict())

    return result


def _run_seismic_earth_pressure(params: dict) -> dict:
    """Mononobe-Okabe seismic earth pressure analysis."""
    phi = params["phi"]
    delta = params.get("delta", 2.0 / 3.0 * phi)
    kh = params["kh"]
    kv = params.get("kv", 0.0)
    beta = params.get("beta", 0.0)
    i = params.get("i", 0.0)

    KAE = mononobe_okabe_KAE(phi, delta, kh, kv, beta, i)
    KA = rankine_Ka(phi)

    result = {
        "KAE": round(KAE, 4),
        "KA_static": round(KA, 4),
    }

    # Compute KPE if requested
    if params.get("include_passive", False):
        KPE = mononobe_okabe_KPE(phi, delta, kh, kv, beta, i)
        result["KPE"] = round(KPE, 4)

    # Compute pressure resultants if gamma and H given
    gamma = params.get("gamma")
    H = params.get("H")
    if gamma is not None and H is not None:
        pressures = seismic_earth_pressure(gamma, H, KAE, KA)
        result.update(pressures)

    result.update({
        "phi_deg": phi,
        "delta_deg": round(delta, 1),
        "kh": kh,
        "kv": kv,
    })

    return result


def _run_liquefaction_evaluation(params: dict) -> dict:
    """Full liquefaction triggering evaluation."""
    results = evaluate_liquefaction(
        layer_depths=params["depths"],
        layer_N160=params["N160"],
        layer_FC=params["FC"],
        layer_gamma=params["gamma"],
        amax_g=params["amax_g"],
        gwt_depth=params["gwt_depth"],
        M=params.get("magnitude", 7.5),
    )

    n_liq = sum(1 for r in results if r.get("liquefiable", False))
    min_fos = min((r["FOS_liq"] for r in results), default=99.9)

    return {
        "layer_results": results,
        "n_layers_evaluated": len(results),
        "n_liquefiable": n_liq,
        "min_FOS_liq": round(min_fos, 3),
        "amax_g": params["amax_g"],
        "magnitude": params.get("magnitude", 7.5),
        "gwt_depth_m": params["gwt_depth"],
    }


def _run_residual_strength(params: dict) -> dict:
    """Post-liquefaction residual strength."""
    N160cs = params["N160cs"]
    sigma_v_eff = params.get("sigma_v_eff")
    method = params.get("method", "seed_harder")

    Sr = post_liquefaction_strength(N160cs, sigma_v_eff, method)

    result = {
        "Sr_kPa": round(Sr, 1),
        "N160cs": N160cs,
        "method": method,
    }
    if sigma_v_eff is not None:
        result["sigma_v_eff_kPa"] = sigma_v_eff
        result["Sr_ratio"] = round(Sr / sigma_v_eff, 4) if sigma_v_eff > 0 else None

    return result


def _run_csr_crr_check(params: dict) -> dict:
    """Quick CSR/CRR check at a single depth."""
    z = params["depth"]
    N160 = params["N160"]
    FC = params.get("FC", 5.0)
    amax_g = params["amax_g"]
    gwt_depth = params["gwt_depth"]
    gamma = params["gamma"]
    M = params.get("magnitude", 7.5)
    gamma_w = params.get("gamma_w", 9.81)

    sigma_v = gamma * z
    if z <= gwt_depth:
        sigma_v_eff = sigma_v
    else:
        sigma_v_eff = sigma_v - gamma_w * (z - gwt_depth)
    if sigma_v_eff <= 0:
        sigma_v_eff = 1.0

    N160cs = fines_correction(N160, FC)
    CSR = compute_CSR(amax_g, sigma_v, sigma_v_eff, z, M)
    CRR = CRR_from_N160cs(N160cs)
    FOS = CRR / CSR if CSR > 0 else 99.9

    return {
        "depth_m": z,
        "sigma_v_kPa": round(sigma_v, 1),
        "sigma_v_eff_kPa": round(sigma_v_eff, 1),
        "rd": round(stress_reduction_rd(z), 4),
        "CSR": round(CSR, 4),
        "N160": N160,
        "N160cs": round(N160cs, 1),
        "FC_pct": FC,
        "CRR": round(CRR, 4),
        "FOS_liq": round(FOS, 3),
        "liquefiable": FOS < 1.0,
        "MSF": round(magnitude_scaling_factor(M), 3),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
        "brief": "AASHTO/NEHRP site classification with Fpga, Fa, Fv coefficients.",
        "description": (
            "Classifies site as A-F based on Vs30, N-bar, or su-bar. "
            "Optionally computes AASHTO site coefficients Fpga, Fa, Fv "
            "when spectral acceleration values (Ss, S1) are provided."
        ),
        "reference": "AASHTO LRFD Section 3.10.3; NEHRP Provisions (FEMA P-1050)",
        "parameters": {
            "vs30": {"type": "float", "required": False, "description": "Average Vs in top 30m (m/s). Preferred method."},
            "n_bar": {"type": "float", "required": False, "description": "Average SPT N in top 30m."},
            "su_bar": {"type": "float", "required": False, "description": "Average undrained shear strength in top 30m (kPa)."},
            "layer_thicknesses": {"type": "array", "required": False, "description": "Layer thicknesses (m). For computing Vs30 or N-bar from profile data."},
            "layer_vs": {"type": "array", "required": False, "description": "Shear wave velocity per layer (m/s). With layer_thicknesses to compute Vs30."},
            "layer_N": {"type": "array", "required": False, "description": "SPT N per layer. With layer_thicknesses to compute N-bar."},
            "su_layer_thicknesses": {"type": "array", "required": False, "description": "Cohesive layer thicknesses (m)."},
            "layer_su": {"type": "array", "required": False, "description": "Undrained strength per cohesive layer (kPa)."},
            "Ss": {"type": "float", "required": False, "description": "Spectral acceleration at 0.2s (g). Provide with S1 for site coefficients."},
            "S1": {"type": "float", "required": False, "description": "Spectral acceleration at 1.0s (g)."},
        },
        "returns": {
            "site_class": "Site class letter (A-F).",
            "Fpga": "PGA site coefficient.",
            "Fa": "Short-period (0.2s) site coefficient.",
            "Fv": "Long-period (1.0s) site coefficient.",
            "SDS_g": "Design spectral acceleration at short period (Fa*Ss).",
            "SD1_g": "Design spectral acceleration at 1.0s (Fv*S1).",
        },
        "related": {
            "seismic_earth_pressure": "Use site coefficients to determine kh for seismic earth pressure.",
            "liquefaction_evaluation": "Use site coefficients and PGA for liquefaction analysis.",
            "pystrata_agent.eql_site_response": "Detailed 1D site response for site-specific analysis.",
            "hvsrpy_agent.hvsr_analysis": "Field measurement for site period / f0.",
        },
        "typical_workflow": (
            "1. Determine site class from Vs30, SPT, or su data (this method)\n"
            "2. Get Fa, Fv coefficients for design spectrum\n"
            "3. Compute SDS = 2/3 * Fa * Ss, SD1 = 2/3 * Fv * S1\n"
            "4. Check liquefaction if site is susceptible (liquefaction_evaluation)\n"
            "5. Get seismic earth pressures for retaining structures (seismic_earth_pressure)"
        ),
        "common_mistakes": [
            "Providing Vs30 without Ss and S1 — site class is determined, but coefficients Fa/Fv won't be computed.",
            "Using N_bar (average SPT) when Vs30 is available — Vs30 is the preferred method.",
        ],
    },
    "seismic_earth_pressure": {
        "category": "Seismic Earth Pressure",
        "brief": "Mononobe-Okabe seismic active/passive earth pressure coefficients and forces.",
        "description": (
            "Computes KAE (seismic active) and optionally KPE (seismic passive) "
            "earth pressure coefficients using the Mononobe-Okabe method. "
            "With gamma and H, also computes total seismic force, static force, "
            "seismic increment, and application height."
        ),
        "reference": "Mononobe & Matsuo (1929); Okabe (1926); AASHTO LRFD 11.6.5; FHWA GEC-3",
        "parameters": {
            "phi": {"type": "float", "required": True, "description": "Soil friction angle (degrees)."},
            "delta": {"type": "float", "required": False, "description": "Wall-soil friction (degrees). Default 2/3*phi."},
            "kh": {"type": "float", "required": True, "description": "Horizontal seismic coefficient (dimensionless, e.g. 0.2)."},
            "kv": {"type": "float", "required": False, "default": 0.0, "description": "Vertical seismic coefficient."},
            "beta": {"type": "float", "required": False, "default": 0.0, "description": "Wall batter from vertical (degrees)."},
            "i": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope (degrees)."},
            "include_passive": {"type": "bool", "required": False, "default": False, "description": "Also compute KPE (passive)."},
            "gamma": {"type": "float", "required": False, "description": "Backfill unit weight (kN/m3). Provide with H for force resultants."},
            "H": {"type": "float", "required": False, "description": "Wall height (m). Provide with gamma for force resultants."},
        },
        "returns": {
            "KAE": "Seismic active earth pressure coefficient.",
            "KA_static": "Static active (Rankine Ka) for comparison.",
            "KPE": "Seismic passive coefficient (if include_passive=True).",
            "PAE_total_kN_per_m": "Total seismic active force (kN/m).",
            "delta_PAE_kN_per_m": "Seismic increment = PAE - PA_static (kN/m).",
            "height_of_application_m": "Height of seismic increment (0.6*H).",
        },
    },
    "liquefaction_evaluation": {
        "category": "Liquefaction",
        "brief": "SPT-based liquefaction triggering evaluation at multiple depths (Youd et al. 2001).",
        "description": (
            "Evaluates liquefaction potential using the Seed-Idriss simplified "
            "procedure with the Youd et al. (2001) NCEER deterministic curve. "
            "Computes CSR, CRR, and factor of safety at each depth. Applies "
            "fines correction and magnitude scaling."
        ),
        "reference": "Youd et al. (2001), ASCE JGGE; Seed & Idriss (1971)",
        "parameters": {
            "depths": {"type": "array", "required": True, "description": "Evaluation depths — midpoint of each layer (m)."},
            "N160": {"type": "array", "required": True, "description": "Corrected SPT (N1)60 at each depth."},
            "FC": {"type": "array", "required": True, "description": "Fines content at each depth (%)."},
            "gamma": {"type": "array", "required": True, "description": "Total unit weight at each depth (kN/m3)."},
            "amax_g": {"type": "float", "required": True, "description": "Peak ground acceleration (fraction of g)."},
            "gwt_depth": {"type": "float", "required": True, "description": "Groundwater depth (m)."},
            "magnitude": {"type": "float", "required": False, "default": 7.5, "description": "Earthquake moment magnitude."},
        },
        "returns": {
            "layer_results": "Per-layer: depth, N160, N160cs, CSR, CRR, FOS_liq, liquefiable.",
            "n_liquefiable": "Number of liquefiable layers.",
            "min_FOS_liq": "Minimum factor of safety against liquefaction.",
        },
    },
    "residual_strength": {
        "category": "Liquefaction",
        "brief": "Post-liquefaction residual undrained strength estimation.",
        "description": (
            "Estimates residual shear strength after liquefaction using "
            "Seed & Harder (1990) lower-bound or Idriss & Boulanger (2008). "
            "Used for post-earthquake stability analysis."
        ),
        "reference": "Seed & Harder (1990); Idriss & Boulanger (2008)",
        "parameters": {
            "N160cs": {"type": "float", "required": True, "description": "Clean-sand corrected SPT (N1)60cs."},
            "sigma_v_eff": {"type": "float", "required": False, "description": "Pre-earthquake effective vertical stress (kPa). Required for Idriss-Boulanger."},
            "method": {"type": "str", "required": False, "default": "seed_harder", "description": "'seed_harder' or 'idriss_boulanger'."},
        },
        "returns": {
            "Sr_kPa": "Residual undrained strength (kPa).",
            "Sr_ratio": "Sr / sigma_v' (if sigma_v_eff provided).",
        },
    },
    "csr_crr_check": {
        "category": "Liquefaction",
        "brief": "Quick CSR/CRR check at a single depth.",
        "description": (
            "Computes cyclic stress ratio (CSR), cyclic resistance ratio (CRR), "
            "and factor of safety against liquefaction at a single depth. "
            "Includes stress reduction factor rd and magnitude scaling."
        ),
        "reference": "Youd et al. (2001); Seed & Idriss (1971)",
        "parameters": {
            "depth": {"type": "float", "required": True, "description": "Evaluation depth (m)."},
            "N160": {"type": "float", "required": True, "description": "Corrected SPT (N1)60."},
            "FC": {"type": "float", "required": False, "default": 5.0, "description": "Fines content (%)."},
            "gamma": {"type": "float", "required": True, "description": "Total unit weight (kN/m3)."},
            "amax_g": {"type": "float", "required": True, "description": "Peak ground acceleration (g)."},
            "gwt_depth": {"type": "float", "required": True, "description": "Groundwater depth (m)."},
            "magnitude": {"type": "float", "required": False, "default": 7.5, "description": "Earthquake magnitude."},
        },
        "returns": {
            "CSR": "Cyclic stress ratio (adjusted to M=7.5).",
            "CRR": "Cyclic resistance ratio.",
            "FOS_liq": "Factor of safety (CRR/CSR).",
            "liquefiable": "True if FOS < 1.0.",
            "rd": "Stress reduction factor.",
            "MSF": "Magnitude scaling factor.",
        },
    },
}


# ---------------------------------------------------------------------------
# Foundry functions
# ---------------------------------------------------------------------------

@function
def seismic_geotech_agent(method: str, parameters_json: str) -> str:
    """
    Seismic geotechnical analysis toolkit.

    Provides site classification (AASHTO/NEHRP), Mononobe-Okabe seismic
    earth pressures, SPT-based liquefaction triggering evaluation, and
    post-liquefaction residual strength estimation.

    Parameters:
        method: The calculation method name. Use seismic_geotech_list_methods() to see options.
        parameters_json: JSON string of parameters. Use seismic_geotech_describe_method() for details.

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
def seismic_geotech_list_methods(category: str = "") -> str:
    """
    Lists available seismic geotechnical methods.

    Parameters:
        category: Optional filter by category (e.g. 'Liquefaction', 'Site Classification').

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
def seismic_geotech_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a seismic geotechnical method.

    Parameters:
        method: The method name (e.g. 'site_classification', 'liquefaction_evaluation').

    Returns:
        JSON string with parameters, types, ranges, defaults, and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
