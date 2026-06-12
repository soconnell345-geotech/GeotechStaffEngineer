"""Reliability adapter — geotechnical probabilistic engines (native, no
optional deps): FOSM, Rosenblueth PEM, Monte Carlo, FORM + the property-
variability knowledge base (COV guidance, combined COV, Vanmarcke spatial
averaging)."""

import math
import re

from funhouse_agent.adapters import clean_result, require_params

_CONVENTIONS = ["fos", "margin"]
_SAMPLING = ["random", "lhs"]
_SCHEMES = ["full", "multiplicative"]
_VR_MODELS = ["exponential", "simple"]
_CATEGORIES = ["inherent", "site_specific", "total_test", "transformation"]
_DISTS = ["normal", "lognormal", "uniform", "triangular"]

_MATH_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "exp", "sin", "cos", "tan", "pi", "asin",
                 "acos", "atan", "atan2", "sinh", "cosh", "tanh", "log10",
                 "ceil", "floor", "radians", "degrees")
}
_MATH_FUNCS.update({"abs": abs, "min": min, "max": max})


def _compile_g(expr: str, var_names):
    """Compile a limit-state/FOS expression string into g(values_dict).

    Same restricted-eval pattern as the pystra adapter: only the variable
    names and math functions are allowed identifiers.
    """
    if not expr or not str(expr).strip():
        raise ValueError(
            "g_expression is required, e.g. 'R - S' (margin) or "
            "'(c + q*tan(radians(phi)))/tau' (FOS).")
    allowed = set(var_names) | set(_MATH_FUNCS)
    for token in re.findall(r"[a-zA-Z_]\w*", expr):
        if token not in allowed:
            raise ValueError(
                f"Unknown identifier '{token}' in g_expression. Allowed "
                f"variables: {sorted(var_names)}; plus math functions "
                f"like sqrt/log/exp/tan/radians.")
    ns = {"__builtins__": {}}
    ns.update(_MATH_FUNCS)  # lambda resolves names via its __globals__
    fn = eval(f"lambda {', '.join(var_names)}: {expr}", ns)

    def g(values):
        return fn(**{k: values[k] for k in var_names})

    return g


def _prep(params: dict, method: str):
    require_params(params, ["variables", "g_expression"], method=method)
    variables = params["variables"]
    if not isinstance(variables, dict) or not variables:
        raise ValueError(
            f"{method}: 'variables' must be a non-empty dict like "
            f"{{'R': {{'mean': 200, 'cov': 0.1, 'dist': 'lognormal'}}}}.")
    convention = params.get("convention", "fos")
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"{method}: convention must be one of {_CONVENTIONS}.")
    g = _compile_g(params["g_expression"], list(variables.keys()))
    return g, variables, params.get("correlation"), convention


def _run_fosm(params: dict) -> dict:
    from reliability import fosm
    g, variables, corr, conv = _prep(params, "fosm")
    res = fosm(g, variables, correlation=corr, convention=conv)
    return clean_result(res.to_dict())


def _run_pem(params: dict) -> dict:
    from reliability import pem
    g, variables, corr, conv = _prep(params, "pem")
    scheme = params.get("scheme", "full")
    if scheme not in _SCHEMES:
        raise ValueError(f"pem: scheme must be one of {_SCHEMES}.")
    res = pem(g, variables, correlation=corr, convention=conv,
              scheme=scheme)
    return clean_result(res.to_dict())


def _run_monte_carlo(params: dict) -> dict:
    from reliability import monte_carlo
    g, variables, corr, conv = _prep(params, "monte_carlo")
    sampling = params.get("sampling", "random")
    if sampling not in _SAMPLING:
        raise ValueError(
            f"monte_carlo: sampling must be one of {_SAMPLING}.")
    res = monte_carlo(
        g, variables, correlation=corr, convention=conv,
        n=int(params.get("n", 10000)), seed=params.get("seed"),
        sampling=sampling, n_bins=int(params.get("n_bins", 30)))
    return clean_result(res.to_dict())


def _run_form(params: dict) -> dict:
    from reliability import form
    g, variables, corr, conv = _prep(params, "form")
    res = form(g, variables, correlation=corr, convention=conv,
               max_iterations=int(params.get("max_iterations", 100)))
    return clean_result(res.to_dict())


def _run_cov_guidance(params: dict) -> dict:
    from reliability import cov_guidance
    require_params(params, ["property"], method="cov_guidance")
    category = params.get("category")
    if category is not None and category not in _CATEGORIES:
        raise ValueError(
            f"cov_guidance: category must be one of {_CATEGORIES}.")
    rows = cov_guidance(params["property"],
                        soil_type=params.get("soil_type"),
                        test=params.get("test"),
                        category=category)
    return clean_result({
        "property": params["property"],
        "n_entries": len(rows),
        "entries": [r.to_dict() for r in rows],
        "note": "COV values in percent, as published. Divide by 100 for "
                "the engines' cov input.",
    })


def _run_combined_cov(params: dict) -> dict:
    from reliability import combined_cov
    require_params(params, ["cov_inherent"], method="combined_cov")
    total = combined_cov(
        float(params["cov_inherent"]),
        float(params.get("cov_measurement", 0.0)),
        float(params.get("cov_transformation", 0.0)),
        float(params.get("variance_reduction", 1.0)))
    return {
        "cov_total": round(total, 6),
        "equation": "cov_total = sqrt(Gamma2*cov_w^2 + cov_e^2 + cov_t^2) "
                    "(UFC 3-220-20 Eq. 7-5; Phoon & Kulhawy 1999)",
        "inputs": {
            "cov_inherent": float(params["cov_inherent"]),
            "cov_measurement": float(params.get("cov_measurement", 0.0)),
            "cov_transformation":
                float(params.get("cov_transformation", 0.0)),
            "variance_reduction": float(params.get("variance_reduction",
                                                   1.0)),
        },
    }


def _run_variance_reduction(params: dict) -> dict:
    from reliability import variance_reduction
    from reliability.spatial import scale_of_fluctuation_guidance
    require_params(params, ["L"], method="variance_reduction")
    model = params.get("model", "exponential")
    if model not in _VR_MODELS:
        raise ValueError(
            f"variance_reduction: model must be one of {_VR_MODELS}.")
    delta = params.get("delta")
    out = {}
    if delta is None:
        soil_type = params.get("soil_type")
        if soil_type is None:
            raise ValueError(
                "variance_reduction: provide 'delta' (scale of "
                "fluctuation, m) or 'soil_type' to use published "
                "guidance (e.g. 'clay', 'sand').")
        rows = scale_of_fluctuation_guidance(soil_type)
        delta = rows[0].delta_v_avg
        out["delta_source"] = (
            f"vertical average for '{rows[0].soil_type}' from "
            f"{rows[0].source}")
        out["delta_guidance"] = [r.to_dict() for r in rows]
    g2 = variance_reduction(float(params["L"]), float(delta), model=model)
    out.update({
        "gamma_squared": round(g2, 6),
        "gamma": round(math.sqrt(g2), 6),
        "L": float(params["L"]),
        "delta": float(delta),
        "model": model,
        "usage": "Apply Gamma2 to the INHERENT cov only: "
                 "cov_avg = sqrt(Gamma2*cov_w^2 + cov_e^2 + cov_t^2) "
                 "(combined_cov method, variance_reduction param).",
    })
    return clean_result(out)


METHOD_REGISTRY = {
    "fosm": _run_fosm,
    "pem": _run_pem,
    "monte_carlo": _run_monte_carlo,
    "form": _run_form,
    "cov_guidance": _run_cov_guidance,
    "combined_cov": _run_combined_cov,
    "variance_reduction": _run_variance_reduction,
}

_VARIABLES_PARAM = {
    "variables": {
        "type": "object", "required": True,
        "description": "Random variables: {name: {mean, cov or std, dist, "
                       "lower, upper}}. dist one of "
                       f"{_DISTS} (default normal). lower/upper truncate "
                       "normal/lognormal or bound uniform/triangular. "
                       "Example: {'su': {'mean': 50, 'cov': 0.3, 'dist': "
                       "'lognormal'}}."},
    "g_expression": {
        "type": "str", "required": True,
        "description": "Performance function of the variable names. With "
                       "convention='fos' write a FACTOR OF SAFETY "
                       "(failure at g<1), e.g. 'R/S'; with 'margin' a "
                       "margin (failure at g<0), e.g. 'R - S'. Math "
                       "functions allowed (sqrt, log, exp, tan, radians, "
                       "...)."},
    "convention": {
        "type": "str", "required": False, "default": "fos",
        "allowed_values": _CONVENTIONS,
        "description": "'fos': g is a factor of safety (failure g<1, "
                       "lognormal beta reported). 'margin': g = R-S "
                       "(failure g<0)."},
    "correlation": {
        "type": "object", "required": False,
        "description": "Pairwise correlations {'a,b': rho} or full "
                       "matrix [[...]] in variable order. Omit for "
                       "independent variables."},
}

_RETURNS_MOMENTS = {
    "g_mean": "Mean of g.", "g_std": "Std dev of g.",
    "g_cov": "COV of g.",
    "beta_normal": "Normal reliability index ((mu-threshold)/sigma).",
    "beta_lognormal": "Duncan (2000) lognormal index (fos convention).",
    "pf_normal": "PHI(-beta_normal).",
    "pf_lognormal": "PHI(-beta_lognormal).",
}

METHOD_INFO = {
    "fosm": {
        "category": "Reliability",
        "brief": "FOSM / Taylor-series reliability (Duncan 2000, UFC "
                 "3-220-20): central differences at +/-1 sigma -> moments "
                 "of g, beta (normal+lognormal), pf, and the per-variable "
                 "variance-contribution table (which variable matters). "
                 "2n+1 g-evaluations; supports correlation.",
        "parameters": _VARIABLES_PARAM,
        "returns": {**_RETURNS_MOMENTS,
                    "variance_contributions_pct":
                        "Per-variable share of Var[g] (%).",
                    "variable_deltas": "g(+1sigma)-g(-1sigma) per variable."},
    },
    "pem": {
        "category": "Reliability",
        "brief": "Rosenblueth point-estimate method (UFC 3-220-20 Eqs. "
                 "7-13/7-14): 2^n evaluations at mu+/-sigma (correlation-"
                 "adjusted weights), or the 2n+1 'multiplicative' reduced "
                 "scheme for many variables.",
        "parameters": {
            **_VARIABLES_PARAM,
            "scheme": {"type": "str", "required": False, "default": "full",
                       "allowed_values": _SCHEMES,
                       "description": "'full' = 2^n points; "
                                      "'multiplicative' = Rosenblueth 2n+1 "
                                      "(uncorrelated only)."},
        },
        "returns": {**_RETURNS_MOMENTS, "n_points": "g evaluations used."},
    },
    "monte_carlo": {
        "category": "Reliability",
        "brief": "Monte Carlo simulation: seeded numpy sampling (optional "
                 "Latin Hypercube), correlation via Cholesky in normal "
                 "space, empirical pf with 95% CI + convergence trace, "
                 "histogram, lognormal-fit beta.",
        "parameters": {
            **_VARIABLES_PARAM,
            "n": {"type": "int", "required": False, "default": 10000,
                  "description": "Realizations. Aim n >= 10/pf."},
            "seed": {"type": "int", "required": False,
                     "description": "RNG seed for reproducibility."},
            "sampling": {"type": "str", "required": False,
                         "default": "random", "allowed_values": _SAMPLING,
                         "description": "'lhs' = Latin Hypercube "
                                        "(variance reduction)."},
            "n_bins": {"type": "int", "required": False, "default": 30,
                       "description": "Histogram bins."},
        },
        "returns": {"pf": "Empirical probability of failure.",
                    "pf_ci95": "95% binomial confidence interval on pf.",
                    "convergence": "[[n, running pf], ...] trace.",
                    "histogram_bins": "Bin edges.",
                    "histogram_counts": "Counts.",
                    **_RETURNS_MOMENTS},
    },
    "form": {
        "category": "Reliability",
        "brief": "Native FORM (Hasofer-Lind / Rackwitz-Fiessler): HL-RF "
                 "iteration with equivalent-normal transformation of "
                 "non-normal variables; exact for the linear/lognormal "
                 "cases. Returns beta, pf, design point x*, and alpha "
                 "sensitivity vector (alpha^2 = uncertainty share).",
        "parameters": {
            **_VARIABLES_PARAM,
            "max_iterations": {"type": "int", "required": False,
                               "default": 100,
                               "description": "HL-RF iteration cap."},
        },
        "returns": {"beta": "Reliability index (geometric, HL).",
                    "pf": "PHI(-beta).",
                    "design_point": "Most probable failure point x*.",
                    "alphas": "Sensitivity vector (negative = resistance).",
                    "alpha_squared_pct": "Per-variable uncertainty share.",
                    "converged": "HL-RF convergence flag."},
    },
    "cov_guidance": {
        "category": "Property Variability",
        "brief": "Published COV ranges for soil/rock properties with full "
                 "provenance: Duncan (2000) Table 1, ISSMGE-TC304 (2021) "
                 "site-specific statistics (clay/sand/rock), Phoon & "
                 "Kulhawy (1999) transformation uncertainty (via UFC "
                 "3-220-20). Use to pick a defensible cov for the engines.",
        "parameters": {
            "property": {"type": "str", "required": True,
                         "description": "Property key or alias: phi/"
                                        "friction_angle, su, gamma/"
                                        "unit_weight, N/spt, qc/cpt, Cc, "
                                        "w, LL, PL, PI, OCR, K0, "
                                        "sigma_ci/ucs, RQD, GSI, ..."},
            "soil_type": {"type": "str", "required": False,
                          "description": "Filter: 'clay', 'sand', 'rock'."},
            "test": {"type": "str", "required": False,
                     "description": "Filter by test, e.g. 'SPT', 'CPT', "
                                    "'VST'."},
            "category": {"type": "str", "required": False,
                         "allowed_values": _CATEGORIES,
                         "description": "Uncertainty component."},
        },
        "returns": {"entries": "[{property, label, cov_min_pct, "
                               "cov_max_pct, cov_mean_pct, category, "
                               "soil_type, test, source}]."},
    },
    "combined_cov": {
        "category": "Property Variability",
        "brief": "Combine inherent + measurement + transformation COVs "
                 "(UFC 3-220-20 Eq. 7-5), optionally with Vanmarcke "
                 "spatial-averaging variance reduction on the inherent "
                 "part. All COVs as fractions (0.3 = 30%).",
        "parameters": {
            "cov_inherent": {"type": "float", "required": True,
                             "description": "Inherent (spatial) COV, "
                                            "fraction."},
            "cov_measurement": {"type": "float", "required": False,
                                "default": 0.0,
                                "description": "Measurement-error COV."},
            "cov_transformation": {"type": "float", "required": False,
                                   "default": 0.0,
                                   "description": "Transformation-model "
                                                  "COV."},
            "variance_reduction": {"type": "float", "required": False,
                                   "default": 1.0,
                                   "description": "Gamma^2 from the "
                                                  "variance_reduction "
                                                  "method (1 = point "
                                                  "property)."},
        },
        "returns": {"cov_total": "Combined COV (fraction)."},
    },
    "variance_reduction": {
        "category": "Property Variability",
        "brief": "Vanmarcke variance reduction Gamma^2(L/delta) for "
                 "spatial averaging over a failure surface/pile length. "
                 "Give delta directly or a soil_type to use published "
                 "scale-of-fluctuation guidance (Cami et al. 2020 / "
                 "ISSMGE-TC304 2021).",
        "parameters": {
            "L": {"type": "float", "required": True,
                  "description": "Averaging length (m)."},
            "delta": {"type": "float", "required": False,
                      "description": "Scale of fluctuation (m). Omit to "
                                     "look up by soil_type."},
            "soil_type": {"type": "str", "required": False,
                          "description": "e.g. 'clay', 'sand', 'silt', "
                                         "'soft clay' — used when delta "
                                         "is omitted (vertical average)."},
            "model": {"type": "str", "required": False,
                      "default": "exponential",
                      "allowed_values": _VR_MODELS,
                      "description": "'exponential' (exact Markov ACF) or "
                                     "'simple' (delta/L)."},
        },
        "returns": {"gamma_squared": "Variance reduction factor.",
                    "gamma": "Std-dev reduction factor.",
                    "delta_guidance": "Published ranges when looked up."},
    },
}
