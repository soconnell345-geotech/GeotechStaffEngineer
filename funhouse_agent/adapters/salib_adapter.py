"""SALib adapter — Sobol and Morris sensitivity analysis.

``sobol_analysis`` / ``morris_analysis`` do the whole job in one call: sample,
evaluate the model at every point, analyze, return the indices only (live
smoke G9: an agent cannot loop over the 1,280-row matrix ``sobol_sample``
hands back). The model is either a safe arithmetic ``expression`` of the
variables or a ``model`` spec that runs one of the calculation modules per
sample point.
"""

import copy
import math
import time

from funhouse_agent.adapters import (
    apply_aliases, clean_result, mark_required, reject_unknown_params,
    require_params,
)


# ---------------------------------------------------------------------------
# One-call analyses: the model as an expression or a dispatch spec
# ---------------------------------------------------------------------------

#: Modules a ``model`` spec may run: calculation modules that write no files
#: and do not call back into a probabilistic engine (no salib / reliability /
#: pystra recursion, no writers, no document or reference tools).
_MODEL_MODULES = frozenset({
    "bearing_capacity", "settlement", "slope_stability", "seismic_geotech",
    "liquefaction", "retaining_walls", "axial_pile", "drilled_shaft",
    "sheet_pile", "lateral_pile", "pile_group", "ground_improvement",
    "wave_equation", "downdrag", "soe", "pavement_design", "fem2d",
    "section_props", "concrete_props", "pynite", "dm7",
})

#: Parameters that name a file to read or write: refused inside a spec, so
#: thousands of evaluations can never touch the file system.
_MODEL_FILE_PARAMS = ("output_path", "output_dir", "file_path",
                      "attachment_key")

_MODEL_SPEC_KEYS = ("agent_name", "method", "parameters", "variable_map",
                    "output")

_TIME_BUDGET_DEFAULT_S = 120.0
_TIME_BUDGET_MAX_S = 600.0

_ONE_CALL_COMMON = ("var_names", "bounds", "expression", "model", "seed",
                    "max_evaluations", "time_budget_s")


def _walk(obj, part):
    """One step of a dotted path: a dict key or a list index."""
    if isinstance(obj, dict):
        if part not in obj:
            raise KeyError(part)
        return obj[part]
    if isinstance(obj, list):
        try:
            return obj[int(part)]
        except (ValueError, IndexError):
            raise KeyError(part) from None
    raise KeyError(part)


def _get_path(obj, path):
    for part in str(path).split("."):
        obj = _walk(obj, part)
    return obj


def _set_path(obj, path, value):
    parts = str(path).split(".")
    for part in parts[:-1]:
        obj = _walk(obj, part)
    last = parts[-1]
    if isinstance(obj, list):
        try:
            obj[int(last)] = value
        except (ValueError, IndexError):
            raise KeyError(last) from None
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        raise KeyError(last)


def _numeric_keys(result, prefix="", depth=0):
    """Dotted paths of the numeric values in a result (for error hints)."""
    out = []
    if not isinstance(result, dict) or depth > 1:
        return out
    for k, v in result.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.append(prefix + k)
        elif isinstance(v, dict):
            out += _numeric_keys(v, prefix + k + ".", depth + 1)
    return out


def _n_evaluations(method, n_vars, params):
    if method == "sobol_analysis":
        n = int(params.get("n_samples", 256))
        second = bool(params.get("calc_second_order", False))
        return n * (2 * n_vars + 2) if second else n * (n_vars + 2)
    return int(params.get("n_trajectories", 20)) * (n_vars + 1)


def _model_from_spec(spec, var_names, bounds, n_evals, time_budget_s, method):
    """A model ``f(values: dict) -> float`` that runs a calculation module.

    Checked before any sampling: the module is on the allowlist, no file
    parameter appears, every variable lands on a parameter, and one trial run
    at the centre of the bounds returns a number at ``output``. The trial's
    run time sets the cost estimate; a design projected past the time budget
    is refused, and the run stops if it overruns anyway.
    """
    from funhouse_agent.dispatch import _canonical_agent_name, call_agent

    if not isinstance(spec, dict):
        raise ValueError(
            f"{method}: model must be a dict {{agent_name, method, "
            f"parameters, output, variable_map (optional)}}, e.g. "
            f"{{'agent_name': 'bearing_capacity', 'method': "
            f"'bearing_capacity_analysis', 'parameters': {{'width': 2, "
            f"'depth': 1, 'unit_weight': 18}}, 'output': 'q_ultimate_kPa', "
            f"'variable_map': {{'phi': 'friction_angle', 'c': 'cohesion'}}}}.")
    spec = apply_aliases(spec, {"module": "agent_name", "agent": "agent_name",
                                "params": "parameters",
                                "fixed_parameters": "parameters",
                                "output_key": "output"})
    reject_unknown_params(spec, _MODEL_SPEC_KEYS, method=f"{method} model")
    require_params(spec, ["agent_name", "method", "output"],
                   method=f"{method} model", valid=_MODEL_SPEC_KEYS)
    agent = _canonical_agent_name(str(spec["agent_name"]))
    if agent not in _MODEL_MODULES:
        raise ValueError(
            f"{method}: model.agent_name '{spec['agent_name']}' is not "
            f"allowed. A model spec runs calculation modules only: "
            f"{sorted(_MODEL_MODULES)}. For anything else, write the model "
            f"as an 'expression' of the variables.")
    target_method = str(spec["method"])
    base = spec.get("parameters") or {}
    if not isinstance(base, dict):
        raise ValueError(f"{method}: model.parameters must be a dict.")
    bad = [k for k in _MODEL_FILE_PARAMS if k in base]
    if bad:
        raise ValueError(
            f"{method}: model.parameters may not name files ({bad}): the "
            f"model runs once per sample point.")
    vmap = spec.get("variable_map") or {}
    if not isinstance(vmap, dict):
        raise ValueError(f"{method}: model.variable_map must be a dict "
                         f"{{variable: parameter}}.")
    unknown = sorted(set(vmap) - set(var_names))
    if unknown:
        raise ValueError(
            f"{method}: model.variable_map names {unknown}, which are not in "
            f"var_names {list(var_names)}.")
    targets = {v: str(vmap.get(v, v)) for v in var_names}
    if any(t.split(".")[0] in _MODEL_FILE_PARAMS for t in targets.values()):
        raise ValueError(f"{method}: a variable may not map to a file "
                         f"parameter.")
    output = str(spec["output"])
    base = copy.deepcopy(base)
    label = f"{agent}.{target_method}"

    def run(values):
        p = copy.deepcopy(base)
        for v, target in targets.items():
            try:
                _set_path(p, target, values[v])
            except KeyError as exc:
                raise ValueError(
                    f"variable '{v}' maps to '{target}', but {exc} is not in "
                    f"model.parameters (a dotted path into a list or dict "
                    f"must already exist there).") from None
        r = call_agent(agent, target_method, p)
        if not isinstance(r, dict):
            raise ValueError(f"{label} returned no result dict")
        if "error" in r:
            raise ValueError(f"{label}: {r['error']}")
        try:
            val = _get_path(r, output)
        except KeyError:
            raise ValueError(
                f"{label} has no output '{output}'. Numeric outputs: "
                f"{_numeric_keys(r)}.") from None
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"{label} output '{output}' is not a number ({val!r}).")
        return float(val)

    # Trial run at the centre of the bounds: a broken spec fails here, once,
    # with the module's own message, and the run time prices the design.
    centre = {v: (float(b[0]) + float(b[1])) / 2.0
              for v, b in zip(var_names, bounds)}
    t0 = time.perf_counter()
    try:
        run(centre)
    except ValueError as exc:
        raise ValueError(
            f"{method}: the model failed its trial run at the centre of the "
            f"bounds {centre}: {exc}") from None
    per_call = time.perf_counter() - t0
    projected = per_call * n_evals
    if projected > time_budget_s:
        raise ValueError(
            f"{method}: one {label} run takes {per_call:.3f} s, so "
            f"{n_evals} evaluations would take about {projected:.0f} s, "
            f"above the {time_budget_s:.0f} s time budget. Use fewer "
            f"samples (n_samples / n_trajectories), screen with "
            f"morris_analysis first, or raise time_budget_s (max "
            f"{_TIME_BUDGET_MAX_S:.0f}).")

    started = [None]

    def model(values):
        now = time.perf_counter()
        if started[0] is None:
            started[0] = now
        elif now - started[0] > 1.5 * time_budget_s:
            raise ValueError(
                f"the run passed 1.5 x the {time_budget_s:.0f} s time "
                f"budget; stopped")
        return run(values)

    return model, {"agent_name": agent, "method": target_method,
                   "output": output, "variable_map": targets,
                   "seconds_per_evaluation": round(per_call, 5)}


def _nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, list):
        return [_nan_to_none(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    return obj


def _one_call(params: dict, method: str, extra_valid) -> dict:
    from salib_agent import has_salib
    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

    params = apply_aliases(params, {"g_expression": "expression",
                                    "model_spec": "model"})
    valid = _ONE_CALL_COMMON + tuple(extra_valid)
    reject_unknown_params(params, valid, method=method)
    require_params(params, ["var_names", "bounds"], method=method,
                   valid=valid)
    var_names = list(params["var_names"])
    bounds = params["bounds"]
    expression = params.get("expression")
    spec = params.get("model")
    if (expression is None) == (spec is None):
        raise ValueError(
            f"{method}: give exactly one of 'expression' (an arithmetic "
            f"expression of the variables, e.g. '(c + 18*z*tan(radians(phi)))"
            f"/40') or 'model' (a calculation-module spec {{agent_name, "
            f"method, parameters, output, variable_map}}).")
    if len(var_names) != len(bounds or []):
        raise ValueError(
            f"{method}: var_names ({len(var_names)}) and bounds "
            f"({len(bounds or [])}) must have the same length.")

    from salib_agent.sensitivity import DEFAULT_MAX_EVALUATIONS
    kwargs = {"seed": params.get("seed", 42),
              "max_evaluations": int(params.get("max_evaluations",
                                                DEFAULT_MAX_EVALUATIONS))}
    if method == "sobol_analysis":
        kwargs["n_samples"] = int(params.get("n_samples", 256))
        kwargs["calc_second_order"] = bool(params.get("calc_second_order",
                                                      False))
    else:
        kwargs["n_trajectories"] = int(params.get("n_trajectories", 20))
        kwargs["num_levels"] = int(params.get("num_levels", 4))
    n_evals = _n_evaluations(method, len(var_names), kwargs)
    model_info = {"kind": "expression", "expression": expression}
    t0 = time.perf_counter()
    if spec is not None:
        budget = float(params.get("time_budget_s", _TIME_BUDGET_DEFAULT_S))
        if not 0 < budget <= _TIME_BUDGET_MAX_S:
            raise ValueError(f"{method}: time_budget_s must be in "
                             f"(0, {_TIME_BUDGET_MAX_S:.0f}].")
        if n_evals > kwargs["max_evaluations"]:
            raise ValueError(
                f"{method}: the design needs {n_evals} model evaluations, "
                f"above max_evaluations ({kwargs['max_evaluations']}). Use "
                f"fewer samples.")
        model, model_info = _model_from_spec(
            spec, var_names, bounds, n_evals, budget, method)
        model_info = {"kind": "module", **model_info}
        kwargs["model"] = model
    else:
        kwargs["expression"] = expression

    if method == "sobol_analysis":
        from salib_agent import sobol_analysis
        result = sobol_analysis(var_names, bounds, **kwargs)
    else:
        from salib_agent import morris_analysis
        result = morris_analysis(var_names, bounds, **kwargs)
    out = clean_result(_nan_to_none(result.to_dict()))
    out["model_evaluations"] = n_evals
    out["model"] = model_info
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return out


def _run_sobol_analysis(params: dict) -> dict:
    return _one_call(params, "sobol_analysis",
                     ("n_samples", "calc_second_order"))


def _run_morris_analysis(params: dict) -> dict:
    return _one_call(params, "morris_analysis",
                     ("n_trajectories", "num_levels"))


# ---------------------------------------------------------------------------
# Two-step methods: sample -> the caller evaluates -> analyze
# ---------------------------------------------------------------------------


def _run_sobol_sample(params: dict) -> dict:
    from salib_agent import sobol_sample, has_salib

    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

    require_params(params, ["var_names", "bounds"], method="sobol_sample")
    sample_matrix = sobol_sample(
        var_names=params["var_names"],
        bounds=params["bounds"],
        n_samples=params.get("n_samples", 1024),
        calc_second_order=params.get("calc_second_order", True),
        seed=params.get("seed", 42),
    )
    return clean_result({
        "sample_matrix": sample_matrix.tolist(),
        "n_rows": sample_matrix.shape[0],
        "n_vars": sample_matrix.shape[1],
        "var_names": list(params["var_names"]),
    })


def _run_sobol_analyze(params: dict) -> dict:
    from salib_agent import sobol_analyze, has_salib

    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

    require_params(params, ["var_names", "bounds", "Y"], method="sobol_analyze")
    result = sobol_analyze(
        var_names=params["var_names"],
        bounds=params["bounds"],
        Y=params["Y"],
        n_samples=params.get("n_samples", 1024),
        calc_second_order=params.get("calc_second_order", True),
        seed=params.get("seed", 42),
    )
    return clean_result(result.to_dict())


def _run_morris_sample(params: dict) -> dict:
    from salib_agent import morris_sample, has_salib

    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

    require_params(params, ["var_names", "bounds"], method="morris_sample")
    sample_matrix = morris_sample(
        var_names=params["var_names"],
        bounds=params["bounds"],
        n_trajectories=params.get("n_trajectories", 20),
        num_levels=params.get("num_levels", 4),
        seed=params.get("seed", 42),
    )
    return clean_result({
        "sample_matrix": sample_matrix.tolist(),
        "n_rows": sample_matrix.shape[0],
        "n_vars": sample_matrix.shape[1],
        "var_names": list(params["var_names"]),
    })


def _run_morris_analyze(params: dict) -> dict:
    from salib_agent import morris_analyze, has_salib

    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

    require_params(params, ["var_names", "bounds", "X", "Y"],
                   method="morris_analyze")
    result = morris_analyze(
        var_names=params["var_names"],
        bounds=params["bounds"],
        X=params["X"],
        Y=params["Y"],
        n_trajectories=params.get("n_trajectories", 20),
        num_levels=params.get("num_levels", 4),
        seed=params.get("seed", 42),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "sobol_analysis": _run_sobol_analysis,
    "morris_analysis": _run_morris_analysis,
    "sobol_sample": _run_sobol_sample,
    "sobol_analyze": _run_sobol_analyze,
    "morris_sample": _run_morris_sample,
    "morris_analyze": _run_morris_analyze,
}

_EXPRESSION_DOC = (
    "The model as a safe arithmetic expression of the variables (the "
    "reliability engines' g_expression rules): +, -, *, /, **, %, "
    "comparisons, 'a if cond else b', and sqrt/log/exp/sin/cos/tan/asin/"
    "acos/atan/atan2/sinh/cosh/tanh/log10/ceil/floor/radians/degrees/abs/"
    "min/max/pi; scientific notation (1e-3) is fine. E.g. "
    "'(c + 18*z*tan(radians(phi)))/40'. Give this OR model (exactly one).")

_MODEL_DOC = (
    "The model as a calculation-module run per sample point, instead of "
    "expression: {agent_name, method, parameters (the fixed inputs), "
    "output (the result key to analyse; a dotted path such as "
    "'results.fos' reaches into a nested result), variable_map (optional "
    "{variable: parameter}; default: each variable sets the parameter of "
    "the same name; a dotted path such as 'layers.0.phi' sets a nested "
    "value that already exists in parameters)}. E.g. {'agent_name': "
    "'bearing_capacity', 'method': 'bearing_capacity_analysis', "
    "'parameters': {'width': 2, 'depth': 1, 'unit_weight': 18}, 'output': "
    "'q_ultimate_kPa', 'variable_map': {'phi': 'friction_angle', 'c': "
    "'cohesion'}}. Calculation modules only (" + ", ".join(
        sorted(_MODEL_MODULES)) + "); no file parameters. One trial run "
    "at the centre of the bounds checks the spec and prices the design "
    "against time_budget_s. Give this OR expression (exactly one).")

_ONE_CALL_RETURNS = {
    "model_evaluations": "Number of model evaluations run.",
    "model": "What was evaluated: {kind: 'expression', expression} or "
             "{kind: 'module', agent_name, method, output, variable_map, "
             "seconds_per_evaluation}.",
    "output_mean": "Mean of the model output over the sample.",
    "output_std": "Standard deviation of the model output over the sample.",
    "elapsed_s": "Wall time of the run (s).",
}

METHOD_INFO = {
    "sobol_analysis": {
        "category": "Sensitivity Analysis",
        "brief": "ONE CALL Sobol variance-based sensitivity: samples, evaluates the model (an expression or a calculation-module spec) at every point, returns S1/ST indices only. Use this, not sobol_sample + sobol_analyze.",
        "parameters": {
            "var_names": {"type": "array", "required": True,
                          "description": "Variable names (at least 2); the names the expression or variable_map uses."},
            "bounds": {"type": "array", "required": True,
                       "description": "[min, max] for each variable, in var_names order (uniform over the range)."},
            "expression": {"type": "str", "required": False,
                           "description": _EXPRESSION_DOC},
            "model": {"type": "dict", "required": False,
                      "description": _MODEL_DOC},
            "n_samples": {"type": "int", "required": False, "default": 256,
                          "description": "Base sample size N (a power of 2). Evaluations = N*(D+2), or N*(2D+2) with calc_second_order."},
            "calc_second_order": {"type": "bool", "required": False,
                                  "default": False,
                                  "description": "Also compute second-order interaction indices S2 (doubles the evaluations)."},
            "seed": {"type": "int", "required": False, "default": 42,
                     "description": "Random seed."},
            "max_evaluations": {"type": "int", "required": False,
                                "default": 20000,
                                "description": "Refuse a design needing more model evaluations than this."},
            "time_budget_s": {"type": "float", "required": False,
                              "default": _TIME_BUDGET_DEFAULT_S,
                              "description": f"model only: refuse a design whose projected run time (trial run x evaluations) exceeds this, in seconds (max {_TIME_BUDGET_MAX_S:.0f})."},
        },
        "returns": {
            "var_names": "Variable names.",
            "n_vars": "Number of variables.",
            "n_samples": "Number of model outputs analysed.",
            "S1": "First-order Sobol indices (var_names order).",
            "S1_conf": "95% confidence intervals for S1.",
            "ST": "Total-order Sobol indices.",
            "ST_conf": "95% confidence intervals for ST.",
            "S2": "Second-order indices (calc_second_order only; null below the diagonal).",
            **_ONE_CALL_RETURNS,
        },
    },
    "morris_analysis": {
        "category": "Sensitivity Analysis",
        "brief": "ONE CALL Morris elementary-effects screening: samples, evaluates the model (an expression or a calculation-module spec) at every point, returns mu*/sigma only. Cheaper than Sobol: evaluations = n_trajectories*(D+1).",
        "parameters": {
            "var_names": {"type": "array", "required": True,
                          "description": "Variable names (at least 2); the names the expression or variable_map uses."},
            "bounds": {"type": "array", "required": True,
                       "description": "[min, max] for each variable, in var_names order."},
            "expression": {"type": "str", "required": False,
                           "description": _EXPRESSION_DOC},
            "model": {"type": "dict", "required": False,
                      "description": _MODEL_DOC},
            "n_trajectories": {"type": "int", "required": False,
                               "default": 20,
                               "description": "Number of Morris trajectories."},
            "num_levels": {"type": "int", "required": False, "default": 4,
                           "description": "Number of grid levels."},
            "seed": {"type": "int", "required": False, "default": 42,
                     "description": "Random seed."},
            "max_evaluations": {"type": "int", "required": False,
                                "default": 20000,
                                "description": "Refuse a design needing more model evaluations than this."},
            "time_budget_s": {"type": "float", "required": False,
                              "default": _TIME_BUDGET_DEFAULT_S,
                              "description": f"model only: refuse a design whose projected run time (trial run x evaluations) exceeds this, in seconds (max {_TIME_BUDGET_MAX_S:.0f})."},
        },
        "returns": {
            "var_names": "Variable names.",
            "n_vars": "Number of variables.",
            "n_trajectories": "Number of trajectories.",
            "mu_star": "Mean absolute elementary effect per variable (importance).",
            "sigma": "Standard deviation of the elementary effects (nonlinearity / interactions).",
            "mu_star_conf": "95% confidence intervals for mu*.",
            **_ONE_CALL_RETURNS,
        },
    },
    "sobol_sample": {
        "category": "Sensitivity Analysis",
        "brief": "Generate Sobol quasi-random sample matrix for variance-based sensitivity analysis (two-step: YOU evaluate the model on every row, then sobol_analyze). Prefer sobol_analysis, which does it in one call.",
        "parameters": {
            "var_names": {
                "type": "array",
                "brief": "List of variable names (at least 2).",
            },
            "bounds": {
                "type": "array",
                "brief": "List of [min, max] bounds for each variable.",
            },
            "n_samples": {
                "type": "int",
                "brief": "Base number of Sobol samples. Total rows = N*(2D+2) for second-order.",
                "default": 1024,
            },
            "calc_second_order": {
                "type": "bool",
                "brief": "Include second-order interaction indices.",
                "default": True,
            },
            "seed": {
                "type": "int",
                "brief": "Random seed for reproducibility.",
                "default": 42,
            },
        },
        "returns": {
            "sample_matrix": "Sample matrix (list of lists), each row is one sample point.",
            "n_rows": "Total number of sample rows.",
            "n_vars": "Number of variables.",
            "var_names": "Variable names.",
        },
    },
    "sobol_analyze": {
        "category": "Sensitivity Analysis",
        "brief": "Compute Sobol first-order and total-order sensitivity indices from model output.",
        "parameters": {
            "var_names": {
                "type": "array",
                "brief": "List of variable names.",
            },
            "bounds": {
                "type": "array",
                "brief": "List of [min, max] bounds for each variable.",
            },
            "Y": {
                "type": "array",
                "brief": "Model output array, one value per sample point from sobol_sample().",
            },
            "n_samples": {
                "type": "int",
                "brief": "Base number of samples used in sobol_sample().",
                "default": 1024,
            },
            "calc_second_order": {
                "type": "bool",
                "brief": "Whether second-order indices were computed.",
                "default": True,
            },
            "seed": {
                "type": "int",
                "brief": "Random seed for resampling.",
                "default": 42,
            },
        },
        "returns": {
            "n_samples": "Number of model evaluations.",
            "n_vars": "Number of variables.",
            "var_names": "Variable names.",
            "S1": "First-order Sobol indices.",
            "S1_conf": "95% confidence intervals for S1.",
            "ST": "Total-order Sobol indices.",
            "ST_conf": "95% confidence intervals for ST.",
            "S2": "Second-order interaction indices (if computed).",
        },
    },
    "morris_sample": {
        "category": "Sensitivity Analysis",
        "brief": "Generate Morris one-at-a-time (OAT) sample matrix for screening (two-step: YOU evaluate the model on every row, then morris_analyze). Prefer morris_analysis, which does it in one call.",
        "parameters": {
            "var_names": {
                "type": "array",
                "brief": "List of variable names (at least 2).",
            },
            "bounds": {
                "type": "array",
                "brief": "List of [min, max] bounds for each variable.",
            },
            "n_trajectories": {
                "type": "int",
                "brief": "Number of Morris trajectories.",
                "default": 20,
            },
            "num_levels": {
                "type": "int",
                "brief": "Number of grid levels.",
                "default": 4,
            },
            "seed": {
                "type": "int",
                "brief": "Random seed for reproducibility.",
                "default": 42,
            },
        },
        "returns": {
            "sample_matrix": "Sample matrix (list of lists).",
            "n_rows": "Total number of sample rows (n_trajectories * (n_vars + 1)).",
            "n_vars": "Number of variables.",
            "var_names": "Variable names.",
        },
    },
    "morris_analyze": {
        "category": "Sensitivity Analysis",
        "brief": "Compute Morris elementary effects (mu*, sigma) for parameter screening.",
        "parameters": {
            "var_names": {
                "type": "array",
                "brief": "List of variable names.",
            },
            "bounds": {
                "type": "array",
                "brief": "List of [min, max] bounds for each variable.",
            },
            "X": {
                "type": "array",
                "brief": "Sample matrix from morris_sample().",
            },
            "Y": {
                "type": "array",
                "brief": "Model output array, one value per sample row.",
            },
            "n_trajectories": {
                "type": "int",
                "brief": "Number of trajectories used in morris_sample().",
                "default": 20,
            },
            "num_levels": {
                "type": "int",
                "brief": "Number of grid levels used in morris_sample().",
                "default": 4,
            },
            "seed": {
                "type": "int",
                "brief": "Random seed.",
                "default": 42,
            },
        },
        "returns": {
            "n_trajectories": "Number of trajectories.",
            "n_vars": "Number of variables.",
            "var_names": "Variable names.",
            "mu_star": "Mean of absolute elementary effects (importance).",
            "sigma": "Standard deviation of elementary effects (nonlinearity/interactions).",
            "mu_star_conf": "95% confidence intervals for mu*.",
        },
    },
}


# One METHOD_INFO style everywhere: explicit required flags matching the code
# above (live smoke wave 1, G7). The one-call methods need exactly one of
# expression / model, which no single parameter carries alone.
mark_required(METHOD_INFO, {
    "sobol_analysis": ["var_names", "bounds"],
    "morris_analysis": ["var_names", "bounds"],
    "sobol_sample": ["var_names", "bounds"],
    "sobol_analyze": ["var_names", "bounds", "Y"],
    "morris_sample": ["var_names", "bounds"],
    "morris_analyze": ["var_names", "bounds", "X", "Y"],
})
