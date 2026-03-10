"""SALib adapter — Sobol and Morris sensitivity analysis."""

from funhouse_agent.adapters import clean_result


def _run_sobol_sample(params: dict) -> dict:
    from salib_agent import sobol_sample, has_salib

    if not has_salib():
        return {"error": "SALib is not installed. Install via: pip install SALib"}

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
    "sobol_sample": _run_sobol_sample,
    "sobol_analyze": _run_sobol_analyze,
    "morris_sample": _run_morris_sample,
    "morris_analyze": _run_morris_analyze,
}

METHOD_INFO = {
    "sobol_sample": {
        "category": "Sensitivity Analysis",
        "brief": "Generate Sobol quasi-random sample matrix for variance-based sensitivity analysis.",
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
        "brief": "Generate Morris one-at-a-time (OAT) sample matrix for screening.",
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
