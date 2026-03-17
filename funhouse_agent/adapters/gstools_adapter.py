"""GSTools adapter — flat dict -> gstools_agent API -> dict."""

from funhouse_agent.adapters import clean_result


def _run_kriging(params: dict) -> dict:
    from gstools_agent import analyze_kriging, has_gstools
    if not has_gstools():
        return {"error": "gstools is not installed. Install with: pip install gstools"}

    result = analyze_kriging(
        x=params["x"],
        y=params["y"],
        values=params["values"],
        model_type=params.get("model_type", "Gaussian"),
        kriging_type=params.get("kriging_type", "ordinary"),
        grid_x_min=params.get("grid_x_min"),
        grid_x_max=params.get("grid_x_max"),
        grid_y_min=params.get("grid_y_min"),
        grid_y_max=params.get("grid_y_max"),
        n_grid_x=params.get("n_grid_x", 50),
        n_grid_y=params.get("n_grid_y", 50),
        variance=params.get("variance"),
        len_scale=params.get("len_scale"),
        nugget=params.get("nugget", 0.0),
        fit_variogram=params.get("fit_variogram", True),
    )
    return clean_result(result.to_dict())


def _run_variogram(params: dict) -> dict:
    from gstools_agent import analyze_variogram, has_gstools
    if not has_gstools():
        return {"error": "gstools is not installed. Install with: pip install gstools"}

    result = analyze_variogram(
        x=params["x"],
        y=params["y"],
        values=params["values"],
        model_type=params.get("model_type", "Gaussian"),
        n_bins=params.get("n_bins", 10),
        nugget=params.get("nugget", 0.0),
    )
    return clean_result(result.to_dict())


def _run_random_field(params: dict) -> dict:
    from gstools_agent import generate_random_field, has_gstools
    if not has_gstools():
        return {"error": "gstools is not installed. Install with: pip install gstools"}

    result = generate_random_field(
        model_type=params.get("model_type", "Gaussian"),
        variance=params.get("variance", 1.0),
        len_scale=params.get("len_scale", 10.0),
        nugget=params.get("nugget", 0.0),
        mean=params.get("mean", 0.0),
        x_min=params.get("x_min", 0.0),
        x_max=params.get("x_max", 100.0),
        y_min=params.get("y_min", 0.0),
        y_max=params.get("y_max", 100.0),
        n_x=params.get("n_x", 50),
        n_y=params.get("n_y", 50),
        seed=params.get("seed", 42),
    )
    return clean_result(result.to_dict())


METHOD_REGISTRY = {
    "kriging": _run_kriging,
    "variogram": _run_variogram,
    "random_field": _run_random_field,
}

METHOD_INFO = {
    "kriging": {
        "category": "Geostatistics",
        "brief": "Krige soil property values onto a regular grid (ordinary/simple/universal).",
        "parameters": {
            "x": {"type": "array", "required": True, "description": "X-coordinates of measurement points."},
            "y": {"type": "array", "required": True, "description": "Y-coordinates of measurement points."},
            "values": {"type": "array", "required": True, "description": "Measured values at each point."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "description": "Covariance model: Gaussian/Exponential/Matern/Spherical/Linear."},
            "kriging_type": {"type": "str", "required": False, "default": "ordinary", "description": "Kriging type: ordinary/simple/universal."},
            "grid_x_min": {"type": "float", "required": False, "description": "Grid X minimum. Auto from data if omitted."},
            "grid_x_max": {"type": "float", "required": False, "description": "Grid X maximum."},
            "grid_y_min": {"type": "float", "required": False, "description": "Grid Y minimum."},
            "grid_y_max": {"type": "float", "required": False, "description": "Grid Y maximum."},
            "n_grid_x": {"type": "int", "required": False, "default": 50, "description": "Grid points in X."},
            "n_grid_y": {"type": "int", "required": False, "default": 50, "description": "Grid points in Y."},
            "variance": {"type": "float", "required": False, "description": "Covariance model variance. Auto-fit if omitted."},
            "len_scale": {"type": "float", "required": False, "description": "Correlation length. Auto-fit if omitted."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance."},
            "fit_variogram": {"type": "bool", "required": False, "default": True, "description": "Auto-fit variogram parameters."},
        },
        "returns": {
            "n_data": "Number of input data points.",
            "model_type": "Covariance model used.",
            "variance": "Fitted/specified variance.",
            "len_scale": "Fitted/specified correlation length.",
        },
    },
    "variogram": {
        "category": "Geostatistics",
        "brief": "Estimate and fit an empirical variogram from spatial data.",
        "parameters": {
            "x": {"type": "array", "required": True, "description": "X-coordinates of measurement points."},
            "y": {"type": "array", "required": True, "description": "Y-coordinates of measurement points."},
            "values": {"type": "array", "required": True, "description": "Measured values at each point."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "description": "Covariance model to fit."},
            "n_bins": {"type": "int", "required": False, "default": 10, "description": "Number of lag bins."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance."},
        },
        "returns": {
            "variance": "Fitted sill variance.",
            "len_scale": "Fitted correlation length.",
            "r_squared": "Goodness of fit.",
        },
    },
    "random_field": {
        "category": "Geostatistics",
        "brief": "Generate a 2D spatial random field for probabilistic soil property modeling.",
        "parameters": {
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "description": "Covariance model."},
            "variance": {"type": "float", "required": False, "default": 1.0, "description": "Field variance."},
            "len_scale": {"type": "float", "required": False, "default": 10.0, "description": "Correlation length."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance."},
            "mean": {"type": "float", "required": False, "default": 0.0, "description": "Mean value."},
            "x_min": {"type": "float", "required": False, "default": 0.0, "description": "X-axis minimum."},
            "x_max": {"type": "float", "required": False, "default": 100.0, "description": "X-axis maximum."},
            "y_min": {"type": "float", "required": False, "default": 0.0, "description": "Y-axis minimum."},
            "y_max": {"type": "float", "required": False, "default": 100.0, "description": "Y-axis maximum."},
            "n_x": {"type": "int", "required": False, "default": 50, "description": "Grid points in X."},
            "n_y": {"type": "int", "required": False, "default": 50, "description": "Grid points in Y."},
            "seed": {"type": "int", "required": False, "default": 42, "description": "Random seed for reproducibility."},
        },
        "returns": {
            "model_type": "Covariance model used.",
            "field_mean": "Actual mean of generated field.",
            "field_std": "Actual std of generated field.",
        },
    },
}
