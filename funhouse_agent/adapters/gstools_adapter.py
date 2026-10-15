"""GSTools adapter — flat dict -> gstools_agent API -> dict.

Four methods, told apart by what they answer:

* ``kriging`` — ONE smoothed best estimate (the mean surface) and its
  kriging variance. It understates the spread and cannot give the
  probability of a total over an area.
* ``variogram`` — the empirical variogram and a fitted covariance model.
* ``random_field`` — an UNconditioned random field: it does not honour data.
* ``conditional_simulation`` — many realizations that honour the data, each
  integrated over plan regions: probabilities of totals (volume above or
  below a level, area, any encounter) with percentiles and exceedance.

Context economy: a kriging grid above ~2,500 points and every array of a
conditional simulation go to CSV files in the conversation's working folder;
the JSON carries their paths and summary statistics, never whole grids (a
60 x 120 grid once went to scratch and the agent could not use it).
"""

import os
from datetime import datetime

from funhouse_agent.adapters import (
    apply_aliases, clean_result, reject_unknown_params, require_params,
    write_plotly_sidecar,
)

_MODEL_TYPES = ["Gaussian", "Exponential", "Matern", "Spherical", "Linear",
                "Stable", "Rational", "Cubic", "HyperSpherical"]

#: A kriging grid with more points than this is written to a CSV file and
#: returned by path instead of as arrays.
KRIGING_INLINE_MAX = 2500

_NOT_INSTALLED = {"error": "gstools is not installed. Install with: pip "
                           "install gstools"}


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _out_path(name: str) -> str:
    """Where a file named ``name`` goes: the conversation's working folder
    when a host set one (a path elsewhere keeps only its file name), else
    resolved like every other tool output."""
    from funhouse_agent._fileio import into_working_folder, resolve_output_path
    return resolve_output_path(into_working_folder(str(name)))


def _save(path: str, content) -> dict:
    """Verified write; returns ``{path, file_exists[, error, rescue_path]}``."""
    from funhouse_agent._fileio import save_verified
    saved = save_verified(path, content)
    out = {"path": saved.get("saved", os.path.abspath(path)),
           "file_exists": bool(saved.get("file_exists"))}
    for key in ("error", "rescue_path"):
        if saved.get(key):
            out[key] = saved[key]
    return out


def _prefix(params, default: str) -> str:
    p = params.get("output_prefix")
    if p is None or not str(p).strip():
        return f"{default}_{_stamp()}"
    p = str(p).strip()
    for ext in (".csv", ".png", ".json"):
        if p.lower().endswith(ext):
            p = p[: -len(ext)]
    return p


# ---------------------------------------------------------------------------
# kriging
# ---------------------------------------------------------------------------

_KRIGING_VALID = (
    "x", "y", "values", "model_type", "kriging_type", "grid_x_min",
    "grid_x_max", "grid_y_min", "grid_y_max", "n_grid_x", "n_grid_y",
    "variance", "len_scale", "nugget", "fit_variogram", "regions",
    "cell_size", "quantity", "margin", "output_prefix")


def _kriging_grid_csv(result) -> str:
    import numpy as np
    from gstools_agent.results import _csv_text
    X, Y = np.meshgrid(result.grid_x, result.grid_y, indexing="ij")
    rows = zip(X.ravel().tolist(), Y.ravel().tolist(),
               np.asarray(result.field).ravel().tolist(),
               np.asarray(result.krige_variance).ravel().tolist())
    return _csv_text(["x", "y", "value", "kriging_variance"], rows)


def _run_kriging(params: dict) -> dict:
    from gstools_agent import analyze_kriging, has_gstools
    if not has_gstools():
        return dict(_NOT_INSTALLED)

    reject_unknown_params(params, _KRIGING_VALID, method="kriging")
    require_params(params, ["x", "y", "values"], method="kriging")
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
        regions=params.get("regions"),
        cell_size=params.get("cell_size"),
        quantity=params.get("quantity", "above"),
        margin=params.get("margin"),
    )
    d = result.to_dict()
    prefix = _prefix(params, "kriging")
    if result.region_cells:
        saved = _save(_out_path(f"{prefix}_regions.csv"),
                      result.regions_csv())
        d["regions_csv"] = saved["path"]
        d["regions_csv_columns"] = (
            "region, level, x, y, kriged_mean, kriging_std, and the Gaussian "
            "pointwise probability that z is "
            f"{result.quantity} the level, per cell")
        for key in ("error", "rescue_path"):
            if key in saved:
                d[f"regions_csv_{key}"] = saved[key]
    n_points = int(result.n_grid_x) * int(result.n_grid_y)
    if n_points > KRIGING_INLINE_MAX and result.field is not None:
        for key in ("field", "krige_variance", "grid_x", "grid_y"):
            d.pop(key, None)
        path = _out_path(f"{prefix}_grid.csv")
        saved = _save(path, _kriging_grid_csv(result))
        d["grid_csv"] = saved["path"]
        d["grid_csv_written"] = saved["file_exists"]
        for key in ("error", "rescue_path"):
            if key in saved:
                d[f"grid_csv_{key}"] = saved[key]
        d["grid_note"] = (
            f"The {result.n_grid_x} x {result.n_grid_y} grid ({n_points:,} "
            f"points) is in grid_csv (columns x, y, value, kriging_variance) "
            f"instead of inline; field_min / field_max / field_mean and "
            f"krige_variance_max summarise it.")
    return clean_result(d)


# ---------------------------------------------------------------------------
# variogram / random_field
# ---------------------------------------------------------------------------

def _run_variogram(params: dict) -> dict:
    from gstools_agent import analyze_variogram, has_gstools
    if not has_gstools():
        return dict(_NOT_INSTALLED)

    reject_unknown_params(
        params, ("x", "y", "values", "model_type", "n_bins", "nugget"),
        method="variogram")
    require_params(params, ["x", "y", "values"], method="variogram")
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
        return dict(_NOT_INSTALLED)

    reject_unknown_params(
        params,
        ("model_type", "variance", "len_scale", "nugget", "mean", "x_min",
         "x_max", "y_min", "y_max", "n_x", "n_y", "seed"),
        method="random_field")
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


# ---------------------------------------------------------------------------
# conditional_simulation
# ---------------------------------------------------------------------------

_COND_VALID = (
    "x", "y", "values", "regions", "names", "model_type", "variance",
    "len_scale", "nugget", "fit_variogram", "kriging_type",
    "measurement_error", "anis", "angle_deg", "cell_size", "margin",
    "quantity", "n_realizations", "seed", "simulation_method", "bounds",
    "bounds_mode", "report_values", "figures", "output_prefix")

#: GSTools' own name for the measurement error, and common spellings.
_COND_ALIASES = {"cond_err": "measurement_error",
                 "n_realisations": "n_realizations",
                 "ens_no": "n_realizations"}


def _run_conditional_simulation(params: dict) -> dict:
    from gstools_agent import analyze_conditional_simulation, has_gstools
    if not has_gstools():
        return dict(_NOT_INSTALLED)

    params = apply_aliases(params, _COND_ALIASES)
    reject_unknown_params(params, _COND_VALID,
                          method="conditional_simulation")
    require_params(params, ["x", "y", "values", "regions"],
                   method="conditional_simulation", valid=_COND_VALID)
    kw = {k: params[k] for k in _COND_VALID
          if params.get(k) is not None
          and k not in ("figures", "output_prefix")}
    result = analyze_conditional_simulation(**kw)
    d = result.to_dict()

    prefix = _prefix(params, "conditional_sim")
    files = {}
    for key, name, text in (
            ("realizations_csv", f"{prefix}_realizations.csv",
             result.realizations_csv),
            ("probability_csv", f"{prefix}_probability.csv",
             result.probability_csv),
            ("kriging_csv", f"{prefix}_kriging.csv", result.kriging_csv)):
        saved = _save(_out_path(name), text())
        files[key] = saved["path"]
        if not saved["file_exists"]:
            d.setdefault("file_errors", []).append(
                {key: saved.get("error", "not written"),
                 "rescue_path": saved.get("rescue_path")})
    files["contents"] = {
        "realizations_csv": "one row per realization: the seed it was "
                            "drawn with, then the volume and area for every "
                            "region@level",
        "probability_csv": ("region, level, x, y and the probability that "
                            f"z is {result.quantity} the level at each cell"),
        "kriging_csv": ("x, y, kriged mean, kriging variance and std (the "
                        "continuous part) at every lattice cell"),
    }
    d["files"] = files

    if params.get("figures", True):
        from gstools_agent.conditional_figures import (
            exceedance_figure, probability_map_figure,
        )
        figs = []
        for kind, fn, what in (
                ("exceedance", exceedance_figure,
                 "exceedance curve P(V > v) per region@level, P50 and P90 "
                 "marked"),
                ("probability_map", probability_map_figure,
                 "plan map, to scale, of the probability at each cell for "
                 "the first region and level, with the outlines and the "
                 "named observations")):
            try:
                png, pfig, _meta = fn(result)
            except Exception as exc:                       # noqa: BLE001
                d.setdefault("warnings", []).append(
                    f"The {kind} figure could not be drawn: {exc}")
                continue
            saved = _save(_out_path(f"{prefix}_{kind}.png"), png)
            entry = {"figure": kind, "shows": what,
                     "output_path": saved["path"],
                     "file_exists": saved["file_exists"]}
            if saved["file_exists"] and pfig is not None:
                try:
                    side = write_plotly_sidecar(saved["path"], pfig.to_json())
                except Exception:                          # noqa: BLE001
                    side = None
                if side:
                    entry.update(side)
            for key in ("error", "rescue_path"):
                if key in saved:
                    entry[key] = saved[key]
            entry["html_img_tag"] = (
                f'<img src="{saved["path"]}" alt="{kind}" '
                f'style="width:100%;max-width:640px;">')
            figs.append(entry)
        d["figures"] = figs
        d["embed_note"] = (
            "Each figure is a PNG plus an interactive chart; the web app "
            "shows the interactive chart under your reply. Paste "
            "html_img_tag into report HTML and html_to_pdf embeds the PNG. "
            "Do not put a markdown image link to a local path in the reply.")
    return clean_result(d)


METHOD_REGISTRY = {
    "kriging": _run_kriging,
    "variogram": _run_variogram,
    "random_field": _run_random_field,
    "conditional_simulation": _run_conditional_simulation,
}

_REGIONS_DESC = (
    "Plan regions: [{name, polygon: [[x, y], ...], level}], level one "
    "number or a list of levels to sweep (same units as values, e.g. an "
    "elevation in m). Totals over the cells whose centre is inside the "
    "polygon (concave shapes fine): volume_above = sum(max(z - level, 0) x "
    "cell area), area_above = cell area where z > level (the 'below' "
    "versions mirror it).")

METHOD_INFO = {
    "kriging": {
        "category": "Geostatistics",
        "brief": ("Kriging (ordinary/simple/universal): ONE smoothed best "
                  "estimate of a property between points -- the mean "
                  "surface and its kriging variance. It understates the "
                  "spread and cannot give the probability of a total over "
                  "an area (use conditional_simulation)."),
        "parameters": {
            "x": {"type": "array", "required": True, "description": "X-coordinates of measurement points (m)."},
            "y": {"type": "array", "required": True, "description": "Y-coordinates of measurement points (m)."},
            "values": {"type": "array", "required": True, "description": "Measured values at each point."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "allowed_values": _MODEL_TYPES, "description": "GSTools covariance model."},
            "kriging_type": {"type": "str", "required": False, "default": "ordinary", "allowed_values": ["ordinary", "simple", "universal"], "description": "ordinary = unknown constant mean; simple = known mean (the sample mean of these observations); universal = linear drift in x and y."},
            "grid_x_min": {"type": "float", "required": False, "description": "Grid X minimum (m). Omitted: the data's x range less 10%."},
            "grid_x_max": {"type": "float", "required": False, "description": "Grid X maximum (m). Omitted: the data's x range plus 10%."},
            "grid_y_min": {"type": "float", "required": False, "description": "Grid Y minimum (m). Omitted: from the data, as x."},
            "grid_y_max": {"type": "float", "required": False, "description": "Grid Y maximum (m). Omitted: from the data, as x."},
            "n_grid_x": {"type": "int", "required": False, "default": 50, "description": "Grid points in X (1 with min = max for one coordinate)."},
            "n_grid_y": {"type": "int", "required": False, "default": 50, "description": "Grid points in Y."},
            "variance": {"type": "float", "required": False, "description": "Sill variance (value units squared). Omitted: the sample variance of THESE observations. With fit_variogram=true (the default) it is only the starting guess and is overridden by the fit."},
            "len_scale": {"type": "float", "required": False, "description": "Correlation length (m). Omitted: a third of the data extent. With fit_variogram=true it is only the starting guess and is overridden by the fit."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance (measurement error + variation below the data spacing); with a nugget the surface does not pass exactly through the data. With fit_variogram=true the nugget is fitted too."},
            "fit_variogram": {"type": "bool", "required": False, "default": True, "description": "Fit variance, len_scale AND nugget to the empirical variogram, overriding values given for them (the fitted values are returned). Refused with too few lag classes: then set false and give variance and len_scale."},
            "regions": {"type": "array", "required": False, "description": _REGIONS_DESC + " Here from the kriging only: the kriged surface's totals, and Gaussian POINTWISE expectations (expected volume, expected area, mean cell probability). These are expectations cell by cell; the probability that a TOTAL exceeds a value needs conditional_simulation."},
            "cell_size": {"type": "float", "required": False, "description": "Cell size (m) for the region totals; default the smallest round size giving at most ~3,000 cells."},
            "quantity": {"type": "str", "required": False, "default": "above", "allowed_values": ["above", "below"], "description": "Region totals above (e.g. rock above an excavation level) or below (e.g. fill to a grade) the level."},
            "margin": {"type": "float", "required": False, "description": "Margin (m) of the region lattice around the regions."},
            "output_prefix": {"type": "str", "required": False, "description": "File name stem for the files this call writes (<stem>_grid.csv for a grid over 2,500 points, <stem>_regions.csv with regions); default kriging_<time>."},
        },
        "returns": {
            "n_data": "Number of observations used.",
            "n_grid_x": "Grid points in x.",
            "n_grid_y": "Grid points in y.",
            "model_type": "Covariance model used.",
            "kriging_type": "Kriging type used.",
            "variance": "Sill variance used (fitted when fit_variogram).",
            "len_scale": "Correlation length used, m (fitted when fit_variogram).",
            "nugget": "Nugget used (fitted when fit_variogram).",
            "field": "Kriged values, nested list [n_grid_x][n_grid_y] (field[i][j] at grid_x[i], grid_y[j]) -- ONLY for grids of up to 2,500 points.",
            "krige_variance": "Kriging variance at each grid point, same shape (GSTools: the variance of the value at the point, nugget included) -- only up to 2,500 points.",
            "grid_x": "Grid x coordinates (up to 2,500 points).",
            "grid_y": "Grid y coordinates (up to 2,500 points).",
            "grid_csv": "Grids over 2,500 points: path of a CSV in the working folder with columns x, y, value, kriging_variance, in place of field / krige_variance / grid_x / grid_y.",
            "grid_csv_written": "True when grid_csv was written and verified.",
            "grid_note": "Says the grid went to grid_csv.",
            "field_min": "Smallest kriged value on the grid.",
            "field_max": "Largest kriged value on the grid.",
            "field_mean": "Mean kriged value over the grid.",
            "krige_variance_max": "Largest kriging variance on the grid.",
            "regions": ("With regions, per region and level: polygon_area, cell_area_sum, n_cells, cell_size; volume_above / area_above (or _below) of the "
                        "KRIGED surface; expected_volume_above, expected_area_above, mean_cell_probability_above (expected fraction of the area), "
                        "max_cell_probability_above (Gaussian pointwise from the kriging mean and std); mean_kriged_value; max_kriging_std; "
                        "nearest_observation_to_centroid (m)."),
            "regions_csv": "With regions: CSV path, per cell: region, level, x, y, kriged_mean, kriging_std, pointwise probability.",
            "regions_csv_columns": "The columns of regions_csv.",
            "warnings": "Poorly constrained or degenerate variogram fit.",
            "notes": "Defaults computed from these observations, the fit override, kriging type details, what the region numbers mean.",
        },
    },
    "variogram": {
        "category": "Geostatistics",
        "brief": "Estimate and fit an empirical variogram from spatial data (sill, range, nugget).",
        "parameters": {
            "x": {"type": "array", "required": True, "description": "X-coordinates of measurement points (m)."},
            "y": {"type": "array", "required": True, "description": "Y-coordinates of measurement points (m)."},
            "values": {"type": "array", "required": True, "description": "Measured values at each point."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "allowed_values": _MODEL_TYPES, "description": "Covariance model to fit."},
            "n_bins": {"type": "int", "required": False, "default": 10, "description": "Number of lag classes between 0 and the largest pair separation."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance."},
        },
        "returns": {
            "variance": "Fitted sill variance.",
            "len_scale": "Fitted correlation length (m).",
            "nugget": "Fitted nugget.",
            "bin_center": "Lag distances of the populated classes.",
            "gamma": "Empirical semivariance per class.",
            "pair_counts": "Point pairs per class.",
            "warnings": "Poorly constrained or degenerate fit.",
        },
    },
    "random_field": {
        "category": "Geostatistics",
        "brief": ("Generate ONE UNconditioned 2D random field (mean + "
                  "correlated variation). It does NOT honour any data; for "
                  "realizations that pass through observations use "
                  "conditional_simulation."),
        "parameters": {
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "allowed_values": _MODEL_TYPES, "description": "Covariance model."},
            "variance": {"type": "float", "required": False, "default": 1.0, "description": "Field variance."},
            "len_scale": {"type": "float", "required": False, "default": 10.0, "description": "Correlation length (m)."},
            "nugget": {"type": "float", "required": False, "default": 0.0, "description": "Nugget variance."},
            "mean": {"type": "float", "required": False, "default": 0.0, "description": "Mean value."},
            "x_min": {"type": "float", "required": False, "default": 0.0, "description": "X-axis minimum (m)."},
            "x_max": {"type": "float", "required": False, "default": 100.0, "description": "X-axis maximum (m)."},
            "y_min": {"type": "float", "required": False, "default": 0.0, "description": "Y-axis minimum (m)."},
            "y_max": {"type": "float", "required": False, "default": 100.0, "description": "Y-axis maximum (m)."},
            "n_x": {"type": "int", "required": False, "default": 50, "description": "Grid points in X."},
            "n_y": {"type": "int", "required": False, "default": 50, "description": "Grid points in Y."},
            "seed": {"type": "int", "required": False, "default": 42, "description": "Random seed for reproducibility."},
        },
        "returns": {
            "model_type": "Covariance model used.",
            "field": "The generated field [n_x][n_y].",
            "field_mean": "Actual mean of generated field.",
            "field_std": "Actual std of generated field.",
        },
    },
    "conditional_simulation": {
        "category": "Geostatistics",
        "brief": ("Realizations that honour the data (conditioned random "
                  "field), each integrated over plan polygons: probabilities "
                  "of a TOTAL over an area -- volume above/below a level "
                  "(rock to excavate, fill), area, any encounter -- with "
                  "percentiles, exceedance, CSV files and figures."),
        "parameters": {
            "x": {"type": "array", "required": True, "description": "X-coordinates of the observations (m), at least 3."},
            "y": {"type": "array", "required": True, "description": "Y-coordinates of the observations (m)."},
            "values": {"type": "array", "required": True, "description": "Observed values, e.g. top-of-rock elevation (m) at each boring."},
            "regions": {"type": "array", "required": True, "description": _REGIONS_DESC + " Every region and level uses the SAME realizations, so differences between footprints or levels are meaningful."},
            "names": {"type": "array", "required": False, "description": "Observation names (e.g. boring IDs), one per point; label the map and files."},
            "model_type": {"type": "str", "required": False, "default": "Gaussian", "allowed_values": _MODEL_TYPES, "description": "GSTools covariance model (Exponential is rougher than Gaussian)."},
            "variance": {"type": "float", "required": False, "description": "Sill variance (value units squared, m2 for elevations). Omitted: the sample variance of these observations, flagged. The percentiles follow it -- state its basis."},
            "len_scale": {"type": "float", "required": False, "description": "Correlation length (m). Omitted: a third of the data extent, flagged. State its basis."},
            "nugget": {"type": "float", "required": False, "description": "Nugget variance (error + variation below the data spacing); default 0, or the largest measurement_error. With a nugget the field does not pass exactly through the data."},
            "fit_variogram": {"type": "bool", "required": False, "default": False, "description": "Fit variance, len_scale (and the nugget if not given) to the empirical variogram. Off by default: with a handful of borings a fitted variogram is unstable."},
            "kriging_type": {"type": "str", "required": False, "default": "ordinary", "allowed_values": ["ordinary", "simple", "universal"], "description": "ordinary = unknown constant mean; simple = known mean (the sample mean); universal = linear drift in x and y (variance is then the residual's)."},
            "measurement_error": {"type": "float", "required": False, "description": "Error variance at the observations (GSTools cond_err), one number or one per point; not above the nugget (GSTools counts it in the nugget). The field then need not pass through the data."},
            "anis": {"type": "float", "required": False, "default": 1.0, "description": "Anisotropy ratio (minor / main correlation length)."},
            "angle_deg": {"type": "float", "required": False, "default": 0.0, "description": "Direction of the main correlation length, degrees counter-clockwise from +x (GSTools angles, in degrees here)."},
            "cell_size": {"type": "float", "required": False, "description": "Cell size (m); default the smallest round size giving <= ~3,000 cells; over 5,000 cells refused (the message names a size)."},
            "margin": {"type": "float", "required": False, "description": "Margin (m) of the lattice around the regions (kriged-mean file, map)."},
            "quantity": {"type": "str", "required": False, "default": "above", "allowed_values": ["above", "below"], "description": "above = volume/area of the field above the level (e.g. rock to excavate); below = below it (e.g. fill)."},
            "n_realizations": {"type": "int", "required": False, "default": 1000, "description": "Realizations (max 20,000). The convergence table shows if more are needed."},
            "seed": {"type": "int", "required": False, "default": 42, "description": "Realization i uses seed + i (reproducible)."},
            "simulation_method": {"type": "str", "required": False, "default": "kriging_residual", "allowed_values": ["kriging_residual", "condsrf"], "description": "kriging_residual = conditioning by kriging (Journel & Huijbregts 1978): exact conditional draws, right correlation between cells, seconds. condsrf = GSTools CondSRF (kriged field + unconditional field scaled by the kriging std): right pointwise variance, unconditioned correlation, so totals can spread too much; slower. condsrf reproduces a CondSRF notebook."},
            "bounds": {"type": "array", "required": False, "description": "Inequality data [{x, y, max} or {x, y, min}, optional name], e.g. a boring that ended above rock: rock surface below its bottom elevation -> max."},
            "bounds_mode": {"type": "str", "required": False, "default": "report", "allowed_values": ["report", "reject"], "description": "report = how often each bound is violated; reject = keep only realizations satisfying every bound (rejection sampling, not inequality kriging; at most 10 x n_realizations draws; acceptance rate reported)."},
            "report_values": {"type": "array", "required": False, "description": "Totals for the exceedance table P(V > v); default round values per region and level."},
            "figures": {"type": "bool", "required": False, "default": True, "description": "Draw the exceedance curve and the probability map (PNG + interactive)."},
            "output_prefix": {"type": "str", "required": False, "description": "File name stem for the CSVs and figures (default conditional_sim_<time>)."},
        },
        "returns": {
            "results": ("Per region and level: polygon_area, cell_area_sum, volume_above / area_above (or _below) "
                        "{mean, std, P5..P95}, p_any = P(volume > 0), exceedance [{value, p_exceed, std_error}], "
                        "convergence [{n, mean, P90}], kriged_surface (the deterministic total), pointwise_expectation "
                        "(expected total from kriging mean + variance; the realizations' mean should match), "
                        "nearest_observation_to_centroid."),
            "definitions": "What each total and percentile means (P90 = 90% chance of not being exceeded).",
            "covariance_model": "Model actually used (and whether fitted).",
            "method_basis": "How the realizations were made, with references.",
            "grid": "Cell size, cell area, cells simulated, lattice.",
            "data_check": "Largest deviation of the realizations from the observations.",
            "defaults_from_data": "Defaults computed from THESE observations, with their basis.",
            "non_finite_redrawn": "Only if it happened: NaN realizations redrawn with fresh seeds.",
            "bounds": "Per bound: kriged value, kriging std, violated_fraction; acceptance_rate under reject.",
            "files": "realizations_csv, probability_csv, kriging_csv paths in the working folder.",
            "figures": "PNG paths with .plotly.json twins: exceedance curve, probability map.",
            "warnings": "Few observations, regions far from data, level outside the data range, coarse cells, unconverged P90, low acceptance.",
            "assumptions": "Gaussian stationary field, how the nugget enters, common random numbers.",
        },
    },
}
