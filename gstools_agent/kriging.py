"""
Kriging interpolation wrapper using GSTools.

Spatially interpolates soil properties (SPT N, Vs, etc.) from point
measurements onto a regular grid.
"""

import numpy as np

from gstools_agent.gstools_utils import import_gstools
from gstools_agent.results import KrigingResult


_VALID_MODELS = {
    "Gaussian", "Exponential", "Matern", "Spherical", "Linear",
    "Stable", "Rational", "Cubic", "HyperSpherical",
}

_VALID_KRIGE_TYPES = {"ordinary", "simple", "universal"}


def _validate_kriging_inputs(x, y, values, model_type, kriging_type,
                             grid_x_min, grid_x_max, grid_y_min, grid_y_max,
                             n_grid_x, n_grid_y):
    """Validate kriging inputs."""
    if len(x) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(x)}")
    if len(x) != len(y) or len(x) != len(values):
        raise ValueError(
            f"x, y, values must have same length: "
            f"x={len(x)}, y={len(y)}, values={len(values)}"
        )
    if model_type not in _VALID_MODELS:
        raise ValueError(
            f"model_type must be one of {sorted(_VALID_MODELS)}, got '{model_type}'"
        )
    if kriging_type not in _VALID_KRIGE_TYPES:
        raise ValueError(
            f"kriging_type must be one of {sorted(_VALID_KRIGE_TYPES)}, got '{kriging_type}'"
        )
    # One grid line (n = 1, min == max) is a valid request: kriging at a
    # single coordinate, e.g. one point (live smoke GS-1 asked for (5, 5)).
    for axis, lo, hi, n in (("x", grid_x_min, grid_x_max, n_grid_x),
                            ("y", grid_y_min, grid_y_max, n_grid_y)):
        if n < 1:
            raise ValueError(f"n_grid_{axis} must be >= 1")
        if n == 1:
            if hi != lo:
                raise ValueError(
                    f"n_grid_{axis} = 1 krigs on one line: give "
                    f"grid_{axis}_min = grid_{axis}_max (the coordinate)")
        elif hi <= lo:
            raise ValueError(
                f"grid_{axis}_max must be > grid_{axis}_min when "
                f"n_grid_{axis} >= 2 (for a single point use n_grid_{axis}=1 "
                f"and grid_{axis}_min = grid_{axis}_max)")


def analyze_kriging(
    x,
    y,
    values,
    model_type="Gaussian",
    kriging_type="ordinary",
    grid_x_min=None,
    grid_x_max=None,
    grid_y_min=None,
    grid_y_max=None,
    n_grid_x=50,
    n_grid_y=50,
    variance=None,
    len_scale=None,
    nugget=0.0,
    fit_variogram=True,
    regions=None,
    cell_size=None,
    quantity="above",
    margin=None,
) -> KrigingResult:
    """Krige soil property values onto a regular grid.

    Kriging gives ONE smoothed best estimate (the mean surface). It
    understates the spread of the field and cannot give the probability of a
    total over an area; for that use
    :func:`gstools_agent.analyze_conditional_simulation`.

    Parameters
    ----------
    x, y : array-like
        Coordinates of measurement points.
    values : array-like
        Measured values at each point (e.g. SPT N-values).
    model_type : str
        Covariance model. Default 'Gaussian'.
    kriging_type : str
        Kriging type: 'ordinary', 'simple', or 'universal'. Default 'ordinary'.
    grid_x_min, grid_x_max : float or None
        Grid extent in X. None = auto from data bounds with 10% buffer.
    grid_y_min, grid_y_max : float or None
        Grid extent in Y.
    n_grid_x, n_grid_y : int
        Number of grid points. Default 50 each.
    variance : float or None
        Sill variance. None = estimate from data.
    len_scale : float or None
        Correlation length. None = estimate from data.
    nugget : float
        Nugget variance. Default 0.
    fit_variogram : bool
        If True, fit variogram model to data. Default True.
    regions : list of dict or None
        ``{name, polygon: [[x, y], ...], level}`` (level one number or a
        list): the kriged surface's volume and area above (or below) each
        level over the cells whose centre is inside the polygon.
    cell_size : float or None
        Cell size (m) of the region lattice (separate from the output grid);
        None = the smallest round size giving at most ~3,000 cells.
    quantity : str
        'above' (default) or 'below' the level.
    margin : float or None
        Lattice margin around the regions (m).

    Returns
    -------
    KrigingResult
        Kriged field, variance, and model parameters.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    val_arr = np.asarray(values, dtype=float)

    # Auto grid bounds
    buf = 0.1
    if grid_x_min is None:
        rng = x_arr.max() - x_arr.min()
        grid_x_min = x_arr.min() - buf * max(rng, 1.0)
    if grid_x_max is None:
        rng = x_arr.max() - x_arr.min()
        grid_x_max = x_arr.max() + buf * max(rng, 1.0)
    if grid_y_min is None:
        rng = y_arr.max() - y_arr.min()
        grid_y_min = y_arr.min() - buf * max(rng, 1.0)
    if grid_y_max is None:
        rng = y_arr.max() - y_arr.min()
        grid_y_max = y_arr.max() + buf * max(rng, 1.0)

    _validate_kriging_inputs(
        x_arr, y_arr, val_arr, model_type, kriging_type,
        grid_x_min, grid_x_max, grid_y_min, grid_y_max,
        n_grid_x, n_grid_y,
    )

    gs = import_gstools()

    # Build covariance model. Data-dependent defaults are computed here, from
    # THIS call's observations, and echoed in the notes.
    warnings = []
    notes = []
    model_cls = getattr(gs, model_type)
    role = ("the starting value for the fit" if fit_variogram else "used")
    if variance is None:
        variance = float(np.var(val_arr))
        notes.append(f"variance not given: the sample variance of the "
                     f"{len(val_arr)} observations ({variance:.4g}) is {role}.")
    if len_scale is None:
        # Initial guess: 1/3 of data extent
        extent = max(x_arr.max() - x_arr.min(), y_arr.max() - y_arr.min())
        len_scale = extent / 3.0
        notes.append(f"len_scale not given: one third of the data extent "
                     f"({len_scale:.4g} m) is {role}.")
    if fit_variogram:
        notes.append("fit_variogram=true: variance, len_scale and nugget "
                     "are fitted to the empirical variogram; values given "
                     "for them are only starting guesses (the fitted values "
                     "are reported).")

    model = model_cls(dim=2, var=variance, len_scale=len_scale, nugget=nugget)

    # Fit variogram if requested (same data-based lag classes and the same
    # refusal / degeneracy checks as analyze_variogram — live smoke GS-1).
    if fit_variogram:
        from gstools_agent.variogram import empirical_variogram, fit_warnings
        try:
            bin_center, gamma, _counts, warnings = empirical_variogram(
                gs, x_arr, y_arr, val_arr)
        except ValueError as exc:
            raise ValueError(
                f"{exc} To krige anyway, pass fit_variogram=false with "
                f"variance and len_scale.") from None
        model.fit_variogram(bin_center, gamma, nugget=nugget >= 0)
        warnings += fit_warnings(model, val_arr, float(
            np.max(np.hypot(x_arr[:, None] - x_arr[None, :],
                            y_arr[:, None] - y_arr[None, :]))))

    # Run kriging
    if kriging_type == "simple":
        # Simple kriging needs the field's (known) mean; gstools defaults it
        # to 0, which pulls every estimate toward zero away from the data.
        sk_mean = float(np.mean(val_arr))
        krig = gs.krige.Simple(model, [x_arr, y_arr], val_arr, mean=sk_mean)
        notes.append(f"Simple kriging about the sample mean {sk_mean:.4g}.")
    elif kriging_type == "universal":
        # Universal kriging needs a drift; a linear trend in x and y.
        krig = gs.krige.Universal(model, [x_arr, y_arr], val_arr,
                                  drift_functions="linear")
        notes.append("Universal kriging with a linear drift in x and y.")
    else:
        krig = gs.krige.Ordinary(model, [x_arr, y_arr], val_arr)

    grid_x = np.linspace(grid_x_min, grid_x_max, n_grid_x)
    grid_y = np.linspace(grid_y_min, grid_y_max, n_grid_y)
    field, krige_var = krig.structured([grid_x, grid_y])

    region_rows = region_cells = None
    if regions is not None:
        region_rows, region_cells = _region_totals(
            krig, model, regions, cell_size, quantity, margin, x_arr, y_arr,
            notes)

    return KrigingResult(
        n_data=len(x_arr),
        n_grid_x=n_grid_x,
        n_grid_y=n_grid_y,
        model_type=model_type,
        variance=float(model.var),
        len_scale=float(model.len_scale),
        nugget=float(model.nugget),
        kriging_type=kriging_type,
        field=field,
        krige_variance=krige_var,
        grid_x=grid_x,
        grid_y=grid_y,
        warnings=warnings or None,
        notes=notes or None,
        regions=region_rows,
        region_cells=region_cells,
        quantity=quantity,
    )


#: Most lattice cells inside the regions for the kriged-surface totals.
_MAX_REGION_CELLS = 200000


def _region_totals(krig, model, regions, cell_size, quantity, margin,
                   x_obs, y_obs, notes):
    """Volume / area of the KRIGED surface above (below) each region level."""
    from gstools_agent.regions import (
        QUANTITIES, auto_cell_size, build_lattice, integrate, parse_regions,
        pointwise_gaussian, region_geometry, region_masks,
    )
    if quantity not in QUANTITIES:
        raise ValueError(f"quantity must be one of {QUANTITIES}, got "
                         f"'{quantity}'")
    regs = parse_regions(regions, method="kriging")
    cell = auto_cell_size(regs) if cell_size is None else float(cell_size)
    if not cell > 0:
        raise ValueError("cell_size must be > 0 (m)")
    lat = build_lattice(regs, cell, margin)
    if lat["nx"] * lat["ny"] > 50 * _MAX_REGION_CELLS:
        raise ValueError(f"cell_size {cell:g} m makes a {lat['nx']} x "
                         f"{lat['ny']} lattice over the regions; use a larger "
                         f"cell_size.")
    masks = region_masks(regs, lat["px"], lat["py"], cell)
    union = np.zeros(lat["nx"] * lat["ny"], dtype=bool)
    for m in masks:
        union |= m
    n_cells = int(union.sum())
    if n_cells > _MAX_REGION_CELLS:
        raise ValueError(f"{n_cells:,} cells of {cell:g} m lie inside the "
                         f"regions (limit {_MAX_REGION_CELLS:,}); use a "
                         f"larger cell_size.")
    for reg, m in zip(regs, masks):
        if not m.any():
            raise ValueError(f"region '{reg['name']}' holds no cell centre "
                             f"at cell_size {cell:g} m; use a smaller "
                             f"cell_size.")
    px, py = lat["px"][union], lat["py"][union]
    zc, vc = krig([px, py], mesh_type="unstructured", store=False)
    zc = np.asarray(zc, dtype=float)
    sd = np.sqrt(np.clip(np.asarray(vc, dtype=float) - float(model.nugget),
                         0.0, None))
    cmask = [m[union] for m in masks]
    area = cell * cell
    geom = region_geometry(regs, cmask, area, x_obs, y_obs)
    q = quantity
    rows, cell_rows = [], []
    for reg, m, g in zip(regs, cmask, geom):
        vol, ar = integrate(zc[m], reg["levels"], area, quantity)
        for li, lv in enumerate(reg["levels"]):
            p, e = pointwise_gaussian(zc[m], sd[m], lv, quantity)
            rows.append({
                "region": reg["name"], "level": float(lv),
                "polygon_area": g["polygon_area"],
                "cell_area_sum": g["cell_area_sum"],
                "n_cells": g["n_cells"], "cell_size": cell,
                f"volume_{q}": float(vol[li]),
                f"area_{q}": float(ar[li]),
                f"expected_volume_{q}": float(e.sum() * area),
                f"expected_area_{q}": float(p.sum() * area),
                f"mean_cell_probability_{q}": float(p.mean()),
                f"max_cell_probability_{q}": float(p.max()),
                "mean_kriged_value": float(zc[m].mean()),
                "max_kriging_std": float(sd[m].max()),
                "nearest_observation_to_centroid": g[
                    "nearest_observation_distance"],
            })
            cell_rows.append((reg["name"], float(lv), px[m], py[m], zc[m],
                              sd[m], p))
    notes.append(
        f"regions: volume_{q} and area_{q} are totals of the KRIGED (mean) "
        f"surface -- one smoothed estimate, no spread. expected_volume_{q} "
        f"and expected_area_{q} add, cell by cell, the Gaussian expectation "
        f"from the kriging mean and variance (sigma phi(d) + (mu - level) "
        f"Phi(d), d = (mu - level) / sigma; and P = Phi(d)), so they are the "
        f"EXPECTED totals and mean_cell_probability_{q} is the expected "
        f"fraction of the area {q} the level. These are pointwise "
        f"expectations: the probability that a TOTAL exceeds a value (a P90 "
        f"volume, the chance of any rock) needs conditional_simulation.")
    return rows, cell_rows
