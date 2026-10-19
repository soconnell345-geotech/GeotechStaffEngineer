"""
Result dataclasses for GSTools agent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class KrigingResult:
    """Results from kriging interpolation.

    Attributes
    ----------
    n_data : int
        Number of input data points.
    n_grid_x, n_grid_y : int
        Grid dimensions.
    model_type : str
        Covariance model name (e.g. 'Gaussian', 'Exponential').
    variance : float
        Fitted variance (sill).
    len_scale : float
        Fitted correlation length.
    nugget : float
        Nugget variance.
    kriging_type : str
        Kriging type ('ordinary', 'simple', 'universal').
    field : np.ndarray or None
        Interpolated field (n_grid_x x n_grid_y).
    krige_variance : np.ndarray or None
        Kriging variance field.
    grid_x : np.ndarray or None
        X coordinates of grid.
    grid_y : np.ndarray or None
        Y coordinates of grid.
    """
    n_data: int = 0
    n_grid_x: int = 0
    n_grid_y: int = 0
    model_type: str = ""
    variance: float = 0.0
    len_scale: float = 0.0
    nugget: float = 0.0
    kriging_type: str = "ordinary"
    field: Optional[np.ndarray] = None
    krige_variance: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None
    # (``field`` is a data attribute here, so no dataclasses.field below)
    warnings: Optional[list] = None
    notes: Optional[list] = None
    #: Per region and level: the kriged surface's volume / area above (or
    #: below) the level -- set only when ``regions`` were given.
    regions: Optional[list] = None
    #: Not serialized: per region/level (name, level, x, y, mean, std, p)
    #: cell arrays behind ``regions``, for :meth:`regions_csv`.
    region_cells: Optional[list] = None
    quantity: str = "above"

    def regions_csv(self) -> str:
        """Per region and level, every cell: the kriged mean, the kriging
        std (continuous part) and the Gaussian pointwise probability that
        the value is above (below) the level."""
        rel = "gt" if self.quantity == "above" else "lt"
        rows = []
        for name, lv, xs, ys, mu, sd, p in (self.region_cells or []):
            for a, b, c, e, f in zip(xs, ys, mu, sd, p):
                rows.append([name, float(lv), float(a), float(b), float(c),
                             float(e), float(f)])
        return _csv_text(["region", "level", "x", "y", "kriged_mean",
                          "kriging_std", f"p_z_{rel}_level"], rows)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  KRIGING INTERPOLATION RESULTS",
            "=" * 60,
            f"  Data points:        {self.n_data}",
            f"  Grid:               {self.n_grid_x} x {self.n_grid_y}",
            f"  Kriging type:       {self.kriging_type}",
            f"  Model:              {self.model_type}",
            f"  Variance (sill):    {self.variance:.4f}",
            f"  Correlation length: {self.len_scale:.2f}",
            f"  Nugget:             {self.nugget:.4f}",
        ]
        if self.field is not None:
            lines.append(f"  Field range:        [{self.field.min():.3f}, {self.field.max():.3f}]")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n_data": self.n_data,
            "n_grid_x": self.n_grid_x,
            "n_grid_y": self.n_grid_y,
            "model_type": self.model_type,
            "variance": float(self.variance),
            "len_scale": float(self.len_scale),
            "nugget": float(self.nugget),
            "kriging_type": self.kriging_type,
        }
        if self.field is not None:
            d["field"] = self.field.tolist()
            d["field_min"] = float(self.field.min())
            d["field_max"] = float(self.field.max())
            d["field_mean"] = float(self.field.mean())
        if self.krige_variance is not None:
            d["krige_variance"] = self.krige_variance.tolist()
            d["krige_variance_max"] = float(self.krige_variance.max())
        if self.grid_x is not None:
            d["grid_x"] = self.grid_x.tolist()
        if self.grid_y is not None:
            d["grid_y"] = self.grid_y.tolist()
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.notes:
            d["notes"] = list(self.notes)
        if self.regions:
            d["regions"] = [dict(r) for r in self.regions]
        return d

    def plot_field(self, ax=None, show=True, **kwargs):
        """Plot kriged field as contour map."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        setup_engineering_plot(ax, "Kriging Interpolation", "X", "Y")

        if self.field is not None and self.grid_x is not None:
            X, Y = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
            cf = ax.contourf(X, Y, self.field, levels=20, **kwargs)
            plt.colorbar(cf, ax=ax)

        if show:
            plt.show()
        return ax


@dataclass
class VariogramResult:
    """Results from empirical variogram estimation and model fitting.

    Attributes
    ----------
    n_data : int
        Number of input data points.
    n_bins : int
        Number of variogram bins.
    model_type : str
        Fitted covariance model name.
    variance : float
        Fitted variance (sill).
    len_scale : float
        Fitted correlation length.
    nugget : float
        The nugget: fitted, or as given (``nugget_fitted``).
    nugget_fitted : bool
        Whether the nugget was fitted (True) or fixed at the value given.
    bin_center : np.ndarray or None
        Bin centers (lag distances).
    gamma : np.ndarray or None
        Empirical variogram values.
    pair_counts : np.ndarray or None
        Number of point pairs in each lag class.
    warnings : list of str
        Poorly constrained or degenerate fit flags.
    """
    n_data: int = 0
    n_bins: int = 0
    model_type: str = ""
    variance: float = 0.0
    len_scale: float = 0.0
    nugget: float = 0.0
    nugget_fitted: bool = True
    bin_center: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None
    pair_counts: Optional[np.ndarray] = None
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  VARIOGRAM ANALYSIS RESULTS",
            "=" * 60,
            f"  Data points:        {self.n_data}",
            f"  Bins:               {self.n_bins}",
            f"  Best-fit model:     {self.model_type}",
            f"  Variance (sill):    {self.variance:.4f}",
            f"  Correlation length: {self.len_scale:.2f}",
            f"  Nugget:             {self.nugget:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n_data": self.n_data,
            "n_bins": self.n_bins,
            "model_type": self.model_type,
            "variance": float(self.variance),
            "len_scale": float(self.len_scale),
            "nugget": float(self.nugget),
            "nugget_fitted": bool(self.nugget_fitted),
        }
        if self.bin_center is not None:
            d["bin_center"] = [float(x) for x in self.bin_center]
        if self.gamma is not None:
            d["gamma"] = [float(x) for x in self.gamma]
        if self.pair_counts is not None:
            d["pair_counts"] = [int(c) for c in self.pair_counts]
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def plot_variogram(self, ax=None, show=True, **kwargs):
        """Plot empirical variogram with fitted model."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))
        setup_engineering_plot(ax, "Variogram", "Lag Distance", "Semivariance")

        if self.bin_center is not None and self.gamma is not None:
            ax.plot(self.bin_center, self.gamma, 'ko', markersize=5,
                    label='Empirical')
            # Plot fitted model
            lags = np.linspace(0, self.bin_center.max(), 100)
            gs = import_gstools_safe()
            if gs is not None:
                model_cls = getattr(gs, self.model_type, None)
                if model_cls:
                    model = model_cls(
                        dim=2, var=self.variance, len_scale=self.len_scale,
                        nugget=self.nugget,
                    )
                    ax.plot(lags, model.variogram(lags), 'r-',
                            label=f'{self.model_type} fit')
            ax.legend(fontsize=8)

        if show:
            plt.show()
        return ax


def import_gstools_safe():
    """Try to import gstools, return None if not available."""
    try:
        import gstools
        return gstools
    except ImportError:
        return None


@dataclass
class RandomFieldResult:
    """Results from spatial random field generation.

    Attributes
    ----------
    n_grid_x, n_grid_y : int
        Grid dimensions.
    model_type : str
        Covariance model name.
    variance : float
        Variance of the field.
    len_scale : float
        Correlation length.
    mean : float
        Mean value of the field.
    seed : int
        Random seed used.
    field : np.ndarray or None
        Generated random field.
    grid_x, grid_y : np.ndarray or None
        Grid coordinates.
    """
    n_grid_x: int = 0
    n_grid_y: int = 0
    model_type: str = ""
    variance: float = 0.0
    len_scale: float = 0.0
    mean: float = 0.0
    seed: int = 0
    field: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  RANDOM FIELD RESULTS",
            "=" * 60,
            f"  Grid:               {self.n_grid_x} x {self.n_grid_y}",
            f"  Model:              {self.model_type}",
            f"  Variance:           {self.variance:.4f}",
            f"  Correlation length: {self.len_scale:.2f}",
            f"  Mean:               {self.mean:.4f}",
            f"  Seed:               {self.seed}",
        ]
        if self.field is not None:
            lines.append(f"  Field range:        [{self.field.min():.3f}, {self.field.max():.3f}]")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n_grid_x": self.n_grid_x,
            "n_grid_y": self.n_grid_y,
            "model_type": self.model_type,
            "variance": float(self.variance),
            "len_scale": float(self.len_scale),
            "mean": float(self.mean),
            "seed": self.seed,
        }
        if self.field is not None:
            d["field"] = self.field.tolist()
            d["field_min"] = float(self.field.min())
            d["field_max"] = float(self.field.max())
            d["field_mean"] = float(self.field.mean())
            d["field_std"] = float(self.field.std())
        if self.grid_x is not None:
            d["grid_x"] = self.grid_x.tolist()
        if self.grid_y is not None:
            d["grid_y"] = self.grid_y.tolist()
        return d

    def plot_field(self, ax=None, show=True, **kwargs):
        """Plot random field."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        setup_engineering_plot(ax, "Random Field", "X", "Y")

        if self.field is not None and self.grid_x is not None:
            X, Y = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
            cf = ax.contourf(X, Y, self.field, levels=20, **kwargs)
            plt.colorbar(cf, ax=ax)

        if show:
            plt.show()
        return ax


def _csv_text(header, rows) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([f"{v:.6g}" if isinstance(v, (float, np.floating))
                    else v for v in r])
    return buf.getvalue()


def _label(name, level) -> str:
    return f"{name}@{level:g}"


@dataclass
class ConditionalSimulationResult:
    """Results of :func:`gstools_agent.analyze_conditional_simulation`.

    ``to_dict()`` carries the small JSON (model, grid, per region and level
    statistics, exceedance table, convergence, the kriged-surface answer,
    bounds, warnings). The arrays behind it -- every kept realization's
    totals, the per-cell exceedance counts, the kriged mean and variance --
    stay on the object for the CSV writers (:meth:`realizations_csv`,
    :meth:`probability_csv`, :meth:`kriging_csv`) and the figures
    (``gstools_agent.conditional_figures``), never in the JSON.
    """
    simulation_method: str = "kriging_residual"
    method_basis: str = ""
    n_data: int = 0
    model_type: str = ""
    variance: float = 0.0
    len_scale: float = 0.0
    nugget: float = 0.0
    anis: float = 1.0
    angle_deg: float = 0.0
    fitted: bool = False
    kriging_type: str = "ordinary"
    measurement_error: Optional[object] = None
    data_exact: bool = True
    quantity: str = "above"
    cell_size: float = 0.0
    cell_area: float = 0.0
    n_cells: int = 0
    lattice: dict = field(default_factory=dict)
    n_realizations: int = 0
    n_kept: int = 0
    attempts: int = 0
    acceptance_rate: float = 1.0
    seed: int = 0
    bounds_mode: Optional[str] = None
    bounds: list = field(default_factory=list)
    results: list = field(default_factory=list)
    data_check: dict = field(default_factory=dict)
    #: Defaults worked out from THIS call's observations (variance,
    #: len_scale, simple-kriging mean) with their basis, plus the
    #: observations' own summary.
    defaults_from_data: dict = field(default_factory=dict)
    #: Non-finite realizations replaced: [{replaced_seed, redrawn_seed}].
    redrawn: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    runtime_s: float = 0.0
    # --- arrays (not serialized) ---
    obs_x: Optional[np.ndarray] = None
    obs_y: Optional[np.ndarray] = None
    obs_values: Optional[np.ndarray] = None
    obs_names: Optional[list] = None
    regions_geom: Optional[list] = None
    cell_x: Optional[np.ndarray] = None
    cell_y: Optional[np.ndarray] = None
    cell_masks: Optional[list] = None
    levels_all: Optional[list] = None
    exceed_counts: Optional[np.ndarray] = None
    realization_mean: Optional[np.ndarray] = None
    kriged_cells: Optional[np.ndarray] = None
    realization_table: Optional[np.ndarray] = None
    realization_ids: Optional[np.ndarray] = None
    rl_pairs: Optional[list] = None
    kriging_x: Optional[np.ndarray] = None
    kriging_y: Optional[np.ndarray] = None
    kriging_mean: Optional[np.ndarray] = None
    kriging_var: Optional[np.ndarray] = None
    kriging_scope: str = "lattice"
    lattice_full: Optional[dict] = None
    lattice_union: Optional[np.ndarray] = None
    bounds_input: Optional[list] = None

    # -- convenience ---------------------------------------------------------

    def probability(self, region, level) -> np.ndarray:
        """P(z > level) (or < for 'below') at each cell of ``region``."""
        names = [g["name"] for g in self.regions_geom]
        ri = names.index(region)
        li = self.levels_all.index(float(level))
        return (self.exceed_counts[li][self.cell_masks[ri]]
                / float(self.n_kept))

    def totals(self, region, level, what="volume") -> np.ndarray:
        """Every kept realization's volume (or area) for one region/level."""
        i = self.rl_pairs.index((region, float(level)))
        return self.realization_table[:, 2 * i + (0 if what == "volume"
                                                   else 1)]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        q = self.quantity
        d = {
            "analysis": "Conditional simulation over plan regions",
            "simulation_method": self.simulation_method,
            "method_basis": self.method_basis,
            "n_data": self.n_data,
            "covariance_model": {
                "model_type": self.model_type,
                "variance": float(self.variance),
                "len_scale": float(self.len_scale),
                "nugget": float(self.nugget),
                "anis": float(self.anis),
                "angle_deg": float(self.angle_deg),
                "fitted_to_data": bool(self.fitted),
            },
            "kriging_type": self.kriging_type,
            "measurement_error": self.measurement_error,
            "field_passes_through_data": bool(self.data_exact),
            "quantity": q,
            "definitions": {
                f"volume_{q}": (
                    "sum of max(z - level, 0) x cell area over cells whose "
                    "centre is inside the polygon (m3 when z is an elevation "
                    "or thickness in m)" if q == "above" else
                    "sum of max(level - z, 0) x cell area over cells whose "
                    "centre is inside the polygon (m3 when z is in m)"),
                f"area_{q}": (f"sum of cell areas where z is {q} the level "
                              "(m2)"),
                "Pnn": ("percentile of the realizations: P90 has a 90% chance "
                        "of NOT being exceeded (10% chance of more)"),
                "p_any": (f"probability that some cell inside the polygon is "
                          f"{q} the level (volume > 0)"),
                "kriged_surface": (
                    "the same total computed on the kriged (mean) surface; "
                    "the realizations' mean is at least this large, because "
                    "only the parts beyond the level count (the mean of "
                    "max(z - level, 0) is at least max(mean - level, 0))"),
                "pointwise_expectation": (
                    "the EXPECTED total from the kriging mean and variance "
                    "cell by cell (Gaussian: sigma phi(d) + (mu - level) "
                    "Phi(d), d = (mu - level)/sigma; area = sum of P), "
                    "summed. The realizations' mean should match it -- a "
                    "Monte Carlo check; it says nothing about the "
                    "probability of the total"),
            },
            "grid": {
                "cell_size": float(self.cell_size),
                "cell_area": float(self.cell_area),
                "n_cells_simulated": int(self.n_cells),
                "lattice": {k: (float(v) if isinstance(v, (float,
                                                           np.floating))
                                else int(v))
                            for k, v in self.lattice.items()},
            },
            "n_realizations": int(self.n_kept),
            "seed": int(self.seed),
            "common_random_numbers": (
                "every region and level is evaluated on the same "
                "realizations, so differences between them are meaningful"),
            "results": self.results,
            "data_check": self.data_check,
            "defaults_from_data": self.defaults_from_data,
            "runtime_s": round(float(self.runtime_s), 2),
        }
        if self.redrawn:
            d["non_finite_redrawn"] = {
                "count": len(self.redrawn),
                "seeds": self.redrawn[:20],
                "note": ("realizations with a non-finite (NaN) value were "
                         "replaced by fresh draws; none reached the "
                         "statistics"),
            }
        if self.bounds_mode:
            d["bounds_mode"] = self.bounds_mode
            d["bounds"] = self.bounds
            d["attempts"] = int(self.attempts)
            d["acceptance_rate"] = float(self.acceptance_rate)
            if self.bounds_mode == "reject":
                d["bounds_method"] = (
                    "rejection sampling: realizations violating any bound "
                    "are discarded and more are drawn (not inequality "
                    "kriging)")
        if self.warnings:
            d["warnings"] = list(self.warnings)
        if self.notes:
            d["notes"] = list(self.notes)
        if self.assumptions:
            d["assumptions"] = list(self.assumptions)
        return d

    def summary(self) -> str:
        q = self.quantity
        lines = [
            "=" * 64,
            "  CONDITIONAL SIMULATION",
            "=" * 64,
            f"  Method:        {self.simulation_method} "
            f"({self.kriging_type} kriging)",
            f"  Model:         {self.model_type}, var {self.variance:.4g}, "
            f"len_scale {self.len_scale:.4g} m, nugget {self.nugget:.4g}",
            f"  Grid:          {self.n_cells} cells of {self.cell_size:g} m",
            f"  Realizations:  {self.n_kept} (seed {self.seed})",
        ]
        for r in self.results:
            v = r[f"volume_{q}"]
            lines.append(
                f"  {r['region']} @ {r['level']:g}: volume {q} mean "
                f"{v['mean']:.4g}, P50 {v['P50']:.4g}, P90 {v['P90']:.4g}; "
                f"P(>0) {r['p_any']:.3f}; kriged surface "
                f"{r['kriged_surface'][f'volume_{q}']:.4g}")
        lines.append("=" * 64)
        return "\n".join(lines)

    # -- files ---------------------------------------------------------------

    def realizations_csv(self) -> str:
        """One row per kept realization: the seed it was drawn with, then
        the volume and area for every region/level."""
        q = self.quantity
        header = ["seed"]
        for name, lv in self.rl_pairs:
            header += [f"volume_{q}:{_label(name, lv)}",
                       f"area_{q}:{_label(name, lv)}"]
        rows = ([int(i)] + [float(v) for v in row]
                for i, row in zip(self.realization_ids,
                                  self.realization_table))
        return _csv_text(header, rows)

    def probability_csv(self) -> str:
        """Per region and level, every cell: x, y and the probability that
        the field is above (below) the level there."""
        rel = "gt" if self.quantity == "above" else "lt"
        rows = []
        for g, m in zip(self.regions_geom, self.cell_masks):
            for lv in g["levels"]:
                p = self.probability(g["name"], lv)
                for xx, yy, pp in zip(self.cell_x[m], self.cell_y[m], p):
                    rows.append([g["name"], float(lv), float(xx), float(yy),
                                 float(pp)])
        return _csv_text(["region", "level", "x", "y", f"p_z_{rel}_level"],
                         rows)

    def kriging_csv(self) -> str:
        """The kriged mean and the kriging variance (continuous part) at
        every lattice cell (or every simulated cell on a very large
        lattice)."""
        rows = ([float(a), float(b), float(c), float(e), float(np.sqrt(e))]
                for a, b, c, e in zip(self.kriging_x, self.kriging_y,
                                      self.kriging_mean, self.kriging_var))
        return _csv_text(["x", "y", "kriged_mean", "kriging_variance",
                          "kriging_std"], rows)
