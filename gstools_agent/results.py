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
        Fitted nugget.
    bin_center : np.ndarray or None
        Bin centers (lag distances).
    gamma : np.ndarray or None
        Empirical variogram values.
    """
    n_data: int = 0
    n_bins: int = 0
    model_type: str = ""
    variance: float = 0.0
    len_scale: float = 0.0
    nugget: float = 0.0
    bin_center: Optional[np.ndarray] = None
    gamma: Optional[np.ndarray] = None

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
        }
        if self.bin_center is not None:
            d["bin_center"] = [float(x) for x in self.bin_center]
        if self.gamma is not None:
            d["gamma"] = [float(x) for x in self.gamma]
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
