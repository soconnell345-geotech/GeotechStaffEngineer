"""
Results container and plotting for lateral pile analysis.

Provides a clean interface to access analysis results and generate
standard engineering plots of deflection, moment, shear, and soil reaction
versus depth.

Matplotlib is optional — plotting methods will raise ImportError with
a helpful message if matplotlib is not installed.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Results:
    """Container for lateral pile analysis results.

    Attributes
    ----------
    z : numpy.ndarray
        Depth array (m).
    deflection : numpy.ndarray
        Lateral deflection (m).
    slope : numpy.ndarray
        Rotation/slope (radians).
    moment : numpy.ndarray
        Bending moment (kN-m).
    shear : numpy.ndarray
        Shear force (kN).
    soil_reaction : numpy.ndarray
        Soil reaction p (kN/m).
    Es : numpy.ndarray
        Secant soil modulus (kN/m^2).
    iterations : int
        Number of solver iterations.
    converged : bool
        Whether the solver converged.
    pile_length : float
        Pile length (m).
    pile_diameter : float
        Pile diameter (m).
    Vt : float
        Applied lateral load (kN).
    Mt : float
        Applied moment (kN-m).
    Q : float
        Applied axial load (kN).
    """
    z: np.ndarray
    deflection: np.ndarray
    slope: np.ndarray
    moment: np.ndarray
    shear: np.ndarray
    soil_reaction: np.ndarray
    Es: np.ndarray
    iterations: int
    converged: bool
    pile_length: float
    pile_diameter: float
    Vt: float
    Mt: float
    Q: float

    @property
    def y_top(self) -> float:
        """Pile head deflection (m)."""
        return float(self.deflection[0])

    @property
    def rotation_top(self) -> float:
        """Pile head rotation (radians)."""
        return float(self.slope[0])

    @property
    def max_moment(self) -> float:
        """Maximum bending moment magnitude (kN-m)."""
        return float(np.max(np.abs(self.moment)))

    @property
    def max_moment_depth(self) -> float:
        """Depth of maximum bending moment (m)."""
        idx = np.argmax(np.abs(self.moment))
        return float(self.z[idx])

    @property
    def max_deflection(self) -> float:
        """Maximum deflection magnitude (m)."""
        return float(np.max(np.abs(self.deflection)))

    @property
    def max_shear(self) -> float:
        """Maximum shear force magnitude (kN)."""
        return float(np.max(np.abs(self.shear)))

    def depth_of_zero_deflection(self) -> float:
        """Depth at which deflection first crosses zero (m).

        Returns
        -------
        float
            Depth of first zero crossing, or pile_length if none found.
        """
        for i in range(len(self.deflection) - 1):
            if self.deflection[i] * self.deflection[i + 1] < 0:
                # Linear interpolation
                frac = abs(self.deflection[i]) / (
                    abs(self.deflection[i]) + abs(self.deflection[i + 1])
                )
                return float(self.z[i] + frac * (self.z[i + 1] - self.z[i]))
        return float(self.pile_length)

    def summary(self) -> str:
        """Return a text summary of key results."""
        lines = [
            "Lateral Pile Analysis Results",
            "=" * 40,
            f"Pile length:          {self.pile_length:.2f} m",
            f"Pile diameter:        {self.pile_diameter:.3f} m",
            f"Applied lateral load: {self.Vt:.1f} kN",
            f"Applied moment:       {self.Mt:.1f} kN-m",
            f"Applied axial load:   {self.Q:.1f} kN",
            "",
            f"Pile head deflection: {self.y_top*1000:.2f} mm",
            f"Pile head rotation:   {self.rotation_top*1000:.4f} mrad",
            f"Max bending moment:   {self.max_moment:.1f} kN-m at {self.max_moment_depth:.2f} m",
            f"Max shear force:      {self.max_shear:.1f} kN",
            f"Max deflection:       {self.max_deflection*1000:.2f} mm",
            f"Zero deflection at:   {self.depth_of_zero_deflection():.2f} m",
            "",
            f"Solver iterations:    {self.iterations}",
            f"Converged:            {self.converged}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export results as a dictionary (for JSON serialization)."""
        return {
            'z': self.z.tolist(),
            'deflection_m': self.deflection.tolist(),
            'slope_rad': self.slope.tolist(),
            'moment_kNm': self.moment.tolist(),
            'shear_kN': self.shear.tolist(),
            'soil_reaction_kN_per_m': self.soil_reaction.tolist(),
            'y_top_m': self.y_top,
            'rotation_top_rad': self.rotation_top,
            'max_moment_kNm': self.max_moment,
            'max_moment_depth_m': self.max_moment_depth,
            'max_deflection_m': self.max_deflection,
            'iterations': self.iterations,
            'converged': self.converged,
        }

    # ---- Plotting methods ----

    def plot_deflection(self, ax=None, show: bool = True, **kwargs):
        """Plot lateral deflection vs depth.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        show : bool
            Whether to call plt.show(). Default True.
        **kwargs
            Additional keyword arguments passed to ax.plot().
        """
        ax, plt = self._get_axes(ax)
        ax.plot(self.deflection * 1000, self.z, **kwargs)
        ax.set_xlabel('Deflection (mm)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Lateral Deflection vs Depth')
        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        if show and plt is not None:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_moment(self, ax=None, show: bool = True, **kwargs):
        """Plot bending moment vs depth."""
        ax, plt = self._get_axes(ax)
        ax.plot(self.moment, self.z, **kwargs)
        ax.set_xlabel('Bending Moment (kN-m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Bending Moment vs Depth')
        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        if show and plt is not None:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_shear(self, ax=None, show: bool = True, **kwargs):
        """Plot shear force vs depth."""
        ax, plt = self._get_axes(ax)
        ax.plot(self.shear, self.z, **kwargs)
        ax.set_xlabel('Shear Force (kN)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Shear Force vs Depth')
        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        if show and plt is not None:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_soil_reaction(self, ax=None, show: bool = True, **kwargs):
        """Plot soil reaction vs depth."""
        ax, plt = self._get_axes(ax)
        ax.plot(self.soil_reaction, self.z, **kwargs)
        ax.set_xlabel('Soil Reaction p (kN/m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Soil Reaction vs Depth')
        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        if show and plt is not None:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, show: bool = True):
        """Plot all four response profiles in a 2x2 grid."""
        plt = _import_matplotlib()
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))
        self.plot_deflection(ax=axes[0], show=False)
        self.plot_moment(ax=axes[1], show=False)
        self.plot_shear(ax=axes[2], show=False)
        self.plot_soil_reaction(ax=axes[3], show=False)
        fig.suptitle(
            f'Lateral Pile Analysis: Vt={self.Vt:.0f} kN, Mt={self.Mt:.0f} kN-m, Q={self.Q:.0f} kN',
            fontsize=12
        )
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def _get_axes(self, ax):
        """Get or create matplotlib axes."""
        plt = _import_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        return ax, plt


def _import_matplotlib():
    """Import matplotlib, raising a helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )
