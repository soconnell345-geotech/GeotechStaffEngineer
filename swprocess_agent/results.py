"""
Result dataclasses for swprocess agent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DispersionResult:
    """Results from MASW dispersion analysis.

    Attributes
    ----------
    n_channels : int
        Number of sensors in the array.
    spacing_m : float
        Sensor spacing in meters.
    transform : str
        Wavefield transform used (phase_shift, fk, fdbf).
    n_freq : int
        Number of frequency bins.
    n_vel : int
        Number of velocity bins.
    frequencies : np.ndarray or None
        Frequency vector (Hz).
    velocities_grid : np.ndarray or None
        Velocity vector for the power grid (m/s).
    power : np.ndarray or None
        Wavefield transform power (n_vel x n_freq).
    disp_freq : np.ndarray or None
        Dispersion curve frequencies (Hz).
    disp_vel : np.ndarray or None
        Dispersion curve phase velocities (m/s).
    """
    n_channels: int = 0
    spacing_m: float = 0.0
    transform: str = ""
    n_freq: int = 0
    n_vel: int = 0
    frequencies: Optional[np.ndarray] = None
    velocities_grid: Optional[np.ndarray] = None
    power: Optional[np.ndarray] = None
    disp_freq: Optional[np.ndarray] = None
    disp_vel: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  MASW DISPERSION ANALYSIS RESULTS",
            "=" * 60,
            f"  Channels:    {self.n_channels}",
            f"  Spacing:     {self.spacing_m:.2f} m",
            f"  Transform:   {self.transform}",
            f"  Freq bins:   {self.n_freq}",
            f"  Vel bins:    {self.n_vel}",
        ]
        if self.disp_freq is not None:
            lines.append(f"  Disp. curve: {len(self.disp_freq)} points")
            lines.append(f"  Freq range:  [{self.disp_freq.min():.1f}, {self.disp_freq.max():.1f}] Hz")
            lines.append(f"  Vel range:   [{self.disp_vel.min():.1f}, {self.disp_vel.max():.1f}] m/s")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n_channels": self.n_channels,
            "spacing_m": float(self.spacing_m),
            "transform": self.transform,
            "n_freq": self.n_freq,
            "n_vel": self.n_vel,
        }
        if self.disp_freq is not None:
            d["disp_freq_hz"] = [float(x) for x in self.disp_freq]
        if self.disp_vel is not None:
            d["disp_vel_mps"] = [float(x) for x in self.disp_vel]
        if self.frequencies is not None:
            d["freq_min"] = float(self.frequencies.min())
            d["freq_max"] = float(self.frequencies.max())
        if self.velocities_grid is not None:
            d["vel_min"] = float(self.velocities_grid.min())
            d["vel_max"] = float(self.velocities_grid.max())
        # Omit power grid from JSON (too large) — keep disp curve only
        return d

    def plot_dispersion(self, ax=None, show=True, **kwargs):
        """Plot dispersion image and extracted curve."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        setup_engineering_plot(ax, "MASW Dispersion", "Frequency (Hz)",
                               "Phase Velocity (m/s)")

        if self.power is not None and self.frequencies is not None:
            F, V = np.meshgrid(self.frequencies, self.velocities_grid)
            ax.pcolormesh(F, V, self.power, shading='auto', cmap='hot_r')

        if self.disp_freq is not None and self.disp_vel is not None:
            ax.plot(self.disp_freq, self.disp_vel, 'co-', markersize=3,
                    linewidth=1.5, label='Dispersion curve')
            ax.legend(fontsize=8)

        if show:
            plt.show()
        return ax
