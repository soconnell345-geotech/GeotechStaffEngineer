"""
Result dataclasses for PySeismoSoil agent.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CurveResult:
    """Results from nonlinear soil curve generation.

    Attributes
    ----------
    model : str
        Model name ('MKZ' or 'HH').
    params : dict
        Model parameters used.
    n_points : int
        Number of strain points.
    strain_pct : list of float
        Shear strain in percent.
    G_Gmax : list of float
        Modulus reduction G/Gmax.
    damping_pct : list of float
        Damping ratio in percent.
    """
    model: str = ""
    params: dict = field(default_factory=dict)
    n_points: int = 0
    strain_pct: Optional[np.ndarray] = None
    G_Gmax: Optional[np.ndarray] = None
    damping_pct: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  NONLINEAR SOIL CURVES — {self.model} Model",
            "=" * 60,
            f"  Parameters:  {self.params}",
            f"  Points:      {self.n_points}",
        ]
        if self.strain_pct is not None:
            lines.append(f"  Strain range: [{self.strain_pct[0]:.1e}, {self.strain_pct[-1]:.1e}] %")
        if self.G_Gmax is not None:
            lines.append(f"  G/Gmax range: [{self.G_Gmax.min():.4f}, {self.G_Gmax.max():.4f}]")
        if self.damping_pct is not None:
            lines.append(f"  Damping range: [{self.damping_pct.min():.2f}, {self.damping_pct.max():.2f}] %")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "params": self.params,
            "n_points": self.n_points,
        }
        if self.strain_pct is not None:
            d["strain_pct"] = [float(x) for x in self.strain_pct]
        if self.G_Gmax is not None:
            d["G_Gmax"] = [float(x) for x in self.G_Gmax]
        if self.damping_pct is not None:
            d["damping_pct"] = [float(x) for x in self.damping_pct]
        return d

    def plot_curves(self, ax=None, show=True, **kwargs):
        """Plot G/Gmax and damping curves."""
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            ax1 = ax
            ax2 = ax.twinx()

        if self.strain_pct is not None and self.G_Gmax is not None:
            setup_engineering_plot(ax1, f"{self.model} — Modulus Reduction",
                                  "Shear Strain (%)", "G/Gmax")
            ax1.semilogx(self.strain_pct, self.G_Gmax, 'b-', linewidth=2)
            ax1.set_ylim(0, 1.05)

        if self.strain_pct is not None and self.damping_pct is not None:
            setup_engineering_plot(ax2, f"{self.model} — Damping",
                                  "Shear Strain (%)", "Damping (%)")
            ax2.semilogx(self.strain_pct, self.damping_pct, 'r-', linewidth=2)

        plt.tight_layout()
        if show:
            plt.show()
        return ax1


@dataclass
class VsProfileResult:
    """Results from Vs profile site characterization.

    Attributes
    ----------
    n_layers : int
        Number of soil layers (excluding halfspace).
    vs30 : float
        Time-averaged shear wave velocity in top 30 m (m/s).
    f0_bh : float
        Fundamental frequency from Borcherdt-Hartzell (Hz).
    f0_ro : float
        Fundamental frequency from Roesset method (Hz).
    z1 : float
        Depth to Vs >= 1000 m/s (m). Basin depth proxy.
    z_max : float
        Maximum profile depth (m).
    thicknesses : list of float
        Layer thicknesses (m). Last = 0 for halfspace.
    vs_values : list of float
        Layer Vs values (m/s).
    depth_array : list of float
        Interface depths (m).
    """
    n_layers: int = 0
    vs30: float = 0.0
    f0_bh: float = 0.0
    f0_ro: float = 0.0
    z1: float = 0.0
    z_max: float = 0.0
    thicknesses: List[float] = field(default_factory=list)
    vs_values: List[float] = field(default_factory=list)
    depth_array: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Vs PROFILE SITE CHARACTERIZATION",
            "=" * 60,
            f"  Layers:       {self.n_layers}",
            f"  Vs30:         {self.vs30:.1f} m/s",
            f"  f0 (BH):      {self.f0_bh:.3f} Hz",
            f"  f0 (RO):      {self.f0_ro:.3f} Hz",
            f"  z1:           {self.z1:.1f} m",
            f"  z_max:        {self.z_max:.1f} m",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "vs30": float(self.vs30),
            "f0_bh": float(self.f0_bh),
            "f0_ro": float(self.f0_ro),
            "z1": float(self.z1),
            "z_max": float(self.z_max),
            "thicknesses": [float(x) for x in self.thicknesses],
            "vs_values": [float(x) for x in self.vs_values],
            "depth_array": [float(x) for x in self.depth_array],
        }
