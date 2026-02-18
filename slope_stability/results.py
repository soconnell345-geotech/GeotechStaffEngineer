"""
Result containers for slope stability analysis.

Each dataclass stores analysis outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: meters, kPa, kN/m, degrees.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class SliceData:
    """Per-slice data for detailed output and plotting.

    Attributes
    ----------
    x_mid : float
        x-coordinate of slice midpoint (m).
    z_top : float
        Ground surface elevation at midpoint (m).
    z_base : float
        Slip surface elevation at midpoint (m).
    width : float
        Slice width b (m).
    height : float
        Slice height h (m).
    alpha_deg : float
        Base inclination angle (degrees).
    weight : float
        Total slice weight W (kN/m).
    pore_pressure : float
        Pore water pressure at base u (kPa).
    c : float
        Cohesion at base (kPa).
    phi : float
        Friction angle at base (degrees).
    base_length : float
        Length of slice base dl (m).
    """
    x_mid: float = 0.0
    z_top: float = 0.0
    z_base: float = 0.0
    width: float = 0.0
    height: float = 0.0
    alpha_deg: float = 0.0
    weight: float = 0.0
    pore_pressure: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    base_length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_mid_m": round(self.x_mid, 3),
            "z_top_m": round(self.z_top, 3),
            "z_base_m": round(self.z_base, 3),
            "width_m": round(self.width, 3),
            "height_m": round(self.height, 3),
            "alpha_deg": round(self.alpha_deg, 2),
            "weight_kN_per_m": round(self.weight, 1),
            "pore_pressure_kPa": round(self.pore_pressure, 1),
            "c_kPa": round(self.c, 1),
            "phi_deg": round(self.phi, 1),
            "base_length_m": round(self.base_length, 3),
        }


@dataclass
class SlopeStabilityResult:
    """Results from slope stability analysis.

    Attributes
    ----------
    FOS : float
        Factor of safety.
    method : str
        Analysis method ('Fellenius', 'Bishop', 'Spencer').
    xc : float
        Circle center x-coordinate (m).
    yc : float
        Circle center z-coordinate / elevation (m).
    radius : float
        Circle radius (m).
    x_entry : float
        Slip surface entry x-coordinate (m).
    x_exit : float
        Slip surface exit x-coordinate (m).
    is_stable : bool
        True if FOS >= FOS_required.
    FOS_required : float
        Minimum required FOS.
    theta_spencer : float or None
        Interslice force angle from Spencer (degrees).
    FOS_fellenius : float or None
        Fellenius FOS for comparison.
    FOS_bishop : float or None
        Bishop FOS for comparison.
    n_slices : int
        Number of slices used.
    has_seismic : bool
        Whether seismic load was applied.
    kh : float
        Horizontal seismic coefficient.
    slice_data : list of SliceData or None
        Per-slice breakdown.
    """
    FOS: float = 0.0
    method: str = ""
    xc: float = 0.0
    yc: float = 0.0
    radius: float = 0.0
    x_entry: float = 0.0
    x_exit: float = 0.0
    is_stable: bool = False
    FOS_required: float = 1.5
    theta_spencer: Optional[float] = None
    FOS_fellenius: Optional[float] = None
    FOS_bishop: Optional[float] = None
    n_slices: int = 0
    has_seismic: bool = False
    kh: float = 0.0
    slice_data: Optional[List[SliceData]] = None

    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "UNSTABLE"
        lines = [
            "=" * 60,
            "  SLOPE STABILITY ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Method:           {self.method}",
            f"  Factor of Safety: {self.FOS:.3f}  [{status}]",
            f"  Required FOS:     {self.FOS_required:.2f}",
            "",
            f"  Critical Circle:",
            f"    Center (xc, yc) = ({self.xc:.2f}, {self.yc:.2f}) m",
            f"    Radius R        = {self.radius:.2f} m",
            f"    Entry x         = {self.x_entry:.2f} m",
            f"    Exit x          = {self.x_exit:.2f} m",
            "",
            f"  Slices: {self.n_slices}",
        ]
        if self.has_seismic:
            lines.append(f"  Seismic kh:       {self.kh:.3f}")
        if self.FOS_fellenius is not None:
            lines.append(f"  FOS (Fellenius):  {self.FOS_fellenius:.3f}")
        if self.FOS_bishop is not None and self.method != "Bishop":
            lines.append(f"  FOS (Bishop):     {self.FOS_bishop:.3f}")
        if self.theta_spencer is not None:
            lines.append(f"  Spencer theta:    {self.theta_spencer:.2f} deg")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "FOS": round(self.FOS, 4),
            "method": self.method,
            "is_stable": self.is_stable,
            "FOS_required": self.FOS_required,
            "xc_m": round(self.xc, 2),
            "yc_m": round(self.yc, 2),
            "radius_m": round(self.radius, 2),
            "x_entry_m": round(self.x_entry, 2),
            "x_exit_m": round(self.x_exit, 2),
            "n_slices": self.n_slices,
            "has_seismic": self.has_seismic,
            "kh": self.kh,
        }
        if self.FOS_fellenius is not None:
            d["FOS_fellenius"] = round(self.FOS_fellenius, 4)
        if self.FOS_bishop is not None:
            d["FOS_bishop"] = round(self.FOS_bishop, 4)
        if self.theta_spencer is not None:
            d["theta_spencer_deg"] = round(self.theta_spencer, 2)
        if self.slice_data is not None:
            d["slice_data"] = [s.to_dict() for s in self.slice_data]
        return d


@dataclass
class SearchResult:
    """Results from a critical surface search.

    Attributes
    ----------
    critical : SlopeStabilityResult
        Result for the critical (minimum FOS) surface.
    n_surfaces_evaluated : int
        Total number of trial surfaces evaluated.
    grid_fos : list of dict
        Grid of {xc, yc, R, FOS} for all evaluated surfaces.
    """
    critical: Optional[SlopeStabilityResult] = None
    n_surfaces_evaluated: int = 0
    grid_fos: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CRITICAL SURFACE SEARCH RESULTS",
            "=" * 60,
            "",
            f"  Surfaces evaluated: {self.n_surfaces_evaluated}",
        ]
        if self.critical:
            lines.append(f"  Minimum FOS:       {self.critical.FOS:.3f}")
            lines.append(f"  Method:            {self.critical.method}")
            lines.append(
                f"  Critical circle:   "
                f"({self.critical.xc:.2f}, {self.critical.yc:.2f}), "
                f"R={self.critical.radius:.2f} m"
            )
            status = "STABLE" if self.critical.is_stable else "UNSTABLE"
            lines.append(f"  Assessment:        {status}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def plot_fos_contour(self, ax=None, show=True, **kwargs):
        """Plot FOS at each trial circle center as a scatter/contour.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.grid_fos:
            raise ValueError("No grid FOS data available for plotting.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        xc_vals = [g['xc'] for g in self.grid_fos]
        yc_vals = [g['yc'] for g in self.grid_fos]
        fos_vals = [g['FOS'] for g in self.grid_fos]
        sc = ax.scatter(xc_vals, yc_vals, c=fos_vals, cmap='RdYlGn',
                        edgecolors='black', linewidth=0.5, s=60, **kwargs)
        plt.colorbar(sc, ax=ax, label='Factor of Safety')
        if self.critical:
            ax.plot(self.critical.xc, self.critical.yc, 'r*', markersize=15,
                    markeredgecolor='black', label=f'Critical (FOS={self.critical.FOS:.3f})')
            ax.legend()
        setup_engineering_plot(ax, "Critical Surface Search",
                              "Circle Center X (m)", "Circle Center Y (m)")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        d = {"n_surfaces_evaluated": self.n_surfaces_evaluated}
        if self.critical:
            d["critical"] = self.critical.to_dict()
        return d
