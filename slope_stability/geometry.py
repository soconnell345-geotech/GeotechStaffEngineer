"""
Slope geometry and soil layer definitions for limit equilibrium analysis.

Slope is defined by a ground surface profile (x, z) and soil layers
by elevation ranges. All units SI: meters, kPa, kN/m3, degrees.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
    Abramson et al. (2002) — Slope Stability and Stabilization Methods
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SlopeSoilLayer:
    """A soil layer within the slope, defined by elevation boundaries.

    Parameters
    ----------
    name : str
        Layer identifier / description.
    top_elevation : float
        Top elevation of layer (m). Higher = up.
    bottom_elevation : float
        Bottom elevation of layer (m).
    gamma : float
        Total unit weight (kN/m3).
    gamma_sat : float, optional
        Saturated unit weight (kN/m3). If None, uses gamma.
    phi : float
        Effective friction angle (degrees). For drained analysis.
    c_prime : float
        Effective cohesion (kPa). For drained analysis.
    cu : float
        Undrained shear strength (kPa). For undrained (phi=0) analysis.
    analysis_mode : str
        'drained' (uses c', phi') or 'undrained' (uses cu, phi=0).
        Default 'drained'.
    """
    name: str
    top_elevation: float
    bottom_elevation: float
    gamma: float
    gamma_sat: Optional[float] = None
    phi: float = 0.0
    c_prime: float = 0.0
    cu: float = 0.0
    analysis_mode: str = "drained"

    def __post_init__(self):
        if self.bottom_elevation >= self.top_elevation:
            raise ValueError(
                f"Layer '{self.name}': bottom_elevation ({self.bottom_elevation}) "
                f"must be less than top_elevation ({self.top_elevation})"
            )
        if self.gamma <= 0:
            raise ValueError(
                f"Layer '{self.name}': gamma must be positive, got {self.gamma}"
            )
        if self.gamma_sat is None:
            self.gamma_sat = self.gamma
        if self.analysis_mode not in ("drained", "undrained"):
            raise ValueError(
                f"analysis_mode must be 'drained' or 'undrained', "
                f"got '{self.analysis_mode}'"
            )
        if self.analysis_mode == "drained":
            if self.phi < 0:
                raise ValueError(f"phi must be non-negative, got {self.phi}")
            if self.c_prime < 0:
                raise ValueError(f"c_prime must be non-negative, got {self.c_prime}")
        else:
            if self.cu < 0:
                raise ValueError(f"cu must be non-negative, got {self.cu}")

    @property
    def thickness(self) -> float:
        """Layer thickness (m)."""
        return self.top_elevation - self.bottom_elevation

    @property
    def shear_strength_params(self) -> Tuple[float, float]:
        """Return (cohesion, phi) depending on analysis mode.

        For drained: (c', phi')
        For undrained: (cu, 0)
        """
        if self.analysis_mode == "undrained":
            return (self.cu, 0.0)
        return (self.c_prime, self.phi)


@dataclass
class SlopeGeometry:
    """Complete slope definition: ground surface + soil layers + water.

    Parameters
    ----------
    surface_points : list of (float, float)
        Ground surface profile as (x, z) coordinates, left to right.
        z = elevation. Must have at least 2 points.
    soil_layers : list of SlopeSoilLayer
        Soil layers sorted by decreasing top_elevation.
    gwt_points : list of (float, float), optional
        Groundwater table as (x, z_gwt) points. None = no water.
    surcharge : float
        Uniform vertical surcharge on slope surface (kPa). Default 0.
    surcharge_x_range : tuple of (float, float), optional
        (x_start, x_end) over which surcharge is applied.
        None = entire surface.
    reinforcement_force : float
        Horizontal reinforcement force (kN/m). Default 0.
    reinforcement_elevation : float, optional
        Elevation at which reinforcement force acts (m).
    kh : float
        Horizontal seismic coefficient. Default 0 (no seismic).
    """
    surface_points: List[Tuple[float, float]]
    soil_layers: List[SlopeSoilLayer]
    gwt_points: Optional[List[Tuple[float, float]]] = None
    surcharge: float = 0.0
    surcharge_x_range: Optional[Tuple[float, float]] = None
    reinforcement_force: float = 0.0
    reinforcement_elevation: Optional[float] = None
    kh: float = 0.0

    def __post_init__(self):
        if len(self.surface_points) < 2:
            raise ValueError("Surface must have at least 2 points")
        xs = [p[0] for p in self.surface_points]
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("Surface points must be sorted left-to-right by x")
        if len(self.soil_layers) == 0:
            raise ValueError("At least one soil layer is required")
        if self.kh < 0:
            raise ValueError(f"kh must be non-negative, got {self.kh}")
        if self.surcharge < 0:
            raise ValueError(f"surcharge must be non-negative, got {self.surcharge}")

    def ground_elevation_at(self, x: float) -> float:
        """Linearly interpolate ground surface elevation at x.

        Extrapolates as constant beyond surface extent.
        """
        pts = self.surface_points
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return z0 + t * (z1 - z0)
        return pts[-1][1]

    def gwt_elevation_at(self, x: float) -> Optional[float]:
        """Linearly interpolate GWT elevation at x. None if no GWT."""
        if self.gwt_points is None:
            return None
        pts = self.gwt_points
        if len(pts) == 0:
            return None
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return z0 + t * (z1 - z0)
        return pts[-1][1]

    def layer_at_elevation(self, z: float) -> Optional[SlopeSoilLayer]:
        """Return the soil layer that contains elevation z.

        Searches from top to bottom. Returns None if z is outside
        all layers.
        """
        for layer in self.soil_layers:
            if layer.bottom_elevation <= z <= layer.top_elevation:
                return layer
        return None

    @property
    def x_range(self) -> Tuple[float, float]:
        """(x_min, x_max) of the ground surface."""
        return (self.surface_points[0][0], self.surface_points[-1][0])

    @property
    def z_range(self) -> Tuple[float, float]:
        """(z_min, z_max) of the ground surface."""
        zs = [p[1] for p in self.surface_points]
        return (min(zs), max(zs))

    def surcharge_at(self, x: float) -> float:
        """Return surcharge pressure at x (kPa)."""
        if self.surcharge <= 0:
            return 0.0
        if self.surcharge_x_range is None:
            return self.surcharge
        x_lo, x_hi = self.surcharge_x_range
        if x_lo <= x <= x_hi:
            return self.surcharge
        return 0.0
