"""
Circular slip surface geometry.

Handles circle-ground intersection, entry/exit points, base
inclination angle alpha, and validation that the circle actually
passes through the slope.

References:
    Duncan, Wright & Brandon (2014) — Chapter 6
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional

from slope_stability.geometry import SlopeGeometry


@dataclass
class CircularSlipSurface:
    """A circular slip surface for limit equilibrium analysis.

    Parameters
    ----------
    xc : float
        x-coordinate of circle center (m).
    yc : float
        z-coordinate (elevation) of circle center (m).
        Should be above the slope surface.
    radius : float
        Circle radius (m).
    """
    xc: float
    yc: float
    radius: float

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

    def slip_elevation_at(self, x: float) -> Optional[float]:
        """Return slip surface elevation at x (lower arc).

        Returns None if x is outside the circle.
        """
        dx = x - self.xc
        if abs(dx) >= self.radius:
            return None
        dz = math.sqrt(self.radius**2 - dx**2)
        return self.yc - dz

    def tangent_angle_at(self, x: float) -> float:
        """Base inclination angle alpha at x (radians).

        Alpha is the angle of the slip surface tangent to horizontal.
        Positive when the base slopes upward from left to right
        (left portion of circle, where weight drives sliding).

        For the lower arc of a circle centered at (xc, yc):
            dz/dx = (x - xc) / sqrt(R^2 - (x-xc)^2)
            alpha = atan(dz/dx)
        """
        dx = x - self.xc
        r2 = self.radius**2
        dx2 = dx**2
        if dx2 >= r2:
            return math.pi / 2 if dx > 0 else -math.pi / 2
        dz = math.sqrt(r2 - dx2)
        return math.atan(dx / dz)

    def find_entry_exit(self, geom: SlopeGeometry) -> Tuple[float, float]:
        """Find x-coordinates where slip circle intersects ground surface.

        Uses bisection on f(x) = z_ground(x) - z_slip(x).
        Where f(x) > 0, the ground is above the slip surface (inside slope).
        Where f(x) < 0, the ground is below the slip surface (exposed).

        Returns
        -------
        (x_entry, x_exit) : tuple of float
            x_entry < x_exit.

        Raises
        ------
        ValueError
            If circle does not intersect ground surface properly.
        """
        x_min, x_max = geom.x_range
        # Circle horizontal extent
        x_left = self.xc - self.radius
        x_right = self.xc + self.radius

        # Effective search range
        search_lo = max(x_min, x_left)
        search_hi = min(x_max, x_right)

        if search_lo >= search_hi:
            raise ValueError(
                "Circle does not overlap with ground surface x-range"
            )

        def f(x):
            """Positive = ground above slip, negative = ground below slip."""
            z_ground = geom.ground_elevation_at(x)
            z_slip = self.slip_elevation_at(x)
            if z_slip is None:
                return -999.0
            return z_ground - z_slip

        # Sample to find sign changes
        n_sample = 200
        dx = (search_hi - search_lo) / n_sample
        crossings = []
        x_prev = search_lo
        f_prev = f(x_prev)

        for i in range(1, n_sample + 1):
            x_curr = search_lo + i * dx
            f_curr = f(x_curr)
            if f_prev * f_curr < 0:
                # Sign change — bisect to find crossing
                a, b = x_prev, x_curr
                for _ in range(50):
                    mid = (a + b) / 2.0
                    f_mid = f(mid)
                    if f_mid * f(a) < 0:
                        b = mid
                    else:
                        a = mid
                    if b - a < 1e-6:
                        break
                crossings.append((a + b) / 2.0)
            x_prev = x_curr
            f_prev = f_curr

        if len(crossings) < 2:
            raise ValueError(
                f"Circle (xc={self.xc:.2f}, yc={self.yc:.2f}, R={self.radius:.2f}) "
                f"does not intersect the ground surface at 2 points "
                f"(found {len(crossings)} crossing(s))"
            )

        # Use first and last crossings
        return (crossings[0], crossings[-1])
