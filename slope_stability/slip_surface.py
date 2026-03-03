"""
Slip surface geometry — circular and polyline (noncircular).

Handles ground intersection, entry/exit points, base inclination
angle alpha, and validation.

Both CircularSlipSurface and PolylineSlipSurface share the same
duck-typed interface:
- slip_elevation_at(x) -> Optional[float]
- tangent_angle_at(x) -> float
- find_entry_exit(geom) -> (x_entry, x_exit)
- is_circular -> bool

References:
    Duncan, Wright & Brandon (2014) — Chapters 6-7
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

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

    @property
    def is_circular(self) -> bool:
        return True

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


@dataclass
class PolylineSlipSurface:
    """A noncircular (polyline) slip surface for limit equilibrium analysis.

    The slip surface is defined by a series of (x, z) points connected
    by straight line segments. This is the natural representation for
    Spencer's method applied to noncircular surfaces.

    Parameters
    ----------
    points : list of (float, float)
        Polyline vertices as (x, z) pairs, sorted left-to-right.
        At least 2 points required. x-coordinates must be monotonically
        increasing.

    References
    ----------
    Duncan, Wright & Brandon (2014) — Chapter 7
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
    """
    points: List[Tuple[float, float]]

    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError(
                f"PolylineSlipSurface requires at least 2 points, "
                f"got {len(self.points)}"
            )
        xs = [p[0] for p in self.points]
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError(
                "PolylineSlipSurface points must have monotonically "
                "increasing x-coordinates"
            )

    @property
    def is_circular(self) -> bool:
        return False

    def slip_elevation_at(self, x: float) -> Optional[float]:
        """Return slip surface elevation at x via linear interpolation.

        Returns None if x is outside the polyline x-range.
        """
        pts = self.points
        if x < pts[0][0] or x > pts[-1][0]:
            return None
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                return z0 + t * (z1 - z0)
        return pts[-1][1]

    def tangent_angle_at(self, x: float) -> float:
        """Base inclination angle alpha at x (radians).

        Returns the angle of the line segment containing x.
        Positive = base slopes upward left-to-right.
        """
        pts = self.points
        # Find the segment containing x
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                dx = x1 - x0
                dz = z1 - z0
                if abs(dx) < 1e-12:
                    return math.pi / 2 if dz > 0 else -math.pi / 2
                return math.atan2(dz, dx)
        # Outside range — use nearest segment
        if x < pts[0][0]:
            dx = pts[1][0] - pts[0][0]
            dz = pts[1][1] - pts[0][1]
        else:
            dx = pts[-1][0] - pts[-2][0]
            dz = pts[-1][1] - pts[-2][1]
        if abs(dx) < 1e-12:
            return math.pi / 2 if dz > 0 else -math.pi / 2
        return math.atan2(dz, dx)

    def find_entry_exit(self, geom: SlopeGeometry) -> Tuple[float, float]:
        """Find x-coordinates where polyline intersects ground surface.

        For polyline surfaces, uses the first and last points that are
        at or near the ground surface. If the polyline endpoints are
        below ground, finds intersection via bisection.

        Returns
        -------
        (x_entry, x_exit) : tuple of float
            x_entry < x_exit.
        """
        pts = self.points
        x_entry = pts[0][0]
        x_exit = pts[-1][0]

        # Refine entry: find where polyline crosses ground surface
        # near the first point
        x_min, x_max = geom.x_range
        search_lo = max(x_min, x_entry)
        search_hi = min(x_max, x_exit)

        def f(x):
            z_ground = geom.ground_elevation_at(x)
            z_slip = self.slip_elevation_at(x)
            if z_slip is None:
                return -999.0
            return z_ground - z_slip

        # Sample for sign changes
        n_sample = 200
        if search_hi <= search_lo:
            return (x_entry, x_exit)

        dx = (search_hi - search_lo) / n_sample
        crossings = []
        x_prev = search_lo
        f_prev = f(x_prev)

        for i in range(1, n_sample + 1):
            x_curr = search_lo + i * dx
            f_curr = f(x_curr)
            if f_prev * f_curr < 0:
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

        if len(crossings) >= 2:
            return (crossings[0], crossings[-1])
        elif len(crossings) == 1:
            # One crossing found — use polyline endpoints
            if crossings[0] < (search_lo + search_hi) / 2:
                return (crossings[0], x_exit)
            else:
                return (x_entry, crossings[0])

        # No crossings — use polyline endpoints directly
        return (x_entry, x_exit)
