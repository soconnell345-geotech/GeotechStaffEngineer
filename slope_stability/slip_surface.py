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

A slip surface is the part of the trial surface below the ground surface,
between where it enters the soil and where it leaves it: the base of ONE
sliding mass. A trial surface that is below the ground over more than one
stretch (it leaves the ground and comes back into it) bounds more than one
mass, so ``find_entry_exit`` refuses it with ``GroundCrossingError`` rather
than taking it from its first crossing to its last (GM-122; see DESIGN.md
"One sliding mass"). A stretch that only touches the ground (its greatest
depth under 1 % of the deepest stretch's) carries no mass and is left out.

References:
    Duncan, Wright & Brandon (2014) — Chapters 6-7
    Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from slope_stability.geometry import SlopeGeometry


# A polyline within this of the ground (m) is on it: it neither starts nor
# ends a stretch below the ground (its ends are usually placed on it).
_ON_GROUND_TOL = 1e-6

# A stretch below the ground whose greatest depth is under this fraction of
# the deepest stretch's only touches the ground: it carries no mass and is
# left out. A circle passing just over a toe dips a fraction of a millimetre
# under the ground in front of it (0.2 mm under 4 m of slope in V-042).
_TOUCH_FRACTION = 0.01


def _stretches_below(samples, crossings, tol=_ON_GROUND_TOL):
    """The stretches (x_from, x_to) where a surface is below the ground.

    ``samples`` are (x, depth below the ground) in x order and
    ``crossings`` the bisected crossings of the ground between them. A
    stretch is a run of samples more than ``tol`` below the ground; a
    sample within ``tol`` of the ground (a surface built through a point on
    it, a toe it passes through) neither starts nor ends one. A stretch
    starts and ends at its crossings, else at its last or first sample on
    the ground. One below the ground at the first or last sample runs out
    of the sampled range there and starts or ends at that sample.
    """
    stretches = []
    run = None
    last_end = -math.inf
    for i, (x, depth) in enumerate(samples):
        if depth > tol:
            if run is None:
                before = [c for c in crossings if last_end < c <= x]
                start = (before[-1] if before
                         else x if i == 0 else samples[i - 1][0])
                run = [start, x]
            run[1] = x
        elif depth < -tol and run is not None:
            after = [c for c in crossings if run[1] <= c <= x]
            end = after[-1] if after else samples[i - 1][0]
            stretches.append((run[0], end))
            last_end = end
            run = None
    if run is not None:
        x_last, depth_last = samples[-1]
        after = [c for c in crossings if run[1] <= c <= x_last]
        end = (x_last if depth_last > tol or not after else after[-1])
        stretches.append((run[0], end))
    return stretches


def _masses(stretches, depth_at, xs_extra=()):
    """The stretches below the ground that carry a mass (GM-122).

    ``depth_at(x)`` is the depth of the surface below the ground at x. A
    stretch whose greatest depth is under ``_TOUCH_FRACTION`` of the
    deepest stretch's is a touch, not a mass, and is dropped. The depth is
    read at 63 points across each stretch and at ``xs_extra`` inside it
    (the ground's and a polyline's vertices).
    """
    depths = []
    for a, b in stretches:
        xs = [a + (b - a) * k / 64 for k in range(1, 64)]
        xs += [x for x in xs_extra if a < x < b]
        depths.append(max(depth_at(x) for x in xs))
    deepest = max(depths)
    return [tuple(s) for s, d in zip(stretches, depths)
            if d >= _TOUCH_FRACTION * deepest]


class GroundCrossingError(ValueError):
    """A trial surface below the ground over more than one stretch (GM-122).

    It leaves the ground and comes back into it, so the soil above it is
    more than one sliding mass and the stretch in the air between them is
    part of none. ``stretches`` lists the (x_from, x_to) stretches below
    the ground. A ``ValueError``, so every caller that refuses an
    inadmissible surface refuses this one; the searches also count it
    (``SearchResult.n_rejected_ground_crossings``).
    """

    def __init__(self, what: str, stretches, x_ends, about: bool = False):
        self.stretches = [(float(a), float(b)) for a, b in stretches]
        x_min, x_max = x_ends
        tol = 1e-6 * max(1.0, abs(x_max - x_min))

        def at(x):
            end = (abs(x - x_min) <= tol or abs(x - x_max) <= tol)
            return f"{x:.2f}" + (" (the end of the profile)" if end else "")

        where = ", ".join(f"x {at(a)} to {at(b)}" for a, b in self.stretches)
        super().__init__(
            f"{what} is below the ground surface over "
            f"{len(self.stretches)} separate stretches "
            f"({'about ' if about else ''}{where}): it leaves the ground and "
            f"comes back into it, so the soil above it is more than one "
            f"sliding mass. A slip surface enters the ground once and leaves "
            f"it once; give a surface that bounds one of these masses (a "
            f"circle through its two ends, or a polyline).")


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
        GroundCrossingError
            If the circle is below the ground over more than one stretch
            (GM-122): it crosses the ground more than twice, or runs out of
            the profile below the ground beside a stretch it closes.
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
        samples = [(x_prev, f_prev)]

        for i in range(1, n_sample + 1):
            x_curr = search_lo + i * dx
            f_curr = f(x_curr)
            samples.append((x_curr, f_curr))
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

        # One sliding mass (GM-122). The stretches below the ground run
        # between crossings, and from the end of the profile where the
        # circle is below the ground there. More than one: the circle leaves
        # the ground and comes back into it (say it dips under the level
        # ground in front of a toe, then rises through the air to the face),
        # and taking it from its first crossing to its last made the stretch
        # in the air slices and two masses one. A stretch that only touches
        # the ground is left out (``_masses``).
        stretches = _stretches_below(samples, crossings)
        if len(stretches) > 1:
            stretches = _masses(stretches, f,
                                [p[0] for p in geom.surface_points])
            if len(stretches) > 1:
                raise GroundCrossingError(
                    f"Circle (xc={self.xc:.2f}, yc={self.yc:.2f}, "
                    f"R={self.radius:.2f})", stretches, (x_min, x_max))

        # Its one stretch below the ground, from where it enters the ground
        # to where it leaves it. Below the ground at an end of the profile,
        # it does neither there: it runs out of the model.
        runs_out = bool(stretches) and (
            (samples[0][1] > _ON_GROUND_TOL
             and stretches[0][0] == samples[0][0])
            or (samples[-1][1] > _ON_GROUND_TOL
                and stretches[0][1] == samples[-1][0]))
        if not stretches or runs_out:
            n = len([c for c in crossings
                     if not stretches
                     or stretches[0][0] <= c <= stretches[0][1]])
            raise ValueError(
                f"Circle (xc={self.xc:.2f}, yc={self.yc:.2f}, R={self.radius:.2f}) "
                f"does not intersect the ground surface at 2 points "
                f"(found {n} crossing(s))"
            )
        return stretches[0]

    def reverse_curvature_crack(self, geom: SlopeGeometry,
                                x_entry: Optional[float] = None,
                                x_exit: Optional[float] = None):
        """The crack closing a "reverse curvature" circle, or None.

        Where the circle's centre lies below the ground at an end of the
        surface, the circle would rise past the centre's level and turn back
        over itself (Slide2's "reverse curvature" surface). Vertical slices
        follow it only up to the centre's level, where the arc is vertical
        and ``find_entry_exit`` ends it, so the surface is closed by a
        vertical crack from there to the ground, dry and without shear —
        as Slide2's default "Create tension crack for reverse curvature"
        closes it. Returns {'x', 'z_ground', 'depth'} or None.
        """
        if x_entry is None or x_exit is None:
            try:
                x_entry, x_exit = self.find_entry_exit(geom)
            except ValueError:
                return None
        tol = 1e-3 * self.radius
        for x_end, x_side in ((x_exit, self.xc + self.radius),
                              (x_entry, self.xc - self.radius)):
            if abs(x_end - x_side) > tol:
                continue
            z_g = geom.ground_elevation_at(x_end)
            depth = z_g - self.yc
            if depth > tol:
                return {"x": x_end, "z_ground": z_g, "depth": depth}
        return None


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

        Raises
        ------
        GroundCrossingError
            If the polyline is below the ground over more than one stretch
            between its ends (GM-122): it rises into the air and comes back
            into the ground.
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
        samples = [(x_prev, f_prev)]

        for i in range(1, n_sample + 1):
            x_curr = search_lo + i * dx
            f_curr = f(x_curr)
            samples.append((x_curr, f_curr))
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

        # One sliding mass (GM-122): below the ground over one stretch
        # between its ends. A point on the ground (within _ON_GROUND_TOL)
        # neither starts nor ends a stretch, and a stretch that only
        # touches the ground is left out (``_masses``).
        stretches = []
        below = False
        for x, fx in samples:
            if fx > _ON_GROUND_TOL:
                if not below:
                    stretches.append([x, x])
                    below = True
                stretches[-1][1] = x
            elif fx < -_ON_GROUND_TOL:
                below = False
        if len(stretches) > 1:
            masses = _masses(
                stretches, f,
                [p[0] for p in geom.surface_points] + [p[0] for p in pts])
            if len(masses) > 1:
                raise GroundCrossingError(
                    f"Polyline ({len(pts)} points, x {pts[0][0]:.2f} to "
                    f"{pts[-1][0]:.2f})", masses, (x_min, x_max), about=True)
            # the one mass: its own crossings, else the polyline's ends
            lo, hi = masses[0]
            left = [c for c in crossings if lo - dx <= c <= lo]
            right = [c for c in crossings if hi <= c <= hi + dx]
            return (left[-1] if left else x_entry,
                    right[0] if right else x_exit)

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
