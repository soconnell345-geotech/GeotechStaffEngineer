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
between two points where it meets the ground: the base of ONE sliding mass.

A circle (GM-343, "Where a circle ends"; DESIGN.md "Where a circle ends").
A circle's trial surface is an arc below the ground between two points
where the circle meets the ground. The meeting points are found from the
ground's segments and corners (``circle_ground_meetings``), not by
sampling, so a toe or any other corner divides the arc. A circle that
meets the ground at more than two points (crossing it, or touching it: a
corner it passes through, or a stretch that dips under the ground by less
than the touch allowance) is split at those points, and each arc with
ground above it is a trial surface of its own (``CircularSlipSurface.arcs``):
a circle carries the two ends of the arc it stands for (``x_start``,
``x_end``), and the slices are cut between them. Nothing is refused for
being one of two masses. A circle given without its ends that bounds one
arc is that arc; one that bounds several has no single pair of ends, and
``find_entry_exit`` says so (``SeveralArcsError``): `analyze_slope` and
the searches score each arc. The family of circles ending on the face:
FHWA NHI-06-088 p. 6-38; a search covering face exits at every height:
GEC 11 p. 9-6. The split itself is the module's ruling (2026-11-13): no
held reference prints a rule for a circle that meets the ground more than
twice (Slide2 refuses it, the rule before, GM-122).

A polyline that is below the ground over more than one stretch (it leaves
the ground and comes back into it) is still refused with
``GroundCrossingError`` (GM-122). A stretch that only touches the ground
(its greatest depth under 1 % of the deepest stretch's) carries no mass
and is left out, for a circle as for a polyline.

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

# The touch allowance: this fraction of the greatest depth of the circle (or
# polyline) below the ground. A stretch below the ground whose greatest depth
# is under it only touches the ground: it carries no mass and is left out.
# For a circle (GM-343) a ground corner the arc passes under by less than it,
# where the depth has a local minimum, is a touch too: a point where an arc
# given by its two ends may end (``CircleMeetings.touched``). A circle given
# without its ends is not split there (`_SPLIT_AT_TOUCHED_CORNERS`). (The
# value is the module's own, older than GM-343; no printed source. Only its
# role changed there.)
_TOUCH_FRACTION = 0.01


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


# Where a circle's side (its vertical tangent, at the centre's level) is the
# end of an arc, the end is taken this fraction of the radius inside it, so
# that the arc has an elevation there (the lower arc is defined strictly
# inside the circle's width).
_SIDE_INSET = 1e-9

# Two meeting points closer than this fraction of the radius are one (a
# corner found from both of its segments), and an arc within it of a corner
# passes through the corner (a circle built through a toe).
_MEET_MERGE = 1e-9

# Whether a circle given without its ends is split at a corner it passes
# under by less than the touch allowance (the other reading of GM-343's
# rule, "touching it within the touch allowance"). Off: measured with it
# on, a search's minimum moved with the allowance (an arc may pass up to
# the allowance under a toe and be ended there, which leaves a small free
# face): the vertical cut read 0.844 where its toe circles read 0.8514
# (Taylor 0.8511), and V-042's entry / exit search 1.3269 for 1.3335
# (printed 1.334). `module_work/gm140/CIRCLE_ENDS_CHECK.md`, "Calls made".
_SPLIT_AT_TOUCHED_CORNERS = False


@dataclass
class CircleMeetings:
    """Where a circle's lower arc meets the ground (GM-343).

    ``arcs``: the (x_from, x_to) arcs below the ground between two meeting
    points, in x order, each a trial surface of its own. ``touches``: the
    stretches left out because they only touch the ground (within the
    touch allowance). ``runs_out``: the stretches below the ground that
    run out of the profile at one of its ends (not between two meeting
    points: not trial surfaces). ``points``: every meeting point's x
    (crossings, corners the arc passes through, and a circle's side where
    the ground is above the centre's level). ``touched``: the (x, depth)
    of each ground corner inside an arc that the arc passes under by less
    than the allowance, at a local minimum of its depth: the arc is not
    split there, but an arc given by its two ends may end there.
    ``allowance``: the touch allowance (m). ``n_crossings``: the points
    where the arc crosses the ground or touches it exactly."""
    arcs: List[Tuple[float, float]]
    touches: List[Tuple[float, float]]
    runs_out: List[Tuple[float, float]]
    points: List[float]
    allowance: float
    n_crossings: int
    touched: List[Tuple[float, float]] = field(default_factory=list)


def circle_ground_meetings(xc: float, yc: float, radius: float,
                           surface_points) -> CircleMeetings:
    """The arcs of a circle below the ground, from the ground's own
    segments and corners (GM-343).

    The ground is the polyline ``surface_points`` and the slip surface the
    circle's lower arc. On each ground segment the circle and the line
    meet where (1 + m^2) u^2 + 2 k m u + k^2 - R^2 = 0, with u = x - xc, m
    the segment's slope and k = z0 - yc + m (xc - x0); the roots on the
    lower arc (k + m u <= 0) inside the segment are meeting points. The
    depth of the arc below the ground, d(x), is concave along a segment
    (the lower arc is convex), so between two neighbouring meeting points
    the arc is wholly below the ground or wholly above it, read at the
    midpoint; d has a local minimum only at a ground corner, and its
    greatest value on a segment is where the arc's slope equals the
    segment's, x = xc + R m / sqrt(1 + m^2).

    The rule: a stretch below the ground between two meeting points is an
    arc. A corner the arc passes through from below (its depth there zero,
    to ``_MEET_MERGE`` of the radius: a circle built through a toe) is a
    meeting point and splits the stretch. A stretch (or part of one) whose
    greatest depth is under the touch allowance (``_TOUCH_FRACTION`` of the
    circle's greatest depth below the ground) is a touch and is left out.
    A corner inside an arc that the arc passes under by less than the
    allowance, at a local minimum of its depth, is listed in ``touched``:
    an arc given by its two ends may end there (`given_arc`); the circle
    is not split there. A stretch that reaches an end of the
    profile below the ground runs out of the model and is not an arc. A
    circle's side inside the profile with the ground above the centre's
    level ends an arc there (the arc is vertical: a "reverse curvature"
    circle, closed by the crack of EM 1110-2-1902 C-10c, `cracks`).
    """
    pts = surface_points
    x_min, x_max = pts[0][0], pts[-1][0]
    lo = max(x_min, xc - radius)
    hi = min(x_max, xc + radius)
    if lo >= hi:
        raise ValueError(
            "Circle does not overlap with ground surface x-range")
    merge = _MEET_MERGE * max(radius, 1.0)
    r2 = radius * radius

    def slip_z(x):
        return yc - math.sqrt(max(r2 - (x - xc) ** 2, 0.0))

    def ground_z(x):
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
            if x0 <= x <= x1:
                return z0 + (x - x0) / (x1 - x0) * (z1 - z0)
        return pts[0][1] if x < x_min else pts[-1][1]

    def depth(x):
        return ground_z(x) - slip_z(x)

    roots = []
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        if x1 < lo or x0 > hi:
            continue
        m = (z1 - z0) / (x1 - x0)
        k = z0 - yc + m * (xc - x0)
        qa = 1.0 + m * m
        disc = (k * m) ** 2 - qa * (k * k - r2)
        if disc < 0.0:
            continue
        root = math.sqrt(disc)
        for u in ((-k * m - root) / qa, (-k * m + root) / qa):
            x = xc + u
            if k + m * u > merge:           # the upper arc
                continue
            if x < max(x0, lo) - merge or x > min(x1, hi) + merge:
                continue
            roots.append(min(max(x, lo), hi))
    roots.sort()
    crossings = []
    for x in roots:
        if not crossings or x - crossings[-1] > merge:
            crossings.append(x)

    # the stretches below the ground, between neighbouring meeting points
    bounds = [lo] + [x for x in crossings
                     if x - lo > merge and hi - x > merge] + [hi]
    side_lo = xc - radius >= x_min      # the circle's side, not the profile's end
    side_hi = xc + radius <= x_max
    stretches = []                      # [a, b, runs out]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= merge or depth(0.5 * (a + b)) <= 0.0:
            continue
        out = ((a == lo and not side_lo and depth(lo) > _ON_GROUND_TOL)
               or (b == hi and not side_hi and depth(hi) > _ON_GROUND_TOL))
        stretches.append([a, b, out])

    def greatest_depth(a, b):
        xs = [a, b] + [p[0] for p in pts if a < p[0] < b]
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
            m = (z1 - z0) / (x1 - x0)
            x = xc + radius * m / math.sqrt(1.0 + m * m)
            if max(a, x0) < x < min(b, x1):
                xs.append(x)
        return max(depth(x) for x in xs)

    if not stretches:
        return CircleMeetings([], [], [], crossings, 0.0, len(crossings))
    allowance = _TOUCH_FRACTION * max(greatest_depth(a, b)
                                      for a, b, _ in stretches)

    # a corner the arc passes through from below (a local minimum of the
    # depth, zero to the merge tolerance) is a meeting point; one it passes
    # under by less than the allowance is a touch, where a given arc may end
    points = list(crossings)
    touched = []
    split_under = allowance if _SPLIT_AT_TOUCHED_CORNERS else merge
    parts = []
    for a, b, out in stretches:
        cuts = [a]
        for i in range(1, len(pts) - 1):
            x, z = pts[i]
            if not (a + merge < x < b - merge):
                continue
            under = z - slip_z(x)
            if not (0.0 <= under < max(allowance, merge)):
                continue
            slope = (x - xc) / math.sqrt(max(r2 - (x - xc) ** 2, 1e-300))
            m_left = (z - pts[i - 1][1]) / (x - pts[i - 1][0])
            m_right = (pts[i + 1][1] - z) / (pts[i + 1][0] - x)
            if m_left <= slope <= m_right:
                if under < split_under:
                    cuts.append(x)
                    points.append(x)
                else:
                    touched.append((x, under))
        cuts.append(b)
        for j, (c, d) in enumerate(zip(cuts[:-1], cuts[1:])):
            parts.append((c, d, out and ((j == 0 and c == lo)
                                         or (j == len(cuts) - 2
                                             and d == hi))))

    arcs, touches, runs_out = [], [], []
    inset = _SIDE_INSET * radius
    for a, b, out in parts:
        if greatest_depth(a, b) < allowance:
            touches.append((a, b))
        elif out:
            runs_out.append((a, b))
        else:
            if side_lo and a == lo:
                a = lo + inset
                points.append(a)
            if side_hi and b == hi:
                b = hi - inset
                points.append(b)
            arcs.append((a, b))
    touched = [(x, d) for x, d in touched
               if any(a < x < b for a, b in arcs)]
    return CircleMeetings(arcs, touches, runs_out, sorted(set(points)),
                          allowance, len(crossings), touched)


class SeveralArcsError(ValueError):
    """A circle given without its ends that bounds more than one arc below
    the ground (GM-343): it has no single entry and exit. ``arcs`` lists
    the (x_from, x_to) arcs. Not a refusal of the circle: `analyze_slope`
    and the searches score each arc (``CircularSlipSurface.arcs``); this is
    raised only where one arc is needed and none was named."""

    def __init__(self, circle, arcs):
        self.arcs = [(float(a), float(b)) for a, b in arcs]
        where = ", ".join(f"x {a:.2f} to {b:.2f}" for a, b in self.arcs)
        super().__init__(
            f"Circle (xc={circle.xc:.2f}, yc={circle.yc:.2f}, "
            f"R={circle.radius:.2f}) meets the ground at more than two "
            f"points and bounds {len(self.arcs)} arcs below it ({where}). "
            f"Each arc is a trial surface of its own: name the one meant "
            f"(CircularSlipSurface(xc, yc, radius, x_start, x_end), or one "
            f"of circle.arcs(geom)).")


def describe_arc(slip, meetings: "CircleMeetings") -> Optional[dict]:
    """Which arc of its circle a bound circle is, for a result (GM-343):
    {'x_start', 'x_end', 'index' (1-based, in x order), 'n_arcs',
    'arcs' (every arc of the circle), 'description'}."""
    if slip.x_start is None or slip.x_end is None:
        return None
    arcs = list(meetings.arcs)
    index = None
    for i, (a, b) in enumerate(arcs):
        if abs(a - slip.x_start) <= 1e-6 and abs(b - slip.x_end) <= 1e-6:
            index = i + 1
    n = len(arcs)
    where = f"x {slip.x_start:.3f} to {slip.x_end:.3f}"
    if index is None:
        text = (f"the arc from {where}, its ends as given: an end is at a "
                f"ground corner the circle passes just under, within the "
                f"touch allowance of {meetings.allowance:.3f} m")
    elif n == 1:
        text = f"the circle's one arc below the ground, {where}"
    else:
        order = ("lowest-x" if index == 1 else
                 "highest-x" if index == n else f"number {index} in x")
        text = (f"arc {index} of the {n} arcs this circle bounds below the "
                f"ground (the {order} mass), {where}")
    return {"x_start": float(slip.x_start), "x_end": float(slip.x_end),
            "index": index, "n_arcs": n,
            "arcs": [(float(a), float(b)) for a, b in arcs],
            "description": text}


def given_arc(slip, geom) -> "CircularSlipSurface":
    """A circle whose two ends were given, with each end set on the point
    where the circle meets the ground (GM-343).

    Each given end must lie within a thousandth of the radius (a result's
    ends may have been rounded) of an end of one of the circle's arcs, or
    of a ground corner inside that arc which the arc passes under by less
    than the touch allowance (``CircleMeetings.touched``: a touch within
    the allowance counts as a meeting point, so Taylor's toe circle
    written to two decimals, which passes 7 mm under the toe, can be
    ended at the toe). Both ends must belong to the same arc, so the
    ground is above the surface all the way between them. Anything else
    is refused with the circle's own arcs listed: ends placed anywhere
    else would cut the sliding mass with a vertical face no method was
    told of."""
    found = slip.meetings(geom)
    tol = 1e-3 * slip.radius
    for a, b in found.arcs:
        ends = [a] + [x for x, _ in found.touched if a < x < b] + [b]
        lo = [x for x in ends if abs(x - slip.x_start) <= tol]
        hi = [x for x in ends if abs(x - slip.x_end) <= tol]
        if lo and hi and lo[0] < hi[0]:
            return slip.arc(min(lo, key=lambda x: abs(x - slip.x_start)),
                            min(hi, key=lambda x: abs(x - slip.x_end)))
    where = ", ".join(f"x {a:.3f} to {b:.3f}" for a, b in found.arcs)
    at = ", ".join(f"x {x:.3f}" for x, _ in found.touched)
    raise ValueError(
        f"Circle (xc={slip.xc:.2f}, yc={slip.yc:.2f}, R={slip.radius:.2f}): "
        f"the ends given (x {slip.x_start:.3f} to {slip.x_end:.3f}) are not "
        f"the two ends of an arc of this circle below the ground. Its "
        f"arcs: {where or 'none'}"
        + (f"; an arc may also end at the ground corner it passes just "
           f"under at {at}" if at else "") + ".")


def circle_of(result) -> "CircularSlipSurface":
    """The circle a result reports, standing for the arc it reports
    (``SlopeStabilityResult.arc``; GM-343), so that a later analysis of
    the governing surface keeps its two ends. A result without ``arc`` is
    a circle with one arc: the circle alone."""
    arc = getattr(result, "arc", None)
    if arc:
        return CircularSlipSurface(result.xc, result.yc, result.radius,
                                   arc["x_start"], arc["x_end"])
    return CircularSlipSurface(result.xc, result.yc, result.radius)


class GroundCrossingError(ValueError):
    """A polyline below the ground over more than one stretch (GM-122).

    (A circle is no longer refused for it: it is split at the points where
    it meets the ground and each arc is a trial surface, GM-343.)

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
    x_start, x_end : float, optional
        The two ends of the arc this circle stands for (GM-343): the x of
        the two points where the arc meets the ground, ``x_start`` <
        ``x_end``. Both or neither. Left out, the circle stands for its one
        arc below the ground; where it bounds several (it meets the ground
        at more than two points), `analyze_slope` and the searches score
        each (``arcs``).
    """
    xc: float
    yc: float
    radius: float
    x_start: Optional[float] = None
    x_end: Optional[float] = None

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
        if (self.x_start is None) != (self.x_end is None):
            raise ValueError(
                "Give both ends of the arc (x_start and x_end) or neither")
        if self.x_start is not None:
            if not self.x_start < self.x_end:
                raise ValueError(
                    f"The arc's ends must be in order: x_start "
                    f"{self.x_start} is not below x_end {self.x_end}")
            if (self.x_start <= self.xc - self.radius
                    or self.x_end >= self.xc + self.radius):
                raise ValueError(
                    f"The arc's ends (x {self.x_start} to {self.x_end}) "
                    f"must lie inside the circle's width "
                    f"(x {self.xc - self.radius} to "
                    f"{self.xc + self.radius})")

    @property
    def is_bound(self) -> bool:
        """Whether the circle carries the two ends of its arc."""
        return self.x_start is not None

    def meetings(self, geom: SlopeGeometry) -> "CircleMeetings":
        """Where the circle meets the ground (``circle_ground_meetings``)."""
        return circle_ground_meetings(self.xc, self.yc, self.radius,
                                      geom.surface_points)

    def arc(self, x_start: float, x_end: float) -> "CircularSlipSurface":
        """This circle standing for the arc between two ends."""
        return CircularSlipSurface(self.xc, self.yc, self.radius,
                                   x_start, x_end)

    def arcs(self, geom: SlopeGeometry) -> List["CircularSlipSurface"]:
        """The circle's trial surfaces (GM-343): one circle per arc below
        the ground between two points where it meets the ground, each
        carrying its two ends, in x order. A circle that already carries
        its ends is its own one arc. Empty where no arc lies below the
        ground between two meeting points."""
        if self.is_bound:
            return [self]
        return [self.arc(a, b) for a, b in self.meetings(geom).arcs]

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
        """The two ends of the circle's arc: where it meets the ground.

        A circle that carries its ends returns them. One that does not
        returns the ends of its one arc below the ground, found from the
        ground's segments and corners (``circle_ground_meetings``, GM-343;
        before, by sampling at 200 points and bisection, which stepped
        over a toe a circle passed a hair above).

        Returns
        -------
        (x_entry, x_exit) : tuple of float
            x_entry < x_exit.

        Raises
        ------
        SeveralArcsError
            The circle carries no ends and bounds more than one arc: each
            is a trial surface of its own (``arcs``), and none was named.
        ValueError
            No arc of the circle lies below the ground between two points
            where it meets the ground (it misses the ground, or runs out
            of the profile below it).
        """
        if self.x_start is not None:
            return (self.x_start, self.x_end)
        found = self.meetings(geom)
        if len(found.arcs) > 1:
            raise SeveralArcsError(self, found.arcs)
        if not found.arcs:
            raise ValueError(
                f"Circle (xc={self.xc:.2f}, yc={self.yc:.2f}, R={self.radius:.2f}) "
                f"does not intersect the ground surface at 2 points "
                f"(found {found.n_crossings} crossing(s))"
            )
        return found.arcs[0]

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
