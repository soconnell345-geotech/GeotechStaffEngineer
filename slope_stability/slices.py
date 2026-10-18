"""
Slice discretization for method of slices.

Cuts the soil mass between the slip surface entry and exit into
vertical slices and computes all geometric and force quantities
for each slice.

Where the slices are cut (GM-131, GM-134). Every slip surface has a slice
boundary at every break in the ground surface, every point where it
crosses a layer boundary (but the model's base, below) or the water table
(Slide2's "Slice at piezo-surface intersection"), each edge of a
surcharge and the edge of a tension crack, and a noncircular (polyline)
surface also at every corner, so that the base of each slice lies in one
material, under one water table, and its top is one straight segment
under one load. The stretches between those boundaries share out about
``n_slices`` slices in proportion to their width, at least one each.

Each slice, on a circle as on a polyline, takes its base angle and base
length from the chord between its two boundaries on the slip surface
(on a polyline the chord IS the surface); its base elevation stays on
the surface at its midpoint, so its weight is the area under the arc.
On a circle the chord is the straight base the methods assume (Duncan,
Wright & Brandon, Ch. 6): its perpendicular bisector passes through the
centre, so the base normal force acts through the centre, and its length
is the b / cos(alpha) that Bishop's c' b / m_alpha takes. Bishop's
driving moment, W (x_mid - xc) / R, does not use alpha. The circle's
tangent at each slice's midpoint, used before GM-134, gives segments
that do not join: where the arc turns vertical (a circle closed at its
vertical tangent, a reverse-curvature end, a vertical cut) the last
slice's base, width / cos(alpha), falls short of the arc by about a
factor of the square root of two however thin the slice is, and the FOS
converged only as one over the square root of the count (4.7 % low at
30 slices on one toe circle, 0.4 % at 3,840). The chord converges as one
over the count. Elsewhere the two agree within about 0.3 %.

Cut at an even spacing instead, a slice across a corner took one
segment's angle for its whole weight, and a polyline's FOS jumped with
the slice count (0.65 at 30 slices, 1.18 at 31, 0.97 with a boundary at
every corner, on one Taylor surface); a circle's slice across a layer
crossing took one material for its whole base (V-043's circle 1.246,
1.190, 1.180 and 1.232 on 25, 29, 30 and 31 slices; with the boundaries
1.208 to 1.211 on 25 to 45 slices, 1.213 on 3,840, Slide2 1.209).

The model's base (GM-128). A surface may touch the bottom of the deepest
layer, Taylor's firm base, but not pass under it: it would pass through
ground the model does not have. Whether it does is measured on the
surface itself, not on its slices (``base_depth``: the greatest depth of
the surface under that bottom, exact for a circle and a polyline), so the
answer does not depend on the slice count. A surface within
``base_tolerance`` of the bottom, a thousandth of the model's height,
touches it and is kept with every slice; one that reaches further is
refused at any count.

References:
    Duncan, Wright & Brandon (2014) — Chapter 6 (straight slice bases;
    the normal force on a circle's base acts through its centre)
    Rocscience, Slide2 help: slice data ("Slide2 uses the mid-point of a
    slice base to calculate the length of the moment arm"); Advanced project settings ("Slice at
    piezo-surface intersection"); Sarma non-vertical slice method
    ("Slice boundaries will be located at each vertex of a failure
    surface"); error -116, "Not enough slices to analyze the surface"
    (more crossings of material boundaries than slices)
"""

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from geotech_common.water import GAMMA_W
from slope_stability.geometry import (
    SlopeGeometry, SlopeSoilLayer, build_pore_pressure_interpolator,
)


@dataclass
class Slice:
    """Data for a single vertical slice.

    All forces per unit width of slope (kN/m).

    Attributes
    ----------
    x_left : float
        x-coordinate of left side of slice.
    x_right : float
        x-coordinate of right side of slice.
    x_mid : float
        x-coordinate of slice midpoint.
    width : float
        Slice width b (m).
    z_top : float
        Ground surface elevation at slice midpoint.
    z_base : float
        Slip surface elevation at slice midpoint.
    height : float
        Slice height h = z_top - z_base (m).
    alpha : float
        Inclination of slice base to horizontal (radians): the chord's
        between the slice's two boundaries on the slip surface.
    base_length : float
        Length of the slice base, that chord, dl = width / cos(alpha) (m).
    weight : float
        Total weight W of the slice (kN/m).
    pore_pressure : float
        Average pore pressure u on slice base (kPa).
    c : float
        Cohesion (c' or cu) at slice base (kPa).
    phi : float
        Friction angle (phi' or 0) at slice base (degrees).
    surcharge_force : float
        Vertical surcharge force on slice (kN/m).
    seismic_force : float
        Horizontal seismic force = kh * W (kN/m).
    z_centroid : float
        Elevation of slice centroid (for seismic moment arms).
    """
    x_left: float = 0.0
    x_right: float = 0.0
    x_mid: float = 0.0
    width: float = 0.0
    z_top: float = 0.0
    z_base: float = 0.0
    height: float = 0.0
    alpha: float = 0.0
    base_length: float = 0.0
    weight: float = 0.0
    pore_pressure: float = 0.0
    c: float = 0.0
    phi: float = 0.0
    surcharge_force: float = 0.0
    seismic_force: float = 0.0
    z_centroid: float = 0.0
    in_tension_crack: bool = False
    crack_water_force: float = 0.0
    crack_water_z: float = 0.0
    pond_force: float = 0.0      # vertical pond water weight (also in weight)
    pond_hforce: float = 0.0     # signed horizontal pond thrust, +x (kN/m)
    pond_z: float = 0.0          # elevation of pond thrust line of action


@dataclass
class SliceForces:
    """Computed forces on a single slice (Fellenius decomposition).

    All forces per unit width of slope (kN/m).

    Attributes
    ----------
    W : float
        Total weight including surcharge (kN/m).
    N_prime : float
        Effective normal force on base = W*cos(alpha) - U (kN/m).
        Clamped to zero if tensile.
    S_mobilized : float
        Driving shear = W*sin(alpha) (kN/m).
    T_available : float
        Available shear resistance = c*dl + N'*tan(phi) (kN/m).
    U : float
        Pore water force on base = u*dl (kN/m).
    surcharge : float
        Vertical surcharge force (kN/m).
    seismic : float
        Horizontal seismic force (kN/m).
    alpha_deg : float
        Base inclination angle (degrees).
    """
    W: float
    N_prime: float
    S_mobilized: float
    T_available: float
    U: float
    surcharge: float
    seismic: float
    alpha_deg: float


def compute_slice_forces(s: Slice) -> SliceForces:
    """Compute Fellenius-style forces acting on a single slice.

    Parameters
    ----------
    s : Slice
        Slice with geometry and strength data from build_slices().

    Returns
    -------
    SliceForces
        All computed forces on the slice.
    """
    alpha_rad = s.alpha
    alpha_deg = math.degrees(alpha_rad)

    W = s.weight + s.surcharge_force
    U = s.pore_pressure * s.base_length
    N_prime = W * math.cos(alpha_rad) - U
    if N_prime < 0.0:
        N_prime = 0.0

    S_mobilized = W * math.sin(alpha_rad)
    phi_rad = math.radians(s.phi)
    T_available = s.c * s.base_length + N_prime * math.tan(phi_rad)

    return SliceForces(
        W=W,
        N_prime=N_prime,
        S_mobilized=S_mobilized,
        T_available=T_available,
        U=U,
        surcharge=s.surcharge_force,
        seismic=s.seismic_force,
        alpha_deg=alpha_deg,
    )


# A break in the ground or a crossing closer than this fraction of the
# surface's horizontal span to another boundary is merged into it: a layer
# boundary through a corner, a ground break at a corner, rounding. A corner
# of the surface is never merged away, however close: the slice across it
# would take its chord across the corner (a 4 m step 1 mm wide at the entry
# of a 38 m surface gave a 1.25 m slice at -74.5 deg and FOS 2.139 on 30
# slices, 2.259 on 60, against 2.520 on both with the corner kept). (Slide2
# refuses a surface whose average slice width is under 1e-4 of the soil
# region's width, its error -106.)
_BOUNDARY_MERGE = 1e-4

# A corner this close (m) to the entry or exit is that end: find_entry_exit
# bisects to 1e-6 m, so a vertex placed on the ground at an end comes back
# a hair inside it.
_END_MERGE = 1e-6


def _crossings(f, xs: List[float], tol: float) -> List[float]:
    """Where the piecewise-linear f, linear between the sorted ``xs``,
    crosses zero strictly inside (xs[0], xs[-1]): each sign change between
    neighbours by linear interpolation, and each interior x where |f| <=
    ``tol`` (a surface running along a boundary gives its two ends)."""
    vals = [f(x) for x in xs]
    signs = [0 if v is None or abs(v) <= tol else (1 if v > 0 else -1)
             for v in vals]
    out = []
    for i in range(len(xs) - 1):
        if 0 < i and signs[i] == 0 and vals[i] is not None:
            out.append(xs[i])
        if signs[i] * signs[i + 1] < 0:
            v0, v1 = vals[i], vals[i + 1]
            out.append(xs[i] + (xs[i + 1] - xs[i]) * v0 / (v0 - v1))
    return out


def _circle_crossings(slip, xs: List[float], z_line) -> List[float]:
    """Where the lower arc of the circle ``slip`` crosses ``z_line``, which
    is straight between the sorted ``xs``: on each piece, the roots of the
    circle-line quadratic in u = x - xc, (1 + m^2) u^2 + 2 m d u + d^2 -
    R^2 = 0 with the line z - yc = d + m u, kept where the line lies below
    the centre (the lower arc). A line that only touches the circle gives
    its one point."""
    xc, yc, r = slip.xc, slip.yc, slip.radius
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        if b <= a:
            continue
        za, zb = z_line(a), z_line(b)
        m = (zb - za) / (b - a)
        d = za - yc + m * (xc - a)
        k = 1.0 + m * m
        disc = k * r * r - d * d
        if disc < 0.0:
            continue
        sq = math.sqrt(disc)
        for u in ((-m * d - sq) / k, (-m * d + sq) / k):
            x = xc + u
            if a <= x <= b and d + m * u <= 0.0:
                out.append(x)
    return out


def _crack_base(geom: SlopeGeometry, x_entry: float,
                x_exit: float) -> Optional[float]:
    """Elevation of the tension crack's base at the crest end of the
    surface (the entry, or the exit with ``tension_crack_side='exit'``),
    or None without a crack. A slice on that half of the surface whose
    base lies above it is in the crack (``build_slices``)."""
    if geom.tension_crack_depth <= 0:
        return None
    x_crest = x_entry if geom.tension_crack_side == "entry" else x_exit
    return geom.ground_elevation_at(x_crest) - geom.tension_crack_depth


def _surcharge_edges(geom: SlopeGeometry) -> List[float]:
    """The x of each end of each loaded surcharge zone."""
    xs = []
    if geom.surcharge > 0 and geom.surcharge_x_range is not None:
        xs += [float(x) for x in geom.surcharge_x_range]
    for p, x0, x1 in geom.surcharges or ():
        if p > 0:
            xs += [float(x0), float(x1)]
    return xs


def slice_breaks(geom: SlopeGeometry, slip, x_entry: float,
                 x_exit: float) -> List[float]:
    """The x, strictly between the entry and the exit, where a slice
    boundary must fall besides a polyline's corners (GM-131, GM-134):
    each break in the ground surface; each crossing of a layer's top or
    bottom (but the model's base, GM-128), of the water table and of the
    tension crack's base on the crest half of the surface (with the middle
    of the surface, where that half ends, when the crack reaches it); and
    each edge of a surcharge. A polyline is straight between its corners,
    and each line here
    between its vertices, so the crossings are exact; a circle's are the
    roots of the circle-line quadratic on each straight piece."""
    span = x_exit - x_entry
    circular = getattr(slip, "is_circular", True)

    def inside(xs):
        return [x for x in xs if x_entry < x < x_exit]

    corners = ([] if circular else
               inside(p[0] for p in (getattr(slip, "points", []) or [])))

    def crossings_of(vxs, z_line):
        xs = sorted({x_entry, x_exit, *corners, *inside(vxs)})
        if circular:
            return _circle_crossings(slip, xs, z_line)

        def diff(x):
            z = slip.slip_elevation_at(x)
            return None if z is None else z - z_line(x)

        return _crossings(diff, xs, 1e-9 * max(1.0, span))

    out = inside(p[0] for p in geom.surface_points)
    # Crossings of the layer boundaries and the water table: each is
    # straight between its vertices (level beyond them). Not the bottom of
    # the deepest layer, the model's base: a surface that crosses it only
    # touches it (``base_tolerance``; a deeper one is refused), and the
    # slice under it takes the same layer as the slice above (GM-128).
    def on_base(layer, x):
        return layer.bottom_at(x) <= min(
            L.bottom_at(x) for L in geom.soil_layers) + 1e-9

    for layer in geom.soil_layers:
        out += crossings_of([], layer.top_at)
        bpts = layer.bottom_boundary_points
        out += [x for x in crossings_of([p[0] for p in bpts] if bpts else [],
                                        layer.bottom_at)
                if not on_base(layer, x)]
    if geom.gwt_points:
        out += crossings_of([p[0] for p in geom.gwt_points],
                            geom.gwt_elevation_at)
    # The tension crack: slices on the crest half whose base is above its
    # base are in it, so it ends where the surface crosses that level on
    # the crest half, or at the middle of the surface.
    z_crack = _crack_base(geom, x_entry, x_exit)
    if z_crack is not None:
        x_half = 0.5 * (x_entry + x_exit)
        entry_side = geom.tension_crack_side == "entry"
        out += [x for x in crossings_of([], lambda x: z_crack)
                if (x <= x_half if entry_side else x >= x_half)]
        z_half = slip.slip_elevation_at(x_half)
        if z_half is not None and z_half >= z_crack:
            out.append(x_half)
    out += _surcharge_edges(geom)
    return inside(out)


def slice_boundaries(geom: SlopeGeometry, slip, n_slices: int,
                     x_entry: Optional[float] = None,
                     x_exit: Optional[float] = None,
                     extra=()) -> List[float]:
    """The x of every slice boundary, entry to exit, as ``build_slices``
    cuts them (GM-131, GM-134; see the module docstring).

    A boundary at each break in the ground surface, each crossing of a
    layer's top or bottom, of the water table or of a tension crack's
    base, and each surcharge edge (``slice_breaks``), at each corner of a
    noncircular surface, and at each x in ``extra`` (the other stage's
    breaks in a rapid drawdown, so both stages are cut alike), strictly
    between the entry and the exit. Every corner is kept (one within
    ``_END_MERGE`` of an end is that end); any other boundary within
    ``_BOUNDARY_MERGE`` of the span of another is merged into it. The
    stretches between them share out ``max(n_slices, stretches)`` slices
    in proportion to their width, at least one each (each further slice
    goes to the stretch whose slices are widest), and each stretch is cut
    evenly; with no boundary inside, ``n_slices`` slices of equal width.
    """
    if x_entry is None or x_exit is None:
        x_entry, x_exit = slip.find_entry_exit(geom)
    span = x_exit - x_entry
    tol = _BOUNDARY_MERGE * span

    corners = []
    if not getattr(slip, "is_circular", True):
        corners = [p[0] for p in (getattr(slip, "points", []) or [])
                   if x_entry < p[0] < x_exit]
    breaks = slice_breaks(geom, slip, x_entry, x_exit)
    breaks += [x for x in extra if x_entry < x < x_exit]

    bounds = sorted({x_entry, x_exit, *(
        x for x in corners
        if x - x_entry > _END_MERGE and x_exit - x > _END_MERGE)})
    for x in sorted(set(breaks)):
        i = bisect.bisect_left(bounds, x)
        if (x - bounds[i - 1] >= tol and bounds[i] - x >= tol):
            bounds.insert(i, x)

    widths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    counts = [1] * len(widths)
    for _ in range(n_slices - len(widths)):
        k = max(range(len(widths)), key=lambda i: widths[i] / counts[i])
        counts[k] += 1
    out = [x_entry]
    for a, b, c in zip(bounds[:-1], bounds[1:], counts):
        out += [a + (b - a) * j / c for j in range(1, c)] + [b]
    return out


def _boundary_elevation(slip, x: float) -> Optional[float]:
    """The slip surface's elevation at a slice boundary. A circle cut off at
    its vertical tangent (reverse curvature, x = xc +/- R) ends on the
    circle at the centre's level, where ``slip_elevation_at`` says outside."""
    z = slip.slip_elevation_at(x)
    if z is None and getattr(slip, "is_circular", False):
        r = slip.radius
        if abs(abs(x - slip.xc) - r) <= 1e-9 * max(1.0, r):
            z = slip.yc
    return z


# A surface within this fraction of the model's height of the bottom of the
# deepest layer touches it (GM-128). A circle given as tangent with its
# centre and radius to the centimetre, as published examples give them, lies
# within it on a model 10 m or more high. The FOS of a circle reaching that
# far under a firm base is lower than the circle tangent to it by no more
# than the slice count moves it: Taylor's D = 2 circle 0.02 %, V-043's
# 0.06 %, Slide2 #85's 0.26 %, against Taylor's circle at 0.7045 on 25
# slices and 0.7063 on 100 (0.26 %). A fraction rather than a length,
# because the FOS changes with the depth of a circle over the model's own
# length scale. A search keeps to surfaces inside the model
# (``search._SEARCH_BASE_TOUCH``).
_BASE_TOUCH = 1e-3


def model_height(geom: SlopeGeometry) -> float:
    """The model's height (m): from the lowest point of the bottom of the
    deepest layer to the highest ground."""
    z_bottom = min(
        min(z for _, z in L.bottom_boundary_points)
        if L.bottom_boundary_points else L.bottom_elevation
        for L in geom.soil_layers)
    return max(z for _, z in geom.surface_points) - z_bottom


def base_tolerance(geom: SlopeGeometry) -> float:
    """How far (m) a slip surface may reach below the bottom of the deepest
    layer and still touch it (GM-128): a thousandth of the model's height
    (``model_height``)."""
    return _BASE_TOUCH * model_height(geom)


def base_clearance(geom: SlopeGeometry, slip) -> float:
    """A lower bound (m) on the height of the slip surface above the bottom
    of the deepest layer, anywhere: the surface's lowest point (a circle's
    yc - R, a polyline's lowest corner) less the highest that bottom gets
    (the least over the layers of each one's highest bottom point). Cheap;
    where it is not negative the surface is never below the base."""
    top_of_base = min(
        max(z for _, z in L.bottom_boundary_points)
        if L.bottom_boundary_points else L.bottom_elevation
        for L in geom.soil_layers)
    if getattr(slip, "is_circular", True):
        lowest = slip.yc - slip.radius
    else:
        lowest = min((z for _, z in getattr(slip, "points", []) or ()),
                     default=math.inf)
    return lowest - top_of_base


def base_depth(geom: SlopeGeometry, slip, x_entry: float,
               x_exit: float) -> Tuple[float, float]:
    """(d, x): the greatest depth d (m) of the slip surface below the bottom
    of the deepest layer between ``x_entry`` and ``x_exit``, and the x where
    it is; d <= 0 where the surface is never below it (GM-128; then 0 at
    ``x_entry`` when ``base_clearance`` already says so).

    Exact, whatever the slicing. Between the corners of a polyline and the
    vertices of the layer bottoms the deepest bottom is the least of
    straight lines, so its height above the surface is concave (the lower
    arc of a circle is convex, a polyline straight) and peaks at a piece's
    end (either side of a vertical step in a bottom), where two bottoms
    cross, or, on a circle, where the arc runs parallel to one of the
    bottoms."""
    if base_clearance(geom, slip) >= 0.0:
        return 0.0, x_entry
    circular = getattr(slip, "is_circular", True)
    xs = {x_entry, x_exit}
    for L in geom.soil_layers:
        xs.update(p[0] for p in (L.bottom_boundary_points or ()))
    if not circular:
        xs.update(p[0] for p in (getattr(slip, "points", []) or []))
    xs = sorted(x for x in xs if x_entry <= x <= x_exit)

    best = (-math.inf, x_entry)
    layers = geom.soil_layers
    for a, b in zip(xs[:-1], xs[1:]):
        if b <= a:
            continue
        # Each bottom's line on this piece, from two points inside it: at a
        # vertical step (two bottom points at one x) `bottom_at` gives the
        # left value, so the piece's own ends would read the wrong side.
        qa, qb = a + 0.25 * (b - a), a + 0.75 * (b - a)
        lines = []
        for L in layers:
            za, zb = L.bottom_at(qa), L.bottom_at(qb)
            m = (zb - za) / (qb - qa)
            lines.append((za - m * (qa - a), m))
        cands = {a, b}
        if circular:
            for _, m in lines:
                x = slip.xc + m * slip.radius / math.sqrt(1.0 + m * m)
                if a < x < b:
                    cands.add(x)
        for i, (z1, m1) in enumerate(lines):
            for z2, m2 in lines[i + 1:]:
                if m1 != m2:
                    x = a + (z2 - z1) / (m1 - m2)
                    if a < x < b:
                        cands.add(x)
        for x in cands:
            z = _boundary_elevation(slip, x)
            if z is not None:
                d = min(z0 + m * (x - a) for z0, m in lines) - z
                if d > best[0]:
                    best = (d, x)
    return best


def build_slices(geom: SlopeGeometry,
                 slip,
                 n_slices: int = 30,
                 extra_boundaries=()) -> List[Slice]:
    """Discretize the sliding mass into vertical slices.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope geometry with surface, layers, and water table.
    slip : CircularSlipSurface or PolylineSlipSurface
        Trial slip surface (duck-typed — must have slip_elevation_at,
        tangent_angle_at, and find_entry_exit methods).
    n_slices : int
        Number of slices. Default 30. The surface is cut with a boundary
        at every ground break, layer crossing, water-table crossing,
        surcharge edge and tension-crack edge, and every corner of a
        noncircular surface (``slice_boundaries``; GM-131, GM-134), into
        this many slices, or one per stretch between those boundaries
        where there are more stretches.
    extra_boundaries : sequence of float
        Further x to cut at (``slice_boundaries``' ``extra``).

    Returns
    -------
    list of Slice
    """
    if n_slices < 3:
        raise ValueError(f"Need at least 3 slices, got {n_slices}")

    x_entry, x_exit = slip.find_entry_exit(geom)
    # Under the bottom of the deepest layer the surface would pass through
    # ground the model does not have, so it may at most touch it (SS-5,
    # GM-128). Measured on the surface, it is refused or kept at any slice
    # count; a surface that touches keeps every slice.
    depth, x_deep = base_depth(geom, slip, x_entry, x_exit)
    touch = base_tolerance(geom)
    if depth > touch:
        raise ValueError(
            "Slip surface passes below the bottom of the deepest soil "
            f"layer: {depth:.3f} m below it at x = {x_deep:.2f} (a surface "
            f"may touch it, within {touch:.3f} m, a thousandth of the "
            "model's height). Use a shallower trial surface or extend the "
            "soil layers."
        )
    # a boundary at every ground break, crossing, surcharge and crack edge
    # (and every corner of a polyline); each slice takes the base angle and
    # length of the chord between its boundaries, on a circle as on a
    # polyline.
    xs = slice_boundaries(geom, slip, n_slices, x_entry, x_exit,
                          extra_boundaries)
    cuts = [(a, b, b - a) for a, b in zip(xs[:-1], xs[1:])]

    # Discrete pore-pressure field (flow-net / TIN): build the interpolator once
    # per call for the hot path; None when the piezometric-line / ru model is used.
    pp_interp = (build_pore_pressure_interpolator(geom.pore_pressure_points)
                 if geom.pore_pressure_points is not None else None)

    # Tension crack: compute the crack base elevation at the crest end of the
    # slip surface (entry = low-x by default, or exit = high-x). Only slices on
    # that half of the surface can be in the crack.
    x_midpoint = (x_entry + x_exit) / 2.0
    crack_side = geom.tension_crack_side
    crack_truncate = geom.tension_crack_model == "truncation"
    crack_base_elev = _crack_base(geom, x_entry, x_exit)

    slices = []
    for x_left, x_right, width in cuts:
        x_mid = (x_left + x_right) / 2.0

        z_top = geom.ground_elevation_at(x_mid)
        z_base = slip.slip_elevation_at(x_mid)
        if z_base is None:
            continue

        height = z_top - z_base
        if height <= 0:
            continue

        z_l = _boundary_elevation(slip, x_left)
        z_r = _boundary_elevation(slip, x_right)
        if z_l is not None and z_r is not None:
            alpha = math.atan2(z_r - z_l, width)
            base_length = math.hypot(width, z_r - z_l)
        else:
            alpha = slip.tangent_angle_at(x_mid)
            cos_alpha = math.cos(alpha)
            if abs(cos_alpha) < 1e-10:
                cos_alpha = 1e-10
            base_length = width / abs(cos_alpha)

        # GWT at slice midpoint
        gwt_elev = geom.gwt_elevation_at(x_mid)

        # Weight through multiple layers
        weight = _compute_slice_weight(geom, x_mid, z_top, z_base, width, gwt_elev)

        # Ponded water (auto-detected): GWT above the ground surface means
        # standing water on the slice. The hydrostatic pressure on the
        # submerged ground-surface segment resolves into
        #   - a vertical component = weight of the water column
        #     (added to the slice weight), and
        #   - a signed horizontal component on inclined ground
        #     Fx = gamma_w * (d_l^2 - d_r^2)/2  (d = pool depth at the
        #     slice edges, clipped at 0), pushing toward the soil — the
        #     classic external-water buttress.
        # The pore pressure at the base already uses the full head to the
        # pool surface, so with both components a fully submerged slope
        # reproduces the buoyant-weight equivalence (boundary water
        # pressures + internal u == buoyancy).
        pond_weight = 0.0
        pond_hforce = 0.0
        pond_z = z_top
        if gwt_elev is not None and gwt_elev > z_top:
            pond_weight = GAMMA_W * (gwt_elev - z_top) * width
        if gwt_elev is not None:
            d_l = max(gwt_elev - geom.ground_elevation_at(x_left), 0.0)
            d_r = max(gwt_elev - geom.ground_elevation_at(x_right), 0.0)
            if d_l > 0.0 or d_r > 0.0:
                pond_hforce = 0.5 * GAMMA_W * (d_l * d_l - d_r * d_r)
                if abs(pond_hforce) > 1e-12:
                    # line of action: centroid of the trapezoidal pressure
                    # diagram on the projected vertical face
                    depth = (2.0 * (d_l ** 3 - d_r ** 3)
                             / (3.0 * (d_l * d_l - d_r * d_r)))
                    pond_z = gwt_elev - depth

        # Pore pressure at base
        pore_pressure = _pore_pressure_at_base(z_base, gwt_elev)

        # Strength parameters from layer at base midpoint
        base_layer = geom.layer_at_point(x_mid, z_base)
        if base_layer is None:
            bottoms = [L.bottom_at(x_mid) for L in geom.soil_layers]
            if z_base < min(bottoms):
                # touching the base, within base_tolerance: the deepest
                # layer's strength (GM-128)
                base_layer = geom.soil_layers[bottoms.index(min(bottoms))]
            else:
                # Fallback: use bottom-most layer
                base_layer = geom.soil_layers[-1]
        # Apply Ru if no GWT pore pressure and layer has Ru > 0
        if pp_interp is None and pore_pressure <= 0 and base_layer.ru > 0:
            # u = Ru * gamma * h  where h = overburden depth
            overburden_h = z_top - z_base
            pore_pressure = base_layer.ru * base_layer.gamma * overburden_h

        # Discrete pore-pressure field overrides the piezometric-line / ru value
        # at the slice base (the ponded-water buttress above still uses gwt_elev).
        if pp_interp is not None:
            pore_pressure = pp_interp(x_mid, z_base)

        # Strength model evaluation (SHANSEP / Hoek-Brown need stress
        # estimates at the base: Fellenius normal and vertical effective)
        if base_layer.strength_model == "mohr_coulomb":
            c, phi = base_layer.shear_strength_params
        else:
            sigma_v = (weight / width if width > 0 else 0.0) - pore_pressure
            alpha_cos = math.cos(alpha)
            dl = width / max(abs(alpha_cos), 1e-10)
            sigma_n = (weight * alpha_cos / dl if dl > 0 else 0.0) \
                - pore_pressure
            # alpha (base inclination) drives the 'anisotropic' su(alpha) model;
            # ignored by shansep / hoek_brown.
            c, phi = base_layer.strength_at(sigma_n, sigma_v, alpha=alpha)

        # Tension crack: slices on the CREST half whose base is above the crack
        # bottom either lose shear resistance (strength model — open crack face)
        # or are removed from the sliding mass (truncation model — the mass ends
        # at the vertical crack face). The x-position guard scopes it to the
        # crest side (entry = low-x default, exit = high-x).
        in_crack = False
        if crack_base_elev is not None and z_base >= crack_base_elev:
            on_crest_side = (x_mid <= x_midpoint if crack_side == "entry"
                             else x_mid >= x_midpoint)
            if on_crest_side:
                if crack_truncate:
                    continue           # truncate the mass at the crack face
                in_crack = True
                c = 0.0
                phi = 0.0

        # Surcharge
        surcharge_force = geom.surcharge_at(x_mid) * width

        # Seismic horizontal force — applied to the SOIL mass only
        # (hydrodynamic effects on ponded water are not modeled)
        seismic_force = geom.kh * weight

        # add pond water weight after seismic so kh acts on soil only
        weight = weight + pond_weight

        # Centroid elevation (for seismic moment arm)
        z_centroid = z_base + height / 2.0

        slices.append(Slice(
            x_left=x_left,
            x_right=x_right,
            x_mid=x_mid,
            width=width,
            z_top=z_top,
            z_base=z_base,
            height=height,
            alpha=alpha,
            base_length=base_length,
            weight=weight,
            pore_pressure=pore_pressure,
            c=c,
            phi=phi,
            surcharge_force=surcharge_force,
            seismic_force=seismic_force,
            z_centroid=z_centroid,
            in_tension_crack=in_crack,
            pond_force=pond_weight,
            pond_hforce=pond_hforce,
            pond_z=pond_z,
        ))

    # Tension crack water force: hydrostatic thrust on the retained face,
    # F_w = 0.5 * gamma_w * z_w^2 acting at z_w/3 above the crack base (treated
    # downstream as an always-driving magnitude). Apply it to the retained slice
    # adjacent to the crack face: the first non-cracked slice for an entry-side
    # crack, the last non-cracked for an exit-side crack (for truncation the
    # cracked slices are already gone, so this is the boundary retained slice).
    if crack_base_elev is not None and geom.tension_crack_water_depth > 0 and slices:
        z_w = geom.tension_crack_water_depth
        crack_water_f = 0.5 * GAMMA_W * z_w ** 2
        crack_water_elev = crack_base_elev + z_w / 3.0
        target = None
        ordered = slices if crack_side == "entry" else list(reversed(slices))
        for s in ordered:
            if not s.in_tension_crack:
                target = s
                break
        if target is None:
            target = ordered[-1]
        target.crack_water_force = crack_water_f
        target.crack_water_z = crack_water_elev

    return slices


def _compute_slice_weight(geom: SlopeGeometry,
                          x_mid: float,
                          z_top: float,
                          z_base: float,
                          width: float,
                          gwt_elev: Optional[float]) -> float:
    """Compute weight of a single slice through multiple soil layers.

    Traverses layers from top to bottom within the slice height,
    using gamma above GWT and gamma_sat below GWT.
    """
    weight = 0.0

    # Collect layers that overlap with (z_base, z_top)
    for layer in geom.soil_layers:
        # Clip layer to slice elevation range
        lay_top = min(layer.top_at(x_mid), z_top)
        lay_bot = max(layer.bottom_at(x_mid), z_base)

        if lay_top <= lay_bot:
            continue

        layer_thickness = lay_top - lay_bot

        if gwt_elev is not None:
            # Part above GWT uses gamma, part below uses gamma_sat
            above_gwt = max(0.0, lay_top - max(gwt_elev, lay_bot))
            below_gwt = max(0.0, min(gwt_elev, lay_top) - lay_bot)
            weight += (above_gwt * layer.gamma + below_gwt * layer.gamma_sat) * width
        else:
            weight += layer_thickness * layer.gamma * width

    # If no layers cover the slice (shouldn't happen normally), estimate
    if weight == 0.0 and z_top > z_base:
        fallback_layer = geom.soil_layers[0]
        weight = (z_top - z_base) * fallback_layer.gamma * width

    return weight


def _pore_pressure_at_base(z_base: float,
                           gwt_elev: Optional[float]) -> float:
    """Compute pore water pressure at the midpoint of the slice base.

    u = gamma_w * (z_gwt - z_base) if z_base < z_gwt, else 0.

    ASSUMPTION (SS-3): the piezometric head is taken as the full vertical
    distance from the slice base to the GWT surface (hydrostatic, no-flow
    conditions). For seepage parallel to an inclined phreatic surface the
    rigorous head is reduced (factor ~cos^2(beta) of the phreatic slope,
    or per a flownet), so this slightly OVERESTIMATES u — conservative —
    on slopes with steeply inclined water tables. Use the layer ``ru``
    coefficient as the alternative pore-pressure model when a calibrated
    value is available.
    """
    if gwt_elev is None:
        return 0.0
    depth_below_gwt = gwt_elev - z_base
    if depth_below_gwt <= 0:
        return 0.0
    return GAMMA_W * depth_below_gwt
