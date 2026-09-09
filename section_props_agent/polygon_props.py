"""Exact section-property integration over polygonal outlines.

Every geometric property this module returns is a closed-form integral over
the section outline (Green's theorem), so there is no mesh, no solver and no
discretisation error: a polygon's area, centroid, second moments and plastic
moduli come out exact to floating point.

Regions are described as a list of rings, each ``(points, sign)`` with
``sign = +1`` for solid material and ``-1`` for a hole. Ring orientation is
normalised internally, so callers need not care whether they wound a ring
clockwise or anti-clockwise.

Sign conventions follow the structural convention used throughout the module:
``ixx`` is the second moment about the centroidal x-axis (i.e. the integral of
y^2 dA), and ``phi`` is the angle of the major principal (11) axis measured
from the x-axis.
"""

import math

__all__ = [
    "ring_integrals",
    "region_properties",
    "clip_ring_below",
    "plastic_modulus_x",
    "polygon_is_simple",
    "ring_perimeter",
]

_TOL = 1e-12


def _closed(points):
    """Return the ring as a list of (x, y) floats, without a repeated last point."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2 and abs(pts[0][0] - pts[-1][0]) < _TOL \
            and abs(pts[0][1] - pts[-1][1]) < _TOL:
        pts = pts[:-1]
    return pts


def ring_integrals(points):
    """Area and area-moment integrals of one closed ring, about the origin.

    Returns ``(area, qx, qy, ix, iy, ixy)`` where ``qx = int(y dA)``,
    ``qy = int(x dA)``, ``ix = int(y^2 dA)``, ``iy = int(x^2 dA)`` and
    ``ixy = int(x y dA)``. The values are signed by the ring's winding
    direction (positive anti-clockwise).
    """
    pts = _closed(points)
    n = len(pts)
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    a = qx = qy = ix = iy = ixy = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        qy += (x0 + x1) * cross
        qx += (y0 + y1) * cross
        ix += (y0 * y0 + y0 * y1 + y1 * y1) * cross
        iy += (x0 * x0 + x0 * x1 + x1 * x1) * cross
        ixy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cross

    return a / 2.0, qx / 6.0, qy / 6.0, ix / 12.0, iy / 12.0, ixy / 24.0


def ring_perimeter(points):
    """Closed-outline length of one ring."""
    pts = _closed(points)
    n = len(pts)
    if n < 2:
        return 0.0
    return sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n))


def _signed(rings):
    """Normalise ``(points, sign)`` rings to anti-clockwise, carrying the sign."""
    out = []
    for points, sign in rings:
        pts = _closed(points)
        area, *_ = ring_integrals(pts)
        if area < 0:
            pts = pts[::-1]
        out.append((pts, 1.0 if sign >= 0 else -1.0))
    return out


def region_properties(rings):
    """Geometric properties of a multi-ring region.

    Parameters
    ----------
    rings : list of (points, sign)
        ``sign`` is +1 for material, -1 for a hole.

    Returns
    -------
    dict
        ``area``, ``cx``, ``cy``, ``ixx``, ``iyy``, ``ixy`` (centroidal),
        ``i11``, ``i22``, ``phi_deg``, ``rx``, ``ry``, plus the extreme-fibre
        distances ``x_min``/``x_max``/``y_min``/``y_max`` and the elastic
        moduli ``zxx_plus``/``zxx_minus``/``zyy_plus``/``zyy_minus``.
    """
    norm = _signed(rings)
    area = qx = qy = ix = iy = ixy0 = 0.0
    for pts, sign in norm:
        a, mqx, mqy, mix, miy, mixy = ring_integrals(pts)
        area += sign * a
        qx += sign * mqx
        qy += sign * mqy
        ix += sign * mix
        iy += sign * miy
        ixy0 += sign * mixy

    if area <= _TOL:
        raise ValueError("region has zero or negative area")

    cx = qy / area
    cy = qx / area
    ixx = ix - area * cy * cy
    iyy = iy - area * cx * cx
    ixy = ixy0 - area * cx * cy

    # A symmetric section's product moment is analytically zero; what survives
    # the integration is round-off. Left in place it would flip the principal
    # axis angle by 180 degrees (atan2 is sign-sensitive at the origin), so
    # snap it to a true zero relative to the section's own scale.
    if abs(ixy) < 1e-10 * max(abs(ixx), abs(iyy), 1.0):
        ixy = 0.0

    # principal axes
    avg = 0.5 * (ixx + iyy)
    dif = 0.5 * (ixx - iyy)
    root = math.hypot(dif, ixy)
    i11 = avg + root
    i22 = avg - root
    # matches the retired library's convention: phi = atan2(ixx - i11, ixy)
    phi_deg = math.degrees(math.atan2(ixx - i11, ixy))

    xs = [p[0] for pts, _ in norm for p in pts]
    ys = [p[1] for pts, _ in norm for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return {
        "area": area,
        "cx": cx, "cy": cy,
        "ixx": ixx, "iyy": iyy, "ixy": ixy,
        "i11": i11, "i22": i22, "phi_deg": phi_deg,
        "rx": math.sqrt(ixx / area) if ixx > 0 else 0.0,
        "ry": math.sqrt(iyy / area) if iyy > 0 else 0.0,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "zxx_plus": ixx / abs(y_max - cy) if abs(y_max - cy) > _TOL else 0.0,
        "zxx_minus": ixx / abs(y_min - cy) if abs(y_min - cy) > _TOL else 0.0,
        "zyy_plus": iyy / abs(x_max - cx) if abs(x_max - cx) > _TOL else 0.0,
        "zyy_minus": iyy / abs(x_min - cx) if abs(x_min - cx) > _TOL else 0.0,
    }


# ------------------------------------------------------------------ clipping

def clip_ring_below(points, y_level):
    """Sutherland-Hodgman clip of one ring to the half-plane y <= y_level.

    Valid for any simple polygon (convex or not): the clip region is a
    half-plane, so the result is a single closed ring, possibly with
    zero-width slivers along the cut that contribute nothing to the
    integrals.
    """
    pts = _closed(points)
    if not pts:
        return []
    out = []
    n = len(pts)
    for i in range(n):
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        cur_in = cur[1] <= y_level
        nxt_in = nxt[1] <= y_level
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            dy = nxt[1] - cur[1]
            if abs(dy) > _TOL:
                t = (y_level - cur[1]) / dy
                out.append((cur[0] + t * (nxt[0] - cur[0]), y_level))
    return out


def _area_below(norm_rings, y_level):
    total = 0.0
    for pts, sign in norm_rings:
        clipped = clip_ring_below(pts, y_level)
        if len(clipped) >= 3:
            total += sign * ring_integrals(clipped)[0]
    return total


def _first_moment_below(norm_rings, y_level, about):
    """int |y - about| dA over the part of the region below y_level."""
    total = 0.0
    for pts, sign in norm_rings:
        clipped = clip_ring_below(pts, y_level)
        if len(clipped) >= 3:
            a, qx, *_ = ring_integrals(clipped)
            total += sign * (qx - about * a)
    return total


def plastic_modulus_x(rings, tol=1e-9, max_iter=200):
    """Plastic section modulus about the x-axis (and the plastic NA depth).

    The plastic neutral axis is the horizontal line that halves the area; the
    modulus is the sum of the magnitudes of the first moments of the two
    halves about that line. Found by bisection on the exactly-integrated
    clipped area, so the only error is the root-finding tolerance.

    Returns ``(sx, y_pna)``.
    """
    norm = _signed(rings)
    total = sum(sign * ring_integrals(pts)[0] for pts, sign in norm)
    if total <= _TOL:
        raise ValueError("region has zero or negative area")

    ys = [p[1] for pts, _ in norm for p in pts]
    lo, hi = min(ys), max(ys)
    target = 0.5 * total
    span = hi - lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _area_below(norm, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(span, 1.0):
            break
    y_pna = 0.5 * (lo + hi)

    q_below = _first_moment_below(norm, y_pna, y_pna)
    # the whole region's first moment about the PNA, minus the lower part
    q_all = sum(sign * (ring_integrals(pts)[1] - y_pna * ring_integrals(pts)[0])
                for pts, sign in norm)
    q_above = q_all - q_below
    return abs(q_above) + abs(q_below), y_pna


def _rotate(rings):
    """Swap x and y so an x-axis routine can serve the y-axis."""
    return [([(y, x) for x, y in pts], sign) for pts, sign in rings]


def plastic_modulus_y(rings, **kw):
    """Plastic section modulus about the y-axis."""
    sy, x_pna = plastic_modulus_x(_rotate(rings), **kw)
    return sy, x_pna


# ---------------------------------------------------------------- validation

def _seg_intersect(p1, p2, p3, p4):
    """True if closed segments p1p2 and p3p4 cross at an interior point."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(v) < 1e-9:
            return 0
        return 1 if v > 0 else -1

    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_seg(p1, p2, p3):
        return True
    if o2 == 0 and on_seg(p1, p2, p4):
        return True
    if o3 == 0 and on_seg(p3, p4, p1):
        return True
    if o4 == 0 and on_seg(p3, p4, p2):
        return True
    return False


def polygon_is_simple(points):
    """True if the closed polygon has no self-intersections and non-zero area.

    Replaces the shapely ``is_valid`` check the library path used, so the
    module carries no geometry-library dependency.
    """
    pts = _closed(points)
    n = len(pts)
    if n < 3:
        return False
    if abs(ring_integrals(pts)[0]) <= _TOL:
        return False
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i:
                continue
            # skip adjacent segments: they legitimately share an endpoint
            if (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1, b2 = pts[j], pts[(j + 1) % n]
            if _seg_intersect(a1, a2, b1, b2):
                return False
    return True
