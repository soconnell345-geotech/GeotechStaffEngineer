"""St. Venant torsion and warping constants, from the published solutions.

Where a shape has an exact or industry-standard closed form, that form is
used -- it is what steel design tables and codes quote, and for the solid
rectangle the series solution is the exact elasticity answer:

======================  ==========================================
shape                   J (torsion constant)
======================  ==========================================
rectangle (solid)       exact St. Venant series (Timoshenko & Goodier,
                        Theory of Elasticity, 3rd ed., Art. 109)
circle / CHS            polar second moment (exact)
RHS (closed box)        Bredt thin-wall single-cell formula
I-section               El Darwish & Johnston (1965) / AISC Design
                        Guide 9 -- flange+web strips plus the fillet term
arbitrary polygon       finite-difference solve of the Prandtl stress
                        function (no closed form exists)
======================  ==========================================

Warping constants follow the same rule: the doubly-symmetric I-section uses
the standard Cw = Iy*h0^2/4; circular sections do not warp (Cw = 0); for the
solid rectangle, the closed box and arbitrary polygons no defensible closed
form exists, so ``None`` is returned rather than a fabricated number (warping
restraint is neglected for solid and closed sections in design practice).
"""

import math

__all__ = [
    "rectangle_torsion",
    "circle_torsion",
    "chs_torsion",
    "rhs_torsion",
    "i_section_torsion",
    "i_section_warping",
    "polygon_torsion_fd",
]


def rectangle_torsion(b, d, n_terms=60):
    """Exact St. Venant torsion constant of a solid rectangle b x d (mm).

    J = a*t^3 * [1/3 - (64/pi^5)*(t/a) * sum_{k odd} tanh(k*pi*a/(2t))/k^5]
    with ``a`` the long side and ``t`` the short side. Converges to machine
    precision within a handful of terms.
    """
    a = max(float(b), float(d))
    t = min(float(b), float(d))
    if t <= 0.0:
        return 0.0
    total = 0.0
    for k in range(1, 2 * n_terms, 2):
        arg = k * math.pi * a / (2.0 * t)
        # tanh saturates to 1.0 quickly; guard the overflow for slender shapes
        th = 1.0 if arg > 350.0 else math.tanh(arg)
        total += th / k ** 5
    return a * t ** 3 * (1.0 / 3.0 - (64.0 / math.pi ** 5) * (t / a) * total)


def circle_torsion(d):
    """Polar second moment of a solid circle (exact)."""
    return math.pi * float(d) ** 4 / 32.0


def chs_torsion(d, t):
    """Polar second moment of a circular hollow section (exact)."""
    d_o = float(d)
    d_i = d_o - 2.0 * float(t)
    return math.pi * (d_o ** 4 - d_i ** 4) / 32.0


def rhs_torsion(d, b, t, r_out=None):
    """Bredt single-cell torsion constant of a closed rectangular hollow section.

    J = 4 * Am^2 * t / p, with ``Am`` the area enclosed by the mid-thickness
    perimeter and ``p`` that perimeter's length; corner radii are taken at the
    mid-thickness line (r_m = r_out - t/2).
    """
    d, b, t = float(d), float(b), float(t)
    r_out = 2.0 * t if r_out is None else float(r_out)
    r_m = max(r_out - t / 2.0, 0.0)
    bm, dm = b - t, d - t          # mid-thickness box dimensions
    # enclosed area: full box less the four corner cut-offs
    area_m = bm * dm - (4.0 - math.pi) * r_m ** 2
    # mid-line length: straight runs + four quarter arcs
    perim_m = 2.0 * (bm + dm) - 8.0 * r_m + 2.0 * math.pi * r_m
    if perim_m <= 0.0:
        return 0.0
    return 4.0 * area_m ** 2 * t / perim_m


def i_section_torsion(d, b, t_f, t_w, r=0.0):
    """I-section torsion constant, El Darwish & Johnston (1965).

    J = (1/3)[2*b*tf^3 + (d - 2*tf)*tw^3] + 2*alpha*D^4

    The fillet term is what separates a rolled shape's tabulated J from the
    naive sum of rectangles; without it J is ~10% low on a typical rolled
    section.
    """
    d, b, t_f, t_w, r = (float(v) for v in (d, b, t_f, t_w, r))
    j_strips = (2.0 * b * t_f ** 3 + (d - 2.0 * t_f) * t_w ** 3) / 3.0
    if r <= 0.0:
        return j_strips
    alpha = (-0.042
             + 0.2204 * (t_w / t_f)
             + 0.1355 * (r / t_f)
             - 0.0865 * (t_w * r / t_f ** 2)
             - 0.0725 * (t_w / t_f) ** 2)
    big_d = ((t_f + r) ** 2 + t_w * r + t_w ** 2 / 4.0) / (2.0 * r + t_f)
    return j_strips + 2.0 * alpha * big_d ** 4


def i_section_warping(d, b, t_f, t_w, r=0.0):
    """Warping constant of a doubly-symmetric I-section: Cw = Iy * h0^2 / 4.

    ``h0`` is the flange-centroid separation (d - t_f). Root radii are
    ignored, which is the same assumption behind the tabulated Cw values.
    """
    d, b, t_f, t_w = (float(v) for v in (d, b, t_f, t_w))
    h_w = d - 2.0 * t_f
    iy = 2.0 * (t_f * b ** 3 / 12.0) + h_w * t_w ** 3 / 12.0
    h0 = d - t_f
    return iy * h0 ** 2 / 4.0


# ------------------------------------------------------- arbitrary polygons

def polygon_torsion_fd(points, n_across=150, max_cells=1_200_000,
                       richardson=True):
    """Torsion constant of an arbitrary simply-connected polygon.

    Solves the Prandtl stress function problem ``lap(phi) = -2`` with
    ``phi = 0`` on the boundary by finite differences on a uniform grid, then
    ``J = 2 * integral(phi dA)``.

    Holding ``phi = 0`` at the first grid node outside the outline places the
    zero contour about half a cell beyond the true boundary, which makes the
    raw solve first-order accurate in the cell size (measured: the error
    halves exactly with each grid halving). Solving twice and extrapolating
    to zero cell size therefore removes almost all of it -- on a rectangle,
    where the exact series answer is known, the extrapolated value lands
    within 0.02%. Re-entrant corners hold a genuine stress singularity that
    no uniform grid resolves cleanly, so an outline with a sharp inside
    corner stays the least accurate case.

    Exact closed forms are used for every parametric shape, so this path only
    serves genuinely arbitrary outlines.

    Parameters
    ----------
    points : list of (x, y)
    n_across : int
        Cells across the larger bounding-box dimension for the coarse solve.
    max_cells : int
        Safety cap on the grid size.
    richardson : bool
        Solve again at half the cell size and extrapolate. Roughly triples
        the cost and cuts the error by one to two orders of magnitude.
    """
    coarse = _prandtl_solve(points, n_across, max_cells)
    if not richardson:
        return coarse
    fine = _prandtl_solve(points, 2 * n_across, max_cells)
    # first-order error model: J(h) = J0 + C*h  ->  J0 = 2*J(h/2) - J(h)
    return 2.0 * fine - coarse


def _prandtl_solve(points, n_across, max_cells):
    """One finite-difference Prandtl solve; see ``polygon_torsion_fd``."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg

    pts = [(float(x), float(y)) for x, y in points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, y1 - y0)
    if span <= 0.0:
        return 0.0

    h = span / float(n_across)
    nx = int(math.ceil((x1 - x0) / h)) + 2
    ny = int(math.ceil((y1 - y0) / h)) + 2
    while nx * ny > max_cells:
        h *= 1.25
        nx = int(math.ceil((x1 - x0) / h)) + 2
        ny = int(math.ceil((y1 - y0) / h)) + 2

    # Offset the nodes by half a cell. On an axis-aligned outline an
    # un-offset grid drops nodes exactly on the boundary, where the
    # inside/outside test turns on round-off; that makes the error jump
    # around instead of falling smoothly with the cell size, and the
    # extrapolation below depends on it falling smoothly.
    gx = x0 - 0.5 * h + np.arange(nx) * h
    gy = y0 - 0.5 * h + np.arange(ny) * h
    gxx, gyy = np.meshgrid(gx, gy, indexing="ij")

    # even-odd ray crossing, vectorised over the whole grid
    inside = np.zeros(gxx.shape, dtype=bool)
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[(i + 1) % n]
        if yi == yj:
            continue
        cond = ((gyy >= min(yi, yj)) & (gyy < max(yi, yj)))
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = xi + (gyy - yi) * (xj - xi) / (yj - yi)
        inside ^= cond & (gxx < x_cross)

    idx = -np.ones(gxx.shape, dtype=np.int64)
    unknowns = np.flatnonzero(inside.ravel())
    if unknowns.size == 0:
        return 0.0
    idx.ravel()[unknowns] = np.arange(unknowns.size)

    # Assemble 4*phi_i - sum(phi_neighbours) = 2h^2, i.e. the NEGATIVE
    # Laplacian, which is symmetric positive definite -- so it can be solved by
    # conjugate gradients. A direct sparse factorisation of a 2D Laplacian
    # fills in badly (hundreds of MB on a 150k-node grid, enough to push a
    # full test run out of memory); CG holds only a few vectors.
    rows, cols, vals = [], [], []
    flat_idx = idx.ravel()
    for k, node in enumerate(unknowns):
        rows.append(k); cols.append(k); vals.append(4.0)
        i, j = divmod(int(node), ny)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < nx and 0 <= jj < ny:
                nb = flat_idx[ii * ny + jj]
                if nb >= 0:                      # outside nodes hold phi = 0
                    rows.append(k); cols.append(int(nb)); vals.append(-1.0)

    a = csr_matrix((vals, (rows, cols)),
                   shape=(unknowns.size, unknowns.size))
    rhs = np.full(unknowns.size, 2.0 * h * h)
    try:
        phi, info = cg(a, rhs, rtol=1e-10, atol=0.0, maxiter=20000)
    except TypeError:                       # scipy < 1.14 spelled it `tol`
        phi, info = cg(a, rhs, tol=1e-10, atol=0.0, maxiter=20000)
    if info != 0:
        raise RuntimeError(
            f"torsion solve did not converge (scipy cg info={info})")
    return float(2.0 * phi.sum() * h * h)
