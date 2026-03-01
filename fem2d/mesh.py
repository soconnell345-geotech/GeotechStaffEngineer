"""
2D mesh generation for geotechnical FEM using scipy.spatial.Delaunay.

Provides:
- Rectangular and polygonal domain meshing
- Variable-density point seeding
- Point-in-polygon testing (ray casting)
- Soil layer assignment by centroid elevation
- Laplacian smoothing
- Mesh quality metrics
"""

import numpy as np
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------
# Point-in-polygon (vectorized ray casting)
# ---------------------------------------------------------------------------

def points_in_polygon(points, polygon):
    """Test which points lie inside a polygon (vectorized).

    Parameters
    ----------
    points : (M, 2) array
    polygon : (N, 2) array — vertices in order (not self-closing).

    Returns
    -------
    inside : (M,) bool array
    """
    x = points[:, 0]
    y = points[:, 1]
    n = len(polygon)
    inside = np.zeros(len(points), dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        cond1 = (yi > y) != (yj > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
        cond2 = x < x_int
        inside ^= (cond1 & cond2)
        j = i
    return inside


def point_in_polygon(point, polygon):
    """Scalar version of point-in-polygon test."""
    return points_in_polygon(np.array([point]), polygon)[0]


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def generate_rect_mesh(x_min, x_max, y_min, y_max, nx, ny):
    """Generate a structured triangular mesh for a rectangular domain.

    Splits each quad cell into 2 triangles (shortest diagonal).

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float — domain bounds.
    nx, ny : int — number of elements in each direction.

    Returns
    -------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — triangle connectivity.
    """
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            # Split along shorter diagonal
            d02 = np.linalg.norm(nodes[n2] - nodes[n0])
            d13 = np.linalg.norm(nodes[n3] - nodes[n1])
            if d02 <= d13:
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
            else:
                elements.append([n0, n1, n3])
                elements.append([n1, n2, n3])

    return nodes, np.array(elements)


def generate_slope_mesh(surface_points, depth, nx=30, ny=15,
                        x_extend_left=0.0, x_extend_right=0.0):
    """Generate a triangular mesh for a slope cross-section.

    Creates a rectangular domain from the surface profile down to a
    specified depth, then uses Delaunay triangulation with surface-
    conforming node placement.

    Parameters
    ----------
    surface_points : list of (x, z) tuples — ground surface profile,
        sorted by x.
    depth : float — depth below lowest surface point.
    nx : int — approximate number of horizontal divisions.
    ny : int — number of vertical divisions.
    x_extend_left, x_extend_right : float — extra horizontal extent.

    Returns
    -------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array
    """
    surf = np.array(surface_points)
    x_min = surf[:, 0].min() - x_extend_left
    x_max = surf[:, 0].max() + x_extend_right
    y_min = surf[:, 1].min() - depth

    # Generate grid nodes
    x_vals = np.linspace(x_min, x_max, nx + 1)
    y_base = np.linspace(y_min, 0, ny + 1)

    all_pts = []
    for x in x_vals:
        y_surf = np.interp(x, surf[:, 0], surf[:, 1])
        y_col = np.linspace(y_min, y_surf, ny + 1)
        for y in y_col:
            all_pts.append([x, y])

    # Add surface points themselves for accuracy
    for x, y in surface_points:
        all_pts.append([x, y])

    all_pts = np.array(all_pts)

    # Remove near-duplicates
    all_pts = _remove_duplicates(all_pts, tol=1e-6)

    # Delaunay triangulation
    tri = Delaunay(all_pts)
    elements = tri.simplices

    # Remove degenerate triangles (zero area)
    areas = _triangle_areas(all_pts, elements)
    valid = areas > 1e-12
    elements = elements[valid]

    # Ensure CCW ordering
    elements = _ensure_ccw(all_pts, elements)

    return all_pts, elements


def generate_polygon_mesh(polygon, element_size, interior_boundaries=None):
    """Generate a triangular mesh for a polygonal domain.

    Parameters
    ----------
    polygon : (N, 2) array — domain boundary (CCW ordering).
    element_size : float — target element size (m).
    interior_boundaries : list of (M, 2) arrays, optional —
        internal boundaries (e.g., soil layer interfaces) to add as
        node constraints.

    Returns
    -------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array
    """
    # Boundary points
    bnd_pts = _sample_boundary(polygon, element_size)
    n_boundary = len(bnd_pts)

    # Interior boundary points
    int_bnd_pts = []
    if interior_boundaries:
        for bnd in interior_boundaries:
            pts = _sample_polyline(bnd, element_size)
            int_bnd_pts.append(pts)

    # Interior grid points
    interior_pts = _seed_interior_grid(polygon, element_size)

    # Combine all points
    parts = [bnd_pts]
    for pts in int_bnd_pts:
        parts.append(pts)
    parts.append(interior_pts)
    all_pts = np.vstack(parts)
    all_pts = _remove_duplicates(all_pts, tol=element_size * 0.3)

    # Triangulate
    if len(all_pts) < 3:
        raise ValueError("Too few points for triangulation")
    tri = Delaunay(all_pts)

    # Remove exterior triangles
    centroids = all_pts[tri.simplices].mean(axis=1)
    inside = points_in_polygon(centroids, polygon)
    elements = tri.simplices[inside]

    # Remove degenerate
    areas = _triangle_areas(all_pts, elements)
    elements = elements[areas > 1e-12]

    # Laplacian smooth interior nodes
    boundary_idx = set(range(n_boundary))
    all_pts = _laplacian_smooth(all_pts, elements, boundary_idx, n_iter=3)

    # Ensure CCW
    elements = _ensure_ccw(all_pts, elements)

    return all_pts, elements


# ---------------------------------------------------------------------------
# Soil layer assignment
# ---------------------------------------------------------------------------

def assign_layers_by_elevation(nodes, elements, layer_boundaries):
    """Assign material layer index to each element by centroid elevation.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array
    layer_boundaries : list of float — layer bottom elevations, from
        top to bottom. E.g., [5.0, 0.0, -10.0] defines 2 layers:
        above 5.0 (layer 0), 5.0–0.0 (layer 1), below 0.0 (layer 2).

    Returns
    -------
    layer_ids : (n_elements,) int array
    """
    centroids = nodes[elements].mean(axis=1)
    cy = centroids[:, 1]
    layer_ids = np.zeros(len(elements), dtype=int)

    for i, bound in enumerate(layer_boundaries):
        layer_ids[cy < bound] = i + 1

    return layer_ids


def assign_layers_by_polylines(nodes, elements, layer_polylines):
    """Assign layer IDs using polyline boundaries.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array
    layer_polylines : list of (M, 2) arrays — layer boundaries from
        top to bottom. Each defines the bottom of a layer.

    Returns
    -------
    layer_ids : (n_elements,) int array
    """
    centroids = nodes[elements].mean(axis=1)
    cx = centroids[:, 0]
    cy = centroids[:, 1]
    layer_ids = np.zeros(len(elements), dtype=int)

    for i, polyline in enumerate(layer_polylines):
        poly_x = polyline[:, 0]
        poly_y = polyline[:, 1]
        bound_y = np.interp(cx, poly_x, poly_y)
        layer_ids[cy < bound_y] = i + 1

    return layer_ids


# ---------------------------------------------------------------------------
# Boundary condition detection
# ---------------------------------------------------------------------------

def detect_boundary_nodes(nodes, tol=0.01):
    """Detect standard geotechnical boundary condition nodes.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    tol : float — tolerance for edge detection.

    Returns
    -------
    dict with keys:
        'fixed_base': node indices at bottom (u=v=0)
        'roller_left': node indices on left edge (u=0)
        'roller_right': node indices on right edge (u=0)
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min = y.min()

    base = np.where(np.abs(y - y_min) < tol)[0]
    left = np.where(np.abs(x - x_min) < tol)[0]
    right = np.where(np.abs(x - x_max) < tol)[0]

    # Remove base nodes from roller lists
    base_set = set(base)
    left = np.array([n for n in left if n not in base_set])
    right = np.array([n for n in right if n not in base_set])

    return {
        'fixed_base': base,
        'roller_left': left,
        'roller_right': right,
    }


# ---------------------------------------------------------------------------
# Mesh quality
# ---------------------------------------------------------------------------

def triangle_quality(nodes, elements):
    """Compute quality metrics for triangular elements.

    Returns
    -------
    dict with 'min_angles', 'aspect_ratios', 'areas' — all (n_elements,) arrays.
    """
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    # Areas via cross product
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    areas = 0.5 * np.abs(cross)

    # Minimum angle
    def _angle(a, b, c_arr):
        ab = b - a
        ac = c_arr - a
        cos_a = np.sum(ab * ac, axis=1) / (
            np.linalg.norm(ab, axis=1) * np.linalg.norm(ac, axis=1) + 1e-30)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    a0 = _angle(v0, v1, v2)
    a1 = _angle(v1, v2, v0)
    a2 = _angle(v2, v0, v1)
    min_angles = np.minimum(np.minimum(a0, a1), a2)

    # Aspect ratio
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest_alt = 2.0 * areas / (longest + 1e-30)
    aspect_ratios = longest / (shortest_alt + 1e-30)

    return {
        'min_angles': min_angles,
        'aspect_ratios': aspect_ratios,
        'areas': areas,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_boundary(polygon, spacing):
    """Place points along polygon edges at given spacing."""
    pts = []
    n = len(polygon)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        edge = p1 - p0
        edge_len = np.linalg.norm(edge)
        n_seg = max(1, int(round(edge_len / spacing)))
        for j in range(n_seg):
            pts.append(p0 + (j / n_seg) * edge)
    return np.array(pts)


def _sample_polyline(polyline, spacing):
    """Place points along an open polyline."""
    pts = []
    for i in range(len(polyline) - 1):
        p0 = polyline[i]
        p1 = polyline[i + 1]
        edge = p1 - p0
        edge_len = np.linalg.norm(edge)
        n_seg = max(1, int(round(edge_len / spacing)))
        for j in range(n_seg + 1):
            pts.append(p0 + (j / n_seg) * edge)
    return np.array(pts) if pts else np.empty((0, 2))


def _seed_interior_grid(polygon, spacing):
    """Seed uniformly spaced interior points inside a polygon."""
    bounds = polygon.min(axis=0), polygon.max(axis=0)
    x = np.arange(bounds[0][0] + spacing / 2, bounds[1][0], spacing)
    y = np.arange(bounds[0][1] + spacing / 2, bounds[1][1], spacing)
    if len(x) == 0 or len(y) == 0:
        return np.empty((0, 2))
    xx, yy = np.meshgrid(x, y)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])
    mask = points_in_polygon(candidates, polygon)
    return candidates[mask]


def _remove_duplicates(pts, tol=1e-6):
    """Remove near-duplicate points."""
    if len(pts) <= 1:
        return pts
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=tol)
    remove = set()
    for i, j in pairs:
        remove.add(max(i, j))
    keep = np.array([i for i in range(len(pts)) if i not in remove])
    return pts[keep]


def _triangle_areas(nodes, elements):
    """Compute signed areas for triangles."""
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    return 0.5 * np.abs(cross)


def _ensure_ccw(nodes, elements):
    """Ensure all triangles have counter-clockwise ordering."""
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - \
            (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    # Swap nodes 1 and 2 for clockwise triangles
    cw = cross < 0
    elements[cw, 1], elements[cw, 2] = (
        elements[cw, 2].copy(), elements[cw, 1].copy())
    return elements


def _laplacian_smooth(nodes, elements, fixed_nodes, n_iter=3, weight=0.4):
    """Laplacian smoothing of interior nodes."""
    n = len(nodes)
    # Build adjacency
    neighbors = [set() for _ in range(n)]
    for tri in elements:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbors[tri[i]].add(tri[j])

    smoothed = nodes.copy()
    for _ in range(n_iter):
        new_pos = smoothed.copy()
        for i in range(n):
            if i in fixed_nodes or len(neighbors[i]) == 0:
                continue
            nbr = list(neighbors[i])
            avg = smoothed[nbr].mean(axis=0)
            new_pos[i] = smoothed[i] + weight * (avg - smoothed[i])
        smoothed = new_pos
    return smoothed
