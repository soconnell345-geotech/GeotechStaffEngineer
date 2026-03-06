"""
Quad grid generation and sub-triangle decomposition for explicit FDM.

Provides:
- Rectangular quad grid generation (row-by-row from bottom-left)
- Mixed discretization sub-triangle decomposition (2 overlays × 2 tris)
- CST B-matrix and area for each sub-triangle
- Boundary detection for standard geotechnical BCs
- Lumped mass calculation

Zone node ordering: CCW [SW, SE, NE, NW] = [0, 1, 2, 3].
"""

import numpy as np


def generate_quad_grid(x_min, x_max, y_min, y_max, nx, ny):
    """Generate a rectangular quad grid.

    Nodes numbered row-by-row from bottom-left.
    Zones use CCW ordering: [SW, SE, NE, NW].

    Parameters
    ----------
    x_min, x_max : float — horizontal extent (m).
    y_min, y_max : float — vertical extent (m).
    nx : int — number of zones in x-direction.
    ny : int — number of zones in y-direction.

    Returns
    -------
    nodes : (n_gp, 2) array — gridpoint coordinates.
    zones : (n_zones, 4) int array — zone connectivity (CCW).
    """
    n_gp_x = nx + 1
    n_gp_y = ny + 1

    x = np.linspace(x_min, x_max, n_gp_x)
    y = np.linspace(y_min, y_max, n_gp_y)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    zones = np.empty((nx * ny, 4), dtype=int)
    idx = 0
    for j in range(ny):
        for i in range(nx):
            sw = j * n_gp_x + i
            se = sw + 1
            ne = se + n_gp_x
            nw = sw + n_gp_x
            zones[idx] = [sw, se, ne, nw]
            idx += 1

    return nodes, zones


def build_sub_triangles(zones):
    """Build 4 sub-triangles per zone for mixed discretization.

    Each quad zone is split into 2 overlapping triangle pairs:
        Overlay A: tri(0,1,2), tri(0,2,3)   [diagonal 0-2]
        Overlay B: tri(0,1,3), tri(1,2,3)   [diagonal 1-3]

    Parameters
    ----------
    zones : (n_zones, 4) int array — zone connectivity [SW, SE, NE, NW].

    Returns
    -------
    sub_tris : (n_zones, 4, 3) int array — node indices per sub-triangle.
        sub_tris[z, 0:2] = overlay A triangles
        sub_tris[z, 2:4] = overlay B triangles
    """
    n_zones = len(zones)
    sub_tris = np.empty((n_zones, 4, 3), dtype=int)

    for z in range(n_zones):
        n0, n1, n2, n3 = zones[z]
        # Overlay A (diagonal 0-2)
        sub_tris[z, 0] = [n0, n1, n2]
        sub_tris[z, 1] = [n0, n2, n3]
        # Overlay B (diagonal 1-3)
        sub_tris[z, 2] = [n0, n1, n3]
        sub_tris[z, 3] = [n1, n2, n3]

    return sub_tris


def _cst_B_area(coords):
    """Compute CST B-matrix (3×6) and area for a triangle.

    Parameters
    ----------
    coords : (3, 2) array — triangle node coordinates.

    Returns
    -------
    B : (3, 6) array — strain-displacement matrix.
    A : float — triangle area.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = abs(A2) / 2.0

    b1, c1 = y2 - y3, x3 - x2
    b2, c2 = y3 - y1, x1 - x3
    b3, c3 = y1 - y2, x2 - x1

    B = (1.0 / A2) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, A


def compute_sub_triangle_geometry(nodes, sub_tris):
    """Precompute B-matrices and areas for all sub-triangles.

    Parameters
    ----------
    nodes : (n_gp, 2) array — gridpoint coordinates.
    sub_tris : (n_zones, 4, 3) int array — sub-triangle connectivity.

    Returns
    -------
    B_all : (n_zones, 4, 3, 6) array — B-matrix per sub-triangle.
    areas : (n_zones, 4) array — area per sub-triangle.
    """
    n_zones = sub_tris.shape[0]
    B_all = np.zeros((n_zones, 4, 3, 6))
    areas = np.zeros((n_zones, 4))

    for z in range(n_zones):
        for s in range(4):
            tri_nodes = sub_tris[z, s]
            coords = nodes[tri_nodes]
            B, A = _cst_B_area(coords)
            B_all[z, s] = B
            areas[z, s] = A

    return B_all, areas


def detect_boundary_gridpoints(nodes, tol=1e-6):
    """Detect boundary gridpoints for standard geotechnical BCs.

    Standard BCs:
    - fixed_base: bottom row — fix both x and y
    - roller_left: left column (excluding corners) — fix x
    - roller_right: right column (excluding corners) — fix x

    Parameters
    ----------
    nodes : (n_gp, 2) array
    tol : float — coordinate tolerance.

    Returns
    -------
    bc : dict with keys 'fixed_base', 'roller_left', 'roller_right'.
        Each value is an array of node indices.
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min = y.min()

    base = np.where(np.abs(y - y_min) < tol)[0]
    left = np.where((np.abs(x - x_min) < tol) & (np.abs(y - y_min) >= tol))[0]
    right = np.where(
        (np.abs(x - x_max) < tol) & (np.abs(y - y_min) >= tol))[0]

    return {
        'fixed_base': base,
        'roller_left': left,
        'roller_right': right,
    }


def compute_lumped_mass(nodes, zones, sub_tris, areas, rho, t=1.0):
    """Compute lumped nodal mass from zone areas and density.

    Mass is distributed equally to the 4 corner nodes of each zone,
    using the average area of its 4 sub-triangles (which equals the
    zone area since overlay tris cover the same quad).

    Parameters
    ----------
    nodes : (n_gp, 2) array
    zones : (n_zones, 4) int array
    sub_tris : (n_zones, 4, 3) int array
    areas : (n_zones, 4) array — sub-triangle areas.
    rho : float or (n_zones,) array — mass density (kN·s²/m⁴).
    t : float — out-of-plane thickness (m).

    Returns
    -------
    mass : (n_gp,) array — lumped mass per gridpoint.
    """
    n_gp = len(nodes)
    n_zones = len(zones)
    mass = np.zeros(n_gp)

    rho_arr = np.broadcast_to(rho, (n_zones,))

    for z in range(n_zones):
        # Zone area = average of overlay A area + overlay B area
        # Overlay A: sub_tris 0,1; Overlay B: sub_tris 2,3
        zone_area = (areas[z, 0] + areas[z, 1] + areas[z, 2] +
                     areas[z, 3]) / 2.0
        zone_mass = rho_arr[z] * zone_area * t
        # Distribute equally to 4 corner nodes
        for node_id in zones[z]:
            mass[node_id] += zone_mass / 4.0

    return mass
