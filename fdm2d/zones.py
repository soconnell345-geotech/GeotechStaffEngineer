"""
Zone-level operations for explicit FDM.

Provides:
- Strain rate computation from gridpoint velocities
- Mixed discretization (Marti & Cundall 1982)
- Internal force computation (overlay-averaged)
- Gravity and surface pressure forces

All operations are element-by-element — no global matrix assembly.
"""

import numpy as np


def compute_strain_rates(velocities, sub_tris, B_all, n_zones):
    """Compute strain rates from gridpoint velocities.

    For each sub-triangle, strain_rate = B @ v_element, where
    v_element = [vx0, vy0, vx1, vy1, vx2, vy2] from the triangle nodes.

    Parameters
    ----------
    velocities : (n_gp, 2) array — gridpoint velocities.
    sub_tris : (n_zones, 4, 3) int array — sub-triangle connectivity.
    B_all : (n_zones, 4, 3, 6) array — precomputed B-matrices.
    n_zones : int

    Returns
    -------
    strain_rates : (n_zones, 4, 3) array — [eps_xx_dot, eps_yy_dot,
        gamma_xy_dot] per sub-triangle.
    """
    strain_rates = np.zeros((n_zones, 4, 3))

    for z in range(n_zones):
        for s in range(4):
            tri_nodes = sub_tris[z, s]
            # Build element velocity vector [vx0, vy0, vx1, vy1, vx2, vy2]
            v_elem = velocities[tri_nodes].ravel()  # (6,)
            strain_rates[z, s] = B_all[z, s] @ v_elem

    return strain_rates


def apply_mixed_discretization(strain_rates, areas):
    """Apply mixed discretization to prevent volumetric locking.

    Averages volumetric strain rates across all 4 sub-triangles of
    each zone while preserving deviatoric strain rates per sub-triangle.

    Parameters
    ----------
    strain_rates : (n_zones, 4, 3) array — raw strain rates.
    areas : (n_zones, 4) array — sub-triangle areas.

    Returns
    -------
    corrected : (n_zones, 4, 3) array — corrected strain rates.
    """
    n_zones = strain_rates.shape[0]
    corrected = strain_rates.copy()

    for z in range(n_zones):
        # Volumetric strain rate = eps_xx + eps_yy
        # Area-weighted average across all 4 sub-triangles
        total_area = areas[z].sum()
        if total_area < 1e-30:
            continue

        eps_vol_avg = 0.0
        for s in range(4):
            eps_vol = strain_rates[z, s, 0] + strain_rates[z, s, 1]
            eps_vol_avg += eps_vol * areas[z, s]
        eps_vol_avg /= total_area

        # Replace volumetric part, keep deviatoric
        for s in range(4):
            eps_vol_local = strain_rates[z, s, 0] + strain_rates[z, s, 1]
            correction = (eps_vol_avg - eps_vol_local) / 2.0
            corrected[z, s, 0] += correction
            corrected[z, s, 1] += correction

    return corrected


def compute_internal_forces(nodes, sub_tris, B_all, areas, stresses,
                            n_gp, t=1.0):
    """Compute internal forces from sub-triangle stresses.

    F_int = t * A * B^T @ sigma, averaged across the two overlays.

    Parameters
    ----------
    nodes : (n_gp, 2) array
    sub_tris : (n_zones, 4, 3) int array
    B_all : (n_zones, 4, 3, 6) array
    areas : (n_zones, 4) array
    stresses : (n_zones, 4, 3) array — sub-triangle stresses.
    n_gp : int
    t : float — thickness.

    Returns
    -------
    F_int : (n_gp, 2) array — internal nodal forces.
    """
    n_zones = sub_tris.shape[0]
    F_int = np.zeros((n_gp, 2))

    # Compute forces from each overlay separately, then average
    F_A = np.zeros((n_gp, 2))
    F_B = np.zeros((n_gp, 2))

    for z in range(n_zones):
        for s in range(4):
            f_elem = t * areas[z, s] * (B_all[z, s].T @ stresses[z, s])
            tri_nodes = sub_tris[z, s]
            target = F_A if s < 2 else F_B
            for loc, nid in enumerate(tri_nodes):
                target[nid, 0] += f_elem[2 * loc]
                target[nid, 1] += f_elem[2 * loc + 1]

    F_int = (F_A + F_B) / 2.0
    return F_int


def compute_gravity_forces(nodes, zones, areas, gamma, n_gp, t=1.0):
    """Compute gravity body forces.

    Gravity acts downward (negative y in tension-positive convention).
    Body force per unit volume: bx=0, by=-gamma.
    Force distributed equally to 4 corner nodes per zone.

    Parameters
    ----------
    nodes : (n_gp, 2) array
    zones : (n_zones, 4) int array
    areas : (n_zones, 4) array — sub-triangle areas.
    gamma : float or (n_zones,) array — unit weight (kN/m³).
    n_gp : int
    t : float — thickness.

    Returns
    -------
    F_grav : (n_gp, 2) array — gravity nodal forces.
    """
    n_zones = len(zones)
    F_grav = np.zeros((n_gp, 2))
    gamma_arr = np.broadcast_to(gamma, (n_zones,))

    for z in range(n_zones):
        # Zone area from overlay average
        zone_area = (areas[z, 0] + areas[z, 1] + areas[z, 2] +
                     areas[z, 3]) / 2.0
        total_weight = gamma_arr[z] * zone_area * t
        # Distribute equally to 4 corner nodes, downward
        for nid in zones[z]:
            F_grav[nid, 1] -= total_weight / 4.0

    return F_grav


def compute_surface_pressure(nodes, edges, qx, qy, n_gp, t=1.0):
    """Compute surface pressure forces on boundary edges.

    Parameters
    ----------
    nodes : (n_gp, 2) array
    edges : list of (node_i, node_j) — boundary edge pairs.
    qx : float — horizontal pressure (kPa, tension-positive).
    qy : float — vertical pressure (kPa, tension-positive).
    n_gp : int
    t : float — thickness.

    Returns
    -------
    F_surf : (n_gp, 2) array — surface pressure forces.
    """
    F_surf = np.zeros((n_gp, 2))

    for ni, nj in edges:
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        edge_len = np.sqrt(dx ** 2 + dy ** 2)
        # Force = pressure × edge_length × thickness / 2 per node
        fx = qx * edge_len * t / 2.0
        fy = qy * edge_len * t / 2.0
        F_surf[ni, 0] += fx
        F_surf[ni, 1] += fy
        F_surf[nj, 0] += fx
        F_surf[nj, 1] += fy

    return F_surf


def zone_averaged_stress(stresses):
    """Average sub-triangle stresses to zone-level for reporting.

    Parameters
    ----------
    stresses : (n_zones, 4, 3) array — sub-triangle stresses.

    Returns
    -------
    zone_stress : (n_zones, 3) array — zone-averaged [sxx, syy, txy].
    """
    return stresses.mean(axis=1)
