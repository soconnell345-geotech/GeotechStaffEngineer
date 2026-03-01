"""
Global FEM assembly, boundary condition application, and linear solver.

Assembles element stiffness matrices and force vectors into sparse global
system, applies boundary conditions via penalty method, and solves using
scipy.sparse.linalg.spsolve.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from fem2d.elements import (
    cst_stiffness, cst_body_force, cst_stress,
    q4_stiffness, q4_body_force, q4_stress,
)
from fem2d.materials import elastic_D


# ---------------------------------------------------------------------------
# DOF mapping
# ---------------------------------------------------------------------------

def element_dofs(conn):
    """Map element node connectivity to global DOF indices.

    Parameters
    ----------
    conn : array-like of int — global node indices for one element.

    Returns
    -------
    dofs : (n_dof_per_element,) array of int
    """
    dofs = []
    for n in conn:
        dofs.extend([2 * n, 2 * n + 1])
    return np.array(dofs, dtype=int)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble_stiffness(nodes, elements, D, t=1.0):
    """Assemble global stiffness matrix (sparse CSR).

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 4) array — connectivity.
    D : (3, 3) array or list of (3, 3) arrays — constitutive matrix.
        If a list, D[e] is used for element e.
    t : float — thickness.

    Returns
    -------
    K : sparse CSR matrix (n_dof × n_dof)
    """
    n_dof = 2 * len(nodes)
    rows, cols, vals = [], [], []

    D_is_list = isinstance(D, (list, np.ndarray)) and np.ndim(D) == 3

    for e in range(len(elements)):
        conn = elements[e]
        coords = nodes[conn]
        De = D[e] if D_is_list else D
        n_nodes_e = len(conn)

        if n_nodes_e == 3:
            Ke = cst_stiffness(coords, De, t)
        else:
            Ke = q4_stiffness(coords, De, t)

        dofs = element_dofs(conn)
        ne = len(dofs)
        for i in range(ne):
            for j in range(ne):
                rows.append(dofs[i])
                cols.append(dofs[j])
                vals.append(Ke[i, j])

    K = coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


def assemble_gravity(nodes, elements, gamma, t=1.0):
    """Assemble gravity body force vector.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 4) array
    gamma : float or (n_elements,) array — unit weight (kN/m³).
        If array, gamma[e] is used for element e.
    t : float

    Returns
    -------
    F : (n_dof,) array
    """
    n_dof = 2 * len(nodes)
    F = np.zeros(n_dof)

    gamma_is_array = hasattr(gamma, '__len__')

    for e in range(len(elements)):
        conn = elements[e]
        coords = nodes[conn]
        g = gamma[e] if gamma_is_array else gamma

        if len(conn) == 3:
            fe = cst_body_force(coords, 0.0, -g, t)
        else:
            fe = q4_body_force(coords, 0.0, -g, t)

        dofs = element_dofs(conn)
        F[dofs] += fe

    return F


def assemble_surface_load(nodes, surface_edges, qx=0.0, qy=0.0, t=1.0):
    """Assemble surface traction load vector.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    surface_edges : list of (node_i, node_j) — edges on the loaded surface.
    qx, qy : float — traction (kPa) in x and y directions.
        Negative qy = downward pressure.
    t : float

    Returns
    -------
    F : (n_dof,) array
    """
    n_dof = 2 * len(nodes)
    F = np.zeros(n_dof)

    for ni, nj in surface_edges:
        L = np.linalg.norm(nodes[nj] - nodes[ni])
        # Equal distribution to both nodes (linear edge)
        f_node = t * L / 2.0
        F[2 * ni] += f_node * qx
        F[2 * ni + 1] += f_node * qy
        F[2 * nj] += f_node * qx
        F[2 * nj + 1] += f_node * qy

    return F


# ---------------------------------------------------------------------------
# Boundary conditions (penalty method)
# ---------------------------------------------------------------------------

def apply_bcs_penalty(K, F, bc_nodes, penalty=None):
    """Apply boundary conditions using penalty method.

    Parameters
    ----------
    K : sparse CSR matrix
    F : (n_dof,) array
    bc_nodes : dict from detect_boundary_nodes() with keys:
        'fixed_base', 'roller_left', 'roller_right'
    penalty : float, optional — penalty value. Default: 1e20.

    Returns
    -------
    K_mod : sparse CSR matrix (modified)
    F_mod : (n_dof,) array (modified)
    """
    if penalty is None:
        penalty = 1e20

    K_lil = K.tolil()
    F_mod = F.copy()

    # Fixed base: u = v = 0
    for n in bc_nodes.get('fixed_base', []):
        for dof in [2 * n, 2 * n + 1]:
            K_lil[dof, dof] += penalty
            F_mod[dof] += penalty * 0.0

    # Roller left/right: u = 0
    for key in ['roller_left', 'roller_right']:
        for n in bc_nodes.get(key, []):
            dof = 2 * n  # x-DOF
            K_lil[dof, dof] += penalty
            F_mod[dof] += penalty * 0.0

    return K_lil.tocsr(), F_mod


def apply_prescribed_displacements(K, F, dof_val_pairs, penalty=None):
    """Apply arbitrary prescribed displacements.

    Parameters
    ----------
    K : sparse CSR matrix
    F : (n_dof,) array
    dof_val_pairs : list of (dof_index, displacement_value)
    penalty : float

    Returns
    -------
    K_mod, F_mod
    """
    if penalty is None:
        penalty = 1e20

    K_lil = K.tolil()
    F_mod = F.copy()

    for dof, val in dof_val_pairs:
        K_lil[dof, dof] += penalty
        F_mod[dof] += penalty * val

    return K_lil.tocsr(), F_mod


# ---------------------------------------------------------------------------
# Linear solve
# ---------------------------------------------------------------------------

def solve_linear(K, F):
    """Solve K·u = F using sparse direct solver.

    Parameters
    ----------
    K : sparse matrix
    F : (n_dof,) array

    Returns
    -------
    u : (n_dof,) array — nodal displacements.
    """
    return spsolve(K.tocsc(), F)


# ---------------------------------------------------------------------------
# Stress recovery
# ---------------------------------------------------------------------------

def recover_element_stresses(nodes, elements, D, u):
    """Compute element stresses from nodal displacements.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 4) array
    D : (3, 3) array or list of (3, 3) arrays
    u : (n_dof,) array — nodal displacements.

    Returns
    -------
    stresses : (n_elements, 3) array — [sigma_x, sigma_y, tau_xy] per element.
    strains : (n_elements, 3) array — [eps_x, eps_y, gamma_xy] per element.
    """
    D_is_list = isinstance(D, (list, np.ndarray)) and np.ndim(D) == 3
    n_elem = len(elements)
    stresses = np.zeros((n_elem, 3))
    strains = np.zeros((n_elem, 3))

    for e in range(n_elem):
        conn = elements[e]
        coords = nodes[conn]
        dofs = element_dofs(conn)
        u_e = u[dofs]
        De = D[e] if D_is_list else D

        if len(conn) == 3:
            sig, eps = cst_stress(coords, De, u_e)
        else:
            sig, eps = q4_stress(coords, De, u_e)

        stresses[e] = sig
        strains[e] = eps

    return stresses, strains


def nodal_stresses(nodes, elements, element_stresses):
    """Average element stresses to nodes.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 4) array
    element_stresses : (n_elements, 3) array

    Returns
    -------
    sigma_nodal : (n_nodes, 3) array — averaged nodal stresses.
    """
    n_nodes = len(nodes)
    sigma_sum = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    for e in range(len(elements)):
        for n in elements[e]:
            sigma_sum[n] += element_stresses[e]
            count[n] += 1

    count[count == 0] = 1  # avoid division by zero
    return sigma_sum / count[:, np.newaxis]
