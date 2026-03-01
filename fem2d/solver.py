"""
FEM solvers: linear elastic and nonlinear Newton-Raphson.

The linear solver assembles K and F once and solves directly.
The nonlinear solver implements incremental gravity loading with
Newton-Raphson iteration for elastoplastic materials.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from fem2d.elements import cst_B, cst_area
from fem2d.materials import elastic_D, mc_return_mapping


def element_dofs(conn):
    """Map element connectivity to global DOF indices."""
    dofs = []
    for n in conn:
        dofs.extend([2 * n, 2 * n + 1])
    return np.array(dofs, dtype=int)


def solve_elastic(nodes, elements, D, gamma, bc_nodes, t=1.0,
                  surface_loads=None):
    """Solve a linear elastic problem.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    D : (3, 3) array or list — constitutive matrix.
    gamma : float or (n_elements,) array — unit weight (kN/m³).
    bc_nodes : dict from mesh.detect_boundary_nodes().
    t : float — thickness.
    surface_loads : list of (edge_nodes, qx, qy), optional.

    Returns
    -------
    u : (n_dof,) array — displacements.
    stresses : (n_elements, 3) array — element stresses.
    strains : (n_elements, 3) array — element strains.
    """
    from fem2d.assembly import (
        assemble_stiffness, assemble_gravity, assemble_surface_load,
        apply_bcs_penalty, solve_linear, recover_element_stresses,
    )

    K = assemble_stiffness(nodes, elements, D, t)
    F = assemble_gravity(nodes, elements, gamma, t)

    if surface_loads:
        for edges, qx, qy in surface_loads:
            F += assemble_surface_load(nodes, edges, qx, qy, t)

    K_bc, F_bc = apply_bcs_penalty(K, F, bc_nodes)
    u = solve_linear(K_bc, F_bc)
    stresses, strains = recover_element_stresses(nodes, elements, D, u)

    return u, stresses, strains


def solve_nonlinear(nodes, elements, material_props, gamma, bc_nodes,
                    t=1.0, n_steps=10, max_iter=100, tol=1e-5):
    """Solve a nonlinear (Mohr-Coulomb) problem with Newton-Raphson.

    Uses incremental gravity loading with full Newton-Raphson iteration.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', 'c', 'phi', 'psi', 'gamma'}.
        If shorter than n_elements, the last entry is repeated.
    gamma : float or (n_elements,) array — unit weight.
    bc_nodes : dict from mesh.detect_boundary_nodes().
    t : float
    n_steps : int — number of load increments.
    max_iter : int — max NR iterations per step.
    tol : float — convergence tolerance on residual norm ratio.

    Returns
    -------
    converged : bool
    u : (n_dof,) array — displacements.
    stresses : (n_elements, 3) array — element stresses.
    strains : (n_elements, 3) array — element strains.
    """
    from fem2d.assembly import assemble_gravity, apply_bcs_penalty

    n_dof = 2 * len(nodes)
    n_elem = len(elements)

    # Expand material properties
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Pre-compute element B matrices and areas (CST only)
    elem_data = []
    for e in range(n_elem):
        coords = nodes[elements[e]]
        B, A = cst_B(coords)
        elem_data.append((B, A))

    # Full gravity load
    F_gravity = assemble_gravity(nodes, elements, gamma, t)
    F_ext_norm = np.linalg.norm(F_gravity)
    if F_ext_norm < 1e-30:
        # No gravity — return zero solution
        return True, np.zeros(n_dof), np.zeros((n_elem, 3)), np.zeros((n_elem, 3))

    # Initialize state
    u = np.zeros(n_dof)
    sigma_gp = np.zeros((n_elem, 3))  # converged stress at each element
    epsilon_gp = np.zeros((n_elem, 3))

    # Penalty for BCs
    penalty = 1e20
    bc_dofs = set()
    for n in bc_nodes.get('fixed_base', []):
        bc_dofs.add(2 * n)
        bc_dofs.add(2 * n + 1)
    for key in ['roller_left', 'roller_right']:
        for n in bc_nodes.get(key, []):
            bc_dofs.add(2 * n)

    converged_overall = True

    for step in range(1, n_steps + 1):
        F_ext = (step / n_steps) * F_gravity

        for iteration in range(max_iter):
            # Assemble internal forces and tangent stiffness
            F_int = np.zeros(n_dof)
            rows, cols, vals = [], [], []

            for e in range(n_elem):
                B, A = elem_data[e]
                conn = elements[e]
                dofs = element_dofs(conn)
                u_e = u[dofs]
                mp = material_props[e]

                # Strain
                eps = B @ u_e
                d_eps = eps - epsilon_gp[e]

                # Trial stress and return mapping
                D_e = elastic_D(mp['E'], mp['nu'])
                sigma_trial = sigma_gp[e] + D_e @ d_eps

                if mp.get('c', 0) > 0 or mp.get('phi', 0) > 0:
                    sigma_new, D_ep, _ = mc_return_mapping(
                        sigma_trial, mp['E'], mp['nu'],
                        mp.get('c', 0), mp.get('phi', 0),
                        mp.get('psi', 0))
                else:
                    sigma_new = sigma_trial
                    D_ep = D_e

                # Internal force: f_int_e = t * A * B^T * sigma
                f_int_e = t * A * (B.T @ sigma_new)
                F_int[dofs] += f_int_e

                # Tangent stiffness
                Ke = t * A * (B.T @ D_ep @ B)
                ne = len(dofs)
                for i in range(ne):
                    for j in range(ne):
                        rows.append(dofs[i])
                        cols.append(dofs[j])
                        vals.append(Ke[i, j])

            # Build tangent stiffness
            K_T = coo_matrix((vals, (rows, cols)),
                             shape=(n_dof, n_dof)).tocsr()

            # Residual
            R = F_ext - F_int

            # Apply BCs to residual and tangent
            K_T_lil = K_T.tolil()
            for dof in bc_dofs:
                K_T_lil[dof, dof] += penalty
                R[dof] = -penalty * u[dof]  # enforce u=0

            K_T_csc = K_T_lil.tocsc()

            # Check convergence
            R_free = R.copy()
            for dof in bc_dofs:
                R_free[dof] = 0.0
            R_norm = np.linalg.norm(R_free)

            if R_norm / F_ext_norm < tol:
                break  # Converged

            if R_norm > 1e12 or np.any(np.isnan(R)):
                converged_overall = False
                break

            # Solve for displacement increment
            try:
                du = spsolve(K_T_csc, R)
            except Exception:
                converged_overall = False
                break

            if np.any(np.isnan(du)) or np.any(np.abs(du) > 1e6):
                converged_overall = False
                break

            u += du

        else:
            # max_iter reached without converging
            converged_overall = False

        if not converged_overall:
            break

        # Update converged state
        for e in range(n_elem):
            B, A = elem_data[e]
            conn = elements[e]
            dofs = element_dofs(conn)
            u_e = u[dofs]
            eps = B @ u_e
            d_eps = eps - epsilon_gp[e]
            mp = material_props[e]
            D_e = elastic_D(mp['E'], mp['nu'])
            sigma_trial = sigma_gp[e] + D_e @ d_eps

            if mp.get('c', 0) > 0 or mp.get('phi', 0) > 0:
                sigma_new, _, _ = mc_return_mapping(
                    sigma_trial, mp['E'], mp['nu'],
                    mp.get('c', 0), mp.get('phi', 0),
                    mp.get('psi', 0))
            else:
                sigma_new = sigma_trial

            sigma_gp[e] = sigma_new
            epsilon_gp[e] = eps

    # Final stresses
    stresses = sigma_gp.copy()
    strains = epsilon_gp.copy()

    return converged_overall, u, stresses, strains
