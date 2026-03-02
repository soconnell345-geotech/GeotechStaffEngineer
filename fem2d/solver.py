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
from fem2d.assembly import element_dofs
from fem2d.materials import elastic_D, mc_return_mapping, hs_return_mapping


def solve_elastic(nodes, elements, D, gamma, bc_nodes, t=1.0,
                  surface_loads=None, pore_pressures=None):
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
    pore_pressures : (n_nodes,) array, optional — nodal pore pressures.
        When provided, adds pore pressure equivalent forces to F_ext.

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

    if pore_pressures is not None:
        from fem2d.porewater import pore_pressure_force
        F += pore_pressure_force(nodes, elements, pore_pressures, t)

    K_bc, F_bc = apply_bcs_penalty(K, F, bc_nodes)
    u = solve_linear(K_bc, F_bc)
    stresses, strains = recover_element_stresses(nodes, elements, D, u)

    return u, stresses, strains


def _do_return_mapping(mp, sigma_trial, elem_state_e):
    """Dispatch constitutive return mapping based on material model.

    Parameters
    ----------
    mp : dict — material properties for one element.
    sigma_trial : (3,) array — trial stress.
    elem_state_e : dict or None — HS state for this element.

    Returns
    -------
    sigma_new : (3,) array
    D_ep : (3, 3) array
    yielded : bool
    state_new : dict or None — updated HS state (None for MC/elastic).
    """
    model = mp.get('model', '')

    if model == 'hs':
        state = elem_state_e if elem_state_e is not None else {
            'gamma_p_s': 0.0,
            'sigma_prev': np.zeros(3),
            'loading': True,
        }
        sigma_new, D_ep, yielded, state_new = hs_return_mapping(
            sigma_trial, state,
            E50_ref=mp['E50_ref'], Eur_ref=mp['Eur_ref'],
            m=mp['m'], p_ref=mp['p_ref'], R_f=mp['R_f'],
            nu=mp['nu'], c=mp.get('c', 0), phi_deg=mp.get('phi', 0),
            psi_deg=mp.get('psi', 0))
        return sigma_new, D_ep, yielded, state_new

    # MC or elastic path
    if mp.get('c', 0) > 0 or mp.get('phi', 0) > 0:
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, mp['E'], mp['nu'],
            mp.get('c', 0), mp.get('phi', 0),
            mp.get('psi', 0))
        return sigma_new, D_ep, yielded, None
    else:
        D_e = elastic_D(mp['E'], mp['nu'])
        return sigma_trial.copy(), D_e, False, None


def solve_nonlinear(nodes, elements, material_props, gamma, bc_nodes,
                    t=1.0, n_steps=10, max_iter=100, tol=1e-5,
                    beam_elements=None, rotation_dof_map=None,
                    pore_pressures=None):
    """Solve a nonlinear (MC/HS) problem with Newton-Raphson.

    Uses incremental gravity loading with full Newton-Raphson iteration.
    Supports Mohr-Coulomb, Hardening Soil, and elastic materials.
    Optionally includes beam elements for structural members.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', 'c', 'phi', 'psi', 'gamma'}.
        For HS: add {'model': 'hs', 'E50_ref', 'Eur_ref', 'm', 'p_ref', 'R_f'}.
        If shorter than n_elements, the last entry is repeated.
    gamma : float or (n_elements,) array — unit weight.
    bc_nodes : dict from mesh.detect_boundary_nodes().
    t : float
    n_steps : int — number of load increments.
    max_iter : int — max NR iterations per step.
    tol : float — convergence tolerance on residual norm ratio.
    beam_elements : list of BeamElement, optional — structural beam elements.
    rotation_dof_map : dict, optional — {node_id: rotation_dof_index}.
    pore_pressures : (n_nodes,) array, optional — nodal pore pressures.
        When provided, adds pore pressure equivalent forces to F_ext.
        Constitutive model sees effective stress (total - pore pressure).

    Returns
    -------
    converged : bool
    u : (n_dof,) array — displacements.
    stresses : (n_elements, 3) array — element stresses.
    strains : (n_elements, 3) array — element strains.
    """
    from fem2d.assembly import assemble_gravity, apply_bcs_penalty

    n_nodes = len(nodes)
    n_elem = len(elements)

    # Total DOFs: translational + rotation (if beams present)
    if beam_elements and rotation_dof_map:
        n_dof = 2 * n_nodes + len(rotation_dof_map)
    else:
        n_dof = 2 * n_nodes

    # Expand material properties
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Pre-compute element pore pressures (centroidal average)
    if pore_pressures is not None:
        from fem2d.porewater import element_pore_pressures as _elem_pp
        from fem2d.porewater import effective_stress_correction as _eff_corr
        pp_elem = _elem_pp(nodes, elements, pore_pressures)
    else:
        pp_elem = None

    # Pre-compute element B matrices and areas (CST only)
    elem_data = []
    for e in range(n_elem):
        coords = nodes[elements[e]]
        B, A = cst_B(coords)
        elem_data.append((B, A))

    # Full gravity load (soil elements)
    F_gravity_soil = assemble_gravity(nodes, elements, gamma, t)
    # Add pore pressure equivalent forces
    if pore_pressures is not None:
        from fem2d.porewater import pore_pressure_force
        F_gravity_soil = F_gravity_soil + pore_pressure_force(
            nodes, elements, pore_pressures, t)
    # Extend to full DOF size if beams present
    F_gravity = np.zeros(n_dof)
    F_gravity[:2 * n_nodes] = F_gravity_soil

    # Add beam gravity if present
    if beam_elements and rotation_dof_map:
        from fem2d.assembly import (
            assemble_beam_gravity, beam_element_dofs,
            assemble_beam_stiffness,
        )
        from fem2d.elements import beam2d_stiffness, beam2d_internal_forces
        F_beam_grav = assemble_beam_gravity(
            nodes, beam_elements, rotation_dof_map, n_dof)
        F_gravity += F_beam_grav

    F_ext_norm = np.linalg.norm(F_gravity)
    if F_ext_norm < 1e-30:
        return True, np.zeros(n_dof), np.zeros((n_elem, 3)), np.zeros((n_elem, 3))

    # Initialize state
    u = np.zeros(n_dof)
    sigma_gp = np.zeros((n_elem, 3))
    epsilon_gp = np.zeros((n_elem, 3))

    # HS per-element state
    has_hs = any(mp.get('model') == 'hs' for mp in material_props)
    elem_state = [None] * n_elem
    if has_hs:
        for e in range(n_elem):
            if material_props[e].get('model') == 'hs':
                elem_state[e] = {
                    'gamma_p_s': 0.0,
                    'sigma_prev': np.zeros(3),
                    'loading': True,
                }

    # Penalty for BCs
    penalty = 1e20
    bc_dofs = set()
    for n in bc_nodes.get('fixed_base', []):
        bc_dofs.add(2 * n)
        bc_dofs.add(2 * n + 1)
        # Fix rotation DOF at base for beam nodes
        if rotation_dof_map and n in rotation_dof_map:
            bc_dofs.add(rotation_dof_map[n])
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
            tentative_states = [None] * n_elem

            for e in range(n_elem):
                B, A = elem_data[e]
                conn = elements[e]
                dofs = element_dofs(conn)
                u_e = u[dofs]
                mp = material_props[e]

                # Strain
                eps = B @ u_e
                d_eps = eps - epsilon_gp[e]

                # Trial stress (total)
                D_e = elastic_D(mp['E'], mp['nu'])
                sigma_trial = sigma_gp[e] + D_e @ d_eps

                # Convert to effective stress for return mapping
                if pp_elem is not None:
                    sigma_trial_eff = _eff_corr(sigma_trial, pp_elem[e])
                else:
                    sigma_trial_eff = sigma_trial

                # Return mapping on effective stress
                sigma_new_eff, D_ep, _, state_new = _do_return_mapping(
                    mp, sigma_trial_eff, elem_state[e])
                tentative_states[e] = state_new

                # Store total stress (add back pore pressure)
                if pp_elem is not None:
                    sigma_new = sigma_new_eff + pp_elem[e] * np.array([1., 1., 0.])
                else:
                    sigma_new = sigma_new_eff

                # Internal force: f_int_e = t * A * B^T * sigma (total)
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

            # Add beam contributions
            if beam_elements and rotation_dof_map:
                from fem2d.assembly import beam_element_dofs as _beam_dofs
                from fem2d.elements import (
                    beam2d_stiffness as _b2d_K,
                    beam2d_internal_forces as _b2d_f,
                )
                for beam in beam_elements:
                    coords_ij = np.array([
                        nodes[beam.node_i], nodes[beam.node_j]])
                    K_beam, _, _ = _b2d_K(coords_ij, beam.EA, beam.EI)
                    bdofs = _beam_dofs(
                        beam.node_i, beam.node_j, rotation_dof_map)
                    # Internal forces from beam
                    u_beam = u[bdofs]
                    f_int_beam = K_beam @ u_beam
                    F_int[bdofs] += f_int_beam
                    # Tangent stiffness
                    nb = len(bdofs)
                    for i in range(nb):
                        for j in range(nb):
                            rows.append(bdofs[i])
                            cols.append(bdofs[j])
                            vals.append(K_beam[i, j])

            # Build tangent stiffness
            K_T = coo_matrix((vals, (rows, cols)),
                             shape=(n_dof, n_dof)).tocsr()

            # Residual
            R = F_ext - F_int

            # Apply BCs to residual and tangent
            K_T_lil = K_T.tolil()
            for dof in bc_dofs:
                K_T_lil[dof, dof] += penalty
                R[dof] = -penalty * u[dof]

            K_T_csc = K_T_lil.tocsc()

            # Check convergence
            R_free = R.copy()
            for dof in bc_dofs:
                R_free[dof] = 0.0
            R_norm = np.linalg.norm(R_free)

            if R_norm / F_ext_norm < tol:
                break

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

            # Effective stress for return mapping
            if pp_elem is not None:
                sigma_trial_eff = _eff_corr(sigma_trial, pp_elem[e])
            else:
                sigma_trial_eff = sigma_trial

            sigma_new_eff, _, _, state_new = _do_return_mapping(
                mp, sigma_trial_eff, elem_state[e])

            # Store total stress
            if pp_elem is not None:
                sigma_new = sigma_new_eff + pp_elem[e] * np.array([1., 1., 0.])
            else:
                sigma_new = sigma_new_eff

            sigma_gp[e] = sigma_new
            epsilon_gp[e] = eps
            if state_new is not None:
                elem_state[e] = state_new

    stresses = sigma_gp.copy()
    strains = epsilon_gp.copy()

    return converged_overall, u, stresses, strains
