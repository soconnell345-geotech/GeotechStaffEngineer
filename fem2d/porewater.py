"""
Pore water pressures, steady-state seepage, and coupled Biot consolidation.

Provides:
- Static pore pressure computation from GWT definition
- Effective stress correction for total → effective stress
- Pore pressure equivalent nodal forces
- Steady-state seepage solver (Laplace equation, CST flow elements)
- Coupled Biot consolidation (staggered scheme)

Sign convention (tension-positive code):
    Compression = negative sigma
    Pore pressure u > 0 below GWT
    Effective stress: sigma' = sigma + u * m  where m = [1, 1, 0]^T
    (adding positive u makes effective stress less compressive = less confining)
    Total from effective: sigma_total = sigma' - u * m

References:
    Biot (1941) — General theory of three-dimensional consolidation
    Verruijt (1969) — Elastic storage of aquifers
    Smith & Griffiths (2004) — Programming the Finite Element Method
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from fem2d.elements import cst_B, cst_area
from fem2d.assembly import element_dofs


# ---------------------------------------------------------------------------
# Voigt identity vector for pore pressure coupling
# ---------------------------------------------------------------------------

_M_VOIGT = np.array([1.0, 1.0, 0.0])  # [sigma_x, sigma_y, tau_xy]


# ===========================================================================
# Phase 1: Static Pore Pressure Field
# ===========================================================================

def compute_pore_pressures(nodes, gwt, gamma_w=9.81):
    """Compute nodal pore pressures from groundwater table definition.

    Parameters
    ----------
    nodes : (n_nodes, 2) array — node coordinates [x, y].
    gwt : float, (M, 2) array, or (n_nodes,) array
        - float: constant GWT elevation (hydrostatic below this level).
        - (M, 2) array: polyline [(x1, z_gwt1), (x2, z_gwt2), ...].
          GWT elevation is linearly interpolated between points.
        - (n_nodes,) array: per-node prescribed head (for artesian).
    gamma_w : float — unit weight of water (kN/m^3). Default 9.81.

    Returns
    -------
    pore_pressures : (n_nodes,) array — u >= 0 below GWT, 0 above.
    """
    nodes = np.asarray(nodes)
    n_nodes = len(nodes)
    pp = np.zeros(n_nodes)

    gwt_arr = np.asarray(gwt)

    if gwt_arr.ndim == 0:
        # Constant GWT elevation
        z_gwt = float(gwt_arr)
        for i in range(n_nodes):
            depth_below = z_gwt - nodes[i, 1]
            if depth_below > 0:
                pp[i] = gamma_w * depth_below

    elif gwt_arr.ndim == 1 and len(gwt_arr) == n_nodes:
        # Per-node prescribed head (artesian)
        for i in range(n_nodes):
            depth_below = gwt_arr[i] - nodes[i, 1]
            if depth_below > 0:
                pp[i] = gamma_w * depth_below

    elif gwt_arr.ndim == 2 and gwt_arr.shape[1] == 2:
        # Polyline GWT: interpolate z_gwt at each node's x-coordinate
        gwt_sorted = gwt_arr[np.argsort(gwt_arr[:, 0])]
        x_gwt = gwt_sorted[:, 0]
        z_gwt = gwt_sorted[:, 1]
        for i in range(n_nodes):
            z_gwt_at_node = np.interp(nodes[i, 0], x_gwt, z_gwt)
            depth_below = z_gwt_at_node - nodes[i, 1]
            if depth_below > 0:
                pp[i] = gamma_w * depth_below

    else:
        raise ValueError(
            "gwt must be a float (constant elevation), (M,2) array "
            "(polyline), or (n_nodes,) array (per-node head).")

    return pp


def element_pore_pressures(nodes, elements, nodal_pp):
    """Average nodal pore pressures to element centroids.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    nodal_pp : (n_nodes,) array — nodal pore pressures.

    Returns
    -------
    (n_elements,) array of centroidal pore pressures.
    """
    nodal_pp = np.asarray(nodal_pp)
    n_elem = len(elements)
    pp_elem = np.zeros(n_elem)
    for e in range(n_elem):
        pp_elem[e] = nodal_pp[elements[e]].mean()
    return pp_elem


def effective_stress_correction(sigma_total_3, u_pore):
    """Convert total stress to effective stress (tension-positive).

    sigma_eff = sigma_total + u * m  where m = [1, 1, 0]^T

    In tension-positive convention, compression is negative. Adding
    positive u makes effective stress less compressive (less negative),
    i.e. lower confining pressure → lower shear strength.

    Parameters
    ----------
    sigma_total_3 : (3,) array — [sigma_x, sigma_y, tau_xy] total stress.
    u_pore : float — pore water pressure (positive below GWT).

    Returns
    -------
    (3,) array — effective stress.
    """
    return np.asarray(sigma_total_3) + u_pore * _M_VOIGT


def pore_pressure_force(nodes, elements, nodal_pp, t=1.0,
                        active_elements=None):
    """Assemble equivalent nodal force vector from pore pressures.

    For each CST element:
        f_p = t * A * B^T * m * u_avg
    where m = [1, 1, 0]^T and u_avg = mean of nodal pore pressures.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    nodal_pp : (n_nodes,) array — nodal pore pressures.
    t : float — thickness.
    active_elements : set of int, optional — element indices to include.
        None means all elements are active.

    Returns
    -------
    F_pp : (2*n_nodes,) force vector to add to external loads.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    nodal_pp = np.asarray(nodal_pp)
    n_dof = 2 * len(nodes)
    F_pp = np.zeros(n_dof)

    if active_elements is not None:
        active_elements = set(active_elements)

    for e in range(len(elements)):
        if active_elements is not None and e not in active_elements:
            continue
        conn = elements[e]
        coords = nodes[conn]
        B, A = cst_B(coords)
        u_avg = nodal_pp[conn].mean()

        # f_p = t * A * B^T * m * u_avg
        f_e = t * A * (B.T @ _M_VOIGT) * u_avg
        dofs = element_dofs(conn)
        F_pp[dofs] += f_e

    return F_pp


# ===========================================================================
# Phase 2: Steady-State Seepage Solver
# ===========================================================================

def cst_permeability_matrix(coords, k, t=1.0):
    """CST element permeability (flow) matrix.

    H_e = k * t * A * G^T * G   where G = [dN/dx; dN/dy] (2x3)
    Uses same shape function derivatives as CST B-matrix.

    Parameters
    ----------
    coords : (3, 2) array — element node coordinates.
    k : float — isotropic hydraulic conductivity (m/s).
    t : float — thickness.

    Returns
    -------
    H_e : (3, 3) array — element permeability matrix.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = abs(A2) / 2.0

    # Shape function derivatives: dN/dx = b_i / (2A), dN/dy = c_i / (2A)
    b1, c1 = y2 - y3, x3 - x2
    b2, c2 = y3 - y1, x1 - x3
    b3, c3 = y1 - y2, x2 - x1

    # Gradient matrix G (2x3): [dN1/dx, dN2/dx, dN3/dx; dN1/dy, dN2/dy, dN3/dy]
    G = (1.0 / A2) * np.array([
        [b1, b2, b3],
        [c1, c2, c3],
    ])

    # H_e = k * t * A * G^T * G
    return k * t * A * (G.T @ G)


def assemble_flow_system(nodes, elements, k, t=1.0):
    """Assemble global permeability matrix H and zero RHS vector.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    k : float or (n_elements,) array — hydraulic conductivity (m/s).
    t : float — thickness.

    Returns
    -------
    H : sparse CSR (n_nodes x n_nodes) — global permeability matrix.
    q : (n_nodes,) array — RHS vector (initially zero).
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_nodes = len(nodes)
    k_arr = np.asarray(k)
    k_is_array = k_arr.ndim > 0 and len(k_arr) == len(elements)

    rows, cols, vals = [], [], []

    for e in range(len(elements)):
        conn = elements[e]
        coords = nodes[conn]
        ke = k_arr[e] if k_is_array else float(k_arr)
        H_e = cst_permeability_matrix(coords, ke, t)

        for i in range(3):
            for j in range(3):
                rows.append(conn[i])
                cols.append(conn[j])
                vals.append(H_e[i, j])

    H = coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    q = np.zeros(n_nodes)
    return H, q


def apply_head_bcs(H, q, prescribed_heads, penalty=1e20):
    """Apply Dirichlet head BCs via penalty method.

    Parameters
    ----------
    H : sparse matrix — global permeability matrix.
    q : (n_nodes,) array — RHS vector.
    prescribed_heads : list of (node_id, head_value)
    penalty : float — penalty value.

    Returns
    -------
    H_mod : sparse CSR matrix
    q_mod : (n_nodes,) array
    """
    H_lil = H.tolil()
    q_mod = q.copy()

    for node, head_val in prescribed_heads:
        H_lil[node, node] += penalty
        q_mod[node] += penalty * head_val

    return H_lil.tocsr(), q_mod


def seepage_velocity(nodes, elements, head, k):
    """Compute element Darcy velocities from head field.

    v = -k * grad(h), grad computed via CST shape function derivatives.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    head : (n_nodes,) array — total head at each node.
    k : float or (n_elements,) array — hydraulic conductivity.

    Returns
    -------
    velocity : (n_elements, 2) array of [vx, vy] per element.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    head = np.asarray(head)
    k_arr = np.asarray(k)
    k_is_array = k_arr.ndim > 0 and len(k_arr) == len(elements)

    n_elem = len(elements)
    velocity = np.zeros((n_elem, 2))

    for e in range(n_elem):
        conn = elements[e]
        coords = nodes[conn]
        ke = k_arr[e] if k_is_array else float(k_arr)

        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)

        b1, c1 = y2 - y3, x3 - x2
        b2, c2 = y3 - y1, x1 - x3
        b3, c3 = y1 - y2, x2 - x1

        G = (1.0 / A2) * np.array([
            [b1, b2, b3],
            [c1, c2, c3],
        ])

        h_e = head[conn]
        grad_h = G @ h_e  # [dh/dx, dh/dy]
        velocity[e] = -ke * grad_h

    return velocity


def solve_seepage(nodes, elements, k, head_bcs, t=1.0,
                  gamma_w=9.81, flow_bcs=None):
    """Solve steady-state seepage problem.

    Solves the Laplace equation for hydraulic head using CST elements,
    then computes pore pressures from the head field.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    k : float or (n_elements,) array — hydraulic conductivity (m/s).
    head_bcs : list of (node_id, head_value) — Dirichlet BCs.
    t : float — thickness.
    gamma_w : float — unit weight of water (kN/m^3).
    flow_bcs : list of (node_id, flow_rate), optional — Neumann BCs
        (added to RHS).

    Returns
    -------
    dict with keys:
        head : (n_nodes,) — total head at each node.
        pore_pressures : (n_nodes,) — u = gamma_w * (h - z), clipped >= 0.
        velocity : (n_elements, 2) — Darcy velocity per element.
        flow_rate : float — total flow through domain.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)

    H, q = assemble_flow_system(nodes, elements, k, t)

    # Apply Neumann BCs (prescribed flow)
    if flow_bcs:
        for node, flow_val in flow_bcs:
            q[node] += flow_val

    # Apply Dirichlet BCs
    H_bc, q_bc = apply_head_bcs(H, q, head_bcs)

    # Solve
    head = spsolve(H_bc.tocsc(), q_bc)

    # Pore pressures: u = gamma_w * (h - z)
    pp = gamma_w * (head - nodes[:, 1])
    pp = np.maximum(pp, 0.0)

    # Velocity
    vel = seepage_velocity(nodes, elements, head, k)

    # Total flow rate: sum of |v| * A for all elements
    flow_rate = 0.0
    for e in range(len(elements)):
        coords = nodes[elements[e]]
        A = cst_area(coords)
        v_mag = np.linalg.norm(vel[e])
        flow_rate += v_mag * A * t

    return {
        'head': head,
        'pore_pressures': pp,
        'velocity': vel,
        'flow_rate': flow_rate,
    }


# ===========================================================================
# Phase 3: Coupled Biot Consolidation
# ===========================================================================

def cst_coupling_matrix(coords, t=1.0):
    """CST solid-fluid coupling matrix Q_e.

    Q_e = t * A * B^T * m * N_avg^T
    where m = [1, 1, 0]^T, B is strain-displacement (3x6),
    and N_avg = [1/3, 1/3, 1/3] for CST.

    For CST: Q_e = (t * A / 3) * B^T * [1;1;0] * [1,1,1]

    Parameters
    ----------
    coords : (3, 2) array — element node coordinates.
    t : float — thickness.

    Returns
    -------
    Q_e : (6, 3) coupling matrix (displacement DOFs x pressure DOFs).
    """
    B, A = cst_B(coords)
    # B^T * m gives (6,) vector; outer product with [1/3, 1/3, 1/3]
    BT_m = B.T @ _M_VOIGT  # (6,)
    N_avg = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    Q_e = t * A * np.outer(BT_m, N_avg)
    return Q_e


def cst_compressibility_matrix(coords, n_w=2.2e6, t=1.0):
    """CST fluid compressibility matrix S_e.

    S_e = (t * A / (9 * n_w)) * [[2,1,1],[1,2,1],[1,1,2]]
    (consistent mass-type matrix for pressure DOFs)

    Parameters
    ----------
    coords : (3, 2) array — element node coordinates.
    n_w : float — bulk modulus of water (kPa). Large value for
        incompressible fluid (effectively S -> 0, pure Terzaghi).
    t : float — thickness.

    Returns
    -------
    S_e : (3, 3) compressibility matrix.
    """
    A = cst_area(coords)
    # Consistent mass matrix for CST: (A/12) * [[2,1,1],[1,2,1],[1,1,2]]
    # Divided by n_w for compressibility
    factor = t * A / (12.0 * n_w)
    S_e = factor * np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    return S_e


def assemble_coupling(nodes, elements, t=1.0):
    """Assemble global coupling matrix Q.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    t : float — thickness.

    Returns
    -------
    Q : sparse CSR (2*n_nodes x n_nodes) — coupling matrix.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_nodes = len(nodes)

    rows, cols, vals = [], [], []

    for e in range(len(elements)):
        conn = elements[e]
        coords = nodes[conn]
        Q_e = cst_coupling_matrix(coords, t)
        dofs_u = element_dofs(conn)  # (6,) displacement DOFs

        for i in range(6):
            for j in range(3):
                rows.append(dofs_u[i])
                cols.append(conn[j])
                vals.append(Q_e[i, j])

    Q = coo_matrix((vals, (rows, cols)),
                    shape=(2 * n_nodes, n_nodes)).tocsr()
    return Q


def assemble_compressibility(nodes, elements, n_w=2.2e6, t=1.0):
    """Assemble global compressibility matrix S.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    n_w : float — bulk modulus of water (kPa).
    t : float — thickness.

    Returns
    -------
    S : sparse CSR (n_nodes x n_nodes) — compressibility matrix.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_nodes = len(nodes)

    rows, cols, vals = [], [], []

    for e in range(len(elements)):
        conn = elements[e]
        coords = nodes[conn]
        S_e = cst_compressibility_matrix(coords, n_w, t)

        for i in range(3):
            for j in range(3):
                rows.append(conn[i])
                cols.append(conn[j])
                vals.append(S_e[i, j])

    S = coo_matrix((vals, (rows, cols)),
                    shape=(n_nodes, n_nodes)).tocsr()
    return S


def solve_consolidation(nodes, elements, material_props, gamma, bc_nodes,
                        k, head_bcs, time_steps, t=1.0,
                        gamma_w=9.81, n_w=2.2e6,
                        pore_pressures_0=None, surface_loads=None):
    """Solve Biot consolidation using staggered scheme.

    Staggered (sequential) at each time step dt:
    1. Displacement: K * u_{n+1} = F_ext - Q * p_n
    2. Pressure: (S/dt + H) * p_{n+1} = S * p_n / dt - Q^T * (u_{n+1} - u_n) / dt
    Apply head BCs to pressure system.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', ...}. Only elastic materials supported.
    gamma : float or (n_elements,) array — unit weight (kN/m^3).
    bc_nodes : dict — from detect_boundary_nodes().
    k : float or (n_elements,) array — hydraulic conductivity (m/s).
    head_bcs : list of (node_id, head_value) — drainage BCs (fixed head).
    time_steps : array-like — time points (s), e.g. [0, 100, 1000].
    t : float — thickness.
    gamma_w : float — unit weight of water (kN/m^3).
    n_w : float — bulk modulus of water (kPa).
    pore_pressures_0 : (n_nodes,) array, optional — initial pore pressures.
    surface_loads : list of (edge_nodes, qx, qy), optional — surface tractions.

    Returns
    -------
    dict with keys:
        times : (n_steps,) array
        displacements : (n_steps, 2*n_nodes) array
        pore_pressures : (n_steps, n_nodes) array
        settlements : (n_steps,) array — max surface settlement at each step
        max_settlement_m : float
        max_excess_pore_pressure_kPa : float
        degree_of_consolidation : float — U at final time
        converged : bool
    """
    from fem2d.assembly import (
        assemble_stiffness, assemble_gravity, assemble_surface_load,
        apply_bcs_penalty,
    )
    from fem2d.materials import elastic_D

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    time_steps = np.asarray(time_steps, dtype=float)
    n_nodes = len(nodes)
    n_elem = len(elements)
    n_steps = len(time_steps)
    n_dof_u = 2 * n_nodes

    # Expand material props
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Build global stiffness from elastic D matrices
    D_list = []
    for mp in material_props:
        D_list.append(elastic_D(mp['E'], mp['nu']))
    D_array = np.array(D_list)
    K = assemble_stiffness(nodes, elements, D_array, t)

    # External force (gravity + surface loads)
    F_ext = assemble_gravity(nodes, elements, gamma, t)
    if surface_loads:
        for edges, qx, qy in surface_loads:
            F_ext += assemble_surface_load(nodes, edges, qx, qy, t)

    # Assembly flow/coupling matrices
    Q = assemble_coupling(nodes, elements, t)
    H_flow, _ = assemble_flow_system(nodes, elements, k, t)
    S = assemble_compressibility(nodes, elements, n_w, t)

    # Displacement BCs
    penalty = 1e20
    bc_dofs = set()
    for n in bc_nodes.get('fixed_base', []):
        bc_dofs.add(2 * n)
        bc_dofs.add(2 * n + 1)
    for key in ['roller_left', 'roller_right']:
        for n in bc_nodes.get(key, []):
            bc_dofs.add(2 * n)

    # Apply displacement BCs to K
    K_bc, F_ext_bc = apply_bcs_penalty(K, F_ext, bc_nodes)

    # Initial pore pressures
    if pore_pressures_0 is not None:
        p = np.asarray(pore_pressures_0, dtype=float).copy()
    else:
        p = np.zeros(n_nodes)

    # Initial displacement (equilibrium under initial pore pressure + gravity)
    F_init = F_ext_bc + pore_pressure_force(nodes, elements, p, t)
    # Re-apply BCs to the combined force
    _, F_init_bc = apply_bcs_penalty(K, F_init, bc_nodes)
    u = spsolve(K_bc.tocsc(), F_init_bc)

    # Storage for time history
    u_history = np.zeros((n_steps, n_dof_u))
    p_history = np.zeros((n_steps, n_nodes))
    settlements = np.zeros(n_steps)

    # Store initial state
    u_history[0] = u
    p_history[0] = p

    # Surface nodes for settlement tracking
    y_max = nodes[:, 1].max()
    surface_mask = np.abs(nodes[:, 1] - y_max) < 0.01 * (y_max - nodes[:, 1].min() + 1)

    if surface_mask.any():
        settlements[0] = np.min(u[1::2][surface_mask])  # most negative = most settlement
    converged = True

    # Time stepping
    for step in range(1, n_steps):
        dt = time_steps[step] - time_steps[step - 1]
        if dt <= 0:
            u_history[step] = u
            p_history[step] = p
            settlements[step] = settlements[step - 1]
            continue

        u_prev = u.copy()
        p_prev = p.copy()

        # Step 1: Displacement with current pore pressure
        F_pp = pore_pressure_force(nodes, elements, p, t)
        F_total = F_ext + F_pp
        K_bc_step, F_bc_step = apply_bcs_penalty(K, F_total, bc_nodes)
        try:
            u = spsolve(K_bc_step.tocsc(), F_bc_step)
        except Exception:
            converged = False
            u_history[step:] = u_prev
            p_history[step:] = p_prev
            settlements[step:] = settlements[step - 1]
            break

        # Step 2: Pressure update
        # (S/dt + H) * p_{n+1} = S * p_n / dt - Q^T * (u_{n+1} - u_n) / dt
        du = u - u_prev
        rhs_p = (S @ p_prev) / dt - (Q.T @ du) / dt
        A_p = S / dt + H_flow

        # Apply head BCs to pressure system
        A_p_bc, rhs_p_bc = apply_head_bcs(A_p, rhs_p, head_bcs)
        try:
            p = spsolve(A_p_bc.tocsc(), rhs_p_bc)
        except Exception:
            converged = False
            p = p_prev
            u_history[step:] = u
            p_history[step:] = p
            settlements[step:] = settlements[step - 1]
            break

        # Clip negative pore pressures (suction not modeled)
        p = np.maximum(p, 0.0)

        u_history[step] = u
        p_history[step] = p
        if surface_mask.any():
            settlements[step] = np.min(u[1::2][surface_mask])

    # Compute summary statistics
    max_settlement = float(np.min(settlements))  # most negative
    max_pp = float(np.max(p_history))

    # Degree of consolidation: U = current_settlement / final_settlement
    # (settlement is negative, so use ratios of absolute values)
    s_final = abs(settlements[-1]) if abs(settlements[-1]) > 0 else 1.0
    s_elastic = abs(settlements[0]) if abs(settlements[0]) > 0 else 0.0
    if s_final > 1e-12:
        degree_of_consolidation = abs(settlements[-1]) / s_final
    else:
        degree_of_consolidation = 1.0

    return {
        'times': time_steps,
        'displacements': u_history,
        'pore_pressures': p_history,
        'settlements': settlements,
        'max_settlement_m': max_settlement,
        'max_excess_pore_pressure_kPa': max_pp,
        'degree_of_consolidation': degree_of_consolidation,
        'converged': converged,
    }
