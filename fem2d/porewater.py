"""
Pore water pressures, steady-state seepage, and coupled Biot consolidation.

Provides:
- Static pore pressure computation from GWT definition
- Effective stress correction for total → effective stress
- Pore pressure equivalent nodal forces
- Steady-state seepage solver (Laplace equation, CST flow elements)
- Coupled Biot consolidation (staggered default; monolithic u-p option for the
  load-induced undrained transient, ``solve_consolidation(scheme="monolithic")``)

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
from scipy.sparse import coo_matrix, lil_matrix, bmat
from scipy.sparse.linalg import spsolve, splu

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
        Source basis: physical constant (rho_w*g = 1000*9.81/1000 = 9.81
        kN/m^3), not a correlation; see geotech_common.water.GAMMA_W.

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
        Source basis: PHYSICAL CONSTANT — the bulk modulus of water is
        ~2.2e6 kPa (2.2 GPa). Not a correlation or chart read.
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
                        pore_pressures_0=None, surface_loads=None,
                        scheme="staggered", theta=1.0,
                        initial_surface_loads=None, fill_elements=None):
    """Solve Biot consolidation using the staggered or monolithic u-p scheme.

    Two construction stages (2026-10-09, per-stage movements). Stage 0, the
    INITIAL state: the self-weight of the soil in place before loading, in
    drained equilibrium with the hydrostatic ``pore_pressures_0`` field, plus
    any ``initial_surface_loads`` (e.g. ponded water above the ground
    surface). Stage 1, the LOAD stage: ``surface_loads`` plus the self-weight
    of any ``fill_elements`` (soil placed as fill during construction, with
    the load), followed by consolidation. The stage-0 field and the stage-1
    increment are returned separately, so the caller reports movements per
    stage and chooses which stage they are measured from. A soil with zero
    unit weight everywhere is a weightless analysis: the initial stage then
    carries no load at all (no self-weight, no water force).

    Formulation (2026-10-09). ``pore_pressures_0`` is the initial EQUILIBRIUM
    pore field (hydrostatic; zeros if omitted) and self-weight is in
    equilibrium with it before loading. The diffusion unknown is the EXCESS
    pore pressure over that field, so a hydrostatic field carries no flow and
    never "dissipates". ``head_bcs`` give the drained boundaries as TOTAL HEAD
    h (m); they become pore pressures u = gamma_w * (h - z), clipped >= 0, and
    then excess pressures u - p_init. Times are measured from the instant of
    loading: t = 0 is prepended when the schedule starts later.

    ``scheme="staggered"`` (**default**): sequential split at each dt:
      1. Displacement: K * u_{n+1} = F_ext - Q * (p_init + pex_n)
      2. Excess pressure: (S/dt + H/gamma_w) * pex_{n+1}
                          = S * pex_n / dt - Q^T*(u_{n+1}-u_n)/dt
    with k the hydraulic conductivity (m/s). It transports an excess field set
    up by the boundaries but does NOT convert an applied total-stress increment
    into excess pore pressure (no undrained transient: drained at every step,
    U = 1). Use the monolithic scheme for consolidation under a load.

    ``scheme="monolithic"``: solve displacement AND excess pore pressure
    SIMULTANEOUSLY from the coupled Biot block system (theta-method in time),
    driven by the surface LOAD increment only (self-weight is the initial
    state; linear superposition):

        | K      -Q            | | u_{n+1} |   | F_load                           |
        | Q^T   (S + theta dt H)| | p_{n+1} | = | Q^T u_n + S p_n - (1-theta)dt H p_n |

    The load is applied UNDRAINED at t=0 (the first block solve with no flow gives
    the instantaneous excess pore pressure p0), then it dissipates through H — the
    full Terzaghi/Biot consolidation transient. Here ``k`` is the MOBILITY
    (m^2/(kPa.s)) = hydraulic conductivity / gamma_w. ``theta`` in [0.5, 1];
    1.0 = backward Euler (unconditionally stable, recommended for the early
    undrained boundary layer).

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', ...}. Only elastic materials supported.
    gamma : float or (n_elements,) array — unit weight (kN/m^3). Enters the
        staggered displacement step; the monolithic increment excludes it.
    bc_nodes : dict — from detect_boundary_nodes().
    k : float or (n_elements,) array — hydraulic conductivity (m/s) for
        "staggered"; mobility (m^2/(kPa.s)) for "monolithic".
    head_bcs : list of (node_id, total_head_m) — drained boundaries.
    time_steps : array-like — times since loading (s), e.g. [0, 100, 1000].
    t : float — thickness.
    gamma_w : float — unit weight of water (kN/m^3).
    n_w : float — bulk modulus of water (kPa).
    pore_pressures_0 : (n_nodes,) array, optional — initial equilibrium pore
        pressures (kPa).
    surface_loads : list of (edge_nodes, qx, qy), optional — surface tractions
        applied in the load stage.
    initial_surface_loads : list of (edge_nodes, qx, qy), optional — tractions
        that belong to the INITIAL state (ponded water above the surface).
        The staggered scheme also carries them in every step (it is
        cumulative); the monolithic increment never sees them.
    fill_elements : iterable of int, optional — elements placed as fill
        during construction, ABOVE the water table (ValueError otherwise):
        their self-weight leaves the initial state and joins the load stage
        (monolithic: applied UNDRAINED with the load, so it raises excess
        pore pressure and consolidates).

    Returns
    -------
    dict with keys:
        times : (n_steps,) array — from 0 (the loading instant)
        displacements : (n_steps, 2*n_nodes) array — CUMULATIVE (initial
            stage + load stage) for "staggered", the LOAD-STAGE increment for
            "monolithic" (as before; see the per-stage keys below)
        initial_stage_displacements : (2*n_u,) — the initial stage
        load_stage_displacements : (n_steps, 2*n_u) — the load stage alone
        displacement_nodes : (n_u, 2) — the nodes these arrays refer to (the
            T6 nodes for "monolithic", the input nodes for "staggered")
        surface_node_mask : (n_u,) bool — the ground-surface nodes
        pore_pressures : (n_steps, n_nodes) array — TOTAL for "staggered",
            EXCESS for "monolithic" (as before; see the next two keys)
        excess_pore_pressures, total_pore_pressures : (n_steps, n_nodes)
        settlements : (n_steps,) array — max surface settlement at each step
            (monolithic: from the load increment only; staggered: cumulative)
        max_settlement_m : float
        max_excess_pore_pressure_kPa : float — largest |excess| at any time
        degree_of_consolidation : float — U at the final time
        degree_of_consolidation_history : (n_steps,) array — U at every time.
            Monolithic: area-weighted excess-pore-pressure dissipation
            1 - avg p(t) / avg p0; staggered: 1 (no undrained transient).
        final_drained_settlement_m : float (monolithic) — the end state
        converged : bool
        scheme : str
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

    if scheme not in ("staggered", "monolithic"):
        raise ValueError(
            f"scheme must be 'staggered' or 'monolithic', got '{scheme}'")

    # Time origin: t = 0 is the instant the load is applied. A schedule that
    # starts later gets t = 0 prepended, so the first requested time is a real
    # elapsed time (before 2026-10-09 its first entry was silently taken as
    # the loading instant: one requested time always gave U = 0).
    if time_steps.size == 0:
        raise ValueError("time_steps must contain at least one time (s)")
    if time_steps[0] < 0:
        raise ValueError(
            f"time_steps are times since loading (s) and must be >= 0; "
            f"got {time_steps[0]}")
    if time_steps[0] > 0:
        time_steps = np.concatenate([[0.0], time_steps])
    n_steps = len(time_steps)
    n_dof_u = 2 * n_nodes

    # Initial pore pressures: the equilibrium (hydrostatic) field the load
    # increment starts from. Excess pore pressure is measured from it.
    if pore_pressures_0 is not None:
        p_init = np.asarray(pore_pressures_0, dtype=float).copy()
    else:
        p_init = np.zeros(n_nodes)

    # Drained boundaries are given as TOTAL HEAD h (m). Convert to pore
    # pressure, u = gamma_w * (h - z) clipped >= 0 (the convention of
    # compute_pore_pressures / solve_seepage), then to EXCESS over p_init.
    # Before 2026-10-09 the head in m was written straight in as a pressure
    # in kPa (gwt = 20 m gave a 20 kPa boundary pressure).
    excess_bcs = [
        (int(node), gamma_w * max(float(head) - nodes[int(node), 1], 0.0)
         - p_init[int(node)])
        for node, head in head_bcs
    ]

    # Construction stages: the soil in place before loading (initial state)
    # versus fill placed with the load. A zero unit weight everywhere is a
    # weightless analysis: the initial state then carries no load at all
    # (weightless soil under a water table would otherwise be "buoyed up").
    weightless = not np.any(np.asarray(gamma, dtype=float) != 0.0)
    fill_set = (set(int(e) for e in fill_elements)
                if fill_elements is not None else set())
    if fill_set:
        # Fill placed above the water table only: underwater (hydraulic)
        # fill would need the water standing on the soil in place before
        # placement as an initial-state traction — not modelled.
        fill_nodes = np.unique(elements[sorted(fill_set)].ravel())
        if np.any(p_init[fill_nodes] > 1e-9):
            raise ValueError(
                "fill_elements must lie above the water table (zero initial "
                "pore pressure in and under them): fill placed under water "
                "is not modelled. Lower gwt to the top of the soil in place "
                "or below it, or leave the layer out of the fill.")
    in_place = [e for e in range(n_elem) if e not in fill_set]
    init_loads = [] if weightless else list(initial_surface_loads or [])

    if scheme == "monolithic":
        # Monolithic u-p uses a Taylor-Hood (T6 displacement / T3 pressure)
        # pairing to satisfy the LBB (inf-sup) condition; self-assembles on a
        # T6 mesh derived from the CST input mesh.
        res = _monolithic_taylor_hood(
            nodes, elements, material_props, gamma, bc_nodes, k, excess_bcs,
            time_steps, t, n_w, float(theta), surface_loads,
            p_init=p_init, initial_surface_loads=init_loads,
            fill_elements=fill_set, weightless=weightless)
        ex = np.asarray(res['pore_pressures'])
        res['excess_pore_pressures'] = ex
        res['total_pore_pressures'] = ex + p_init[None, :ex.shape[1]]
        return res

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

    # External force (gravity + surface loads + initial-state tractions such
    # as ponded water; the staggered displacement is cumulative)
    F_ext = assemble_gravity(nodes, elements, gamma, t)
    if surface_loads:
        for edges, qx, qy in surface_loads:
            F_ext += assemble_surface_load(nodes, edges, qx, qy, t)
    for edges, qx, qy in init_loads:
        F_ext += assemble_surface_load(nodes, edges, qx, qy, t)

    def _pp_force(p_total):
        # Water force in the displacement step. Weightless analysis: the
        # hydrostatic field is only the reference for the excess, so only
        # the excess acts (see the stage note in the docstring).
        p_act = p_total - p_init if weightless else p_total
        return pore_pressure_force(nodes, elements, p_act, t)

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

    # Darcy flow in pressure form: q = -(k/gamma_w) grad(p) for the EXCESS
    # pressure (the hydrostatic initial field carries no flow). k here is the
    # hydraulic conductivity (m/s), so the pressure-equation flow matrix is
    # H/gamma_w (before 2026-10-09 k was used as if it were the mobility,
    # gamma_w times too fast, and the flow acted on the TOTAL pressure with
    # no elevation term, so a hydrostatic field "consolidated" away).
    H_flow = H_flow / gamma_w

    # Apply displacement BCs to K
    K_bc, F_ext_bc = apply_bcs_penalty(K, F_ext, bc_nodes)

    # Pore pressures: p = p_init (equilibrium) + p_ex (excess, the unknown of
    # the diffusion equation).
    p_ex = np.zeros(n_nodes)
    p = p_init + p_ex

    # Initial displacement (equilibrium under initial pore pressure + gravity)
    F_init = F_ext_bc + _pp_force(p)
    # Re-apply BCs to the combined force
    _, F_init_bc = apply_bcs_penalty(K, F_init, bc_nodes)
    u = spsolve(K_bc.tocsc(), F_init_bc)

    # The initial stage alone: self-weight of the soil in place before
    # loading (not the fill) in drained equilibrium with the hydrostatic
    # field, plus the initial-state tractions. Linear and drained, so the
    # load stage is the cumulative field minus this one.
    if weightless:
        u_stage0 = np.zeros(n_dof_u)
    else:
        F0 = assemble_gravity(nodes, elements, gamma, t,
                              active_elements=in_place)
        F0 = F0 + pore_pressure_force(nodes, elements, p_init, t,
                                      active_elements=in_place)
        for edges, qx, qy in init_loads:
            F0 += assemble_surface_load(nodes, edges, qx, qy, t)
        _, F0_bc = apply_bcs_penalty(K, F0, bc_nodes)
        u_stage0 = spsolve(K_bc.tocsc(), F0_bc)

    # Storage for time history
    u_history = np.zeros((n_steps, n_dof_u))
    p_history = np.zeros((n_steps, n_nodes))
    pex_history = np.zeros((n_steps, n_nodes))
    settlements = np.zeros(n_steps)

    # Store initial state
    u_history[0] = u
    p_history[0] = p
    pex_history[0] = p_ex

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
            pex_history[step] = p_ex
            settlements[step] = settlements[step - 1]
            continue

        u_prev = u.copy()
        p_ex_prev = p_ex.copy()

        # Step 1: Displacement with current (total) pore pressure
        F_pp = _pp_force(p)
        F_total = F_ext + F_pp
        K_bc_step, F_bc_step = apply_bcs_penalty(K, F_total, bc_nodes)
        try:
            u = spsolve(K_bc_step.tocsc(), F_bc_step)
        except Exception:
            converged = False
            u_history[step:] = u_prev
            p_history[step:] = p
            pex_history[step:] = p_ex
            settlements[step:] = settlements[step - 1]
            break

        # Step 2: Excess pressure update
        # (S/dt + H/gw) * pex_{n+1} = S * pex_n / dt - Q^T (u_{n+1}-u_n)/dt
        du = u - u_prev
        rhs_p = (S @ p_ex_prev) / dt - (Q.T @ du) / dt
        A_p = S / dt + H_flow

        # Drained boundaries: prescribed excess pressure
        A_p_bc, rhs_p_bc = apply_head_bcs(A_p, rhs_p, excess_bcs)
        try:
            p_ex = spsolve(A_p_bc.tocsc(), rhs_p_bc)
        except Exception:
            converged = False
            p_ex = p_ex_prev
            u_history[step:] = u
            p_history[step:] = p
            pex_history[step:] = p_ex
            settlements[step:] = settlements[step - 1]
            break

        # Clip negative TOTAL pore pressures (suction not modeled)
        p = np.maximum(p_init + p_ex, 0.0)
        p_ex = p - p_init

        u_history[step] = u
        p_history[step] = p
        pex_history[step] = p_ex
        if surface_mask.any():
            settlements[step] = np.min(u[1::2][surface_mask])

    # Compute summary statistics
    max_settlement = float(np.min(settlements))  # most negative
    max_pp = float(np.max(np.abs(pex_history)))

    # Degree of consolidation: the staggered split never converts the load
    # into excess pore pressure (drained at every step), so U is 1 at every
    # time; reported as such, with a note in the analysis wrapper.
    doc_history = np.ones(n_steps)
    degree_of_consolidation = 1.0

    return {
        'times': time_steps,
        'displacements': u_history,
        'initial_stage_displacements': u_stage0,
        'load_stage_displacements': u_history - u_stage0[None, :],
        'displacement_nodes': nodes,
        'surface_node_mask': surface_mask,
        'pore_pressures': p_history,
        'excess_pore_pressures': pex_history,
        'total_pore_pressures': p_history,
        'settlements': settlements,
        'max_settlement_m': max_settlement,
        'max_excess_pore_pressure_kPa': max_pp,
        'degree_of_consolidation': degree_of_consolidation,
        'degree_of_consolidation_history': doc_history,
        'converged': converged,
        'scheme': 'staggered',
    }


def _same_dt(dt_a, dt_b):
    """True if two time-step sizes are equal to working precision, so the coupled
    A-block LU factorization can be reused. A patchable seam: forcing this to
    False makes the monolithic solver refactor every step (the naive path), which
    a test uses to confirm the cached path is numerically identical."""
    return bool(np.isclose(dt_a, dt_b, rtol=1e-12, atol=0.0))


def _monolithic_consolidation(K_bc, F_ext_bc, Q, H, S, excess_bcs,
                              time_steps, theta, surface_mask):
    """Monolithic (coupled) u-p Biot consolidation, theta time-stepping.

    Solves the block system for (u, p) simultaneously so an applied total-stress
    increment is instantly split into effective stress + excess pore pressure
    (the undrained response), which then dissipates through the flow matrix H.
    K_bc / F_ext_bc already carry the displacement BCs (penalty); Q is
    (n_dof_u x n_dof_p), H and S are (n_dof_p x n_dof_p). Dimensions are inferred
    from the matrices, so the same solver serves the CST and the Taylor-Hood
    (T6/T3) pairings. ``p`` is the EXCESS pore pressure and ``F_ext_bc`` the
    load INCREMENT (self-weight and the hydrostatic field are the initial
    state); ``excess_bcs`` are (node, prescribed excess pressure kPa) on the
    drained boundary, applied to the pressure block by penalty. See
    ``solve_consolidation``.
    """
    time_steps = np.asarray(time_steps, dtype=float)
    n_steps = len(time_steps)
    n_dof_u = K_bc.shape[0]
    n_dof_p = S.shape[0]
    penalty = 1.0e20
    if not (0.5 <= theta <= 1.0):
        raise ValueError(f"theta must be in [0.5, 1.0], got {theta}")

    # Pressure (drainage) BCs by penalty: prescribed EXCESS pore pressure
    # (kPa), already converted from head by solve_consolidation.
    p_pen = np.zeros(n_dof_p)
    p_val = np.zeros(n_dof_p)
    for node, p_excess in excess_bcs:
        p_pen[int(node)] = penalty
        p_val[int(node)] = float(p_excess)
    idx = np.arange(n_dof_p)
    P_bc = coo_matrix((p_pen, (idx, idx)), shape=(n_dof_p, n_dof_p)).tocsr()
    p_rhs_bc = p_pen * p_val

    Qt = Q.T.tocsr()

    def _factor(A22):
        # LU-factorize the coupled block [[K, -Q], [Qt, A22]]. The block is
        # constant for a fixed dt (K/Q/S/H/P_bc do not change), so the
        # factorization is reused across every step that shares dt.
        A = bmat([[K_bc, -Q], [Qt, A22]]).tocsc()
        return splu(A)

    def _solve_block_lu(lu, rhs_p):
        rhs = np.concatenate([F_ext_bc, rhs_p])
        x = lu.solve(rhs)
        return x[:n_dof_u], x[n_dof_u:]

    # t = 0: undrained response (no flow term) — the instantaneous p0 field.
    u, p = _solve_block_lu(_factor((S + P_bc).tocsr()), p_rhs_bc.copy())

    u_hist = np.zeros((n_steps, n_dof_u))
    p_hist = np.zeros((n_steps, n_dof_p))
    settlements = np.zeros(n_steps)
    u_hist[0] = u
    p_hist[0] = p
    if surface_mask.any():
        settlements[0] = float(np.min(u[1::2][surface_mask]))
    converged = True

    # Factorization cache: the A-block depends only on dt, so a uniform-time-step
    # schedule (the common case) factorizes ONCE and reuses; it is rebuilt only
    # when dt changes. lu.solve uses the same SuperLU backend as spsolve, so the
    # results are numerically identical to a per-step rebuild.
    lu = None
    lu_dt = None
    for step in range(1, n_steps):
        dt = time_steps[step] - time_steps[step - 1]
        if dt <= 0:
            u_hist[step] = u
            p_hist[step] = p
            settlements[step] = settlements[step - 1]
            continue
        rhs_p = Qt @ u + S @ p - (1.0 - theta) * dt * (H @ p) + p_rhs_bc
        try:
            if lu is None or not _same_dt(dt, lu_dt):
                lu = _factor((S + theta * dt * H + P_bc).tocsr())
                lu_dt = dt
            u, p = _solve_block_lu(lu, rhs_p)
        except Exception:
            converged = False
            u_hist[step:] = u
            p_hist[step:] = p
            settlements[step:] = settlements[step - 1]
            break
        u_hist[step] = u
        p_hist[step] = p
        if surface_mask.any():
            settlements[step] = float(np.min(u[1::2][surface_mask]))

    # Degree of consolidation from excess-pore-pressure dissipation, at EVERY
    # time: U(t) = 1 - avg p(t) / avg p0, the averages weighted by each pressure
    # node's tributary area (row sums of the storage matrix S, which is
    # proportional to the lumped mass for a uniform fluid modulus) so U is the
    # area average Terzaghi defines, not a node count. p_hist[0] is the
    # instantaneous undrained t=0 field (U = 0); full dissipation gives U = 1.
    # Guard a ~zero p0 (no applied load) -> U = 1.
    w = np.asarray(S.sum(axis=1)).ravel()
    avg_p0 = float(w @ p_hist[0])
    if abs(avg_p0) > 1e-12:
        doc_hist = np.clip(1.0 - (p_hist @ w) / avg_p0, 0.0, 1.0)
    else:
        doc_hist = np.ones(n_steps)
    return {
        'times': time_steps,
        'displacements': u_hist,
        'pore_pressures': p_hist,
        'settlements': settlements,
        'max_settlement_m': float(np.min(settlements)),
        'max_excess_pore_pressure_kPa': float(np.max(np.abs(p_hist))),
        'initial_excess_pore_pressure_avg_kPa': (
            avg_p0 / float(w.sum()) if w.sum() > 0 else 0.0),
        # U at the final time step, and at every time.
        'degree_of_consolidation': float(doc_hist[-1]),
        'degree_of_consolidation_history': doc_hist,
        'converged': converged,
        'scheme': 'monolithic',
    }


def assemble_coupling_taylor_hood(nodes6, elements6, t=1.0, n_gp=3,
                                  active_elements=None):
    """Taylor-Hood solid-fluid coupling Q (T6 displacement x T3 corner pressure).

    Q_e = integral_T B_u^T m N_p dA, with B_u the T6 strain-displacement matrix
    (3x12) and N_p = [L1, L2, L3] the linear (T3) pressure shape on the corner
    nodes. Global Q is (2*n6 x n_corner); pressure DOFs live on the corner nodes
    only (indices 0..n_corner-1 in the convert_to_t6 node ordering).
    ``active_elements`` (optional set of element indices) assembles over a
    subset with the full global shape (e.g. the hydrostatic water force on
    the soil present in the initial stage only).
    """
    from fem2d.elements import t6_B_detJ, TRI_GAUSS

    nodes6 = np.asarray(nodes6, dtype=float)
    elements6 = np.asarray(elements6, dtype=int)
    n6 = len(nodes6)
    n_corner = int(elements6[:, :3].max()) + 1
    pts, wts = TRI_GAUSS[n_gp]
    if active_elements is not None:
        active_elements = set(int(e) for e in active_elements)

    rows, cols, vals = [], [], []
    for e in range(len(elements6)):
        if active_elements is not None and e not in active_elements:
            continue
        conn = elements6[e]
        coords = nodes6[conn]                 # (6, 2)
        Qe = np.zeros((12, 3))
        for L, w in zip(pts, wts):
            B, detJ, _ = t6_B_detJ(coords, L)
            Np = np.asarray(L, dtype=float)    # corner linear shape = area coords
            BTm = B.T @ _M_VOIGT               # (12,)
            Qe += np.outer(BTm, Np) * (0.5 * w * detJ * t)
        dofs_u = element_dofs(conn)            # (12,)
        corners = conn[:3]
        for i in range(12):
            for j in range(3):
                rows.append(dofs_u[i])
                cols.append(int(corners[j]))
                vals.append(Qe[i, j])
    return coo_matrix((vals, (rows, cols)),
                      shape=(2 * n6, n_corner)).tocsr()


def _monolithic_taylor_hood(nodes, elements, material_props, gamma, bc_nodes,
                            k, excess_bcs, time_steps, t, n_w, theta,
                            surface_loads, p_init=None,
                            initial_surface_loads=None, fill_elements=None,
                            weightless=False):
    """Monolithic u-p consolidation with the Taylor-Hood (T6/T3) pairing.

    Converts the CST input mesh to T6, assembles the quadratic-displacement
    stiffness + the mixed coupling, uses the corner (CST) skeleton for the
    linear-pressure flow/compressibility matrices, and time-steps the coupled
    block system (theta-method) — LBB-stable, so the drained-boundary pressure
    overshoot of the equal-order pairing is avoided.

    The transient is driven by the LOAD STAGE only: the applied surface loads
    plus the effective self-weight of any ``fill_elements`` (soil placed with
    the load). The self-weight of the soil already in place is in equilibrium
    with the initial hydrostatic pore field before loading — the INITIAL
    stage — so, the system being linear, it is left out of the increment by
    superposition and solved on its own (drained) as
    ``initial_stage_displacements``. (Before 2026-10-09 all gravity was
    applied undrained at t = 0 together with the load, so p0 carried the
    self-weight response: 386 kPa under a 100 kPa load on a 20 m column.)
    Which stage the reported movements are measured from is the caller's
    choice (``analyze_consolidation(reset_displacements_after=...)``).
    """
    from fem2d.mesh import convert_to_t6, t6_boundary_edges, detect_boundary_nodes
    from fem2d.assembly import (
        assemble_stiffness, assemble_surface_load, apply_bcs_penalty,
        assemble_gravity,
    )
    from fem2d.materials import elastic_D

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_elem = len(elements)
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Quadratic-displacement (T6) mesh; corners keep their indices.
    nodes6, elem6 = convert_to_t6(nodes, elements)

    D_array = np.array([elastic_D(mp['E'], mp['nu']) for mp in material_props])
    K = assemble_stiffness(nodes6, elem6, D_array, t)

    # Load-stage increment (see docstring): the surface loads, plus the
    # effective self-weight of any fill placed with the load. The self-weight
    # of the soil already in place is the initial stage, solved below.
    F_ext = np.zeros(2 * len(nodes6))
    if surface_loads:
        for edges, qx, qy in surface_loads:
            edges3 = t6_boundary_edges(elem6, edges)
            F_ext += assemble_surface_load(nodes6, edges3, qx, qy, t)

    n_corner = int(elem6[:, :3].max()) + 1
    p_corner = (np.zeros(n_corner) if p_init is None
                else np.asarray(p_init, dtype=float)[:n_corner])
    fill = set(fill_elements or ())

    # Displacement BCs on the T6 mesh (re-detect so midside boundary nodes are
    # constrained too); geometry matches the CST detection.
    bc6 = detect_boundary_nodes(nodes6)

    # Linear-pressure (T3) flow + compressibility on the corner skeleton
    # (identical to the original CST element matrices).
    H_p, _ = assemble_flow_system(nodes, elements, k, t)
    S_p = assemble_compressibility(nodes, elements, n_w, t)

    Q = assemble_coupling_taylor_hood(nodes6, elem6, t)

    # Water force of the hydrostatic field on the soil (Q p: tension-positive
    # effective stress, so below the water table it buoys the soil up).
    F_water_all = Q @ p_corner
    F_water_fill = np.zeros_like(F_water_all)
    if fill and not weightless:
        F_water_fill = assemble_coupling_taylor_hood(
            nodes6, elem6, t, active_elements=fill) @ p_corner
        F_ext += assemble_gravity(nodes6, elem6, gamma, t,
                                  active_elements=fill) + F_water_fill

    K_bc, F_ext_bc = apply_bcs_penalty(K, F_ext, bc6)

    y_max = nodes6[:, 1].max()
    surface_mask = np.abs(nodes6[:, 1] - y_max) < \
        0.01 * (y_max - nodes6[:, 1].min() + 1)

    res = _monolithic_consolidation(K_bc, F_ext_bc, Q, H_p, S_p, excess_bcs,
                                    time_steps, theta, surface_mask)
    res['scheme'] = 'monolithic_taylor_hood'

    # The INITIAL stage: drained self-weight of the soil in place (not the
    # fill) with the hydrostatic water force, plus the initial-state
    # tractions (ponded water). Zero for a weightless analysis.
    u0 = np.zeros(2 * len(nodes6))
    if not weightless:
        in_place = [e for e in range(len(elem6)) if e not in fill]
        F0 = assemble_gravity(nodes6, elem6, gamma, t,
                              active_elements=in_place)
        F0 = F0 + (F_water_all - F_water_fill)
        for edges, qx, qy in initial_surface_loads or []:
            F0 += assemble_surface_load(
                nodes6, t6_boundary_edges(elem6, edges), qx, qy, t)
        if np.any(F0 != 0.0):
            _, F0_bc = apply_bcs_penalty(K, F0, bc6)
            u0 = spsolve(K_bc.tocsc(), F0_bc)
    res['initial_stage_displacements'] = u0
    res['load_stage_displacements'] = res['displacements']
    res['displacement_nodes'] = nodes6
    res['surface_node_mask'] = surface_mask
    # The fully drained end state of the same load (the settlement the
    # transient tends to), for U by settlement and as a check.
    try:
        u_dr = spsolve(K_bc.tocsc(), F_ext_bc)
        res['final_drained_settlement_m'] = (
            float(np.min(u_dr[1::2][surface_mask])) if surface_mask.any()
            else 0.0)
    except Exception:
        res['final_drained_settlement_m'] = None
    return res
