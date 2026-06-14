"""
FEM solvers: linear elastic and nonlinear (Newton-Raphson / initial-stiffness).

The linear solver assembles K and F once and solves directly.

The nonlinear solver is a vectorized Gauss-point core supporting CST, T6,
and Q4 elements with per-Gauss-point 4-component stress state
[sxx, syy, szz, txy] (tension-positive) and the 3D principal-stress
Mohr-Coulomb return mapping (fem2d.materials.mc_return_principal).

Two solution methods:

- ``method='elastic'`` (default): constant-stiffness (initial stiffness /
  modified NR) iteration in the style of Smith & Griffiths "Programming the
  Finite Element Method" and Griffiths & Lane (1999). The elastic global
  stiffness is factorized once (scipy splu) and reused for every iteration
  of every load step — iterations are back-substitutions only. Convergence
  on residual ratio OR displacement-increment ratio (dual criterion).
  Non-convergence within ``max_iter`` is the SRM failure indicator.
- ``method='tangent'``: Newton-Raphson with the continuum elastoplastic
  tangent, reformed every ``reform_interval`` iterations, with divergence
  detection, step cutback, and optional line search.

Per-element Hardening Soil state is kept (per-Gauss-point arrays inside the
per-element dict); HS uses its hyperbolic in-plane tangent model with szz
tracked elastically (documented limitation).
"""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve, splu

from fem2d.elements import cst_B, cst_area, TRI_GAUSS, t6_shape_derivs, \
    q4_shape_derivs, _GAUSS_PTS_2x2
from fem2d.assembly import element_dofs
from fem2d.materials import (
    elastic_D, mc_return_mapping, mc_return_principal, hs_return_mapping,
)


def solve_elastic(nodes, elements, D, gamma, bc_nodes, t=1.0,
                  surface_loads=None, pore_pressures=None):
    """Solve a linear elastic problem.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3/4/6) array — connectivity.
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
        elements_arr = np.asarray(elements)
        if elements_arr.shape[1] == 3:
            from fem2d.porewater import pore_pressure_force
            F += pore_pressure_force(nodes, elements, pore_pressures, t)
        else:
            gp = _gp_precompute(nodes, elements_arr, t)
            F += _pore_pressure_force_gp(gp, np.asarray(pore_pressures))

    K_bc, F_bc = apply_bcs_penalty(K, F, bc_nodes)
    u = solve_linear(K_bc, F_bc)
    stresses, strains = recover_element_stresses(nodes, elements, D, u)

    return u, stresses, strains


# ===========================================================================
# Gauss-point precompute
# ===========================================================================

def _gp_precompute(nodes, elements, t=1.0, n_gp=None):
    """Precompute B matrices, weighted Jacobians, and DOF maps.

    Returns dict with:
        B : (n_e, n_gp, 3, ndof_e) strain-displacement matrices
        w : (n_e, n_gp) integration weights (incl. thickness)
        N : (n_gp, nen) shape function values at the Gauss points
        dofs : (n_e, ndof_e) int — global DOF indices
        rows, cols : flattened assembly index arrays for K
        nen, n_gp, ndof_e
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_e = len(elements)
    nen = elements.shape[1]
    coords = nodes[elements]  # (n_e, nen, 2)

    if nen == 3:
        ngp = 1
        x = coords[:, :, 0]
        y = coords[:, :, 1]
        A2 = (x[:, 0] * (y[:, 1] - y[:, 2]) + x[:, 1] * (y[:, 2] - y[:, 0])
              + x[:, 2] * (y[:, 0] - y[:, 1]))  # signed 2A
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0],
                      y[:, 0] - y[:, 1]], axis=1)  # (n_e, 3)
        cc = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2],
                       x[:, 1] - x[:, 0]], axis=1)
        B = np.zeros((n_e, 1, 3, 6))
        B[:, 0, 0, 0::2] = b / A2[:, None]
        B[:, 0, 1, 1::2] = cc / A2[:, None]
        B[:, 0, 2, 0::2] = cc / A2[:, None]
        B[:, 0, 2, 1::2] = b / A2[:, None]
        w = (0.5 * np.abs(A2) * t)[:, None]
        N = np.full((1, 3), 1.0 / 3.0)
    elif nen == 6:
        ngp = n_gp if n_gp else 3
        pts, wts = TRI_GAUSS[ngp]
        B = np.zeros((n_e, ngp, 3, 12))
        w = np.zeros((n_e, ngp))
        N = np.zeros((ngp, 6))
        for g in range(ngp):
            Ng, dxi, deta = t6_shape_derivs(pts[g])
            N[g] = Ng
            Dn = np.array([dxi, deta])  # (2, 6)
            J = np.einsum('ki,eij->ekj', Dn, coords)  # (n_e, 2, 2)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            dNx = (J[:, 1, 1, None] * dxi[None] -
                   J[:, 0, 1, None] * deta[None]) / detJ[:, None]
            dNy = (-J[:, 1, 0, None] * dxi[None] +
                   J[:, 0, 0, None] * deta[None]) / detJ[:, None]
            B[:, g, 0, 0::2] = dNx
            B[:, g, 1, 1::2] = dNy
            B[:, g, 2, 0::2] = dNy
            B[:, g, 2, 1::2] = dNx
            w[:, g] = 0.5 * wts[g] * detJ * t
        if np.any(w <= 0):
            raise ValueError("Non-positive Jacobian in T6 mesh")
    elif nen == 4:
        ngp = 4
        B = np.zeros((n_e, 4, 3, 8))
        w = np.zeros((n_e, 4))
        N = np.zeros((4, 4))
        for g, (xi, eta) in enumerate(_GAUSS_PTS_2x2):
            dxi, deta, Ng = q4_shape_derivs(xi, eta)
            N[g] = Ng
            Dn = np.array([dxi, deta])
            J = np.einsum('ki,eij->ekj', Dn, coords)
            detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            dNx = (J[:, 1, 1, None] * dxi[None] -
                   J[:, 0, 1, None] * deta[None]) / detJ[:, None]
            dNy = (-J[:, 1, 0, None] * dxi[None] +
                   J[:, 0, 0, None] * deta[None]) / detJ[:, None]
            B[:, g, 0, 0::2] = dNx
            B[:, g, 1, 1::2] = dNy
            B[:, g, 2, 0::2] = dNy
            B[:, g, 2, 1::2] = dNx
            w[:, g] = detJ * t  # Gauss weight = 1 for 2x2
    else:
        raise ValueError(f"Unsupported element with {nen} nodes")

    dofs = np.empty((n_e, 2 * nen), dtype=int)
    dofs[:, 0::2] = 2 * elements
    dofs[:, 1::2] = 2 * elements + 1
    ndof_e = 2 * nen
    rows = np.repeat(dofs[:, :, None], ndof_e, axis=2).ravel()
    cols = np.repeat(dofs[:, None, :], ndof_e, axis=1).ravel()

    return {'B': B, 'w': w, 'N': N, 'dofs': dofs, 'rows': rows,
            'cols': cols, 'nen': nen, 'n_gp': ngp, 'ndof_e': ndof_e,
            'elements': elements}


def _pore_pressure_force_gp(gp, nodal_pp):
    """Pore pressure equivalent force, any element type.

    f_p = sum_gp w * B^T m * u(gp),  m = [1, 1, 0]^T.
    """
    pp_e = nodal_pp[gp['elements']]              # (n_e, nen)
    u_gp = np.einsum('gi,ei->eg', gp['N'], pp_e)  # (n_e, n_gp)
    Bm = gp['B'][:, :, 0, :] + gp['B'][:, :, 1, :]  # (n_e, n_gp, ndof_e)
    fe = np.einsum('egi,eg,eg->ei', Bm, u_gp, gp['w'])
    F = np.zeros(2 * int(gp['dofs'].max() // 2 + 1))
    np.add.at(F, gp['dofs'], fe)
    return F


# ===========================================================================
# Material arrays
# ===========================================================================

_MODEL_ELASTIC, _MODEL_MC, _MODEL_HS = 0, 1, 2


def _material_arrays(material_props, n_elem):
    """Expand the per-element material dict list into flat arrays."""
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    E = np.zeros(n_elem)
    nu = np.zeros(n_elem)
    c = np.zeros(n_elem)
    phi = np.zeros(n_elem)
    psi = np.zeros(n_elem)
    model = np.zeros(n_elem, dtype=int)

    for e, mp in enumerate(material_props):
        c[e] = mp.get('c', 0.0)
        phi[e] = mp.get('phi', 0.0)
        psi[e] = mp.get('psi', 0.0)
        nu[e] = mp['nu']
        if mp.get('model') == 'hs':
            model[e] = _MODEL_HS
            # Representative elastic stiffness for trial/elastic operator:
            # use unload/reload modulus (standard for HS predictors)
            E[e] = mp.get('E', mp['Eur_ref'])
            nu[e] = mp['nu']
        else:
            E[e] = mp['E']
            if c[e] > 0 or phi[e] > 0:
                model[e] = _MODEL_MC

    return {'E': E, 'nu': nu, 'c': c, 'phi': phi, 'psi': psi,
            'model': model, 'props': list(material_props)}


# ===========================================================================
# Nonlinear context (build once, run many — used by SRM)
# ===========================================================================

def build_nl_context(nodes, elements, material_props, gamma, bc_nodes,
                     t=1.0, beam_elements=None, rotation_dof_map=None,
                     pore_pressures=None, active_elements=None,
                     active_beams=None, surface_loads=None,
                     strut_springs=None, n_gp=None):
    """Precompute everything load- and mesh-dependent for nonlinear runs.

    The returned context can be reused across multiple `run_nl` calls with
    different strength parameters (SRM trials) — the elastic stiffness
    factorization is computed lazily once and shared.
    """
    from fem2d.assembly import assemble_gravity, assemble_surface_load

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes = len(nodes)
    n_elem = len(elements)

    gp = _gp_precompute(nodes, elements, t, n_gp)
    mats = _material_arrays(material_props, n_elem)

    # Active element mask
    act = np.ones(n_elem, dtype=bool)
    if active_elements is not None:
        act[:] = False
        act[sorted(active_elements)] = True
    w_act = gp['w'] * act[:, None]

    # DOF count
    if beam_elements and rotation_dof_map:
        n_dof = 2 * n_nodes + len(rotation_dof_map)
    else:
        n_dof = 2 * n_nodes

    # External load: gravity + pore pressure + surface loads (full level)
    F_soil = assemble_gravity(nodes, elements, gamma, t,
                              active_elements=active_elements)
    if pore_pressures is not None:
        pp = np.asarray(pore_pressures, dtype=float)
        F_pp = _pore_pressure_force_gp(
            {**gp, 'w': w_act}, pp)
        F_soil = F_soil + F_pp[:len(F_soil)]
    if surface_loads:
        for edges, qx, qy in surface_loads:
            F_soil += assemble_surface_load(nodes, edges, qx, qy, t)

    F_ext = np.zeros(n_dof)
    F_ext[:2 * n_nodes] = F_soil

    # Constant linear stiffness: beams + struts
    K_lin = None
    if beam_elements and rotation_dof_map:
        from fem2d.assembly import (
            assemble_beam_stiffness, assemble_beam_gravity)
        K_lin = assemble_beam_stiffness(
            nodes, beam_elements, rotation_dof_map, n_dof,
            active_beams=active_beams)
        F_ext += assemble_beam_gravity(
            nodes, beam_elements, rotation_dof_map, n_dof,
            active_beams=active_beams)
    if strut_springs:
        rows, cols, vals = [], [], []
        for node_id, s_k in strut_springs:
            rows.append(2 * node_id)
            cols.append(2 * node_id)
            vals.append(s_k)
        K_struts = coo_matrix((vals, (rows, cols)),
                              shape=(n_dof, n_dof)).tocsr()
        K_lin = K_struts if K_lin is None else K_lin + K_struts

    # BC DOFs (penalty)
    bc_dofs = set()
    for n in bc_nodes.get('fixed_base', []):
        bc_dofs.add(2 * n)
        bc_dofs.add(2 * n + 1)
        if rotation_dof_map and n in rotation_dof_map:
            bc_dofs.add(rotation_dof_map[n])
    for key in ('roller_left', 'roller_right'):
        for n in bc_nodes.get(key, []):
            bc_dofs.add(2 * n)
    # Roller base: v = 0 (horizontal symmetry plane / smooth rigid base)
    for n in bc_nodes.get('roller_base', []):
        bc_dofs.add(2 * n + 1)
    # Floating DOFs (nodes not in any active element) — fix to zero
    if active_elements is not None:
        active_nodes = set(elements[act].ravel().tolist())
        beam_nodes = set()
        if beam_elements:
            ab = active_beams
            for idx, bm in enumerate(beam_elements):
                if ab is not None and idx not in ab:
                    continue
                beam_nodes.add(bm.node_i)
                beam_nodes.add(bm.node_j)
        for ni in range(n_nodes):
            if ni not in active_nodes and ni not in beam_nodes:
                bc_dofs.add(2 * ni)
                bc_dofs.add(2 * ni + 1)
    bc_dofs = np.array(sorted(bc_dofs), dtype=int)

    penalty = 1e20
    pen_diag = coo_matrix(
        (np.full(len(bc_dofs), penalty), (bc_dofs, bc_dofs)),
        shape=(n_dof, n_dof)).tocsr()

    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[bc_dofs] = False

    return {
        'nodes': nodes, 'elements': elements, 'gp': gp, 'mats': mats,
        'act': act, 'w_act': w_act, 'n_dof': n_dof, 'n_nodes': n_nodes,
        'n_elem': n_elem, 'F_ext': F_ext, 'K_lin': K_lin,
        'bc_dofs': bc_dofs, 'free_mask': free_mask, 'penalty': penalty,
        'pen_diag': pen_diag, 't': t,
        'beam_elements': beam_elements,
        'rotation_dof_map': rotation_dof_map,
        '_K_el_factor': None,
    }


def _elastic_Dep(mats_E, mats_nu):
    """Vectorized elastic in-plane D matrices, (N, 3, 3)."""
    cfac = mats_E / ((1.0 + mats_nu) * (1.0 - 2.0 * mats_nu))
    D = np.zeros((len(mats_E), 3, 3))
    D[:, 0, 0] = cfac * (1.0 - mats_nu)
    D[:, 1, 1] = cfac * (1.0 - mats_nu)
    D[:, 0, 1] = cfac * mats_nu
    D[:, 1, 0] = cfac * mats_nu
    D[:, 2, 2] = cfac * (1.0 - 2.0 * mats_nu) / 2.0
    return D


def _assemble_K_soil(ctx, Dep_gp):
    """Assemble soil tangent stiffness from per-GP D matrices."""
    gp = ctx['gp']
    Ke = np.einsum('egki,egkl,eglj,eg->eij',
                   gp['B'], Dep_gp, gp['B'], ctx['w_act'], optimize=True)
    K = coo_matrix((Ke.ravel(), (gp['rows'], gp['cols'])),
                   shape=(ctx['n_dof'], ctx['n_dof'])).tocsr()
    return K


def _build_operator(ctx, Dep_gp):
    """Full system matrix: soil + linear members + BC penalty."""
    K = _assemble_K_soil(ctx, Dep_gp)
    if ctx['K_lin'] is not None:
        K = K + ctx['K_lin']
    K = K + ctx['pen_diag']
    return K.tocsc()


def _elastic_factor(ctx, mats):
    """Lazy shared splu factorization of the elastic operator."""
    if ctx['_K_el_factor'] is None:
        gp = ctx['gp']
        De3 = _elastic_Dep(mats['E'], mats['nu'])
        Dep_gp = np.repeat(De3[:, None], gp['n_gp'], axis=1)
        ctx['_K_el_factor'] = splu(_build_operator(ctx, Dep_gp))
    return ctx['_K_el_factor']


def _constitutive(ctx, mats, u, sig_gp, eps_gp, hs_state, want_tangent):
    """Strain -> trial stress -> return mapping, fully vectorized.

    Returns (sig_new (n_e,ngp,4), eps_new (n_e,ngp,3),
             Dep (n_e,ngp,3,3) or None, hs_tentative list)
    """
    gp = ctx['gp']
    n_e, ngp = gp['w'].shape
    E, nu = mats['E'], mats['nu']
    model = mats['model']

    u_e = u[gp['dofs']]                                     # (n_e, ndof_e)
    eps_new = np.einsum('egki,ei->egk', gp['B'], u_e)        # (n_e, ngp, 3)
    d_eps = eps_new - eps_gp

    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    tr = d_eps[:, :, 0] + d_eps[:, :, 1]
    sig_tr = sig_gp.copy()
    sig_tr[:, :, 0] += lam[:, None] * tr + 2.0 * G[:, None] * d_eps[:, :, 0]
    sig_tr[:, :, 1] += lam[:, None] * tr + 2.0 * G[:, None] * d_eps[:, :, 1]
    sig_tr[:, :, 2] += lam[:, None] * tr
    sig_tr[:, :, 3] += G[:, None] * d_eps[:, :, 2]

    sig_new = sig_tr
    Dep = None
    if want_tangent:
        De3 = _elastic_Dep(E, nu)
        Dep = np.repeat(De3[:, None], ngp, axis=1)

    # --- MC elements: vectorized principal return ---
    mc_mask = (model == _MODEL_MC) & ctx['act']
    if np.any(mc_mask):
        je = np.where(mc_mask)[0]
        sig_flat = sig_tr[je].reshape(-1, 4)
        rep = np.repeat
        s_ret, Dep_ret, _, _ = mc_return_principal(
            sig_flat, rep(E[je], ngp), rep(nu[je], ngp), rep(mats['c'][je], ngp),
            rep(mats['phi'][je], ngp), rep(mats['psi'][je], ngp),
            want_tangent=want_tangent)
        sig_new[je] = s_ret.reshape(len(je), ngp, 4)
        if want_tangent:
            Dep[je] = Dep_ret.reshape(len(je), ngp, 3, 3)

    # --- HS elements: per-element loop (rare), in-plane model ---
    hs_tent = [None] * n_e
    hs_idx = np.where((model == _MODEL_HS) & ctx['act'])[0]
    for e in hs_idx:
        mp = mats['props'][e]
        st = hs_state[e]
        if st is None:
            st = {'gamma_p_s': np.zeros(ngp),
                  'sigma_prev': np.zeros((ngp, 3)),
                  'loading': True}
        else:
            st = _hs_state_to_gp(st, ngp)
        gps_new = np.zeros(ngp)
        sprev_new = np.zeros((ngp, 3))
        loading_any = False
        for g in range(ngp):
            sig3 = sig_tr[e, g, [0, 1, 3]]
            st_g = {'gamma_p_s': float(st['gamma_p_s'][g]),
                    'sigma_prev': st['sigma_prev'][g].copy(),
                    'loading': st['loading']}
            s3n, D3, _, st_n = hs_return_mapping(
                sig3, st_g, E50_ref=mp['E50_ref'], Eur_ref=mp['Eur_ref'],
                m=mp['m'], p_ref=mp['p_ref'], R_f=mp['R_f'],
                nu=mp['nu'], c=mp.get('c', 0), phi_deg=mp.get('phi', 0),
                psi_deg=mp.get('psi', 0))
            sig_new[e, g, 0] = s3n[0]
            sig_new[e, g, 1] = s3n[1]
            sig_new[e, g, 3] = s3n[2]
            # szz stays elastic (already updated in sig_tr)
            if want_tangent:
                Dep[e, g] = D3
            gps_new[g] = st_n['gamma_p_s']
            sprev_new[g] = st_n['sigma_prev']
            loading_any = loading_any or st_n.get('loading', True)
        hs_tent[e] = {'gamma_p_s': gps_new, 'sigma_prev': sprev_new,
                      'loading': loading_any}

    return sig_new, eps_new, Dep, hs_tent


def _hs_state_to_gp(st, ngp):
    """Normalize an HS state dict to per-GP array form."""
    gps = np.asarray(st.get('gamma_p_s', 0.0), dtype=float)
    if gps.ndim == 0:
        gps = np.full(ngp, float(gps))
    sp = np.asarray(st.get('sigma_prev', np.zeros(3)), dtype=float)
    if sp.ndim == 1:
        sp = np.tile(sp, (ngp, 1))
    return {'gamma_p_s': gps, 'sigma_prev': sp,
            'loading': st.get('loading', True)}


def _internal_force(ctx, sig_gp):
    """F_int from per-GP stresses (in-plane components)."""
    gp = ctx['gp']
    sig_ip = sig_gp[:, :, [0, 1, 3]]
    fe = np.einsum('egki,egk,eg->ei', gp['B'], sig_ip, ctx['w_act'],
                   optimize=True)
    F = np.zeros(ctx['n_dof'])
    np.add.at(F, gp['dofs'], fe)
    return F


def run_nl(ctx, n_steps=10, max_iter=100, tol=1e-5,
           u_init=None, sigma_gp_init=None, strain_gp_init=None,
           hs_state_init=None, method='elastic', reform_interval=1,
           disp_tol=None, disp_residual_cap=0.01, max_cutbacks=3,
           line_search=False, mats_override=None,
           stall_window=None, stall_ratio=0.98, f_int_offset=None):
    """Run the incremental-iterative nonlinear solution on a context.

    Parameters
    ----------
    ctx : dict from build_nl_context().
    n_steps : int — load increments (proportional ramp of F_ext).
    max_iter : int — iteration ceiling per load step.
    tol : float — residual convergence: ||R_free|| / ||F_ext|| < tol.
    disp_tol : float or None — optional displacement-increment criterion:
        ||du|| / ||u|| < disp_tol, accepted only if the residual ratio is
        also below `disp_residual_cap`. Default None (residual only):
        near collapse the constant-stiffness iteration can stall with
        small du but a large residual, so a loose dual criterion
        overestimates SRM FOS (verified against Griffiths & Lane Ex. 1).
    method : 'elastic' (constant-stiffness, factorize once) or
        'tangent' (NR with continuum tangent).
    reform_interval : int — tangent reform interval ('tangent' only;
        1 = full NR, k>1 = modified NR).
    max_cutbacks : int — load-step halvings on divergence ('tangent' only).
    line_search : bool — backtracking line search on the residual norm.
    mats_override : dict, optional — replaces ctx['mats'] (SRM trials).
    f_int_offset : (n_dof,) array, optional — internal-force baseline to
        subtract from the residual. Used for an initial-stress release
        (excavation/unloading): with a pre-stressed `sigma_gp_init` whose
        equilibrium internal force is `F0 = integral(B^T sigma_init)`,
        passing `f_int_offset = F0` and `ctx['F_ext'] = -F0` makes the
        residual `R = F_ext - (F_int - F0)`, so at convergence the
        body relaxes to equilibrium with the released boundary traction-
        free (rather than re-applying the in-situ field as an external
        load). None = standard residual `R = F_ext - F_int`.

    Returns
    -------
    dict with keys: converged, u, sigma_gp, eps_gp, hs_state,
        iterations (list per step), n_iter_total.
    """
    mats = mats_override if mats_override is not None else ctx['mats']
    n_dof = ctx['n_dof']
    n_e = ctx['n_elem']
    ngp = ctx['gp']['n_gp']
    bc = ctx['bc_dofs']
    free = ctx['free_mask']
    penalty = ctx['penalty']
    F_full = ctx['F_ext']
    F_norm = np.linalg.norm(F_full[free])

    u = np.array(u_init, dtype=float) if u_init is not None \
        else np.zeros(n_dof)
    sig_gp = np.array(sigma_gp_init, dtype=float) if sigma_gp_init is not None \
        else np.zeros((n_e, ngp, 4))
    eps_gp = np.array(strain_gp_init, dtype=float) if strain_gp_init is not None \
        else np.zeros((n_e, ngp, 3))
    hs_state = list(hs_state_init) if hs_state_init is not None \
        else [None] * n_e

    if F_norm < 1e-30:
        return {'converged': True, 'u': u, 'sigma_gp': sig_gp,
                'eps_gp': eps_gp, 'hs_state': hs_state,
                'iterations': [], 'n_iter_total': 0}

    if method == 'elastic':
        lu = _elastic_factor(ctx, mats)

    iters_per_step = []
    converged_overall = True
    n_iter_total = 0

    # Pseudo-time stepping with optional cutback
    lam_done = 0.0
    d_lam = 1.0 / n_steps
    cutbacks = 0

    while lam_done < 1.0 - 1e-12:
        lam_t = min(1.0, lam_done + d_lam)
        F_ext = lam_t * F_full

        # Snapshot committed state for possible cutback
        u_save = u.copy()

        step_conv = False
        lu_t = None
        n_it = 0
        R_norm_prev = None
        r_hist = []
        for it in range(max_iter):
            n_it = it + 1
            want_tan = (method == 'tangent') and (it % reform_interval == 0)
            sig_new, eps_new, Dep, hs_tent = _constitutive(
                ctx, mats, u, sig_gp, eps_gp, hs_state,
                want_tangent=want_tan)

            F_int = _internal_force(ctx, sig_new)
            if ctx['K_lin'] is not None:
                F_int += ctx['K_lin'] @ u
            if f_int_offset is not None:
                F_int = F_int - f_int_offset

            R = F_ext - F_int
            R[bc] = -penalty * u[bc]
            R_norm = np.linalg.norm(np.where(free, R, 0.0))
            r_ratio = R_norm / F_norm
            r_hist.append(r_ratio)

            if r_ratio < tol:
                step_conv = True
                break
            if not np.isfinite(R_norm) or R_norm > 1e12:
                break
            # Stagnation cut: residual not improving over the window ->
            # treat as non-convergence (saves burning the full ceiling)
            if stall_window and len(r_hist) > stall_window:
                if (min(r_hist[-stall_window:]) >
                        stall_ratio * min(r_hist[:-stall_window])):
                    break

            if method == 'tangent':
                if want_tan:
                    try:
                        lu_t = splu(_build_operator(ctx, Dep))
                    except RuntimeError:
                        break
                du = lu_t.solve(R)
            else:
                du = lu.solve(R)

            if not np.all(np.isfinite(du)) or np.max(np.abs(du)) > 1e6:
                break

            if line_search and R_norm_prev is not None \
                    and R_norm > R_norm_prev:
                # Backtrack on the previous (overshooting) update
                s = 1.0
                for _ in range(3):
                    s *= 0.5
                    u_try = u + (s - 1.0) * du_prev
                    sig_t, _, _, _ = _constitutive(
                        ctx, mats, u_try, sig_gp, eps_gp, hs_state, False)
                    F_t = _internal_force(ctx, sig_t)
                    if ctx['K_lin'] is not None:
                        F_t += ctx['K_lin'] @ u_try
                    if f_int_offset is not None:
                        F_t = F_t - f_int_offset
                    R_t = F_ext - F_t
                    R_t[bc] = -penalty * u_try[bc]
                    R_t_norm = np.linalg.norm(np.where(free, R_t, 0.0))
                    if R_t_norm < R_norm:
                        u = u_try
                        R_norm = R_t_norm
                        break

            u_norm = max(np.linalg.norm(u), 1e-12)
            du_norm = np.linalg.norm(du)
            u = u + du
            du_prev = du
            R_norm_prev = R_norm

            # Displacement-change criterion (Griffiths & Lane)
            if disp_tol is not None and du_norm / u_norm < disp_tol \
                    and r_ratio < disp_residual_cap:
                step_conv = True
                # State must reflect the final u
                sig_new, eps_new, _, hs_tent = _constitutive(
                    ctx, mats, u, sig_gp, eps_gp, hs_state, False)
                break

        n_iter_total += n_it

        if step_conv:
            # Commit state
            actm = ctx['act']
            sig_gp[actm] = sig_new[actm]
            eps_gp[actm] = eps_new[actm]
            for e in np.where(actm)[0]:
                if hs_tent[e] is not None:
                    hs_state[e] = hs_tent[e]
            lam_done = lam_t
            iters_per_step.append(n_it)
            continue

        # Step failed
        if method == 'tangent' and cutbacks < max_cutbacks:
            cutbacks += 1
            d_lam *= 0.5
            u = u_save
            continue

        u = u_save
        converged_overall = False
        iters_per_step.append(n_it)
        break

    return {'converged': converged_overall, 'u': u, 'sigma_gp': sig_gp,
            'eps_gp': eps_gp, 'hs_state': hs_state,
            'iterations': iters_per_step, 'n_iter_total': n_iter_total,
            'r_history': r_hist}


# ===========================================================================
# Backward-compatible wrapper
# ===========================================================================

def _legacy_state_in(sigma_init, strain_init, mats, n_e, ngp):
    """Convert legacy (n_elem, 3) state arrays to per-GP form."""
    sig_gp = None
    eps_gp = None
    if sigma_init is not None:
        sigma_init = np.asarray(sigma_init, dtype=float)
        if sigma_init.ndim == 3:
            sig_gp = sigma_init
        else:
            sig_gp = np.zeros((n_e, ngp, 4))
            sig_gp[:, :, 0] = sigma_init[:, None, 0]
            sig_gp[:, :, 1] = sigma_init[:, None, 1]
            sig_gp[:, :, 3] = sigma_init[:, None, 2]
            # Estimate szz from plane-strain elasticity
            sig_gp[:, :, 2] = (mats['nu'] * (sigma_init[:, 0] +
                                             sigma_init[:, 1]))[:, None]
    if strain_init is not None:
        strain_init = np.asarray(strain_init, dtype=float)
        if strain_init.ndim == 3:
            eps_gp = strain_init
        else:
            eps_gp = np.repeat(strain_init[:, None, :], ngp, axis=1)
    return sig_gp, eps_gp


def solve_nonlinear(nodes, elements, material_props, gamma, bc_nodes,
                    t=1.0, n_steps=10, max_iter=100, tol=1e-5,
                    beam_elements=None, rotation_dof_map=None,
                    pore_pressures=None,
                    active_elements=None, active_beams=None,
                    u_init=None, sigma_init=None, strain_init=None,
                    state_init=None, surface_loads=None,
                    return_state=False, strut_springs=None,
                    method='elastic', reform_interval=1,
                    disp_tol=None, max_cutbacks=3, line_search=False,
                    n_gp=None, return_gp=False, _ctx=None,
                    initial_stress_relaxation=False):
    """Solve a nonlinear (MC/HS) problem.

    Vectorized Gauss-point core with 4-component stress state and 3D
    principal-stress Mohr-Coulomb return mapping. Supports CST (3-node),
    T6 (6-node), and Q4 (4-node) soil elements plus beam elements.

    Parameters (beyond the legacy ones)
    ----------
    method : 'elastic' (default) — constant-stiffness iteration with a
        single splu factorization (Griffiths & Lane style; robust failure
        detection for SRM); 'tangent' — Newton-Raphson with continuum
        elastoplastic tangent, reform interval, and divergence cutback.
    reform_interval : int — K_T reform interval for 'tangent'.
    disp_tol : float or None — optional displacement-increment criterion
        ||du||/||u|| (dual with residual `tol`). Default None.
    max_cutbacks : int — load-step halvings on divergence ('tangent').
    line_search : bool — backtracking line search ('tangent').
    n_gp : int, optional — Gauss rule override for T6 (3 or 6).
    return_gp : bool — append (sigma_gp, eps_gp) per-Gauss-point arrays
        (4-component stress incl. sigma_zz) to the returned tuple.
    initial_stress_relaxation : bool — initial-stress release / unloading.
        Requires `sigma_init` (the in-situ field). The equilibrium internal
        force `F0 = integral(B^T sigma_init)` is computed and the analysis
        is driven by the release load `F_ext = -F0` (ramped over `n_steps`),
        with the residual offset by `F0` so the structure relaxes to a NEW
        equilibrium in which any unsupported (traction-free) boundary —
        e.g. an unlined tunnel/cavity wall, or an excavated face — carries
        zero traction. Any `surface_loads` are ADDED to the release load (so
        a residual support pressure can be retained on part of the
        boundary). The Mohr-Coulomb / HS yield check sees the true total
        stress `sigma_init + Delta sigma` throughout, so the elasto-plastic
        unloading path is correct. Use symmetry BCs (`roller_left/right`
        for u=0, `roller_base` for v=0) plus a fixed or far-field boundary.

    Returns
    -------
    converged : bool
    u : (n_dof,) array
    stresses : (n_elements, 3) array — element-average in-plane stress.
    strains : (n_elements, 3) array — element-average in-plane strain.
    [elem_state] : list — per-element HS state (return_state=True).
    [sigma_gp, eps_gp] : per-GP arrays (return_gp=True).
    """
    elements = np.asarray(elements, dtype=int)
    n_e = len(elements)

    ctx = _ctx
    if ctx is None:
        ctx = build_nl_context(
            nodes, elements, material_props, gamma, bc_nodes, t=t,
            beam_elements=beam_elements, rotation_dof_map=rotation_dof_map,
            pore_pressures=pore_pressures, active_elements=active_elements,
            active_beams=active_beams, surface_loads=surface_loads,
            strut_springs=strut_springs, n_gp=n_gp)
    ngp = ctx['gp']['n_gp']
    mats = ctx['mats']

    sig_gp0, eps_gp0 = _legacy_state_in(sigma_init, strain_init, mats,
                                        n_e, ngp)

    # When initial state is provided (staged construction), apply the full
    # load directly — the initial state already represents equilibrium.
    n_steps_actual = 1 if u_init is not None else n_steps

    f_int_offset = None
    if initial_stress_relaxation:
        if sigma_init is None:
            raise ValueError(
                "initial_stress_relaxation=True requires sigma_init "
                "(the in-situ stress field to be released).")
        # F0 = integral(B^T sigma_init): the equilibrium internal force of
        # the in-situ field. The release load is -F0 (ramped); any prescribed
        # surface_loads already in ctx['F_ext'] are retained (added).
        F0 = _internal_force(ctx, sig_gp0)
        ctx = dict(ctx)
        ctx['F_ext'] = ctx['F_ext'] - F0
        f_int_offset = F0
        # Relaxation is an unloading path — always ramp over n_steps even if
        # u_init is supplied.
        n_steps_actual = n_steps

    res = run_nl(ctx, n_steps=n_steps_actual, max_iter=max_iter, tol=tol,
                 u_init=u_init, sigma_gp_init=sig_gp0,
                 strain_gp_init=eps_gp0, hs_state_init=state_init,
                 method=method, reform_interval=reform_interval,
                 disp_tol=disp_tol, max_cutbacks=max_cutbacks,
                 line_search=line_search, f_int_offset=f_int_offset)

    sig_gp = res['sigma_gp']
    eps_gp = res['eps_gp']
    stresses = sig_gp[:, :, [0, 1, 3]].mean(axis=1)
    strains = eps_gp.mean(axis=1)

    out = (res['converged'], res['u'], stresses, strains)
    if return_state:
        out = out + (list(res['hs_state']),)
    if return_gp:
        out = out + (sig_gp, eps_gp)
    return out


def _do_return_mapping(mp, sigma_trial, elem_state_e):
    """Legacy single-point constitutive dispatch (kept for API compat).

    Works on in-plane 3-component stress [sxx, syy, txy] like the
    original element-level solver. The vectorized solver core uses
    mc_return_principal / _constitutive instead.
    """
    model = mp.get('model', '')

    if model == 'hs':
        state = elem_state_e if elem_state_e is not None else {
            'gamma_p_s': 0.0,
            'sigma_prev': np.zeros(3),
            'loading': True,
        }
        return hs_return_mapping(
            sigma_trial, state,
            E50_ref=mp['E50_ref'], Eur_ref=mp['Eur_ref'],
            m=mp['m'], p_ref=mp['p_ref'], R_f=mp['R_f'],
            nu=mp['nu'], c=mp.get('c', 0), phi_deg=mp.get('phi', 0),
            psi_deg=mp.get('psi', 0))

    if mp.get('c', 0) > 0 or mp.get('phi', 0) > 0:
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, mp['E'], mp['nu'],
            mp.get('c', 0), mp.get('phi', 0),
            mp.get('psi', 0))
        return sigma_new, D_ep, yielded, None

    D_e = elastic_D(mp['E'], mp['nu'])
    return sigma_trial.copy(), D_e, False, None
