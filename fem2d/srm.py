"""
Strength Reduction Method (SRM) for slope stability FOS.

Progressively reduces c and tan(phi) by a Strength Reduction Factor
(SRF) until the FEM model fails. The critical SRF = FOS.

Failure detection (Griffiths & Lane, 1999):
- primary: non-convergence of the constant-stiffness iteration within the
  iteration ceiling (no stress distribution satisfies both the MC criterion
  and global equilibrium);
- secondary: dimensionless-displacement blowup — E_ref*delta_max/(gamma*H^2)
  exceeding `blowup_factor` times its value at the lowest converged SRF
  (the "knee" of the SRF vs displacement curve, GL99 Fig. 2).

Algorithm:
1. Build the nonlinear context once — the elastic global stiffness is
   factorized once and shared by ALL SRF trials (it does not depend on
   c/phi), so each trial costs only back-substitutions.
2. Verify the lower bound converges, step SRF by `bracket_step` (0.1)
   until failure to bracket, then bisect to `tol` (default 0.01).

Strength reduction policy:
- c_red = c/SRF, phi_red = arctan(tan(phi)/SRF)
- psi_red = min(psi, phi_red)  (PLAXIS convention; GL99 used psi=0)
- `srm_field`: 'c_phi' (default, both), 'c' (cohesion only),
  'phi' (friction only)
- HS materials: c and phi reduced, stiffness parameters unchanged.

Reference: Griffiths & Lane (1999), Geotechnique 49(3) 387-403.
"""

import math
import numpy as np

from fem2d.solver import build_nl_context, run_nl, _material_arrays
from fem2d.mesh import detect_boundary_nodes


def _reduce_props(material_props, srf, srm_field):
    """Apply strength reduction to a material property list."""
    reduced = []
    for mp in material_props:
        rp = dict(mp)
        c_orig = mp.get('c', 0.0)
        phi_orig = mp.get('phi', 0.0)
        psi_orig = mp.get('psi', 0.0)

        if srm_field in ('c_phi', 'c'):
            rp['c'] = c_orig / srf
        if srm_field in ('c_phi', 'phi') and phi_orig > 0:
            rp['phi'] = math.degrees(
                math.atan(math.tan(math.radians(phi_orig)) / srf))
        # Dilation capped at the reduced friction angle (PLAXIS policy)
        rp['psi'] = min(psi_orig, rp.get('phi', phi_orig))
        reduced.append(rp)
    return reduced


def _plastic_gp_points(ctx, sig_gp, material_props, srf, srm_field):
    """Gauss points on the reduced-strength MC yield surface (additive).

    At a converged state the return mapping leaves plastic Gauss-point
    stresses ON the yield surface (f = 0) and elastic ones inside
    (f < 0), so flagging f >= -tol identifies the plastic points the
    same way PLAXIS plastic-point plots do. Uses the in-plane Mohr
    circle criterion of materials.mc_return_mapping (tension-positive):
    f = q + p*sin(phi) - c*cos(phi), with c/phi REDUCED at the given
    SRF.

    Returns dict: {'points': [(x, y), ...], 'n_plastic': int,
    'n_gp_total': int, 'srf': float} or None on failure.
    """
    try:
        n_elem = ctx['n_elem']
        reduced = _reduce_props(material_props, srf, srm_field)
        mats = _material_arrays(reduced, n_elem)
        c = np.asarray(mats['c'], dtype=float)[:, None]
        phi = np.radians(np.asarray(mats['phi'], dtype=float))[:, None]
        sxx = sig_gp[:, :, 0]
        syy = sig_gp[:, :, 1]
        txy = sig_gp[:, :, 3]
        pm = 0.5 * (sxx + syy)
        q = np.sqrt((0.5 * (sxx - syy)) ** 2 + txy ** 2)
        f = q + pm * np.sin(phi) - c * np.cos(phi)
        scale = np.maximum(
            np.abs(c * np.cos(phi)) + np.abs(pm * np.sin(phi)), 1.0)
        plastic = f >= -1e-3 * scale
        gp = ctx['gp']
        coords = np.einsum('gi,eij->egj', gp['N'],
                           ctx['nodes'][gp['elements']])
        pts = coords[plastic]
        return {
            'points': [(float(x), float(y)) for x, y in pts],
            'n_plastic': int(plastic.sum()),
            'n_gp_total': int(plastic.size),
            'srf': float(srf),
        }
    except (KeyError, ValueError, IndexError):
        return None


def strength_reduction(nodes, elements, material_props, gamma, bc_nodes=None,
                       t=1.0, srf_range=(0.5, 3.0), tol=0.01,
                       n_load_steps=2, max_nr_iter=1000, nr_tol=1e-5,
                       pore_pressures=None, srm_field='c_phi',
                       bracket_step=0.1, blowup_factor=15.0,
                       n_gp=None, disp_tol=None, stall_window=None,
                       h_ref=None, nr_method='elastic', nr_fallback=False):
    """Find slope stability FOS using the Strength Reduction Method.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 6) array — CST or T6 connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', 'c', 'phi', 'psi', 'gamma'} (+ HS params).
    gamma : float or array — unit weight (kN/m³).
    bc_nodes : dict, optional — from detect_boundary_nodes().
    t : float — thickness.
    srf_range : (float, float) — search range for SRF.
    tol : float — bisection tolerance on FOS (default 0.01).
    n_load_steps : int — gravity load increments per SRF trial.
        Default 2; Griffiths & Lane showed FOS is insensitive to the
        gravity increment size for elastic-perfectly-plastic MC.
    max_nr_iter : int — iteration ceiling per load step (1000, the
        Griffiths & Lane value; iterations are cheap back-substitutions).
    nr_tol : float — residual convergence tolerance.
    pore_pressures : (n_nodes,) array, optional — held constant during
        strength reduction.
    srm_field : 'c_phi' | 'c' | 'phi' — which strengths to reduce.
    bracket_step : float — coarse SRF increment for bracketing.
    blowup_factor : float — dimensionless-displacement blowup threshold
        (multiple of the value at the lowest converged SRF). Set None to
        disable and use pure non-convergence (GL99).
    n_gp : int, optional — Gauss rule override for T6.
    disp_tol : float or None — displacement-increment convergence
        criterion. Default None: convergence is judged on the residual
        alone, which is the correct failure indicator for the
        constant-stiffness scheme (the dual criterion can accept
        stalled near-collapse states and overestimate FOS).
    stall_window : int or None — abort a trial early when the residual
        has not improved by >2% over this many iterations. Default
        None (off): on coarse meshes slow-but-real convergence can
        be misclassified as failure, biasing FOS low. Opt-in speedup.
    h_ref : float, optional — reference height for the dimensionless
        displacement E_ref*delta_max/(gamma_ref*h_ref^2) (Griffiths &
        Lane). Default: total mesh height. Pass the SLOPE height to
        compare against published curves.

    Returns
    -------
    dict with keys:
        'FOS': float — factor of safety.
        'converged': bool — whether bracketing succeeded.
        'fos_basis': 'nonconvergence' | 'blowup' | 'range_exhausted'
        'u_failure': (n_dof,) array — displacements at last stable SRF.
        'stresses_failure': (n_elements, 3) array (element-avg in-plane).
        'n_srf_trials': int — number of SRF evaluations.
        'srf_history': list of dict — per-trial: 'srf', 'max_disp_m',
            'dimensionless_disp', 'converged', 'failed', 'n_iter'.
        'srf_curve': (srf, E_ref*dmax/(gamma*H^2)) arrays for plotting.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    if bc_nodes is None:
        bc_nodes = detect_boundary_nodes(nodes)

    n_elem = len(elements)
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Build context once — elastic K factorization shared across all trials
    ctx = build_nl_context(
        nodes, elements, material_props, gamma, bc_nodes, t=t,
        pore_pressures=pore_pressures, n_gp=n_gp)

    # Dimensionless displacement scaling (Griffiths & Lane):
    # E_ref * delta_max / (gamma_ref * H^2)
    E_ref = float(np.max(ctx['mats']['E']))
    gamma_arr = np.asarray(gamma, dtype=float)
    gamma_ref = float(gamma_arr.mean()) if gamma_arr.ndim else float(gamma_arr)
    H_ref = float(h_ref) if h_ref is not None else \
        float(nodes[:, 1].max() - nodes[:, 1].min())
    dim_scale = E_ref / max(gamma_ref * H_ref ** 2, 1e-12)

    def _max_disp(u):
        n2 = 2 * len(nodes)
        ux = u[:n2:2]
        uy = u[1:n2:2]
        return float(np.max(np.sqrt(ux ** 2 + uy ** 2)))

    n_trials = 0
    srf_history = []
    baseline_dim = [None]  # dimensionless disp at lowest converged SRF
    blowup_seen = [False]

    def _try_srf(srf):
        nonlocal n_trials
        n_trials += 1
        reduced = _reduce_props(material_props, srf, srm_field)
        mats = _material_arrays(reduced, n_elem)
        res = run_nl(ctx, n_steps=n_load_steps, max_iter=max_nr_iter,
                     tol=nr_tol, disp_tol=disp_tol, method=nr_method,
                     mats_override=mats, stall_window=stall_window)
        conv = res['converged']
        extended = False
        if not conv and nr_fallback and nr_method == 'elastic':
            # OPT-IN (nr_fallback=True), experimental: see UPGRADE_PLAN
            # "SRM failure-detection mesh consistency". On low-c/low-phi
            # faces the elastic stall point and the tangent rescue point
            # disagree across meshes (ACADS 1a: 0.80-1.32 spread), so the
            # fallback is NOT part of the validated default behavior.
            extended = True
            res = run_nl(ctx, n_steps=n_load_steps,
                         max_iter=max(200, max_nr_iter // 4),
                         tol=nr_tol, disp_tol=disp_tol, method='tangent',
                         mats_override=mats, stall_window=stall_window)
            conv = res['converged']
        dmax = _max_disp(res['u'])
        dim_disp = dmax * dim_scale
        failed = not conv
        if conv and blowup_factor is not None:
            if baseline_dim[0] is None:
                baseline_dim[0] = max(dim_disp, 1e-12)
            elif dim_disp > blowup_factor * baseline_dim[0]:
                failed = True
                blowup_seen[0] = True
        srf_history.append({
            'srf': float(srf),
            'max_disp_m': dmax,
            'dimensionless_disp': dim_disp,
            'converged': bool(conv),
            'failed': bool(failed),
            'n_iter': res['n_iter_total'],
            'extended': extended,
        })
        return (not failed), res

    def _result(fos, converged, last_res, basis, stable_srf=None):
        sig_gp = last_res['sigma_gp']
        srf_for_state = stable_srf if stable_srf is not None else fos
        return {
            'FOS': round(float(fos), 3),
            'converged': converged,
            'fos_basis': basis,
            'u_failure': last_res['u'],
            'stresses_failure': sig_gp[:, :, [0, 1, 3]].mean(axis=1),
            'n_srf_trials': n_trials,
            'srf_history': srf_history,
            'srf_curve': (
                np.array([h['srf'] for h in srf_history if h['converged']]),
                np.array([h['dimensionless_disp'] for h in srf_history
                          if h['converged']]),
            ),
            'plastic_gp': _plastic_gp_points(
                ctx, sig_gp, material_props, srf_for_state, srm_field),
        }

    srf_low, srf_high = srf_range

    # Verify lower bound is stable
    ok_low, res_low = _try_srf(srf_low)
    if not ok_low:
        return _result(srf_low, False, res_low, 'nonconvergence',
                       stable_srf=srf_low)

    # Bracket failure: step upward from max(1.0, srf_low)
    srf = max(1.0, srf_low)
    last_stable_srf = srf_low
    last_stable_res = res_low
    bracketed = False

    while srf <= srf_high + 1e-12:
        ok, res = _try_srf(srf)
        if ok:
            last_stable_srf = srf
            last_stable_res = res
            srf = round(srf + bracket_step, 10)
        else:
            srf_lo_b = last_stable_srf
            srf_hi_b = srf
            bracketed = True
            break

    if not bracketed:
        return _result(srf_high, True, last_stable_res, 'range_exhausted',
                       stable_srf=last_stable_srf)

    # Bisection
    while (srf_hi_b - srf_lo_b) > tol:
        srf_mid = 0.5 * (srf_lo_b + srf_hi_b)
        ok, res = _try_srf(srf_mid)
        if ok:
            srf_lo_b = srf_mid
            last_stable_res = res
            last_stable_srf = srf_mid
        else:
            srf_hi_b = srf_mid

    fos = 0.5 * (srf_lo_b + srf_hi_b)
    basis = 'blowup' if blowup_seen[0] else 'nonconvergence'
    return _result(fos, True, last_stable_res, basis,
                   stable_srf=last_stable_srf)
