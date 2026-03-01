"""
Strength Reduction Method (SRM) for slope stability FOS.

Progressively reduces c and tan(phi) by a Strength Reduction Factor
(SRF) until the FEM model fails to converge. The critical SRF = FOS.

Algorithm:
1. Establish gravity equilibrium at SRF = 1.0
2. Increment SRF by 0.1 until non-convergence to bracket failure
3. Bisect [SRF_low, SRF_high] to refine FOS to desired tolerance

Reference: Griffiths & Lane (1999), Géotechnique 49(3)
"""

import math
import numpy as np

from fem2d.solver import solve_nonlinear
from fem2d.mesh import detect_boundary_nodes


def strength_reduction(nodes, elements, material_props, gamma, bc_nodes=None,
                       t=1.0, srf_range=(0.5, 3.0), tol=0.02,
                       n_load_steps=10, max_nr_iter=100, nr_tol=1e-5):
    """Find slope stability FOS using the Strength Reduction Method.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
        Each dict: {'E', 'nu', 'c', 'phi', 'psi', 'gamma'}.
    gamma : float or array — unit weight (kN/m³).
    bc_nodes : dict, optional — from detect_boundary_nodes().
        Auto-detected if None.
    t : float — thickness.
    srf_range : (float, float) — search range for SRF.
    tol : float — bisection tolerance for FOS.
    n_load_steps : int — gravity load increments per SRF trial.
    max_nr_iter : int — max Newton-Raphson iterations.
    nr_tol : float — NR convergence tolerance.

    Returns
    -------
    dict with keys:
        'FOS': float — factor of safety.
        'converged': bool — whether bracketing succeeded.
        'u_failure': (n_dof,) array — displacements at last converged SRF.
        'stresses_failure': (n_elements, 3) array.
        'n_srf_trials': int — number of SRF evaluations.
    """
    if bc_nodes is None:
        bc_nodes = detect_boundary_nodes(nodes)

    n_elem = len(elements)

    # Expand material_props
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    def _try_srf(srf):
        """Run FEM with reduced strength, return (converged, u, stresses)."""
        reduced_props = []
        for mp in material_props:
            rp = dict(mp)
            c_orig = mp.get('c', 0.0)
            phi_orig = mp.get('phi', 0.0)
            psi_orig = mp.get('psi', 0.0)

            rp['c'] = c_orig / srf
            if phi_orig > 0:
                rp['phi'] = math.degrees(
                    math.atan(math.tan(math.radians(phi_orig)) / srf))
            else:
                rp['phi'] = 0.0
            rp['psi'] = 0.0  # Zero dilation for SRM
            reduced_props.append(rp)

        converged, u, stresses, strains = solve_nonlinear(
            nodes, elements, reduced_props, gamma, bc_nodes,
            t=t, n_steps=n_load_steps, max_iter=max_nr_iter, tol=nr_tol)
        return converged, u, stresses

    n_trials = 0

    # Phase 1: bracket failure by incrementing SRF
    srf_low = srf_range[0]
    srf_high = srf_range[1]

    # Verify lower bound converges
    n_trials += 1
    conv_low, u_low, s_low = _try_srf(srf_low)
    if not conv_low:
        return {
            'FOS': srf_low,
            'converged': False,
            'u_failure': np.zeros(2 * len(nodes)),
            'stresses_failure': np.zeros((n_elem, 3)),
            'n_srf_trials': n_trials,
        }

    # Increment from 1.0 upward to find non-convergence
    srf = 1.0
    d_srf = 0.1
    last_conv_srf = srf_low
    last_conv_u = u_low
    last_conv_s = s_low

    while srf <= srf_high:
        n_trials += 1
        conv, u, s = _try_srf(srf)
        if conv:
            last_conv_srf = srf
            last_conv_u = u
            last_conv_s = s
            srf += d_srf
        else:
            # Bracket found: [srf - d_srf, srf]
            srf_low = last_conv_srf
            srf_high = srf
            break
    else:
        # Never failed — FOS > srf_high
        return {
            'FOS': srf_high,
            'converged': True,
            'u_failure': last_conv_u,
            'stresses_failure': last_conv_s,
            'n_srf_trials': n_trials,
        }

    # Phase 2: bisection to refine FOS
    while (srf_high - srf_low) > tol:
        srf_mid = (srf_low + srf_high) / 2.0
        n_trials += 1
        conv, u, s = _try_srf(srf_mid)
        if conv:
            srf_low = srf_mid
            last_conv_u = u
            last_conv_s = s
        else:
            srf_high = srf_mid

    fos = (srf_low + srf_high) / 2.0

    return {
        'FOS': round(fos, 3),
        'converged': True,
        'u_failure': last_conv_u,
        'stresses_failure': last_conv_s,
        'n_srf_trials': n_trials,
    }
