"""
Explicit time-stepping solver for 2D FDM.

Central-difference time integration with local (non-viscous) damping.
Converges to static equilibrium via dynamic relaxation.

No global stiffness matrix — all operations are element-by-element.
"""

import math
import numpy as np

from fdm2d.materials import elastic_D, bulk_shear_moduli, wave_speed, mc_return_mapping
from fdm2d.zones import (
    compute_strain_rates, apply_mixed_discretization,
    compute_internal_forces, compute_gravity_forces,
    compute_surface_pressure,
)


def critical_timestep(nodes, zones, material_props, safety=0.5):
    """Compute critical timestep for explicit integration.

    dt = safety * min(sqrt(A_zone) / v_p) over all zones.

    Parameters
    ----------
    nodes : (n_gp, 2) array
    zones : (n_zones, 4) int array
    material_props : dict — must have 'E', 'nu', 'gamma'.
    safety : float — safety factor (default 0.5).

    Returns
    -------
    dt : float — critical timestep (seconds).
    """
    E = material_props['E']
    nu = material_props['nu']
    gamma = material_props['gamma']
    rho = gamma / 9.81

    K, G = bulk_shear_moduli(E, nu)
    vp = wave_speed(K, G, rho)

    dt_min = float('inf')
    for z in range(len(zones)):
        zone_nodes = nodes[zones[z]]
        # Zone area via shoelace
        x = zone_nodes[:, 0]
        y = zone_nodes[:, 1]
        n = len(x)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j] - x[j] * y[i]
        area = abs(area) / 2.0

        char_len = math.sqrt(area)
        dt_zone = char_len / vp
        if dt_zone < dt_min:
            dt_min = dt_zone

    return safety * dt_min


def solve_explicit(nodes, zones, sub_tris, B_all, areas, material_props,
                   gamma, bc_fixed, bc_values, mass, n_gp, t=1.0,
                   max_steps=100000, tol=1e-5, damping=0.8,
                   report_interval=1000, surface_loads=None,
                   dt=None):
    """Explicit time-stepping solver with local damping.

    Steps each iteration:
    1. Strain rates from velocities
    2. Mixed discretization correction
    3. Stress update: sigma += D * d_epsilon, then MC return mapping
    4. Internal forces from stresses
    5. Net force = gravity + surface - internal
    6. Damped force
    7. Velocity update with BCs
    8. Position update (small deformation: skip)
    9. Convergence check

    Parameters
    ----------
    nodes : (n_gp, 2) array — initial gridpoint positions.
    zones : (n_zones, 4) int array
    sub_tris : (n_zones, 4, 3) int array
    B_all : (n_zones, 4, 3, 6) array
    areas : (n_zones, 4) array
    material_props : dict — 'E', 'nu', 'gamma', optionally 'c', 'phi', 'psi'.
    gamma : float or (n_zones,) array — unit weight.
    bc_fixed : (n_gp, 2) bool array — True where DOF is fixed.
    bc_values : (n_gp, 2) array — prescribed velocity (usually 0).
    mass : (n_gp,) array — lumped mass per gridpoint.
    n_gp : int
    t : float — thickness.
    max_steps : int — maximum timesteps.
    tol : float — convergence tolerance on force ratio.
    damping : float — local damping coefficient (0 to 1).
    report_interval : int — convergence check interval.
    surface_loads : list of (edges, qx, qy), optional.
    dt : float, optional — override timestep.

    Returns
    -------
    converged : bool
    positions : (n_gp, 2) array — final positions.
    displacements : (n_gp, 2) array
    stresses : (n_zones, 4, 3) array — sub-triangle stresses.
    velocities : (n_gp, 2) array
    n_steps : int
    force_ratio : float — final force ratio.
    history : list of float — force ratio history.
    """
    n_zones = len(zones)
    E = material_props['E']
    nu = material_props['nu']
    c = material_props.get('c', 0.0)
    phi = material_props.get('phi', 0.0)
    psi = material_props.get('psi', 0.0)
    is_mc = c > 0 or phi > 0

    D = elastic_D(E, nu)

    # Compute timestep if not given
    if dt is None:
        dt = critical_timestep(nodes, zones, material_props)

    # Initialize
    pos = nodes.copy()
    vel = np.zeros((n_gp, 2))
    stresses = np.zeros((n_zones, 4, 3))

    # Gravity forces (constant)
    F_grav = compute_gravity_forces(nodes, zones, areas, gamma, n_gp, t)

    # Surface forces (constant)
    F_surf = np.zeros((n_gp, 2))
    if surface_loads:
        for edges, qx, qy in surface_loads:
            F_surf += compute_surface_pressure(
                nodes, edges, qx, qy, n_gp, t)

    # Total applied force magnitude for convergence check
    F_applied = F_grav + F_surf
    f_app_mag = np.sqrt(np.sum(F_applied ** 2))
    if f_app_mag < 1e-30:
        # No forces — already at equilibrium
        disp = np.zeros((n_gp, 2))
        return True, pos, disp, stresses, vel, 0, 0.0, [0.0]

    history = []
    force_ratio = 1.0

    for step in range(max_steps):
        # 1. Strain rates from velocities
        strain_rates = compute_strain_rates(vel, sub_tris, B_all, n_zones)

        # 2. Mixed discretization
        strain_rates = apply_mixed_discretization(strain_rates, areas)

        # 3. Stress update
        d_eps = strain_rates * dt
        for z in range(n_zones):
            for s in range(4):
                sigma_trial = stresses[z, s] + D @ d_eps[z, s]
                if is_mc:
                    sigma_new, _ = mc_return_mapping(
                        sigma_trial, E, nu, c, phi, psi)
                    stresses[z, s] = sigma_new
                else:
                    stresses[z, s] = sigma_trial

        # 4. Internal forces
        F_int = compute_internal_forces(
            pos, sub_tris, B_all, areas, stresses, n_gp, t)

        # 5. Net force
        F_net = F_grav + F_surf - F_int

        # 6. Local damping: F_damped = F_net - alpha * sign(v) * |F_net|
        sign_v = np.sign(vel)
        F_damped = F_net - damping * sign_v * np.abs(F_net)

        # 7. Velocity update
        for i in range(n_gp):
            if mass[i] > 1e-30:
                vel[i, 0] += (dt / mass[i]) * F_damped[i, 0]
                vel[i, 1] += (dt / mass[i]) * F_damped[i, 1]

        # Apply velocity BCs
        vel[bc_fixed] = bc_values[bc_fixed]

        # 8. Position update (small deformation — for displacement tracking)
        pos += dt * vel

        # 9. Convergence check
        if (step + 1) % report_interval == 0:
            F_unbal = F_grav + F_surf - F_int
            # Zero out fixed DOF contributions
            F_unbal[bc_fixed] = 0.0
            f_unbal_max = np.max(np.abs(F_unbal))
            force_ratio = f_unbal_max / f_app_mag
            history.append(force_ratio)

            if force_ratio < tol:
                disp = pos - nodes
                return (True, pos, disp, stresses, vel,
                        step + 1, force_ratio, history)

    # Did not converge
    disp = pos - nodes
    return False, pos, disp, stresses, vel, max_steps, force_ratio, history
