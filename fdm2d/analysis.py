"""
High-level analysis functions for 2D explicit FDM.

Provides the public API:
- analyze_gravity() — elastic/MC gravity loading of a soil column
- analyze_foundation() — strip load on elastic half-space
"""

import numpy as np

from fdm2d.grid import (
    generate_quad_grid, build_sub_triangles,
    compute_sub_triangle_geometry, detect_boundary_gridpoints,
    compute_lumped_mass,
)
from fdm2d.solver import solve_explicit
from fdm2d.zones import zone_averaged_stress
from fdm2d.results import FDMResult


def _build_bc_arrays(nodes, bc_dict):
    """Convert boundary dict to fixed/value arrays.

    Parameters
    ----------
    nodes : (n_gp, 2) array
    bc_dict : dict with 'fixed_base', 'roller_left', 'roller_right'.

    Returns
    -------
    bc_fixed : (n_gp, 2) bool array
    bc_values : (n_gp, 2) array — prescribed velocities (0).
    """
    n_gp = len(nodes)
    bc_fixed = np.zeros((n_gp, 2), dtype=bool)
    bc_values = np.zeros((n_gp, 2))

    # Fixed base: both x and y
    for nid in bc_dict['fixed_base']:
        bc_fixed[nid, 0] = True
        bc_fixed[nid, 1] = True

    # Roller left: fix x
    for nid in bc_dict['roller_left']:
        bc_fixed[nid, 0] = True

    # Roller right: fix x
    for nid in bc_dict['roller_right']:
        bc_fixed[nid, 0] = True

    return bc_fixed, bc_values


def _build_result(nodes, zones, displacements, stresses, velocities,
                  converged, n_timesteps, force_ratio, history,
                  analysis_type):
    """Build FDMResult from solver output."""
    ux = displacements[:, 0]
    uy = displacements[:, 1]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2)

    # Zone-averaged stresses for reporting
    zone_stress = zone_averaged_stress(stresses)

    result = FDMResult(
        analysis_type=analysis_type,
        n_gridpoints=len(nodes),
        n_zones=len(zones),
        max_displacement_m=float(disp_mag.max()),
        max_displacement_x_m=float(np.abs(ux).max()),
        max_displacement_y_m=float(np.abs(uy).max()),
        converged=converged,
        n_timesteps=n_timesteps,
        final_force_ratio=force_ratio,
        nodes=nodes,
        zones=zones,
        displacements=displacements,
        stresses=stresses,
        velocities=velocities,
        force_ratio_history=history,
    )

    if len(zone_stress) > 0:
        result.max_sigma_xx_kPa = float(np.max(np.abs(zone_stress[:, 0])))
        result.max_sigma_yy_kPa = float(np.max(zone_stress[:, 1]))
        result.min_sigma_yy_kPa = float(np.min(zone_stress[:, 1]))
        result.max_tau_xy_kPa = float(np.max(np.abs(zone_stress[:, 2])))

    return result


def analyze_gravity(width, depth, gamma, E, nu, nx=10, ny=10, t=1.0,
                    c=0.0, phi=0.0, psi=0.0,
                    max_steps=100000, tol=1e-5, damping=0.8):
    """Elastic/MC gravity analysis of a rectangular soil column.

    Parameters
    ----------
    width : float — domain width (m).
    depth : float — domain depth (m).
    gamma : float — unit weight (kN/m³).
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.
    nx, ny : int — mesh density.
    t : float — thickness.
    c : float — cohesion (kPa), default 0 (elastic).
    phi : float — friction angle (degrees), default 0 (elastic).
    psi : float — dilation angle (degrees), default 0.
    max_steps : int — maximum timesteps.
    tol : float — convergence tolerance.
    damping : float — local damping coefficient.

    Returns
    -------
    FDMResult
    """
    nodes, zones = generate_quad_grid(0, width, -depth, 0, nx, ny)
    sub_tris = build_sub_triangles(zones)
    B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
    bc_dict = detect_boundary_gridpoints(nodes)
    bc_fixed, bc_values = _build_bc_arrays(nodes, bc_dict)

    rho = gamma / 9.81
    mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho, t)

    material_props = {
        'E': E, 'nu': nu, 'gamma': gamma,
        'c': c, 'phi': phi, 'psi': psi,
    }

    result = solve_explicit(
        nodes, zones, sub_tris, B_all, areas, material_props,
        gamma, bc_fixed, bc_values, mass, len(nodes), t,
        max_steps=max_steps, tol=tol, damping=damping)

    converged, pos, disp, stresses, vel, n_steps, fr, history = result

    return _build_result(nodes, zones, disp, stresses, vel,
                         converged, n_steps, fr, history,
                         analysis_type="gravity")


def analyze_foundation(B, q, depth, E, nu, gamma=0.0,
                       nx=20, ny=10, t=1.0,
                       c=0.0, phi=0.0, psi=0.0,
                       max_steps=100000, tol=1e-5, damping=0.8):
    """Elastic analysis of a strip foundation on a half-space.

    Parameters
    ----------
    B : float — foundation width (m).
    q : float — applied pressure (kPa, positive downward).
    depth : float — domain depth (m).
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.
    gamma : float — soil unit weight (kN/m³). Default 0.
    nx, ny : int — mesh density.
    t : float — thickness.
    c : float — cohesion (kPa), default 0.
    phi : float — friction angle (degrees), default 0.
    psi : float — dilation angle (degrees), default 0.
    max_steps : int — maximum timesteps.
    tol : float — convergence tolerance.
    damping : float — local damping coefficient.

    Returns
    -------
    FDMResult
    """
    # Domain: 3B on each side
    x_extent = 3.0 * B
    nodes, zones = generate_quad_grid(
        -x_extent, x_extent, -depth, 0, nx, ny)
    sub_tris = build_sub_triangles(zones)
    B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
    bc_dict = detect_boundary_gridpoints(nodes)
    bc_fixed, bc_values = _build_bc_arrays(nodes, bc_dict)

    mat_gamma = gamma if gamma > 0 else 1.0  # Need nonzero for mass
    rho = mat_gamma / 9.81
    mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho, t)

    material_props = {
        'E': E, 'nu': nu, 'gamma': mat_gamma,
        'c': c, 'phi': phi, 'psi': psi,
    }

    # Find surface edges under the foundation
    x_tol = (2 * x_extent / nx) * 0.6  # tolerance scales with mesh spacing
    surface_nodes = np.where(np.abs(nodes[:, 1]) < 1e-6)[0]
    loaded_nodes = surface_nodes[
        (nodes[surface_nodes, 0] >= -B / 2 - x_tol) &
        (nodes[surface_nodes, 0] <= B / 2 + x_tol)]
    loaded_nodes = loaded_nodes[np.argsort(nodes[loaded_nodes, 0])]

    surface_edges = []
    for i in range(len(loaded_nodes) - 1):
        surface_edges.append((loaded_nodes[i], loaded_nodes[i + 1]))

    # Pressure: qy=-q (downward), qx=0
    surface_loads = [(surface_edges, 0.0, -q)]

    result = solve_explicit(
        nodes, zones, sub_tris, B_all, areas, material_props,
        gamma, bc_fixed, bc_values, mass, len(nodes), t,
        max_steps=max_steps, tol=tol, damping=damping,
        surface_loads=surface_loads)

    converged, pos, disp, stresses, vel, n_steps, fr, history = result

    return _build_result(nodes, zones, disp, stresses, vel,
                         converged, n_steps, fr, history,
                         analysis_type="foundation")
