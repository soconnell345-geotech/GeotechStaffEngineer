"""
High-level analysis functions for 2D FEM.

Provides the public API:
- analyze_gravity() — elastic gravity loading
- analyze_foundation() — strip load on elastic half-space
- analyze_slope_srm() — slope stability via Strength Reduction Method
"""

import math
import numpy as np

from fem2d.mesh import (
    generate_rect_mesh, generate_slope_mesh, detect_boundary_nodes,
    assign_layers_by_elevation,
)
from fem2d.materials import elastic_D
from fem2d.solver import solve_elastic, solve_nonlinear
from fem2d.srm import strength_reduction
from fem2d.results import FEMResult


def analyze_gravity(width, depth, gamma, E, nu, nx=20, ny=10, t=1.0):
    """Elastic gravity analysis of a rectangular soil column.

    Parameters
    ----------
    width : float — domain width (m).
    depth : float — domain depth (m).
    gamma : float — unit weight (kN/m³).
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.
    nx, ny : int — mesh density.
    t : float — thickness.

    Returns
    -------
    FEMResult
    """
    nodes, elements = generate_rect_mesh(0, width, -depth, 0, nx, ny)
    bc_nodes = detect_boundary_nodes(nodes)
    D = elastic_D(E, nu)

    u, stresses, strains = solve_elastic(
        nodes, elements, D, gamma, bc_nodes, t)

    return _build_result(nodes, elements, u, stresses, strains,
                         analysis_type="elastic")


def analyze_foundation(B, q, depth, E, nu, gamma=0.0, nx=30, ny=15, t=1.0):
    """Elastic analysis of a strip foundation on a half-space.

    Parameters
    ----------
    B : float — foundation width (m).
    q : float — applied pressure (kPa, positive downward).
    depth : float — domain depth (m).
    E : float — Young's modulus (kPa).
    nu : float — Poisson's ratio.
    gamma : float — soil unit weight (kN/m³). Default 0 (no gravity).
    nx, ny : int — mesh density.
    t : float

    Returns
    -------
    FEMResult
    """
    # Domain: 3B on each side, depth below
    x_extent = 3.0 * B
    nodes, elements = generate_rect_mesh(
        -x_extent, x_extent, -depth, 0, nx, ny)
    bc_nodes = detect_boundary_nodes(nodes)
    D = elastic_D(E, nu)

    # Find surface edges under the foundation
    x_tol = 0.01
    surface_nodes = np.where(np.abs(nodes[:, 1]) < x_tol)[0]
    loaded_nodes = surface_nodes[
        (nodes[surface_nodes, 0] >= -B / 2 - x_tol) &
        (nodes[surface_nodes, 0] <= B / 2 + x_tol)]
    loaded_nodes = loaded_nodes[np.argsort(nodes[loaded_nodes, 0])]

    surface_edges = []
    for i in range(len(loaded_nodes) - 1):
        surface_edges.append((loaded_nodes[i], loaded_nodes[i + 1]))

    surface_loads = [(surface_edges, 0.0, -q)]

    u, stresses, strains = solve_elastic(
        nodes, elements, D, gamma, bc_nodes, t,
        surface_loads=surface_loads)

    return _build_result(nodes, elements, u, stresses, strains,
                         analysis_type="elastic")


def analyze_slope_srm(surface_points, soil_layers, depth=None,
                      nx=30, ny=15, x_extend=None,
                      srf_tol=0.02, n_load_steps=10, t=1.0):
    """Slope stability FOS via Strength Reduction Method.

    Parameters
    ----------
    surface_points : list of (x, z) tuples — ground surface profile.
    soil_layers : list of dict — soil properties, each with:
        'name', 'bottom_elevation', 'E', 'nu', 'c', 'phi',
        'psi' (optional, default 0), 'gamma'.
    depth : float, optional — depth below lowest surface. Default 2×H.
    nx, ny : int — mesh density.
    x_extend : float, optional — horizontal extension. Default 2×width.
    srf_tol : float — SRF bisection tolerance.
    n_load_steps : int — gravity increments.
    t : float

    Returns
    -------
    FEMResult with FOS
    """
    surf = np.array(surface_points)
    z_min_surf = surf[:, 1].min()
    z_max_surf = surf[:, 1].max()
    H = z_max_surf - z_min_surf

    if depth is None:
        depth = max(2.0 * H, 10.0)
    if x_extend is None:
        x_extend = max(2.0 * (surf[:, 0].max() - surf[:, 0].min()), 20.0)

    # Generate mesh
    nodes, elements = generate_slope_mesh(
        surface_points, depth, nx, ny,
        x_extend_left=x_extend * 0.3, x_extend_right=x_extend * 0.3)
    bc_nodes = detect_boundary_nodes(nodes)

    # Assign layers to elements
    layer_bottoms = [sl['bottom_elevation'] for sl in soil_layers]
    layer_ids = assign_layers_by_elevation(nodes, elements, layer_bottoms)

    # Build per-element material properties and gamma array
    material_props = []
    gamma_arr = np.zeros(len(elements))

    for e in range(len(elements)):
        lid = min(layer_ids[e], len(soil_layers) - 1)
        sl = soil_layers[lid]
        mp = {
            'E': sl.get('E', 30000),
            'nu': sl.get('nu', 0.3),
            'c': sl.get('c', 0),
            'phi': sl.get('phi', 0),
            'psi': sl.get('psi', 0),
            'gamma': sl.get('gamma', 18),
        }
        material_props.append(mp)
        gamma_arr[e] = mp['gamma']

    # Run SRM
    srm_result = strength_reduction(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        t=t, tol=srf_tol, n_load_steps=n_load_steps)

    result = _build_result(
        nodes, elements, srm_result['u_failure'],
        srm_result['stresses_failure'], None,
        analysis_type="srm")
    result.FOS = srm_result['FOS']
    result.converged = srm_result['converged']
    result.n_srf_trials = srm_result['n_srf_trials']
    return result


def _build_result(nodes, elements, u, stresses, strains, analysis_type):
    """Build a FEMResult from raw arrays."""
    n_dof = len(u)
    ux = u[0::2]
    uy = u[1::2]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2)

    result = FEMResult(
        analysis_type=analysis_type,
        n_nodes=len(nodes),
        n_elements=len(elements),
        max_displacement_m=float(disp_mag.max()),
        max_displacement_x_m=float(np.abs(ux).max()),
        max_displacement_y_m=float(np.abs(uy).max()),
        converged=True,
        nodes=nodes,
        elements=elements,
        displacements=u,
        stresses=stresses,
        strains=strains,
    )

    if stresses is not None and len(stresses) > 0:
        result.max_sigma_xx_kPa = float(np.max(np.abs(stresses[:, 0])))
        result.max_sigma_yy_kPa = float(np.max(stresses[:, 1]))
        result.min_sigma_yy_kPa = float(np.min(stresses[:, 1]))
        result.max_tau_xy_kPa = float(np.max(np.abs(stresses[:, 2])))

    return result
