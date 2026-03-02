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
                      srf_tol=0.02, n_load_steps=10, t=1.0,
                      gwt=None, gamma_w=9.81):
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
    gwt : float, (M,2) array, or (n_nodes,) array, optional
        Groundwater table. See compute_pore_pressures() for formats.
    gamma_w : float — unit weight of water (kN/m^3). Default 9.81.

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

    # Compute pore pressures if GWT specified
    pp = None
    if gwt is not None:
        from fem2d.porewater import compute_pore_pressures
        pp = compute_pore_pressures(nodes, gwt, gamma_w)

    # Run SRM
    srm_result = strength_reduction(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        t=t, tol=srf_tol, n_load_steps=n_load_steps,
        pore_pressures=pp)

    result = _build_result(
        nodes, elements, srm_result['u_failure'],
        srm_result['stresses_failure'], None,
        analysis_type="srm")
    result.FOS = srm_result['FOS']
    result.converged = srm_result['converged']
    result.n_srf_trials = srm_result['n_srf_trials']
    return result


def create_wall_elements(nodes, x_wall, y_top, y_bottom, EA, EI,
                         weight_per_m=0.0, tol=0.5):
    """Create beam elements along a vertical wall line in the mesh.

    Finds mesh nodes near x=x_wall between y_bottom and y_top, sorts by
    elevation, and connects them with BeamElement objects.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    x_wall : float — x-coordinate of the wall line.
    y_top : float — top elevation of the wall.
    y_bottom : float — bottom (tip) elevation of the wall.
    EA : float — axial stiffness (kN).
    EI : float — flexural stiffness (kN*m^2).
    weight_per_m : float — self-weight per unit length (kN/m).
    tol : float — horizontal tolerance for finding wall nodes (m).

    Returns
    -------
    beam_elements : list of BeamElement
    wall_node_ids : list of int — sorted by elevation (top to bottom).
    """
    from fem2d.elements import BeamElement

    # Find nodes near x_wall within elevation range
    mask = (
        (np.abs(nodes[:, 0] - x_wall) < tol) &
        (nodes[:, 1] >= y_bottom - tol) &
        (nodes[:, 1] <= y_top + tol)
    )
    node_ids = np.where(mask)[0]
    if len(node_ids) < 2:
        return [], []

    # Sort by elevation (top to bottom)
    node_ids = node_ids[np.argsort(-nodes[node_ids, 1])]

    beam_elements = []
    for k in range(len(node_ids) - 1):
        beam_elements.append(BeamElement(
            node_i=int(node_ids[k]),
            node_j=int(node_ids[k + 1]),
            EA=EA, EI=EI,
            weight_per_m=weight_per_m,
        ))

    return beam_elements, list(node_ids)


def analyze_excavation(width, depth, wall_depth, soil_layers, wall_EI,
                       wall_EA, nx=30, ny=15, t=1.0, n_steps=10,
                       gwt=None, gamma_w=9.81):
    """Analyze a braced excavation with a sheet pile wall.

    Creates a rectangular domain with a vertical wall on the left side
    of an excavation. Excavation is modeled by removing gravity from
    elements inside the excavated zone.

    Parameters
    ----------
    width : float — excavation width (m).
    depth : float — excavation depth (m).
    wall_depth : float — total wall depth below surface (m).
    soil_layers : list of dict — soil properties (same format as analyze_slope_srm).
    wall_EI : float — wall flexural stiffness (kN*m^2/m).
    wall_EA : float — wall axial stiffness (kN/m).
    nx, ny : int — mesh density.
    t : float — thickness.
    n_steps : int — load steps.
    gwt : float, (M,2) array, or (n_nodes,) array, optional
        Groundwater table. See compute_pore_pressures() for formats.
    gamma_w : float — unit weight of water (kN/m^3). Default 9.81.

    Returns
    -------
    FEMResult with beam force results.
    """
    from fem2d.elements import BeamElement, beam2d_internal_forces
    from fem2d.assembly import (
        build_rotation_dof_map, beam_element_dofs,
    )
    from fem2d.results import BeamForceResult

    # Domain: wall at x=0, excavation to the right [0, width]
    # Extend left by 2*wall_depth, right by width + 2*wall_depth
    x_left = -2.0 * wall_depth
    x_right = width + 2.0 * wall_depth
    y_top = 0.0
    y_bottom = -max(wall_depth + depth, 2.0 * wall_depth)

    nodes, elements = generate_rect_mesh(
        x_left, x_right, y_bottom, y_top, nx, ny)
    bc_nodes = detect_boundary_nodes(nodes)

    # Assign layers
    if len(soil_layers) > 1:
        layer_bottoms = [sl['bottom_elevation'] for sl in soil_layers]
        layer_ids = assign_layers_by_elevation(nodes, elements, layer_bottoms)
    else:
        layer_ids = np.zeros(len(elements), dtype=int)

    # Build per-element material properties
    material_props = []
    gamma_arr = np.zeros(len(elements))
    centroids = nodes[elements].mean(axis=1)

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
        # Copy HS params if present
        if sl.get('model') == 'hs':
            mp['model'] = 'hs'
            for key in ('E50_ref', 'Eur_ref', 'm', 'p_ref', 'R_f'):
                mp[key] = sl[key]
        material_props.append(mp)

        # Reduce gamma to zero inside excavation zone (right of wall, above depth)
        cx, cy = centroids[e]
        if cx > 0 and cy > -depth:
            gamma_arr[e] = 0.0
        else:
            gamma_arr[e] = mp['gamma']

    # Compute pore pressures if GWT specified
    pp = None
    if gwt is not None:
        from fem2d.porewater import compute_pore_pressures
        pp = compute_pore_pressures(nodes, gwt, gamma_w)

    # Create wall elements at x=0
    # Tolerance scales with mesh spacing so coarse meshes still find wall nodes
    dx_mesh = (x_right - x_left) / nx
    wall_tol = max(dx_mesh * 0.6, 0.5)
    beam_elems, wall_nodes = create_wall_elements(
        nodes, x_wall=0.0, y_top=0.0, y_bottom=-wall_depth,
        EA=wall_EA, EI=wall_EI, tol=wall_tol)

    if not beam_elems:
        # Fallback: no wall nodes found, run without beams
        converged, u, stresses, strains = solve_nonlinear(
            nodes, elements, material_props, gamma_arr, bc_nodes,
            t=t, n_steps=n_steps, pore_pressures=pp)
        return _build_result(nodes, elements, u, stresses, strains,
                             analysis_type="excavation")

    # Build rotation DOF map and solve with beams
    rotation_dof_map, n_dof_total = build_rotation_dof_map(
        len(nodes), beam_elems)

    converged, u, stresses, strains = solve_nonlinear(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        t=t, n_steps=n_steps,
        beam_elements=beam_elems, rotation_dof_map=rotation_dof_map,
        pore_pressures=pp)

    result = _build_result(nodes, elements,
                           u[:2 * len(nodes)],  # translational DOFs only
                           stresses, strains,
                           analysis_type="excavation")
    result.converged = converged

    # Extract beam forces
    beam_force_results = []
    for idx, beam in enumerate(beam_elems):
        coords_ij = np.array([nodes[beam.node_i], nodes[beam.node_j]])
        bdofs = beam_element_dofs(beam.node_i, beam.node_j, rotation_dof_map)
        u_beam = u[bdofs]
        forces = beam2d_internal_forces(coords_ij, beam.EA, beam.EI, u_beam)
        beam_force_results.append(BeamForceResult(
            element_index=idx,
            node_i=beam.node_i, node_j=beam.node_j,
            axial_i=forces['axial_i'], shear_i=forces['shear_i'],
            moment_i=forces['moment_i'],
            axial_j=forces['axial_j'], shear_j=forces['shear_j'],
            moment_j=forces['moment_j'],
            length=forces['length'],
        ))

    result.n_beam_elements = len(beam_elems)
    result.beam_forces = beam_force_results
    if beam_force_results:
        result.max_beam_moment_kNm_per_m = max(
            max(abs(bf.moment_i), abs(bf.moment_j))
            for bf in beam_force_results)
        result.max_beam_shear_kN_per_m = max(
            max(abs(bf.shear_i), abs(bf.shear_j))
            for bf in beam_force_results)

    return result


def analyze_seepage(nodes, elements, k, head_bcs, t=1.0, gamma_w=9.81):
    """High-level steady-state seepage analysis.

    Solves the Laplace equation for hydraulic head using CST elements.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    k : float or (n_elements,) array — hydraulic conductivity (m/s).
    head_bcs : list of (node_id, head_value) — Dirichlet BCs.
    t : float — thickness.
    gamma_w : float — unit weight of water (kN/m^3).

    Returns
    -------
    SeepageResult
    """
    from fem2d.porewater import solve_seepage
    from fem2d.results import SeepageResult

    result_dict = solve_seepage(nodes, elements, k, head_bcs, t, gamma_w)

    vel = result_dict['velocity']
    v_mag = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)

    return SeepageResult(
        n_nodes=len(nodes),
        n_elements=len(elements),
        max_head_m=float(np.max(result_dict['head'])),
        min_head_m=float(np.min(result_dict['head'])),
        max_pore_pressure_kPa=float(np.max(result_dict['pore_pressures'])),
        max_velocity_m_per_s=float(np.max(v_mag)),
        total_flow_m3_per_s_per_m=result_dict['flow_rate'],
        head=result_dict['head'],
        pore_pressures=result_dict['pore_pressures'],
        velocity=result_dict['velocity'],
        nodes=np.asarray(nodes),
        elements=np.asarray(elements),
    )


def analyze_consolidation(width, depth, soil_layers, k, load_q,
                          time_points, gwt=0.0, gamma_w=9.81,
                          nx=10, ny=20, t=1.0, n_w=2.2e6):
    """1D-like consolidation of a loaded soil column.

    Sets up rectangular domain, applies surface load, tracks
    settlement and pore pressure dissipation over time.

    Parameters
    ----------
    width : float — domain width (m).
    depth : float — domain depth (m).
    soil_layers : list of dict — soil properties, each with:
        'E', 'nu', 'gamma' (and optionally 'bottom_elevation').
    k : float — hydraulic conductivity (m/s).
    load_q : float — surface load (kPa, positive downward).
    time_points : array-like — time points (s).
    gwt : float — GWT elevation (m). Default 0.0 (at surface).
    gamma_w : float — unit weight of water (kN/m^3).
    nx, ny : int — mesh density.
    t : float — thickness.
    n_w : float — bulk modulus of water (kPa).

    Returns
    -------
    ConsolidationResult
    """
    from fem2d.porewater import solve_consolidation, compute_pore_pressures
    from fem2d.results import ConsolidationResult

    nodes, elements = generate_rect_mesh(0, width, -depth, 0, nx, ny)
    bc_nodes = detect_boundary_nodes(nodes)

    # Assign layers
    if len(soil_layers) > 1:
        layer_bottoms = [sl['bottom_elevation'] for sl in soil_layers]
        layer_ids = assign_layers_by_elevation(nodes, elements, layer_bottoms)
    else:
        layer_ids = np.zeros(len(elements), dtype=int)

    # Build material props and gamma
    material_props = []
    gamma_arr = np.zeros(len(elements))
    for e in range(len(elements)):
        lid = min(layer_ids[e], len(soil_layers) - 1)
        sl = soil_layers[lid]
        mp = {
            'E': sl.get('E', 30000),
            'nu': sl.get('nu', 0.3),
        }
        material_props.append(mp)
        gamma_arr[e] = sl.get('gamma', 18)

    # Surface load: find top edges
    x_tol = 0.01
    surface_nodes = np.where(np.abs(nodes[:, 1]) < x_tol)[0]
    surface_nodes = surface_nodes[np.argsort(nodes[surface_nodes, 0])]
    surface_edges = []
    for i in range(len(surface_nodes) - 1):
        surface_edges.append((surface_nodes[i], surface_nodes[i + 1]))
    surface_loads = [(surface_edges, 0.0, -load_q)]

    # Drainage BCs: top surface is drained (head = gwt elevation)
    head_bcs = [(int(n), float(gwt)) for n in surface_nodes]

    # Initial pore pressures (hydrostatic from GWT)
    pp_0 = compute_pore_pressures(nodes, gwt, gamma_w)

    result_dict = solve_consolidation(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        k=k, head_bcs=head_bcs, time_steps=np.asarray(time_points),
        t=t, gamma_w=gamma_w, n_w=n_w,
        pore_pressures_0=pp_0, surface_loads=surface_loads)

    return ConsolidationResult(
        n_nodes=len(nodes),
        n_elements=len(elements),
        n_time_steps=len(time_points),
        times=result_dict['times'],
        max_settlement_m=result_dict['max_settlement_m'],
        max_excess_pore_pressure_kPa=result_dict['max_excess_pore_pressure_kPa'],
        degree_of_consolidation=result_dict['degree_of_consolidation'],
        converged=result_dict['converged'],
        displacements=result_dict['displacements'],
        pore_pressures=result_dict['pore_pressures'],
        settlements=result_dict['settlements'],
    )


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
