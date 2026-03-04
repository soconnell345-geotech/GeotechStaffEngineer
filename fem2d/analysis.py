"""
High-level analysis functions for 2D FEM.

Provides the public API:
- analyze_gravity() — elastic gravity loading
- analyze_foundation() — strip load on elastic half-space
- analyze_slope_srm() — slope stability via Strength Reduction Method
- analyze_staged() — staged construction (multi-phase)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fem2d.mesh import (
    generate_rect_mesh, generate_slope_mesh, detect_boundary_nodes,
    assign_layers_by_elevation,
)
from fem2d.materials import elastic_D
from fem2d.solver import solve_elastic, solve_nonlinear
from fem2d.srm import strength_reduction
from fem2d.results import FEMResult, PhaseResult, StagedConstructionResult


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
                      gwt=None, gamma_w=9.81,
                      max_iter=100, tol=1e-5,
                      layer_polylines=None):
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
    if layer_polylines:
        from fem2d.mesh import assign_layers_by_polylines
        layer_ids = assign_layers_by_polylines(nodes, elements, layer_polylines)
    else:
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
        max_nr_iter=max_iter, nr_tol=tol,
        pore_pressures=pp)

    result = _build_result(
        nodes, elements, srm_result['u_failure'],
        srm_result['stresses_failure'], None,
        analysis_type="srm")
    result.FOS = srm_result['FOS']
    result.converged = srm_result['converged']
    result.n_srf_trials = srm_result['n_srf_trials']
    result.srf_history = srm_result.get('srf_history')
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
                       gwt=None, gamma_w=9.81, struts=None,
                       max_iter=100, tol=1e-5,
                       layer_polylines=None):
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
    struts : list of dict, optional
        Horizontal strut supports. Each dict: {'depth': float (m below surface),
        'stiffness': float (kN/m/m)}. Adds spring stiffness at wall nodes.
    max_iter : int — maximum Newton-Raphson iterations per step.
    tol : float — convergence tolerance.

    Returns
    -------
    FEMResult with beam force results and optional strut forces.
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
    if layer_polylines:
        from fem2d.mesh import assign_layers_by_polylines
        layer_ids = assign_layers_by_polylines(nodes, elements, layer_polylines)
    elif len(soil_layers) > 1:
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
            t=t, n_steps=n_steps, max_iter=max_iter, tol=tol,
            pore_pressures=pp)
        return _build_result(nodes, elements, u, stresses, strains,
                             analysis_type="excavation")

    # Build rotation DOF map and solve with beams
    rotation_dof_map, n_dof_total = build_rotation_dof_map(
        len(nodes), beam_elems)

    # Build strut spring list: [(node_id, stiffness), ...]
    strut_node_map = []
    if struts:
        for strut in struts:
            s_depth = strut['depth']
            s_k = strut['stiffness']
            if s_k <= 0:
                continue
            # Find wall node closest to y = -s_depth on the wall line (x≈0)
            target_y = -s_depth
            best_node = None
            best_dist = float('inf')
            for nid in wall_nodes:
                dist = abs(nodes[nid, 1] - target_y)
                if dist < best_dist:
                    best_dist = dist
                    best_node = nid
            if best_node is not None:
                strut_node_map.append((best_node, s_k))

    converged, u, stresses, strains = solve_nonlinear(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        t=t, n_steps=n_steps, max_iter=max_iter, tol=tol,
        beam_elements=beam_elems, rotation_dof_map=rotation_dof_map,
        pore_pressures=pp,
        strut_springs=strut_node_map if strut_node_map else None)

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

    # Extract strut forces: F = k * u_horizontal
    if strut_node_map:
        strut_force_results = []
        for node_id, s_k in strut_node_map:
            u_horiz = u[2 * node_id]  # horizontal DOF
            force = s_k * u_horiz
            strut_force_results.append({
                'depth_m': float(-nodes[node_id, 1]),
                'stiffness_kN_per_m': float(s_k),
                'force_kN_per_m': float(force),
                'node_id': int(node_id),
            })
        result.strut_forces = strut_force_results

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
                          nx=10, ny=20, t=1.0, n_w=2.2e6,
                          layer_polylines=None):
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
    if layer_polylines:
        from fem2d.mesh import assign_layers_by_polylines
        layer_ids = assign_layers_by_polylines(nodes, elements, layer_polylines)
    elif len(soil_layers) > 1:
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


@dataclass
class ConstructionPhase:
    """Definition of one construction phase.

    Attributes
    ----------
    name : str — descriptive phase name.
    active_soil_groups : list of str — group names to activate.
    active_beam_ids : list of int, optional — beam indices to activate.
        None means no beams active in this phase.
    surface_loads : list of (edges, qx, qy), optional.
    gwt : float, (M,2) array, or None — groundwater table for this phase.
        None means no pore pressures.
    n_steps : int — gravity load increments for this phase.
    reset_displacements : bool — zero u at start of this phase.
    """
    name: str = "Phase"
    active_soil_groups: List[str] = field(default_factory=list)
    active_beam_ids: Optional[List[int]] = None
    surface_loads: Optional[List] = None
    gwt: Any = None
    n_steps: int = 5
    reset_displacements: bool = False


def assign_element_groups(nodes, elements, regions):
    """Assign elements to named groups by centroid bounding box.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3 or 4) array — connectivity.
    regions : dict of str -> dict with keys 'x_min','x_max','y_min','y_max'.

    Returns
    -------
    groups : dict of str -> list of int (element indices).
        Elements matching no region go into '_default'.
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    n_elem = len(elements)

    # Compute centroids
    centroids = np.zeros((n_elem, 2))
    for e in range(n_elem):
        centroids[e] = nodes[elements[e]].mean(axis=0)

    groups = {name: [] for name in regions}
    groups['_default'] = []
    assigned = set()

    for name, bbox in regions.items():
        x_min = bbox.get('x_min', -np.inf)
        x_max = bbox.get('x_max', np.inf)
        y_min = bbox.get('y_min', -np.inf)
        y_max = bbox.get('y_max', np.inf)
        for e in range(n_elem):
            cx, cy = centroids[e]
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                groups[name].append(e)
                assigned.add(e)

    # Unassigned elements go to _default
    for e in range(n_elem):
        if e not in assigned:
            groups['_default'].append(e)

    return groups


def analyze_staged(nodes, elements, material_props, gamma, bc_nodes,
                   element_groups, phases, beam_elements=None,
                   t=1.0, max_iter=100, tol=1e-5, gamma_w=9.81):
    """Staged construction analysis.

    Solves a sequence of construction phases. Each phase activates a
    subset of soil element groups and (optionally) beam elements.
    Displacements, stresses, and strains carry forward cumulatively.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    elements : (n_elements, 3) array — CST connectivity.
    material_props : list of dict — per-element material properties.
    gamma : float or (n_elements,) array — unit weight.
    bc_nodes : dict from detect_boundary_nodes().
    element_groups : dict of str -> list of int — from assign_element_groups().
    phases : list of ConstructionPhase
    beam_elements : list of BeamElement, optional
    t : float — thickness.
    max_iter : int — max NR iterations per step.
    tol : float — convergence tolerance.
    gamma_w : float — unit weight of water.

    Returns
    -------
    StagedConstructionResult
    """
    from fem2d.assembly import (
        build_rotation_dof_map, beam_element_dofs,
    )
    from fem2d.results import BeamForceResult

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes_count = len(nodes)
    n_elem = len(elements)

    # Expand material properties
    if len(material_props) < n_elem:
        material_props = list(material_props) + \
            [material_props[-1]] * (n_elem - len(material_props))

    # Build rotation DOF map if beams present
    rotation_dof_map = None
    n_dof_total = 2 * n_nodes_count
    if beam_elements:
        rotation_dof_map, n_dof_total = build_rotation_dof_map(
            n_nodes_count, beam_elements)

    # Initialize cumulative state
    u = np.zeros(n_dof_total)
    sigma = np.zeros((n_elem, 3))
    strain = np.zeros((n_elem, 3))
    elem_state = [None] * n_elem

    phase_results = []
    all_converged = True

    for pi, phase in enumerate(phases):
        # 1. Compute active elements from group names
        active_elems = set()
        for group_name in phase.active_soil_groups:
            if group_name in element_groups:
                active_elems.update(element_groups[group_name])

        # 2. Compute active beams
        active_bms = None
        if phase.active_beam_ids is not None:
            active_bms = set(phase.active_beam_ids)

        # 3. Compute pore pressures if gwt provided
        pp = None
        if phase.gwt is not None:
            from fem2d.porewater import compute_pore_pressures
            pp = compute_pore_pressures(nodes, phase.gwt, gamma_w)

        # 4. Reset displacements if requested
        if phase.reset_displacements:
            u = np.zeros(n_dof_total)

        # 5. Handle empty active elements gracefully
        if len(active_elems) == 0:
            pr = PhaseResult(
                phase_name=phase.name,
                phase_index=pi,
                n_active_elements=0,
                n_active_beams=len(active_bms) if active_bms else 0,
                converged=True,
                displacements=u[:2 * n_nodes_count].copy(),
                stresses=sigma.copy(),
                strains=strain.copy(),
            )
            phase_results.append(pr)
            continue

        # 6. Call solve_nonlinear with cumulative state
        result = solve_nonlinear(
            nodes, elements, material_props, gamma, bc_nodes,
            t=t, n_steps=phase.n_steps, max_iter=max_iter, tol=tol,
            beam_elements=beam_elements,
            rotation_dof_map=rotation_dof_map,
            pore_pressures=pp,
            active_elements=active_elems,
            active_beams=active_bms,
            u_init=u, sigma_init=sigma, strain_init=strain,
            state_init=elem_state,
            surface_loads=phase.surface_loads,
            return_state=True,
        )

        converged, u_new, sigma_new, strain_new, state_new = result

        # 7. Update cumulative state
        u = u_new
        sigma = sigma_new
        strain = strain_new
        elem_state = state_new

        # 8. Extract beam forces for active beams
        beam_force_results = []
        n_active_beams = 0
        if beam_elements and active_bms:
            from fem2d.elements import beam2d_internal_forces
            n_active_beams = len(active_bms)
            for idx in sorted(active_bms):
                if idx >= len(beam_elements):
                    continue
                beam = beam_elements[idx]
                coords_ij = np.array([
                    nodes[beam.node_i], nodes[beam.node_j]])
                bdofs = beam_element_dofs(
                    beam.node_i, beam.node_j, rotation_dof_map)
                u_beam = u[bdofs]
                forces = beam2d_internal_forces(
                    coords_ij, beam.EA, beam.EI, u_beam)
                beam_force_results.append(BeamForceResult(
                    element_index=idx,
                    node_i=beam.node_i, node_j=beam.node_j,
                    axial_i=forces['axial_i'],
                    shear_i=forces['shear_i'],
                    moment_i=forces['moment_i'],
                    axial_j=forces['axial_j'],
                    shear_j=forces['shear_j'],
                    moment_j=forces['moment_j'],
                    length=forces['length'],
                ))

        # 9. Build PhaseResult
        u_trans = u[:2 * n_nodes_count]
        ux = u_trans[0::2]
        uy = u_trans[1::2]
        disp_mag = np.sqrt(ux**2 + uy**2)

        pr = PhaseResult(
            phase_name=phase.name,
            phase_index=pi,
            n_active_elements=len(active_elems),
            n_active_beams=n_active_beams,
            converged=converged,
            max_displacement_m=float(disp_mag.max()),
            max_displacement_x_m=float(np.abs(ux).max()),
            max_displacement_y_m=float(np.abs(uy).max()),
            displacements=u_trans.copy(),
            stresses=sigma.copy(),
            strains=strain.copy(),
        )

        # Stress statistics from active elements only
        active_list = sorted(active_elems)
        if len(active_list) > 0:
            active_stresses = sigma[active_list]
            pr.max_sigma_xx_kPa = float(np.max(np.abs(active_stresses[:, 0])))
            pr.max_sigma_yy_kPa = float(np.max(active_stresses[:, 1]))
            pr.min_sigma_yy_kPa = float(np.min(active_stresses[:, 1]))
            pr.max_tau_xy_kPa = float(np.max(np.abs(active_stresses[:, 2])))

        if beam_force_results:
            pr.n_beam_elements = len(beam_force_results)
            pr.beam_forces = beam_force_results
            pr.max_beam_moment_kNm_per_m = max(
                max(abs(bf.moment_i), abs(bf.moment_j))
                for bf in beam_force_results)
            pr.max_beam_shear_kN_per_m = max(
                max(abs(bf.shear_i), abs(bf.shear_j))
                for bf in beam_force_results)

        phase_results.append(pr)

        # 10. Break if not converged
        if not converged:
            all_converged = False
            break

    return StagedConstructionResult(
        n_phases=len(phase_results),
        n_nodes=n_nodes_count,
        n_elements=n_elem,
        converged=all_converged,
        phases=phase_results,
        nodes=nodes,
        elements=elements,
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
