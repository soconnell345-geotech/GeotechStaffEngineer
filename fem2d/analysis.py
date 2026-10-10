"""
High-level analysis functions for 2D FEM.

Provides the public API:
- analyze_gravity() — elastic gravity loading
- analyze_foundation() — strip load on elastic half-space
- analyze_footing_capacity() — ultimate bearing capacity by load-control collapse
- analyze_slope_srm() — slope stability via Strength Reduction Method
- analyze_staged() — staged construction (multi-phase)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fem2d.mesh import (
    generate_rect_mesh, generate_slope_mesh, detect_boundary_nodes,
    assign_layers_by_elevation, convert_to_t6, t6_boundary_edges,
)
from fem2d.materials import elastic_D
from fem2d.solver import (
    solve_elastic, solve_nonlinear, build_nl_context, run_nl,
)
from fem2d.srm import strength_reduction
from fem2d.results import FEMResult, PhaseResult, StagedConstructionResult


def analyze_gravity(width, depth, gamma, E, nu, nx=20, ny=10, t=1.0,
                    element_type='t6'):
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
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
    bc_nodes = detect_boundary_nodes(nodes)
    D = elastic_D(E, nu)

    u, stresses, strains = solve_elastic(
        nodes, elements, D, gamma, bc_nodes, t)

    return _build_result(nodes, elements, u, stresses, strains,
                         analysis_type="elastic")


def analyze_foundation(B, q, depth, E, nu, gamma=0.0, nx=30, ny=15, t=1.0,
                       element_type='t6'):
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
    n_corner = len(nodes)
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
    bc_nodes = detect_boundary_nodes(nodes)
    D = elastic_D(E, nu)

    # Find surface edges under the foundation
    x_tol = 0.01
    surface_nodes = np.where(np.abs(nodes[:n_corner, 1]) < x_tol)[0]
    loaded_nodes = surface_nodes[
        (nodes[surface_nodes, 0] >= -B / 2 - x_tol) &
        (nodes[surface_nodes, 0] <= B / 2 + x_tol)]
    loaded_nodes = loaded_nodes[np.argsort(nodes[loaded_nodes, 0])]

    surface_edges = []
    for i in range(len(loaded_nodes) - 1):
        surface_edges.append((loaded_nodes[i], loaded_nodes[i + 1]))
    if element_type == 't6':
        surface_edges = t6_boundary_edges(elements, surface_edges)

    surface_loads = [(surface_edges, 0.0, -q)]

    u, stresses, strains = solve_elastic(
        nodes, elements, D, gamma, bc_nodes, t,
        surface_loads=surface_loads)

    return _build_result(nodes, elements, u, stresses, strains,
                         analysis_type="elastic")


def bearing_capacity_factors(phi):
    """Classical bearing-capacity factors (Nc, Nq, Ngamma) for a strip footing.

    Prandtl–Reissner Nq and Nc with the Vesic (1973) Ngamma:
        Nq = e^(pi tan phi) tan^2(45 + phi/2)
        Nc = (Nq - 1) / tan phi           (Nc -> 2 + pi for phi = 0)
        Ngamma = 2 (Nq + 1) tan phi
    Closed-form only (no cross-module import) — used to size the collapse load
    ramp and as a reference in `analyze_footing_capacity`.
    """
    phi_r = math.radians(phi)
    if phi <= 1e-9:
        return 2.0 + math.pi, 1.0, 0.0
    Nq = math.exp(math.pi * math.tan(phi_r)) * math.tan(math.radians(45.0 + phi / 2.0)) ** 2
    Nc = (Nq - 1.0) / math.tan(phi_r)
    Ngamma = 2.0 * (Nq + 1.0) * math.tan(phi_r)
    return Nc, Nq, Ngamma


def analyze_footing_capacity(B, c, phi=0.0, gamma=0.0, E=1e5, nu=0.3, psi=0.0,
                             surcharge=0.0, q_max=None, n_load_steps=45,
                             domain_depth=None, domain_half_width=None,
                             nx=40, ny=20, element_type='t6',
                             max_iter=1000, tol=1e-4, q_applied=None):
    """Ultimate bearing capacity of a rigid strip footing by FEM load control.

    The FE analogue of a bearing-capacity calculation, analogous to how
    ``analyze_slope_srm`` wraps the slope workflow: it builds a half-space mesh,
    applies a uniform pressure over the footing width B, and ramps that pressure
    over ``n_load_steps`` increments until the mesh can no longer carry it. The
    last converged load level is the ultimate bearing pressure ``q_ult`` (the
    Griffiths-style "collapse = non-convergence" criterion). Validated against
    the Prandtl closed form q_ult = (2 + pi) c, Nc = 5.14 (VALIDATION.md §3;
    ~2% band with T6). **Use element_type='t6'** — CST locks and never collapses
    for this isochoric mechanism (VALIDATION.md §3).

    Parameters
    ----------
    B : float — footing width (m).
    c : float — cohesion / undrained shear strength (kPa).
    phi : float — friction angle (deg). Default 0 (Prandtl / undrained).
    gamma : float — soil unit weight (kN/m^3). Applied as a body force ramped
        with the load; for the classical weightless bearing mechanism use the
        default 0 (the validated basis). Default 0.
    E, nu, psi : float — elastic moduli (kPa / -) and dilation angle (deg).
    surcharge : float — uniform surcharge q beside the footing (kPa), applied
        as an initial (unramped) pressure on the surface outside B. Default 0.
    q_max : float, optional — top of the load ramp (kPa). Default:
        1.6 x the closed-form estimate c*Nc + surcharge*Nq + 0.5*gamma*B*Ngamma,
        chosen so collapse falls inside the ramp. Pass explicitly for c=phi=0.
    n_load_steps : int — load increments; collapse-load resolution is
        q_max / n_load_steps. Default 45.
    domain_depth, domain_half_width : float, optional — mesh extent below and to
        each side of the footing (m). Defaults 5*B each (10B wide, Prandtl-grade).
    nx, ny : int — mesh density. Default 40 x 20.
    element_type : 't6' (default) or 'cst'.
    q_applied : float, optional — a working/design pressure (kPa). If given, the
        bearing factor of safety q_ult / q_applied is reported.

    Returns
    -------
    FEMResult with additional attributes:
        q_ult_kPa — ultimate bearing pressure (last converged level).
        Nc_backfigured — q_ult/c when this is a pure-cohesion (surcharge=gamma=0)
            case, else None (compare to Prandtl Nc = 5.14).
        bearing_capacity_factors — {'Nc','Nq','Ngamma'} closed-form reference.
        q_ult_estimate_kPa — the closed-form estimate used to size the ramp.
        bearing_FOS — q_ult/q_applied when q_applied is given, else None.
        collapse_bracketed — False if the footing carried the full q_max (then
            q_ult is a lower bound; raise q_max).
        q_max_kPa, n_steps_converged, collapse_load_fraction.
    """
    if B <= 0:
        raise ValueError(f"footing width B must be positive, got {B}")
    if element_type not in ('t6', 'cst'):
        raise ValueError(f"element_type must be 't6' or 'cst', got {element_type!r}")

    Nc, Nq, Ngamma = bearing_capacity_factors(phi)
    q_ult_est = c * Nc + surcharge * Nq + 0.5 * gamma * B * Ngamma
    if q_max is None:
        if q_ult_est <= 0:
            raise ValueError(
                "Cannot auto-size q_max for a weightless cohesionless footing "
                "(c=0, gamma=0, surcharge=0); pass q_max explicitly.")
        q_max = 1.6 * q_ult_est
    if q_max <= 0:
        raise ValueError(f"q_max must be positive, got {q_max}")

    half_w = domain_half_width if domain_half_width is not None else 5.0 * B
    dep = domain_depth if domain_depth is not None else 5.0 * B

    nodes, elements = generate_rect_mesh(-half_w, half_w, -dep, 0.0, nx, ny)
    n_corner = len(nodes)
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
    bc_nodes = detect_boundary_nodes(nodes)

    # Surface edges under the footing (|x| <= B/2)
    x_tol = 1e-6 + B * 1e-6
    surf = np.where(np.abs(nodes[:n_corner, 1]) < 1e-9)[0]
    loaded = surf[(nodes[surf, 0] >= -B / 2 - x_tol) &
                  (nodes[surf, 0] <= B / 2 + x_tol)]
    loaded = loaded[np.argsort(nodes[loaded, 0])]
    if len(loaded) < 2:
        raise ValueError(
            "Footing spans fewer than 2 surface nodes; increase nx or B.")
    edges = [(loaded[i], loaded[i + 1]) for i in range(len(loaded) - 1)]
    if element_type == 't6':
        edges = t6_boundary_edges(elements, edges)

    surface_loads = [(edges, 0.0, -q_max)]

    # Optional surcharge beside the footing (unramped): applied via an initial
    # equilibrium is out of scope; here surcharge only sizes the ramp and the
    # reported factors. (The validated basis is the weightless Prandtl footing.)

    props = [{'E': E, 'nu': nu, 'c': c, 'phi': phi, 'psi': psi, 'gamma': gamma}]
    ctx = build_nl_context(nodes, elements, props, gamma, bc_nodes,
                           surface_loads=surface_loads)
    res = run_nl(ctx, n_steps=n_load_steps, max_iter=max_iter, tol=tol,
                 method='elastic')

    n_ok = len(res['iterations']) - (0 if res['converged'] else 1)
    n_ok = max(0, n_ok)
    frac = n_ok / n_load_steps
    q_ult = q_max * frac
    collapse_bracketed = not res['converged']  # full-load convergence => no collapse

    sig = res['sigma_gp'][:, :, [0, 1, 3]].mean(axis=1)
    result = _build_result(nodes, elements, res['u'], sig, None,
                           analysis_type="footing_capacity")
    result.converged = collapse_bracketed
    result.q_ult_kPa = float(q_ult)
    result.q_max_kPa = float(q_max)
    result.n_steps_converged = int(n_ok)
    result.collapse_load_fraction = float(frac)
    result.collapse_bracketed = bool(collapse_bracketed)
    result.bearing_capacity_factors = {'Nc': Nc, 'Nq': Nq, 'Ngamma': Ngamma}
    result.q_ult_estimate_kPa = float(q_ult_est)
    result.Nc_backfigured = (float(q_ult / c)
                             if (c > 0 and surcharge == 0 and gamma == 0)
                             else None)
    result.bearing_FOS = (float(q_ult / q_applied)
                          if (q_applied and q_applied > 0) else None)
    return result


def analyze_slope_srm(surface_points, soil_layers, depth=None,
                      nx=30, ny=15, x_extend=None,
                      srf_tol=0.02, n_load_steps=2, t=1.0,
                      gwt=None, gamma_w=9.81,
                      max_iter=1000, tol=1e-5,
                      layer_polylines=None,
                      element_type='t6', srm_field='c_phi',
                      blowup_factor=15.0, srf_range=(0.5, 3.0),
                      n_gp=None, nr_method='elastic', nr_fallback=False,
                      compute_local_fos=False, local_fos_cap=10.0):
    """Slope stability FOS via Strength Reduction Method.

    Parameters
    ----------
    surface_points : list of (x, z) tuples — ground surface profile.
    soil_layers : list of dict — soil properties, each with:
        'name', 'bottom_elevation', 'E', 'nu', 'c', 'phi',
        'psi' (optional, default 0), 'gamma'.
    depth : float, optional — depth below lowest surface. Default 2×H.
    nx, ny : int — mesh density.
    x_extend : float, optional — extra flat margin added to BOTH sides
        (0.3x each). Default max(0.5*H, 5). Pass 0 when the profile
        already includes adequate crest/toe margins (recommended).
    srf_tol : float — SRF bisection tolerance.
    n_load_steps : int — gravity increments.
    t : float
    gwt : float, (M,2) array, or (n_nodes,) array, optional
        Groundwater table. See compute_pore_pressures() for formats.
    gamma_w : float — unit weight of water (kN/m^3). Default 9.81.
    element_type : 't6' (default, quadratic — recommended) or 'cst'.
    srm_field : 'c_phi' | 'c' | 'phi' — which strengths to reduce.
    blowup_factor : float or None — displacement-blowup failure threshold
        (multiple of the dimensionless displacement at the lowest stable
        SRF). None = pure non-convergence criterion (Griffiths & Lane).
    srf_range : (float, float) — SRF search range.
    n_gp : int, optional — T6 Gauss rule override (3 or 6).

    Returns
    -------
    FEMResult with FOS, srf_history, srf_curve, fos_basis
    """
    surf = np.array(surface_points)
    z_min_surf = surf[:, 1].min()
    z_max_surf = surf[:, 1].max()
    H = z_max_surf - z_min_surf

    if depth is None:
        depth = max(2.0 * H, 10.0)
    if x_extend is None:
        # Depth-aware default: deep-seated mechanisms (weak foundation
        # layers) spread laterally about as far as the model is deep, so
        # the margin must scale with depth below the toe or the roller
        # boundaries truncate the failure mass (non-convergence at every
        # SRF). The old default (2x domain width) went the other way and
        # starved the slope face of elements at fixed nx, overpredicting
        # FOS by ~50%+ (verified against Bishop on a shared geometry).
        x_extend = max(0.5 * H, depth, 5.0)

    # Generate mesh
    nodes, elements = generate_slope_mesh(
        surface_points, depth, nx, ny,
        x_extend_left=x_extend * 0.3, x_extend_right=x_extend * 0.3)
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
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
        pore_pressures=pp, srm_field=srm_field,
        blowup_factor=blowup_factor, srf_range=srf_range, n_gp=n_gp,
        h_ref=H, nr_method=nr_method, nr_fallback=nr_fallback)

    result = _build_result(
        nodes, elements, srm_result['u_failure'],
        srm_result['stresses_failure'], None,
        analysis_type="srm")
    result.FOS = srm_result['FOS']
    result.converged = srm_result['converged']
    result.n_srf_trials = srm_result['n_srf_trials']
    result.srf_history = srm_result.get('srf_history')
    result.srf_curve = srm_result.get('srf_curve')
    result.fos_basis = srm_result.get('fos_basis')
    result.plastic_points = srm_result.get('plastic_gp')
    if compute_local_fos:
        # Local FOS at the critical-SRF stress field, evaluated with the
        # ORIGINAL (un-reduced) strengths: the low-FOS band traces the slip
        # surface and its minimum ~ the global SRM FOS (see fem2d/local_fos.py).
        from fem2d.local_fos import local_fos_field
        c_arr = np.array([mp.get('c', 0.0) for mp in material_props], dtype=float)
        phi_arr = np.array([mp.get('phi', 0.0) for mp in material_props],
                           dtype=float)
        result.local_fos = local_fos_field(result, c_arr, phi_arr,
                                            cap=local_fos_cap)
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
                       layer_polylines=None, element_type='t6'):
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
    if element_type == 't6':
        # Beams couple at corner AND midside nodes along the wall line
        # (each T6 edge contributes two beam segments).
        nodes, elements = convert_to_t6(nodes, elements)
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


def _consolidation_schedule(out_times, per_decade=25, max_steps=300):
    """Fine time-step schedule (s) containing every requested output time.

    t = 0, then a geometric sequence from 1/100 of the first positive output
    time to the last, merged with the output times themselves (which are
    kept exactly, so they can be picked out of the solution afterwards).
    """
    out_times = np.asarray(out_times, dtype=float)
    pos = out_times[out_times > 0]
    if len(pos) == 0:
        return np.array([0.0])
    t_lo, t_hi = pos[0] / 100.0, pos[-1]
    decades = max(np.log10(t_hi / t_lo), 1e-9)
    n = int(np.ceil(decades * min(per_decade, max_steps / decades))) + 1
    grid = np.logspace(np.log10(t_lo), np.log10(t_hi), max(n, 2))
    # Drop grid points that merely duplicate an output time to round-off.
    near = np.min(np.abs(grid[:, None] - pos[None, :]) / pos[None, :], axis=1)
    grid = grid[near > 1e-6]
    return np.unique(np.concatenate([[0.0], grid, out_times]))


_CUMULATIVE_OPTION = "cumulative from start"
_INITIAL_OPTION = "from end of the initial (gravity) stage"


def _fmt_movement(m):
    """'settlement 41.0 mm, max |u| 41.2 mm' from a movement summary."""
    txt = f"settlement {abs(m['settlement_m']) * 1000:.1f} mm"
    if m.get('heave_m', 0.0) * 1000 >= 0.05:
        txt += f", heave {m['heave_m'] * 1000:.1f} mm"
    if 'total_m' in m:
        txt += f", max |u| {m['total_m'] * 1000:.1f} mm"
    return txt


def _reference_judgment(initial_name, where, from_initial, cumulative,
                        option_hint):
    """The reference-stage choice as a ``judgment`` record: emitted only
    when the default was applied and it changes the reported numbers
    materially (callers check). ``from_initial`` / ``cumulative`` are
    movement summaries (dicts with settlement_m and optionally heave_m /
    total_m) at ``where`` (e.g. "stage 2 'Load'")."""
    return {
        "question": "Which stage are movements measured from?",
        "options": [
            {"name": _INITIAL_OPTION,
             "source": "common FE practice (e.g. PLAXIS: reset "
                       "displacements to zero after the K0/gravity stage)",
             "assumptions": f"the soil of {initial_name} was in place "
                            "before construction; its self-weight movement "
                            "is not a construction movement",
             "applies": True,
             "result": f"{where}: {_fmt_movement(from_initial)}"},
            {"name": _CUMULATIVE_OPTION,
             "source": "the total FE displacement since the start of the "
                       "analysis",
             "assumptions": "the self-weight movement happens during "
                            "construction (e.g. the soil is placed as fill)",
             "applies": True,
             "result": f"{where}: {_fmt_movement(cumulative)}"},
        ],
        "used": _INITIAL_OPTION,
        "why": "reset_displacements_after was not given, so construction "
               "movements exclude the initial self-weight movement. "
               + option_hint,
    }


def _movement_differs(a, b):
    """True when two movement summaries differ materially (settlement or
    largest |u|)."""
    from fem2d.results import _material_difference
    keys = [k for k in ("settlement_m", "total_m") if k in a and k in b]
    return any(_material_difference(a[k], b[k]) for k in keys)


def analyze_consolidation(width, depth, soil_layers, k, load_q,
                          time_points, gwt=0.0, gamma_w=9.81,
                          nx=10, ny=20, t=1.0, n_w=2.2e6,
                          layer_polylines=None,
                          consolidation_scheme="staggered", theta=1.0,
                          reset_displacements_after=None):
    """1D-like consolidation of a loaded soil column.

    Sets up rectangular domain, applies surface load, tracks
    settlement and pore pressure dissipation over time.

    Construction stages (2026-10-09). The analysis has two stages, each
    reported with its own movement and the cumulative movement
    (``ConsolidationResult.stages``): stage 0 ``initial`` — the self-weight
    of the soil in place, drained, in equilibrium with the hydrostatic water
    table (plus ponded water when ``gwt`` is above the surface); stage 1
    ``load`` — ``load_q`` plus the self-weight of any layer flagged
    ``'fill': True`` (placed during construction, with the load), then
    consolidation. ``reset_displacements_after`` picks the stage the
    reported settlements (``settlements``, ``surface_settlement_m_by_time``,
    ``max_settlement_m``) are measured from: None (default) or ``"initial"``
    — the end of the initial stage, so they are the load stage alone (common
    FE practice; the result carries a ``judgment`` record when this default
    changes the numbers materially); ``"start"`` — cumulative from the start,
    self-weight included. Phasing, not a global switch, decides whether
    self-weight movement counts: flag a layer ``fill`` when it is placed
    during construction.

    ``consolidation_scheme`` selects the Biot solver: "staggered" (**default**,
    sequential split — transports a pore field but does not create excess pore
    pressure from an applied load) or "monolithic" (coupled u-p, Taylor-Hood
    T6/T3, ``theta``-time-stepping — reproduces the load-induced undrained response
    p0 and the Terzaghi consolidation transient). Pass the flow ``k`` as the
    MOBILITY (m^2/(kPa.s)) and ``n_w`` as the Biot modulus M (kPa) for the
    monolithic transient; use a fine ``time_points`` schedule and theta in
    [0.5, 1] for the tightest decay match.

    Parameters
    ----------
    width : float — domain width (m).
    depth : float — domain depth (m).
    soil_layers : list of dict — soil properties, each with:
        'E', 'nu', 'gamma' (and optionally 'bottom_elevation', and
        'fill': True for a layer placed as fill during construction; fill
        layers must be the top layers and lie above the water table).
    k : float — hydraulic conductivity (m/s).
    load_q : float — surface load (kPa, positive downward).
    time_points : array-like — OUTPUT times since loading (s); t = 0 (the
        loading instant) is always reported first. The monolithic scheme
        integrates on a finer internal schedule between them.
    gwt : float — GWT elevation (m). Default 0.0 (at surface).
    gamma_w : float — unit weight of water (kN/m^3).
    nx, ny : int — mesh density.
    t : float — thickness.
    n_w : float — bulk modulus of water (kPa).
    reset_displacements_after : None, "initial" (or 0), or "start" — see
        above. Default None = after the initial stage.

    Returns
    -------
    ConsolidationResult
    """
    from fem2d.porewater import solve_consolidation, compute_pore_pressures
    from fem2d.results import ConsolidationResult, movement_summary

    # Which stage the reported movements are measured from.
    ref = reset_displacements_after
    if ref is None:
        ref_idx, chosen_by = 0, "default"
    else:
        key = str(ref).strip().lower()
        if key in ("initial", "0", "first"):
            ref_idx, chosen_by = 0, "user"
        elif key in ("start", "none"):
            ref_idx, chosen_by = -1, "user"
        else:
            raise ValueError(
                f"reset_displacements_after={ref!r}: use 'initial' (movements "
                f"from the end of the initial self-weight stage, the default) "
                f"or 'start' (cumulative from the start of the analysis)")

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
    fill_elements = []
    fill_names = []
    for e in range(len(elements)):
        lid = min(layer_ids[e], len(soil_layers) - 1)
        sl = soil_layers[lid]
        mp = {
            'E': sl.get('E', 30000),
            'nu': sl.get('nu', 0.3),
        }
        material_props.append(mp)
        gamma_arr[e] = sl.get('gamma', 18)
        if sl.get('fill'):
            # Placed as fill during construction: its self-weight belongs
            # to the load stage, not the initial state.
            fill_elements.append(e)
            name = sl.get('name', f"layer {lid}")
            if name not in fill_names:
                fill_names.append(name)
    if fill_elements:
        cy = nodes[elements].mean(axis=1)[:, 1]
        in_place_cy = np.delete(cy, fill_elements)
        if len(in_place_cy) and cy[fill_elements].min() < in_place_cy.max():
            raise ValueError(
                "soil_layers flagged 'fill' must be the TOP layers (placed on "
                "the soil in place); a fill layer lies below an unflagged "
                "one.")

    # Surface load: find top edges
    x_tol = 0.01
    surface_nodes = np.where(np.abs(nodes[:, 1]) < x_tol)[0]
    surface_nodes = surface_nodes[np.argsort(nodes[surface_nodes, 0])]
    surface_edges = []
    for i in range(len(surface_nodes) - 1):
        surface_edges.append((surface_nodes[i], surface_nodes[i + 1]))
    surface_loads = [(surface_edges, 0.0, -load_q)]

    # Water above the ground surface (gwt > 0) is ponded water: it loads the
    # surface in the initial state (without it the hydrostatic pore pressure
    # would lift the soil).
    pond_m = max(float(gwt), 0.0)
    initial_surface_loads = ([(surface_edges, 0.0, -gamma_w * pond_m)]
                             if pond_m > 0 else None)

    # Drainage BCs: top surface is drained (head = gwt elevation)
    head_bcs = [(int(n), float(gwt)) for n in surface_nodes]

    # Initial pore pressures (hydrostatic from GWT)
    pp_0 = compute_pore_pressures(nodes, gwt, gamma_w)

    # time_points are OUTPUT times measured from loading; t = 0 (the loading
    # instant) is always reported first.
    out_times = np.asarray(time_points, dtype=float).ravel()
    if out_times.size == 0 or np.any(out_times < 0):
        raise ValueError(
            "time_points must be one or more times since loading (s), >= 0")
    out_times = np.unique(np.concatenate([[0.0], out_times]))
    schedule = out_times
    if consolidation_scheme == "monolithic":
        # The coupled solve steps through a fine geometric schedule between
        # the requested times, so a sparse output list (e.g. 1e3, 1e7, 1e8 s)
        # is integrated accurately instead of in three giant steps (one
        # Crank-Nicolson step of 1e7 s oscillates). Starts two decades below
        # the first output time, ~25 steps per decade, <= ~300 steps.
        schedule = _consolidation_schedule(out_times)

    result_dict = solve_consolidation(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        k=k, head_bcs=head_bcs, time_steps=schedule,
        t=t, gamma_w=gamma_w, n_w=n_w,
        pore_pressures_0=pp_0, surface_loads=surface_loads,
        scheme=consolidation_scheme, theta=theta,
        initial_surface_loads=initial_surface_loads,
        fill_elements=fill_elements)

    # Per-stage fields: the initial stage, and the load stage per time.
    u0 = np.asarray(result_dict['initial_stage_displacements'], dtype=float)
    nodes_u = np.asarray(result_dict['displacement_nodes'], dtype=float)
    smask = np.asarray(result_dict['surface_node_mask'], dtype=bool)

    def _surface(rows):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        if not smask.any():
            return np.zeros(len(rows))
        return rows[:, 1::2][:, smask].min(axis=1)

    # Largest reported settlement over the whole (internal) schedule.
    load_full = np.asarray(result_dict['load_stage_displacements'],
                           dtype=float)
    max_settlement = float(_surface(
        load_full if ref_idx == 0 else load_full + u0[None, :]).min())

    # Report only the requested times.
    if len(schedule) != len(out_times):
        idx = np.searchsorted(result_dict['times'], out_times)
        for key in ('times', 'displacements', 'pore_pressures', 'settlements',
                    'excess_pore_pressures', 'total_pore_pressures',
                    'degree_of_consolidation_history',
                    'load_stage_displacements'):
            if result_dict.get(key) is not None:
                result_dict[key] = np.asarray(result_dict[key])[idx]

    times_out = np.asarray(result_dict['times'], dtype=float)
    load_u = np.asarray(result_dict['load_stage_displacements'], dtype=float)
    cum_u = load_u + u0[None, :]
    load_s, cum_s = _surface(load_u), _surface(cum_u)
    init_s = float(_surface(u0)[0])
    rep_u, rep_s = (load_u, load_s) if ref_idx == 0 else (cum_u, cum_s)
    U_hist = result_dict.get('degree_of_consolidation_history')

    weightless = not np.any(gamma_arr != 0.0)
    if weightless:
        init_desc = ("weightless soil (unit weight 0): the initial stage "
                     "carries no load")
    else:
        init_desc = ("self-weight of the soil in place, drained, in "
                     "equilibrium with the hydrostatic water table")
        if pond_m > 0:
            init_desc += f" and {pond_m:g} m of ponded water"
        if fill_names:
            init_desc += f" (fill layers {fill_names} not yet placed)"
    load_desc = f"surface load {load_q:g} kPa"
    if fill_names:
        load_desc += f" + self-weight of fill layers {fill_names}"
    load_desc += (", applied undrained, then consolidation"
                  if consolidation_scheme == "monolithic"
                  else ", drained (staggered scheme: no transient)")

    def _r(v):
        return round(float(v), 6) + 0.0

    stages = [
        {"stage": 0, "name": "initial", "description": init_desc,
         "surface_settlement_m": _r(init_s),
         "cumulative_surface_settlement_m": _r(init_s),
         "delta_displacement": movement_summary(nodes_u, u0),
         "cumulative_displacement": movement_summary(nodes_u, u0)},
        {"stage": 1, "name": "load", "description": load_desc,
         "time_s": [float(x) for x in times_out],
         "surface_settlement_m_by_time": [_r(v) for v in load_s],
         "cumulative_surface_settlement_m_by_time": [_r(v) for v in cum_s],
         "degree_of_consolidation_by_time": (
             [round(float(x), 4) for x in np.asarray(U_hist)]
             if U_hist is not None else None),
         "surface_settlement_m": _r(load_s[-1]),
         "cumulative_surface_settlement_m": _r(cum_s[-1]),
         "delta_displacement": movement_summary(nodes_u, load_u[-1]),
         "cumulative_displacement": movement_summary(nodes_u, cum_u[-1])},
    ]
    measured_from = ("end of stage 0 'initial' (self-weight)" if ref_idx == 0
                     else "start of analysis")
    displacement_reference = {
        "reset_after_stage": 0 if ref_idx == 0 else None,
        "measured_from": measured_from,
        "chosen_by": chosen_by,
    }
    judgment = None
    if chosen_by == "default":
        a = {"settlement_m": float(load_s[-1])}
        b = {"settlement_m": float(cum_s[-1])}
        if _movement_differs(a, b):
            judgment = _reference_judgment(
                "the column", f"surface at t = {times_out[-1]:.3g} s",
                a, b,
                "Pass reset_displacements_after='start' for cumulative "
                "settlement, or flag a layer 'fill': true when it is placed "
                "during construction.")

    notes = [
        "Times are measured from the instant the load is applied; time_s[0] "
        "= 0 is that instant (added if time_points did not start at 0).",
        "gwt is the water-table ELEVATION (m) in the model frame: the ground "
        "surface is at 0 and the base at -depth.",
        f"Settlements (max_settlement_m, surface_settlement_m_by_time) are "
        f"measured from the {measured_from}; stages[] gives each stage's own "
        f"movement and the cumulative movement.",
    ]
    if consolidation_scheme == "monolithic":
        notes.append(
            "Excess pore pressure comes from the load stage only (the "
            "surface load, plus any layer flagged fill): the self-weight of "
            "the soil in place is the initial stage, in drained equilibrium "
            "with the hydrostatic water before loading.")
        notes.append(
            "Early-time U is only as good as the mesh at the drained "
            "boundary: elements there should be thinner than sqrt(c t) "
            "(c = k / (1/n_w + 1/M_oed)); refine ny for small times.")
        if fill_names:
            notes.append(
                "Fill layers are saturated like the rest of the column in "
                "the monolithic scheme: above the water table that "
                "overstates their excess pore pressure.")
    else:
        notes.append(
            "The staggered scheme does not turn the applied load into excess "
            "pore pressure: it is drained at every step (U = 1, no "
            "consolidation transient). Use consolidation_scheme='monolithic' "
            "(k as the mobility m^2/(kPa.s) = hydraulic conductivity / "
            "gamma_w) for consolidation under the load.")

    return ConsolidationResult(
        n_nodes=len(nodes),
        n_elements=len(elements),
        n_time_steps=len(result_dict['times']),
        times=result_dict['times'],
        max_settlement_m=max_settlement,
        max_excess_pore_pressure_kPa=result_dict['max_excess_pore_pressure_kPa'],
        degree_of_consolidation=result_dict['degree_of_consolidation'],
        converged=result_dict['converged'],
        displacements=rep_u,
        pore_pressures=result_dict['pore_pressures'],
        settlements=rep_s,
        degree_of_consolidation_history=U_hist,
        excess_pore_pressures=result_dict.get('excess_pore_pressures'),
        final_drained_settlement_m=result_dict.get(
            'final_drained_settlement_m'),
        scheme=consolidation_scheme,
        notes=notes,
        stages=stages,
        displacement_reference=displacement_reference,
        judgment=judgment,
        initial_stage_displacements=u0,
        load_stage_displacements=load_u,
        displacement_nodes=nodes_u,
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
    reset_displacements : bool — report this phase's displacements (and
        later ones) from the START of this phase, i.e. the end of the
        previous one (PLAXIS "reset displacements to zero"). Reporting
        only: the solver's state is never altered. (Before 2026-10-09 it
        zeroed u in the solver, which then re-converged to the same total
        displacement, so it had no effect.)
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


def _resolve_reset_stage(value, phase_names):
    """``reset_displacements_after`` -> None (not given), -1 (no reset:
    cumulative from the start) or a stage index."""
    if value is None:
        return None
    hint = (f"use a stage index (0..{len(phase_names) - 1}), a stage name "
            f"{phase_names}, 'initial' (the first stage) or 'start' "
            f"(no reset: cumulative from the start)")
    if isinstance(value, bool):
        raise ValueError(f"reset_displacements_after={value!r}: {hint}")
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in ("start", "none"):
            return -1
        if v in phase_names:
            return phase_names.index(v)
        if v.lower() in ("initial", "first"):
            return 0
        try:
            value = int(v)
        except ValueError:
            raise ValueError(
                f"reset_displacements_after={value!r}: {hint}") from None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"reset_displacements_after={value!r}: {hint}") \
            from None
    if idx != value or not 0 <= idx < len(phase_names):
        raise ValueError(f"reset_displacements_after={value!r}: {hint}")
    return idx


def analyze_staged(nodes, elements, material_props, gamma, bc_nodes,
                   element_groups, phases, beam_elements=None,
                   t=1.0, max_iter=100, tol=1e-5, gamma_w=9.81,
                   reset_displacements_after=None):
    """Staged construction analysis.

    Solves a sequence of construction phases. Each phase activates a
    subset of soil element groups and (optionally) beam elements.
    Displacements, stresses, and strains carry forward cumulatively
    (per Gauss point). An element activated in a phase is placed
    stress-free on the deformed mesh: its strain counts from activation.

    Movements per stage (2026-10-09). Every phase reports its own movement
    (``delta_displacement``), the movement from the start of the analysis
    (``cumulative_displacement``; also ``max_displacement_*``, unchanged) and
    the movement from a reference stage (``displacement_since_reference``).
    Phasing decides whether self-weight counts: soil placed during
    construction (a fill group activated in a later phase) moves in that
    phase's delta; soil in place from the first phase is the initial state.

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
    reset_displacements_after : int, str or None — the stage at whose END
        ``displacement_since_reference`` is zeroed (later stages are
        measured from it; stages up to it from the start). A stage index, a
        stage name, ``"initial"`` (the first stage) or ``"start"`` (no
        reset: cumulative). Default None: if no phase sets
        ``reset_displacements``, the FIRST stage is taken as the initial
        (gravity / K0) stage and displacements are reset after it — common
        FE practice, so construction movements exclude self-weight — and
        the result carries a ``judgment`` record when that default changes
        the reported numbers materially. A phase's own
        ``reset_displacements=True`` also resets at its start.

    Returns
    -------
    StagedConstructionResult
    """
    from fem2d.assembly import (
        build_rotation_dof_map, beam_element_dofs,
    )
    from fem2d.results import BeamForceResult, movement_summary

    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    n_nodes_count = len(nodes)
    n_elem = len(elements)

    # Reference stage(s) for reporting: the stage ends at which the
    # reported (since-reference) displacements are zeroed.
    phase_names = [p.name for p in phases]
    user_ref = _resolve_reset_stage(reset_displacements_after, phase_names)
    phase_resets = {j - 1 for j, p in enumerate(phases)
                    if j >= 1 and p.reset_displacements}
    if user_ref is not None:
        reset_points = set(phase_resets)
        if user_ref >= 0:
            reset_points.add(user_ref)
        chosen_by = "user"
    elif any(p.reset_displacements for p in phases):
        reset_points, chosen_by = set(phase_resets), "user"
    else:
        reset_points = {0} if phases else set()
        chosen_by = "default"

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

    # Initialize cumulative state. The Gauss-point stress/strain arrays carry
    # between phases (element averages lose the T6 variation and the
    # out-of-plane stress of a plastic point).
    u = np.zeros(n_dof_total)
    sigma = np.zeros((n_elem, 3))
    strain = np.zeros((n_elem, 3))
    sig_gp = None
    eps_gp = None
    elem_state = [None] * n_elem
    prev_active = set()
    gp_cache = {}

    def _kinematic_strain(u_vec):
        """Strain B u at every Gauss point (n_e, n_gp, 3)."""
        if 'gp' not in gp_cache:
            from fem2d.solver import _gp_precompute
            gp_cache['gp'] = _gp_precompute(nodes, elements, t)
        gp = gp_cache['gp']
        return np.einsum('egki,ei->egk', gp['B'], u_vec[gp['dofs']])

    def _active_node_mask(active_elems, active_bms):
        mask = np.zeros(n_nodes_count, dtype=bool)
        if active_elems:
            mask[np.unique(elements[sorted(active_elems)].ravel())] = True
        if beam_elements and active_bms:
            for idx in active_bms:
                if 0 <= idx < len(beam_elements):
                    mask[beam_elements[idx].node_i] = True
                    mask[beam_elements[idx].node_j] = True
        return mask

    u_ends = []          # translational u at the end of each phase
    phase_results = []
    all_converged = True

    def _stage_movements(pr, pi, u_trans, mask):
        """Delta / cumulative / since-reference movements of phase pi."""
        prev = u_ends[pi - 1] if pi > 0 else np.zeros_like(u_trans)
        mask2 = np.repeat(mask, 2)
        delta = np.where(mask2, u_trans - prev, 0.0)
        earlier = [j for j in reset_points if j < pi]
        ref_k = max(earlier) if earlier else None
        u_ref = u_ends[ref_k] if ref_k is not None else np.zeros_like(u_trans)
        since = np.where(mask2, u_trans - u_ref, 0.0)
        pr.delta_displacements = delta
        pr.displacements_since_reference = since
        pr.active_node_mask = mask
        pr.delta_displacement = movement_summary(nodes, delta, mask)
        pr.cumulative_displacement = movement_summary(nodes, u_trans, mask)
        pr.displacement_since_reference = movement_summary(nodes, since, mask)
        pr.reference_stage_index = ref_k
        pr.reference_stage = (f"end of stage {ref_k} '{phases[ref_k].name}'"
                              if ref_k is not None else "start of analysis")
        u_ends.append(u_trans.copy())

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

        # 4. Elements activated in this phase are placed stress-free on the
        #    deformed mesh: zero stress, strain = B u now (so only movement
        #    after placement strains them), fresh HS state. Without this a
        #    fill's strain counted the movement its shared nodes made before
        #    it existed. (reset_displacements is reporting only — step 9.)
        newly = sorted(active_elems - prev_active)
        if newly and sig_gp is not None:
            sig_gp = np.array(sig_gp, dtype=float)
            eps_gp = np.array(eps_gp, dtype=float)
            sig_gp[newly] = 0.0
            eps_gp[newly] = _kinematic_strain(u)[newly]
            for e in newly:
                elem_state[e] = None
        prev_active = set(active_elems)

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
            _stage_movements(pr, pi, u[:2 * n_nodes_count].copy(),
                             _active_node_mask(active_elems, active_bms))
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
            u_init=u,
            sigma_init=sig_gp if sig_gp is not None else sigma,
            strain_init=eps_gp if eps_gp is not None else strain,
            state_init=elem_state,
            surface_loads=phase.surface_loads,
            return_state=True,
            return_gp=True,
        )

        (converged, u_new, sigma_new, strain_new, state_new,
         sig_gp, eps_gp) = result

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

        # Movements: this stage's own, cumulative, and since the reference.
        _stage_movements(pr, pi, u_trans.copy(),
                         _active_node_mask(active_elems, active_bms))
        phase_results.append(pr)

        # 10. Break if not converged
        if not converged:
            all_converged = False
            break

    # Which stage the reported (since-reference) movements start from.
    resets = sorted(j for j in reset_points if j < len(phases))
    displacement_reference = {
        "reset_after_stages": resets,
        "measured_from": (
            "; ".join(f"end of stage {j} '{phases[j].name}'" for j in resets)
            if resets else "start of analysis"),
        "chosen_by": chosen_by,
    }
    judgment = None
    if chosen_by == "default" and len(phase_results) > 1:
        last = phase_results[-1]
        if _movement_differs(last.displacement_since_reference,
                             last.cumulative_displacement):
            judgment = _reference_judgment(
                f"stage 0 '{phases[0].name}'",
                f"stage {last.phase_index} '{last.phase_name}'",
                last.displacement_since_reference,
                last.cumulative_displacement,
                "Pass reset_displacements_after='start' for cumulative "
                "movements, or a stage index or name; soil placed during "
                "construction belongs in its own stage.")
    notes = [
        "max_displacement_m / _x_m / _y_m and the displacements array are "
        "cumulative from the start of the analysis (every stage, gravity "
        "included).",
        "delta_displacement is each stage's own movement; "
        "displacement_since_reference is measured from "
        "displacement_reference (settlement_m negative = down, heave_m "
        "positive = up, horizontal_m signed +x).",
        "An element activated in a stage is placed stress-free on the "
        "deformed mesh; its nodes' movement counts from placement.",
    ]

    return StagedConstructionResult(
        n_phases=len(phase_results),
        n_nodes=n_nodes_count,
        n_elements=n_elem,
        converged=all_converged,
        phases=phase_results,
        nodes=nodes,
        elements=elements,
        displacement_reference=displacement_reference,
        judgment=judgment,
        notes=notes,
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
