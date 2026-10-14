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


# Elements across the slope height below which the strength-reduction FOS
# moved by more than ~2 % against a finer mesh (VALIDATION.md section 7:
# +2.7 % at 3 elements; within 1.7 % at 4.6 and more).
_SRM_MIN_ELEMENTS_ACROSS_SLOPE = 4
_SRM_MESH_CHECK_TOL_PCT = 5.0
# At or below this cohesion a slope face is treated as cohesionless (a
# larger c' makes a thin slide parallel to the face stronger than
# tan(phi')/tan(beta)).
_COHESIONLESS_C_KPA = 0.5


def _element_size(corners):
    """Per element: the side of the square two such triangles would make,
    sqrt(2 * area)."""
    x, y = corners[:, :, 0], corners[:, :, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    return np.sqrt(2.0 * area)


def _layer_warnings(nodes, elements, layer_ids, soil_layers, z_top, z_bottom,
                    elevation_layers):
    """A soil layer with no elements, or thinner than the elements there, is
    lost or reduced to a jagged band (elements take the layer of their
    centroid)."""
    warnings = []
    corners = nodes[np.asarray(elements)[:, :3]]
    size = _element_size(corners)
    # elements below the last bottom take the last layer
    ids = np.minimum(np.asarray(layer_ids), len(soil_layers) - 1)
    top = z_top
    for i, sl in enumerate(soil_layers):
        name = sl.get('name', f'layer {i}')
        mine = ids == i
        if elevation_layers:
            bottom = (z_bottom if i == len(soil_layers) - 1
                      else max(sl['bottom_elevation'], z_bottom))
            thick = top - bottom
            top = min(top, sl['bottom_elevation'])
            if thick <= 0:
                continue
        if not np.any(mine):
            warnings.append(
                f"Soil layer {name!r} has no elements: it is thinner than "
                "the mesh there, so the model leaves it out. Refine ny "
                "(more rows) until the layer is two elements thick.")
            continue
        if elevation_layers:
            h = float(np.median(size[mine]))
            if thick < h:
                warnings.append(
                    f"Soil layer {name!r} is {thick:.2f} m thick but the "
                    f"elements in it are about {h:.1f} m across: it is "
                    "modelled as a jagged band of whole elements, and a "
                    "failure along it is poorly represented. Refine ny until "
                    "the layer is two elements thick.")
    return warnings


def _cohesionless_face_warnings(surf, soil_layers, fos):
    """c' ~ 0 at a dry slope face: the critical slide is a shallow one
    parallel to the face, FOS = tan(phi') / tan(beta) (infinite slope). A
    mesh captures it only with small elements at the face, so a
    strength-reduction FOS above it is too high (Duncan Ex 2, c' 0.1 kPa,
    phi' 40 deg, 2:1: 1.81 / 1.74 / 1.71 on 15x8 / 30x15 / 45x24 against
    1.68)."""
    if fos is None:
        return []
    bottoms = [sl['bottom_elevation'] for sl in soil_layers]
    worst = None
    for (x0, z0), (x1, z1) in zip(surf[:-1], surf[1:]):
        if x1 == x0 or z1 == z0:
            continue
        beta = math.atan(abs(z1 - z0) / abs(x1 - x0))
        zm = 0.5 * (z0 + z1)
        idx = min(sum(zm < b for b in bottoms), len(soil_layers) - 1)
        sl = soil_layers[idx]
        c, phi = sl.get('c', 0.0), sl.get('phi', 0.0)
        if c > _COHESIONLESS_C_KPA or phi <= 0:
            continue
        f_inf = math.tan(math.radians(phi)) / math.tan(beta)
        if worst is None or f_inf < worst[0]:
            worst = (f_inf, math.degrees(beta), sl, c, phi)
    if worst is None or fos <= 1.02 * worst[0]:
        return []
    f_inf, beta_deg, sl, c, phi = worst
    return [
        f"The soil at the slope face ({sl.get('name', 'surface layer')!r}, "
        f"c' = {c:g} kPa, phi' = {phi:g} deg) is cohesionless: the critical "
        "slide there is shallow and parallel to the face, with factor of "
        f"safety tan(phi')/tan(beta) = {f_inf:.2f} on the {beta_deg:.1f} deg "
        f"face (infinite slope, dry). The strength-reduction FOS {fos:.2f} "
        "is higher because the elements at the face are too large to "
        f"capture that slide: refine the mesh at the face, or take "
        f"{f_inf:.2f}."]


def _srm_warnings(nodes, elements, z_toe, H, srm_result, srf_range):
    """Plain-language warnings for a strength-reduction result."""
    warnings = []
    corners = nodes[np.asarray(elements)[:, :3]]
    if H > 0:
        size = _element_size(corners)
        above_toe = corners[:, :, 1].mean(axis=1) > z_toe
        if np.any(above_toe):
            h = float(np.median(size[above_toe]))
            n_across = H / h
            if n_across < _SRM_MIN_ELEMENTS_ACROSS_SLOPE:
                warnings.append(
                    f"The elements on the slope are about {h:.1f} m across, "
                    f"so only {n_across:.1f} of them span the {H:.1f} m "
                    "height from toe to crest. With fewer than "
                    f"{_SRM_MIN_ELEMENTS_ACROSS_SLOPE} the strength-reduction "
                    "factor of safety can depend on the mesh: refine nx and "
                    "ny, or pass mesh_check=True to compare with a finer "
                    "mesh.")
    basis = srm_result.get('fos_basis')
    lo, hi = srf_range
    if basis == 'range_exhausted':
        warnings.append(
            f"The slope did not fail at any strength reduction factor up to "
            f"{hi}: the factor of safety is at least {hi}. Raise srf_range "
            "to find it.")
    elif not srm_result.get('converged', True):
        warnings.append(
            f"The slope already fails at the lowest strength reduction "
            f"factor tried ({lo}): the factor of safety is below {lo}. Lower "
            "srf_range to find it.")
    return warnings


def analyze_slope_srm(surface_points, soil_layers, depth=None,
                      nx=30, ny=15, x_extend=None,
                      srf_tol=0.02, n_load_steps=2, t=1.0,
                      gwt=None, gamma_w=9.81,
                      max_iter=1000, tol=1e-5,
                      layer_polylines=None,
                      element_type='t6', srm_field='c_phi',
                      blowup_factor=15.0, srf_range=(0.5, 3.0),
                      n_gp=None, nr_method='elastic', nr_fallback=False,
                      compute_local_fos=False, local_fos_cap=10.0,
                      settled_ratio=0.1, mesh_check=False):
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
        (0.3x each). Default max(0.5*H, depth, 5). Pass 0 when the
        profile already includes adequate crest/toe margins
        (recommended).
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
    settled_ratio : float or None — a trial that reaches the iteration
        ceiling still counts as stable when its displacements have stopped
        moving (see `strength_reduction`). Default 0.1.
    mesh_check : bool — also run the analysis on a mesh 1.5 times finer
        in each direction and report both factors of safety
        (``result.mesh_check``), with a warning when they differ by more
        than 5 %. Roughly triples the run time. Default False.

    Returns
    -------
    FEMResult with FOS, srf_history, srf_curve, fos_basis, warnings
    (plain-language: a slope covered by too few elements, a soil layer
    with no elements or thinner than its elements, a cohesionless dry face
    whose FOS came out above tan(phi')/tan(beta), a search range that did
    not bracket failure, a finer mesh that disagrees) and, with
    mesh_check, ``mesh_check``.
    """
    surf = np.array(surface_points)
    z_min_surf = surf[:, 1].min()
    z_max_surf = surf[:, 1].max()
    H = z_max_surf - z_min_surf
    for sl in soil_layers:
        nu_l = sl.get('nu', 0.3)
        if not -1.0 < nu_l < 0.5:
            raise ValueError(
                f"soil layer {sl.get('name', '?')!r}: Poisson's ratio "
                f"{nu_l} must be below 0.5 (use 0.49 to 0.499 for "
                "undrained behaviour)")

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
        h_ref=H, nr_method=nr_method, nr_fallback=nr_fallback,
        settled_ratio=settled_ratio)

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
    result.warnings.extend(_srm_warnings(
        nodes, elements, z_min_surf, H, srm_result, srf_range))
    result.warnings.extend(_layer_warnings(
        nodes, elements, layer_ids, soil_layers, z_max_surf,
        float(nodes[:, 1].min()), elevation_layers=not layer_polylines))
    if gwt is None and not layer_polylines:
        result.warnings.extend(_cohesionless_face_warnings(
            [tuple(p) for p in surf], soil_layers, result.FOS))
    if mesh_check:
        nx2, ny2 = int(math.ceil(1.5 * nx)), int(math.ceil(1.5 * ny))
        fine = analyze_slope_srm(
            surface_points, soil_layers, depth=depth, nx=nx2, ny=ny2,
            x_extend=x_extend, srf_tol=srf_tol, n_load_steps=n_load_steps,
            t=t, gwt=gwt, gamma_w=gamma_w, max_iter=max_iter, tol=tol,
            layer_polylines=layer_polylines, element_type=element_type,
            srm_field=srm_field, blowup_factor=blowup_factor,
            srf_range=srf_range, n_gp=n_gp, nr_method=nr_method,
            nr_fallback=nr_fallback, settled_ratio=settled_ratio)
        change = (fine.FOS - result.FOS) / result.FOS * 100.0
        result.mesh_check = {
            'mesh': f"{nx}x{ny}", 'FOS': result.FOS,
            'finer_mesh': f"{nx2}x{ny2}", 'FOS_finer_mesh': fine.FOS,
            'change_pct': round(change, 1),
        }
        if abs(change) > _SRM_MESH_CHECK_TOL_PCT:
            result.warnings.append(
                f"Factor of safety {result.FOS:.2f} on the {nx}x{ny} mesh "
                f"but {fine.FOS:.2f} on {nx2}x{ny2} ({change:+.0f} %): the "
                "answer depends on the mesh. Refine until two meshes agree "
                "(srm_mesh_refinement_study runs a sequence).")
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

    Finds the column of mesh nodes nearest x=x_wall (within ``tol``)
    between y_bottom and y_top, sorts it by elevation, and connects it with
    BeamElement objects.

    Only the NEAREST column is used (GM-70, 2026-10-14). Before, every node
    within ``tol`` was taken; on a T6 mesh the midside nodes of horizontal
    edges sit at x_wall +/- dx/2, inside the 0.6 dx tolerance
    ``analyze_excavation`` passes, so the "wall" hopped between three node
    columns (horizontal beam segments across the excavation face) and its
    moments came out several times too large.

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

    # Keep only the column nearest the wall line (GM-70; see docstring),
    # from the node nearest y_top to the node nearest y_bottom (``tol`` is
    # a horizontal search radius; used vertically it let a T6 wall run up
    # to tol past its tip).
    dist = np.abs(nodes[node_ids, 0] - x_wall)
    span = max(float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])), 1.0)
    eps = 1e-9 * span
    node_ids = node_ids[dist <= dist.min() + eps]
    ys = nodes[node_ids, 1]
    y_hi = ys[np.argmin(np.abs(ys - y_top))]
    y_lo = ys[np.argmin(np.abs(ys - y_bottom))]
    node_ids = node_ids[(ys <= y_hi + eps) & (ys >= y_lo - eps)]
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


def _graded_coords(levels, n_target):
    """Coordinates through every value in ``levels`` (sorted, distinct):
    each interval divided evenly, at about the spacing
    (levels[-1] - levels[0]) / n_target, and at least once."""
    levels = np.asarray(levels, dtype=float)
    h = (levels[-1] - levels[0]) / max(int(n_target), 1)
    coords = [levels[0]]
    for a, b in zip(levels[:-1], levels[1:]):
        k = max(1, int(round((b - a) / h)))
        coords.extend(np.linspace(a, b, k + 1)[1:])
    return np.array(coords)


def _layer_k0(sl):
    """At-rest coefficient of a soil layer and where it came from: the
    layer's own 'K0'; else Jaky's 1 - sin(phi') for a frictional layer;
    else nu / (1 - nu), the ratio one-dimensional elastic loading gives."""
    if sl.get('K0') is not None:
        return float(sl['K0']), "given"
    phi = float(sl.get('phi', 0.0) or 0.0)
    if phi > 0:
        return 1.0 - np.sin(np.radians(phi)), "Jaky, 1 - sin(phi')"
    nu = float(sl.get('nu', 0.3))
    return nu / (1.0 - nu), "nu / (1 - nu) (phi = 0)"


def _geostatic_stresses(nodes, elements, soil_layers, layer_ids,
                        layer_polylines, pp, t):
    """In-situ (K0) stresses at every Gauss point, tension positive:
    sigma_v' from the overburden of the soil layers (and any water ponded
    on the ground) less the pore pressure; sigma_h' = sigma_z' = K0
    sigma_v'; no shear. The layers are taken as the layer boundaries say
    (horizontal at ``bottom_elevation``, or ``layer_polylines``); the
    elements take the layer of their centroid, so where a boundary cuts an
    element the field is slightly out of balance with the element weights,
    which the in-situ stage settles."""
    from fem2d.solver import _gp_precompute

    gp = _gp_precompute(nodes, elements, t)
    N = gp['N']
    xe = nodes[elements]
    xg = np.einsum('gi,ei->eg', N, xe[:, :, 0])
    yg = np.einsum('gi,ei->eg', N, xe[:, :, 1])
    n_l = len(soil_layers)
    if layer_polylines:
        bottoms = [
            (lambda x, pl=np.asarray(p, dtype=float):
             np.interp(x, pl[:, 0], pl[:, 1]))
            for p in layer_polylines][:n_l - 1]
    elif n_l > 1:
        bottoms = [
            (lambda x, b=float(sl['bottom_elevation']): np.full_like(x, b))
            for sl in soil_layers[:n_l - 1]]
    else:
        bottoms = []
    y_ground = float(np.max(nodes[:, 1]))
    sv = np.zeros_like(xg)
    top = np.full_like(xg, y_ground)
    for i in range(len(bottoms) + 1):
        bot = bottoms[i](xg) if i < len(bottoms) \
            else np.full_like(xg, -np.inf)
        g = float(soil_layers[min(i, n_l - 1)].get('gamma', 18))
        sv += g * np.clip(top - np.maximum(bot, yg), 0.0, None)
        top = np.minimum(top, bot)
    p = np.zeros_like(xg)
    p_top = np.zeros_like(xg)
    if pp is not None:
        pp = np.clip(np.asarray(pp, dtype=float), 0.0, None)
        p = np.clip(np.einsum('gi,ei->eg', N, pp[elements]), 0.0, None)
        top_nodes = np.where(np.abs(nodes[:, 1] - y_ground) < 1e-9)[0]
        order = top_nodes[np.argsort(nodes[top_nodes, 0])]
        p_top = np.interp(xg, nodes[order, 0], pp[order])
    sv_eff = np.clip(sv + p_top - p, 0.0, None)
    k0 = np.array([_layer_k0(soil_layers[min(int(lid), n_l - 1)])[0]
                   for lid in layer_ids])
    sig = np.zeros(xg.shape + (4,))
    sig[:, :, 0] = -k0[:, None] * sv_eff
    sig[:, :, 1] = -sv_eff
    sig[:, :, 2] = -k0[:, None] * sv_eff
    return sig


# analyze_excavation accepts a load step that reaches the iteration ceiling
# with its displacements standing still (largest nodal change over the
# second half of the iterations < 0.1 of the first half) and a residual
# ratio below 1e-2 or still falling: the criterion the strength reduction
# uses (DESIGN.md, "Failure criterion"). In the active wedge behind a wall
# the constant-stiffness iteration converges slowly (the wedge is at the
# limit state and the wall's bending controls its movement) and stalls
# between 3e-5 and 1e-2 on the meshes measured (DESIGN.md, GM-71).
_EXC_SETTLED_RATIO = 0.1
_EXC_SETTLED_CAP = 1e-2


def analyze_excavation(width, depth, wall_depth, soil_layers, wall_EI,
                       wall_EA, nx=30, ny=15, t=1.0, n_steps=10,
                       gwt=None, gamma_w=9.81, struts=None,
                       max_iter=1000, tol=1e-5,
                       layer_polylines=None, element_type='t6',
                       lifts=None, wall_contact='smooth'):
    """Excavation in front of an embedded wall, staged (GM-71, 2026-10-14).

    The model is half of a symmetric excavation, in plane strain: the wall
    on x = 0, the retained soil to its left (to x = -2 wall_depth), the
    excavation from the wall to its centre line at x = width / 2, which is
    a line of symmetry (rollers); the base is fixed. A one-sided cut in
    front of open ground is the limit of a wide excavation: give a width
    of at least twice the passive zone in front of the wall.

    Stages (``analyze_staged``):

    0. In-situ stresses (K0 procedure): every element at sigma_v' from its
       overburden (less the pore pressure) and sigma_h' = K0 sigma_v'
       (each layer's 'K0', else Jaky's 1 - sin phi', else nu/(1-nu) for
       phi = 0), no displacement; the stage settles any out-of-balance.
       Displacements are reported from the end of this stage.
    1..n. One stage per lift. The wall (beam elements on x = 0, from the
       ground to -wall_depth) is wished in place at the start of the first
       lift, stress-free. Each lift removes the elements between the
       previous and the new excavation level; the forces they carried are
       released onto the remaining mesh (the stage's out-of-balance load is
       ramped over ``n_steps``). A strut is installed, stress-free, at the
       start of the first lift after the excavation has reached its level,
       so it carries only the movement after its installation.

    Before 2026-10-14 this function switched off the self-weight of the
    excavated elements and kept their stiffness, in one gravity solve; the
    wall head moved into the retained soil and the wall forces were not
    those of an excavation.

    Parameters
    ----------
    width : float — full excavation width (m); the model spans half of it.
    depth : float — final excavation depth below the ground (m).
    wall_depth : float — wall tip depth below the ground (m), > depth.
    soil_layers : list of dict — 'E', 'nu', 'c', 'phi', 'psi', 'gamma'
        (total unit weight), 'bottom_elevation' when there is more than
        one layer, optional 'K0' and the Hardening Soil keys.
    wall_EI : float — wall flexural stiffness (kN*m^2/m).
    wall_EA : float — wall axial stiffness (kN/m).
    nx, ny : int — about this many elements across and down (the mesh has
        a node column on the wall and a node row at every excavation level,
        strut level and the wall tip).
    t : float — thickness.
    n_steps : int — load steps in each stage.
    gwt : float, (M,2) array, or (n_nodes,) array, optional — groundwater
        table (compute_pore_pressures formats), the same in every stage.
        Where it is above the excavation level the excavation is kept full
        of water to it (no dewatering, no seepage; a warning says so).
    gamma_w : float — unit weight of water (kN/m^3).
    struts : list of dict, optional — {'depth': m below the ground (at or
        above the excavation level, < depth), 'stiffness': kN/m per m run}.
        A horizontal spring at the wall node nearest the depth (a strut, or
        an anchor's axial stiffness); stiffness <= 0 is skipped. No preload.
    max_iter : int — iteration ceiling per load step (default 1000 since
        2026-10-14, was 100: the active wedge behind a wall in sand needs
        several hundred constant-stiffness iterations a step). A step that
        reaches it with its displacements standing still is accepted, as in
        the strength reduction (see ``notes``).
    tol : float — residual tolerance.
    layer_polylines : list of (M,2) arrays, optional — layer bottoms.
    element_type : 't6' (default) or 't3'.
    lifts : list of float, optional — excavation depths (m) at the end of
        each lift, increasing; the last is taken to ``depth``. Default: one
        lift to each strut level, then to ``depth``. A stop is added at a
        strut level that a lift would pass without stopping.
    wall_contact : 'smooth' (default) or 'bonded'. 'smooth': the wall has
        its own nodes, tied to the soil horizontally and, at its tip, also
        vertically; the soil slides down or up the wall freely (no wall
        friction, delta = 0: the Rankine active pressure the USS Steel Sheet
        Piling Design Manual and Caltrans design methods use on the
        retained side). 'bonded': the wall shares the soil's nodes (no
        slip at all; before 2026-10-14 the only option). A bonded wall
        hangs the retained soil: on the USS cantilever benchmark its axial
        force reached 1.5 times the horizontal thrust above the excavation
        level (more than any wall friction angle up to phi' allows) and the
        thrust fell below the Coulomb active value with delta = phi'.
        There are no interface elements yet (a wall friction delta between
        0 and phi' needs them).

    Returns
    -------
    FEMResult (analysis_type "excavation"), at the end of the last stage:
        displacements and ``max_displacement_*`` — movement since the end of
            the in-situ stage (the excavation's own movement; before
            2026-10-14 the gravity movement of the one solve);
        elements, stresses, strains — the soil left at the end (the
            excavated elements are not in it); stresses are effective
            (in-situ plus change);
        beam_forces, max_beam_moment_kNm_per_m, max_beam_shear_kN_per_m —
            the wall at the end of the last stage (``wall_envelope`` holds
            the largest over all stages);
        strut_forces — per strut: 'force_kN_per_m' = stiffness x the wall's
            movement toward the excavation since installation (positive =
            the strut holds the wall back; before 2026-10-14 it counted the
            movement since the start), plus 'installed_in_stage';
        excavation_stages — one row per lift (wall head and largest wall
            movement, moment, shear, strut forces); notes; warnings;
        staged — the StagedConstructionResult (not serialized).

    Sign: x is positive from the retained soil toward the excavation, so a
    wall moving toward the excavation has u_x > 0.
    """
    width = float(width)
    depth = float(depth)
    wall_depth = float(wall_depth)
    if width <= 0:
        raise ValueError(f"width must be > 0 m (got {width})")
    if depth <= 0:
        raise ValueError(f"depth must be > 0 m (got {depth})")
    if wall_depth <= depth:
        raise ValueError(
            f"wall_depth ({wall_depth} m) must be greater than depth "
            f"({depth} m): the wall has to reach below the excavation level")
    if not soil_layers:
        raise ValueError("soil_layers is empty")
    if wall_contact not in ('smooth', 'bonded'):
        raise ValueError(f"wall_contact must be 'smooth' or 'bonded' (got "
                         f"{wall_contact!r})")

    # Struts: (depth, stiffness, input index)
    strut_defs = []
    for j, s in enumerate(struts or []):
        k = float(s['stiffness'])
        d = float(s['depth'])
        if k <= 0:
            continue
        if d < 0 or d >= depth - 1e-9:
            raise ValueError(
                f"struts[{j}]: depth {d} m must be at or below the ground "
                f"and above the final excavation level ({depth} m); a strut "
                f"is installed once the excavation has reached its level")
        strut_defs.append((d, k, j))

    # Lifts: excavation depth at the end of each stage
    if lifts is None:
        lift_depths = [depth]
    else:
        lift_depths = [float(d) for d in lifts]
        if not lift_depths or lift_depths[0] <= 0 or any(
                b <= a for a, b in zip(lift_depths[:-1], lift_depths[1:])):
            raise ValueError(f"lifts must be increasing depths > 0 (got "
                             f"{list(lifts)})")
        if lift_depths[-1] > depth + 1e-9:
            raise ValueError(f"lifts go below the excavation depth {depth} m")
        if lift_depths[-1] < depth - 1e-9:
            lift_depths.append(depth)
        else:
            lift_depths[-1] = depth
    for d, _, _ in strut_defs:
        # the first lift that reaches the strut level must not be the last
        first = next(b for b in lift_depths if b >= d - 1e-9)
        if abs(first - depth) < 1e-9 and d > 0:
            lift_depths = sorted(set(lift_depths) | {d})

    # Domain: the retained side as before; the excavation to its centre line
    x_left = -2.0 * wall_depth
    x_right = 0.5 * width
    y_top = 0.0
    y_bottom = -max(wall_depth + depth, 2.0 * wall_depth)
    h_y = (y_top - y_bottom) / max(int(ny), 1)
    levels = sorted({y_top, y_bottom, -wall_depth}
                    | {-d for d in lift_depths})
    for d, _, _ in strut_defs:
        if min(abs(-d - lv) for lv in levels) > 0.25 * h_y:
            levels = sorted(set(levels) | {-d})
    xs = _graded_coords([x_left, 0.0, x_right], nx)
    ys = _graded_coords(levels, ny)
    n_cx, n_cy = len(xs) - 1, len(ys) - 1
    idx_nodes, elements = generate_rect_mesh(0.0, float(n_cx), 0.0,
                                             float(n_cy), n_cx, n_cy)
    ix = np.rint(idx_nodes[:, 0]).astype(int)
    iy = np.rint(idx_nodes[:, 1]).astype(int)
    nodes = np.column_stack([xs[ix], ys[iy]])
    if element_type == 't6':
        # Beams couple at corner AND midside nodes along the wall line
        # (each T6 edge contributes two beam segments).
        nodes, elements = convert_to_t6(nodes, elements)
    bc_nodes = detect_boundary_nodes(nodes)
    n_elem = len(elements)

    # Layers and materials
    if layer_polylines:
        from fem2d.mesh import assign_layers_by_polylines
        layer_ids = assign_layers_by_polylines(nodes, elements,
                                               layer_polylines)
    elif len(soil_layers) > 1:
        layer_bottoms = [sl['bottom_elevation'] for sl in soil_layers]
        layer_ids = assign_layers_by_elevation(nodes, elements,
                                               layer_bottoms)
    else:
        layer_ids = np.zeros(n_elem, dtype=int)
    layer_ids = np.minimum(layer_ids, len(soil_layers) - 1)
    material_props = []
    gamma_arr = np.zeros(n_elem)
    for e in range(n_elem):
        sl = soil_layers[int(layer_ids[e])]
        mp = {
            'E': sl.get('E', 30000),
            'nu': sl.get('nu', 0.3),
            'c': sl.get('c', 0),
            'phi': sl.get('phi', 0),
            'psi': sl.get('psi', 0),
            'gamma': sl.get('gamma', 18),
        }
        if sl.get('model') == 'hs':
            mp['model'] = 'hs'
            for key in ('E50_ref', 'Eur_ref', 'm', 'p_ref', 'R_f'):
                mp[key] = sl[key]
        material_props.append(mp)
        gamma_arr[e] = mp['gamma']

    # Element groups: the retained soil, and one group per lift
    centroids = nodes[elements].mean(axis=1)
    lift_of = np.full(n_elem, -1)
    prev = 0.0
    for k, d in enumerate(lift_depths):
        sel = ((centroids[:, 0] > 0.0) & (centroids[:, 1] > -d)
               & (centroids[:, 1] < -prev) & (lift_of < 0))
        lift_of[sel] = k
        prev = d
    groups = {'soil': [int(e) for e in np.where(lift_of < 0)[0]]}
    for k in range(len(lift_depths)):
        groups[f'lift_{k + 1}'] = [int(e) for e in np.where(lift_of == k)[0]]
    all_groups = list(groups)

    # The wall: one node column on x = 0 from the ground to the tip
    spacing = float(min(np.min(np.diff(xs)), np.min(np.diff(ys))))
    beam_elems, wall_nodes = create_wall_elements(
        nodes, x_wall=0.0, y_top=0.0, y_bottom=-wall_depth,
        EA=wall_EA, EI=wall_EI, tol=0.25 * spacing)
    if not beam_elems:
        raise RuntimeError("no wall nodes found on x = 0")
    springs = []
    zip_ties, wall_ties, right_ties = [], [], []    # spring indices
    if wall_contact == 'smooth':
        # Cut the soil along the wall above its tip: the elements right of
        # the wall get their own copies of the column's nodes. In the
        # in-situ stage the two faces are tied together (x and y), so the
        # soil is whole; from the first lift the wall has its own nodes,
        # tied to each face horizontally only (no wall friction) and to the
        # soil at its tip in both directions, where it bears on the soil.
        # A face tie on the excavation side goes with the soil above the
        # excavation level.
        col = list(wall_nodes)                  # top to tip
        upper = col[:-1]
        n0 = len(nodes)
        right_ids = list(range(n0, n0 + len(upper)))
        wall_ids = list(range(n0 + len(upper), n0 + len(upper) + len(col)))
        nodes = np.vstack([nodes, nodes[upper], nodes[col]])
        to_right = dict(zip(upper, right_ids))
        cx = nodes[elements[:, :3]].mean(axis=1)[:, 0]
        elements = np.array(elements)
        for e in np.where(cx > 0.0)[0]:
            elements[e] = [to_right.get(int(n), int(n)) for n in elements[e]]
        to_wall = dict(zip(col, wall_ids))
        for bm in beam_elems:
            bm.node_i = to_wall[bm.node_i]
            bm.node_j = to_wall[bm.node_j]
        wall_nodes = wall_ids
        h_min = min(float(np.hypot(*(nodes[b.node_j] - nodes[b.node_i])))
                    for b in beam_elems)
        e_max = max(float(sl.get('E', 30000)) for sl in soil_layers)
        k_tie = 1e3 * max(e_max, float(wall_EA) / h_min,
                          12.0 * float(wall_EI) / h_min ** 3)
        for left, right in zip(upper, right_ids):
            zip_ties.append(len(springs))
            springs.append({'node': right, 'node_b': left, 'kx': k_tie,
                            'ky': k_tie})
        for w, left in zip(wall_ids, col):
            wall_ties.append(len(springs))
            springs.append({'node': w, 'node_b': left, 'kx': k_tie,
                            'ky': k_tie if left == col[-1] else 0.0})
        for w, right in zip(wall_ids, right_ids):
            right_ties.append((len(springs), float(nodes[right, 1])))
            springs.append({'node': w, 'node_b': right, 'kx': k_tie})
    n_ties = len(springs)

    # Struts as springs at the wall node nearest their level
    for d, k, j in strut_defs:
        nid = min(wall_nodes, key=lambda n: abs(nodes[n, 1] + d))
        springs.append({'node': int(nid), 'kx': k})

    pp = None
    if gwt is not None:
        from fem2d.porewater import compute_pore_pressures
        pp = compute_pore_pressures(nodes, gwt, gamma_w)

    # Stages
    phases = [ConstructionPhase(name="In-situ stresses (K0)",
                                active_soil_groups=all_groups, gwt=gwt,
                                n_steps=n_steps,
                                active_spring_ids=list(zip_ties))]
    installed = []
    install_stage = {}
    excavated = set()
    reached = 0.0
    for k, d in enumerate(lift_depths):
        new = [i for i, (sd, _, _) in enumerate(strut_defs)
               if i not in installed and sd <= reached + 1e-9]
        for i in new:
            installed.append(i)
            install_stage[i] = k + 1
        excavated.add(f'lift_{k + 1}')
        name = f"Excavate to {d:.2f} m"
        if new:
            name += " (struts at " + ", ".join(
                f"{strut_defs[i][0]:.2f}" for i in new) + " m installed)"
        phases.append(ConstructionPhase(
            name=name,
            active_soil_groups=[g for g in all_groups if g not in excavated],
            active_beam_ids=list(range(len(beam_elems))),
            active_spring_ids=list(wall_ties)
            + [i for i, y in right_ties if y <= -d + 1e-9]
            + [n_ties + i for i in installed],
            gwt=gwt, n_steps=n_steps))
        reached = d

    sig0 = _geostatic_stresses(nodes, elements, soil_layers, layer_ids,
                               layer_polylines, pp, t)
    staged = analyze_staged(
        nodes, elements, material_props, gamma_arr, bc_nodes,
        element_groups=groups, phases=phases, beam_elements=beam_elems,
        t=t, max_iter=max_iter, tol=tol, gamma_w=gamma_w,
        reset_displacements_after=0, springs=springs,
        initial_stresses=sig0, water_on_free_faces=True,
        settled_ratio=_EXC_SETTLED_RATIO,
        settled_residual_cap=_EXC_SETTLED_CAP)

    # Result: the end of the last stage reached
    last = staged.phases[-1]
    n_n = len(nodes)
    u_rep = np.asarray(last.displacements_since_reference, dtype=float)
    n_lifts_done = len(staged.phases) - 1
    # The result's mesh is the soil left at the end (the full mesh and every
    # stage are in result.staged).
    keep = np.where(~((lift_of >= 0) & (lift_of < n_lifts_done)))[0]
    stresses = np.array(last.stresses, dtype=float)[keep]
    strains = np.array(last.strains, dtype=float)[keep]
    result = _build_result(nodes, np.asarray(elements)[keep],
                           u_rep[:2 * n_n], stresses, strains,
                           analysis_type="excavation")
    result.converged = bool(staged.converged
                            and len(staged.phases) == len(phases))
    result.staged = staged

    if last.beam_forces:
        result.n_beam_elements = len(last.beam_forces)
        result.beam_forces = last.beam_forces
        result.max_beam_moment_kNm_per_m = last.max_beam_moment_kNm_per_m
        result.max_beam_shear_kN_per_m = last.max_beam_shear_kN_per_m

    def _strut_rows(pr):
        rows = []
        for s in pr.spring_forces or []:
            i = s['spring'] - n_ties
            if i < 0:
                continue                    # a wall-to-soil tie
            rows.append({
                'depth_m': float(-nodes[s['node_id'], 1]),
                'stiffness_kN_per_m': float(strut_defs[i][1]),
                'force_kN_per_m': float(s['force_x_kN_per_m']),
                'node_id': int(s['node_id']),
                'installed_in_stage': int(s['placed_in_phase']),
            })
        return rows

    if strut_defs and _strut_rows(last):
        result.strut_forces = _strut_rows(last)

    # One row per excavation stage
    wall_y = nodes[wall_nodes, 1]
    stage_rows = []
    env_m = env_v = None
    for pr in staged.phases[1:]:
        k = pr.phase_index - 1
        us = np.asarray(pr.displacements_since_reference, dtype=float)
        wux = us[2 * np.asarray(wall_nodes)]
        i_max = int(np.argmax(np.abs(wux)))
        row = {
            'stage': pr.phase_index,
            'name': pr.phase_name,
            'excavation_depth_m': float(lift_depths[k]),
            'converged': bool(pr.converged),
            'struts_installed_m': [float(strut_defs[i][0]) for i in
                                   install_stage if install_stage[i] == k + 1],
            'wall_head_ux_m': float(wux[0]),
            'max_wall_ux_m': float(wux[i_max]),
            'max_wall_ux_depth_m': float(-wall_y[i_max]),
            'max_beam_moment_kNm_per_m': float(pr.max_beam_moment_kNm_per_m),
            'max_beam_shear_kN_per_m': float(pr.max_beam_shear_kN_per_m),
            'strut_forces_kN_per_m': [r['force_kN_per_m']
                                      for r in _strut_rows(pr)],
            'settled_steps': len(pr.settled_steps or []),
        }
        stage_rows.append(row)
        if env_m is None or row['max_beam_moment_kNm_per_m'] > env_m[0]:
            env_m = (row['max_beam_moment_kNm_per_m'], row['name'])
        if env_v is None or row['max_beam_shear_kN_per_m'] > env_v[0]:
            env_v = (row['max_beam_shear_kN_per_m'], row['name'])
    result.excavation_stages = stage_rows
    if env_m is not None:
        result.wall_envelope = {
            'max_beam_moment_kNm_per_m': env_m[0], 'moment_stage': env_m[1],
            'max_beam_shear_kN_per_m': env_v[0], 'shear_stage': env_v[1],
        }

    k0_txt = "; ".join(
        f"{sl.get('name', f'layer {i + 1}')}: K0 = {_layer_k0(sl)[0]:.3f} "
        f"({_layer_k0(sl)[1]})" for i, sl in enumerate(soil_layers))
    result.notes = [
        "Staged excavation: in-situ K0 stresses with no displacement, then "
        f"{len(lift_depths)} lift(s) to "
        + ", ".join(f"{d:.2f}" for d in lift_depths) + " m; each lift's "
        "elements are removed and the forces they carried released onto "
        "the remaining mesh. The wall is wished in place at the first lift "
        "and each strut installed once the excavation reached its level, "
        "both stress-free.",
        f"In-situ stresses: {k0_txt}.",
        "Displacements are measured from the end of the in-situ stage; "
        "x is positive toward the excavation (u_x > 0 = the wall moves "
        "toward the excavation).",
        f"Half of a symmetric excavation {width:g} m wide: the right "
        f"boundary at x = {x_right:g} m is its centre line (rollers). "
        "The wall shares nodes with the soil on both sides (no interface "
        "elements): fully rough, and the soil is not separated by it.",
        "Strut force = stiffness x movement toward the excavation since "
        "installation (positive = the strut holds the wall back); no "
        "preload.",
    ]
    settled = [(p.phase_name, s) for p in staged.phases
               for s in (p.settled_steps or [])]
    if settled:
        worst = max(s['residual'] for _, s in settled)
        result.notes.append(
            f"{len(settled)} load step(s) reached the iteration ceiling with "
            "the displacements standing still and were accepted (largest "
            f"residual ratio {worst:.1e}; tolerance {tol:g}): Gauss points "
            "at the Mohr-Coulomb apex switching return region stall the "
            "residual without moving the mesh.")
    warnings = []
    if not result.converged:
        bad = next((p for p in staged.phases if not p.converged), None)
        warnings.append(
            f"Stage '{bad.phase_name if bad else phases[len(staged.phases)].name}' "
            f"did not converge within {max_iter} iterations a step; the "
            "results are those of the last stage reached. Try more n_steps "
            "or a higher max_iter, or the excavation may not be stable.")
    if pp is not None:
        exc_nodes = np.where((nodes[:, 0] > 1e-9)
                             & (np.abs(nodes[:, 1] + depth) < 1e-9))[0]
        if len(exc_nodes) and float(np.max(pp[exc_nodes])) > 1e-6:
            warnings.append(
                "The groundwater table is above the excavation level: the "
                "excavation is modelled kept full of water to the table "
                "(water pressure on its faces, no dewatering, no seepage), "
                "so the water pressures on the two sides of the wall "
                "balance. A dewatered excavation also carries the net "
                "water pressure; these wall forces do not include it.")
    plastic = any(float(sl.get('c', 0) or 0) > 0
                  or float(sl.get('phi', 0) or 0) > 0
                  or sl.get('model') == 'hs' for sl in soil_layers)
    if element_type != 't6' and plastic:
        warnings.append(
            "T3 (constant-strain) elements lock in plastic flow (DESIGN.md, "
            "Element Types): on the cantilever benchmark (VALIDATION.md "
            "section 9) T3 meshes gave a quarter (30 x 15) to a half "
            "(60 x 30) of the T6 wall moment, with a retained-side thrust "
            "below the active value. Use element_type='t6' for wall "
            "forces.")
    phi_max = max(float(sl.get('phi', 0) or 0) for sl in soil_layers)
    passive = (wall_depth - depth) * np.tan(np.radians(45 + phi_max / 2))
    if x_right < passive:
        result.notes.append(
            f"The excavation's half-width ({x_right:.1f} m) is less than "
            f"the Rankine passive zone in front of the wall "
            f"({passive:.1f} m): the centre line limits it, as it does in a "
            "symmetric excavation this wide. For a one-sided cut give a "
            "larger width.")
    result.warnings = warnings
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
    active_spring_ids : list of int, optional — indices into
        ``analyze_staged(springs=...)`` active in this phase (struts,
        anchors, props as linear springs). None means no springs.
    """
    name: str = "Phase"
    active_soil_groups: List[str] = field(default_factory=list)
    active_beam_ids: Optional[List[int]] = None
    surface_loads: Optional[List] = None
    gwt: Any = None
    n_steps: int = 5
    reset_displacements: bool = False
    active_spring_ids: Optional[List[int]] = None


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


def _normalize_springs(springs, n_nodes):
    """``springs`` -> list of (node, kx, ky, node_b). Each spring is a
    dict ({'node' or 'node_id', 'kx' or 'stiffness', 'ky', 'node_b'}) or a
    tuple (node, kx[, ky[, node_b]]), stiffness in kN/m per m run; node_b
    None = a spring to ground, else a tie between the two nodes."""
    out = []
    for j, s in enumerate(springs or []):
        if isinstance(s, dict):
            node = s.get('node', s.get('node_id'))
            kx = s.get('kx', s.get('stiffness', 0.0))
            ky = s.get('ky', 0.0)
            node_b = s.get('node_b')
        else:
            node, kx = s[0], s[1]
            ky = s[2] if len(s) > 2 else 0.0
            node_b = s[3] if len(s) > 3 else None
        for nd in (node, node_b):
            if nd is not None and not 0 <= int(nd) < n_nodes:
                raise ValueError(f"springs[{j}]: node {nd!r} is not a mesh "
                                 f"node")
        if node is None:
            raise ValueError(f"springs[{j}]: no node")
        out.append((int(node), float(kx or 0.0), float(ky or 0.0),
                    None if node_b is None else int(node_b)))
    return out


def analyze_staged(nodes, elements, material_props, gamma, bc_nodes,
                   element_groups, phases, beam_elements=None,
                   t=1.0, max_iter=100, tol=1e-5, gamma_w=9.81,
                   reset_displacements_after=None, springs=None,
                   initial_stresses=None, water_on_free_faces=True,
                   settled_ratio=None, settled_residual_cap=1e-2):
    """Staged construction analysis.

    Solves a sequence of construction phases. Each phase activates a
    subset of soil element groups and (optionally) beam elements and
    springs. Displacements, stresses, and strains carry forward
    cumulatively (per Gauss point). An element activated in a phase is
    placed stress-free on the deformed mesh: its strain counts from
    activation. A beam or spring activated in a phase is placed the same
    way (since 2026-10-14, GM-61): it carries only the movement after its
    placement. Before, a beam left out of a phase (``active_beam_ids=None``)
    still had its stiffness in the solve and, once reported, its forces
    counted every movement since the start of the analysis.

    Removing elements releases the forces they carried onto the rest of the
    mesh: a phase's out-of-balance load runs from the internal force of the
    state it starts from (on this phase's active elements) to this phase's
    load, so the excavated elements' stresses drop out and their weight
    goes with them.

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
    springs : list, optional — linear springs (struts, anchors, props, or
        ties between two nodes): dicts ``{'node': id, 'kx': kN/m/m,
        'ky': kN/m/m, 'node_b': id or None}`` (also ``node_id`` /
        ``stiffness``) or tuples ``(node, kx[, ky[, node_b]])``; without
        ``node_b`` the spring goes to ground. A phase switches them on with
        ``active_spring_ids``. Each phase reports ``spring_forces``: k
        times the stretch since placement (positive when the node moved in
        +x / +y relative to ground or to node_b; the spring pulls back).
    initial_stresses : array, optional — the stress state at the start of
        the first phase (effective, tension positive), per Gauss point
        ``(n_elem, n_gp, 4)`` as [sxx, syy, szz, txy], or per element
        ``(n_elem, 4)``. With it the first phase starts from these stresses
        and zero displacement (a K0 procedure: generate them in equilibrium
        with the first phase's load and the first phase only settles any
        out-of-balance). Elements not active in the first phase get none.
    water_on_free_faces : bool — in a phase with a groundwater table,
        water stands against every free face of the active mesh where the
        pore pressure is positive (``porewater.standing_water_forces``):
        an excavation below the table is kept full of water to it. Default
        True (since 2026-10-14). False reproduces the earlier load, in
        which such a face carried the pore pressure outward unbalanced.
    settled_ratio, settled_residual_cap : accept a load step that reaches
        ``max_iter`` with its displacements standing still (the largest
        nodal change over the second half of its iterations below
        ``settled_ratio`` of the first half) and its residual ratio below
        ``settled_residual_cap`` or still falling (``solver.run_nl``; the
        stall of Gauss points switching at the Mohr-Coulomb apex). Default
        None: off, every step must meet ``tol``. Accepted steps are listed
        on each phase (``settled_steps``).

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
    spring_list = _normalize_springs(springs, n_nodes_count)

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

    def _gp():
        if 'gp' not in gp_cache:
            from fem2d.solver import _gp_precompute
            gp_cache['gp'] = _gp_precompute(nodes, elements, t)
        return gp_cache['gp']

    def _kinematic_strain(u_vec):
        """Strain B u at every Gauss point (n_e, n_gp, 3)."""
        gp = _gp()
        return np.einsum('egki,ei->egk', gp['B'], u_vec[gp['dofs']])

    if initial_stresses is not None:
        ngp = _gp()['n_gp']
        s0 = np.asarray(initial_stresses, dtype=float)
        if s0.ndim == 2 and s0.shape == (n_elem, 4):
            s0 = np.repeat(s0[:, None, :], ngp, axis=1)
        if s0.shape != (n_elem, ngp, 4):
            raise ValueError(
                f"initial_stresses has shape {s0.shape}; expected "
                f"({n_elem}, {ngp}, 4) per Gauss point or ({n_elem}, 4) per "
                f"element, as [sxx, syy, szz, txy] (tension positive)")
        sig_gp = s0.copy()
        eps_gp = np.zeros((n_elem, ngp, 3))
        sigma = sig_gp[:, :, [0, 1, 3]].mean(axis=1)

    # Members (beams, springs) placed stress-free: the displacement of their
    # DOFs at placement, kept while they stay active (GM-61).
    beam_placed = {}     # beam index -> (dofs, u at placement)
    spring_placed = {}   # spring index -> (ux, uy) at placement
    beam_K = {}

    def _beam_dofs_K(idx):
        if idx not in beam_K:
            from fem2d.elements import beam2d_stiffness
            bm = beam_elements[idx]
            coords_ij = np.array([nodes[bm.node_i], nodes[bm.node_j]])
            K_e, _, _ = beam2d_stiffness(coords_ij, bm.EA, bm.EI)
            beam_K[idx] = (beam_element_dofs(bm.node_i, bm.node_j,
                                             rotation_dof_map), K_e)
        return beam_K[idx]

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

        # 2. Compute active beams and springs (None = none in this phase;
        #    before 2026-10-14 None reached the solver as "all beams").
        active_bms = set()
        if phase.active_beam_ids is not None and beam_elements:
            active_bms = {int(i) for i in phase.active_beam_ids
                          if 0 <= int(i) < len(beam_elements)}
        active_spr = set()
        if phase.active_spring_ids is not None:
            active_spr = {int(i) for i in phase.active_spring_ids
                          if 0 <= int(i) < len(spring_list)}

        # 3. Compute pore pressures if gwt provided
        pp = None
        water = None
        if phase.gwt is not None:
            from fem2d.porewater import (compute_pore_pressures,
                                         standing_water_forces)
            pp = compute_pore_pressures(nodes, phase.gwt, gamma_w)
            if water_on_free_faces and active_elems:
                water = standing_water_forces(nodes, elements, pp,
                                              active_elems, t)

        # 4. Elements activated in this phase are placed stress-free on the
        #    deformed mesh: zero stress, strain = B u now (so only movement
        #    after placement strains them), fresh HS state. Without this a
        #    fill's strain counted the movement its shared nodes made before
        #    it existed. (reset_displacements is reporting only — step 9.)
        #    Elements of the first phase keep any initial_stresses.
        newly = sorted(active_elems - prev_active)
        if pi == 0 and initial_stresses is not None:
            idle = sorted(set(range(n_elem)) - active_elems)
            sig_gp[idle] = 0.0
            newly = []
        if newly and sig_gp is not None:
            sig_gp = np.array(sig_gp, dtype=float)
            eps_gp = np.array(eps_gp, dtype=float)
            sig_gp[newly] = 0.0
            eps_gp[newly] = _kinematic_strain(u)[newly]
            for e in newly:
                elem_state[e] = None
        prev_active = set(active_elems)

        # 4b. Beams and springs placed in this phase start from the current
        #     displacement; ones no longer active are forgotten.
        for idx in list(beam_placed):
            if idx not in active_bms:
                del beam_placed[idx]
        for idx in sorted(active_bms - set(beam_placed)):
            dofs, _ = _beam_dofs_K(idx)
            beam_placed[idx] = (dofs, u[dofs].copy(), pi)
        for j in list(spring_placed):
            if j not in active_spr:
                del spring_placed[j]
        def _spring_stretch(j):
            node, _, _, node_b = spring_list[j]
            dx, dy = float(u[2 * node]), float(u[2 * node + 1])
            if node_b is not None:
                dx -= float(u[2 * node_b])
                dy -= float(u[2 * node_b + 1])
            return dx, dy

        for j in sorted(active_spr - set(spring_placed)):
            spring_placed[j] = _spring_stretch(j) + (pi,)
        lin_offset = None
        if beam_placed or spring_placed:
            lin_offset = np.zeros(n_dof_total)
            for idx, (dofs, u0, _) in beam_placed.items():
                lin_offset[dofs] += beam_K[idx][1] @ u0
            for j, (u0x, u0y, _) in spring_placed.items():
                node, kx, ky, node_b = spring_list[j]
                lin_offset[2 * node] += kx * u0x
                lin_offset[2 * node + 1] += ky * u0y
                if node_b is not None:
                    lin_offset[2 * node_b] -= kx * u0x
                    lin_offset[2 * node_b + 1] -= ky * u0y
        phase_springs = [spring_list[j] for j in sorted(active_spr)]

        # 5. Handle empty active elements gracefully
        if len(active_elems) == 0:
            pr = PhaseResult(
                phase_name=phase.name,
                phase_index=pi,
                n_active_elements=0,
                n_active_beams=len(active_bms),
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
            strut_springs=phase_springs or None,
            return_state=True,
            return_gp=True,
            lin_offset=lin_offset,
            extra_forces=water,
            settled_ratio=settled_ratio,
            settled_residual_cap=settled_residual_cap,
            return_info=True,
        )

        (converged, u_new, sigma_new, strain_new, state_new,
         sig_gp, eps_gp, run_info) = result

        # 7. Update cumulative state
        u = u_new
        sigma = sigma_new
        strain = strain_new
        elem_state = state_new

        # 8. Extract beam forces for active beams (from the movement since
        #    each beam's placement)
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
                bdofs, u0, _ = beam_placed[idx]
                u_beam = u[bdofs] - u0
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
        if spring_placed:
            sf = []
            for j in sorted(spring_placed):
                node, kx, ky, node_b = spring_list[j]
                u0x, u0y, placed_in = spring_placed[j]
                sx, sy = _spring_stretch(j)
                sf.append({
                    'spring': j, 'node_id': int(node),
                    'node_b': None if node_b is None else int(node_b),
                    'x_m': float(nodes[node, 0]),
                    'y_m': float(nodes[node, 1]),
                    'kx_kN_per_m': kx, 'ky_kN_per_m': ky,
                    'force_x_kN_per_m': kx * (sx - u0x),
                    'force_y_kN_per_m': ky * (sy - u0y),
                    'placed_in_phase': int(placed_in),
                })
            pr.spring_forces = sf

        pr.settled_steps = list(run_info['settled'])
        pr.final_residual = float(run_info['final_residual'])

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
        "deformed mesh; its nodes' movement counts from placement. Beams "
        "and springs likewise carry only the movement after their "
        "placement.",
    ]
    if water_on_free_faces and any(p.gwt is not None for p in phases):
        notes.append(
            "Water stands against every free face where the pore pressure "
            "is positive (an excavation below the water table is kept full "
            "of water to it; no dewatering or seepage is modelled).")

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
