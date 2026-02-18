"""
1D effective-stress site response analysis using OpenSees.

Builds a 1D soil column with SSPquadUP elements, applies earthquake
ground motion through a Lysmer-Kuhlmeyer viscous dashpot at the base,
and extracts surface motion, depth profiles, and response spectra.

Materials:
  Sand: PressureDependMultiYield02
  Clay: PressureIndependMultiYield

References:
    Lysmer, J. & Kuhlemeyer, R.L. (1969). "Finite Dynamic Model for
    Infinite Media." J. Eng. Mech. Div., ASCE, 95(4), 859-877.

    Yang, Z., Elgamal, A., & Parra, E. (2003). "Computational Model
    for Cyclic Mobility and Associated Shear Deformation." J. Geotech.
    Geoenviron. Eng., 129(12), 1119-1127.
"""

import math

import numpy as np

from opensees_agent.results import SiteResponseResult


# ===========================================================================
# Input validation
# ===========================================================================

def _validate_site_response_inputs(layers, gwt_depth, bedrock_Vs,
                                   bedrock_density, damping,
                                   scale_factor, n_elem_per_layer):
    """Validate all site response inputs, raising ValueError for bad values."""
    # -- layers --
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list of dicts")

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{i}] must be a dict")

        for key in ("thickness", "Vs", "density", "material_type"):
            if key not in layer:
                raise ValueError(
                    f"layers[{i}] missing required key '{key}'")

        if layer["thickness"] <= 0:
            raise ValueError(
                f"layers[{i}]['thickness'] must be positive, "
                f"got {layer['thickness']}")
        if layer["Vs"] <= 0:
            raise ValueError(
                f"layers[{i}]['Vs'] must be positive, got {layer['Vs']}")
        if layer["density"] <= 0:
            raise ValueError(
                f"layers[{i}]['density'] must be positive, "
                f"got {layer['density']}")

        mat = layer["material_type"].lower().strip()
        if mat not in ("sand", "clay"):
            raise ValueError(
                f"layers[{i}]['material_type'] must be 'sand' or 'clay', "
                f"got '{layer['material_type']}'")

        if mat == "sand":
            if "phi" not in layer:
                raise ValueError(
                    f"layers[{i}] (sand) requires 'phi' (friction angle)")
            if layer["phi"] <= 0:
                raise ValueError(
                    f"layers[{i}]['phi'] must be positive, got {layer['phi']}")
        if mat == "clay":
            if "su" not in layer:
                raise ValueError(
                    f"layers[{i}] (clay) requires 'su' "
                    f"(undrained shear strength)")
            if layer["su"] <= 0:
                raise ValueError(
                    f"layers[{i}]['su'] must be positive, got {layer['su']}")

    # -- scalar params --
    if gwt_depth < 0:
        raise ValueError(f"gwt_depth must be >= 0, got {gwt_depth}")
    if bedrock_Vs <= 0:
        raise ValueError(f"bedrock_Vs must be positive, got {bedrock_Vs}")
    if bedrock_density <= 0:
        raise ValueError(
            f"bedrock_density must be positive, got {bedrock_density}")
    if not (0 <= damping < 1):
        raise ValueError(f"damping must be in [0, 1), got {damping}")
    if scale_factor <= 0:
        raise ValueError(
            f"scale_factor must be positive, got {scale_factor}")
    if n_elem_per_layer < 1:
        raise ValueError(
            f"n_elem_per_layer must be >= 1, got {n_elem_per_layer}")


# ===========================================================================
# Public API
# ===========================================================================

def analyze_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    gwt_depth=0.0,
    bedrock_Vs=760.0,
    bedrock_density=2.4,
    damping=0.02,
    scale_factor=1.0,
    n_elem_per_layer=4,
):
    """Run 1D effective-stress site response analysis.

    Builds a layered soil column in OpenSees using SSPquadUP elements
    with PressureDependMultiYield02 (sand) and PressureIndependMultiYield
    (clay) constitutive models. Input motion is applied at the base
    through a Lysmer-Kuhlemeyer viscous dashpot.

    Parameters
    ----------
    layers : list of dict
        Soil layers from top to bottom. Each dict must contain:
        ``thickness`` (m), ``Vs`` (m/s), ``density`` (Mg/m3),
        ``material_type`` ('sand' or 'clay').
        Sand requires: ``phi`` (degrees). Optional: ``K0``.
        Clay requires: ``su`` (kPa).
        Optional for all: ``n_surf`` (default 20), ``e_init``.
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    gwt_depth : float
        Groundwater table depth from surface (m). Default 0.0 (at surface).
    bedrock_Vs : float
        Bedrock shear wave velocity (m/s). Default 760.
    bedrock_density : float
        Bedrock mass density (Mg/m3). Default 2.4.
    damping : float
        Target Rayleigh damping ratio. Default 0.02.
    scale_factor : float
        Scale factor applied to input acceleration. Default 1.0.
    n_elem_per_layer : int
        Number of elements per soil layer. Default 4.

    Returns
    -------
    SiteResponseResult
        Analysis results with surface motion, spectra, and depth profiles.

    Raises
    ------
    ValueError
        For invalid input parameters.
    ImportError
        If openseespy is not installed.
    """
    _validate_site_response_inputs(
        layers, gwt_depth, bedrock_Vs, bedrock_density,
        damping, scale_factor, n_elem_per_layer)

    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_motion = validate_motion_input(motion, accel_history, dt)

    motion_name = motion if motion else "custom"

    from opensees_agent.opensees_utils import fresh_model
    ops = fresh_model(ndm=2, ndf=3)

    try:
        return _run_site_response_model(
            ops, layers, accel_g, dt_motion,
            gwt_depth, bedrock_Vs, bedrock_density,
            damping, scale_factor, n_elem_per_layer,
            motion_name)
    finally:
        ops.wipe()


# ===========================================================================
# Model building & analysis
# ===========================================================================

def _run_site_response_model(ops, layers, accel_g, dt_motion,
                             gwt_depth, bedrock_Vs, bedrock_density,
                             damping, scale_factor, n_elem_per_layer,
                             motion_name):
    """Build model, run gravity + dynamic, extract results."""
    g_accel = 9.81  # m/s^2
    col_width = 1.0  # m (plane-strain unit width)
    total_depth = sum(L["thickness"] for L in layers)

    # ------------------------------------------------------------------
    # 1. Mesh generation: compute element layout bottom-to-top
    # ------------------------------------------------------------------
    # Layers are given top-to-bottom; we reverse for building bottom-up
    layer_elems = []  # (layer_dict, elem_height, n_elems) per layer
    for L in reversed(layers):
        n_e = n_elem_per_layer
        h = L["thickness"] / n_e
        layer_elems.append((L, h, n_e))

    n_elem_total = n_elem_per_layer * len(layers)
    n_nodes_col = n_elem_total + 1  # nodes per column (left or right)

    # Node tag helpers (1-indexed)
    def left_node(i):
        return i + 1

    def right_node(i):
        return n_nodes_col + i + 1

    # Compute node elevations (from base = 0 to surface = total_depth)
    elevations = [0.0]
    for L_dict, h_elem, n_e in layer_elems:
        for _ in range(n_e):
            elevations.append(elevations[-1] + h_elem)

    # ------------------------------------------------------------------
    # 2. Create nodes
    # ------------------------------------------------------------------
    for i in range(n_nodes_col):
        ops.node(left_node(i), 0.0, elevations[i])
        ops.node(right_node(i), col_width, elevations[i])

    # ------------------------------------------------------------------
    # 3. Boundary conditions
    # ------------------------------------------------------------------
    # Base nodes: fix vertical, free horizontal, free pore pressure
    ops.fix(left_node(0), 0, 1, 0)
    ops.fix(right_node(0), 0, 1, 0)

    # Tie left-right node pairs (periodic BCs for 1D column)
    for i in range(n_nodes_col):
        if i == 0:
            # Base pair: tie x only (y already fixed)
            ops.equalDOF(left_node(0), right_node(0), 1)
        else:
            ops.equalDOF(left_node(i), right_node(i), 1, 2)

    # Surface nodes: fix pore pressure (drained surface)
    top_idx = n_nodes_col - 1
    ops.fix(left_node(top_idx), 0, 0, 1)
    ops.fix(right_node(top_idx), 0, 0, 1)

    # Nodes above GWT: fix pore pressure (no pore water)
    for i in range(n_nodes_col):
        depth_from_surface = total_depth - elevations[i]
        if depth_from_surface < gwt_depth and i != top_idx:
            ops.fix(left_node(i), 0, 0, 1)
            ops.fix(right_node(i), 0, 0, 1)

    # ------------------------------------------------------------------
    # 4. Create materials (one per layer)
    # ------------------------------------------------------------------
    mat_tags = []
    perms = []  # permeability per layer
    e_inits = []  # void ratio per layer
    mat_tag = 1
    depth_bottom = 0.0  # building bottom-up
    for L_dict, h_elem, n_e in layer_elems:
        thickness = L_dict["thickness"]
        depth_top = depth_bottom
        depth_mid = depth_bottom + thickness / 2.0
        depth_bottom_layer = depth_bottom + thickness

        # Depth from surface for stress calculation
        depth_from_surface_mid = total_depth - depth_mid

        mat_type = L_dict["material_type"].lower().strip()

        if mat_type == "sand":
            perm = _create_sand_material(
                ops, mat_tag, L_dict, depth_from_surface_mid, gwt_depth)
        else:
            perm = _create_clay_material(
                ops, mat_tag, L_dict, depth_from_surface_mid, gwt_depth)

        e_init = L_dict.get("e_init", 0.65 if mat_type == "sand" else 0.85)

        mat_tags.append(mat_tag)
        perms.append(perm)
        e_inits.append(e_init)
        mat_tag += 1
        depth_bottom = depth_bottom_layer

    # ------------------------------------------------------------------
    # 5. Create elements (SSPquadUP)
    # ------------------------------------------------------------------
    elem_tag = 1
    elem_mat_idx = []  # which layer index each element belongs to
    ei = 0  # element counter within current layer
    layer_idx = 0

    for layer_idx_local, (L_dict, h_elem, n_e) in enumerate(layer_elems):
        for _ in range(n_e):
            n1 = left_node(ei)
            n2 = right_node(ei)
            n3 = right_node(ei + 1)
            n4 = left_node(ei + 1)

            perm_val = perms[layer_idx_local]
            e_val = e_inits[layer_idx_local]

            ops.element('SSPquadUP', elem_tag,
                        n1, n2, n3, n4,
                        mat_tags[layer_idx_local],
                        1.0,       # thickness (plane strain)
                        2.2e6,     # fluid bulk modulus (water, kPa)
                        1.0,       # fluid mass density (Mg/m3)
                        perm_val,  # horizontal permeability
                        perm_val,  # vertical permeability
                        e_val,     # void ratio
                        1.0e-5)    # alpha (numerical damping)

            elem_mat_idx.append(layer_idx_local)
            elem_tag += 1
            ei += 1

    # ------------------------------------------------------------------
    # 6. Lysmer-Kuhlemeyer dashpot at base
    # ------------------------------------------------------------------
    total_nodes = 2 * n_nodes_col
    dashpot_node = total_nodes + 1
    ops.node(dashpot_node, 0.0, 0.0)
    ops.fix(dashpot_node, 1, 1, 1)

    # Dashpot coefficient: c = rho_bedrock * Vs_bedrock * area
    c_dashpot = bedrock_density * bedrock_Vs * col_width  # kN*s/m

    dashpot_mat = mat_tag + 100
    ops.uniaxialMaterial('Viscous', dashpot_mat, c_dashpot, 1.0)

    dashpot_elem = elem_tag + 100
    ops.element('zeroLength', dashpot_elem,
                dashpot_node, left_node(0),
                '-mat', dashpot_mat, '-dir', 1)

    # ------------------------------------------------------------------
    # 7. Gravity analysis (elastic stage)
    # ------------------------------------------------------------------
    # Body forces: gravity applied through element self-weight
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # Apply gravity body forces as nodal loads
    # Each element's weight is distributed to its 4 nodes
    for e_idx in range(n_elem_total):
        li = elem_mat_idx[e_idx]
        rho = layer_elems[li][0]["density"]
        h_e = layer_elems[li][1]
        # Weight per node = rho * g * h * width / 4 (4 nodes per element)
        w_node = -rho * g_accel * h_e * col_width / 4.0

        n1 = left_node(e_idx)
        n2 = right_node(e_idx)
        n3 = right_node(e_idx + 1)
        n4 = left_node(e_idx + 1)

        ops.load(n1, 0.0, w_node, 0.0)
        ops.load(n2, 0.0, w_node, 0.0)
        ops.load(n3, 0.0, w_node, 0.0)
        ops.load(n4, 0.0, w_node, 0.0)

    # Set all materials to elastic stage (stage 0)
    for mt in mat_tags:
        ops.updateMaterialStage('-material', mt, '-stage', 0)

    ops.constraints('Penalty', 1.0e14, 1.0e14)
    ops.test('NormDispIncr', 1.0e-4, 35, 0)
    ops.algorithm('KrylovNewton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    ops.analyze(10, 500.0)

    # ------------------------------------------------------------------
    # 8. Switch to elastoplastic (stage 1)
    # ------------------------------------------------------------------
    for mt in mat_tags:
        ops.updateMaterialStage('-material', mt, '-stage', 1)

    ops.analyze(10, 500.0)

    # ------------------------------------------------------------------
    # 9. Record initial stresses for ru computation
    # ------------------------------------------------------------------
    initial_sigma_v = np.zeros(n_elem_total)
    for e in range(n_elem_total):
        stress = ops.eleResponse(e + 1, 'stress')
        # stress[1] = sigma_yy (negative = compression in OpenSees)
        initial_sigma_v[e] = -stress[1] if len(stress) > 1 else 1.0

    # ------------------------------------------------------------------
    # 10. Set up dynamic loading
    # ------------------------------------------------------------------
    accel_scaled = accel_g * scale_factor

    # Integrate acceleration to velocity
    velocity = np.cumsum(accel_scaled * g_accel * dt_motion)

    # Force at base: factor 2 for outcrop motion assumption
    force_vals = 2.0 * c_dashpot * velocity

    # Path time series for the force
    ts_dyn = 100
    pat_dyn = 100

    ops.timeSeries('Path', ts_dyn, '-dt', dt_motion,
                   '-values', *force_vals.tolist(), '-factor', 1.0)
    ops.pattern('Plain', pat_dyn, ts_dyn)
    ops.load(left_node(0), 1.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 11. Rayleigh damping
    # ------------------------------------------------------------------
    # Average Vs (travel-time weighted)
    Vs_avg = total_depth / sum(
        L["thickness"] / L["Vs"] for L in layers)
    T1 = 4.0 * total_depth / Vs_avg  # fundamental period
    f1 = 1.0 / T1
    f2 = 5.0 * f1  # target 5th mode approx
    omega1 = 2.0 * math.pi * f1
    omega2 = 2.0 * math.pi * f2
    a0_ray = 2.0 * damping * omega1 * omega2 / (omega1 + omega2)
    a1_ray = 2.0 * damping / (omega1 + omega2)

    ops.rayleigh(a0_ray, a1_ray, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 12. Dynamic analysis
    # ------------------------------------------------------------------
    ops.wipeAnalysis()
    ops.constraints('Penalty', 1.0e14, 1.0e14)
    ops.test('NormDispIncr', 1.0e-3, 35, 0)
    ops.algorithm('KrylovNewton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    n_steps = len(accel_g)
    surface_accel = np.zeros(n_steps)
    max_accel = np.zeros(n_elem_total)
    max_strain = np.zeros(n_elem_total)
    max_ru = np.zeros(n_elem_total)

    surface_node = left_node(top_idx)

    for step in range(n_steps):
        ok = ops.analyze(1, dt_motion)
        if ok != 0:
            # Fallback: ModifiedNewton
            ops.algorithm('ModifiedNewton')
            ok = ops.analyze(1, dt_motion)
            ops.algorithm('KrylovNewton')
        if ok != 0:
            # Fallback: substeps
            for _sub in range(10):
                ok = ops.analyze(1, dt_motion / 10.0)
                if ok != 0:
                    break

        # Surface acceleration
        a_surf = ops.nodeAccel(surface_node, 1)
        surface_accel[step] = a_surf / g_accel

        # Element profiles
        for e in range(n_elem_total):
            e_tag = e + 1
            stress = ops.eleResponse(e_tag, 'stress')
            strain = ops.eleResponse(e_tag, 'strain')

            # Max acceleration (average of top and bottom node accels)
            a_bot = abs(ops.nodeAccel(left_node(e), 1))
            a_top = abs(ops.nodeAccel(left_node(e + 1), 1))
            a_avg = (a_bot + a_top) / 2.0 / g_accel
            if a_avg > max_accel[e]:
                max_accel[e] = a_avg

            # Max shear strain
            if len(strain) > 2:
                gamma = abs(strain[2])  # engineering shear strain
                if gamma > max_strain[e]:
                    max_strain[e] = gamma

            # Pore pressure ratio
            if len(stress) > 1 and initial_sigma_v[e] > 1.0:
                sigma_v_curr = -stress[1]
                ru = 1.0 - sigma_v_curr / initial_sigma_v[e]
                ru = max(0.0, min(ru, 1.0))
                if ru > max_ru[e]:
                    max_ru[e] = ru

    # ------------------------------------------------------------------
    # 13. Post-process
    # ------------------------------------------------------------------
    time_arr = np.arange(n_steps) * dt_motion
    pga_input = float(np.max(np.abs(accel_scaled)))
    pga_surface = float(np.max(np.abs(surface_accel)))
    amplification = pga_surface / pga_input if pga_input > 0 else 0.0

    # Element center depths (from surface)
    depths = np.zeros(n_elem_total)
    for e in range(n_elem_total):
        elev_center = (elevations[e] + elevations[e + 1]) / 2.0
        depths[e] = total_depth - elev_center

    # Response spectra
    from opensees_agent.opensees_utils import compute_response_spectrum
    periods = np.logspace(-2, 1, 100)
    _, Sa_surface = compute_response_spectrum(
        surface_accel, dt_motion, periods)
    _, Sa_input = compute_response_spectrum(
        accel_scaled, dt_motion, periods)

    return SiteResponseResult(
        total_depth_m=total_depth,
        n_layers=len(layers),
        motion_name=motion_name,
        pga_input_g=pga_input,
        pga_surface_g=pga_surface,
        amplification_factor=amplification,
        time=time_arr,
        surface_accel_g=surface_accel,
        depths=depths,
        max_strain_pct=max_strain * 100.0,
        max_accel_g=max_accel,
        max_pore_pressure_ratio=max_ru,
        periods=periods,
        Sa_surface_g=Sa_surface,
        Sa_input_g=Sa_input,
    )


# ===========================================================================
# Material creation helpers
# ===========================================================================

def _create_sand_material(ops, mat_tag, layer, depth_from_surface, gwt_depth):
    """Create PressureDependMultiYield02 material for a sand layer.

    Returns permeability (m/s) for use in element creation.
    """
    Vs = layer["Vs"]
    rho = layer["density"]
    phi = layer["phi"]
    K0 = layer.get("K0", 1.0 - math.sin(math.radians(phi)))
    n_surf = layer.get("n_surf", 20)
    e_init = layer.get("e_init", 0.65)

    # Gmax = rho * Vs^2 (kPa)
    Gmax = rho * Vs ** 2
    # Bulk modulus (drained nu ~ 0.3)
    nu_drain = 0.3
    Bmax = 2.0 * Gmax * (1.0 + nu_drain) / (3.0 * (1.0 - 2.0 * nu_drain))

    # Reference confining pressure at layer mid-depth
    gamma_w = 9.81
    depth_below_gwt = max(0.0, depth_from_surface - gwt_depth)
    sigma_v = rho * 9.81 * depth_from_surface - gamma_w * depth_below_gwt
    sigma_mean = max(sigma_v * (1.0 + 2.0 * K0) / 3.0, 5.0)

    gamma_max = 0.1  # 10% peak shear strain

    # Contraction/dilation parameters (conservative defaults)
    # c1, c3, c2 — contraction
    c1, c3, c2 = 0.067, 0.23, 0.0
    # d1, d3, d2 — dilation
    d1, d3, d2 = 0.06, 0.27, 0.0

    perm = 1.0e-4  # m/s (sand)

    ops.nDMaterial('PressureDependMultiYield02', mat_tag,
                   2,            # ndm
                   rho,          # mass density (Mg/m3)
                   Gmax,         # reference shear modulus (kPa)
                   Bmax,         # reference bulk modulus (kPa)
                   phi,          # friction angle (degrees)
                   gamma_max,    # peak shear strain
                   sigma_mean,   # reference confining pressure (kPa)
                   0.5,          # pressure dependence coefficient
                   n_surf,       # number of yield surfaces
                   c1, c3, c2,   # contraction params
                   d1, d3, d2,   # dilation params
                   0.0, 0.0, 0.0, 0.0, 0.0,  # liquefaction params
                   e_init)       # void ratio

    return perm


def _create_clay_material(ops, mat_tag, layer, depth_from_surface, gwt_depth):
    """Create PressureIndependMultiYield material for a clay layer.

    Returns permeability (m/s) for use in element creation.
    """
    Vs = layer["Vs"]
    rho = layer["density"]
    su = layer["su"]
    n_surf = layer.get("n_surf", 20)

    Gmax = rho * Vs ** 2
    # Nearly incompressible for undrained clay
    nu_undrained = 0.495
    Bmax = 2.0 * Gmax * (1.0 + nu_undrained) / (3.0 * (1.0 - 2.0 * nu_undrained))
    # Cap to prevent numerical issues
    Bmax = min(Bmax, Gmax * 100.0)

    gamma_max = 0.1  # 10%
    sigma_mean = max(rho * 9.81 * depth_from_surface * 0.67, 5.0)

    perm = 1.0e-7  # m/s (clay, effectively undrained)

    ops.nDMaterial('PressureIndependMultiYield', mat_tag,
                   2,            # ndm
                   rho,          # mass density
                   Gmax,         # reference shear modulus (kPa)
                   Bmax,         # reference bulk modulus (kPa)
                   su,           # cohesion / undrained shear strength (kPa)
                   gamma_max,    # peak shear strain
                   0.0,          # friction angle (0 for total stress)
                   sigma_mean,   # reference pressure (kPa)
                   0.0,          # pressure dependence coefficient
                   n_surf)       # number of yield surfaces

    return perm
