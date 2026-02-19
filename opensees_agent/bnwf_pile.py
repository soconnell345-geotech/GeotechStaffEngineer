"""
BNWF laterally-loaded pile analysis using OpenSees.

Builds a Beam on Nonlinear Winkler Foundation (BNWF) model:
  - Pile: elasticBeamColumn elements (2D beam, ndm=2, ndf=3)
  - Lateral springs: PySimple1 uniaxial material on zeroLength elements
  - Shaft springs: TzSimple1 on the same zero-length elements (vertical DOF)
  - Tip spring: QzSimple1 at the pile toe

Reuses existing p-y curve models from lateral_pile/py_curves.py to compute
pult and y50 for the PySimple1 materials.

References:
    - OpenSeesPy BNWF lateral pile example:
      https://openseespydoc.readthedocs.io/en/latest/src/pile.html
    - API RP2A-WSD (2000), Section 6.8 (p-y curves for sand)
    - Matlock, H. (1970). OTC 1204 (p-y curves for soft clay)
"""

import math

import numpy as np

from opensees_agent.results import BNWFPileResult


# ── Supported p-y model names → lateral_pile class mapping ──

_PY_MODEL_MAP = {
    "matlock": "SoftClayMatlock",
    "jeanjean": "SoftClayJeanjean",
    "stiff_clay_below_wt": "StiffClayBelowWT",
    "stiff_clay_above_wt": "StiffClayAboveWT",
    "api_sand": "SandAPI",
    "reese_sand": "SandReese",
    "weak_rock": "WeakRock",
    "liquefied_sand": "SandLiquefied",
}

# Which models are "clay" (soilType=1) vs "sand" (soilType=2) for PySimple1
_CLAY_MODELS = {"matlock", "jeanjean", "stiff_clay_below_wt", "stiff_clay_above_wt"}
_SAND_MODELS = {"api_sand", "reese_sand", "liquefied_sand"}


def _validate_bnwf_inputs(pile_length, pile_diameter, wall_thickness, E_pile,
                          layers, lateral_load, moment, axial_load,
                          head_condition, pile_above_ground, n_elem_per_meter):
    """Validate BNWF pile analysis inputs."""
    if pile_length <= 0:
        raise ValueError(f"pile_length must be positive, got {pile_length}")
    if pile_diameter <= 0:
        raise ValueError(f"pile_diameter must be positive, got {pile_diameter}")
    if wall_thickness < 0:
        raise ValueError(f"wall_thickness must be >= 0, got {wall_thickness}")
    if wall_thickness > 0 and wall_thickness >= pile_diameter / 2:
        raise ValueError(
            f"wall_thickness ({wall_thickness}) must be < pile_diameter/2 "
            f"({pile_diameter / 2})")
    if E_pile <= 0:
        raise ValueError(f"E_pile must be positive, got {E_pile}")
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list of dicts")
    if lateral_load == 0 and moment == 0:
        raise ValueError("At least one of lateral_load or moment must be non-zero")
    if head_condition not in ('free', 'fixed'):
        raise ValueError(
            f"head_condition must be 'free' or 'fixed', got '{head_condition}'")
    if pile_above_ground < 0:
        raise ValueError(
            f"pile_above_ground must be >= 0, got {pile_above_ground}")
    if n_elem_per_meter <= 0:
        raise ValueError(
            f"n_elem_per_meter must be positive, got {n_elem_per_meter}")

    # Validate each layer
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{i}] must be a dict, got {type(layer).__name__}")
        for key in ("top", "bottom", "py_model"):
            if key not in layer:
                raise ValueError(f"layers[{i}] missing required key '{key}'")
        if layer["top"] < 0:
            raise ValueError(f"layers[{i}]['top'] must be >= 0, got {layer['top']}")
        if layer["bottom"] <= layer["top"]:
            raise ValueError(
                f"layers[{i}]['bottom'] ({layer['bottom']}) must be > "
                f"'top' ({layer['top']})")
        model_name = layer["py_model"].lower().strip()
        if model_name not in _PY_MODEL_MAP:
            raise ValueError(
                f"layers[{i}]['py_model'] = '{layer['py_model']}' not recognized. "
                f"Available: {', '.join(sorted(_PY_MODEL_MAP.keys()))}")


def _build_py_model(layer_dict):
    """Instantiate a lateral_pile p-y model from a layer dict.

    Returns the p-y model object and the model name (lowercase).
    """
    from lateral_pile.py_curves import (
        SoftClayMatlock, SoftClayJeanjean, StiffClayBelowWT,
        StiffClayAboveWT, SandAPI, SandReese, WeakRock,
        SandLiquefied,
    )

    class_map = {
        "SoftClayMatlock": SoftClayMatlock,
        "SoftClayJeanjean": SoftClayJeanjean,
        "StiffClayBelowWT": StiffClayBelowWT,
        "StiffClayAboveWT": StiffClayAboveWT,
        "SandAPI": SandAPI,
        "SandReese": SandReese,
        "WeakRock": WeakRock,
        "SandLiquefied": SandLiquefied,
    }

    model_name = layer_dict["py_model"].lower().strip()
    cls_name = _PY_MODEL_MAP[model_name]
    cls = class_map[cls_name]

    # Build kwargs by excluding metadata keys
    skip_keys = {"top", "bottom", "py_model", "description"}
    kwargs = {k: v for k, v in layer_dict.items() if k not in skip_keys}

    # Map common aliases
    if "su" in kwargs and cls_name == "SoftClayMatlock":
        kwargs["c"] = kwargs.pop("su")
    if "su" in kwargs and cls_name in ("StiffClayBelowWT", "StiffClayAboveWT"):
        kwargs["c"] = kwargs.pop("su")

    return cls(**kwargs), model_name


def _get_py_params(py_model, z, b, elem_length):
    """Extract pult and y50 from a p-y model at depth z.

    Parameters
    ----------
    py_model : p-y model instance
        One of the lateral_pile p-y model classes.
    z : float
        Depth below ground surface (m).
    b : float
        Pile diameter (m).
    elem_length : float
        Tributary length for this spring (m).

    Returns
    -------
    pult : float
        Ultimate lateral resistance (kN), already multiplied by tributary length.
    y50 : float
        Displacement at 50% of pult (m).
    """
    # Generate p-y curve and find pult and y50
    y_arr, p_arr = py_model.get_py_curve(z, b, n_points=200)

    # pult = max of the curve (force per unit length) × tributary length
    p_max = float(np.max(p_arr))  # kN/m
    pult = p_max * elem_length    # kN (force)

    if pult <= 0:
        # At surface or zero resistance — use small value
        pult = 0.001 * elem_length
        y50 = 0.001 * b
        return pult, y50

    # y50 = displacement at 50% of p_max
    p_target = 0.5 * p_max
    # Find first crossing
    idx = np.searchsorted(p_arr, p_target)
    if idx == 0:
        y50 = y_arr[1] if len(y_arr) > 1 else 0.001 * b
    elif idx >= len(y_arr):
        y50 = y_arr[-1] * 0.5
    else:
        # Linear interpolation between bracketing points
        y_lo, y_hi = y_arr[idx - 1], y_arr[idx]
        p_lo, p_hi = p_arr[idx - 1], p_arr[idx]
        if p_hi > p_lo:
            frac = (p_target - p_lo) / (p_hi - p_lo)
            y50 = y_lo + frac * (y_hi - y_lo)
        else:
            y50 = y_lo

    y50 = max(y50, 1.0e-6)  # prevent zero
    return pult, y50


def _get_layer_at_depth(layers_parsed, z):
    """Return (py_model, model_name) for a given depth z."""
    for top, bottom, py_model, model_name in layers_parsed:
        if top <= z <= bottom:
            return py_model, model_name
    # If z is beyond last layer, use the last layer
    return layers_parsed[-1][2], layers_parsed[-1][3]


def analyze_bnwf_pile(
    pile_length,
    pile_diameter,
    wall_thickness,
    E_pile,
    layers,
    lateral_load=0.0,
    moment=0.0,
    axial_load=0.0,
    head_condition='free',
    pile_above_ground=0.0,
    n_elem_per_meter=5,
):
    """Run BNWF lateral pile analysis using OpenSees.

    Parameters
    ----------
    pile_length : float
        Embedded pile length (m).
    pile_diameter : float
        Outer diameter (m).
    wall_thickness : float
        Pipe pile wall thickness (m). Use 0 for solid circular section.
    E_pile : float
        Young's modulus of pile (kPa).
    layers : list of dict
        Soil layers. Each dict must have:
          - top (float): depth to top of layer (m, from ground surface)
          - bottom (float): depth to bottom (m)
          - py_model (str): one of 'matlock', 'jeanjean', 'stiff_clay_below_wt',
            'stiff_clay_above_wt', 'api_sand', 'reese_sand', 'weak_rock'
          - Additional parameters required by the p-y model (phi, gamma, k,
            su/c, eps50, Gmax, etc.)
    lateral_load : float
        Lateral force at pile head (kN). Default 0.
    moment : float
        Moment at pile head (kN-m). Default 0.
    axial_load : float
        Axial force (kN, compression positive). Default 0.
    head_condition : str
        'free' or 'fixed'. Default 'free'.
    pile_above_ground : float
        Free length above ground surface (m). Default 0.
    n_elem_per_meter : float
        Mesh density (elements per meter). Default 5.

    Returns
    -------
    BNWFPileResult

    Raises
    ------
    ValueError
        For invalid inputs.
    ImportError
        If openseespy is not installed.
    """
    _validate_bnwf_inputs(
        pile_length, pile_diameter, wall_thickness, E_pile,
        layers, lateral_load, moment, axial_load,
        head_condition, pile_above_ground, n_elem_per_meter)

    # Parse layers into (top, bottom, py_model_instance, model_name)
    layers_parsed = []
    for layer_dict in layers:
        py_model, model_name = _build_py_model(layer_dict)
        layers_parsed.append((
            layer_dict["top"], layer_dict["bottom"],
            py_model, model_name))

    # Sort by top depth
    layers_parsed.sort(key=lambda x: x[0])

    from opensees_agent.opensees_utils import fresh_model
    ops = fresh_model(ndm=2, ndf=3)

    try:
        return _build_and_solve(
            ops, pile_length, pile_diameter, wall_thickness, E_pile,
            layers_parsed, lateral_load, moment, axial_load,
            head_condition, pile_above_ground, n_elem_per_meter)
    finally:
        ops.wipe()


def _build_and_solve(
    ops, pile_length, pile_diameter, wall_thickness, E_pile,
    layers_parsed, lateral_load, moment, axial_load,
    head_condition, pile_above_ground, n_elem_per_meter,
):
    """Build the BNWF model, run static analysis, extract results."""
    b = pile_diameter

    # ── Section properties ──
    if wall_thickness > 0:
        # Hollow pipe pile
        r_out = b / 2.0
        r_in = r_out - wall_thickness
        A_pile = math.pi * (r_out ** 2 - r_in ** 2)
        I_pile = math.pi / 4.0 * (r_out ** 4 - r_in ** 4)
    else:
        # Solid circular
        A_pile = math.pi * (b / 2.0) ** 2
        I_pile = math.pi / 4.0 * (b / 2.0) ** 4

    EI = E_pile * I_pile  # kN-m^2
    EA = E_pile * A_pile  # kN

    # ── Mesh ──
    total_length = pile_above_ground + pile_length
    n_elem = max(int(round(total_length * n_elem_per_meter)), 10)
    elem_length = total_length / n_elem
    n_nodes = n_elem + 1

    # Node depths measured from pile head (top)
    # depth=0 is at pile head, depth=pile_above_ground is at ground surface
    node_depths = np.linspace(0, total_length, n_nodes)

    # ── Node numbering ──
    # Soil nodes (fixed): 1 .. n_nodes
    # Spring nodes (free): n_nodes+1 .. 2*n_nodes
    # Pile nodes: 2*n_nodes+1 .. 3*n_nodes
    soil_base = 0
    spring_base = n_nodes
    pile_base = 2 * n_nodes

    def soil_node(i):
        return soil_base + i + 1

    def spring_node(i):
        return spring_base + i + 1

    def pile_node(i):
        return pile_base + i + 1

    # ── Create pile nodes ──
    for i in range(n_nodes):
        depth = node_depths[i]
        x = 0.0
        y = -depth  # y-axis points up, pile goes down
        ops.node(pile_node(i), x, y)

    # ── Create soil + spring nodes (only for embedded portion) ──
    spring_nodes_created = set()
    for i in range(n_nodes):
        depth = node_depths[i]
        z_below_ground = depth - pile_above_ground  # depth below ground surface

        if z_below_ground < 0:
            continue  # above ground — no spring

        x = 0.0
        y = -depth

        ops.node(soil_node(i), x, y)
        ops.node(spring_node(i), x, y)

        # Fix soil node (all DOFs)
        ops.fix(soil_node(i), 1, 1, 1)

        spring_nodes_created.add(i)

    # ── Boundary conditions ──
    # Pile tip: pin (no translation, free rotation)
    i_tip = n_nodes - 1
    ops.fix(pile_node(i_tip), 1, 1, 0)

    # Pile head boundary
    if head_condition == 'fixed':
        ops.fix(pile_node(0), 0, 0, 1)  # fix rotation at head

    # ── PySimple1 / TzSimple1 materials and zeroLength elements ──
    mat_tag = 100
    ele_tag = 1000

    for i in spring_nodes_created:
        depth = node_depths[i]
        z_below_ground = depth - pile_above_ground

        # Tributary length for this node
        if i == 0 or i == n_nodes - 1:
            trib = elem_length / 2.0
        else:
            trib = elem_length

        # Get p-y model at this depth
        py_model, model_name = _get_layer_at_depth(layers_parsed, z_below_ground)

        # Compute PySimple1 parameters
        pult, y50 = _get_py_params(py_model, z_below_ground, b, trib)

        # Determine soilType: 1=clay, 2=sand
        soil_type = 1 if model_name in _CLAY_MODELS else 2

        # Drag resistance Cd
        Cd = 0.0  # static analysis

        # Create lateral spring material (PySimple1)
        py_mat_tag = mat_tag
        mat_tag += 1
        ops.uniaxialMaterial('PySimple1', py_mat_tag, soil_type,
                             pult, y50, Cd)

        # Create vertical spring material (TzSimple1) — shaft friction
        tz_mat_tag = mat_tag
        mat_tag += 1
        # Simple estimate: tult = alpha * sigma_v * pi * D * trib
        # Use a default small t-z: tult = 0.5 * pult (rough approx)
        # z50_tz = 0.5% of pile diameter
        tult = 0.5 * pult
        z50_tz = 0.005 * b
        ops.uniaxialMaterial('TzSimple1', tz_mat_tag, 2,
                             max(tult, 0.001), max(z50_tz, 1e-6), 0.0)

        # zeroLength element connecting soil to spring node
        ops.element('zeroLength', ele_tag, soil_node(i), spring_node(i),
                    '-mat', py_mat_tag, tz_mat_tag,
                    '-dir', 1, 2)
        ele_tag += 1

        # Connect spring node to pile node via equalDOF
        ops.equalDOF(pile_node(i), spring_node(i), 1, 2)

    # ── Tip spring (QzSimple1) ──
    # Apply only if tip node has a spring
    if i_tip in spring_nodes_created:
        z_tip = pile_length
        py_model_tip, model_name_tip = _get_layer_at_depth(layers_parsed, z_tip)
        # Tip bearing: rough estimate qult = 10 * pult_per_m * D
        pult_tip, _ = _get_py_params(py_model_tip, z_tip, b, 1.0)
        qult = 10.0 * pult_tip * b  # kN
        z50_q = 0.01 * b  # 1% of diameter
        qz_mat_tag = mat_tag
        mat_tag += 1
        ops.uniaxialMaterial('QzSimple1', qz_mat_tag, 2,
                             max(qult, 0.001), max(z50_q, 1e-6), 0.0)

        # Replace the existing vertical spring at tip with QzSimple1
        # (Already created TzSimple1 above — add a separate qz element
        #  on a new pair of nodes for the tip bearing)
        # Simpler approach: just update the vertical material in-place
        # Since we can't update, add an additional zeroLength for tip bearing
        ops.node(3 * n_nodes + 1, 0.0, -total_length)
        ops.fix(3 * n_nodes + 1, 1, 1, 1)
        ops.element('zeroLength', ele_tag, 3 * n_nodes + 1, spring_node(i_tip),
                    '-mat', qz_mat_tag, '-dir', 2)
        ele_tag += 1

    # ── Pile beam elements ──
    beam_ele_start = ele_tag
    for i in range(n_elem):
        ops.element('elasticBeamColumn', ele_tag,
                    pile_node(i), pile_node(i + 1),
                    A_pile, E_pile, I_pile)
        ele_tag += 1

    # ── Apply loads ──
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    head = pile_node(0)
    ops.load(head, lateral_load, -axial_load, moment)

    # ── Static analysis ──
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-5, 40, 0)
    ops.algorithm('Newton')

    # Load in increments for nonlinear convergence
    n_steps = 20
    ops.integrator('LoadControl', 1.0 / n_steps)
    ops.analysis('Static')

    converged = True
    for step in range(n_steps):
        ok = ops.analyze(1)
        if ok != 0:
            # Try modified Newton
            ops.algorithm('ModifiedNewton')
            ok = ops.analyze(1)
            ops.algorithm('Newton')
        if ok != 0:
            converged = False
            break

    # ── Extract results ──
    z_arr = node_depths.copy()
    defl_arr = np.zeros(n_nodes)
    moment_arr = np.zeros(n_nodes)
    shear_arr = np.zeros(n_nodes)
    soil_rxn_arr = np.zeros(n_nodes)

    for i in range(n_nodes):
        disp = ops.nodeDisp(pile_node(i))
        defl_arr[i] = disp[0]  # lateral displacement

    # Extract element forces for moment and shear
    for i in range(n_elem):
        e_tag = beam_ele_start + i
        forces = ops.eleForce(e_tag)
        # elasticBeamColumn 2D returns [Fx_i, Fy_i, Mz_i, Fx_j, Fy_j, Mz_j]
        # Node i is pile_node(i), node j is pile_node(i+1)
        shear_arr[i] += forces[1]   # Vy at node i
        moment_arr[i] += forces[2]  # Mz at node i
        if i == n_elem - 1:
            shear_arr[i + 1] = forces[4]  # Vy at last node
            moment_arr[i + 1] = forces[5]  # Mz at last node

    # Soil reaction: from spring forces
    for i in spring_nodes_created:
        ops.reactions()
        rxn = ops.nodeReaction(soil_node(i))
        soil_rxn_arr[i] = -rxn[0] / elem_length  # kN/m (negate reaction)

    # ── Key results ──
    head_disp = ops.nodeDisp(head)
    y_top = head_disp[0]
    rotation_top = head_disp[2]

    max_defl = float(np.max(np.abs(defl_arr)))
    max_moment = float(np.max(np.abs(moment_arr)))
    max_moment_idx = int(np.argmax(np.abs(moment_arr)))
    max_moment_depth = float(z_arr[max_moment_idx])

    return BNWFPileResult(
        pile_length=pile_length,
        pile_diameter=pile_diameter,
        lateral_force_kN=lateral_load,
        moment_kNm=moment,
        axial_force_kN=axial_load,
        z=z_arr,
        deflection_m=defl_arr,
        moment_profile_kNm=moment_arr,
        shear_profile_kN=shear_arr,
        soil_reaction_kN_per_m=soil_rxn_arr,
        y_top_m=y_top,
        rotation_top_rad=rotation_top,
        max_moment_kNm=max_moment,
        max_moment_depth_m=max_moment_depth,
        max_deflection_m=max_defl,
        converged=converged,
    )
