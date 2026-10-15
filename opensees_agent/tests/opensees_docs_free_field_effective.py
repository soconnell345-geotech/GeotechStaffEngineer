"""
Port of the OpenSeesPy docs' effective stress site response example.

A TEST HELPER, not app code (GM-83). The example is "Effective Stress
Site Response Analysis of a Layered Soil Column", OpenSeesPy
documentation section 14.4.2 (freeFieldEffective.py,
https://openseespydoc.readthedocs.io/en/latest/src/freeFieldEffective.html),
converted to openseespy by Harsh Mistry (University of Manchester) from
the OpenSees wiki example by Christopher McGann, HyungSuk Shin, Pedro
Arduino and Peter Mackenzie-Helnwein (University of Washington),
https://opensees.berkeley.edu/wiki/index.php/Effective_Stress_Site_Response_Analysis_of_a_Layered_Soil_Column
It is the recipe ``analyze_site_response`` adapts.

The port follows freeFieldEffective.py in its order and with its numbers:
a 30 m column of three PressureDependMultiYield02 layers (bottom to top:
20 m dense sand, rho 2.45, phi 39; 8 m loose sand, rho 2.24, phi 32; a
2 m dry crust, rho 1.8, phi 32) with the water table 2 m down, on a 2 %
slope (gravity tilted), meshed with 60 nine-node `9_4_QuadUP` elements of
0.5 m; a Lysmer-Kuhlemeyer dashpot (rock 700 m/s, 2.5 Mg/m3); gravity in
10 elastic and 40 plastic steps of 500 s with permeability 1 m/s, then the
permeabilities set to 1e-4 m/s; the clock set back to 0; the force
c * v(t) on the base; Rayleigh damping of 2 % at 0.2 and 20 Hz; Newmark
(0.5, 0.25) with Penalty constraints, KrylovNewton and NormDispIncr 1e-3.

What differs from the Python, none of it numerical:
  * the recorders (and the node list written to Node_record.txt) are
    replaced by reading the top node's acceleration and the pore pressure
    after each step;
  * ``np.float`` / ``np.int`` (removed from numpy) become float / int;
  * the velocity is passed as values, not read from velocityHistory.txt
    (that file is the same Gilroy No. 1 velocity rounded to three
    significant figures; the test passes the trapezoid of its stored
    record, within 5e-4 m/s of it);
  * the convergence tests print nothing;
  * a failed dynamic step raises; the example would instead halve the
    step for the rest of the run (twice at most). No step fails on the
    stored record, so the retry is not ported.

Two things the Python does that a reader should know: its ``sElemY``
loop reads ``layerThick[i - 1] / nElemY[i - 1]``, off by one, but every
layer's elements are 0.5 m so it does no harm; and, as in the Tcl, the
time-step check divides by the square root of the shear wave velocity
and keeps the record's 0.005 s.
"""

import math

import numpy as np

G_ACCEL = 9.81


def run_free_field_effective(velocity, motion_dt=0.005, motion_steps=None,
                             grade=2.0, algorithm="KrylovNewton",
                             module_inputs=False, rayleigh_hz=(0.2, 20.0),
                             damp=0.02):
    """Run freeFieldEffective.py with ``velocity`` (m/s) as its velocity
    history. The keyword arguments default to the example's; they exist so
    a study can change one thing and see what it does: ``grade`` (%, 0 =
    level ground), ``algorithm`` (the dynamic stage's), the Rayleigh
    damping, and ``module_inputs=True``, which sets the inputs
    ``analyze_site_response`` cannot set as the example does as the module
    sets them: every bulk modulus 2(1 + 0.3) / (3(1 - 0.6)) times the shear
    modulus (the module's drained Poisson's ratio 0.3; the example's are
    0.32 and 0.29), the pore fluid's bulk modulus 2.2e6 kPa (the example's
    6.88e6 and 5.06e6), and the dense layer's contraction and dilation
    constants the module's Dr = 40 % ones (0.067, 0.23, 0.06, 0.27; the
    example's 0.010, 0.0, 0.35, 0.0) -- its void ratio 0.47 the module can
    take.

    Returns ``time``, ``surface_accel_g`` (the top node), ``pp_nodes``
    ((node, elevation) of the corner nodes up the left side, which carry
    the pore pressure), ``pp0`` (their pore pressure after gravity, kPa)
    and ``max_excess_pp`` (their largest rise above it during shaking).
    The u-p elements keep pore pressure as the "velocity" of DOF 3, which
    is what the example's recorder reads.
    """
    import openseespy.opensees as op

    velocity = np.asarray(velocity, dtype=float)
    if motion_steps is None:
        motion_steps = len(velocity)

    op.wipe()

    # ------------------------------------------------------------------
    #  1. DEFINE SOIL AND MESH GEOMETRY
    # ------------------------------------------------------------------
    soilThick = 30.0
    numLayers = 3
    layerThick = [20.0, 8.0, 2.0]
    waterTable = 2.0
    layerBound = np.zeros((numLayers, 1))
    layerBound[0] = layerThick[0]
    for i in range(1, numLayers):
        layerBound[i] = layerBound[i - 1] + layerThick[i]
    nElemX = 1
    nNodeX = 2 * nElemX + 1
    sElemX = 2.0
    nElemY = [40, 16, 4]
    nElemT = 60
    sElemY = np.zeros((numLayers, 1))
    for i in range(numLayers):
        sElemY[i] = [layerThick[i - 1] / nElemY[i - 1]]
    nNodeY = 2 * nElemT + 1
    nNodeT = nNodeX * nNodeY

    # ------------------------------------------------------------------
    #  2. CREATE PORE PRESSURE NODES AND FIXITIES
    # ------------------------------------------------------------------
    op.model('basic', '-ndm', 2, '-ndf', 3)
    count = 1
    layerNodeCount = 0
    dry_Node = np.zeros((500, 1))
    pp_nodes = []
    for k in range(1, numLayers + 1):
        for i in range(1, nNodeX + 1, 2):
            if k == 1:
                bump = 1
            else:
                bump = 0
            j_end = 2 * nElemY[k - 1] + bump
            for j in range(1, j_end + 1, 2):
                xCoord = (i - 1) * (sElemX / 2)
                yctr = j + layerNodeCount
                yCoord = (yctr - 1) * (float(sElemY[k - 1][0])) / 2
                nodeNum = i + ((yctr - 1) * nNodeX)
                op.node(nodeNum, xCoord, yCoord)
                if i == 1:
                    pp_nodes.append((nodeNum, yCoord))
                waterHeight = soilThick - waterTable
                if yCoord >= waterHeight:
                    dry_Node[count] = nodeNum
                    count = count + 1
        layerNodeCount = yctr + 1
    dryNode = np.trim_zeros(dry_Node.ravel())
    for i in range(count - 1):
        n_dryNode = int(dryNode[i])
        op.fix(n_dryNode, 0, 0, 1)
    op.fix(1, 0, 1, 0)
    op.fix(3, 0, 1, 0)
    for i in range(1, ((3 * nNodeY) - 2), 6):
        op.equalDOF(i, i + 2, 1, 2)

    # ------------------------------------------------------------------
    #  3. CREATE INTERIOR NODES AND FIXITIES
    # ------------------------------------------------------------------
    op.model('basic', '-ndm', 2, '-ndf', 2)
    xCoord = float(sElemX / 2)
    layerNodeCount = 0
    for k in range(1, numLayers + 1):
        if k == 1:
            bump = 1
        else:
            bump = 0
        j_end = 2 * nElemY[k - 1] + bump
        for j in range(1, j_end + 1, 1):
            yctr = j + layerNodeCount
            yCoord = (yctr - 1) * (float(sElemY[k - 1][0])) / 2
            nodeNum = (3 * yctr) - 1
            op.node(nodeNum, xCoord, yCoord)
        layerNodeCount = yctr
    layerNodeCount = 0
    for k in range(1, numLayers + 1):
        for j in range(1, ((nElemY[k - 1]) + 1)):
            yctr = j + layerNodeCount
            yCoord = float(sElemY[k - 1][0]) * (yctr - 0.5)
            nodeNumL = (6 * yctr) - 2
            nodeNumR = nodeNumL + 2
            op.node(nodeNumL, 0.0, yCoord)
            op.node(nodeNumR, sElemX, yCoord)
        layerNodeCount = yctr
    op.fix(2, 0, 1)
    for i in range(1, ((3 * nNodeY) - 6), 6):
        op.equalDOF(i, i + 1, 1, 2)
        op.equalDOF(i + 3, i + 4, 1, 2)
        op.equalDOF(i + 3, i + 5, 1, 2)
    op.equalDOF(nNodeT - 2, nNodeT - 1, 1, 2)

    # ------------------------------------------------------------------
    #  4. CREATE SOIL MATERIALS
    # ------------------------------------------------------------------
    slope = math.atan(grade / 100.0)
    g = -9.81
    xwgt_var = g * (math.sin(slope))
    ywgt_var = g * (math.cos(slope))
    thick = [1.0, 1.0, 1.0]
    xWgt = [xwgt_var, xwgt_var, xwgt_var]
    yWgt = [ywgt_var, ywgt_var, ywgt_var]
    uBulk = [6.88E6, 5.06E6, 5.0E-6]
    hPerm = [1.0E-4, 1.0E-4, 1.0E-4]
    vPerm = [1.0E-4, 1.0E-4, 1.0E-4]
    B12, B3 = 2.2e5, 2.6e5
    dense = (0.010, 0.0, 0.35, 0.0)
    if module_inputs:
        b_over_g = 2.0 * (1.0 + 0.3) / (3.0 * (1.0 - 2.0 * 0.3))
        B12, B3 = b_over_g * 9.0e4, b_over_g * 1.3e5
        uBulk = [2.2e6, 2.2e6, 5.0E-6]
        dense = (0.067, 0.23, 0.06, 0.27)
    op.nDMaterial('PressureDependMultiYield02', 3, 2, 1.8, 9.0e4, B12, 32,
                  0.1, 101.0, 0.5, 26, 0.067, 0.23, 0.06, 0.27, 20, 5.0,
                  3.0, 1.0, 0.0, 0.77, 0.9, 0.02, 0.7, 101.0)
    op.nDMaterial('PressureDependMultiYield02', 2, 2, 2.24, 9.0e4, B12,
                  32, 0.1, 101.0, 0.5, 26, 0.067, 0.23, 0.06, 0.27, 20,
                  5.0, 3.0, 1.0, 0.0, 0.77, 0.9, 0.02, 0.7, 101.0)
    op.nDMaterial('PressureDependMultiYield02', 1, 2, 2.45, 1.3e5, B3,
                  39, 0.1, 101.0, 0.5, 26, *dense, 20, 5.0,
                  3.0, 1.0, 0.0, 0.47, 0.9, 0.02, 0.7, 101.0)

    # ------------------------------------------------------------------
    #  5. CREATE SOIL ELEMENTS
    # ------------------------------------------------------------------
    for j in range(1, nElemT + 1):
        nI = (6 * j) - 5
        nJ = nI + 2
        nK = nI + 8
        nL = nI + 6
        nM = nI + 1
        nN = nI + 5
        nP = nI + 7
        nQ = nI + 3
        nR = nI + 4
        lowerBound = 0.0
        for i in range(1, numLayers + 1):
            if (j * sElemY[i - 1] <= layerBound[i - 1]
                    and j * sElemY[i - 1] > lowerBound):
                # permeabilities 1.0 m/s for the gravity analysis
                op.element('9_4_QuadUP', j, nI, nJ, nK, nL, nM, nN, nP, nQ,
                           nR, thick[i - 1], i, uBulk[i - 1], 1.0, 1.0, 1.0,
                           xWgt[i - 1], yWgt[i - 1])
            lowerBound = layerBound[i - 1]

    # ------------------------------------------------------------------
    #  6. LYSMER DASHPOT
    # ------------------------------------------------------------------
    dashF = nNodeT + 1
    dashS = nNodeT + 2
    op.node(dashF, 0.0, 0.0)
    op.node(dashS, 0.0, 0.0)
    op.fix(dashF, 1, 1)
    op.fix(dashS, 0, 1)
    op.equalDOF(1, dashS, 1)
    colArea = sElemX * thick[0]
    rockVS = 700.0
    rockDen = 2.5
    dashpotCoeff = rockVS * rockDen
    op.uniaxialMaterial('Viscous', numLayers + 1, dashpotCoeff * colArea, 1)
    op.element('zeroLength', nElemT + 1, dashF, dashS, '-mat',
               numLayers + 1, '-dir', 1)

    # (7. gravity recorders: not needed, see the module docstring)

    # ------------------------------------------------------------------
    #  8. DEFINE ANALYSIS PARAMETERS
    # ------------------------------------------------------------------
    motionDT = motion_dt
    motionSteps = motion_steps
    # (0.02 at 0.2 and 20 Hz in the example)
    omega1 = 2 * np.pi * rayleigh_hz[0]
    omega2 = 2 * np.pi * rayleigh_hz[1]
    a0 = 2 * damp * omega1 * omega2 / (omega1 + omega2)
    a1 = 2 * damp / (omega1 + omega2)
    vsMax = 250.0
    duration = motionDT * motionSteps
    minSize = sElemY[0]
    for i in range(2, numLayers + 1):
        if sElemY[i - 1] <= minSize:
            minSize = sElemY[i - 1]
    kTrial = float(minSize[0]) / (vsMax ** 0.5)
    if motionDT <= kTrial:
        nSteps = motionSteps
        dT = motionDT
    else:
        nSteps = int(math.floor(duration / kTrial) + 1)
        dT = duration / nSteps
    gamma = 0.5
    beta = 0.25

    # ------------------------------------------------------------------
    #  9. GRAVITY ANALYSIS
    # ------------------------------------------------------------------
    op.updateMaterialStage('-material', 1, '-stage', 0)
    op.updateMaterialStage('-material', 2, '-stage', 0)
    op.updateMaterialStage('-material', 3, '-stage', 0)
    op.constraints('Penalty', 1.0E14, 1.0E14)
    op.test('NormDispIncr', 1e-4, 35, 0)
    op.algorithm('KrylovNewton')
    op.numberer('RCM')
    op.system('ProfileSPD')
    op.integrator('Newmark', gamma, beta)
    op.analysis('Transient')
    _analyze(op, 10, 5.0E2, "elastic gravity")
    op.updateMaterialStage('-material', 1, '-stage', 1)
    op.updateMaterialStage('-material', 2, '-stage', 1)
    op.updateMaterialStage('-material', 3, '-stage', 1)
    _analyze(op, 40, 5.0e2, "plastic gravity")

    # ------------------------------------------------------------------
    #  10. UPDATE ELEMENT PERMEABILITY VALUES FOR POST-GRAVITY ANALYSIS
    # ------------------------------------------------------------------
    ctr = 10000.0
    for i in range(1, nElemT + 1):
        op.parameter(int(ctr + 1.0), 'element', i, 'vPerm')
        op.parameter(int(ctr + 2.0), 'element', i, 'hPerm')
        ctr = ctr + 2.0
    ctr = 10000.0
    for j in range(1, nElemT + 1):
        lowerBound = 0.0
        for i in range(1, numLayers + 1):
            if (j * sElemY[i - 1] <= layerBound[i - 1]
                    and j * sElemY[i - 1] > lowerBound):
                op.updateParameter(int(ctr + 1.0), vPerm[i - 1])
                op.updateParameter(int(ctr + 2.0), hPerm[i - 1])
            lowerBound = layerBound[i - 1]
        ctr = ctr + 2.0

    # Initial pore pressure along the left column (for excess pressure)
    pp0 = {n: op.nodeVel(n, 3) for n, _ in pp_nodes}

    # ------------------------------------------------------------------
    #  11. CREATE POST-GRAVITY RECORDERS
    # ------------------------------------------------------------------
    op.setTime(0.0)
    op.wipeAnalysis()

    # ------------------------------------------------------------------
    #  12. DYNAMIC ANALYSIS
    # ------------------------------------------------------------------
    op.model('basic', '-ndm', 2, '-ndf', 3)
    cFactor = colArea * dashpotCoeff
    op.timeSeries('Path', 2, '-dt', motionDT, '-values',
                  *velocity.tolist(), '-factor', cFactor)
    op.pattern('Plain', 10, 2)
    op.load(1, 1.0, 0.0, 0.0)
    op.constraints('Penalty', 1.0E16, 1.0E16)
    op.test('NormDispIncr', 1e-3, 35, 0)
    op.algorithm(algorithm)
    op.numberer('RCM')
    op.system('ProfileSPD')
    op.integrator('Newmark', gamma, beta)
    op.rayleigh(a0, a1, 0.0, 0.0)
    op.analysis('Transient')

    top = nNodeT - 2  # the left surface node (pore pressure column)
    time = np.zeros(nSteps + 1)
    accel = np.zeros(nSteps + 1)
    max_excess = {n: 0.0 for n, _ in pp_nodes}
    for step in range(1, nSteps + 1):
        _analyze(op, 1, dT, f"dynamic step {step}")
        time[step] = op.getTime()
        accel[step] = op.nodeAccel(top, 1) / G_ACCEL
        for n, _ in pp_nodes:
            ex = op.nodeVel(n, 3) - pp0[n]
            if ex > max_excess[n]:
                max_excess[n] = ex
    op.wipe()
    return dict(time=time, surface_accel_g=accel, nSteps=nSteps, dT=dT,
                pp_nodes=pp_nodes, max_excess_pp=max_excess, pp0=pp0)


def _analyze(op, n, dt, what):
    if op.analyze(n, dt) != 0:
        raise RuntimeError(f"freeFieldEffective port: {what} did not "
                           f"converge")


def max_ru_at_depths(result, depths=(3.0, 7.0, 16.0)):
    """Largest excess pore pressure over the initial vertical effective
    stress (the weight above, water 2 m down) at the corner nodes at these
    depths: the r_u the example's wiki page plots at 3, 7 and 16 m."""
    rho = lambda y: 2.45 if y <= 20.0 else (2.24 if y <= 28.0 else 1.8)
    out = []
    for d in depths:
        n = next(n for n, y in result["pp_nodes"]
                 if abs((30.0 - y) - d) < 1e-6)
        ys = np.linspace(30.0 - d, 30.0, 401)
        sv = np.trapezoid([rho(v) * G_ACCEL for v in ys], ys)
        out.append(result["max_excess_pp"][n] / (sv - G_ACCEL * (d - 2.0)))
    return out


# ===========================================================================
# The same column through opensees_agent.analyze_site_response
# ===========================================================================

def closest_api_inputs():
    """analyze_site_response's inputs for the example's column, as close as
    its arguments allow.

    The module meshes every layer with the same number of elements, so the
    three layers go in as fifteen 2 m sublayers of four 0.5 m elements (the
    example's mesh). The module's sand modulus is rho * Vs^2 at its
    reference pressure, the mean effective stress at mid-sublayer
    (sigma'v (1 + 2 K0) / 3), with pressure exponent 0.5; each sublayer's
    Vs is set so that equals the example's Gr (p / 101)^0.5 (Gr 9e4 kPa
    loose and crust, 1.3e5 dense), which gives the same G at every
    pressure. K0 0.47 / 0.40 (the example's elastic nu / (1 - nu)) only
    places that reference pressure. Friction angles, densities, void
    ratios (0.77; 0.47 dense), water table 2 m, rock 700 m/s and 2.5 Mg/m3,
    permeability 1e-4 m/s and the 20 yield surfaces map exactly; damping
    is the example's Rayleigh ratio at the module's site frequency
    (0.40 % at 1.77 Hz). What it cannot set is listed under
    ``run_free_field_effective(module_inputs=True)``, plus the slope (the
    module's column is level), the element (4-node SSPquadUP against
    9-node 9_4_QuadUP) and the gravity permeability (the example runs
    gravity at 1 m/s, the module at 1e-4).
    """
    from opensees_agent.site_response import _sigma_v_eff
    from opensees_agent.tests.opensees_wiki_free_field import rayleigh_ratio
    spec = [(2.0, 1.8, 32.0, 9.0e4, 0.77, 0.47),     # dry crust
            (8.0, 2.24, 32.0, 9.0e4, 0.77, 0.47),    # loose sand
            (20.0, 2.45, 39.0, 1.3e5, 0.47, 0.40)]   # dense sand
    layers, gr = [], []
    for H, rho, phi, Gr, e0, K0 in spec:
        for _ in range(int(round(H / 2.0))):
            layers.append({"thickness": 2.0, "density": rho,
                           "material_type": "sand", "phi": phi,
                           "e_init": e0, "K0": K0, "Vs": 1.0})
            gr.append(Gr)
    top = 0.0
    for L, Gr in zip(layers, gr):
        sv = _sigma_v_eff(layers, top + 1.0, 2.0)
        p = max(sv * (1.0 + 2.0 * L["K0"]) / 3.0, 5.0)
        L["Vs"] = math.sqrt(Gr * math.sqrt(p / 101.0) / L["density"])
        top += 2.0
    vs_avg = 30.0 / sum(L["thickness"] / L["Vs"] for L in layers)
    f1 = vs_avg / (4.0 * 30.0)
    return dict(layers=layers, gwt_depth=2.0, bedrock_Vs=700.0,
                bedrock_density=2.5, damping=rayleigh_ratio(f1),
                n_elem_per_layer=4), f1
