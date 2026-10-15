"""
Port of the OpenSees wiki example freeFieldSingle.tcl to openseespy.

A TEST HELPER, not app code (GM-83). The example is "Site Response
Analysis of a Layered Soil Column (Total Stress Analysis)", prepared by
Christopher McGann and Pedro Arduino, University of Washington (the Tcl
header also names HyungSuk Shin and Peter Mackenzie-Helnwein):
https://opensees.berkeley.edu/wiki/index.php/Site_Response_Analysis_of_a_Layered_Soil_Column_(Total_Stress_Analysis)
(files in the page's siteResponseAnalysis.zip, dated 2010-09-15).

The port follows the Tcl line for line, in its order and with its
numbers: a 40 m layer (rho 1.7 Mg/m3, Vs 250 m/s, Poisson's ratio 0),
PressureIndependMultiYield (cohesion 95 kPa, peak shear strain 0.05,
phi 0, refPress 80, pressure coefficient 0, 25 surfaces) on four-node
`quad` elements in plane strain sized for 100 Hz at 10 elements a
wavelength (160 elements of 0.25 m); a Lysmer-Kuhlemeyer dashpot
(rock Vs 760 m/s, rho 2.4 Mg/m3) on a zeroLength element; the motion as a
force history c * v(t) on the base node (Joyner & Chen 1975); gravity as
element body forces (the unit weight) in a transient analysis of 10
elastic and 40 elastoplastic steps of 500 s; the clock set back to 0;
Rayleigh damping of 2 % at 0.2 and 20 Hz; Newmark (0.5, 0.25) at the
record's 0.005 s step.

What differs from the Tcl, none of it numerical:
  * the recorders are replaced by reading the top node's acceleration
    after each step (the Tcl records every node at -dT 0.005, the
    analysis step, and the example's accelPlot.m plots the top node);
  * the convergence tests print nothing (the Tcl's print flag 1 only
    prints the norms);
  * the velocity history is passed as values (`-values`) instead of read
    from velocityHistory.out (`-filePath`); the values are the same;
  * a failed step raises instead of ending the run quietly (the Tcl does
    not check `analyze`'s return value).

The clock and the input force (checked, GM-83): the Tcl DOES reset the
clock (`setTime 0.0`, then `wipeAnalysis`) after the gravity stages,
which end at t = 25,000 s, and its Path series has no start time, so the
series starts at t = 0 and value k applies at t = k * 0.005 s. The first
dynamic step lands at t = 0.005 s with force c * v[1]; v[0] = 0 (the
example's processMotion.m integrates the record with cumtrapz, which
starts at 0), so the column starts at rest. The record has 7,990 values
(t = 0 to 39.945 s) and the analysis runs 7,990 steps, to t = 39.95 s:
the last step reads past the end of the series, where Path returns 0 (no
-useLast), so the force steps from c * v[7989] (v = -2.6e-6 m/s, 1.2e-3
kN on the 0.25 m column) to 0. Harmless. Without the clock reset the
whole series would lie in the past and the column would not move -- the
defect GM-77 found in our own module.

Two things the Tcl does that a reader should know:
  * The time-step check `kTrial = sizeEleX / pow(Vs, 0.5)` divides by the
    SQUARE ROOT of Vs (0.0158 s here), not Vs (the wave's crossing time of
    one element, 0.001 s), so it always keeps the record's 0.005 s step.
    Newmark's average-acceleration rule is unconditionally stable, so the
    run is stable either way; the 0.005 s step is fine for the 0-20 Hz a
    response spectrum uses.
  * Poisson's ratio 0 makes the elastic gravity stage leave NO horizontal
    stress (K0 = nu / (1 - nu) = 0), so below about 11 m (where the
    vertical stress passes 2c = 190 kPa, the von Mises limit for a
    vertical-only stress) the column starts its elastoplastic stage
    outside the failure surface; PressureIndependMultiYield then scales
    the stress back onto the surface and the 40 plastic gravity steps
    settle the column at failure in vertical-horizontal shear. The
    example's own text chose nu = 0 so that no vertical motion appears.
"""

import math
import os

import numpy as np

# The example's record step and length (freeFieldSingle.tcl, section 1)
MOTION_DT = 0.005
MOTION_STEPS = 7990

G_ACCEL = 9.81

# The first 20 s of the example's GilroyNo1EW.out in micro-g (the file's
# header says how it was derived; the full record is 121 KB of text).
RECORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "gilroy1_ew_first20s_microg.txt")

# The column's site frequency Vs / 4H (250 / 160 = 1.5625 Hz, period
# 0.64 s) and the spectral periods the comparison reports (5 % damping)
SITE_FREQUENCY_HZ = 250.0 / (4.0 * 40.0)
PERIODS_S = (0.1, 0.2, 0.3, 0.4, 0.5, 0.64, 1.0, 1.5)


def load_record():
    """The stored input motion: (acceleration in g, time step in s)."""
    microg = np.loadtxt(RECORD_FILE, comments="#").reshape(-1)
    return microg * 1.0e-6, MOTION_DT


def rayleigh_ratio(f_hz, rayleigh_hz=(0.2, 20.0), damp=0.02):
    """Damping ratio at f_hz of the example's Rayleigh damping (2 % at
    0.2 and 20 Hz): a0 / (2 w) + a1 w / 2. At the site frequency it is
    0.41 %; at 10 Hz 1.0 %; at 20 Hz 2 %."""
    pi = 3.141592654
    w1, w2 = 2 * pi * rayleigh_hz[0], 2 * pi * rayleigh_hz[1]
    a0 = 2 * damp * w1 * w2 / (w1 + w2)
    a1 = 2 * damp / (w1 + w2)
    w = 2 * pi * f_hz
    return a0 / (2 * w) + a1 * w / 2


def summarize(surface_accel_g, input_accel_g, dt=MOTION_DT,
              periods=PERIODS_S):
    """PGA (g), its time (s), and per period the 5 %-damped spectral
    acceleration of the surface motion (g) and its ratio to the input's
    (the amplification spectrum), all on the same samples."""
    from opensees_agent.opensees_utils import compute_response_spectrum
    n = min(len(surface_accel_g), len(input_accel_g))
    surf = np.asarray(surface_accel_g, dtype=float)[:n]
    inp = np.asarray(input_accel_g, dtype=float)[:n]
    _, sa = compute_response_spectrum(surf, dt, np.asarray(periods))
    _, sa_in = compute_response_spectrum(inp, dt, np.asarray(periods))
    k = int(np.argmax(np.abs(surf)))
    return dict(pga_g=float(abs(surf[k])), pga_time_s=k * dt,
                sa_g=[float(x) for x in sa],
                amplification=[float(x) for x in sa / sa_in])


def velocity_from_acceleration(accel_g, dt=MOTION_DT):
    """Velocity (m/s) from acceleration (g), as the example's
    processMotion.m does: a * 9.81, cumtrapz (starting at 0), times dt."""
    a = np.asarray(accel_g, dtype=float) * G_ACCEL
    return np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)))


def run_free_field_single(velocity, motion_dt=MOTION_DT, motion_steps=None,
                          elastoplastic=True, nu=0.0,
                          rayleigh_hz=(0.2, 20.0), damp=0.02,
                          gamma_peak=0.05):
    """Run freeFieldSingle.tcl with ``velocity`` (m/s) as velocityHistory.out.

    ``motion_steps`` defaults to the number of velocity values (the Tcl's
    7,990 for the full record). The keyword arguments after it default to
    the Tcl's values; they exist only so a test or a study can change one
    thing at a time and see what it does: ``elastoplastic=False`` skips
    ``updateMaterialStage`` (the soil stays linear elastic); ``nu``,
    ``rayleigh_hz`` / ``damp`` and ``gamma_peak`` are the three inputs
    ``analyze_site_response`` cannot set as the Tcl does (see
    ``API_EXPRESSIBLE``).

    Returns a dict: ``time`` (s, starting at 0) and ``surface_accel_g``
    (the top node, starting at rest at t = 0, as the example's
    respSpectra.m prepends), plus the model facts the Tcl prints.
    """
    import openseespy.opensees as ops

    velocity = np.asarray(velocity, dtype=float)
    if motion_steps is None:
        motion_steps = len(velocity)

    ops.wipe()

    # ------------------------------------------------------------------
    #  1. DEFINE ANALYSIS PARAMETERS
    # ------------------------------------------------------------------
    # ---SOIL GEOMETRY
    soilThick = 40.0
    # ---MATERIAL PROPERTIES
    rho = 1.7
    Vs = 250.0
    G = rho * Vs * Vs
    # (nu = 0.0 in the Tcl)
    E = 2 * G * (1 + nu)
    bulk = E / (3 * (1 - 2 * nu))
    cohesion = 95.0
    gammaPeak = gamma_peak  # 0.05 in the Tcl
    phi = 0.0
    refPress = 80.0
    pressCoeff = 0.0
    rockVS = 760
    rockDen = 2.4
    # ---GROUND MOTION PARAMETERS
    motionDT = motion_dt
    motionSteps = motion_steps
    # ---WAVELENGTH PARAMETERS
    fMax = 100.0
    wave = Vs / fMax
    nEle = 10
    # ---RAYLEIGH DAMPING PARAMETERS (0.2 and 20 Hz in the Tcl)
    pi = 3.141592654
    omega1 = 2 * pi * rayleigh_hz[0]
    omega2 = 2 * pi * rayleigh_hz[1]
    a0 = 2 * damp * omega1 * omega2 / (omega1 + omega2)
    a1 = 2 * damp / (omega1 + omega2)
    # ---ANALYSIS PARAMETERS
    gamma = 0.5
    beta = 0.25

    # ------------------------------------------------------------------
    #  2. DEFINE MESH GEOMETRY
    # ------------------------------------------------------------------
    hTrial = wave / nEle
    nTrial = soilThick / hTrial
    if nTrial > math.floor(soilThick / hTrial):
        numEleY = int(math.floor(soilThick / hTrial) + 1)
    else:
        numEleY = int(nTrial)
    sizeEleY = soilThick / numEleY
    sizeEleX = sizeEleY
    numNodeY = 2 * (numEleY + 1)

    # ------------------------------------------------------------------
    #  3. DEFINE NODES FOR SOIL ELEMENTS
    # ------------------------------------------------------------------
    ops.model('BasicBuilder', '-ndm', 2, '-ndf', 2)
    yCoord = 0.0
    count = 0
    for j in range(1, numNodeY + 1, 2):
        ops.node(j, 0.0, yCoord + count * sizeEleY)
        ops.node(j + 1, sizeEleX, yCoord + count * sizeEleY)
        count = count + 1

    # ------------------------------------------------------------------
    #  4. DEFINE DASHPOT NODES
    # ------------------------------------------------------------------
    ops.node(2000, 0.0, 0.0)
    ops.node(2001, 0.0, 0.0)

    # ------------------------------------------------------------------
    #  5. DEFINE BOUNDARY CONDITIONS AND EQUAL DOF
    # ------------------------------------------------------------------
    ops.fix(1, 0, 1)
    ops.fix(2, 0, 1)
    ops.fix(2000, 1, 1)
    ops.fix(2001, 0, 1)
    for k in range(3, numNodeY + 1, 2):
        ops.equalDOF(k, k + 1, 1, 2)
    ops.equalDOF(1, 2, 1)
    ops.equalDOF(1, 2001, 1)

    # ------------------------------------------------------------------
    #  6. DEFINE SOIL MATERIALS
    # ------------------------------------------------------------------
    ops.nDMaterial('PressureIndependMultiYield', 1, 2, rho, G, bulk,
                   cohesion, gammaPeak, phi, refPress, pressCoeff, 25)

    # ------------------------------------------------------------------
    #  7. DEFINE SOIL ELEMENTS
    # ------------------------------------------------------------------
    wgtX = 0.0
    wgtY = -9.81 * rho
    for j in range(1, numEleY + 1):
        nI = 2 * j - 1
        nJ = nI + 1
        nK = nI + 3
        nL = nI + 2
        ops.element('quad', j, nI, nJ, nK, nL, 1.0, 'PlaneStrain', 1,
                    0.0, 0.0, wgtX, wgtY)

    # ------------------------------------------------------------------
    #  8. DEFINE MATERIAL AND ELEMENTS FOR VISCOUS DAMPERS
    # ------------------------------------------------------------------
    mC = sizeEleX * rockDen * rockVS
    ops.uniaxialMaterial('Viscous', 4000, mC, 1)
    ops.element('zeroLength', 5000, 2000, 2001, '-mat', 4000, '-dir', 1)

    # (8. gravity recorders: not needed, see the module docstring)

    # ------------------------------------------------------------------
    #  9. APPLY GRAVITY LOADING
    # ------------------------------------------------------------------
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1e-5, 30, 0)
    ops.algorithm('Newton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', gamma, beta)
    ops.analysis('Transient')

    _analyze(ops, 10, 5.0e2, "elastic gravity")

    if elastoplastic:
        ops.updateMaterialStage('-material', 1, '-stage', 1)
        _analyze(ops, 40, 5.0e2, "plastic gravity")
    else:
        # linear elastic study: the same 40 steps, stage 0 kept
        _analyze(ops, 40, 5.0e2, "gravity (elastic kept)")

    # ------------------------------------------------------------------
    #  10. CREATE POST-GRAVITY RECORDERS
    # ------------------------------------------------------------------
    ops.setTime(0.0)
    ops.wipeAnalysis()

    # ------------------------------------------------------------------
    #  11. DYNAMIC ANALYSIS
    # ------------------------------------------------------------------
    cFactor = sizeEleX * rockDen * rockVS
    ops.timeSeries('Path', 1, '-dt', motionDT,
                   '-values', *velocity.tolist(), '-factor', cFactor)
    ops.pattern('Plain', 10, 1)
    ops.load(1, 1.0, 0.0)

    # ---DETERMINE STABLE ANALYSIS TIME STEP USING CFL CONDITION
    duration = motionDT * motionSteps
    kTrial = sizeEleX / (pow(Vs, 0.5))
    if motionDT <= kTrial:
        nSteps = motionSteps
        dT = motionDT
    else:
        nSteps = int(math.floor(duration / kTrial) + 1)
        dT = duration / nSteps

    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1e-3, 15, 0)
    ops.algorithm('Newton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', gamma, beta)
    ops.rayleigh(a0, a1, 0.0, 0.0)
    ops.analysis('Transient')

    top = numNodeY  # the node accelPlot.m plots (equalDOF to numNodeY - 1)
    time = np.zeros(nSteps + 1)
    accel = np.zeros(nSteps + 1)
    for step in range(1, nSteps + 1):
        _analyze(ops, 1, dT, f"dynamic step {step}")
        time[step] = ops.getTime()
        accel[step] = ops.nodeAccel(top, 1) / G_ACCEL

    ops.wipe()
    return dict(time=time, surface_accel_g=accel, numEleY=numEleY,
                sizeEleX=sizeEleX, a0=a0, a1=a1, nSteps=nSteps, dT=dT,
                dashpot_c=mC, bulk=bulk)


def _analyze(ops, n, dt, what):
    ok = ops.analyze(n, dt)
    if ok != 0:
        raise RuntimeError(f"freeFieldSingle port: {what} did not converge")


# ===========================================================================
# The same column through opensees_agent.analyze_site_response
# ===========================================================================

def closest_api_inputs():
    """analyze_site_response's inputs for the example's column, as close as
    its arguments allow (the accelerations come from ``load_record``).

    Mapped exactly: thickness, Vs, density, the rock (760 m/s, 2.4 Mg/m3),
    the mesh (160 elements of 0.25 m), 25 yield surfaces, the cohesion
    (the module sets cohesi = sqrt(3)/2 * su, so su = 2 * 95 / sqrt(3)
    gives the Tcl's 95 kPa), the input force (c * v, v by trapezoid from
    0), the time step, the gravity stages and the clock reset. A dry
    column: the water table at the base drains every node, so the module's
    SSPquadUP elements carry no pore pressure (checked: on this column a
    dry SSPquadUP, an SSPquad and the Tcl's quad give the same surface
    motion to 1e-12).

    Not expressible, and what is done instead:
      * Poisson's ratio: the module's clay is nearly incompressible
        (nu = 0.495, K0 about 1 after gravity); the Tcl's is 0 (K0 = 0);
      * Rayleigh damping: the module puts the target ratio at the site
        frequency and five times it; the Tcl puts 2 % at 0.2 and 20 Hz,
        which is 0.41 % at the site frequency. ``damping`` is set to that
        0.41 %; above about 8 Hz the module's column is then less damped
        than the Tcl's (0.9 % against 2 % at 20 Hz);
      * peak shear strain: 0.1 in the module, 0.05 in the Tcl (the
        backbone's reference strain moves 0.6 %).
    """
    return dict(
        layers=[{"thickness": 40.0, "Vs": 250.0, "density": 1.7,
                 "material_type": "clay", "su": 2.0 * 95.0 / math.sqrt(3.0),
                 "n_surf": 25}],
        gwt_depth=40.0, bedrock_Vs=760.0, bedrock_density=2.4,
        damping=rayleigh_ratio(SITE_FREQUENCY_HZ),
        n_elem_per_layer=160)


# The port with the three inputs above set as analyze_site_response sets
# them: what the module's column should reproduce exactly.
API_EXPRESSIBLE = dict(
    nu=0.495,
    rayleigh_hz=(SITE_FREQUENCY_HZ, 5.0 * SITE_FREQUENCY_HZ),
    damp=rayleigh_ratio(SITE_FREQUENCY_HZ),
    gamma_peak=0.1)


def reference_numbers():
    """Run the port three ways on the stored record and summarize each:
    as written; linear elastic (no updateMaterialStage); and with the
    inputs analyze_site_response cannot set as the Tcl does set as it sets
    them (``API_EXPRESSIBLE``). The test file's REFERENCE is this output.

    On the 20 s record the port's last step (t = 20.0 s) reads past the
    end of the series and the base force drops to 0; ``summarize`` keeps
    only the record's own 4,000 samples, so that step is not compared."""
    accel, dt = load_record()
    vel = velocity_from_acceleration(accel, dt)
    runs = {"as_written": {}, "linear": {"elastoplastic": False},
            "api_expressible": API_EXPRESSIBLE}
    out = {}
    for name, kw in runs.items():
        r = run_free_field_single(vel, motion_dt=dt, **kw)
        out[name] = summarize(r["surface_accel_g"], accel, dt)
    out["input_sa_g"] = summarize(accel, accel, dt)["sa_g"]
    return out
