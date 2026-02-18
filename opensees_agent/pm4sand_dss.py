"""
PM4Sand undrained cyclic direct simple shear analysis.

Builds a single SSPquadUP element with the PM4Sand material model,
applies consolidation under a specified vertical stress, then drives
stress-controlled undrained cyclic loading to evaluate liquefaction
triggering (number of cycles to ru threshold).

Reference:
    Boulanger, R.W. & Ziotopoulou, K. (2017). "PM4Sand (Version 3.1):
    A Sand Plasticity Model for Earthquake Engineering Applications."
    Report No. UCD/CGM-17/01. UC Davis.
"""

import math

import numpy as np

from opensees_agent.results import PM4SandDSSResult


def _validate_pm4sand_inputs(Dr, G0, hpo, Den, sigma_v, CSR, K0, n_cycles):
    """Validate PM4Sand DSS inputs, raising ValueError for bad values."""
    if not (0.0 < Dr <= 1.0):
        raise ValueError(f"Dr must be in (0, 1], got {Dr}")
    if G0 <= 0:
        raise ValueError(f"G0 must be positive, got {G0}")
    if hpo <= 0:
        raise ValueError(f"hpo must be positive, got {hpo}")
    if Den <= 0:
        raise ValueError(f"Den (mass density) must be positive, got {Den}")
    if sigma_v <= 0:
        raise ValueError(f"sigma_v must be positive, got {sigma_v}")
    if CSR <= 0:
        raise ValueError(f"CSR must be positive, got {CSR}")
    if not (0.0 < K0 < 2.0):
        raise ValueError(f"K0 must be in (0, 2), got {K0}")
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")


def analyze_pm4sand_dss(
    # Required PM4Sand material parameters
    Dr,
    G0,
    hpo,
    Den,
    # Loading conditions
    sigma_v=100.0,
    CSR=0.15,
    K0=0.5,
    # Optional PM4Sand secondary parameters (-1 = use default)
    P_atm=101.325,
    h0=-1.0,
    e_max=0.8,
    e_min=0.5,
    nb=0.5,
    nd=0.1,
    Ado=-1.0,
    z_max=-1.0,
    c_z=250.0,
    c_e=-1.0,
    phi_cv=33.0,
    nu=None,
    g_degr=2.0,
    c_dr=-1.0,
    c_kaf=-1.0,
    Q=10.0,
    R=1.5,
    m_par=0.01,
    F_sed=-1.0,
    p_sed=-1.0,
    # Analysis control
    n_cycles=30,
    ru_threshold=0.95,
    strain_increment=5.0e-6,
    dev_disp_limit=0.03,
):
    """Run PM4Sand undrained cyclic direct simple shear analysis.

    Parameters
    ----------
    Dr : float
        Relative density (0 to 1, e.g. 0.55).
    G0 : float
        Shear modulus coefficient (dimensionless, typically 400-900).
    hpo : float
        Contraction rate parameter (typically 0.05-2.0).
    Den : float
        Mass density (Mg/m3, e.g. 1.7).
    sigma_v : float
        Initial vertical effective stress (kPa). Default 100.
    CSR : float
        Cyclic stress ratio to apply. Default 0.15.
    K0 : float
        Coefficient of lateral earth pressure. Default 0.5.
    P_atm : float
        Atmospheric pressure (kPa). Default 101.325.
    h0 through p_sed : float
        PM4Sand secondary parameters. Use -1 for material defaults.
    n_cycles : int
        Maximum number of loading cycles. Default 30.
    ru_threshold : float
        Pore pressure ratio defining liquefaction. Default 0.95.
    strain_increment : float
        Displacement increment per analysis step (m). Default 5e-6.
    dev_disp_limit : float
        Max deviatoric displacement before stopping a half-cycle (m).
        Default 0.03.

    Returns
    -------
    PM4SandDSSResult
        Analysis results with time histories and summary.

    Raises
    ------
    ValueError
        For invalid input parameters.
    ImportError
        If openseespy is not installed.
    """
    _validate_pm4sand_inputs(Dr, G0, hpo, Den, sigma_v, CSR, K0, n_cycles)

    # Compute nu from K0 if not provided
    if nu is None:
        nu = K0 / (1.0 + K0)

    # Initial void ratio
    e_ini = e_max - (e_max - e_min) * Dr

    # Stress sign convention: OpenSees uses negative for compression
    sig_v0 = -sigma_v  # kPa (negative = compression)
    target_shear = CSR * sigma_v  # positive shear stress target

    from opensees_agent.opensees_utils import fresh_model
    ops = fresh_model(ndm=2, ndf=3)

    try:
        return _run_pm4sand_model(
            ops, Dr, G0, hpo, Den, P_atm, h0, e_max, e_min,
            nb, nd, Ado, z_max, c_z, c_e, phi_cv, nu, g_degr,
            c_dr, c_kaf, Q, R, m_par, F_sed, p_sed,
            sigma_v, sig_v0, K0, CSR, target_shear, e_ini,
            n_cycles, ru_threshold, strain_increment, dev_disp_limit,
        )
    finally:
        ops.wipe()


def _run_pm4sand_model(
    ops, Dr, G0, hpo, Den, P_atm, h0, e_max, e_min,
    nb, nd, Ado, z_max, c_z, c_e, phi_cv, nu, g_degr,
    c_dr, c_kaf, Q, R, m_par, F_sed, p_sed,
    sigma_v, sig_v0, K0, CSR, target_shear, e_ini,
    n_cycles, ru_threshold, strain_increment, dev_disp_limit,
):
    """Internal: build model, run analysis, extract results."""
    perm = 1.0e-9  # very low permeability for undrained

    # ── Nodes (unit square element) ──
    ops.node(1, 0.0, 0.0)
    ops.node(2, 1.0, 0.0)
    ops.node(3, 1.0, 1.0)
    ops.node(4, 0.0, 1.0)

    # ── Boundary conditions ──
    ops.fix(1, 1, 1, 1)  # bottom nodes fixed
    ops.fix(2, 1, 1, 1)
    ops.fix(3, 0, 0, 1)  # top nodes: free x,y; fixed pore pressure
    ops.fix(4, 0, 0, 1)
    ops.equalDOF(3, 4, 1, 2)  # tie top nodes together

    # ── PM4Sand material ──
    ops.nDMaterial('PM4Sand', 1, Dr, G0, hpo, Den,
                   P_atm, h0, e_max, e_min, nb, nd, Ado,
                   z_max, c_z, c_e, phi_cv, nu, g_degr,
                   c_dr, c_kaf, Q, R, m_par, F_sed, p_sed)

    # ── Element ──
    ops.element('SSPquadUP', 1, 1, 2, 3, 4, 1,
                1.0, 2.2e6, 1.0, perm, perm, e_ini, 1.0e-5)

    # ── Rayleigh damping ──
    damp = 0.02
    omega1, omega2 = 0.2, 20.0
    a1 = 2.0 * damp / (omega1 + omega2)
    a0 = a1 * omega1 * omega2

    # ── Analysis configuration ──
    ops.constraints('Transformation')
    ops.test('NormDispIncr', 1.0e-5, 35, 0)
    ops.algorithm('Newton')
    ops.numberer('RCM')
    ops.system('FullGeneral')
    ops.integrator('Newmark', 5.0 / 6.0, 4.0 / 9.0)
    ops.rayleigh(a1, a0, 0.0, 0.0)
    ops.analysis('Transient')

    # ═════════════════════════════════════════════════════════════
    # Phase 1: Consolidation (elastic stage)
    # ═════════════════════════════════════════════════════════════
    p_node = sig_v0 / 2.0  # split vertical load between 2 top nodes
    ops.timeSeries('Path', 1, '-values', 0, 1, 1,
                   '-time', 0.0, 100.0, 1.0e10)
    ops.pattern('Plain', 1, 1, '-factor', 1.0)
    ops.load(3, 0.0, p_node, 0.0)
    ops.load(4, 0.0, p_node, 0.0)
    ops.updateMaterialStage('-material', 1, '-stage', 0)
    ops.analyze(100, 1.0)

    v_disp = ops.nodeDisp(3, 2)

    # Apply vertical displacement constraint to hold stress
    ops.timeSeries('Path', 2, '-values', 1.0, 1.0, 1.0,
                   '-time', 100.0, 80000.0, 1.0e10, '-factor', 1.0)
    ops.pattern('Plain', 2, 2, '-factor', 1.0)
    ops.sp(3, 2, v_disp)
    ops.sp(4, 2, v_disp)

    # Close drainage (remove pore pressure fixity)
    for i in range(1, 5):
        ops.remove('sp', i, 3)

    ops.analyze(25, 1.0)

    # Switch to elastoplastic stage
    ops.updateMaterialStage('-material', 1, '-stage', 1)
    ops.setParameter('-val', 0, '-ele', 1, 'FirstCall', '1')
    ops.analyze(25, 1.0)

    # Update Poisson's ratio
    ops.setParameter('-val', nu, '-ele', 1, 'poissonRatio', '1')

    # ═════════════════════════════════════════════════════════════
    # Phase 2: Stress-controlled cyclic loading
    # ═════════════════════════════════════════════════════════════
    control_disp = 1.1 * dev_disp_limit
    num_cycle = 0.0
    ts_tag = 3
    pat_tag = 3

    # Storage for time histories
    times = []
    stresses_v = []
    stresses_shear = []
    strains_shear = []

    def _record():
        """Capture current state."""
        t = ops.getTime()
        stress = ops.eleResponse(1, 'stress')
        strain = ops.eleResponse(1, 'strain')
        times.append(t)
        stresses_v.append(-stress[1])    # σ'v (make positive)
        stresses_shear.append(stress[2])  # τ
        strains_shear.append(strain[2] * 100.0)  # convert to %

    _record()  # initial state

    liquefied = False
    liq_cycle = float('inf')

    while num_cycle < n_cycles:
        # ── Quarter-cycle 1: load positive ──
        h_disp = ops.nodeDisp(3, 1)
        cur_time = ops.getTime()
        steps = control_disp / strain_increment
        time_end = cur_time + steps

        ops.timeSeries('Path', ts_tag, '-values', h_disp, control_disp,
                       control_disp, '-time', cur_time, time_end, 1.0e10,
                       '-factor', 1.0)
        ops.pattern('Plain', pat_tag, ts_tag, '-fact', 1.0)
        ops.sp(3, 1, 1.0)

        b = ops.eleResponse(1, 'stress')
        while b[2] <= target_shear:
            ok = ops.analyze(1, 1.0)
            if ok != 0:
                break
            b = ops.eleResponse(1, 'stress')
            _record()
            h_disp = ops.nodeDisp(3, 1)
            if h_disp >= dev_disp_limit:
                break
            # Check liquefaction
            if stresses_v[-1] > 0 and not liquefied:
                ru_now = 1.0 - stresses_v[-1] / sigma_v
                if ru_now >= ru_threshold:
                    liquefied = True
                    liq_cycle = num_cycle + 0.25

        num_cycle += 0.25
        if liquefied:
            break

        # ── Half-cycle: unload to negative ──
        h_disp = ops.nodeDisp(3, 1)
        cur_time = ops.getTime()
        ops.remove('loadPattern', pat_tag)
        ops.remove('timeSeries', ts_tag)
        ops.remove('sp', 3, 1)

        steps = (control_disp + abs(h_disp)) / strain_increment
        time_end = cur_time + steps

        ops.timeSeries('Path', ts_tag, '-values', h_disp, -control_disp,
                       -control_disp, '-time', cur_time, time_end, 1.0e10,
                       '-factor', 1.0)
        ops.pattern('Plain', pat_tag, ts_tag)
        ops.sp(3, 1, 1.0)

        b = ops.eleResponse(1, 'stress')
        while b[2] > -target_shear:
            ok = ops.analyze(1, 1.0)
            if ok != 0:
                break
            b = ops.eleResponse(1, 'stress')
            _record()
            h_disp = ops.nodeDisp(3, 1)
            if h_disp <= -dev_disp_limit:
                break
            if stresses_v[-1] > 0 and not liquefied:
                ru_now = 1.0 - stresses_v[-1] / sigma_v
                if ru_now >= ru_threshold:
                    liquefied = True
                    liq_cycle = num_cycle + 0.5

        num_cycle += 0.5
        if liquefied:
            break

        # ── Quarter-cycle 3: reload to zero ──
        h_disp = ops.nodeDisp(3, 1)
        cur_time = ops.getTime()
        ops.remove('loadPattern', pat_tag)
        ops.remove('timeSeries', ts_tag)
        ops.remove('sp', 3, 1)

        steps = (control_disp + abs(h_disp)) / strain_increment
        time_end = cur_time + steps

        ops.timeSeries('Path', ts_tag, '-values', h_disp, control_disp,
                       control_disp, '-time', cur_time, time_end, 1.0e10,
                       '-factor', 1.0)
        ops.pattern('Plain', pat_tag, ts_tag, '-fact', 1.0)
        ops.sp(3, 1, 1.0)

        b = ops.eleResponse(1, 'stress')
        while b[2] <= 0.0:
            ok = ops.analyze(1, 1.0)
            if ok != 0:
                break
            b = ops.eleResponse(1, 'stress')
            _record()
            h_disp = ops.nodeDisp(3, 1)
            if h_disp >= dev_disp_limit:
                break
            if stresses_v[-1] > 0 and not liquefied:
                ru_now = 1.0 - stresses_v[-1] / sigma_v
                if ru_now >= ru_threshold:
                    liquefied = True
                    liq_cycle = num_cycle + 0.25

        num_cycle += 0.25
        if liquefied:
            break

        ops.remove('loadPattern', pat_tag)
        ops.remove('timeSeries', ts_tag)
        ops.remove('sp', 3, 1)

    # ═════════════════════════════════════════════════════════════
    # Build result
    # ═════════════════════════════════════════════════════════════
    time_arr = np.array(times)
    sv_arr = np.array(stresses_v)
    tau_arr = np.array(stresses_shear)
    gamma_arr = np.array(strains_shear)

    # Compute ru array
    ru_arr = np.where(sv_arr > 0, 1.0 - sv_arr / sigma_v, 0.0)
    ru_arr = np.clip(ru_arr, 0.0, 1.0)

    return PM4SandDSSResult(
        Dr=Dr,
        sigma_v_kPa=sigma_v,
        CSR_applied=CSR,
        K0=K0,
        n_cycles_to_liq=liq_cycle if liquefied else float('inf'),
        liquefied=liquefied,
        max_ru=float(np.max(ru_arr)) if len(ru_arr) > 0 else 0.0,
        max_shear_strain_pct=float(np.max(np.abs(gamma_arr))) if len(gamma_arr) > 0 else 0.0,
        time=time_arr,
        shear_stress_kPa=tau_arr,
        shear_strain_pct=gamma_arr,
        vert_eff_stress_kPa=sv_arr,
        ru=ru_arr,
    )
