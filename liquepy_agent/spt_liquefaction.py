"""
SPT-based liquefaction triggering analysis using the Boulanger & Idriss (2014)
procedure.

liquepy does NOT ship a packaged SPT triggering object the way it ships
``run_bi2014`` for CPT — its only out-of-the-box SPT entry points are *field
correlations* (Vs, Dr, G0 from N), not triggering. However, liquepy DOES expose
every B&I-2014 SPT *building block* as tested module-level functions in
``liquepy.trigger.boulanger_and_idriss_2014``:

    calc_rd(depth, magnitude)              # Eq 2.14a stress-reduction factor
    calc_csr(sigma_veff, sigma_v, pga, rd) # Eq 2.2  cyclic stress ratio
    calc_crr_m7p5_from_n1_60cs(ncs, c_0)   # B&I-2014 SPT CRR_M7.5 curve
    calc_k_sigma_w_n1_60cs(sigma', ncs)    # Eq 2.16a overburden correction

This module assembles those into a full SPT triggering evaluation. The only
piece liquepy lacks for SPT is the fines correction Δ(N1)60 (it ships the CPT
``calc_delta_q_c1n`` but no SPT analogue), so the B&I-2014 SPT fines equation
(Eq 2.23) and the SPT MSF (Eq 2.20/2.21, msf_max from (N1)60cs) are implemented
here on top of liquepy's CRR/rd/CSR/K_sigma. This keeps the numerically
load-bearing curve fits in liquepy's tested code.

All units are SI: kPa, m, g (acceleration).

References
----------
Boulanger & Idriss (2014), "CPT and SPT based liquefaction triggering
    procedures", Report UCD/CGM-14/01. Eqs 2.2, 2.14a, 2.16a, 2.19/2.20/2.21,
    2.23, 2.24/2.25b.
"""

import math

import numpy as np

from liquepy_agent.liquepy_utils import import_liquepy_trigger
from liquepy_agent.results import SPTLiquefactionResult


def bi2014_spt_fines_correction(n1_60: float, fc: float) -> float:
    """Clean-sand equivalent (N1)60cs from (N1)60 and fines content (B&I 2014).

    Δ(N1)60 = exp(1.63 + 9.7/(FC+0.01) − (15.7/(FC+0.01))²)   (Eq 2.23)
    (N1)60cs = (N1)60 + Δ(N1)60

    Parameters
    ----------
    n1_60 : float
        Corrected SPT blow count (N1)60.
    fc : float
        Fines content (% passing #200), clipped to [0, 100].

    Returns
    -------
    float
        Clean-sand equivalent (N1)60cs.
    """
    fc = min(max(fc, 0.0), 100.0)
    delta = math.exp(1.63 + 9.7 / (fc + 0.01) - (15.7 / (fc + 0.01)) ** 2)
    return n1_60 + delta


def bi2014_spt_msf(magnitude: float, n1_60cs: float) -> float:
    """Magnitude scaling factor for the SPT B&I-2014 procedure.

    MSF_max = 1.09 + ((N1)60cs/31.5)²,  capped at 2.2  (Eq 2.21, SPT form)
    MSF = 1 + (MSF_max − 1)(8.64·e^(−M/4) − 1.325)       (Eq 2.20)

    Mirrors liquepy's ``calc_msf`` (which is written in the CPT q_c1n_cs form);
    only the msf_max correlation differs between CPT and SPT.

    Returns exactly 1.0 at M = 7.5 (the reference magnitude), matching liquepy's
    ``calc_msf`` convention — the closed form is ~1 there but not bit-exact.
    """
    if magnitude == 7.5:
        return 1.0
    msf_max = 1.09 + (n1_60cs / 31.5) ** 2
    msf_max = min(msf_max, 2.2)
    return 1.0 + (msf_max - 1.0) * (8.64 * math.exp(-magnitude / 4.0) - 1.325)


def _validate_spt_inputs(depth, n1_60, fc, gamma, amax_g, gwt_depth, m_w):
    n = len(depth)
    if n == 0:
        raise ValueError("depth array is required and must be non-empty")
    for name, arr in (("n1_60", n1_60), ("fc", fc), ("gamma", gamma)):
        if len(arr) != n:
            raise ValueError(
                f"{name} length ({len(arr)}) must match depth length ({n})"
            )
    if amax_g <= 0:
        raise ValueError(f"amax_g must be positive, got {amax_g}")
    if m_w <= 0:
        raise ValueError(f"m_w must be positive, got {m_w}")
    if gwt_depth < 0:
        raise ValueError(f"gwt_depth must be >= 0, got {gwt_depth}")


def analyze_spt_liquefaction(
    depth,
    n1_60,
    fc,
    gamma,
    amax_g,
    gwt_depth,
    m_w=7.5,
    c_0=2.8,
    gamma_w=9.81,
    p_a=101.0,
) -> SPTLiquefactionResult:
    """Run SPT-based liquefaction triggering (Boulanger & Idriss 2014).

    FoS = (CRR_M7.5 · MSF · K_sigma) / CSR, per layer.

    Parameters
    ----------
    depth : array_like
        Mid-depth of each evaluation layer (m).
    n1_60 : array_like
        Corrected SPT blow count (N1)60 at each depth.
    fc : array_like
        Fines content (%) at each depth.
    gamma : array_like
        Total unit weight (kN/m3) at each depth.
    amax_g : float
        Peak ground acceleration (fraction of g).
    gwt_depth : float
        Depth to groundwater table (m below surface).
    m_w : float, optional
        Moment magnitude. Default 7.5.
    c_0 : float, optional
        CRR curve fitting parameter (2.8 = 16th-percentile/deterministic,
        2.6 = median). Default 2.8.
    gamma_w : float, optional
        Unit weight of water (kN/m3). Default 9.81.
    p_a : float, optional
        Atmospheric pressure (kPa). Default 101.0.

    Returns
    -------
    SPTLiquefactionResult
        Per-layer triggering result.
    """
    depth = np.asarray(depth, dtype=float)
    n1_60 = np.asarray(n1_60, dtype=float)
    fc = np.asarray(fc, dtype=float)
    gamma = np.asarray(gamma, dtype=float)

    _validate_spt_inputs(depth, n1_60, fc, gamma, amax_g, gwt_depth, m_w)

    bi = import_liquepy_trigger().boulanger_and_idriss_2014

    n = len(depth)
    n1_60cs = np.zeros(n)
    sigma_v = np.zeros(n)
    sigma_veff = np.zeros(n)
    csr = np.zeros(n)
    crr = np.zeros(n)
    fos = np.zeros(n)

    for i in range(n):
        z = depth[i]

        # Total stress (gamma * z is a simple approximation when only the
        # mid-depth gamma is supplied — matches seismic_geotech's convention).
        sv = gamma[i] * z
        if z <= gwt_depth:
            sve = sv
        else:
            sve = sv - gamma_w * (z - gwt_depth)
        sve = max(sve, 1.0)  # guard against division by zero
        sigma_v[i] = sv
        sigma_veff[i] = sve

        # Clean-sand equivalent blow count (B&I fines correction).
        ncs = bi2014_spt_fines_correction(float(n1_60[i]), float(fc[i]))
        n1_60cs[i] = ncs

        # CSR (B&I rd + cyclic stress ratio, from liquepy's tested functions).
        rd = float(bi.calc_rd(z, m_w))
        csr[i] = float(bi.calc_csr(sigma_veff=sve, sigma_v=sv, pga=amax_g, rd=rd))

        # CRR_M7.5 (liquepy's tested B&I SPT curve), then magnitude + overburden
        # corrections per B&I-2014.
        crr_m7p5 = float(bi.calc_crr_m7p5_from_n1_60cs(ncs, c_0=c_0))
        msf = bi2014_spt_msf(m_w, ncs)
        k_sigma = float(bi.calc_k_sigma_w_n1_60cs(sve, ncs, pa=p_a))
        crr[i] = crr_m7p5 * msf * k_sigma

        fos[i] = crr[i] / csr[i] if csr[i] > 0 else 99.9

    liquefiable = fos < 1.0
    min_fos = float(np.min(fos)) if n > 0 else 0.0
    n_liq = int(np.sum(liquefiable))

    return SPTLiquefactionResult(
        n_layers=n,
        gwt_depth_m=float(gwt_depth),
        amax_g=float(amax_g),
        m_w=float(m_w),
        min_fos=min_fos,
        n_liquefiable=n_liq,
        depth=depth,
        n1_60=n1_60,
        n1_60cs=n1_60cs,
        fines_content=fc,
        sigma_v=sigma_v,
        sigma_veff=sigma_veff,
        csr=csr,
        crr=crr,
        factor_of_safety=fos,
        liquefiable=liquefiable,
    )
