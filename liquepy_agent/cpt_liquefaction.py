"""
CPT-based liquefaction triggering analysis using liquepy.

Implements the Boulanger & Idriss (2014) CPT-based simplified procedure
with post-triggering strain and index calculations.
"""

import numpy as np

from liquepy_agent.liquepy_utils import import_liquepy_trigger, import_liquepy_field
from liquepy_agent.results import CPTLiquefactionResult


def _validate_cpt_inputs(depth, q_c, f_s, u_2, gwl, pga, m_w):
    """Validate CPT liquefaction analysis inputs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if depth is None or len(depth) == 0:
        raise ValueError("depth array is required and must be non-empty")
    if q_c is None or len(q_c) == 0:
        raise ValueError("q_c array is required and must be non-empty")
    if f_s is None or len(f_s) == 0:
        raise ValueError("f_s array is required and must be non-empty")

    n = len(depth)
    if len(q_c) != n:
        raise ValueError(f"q_c length ({len(q_c)}) must match depth length ({n})")
    if len(f_s) != n:
        raise ValueError(f"f_s length ({len(f_s)}) must match depth length ({n})")
    if u_2 is not None and len(u_2) != n:
        raise ValueError(f"u_2 length ({len(u_2)}) must match depth length ({n})")

    if gwl < 0:
        raise ValueError(f"gwl must be >= 0, got {gwl}")
    if pga <= 0:
        raise ValueError(f"pga must be positive, got {pga}")
    if m_w <= 0:
        raise ValueError(f"m_w must be positive, got {m_w}")


def _calc_ldi_safe(shear_strain, depth, z_max=None):
    """Calculate Lateral Displacement Index using np.trapezoid.

    This is a safe replacement for liquepy.trigger.calc_ldi which uses
    the removed np.trapz on numpy >= 2.0.

    Parameters
    ----------
    shear_strain : ndarray
        Shear strain (decimal).
    depth : ndarray
        Depth array (m).
    z_max : float, optional
        Maximum integration depth.

    Returns
    -------
    float
        Lateral Displacement Index (m).
    """
    if z_max is not None:
        mask = depth <= z_max
        shear_strain = shear_strain[mask]
        depth = depth[mask]

    if len(depth) < 2:
        return 0.0

    return float(np.trapezoid(y=shear_strain, x=depth))


def analyze_cpt_liquefaction(
    depth,
    q_c,
    f_s,
    u_2=None,
    gwl=1.0,
    pga=0.25,
    m_w=7.5,
    a_ratio=0.8,
    i_c_limit=2.6,
    cfc=0.0,
    unit_wt_method="robertson2009",
    gamma_predrill=17.0,
    s_g=2.65,
    p_a=101.0,
) -> CPTLiquefactionResult:
    """Run CPT-based liquefaction triggering analysis (Boulanger & Idriss 2014).

    Performs full triggering analysis and computes post-triggering indices:
    LPI, LSN, LDI, volumetric strain, and shear strain.

    Parameters
    ----------
    depth : array_like
        Depth from surface (m). Must be monotonically increasing.
    q_c : array_like
        Cone tip resistance (kPa).
    f_s : array_like
        Sleeve friction (kPa).
    u_2 : array_like, optional
        Pore pressure behind cone tip (kPa). Default: zeros.
    gwl : float, optional
        Groundwater level depth (m below surface). Default: 1.0.
    pga : float, optional
        Peak ground acceleration (g). Default: 0.25.
    m_w : float, optional
        Moment magnitude. Default: 7.5.
    a_ratio : float, optional
        Cone area ratio. Default: 0.8.
    i_c_limit : float, optional
        I_c limit for liquefiable material. Default: 2.6.
    cfc : float, optional
        Fines content correction factor. Default: 0.0.
    unit_wt_method : str, optional
        Unit weight method: 'robertson2009' or 'void_ratio'. Default: 'robertson2009'.
    gamma_predrill : float, optional
        Unit weight above pre-drill depth (kN/m3). Default: 17.0.
    s_g : float, optional
        Specific gravity of solids. Default: 2.65.
    p_a : float, optional
        Atmospheric pressure (kPa). Default: 101.0.

    Returns
    -------
    CPTLiquefactionResult
        Result with triggering factors of safety, indices, and profiles.
    """
    depth = np.asarray(depth, dtype=float)
    q_c = np.asarray(q_c, dtype=float)
    f_s = np.asarray(f_s, dtype=float)
    if u_2 is None:
        u_2 = np.zeros_like(depth)
    else:
        u_2 = np.asarray(u_2, dtype=float)

    _validate_cpt_inputs(depth, q_c, f_s, u_2, gwl, pga, m_w)

    # Import liquepy components
    trigger = import_liquepy_trigger()
    field_mod = import_liquepy_field()

    # Create CPT object
    cpt = field_mod.CPT(depth, q_c, f_s, u_2, gwl, a_ratio=a_ratio)

    # Run Boulanger & Idriss 2014 triggering
    bi = trigger.run_bi2014(
        cpt, pga=pga, m_w=m_w, gwl=gwl,
        p_a=p_a, cfc=cfc, i_c_limit=i_c_limit,
        gamma_predrill=gamma_predrill,
        unit_wt_method=unit_wt_method,
        s_g=s_g,
    )

    fos = np.asarray(bi.factor_of_safety, dtype=float)
    csr_arr = np.asarray(bi.csr, dtype=float)
    crr_arr = np.asarray(bi.crr, dtype=float)
    q_c1n_cs_arr = np.asarray(bi.q_c1n_cs, dtype=float)
    i_c_arr = np.asarray(bi.i_c, dtype=float)
    fc_arr = np.asarray(bi.fines_content, dtype=float)
    sigma_v_arr = np.asarray(bi.sigma_v, dtype=float)
    sigma_veff_arr = np.asarray(bi.sigma_veff, dtype=float)

    # Post-triggering: volumetric strain (Zhang et al. 2002)
    vol_strain = np.asarray(
        trigger.calc_volumetric_strain_zhang_2002(fos, q_c1n_cs_arr),
        dtype=float,
    )

    # Post-triggering: relative density (Zhang 2002)
    q_c1n_arr = np.asarray(bi.q_c1n, dtype=float)
    dr = np.asarray(
        trigger.calc_relative_density_zhang_2002(q_c1n_arr),
        dtype=float,
    )

    # Post-triggering: shear strain (Zhang et al. 2004)
    shear_strain = np.asarray(
        trigger.calc_shear_strain_zhang_2004(fos, dr),
        dtype=float,
    )

    # LPI (Liquefaction Potential Index)
    lpi_val = float(trigger.calc_lpi(fos, depth))

    # LSN (Liquefaction Severity Number)
    # calc_lsn expects volumetric strain in percentage
    lsn_val = float(trigger.calc_lsn(vol_strain * 100, depth))

    # LDI (Lateral Displacement Index) — use safe version (np.trapezoid)
    ldi_val = _calc_ldi_safe(shear_strain, depth)

    # Min FoS in liquefiable zone (below GWL, Ic < limit)
    liquefiable = (depth >= gwl) & (i_c_arr < i_c_limit)
    if np.any(liquefiable):
        min_fos = float(np.min(fos[liquefiable]))
    else:
        min_fos = float(np.min(fos)) if len(fos) > 0 else 0.0

    # Max settlement (sum of volumetric strain * layer thickness)
    if len(depth) > 1:
        dz = np.diff(depth)
        # Use midpoint strains
        mid_strain = (vol_strain[:-1] + vol_strain[1:]) / 2
        settlement_m = float(np.sum(mid_strain * dz))
        max_settlement_mm = settlement_m * 1000
    else:
        max_settlement_mm = 0.0

    return CPTLiquefactionResult(
        n_points=len(depth),
        gwl_m=float(gwl),
        pga_g=float(pga),
        m_w=float(m_w),
        i_c_limit=float(i_c_limit),
        lpi=lpi_val,
        lsn=lsn_val,
        ldi_m=ldi_val,
        min_fos=min_fos,
        max_settlement_mm=max_settlement_mm,
        depth=depth,
        factor_of_safety=fos,
        csr=csr_arr,
        crr=crr_arr,
        q_c1n_cs=q_c1n_cs_arr,
        i_c=i_c_arr,
        fines_content=fc_arr,
        sigma_v=sigma_v_arr,
        sigma_veff=sigma_veff_arr,
        volumetric_strain=vol_strain,
        shear_strain=shear_strain,
        relative_density=dr,
    )
