"""
CPT-based field correlations using liquepy.

Computes shear wave velocity, relative density, undrained strength ratio,
and permeability from CPT data using published correlations.
"""

import numpy as np

from liquepy_agent.liquepy_utils import import_liquepy_trigger, import_liquepy_field
from liquepy_agent.results import FieldCorrelationsResult


_VS_METHODS = {"mcgann2015", "robertson2009", "andrus2007"}


def _validate_correlation_inputs(depth, q_c, f_s, u_2, gwl, vs_method):
    """Validate field correlation inputs.

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
    if vs_method not in _VS_METHODS:
        raise ValueError(
            f"vs_method must be one of {sorted(_VS_METHODS)}, got '{vs_method}'"
        )


def analyze_field_correlations(
    depth,
    q_c,
    f_s,
    u_2=None,
    gwl=1.0,
    a_ratio=0.8,
    vs_method="mcgann2015",
    i_c_limit=2.6,
    p_a=101.0,
    s_g=2.65,
    gamma_predrill=17.0,
) -> FieldCorrelationsResult:
    """Compute field correlations from CPT data.

    Uses liquepy's built-in correlations for shear wave velocity,
    relative density, undrained strength ratio, and permeability.

    Parameters
    ----------
    depth : array_like
        Depth from surface (m).
    q_c : array_like
        Cone tip resistance (kPa).
    f_s : array_like
        Sleeve friction (kPa).
    u_2 : array_like, optional
        Pore pressure behind cone tip (kPa). Default: zeros.
    gwl : float, optional
        Groundwater level depth (m below surface). Default: 1.0.
    a_ratio : float, optional
        Cone area ratio. Default: 0.8.
    vs_method : str, optional
        Vs correlation: 'mcgann2015', 'robertson2009', or 'andrus2007'.
        Default: 'mcgann2015'.
    i_c_limit : float, optional
        I_c limit for soil classification. Default: 2.6.
    p_a : float, optional
        Atmospheric pressure (kPa). Default: 101.0.
    s_g : float, optional
        Specific gravity of solids. Default: 2.65.
    gamma_predrill : float, optional
        Unit weight above pre-drill (kN/m3). Default: 17.0.

    Returns
    -------
    FieldCorrelationsResult
        Result with Vs, Dr, su/σv', permeability, and Ic profiles.
    """
    depth = np.asarray(depth, dtype=float)
    q_c = np.asarray(q_c, dtype=float)
    f_s = np.asarray(f_s, dtype=float)
    if u_2 is None:
        u_2 = np.zeros_like(depth)
    else:
        u_2 = np.asarray(u_2, dtype=float)

    _validate_correlation_inputs(depth, q_c, f_s, u_2, gwl, vs_method)

    trigger = import_liquepy_trigger()
    field_mod = import_liquepy_field()

    # Create CPT and run B&I 2014 (needed for Ic, q_c1n, sigma_veff)
    cpt = field_mod.CPT(depth, q_c, f_s, u_2, gwl, a_ratio=a_ratio)
    bi = trigger.run_bi2014(
        cpt, pga=0.25, m_w=7.5, gwl=gwl,
        p_a=p_a, i_c_limit=i_c_limit,
        gamma_predrill=gamma_predrill, s_g=s_g,
    )

    i_c_arr = np.asarray(bi.i_c, dtype=float)
    q_c1n_arr = np.asarray(bi.q_c1n, dtype=float)
    sigma_veff_arr = np.asarray(bi.sigma_veff, dtype=float)
    q_t_arr = np.asarray(bi.q_t, dtype=float)
    big_q_arr = np.asarray(bi.big_q, dtype=float)

    # Import correlations
    from liquepy.field import correlations as corr

    # Relative density (Boulanger et al. 2014)
    dr = np.asarray(
        corr.calc_relative_density_boulanger_et_al_2014_cpt_values(q_c1n_arr),
        dtype=float,
    )

    # Undrained strength ratio (Robertson 2009)
    su_ratio = np.asarray(
        corr.est_undrained_strength_ratio_robertson_2009(big_q_arr),
        dtype=float,
    )

    # Permeability (Robertson & Cabal 2012)
    perm = np.asarray(
        corr.est_permeability_robertson_and_cabal_2012(i_c_arr),
        dtype=float,
    )

    # Shear wave velocity
    if vs_method == "mcgann2015":
        vs = np.asarray(
            corr.calc_shear_vel_mcgann_2015_cpt(cpt),
            dtype=float,
        )
    elif vs_method == "robertson2009":
        # Robertson 2009 uses Pa units (Pa, not kPa)
        vs = np.asarray(
            corr.calc_shear_vel_robertson_2009_cpt(
                i_c_arr, sigma_veff_arr * 1000, q_t_arr * 1000, p_atm=p_a * 1000,
            ),
            dtype=float,
        )
    elif vs_method == "andrus2007":
        vs = np.asarray(
            corr.calc_shear_vel_andrus_et_al_2007_cpt(
                i_c_arr, depth, q_t_arr,
            ),
            dtype=float,
        )
    else:
        vs = np.full_like(depth, np.nan)

    return FieldCorrelationsResult(
        n_points=len(depth),
        gwl_m=float(gwl),
        vs_method=vs_method,
        depth=depth,
        vs_m_per_s=vs,
        relative_density=dr,
        su_ratio=su_ratio,
        permeability_cm_per_s=perm,
        i_c=i_c_arr,
    )
