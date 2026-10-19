"""
CPT-based liquefaction triggering analysis using liquepy.

Implements the Boulanger & Idriss (2014) CPT-based simplified procedure
with post-triggering strain and index calculations.
"""

import math

import numpy as np

from liquepy_agent.liquepy_utils import import_liquepy_trigger, import_liquepy_field
from liquepy_agent.results import CPTLiquefactionResult

#: Why a run without a water table is refused (BACKLOG GM-133): until
#: 2026-10-17 the module took 1.0 m without saying so.
WATER_TABLE_REQUIRED = (
    "gwl is required: the depth to the water table (m below the ground "
    "surface) during the design earthquake. It sets the effective stress in "
    "CSR and which points are saturated and can liquefy, so there is no "
    "default: the references take the groundwater level as a site input "
    "(GEC 5, FHWA-NHI-16-072, p. 12-23: \"establish the location of the "
    "groundwater level\"). Give the measured or design level; if the site "
    "data give none, ask, or state the level you assume and give it.")

#: Why a run without a peak ground acceleration is refused (GM-133).
PGA_REQUIRED = (
    "pga is required: the peak ground acceleration (g) at the ground "
    "surface for the design earthquake. It sets CSR, so there is no "
    "default; it comes from the site's seismic hazard. Give the design "
    "value; if none is given, ask, or state the value you assume and give "
    "it.")


#: Most rows :func:`depth_profile` gives (BACKLOG GM-201). A CPT has a
#: reading every few centimetres, hundreds over a pile's length, and the
#: agent's tool results are kept under 8000 characters
#: (``funhouse_agent/deep/tools.py`` DEFAULT_MAX_RESULT_CHARS); a longer
#: CPT comes in depth bands.
PROFILE_MAX_ROWS = 40

#: FS_liq where the bands split: FHWA GEC-10 §9.4.2.2 (p. 9-42), "layers
#: predicted to undergo liquefaction (FSliq ≤ 1.0)".
PROFILE_FS_SPLIT = 1.0


def _fs_row(f: float) -> float:
    """FS to 3 decimals, rounded up, so a reading above
    :data:`PROFILE_FS_SPLIT` never reads as at or below it."""
    return math.ceil(f * 1000.0 - 1e-9) / 1000.0


def depth_profile(depth, factor_of_safety, volumetric_strain,
                  max_rows: int = PROFILE_MAX_ROWS):
    """The CPT's factor of safety against liquefaction and post-liquefaction
    volumetric strain by depth, as rows a tool result can carry (GM-201).

    Each reading stands for the depth halfway to its neighbours (the first
    and last half a spacing beyond their reading, the first not above the
    ground surface), as ``downdrag.liquefied_layers_from_result`` takes a
    profile. Up to ``max_rows`` readings give one row each. A longer CPT
    is summarised in depth bands of whole readings:

    - a band never holds readings on both sides of FS_liq = 1.0 (GEC-10
      §9.4.2.2), so at that threshold the liquefied depths are the
      readings'; only when the readings cross 1.0 more often than there
      are rows are the thinnest runs counted with the thicker soil around
      them (the note gives the thickness);
    - the bands are about equally thick within that;
    - ``FOS_liq`` is the band's lowest reading on its side of 1.0, and
      ``volumetric_strain`` the band's average (strain x thickness = the
      band's reconsolidation settlement).

    Parameters
    ----------
    depth, factor_of_safety, volumetric_strain : array_like
        The readings (m; FS_liq; decimal strain), as
        :func:`analyze_cpt_liquefaction` returns them.
    max_rows : int
        The most rows to give.

    Returns
    -------
    (rows, note) : (list of dict, str or None)
        Rows of ``top_m``, ``bottom_m``, ``FOS_liq`` and
        ``volumetric_strain``, in depth order; ``note`` says how the bands
        were made (None for one row per reading).
    """
    z = np.asarray(depth, dtype=float).ravel()
    fs = np.asarray(factor_of_safety, dtype=float).ravel()
    ev = np.asarray(volumetric_strain, dtype=float).ravel()
    if not len(z) == len(fs) == len(ev):
        raise ValueError("depth, factor_of_safety and volumetric_strain "
                         "must be the same length")
    n = len(z)
    if n < 2:
        raise ValueError("a depth profile needs at least two readings")
    order = np.argsort(z, kind="stable")
    z, fs, ev = z[order], fs[order], ev[order]
    edges = np.empty(n + 1)
    edges[0] = max(0.0, z[0] - 0.5 * (z[1] - z[0]))
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    thick = np.diff(edges)
    liq = np.isfinite(fs) & (fs <= PROFILE_FS_SPLIT)

    def row(i0, i1, liquefied):
        th = thick[i0:i1]
        own = liq[i0:i1] == liquefied
        f = float(np.min(fs[i0:i1][own] if own.any() else fs[i0:i1]))
        e = (float(np.sum(ev[i0:i1] * th) / np.sum(th)) if np.sum(th) > 0
             else float(ev[i0]))
        # Zhang's curves interpolate a hair below zero where FS is capped
        return {"top_m": round(float(edges[i0]), 2),
                "bottom_m": round(float(edges[i1]), 2),
                "FOS_liq": _fs_row(f),
                "volumetric_strain": round(max(e, 0.0), 5) + 0.0}

    if n <= max_rows:
        return [row(i, i + 1, bool(liq[i])) for i in range(n)], None

    # runs of readings on one side of FS_liq = 1.0
    runs, i0 = [], 0
    for i in range(1, n + 1):
        if i == n or liq[i] != liq[i0]:
            runs.append([i0, i, bool(liq[i0])])
            i0 = i

    def size(r):
        return edges[r[1]] - edges[r[0]]

    # more crossings than rows: the thinnest run goes with the soil around
    # it (its neighbours are the other kind, so the three become one)
    while len(runs) > max_rows:
        k = min(range(len(runs)), key=lambda j: size(runs[j]))
        lo, hi = max(k - 1, 0), min(k + 1, len(runs) - 1)
        kind = runs[k - 1][2] if k > 0 else runs[k + 1][2]
        runs[lo:hi + 1] = [[runs[lo][0], runs[hi][1], kind]]
    # then the rows left split the runs into bands of about equal thickness
    pieces = [1] * len(runs)
    while sum(pieces) < max_rows:
        can = [j for j, r in enumerate(runs) if r[1] - r[0] > pieces[j]]
        if not can:
            break
        j = max(can, key=lambda j: size(runs[j]) / pieces[j])
        pieces[j] += 1
    rows = []
    for (a, b, kind), m in zip(runs, pieces):
        for chunk in np.array_split(np.arange(a, b), m):
            rows.append(row(int(chunk[0]), int(chunk[-1]) + 1, kind))
    widest = max(size(r) / m for r, m in zip(runs, pieces))
    split = f"{PROFILE_FS_SPLIT:.1f}"
    note = (
        f"{n} readings summarised in {len(rows)} depth bands, up to about "
        f"{widest:.2g} m thick, to keep the result small. FOS_liq is the "
        f"band's lowest reading on its side of FS_liq = {split}; "
        f"volumetric_strain is the band's average, so strain x thickness "
        f"is its settlement.")
    lenses = [float(np.sum(thick[a:b][liq[a:b] != kind]))
              for a, b, kind in runs]
    wet = sum(x for x, r in zip(lenses, runs) if not r[2])
    dry = sum(x for x, r in zip(lenses, runs) if r[2])
    if wet + dry > 0:
        note += (
            f" The readings cross {split} more often than there are rows, "
            f"so the thinnest runs are counted with the soil around them: "
            f"{wet:.2f} m with FS_liq <= {split} inside bands above it, "
            f"{dry:.2f} m above it inside liquefied bands; elsewhere no band "
            f"holds readings on both sides of {split} (GEC-10 §9.4.2.2).")
    else:
        note += (
            f" No band holds readings on both sides of {split} (GEC-10 "
            f"§9.4.2.2), so at that threshold the liquefied depths are the "
            f"readings' own.")
    return rows, note


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

    if gwl is None:
        raise ValueError(WATER_TABLE_REQUIRED)
    if pga is None:
        raise ValueError(PGA_REQUIRED)
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
    gwl=None,
    pga=None,
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
    gwl : float
        Depth to the water table during the design earthquake (m below
        surface). Required: a site input, with no default
        (:data:`WATER_TABLE_REQUIRED` says why; 1.0 m until 2026-10-17).
    pga : float
        Peak ground acceleration (g). Required (:data:`PGA_REQUIRED`;
        0.25 until 2026-10-17).
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
        choices=cpt_choices(i_c_limit, cfc, gwl),
    )


def cpt_choices(i_c_limit: float = 2.6, cfc: float = 0.0,
                gwl: float = None) -> list:
    """What :func:`analyze_cpt_liquefaction` chose (GM-115), as a result's
    ``choices`` (``geotech_common.choices``); the water table second, as
    given (GM-133: it is required). The procedure is this function's own,
    not an input, so it is the module's; the water table is as the call
    gave it (GM-180)."""
    from geotech_common.choices import MODULE, choice, how_given
    out = [
        choice("triggering procedure",
               "Boulanger & Idriss (2014) CPT (liquepy run_bi2014)", MODULE,
               f"rd, MSF and K_sigma per B&I (2014); fines content from Ic "
               f"with C_FC = {cfc:g}"),
    ]
    if gwl is not None:
        out.append(choice(
            "water table", f"{gwl:g} m below the ground surface",
            how_given(gwl, None, name="gwl"),
            "the depth to groundwater during the design earthquake; it "
            "sets sigma'v in CSR and which points can liquefy. A site "
            "input with no default (GEC 5, FHWA-NHI-16-072, p. 12-23)"))
    return out + [
        choice("CRR curve C0", "2.8", MODULE,
               "B&I (2014)'s deterministic CPT curve, liquepy's value; this "
               "route does not take C0"),
        choice("points counted as liquefiable",
               f"below the water table with Ic < {i_c_limit:g}",
               how_given(i_c_limit, 2.6, name="i_c_limit"),
               "the minimum FS is taken over these points (min_fos)"),
        choice("post-triggering strains",
               "volumetric Zhang et al. (2002), shear Zhang et al. (2004)",
               MODULE, "LSN and the settlement from the volumetric strain, "
               "LDI from the shear strain"),
    ]
