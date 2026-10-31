"""
AASHTO seismic site classification, as FHWA GEC-12 Vol I prints it.

Classifies a site as A-E based on:
- Vs30: weighted harmonic mean shear wave velocity in top 30m
- N-bar: weighted average SPT N in top 30m
- su-bar: weighted average undrained shear strength in top 30m

Also provides site coefficients Fpga, Fa, Fv from AASHTO tables.

All units are SI: m/s, kPa, blows/ft.

Checked against the print held (GM-293, module_work/gm293/CHECK.md): the
classes are GEC-12 Vol I (FHWA-NHI-16-009) Table 7-26 (p. 379, after
AASHTO 2014), printed in ft/s and ksf and converted exactly here
(1 ft = 0.3048 m, 1 ksf = 47.88 kPa); the averages are taken over the upper
100 ft (GEC-12 p. 379; GEC-7, FHWA-NHI-14-007, p. 92), 30 m here, as ASCE 7-22 Ch. 20
writes "100 ft (30 m)"; the site factors are GEC-12 Tables 7-27 to 7-29
(p. 380) and As = Fpga PGA its Eq. 7-78 (p. 381). Not checked by this
tool: Site Class E by more than 10 ft of soft clay, and Site Class F
(Table 7-26).

References:
    FHWA GEC-12 Vol I (FHWA-NHI-16-009) Sec. 7.4.2.1, Tables 7-26 to 7-29,
    Eqs. 7-77 to 7-79 and Fig. 7-77 (pp. 379-381), after AASHTO (2014)
    AASHTO LRFD Bridge Design Specifications, Section 3.10.3
"""

import warnings
from typing import List, Optional, Dict, Any, Tuple

from seismic_geotech.results import SiteClassResult


def compute_vs30(layer_thicknesses: List[float],
                 layer_vs: List[float]) -> float:
    """Compute Vs30 (weighted harmonic mean of Vs in top 30m).

    Vs30 = 30 / sum(di / Vsi)

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each layer (m).
    layer_vs : list of float
        Shear wave velocity of each layer (m/s).

    Returns
    -------
    float
        Vs30 (m/s).
    """
    total = 0.0
    depth_used = 0.0
    for h, vs in zip(layer_thicknesses, layer_vs):
        if vs <= 0:
            raise ValueError(f"Shear wave velocity must be positive, got {vs}")
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / vs
        depth_used += h_used

    if depth_used < 30.0:
        warnings.warn(
            f"Profile depth ({depth_used}m) < 30m; Vs30 based on available depth"
        )
    if total <= 0:
        return 0.0
    return depth_used / total


def compute_n_bar(layer_thicknesses: List[float],
                  layer_N: List[float]) -> float:
    """Compute N-bar (weighted average SPT N in top 30m).

    N-bar = 30 / sum(di / Ni)

    Ni is the field blow count, without corrections, limited to 100 blows
    per ft (FHWA GEC-7 p. 93, under its Eq. 4.2: "This value must be as
    measured in the field without corrections. Ni must be limited to 100
    bpf where refusal is encountered"). Until GM-293 no limit was applied.

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each layer (m).
    layer_N : list of float
        Field SPT N-value for each layer (blows/ft); above 100 is taken
        as 100.

    Returns
    -------
    float
        Average N-bar value.
    """
    total = 0.0
    depth_used = 0.0
    for h, N in zip(layer_thicknesses, layer_N):
        if N <= 0:
            raise ValueError(f"SPT N must be positive, got {N}")
        N = min(N, 100.0)
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / N
        depth_used += h_used

    if depth_used < 30.0:
        warnings.warn(
            f"Profile depth ({depth_used}m) < 30m; N-bar based on available depth"
        )
    if total <= 0:
        return 0.0
    return depth_used / total


def compute_su_bar(layer_thicknesses: List[float],
                   layer_su: List[float]) -> float:
    """Compute su-bar (weighted average cu in top 30m of cohesive soil).

    su-bar = sum(di) / sum(di / sui)  (harmonic mean)

    Parameters
    ----------
    layer_thicknesses : list of float
        Thickness of each cohesive layer (m).
    layer_su : list of float
        Undrained shear strength of each layer (kPa).

    Returns
    -------
    float
        Average su-bar (kPa).
    """
    total = 0.0
    depth_used = 0.0
    for h, su in zip(layer_thicknesses, layer_su):
        if su <= 0:
            raise ValueError(f"Undrained strength must be positive, got {su}")
        remaining = 30.0 - depth_used
        if remaining <= 0:
            break
        h_used = min(h, remaining)
        total += h_used / su
        depth_used += h_used

    if total <= 0:
        return 0.0
    return depth_used / total


def classify_site(vs30: float = None,
                  n_bar: float = None,
                  su_bar: float = None) -> str:
    """Classify site per AASHTO, GEC-12 Vol I Table 7-26 (p. 379).

    The boundaries are the printed ones converted exactly: 5,000, 2,500,
    1,200 and 600 ft/s are 1524, 762, 365.76 and 182.88 m/s; 2.0 and 1.0
    ksf are 95.76 and 47.88 kPa. Until GM-293 they were 1500, 760, 360 and
    180 m/s and 100 and 50 kPa, so a site between the two (say Vs30 =
    181 m/s) took the stiffer class.

    Parameters
    ----------
    vs30 : float, optional
        Average shear wave velocity in top 30m (m/s).
    n_bar : float, optional
        Average field SPT N in top 30m.
    su_bar : float, optional
        Average undrained shear strength in top 30m (kPa).

    Returns
    -------
    str
        Site class: "A", "B", "C", "D" or "E". Class F (Table 7-26: peat,
        very high plasticity or very thick soft clay) and Class E by more
        than 10 ft of soft clay are not checked here.

    References
    ----------
    FHWA GEC-12 Vol I Table 7-26 (after AASHTO 2014)
    """
    # Prefer Vs30, then N-bar, then su-bar
    if vs30 is not None:
        if vs30 > 1524.0:
            return "A"
        elif vs30 > 762.0:
            return "B"
        elif vs30 > 365.76:
            return "C"
        elif vs30 > 182.88:
            return "D"
        else:
            return "E"

    if n_bar is not None:
        if n_bar > 50:
            return "C"
        elif n_bar >= 15:
            return "D"
        else:
            return "E"

    if su_bar is not None:
        if su_bar > 95.76:
            return "C"
        elif su_bar >= 47.88:
            return "D"
        else:
            return "E"

    raise ValueError("At least one of vs30, n_bar, or su_bar must be provided")


def site_coefficients(site_class: str, Ss: float, S1: float,
                      pga: Optional[float] = None) -> SiteClassResult:
    """Compute site coefficients Fpga, Fa, Fv per AASHTO.

    Parameters
    ----------
    site_class : str
        Site class "A" through "E".
    Ss : float
        Spectral acceleration at 0.2s period (g).
    S1 : float
        Spectral acceleration at 1.0s period (g).
    pga : float, optional
        Peak ground acceleration coefficient on rock (g). If given,
        Fpga is interpolated against PGA (GEC-12 Table 7-27, PGA 0.1 to
        0.5) and As = Fpga PGA (GEC-12 Eq. 7-78). If omitted, Fpga and As
        are None: Fpga is read against PGA, and no page prints it from
        Ss. Until GM-293 Fpga was then Fa(Ss), the same table values read
        against Ss.

    Returns
    -------
    SiteClassResult
        Site classification results with Fpga, Fa, Fv (and pga, As).

    References
    ----------
    FHWA GEC-12 Vol I Tables 7-27 (Fpga vs PGA), 7-28 (Fa vs Ss) and 7-29
    (Fv vs S1), p. 380, after AASHTO (2014): checked cell by cell (GM-293).
    AASHTO LRFD Tables 3.10.3.2-1/2/3 (5th Edition numbering; the 9th
    Edition renumbers the identical tables 3.10.3.1-1/2/3).

    Provenance (VERIFIED, 2026-07-18 wiki-verification): all 75 table cells,
    the column breakpoints, the straight-line-interpolation rule, and the
    Class-F site-specific footnote verified cell-by-cell against AASHTO LRFD
    5th Edition (2010) Tables 3.10.3.2-1/2/3 (printed pp. 3-88/3-89, source
    PDF in the owner's library) — 75/75 match, incl. the Fpga=Fa value-set
    aliasing, which is faithful to print. The 9th-Ed copy in the library is
    DRM-locked; the tables are edition-stable (renumbered only). Ledger:
    module_work/wiki_verification/aashto_site_factors.md.
    """
    # Fa table: rows = site class, columns = Ss values
    # Ss:     0.25  0.50  0.75  1.00  1.25
    _Fa_table = {
        "A": [0.8, 0.8, 0.8, 0.8, 0.8],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [1.2, 1.2, 1.1, 1.0, 1.0],
        "D": [1.6, 1.4, 1.2, 1.1, 1.0],
        "E": [2.5, 1.7, 1.2, 0.9, 0.9],
    }
    _Fa_Ss = [0.25, 0.50, 0.75, 1.00, 1.25]

    # Fv table: rows = site class, columns = S1 values
    # S1:     0.10  0.20  0.30  0.40  0.50
    _Fv_table = {
        "A": [0.8, 0.8, 0.8, 0.8, 0.8],
        "B": [1.0, 1.0, 1.0, 1.0, 1.0],
        "C": [1.7, 1.6, 1.5, 1.4, 1.3],
        "D": [2.4, 2.0, 1.8, 1.6, 1.5],
        "E": [3.5, 3.2, 2.8, 2.4, 2.4],
    }
    _Fv_S1 = [0.10, 0.20, 0.30, 0.40, 0.50]

    # Fpga table (AASHTO 3.10.3.2-1): same coefficient values as the Fa
    # table, but indexed against PGA with breakpoints 0.10-0.50 g.
    _Fpga_table = _Fa_table
    _Fpga_PGA = [0.10, 0.20, 0.30, 0.40, 0.50]

    if site_class.upper() == "F":
        raise ValueError("Site Class F requires site-specific analysis")

    sc = site_class.upper()
    if sc not in _Fa_table:
        raise ValueError(f"Invalid site class '{site_class}'. Use A-E.")

    Fa = _interpolate_table(_Fa_Ss, _Fa_table[sc], Ss)
    Fv = _interpolate_table(_Fv_S1, _Fv_table[sc], S1)
    # Fpga is read against PGA only (GEC-12 Table 7-27); without PGA there
    # is no Fpga and no As (GM-293)
    Fpga = (round(_interpolate_table(_Fpga_PGA, _Fpga_table[sc], pga), 3)
            if pga is not None else None)

    return SiteClassResult(
        site_class=sc,
        Fpga=Fpga,
        Fa=round(Fa, 3),
        Fv=round(Fv, 3),
        Ss=Ss,
        S1=S1,
        pga=pga,
    )


def _interpolate_table(x_vals: List[float], y_vals: List[float],
                       x: float) -> float:
    """Linear interpolation within AASHTO coefficient tables (GEC-12
    p. 379: "Straight line interpolation should be used for intermediate
    values in any of the three tables"), the end values held beyond the
    first and last columns ("PGA < 0.1", "PGA > 0.5")."""
    if x <= x_vals[0]:
        return y_vals[0]
    if x >= x_vals[-1]:
        return y_vals[-1]

    for i in range(len(x_vals) - 1):
        if x_vals[i] <= x <= x_vals[i + 1]:
            frac = (x - x_vals[i]) / (x_vals[i + 1] - x_vals[i])
            return y_vals[i] + frac * (y_vals[i + 1] - y_vals[i])

    return y_vals[-1]
