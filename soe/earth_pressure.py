"""
Earth pressure calculations for support of excavation design.

Implements classical Rankine/Coulomb coefficients (copied from sheet_pile module
per no-cross-module-import convention) plus Terzaghi-Peck apparent earth
pressure envelopes for braced/anchored excavations.

All units SI: kPa, kN/m³, degrees, meters.

References:
    Rankine (1857), Coulomb (1776)
    Terzaghi, K. & Peck, R.B. (1967) Soil Mechanics in Engineering Practice
    FHWA-IF-99-015, GEC-4: Ground Anchors and Anchored Systems
    Peck, R.B. (1969) Deep Excavation and Tunneling in Soft Ground, SOA Report
"""

import math
import warnings
from typing import List, Optional, Tuple


# ============================================================================
# Classical earth pressure coefficients (from sheet_pile/earth_pressure.py)
# ============================================================================

def rankine_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient Ka = tan²(45° - phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 - phi_rad / 2) ** 2


def rankine_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient Kp = tan²(45° + phi/2)."""
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4 + phi_rad / 2) ** 2


def coulomb_Ka(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb active earth pressure coefficient."""
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    num = math.sin(alpha + phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha - delta)
    term2 = math.sin(phi + delta) * math.sin(phi - beta)
    term3 = math.sin(alpha - delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Ka calculation")

    denom = term1 * (1 + math.sqrt(term2 / term3)) ** 2
    return num / denom


def coulomb_Kp(phi_deg: float, delta_deg: float = 0.0,
               alpha_deg: float = 90.0, beta_deg: float = 0.0) -> float:
    """Coulomb passive earth pressure coefficient."""
    phi = math.radians(phi_deg)
    delta = math.radians(delta_deg)
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    num = math.sin(alpha + phi) ** 2
    term1 = math.sin(alpha) ** 2 * math.sin(alpha + delta)
    term2 = math.sin(phi + delta) * math.sin(phi + beta)
    term3 = math.sin(alpha + delta) * math.sin(alpha + beta)

    if term3 <= 0:
        raise ValueError("Invalid geometry for Coulomb Kp calculation")

    denom = term1 * (1 - math.sqrt(term2 / term3)) ** 2
    if denom <= 0:
        warnings.warn("Coulomb Kp computation failed; using Rankine Kp")
        return rankine_Kp(phi_deg)
    return num / denom


# ----------------------------------------------------------------------------
# Caquot-Kerisel (1948) log-spiral passive coefficient
# (copied from sheet_pile/earth_pressure.py per no-cross-module-import convention)
# ----------------------------------------------------------------------------

# Base Kp at delta = phi (full wall friction), vertical wall, level backfill,
# from the Caquot & Kerisel (1948) log-spiral charts (Caltrans T&S Fig 4-20 /
# NAVFAC DM-7.2). phi = 30 deg anchored to the Caltrans Fig 4-20 read (6.30).
_CK_KP0_PHI = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
_CK_KP0 = [2.20, 3.10, 4.40, 6.30, 9.20, 14.0]

# Wall-friction reduction factor R = Kp(delta)/Kp(delta=phi) (Caltrans Matrix
# 4-1 / NAVFAC DM-7.2); tabulated for phi 30/32/35 and delta/phi 0.40/0.44/0.50.
_CK_R_PHI = [30.0, 32.0, 35.0]
_CK_R_RATIO = [0.40, 0.44, 0.50, 1.00]
_CK_R = {
    30.0: {0.40: 0.686, 0.44: 0.710, 0.50: 0.746, 1.00: 1.000},
    32.0: {0.40: 0.653, 0.44: 0.679, 0.50: 0.717, 1.00: 1.000},
    35.0: {0.40: 0.603, 0.44: 0.631, 0.50: 0.674, 1.00: 1.000},
}


def _lininterp(x, xs, ys):
    """Clamped piecewise-linear interpolation."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def caquot_kerisel_Kp(phi_deg: float, delta_deg: float = None,
                      Kp_initial: float = None) -> float:
    """Caquot-Kerisel (1948) log-spiral passive coefficient Kp' = R * Kp0.

    Kp0 is the passive coefficient at delta = phi; R = Kp(delta)/Kp(delta=phi)
    <= 1 reduces it for the actual wall-friction ratio delta/phi. Unlike Coulomb
    (planar wedge), the log-spiral surface does not over-predict Kp at high
    delta/phi. For phi = 30 deg, delta/phi = 0.5, Kp' = 6.30 * 0.746 = 4.70
    (Caltrans Ex 8-1). Below the lowest tabulated delta/phi column (0.40) R is
    interpolated down to the Rankine anchor R(0) = Kp_rankine(phi)/Kp0, so a
    smooth wall (delta = 0) returns exactly the Rankine Kp (not 0.686*Kp0). See
    sheet_pile.earth_pressure.caquot_kerisel_Kp.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    delta_deg : float, optional
        Wall friction angle (degrees). Default None -> delta = phi (R = 1.0).
    Kp_initial : float, optional
        Chart-read base Kp at delta = phi. If None, from the digitized table.

    References
    ----------
    Caquot & Kerisel (1948); Caltrans T&S Manual Sec 4-6 / Fig 4-20 / Matrix 4-1;
    NAVFAC DM-7.2.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    Kp0 = Kp_initial if Kp_initial is not None else _lininterp(phi_deg, _CK_KP0_PHI, _CK_KP0)
    if delta_deg is None or phi_deg <= 0:
        return Kp0
    ratio = min(max(delta_deg / phi_deg, 0.0), 1.0)
    if ratio >= 1.0:
        return Kp0

    def _R_tab(r):
        # Bilinear over the tabulated (phi, delta/phi) grid, clamped in phi.
        phi_c = min(max(phi_deg, _CK_R_PHI[0]), _CK_R_PHI[-1])
        r_row = [_lininterp(r, _CK_R_RATIO, [_CK_R[p][k] for k in _CK_R_RATIO])
                 for p in _CK_R_PHI]
        return _lininterp(phi_c, _CK_R_PHI, r_row)

    lowest = _CK_R_RATIO[0]  # 0.40 -- lowest tabulated delta/phi column
    if ratio >= lowest:
        R = _R_tab(ratio)
    else:
        # Below the lowest tabulated column the table would clamp to R(0.40),
        # over-predicting passive resistance for near-smooth walls. Interpolate R
        # linearly between the Rankine anchor R(0) = Kp_rankine(phi)/Kp0 -- which
        # makes delta = 0 return exactly the Rankine Kp -- and R(0.40).
        R0 = rankine_Kp(phi_deg) / Kp0
        R = R0 + (ratio / lowest) * (_R_tab(lowest) - R0)
    return Kp0 * R


def K0(phi_deg: float) -> float:
    """At-rest earth pressure coefficient (Jaky). K0 = 1 - sin(phi)."""
    return 1.0 - math.sin(math.radians(phi_deg))


def active_pressure(gamma: float, z: float, Ka: float,
                    c: float = 0.0, q_surcharge: float = 0.0) -> float:
    """Active lateral pressure: sigma_a = Ka*(gamma*z + q) - 2*c*sqrt(Ka)."""
    return Ka * (gamma * z + q_surcharge) - 2.0 * c * math.sqrt(Ka)


def passive_pressure(gamma: float, z: float, Kp: float,
                     c: float = 0.0) -> float:
    """Passive lateral pressure: sigma_p = Kp*gamma*z + 2*c*sqrt(Kp)."""
    return Kp * gamma * z + 2.0 * c * math.sqrt(Kp)


def tension_crack_depth(c: float, gamma: float, Ka: float,
                        q_surcharge: float = 0.0) -> float:
    """Depth of tension crack in cohesive soil behind active wall."""
    if c <= 0 or Ka <= 0 or gamma <= 0:
        return 0.0
    z_crack = (2.0 * c / math.sqrt(Ka) - q_surcharge) / gamma
    return max(z_crack, 0.0)


# ============================================================================
# Apparent earth pressure envelopes (Terzaghi & Peck 1967, FHWA GEC-4)
# ============================================================================

def apparent_pressure_sand(gamma: float, H: float, Ka: float) -> float:
    """Apparent earth pressure for braced excavations in sand.

    Returns the uniform (rectangular) envelope ordinate:
        p = 0.65 * Ka * gamma * H

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    Ka : float
        Active earth pressure coefficient.

    Returns
    -------
    float
        Uniform apparent pressure (kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 3.
    FHWA-IF-99-015, Section 5.2.1.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    return 0.65 * Ka * gamma * H


def apparent_pressure_soft_clay(gamma: float, H: float, cu: float,
                                m: float = 1.0) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in soft to medium clay.

    Applies when stability number N = gamma*H/cu > 4.

    Returns a uniform (rectangular) envelope:
        Ka_apparent = 1 - m * (4*cu / (gamma*H))
        p = Ka_apparent * gamma * H

    where m = 1.0 for most cases (Peck 1969), m = 0.4 if movements are
    critical and excavation has no significant soft layer below the base.

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).
    m : float
        Empirical coefficient. Default 1.0.

    Returns
    -------
    tuple of (str, float)
        ("uniform", pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 4.
    FHWA-IF-99-015, Section 5.2.2.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")

    N = gamma * H / cu  # stability number
    Ka_apparent = 1.0 - m * (4.0 * cu / (gamma * H))
    # Ka_apparent should not be less than 0.25 per FHWA guidance
    Ka_apparent = max(Ka_apparent, 0.25)
    p = Ka_apparent * gamma * H
    return ("uniform", p)


def apparent_pressure_stiff_clay(gamma: float, H: float,
                                 cu: float) -> Tuple[str, float]:
    """Apparent earth pressure for braced excavations in stiff clay.

    Applies when stability number N = gamma*H/cu <= 4.

    Returns a trapezoidal envelope with maximum ordinate between
    0.2*gamma*H and 0.4*gamma*H. Uses 0.2 + 0.4*(N/4) interpolation
    based on stability number.

    The trapezoidal shape is:
    - Zero at ground surface
    - Increases linearly to max over top 0.25*H
    - Constant max from 0.25*H to 0.75*H
    - Decreases linearly to zero at H (or to a reduced value)

    Parameters
    ----------
    gamma : float
        Average unit weight (kN/m³).
    H : float
        Total excavation depth (m).
    cu : float
        Average undrained shear strength (kPa).

    Returns
    -------
    tuple of (str, float)
        ("trapezoidal", max_pressure in kPa).

    References
    ----------
    Terzaghi & Peck (1967); Peck (1969), Fig. 5.
    FHWA-IF-99-015, Section 5.2.3.
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if cu <= 0:
        raise ValueError("Undrained shear strength cu must be positive")

    N = gamma * H / cu
    # Interpolate coefficient between 0.2 and 0.4 based on stability number
    coeff = min(0.2 + 0.2 * (N / 4.0), 0.4)
    p_max = coeff * gamma * H
    return ("trapezoidal", p_max)


def select_apparent_pressure(soil_layers, H: float,
                             surcharge: float = 0.0) -> dict:
    """Auto-select apparent pressure diagram from soil profile.

    Determines the controlling soil type over the excavation depth and
    returns the appropriate apparent pressure envelope.

    Parameters
    ----------
    soil_layers : list of SOEWallLayer
        Soil layers from ground surface downward.
    H : float
        Excavation depth (m).
    surcharge : float
        Surface surcharge (kPa). Default 0.

    Returns
    -------
    dict
        Keys: "type" ("sand", "soft_clay", "stiff_clay"),
              "shape" ("uniform" or "trapezoidal"),
              "max_pressure_kPa" (peak ordinate),
              "stability_number" (gamma*H/cu for clay, None for sand).
    """
    if H <= 0:
        raise ValueError("Excavation depth H must be positive")
    if not soil_layers:
        raise ValueError("At least one soil layer is required")

    # Compute weighted averages over excavation depth
    total_gamma_h = 0.0
    total_cu_h = 0.0
    total_phi_h = 0.0
    sand_thickness = 0.0
    clay_thickness = 0.0
    remaining = H

    for layer in soil_layers:
        h = min(layer.thickness, remaining)
        total_gamma_h += layer.unit_weight * h
        total_cu_h += layer.cohesion * h
        total_phi_h += layer.friction_angle * h
        if layer.soil_type == "sand":
            sand_thickness += h
        else:
            clay_thickness += h
        remaining -= h
        if remaining <= 0:
            break

    gamma_avg = total_gamma_h / H
    cu_avg = total_cu_h / H
    phi_avg = total_phi_h / H

    # Determine controlling soil type
    if sand_thickness >= clay_thickness:
        # Predominantly sand
        Ka = rankine_Ka(phi_avg)
        p_max = apparent_pressure_sand(gamma_avg, H, Ka)
        return {
            "type": "sand",
            "shape": "uniform",
            "max_pressure_kPa": round(p_max, 2),
            "stability_number": None,
            "Ka": round(Ka, 4),
            "gamma_avg": round(gamma_avg, 2),
        }
    else:
        # Predominantly clay
        N = gamma_avg * H / cu_avg if cu_avg > 0 else float("inf")

        if N > 4:
            # Soft to medium clay
            shape, p_max = apparent_pressure_soft_clay(gamma_avg, H, cu_avg)
            return {
                "type": "soft_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_max, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
            }
        else:
            # Stiff clay
            shape, p_max = apparent_pressure_stiff_clay(
                gamma_avg, H, cu_avg
            )
            return {
                "type": "stiff_clay",
                "shape": shape,
                "max_pressure_kPa": round(p_max, 2),
                "stability_number": round(N, 2),
                "cu_avg": round(cu_avg, 2),
                "gamma_avg": round(gamma_avg, 2),
            }


def fhwa_apparent_pressure_anchored_wall(
    H: float,
    anchor_depths: List[float],
    gamma: float,
    phi: float,
    surcharge: float = 0.0,
    spacing: float = 1.0,
    inclination_deg: float = 0.0,
    Ka: float = None,
) -> dict:
    """FHWA/GEC-4 apparent-pressure anchored-wall design (tributary method).

    Builds the FHWA apparent earth-pressure envelope for a multi-level anchored
    wall in sand and distributes it to the anchors by the tributary-area (hinge)
    method — the GEC-4 (FHWA-IF-99-015) Appendix A procedure.

    The maximum apparent-pressure ordinate is

        pe = 0.65 * Ka * gamma * H^2 / (H - H1/3 - Hn+1/3)

    where H1 = depth to the topmost anchor and Hn+1 = height from the lowest
    anchor to the base. (The total apparent earth load is
    PT = 0.65*Ka*gamma*H^2 = 1.3 * (1/2 Ka gamma H^2), i.e. 1.3x the triangular
    Rankine total.) A uniform surcharge adds ps = Ka*q over the full height.

    Tributary horizontal anchor loads (per unit width of wall):
      - top anchor:     (2/3 H1 + H2/2) pe + (H1 + H2/2) ps
      - interior anchor: (Hi/2 + Hi+1/2) (pe + ps)
      - bottom anchor:  (Hn/2 + 23/48 Hn+1) pe + (Hn/2 + Hn+1/2) ps
      - subgrade reaction R = 3/16 Hn+1 pe + Hn+1/2 ps
    Hinge (simple-span) moments: top span (13/54) H1^2 (pe+ps); interior/lowest
    spans (1/10) Hi^2 (pe+ps). Anchor design loads DL = TH * spacing / cos(incl).

    Validated vs GEC-4 Design Example 1 (2-anchor, SI): pe=43.6, ps=3.2,
    TH1=168, TH2=172, R=37, M=66 kN-m/m, DL1=435, DL2=445 kN. For a SINGLE
    anchor (n=1) the envelope pe (= the Caltrans Ex 8-1 max ordinate sigma_a),
    the total load PT, and the upper-tributary load are returned; the single-
    anchor TOTAL anchor force and embedment come from the free-earth-support
    solve (``sheet_pile.analyze_anchored(pressure_method="log_spiral")``).

    Parameters
    ----------
    H : float
        Total wall height (m).
    anchor_depths : list of float
        Depths of the anchor levels below the top of wall (m).
    gamma : float
        Retained-soil unit weight (kN/m3).
    phi : float
        Retained-soil friction angle (deg).
    surcharge : float, optional
        Uniform surface surcharge q (kPa). Default 0.
    spacing : float, optional
        Horizontal anchor spacing (m) for the design loads. Default 1.0.
    inclination_deg : float, optional
        Anchor inclination from horizontal (deg). Default 0.
    Ka : float, optional
        Active coefficient. If None, Rankine Ka(phi).

    Returns
    -------
    dict
        {Ka, pe_kPa, ps_kPa, PT_total_kN_per_m, anchors[], subgrade_reaction_kN_per_m,
         max_moment_kN_m_per_m, ...}. Each anchors[] entry: {depth_m,
         TH_kN_per_m (horizontal tributary load), design_load_kN}.

    References
    ----------
    FHWA-IF-99-015 (GEC-4), Ground Anchors and Anchored Systems, App. A;
    Caltrans T&S Manual Ch. 8.
    """
    if H <= 0:
        raise ValueError("Wall height H must be positive")
    if not anchor_depths:
        raise ValueError("At least one anchor depth is required")
    d = sorted(float(x) for x in anchor_depths)
    if d[0] <= 0 or d[-1] >= H:
        raise ValueError("Anchor depths must be between 0 and H")
    n = len(d)
    if Ka is None:
        Ka = rankine_Ka(phi)

    H1 = d[0]
    Hn1 = H - d[-1]
    denom = H - H1 / 3.0 - Hn1 / 3.0
    pe = 0.65 * Ka * gamma * H ** 2 / denom
    ps = Ka * surcharge
    PT_total = 0.65 * Ka * gamma * H ** 2

    # Spans between successive supports: span[i] = d[i] - d[i-1] (i>=1).
    cos_incl = math.cos(math.radians(inclination_deg))
    anchors = []
    moments = []

    if n == 1:
        # Single anchor: report the upper tributary (well-defined); the total
        # anchor force needs the FES embedment solve.
        TH_upper = (2.0 / 3.0) * H1 * pe + H1 * ps
        anchors.append({
            "depth_m": round(d[0], 3),
            "TH_upper_kN_per_m": round(TH_upper, 3),
            "TH_kN_per_m": None,   # total needs FES solve (see docstring)
            "design_load_kN": None,
        })
        moments.append((13.0 / 54.0) * H1 ** 2 * (pe + ps))
    else:
        for i in range(n):
            if i == 0:
                H2 = d[1] - d[0]
                TH = (2.0 / 3.0 * H1 + H2 / 2.0) * pe + (H1 + H2 / 2.0) * ps
            elif i == n - 1:
                Hn = d[i] - d[i - 1]
                TH = (Hn / 2.0 + 23.0 / 48.0 * Hn1) * pe + (Hn / 2.0 + Hn1 / 2.0) * ps
            else:
                Hi = d[i] - d[i - 1]
                Hi1 = d[i + 1] - d[i]
                TH = (Hi / 2.0 + Hi1 / 2.0) * (pe + ps)
            anchors.append({
                "depth_m": round(d[i], 3),
                "TH_kN_per_m": round(TH, 3),
                "design_load_kN": round(TH * spacing / cos_incl, 2),
            })
        # Hinge moments: top span + each interior/lowest span.
        moments.append((13.0 / 54.0) * H1 ** 2 * (pe + ps))
        for i in range(1, n):
            span = d[i] - d[i - 1]
            moments.append((1.0 / 10.0) * span ** 2 * (pe + ps))
        moments.append((1.0 / 10.0) * Hn1 ** 2 * (pe + ps))

    R = (3.0 / 16.0) * Hn1 * pe + (Hn1 / 2.0) * ps

    return {
        "Ka": round(Ka, 4),
        "pe_kPa": round(pe, 3),
        "ps_kPa": round(ps, 3),
        "PT_total_kN_per_m": round(PT_total, 2),
        "H1_m": round(H1, 3),
        "Hn1_m": round(Hn1, 3),
        "n_anchors": n,
        "anchors": anchors,
        "subgrade_reaction_kN_per_m": round(R, 3),
        "max_moment_kN_m_per_m": round(max(moments), 3),
        "spacing_m": spacing,
        "inclination_deg": inclination_deg,
    }


def get_pressure_at_depth(z: float, H: float, shape: str,
                          p_max: float) -> float:
    """Return apparent pressure ordinate at depth z for given envelope shape.

    Parameters
    ----------
    z : float
        Depth from top of wall (m).
    H : float
        Total excavation depth (m).
    shape : str
        "uniform" or "trapezoidal".
    p_max : float
        Maximum pressure ordinate (kPa).

    Returns
    -------
    float
        Apparent pressure at depth z (kPa).
    """
    if z < 0 or z > H:
        return 0.0

    if shape == "uniform":
        return p_max

    elif shape == "trapezoidal":
        # Trapezoidal: ramps from 0 to p_max over top 0.25H,
        # constant from 0.25H to 0.75H, ramps to 0 at H.
        if z <= 0.25 * H:
            return p_max * (z / (0.25 * H)) if H > 0 else 0.0
        elif z <= 0.75 * H:
            return p_max
        else:
            return p_max * ((H - z) / (0.25 * H)) if H > 0 else 0.0

    return p_max  # fallback
