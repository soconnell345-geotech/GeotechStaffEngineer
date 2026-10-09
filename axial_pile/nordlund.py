"""
Nordlund Method for skin friction in cohesionless soils.

Computes skin friction using Nordlund's (1963, 1979) method and
end bearing using Meyerhof's method for driven piles in sand/gravel.

All units are SI: kPa, meters, kN, degrees.

.. warning:: **Chart-fit simplifications vs rigorous Nordlund.**
    The rigorous Nordlund method reads several design charts
    (GEC-12 Figures 7-4 through 7-8 and 7-13). This module uses
    simplified curve fits that drop some of the chart dependencies:

    * ``nordlund_Kd`` ignores the displaced-volume (V/V0) family of
      curves — it fixes the typical displacement-pile curve;
    * ``alpha_t_factor`` ignores phi and returns the maximum (1.0) for
      D/b > 5, which can over-predict tip resistance (mitigated by the
      Meyerhof limiting q_L cap);
    * ``nordlund_CF`` keys only on delta/phi (a linear fit), not on phi.

    Results can therefore deviate from a rigorous chart-based Nordlund
    calculation — typically within the chart-reading scatter for common
    displacement piles at phi = 28-38 deg, but verify against the full
    charts (or a load test / driving criteria) where tip resistance or
    low-displacement piles (H-piles, open pipes) govern the design.

References:
    Nordlund, R.L. (1963, 1979)
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7, Sections 7.2.1
    FHWA Soils & Foundations Reference Manual, Vol II
    Meyerhof, G.G. (1976) — Bearing capacity and settlement of pile foundations
"""

import math
import warnings
from typing import Optional

import numpy as np


def delta_from_phi(phi_deg: float, pile_material: str = "steel") -> float:
    """Estimate pile-soil friction angle delta.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    pile_material : str, optional
        "steel" (delta/phi ≈ 0.67-0.83), "concrete" (≈ 0.80-1.0),
        "timber" (≈ 0.80-1.0). Default "steel".

    Returns
    -------
    float
        Pile-soil friction angle delta (degrees).

    References
    ----------
    FHWA GEC-12, Table 7-1.
    """
    ratios = {
        "steel": 0.75,
        "concrete": 0.90,
        "timber": 0.90,
    }
    pile_material = pile_material.lower()
    ratio = ratios.get(pile_material, 0.75)
    return phi_deg * ratio


def nordlund_Kd(phi_deg: float, omega_deg: float = 0.0) -> float:
    """Coefficient of lateral earth pressure Kd for Nordlund method.

    Simplified from Nordlund's charts. For uniform (non-tapered) piles
    (omega=0), Kd depends primarily on phi.

    .. warning:: This fit IGNORES the displaced-volume (V/V0) curve
        family of the GEC-12 Kd charts — it represents a typical
        displacement pile (V/V0 ~ 0.5-1.0). Low-displacement piles
        (H-piles, open-ended pipes) have lower Kd than returned here;
        see the module-level warning.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    omega_deg : float, optional
        Pile taper angle (degrees). Default 0 (uniform pile).

    Returns
    -------
    float
        Coefficient of lateral earth pressure Kd.

    References
    ----------
    FHWA GEC-12, Figures 7-3 through 7-7 (Nordlund Kd charts).
    Simplified curve fit for omega=0.
    """
    # Simplified Kd for omega=0 (uniform piles)
    # From FHWA GEC-12 Figure 7-5 (V/V0 = 0.1 to 1.0 curves)
    # Using V/V0 ≈ 0.5 to 1.0 for typical displacement piles
    if omega_deg > 0:
        # Tapered piles have higher Kd; simplified increase
        kd_factor = 1.0 + 0.5 * omega_deg / 10.0  # rough approximation
    else:
        kd_factor = 1.0

    # Kd vs phi (omega=0, displacement pile, from FHWA charts)
    if phi_deg <= 25:
        Kd = 0.7
    elif phi_deg <= 30:
        Kd = 0.7 + (phi_deg - 25) * (1.0 - 0.7) / 5
    elif phi_deg <= 35:
        Kd = 1.0 + (phi_deg - 30) * (1.5 - 1.0) / 5
    elif phi_deg <= 40:
        Kd = 1.5 + (phi_deg - 35) * (2.5 - 1.5) / 5
    else:
        Kd = 2.5 + (phi_deg - 40) * 0.2

    return Kd * kd_factor


def nordlund_CF(delta_phi_ratio: float) -> float:
    """Correction factor CF for Kd when delta/phi differs from chart value.

    Parameters
    ----------
    delta_phi_ratio : float
        Actual delta/phi ratio.

    Returns
    -------
    float
        Correction factor CF. Approximately 1.0 when delta/phi matches
        the value used to develop the Kd charts.

    .. warning:: Simplified linear fit keyed ONLY on delta/phi; the
        GEC-12 Figure 7-8 chart also varies with phi. See the
        module-level warning.

    References
    ----------
    FHWA GEC-12, Figure 7-8.
    """
    # CF ≈ 1.0 for typical values; simplified linear adjustment
    # CF chart shows CF varies from ~0.4 to 1.5 based on delta/phi
    # For delta/phi = 0.75 (typical steel), CF ≈ 1.0
    return max(0.5, min(delta_phi_ratio / 0.75, 1.5))


def skin_friction_cohesionless(phi_deg: float, sigma_v: float,
                               pile_perimeter: float,
                               layer_thickness: float,
                               pile_material: str = "steel",
                               delta_phi_ratio: Optional[float] = None,
                               omega_deg: float = 0.0) -> float:
    """Compute skin friction in a cohesionless soil layer (Nordlund).

    Qs = Kd * CF * sigma_v' * sin(delta) * perimeter * dz

    Parameters
    ----------
    phi_deg : float
        Soil friction angle (degrees).
    sigma_v : float
        Effective overburden pressure at the center of the layer (kPa).
    pile_perimeter : float
        Pile perimeter (m).
    layer_thickness : float
        Layer thickness (m).
    pile_material : str, optional
        Pile material. Default "steel".
    delta_phi_ratio : float, optional
        delta/phi ratio. If None, determined from pile_material.
    omega_deg : float, optional
        Pile taper angle (degrees). Default 0.

    Returns
    -------
    float
        Skin friction from this layer (kN).

    References
    ----------
    FHWA GEC-12, Eq 7-1.
    """
    if delta_phi_ratio is None:
        delta_deg = delta_from_phi(phi_deg, pile_material)
    else:
        delta_deg = phi_deg * delta_phi_ratio

    delta_rad = math.radians(delta_deg)
    omega_rad = math.radians(omega_deg)

    Kd = nordlund_Kd(phi_deg, omega_deg)
    CF = nordlund_CF(delta_deg / phi_deg if phi_deg > 0 else 0.75)

    fs = Kd * CF * sigma_v * math.sin(delta_rad + omega_rad) / math.cos(omega_rad)
    return fs * pile_perimeter * layer_thickness


def nordlund_Nq_prime(phi_deg: float) -> float:
    """Bearing capacity factor Nq' for pile tip (Meyerhof, 1976).

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).

    Returns
    -------
    float
        Nq' factor.

    References
    ----------
    FHWA GEC-12, Figure 7-14 (after Meyerhof, 1976).
    """
    # Meyerhof Nq' for driven piles (from FHWA charts)
    # Interpolated from Figure 7-14
    phi = phi_deg
    if phi <= 20:
        return 8.0
    elif phi <= 25:
        return 8.0 + (phi - 20) * (12 - 8) / 5
    elif phi <= 28:
        return 12 + (phi - 25) * (20 - 12) / 3
    elif phi <= 30:
        return 20 + (phi - 28) * (35 - 20) / 2
    elif phi <= 32:
        return 35 + (phi - 30) * (55 - 35) / 2
    elif phi <= 34:
        return 55 + (phi - 32) * (90 - 55) / 2
    elif phi <= 36:
        return 90 + (phi - 34) * (130 - 90) / 2
    elif phi <= 38:
        return 130 + (phi - 36) * (200 - 130) / 2
    elif phi <= 40:
        return 200 + (phi - 38) * (300 - 200) / 2
    elif phi <= 42:
        return 300 + (phi - 40) * (400 - 300) / 2
    else:
        return 400 + (phi - 42) * 50


def alpha_t_factor(Db_ratio: float) -> float:
    """Dimensionless factor alpha_t for Nordlund end bearing.

    Parameters
    ----------
    Db_ratio : float
        Pile depth / pile diameter ratio (D/b).

    Returns
    -------
    float
        alpha_t factor (0 to 1). Approaches 1.0 for deep piles.

    .. warning:: Simplified fit that IGNORES phi (the GEC-12 Figure 7-13
        chart varies with phi) and returns the maximum (1.0) for
        D/b > 5 — this can over-predict tip resistance for high-phi
        soils, though the Meyerhof limiting q_L cap bounds the error.
        See the module-level warning.

    References
    ----------
    FHWA GEC-12, Figure 7-13.
    """
    # alpha_t approaches 1.0 quickly; simplified curve
    if Db_ratio <= 0:
        return 0.0
    elif Db_ratio <= 5:
        return 0.5 + 0.5 * Db_ratio / 5
    else:
        return 1.0


# GEC-12 Figure 7-15 (after Meyerhof 1976), measured off the printed figure
# (GEC 12 Vol 1, pdf page index 283) on 2026-10-08. The same nodes are
# geotech_references.gec_12.figure_7_15_limiting_toe_resistance, in tsf.
# The axis starts at 30 deg and the curve ends at 43.75 deg.
TSF_TO_KPA = 95.76
_FIG_7_15_PHI = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 43.75]
_FIG_7_15_QL_TSF = [7.1, 10.1, 16.0, 24.5, 35.9, 53.7, 75.2, 102.3, 133.6,
                    168.3, 208.5, 251.6, 296.0, 339.2, 368.0]
_FIG_7_15_QL_KPA = [q * TSF_TO_KPA for q in _FIG_7_15_QL_TSF]
TOE_LIMIT_PHI_MIN = _FIG_7_15_PHI[0]
TOE_LIMIT_PHI_MAX = _FIG_7_15_PHI[-1]


def _limiting_tip_resistance(phi_deg: float) -> float:
    """Limiting unit tip resistance q_L from GEC-12 Figure 7-15 (Meyerhof, 1976).

    Piecewise linear interpolation of the chart's 1-degree nodes; outside
    the chart (below 30 deg, above 43.75 deg) the curve's end value is used
    and :func:`toe_limit_chart_note` says so.

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).

    Returns
    -------
    float
        Limiting unit tip resistance q_L (kPa).

    References
    ----------
    FHWA GEC-12, Figure 7-15 (after Meyerhof, 1976).

    Provenance.
    - 2026-07-19: replaced a table about 10x unconservative at phi = 30.
    - 2026-10-08: re-measured off the figure (gridline-fitted; a second
      pixel read and two vision models agree within about 1 tsf).
      - The 2026-07-19 table was still +41 % at 30 deg, +25 % at 32 deg and
        +11 % at 34 deg, and 3-6 % low from 38 to 43 deg.
      - It also carried nodes at 26, 28, 44 and 45 deg that are not on the
        chart.
      - GEC-12's worked example (NHI-06-089, the 428.1 kip toe plateau at a
        toe phi of 40) now agrees to +0.3 % (429.3 kips) instead of -4.6 %.
    """
    phi = min(max(phi_deg, TOE_LIMIT_PHI_MIN), TOE_LIMIT_PHI_MAX)
    return float(np.interp(phi, _FIG_7_15_PHI, _FIG_7_15_QL_KPA))


def toe_limit_chart_note(phi_deg: float, governed: Optional[bool] = None
                         ) -> Optional[str]:
    """Say when the toe friction angle is outside GEC-12 Figure 7-15.

    Returns None inside the chart (30-43.75 deg). Outside it, returns the
    warning to show the user: which end value was used, and, below 30 deg,
    that the true limit is lower. ``governed`` (whether the limit capped the
    toe resistance) adds one clause when known.
    """
    if TOE_LIMIT_PHI_MIN <= phi_deg <= TOE_LIMIT_PHI_MAX:
        return None
    if phi_deg < TOE_LIMIT_PHI_MIN:
        end, where = _FIG_7_15_QL_TSF[0], "below"
        tail = (" The chart's curve falls toward zero below 30 deg, so the "
                "true limit is LOWER than this and the toe resistance may be "
                "overstated; check it by another method.")
    else:
        end, where = _FIG_7_15_QL_TSF[-1], "above"
        tail = (" The curve stops at 43.75 deg, so holding its end value is "
                "on the low side of any extrapolation.")
    note = (f"Toe friction angle {phi_deg:g} deg is {where} GEC-12 Figure "
            f"7-15 (it spans 30-43.75 deg). The limiting toe resistance was "
            f"held at the chart's end value, {end:g} tsf "
            f"({end * TSF_TO_KPA:,.0f} kPa).{tail}")
    if governed is True:
        note += " The limit governed the toe resistance here."
    elif governed is False and phi_deg < TOE_LIMIT_PHI_MIN:
        note += (" It did not govern here, but a lower true limit might "
                 "have.")
    return note


def end_bearing_cohesionless(phi_deg: float, sigma_v_tip: float,
                              tip_area: float,
                              pile_depth: float,
                              pile_width: float,
                              notes: Optional[list] = None) -> float:
    """Compute end bearing in cohesionless soil (Nordlund/Meyerhof).

    Qt = alpha_t * Nq' * sigma_v' * At

    With limiting value: qt_limit from Meyerhof (1976).

    Parameters
    ----------
    phi_deg : float
        Soil friction angle at pile tip (degrees).
    sigma_v_tip : float
        Effective overburden at pile tip (kPa).
    tip_area : float
        Pile tip area (m²).
    pile_depth : float
        Pile embedment depth (m).
    pile_width : float
        Pile width or diameter (m).
    notes : list, optional
        When given, a warning about a toe friction angle outside Figure
        7-15 is appended here (for the result to carry to the user). When
        omitted, the same warning is raised with :func:`warnings.warn`.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    FHWA GEC-12, Eq 7-3 and limiting qt from Figure 7-15.
    """
    Db = pile_depth / pile_width if pile_width > 0 else 0
    at = alpha_t_factor(Db)
    Nq = nordlund_Nq_prime(phi_deg)

    qt = at * Nq * sigma_v_tip

    # Limiting tip resistance from GEC-12 Figure 7-15 (Meyerhof, 1976)
    qt_limit = _limiting_tip_resistance(phi_deg)
    note = toe_limit_chart_note(phi_deg, governed=qt >= qt_limit)
    if note:
        if notes is not None:
            if note not in notes:
                notes.append(note)
        else:
            warnings.warn(note, stacklevel=2)

    qt = min(qt, qt_limit)
    return qt * tip_area
