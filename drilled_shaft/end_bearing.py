"""
End bearing methods for drilled shafts.

Implements tip resistance for:
- Cohesive soils (Nc method)
- Cohesionless soils (N60 correlation)
- Rock (UCS-based)

All units are SI: kPa, kN, meters.

References:
    FHWA GEC-10, Chapter 13
    Brown, Turner & Castelli (2010)
    O'Neill & Reese (1999)
    AASHTO LRFD Bridge Design Specifications, Section 10.8.3.5.1c
"""

import math


# GEC-10 / O'Neill-Reese limiting net unit end bearing in clay:
# 80 ksf = 3830 kPa (~3.8 MPa). Higher values require load-test
# verification.
QB_MAX_COHESIVE = 3830.0  # kPa

# Base diameter above which the O'Neill-Reese large-base reduction
# applies in cohesive soil: 75 in = 1.90 m.
LARGE_BASE_DIAMETER = 1.90  # m

_M_TO_IN = 39.3701
_KPA_TO_KSF = 1.0 / 47.8803


def Nc_drilled_shaft(L_over_D: float) -> float:
    """Bearing capacity factor Nc for drilled shaft in cohesive soil.

    O'Neill & Reese (1999) / AASHTO form:
        Nc = 6 * (1 + 0.2 * L/D) <= 9
    which increases from 6.0 at L/D=0 to the maximum 9.0 at L/D = 2.5.
    (The previous linearization min(6 + L/D, 9) reached 9 at L/D = 3,
    slightly under-predicting Nc for 2.5 < L/D < 3.)

    Parameters
    ----------
    L_over_D : float
        Ratio of shaft length to diameter.

    Returns
    -------
    float
        Bearing capacity factor Nc.

    References
    ----------
    GEC-10; O'Neill & Reese (1999); AASHTO LRFD 10.8.3.5.1c
    """
    return min(6.0 * (1.0 + 0.2 * L_over_D), 9.0)


def end_bearing_cohesive(cu: float, tip_area: float,
                         L_over_D: float = 10.0,
                         base_diameter: float = None,
                         shaft_length: float = None) -> float:
    """End bearing in cohesive soil.

    qb = Nc * cu, capped at the GEC-10/O'Neill-Reese limiting net unit
    end bearing of 80 ksf (3830 kPa). For base (bell) diameters larger
    than 1.90 m (75 in), the O'Neill & Reese (1999) / AASHTO large-base
    reduction is applied:

        qb_red = Fr * qb
        Fr = 2.5 / (a * Bb_in + 2.5 * b) <= 1.0
        a  = 0.0071 + 0.0021 * (L / Bb) <= 0.015
        b  = 0.45 * sqrt(cu_ksf), clamped to [0.5, 1.5]

    (Bb_in = base diameter in inches; cu_ksf = cu in ksf.)

    Qb = qb * A_tip

    Parameters
    ----------
    cu : float
        Undrained shear strength at tip (kPa).
    tip_area : float
        Shaft tip (base/bell) area (m²).
    L_over_D : float, optional
        Length to SHAFT diameter ratio (for Nc). Default 10.0 (full Nc=9).
    base_diameter : float, optional
        Base (bell) diameter (m). If provided and > 1.90 m, the
        large-base reduction factor Fr is applied. Default None (no
        reduction — preserves behavior for normal-size bases).
    shaft_length : float, optional
        Shaft length L (m), used in the Fr coefficient ``a``. If None
        when the reduction applies, L is estimated as
        ``L_over_D * base_diameter`` (conservative for belled shafts).

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10; O'Neill & Reese (1999); AASHTO LRFD 10.8.3.5.1c
    """
    return unit_end_bearing_cohesive(
        cu, L_over_D, base_diameter, shaft_length)["qb"] * tip_area


def unit_end_bearing_cohesive(cu: float, L_over_D: float = 10.0,
                              base_diameter: float = None,
                              shaft_length: float = None) -> dict:
    """The unit end bearing in cohesive soil and each step to it (GM-148).

    One place for the arithmetic of :func:`end_bearing_cohesive`, so the
    calc package prints the values the analysis used: ``Nc``,
    ``qb_uncapped`` (Nc cu), ``capped`` (the 3830 kPa limit applied),
    ``Fr`` (None where the base is 1.90 m or smaller), ``a``, ``b``,
    ``L`` (the length in ``a``) and ``qb`` (kPa).
    """
    Nc = Nc_drilled_shaft(L_over_D)
    qb0 = Nc * cu
    qb = min(qb0, QB_MAX_COHESIVE)
    out = {"Nc": Nc, "qb_uncapped": qb0, "capped": qb0 > QB_MAX_COHESIVE,
           "Fr": None, "a": None, "b": None, "L": None}
    if base_diameter is not None and base_diameter > LARGE_BASE_DIAMETER:
        L = shaft_length if shaft_length is not None \
            else L_over_D * base_diameter
        a = min(0.0071 + 0.0021 * (L / base_diameter), 0.015)
        b = 0.45 * math.sqrt(cu * _KPA_TO_KSF)
        b = min(max(b, 0.5), 1.5)
        Fr = min(2.5 / (a * base_diameter * _M_TO_IN + 2.5 * b), 1.0)
        qb *= Fr
        out.update(Fr=Fr, a=a, b=b, L=L)
    out["qb"] = qb
    return out


def end_bearing_cohesionless(N60: float, tip_area: float,
                             diameter: float = 1.0) -> float:
    """End bearing in cohesionless soil.

    qb = 57.5 * N60 (kPa), with N60 capped at 50.
    For D > 1.27m, apply reduction factor 1.27/D.

    Parameters
    ----------
    N60 : float
        Energy-corrected SPT blow count at tip.
    tip_area : float
        Shaft tip (base/bell) area (m²).
    diameter : float, optional
        BASE diameter (m) — the bell diameter for belled shafts, since
        the 1.27/D large-diameter reduction is governed by the bearing
        surface size. Default 1.0.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10 Section 13.3.4.3; O'Neill & Reese (1999)
    """
    return unit_end_bearing_cohesionless(N60, diameter)["qb"] * tip_area


def unit_end_bearing_cohesionless(N60: float, diameter: float = 1.0) -> dict:
    """The unit end bearing in cohesionless soil and each step (GM-148):
    ``N60_capped``, ``qb_unreduced`` (57.5 N60), ``reduction`` (1.27/D
    with D the base diameter, None at 1.27 m or smaller) and ``qb``
    (kPa); see :func:`end_bearing_cohesionless`."""
    N60_capped = min(N60, 50.0)
    qb0 = 57.5 * N60_capped  # kPa
    reduction = 1.27 / diameter if diameter > 1.27 else None
    return {"N60_capped": N60_capped, "qb_unreduced": qb0,
            "reduction": reduction,
            "qb": qb0 * reduction if reduction else qb0}


def end_bearing_rock(qu: float, tip_area: float,
                     RQD: float = 100.0) -> float:
    """End bearing in rock.

    qb = 2.5 * qu for intact rock (RQD > 70%).
    Reduced for fractured rock based on RQD.

    Parameters
    ----------
    qu : float
        Unconfined compressive strength (kPa).
    tip_area : float
        Shaft tip area (m²).
    RQD : float, optional
        Rock Quality Designation (%). Default 100.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10 Section 13.3.4.4; Canadian Foundation Engineering Manual
    """
    return rock_end_bearing_multiplier(RQD) * qu * tip_area


def rock_end_bearing_multiplier(RQD: float) -> float:
    """q_b / q_u for :func:`end_bearing_rock`: 2.5 at RQD >= 70 %, then
    linear to 1.5 at 50 %, 0.8 at 25 % and 0.4 at 0 %."""
    if RQD >= 70:
        return 2.5
    if RQD >= 50:
        return 1.5 + (RQD - 50) / 20 * 1.0
    if RQD >= 25:
        return 0.8 + (RQD - 25) / 25 * 0.7
    return 0.4 + RQD / 25 * 0.4
