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
"""


def Nc_drilled_shaft(L_over_D: float) -> float:
    """Bearing capacity factor Nc for drilled shaft in cohesive soil.

    Nc increases from 6.0 at L/D=0 to 9.0 at L/D>=3.

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
    GEC-10 Section 13.3.4.2; O'Neill & Reese (1999)
    """
    return min(6.0 + L_over_D, 9.0)


def end_bearing_cohesive(cu: float, tip_area: float,
                         L_over_D: float = 10.0) -> float:
    """End bearing in cohesive soil.

    qb = Nc * cu
    Qb = qb * A_tip

    Parameters
    ----------
    cu : float
        Undrained shear strength at tip (kPa).
    tip_area : float
        Shaft tip area (m²).
    L_over_D : float, optional
        Length to diameter ratio. Default 10.0 (full Nc=9).

    Returns
    -------
    float
        End bearing capacity (kN).
    """
    Nc = Nc_drilled_shaft(L_over_D)
    qb = Nc * cu
    return qb * tip_area


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
        Shaft tip area (m²).
    diameter : float, optional
        Shaft diameter (m). Default 1.0.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10 Section 13.3.4.3; O'Neill & Reese (1999)
    """
    N60_capped = min(N60, 50.0)
    qb = 57.5 * N60_capped  # kPa

    # Large diameter reduction
    if diameter > 1.27:
        qb *= 1.27 / diameter

    return qb * tip_area


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
    if RQD >= 70:
        qb = 2.5 * qu
    elif RQD >= 50:
        # Linear interpolation: 2.5 at RQD=70, 1.5 at RQD=50
        qb = (1.5 + (RQD - 50) / 20 * 1.0) * qu
    elif RQD >= 25:
        # Further reduction: 1.5 at RQD=50, 0.8 at RQD=25
        qb = (0.8 + (RQD - 25) / 25 * 0.7) * qu
    else:
        # Very fractured: 0.8 at RQD=25, 0.4 at RQD=0
        qb = (0.4 + RQD / 25 * 0.4) * qu

    return qb * tip_area
