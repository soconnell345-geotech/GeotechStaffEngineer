"""
Side resistance methods for drilled shafts.

Implements:
- Alpha method for cohesive soils (O'Neill & Reese 1999, Brown et al. 2010)
- Beta method for cohesionless soils (Brown et al. 2010)
- Rock socket method (Horvath & Kenney, O'Neill et al. 1996)

All units are SI: kPa, kN, meters.

References:
    FHWA GEC-10, Chapter 13
    Brown, Turner & Castelli (2010), FHWA-NHI-10-016
    O'Neill & Reese (1999), FHWA-RD-99-049
"""

import math


# Atmospheric pressure (kPa)
PA = 101.325


def alpha_cohesive(cu: float, pa: float = PA) -> float:
    """Alpha factor for side resistance in cohesive soil per GEC-10.

    Parameters
    ----------
    cu : float
        Undrained shear strength (kPa).
    pa : float, optional
        Atmospheric pressure (kPa). Default 101.325.

    Returns
    -------
    float
        Alpha factor (dimensionless).

    References
    ----------
    Brown et al. (2010), GEC-10 Figure 13-5 / Section 13.3.3.2
    """
    ratio = cu / pa
    if ratio <= 1.5:
        return 0.55
    else:
        # Linear decrease above cu/pa = 1.5
        alpha = 0.55 - 0.1 * (ratio - 1.5)
        return max(alpha, 0.35)


def side_resistance_cohesive(cu: float, shaft_perimeter: float,
                             segment_thickness: float,
                             alpha: float = None) -> float:
    """Unit side resistance and total for a cohesive segment.

    fs = alpha * cu
    Qs = fs * perimeter * thickness

    Parameters
    ----------
    cu : float
        Undrained shear strength (kPa).
    shaft_perimeter : float
        Shaft circumference (m).
    segment_thickness : float
        Segment height contributing to side resistance (m).
    alpha : float, optional
        Alpha factor. If None, computed from cu.

    Returns
    -------
    float
        Side resistance for this segment (kN).
    """
    if alpha is None:
        alpha = alpha_cohesive(cu)
    fs = alpha * cu
    return fs * shaft_perimeter * segment_thickness


def beta_cohesionless(z: float) -> float:
    """Beta factor for side resistance in cohesionless soil per GEC-10.

    beta = 1.5 - 0.245 * sqrt(z_ft), clamped to [0.25, 1.2]

    The original O'Neill & Reese (1999) formula uses z in feet.
    This implementation converts internally: z_ft = z_m * 3.28084.

    Parameters
    ----------
    z : float
        Depth below ground surface (m).

    Returns
    -------
    float
        Beta factor (dimensionless).

    References
    ----------
    Brown et al. (2010), GEC-10 Section 13.3.3.3
    O'Neill & Reese (1999), FHWA-RD-99-049
    """
    beta = 1.5 - 0.245 * math.sqrt(max(z, 0) * 3.28084)
    return max(0.25, min(beta, 1.2))


def side_resistance_cohesionless(sigma_v: float, beta: float,
                                 shaft_perimeter: float,
                                 segment_thickness: float) -> float:
    """Side resistance for a cohesionless segment.

    fs = beta * sigma_v', capped at 200 kPa
    Qs = fs * perimeter * thickness

    Parameters
    ----------
    sigma_v : float
        Effective vertical stress at segment midpoint (kPa).
    beta : float
        Beta factor.
    shaft_perimeter : float
        Shaft circumference (m).
    segment_thickness : float
        Segment height (m).

    Returns
    -------
    float
        Side resistance for this segment (kN).
    """
    fs = min(beta * sigma_v, 200.0)
    return fs * shaft_perimeter * segment_thickness


def side_resistance_rock(qu: float, shaft_perimeter: float,
                         segment_thickness: float,
                         C: float = 0.65, alpha_E: float = 1.0) -> float:
    """Side resistance for a rock socket segment.

    fs = C * alpha_E * sqrt(qu * pa)

    Equivalent to: qs = C * alpha_E * pa * sqrt(qu / pa)
    where pa = 101.325 kPa (atmospheric pressure).

    Parameters
    ----------
    qu : float
        Unconfined compressive strength of rock (kPa).
    shaft_perimeter : float
        Socket circumference (m).
    segment_thickness : float
        Socket segment height (m).
    C : float, optional
        Socket roughness/fitting coefficient. Default 0.65
        (Horvath & Kenney base coefficient per GEC-10).
    alpha_E : float, optional
        Rock mass reduction factor for jointing. Default 1.0 (intact).

    Returns
    -------
    float
        Side resistance for this segment (kN).

    References
    ----------
    Horvath & Kenney (1979); GEC-10 Section 13.3.3.4 (Eq. 13-15);
    AASHTO LRFD Section 10.8.3.5.4b-1
    """
    fs = C * alpha_E * math.sqrt(qu * PA)
    return fs * shaft_perimeter * segment_thickness
