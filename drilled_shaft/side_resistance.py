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


def su_to_ciuc(su: float, sigma_v0: float, test_type: str = "uc") -> float:
    """Convert a lab undrained shear strength to its CIUC-equivalent value.

    GEC-10 (2018) calibrates the rational alpha method (``alpha_cohesive_rational``)
    to CIUC (isotropically-consolidated undrained compression) strength, so a
    routine UC or UU laboratory strength is first normalized by the Chen &
    Kulhawy (1993) regressions (GEC-10 Eq. 10-16 / 10-17):

        UC -> CIUC:  su(UC) / su(CIUC) = 0.893 + 0.513 * log10(su(UC) / sigma'v0)
        UU -> CIUC:  su(UU) / su(CIUC) = 0.911 + 0.499 * log10(su(UU) / sigma'v0)

    su(CIUC) = su / [a + b * log10(su / sigma'v0)].

    Note: the GEC-10 Appendix A worked example (Layer 4) applies the **UC** pair
    (0.893, 0.513) to its mean strength, so ``test_type="uc"`` reproduces that
    example (su=1,750 psf, sigma'vo=2,114 psf -> su(CIUC)=2,057 psf).

    Parameters
    ----------
    su : float
        Measured undrained shear strength (kPa).
    sigma_v0 : float
        Effective vertical stress at the layer mid-depth (kPa). Must be > 0.
    test_type : str, optional
        "uc" (default), "uu", or "ciuc". "ciuc" returns ``su`` unchanged.

    Returns
    -------
    float
        CIUC-equivalent undrained shear strength (kPa).

    References
    ----------
    Chen & Kulhawy (1993); GEC-10 (FHWA-NHI-18-024) Eq. 10-16 / 10-17.
    """
    tt = test_type.lower()
    if tt == "ciuc":
        return su
    if su <= 0 or sigma_v0 <= 0:
        raise ValueError(
            "su and sigma_v0 must be positive for the CIUC transform "
            f"(got su={su}, sigma_v0={sigma_v0})"
        )
    if tt == "uc":
        a, b = 0.893, 0.513
    elif tt == "uu":
        a, b = 0.911, 0.499
    else:
        raise ValueError(
            f"test_type must be 'uc', 'uu', or 'ciuc', got '{test_type}'"
        )
    denom = a + b * math.log10(su / sigma_v0)
    return su / denom


def alpha_cohesive_rational(su_ciuc: float, pa: float = PA) -> float:
    """Rational GEC-10 alpha factor for cohesive side resistance (Chen 2011).

    alpha = 0.30 + 0.17 / (su(CIUC) / pa)

    This is the FHWA-NHI-18-024 Figure 10-6 regression (Chen et al. 2011), which
    supersedes the AASHTO piecewise ``alpha_cohesive`` (0.55 for cu/pa <= 1.5)
    for GEC-10 design. The input MUST be the CIUC-equivalent strength — convert a
    UC/UU lab strength with ``su_to_ciuc`` first — and the mobilized unit side
    resistance is ``fs = alpha * su(CIUC)`` (the transformed strength, not the raw
    UU value).

    Parameters
    ----------
    su_ciuc : float
        CIUC-equivalent undrained shear strength (kPa). Must be > 0. The Chen
        regression is fitted over roughly su(CIUC) = 25-500 kPa.
    pa : float, optional
        Atmospheric pressure (kPa). Default 101.325.

    Returns
    -------
    float
        Alpha factor (dimensionless).

    References
    ----------
    Chen et al. (2011); GEC-10 (FHWA-NHI-18-024) Fig. 10-6 / Section 10.3.5.2.
    """
    if su_ciuc <= 0:
        raise ValueError(f"su_ciuc must be positive, got {su_ciuc}")
    return 0.30 + 0.17 / (su_ciuc / pa)


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


def beta_cohesionless(z: float, N60: float = None) -> float:
    """Beta factor for side resistance in cohesionless soil per GEC-10.

    beta = 1.5 - 0.245 * sqrt(z_ft), clamped to [0.25, 1.2]

    The original O'Neill & Reese (1999) formula uses z in feet.
    This implementation converts internally: z_ft = z_m * 3.28084.

    The formula (and the 200 kPa fs cap) presumes N60 >= 15. For looser
    sands (N60 < 15), O'Neill & Reese reduce beta proportionally:

        beta_reduced = (N60 / 15) * beta

    applied here to the clamped base value when ``N60`` is provided and
    less than 15. When ``N60`` is None (not measured), no reduction is
    applied — the caller is assuming a medium-dense or better sand.

    Parameters
    ----------
    z : float
        Depth below ground surface (m).
    N60 : float, optional
        Energy-corrected SPT blow count for the layer. Default None
        (no reduction; assumes N60 >= 15).

    Returns
    -------
    float
        Beta factor (dimensionless).

    References
    ----------
    Brown et al. (2010), GEC-10 Section 13.3.3.3
    O'Neill & Reese (1999), FHWA-RD-99-049
    AASHTO LRFD 10.8.3.5.2b (N60 < 15 reduction)
    """
    beta = 1.5 - 0.245 * math.sqrt(max(z, 0) * 3.28084)
    beta = max(0.25, min(beta, 1.2))
    if N60 is not None and 0 < N60 < 15:
        beta *= N60 / 15.0
    return beta


def phi_prime_from_N1_60(N1_60: float) -> float:
    """Effective friction angle phi' (degrees) from (N1)60 for granular soils.

    phi' = 27.5 + 9.2 * log10[(N1)60]

    Used to seed the GEC-10 rational beta chain (``beta_cohesionless_rational``).
    In the GEC-10 Appendix A example, (N1)60 = 21 -> phi' = 40 deg.

    References
    ----------
    GEC-10 (FHWA-NHI-18-024) Appendix A; Kulhawy & Mayne (1990).
    """
    if N1_60 <= 0:
        raise ValueError(f"N1_60 must be positive, got {N1_60}")
    return 27.5 + 9.2 * math.log10(N1_60)


def preconsolidation_stress(N60: float, pa: float = PA, m: float = 0.6) -> float:
    """Preconsolidation (yield) stress sigma'p (kPa) from N60 for granular soils.

    sigma'p = pa * 0.47 * (N60)^m,  with m = 0.6 (Mayne 2007).

    Feeds OCR = sigma'p / sigma'v in the rational beta chain. In the GEC-10
    Appendix A example, N60 = 30 -> sigma'p = 7,654 psf.

    References
    ----------
    Mayne (2007); GEC-10 (FHWA-NHI-18-024) Appendix A.
    """
    if N60 <= 0:
        raise ValueError(f"N60 must be positive, got {N60}")
    return pa * 0.47 * N60 ** m


def k0_from_ocr(phi: float, OCR: float) -> float:
    """At-rest earth-pressure coefficient Ko = (1 - sin phi') * OCR^(sin phi').

    Mayne & Kulhawy (1982). ``phi`` in degrees.
    """
    if OCR <= 0:
        raise ValueError(f"OCR must be positive, got {OCR}")
    s = math.sin(math.radians(phi))
    return (1.0 - s) * OCR ** s


def beta_cohesionless_rational(phi: float, OCR: float, delta: float = None,
                               cap_at_Kp: bool = True) -> float:
    """Rational GEC-10 beta factor for cohesionless side resistance.

        beta = Ko * tan(delta),   Ko = (1 - sin phi') * OCR^(sin phi')  (<= Kp),
        delta = phi' by default (cast-in-place concrete against sand).

    This is the Chen & Kulhawy / Mayne rational beta used in the GEC-10
    (FHWA-NHI-18-024) Appendix A example (phi'=40, OCR=1.65 -> Ko=0.49,
    beta=0.41). It is distinct from the depth-based O'Neill & Reese
    ``beta_cohesionless`` (1.5 - 0.245*sqrt(z_ft), clamped to [0.25, 1.2]).

    Parameters
    ----------
    phi : float
        Effective friction angle phi' (degrees). Typically from
        ``phi_prime_from_N1_60``.
    OCR : float
        Overconsolidation ratio (sigma'p / sigma'v). Must be > 0.
    delta : float, optional
        Shaft-soil interface friction angle (degrees). Default None -> delta = phi'.
    cap_at_Kp : bool, optional
        Limit Ko to the Rankine passive coefficient Kp = tan^2(45 + phi'/2)
        per the GEC-10 note (Ko < Kp). Default True.

    Returns
    -------
    float
        Beta factor (dimensionless).

    References
    ----------
    Chen & Kulhawy (2002); Mayne & Kulhawy (1982);
    GEC-10 (FHWA-NHI-18-024) Appendix A.
    """
    if delta is None:
        delta = phi
    Ko = k0_from_ocr(phi, OCR)
    if cap_at_Kp:
        Kp = math.tan(math.radians(45.0 + phi / 2.0)) ** 2
        Ko = min(Ko, Kp)
    return Ko * math.tan(math.radians(delta))


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
