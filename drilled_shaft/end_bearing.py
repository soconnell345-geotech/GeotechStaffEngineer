"""
End bearing methods for drilled shafts.

GEC-10 (FHWA-NHI-18-024, 2018) as printed, the default:

- cohesionless soil: q_BN = 0.60 N60 tsf = 57.5 N60 kPa, N60 limited to 50
  (Eq. 10-14, pp. 10-14, 10-15);
- cohesive soil: q_BN = N*c su (Eq. 10-19), N*c from Table 10-2 by su, and
  q_BN = (2/3)[1 + D/(6B)] N*c su where the depth D is less than 3B (Eq.
  10-20; p. 10-19);
- rock: q_BN = N*cr qu, N*cr = 2.5 (Eq. 10-23, p. 10-22).

The older methods are named options: in cohesive soil O'Neill and Reese
(1999) as FHWA NHI-06-089 prints it (Eqs. 9-39, 9-40, p. 9-131); the
large-base reductions of O'Neill and Reese (1999), 1.27/B in cohesionless
soil and F_r in cohesive soil, which GEC-10 (2018) does not print; and in
rock a reduction of q_b/q_u by RQD.

All units are SI: kPa, kN, meters.

References:
    FHWA GEC-10 (FHWA-NHI-18-024), Chapter 10, Sec. 10.3.5
    FHWA NHI-06-089, Sec. 9.12.2 (O'Neill & Reese 1999 as printed there)
    O'Neill & Reese (1999), FHWA-IF-99-025 (the large-base reductions; not
    in the reference library)
"""

import math

from drilled_shaft.side_resistance import KPA_PER_KSF

#: psf in one kPa.
PSF_PER_KPA = 1000.0 / KPA_PER_KSF

#: The cohesive base methods: GEC-10 (2018) Eqs. 10-19, 10-20 and Table
#: 10-2 (the default); O'Neill and Reese (1999) as FHWA NHI-06-089 prints it.
CLAY_BASE_METHODS = ("gec10", "oneill_reese")

#: The rock base methods: GEC-10 (2018) Eq. 10-23 with N*cr = 2.5 (the
#: default); q_b/q_u reduced by RQD.
ROCK_BASE_METHODS = ("gec10", "rqd")

#: GEC-10 (2018) Table 10-2 (p. 10-19): (su in psf, N*c). 'Linear
#: interpolation can be used for values between those tabulated.'
TABLE_10_2 = ((500.0, 6.5), (1000.0, 8.0), (2000.0, 9.0))

#: O'Neill and Reese's limiting unit end bearing in clay as FHWA NHI-06-089
#: prints it (p. 9-131): 80 ksf = 3830 kPa, 'not a theoretical limit but a
#: limit based on the largest measured values'. GEC-10 (2018) prints none.
QB_MAX_COHESIVE = 3830.0  # kPa

#: NHI-06-089 p. 9-131: 'If the soil within 2.0 diameters of the tip has
#: sut < 0.5 ksf, the value of Nc should be multiplied by 0.67.'
SOFT_CLAY_SU = 0.5 * KPA_PER_KSF  # kPa
SOFT_CLAY_NC_FACTOR = 0.67

# Base diameter above which the O'Neill-Reese large-base reduction
# applies in cohesive soil: 75 in = 1.90 m.
LARGE_BASE_DIAMETER = 1.90  # m

#: Base diameter above which the O'Neill-Reese reduction 1.27/B applies
#: in cohesionless soil (m).
LARGE_BASE_DIAMETER_SAND = 1.27  # m

_M_TO_IN = 39.3701
_KPA_TO_KSF = 1.0 / KPA_PER_KSF


def Nc_drilled_shaft(L_over_D: float) -> float:
    """Bearing capacity factor Nc of O'Neill and Reese (1999) in cohesive
    soil, as FHWA NHI-06-089 prints it (Eq. 9-40, p. 9-131):

        Nc = 6.0 [1 + 0.2 (z/D)] <= 9

    6.0 at z/D = 0, reaching 9.0 at z/D = 2.5. The method of
    ``clay_base_method='oneill_reese'``; GEC-10 (2018) takes N*c from
    Table 10-2 instead (:func:`nc_table_10_2`).

    Parameters
    ----------
    L_over_D : float
        Depth of penetration over shaft diameter, z/D.

    Returns
    -------
    float
        Bearing capacity factor Nc.
    """
    return min(6.0 * (1.0 + 0.2 * L_over_D), 9.0)


def nc_table_10_2(su: float) -> float:
    """N*c of GEC-10 (2018) Table 10-2 (p. 10-19) at su (kPa): 6.5 at 500
    psf, 8.0 at 1,000 psf, 9.0 at 2,000 psf and more, by linear
    interpolation between. The table starts at 500 psf; below it N*c is
    held at 6.5 (the result's choices say so)."""
    su_psf = su * PSF_PER_KPA
    if su_psf <= TABLE_10_2[0][0]:
        return TABLE_10_2[0][1]
    for (s0, n0), (s1, n1) in zip(TABLE_10_2, TABLE_10_2[1:]):
        if su_psf <= s1:
            return n0 + (n1 - n0) * (su_psf - s0) / (s1 - s0)
    return TABLE_10_2[-1][1]


def nc_below_table_10_2(su: float) -> float:
    """N*c at su (kPa) below 500 psf on the line through Table 10-2's
    first two rows (6.5 at 500 psf, 8.0 at 1,000 psf), which GEC-10 does
    not print: the alternative to holding 6.5. Not less than 0."""
    (s0, n0), (s1, n1) = TABLE_10_2[:2]
    return max(n0 - (n1 - n0) * (s0 - su * PSF_PER_KPA) / (s1 - s0), 0.0)


def end_bearing_cohesive(cu: float, tip_area: float,
                         L_over_D: float = 10.0,
                         base_diameter: float = None,
                         shaft_length: float = None,
                         method: str = "gec10",
                         large_base_reduction: bool = False) -> float:
    """End bearing in cohesive soil (kN): :func:`unit_end_bearing_cohesive`
    times the base area.

    Parameters
    ----------
    cu : float
        Undrained shear strength at the base (kPa).
    tip_area : float
        Base (bell) area (m²).
    L_over_D : float, optional
        Depth of the base over the shaft diameter: z/D in the O'Neill and
        Reese Nc, and D/B in GEC-10's Eq. 10-20 where ``shaft_length`` and
        ``base_diameter`` are not both given. Default 10.0.
    base_diameter : float, optional
        Base (bell) diameter B (m).
    shaft_length : float, optional
        Depth of the base D (m).
    method : str, optional
        "gec10" (default): GEC-10 (2018) Eqs. 10-19 and 10-20 with Table
        10-2. "oneill_reese": FHWA NHI-06-089 Eqs. 9-39, 9-40.
    large_base_reduction : bool, optional
        O'Neill and Reese's (1999) F_r for a base wider than 1.90 m.
        Default False: GEC-10 (2018) prints none.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10 (FHWA-NHI-18-024) Eqs. 10-19, 10-20, Table 10-2 (p. 10-19);
    FHWA NHI-06-089 Eqs. 9-39, 9-40 (p. 9-131); O'Neill & Reese (1999)
    """
    return unit_end_bearing_cohesive(
        cu, L_over_D, base_diameter, shaft_length, method,
        large_base_reduction)["qb"] * tip_area


def unit_end_bearing_cohesive(cu: float, L_over_D: float = 10.0,
                              base_diameter: float = None,
                              shaft_length: float = None,
                              method: str = "gec10",
                              large_base_reduction: bool = False) -> dict:
    """The unit end bearing in cohesive soil and each step to it (GM-148,
    GM-244), so the calc package prints the values the analysis used.

    ``method`` "gec10" (GEC-10 (2018), p. 10-19): ``su_psf``, ``Nc`` from
    Table 10-2, ``below_table`` (su under 500 psf, N*c held at 6.5),
    ``D_over_B`` and ``equation`` ("10-19" where D >= 3B, else "10-20"
    with ``depth_factor`` (2/3)[1 + D/(6B)]). ``method`` "oneill_reese"
    (FHWA NHI-06-089 p. 9-131): ``Nc`` = 6[1 + 0.2 L/D] <= 9,
    ``soft_clay`` (su < 0.5 ksf, Nc x 0.67), ``qb_uncapped`` and
    ``capped`` (the 80 ksf limit). Both: ``method``, ``qb_unreduced``
    (kPa, before F_r), ``Fr`` (None unless ``large_base_reduction`` and a
    base wider than 1.90 m), ``a``, ``b``, ``L`` and ``qb`` (kPa).
    """
    if method not in CLAY_BASE_METHODS:
        raise ValueError(f"clay_base_method must be one of "
                         f"{list(CLAY_BASE_METHODS)}, got {method!r}")
    out = {"method": method, "Fr": None, "a": None, "b": None, "L": None}
    if method == "gec10":
        if shaft_length is not None and base_diameter:
            D_over_B = shaft_length / base_diameter
        else:
            D_over_B = L_over_D
        Nc = nc_table_10_2(cu)
        factor = 1.0 if D_over_B >= 3.0 else 2.0 / 3.0 * (1 + D_over_B / 6)
        qb = factor * Nc * cu
        out.update(su_psf=cu * PSF_PER_KPA, Nc=Nc,
                   below_table=cu * PSF_PER_KPA < TABLE_10_2[0][0],
                   D_over_B=D_over_B,
                   equation="10-19" if D_over_B >= 3.0 else "10-20",
                   depth_factor=factor)
    else:
        Nc = Nc_drilled_shaft(L_over_D)
        soft = cu < SOFT_CLAY_SU
        if soft:
            Nc *= SOFT_CLAY_NC_FACTOR
        qb0 = Nc * cu
        qb = min(qb0, QB_MAX_COHESIVE)
        out.update(Nc=Nc, soft_clay=soft, qb_uncapped=qb0,
                   capped=qb0 > QB_MAX_COHESIVE)
    out["qb_unreduced"] = qb
    if large_base_reduction and base_diameter is not None \
            and base_diameter > LARGE_BASE_DIAMETER:
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
                             diameter: float = 1.0,
                             large_base_reduction: bool = False) -> float:
    """End bearing in cohesionless soil (kN).

    qb = 57.5 * N60 (kPa), with N60 limited to 50: GEC-10 (2018) Eq. 10-14,
    q_BN = 0.60 N60 tsf (p. 10-14), 'limited to cohesionless soils with SPT
    N-values of 50 blows per foot or less' (p. 10-15). With
    ``large_base_reduction``, O'Neill and Reese's (1999) 1.27/B for a base
    wider than 1.27 m, which GEC-10 (2018) does not print.

    Parameters
    ----------
    N60 : float
        Energy-corrected SPT blow count at tip.
    tip_area : float
        Shaft tip (base/bell) area (m²).
    diameter : float, optional
        BASE diameter B (m), the bell's for a belled shaft: the 1.27/B
        reduction is governed by the bearing surface. Default 1.0.
    large_base_reduction : bool, optional
        Apply 1.27/B where B > 1.27 m. Default False.

    Returns
    -------
    float
        End bearing capacity (kN).

    References
    ----------
    GEC-10 (FHWA-NHI-18-024) Eq. 10-14, p. 10-14 (N60 limited to 50, p.
    10-15); FHWA NHI-06-089 Eq. 9-45a (p. 9-133) prints the same q_t =
    1.2 N60 ksf; the 1.27/B reduction is O'Neill & Reese (1999)
    """
    return unit_end_bearing_cohesionless(
        N60, diameter, large_base_reduction)["qb"] * tip_area


def unit_end_bearing_cohesionless(N60: float, diameter: float = 1.0,
                                  large_base_reduction: bool = False
                                  ) -> dict:
    """The unit end bearing in cohesionless soil and each step (GM-148):
    ``N60_capped``, ``qb_unreduced`` (57.5 N60), ``reduction`` (1.27/B
    with B the base diameter, None unless ``large_base_reduction`` and B >
    1.27 m) and ``qb`` (kPa); see :func:`end_bearing_cohesionless`."""
    N60_capped = min(N60, 50.0)
    qb0 = 57.5 * N60_capped  # kPa
    reduction = (LARGE_BASE_DIAMETER_SAND / diameter
                 if large_base_reduction
                 and diameter > LARGE_BASE_DIAMETER_SAND else None)
    return {"N60_capped": N60_capped, "qb_unreduced": qb0,
            "reduction": reduction,
            "qb": qb0 * reduction if reduction else qb0}


def end_bearing_rock(qu: float, tip_area: float,
                     RQD: float = 100.0, method: str = "gec10") -> float:
    """End bearing in rock (kN).

    ``method`` "gec10" (default): q_BN = N*cr qu with N*cr = 2.5, GEC-10
    (2018) Eq. 10-23 (p. 10-22), 'recommended for design when qu is the
    sole parameter used for establishing qBN' and the base bears on rock
    massive or tightly jointed to one diameter below it, with no cavities
    or voids beneath and a clean base. ``method`` "rqd": q_b/q_u from
    :func:`rock_end_bearing_multiplier` by RQD.

    Parameters
    ----------
    qu : float
        Unconfined compressive strength (kPa).
    tip_area : float
        Shaft tip area (m²).
    RQD : float, optional
        Rock Quality Designation (%), used by ``method`` "rqd". Default 100.
    method : str, optional
        "gec10" or "rqd".

    Returns
    -------
    float
        End bearing capacity (kN).
    """
    return rock_base_factor(RQD, method) * qu * tip_area


def rock_base_factor(RQD: float, method: str = "gec10") -> float:
    """q_b / q_u in rock: N*cr = 2.5 (GEC-10 Eq. 10-23) for ``method``
    "gec10", :func:`rock_end_bearing_multiplier` for "rqd"."""
    if method not in ROCK_BASE_METHODS:
        raise ValueError(f"rock_base_method must be one of "
                         f"{list(ROCK_BASE_METHODS)}, got {method!r}")
    return 2.5 if method == "gec10" else rock_end_bearing_multiplier(RQD)


def rock_end_bearing_multiplier(RQD: float) -> float:
    """q_b / q_u of ``rock_base_method='rqd'``: 2.5 at RQD >= 70 %, then
    linear to 1.5 at 50 %, 0.8 at 25 % and 0.4 at 0 %. This module's
    earlier rock base, cited to GEC-10 (2010 edition) Sec. 13.3.4.4 and
    the Canadian Foundation Engineering Manual, neither in the reference
    library, so not checked against a printed page. GEC-10 (2018) gives
    N*cr = 2.5 (Eq. 10-23) for competent rock and the Hoek-Brown base
    (Eqs. 10-24 to 10-29, pp. 10-23, 10-24) for a fractured rock mass."""
    if RQD >= 70:
        return 2.5
    if RQD >= 50:
        return 1.5 + (RQD - 50) / 20 * 1.0
    if RQD >= 25:
        return 0.8 + (RQD - 25) / 25 * 0.7
    return 0.4 + RQD / 25 * 0.4
