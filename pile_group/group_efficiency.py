"""
Pile group resistance, lateral group effects and group settlement, as
FHWA GEC-12 (FHWA-NHI-16-009, Vol. I ch. 7) and UFC 3-220-20 (2025,
ch. 6) print them (checked against the print in GM-286,
``module_work/gm286/CHECK.md``).

- :func:`group_efficiency_factor`: the group efficiency factor eta_g,
  GEC-12 Secs. 7.2.2.1 and 7.2.2.2 (driven piles, pp. 282-284) and UFC
  3-220-20 Table 6-26 (p. 432; driven piles, drilled shafts, CFAs, DDPs,
  micropiles).
- :func:`block_failure_resistance`: the group's nominal resistance to
  block failure, GEC-12 Eqs. 7-36 and 7-37 (p. 286); UFC Eqs. 6-38 and
  6-39 print the same.
- :func:`p_multiplier`: the lateral p-multiplier of a row, as AASHTO
  prints it (GEC-12 pp. 367 and 370; UFC Table 6-38, p. 466), or by
  Mokwa's (1999) equations (UFC Table 6-38).
- :func:`group_settlement_equivalent_raft`: the equivalent footing,
  GEC-12 Sec. 7.3.5.3 (Eq. 7-55, Fig. 7-44); UFC Sec. 6-5.8.4.
- :func:`meyerhof_group_settlement`: Meyerhof (1976), GEC-12 Eqs. 7-50
  to 7-52 (p. 319); UFC Eqs. 6-46, 6-47.

All units are SI: meters, kPa, kN.
"""

import math
from typing import Optional

#: 1 ksf in kPa: GEC-12 and UFC set the eta_g rows at su = 2 ksf.
_KPA_PER_KSF = 47.88
#: su below which a clay's eta_g row depends on the cap's contact (2 ksf).
SU_LIMIT_KPA = 2.0 * _KPA_PER_KSF

#: Pile types UFC 3-220-20 Table 6-26 prints a column for.
PILE_TYPES = ("driven", "drilled_shaft", "cfa", "ddp", "micropile")
SOIL_TYPES = ("cohesionless", "cohesive")

_PILE_TYPE_TEXT = {
    "driven": "driven piles", "drilled_shaft": "drilled shafts",
    "cfa": "CFA piles", "ddp": "drilled displacement piles (DDPs)",
    "micropile": "micropiles",
}

# UFC 3-220-20 Table 6-26 (p. 432), by pile type and row: a constant
# eta_g, or the printed (s/b, eta_g) points, interpolated linearly between
# ("Interpolation can be applied to evaluate eta_g for intermediate
# spacings", p. 431) and held at the last point beyond it. The driven-pile
# column is GEC-12 Secs. 7.2.2.1 and 7.2.2.2 (pp. 282, 284). Rows:
# coarse-grained (cohesionless) soil, cap not in contact / in contact; and
# fine-grained (cohesive) soil, su < 2 ksf, cap not in contact / in
# contact; su >= 2 ksf.
_ETA_G = {
    "driven": {
        "cohesionless_free": 1.0, "cohesionless_contact": 1.0,
        "soft_free": ((2.5, 0.65), (6.0, 1.0)), "soft_contact": 1.0,
        "stiff": 1.0},
    "drilled_shaft": {
        "cohesionless_free": ((2.5, 0.65), (3.0, 0.80), (4.0, 1.0)),
        "cohesionless_contact": 1.0,
        "soft_free": 1.0, "soft_contact": 1.0, "stiff": 1.0},
    "cfa": {
        "cohesionless_free": ((2.5, 0.65), (6.0, 1.0)),
        "cohesionless_contact": ((2.5, 0.65), (6.0, 1.0)),
        "soft_free": 1.0, "soft_contact": 1.0, "stiff": 1.0},
    "ddp": {
        "cohesionless_free": 1.0, "cohesionless_contact": 1.0,
        "soft_free": ((3.0, 0.7), (6.0, 1.0)), "soft_contact": 1.0,
        "stiff": 1.0},
    "micropile": {
        "cohesionless_free": 1.0, "cohesionless_contact": 1.0,
        "soft_free": ((2.5, 0.65), (3.0, 0.70), (6.0, 1.0)),
        "soft_contact": 1.0, "stiff": 1.0},
}

# The minimum spacing, s/b, Table 6-26's footnotes print for each pile
# type (driven piles: AASHTO's 2.5, also GEC-12 p. 282). Below it no row
# applies.
_MIN_S_OVER_B = {"driven": 2.5, "drilled_shaft": 2.5, "cfa": 3.0,
                 "ddp": 3.0, "micropile": 3.0}

_ROW_TEXT = {
    "cohesionless_free": "cohesionless (coarse-grained) soil, pile cap not "
                         "in contact with the ground",
    "cohesionless_contact": "cohesionless (coarse-grained) soil, pile cap "
                            "in contact with the ground",
    "soft_free": "cohesive (fine-grained) soil, su < 2 ksf, pile cap not "
                 "in contact with the ground",
    "soft_contact": "cohesive (fine-grained) soil, su < 2 ksf, pile cap in "
                    "contact with the ground",
    "stiff": "cohesive (fine-grained) soil, su >= 2 ksf",
}


def _read(entry, s_over_b: float) -> float:
    """eta_g from one row of Table 6-26: a constant, or the printed
    points interpolated linearly and held at the last."""
    if not isinstance(entry, tuple):
        return float(entry)
    if s_over_b <= entry[0][0]:
        return entry[0][1]
    for (s0, e0), (s1, e1) in zip(entry, entry[1:]):
        if s_over_b <= s1:
            return e0 + (e1 - e0) * (s_over_b - s0) / (s1 - s0)
    return entry[-1][1]


def _row(soil_type: str, cap_in_contact: bool, soft: bool) -> str:
    if soil_type == "cohesionless":
        return ("cohesionless_contact" if cap_in_contact
                else "cohesionless_free")
    if not soft:
        return "stiff"
    return "soft_contact" if cap_in_contact else "soft_free"


def group_efficiency_factor(soil_type: str, s_over_b: float,
                            cap_in_contact: Optional[bool] = None,
                            su_kPa: Optional[float] = None,
                            pile_type: str = "driven") -> dict:
    """The group efficiency factor eta_g, as GEC-12 and UFC 3-220-20 print it.

    eta_g = R_ng / (n R_n) (GEC-12 Eq. 7-35); the group's resistance is
    the lesser of n eta_g R_n and its resistance to block failure (GEC-12
    p. 284 step 4; UFC Eq. 6-37), see :func:`block_failure_resistance`.

    Driven piles, GEC-12 Secs. 7.2.2.1 and 7.2.2.2 (pp. 282, 284):
    cohesionless soil not underlain by a weak deposit, 1.0 (jetting or
    predrilling "can result in group efficiencies less than 1", no value
    printed); clay with su < 2 ksf and the cap not in firm contact with
    the ground, 0.65 at s = 2.5 diameters to 1.0 at 6, linear between;
    su < 2 ksf with the cap in firm contact, or su above 2 ksf, 1.0. UFC
    3-220-20 Table 6-26 (p. 432) prints the same for driven piles and
    the other pile types' columns.

    Parameters
    ----------
    soil_type : str
        'cohesionless' (UFC's coarse-grained) or 'cohesive'
        (fine-grained): the stratum providing most of the resistance, the
        controlling stratum (GEC-12 p. 281).
    s_over_b : float
        Centre-to-centre spacing over the pile diameter or width.
    cap_in_contact : bool, optional
        Whether the pile cap is in firm contact with the ground. Left out
        where the row depends on it, the lower eta_g is taken (cap not in
        contact) and the result says so (``cap_in_contact_taken``), with
        the other value as ``alternative``.
    su_kPa : float, optional
        Undrained shear strength of the clay (kPa), set against 2 ksf
        (95.76 kPa). Left out where the row depends on it, su < 2 ksf (the
        lower eta_g) is taken and the result says so (``su_taken``).
    pile_type : str
        'driven' (default; GEC-12's subject), 'drilled_shaft', 'cfa',
        'ddp' or 'micropile': the column of UFC Table 6-26.

    Returns
    -------
    dict
        ``eta_g``; ``row`` (the printed row, in words); ``points`` (the
        printed (s/b, eta_g) points read, or None for a constant);
        ``cap_in_contact`` and ``su_below_2ksf`` as used (None where the
        row did not need them); ``cap_in_contact_taken`` / ``su_taken``
        (True where left out and taken); ``alternative`` (eta_g had the
        taken inputs gone the other way, or None).

    Raises
    ------
    ValueError
        For an unknown soil or pile type, or s/b below the minimum spacing
        Table 6-26's footnotes print for the pile type.
    """
    if soil_type not in SOIL_TYPES:
        raise ValueError(f"soil_type must be one of {list(SOIL_TYPES)}, "
                         f"got {soil_type!r}")
    if pile_type not in PILE_TYPES:
        raise ValueError(f"pile_type must be one of {list(PILE_TYPES)}, "
                         f"got {pile_type!r}")
    if not (s_over_b > 0 and math.isfinite(s_over_b)):
        raise ValueError(f"s/b must be positive, got {s_over_b}")
    s_min = _MIN_S_OVER_B[pile_type]
    if s_over_b < s_min - 1e-9:
        raise ValueError(
            f"s/b = {s_over_b:.2f} is below the minimum spacing UFC "
            f"3-220-20 Table 6-26 prints for {_PILE_TYPE_TEXT[pile_type]}, "
            f"s/b = {s_min:g}; no group efficiency is printed below it.")
    table = _ETA_G[pile_type]

    def eta(contact, soft):
        return _read(table[_row(soil_type, contact, soft)], s_over_b)

    soft = None
    su_taken = False
    if soil_type == "cohesive":
        if su_kPa is not None:
            if not su_kPa > 0:
                raise ValueError(f"su must be positive, got {su_kPa}")
            soft = su_kPa < SU_LIMIT_KPA
        elif any(eta(c, True) != eta(c, False)
                 for c in ((False, True) if cap_in_contact is None
                           else (cap_in_contact,))):
            soft, su_taken = True, True
    contact = cap_in_contact
    contact_taken = False
    if contact is None:
        rows_differ = (eta(False, bool(soft)) != eta(True, bool(soft))
                       if soil_type == "cohesive" else
                       eta(False, None) != eta(True, None))
        if rows_differ:
            contact, contact_taken = False, True
    used_contact = bool(contact)
    used_soft = bool(soft)
    value = eta(used_contact, used_soft)
    alternative = None
    if contact_taken or su_taken:
        alternative = eta(True if contact_taken else used_contact,
                          False if su_taken else used_soft)
    entry = table[_row(soil_type, used_contact, used_soft)]
    return {
        "eta_g": value,
        "pile_type": pile_type,
        "soil_type": soil_type,
        "s_over_b": s_over_b,
        "row": _ROW_TEXT[_row(soil_type, used_contact, used_soft)],
        "points": list(entry) if isinstance(entry, tuple) else None,
        "cap_in_contact": None if contact is None else used_contact,
        "cap_in_contact_taken": contact_taken,
        "su_below_2ksf": soft,
        "su_taken": su_taken,
        "alternative": alternative,
    }


def block_failure_resistance(B: float, Z: float, D: float,
                             su1: float, su2: float) -> dict:
    """Nominal resistance of a pile group to block failure.

    GEC-12 Eq. 7-36 (p. 286):  R_ng = 2 D (B + Z) su1 + B Z su2 Nc

    with Nc by GEC-12 Eq. 7-37:  Nc = 5 [1 + D/(5B)] [1 + B/(5Z)] <= 9

    ("generally 9", but by the equation "for pile groups with small pile
    embedment depths and/or large widths"; the equation gives 9 wherever
    9 applies). UFC 3-220-20 Eqs. 6-38 and 6-39 (p. 433) print the same
    in its own symbols (its Z the embedded length, L the group length).

    Parameters
    ----------
    B, Z : float
        Width and length of the pile group (m), to the outer faces of the
        outer piles (GEC-12 Fig. 7-31; Vol. III p. 62 takes them "from
        exterior pile edge to exterior pile edge"). B is the smaller; given
        the other way round, they are swapped.
    D : float
        Pile embedded length (m).
    su1 : float
        Weighted average undrained shear strength over the embedded length
        along the group's perimeter (kPa); for cohesionless soils along
        the sides, their weighted average unit shaft resistance (GEC-12
        p. 287; UFC's f_s,1).
    su2 : float
        Average undrained shear strength below the pile toes to a depth of
        2B (kPa; GEC-12 p. 286; UFC p. 433: 2B to 3B, whichever gives the
        lower), that of the weak layer for a group in cohesionless soil
        over it.

    Returns
    -------
    dict
        ``R_ng_kN``, ``Nc`` (as used), ``Nc_eq_7_37`` (the equation before
        the limit of 9), ``side_kN``, ``base_kN``, ``B_m``, ``Z_m``,
        ``D_m``, ``su1_kPa``, ``su2_kPa``.
    """
    for name, v in (("B", B), ("Z", Z), ("D", D), ("su1", su1),
                    ("su2", su2)):
        if not (v > 0 and math.isfinite(v)):
            raise ValueError(f"{name} must be positive, got {v}")
    B, Z = min(B, Z), max(B, Z)
    nc_eq = 5.0 * (1.0 + D / (5.0 * B)) * (1.0 + B / (5.0 * Z))
    Nc = min(nc_eq, 9.0)
    side = 2.0 * D * (B + Z) * su1
    base = B * Z * su2 * Nc
    return {
        "R_ng_kN": side + base,
        "Nc": Nc,
        "Nc_eq_7_37": nc_eq,
        "side_kN": side,
        "base_kN": base,
        "B_m": B, "Z_m": Z, "D_m": D,
        "su1_kPa": su1, "su2_kPa": su2,
    }


def group_plan_dimensions(n_rows: int, n_cols: int, spacing_x: float,
                          spacing_y: float, pile_diameter: float) -> tuple:
    """Width and length of a rectangular group (m), to the outer faces of
    the outer piles (GEC-12 Fig. 7-31): (n_cols - 1) s_x + d along x and
    (n_rows - 1) s_y + d along y, the smaller first."""
    if pile_diameter <= 0:
        raise ValueError(f"Pile diameter must be positive, got {pile_diameter}")
    bx = (n_cols - 1) * spacing_x + pile_diameter
    by = (n_rows - 1) * spacing_y + pile_diameter
    return min(bx, by), max(bx, by)


def block_failure_capacity(n_rows: int, n_cols: int,
                           spacing_x: float, spacing_y: float,
                           pile_length: float,
                           cu: float,
                           pile_diameter: float,
                           su2: Optional[float] = None) -> float:
    """Block failure resistance (kN) of a rectangular group: GEC-12 Eqs.
    7-36 and 7-37 (:func:`block_failure_resistance`) on the group's plan
    dimensions to the outer pile faces, with ``cu`` as su1 and ``su2``
    (default ``cu``, one strength for both) below the toes."""
    B, Z = group_plan_dimensions(n_rows, n_cols, spacing_x, spacing_y,
                                 pile_diameter)
    return block_failure_resistance(B, Z, pile_length, cu,
                                    cu if su2 is None else su2)["R_ng_kN"]


# The AASHTO p-multipliers by row (rows 1, 2, 3 and 4+), at s/b = 3 and 5
# as GEC-12 prints them (pp. 367, 370: "0.8 for the lead row, 0.4 for the
# second row, and 0.3 for the third and subsequent rows" at 3b; 1.0, 0.85
# and 0.7 at 5b; "Interpolate for center to center pile spacing between 3b
# and 5b"), and 1.00 at s/b = 6, UFC Table 6-38's "AASHTO (2020) guidance,
# extrapolated to a pile spacing of 6 b" (p. 465).
_PM_AASHTO = ((3.0, (0.80, 0.40, 0.30, 0.30)),
              (5.0, (1.00, 0.85, 0.70, 0.70)),
              (6.0, (1.00, 1.00, 1.00, 1.00)))
# Mokwa (1999), UFC Table 6-38: p_m = a (s/b) + c <= 1, rows 1, 2, 3, 4+.
_PM_MOKWA = ((0.06, 0.64), (0.11, 0.34), (0.14, 0.16), (0.16, 0.04))

P_MULTIPLIER_METHODS = ("aashto", "mokwa")


def p_multiplier(row_position: int, spacing_diameter_ratio: float,
                 method: str = "aashto") -> float:
    """Lateral p-multiplier p_m of a row of piles in a group.

    'aashto' (default): AASHTO's p_m as GEC-12 prints it (pp. 367, 370)
    at s/b = 3 (0.80, 0.40, 0.30 for rows 1, 2, 3 and higher) and 5
    (1.00, 0.85, 0.70), interpolated linearly between; from 5 to 6
    interpolated to 1.00 at 6 and 1.00 beyond, UFC 3-220-20 Table 6-38's
    extension to 6b (p. 465), Table 6-38 printing 1.00 for s/b > 6.
    GEC-12 Vol. III takes 0.90, 0.625 and 0.5 at 4b (pp. 56, 137).

    'mokwa': Mokwa (1999), UFC Table 6-38: p_m = 0.06 (s/b) + 0.64,
    0.11 (s/b) + 0.34, 0.14 (s/b) + 0.16 and 0.16 (s/b) + 0.04 for rows
    1, 2, 3 and 4 and higher, each <= 1 (UFC Table B-32 works s/b = 3).

    Rows are numbered from the lead row, 1, the row farthest in the
    direction of the load (UFC Fig. 6-25).

    Raises ValueError below s/b = 3: GEC-12 prints p_m from 3b, and UFC
    p. 465: "it is not advisable to install elements closer than 3 b".
    """
    if method not in P_MULTIPLIER_METHODS:
        raise ValueError(f"p_multiplier method must be one of "
                         f"{list(P_MULTIPLIER_METHODS)}, got {method!r}")
    if isinstance(row_position, bool) or int(row_position) != row_position \
            or row_position < 1:
        raise ValueError(f"row_position must be a whole number, 1 (the "
                         f"lead row) or more; got {row_position!r}")
    sd = spacing_diameter_ratio
    if not (sd > 0 and math.isfinite(sd)):
        raise ValueError(f"s/D must be positive, got {sd}")
    if sd < 3.0 - 1e-9:
        raise ValueError(
            f"s/D = {sd:.2f} is below 3: GEC-12 (pp. 367, 370) and UFC "
            f"3-220-20 Table 6-38 print p-multipliers from s/D = 3, and "
            f"UFC p. 465 says it is not advisable to install piles closer "
            f"than 3 diameters.")
    i = min(int(row_position), 4) - 1
    if method == "mokwa":
        a, c = _PM_MOKWA[i]
        return min(a * sd + c, 1.0)
    if sd >= 6.0:
        return 1.0
    for (s0, p0), (s1, p1) in zip(_PM_AASHTO, _PM_AASHTO[1:]):
        if sd <= s1:
            return p0[i] + (p1[i] - p0[i]) * (sd - s0) / (s1 - s0)
    return 1.0


def _influence_depth_10pct(B: float, L: float) -> float:
    """Depth below the equivalent footing where the 2V:1H stress falls to
    10 % of the footing's bearing stress: (B + z)(L + z) = 10 B L."""
    return (-(B + L) + math.sqrt((B + L) ** 2 + 36.0 * B * L)) / 2.0


def group_settlement_equivalent_raft(
    n_rows: int,
    n_cols: int,
    spacing: float,
    pile_diameter: float,
    pile_length: float,
    load_kN: float,
    soil_modulus_kPa: float,
    num_sublayers: int = 10,
    influence_depth_m: Optional[float] = None,
) -> dict:
    """Group settlement by the equivalent footing (GEC-12 Sec. 7.3.5.3).

    Terzaghi and Peck's (1967) equivalent footing at 1/3 D above the pile
    toe, i.e. 2/3 D below the top of the embedded length (GEC-12 p. 320,
    Fig. 7-44; UFC 3-220-20 Table 6-32's first row, "Sand or clay": z1 =
    2Zb/3, z2 = 0), with plan area B x Z the perimeter dimensions of the
    group (GEC-12 p. 320, to the outer pile faces as in Fig. 7-31):

        Bg = (n_cols - 1) * s + d,   Lg = (n_rows - 1) * s + d

    The load spreads at 2V:1H (Fig. 7-44's 1H:2V; GEC-12 Eq. 7-55, UFC
    Eq. 6-51; the text of p. 320 says "side slopes at 30 deg", which the
    figure and the equation do not take):

        delta_sigma(z) = Q / [(Bg + z) * (Lg + z)]

    Settlement sums the one-dimensional compression of sublayers,

        S = SUM( delta_sigma_i * dz_i / M )

    M the constrained modulus, Es (1 - nu) / [(1 + nu)(1 - 2 nu)] (UFC
    Eq. 6-54, elastic compression of coarse-grained soil). The sum runs to
    the depth below the footing where delta_sigma falls to 10 % of the
    footing's bearing stress Q / (Bg Lg) (UFC p. 449: "This depth is
    typically selected where the change in stress is less than 10% of the
    bearing stress of the equivalent footing"), or to
    ``influence_depth_m`` where given (the compressible layer's base, for
    one). GEC-12 Fig. 7-46 and UFC Table 6-32 place the footing elsewhere
    for other profiles (soft over firm clay, toe-bearing piles over soft
    clay, sand over clay, layered soils); this function takes the uniform
    case.

    Parameters
    ----------
    n_rows, n_cols : int
        Rows and columns of the group (>= 1).
    spacing : float
        Center-to-center pile spacing, s (m), both directions.
    pile_diameter : float
        Pile diameter or width, d (m).
    pile_length : float
        Pile embedded length, D (m).
    load_kN : float
        The unfactored permanent load on the group, Q (kN) (GEC-12 p. 320:
        AASHTO "states that the load used in calculating group settlement
        is the permanently applied load").
    soil_modulus_kPa : float
        Constrained modulus M of the soil below the footing (kPa); a
        weighted average for layered soils.
    num_sublayers : int
        Sublayers the influence zone is divided into. Default 10.
    influence_depth_m : float, optional
        Depth below the equivalent footing to sum to (m). Left out: the
        10 % depth above.

    Returns
    -------
    dict
        ``settlement_m``, ``settlement_mm``, ``Bg_m``, ``Lg_m``,
        ``raft_depth_m`` (below the top of the embedded length),
        ``influence_depth_m``, ``max_stress_kPa`` (the footing's bearing
        stress).

    References
    ----------
    FHWA GEC-12 (FHWA-NHI-16-009), Sec. 7.3.5.3, Eq. 7-55, Fig. 7-44.
    UFC 3-220-20 (2025), Sec. 6-5.8.4: Table 6-32, Eqs. 6-51, 6-54, p. 449.
    """
    # --- Input validation ---
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if pile_diameter <= 0:
        raise ValueError(f"Pile diameter must be positive, got {pile_diameter}")
    if pile_length <= 0:
        raise ValueError(f"Pile length must be positive, got {pile_length}")
    if load_kN <= 0:
        raise ValueError(f"Load must be positive, got {load_kN}")
    if soil_modulus_kPa <= 0:
        raise ValueError(f"Soil modulus must be positive, got {soil_modulus_kPa}")
    if num_sublayers < 1:
        raise ValueError(f"num_sublayers must be >= 1, got {num_sublayers}")
    if influence_depth_m is not None and not influence_depth_m > 0:
        raise ValueError(f"influence_depth_m must be positive, got "
                         f"{influence_depth_m}")

    # --- Equivalent footing: plan to the outer pile faces ---
    Bg = (n_cols - 1) * spacing + pile_diameter
    Lg = (n_rows - 1) * spacing + pile_diameter

    # --- Its depth: 1/3 D above the toe (Terzaghi and Peck 1967) ---
    raft_depth = (2.0 / 3.0) * pile_length

    # --- Influence zone: to 10 % of the bearing stress, or as given ---
    influence_depth = (_influence_depth_10pct(Bg, Lg)
                       if influence_depth_m is None else influence_depth_m)

    # --- Discretize and integrate settlement ---
    dz = influence_depth / num_sublayers
    settlement = 0.0

    for i in range(num_sublayers):
        z_mid = (i + 0.5) * dz  # mid-depth of sublayer below the footing

        # 2V:1H stress distribution (GEC-12 Eq. 7-55)
        delta_sigma = load_kN / ((Bg + z_mid) * (Lg + z_mid))

        settlement += delta_sigma * dz / soil_modulus_kPa

    # Bearing stress of the equivalent footing (z = 0)
    max_stress = load_kN / (Bg * Lg)

    return {
        "settlement_m": round(settlement, 6),
        "settlement_mm": round(settlement * 1000.0, 3),
        "Bg_m": round(Bg, 4),
        "Lg_m": round(Lg, 4),
        "raft_depth_m": round(raft_depth, 4),
        "influence_depth_m": round(influence_depth, 4),
        "max_stress_kPa": round(max_stress, 3),
    }


# --- Unit-conversion constants for the Meyerhof (1976) SPT method ------------
# The Meyerhof formula's leading coefficient "4" is calibrated for US customary
# units (settlement in INCHES, pressure pf in ksf, group width B in FEET).
_KSF_PER_KPA = 1.0 / _KPA_PER_KSF   # 1 kPa = 1/47.88 ksf
_FT_PER_M = 1.0 / 0.3048     # 1 m = 1/0.3048 ft
_MM_PER_IN = 25.4            # 1 in = 25.4 mm


def meyerhof_group_settlement(
    group_width: float,
    group_length: float,
    N160: float,
    load_kN: Optional[float] = None,
    pf_kPa: Optional[float] = None,
    embedment_DB: float = 0.0,
    silty_sand: bool = False,
) -> dict:
    """Meyerhof (1976) SPT pile-group settlement, GEC-12 Eqs. 7-50 to 7-52.

    GEC-12 Sec. 7.3.5.2.1 (p. 319), for "a pile group in a homogeneous
    sand deposit not underlain by a more compressible soil at a greater
    depth" (conservative); in US customary units (settlement in INCHES):

        S[in] = 4 * pf[ksf] * If * sqrt(B[ft]) / N1(60)      (Eq. 7-50)
        S[in] = 8 * pf[ksf] * If * sqrt(B[ft]) / N1(60)      (Eq. 7-51,
                                                              silty sand)
        If = 1 - D'/(8 B) >= 0.5                            (Eq. 7-52)

    where pf = the design foundation pressure, Q / (B Z) (GEC-12 Vol. III
    p. 62: the unfactored permanent load over the group's plan area, B and
    Z "from exterior pile edge to exterior pile edge"); B = the group's
    width (the smaller plan dimension); N1(60) = the average corrected SPT
    N within a depth B below the pile toe; D' = 2/3 of the embedment
    ``embedment_DB``.

    The print defines D' two ways. GEC-12 Vol. I (p. 319): "2/3 pile
    embedded length"; UFC 3-220-20 Eq. 6-47, If = max(1 - Z/(12B), 0.5),
    Z the group's embedded length (Fig. 6-15), is the same. GEC-12 Vol.
    III Block 15 (p. 63, Table D-23): 2/3 of DB, the "pile embedded length
    into bearing stratum". ``embedment_DB`` takes whichever the engineer
    reads; left at 0, If = 1.0, the upper bound. UFC Eq. 6-46 prints the
    expression without the leading 4; GEC-12 Eq. 7-50 and its worked
    example (Vol. III p. 63, 1.04 in) carry it, and this function follows
    GEC-12. For silty sand UFC doubles the estimate, as Eq. 7-51 does.

    This module is SI: the inputs (m, kN, kPa) are converted to US units,
    the published formula applied verbatim, and the settlement (inches)
    converted back to mm:

        pf[ksf] = pf[kPa] / 47.88,  B[ft] = B[m] / 0.3048,  S[mm] = 25.4 S[in]

    Parameters
    ----------
    group_width : float
        Group plan width, B (m), the smaller plan dimension; given larger
        than ``group_length``, the two are swapped.
    group_length : float
        Group plan length, Z (m). Forms the plan area with the width when
        ``load_kN`` is given.
    N160 : float
        Average corrected SPT N1(60) within a depth B below the pile toe.
    load_kN : float, optional
        Unfactored permanent load on the group, Q (kN); pf = Q / (B Z).
    pf_kPa : float, optional
        Design foundation pressure pf (kPa), given directly; takes
        precedence over ``load_kN``.
    embedment_DB : float, optional
        The embedment whose 2/3 is D' (m), see above. Default 0.0
        (If = 1.0).
    silty_sand : bool, optional
        True: Eq. 7-51 (coefficient 8) for silty sand. Default False
        (Eq. 7-50).

    Returns
    -------
    dict
        ``settlement_mm``, ``settlement_in``, ``pf_kPa``, ``pf_ksf``,
        ``influence_factor``, ``B_m``, ``N160``, ``coefficient`` (4 or 8),
        ``equation`` ('7-50' or '7-51').

    References
    ----------
    Meyerhof, G.G. (1976), J. Geotech. Eng. Div., ASCE, 102(GT3), 197-228.
    FHWA GEC-12 (FHWA-NHI-16-009) Vol. I Sec. 7.3.5.2.1; Vol. III
    (FHWA-NHI-16-064) Block 15, Table D-23. UFC 3-220-20 Sec. 6-5.8.2.
    """
    # --- Input validation ---
    if group_width <= 0:
        raise ValueError(f"group_width must be positive, got {group_width}")
    if group_length <= 0:
        raise ValueError(f"group_length must be positive, got {group_length}")
    if N160 <= 0:
        raise ValueError(f"N160 must be positive, got {N160}")
    if embedment_DB < 0:
        raise ValueError(f"embedment_DB must be >= 0, got {embedment_DB}")

    # B is the smaller plan dimension by definition; Z the larger.
    B_m = min(group_width, group_length)
    Z_m = max(group_width, group_length)

    # --- Design foundation pressure pf (kPa) ---
    if pf_kPa is not None:
        if pf_kPa <= 0:
            raise ValueError(f"pf_kPa must be positive, got {pf_kPa}")
        pf = float(pf_kPa)
    elif load_kN is not None:
        if load_kN <= 0:
            raise ValueError(f"load_kN must be positive, got {load_kN}")
        pf = load_kN / (B_m * Z_m)
    else:
        raise ValueError(
            "Provide either load_kN (total permanent load, kN) or "
            "pf_kPa (design foundation pressure, kPa)."
        )

    # --- Embedment influence factor (Eq. 7-52, not below 0.5) ---
    If = 1.0 - (2.0 / 3.0 * embedment_DB) / (8.0 * B_m)
    If = max(If, 0.5)

    # --- SI -> US, apply the published US-customary formula, US -> SI ---
    coefficient = 8.0 if silty_sand else 4.0
    pf_ksf = pf * _KSF_PER_KPA
    B_ft = B_m * _FT_PER_M
    S_in = coefficient * pf_ksf * If * math.sqrt(B_ft) / N160
    S_mm = S_in * _MM_PER_IN

    return {
        "settlement_mm": round(S_mm, 3),
        "settlement_in": round(S_in, 4),
        "pf_kPa": round(pf, 3),
        "pf_ksf": round(pf_ksf, 4),
        "influence_factor": round(If, 4),
        "B_m": round(B_m, 4),
        "N160": N160,
        "coefficient": int(coefficient),
        "equation": "7-51" if silty_sand else "7-50",
    }
