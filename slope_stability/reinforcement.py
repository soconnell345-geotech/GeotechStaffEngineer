"""
Reinforcement integration for limit equilibrium slope stability.

Wires soil nails (nails.py), simple anchors, and geosynthetic layers into
the FOS equations of all methods. Each element crossing the slip surface
contributes an allowable tension T (kN per metre of slope run) applied at
the crossing point, directed along the element toward its anchorage
(into the stable zone).

Where the force acts (Wright & Duncan 1991). Each support is a
concentrated force applied at the base of the slice whose base it crosses
(Slide2: "a force is applied at the point of intersection of the slip
surface with the support"). It enters every equilibrium equation the method
satisfies:

* Bishop's simplified method: the moment about the circle centre, and the
  VERTICAL component in that slice's vertical force equilibrium, so it
  changes the base normal force and, where phi > 0, the shear strength.
  The horizontal component enters no equation Bishop satisfies except the
  moment. Wright & Duncan (1991), conclusion 6: "In Bishop's simplified
  procedure, which satisfies vertical force equilibrium, the vertical
  components of the reinforcement forces should be included in the
  equations of vertical equilibrium."
* Ordinary Method of Slices (Fellenius): no number where a support
  crosses the surface (GM-140 step 8). Duncan, Wright and Brandon (2014,
  p. 93) print Bishop's simplified method with known forces (Eq. 6.78) and
  no such equation for this method; no other reference held prints one.
* Spencer / Morgenstern-Price / GLE / Janbu (``gle.py``): the horizontal
  component in that slice's horizontal equilibrium, the vertical component
  in its vertical equilibrium, and the moment exactly.

GM-140 step 8 (``slope_stability.supports``): each crossing is found
exactly, the force acts along the support's own orientation toward its
anchorage (or, as a stated option, parallel to the slip surface; Duncan,
Wright and Brandon 2014, Sec. 8.6), it is no larger than the support can
carry at the crossing (a layer's pullout from each end, Sec. 8.4; a nail's
facing term, GEC 7 Eq. 5.2), a support whose pull assists the sliding is
slack by default (OWN-59), and a crossing on a slice boundary acts on the
slice holding the part of the support inside the sliding mass (GM-353).

In the circular methods the moment is the signed moment of the force about
the centre (Eq. 6.70; M_n of Eq. 6.78): an inclined support whose line of
action passes on the far side of the centre adds to the driving moment, as
it does in the GLE. On a circle that moment is the support's component along the slip
surface times R, so a support crossing a steep part of the circle close to
the level of the centre counts for little in Bishop's method (the warning
in ``search_supplements._support_arm_warning``). Until 2026-10-14 the
circular methods took the moment's magnitude, so such a support counted as
stabilizing, and left the vertical component out of the base normal
(GM-59).

Active or passive (Duncan & Wright 2005, Ch. 8, Methods A and B; Slide2
"Active/Passive Force Application"), per element (``support_convention``):

* 'active' (DEFAULT, Method A): F = soil resisting / (driving - support).
  The support force is NOT divided by the factor of safety, so the input is
  the ALLOWABLE force. Duncan & Wright prefer it: "the soil strength and the
  reinforcement forces have different sources of uncertainty, factoring them
  separately makes it possible to reflect these differences" (as quoted by
  Slide2). Tiebacks and pretensioned anchors act this way.
* 'passive' (Method B): F = (soil resisting + support) / driving. "Both soil
  strength and support forces are divided by the factor of safety", so the
  input is the ULTIMATE force. Here the support force acts as T / F in every
  equilibrium equation above (its moment, its vertical and its horizontal
  component). Slide2: passive "will always give a lower Factor of Safety
  than Active support" for the same force.

Nail capacity per GEC-7: T = min(pullout, tensile) where
pullout = bond_stress * pi * D_drillhole * L_behind (length beyond the
slip surface) and tensile = fy * A_bar, both divided by the horizontal
spacing to give per-metre-run values.

References
----------
Wright, S.G. & Duncan, J.M. (1991). "Limit equilibrium stability analyses
    for reinforced slopes." Transportation Research Record 1330, 40-46.
Duncan, J.M. & Wright, S.G. (2005); Duncan, Wright & Brandon (2014). Soil
    Strength and Slope Stability, Ch. 8 (Methods A and B).
Rocscience Slide2 help: "Active/Passive Force Application" and "Overview of
    Support Implementation".
Lazarte et al. (2003) — FHWA GEC-7: Soil Nail Walls (also in-house
    geotech-references/gec_7).
Byrne et al. (1998) — FHWA-SA-96-069R Soil Nailing Manual.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from slope_stability.nails import SoilNail

SUPPORT_CONVENTIONS = ("active", "passive")


def _check_direction(value: str) -> None:
    from slope_stability.supports import FORCE_DIRECTIONS
    if value not in FORCE_DIRECTIONS:
        raise ValueError(
            "force_direction must be 'along_reinforcement' (the support's "
            "own orientation) or 'along_slip_surface' (Duncan, Wright and "
            f"Brandon 2014, Sec. 8.6), got {value!r}")


def _check_convention(value: str) -> None:
    if value not in SUPPORT_CONVENTIONS:
        raise ValueError("support_convention must be 'active' or 'passive', "
                         f"got '{value}'")


@dataclass
class Geosynthetic:
    """A horizontal geosynthetic reinforcement layer.

    Parameters
    ----------
    elevation : float
        Elevation of the layer (m).
    T_allow : float
        Allowable tensile capacity per metre of slope run (kN/m).
        The user is responsible for reducing for pullout/durability —
        the full T_allow is applied when the layer crosses the slip
        surface (standard simple treatment).
    x_start, x_end : float, optional
        Horizontal extent. Defaults to unbounded.
    support_convention : str
        'active' (default, Method A: the force reduces the driving side,
        give the allowable force) or 'passive' (Method B: the force is
        divided by the factor of safety, give the ultimate force).
    F_star, alpha : float, optional
        The pullout resistance factor and the scale effect correction
        factor (GEC 11 Eq. 3-2 and Table 3-6, pp. 3-16 to 3-17; Duncan,
        Wright and Brandon 2014, Table 8.3). Findings about the fill and
        the product: no default. Given, the force at a crossing is the
        lesser of ``T_allow`` and the pullout resistance
        (``slope_stability.supports``).
    pullout_factor : float, optional
        What the pullout resistance is divided by under Method A (Duncan,
        Wright and Brandon 2014, Table 8.4: F_R = 1.3, 1.5 or 2.0; GEC 11
        p. 9-5: 1.5 granular, 2 cohesive). Asked for when F_star and alpha
        are given and the layer is active; not used when passive.
    pullout_basis : str, optional
        'dwb2014' (pullout from each end of the layer; the default with
        F_star and alpha), 'gec11' (behind the surface only, on the
        effective stress, with ``coverage_ratio``, at least 1 m),
        'given' (``development_length``) or 'not_checked' (the full
        ``T_allow`` at every crossing; the case when no pullout input is
        given, said in the result).
    development_length : float, optional
        Length from an end of the layer over which its force rises
        linearly from zero to ``T_allow`` (m): the pullout given directly
        (Duncan, Wright and Brandon 2014, Fig. 8.2).
    anchored_end : str, optional
        'start' (the -x end) or 'end' (the +x end): an end fixed to a
        facing, from which no pullout is measured.
    coverage_ratio : float
        Rc = b / Sh (GEC 11 Eq. 9-9); 1 for a continuous sheet. Used by
        'gec11' only.
    force_direction : str
        'along_reinforcement' (default: horizontal, the layer's own
        orientation) or 'along_slip_surface' (Duncan, Wright and Brandon
        2014, Sec. 8.6, pp. 163 to 164).
    """
    elevation: float
    T_allow: float
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    support_convention: str = "active"
    F_star: Optional[float] = None
    alpha: Optional[float] = None
    pullout_factor: Optional[float] = None
    pullout_basis: Optional[str] = None
    development_length: Optional[float] = None
    anchored_end: Optional[str] = None
    coverage_ratio: float = 1.0
    force_direction: str = "along_reinforcement"

    def __post_init__(self):
        if self.T_allow <= 0:
            raise ValueError(f"T_allow must be positive, got {self.T_allow}")
        _check_convention(self.support_convention)
        _check_direction(self.force_direction)
        from slope_stability.supports import PULLOUT_BASES
        if (self.pullout_basis is not None
                and self.pullout_basis not in PULLOUT_BASES):
            raise ValueError(
                "pullout_basis must be one of "
                + ", ".join(repr(b) for b in PULLOUT_BASES)
                + f", got {self.pullout_basis!r}")
        for name in ("F_star", "alpha", "pullout_factor",
                     "development_length"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if (self.F_star is None) != (self.alpha is None):
            raise ValueError("Give both F_star and alpha, or neither")
        if self.pullout_basis == "given" and not self.development_length:
            raise ValueError(
                "pullout_basis='given' needs development_length")
        if (self.pullout_basis in ("dwb2014", "gec11")
                and self.F_star is None):
            raise ValueError(
                f"pullout_basis={self.pullout_basis!r} needs F_star and "
                f"alpha")
        if self.anchored_end not in (None, "start", "end"):
            raise ValueError("anchored_end must be 'start' or 'end', got "
                             f"{self.anchored_end!r}")
        if not 0 < self.coverage_ratio <= 1:
            raise ValueError("coverage_ratio must be above 0 and at most "
                             f"1, got {self.coverage_ratio}")


@dataclass
class Anchor:
    """A simple tieback/anchor with user-specified allowable tension.

    Parameters
    ----------
    x_head, z_head : float
        Head position on the slope face (m).
    length : float
        Total anchor length (m).
    inclination : float
        Degrees below horizontal (positive = downward). Default 15.
    T_allow : float
        Allowable tension per metre of slope run (kN/m). The anchor
        contributes full T_allow when it crosses the slip surface with
        its bond zone beyond it (no pullout model — use SoilNail for
        bond-controlled elements).
    support_convention : str
        'active' (default, Method A: the force reduces the driving side,
        give the allowable force; a tensioned tieback acts on the mass
        before it moves) or 'passive' (Method B: the force is divided by
        the factor of safety, give the ultimate force).
    force_direction : str
        'along_reinforcement' (default: along the anchor) or
        'along_slip_surface' (Duncan, Wright and Brandon 2014, Sec. 8.6,
        pp. 163 to 164).
    """
    x_head: float
    z_head: float
    length: float
    T_allow: float
    inclination: float = 15.0
    support_convention: str = "active"
    force_direction: str = "along_reinforcement"

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.T_allow <= 0:
            raise ValueError(f"T_allow must be positive, got {self.T_allow}")
        _check_convention(self.support_convention)
        _check_direction(self.force_direction)


@dataclass
class StabilizingPile:
    """A single row of vertical stabilizing / micro piles crossing the slope.

    The row provides a lateral resisting force where the piles cross the slip
    surface. Two ways to set the resistance:

    * ``shear_capacity`` — the limiting shear resistance of ONE pile (kN). The
      per-metre force applied to the sliding mass is ``shear_capacity/spacing``
      (Slide2 verification #54 / Yamagami 2000 style: a directly-specified
      passive pile shear).
    * Ito & Matsui (1975) plastic-deformation lateral force — set
      ``ito_matsui=True`` and give the pile ``diameter`` (so the clear spacing
      D2 = spacing - diameter). The lateral force per pile is the Ito-Matsui
      pressure integrated from the pile head to the slip surface; divided by
      ``spacing`` for the per-metre force. Soil c/phi/gamma default to the layer
      the pile passes through.

    Parameters
    ----------
    x : float
        Pile-row location (m). The piles are vertical.
    shear_capacity : float, optional
        Limiting shear per pile (kN). Mutually exclusive with ``ito_matsui``.
    spacing : float
        Center-to-center pile spacing along the row, D1 (m). Default 1.0.
    z_head : float, optional
        Pile head elevation (m). Default: ground surface at ``x``.
    z_toe : float, optional
        Pile toe elevation (m). Default: unbounded below (crosses wherever the
        slip surface is below the head).
    ito_matsui : bool
        Use the Ito & Matsui (1975) plastic-deformation lateral force.
    diameter : float, optional
        Pile diameter/width B (m); required for ``ito_matsui`` (D2 = spacing-B).
    c, phi, gamma : float, optional
        Soil strength/unit weight for the Ito-Matsui force. Default: taken from
        the soil layer at the pile mid-depth.
    force_direction : str
        'horizontal' (default: the pile's shear, perpendicular to the
        vertical pile, against the sliding) or 'along_slip_surface'
        (parallel to the slip surface at the crossing, against the
        sliding): the two directions Duncan, Wright and Brandon (2014,
        Sec. 8.6, pp. 163 to 164) print for a support's force. 'normal'
        (perpendicular to the slip surface), offered until GM-140 step 8,
        is refused: no reference held prints it, and on a circle such a
        force has no moment about the centre.
    support_convention : str
        'active' (**default**, Duncan & Wright Method A, Slide2 "Active")
        applies the unfactored force, reducing the driving side; 'passive'
        (Method B, Slide2 "Passive") divides it by the factor of safety, on
        the resisting side. Active gives a larger FOS gain for the same
        force; passive is the more conservative treatment matching a pile
        mobilised only as the mass moves. Every method honours it (since
        GM-59; the rigorous GLE used to apply the pile force unfactored
        either way).
    """
    x: float
    shear_capacity: Optional[float] = None
    spacing: float = 1.0
    z_head: Optional[float] = None
    z_toe: Optional[float] = None
    ito_matsui: bool = False
    diameter: Optional[float] = None
    c: Optional[float] = None
    phi: Optional[float] = None
    gamma: Optional[float] = None
    force_direction: str = "horizontal"
    support_convention: str = "active"

    def __post_init__(self):
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")
        if self.ito_matsui:
            if self.diameter is None or self.diameter <= 0:
                raise ValueError("ito_matsui piles need a positive diameter")
            if self.diameter >= self.spacing:
                raise ValueError("diameter must be < spacing (need D2 = spacing "
                                 "- diameter > 0)")
        elif self.shear_capacity is None or self.shear_capacity <= 0:
            raise ValueError("give a positive shear_capacity or set "
                             "ito_matsui=True with a diameter")
        if self.force_direction == "normal":
            raise ValueError(
                "force_direction='normal' (perpendicular to the slip "
                "surface) is no longer offered: no reference held prints "
                "it, and on a circle such a force has no moment about the "
                "centre. Use 'horizontal' (the pile's shear, the default) "
                "or 'along_slip_surface' (Duncan, Wright and Brandon 2014, "
                "Sec. 8.6, pp. 163 to 164).")
        if self.force_direction not in ("horizontal", "along_slip_surface"):
            raise ValueError("force_direction must be 'horizontal' or "
                             "'along_slip_surface', got "
                             f"{self.force_direction!r}")
        if self.support_convention not in ("active", "passive"):
            raise ValueError("support_convention must be 'active' or 'passive', "
                             f"got '{self.support_convention}'")


@dataclass
class ReinforcementForce:
    """A resolved reinforcement force at a slip-surface crossing.

    Attributes
    ----------
    x, z : float
        Crossing point on the slip surface.
    T : float
        Force magnitude (kN per metre of slope run).
    dir_x, dir_z : float
        Unit direction of the force ON the sliding mass (toward the
        anchorage / stable zone).
    kind : str
        'nail', 'anchor', 'geosynthetic' or 'pile'.
    index : int
        Index within its input list.
    controlled_by : str
        'pullout', 'pullout_front', 'embedment', 'tensile', 'facing',
        'allowable', 'shear_capacity' or 'ito_matsui'.
    side : float
        -1 or +1: the x sense, from the crossing, of the part of the
        support inside the sliding mass (0 for a vertical pile). Decides
        the slice a crossing on a slice boundary acts on
        (``supports.slice_index_for``, GM-353).
    note : str
        What limited the force, in plain words.
    passive : bool
        If True, the force is applied with the PASSIVE convention (Method B:
        divided by the factor of safety, on the resisting side) rather than
        the default ACTIVE convention (Method A: unfactored, reducing the
        driving side). Default False (active).
    """
    x: float
    z: float
    T: float
    dir_x: float
    dir_z: float
    kind: str
    index: int
    controlled_by: str = "allowable"
    passive: bool = False
    side: float = 0.0
    note: str = ""


def _into_slope_sign(geom, x_head: float) -> float:
    """+1 if the slope rises toward +x at the head, else -1.

    The nail/anchor shank points into the slope (toward the crest /
    anchorage side); choosing the direction from the ground geometry
    prevents spurious 'crossings' out the slope face.
    """
    x_min, x_max = geom.x_range
    probe = max((x_max - x_min) / 50.0, 0.5)
    z_plus = geom.ground_elevation_at(min(x_head + probe, x_max))
    z_minus = geom.ground_elevation_at(max(x_head - probe, x_min))
    return 1.0 if z_plus >= z_minus else -1.0


def _line_slip_intersection(slip, geom, x_head, z_head, beta_rad, length,
                            n_sample: int = 200):
    """First crossing of an inclined line with the slip surface.

    The line runs from the head at ``beta_rad`` below horizontal,
    pointing into the slope (toward higher ground). The head must start
    above the slip surface; the line is followed until it passes below
    it.

    Returns (x_int, z_int, t, dir_x, dir_z) or None.
    """
    for sgn in (_into_slope_sign(geom, x_head),):
        dx = sgn * math.cos(beta_rad)
        dz = -math.sin(beta_rad)

        def f(t):
            x = x_head + t * dx
            z = z_head + t * dz
            zs = slip.slip_elevation_at(x)
            if zs is None:
                return None
            return z - zs  # positive above slip surface

        f0 = f(0.0)
        if f0 is None or f0 < 0:
            continue  # head not above the slip surface in this direction
        t_prev, f_prev = 0.0, f0
        found = None
        for i in range(1, n_sample + 1):
            t = length * i / n_sample
            ft = f(t)
            if ft is None:
                t_prev, f_prev = t, None
                continue
            if f_prev is not None and f_prev > 0 >= ft:
                a, b = t_prev, t
                for _ in range(60):
                    m = 0.5 * (a + b)
                    fm = f(m)
                    if fm is None:
                        break
                    if fm > 0:
                        a = m
                    else:
                        b = m
                    if b - a < 1e-8 * length:
                        break
                t_int = 0.5 * (a + b)
                found = (x_head + t_int * dx, z_head + t_int * dz, t_int,
                         dx, dz)
                break
            t_prev, f_prev = t, ft
        if found:
            return found
    return None


def _ito_matsui_coeffs(c: float, phi: float, gamma: float,
                       D1: float, D2: float):
    """Return (P_c, P_g) so the Ito-Matsui pressure is p(z) = P_c + P_g*z.

    Original Ito & Matsui (1975) plastic-deformation solution (their Eq. 13 for
    c-phi soil, Eq. 23 for phi=0). N_phi = tan^2(45+phi/2); s=sqrt(N_phi),
    t=tan(phi). With
        A  = (D1/D2)^(s*t + N_phi - 1) * exp[ (D1-D2)/D2 * N_phi*t*tan(pi/8+phi/4) ]
        Fc = (2t + 2s + 1/s) / (s*t + N_phi - 1)
    the pressure per unit depth on one pile is
        p(z) = c*D1*[ (A - 2 s t - 1)/(N_phi*t) + Fc ]
             - c*( D1*Fc - 2*D2/s )
             + (gamma*z/N_phi)*(D1*A - D2).
    NOTE the exponential uses tan(pi/8 + phi/4) and the first-term coefficient is
    1/(N_phi*tanphi): these are the ORIGINAL Ito & Matsui (1975) printed forms
    (Eq. 13, confirmed by their Fig. 2 wedge half-angle pi/8 + phi/4). A widely
    copied SECONDARY reproduction (Hassiotis, Chameau & Gunaratne 1997) prints
    tan(pi/8 + phi/2) and 1/(sqrt(N_phi)*tanphi); those variants overestimate the
    force (up to ~+150% for tight spacing / high phi) and are NOT used here. The
    D1*Fc contributions cancel, but the printed grouping is kept for traceability.
    The phi=0 cohesive limit is
        p(z) = c*{ D1*(3 ln(D1/D2) + (D1-D2)/D2*tan(pi/8)) - 2*(D1-D2) }
             + gamma*z*(D1-D2).
    """
    if D2 <= 0 or D2 >= D1:
        raise ValueError("need 0 < D2 < D1 (D2 = spacing - pile diameter)")
    if phi < 1e-6:
        P_c = c * (D1 * (3.0 * math.log(D1 / D2)
                         + (D1 - D2) / D2 * math.tan(math.pi / 8.0))
                   - 2.0 * (D1 - D2))
        return P_c, gamma * (D1 - D2)
    phir = math.radians(phi)
    Nphi = math.tan(math.radians(45.0 + phi / 2.0)) ** 2
    s = math.sqrt(Nphi)
    t = math.tan(phir)
    m = s * t + Nphi - 1.0
    A = (D1 / D2) ** m * math.exp((D1 - D2) / D2 * Nphi * t
                                  * math.tan(math.pi / 8.0 + phir / 4.0))
    Fc = (2.0 * t + 2.0 * s + 1.0 / s) / m
    P_c = (c * D1 * ((A - 2.0 * s * t - 1.0) / (Nphi * t) + Fc)
           - c * (D1 * Fc - 2.0 * D2 / s))
    P_g = (gamma / Nphi) * (D1 * A - D2)
    return P_c, P_g


def ito_matsui_pressure(c: float, phi: float, gamma: float, z: float,
                        D1: float, D2: float) -> float:
    """Ito & Matsui (1975) lateral force per unit depth on one pile at depth z.

    Plastic-deformation theory for a row of piles in c-phi soil. ``z`` is the
    depth below the top of the moving layer, D1 the center-to-center pile
    spacing and D2 the clear spacing (D1 - pile diameter). See
    ``_ito_matsui_coeffs`` for the equation.

    References: Ito, T. & Matsui, T. (1975), "Methods to estimate lateral force
    acting on stabilizing piles," Soils and Foundations 15(4), 43-59
    (DOI 10.3208/sandf1972.15.4_43) -- the ORIGINAL form, as used by Rocscience
    Slide2 (verification #106). Hand check: c=10, phi=20, gamma=18, z=5, D1=2.0,
    D2=1.5 -> p = 105.079 kN/m per m depth.
    """
    P_c, P_g = _ito_matsui_coeffs(c, phi, gamma, D1, D2)
    return P_c + P_g * z


def ito_matsui_lateral_force(c: float, phi: float, gamma: float,
                             D1: float, D2: float,
                             z_top: float, z_bot: float) -> float:
    """Total Ito & Matsui (1975) lateral force on ONE pile (kN).

    Integrates ``ito_matsui_pressure`` over the moving layer, from the pile head
    (``z_top``) down to the slip surface (``z_bot``), depth measured below the
    head. Because p(z) = P_c + P_g*z is linear in depth, the integral over a
    layer of thickness H = z_top - z_bot is closed-form:  F = P_c*H + P_g*H^2/2.
    Divide by the spacing D1 for the force per metre of slope run.
    """
    H = z_top - z_bot
    if H <= 0:
        return 0.0
    P_c, P_g = _ito_matsui_coeffs(c, phi, gamma, D1, D2)
    return P_c * H + P_g * H * H / 2.0


def compute_reinforcement_forces(geom, slip,
                                 x_entry: float,
                                 x_exit: float) -> List[ReinforcementForce]:
    """Resolve all reinforcement crossings for a trial slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        May carry ``nails``, ``anchors``, ``geosynthetics``,
        ``stabilizing_piles`` and the legacy
        ``reinforcement_force``/``reinforcement_elevation`` pair (a
        horizontal allowable force; refused without its elevation).
    slip : slip surface (circular or polyline)
    x_entry, x_exit : float
        The two ends of the slip surface.

    Returns
    -------
    list of ReinforcementForce
        The forces the methods take: ``supports.resolve_supports(...)
        .forces`` (GM-140 step 8: each crossing found exactly, the force
        along the stated direction and no larger than the support can
        carry there; a support whose pull assists the sliding is slack
        by default and is not in the list).
    """
    from slope_stability.supports import resolve_supports
    return resolve_supports(geom, slip, x_entry, x_exit).forces


# ---------------------------------------------------------------------------
# Contributions for the classical methods (Fellenius, Bishop)
# ---------------------------------------------------------------------------

def support_components(f: ReinforcementForce,
                       d_sign: float) -> Tuple[float, float]:
    """(Fx, Fz) of a support force on the sliding mass, Fz up-positive.

    The force acts along the direction the support was resolved with
    (``supports.resolve_supports``: toward its anchorage, or parallel to
    the slip surface against the sliding), as the rigorous GLE takes it
    (``gle._assign_reinforcement``). ``d_sign`` is kept for the callers
    and no longer sets a sign: until GM-140 step 8 the horizontal
    component was forced to point against the sliding whatever the
    direction (which hid a geosynthetic's wrong direction, GM-82)."""
    return (f.T * f.dir_x, f.T * f.dir_z)


def support_moment(f: ReinforcementForce, xc: float, yc: float,
                   d_sign: float) -> float:
    """Signed moment of a support force about (xc, yc), counter-clockwise
    positive (kN*m per m): (x - xc)*Fz - (z - yc)*Fx, with the components of
    ``support_components``.

    On a circle only the force's component along the slip surface where it
    crosses has a moment about the centre (the normal component passes
    through it), so this is that component times R: a support crossing a
    steep part of the circle, close to the level of the centre, has a short
    arm and little moment."""
    fx, fz = support_components(f, d_sign)
    return (f.x - xc) * fz - (f.z - yc) * fx


def _moment_arm_sum(forces, xc, yc, radius, *, passive: bool,
                    d_sign: Optional[float] = None) -> float:
    """Sum of the support moments / R over the forces matching ``passive``.

    With ``d_sign`` (+1 when the mass turns clockwise about the centre, i.e.
    the gravity moment sum of W*(x - xc) is positive; -1 otherwise) the
    moment is SIGNED, positive where it resists the sliding: an inclined
    support whose line of action passes on the far side of the centre adds
    to the driving moment, as in the rigorous GLE. Without it (``None``) the
    magnitude of the arm is taken, every support counted as resisting: the
    form the circular methods used until GM-59."""
    total = 0.0
    for f in forces:
        if bool(f.passive) is not passive:
            continue
        if d_sign is None:
            d_perp = abs((f.x - xc) * f.dir_z - (f.z - yc) * f.dir_x)
            total += f.T * d_perp / radius
        else:
            total += d_sign * support_moment(f, xc, yc, d_sign) / radius
    return total


def moment_reduction(forces: List[ReinforcementForce],
                     xc: float, yc: float, radius: float,
                     d_sign: Optional[float] = None) -> float:
    """Reduction of the (normalized) driving moment for circular methods.

    Returns the moment of the ACTIVE-convention forces about the centre / R
    — subtract from the driving sum of Fellenius/Bishop (which is the
    driving moment divided by R). Pass ``d_sign`` (the sliding sense, see
    ``_moment_arm_sum``) for the signed moment, as the methods do; without
    it every force counts as resisting. Passive-convention forces are
    handled by ``moment_resistance`` instead (added to the resisting side).
    """
    return _moment_arm_sum(forces, xc, yc, radius, passive=False,
                           d_sign=d_sign)


def moment_resistance(forces: List[ReinforcementForce],
                      xc: float, yc: float, radius: float,
                      d_sign: Optional[float] = None) -> float:
    """Contribution of the PASSIVE-convention forces to the resisting moment.

    Returns their moment about the centre / R — ADD to the resisting sum of
    Fellenius/Bishop (Method B: F = (resisting + support) / driving, so the
    support is divided by F with the soil strength). Zero when no force is
    passive.
    """
    return _moment_arm_sum(forces, xc, yc, radius, passive=True,
                           d_sign=d_sign)


def slice_index_at(slices, x: float, side: float = 0.0) -> Optional[int]:
    """Index of the slice whose base the crossing at ``x`` lies on, or
    None; on a slice boundary, by the rule of
    ``supports.slice_index_for`` (GM-353), as ``gle`` assigns it."""
    from slope_stability.supports import slice_index_for
    return slice_index_for([(s.x_left, s.x_right) for s in slices], x, side)


def slice_support_forces(forces: List[ReinforcementForce], slices, *,
                         passive: bool, d_sign: float) -> dict:
    """{slice index: (Fx, Fz)} — the support forces matching ``passive``
    summed per slice whose base they cross (Wright & Duncan 1991: each is
    a concentrated force on the base of that slice), as resolved by
    ``support_components``. Fz is up-positive."""
    out = {}
    for f in forces:
        if bool(f.passive) is not passive:
            continue
        i = slice_index_at(slices, f.x, getattr(f, "side", 0.0))
        if i is None:
            continue
        cx, cz = support_components(f, d_sign)
        fx, fz = out.get(i, (0.0, 0.0))
        out[i] = (fx + cx, fz + cz)
    return out


def _base_angle_at(slices, x: float) -> float:
    """Base inclination (rad) of the slice whose base the crossing x falls in.

    Used to resolve the vertical component of a reinforcement force onto the
    base tangent for the force-equilibrium (OMS-noncircular) driving reduction.
    Returns 0.0 when no slice list is available.
    """
    if not slices:
        return 0.0
    for s in slices:
        if s.x_left <= x <= s.x_right:
            return s.alpha
    # x outside the slice span: use the nearest end slice
    return slices[0].alpha if x < slices[0].x_left else slices[-1].alpha


def _horizontal_proj_sum(forces, slices, *, passive: bool) -> float:
    """Sum of the base-tangent projection of the forces matching ``passive``."""
    total = 0.0
    for f in forces:
        if bool(f.passive) is not passive:
            continue
        total += f.T * abs(f.dir_x)
        if slices is not None and abs(f.dir_z) > 1e-12:
            alpha = _base_angle_at(slices, f.x)
            total += f.T * abs(f.dir_z) * abs(math.sin(alpha))
    return total


def horizontal_reduction(forces: List[ReinforcementForce],
                         slices=None) -> float:
    """Reduction of the driving force for force-equilibrium (OMS-noncircular).

    The ACTIVE-convention reinforcement force is resolved into the base-driving
    direction and subtracted from the driving sum (assumed stabilizing):

    * the horizontal component contributes ``T*|dir_x|`` (unchanged — this is the
      only term for nails/anchors/geosynthetics/horizontal piles, whose
      ``dir_z == 0``, so their behaviour is byte-identical);
    * the vertical component contributes ``T*|dir_z|*|sin(alpha)|`` — its
      projection onto the base tangent at the crossing, where ``alpha`` is the
      base inclination there. A vertical force resists tangential sliding on an
      inclined base and does nothing on a flat one. ``slices`` supplies the base
      geometry; when omitted the vertical term is zero (legacy).

    Passive-convention forces are handled by ``horizontal_resistance`` instead.
    """
    return _horizontal_proj_sum(forces, slices, passive=False)


def horizontal_resistance(forces: List[ReinforcementForce],
                          slices=None) -> float:
    """Base-tangent projection of the PASSIVE-convention forces (add to
    resisting). Zero when no force is passive, so the active default is
    byte-identical."""
    return _horizontal_proj_sum(forces, slices, passive=True)
