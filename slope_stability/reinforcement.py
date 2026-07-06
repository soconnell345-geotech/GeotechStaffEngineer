"""
Reinforcement integration for limit equilibrium slope stability.

Wires soil nails (nails.py), simple anchors, and geosynthetic layers into
the FOS equations of all methods. Each element crossing the slip surface
contributes an allowable tension T (kN per metre of slope run) applied at
the crossing point, directed along the element toward its anchorage
(into the stable zone).

Convention: ACTIVE (Method A): T is applied as an unfactored external
force in the equilibrium equations — it reduces the driving moment/force
rather than being added to the (strength-factored) resisting side. This
matches the FHWA GEC-7 recommendation for allowable nail forces and
Slide2's "Active" reinforcement convention.

Nail capacity per GEC-7: T = min(pullout, tensile) where
pullout = bond_stress * pi * D_drillhole * L_behind (length beyond the
slip surface) and tensile = fy * A_bar, both divided by the horizontal
spacing to give per-metre-run values.

References
----------
Lazarte et al. (2003) — FHWA GEC-7: Soil Nail Walls (also in-house
    geotech-references/gec_7).
Byrne et al. (1998) — FHWA-SA-96-069R Soil Nailing Manual.
Rocscience Slide2 documentation — reinforcement conventions.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from slope_stability.nails import SoilNail


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
    """
    elevation: float
    T_allow: float
    x_start: Optional[float] = None
    x_end: Optional[float] = None

    def __post_init__(self):
        if self.T_allow <= 0:
            raise ValueError(f"T_allow must be positive, got {self.T_allow}")


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
    """
    x_head: float
    z_head: float
    length: float
    T_allow: float
    inclination: float = 15.0

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.T_allow <= 0:
            raise ValueError(f"T_allow must be positive, got {self.T_allow}")


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
        'horizontal' (default) or 'normal' (perpendicular to the slip surface
        at the crossing).
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
        if self.force_direction not in ("horizontal", "normal"):
            raise ValueError("force_direction must be 'horizontal' or 'normal'")


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
        'nail', 'anchor', or 'geosynthetic'.
    index : int
        Index within its input list.
    controlled_by : str
        'pullout', 'tensile', or 'allowable'.
    """
    x: float
    z: float
    T: float
    dir_x: float
    dir_z: float
    kind: str
    index: int
    controlled_by: str = "allowable"


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


def ito_matsui_pressure(c: float, phi: float, gamma: float, z: float,
                        D1: float, D2: float) -> float:
    """Ito & Matsui (1975) lateral force per unit depth on one pile at depth z.

    Plastic-deformation theory for a row of piles in c-phi soil. ``z`` is the
    depth below the top of the moving layer, D1 the center-to-center pile
    spacing and D2 the clear spacing (D1 - pile diameter). The pressure is

        p(z) = P_c  +  (gamma*z / N_phi) * (D1*A - D2)

    with N_phi = tan^2(45+phi/2), and (writing s=sqrt(N_phi), t=tan(phi),
    m = s*t + N_phi - 1, and Cterm = (2 t + 2 s + 1/s)/m):

        A   = (D1/D2)^m * exp[ (D1-D2)/D2 * N_phi * t * tan(pi/8 + phi/2) ]
        P_c = c * { D1 * [ (A - 2 s t - 1)/(s t) + Cterm ] - D2 * Cterm }

    For phi = 0 the general form is singular; Ito & Matsui give the cohesive
    limit  p(z) = c*D1*[ 3 ln(D1/D2) + (D1-D2)/D2 ] + gamma*z*(D1-D2)  (used
    below when phi < 1e-6).

    References: Ito, T. & Matsui, T. (1975), "Methods to estimate lateral force
    acting on stabilizing piles," Soils and Foundations 15(4), 43-59; equation
    as reproduced in Hassiotis, Chameau & Gunaratne (1997), "Design method for
    stabilizing piles in slopes," J. Geotech. Geoenviron. Eng. 123(4)
    (exponential argument tan(pi/8 + phi/2)). Also the basis of the Rocscience
    Slide2 Ito & Matsui pile support (verification #106). Validated here by the
    #106 spacing trend (force falls as spacing/diameter grows), not an exact FOS.
    """
    if D2 <= 0 or D2 >= D1:
        raise ValueError("need 0 < D2 < D1")
    if phi < 1e-6:
        return (c * D1 * (3.0 * math.log(D1 / D2) + (D1 - D2) / D2)
                + gamma * z * (D1 - D2))
    phir = math.radians(phi)
    Nphi = math.tan(math.radians(45.0 + phi / 2.0)) ** 2
    s = math.sqrt(Nphi)
    t = math.tan(phir)
    m = s * t + Nphi - 1.0
    A = (D1 / D2) ** m * math.exp((D1 - D2) / D2 * Nphi * t
                                  * math.tan(math.pi / 8.0 + phir / 2.0))
    Cterm = (2.0 * t + 2.0 * s + 1.0 / s) / m
    P_c = c * (D1 * ((A - 2.0 * s * t - 1.0) / (s * t) + Cterm) - D2 * Cterm)
    return P_c + (gamma * z / Nphi) * (D1 * A - D2)


def ito_matsui_lateral_force(c: float, phi: float, gamma: float,
                             D1: float, D2: float,
                             z_top: float, z_bot: float) -> float:
    """Total Ito & Matsui (1975) lateral force on ONE pile (kN).

    Integrates ``ito_matsui_pressure`` over the moving layer, from the pile head
    (``z_top``) down to the slip surface (``z_bot``), with depth measured below
    the head. Because p(z) is linear in the overburden gamma*z, the integral over
    a layer of thickness H = z_top - z_bot is closed-form:

        F = P_c * H + (gamma / N_phi) * (D1*A - D2) * H^2 / 2      (phi > 0)
        F = c*D1*[3 ln(D1/D2)+(D1-D2)/D2]*H + gamma*(D1-D2)*H^2/2  (phi = 0)

    Divide by the spacing D1 for the force per metre of slope run.
    """
    H = z_top - z_bot
    if H <= 0:
        return 0.0
    if phi < 1e-6:
        pc = c * D1 * (3.0 * math.log(D1 / D2) + (D1 - D2) / D2)
        return pc * H + gamma * (D1 - D2) * H * H / 2.0
    phir = math.radians(phi)
    Nphi = math.tan(math.radians(45.0 + phi / 2.0)) ** 2
    s = math.sqrt(Nphi)
    t = math.tan(phir)
    m = s * t + Nphi - 1.0
    A = (D1 / D2) ** m * math.exp((D1 - D2) / D2 * Nphi * t
                                  * math.tan(math.pi / 8.0 + phir / 2.0))
    Cterm = (2.0 * t + 2.0 * s + 1.0 / s) / m
    P_c = c * (D1 * ((A - 2.0 * s * t - 1.0) / (s * t) + Cterm) - D2 * Cterm)
    return P_c * H + (gamma / Nphi) * (D1 * A - D2) * H * H / 2.0


def compute_reinforcement_forces(geom, slip,
                                 x_entry: float,
                                 x_exit: float) -> List[ReinforcementForce]:
    """Resolve all reinforcement crossings for a trial slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        May carry ``nails``, ``anchors``, ``geosynthetics`` lists and the
        legacy ``reinforcement_force``/``reinforcement_elevation`` pair
        (treated as a horizontal allowable force).
    slip : slip surface (circular or polyline)
    x_entry, x_exit : float
        Slip surface entry/exit (limits the search window).

    Returns
    -------
    list of ReinforcementForce
    """
    forces: List[ReinforcementForce] = []

    nails = getattr(geom, "nails", None) or []
    for i, nail in enumerate(nails):
        beta = math.radians(nail.inclination)
        hit = _line_slip_intersection(slip, geom, nail.x_head, nail.z_head,
                                      beta, nail.length)
        if hit is None:
            continue
        x_int, z_int, t_int, dx, dz = hit
        L_behind = max(nail.length - t_int, 0.0)
        ddh_m = nail.drill_hole_diameter / 1000.0
        T_pull = nail.bond_stress * math.pi * ddh_m * L_behind / nail.spacing_h
        T_tens = nail.tensile_capacity_kN / nail.spacing_h
        if T_pull <= 0:
            continue
        if T_pull <= T_tens:
            T, ctrl = T_pull, "pullout"
        else:
            T, ctrl = T_tens, "tensile"
        forces.append(ReinforcementForce(
            x=x_int, z=z_int, T=T, dir_x=dx, dir_z=dz,
            kind="nail", index=i, controlled_by=ctrl))

    anchors = getattr(geom, "anchors", None) or []
    for i, a in enumerate(anchors):
        beta = math.radians(a.inclination)
        hit = _line_slip_intersection(slip, geom, a.x_head, a.z_head,
                                      beta, a.length)
        if hit is None:
            continue
        x_int, z_int, t_int, dx, dz = hit
        if a.length - t_int <= 0.05 * a.length:
            continue  # bond zone not beyond the surface
        forces.append(ReinforcementForce(
            x=x_int, z=z_int, T=a.T_allow, dir_x=dx, dir_z=dz,
            kind="anchor", index=i, controlled_by="allowable"))

    geos = list(getattr(geom, "geosynthetics", None) or [])
    # Legacy single horizontal reinforcement force on SlopeGeometry
    legacy_F = getattr(geom, "reinforcement_force", 0.0) or 0.0
    legacy_z = getattr(geom, "reinforcement_elevation", None)
    if legacy_F > 0 and legacy_z is not None:
        geos.append(Geosynthetic(elevation=legacy_z, T_allow=legacy_F))

    for i, g in enumerate(geos):
        x_lo = max(x_entry, g.x_start if g.x_start is not None else x_entry)
        x_hi = min(x_exit, g.x_end if g.x_end is not None else x_exit)
        if x_hi <= x_lo:
            continue
        # find crossing of slip surface with the layer elevation
        n = 200
        x_prev = x_lo
        zs_prev = slip.slip_elevation_at(x_prev)
        f_prev = None if zs_prev is None else g.elevation - zs_prev
        crossing = None
        for k in range(1, n + 1):
            x = x_lo + (x_hi - x_lo) * k / n
            zs = slip.slip_elevation_at(x)
            f = None if zs is None else g.elevation - zs
            if f_prev is not None and f is not None and f_prev * f < 0:
                a_, b_ = x_prev, x
                for _ in range(60):
                    m = 0.5 * (a_ + b_)
                    zm = slip.slip_elevation_at(m)
                    fm = None if zm is None else g.elevation - zm
                    if fm is None:
                        break
                    if fm * f_prev > 0:
                        a_ = m
                    else:
                        b_ = m
                    if b_ - a_ < 1e-8:
                        break
                crossing = 0.5 * (a_ + b_)
                break
            x_prev, f_prev = x, f
        if crossing is None:
            continue
        # direction: horizontal, toward the anchored (stable) side.
        # The anchored side is where the layer is BELOW the slip surface
        # (inside stable ground): determine from the local slope of the
        # slip surface — anchorage lies on the deeper side.
        zs_left = slip.slip_elevation_at(max(crossing - 0.5, x_lo))
        zs_right = slip.slip_elevation_at(min(crossing + 0.5, x_hi))
        if zs_left is not None and zs_right is not None:
            dir_x = 1.0 if zs_right < zs_left else -1.0
        else:
            dir_x = 1.0
        forces.append(ReinforcementForce(
            x=crossing, z=g.elevation, T=g.T_allow, dir_x=dir_x, dir_z=0.0,
            kind="geosynthetic", index=i, controlled_by="allowable"))

    piles = getattr(geom, "stabilizing_piles", None) or []
    for i, p in enumerate(piles):
        if p.x < x_entry or p.x > x_exit:
            continue
        z_slip = slip.slip_elevation_at(p.x)
        if z_slip is None:
            continue
        z_head = p.z_head if p.z_head is not None else geom.ground_elevation_at(p.x)
        if z_slip >= z_head:
            continue                       # slip surface above the pile head
        if p.z_toe is not None and z_slip <= p.z_toe:
            continue                       # slip surface below the pile toe
        if p.ito_matsui:
            lay = geom.layer_at_point(p.x, 0.5 * (z_head + z_slip))
            c = p.c if p.c is not None else (lay.c_prime if lay else 0.0)
            phi = p.phi if p.phi is not None else (lay.phi if lay else 0.0)
            gamma = p.gamma if p.gamma is not None else (lay.gamma if lay else 0.0)
            F_pile = ito_matsui_lateral_force(
                c=c, phi=phi, gamma=gamma, D1=p.spacing, D2=p.spacing - p.diameter,
                z_top=z_head, z_bot=z_slip)
            ctrl = "ito_matsui"
        else:
            F_pile = p.shear_capacity
            ctrl = "shear_capacity"
        T = F_pile / p.spacing             # per metre of slope run
        if T <= 0:
            continue
        if p.force_direction == "normal":
            # perpendicular to the slip surface (resisting), pointing into the
            # stable mass; approximate the local slip tangent by finite diff
            zl = slip.slip_elevation_at(p.x - 0.25)
            zr = slip.slip_elevation_at(p.x + 0.25)
            if zl is not None and zr is not None:
                tx, tz = 0.5, (zr - zl)
                norm = math.hypot(tx, tz)
                dir_x, dir_z = -tz / norm, tx / norm    # rotate tangent +90
                if dir_z < 0:                            # point upward (resisting)
                    dir_x, dir_z = -dir_x, -dir_z
            else:
                dir_x, dir_z = _into_slope_sign(geom, p.x), 0.0
        else:
            dir_x, dir_z = _into_slope_sign(geom, p.x), 0.0
        forces.append(ReinforcementForce(
            x=p.x, z=z_slip, T=T, dir_x=dir_x, dir_z=dir_z,
            kind="pile", index=i, controlled_by=ctrl))

    return forces


# ---------------------------------------------------------------------------
# Contributions for the classical methods (active convention)
# ---------------------------------------------------------------------------

def moment_reduction(forces: List[ReinforcementForce],
                     xc: float, yc: float, radius: float) -> float:
    """Reduction of the (normalized) driving moment for circular methods.

    Returns sum(T * d_perp) / R — subtract from the driving sum of
    Fellenius/Bishop (which is the driving moment divided by R).
    The perpendicular arm is taken as a magnitude: reinforcement crossing
    the surface is assumed stabilizing (standard practice).
    """
    total = 0.0
    for f in forces:
        d_perp = abs((f.x - xc) * f.dir_z - (f.z - yc) * f.dir_x)
        total += f.T * d_perp / radius
    return total


def horizontal_reduction(forces: List[ReinforcementForce]) -> float:
    """Reduction of the driving force for force-equilibrium methods.

    Magnitude of the horizontal components (assumed stabilizing).
    """
    return sum(f.T * abs(f.dir_x) for f in forces)
