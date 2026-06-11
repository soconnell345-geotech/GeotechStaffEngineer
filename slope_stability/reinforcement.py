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
