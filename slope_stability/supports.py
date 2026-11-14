"""
Supports as printed (GM-140 step 8).

The rule. A support enters a solution as a known force, the way the print
gives it: by Method A or Method B, along the reinforcement's own direction,
and no larger than the reinforcement can carry where the slip surface
crosses it. `resolve_supports` is the one place that turns the supports of a
geometry into the forces every method takes (``reinforcement.
compute_reinforcement_forces`` returns its ``forces``).

Print (Duncan, Wright and Brandon 2014, "Soil Strength and Slope Stability",
2nd ed.; the owner's copy, cited by page and equation):

* Sec. 8.2, pp. 159 to 160. Method A: the forces are allowable forces and
  are not divided by the computed factor of safety,
  F = soil resisting moment / (overturning moment - reinforcement moment)
  (Eq. 8.1; Eq. 8.2 in stresses). Method B: the forces are ultimate
  (long-term capacity) forces and are divided by F as the soil strength
  is, F = (soil resisting moment + reinforcement moment) / overturning
  moment (Eq. 8.3; Eq. 8.4). Method A is preferable (p. 160).
* Sec. 8.6, pp. 163 to 164. The force acts along the reinforcement's
  original orientation (recommended, the more conservative); a force
  parallel to the slip surface is the other extreme.
* Sec. 8.4, pp. 161 to 163. The long-term capacity along a layer is the
  lesser of T_lim and the pullout resistance measured from each end
  (Fig. 8.2). Pullout from the end: T_po = 2 gamma z alpha F* L_e under a
  constant depth of fill z (Eq. 8.8) and T_po = gamma tan(beta) alpha F*
  L_e^2 beneath the slope face (Eq. 8.9). Both are the integral of
  2 alpha F* (gamma z) along the embedded length, which is what is
  evaluated here, on the ground as the geometry gives it. T_all = T_LTDS /
  F_R (Eq. 8.10), F_R = 1.3, 1.5 or 2.0 by consequence and uncertainty
  (Table 8.4): no single value is printed, so it is asked for.
* FHWA GEC 11 (FHWA-NHI-10-024/025): Pr = F* alpha sigma'v Le C (Eq. 3-2,
  p. 3-16; C = 2 for sheets, strips and grids); Le = Tmax FS / (F* alpha
  sigma'v Rc C) (Eq. 9-9, p. 9-13), FS = 1.5 for granular and 2 for
  cohesive soils (p. 9-5); "Only count reinforcement that extends 3 ft
  (1 m) beyond the surface" (p. 9-14); the capacity of a layer is "the
  minimum of its allowable pullout resistance behind the potential failure
  surface or its long-term allowable design strength" (Sec. 9.1, p. 9-1).
  The method of analysis of GEC 11's Sec. 9.2 (p. 9-24: the reinforcing
  force "contributed to the resisting moment") is Method B's form, and its
  Eq. 9-9 still divides the pullout by FS: under ``pullout_basis='gec11'``
  the pullout is divided by ``pullout_factor`` under Method A and Method B
  alike. In the form that reduces the driving moment (Method A's),
  FS = MR / (MD - Ts D) (Eq. 9-17, p. 9-24): "the
  stability factory of safety is not applied to the reinforcement. For
  such computations ... the allowable strength of the reinforcement Tal
  must be divided by a required minimum factor of safety FSR = 1.3 to
  provide an equivalent material uncertainty" (said with Method A; the
  allowable force is the engineer's input, not changed here). Continuous
  sheets are taken as acting tangentially to the circle (D = R, the
  definitions under Eq. 9-1, p. 9-7): the print of the
  ``force_direction='along_slip_surface'`` option.
* FHWA GEC 7 (FHWA-NHI-14-007) Eq. 5.2, p. 105: Tmax_i = min(R_T, R_F,
  R_PO): pullout behind the surface, the bar, and the facing. Fig. 5.7
  (p. 106; the library holds its caption, "Limitations to tensile forces
  in nails") is read as the facing resistance plus the pullout of the
  length in front of the surface limiting the load at the surface: a
  reading of the figure, the literal min(R_T, R_F, R_PO) of Eq. 5.2 being
  the other form.
* OWN-59 (the owner, 2026-10-20). A support whose pull assists the sliding:
  both readings are offered; slack (the support counts nothing) is the
  default.

What is the module's own (no page found; listed in STEP8_CHECK.md under
"Calls made"): the slice a support on a slice boundary acts on (the slice
that holds the part of the support inside the sliding mass); the test for
"assists the sliding" (the force's component along the slip surface at the
crossing points down the slope); a layer whose pullout inputs are not given
carries its full force and the result says the pullout was not checked.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from geotech_common.water import GAMMA_W

#: What a support whose pull assists the sliding counts (OWN-59).
ASSISTING_SUPPORT = ("slack", "as_given")

#: The direction a support's force acts in (Duncan, Wright and Brandon
#: 2014, Sec. 8.6, pp. 163 to 164).
FORCE_DIRECTIONS = ("along_reinforcement", "along_slip_surface")
#: A stabilizing pile's: its shear, perpendicular to the (vertical) pile,
#: or parallel to the slip surface.
PILE_FORCE_DIRECTIONS = ("horizontal", "along_slip_surface")

#: How a geosynthetic's pullout resistance is found.
PULLOUT_BASES = ("dwb2014", "gec11", "given", "not_checked")

#: GEC 11 p. 9-14: only reinforcement that extends 1 m (3 ft) beyond the
#: surface counts.
GEC11_MIN_EMBEDMENT_M = 1.0

#: A crossing this close (m) to a slice boundary is on the boundary.
BOUNDARY_TOL = 1e-6

_EPS = 1e-9


@dataclass
class SupportResolution:
    """The supports of a geometry on one slip surface.

    forces : the forces the methods take (``ReinforcementForce``).
    not_counted : supports that cross the surface and count nothing, each
        ``{"kind", "index", "x_m", "z_m", "T_kN_per_m", "reason"}``.
    notes : statements about what was and was not checked (plain words).
    """
    forces: list = field(default_factory=list)
    not_counted: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry of a crossing: exact (GM-140 step 8, item 9)
# ---------------------------------------------------------------------------

def _surface_segments(slip, x_entry: float, x_exit: float):
    """A polyline surface's segments between its two ends."""
    pts = slip.points
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
            if pts[i + 1][0] >= x_entry - _EPS and pts[i][0] <= x_exit + _EPS]


def slip_z(slip, x: float) -> Optional[float]:
    """The surface's elevation at x, the two ends included (a circle's
    ``slip_elevation_at`` is None at its own vertical ends only)."""
    return slip.slip_elevation_at(x)


def line_crossings(slip, x_entry: float, x_exit: float, x0: float,
                   z0: float, dx: float, dz: float,
                   length: float) -> List[Tuple[float, float, float]]:
    """Every point where the line from (x0, z0) along the unit direction
    (dx, dz) meets the slip surface between its two ends within ``length``:
    [(t, x, z)] in order of t, found exactly: the line with the circle
    (the lower arc), or with each segment of a polyline. Before GM-140
    step 8 the line was sampled at 200 points, which missed a crossing
    within one step of a circle's vertical end."""
    out = []
    tol = 1e-9 * max(1.0, length)
    if getattr(slip, "is_circular", False):
        ax, az = x0 - slip.xc, z0 - slip.yc
        b = ax * dx + az * dz
        c = ax * ax + az * az - slip.radius ** 2
        disc = b * b - c
        if disc < 0:
            return out
        root = math.sqrt(disc)
        for t in sorted({-b - root, -b + root}):
            if t < -tol or t > length + tol:
                continue
            x, z = x0 + t * dx, z0 + t * dz
            if z > slip.yc + tol:               # the upper arc
                continue
            if x < x_entry - 1e-7 or x > x_exit + 1e-7:
                continue
            out.append((max(t, 0.0), x, z))
        return out
    for (xa, za), (xb, zb) in _surface_segments(slip, x_entry, x_exit):
        ex, ez = xb - xa, zb - za
        den = dx * ez - dz * ex
        if abs(den) < 1e-14:                    # parallel
            continue
        t = ((xa - x0) * ez - (za - z0) * ex) / den
        s = ((xa - x0) * dz - (za - z0) * dx) / den
        if t < -tol or t > length + tol or s < -1e-12 or s > 1 + 1e-12:
            continue
        x, z = x0 + t * dx, z0 + t * dz
        if x < x_entry - 1e-7 or x > x_exit + 1e-7:
            continue
        if out and abs(out[-1][0] - t) <= tol:  # a vertex, met twice
            continue
        out.append((max(t, 0.0), x, z))
    out.sort()
    return out


def level_crossings(slip, x_entry: float, x_exit: float,
                    z: float) -> List[Tuple[float, float]]:
    """Every point where the level line at elevation z passes through the
    slip surface between its two ends: [(x, s)] in x order, found exactly.
    s = +1 where the surface rises through the level toward +x (so the
    level line is inside the sliding mass on the -x side of the crossing
    and in the ground below the surface on the +x side), -1 the reverse."""
    out = []
    if getattr(slip, "is_circular", False):
        h = slip.yc - z
        if h <= 0 or h >= slip.radius:
            return out
        half = math.sqrt(slip.radius ** 2 - h * h)
        for x, s in ((slip.xc - half, -1.0), (slip.xc + half, 1.0)):
            if x_entry - 1e-7 <= x <= x_exit + 1e-7:
                out.append((x, s))
        return out
    for (xa, za), (xb, zb) in _surface_segments(slip, x_entry, x_exit):
        if (za - z) * (zb - z) > 0 or za == zb:
            continue
        if zb == z and za != z:
            continue        # met at the next segment's start, once
        x = xa + (z - za) * (xb - xa) / (zb - za)
        if not (x_entry - 1e-7 <= x <= x_exit + 1e-7):
            continue
        if za == z:
            # starts on the level: a crossing only if the surface came
            # from the other side of it
            before = slip_z(slip, xa - 1e-6)
            if before is None or (before - z) * (zb - z) >= 0:
                continue
        out.append((x, 1.0 if zb > za else -1.0))
    return out


def base_slope(slip, x: float, side: float = 0.0) -> float:
    """dz/dx of the slip surface at x; at a polyline's corner, of the
    segment on the ``side`` (-1 toward -x, +1 toward +x) of it."""
    if getattr(slip, "is_circular", False):
        ddx = x - slip.xc
        dz = math.sqrt(max(slip.radius ** 2 - ddx * ddx, 1e-30))
        return ddx / dz
    pts = slip.points
    n = len(pts) - 1
    for i in range(n):
        (xa, za), (xb, zb) = pts[i], pts[i + 1]
        if xa - _EPS <= x <= xb + _EPS:
            if abs(x - xb) <= 1e-9 and side > 0 and i + 1 < n:
                continue
            return (zb - za) / (xb - xa)
    (xa, za), (xb, zb) = (pts[0], pts[1]) if x < pts[0][0] else (pts[-2],
                                                                 pts[-1])
    return (zb - za) / (xb - xa)


def crest_side(geom, slip, x_entry: float, x_exit: float) -> float:
    """+1 where the surface's upper end is its +x end (the mass slides
    toward -x), -1 where it is its -x end."""
    za = geom.ground_elevation_at(x_entry)
    zb = geom.ground_elevation_at(x_exit)
    return 1.0 if zb >= za else -1.0


def up_slope_tangent(slip, x: float, crest: float,
                     side: float = 0.0) -> Tuple[float, float]:
    """Unit vector along the slip surface at x pointing against the
    sliding (its horizontal part toward the crest)."""
    m = base_slope(slip, x, side)
    n = math.hypot(1.0, m)
    return (crest / n, crest * m / n)


# ---------------------------------------------------------------------------
# What a geosynthetic layer can carry at a crossing (D34)
# ---------------------------------------------------------------------------

def buried_extent(geom, z: float, x: float) -> Tuple[float, float]:
    """The stretch (a, b) around x over which the level z is below the
    ground: the ends are where the ground comes down to z (found on the
    ground's own segments) or the ends of the model."""
    pts = geom.surface_points
    a, b = pts[0][0], pts[-1][0]
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        if (za - z) * (zb - z) < 0:
            xi = xa + (z - za) * (xb - xa) / (zb - za)
        elif za == z and zb != z:
            xi = xa
        else:
            continue
        if xi <= x:
            a = max(a, xi)
        else:
            b = min(b, xi)
    return a, b


def overburden(geom, x: float, z: float) -> float:
    """Total vertical stress gamma z at (x, z): the soil column above it,
    each soil at its own unit weight (kPa)."""
    return sum(lay.gamma * (top - bot)
               for lay, bot, top in geom.column_at(x, z_low=z))


def _sigma_v(geom, x: float, z: float, effective: bool) -> float:
    s = overburden(geom, x, z)
    if effective:
        zw = geom.gwt_elevation_at(x)
        if zw is not None and zw > z:
            s -= GAMMA_W * (zw - z)
    return max(s, 0.0)


def stress_integral(geom, z: float, xa: float, xb: float,
                    effective: bool = False) -> float:
    """Integral of the vertical stress on the level z from xa to xb
    (kN/m): Simpson's rule on each stretch between the ground's corners,
    exact where the stress varies linearly along the stretch (level
    ground, Eq. 8.8, and a uniform slope face, Eq. 8.9)."""
    lo, hi = min(xa, xb), max(xa, xb)
    if hi - lo <= 0:
        return 0.0
    cuts = [lo] + [p[0] for p in geom.surface_points
                   if lo < p[0] < hi] + [hi]
    total = 0.0
    n = 8
    for p, q in zip(cuts, cuts[1:]):
        h = (q - p) / n
        acc = 0.0
        for k in range(n + 1):
            w = 1 if k in (0, n) else (4 if k % 2 else 2)
            acc += w * _sigma_v(geom, p + k * h, z, effective)
        total += acc * h / 3.0
    return total


def geosynthetic_pullout_basis(g) -> str:
    """The basis a layer's pullout is found on: the one named, else by the
    inputs given: 'given' with a development length, 'dwb2014' with F* and
    alpha, else 'not_checked'."""
    named = getattr(g, "pullout_basis", None)
    if named:
        return named
    if getattr(g, "development_length", None):
        return "given"
    if getattr(g, "F_star", None) and getattr(g, "alpha", None):
        return "dwb2014"
    return "not_checked"


def layer_capacity(geom, g, x_c: float, anchorage: float, passive: bool
                   ) -> Tuple[float, str, str]:
    """(T, controlled_by, note): the force a geosynthetic layer can carry
    where the slip surface crosses it at x_c, its anchorage on the
    ``anchorage`` side (+1 toward +x): the lesser of its strength
    (``T_allow``) and its pullout resistance (see the module docstring).
    """
    basis = geosynthetic_pullout_basis(g)
    T = g.T_allow
    if basis == "not_checked":
        return T, "allowable", "pullout not checked"
    a, b = buried_extent(geom, g.elevation, x_c)
    if g.x_start is not None:
        a = max(a, g.x_start)
    if g.x_end is not None:
        b = min(b, g.x_end)
    end_behind = b if anchorage > 0 else a
    end_front = a if anchorage > 0 else b
    held = getattr(g, "anchored_end", None)     # 'start' (-x) or 'end' (+x)
    behind_held = held == ("end" if anchorage > 0 else "start")
    front_held = held == ("start" if anchorage > 0 else "end")
    L_b = abs(end_behind - x_c)
    L_f = abs(x_c - end_front)

    if basis == "given":
        Ld = g.development_length
        caps = []
        if not behind_held:
            caps.append((T * min(1.0, L_b / Ld), "pullout",
                         f"{L_b:.2f} m behind the surface"))
        if not front_held:
            caps.append((T * min(1.0, L_f / Ld), "pullout_front",
                         f"{L_f:.2f} m in front of the surface"))
        how = (f"the force develops linearly over {Ld:g} m from an end "
               f"(development_length; Duncan, Wright and Brandon 2014, "
               f"Fig. 8.2)")
    else:
        if not (getattr(g, "F_star", None) and getattr(g, "alpha", None)):
            raise ValueError(
                f"A geosynthetic's pullout by '{basis}' needs F_star and "
                f"alpha (the pullout resistance factor and the scale "
                f"effect factor: findings about the fill and the product, "
                f"GEC 11 Table 3-6, p. 3-17; Duncan, Wright and Brandon "
                f"2014, Table 8.3). Give them, or a development_length, "
                f"or pullout_basis='not_checked'.")
        gec = basis == "gec11"
        factor = 1.0
        if gec:
            # GEC 11 Eq. 9-9 (p. 9-13) carries FS whatever the method:
            # its own rotational analysis is Method B's form (p. 9-24)
            factor = getattr(g, "pullout_factor", None)
            if not factor:
                raise ValueError(
                    "A geosynthetic's pullout by 'gec11' is divided by the "
                    "factor of safety FS of GEC 11 Eq. 9-9 (p. 9-13) under "
                    "Method A and Method B alike, and no page prints one "
                    "value for every case: give pullout_factor (GEC 11 p. "
                    "9-5: 1.5 for granular soils, 2 for cohesive soils).")
        elif not passive:
            factor = getattr(g, "pullout_factor", None)
            if not factor:
                raise ValueError(
                    "An active (Method A) geosynthetic's pullout "
                    "resistance is divided by a factor that no page "
                    "prints one value of: give pullout_factor (Duncan, "
                    "Wright and Brandon 2014, Table 8.4: F_R = 1.3, 1.5 "
                    "or 2.0 by consequence and uncertainty; GEC 11 p. "
                    "9-5: 1.5 for granular soils, 2 for cohesive soils). "
                    "A passive (Method B) layer's is not divided: its "
                    "ultimate force is divided by the factor of safety.")
        rc = getattr(g, "coverage_ratio", 1.0) if gec else 1.0
        k = 2.0 * g.alpha * g.F_star * rc / factor

        def pull(x_end):
            return k * stress_integral(geom, g.elevation, x_c, x_end,
                                       effective=gec)
        caps = []
        if not behind_held:
            if gec and L_b < GEC11_MIN_EMBEDMENT_M:
                caps.append((0.0, "embedment",
                             f"{L_b:.2f} m behind the surface, less than "
                             f"the 1 m that counts (GEC 11 p. 9-14)"))
            else:
                caps.append((pull(end_behind), "pullout",
                             f"{L_b:.2f} m behind the surface"))
        if not gec and not front_held:
            caps.append((pull(end_front), "pullout_front",
                         f"{L_f:.2f} m in front of the surface"))
        how = ("GEC 11 Eq. 9-9 (p. 9-13), behind the surface only, on the "
               "effective vertical stress" if gec else
               "Duncan, Wright and Brandon (2014) Eqs. 8.8 and 8.9, from "
               "each end (Fig. 8.2)")
        if gec:
            how += (f", divided by FS = {factor:g} under Method A and "
                    f"Method B alike, as Eq. 9-9 prints it")
        elif not passive:
            how += f", divided by {factor:g}"
    ctrl, note = "allowable", how
    for cap, name, where in caps:
        if cap < T:
            T, ctrl, note = cap, name, f"{where}; {how}"
    return T, ctrl, note


# ---------------------------------------------------------------------------
# The supports of a geometry on one surface
# ---------------------------------------------------------------------------

def _in_mass(geom, slip, x_entry, x_exit, x, z) -> bool:
    if x < x_entry - 1e-7 or x > x_exit + 1e-7:
        return False
    zs = slip_z(slip, min(max(x, x_entry), x_exit))
    return zs is None or z >= zs - 1e-7


def check_legacy_force(geom) -> None:
    """Refuse the legacy single force given without an elevation (D37):
    it has no line of action, and was left out without notice."""
    force = getattr(geom, "reinforcement_force", 0.0) or 0.0
    if force > 0 and getattr(geom, "reinforcement_elevation", None) is None:
        raise ValueError(
            "reinforcement_force is given without reinforcement_elevation: "
            "a support's force acts where the reinforcement crosses the "
            "slip surface (Duncan, Wright and Brandon 2014, Eq. 6.70, p. "
            "92), so it needs the elevation it acts at. Give "
            "reinforcement_elevation, or the layer as a geosynthetic "
            "(geosynthetics), or leave the force out. Until GM-140 step 8 "
            "such a force was left out of the analysis without notice.")


def resolve_supports(geom, slip, x_entry: float,
                     x_exit: float) -> SupportResolution:
    """The supports of ``geom`` on the slip surface between its two ends:
    each crossing found exactly, the force along the stated direction, no
    larger than the support can carry there, and a support whose pull
    assists the sliding slack unless ``geom.assisting_support`` is
    'as_given' (see the module docstring)."""
    from slope_stability.reinforcement import (
        ReinforcementForce, Geosynthetic, _into_slope_sign,
        ito_matsui_lateral_force)
    check_legacy_force(geom)
    res = SupportResolution()
    crest = crest_side(geom, slip, x_entry, x_exit)
    reading = getattr(geom, "assisting_support", "slack")

    def add(f, mass_side):
        # the direction: the support's own, or parallel to the surface
        up = up_slope_tangent(slip, f.x, crest, mass_side)
        f.side = mass_side
        if f.T <= 0:
            res.not_counted.append(_entry(f, f.note or "no capacity at "
                                          "the crossing"))
            return
        along = f.dir_x * up[0] + f.dir_z * up[1]
        if along < -1e-9:
            if reading == "slack":
                res.not_counted.append(_entry(
                    f, "its pull along the slip surface where it crosses "
                    "points down the slope, so it would assist the "
                    "sliding: taken as slack, it counts nothing (OWN-59; "
                    "assisting_support='as_given' applies the force as "
                    "it stands)"))
                return
            f.note = (f.note + "; " if f.note else "") + (
                "its pull along the slip surface points down the slope; "
                "applied as it stands (assisting_support='as_given')")
        res.forces.append(f)

    def direct(f, element, mass_side):
        if getattr(element, "force_direction",
                   "along_reinforcement") == "along_slip_surface":
            f.dir_x, f.dir_z = up_slope_tangent(slip, f.x, crest, mass_side)
        return f

    # --- nails -----------------------------------------------------------
    for i, nail in enumerate(getattr(geom, "nails", None) or []):
        hit = _first_crossing(geom, slip, x_entry, x_exit, nail.x_head,
                              nail.z_head, nail.inclination, nail.length,
                              _into_slope_sign(geom, nail.x_head))
        if hit is None:
            continue
        t_int, x_int, z_int, dx, dz = hit
        rate = (nail.bond_stress * math.pi
                * nail.drill_hole_diameter / 1000.0)        # kN/m per nail
        caps = [(rate * max(nail.length - t_int, 0.0), "pullout"),
                (nail.tensile_capacity_kN, "tensile")]
        facing = getattr(nail, "facing_capacity", None)
        if facing is not None:
            # GEC 7 Eq. 5.2 (p. 105) with Fig. 5.7 (p. 106), read: in
            # front of the surface the load is limited by the facing and
            # the pullout of the length between the facing and the
            # surface (a reading of the figure; the literal Eq. 5.2 is
            # min(R_T, R_F, R_PO))
            caps.append((facing + rate * t_int, "facing"))
        cap, ctrl = min(caps)
        f = ReinforcementForce(
            x=x_int, z=z_int, T=cap / nail.spacing_h, dir_x=dx, dir_z=dz,
            kind="nail", index=i, controlled_by=ctrl,
            passive=(nail.support_convention == "passive"))
        add(direct(f, nail, -math.copysign(1.0, dx)),
            -math.copysign(1.0, dx))
    nails = getattr(geom, "nails", None) or []
    if nails and any(getattr(n, "facing_capacity", None) is None
                     for n in nails):
        res.notes.append(
            "Soil nails with no facing_capacity: the facing term of GEC 7 "
            "Eq. 5.2 (p. 105; the facing resistance with the pullout of "
            "the length in front of the slip surface) was not checked for "
            "them. Their load is the lesser of the pullout behind the "
            "surface and the bar.")

    # --- anchors ---------------------------------------------------------
    for i, a in enumerate(getattr(geom, "anchors", None) or []):
        hit = _first_crossing(geom, slip, x_entry, x_exit, a.x_head,
                              a.z_head, a.inclination, a.length,
                              _into_slope_sign(geom, a.x_head))
        if hit is None:
            continue
        t_int, x_int, z_int, dx, dz = hit
        if a.length - t_int <= 0.05 * a.length:
            continue  # bond zone not beyond the surface
        f = ReinforcementForce(
            x=x_int, z=z_int, T=a.T_allow, dir_x=dx, dir_z=dz,
            kind="anchor", index=i, controlled_by="allowable",
            passive=(a.support_convention == "passive"))
        add(direct(f, a, -math.copysign(1.0, dx)), -math.copysign(1.0, dx))

    # --- geosynthetics (and the legacy single force) ---------------------
    geos = list(getattr(geom, "geosynthetics", None) or [])
    legacy_F = getattr(geom, "reinforcement_force", 0.0) or 0.0
    if legacy_F > 0:
        geos.append(Geosynthetic(elevation=geom.reinforcement_elevation,
                                 T_allow=legacy_F))
    unchecked = 0
    for i, g in enumerate(geos):
        passive = getattr(g, "support_convention", "active") == "passive"
        crossed = False
        for x_c, s in level_crossings(slip, x_entry, x_exit, g.elevation):
            if g.x_start is not None and x_c < g.x_start - 1e-9:
                continue
            if g.x_end is not None and x_c > g.x_end + 1e-9:
                continue
            if geom.ground_elevation_at(x_c) <= g.elevation + 1e-7:
                continue        # at or above the ground there: not buried
            crossed = True
            T, ctrl, note = layer_capacity(geom, g, x_c, s, passive)
            f = ReinforcementForce(
                x=x_c, z=g.elevation, T=T, dir_x=s, dir_z=0.0,
                kind="geosynthetic", index=i, controlled_by=ctrl,
                passive=passive, note=note)
            add(direct(f, g, -s), -s)
        if crossed and geosynthetic_pullout_basis(g) == "not_checked":
            unchecked += 1
    if unchecked:
        res.notes.append(
            f"{unchecked} geosynthetic layer(s) crossing the surface carry "
            f"their full force (T_allow) there: their pullout resistance "
            f"was not checked, because no pullout input was given. The "
            f"capacity of a layer is the lesser of its strength and its "
            f"pullout resistance on each side of the crossing (Duncan, "
            f"Wright and Brandon 2014, Fig. 8.2, p. 162; GEC 11 Sec. 9.1, "
            f"p. 9-1), which matters for surfaces near a layer's ends. "
            f"Give the layer's ends (x_start, x_end) with F_star, alpha "
            f"and pullout_factor, or a development_length.")

    # --- stabilizing piles -----------------------------------------------
    for i, p in enumerate(getattr(geom, "stabilizing_piles", None) or []):
        if p.x < x_entry or p.x > x_exit:
            continue
        z_slip = slip_z(slip, p.x)
        if z_slip is None:
            continue
        z_head = (p.z_head if p.z_head is not None
                  else geom.ground_elevation_at(p.x))
        if z_slip >= z_head:
            continue                       # slip surface above the pile head
        if p.z_toe is not None and z_slip <= p.z_toe:
            continue                       # slip surface below the pile toe
        if p.ito_matsui:
            lay = geom.layer_at_point(p.x, 0.5 * (z_head + z_slip))
            c = p.c if p.c is not None else (lay.c_prime if lay else 0.0)
            phi = p.phi if p.phi is not None else (lay.phi if lay else 0.0)
            gamma = (p.gamma if p.gamma is not None
                     else (lay.gamma if lay else 0.0))
            F_pile = ito_matsui_lateral_force(
                c=c, phi=phi, gamma=gamma, D1=p.spacing,
                D2=p.spacing - p.diameter, z_top=z_head, z_bot=z_slip)
            ctrl = "ito_matsui"
        else:
            F_pile = p.shear_capacity
            ctrl = "shear_capacity"
        T = F_pile / p.spacing             # per metre of slope run
        if T <= 0:
            continue
        if p.force_direction == "along_slip_surface":
            dir_x, dir_z = up_slope_tangent(slip, p.x, crest)
        else:
            dir_x, dir_z = crest, 0.0
        f = ReinforcementForce(
            x=p.x, z=z_slip, T=T, dir_x=dir_x, dir_z=dir_z,
            kind="pile", index=i, controlled_by=ctrl,
            passive=(p.support_convention == "passive"))
        add(f, 0.0)
    return res


def _entry(f, reason: str) -> Dict[str, Any]:
    return {"kind": f.kind, "index": f.index, "x_m": round(f.x, 3),
            "z_m": round(f.z, 3), "T_kN_per_m": round(f.T, 2),
            "reason": reason}


def _first_crossing(geom, slip, x_entry, x_exit, x_head, z_head,
                    inclination_deg, length, into):
    """Where a nail or an anchor first leaves the sliding mass: (t, x, z,
    dir_x, dir_z), or None when its head is not in the mass or it does not
    reach the surface. The direction is the element's own, from its head
    toward its anchorage (``into`` = +1 where the slope rises toward
    +x)."""
    beta = math.radians(inclination_deg)
    dx, dz = into * math.cos(beta), -math.sin(beta)
    if not _in_mass(geom, slip, x_entry, x_exit, x_head, z_head):
        return None
    for t, x, z in line_crossings(slip, x_entry, x_exit, x_head, z_head,
                                  dx, dz, length):
        if t > 1e-9 * max(1.0, length):
            return t, x, z, dx, dz
    return None


# ---------------------------------------------------------------------------
# The slice a support acts on (GM-353)
# ---------------------------------------------------------------------------

def slice_index_for(bounds: List[Tuple[float, float]], x: float,
                    side: float = 0.0) -> Optional[int]:
    """Index of the slice (``bounds`` = [(x_left, x_right)] in x order)
    whose base a support crossing the slip surface at x acts on.

    The slice whose base holds the crossing. Where the crossing is on a
    slice boundary (within `BOUNDARY_TOL`): the slice on the ``side`` of it
    that holds the part of the support inside the sliding mass (-1 toward
    -x, +1 toward +x), since that slice is the body the support pulls on;
    with no side (a vertical pile), the slice toward -x. Before GM-140
    step 8 a rounding decided it (GM-353). The module's own rule: no
    reference held says which slice a force on a boundary belongs to."""
    n = len(bounds)
    for i, (xl, xr) in enumerate(bounds):
        if xl - BOUNDARY_TOL <= x <= xr + BOUNDARY_TOL:
            if (abs(x - xr) <= BOUNDARY_TOL and side > 0 and i + 1 < n
                    and abs(bounds[i + 1][0] - xr) <= BOUNDARY_TOL):
                return i + 1
            return i
    return None


# ---------------------------------------------------------------------------
# What a result says about its supports
# ---------------------------------------------------------------------------

def has_supports(geom) -> bool:
    return bool(getattr(geom, "nails", None)
                or getattr(geom, "anchors", None)
                or getattr(geom, "geosynthetics", None)
                or getattr(geom, "stabilizing_piles", None)
                or (getattr(geom, "reinforcement_force", 0.0) or 0.0) > 0)


_METHOD_A = ("Method A (active): the support forces are allowable forces, "
             "not divided by the factor of safety")
_METHOD_B = ("Method B (passive): the support forces are ultimate "
             "(long-term capacity) forces, divided by the factor of "
             "safety as the soil strength is")


def support_statements(geom) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(choices, notes): what a result with supports says about them
    (``geotech_common.choices``): Method A or B and the measure of force
    the input is taken to be, the direction of the forces, what a support
    whose pull assists the sliding counts, and how the pullout of the
    geosynthetic layers is found. Empty without supports."""
    from geotech_common.choices import MODULE, choice, how_given
    if not has_supports(geom):
        return [], []
    elements = []
    for name in ("nails", "anchors", "geosynthetics", "stabilizing_piles"):
        elements += list(getattr(geom, name, None) or [])
    conv = {getattr(e, "support_convention", "active") for e in elements}
    if (getattr(geom, "reinforcement_force", 0.0) or 0.0) > 0:
        conv.add("active")
    if conv == {"active"}:
        chosen = _METHOD_A
    elif conv == {"passive"}:
        chosen = _METHOD_B
    else:
        chosen = ("Method A for the supports marked active and Method B "
                  "for those marked passive")
    choices = [choice(
        "how the support forces enter the factor of safety",
        chosen,
        how_given("active" if conv == {"active"} else "mixed", "active",
                  within="supports"),
        "Duncan, Wright and Brandon (2014) Sec. 8.2, pp. 159 to 160. "
        "Method A: F = soil resisting moment / (overturning moment - "
        "reinforcement moment), Eqs. 8.1 and 8.2; enter the allowable "
        "force. Method B: F = (soil resisting moment + reinforcement "
        "moment) / overturning moment, Eqs. 8.3 and 8.4; enter the "
        "ultimate force. Method A is preferable (p. 160). GEC 11 p. "
        "9-24, Eq. 9-17, FS = MR / (MD - Ts D), the form that reduces the "
        "driving moment: there \"the stability factory of safety is not "
        "applied to the reinforcement\" and \"the allowable strength of "
        "the reinforcement Tal must be divided by a required minimum "
        "factor of safety FSR = 1.3\"; the allowable force is the "
        "input, and nothing here divides it. In Bishop's "
        "simplified method the forces enter by Eq. 6.78 (pp. 91 to 93): "
        "the moment about the centre, and the vertical component in the "
        "slice's vertical equilibrium; a horizontal force enters the "
        "moment only",
        [_METHOD_A + " (support_convention 'active')",
         _METHOD_B + " (support_convention 'passive')"])]

    dirs = {getattr(e, "force_direction", "along_reinforcement")
            for e in elements}
    parallel = "along_slip_surface" in dirs
    choices.append(choice(
        "direction of the support forces",
        ("parallel to the slip surface where marked (force_direction "
         "'along_slip_surface'), the others along the support's own "
         "orientation" if parallel and len(dirs) > 1 else
         "parallel to the slip surface (force_direction "
         "'along_slip_surface')" if parallel else
         "along the support's own orientation (a stabilizing pile's "
         "shear: horizontal)"),
        how_given("along_slip_surface" if parallel else
                  "along_reinforcement", "along_reinforcement",
                  within="supports"),
        "Duncan, Wright and Brandon (2014) Sec. 8.6, pp. 163 to 164: the "
        "reinforcement's original orientation is recommended, the more "
        "conservative; parallel to the slip surface is the other extreme, "
        "the option's print GEC 11 p. 9-7 (continuous sheet reinforcement "
        "\"assumed to act tangentially to the circle\", the moment arm D "
        "= R under Eq. 9-1)",
        ["along the support's own orientation (force_direction "
         "'along_reinforcement'; a pile's 'horizontal')",
         "parallel to the slip surface (force_direction "
         "'along_slip_surface')"]))

    reading = getattr(geom, "assisting_support", "slack")
    choices.append(choice(
        "a support whose pull assists the sliding",
        ("counts nothing (slack)" if reading == "slack" else
         "applied as it stands"),
        how_given(reading, "slack", name="assisting_support"),
        "the owner's call (OWN-59): no reference held prints one. The "
        "test, this module's own: the force's component along the slip "
        "surface where the support crosses it points down the slope",
        ["counts nothing (assisting_support 'slack')",
         "applied as it stands (assisting_support 'as_given')"]))

    geos = list(getattr(geom, "geosynthetics", None) or [])
    bases = sorted({geosynthetic_pullout_basis(g) for g in geos})
    if (getattr(geom, "reinforcement_force", 0.0) or 0.0) > 0:
        bases = sorted(set(bases) | {"not_checked"})
    if bases:
        text = {
            "dwb2014": "from each end of the layer by Duncan, Wright and "
                       "Brandon (2014) Eqs. 8.8 and 8.9 (pullout_basis "
                       "'dwb2014')",
            "gec11": "behind the surface by GEC 11 Eq. 9-9, divided by "
                     "FS (pullout_factor) under Method A and Method B, at "
                     "least 1 m (pullout_basis 'gec11')",
            "given": "from the development length given "
                     "(development_length)",
            "not_checked": "not checked: the full T_allow at every "
                           "crossing (no pullout input given)"}
        choices.append(choice(
            "pullout resistance of the geosynthetic layers",
            "; ".join(text[b] for b in bases), MODULE,
            "the force a layer carries at the slip surface is the lesser "
            "of its strength and its pullout resistance (Duncan, Wright "
            "and Brandon 2014, Sec. 8.4, Fig. 8.2; GEC 11 Sec. 9.1, p. "
            "9-1). With F_star and alpha given the default is the "
            "book's, whose Methods A and B the forces are entered by: "
            "pullout from each end, 2 alpha F* gamma z integrated along "
            "the embedded length (Eqs. 8.8 and 8.9), divided by "
            "pullout_factor under Method A (Table 8.4 prints 1.3, 1.5 "
            "and 2.0; none is a default). GEC 11's form (behind the "
            "surface only, on the effective stress, with the coverage "
            "ratio, 1 m at least; divided by FS, 1.5 granular, 2 "
            "cohesive, p. 9-5, under either method, as Eq. 9-9 on p. 9-13 "
            "prints it) is the option",
            [text["dwb2014"], text["gec11"], text["given"],
             text["not_checked"]]))
    return choices, []


def not_counted_warning(not_counted) -> Optional[str]:
    """The plain statement of the supports that cross the surface and
    count nothing, or None."""
    if not not_counted:
        return None
    parts = "; ".join(
        f"{r['kind']} {r['index']} at x = {r['x_m']:.2f} m "
        f"({r['T_kN_per_m']:.1f} kN/m): {r['reason']}"
        for r in not_counted)
    return (f"{len(not_counted)} support(s) cross this surface and count "
            f"nothing in the factor of safety: {parts}.")


def add_support_statements(result, geom) -> None:
    """Put ``support_statements(geom)`` on a result (a single surface's or
    a search's), after its own ``choices``; on a search, off its critical
    surface, so they are said once."""
    choices, notes = support_statements(geom)
    if not choices:
        return
    whats = {c["what"] for c in choices}
    have = [c for c in (getattr(result, "choices", None) or [])
            if c["what"] not in whats]
    result.choices = have + choices
    for n in notes:
        if n not in result.warnings:
            result.warnings.append(n)
    critical = getattr(result, "critical", None)
    if critical is not None and getattr(critical, "choices", None):
        critical.choices = [c for c in critical.choices
                            if c["what"] not in whats]
