"""Tension cracks and reverse-curvature circles as printed (GM-140 step 4).

The rule, one for every method, surface family and caller:

* A tension crack ends the slip surface. The soil upslope of the crack is
  left out of the computation, the slices end at a vertical boundary at the
  crack, and water in the crack acts on that boundary (EM 1110-2-1902,
  31 Oct 2003, 2-2f(3), p. 2-6: the soil upslope from the crack "is then
  ignored in the stability computations ... by terminating the slices near
  the crest of the slope with a slice having a vertical boundary"; C-10c,
  p. C-36: the slip surface is terminated where it reaches the bottom of
  the crack; Slide v6 Tutorial 16, p. 16-1: "A tension crack essentially
  terminates the slip surface").
* The crack is at the upper end of the slip surface, the end where the
  ground is higher (EM 2-2f(3): "near the crest of the slope").
* Its depth is one of three (``DEPTH_BASES``):

  - ``given``: the depth the caller gives;
  - ``rankine``: the depth over which Rankine's active pressures are
    negative, with the developed strengths (EM C-10c(2), p. C-36, Eq. C-36,
    "for total stresses and homogeneous soil"; the same depth as Eq. 2-4,
    p. 2-9)::

        d_crack = 2 c_D / (gamma tan(45 deg - phi_D / 2))     (Eq. C-36)
        c_D = c / F                                            (Eq. 2-5)
        phi_D = arctan(tan(phi) / F)                           (Eq. 2-6)

    F being the factor of safety of the surface itself, so the solution is
    repeated until the depth stops changing. Where Eq. C-36 gives a depth
    greater than the thickness of the soil it was computed for, the crack
    stops at that soil's bottom (C-10c: the crack is as deep as the
    embankment and does not extend into the foundation);
  - ``min_fos``: the depth giving the minimum factor of safety, from a scan
    of depths (EM 2-2f(4), p. 2-6; C-10c(1), p. C-36).

* The crack is a condition of the analysis, set by the engineer and the
  same on every trial surface; a circle's shape never switches it on (the
  orchestrator's ruling on the step 4 review's F1, refining OWN-103). A
  reverse-curvature circle (its centre below the ground at its upper end,
  so that the arc would turn back over itself) is closed by the crack asked
  for where its overhang lies within that crack's depth; where the overhang
  runs deeper, the circle is discarded, counted and located (OWN-103's
  limit). With no crack asked for, a reverse-curvature circle is not an
  admissible surface: it is discarded, counted and located, and naming a
  crack admits those within its depth. No printed source gives a crack at
  the lower end of a surface, so a circle that overhangs there is
  discarded as well.
* A surface that nowhere passes below the crack's depth (measured below
  the ground above it) lies wholly within the cracked depth and is no
  surface: it is refused and counted (the step 4 review's D1). The rule is
  geometric; no threshold on F or on the size of the mass.

Every solve of one surface goes through ``solve_with_crack``: it hands the
solver a copy of the geometry with the crack fixed (``fixed_geometry``),
and ``slices.build_slices`` reads the crack from there. ``build_slices``
on a geometry whose crack still depends on the factor of safety raises
``CrackDepthNeeded``, so no caller can slice such a surface on a silent
basis.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

GIVEN = "given"
RANKINE = "rankine"
MIN_FOS = "min_fos"
#: How a tension crack's depth is set (``SlopeGeometry.tension_crack_depth_basis``).
DEPTH_BASES = (GIVEN, RANKINE, MIN_FOS)

CREST = "crest"
ENTRY = "entry"
EXIT = "exit"
#: Which end of the surface the crack is at (``tension_crack_side``).
SIDES = (CREST, ENTRY, EXIT)

TRUNCATION = "truncation"
STRENGTH = "strength"
#: What the crack does to the soil upslope of it (``tension_crack_model``).
MODELS = (TRUNCATION, STRENGTH)

#: The Rankine depth has stopped changing when two successive solutions
#: give depths this close (m).
DEPTH_TOL = 1e-3
#: Solutions repeated at most this many times for the Rankine depth.
MAX_ITERATIONS = 25
#: Equal steps of the scan for the depth of minimum factor of safety.
SCAN_STEPS = 20

_EM_TERMINATE = ("EM 1110-2-1902 (2003) 2-2f(3), p. 2-6, and C-10c, "
                 "p. C-36; Slide v6 Tutorial 16, p. 16-1")
_EM_RANKINE = "EM 1110-2-1902 (2003) Eq. C-36, p. C-36, with Eqs. 2-5 and 2-6"
_EM_SCAN = "EM 1110-2-1902 (2003) 2-2f(4), p. 2-6, and C-10c(1), p. C-36"


class CrackDepthNeeded(ValueError):
    """The crack on this surface depends on the surface's factor of safety
    (a Rankine or minimum-FOS depth, or a reverse-curvature circle), and
    the slices were asked for without it. Solve through
    ``solve_with_crack`` (``analyze_slope`` and the searches do)."""


#: Why the crack rule refuses a surface, each counted on its own in a
#: search's result (``SearchResult.crack_refused``; the step 4 review's
#: D5), with the words the result and the search's choices use.
NO_CRACK_ASKED = "reverse_curvature_no_crack"
DEEPER_THAN_CRACK = "reverse_curvature_deeper_than_crack"
NO_CRACK_AT_END = "reverse_curvature_no_crack_at_end"
DISCARD_ASKED = "reverse_curvature_discard_asked"
WITHIN_CRACK = "within_cracked_depth"
SCAN_NO_MINIMUM = "scan_no_minimum"
BOTTOM_NOT_REACHED = "crack_bottom_not_reached"
REFUSAL_TEXT = {
    NO_CRACK_ASKED: (
        "reverse-curvature circles with no tension crack asked for: not "
        "admissible surfaces; naming a crack (tension_crack_depth, or "
        "tension_crack_depth_basis 'rankine' or 'min_fos') admits those "
        "whose overhang lies within its depth"),
    DEEPER_THAN_CRACK: (
        "reverse-curvature circles whose overhang runs deeper than the "
        "crack asked for, which would close them (OWN-103)"),
    NO_CRACK_AT_END: (
        "circles with reverse curvature at an end where no crack is (the "
        "lower end, or the end away from a crack put at the other one)"),
    DISCARD_ASKED: (
        "reverse-curvature circles discarded as asked "
        "(reverse_curvature='discard')"),
    WITHIN_CRACK: (
        "surfaces that nowhere pass below the crack's depth, measured below "
        "the ground above them: they lie wholly within the cracked depth "
        "and are no slip surface"),
    SCAN_NO_MINIMUM: (
        "surfaces on which the scan for the crack depth of minimum factor "
        "of safety found no minimum"),
    BOTTOM_NOT_REACHED: (
        "surfaces on which the crack's bottom is not reached at the upper "
        "end: from the middle of the surface to its crest end the surface "
        "does not pass below the crack's bottom, so it has no vertical "
        "boundary at the crack (EM 1110-2-1902 C-10c, p. C-36: the slip "
        "surface is terminated where it reaches the bottom of the crack)"),
}
#: The refusals of reverse-curvature circles.
REVERSE_REFUSALS = (NO_CRACK_ASKED, DEEPER_THAN_CRACK, NO_CRACK_AT_END,
                    DISCARD_ASKED)


class CrackRuleRefused(ValueError):
    """A surface the crack rule refuses (``REFUSAL_TEXT``): it has no
    factor of safety. ``outcome.refusal`` says why."""

    def __init__(self, message: str, outcome: "CrackOutcome"):
        super().__init__(message)
        self.outcome = outcome


class ReverseCurvatureDiscarded(CrackRuleRefused):
    """A reverse-curvature circle the crack rule discards: no crack asked
    for, an overhang deeper than the crack (OWN-103), an overhang at an end
    no crack is at, or every such circle discarded as asked."""


def refusal_error(outcome: "CrackOutcome") -> CrackRuleRefused:
    """The error a single-surface caller raises for a refused surface."""
    cls = (ReverseCurvatureDiscarded if outcome.refusal in REVERSE_REFUSALS
           else CrackRuleRefused)
    return cls("No factor of safety: " + outcome.reason + ".", outcome)


def bottom_not_reached_error(geom, crack: Dict[str, Any], x_entry: float,
                             x_exit: float) -> CrackRuleRefused:
    """The error ``slices`` raises where the surface, from its middle to
    its crest end, never passes below the crack's bottom: there the slices
    could only end at the middle of the surface, a stub carrying the
    crack's full water force, which no printed method describes (the
    step 4 review's D1 found a search reading F = 0.052 on such stubs).
    Refused and counted, never solved."""
    x_end = x_entry if crack["side"] == ENTRY else x_exit
    reason = ("the crack's bottom (el. %.3f, %.3f m below the ground at the "
              "surface's upper end, x = %.3f) is not reached at the upper "
              "end: between the middle of the surface and that end it does "
              "not pass below the crack's bottom, so it has no vertical "
              "boundary at the crack" % (crack["base"], crack["depth"], x_end))
    outcome = CrackOutcome(
        depth=crack["depth"], side=crack["side"],
        model=geom.tension_crack_model, x_end=x_end,
        z_ground=geom.ground_elevation_at(x_end), discarded=True,
        reason=reason, refusal=BOTTOM_NOT_REACHED)
    return CrackRuleRefused("No factor of safety: " + reason + ".", outcome)


def rankine_crack_depth(c: float, phi_deg: float, gamma: float,
                        fos: float = 1.0) -> float:
    """Depth (m) over which Rankine's active earth pressures are negative,
    with the developed strengths: EM 1110-2-1902 (2003) Eq. C-36, p. C-36,

        d_crack = 2 c_D / (gamma tan(45 deg - phi_D / 2)),

    c_D = c / F (Eq. 2-5) and phi_D = arctan(tan(phi) / F) (Eq. 2-6),
    p. 2-9. Eq. 2-4 (p. 2-9), z_t = (2 c_D / gamma) tan(45 deg + phi_D / 2),
    is the same depth. ``fos`` = 1 gives the depth with the full strengths.
    """
    if c <= 0.0 or gamma <= 0.0 or fos <= 0.0:
        return 0.0
    c_d = c / fos
    phi_d = math.atan(math.tan(math.radians(phi_deg)) / fos)
    return 2.0 * c_d / (gamma * math.tan(math.pi / 4.0 - phi_d / 2.0))


def crest_side(geom, x_entry: float, x_exit: float) -> str:
    """The end of the surface where the ground is higher: 'entry' (low x)
    or 'exit' (high x). EM 1110-2-1902 2-2f(3), p. 2-6: the crack is "near
    the crest of the slope" and the soil upslope of it is ignored. Ends at
    one level (no crest): 'entry'."""
    return (EXIT if geom.ground_elevation_at(x_exit)
            > geom.ground_elevation_at(x_entry) + 1e-9 else ENTRY)


def crack_side(geom, x_entry: float, x_exit: float) -> str:
    """The side the geometry's crack is on for this surface: the given
    side, or the crest's (``crest_side``) by default."""
    side = geom.tension_crack_side
    return crest_side(geom, x_entry, x_exit) if side == CREST else side


def reverse_ends(geom, slip, x_entry: float, x_exit: float
                 ) -> List[Dict[str, Any]]:
    """The ends at which a circle has reverse curvature: its centre lies
    below the ground there, so the arc is cut at its vertical tangent
    (x = xc -/+ R) at the centre's level. Each: {'side', 'x', 'z_ground',
    'depth'} with ``depth`` the overhang's depth, from the ground down to
    the centre's level. Empty for a polyline."""
    if not getattr(slip, "is_circular", False):
        return []
    tol = 1e-3 * slip.radius
    out = []
    for side, x_end, x_tan in ((EXIT, x_exit, slip.xc + slip.radius),
                               (ENTRY, x_entry, slip.xc - slip.radius)):
        if abs(x_end - x_tan) > tol:
            continue
        z_g = geom.ground_elevation_at(x_end)
        if z_g - slip.yc > tol:
            out.append({"side": side, "x": x_end, "z_ground": z_g,
                        "depth": z_g - slip.yc})
    return out


@dataclass
class CrackPlan:
    """What one surface's crack needs before its slices can be cut."""
    side: str
    x_end: float                       # the surface's end on that side
    z_ground: float                    # the ground there
    basis: str                         # GIVEN / RANKINE / MIN_FOS
    depth: float = 0.0                 # the given depth (GIVEN)
    overhang: Optional[float] = None   # reverse curvature: its depth
    discard: Optional[str] = None      # why the surface is refused, now
    refusal: str = ""                  # that reason's key (REFUSAL_TEXT)
    scan_limit: float = 0.0            # deepest depth a scan tries
    x_other: float = 0.0               # the surface's other end
    # the greatest depth of the surface below the ground above it
    # (``depth_below_ground``): a crack this deep or deeper holds it whole
    d_max: float = math.inf
    slip: Any = None
    # Rankine's soil: the one at the ground at the crack
    c: float = 0.0
    phi: float = 0.0
    gamma: float = 0.0
    soil: str = ""
    soil_thickness: float = math.inf
    rankine_refused: str = ""          # why Eq. C-36 does not apply here

    @property
    def needs_solution(self) -> bool:
        return self.discard is None and self.basis != GIVEN


@dataclass
class CrackOutcome:
    """The crack a surface was solved with (or why it was discarded)."""
    depth: float = 0.0
    basis: str = GIVEN
    side: str = ENTRY
    model: str = TRUNCATION
    x_end: float = 0.0
    z_ground: float = 0.0
    water_depth: float = 0.0
    reverse_curvature: bool = False
    overhang: Optional[float] = None
    discarded: bool = False
    reason: str = ""
    refusal: str = ""                  # why refused (REFUSAL_TEXT key)
    # where the crack is: the x at which the slices end (the surface
    # reaches the crack's bottom), and the ground there
    x_crack: Optional[float] = None
    z_ground_crack: Optional[float] = None
    fos: Optional[float] = None        # the F the Rankine depth was taken at
    iterations: int = 0
    converged: bool = True
    capped: bool = False               # stopped at the bottom of its soil
    soil: str = ""
    rankine_refused: str = ""
    scan: List[Tuple[float, float]] = field(default_factory=list)
    fixed: Optional[Dict[str, Any]] = None   # for ``geometry_of``

    def to_dict(self) -> Dict[str, Any]:
        # The crack is where the slices end (the step 4 review's D3), its
        # bottom the depth below the ground at the surface's end on its side.
        x = self.x_end if self.x_crack is None else self.x_crack
        z_top = (self.z_ground if self.z_ground_crack is None
                 else self.z_ground_crack)
        d: Dict[str, Any] = {
            "depth_m": round(self.depth, 3),
            "depth_basis": self.basis,
            "side": self.side,
            "model": self.model,
            "x_m": round(x, 3),
            "ground_z_m": round(z_top, 3),
            "base_z_m": round(self.z_ground - self.depth, 3),
            "water_depth_m": round(self.water_depth, 3),
        }
        if self.reverse_curvature:
            d["reverse_curvature"] = True
            d["overhang_depth_m"] = round(self.overhang or 0.0, 3)
        if self.discarded:
            d["discarded"] = True
            d["refusal"] = self.refusal
        if self.reason:
            d["note"] = self.reason
        if self.basis == RANKINE:
            d["fos_for_depth"] = (None if self.fos is None
                                  else round(self.fos, 4))
            d["iterations"] = self.iterations
            d["converged"] = self.converged
            d["stopped_at_soil_bottom"] = self.capped
            d["soil"] = self.soil
        if self.rankine_refused:
            d["rankine_not_used"] = self.rankine_refused
        if self.scan:
            d["scan"] = [{"depth_m": round(a, 3), "FOS": round(b, 4)}
                         for a, b in self.scan]
        return d

    def statement(self) -> str:
        """One plain sentence for a result's warnings."""
        if self.discarded:
            return self.reason
        what = {GIVEN: "given",
                RANKINE: (f"Rankine's depth with the developed strengths at "
                          f"F = {self.fos:.3f} ({_EM_RANKINE})"
                          if self.fos is not None else
                          "Rankine's depth with the full strengths (the "
                          "method found no factor of safety to develop "
                          f"them at; {_EM_RANKINE})"),
                MIN_FOS: (f"the depth of minimum factor of safety from a "
                          f"scan of {len(self.scan)} depths ({_EM_SCAN})"),
                }[self.basis]
        x = self.x_end if self.x_crack is None else self.x_crack
        s = (f"Tension crack {self.depth:.2f} m deep on the "
             f"{'low-x' if self.side == ENTRY else 'high-x'} side of the "
             f"surface, at x = {x:.2f} where the surface reaches its bottom "
             f"(z = {self.z_ground - self.depth:.2f}, the ground at the "
             f"surface's end there, z = {self.z_ground:.2f}, less the "
             f"depth), depth {what}")
        if self.capped:
            s += (f"; Eq. C-36's depth passes the bottom of the soil it was "
                  f"computed for ({self.soil}), so the crack stops there "
                  f"(EM C-10c, p. C-36)")
        if self.rankine_refused:
            s += (f"; Rankine's depth was not used ({self.rankine_refused}), "
                  f"so the depth is the scan's")
        if self.basis == RANKINE and not self.converged \
                and self.fos is not None:
            s += (f"; the depth was still changing after {self.iterations} "
                  f"solutions")
        if self.model == TRUNCATION:
            s += (". The slip surface ends at the bottom of the crack and "
                  "the soil upslope of it is left out (" + _EM_TERMINATE
                  + ")")
        else:
            s += (". The cracked soil is kept as weight without strength "
                  "(tension_crack_model='strength'; the printed model ends "
                  "the surface at the crack)")
        if self.reverse_curvature:
            s += (f". The circle is a reverse-curvature circle (its centre "
                  f"is {self.overhang:.2f} m below the ground at its end, "
                  f"so the arc would turn back over itself) and the crack "
                  f"asked for closes it; a circle overhanging deeper than "
                  f"the crack is discarded")
        return s + "."


def _crest_half_low(slip, x_entry: float, x_exit: float, side: str
                    ) -> Optional[float]:
    """The lowest elevation of the slip surface on the crest half."""
    x_half = 0.5 * (x_entry + x_exit)
    a, b = (x_entry, x_half) if side == ENTRY else (x_half, x_exit)
    xs = [a + (b - a) * i / 40.0 for i in range(41)]
    if getattr(slip, "is_circular", False) and a <= slip.xc <= b:
        xs.append(slip.xc)
    xs += [p[0] for p in (getattr(slip, "points", None) or ())
           if a <= p[0] <= b]
    zs = [z for z in (slip.slip_elevation_at(x) for x in xs) if z is not None]
    return min(zs) if zs else None


def depth_below_ground(geom, slip, x_entry: float, x_exit: float) -> float:
    """The greatest depth of the slip surface below the ground above it,
    between its ends. The ground is straight between its vertices: a
    polyline's depth is then greatest at a vertex of either, and a
    circle's on each straight piece at an end or where the arc runs
    parallel to the ground (x = xc + s R / sqrt(1 + s^2), s the piece's
    slope)."""
    xs = {x_entry, x_exit}
    xs.update(p[0] for p in geom.surface_points if x_entry < p[0] < x_exit)
    xs.update(p[0] for p in (getattr(slip, "points", None) or ())
              if x_entry < p[0] < x_exit)
    if getattr(slip, "is_circular", False):
        bounds = sorted(xs)
        for a, b in zip(bounds[:-1], bounds[1:]):
            s = ((geom.ground_elevation_at(b) - geom.ground_elevation_at(a))
                 / (b - a))
            x = slip.xc + s * slip.radius / math.sqrt(1.0 + s * s)
            if a < x < b:
                xs.add(x)
    best = 0.0
    for x in xs:
        z = slip.slip_elevation_at(x)
        if z is not None:
            best = max(best, geom.ground_elevation_at(x) - z)
    return best


def _crack_x(slip, plan: "CrackPlan", depth: float) -> Optional[float]:
    """Where the crack is: the x at which the slip surface, walked from the
    middle of the surface towards the crack's end, first rises to the
    crack's bottom, which is where the slices end (``slices._kept_span``);
    the middle of the surface where its crack half is all above it."""
    if slip is None:
        return None
    z_base = plan.z_ground - depth
    x_half = 0.5 * (plan.x_end + plan.x_other)

    def above(x):
        z = slip.slip_elevation_at(x)
        return z is None or z >= z_base

    if above(x_half):
        return x_half
    n = 400
    prev = x_half
    for i in range(1, n + 1):
        x = x_half + (plan.x_end - x_half) * i / n
        if above(x):
            a, b = prev, x
            for _ in range(60):
                m = 0.5 * (a + b)
                a, b = (a, m) if above(m) else (m, b)
            return b
        prev = x
    return plan.x_end


def _rankine_soil(geom, plan: CrackPlan) -> None:
    """Fill the plan with the soil Eq. C-36 is computed for (the one at the
    ground at the crack), or with why the equation does not apply there:
    it is printed "for total stresses and homogeneous soil" (EM C-10c(2),
    p. C-36), with c and phi."""
    x = plan.x_end
    column = geom.column_at(x) if hasattr(geom, "column_at") else None
    layer = geom.layer_at_point(x, plan.z_ground - 1e-6)
    if layer is None:
        plan.rankine_refused = "no soil at the ground at the crack"
        return
    plan.soil = layer.name
    if layer.strength_model == "anisotropic":
        # Rankine's ACTIVE pressures: the undrained strength of the active
        # zone, phi = 0 (the module's reading; Eq. C-36 is printed for one
        # c and phi)
        plan.c, plan.phi, plan.gamma = layer.su_active, 0.0, layer.gamma
    elif layer.strength_model != "mohr_coulomb":
        plan.rankine_refused = (
            f"the soil at the crack ({layer.name}) has a "
            f"'{layer.strength_model}' strength, and Eq. C-36 is printed "
            f"for c and phi")
        return
    else:
        plan.c, plan.phi = layer.shear_strength_params
        plan.gamma = layer.gamma
    bottom = layer.bottom_at(x)
    if column:
        # the soil's own thickness at the crack, one material per point
        for lay, bot_z, top_z in column:
            if lay is layer and top_z >= plan.z_ground - 1e-6:
                bottom = bot_z
                break
    plan.soil_thickness = max(plan.z_ground - bottom, 0.0)


def _water_in_depth(geom, plan: CrackPlan, depth: float) -> str:
    """Why Eq. C-36 does not apply to an effective-stress soil cracked to
    ``depth``: pore water pressure inside the cracked depth (EM 2-2f(6),
    p. 2-9, prints no equation for that case). '' when it applies."""
    layer = geom.layer_at_point(plan.x_end, plan.z_ground - 1e-6)
    if layer is None or layer.analysis_mode != "drained" or depth <= 0:
        return ""
    if geom.pore_pressure_points is not None:
        return ("the soil at the crack is drained and its pore pressures "
                "come from a grid")
    if layer.ru > 0:
        return "the soil at the crack is drained and has ru > 0"
    gwt = geom.gwt_elevation_at(plan.x_end)
    if gwt is not None and gwt > plan.z_ground - depth + 1e-9:
        return ("the soil at the crack is drained and the water table is "
                "inside the cracked depth; Eq. C-36 is printed for total "
                "stresses")
    return ""


def may_need_solution(geom, slip) -> bool:
    """A cheap test, without the surface's ends: False only when the crack
    on this surface cannot depend on its factor of safety (a given depth
    or none, and no reverse curvature)."""
    if getattr(geom, "_crack_fixed", None) is not None:
        return False
    if geom.tension_crack_depth_basis != GIVEN:
        return True
    if not getattr(slip, "is_circular", False):
        return False
    x_lo, x_hi = geom.x_range
    for x in (slip.xc - slip.radius, slip.xc + slip.radius):
        if x_lo <= x <= x_hi and geom.ground_elevation_at(x) > slip.yc:
            return True
    return False


def plan_crack(geom, slip, x_entry: Optional[float] = None,
               x_exit: Optional[float] = None) -> Optional[CrackPlan]:
    """What this surface's crack needs: None for no crack; a plan with
    ``basis`` GIVEN for a depth the slices can be cut with at once; one
    that ``needs_solution`` (a Rankine or minimum-FOS depth); or one with
    ``discard`` set. Raises what ``find_entry_exit`` raises."""
    if getattr(geom, "_crack_fixed", None) is not None:
        return None
    if x_entry is None or x_exit is None:
        x_entry, x_exit = slip.find_entry_exit(geom)
    side = crack_side(geom, x_entry, x_exit)
    basis = geom.tension_crack_depth_basis
    depth = geom.tension_crack_depth
    has_crack = depth > 0 or basis != GIVEN
    rev = reverse_ends(geom, slip, x_entry, x_exit)
    overhang = None
    discard = None
    refusal = ""
    if rev:
        upper = crest_side(geom, x_entry, x_exit)
        low = [e for e in rev if e["side"] != upper]
        at_crest = [e for e in rev if e["side"] == upper]
        if geom.reverse_curvature == "discard":
            e = rev[0]
            refusal = DISCARD_ASKED
            discard = (f"a reverse-curvature circle (its centre "
                       f"{e['depth']:.2f} m below the ground at x = "
                       f"{e['x']:.2f}), discarded as asked "
                       f"(reverse_curvature='discard')")
        elif low:
            e = low[0]
            refusal = NO_CRACK_AT_END
            discard = (f"the circle has reverse curvature at its lower end "
                       f"(x = {e['x']:.2f}, its centre {e['depth']:.2f} m "
                       f"below the ground there); a tension crack is at the "
                       f"upper end of a slip surface (EM 1110-2-1902 "
                       f"2-2f(3), p. 2-6) and no printed crack closes this "
                       f"one")
        elif not has_crack:
            # The crack is a condition of the analysis, set by the engineer;
            # a circle's shape never switches it on (the step 4 review's
            # F1, the orchestrator's ruling).
            e = at_crest[0]
            refusal = NO_CRACK_ASKED
            discard = (f"a reverse-curvature circle: its centre is "
                       f"{e['depth']:.2f} m below the ground where it ends "
                       f"(x = {e['x']:.2f}), so the arc would turn back "
                       f"over itself, and no tension crack was asked for "
                       f"to close it, so it is not an admissible surface; "
                       f"naming a crack (tension_crack_depth, or "
                       f"tension_crack_depth_basis 'rankine' or 'min_fos') "
                       f"admits such circles whose overhang lies within "
                       f"the crack's depth")
        elif side != upper:
            e = at_crest[0]
            refusal = NO_CRACK_AT_END
            discard = (f"the circle has reverse curvature at its upper end "
                       f"(x = {e['x']:.2f}) and the tension crack was put "
                       f"at the other end (tension_crack_side="
                       f"'{geom.tension_crack_side}'), so no crack closes "
                       f"it")
        else:
            e = at_crest[0]
            side = upper
        overhang = e["depth"]       # discarded or closed, it is located
    elif not has_crack:
        return None
    x_end, x_other = (x_entry, x_exit) if side == ENTRY else (x_exit, x_entry)
    z_g = geom.ground_elevation_at(x_end)
    plan = CrackPlan(side=side, x_end=x_end, z_ground=z_g, basis=basis,
                     depth=depth, overhang=overhang, discard=discard,
                     refusal=refusal, x_other=x_other, slip=slip)
    if discard is not None:
        return plan
    plan.d_max = depth_below_ground(geom, slip, x_entry, x_exit)
    if basis == GIVEN:
        if overhang is not None and depth < overhang - 1e-9:
            plan.discard = _too_deep(plan, depth, "the given crack depth")
            plan.refusal = DEEPER_THAN_CRACK
        elif depth > 0 and plan.d_max <= depth + 1e-9:
            plan.discard = _within(plan, depth, "the given crack depth")
            plan.refusal = WITHIN_CRACK
        return plan
    low = _crest_half_low(slip, x_entry, x_exit, side)
    plan.scan_limit = max(z_g - low, 0.0) if low is not None else 0.0
    if basis == RANKINE:
        _rankine_soil(geom, plan)
        if not plan.rankine_refused and plan.c <= 0.0:
            # no cohesion, no tension: no crack (Eq. C-36 gives 0)
            if overhang is not None:
                plan.discard = _too_deep(
                    plan, 0.0, f"Eq. C-36's crack depth in {plan.soil} "
                    f"(c = 0)")
                plan.refusal = DEEPER_THAN_CRACK
            else:
                plan.basis, plan.depth = GIVEN, 0.0
    return plan


def _too_deep(plan: CrackPlan, depth: float, what: str) -> str:
    return (f"a reverse-curvature circle: its centre is "
            f"{plan.overhang:.2f} m below the ground where the circle ends "
            f"(x = {plan.x_end:.2f}, ground z = {plan.z_ground:.2f}), so "
            f"its overhanging part runs deeper than {what}, {depth:.2f} m; "
            f"a crack is justified only to the depth the soil can be in "
            f"tension (EM 1110-2-1902 C-10c, p. C-36), so the circle is "
            f"discarded")


def _within(plan: CrackPlan, depth: float, what: str) -> str:
    return (f"the surface nowhere passes below {what}, {depth:.2f} m: it "
            f"is at most {plan.d_max:.2f} m below the ground above it, so "
            f"it lies wholly within the cracked depth and is no slip "
            f"surface (EM 1110-2-1902 2-2f(3), p. 2-6, and C-10c, p. C-36: "
            f"the slip surface ends at the bottom of the crack)")


def fixed_geometry(geom, plan: CrackPlan, depth: float):
    """A copy of the geometry with this surface's crack fixed at ``depth``
    on the plan's side, for ``build_slices``. A reverse-curvature circle's
    crack reaches at least to the centre's level (where the arc is cut) and
    always ends the surface."""
    g = copy.copy(geom)
    d = depth if plan.overhang is None else max(depth, plan.overhang)
    g._crack_fixed = {
        "side": plan.side, "depth": d,
        "truncate": (geom.tension_crack_model == TRUNCATION
                     or plan.overhang is not None),
        "water_depth": min(geom.tension_crack_water_depth, d),
    }
    return g


def crack_for_slices(geom, slip, x_entry: float, x_exit: float
                     ) -> Optional[Dict[str, Any]]:
    """The crack ``build_slices`` cuts with: None, or {'side', 'depth',
    'truncate', 'water_depth', 'base'} with ``base`` the elevation of the
    crack's bottom (the ground at the surface's end on that side, less the
    depth). Raises ``CrackDepthNeeded`` where the depth depends on the
    surface's factor of safety and no solve has fixed it, and
    ``ReverseCurvatureDiscarded`` for a circle no crack closes."""
    fixed = getattr(geom, "_crack_fixed", None)
    if fixed is None:
        plan = plan_crack(geom, slip, x_entry, x_exit)
        if plan is None:
            return None
        if plan.discard is not None:
            raise refusal_error(_discarded(geom, plan))
        if plan.basis != GIVEN:
            raise CrackDepthNeeded(
                "The tension crack on this surface depends on the surface's "
                "factor of safety ("
                + ("a reverse-curvature circle closed by a crack of "
                   if plan.overhang is not None else "")
                + {RANKINE: "Rankine's depth with the developed strengths",
                   MIN_FOS: "the depth of minimum factor of safety"}[
                       plan.basis]
                + "): solve it with slope_stability.cracks.solve_with_crack "
                "(analyze_slope and the searches do).")
        if plan.depth <= 0:
            return None
        fixed = fixed_geometry(geom, plan, plan.depth)._crack_fixed
    if fixed["depth"] <= 0:
        return None
    x_end = x_entry if fixed["side"] == ENTRY else x_exit
    out = dict(fixed)
    out["base"] = geom.ground_elevation_at(x_end) - fixed["depth"]
    return out


def _discarded(geom, plan: CrackPlan, depth: float = 0.0,
               reason: Optional[str] = None, refusal: Optional[str] = None,
               **kw) -> CrackOutcome:
    return CrackOutcome(
        depth=depth, basis=plan.basis, side=plan.side,
        model=geom.tension_crack_model, x_end=plan.x_end,
        z_ground=plan.z_ground, reverse_curvature=bool(plan.overhang),
        overhang=plan.overhang, discarded=True,
        reason=(reason or plan.discard or ""),
        refusal=(refusal or plan.refusal), **kw)


def _outcome(geom, plan: CrackPlan, depth: float, basis: str, **kw
             ) -> CrackOutcome:
    d = depth if plan.overhang is None else max(depth, plan.overhang)
    x_c = _crack_x(plan.slip, plan, d)
    return CrackOutcome(
        depth=d, basis=basis, side=plan.side,
        model=(TRUNCATION if plan.overhang is not None
               else geom.tension_crack_model),
        x_end=plan.x_end, z_ground=plan.z_ground,
        x_crack=x_c,
        z_ground_crack=(None if x_c is None
                        else geom.ground_elevation_at(x_c)),
        water_depth=min(geom.tension_crack_water_depth, d),
        reverse_curvature=plan.overhang is not None,
        overhang=plan.overhang,
        fixed=fixed_geometry(geom, plan, depth)._crack_fixed, **kw)


def geometry_of(geom, outcome: Optional[CrackOutcome]):
    """The geometry with the crack of ``outcome`` fixed (``geom`` itself
    without one)."""
    if outcome is None or outcome.fixed is None:
        return geom
    g = copy.copy(geom)
    g._crack_fixed = dict(outcome.fixed)
    return g


def geometry_at_fos(geom, slip, fos: Optional[float] = None):
    """A geometry to cut a solved surface's slices with where only their
    count or their weights are wanted and the solve is not repeated: a
    Rankine crack at the given factor of safety (the full strengths without
    one), a scanned one at its shallowest depth."""
    if not may_need_solution(geom, slip):
        return geom
    plan = plan_crack(geom, slip)
    if plan is None or plan.discard is not None or plan.basis == GIVEN:
        return geom
    d = 0.0
    if plan.basis == RANKINE and not plan.rankine_refused:
        d = min(rankine_crack_depth(plan.c, plan.phi, plan.gamma,
                                    fos if fos and 0 < fos < 999 else 1.0),
                plan.soil_thickness)
    return fixed_geometry(geom, plan, d)


def given_outcome(geom, slip, x_entry: Optional[float] = None,
                  x_exit: Optional[float] = None) -> Optional[CrackOutcome]:
    """The outcome for a crack that needs no solution (a given depth),
    or None (no crack, or the geometry's crack is already fixed)."""
    plan = plan_crack(geom, slip, x_entry, x_exit)
    if plan is None or plan.discard is not None or plan.basis != GIVEN \
            or plan.depth <= 0:
        return None
    return _outcome(geom, plan, plan.depth, GIVEN)


def solve_with_crack(geom, slip, solve: Callable[[Any], Any],
                     fos_of: Callable[[Any], Optional[float]] = lambda r: r,
                     plan: Optional[CrackPlan] = None,
                     fos_for_depth: Optional[float] = None,
                     ) -> Tuple[Any, Optional[CrackOutcome]]:
    """Solve one surface with its crack as printed.

    ``solve(g)`` solves the surface on geometry ``g`` (a copy of ``geom``
    with the crack fixed) and returns anything; ``fos_of`` reads the factor
    of safety from it (None, or not finite, or 999 and above: no solution).
    Returns (the last solution, the crack's ``CrackOutcome``); the outcome
    is None without a crack, and has ``discarded`` set, with the solution
    None, for a reverse-curvature circle the crack does not close.

    ``fos_for_depth``: take Rankine's depth at this factor of safety and
    do not repeat the solution (the yield acceleration, where F = 1).
    """
    if plan is None:
        plan = plan_crack(geom, slip)
    if plan is None:
        return solve(geom), None
    if plan.discard is not None:
        return None, _discarded(geom, plan)

    def run(depth):
        r = solve(fixed_geometry(geom, plan, depth))
        f = fos_of(r)
        ok = (f is not None and isinstance(f, (int, float))
              and math.isfinite(f) and 0.0 < f < 999.0)
        return r, (float(f) if ok else None)

    if plan.basis == GIVEN:
        r, f = run(plan.depth)
        return r, _outcome(geom, plan, plan.depth, GIVEN, fos=f)

    refused = plan.rankine_refused
    if plan.basis == RANKINE and not refused:
        def depth_at(f):
            d = rankine_crack_depth(plan.c, plan.phi, plan.gamma, f)
            return min(d, plan.soil_thickness), d > plan.soil_thickness

        def within(f_at, n_done):
            # a crack holding the whole surface (the step 4 review's D1)
            return None, _discarded(
                geom, plan, depth,
                _within(plan, depth,
                        f"Rankine's crack depth with the developed strengths "
                        f"at F = {f_at:.3f} ({_EM_RANKINE})"),
                refusal=WITHIN_CRACK, fos=f_at, iterations=n_done,
                capped=capped, soil=plan.soil)

        f_k = 1.0 if fos_for_depth is None else fos_for_depth
        depth, capped = depth_at(f_k)
        r = f = None
        n = 0
        if depth >= plan.d_max - 1e-9 and fos_for_depth is None:
            # Eq. C-36 with the full strengths (F = 1, the first solution's
            # depth) holds the whole surface: start instead from the
            # surface's own F without a crack.
            r, f = run(0.0)
            n = 1
            if f is None:
                return r, _outcome(geom, plan, 0.0, RANKINE, fos=None,
                                   iterations=n, converged=False,
                                   capped=False, soil=plan.soil)
            f_k = f
            depth, capped = depth_at(f_k)
        converged = fos_for_depth is not None
        while n < MAX_ITERATIONS:
            if depth >= plan.d_max - 1e-9:
                return within(f_k, n)
            r, f = run(depth)
            n += 1
            if f is None:
                return r, _outcome(geom, plan, depth, RANKINE, fos=None,
                                   iterations=n, converged=False,
                                   capped=capped, soil=plan.soil)
            if fos_for_depth is not None:
                break
            new_depth, new_capped = depth_at(f)
            if abs(new_depth - depth) <= DEPTH_TOL:
                converged = True
                f_k = f
                break
            f_k, depth, capped = f, new_depth, new_capped
        refused = _water_in_depth(geom, plan, depth)
        if not refused:
            if plan.overhang is not None and depth < plan.overhang - DEPTH_TOL:
                return None, _discarded(
                    geom, plan, depth,
                    _too_deep(plan, depth,
                              f"Rankine's crack depth with the developed "
                              f"strengths at F = {f:.3f} ({_EM_RANKINE})"),
                    refusal=DEEPER_THAN_CRACK,
                    fos=f, iterations=n, converged=converged, capped=capped,
                    soil=plan.soil)
            return r, _outcome(geom, plan, depth, RANKINE, fos=f_k,
                               iterations=n, converged=converged,
                               capped=capped, soil=plan.soil)

    # The scan: the depth of minimum factor of safety (EM 2-2f(4), C-10c(1)).
    # Depths are tried from the shallowest down, in equal steps to the
    # lowest point of the surface's upper half, and the first minimum met
    # is taken: past the depth that removes the tension, a deeper crack
    # removes compression and the factor of safety rises (EM 2-2f(4):
    # "tensile but not compressive stresses are eliminated"). A factor of
    # safety still falling at the last depth has no minimum: the crack
    # would take the whole upper half of the surface, and no depth is
    # reported.
    lo = plan.overhang or 0.0
    hi = plan.scan_limit
    table: List[Tuple[float, float]] = []
    kept: Dict[float, Any] = {}
    found = None
    for i in range(SCAN_STEPS + 1):
        if hi <= lo:
            break
        d = lo + (hi - lo) * i / SCAN_STEPS
        if d >= plan.d_max - 1e-9:
            break                   # a crack this deep holds the surface (D1)
        try:
            r, f = run(d)
        except (ValueError, ZeroDivisionError, RuntimeError):
            f = None
        if f is None:
            break                   # no solution this deep: the scan ends
        table.append((d, f))
        kept[d] = r
        if len(table) >= 2 and f > table[-2][1]:
            found = table[-2]
            break
    if found is None:
        why = ("the factor of safety keeps falling as the crack deepens to "
               f"{table[-1][0]:.2f} m, the deepest tried: there is no depth "
               "of minimum factor of safety (" + _EM_SCAN + "); give the "
               "depth" if len(table) >= 2 else
               "the scan for the crack depth of minimum factor of safety "
               "found no solution at two depths or more")
        if refused:
            why += f" (Rankine's depth was not used: {refused})"
        return None, _discarded(geom, plan, table[-1][0] if table else lo,
                                why, refusal=SCAN_NO_MINIMUM, scan=table,
                                rankine_refused=refused)
    d_min, f_min = found
    if plan.overhang is not None and d_min <= lo + 1e-12:
        # F rises with depth from the overhang's own depth down: the depth
        # of minimum F is not below the overhang, so no crack of the
        # printed depth closes the circle.
        return None, _discarded(
            geom, plan, d_min,
            _too_deep(plan, d_min,
                      f"the crack depth of minimum factor of safety "
                      f"({_EM_SCAN}), which is no more than"),
            refusal=DEEPER_THAN_CRACK, scan=table, rankine_refused=refused)
    return kept[d_min], _outcome(geom, plan, d_min, MIN_FOS, fos=f_min,
                                 scan=table, rankine_refused=refused)


# ---------------------------------------------------------------------------
# Surfaces a search's crack rule refuses: counted by reason and located
# ---------------------------------------------------------------------------

#: The surfaces the crack rule refused in the search running in this
#: context, by reason (``REFUSAL_TEXT``): {reason: {'n', 'x_min', 'x_max',
#: 'kept': the first few, located}}.
_DISCARDS: contextvars.ContextVar = contextvars.ContextVar(
    "slope_crack_refusals", default=None)
_KEEP = 5


def _new_entry() -> Dict[str, Any]:
    return {"n": 0, "x_min": math.inf, "x_max": -math.inf, "kept": []}


def count_discard(slip, outcome: CrackOutcome) -> None:
    """Count a surface the crack rule refused, under its own reason."""
    tally = _DISCARDS.get()
    if tally is None:
        return
    e = tally.setdefault(outcome.refusal or DEEPER_THAN_CRACK, _new_entry())
    e["n"] += 1
    e["x_min"] = min(e["x_min"], outcome.x_end)
    e["x_max"] = max(e["x_max"], outcome.x_end)
    if len(e["kept"]) < _KEEP:
        rec: Dict[str, Any] = {"x_end_m": round(outcome.x_end, 3)}
        if getattr(slip, "is_circular", False):
            rec.update(xc=round(slip.xc, 3), yc=round(slip.yc, 3),
                       radius=round(slip.radius, 3))
        if outcome.overhang:
            rec["overhang_depth_m"] = round(outcome.overhang, 3)
        rec["crack_depth_m"] = round(outcome.depth, 3)
        rec["why"] = outcome.reason
        e["kept"].append(rec)


def discards_so_far() -> Dict[str, int]:
    """The surfaces refused so far by the crack rule in the search running
    in this context (the searches run inside it included), by reason."""
    tally = _DISCARDS.get()
    return {k: e["n"] for k, e in (tally or {}).items() if e["n"]}


def refusal_records(tally: Dict[str, Dict[str, Any]]
                    ) -> Dict[str, Dict[str, Any]]:
    """``SearchResult.crack_refused``: for each reason, how many, what it
    means, where their ends are and the first few."""
    return {k: {"n": e["n"], "what": REFUSAL_TEXT.get(k, k),
                "x_end_range_m": [round(e["x_min"], 3),
                                  round(e["x_max"], 3)],
                "first": list(e["kept"])}
            for k, e in tally.items() if e["n"]}


def counts_reverse_discards(fn):
    """Count, by reason, and locate the surfaces the crack rule refuses in
    a search, and put them on the ``SearchResult`` it returns (or on that
    result's ``search``) as ``crack_refused`` and, for the reverse-curvature
    circles, ``n_discarded_reverse_curvature``. A search run inside another
    adds its surfaces to the outer one."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = _DISCARDS.get()
        tally: Dict[str, Dict[str, Any]] = {}
        token = _DISCARDS.set(tally)
        box: Dict[str, Any] = {}
        token_r = _REPORT.set(box)
        try:
            result = fn(*args, **kwargs)
        finally:
            _REPORT.reset(token_r)
            _DISCARDS.reset(token)
            if outer is not None:
                for k, e in tally.items():
                    o = outer.setdefault(k, _new_entry())
                    o["n"] += e["n"]
                    o["x_min"] = min(o["x_min"], e["x_min"])
                    o["x_max"] = max(o["x_max"], e["x_max"])
                    o["kept"] += e["kept"][:_KEEP - len(o["kept"])]
        target = (result if hasattr(result, "crack_refused")
                  else getattr(result, "search", None))
        if target is not None and hasattr(target, "crack_refused"):
            target.n_discarded_reverse_curvature = sum(
                tally[k]["n"] for k in REVERSE_REFUSALS if k in tally)
            target.crack_refused = refusal_records(tally)
            crit = getattr(target, "critical", None)
            if crit is not None and not getattr(crit, "tension_crack", None):
                apply_to_result(crit, box.get("outcome"))
        return result
    return wrapper


#: The crack of the surface a search reports (``note_report``), applied
#: to its result by ``counts_reverse_discards``.
_REPORT: contextvars.ContextVar = contextvars.ContextVar(
    "slope_crack_report", default=None)


def note_report(outcome: Optional[CrackOutcome]) -> None:
    box = _REPORT.get()
    if box is not None:
        box["outcome"] = outcome


def apply_to_result(result, outcome: Optional[CrackOutcome]) -> None:
    """Put the crack a surface was solved with on its result: the depth
    and water depth used, the ``tension_crack`` record and a warning line."""
    if result is None or outcome is None or outcome.depth <= 0:
        return
    result.tension_crack_depth = outcome.depth
    result.tension_crack_water_depth = outcome.water_depth
    result.tension_crack = outcome.to_dict()
    line = outcome.statement()
    if line not in result.warnings:
        result.warnings.append(line)


def _first_given(*hows: str) -> str:
    """'default' only when every input was left at its default."""
    return next((h for h in hows if h != "default"), "default")


def crack_choices(geom, refused: Optional[Dict[str, int]] = None,
                  circles: bool = True) -> List[Dict[str, Any]]:
    """The ``choices`` entries for the crack rule (``geotech_common.choices``):
    the crack's model, side and depth when one is asked for, what is done
    with reverse-curvature circles, and the surfaces the rule refused, by
    reason (``refused``, from ``discards_so_far``)."""
    from geotech_common.choices import choice, how_given
    out = []
    basis = geom.tension_crack_depth_basis
    if geom.tension_crack_depth > 0 or basis != GIVEN:
        out.append(choice(
            "tension crack",
            ("the slip surface ends at the crack"
             if geom.tension_crack_model == TRUNCATION else
             "the cracked soil kept as weight without strength"),
            how_given(geom.tension_crack_model, TRUNCATION,
                      name="tension_crack_model"),
            "the soil upslope of the crack is left out and the slices end "
            "at a vertical boundary there, on which the crack's water acts "
            "(" + _EM_TERMINATE + ")",
            ["the cracked soil kept as driving weight without strength "
             "(tension_crack_model='strength'), the module's model before "
             "GM-140; no printed source",
             "the surface ended at the crack "
             "(tension_crack_model='truncation', the default)"]))
        out.append(choice(
            "tension crack side",
            {CREST: "the end of the surface where the ground is higher",
             ENTRY: "the low-x end", EXIT: "the high-x end"}[
                 geom.tension_crack_side],
            how_given(geom.tension_crack_side, CREST,
                      name="tension_crack_side"),
            "the crack is near the crest of the slope (EM 1110-2-1902 "
            "(2003) 2-2f(3), p. 2-6)",
            ["a given end (tension_crack_side='entry' or 'exit')"]))
        out.append(choice(
            "tension crack depth",
            {GIVEN: f"given, {geom.tension_crack_depth:g} m",
             RANKINE: "Rankine's depth with the developed strengths",
             MIN_FOS: "the depth of minimum factor of safety"}[basis],
            how_given(basis, GIVEN, name="tension_crack_depth_basis"),
            {GIVEN: "the depth the caller gave",
             RANKINE: (_EM_RANKINE + ": d = 2 c_D / (gamma tan(45 - "
                       "phi_D/2)), c_D = c/F, tan(phi_D) = tan(phi)/F, F "
                       "the surface's own, the solution repeated until the "
                       "depth changes by under 1 mm; printed for total "
                       "stresses and one soil, so it stops at the bottom "
                       "of the soil at the crack, and the scan is used "
                       "where that soil is drained with water in the "
                       "cracked depth (EM 2-2f(6), p. 2-9, prints no "
                       "equation there; the module's reading) or has no c "
                       "and phi"),
             MIN_FOS: (_EM_SCAN + f": {SCAN_STEPS + 1} depths in equal "
                       "steps down to the lowest point of the surface's "
                       "upper half")}[basis],
            [f"tension_crack_depth_basis='{b}'" for b in DEPTH_BASES
             if b != basis]))
    refused = dict(refused or {})
    if circles:
        discard = geom.reverse_curvature == "discard"
        crack = geom.tension_crack_depth > 0 or basis != GIVEN
        n_rev = sum(refused.get(k, 0) for k in REVERSE_REFUSALS)
        out.append(choice(
            "reverse-curvature circles",
            ("discarded" if discard else
             "closed by the tension crack asked for; discarded where the "
             "overhang is deeper" if crack else
             "discarded: no tension crack was asked for")
            + f" ({n_rev} discarded)",
            how_given(geom.reverse_curvature, "tension_crack",
                      name="reverse_curvature"),
            "a circle whose centre is below the ground at its upper end "
            "would turn back over itself. The tension crack is a condition "
            "of the analysis, set by the engineer and the same on every "
            "trial surface; a circle's shape never switches it on. With a "
            "crack asked for, every surface ends at the crack's bottom, and "
            "a reverse-curvature circle whose overhang lies within the "
            "crack's depth is closed by that crack; one overhanging deeper "
            "is not a mechanism the method describes (a crack is justified "
            "only to the depth the soil can be in tension, EM 1110-2-1902 "
            "(2003) C-10c, p. C-36; OWN-103). With no crack asked for, such "
            "a circle is not an admissible surface; naming a crack admits "
            "those within its depth. A circle overhanging at its lower end "
            "is discarded: no printed crack is there. The circles discarded "
            "are counted by reason and located in the result "
            "(crack_refused)",
            ["discard every such circle (reverse_curvature='discard')",
             "name a crack (tension_crack_depth, or "
             "tension_crack_depth_basis 'rankine' or 'min_fos') to admit "
             "the circles whose overhang lies within its depth"
             if not crack else
             "no crack (tension_crack_depth 0): every such circle "
             "discarded"]))
    others = {k: n for k, n in refused.items()
              if k not in REVERSE_REFUSALS and n}
    if others:
        out.append(choice(
            "surfaces the crack rule refused",
            "; ".join(f"{n} {REFUSAL_TEXT.get(k, k)}"
                      for k, n in others.items()),
            "module",
            "a surface wholly within the cracked depth has no vertical "
            "boundary at the crack's bottom, so it is no slip surface "
            "(EM 1110-2-1902 2-2f(3), p. 2-6; C-10c, p. C-36); a scan with "
            "no minimum gives no depth (" + _EM_SCAN + "). Counted and "
            "located in the result (crack_refused)",
            ["give the crack's depth (tension_crack_depth)"]))
    return out
