"""
Critical slip surface search algorithms.

Algorithms:
1. Grid search — 2D grid of trial centers (xc, yc), optimizes radius at each.
2. Random noncircular — random polyline generation with Spencer's method.
3. PSO noncircular — Particle Swarm Optimization over control points.
4. Weak-layer biased — SNIFF-inspired search biasing toward weak layers.
5. Entry/exit region — generates trial arcs from entry/exit zones.

References:
    Duncan, Wright & Brandon (2014) — Chapter 14
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    Borselli (2002) — SNIFF Random Search concept
"""

import contextvars
import functools
import math
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any, Callable

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface, GroundCrossingError,
)
from slope_stability.slices import build_slices
from slope_stability.methods import (
    fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos,
)
from slope_stability.gle import janbu_fos
from slope_stability.results import SearchResult, SlopeStabilityResult


_FOS_MAX = 999.9


# Trial surfaces refused because they leave the ground and come back into it
# (GM-122, ``GroundCrossingError``): counted by `_compute_fos` and
# `_check_entry_exit` into the tally of the search running in this context.
_GROUND_REFUSALS: contextvars.ContextVar = contextvars.ContextVar(
    "slope_ground_refusals", default=None)


def _count_ground_refusal() -> None:
    tally = _GROUND_REFUSALS.get()
    if tally is not None:
        tally[0] += 1


def _counts_ground_refusals(fn):
    """Count the trial surfaces a search refuses because they leave the
    ground and come back into it (GM-122) and put the count on the
    ``SearchResult`` it returns (or on that result's ``search``) as
    ``n_rejected_ground_crossings``. A search run inside another adds its
    count to the outer one, so the outermost result counts every refusal,
    its supplementary families' included."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = _GROUND_REFUSALS.get()
        tally = [0]
        token = _GROUND_REFUSALS.set(tally)
        try:
            result = fn(*args, **kwargs)
        finally:
            _GROUND_REFUSALS.reset(token)
            if outer is not None:
                outer[0] += tally[0]
        target = (result if isinstance(result, SearchResult)
                  else getattr(result, "search", None))
        if isinstance(target, SearchResult):
            target.n_rejected_ground_crossings = tally[0]
        return result
    return wrapper


def _ordered_windows(x_entry_range, x_exit_range):
    """Entry / exit windows as the searches read them (GM-124).

    A surface's entry is its low-x end and its exit its high-x end
    (``find_entry_exit``), whichever way the slope faces. A window given
    high bound first is read low bound first, and an entry window lying
    wholly at or right of the exit window is swapped with it: as given, no
    surface could start in one and end in the other, and the search scored
    nothing and said nothing. Returns (x_entry_range, x_exit_range, note);
    ``note`` says what was changed, for the result's warnings, else None.
    """
    def low_first(w):
        return None if w is None else (min(w[0], w[1]), max(w[0], w[1]))

    def text(w):
        return f"x {w[0]:.2f} to {w[1]:.2f}"

    entry, exit_ = low_first(x_entry_range), low_first(x_exit_range)
    notes = []
    for name, given, used in (("entry", x_entry_range, entry),
                              ("exit", x_exit_range, exit_)):
        if given is not None and given[0] > given[1]:
            notes.append(f"The {name} window was given high bound first "
                         f"({given[0]:g}, {given[1]:g}); it was read as "
                         f"{text(used)}.")
    if entry is not None and exit_ is not None and entry[0] >= exit_[1]:
        notes.append(
            f"The entry window ({text(entry)}) lies right of the exit window "
            f"({text(exit_)}), so no surface could start in one and end in "
            f"the other. Entry is the low-x end of a slip surface and exit "
            f"the high-x end, whichever way the slope faces, so the windows "
            f"were swapped: entry {text(exit_)}, exit {text(entry)}.")
        entry, exit_ = exit_, entry
    return entry, exit_, (" ".join(notes) or None)


def _window_span(x_entry_range, x_exit_range) -> Optional[float]:
    """Characteristic span of a surface bracketed by the entry/exit windows.

    The centre-to-centre distance between the entry and exit windows — a real
    critical surface spans roughly this. Used to anchor the noncircular span
    floor (``_noncircular_admissible``) to the actual problem rather than the
    full model width. Returns None if either window is missing.
    """
    if x_entry_range is None or x_exit_range is None:
        return None
    entry_c = 0.5 * (x_entry_range[0] + x_entry_range[1])
    exit_c = 0.5 * (x_exit_range[0] + x_exit_range[1])
    return abs(exit_c - entry_c)


def _new_reject_stats() -> Dict[str, Any]:
    """Fresh per-search rejection tally (see ``_compute_fos``). ``refused``
    keeps the lowest surface whose solution did not hold (`_solution_fails`)
    so the result can say so."""
    return {"geometry": 0, "nonconverged": 0, "jagged": 0, "refused": None}


def _bump(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] += 1


def _rejection_kwargs(stats: Dict[str, int], n_evaluated: int) -> Dict[str, int]:
    """SearchResult counter kwargs + a ``warnings.warn`` when a majority of the
    trial surfaces were rejected (an otherwise-silent under-resolution)."""
    total_rej = stats["geometry"] + stats["nonconverged"] + stats["jagged"]
    if n_evaluated > 0 and total_rej > 0.5 * n_evaluated:
        warnings.warn(
            f"Noncircular search rejected {total_rej}/{n_evaluated} trial "
            f"surfaces ({stats['geometry']} geometry, {stats['nonconverged']} "
            f"non-converged, {stats['jagged']} jagged). The critical surface may "
            f"be under-resolved -- widen the entry/exit windows, raise n_trials, "
            f"or check the geometry.")
    return {
        "n_rejected_geometry": stats["geometry"],
        "n_rejected_nonconverged": stats["nonconverged"],
        "n_rejected_jagged": stats["jagged"],
        "refused_lowest": stats.get("refused"),
    }


def _note_refused(stats, slip, fos: float, reason: str) -> None:
    """Keep the lowest refused surface in the tally (see
    ``SearchResult.refused_lowest``)."""
    if stats is None:
        return
    old = stats.get("refused")
    if old is not None and old["FOS"] <= fos:
        return
    pts = slip.points
    stats["refused"] = {
        "FOS": round(fos, 4), "reason": reason,
        "x_entry": round(pts[0][0], 3), "x_exit": round(pts[-1][0], 3),
        "points": [(round(x, 3), round(z, 3)) for x, z in pts]}


def _noncircular_admissible(slip, slices, geom, n_slices_requested: int,
                            window_span: Optional[float] = None) -> bool:
    """Reject grossly-degenerate noncircular trial surfaces (SS-6, geometry).

    Cheap geometric sanity checks the team-lead's list calls for — they weed
    out sliver / non-spanning surfaces before the (more expensive) rigorous
    solve. The *jagged-but-spanning* zig-zags that used to win searches with a
    spurious low FOS are caught separately and more reliably by REQUIRING the
    rigorous GLE to converge (see ``_rigorous_noncircular_fos``): the GLE
    reports ``converged=False`` on those, whereas valid smooth surfaces
    converge. This geometric check is deliberately lenient so the random /
    weak-layer generators keep exploring.

    * **Too few surviving slices** — a surface that self-clips or barely enters
      the ground leaves a tiny sliver mass (build_slices drops below-model
      slices); not a usable failure surface.
    * **Too short a horizontal span** — the span floor is anchored to the
      caller's entry/exit WINDOW extent (``window_span``: a real surface roughly
      spans the window) when supplied, else to the local slope HEIGHT (0.5*H).
      The old floor of 15% of the FULL model width silently rejected a legitimate
      localized failure in a wide model — a wide right-of-way or a long tailwater
      bench blew the floor up far past a real slide length.
    """
    if len(slices) < max(5, n_slices_requested // 3):
        return False
    span = slices[-1].x_right - slices[0].x_left
    if window_span is not None and window_span > 0:
        min_span = 0.5 * window_span
    else:
        zs = [p[1] for p in geom.surface_points]
        slope_height = (max(zs) - min(zs)) if zs else 0.0
        min_span = 0.5 * slope_height
    if min_span > 0 and span < min_span:
        return False
    return True


# A suspiciously LOW FOS on a jagged (non-concave) noncircular surface is a
# solver artifact, not a real critical surface. We geometry-check only such
# low-FOS surfaces, so the random/weak-layer generators keep exploring the many
# high-FOS jagged trials (which harmlessly lose the search) while a spurious low
# value can never win.
_LOW_FOS_JAGGED_GATE = 1.5
_JAG_MAX_SEGMENT_DEG = 80.0     # near-vertical segment
_JAG_MAX_DROP_DEG = 25.0        # concavity reversal from the running-max angle


def _is_jagged(slip) -> bool:
    """True if the polyline is non-concave / spiky (kinematically inadmissible).

    A real slip surface is concave-up: its segment inclination increases
    roughly monotonically from the driving end to the resisting end. A surface
    with a near-vertical segment, a non-monotonic x, or a segment angle that
    DROPS more than ``_JAG_MAX_DROP_DEG`` below the running maximum (a
    back-and-forth zig-zag / bump) is flagged.
    """
    pts = getattr(slip, "points", None)
    if not pts or len(pts) < 3:
        return False
    max_seg = math.radians(_JAG_MAX_SEGMENT_DEG)
    max_drop = math.radians(_JAG_MAX_DROP_DEG)
    running_max = -math.inf
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dz = pts[i + 1][1] - pts[i][1]
        if dx <= 0:
            return True
        ang = math.atan2(dz, dx)
        if abs(ang) > max_seg or ang < running_max - max_drop:
            return True
        running_max = max(running_max, ang)
    return False


def _noncircular_axis_point(slices, geom) -> Optional[Tuple[float, float]]:
    """A deterministic, robust moment axis for the noncircular convergence gate.

    Midpoint of the surviving span, half a span above the highest ground point —
    the same centroid-above-surface heuristic the validated composite-surface
    path passes explicitly (test_validation uses the parent circle centre). This
    pins the axis for the convergence check instead of leaving it to gle's
    internal least-squares fit, which can land absurdly far away on a
    near-straight surface and spuriously fail convergence. A converged GLE FOS is
    axis-independent (both moment and force equilibrium are enforced), so this
    does not move the reported FOS of valid surfaces.
    """
    if not slices:
        return None
    x_lo = slices[0].x_left
    x_hi = slices[-1].x_right
    span = max(x_hi - x_lo, 1.0)
    z_top = max(p[1] for p in geom.surface_points) if geom is not None else \
        max(s.z_base for s in slices)
    return (0.5 * (x_lo + x_hi), z_top + 0.5 * span)


def _has_reinforcement(geom) -> bool:
    """True when the geometry carries any reinforcement that
    `compute_reinforcement_forces` resolves (nails, anchors, geosynthetics,
    stabilizing piles, or the legacy single horizontal force)."""
    if (getattr(geom, "nails", None) or getattr(geom, "anchors", None)
            or getattr(geom, "geosynthetics", None)
            or getattr(geom, "stabilizing_piles", None)):
        return True
    return ((getattr(geom, "reinforcement_force", 0.0) or 0.0) > 0
            and getattr(geom, "reinforcement_elevation", None) is not None)


def trial_reinforcement(geom, slip):
    """Reinforcement crossings on a trial surface, resolved exactly as
    `analyze_slope` resolves them (`compute_reinforcement_forces` between the
    surface's entry and exit), or None when there are none (GM-53: the
    search used to score every trial surface without its reinforcement)."""
    if not _has_reinforcement(geom):
        return None
    from slope_stability.reinforcement import compute_reinforcement_forces
    x_entry, x_exit = slip.find_entry_exit(geom)
    return compute_reinforcement_forces(geom, slip, x_entry, x_exit) or None


def _reinforced_rigorous_fos(slices, slip, f_interslice: str, tol: float,
                             reinf_forces) -> Optional[float]:
    """Rigorous GLE FOS of a REINFORCED circular trial surface, or None when
    the GLE does not converge. `spencer_fos` / `morgenstern_price_fos` fall
    back to the legacy solver on non-convergence, and that solver ignores
    reinforcement: in a search the unreinforced value of a heavily anchored
    surface then wins (Slide2 #85, Spencer: 0.86 against Bishop 1.26). A
    search rejects such a surface instead."""
    from slope_stability.gle import gle_fos
    try:
        res = gle_fos(slices, slip, f_interslice=f_interslice,
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf_forces)
        if res.converged and 0.05 < res.fos < _FOS_MAX:
            return res.fos
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


def _rigorous_noncircular_fos(slices, slip, f_interslice: str,
                              tol: float, geom=None,
                              reinf_forces=None) -> Optional[float]:
    """Rigorous GLE FOS for a noncircular surface, REQUIRING convergence.

    Returns None when the GLE does not converge, so the caller REJECTS the
    surface instead of silently accepting the legacy Spencer approximation —
    which returns a spurious low FOS (~0.05-0.2) on jagged/degenerate surfaces
    that a search would then wrongly pick as critical. The rigorous GLE reports
    ``converged=False`` on those surfaces at every slice count (verified on the
    ACADS-4 zig-zag: GLE constant 0.15 / half-sine 0.22, both non-converged,
    vs a Fellenius 8.2 — the methods disagree because the surface is
    kinematically inadmissible). Valid smooth surfaces converge and return the
    same value as before.

    The moment axis for the convergence check is tried in two steps: first gle's
    internal least-squares fit (``axis_point=None`` — byte-identical to the
    historical behaviour, so a surface that already converged is unchanged), and
    only if THAT fails to converge, a robust pinned axis
    (``_noncircular_axis_point``, when ``geom`` is supplied). The second pass
    rescues a valid surface whose least-squares centre lands absurdly far away
    (spuriously failing convergence) without ever removing a convergence the
    first pass already found.
    """
    from slope_stability.gle import gle_fos
    axes = [None]
    if geom is not None:
        pinned = _noncircular_axis_point(slices, geom)
        if pinned is not None:
            axes.append(pinned)
    for axis_point in axes:
        try:
            res = gle_fos(slices, slip, f_interslice=f_interslice,
                          tol=min(tol, 1e-4) * 0.1, axis_point=axis_point,
                          reinf_forces=reinf_forces)
            if res.converged and 0.05 < res.fos < _FOS_MAX:
                return res.fos
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    return None


# A rigorous noncircular solution is confirmed when a second moment axis
# gives a converged solution within this fraction of its FOS, and the
# surface cut into one slice more, or twice as many, still has one within
# the second fraction (loose: a jagged random surface's FOS can move by a
# third between 30 and 31 slices and still be an equilibrium solution at
# each; and GLE finds no solution at some slice counts of valid surfaces).
_AXIS_AGREE = 0.02
_SLICE_AGREE = 0.5


def _converged_about(slices, slip, f_interslice: str, tol: float, geom,
                     reinf_forces=None) -> List[float]:
    """FOS of every converged rigorous solution of the slices about three
    moment axes: gle's least-squares fit, the pinned axis
    (`_noncircular_axis_point`) and the pinned axis raised by the span."""
    from slope_stability.gle import gle_fos
    axes = [None]
    pinned = _noncircular_axis_point(slices, geom) if geom is not None \
        else None
    if pinned is not None:
        span = max(slices[-1].x_right - slices[0].x_left, 1.0)
        axes += [pinned, (pinned[0], pinned[1] + span)]
    out = []
    for axis_point in axes:
        try:
            res = gle_fos(slices, slip, f_interslice=f_interslice,
                          tol=min(tol, 1e-4) * 0.1, axis_point=axis_point,
                          reinf_forces=reinf_forces)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if res.converged and 0.05 < res.fos < _FOS_MAX:
            out.append(res.fos)
    return out


def _solution_fails(geom, slip, f_interslice: str, tol: float,
                    n_slices: int, slices, fos: float,
                    reinf_forces=None) -> Optional[str]:
    """Why the rigorous solution of FOS ``fos`` on a noncircular surface
    is not an equilibrium solution of the surface but of one moment axis or
    one slicing; None when it holds.

    Spencer and GLE satisfy force and moment equilibrium, so their solution
    is the same about any moment axis (`_noncircular_axis_point` relies on
    that), and a surface's FOS does not hang on the number of slices. A
    "converged" state that fails either test is a solver artefact, and
    searches had been reporting them as critical: the least-squares axis of
    a surface with a near-vertical segment can make F_m and F_f meet where
    no other axis balances (a 10 m 1:1 cut, weak clay over sandstone: 1.06
    on a surface 13 m deep in the sandstone, no solution about any other
    axis, Fellenius 21.7); the pinned axis can catch a second, spurious root
    (V-043's sand fill: 0.717 at lambda -0.35 at 30 slices, where 29 and 31
    slices give 1.90 and 1.55 at lambda +0.2 about every axis); and a
    surface diving at 75 deg into rock converged about two axes at 30
    slices only (0.73, with Janbu 0.42 and Fellenius 8.5). So: two of the
    three axes agree within ``_AXIS_AGREE``, and with one slice more or
    twice as many some axis still converges, within ``_SLICE_AGREE`` of the
    FOS.
    """
    here = _converged_about(slices, slip, f_interslice, tol, geom,
                            reinf_forces)
    if sum(abs(f - fos) <= _AXIS_AGREE * fos for f in here) < 2:
        return "it solved about one moment axis only"
    for n in (n_slices + 1, 2 * n_slices):
        try:
            more = build_slices(geom, slip, n)
        except ValueError:
            continue
        there = _converged_about(more, slip, f_interslice, tol, geom,
                                 reinf_forces)
        if any(abs(f - fos) <= _SLICE_AGREE * fos for f in there):
            return None
    return (f"it solved at {n_slices} slices only, not at "
            f"{n_slices + 1} or {2 * n_slices}")


def _compute_fos(geom: SlopeGeometry,
                 slip,
                 method: str,
                 n_slices: int,
                 tol: float = 1e-4,
                 window_span: Optional[float] = None,
                 reject_stats: Optional[Dict[str, int]] = None,
                 fos_fn: Optional[Callable] = None,
                 confirm_below: Optional[float] = None) -> float:
    """Safely compute FOS, returning _FOS_MAX on any error.

    ``confirm_below``: a noncircular surface scored by Spencer /
    Morgenstern-Price / GLE at a FOS below this (a search's best so far) is
    kept only if its solution holds about a second moment axis and at
    another slicing (`_solution_fails`); otherwise it is rejected as not
    converged and, if the lowest so refused, kept in
    ``reject_stats['refused']`` so the result says so. Searches
    pass their running best, so the check runs only when a surface would
    become the critical one.

    ``window_span`` anchors the noncircular span floor to the caller's entry/exit
    window (see ``_noncircular_admissible``). ``reject_stats`` (optional) is a
    mutable tally the generators pass so that silent noncircular rejections
    (geometry / non-converged / jagged) are counted rather than lost.

    ``fos_fn`` (optional) substitutes a caller-supplied per-surface FOS
    evaluator ``fos_fn(geom, slip) -> float`` for the built-in method dispatch,
    while KEEPING the same geometry-admissibility, non-convergence, and jagged
    guards. This is how ``search_rapid_drawdown`` reuses every search loop
    (grid / entry-exit / random / DE) with the rapid-drawdown FOS substituted per
    surface instead of an ordinary drained/undrained solve. Default ``None``
    leaves the method dispatch byte-identical.

    Reinforcement crossing the trial surface is resolved and applied exactly
    as ``analyze_slope`` applies it (GM-53; ``trial_reinforcement``). A
    reinforced circle scored by Spencer / M-P / GLE must converge in the
    rigorous GLE (``_reinforced_rigorous_fos``): the legacy fallback ignores
    reinforcement, so the surface is rejected instead.
    """
    try:
        slices = build_slices(geom, slip, n_slices)
        if len(slices) < 3:
            _bump(reject_stats, "geometry")
            return _FOS_MAX
        # Reinforcement on THIS trial surface, as analyze_slope applies it
        # (a caller-supplied fos_fn resolves its own).
        reinf = trial_reinforcement(geom, slip) if fos_fn is None else None

        # Noncircular guard (SS-6): reject kinematically-degenerate trial
        # surfaces and REQUIRE the rigorous GLE to converge. This kills the
        # weak-layer/random-polyline search pathology where a jagged surface
        # makes the rigorous solver diverge, the legacy Spencer fallback
        # returns a spurious low FOS (~0.05-0.2), and the search wrongly picks
        # it as critical. Circular surfaces are untouched (their GLE converges;
        # bishop/grid/entry-exit paths are byte-identical below).
        if not getattr(slip, 'is_circular', True):
            if not _noncircular_admissible(slip, slices, geom, n_slices,
                                           window_span):
                _bump(reject_stats, "geometry")
                return _FOS_MAX
            if fos_fn is not None:
                fos = fos_fn(geom, slip)
            elif method in ("morgenstern_price", "gle"):
                fos = _rigorous_noncircular_fos(slices, slip, "half_sine", tol,
                                                geom, reinf_forces=reinf)
            elif method == "fellenius":
                fos = fellenius_fos(slices, slip, reinf_forces=reinf)
            elif method == "janbu":
                fos, _u, _f0 = janbu_fos(slices, slip,
                                         tol=min(tol, 1e-4) * 0.1,
                                         reinf_forces=reinf)
            else:  # spencer / bishop-fallback / default
                fos = _rigorous_noncircular_fos(slices, slip, "constant", tol,
                                                geom, reinf_forces=reinf)
            if fos is None or fos < 0.05 or math.isnan(fos) or math.isinf(fos):
                _bump(reject_stats, "nonconverged")
                return _FOS_MAX
            if (confirm_below is not None and fos < confirm_below
                    and fos_fn is None
                    and method not in ("fellenius", "janbu")):
                why = _solution_fails(
                    geom, slip,
                    "half_sine" if method in ("morgenstern_price", "gle")
                    else "constant", tol, n_slices, slices, fos, reinf)
                if why:
                    _bump(reject_stats, "nonconverged")
                    _note_refused(reject_stats, slip, fos, why)
                    return _FOS_MAX
            # Spurious-low-FOS guard: a low FOS on a jagged surface is a solver
            # artifact (SS-6), not a real critical surface — reject it.
            if fos < _LOW_FOS_JAGGED_GATE and _is_jagged(slip):
                _bump(reject_stats, "jagged")
                return _FOS_MAX
            return fos

        # A reverse-curvature circle (its centre below the ground where it
        # comes out) is closed by a dry crack unless the geometry asks for
        # such circles to be discarded (Slide2's two choices; GM-59).
        if (getattr(geom, "reverse_curvature", "tension_crack") == "discard"
                and slip.reverse_curvature_crack(geom) is not None):
            return _FOS_MAX

        if fos_fn is not None:
            fos = fos_fn(geom, slip)
        elif method == "fellenius":
            fos = fellenius_fos(slices, slip, reinf_forces=reinf)
        elif method == "bishop":
            fos = bishop_fos(slices, slip, tol=tol, reinf_forces=reinf)
        elif method == "spencer":
            if reinf:
                fos = _reinforced_rigorous_fos(slices, slip, "constant", tol,
                                               reinf)
            else:
                fos, _ = spencer_fos(slices, slip, tol=tol)
        elif method in ("morgenstern_price", "gle"):
            if reinf:
                fos = _reinforced_rigorous_fos(slices, slip, "half_sine", tol,
                                               reinf)
            else:
                fos, _ = morgenstern_price_fos(slices, slip, tol=tol)
        elif method == "janbu":
            fos, _, _ = janbu_fos(slices, slip, tol=min(tol, 1e-4) * 0.1,
                                  reinf_forces=reinf)
        else:
            fos = bishop_fos(slices, slip, tol=tol, reinf_forces=reinf)

        # Guard against non-convergence producing negative or absurd values
        # FOS < 0.05 is physically meaningless and indicates numerical issues
        if fos is None or fos < 0.05 or math.isnan(fos) or math.isinf(fos):
            return _FOS_MAX
        return fos
    except GroundCrossingError:
        # not one sliding mass (GM-122): refused, and counted
        _count_ground_refusal()
        return _FOS_MAX
    except (ValueError, ZeroDivisionError, RuntimeError):
        return _FOS_MAX


def _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
    """Check if slip surface entry/exit are within allowed x-ranges.

    Returns True if acceptable, False if should be rejected.
    """
    if x_entry_range is None and x_exit_range is None:
        return True
    try:
        x_entry, x_exit = slip.find_entry_exit(geom)
    except GroundCrossingError:
        _count_ground_refusal()        # not one sliding mass (GM-122)
        return False
    except ValueError:
        return False

    if x_entry_range is not None:
        if x_entry < x_entry_range[0] or x_entry > x_entry_range[1]:
            return False
    if x_exit_range is not None:
        if x_exit < x_exit_range[0] or x_exit > x_exit_range[1]:
            return False
    return True


def optimize_radius(geom: SlopeGeometry,
                    xc: float, yc: float,
                    method: str = "bishop",
                    n_slices: int = 30,
                    r_min: Optional[float] = None,
                    r_max: Optional[float] = None,
                    n_radii: int = 20,
                    tol: float = 1e-4,
                    x_entry_range: Optional[Tuple[float, float]] = None,
                    x_exit_range: Optional[Tuple[float, float]] = None,
                    fos_fn: Optional[Callable] = None,
                    admissible: Optional[Callable] = None,
                    ) -> Tuple[float, float]:
    """Find the radius that minimizes FOS for a given circle center.

    Uses a coarse scan followed by golden-section refinement.
    ``admissible(slip)`` (optional) rejects a trial circle before it is
    scored, as the entry/exit windows do.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    xc, yc : float
        Circle center coordinates.
    method : str
        'fellenius', 'bishop', or 'spencer'. Default 'bishop'.
    n_slices : int
        Number of slices.
    r_min, r_max : float, optional
        Radius search bounds. Auto-computed if None.
    n_radii : int
        Number of trial radii in coarse scan.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range. Surfaces outside are rejected.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range. Surfaces outside are rejected.

    Returns
    -------
    (R_opt, FOS_min) : tuple
        Optimal radius and minimum FOS.
    """
    # Auto-compute radius bounds
    if r_min is None or r_max is None:
        zs = [p[1] for p in geom.surface_points]
        z_max_ground = max(zs)

        dist_to_ground = yc - z_max_ground
        if dist_to_ground <= 0:
            dist_to_ground = 0.5
        if r_min is None:
            r_min = max(dist_to_ground + 0.5, 1.0)

        min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
        max_dist = math.sqrt((xc - geom.surface_points[0][0])**2 +
                             (yc - min_layer_bot)**2)
        if r_max is None:
            r_max = max(max_dist, r_min + 5.0)

    if r_min >= r_max:
        r_max = r_min + 5.0

    # Coarse scan
    best_r = r_min
    best_fos = _FOS_MAX

    for i in range(n_radii):
        r = r_min + (r_max - r_min) * i / max(n_radii - 1, 1)
        slip = CircularSlipSurface(xc, yc, r)

        # Entry/exit range check
        if not _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
            continue
        if admissible is not None and not admissible(slip):
            continue

        fos = _compute_fos(geom, slip, method, n_slices, tol=tol, fos_fn=fos_fn)
        if fos < best_fos:
            best_fos = fos
            best_r = r

    # Golden-section refinement around best_r. Each step's new interior
    # point is the previous step's other one (golden ** 2 = 1 - golden), so
    # its FOS is taken from `scored` instead of being solved again (GM-87:
    # the same brackets and result, about 14 fewer solves per centre).
    golden = (math.sqrt(5) - 1) / 2
    margin = (r_max - r_min) / n_radii
    a = max(r_min, best_r - 2 * margin)
    b = min(r_max, best_r + 2 * margin)
    scored = {}

    def fos_at(r):
        key = round(r, 9)
        if key not in scored:
            slip = CircularSlipSurface(xc, yc, r)
            ok = _check_entry_exit(geom, slip, x_entry_range, x_exit_range)
            if ok and admissible is not None:
                ok = admissible(slip)
            scored[key] = _compute_fos(geom, slip, method, n_slices, tol=tol,
                                       fos_fn=fos_fn) if ok else _FOS_MAX
        return scored[key]

    for _ in range(20):
        if b - a < 0.01:
            break
        r1 = b - golden * (b - a)
        r2 = a + golden * (b - a)
        f1 = fos_at(r1)
        f2 = fos_at(r2)

        if f1 < f2:
            b = r2
            if f1 < best_fos:
                best_fos = f1
                best_r = r1
        else:
            a = r1
            if f2 < best_fos:
                best_fos = f2
                best_r = r2

    return (best_r, best_fos)


# ── Local refinement of the centre grid (GM-87) ──────────────────────────
#
# A grid of centres finds the critical circle only to within its spacing.
# The automatic grid spans the whole profile, so on a long profile it is
# coarse: Duncan Ex 1 (2:1 slope in clay, firm base H below the toe) drawn
# with 4 H of level ground each side has its 10 x 10 centres 1.1 H apart in
# x, and the best of them gave 0.739 where Taylor's critical circle (FHWA
# NHI-06-088 Fig. 6-15, after Taylor 1948), whose centre falls between two
# of them, gives 0.705. The radius was never the problem: at each centre
# `optimize_radius` already finds the circle tangent to the firm base. So
# the grid's lowest centres are refined by a compass (pattern) search on the
# centre: from the grid point, the four centres half a grid step away in x
# and in y are scored (each with its own optimised radius, exactly as on the
# grid); the search moves to any lower one, and halves the step when none
# is lower, down to a sixteenth of the grid spacing. It stays inside the
# grid's own centre ranges, so caller-given ranges are honoured.
_REFINE_LEVELS = 4            # steps of 1/2, 1/4, 1/8 and 1/16 of the grid spacing
_REFINE_STARTS = 3            # grid minima refined, at most ...
_REFINE_WITHIN = 0.10         # ... those within 10 % of the lowest grid FOS
_REFINE_MAX_CENTRES = 40      # centres scored from one start, at most
# Spencer / Morgenstern-Price / GLE walk with Bishop (the same critical
# circles to about 1 % and some 30 times cheaper, as the toe-circle family
# does) and re-optimise the radius at each walk's end with the method
# itself; the method's own FOS decides. Not where supports are modelled:
# Bishop counts a support only through its moment about the centre (GM-59),
# and the two methods' critical circles part (Slide2 #85, anchored: the
# Bishop walk found nothing below Spencer's grid minimum 1.581, Spencer's
# own walk 1.574), so a reinforced model walks with the method itself.
_REFINE_SCREENED = ("spencer", "morgenstern_price", "gle")
_REFINE_SCREEN = True
# A caller-supplied per-surface solve (rapid drawdown: two or three GLE
# solves a surface) refines its lowest grid centre only, three levels, at
# most 12 centres.
_REFINE_FOS_FN = (1, 3, 12)          # (starts, levels, centres per start)


def _dips_below_base(geom: SlopeGeometry, slip, n_slices: int) -> bool:
    """True when a slice of the trial surface would have its base below the
    deepest soil layer.

    `build_slices` refuses a surface with such a slice inside the slip mass,
    but drops up to ``max(2, 3 % of n_slices)`` of them at its ends as
    tangential slivers (SS-5). Where the toe sits on the base (Slide2 #85) a
    circle dipping just under the base then sheds its toe slices and scores
    low: #85 without its support, Spencer 0.867 on a circle refused at 90
    slices, against 0.883 for the base-tangent toe circle. The refinement
    walk keeps to circles that at most touch the base, as the toe-circle
    family does.
    """
    if getattr(slip, "is_circular", False):
        # No layer bottom anywhere is higher than this, so a circle whose
        # lowest point is not below it cannot dip under the base.
        top_of_base = min(
            max(z for _, z in L.bottom_boundary_points)
            if L.bottom_boundary_points else L.bottom_elevation
            for L in geom.soil_layers)
        if slip.yc - slip.radius >= top_of_base:
            return False
    try:
        x_en, x_ex = slip.find_entry_exit(geom)
    except ValueError:
        return False                  # _compute_fos rejects it anyway
    dx = (x_ex - x_en) / n_slices
    for i in range(n_slices):
        x = x_en + (i + 0.5) * dx
        z = slip.slip_elevation_at(x)
        if z is None or geom.ground_elevation_at(x) - z <= 0:
            continue
        if (geom.layer_at_point(x, z) is None
                and z < min(L.bottom_at(x) for L in geom.soil_layers)):
            return True
    return False


def _grid_minima(fos, nx: int, ny: int) -> List[Tuple[float, int, int]]:
    """(FOS, ix, iy) of the grid's local minima, lowest first: grid points
    with no lower point among the eight around them, a minimum next to a
    lower one being the same valley and left out."""
    found = []
    for ix in range(nx):
        for iy in range(ny):
            f = fos[ix][iy]
            if f >= _FOS_MAX:
                continue
            if all(f <= fos[jx][jy]
                   for jx in range(max(ix - 1, 0), min(ix + 2, nx))
                   for jy in range(max(iy - 1, 0), min(iy + 2, ny))):
                found.append((f, ix, iy))
    found.sort()
    out = []
    for f, ix, iy in found:
        if not any(abs(ix - jx) <= 1 and abs(iy - jy) <= 1
                   for _, jx, jy in out):
            out.append((f, ix, iy))
    return out


def _refine_centre(score, start, steps, bounds,
                   max_centres: int = _REFINE_MAX_CENTRES,
                   levels: int = _REFINE_LEVELS):
    """Compass search on the circle centre from ``start`` = (FOS, xc, yc, R).

    ``score(xc, yc)`` gives (R, FOS) of the centre's best circle. The four
    centres ``steps`` = (hx, hy) away are tried, the last direction that
    lowered the FOS first; the search moves to the first lower one and
    halves the steps when none is lower (``levels`` levels). Centres are
    kept inside ``bounds`` = ((x_lo, x_hi), (y_lo, y_hi)). Returns the
    lowest (FOS, xc, yc, R) found.
    """
    (x_lo, x_hi), (y_lo, y_hi) = bounds
    hx, hy = steps
    best = start
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    n = 0
    for _ in range(levels):
        moved = True
        while moved:
            moved = False
            for k, (dx, dy) in enumerate(dirs):
                xc = min(max(best[1] + dx * hx, x_lo), x_hi)
                yc = min(max(best[2] + dy * hy, y_lo), y_hi)
                if xc == best[1] and yc == best[2]:
                    continue
                if n >= max_centres:
                    return best
                n += 1
                r, f = score(xc, yc)
                if f < best[0]:
                    best = (f, xc, yc, r)
                    dirs.insert(0, dirs.pop(k))
                    moved = True
                    break
        hx *= 0.5
        hy *= 0.5
    return best


@_counts_ground_refusals
def grid_search(geom: SlopeGeometry,
                x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                nx: int = 10,
                ny: int = 10,
                method: str = "bishop",
                n_slices: int = 30,
                tol: float = 1e-4,
                x_entry_range: Optional[Tuple[float, float]] = None,
                x_exit_range: Optional[Tuple[float, float]] = None,
                fos_fn: Optional[Callable] = None,
                refine: bool = False,
                ) -> SearchResult:
    """Grid search for the critical circular slip surface.

    With ``refine`` the grid's lowest centres are then refined by a local
    compass search on the centre (GM-87; see ``_refine_centre``), inside the
    same centre ranges and entry/exit windows, on circles that at most touch
    the bottom of the deepest layer (``_dips_below_base``); the centres it
    scores are added to ``n_surfaces_evaluated``, and those scored with the
    method itself to ``grid_fos``.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float)
        (x_min, x_max) for circle center search.
    y_range : (float, float)
        (y_min, y_max) for circle center search.
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range.
    refine : bool
        Refine the lowest grid centres locally (GM-87). Default False here;
        ``search_critical_surface`` turns it on.

    Returns
    -------
    SearchResult
    """
    x_entry_range, x_exit_range, window_note = _ordered_windows(
        x_entry_range, x_exit_range)
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    lattice = [[(_FOS_MAX, 0.0, 0.0, 0.0)] * ny for _ in range(nx)]

    def score(xc, yc):
        nonlocal n_evaluated
        r_opt, fos_opt = optimize_radius(
            geom, xc, yc, method=method, n_slices=n_slices, tol=tol,
            x_entry_range=x_entry_range, x_exit_range=x_exit_range,
            fos_fn=fos_fn,
        )
        n_evaluated += 1
        grid_fos.append({
            "xc": round(xc, 2),
            "yc": round(yc, 2),
            "R": round(r_opt, 2),
            "FOS": round(fos_opt, 4),
        })
        return r_opt, fos_opt

    for ix in range(nx):
        xc = x_lo + (x_hi - x_lo) * ix / max(nx - 1, 1) if nx > 1 else (x_lo + x_hi) / 2
        for iy in range(ny):
            yc = y_lo + (y_hi - y_lo) * iy / max(ny - 1, 1) if ny > 1 else (y_lo + y_hi) / 2

            r_opt, fos_opt = score(xc, yc)
            lattice[ix][iy] = (fos_opt, xc, yc, r_opt)

            if fos_opt < best_fos:
                best_fos = fos_opt
                best_xc = xc
                best_yc = yc
                best_r = r_opt

    # Local refinement of the lowest grid centres (GM-87)
    if refine and best_fos < _FOS_MAX:
        steps = ((x_hi - x_lo) / max(nx - 1, 1) / 2.0,
                 (y_hi - y_lo) / max(ny - 1, 1) / 2.0)
        bounds = ((min(x_lo, x_hi), max(x_lo, x_hi)),
                  (min(y_lo, y_hi), max(y_lo, y_hi)))
        screen = (_REFINE_SCREEN and fos_fn is None
                  and method in _REFINE_SCREENED
                  and not _has_reinforcement(geom))
        if fos_fn is None:
            n_starts, levels, max_centres = (_REFINE_STARTS, _REFINE_LEVELS,
                                             _REFINE_MAX_CENTRES)
        else:
            n_starts, levels, max_centres = _REFINE_FOS_FN

        def admissible(slip):
            return not _dips_below_base(geom, slip, n_slices)

        def strict(xc, yc, m):
            """(R, FOS) of the centre's best circle that at most touches
            the base, scored with method ``m``; the method's own scores
            go into grid_fos."""
            nonlocal n_evaluated
            n_evaluated += 1
            r_opt, fos_opt = optimize_radius(
                geom, xc, yc, method=m, n_slices=n_slices, tol=tol,
                x_entry_range=x_entry_range, x_exit_range=x_exit_range,
                fos_fn=fos_fn, admissible=admissible)
            if m == method:
                grid_fos.append({"xc": round(xc, 2), "yc": round(yc, 2),
                                 "R": round(r_opt, 2),
                                 "FOS": round(fos_opt, 4)})
            return r_opt, fos_opt

        walk_method = "bishop" if screen else method
        seen = {}

        def walk_score(xc, yc):
            key = (round(xc, 6), round(yc, 6))
            if key not in seen:
                seen[key] = strict(xc, yc, walk_method)
            return seen[key]

        minima = _grid_minima([[p[0] for p in col] for col in lattice],
                              nx, ny)
        grid_best = best_fos
        for f0, ix, iy in minima[:n_starts]:
            if f0 > (1.0 + _REFINE_WITHIN) * grid_best:
                break
            _, x0, y0, _ = lattice[ix][iy]
            r0, fs = walk_score(x0, y0)
            f, xc, yc, r = _refine_centre(walk_score, (fs, x0, y0, r0), steps,
                                          bounds, max_centres=max_centres,
                                          levels=levels)
            if screen:
                if (xc, yc) == (x0, y0):
                    continue
                r, f = strict(xc, yc, method)
            if f < best_fos:
                best_fos, best_xc, best_yc, best_r = f, xc, yc, r

    # Build result for the critical surface
    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        warnings=[window_note] if window_note else [],
    )


@_counts_ground_refusals
def search_noncircular(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    method: str = "spencer",
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Search for the critical noncircular slip surface using random polylines.

    Generates random polyline slip surfaces within the specified entry/exit
    ranges and evaluates Spencer's FOS for each.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range (where slip enters ground).
    x_exit_range : (float, float)
        Allowed exit x-coordinate range (where slip exits ground).
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis. Default 30.
    tol : float
        Convergence tolerance. Default 1e-4.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult
    """
    if seed is not None:
        random.seed(seed)

    x_lo_entry, x_hi_entry = x_entry_range
    x_lo_exit, x_hi_exit = x_exit_range
    window_span = _window_span(x_entry_range, x_exit_range)

    # Get ground surface elevation range for depth generation
    z_min_ground = min(z for _, z in geom.surface_points)
    z_max_ground = max(z for _, z in geom.surface_points)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []
    trial_surfaces = []
    reject_stats = _new_reject_stats()

    for _ in range(n_trials):
        # Random entry and exit x-coordinates
        x_entry = random.uniform(x_lo_entry, x_hi_entry)
        x_exit = random.uniform(x_lo_exit, x_hi_exit)

        if x_exit <= x_entry + 1.0:
            continue

        # Ground elevation at entry and exit
        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        # Generate intermediate points
        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            # Random depth below ground surface (but above lowest layer)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth = random.uniform(0.5, max_depth * 0.9)
            z_pt = z_ground - depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                               window_span=window_span,
                               reject_stats=reject_stats, fos_fn=fos_fn,
                               confirm_below=best_fos)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })
            if fos < 900:
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in points],
                })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    # Build result
    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.replace("_", "-").title(),
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


# ── PSO Noncircular Search ────────────────────────────────────────────────

@_counts_ground_refusals
def search_pso(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_particles: int = 30,
    n_iterations: int = 50,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> SearchResult:
    """Particle Swarm Optimization for critical noncircular slip surface.

    Each particle represents a polyline slip surface defined by control
    point coordinates. The swarm evolves toward the minimum FOS surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_particles : int
        Swarm size. Default 30.
    n_iterations : int
        Maximum iterations. Default 50.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.
    w : float
        Inertia weight. Default 0.7.
    c1 : float
        Cognitive (personal best) coefficient. Default 1.5.
    c2 : float
        Social (global best) coefficient. Default 1.5.

    Returns
    -------
    SearchResult

    References
    ----------
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    n_interior = max(n_points - 2, 1)
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    # Particle = [x_entry, x_exit, depth_frac_1, ..., depth_frac_n_interior]
    dim = 2 + n_interior

    def _random_particle():
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        depths = [random.uniform(0.05, 0.95) for _ in range(n_interior)]
        return [x_entry, x_exit] + depths

    def _particle_to_slip(pos):
        x_entry = max(x_entry_range[0], min(x_entry_range[1], pos[0]))
        x_exit = max(x_exit_range[0], min(x_exit_range[1], pos[1]))
        if x_exit <= x_entry + 1.0:
            return None

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth_frac = max(0.05, min(0.95, pos[2 + j]))
            z_pt = z_ground - depth_frac * max_depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))
        try:
            return PolylineSlipSurface(points=points)
        except (ValueError, IndexError):
            return None

    best_seen = [_FOS_MAX]

    def _evaluate(pos):
        slip = _particle_to_slip(pos)
        if slip is None:
            return _FOS_MAX
        fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol,
                           window_span=window_span, reject_stats=reject_stats,
                           confirm_below=best_seen[0])
        best_seen[0] = min(best_seen[0], fos)
        return fos

    # Initialize swarm
    positions = [_random_particle() for _ in range(n_particles)]
    velocities = [[random.uniform(-0.3, 0.3) for _ in range(dim)]
                  for _ in range(n_particles)]
    fitness = [_evaluate(p) for p in positions]

    pbest_pos = [list(p) for p in positions]
    pbest_fit = list(fitness)

    gbest_idx = min(range(n_particles), key=lambda i: fitness[i])
    gbest_pos = list(positions[gbest_idx])
    gbest_fit = fitness[gbest_idx]

    n_evaluated = n_particles
    grid_fos = []
    trial_surfaces = []

    for i in range(n_particles):
        slip = _particle_to_slip(positions[i])
        if slip and fitness[i] < _FOS_MAX:
            mid = slip.points[len(slip.points) // 2]
            grid_fos.append({
                "xc": round(mid[0], 2),
                "yc": round(mid[1], 2),
                "R": 0.0,
                "FOS": round(fitness[i], 4),
            })
            trial_surfaces.append({
                "FOS": round(fitness[i], 4),
                "points": [(round(px, 3), round(pz, 3))
                           for px, pz in slip.points],
            })

    for iteration in range(n_iterations):
        w_iter = w - (w - 0.4) * iteration / max(n_iterations - 1, 1)

        for i in range(n_particles):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                velocities[i][d] = (
                    w_iter * velocities[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - positions[i][d])
                    + c2 * r2 * (gbest_pos[d] - positions[i][d])
                )
                if d < 2:
                    velocities[i][d] = max(-5.0, min(5.0, velocities[i][d]))
                else:
                    velocities[i][d] = max(-0.3, min(0.3, velocities[i][d]))
                positions[i][d] += velocities[i][d]

            positions[i][0] = max(x_entry_range[0], min(x_entry_range[1], positions[i][0]))
            positions[i][1] = max(x_exit_range[0], min(x_exit_range[1], positions[i][1]))
            for d in range(2, dim):
                positions[i][d] = max(0.05, min(0.95, positions[i][d]))

            fos = _evaluate(positions[i])
            n_evaluated += 1
            fitness[i] = fos

            slip = _particle_to_slip(positions[i])
            if slip and fos < _FOS_MAX:
                mid = slip.points[len(slip.points) // 2]
                grid_fos.append({
                    "xc": round(mid[0], 2),
                    "yc": round(mid[1], 2),
                    "R": 0.0,
                    "FOS": round(fos, 4),
                })
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in slip.points],
                })

            if fos < pbest_fit[i]:
                pbest_fit[i] = fos
                pbest_pos[i] = list(positions[i])

            if fos < gbest_fit:
                gbest_fit = fos
                gbest_pos = list(positions[i])

    critical = None
    if gbest_fit < _FOS_MAX:
        best_slip = _particle_to_slip(gbest_pos)
        if best_slip is not None:
            try:
                x_entry, x_exit = best_slip.find_entry_exit(geom)
                slices = build_slices(geom, best_slip, n_slices)
                critical = SlopeStabilityResult(
                    FOS=gbest_fit,
                    method="Spencer",
                    xc=0.0,
                    yc=0.0,
                    radius=0.0,
                    x_entry=x_entry,
                    x_exit=x_exit,
                    n_slices=len(slices),
                    has_seismic=geom.kh > 0,
                    kh=geom.kh,
                    slip_points=list(best_slip.points),
                )
            except ValueError:
                pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


# ── Weak-Layer Biased Search (SNIFF-inspired) ────────────────────────────

@_counts_ground_refusals
def search_weak_layer_biased(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> SearchResult:
    """Weak-layer biased random search inspired by SSAP's SNIFF algorithm.

    Biases random surface generation toward passing through soil layers
    with the lowest shear strength. Surfaces are weighted to preferentially
    route through weak zones.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices. Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult

    References
    ----------
    Borselli (2002) — SNIFF Random Search concept (SSAP2010)
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    # Compute weakness score for each layer: inverse of shear strength
    layer_weakness = []
    for layer in geom.soil_layers:
        c, phi = layer.shear_strength_params
        strength = c + 50.0 * math.tan(math.radians(phi))
        if strength <= 0:
            strength = 0.1
        layer_weakness.append(1.0 / strength)

    total_weakness = sum(layer_weakness)
    if total_weakness > 0:
        layer_weights = [lw / total_weakness for lw in layer_weakness]
    else:
        layer_weights = [1.0 / len(geom.soil_layers)] * len(geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []
    trial_surfaces = []

    for _ in range(n_trials):
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        if x_exit <= x_entry + 1.0:
            continue

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        target_idx = _weighted_choice(layer_weights)
        target_layer = geom.soil_layers[target_idx]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)

            if random.random() < 0.7:
                layer_mid = (target_layer.top_elevation + target_layer.bottom_elevation) / 2.0
                thickness = target_layer.top_elevation - target_layer.bottom_elevation
                noise = random.gauss(0, max(thickness / 4.0, 0.1))
                z_pt = layer_mid + noise
                z_pt = max(min_layer_bot, min(z_ground - 0.3, z_pt))
            else:
                max_depth = z_ground - min_layer_bot
                if max_depth <= 0:
                    max_depth = 1.0
                depth = random.uniform(0.5, max_depth * 0.9)
                z_pt = z_ground - depth

            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol,
                               window_span=window_span,
                               reject_stats=reject_stats,
                               confirm_below=best_fos)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })
            if fos < 900:
                trial_surfaces.append({
                    "FOS": round(fos, 4),
                    "points": [(round(px, 3), round(pz, 3))
                               for px, pz in points],
                })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method="Spencer",
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        trial_surfaces=trial_surfaces,
        **_rejection_kwargs(reject_stats, n_evaluated),
    )


def _weighted_choice(weights):
    """Weighted random selection, returns index."""
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1


# ── Entry/Exit Region Search ─────────────────────────────────────────────

@_counts_ground_refusals
def search_entry_exit(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_entry: int = 10,
    n_exit: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Entry/exit region search for circular slip surfaces.

    Instead of searching on a (xc, yc) grid, generates trial arcs
    connecting entry and exit points on the ground surface. For each
    entry/exit pair, finds the optimal circle passing through both points.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        x-range of the surface's entry, its low-x end: the end with the
        smaller x, whichever way the slope faces (the toe side of a slope
        that rises to the right, the crest side of one that falls to it).
    x_exit_range : (float, float)
        x-range of the surface's exit, its high-x end. Windows given the
        other way round are swapped and ``warnings`` says so (GM-124).
    n_entry : int
        Number of trial entry points. Default 10.
    n_exit : int
        Number of trial exit points. Default 10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.

    Returns
    -------
    SearchResult
    """
    x_entry_range, x_exit_range, window_note = _ordered_windows(
        x_entry_range, x_exit_range)
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    entry_xs = [x_entry_range[0] + (x_entry_range[1] - x_entry_range[0]) * i / max(n_entry - 1, 1)
                for i in range(n_entry)] if n_entry > 1 else [(x_entry_range[0] + x_entry_range[1]) / 2]
    exit_xs = [x_exit_range[0] + (x_exit_range[1] - x_exit_range[0]) * i / max(n_exit - 1, 1)
               for i in range(n_exit)] if n_exit > 1 else [(x_exit_range[0] + x_exit_range[1]) / 2]

    for x_en in entry_xs:
        z_en = geom.ground_elevation_at(x_en)
        for x_ex in exit_xs:
            if x_ex <= x_en + 1.0:
                continue
            z_ex = geom.ground_elevation_at(x_ex)

            # Circle center lies on perpendicular bisector of chord
            mid_x = (x_en + x_ex) / 2.0
            mid_z = (z_en + z_ex) / 2.0
            dx = x_ex - x_en
            dz = z_ex - z_en
            chord_len = math.sqrt(dx**2 + dz**2)
            if chord_len < 0.1:
                continue

            perp_x = -dz / chord_len
            perp_z = dx / chord_len
            half_chord = chord_len / 2.0

            t_min = 0.5
            t_max = 3.0 * chord_len
            n_t = 15

            local_best_fos = _FOS_MAX
            local_best_t = t_min

            for it in range(n_t):
                t = t_min + (t_max - t_min) * it / max(n_t - 1, 1)
                xc = mid_x + t * perp_x
                yc = mid_z + t * perp_z
                r = math.sqrt(half_chord**2 + t**2)

                if yc < mid_z:
                    continue

                slip = CircularSlipSurface(xc, yc, r)
                fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                                   fos_fn=fos_fn)
                n_evaluated += 1

                if fos < local_best_fos:
                    local_best_fos = fos
                    local_best_t = t

            # Golden-section refinement
            margin = (t_max - t_min) / n_t
            a = max(t_min, local_best_t - 2 * margin)
            b = min(t_max, local_best_t + 2 * margin)
            golden = (math.sqrt(5) - 1) / 2

            for _ in range(15):
                if b - a < 0.05:
                    break
                t1 = b - golden * (b - a)
                t2 = a + golden * (b - a)

                xc1 = mid_x + t1 * perp_x
                yc1 = mid_z + t1 * perp_z
                r1 = math.sqrt(half_chord**2 + t1**2)

                xc2 = mid_x + t2 * perp_x
                yc2 = mid_z + t2 * perp_z
                r2 = math.sqrt(half_chord**2 + t2**2)

                f1 = _compute_fos(geom, CircularSlipSurface(xc1, yc1, r1),
                                  method, n_slices, tol=tol, fos_fn=fos_fn)
                f2 = _compute_fos(geom, CircularSlipSurface(xc2, yc2, r2),
                                  method, n_slices, tol=tol, fos_fn=fos_fn)
                n_evaluated += 2

                if f1 < f2:
                    b = t2
                    if f1 < local_best_fos:
                        local_best_fos = f1
                        local_best_t = t1
                else:
                    a = t1
                    if f2 < local_best_fos:
                        local_best_fos = f2
                        local_best_t = t2

            t_final = local_best_t
            xc_f = mid_x + t_final * perp_x
            yc_f = mid_z + t_final * perp_z
            r_f = math.sqrt(half_chord**2 + t_final**2)

            grid_fos.append({
                "xc": round(xc_f, 2),
                "yc": round(yc_f, 2),
                "R": round(r_f, 2),
                "FOS": round(local_best_fos, 4),
            })

            if local_best_fos < best_fos:
                best_fos = local_best_fos
                best_xc = xc_f
                best_yc = yc_f
                best_r = r_f

    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    notes = [window_note] if window_note else []
    if n_evaluated == 0:
        notes.append(
            f"No circle was scored: no entry point (x {x_entry_range[0]:.2f} "
            f"to {x_entry_range[1]:.2f}) lies more than 1 m left of an exit "
            f"point (x {x_exit_range[0]:.2f} to {x_exit_range[1]:.2f}). Entry "
            f"is the low-x end of a slip surface and exit the high-x end.")
    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
        warnings=notes,
    )


# ── Differential-Evolution Noncircular Refinement ───────────────────────

@_counts_ground_refusals
def search_de(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_points: int = 7,
    method: str = "spencer",
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    maxiter: int = 30,
    popsize: int = 15,
    seed_surface=None,
    n_seed_trials: int = 150,
    fos_fn: Optional[Callable] = None,
) -> SearchResult:
    """Noncircular critical-surface optimization via differential evolution.

    Optimizes a polyline slip surface parameterized as
    [x_entry, x_exit, depth_frac_1..k] (k = n_points - 2 interior
    vertices at evenly spaced x, each a fraction of the local depth to
    the lowest layer bottom). The parameterization enforces monotonic x
    and keeps vertices inside the ground; convex bumps (a vertex above
    the chord of its neighbours by more than 2% of the slope height)
    are penalized to keep surfaces kinematically reasonable.

    Seeded with the best surface from a short random search (or a
    user-provided ``seed_surface``) so DE refines rather than starts
    cold.

    Parameters
    ----------
    geom : SlopeGeometry
    x_entry_range, x_exit_range : (float, float)
        Entry/exit x windows on the ground surface.
    n_points : int
        Total polyline vertices (>= 4 recommended for DE). Default 7.
    method : str
        FOS method for evaluation. Default 'spencer'.
    n_slices, tol, seed : usual meanings.
    maxiter, popsize : int
        scipy differential_evolution budget. Default 30 / 15.
    seed_surface : PolylineSlipSurface, optional
        Starting surface; its vertices are resampled onto this
        parameterization. Default: best of a short random search.
    n_seed_trials : int
        Trials for the seeding random search. Default 150.

    Returns
    -------
    SearchResult

    References
    ----------
    Storn & Price (1997) — Differential Evolution.
    Cheng et al. (2007) — heuristic optimization of noncircular surfaces.
    """
    from scipy.optimize import differential_evolution
    import numpy as np

    rng = np.random.default_rng(seed)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    z_max_ground = max(z for _, z in geom.surface_points)
    slope_h = z_max_ground - min(z for _, z in geom.surface_points)
    n_interior = max(n_points - 2, 1)
    dim = 2 + n_interior
    window_span = _window_span(x_entry_range, x_exit_range)
    reject_stats = _new_reject_stats()

    def _to_slip(v):
        x_en = float(min(max(v[0], x_entry_range[0]), x_entry_range[1]))
        x_ex = float(min(max(v[1], x_exit_range[0]), x_exit_range[1]))
        if x_ex <= x_en + 1.0:
            return None
        pts = [(x_en, geom.ground_elevation_at(x_en))]
        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x = x_en + frac * (x_ex - x_en)
            zg = geom.ground_elevation_at(x)
            max_depth = zg - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            df = float(min(max(v[2 + j], 0.02), 0.95))
            pts.append((x, zg - df * max_depth))
        pts.append((x_ex, geom.ground_elevation_at(x_ex)))
        try:
            return PolylineSlipSurface(points=pts)
        except ValueError:
            return None

    best_seen = [_FOS_MAX]

    def _objective(v):
        slip = _to_slip(v)
        if slip is None:
            return _FOS_MAX
        fos = _compute_fos(geom, slip, method, n_slices, tol=tol,
                           window_span=window_span, reject_stats=reject_stats,
                           fos_fn=fos_fn, confirm_below=best_seen[0])
        best_seen[0] = min(best_seen[0], fos)
        # Convexity penalty: vertex above the chord of its neighbours
        pts = slip.points
        pen = 0.0
        bump_tol = 0.02 * max(slope_h, 1.0)
        for i in range(1, len(pts) - 1):
            x0, z0 = pts[i - 1]
            x1, z1 = pts[i + 1]
            t = (pts[i][0] - x0) / (x1 - x0) if x1 > x0 else 0.5
            z_chord = z0 + t * (z1 - z0)
            if pts[i][1] > z_chord + bump_tol:
                pen += 10.0 * (pts[i][1] - z_chord - bump_tol) / max(slope_h, 1.0)
        return fos + pen

    # ---- seeding -----------------------------------------------------------
    if seed_surface is None and n_seed_trials > 0:
        seed_res = search_noncircular(
            geom, x_entry_range, x_exit_range, n_trials=n_seed_trials,
            n_points=n_points, n_slices=n_slices, tol=tol, seed=seed,
            method=method, fos_fn=fos_fn,
        )
        if seed_res.critical is not None and seed_res.critical.slip_points:
            seed_surface = PolylineSlipSurface(
                points=seed_res.critical.slip_points)

    x0 = None
    if seed_surface is not None:
        pts = seed_surface.points
        x_en, x_ex = pts[0][0], pts[-1][0]
        x_en = min(max(x_en, x_entry_range[0]), x_entry_range[1])
        x_ex = min(max(x_ex, x_exit_range[0]), x_exit_range[1])
        v = [x_en, x_ex]
        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x = x_en + frac * (x_ex - x_en)
            z_slip = seed_surface.slip_elevation_at(x)
            zg = geom.ground_elevation_at(x)
            max_depth = zg - min_layer_bot
            df = 0.3 if (z_slip is None or max_depth <= 0) else \
                min(max((zg - z_slip) / max_depth, 0.02), 0.95)
            v.append(df)
        x0 = v

    bounds = ([x_entry_range, x_exit_range]
              + [(0.02, 0.95)] * n_interior)

    # init population: latin hypercube + seed row
    init = "latinhypercube"
    if x0 is not None:
        pop = np.empty((max(popsize, 5) * 1, dim))
        for i in range(pop.shape[0]):
            for d, (lo, hi) in enumerate(bounds):
                pop[i, d] = rng.uniform(lo, hi)
        pop[0, :] = x0
        init = pop

    result = differential_evolution(
        _objective, bounds, init=init, maxiter=maxiter,
        popsize=popsize, seed=seed, tol=1e-6, polish=False,
        mutation=(0.4, 1.0), recombination=0.8,
    )

    n_evaluated = int(result.nfev)
    best_slip = _to_slip(result.x)
    best_fos = float(result.fun)

    critical = None
    if best_slip is not None and best_fos < _FOS_MAX:
        # objective may include penalty; recompute clean FOS
        clean = _compute_fos(geom, best_slip, method, n_slices, tol=tol,
                             window_span=window_span, fos_fn=fos_fn,
                             reject_stats=reject_stats,
                             confirm_below=math.inf)
        try:
            if clean >= _FOS_MAX:     # its solution did not hold (above)
                raise ValueError("no confirmed solution")
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=clean,
                method=method.replace("_", "-").title(),
                xc=0.0, yc=0.0, radius=0.0,
                x_entry=x_entry, x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0, kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=[],
        **_rejection_kwargs(reject_stats, n_evaluated),
    )
