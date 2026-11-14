"""Where a circle ends (GM-343; GM-140, after step 3).

The rule. A circle's trial surface is an arc below the ground between two
points where the circle meets the ground. A circle that meets the ground at
more than two points (crossing it, or touching it within the touch
allowance) is split at those points. Each arc with ground above it is a
trial surface of its own, solved and screened like any other, and a search
scores each one. A mass with no driving moment is refused, as any such
surface is. Nothing is refused merely for being one of two masses, and no
arc is carried under the ground in front of the toe. The result says which
arc it is.

A touch: a stretch that dips under the ground by less than the touch
allowance carries no mass and is no arc; a corner the arc passes through
splits it; a corner it passes just under (within the allowance) is where an
arc given by its two ends may end, and does not split a circle given
without them (DESIGN.md says why: the search's minimum would move with the
allowance).

Print. The family of circles with their lower end on the face: FHWA
NHI-06-088 p. 6-38. A search that covers face exits at all heights: GEC 11
p. 9-6. Taylor's stability number for a vertical face in a phi = 0 soil,
3.83: F = 3.83 cu / (gamma H). The split itself is the module's ruling
(2026-11-13): no held reference prints a rule for a circle that meets the
ground more than twice.

Every geometry here is made up.
"""

import functools
import math

import pytest

from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.newmark import yield_acceleration
from slope_stability.probabilistic import fosm_fos
from slope_stability.rapid_drawdown import rapid_drawdown_fos
from slope_stability.search import _compute_fos, grid_search
from slope_stability.search_supplements import search_toe_circles
from slope_stability.slices import build_slices
from slope_stability.slip_surface import (
    CircularSlipSurface, SeveralArcsError, circle_ground_meetings,
    circle_of,
)

# Taylor's stability number for a vertical face, phi = 0: 3.83
_TAYLOR = 3.83 * 20.0 / (18.0 * 5.0)

# Taylor's circle for the cut below: centre (13.02, 10.96), through the toe
# (20, 0), to the crest at x 24.57. By hand (moments about the centre, the
# sector less the triangles): F = 0.8514. To two decimals its radius is
# 13.00, which passes 7 mm under the toe.
_TAYLOR_CIRCLE = dict(xc=13.02, yc=10.96,
                      radius=math.hypot(20.0 - 13.02, 10.96))


def _vertical_cut():
    """A vertical 5 m cut in undrained clay, cu 20 kPa, 18 kN/m3, with
    20 m of level ground in front of the toe."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (20.001, 5.0), (45, 5.0)],
        soil_layers=[SlopeSoilLayer("clay", 5.0, -10.0, 18.0, cu=20.0,
                                    analysis_mode="undrained")])


@functools.lru_cache(maxsize=None)
def _default_cut_search():
    """The default Bishop search of the vertical cut (run once for the
    tests that read it)."""
    return search_critical_surface(_vertical_cut(), method="bishop")


def _cut_over_rock():
    """A 10 m 1H:1V cut, 3.5 m of weak clay over a strong layer."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("clay", 10, 6.5, 18.5, phi=22, c_prime=3),
            SlopeSoilLayer("strong", 6.5, -40, 23, phi=40, c_prime=200),
        ])


def _valley():
    """Two 10 m 1H:1V slopes facing each other across a 10 m floor."""
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (20, 0), (30, 0), (40, 10),
                        (55, 10)],
        soil_layers=[SlopeSoilLayer("clay", 10, -20, 18.0, phi=20,
                                    c_prime=5)])


# ---------------------------------------------------------------------------
# the meeting points come from the ground's segments and corners
# ---------------------------------------------------------------------------

def test_meeting_points_are_exact_and_a_corner_divides_the_arc():
    """The points where a circle meets the ground are found from the
    ground's segments and corners, not by sampling: a circle through a toe
    has a meeting point at the toe, to the last figure, and the toe divides
    the arc."""
    geom = _vertical_cut()
    r = math.hypot(20.0 - 14.0, 0.0 - 9.0)       # through the toe (20, 0)
    found = circle_ground_meetings(14.0, 9.0, r, geom.surface_points)
    assert len(found.arcs) == 2
    (a0, b0), (a1, b1) = found.arcs
    assert a0 == pytest.approx(8.0, abs=1e-9)    # the floor, mirrored
    assert b0 == pytest.approx(20.0, abs=1e-9)
    assert a1 == pytest.approx(20.0, abs=1e-9)   # the toe ends both arcs
    # the crest end, on the level ground at z = 5
    assert b1 == pytest.approx(14.0 + math.sqrt(r * r - 16.0), abs=1e-9)


def test_a_circle_a_hair_above_the_toe_is_split_there():
    """A circle passing 2e-5 m above the toe leaves the ground there and
    comes back into it on the face: two arcs. (Sampling at 200 points, a
    step of about 0.13 m, stepped over that gap and read one surface from
    the far floor crossing: Taylor's circle read 2.676 for 0.851.)"""
    geom = _vertical_cut()
    r = math.hypot(6.0, 9.0) * (1.0 - 1e-6)
    slip = CircularSlipSurface(14.0, 9.0, r)
    arcs = slip.arcs(geom)
    assert len(arcs) == 2
    assert arcs[0].x_end < 20.0 < arcs[1].x_start
    assert arcs[1].x_start - arcs[0].x_end < 1e-3
    with pytest.raises(SeveralArcsError, match="bounds 2 arcs"):
        slip.find_entry_exit(geom)               # none was named


def test_a_circle_with_two_meeting_points_is_its_one_arc():
    """A circle that meets the ground at two points is one arc, given with
    its ends or without: the same surface, the same factor of safety."""
    geom = _cut_over_rock()
    slip = CircularSlipSurface(22.0, 14.0, 9.0)
    (arc,) = slip.arcs(geom)
    assert slip.find_entry_exit(geom) == (arc.x_start, arc.x_end)
    one = analyze_slope(geom, slip_surface=slip)
    assert one.arc is None
    bound = analyze_slope(geom, slip_surface=arc)
    assert bound.FOS == one.FOS
    assert bound.arc["n_arcs"] == 1


# ---------------------------------------------------------------------------
# (a) Taylor's vertical cut
# ---------------------------------------------------------------------------

def test_taylors_circle_ends_at_the_toe():
    """Taylor's circle for the vertical cut (stability number 3.83),
    ended at the toe: F = 0.8514 by hand. The arc under the floor in front
    of the toe (14 m more arc, no more driving moment: the lens under the
    level floor is symmetric about the centre) is a surface of its own,
    refused for having no driving moment; it is not carried into the
    slope's surface (2.676 before)."""
    geom = _vertical_cut()
    res = analyze_slope(geom, method="bishop", n_slices=120,
                        **_TAYLOR_CIRCLE)
    assert res.FOS == pytest.approx(0.8514, rel=0.005)
    assert res.x_entry == pytest.approx(20.0, abs=1e-6)
    assert res.x_exit == pytest.approx(24.57, abs=0.01)
    # the result says which arc it is, and what became of the other
    assert res.arc["index"] == 2 and res.arc["n_arcs"] == 2
    assert "arc 2 of the 2 arcs" in res.arc["description"]
    (other,) = res.arc["others"]
    assert other["x_end"] == pytest.approx(20.0, abs=1e-6)
    assert other["not_scored"] == "no_driving"
    assert res.arc["not_scored"] == {"no_driving": 1}
    assert "bounds 2 arcs" in res.warnings[0]
    # the arc named by its two ends is the same surface
    named = analyze_slope(
        geom, method="bishop", n_slices=120,
        slip_surface=CircularSlipSurface(
            res.xc, res.yc, res.radius, res.x_entry, res.x_exit))
    assert named.FOS == pytest.approx(res.FOS, rel=1e-9)
    # and the slices stop at the toe
    slices = build_slices(geom, circle_of(res), 30)
    assert min(s.x_mid - 0.5 * s.width for s in slices) >= 20.0 - 1e-6


def test_vertical_cut_default_search_reads_taylor():
    """The default Bishop search within 2 % of Taylor's 0.851 (0.935
    before, on a circle dipping 0.048 m under the floor, inside the touch
    allowance; circles leaving the face just above the toe were refused as
    two masses)."""
    res = _default_cut_search()
    assert res.critical.FOS == pytest.approx(_TAYLOR, rel=0.02)
    # the governing surface ends at the toe, or on the face above it
    assert res.critical.x_entry >= 20.0 - 1e-6
    assert res.n_rejected_ground_crossings == 0
    assert sum(res.circles_split.values()) > 0


def test_spencer_on_taylors_arc_has_no_solution_and_says_so():
    """Check (c): on the arc ended at the toe, solved on the circle,
    Spencer's method finds no lambda at which the moment and force factors
    of safety agree. It reports no solution with step 2's reason, never a
    forced value."""
    geom = _vertical_cut()
    arc = CircularSlipSurface(**_TAYLOR_CIRCLE).arcs(geom)[1]
    res = analyze_slope(geom, slip_surface=arc, method="spencer",
                        n_slices=60)
    assert res.FOS is None and not res.method_solved
    assert any("no equilibrium solution" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# a circle that meets the ground at four points gives two arcs
# ---------------------------------------------------------------------------

def test_four_points_two_arcs_each_scored_and_the_governing_arc_named():
    """A circle across a valley is below the ground under each slope and
    in the air over the floor: two arcs, each a slip surface of its own
    with its own factor of safety. The result is the lower and names its
    arc; the other is listed with its value. (Refused before, as two
    masses.)"""
    geom = _valley()
    slip = CircularSlipSurface(24.5, 16.0, 15.7)
    found = slip.meetings(geom)
    assert found.n_crossings == 4 and len(found.arcs) == 2
    res = analyze_slope(geom, slip_surface=slip, method="bishop")
    each = [analyze_slope(geom, slip_surface=a, method="bishop").FOS
            for a in slip.arcs(geom)]
    assert each == pytest.approx([0.919, 1.591], abs=0.002)
    assert res.FOS == min(each)
    assert res.arc["index"] == 1
    assert (res.x_entry, res.x_exit) == found.arcs[0]
    (other,) = res.arc["others"]
    assert other["value"] == pytest.approx(each[1])
    assert "not_scored" not in other
    # the searches' scorer: the lowest arc's score, and the arc remembered
    trial = CircularSlipSurface(24.5, 16.0, 15.7)
    assert _compute_fos(geom, trial, "bishop", 30) == pytest.approx(
        each[0])
    assert trial._governing_arc == found.arcs[0]


def test_a_search_counts_the_circles_split_and_names_the_governing_arc():
    """A search over one centre whose circles pass under the floor in
    front of the toe: each split circle is counted with the number of arcs
    it gave, each arc under the floor is refused for having no driving
    moment and counted under that reason, and the governing result names
    its arc. Its two ends are kept by a later analysis."""
    geom = _vertical_cut()
    res = grid_search(geom, x_range=(13.02, 13.02), y_range=(10.96, 10.96),
                      nx=1, ny=1, method="bishop")
    assert res.critical.FOS == pytest.approx(_TAYLOR, rel=0.02)
    n_split = res.circles_split[2]
    assert n_split > 0 and set(res.circles_split) == {2}
    (rec,) = res.no_solution
    assert rec["no_driving"] == n_split and rec["not_converged"] == 0
    assert res.n_rejected_ground_crossings == 0
    arc = res.critical.arc
    assert arc["n_arcs"] == 2 and arc["index"] == 2
    assert arc["x_start"] == pytest.approx(20.0, abs=1e-6)
    assert res.critical.x_entry == arc["x_start"]
    again = analyze_slope(geom, slip_surface=circle_of(res.critical),
                          method="bishop")
    assert again.FOS == pytest.approx(res.critical.FOS, rel=1e-6)
    assert again.x_entry == arc["x_start"]
    assert "circles_split" in res.to_dict()
    assert "split there" in res.summary()
    # no windows: every arc is scored, none is left out
    assert res.arcs_outside_windows == {}
    assert "arcs_outside_windows" not in res.to_dict()


def test_arcs_outside_the_windows_are_counted_not_scored():
    """The review of GM-343 (ruling 2). An arc of a split circle that lies
    outside the search's entry / exit windows is not scored; it is counted
    by family beside circles_split, and the statements say "each arc
    inside the windows". On the vertical cut the default search's
    automatic windows leave the lenses under the floor out: counted there,
    not as no_driving. With windows given at one centre, the lens of each
    split circle is left out and counted, and none is scored."""
    res = _default_cut_search()
    out = res.arcs_outside_windows
    assert out.get("circular", 0) > 0       # the main family, as no_solution
    assert all(n > 0 for n in out.values())
    assert res.to_dict()["arcs_outside_windows"] == out
    text = res.summary()
    assert "each arc inside the windows scored" in text
    assert "outside the windows, not scored" in text
    (said,) = [c for c in res.choices
               if c["what"].startswith("circles that meet the ground")]
    assert "each arc inside the windows scored" in said["chosen"]
    # one centre, windows given: entry at the toe, exit on the crest
    geom = _vertical_cut()
    win = grid_search(geom, x_range=(13.02, 13.02), y_range=(10.96, 10.96),
                      nx=1, ny=1, method="bishop",
                      x_entry_range=(19.0, 21.0), x_exit_range=(21.0, 45.0))
    n_split = win.circles_split[2]
    assert n_split > 0
    # each split circle's lens (from x < 19) is outside the entry window
    assert win.arcs_outside_windows["main"] >= n_split
    assert all(r["no_driving"] == 0 for r in win.no_solution)
    assert win.critical.x_entry == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# a toe circle ends at the toe
# ---------------------------------------------------------------------------

def test_a_toe_circle_ends_at_the_toe():
    """The toe-circle search sets each arc's end at the toe (FHWA
    NHI-06-088 p. 6-38): no arc is carried under the floor in front of it,
    and the radius is the circle's own (it was shrunk by a millionth to
    pass just above the toe)."""
    geom = _vertical_cut()
    toe = search_toe_circles(geom, method="bishop", above_toe=False)
    assert toe["FOS"] == pytest.approx(_TAYLOR, rel=0.02)
    assert toe["toe"] == (20, 0)
    assert toe["arc"][0] == pytest.approx(20.0, abs=1e-6)
    # the circle passes through the toe itself
    assert math.hypot(toe["xc"] - 20.0, toe["yc"]) == pytest.approx(
        toe["radius"], rel=1e-9)
    slip = CircularSlipSurface(toe["xc"], toe["yc"], toe["radius"],
                               *toe["arc"])
    slices = build_slices(geom, slip, 30)
    assert min(s.x_mid - 0.5 * s.width for s in slices) >= 20.0 - 1e-6


# ---------------------------------------------------------------------------
# a touch within the allowance is a meeting point
# ---------------------------------------------------------------------------

def test_a_circle_touching_the_floor_within_the_allowance_is_split_there():
    """A circle whose lowest point dips 0.04 m under the floor (under the
    touch allowance, 1 % of its greatest depth below the ground) touches
    it: that stretch carries no mass and is no arc, and the circle's arc
    is the one from the face to the crest."""
    geom = _vertical_cut()
    found = CircularSlipSurface(18.48, 6.23, 6.27).meetings(geom)
    assert found.n_crossings == 4
    (touch,) = found.touches
    assert 17.7 < touch[0] < touch[1] < 19.2
    (arc,) = found.arcs
    assert arc[0] == pytest.approx(20.0, abs=1e-3)   # on the face


def test_an_arc_may_be_ended_at_a_toe_it_passes_just_under():
    """Taylor's circle to two decimals (R 13.00) passes 0.007 m under the
    toe, less than the touch allowance (0.05 m here), and runs on under
    the floor: given without ends it is that one surface, two crossings,
    as before (2.68), and the result says what it passed under. A touch
    within the allowance counts as a meeting point: the arc may be ended
    there by giving its two ends, and reads Taylor's value. A circle
    passing 0.5 m under the toe cannot be ended there."""
    geom = _vertical_cut()
    slip = CircularSlipSurface(13.02, 10.96, 13.00)
    found = slip.meetings(geom)
    assert len(found.arcs) == 1 and found.n_crossings == 2
    ((x_toe, under),) = found.touched
    assert x_toe == 20 and under == pytest.approx(0.0072, abs=0.0002)
    assert under < found.allowance
    one = analyze_slope(geom, slip_surface=slip, method="bishop",
                        n_slices=120)
    assert one.FOS == pytest.approx(2.677, abs=0.005)
    assert any("under the ground corner at x = 20.00" in w
               for w in one.warnings)
    ended = analyze_slope(
        geom, method="bishop", n_slices=120,
        slip_surface=CircularSlipSurface(13.02, 10.96, 13.00, 20.0, 24.57))
    assert ended.FOS == pytest.approx(0.8514, rel=0.005)
    assert ended.x_entry == 20
    assert "its ends as given" in ended.arc["description"]
    with pytest.raises(ValueError, match="not the two ends of an arc"):
        analyze_slope(geom, method="bishop", slip_surface=CircularSlipSurface(
            13.02, 10.96, 13.50, 20.0, 24.9))


# ---------------------------------------------------------------------------
# an arc with no driving moment is refused and counted
# ---------------------------------------------------------------------------

def test_an_arc_with_no_driving_moment_is_refused_and_counted():
    """The lens under a level floor is symmetric about the circle's
    centre: no driving moment. As a surface of its own it has no factor of
    safety (GM-140 step 2), in `analyze_slope` and in a search's scorer."""
    geom = _cut_over_rock()
    slip = CircularSlipSurface(10.0, 39.5, 40.0)
    lens, slope = slip.arcs(geom)
    assert (lens.x_start, lens.x_end) == pytest.approx((3.695, 16.305),
                                                       abs=0.001)
    alone = analyze_slope(geom, slip_surface=lens, method="bishop",
                          n_slices=40)
    assert alone.FOS is None
    assert any("driving" in w for w in alone.warnings)
    res = analyze_slope(geom, slip_surface=slip, method="bishop",
                        n_slices=40)
    assert res.arc["index"] == 2
    assert res.arc["not_scored"] == {"no_driving": 1}
    # the slope's mass on its own (Spencer 7.40 measured for GM-122)
    sp = analyze_slope(geom, slip_surface=slip, method="spencer",
                       n_slices=40)
    assert sp.FOS == pytest.approx(7.39, abs=0.02)
    assert (sp.x_entry, sp.x_exit) == pytest.approx((21.06, 37.01),
                                                    abs=0.01)


# ---------------------------------------------------------------------------
# the rule in the other single-surface analyses
# ---------------------------------------------------------------------------

def test_the_other_analyses_of_one_circle_take_its_lowest_arc():
    """Rapid drawdown, the yield coefficient and FOSM on a circle that
    bounds two arcs analyse each arc as a surface of its own (FOSM: the
    arc with the lowest factor of safety at the values given, held for
    every evaluation), and say which arc the result is."""
    geom = _valley()
    circle = dict(xc=24.5, yc=16.0, radius=15.7)
    ends = CircularSlipSurface(**circle).meetings(geom).arcs
    ky = yield_acceleration(geom, method="spencer", n_slices=30, **circle)
    each = [yield_acceleration(geom, method="spencer", n_slices=30,
                               slip_surface=CircularSlipSurface(
                                   24.5, 16.0, 15.7, a, b)).ky
            for a, b in ends]
    assert ky.ky == min(k for k in each if k is not None)
    assert ky.arc["n_arcs"] == 2
    dd = rapid_drawdown_fos(geom, 8.0, 2.0, method="duncan_3stage", n_slices=30,
                            **circle)
    assert dd.arc["n_arcs"] == 2 and "bounds 2 arcs" in dd.warnings[0]
    fosm = fosm_fos(geom, {"c_prime": {"mean": 5.0, "cov": 0.3}}, method="bishop",
                    **circle)
    low = analyze_slope(geom, method="bishop", **circle)
    assert fosm.fos_mlv == pytest.approx(low.FOS, rel=1e-9)


def test_ends_that_are_not_an_arc_of_the_circle_are_refused():
    geom = _cut_over_rock()
    with pytest.raises(ValueError, match="not the two ends of an arc"):
        analyze_slope(geom, slip_surface=CircularSlipSurface(
            22.0, 14.0, 9.0, 24.0, 27.0))
    with pytest.raises(ValueError, match="both ends"):
        CircularSlipSurface(22.0, 14.0, 9.0, x_start=24.0)
