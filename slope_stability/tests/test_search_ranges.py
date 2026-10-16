"""Automatic search ranges set from the slope, not the model width (GM-57).

The ranges come from the slope's faces, its height H and the depth d from
the toe to the base (``slope_stability/search_ranges.py``). Before GM-57
they were fractions of the model width: entry in the first 40 %, exit in the
last 40 %, centres over the whole profile. Measured on the models below
with that code (Bishop, default search):

- a c-phi 2:1 slope drawn with 10 H of level ground each side: the circle
  search found no admissible circle at all (the toe circle family alone gave
  the answer); drawn off-centre (0.5 H behind the crest and 6 H in front of
  the toe, or the reverse) it read 3.16 / 3.20 against 1.645;
- the soft-foundation slope (Taylor D = 3) drawn with 4 H of level ground on
  one side and 12 H on the other: 0.782 either way (+15 % against Taylor's
  0.679; a toe circle, the deep base circle out of reach), against 0.676
  with 4 H or 12 H each side.
"""

import math

import pytest

from slope_stability.analysis import search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.rapid_drawdown import search_rapid_drawdown
from slope_stability.results import SlopeStabilityResult
from slope_stability.search_ranges import automatic_ranges, resolve_ranges
from slope_stability.search_supplements import _clipped_range_warning

H = 10.0


def _slope(layers, behind, front, run=2 * H, height=H, rises=True):
    """A simple slope: `front` of level ground beyond the toe, the face,
    `behind` of level ground behind the crest. Returns (geom, toe x)."""
    if rises:
        pts = ([(0.0, 0.0)] if front > 0 else []) + [
            (front, 0.0), (front + run, height),
            (front + run + behind, height)]
        toe = front
    else:
        pts = [(0.0, height), (behind, height), (behind + run, 0.0)] + (
            [(behind + run + front, 0.0)] if front > 0 else [])
        toe = behind + run
    return SlopeGeometry(surface_points=pts, soil_layers=layers), toe


def _cphi():
    return [SlopeSoilLayer("fill", top_elevation=H, bottom_elevation=-H,
                           gamma=19.0, phi=25.0, c_prime=10.0)]


def _soft_clay(depth_factor=3):
    """Taylor's soft foundation: su / (gamma H) = 0.12, the firm base
    (depth_factor - 1) H below the toe."""
    return [SlopeSoilLayer("clay", top_elevation=H,
                           bottom_elevation=-(depth_factor - 1) * H,
                           gamma=18.0, cu=0.12 * 18.0 * H,
                           analysis_mode="undrained")]


# ---------------------------------------------------------------------------
# The ranges themselves
# ---------------------------------------------------------------------------

def test_reach_from_height_and_depth_to_base():
    """2:1 slope, base H below the toe (Taylor D = 2): grid top 2 H above
    the crest (Y = 3 H above the toe), reach sqrt(d (2Y + d)) = 2.65 H
    beyond the toe and sqrt((Y + d)^2 - (Y - H)^2) = 3.46 H behind the
    crest; Taylor's critical circle comes out 1.19 H and goes in 1.76 H."""
    g, toe = _slope(_soft_clay(2), behind=6 * H, front=6 * H)
    a = automatic_ranges(g)
    (s,) = a.slopes
    assert s["height"] == pytest.approx(H)
    assert s["depth_to_base"] == pytest.approx(H)
    assert s["reach_toe"] == pytest.approx(math.sqrt(7.0) * H, rel=1e-6)
    assert s["reach_crest"] == pytest.approx(math.sqrt(12.0) * H, rel=1e-6)
    crest = toe + 2 * H
    assert a.x_entry_range == pytest.approx((toe - math.sqrt(7.0) * H,
                                             toe + H))
    assert a.x_exit_range == pytest.approx((crest - H,
                                            crest + math.sqrt(12.0) * H))
    assert a.x_range == pytest.approx((a.x_entry_range[0],
                                       a.x_exit_range[1]))
    assert a.y_range == pytest.approx((H + 0.1 * H, 3 * H))
    assert a.clipped == {}


def test_ranges_do_not_depend_on_the_model_width():
    """Drawn with 6 H or 20 H of level ground each side, the slope gets the
    same ranges (measured from its toe)."""
    out = []
    for flat in (6 * H, 20 * H):
        g, toe = _slope(_soft_clay(2), behind=flat, front=flat)
        a = automatic_ranges(g)
        out.append([v - toe for v in a.x_entry_range + a.x_exit_range]
                   + list(a.y_range))
    assert out[0] == pytest.approx(out[1])


def test_deep_base_raises_the_grid_top():
    """With the base 3 H below the toe the centre grid's top is d = 3 H
    above the crest (Janbu's midpoint circles rise with d), not 2 H."""
    g, _ = _slope(_soft_clay(4), behind=10 * H, front=10 * H)
    a = automatic_ranges(g)
    assert a.y_range[1] == pytest.approx(H + 3 * H)


def test_model_end_cuts_the_range_and_it_is_recorded():
    """A model 1 H each side of the face is shorter than the slope's reach:
    the windows stop at the ends of the profile and `clipped` says where
    the slope called for them; the search result carries it."""
    g, toe = _slope(_soft_clay(2), behind=H, front=H)
    a = automatic_ranges(g)
    assert a.x_entry_range[0] == 0.0
    assert a.x_exit_range[1] == pytest.approx(4 * H)
    assert a.clipped["entry_lo"] == pytest.approx(toe - math.sqrt(7.0) * H)
    assert a.clipped["exit_hi"] == pytest.approx(
        toe + 2 * H + math.sqrt(12.0) * H)
    res = search_critical_surface(g, method="bishop", nx=4, ny=4)
    sr = res.search_ranges
    assert sr["automatic"] == ["x_entry_range", "x_exit_range", "x_range",
                               "y_range"]
    assert set(sr["clipped"]) == {"entry_lo", "exit_hi"}
    assert sr["slopes"][0]["reach_toe"] == pytest.approx(
        math.sqrt(7.0) * H, abs=1e-3)
    assert "Automatic ranges" in res.summary()
    assert res.to_dict()["search_ranges"] == sr


def test_base_at_the_toe_gives_no_reach_beyond_it():
    """A slope standing on its base (d = 0): no circle can come out beyond
    the toe, so the range stops there (a small slack) and a model that
    starts at the toe is not cut."""
    layers = [SlopeSoilLayer("clay", top_elevation=H, bottom_elevation=0.0,
                             gamma=18.0, cu=40.0, analysis_mode="undrained")]
    g, toe = _slope(layers, behind=4 * H, front=0.0)
    a = automatic_ranges(g)
    assert a.slopes[0]["depth_to_base"] == 0.0
    assert a.slopes[0]["reach_toe"] == pytest.approx(0.02 * H)
    assert "entry_lo" not in a.clipped


def test_faces_both_ways_join_their_windows():
    """An embankment: each face's windows are joined, so failures of
    either face are inside them."""
    layers = [SlopeSoilLayer("fill", top_elevation=H, bottom_elevation=-H,
                             gamma=19.0, phi=30.0, c_prime=5.0)]
    pts = [(0, 0), (40, 0), (60, H), (70, H), (90, 0), (130, 0)]
    a = automatic_ranges(SlopeGeometry(surface_points=pts,
                                       soil_layers=layers))
    assert sorted(s["rises"] for s in a.slopes) == [-1, 1]
    # left face rising: toe at 40; right face falling: toe at 90
    assert a.x_entry_range[0] < 40.0 and a.x_exit_range[1] > 90.0
    assert a.x_entry_range[1] == pytest.approx(80.0)     # right face's middle
    assert a.x_exit_range[0] == pytest.approx(50.0)      # left face's middle


def test_small_opposite_face_left_out():
    """A ditch a tenth as deep as the slope is high adds no face group."""
    pts = [(0, 0), (20, 0), (40, H), (50, H), (51, H - 1), (52, H),
           (100, H)]
    a = automatic_ranges(SlopeGeometry(surface_points=pts,
                                       soil_layers=_cphi()))
    assert [s["rises"] for s in a.slopes] == [1]


def test_level_ground_falls_back_to_fractions():
    g = SlopeGeometry(surface_points=[(0, 5), (50, 5)], soil_layers=_cphi())
    a = automatic_ranges(g)
    assert a.slopes == []
    assert a.x_entry_range == (0, 20.0) and a.x_exit_range == (30.0, 50)


def test_given_windows_widen_the_automatic_centre_range():
    """A caller-given window outside the slope's own reach stays inside the
    automatic centre grid."""
    g, toe = _slope(_cphi(), behind=20 * H, front=20 * H)
    en, ex, xr, yr, auto = resolve_ranges(g, x_entry_range=(0.0, 10.0))
    assert en == (0.0, 10.0)
    assert xr[0] == 0.0 and xr[1] == auto.x_range[1]


def test_clipped_range_warning_text():
    """The governing surface ending at a cut range is said in words."""
    g, toe = _slope(_soft_clay(2), behind=H, front=H)
    a = automatic_ranges(g)
    crit = SlopeStabilityResult(FOS=1.0, method="Bishop", x_entry=0.05,
                                x_exit=2.5 * H)
    w = _clipped_range_warning(g, crit, (a.x_entry_range, a.x_exit_range),
                               (True, True), a)
    assert w.startswith("The critical surface ends at x = 0.05")
    assert "reach runs to x = " in w
    assert _clipped_range_warning(g, crit, (a.x_entry_range,
                                            a.x_exit_range),
                                  (False, True), a) is None


# ---------------------------------------------------------------------------
# A failure confined to part of the face: the search is directed, and the
# result says what it covered (FHWA NHI-06-088 p. 6-38)
# ---------------------------------------------------------------------------

def _weak_over_rock():
    """A 10 m 1:1 cut: 4 m of weak soil over rock, which meets the face
    6 m above the toe."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("weak", top_elevation=10, bottom_elevation=6,
                           gamma=18, phi=26, c_prime=2),
            SlopeSoilLayer("rock", top_elevation=6, bottom_elevation=-10,
                           gamma=21, phi=40, c_prime=150)])


def test_weak_layer_over_rock_undirected_reads_high_and_says_where():
    """The automatic ranges are for the slope as a whole; the critical
    circle here lies in the weak layer above the rock's outcrop, 4.6 m long
    (0.885 from a 20 x 20 search with both windows over the whole profile),
    which the default windows (ends either side of the face's mid-length)
    do not reach: 4.82 (5.25 with windows from the model width). Seeing
    that the geometry needs a directed search is the engineer's or the
    agent's judgment; the result gives what it needs: the ranges used and
    where the governing surface lies, through which layers."""
    res = search_critical_surface(_weak_over_rock(), method="bishop")
    assert res.critical.FOS == pytest.approx(4.82, abs=0.05)
    gl = res.governing_location
    assert "rock" in gl["layers"]
    assert gl["depth"] > 1.0                   # a slide, not a sliver
    assert gl["slope"]["toe"] == [20, 0] and gl["slope"]["height"] == 10
    assert gl["x_entry"] == pytest.approx(res.critical.x_entry, abs=1e-3)
    assert res.search_ranges["x_entry_range"] == [0, 25.0]
    d = res.to_dict()
    assert d["governing_location"] == gl
    assert "through rock" in res.summary()


@pytest.mark.parametrize("kw, tol", [
    (dict(), 0.015),                           # the default 10 x 10 grid
    (dict(nx=20, ny=20), 0.006),
    (dict(surface_type="entry_exit", nx=20, ny=20), 0.006),
])
def test_weak_layer_over_rock_directed_search_finds_it(kw, tol):
    """Directed with windows that put the surface's ends on the weak
    layer's outcrop and behind the crest, the search finds the circle in
    the weak layer: 0.895 on the default grid, 0.889 / 0.888 on 20 x 20,
    against the full-profile reference 0.885."""
    g = _weak_over_rock()
    ref = search_critical_surface(
        g, method="bishop", x_entry_range=(0, 60), x_exit_range=(0, 60),
        x_range=(10, 50), y_range=(8, 40), nx=20, ny=20)
    assert ref.critical.FOS == pytest.approx(0.885, abs=0.005)
    res = search_critical_surface(g, method="bishop",
                                  x_entry_range=(25.5, 28.0),
                                  x_exit_range=(29.0, 40.0), **kw)
    assert res.critical.FOS == pytest.approx(ref.critical.FOS, rel=tol)
    assert res.critical.FOS >= ref.critical.FOS - 1e-3
    assert res.governing_location["layers"][-1] == "weak"
    assert res.search_ranges["automatic"] == (
        [] if kw.get("surface_type") else ["x_range", "y_range"])


# ---------------------------------------------------------------------------
# The same slope drawn narrow and wide gives the same answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behind, front", [
    (2 * H, 2 * H), (10 * H, 10 * H), (0.5 * H, 6 * H), (6 * H, 0.5 * H)])
def test_cphi_slope_narrow_or_wide_model(behind, front):
    """The circle search on its own (no toe-circle family) finds the toe
    circle the family finds, 1.645, however the model is drawn: 1.645-1.655
    (before GM-57: none at 10 H each side, 3.16 / 3.20 off-centre)."""
    g, toe = _slope(_cphi(), behind=behind, front=front)
    res = search_critical_surface(g, method="bishop", toe_circles=False)
    c = res.critical
    assert res.governing_surface == "circular"
    assert c.FOS == pytest.approx(1.645, rel=0.008)
    assert c.x_entry == pytest.approx(toe, abs=0.3 * H)       # toe circle


@pytest.mark.parametrize("behind, front", [
    (4 * H, 4 * H), (12 * H, 12 * H), (4 * H, 12 * H), (12 * H, 4 * H)])
def test_soft_foundation_circle_exits_beyond_the_toe(behind, front):
    """Taylor D = 3 (2:1 slope, su / (gamma H) = 0.12, base 2 H below the
    toe; FHWA NHI-06-088 Fig. 6-15, F = 0.679): the deep circle tangent to
    the base comes out 2.69 H beyond the toe, wherever the model ends past
    that. Default search: 0.676 at 30 slices on every model (before
    GM-57: 0.782 with 4 H on one side and 12 H on the other)."""
    g, toe = _slope(_soft_clay(3), behind=behind, front=front, rises=False)
    res = search_critical_surface(g, method="bishop")
    c = res.critical
    assert res.governing_surface == "circular"
    assert c.FOS == pytest.approx(0.679, rel=0.01)
    assert c.yc - c.radius == pytest.approx(-2 * H, abs=0.05)
    assert (c.x_exit - toe) / H == pytest.approx(2.69, abs=0.1)
    assert not res.warnings


def test_rapid_drawdown_search_uses_the_slope_ranges():
    """search_rapid_drawdown sets its ranges the same way and reports them."""
    layers = [SlopeSoilLayer("shell", top_elevation=24.0,
                             bottom_elevation=-10.0, gamma=19.0,
                             gamma_sat=20.0, phi=37.0, c_prime=0.0)]
    g = SlopeGeometry(surface_points=[(0, 0), (30, 24), (75, 24)],
                      soil_layers=layers)
    res = search_rapid_drawdown(g, 20.0, 5.0, method="corps_2stage",
                                surface_type="circular", nx=3, ny=3,
                                n_slices=20, toe_circles=False,
                                infinite_slope=False, refine=False)
    sr = res.search.search_ranges
    a = automatic_ranges(g)
    assert sr["x_exit_range"] == pytest.approx(list(a.x_exit_range),
                                               abs=1e-3)
    assert sr["y_range"] == pytest.approx(list(a.y_range), abs=1e-3)
