"""GM-140 step 7a: the noncircular search finished as the print describes
one (the account's D12, D43 and the noncircular part of D9; GM-117).

Every value here is made up. The printed rules:

* EM 1110-2-1902 (2003) C-1g(3), p. C-10, Fig. C-8: hold all points but
  one, shift that point a small distance in two directions, move the
  surface, repeat until F stops falling and the moves become minimal.
* EM 4-2e(2)(b), p. 4-7: the final shift increments no more than 10 to 25
  percent of the thickness of the thinnest stratum.
* EM 4-2e(1)(b), pp. 4-3 and 4-6; C-1g(4), p. C-11; FHWA NHI-06-088 p.
  6-22: several locally critical surfaces, independently started searches.
* EM 4-2d(2), p. 4-2: the output names the most critical surface and the
  other surfaces analysed.
* EM 4-2e(1)(a), p. 4-3: the search must not jump from one face to another.
* EM 4-2e(2)(a), p. 4-7, and Eq. C-32, p. C-34: the toe; the passive exit
  inclination 45 - phi'/2.
* EM C-1f(2), p. C-6, and C-1g(2), pp. C-8 to C-10: a wedge along a weak
  layer, its central block's two ends varied, the wedges at 45 +/- phi'_D/2.
"""

import math

import pytest

from slope_stability.analysis import search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.rapid_drawdown import (rapid_drawdown_fos,
                                            search_rapid_drawdown)
from slope_stability.search import _compute_fos, _layer_band_at
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability import search_refine as SR


def _weak_layer_slope():
    """A 10 m slope rising toward +x (toe at x = 20, crest at x = 40), with
    a 1 m weak layer (el. 1 to 2) above the toe level in a stiffer soil."""
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (20.0, 0.0), (40.0, 10.0), (70.0, 10.0)],
        soil_layers=[
            SlopeSoilLayer("upper", top_elevation=10.0, bottom_elevation=2.0,
                           gamma=19.0, phi=30.0, c_prime=12.0),
            SlopeSoilLayer("weak", top_elevation=2.0, bottom_elevation=1.0,
                           gamma=19.0, phi=12.0, c_prime=0.0),
            SlopeSoilLayer("lower", top_elevation=1.0, bottom_elevation=-6.0,
                           gamma=19.0, phi=30.0, c_prime=12.0)])


_WINDOWS = dict(x_entry_range=(5.0, 32.0), x_exit_range=(40.0, 65.0))


@pytest.fixture(scope="module")
def weak_layer_search():
    return {refine: search_critical_surface(
        _weak_layer_slope(), surface_type="weak_layer", method="spencer",
        n_slices=30, refine=refine, **_WINDOWS) for refine in (False, True)}


def test_wedge_stage_reaches_a_wedge_family_declared_before_scoring(
        weak_layer_search):
    """EM C-1g(2), pp. C-8 to C-10: the wedge search varies the two ends of
    the central block's base, the wedges at 45 -/+ phi'_D/2. The family is
    declared here before it is scored: the base on the weak layer's
    mid-line (el. 1.5), its ends at the windows' span's quarter points, the
    wedges at C-1g(2)'s angles on F = 1 with the upper soil's phi' = 30.
    The wedge stage (refine=False) reads no more than its lowest plus the
    slicing difference (the family's best at 30 against 60 slices)."""
    g = _weak_layer_slope()
    lo, hi = 5.0, 65.0
    xs = [lo + (hi - lo) * k / 4 for k in range(5)]
    a_toe = 45.0 - 30.0 / 2.0           # passive, toe side (low x)
    a_crest = 45.0 + 30.0 / 2.0         # active
    best = (math.inf, None)
    for i, xa in enumerate(xs):
        for xb in xs[i + 1:]:
            p0 = SR._wedge_leg(g, xa, 1.5, a_toe, -1)
            p1 = SR._wedge_leg(g, xb, 1.5, a_crest, +1)
            if p0 is None or p1 is None:
                continue
            pts = [p0] + [(x, 1.5) for x in xs if xa <= x <= xb] + [p1]
            if not (5.0 <= p0[0] <= 32.0 and 40.0 <= p1[0] <= 65.0):
                continue
            f = _compute_fos(g, PolylineSlipSurface(points=pts), "spencer",
                             30, window_span=_span())
            if f < best[0]:
                best = (f, pts)
    assert best[1] is not None
    f60 = _compute_fos(g, PolylineSlipSurface(points=best[1]), "spencer", 60,
                       window_span=_span())
    slicing = abs(f60 - best[0])
    stage = weak_layer_search[False]
    assert stage.critical is not None
    assert stage.critical.FOS <= best[0] + slicing + 1e-6
    assert stage.refinement["refined"] is False
    assert all(m["start"].startswith("wedge along")
               for m in stage.local_minima)


def _span():
    from slope_stability.search import _window_span
    return _window_span(_WINDOWS["x_entry_range"], _WINDOWS["x_exit_range"])


def test_refined_result_is_no_more_than_the_wedge_stage(weak_layer_search):
    """EM C-1g(3), p. C-10: the refinement moves a surface only where F
    falls, so the refined result is no more than the wedge stage."""
    assert (weak_layer_search[True].critical.FOS
            <= weak_layer_search[False].critical.FOS + 1e-9)


def test_every_layer_within_the_search_depth_has_a_wedge(weak_layer_search):
    """Call 3: one wedge start for every layer within the search's depth
    that holds ground below the ground surface in the windows' span."""
    g = _weak_layer_slope()
    rec = weak_layer_search[True].refinement
    named = {w["layer"] for w in rec["wedges"]}
    for layer in g.soil_layers:
        holds = any(
            (band := _layer_band_at(g, layer, x)) is not None
            and 0.5 * (band[0] + band[1]) < g.ground_elevation_at(x)
            for x in [5.0 + 3.0 * k for k in range(21)])
        if holds:
            assert layer.name in named


def test_local_minima_listed_lowest_first_headline_lowest(weak_layer_search):
    """EM 4-2d(2), p. 4-2, and 4-2e(1)(b): each start's minimum is listed,
    lowest first, with its flags; the headline is the lowest."""
    res = weak_layer_search[True]
    scored = [m for m in res.local_minima if m["FOS"] is not None]
    assert len(scored) >= 2
    fos = [m["FOS"] for m in scored]
    assert fos == sorted(fos)
    assert res.critical.FOS == pytest.approx(fos[0], abs=1e-4)
    for m in scored:
        assert "flags" in m and m["FOS"] <= m["F_start"] + 1e-9


def _two_strata():
    """Strata 2 m and 5 m thick under a 6 m slope (the deepest 5 m)."""
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (10.0, 0.0), (22.0, 6.0), (40.0, 6.0)],
        soil_layers=[
            SlopeSoilLayer("top", top_elevation=6.0, bottom_elevation=-1.0,
                           gamma=19.0, phi=28.0, c_prime=8.0),
            SlopeSoilLayer("mid", top_elevation=-1.0, bottom_elevation=-3.0,
                           gamma=19.0, phi=25.0, c_prime=5.0),
            SlopeSoilLayer("deep", top_elevation=-3.0, bottom_elevation=-8.0,
                           gamma=19.0, phi=32.0, c_prime=10.0)])


def test_final_increment_is_ten_percent_of_the_thinnest_stratum():
    """EM 4-2e(2)(b), p. 4-7: the final increments no more than 10 to 25
    percent of the thinnest stratum; call 2: 10 %. With strata 2 m and 5 m
    thick (and the top layer cut by the ground), the result states a final
    increment of no more than 10 % of 2 m and names the stratum."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=1, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0), circle_check=False)
    fi = res.refinement["final_increment"]
    assert fi["stratum"] == "mid"
    assert fi["thickness"] == pytest.approx(2.0)
    assert fi["increment"] <= 0.10 * 2.0 + 1e-9
    assert res.refinement["refined"] is True


def test_refine_fraction_above_the_printed_range_is_refused():
    with pytest.raises(ValueError, match="10 to 25 percent"):
        search_critical_surface(
            _two_strata(), surface_type="noncircular", method="spencer",
            n_trials=20, seed=1, x_entry_range=(2.0, 12.0),
            x_exit_range=(22.0, 38.0), circle_check=False,
            refine_fraction=0.3)


def test_refinement_from_the_critical_circle_never_ends_above_it():
    """A refinement started from the critical circle moves only where F
    falls; the headline is no more than the circle's polyline."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=1, nx=6, ny=6, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0))
    circ = [m for m in res.local_minima if m["start"] == "critical circle"]
    assert circ and circ[0]["FOS"] is not None
    assert circ[0]["FOS"] <= circ[0]["F_start"] + 1e-9
    assert res.circle_check["checked"]
    if res.circle_check.get("FOS_as_polyline") is not None:
        assert res.critical.FOS <= res.circle_check["FOS_as_polyline"] + 1e-9
    assert res.critical.FOS <= circ[0]["F_start"] + 1e-9


def test_circle_check_false_gives_no_circle_starts_and_says_so():
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=40, seed=1, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0), circle_check=False)
    assert "circle_check=False" in res.refinement["circle_starts"]
    assert not any("circle" in m["start"] for m in res.local_minima)


def _embankment():
    """A two-faced embankment: left face 10 to 20, crest 20 to 30, right
    face 30 to 45."""
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (10.0, 0.0), (20.0, 8.0), (30.0, 8.0),
                        (45.0, 0.0), (60.0, 0.0)],
        soil_layers=[SlopeSoilLayer("fill", top_elevation=8.0,
                                    bottom_elevation=-6.0, gamma=19.0,
                                    phi=25.0, c_prime=6.0)])


def test_no_refined_surface_jumps_to_the_other_face():
    """EM 4-2e(1)(a), p. 4-3: an end does not cross the crest from its
    start, nor move onto the other face. A start with its upper end on
    the crest near the left crest and its lower end on the right face,
    refined with automatic windows (shifts free along the ground): no end
    of the refined surface lies on the left face."""
    g = _embankment()

    def score(pts):
        return _compute_fos(g, PolylineSlipSurface(points=pts), "spencer", 30)

    start = [(21.0, 8.0), (26.0, 4.0), (32.0, 2.0), (38.0, 3.7333)]
    start = SR.with_vertices(start)
    r = SR.refine_polyline(g, start, score, d_start=1.0, d_final=0.25,
                           windows=((0.0, 60.0), (0.0, 60.0)),
                           windows_given=(False, False))
    assert r["FOS"] <= score(start) + 1e-9
    xs = (r["points"][0][0], r["points"][-1][0])
    for x in xs:
        assert not (10.0 < x < 20.0)
    assert xs[0] >= 20.0 - 1e-9          # not across the left crest
    assert xs[1] >= 30.0 - 1e-9          # not across the right crest
    assert SR.end_allowed(SR._face_spans(g), 21.0, 22.0)
    assert not SR.end_allowed(SR._face_spans(g), 21.0, 19.0)
    assert not SR.end_allowed(SR._face_spans(g), 35.0, 29.0)


def test_no_refined_exit_is_steeper_than_eq_c32_or_its_start():
    """EM 4-2e(2)(a), p. 4-7, and Eq. C-32, p. C-34 (call 4): no shift
    makes the toe-end segment steeper than 45 - phi'/2 of the exit
    material, nor, where the start is steeper, than the start's exit."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=2, nx=6, ny=6, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0))
    for m in res.local_minima:
        if not m.get("refined") or m["points"] is None:
            continue
        pts = m["points"]
        toe = SR._toe_index(_two_strata(), pts)
        assert SR.exit_angle(pts, toe) <= m["toe_limit_deg"] + 1e-3
        assert m["toe_limit_deg"] >= m["eq_c32_deg"] - 1e-9


def test_drawdown_noncircular_search_refines_with_the_drawdown_f():
    """A drawdown noncircular search refines with the drawdown F: each
    listed minimum is the staged solve's value on its surface."""
    g = SlopeGeometry(
        surface_points=[(0.0, 0.0), (10.0, 0.0), (30.0, 10.0), (45.0, 10.0)],
        soil_layers=[SlopeSoilLayer("fill", top_elevation=10.0,
                                    bottom_elevation=-4.0, gamma=19.0,
                                    phi=28.0, c_prime=8.0)],
        gwt_points=[(0.0, 8.0), (45.0, 8.0)])
    res = search_rapid_drawdown(
        g, 8.0, 1.0, surface_type="noncircular", n_trials=20, seed=1,
        n_slices=30, x_entry_range=(2.0, 14.0), x_exit_range=(28.0, 44.0))
    s = res.search
    assert s.refinement.get("refined") is True
    scored = [m for m in s.local_minima if m["FOS"] is not None]
    assert scored
    top = scored[0]
    dd = rapid_drawdown_fos(g, 8.0, 1.0, slip_surface=PolylineSlipSurface(
        points=top["points"]), n_slices=30)
    assert dd.FOS == pytest.approx(top["FOS"], abs=2e-3)
    assert res.FOS <= top["FOS"] + 1e-9


def test_a_refinement_the_budget_stops_names_the_starts_it_left():
    """GM-123: the refinement is a stage of the time budget; one it stops
    names the starts left unrefined, never a silent stop."""
    res = search_critical_surface(
        _weak_layer_slope(), surface_type="noncircular", method="spencer",
        n_trials=50, seed=1, circle_check=False, time_budget_s=0.05,
        **_WINDOWS)
    assert res.refinement["unrefined"]
    stops = (res.search_ranges.get("time_budget") or {})
    text = repr(stops)
    assert "noncircular refinement" in text and "unrefined" in text


def test_default_circular_search_lists_its_grid_minima_and_is_unrefined_by_7a():
    """A circular search lists its separate grid minima (a record) and has
    no noncircular refinement; its values are those of the grid search."""
    res = search_critical_surface(_two_strata(), method="bishop", nx=6, ny=6)
    assert res.local_minima
    assert not res.refinement
    first = res.local_minima[0]
    assert first["refined"] is True
    for m in res.local_minima:
        if not m["refined"]:
            assert m["why_unrefined"]
