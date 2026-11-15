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
    that holds ground below the ground surface in the windows' span. The
    fix round's D6: the central block runs on the mid-line between the
    layer's bottom and its top cut at the ground, so the upper layer, whose
    top the face cuts, gets a wedge too; a layer with none says why."""
    g = _weak_layer_slope()
    rec = weak_layer_search[True].refinement
    named = {w["layer"] for w in rec["wedges"]}
    for layer in g.soil_layers:
        holds = any(
            (band := _layer_band_at(g, layer, x)) is not None
            and band[0] < min(band[1], g.ground_elevation_at(x)) - 1e-6
            for x in [5.0 + 3.0 * k for k in range(21)])
        if holds:
            assert layer.name in named
    upper = [w for w in rec["wedges"] if w["layer"] == "upper"]
    assert upper and upper[0]["FOS"] is not None
    for w in rec["wedges"]:
        assert w["FOS"] is not None or w["why"]


def test_the_wedge_ends_are_halved_down_to_the_final_increment(
        weak_layer_search):
    """The fix round's ruling 10: after the 21-position grid the best
    wedge's two ends are halved, one at a time, the inclinations fixed,
    down to the final increment (its last level), in the wedge stage alone
    (refine=False) as in the refined search."""
    for refine in (False, True):
        rec = weak_layer_search[refine].refinement
        d_final = rec["final_increment"]["increment"]
        for w in rec["wedges"]:
            if w["FOS"] is None:
                continue
            levels = w["end_levels"]
            assert levels == sorted(levels, reverse=True)
            assert levels[-1] == pytest.approx(d_final, abs=1e-6)
            assert levels[0] <= 0.5 * 60.0 / 20 + 1e-9    # half the spacing


def test_the_wedge_stage_alone_says_its_inclinations_are_fixed(
        weak_layer_search):
    """The fix round's ruling 10: a result of the wedge stage alone carries
    a warning that the inclinations are fixed at EM C-1g(2)'s values and
    that the stage is not a minimum over inclinations or shape; the
    refined search does not."""
    alone = [w for w in weak_layer_search[False].warnings
             if w.startswith("The wedge stage alone")]
    assert len(alone) == 1
    assert "C-1g(2)" in alone[0] and "not a minimum over the inclinations" \
        in alone[0]
    assert not [w for w in weak_layer_search[True].warnings
                if w.startswith("The wedge stage alone")]


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
        assert "flags" in m
        assert m["F_start"] is None or m["FOS"] <= m["F_start"] + 1e-9


def test_each_start_is_scored_as_it_stands_and_its_minimum_is_no_more():
    """The fix round's ruling 9(b), test (i): every start is scored as it
    stands before it is refined, and its minimum is the lower of the two,
    labelled which; so no start's minimum is above the start as it
    stands (made-up slope)."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=3, nx=6, ny=6, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0))
    scored = [m for m in res.local_minima if m["FOS"] is not None]
    assert scored
    for m in scored:
        assert m["F_start"] is not None
        assert m["FOS"] <= m["F_start"] + 1e-9
        assert m["minimum"] in ("as it stands", "refined")
        if m["minimum"] == "refined":
            assert m["FOS"] == pytest.approx(m["F_refined"], abs=1e-4)
        else:
            assert m["FOS"] == pytest.approx(m["F_start"], abs=1e-4)


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
    increment of no more than 10 % of 2 m and names the stratum. The fix
    round's ruling 9(a): the final increment is the lesser of that and a
    quarter of the starting increment (0.10 x the 6 m relief = 0.6 m), and
    the starting increment is never raised to meet it."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=1, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0), circle_check=False)
    fi = res.refinement["final_increment"]
    assert fi["stratum"] == "mid"
    assert fi["thickness"] == pytest.approx(2.0)
    assert fi["increment"] <= 0.10 * 2.0 + 1e-9
    assert res.refinement["start_increment"] == pytest.approx(0.6)
    assert fi["increment"] == pytest.approx(min(0.2, 0.6 / 4))
    assert res.refinement["refined"] is True


def test_a_slope_of_one_thick_stratum_still_halves_at_least_twice():
    """The fix round's ruling 9(a) (the review's Q2): where the thinnest
    stratum is thick, a quarter of the starting increment sets the final
    increment, so the refinement halves at least twice, and choices states
    the rule as the module's call."""
    g = _embankment()
    res = search_critical_surface(
        g, surface_type="noncircular", method="spencer", n_trials=20,
        seed=1, x_entry_range=(2.0, 12.0), x_exit_range=(18.0, 30.0),
        circle_check=False)
    rec = res.refinement
    fi = rec["final_increment"]
    assert fi["governed_by"] == "a quarter of the starting increment"
    assert fi["increment"] == pytest.approx(rec["start_increment"] / 4,
                                            abs=1e-4)
    assert fi["fraction_of_stratum"] > fi["increment"]
    levels = [m["levels"] for m in res.local_minima if m.get("levels")]
    assert levels and all(len(lv) >= 3 for lv in levels)
    text = [c for c in res.choices
            if c["what"] == "final shift increment of the noncircular "
                            "refinement"]
    assert text and "a quarter of the starting increment" in \
        text[0]["chosen"]


def test_refine_fraction_above_the_printed_range_is_refused():
    with pytest.raises(ValueError, match="10 to 25 percent"):
        search_critical_surface(
            _two_strata(), surface_type="noncircular", method="spencer",
            n_trials=20, seed=1, x_entry_range=(2.0, 12.0),
            x_exit_range=(22.0, 38.0), circle_check=False,
            refine_fraction=0.3)


def test_each_circle_start_reports_three_values_and_says_where_above():
    """The fix round's ruling 9(b), test (ii): each circle start reports
    the circle's own F, its slice-base polyline's F on the search's slices
    (the start as it stands) and the refined F; its minimum is the lower
    of the last two, so the headline is never above the circle's polyline.
    Where the headline is above the circle's own F the result says so with
    both values (step 6's statement, worded for the refined search)."""
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=60, seed=1, nx=6, ny=6, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0))
    circles = [m for m in res.local_minima if "F_circle" in m]
    assert any(m["start"] == "critical circle" for m in circles)
    for m in circles:
        assert m["F_circle"] is not None and m["F_start"] is not None
        assert m["F_refined"] is not None and m["refined"] is True
        assert m["FOS"] == pytest.approx(min(m["F_start"], m["F_refined"]),
                                         abs=1e-4)
    cc = res.circle_check
    assert cc["checked"] and cc["FOS_as_polyline"] is not None
    # (circle_check records the polyline's F to 4 decimals)
    assert res.critical.FOS <= cc["FOS_as_polyline"] + 5e-5
    circle = cc["circle"]["FOS"]
    if res.critical.FOS > circle + 1e-9:
        said = [w for w in res.warnings if "above the critical circle" in w]
        assert said and f"{circle:.4f}" in said[0]
        assert f"{res.critical.FOS:.4f}" in said[0]


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


def _drawdown_slope():
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (10.0, 0.0), (30.0, 10.0), (45.0, 10.0)],
        soil_layers=[SlopeSoilLayer("fill", top_elevation=10.0,
                                    bottom_elevation=-4.0, gamma=19.0,
                                    phi=28.0, c_prime=8.0)],
        gwt_points=[(0.0, 8.0), (45.0, 8.0)])


def test_drawdown_refine_false_runs_the_sampler_alone():
    """The fix round's D5: search_rapid_drawdown passes refine,
    refine_fraction and which windows were given to its noncircular kinds,
    so refine=False runs the sampler alone, and the result's choices carry
    the refinement's calls (refinement_choices)."""
    kw = dict(surface_type="noncircular", n_trials=20, seed=1, n_slices=30,
              x_entry_range=(2.0, 14.0), x_exit_range=(28.0, 44.0))
    off = search_rapid_drawdown(_drawdown_slope(), 8.0, 1.0, refine=False,
                                **kw)
    assert off.search.refinement == {"refined": False}
    assert not off.search.local_minima
    whats = {c["what"]: c for c in off.search.choices}
    assert whats["noncircular refinement"]["chosen"] == "off"
    on = search_rapid_drawdown(_drawdown_slope(), 8.0, 1.0,
                               refine_fraction=0.2, **kw)
    whats = {c["what"]: c for c in on.search.choices}
    assert whats["noncircular refinement"]["chosen"] == "on"
    assert "final shift increment of the noncircular refinement" in whats
    assert on.search.refinement["final_increment"]["fraction"] == 0.2
    assert on.FOS <= off.FOS + 1e-9


def test_drawdown_noncircular_search_refines_with_the_drawdown_f():
    """A drawdown noncircular search refines with the drawdown F: each
    listed minimum is the staged solve's value on its surface."""
    g = _drawdown_slope()
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
    # the fix round's D3 / ruling 9(b): every start is scored as it stands
    # first, so the result has a surface, and a start the budget left is
    # listed not refined with its value as it stands
    assert res.critical is not None
    left = [m for m in res.local_minima if m.get("why_unrefined")]
    for m in left:
        assert "not refined" in m["why_unrefined"]
        if m["FOS"] is not None:
            assert m["FOS"] == pytest.approx(m["F_start"], abs=1e-4)


def test_a_wedge_stage_the_budget_stops_records_the_stop():
    """The fix round's D3: a stop of the wedge stage is never silent. With
    refine=False and a budget too short for it, search_ranges.time_budget
    and the warning name the stage, the layers and the positions not
    built, and each layer not built says why."""
    res = search_critical_surface(
        _weak_layer_slope(), surface_type="weak_layer", method="spencer",
        n_slices=30, refine=False, circle_check=False, time_budget_s=0.05,
        **_WINDOWS)
    stops = (res.search_ranges.get("time_budget") or {}).get("stopped", [])
    wedge = [s for s in stops if s["stage"] == "wedge stage"]
    assert wedge and wedge[0]["unit"] == "low-end positions"
    assert "weak" in wedge[0]["not_covered"]
    assert any("The wedge stage stopped" in w for w in res.warnings)
    left = [w for w in res.refinement["wedges"] if w["FOS"] is None]
    assert left and all("time budget" in w["why"] for w in left)


def test_a_none_circle_value_is_no_value():
    """The fix round's D1: a grid minimum whose F is None (a walk that
    found no solution, or a grid valued by the Bishop screen) is kept as a
    circle start with F None and sorted last; it does not break the sort
    (made-up records)."""
    class _R:
        pass
    r = _R()
    r.critical = SR._Circle(20.0, 20.0, 15.0, 8.0, 32.0, FOS=1.40)
    r.local_minima = [
        {"start": "grid minimum 1", "FOS_grid": 1.41, "xc": 21.0, "yc": 20.0,
         "radius": 15.0, "refined": True, "FOS": None,
         "refined_centre": {"xc": 21.5, "yc": 20.0, "radius": 15.0}},
        {"start": "grid minimum 2", "FOS_grid": 1.6, "xc": 30.0, "yc": 25.0,
         "radius": 18.0, "refined": False, "FOS": None,
         "values_by": "the Bishop screen", "why_unrefined": "x"},
        {"start": "grid minimum 3", "FOS_grid": 1.5, "xc": 26.0, "yc": 22.0,
         "radius": 16.0, "refined": True, "FOS": 1.45,
         "refined_centre": {"xc": 25.0, "yc": 22.0, "radius": 16.5}}]
    r.family_surfaces = {"toe_circle": {"xc": 18.0, "yc": 18.0,
                                        "radius": 14.0, "FOS": None}}
    got = SR.circle_starts_of(r)
    labels = [label for label, _c in got]
    assert labels[:3] == ["critical circle", "grid minimum 1",
                          "grid minimum 3"]
    by = dict(got)
    assert by["grid minimum 1"].FOS == 1.41            # the node's own F
    assert by["grid minimum 3"].radius == 16.5          # where the walk ended
    assert by["grid minimum 2"].FOS is None
    assert by["toe circle"].FOS is None
    assert set(labels[3:]) == {"grid minimum 2", "toe circle"}


@pytest.fixture(scope="module")
def b14_searches():
    """Section 11's B14 (made up: a 10 m 1:1 cut, 4 m of weak soil over
    rock), Spencer: random with the default windows, and differential
    evolution directed at the weak layer's outcrop."""
    from slope_stability.tests.test_search_ranges import _weak_over_rock
    return {
        "noncircular": search_critical_surface(
            _weak_over_rock(), method="spencer", surface_type="noncircular",
            seed=1),
        "noncircular_de": search_critical_surface(
            _weak_over_rock(), method="spencer",
            surface_type="noncircular_de", x_entry_range=(25.5, 28.0),
            x_exit_range=(29.0, 40.0), seed=1)}


@pytest.mark.parametrize("kind", ["noncircular", "noncircular_de"])
def test_b14_ends_no_more_than_its_circle_or_says_so(b14_searches, kind):
    """The fix round's ruling 8: inside the refinement the span floor is
    the lesser of the module's floor and the start's own span, so B14's
    critical circle (4.66 m long, under the 5 m floor) is a start, scored
    as it stands and refined. The result is no more than the circle search
    by the same method or, where above it, says so with both values
    (ruling 9's test (ii)). Master read 6.366 (random) and 0.897 (DE)."""
    res = b14_searches[kind]
    cc = res.circle_check
    circle = cc["circle"]["FOS"]
    assert circle == pytest.approx(0.8822, abs=5e-4)    # circular, unchanged
    start = [m for m in res.local_minima if m["start"] == "critical circle"]
    assert start and start[0]["F_start"] is not None
    assert start[0]["F_circle"] == pytest.approx(circle, abs=1e-4)
    # (circle_check records the polyline's F to 4 decimals)
    assert res.critical.FOS <= cc["FOS_as_polyline"] + 5e-5
    if res.critical.FOS > circle + 1e-9:
        said = [w for w in res.warnings if "above the critical circle" in w]
        assert said and f"{circle:.4f}" in said[0]
        assert f"{res.critical.FOS:.4f}" in said[0]


def test_a_start_shorter_than_the_span_floor_keeps_its_own_span():
    """The fix round's ruling 8: inside the refinement the span floor is
    the lesser of the module's floor (half the slope's height without
    given windows) and the start's own span."""
    from slope_stability.search import (module_span_floor,
                                        start_span_floor)
    g = _weak_layer_slope()
    assert module_span_floor(g) == pytest.approx(5.0)
    short = [(20.0, 0.0), (21.5, -0.6), (23.0, -0.4), (24.0, 2.0)]
    floor = start_span_floor(g, short, 30)
    assert floor is not None and floor < 5.0
    assert floor == pytest.approx(4.0, abs=0.05)
    f_screen = _compute_fos(g, PolylineSlipSurface(points=short), "spencer",
                            30)
    f_own = _compute_fos(g, PolylineSlipSurface(points=short), "spencer", 30,
                         span_floor=floor)
    assert f_screen >= 900 and f_own < 900


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


def test_a_seed_surface_given_as_points_is_one_of_the_starts():
    """EM 4-2e(2), p. 4-7: the caller's initial surface is a start of the
    refinement, given to search_critical_surface as a list of points."""
    seed = [(4.0, 0.0), (12.0, -2.0), (20.0, 1.0), (26.0, 6.0)]
    res = search_critical_surface(
        _two_strata(), surface_type="noncircular", method="spencer",
        n_trials=20, seed=1, x_entry_range=(2.0, 12.0),
        x_exit_range=(22.0, 38.0), circle_check=False, seed_surface=seed)
    starts = [m["start"] for m in res.local_minima]
    assert "the caller's surface (seed_surface)" in starts
