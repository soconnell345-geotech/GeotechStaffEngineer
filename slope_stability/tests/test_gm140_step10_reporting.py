"""GM-140 step 10: a search's result and report show what the print asks
an engineer to be shown. Made-up slopes; no value is computed here that
the search did not store.

* D46: contours of the minimum factor of safety over the trial centres,
  FHWA NHI-06-088 Fig. 6-13, p. 6-23; GEC 11 Vol. II p. 9-24; Duncan,
  Buchignani and De Wet (1987), Fig. 28 (the contours should enclose the
  minimum).
* D47: "ordered output and plotting capability for the failure surface of
  10 minimum safety factors", FHWA NHI-06-088 p. 6-49; EM 1110-2-1902
  4-3b(9)(b), p. 4-8 (the critical surface and the other significant
  surfaces with their factors of safety, for circles centre and radius).
* EM 1110-2-1902 4-2d(2), p. 4-2: the engineer checks slice weights,
  strengths and pore pressures at the bottoms of slices (the slice table
  first).
"""

import pytest

from slope_stability.analysis import search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.report_views import centre_contours, lowest_surfaces
from slope_stability.results import SearchResult


def _geom():
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (20, 5), (30, 5)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=10,
                                    bottom_elevation=0, gamma=18,
                                    c_prime=10, phi=25)])


def _search(x_range, y_range, **kw):
    return search_critical_surface(
        _geom(), method="bishop", nx=5, ny=5, x_range=x_range,
        y_range=y_range, **kw)


def _synthetic(values):
    """A 3 x 3 grid search result with the given per-centre minima
    (rows by centre height, bottom first)."""
    grid = []
    for iy, row in enumerate(values):
        for ix, f in enumerate(row):
            grid.append({"xc": float(ix), "yc": float(iy), "R": 5.0,
                         "FOS": f, "node": (ix, iy)})
    return SearchResult(grid_fos=grid, centre_grid={
        "x_range": (0.0, 2.0), "y_range": (0.0, 2.0), "nx": 3, "ny": 3})


def test_closed_when_the_lowest_inside_is_below_every_edge_value():
    """D46: a contour between the lowest inside value and the lowest edge
    value encloses the minimum without reaching the grid's edge."""
    c = centre_contours(_synthetic([[2.0, 1.9, 2.0],
                                    [1.8, 1.5, 1.9],
                                    [2.0, 2.0, 2.1]]))
    assert c["closed"] is True
    assert c["minimum"] == {"FOS": 1.5, "xc": 1.0, "yc": 1.0,
                            "node": (1, 1), "on_edge": False}
    assert c["edge_minimum"]["FOS"] == 1.8
    assert c["fos"][1][1] == 1.5
    assert "closes inside" in c["statement"]


def test_not_closed_when_an_edge_value_is_at_or_below_the_inside():
    """D46: the lowest value on the grid's edge: no contour encloses the
    minimum; the statement names the edge to widen."""
    c = centre_contours(_synthetic([[2.0, 1.9, 2.0],
                                    [1.6, 1.7, 1.9],
                                    [2.0, 2.0, 2.1]]))
    assert c["closed"] is False
    assert c["minimum"]["on_edge"] is True
    assert "left" in c["edge_minimum"]["edge"]
    assert "does not close" in c["statement"]
    assert "x_range / y_range" in c["statement"]


def test_screen_values_and_unsolved_centres_are_not_contoured():
    """A centre a screen alone scored, or one the method solved no circle
    about (999.9), has no value on the contour grid."""
    r = _synthetic([[2.0, 1.9, 2.0], [1.8, 1.5, 1.9], [2.0, 999.9, 2.1]])
    r.grid_fos[0]["method"] = "bishop"     # a screen value only
    c = centre_contours(r)
    assert c["fos"][0][0] is None and c["fos"][2][1] is None
    assert c["n_scored"] == 7 and c["n_screen_only"] == 1


def test_search_with_its_minimum_inside_the_grid():
    r = _search((10, 25), (10, 22))
    c = centre_contours(r)
    assert c["closed"] is True and c["minimum"]["on_edge"] is False
    assert len(c["x"]) == len(c["y"]) == 5
    # every value contoured is one the search stored for that centre
    stored = {g["node"]: g["FOS"] for g in r.grid_fos
              if "node" in g and g["FOS"] < 900}
    for (ix, iy), f in stored.items():
        assert c["fos"][iy][ix] == f
    assert r.to_dict()["centre_contours"]["closed"] is True


def test_search_with_its_minimum_on_the_edge_of_the_grid():
    r = _search((10, 25), (10, 15))
    c = centre_contours(r)
    assert c["closed"] is False and c["minimum"]["on_edge"] is True
    assert "top" in c["edge_minimum"]["edge"]
    assert "Centre grid: The lowest contour does not close" in r.summary()


def test_no_contours_for_a_search_without_a_centre_grid():
    r = search_critical_surface(_geom(), method="bishop", nx=4, ny=4,
                                surface_type="entry_exit")
    assert centre_contours(r) is None
    assert "centre_contours" not in r.to_dict()


def test_the_ordered_list_is_sorted_and_complete():
    """D47: the ten lowest surfaces in order, each with its factor of
    safety, family, method and geometry; the first is the critical
    surface."""
    r = _search((10, 25), (10, 22))
    low = lowest_surfaces(r)
    assert len(low) == 10
    assert [s["rank"] for s in low] == list(range(1, 11))
    assert [s["FOS"] for s in low] == sorted(s["FOS"] for s in low)
    for s in low:
        assert s["family"] and s["method"] == "bishop"
        assert {"xc", "yc", "radius"} <= set(s) or s.get("points")
    assert low[0]["FOS"] == pytest.approx(r.critical.FOS, abs=1e-4)
    # nothing the search stored below the tenth is left out
    # (one surface listed once: keyed by its centre and radius)
    pool = {}
    for g in r.grid_fos:
        if g["FOS"] < 900 and not g.get("method"):
            key = (round(g["xc"], 2), round(g["yc"], 2), round(g["R"], 2))
            pool[key] = min(pool.get(key, 9e9), g["FOS"])
    for v in r.family_surfaces.values():
        key = (round(v["xc"], 2), round(v["yc"], 2), round(v["radius"], 2))
        pool[key] = min(pool.get(key, 9e9), v["FOS"])
    assert low[-1]["FOS"] <= sorted(pool.values())[9] + 1e-9
    assert r.to_dict()["lowest_surfaces"] == low


def test_family_minima_are_in_the_list_with_their_geometry():
    r = _search((10, 25), (10, 22))
    for family, s in r.family_surfaces.items():
        assert s["FOS"] == r.candidates[family]
        assert "radius" in s or "points" in s


def test_the_report_puts_the_slice_table_first():
    """EM 1110-2-1902 4-2d(2), p. 4-2: the report's first table is the
    slice table (weights, strengths and pore pressures at the bases)."""
    from calc_package.data_model import TableData
    from slope_stability.analysis import analyze_slope
    from slope_stability.calc_steps import get_calc_steps
    geom = _geom()
    res = analyze_slope(geom, xc=15, yc=18, radius=13,
                        include_slice_data=True)
    sections = get_calc_steps(res, {"geom": geom})
    first = next(it for sec in sections for it in sec.items
                 if isinstance(it, TableData))
    assert sections[0].title == "Slice Force Table"
    assert first.title == "Per-Slice Forces"
    assert "W (kN/m)" in first.headers and "u (kPa)" in first.headers


def test_the_report_carries_the_contours_and_the_ordered_list():
    from calc_package.data_model import TableData
    from slope_stability.analysis import analyze_slope
    from slope_stability.calc_steps import get_calc_steps, get_figures
    geom = _geom()
    r = _search((10, 25), (10, 22))
    res = analyze_slope(geom, xc=r.critical.xc, yc=r.critical.yc,
                        radius=r.critical.radius, include_slice_data=True)
    analysis = {"geom": geom, "search": r}
    titles = [it.title for sec in get_calc_steps(res, analysis)
              for it in sec.items if isinstance(it, TableData)]
    assert "Minimum FOS over the Circle Centres" in titles
    assert "The 10 Lowest Surfaces" in titles
    pytest.importorskip("matplotlib")
    from slope_stability.calc_steps import CONTOUR_FIGURE_TITLE
    figs = [f.title for f in get_figures(res, analysis)]
    assert CONTOUR_FIGURE_TITLE in figs     # the contours are drawn


# ---------------------------------------------------------------------------
# The fix round after the review: D1 (closure), D5 (no contour), Q1 (the
# screen's values), Q3 (the pool) and the governing centre off the grid
# ---------------------------------------------------------------------------

def _grid_with_screen(method_values, screen_values, screen_only=()):
    """A 3 x 3 screened grid: each centre the method re-scored carries its
    value (999.9: no solution) and the screen's (``screen_FOS``); a centre
    in ``screen_only`` has the screen's value alone."""
    grid = []
    for iy in range(3):
        for ix in range(3):
            if (ix, iy) in screen_only:
                grid.append({"xc": float(ix), "yc": float(iy), "R": 5.0,
                             "FOS": screen_values[iy][ix], "node": (ix, iy),
                             "method": "bishop"})
            else:
                grid.append({"xc": float(ix), "yc": float(iy), "R": 5.0,
                             "FOS": method_values[iy][ix], "node": (ix, iy),
                             "screen_FOS": screen_values[iy][ix],
                             "screen_method": "bishop"})
    return SearchResult(grid_fos=grid, named_method="spencer", centre_grid={
        "x_range": (0.0, 2.0), "y_range": (0.0, 2.0), "nx": 3, "ny": 3})


def test_cannot_be_judged_when_a_centre_next_to_the_low_ones_has_none():
    """D1: closed only where every centre next to the centres below the
    midway level has a value; one without leaves it "cannot be judged"
    (closed None), never closed."""
    c = centre_contours(_synthetic([[2.0, 1.9, 2.0],
                                    [1.8, 1.5, 1.9],
                                    [2.0, 999.9, 2.1]]))
    assert c["closed"] is None and c["verdict"] == "unjudged"
    assert "cannot be judged" in c["statement"]
    assert "8 of 9 centres" in c["statement"]
    assert c["minimum"]["FOS"] == 1.5


def test_cannot_be_judged_without_a_value_inside_the_edge():
    c = centre_contours(_synthetic([[2.0, 1.9, 2.0],
                                    [1.8, 999.9, 1.9],
                                    [2.0, 2.0, 2.1]]))
    assert c["closed"] is None
    assert "no centre inside the grid's edge has one" in c["statement"]


def test_the_closed_contour_is_drawn_at_the_midway_level():
    """The level the statement judges closed (midway between the lowest
    inside and the lowest edge value) is among those drawn; every level
    drawn is spanned by a cell of four valued centres."""
    from slope_stability.report_views import contour_levels
    c = centre_contours(_synthetic([[2.0, 1.9, 2.0],
                                    [1.8, 1.5, 1.9],
                                    [2.0, 2.0, 2.1]]))
    assert c["closed"] is True and c["level"] == pytest.approx(1.65)
    levels = contour_levels(c)
    assert 1.65 in levels
    assert all(1.5 < v < 2.1 for v in levels)


def test_the_screens_values_are_contoured_where_the_methods_cannot_be():
    """Q1: where the method has too few values to judge the contours and
    the search screened its circles, the screen's values are contoured,
    named "Bishop's simplified method (the screen)" in every line."""
    from slope_stability.report_views import SCREEN_TEXT
    r = _grid_with_screen(
        [[2.2, 999.9, 2.2], [999.9, 999.9, 999.9], [2.3, 999.9, 2.4]],
        [[2.0, 1.9, 2.0], [1.8, 1.5, 1.9], [2.0, 2.0, 2.1]],
        screen_only={(1, 0)})
    c = centre_contours(r)
    assert c["basis"] == "screen" and c["values_by"] == SCREEN_TEXT
    assert c["closed"] is True and c["fos"][1][1] == 1.5
    assert c["n_by_method"] == 4
    assert SCREEN_TEXT in c["statement"]
    assert "too few to judge the contours by" in c["statement"]


def test_the_methods_values_stay_where_the_screen_is_no_better():
    """A screen as short of values as the method shows nothing more: the
    method's values stay, and the contours cannot be judged."""
    r = _synthetic([[2.0, 1.9, 2.0], [1.8, 1.5, 1.9], [2.0, 999.9, 2.1]])
    r.grid_fos[0]["method"] = "bishop"
    c = centre_contours(r)
    assert c["basis"] == "method" and c["closed"] is None


def test_the_statement_says_when_the_governing_centre_is_off_the_grid():
    from slope_stability.analysis import analyze_slope
    r = _synthetic([[2.0, 1.9, 2.0], [1.8, 1.5, 1.9], [2.0, 2.0, 2.1]])
    r.critical = analyze_slope(_geom(), xc=15, yc=18, radius=13)
    r.governing_surface = "toe_circle"
    c = centre_contours(r)
    assert "lies outside the centre grid" in c["outside_line"]
    assert c["outside_line"] in c["statement"]
    assert "(15.00, 18.00)" in c["outside_line"]


def test_no_contour_drawn_is_said_in_the_figure_title_and_caption():
    """D5: without a contour the figure is titled for the lowest surfaces
    and its caption says why no contour is drawn."""
    from slope_stability.calc_steps import (
        CONTOUR_FIGURE_TITLE, LOWEST_FIGURE_TITLE, contour_figure_text)
    r = search_critical_surface(_geom(), method="bishop", nx=4, ny=4,
                                surface_type="entry_exit")
    title, caption = contour_figure_text(r)
    assert title == LOWEST_FIGURE_TITLE
    assert ("No contour of the minimum FOS over the circle centres is "
            "drawn: the search has no grid of circle centres") in caption
    sparse = _synthetic([[2.0, 999.9, 2.0], [999.9, 1.5, 999.9],
                         [2.0, 999.9, 2.1]])
    title, caption = contour_figure_text(sparse)
    assert title == LOWEST_FIGURE_TITLE
    assert "no four neighbouring centres all have one" in caption
    closed = _search((10, 25), (10, 22))
    assert contour_figure_text(closed)[0] == CONTOUR_FIGURE_TITLE


def test_the_lowest_surfaces_come_from_every_familys_ten_lowest():
    """Q3: the pool is each family's ten lowest by the method that scored
    it (more than one circle about a centre allowed, none a screen's
    value), the geometry as scored: each surface listed, analysed again
    by its method, gives its listed factor of safety."""
    from slope_stability.analysis import analyze_slope
    from slope_stability.slip_surface import CircularSlipSurface
    r = _search((10, 25), (10, 22))
    assert set(r.family_lowest) == set(r.candidates)
    for family, kept in r.family_lowest.items():
        assert 0 < len(kept) <= 10
        assert [s["FOS"] for s in kept] == sorted(s["FOS"] for s in kept)
        assert kept[0]["FOS"] == pytest.approx(r.candidates[family],
                                               abs=1e-4)
    low = lowest_surfaces(r)
    assert len(low) == 10
    scored_by = {(m["family"], m["method"]) for m in r.candidate_methods}
    assert all((s["family"], s["method"]) in scored_by for s in low)
    for s in low:
        slip = CircularSlipSurface(s["xc"], s["yc"], s["radius"],
                                   *(s.get("arc") or ()))
        again = analyze_slope(_geom(), slip_surface=slip, method=s["method"])
        assert again.FOS == pytest.approx(s["FOS"], abs=2e-4), s


# ---------------------------------------------------------------------------
# Items 7, 8 and 11: the statements the report prints
# ---------------------------------------------------------------------------

def _sand_geom():
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (25, 0), (40, 0)],
        soil_layers=[SlopeSoilLayer(name="sand", top_elevation=10,
                                    bottom_elevation=-10, gamma=19,
                                    c_prime=0.0, phi=32)])


def test_a_circle_through_cohesionless_soil_at_the_face_says_so():
    """The module audit's row 9: a single circle through c' = 0 soil at the
    face overestimates against the infinite-slope mechanism, which the
    search scores with planes. A statement only."""
    from slope_stability.analysis import analyze_slope
    res = analyze_slope(_sand_geom(), xc=20, yc=20, radius=20)
    said = [w for w in res.warnings if "infinite-slope mechanism" in w]
    assert len(said) == 1
    assert "c' = 0" in said[0] and "not changed" in said[0]
    clay = analyze_slope(_geom(), xc=15, yc=18, radius=13)
    assert not [w for w in clay.warnings if "infinite-slope" in w]


def test_the_report_names_the_method_of_each_family_and_a_stand_in():
    """Step 6's statements in the report: beside each family's minimum the
    method that scored it; a plane family under Bishop's simplified method
    is scored by Spencer's method, with the reason."""
    from slope_stability.calc_steps import _family_methods_table
    r = search_critical_surface(_sand_geom(), method="bishop", nx=4, ny=4)
    t = _family_methods_table(r)
    assert t is not None and t.title == "The Method Each Family Was Scored With"
    fams = {row[0]: row for row in t.rows}
    assert len(fams) == len(r.candidate_methods)
    planes = [row for row in t.rows if row[2] == "Spencer's method"]
    assert planes and all("stand-in for Bishop's simplified method"
                          in row[4] for row in planes)
    assert all("circular slip surfaces" in row[4]
               or "centre of a circle" in row[4] for row in planes)


def test_the_report_words_the_unsolved_line_the_janbu_f0_and_the_checks():
    """Janbu's f0 where Janbu's method stands in (applied, not found in
    print, its value, the uncorrected FOS); the unsolved line as the result
    words it (UFC 3-220-10 p. 362); the circle check; an Ordinary Method of
    Slices search crossed by supports refused with nothing standing in."""
    from slope_stability.calc_steps import _family_methods_table
    r = SearchResult(
        named_method="spencer", method_used="janbu",
        named_method_lowest={"FOS": None, "note": "Spencer's method "
                             "scored no family of this search"},
        candidate_methods=[
            {"family": "noncircular", "method": "janbu", "f_interslice":
             None, "FOS": 1.234, "named": False, "reason": "spencer's "
             "method solved none of the family's surfaces",
             "janbu_f0": 1.051, "FOS_uncorrected": 1.174}],
        screen_unsolved={"n": 2, "k": 5, "N": 7, "lowest_screen_FOS": 1.1,
                         "by_family": {}},
        circle_check={"checked": False, "note": "no circle"},
        no_solution=[{"family": "circular", "method": "fellenius", "n": 3,
                      "with_supports": 3}])
    t = _family_methods_table(r)
    row = t.rows[0]
    assert "f0 = 1.051 applied, not found in print" in row[4]
    assert "uncorrected FOS 1.174" in row[4]
    assert ("Of the 5 circle(s) re-scored from the screen's lowest, 2 have "
            "no solution by Spencer's method; the screen scored 7 circle(s) "
            "below the reported minimum") in t.notes
    assert "lowest screen value 1.100 (UFC 3-220-10 p. 362)" in t.notes
    assert "method_used (the governing surface): Janbu's" in t.notes
    assert "Not checked against a circle" in t.notes
    assert "nothing standing in" in t.notes
