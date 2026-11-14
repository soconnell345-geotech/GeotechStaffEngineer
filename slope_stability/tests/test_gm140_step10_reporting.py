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
    figs = [f.title for f in get_figures(res, analysis)]
    assert ("Minimum FOS over the Circle Centres and the Ten Lowest "
            "Surfaces") in figs
