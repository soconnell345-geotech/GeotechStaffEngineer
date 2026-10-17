"""The time budget of a critical-surface search, and its Bishop screen
(backlog GM-123).

A 30 x 30 entry/exit grid with 60 slices scored with Spencer took 957 s in
a live run, and a second one ran past the agent's turn, so the user got no
answer. A search given a budget stops at it and reports the lowest surface in
what it covered, what that was, and what it did not cover. The library has
no budget unless given one, so its results never depend on the machine's
speed; the agent's tools give 180 s (funhouse_agent tests). The budgets
here are tiny so the tests stay fast.
"""

import json
from pathlib import Path

import pytest

from slope_stability import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import search_critical_surface
from slope_stability import search
from slope_stability.search import search_budget


def _geom():
    return SlopeGeometry(
        surface_points=[(0, 5), (5, 5), (15, 15), (20, 15)],
        soil_layers=[SlopeSoilLayer(name="m", top_elevation=15,
                                    bottom_elevation=0, gamma=18, phi=30,
                                    c_prime=10)])


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def _spencer_grid(budget):
    return search_critical_surface(
        _geom(), surface_type="entry_exit", nx=12, ny=12, method="spencer",
        n_slices=30, time_budget_s=budget)


def test_budget_hit_returns_best_so_far_with_coverage():
    r = _spencer_grid(0.3)
    tb = r.search_ranges["time_budget"]
    assert tb["time_budget_s"] == 0.3
    stop = tb["stopped"][0]
    assert stop["stage"] == "entry/exit grid"
    assert stop["unit"] == "entry/exit pairs"
    assert 1 <= stop["scored"] < stop["of"] == 144
    assert "coarsest pass" in stop["covered"] or "every" in stop["covered"]
    assert stop["not_covered"].startswith(f"{144 - stop['scored']} of the 144")
    # The lowest surface in what was covered is reported.
    assert r.critical is not None
    assert 0 < len(r.grid_fos) <= stop["scored"]
    assert r.critical.FOS <= min(e["FOS"] for e in r.grid_fos) + 1e-4
    w = r.warnings[0]
    assert w.startswith("Cut short by its time budget: the reported FOS may "
                        "be too high (unconservative)")
    assert "The budget was 0.3 s" in w
    assert "Covered:" in w and "Not scored:" in w
    # A rigorous method: the Bishop screen is the first lever named.
    assert w.index("(bishop_screen") < w.index("raise time_budget_s")
    assert "raise time_budget_s (at most 600 s" in w
    c = _by_what(r)["time budget of the search"]
    assert c["chosen"] == "0.3 s" and c["how"] == "given"
    assert "stopped at the budget" in c["basis"]


def test_projection_warns():
    r = _spencer_grid(0.3)
    stop = r.search_ranges["time_budget"]["stopped"][0]
    assert stop["projected_s"] > 0.3
    assert (f"all 144 would take about {stop['projected_s']:.0f} s"
            in r.warnings[0])


def test_raising_the_budget_covers_the_whole_search():
    kw = dict(surface_type="entry_exit", nx=6, ny=6, n_slices=20)
    cut = search_critical_surface(_geom(), time_budget_s=1e-4, **kw)
    assert cut.search_ranges["time_budget"]["stopped"][0]["scored"] < 36
    full = search_critical_surface(_geom(), time_budget_s=60, **kw)
    free = search_critical_surface(_geom(), time_budget_s=None, **kw)
    assert "time_budget" not in full.search_ranges
    assert not any("time budget" in w for w in full.warnings)
    # A search inside its budget is the search without one.
    assert full.critical.FOS == free.critical.FOS
    assert full.grid_fos == free.grid_fos
    c = _by_what(full)["time budget of the search"]
    assert c["chosen"] == "60 s" and c["how"] == "given"
    assert c["basis"] == "the search finished inside it"
    # No run time in choices: they do not change from run to run.
    free_c = _by_what(free)["time budget of the search"]
    assert free_c["chosen"] == "no limit" and "basis" not in free_c


def test_library_default_is_no_budget():
    r = search_critical_surface(_geom(), surface_type="entry_exit",
                                nx=4, ny=4)
    assert "time_budget" not in r.search_ranges
    c = _by_what(r)["time budget of the search"]
    assert c["chosen"] == "no limit" and c["how"] == "default"
    assert c["alternatives"] == ["a time budget (time_budget_s)"]


def test_finished_centre_grid_with_refinement_is_the_unbudgeted_one():
    kw = dict(nx=6, ny=6, n_slices=20)          # refine on, as by default
    given = search_critical_surface(_geom(), time_budget_s=600, **kw)
    free = search_critical_surface(_geom(), **kw)
    assert "time_budget" not in given.search_ranges
    a, b = given.critical, free.critical
    assert (a.FOS, a.xc, a.yc, a.radius) == (b.FOS, b.xc, b.yc, b.radius)
    assert given.grid_fos == free.grid_fos
    assert given.n_surfaces_evaluated == free.n_surfaces_evaluated


def test_raising_past_600_s_says_it_cannot_cover_the_search():
    from slope_stability.search_supplements import note_budget
    from slope_stability.results import SearchResult
    from slope_stability.search import SearchBudget
    budget = SearchBudget(180)
    budget.stops.append({"stage": "entry/exit grid", "scored": 98,
                         "of": 900, "unit": "entry/exit pairs",
                         "covered": "c", "not_covered": "n", "time_s": 144.0,
                         "projected_s": 1327})
    r = SearchResult()
    note_budget(r, budget, method="bishop")
    assert ("raise time_budget_s (at most 600 s, which alone cannot cover "
            "this search: about 1327 s projected)") in r.warnings[0]
    assert "bishop_screen" not in r.warnings[0]     # not for Bishop


# The 30 x 30 entry/exit map of the live-run slope (residual clay over
# sandstone; windows 20-29 m and 29.5-45 m), scored with Bishop at 60
# slices. Its minimum lies in one entry row 0.31 m wide, just above the
# clay / sandstone contact: a coarse lattice steps over it.
_MAP = json.loads((Path(__file__).parent / "data"
                   / "entry_exit_bishop_map.json").read_text())


def _points_to_within(order, rel):
    """How many points ``order`` scores before one within ``rel`` of the
    map's minimum."""
    target = (1.0 + rel) * min(min(row) for row in _MAP)
    n = 0
    for i, j in order:
        n += 1
        if _MAP[i][j] <= target:
            return n
    return n


def test_scan_finds_a_narrow_valley_early():
    from slope_stability.search import _scan_grid
    seen = []

    def score(i, j):
        seen.append((i, j))
        return _MAP[i][j]

    sweep = _scan_grid(30, 30, score)
    assert sweep["done"] == len(set(seen)) == len(seen) == 900
    new = _points_to_within(seen, 0.01)

    def coarse_first():                  # every 16th, 8th, ... point
        s, out = 16, []
        while s:
            for i in sorted(set(range(0, 30, s)) | {29}):
                for j in sorted(set(range(0, 30, s)) | {29}):
                    if (i, j) not in out:
                        out.append((i, j))
            s //= 2
        return out

    old = _points_to_within(coarse_first(), 0.01)
    assert new <= 60 and old >= 600, (new, old)          # 44 against 713


def test_de_cut_short_returns_a_surface():
    r = search_critical_surface(_geom(), surface_type="noncircular_de",
                                seed=1, time_budget_s=1e-4)
    assert r.critical is not None and r.critical.FOS < 100
    assert r.warnings[0].startswith("Cut short by its time budget")


def test_budget_warning_comes_before_the_swapped_windows_note():
    r = search_critical_surface(
        _geom(), surface_type="entry_exit", nx=6, ny=6, method="spencer",
        n_slices=20, x_entry_range=(12, 20), x_exit_range=(0, 6),
        time_budget_s=1e-4)
    assert r.warnings[0].startswith("Cut short by its time budget")
    assert "swapped" in r.warnings[1]


def test_screen_asked_for_but_not_applied_says_why():
    r = search_critical_surface(_geom(), surface_type="noncircular",
                                n_trials=30, seed=1, method="spencer",
                                bishop_screen=True)
    c = _by_what(r)["Bishop screen of the main search"]
    assert c["chosen"] is False
    assert c["basis"].startswith("asked for (bishop_screen) but not applied:"
                                 " noncircular searches are not screened")
    r = search_critical_surface(_geom(), surface_type="entry_exit", nx=3,
                                ny=3, bishop_screen=True)       # Bishop
    c = _by_what(r)["Bishop screen of the main search"]
    assert c["chosen"] is False and "the method is bishop" in c["basis"]


def test_screen_says_when_its_cap_left_points_out(monkeypatch):
    from slope_stability import search
    monkeypatch.setattr(search, "_SCREEN_MIN", 2)
    monkeypatch.setattr(search, "_SCREEN_MAX", 2)
    monkeypatch.setattr(search, "_SCREEN_WITHIN", 10.0)   # all within
    r = search_critical_surface(
        _geom(), surface_type="entry_exit", nx=3, ny=3, method="spencer",
        n_slices=20, bishop_screen=True, toe_circles=False,
        infinite_slope=False)
    basis = _by_what(r)["Bishop screen of the main search"]["basis"]
    assert "2 re-scored, the most it re-scores:" in basis
    assert "more within" in basis and "of the screen's lowest were not"         in basis


def test_centre_grid_stops_coarse_first():
    r = search_critical_surface(_geom(), nx=8, ny=8, method="spencer",
                                n_slices=20, time_budget_s=1e-4)
    tb = r.search_ranges["time_budget"]
    stop = tb["stopped"][0]
    assert stop["stage"] == "centre grid" and 1 <= stop["scored"] < 64
    assert "the refinement of the lowest grid centres" in tb["not_run"]
    assert r.critical is not None


def test_noncircular_search_stops_at_the_budget():
    r = search_critical_surface(_geom(), surface_type="noncircular",
                                n_trials=100000, seed=1, time_budget_s=0.2)
    stop = r.search_ranges["time_budget"]["stopped"][0]
    assert stop["stage"] == "random noncircular search"
    assert 1 <= stop["scored"] < 100000
    assert r.warnings[0].startswith("Cut short by its time budget")


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        with search_budget(0):
            pass


@pytest.mark.parametrize("surface_type", ["entry_exit", "circular"])
def test_bishop_screen_finds_the_same_surface(surface_type, monkeypatch):
    # A short list of 3 to 5 of the 9 points, so some stay screen values.
    monkeypatch.setattr(search, "_SCREEN_MIN", 3)
    monkeypatch.setattr(search, "_SCREEN_MAX", 5)
    kw = dict(surface_type=surface_type, nx=3, ny=3, method="spencer",
              n_slices=20, time_budget_s=None, toe_circles=False,
              infinite_slope=False)
    full = search_critical_surface(_geom(), **kw)
    screened = search_critical_surface(_geom(), bishop_screen=True, **kw)
    assert screened.critical.FOS == pytest.approx(full.critical.FOS,
                                                  abs=1e-3)
    assert screened.critical.method == full.critical.method
    tagged = [e for e in screened.grid_fos if e.get("method") == "bishop"]
    assert 0 < len(tagged) < len(screened.grid_fos)
    c = _by_what(screened)["Bishop screen of the main search"]
    assert c["chosen"] is True and c["how"] == "given"
    assert "Bishop screen of the main search" not in _by_what(full)
