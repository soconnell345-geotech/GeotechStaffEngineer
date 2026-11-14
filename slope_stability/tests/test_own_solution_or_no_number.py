"""A method's own solution, or no number (GM-140 step 2).

The rule. A factor of safety reported under a method's name is that
method's own converged solution on that surface. Where the method has no
solution there is no number: ``FOS`` is None, ``method_solved`` is False
and the result says why. It holds for one surface, every search family,
the yield acceleration, FOSM and Monte Carlo.

Print: UFC 3-220-10 p. 362, "Critical failure surfaces can be missed as a
result of non-convergence. The engineer should be able to identify when
convergence issues are present."; EM 1110-2-1902 C-10d, pp. C-37 to C-38
(ways to avoid non-convergence); Bishop (1955) Eq. 13, p. 10; Spencer
(1967) Eqs. 7 and 8, p. 15; Morgenstern and Price (1965) Eq. 18, p. 85.

Every geometry here is made up. Each test failed on master before the
change (the departures D23, D27, D30 and D49 of
module_work/slope_integral_review/ACCOUNT.md).
"""

import math
import warnings

import pytest

import slope_stability
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import (CircularSlipSurface,
                                          PolylineSlipSurface)
from slope_stability.slices import build_slices
from slope_stability.analysis import (analyze_slope, compare_methods_table,
                                      search_critical_surface)
from slope_stability.methods import (bishop_fos, fellenius_fos, spencer_fos,
                                     morgenstern_price_fos)
from slope_stability.gle import janbu_fos
from slope_stability.solution import (solve_method, NoSolution, METHODS,
                                      MethodSolution)
from slope_stability.search import _compute_fos, _FOS_MAX
from slope_stability.newmark import yield_acceleration
from slope_stability.probabilistic import fosm_fos, monte_carlo_fos


def _clay_slope():
    """A made-up 8 m clay slope, phi = 0, su = 30 kPa."""
    return SlopeGeometry(
        surface_points=[(0, 0), (12, 0), (24, 8), (40, 8)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=8.0,
                                    bottom_elevation=-10.0, gamma=18.0,
                                    cu=30.0, analysis_mode="undrained")])


def _fill_slope(c=12.0, phi=22.0):
    """A made-up 10 m fill slope with cohesion and friction."""
    return SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (16, 10), (34, 10)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=10.0,
                                    bottom_elevation=-6.0, gamma=19.0,
                                    phi=phi, c_prime=c)])


# The circle of the fill slope on which Spencer's method solves with a
# cohesion up to about 28 kPa and has no solution at 30 kPa or above (the
# top of the circle goes into tension and no lambda in range makes the
# moment and force factors of safety agree). Until GM-344 the solver read
# a value up to about 40 kPa (1.645 at 30 kPa, 1.917 at 38): the friction
# on the base in tension was left out of some of its equations and kept in
# others, and those values are roots of neither Spencer's equations as
# printed nor of the slice taken frictionless throughout (a second
# computation finds no root from -5 to 60 degrees at 30 kPa under either;
# module_work/gm140/GLE_ROOT_CHECK.md).
_FILL_CIRCLE = dict(xc=9.5, yc=11.0, radius=11.2)

# A circle of the fill slope with a cohesion of 40 kPa on which Spencer's
# method has no solution. The step 2 review's own Spencer (Spencer 1967
# Eqs. 7 and 8, p. 15; 30 slices; nothing from this module) finds the force
# and moment factors of safety apart by at least 0.043 for every side-force
# inclination from -30 to 60 degrees: they never meet. Master reported
# 1.766 here, from the old approximate solver. Not used: GM-139's kind of
# circle, its centre below the crest of a phi = 0 slope. One such circle,
# (30, 4, 9) of the clay slope, does have a Spencer solution, with a very
# high factor of safety (theta 0.65 degrees, F 16.47 on 30 slices by the
# review's own Spencer); the module's solver finds none there
# (module_work/gm140/STEP2_CHECK.md, "noted, not fixed").
_NO_SPENCER_CIRCLE = dict(xc=8.0, yc=12.0, radius=12.0)


# ---------------------------------------------------------------------------
# One surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["spencer", "morgenstern_price", "gle"])
def test_a_circle_with_no_spencer_solution_has_no_number(method):
    """D23/D30. On this circle no side-force inclination makes Spencer's
    moment and force factors of safety agree (Spencer 1967 Eqs. 7 and 8,
    p. 15; checked independently, see _NO_SPENCER_CIRCLE). The old
    approximate solver's value was reported in its place (1.766 on
    master). There is no number (UFC 3-220-10 p. 362). Whether
    Morgenstern-Price (half-sine) has a solution here was not checked
    independently, so for it and the GLE the test holds the rule only:
    their own solution, or no number with the reason. (The module's solver
    finds none.)"""
    g = _fill_slope(c=40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # and no Python warning
        r = analyze_slope(g, method=method, **_NO_SPENCER_CIRCLE)
    if method != "spencer" and r.method_solved:
        assert r.FOS is not None and not r.methods_without_solution
        return
    assert r.FOS is None and r.is_stable is None
    assert r.method_solved is False and r.method_used is None
    assert r.methods_without_solution == [
        "Spencer" if method == "spencer" else "Morgenstern-Price"]
    (w,) = [w for w in r.warnings if "no equilibrium solution" in w]
    assert "No factor of safety is reported" in w
    assert "UFC 3-220-10 p. 362" in w
    d = r.to_dict()
    assert d["FOS"] is None and d["method_solved"] is False
    assert d["method_used"] is None
    # Bishop's simplified method solves the same circle: its own number,
    # 1.7268 by the review's own Bishop (1955 Eq. 13, p. 10) on 30 slices
    b = analyze_slope(g, method="bishop", **_NO_SPENCER_CIRCLE)
    assert b.method_solved
    assert b.FOS_bishop == pytest.approx(1.7268, rel=0.01)
    # GM-140 step 3 (OWN-102; Duncan 1996, p. 580): on this circle Bishop's
    # value is below the Ordinary Method of Slices value (1.750), so the
    # latter is the factor of safety reported, and the result says so.
    assert b.FOS == pytest.approx(b.FOS_fellenius)
    assert b.FOS_fellenius > b.FOS_bishop
    assert "Ordinary Method of Slices" in b.method_used


def test_no_approximate_solver_is_left_in_the_package():
    """D30; the module audit's row 6 (slope_stability). The old
    approximate Spencer and Morgenstern-Price solvers supply no value
    anywhere: they are removed, not kept beside the rigorous ones."""
    import slope_stability.methods as m
    for name in ("spencer_fos_legacy", "morgenstern_price_fos_legacy"):
        assert not hasattr(m, name)
        assert not hasattr(slope_stability, name)
        assert name not in slope_stability.__all__
    assert not hasattr(m, "solve_rigorous")     # solve_method replaced it


def test_solver_functions_raise_where_they_have_no_solution():
    """The method functions give their own solution or raise NoSolution
    with the reason: none returns another solver's value or the 999.9
    marker (D30, D49). Spencer has no solution on this circle; for
    Morgenstern-Price (not checked independently here) the test holds the
    rule only."""
    g = _fill_slope(c=40.0)
    slip = CircularSlipSurface(**_NO_SPENCER_CIRCLE)
    slices = build_slices(g, slip, 30)
    with pytest.raises(NoSolution) as exc:
        spencer_fos(slices, slip)
    assert exc.value.reason
    try:
        f_mp = morgenstern_price_fos(slices, slip)
    except NoSolution as e:
        assert e.reason
    else:
        assert 0.0 < f_mp < 100.0
    assert isinstance(NoSolution("x"), ValueError)


def test_compare_methods_gives_no_number_for_a_method_without_solution():
    """A comparison table carries each method's own solution; a method
    with none has no row and is named (D30). Spencer has none on this
    circle."""
    g = _fill_slope(c=40.0)
    r = analyze_slope(g, method="bishop", compare_methods=True,
                      **_NO_SPENCER_CIRCLE)
    assert r.method_solved and r.FOS is not None
    assert r.FOS_spencer is None and r.theta_spencer is None
    assert "Spencer" in r.methods_without_solution
    t = compare_methods_table(g, **_NO_SPENCER_CIRCLE)
    names = [row["method"] for row in t["rows"]]
    assert not any(n.startswith("Spencer") for n in names)
    assert t["methods_without_solution"] == r.methods_without_solution
    assert all(isinstance(row["FOS"], float) for row in t["rows"])
    assert "legacy" not in t["summary"]
    assert "no solution on this surface" in t["summary"]


def test_bishop_stopped_at_its_iteration_limit_gives_no_number():
    """D27. Bishop's iteration on F (Bishop 1955 Eq. 13, p. 10) that is not
    within its tolerance at its iteration limit has no solution. Before,
    the last iterate was returned as the factor of safety with a Python
    warning only (UFC 3-220-10 p. 362: the engineer should be able to
    identify convergence issues)."""
    g = _fill_slope()
    slip = CircularSlipSurface(**_FILL_CIRCLE)
    slices = build_slices(g, slip, 30)
    converged = bishop_fos(slices, slip)
    assert converged == pytest.approx(1.05, abs=0.03)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(NoSolution, match="not within .* after 2 iter"):
            bishop_fos(slices, slip, tol=1e-12, max_iter=2, fos_initial=9.0)


def test_bishop_without_solution_is_no_number_in_a_result(monkeypatch):
    """D27, in a result and in a search: where Bishop's iteration does not
    converge the result's FOS is None with the reason, and a search does
    not score the circle."""
    import slope_stability.methods as m
    real = m.bishop_fos

    def stopped(slices, slip, tol=1e-4, max_iter=50, **kw):
        return real(slices, slip, tol=1e-12, max_iter=2, fos_initial=9.0,
                    **{k: v for k, v in kw.items() if k != "fos_initial"})

    monkeypatch.setattr(m, "bishop_fos", stopped)
    g = _fill_slope()
    r = analyze_slope(g, method="bishop", **_FILL_CIRCLE)
    assert r.FOS is None and r.method_solved is False
    assert r.methods_without_solution == ["Bishop"]
    assert any("Bishop's simplified method has no solution on this surface"
               in w and "not within" in w for w in r.warnings)
    slip = CircularSlipSurface(**_FILL_CIRCLE)
    assert _compute_fos(g, slip, "bishop", 30) == _FOS_MAX


def test_no_driving_moment_is_no_number_not_the_marker():
    """D49. A circle with no net driving moment (level ground, the mass
    balanced about the centre) has no factor of safety: the ratio of
    resisting to driving has no meaning there, and neither the solvers'
    999.9 marker nor an astronomical ratio leaves a solver as one."""
    g = SlopeGeometry(
        surface_points=[(0, 5), (40, 5)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=5.0,
                                    bottom_elevation=-10.0, gamma=18.0,
                                    phi=30.0, c_prime=5.0)])
    slip = CircularSlipSurface(20.0, 9.0, 8.0)
    slices = build_slices(g, slip, 30)
    for method in ("fellenius", "bishop"):
        sol = solve_method(slices, slip, method)
        assert sol.fos is None and not sol.solved
        assert "driving" in sol.reason and sol.kind == "no_driving"
        r = analyze_slope(g, xc=20.0, yc=9.0, radius=8.0, method=method)
        assert r.FOS is None and r.method_solved is False
        assert r.to_dict()["FOS"] is None
        assert _compute_fos(g, slip, method, 30) == _FOS_MAX
    # the solver functions themselves raise where the driving sum is zero
    # or negative (they used to return 999.9)
    import copy
    flipped = [copy.copy(s) for s in slices]
    for s in flipped:
        s.weight, s.surcharge_force = 0.0, 0.0
    for fn in (fellenius_fos, bishop_fos):
        with pytest.raises(NoSolution, match="no net driving"):
            fn(flipped, slip)


def test_solve_method_is_the_solver_of_a_result():
    """Item 9: one solver for a method. `solve_method` on the surface's
    slices is what `analyze_slope` reports, for every method."""
    g = _fill_slope()
    slip = CircularSlipSurface(**_FILL_CIRCLE)
    slices = build_slices(g, slip, 30)
    for method in METHODS:
        sol = solve_method(slices, slip, method)
        assert isinstance(sol, MethodSolution) and sol.solved
        r = analyze_slope(g, method=method, **_FILL_CIRCLE)
        assert r.FOS == sol.fos


# ---------------------------------------------------------------------------
# Unknown names
# ---------------------------------------------------------------------------

def test_unknown_method_is_refused_everywhere():
    """D49. An unknown method name ran Bishop's simplified method without
    saying so. It is refused, with the list of names, in every entry
    point."""
    g = _fill_slope()
    slip = CircularSlipSurface(**_FILL_CIRCLE)
    slices = build_slices(g, slip, 30)
    variables = {"c_prime": {"mean": 12.0, "std": 2.0}}
    calls = [
        lambda: analyze_slope(g, method="bishops", **_FILL_CIRCLE),
        lambda: search_critical_surface(g, method="bishops", nx=3, ny=3),
        lambda: search_critical_surface(g, method="bishops",
                                        surface_type="noncircular",
                                        n_trials=5, seed=1),
        lambda: slope_stability.grid_search(g, (0, 20), (11, 25), 3, 3,
                                            "bishops", 30),
        lambda: solve_method(slices, slip, "bishops"),
        lambda: _compute_fos(g, slip, "bishops", 30),
        lambda: yield_acceleration(g, method="bishops", **_FILL_CIRCLE),
        lambda: fosm_fos(g, variables, method="bishops", **_FILL_CIRCLE),
        lambda: monte_carlo_fos(g, variables, method="bishops", n=3,
                                seed=1, **_FILL_CIRCLE),
    ]
    for call in calls:
        with pytest.raises(ValueError) as exc:
            call()
        text = str(exc.value)
        assert "Unknown method 'bishops'" in text
        for name in METHODS:
            assert repr(name) in text


def test_unknown_interslice_function_is_refused():
    """D45/D49. An unknown f_interslice is refused with the list of names
    (a comparison used to accept any string where it was not used)."""
    g = _fill_slope()
    for call in (
            lambda: analyze_slope(g, method="gle", f_interslice="sine",
                                  **_FILL_CIRCLE),
            lambda: analyze_slope(g, method="bishop", compare_methods=True,
                                  f_interslice="sine", **_FILL_CIRCLE),
            lambda: compare_methods_table(g, f_interslice="sine",
                                          **_FILL_CIRCLE)):
        with pytest.raises(ValueError) as exc:
            call()
        assert "Unknown interslice function 'sine'" in str(exc.value)
        assert "'half_sine'" in str(exc.value)


def test_unknown_interslice_function_is_refused_whatever_the_method():
    """Item 7 (the step 2 review's D4). An unknown f_interslice was
    accepted, and not used, unless the method was Morgenstern-Price or the
    GLE (or a comparison was asked for). It is refused with the list of
    names whatever the method, on one surface and by the solver. The
    yield acceleration, FOSM and Monte Carlo take no f_interslice (their
    Morgenstern-Price and GLE solves use the half-sine function), so none
    of them can accept an unknown one. A search takes it since GM-140
    step 6, and refuses an unknown one with the same list."""
    g = _fill_slope()
    slip = CircularSlipSurface(**_FILL_CIRCLE)
    slices = build_slices(g, slip, 30)
    calls = [lambda m=m: analyze_slope(g, method=m, f_interslice="sine",
                                       **_FILL_CIRCLE) for m in METHODS]
    calls += [lambda m=m: solve_method(slices, slip, m,
                                       f_interslice="sine")
              for m in METHODS]
    calls += [lambda m=m: search_critical_surface(
        g, method=m, f_interslice="sine", nx=3, ny=3) for m in METHODS]
    for call in calls:
        with pytest.raises(ValueError) as exc:
            call()
        assert "Unknown interslice function 'sine'" in str(exc.value)
        for name in ("'constant'", "'half_sine'", "'clipped_sine'",
                     "'trapezoidal'"):
            assert name in str(exc.value)
    variables = {"c_prime": {"mean": 12.0, "std": 2.0}}
    for call in (
            lambda: yield_acceleration(g, f_interslice="sine",
                                       **_FILL_CIRCLE),
            lambda: fosm_fos(g, variables, f_interslice="sine",
                             **_FILL_CIRCLE),
            lambda: monte_carlo_fos(g, variables, n=3, seed=1,
                                    f_interslice="sine", **_FILL_CIRCLE)):
        with pytest.raises(TypeError, match="f_interslice"):
            call()


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

def test_spencer_circular_search_scores_no_circle_by_another_solver():
    """D23. A Spencer circular search scored a circle whose rigorous
    solution did not converge by the old approximate solver, with no
    warning. Now every circle is scored by Spencer's own solution or not
    scored, the circles without a solution are counted by family and by
    method, and the result warns when the governing family lost trials
    (UFC 3-220-10 p. 362)."""
    g = _clay_slope()
    res = search_critical_surface(g, method="spencer", nx=4, ny=4)
    c = res.critical
    assert c is not None and c.method_solved and c.method_used == ""
    # the governing circle is Spencer's own solution, to the last digit
    again = analyze_slope(g, xc=c.xc, yc=c.yc, radius=c.radius,
                          method="spencer")
    assert again.method_solved and again.FOS == pytest.approx(c.FOS,
                                                              rel=1e-9)
    assert res.n_no_solution > 0
    assert res.n_no_solution == sum(r["n"] for r in res.no_solution)
    for rec in res.no_solution:
        assert set(rec) == {"family", "method", "n", "not_converged",
                            "no_driving", "below_0_05"}
        assert rec["n"] == (rec["not_converged"] + rec["no_driving"]
                            + rec["below_0_05"])
        assert rec["family"] in ("circular", "toe_circle",
                                 "circle_above_toe", "infinite_slope")
        assert rec["method"] in METHODS
    d = res.to_dict()
    assert d["no_solution"] == res.no_solution
    assert not any("legacy" in w for w in res.warnings)
    lost = [r for r in res.no_solution
            if r["family"] == res.governing_surface and r["not_converged"]]
    warned = [w for w in res.warnings if "without a solution" in w]
    assert bool(lost) == bool(warned)
    if warned:
        assert "UFC 3-220-10 p. 362" in warned[0]
        assert "cannot be ruled out" in warned[0]


def test_a_search_in_which_no_trial_solves_says_so(monkeypatch):
    """The step 2 review's D5. Where no trial surface solves there is no
    critical surface, so no governing family, and the search's warnings
    said nothing. They now say that no trial surface had a solution by the
    method, with the counts by family, method and reason. Every solve is
    withheld here (the solver itself is tested above)."""
    import slope_stability.search as search_mod

    def withheld(slices, slip, method, *a, **k):
        return MethodSolution(method=method, fos=None, solved=False,
                              reason="withheld", kind="not_converged")

    monkeypatch.setattr(search_mod, "_method_solution", withheld)
    for kw in (dict(nx=3, ny=3),
               dict(surface_type="noncircular", n_trials=20, seed=1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # the noncircular summary
            res = search_critical_surface(_fill_slope(), method="spencer",
                                          **kw)
        assert res.critical is None and res.n_no_solution > 0
        (w,) = [w for w in res.warnings if w.startswith(
            "No trial surface had a solution by the method")]
        assert f"{res.n_no_solution} trial surface(s)" in w
        for r in res.no_solution:
            assert (f"{r['family']}, " in w
                    and f"{r['n']} ({r['not_converged']} did not converge)"
                    in w)
        assert "UFC 3-220-10 p. 362" in w


def test_a_value_below_0_05_is_counted_under_its_own_reason(monkeypatch):
    """The step 2 review's D6. A search does not take a solved value below
    0.05 as a factor of safety (the existing screen). Such a surface was
    counted as not converged; it has its own reason in the count. Every
    solve gives 0.01 here."""
    import slope_stability.search as search_mod

    def tiny(slices, slip, method, *a, **k):
        return MethodSolution(method=method, fos=0.01, solved=True)

    monkeypatch.setattr(search_mod, "_method_solution", tiny)
    res = slope_stability.grid_search(_fill_slope(), (0.0, 20.0),
                                      (11.0, 25.0), 3, 3, "spencer", 30)
    assert res.critical is None and res.no_solution
    for r in res.no_solution:
        assert r["below_0_05"] == r["n"] > 0
        assert r["not_converged"] == r["no_driving"] == 0
    assert "solved below 0.05" in res.summary()


def test_no_number_row_says_why_in_the_calc_package():
    """The step 2 review's D6. The method-comparison row of a method with
    no number said "does not converge" also where the surface has no net
    driving moment: it says which."""
    from slope_stability.calc_steps import _comparison_section
    level = SlopeGeometry(
        surface_points=[(0, 5), (40, 5)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=5.0,
                                    bottom_elevation=-10.0, gamma=18.0,
                                    phi=30.0, c_prime=5.0)])
    r = analyze_slope(level, xc=20.0, yc=9.0, radius=8.0, method="bishop",
                      compare_methods=True)
    rows = _comparison_section(r, {}).items[0].rows
    held = [row for row in rows if row[0] in ("Fellenius", "Bishop")]
    assert held and all("no net driving moment or force" in row[2]
                        for row in held)
    r = analyze_slope(_fill_slope(c=40.0), method="bishop",
                      compare_methods=True, **_NO_SPENCER_CIRCLE)
    rows = _comparison_section(r, {}).items[0].rows
    (sp,) = [row for row in rows if row[0] == "Spencer"]
    assert sp[1] == "no solution" and "does not converge" in sp[2]


def test_search_and_single_surface_use_one_solver_on_a_polyline():
    """Item 9. A noncircular search accepted a solution about a second,
    fixed moment axis that the single-surface solver did not use, so it
    could report a surface `analyze_slope` had no solution on. The search's
    score of a polyline is `analyze_slope`'s FOS on it, or the surface is
    not scored."""
    g = _fill_slope(c=6.0, phi=28.0)
    res = search_critical_surface(g, method="spencer",
                                  surface_type="noncircular", n_trials=120,
                                  seed=4)
    c = res.critical
    assert c is not None
    slip = PolylineSlipSurface(points=c.slip_points)
    one = analyze_slope(g, slip_surface=slip, method="spencer")
    assert one.method_solved and one.FOS == pytest.approx(c.FOS, rel=1e-9)
    # Over a made-up family of polylines: the search's score is the
    # single-surface FOS to the last digit, and a polyline the
    # single-surface solver cannot solve is not scored.
    n_scored = n_unsolved = 0
    for x0 in (4.0, 6.0, 8.0, 9.5):
        for x1 in (20.0, 24.0, 28.0):
            for depth in (1.0, 2.5, 4.0, 6.0):
                pts = [(x0, 0.0), (0.6 * x0 + 0.4 * x1, -depth),
                       (0.3 * x0 + 0.7 * x1, 4.0 - depth), (x1, 10.0)]
                s = PolylineSlipSurface(points=pts)
                score = _compute_fos(g, s, "spencer", 30,
                                     m_alpha_check=False)
                one = analyze_slope(g, slip_surface=s, method="spencer")
                if score < _FOS_MAX:
                    n_scored += 1
                    assert one.method_solved
                    assert one.FOS == pytest.approx(score, rel=1e-9)
                if not one.method_solved:
                    n_unsolved += 1
                    assert one.FOS is None and score == _FOS_MAX
    assert n_scored >= 10 and n_unsolved >= 1


def test_search_count_is_by_family_and_method():
    """Item 8. The count names the family and the method that had no
    solution; a search called directly counts under 'main'."""
    g = _clay_slope()
    res = slope_stability.grid_search(g, (10.0, 34.0), (9.0, 24.0), 4, 4,
                                      "spencer", 30)
    assert res.no_solution and all(r["family"] == "main"
                                   and r["method"] == "spencer"
                                   for r in res.no_solution)
    assert res.to_dict()["no_solution"] == res.no_solution
    assert f"no solution on them: {res.n_no_solution}" in res.summary()


# ---------------------------------------------------------------------------
# Yield acceleration
# ---------------------------------------------------------------------------

def test_yield_acceleration_uses_spencers_own_solution_at_every_kh(
        monkeypatch):
    """D30. The yield coefficient by Spencer rested on the old approximate
    solver's factor of safety at any trial kh where the rigorous solution
    did not converge. Every trial kh is Spencer's own solution
    (`solve_method`), and ky is the kh at which that solution is 1."""
    import slope_stability.solution as sol_mod
    g = _fill_slope()
    seen = []
    real = sol_mod.solve_method

    def spy(slices, slip, method, *a, **k):
        out = real(slices, slip, method, *a, **k)
        seen.append((method, out.solved))
        return out

    monkeypatch.setattr(sol_mod, "solve_method", spy)
    ya = yield_acceleration(g, method="spencer", **_FILL_CIRCLE)
    assert ya.converged and ya.ky is not None and 0.0 < ya.ky < 0.5
    assert seen and all(m == "spencer" for m, _ in seen)
    monkeypatch.undo()
    gk = _fill_slope()
    gk.kh = ya.ky
    at = analyze_slope(gk, method="spencer", n_slices=50, **_FILL_CIRCLE)
    assert at.method_solved and at.FOS == pytest.approx(1.0, abs=2e-3)


def test_yield_acceleration_without_a_solution_gives_no_number():
    """Where the method has no solution at a trial kh the bisection needs,
    there is no yield coefficient: ky is None and the result says at which
    kh and why. No other solver is asked."""
    ya = yield_acceleration(_fill_slope(c=40.0), method="spencer",
                            **_NO_SPENCER_CIRCLE)
    assert ya.ky is None and ya.ay is None and not ya.converged
    assert "has no solution on this surface at kh = " in ya.reason
    assert "No yield coefficient is reported" in ya.reason
    d = ya.to_dict()
    assert d["ky"] is None and d["ay_m_s2"] is None
    assert d["reason"] == ya.reason and "no_solution_kh" in d


# ---------------------------------------------------------------------------
# FOSM and Monte Carlo
# ---------------------------------------------------------------------------

def test_fosm_meeting_a_surface_with_no_solution_gives_no_result():
    """D49. FOSM took the 999.9 marker (or the approximate solver's value)
    as a factor of safety at a perturbed value. Where the method has no
    solution at mean plus or minus one standard deviation, FOSM gives no
    result and names the variable and the perturbed value. (No printed
    rule was found for this; it is the module's own.)"""
    # (GM-344: the mean was 38 kPa, where Spencer's method no longer has a
    # solution on this circle; see _FILL_CIRCLE)
    g = _fill_slope(c=22.0)
    with pytest.raises(ValueError) as exc:
        fosm_fos(g, {"c_prime": {"mean": 22.0, "std": 8.0}},
                 method="spencer", **_FILL_CIRCLE)
    text = str(exc.value)
    assert "FOSM gives no result" in text
    assert "c_prime = 30" in text
    assert "mean 22 plus one standard deviation 8" in text
    assert "Spencer's method has no solution" in text
    # with a smaller spread every evaluation solves, and FOSM runs
    ok = fosm_fos(g, {"c_prime": {"mean": 22.0, "std": 2.0}},
                  method="spencer", **_FILL_CIRCLE)
    assert ok.fos_mlv == pytest.approx(
        analyze_slope(g, method="spencer", **_FILL_CIRCLE).FOS, rel=1e-9)


def test_monte_carlo_counts_realizations_with_no_solution():
    """D49. A realization in which the method has no solution is never a
    sample: it is counted, left out of the statistics, and the probability
    of failure is bounded both ways (all of them failures, none of them).
    (No printed rule was found for this; the result says it is the
    module's own.)"""
    g = _fill_slope(c=38.0)
    variables = {"c_prime": {"mean": 38.0, "std": 12.0}}
    mc = monte_carlo_fos(g, variables, method="spencer", n=120, seed=7,
                         **_FILL_CIRCLE)
    assert mc.n_no_solution > 0
    assert mc.n + mc.n_no_solution == 120
    assert len(mc.samples) == mc.n
    assert all(0.0 < f < 100.0 for f in mc.samples)     # no marker value
    assert mc.pf == pytest.approx(mc.n_failed / mc.n)
    assert mc.pf_lower == pytest.approx(mc.n_failed / 120)
    assert mc.pf_upper == pytest.approx(
        (mc.n_failed + mc.n_no_solution) / 120)
    assert mc.pf_lower <= mc.pf <= mc.pf_upper
    d = mc.to_dict()
    assert d["n_no_solution"] == mc.n_no_solution
    assert d["pf_bounds"] == [round(mc.pf_lower, 6), round(mc.pf_upper, 6)]
    assert "module's own" in d["no_solution_note"]
    assert "NO SOLUTION" in mc.summary()
    # Bishop solves every realization of the same circle: nothing to count
    b = monte_carlo_fos(g, variables, method="bishop", n=120, seed=7,
                        **_FILL_CIRCLE)
    assert b.n_no_solution == 0 and b.n == 120
    assert b.pf_lower == b.pf_upper == b.pf
    assert "n_no_solution" not in b.to_dict()


def test_reliability_wrapper_never_takes_the_marker_as_a_sample():
    """`reliability.wrappers.slope_reliability` delegates to the same two
    functions, so the rule holds there too."""
    from reliability.wrappers import slope_reliability
    g = _fill_slope(c=38.0)
    variables = {"c_prime": {"mean": 38.0, "std": 12.0}}
    with pytest.raises(ValueError, match="FOSM gives no result"):
        slope_reliability(g, variables, engine="fosm", method="spencer",
                          **_FILL_CIRCLE)
    mc = slope_reliability(g, variables, engine="monte_carlo",
                           method="spencer", n=60, seed=7, **_FILL_CIRCLE)
    assert mc.n_no_solution > 0 and max(mc.samples) < 100.0


# ---------------------------------------------------------------------------
# The solver's lambda search (item 3)
# ---------------------------------------------------------------------------

def test_the_robust_lambda_search_runs_without_supports():
    """Item 3. The GLE's robust search for lambda (the scan started from
    the nearest lambda with a state when lambda = 0 has none, a lambda with
    no state stepped over, an unclosed bracket passed over) ran only where
    supports were present. It is the one search for every run. On this
    circle of the fill slope there is no state at lambda = 0; the search
    used to stop there (and Spencer's value came from the approximate
    solver), and now finds Spencer's own solution (EM 1110-2-1902 C-10d,
    pp. C-37 to C-38: a sensible starting point avoids non-convergence)."""
    from slope_stability.gle import gle_fos
    g = _fill_slope()
    slip = CircularSlipSurface(6.0, 15.0, 12.0)
    slices = build_slices(g, slip, 30)
    res = gle_fos(slices, slip, f_interslice="constant")
    assert res.converged
    assert abs(res.fos_moment - res.fos_force) < 1e-3
    # (GM-344: every state of the scan is now the direct march's, which
    # has a state at lambda = 0 here, so the scan starts there and the
    # "bracket started from" note is no longer raised on this circle)
    r = analyze_slope(g, xc=6.0, yc=15.0, radius=12.0, method="spencer")
    assert r.method_solved and r.FOS == pytest.approx(res.fos, rel=1e-9)


@pytest.mark.xfail(strict=True, reason=(
    "The step 2 review's D1, not applied: the module finds no Spencer "
    "solution on this circle, which has one (with phi = 0, the moment "
    "ratio worked out in the test). Making the direct march the fallback on every run where the "
    "fixed-point step cannot be taken finds it (2.9458) but also gives "
    "states resting on the m_alpha clamp, with base tension, on surfaces "
    "GM-131 found to have no Spencer solution (test_search_robustness, "
    "test_m_alpha), loses the yield coefficient of an ordinary slope "
    "(test_slope_stability_adapter::TestNewmark) and quadruples a "
    "noncircular search's run time (module_work/gm140/STEP2_CHECK.md, "
    "section 11). Decided in GM-140 step 3 (item 7): not applied. With "
    "the rule added that a marched state resting on the m_alpha clamp or "
    "on a base in tension is no state, the march finds this circle "
    "(2.9458) and no longer solves the GM-131 surfaces, but it still "
    "takes away a state the lambda search had "
    "(test_the_robust_lambda_search_runs_without_supports), an ordinary "
    "slope's yield coefficient (test_newmark::"
    "test_yield_acceleration_self_consistent: no solution at kh = 1.0) "
    "and changes two searches (module_work/gm140/STEP3_CHECK.md)."))
def test_the_march_is_tried_where_the_lagged_iteration_cannot_step():
    """Item 3 (the step 2 review's D1). Where the lagged iteration at a
    fixed lambda cannot take a step (a denominator at or below 1e-9), the
    direct march is tried only with supports. On this made-up phi = 0
    circle Spencer's method gives no number ("no lambda from -1.5 to 1.5
    gives an equilibrium state"). It has a solution: with phi = 0 the
    moment about the centre does not depend on the side forces, so
    Spencer's F, which satisfies moment equilibrium (Spencer 1967 Eq. 8,
    p. 15), is the moment ratio cu R (arc length) / sum(W x), x the moment
    arm of each slice's weight about the centre. The target below is that
    arithmetic on 600 equal slices, nothing from this module (3.084; the
    step 3 review got the same; the 2.9435 the step 2 review gave was
    wrong). The circle's centre is 3.0 m below the crest where it ends, so
    it is a reverse-curvature circle: since GM-140 step 4 it is no surface
    unless a crack is asked for, so a dry 3.0 m crack is given, which
    closes it at x = 41, where the arc turns vertical (EM 1110-2-1902
    C-10c; the module's Bishop reads 3.068 on 30 slices of the same
    surface; the test below checks the target)."""
    g, target = _march_circle()
    r = analyze_slope(g, xc=32.0, yc=7.0, radius=9.0, method="spencer",
                      n_slices=30)
    assert r.method_solved and r.FOS == pytest.approx(target, rel=0.01)


def test_the_march_circle_s_target_is_the_moment_ratio():
    """The strict xfail above holds its target: the moment ratio on 600
    slices, 3.084, and Bishop's simplified method, which with phi = 0
    gives the same ratio slice by slice, reads it within 1 % on 30 slices
    of the same surface, closed by the 3.0 m crack at x = 41."""
    g, target = _march_circle()
    assert target == pytest.approx(3.084, abs=0.001)
    b = analyze_slope(g, xc=32.0, yc=7.0, radius=9.0, method="bishop",
                      n_slices=30)
    assert b.tension_crack["x_m"] == pytest.approx(41.0, abs=0.01)
    assert b.FOS == pytest.approx(target, rel=0.01)


def _march_circle():
    """The step 2 review's phi = 0 circle ((32, 7), R 9; cu 25 kPa, gamma
    18) with the dry 3.0 m crack that closes it, and its factor of safety
    as the moment ratio cu R (arc length) / sum(W x) on 600 equal slices
    (with phi = 0 every method satisfying moment equilibrium gives it)."""
    xc, yc, R, cu, gamma = 32.0, 7.0, 9.0, 25.0, 18.0

    def ground(x):
        return 0.0 if x <= 15.0 else min(10.0, (x - 15.0) * 10.0 / 15.0)

    # the lower end, on the face z = (x - 15) 2/3:
    # (x - 32)^2 + (2x/3 - 17)^2 = 81
    a, b, c = 13.0 / 9.0, -64.0 - 68.0 / 3.0, 1024.0 + 289.0 - 81.0
    x_lo = (-b - math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    x_hi = xc + R                              # the crack, at x = 41
    n = 600
    w = (x_hi - x_lo) / n
    moment = 0.0
    for i in range(n):
        x = x_lo + (i + 0.5) * w
        z_arc = yc - math.sqrt(max(R * R - (x - xc) ** 2, 0.0))
        moment += gamma * w * (ground(x) - z_arc) * (x - xc)
    angle_lo = math.atan2(x_lo - xc, yc - ground(x_lo))
    arc = R * (math.pi / 2.0 - angle_lo)       # up to the vertical at 41
    g = SlopeGeometry(
        surface_points=[(0, 0), (15, 0), (30, 10), (50, 10)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=10.0,
                                    bottom_elevation=-12.0, gamma=gamma,
                                    cu=cu, analysis_mode="undrained")],
        tension_crack_depth=3.0)
    return g, cu * R * arc / moment


def test_an_iteration_that_does_not_settle_is_not_a_state():
    """Item 3 / D27 for the GLE. At a fixed lambda the iteration on
    {N, E, X, F} that ran its limit without settling returned its last
    iterate as a state, and nothing flagged it, so a "converged" Spencer
    value could rest on it (1.766 on this circle, with a cohesion of
    40 kPa). The last iterate is not an equilibrium state: there is no
    state at that lambda, and here no Spencer solution."""
    g = _fill_slope(c=40.0)
    r = analyze_slope(g, xc=8.0, yc=12.0, radius=12.0, method="spencer")
    assert r.FOS is None and r.method_solved is False
