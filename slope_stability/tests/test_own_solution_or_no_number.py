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
# cohesion up to about 40 kPa and has no solution at 50 kPa (the top of
# the circle goes into tension and no lambda in range makes the moment and
# force factors of safety agree).
_FILL_CIRCLE = dict(xc=9.5, yc=11.0, radius=11.2)

# An odd circle of the clay slope: its centre (el. 4) is below the crest
# (el. 8), as GM-139's was.
_ODD_CIRCLE = dict(xc=30.0, yc=4.0, radius=9.0)


# ---------------------------------------------------------------------------
# One surface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["spencer", "morgenstern_price", "gle"])
def test_odd_circle_the_rigorous_methods_cannot_solve_has_no_number(method):
    """GM-139, D23/D30. On a circle with its centre below the ground of a
    phi = 0 slope the rigorous methods find no lambda at which the moment
    and force factors of safety agree (Spencer 1967 Eqs. 7 and 8, p. 15).
    The old approximate solver's value was reported in its place (17.68
    here, labelled; about 0.05 on GM-139's circle). There is no number
    (UFC 3-220-10 p. 362)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # and no Python warning
        r = analyze_slope(_clay_slope(), method=method, **_ODD_CIRCLE)
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
    # Bishop's simplified method solves the same circle: its own number
    b = analyze_slope(_clay_slope(), method="bishop", **_ODD_CIRCLE)
    assert b.method_solved and b.FOS == pytest.approx(17.684, abs=0.01)


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
    marker (D30, D49)."""
    g = _clay_slope()
    slip = CircularSlipSurface(**_ODD_CIRCLE)
    slices = build_slices(g, slip, 30)
    for fn in (spencer_fos, morgenstern_price_fos):
        with pytest.raises(NoSolution) as exc:
            fn(slices, slip)
        assert exc.value.reason
    assert isinstance(NoSolution("x"), ValueError)


def test_compare_methods_gives_no_number_for_a_method_without_solution():
    """A comparison table carries each method's own solution; a method
    with none has no row and is named (D30)."""
    r = analyze_slope(_clay_slope(), method="bishop", compare_methods=True,
                      **_ODD_CIRCLE)
    assert r.method_solved and r.FOS is not None
    assert r.FOS_spencer is None and r.FOS_morgenstern_price is None
    assert r.theta_spencer is None and r.lambda_mp is None
    assert {"Spencer", "Morgenstern-Price"} <= set(
        r.methods_without_solution)
    t = compare_methods_table(_clay_slope(), **_ODD_CIRCLE)
    names = [row["method"] for row in t["rows"]]
    assert not any(n.startswith(("Spencer", "Morgenstern")) for n in names)
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
                            "no_driving"}
        assert rec["n"] == rec["not_converged"] + rec["no_driving"]
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
    ya = yield_acceleration(_clay_slope(), method="spencer", **_ODD_CIRCLE)
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
    g = _fill_slope(c=38.0)
    with pytest.raises(ValueError) as exc:
        fosm_fos(g, {"c_prime": {"mean": 38.0, "std": 12.0}},
                 method="spencer", **_FILL_CIRCLE)
    text = str(exc.value)
    assert "FOSM gives no result" in text
    assert "c_prime = 50" in text
    assert "mean 38 plus one standard deviation 12" in text
    assert "Spencer's method has no solution" in text
    # with a smaller spread every evaluation solves, and FOSM runs
    ok = fosm_fos(g, {"c_prime": {"mean": 38.0, "std": 2.0}},
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
    assert any(w.startswith("no equilibrium state at lambda = 0; bracket "
                            "started from") for w in res.warnings)
    r = analyze_slope(g, xc=6.0, yc=15.0, radius=12.0, method="spencer")
    assert r.method_solved and r.FOS == pytest.approx(res.fos, rel=1e-9)


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
