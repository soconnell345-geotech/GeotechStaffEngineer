"""One method across the search (GM-140 step 6).

The rule. Every family of surfaces in a search is scored by the method
named, with the interslice function named, so the reported minimum compares
like with like. A family the named method cannot score is reported under
the method that did score it, apart from the others.

Print: EM 1110-2-1902 4-3b(9)(a), p. 4-8 (state the method used); EM
C-2b, p. C-16 (the Ordinary Method of Slices "can only be used with
circular slip surfaces"); EM C-5a, p. C-26 (Spencer's method extended to
noncircular surfaces); FHWA NHI-06-088 Table 6-1, p. 6-21 (Bishop's the
minimum method, Spencer's where available); UFC 3-220-10 p. 362 ("Critical
failure surfaces can be missed as a result of non-convergence. The engineer
should be able to identify when convergence issues are present."); EM
4-2e(2), p. 4-7 (the initial surface of a noncircular search should
correspond to the estimated critical surface).

Every geometry here is made up. Each test failed on master before the
change (the departures D13, D42, D26 and D49's label part of
module_work/slope_integral_review/ACCOUNT.md), unless its docstring says it
guards something that already held.
"""

import math
import warnings

import pytest

from slope_stability import search as S
from slope_stability import search_supplements as SS
from slope_stability.analysis import (analyze_slope, search_critical_surface,
                                      search_rapid_drawdown)
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.search import (search_noncircular, search_pso,
                                    search_weak_layer_biased, search_de)
from slope_stability.search_supplements import (
    CIRCULAR, NONCIRCULAR, TOE_CIRCLE, ABOVE_TOE, INFINITE_SLOPE,
    check_infinite_slope)
from slope_stability.solution import MethodSolution


def _sand_face():
    """A made-up 10 m face at 2H:1V in dry sand, phi' 35 deg, c' = 0."""
    return SlopeGeometry(
        surface_points=[(0, 0), (8, 0), (28, 10), (46, 10)],
        soil_layers=[SlopeSoilLayer(name="sand", top_elevation=10.0,
                                    bottom_elevation=-8.0, gamma=19.0,
                                    phi=35.0, c_prime=0.0)])


_SAND_CLOSED = math.tan(math.radians(35.0)) / 0.5        # 1.4004
# windows that make every circle span the face (its toe at x = 8, its crest
# at x = 28), so the planes can govern
_OFF_FACE = dict(x_entry_range=(0.0, 8.5), x_exit_range=(27.5, 46.0))


def _fill():
    """A made-up 9 m fill slope with cohesion and friction."""
    return SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (25, 9), (44, 9)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=9.0,
                                    bottom_elevation=-7.0, gamma=19.0,
                                    phi=24.0, c_prime=9.0)])


def _two_layers():
    """A made-up 9 m slope, a stiff fill over a soft clay."""
    return SlopeGeometry(
        surface_points=[(0, 9), (11, 9), (29, 0), (48, 0)],
        soil_layers=[
            SlopeSoilLayer(name="fill", top_elevation=9.0,
                           bottom_elevation=1.5, gamma=19.0, phi=33.0,
                           c_prime=15.0),
            SlopeSoilLayer(name="soft clay", top_elevation=1.5,
                           bottom_elevation=-9.0, gamma=17.0, phi=11.0,
                           c_prime=6.0)])


def _weak_cap():
    """A made-up 8 m cut at 1H:1V: 3 m of weak soil over a strong one, so a
    circle passing above the toe governs."""
    return SlopeGeometry(
        surface_points=[(0, 0), (16, 0), (24, 8), (50, 8)],
        soil_layers=[
            SlopeSoilLayer(name="weak", top_elevation=8.0,
                           bottom_elevation=5.0, gamma=18.0, phi=24.0,
                           c_prime=2.0),
            SlopeSoilLayer(name="strong", top_elevation=5.0,
                           bottom_elevation=-9.0, gamma=22.0, phi=38.0,
                           c_prime=150.0)])


def _by_family(res, family):
    return [r for r in res.candidate_methods if r["family"] == family]


def _search(geom, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return search_critical_surface(geom, **kw)


@pytest.fixture
def solves(monkeypatch):
    """Record every (method, interslice function, circular) the search's
    solver is asked for."""
    seen = []
    real = S.solve_method

    def spy(slices, slip, method, f_interslice="half_sine", tol=1e-4,
            reinf_forces=None):
        seen.append((method, f_interslice,
                     bool(getattr(slip, "is_circular", True))))
        return real(slices, slip, method, f_interslice, tol, reinf_forces)

    monkeypatch.setattr(S, "solve_method", spy)
    return seen


# ---------------------------------------------------------------------------
# Item 2: the planes are scored by the method named
# ---------------------------------------------------------------------------

def test_spencer_named_scores_the_planes_by_spencer():
    """One method across the search: with Spencer's method named the
    shallow planes are scored by Spencer's method (before, always by
    Janbu's simplified method), and the family's entry says so
    (EM 1110-2-1902 4-3b(9)(a), p. 4-8: state the method used)."""
    res = _search(_sand_face(), method="spencer", **_OFF_FACE)
    (rec,) = _by_family(res, INFINITE_SLOPE)
    assert rec["method"] == "spencer" and rec["named"]
    assert rec["f_interslice"] == "constant"
    assert rec["reason"] == ""
    (check,) = res.infinite_slope_checks
    assert check["method"] == "spencer"
    # GM-56: how many of the face's planes the named method solves
    assert check["n_solved_by_named"] == check["n_planes"] == 6
    # the closed form, tan(phi') / tan(beta) (Duncan, Wright and Brandon
    # 2014, Ch. 14; `infinite_slope_fos`)
    assert rec["FOS"] == pytest.approx(_SAND_CLOSED, rel=0.005)
    if res.governing_surface == INFINITE_SLOPE:
        assert res.critical.method == "Spencer"
        assert res.method_used == ""


def test_janbu_named_scores_the_planes_by_janbu():
    """(A guard: Janbu's simplified method scored the planes before, and
    still does when it is the method named.)"""
    res = _search(_sand_face(), method="janbu", **_OFF_FACE)
    (rec,) = _by_family(res, INFINITE_SLOPE)
    assert rec["method"] == "janbu" and rec["named"]
    assert rec["f_interslice"] is None


# ---------------------------------------------------------------------------
# Item 1: the interslice function named, in every family and kind of search
# ---------------------------------------------------------------------------

def test_gle_with_the_constant_function_in_every_family(solves):
    """GLE with the constant function named: every family's entry says
    GLE, constant, and no solve of the search takes another function
    (before, a search always took the half-sine function: D13)."""
    res = _search(_sand_face(), method="gle", f_interslice="constant",
                  nx=5, ny=5)
    families = {r["family"] for r in res.candidate_methods}
    assert families == {CIRCULAR, TOE_CIRCLE, ABOVE_TOE, INFINITE_SLOPE}
    for rec in res.candidate_methods:
        assert rec["method"] == "gle" and rec["named"], rec
        assert rec["f_interslice"] == "constant", rec
    assert res.named_method == "gle" and res.f_interslice == "constant"
    assert {f for m, f, _ in solves if m == "gle"} == {"constant"}
    assert res.critical.method == "GLE"


def test_gle_constant_search_is_the_spencer_search():
    """The GLE with the constant function is Spencer's method (Spencer
    1967): the two searches report the same factor of safety. Before, the
    GLE search took the half-sine function whatever was named."""
    kw = dict(nx=5, ny=5, toe_circles=False)
    gle = _search(_fill(), method="gle", f_interslice="constant", **kw)
    spencer = _search(_fill(), method="spencer", **kw)
    half = _search(_fill(), method="gle", **kw)
    assert gle.critical.FOS == pytest.approx(spencer.critical.FOS, abs=2e-4)
    assert half.f_interslice == "half_sine"
    assert abs(half.critical.FOS - spencer.critical.FOS) > 2e-4


@pytest.mark.parametrize("surface_type", ["noncircular", "noncircular_de",
                                          "pso", "weak_layer"])
def test_every_noncircular_search_takes_the_function_named(solves,
                                                           surface_type):
    """Morgenstern-Price with the trapezoidal function named: every
    polyline of every kind of noncircular search is solved with it."""
    res = _search(_two_layers(), method="morgenstern_price",
                  f_interslice="trapezoidal", surface_type=surface_type,
                  n_trials=40, nx=5, ny=4, seed=3, circle_check=False,
                  infinite_slope=False)
    polylines = {(m, f) for m, f, circular in solves if not circular}
    assert polylines == {("morgenstern_price", "trapezoidal")}
    if res.critical is not None:
        assert res.critical.method == "Morgenstern-Price"
        (rec,) = _by_family(res, NONCIRCULAR)
        assert rec["method"] == "morgenstern_price"
        assert rec["f_interslice"] == "trapezoidal"


def test_an_unknown_interslice_function_is_refused():
    with pytest.raises(ValueError, match="Unknown interslice function"):
        search_critical_surface(_fill(), method="gle",
                                f_interslice="half_cosine")
    with pytest.raises(ValueError, match="Unknown interslice function"):
        search_rapid_drawdown(_fill(), 8.0, 2.0, f_interslice="half_cosine")


# ---------------------------------------------------------------------------
# Item 1: the particle swarm and the weak-layer search use the method named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surface_type", ["pso", "weak_layer"])
def test_pso_and_weak_layer_use_morgenstern_price_when_named(solves,
                                                             surface_type):
    """A particle-swarm and a weak-layer search with Morgenstern-Price
    named use it (before, both always used Spencer's method: D13)."""
    res = _search(_two_layers(), method="morgenstern_price",
                  surface_type=surface_type, n_trials=60, nx=5, ny=4,
                  seed=2, circle_check=False, infinite_slope=False)
    assert {m for m, _f, circular in solves if not circular} == {
        "morgenstern_price"}
    assert res.critical is not None
    assert res.critical.method == "Morgenstern-Price"
    (rec,) = _by_family(res, NONCIRCULAR)
    assert rec["method"] == "morgenstern_price" and rec["named"]
    assert rec["f_interslice"] == "half_sine"
    assert res.method_used == ""


def test_the_search_functions_take_a_method(solves):
    """`search_pso` and `search_weak_layer_biased` take the method (before,
    they had no such argument)."""
    kw = dict(x_entry_range=(2.0, 14.0), x_exit_range=(24.0, 44.0),
              n_slices=20, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pso = search_pso(_two_layers(), n_particles=12, n_iterations=3,
                         method="janbu", **kw)
        weak = search_weak_layer_biased(_two_layers(), n_trials=40,
                                        method="janbu", **kw)
    assert {m for m, _f, _c in solves} == {"janbu"}
    for res in (pso, weak):
        assert res.critical is None or res.critical.method == "Janbu"


# ---------------------------------------------------------------------------
# Items 4 and 5: a family the named method cannot score, and the headline
# ---------------------------------------------------------------------------

def test_bishop_named_on_a_cohesionless_face():
    """Bishop named on a cohesionless face: the planar family is reported
    apart, under the method that scored it, and the headline and the named
    method's own lowest value are both in the result."""
    res = _search(_sand_face(), method="bishop", **_OFF_FACE)
    (planar,) = _by_family(res, INFINITE_SLOPE)
    assert planar["method"] == "spencer" and not planar["named"]
    assert "centre of a circle" in planar["reason"]
    assert "Bishop 1955" in planar["reason"]
    for family in (CIRCULAR, TOE_CIRCLE, ABOVE_TOE):
        for rec in _by_family(res, family):
            assert rec["method"] == "bishop" and rec["named"]
    # the lowest over all families governs: the planes here
    assert res.governing_surface == INFINITE_SLOPE
    assert res.critical.FOS == pytest.approx(planar["FOS"], abs=1e-4)
    assert res.critical.method == "Spencer"
    assert res.method_used == "spencer"
    own = res.named_method_lowest
    named = [r for r in res.candidate_methods if r["named"]]
    assert own["method"] == "bishop"
    assert own["FOS"] == pytest.approx(min(r["FOS"] for r in named))
    assert own["FOS"] > res.critical.FOS
    assert res.named_method == "bishop"
    w = res.warnings[0]
    assert "Spencer's method" in w and "Bishop's simplified method" in w
    assert "method named" in w and f"{own['FOS']:.3f}" in w
    d = res.to_dict()
    assert d["method_used"] == "spencer"
    assert d["named_method_lowest"]["FOS"] == own["FOS"]
    assert d["candidate_methods"] == res.candidate_methods
    chosen = {c["what"]: c for c in res.choices}
    apart = chosen["families scored by another method than the one named"]
    assert "infinite slope: Spencer's method" in apart["chosen"]
    assert "C-5a, p. C-26" in apart["basis"]
    assert "Table 6-1, p. 6-21" in apart["basis"]


def test_the_named_method_governs_without_a_statement():
    """Where the governing family was scored by the method named, nothing
    is said of another method."""
    res = _search(_fill(), method="bishop")
    assert res.method_used == "" and res.named_method_lowest is None
    assert all(r["named"] for r in res.candidate_methods)
    assert "method_used" not in res.to_dict()


def test_the_ordinary_method_of_slices_named_on_planes_and_polylines():
    """The Ordinary Method of Slices "can only be used with circular slip
    surfaces" (EM 1110-2-1902 C-2b, p. C-16; D26): named in a search, the
    planes and the polylines are scored by Spencer's method and reported
    apart. Before, it scored both."""
    res = _search(_sand_face(), method="fellenius", **_OFF_FACE)
    (planar,) = _by_family(res, INFINITE_SLOPE)
    assert planar["method"] == "spencer" and not planar["named"]
    assert "C-2b, p. C-16" in planar["reason"]
    non = _search(_fill(), method="fellenius", surface_type="noncircular",
                  n_trials=60, seed=4, circle_check=False)
    (rec,) = _by_family(non, NONCIRCULAR)
    assert rec["method"] == "spencer" and not rec["named"]
    assert non.critical.method == "Spencer"
    assert non.method_used == "spencer"
    # no family of this search was scored by the method named
    assert non.named_method_lowest["FOS"] is None
    assert "scored no family" in non.named_method_lowest["note"]


def test_a_noncircular_search_named_for_bishop_is_labelled_spencer():
    """A polyline is scored by Spencer's method when Bishop's is named, and
    the direct search's result says Spencer (before, `search_noncircular`
    labelled it 'Bishop': D49)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = search_noncircular(_fill(), (2.0, 14.0), (22.0, 40.0),
                                 n_trials=40, seed=5, method="bishop")
    assert res.critical is not None
    assert res.critical.method == "Spencer"


def _unsolved(method):
    return MethodSolution(method=method, fos=None, solved=False,
                          reason="made not to converge, for the test",
                          kind="not_converged")


def test_a_method_that_solves_none_of_the_planes_gives_way_in_order(
        monkeypatch):
    """Item 4's order. Where the named method solves none of a face's
    planes the face is scored by Spencer's method; where Spencer's solves
    none either, by Janbu's simplified method; and the family is still
    reported, under the method that scored it, with the reason (GM-56: a
    rigorous method named must not lose the infinite-slope mechanism)."""
    real = S.solve_method
    fail = {"morgenstern_price"}

    def solver(slices, slip, method, *a, **kw):
        if method in fail and not getattr(slip, "is_circular", True):
            return _unsolved(method)
        return real(slices, slip, method, *a, **kw)

    monkeypatch.setattr(S, "solve_method", solver)
    res = _search(_sand_face(), method="morgenstern_price", nx=4, ny=4,
                  toe_circles=False, **_OFF_FACE)
    (rec,) = _by_family(res, INFINITE_SLOPE)
    assert rec["method"] == "spencer" and not rec["named"]
    assert "solved none of the family's surfaces" in rec["reason"]
    (check,) = res.infinite_slope_checks
    assert check["n_solved_by_named"] == 0 and check["n_planes"] == 6
    # the planes the named method lost are counted (GM-140 step 2)
    lost = [r for r in res.no_solution if r["family"] == INFINITE_SLOPE]
    assert sum(r["not_converged"] for r in lost
               if r["method"] == "morgenstern_price") == 6

    fail.add("spencer")
    res = _search(_sand_face(), method="morgenstern_price", nx=4, ny=4,
                  toe_circles=False, **_OFF_FACE)
    (rec,) = _by_family(res, INFINITE_SLOPE)
    assert rec["method"] == "janbu" and not rec["named"]
    assert "Spencer's method solved none of them either" in rec["reason"]
    assert rec["FOS"] == pytest.approx(_SAND_CLOSED, rel=0.005)
    assert res.candidates[INFINITE_SLOPE] == rec["FOS"]


def test_a_main_search_the_method_solves_nothing_of_gives_way(monkeypatch):
    """The main family too: a noncircular search whose method solves none
    of its polylines is scored by Spencer's method and reported under
    it."""
    real = S.solve_method

    def solver(slices, slip, method, *a, **kw):
        if method == "morgenstern_price":
            return _unsolved(method)
        return real(slices, slip, method, *a, **kw)

    monkeypatch.setattr(S, "solve_method", solver)
    res = _search(_fill(), method="morgenstern_price",
                  surface_type="noncircular", n_trials=40, seed=6,
                  circle_check=False, infinite_slope=False)
    (rec,) = _by_family(res, NONCIRCULAR)
    assert rec["method"] == "spencer" and not rec["named"]
    assert res.method_used == "spencer"
    assert res.critical.method == "Spencer"


# ---------------------------------------------------------------------------
# Item 6: truthful labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method, label", [
    ("morgenstern_price", "Morgenstern-Price"), ("gle", "GLE"),
    ("spencer", "Spencer"), ("bishop", "Bishop")])
def test_a_governing_toe_circle_is_labelled_with_its_method(method, label):
    """A result of the toe-circle families carries its method's own label
    (before: 'Morgenstern_price' and 'Gle'; D49, GM-81), and its entry
    names the method, the function and the Bishop screen that picked the
    circles."""
    res = _search(_weak_cap(), method=method, n_slices=30, nx=4, ny=4,
                  infinite_slope=False)
    assert res.governing_surface in (TOE_CIRCLE, ABOVE_TOE)
    assert res.critical.method == label
    (rec,) = _by_family(res, res.governing_surface)
    assert rec["method"] == method and rec["named"]
    assert rec["FOS"] == pytest.approx(res.critical.FOS, abs=1e-4)
    if method == "bishop":
        assert "screen" not in rec and rec["f_interslice"] is None
    else:
        # a stated run-time device (item 3): the value is the method's own
        assert rec["screen"] == "bishop"
        chosen = {c["what"]: c for c in res.choices}
        device = chosen["screens by another method (run-time devices)"]
        assert "scanned by Bishop's simplified method" in device["chosen"]
        from slope_stability.slip_surface import circle_of
        again = analyze_slope(
            _weak_cap(), slip_surface=circle_of(res.critical),
            method=method, n_slices=30)
        assert again.FOS == pytest.approx(res.critical.FOS, rel=2e-3)


def test_a_grid_circle_by_morgenstern_price_is_labelled_so():
    res = _search(_fill(), method="morgenstern_price", nx=4, ny=4,
                  toe_circles=False, infinite_slope=False)
    assert res.critical.method == "Morgenstern-Price"
    entry = _search(_fill(), method="gle", nx=4, ny=4,
                    surface_type="entry_exit", toe_circles=False,
                    infinite_slope=False)
    assert entry.critical.method == "GLE"


# ---------------------------------------------------------------------------
# Surfaces the method could not solve, lower by the screen
# ---------------------------------------------------------------------------

def test_screen_values_of_surfaces_the_method_could_not_solve(monkeypatch):
    """Where surfaces the named method could not solve have a lower value
    by the screen than the minimum reported, the result says so, with their
    count and the lowest screen value (UFC 3-220-10 p. 362: critical
    surfaces can be missed through non-convergence; the engineer should be
    able to tell)."""
    plain = _search(_fill(), method="spencer", bishop_screen=True, nx=6,
                    ny=6, toe_circles=False, infinite_slope=False,
                    refine=False)
    assert plain.screen_unsolved == {}
    xc = plain.critical.xc
    real = S.solve_method

    def solver(slices, slip, method, *a, **kw):
        # Spencer's method made to have no solution on the circles about
        # the critical centre and beside it
        if (method == "spencer" and getattr(slip, "is_circular", False)
                and abs(slip.xc - xc) < 3.0):
            return _unsolved(method)
        return real(slices, slip, method, *a, **kw)

    monkeypatch.setattr(S, "solve_method", solver)
    res = _search(_fill(), method="spencer", bishop_screen=True, nx=6,
                  ny=6, toe_circles=False, infinite_slope=False,
                  refine=False)
    rec = res.screen_unsolved
    assert rec["n"] >= 1
    assert rec["lowest_screen_FOS"] < res.critical.FOS
    assert rec["by_family"][CIRCULAR]["screen"] == "bishop"
    assert res.to_dict()["screen_unsolved"] == rec
    (w,) = [w for w in res.warnings if "screen_unsolved" in w]
    assert f"{rec['n']} surface(s)" in w
    assert f"{rec['lowest_screen_FOS']:.3f}" in w
    assert "UFC 3-220-10 p. 362" in w


def test_screen_unsolved_record_counts_only_those_below_the_minimum():
    notes = [("main", "bishop", 1.10), ("main", "bishop", 1.31),
             ("toe_circle", "bishop", 1.05), ("toe_circle", "bishop", 1.40)]
    rec = S.screen_unsolved_record(notes, 1.20, rename={"main": CIRCULAR})
    assert rec == {"n": 2, "lowest_screen_FOS": 1.05, "by_family": {
        CIRCULAR: {"n": 1, "lowest_screen_FOS": 1.10, "screen": "bishop"},
        TOE_CIRCLE: {"n": 1, "lowest_screen_FOS": 1.05,
                     "screen": "bishop"}}}
    assert S.screen_unsolved_record(notes, 1.0) == {}
    assert S.screen_unsolved_record(notes, None)["n"] == 4


# ---------------------------------------------------------------------------
# From the GLE step: the low m_alpha finding and the planar family
# ---------------------------------------------------------------------------

def test_the_planar_family_keeps_its_findings_with_its_lowest_value():
    """The planar family's report keeps what the printed acceptability
    checks find on its lowest plane with the family's lowest value (the
    low m_alpha of a steep end among them: EM 1110-2-1902 C-10b), as the
    single-surface analysis of that plane finds it."""
    geom = _sand_face()
    out = check_infinite_slope(geom, method="spencer")
    (check,) = out["checks"]
    one = analyze_slope(geom, slip_surface=out["slip"], method="spencer")
    assert one.FOS == pytest.approx(check["FOS"], abs=1e-4)
    assert sorted(check["flags"]) == sorted(one.flags)
    res = _search(geom, method="spencer", **_OFF_FACE)
    (rec,) = _by_family(res, INFINITE_SLOPE)
    assert rec["flags"] == check["flags"]


# ---------------------------------------------------------------------------
# A noncircular minimum checked against the circle
# ---------------------------------------------------------------------------

def test_a_noncircular_minimum_is_checked_against_the_circle():
    """A noncircular search does not report a minimum above the critical
    circle's value by the same method without saying so: the circular
    search is run first and its critical circle, as the polyline of its
    slice-base points, is one of the trials (EM 1110-2-1902 4-2e(2),
    p. 4-7: the initial surface should correspond to the estimated
    critical surface). Before, 30 random polylines reported their own
    minimum alone."""
    kw = dict(method="spencer", n_slices=25)
    circle = _search(_fill(), infinite_slope=False, **kw)
    res = _search(_fill(), surface_type="noncircular", n_trials=30, seed=7,
                  infinite_slope=False, **kw)
    cc = res.circle_check
    assert cc["checked"] is True
    assert cc["circle"]["FOS"] == pytest.approx(circle.critical.FOS,
                                                abs=1e-4)
    # no more than the circle's value plus the difference of discretisation
    # (the arc against the polyline of its 25 slice bases)
    assert res.critical.FOS <= circle.critical.FOS * 1.003
    assert cc["noncircular_minimum"] > circle.critical.FOS
    assert cc["governs"] is True
    assert cc["FOS_as_polyline"] == pytest.approx(res.critical.FOS, abs=1e-4)
    assert not res.critical.is_circular
    assert len(res.critical.slip_points) >= 20
    (w,) = [w for w in res.warnings if "circle_check" in w]
    assert f"{cc['noncircular_minimum']:.3f}" in w
    assert "4-2e(2), p. 4-7" in w
    chosen = {c["what"]: c for c in res.choices}
    assert chosen["noncircular minimum checked against the critical "
                  "circle"]["chosen"] is True


def test_a_search_that_ran_no_circle_says_it_was_not_checked():
    """Where no circular search ran, the result says the noncircular
    minimum was not checked against a circle."""
    res = _search(_fill(), method="spencer", surface_type="noncircular",
                  n_trials=30, seed=7, circle_check=False,
                  infinite_slope=False)
    assert res.circle_check["checked"] is False
    assert "not checked against a circle" in res.circle_check["note"]
    assert res.to_dict()["circle_check"]["checked"] is False
    chosen = {c["what"]: c for c in res.choices}
    assert chosen["noncircular minimum checked against the critical "
                  "circle"]["chosen"] is False
    for fn, kw in ((search_noncircular, dict(n_trials=20)),
                   (search_pso, dict(n_particles=8, n_iterations=2)),
                   (search_weak_layer_biased, dict(n_trials=20)),
                   (search_de, dict(maxiter=2, popsize=4,
                                    n_seed_trials=20))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = fn(_fill(), (2.0, 14.0), (22.0, 40.0), seed=1,
                        n_slices=20, **kw)
        assert direct.circle_check["checked"] is False, fn.__name__
        assert "no circular search ran" in direct.circle_check["note"]
    assert _search(_fill(), method="bishop").circle_check == {}


def test_the_circle_as_a_polyline_is_the_circle():
    """The polyline through a circle's slice-base points scores what the
    circle scores, to the difference of discretisation."""
    geom = _fill()
    circle = _search(geom, method="spencer", n_slices=30, nx=5, ny=5,
                     toe_circles=False, infinite_slope=False).critical
    poly = S.circle_as_polyline(geom, circle, 30)
    assert poly.points[0][0] == pytest.approx(circle.x_entry, abs=1e-6)
    assert poly.points[-1][0] == pytest.approx(circle.x_exit, abs=1e-6)
    one = analyze_slope(geom, slip_surface=poly, method="spencer",
                        n_slices=30)
    assert one.FOS == pytest.approx(circle.FOS, rel=0.003)


# ---------------------------------------------------------------------------
# The drawdown search
# ---------------------------------------------------------------------------

def _dam():
    """A made-up 9 m bank: a sand shell on a clay with an R envelope."""
    return SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (28, 9), (46, 9)],
        soil_layers=[
            SlopeSoilLayer(name="shell", top_elevation=9.0,
                           bottom_elevation=0.0, gamma=19.0, gamma_sat=20.0,
                           phi=36.0, c_prime=0.0),
            SlopeSoilLayer(name="clay", top_elevation=0.0,
                           bottom_elevation=-8.0, gamma=18.0, gamma_sat=19.0,
                           phi=27.0, c_prime=4.0, R_c=22.0, R_phi=15.0)])


def test_the_drawdown_search_names_its_solve_for_every_family():
    """The drawdown search takes the interslice function and every family
    is scored by the one staged solve with it; each family's entry names
    the procedure and the function, and the surface's label follows the
    function (before: 'Spencer' whatever function was named)."""
    kw = dict(method="corps_2stage", surface_type="circular", nx=3, ny=3,
              n_slices=20, refine=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        const = search_rapid_drawdown(_dam(), 7.5, 2.0, **kw)
        half = search_rapid_drawdown(_dam(), 7.5, 2.0,
                                     f_interslice="half_sine", **kw)
    for res, f, label in ((const, "constant", "Spencer"),
                          (half, "half_sine", "Morgenstern-Price")):
        s = res.search
        assert s.candidate_methods, f
        assert {r["family"] for r in s.candidate_methods} == set(
            s.candidates)
        for rec in s.candidate_methods:
            assert rec["method"] == "corps_2stage" and rec["named"]
            assert rec["f_interslice"] == f
        assert s.named_method == "corps_2stage" and s.f_interslice == f
        assert s.method_used == ""
        assert s.critical.method == label
