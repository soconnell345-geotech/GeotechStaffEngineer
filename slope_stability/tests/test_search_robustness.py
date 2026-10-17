"""SS-6 (v5.3 B2) regression tests: the weak-layer / noncircular-search
degenerate-surface guard. Jagged, near-vertical, or sliver polyline trial
surfaces must NOT be reported as the critical surface with a spurious low FOS.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.search import (
    _noncircular_admissible, _is_jagged, _compute_fos, _FOS_MAX,
    _new_reject_stats, _rejection_kwargs, _window_span,
    search_noncircular,
)


def _flat_geom():
    return SlopeGeometry(
        surface_points=[(20.0, 28.0), (43.0, 28.0), (68.0, 40.0), (84.0, 40.0)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=40.0,
                                    bottom_elevation=15.0, gamma=18.84,
                                    phi=20.0, c_prime=20.0)])


# The exact jagged surface a weak-layer search reported as "critical"
# (Spencer ~0.15) before the fix — base angles oscillate -79/+78/-36/-62/+80 deg.
_JAGGED = [(48.26, 30.53), (50.29, 20.07), (52.32, 29.39),
           (54.35, 27.94), (56.38, 24.17), (58.41, 35.4)]


def test_degenerate_jagged_surface_rejected():
    """The core fix: the ACADS-4 zig-zag makes the rigorous GLE non-converge AND
    trips the low-FOS jaggedness gate, so `_compute_fos` returns _FOS_MAX for it
    -- a search can no longer pick its spurious ~0.15 FOS -- robustly at every
    slice count (the pre-fix legacy Spencer fallback returned ~0.15)."""
    geom = _flat_geom()
    slip = PolylineSlipSurface(points=_JAGGED)
    assert _is_jagged(slip) is True
    for ns in (20, 30, 40, 60):
        assert _compute_fos(geom, slip, "spencer", ns) == _FOS_MAX
        assert _compute_fos(geom, slip, "gle", ns) == _FOS_MAX


def test_low_fos_gate_only_applies_to_jagged():
    """A jagged surface with a HIGH FOS is NOT geometry-rejected (the gate is
    scoped to low FOS, so the random search keeps exploring); it is a smooth
    LOW-FOS surface that must survive, and a jagged LOW-FOS one that must not."""
    slip = PolylineSlipSurface(points=_JAGGED)
    # jagged -> flagged
    assert _is_jagged(slip) is True
    # smooth concave seam-following surface -> not flagged
    smooth = PolylineSlipSurface(points=[
        (41.99, 28.0), (49.10, 26.66), (56.20, 28.25),
        (63.31, 30.30), (70.42, 35.56), (77.53, 40.0)])
    assert _is_jagged(smooth) is False


def test_sliver_surface_rejected_by_geometry():
    """A too-short-span / too-few-slice sliver is rejected by the geometric
    admissibility check before the solve."""
    geom = _flat_geom()               # slope width 64 m -> min span ~9.6 m
    sliver = PolylineSlipSurface(points=[(50.0, 28.0), (52.0, 25.0),
                                         (54.0, 28.0)])
    slices = build_slices(geom, sliver, 30)
    assert _noncircular_admissible(sliver, slices, geom, 30) is False
    assert _compute_fos(geom, sliver, "spencer", 30) == _FOS_MAX


def test_smooth_valid_noncircular_surface_is_admissible():
    """A smooth concave weak-seam-following surface passes the guards and gets a
    real (converged) FOS -- the fix does not reject valid surfaces."""
    geom = _flat_geom()
    smooth = PolylineSlipSurface(points=[
        (41.99, 28.0), (49.10, 26.66), (56.20, 28.25),
        (63.31, 30.30), (70.42, 35.56), (77.53, 40.0)])
    slices = build_slices(geom, smooth, 40)
    assert _noncircular_admissible(smooth, slices, geom, 40) is True
    fos = _compute_fos(geom, smooth, "spencer", 40)
    assert 0.5 < fos < 3.0 and fos < _FOS_MAX


def _wide_geom():
    """A 10 m slope embedded in a 200 m-wide model (long toe + long crest bench),
    where a fraction-of-model-width span floor (old 0.15*200 = 30 m) would wrongly
    reject a legitimate localized failure only ~14 m across."""
    return SlopeGeometry(
        surface_points=[(0, 0), (80, 0), (100, 10), (200, 10)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10.0,
                                    bottom_elevation=-20.0, gamma=19.0,
                                    phi=25.0, c_prime=10.0)])


def test_span_floor_anchored_to_slope_not_model_width():
    """Finding #3(a): a localized ~14 m failure in a 200 m-wide model is admissible
    (old 15%-of-width floor = 30 m would have silently rejected it); the floor is
    anchored to slope height (0.5*H) or the entry/exit window, not model width."""
    geom = _wide_geom()
    surf = PolylineSlipSurface(points=[(88, 4), (93, 1), (98, 3), (102, 10)])
    slices = build_slices(geom, surf, 30)
    span = slices[-1].x_right - slices[0].x_left
    assert span < 0.15 * 200                       # would fail the old floor
    assert _noncircular_admissible(surf, slices, geom, 30) is True
    assert _noncircular_admissible(surf, slices, geom, 30,
                                   window_span=25.0) is True
    # a genuine sliver is still rejected (0.5*H = 5 m floor)
    sliver = PolylineSlipSurface(points=[(92, 6.0), (94, 3.5), (96, 7.0)])
    sl2 = build_slices(geom, sliver, 30)
    assert _noncircular_admissible(sliver, sl2, geom, 30) is False


def test_window_span_helper():
    assert _window_span((5, 15), (25, 45)) == pytest.approx(25.0)   # 35 - 10
    assert _window_span(None, (25, 45)) is None


def test_reject_stats_counts_geometry():
    """Finding #3(b): a sliver rejection is tallied under 'geometry'."""
    geom = _flat_geom()
    stats = _new_reject_stats()
    sliver = PolylineSlipSurface(points=[(50.0, 28.0), (52.0, 25.0), (54.0, 28.0)])
    assert _compute_fos(geom, sliver, "spencer", 30, reject_stats=stats) == _FOS_MAX
    assert stats == {"geometry": 1, "nonconverged": 0, "jagged": 0,
                     "refused": None}


def test_reject_stats_counts_degenerate_jagged():
    """The ACADS-4 zig-zag is tallied as a rejection (non-converged or, if it
    converges to a spurious low FOS via the pinned-axis fallback, jagged) -- never
    scored as a real surface."""
    geom = _flat_geom()
    stats = _new_reject_stats()
    jag = PolylineSlipSurface(points=_JAGGED)
    # since GM-127 its m_alpha < 0.2 leaves it out first (Slide2's -112)
    info = {}
    assert _compute_fos(geom, jag, "spencer", 40, reject_stats=stats,
                        info=info) == _FOS_MAX
    assert info["m_alpha_min"] < 0.2
    assert stats["nonconverged"] + stats["jagged"] == 0
    # with that check off, the other gates still refuse it
    assert _compute_fos(geom, jag, "spencer", 40, reject_stats=stats,
                        m_alpha_check=False) == _FOS_MAX
    assert stats["geometry"] == 0
    assert stats["nonconverged"] + stats["jagged"] == 1


def test_majority_rejection_warns():
    """A search that rejects a MAJORITY of its trials warns instead of silently
    returning an under-resolved result; a minority does not."""
    with pytest.warns(UserWarning, match="rejected 10/15"):
        kw = _rejection_kwargs({"geometry": 8, "nonconverged": 2, "jagged": 0}, 15)
    assert kw["n_rejected_geometry"] == 8 and kw["n_rejected_nonconverged"] == 2
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        _rejection_kwargs({"geometry": 1, "nonconverged": 0, "jagged": 0}, 15)
    assert not any("rejected" in str(w.message) for w in rec)


def test_search_result_exposes_rejection_counts():
    """A real noncircular search populates the SearchResult rejection counters
    (and to_dict surfaces them when nonzero)."""
    geom = _flat_geom()
    res = search_noncircular(geom, x_entry_range=(41, 50), x_exit_range=(70, 78),
                             n_trials=40, n_slices=30, seed=3)
    for v in (res.n_rejected_geometry, res.n_rejected_nonconverged,
              res.n_rejected_jagged):
        assert isinstance(v, int) and v >= 0
    total = (res.n_rejected_geometry + res.n_rejected_nonconverged
             + res.n_rejected_jagged)
    d = res.to_dict()
    if total:
        assert d["n_rejected"]["nonconverged"] == res.n_rejected_nonconverged


def test_circular_surfaces_unaffected():
    """The guards are noncircular-only; a valid circular surface still evaluates
    to a real FOS identical to a direct bishop_fos call."""
    from slope_stability.methods import bishop_fos
    geom = SlopeGeometry(
        surface_points=[(20.0, 35.0), (40.0, 35.0), (60.0, 25.0), (70.0, 25.0)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=35.0,
                                    bottom_elevation=15.0, gamma=20.0,
                                    phi=10.0, c_prime=32.0)])
    slip = CircularSlipSurface(51.96, 42.94, 20.47)   # V-026 critical circle
    fos = _compute_fos(geom, slip, "bishop", 60)
    assert fos < _FOS_MAX
    assert fos == pytest.approx(bishop_fos(build_slices(geom, slip, 60), slip),
                                rel=1e-9)


# ---------------------------------------------------------------------------
# SS-5 generalized (v5.4 F2c): below-the-rigid-base one-sided-plunge rejection.
# A circle that drops below the deepest soil layer right after entry and
# re-emerges only near the exit leaves a ONE-SIDED surviving fragment; the old
# interior-only SS-5 check missed it, so a spurious low FOS could win a search
# (Slide2 #85 steep clay: ~0.72 on 12 of 40 slices, ~4 ft below the rigid base).
# ---------------------------------------------------------------------------

def _thin_base_geom():
    """Steep saturated-clay slope on a rigid base at el 10 (Slide2 #85 family)."""
    return SlopeGeometry(
        surface_points=[(15.0, 10.0), (25.0, 30.0), (57.0, 30.0)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=30.0,
            bottom_elevation=10.0, gamma=15.4, phi=0.0, c_prime=16.8, cu=16.8,
            analysis_mode="undrained")])


def test_below_base_one_sided_plunge_rejected():
    """A circle dipping well below the deepest layer over most of its span
    (one-sided fragment) is rejected: build_slices raises and _compute_fos
    returns _FOS_MAX, so a search cannot pick its spurious low FOS."""
    geom = _thin_base_geom()
    slip = CircularSlipSurface(xc=29.3, yc=33.0, radius=26.9)   # bottom el ~6.1
    with pytest.raises(ValueError, match="below the bottom of the deepest"):
        build_slices(geom, slip, 40)
    assert _compute_fos(geom, slip, "bishop", 40) == _FOS_MAX


def test_search_returns_base_admissible_critical_surface():
    """With the guard, the circular-search critical surface is physically
    ADMISSIBLE -- its lowest arc point (yc - R) sits at/above the deepest layer
    bottom (the rigid base el 10), instead of a below-base degenerate fragment.
    A real FOS is returned (not _FOS_MAX)."""
    from slope_stability.analysis import search_critical_surface
    res = search_critical_surface(_thin_base_geom(), method="bishop",
                                  surface_type="circular", nx=20, ny=20,
                                  n_slices=40)
    c = res.critical
    assert 0.0 < c.FOS < _FOS_MAX
    # lowest arc point at/above the deepest layer bottom (el 10)
    assert c.yc - c.radius >= 10.0 - 0.25


# ---------------------------------------------------------------------------
# GM-87: the local refinement of the centre grid. It may only lower the
# grid's minimum, it stays inside the grid's centre ranges, and it keeps to
# circles that at most touch the base: build_slices drops up to
# max(2, 3 % of n_slices) below-base END slices as tangential slivers, and
# where the toe sits on the rigid base a circle dipping just under it sheds
# its toe slices and scores low. Without that rule the walk found such a
# circle on this model (0.265, refused at 90 slices).
# ---------------------------------------------------------------------------

_THIN_BOX = ((15.0, 57.0), (31.0, 70.0))      # profile width x (crest+1, +2H)


def test_dips_below_base():
    """A circle tangent to the base inside the slip mass is admissible. One
    whose lowest point is 8 mm under the base just beyond the toe (as the
    walk without the rule found) loses two toe slices at 30 slices, scores
    0.265 against 0.278, and is refused at 90 slices."""
    from slope_stability.search import _dips_below_base
    geom = _thin_base_geom()
    touch = CircularSlipSurface(22.0, 32.0, 22.0)       # lowest point el 10
    assert not _dips_below_base(geom, touch, 30)
    assert len(build_slices(geom, touch, 90)) == 90
    shed = CircularSlipSurface(15.88, 10.0 + 28.06 - 0.008, 28.06)
    assert _dips_below_base(geom, shed, 30)
    assert len(build_slices(geom, shed, 30)) == 28      # slivers dropped
    assert _compute_fos(geom, shed, "bishop", 30) == pytest.approx(0.265,
                                                                   abs=0.002)
    with pytest.raises(ValueError, match="below the bottom of the deepest"):
        build_slices(geom, shed, 90)


@pytest.mark.parametrize("nx", [4, 10])
def test_refinement_lowers_the_grid_on_admissible_circles(nx):
    """The refined grid is below the grid's own minimum (0.281 / 0.278) and
    its circle builds at 90 slices too: 0.278, not the 0.265 sliver
    circle."""
    from slope_stability.search import grid_search
    geom = _thin_base_geom()
    plain = grid_search(geom, *_THIN_BOX, nx=nx, ny=nx, method="bishop")
    ref = grid_search(geom, *_THIN_BOX, nx=nx, ny=nx, method="bishop",
                      refine=True)
    c = ref.critical
    assert c.FOS < plain.critical.FOS
    assert c.FOS == pytest.approx(0.278, abs=0.002)
    assert len(build_slices(geom, CircularSlipSurface(c.xc, c.yc, c.radius),
                            90)) == 90
    (x_lo, x_hi), (y_lo, y_hi) = _THIN_BOX
    assert x_lo <= c.xc <= x_hi and y_lo <= c.yc <= y_hi
    assert ref.n_surfaces_evaluated > plain.n_surfaces_evaluated
    assert len(plain.grid_fos) == nx * nx


# ---------------------------------------------------------------------------
# A "converged" rigorous state that is not an equilibrium solution (GM-57
# review / GM-116): the searches keep a new best noncircular surface only if
# its solution holds about a second moment axis and with one slice more.
# ---------------------------------------------------------------------------

def _cut_weak_clay_over_sandstone():
    """A 10 m 1:1 cut: 3.5 m of weak clay over sandstone."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer(name="residual clay", top_elevation=10,
                           bottom_elevation=6.5, gamma=18.5, c_prime=3,
                           phi=22),
            SlopeSoilLayer(name="sandstone", top_elevation=6.5,
                           bottom_elevation=-20, gamma=23, c_prime=200,
                           phi=40)])


def _v043_sand():
    return SlopeGeometry(
        surface_points=[(0, 9), (10, 9), (20, 3), (30, 3)],
        soil_layers=[
            SlopeSoilLayer(name="fill", top_elevation=9, bottom_elevation=3.0,
                           c_prime=0.0, phi=37.0, gamma=17.0, gamma_sat=17.0),
            SlopeSoilLayer(name="softclay", top_elevation=3,
                           bottom_elevation=0.0, c_prime=20.0, phi=0.0,
                           gamma=20.0, gamma_sat=20.0, cu=20.0,
                           analysis_mode="undrained")])


def _weak_over_rock():
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("weak", top_elevation=10, bottom_elevation=6,
                           gamma=18, phi=26, c_prime=2),
            SlopeSoilLayer("rock", top_elevation=6, bottom_elevation=-10,
                           gamma=21, phi=40, c_prime=150)])


# Three surfaces searches reported as critical, each with a rigorous state
# GLE called converged at 30 slices, none an equilibrium solution:
_SPURIOUS = [
    # 13 m deep in the sandstone: 1.056 about gle's least-squares axis only
    # (Fellenius 21.7, Spencer by analyze_slope: no solution).
    (_cut_weak_clay_over_sandstone,
     [(18.031825345654877, 0.0), (25.918399678269633, -10.16448190701636),
      (33.80470085301144, -11.176172259294903),
      (41.692487049717926, -13.264717299624271),
      (49.57783909564785, 10.0)], 1.0563),
    # V-043's sand fill, DE seed 1 before GM-57: 0.717 at lambda -0.35 about
    # the pinned axis only; 29 / 31 slices give 1.90 / 1.55 at lambda +0.2.
    (_v043_sand,
     [(7.600990431562794, 9.0), (10.527033462319487, 2.6023385922059274),
      (13.453076493076178, 0.4049644518072615),
      (16.37911952383287, 0.7327078388145667),
      (19.305162554589565, 3.4169024672462616)], 0.7172),
    # diving at 75 deg into rock: 0.731 / 0.738 about two axes at 30 slices,
    # no solution at 29, 31 or 60 (Janbu 0.42, Fellenius 8.5).
    (_weak_over_rock,
     [(26.98652054187706, 6.986520541877059),
      (28.276212349877795, 2.092917772552611),
      (29.56590415787853, -3.734401082000133),
      (30.855595965879267, -2.0470654447606655),
      (32.14528777388, 1.2942698853560426),
      (33.43497958188074, 4.128710153451413),
      (34.72467138988147, 10.0)], 0.7308),
]


@pytest.mark.parametrize("make, pts, spurious, why", [
    (m, p, f, w) for (m, p, f), w in zip(_SPURIOUS, (
        "one moment axis only", "one moment axis only",
        "at 30 slices only, not at 31 or 60"))])
def test_spurious_rigorous_state_is_not_kept(make, pts, spurious, why):
    """Each also has m_alpha < 0.2 on a slice, so since GM-127 a search
    leaves it out before its solution is checked (Slide2's -112); with that
    check off, the solution check refuses it and keeps it to report."""
    geom = make()
    slip = PolylineSlipSurface(points=pts)
    info = {}
    assert _compute_fos(geom, slip, "spencer", 30, info=info) == _FOS_MAX
    assert info["m_alpha_min"] < 0.2
    assert _compute_fos(geom, slip, "spencer", 30,
                        m_alpha_check=False) == pytest.approx(
        spurious, abs=1e-3)                    # what the scorer alone gives
    stats = _new_reject_stats()
    assert _compute_fos(geom, slip, "spencer", 30, reject_stats=stats,
                        confirm_below=float("inf"),
                        m_alpha_check=False) == _FOS_MAX
    ref = stats["refused"]                     # kept, to be reported
    assert ref["FOS"] == pytest.approx(spurious, abs=1e-3)
    assert why in ref["reason"]
    assert ref["x_entry"] == pytest.approx(pts[0][0], abs=1e-3)


def test_valid_noncircular_solution_is_kept():
    """A surface whose solution holds about every axis and with one slice
    more keeps its FOS when confirmed."""
    geom = _v043_sand()
    slip = PolylineSlipSurface(points=[(4.8, 9.0), (9.22, 2.04),
                                       (13.64, 0.37), (18.05, 0.23),
                                       (22.47, geom.ground_elevation_at(22.47))])
    plain = _compute_fos(geom, slip, "spencer", 30)
    assert plain < 2.0
    assert _compute_fos(geom, slip, "spencer", 30,
                        confirm_below=float("inf")) == pytest.approx(plain)


@pytest.mark.parametrize("seed", [1, 2])
def test_noncircular_search_reports_a_reproducible_surface(seed):
    """GM-116 review: on the weak clay over sandstone the random noncircular
    search reported FS 0.90 (1.06 later) on the spurious deep surface above.
    Now the surface it reports is one analyze_slope solves with Spencer to
    the same FOS."""
    from slope_stability.analysis import analyze_slope, search_critical_surface
    geom = _cut_weak_clay_over_sandstone()
    res = search_critical_surface(geom, method="spencer",
                                  surface_type="noncircular", seed=seed,
                                  n_trials=200)
    c = res.critical
    again = analyze_slope(geom, slip_surface=PolylineSlipSurface(
        points=list(c.slip_points)), method="spencer", n_slices=30)
    assert again.method_solved
    assert again.FOS == pytest.approx(c.FOS, rel=1e-3)


def test_a_refused_lowest_surface_is_reported(monkeypatch):
    """GM-57 re-review: when the check refuses the lowest surface a search
    found, the result says so (FOS, where, why) instead of only counting
    it. Directed DE on the weak layer over rock: every surface it reaches
    below the rock outcrop solves about one moment axis only; none holds,
    so no surface is reported, and the warning says why. Those surfaces
    also have m_alpha < 0.2, which since GM-127 leaves them out first
    (`test_m_alpha.py` has that search as it runs); the solution check is
    measured here with the m_alpha check off, as Slide2 allows."""
    from slope_stability import search as search_mod
    from slope_stability.analysis import search_critical_surface
    monkeypatch.setattr(search_mod, "M_ALPHA_MIN", -float("inf"))
    res = search_critical_surface(_weak_over_rock(), method="spencer",
                                  surface_type="noncircular_de", seed=1,
                                  x_entry_range=(25.5, 28.0),
                                  x_exit_range=(29.0, 40.0))
    ref = res.refused_lowest
    assert ref is not None and "one moment axis only" in ref["reason"]
    assert res.critical is None
    w = [w for w in res.warnings if w.startswith("The lowest surface found")]
    assert len(w) == 1 and "No other surface held" in w[0]
    assert res.to_dict()["refused_lowest"]["FOS"] == ref["FOS"]

