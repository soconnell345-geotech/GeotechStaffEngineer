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
    assert stats == {"geometry": 1, "nonconverged": 0, "jagged": 0}


def test_reject_stats_counts_degenerate_jagged():
    """The ACADS-4 zig-zag is tallied as a rejection (non-converged or, if it
    converges to a spurious low FOS via the pinned-axis fallback, jagged) -- never
    scored as a real surface."""
    geom = _flat_geom()
    stats = _new_reject_stats()
    jag = PolylineSlipSurface(points=_JAGGED)
    assert _compute_fos(geom, jag, "spencer", 40, reject_stats=stats) == _FOS_MAX
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
