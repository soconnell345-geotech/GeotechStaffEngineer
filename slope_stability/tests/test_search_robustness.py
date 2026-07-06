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
