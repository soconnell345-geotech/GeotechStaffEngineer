"""GM-131: a noncircular slip surface is sliced at its corners.

Even slices across a corner of a polyline took one segment's base angle
for their whole weight. One surface on Taylor's D = 2 profile read 0.650
on 30 slices, 0.807 on 29, 1.182 on 31 and 0.862 on 45, against 0.97
wherever every corner fell on a slice boundary; a search's minimum picked
out the low errors, and a fixed surface could err either way. Now every
vertex of the slip surface is a slice boundary, as are the breaks in the
ground and the surface's crossings of each layer boundary and of the water
table (Duncan, Wright & Brandon 2014, Ch. 6; Slide2 puts a slice boundary
at each vertex of a failure surface and at the piezometric surface); the
stretches between share out about ``n_slices`` slices in proportion to
their width, and each slice takes its base angle and length from the chord
between its two boundaries. Circles get the same boundaries, less the
corners, and the same chords since GM-134 (`test_circle_slicing.py`).
"""
import math

import pytest

from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.probabilistic import fosm_fos
from slope_stability.slices import build_slices, slice_boundaries
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)

H = 40 * 0.3048                     # Taylor D = 2: a 2H:1V slope 40 ft high
TAYLOR_D2 = 0.706                   # cu / (gamma H) = 0.120, N = 0.170

# A noncircular search's seed-3 surface on that profile, corners 12.06 m
# apart (the surface that showed the defect).
_SEED3 = [(26.862431427577228, 12.192),
          (38.92194369089153, 4.0050429263228615),
          (50.98145595420583, -1.7056566574780607),
          (63.04096821752013, -10.174301419905913),
          (75.10048048083443, 0.0)]


def _taylor_d2():
    """phi = 0 clay, cu = 0.12 gamma H, firm base H below the toe."""
    return SlopeGeometry(
        surface_points=[(0, H), (4 * H, H), (6 * H, 0), (10 * H, 0)],
        soil_layers=[SlopeSoilLayer("clay", H, -H, 20.0, cu=0.12 * 20.0 * H,
                                    analysis_mode="undrained")])


def _layered():
    """Two layers and a sloping water table; the ground breaks at x = 20
    and 30, inside the surface below."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("upper", 10, 4, 18.0, phi=28, c_prime=5),
            SlopeSoilLayer("lower", 4, -15, 19.0, phi=32, c_prime=10)],
        gwt_points=[(0, -6.0), (50, 4.0)])


_LAYERED_PTS = [(15.0, 0.0), (21.0, -3.0), (32.0, 1.0), (38.0, 10.0)]


def test_corners_ground_breaks_and_crossings_are_slice_boundaries():
    geom = _layered()
    slip = PolylineSlipSurface(points=_LAYERED_PTS)
    xs = slice_boundaries(geom, slip, 30)
    assert xs[0] == pytest.approx(15.0) and xs[-1] == pytest.approx(38.0)
    assert all(b > a for a, b in zip(xs, xs[1:]))
    assert len(xs) - 1 == 30

    def has(x):
        return any(abs(b - x) < 1e-6 for b in xs)

    assert has(21.0) and has(32.0)                 # the surface's corners
    assert has(20.0) and has(30.0)                 # the ground's breaks
    assert has(34.0)              # it rises through el. 4, the layer boundary
    # it dips below the water table (-6 + 0.2 x) and rises through it again
    assert has(13.5 / 0.7) and has(85.0 / 3.0)


def test_a_corner_close_to_another_boundary_is_kept():
    """A 4 m step 1 mm wide at the entry of a 38 m surface: its corner is a
    boundary, so the first slice is the step itself (when it was merged
    into the entry, a 1.25 m slice took its chord across the corner at
    -74.5 deg: 2.139 on 30 slices, 2.259 on 60)."""
    geom = SlopeGeometry(
        surface_points=[(0, 10), (20, 10), (40, 0), (70, 0)],
        soil_layers=[SlopeSoilLayer("clay", 10, -10, 19.0, c_prime=10,
                                    phi=25)])
    slip = PolylineSlipSurface(points=[(10.0, 10.0), (10.001, 6.0),
                                       (25.0, 0.0), (38.0, -1.0),
                                       (48.0, 0.0)])
    xs = slice_boundaries(geom, slip, 30)
    assert any(abs(x - 10.001) < 1e-9 for x in xs)
    first = build_slices(geom, slip, 30)[0]
    assert first.width == pytest.approx(0.001, abs=1e-6)
    assert math.degrees(first.alpha) == pytest.approx(-90.0, abs=0.1)
    fos = [analyze_slope(geom, slip_surface=slip, method="spencer",
                         n_slices=n).FOS for n in (30, 60)]
    assert fos == pytest.approx([2.521] * 2, abs=0.002)


def test_a_corner_at_an_end_is_that_end():
    """A vertex on the ground at the entry comes back from find_entry_exit
    a hair inside it; it is the entry, not a slice of its own."""
    geom = _layered()
    slip = PolylineSlipSurface(points=_LAYERED_PTS)
    xs = slice_boundaries(geom, slip, 30)
    assert xs[1] - xs[0] > 0.1 and xs[-1] - xs[-2] > 0.1


def test_chord_base_angle_and_length():
    geom = _layered()
    slip = PolylineSlipSurface(points=_LAYERED_PTS)
    xs = slice_boundaries(geom, slip, 30)
    slices = build_slices(geom, slip, 30)
    assert len(slices) == 30
    for s, a, b in zip(slices, xs, xs[1:]):
        dz = slip.slip_elevation_at(b) - slip.slip_elevation_at(a)
        assert s.width == pytest.approx(b - a)
        assert s.alpha == pytest.approx(math.atan2(dz, b - a))
        assert s.base_length == pytest.approx(math.hypot(b - a, dz))


def test_stretches_share_the_slices_by_width():
    """No slice more than twice as wide as another, and with fewer slices
    asked for than the eight stretches, one slice a stretch."""
    geom = _layered()
    slip = PolylineSlipSurface(points=_LAYERED_PTS)
    xs = slice_boundaries(geom, slip, 30)
    widths = [b - a for a, b in zip(xs, xs[1:])]
    assert max(widths) < 2.0 * min(widths)
    few = slice_boundaries(geom, slip, 3)
    assert [round(x, 3) for x in few] == [15.0, 19.286, 20.0, 21.0, 28.333,
                                          30.0, 32.0, 34.0, 38.0]


def test_a_circle_is_cut_at_the_ground_breaks_and_takes_its_chords():
    """GM-134: the crest and toe of Taylor's D = 2 slope are slice
    boundaries on a circle through them; each slice takes the angle and
    length of the chord between its boundaries, as a polyline's does, and
    its base elevation at its midpoint stays on the arc."""
    geom = _taylor_d2()
    slip = CircularSlipSurface(xc=5 * H, yc=1.908 * H,
                               radius=2.908 * H - 1e-6)
    xs = slice_boundaries(geom, slip, 40)
    assert any(abs(x - 4 * H) < 1e-9 for x in xs)
    assert any(abs(x - 6 * H) < 1e-9 for x in xs)
    slices = build_slices(geom, slip, 40)
    assert len(slices) == 40
    w = [s.width for s in slices]
    assert max(w) < 1.2 * min(w)
    for s, a, b in zip(slices, xs, xs[1:]):
        dz = slip.slip_elevation_at(b) - slip.slip_elevation_at(a)
        assert s.width == pytest.approx(b - a)
        assert s.alpha == pytest.approx(math.atan2(dz, b - a))
        assert s.base_length == pytest.approx(math.hypot(b - a, dz))
        assert s.z_base == pytest.approx(slip.slip_elevation_at(s.x_mid))


def test_a_circle_without_a_boundary_inside_is_cut_evenly():
    """No ground break, crossing or edge between the ends: n_slices
    slices of equal width."""
    geom = SlopeGeometry(
        surface_points=[(0, 20), (40, 0)],
        soil_layers=[SlopeSoilLayer("clay", 20, -20, 19.0, c_prime=10,
                                    phi=25)])
    slip = CircularSlipSurface(xc=20.0, yc=25.0, radius=18.0)
    w = [s.width for s in build_slices(geom, slip, 30)]
    assert len(w) == 30
    assert max(w) == pytest.approx(min(w), rel=1e-9)


def test_fixed_polyline_fos_does_not_depend_on_the_slice_count():
    """The seed-3 surface: 0.650 / 0.807 / 1.182 / 0.862 on 30 / 29 / 31 /
    45 even slices before; within 0.3 % from 29 to 45 slices now."""
    geom = _taylor_d2()
    slip = PolylineSlipSurface(points=_SEED3)
    fos = [analyze_slope(geom, slip_surface=slip, method="spencer",
                         n_slices=n).FOS for n in range(29, 46)]
    assert min(fos) == pytest.approx(0.9704, abs=0.001)
    assert max(fos) == pytest.approx(0.9732, abs=0.001)
    assert (max(fos) - min(fos)) / min(fos) < 0.01


def test_fixed_polyline_in_analyze_slope_and_fosm_agree():
    """A polyline given to analyze_slope solves on the slices asked for,
    and FOSM's mean-value FOS on it is the same number."""
    geom = _taylor_d2()
    slip = PolylineSlipSurface(points=_SEED3)
    r = analyze_slope(geom, slip_surface=slip, method="spencer", n_slices=31)
    assert r.method_solved
    assert r.n_slices == 31
    assert r.FOS == pytest.approx(0.9708, abs=0.001)          # 1.182 before
    f = fosm_fos(geom, {"cu:clay": {"mean": 0.12 * 20.0 * H, "cov": 0.1}},
                 slip_surface=slip, method="spencer", n_slices=31)
    assert f.fos_mlv == pytest.approx(r.FOS, rel=1e-6)


def test_taylor_d2_noncircular_search_near_the_circle():
    """Differential evolution on Taylor's D = 2 profile: 0.680, 3.7 % below
    the 0.706 circle (0.610 before, -13.6 %, as even slices under-read the
    corners). Spencer on a five-point polyline in phi = 0 clay; the 3.7 %
    is the polyline's freedom, not slicing: its surface reads the same on
    30 to 240 slices."""
    geom = _taylor_d2()
    res = search_critical_surface(geom, surface_type="noncircular_de",
                                  method="spencer", seed=1)
    c = res.critical
    assert c.FOS == pytest.approx(0.680, abs=0.005)
    assert c.FOS == pytest.approx(TAYLOR_D2, rel=0.05)
    slip = PolylineSlipSurface(points=list(c.slip_points))
    again = [analyze_slope(geom, slip_surface=slip, method="spencer",
                           n_slices=n).FOS for n in (30, 60, 240)]
    assert again == pytest.approx([c.FOS] * 3, rel=0.002)
