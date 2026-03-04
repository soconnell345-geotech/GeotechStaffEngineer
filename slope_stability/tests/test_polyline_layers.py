"""Tests for non-horizontal (polyline boundary) soil layers."""
import math
import pytest
from slope_stability.geometry import (
    SlopeSoilLayer, SlopeGeometry, _interp_polyline,
)
from slope_stability.slices import build_slices
from slope_stability.slip_surface import CircularSlipSurface


class TestInterpPolyline:
    def test_single_point(self):
        assert _interp_polyline([(5, 10)], 5) == 10

    def test_left_extrapolation(self):
        pts = [(0, 0), (10, 5)]
        assert _interp_polyline(pts, -5) == 0

    def test_right_extrapolation(self):
        pts = [(0, 0), (10, 5)]
        assert _interp_polyline(pts, 15) == 5

    def test_midpoint(self):
        pts = [(0, 0), (10, 10)]
        assert _interp_polyline(pts, 5) == pytest.approx(5.0)

    def test_multi_segment(self):
        pts = [(0, 0), (5, 5), (10, 0)]
        assert _interp_polyline(pts, 2.5) == pytest.approx(2.5)
        assert _interp_polyline(pts, 7.5) == pytest.approx(2.5)

    def test_at_node(self):
        pts = [(0, 0), (5, 5), (10, 0)]
        assert _interp_polyline(pts, 5) == pytest.approx(5.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _interp_polyline([], 0)


class TestSlopeSoilLayerPolyline:
    def test_bottom_at_flat(self):
        layer = SlopeSoilLayer("L", top_elevation=10, bottom_elevation=0, gamma=18)
        assert layer.bottom_at(5) == 0

    def test_bottom_at_polyline(self):
        layer = SlopeSoilLayer(
            "L", top_elevation=10, bottom_elevation=0, gamma=18,
            bottom_boundary_points=[(0, 2), (10, 5)])
        assert layer.bottom_at(5) == pytest.approx(3.5)

    def test_top_at(self):
        layer = SlopeSoilLayer("L", top_elevation=10, bottom_elevation=0, gamma=18)
        assert layer.top_at(5) == 10

    def test_polyline_skips_bottom_ge_top_check(self):
        # With polyline, flat bottom_elevation is approximate — allow bot >= top
        layer = SlopeSoilLayer(
            "L", top_elevation=5, bottom_elevation=5, gamma=18,
            bottom_boundary_points=[(0, 0), (10, 3)])
        assert layer.bottom_at(5) == pytest.approx(1.5)


class TestLayerAtPoint:
    def test_horizontal_layers(self):
        layers = [
            SlopeSoilLayer("A", top_elevation=10, bottom_elevation=5, gamma=18),
            SlopeSoilLayer("B", top_elevation=5, bottom_elevation=0, gamma=18),
        ]
        geom = SlopeGeometry(
            surface_points=[(0, 10), (20, 10)],
            soil_layers=layers)
        assert geom.layer_at_point(10, 7).name == "A"
        assert geom.layer_at_point(10, 3).name == "B"
        assert geom.layer_at_point(10, -1) is None

    def test_sloped_boundary(self):
        layers = [
            SlopeSoilLayer(
                "Sand", top_elevation=10, bottom_elevation=2, gamma=18, phi=30,
                bottom_boundary_points=[(0, 5), (20, 2)]),
            SlopeSoilLayer(
                "Clay", top_elevation=5, bottom_elevation=-5, gamma=17, phi=0, cu=50,
                analysis_mode="undrained"),
        ]
        geom = SlopeGeometry(
            surface_points=[(0, 10), (20, 10)],
            soil_layers=layers)
        # At x=0, boundary is at z=5 — point at z=6 is in Sand
        assert geom.layer_at_point(0, 6).name == "Sand"
        # At x=20, boundary is at z=2 — point at z=3 is in Sand
        assert geom.layer_at_point(20, 3).name == "Sand"
        # At x=20, point at z=1 is in Clay
        assert geom.layer_at_point(20, 1).name == "Clay"


class TestBuildSlicesWithPolyline:
    def test_correct_strength_params(self):
        """Slices should pick up correct c/phi based on x-position."""
        layers = [
            SlopeSoilLayer(
                "Sand", top_elevation=10, bottom_elevation=0, gamma=18,
                phi=35, c_prime=0,
                bottom_boundary_points=[(0, 8), (50, 2)]),
            SlopeSoilLayer(
                "Clay", top_elevation=8, bottom_elevation=-10, gamma=17,
                phi=0, c_prime=0, cu=50, analysis_mode="undrained"),
        ]
        geom = SlopeGeometry(
            surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
            soil_layers=layers)
        slip = CircularSlipSurface(xc=15, yc=20, radius=18)
        slices = build_slices(geom, slip, n_slices=20)
        assert len(slices) > 0
        # Near x=0 (boundary at z=8), deep slices should be in Clay
        # Near x=40 (boundary at z=2.8), shallow slices should be in Sand
