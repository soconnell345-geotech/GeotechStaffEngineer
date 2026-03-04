"""Tests for layer_polylines parameter in FEM analysis functions."""

import numpy as np
import pytest

from fem2d.analysis import analyze_slope_srm, analyze_excavation
from fem2d.mesh import generate_rect_mesh, assign_layers_by_polylines


class TestLayerPolylinesParam:
    """Verify layer_polylines parameter works in analysis functions."""

    def test_srm_accepts_layer_polylines(self):
        """analyze_slope_srm should accept and use layer_polylines."""
        surface = [(0, 5), (5, 5), (15, 0), (20, 0)]
        layers = [
            {"name": "Sand", "bottom_elevation": 2, "E": 30000,
             "nu": 0.3, "gamma": 18, "c": 5, "phi": 30, "psi": 0},
            {"name": "Clay", "bottom_elevation": -10, "E": 20000,
             "nu": 0.35, "gamma": 17, "c": 20, "phi": 15, "psi": 0},
        ]
        # Sloped boundary from z=4 at x=-5 to z=1 at x=30
        polyline = np.array([[-5, 4], [30, 1]], dtype=float)

        result = analyze_slope_srm(
            surface, layers, nx=8, ny=6,
            layer_polylines=[polyline])
        assert result.FOS > 0
        assert result.n_nodes > 0

    def test_srm_without_polylines_unchanged(self):
        """analyze_slope_srm without polylines should still work."""
        surface = [(0, 5), (5, 5), (15, 0), (20, 0)]
        layers = [
            {"name": "Sand", "bottom_elevation": 2, "E": 30000,
             "nu": 0.3, "gamma": 18, "c": 5, "phi": 30, "psi": 0},
            {"name": "Clay", "bottom_elevation": -10, "E": 20000,
             "nu": 0.35, "gamma": 17, "c": 20, "phi": 15, "psi": 0},
        ]
        result = analyze_slope_srm(surface, layers, nx=8, ny=6)
        assert result.FOS > 0

    def test_excavation_accepts_layer_polylines(self):
        """analyze_excavation should accept and use layer_polylines."""
        layers = [
            {"name": "Fill", "bottom_elevation": -3, "E": 20000,
             "nu": 0.3, "gamma": 18, "c": 5, "phi": 25, "psi": 0},
            {"name": "Clay", "bottom_elevation": -20, "E": 30000,
             "nu": 0.35, "gamma": 19, "c": 30, "phi": 20, "psi": 0},
        ]
        polyline = np.array([[-30, -3], [40, -4]], dtype=float)

        result = analyze_excavation(
            width=8, depth=4, wall_depth=8,
            soil_layers=layers, wall_EI=50000, wall_EA=5e6,
            nx=8, ny=8,
            layer_polylines=[polyline])
        assert result.n_nodes > 0

    def test_assign_layers_by_polylines_basic(self):
        """Verify assign_layers_by_polylines gives different results
        than assign_layers_by_elevation for a sloped boundary."""
        from fem2d.mesh import assign_layers_by_elevation
        nodes, elements = generate_rect_mesh(0, 20, -10, 0, 10, 5)

        # Horizontal boundary at -3
        layer_ids_flat = assign_layers_by_elevation(
            nodes, elements, [-3])

        # Sloped boundary from z=-1 (left) to z=-5 (right)
        polyline = np.array([[0, -1], [20, -5]], dtype=float)
        layer_ids_sloped = assign_layers_by_polylines(
            nodes, elements, [polyline])

        # Results should differ — the sloped boundary assigns
        # differently than the flat one
        assert not np.array_equal(layer_ids_flat, layer_ids_sloped)

    def test_srm_polylines_vs_flat_different_fos(self):
        """SRM with sloped boundary should give different FOS than flat."""
        surface = [(0, 5), (5, 5), (15, 0), (20, 0)]
        layers = [
            {"name": "Sand", "bottom_elevation": 0, "E": 30000,
             "nu": 0.3, "gamma": 18, "c": 2, "phi": 35, "psi": 0},
            {"name": "Weak", "bottom_elevation": -15, "E": 15000,
             "nu": 0.35, "gamma": 17, "c": 10, "phi": 10, "psi": 0},
        ]
        fos_flat = analyze_slope_srm(
            surface, layers, nx=8, ny=6).FOS

        # Sloped: weak layer is very shallow on left, deep on right
        polyline = np.array([[-10, 3], [30, -2]], dtype=float)
        fos_sloped = analyze_slope_srm(
            surface, layers, nx=8, ny=6,
            layer_polylines=[polyline]).FOS

        # They should differ (not necessarily which is bigger)
        assert fos_flat != pytest.approx(fos_sloped, abs=0.01)
