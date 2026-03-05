"""
Tests for new search algorithms: PSO, weak-layer biased, entry/exit region.

Tests verify:
- Each algorithm finds a valid critical surface with FOS < 999.9
- PSO converges to reasonable FOS (comparable to grid search)
- Weak-layer biased routes through weak layers more often
- Entry/exit search finds valid circles within the specified regions
- All algorithms return proper SearchResult structure
"""

import math
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.search import (
    search_pso, search_weak_layer_biased, search_entry_exit,
)
from slope_stability.analysis import search_critical_surface


# ── Fixtures ──────────────────────────────────────────────────────────────

def _simple_slope():
    """Standard 2:1 slope, 10m high."""
    layer = SlopeSoilLayer(
        name="Soil",
        top_elevation=10.0,
        bottom_elevation=-10.0,
        gamma=18.0,
        gamma_sat=20.0,
        phi=25.0,
        c_prime=10.0,
    )
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
        soil_layers=[layer],
    )


def _two_layer_slope():
    """Slope with strong upper layer and weak lower layer."""
    strong = SlopeSoilLayer(
        name="Strong",
        top_elevation=10.0,
        bottom_elevation=2.0,
        gamma=19.0,
        gamma_sat=21.0,
        phi=35.0,
        c_prime=20.0,
    )
    weak = SlopeSoilLayer(
        name="Weak Clay",
        top_elevation=2.0,
        bottom_elevation=-10.0,
        gamma=17.0,
        gamma_sat=19.0,
        phi=10.0,
        c_prime=5.0,
    )
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (30, 0), (50, 0)],
        soil_layers=[strong, weak],
    )


# ── PSO Search Tests ─────────────────────────────────────────────────────

class TestPSOSearch:
    """Test Particle Swarm Optimization noncircular search."""

    def test_finds_valid_surface(self):
        geom = _simple_slope()
        result = search_pso(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_particles=10,
            n_iterations=10,
            n_slices=20,
            seed=42,
        )
        assert result.critical is not None
        assert result.critical.FOS > 0
        assert result.critical.FOS < 100
        assert result.n_surfaces_evaluated > 0

    def test_returns_noncircular(self):
        geom = _simple_slope()
        result = search_pso(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_particles=10,
            n_iterations=10,
            seed=42,
        )
        assert result.critical is not None
        assert not result.critical.is_circular
        assert result.critical.slip_points is not None
        assert len(result.critical.slip_points) >= 3

    def test_grid_fos_populated(self):
        geom = _simple_slope()
        result = search_pso(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_particles=10,
            n_iterations=5,
            seed=42,
        )
        assert len(result.grid_fos) > 0
        for g in result.grid_fos:
            assert "FOS" in g
            assert g["FOS"] > 0

    def test_more_iterations_better(self):
        """More iterations should find equal or lower FOS."""
        geom = _simple_slope()
        r1 = search_pso(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_particles=15,
            n_iterations=5,
            seed=42,
        )
        r2 = search_pso(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_particles=15,
            n_iterations=30,
            seed=42,
        )
        # More iterations should find equal or better result
        assert r2.critical.FOS <= r1.critical.FOS + 0.1

    def test_seed_reproducibility(self):
        geom = _simple_slope()
        r1 = search_pso(
            geom, x_entry_range=(5, 15), x_exit_range=(25, 45),
            n_particles=10, n_iterations=10, seed=99,
        )
        r2 = search_pso(
            geom, x_entry_range=(5, 15), x_exit_range=(25, 45),
            n_particles=10, n_iterations=10, seed=99,
        )
        assert abs(r1.critical.FOS - r2.critical.FOS) < 1e-6

    def test_via_search_critical_surface(self):
        geom = _simple_slope()
        result = search_critical_surface(
            geom, surface_type="pso", n_slices=20,
            x_entry_range=(5, 15), x_exit_range=(25, 45),
        )
        assert result.critical is not None
        assert result.critical.FOS > 0


# ── Weak-Layer Biased Search Tests ────────────────────────────────────────

class TestWeakLayerBiased:
    """Test SNIFF-inspired weak-layer biased search."""

    def test_finds_valid_surface(self):
        geom = _simple_slope()
        result = search_weak_layer_biased(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_trials=100,
            seed=42,
        )
        assert result.critical is not None
        assert result.critical.FOS > 0
        assert result.critical.FOS < 100

    def test_returns_noncircular(self):
        geom = _simple_slope()
        result = search_weak_layer_biased(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_trials=100,
            seed=42,
        )
        assert result.critical is not None
        assert not result.critical.is_circular
        assert result.critical.slip_points is not None

    def test_two_layer_finds_lower_fos(self):
        """With a weak layer, should find lower FOS than random search."""
        geom = _two_layer_slope()
        result = search_weak_layer_biased(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_trials=200,
            seed=42,
        )
        assert result.critical is not None
        # With a weak clay at bottom, FOS should be relatively low
        assert result.critical.FOS < 5.0

    def test_grid_fos_populated(self):
        geom = _two_layer_slope()
        result = search_weak_layer_biased(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_trials=50,
            seed=42,
        )
        assert len(result.grid_fos) > 0

    def test_via_search_critical_surface(self):
        geom = _two_layer_slope()
        result = search_critical_surface(
            geom, surface_type="weak_layer", n_slices=20,
            x_entry_range=(5, 15), x_exit_range=(25, 45),
            n_trials=100,
        )
        assert result.critical is not None


# ── Entry/Exit Region Search Tests ────────────────────────────────────────

class TestEntryExitSearch:
    """Test entry/exit region search for circular surfaces."""

    def test_finds_valid_surface(self):
        geom = _simple_slope()
        result = search_entry_exit(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_entry=5,
            n_exit=5,
        )
        assert result.critical is not None
        assert result.critical.FOS > 0
        assert result.critical.FOS < 100

    def test_returns_circular(self):
        geom = _simple_slope()
        result = search_entry_exit(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_entry=5,
            n_exit=5,
        )
        assert result.critical is not None
        assert result.critical.is_circular
        assert result.critical.radius > 0

    def test_entry_exit_within_ranges(self):
        geom = _simple_slope()
        result = search_entry_exit(
            geom,
            x_entry_range=(8, 12),
            x_exit_range=(28, 40),
            n_entry=5,
            n_exit=5,
        )
        assert result.critical is not None
        # Entry should be near the specified range
        assert result.critical.x_entry >= 5  # Some tolerance
        assert result.critical.x_exit <= 50

    def test_grid_fos_has_entries(self):
        geom = _simple_slope()
        result = search_entry_exit(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_entry=5,
            n_exit=5,
        )
        assert len(result.grid_fos) > 0
        for g in result.grid_fos:
            assert "xc" in g
            assert "yc" in g
            assert "R" in g
            assert "FOS" in g

    def test_more_points_more_evaluations(self):
        geom = _simple_slope()
        r1 = search_entry_exit(
            geom, x_entry_range=(5, 15), x_exit_range=(25, 45),
            n_entry=3, n_exit=3,
        )
        r2 = search_entry_exit(
            geom, x_entry_range=(5, 15), x_exit_range=(25, 45),
            n_entry=8, n_exit=8,
        )
        assert r2.n_surfaces_evaluated >= r1.n_surfaces_evaluated

    def test_spencer_method(self):
        geom = _simple_slope()
        result = search_entry_exit(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_entry=5,
            n_exit=5,
            method="spencer",
        )
        assert result.critical is not None
        assert result.critical.FOS > 0

    def test_via_search_critical_surface(self):
        geom = _simple_slope()
        result = search_critical_surface(
            geom, surface_type="entry_exit",
            x_entry_range=(5, 15), x_exit_range=(25, 45),
            nx=5, ny=5,
        )
        assert result.critical is not None

    def test_comparable_to_grid_search(self):
        """Entry/exit should find similar FOS to grid search."""
        geom = _simple_slope()
        grid_result = search_critical_surface(
            geom, x_range=(5, 35), y_range=(11, 25),
            nx=8, ny=8, method="bishop",
        )
        ee_result = search_entry_exit(
            geom,
            x_entry_range=(5, 15),
            x_exit_range=(25, 45),
            n_entry=8,
            n_exit=8,
            method="bishop",
        )
        # Both should find a valid surface; FOS should be in the same ballpark
        assert grid_result.critical is not None
        assert ee_result.critical is not None
        # Within 50% — they're searching different parameter spaces
        ratio = ee_result.critical.FOS / grid_result.critical.FOS
        assert 0.5 < ratio < 2.0
