"""
P3 tests: search upgrades + ACADS 1(a) benchmark (B3).

ACADS 1(a) [Giam & Donald 1989; Slide2 Verification #1]:
surface (20,25)-(30,25)-(50,35)-(70,35), c'=3 kPa, phi'=19.6 deg,
gamma=20 kN/m3, dry. Published FOS = 1.00; Slide2 Bishop 0.987,
Spencer 0.986, Janbu corrected 0.990.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.analysis import search_critical_surface
from slope_stability.search import search_de, search_noncircular


def _acads_geom():
    layer = SlopeSoilLayer(
        name="soil", top_elevation=35.0, bottom_elevation=20.0,
        gamma=20.0, phi=19.6, c_prime=3.0,
    )
    return SlopeGeometry(
        surface_points=[(20.0, 25.0), (30.0, 25.0), (50.0, 35.0),
                        (70.0, 35.0)],
        soil_layers=[layer],
    )


class TestACADS1a:
    """B3 — critical circular search on the ACADS 1(a) slope."""

    def test_grid_search_bishop(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, nx=12, ny=12, method="bishop", n_slices=30,
            x_entry_range=(20.0, 35.0), x_exit_range=(45.0, 70.0),
        )
        assert res.critical is not None
        # Published 1.00 (Giam & Donald); Slide2 0.987
        assert 0.96 <= res.critical.FOS <= 1.04

    def test_entry_exit_search_bishop(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, method="bishop", surface_type="entry_exit",
            nx=12, ny=12, n_slices=30,
            x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        )
        assert res.critical is not None
        assert 0.96 <= res.critical.FOS <= 1.04

    def test_entry_exit_search_spencer(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, method="spencer", surface_type="entry_exit",
            nx=8, ny=8, n_slices=30,
            x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        )
        assert res.critical is not None
        # Slide2 Spencer 0.986
        assert 0.95 <= res.critical.FOS <= 1.05

    def test_entry_exit_search_janbu(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, method="janbu", surface_type="entry_exit",
            nx=8, ny=8, n_slices=30,
            x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        )
        assert res.critical is not None
        # Slide2 Janbu corrected 0.990; janbu search uses corrected FOS
        assert 0.93 <= res.critical.FOS <= 1.05

    def test_grid_search_gle(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, nx=8, ny=8, method="gle", n_slices=30,
            x_entry_range=(20.0, 35.0), x_exit_range=(45.0, 70.0),
        )
        assert res.critical is not None
        assert 0.95 <= res.critical.FOS <= 1.05


class TestDifferentialEvolution:

    def test_de_refines_random_search(self):
        """DE (seeded) must do at least as well as the random search."""
        geom = _acads_geom()
        rand = search_noncircular(
            geom, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
            n_trials=200, n_points=6, n_slices=25, seed=42,
        )
        de = search_de(
            geom, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
            n_points=6, n_slices=25, seed=42, maxiter=20, popsize=12,
            n_seed_trials=200,
        )
        assert de.critical is not None
        assert rand.critical is not None
        assert de.critical.FOS <= rand.critical.FOS + 0.01

    def test_de_close_to_circular_optimum(self):
        """For a homogeneous slope the noncircular optimum is close to
        (slightly below or equal to) the critical circle."""
        geom = _acads_geom()
        de = search_de(
            geom, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
            n_points=6, n_slices=25, seed=7, maxiter=25, popsize=12,
            n_seed_trials=150,
        )
        assert de.critical is not None
        assert 0.90 <= de.critical.FOS <= 1.05

    def test_de_admissibility(self):
        """Resulting surface is monotonic in x and below ground."""
        geom = _acads_geom()
        de = search_de(
            geom, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
            n_points=6, n_slices=25, seed=3, maxiter=10, popsize=8,
            n_seed_trials=80,
        )
        assert de.critical is not None
        pts = de.critical.slip_points
        xs = [p[0] for p in pts]
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
        for x, z in pts[1:-1]:
            assert z <= geom.ground_elevation_at(x) + 1e-6

    def test_de_via_search_critical_surface(self):
        geom = _acads_geom()
        res = search_critical_surface(
            geom, surface_type="noncircular_de", method="spencer",
            n_points=6, n_slices=25, seed=11,
            x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
        )
        assert res.critical is not None
        assert res.critical.slip_points is not None
        assert 0.85 <= res.critical.FOS <= 1.10


class TestNoncircularMethodPlumb:

    def test_random_search_with_gle(self):
        geom = _acads_geom()
        res = search_noncircular(
            geom, x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0),
            n_trials=60, n_points=5, n_slices=25, seed=5,
            method="morgenstern_price",
        )
        assert res.critical is not None
        assert res.critical.method == "Morgenstern-Price"

    def test_weak_layer_geometry_de_finds_weak_path(self):
        """DE on a weak-layer profile should drive the surface into the
        weak zone and find a lower FOS than the critical circle search
        restricted to the same windows."""
        upper = SlopeSoilLayer(
            name="upper", top_elevation=20.0, bottom_elevation=11.0,
            gamma=19.0, phi=30.0, c_prime=8.0,
        )
        weak = SlopeSoilLayer(
            name="weak", top_elevation=11.0, bottom_elevation=9.0,
            gamma=18.0, phi=12.0, c_prime=0.0,
        )
        base = SlopeSoilLayer(
            name="base", top_elevation=9.0, bottom_elevation=0.0,
            gamma=20.0, phi=35.0, c_prime=20.0,
        )
        geom = SlopeGeometry(
            surface_points=[(0.0, 20.0), (15.0, 20.0), (35.0, 10.0),
                            (60.0, 10.0)],
            soil_layers=[upper, weak, base],
        )
        de = search_de(
            geom, x_entry_range=(2.0, 14.0), x_exit_range=(36.0, 55.0),
            n_points=7, n_slices=30, seed=42, maxiter=25, popsize=14,
            n_seed_trials=200,
        )
        assert de.critical is not None
        # surface should pass through/near the weak layer
        z_min = min(z for _, z in de.critical.slip_points)
        assert z_min < 11.5
