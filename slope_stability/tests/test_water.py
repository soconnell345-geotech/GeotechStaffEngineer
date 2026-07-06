"""
P7 tests: ponded water (GWT above ground surface).

Auto-detected treatment: the hydrostatic pressure on the submerged
ground surface resolves into a vertical water-column weight per slice
plus a signed horizontal thrust on inclined ground (the external-water
buttress), and the base pore pressure uses the full head to the pool.

Classic check: a fully submerged slope under a horizontal pool analyzed
with total unit weights + full-head pore pressures + pond loads must
match the same slope analyzed dry with buoyant unit weight
(gamma' = gamma_sat - gamma_w) — the textbook equivalence. This holds
to ~0.1% for Bishop / Spencer / M-P. The Ordinary Method of Slices is
the documented exception: N' = W*cos(a) - u*l over-subtracts the pore
force on steep bases, so OMS lands far on the low (conservative) side
for submerged slopes (Duncan, Wright & Brandon 2014, Ch. 6).
"""

import pytest

from geotech_common.water import GAMMA_W
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import bishop_fos, fellenius_fos
from slope_stability.gle import gle_fos
from slope_stability.analysis import analyze_slope, rapid_drawdown_fos


SURFACE = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
CIRCLE = dict(xc=30.0, yc=32.0, radius=26.0)


class TestPondedWater:

    def test_pond_weight_added_to_slices(self):
        """GWT 3 m above the toe bench: slices there carry the water
        column weight."""
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=18.0, gamma_sat=20.0, phi=28.0, c_prime=5.0,
        )
        z_pool = 13.0
        geom = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, z_pool), (70.0, z_pool)],
        )
        slip = CircularSlipSurface(**CIRCLE)
        slices = build_slices(geom, slip, 40)
        # pick a slice on the toe bench (ground at 10, pool at 13)
        bench = [s for s in slices if s.z_top < 10.5]
        assert bench
        s = bench[0]
        # soil part: gamma_sat below GWT
        soil_w = 20.0 * (s.z_top - s.z_base) * s.width
        pond_w = GAMMA_W * (z_pool - s.z_top) * s.width
        assert s.weight == pytest.approx(soil_w + pond_w, rel=1e-6)
        assert s.pond_force == pytest.approx(pond_w, rel=1e-6)
        # flat bench: no horizontal thrust on this slice
        assert s.pond_hforce == pytest.approx(0.0, abs=1e-9)
        # seismic would act on soil only (kh=0 here -> just check 0)
        assert s.seismic_force == 0.0

    def test_pond_thrust_on_submerged_face(self):
        """Inclined submerged face carries a horizontal thrust pushing
        into the slope (here: surface ascends to the right -> +x)."""
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=18.0, gamma_sat=20.0, phi=28.0, c_prime=5.0,
        )
        z_pool = 14.0
        geom = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, z_pool), (70.0, z_pool)],
        )
        slip = CircularSlipSurface(**CIRCLE)
        slices = build_slices(geom, slip, 40)
        # submerged inclined face: x in (20, 28) where ground < pool
        face = [s for s in slices if 20.5 < s.x_mid < 27.5]
        assert face
        for s in face:
            assert s.pond_hforce > 0.0
            # line of action: between the segment's lower edge and pool
            z_lo = geom.ground_elevation_at(s.x_left)
            assert z_lo <= s.pond_z <= z_pool
        # total thrust equals the hydrostatic resultant on the projected
        # vertical face: 0.5 * gamma_w * h^2 (h = pool depth at the toe
        # of the inclined face), shared across slices
        h = z_pool - 10.0
        total = sum(s.pond_hforce for s in slices)
        assert total == pytest.approx(0.5 * GAMMA_W * h * h, rel=1e-6)

    def test_submerged_equals_buoyant(self):
        """Fully submerged slope == dry slope with buoyant unit weight."""
        gamma_sat = 20.0
        sub = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=gamma_sat, gamma_sat=gamma_sat, phi=30.0, c_prime=4.0,
        )
        geom_sub = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[sub],
            gwt_points=[(0.0, 22.0), (70.0, 22.0)],  # pool above crest
        )
        buoy = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=gamma_sat - GAMMA_W, phi=30.0, c_prime=4.0,
        )
        geom_buoy = SlopeGeometry(surface_points=SURFACE, soil_layers=[buoy])

        slip = CircularSlipSurface(**CIRCLE)
        s_sub = build_slices(geom_sub, slip, 40)
        s_buoy = build_slices(geom_buoy, slip, 40)

        f_sub = bishop_fos(s_sub, slip)
        f_buoy = bishop_fos(s_buoy, slip)
        assert f_sub == pytest.approx(f_buoy, rel=0.01)

        # rigorous Spencer / M-P satisfy full equilibrium -> equivalence
        # is essentially exact
        for f_name in ("constant", "half_sine"):
            r_sub = gle_fos(s_sub, slip, f_name)
            r_buoy = gle_fos(s_buoy, slip, f_name)
            assert r_sub.converged and r_buoy.converged
            assert r_sub.fos == pytest.approx(r_buoy.fos, rel=0.005)

        # OMS: N' = W*cos(a) - u*l over-subtracts the pore force on
        # steep bases with full-head u — the known OMS submerged-slope
        # inaccuracy (Duncan, Wright & Brandon 2014, Ch. 6). Assert it
        # stays on the conservative side and within the documented band.
        f_sub_f = fellenius_fos(s_sub, slip)
        f_buoy_f = fellenius_fos(s_buoy, slip)
        assert f_sub_f < f_buoy_f
        assert f_sub_f > 0.6 * f_buoy_f

    def test_pond_buttress_raises_fos(self):
        """At the SAME internal GWT, accounting for the pond loads
        (weight + horizontal thrust) raises FOS vs ignoring the pond
        (the pre-P7 behavior: full-head u with no external water loads
        was doubly conservative)."""
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=18.0, gamma_sat=20.0, phi=25.0, c_prime=5.0,
        )
        geom_pond = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, 14.0), (70.0, 14.0)],
        )
        slip = CircularSlipSurface(**CIRCLE)
        slices = build_slices(geom_pond, slip, 40)
        fos_with = bishop_fos(slices, slip)
        # strip the pond loads (legacy behavior), same pore pressures
        for s in slices:
            s.weight -= s.pond_force
            s.pond_force = 0.0
            s.pond_hforce = 0.0
        fos_without = bishop_fos(slices, slip)
        assert fos_with > 1.2 * fos_without

    def test_partial_pond_analyze_slope(self):
        """analyze_slope auto-detects the pond (no special flags) and
        the result stays physical: raising the pool from the bench to
        4 m above it adds pore pressure under the whole mass but also
        the buttress, so FOS drops by only a modest amount."""
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=18.0, gamma_sat=20.0, phi=25.0, c_prime=5.0,
        )
        geom_bench = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, 10.0), (40.0, 10.0), (70.0, 10.0)],
        )
        geom_pond = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, 14.0), (70.0, 14.0)],
        )
        res_bench = analyze_slope(geom_bench, method="bishop", n_slices=40,
                                  **CIRCLE)
        res_pond = analyze_slope(geom_pond, method="bishop", n_slices=40,
                                 **CIRCLE)
        assert res_pond.FOS < res_bench.FOS          # more u overall
        assert res_pond.FOS > 0.9 * res_bench.FOS    # buttress limits it

    def test_gle_submerged_converges(self):
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=20.0, gamma_sat=20.0, phi=30.0, c_prime=4.0,
        )
        geom = SlopeGeometry(
            surface_points=SURFACE, soil_layers=[layer],
            gwt_points=[(0.0, 22.0), (70.0, 22.0)],
        )
        res = analyze_slope(geom, method="gle", n_slices=40, **CIRCLE)
        assert res.FOS > 0.5


class TestRapidDrawdown:
    """The rapid-drawdown method is now implemented (v5.3 B2a); the old
    NotImplementedError stub is closed. Full theory + validation in
    slope_stability/rapid_drawdown.py and tests/test_rapid_drawdown.py."""

    def _geom(self):
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=19.0, phi=25.0, c_prime=5.0, R_c=30.0, R_phi=12.0,
        )
        return SlopeGeometry(surface_points=SURFACE, soil_layers=[layer])

    def test_returns_result_not_raises(self):
        res = rapid_drawdown_fos(self._geom(), 18.0, 10.0, method="duncan_3stage",
                                 **CIRCLE)
        assert res.FOS > 0
        assert res.stage1_fos > 0

    def test_requires_slip_surface(self):
        with pytest.raises(ValueError):
            rapid_drawdown_fos(self._geom(), 18.0, 10.0)   # no circle/surface
