"""
P8 tests: per-slice force table, thrust line, method-comparison helper.
"""

import pytest

from slope_stability import analyze_slope, compare_methods_table
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface


SURFACE = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
CIRCLE = dict(xc=30.0, yc=32.0, radius=26.0)


def _geom():
    layer = SlopeSoilLayer(
        name="soil", top_elevation=20.0, bottom_elevation=-15.0,
        gamma=18.0, gamma_sat=20.0, phi=25.0, c_prime=5.0,
    )
    return SlopeGeometry(
        surface_points=SURFACE, soil_layers=[layer],
        gwt_points=[(0.0, 8.0), (70.0, 8.0)],
    )


class TestSliceForceTable:

    def test_bishop_force_table(self):
        """Non-rigorous method: Fellenius-decomposition forces, no E/X."""
        res = analyze_slope(_geom(), method="bishop", n_slices=30,
                            include_slice_data=True, **CIRCLE)
        assert res.slice_data is not None
        for sd in res.slice_data:
            # N' = W cos a - u l (clamped) and consistency with stresses
            assert sd.N_eff_kN == pytest.approx(
                sd.normal_stress_kPa * sd.base_length, rel=1e-9)
            assert sd.U_base_kN == pytest.approx(
                sd.pore_pressure * sd.base_length, rel=1e-9)
            assert sd.E_left_kN is None
            assert sd.X_left_kN is None
        assert res.thrust_line is None

    def test_gle_force_table_and_thrust(self):
        """Rigorous method: interslice E/X populated, E ~ 0 at the ends,
        thrust line returned at every slice boundary."""
        res = analyze_slope(_geom(), method="gle", n_slices=30,
                            include_slice_data=True, **CIRCLE)
        sds = res.slice_data
        assert sds is not None
        # ends carry no interslice normal force
        assert sds[0].E_left_kN == pytest.approx(0.0, abs=1e-6)
        assert sds[-1].E_right_kN == pytest.approx(0.0, abs=1e-6)
        # interior E is compressive (positive) and continuous between
        # neighbouring slices
        mid = len(sds) // 2
        assert sds[mid].E_left_kN > 0.0
        for a, b in zip(sds[:-1], sds[1:]):
            assert a.E_right_kN == pytest.approx(b.E_left_kN, abs=1e-9)
            assert a.X_right_kN == pytest.approx(b.X_left_kN, abs=1e-9)
        # mobilized shear = available resistance / FOS on every base
        for sd in sds:
            assert sd.S_mob_kN * res.FOS == pytest.approx(
                sd.shear_resistance_kPa * sd.base_length, rel=1e-6)
        # thrust line: one point per boundary, inside the slope body for
        # interior boundaries (between base and ground, with margin for
        # the near-zero-E end boundaries)
        assert res.thrust_line is not None
        assert len(res.thrust_line) == len(sds) + 1
        for (x, z), sd in zip(res.thrust_line[1:-1], sds[1:]):
            assert sd.z_base - 1.0 <= z <= sd.z_top + 1.0

    def test_thrust_line_clamped_to_section_at_ends(self):
        """Near the slip exit/entry the interslice normal E -> 0, so the raw
        point of application z = z_base - m/E (gle.py) can spike out of the
        section; the reported thrust line is clamped to the physical section
        (slip surface <= z <= ground) at EVERY boundary — including the
        previously-spiking end boundaries the older check had to skip."""
        geom = _geom()
        res = analyze_slope(geom, method="gle", n_slices=30,
                            include_slice_data=True, **CIRCLE)
        slip = CircularSlipSurface(**CIRCLE)
        assert res.thrust_line is not None
        for x, z in res.thrust_line:
            z_slip = slip.slip_elevation_at(x)
            z_ground = geom.ground_elevation_at(x)
            if z_slip is None or z_ground is None:
                continue
            lo, hi = min(z_slip, z_ground), max(z_slip, z_ground)
            assert lo - 1e-6 <= z <= hi + 1e-6

    def test_to_dict_force_table(self):
        res = analyze_slope(_geom(), method="spencer", n_slices=20,
                            include_slice_data=True, **CIRCLE)
        d = res.to_dict()
        assert "thrust_line" in d
        row = d["slice_data"][5]
        for key in ("weight_kN_per_m", "N_eff_kN_per_m", "S_mob_kN_per_m",
                    "U_base_kN_per_m", "E_left_kN_per_m", "X_left_kN_per_m",
                    "alpha_deg"):
            assert key in row


class TestCompareMethodsTable:

    def test_all_methods_present(self):
        tab = compare_methods_table(_geom(), n_slices=30, **CIRCLE)
        names = [r["method"] for r in tab["rows"]]
        assert "Fellenius (OMS)" in names
        assert "Bishop simplified" in names
        assert "Janbu simplified (corrected)" in names
        assert "Spencer" in names
        assert any(n.startswith("Morgenstern-Price") for n in names)
        assert tab["surface"]["type"] == "circular"
        assert "Method comparison" in tab["summary"]

    def test_values_match_analyze_slope(self):
        tab = compare_methods_table(_geom(), n_slices=30, **CIRCLE)
        by_name = {r["method"]: r["FOS"] for r in tab["rows"]}
        res_b = analyze_slope(_geom(), method="bishop", n_slices=30, **CIRCLE)
        assert by_name["Bishop simplified"] == pytest.approx(res_b.FOS,
                                                             rel=1e-3)
        # method ordering: OMS lowest, rigorous methods close to Bishop
        assert by_name["Fellenius (OMS)"] < by_name["Bishop simplified"]
        assert by_name["Spencer"] == pytest.approx(
            by_name["Bishop simplified"], rel=0.05)

    def test_noncircular_surface(self):
        from slope_stability.slip_surface import PolylineSlipSurface
        pts = [(16.0, 10.0), (24.0, 6.5), (34.0, 6.0), (44.0, 11.0),
               (52.0, 20.0)]
        slip = PolylineSlipSurface(points=pts)
        tab = compare_methods_table(_geom(), slip_surface=slip, n_slices=30)
        names = [r["method"] for r in tab["rows"]]
        assert "Bishop simplified" not in names      # circular only
        assert "Spencer" in names
        assert tab["surface"]["type"] == "noncircular"
