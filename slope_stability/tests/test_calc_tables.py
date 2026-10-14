"""slope_stability calculation tables (backlog GM-39): the values in the
tables are the result's and the search's own."""

import pytest

from slope_stability.analysis import analyze_slope, search_critical_surface
from slope_stability.calc_steps import get_tables
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer


def _geom():
    return SlopeGeometry(
        surface_points=[(0, 10), (10, 10), (20, 0), (40, 0)],
        soil_layers=[SlopeSoilLayer(name="clay", top_elevation=10,
                                    bottom_elevation=-10, gamma=19.0,
                                    phi=25.0, c_prime=10.0)],
    )


#: A circle that enters the crest and leaves beyond the toe.
CIRCLE = dict(xc=20.0, yc=15.0, radius=15.5)


@pytest.fixture(scope="module")
def spencer():
    return analyze_slope(_geom(), method="spencer", include_slice_data=True,
                         compare_methods=True, **CIRCLE)


def _by_name(tables):
    return {t.name: t for t in tables}


def test_stresses_are_the_slice_data(spencer):
    t = _by_name(spencer.get_tables())["slip_surface_stresses"]
    assert t.printed and t.n_rows == len(spencer.slice_data)
    assert t.column("Effective normal stress σ'n") == [
        s.normal_stress_kPa for s in spencer.slice_data]
    assert t.column("Shear strength τ_f") == [
        s.shear_resistance_kPa for s in spencer.slice_data]
    lengths = [s.base_length for s in spencer.slice_data]
    assert t.column("Distance along the surface")[-1] == pytest.approx(
        sum(lengths) - lengths[-1] / 2.0)


def test_slice_forces_carry_every_slice_quantity(spencer):
    t = _by_name(spencer.get_tables())["slice_forces"]
    assert not t.printed
    s = spencer.slice_data[3]
    row = t.rows[3]
    assert row[t.column_index("Weight W")] == s.weight
    assert row[t.column_index("Effective normal force N'")] == s.N_eff_kN
    assert row[t.column_index("Interslice normal force E right")] == \
        s.E_right_kN
    assert row[t.column_index("Interslice shear force X left")] == \
        s.X_left_kN


def test_fos_by_method_is_the_results_own(spencer):
    t = _by_name(spencer.get_tables())["fos_by_method"]
    fos = dict(zip(t.column("Method"), t.column("FOS")))
    assert fos["Spencer"] == spencer.FOS_spencer
    assert fos["Bishop simplified"] == spencer.FOS_bishop
    assert fos["Ordinary method of slices (Fellenius)"] == \
        spencer.FOS_fellenius


def test_circle_and_thrust_line(spencer):
    tables = _by_name(spencer.get_tables())
    assert tables["slip_surface"].rows == [[
        spencer.xc, spencer.yc, spencer.radius, spencer.x_entry,
        spencer.x_exit]]
    assert tables["thrust_line"].n_rows == len(spencer.thrust_line)


def test_fellenius_has_no_interslice_columns():
    r = analyze_slope(_geom(), method="fellenius", include_slice_data=True,
                      **CIRCLE)
    t = _by_name(r.get_tables())["slice_forces"]
    assert "Interslice normal force E left" not in [c.name for c in
                                                    t.columns]


def test_search_record_is_the_grid():
    search = search_critical_surface(_geom(), method="bishop", nx=4, ny=4)
    crit = search.critical
    result = analyze_slope(_geom(), xc=crit.xc, yc=crit.yc,
                           radius=crit.radius, method="bishop",
                           include_slice_data=True)
    tables = _by_name(get_tables(result, {"search": search}))
    t = tables["trial_surfaces"]
    assert not t.printed
    assert t.column("FOS") == [g["FOS"] for g in search.grid_fos]
    fam = tables["surface_families"]
    governs = [r for r in fam.rows if r[2]]
    assert len(governs) == 1
    assert governs[0][1] == pytest.approx(crit.FOS, abs=1e-3)
    assert "slip_surface_stresses" in tables
