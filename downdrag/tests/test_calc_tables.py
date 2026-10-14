"""downdrag calculation tables (backlog GM-39): the values in the tables
are the result's own."""

import pytest

from downdrag import DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile
from downdrag.calc_steps import get_tables


@pytest.fixture(scope="module")
def solved():
    soil = DowndragSoilProfile(layers=[
        DowndragSoilLayer(soil_type="cohesive", thickness=10.0,
                          unit_weight=17.0, cu=30.0, settling=True, Cc=0.4,
                          e0=1.1),
        DowndragSoilLayer(soil_type="cohesionless", thickness=15.0,
                          unit_weight=19.0, phi=34.0),
    ], gwt_depth=1.0)
    analysis = DowndragAnalysis(pile_length=20.0, pile_diameter=0.4,
                                pile_E=30e6, Q_dead=600.0, soil=soil,
                                fill_thickness=2.0)
    return analysis.compute(), analysis


def _by_name(tables):
    return {t.name: t for t in tables}


def test_profile_table_is_the_result_arrays(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["load_and_settlement"]
    assert t.printed and t.n_rows == len(result.z)
    assert t.column("Axial load in pile Q") == [float(q) for q in
                                                result.axial_load]
    assert t.column("Unit skin friction f_s") == [
        float(f) for f in result.unit_skin_friction]
    assert t.column("Soil settlement") == [
        float(s) for s in result.soil_settlement_profile]
    assert t.column("Axial load in pile Q")[0] == pytest.approx(
        result.Q_dead)


def test_method_comparison_is_the_results_rows(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["neutral_plane_methods"]
    assert not t.printed
    rows = result.neutral_plane_comparison
    assert t.column("Method") == [r["name"] for r in rows]
    used = [r for r in t.rows if r[t.column_index("Headline")]]
    assert len(used) == 1
    assert used[0][t.column_index("Neutral plane depth z_np")] == \
        pytest.approx(result.neutral_plane_depth)


def test_summary_and_layers_are_csv_only(solved):
    result, analysis = solved
    tables = _by_name(get_tables(result, analysis))
    s = tables["downdrag_summary"]
    assert not s.printed
    v = dict(zip(s.column("Symbol"), s.column("Value")))
    assert v["Q_np"] == result.max_pile_load
    assert v["Q_np - Q_dead"] == result.drag_force
    assert tables["soil_layers"].column("Bottom") == [10.0, 25.0]
