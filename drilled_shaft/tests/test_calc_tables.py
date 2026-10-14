"""drilled_shaft calculation tables (backlog GM-39): the values in the
tables are the result's own."""

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import CAPACITY_TABLE_POINTS, get_tables


@pytest.fixture(scope="module")
def solved():
    shaft = DrillShaft(diameter=1.2, length=20.0)
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(thickness=8.0, soil_type="cohesive", unit_weight=18.0,
                       cu=60.0),
        ShaftSoilLayer(thickness=10.0, soil_type="cohesionless",
                       unit_weight=19.5, phi=34.0, N60=25),
        ShaftSoilLayer(thickness=6.0, soil_type="rock", unit_weight=22.0,
                       qu=5000.0, RQD=80.0),
    ], gwt_depth=3.0)
    analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
    return analysis.compute(), analysis


def _by_name(tables):
    return {t.name: t for t in tables}


def test_side_resistance_rows_are_the_layer_breakdown(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["side_resistance"]
    lb = result.layer_breakdown
    assert t.column("Side resistance Q_s") == [x["side_resistance_kN"]
                                                for x in lb]
    assert t.column("Unit side resistance f_s") == [x["fs_kPa"] for x in lb]
    assert t.column("Effective top") == [x["effective_top_m"] for x in lb]
    assert t.column("Cumulative Q_s")[-1] == pytest.approx(result.Q_skin,
                                                          abs=0.05 * len(lb))


def test_load_transfer_ends_at_the_base_resistance(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["load_transfer"]
    q = t.column("Axial load in shaft Q")
    assert q[0] == result.Q_ultimate
    assert q[-1] == pytest.approx(result.Q_tip, abs=0.05 * (t.n_rows - 1))


def test_summary_splits_side_resistance_by_soil_type(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["capacity_summary"]
    assert not t.printed
    v = dict(zip(t.column("Symbol"), t.column("Value")))
    assert v["Q_s,clay"] + v["Q_s,sand"] + v["Q_s,rock"] == \
        pytest.approx(result.Q_skin)
    assert v["Q_b"] == result.Q_tip and v["Q_ult"] == result.Q_ultimate


def test_capacity_against_length_matches_the_analysis(solved):
    result, analysis = solved
    t = _by_name(get_tables(result, analysis))["capacity_vs_length"]
    expected = analysis.capacity_vs_depth(
        depth_min=3.0, depth_max=analysis.soil.total_thickness,
        n_points=CAPACITY_TABLE_POINTS)
    assert t.column("Ultimate capacity Q_ult") == [r["Q_ultimate_kN"]
                                                   for r in expected]


def test_layer_echo(solved):
    result, analysis = solved
    t = _by_name(get_tables(result, analysis))["soil_layers"]
    assert not t.printed
    assert t.column("Soil type") == ["cohesive", "cohesionless", "rock"]
    assert t.column("Bottom") == [8.0, 18.0, 24.0]
