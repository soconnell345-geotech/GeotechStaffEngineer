"""axial_pile calculation tables (backlog GM-39): the values in the tables
are the result's own."""

import pytest

from axial_pile import (AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile,
                        PileSection)
from axial_pile.calc_steps import CAPACITY_TABLE_POINTS, get_tables


@pytest.fixture(scope="module")
def solved():
    pile = PileSection(name="HP 12x53", pile_type="h_pile", area=0.0101,
                       perimeter=1.22, tip_area=0.0101, width=0.31)
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(soil_type="cohesive", thickness=6.0, unit_weight=18.0,
                       cohesion=40.0),
        AxialSoilLayer(soil_type="cohesionless", thickness=16.0,
                       unit_weight=19.0, friction_angle=33.0),
    ], gwt_depth=2.0)
    analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=18.0)
    return analysis.compute(), analysis


def _by_name(tables):
    return {t.name: t for t in tables}


def test_side_resistance_rows_are_the_layer_breakdown(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["side_resistance"]
    lb = result.layer_breakdown
    assert t.column("Side resistance Q_s") == [x["skin_friction_kN"]
                                                for x in lb]
    assert t.column("σ'v at mid-depth") == [x["sigma_v_kPa"] for x in lb]
    assert t.column("Cumulative Q_s")[-1] == pytest.approx(result.Q_skin)
    assert t.column("Method") == ["Tomlinson α", "Nordlund"]


def test_unit_side_resistance_is_qs_over_shaft_area(solved):
    result, analysis = solved
    t = _by_name(result.get_tables())["side_resistance"]
    for row, lb in zip(t.rows, result.layer_breakdown):
        area = analysis.pile.perimeter * (lb["depth_bottom_m"]
                                          - lb["depth_top_m"])
        assert row[t.column_index("Unit side resistance f_s")] == \
            pytest.approx(lb["skin_friction_kN"] / area)


def test_load_transfer_runs_from_q_ult_to_q_tip(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["load_transfer"]
    q = t.column("Axial load in pile Q")
    assert q[0] == result.Q_ultimate
    assert q[-1] == pytest.approx(result.Q_tip)
    assert t.column("Depth z")[-1] == result.pile_length


def test_capacity_summary_is_csv_only(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["capacity_summary"]
    assert not t.printed
    values = dict(zip(t.column("Symbol"), t.column("Value")))
    assert values["Q_ult"] == result.Q_ultimate
    assert values["Q_all"] == result.Q_allowable


def test_capacity_against_length_matches_the_analysis(solved):
    result, analysis = solved
    t = _by_name(get_tables(result, analysis))["capacity_vs_length"]
    assert t.n_rows == CAPACITY_TABLE_POINTS
    expected = analysis.capacity_vs_depth(
        depth_min=3.0, depth_max=analysis.soil.total_thickness,
        n_points=CAPACITY_TABLE_POINTS)
    assert t.column("Ultimate capacity Q_ult") == [r["Q_ultimate_kN"]
                                                   for r in expected]
    # the row at the design length is the design result
    i = t.column("Pile length L").index(18.0) if 18.0 in t.column(
        "Pile length L") else None
    if i is not None:
        assert t.rows[i][3] == pytest.approx(result.Q_ultimate, abs=0.05)


def test_layer_echo(solved):
    result, analysis = solved
    t = _by_name(get_tables(result, analysis))["soil_layers"]
    assert not t.printed
    assert t.column("Bottom") == [6.0, 22.0]
