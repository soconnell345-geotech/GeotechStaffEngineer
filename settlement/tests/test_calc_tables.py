"""settlement calculation tables (backlog GM-39): the values in the tables
are the result's own."""

import pytest

from settlement import ConsolidationLayer, SettlementAnalysis
from settlement.calc_steps import get_tables


@pytest.fixture(scope="module")
def solved():
    analysis = SettlementAnalysis(
        q_applied=150.0, B=3.0, L=3.0, footing_shape="square",
        stress_method="boussinesq", Es_immediate=20000.0,
        consolidation_layers=[
            ConsolidationLayer(thickness=2.0, depth_to_center=3.0, e0=0.9,
                               Cc=0.35, Cr=0.05, sigma_v0=45.0),
            ConsolidationLayer(thickness=3.0, depth_to_center=5.5, e0=0.8,
                               Cc=0.30, Cr=0.04, sigma_v0=70.0,
                               sigma_p=100.0, description="OC clay"),
        ],
        cv=2.0, drainage="double")
    return analysis.compute(), analysis


def _by_name(tables):
    return {t.name: t for t in tables}


def test_per_layer_rows_are_the_results_own(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["consolidation_per_layer"]
    layers = result.consolidation_layers
    assert t.column("Settlement S_c") == [x["settlement_mm"] for x in layers]
    assert t.column("Stress increase Δσ") == [
        x["delta_sigma_kPa"] for x in layers]
    assert sum(t.column("Settlement S_c")) == pytest.approx(
        result.consolidation * 1000, abs=0.01)


def test_time_curve_is_the_results_curve(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["time_settlement"]
    assert t.n_rows == len(result.time_settlement_curve)
    assert t.column("Settlement S") == [s for _, s in
                                        result.time_settlement_curve]
    assert t.column("Time t") == [x for x, _ in result.time_settlement_curve]
    # U as to_dict reports it
    d = result.to_dict()["time_settlement_curve"]
    for u, p in zip(t.column("Average degree of consolidation U"), d):
        assert round(u, 1) == p["U_percent"]


def test_summary_is_csv_only_and_sums(solved):
    result, _ = solved
    t = _by_name(result.get_tables())["settlement_summary"]
    assert not t.printed
    v = dict(zip(t.column("Symbol"), t.column("Value")))
    assert v["S"] == result.total
    assert v["t90"] == result.t90_years


def test_calc_package_table_adds_the_layer_inputs(solved):
    result, analysis = solved
    t = _by_name(get_tables(result, analysis))["consolidation_per_layer"]
    assert t.column("σ'p") == [45.0, 100.0]
    assert t.column("Cc") == [0.35, 0.30]
    assert t.column("Settlement S_c") == [
        x["settlement_mm"] for x in result.consolidation_layers]


def test_no_consolidation_gives_only_the_summary():
    result = SettlementAnalysis(q_applied=100.0, B=2.0, L=2.0,
                                Es_immediate=15000.0).compute()
    assert [t.name for t in result.get_tables()] == ["settlement_summary"]
