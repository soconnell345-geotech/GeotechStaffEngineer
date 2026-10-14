"""seismic_geotech calculation tables (backlog GM-39): the liquefaction
table is the result's own per-depth evaluation."""

from seismic_geotech import evaluate_liquefaction
from seismic_geotech.results import LiquefactionResult, SiteClassResult


def test_liquefaction_table_is_the_layer_results():
    layers = evaluate_liquefaction(
        layer_depths=[2.0, 4.0, 6.0, 8.0], layer_N160=[8, 12, 20, 30],
        layer_FC=[5, 15, 10, 35], layer_gamma=[18, 19, 19, 20],
        amax_g=0.3, gwt_depth=1.5, M=7.0)
    result = LiquefactionResult(layer_results=layers, amax_g=0.3,
                                magnitude=7.0, gwt_depth=1.5)
    (t,) = result.get_tables()
    assert t.n_rows == 4
    assert t.column("FS_liq") == [r["FOS_liq"] for r in layers]
    assert t.column("CSR (M 7.5)") == [r["CSR"] for r in layers]
    assert t.column("Liquefiable") == [r["liquefiable"] for r in layers]
    assert min(t.column("FS_liq")) == result.min_FOS


def test_results_without_tables_give_none():
    assert LiquefactionResult().get_tables() == []
    assert not hasattr(SiteClassResult(), "get_tables")
