"""The ``downdrag.post_liquefaction_downdrag`` tool (BACKLOG GM-03): the
liquefaction tool's output in, the post-liquefaction neutral plane out."""

import pytest

from funhouse_agent import dispatch as D

_LAYERS = [
    {"thickness": 6, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 30, "beta": 0.3},
    {"thickness": 4, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 30, "beta": 0.3},
    {"thickness": 20, "soil_type": "cohesionless", "unit_weight": 20,
     "phi": 36, "beta": 0.45},
]
_STATIC = {"layers": _LAYERS, "pile_length": 20.0, "pile_diameter": 0.5,
           "pile_perimeter": 2.0, "pile_area": 0.25, "Q_dead": 600.0,
           "Nt": 30.0}


def _call(**params):
    return D.call_agent("downdrag", "post_liquefaction_downdrag",
                        {**_STATIC, **params})


def test_the_tool_runs_the_module_and_lists_its_choices():
    r = _call(liquefied_layers=[{"top": 6, "bottom": 10,
                                 "volumetric_strain": 0.03}])
    assert "error" not in r, r
    assert list(r)[:3] == ["post_liquefaction_basis", "warnings",
                           "methods_range_after"]
    assert r["neutral_plane_after_m"] == pytest.approx(18.13, abs=0.01)
    assert r["reconsolidation_settlement_mm"] == pytest.approx(120.0)
    assert r["defaults_used"]["liquefied_side_resistance"] == \
        "half_pre_earthquake"
    # the FS_liq threshold applies only to a liquefaction result
    assert "fs_liq_threshold" not in r["defaults_used"]
    how = {c["what"]: c["how"] for c in r["choices"]}
    assert how["side resistance of the liquefied layers"] == "default"
    assert how["liquefied layers"] == "given"


def test_the_liquefaction_tool_output_goes_straight_in():
    liq = D.call_agent("liquefaction", "liquefaction_analysis", {
        "depth": [3, 5, 7, 9, 11], "N160": [20, 8, 7, 9, 30],
        "FC": [5, 5, 10, 5, 5], "gamma": [18, 19, 19, 19, 20],
        "amax_g": 0.3, "gwt_depth": 2.0, "magnitude": 7.5})
    assert "error" not in liq, liq
    r = _call(gwt_depth=2.0, liquefaction_result=liq,
              volumetric_strain=0.025)
    assert "error" not in r, r
    liquefied = [row["depth_m"] for row in liq["layer_results"]
                 if row["FOS_liq"] <= 1.0]
    assert liquefied == [3.0, 5.0, 7.0, 9.0]
    # each reading halfway to its neighbours, not above the water table
    assert [(z["top_m"], z["bottom_m"]) for z in r["liquefied_zones"]] == \
        [(2.0, 10.0)]
    assert r["defaults_used"]["fs_liq_threshold"] == 1.0
    how = {c["what"]: c["how"] for c in r["choices"]}
    assert how["liquefied layers"] == "module"
    assert how["reconsolidation volumetric strain"] == "given"


def test_an_spt_result_without_a_strain_says_what_to_give():
    liq = D.call_agent("liquefaction", "liquefaction_analysis", {
        "depth": [3, 5, 7], "N160": [8, 7, 30], "FC": [5, 5, 5],
        "gamma": [19, 19, 20], "amax_g": 0.3, "gwt_depth": 1.0,
        "magnitude": 7.5})
    r = _call(gwt_depth=1.0, liquefaction_result=liq)
    assert "error" in r
    assert "Give volumetric_strain" in r["error"]


def test_a_misspelt_layer_key_is_refused_with_the_did_you_mean():
    r = _call(liquefied_layers=[{"top": 6, "bottom": 10, "strain": 0.03}])
    assert "strain -> volumetric_strain" in r["error"]
    r = _call(liquefied_layers=[{"top": 6, "bottom": 10,
                                 "volumetric_strain": 0.03}],
              liquified_side_resistance="zero")
    assert "unknown parameter(s) ['liquified_side_resistance']" in r["error"]


def test_the_static_tool_is_unchanged_by_the_shared_builder():
    static = D.call_agent("downdrag", "downdrag_analysis", dict(_STATIC))
    r = _call(liquefied_layers=[{"top": 6, "bottom": 10,
                                 "volumetric_strain": 0.03}])
    assert r["neutral_plane_before_m"] == pytest.approx(
        static["neutral_plane_depth_m"])
    assert r["drag_force_before_kN"] == pytest.approx(
        static["drag_force_kN"])
