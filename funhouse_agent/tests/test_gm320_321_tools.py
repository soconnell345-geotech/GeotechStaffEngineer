"""GM-320 and GM-321 at the tools: the settlement refusals and the
braced wall's per-support loads. Made-up values."""

import math

import pytest

from funhouse_agent.adapters.calc_package import METHOD_REGISTRY as CALC
from funhouse_agent.adapters.settlement import METHOD_REGISTRY as SETTLEMENT
from funhouse_agent.adapters.soe import METHOD_INFO as SOE_INFO
from funhouse_agent.adapters.soe import METHOD_REGISTRY as SOE

_CLAY = {"thickness": 2.0, "e0": 0.9, "Cc": 0.3, "Cr": 0.05}
_LAYERS = [dict(_CLAY, depth_to_center=3.0, sigma_v0=50.0),
           dict(_CLAY, depth_to_center=5.0, sigma_v0=70.0),
           dict(_CLAY, depth_to_center=7.0, sigma_v0=90.0)]


@pytest.mark.parametrize("stresses", [[40.0, 30.0], [40.0, 30.0, 20.0, 10.0],
                                      []])
def test_delta_sigmas_of_another_length_refused(stresses):
    with pytest.raises(ValueError) as e:
        SETTLEMENT["consolidation_settlement"](
            {"layers": _LAYERS, "delta_sigmas": stresses})
    assert f"delta_sigmas has {len(stresses)} value(s) and layers has 3" \
        in str(e.value)


def test_delta_sigmas_not_a_list_refused():
    with pytest.raises(ValueError, match="must be a list"):
        SETTLEMENT["consolidation_settlement"](
            {"layers": _LAYERS, "delta_sigmas": 40.0})


def test_delta_sigmas_one_per_layer_runs():
    out = SETTLEMENT["consolidation_settlement"](
        {"layers": _LAYERS, "delta_sigmas": [40.0, 30.0, 20.0]})
    assert len(out["layer_breakdown"]) == 3
    assert out["consolidation_settlement_mm"] == pytest.approx(
        sum(r["settlement_mm"] for r in out["layer_breakdown"]), abs=0.01)


_SCH = {"q_net": 120.0, "q_overburden": 20.0, "B": 2.0, "gamma_soil": 18.0,
        "layers": [{"depth_top": 0.0, "depth_bottom": 2.0, "Es": 12000.0}]}


def test_schmertmann_short_profile_refused_then_stated():
    with pytest.raises(ValueError, match="2 to 4 m below the footing base "
                                         "is not covered"):
        SETTLEMENT["schmertmann_settlement"](_SCH)
    cont = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="last_layer_continued"))
    rock = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="incompressible"))
    clay = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="settled_by_another_method"))
    assert rock["schmertmann_settlement_mm"] < cont["schmertmann_settlement_mm"]
    assert clay["schmertmann_settlement_mm"] == rock["schmertmann_settlement_mm"]
    for out, word in ((cont, "continued"), (rock, "incompressible"),
                      (clay, "not granular soil")):
        assert out["influence_depth_m"] == 4.0
        assert out["layers_bottom_m"] == 2.0
        assert out["depths_not_covered_m"] == [[2.0, 4.0]]
        assert word in out["choices"][0]["chosen"]
        assert "do not cover" in out["warnings"][0]
        assert "(below_profile)" in out["warnings"][0]


def test_schmertmann_gap_refused_then_listed():
    gap = dict(_SCH, layers=[
        {"depth_top": 0.0, "depth_bottom": 1.0, "Es": 12000.0},
        {"depth_top": 3.0, "depth_bottom": 4.0, "Es": 12000.0}])
    with pytest.raises(ValueError, match="1 to 3 m below the footing base "
                                         "is not covered"):
        SETTLEMENT["schmertmann_settlement"](gap)
    out = SETTLEMENT["schmertmann_settlement"](
        dict(gap, below_profile="incompressible"))
    assert out["depths_not_covered_m"] == [[1.0, 3.0]]
    assert "1 to 3 m" in out["warnings"][0]


def test_schmertmann_full_profile_has_no_note():
    full = dict(_SCH, layers=_SCH["layers"] + [
        {"depth_top": 2.0, "depth_bottom": 4.0, "Es": 12000.0}])
    out = SETTLEMENT["schmertmann_settlement"](full)
    assert "choices" not in out and "warnings" not in out
    assert "below_profile" not in out
    cont = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="last_layer_continued"))
    assert out["schmertmann_settlement_mm"] == cont["schmertmann_settlement_mm"]


@pytest.mark.parametrize("tool", ["combined_settlement_analysis",
                                  "settlement_package"])
def test_combined_tools_name_their_own_parameter(tool):
    """The refusal names schmertmann_below_profile (review B2)."""
    p = {"q_applied": 140.0, "q_overburden": 20.0, "B": 2.0,
         "immediate_method": "schmertmann", "gamma_soil": 18.0,
         "schmertmann_layers": _SCH["layers"]}
    call = SETTLEMENT[tool] if tool in SETTLEMENT else CALC[tool]
    with pytest.raises(ValueError) as e:
        call(p)
    msg = str(e.value)
    assert "with schmertmann_below_profile:" in msg
    assert "with below_profile" not in msg


def test_combined_analysis_takes_schmertmann_below_profile():
    p = {"q_applied": 140.0, "q_overburden": 20.0, "B": 2.0,
         "immediate_method": "schmertmann", "gamma_soil": 18.0,
         "schmertmann_layers": _SCH["layers"]}
    with pytest.raises(ValueError, match="is not covered"):
        SETTLEMENT["combined_settlement_analysis"](p)
    out = SETTLEMENT["combined_settlement_analysis"](
        dict(p, schmertmann_below_profile="incompressible"))
    assert any("do not cover" in w and "(schmertmann_below_profile)" in w
               for w in out["warnings"])


def test_app_a7_arrangement_through_the_tool():
    """NHI-06-089 App. A.7's arrangement (pp. A.7-10 to A.7-13) with
    made-up numbers: a 2 m strip on 3 m of sand over clay, the diagram to
    4B = 8 m, the clay consolidated. Hand-worked in
    settlement/tests/test_gm320_short_profile.py::test_app_a7_arrangement_by_hand:
    Se = 0.9 x 150 x 1.44728 / 30 000 = 6.51 mm."""
    p = {"q_applied": 180.0, "q_overburden": 30.0, "B": 2.0,
         "shape": "strip", "immediate_method": "schmertmann",
         "gamma_soil": 18.0,
         "schmertmann_layers": [{"depth_top": 0.0, "depth_bottom": 3.0,
                                 "Es": 30000.0}],
         "consolidation_layers": [dict(_CLAY, thickness=4.0,
                                       depth_to_center=5.0, sigma_v0=80.0)]}
    with pytest.raises(ValueError) as e:
        SETTLEMENT["combined_settlement_analysis"](p)
    assert "'settled_by_another_method'" in str(e.value)
    assert "'incompressible'" not in str(e.value)
    with pytest.raises(ValueError, match="consolidates its consolidation"):
        SETTLEMENT["combined_settlement_analysis"](
            dict(p, schmertmann_below_profile="incompressible"))
    out = SETTLEMENT["combined_settlement_analysis"](
        dict(p, schmertmann_below_profile="settled_by_another_method"))
    assert out["immediate_mm"] == pytest.approx(6.51, abs=0.01)
    assert out["consolidation_mm"] > 0
    entry = [c for c in out["choices"]
             if c["what"] == "ground the Schmertmann layers do not cover"][0]
    assert "not granular soil" in entry["chosen"]
    assert "3 to 8 m" in entry["chosen"]
    assert "App. A.7" in entry["basis"]
    text = " ".join(out["warnings"]) + " ".join(
        str(c["chosen"]) for c in out["choices"])
    assert "incompressible" not in text


_WALL = {"excavation_depth": 7.0,
         "layers": [{"thickness": 16.0, "unit_weight": 18.5,
                     "friction_angle": 33.0, "soil_type": "sand"}]}


def test_braced_excavation_returns_per_metre_and_per_support_loads():
    out = SOE["braced_excavation"](dict(_WALL, supports=[
        {"depth": 2.0, "spacing": 3.0},
        {"depth": 5.0, "support_type": "anchor", "spacing": 2.4,
         "angle_deg": 20.0}]))
    plain = SOE["braced_excavation"](dict(_WALL, supports=[
        {"depth": 2.0}, {"depth": 5.0}]))
    r1, r2 = out["support_reactions"]
    assert [r["load_kN_per_m"] for r in out["support_reactions"]] == \
        [r["load_kN_per_m"] for r in plain["support_reactions"]]
    assert r1["axial_load_kN_per_support"] == pytest.approx(
        r1["load_kN_per_m"] * 3.0, abs=0.05)
    assert r2["axial_load_kN_per_support"] == pytest.approx(
        r2["load_kN_per_m"] * 2.4 / math.cos(math.radians(20.0)), abs=0.05)
    for r in plain["support_reactions"]:
        assert r["spacing_m"] is None
        assert r["axial_load_kN_per_support"] is None
    assert any("No spacing given" in n for n in plain["notes"])


def test_every_promised_support_key_is_returned():
    text = SOE_INFO["braced_excavation"]["returns"]["support_reactions"]
    out = SOE["braced_excavation"](dict(_WALL, supports=[{"depth": 2.0}]))
    for key in ("load_kN_per_m", "axial_load_kN_per_m",
                "horizontal_load_kN_per_support",
                "axial_load_kN_per_support", "vertical_load_kN_per_support",
                "spacing_m", "angle_deg"):
        assert key in text and key in out["support_reactions"][0]


def test_null_angle_is_horizontal_as_before():
    null = SOE["braced_excavation"](dict(_WALL, supports=[
        {"depth": 2.0, "spacing": 3.0, "angle_deg": None}]))
    flat = SOE["braced_excavation"](dict(_WALL, supports=[
        {"depth": 2.0, "spacing": 3.0}]))
    assert null["support_reactions"] == flat["support_reactions"]
    assert null["support_reactions"][0]["angle_deg"] == 0.0


@pytest.mark.parametrize("support, text", [
    ({"depth": 2.0, "spacing": 0}, r"supports\[0\].*spacing must be positive"),
    ({"depth": 2.0, "angle_deg": 95}, r"supports\[0\].*less than 90"),
    ({"depth": 2.0, "preload_kN_per_m": 80}, "preload .*is not an input"),
    ({"depth": 2.0, "angle_deg": "15"},
     r"supports\[0\].*angle_deg must be a number"),
    ({"depth": 2.0, "spacing": "3"},
     r"supports\[0\].*spacing must be a number"),
])
def test_bad_support_inputs_refused(support, text):
    with pytest.raises(ValueError, match=text):
        SOE["braced_excavation"](dict(_WALL, supports=[support]))
