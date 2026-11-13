"""GM-320 and GM-321 at the tools: the settlement refusals and the
braced wall's per-support loads. Made-up values."""

import math

import pytest

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
    with pytest.raises(ValueError, match="from 2 to 4 m is not covered"):
        SETTLEMENT["schmertmann_settlement"](_SCH)
    cont = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="last_layer_continued"))
    rock = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="incompressible"))
    assert rock["schmertmann_settlement_mm"] < cont["schmertmann_settlement_mm"]
    for out, word in ((cont, "continued"), (rock, "incompressible")):
        assert out["influence_depth_m"] == 4.0
        assert out["layers_bottom_m"] == 2.0
        assert word in out["choices"][0]["chosen"]
        assert "stop at 2 m" in out["warnings"][0]


def test_schmertmann_full_profile_has_no_note():
    full = dict(_SCH, layers=_SCH["layers"] + [
        {"depth_top": 2.0, "depth_bottom": 4.0, "Es": 12000.0}])
    out = SETTLEMENT["schmertmann_settlement"](full)
    assert "choices" not in out and "warnings" not in out
    assert "below_profile" not in out
    cont = SETTLEMENT["schmertmann_settlement"](
        dict(_SCH, below_profile="last_layer_continued"))
    assert out["schmertmann_settlement_mm"] == cont["schmertmann_settlement_mm"]


def test_combined_analysis_takes_schmertmann_below_profile():
    p = {"q_applied": 140.0, "q_overburden": 20.0, "B": 2.0,
         "immediate_method": "schmertmann", "gamma_soil": 18.0,
         "schmertmann_layers": _SCH["layers"]}
    with pytest.raises(ValueError, match="is not covered"):
        SETTLEMENT["combined_settlement_analysis"](p)
    out = SETTLEMENT["combined_settlement_analysis"](
        dict(p, schmertmann_below_profile="incompressible"))
    assert any("stop at 2 m" in w for w in out["warnings"])


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


@pytest.mark.parametrize("support, text", [
    ({"depth": 2.0, "spacing": 0}, r"supports\[0\].*spacing must be positive"),
    ({"depth": 2.0, "angle_deg": 95}, r"supports\[0\].*less than 90"),
    ({"depth": 2.0, "preload_kN_per_m": 80}, "preload .*is not an input"),
])
def test_bad_support_inputs_refused(support, text):
    with pytest.raises(ValueError, match=text):
        SOE["braced_excavation"](dict(_WALL, supports=[support]))
