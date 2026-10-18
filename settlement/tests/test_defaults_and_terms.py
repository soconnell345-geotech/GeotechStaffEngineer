"""GM-142 (defaults made correct or stated) and GM-143 (the result carries
the values the analysis used, and the calc steps print them) for the
settlement module."""

import math

import pytest

from settlement import (
    ConsolidationLayer, SchmertmannLayer, SettlementAnalysis,
    consolidation_settlement_layer, schmertmann_settlement,
    secondary_settlement,
)
from settlement.consolidation import consolidation_layer_terms
from settlement.immediate import schmertmann_terms
from settlement.stress_distribution import (
    approximate_2to1, approximate_2to1_strip, boussinesq_center_rectangular,
    boussinesq_center_strip, stress_at_depth, westergaard_center_strip,
    westergaard_rectangular,
)


def _clay(e0=0.9, z=4.0, H=2.0):
    return ConsolidationLayer(thickness=H, depth_to_center=z, e0=e0,
                              Cc=0.3, Cr=0.05, sigma_v0=60.0)


def _sands():
    return [SchmertmannLayer(depth_top=0.0, depth_bottom=4.0, Es=15000.0),
            SchmertmannLayer(depth_top=4.0, depth_bottom=10.0, Es=25000.0)]


# ---------------------------------------------------------------- strips

@pytest.mark.parametrize("z", [0.5, 1.5, 4.0, 10.0])
def test_strip_closed_forms_are_the_long_rectangle_limit(z):
    q, B = 100.0, 2.0
    assert approximate_2to1_strip(q, B, z) == pytest.approx(
        approximate_2to1(q, B, 1e6, z), rel=1e-4)
    assert boussinesq_center_strip(q, B, z) == pytest.approx(
        boussinesq_center_rectangular(q, B, 1e4, z), rel=1e-3)
    # the centre: four corner rectangles of B/2 x L/2
    assert westergaard_center_strip(q, B, z) == pytest.approx(
        4.0 * westergaard_rectangular(q, B / 2.0, 5e3, z), rel=1e-3)


def test_stress_at_depth_takes_an_infinite_L_as_a_strip():
    for method in ("2:1", "boussinesq", "westergaard"):
        assert stress_at_depth(100.0, 2.0, math.inf, 3.0, method=method) \
            > stress_at_depth(100.0, 2.0, 2.0, 3.0, method=method)
    with pytest.raises(ValueError):
        stress_at_depth(100.0, 2.0, math.inf, 3.0, method="boussinesq",
                        location="corner")


def test_a_strip_with_no_L_is_a_strip_not_a_square():
    # Before GM-142 a strip with no L was run as a B x B square: at z = 2B
    # the 2:1 stress was 16.7 kPa instead of the strip's 50 kPa.
    a = SettlementAnalysis(q_applied=100.0, B=2.0, footing_shape="strip",
                           consolidation_layers=[_clay(z=4.0)])
    r = a.compute()
    assert r.L is None
    assert r.consolidation_terms[0]["delta_sigma"] == pytest.approx(
        approximate_2to1_strip(100.0, 2.0, 4.0))


def test_strip_with_no_L_under_schmertmann_is_plane_strain():
    a = SettlementAnalysis(q_applied=150.0, B=2.0, footing_shape="strip",
                           immediate_method="schmertmann",
                           schmertmann_layers=_sands(), gamma_soil=18.0)
    r = a.compute()
    t = r.schmertmann_terms
    assert t["strip"] is True
    assert t["z_max"] == pytest.approx(4.0 * 2.0)
    assert r.immediate * 1000 == pytest.approx(t["Se"] * 1000, abs=1e-6)


def test_elastic_strip_with_no_L_is_refused_unless_Iw_is_given():
    with pytest.raises(ValueError, match="strip footing with no length L"):
        SettlementAnalysis(q_applied=100.0, B=2.0, footing_shape="strip",
                           Es_immediate=10000.0).compute()
    r = SettlementAnalysis(q_applied=100.0, B=2.0, footing_shape="strip",
                           Es_immediate=10000.0, Iw_immediate=2.0).compute()
    assert r.Iw == 2.0 and r.immediate > 0


def test_a_rectangle_with_no_L_is_refused():
    with pytest.raises(ValueError, match="needs its length L"):
        SettlementAnalysis(q_applied=100.0, B=2.0,
                           footing_shape="rectangular")


def test_a_square_with_no_L_takes_L_equal_to_B_and_says_so():
    a = SettlementAnalysis(q_applied=100.0, B=2.5,
                           consolidation_layers=[_clay()])
    assert a.L == 2.5 and a.L_given is False
    r = a.compute()
    length = {c["what"]: c for c in r.choices}["footing length"]
    assert length["how"] == "default" and "L = B = 2.5" in length["chosen"]


# ---------------------------------------------------------- secondary e0

def test_secondary_uses_each_layers_own_e0():
    layers = [_clay(e0=0.9, z=3.0), _clay(e0=1.4, z=5.0)]
    a = SettlementAnalysis(q_applied=80.0, B=3.0,
                           consolidation_layers=layers, C_alpha=0.01,
                           t_secondary=30.0, cv=2.0, Hdr=2.0)
    r = a.compute()
    t = r.secondary_terms
    assert t["e0_given"] is None
    expected = sum(secondary_settlement(0.01, l.thickness, t["t1_years"],
                                        t["t2_years"], e0=l.e0)
                   for l in layers)
    assert r.secondary == pytest.approx(expected)
    assert [x["e0"] for x in t["layers"]] == [0.9, 1.4]
    e0 = {c["what"]: c for c in r.choices}[
        "void ratio e0 for secondary compression"]
    assert e0["how"] == "module" and "0.9 to 1.4" in e0["chosen"]


def test_a_given_e0_secondary_still_applies_to_every_layer():
    layers = [_clay(e0=0.9, z=3.0), _clay(e0=1.4, z=5.0)]
    r = SettlementAnalysis(q_applied=80.0, B=3.0,
                           consolidation_layers=layers, C_alpha=0.01,
                           t_secondary=30.0, cv=2.0, Hdr=2.0,
                           e0_secondary=1.0).compute()
    t = r.secondary_terms
    assert r.secondary == pytest.approx(secondary_settlement(
        0.01, 4.0, t["t1_years"], t["t2_years"], e0=1.0))


# ------------------------------------------- GM-143: the terms it used

def test_result_terms_are_the_modules_own():
    layers = [_clay(z=3.0), _clay(e0=1.1, z=5.0)]
    a = SettlementAnalysis(q_applied=120.0, B=2.0, L=3.0,
                           footing_shape="rectangular",
                           stress_method="boussinesq",
                           consolidation_layers=layers,
                           immediate_method="schmertmann",
                           schmertmann_layers=_sands(), gamma_soil=18.0)
    r = a.compute()
    for layer, terms in zip(layers, r.consolidation_terms):
        ds = stress_at_depth(120.0, 2.0, 3.0, layer.depth_to_center,
                             method="boussinesq")
        assert terms["delta_sigma"] == pytest.approx(ds)
        assert terms == pytest.approx(consolidation_layer_terms(layer, ds))
        assert terms["Sc"] == pytest.approx(
            consolidation_settlement_layer(layer, ds))
    t = schmertmann_terms(120.0, 0.0, 2.0, _sands(), "rectangular", 0.0,
                          L=3.0, gamma_soil=18.0)
    assert r.schmertmann_terms["Se"] == pytest.approx(t["Se"])
    assert t["Se"] == pytest.approx(schmertmann_settlement(
        120.0, 0.0, 2.0, _sands(), "rectangular", 0.0, L=3.0,
        gamma_soil=18.0))


def test_calc_steps_print_the_results_own_influence_factor_and_stress():
    from settlement.calc_steps import get_calc_steps
    a = SettlementAnalysis(q_applied=100.0, B=2.0, footing_shape="strip",
                           consolidation_layers=[_clay(z=4.0)])
    r = a.compute()
    text = repr(get_calc_steps(r, a))
    # the strip's 2:1 stress at z = 2B, 33.3 kPa; the B x B square's
    # (the pre-GM-142 reading of a strip with no L) is 11.1 kPa
    assert "\u0394\u03c3 = 33.3 kPa" in text
    assert "11.1" not in text
