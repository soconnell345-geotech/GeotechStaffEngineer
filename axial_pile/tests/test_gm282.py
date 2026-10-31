"""BACKLOG GM-282: axial_pile checked against GEC-12 chapter 7 as printed.

The factored resistance by Table 7-1 (p. 224): Eq. 7-3 in compression and
Eq. 7-38 in tension, each part of the resistance times the factor of the
method it was worked out with, and none where AASHTO gives none; the factor
of safety's basis. Page numbers are the printed GEC-12 Vol 1
(FHWA-NHI-16-009). Soil values are made up."""

import pytest

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_h_pile,
    make_pipe_pile,
)
from axial_pile.calc_steps import get_calc_steps


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def _sand():
    return AxialSoilProfile([
        AxialSoilLayer(5, "cohesionless", 18.2, friction_angle=31),
        AxialSoilLayer(30, "cohesionless", 19.6, friction_angle=35)],
        gwt_depth=2.2)


def _clay():
    return AxialSoilProfile([AxialSoilLayer(30, "cohesive", 17.7,
                                            cohesion=48)], gwt_depth=1.5)


def _layered():
    return AxialSoilProfile([
        AxialSoilLayer(3.5, "cohesionless", 18.1, friction_angle=30),
        AxialSoilLayer(7, "cohesive", 17.4, cohesion=66),
        AxialSoilLayer(20, "cohesionless", 19.9, friction_angle=37)],
        gwt_depth=2.7)


def test_sand_takes_nordlunds_factors():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.36, 0.011, closed_end=True),
                          soil=_sand(), pile_length=14,
                          include_uplift=True).compute()
    assert r.Q_factored == pytest.approx(0.45 * r.Q_ultimate)
    assert r.Q_uplift_factored == pytest.approx(0.35 * r.Q_skin)
    c = _by_what(r)["resistance factors, compression (LRFD)"]
    assert "Table 7-1" in c["basis"] and "Eq. 7-3" in c["basis"]
    assert "20 %" in c["basis"] and "p. 225" in c["basis"]
    up = _by_what(r)["resistance factors, uplift (LRFD)"]
    assert "Eq. 7-38" in up["basis"] and "0.35" in up["chosen"]
    d = r.to_dict()
    assert d["Q_factored_kN"] == pytest.approx(round(r.Q_factored, 1))
    assert d["resistance_factors"]["compression"]["toe"]["phi"] == 0.45


def test_clay_takes_the_alpha_methods_factors():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.36, 0.011, closed_end=True),
                          soil=_clay(), pile_length=14,
                          include_uplift=True).compute()
    # Table 7-1: alpha method 0.35 on the shaft and the 9 su toe; 0.25 up
    assert r.Q_factored == pytest.approx(0.35 * r.Q_ultimate)
    assert r.Q_uplift_factored == pytest.approx(0.25 * r.Q_skin)


def test_a_layered_profile_factors_each_layer_by_its_method():
    """GEC-12's worked example (Vol 3 p. 111): 0.45 on the Nordlund layers,
    0.35 on the alpha-method layers; p. 115 in tension, 0.35 and 0.25."""
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_layered(),
                          pile_length=18, include_uplift=True).compute()
    sand = sum(lb["skin_friction_kN"] for lb in r.layer_breakdown
               if lb["method"] == "nordlund")
    clay = sum(lb["skin_friction_kN"] for lb in r.layer_breakdown
               if lb["method"] == "tomlinson")
    assert sand > 0 and clay > 0
    assert r.Q_factored == pytest.approx(0.45 * sand + 0.35 * clay
                                         + 0.45 * r.Q_tip)
    assert r.Q_uplift_factored == pytest.approx(0.35 * sand + 0.25 * clay)
    rf = r.resistance_factors["compression"]["shaft"]
    assert rf["nordlund"]["phi"] == 0.45 and rf["tomlinson"]["phi"] == 0.35


def test_the_beta_method_has_no_compression_factor():
    """p. 258: AASHTO gives no factor for the beta method by Fellenius (Table
    7-1 'Differs in Manual'); the field method's phi_dyn is used. Table 7-1
    prints phi_up 0.20 for it in tension."""
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_layered(),
                          pile_length=18, method="beta",
                          include_uplift=True).compute()
    assert r.Q_factored is None
    comp = r.resistance_factors["compression"]
    assert "p. 258" in comp["not_factored"]
    assert "Table 7-2" in comp["not_factored"]
    assert r.Q_uplift_factored == pytest.approx(0.20 * r.Q_skin)
    assert "p. 258" in r.resistance_factors["uplift"]["note"]
    c = _by_what(r)["resistance factors, compression (LRFD)"]
    assert "R_r not computed" in c["chosen"]
    assert r.to_dict()["Q_factored_kN"] is None


def test_eq_7_90_has_no_factor():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.32, 0.009, closed_end=False),
                          soil=_clay(), pile_length=14,
                          include_uplift=True).compute()
    assert r.toe_details["governs"] == "Eq. 7-90"
    assert r.Q_factored is None and r.Q_uplift_factored is None
    assert "Eq. 7-90" in r.resistance_factors["compression"]["not_factored"]


def test_an_unplugged_open_pipe_factors_its_inside_friction_too():
    """The inside friction is the outside's on the inner perimeter, scaled:
    it takes the same factor; the plug weight (Eq. 7-89) is subtracted in
    full."""
    soil = AxialSoilProfile([AxialSoilLayer(20, "cohesionless", 17.6,
                                            friction_angle=30)],
                            gwt_depth=1.0)
    r = AxialPileAnalysis(pile=make_pipe_pile(0.46, 0.010, closed_end=False),
                          soil=soil, pile_length=6).compute()
    assert r.plug_weight > 0                    # the unplugged case governs
    assert r.Q_factored == pytest.approx(
        0.45 * (r.Q_skin + r.Q_tip) - r.plug_weight)


def test_api_alpha_layers_keep_the_alpha_methods_factor():
    soil = AxialSoilProfile([AxialSoilLayer(30, "cohesive", 19.2,
                                            cohesion=205)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=make_pipe_pile(0.36, 0.011, closed_end=True),
                          soil=soil, pile_length=12,
                          adhesion_beyond_chart="api").compute()
    assert r.layer_breakdown[0]["adhesion_source"] == "API alpha"
    assert r.Q_factored == pytest.approx(0.35 * r.Q_ultimate)
    c = _by_what(r)["resistance factors, compression (LRFD)"]
    assert "p. 247" in c["basis"]


def test_factored_resistance_against_depth():
    a = AxialPileAnalysis(pile=make_pipe_pile(0.36, 0.011, closed_end=True),
                          soil=_sand(), pile_length=20)
    rows = a.capacity_vs_depth(depth_min=6, depth_max=20, n_points=3)
    for row in rows:
        assert row["Q_factored_kN"] == pytest.approx(
            0.45 * row["Q_ultimate_kN"], abs=0.1)


def test_the_calc_steps_show_the_factored_resistance():
    for method in ("auto", "beta"):
        a = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_layered(),
                              pile_length=18, method=method,
                              include_uplift=True)
        r = a.compute()
        steps = [i for s in get_calc_steps(r, a) for i in s.items
                 if getattr(i, "title", "").startswith("Factored")]
        assert [s.title for s in steps] == [
            "Factored Axial Resistance (LRFD)",
            "Factored Uplift Resistance (LRFD)"]
        assert steps[0].reference == "FHWA GEC-12 Eq. 7-3, Table 7-1"
        if method == "beta":
            assert steps[0].substitution == "not computed"


def test_the_factor_of_safety_says_its_basis():
    """GEC-12 (2016) gives no factor of safety; the default 2.5 is FHWA
    NHI-06-089 Table 9-5's row for indicator piles with wave equation
    analysis (p. 9-25)."""
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_sand(),
                          pile_length=14).compute()
    c = _by_what(r)["factor of safety (allowable capacity)"]
    assert c["chosen"] == "2.5" and c["how"] == "default"
    assert "Table 9-5" in c["basis"] and "indicator piles" in c["basis"]
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=_sand(),
                          pile_length=14, factor_of_safety=2.0).compute()
    assert _by_what(r)["factor of safety (allowable capacity)"][
        "how"] == "given"
