"""BACKLOG GM-237: the driven-pile alpha method's loose ends from GM-229.

- a clay layer's site-specific alpha (or Ca) in place of the chart, with
  where it came from (GEC-12 Vol I p. 246; NHI-06-089 Example 9-3, p. 9-58);
- a clay layer's own call on "highly overconsolidated" (GEC-12 p. 247 and
  Step 1, p. 250), in place of the su / sigma'vo line for that layer;
- Fig. 7-18 for a clay directly under a softer clay: the bottom graph is
  for stiff clays "without any different overlying strata" (p. 247) and the
  middle graph needs the clay above to be soft, which GEC-12 gives no su to
  decide, so the run asks for the layer's case where the two differ.

Values are made up. Page numbers are the printed GEC-12 Vol 1."""

import pytest

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_concrete_pile,
    make_pipe_pile,
)
from axial_pile.calc_steps import _auto_method_notes
from axial_pile.capacity import JudgmentNeeded
from axial_pile.tomlinson import SOFT_OVER_STIFF_CLAY, STIFF_CLAY

BASIS = "made-up load test at the site"


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def _two_clays(**lower):
    return AxialSoilProfile(layers=[
        AxialSoilLayer(4.0, "cohesive", 17.5, cohesion=25.0),
        AxialSoilLayer(12.0, "cohesive", 19.5, cohesion=55.0, **lower)])


# ------------------------------------------------ site-specific adhesion

def test_a_layers_site_alpha_replaces_the_chart():
    pile = make_concrete_pile(0.35)
    soil = _two_clays(alpha=0.8, adhesion_basis=BASIS)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0).compute()
    upper, lower = r.layer_breakdown
    assert upper["adhesion_source"] == "Fig. 7-17"
    assert lower["adhesion_source"] == "given"
    assert lower["alpha"] == pytest.approx(0.8)
    assert lower["skin_friction_kN"] == pytest.approx(
        0.8 * 55.0 * pile.perimeter * 9.0)
    c = _by_what(r)["site-specific adhesion (alpha method)"]
    assert c["how"] == "given" and BASIS in c["chosen"]
    assert "p. 246" in c["basis"] and "Example 9-3" in c["basis"]
    # one layer still read a chart, so the chart is still stated
    assert "adhesion chart (alpha-method)" in _by_what(r)


def test_a_layers_site_adhesion_in_kpa():
    pile = make_concrete_pile(0.35)
    soil = _two_clays(adhesion=40.0, adhesion_basis=BASIS)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0).compute()
    lower = r.layer_breakdown[1]
    assert lower["skin_friction_kN"] == pytest.approx(
        40.0 * pile.perimeter * 9.0)
    assert lower["alpha"] == pytest.approx(40.0 / 55.0)


@pytest.mark.parametrize("chart", ["fig_7_17", "fig_7_18", "smaller"])
def test_no_chart_is_stated_where_every_clay_gives_its_own(chart):
    pile = make_concrete_pile(0.35)
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(4.0, "cohesive", 17.5, cohesion=25.0, alpha=1.0,
                       adhesion_basis=BASIS),
        AxialSoilLayer(12.0, "cohesive", 19.5, cohesion=55.0, alpha=1.0,
                       adhesion_basis=BASIS)])
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0,
                          adhesion_chart=chart).compute()
    whats = _by_what(r)
    assert "adhesion chart (alpha-method)" not in whats
    assert "adhesion (alpha-method)" not in whats
    assert "Fig. 7-18 stratigraphy case" not in whats
    assert r.Q_skin == pytest.approx(
        (25.0 * 4.0 + 55.0 * 9.0) * pile.perimeter)
    notes = _auto_method_notes(r, None)
    assert "Fig. 7-17" not in notes and "given" in notes and BASIS in notes


def test_a_site_alpha_needs_no_highly_overconsolidated_line():
    # su 200 kPa (4.2 ksf) is past Fig. 7-17's curves: without a line the
    # chart's run is refused, with the layer's own alpha it is not
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(15.0, "cohesive", 20.0, cohesion=200.0)])
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    with pytest.raises(JudgmentNeeded):
        AxialPileAnalysis(pile=pile, soil=soil, pile_length=12.0).compute()
    soil.layers[0].alpha = 0.45
    soil.layers[0].adhesion_basis = BASIS
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=12.0).compute()
    assert r.layer_breakdown[0]["alpha"] == pytest.approx(0.45)


def test_a_site_alpha_keeps_the_shrink_swell_zone():
    pile = make_concrete_pile(0.35)
    soil = _two_clays(alpha=0.8, adhesion_basis=BASIS)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0,
                          shrink_swell_depth=6.0).compute()
    lower = r.layer_breakdown[1]
    assert lower["skin_friction_kN"] == pytest.approx(
        0.8 * 55.0 * pile.perimeter * 7.0)


@pytest.mark.parametrize("kwargs, words", [
    ({"alpha": 0.9}, "adhesion_basis"),
    ({"alpha": 0.9, "adhesion": 30.0, "adhesion_basis": BASIS}, "not both"),
    ({"adhesion_basis": BASIS}, "give the value"),
    ({"alpha": -0.2, "adhesion_basis": BASIS}, "> 0"),
    ({"alpha": 0.9, "adhesion_basis": "  "}, "adhesion_basis"),
    ({"highly_overconsolidated": "yes"}, "true or false"),
])
def test_layer_inputs_refused(kwargs, words):
    with pytest.raises(ValueError, match=words):
        AxialSoilLayer(5.0, "cohesive", 18.0, cohesion=40.0, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.9, "adhesion_basis": BASIS},
    {"highly_overconsolidated": True}])
def test_alpha_inputs_on_a_sand_layer_refused(kwargs):
    with pytest.raises(ValueError, match="cohesive layer"):
        AxialSoilLayer(5.0, "cohesionless", 18.0, friction_angle=32.0,
                       **kwargs)


def test_inputs_the_run_would_not_use_are_refused():
    pile = make_concrete_pile(0.35)
    with pytest.raises(ValueError, match="beta"):
        AxialPileAnalysis(pile=pile, soil=_two_clays(
            alpha=0.8, adhesion_basis=BASIS), pile_length=13.0,
            method="beta")
    with pytest.raises(ValueError, match="chart_end"):
        AxialPileAnalysis(pile=pile, soil=_two_clays(
            highly_overconsolidated=True), pile_length=13.0,
            adhesion_beyond_chart="chart_end")


# ------------------------------------ a layer's own highly overconsolidated

def _stiff(flag=None):
    return AxialSoilProfile(layers=[
        AxialSoilLayer(15.0, "cohesive", 20.0, cohesion=200.0,
                       highly_overconsolidated=flag)])


def test_a_layers_flag_takes_api_alpha_or_the_curves_end():
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    high = AxialPileAnalysis(pile=pile, soil=_stiff(True),
                             pile_length=12.0).compute()
    assert high.layer_breakdown[0]["adhesion_source"] == "API alpha"
    low = AxialPileAnalysis(pile=pile, soil=_stiff(False),
                            pile_length=12.0).compute()
    assert low.layer_breakdown[0]["adhesion_source"] == "Fig. 7-17 end"
    c = _by_what(high)["highly overconsolidated clay (alpha method)"]
    assert c["chosen"].startswith("as each layer gives it")
    assert c["how"] == "given"
    assert any("highly overconsolidated" in w and "given" in w
               for w in high.warnings)


def test_a_layers_flag_wins_over_the_line():
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    # su / sigma'vo here is well above 0.5: the line alone takes API alpha
    by_line = AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=12.0,
                                highly_overconsolidated_su_ratio=0.5
                                ).compute()
    assert by_line.layer_breakdown[0]["adhesion_source"] == "API alpha"
    flagged = AxialPileAnalysis(pile=pile, soil=_stiff(False),
                                pile_length=12.0,
                                highly_overconsolidated_su_ratio=0.5
                                ).compute()
    assert flagged.layer_breakdown[0]["adhesion_source"] == "Fig. 7-17 end"
    assert "not highly overconsolidated (given)" in _by_what(flagged)[
        "highly overconsolidated clay (alpha method)"]["chosen"]


# ------------------------- a softer clay over a firmer clay (Figs. 7-17/18)

def test_fig_7_17_reads_d_from_the_ground_surface_whatever_lies_above():
    # p. 246 and the figure's legend (p. 248): D from the ground surface to
    # the bottom of the clay layer, or to the toe if less; Fig. 7-17 has no
    # stratigraphy case, and its D = 40b solid curve passes above Ca = su
    pile = make_concrete_pile(0.3)
    soil = _two_clays()
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0).compute()
    lower = r.layer_breakdown[1]
    assert lower["adhesion_source"] == "Fig. 7-17"
    assert lower["D_over_b"] == pytest.approx(13.0 / 0.3)
    assert lower["alpha"] > 1.0


@pytest.mark.parametrize("chart", ["fig_7_18", "smaller"])
def test_fig_7_18_asks_for_the_case_of_a_clay_under_a_softer_clay(chart):
    pile = make_concrete_pile(0.3)
    with pytest.raises(JudgmentNeeded, match="fig_7_18_case"):
        AxialPileAnalysis(pile=pile, soil=_two_clays(), pile_length=13.0,
                          adhesion_chart=chart).compute()


def test_fig_7_18_takes_the_case_given():
    pile = make_concrete_pile(0.3)
    soft = AxialPileAnalysis(
        pile=pile, soil=_two_clays(fig_7_18_case=SOFT_OVER_STIFF_CLAY),
        pile_length=13.0, adhesion_chart="fig_7_18").compute()
    lower = soft.layer_breakdown[1]
    assert lower["fig_7_18_case"] == SOFT_OVER_STIFF_CLAY
    # D from the top of the firmer clay (4 m) to the toe (13 m)
    assert lower["D_over_b_fig_7_18"] == pytest.approx(9.0 / 0.3)
    stiff = AxialPileAnalysis(
        pile=pile, soil=_two_clays(fig_7_18_case=STIFF_CLAY),
        pile_length=13.0, adhesion_chart="fig_7_18").compute()
    assert stiff.layer_breakdown[1]["alpha"] > lower["alpha"]


def _soft_over_firm(upper=None, lower=None):
    return AxialSoilProfile(layers=[
        AxialSoilLayer(8.0, "cohesive", 17.0, cohesion=30.0, **(upper or {})),
        AxialSoilLayer(20.0, "cohesive", 19.0, cohesion=90.0, **(lower or {}))],
        gwt_depth=1.0)


def test_the_ask_says_to_give_the_profiles_case_on_the_top_clay_layer():
    # a case given on the lower layer starts its stratum there (GM-144), so
    # the profile's bottom graph, D from the ground surface, is had only by
    # giving 'stiff_clay' on the stratum's top clay layer
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    with pytest.raises(JudgmentNeeded) as ask:
        AxialPileAnalysis(pile=pile, soil=_soft_over_firm(), pile_length=20.0,
                          adhesion_chart="fig_7_18").compute()
    msg = str(ask.value)
    assert "clay stratum from 0.00 m" in msg
    assert "on the TOP clay layer of the stratum" in msg
    r = AxialPileAnalysis(
        pile=pile, soil=_soft_over_firm(upper={"fig_7_18_case": STIFF_CLAY}),
        pile_length=20.0, adhesion_chart="fig_7_18").compute()
    lower = r.layer_breakdown[1]
    assert lower["fig_7_18_case"] == STIFF_CLAY
    assert lower["D_over_b_fig_7_18"] == pytest.approx(20.0 / 0.4)
    assert lower["alpha"] == pytest.approx(0.941, abs=5e-4)
    assert r.Q_ultimate == pytest.approx(1674.5, abs=0.1)


def test_the_ask_names_the_chart_each_ca_came_from():
    # under 'smaller' the run's Ca here is Fig. 7-17's, not the bottom graph's
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    fig_7_17 = AxialPileAnalysis(pile=pile, soil=_soft_over_firm(),
                                 pile_length=20.0).compute()
    ca_17 = fig_7_17.layer_breakdown[1]["alpha"] * 90.0
    with pytest.raises(JudgmentNeeded) as ask:
        AxialPileAnalysis(pile=pile, soil=_soft_over_firm(), pile_length=20.0,
                          adhesion_chart="smaller").compute()
    msg = str(ask.value)
    assert f"Ca {ca_17:.1f} kPa by Fig. 7-17," in msg
    assert "(bottom graph)" not in msg
    assert "kPa by the soft clay over stiff clay graph" in msg
    with pytest.raises(JudgmentNeeded) as ask:
        AxialPileAnalysis(pile=pile, soil=_soft_over_firm(), pile_length=20.0,
                          adhesion_chart="fig_7_18").compute()
    assert "kPa by the stiff clay without any different overlying strata " \
           "(bottom graph)" in str(ask.value)


def test_a_layer_with_its_own_alpha_is_not_listed_under_a_fig_7_18_case():
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    r = AxialPileAnalysis(
        pile=pile, soil=_soft_over_firm(
            upper={"alpha": 1.0, "adhesion_basis": BASIS},
            lower={"fig_7_18_case": SOFT_OVER_STIFF_CLAY}),
        pile_length=20.0, adhesion_chart="smaller").compute()
    chosen = _by_what(r)["Fig. 7-18 stratigraphy case"]["chosen"]
    assert "0-8 m" not in chosen and "none (Fig. 7-17)" not in chosen
    assert "8-20 m" in chosen


def test_a_firmer_clay_over_a_softer_one_keeps_the_profiles_case():
    pile = make_concrete_pile(0.3)
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(4.0, "cohesive", 19.5, cohesion=55.0),
        AxialSoilLayer(12.0, "cohesive", 17.5, cohesion=45.0)])
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.0,
                          adhesion_chart="fig_7_18").compute()
    assert [lb["fig_7_18_case"] for lb in r.layer_breakdown] == [
        STIFF_CLAY, STIFF_CLAY]


def test_the_default_chart_never_asks_for_a_case():
    pile = make_concrete_pile(0.3)
    r = AxialPileAnalysis(pile=pile, soil=_two_clays(),
                          pile_length=13.0).compute()
    assert "Fig. 7-18 stratigraphy case" not in _by_what(r)
