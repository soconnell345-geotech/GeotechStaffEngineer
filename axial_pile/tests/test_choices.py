"""What an axial pile result says it chose (backlog GM-115, GM-121): method
"auto" resolved by soil type, the toe method, the GEC-12 charts and limits
it applied and the judgments GEC-12 leaves to the engineer."""

import pytest

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_concrete_pile,
    make_h_pile, make_pipe_pile,
)

KSF = 47.880259


def _soil():
    return AxialSoilProfile(layers=[
        AxialSoilLayer(10, "cohesionless", 18.0, friction_angle=32),
        AxialSoilLayer(10, "cohesive", 17.0, cohesion=60),
    ], gwt_depth=2.0)


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def test_auto_says_which_method_each_depth_got():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=True),
                          soil=_soil(), pile_length=15).compute()
    c = _by_what(r)
    side = c["side resistance method"]
    assert side["chosen"] == "Nordlund 0-10 m; Tomlinson alpha 10-15 m"
    assert side["how"] == "module" and "GEC-12" in side["basis"]
    assert any("beta" in a for a in side["alternatives"])
    assert c["toe resistance method"]["chosen"] == "9 su"
    # citations are the printed GEC-12 (FHWA-NHI-16-009)
    assert "Sec. 7.2.1.3.1" in side["basis"]
    assert "Sec. 7.2.1.3.2" in side["basis"]
    assert "Eq. 7-11" in c["toe resistance method"]["basis"]
    assert "Fig. 7-9" in c["Nordlund delta/phi"]["basis"]
    # the per-layer methods the choice names are the ones that ran
    assert [lb["method"] for lb in r.layer_breakdown] == ["nordlund",
                                                          "tomlinson"]
    assert c["Nordlund delta/phi"]["chosen"] == (
        "0.80 (Fig. 7-9 curve a, pipe piles)")
    assert c["Nordlund delta/phi"]["how"] == "module"
    assert c["displaced volume V (Nordlund)"]["chosen"].startswith("3.04")
    assert "Fig. 7-17" in c["adhesion (alpha-method)"]["chosen"]
    assert "open-ended pipe" not in c
    d = r.to_dict()
    assert d["choices"] == r.choices and d["method"] == "auto"
    assert "Choices:" in r.summary()


def test_beta_is_given_and_its_clay_phi_fallback_is_reported():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=True),
                          soil=_soil(), pile_length=15,
                          method="beta").compute()
    c = _by_what(r)
    assert c["side resistance method"]["how"] == "given"
    fallback = c["phi' of cohesive layers given no friction angle (beta "
                 "method)"]
    assert fallback == {**fallback, "chosen": "25 deg", "how": "default"}
    assert "Table 7-9" in fallback["basis"]


def test_open_pipe_reports_the_governing_case():
    r = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=False),
                          soil=_soil(), pile_length=15).compute()
    pipe = _by_what(r)["open-ended pipe"]
    assert pipe["how"] == "module"
    assert pipe["chosen"].split(" ")[0] in ("plugged", "unplugged")
    assert len(pipe["alternatives"]) == 2
    assert "Sec. 7.10.7" in pipe["basis"]
    c = _by_what(r)
    inside = c["interior unit side resistance (open pipe)"]
    assert inside["how"] == "default" and inside["chosen"].startswith("0.333")
    assert "1/3 to 1/2" in inside["basis"]
    assert c["soil plug weight Wp (open pipe)"]["how"] == "module"
    V = c["displaced volume V (Nordlund)"]
    assert "gross area" in V["chosen"] and "steel area" in V["alternatives"][0]
    given = AxialPileAnalysis(
        pile=make_pipe_pile(0.6, 0.012, closed_end=False), soil=_soil(),
        pile_length=15, inside_friction_ratio=0.5).compute()
    assert _by_what(given)["interior unit side resistance (open pipe)"][
        "how"] == "given"


def test_toe_stress_limit_is_named():
    # GEC-12 Sec. 7.2.1.3.1 Step 8 limits sigma'p to 3 ksf.
    deep = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesionless", 19.0, friction_angle=34)],
        gwt_depth=30.0)
    pile = make_pipe_pile(0.6, 0.012, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=deep, pile_length=20).compute()
    toe = _by_what(r)["sigma'p at the toe (Nordlund, Step 8)"]
    assert r.sigma_v_tip > 3 * KSF
    assert toe["chosen"].startswith(f"{3 * KSF:.1f} kPa (limit")
    assert toe["how"] == "default"
    assert toe["basis"].startswith("Nordlund method, Step 8")
    assert "p. 244" in toe["basis"]
    off = AxialPileAnalysis(pile=pile, soil=deep, pile_length=20,
                            toe_stress_limit=None).compute()
    toe = _by_what(off)["sigma'p at the toe (Nordlund, Step 8)"]
    assert toe["chosen"].startswith("not limited") and toe["how"] == "given"
    beta = AxialPileAnalysis(pile=pile, soil=deep, pile_length=20,
                             method="beta").compute()
    assert "sigma'p at the toe (Nordlund, Step 8)" not in _by_what(beta)


def _stiff():
    """A clay past Fig. 7-17's curves (su 4 ksf): su / sigma'vo about 2.4
    at the middle of a 15 m pile."""
    return AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=4.0 * KSF)],
        gwt_depth=1.0)


def test_past_the_chart_the_line_is_the_engineers():
    """GM-229 (owner OWN-63): past Fig. 7-17's curves GEC-12 recommends
    API alpha for highly overconsolidated clays (p. 247, alpha-method Step
    1) and gives no su / sigma'vo that marks one, so the default run is
    refused until the engineer's line is given."""
    from axial_pile.capacity import JudgmentNeeded
    pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
    with pytest.raises(JudgmentNeeded) as e:
        AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15).compute()
    msg = str(e.value)
    assert msg.startswith("alpha method (GEC-12 Sec. 7.2.1.3.2)")
    for text in ("su = 4.00 ksf", "Fig. 7-17's curves (3.05 ksf)",
                 "engineer's judgment", "highly_overconsolidated_su_ratio",
                 "'chart_end' or 'api'", "p. 247", "Step 1, p. 250"):
        assert text in msg, text
    # su / sigma'vo at the middle of the 15 m pile, given back in the error
    sv = _stiff().effective_stress_at_depth(7.5)
    assert f"su / sigma'vo is {4.0 * KSF / sv:.2f}" in msg
    # a pile that stays within the chart needs no line
    soft = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=60)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=pile, soil=soft, pile_length=15).compute()
    assert not any(c["what"].startswith("su past") for c in r.choices)
    # capacity_vs_depth refuses too, rather than leaving the depths out
    with pytest.raises(JudgmentNeeded):
        AxialPileAnalysis(pile=pile, soil=_stiff(),
                          pile_length=15).capacity_vs_depth(n_points=3)


@pytest.mark.parametrize("line, high", [(2.0, True), (3.0, False)])
def test_each_layer_says_its_side_of_the_line(line, high):
    pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15,
                          highly_overconsolidated_su_ratio=line).compute()
    (lb,) = r.layer_breakdown
    sv = _stiff().effective_stress_at_depth(7.5)
    assert lb["su_over_sigma_v"] == pytest.approx(4.0 * KSF / sv)
    assert lb["highly_overconsolidated"] is high
    assert lb["adhesion_source"] == ("API alpha" if high else "Fig. 7-17 end")
    c = _by_what(r)
    side = c["su past Fig. 7-17 (alpha method)"]
    assert side["how"] == "default"
    assert side["chosen"].startswith("API RP 2A's alpha where the clay is "
                                     "highly overconsolidated")
    assert "Table D-100" in side["basis"] and "p. 247" in side["basis"]
    assert any("'chart_end'" in a for a in side["alternatives"])
    assert any("'api'" in a for a in side["alternatives"])
    oc = c["highly overconsolidated clay (alpha method)"]
    assert oc["how"] == "given"
    assert oc["chosen"].startswith(f"su / sigma'vo at or above {line:g}: "
                                   f"0-15 m, su / sigma'vo "
                                   f"{lb['su_over_sigma_v']:.2f}: ")
    assert ("highly overconsolidated, API alpha" in oc["chosen"]) is high
    assert "engineer's judgment" in oc["basis"]
    assert "does not define" in oc["basis"]      # Eqs. 7-13/7-14's 1.0
    assert any("engineer's judgment" in w and "GEC-12 gives no value" in w
               for w in r.warnings)


def test_chart_end_or_api_for_every_layer_needs_no_line():
    pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
    end = AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15,
                            adhesion_beyond_chart="chart_end").compute()
    c = _by_what(end)["su past Fig. 7-17 (alpha method)"]
    assert c["how"] == "given"
    assert c["chosen"] == "Ca held at the curve's end in every such layer"
    assert "highly overconsolidated clay (alpha method)" not in _by_what(end)
    api = AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15,
                            adhesion_beyond_chart="api").compute()
    c = _by_what(api)["su past Fig. 7-17 (alpha method)"]
    assert c["how"] == "given" and "API" in c["chosen"]
    assert api.Q_skin > end.Q_skin
    # the line given beside them changes nothing
    again = AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15,
                              adhesion_beyond_chart="api",
                              highly_overconsolidated_su_ratio=5.0).compute()
    assert again.Q_skin == api.Q_skin
    with pytest.raises(ValueError, match="highly_overconsolidated_su_ratio"):
        AxialPileAnalysis(pile=pile, soil=_stiff(), pile_length=15,
                          highly_overconsolidated_su_ratio=0.0)


def test_ca_above_su_and_fig_7_18_are_named():
    """GM-229: Fig. 7-17 as drawn (alpha 1.106 at su = 1.0 ksf, D = 40b),
    stated in the choices; D from the ground surface (p. 246) stated with
    its page; Fig. 7-18 as an alternative."""
    clay = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=1.0 * KSF)],
        gwt_depth=1.0)
    pile = make_concrete_pile(0.3556)
    r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20).compute()
    c = _by_what(r)
    above = c["Ca above su (alpha method)"]
    assert above["how"] == "module"
    assert above["chosen"] == ("Fig. 7-17 as drawn, no limit on Ca; Ca above "
                               "su in 1 layer(s), alpha up to 1.11")
    assert "Steps 1 and 2 (p. 250)" in above["basis"]
    assert "API RP 2A's" in above["basis"] and "p. 252" in above["basis"]
    assert r.layer_breakdown[0]["alpha"] == pytest.approx(1.106)
    D = c["D for Fig. 7-17 (alpha method)"]
    assert D["chosen"].startswith("from the ground surface to the bottom of "
                                  "each clay layer, or to the toe if less")
    assert "p. 246" in D["basis"] and "p. 248" in D["basis"]
    # Fig. 7-18 and the smaller of the two are offered (GM-144)
    chart = c["adhesion chart (alpha-method)"]
    assert chart["how"] == "default" and "smaller value" in chart["basis"]
    assert any("'fig_7_18'" in a for a in chart["alternatives"])
    assert any("'smaller'" in a for a in chart["alternatives"])


def test_h_pile_takes_the_corrugated_curve_with_its_page():
    """GM-229 (OWN-63, no change): an H-pile on the corrugated pile curve
    times the box area, p. 247, stated in the choices."""
    clay = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=60)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=clay,
                          pile_length=15).compute()
    a = _by_what(r)["adhesion (alpha-method)"]
    assert a["chosen"].startswith("Fig. 7-17, the curves for concrete, "
                                  "timber, corrugated steel piles")
    assert "box perimeter" in a["chosen"] and a["how"] == "default"
    assert "corrugated pile curve times the box area (p. 247)" in a["basis"]


def test_h_pile_toe_area_and_uplift_are_named():
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesionless", 19.0, friction_angle=34)],
        gwt_depth=2.0)
    r = AxialPileAnalysis(pile=make_h_pile("HP12x74"), soil=soil,
                          pile_length=15, include_uplift=True).compute()
    c = _by_what(r)
    assert c["toe area (H-pile)"]["how"] == "default"
    assert c["toe area (H-pile)"]["chosen"].startswith("box area")
    up = c["uplift capacity"]
    assert up["how"] == "default" and up["chosen"].startswith("1 x")
    assert "Eq. 7-38" in up["basis"] and "0.7" in up["alternatives"][0]


def test_beta_cohesive_toe_nt_is_named():
    clay = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 18.0, cohesion=80)], gwt_depth=1)
    pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                          method="beta").compute()
    nt = _by_what(r)["Nt of the cohesive toe layer (beta method)"]
    assert nt["chosen"] == "3" and nt["how"] == "default"
    r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20,
                          method="beta", cohesive_Nt=10).compute()
    assert _by_what(r)["Nt of the cohesive toe layer (beta method)"][
        "how"] == "given"


def test_numbers_unchanged_by_the_choices():
    a = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=True),
                          soil=_soil(), pile_length=15)
    d = a.compute().to_dict()
    d.pop("choices")
    # a 0.6 m pile is wider than GEC-12's 18 in correlation database
    assert set(d) == {"Q_ultimate_kN", "Q_skin_kN", "Q_tip_kN",
                      "Q_allowable_kN", "factor_of_safety", "pile_length_m",
                      "pile_name", "method", "sigma_v_tip_kPa",
                      "toe_details", "warnings"}
