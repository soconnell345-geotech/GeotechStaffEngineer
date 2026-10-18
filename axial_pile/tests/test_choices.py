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
    toe = _by_what(r)["sigma'p at the toe"]
    assert r.sigma_v_tip > 3 * KSF
    assert toe["chosen"].startswith(f"{3 * KSF:.1f} kPa (limit")
    assert toe["how"] == "default" and "Step 8" in toe["basis"]
    off = AxialPileAnalysis(pile=pile, soil=deep, pile_length=20,
                            toe_stress_limit=None).compute()
    toe = _by_what(off)["sigma'p at the toe"]
    assert toe["chosen"].startswith("not limited") and toe["how"] == "given"
    beta = AxialPileAnalysis(pile=pile, soil=deep, pile_length=20,
                             method="beta").compute()
    assert "sigma'p at the toe" not in _by_what(beta)


def test_adhesion_beyond_the_chart_is_named():
    stiff = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=4.0 * KSF)],
        gwt_depth=1.0)
    pile = make_pipe_pile(0.4064, 0.0127, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=stiff, pile_length=15).compute()
    c = _by_what(r)["su beyond Fig. 7-17"]
    assert c["how"] == "default" and c["chosen"] == "Ca held at the curve's end"
    assert "Table D-100" in c["basis"] and "p. 247" in c["basis"]
    assert "API" in c["alternatives"][0]
    api = AxialPileAnalysis(pile=pile, soil=stiff, pile_length=15,
                            adhesion_beyond_chart="api").compute()
    c = _by_what(api)["su beyond Fig. 7-17"]
    assert c["how"] == "given" and "API" in c["chosen"]
    assert api.Q_skin > r.Q_skin
    soft = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=60)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=pile, soil=soft, pile_length=15).compute()
    assert "su beyond Fig. 7-17" not in _by_what(r)


def test_adhesion_limit_and_fig_7_18_are_named():
    """Ca at most su (alpha <= 1) and Fig. 7-18 as an alternative."""
    clay = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=1.0 * KSF)],
        gwt_depth=1.0)
    pile = make_concrete_pile(0.3556)
    r = AxialPileAnalysis(pile=pile, soil=clay, pile_length=20).compute()
    c = _by_what(r)
    lim = c["adhesion limit (alpha-method)"]
    assert lim["how"] == "module" and "governed in 1 layer" in lim["chosen"]
    assert "p. 252" in lim["basis"] and "uncapped" in lim["alternatives"][0]
    assert r.layer_breakdown[0]["alpha"] == pytest.approx(1.0)
    assert any("Fig. 7-18" in a and "smaller value" in a
               for a in c["adhesion (alpha-method)"]["alternatives"])


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
