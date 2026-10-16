"""What an axial pile result says it chose (backlog GM-115): method "auto"
resolved by soil type, the toe method, an open pipe's governing case."""

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_pipe_pile,
)


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
    assert c["Nordlund delta/phi"]["chosen"] == "0.75 (steel pile)"
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
    assert len(pipe["alternatives"]) == 1
    assert "Sec. 7.10.7" in pipe["basis"]


def test_toe_stress_above_3_ksf_is_named():
    # GEC-12 Sec. 7.2.1.3.1 limits sigma'p to 3 ksf; the module does not.
    deep = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesionless", 19.0, friction_angle=34)],
        gwt_depth=30.0)
    r = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=True),
                          soil=deep, pile_length=20).compute()
    toe = _by_what(r)["sigma'v at the toe"]
    assert r.sigma_v_tip > 143.6
    assert toe["chosen"] == f"{r.sigma_v_tip:.1f} kPa, not limited"
    shallow = AxialPileAnalysis(
        pile=make_pipe_pile(0.6, 0.012, closed_end=True), soil=deep,
        pile_length=5).compute()
    assert "sigma'v at the toe" not in _by_what(shallow)


def test_numbers_unchanged_by_the_choices():
    a = AxialPileAnalysis(pile=make_pipe_pile(0.6, 0.012, closed_end=True),
                          soil=_soil(), pile_length=15)
    d = a.compute().to_dict()
    d.pop("choices")
    assert set(d) == {"Q_ultimate_kN", "Q_skin_kN", "Q_tip_kN",
                      "Q_allowable_kN", "factor_of_safety", "pile_length_m",
                      "pile_name", "method", "sigma_v_tip_kPa"}
