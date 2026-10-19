"""BACKLOG GM-144: the GEC-12 chapter 7 parts GM-121 left, each an option
with the old behaviour as its default: Fig. 7-18's stratigraphy cases and
the shrink-swell zone (alpha-method), Table 7-9's beta and Nt by soil type
(beta method), the toe phi averaged from the toe to 3b below (Nordlund Step
7c), and Eq. 7-90's "predominantly cohesive" stated as the reading used.
Page numbers are the printed GEC-12 Vol 1 (FHWA-NHI-16-009)."""

import pytest

from axial_pile import (
    AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_concrete_pile,
    make_h_pile, make_pipe_pile,
)
from axial_pile.beta_method import TABLE_7_9, table_7_9
from axial_pile.tomlinson import (
    SAND_OVER_STIFF_CLAY, SOFT_OVER_STIFF_CLAY, STIFF_CLAY,
    adhesion_fig_7_18, alpha_fig_7_18,
)

KSF = 47.880259


def _by_what(result):
    return {c["what"]: c for c in result.choices}


# ---------------------------------------------------------------- Fig. 7-18

@pytest.mark.parametrize("case, su, D_over_b, alpha", [
    # top graph (p. 249): D < 10b is alpha = 1 everywhere; the D = 20b and
    # D > 40b curves run flat at 0.75 and about 0.39 past su 3 and 3.3 ksf
    (SAND_OVER_STIFF_CLAY, 3.0, 5.0, 1.0),
    (SAND_OVER_STIFF_CLAY, 4.0, 20.0, 0.75),
    (SAND_OVER_STIFF_CLAY, 1.0, 20.0, 1.0),
    (SAND_OVER_STIFF_CLAY, 4.0, 40.0, 0.39),
    (SAND_OVER_STIFF_CLAY, 2.6, 60.0, 0.50),
    # middle graph: D > 20b about 0.74 at su 1 ksf, 0.62 at 3; D = 10b
    # about 0.40 at 1 ksf, 0.25 at 2.9 ksf
    (SOFT_OVER_STIFF_CLAY, 1.0, 20.0, 0.74),
    (SOFT_OVER_STIFF_CLAY, 3.0, 25.0, 0.62),
    (SOFT_OVER_STIFF_CLAY, 1.0, 10.0, 0.40),
    (SOFT_OVER_STIFF_CLAY, 2.9, 10.0, 0.25),
    # bottom graph: D > 40b 1.0 to about 1.55 ksf, 0.50 at 3 ksf, flat 0.30
    # past 4.2; D = 10b 0.69 at 2 ksf, flat 0.24 past 3.6
    (STIFF_CLAY, 1.4, 40.0, 1.0),
    (STIFF_CLAY, 3.0, 40.0, 0.50),
    (STIFF_CLAY, 4.6, 40.0, 0.30),
    (STIFF_CLAY, 2.0, 10.0, 0.69),
    (STIFF_CLAY, 4.0, 10.0, 0.24),
])
def test_fig_7_18_reads_the_printed_curves(case, su, D_over_b, alpha):
    assert alpha_fig_7_18(su * KSF, D_over_b, case, notes=[]) == \
        pytest.approx(alpha, abs=0.012)


def test_fig_7_18_is_linear_in_D_over_b_between_the_curves():
    a20 = alpha_fig_7_18(2.4 * KSF, 20, SAND_OVER_STIFF_CLAY, [])
    a40 = alpha_fig_7_18(2.4 * KSF, 40, SAND_OVER_STIFF_CLAY, [])
    a30 = alpha_fig_7_18(2.4 * KSF, 30, SAND_OVER_STIFF_CLAY, [])
    assert a30 == pytest.approx((a20 + a40) / 2)
    # between D < 10b (alpha 1) and D = 20b
    a15 = alpha_fig_7_18(2.4 * KSF, 15, SAND_OVER_STIFF_CLAY, [])
    assert a15 == pytest.approx((1.0 + a20) / 2)


def test_fig_7_18_says_where_it_holds_an_edge():
    notes = []
    alpha_fig_7_18(0.3 * KSF, 15, SOFT_OVER_STIFF_CLAY, notes)
    assert any("below GEC-12 Figure 7-18" in n for n in notes)
    notes = []
    alpha_fig_7_18(2.0 * KSF, 5, STIFF_CLAY, notes)
    assert any("shallowest curve" in n and "overstated" in n for n in notes)
    notes = []
    ca, source = adhesion_fig_7_18(6.0 * KSF, 40, STIFF_CLAY, notes=notes)
    # past the chart Ca is held at its end, as Fig. 7-17's chart_end is
    assert source == "Fig. 7-18 end"
    assert ca == pytest.approx(0.30 * 5.0 * KSF, rel=0.01)
    assert any("beyond GEC-12 Figure 7-18" in n for n in notes)
    ca, source = adhesion_fig_7_18(6.0 * KSF, 40, STIFF_CLAY,
                                   sigma_v=4.0 * KSF, beyond_chart="api",
                                   notes=[])
    assert source == "API alpha"
    assert ca == pytest.approx(0.5 * 1.5 ** -0.25 * 6.0 * KSF)


def _sand_over_clay(**clay):
    return AxialSoilProfile(layers=[
        AxialSoilLayer(5, "cohesionless", 18.0, friction_angle=32),
        AxialSoilLayer(20, "cohesive", 19.0, cohesion=2.0 * KSF, **clay)],
        gwt_depth=2.0)


def test_fig_7_17_stays_the_default_and_its_numbers_do_not_move():
    pile = make_concrete_pile(0.4)
    r0 = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(),
                           pile_length=15).compute()
    r1 = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(),
                           pile_length=15, adhesion_chart="fig_7_17").compute()
    assert r0.Q_ultimate == r1.Q_ultimate
    clay = r0.layer_breakdown[1]
    assert clay["adhesion_chart"] == "Fig. 7-17"
    assert "fig_7_18_case" not in clay
    chart = _by_what(r0)["adhesion chart (alpha-method)"]
    assert chart["how"] == "default" and "Fig. 7-17" in chart["chosen"]
    assert "Fig. 7-18 stratigraphy case" not in _by_what(r0)


def test_fig_7_18_takes_the_case_and_the_penetration_from_the_profile():
    """Clay under sand: the top graph, D from the top of the clay (5 m) to
    the toe (15 m) = 10 m = 25b; Ca = alpha su (p. 249)."""
    pile = make_concrete_pile(0.4)
    r = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(), pile_length=15,
                          adhesion_chart="fig_7_18").compute()
    clay = r.layer_breakdown[1]
    assert clay["fig_7_18_case"] == SAND_OVER_STIFF_CLAY
    assert clay["D_over_b_fig_7_18"] == pytest.approx(25.0)
    alpha = alpha_fig_7_18(2.0 * KSF, 25.0, SAND_OVER_STIFF_CLAY, [])
    assert clay["alpha"] == pytest.approx(alpha)
    assert clay["skin_friction_kN"] == pytest.approx(
        alpha * 2.0 * KSF * pile.perimeter * 10.0)
    c = _by_what(r)
    assert c["adhesion chart (alpha-method)"]["how"] == "given"
    case = c["Fig. 7-18 stratigraphy case"]
    assert case["how"] == "module" and "top graph" in case["chosen"]
    assert "D = 25.0b" in case["chosen"]
    # Fig. 7-17 is not read, so its curve choice is not reported
    assert "adhesion (alpha-method)" not in c


def test_clay_from_the_ground_surface_takes_the_bottom_graph():
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(10, "cohesive", 18.0, cohesion=60),
        AxialSoilLayer(20, "cohesive", 19.0, cohesion=150)], gwt_depth=1.0)
    pile = make_pipe_pile(0.4, 0.0127, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                          adhesion_chart="fig_7_18").compute()
    upper, lower = r.layer_breakdown
    # one clay stratum from the ground surface: D runs from there
    assert upper["fig_7_18_case"] == lower["fig_7_18_case"] == STIFF_CLAY
    assert upper["D_over_b_fig_7_18"] == pytest.approx(10 / 0.4)
    assert lower["D_over_b_fig_7_18"] == pytest.approx(20 / 0.4)


def test_a_layers_case_wins_and_starts_its_own_stratum():
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(6, "cohesive", 17.0, cohesion=20),
        AxialSoilLayer(20, "cohesive", 19.0, cohesion=120,
                       fig_7_18_case=SOFT_OVER_STIFF_CLAY)], gwt_depth=1.0)
    pile = make_concrete_pile(0.4)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=16,
                          adhesion_chart="fig_7_18").compute()
    soft, stiff = r.layer_breakdown
    assert soft["fig_7_18_case"] == STIFF_CLAY
    assert stiff["fig_7_18_case"] == SOFT_OVER_STIFF_CLAY
    # D from the top of the stiff clay (6 m) to the toe (16 m)
    assert stiff["D_over_b_fig_7_18"] == pytest.approx(25.0)
    assert _by_what(r)["Fig. 7-18 stratigraphy case"]["how"] == "given"
    # "none": Fig. 7-17 alone for that layer
    soil.layers[1].fig_7_18_case = "none"
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=16,
                          adhesion_chart="fig_7_18").compute()
    assert r.layer_breakdown[1]["adhesion_chart"] == "Fig. 7-17"


def test_smaller_takes_the_lesser_ca_of_the_two_figures():
    """p. 246: 'the inexperienced user should select and use the smaller
    value obtained from either figure'."""
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(30, "cohesive", 19.0, cohesion=2.5 * KSF)],
        gwt_depth=1.0)
    pile = make_concrete_pile(0.4)
    runs = {chart: AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                                     adhesion_chart=chart).compute()
            for chart in ("fig_7_17", "fig_7_18", "smaller")}
    q = {k: r.Q_skin for k, r in runs.items()}
    assert q["smaller"] == pytest.approx(min(q["fig_7_17"], q["fig_7_18"]))
    assert q["fig_7_18"] < q["fig_7_17"]   # su 2.5 ksf, bottom graph, 50b
    clay = runs["smaller"].layer_breakdown[0]
    assert clay["Ca_fig_7_18_kPa"] < clay["Ca_fig_7_17_kPa"]
    assert clay["adhesion_chart"] == "Fig. 7-18"
    assert "Ca of the two charts (alpha-method)" in _by_what(runs["smaller"])


def test_fig_7_18_case_is_for_clay_only():
    with pytest.raises(ValueError, match="cohesive layer"):
        AxialSoilLayer(5, "cohesionless", 18.0, friction_angle=30,
                       fig_7_18_case=STIFF_CLAY)
    with pytest.raises(ValueError, match="fig_7_18_case must be one of"):
        AxialSoilLayer(5, "cohesive", 18.0, cohesion=50,
                       fig_7_18_case="stiff")
    with pytest.raises(ValueError, match="adhesion_chart"):
        AxialPileAnalysis(pile=make_concrete_pile(0.4),
                          soil=_sand_over_clay(), pile_length=10,
                          adhesion_chart="tomlinson_1980")


# ------------------------------------------------------- shrink-swell zone

def test_the_shrink_swell_zone_gives_no_shaft_resistance_in_clay():
    """p. 247: 'static resistance calculations should ignore the shaft
    resistance from the adhesion in the shrink-swell zone'."""
    pile = make_concrete_pile(0.4)
    base = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(),
                             pile_length=15).compute()
    r = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(), pile_length=15,
                          shrink_swell_depth=7.0).compute()
    sand0, clay0 = base.layer_breakdown
    sand, clay = r.layer_breakdown
    # the sand above is untouched; 2 of the clay's 10 m are left out
    assert sand["skin_friction_kN"] == sand0["skin_friction_kN"]
    assert clay["shrink_swell_m"] == pytest.approx(2.0)
    assert clay["skin_friction_kN"] == pytest.approx(
        clay0["skin_friction_kN"] * 8.0 / 10.0)
    assert clay["alpha"] == pytest.approx(clay0["alpha"])
    c = _by_what(r)["shrink-swell zone (cohesive soil)"]
    assert c["how"] == "given" and "p. 247" in c["basis"]
    assert "shrink-swell zone (cohesive soil)" not in _by_what(base)
    beta = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(),
                             pile_length=15, method="beta",
                             shrink_swell_depth=7.0).compute()
    beta0 = AxialPileAnalysis(pile=pile, soil=_sand_over_clay(),
                              pile_length=15, method="beta").compute()
    assert beta.Q_skin < beta0.Q_skin


# ------------------------------------------------------------- Table 7-9

@pytest.mark.parametrize("soil", list(TABLE_7_9))
def test_table_7_9_runs_across_each_soil_types_range(soil):
    (p0, p1), (b0, b1), (n0, n1) = TABLE_7_9[soil]
    assert table_7_9(soil, p0, []) == pytest.approx((b0, n0))
    assert table_7_9(soil, p1, []) == pytest.approx((b1, n1))
    assert table_7_9(soil, (p0 + p1) / 2, []) == pytest.approx(
        ((b0 + b1) / 2, (n0 + n1) / 2))
    notes = []
    assert table_7_9(soil, p1 + 5, notes) == pytest.approx((b1, n1))
    assert "outside GEC-12 Table 7-9" in notes[0]


def test_table_7_9_is_printed_as_gec_12_gives_it():
    """GEC-12 Table 7-9 (after Fellenius 2014), p. 256."""
    assert TABLE_7_9 == {
        "clay": ((25, 30), (0.15, 0.35), (3, 30)),
        "silt": ((28, 34), (0.25, 0.50), (20, 40)),
        "sand": ((32, 40), (0.30, 0.90), (30, 150)),
        "gravel": ((35, 45), (0.35, 0.80), (60, 300)),
    }


def _beta_soil():
    return AxialSoilProfile(layers=[
        AxialSoilLayer(10, "cohesive", 18.0, cohesion=60, friction_angle=28),
        AxialSoilLayer(10, "cohesionless", 19.0, friction_angle=36),
        AxialSoilLayer(10, "cohesionless", 20.0, friction_angle=40,
                       soil_class="gravel")], gwt_depth=2.0)


def test_beta_from_table_7_9_by_soil_type():
    pile = make_pipe_pile(0.6, 0.012, closed_end=True)
    base = AxialPileAnalysis(pile=pile, soil=_beta_soil(), pile_length=25,
                             method="beta").compute()
    r = AxialPileAnalysis(pile=pile, soil=_beta_soil(), pile_length=25,
                          method="beta",
                          beta_coefficients="table_7_9").compute()
    clay, sand, gravel = r.layer_breakdown
    assert clay["beta"] == pytest.approx(0.15 + 0.6 * 0.20)    # phi' 28
    assert sand["beta"] == pytest.approx(0.30 + 0.5 * 0.60)    # phi' 36
    assert gravel["beta"] == pytest.approx(0.35 + 0.5 * 0.45)  # phi' 40
    assert gravel["beta_source"] == "Table 7-9 (gravel)"
    # the toe in gravel at phi' 40: Nt 60 + 0.5 x 240
    assert r.toe_details["N_t"] == pytest.approx(180.0)
    c = _by_what(r)["beta and Nt (beta method)"]
    assert c["how"] == "given" and "Steps 2b and 5b" in c["basis"]
    assert "gravel beta 0.35-0.8" in c["chosen"]
    d = _by_what(base)["beta and Nt (beta method)"]
    assert d["how"] == "default" and "Fellenius 1991" in d["chosen"]
    assert base.layer_breakdown[1]["beta_source"] == "from phi'"


def test_a_clay_toe_keeps_cohesive_nt_and_a_silt_toe_takes_the_table():
    pile = make_pipe_pile(0.6, 0.012, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=_beta_soil(), pile_length=8,
                          method="beta",
                          beta_coefficients="table_7_9").compute()
    assert r.toe_details["N_t"] == 3.0
    assert r.toe_details["N_t_source"] == "cohesive_Nt"
    silt = AxialSoilProfile(layers=[
        AxialSoilLayer(20, "cohesive", 18.0, cohesion=40, friction_angle=31,
                       soil_class="silt")], gwt_depth=2.0)
    r = AxialPileAnalysis(pile=pile, soil=silt, pile_length=10,
                          method="beta",
                          beta_coefficients="table_7_9").compute()
    assert r.toe_details["N_t"] == pytest.approx(30.0)
    assert r.layer_breakdown[0]["beta"] == pytest.approx(0.375)
    # a layer's own beta and Nt still win (local experience, Steps 2a/5a)
    silt.layers[0].beta, silt.layers[0].Nt = 0.2, 12.0
    r = AxialPileAnalysis(pile=pile, soil=silt, pile_length=10,
                          method="beta",
                          beta_coefficients="table_7_9").compute()
    assert r.layer_breakdown[0]["beta"] == 0.2
    assert r.toe_details["N_t"] == 12.0
    assert "beta and Nt (beta method)" not in _by_what(r)


def test_soil_class_must_suit_the_soil_type():
    with pytest.raises(ValueError, match="Table 7-9"):
        AxialSoilLayer(5, "cohesive", 18.0, cohesion=50, soil_class="sand")
    with pytest.raises(ValueError, match="Table 7-9"):
        AxialSoilLayer(5, "cohesionless", 18.0, friction_angle=30,
                       soil_class="clay")
    assert AxialSoilLayer(5, "cohesionless", 18.0, friction_angle=30,
                          soil_class="Silt").table_7_9_soil == "silt"


# ------------------------------------------------------- Step 7c toe zone

def _toe_soil():
    return AxialSoilProfile(layers=[
        AxialSoilLayer(14.0, "cohesionless", 19.0, friction_angle=34),
        AxialSoilLayer(0.8, "cohesionless", 19.0, friction_angle=30),
        AxialSoilLayer(10.0, "cohesionless", 20.0, friction_angle=36,
                       toe_friction_angle=40)], gwt_depth=2.0)


def test_the_toe_phi_is_averaged_from_the_toe_to_3b_below():
    """Step 7c (p. 244): the zone from the pile toe to 3 diameters below."""
    pile = make_pipe_pile(0.6, 0.012, closed_end=True)   # 3b = 1.8 m
    base = AxialPileAnalysis(pile=pile, soil=_toe_soil(),
                             pile_length=13.6).compute()
    r = AxialPileAnalysis(pile=pile, soil=_toe_soil(), pile_length=13.6,
                          toe_phi_zone="average_3b").compute()
    # 0.4 m at 34, 0.8 m at 30, 0.6 m at the gravel's toe phi 40
    phi = (0.4 * 34 + 0.8 * 30 + 0.6 * 40) / 1.8
    assert r.toe_details["toe_phi_deg"] == pytest.approx(phi)
    assert base.toe_details["toe_phi_deg"] == 34
    # the same toe as the toe layer given that phi
    soil = _toe_soil()
    soil.layers[0].toe_friction_angle = phi
    same = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.6).compute()
    assert r.Q_tip == pytest.approx(same.Q_tip) and r.Q_tip != base.Q_tip
    c = _by_what(r)["toe friction angle (Nordlund)"]
    assert c["how"] == "given" and "Step 7c" in c["basis"]
    assert "averaged from the toe to 3b below" in c["chosen"]
    d = _by_what(base)["toe friction angle (Nordlund)"]
    assert d["how"] == "default"
    assert d["chosen"] == "34 deg, the toe layer's friction_angle"


def test_the_toe_zone_leaves_clay_out_and_says_so():
    soil = AxialSoilProfile(layers=[
        AxialSoilLayer(10.0, "cohesionless", 19.0, friction_angle=36),
        AxialSoilLayer(5.0, "cohesive", 18.0, cohesion=40)], gwt_depth=2.0)
    pile = make_pipe_pile(0.6, 0.012, closed_end=True)
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=9.4,
                          toe_phi_zone="average_3b").compute()
    assert r.toe_details["toe_phi_deg"] == 36
    assert any("takes in cohesive soil" in w for w in r.warnings)
    # past the profile's bottom the last layer goes on
    r = AxialPileAnalysis(pile=pile, soil=_toe_soil(), pile_length=24.5,
                          toe_phi_zone="average_3b").compute()
    assert r.toe_details["toe_phi_deg"] == 40


# ------------------------------------------- Eq. 7-90, "predominantly"

def test_the_predominantly_cohesive_reading_is_stated():
    """p. 407 gives Eq. 7-90 for open pipes under 24 in 'in predominantly
    cohesive soils'; the module reads it as more than half the length."""
    pile = make_pipe_pile(0.4, 0.0127, closed_end=False)
    mostly_clay = AxialSoilProfile(layers=[
        AxialSoilLayer(3, "cohesionless", 18.0, friction_angle=30),
        AxialSoilLayer(20, "cohesive", 18.0, cohesion=60)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=pile, soil=mostly_clay,
                          pile_length=12).compute()
    c = _by_what(r)["predominantly cohesive soil (Eq. 7-90)"]
    assert c["chosen"].startswith("yes: 9.00 of 12.00 m")
    assert c["how"] == "module" and "more than half" in c["basis"]
    assert _by_what(r)["open-ended pipe"]["chosen"] == "Tomlinson's Eq. 7-90"
    mostly_sand = AxialSoilProfile(layers=[
        AxialSoilLayer(8, "cohesionless", 18.0, friction_angle=30),
        AxialSoilLayer(20, "cohesive", 18.0, cohesion=60)], gwt_depth=1.0)
    r = AxialPileAnalysis(pile=pile, soil=mostly_sand,
                          pile_length=12).compute()
    c = _by_what(r)["predominantly cohesive soil (Eq. 7-90)"]
    assert c["chosen"].startswith("no: 4.00 of 12.00 m")
    assert _by_what(r)["open-ended pipe"]["chosen"] != "Tomlinson's Eq. 7-90"
    # forced, or 24 in and over: no reading to state
    r = AxialPileAnalysis(pile=pile, soil=mostly_clay, pile_length=12,
                          open_pipe_method="tomlinson").compute()
    assert "predominantly cohesive soil (Eq. 7-90)" not in _by_what(r)


@pytest.mark.parametrize("chart", ["fig_7_18", "smaller"])
def test_eq_7_90_takes_ca_from_fig_7_17_whatever_the_chart(chart):
    """p. 407 prints Eq. 7-90 with Ca from Fig. 7-17: Rn = 0.8 Ca As +
    4.5 su Ap. The side-resistance chart does not reach it."""
    pile = make_pipe_pile(0.3556, 0.0095, closed_end=False)
    soil = _sand_over_clay()
    r17 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20).compute()
    r = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                          adhesion_chart=chart).compute()
    assert r.toe_details["governs"] == "Eq. 7-90"
    assert (r.Q_skin, r.Q_tip, r.Q_ultimate) == \
        (r17.Q_skin, r17.Q_tip, r17.Q_ultimate)
    assert r.layer_breakdown[1]["adhesion_chart"] == "Fig. 7-17"
    c = _by_what(r)
    assert ("Ca from Fig. 7-17 as Eq. 7-90 prints it (p. 407), whatever "
            "adhesion chart is chosen for the side resistance"
            in c["open-ended pipe"]["basis"])
    assert "Fig. 7-18 stratigraphy case" not in c
    # Eqs. 7-88/7-89 keep the chosen chart: Fig. 7-18 moves them
    pw = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                           adhesion_chart="fig_7_18",
                           open_pipe_method="paikowsky_whitman").compute()
    pw17 = AxialPileAnalysis(pile=pile, soil=soil, pile_length=20,
                             open_pipe_method="paikowsky_whitman").compute()
    assert pw.Q_skin != pw17.Q_skin


# --------------------------------------------- the defaults do not move

def test_every_new_option_at_its_default_is_the_old_run():
    defaults = {"adhesion_chart": "fig_7_17", "shrink_swell_depth": None,
                "beta_coefficients": "from_phi", "toe_phi_zone": "toe_layer"}
    for pile in (make_h_pile("HP12x74"), make_concrete_pile(0.4),
                 make_pipe_pile(0.4, 0.0127, closed_end=False)):
        for method in ("auto", "beta"):
            for soil in (_sand_over_clay(), _beta_soil(), _toe_soil()):
                a = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.6,
                                      method=method).compute()
                b = AxialPileAnalysis(pile=pile, soil=soil, pile_length=13.6,
                                      method=method, **defaults).compute()
                assert (a.Q_skin, a.Q_tip) == (b.Q_skin, b.Q_tip)
