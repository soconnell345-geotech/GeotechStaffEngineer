"""GM-244: GEC-10 (FHWA-NHI-18-024, 2018) as printed is the default for
drilled shafts; the older methods stay as named options with their
sources, and the result's choices state each with its alternatives' values.

The pages relied on:

* Sec. 10.3.5.2, cohesive soil: Eq. 10-17 (UU to CIUC) and Eq. 10-18,
  alpha = 0.30 + 0.17 / (su(CIUC)/pa), p. 10-16, 'recommended as a revision
  to the expressions for alpha currently given in the AASHTO LRFD Bridge
  Design Specifications'; Example 10-2 (p. 10-18): UU 2,000 psf, sigma'vo
  3,697 psf -> su(CIUC) 2,571 psf, alpha 0.44, f_SN 1,131 psf, R_SN =
  pi (5 ft)(27 ft) 1,131 psf = 479.7 kips over the whole of Layer 3 to the
  base at 60 ft (no bottom zone).
* p. 10-19: q_BN = N*c su (Eq. 10-19), Table 10-2 (6.5 at 500 psf, 8.0 at
  1,000 psf, 9.0 at 2,000 psf or more, linear between), Eq. 10-20 for a
  depth under 3B; Example 10-3 (p. 10-20): 9.0 x 2,571 psf = 23.1 ksf,
  R_BN = 453.6 kips.
* p. 10-22: q_BN = N*cr qu, N*cr = 2.5 (Eq. 10-23); Example 10-4 (pp.
  10-27, 10-28): RQD 37 to 100, 2.5 x 360 ksf = 900 ksf, 25,447 kips.
* Table 8-4 (p. 8-14): resistance factors, rock side 0.50.
* FHWA NHI-06-089 p. 9-133: 'The limiting value of fsi for shafts in
  cohesionless soils is 4 ksf (191 kPa)'.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps
from drilled_shaft.end_bearing import (
    end_bearing_rock, nc_below_table_10_2, nc_table_10_2,
    unit_end_bearing_cohesionless, unit_end_bearing_cohesive)
from drilled_shaft.lrfd import RESISTANCE_FACTORS
from drilled_shaft.side_resistance import (
    DEPTH_BETA_FS_MAX, side_resistance_cohesionless)

FT, PCF, PSF, KIP = 0.3048, 0.157087, 0.0478803, 4.44822


def _choice(r, what):
    (c,) = [c for c in r.choices if c["what"] == what]
    return c


def _figure_10_5(**options):
    """Examples 10-1 to 10-3: two sand layers to 33 ft over the stiff clay
    of Layer 3 (UU 2,000 psf), water at 15 ft, a 5 ft shaft 60 ft deep."""
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(15 * FT, "cohesionless", 115 * PCF, phi=38, N60=9,
                       sigma_p_correlation="silty_sand"),
        ShaftSoilLayer(18 * FT, "cohesionless", 125 * PCF, phi=41, N60=27,
                       sigma_p_correlation="silty_sand"),
        ShaftSoilLayer(40 * FT, "cohesive", 125 * PCF, cu=2000 * PSF,
                       description="Layer 3")], gwt_depth=15 * FT)
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=5 * FT,
                                               length=60 * FT),
                              soil=soil, su_test_type="uu", **options)


def test_examples_10_2_and_10_3_as_printed():
    r = _figure_10_5().compute()
    (clay,) = [lb for lb in r.layer_breakdown
               if lb["soil_type"] == "cohesive"]
    st = clay["alpha_steps"]
    assert st["su_used_kPa"] / PSF == pytest.approx(2571, abs=1)
    assert st["alpha"] == pytest.approx(0.44, abs=0.005)
    assert clay["fs_kPa"] / PSF == pytest.approx(1131, rel=0.002)
    # over the whole 27 ft of Layer 3, to the base
    assert clay["effective_bottom_m"] == pytest.approx(60 * FT, abs=0.01)
    assert clay["side_resistance_kN"] / KIP == pytest.approx(479.7,
                                                             rel=0.002)
    # Example 10-3: N*c 9.0 on the su(CIUC) of Example 10-2
    b = r.base_steps
    assert b["Nc"] == 9.0 and b["equation"] == "10-19"
    assert b["su_from"] == "su_ciuc"
    assert b["qb"] / PSF / 1000 == pytest.approx(23.1, abs=0.05)
    assert r.Q_tip / KIP == pytest.approx(453.6, rel=0.003)


def test_the_clay_choices_give_the_alternatives_values():
    r = _figure_10_5().compute()
    side = _choice(r, "side resistance in cohesive soil")
    assert side["how"] == "default" and "Eq. 10-18" in side["chosen"]
    assert "'recommended as a revision" in side["basis"]
    (aashto,) = side["alternatives"]
    assert aashto.startswith("alpha_method 'aashto'")
    assert "Q_s in cohesive soil" in aashto and "Q_ult" in aashto
    su = _choice(r, "strength su in Eq. 10-18 (the test each cu came from)")
    assert su["how"] == "given" and "UU" in su["chosen"]
    assert len(su["alternatives"]) == 2
    bottom = _choice(r, "side resistance near the base in cohesive soil")
    assert bottom["how"] == "default"
    assert bottom["chosen"].startswith("taken to the base")
    assert "Example 10-2" in bottom["basis"] and "Fig. 9-49" in \
        bottom["basis"]
    # the bottom diameter here is also NHI's bottom 5 ft: one alternative
    (one_d,) = bottom["alternatives"]
    assert one_d.startswith("bottom_exclusion 1.524")
    base = _choice(r, "base resistance in cohesive soil")
    assert "Eq. 10-19" in base["chosen"] and "N*c = 9.00" in base["chosen"]
    assert any(a.startswith("clay_base_method 'oneill_reese'")
               for a in base["alternatives"])
    assert any(a.startswith("su = cu as given") for a in base["alternatives"])


def test_the_older_clay_basis_is_the_named_options():
    """The pre-GM-244 clay: AASHTO alpha on cu, the bottom one diameter
    left out, O'Neill & Reese's Nc."""
    old = _figure_10_5(alpha_method="aashto", bottom_exclusion=5 * FT,
                       clay_base_method="oneill_reese").compute()
    (clay,) = [lb for lb in old.layer_breakdown
               if lb["soil_type"] == "cohesive"]
    assert clay["alpha_steps"]["alpha"] == pytest.approx(0.55)
    assert clay["effective_bottom_m"] == pytest.approx(55 * FT, abs=0.01)
    assert clay["side_resistance_kN"] == pytest.approx(
        0.55 * 2000 * PSF * math.pi * 5 * FT * 22 * FT, rel=1e-4)
    assert old.base_steps["su_from"] == "cu"
    assert old.Q_tip == pytest.approx(
        9.0 * 2000 * PSF * math.pi / 4 * (5 * FT) ** 2, rel=1e-6)


def test_table_10_2_as_printed():
    for psf, nc in ((500, 6.5), (1000, 8.0), (2000, 9.0), (4000, 9.0),
                    (750, 7.25), (1500, 8.5)):
        assert nc_table_10_2(psf * PSF) == pytest.approx(nc)
    # below the table: held at 6.5, the line through its first two rows
    # the alternative
    assert nc_table_10_2(300 * PSF) == 6.5
    assert nc_below_table_10_2(300 * PSF) == pytest.approx(5.9)
    u = unit_end_bearing_cohesive(300 * PSF, 10.0)
    assert u["below_table"] and u["qb"] == pytest.approx(6.5 * 300 * PSF)


def test_soft_clay_below_the_table_is_a_judgment_with_its_alternative():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(30.0, "cohesive", 17.0, cu=20.0)])
    r = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=12.0),
                           soil=soil).compute()
    base = _choice(r, "base resistance in cohesive soil")
    assert "N*c = 6.50" in base["chosen"]
    assert "the engineer's judgment: GEC-10 gives no value" in base["basis"]
    assert any(a.startswith("N*c = ") for a in base["alternatives"])
    # Fig. 10-6 runs su(CIUC)/pa from about 0.25: 20/101.3 is below it
    side = _choice(r, "side resistance in cohesive soil")
    assert "outside it" in side["basis"]


def test_eq_10_20_where_the_depth_is_under_3B():
    u = unit_end_bearing_cohesive(100.0, 1.5, base_diameter=2.0,
                                  shaft_length=3.0)
    assert u["equation"] == "10-20"
    assert u["depth_factor"] == pytest.approx(2 / 3 * (1 + 1.5 / 6))
    assert u["qb"] == pytest.approx(u["depth_factor"] * 9.0 * 100.0)
    deep = unit_end_bearing_cohesive(100.0, 1.5, base_diameter=2.0,
                                     shaft_length=6.0)
    assert deep["equation"] == "10-19" and deep["qb"] == 900.0


def test_example_10_4_base_whatever_the_rqd():
    """RQD 37 to 100 in Example 10-4, and q_BN = 2.5 x 360 ksf all the
    same; the RQD reduction is the 'rqd' option."""
    area = math.pi / 4 * (6 * FT) ** 2
    qu = 360 * 1000 * PSF
    for rqd in (37, 100):
        assert end_bearing_rock(qu, area, RQD=rqd) / KIP == pytest.approx(
            25447, abs=2)
    assert end_bearing_rock(qu, area, RQD=37, method="rqd") < \
        end_bearing_rock(qu, area, RQD=37)


def _rock_shaft(**options):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(10.0, "rock", 23.0, qu=20000.0, RQD=40.0)])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=6.0),
                              soil=soil, **options).compute()


def test_rock_from_the_top_and_its_base_choice():
    """Sec. 10.3.5.3 prints no top zone in rock (Example 10-4 takes the
    socket's whole length): rock carries side resistance from its top."""
    r = _rock_shaft()
    (rock,) = r.layer_breakdown
    assert rock["effective_top_m"] == 0.0
    top = _choice(r, "side resistance near the ground surface")
    assert "rock" in top["chosen"] and top["how"] == "default"
    base = _choice(r, "base resistance in rock")
    assert base["how"] == "default" and "2.5 qu" in base["chosen"]
    assert "RQD is given as 40 %" in base["basis"]
    assert base["alternatives"][0].startswith("rock_base_method 'rqd'")
    assert "Hoek-Brown" in base["alternatives"][1]
    rqd = _rock_shaft(rock_base_method="rqd")
    assert rqd.Q_tip < r.Q_tip
    assert _choice(rqd, "base resistance in rock")["how"] == "given"


def test_no_large_base_reduction_unless_asked():
    sand = ShaftSoilProfile(layers=[
        ShaftSoilLayer(30.0, "cohesionless", 19.0, phi=34.0, N60=30.0)])
    shaft = DrillShaft(diameter=2.0, length=15.0)
    r = DrillShaftAnalysis(shaft=shaft, soil=sand).compute()
    assert r.Q_tip == pytest.approx(57.5 * 30 * shaft.tip_area)
    c = _choice(r, "reduction of base resistance for a large base")
    assert c["how"] == "default" and c["chosen"].startswith("none")
    assert "large_base_reduction true" in c["alternatives"][0]
    opt = DrillShaftAnalysis(shaft=shaft, soil=sand,
                             large_base_reduction=True).compute()
    assert opt.Q_tip == pytest.approx(r.Q_tip * 1.27 / 2.0)
    assert unit_end_bearing_cohesionless(30, 2.0)["reduction"] is None


def test_the_depth_beta_limit_is_4_ksf():
    assert DEPTH_BETA_FS_MAX == pytest.approx(191.5, abs=0.05)
    assert side_resistance_cohesionless(1000.0, 1.0, 1.0, 1.0) == \
        pytest.approx(DEPTH_BETA_FS_MAX)


def test_a_given_phi_wins_over_n1_60():
    """phi' given is used; (N1)60 gives phi' only where none is."""
    def run(**layer):
        soil = ShaftSoilProfile(layers=[ShaftSoilLayer(
            30.0, "cohesionless", 19.0, N60=20.0, **layer)])
        return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0,
                                                   length=12.0),
                                  soil=soil).compute()
    both = run(phi=34.0, N1_60=30.0)
    (lb,) = both.layer_breakdown
    assert lb["beta_steps"]["phi_from"] == "phi"
    assert lb["beta_steps"]["phi_deg"] == 34.0
    c = _choice(both, "phi' of cohesionless soil")
    assert c["how"] == "given" and "34 deg" in c["chosen"]
    assert c["alternatives"][0].startswith("phi' from (N1)60")
    only = run(N1_60=30.0)
    (lb,) = only.layer_breakdown
    assert lb["beta_steps"]["phi_from"] == "N1_60"
    assert lb["beta_steps"]["phi_deg"] == pytest.approx(
        27.5 + 9.2 * math.log10(30.0), abs=1e-3)
    assert "where no phi' is given" in _choice(
        only, "phi' of cohesionless soil")["chosen"]
    # the alternative is the run that takes phi' from (N1)60
    assert f"Q_ult {only.Q_ultimate:,.0f} kN" in c["alternatives"][0]
    with pytest.raises(ValueError, match="phi"):
        ShaftSoilLayer(10.0, "cohesionless", 19.0, N60=20.0)


def test_resistance_factors_follow_table_8_4():
    assert RESISTANCE_FACTORS == {
        "side_cohesive": 0.45, "side_cohesionless": 0.55, "side_rock": 0.50,
        "tip_cohesive": 0.40, "tip_cohesionless": 0.50, "tip_rock": 0.50,
        "uplift_cohesive": 0.35, "uplift_cohesionless": 0.45,
        "uplift_rock": 0.40}


def test_the_calc_steps_print_what_was_used():
    a = _figure_10_5()
    r = a.compute()
    steps = {s.title: s for sec in get_calc_steps(r, a)
             for s in sec.items if hasattr(s, "title")}
    assert "Alpha Method (Cohesive Soils), GEC-10 (2018) Eq. 10-18" in steps
    (clay,) = [s for t, s in steps.items()
               if t.startswith("Layer 3:") and "Alpha" in t]
    assert "Eq. 10-17" in clay.substitution
    assert "= 0.440" in clay.substitution
    assert "Bottom Zone (cohesive soil)" in steps
    nc = steps["Bearing Capacity Factor N*c (Cohesive Base), Table 10-2"]
    assert nc.result_value == "9.00"
    unit = steps["Unit End Bearing"]
    assert float(unit.result_value.replace(",", "")) * a.shaft.tip_area == \
        pytest.approx(r.Q_tip, rel=1e-3)
    rock = _rock_shaft()
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=6.0),
                           soil=ShaftSoilProfile(layers=[ShaftSoilLayer(
                               10.0, "rock", 23.0, qu=20000.0, RQD=40.0)]))
    titles = [s.title for sec in get_calc_steps(rock, a)
              for s in sec.items if hasattr(s, "title")]
    assert "Unit End Bearing (Rock Base), GEC-10 Eq. 10-23" in titles
