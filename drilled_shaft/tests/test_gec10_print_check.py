"""GM-284: drilled_shaft checked against GEC-10 (FHWA-NHI-18-024, 2018) as
printed; module_work/drilled_shaft_print/CHECK.md has the method-by-method
account.

The pages this file pins:

* p. 10-15 (Eq. 10-14): 'N60 is the average value between the base and two
  diameters beneath the base', 'the field value, with no correction for
  overburden stress', limited to N <= 50 (30 tsf). The base took the tip
  layer's N60 alone until GM-284.
* p. 10-21: 'For design, the value of qu used in Equation 10-21 should not
  exceed the compressive strength of the drilled shaft concrete'; Example
  10-4 (pp. 10-27, 10-28): qu 2,500 psi against f'c 4,000 psi.
* p. 10-21: Eq. 10-22, f_SN = 0.65 alpha_E pa sqrt(qu / pa), with Table
  10-3 (alpha_E by RQD, closed or open joints), for caving rock.
* Example 10-5 (p. 10-29): 0.55 x 197.2 + 0.55 x 635.6 + 0.45 x 479.7 +
  0.40 x 453.6 = 855.4 kips.

Values not from the print are made up.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps
from drilled_shaft.lrfd import apply_lrfd
from drilled_shaft.side_resistance import TABLE_10_3, alpha_E_table_10_3

FT, PCF, PSF, KIP, PSI = 0.3048, 0.157087, 0.0478803, 4.44822, 6.894757
PA = 101.325


def _choice(r, what):
    (c,) = [c for c in r.choices if c["what"] == what]
    return c


# -- Eq. 10-14: N60 the mean over 2B below the base ------------------------

def _sand(*layers, L=9.5, D=1.0, **options):
    soil = ShaftSoilProfile(layers=list(layers))
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=D, length=L),
                              soil=soil, **options).compute()


def test_the_sand_base_takes_n60_over_2b_below_it():
    """Base at 9.5 m, B = 1 m: 0.5 m at N60 20 and 1.5 m at 40 within 2B,
    mean 35, q_BN = 57.5 x 35 kPa; the tip layer's 20 alone was taken
    before."""
    r = _sand(ShaftSoilLayer(10, "cohesionless", 19, phi=34, N60=20),
              ShaftSoilLayer(10, "cohesionless", 20, phi=36, N60=40))
    b = r.base_steps
    assert b["N60_mean"] == pytest.approx((0.5 * 20 + 1.5 * 40) / 2.0)
    assert b["N60_tip"] == 20 and b["N60_zone_m"] == [9.5, 11.5]
    assert b["qb"] == pytest.approx(57.5 * 35)
    assert r.Q_tip == pytest.approx(57.5 * 35 * math.pi / 4)
    c = _choice(r, "base resistance in cohesionless soil")
    assert "Eq. 10-14" in c["chosen"] and "N60 = 35.0" in c["chosen"]
    assert "no correction for overburden stress" in c["basis"]
    (alt,) = c["alternatives"]
    assert alt.startswith("N60 = the tip layer's alone, 20")
    assert f"Q_b {57.5 * 20 * math.pi / 4:,.0f} kN" in alt


def test_one_sand_layer_through_2b_is_its_own_n60():
    r = _sand(ShaftSoilLayer(30, "cohesionless", 19, phi=34, N60=25))
    assert r.base_steps["N60_mean"] == 25.0
    assert r.Q_tip == pytest.approx(57.5 * 25 * math.pi / 4)
    assert not _choice(r, "base resistance in cohesionless soil").get(
        "alternatives")


def test_clay_within_2b_is_left_out_and_said():
    """Base at 9.5 m: 1 m of the sand (N60 30), then clay, which gives no
    N60."""
    r = _sand(ShaftSoilLayer(10.5, "cohesionless", 19, phi=34, N60=30),
              ShaftSoilLayer(10, "cohesive", 18, cu=80))
    assert r.base_steps["N60_mean"] == 30.0
    c = _choice(r, "base resistance in cohesionless soil")
    assert "cohesive at 10.5-11.5 m left out (no N60)" in c["chosen"]


def test_the_mean_is_limited_to_50():
    """Mean 60 over 2B: Eq. 10-14 holds to N <= 50, 30 tsf above it."""
    r = _sand(ShaftSoilLayer(10, "cohesionless", 19, phi=34, N60=45),
              ShaftSoilLayer(10, "cohesionless", 20, phi=38, N60=65))
    assert r.base_steps["N60_mean"] == pytest.approx(60.0)
    assert r.base_steps["qb"] == pytest.approx(57.5 * 50)
    c = _choice(r, "base resistance in cohesionless soil")
    assert "limited to 50 (30 tsf)" in c["chosen"]
    assert "load testing" in c["basis"]


def test_a_profile_ending_inside_2b_takes_what_it_gives():
    r = _sand(ShaftSoilLayer(5, "cohesionless", 19, phi=34, N60=20),
              ShaftSoilLayer(5, "cohesionless", 20, phi=36, N60=30), L=9.5)
    assert r.base_steps["N60_mean"] == 30.0
    assert r.base_steps["N60_profile_ends_m"] == 10.0
    assert "the profile ends at 10 m" in _choice(
        r, "base resistance in cohesionless soil")["chosen"]


def test_the_calc_steps_print_the_mean():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(10, "cohesionless", 19, phi=34, N60=20),
        ShaftSoilLayer(10, "cohesionless", 20, phi=36, N60=40)])
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=9.5),
                           soil=soil)
    r = a.compute()
    steps = {s.title: s for sec in get_calc_steps(r, a) for s in sec.items
             if hasattr(s, "title")}
    n = steps["N_60 of Eq. 10-14 (mean over 2B below the base)"]
    assert n.result_value == "35.0" and "p. 10-15" in n.reference
    assert "57.5 × 35.0" in steps[
        "Unit End Bearing (Cohesionless Tip)"].substitution


# -- p. 10-21: qu no higher than the concrete strength --------------------

def _example_10_4(**shaft):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(34 * FT, "cohesionless", 19.0, phi=32.0, N60=20.0),
        ShaftSoilLayer(20 * FT, "rock", 23.0, qu=2500 * PSI, RQD=93.0)])
    return DrillShaftAnalysis(
        shaft=DrillShaft(diameter=6.5 * FT, length=44 * FT,
                         socket_diameter=6.0 * FT, casing_depth=34 * FT,
                         **shaft),
        soil=soil, rock_base_conditions_met=True)


def test_example_10_4_qu_is_below_fc():
    """qu 2,500 psi against f'c 4,000 psi: no limit binds, 5,202 kips."""
    r = _example_10_4(concrete_fc=4000 * PSI).compute()
    assert r.Q_side_rock / KIP == pytest.approx(5202, abs=1)
    c = _choice(r, "qu in the rock side resistance, against the concrete "
                   "strength")
    assert c["how"] == "given" and c["chosen"].startswith("no limit binds")
    assert "should not exceed the compressive strength" in c["basis"]


def test_qu_above_fc_is_limited_and_said():
    """Made-up f'c 10,000 kPa below the example's qu 17,237 kPa."""
    a = _example_10_4(concrete_fc=10000.0)
    r = a.compute()
    (rock,) = [lb for lb in r.layer_breakdown if lb["soil_type"] == "rock"]
    fs = math.sqrt(10000.0 * PA)
    assert rock["fs_kPa"] == pytest.approx(fs, abs=0.05)
    assert rock["rock_steps"]["qu_used_kPa"] == 10000.0
    assert rock["rock_steps"]["qu_limited"]
    assert r.Q_side_rock == pytest.approx(
        fs * math.pi * 6.0 * FT * 10 * FT, rel=1e-6)
    # the base is not limited by it: 2.5 qu as before
    assert r.Q_tip / KIP == pytest.approx(25447, abs=1)
    c = _choice(r, "qu in the rock side resistance, against the concrete "
                   "strength")
    assert c["chosen"].startswith("qu limited to f'c = 10,000 kPa")
    unlimited = _example_10_4().compute().Q_side_rock
    assert f"Q_s in rock {unlimited:,.0f} kN" in c["alternatives"][0]
    rock_step = [s for sec in get_calc_steps(r, a) for s in sec.items
                 if getattr(s, "title", "").endswith("Rock Socket")][0]
    assert "limited to f'c = 10,000 kPa" in rock_step.notes


def test_fc_left_out_is_not_checked_and_said():
    r = _example_10_4().compute()
    c = _choice(r, "qu in the rock side resistance, against the concrete "
                   "strength")
    assert c["how"] == "default" and c["chosen"].startswith("not checked")


def test_fc_is_in_kpa():
    with pytest.raises(ValueError, match="kPa"):
        DrillShaft(diameter=1.0, length=10.0, concrete_fc=28.0)


# -- p. 10-21: Eq. 10-22 with Table 10-3 ------------------------------------

def test_table_10_3_as_printed():
    assert TABLE_10_3 == ((100.0, 1.00, 0.85), (70.0, 0.85, 0.55),
                          (50.0, 0.60, 0.55), (30.0, 0.50, 0.50),
                          (20.0, 0.45, 0.45))
    for rqd, closed, open_ in TABLE_10_3:
        assert alpha_E_table_10_3(rqd, "closed") == closed
        assert alpha_E_table_10_3(rqd, "open") == open_
    # linear between the rows; held at the RQD 20 row below it
    assert alpha_E_table_10_3(60.0) == pytest.approx(0.725)
    assert alpha_E_table_10_3(85.0, "open") == pytest.approx(0.70)
    assert alpha_E_table_10_3(10.0) == 0.45
    with pytest.raises(ValueError, match="rock_joints"):
        alpha_E_table_10_3(50.0, "tight")


def _caving(RQD=60.0, method="caving", **options):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(4.0, "cohesive", 18.0, cu=60.0),
        ShaftSoilLayer(10.0, "rock", 23.0, qu=8000.0, RQD=RQD)])
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=9.0),
                           soil=soil, rock_side_method=method,
                           rock_base_conditions_met=True, **options)
    return a, a.compute()


def test_caving_rock_takes_eq_10_22_with_table_10_3():
    """Made-up rock, RQD 60, closed joints: alpha_E 0.725, f_s = 0.65 x
    0.725 x sqrt(8,000 x 101.325) kPa."""
    a, r = _caving()
    (rock,) = [lb for lb in r.layer_breakdown if lb["soil_type"] == "rock"]
    fs = 0.65 * 0.725 * math.sqrt(8000.0 * PA)
    assert rock["fs_kPa"] == pytest.approx(fs, abs=0.05)
    assert rock["rock_steps"]["equation"] == "10-22"
    assert r.Q_side_rock == pytest.approx(fs * math.pi * 1.2 * 5.0, rel=1e-6)
    m = _choice(r, "rock side resistance method")
    assert m["how"] == "given" and m["chosen"].startswith("Eq. 10-22")
    assert "RQD 60 %: alpha_E 0.725" in m["chosen"]
    normal = DrillShaftAnalysis(shaft=a.shaft, soil=a.soil,
                                rock_base_conditions_met=True).compute()
    assert f"Q_s in rock {normal.Q_side_rock:,.0f} kN" in m["alternatives"][0]
    j = _choice(r, "joint condition in Table 10-3")
    assert j["how"] == "default" and j["chosen"] == "closed joints"
    _, opened = _caving(rock_joints="open")
    assert f"Q_s in rock {opened.Q_side_rock:,.0f} kN" in j["alternatives"][0]
    assert opened.Q_side_rock == pytest.approx(
        r.Q_side_rock * 0.55 / 0.725, rel=1e-6)
    steps = [s for sec in get_calc_steps(r, a) for s in sec.items
             if hasattr(s, "title")]
    assert any(s.title == "Rock Socket Method, Eq. 10-22 (caving rock)"
               for s in steps)
    rock_step = [s for s in steps if s.title.endswith("Rock Socket")][0]
    assert rock_step.substitution.startswith("f_s = 0.65 × 0.725 × ")


def test_rqd_below_the_table_is_held_and_said():
    _, r = _caving(RQD=10.0)
    m = _choice(r, "rock side resistance method")
    assert "held at 0.45" in m["basis"]


def test_the_normal_socket_names_eq_10_22_as_its_alternative():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(10.0, "rock", 23.0, qu=8000.0, RQD=60.0)])
    r = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=5.0),
                           soil=soil, rock_base_conditions_met=True).compute()
    m = _choice(r, "rock side resistance method")
    assert m["how"] == "default" and m["chosen"].startswith("Eq. 10-21")
    assert m["alternatives"][0].startswith("rock_side_method 'caving'")
    assert _choice(r, "rock side resistance coefficient C")["how"] == \
        "default"


def test_caving_takes_no_c():
    with pytest.raises(ValueError, match="takes no C"):
        _caving(rock_side_C=0.65)
    with pytest.raises(ValueError, match="rock_side_method"):
        _caving(method="rough")


# -- Example 10-5: the factored resistance -----------------------------------

def test_example_10_5_factored_resistance():
    """Examples 10-1 to 10-3 factored by Table 8-4: 855.4 kips as printed
    (0.55, 0.55, 0.45, 0.40); the tool 853.0 kips, 0.3 % under, from
    Example 10-1's rounding (V-058, 828.1 against 833 kips)."""
    assert 0.55 * (197.2 + 635.6) + 0.45 * 479.7 + 0.40 * 453.6 == \
        pytest.approx(855.4, abs=0.06)
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(15 * FT, "cohesionless", 115 * PCF, phi=38, N60=9,
                       sigma_p_correlation="silty_sand"),
        ShaftSoilLayer(18 * FT, "cohesionless", 125 * PCF, phi=41, N60=27,
                       sigma_p_correlation="silty_sand"),
        ShaftSoilLayer(40 * FT, "cohesive", 125 * PCF, cu=2000 * PSF)],
        gwt_depth=15 * FT)
    r = DrillShaftAnalysis(shaft=DrillShaft(diameter=5 * FT, length=60 * FT),
                           soil=soil, su_test_type="uu").compute()
    lrfd = apply_lrfd(r)
    assert (lrfd["phi_side_sand"], lrfd["phi_side_clay"],
            lrfd["phi_tip"]) == (0.55, 0.45, 0.40)
    assert lrfd["phi_Qn_kN"] / KIP == pytest.approx(855.4, rel=0.004)
    assert lrfd["phi_Qn_kN"] / KIP == pytest.approx(853.0, abs=0.1)
