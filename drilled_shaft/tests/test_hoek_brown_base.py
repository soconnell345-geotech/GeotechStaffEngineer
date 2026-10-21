"""GM-251: the base of a fractured rock mass, GEC-10 (FHWA-NHI-18-024, 2018)
Sec. 10.3.5.3, as printed.

* Eq. 10-24 (p. 10-23): sigma'1 = sigma'3 + qu [m_b (sigma'3 / qu) + s]^a
  (Hoek et al. 2002), qu of the intact rock.
* Eqs. 10-25 to 10-29 (p. 10-24): m_b / m_i = exp[(GSI - 100) / 28],
  s = exp[(GSI - 100) / 9], a = 1/2 + (1/6)[exp(-GSI/15) - exp(-20/3)],
  A = sigma'vb + qu [m_b (sigma'vb / qu) + s]^a,
  q_BN = A + qu [m_b (A / qu) + s]^a.
* p. 10-24: 'A value of N*cr equal to 2.5 can be used as an upper-bound
  limit to base resistance ..., unless local experience or load tests can
  be used to validate higher values'; 'The recommend resistance factor for
  application to Equation 10-29 is phi = 0.50'; Table 8-4 (p. 8-14) prints
  0.50 for 'Eq. 10-23 or Eq. 10-29'.
* Fig. 10-8 (p. 10-25): N*cr = q_BN / qu against GSI at sigma'vb = 0, for
  m_i 4, 10, 15, 20, 25 and 33, to 'maximum qBN/qu = 2.5'.

No GEC-10 worked example uses Eqs. 10-24 to 10-29: the printed check is
Fig. 10-8. The points below were read from the figure's plotted curves
(the vector paths of p. 10-25, axes GSI 10 to 90 and N*cr 0 to 3).
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps
from drilled_shaft.end_bearing import hoek_brown_base

#: (m_i, GSI, N*cr read from Fig. 10-8)
FIGURE_10_8 = [
    (4, 50.1, 0.283), (4, 70.0, 0.737), (4, 94.8, 2.503),
    (10, 60.0, 0.633), (10, 70.0, 1.015), (10, 88.7, 2.503),
    (33, 70.0, 1.668), (33, 78.7, 2.5),
]


@pytest.mark.parametrize("m_i, GSI, ncr", FIGURE_10_8)
def test_figure_10_8_as_plotted(m_i, GSI, ncr):
    """At sigma'vb = 0, N*cr from Eqs. 10-25 to 10-29, limited to 2.5,
    within 0.02 of the plotted curve."""
    r = hoek_brown_base(qu=1000.0, GSI=GSI, m_i=m_i, sigma_vb=0.0)
    assert r["factor"] == pytest.approx(ncr, abs=0.02)


def test_gsi_100_is_the_intact_rock():
    """GSI 100: m_b = m_i, s = 1, a = 1/2 (Eqs. 10-25 to 10-27), so at
    sigma'vb = 0, A = qu and q_BN = qu [1 + sqrt(m_i + 1)], past 2.5."""
    r = hoek_brown_base(qu=1000.0, GSI=100.0, m_i=10.0, sigma_vb=0.0)
    assert (r["m_b"], r["s"], r["a"]) == pytest.approx((10.0, 1.0, 0.5))
    assert r["A_kPa"] == pytest.approx(1000.0)
    assert r["Ncr_uncapped"] == pytest.approx(1 + math.sqrt(11))
    assert r["capped"] and r["factor"] == 2.5 and r["qb"] == 2500.0


def test_the_equations_step_by_step():
    """GSI 45, m_i 10, qu 20 MPa, sigma'vb 286 kPa, each equation by
    hand."""
    GSI, m_i, qu, sv = 45.0, 10.0, 20000.0, 286.0
    m_b = m_i * math.exp((GSI - 100) / 28)
    s = math.exp((GSI - 100) / 9)
    a = 0.5 + (math.exp(-GSI / 15) - math.exp(-20 / 3)) / 6
    A = sv + qu * (m_b * sv / qu + s) ** a
    q = A + qu * (m_b * A / qu + s) ** a
    r = hoek_brown_base(qu, GSI, m_i, sv)
    assert (r["m_b"], r["s"], r["a"]) == pytest.approx((m_b, s, a))
    assert r["A_kPa"] == pytest.approx(A) and r["qb"] == pytest.approx(q)
    assert not r["capped"] and r["factor"] == pytest.approx(q / qu)


def test_inputs_out_of_range_are_refused():
    with pytest.raises(ValueError, match="GSI must be 0 to 100"):
        hoek_brown_base(1000.0, 120.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="m_i must be positive"):
        ShaftSoilLayer(5.0, "rock", 23.0, qu=1000.0, GSI=50.0, m_i=0.0)
    with pytest.raises(ValueError, match="only a rock layer"):
        ShaftSoilLayer(5.0, "cohesive", 18.0, cu=50.0, GSI=50.0)


def _soil(**rock):
    return ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        ShaftSoilLayer(10.0, "rock", 23.0, qu=20000.0, **rock)])


def _run(soil, **options):
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=14.0),
                              soil=soil, **options).compute()


def _choice(r, what):
    (c,) = [c for c in r.choices if c["what"] == what]
    return c


def test_fractured_rock_with_its_inputs_takes_the_hoek_brown_base():
    """RQD 40 below the base and the conditions of Eq. 10-23 not stated:
    refused before GM-251; with GSI and m_i given, Eqs. 10-24 to 10-29."""
    r = _run(_soil(RQD=40.0, GSI=45.0, m_i=10.0))
    st = r.base_steps
    assert st["equation"] == "10-29"
    assert st["sigma_vb_kPa"] == pytest.approx(8 * 18.5 + 6 * 23.0)
    expect = hoek_brown_base(20000.0, 45.0, 10.0, st["sigma_vb_kPa"])
    area = math.pi / 4 * 1.2 ** 2
    assert r.Q_tip == pytest.approx(expect["qb"] * area)
    base = _choice(r, "base resistance in rock")
    assert base["how"] == "default"
    assert base["chosen"].startswith("GEC-10 (2018) Eqs. 10-24 to 10-29")
    assert "Eq. 10-24 (p. 10-23)" in base["basis"]
    assert "Eqs. 10-25 to 10-27 (p. 10-24)" in base["basis"]
    assert "resistance factor 0.50 (p. 10-24; Table 8-4" in base["basis"]
    met = _choice(r, "rock below the base (the conditions of Eq. 10-23)")
    assert met["chosen"].startswith("not taken as met")
    # Eq. 10-23 with the conditions stated, and the 'rqd' base, with values
    full = _run(_soil(RQD=40.0, GSI=45.0, m_i=10.0),
                rock_base_conditions_met=True)
    assert full.base_steps["equation"] == "10-23"
    assert f"Q_b {full.Q_tip:,.0f} kN" in base["alternatives"][0]
    assert base["alternatives"][1].startswith("rock_base_method 'rqd'")
    # the HB base where Eq. 10-23 is taken, as its alternative
    alt = _choice(full, "base resistance in rock")["alternatives"][1]
    assert "GSI 45 and m_i 10" in alt and f"Q_b {r.Q_tip:,.0f} kN" in alt


def test_the_conditions_stated_not_met_take_the_hoek_brown_base():
    r = _run(_soil(RQD=40.0, GSI=45.0, m_i=10.0),
             rock_base_conditions_met=False)
    assert r.base_steps["equation"] == "10-29"
    met = _choice(r, "rock below the base (the conditions of Eq. 10-23)")
    assert met["how"] == "given" and met["chosen"].startswith("not met")
    # at RQD 100 too: the statement is the engineer's
    r100 = _run(_soil(GSI=45.0, m_i=10.0), rock_base_conditions_met=False)
    assert r100.Q_tip == pytest.approx(r.Q_tip)


def test_refused_only_where_the_inputs_are_missing_and_named():
    with pytest.raises(ValueError) as e:
        _run(_soil(RQD=40.0))
    msg = str(e.value)
    assert "RQD 40 % at 14-15.2 m" in msg
    assert "no GSI and no m_i" in msg and "Table 9-10" in msg
    with pytest.raises(ValueError, match="needs the tip rock layer's m_i"):
        _run(_soil(RQD=40.0, GSI=45.0), rock_base_conditions_met=False)
    with pytest.raises(ValueError, match="GSI and m_i"):
        _run(_soil(), rock_base_conditions_met=False)


def test_the_limit_of_2_5_and_the_unlimited_value_beside_it():
    r = _run(_soil(RQD=40.0, GSI=90.0, m_i=20.0))
    st = r.base_steps
    assert st["capped"] and st["factor"] == 2.5
    assert st["Ncr_uncapped"] > 2.5
    base = _choice(r, "base resistance in rock")
    assert "limited to 2.5 (p. 10-24)" in base["chosen"]
    (unlimited,) = [a for a in base["alternatives"]
                    if "not limited to 2.5" in a]
    area = math.pi / 4 * 1.2 ** 2
    assert f"Q_b {st['qb_uncapped'] * area:,.0f} kN" in unlimited


def test_other_rock_within_one_diameter_below_is_named_with_its_value():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        ShaftSoilLayer(6.5, "rock", 23.0, qu=20000.0, RQD=40.0, GSI=45.0,
                       m_i=10.0),
        ShaftSoilLayer(5.0, "rock", 23.0, qu=8000.0, GSI=30.0, m_i=7.0)])
    r = _run(soil)
    base = _choice(r, "base resistance in rock")
    (other,) = [a for a in base["alternatives"] if "14.5-15.2 m" in a]
    hb = hoek_brown_base(8000.0, 30.0, 7.0, r.base_steps["sigma_vb_kPa"])
    area = math.pi / 4 * 1.2 ** 2
    assert f"Q_b {hb['qb'] * area:,.0f} kN" in other
    assert "which rock governs is the engineer's judgment" in base["basis"]


def test_the_calc_steps_print_eqs_10_28_and_10_29():
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=14.0),
                           soil=_soil(RQD=40.0, GSI=45.0, m_i=10.0))
    r = a.compute()
    steps = {}
    for section in get_calc_steps(r, a):
        if section.title == "End Bearing":
            for item in section.items:
                steps[item.title] = item
    assert "Hoek-Brown Base, Eq. 10-28" in steps
    q = steps["Unit End Bearing (Fractured Rock Mass), GEC-10 Eq. 10-29"]
    assert float(q.result_value.replace(",", "")) == pytest.approx(
        r.base_steps["qb"], abs=0.1)


# GM-251 review: where the conditions of p. 10-22 hold, GEC-10 gives Eq.
# 10-23 (2.5 is 'recommended for design ... when the following conditions
# are met'); the Hoek-Brown base is 'For fractured rock mass' (p. 10-23).
# GSI and m_i given do not make the lower value the base by themselves.

def test_conditions_taken_as_met_keep_eq_10_23_with_hoek_brown_beside():
    """RQD 100 (the default) within one diameter below the base, GSI and
    m_i given, the conditions not stated: Eq. 10-23, 2.5 qu; the
    Hoek-Brown value is given beside it in both choices."""
    r = _run(_soil(GSI=45.0, m_i=10.0))
    assert r.base_steps["equation"] == "10-23"
    assert r.base_steps["factor"] == 2.5
    hb = _run(_soil(GSI=45.0, m_i=10.0), rock_base_conditions_met=False)
    assert hb.base_steps["equation"] == "10-29"
    assert hb.Q_tip < r.Q_tip
    met = _choice(r, "rock below the base (the conditions of Eq. 10-23)")
    assert met["how"] == "default" and met["chosen"].startswith(
        "taken as met")
    assert "GSI 45 and m_i 10" in met["chosen"]
    assert "RQD is 100 where none is given" in met["chosen"]
    (alt,) = [a for a in met["alternatives"]
              if a.startswith("rock_base_conditions_met false")]
    assert f"Q_b {hb.Q_tip:,.0f} kN" in alt
    base = _choice(r, "base resistance in rock")
    assert base["chosen"].startswith("GEC-10 (2018) Eq. 10-23")
    assert any(f"Q_b {hb.Q_tip:,.0f} kN" in a for a in base["alternatives"])


def test_rqd_below_100_without_a_statement_takes_hoek_brown():
    """The same rock at RQD 40: the conditions are not taken as met, and
    the reason given is the layer below the base, not the GSI."""
    r = _run(_soil(RQD=40.0, GSI=45.0, m_i=10.0))
    assert r.base_steps["equation"] == "10-29"
    met = _choice(r, "rock below the base (the conditions of Eq. 10-23)")
    assert "give rock with RQD 40 % at 14-15.2 m" in met["chosen"]


def test_rock_below_the_base_shown_where_eq_10_23_is_taken():
    """A rock layer within one diameter below the base with its own GSI and
    m_i: shown with its Hoek-Brown value where the base takes Eq. 10-23,
    and under the 'rqd' base."""
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        ShaftSoilLayer(6.5, "rock", 23.0, qu=20000.0),
        ShaftSoilLayer(5.0, "rock", 23.0, qu=8000.0, GSI=30.0, m_i=7.0)])
    hb = hoek_brown_base(8000.0, 30.0, 7.0, 8 * 18.5 + 6 * 23.0)
    area = math.pi / 4 * 1.2 ** 2
    for options in ({}, {"rock_base_method": "rqd"}):
        r = _run(soil, **options)
        base = _choice(r, "base resistance in rock")
        (other,) = [a for a in base["alternatives"] if "14.5-15.2 m" in a]
        assert "GSI 30, m_i 7" in other
        assert f"Q_b {hb['qb'] * area:,.0f} kN" in other


@pytest.mark.parametrize("layers, why", [
    # GSI and m_i on the socket rock above the base
    ([ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
      ShaftSoilLayer(3.0, "rock", 23.0, qu=20000.0, GSI=30.0, m_i=7.0),
      ShaftSoilLayer(7.0, "rock", 23.0, qu=20000.0)],
     "above the base at 14 m"),
    # on rock more than one base diameter below the base
    ([ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
      ShaftSoilLayer(8.0, "rock", 23.0, qu=20000.0),
      ShaftSoilLayer(5.0, "rock", 23.0, qu=8000.0, GSI=30.0, m_i=7.0)],
     "more than one base diameter (1.2 m) below the base at 14 m"),
    # the base in clay
    ([ShaftSoilLayer(15.0, "cohesive", 18.5, cu=75.0),
      ShaftSoilLayer(5.0, "rock", 23.0, qu=8000.0, GSI=30.0)],
     "the base at 14 m is in cohesive soil"),
])
def test_gsi_and_m_i_no_base_takes_are_refused(layers, why):
    """As GM-179 and GM-249: an input a layer would not use is refused
    with its reason, not dropped."""
    with pytest.raises(ValueError) as e:
        _run(ShaftSoilProfile(layers=layers))
    msg = str(e.value)
    assert "does not take" in msg and why in msg


def test_capacity_vs_depth_takes_gsi_where_a_trial_base_is():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        ShaftSoilLayer(10.0, "rock", 23.0, qu=20000.0, RQD=40.0, GSI=45.0,
                       m_i=10.0)])
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=14.0),
                           soil=soil)
    curve = a.capacity_vs_depth(depth_min=6.0, depth_max=14.0, n_points=5)
    assert len(curve) == 5
    with pytest.raises(ValueError, match="no trial base from 3 to 7 m"):
        a.capacity_vs_depth(depth_min=3.0, depth_max=7.0, n_points=5)


@pytest.mark.parametrize("soil_type,props", [
    ("cohesive", {"cu": 75.0}),
    ("cohesionless", {"phi": 34.0, "N60": 30.0}),
])
@pytest.mark.parametrize("flag", [True, False])
def test_rock_base_conditions_met_with_the_base_in_soil_is_refused(
        soil_type, props, flag):
    """GM-259: the flag bears on a base in rock (GEC-10 2018 p. 10-22);
    with the base in soil it is refused, not dropped."""
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(20.0, soil_type, 18.5, **props)])
    with pytest.raises(ValueError) as e:
        _run(soil, rock_base_conditions_met=flag)
    msg = str(e.value)
    assert "does not take rock_base_conditions_met" in msg
    assert "p. 10-22" in msg and f"is in {soil_type} soil" in msg


@pytest.mark.parametrize("flag", [True, False])
def test_rock_base_conditions_met_with_the_base_in_rock_still_works(flag):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        ShaftSoilLayer(10.0, "rock", 23.0, qu=20000.0, RQD=40.0, GSI=45.0,
                       m_i=10.0)])
    assert _run(soil, rock_base_conditions_met=flag).Q_tip > 0
