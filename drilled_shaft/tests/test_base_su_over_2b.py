"""GM-252: the clay base's su as GEC-10 (FHWA-NHI-18-024, 2018) prints it.

p. 10-19, under Eqs. 10-19 and 10-20: 'su = mean undrained shear strength
of the cohesive soil over a depth of 2B below the base'. The tool takes the
mean by thickness from the base to 2B below it, B the base's diameter: the
tip layer at its su as its side resistance took it (Example 10-3, p. 10-20,
applies N*c to Example 10-2's su(CIUC)), each cohesive layer below at its
own cu, transformed alike. Cohesionless or rock layers within 2B have no su
and are left out; where the profile ends short of 2B, the mean is over the
soil it gives, and the choice says so. Before GM-252 the tip layer's su
alone was used, now the choice's alternative.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps
from drilled_shaft.end_bearing import nc_table_10_2

AREA = math.pi / 4 * 1.0 ** 2


def _run(layers, **options):
    shaft = options.pop("shaft", DrillShaft(diameter=1.0, length=12.0))
    soil = ShaftSoilProfile(layers=layers, gwt_depth=2.0)
    return DrillShaftAnalysis(shaft=shaft, soil=soil, **options).compute()


def _base(r):
    (c,) = [c for c in r.choices
            if c["what"] == "base resistance in cohesive soil"]
    return c


def _q_b(su):
    return nc_table_10_2(su) * su * AREA


def test_a_weaker_clay_within_2b_lowers_the_base():
    """cu 100 kPa to 13 m over cu 40 kPa; base at 12 m, B 1 m: 2B reaches
    14 m, half in each, su = 70 kPa. Before GM-252: su 100, Q_b 707 kN."""
    r = _run([ShaftSoilLayer(13.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(12.0, "cohesive", 17.0, cu=40.0)])
    assert r.base_steps["su_kPa"] == pytest.approx(70.0)
    assert r.base_steps["su_zone_m"] == [12.0, 14.0]
    assert r.Q_tip == pytest.approx(_q_b(70.0))
    assert r.Q_tip == pytest.approx(465.2, abs=0.1)
    c = _base(r)
    assert "the mean over 2B = 2 m below the base (12-14 m)" in c["chosen"]
    assert "12-13 m 100.0 kPa; 13-14 m 40.0 kPa" in c["chosen"]
    (alone,) = [a for a in c["alternatives"]
                if a.startswith("su = the tip layer's alone")]
    assert f"Q_b {_q_b(100.0):,.0f} kN" in alone   # 707 kN, before GM-252


def test_a_weaker_clay_below_2b_changes_nothing():
    """The same clays with the boundary at 15 m: 2B is all in the tip
    layer, su 100 kPa, Q_b 707 kN as before."""
    r = _run([ShaftSoilLayer(15.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(10.0, "cohesive", 17.0, cu=40.0)])
    assert r.base_steps["su_kPa"] == pytest.approx(100.0)
    assert r.Q_tip == pytest.approx(_q_b(100.0))
    assert r.Q_tip == pytest.approx(706.9, abs=0.1)
    c = _base(r)
    assert "all in the tip layer" in c["chosen"]
    assert not any(a.startswith("su = the tip layer's alone")
                   for a in c.get("alternatives", []))


def test_the_profile_ending_short_is_said_and_no_soil_invented():
    r = _run([ShaftSoilLayer(12.5, "cohesive", 18.0, cu=100.0)])
    assert r.base_steps["su_kPa"] == pytest.approx(100.0)
    assert r.base_steps["su_profile_ends_m"] == 12.5
    assert "the profile ends at 12.5 m, 1.5 m short of 2B" in \
        _base(r)["chosen"]
    # a weaker clay that stops short: the mean over what is given
    r = _run([ShaftSoilLayer(13.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(0.5, "cohesive", 17.0, cu=40.0)])
    assert r.base_steps["su_kPa"] == pytest.approx((100 + 0.5 * 40) / 1.5)
    assert "the profile ends at 13.5 m" in _base(r)["chosen"]


def test_sand_within_2b_has_no_su_and_is_left_out():
    r = _run([ShaftSoilLayer(12.6, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(5.0, "cohesionless", 19.0, phi=32.0,
                             N60=20.0)])
    assert r.base_steps["su_kPa"] == pytest.approx(100.0)
    assert "cohesionless at 12.6-14 m left out (no su)" in _base(r)["chosen"]


def test_a_uu_strength_below_is_transformed_where_it_is():
    """su_test_type 'uu': each layer's cu made su(CIUC) (Eq. 10-17), the
    layer below on sigma'vo at the middle of its part within 2B."""
    from drilled_shaft.side_resistance import su_to_ciuc
    layers = [ShaftSoilLayer(13.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(12.0, "cohesive", 17.0, cu=40.0)]
    r = _run(layers, su_test_type="uu")
    soil = ShaftSoilProfile(layers=layers, gwt_depth=2.0)
    below = su_to_ciuc(40.0, soil.effective_stress_at_depth(13.5), "uu")
    st = r.base_steps
    assert st["su_layers"][1]["su_kPa"] == pytest.approx(below, abs=1e-3)
    assert st["su_kPa"] == pytest.approx((st["su_tip_kPa"] + below) / 2,
                                         abs=1e-3)
    c = _base(r)
    assert any(a.startswith("su = cu as given, 70.0 kPa")
               for a in c["alternatives"])


def test_a_bell_sets_2b():
    """B is the base's diameter: a 2 m bell takes the mean to 4 m below."""
    shaft = DrillShaft(diameter=1.0, length=12.0, bell_diameter=2.0)
    r = _run([ShaftSoilLayer(14.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(12.0, "cohesive", 17.0, cu=40.0)],
             shaft=shaft)
    assert r.base_steps["su_zone_m"] == [12.0, 16.0]
    assert r.base_steps["su_kPa"] == pytest.approx(70.0)


def test_oneill_reese_keeps_the_tip_layers_cu():
    r = _run([ShaftSoilLayer(13.0, "cohesive", 18.0, cu=100.0),
              ShaftSoilLayer(12.0, "cohesive", 17.0, cu=40.0)],
             clay_base_method="oneill_reese")
    assert r.base_steps["su_kPa"] == pytest.approx(100.0)
    assert "su_zone_m" not in r.base_steps


def test_the_calc_steps_print_the_mean():
    a = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.0, length=12.0),
        soil=ShaftSoilProfile(layers=[
            ShaftSoilLayer(13.0, "cohesive", 18.0, cu=100.0),
            ShaftSoilLayer(12.0, "cohesive", 17.0, cu=40.0)], gwt_depth=2.0))
    r = a.compute()
    (section,) = [s for s in get_calc_steps(r, a) if s.title == "End Bearing"]
    (step,) = [i for i in section.items
               if i.title == "Base s_u, the Mean over 2B below the Base"]
    assert step.substitution == "(100.0 × 1.00 + 40.0 × 1.00) / 2.00"
    assert step.result_value == "70.0"
    assert "p. 10-19" in step.reference
