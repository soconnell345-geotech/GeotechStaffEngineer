"""GM-142: a cohesionless tip needs its N60. It took N60 = 15 without
saying so (capacity and calc steps alike); now it is refused, and the
length sweep refuses it too rather than leaving those depths out."""

import pytest

from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile
from drilled_shaft.capacity import DrillShaftAnalysis
from drilled_shaft.end_bearing import end_bearing_cohesionless


def _analysis(N60=0.0, length=12.0):
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(6.0, "cohesive", 18.0, cu=60.0),
        ShaftSoilLayer(10.0, "cohesionless", 19.0, phi=34, N60=N60,
                       description="dense sand"),
    ])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=length),
                              soil=soil)


def test_a_cohesionless_tip_with_no_N60_is_refused():
    with pytest.raises(ValueError, match="no N60") as err:
        _analysis().compute()
    assert "dense sand" in str(err.value) and "57.5 N60" in str(err.value)


def test_a_given_N60_is_what_the_tip_uses():
    a = _analysis(N60=30.0)
    r = a.compute()
    assert r.Q_tip == pytest.approx(
        end_bearing_cohesionless(30.0, a.shaft.tip_area, 1.0))


def test_the_length_sweep_refuses_it_too():
    a = _analysis()
    # tips in the clay only: no N60 needed
    assert len(a.capacity_vs_depth(depth_min=3.0, depth_max=5.5,
                                   n_points=4)) == 4
    with pytest.raises(ValueError, match="no N60"):
        a.capacity_vs_depth(depth_min=3.0, depth_max=12.0, n_points=6)


def test_the_calc_steps_print_the_N60_given():
    from drilled_shaft.calc_steps import get_calc_steps
    a = _analysis(N60=30.0)
    r = a.compute()
    text = repr(get_calc_steps(r, a))
    assert "57.5 × 30" in text
