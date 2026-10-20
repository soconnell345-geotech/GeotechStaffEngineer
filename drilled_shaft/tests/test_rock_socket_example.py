"""GEC-10 Illustrative Example 10-4, rock socket resistance (GM-221).

GEC-10 (FHWA-NHI-18-024) pp. 10-27, 10-28: 6.5 ft casing through 34 ft of
soil and decomposed rock to the top of competent sandstone, a 6 ft socket
10 ft deep, qu = 2,500 psi (360 ksf), RQD 37 to 100 and competent rock
below the tip. As printed:

* f_SN = pa C sqrt(qu / pa) = 2.116 x 1.0 x sqrt(360 / 2.116) = 27.6 ksf
  (Eq. 10-21, C = 1.0 for a 'normal' socket);
* R_SN = 27.6 ksf x (6 ft x pi) x 10 ft = 5,202 kips;
* q_BN = N*cr qu = 2.5 x 360 = 900 ksf (Eq. 10-23);
* R_BN = pi/4 (6 ft)^2 x 900 ksf = 25,447 kips: the base on the SOCKET
  diameter B, not the 6.5 ft casing;
* phi R_BN = 0.50 x 25,447 = 12,723 kips.

Before GM-221 the base was taken on the shaft diameter, pi/4 (6.5 ft)^2 x
900 ksf = 29,865 kips (+17.4 %), and C was fixed at 0.65 (Eq. 10-22 with
alpha_E = 1.0), so the side could not be reproduced.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps, get_input_summary
from drilled_shaft.lrfd import apply_lrfd

FT, KIP = 0.3048, 4.4482216
QU = 2500 * 6.894757  # kPa


def _soil():
    return ShaftSoilProfile(layers=[
        ShaftSoilLayer(34 * FT, "cohesionless", 19.0, phi=32.0, N60=20.0,
                       description="soil and decomposed rock"),
        ShaftSoilLayer(20 * FT, "rock", 23.0, qu=QU, RQD=93.0,
                       description="competent sandstone")])


def _shaft(**kw):
    return DrillShaft(**{"diameter": 6.5 * FT, "length": 44 * FT,
                         "socket_diameter": 6.0 * FT,
                         "casing_depth": 34 * FT, **kw})


def _run(C=None, **kw):
    extra = {} if C is None else {"rock_side_C": C}
    return DrillShaftAnalysis(shaft=_shaft(**kw), soil=_soil(),
                              **extra).compute()


def test_example_10_4_side_and_base_as_printed():
    r = _run(C=1.0)
    (rock,) = [lb for lb in r.layer_breakdown if lb["soil_type"] == "rock"]
    assert rock["fs_kPa"] / 47.880 == pytest.approx(27.6, abs=0.05)
    assert r.Q_side_rock / KIP == pytest.approx(5202, abs=1)
    assert r.Q_side_clay == r.Q_side_sand == 0.0   # all cased
    assert r.Q_tip / KIP == pytest.approx(25447, abs=1)
    assert r.base_diameter == pytest.approx(6.0 * FT)
    lrfd = apply_lrfd(r, "rock")
    assert lrfd["phi_Qt_kN"] / KIP == pytest.approx(12723, abs=1)


def test_the_base_was_on_the_shaft_diameter_before():
    """Before and after on the same shaft: 29,865 kips on the 6.5 ft shaft
    diameter, 25,447 kips on the 6 ft socket."""
    before = 2.5 * QU * math.pi / 4 * (6.5 * FT) ** 2
    assert before / KIP == pytest.approx(29865, abs=1)
    r = _run()
    assert r.Q_tip / KIP == pytest.approx(25447, abs=1)
    assert r.Q_tip / before == pytest.approx((6.0 / 6.5) ** 2)
    # no socket diameter given: the base is the shaft's, as before
    assert _run(socket_diameter=None).Q_tip == pytest.approx(before)


def test_the_default_c_is_unchanged_and_said():
    r = _run()
    assert r.Q_side_rock / KIP == pytest.approx(0.65 * 5202.75, abs=1)
    (c,) = [c for c in r.choices if c["what"].startswith("rock side")]
    assert c["how"] == "default" and c["chosen"].startswith("0.65: Eq. 10-22")
    assert "C = 1.0 is recommended" in c["basis"]
    assert any(a.startswith("1.0: Eq. 10-21") for a in c["alternatives"])
    c1 = [c for c in _run(C=1.0).choices
          if c["what"].startswith("rock side")][0]
    assert c1["how"] == "given" and c1["chosen"].startswith("1.0: Eq. 10-21")


def test_a_bell_still_takes_the_base_and_soil_tips_keep_the_shaft():
    r = _run(bell_diameter=7.0 * FT)
    assert r.base_diameter == pytest.approx(7.0 * FT)
    shaft = _shaft(length=30 * FT, casing_depth=0.0)   # tip in the soil
    r = DrillShaftAnalysis(shaft=shaft, soil=_soil()).compute()
    assert r.base_diameter == pytest.approx(6.5 * FT)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="Socket diameter"):
        _shaft(socket_diameter=0.0)
    with pytest.raises(ValueError, match="rock_side_C"):
        DrillShaftAnalysis(shaft=_shaft(), soil=_soil(), rock_side_C=0.0)


def test_the_capacity_curve_carries_c():
    a = DrillShaftAnalysis(shaft=_shaft(), soil=_soil(), rock_side_C=1.0)
    deepest = a.capacity_vs_depth(depth_min=40 * FT, n_points=2)[-1]
    r = a.compute()
    assert deepest["Q_skin_kN"] == pytest.approx(r.Q_skin, abs=0.1)
    assert deepest["Q_tip_kN"] == pytest.approx(r.Q_tip, abs=0.1)


def test_the_calc_steps_print_the_socket_base_and_c():
    a = DrillShaftAnalysis(shaft=_shaft(), soil=_soil(), rock_side_C=1.0)
    r = a.compute()
    steps = [s for sec in get_calc_steps(r, a) for s in sec.items
             if hasattr(s, "equation")]
    tip = [s for s in steps if s.title == "Shaft Tip Area"][0]
    assert "D_sock" in tip.equation and "Example 10-4" in tip.reference
    assert float(tip.result_value) == pytest.approx(r.base_area, abs=1e-4)
    rock = [s for s in steps if s.title.endswith("Rock Socket")][0]
    assert rock.substitution.startswith("f_s = 1 ×")
    method = [s for s in steps if s.title == "Rock Socket Method"][0]
    assert "C = 1 (rock_side_C)" in method.equation
    a_tip = [i for i in get_input_summary(r, a) if i.name == "A_tip"][0]
    assert float(a_tip.value) == pytest.approx(r.base_area, abs=1e-4)
