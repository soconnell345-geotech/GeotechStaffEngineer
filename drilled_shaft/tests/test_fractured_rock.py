"""Rock not considered part of the socket (GEC-10 p. 8-15, GM-220).

GEC-10 (FHWA-NHI-18-024) p. 8-15: the rock to be removed is defined
"including depth of rock sockets as well as the decomposed or fractured
rock that may be removed but not considered part of the socket for design
purposes". ``fractured_rock_thickness`` is that rock, measured below the top
of the first rock layer: no rock side resistance over it, while the soil
above keeps its own. ``casing_depth`` takes both away.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.calc_steps import get_calc_steps, get_input_summary
from drilled_shaft.side_resistance import PA

QU = 20000.0
#: GEC-10 rock socket unit side resistance as the module takes it (kPa)
FS_ROCK = 0.65 * math.sqrt(QU * PA)


def _soil(*rock):
    rock = rock or (10.0,)
    return ShaftSoilProfile(layers=[
        ShaftSoilLayer(8.0, "cohesive", 18.5, cu=75.0),
        *[ShaftSoilLayer(t, "rock", 23.0, qu=QU, RQD=80.0) for t in rock],
    ], gwt_depth=2.0)


def _run(soil=None, **shaft):
    return DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=14.0, **shaft),
        soil=soil or _soil()).compute()


def _rock_choice(r):
    (c,) = [c for c in r.choices if "socket" in c["what"]]
    return c


def test_the_default_neglects_no_rock_and_says_so():
    r = _run()
    assert r.Q_side_rock == pytest.approx(FS_ROCK * math.pi * 1.2 * 6.0,
                                          rel=1e-6)
    # the module's value; 78489.5 before GM-244 (the clay's alpha 0.55,
    # now GEC-10 Eq. 10-18's 0.530 over the clay's full length)
    assert r.Q_ultimate == pytest.approx(78452.1, abs=0.1)
    c = _rock_choice(r)
    assert c["chosen"] == "none" and c["how"] == "default"
    assert "p. 8-15" in c["basis"]
    assert r.to_dict()["choices"] == r.choices
    assert _run(fractured_rock_thickness=0.0).Q_ultimate == r.Q_ultimate


def test_the_top_of_rock_gives_no_side_resistance_and_the_soil_keeps_its():
    r0, r = _run(), _run(fractured_rock_thickness=1.5)
    assert r.Q_side_rock == pytest.approx(FS_ROCK * math.pi * 1.2 * 4.5,
                                          rel=1e-6)
    assert r.Q_side_clay == r0.Q_side_clay > 0
    assert r.Q_tip == r0.Q_tip
    (rock,) = [lb for lb in r.layer_breakdown if lb["soil_type"] == "rock"]
    assert rock["effective_top_m"] == 9.5
    c = _rock_choice(r)
    assert c["how"] == "given" and "8-9.5 m" in c["chosen"]
    assert ("not considered part of the socket for design purposes"
            in c["basis"])
    assert any("casing_depth" in a for a in c["alternatives"])


def test_casing_to_the_same_depth_also_takes_the_soil():
    r = _run(fractured_rock_thickness=1.5)
    cased = _run(casing_depth=9.5)
    assert cased.Q_side_rock == pytest.approx(r.Q_side_rock)
    assert cased.Q_side_clay == 0.0 < r.Q_side_clay


def test_the_zone_runs_on_through_a_second_rock_layer():
    r = _run(_soil(1.0, 9.0), fractured_rock_thickness=1.5)
    upper, lower = [lb for lb in r.layer_breakdown
                    if lb["soil_type"] == "rock"]
    assert upper["method"] == "excluded"
    assert lower["effective_top_m"] == 9.5
    assert r.Q_side_rock == pytest.approx(FS_ROCK * math.pi * 1.2 * 4.5,
                                          rel=1e-6)


def test_a_zone_deeper_than_the_shaft_leaves_only_the_base():
    r = _run(fractured_rock_thickness=7.0)
    assert r.Q_side_rock == 0.0 and r.Q_tip == _run().Q_tip
    # the base is on rock the caller called fractured: said, not hidden
    c = _rock_choice(r)
    assert "tip (14 m) is within it" in c["chosen"]
    assert "qu and RQD as given" in c["chosen"]
    assert any("layer of its own" in a for a in c["alternatives"])
    # a zone ending at the tip leaves the base on the rock below it
    assert "within" not in _rock_choice(
        _run(fractured_rock_thickness=6.0))["chosen"]


def test_a_zone_below_the_tip_is_said_to_change_nothing():
    # GM-220 review: the shaft ends above the rock; before, the given
    # thickness changed nothing and the result said nothing about it
    r = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=7.0,
                         fractured_rock_thickness=1.5),
        soil=_soil()).compute()
    c = _rock_choice(r)
    assert c["how"] == "given"
    assert "below the shaft tip (7 m), so it changes nothing" in c["chosen"]
    assert r.Q_ultimate == DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=7.0), soil=_soil()
    ).compute().Q_ultimate


def test_with_no_rock_layer_it_is_refused():
    soil = ShaftSoilProfile(layers=[ShaftSoilLayer(20.0, "cohesive", 18.5,
                                                   cu=75.0)])
    with pytest.raises(ValueError, match="no rock layer"):
        _run(soil, fractured_rock_thickness=1.0)
    # and a soil profile states no choice about rock
    assert not any("rock" in c["what"] for c in _run(soil).choices)


def test_a_negative_thickness_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        DrillShaft(diameter=1.2, length=14.0, fractured_rock_thickness=-1.0)


def test_the_capacity_curve_carries_it():
    a = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=14.0,
                         fractured_rock_thickness=1.5), soil=_soil())
    by_depth = {round(p["depth_m"], 2): p for p in
                a.capacity_vs_depth(depth_min=9.0, depth_max=14.0,
                                    n_points=6)}
    def at_9(t):
        return DrillShaftAnalysis(
            shaft=DrillShaft(diameter=1.2, length=9.0,
                             fractured_rock_thickness=t),
            soil=_soil()).compute()
    # 1 m into rock, all of it inside the 1.5 m zone: the clay only
    assert by_depth[9.0]["Q_skin_kN"] == pytest.approx(
        at_9(1.5).Q_skin, rel=1e-3)
    assert at_9(1.5).Q_side_rock == 0.0 < at_9(0.0).Q_side_rock
    assert by_depth[14.0]["Q_skin_kN"] == pytest.approx(
        a.compute().Q_skin, rel=1e-3)


def test_the_calc_steps_print_the_zone():
    a = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=14.0,
                         fractured_rock_thickness=1.5), soil=_soil())
    r = a.compute()
    inputs = {i.name: i.value for i in get_input_summary(r, a)}
    assert inputs["t_fr"] == 1.5
    steps = [s for sec in get_calc_steps(r, a) for s in sec.items]
    (zone,) = [s for s in steps if getattr(s, "title", "")
               == "Rock Not Considered Part of the Socket"]
    assert zone.result_value == "9.50" and "p. 8-15" in zone.reference
    (rock,) = [s for s in steps if "Rock Socket" in getattr(s, "title", "")
               and "Layer" in s.title]
    assert "not considered part of the socket" in rock.notes
    # the default prints no such step
    a0 = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=14.0),
                            soil=_soil())
    assert not [s for sec in get_calc_steps(a0.compute(), a0)
                for s in sec.items
                if "Not Considered" in getattr(s, "title", "")]
