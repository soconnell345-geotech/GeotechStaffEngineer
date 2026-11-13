"""GM-321: a braced wall's supports use their spacing and inclination.

GEC-4 (FHWA-IF-99-015) Eqs. 19 to 21, pp. 66 and 67: Th = Thi s,
T = Th / cos(theta), Tv = T sin(theta); the Caltrans Trenching and Shoring
Manual's Example 8-1 does the same two steps (Eqs. 8-4-21 and 8-4-22,
pp. 8-15, 8-16). Made-up values, except the first test, which is GEC-4
App. A Design Example 1's printed numbers (p. A-7).
"""

import math

import pytest

from soe import (ExcavationGeometry, SOEWallLayer, SupportLevel,
                 analyze_braced_excavation)
from soe.beam_analysis import support_own_loads


def _geometry(supports):
    return ExcavationGeometry(
        excavation_depth=7.0,
        soil_layers=[SOEWallLayer(thickness=16.0, unit_weight=18.5,
                                  friction_angle=33.0, soil_type="sand")],
        support_levels=supports, surcharge=12.0)


def test_gec4_example_1_design_loads():
    # GEC-4 App. A Design Example 1: TH1 = 168 kN/m, anchors at 2.5 m and
    # 15 deg, DL1 = 435 kN (the numbers soe's FHWA function is pinned to)
    own = support_own_loads(168.0, 2.5, 15.0)
    assert own["horizontal_load_kN_per_support"] == pytest.approx(420.0)
    assert own["axial_load_kN_per_support"] == pytest.approx(434.8, abs=0.1)
    assert own["vertical_load_kN_per_support"] == pytest.approx(
        434.82 * math.sin(math.radians(15.0)), abs=0.05)


def test_caltrans_example_8_1_form_by_hand():
    # Caltrans Eq. 8-4-21 (p. 8-15): the horizontal load per foot of wall
    # x the anchor spacing = T_H; Eq. 8-4-22 (p. 8-16): T = T_H / cos 15.
    # Made-up values: 50 kN/m, supports at 3 m, 15 deg.
    #   T_H = 50 x 3 = 150 kN
    #   T   = 150 / cos 15 = 150 / 0.965926 = 155.29 kN
    #   T_v = 155.29 x sin 15 = 155.29 x 0.258819 = 40.19 kN
    own = support_own_loads(50.0, 3.0, 15.0)
    assert own["horizontal_load_kN_per_support"] == 150.0
    assert own["axial_load_kN_per_support"] == pytest.approx(155.29, abs=0.005)
    assert own["vertical_load_kN_per_support"] == pytest.approx(40.19,
                                                                abs=0.005)
    assert own["axial_load_kN_per_m"] == pytest.approx(51.76, abs=0.005)


def test_inclined_support_note_says_the_wall_must_carry_the_vertical():
    flat = analyze_braced_excavation(_geometry(
        [SupportLevel(depth=2.0, spacing=3.0)]))
    assert not any("vertical component" in n for n in flat.notes)
    tilted = analyze_braced_excavation(_geometry(
        [SupportLevel(depth=2.0, spacing=3.0),
         SupportLevel(depth=5.0, spacing=3.0, angle_deg=15.0)]))
    note = [n for n in tilted.notes if "vertical component" in n][0]
    assert "Support(s) 2 inclined" in note
    assert "reported, not applied" in note
    assert "does not check" in note
    assert "Sec. 11-3.04" in note and "A-13" in note


def test_spacing_and_angle_reach_the_result():
    plain = analyze_braced_excavation(_geometry(
        [SupportLevel(depth=2.0), SupportLevel(depth=5.0)]))
    used = analyze_braced_excavation(_geometry(
        [SupportLevel(depth=2.0, spacing=3.0),
         SupportLevel(depth=5.0, support_type="anchor", spacing=2.4,
                      angle_deg=20.0)]))
    for a, b in zip(plain.support_reactions, used.support_reactions):
        # the wall analysis per metre does not move
        assert a["load_kN_per_m"] == b["load_kN_per_m"]
        assert a["horizontal_load_kN_per_support"] is None
        assert a["axial_load_kN_per_support"] is None
        assert a["axial_load_kN_per_m"] == a["load_kN_per_m"]
    r1, r2 = used.support_reactions
    assert r1["horizontal_load_kN_per_support"] == pytest.approx(
        r1["load_kN_per_m"] * 3.0, abs=0.05)
    assert r1["axial_load_kN_per_support"] == \
        r1["horizontal_load_kN_per_support"]
    assert r1["vertical_load_kN_per_support"] == 0.0
    c = math.cos(math.radians(20.0))
    assert r2["axial_load_kN_per_m"] == pytest.approx(
        r2["load_kN_per_m"] / c, abs=0.05)
    assert r2["axial_load_kN_per_support"] == pytest.approx(
        r2["load_kN_per_m"] * 2.4 / c, abs=0.05)
    assert r2["vertical_load_kN_per_support"] == pytest.approx(
        r2["load_kN_per_m"] * 2.4 * math.tan(math.radians(20.0)), abs=0.05)
    assert (r2["spacing_m"], r2["angle_deg"]) == (2.4, 20.0)


def test_result_says_what_the_loads_are():
    plain = analyze_braced_excavation(_geometry([SupportLevel(depth=2.0)]))
    assert any("No spacing given for support(s) 1" in n for n in plain.notes)
    entry = [c for c in plain.choices
             if c["what"].startswith("each support's own load")][0]
    assert entry["how"] == "default" and "per metre of wall only" in \
        entry["chosen"]
    used = analyze_braced_excavation(_geometry(
        [SupportLevel(depth=2.0, spacing=3.0, angle_deg=10.0)]))
    entry = [c for c in used.choices
             if c["what"].startswith("each support's own load")][0]
    assert entry["how"] == "given"
    assert "spacing 3 m, inclined 10 deg" in entry["chosen"]
    assert "GEC-4 Eqs. 19 to 21" in entry["basis"]
    assert "kN axial per support at 3 m, 10 deg" in used.summary()


def test_load_not_found_gives_no_support_load():
    own = support_own_loads(None, 3.0, 10.0)
    assert own["axial_load_kN_per_m"] is None
    assert own["axial_load_kN_per_support"] is None


@pytest.mark.parametrize("kw, text", [
    ({"spacing": 0.0}, "spacing must be positive"),
    ({"spacing": -2.0}, "spacing must be positive"),
    ({"angle_deg": 90.0}, "less than 90"),
    ({"angle_deg": -5.0}, "less than 90"),
])
def test_bad_spacing_or_angle_refused(kw, text):
    with pytest.raises(ValueError, match=text):
        SupportLevel(depth=2.0, **kw)


@pytest.mark.parametrize("kw, text", [
    ({"angle_deg": "15"}, "angle_deg must be a number, got '15'"),
    ({"spacing": "3"}, "spacing must be a number, got '3'"),
    ({"spacing": True}, "spacing must be a number, got True"),
])
def test_spacing_or_angle_not_a_number_refused(kw, text):
    with pytest.raises(ValueError, match=text):
        SupportLevel(depth=2.0, **kw)


def test_null_angle_is_horizontal():
    assert SupportLevel(depth=2.0, angle_deg=None).angle_deg == 0.0


def test_preload_is_not_a_support_input():
    with pytest.raises(TypeError):
        SupportLevel(depth=2.0, preload_kN_per_m=50.0)
