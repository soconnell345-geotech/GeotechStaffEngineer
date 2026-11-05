"""GM-296: water standing on the excavated side of a sheet pile wall.

UFC 3-220-20 Fig. 4-41 (p. 242): "Water levels can be different on
opposite sides due to pumping, tidal fluctuations and other reasons".
Its example, Fig. 4-43 (p. 245), has water 5 ft below the top on both
sides of a 15 ft wall, read off the chart Fig. 4-42 (p. 243), which takes
"the level ... the same on both sides" and gamma = 124.8 pcf = 2 gamma'.
Before GM-296 a passive-side level above the dredge line was taken AT the
dredge line. Test values are made up except the print's own example.
"""

import math

import pytest

from geotech_common.choices import call_inputs
from sheet_pile.cantilever import (
    WallSoilLayer, analyze_cantilever, passive_water_level,
    passive_water_pressure,
)
from sheet_pile.anchored import analyze_anchored
from sheet_pile.earth_pressure import rankine_Ka, rankine_Kp

FT = 0.3048
PCF = 0.157087


def _sand(phi=32.0, gamma=19.0):
    return [WallSoilLayer(thickness=30.0, unit_weight=gamma,
                          friction_angle=phi)]


# ── the level the walls take ──────────────────────────────────────────

def test_left_out_is_the_excavation_level_or_the_deeper_active_level():
    assert passive_water_level(None, None, 5.0) is None
    assert passive_water_level(1.0, None, 5.0) == 5.0
    assert passive_water_level(7.0, None, 5.0) == 7.0
    assert passive_water_level(1.0, 2.0, 5.0) == 2.0


def test_water_pressure_on_the_excavated_face():
    gw = 10.0
    # standing 3 m deep in front of a 5 m excavation
    assert passive_water_pressure(1.0, 5.0, 2.0, gw) == 0.0
    assert passive_water_pressure(4.0, 5.0, 2.0, gw) == pytest.approx(20.0)
    assert passive_water_pressure(7.0, 5.0, 2.0, gw) == pytest.approx(50.0)
    # at or below the dredge line: none above it
    assert passive_water_pressure(4.0, 5.0, 5.0, gw) == 0.0
    assert passive_water_pressure(7.0, 5.0, 6.0, gw) == pytest.approx(10.0)
    assert passive_water_pressure(7.0, 5.0, None, gw) == 0.0


# ── calls that did not move ───────────────────────────────────────────

@pytest.mark.parametrize("gwt_a", [None, 0.0, 2.0, 5.0, 6.5])
def test_left_out_is_the_same_as_before(gwt_a):
    """Left out, the excavated side's level is the excavation level or the
    deeper active level: the same answer as giving it."""
    H = 5.0
    left = analyze_cantilever(H, _sand(), gwt_depth_active=gwt_a)
    given = analyze_cantilever(
        H, _sand(), gwt_depth_active=gwt_a,
        gwt_depth_passive=None if gwt_a is None else max(gwt_a, H))
    assert left.embedment_depth == given.embedment_depth
    assert left.max_moment == given.max_moment
    a_left = analyze_anchored(H, 1.0, _sand(), gwt_depth_active=gwt_a)
    a_given = analyze_anchored(
        H, 1.0, _sand(), gwt_depth_active=gwt_a,
        gwt_depth_passive=None if gwt_a is None else max(gwt_a, H))
    assert a_left.embedment_depth == a_given.embedment_depth
    assert a_left.anchor_force == a_given.anchor_force


def test_pinned_values_before_gm296_unchanged():
    """Two walls run on master before GM-296 (water behind at 1 m, left
    out in front): the numbers did not move."""
    c = analyze_cantilever(5.0, _sand(), gwt_depth_active=1.0)
    assert c.embedment_depth == pytest.approx(16.2567, abs=1e-4)
    assert c.max_moment == pytest.approx(1561.87, abs=0.01)
    a = analyze_anchored(5.0, 1.0, _sand(), gwt_depth_active=1.0)
    assert a.embedment_depth == pytest.approx(6.4276, abs=1e-4)
    assert a.anchor_force == pytest.approx(136.11, abs=0.01)
    assert a.max_moment == pytest.approx(349.48, abs=0.01)


# ── balanced water ────────────────────────────────────────────────────

def test_balanced_water_has_no_net_water_pressure():
    H, hw = 5.0, 2.0
    c = analyze_cantilever(H, _sand(), gwt_depth_active=hw,
                           gwt_depth_passive=hw)
    assert all(abs(row[3]) < 1e-9 for row in c.profile)
    a = analyze_anchored(H, 1.0, _sand(), gwt_depth_active=hw,
                         gwt_depth_passive=hw)
    assert all(abs(row[3]) < 1e-9 for row in a.profile)


def test_water_in_front_shortens_the_wall():
    H = 5.0
    before = analyze_cantilever(H, _sand(), gwt_depth_active=2.0,
                                gwt_depth_passive=H)
    balanced = analyze_cantilever(H, _sand(), gwt_depth_active=2.0,
                                  gwt_depth_passive=2.0)
    # before GM-296 a level above the dredge line was taken at it
    assert before.embedment_depth == pytest.approx(14.478, abs=1e-3)
    assert balanced.embedment_depth == pytest.approx(8.606, abs=2e-3)
    assert balanced.max_moment < before.max_moment
    a0 = analyze_anchored(H, 1.0, _sand(), gwt_depth_active=2.0,
                          gwt_depth_passive=H)
    a1 = analyze_anchored(H, 1.0, _sand(), gwt_depth_active=2.0,
                          gwt_depth_passive=2.0)
    assert a1.anchor_force < a0.anchor_force


def test_net_water_between_two_levels():
    """Water 2 m deep behind and 3 m deep in front of a 5 m excavation:
    the net water pressure is gamma_w x 1 m from the front level down."""
    gw = 9.81
    c = analyze_cantilever(5.0, _sand(), gwt_depth_active=2.0,
                           gwt_depth_passive=3.0)
    for row in c.profile:
        z, water = row[0], row[3]
        want = gw * (max(0.0, z - 2.0) - max(0.0, z - 3.0))
        assert water == pytest.approx(want, abs=1e-9)


# ── UFC 3-220-20 Fig. 4-43 ────────────────────────────────────────────

def _fig_4_43(gwt_passive):
    """Fig. 4-43 on the chart's basis (Fig. 4-42: gamma = 124.8 pcf =
    2 gamma'), Kp / Ka = 14.4 as printed (Rankine phi' 35.65, Kp already
    the allowable one, so FOS_passive 1.0)."""
    H, hw = 15 * FT, 5 * FT
    phi = 2 * math.degrees(math.atan(14.4 ** 0.25)) - 90
    assert rankine_Kp(phi) / rankine_Ka(phi) == pytest.approx(14.4)
    layers = [WallSoilLayer(thickness=40.0, unit_weight=124.8 * PCF,
                            friction_angle=phi)]
    r = analyze_cantilever(H, layers, gwt_depth_active=hw,
                           gwt_depth_passive=gwt_passive, FOS_passive=1.0,
                           gamma_w=62.4 * PCF)
    return r, H, rankine_Ka(phi)


def test_fig_4_43_reproduced_on_the_simplified_method():
    """Print: D/H = 0.95 off the chart (UFC's full method, Fig. 4-41),
    D_design = 1.2 x 14.25 = 17 ft, M/(gamma' Ka H^3) = 0.6. The module is
    the Caltrans Simplified Method (moments about O; GM-290 P4), which
    gives a shorter D0 than the full method: 0.868 H, D = 15.6 ft (-8 %);
    the moment ratio 0.565 (-6 %). Before GM-296 the water in front was
    left out: D0/H 1.46, moment ratio 1.87."""
    r, H, Ka = _fig_4_43(5 * FT)
    assert r.embedment_converged / H == pytest.approx(0.868, abs=0.005)
    assert r.embedment_depth / FT == pytest.approx(15.6, abs=0.1)
    ratio = r.max_moment / (62.4 * PCF * Ka * H ** 3)
    assert ratio == pytest.approx(0.565, abs=0.01)
    assert 0.9 < r.embedment_depth / (17 * FT) < 0.95
    old, _, _ = _fig_4_43(15 * FT)      # the water at the dredge line
    assert old.embedment_converged / H == pytest.approx(1.461, abs=0.005)


# ── what the result says ──────────────────────────────────────────────

def test_choices_and_defaults_used_say_the_level_taken():
    with call_inputs({"excavation_depth", "layers", "gwt_depth_active"},
                     {"gwt_depth_passive"}) as rec:
        r = analyze_cantilever(5.0, _sand(), gwt_depth_active=2.0)
    entry = [c for c in r.choices if "excavated side" in c["what"]]
    assert entry and entry[0]["chosen"] == 5.0
    assert entry[0]["how"] == "default"
    assert rec.took["gwt_depth_passive"] == 5.0
    r = analyze_cantilever(5.0, _sand(), gwt_depth_active=2.0,
                           gwt_depth_passive=2.0)
    entry = [c for c in r.choices if "excavated side" in c["what"]][0]
    assert entry["how"] == "given" and "above the dredge line" in entry["basis"]
    assert not [c for c in analyze_cantilever(5.0, _sand()).choices
                if "excavated side" in c["what"]]
