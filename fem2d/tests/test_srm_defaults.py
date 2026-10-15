"""GM-88: the strength-reduction model's default margins over a deep
undrained foundation, the x_extend pair, and the undrained Poisson's ratio
warning. Fast: the helpers are checked directly; one coarse run checks the
pair reaches the mesh.

Background (fem2d/VALIDATION.md section 11): over undrained (phi = 0) soil
below the toe the critical circle passes below the toe, tangent to the firm
base, and reaches 1.2-1.8 times that soil's depth either side (Taylor 1948;
FHWA NHI-06-088 Figs. 6-14, 6-15). The old default carried the level ground
only 0.6 H past the profile, so the side rollers cut that circle and the
factor of safety read about 4 % low (Duncan Ex 1: 0.648 against 0.660 wide).
"""

import numpy as np
import pytest

from fem2d.analysis import (_deep_undrained_margins, _level_ground,
                            _undrained_nu_warnings, analyze_slope_srm)

H = 12.192
#: Duncan Ex 1's outline: 1 H of level ground behind the crest and beyond
#: the toe, a 2:1 slope.
SURF = np.array([(0, H), (H, H), (3 * H, 0), (4 * H, 0)], float)
BASE = 0.3 * 2 * H          # the old default margin each side at depth 2 H


def _clay(**kw):
    layer = {'name': 'clay', 'bottom_elevation': -H, 'E': 10000, 'nu': 0.49,
             'c': 28.73, 'phi': 0.0, 'gamma': 19.6}
    layer.update(kw)
    return layer


def test_level_ground_at_each_end():
    assert _level_ground(SURF) == pytest.approx((H, H))
    assert _level_ground(np.array([(0, 0), (10, 0)], float)) is None


def test_deep_undrained_soil_gets_twice_its_depth_each_side():
    # the last layer continues to the mesh base, 2 H below the toe
    left, right, depth = _deep_undrained_margins(
        SURF, [_clay()], z_toe=0.0, z_base=-2 * H, base_side=BASE)
    assert depth == pytest.approx(2 * H)
    # level ground H + margin = 2 x 2 H each side
    assert left == pytest.approx(3 * H) and right == pytest.approx(3 * H)


def test_no_change_where_it_is_not_needed():
    drained = _clay(phi=30.0, c=5.0)
    assert _deep_undrained_margins(SURF, [drained], 0.0, -2 * H, BASE) is None
    # undrained crust above the toe, drained soil below it
    crust = _clay(bottom_elevation=0.0)
    assert _deep_undrained_margins(SURF, [crust, drained], 0.0, -2 * H,
                                   BASE) is None
    # a profile that already carries 4 H of level ground each side
    wide = np.array([(0, H), (4 * H, H), (6 * H, 0), (10 * H, 0)], float)
    assert _deep_undrained_margins(wide, [_clay()], 0.0, -2 * H, BASE) is None


def test_a_thinner_undrained_layer_sets_its_own_depth():
    # 6 m of soft clay below the toe over a drained layer, on a profile with
    # only 0.5 m of level ground at each end
    short = np.array([(0, H), (0.5, H), (0.5 + 2 * H, 0), (1 + 2 * H, 0)],
                     float)
    soft = _clay(bottom_elevation=-6.0)
    drained = _clay(name='sand', phi=35.0, c=0.0)
    left, right, depth = _deep_undrained_margins(
        short, [soft, drained], 0.0, -2 * H, base_side=BASE)
    assert depth == pytest.approx(6.0)
    assert left == pytest.approx(12.0 - 0.5)
    assert right == pytest.approx(12.0 - 0.5)
    # 3 m of soft clay: twice it is inside the base margin, no change
    thin = _clay(bottom_elevation=-3.0)
    assert _deep_undrained_margins(short, [thin, drained], 0.0, -2 * H,
                                   BASE) is None


def test_undrained_layer_with_a_drained_poisson_ratio_is_named():
    w = _undrained_nu_warnings([_clay(nu=0.35)])
    assert len(w) == 1
    assert "'clay'" in w[0] and "0.35" in w[0] and "K0" in w[0]
    assert "0.54" in w[0] and "use 0.49" in w[0]
    assert _undrained_nu_warnings([_clay(nu=0.49)]) == []
    assert _undrained_nu_warnings([_clay(nu=0.3, phi=25.0, c=5.0)]) == []


def test_x_extend_as_a_pair():
    drained = [_clay(phi=30.0, c=10.0, nu=0.3, bottom_elevation=-H)]
    r = analyze_slope_srm([tuple(p) for p in SURF], drained, depth=5.0,
                          nx=10, ny=4, srf_tol=0.2, x_extend=(2.0, 7.0))
    assert r.nodes[:, 0].min() == pytest.approx(-2.0)
    assert r.nodes[:, 0].max() == pytest.approx(4 * H + 7.0)
    with pytest.raises(ValueError, match="pair"):
        analyze_slope_srm([tuple(p) for p in SURF], drained, x_extend=(1,))
    with pytest.raises(ValueError, match="negative"):
        analyze_slope_srm([tuple(p) for p in SURF], drained,
                          x_extend=(-1.0, 2.0))
