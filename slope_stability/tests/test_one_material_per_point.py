"""One material per point (GM-140 step 1; the account's departure D1).

At any point below the ground there is one layer, and a slice's weight,
the strength at its base and everything else taken from the column come
from that one lookup (``SlopeGeometry.column_at`` / ``layer_at_point``).
EM 1110-2-1902 2-2d (p. 2-5): total unit weights for all soils, one weight
per point; 4-2d(2) (p. 4-2): the engineer checks the weights of slices.

All values here are made up.
"""

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.slices import build_slices

# A fill over a clay; their boundary rises 1 in 5 from (0, -2) to (40, 6).
_SURF = [(0.0, 0.0), (10.0, 0.0), (30.0, 10.0), (40.0, 10.0)]
_BOUNDARY = [(0.0, -2.0), (40.0, 6.0)]
_G_FILL, _G_CLAY = 20.0, 16.0
_SLIP = [(10.0, 0.0), (18.0, -3.0), (30.0, -1.0), (38.0, 10.0)]


def _layers(lower_top="flat"):
    """The usual description of a sloping boundary: the upper layer has a
    polyline bottom and the lower one a flat top at the boundary's highest
    point (``lower_top='flat'``), or the lower layer's top is the same
    polyline (``'polyline'``)."""
    clay = dict(name="clay", top_elevation=6.0, bottom_elevation=-10.0,
                gamma=_G_CLAY, cu=30.0, analysis_mode="undrained")
    if lower_top == "polyline":
        clay["top_boundary_points"] = _BOUNDARY
    return [
        SlopeSoilLayer(name="fill", top_elevation=10.0, bottom_elevation=6.0,
                       gamma=_G_FILL, phi=30.0, c_prime=5.0,
                       bottom_boundary_points=_BOUNDARY),
        SlopeSoilLayer(**clay),
    ]


def _geom(lower_top="flat"):
    return SlopeGeometry(surface_points=_SURF, soil_layers=_layers(lower_top))


def _ground(x):
    if x <= 10.0:
        return 0.0
    if x >= 30.0:
        return 10.0
    return 0.5 * (x - 10.0)


def _hand_weight(s):
    """Weight of a slice by hand: fill from the ground down to the boundary
    (or the base, where the base is in the fill), clay from the boundary
    down to the base. The ground, the boundary and the base are straight
    across every slice here, so the mid-point column is exact."""
    b = -2.0 + 0.2 * s.x_mid
    g = _ground(s.x_mid)
    fill = g - max(b, s.z_base)
    clay = max(0.0, min(b, g) - s.z_base)
    return s.width * (_G_FILL * fill + _G_CLAY * clay)


@pytest.mark.parametrize("lower_top", ["flat", "polyline"])
@pytest.mark.parametrize("n", [12, 30, 77])
def test_two_layer_sloping_boundary_slice_weights(lower_top, n):
    slip = PolylineSlipSurface(points=_SLIP)
    slices = build_slices(_geom(lower_top), slip, n)
    for s in slices:
        assert s.weight == pytest.approx(_hand_weight(s), rel=1e-9, abs=1e-9)


def test_two_layer_one_slice_by_hand():
    """The slice from x = 24 to 25 (ground break, corners and crossings put
    boundaries at 10, 18, 30 and where the base crosses the boundary; 28
    slices put one-metre slices between 18 and 30).

    At x = 24.5: ground 0.5 (24.5 - 10) = 7.25; boundary -2 + 0.2 (24.5) =
    2.90; base -3 + (24.5 - 18) (2 / 12) = -1.9167. Fill 7.25 - 2.90 = 4.35 m
    at 20 = 87.0; clay 2.90 + 1.9167 = 4.8167 m at 16 = 77.07; W = 164.07
    kN/m for a 1 m slice. Counting the clay from its flat top (6.0) as well
    gives 87.0 + (6.0 + 1.9167) 16 = 213.67."""
    slip = PolylineSlipSurface(points=_SLIP)
    slices = build_slices(_geom(), slip, 28)
    s = next(s for s in slices if abs(s.x_mid - 24.5) < 1e-9)
    assert s.width == pytest.approx(1.0)
    assert s.z_top == pytest.approx(7.25)
    assert s.z_base == pytest.approx(-1.916667, abs=1e-6)
    assert s.weight == pytest.approx(164.07, abs=0.01)


def test_two_layer_total_weight_is_area_times_unit_weight():
    """The sliding mass by its areas. Polygon of the mass: (10,0) (18,-3)
    (30,-1) (38,10) (30,10); area 180.0 m2 by the shoelace rule. The
    boundary z = -2 + 0.2 x cuts it from (10, 0) to (34.2553, 4.8511)
    (where it meets the last leg of the surface), leaving 86.638 m2 of
    clay below it and 93.362 m2 of fill above: 20 (93.362) + 16 (86.638)
    = 3,253.4 kN/m."""
    def shoelace(p):
        return 0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1]
                             - p[(i + 1) % len(p)][0] * p[i][1]
                             for i in range(len(p))))

    # last leg: z = -1 + (11/8)(x - 30) meets z = -2 + 0.2 x
    xc = (-1.0 - 30.0 * 11.0 / 8.0 + 2.0) / (0.2 - 11.0 / 8.0)
    zc = -2.0 + 0.2 * xc
    mass = [(10.0, 0.0), (18.0, -3.0), (30.0, -1.0), (38.0, 10.0),
            (30.0, 10.0)]
    clay = [(10.0, 0.0), (18.0, -3.0), (30.0, -1.0), (xc, zc)]
    a_clay = shoelace(clay)
    a_fill = shoelace(mass) - a_clay
    assert shoelace(mass) == pytest.approx(180.0)
    assert a_clay == pytest.approx(86.638, abs=1e-3)
    slip = PolylineSlipSurface(points=_SLIP)
    slices = build_slices(_geom(), slip, 40)
    assert sum(s.weight for s in slices) == pytest.approx(
        _G_FILL * a_fill + _G_CLAY * a_clay, rel=1e-9)


def test_strength_and_weight_take_the_same_layer():
    """A base in the clay takes the clay's strength; one in the fill, the
    fill's (the same lookup as the weight)."""
    geom = _geom()
    slip = PolylineSlipSurface(points=_SLIP)
    for s in build_slices(geom, slip, 30):
        in_clay = s.z_base < -2.0 + 0.2 * s.x_mid
        assert (s.c, s.phi) == ((30.0, 0.0) if in_clay else (5.0, 30.0))
        assert geom.layer_at_point(s.x_mid, s.z_base).name == (
            "clay" if in_clay else "fill")


def test_column_at_gives_one_layer_per_point():
    geom = _geom()
    col = geom.column_at(20.0)       # ground 5.0, boundary 2.0, base -10
    assert [(L.name, zb, zt) for L, zb, zt in col] == [
        ("fill", pytest.approx(2.0), pytest.approx(5.0)),
        ("clay", pytest.approx(-10.0), pytest.approx(2.0)),
    ]


# --- overlaps: the first listed layer holds the ground, and the result says


def test_overlap_reported_with_the_flat_top():
    over = _geom("flat").layer_overlaps()
    assert len(over) == 1
    o = over[0]
    assert o["layer"] == "clay" and o["held_by"] == ["fill"]
    # the clay's flat top (6.0) is above the boundary from x = 0 to 40,
    # by up to 6 - (-2) = 8 m at x = 0; under the ground it is 2 m at most
    # there (ground 0, boundary -2).
    assert o["x_from"] == pytest.approx(0.0)
    assert o["x_to"] == pytest.approx(40.0)
    assert o["max_overlap_m"] > 0.0


def test_no_overlap_with_a_polyline_top_or_flat_layers():
    assert _geom("polyline").layer_overlaps() == []
    flat = SlopeGeometry(
        surface_points=[(0.0, 5.0), (10.0, 5.0), (20.0, 10.0), (30.0, 10.0)],
        soil_layers=[
            SlopeSoilLayer("a", top_elevation=10.0, bottom_elevation=4.0,
                           gamma=18.0, phi=30.0),
            SlopeSoilLayer("b", top_elevation=4.0, bottom_elevation=-5.0,
                           gamma=19.0, phi=25.0)])
    assert flat.layer_overlaps() == []


def test_analysis_result_reports_the_overlap():
    from slope_stability.analysis import analyze_slope
    res = analyze_slope(_geom("flat"), xc=18.0, yc=22.0, radius=24.0,
                        method="bishop")
    assert res.layer_overlaps and res.layer_overlaps[0]["layer"] == "clay"
    assert "layer_overlaps" in res.to_dict()
    assert "first listed" in res.summary()
    res2 = analyze_slope(_geom("polyline"), xc=18.0, yc=22.0, radius=24.0,
                         method="bishop")
    assert res2.layer_overlaps == []
    assert "layer_overlaps" not in res2.to_dict()
    # the same ground; the flat top adds one slice boundary where the
    # circle crosses el. 6.0, so the two differ by the slicing only
    assert res2.FOS == pytest.approx(res.FOS, rel=1e-5)


def test_a_lens_listed_first_is_weighed_once():
    """A thin layer listed before the layer it sits in holds its own
    ground; the host layer is weighed around it."""
    geom = SlopeGeometry(
        surface_points=[(0.0, 10.0), (40.0, 10.0)],
        soil_layers=[
            SlopeSoilLayer("lens", top_elevation=6.0, bottom_elevation=5.0,
                           gamma=12.0, cu=10.0, analysis_mode="undrained",
                           top_boundary_points=[(10.0, 6.0), (30.0, 8.0)],
                           bottom_boundary_points=[(10.0, 5.0), (30.0, 7.0)]),
            SlopeSoilLayer("host", top_elevation=10.0, bottom_elevation=0.0,
                           gamma=20.0, phi=30.0)])
    col = geom.column_at(20.0)       # lens 6 to 7
    assert sorted((L.name, round(zb, 9), round(zt, 9)) for L, zb, zt in col) \
        == [("host", 0.0, 6.0), ("host", 7.0, 10.0), ("lens", 6.0, 7.0)]
    assert geom.layer_at_point(20.0, 6.5).name == "lens"
    assert geom.layer_at_point(20.0, 8.0).name == "host"
    over = geom.layer_overlaps()
    assert over and over[0]["layer"] == "host" and over[0]["held_by"] == ["lens"]


# --- gaps: refused, with the location


def test_gap_between_layers_refused_with_location():
    with pytest.raises(ValueError) as e:
        SlopeGeometry(
            surface_points=[(0.0, 10.0), (40.0, 10.0)],
            soil_layers=[
                SlopeSoilLayer("a", top_elevation=10.0, bottom_elevation=6.0,
                               gamma=18.0, phi=30.0),
                SlopeSoilLayer("b", top_elevation=4.0, bottom_elevation=0.0,
                               gamma=19.0, phi=25.0)])
    msg = str(e.value)
    assert "gap" in msg and "4.00" in msg and "6.00" in msg


def test_gap_under_a_sloping_top_refused_with_location():
    """The lower layer's top dips below the upper layer's flat bottom from
    x = 20 on: a wedge of ground with no layer."""
    with pytest.raises(ValueError) as e:
        SlopeGeometry(
            surface_points=[(0.0, 10.0), (40.0, 10.0)],
            soil_layers=[
                SlopeSoilLayer("a", top_elevation=10.0, bottom_elevation=5.0,
                               gamma=18.0, phi=30.0),
                SlopeSoilLayer("b", top_elevation=5.0, bottom_elevation=0.0,
                               gamma=19.0, phi=25.0,
                               top_boundary_points=[(0.0, 5.0), (20.0, 5.0),
                                                    (40.0, 3.0)])])
    msg = str(e.value)
    assert "gap" in msg and "x = 20.00" in msg and "40.00" in msg


def test_ground_above_the_top_layer_refused():
    with pytest.raises(ValueError, match="gap"):
        SlopeGeometry(
            surface_points=[(0.0, 5.0), (10.0, 5.0), (20.0, 12.0),
                            (30.0, 12.0)],
            soil_layers=[SlopeSoilLayer("a", top_elevation=10.0,
                                        bottom_elevation=0.0, gamma=18.0,
                                        phi=30.0)])


# --- V-028 (ACADS 4, Slide2 #9): the slice the account measured


def test_v028_slice_weight_is_gamma_h_b():
    """Every layer of V-028 weighs 18.84 kN/m3, so each slice weighs
    18.84 h b. The slice from x = 47.9746 to 48.5348 on the pinned
    surface: ground 28 + (12/25)(48.2547 - 43) = 30.5222, base 26.8149,
    h = 3.7073 m, b = 0.5602 m, W = 39.13 kN/m (the module read 116.0
    with each layer weighed from its flat top at el. 40). The whole mass:
    3,003 kN/m from its area (8,768 before)."""
    from validation_examples.test_published_v028_slope import (
        _v028_geom, _V028_SURFACE)
    slip = PolylineSlipSurface(points=_V028_SURFACE)
    slices = build_slices(_v028_geom(), slip, 50)
    s = next(s for s in slices if s.x_left < 48.25 < s.x_right)
    assert s.width == pytest.approx(0.5602, abs=1e-4)
    assert s.height == pytest.approx(3.7073, abs=1e-4)
    assert s.weight == pytest.approx(39.13, abs=0.01)
    for t in slices:
        assert t.weight == pytest.approx(18.84 * t.height * t.width, rel=1e-9)
    assert sum(t.weight for t in slices) == pytest.approx(3003.0, abs=1.0)
