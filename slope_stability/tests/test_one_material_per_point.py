"""One material per point (GM-140 step 1; the account's departure D1).

At any point below the ground there is one layer, and a slice's weight,
the strength at its base and everything else taken from the column come
from that one lookup (``SlopeGeometry.column_at`` / ``layer_at_point``).
EM 1110-2-1902 Appendix F, F-5d (p. F-19): slices containing two soils
"were divided into two portions for calculating the slice weights", each
portion at its own soil's total unit weight (2-2d, p. 2-5); 4-2d(2) (p.
4-2): the engineer checks the weights of slices.

All values here are made up, except EM 1110-2-1902's example near the
end (Figs. F-10 and F-11, printed inputs and answer, cited there) and
V-028's (Slide2 problem #9).
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
    down to the base. In this section, on this surface, the ground, the
    boundary and the base are straight across every slice (the slices
    break at the ground's and the surface's corners and where the base
    crosses the boundary), so the mid-point column is exact here; it is
    not exact for a slice with a corner of a layer boundary inside it."""
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


# --- the weak-layer search aims at a layer where its polylines put it

# A 10 m slope at 1 on 2, crest on the left. A weak layer 1 m thick dips
# with the slope, its mid-line from el. 13 at x = 0 to el. 5 at x = 50.
# Every layer carries the flat elevations of the whole section (20 and -5),
# as a section built from boundaries has them: far from the weak layer.
_WL_SURF = [(0.0, 20.0), (10.0, 20.0), (30.0, 10.0), (50.0, 10.0)]
_WL_TOP = [(0.0, 13.5), (50.0, 5.5)]
_WL_BOT = [(0.0, 12.5), (50.0, 4.5)]


def _weak_layer_section(weak_top_polyline):
    """The weak layer given both its polylines (``True``), or its bottom
    only, the layer listed before it holding everything above (``False``)."""
    weak = dict(name="weak", top_elevation=20.0, bottom_elevation=-5.0,
                gamma=19.0, phi=10.0, c_prime=0.0,
                bottom_boundary_points=_WL_BOT)
    if weak_top_polyline:
        weak["top_boundary_points"] = _WL_TOP
    return SlopeGeometry(surface_points=_WL_SURF, soil_layers=[
        SlopeSoilLayer(name="upper", top_elevation=20.0, bottom_elevation=-5.0,
                       gamma=19.0, phi=30.0, c_prime=20.0,
                       bottom_boundary_points=_WL_TOP),
        SlopeSoilLayer(**weak),
        SlopeSoilLayer(name="lower", top_elevation=20.0, bottom_elevation=-5.0,
                       gamma=19.0, phi=30.0, c_prime=20.0)])


def _share_of_base_in_weak_layer(points, n=400):
    """Share of the surface's horizontal length whose base lies between
    the weak layer's bottom (12.5 - 0.16 x) and top (13.5 - 0.16 x)."""
    x0, x1 = points[0][0], points[-1][0]
    inside = 0
    for i in range(n):
        x = x0 + (i + 0.5) * (x1 - x0) / n
        for (xa, za), (xb, zb) in zip(points[:-1], points[1:]):
            if xa <= x <= xb:
                z = za + (zb - za) * (x - xa) / (xb - xa)
                break
        if 12.5 - 0.16 * x <= z <= 13.5 - 0.16 * x:
            inside += 1
    return inside / n


@pytest.mark.parametrize("weak_top_polyline", [True, False])
def test_weak_layer_search_runs_in_a_sloping_layer(weak_top_polyline):
    """The weak-layer search draws its trial surfaces toward the target
    layer. The layer is where its top and bottom polylines put it (and,
    with no top polyline, below what the layer listed before it holds),
    not at its flat top_elevation and bottom_elevation: the critical
    surface runs in the weak layer over most of its length. With 8 points
    the two end legs go from the ground to the layer, so five-sevenths of
    the length is the most that can lie in it."""
    from slope_stability.search import search_weak_layer_biased
    res = search_weak_layer_biased(
        _weak_layer_section(weak_top_polyline), x_entry_range=(0.0, 12.0),
        x_exit_range=(30.0, 48.0), n_trials=300, n_points=8, n_slices=20,
        seed=1)
    assert res.critical is not None
    assert _share_of_base_in_weak_layer(res.critical.slip_points) > 0.5


def test_weak_layer_search_band_at_x():
    """The band the weak-layer search aims at, at x = 25 of the section
    above (weak layer from 8.5 to 9.5; ground at 12.5): each layer's own
    ground, the top layer's flat top not cut at the ground. For a layer
    with a flat top and bottom, its top_elevation and bottom_elevation
    themselves."""
    from slope_stability.search import _layer_band_at
    for weak_top_polyline in (True, False):
        geom = _weak_layer_section(weak_top_polyline)
        upper, weak, lower = geom.soil_layers
        assert _layer_band_at(geom, upper, 25.0) == pytest.approx((9.5, 20.0))
        assert _layer_band_at(geom, weak, 25.0) == pytest.approx((8.5, 9.5))
        assert _layer_band_at(geom, lower, 25.0) == pytest.approx((-5.0, 8.5))
    a = SlopeSoilLayer("a", top_elevation=10.0, bottom_elevation=2.0,
                       gamma=19.0, phi=35.0, c_prime=20.0)
    b = SlopeSoilLayer("b", top_elevation=2.0, bottom_elevation=-10.0,
                       gamma=17.0, phi=10.0, c_prime=5.0)
    flat = SlopeGeometry(
        surface_points=[(0.0, 10.0), (10.0, 10.0), (30.0, 0.0), (50.0, 0.0)],
        soil_layers=[a, b])
    assert _layer_band_at(flat, a, 40.0) == (2.0, 10.0)
    assert _layer_band_at(flat, b, 40.0) == (-10.0, 2.0)


# --- EM 1110-2-1902 Figs. F-10 and F-11: two soils on a level boundary

# The manual's units, converted here.
_FT = 0.3048                            # m
_LBF = 4.4482216152605e-3               # kN
_PCF = _LBF / _FT ** 3                  # kN/m3 per lb/ft3
_PSF = _LBF / _FT ** 2                  # kPa per lb/ft2
_KIPS_PER_FT = 1000.0 * _LBF / _FT      # kN/m per kip/ft
_EM_XC, _EM_YC, _EM_R = 101.0, 259.0, 278.0     # ft (Fig. F-10a)


def _em_f10(foundation_pcf=127.0):
    """EM 1110-2-1902 Fig. F-10 (p. F-20), end of construction. Toe at El.
    0; 3 on 1 to El. 58, 2.5 on 1 to El. 91, level crest; foundation
    surface level at El. 0. Circle: centre 101 ft from the toe and 259 ft
    above El. 0, R = 278 ft. Embankment 135 pcf, c = 1,780 psf, phi = 5
    deg; foundation 127 pcf, c = 1,600 psf, phi = 2 deg. No water."""
    from slope_stability.slip_surface import CircularSlipSurface
    ground_ft = [(-100.0, 0.0), (0.0, 0.0), (174.0, 58.0), (256.5, 91.0),
                 (420.0, 91.0)]
    geom = SlopeGeometry(
        surface_points=[(x * _FT, z * _FT) for x, z in ground_ft],
        soil_layers=[
            SlopeSoilLayer(name="embankment", top_elevation=91.0 * _FT,
                           bottom_elevation=0.0, gamma=135.0 * _PCF,
                           phi=5.0, c_prime=1780.0 * _PSF),
            SlopeSoilLayer(name="foundation", top_elevation=0.0,
                           bottom_elevation=-60.0 * _FT,
                           gamma=foundation_pcf * _PCF, phi=2.0,
                           c_prime=1600.0 * _PSF)])
    slip = CircularSlipSurface(xc=_EM_XC * _FT, yc=_EM_YC * _FT,
                               radius=_EM_R * _FT)
    return geom, slip


def test_em_1902_fig_f11_bishop_two_soils_level_boundary():
    """EM 1110-2-1902 Appendix F, F-5d (pp. F-19, F-20) and Figs. F-10 and
    F-11 (pp. F-20, F-21): the Simplified Bishop Method on an embankment
    over a foundation soil, printed F = 1.33. Slices 6 to 10 hold both
    soils and "were divided into two portions for calculating the slice
    weights" (F-5d): a printed check of slice weights by soil. The
    boundary is level, so it does not exercise a sloping boundary and
    reads the same as before GM-140 step 1."""
    from slope_stability.methods import bishop_fos
    geom, slip = _em_f10()
    fos = bishop_fos(build_slices(geom, slip, 30), slip)
    fos = fos[0] if isinstance(fos, tuple) else fos
    # printed F = 1.33: EM 1110-2-1902 Fig. F-11, columns 14 and 15 (sums
    # 698 / 524), p. F-21
    assert fos == pytest.approx(1.33, rel=0.01)


def test_em_1902_fig_f10_weight_in_each_soil():
    """The weight in each soil of Fig. F-10's sliding mass against its
    exact areas (ft2). Foundation: the segment of the circle below El. 0,
    R^2 acos(d/R) - d sqrt(R^2 - d^2) with d = 259. Embankment: the area
    under the ground from the toe to where the circle leaves the crest,
    less the area between El. 0 and the arc where the arc is above El. 0.
    The module's weight in each soil is separated by weighing the mass a
    second time with the foundation's unit weight doubled.

    Two things on the printed page, for which Fig. F-11's total weight
    (column 7 adds to 2,022 kips) is not asserted: column 3 prints the
    slice widths against the slice numbers in reverse order (57 ft is
    slice 10's, at the toe; 17 ft slice 1's), and the printed slices end
    about 5 ft short of where the stated circle leaves the crest (the
    printed widths add to 318 ft; the circle leaves the crest 322.5 ft
    from the toe). The page also takes each slice's area from its mean
    height. The exact areas give 1,724 + 327 = 2,051 kips."""
    import math
    r, d = _EM_R, _EM_YC
    half = math.sqrt(r * r - d * d)                 # 101.01 ft
    a_foundation = r * r * math.acos(d / r) - d * half
    x_out = _EM_XC + math.sqrt(r * r - (d - 91.0) ** 2)      # 322.50 ft
    under_ground = (0.5 * 174.0 * 58.0 + 0.5 * (58.0 + 91.0) * 82.5
                    + 91.0 * (x_out - 256.5))

    def under_root(u):          # integral of sqrt(R^2 - u^2)
        return 0.5 * u * math.sqrt(r * r - u * u) + 0.5 * r * r * math.asin(u / r)

    u0, u1 = half, x_out - _EM_XC
    below_arc = d * (u1 - u0) - (under_root(u1) - under_root(u0))
    a_embankment = under_ground - below_arc
    assert a_foundation == pytest.approx(2577.0, abs=1.0)
    assert a_embankment == pytest.approx(12771.0, abs=1.0)

    geom, slip = _em_f10()
    w = sum(s.weight for s in build_slices(geom, slip, 30))
    geom2, slip2 = _em_f10(foundation_pcf=254.0)
    w2 = sum(s.weight for s in build_slices(geom2, slip2, 30))
    kips_foundation = (w2 - w) / _KIPS_PER_FT
    kips_embankment = w / _KIPS_PER_FT - kips_foundation
    assert kips_foundation == pytest.approx(127.0 * a_foundation / 1000.0,
                                            rel=0.005)
    assert kips_embankment == pytest.approx(135.0 * a_embankment / 1000.0,
                                            rel=0.005)


def test_thin_gap_message_shows_its_thickness():
    """A gap 4 mm thick is printed with decimals enough to see it, with
    its thickness and how to close it."""
    with pytest.raises(ValueError) as e:
        SlopeGeometry(
            surface_points=[(0.0, 12.0), (40.0, 12.0)],
            soil_layers=[
                SlopeSoilLayer("a", top_elevation=12.0, bottom_elevation=10.004,
                               gamma=18.0, phi=30.0),
                SlopeSoilLayer("b", top_elevation=10.0, bottom_elevation=0.0,
                               gamma=19.0, phi=25.0)])
    msg = str(e.value)
    assert "gap 0.0040 m thick" in msg
    assert "10.0000 and 10.0040 m" in msg
    assert "top_elevation" in msg and "top_boundary_points" in msg
