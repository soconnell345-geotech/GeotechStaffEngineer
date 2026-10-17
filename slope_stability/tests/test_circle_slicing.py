"""GM-134: where a circle is sliced, and the slice edges GM-131 left out.

A circle gets a slice boundary at every ground break and every crossing of
a layer boundary, the water table or a tension crack's base, and at each
edge of a surcharge, as a polyline does (GM-131, `test_slice_boundaries.py`);
a polyline now gets the surcharge and crack edges too. Each slice takes the
angle and length of the chord between its boundaries. Cut evenly, a
circle's slice across a crossing took one material, one pore pressure or
one load for its whole base, and the FOS jumped with the slice count: the
critical circle of Slide2 #39 (Tandjiria 2002, sand fill) read 1.246,
1.190, 1.180 and 1.232 on 25, 29, 30 and 31 slices.
"""

import math

import pytest

import slope_stability.rapid_drawdown as rd
from slope_stability.analysis import analyze_slope
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slices import build_slices, slice_boundaries
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)

FT, PSF, PCF = 0.3048, 0.04788, 0.157087

# A 5 m embankment, 2:1 face, a stronger crust over a weaker layer at el 6.
_SURFACE = [(0.0, 10.0), (10.0, 10.0), (20.0, 5.0), (35.0, 5.0)]
# Enters the crest at x = 5.608, leaves the bench at x = 20.796, lowest
# point el 4 at x = 16.
_CIRCLE = (16.0, 16.0, 12.0)


def _geom(**kw):
    return SlopeGeometry(
        surface_points=_SURFACE,
        soil_layers=[
            SlopeSoilLayer(name="crust", top_elevation=10.0,
                           bottom_elevation=6.0, gamma=19.0, phi=30.0,
                           c_prime=5.0),
            SlopeSoilLayer(name="clay", top_elevation=6.0,
                           bottom_elevation=-5.0, gamma=18.0, phi=0.0,
                           c_prime=20.0),
        ], **kw)


def _left_crossing(z, circle=_CIRCLE):
    """Where the lower arc of ``circle`` is at elevation ``z``, left side."""
    xc, yc, r = circle
    return xc - math.sqrt(r * r - (z - yc) ** 2)


def _has(bounds, x, tol=1e-9):
    return any(abs(b - x) <= tol for b in bounds)


def _edges(slices):
    return [slices[0].x_left] + [s.x_right for s in slices]


def test_a_circle_is_cut_at_a_layer_crossing():
    """The circle crosses the top of the clay at x = 9.367: a slice boundary
    there, and at the crest and toe breaks, so each slice's base lies in one
    layer."""
    g = _geom()
    slip = CircularSlipSurface(*_CIRCLE)
    bounds = slice_boundaries(g, slip, 30)
    x6 = _left_crossing(6.0)
    assert x6 == pytest.approx(9.367, abs=1e-3)
    assert _has(bounds, x6) and _has(bounds, 10.0) and _has(bounds, 20.0)
    for a, b in zip(bounds, bounds[1:]):
        za, zb = slip.slip_elevation_at(a), slip.slip_elevation_at(b)
        assert (min(za, zb) >= 6.0 - 1e-9) or (max(za, zb) <= 6.0 + 1e-9)
    assert _edges(build_slices(g, slip, 30)) == pytest.approx(bounds)


def test_a_circle_is_cut_at_the_water_table():
    """A level water table at el 4.5, below the bench: the circle dips under
    it between x = 12.572 and 19.428, both slice boundaries, so each slice's
    base is wholly above or wholly below it (no pore pressure on the dry
    ones)."""
    g = _geom(gwt_points=[(0.0, 4.5), (35.0, 4.5)])
    slip = CircularSlipSurface(*_CIRCLE)
    bounds = slice_boundaries(g, slip, 30)
    half = math.sqrt(12.0 ** 2 - 11.5 ** 2)
    for x in (16.0 - half, 16.0 + half):
        assert _has(bounds, x)
    for s in build_slices(g, slip, 30):
        za = slip.slip_elevation_at(s.x_left)
        zb = slip.slip_elevation_at(s.x_right)
        assert (min(za, zb) >= 4.5 - 1e-9) or (max(za, zb) <= 4.5 + 1e-9)
        if min(za, zb) >= 4.5 - 1e-9:
            assert s.pore_pressure == pytest.approx(0.0, abs=1e-9)
        else:
            assert s.pore_pressure > 0.0


def test_a_circle_is_cut_at_a_surcharge_edge():
    """A 20 kPa strip from x = 6.5 to 8.5 on the crest: both edges are slice
    boundaries, so each slice carries the load over its whole top or not at
    all."""
    g = _geom(surcharge=20.0, surcharge_x_range=(6.5, 8.5))
    slip = CircularSlipSurface(*_CIRCLE)
    assert _has(slice_boundaries(g, slip, 30), 6.5)
    assert _has(slice_boundaries(g, slip, 30), 8.5)
    loaded = 0.0
    for s in build_slices(g, slip, 30):
        if 6.5 - 1e-9 <= s.x_left and s.x_right <= 8.5 + 1e-9:
            assert s.surcharge_force == pytest.approx(20.0 * s.width)
            loaded += s.width
        else:
            assert s.surcharge_force == pytest.approx(0.0, abs=1e-9)
    assert loaded == pytest.approx(2.0)


def test_a_circle_is_cut_at_the_tension_crack():
    """A 2 m crack in the crest: its base, el 8, meets the circle at
    x = 7.056, a slice boundary; the slices before it are in the crack and
    none after it."""
    g = _geom(tension_crack_depth=2.0)
    slip = CircularSlipSurface(*_CIRCLE)
    x8 = _left_crossing(8.0)
    assert x8 == pytest.approx(7.056, abs=1e-3)
    assert _has(slice_boundaries(g, slip, 30), x8)
    sl = build_slices(g, slip, 30)
    assert any(s.in_tension_crack for s in sl)
    for s in sl:
        assert s.in_tension_crack == (s.x_right <= x8 + 1e-9)


def test_a_polyline_is_cut_at_surcharge_and_crack_edges():
    """GM-134 adds the surcharge edges (x = 6 and 8) and the crack's base
    crossing (el 8, x = 6.857) to a polyline's boundaries, besides its
    corners (9, 16), the ground breaks (10, 20) and the top of the clay
    (x = 10.75)."""
    g = _geom(surcharge=20.0, surcharge_x_range=(6.0, 8.0),
              tension_crack_depth=2.0)
    slip = PolylineSlipSurface([(4.0, 10.0), (9.0, 6.5), (16.0, 4.5),
                                (21.0, 5.0)])
    bounds = slice_boundaries(g, slip, 30)
    for x in (6.0, 8.0, 4.0 + 5.0 * 2.0 / 3.5, 9.0, 16.0, 10.0, 20.0,
              9.0 + 0.5 * 7.0 / 2.0):
        assert _has(bounds, x), x
    sl = build_slices(g, slip, 30)
    for s in sl:
        if 6.0 - 1e-9 <= s.x_left and s.x_right <= 8.0 + 1e-9:
            assert s.surcharge_force == pytest.approx(20.0 * s.width)
        else:
            assert s.surcharge_force == pytest.approx(0.0, abs=1e-9)
        assert s.in_tension_crack == (s.x_right <= 6.857 + 1e-3)


def test_rapid_drawdown_slices_both_stages_alike(monkeypatch):
    """Each stage's water-table crossings are boundaries in the other, so
    the full-pool and drawn-down stages are cut at the same x and the stage-2
    strengths map slice for slice."""
    cuts = []
    real = rd.build_slices

    def spy(geom, slip, n, **kw):
        sl = real(geom, slip, n, **kw)
        cuts.append(_edges(sl))
        return sl

    monkeypatch.setattr(rd, "build_slices", spy)
    lay = SlopeSoilLayer(name="fill", top_elevation=110 * FT,
                         bottom_elevation=-5.0, gamma=135 * PCF, phi=30.0,
                         c_prime=0.0, R_c=1200 * PSF, R_phi=16.0)
    g = SlopeGeometry(surface_points=[(0.0, 0.0), (305 * FT, 110 * FT),
                                      (380 * FT, 110 * FT)],
                      soil_layers=[lay])
    r = rd.rapid_drawdown_fos(g, 80 * FT, 24 * FT, xc=169.5 * FT,
                              yc=210 * FT, radius=210 * FT,
                              method="duncan_3stage", n_slices=50)
    assert r.FOS > 0
    assert len(cuts) >= 2
    assert cuts[1] == pytest.approx(cuts[0])
    slip = CircularSlipSurface(169.5 * FT, 210 * FT, 210 * FT)
    for z in (80 * FT, 24 * FT):           # both pools' crossings
        x = _left_crossing(z, (169.5 * FT, 210 * FT, 210 * FT))
        if slip.find_entry_exit(g)[0] < x:
            assert _has(cuts[0], x, 1e-6), z


def _v043_sand():
    """Slide2 #39 (Tandjiria 2002), sand fill on soft clay, and its critical
    circle from an entry-exit search (validation V-043)."""
    g = SlopeGeometry(
        surface_points=[(0, 9), (10, 9), (20, 3), (30, 3)],
        soil_layers=[
            SlopeSoilLayer(name="fill", top_elevation=9, bottom_elevation=3,
                           c_prime=0.0, phi=37.0, gamma=17.0,
                           gamma_sat=17.0, analysis_mode="drained"),
            SlopeSoilLayer(name="softclay", top_elevation=3,
                           bottom_elevation=0.0, c_prime=20.0, phi=0.0,
                           gamma=20.0, gamma_sat=20.0, cu=20.0,
                           analysis_mode="undrained")])
    return g, (15.193905105454997, 10.242328996634253, 9.938523884815131)


def _v031_water():
    """Slide2 #57 (Pockoski & Duncan 2000, test slope 3): two layers and a
    water table the circle dips under (validation V-031)."""
    g = SlopeGeometry(
        surface_points=[(-70 * FT, 100 * FT), (0.0, 100 * FT),
                        (125 * FT, 150 * FT), (200 * FT, 150 * FT)],
        gwt_points=[(-70 * FT, 100 * FT), (0.0, 100 * FT),
                    (125 * FT, 140 * FT), (200 * FT, 140 * FT)],
        soil_layers=[
            SlopeSoilLayer(name="sandy_clay", top_elevation=150 * FT,
                           bottom_elevation=90 * FT, gamma=130 * PCF,
                           phi=35.0, c_prime=300 * PSF),
            SlopeSoilLayer(name="hp_clay", top_elevation=90 * FT,
                           bottom_elevation=85 * FT, gamma=130 * PCF,
                           phi=25.0, c_prime=0.0)])
    return g, (10.5, 60.3, 34.3)


@pytest.mark.parametrize("case, method, published, spread", [
    # Spencer 1.208-1.211 (1.180-1.246 before GM-134), Slide2 1.209
    ("v043", "spencer", 1.209, 0.004),
    # Bishop 1.420-1.421 and Spencer 1.423-1.425 (1.399-1.446 before),
    # Slide2 1.417 and 1.422
    ("v031", "bishop", 1.417, 0.002),
    ("v031", "spencer", 1.422, 0.003),
])
def test_the_fos_does_not_depend_on_the_slice_count(case, method, published,
                                                     spread):
    """On 25 to 45 slices the circle's FOS stays within a few thousandths,
    and within 0.5 % of the published value, with layers (V-043) and with a
    water table (V-031)."""
    g, (xc, yc, r) = _v043_sand() if case == "v043" else _v031_water()
    fos = [analyze_slope(g, method=method, n_slices=n, xc=xc, yc=yc,
                         radius=r).FOS for n in range(25, 46)]
    assert max(fos) - min(fos) < spread
    for f in fos:
        assert f == pytest.approx(published, rel=0.005)
