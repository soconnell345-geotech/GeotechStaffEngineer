"""GM-140 step 5a: the pore pressure on a slice base, as printed.

One function (``slope_stability.pore_pressure.base_pore_pressure``) gives
the pore pressure for every method, every family of surfaces and every
caller. All values here are made up.

Departures of the written account these tests reproduce (each failed on
the code before this step):

* D2  no cos^2 correction for a sloping water table (owner call OWN-104:
      built, off by default, the reading named in the result);
* D3  ru took the base layer's unit weight over the whole column;
* D4  pore pressure was subtracted on total-stress layers;
* D16 the pond's edge was not a slice boundary.
"""

import math

import pytest

from geotech_common.water import GAMMA_W
from slope_stability.analysis import (
    analyze_slope, compare_methods_table, infinite_slope_fos,
    rapid_drawdown_fos, search_critical_surface,
)
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.pore_pressure import (
    base_pore_pressure, pond_edges, water_line_slope, water_statements,
)
from slope_stability.slices import build_slices, slice_boundaries
from slope_stability.slip_surface import (
    CircularSlipSurface, PolylineSlipSurface,
)

# A 2H:1V face 10 m high between level ground, made up.
SURFACE = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
CIRCLE = dict(xc=24.0, yc=34.0, radius=25.0)
POLY = [(14.0, 10.0), (22.0, 7.0), (34.0, 9.0), (52.0, 20.0)]
DEEP = dict(xc=24.0, yc=34.0, radius=29.0)     # reaches el. 5


def _sand(**kw):
    d = dict(name="sand", top_elevation=20.0, bottom_elevation=-10.0,
             gamma=18.0, gamma_sat=20.0, phi=30.0, c_prime=5.0)
    d.update(kw)
    return SlopeSoilLayer(**d)


# The water line: level at el. 9 under the toe bench, rising 1 in 2 under
# the face (alpha_w = atan(0.5), cos^2 = 0.8), level at el. 17 behind.
SLOPING_LINE = [(0.0, 9.0), (22.0, 9.0), (38.0, 17.0), (70.0, 17.0)]
COS2 = 1.0 / (1.0 + 0.5 ** 2)          # cos^2(atan(1/2)) = 0.8


def _geom(line=SLOPING_LINE, layers=None, **kw):
    return SlopeGeometry(surface_points=SURFACE,
                         soil_layers=layers or [_sand()],
                         gwt_points=line, **kw)


# ---------------------------------------------------------------------------
# D2 / OWN-104: a sloping water line read both ways
# ---------------------------------------------------------------------------

class TestSlopingWaterLine:

    def test_slope_of_the_line_above_a_point(self):
        """alpha_w is the slope of the water table from the horizontal
        (FHWA NHI-06-088 Eq. 6-16, p. 6-14), taken on the stretch of the
        line above the point (UFC 3-220-10 Sec. 7-5.1, p. 355: "based on
        the slope of the surface above the specific point")."""
        g = _geom()
        assert water_line_slope(g, 10.0) == 0.0
        assert water_line_slope(g, 30.0) == pytest.approx(math.atan(0.5))
        assert water_line_slope(g, 50.0) == 0.0
        # at a vertex: the flatter of the two stretches (the larger u)
        assert water_line_slope(g, 22.0) == 0.0
        # beyond the line's ends it is carried on level
        assert water_line_slope(g, -5.0) == 0.0

    @pytest.mark.parametrize("slip", [
        CircularSlipSurface(**CIRCLE), PolylineSlipSurface(points=POLY)])
    def test_piezometric_line_is_the_full_head_and_never_corrected(
            self, slip):
        """Default (OWN-104): the line is a piezometric line, measured
        heads; u = gamma_w x the full vertical depth below it."""
        g = _geom()
        assert g.water_line_kind == "piezometric"
        wet = 0
        for s in build_slices(g, slip, 30):
            h = g.gwt_elevation_at(s.x_mid) - s.z_base
            if h > 0:
                wet += 1
                assert s.pore_pressure == pytest.approx(GAMMA_W * h)
                assert s.pore_pressure_from == "line"
            else:
                assert s.pore_pressure == 0.0
        assert wet > 5

    @pytest.mark.parametrize("slip", [
        CircularSlipSurface(**CIRCLE), PolylineSlipSurface(points=POLY)])
    def test_water_table_takes_cos2_under_its_sloping_stretch(self, slip):
        """"When the water table is sloping, use Equation 6-16 to
        calculate the water pressure on the base of the slice:
        u = h_w gamma_w cos^2(alpha_w)" (FHWA NHI-06-088 p. 6-14). Under
        the level stretches the two readings are the same."""
        g = _geom(water_line_kind="water_table")
        n_sloping = n_level = 0
        for s in build_slices(g, slip, 30):
            h = g.gwt_elevation_at(s.x_mid) - s.z_base
            if h <= 0:
                assert s.pore_pressure == 0.0
            elif 22.0 < s.x_mid < 38.0:
                n_sloping += 1
                assert s.pore_pressure == pytest.approx(GAMMA_W * h * COS2)
                assert s.pore_pressure_from == "line_cos2"
            else:
                n_level += 1
                assert s.pore_pressure == pytest.approx(GAMMA_W * h)
                assert s.pore_pressure_from == "line"
        assert n_sloping > 3 and n_level > 1

    def test_one_base_by_hand(self):
        """A base 4 m below a line rising 1 in 2: 9.81 x 4 = 39.24 kPa as
        a piezometric line, 39.24 x 0.8 = 31.392 kPa as a water table."""
        layer = _sand()
        z_line = 9.0 + 0.5 * (30.0 - 22.0)            # 13.0 at x = 30
        z_base = z_line - 4.0
        for kind, u in (("piezometric", 39.24), ("water_table", 31.392)):
            g = _geom(water_line_kind=kind)
            pp = base_pore_pressure(g, 30.0, z_base, 15.0, layer, 100.0)
            assert pp.u == pytest.approx(u, rel=1e-4)

    def test_no_correction_where_the_line_is_above_the_ground(self):
        """Ponded water is hydrostatic: where the line lies above the
        ground the head is not corrected, sloping or not."""
        # a line that rises over the toe bench, above the ground there
        line = [(0.0, 12.0), (20.0, 14.0), (70.0, 14.0)]
        g = _geom(line=line, water_line_kind="water_table")
        layer = g.soil_layers[0]
        z_top = g.ground_elevation_at(10.0)            # 10.0
        z_w = g.gwt_elevation_at(10.0)                 # 13.0 > ground
        pp = base_pore_pressure(g, 10.0, 6.0, z_top, layer, 80.0)
        assert pp.u == pytest.approx(GAMMA_W * (z_w - 6.0))
        assert pp.source == "line"

    def test_the_uncorrected_head_is_the_lower_factor_of_safety(self):
        slip = CircularSlipSurface(**CIRCLE)
        for method in ("fellenius", "bishop", "janbu", "spencer",
                       "morgenstern_price"):
            f_p = analyze_slope(_geom(), slip_surface=slip,
                                method=method).FOS
            f_w = analyze_slope(_geom(water_line_kind="water_table"),
                                slip_surface=slip, method=method).FOS
            assert f_p < f_w, method

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="water_line_kind"):
            _geom(water_line_kind="phreatic")


class TestInfiniteSlopeWithParallelSeepage:
    """The printed case for the correction. FHWA NHI-06-088 Sec. 6.3,
    pp. 6-7 to 6-8 (Figure 6-4, Eqs. 6-6 to 6-10): an infinite slope with
    seepage parallel to it and the water table at the surface carries the
    pore-water force U = gamma_w b h cos(beta) on a base of length
    b / cos(beta), that is u = gamma_w h cos^2(beta), and for c' = 0 "the
    FS for the dry material is reduced by the factor gamma'/gamma_sat"
    (Eq. 6-10): FS = (gamma'/gamma_sat) tan(phi') / tan(beta)."""

    BETA = math.atan(0.5)                  # 2H:1V, made up
    PHI, GSAT = 34.0, 19.5

    def _long_plane(self, kind, depth=0.5):
        # a face 400 m long in plan; the plane lies `depth` below it
        x0, z0, run = 50.0, 10.0, 400.0
        x1, z1 = x0 + run, z0 + 0.5 * run
        surface = [(0.0, z0), (x0, z0), (x1, z1), (x1 + 50.0, z1)]
        layer = SlopeSoilLayer(
            name="sand", top_elevation=z1, bottom_elevation=-50.0,
            gamma=self.GSAT, gamma_sat=self.GSAT, phi=self.PHI,
            c_prime=0.0)
        geom = SlopeGeometry(surface_points=surface, soil_layers=[layer],
                             gwt_points=list(surface),
                             water_line_kind=kind)
        e = 2.0                             # plan length of each end ramp
        xa, xb = x0 + 5.0, x1 - 5.0
        za = z0 + 0.5 * (xa - x0)
        zb = z0 + 0.5 * (xb - x0)
        slip = PolylineSlipSurface(points=[
            (xa, za), (xa + e, za + 0.5 * e - depth),
            (xb - e, zb - 0.5 * e - depth), (xb, zb)])
        return geom, slip

    def _printed(self):
        return ((self.GSAT - GAMMA_W) / self.GSAT
                * math.tan(math.radians(self.PHI)) / math.tan(self.BETA))

    def test_water_table_reproduces_eq_6_10(self):
        geom, slip = self._long_plane("water_table")
        res = analyze_slope(geom, slip_surface=slip, method="fellenius",
                            n_slices=60)
        # FHWA NHI-06-088 Eq. 6-10, p. 6-8: 0.6704 for these made-up
        # values; the two 2 m end ramps of a 390 m plane are the rest
        assert res.FOS == pytest.approx(self._printed(), rel=0.005)
        # and the module's closed form (same equation)
        closed = infinite_slope_fos(
            math.degrees(self.BETA), self.PHI, self.GSAT, depth=0.5,
            water_condition="seepage_parallel").FOS
        assert closed == pytest.approx(self._printed(), rel=1e-9)

    def test_the_base_pressure_is_eq_6_6(self):
        geom, slip = self._long_plane("water_table")
        mid = [s for s in build_slices(geom, slip, 60)
               if 100.0 < s.x_mid < 400.0]
        assert mid
        for s in mid:
            h = s.z_top - s.z_base
            assert s.pore_pressure == pytest.approx(
                GAMMA_W * h * math.cos(self.BETA) ** 2)

    def test_piezometric_reading_is_lower(self):
        """Read as a piezometric line the same input carries the full
        head, gamma_w h, on the plane: F = (gamma_sat cos^2(beta) -
        gamma_w) tan(phi') / (gamma_sat sin(beta) cos(beta)) by the same
        statics: (19.5 x 0.8 - 9.81) x tan 34 deg / (19.5 x 0.4) = 0.5007,
        against Eq. 6-10's 0.6704 here."""
        geom, slip = self._long_plane("piezometric")
        res = analyze_slope(geom, slip_surface=slip, method="fellenius",
                            n_slices=60)
        b = self.BETA
        full = ((self.GSAT * math.cos(b) ** 2 - GAMMA_W)
                * math.tan(math.radians(self.PHI))
                / (self.GSAT * math.sin(b) * math.cos(b)))
        assert res.FOS == pytest.approx(full, rel=0.005)
        assert res.FOS < self._printed()


# ---------------------------------------------------------------------------
# D3: ru in a layered column
# ---------------------------------------------------------------------------

class TestRuInALayeredColumn:
    """EM 1110-2-1902 C-7 (p. C-29) defines ru = u / (gamma z) for a
    uniform slope. In a layered column the module reads gamma z as the
    total overburden stress: the soil column above the base, each soil at
    its own unit weight (the column step 1 weighs)."""

    def _two_soils(self):
        fill = SlopeSoilLayer(name="fill", top_elevation=20.0,
                              bottom_elevation=12.0, gamma=16.0, phi=32.0)
        clay = SlopeSoilLayer(name="clay", top_elevation=12.0,
                              bottom_elevation=-10.0, gamma=20.0, phi=24.0,
                              c_prime=8.0, ru=0.3)
        return SlopeGeometry(surface_points=SURFACE,
                             soil_layers=[fill, clay])

    @pytest.mark.parametrize("slip", [
        CircularSlipSurface(**CIRCLE), PolylineSlipSurface(points=POLY)])
    def test_u_is_ru_times_the_overburden_stress(self, slip):
        g = self._two_soils()
        n = 0
        for s in build_slices(g, slip, 30):
            if s.z_base >= 12.0:
                assert s.pore_pressure == 0.0       # in the fill: no ru
                continue
            h_fill = max(s.z_top - 12.0, 0.0)
            h_clay = min(s.z_top, 12.0) - s.z_base
            sigma_v = 16.0 * h_fill + 20.0 * h_clay
            assert s.pore_pressure == pytest.approx(0.3 * sigma_v)
            assert s.pore_pressure_from == "ru"
            if h_fill > 0:
                n += 1
                # before: 0.3 x 20 x (h_fill + h_clay)
                assert s.pore_pressure < 0.3 * 20.0 * (h_fill + h_clay)
        assert n > 3

    def test_one_column_by_hand(self):
        """3 m of fill at 16 kN/m3 over 2 m of clay at 20 kN/m3, ru 0.3:
        u = 0.3 x (48 + 40) = 26.4 kPa (it was 0.3 x 20 x 5 = 30.0)."""
        g = self._two_soils()
        clay = g.soil_layers[1]
        pp = base_pore_pressure(g, 30.0, 10.0, 15.0, clay,
                                16.0 * 3.0 + 20.0 * 2.0)
        assert pp.u == pytest.approx(26.4)
        s = [s for s in build_slices(
            g, PolylineSlipSurface(points=[(18.0, 10.0), (24.0, 8.0),
                                           (38.0, 9.0), (50.0, 20.0)]), 30)
             if s.z_top > 12.0 and s.z_base < 12.0][0]
        assert s.pore_pressure == pytest.approx(
            0.3 * (16.0 * (s.z_top - 12.0) + 20.0 * (12.0 - s.z_base)))

    def test_a_surcharge_is_not_in_the_overburden(self):
        """gamma z is the soil's weight (EM C-7, p. C-29): a surcharge
        loads the slice but is not counted in ru's overburden."""
        g = self._two_soils()
        g2 = self._two_soils()
        g2.surcharge = 25.0
        slip = CircularSlipSurface(**CIRCLE)
        for a, b in zip(build_slices(g, slip, 30),
                        build_slices(g2, slip, 30)):
            assert a.pore_pressure == pytest.approx(b.pore_pressure)

    def test_the_result_says_how_ru_is_applied(self):
        res = analyze_slope(self._two_soils(), **CIRCLE)
        entry = [c for c in res.choices
                 if c["what"] == "pore pressure from ru"]
        assert len(entry) == 1
        assert "EM 1110-2-1902 C-7" in entry[0]["basis"]
        assert "module's reading" in entry[0]["basis"]

    def test_a_uniform_slope_is_unchanged(self):
        """In one soil the overburden stress is gamma z: ru x gamma x h,
        as EM C-7 prints."""
        g = SlopeGeometry(surface_points=SURFACE,
                          soil_layers=[_sand(ru=0.25)])
        for s in build_slices(g, CircularSlipSurface(**CIRCLE), 30):
            assert s.pore_pressure == pytest.approx(
                0.25 * 18.0 * (s.z_top - s.z_base))


# ---------------------------------------------------------------------------
# D4: no pore pressure on a total-stress base
# ---------------------------------------------------------------------------

class TestTotalStressLayers:
    """"For total stress analyses, pore water pressures are set to zero"
    (EM 1110-2-1902 Table 2-1, p. 2-2); "the pore water pressure should be
    set to zero" (2-2c, p. 2-4)."""

    LINE = [(0.0, 10.0), (20.0, 10.0), (40.0, 16.0), (70.0, 16.0)]

    def _geom(self, **clay):
        fill = _sand(name="fill", bottom_elevation=8.0)
        d = dict(name="clay", top_elevation=8.0, bottom_elevation=-10.0,
                 gamma=17.0, gamma_sat=17.0, cu=40.0,
                 analysis_mode="undrained")
        d.update(clay)
        return SlopeGeometry(surface_points=SURFACE,
                             soil_layers=[fill, SlopeSoilLayer(**d)],
                             gwt_points=self.LINE)

    @pytest.mark.parametrize("slip", [
        CircularSlipSurface(**DEEP), PolylineSlipSurface(points=POLY)])
    def test_no_u_on_an_undrained_base_below_the_water_line(self, slip):
        g = self._geom()
        n_clay = n_fill = 0
        for s in build_slices(g, slip, 30):
            head = g.gwt_elevation_at(s.x_mid) - s.z_base
            if s.z_base < 8.0:
                n_clay += 1
                assert head > 0
                assert s.pore_pressure == 0.0
                # the water is still there: only the base carries none
                assert s.water_pressure == pytest.approx(GAMMA_W * head)
            elif head > 0:
                n_fill += 1
                assert s.pore_pressure == pytest.approx(GAMMA_W * head)
        assert n_clay > 3 and n_fill > 0

    def test_shansep_and_anisotropic_layers_are_total_stress(self):
        for clay in (dict(strength_model="shansep", cu=0.0),
                     dict(strength_model="anisotropic", cu=0.0,
                          su_active=45.0, su_dss=35.0, su_passive=30.0)):
            g = self._geom(**clay)
            below = [s for s in build_slices(
                g, CircularSlipSurface(**DEEP), 30) if s.z_base < 8.0]
            assert below
            assert all(s.pore_pressure == 0.0 for s in below)
            assert all(s.water_pressure > 0.0 for s in below)

    def test_ru_and_a_grid_give_none_either(self):
        g = self._geom(ru=0.4)
        g.gwt_points = None
        below = [s for s in build_slices(
            g, CircularSlipSurface(**DEEP), 30) if s.z_base < 8.0]
        assert below and all(s.pore_pressure == 0.0 for s in below)
        g = self._geom()
        g.pore_pressure_points = [(0.0, -10.0, 150.0), (70.0, -10.0, 150.0),
                                  (0.0, 20.0, 150.0), (70.0, 20.0, 150.0)]
        sl = build_slices(g, CircularSlipSurface(**DEEP), 30)
        assert all(s.pore_pressure == 0.0 for s in sl if s.z_base < 8.0)
        assert all(s.pore_pressure == pytest.approx(150.0)
                   for s in sl if s.z_base > 8.0)

    def test_the_result_says_so(self):
        res = analyze_slope(self._geom(), **DEEP)
        entry = [c for c in res.choices if c["what"]
                 == "pore pressure on bases in total-stress layers"]
        assert len(entry) == 1
        assert "'clay'" in entry[0]["chosen"]
        assert "2-2c" in entry[0]["basis"]
        assert not any("Table 2-1" in w for w in res.warnings)

    def test_phi_given_to_a_total_stress_layer_under_water_warns(self):
        """Table 2-1, note 2 (EM 1110-2-1902 p. 2-2): total-stress
        envelopes with phi > 0 apply only to partially saturated soils."""
        res = analyze_slope(self._geom(phi=12.0), **DEEP)
        w = [w for w in res.warnings if "Table" in w]
        assert len(w) == 1
        assert "'clay'" in w[0] and "note 2" in w[0] and "p. 2-2" in w[0]
        assert "phi = 12" in w[0]

    def test_above_the_water_line_it_does_not_warn(self):
        g = self._geom(phi=12.0)
        g.gwt_points = [(0.0, -20.0), (70.0, -20.0)]
        res = analyze_slope(g, **DEEP)
        assert not any("Table 2-1" in w for w in res.warnings)
        assert not any("total-stress" in c["what"] for c in res.choices)


# ---------------------------------------------------------------------------
# D16: the pond's edge is a slice boundary
# ---------------------------------------------------------------------------

class TestPondEdge:
    """A pool at el. 14 over a 2H:1V face from (20, 10): the water meets
    the ground at x = 28 (FHWA NHI-06-088 p. 6-14 cuts the slices at the
    ground water level's intersections and at breaks in the slope)."""

    POOL = [(0.0, 14.0), (70.0, 14.0)]

    def test_where_the_pond_ends(self):
        assert pond_edges(_geom(line=self.POOL)) == pytest.approx([28.0])
        assert pond_edges(_geom(line=SLOPING_LINE)) == []
        assert pond_edges(_geom(line=None)) == []

    @pytest.mark.parametrize("slip", [
        CircularSlipSurface(**CIRCLE), PolylineSlipSurface(points=POLY)])
    @pytest.mark.parametrize("n", [20, 30, 47])
    def test_the_edge_is_a_slice_boundary(self, slip, n):
        g = _geom(line=self.POOL)
        xs = slice_boundaries(g, slip, n)
        assert any(abs(x - 28.0) < 1e-9 for x in xs)
        # so every slice's top is wholly under the pond or wholly out
        for s in build_slices(g, slip, n):
            d_l = 14.0 - g.ground_elevation_at(s.x_left)
            d_r = 14.0 - g.ground_elevation_at(s.x_right)
            assert not (d_l > 1e-9 and d_r < -1e-9)
            assert not (d_l < -1e-9 and d_r > 1e-9)

    def test_the_pond_weighs_what_it_holds(self):
        """With the edge cut, the slices' pond weights add up to the
        water over the surface, whatever the slice count: gamma_w x the
        area between the pool and the ground from the entry to x = 28."""
        g = _geom(line=self.POOL)
        slip = CircularSlipSurface(**CIRCLE)
        x_entry, _ = slip.find_entry_exit(g)
        assert x_entry < 20.0
        area = 4.0 * (20.0 - x_entry) + 0.5 * 4.0 * 8.0
        for n in (12, 20, 33):
            total = sum(s.pond_force for s in build_slices(g, slip, n))
            assert total == pytest.approx(GAMMA_W * area, rel=1e-9)


# ---------------------------------------------------------------------------
# The result names the reading (OWN-104)
# ---------------------------------------------------------------------------

_WHAT = "pore pressure under the sloping water line"


def _reading(choices):
    return [c for c in choices if c["what"].endswith(_WHAT)]


class TestTheResultNamesTheReading:

    def test_a_single_surface(self):
        res = analyze_slope(_geom(), **CIRCLE)
        (c,) = _reading(res.choices)
        assert "water_line_kind 'piezometric'" in c["chosen"]
        assert c["how"] == "default"
        assert "Eq. 6-16" in c["basis"] and "p. 355" in c["basis"]
        assert "lower factor of safety" in c["basis"]
        assert any("water_table" in a for a in c["alternatives"])
        assert res.to_dict()["choices"] == res.choices
        assert "sloping water line" in res.summary()

        res = analyze_slope(_geom(water_line_kind="water_table"), **CIRCLE)
        (c,) = _reading(res.choices)
        assert "cos^2" in c["chosen"]
        assert c["how"] == "given"
        assert "seepage parallel to a straight water table" in c["basis"]
        assert any("piezometric" in a for a in c["alternatives"])

    def test_a_level_line_needs_no_entry(self):
        level = [(0.0, 12.0), (70.0, 12.0)]
        for kind in ("piezometric", "water_table"):
            res = analyze_slope(_geom(line=level, water_line_kind=kind),
                                **CIRCLE)
            assert _reading(res.choices) == []
        assert analyze_slope(_geom(line=None), **CIRCLE).choices == []

    def test_a_level_line_reads_the_same_either_way(self):
        level = [(0.0, 12.0), (70.0, 12.0)]
        a = analyze_slope(_geom(line=level), **CIRCLE).FOS
        b = analyze_slope(_geom(line=level, water_line_kind="water_table"),
                          **CIRCLE).FOS
        assert a == b

    def test_the_comparison_table(self):
        out = compare_methods_table(_geom(), **CIRCLE)
        assert len(_reading(out["choices"])) == 1

    def test_a_search(self):
        res = search_critical_surface(
            _geom(water_line_kind="water_table"), nx=4, ny=4, n_slices=20)
        (c,) = _reading(res.choices)
        assert "cos^2" in c["chosen"]
        # said once, by the search
        assert _reading(res.critical.choices) == []
        assert len(_reading(res.to_dict()["choices"])) == 1

    def test_a_search_scores_every_surface_with_the_reading(self):
        kw = dict(nx=4, ny=4, n_slices=20, toe_circles=False,
                  infinite_slope=False)
        low = search_critical_surface(_geom(), **kw)
        high = search_critical_surface(
            _geom(water_line_kind="water_table"), **kw)
        assert low.critical.FOS < high.critical.FOS

    def test_a_grid_governs_the_base_and_says_so(self):
        g = _geom(water_line_kind="water_table")
        g.pore_pressure_points = [(0.0, -10.0, 60.0), (70.0, -10.0, 60.0),
                                  (0.0, 20.0, 60.0), (70.0, 20.0, 60.0)]
        res = analyze_slope(g, **CIRCLE)
        assert _reading(res.choices) == []
        (c,) = [c for c in res.choices
                if c["what"] == "pore pressure on slice bases"]
        assert "pore_pressure_points" in c["chosen"]
        for s in build_slices(g, CircularSlipSurface(**CIRCLE), 30):
            assert s.pore_pressure == pytest.approx(60.0)
            assert s.pore_pressure_from == "grid"

    def test_drawdown_stage_one_takes_the_same_function(self):
        """A sloping stage-1 seepage line is read as the geometry says,
        and the drawdown result names the reading."""
        core = SlopeSoilLayer(
            name="core", top_elevation=20.0, bottom_elevation=-10.0,
            gamma=19.0, gamma_sat=20.0, phi=30.0, c_prime=5.0,
            R_c=30.0, R_phi=16.0)
        seep = [(0.0, 18.0), (26.0, 18.0), (42.0, 12.0), (70.0, 12.0)]
        surface = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
        out = {}
        for kind in ("piezometric", "water_table"):
            g = SlopeGeometry(surface_points=surface, soil_layers=[core],
                              water_line_kind=kind)
            out[kind] = rapid_drawdown_fos(
                g, 18.0, 11.0, stage1_phreatic_points=seep, n_slices=30,
                **CIRCLE)
            (c,) = _reading(out[kind].choices)
            assert c["what"].startswith("Stage 1")
            assert len(_reading(out[kind].to_dict()["choices"])) == 1
        assert "piezometric" in _reading(
            out["piezometric"].choices)[0]["chosen"]
        assert "cos^2" in _reading(out["water_table"].choices)[0]["chosen"]
        # a lower stage-1 pore pressure: a higher stage-1 FOS
        assert (out["water_table"].stage1_fos
                > out["piezometric"].stage1_fos)
        # the pools are level: with a flat stage 1 nothing is named
        g = SlopeGeometry(surface_points=surface, soil_layers=[core])
        flat = rapid_drawdown_fos(g, 18.0, 11.0, n_slices=30, **CIRCLE)
        assert _reading(flat.choices) == []

    def test_a_line_given_after_drawdown_takes_the_same_reading(self):
        """A sloping line given after drawdown (drawdown_phreatic_points,
        GM-140 step 9) gives the bases of the later stages their pore
        pressure by the same function, read as the geometry says, and the
        result names that reading too. Made-up values."""
        core = SlopeSoilLayer(
            name="core", top_elevation=20.0, bottom_elevation=-10.0,
            gamma=19.0, gamma_sat=20.0, phi=30.0, c_prime=5.0,
            R_c=30.0, R_phi=16.0)
        after = [(0.0, 11.0), (22.0, 11.0), (42.0, 15.0), (70.0, 15.0)]
        surface = [(0.0, 10.0), (20.0, 10.0), (40.0, 20.0), (70.0, 20.0)]
        out = {}
        for kind in ("piezometric", "water_table"):
            g = SlopeGeometry(surface_points=surface, soil_layers=[core],
                              water_line_kind=kind)
            out[kind] = rapid_drawdown_fos(
                g, 18.0, 11.0, drawdown_phreatic_points=after,
                n_slices=30, **CIRCLE)
            (c,) = _reading(out[kind].choices)
            assert c["what"].startswith("After drawdown")
            assert len(_reading(out[kind].to_dict()["choices"])) == 1
        assert "piezometric" in _reading(
            out["piezometric"].choices)[0]["chosen"]
        assert "cos^2" in _reading(out["water_table"].choices)[0]["chosen"]
        # the line after drawdown is read as the geometry says: a lower
        # pore pressure after drawdown under 'water_table'
        assert (sum(out["water_table"].u_after)
                < sum(out["piezometric"].u_after))


def test_water_statements_without_water_are_empty():
    g = SlopeGeometry(surface_points=SURFACE, soil_layers=[_sand()])
    assert water_statements(g) == ([], [])


def test_a_water_line_at_the_ground_is_not_a_pond():
    """A sloping water line that lies ON the ground (a water table at the
    surface of the face) holds no ponded water: no slice carries a pond
    weight or a pond thrust. (The pool's depth at a slice's sides was
    taken from the line's elevation at the slice's middle, which gave
    each slice on the face a thrust into the slope.) Boundary water
    forces act where there is water outside the slope (FHWA NHI-06-088
    p. 6-13)."""
    g = _geom(line=list(SURFACE))
    for slip in (CircularSlipSurface(**CIRCLE),
                 PolylineSlipSurface(points=POLY)):
        for s in build_slices(g, slip, 30):
            assert s.pond_force == 0.0
            assert s.pond_hforce == pytest.approx(0.0, abs=1e-9)


def test_the_agents_tools_name_the_reading_and_take_the_option():
    """Through the agent's adapter: water_line_kind reaches the geometry,
    an unknown value is refused, and the yield acceleration, FOSM and
    Monte Carlo results (which carry no choices of their own) name the
    reading too."""
    from funhouse_agent.adapters.slope_stability import METHOD_REGISTRY
    layer = dict(name="sand", top_elevation=20.0, bottom_elevation=-10.0,
                 gamma=18.0, gamma_sat=20.0, phi=30.0, c_prime=5.0)
    base = dict(surface_points=[list(p) for p in SURFACE],
                soil_layers=[layer],
                gwt_points=[list(p) for p in SLOPING_LINE], **CIRCLE)
    low = METHOD_REGISTRY["analyze_slope"](dict(base))
    high = METHOD_REGISTRY["analyze_slope"](
        dict(base, water_line_kind="water_table"))
    assert low["FOS"] < high["FOS"]
    assert "piezometric" in _reading(low["choices"])[0]["chosen"]
    assert "cos^2" in _reading(high["choices"])[0]["chosen"]
    with pytest.raises(ValueError, match="water_line_kind"):
        METHOD_REGISTRY["analyze_slope"](
            dict(base, water_line_kind="phreatic"))
    variables = {"phi": {"cov": 0.1}}
    for name, extra in (
            ("yield_acceleration", {}),
            ("fosm_fos", {"variables": variables}),
            ("monte_carlo_fos", {"variables": variables, "n": 20,
                                 "seed": 1})):
        out = METHOD_REGISTRY[name](
            dict(base, water_line_kind="water_table", **extra))
        (c,) = _reading(out["choices"])
        assert "cos^2" in c["chosen"], name


class TestPondThrust:
    """The horizontal resultant of ponded water's pressure on the ground
    across a slice (boundary water forces with total weights, FHWA
    NHI-06-088 p. 6-13), from the pool's depth at the slice's sides."""

    def test_a_level_pool_is_as_before(self):
        """gamma_w (d_l^2 - d_r^2) / 2 at the centroid of the trapezoid on
        the projected face, the form used before this step."""
        from slope_stability.slices import _pond_thrust
        # ground rises 2 m across the slice under a pool 5 m deep at left
        f, z = _pond_thrust(5.0, 3.0, 10.0, 12.0)
        assert f == pytest.approx(0.5 * GAMMA_W * (25.0 - 9.0))
        depth = 2.0 * (125.0 - 27.0) / (3.0 * (25.0 - 9.0))
        assert z == pytest.approx(15.0 - depth)
        # the pool ends inside the slice: only the wetted rise is loaded
        f, z = _pond_thrust(1.0, -3.0, 10.0, 14.0)
        assert f == pytest.approx(0.5 * GAMMA_W * 1.0)
        assert z == pytest.approx(10.0 + 1.0 / 3.0)

    def test_a_sloping_line_above_the_ground(self):
        """Depth 2 m at the left and 1 m at the right over ground rising
        3 m: gamma_w x mean depth 1.5 m x the 3 m rise."""
        from slope_stability.slices import _pond_thrust
        f, _ = _pond_thrust(2.0, 1.0, 10.0, 13.0)
        assert f == pytest.approx(GAMMA_W * 1.5 * 3.0)

    def test_a_line_at_or_under_the_ground_gives_none(self):
        from slope_stability.slices import _pond_thrust
        assert _pond_thrust(0.0, 0.0, 10.0, 13.0)[0] == 0.0
        assert _pond_thrust(-0.5, -0.2, 10.0, 13.0)[0] == 0.0

    def test_a_water_line_at_the_ground_closed_form(self):
        """The thrust pinned as arithmetic (the step 5a review's case,
        ruling 1). Made-up values: a 2H:1V face (beta = atan 0.5, so
        sin(beta) cos(beta) = 0.4 and cos^2(beta) = 0.8), c' = 5 kPa,
        phi' = 34 deg, gamma = gamma_sat = 19.5 kN/m3, the water line ON
        the ground, read as a piezometric line (the default).

        Infinite slope, per metre of plan, the plane z = 2 m (vertical)
        below the ground: W = gamma_sat z; the base is 1 / cos(beta) long;
        u = gamma_w z (the full head under a line at the ground); no pond,
        as the line is nowhere above the ground. OMS (no interslice force):

            N' = W cos(beta) - u / cos(beta)
            F  = [c' / cos(beta) + N' tan(phi')] / (W sin(beta))
               = c' / (gamma_sat z sin(beta) cos(beta))
                 + (gamma_sat cos^2(beta) - gamma_w) tan(phi')
                   / (gamma_sat sin(beta) cos(beta))
               = 5 / 15.6 + (15.6 - 9.81) x 0.67451 / 7.8
               = 0.32051 + 0.50069 = 0.82121

        The finite plane runs 358 m in plan 2 m below the face, with a 2 m
        stretch from the ground at each end. The same statics summed over
        its three straight stretches (each exact by the trapezoid rule, the
        ground and the base being straight there) is what OMS must give:
        slices are cut at the surface's corners. The plane's own stretch is
        the infinite slope. On the steep head stretch (alpha = atan 1.5,
        gamma cos^2(alpha) = 6.0 < gamma_w) W cos(alpha) - u l is
        negative: OMS takes it as printed (GM-140 step 3, DESIGN.md D21),
        and the end stretches add 0.3 % (0.8238); with
        negative_friction_term='zero' it is held at zero and they add
        0.5 % (0.8253). Before step 5a each slice took a thrust from the
        line at its middle, a pool half the slice's rise deep at its lower
        side, and OMS read 0.914.
        """
        b = math.atan(0.5)
        c, tphi, gs, z = 5.0, math.tan(math.radians(34.0)), 19.5, 2.0
        sc = math.sin(b) * math.cos(b)
        closed = (c / (gs * z * sc)
                  + (gs * math.cos(b) ** 2 - GAMMA_W) * tphi / (gs * sc))
        assert closed == pytest.approx(0.82121, abs=1e-5)

        ground = [(-20.0, 0.0), (0.0, 0.0), (400.0, 200.0), (420.0, 200.0)]
        plane = [(20.0, 10.0), (22.0, 9.0), (380.0, 188.0), (382.0, 191.0)]
        face = SlopeSoilLayer(name="face", top_elevation=200.0,
                              bottom_elevation=-50.0, gamma=gs,
                              gamma_sat=gs, phi=34.0, c_prime=c)
        geom = SlopeGeometry(surface_points=ground, soil_layers=[face],
                             gwt_points=list(ground))
        res = analyze_slope(geom, slip_surface=PolylineSlipSurface(
            points=plane), method="fellenius", n_slices=60)
        res0 = analyze_slope(geom, slip_surface=PolylineSlipSurface(
            points=plane), method="fellenius", n_slices=60,
            negative_friction_term="zero")

        def stretch(xa, za, xb, zb, zero=False):
            # the ground (and the line) is z = x / 2 over the whole plane;
            # h, the depth of the base below it, is straight on a stretch
            dx = xb - xa
            alpha = math.atan2(zb - za, dx)
            h = 0.5 * ((0.5 * xa - za) + (0.5 * xb - zb))   # mean depth
            w = gs * h * dx
            length = dx / math.cos(alpha)
            u_force = GAMMA_W * h * length                  # u = gamma_w h
            # W cos(alpha) - u l as printed, negative on the steep head
            # stretch (gamma cos^2(alpha) < gamma_w); held at zero only
            # with negative_friction_term='zero' (D21)
            n_eff = w * math.cos(alpha) - u_force
            if zero:
                n_eff = max(n_eff, 0.0)
            return c * length + n_eff * tphi, w * math.sin(alpha)

        for held, result in ((False, res), (True, res0)):
            parts = [stretch(*plane[i], *plane[i + 1], zero=held)
                     for i in range(3)]
            assert parts[1][0] / parts[1][1] == pytest.approx(closed,
                                                              rel=1e-12)
            by_hand = sum(p[0] for p in parts) / sum(p[1] for p in parts)
            assert result.FOS == pytest.approx(by_hand, rel=1e-6)
        assert res.FOS < res0.FOS


def test_the_planar_check_follows_the_reading():
    """The infinite-slope check of a search reports the closed form with
    the water at the face's surface for the reading the run used, and
    names it (the step 5a review): 'water_table', (gamma'/gamma_sat)
    tan(phi')/tan(beta) (FHWA NHI-06-088 Eq. 6-10, p. 6-8); 'piezometric'
    (the default), the full head, (gamma_sat cos^2(beta) - gamma_w)
    tan(phi') / (gamma_sat sin(beta) cos(beta)). Made-up 2H:1V sand face,
    phi' = 34 deg, gamma_sat = 19.5: 1.3490 x 9.69 / 19.5 = 0.6704 and
    1.3490 x 5.79 / 15.6 = 0.5007."""
    from slope_stability.search_supplements import check_infinite_slope
    sand = _sand(gamma_sat=19.5, phi=34.0, c_prime=0.0)
    dry = math.tan(math.radians(34.0)) / 0.5
    want = {"water_table": dry * (19.5 - GAMMA_W) / 19.5,
            "piezometric": dry * (19.5 * COS2 - GAMMA_W) / (19.5 * COS2)}
    assert want["water_table"] == pytest.approx(0.6704, abs=1e-4)
    assert want["piezometric"] == pytest.approx(0.5007, abs=1e-4)
    for kind, value in want.items():
        g = _geom(line=list(SURFACE), layers=[sand], water_line_kind=kind)
        # windows that exclude the face: the closed forms only, no solve
        (chk,) = check_infinite_slope(
            g, x_entry_range=(-90.0, -80.0))["checks"]
        assert chk["seepage_reading"] == kind
        assert chk["FOS_seepage_closed_form"] == pytest.approx(value,
                                                               abs=1e-4)
        assert chk["FOS_dry_closed_form"] == pytest.approx(dry, abs=1e-4)
