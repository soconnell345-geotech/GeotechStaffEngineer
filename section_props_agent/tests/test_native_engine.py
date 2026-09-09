"""Native section engine: exact integration, torsion, and the FD solver.

These tests anchor the engine against closed-form and published values
rather than against any library output — the polygon integrals are exact, so
they are asserted at machine precision, and the torsion constants are checked
against Roark's tabulated coefficients.
"""

import math

import pytest

from section_props_agent import analyze_polygon_section, analyze_section
from section_props_agent.polygon_props import (
    plastic_modulus_x, plastic_modulus_y, polygon_is_simple, region_properties,
    ring_integrals,
)
from section_props_agent.torsion import (
    chs_torsion, circle_torsion, i_section_torsion, i_section_warping,
    polygon_torsion_fd, rectangle_torsion, rhs_torsion,
)

RECT = [(0.0, 0.0), (300.0, 0.0), (300.0, 500.0), (0.0, 500.0)]


class TestExactIntegration:
    """Green's-theorem integrals are exact — assert them as such."""

    def test_rectangle_moments_are_exact(self):
        p = region_properties([(RECT, 1)])
        assert p["area"] == pytest.approx(300.0 * 500.0, rel=1e-14)
        assert p["cx"] == pytest.approx(150.0, rel=1e-14)
        assert p["cy"] == pytest.approx(250.0, rel=1e-14)
        assert p["ixx"] == pytest.approx(300.0 * 500.0 ** 3 / 12.0, rel=1e-14)
        assert p["iyy"] == pytest.approx(500.0 * 300.0 ** 3 / 12.0, rel=1e-14)
        assert p["ixy"] == 0.0

    def test_triangle_matches_closed_form(self):
        tri = [(0.0, 0.0), (300.0, 0.0), (0.0, 400.0)]
        p = region_properties([(tri, 1)])
        assert p["area"] == pytest.approx(0.5 * 300.0 * 400.0, rel=1e-14)
        assert p["cy"] == pytest.approx(400.0 / 3.0, rel=1e-13)
        # b*h^3/36 about the centroid
        assert p["ixx"] == pytest.approx(300.0 * 400.0 ** 3 / 36.0, rel=1e-13)

    def test_hole_is_subtracted(self):
        outer = [(0.0, 0.0), (200.0, 0.0), (200.0, 300.0), (0.0, 300.0)]
        inner = [(10.0, 10.0), (190.0, 10.0), (190.0, 290.0), (10.0, 290.0)]
        p = region_properties([(outer, 1), (inner, -1)])
        assert p["area"] == pytest.approx(200 * 300 - 180 * 280, rel=1e-13)
        assert p["ixx"] == pytest.approx(
            200 * 300 ** 3 / 12.0 - 180 * 280 ** 3 / 12.0, rel=1e-13)

    def test_ring_winding_does_not_matter(self):
        cw = list(reversed(RECT))
        assert region_properties([(cw, 1)])["area"] == pytest.approx(
            region_properties([(RECT, 1)])["area"], rel=1e-14)

    def test_signed_ring_integral_flips_with_winding(self):
        assert ring_integrals(RECT)[0] == pytest.approx(150000.0, rel=1e-14)
        assert ring_integrals(list(reversed(RECT)))[0] == pytest.approx(
            -150000.0, rel=1e-14)


class TestPlasticModuli:
    def test_rectangle_plastic_modulus(self):
        sx, y_pna = plastic_modulus_x([(RECT, 1)])
        assert sx == pytest.approx(300.0 * 500.0 ** 2 / 4.0, rel=1e-9)
        assert y_pna == pytest.approx(250.0, rel=1e-6)
        sy, x_pna = plastic_modulus_y([(RECT, 1)])
        assert sy == pytest.approx(500.0 * 300.0 ** 2 / 4.0, rel=1e-9)
        assert x_pna == pytest.approx(150.0, rel=1e-6)

    def test_plastic_na_halves_a_tee(self):
        """A tee's plastic axis sits away from its elastic centroid."""
        tee = [(0.0, 0.0), (200.0, 0.0), (200.0, 40.0), (120.0, 40.0),
               (120.0, 300.0), (80.0, 300.0), (80.0, 40.0), (0.0, 40.0)]
        area = region_properties([(tee, 1)])["area"]
        sx, y_pna = plastic_modulus_x([(tee, 1)])
        # the axis must split the area exactly in half
        below = 200.0 * min(y_pna, 40.0) + (
            40.0 * max(y_pna - 40.0, 0.0))
        assert below == pytest.approx(area / 2.0, rel=1e-6)
        assert sx > 0


class TestTorsionClosedForms:
    @pytest.mark.parametrize("ratio,k_roark", [
        (1.0, 0.1406), (1.5, 0.1958), (2.0, 0.229), (2.5, 0.249),
        (3.0, 0.263), (4.0, 0.281), (5.0, 0.291), (10.0, 0.312),
    ])
    def test_rectangle_series_matches_roark(self, ratio, k_roark):
        """J = K*a*b^3 against Roark's tabulated K (Table 10.1, case 4)."""
        b = 50.0
        a = ratio * b
        k = rectangle_torsion(b, a) / (a * b ** 3)
        assert k == pytest.approx(k_roark, rel=0.005)

    def test_circle_is_polar_moment(self):
        assert circle_torsion(200.0) == pytest.approx(
            math.pi * 200.0 ** 4 / 32.0, rel=1e-14)

    def test_chs_is_polar_moment(self):
        j = chs_torsion(400.0, 12.0)
        assert j == pytest.approx(
            math.pi * (400.0 ** 4 - 376.0 ** 4) / 32.0, rel=1e-14)

    def test_rhs_bredt_matches_hand_calc(self):
        d, b, t, r_out = 300.0, 200.0, 10.0, 20.0
        r_m = r_out - t / 2.0
        area_m = (b - t) * (d - t) - (4.0 - math.pi) * r_m ** 2
        perim_m = 2.0 * ((b - t) + (d - t)) - 8.0 * r_m + 2.0 * math.pi * r_m
        assert rhs_torsion(d, b, t, r_out) == pytest.approx(
            4.0 * area_m ** 2 * t / perim_m, rel=1e-12)

    def test_i_section_fillet_term_adds_stiffness(self):
        sharp = i_section_torsion(310.0, 165.0, 11.8, 6.6, r=0.0)
        filleted = i_section_torsion(310.0, 165.0, 11.8, 6.6, r=11.4)
        assert filleted > sharp
        # the fillet is worth ~10-15% on a rolled shape
        assert 1.05 < filleted / sharp < 1.30

    def test_i_section_warping_identity(self):
        """Cw = Iy * h0^2 / 4 for a doubly-symmetric shape."""
        d, b, t_f, t_w = 400.0, 180.0, 14.0, 9.0
        iy = 2.0 * (t_f * b ** 3 / 12.0) + (d - 2 * t_f) * t_w ** 3 / 12.0
        assert i_section_warping(d, b, t_f, t_w) == pytest.approx(
            iy * (d - t_f) ** 2 / 4.0, rel=1e-12)


class TestPolygonTorsionSolver:
    """The FD Prandtl solve is the one approximate path — pin its accuracy."""

    def test_matches_the_exact_rectangle_series(self):
        exact = rectangle_torsion(300.0, 500.0)
        j = polygon_torsion_fd(RECT, n_across=150)
        assert j == pytest.approx(exact, rel=0.002)

    def test_matches_a_square(self):
        sq = [(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]
        assert polygon_torsion_fd(sq, n_across=150) == pytest.approx(
            rectangle_torsion(200.0, 200.0), rel=0.002)

    def test_richardson_beats_the_raw_solve(self):
        exact = rectangle_torsion(300.0, 500.0)
        raw = polygon_torsion_fd(RECT, n_across=150, richardson=False)
        extrapolated = polygon_torsion_fd(RECT, n_across=150)
        assert abs(extrapolated - exact) < abs(raw - exact) / 10.0


class TestShapeResults:
    def test_circle_is_analytic(self):
        r = analyze_section("circle", d=400.0)
        assert r.area_mm2 == pytest.approx(math.pi * 400.0 ** 2 / 4.0, rel=1e-14)
        assert r.ixx_mm4 == pytest.approx(math.pi * 400.0 ** 4 / 64.0, rel=1e-14)
        assert r.sxx_mm3 == pytest.approx(400.0 ** 3 / 6.0, rel=1e-14)
        assert r.perimeter_mm == pytest.approx(math.pi * 400.0, rel=1e-14)
        assert r.gamma_mm6 == 0.0          # a circle does not warp

    def test_chs_area_is_the_annulus(self):
        r = analyze_section("chs", d=400.0, t=12.0)
        assert r.area_mm2 == pytest.approx(
            math.pi * (400.0 ** 2 - 376.0 ** 2) / 4.0, rel=1e-14)

    def test_rhs_area_matches_the_two_rings(self):
        r = analyze_section("rhs", d=300.0, b=200.0, t=10.0)
        # rounded corners: r_out defaults to 2t, r_in to r_out - t
        outer = 200 * 300 - (4 - math.pi) * 20.0 ** 2
        inner = 180 * 280 - (4 - math.pi) * 10.0 ** 2
        assert r.area_mm2 == pytest.approx(outer - inner, rel=1e-4)

    def test_i_section_area_and_ixx(self):
        r = analyze_section("i_section", d=400.0, b=180.0, t_f=14.0, t_w=9.0)
        area = 2 * 180 * 14 + (400 - 2 * 14) * 9
        assert r.area_mm2 == pytest.approx(area, rel=1e-12)

    def test_symmetric_shapes_have_zero_product_moment(self):
        """Round-off must not flip the principal-axis angle."""
        for kw in ({"shape": "rectangle", "d": 500.0, "b": 300.0},
                   {"shape": "rhs", "d": 300.0, "b": 200.0, "t": 10.0},
                   {"shape": "i_section", "d": 310.0, "b": 165.0,
                    "t_f": 11.8, "t_w": 6.6, "r": 11.4}):
            kw = dict(kw)
            r = analyze_section(kw.pop("shape"), **kw)
            assert r.ixy_mm4 == 0.0
            assert r.phi_deg == pytest.approx(0.0, abs=1e-9)

    def test_rotated_rectangle_principal_axes(self):
        """Rotating a section rotates its principal axes and nothing else.

        A 300x100 rectangle is stiffest about the y-axis, so unrotated its
        major (11) axis sits at -90 degrees. Turning the outline by +30
        degrees must carry that axis to -60 while leaving I11 and I22 at the
        unrotated values.
        """
        w, h, ang = 300.0, 100.0, math.radians(30.0)
        base = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        rot = [(x * math.cos(ang) - y * math.sin(ang),
                x * math.sin(ang) + y * math.cos(ang)) for x, y in base]
        flat = analyze_polygon_section(base, warping=False)
        assert flat.phi_deg == pytest.approx(-90.0, abs=1e-9)
        r = analyze_polygon_section(rot, warping=False)
        assert r.i11_mm4 == pytest.approx(h * w ** 3 / 12.0, rel=1e-10)
        assert r.i22_mm4 == pytest.approx(w * h ** 3 / 12.0, rel=1e-10)
        assert r.phi_deg == pytest.approx(-60.0, abs=1e-6)

    def test_warping_false_skips_the_solve(self):
        r = analyze_section("rectangle", d=500.0, b=300.0, warping=False)
        assert r.j_mm4 == 0.0
        assert r.gamma_mm6 is None

    def test_polygon_reproduces_the_parametric_rectangle(self):
        a = analyze_section("rectangle", d=500.0, b=300.0)
        b = analyze_polygon_section(RECT)
        assert b.area_mm2 == pytest.approx(a.area_mm2, rel=1e-12)
        assert b.ixx_mm4 == pytest.approx(a.ixx_mm4, rel=1e-12)
        assert b.sxx_mm3 == pytest.approx(a.sxx_mm3, rel=1e-9)
        assert b.j_mm4 == pytest.approx(a.j_mm4, rel=0.005)


class TestPolygonValidation:
    """Native replacements for the shapely validity check."""

    def test_simple_polygon_accepted(self):
        assert polygon_is_simple(RECT)

    def test_bowtie_rejected(self):
        assert not polygon_is_simple(
            [(0.0, 0.0), (100.0, 100.0), (100.0, 0.0), (0.0, 100.0)])

    def test_zero_area_rejected(self):
        assert not polygon_is_simple([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)])

    def test_concave_polygon_accepted(self):
        el = [(0.0, 0.0), (200.0, 0.0), (200.0, 50.0), (50.0, 50.0),
              (50.0, 300.0), (0.0, 300.0)]
        assert polygon_is_simple(el)
        assert analyze_polygon_section(el, warping=False).area_mm2 == \
            pytest.approx(200 * 50 + 50 * 250, rel=1e-12)


class TestGeometryGuards:
    def test_rhs_wall_too_thick(self):
        with pytest.raises(ValueError, match="too large"):
            analyze_section("rhs", d=100.0, b=80.0, t=40.0)

    def test_chs_wall_too_thick(self):
        with pytest.raises(ValueError, match="too thick"):
            analyze_section("chs", d=100.0, t=50.0)

    def test_i_section_flanges_too_deep(self):
        with pytest.raises(ValueError, match="do not fit"):
            analyze_section("i_section", d=20.0, b=180.0, t_f=14.0, t_w=9.0)
