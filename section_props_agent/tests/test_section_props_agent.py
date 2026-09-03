"""Tests for section_props_agent — anchored to closed-form section formulas."""

import math

import pytest

pytest.importorskip("sectionproperties")

from section_props_agent import (
    analyze_section, analyze_polygon_section, has_sectionproperties,
    SECTION_SHAPES,
)


class TestRectangleExact:
    """Rectangle 100 x 200 mm — every property has an exact closed form."""

    @pytest.fixture(scope="class")
    def rect(self):
        return analyze_section("rectangle", d=200, b=100)

    def test_area_and_centroid(self, rect):
        assert rect.area_mm2 == pytest.approx(100 * 200, rel=1e-9)
        assert rect.cx_mm == pytest.approx(50, abs=1e-6)
        assert rect.cy_mm == pytest.approx(100, abs=1e-6)

    def test_second_moments(self, rect):
        # I = b d^3 / 12
        assert rect.ixx_mm4 == pytest.approx(100 * 200**3 / 12, rel=1e-9)
        assert rect.iyy_mm4 == pytest.approx(200 * 100**3 / 12, rel=1e-9)
        assert rect.ixy_mm4 == pytest.approx(0.0, abs=1.0)

    def test_elastic_and_plastic_moduli(self, rect):
        # Z = b d^2 / 6 ; S = b d^2 / 4
        assert rect.zxx_plus_mm3 == pytest.approx(100 * 200**2 / 6, rel=1e-9)
        assert rect.sxx_mm3 == pytest.approx(100 * 200**2 / 4, rel=1e-6)
        assert rect.syy_mm3 == pytest.approx(200 * 100**2 / 4, rel=1e-6)

    def test_radii_of_gyration(self, rect):
        # r = d / sqrt(12)
        assert rect.rx_mm == pytest.approx(200 / math.sqrt(12), rel=1e-9)
        assert rect.ry_mm == pytest.approx(100 / math.sqrt(12), rel=1e-9)

    def test_torsion_constant(self, rect):
        # exact series solution for 2:1 rectangle: J = 0.2287 * b^3 * d
        # (Roark's Formulas, beta = 0.229 for d/b = 2)
        j_exact = 0.2287 * 100**3 * 200
        assert rect.j_mm4 == pytest.approx(j_exact, rel=0.01)

    def test_to_dict_and_summary(self, rect):
        d = rect.to_dict()
        assert d["area_mm2"] == pytest.approx(20000)
        assert "CROSS-SECTION" in rect.summary()


class TestCircleExact:
    def test_circle(self):
        # 64-segment polygonization underestimates the true circle by
        # ~0.16% in area (1 - sinc(pi/64)) — tolerance reflects that.
        r = analyze_section("circle", d=100, warping=False)
        assert r.area_mm2 == pytest.approx(math.pi * 50**2, rel=3e-3)
        assert r.ixx_mm4 == pytest.approx(math.pi * 100**4 / 64, rel=6e-3)


class TestISection:
    def test_i_section_area_and_ixx(self):
        # 300 deep, 150 wide, 10 flanges, 6 web, no root radius:
        # A = 2*150*10 + 280*6 = 4680
        # Ixx = 150*300^3/12 - 144*280^3/12 = 74,086,000 mm^4... compute:
        r = analyze_section("i_section", d=300, b=150, t_f=10, t_w=6,
                            warping=False)
        assert r.area_mm2 == pytest.approx(4680, rel=1e-6)
        ixx_hand = 150 * 300**3 / 12 - (150 - 6) * 280**3 / 12
        assert r.ixx_mm4 == pytest.approx(ixx_hand, rel=1e-6)


class TestPolygon:
    def test_polygon_matches_rectangle(self):
        p = analyze_polygon_section([(0, 0), (100, 0), (100, 200), (0, 200)],
                                    warping=False)
        assert p.area_mm2 == pytest.approx(20000, rel=1e-9)
        assert p.ixx_mm4 == pytest.approx(100 * 200**3 / 12, rel=1e-9)

    def test_bad_polygon_rejected(self):
        with pytest.raises(ValueError):
            analyze_polygon_section([(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            # self-intersecting bowtie
            analyze_polygon_section([(0, 0), (10, 10), (10, 0), (0, 10)])


class TestValidation:
    def test_unknown_shape(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            analyze_section("t_section", d=100)

    def test_missing_dimension(self):
        with pytest.raises(ValueError, match="missing"):
            analyze_section("rectangle", d=100)

    def test_negative_dimension(self):
        with pytest.raises(ValueError, match="positive"):
            analyze_section("rectangle", d=100, b=-5)

    def test_shape_registry(self):
        assert set(SECTION_SHAPES) == {"rectangle", "circle", "chs", "rhs",
                                       "i_section"}
        assert has_sectionproperties()
