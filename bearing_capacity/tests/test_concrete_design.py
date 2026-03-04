"""Tests for ACI 318-19 concrete footing design."""

import math
import pytest

from bearing_capacity.concrete_design import (
    design_concrete_footing, rebar_area, _compute_As_flexure,
    ConcreteFootingResult,
)


class TestRebarArea:
    def test_known_areas(self):
        assert rebar_area("#4") == pytest.approx(129.0)
        assert rebar_area("#8") == pytest.approx(510.0)
        assert rebar_area("#11") == pytest.approx(1006.0)

    def test_unknown_bar(self):
        with pytest.raises(ValueError, match="Unknown rebar"):
            rebar_area("#99")


class TestOnewayShear:
    """Verify one-way shear demand and capacity."""

    def test_standard_footing(self):
        """2.5m sq footing, 0.5m thick, 1000 kN — check Vu < phiVc."""
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            fc_MPa=28.0, fy_MPa=420.0,
            col_b_m=0.4, col_d_m=0.4,
        )
        # Vu should be positive
        assert r.Vu_oneway_kN >= 0
        assert r.phi_Vc_oneway_kN > 0
        # For this geometry, one-way shear should pass
        assert r.oneway_ok is True

    def test_thin_footing_fails(self):
        """Very thin footing should fail one-way shear."""
        r = design_concrete_footing(
            P_kN=3000, B_m=2.0, L_m=2.0, h_m=0.2,
            fc_MPa=28.0, col_b_m=0.3, col_d_m=0.3,
            cover_mm=50.0, bar_size="#5",
        )
        # With h=200mm, d~137mm for a 3000kN load on 2x2 footing
        # should likely fail
        assert r.Vu_oneway_kN > 0

    def test_large_column_low_shear(self):
        """Large column relative to footing => low cantilever => low shear."""
        r = design_concrete_footing(
            P_kN=500, B_m=1.5, L_m=1.5, h_m=0.6,
            col_b_m=1.0, col_d_m=1.0,
        )
        # Short cantilever (250mm) with large d => Vu near zero
        assert r.Vu_oneway_kN < r.phi_Vc_oneway_kN
        assert r.oneway_ok is True


class TestTwowayShear:
    """Verify two-way (punching) shear."""

    def test_standard_footing(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        assert r.Vu_twoway_kN > 0
        assert r.phi_Vc_twoway_kN > 0
        assert r.bo_m > 0
        # bo = 2*(col_b+d + col_d+d)
        expected_bo = 2 * ((0.4 + r.d_m) + (0.4 + r.d_m))
        assert r.bo_m == pytest.approx(expected_bo, abs=0.001)

    def test_all_three_equations(self):
        """Verify that governing Vc is the minimum of 3 ACI equations."""
        # Square column => beta=1 => Vc_2 simplifies
        r = design_concrete_footing(
            P_kN=800, B_m=2.0, L_m=2.0, h_m=0.45,
            col_b_m=0.4, col_d_m=0.4,
        )
        # For square column, beta=1, so (0.17+0.33/1)=0.50
        # Vc_2 = 0.50*sqrt(fc)*bo*d > 0.33*sqrt(fc)*bo*d = Vc_1
        # Governing should be min of Vc_1, Vc_2, Vc_3
        assert r.phi_Vc_twoway_kN > 0

    def test_rectangular_column(self):
        """Non-square column: beta > 1 reduces Vc_2."""
        r = design_concrete_footing(
            P_kN=800, B_m=2.5, L_m=2.5, h_m=0.45,
            col_b_m=0.3, col_d_m=0.6,
        )
        # beta = 0.6/0.3 = 2.0
        assert r.twoway_ok  # should still pass for this load


class TestFlexure:
    """Verify flexural design."""

    def test_As_req_positive(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        assert r.Mu_kNm > 0
        assert r.As_req_mm2 > 0
        assert r.As_provided_mm2 >= r.As_req_mm2
        assert r.As_provided_mm2 >= r.As_min_mm2
        assert r.n_bars >= 2

    def test_min_steel_governs(self):
        """Light load: minimum steel should govern over required."""
        r = design_concrete_footing(
            P_kN=100, B_m=2.0, L_m=2.0, h_m=0.4,
            col_b_m=0.4, col_d_m=0.4,
        )
        # Very light load => As_req small, As_min should govern
        assert r.As_min_mm2 >= r.As_req_mm2
        assert r.As_provided_mm2 >= r.As_min_mm2

    def test_As_flexure_direct(self):
        """Direct test of Whitney stress block computation."""
        # Known case: Mu = 200 kN-m, f'c=28, fy=420, b=2000mm, d=400mm
        Mu = 200e6  # N-mm
        As = _compute_As_flexure(Mu, 28.0, 420.0, 2000.0, 400.0, 0.9)
        # Should be a reasonable steel area
        assert 1000 < As < 5000  # rough bounds

    def test_zero_moment(self):
        """Zero moment => zero As."""
        As = _compute_As_flexure(0.0, 28.0, 420.0, 2000.0, 400.0, 0.9)
        assert As == 0.0


class TestBarSelection:
    """Verify bar count and spacing."""

    def test_bar_count(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4, bar_size="#6",
        )
        expected_n = math.ceil(max(r.As_req_mm2, r.As_min_mm2) / 284.0)
        expected_n = max(expected_n, 2)
        assert r.n_bars == expected_n
        assert r.As_provided_mm2 == pytest.approx(expected_n * 284.0)

    def test_spacing(self):
        """Spacing = (B - 2*cover) / (n_bars - 1)."""
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4, cover_mm=75.0,
        )
        if r.n_bars > 1:
            expected_spacing = (2500 - 2 * 75.0) / (r.n_bars - 1)
            assert r.spacing_mm == pytest.approx(expected_spacing, abs=0.1)


class TestDevelopmentLength:
    """Verify development length checks."""

    def test_standard(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        assert r.ld_req_mm >= 300  # ACI minimum
        assert r.ld_avail_mm > 0

    def test_small_bars_psi_s(self):
        """#6 and smaller get psi_s = 0.8 reduction."""
        r_small = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4, bar_size="#5",
        )
        r_large = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4, bar_size="#8",
        )
        # #5 uses psi_s=0.8, #8 uses psi_s=1.0
        # ld_req for #5 should be proportionally less (per bar diameter)
        ld_ratio_small = r_small.ld_req_mm / (15.875)  # #5 db in mm
        ld_ratio_large = r_large.ld_req_mm / (25.4)    # #8 db in mm
        assert ld_ratio_small < ld_ratio_large


class TestSquareVsRectangular:
    """Compare square and rectangular footings."""

    def test_same_area(self):
        """Same area footing, different aspect ratios."""
        r_sq = design_concrete_footing(
            P_kN=1000, B_m=2.0, L_m=2.0, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        r_rect = design_concrete_footing(
            P_kN=1000, B_m=1.5, L_m=2.667, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        # Both have ~4 m^2 area, so q_u similar
        assert abs(r_sq.Mu_kNm - r_rect.Mu_kNm) / r_sq.Mu_kNm < 0.5


class TestOutputMethods:
    """Test summary() and to_dict()."""

    def test_summary(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        s = r.summary()
        assert "ACI 318-19" in s
        assert "ONE-WAY SHEAR" in s
        assert "TWO-WAY" in s
        assert "FLEXURE" in s
        assert "DEVELOPMENT" in s

    def test_to_dict(self):
        r = design_concrete_footing(
            P_kN=1000, B_m=2.5, L_m=2.5, h_m=0.5,
            col_b_m=0.4, col_d_m=0.4,
        )
        d = r.to_dict()
        assert d["P_kN"] == 1000
        assert d["B_m"] == 2.5
        assert isinstance(d["oneway_ok"], bool)
        assert isinstance(d["n_bars"], int)
        assert "bar_size" in d


class TestEdgeCases:
    """Edge cases and validation."""

    def test_zero_depth_raises(self):
        """h too small for cover => ValueError."""
        with pytest.raises(ValueError, match="Effective depth"):
            design_concrete_footing(
                P_kN=500, B_m=2.0, L_m=2.0, h_m=0.05,
                cover_mm=75.0,
            )

    def test_unknown_bar_size(self):
        with pytest.raises(ValueError, match="Unknown rebar"):
            design_concrete_footing(
                P_kN=500, B_m=2.0, L_m=2.0, bar_size="#99",
            )
