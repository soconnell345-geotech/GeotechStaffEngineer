"""Comprehensive tests for geotech.dm7_2.chapter3 module.

Tests cover all 11 public functions (Equations 3-1 through 3-10) with valid
inputs, edge cases, and every ValueError validation branch.
"""

import pytest
from geotech.dm7_2.chapter3 import *


# ---------------------------------------------------------------------------
# 1. dry_unit_weight  (Equation 3-1)
#    gamma_d = gamma_t / (1 + w/100)
# ---------------------------------------------------------------------------

class TestDryUnitWeight:
    """Tests for dry_unit_weight(gamma_t, w)."""

    def test_basic_valid(self):
        # 120 / (1 + 15/100) = 120 / 1.15 = 104.347826...
        result = dry_unit_weight(120.0, 15.0)
        assert result == pytest.approx(104.347826086957, rel=1e-4)

    def test_zero_water_content(self):
        # w = 0 => gamma_d = gamma_t
        result = dry_unit_weight(100.0, 0.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_high_water_content(self):
        # 130 / (1 + 50/100) = 130 / 1.50 = 86.666...
        result = dry_unit_weight(130.0, 50.0)
        assert result == pytest.approx(86.6666666667, rel=1e-4)

    def test_si_units(self):
        # 18.5 / (1 + 20/100) = 18.5 / 1.20 = 15.41666...
        result = dry_unit_weight(18.5, 20.0)
        assert result == pytest.approx(15.41666666667, rel=1e-4)

    def test_raises_gamma_t_zero(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            dry_unit_weight(0.0, 10.0)

    def test_raises_gamma_t_negative(self):
        with pytest.raises(ValueError, match="gamma_t must be positive"):
            dry_unit_weight(-5.0, 10.0)

    def test_raises_w_negative(self):
        with pytest.raises(ValueError, match="w must be non-negative"):
            dry_unit_weight(120.0, -1.0)


# ---------------------------------------------------------------------------
# 2. dry_unit_weight_from_saturation  (Equation 3-2)
#    gamma_d = gamma_w / (w_dec / (S/100) + 1/G_s)
# ---------------------------------------------------------------------------

class TestDryUnitWeightFromSaturation:
    """Tests for dry_unit_weight_from_saturation(S, w, G_s, gamma_w)."""

    def test_basic_valid(self):
        # gamma_w / (w_dec/(S/100) + 1/G_s)
        # = 62.4 / (0.15/0.85 + 1/2.70)
        # = 62.4 / (0.176470588 + 0.370370370)
        # = 62.4 / 0.546840959 = 114.1048...
        result = dry_unit_weight_from_saturation(85.0, 15.0, 2.70, 62.4)
        assert result == pytest.approx(114.10484228, rel=1e-4)

    def test_full_saturation(self):
        # S = 100 => gamma_w / (w_dec/1.0 + 1/G_s)
        # = 62.4 / (0.20 + 1/2.65)
        # = 62.4 / (0.20 + 0.377358490)
        # = 62.4 / 0.577358490 = 108.074...
        result = dry_unit_weight_from_saturation(100.0, 20.0, 2.65, 62.4)
        assert result == pytest.approx(108.07438017, rel=1e-4)

    def test_si_units(self):
        # gamma_w=9.81, S=90, w=18, G_s=2.70
        # = 9.81 / (0.18/0.90 + 1/2.70)
        # = 9.81 / (0.20 + 0.370370370)
        # = 9.81 / 0.570370370 = 17.2001...
        result = dry_unit_weight_from_saturation(90.0, 18.0, 2.70, 9.81)
        assert result == pytest.approx(17.200105, rel=1e-4)

    def test_raises_S_zero(self):
        with pytest.raises(ValueError, match="S must be in the range"):
            dry_unit_weight_from_saturation(0.0, 15.0, 2.70)

    def test_raises_S_negative(self):
        with pytest.raises(ValueError, match="S must be in the range"):
            dry_unit_weight_from_saturation(-10.0, 15.0, 2.70)

    def test_raises_S_over_100(self):
        with pytest.raises(ValueError, match="S must be in the range"):
            dry_unit_weight_from_saturation(100.1, 15.0, 2.70)

    def test_raises_w_zero(self):
        with pytest.raises(ValueError, match="w must be positive"):
            dry_unit_weight_from_saturation(85.0, 0.0, 2.70)

    def test_raises_w_negative(self):
        with pytest.raises(ValueError, match="w must be positive"):
            dry_unit_weight_from_saturation(85.0, -5.0, 2.70)

    def test_raises_G_s_zero(self):
        with pytest.raises(ValueError, match="G_s must be positive"):
            dry_unit_weight_from_saturation(85.0, 15.0, 0.0)

    def test_raises_G_s_negative(self):
        with pytest.raises(ValueError, match="G_s must be positive"):
            dry_unit_weight_from_saturation(85.0, 15.0, -2.70)

    def test_raises_gamma_w_zero(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            dry_unit_weight_from_saturation(85.0, 15.0, 2.70, gamma_w=0.0)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            dry_unit_weight_from_saturation(85.0, 15.0, 2.70, gamma_w=-1.0)


# ---------------------------------------------------------------------------
# 3. relative_compaction  (Equation 3-3)
#    R.C. = (gamma_d_field / gamma_d_max) * 100
# ---------------------------------------------------------------------------

class TestRelativeCompaction:
    """Tests for relative_compaction(gamma_d_field, gamma_d_max)."""

    def test_basic_valid(self):
        # (110 / 120) * 100 = 91.666...
        result = relative_compaction(110.0, 120.0)
        assert result == pytest.approx(91.6666666667, rel=1e-4)

    def test_full_compaction(self):
        # (120 / 120) * 100 = 100.0
        result = relative_compaction(120.0, 120.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_zero_field_weight(self):
        # (0 / 120) * 100 = 0.0
        result = relative_compaction(0.0, 120.0)
        assert result == pytest.approx(0.0, rel=1e-4)

    def test_raises_gamma_d_max_zero(self):
        with pytest.raises(ValueError, match="gamma_d_max must be positive"):
            relative_compaction(110.0, 0.0)

    def test_raises_gamma_d_max_negative(self):
        with pytest.raises(ValueError, match="gamma_d_max must be positive"):
            relative_compaction(110.0, -10.0)

    def test_raises_gamma_d_field_negative(self):
        with pytest.raises(ValueError, match="gamma_d_field must be non-negative"):
            relative_compaction(-5.0, 120.0)


# ---------------------------------------------------------------------------
# 4. relative_water_content  (Equation 3-4)
#    delta_w = w_field - w_opt
# ---------------------------------------------------------------------------

class TestRelativeWaterContent:
    """Tests for relative_water_content(w_field, w_opt)."""

    def test_wet_of_optimum(self):
        # 18 - 15 = 3.0
        result = relative_water_content(18.0, 15.0)
        assert result == pytest.approx(3.0, rel=1e-4)

    def test_dry_of_optimum(self):
        # 12 - 15 = -3.0
        result = relative_water_content(12.0, 15.0)
        assert result == pytest.approx(-3.0, rel=1e-4)

    def test_at_optimum(self):
        # 15 - 15 = 0.0
        result = relative_water_content(15.0, 15.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_zero_field_water_content(self):
        # 0 - 15 = -15.0
        result = relative_water_content(0.0, 15.0)
        assert result == pytest.approx(-15.0, rel=1e-4)

    def test_zero_optimum_water_content(self):
        # 10 - 0 = 10.0
        result = relative_water_content(10.0, 0.0)
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_raises_w_field_negative(self):
        with pytest.raises(ValueError, match="w_field must be non-negative"):
            relative_water_content(-1.0, 15.0)

    def test_raises_w_opt_negative(self):
        with pytest.raises(ValueError, match="w_opt must be non-negative"):
            relative_water_content(18.0, -1.0)


# ---------------------------------------------------------------------------
# 5. relative_density_from_void_ratio  (Equation 3-5, void-ratio form)
#    D_r = ((e_max - e) / (e_max - e_min)) * 100
# ---------------------------------------------------------------------------

class TestRelativeDensityFromVoidRatio:
    """Tests for relative_density_from_void_ratio(e, e_max, e_min)."""

    def test_basic_valid(self):
        # ((0.90 - 0.65) / (0.90 - 0.50)) * 100 = (0.25/0.40)*100 = 62.5
        result = relative_density_from_void_ratio(0.65, 0.90, 0.50)
        assert result == pytest.approx(62.5, rel=1e-4)

    def test_loosest_state(self):
        # e = e_max => D_r = 0%
        result = relative_density_from_void_ratio(0.90, 0.90, 0.50)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_densest_state(self):
        # e = e_min => D_r = 100%
        result = relative_density_from_void_ratio(0.50, 0.90, 0.50)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_midpoint(self):
        # e = 0.70, e_max=0.90, e_min=0.50 => (0.20/0.40)*100 = 50%
        result = relative_density_from_void_ratio(0.70, 0.90, 0.50)
        assert result == pytest.approx(50.0, rel=1e-4)

    def test_raises_e_max_equal_e_min(self):
        with pytest.raises(ValueError, match="e_max must be greater than e_min"):
            relative_density_from_void_ratio(0.50, 0.50, 0.50)

    def test_raises_e_max_less_than_e_min(self):
        with pytest.raises(ValueError, match="e_max must be greater than e_min"):
            relative_density_from_void_ratio(0.65, 0.40, 0.90)


# ---------------------------------------------------------------------------
# 6. relative_density_from_dry_unit_weight  (Equation 3-5, density form)
#    D_r = (gamma_d_max / gamma_d_field)
#         * ((gamma_d_field - gamma_d_min) / (gamma_d_max - gamma_d_min))
#         * 100
# ---------------------------------------------------------------------------

class TestRelativeDensityFromDryUnitWeight:
    """Tests for relative_density_from_dry_unit_weight(gamma_d_field, gamma_d_max, gamma_d_min)."""

    def test_basic_valid(self):
        # (110/100) * ((100 - 90) / (110 - 90)) * 100
        # = 1.1 * (10/20) * 100 = 1.1 * 0.5 * 100 = 55.0
        result = relative_density_from_dry_unit_weight(100.0, 110.0, 90.0)
        assert result == pytest.approx(55.0, rel=1e-4)

    def test_densest_state(self):
        # gamma_d_field = gamma_d_max = 110
        # (110/110) * ((110 - 90) / (110 - 90)) * 100 = 1 * 1 * 100 = 100
        result = relative_density_from_dry_unit_weight(110.0, 110.0, 90.0)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_loosest_state(self):
        # gamma_d_field = gamma_d_min = 90
        # (110/90) * ((90 - 90) / (110 - 90)) * 100 = 1.2222 * 0 * 100 = 0
        result = relative_density_from_dry_unit_weight(90.0, 110.0, 90.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_another_valid(self):
        # gamma_d_field=105, gamma_d_max=115, gamma_d_min=85
        # (115/105) * ((105-85)/(115-85)) * 100
        # = 1.095238... * (20/30) * 100
        # = 1.095238... * 0.666666... * 100 = 73.0158...
        result = relative_density_from_dry_unit_weight(105.0, 115.0, 85.0)
        assert result == pytest.approx(73.015873, rel=1e-4)

    def test_raises_gamma_d_field_zero(self):
        with pytest.raises(ValueError, match="gamma_d_field must be positive"):
            relative_density_from_dry_unit_weight(0.0, 110.0, 90.0)

    def test_raises_gamma_d_field_negative(self):
        with pytest.raises(ValueError, match="gamma_d_field must be positive"):
            relative_density_from_dry_unit_weight(-5.0, 110.0, 90.0)

    def test_raises_gamma_d_max_equal_gamma_d_min(self):
        with pytest.raises(ValueError, match="gamma_d_max must be greater than gamma_d_min"):
            relative_density_from_dry_unit_weight(100.0, 90.0, 90.0)

    def test_raises_gamma_d_max_less_than_gamma_d_min(self):
        with pytest.raises(ValueError, match="gamma_d_max must be greater than gamma_d_min"):
            relative_density_from_dry_unit_weight(100.0, 80.0, 90.0)


# ---------------------------------------------------------------------------
# 7. oversize_corrected_water_content  (Equation 3-6)
#    w_T = P_C * w_C + P_F * w_F
# ---------------------------------------------------------------------------

class TestOversizeCorrectedWaterContent:
    """Tests for oversize_corrected_water_content(P_C, w_C, P_F, w_F)."""

    def test_basic_valid(self):
        # 0.30*0.02 + 0.70*0.15 = 0.006 + 0.105 = 0.111
        result = oversize_corrected_water_content(0.30, 0.02, 0.70, 0.15)
        assert result == pytest.approx(0.111, rel=1e-4)

    def test_no_oversize(self):
        # P_C=0, P_F=1 => w_T = 0*0.02 + 1*0.18 = 0.18
        result = oversize_corrected_water_content(0.0, 0.02, 1.0, 0.18)
        assert result == pytest.approx(0.18, rel=1e-4)

    def test_all_oversize(self):
        # P_C=1, P_F=0 => w_T = 1*0.03 + 0*0.18 = 0.03
        result = oversize_corrected_water_content(1.0, 0.03, 0.0, 0.18)
        assert result == pytest.approx(0.03, rel=1e-4)

    def test_both_zero_water_content(self):
        # w_C=0, w_F=0 => w_T = 0
        result = oversize_corrected_water_content(0.30, 0.0, 0.70, 0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_raises_P_C_negative(self):
        with pytest.raises(ValueError, match="P_C must be in the range"):
            oversize_corrected_water_content(-0.1, 0.02, 1.1, 0.15)

    def test_raises_P_C_over_one(self):
        with pytest.raises(ValueError, match="P_C must be in the range"):
            oversize_corrected_water_content(1.1, 0.02, -0.1, 0.15)

    def test_raises_P_F_negative(self):
        with pytest.raises(ValueError, match="P_F must be in the range"):
            oversize_corrected_water_content(0.30, 0.02, -0.1, 0.15)

    def test_raises_P_F_over_one(self):
        with pytest.raises(ValueError, match="P_F must be in the range"):
            oversize_corrected_water_content(0.30, 0.02, 1.1, 0.15)

    def test_raises_sum_not_one(self):
        with pytest.raises(ValueError, match="P_C \\+ P_F must equal 1.0"):
            oversize_corrected_water_content(0.30, 0.02, 0.60, 0.15)

    def test_raises_w_C_negative(self):
        with pytest.raises(ValueError, match="w_C must be non-negative"):
            oversize_corrected_water_content(0.30, -0.01, 0.70, 0.15)

    def test_raises_w_F_negative(self):
        with pytest.raises(ValueError, match="w_F must be non-negative"):
            oversize_corrected_water_content(0.30, 0.02, 0.70, -0.01)


# ---------------------------------------------------------------------------
# 8. oversize_corrected_dry_unit_weight  (Equation 3-7)
#    gamma_dT = (gamma_dF * G_sC * gamma_w)
#             / (gamma_dF * P_C + G_sC * gamma_w * P_F)
# ---------------------------------------------------------------------------

class TestOversizeCorrectedDryUnitWeight:
    """Tests for oversize_corrected_dry_unit_weight(gamma_dF, P_C, G_sC, P_F, gamma_w)."""

    def test_basic_valid(self):
        # num = 115 * 2.65 * 62.4 = 304.75 * 62.4 = 19016.4
        # den = 115 * 0.30 + 2.65 * 62.4 * 0.70 = 34.5 + 115.752 = 150.252
        # result = 19016.4 / 150.252 = 126.5634...
        result = oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, 0.70, 62.4)
        assert result == pytest.approx(126.56337, rel=1e-4)

    def test_no_oversize(self):
        # P_C=0, P_F=1 => gamma_dT = (gamma_dF * G_sC * gamma_w) / (G_sC * gamma_w)
        #                            = gamma_dF
        result = oversize_corrected_dry_unit_weight(115.0, 0.0, 2.65, 1.0, 62.4)
        assert result == pytest.approx(115.0, rel=1e-4)

    def test_si_units(self):
        # gamma_dF=18.0, P_C=0.25, G_sC=2.70, P_F=0.75, gamma_w=9.81
        # num = 18.0 * 2.70 * 9.81 = 476.766
        # den = 18.0 * 0.25 + 2.70 * 9.81 * 0.75 = 4.5 + 19.8652... = 24.3652...
        # but let me be precise: 2.70 * 9.81 = 26.487; 26.487 * 0.75 = 19.86525
        # den = 4.5 + 19.86525 = 24.36525
        # result = 476.766 / 24.36525 = 19.567...
        result = oversize_corrected_dry_unit_weight(18.0, 0.25, 2.70, 0.75, 9.81)
        assert result == pytest.approx(19.567, rel=1e-3)

    def test_raises_gamma_dF_zero(self):
        with pytest.raises(ValueError, match="gamma_dF must be positive"):
            oversize_corrected_dry_unit_weight(0.0, 0.30, 2.65, 0.70)

    def test_raises_gamma_dF_negative(self):
        with pytest.raises(ValueError, match="gamma_dF must be positive"):
            oversize_corrected_dry_unit_weight(-10.0, 0.30, 2.65, 0.70)

    def test_raises_P_C_negative(self):
        with pytest.raises(ValueError, match="P_C must be in the range"):
            oversize_corrected_dry_unit_weight(115.0, -0.1, 2.65, 1.1)

    def test_raises_P_C_over_one(self):
        with pytest.raises(ValueError, match="P_C must be in the range"):
            oversize_corrected_dry_unit_weight(115.0, 1.1, 2.65, -0.1)

    def test_raises_P_F_negative(self):
        with pytest.raises(ValueError, match="P_F must be in the range"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, -0.1)

    def test_raises_P_F_over_one(self):
        with pytest.raises(ValueError, match="P_F must be in the range"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, 1.1)

    def test_raises_sum_not_one(self):
        with pytest.raises(ValueError, match="P_C \\+ P_F must equal 1.0"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, 0.60)

    def test_raises_G_sC_zero(self):
        with pytest.raises(ValueError, match="G_sC must be positive"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 0.0, 0.70)

    def test_raises_G_sC_negative(self):
        with pytest.raises(ValueError, match="G_sC must be positive"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, -2.65, 0.70)

    def test_raises_gamma_w_zero(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, 0.70, gamma_w=0.0)

    def test_raises_gamma_w_negative(self):
        with pytest.raises(ValueError, match="gamma_w must be positive"):
            oversize_corrected_dry_unit_weight(115.0, 0.30, 2.65, 0.70, gamma_w=-1.0)


# ---------------------------------------------------------------------------
# 9. borrow_volume_from_waste_weight  (Equation 3-8)
#    V_B = V_F * (gamma_d_F / gamma_d_B) + W_L / gamma_d_B
# ---------------------------------------------------------------------------

class TestBorrowVolumeFromWasteWeight:
    """Tests for borrow_volume_from_waste_weight(V_F, gamma_d_F, gamma_d_B, W_L)."""

    def test_basic_valid(self):
        # 1000*(120/100) + 5000/100 = 1200 + 50 = 1250.0
        result = borrow_volume_from_waste_weight(1000.0, 120.0, 100.0, 5000.0)
        assert result == pytest.approx(1250.0, rel=1e-4)

    def test_no_waste(self):
        # 1000*(120/100) + 0/100 = 1200.0
        result = borrow_volume_from_waste_weight(1000.0, 120.0, 100.0, 0.0)
        assert result == pytest.approx(1200.0, rel=1e-4)

    def test_zero_fill_volume(self):
        # 0*(120/100) + 5000/100 = 50.0
        result = borrow_volume_from_waste_weight(0.0, 120.0, 100.0, 5000.0)
        assert result == pytest.approx(50.0, rel=1e-4)

    def test_equal_unit_weights(self):
        # 500*(100/100) + 2000/100 = 500 + 20 = 520.0
        result = borrow_volume_from_waste_weight(500.0, 100.0, 100.0, 2000.0)
        assert result == pytest.approx(520.0, rel=1e-4)

    def test_raises_V_F_negative(self):
        with pytest.raises(ValueError, match="V_F must be non-negative"):
            borrow_volume_from_waste_weight(-1.0, 120.0, 100.0, 5000.0)

    def test_raises_gamma_d_F_zero(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            borrow_volume_from_waste_weight(1000.0, 0.0, 100.0, 5000.0)

    def test_raises_gamma_d_F_negative(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            borrow_volume_from_waste_weight(1000.0, -10.0, 100.0, 5000.0)

    def test_raises_gamma_d_B_zero(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            borrow_volume_from_waste_weight(1000.0, 120.0, 0.0, 5000.0)

    def test_raises_gamma_d_B_negative(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            borrow_volume_from_waste_weight(1000.0, 120.0, -10.0, 5000.0)

    def test_raises_W_L_negative(self):
        with pytest.raises(ValueError, match="W_L must be non-negative"):
            borrow_volume_from_waste_weight(1000.0, 120.0, 100.0, -100.0)


# ---------------------------------------------------------------------------
# 10. borrow_volume_from_waste_fraction  (Equation 3-9)
#     V_B = V_F * gamma_d_F / (gamma_d_B * (1 - X_L))
# ---------------------------------------------------------------------------

class TestBorrowVolumeFromWasteFraction:
    """Tests for borrow_volume_from_waste_fraction(V_F, gamma_d_F, gamma_d_B, X_L)."""

    def test_basic_valid(self):
        # 1000*120 / (100*(1-0.10)) = 120000 / 90 = 1333.333...
        result = borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, 0.10)
        assert result == pytest.approx(1333.33333333, rel=1e-4)

    def test_no_waste(self):
        # X_L=0 => 1000*120 / (100*1) = 1200.0
        result = borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, 0.0)
        assert result == pytest.approx(1200.0, rel=1e-4)

    def test_zero_fill_volume(self):
        # V_F=0 => 0
        result = borrow_volume_from_waste_fraction(0.0, 120.0, 100.0, 0.10)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_high_waste_fraction(self):
        # X_L=0.50 => 1000*120/(100*0.50) = 120000/50 = 2400.0
        result = borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, 0.50)
        assert result == pytest.approx(2400.0, rel=1e-4)

    def test_raises_V_F_negative(self):
        with pytest.raises(ValueError, match="V_F must be non-negative"):
            borrow_volume_from_waste_fraction(-1.0, 120.0, 100.0, 0.10)

    def test_raises_gamma_d_F_zero(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            borrow_volume_from_waste_fraction(1000.0, 0.0, 100.0, 0.10)

    def test_raises_gamma_d_F_negative(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            borrow_volume_from_waste_fraction(1000.0, -10.0, 100.0, 0.10)

    def test_raises_gamma_d_B_zero(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            borrow_volume_from_waste_fraction(1000.0, 120.0, 0.0, 0.10)

    def test_raises_gamma_d_B_negative(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            borrow_volume_from_waste_fraction(1000.0, 120.0, -10.0, 0.10)

    def test_raises_X_L_negative(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, -0.01)

    def test_raises_X_L_one(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, 1.0)

    def test_raises_X_L_over_one(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            borrow_volume_from_waste_fraction(1000.0, 120.0, 100.0, 1.5)


# ---------------------------------------------------------------------------
# 11. shrinkage_factor  (Equation 3-10)
#     dV/V_F = gamma_d_F / (gamma_d_B * (1 - X_L)) - 1
# ---------------------------------------------------------------------------

class TestShrinkageFactor:
    """Tests for shrinkage_factor(gamma_d_F, gamma_d_B, X_L)."""

    def test_basic_shrinkage(self):
        # 120 / (100*(1-0.10)) - 1 = 120/90 - 1 = 1.33333... - 1 = 0.33333...
        result = shrinkage_factor(120.0, 100.0, 0.10)
        assert result == pytest.approx(0.33333333, rel=1e-4)

    def test_no_waste_shrinkage(self):
        # X_L=0 => 120/(100*1) - 1 = 1.20 - 1 = 0.20
        result = shrinkage_factor(120.0, 100.0, 0.0)
        assert result == pytest.approx(0.20, rel=1e-4)

    def test_bulking(self):
        # gamma_d_F < gamma_d_B, no waste => bulking (negative)
        # 90/(100*1) - 1 = 0.90 - 1 = -0.10
        result = shrinkage_factor(90.0, 100.0, 0.0)
        assert result == pytest.approx(-0.10, rel=1e-4)

    def test_no_change(self):
        # gamma_d_F = gamma_d_B, no waste => 0
        # 100/(100*1) - 1 = 0
        result = shrinkage_factor(100.0, 100.0, 0.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_high_waste(self):
        # 120/(100*(1-0.50)) - 1 = 120/50 - 1 = 2.4 - 1 = 1.4
        result = shrinkage_factor(120.0, 100.0, 0.50)
        assert result == pytest.approx(1.4, rel=1e-4)

    def test_raises_gamma_d_F_zero(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            shrinkage_factor(0.0, 100.0, 0.10)

    def test_raises_gamma_d_F_negative(self):
        with pytest.raises(ValueError, match="gamma_d_F must be positive"):
            shrinkage_factor(-10.0, 100.0, 0.10)

    def test_raises_gamma_d_B_zero(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            shrinkage_factor(120.0, 0.0, 0.10)

    def test_raises_gamma_d_B_negative(self):
        with pytest.raises(ValueError, match="gamma_d_B must be positive"):
            shrinkage_factor(120.0, -10.0, 0.10)

    def test_raises_X_L_negative(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            shrinkage_factor(120.0, 100.0, -0.01)

    def test_raises_X_L_one(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            shrinkage_factor(120.0, 100.0, 1.0)

    def test_raises_X_L_over_one(self):
        with pytest.raises(ValueError, match="X_L must be in the range"):
            shrinkage_factor(120.0, 100.0, 1.5)
