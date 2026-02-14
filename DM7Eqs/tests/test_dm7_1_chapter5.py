"""Comprehensive tests for geotech.dm7_1.chapter5 (Equations 5-1 through 5-28).

Each public function is tested for:
  - Basic valid input with a hand-calculated expected result
  - Edge cases where applicable
  - ValueError checks for every validation branch
"""

import math

import pytest
from geotech.dm7_1.chapter5 import *


# ===========================================================================
# 5-1  overconsolidation_ratio
# ===========================================================================

class TestOverconsolidationRatio:
    """Tests for Equation 5-1: OCR = sigma_p / sigma_z0."""

    def test_basic(self):
        # sigma_p=4000, sigma_z0=2000 => OCR = 4000/2000 = 2.0
        assert overconsolidation_ratio(4000.0, 2000.0) == pytest.approx(2.0, rel=1e-4)

    def test_normally_consolidated(self):
        # OCR = 1 when sigma_p equals sigma_z0
        assert overconsolidation_ratio(1500.0, 1500.0) == pytest.approx(1.0, rel=1e-4)

    def test_high_ocr(self):
        # sigma_p=10000, sigma_z0=500 => OCR = 20.0
        assert overconsolidation_ratio(10000.0, 500.0) == pytest.approx(20.0, rel=1e-4)

    def test_sigma_p_zero(self):
        # sigma_p = 0 is allowed => OCR = 0
        assert overconsolidation_ratio(0.0, 100.0) == pytest.approx(0.0, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            overconsolidation_ratio(100.0, 0.0)

    def test_sigma_z0_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            overconsolidation_ratio(100.0, -50.0)

    def test_sigma_p_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_p must be non-negative"):
            overconsolidation_ratio(-10.0, 100.0)


# ===========================================================================
# 5-2  undrained_shear_strength_from_ocr
# ===========================================================================

class TestUndrainedShearStrengthFromOcr:
    """Tests for Equation 5-2: s_u = USR_NC * OCR^m * sigma_z0."""

    def test_basic(self):
        # usr_nc=0.25, ocr=4, m=0.8, sigma_z0=2000
        # s_u = 0.25 * 4^0.8 * 2000
        # 4^0.8 = exp(0.8 * ln 4) = exp(0.8 * 1.386294) = exp(1.109035) = 3.031433
        # s_u = 0.25 * 3.031433 * 2000 = 1515.717
        expected = 0.25 * (4.0 ** 0.8) * 2000.0
        assert undrained_shear_strength_from_ocr(0.25, 4.0, 0.8, 2000.0) == pytest.approx(expected, rel=1e-4)

    def test_nc_soil(self):
        # OCR = 1 => s_u = USR_NC * 1^m * sigma_z0 = USR_NC * sigma_z0
        # usr_nc=0.3, ocr=1, m=0.8, sigma_z0=1000 => s_u = 0.3 * 1000 = 300
        assert undrained_shear_strength_from_ocr(0.3, 1.0, 0.8, 1000.0) == pytest.approx(300.0, rel=1e-4)

    def test_zero_ocr(self):
        # ocr=0 => 0^0.8 = 0 => s_u = 0
        assert undrained_shear_strength_from_ocr(0.3, 0.0, 0.8, 1000.0) == pytest.approx(0.0, rel=1e-4)

    def test_usr_nc_negative_raises(self):
        with pytest.raises(ValueError, match="usr_nc must be non-negative"):
            undrained_shear_strength_from_ocr(-0.1, 2.0, 0.8, 1000.0)

    def test_ocr_negative_raises(self):
        with pytest.raises(ValueError, match="ocr must be non-negative"):
            undrained_shear_strength_from_ocr(0.25, -1.0, 0.8, 1000.0)

    def test_sigma_z0_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be non-negative"):
            undrained_shear_strength_from_ocr(0.25, 2.0, 0.8, -100.0)


# ===========================================================================
# 5-3  preconsolidation_stress_from_su
# ===========================================================================

class TestPreconsolidationStressFromSu:
    """Tests for Equation 5-3: sigma_p = sigma_z0 * (s_u / (USR_NC * sigma_z0))^(1/m)."""

    def test_basic(self):
        # su=300, usr_nc=0.25, m=0.8, sigma_z0=1000
        # sigma_p = 1000 * (300 / (0.25 * 1000))^(1/0.8)
        #         = 1000 * (300 / 250)^(1.25)
        #         = 1000 * 1.2^1.25
        # 1.2^1.25 = exp(1.25 * ln 1.2) = exp(1.25 * 0.182322) = exp(0.227903) = 1.25596
        # sigma_p = 1000 * 1.25596 = 1255.96
        expected = 1000.0 * (300.0 / (0.25 * 1000.0)) ** (1.0 / 0.8)
        assert preconsolidation_stress_from_su(300.0, 0.25, 0.8, 1000.0) == pytest.approx(expected, rel=1e-4)

    def test_round_trip_with_eq52(self):
        # If s_u comes from Eq 5-2, Eq 5-3 should recover sigma_p
        # sigma_z0=2000, sigma_p=4000 => OCR=2, usr_nc=0.3, m=0.8
        ocr = 2.0
        usr_nc = 0.3
        m = 0.8
        sigma_z0 = 2000.0
        su = usr_nc * (ocr ** m) * sigma_z0
        sigma_p_recovered = preconsolidation_stress_from_su(su, usr_nc, m, sigma_z0)
        # sigma_p = sigma_z0 * OCR = 4000
        assert sigma_p_recovered == pytest.approx(4000.0, rel=1e-4)

    def test_usr_nc_zero_raises(self):
        with pytest.raises(ValueError, match="usr_nc must be positive"):
            preconsolidation_stress_from_su(300.0, 0.0, 0.8, 1000.0)

    def test_usr_nc_negative_raises(self):
        with pytest.raises(ValueError, match="usr_nc must be positive"):
            preconsolidation_stress_from_su(300.0, -0.1, 0.8, 1000.0)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            preconsolidation_stress_from_su(300.0, 0.25, 0.8, 0.0)

    def test_sigma_z0_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            preconsolidation_stress_from_su(300.0, 0.25, 0.8, -100.0)

    def test_m_zero_raises(self):
        with pytest.raises(ValueError, match="m must be non-zero"):
            preconsolidation_stress_from_su(300.0, 0.25, 0.0, 1000.0)


# ===========================================================================
# 5-4  settlement_from_strain
# ===========================================================================

class TestSettlementFromStrain:
    """Tests for Equation 5-4: s = sum(epsilon_z_i * H_i)."""

    def test_single_layer(self):
        # epsilon_z = 0.02, H = 10 => s = 0.02 * 10 = 0.2
        assert settlement_from_strain([0.02], [10.0]) == pytest.approx(0.2, rel=1e-4)

    def test_multiple_layers(self):
        # epsilon_z = [0.01, 0.03, 0.005], H = [5, 10, 8]
        # s = 0.01*5 + 0.03*10 + 0.005*8 = 0.05 + 0.30 + 0.04 = 0.39
        result = settlement_from_strain([0.01, 0.03, 0.005], [5.0, 10.0, 8.0])
        assert result == pytest.approx(0.39, rel=1e-4)

    def test_zero_strain(self):
        # zero strain => zero settlement
        assert settlement_from_strain([0.0, 0.0], [10.0, 5.0]) == pytest.approx(0.0, rel=1e-4)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="epsilon_z and H must have the same length"):
            settlement_from_strain([0.01, 0.02], [5.0])

    def test_empty_lists_raises(self):
        with pytest.raises(ValueError, match="At least one layer must be provided"):
            settlement_from_strain([], [])

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="Layer thicknesses must be non-negative"):
            settlement_from_strain([0.01], [-5.0])


# ===========================================================================
# 5-5  immediate_settlement
# ===========================================================================

class TestImmediateSettlement:
    """Tests for Equation 5-5: s = (q0 * B / Es) * mu0 * mu1."""

    def test_basic(self):
        # q0=2000, B=6, Es=500000, mu0=0.9, mu1=1.2
        # s = (2000 * 6 / 500000) * 0.9 * 1.2 = 0.024 * 0.9 * 1.2 = 0.02592
        expected = (2000.0 * 6.0 / 500000.0) * 0.9 * 1.2
        assert immediate_settlement(2000.0, 6.0, 500000.0, 0.9, 1.2) == pytest.approx(expected, rel=1e-4)

    def test_zero_load(self):
        # q0=0 => s=0
        assert immediate_settlement(0.0, 6.0, 500000.0, 0.9, 1.2) == pytest.approx(0.0, rel=1e-4)

    def test_zero_width(self):
        # B=0 => s=0
        assert immediate_settlement(2000.0, 0.0, 500000.0, 0.9, 1.2) == pytest.approx(0.0, rel=1e-4)

    def test_Es_zero_raises(self):
        with pytest.raises(ValueError, match="Es must be positive"):
            immediate_settlement(2000.0, 6.0, 0.0, 0.9, 1.2)

    def test_Es_negative_raises(self):
        with pytest.raises(ValueError, match="Es must be positive"):
            immediate_settlement(2000.0, 6.0, -100.0, 0.9, 1.2)

    def test_B_negative_raises(self):
        with pytest.raises(ValueError, match="B must be non-negative"):
            immediate_settlement(2000.0, -1.0, 500000.0, 0.9, 1.2)


# ===========================================================================
# 5-6  corrected_spt_silty_sand
# ===========================================================================

class TestCorrectedSptSiltySand:
    """Tests for Equation 5-6: N'_SM = 15 + 0.5 * (N' - 15)."""

    def test_basic_above_15(self):
        # N'=25 => N'_SM = 15 + 0.5*(25-15) = 15 + 5 = 20
        assert corrected_spt_silty_sand(25.0) == pytest.approx(20.0, rel=1e-4)

    def test_at_15(self):
        # N'=15 => N'_SM = 15 + 0.5*(15-15) = 15
        assert corrected_spt_silty_sand(15.0) == pytest.approx(15.0, rel=1e-4)

    def test_below_15(self):
        # N'=10 => N'_SM = 15 + 0.5*(10-15) = 15 - 2.5 = 12.5
        # (equation still applies mathematically, though manual says only for N'>15)
        assert corrected_spt_silty_sand(10.0) == pytest.approx(12.5, rel=1e-4)

    def test_zero(self):
        # N'=0 => N'_SM = 15 + 0.5*(0-15) = 15 - 7.5 = 7.5
        assert corrected_spt_silty_sand(0.0) == pytest.approx(7.5, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="N_prime must be non-negative"):
            corrected_spt_silty_sand(-1.0)


# ===========================================================================
# 5-7  schmertmann_settlement
# ===========================================================================

class TestSchmertmannSettlement:
    """Tests for Equation 5-7: s = C1 * C2 * (q0 - sigma_z0) * sum(Iz/Es * dz)."""

    def test_basic(self):
        # C1=0.9, C2=1.1, q0=3000, sigma_z0=1000, one layer: Iz=0.5, Es=2000, dz=3
        # s = 0.9 * 1.1 * (3000-1000) * (0.5/2000 * 3)
        #   = 0.99 * 2000 * 0.00075 = 0.99 * 1.5 = 1.485
        expected = 0.9 * 1.1 * 2000.0 * (0.5 / 2000.0 * 3.0)
        assert schmertmann_settlement(0.9, 1.1, 3000.0, 1000.0, [0.5], [2000.0], [3.0]) == pytest.approx(expected, rel=1e-4)

    def test_multiple_layers(self):
        # C1=1.0, C2=1.0, q0=2000, sigma_z0=500, 2 layers
        # Iz=[0.4, 0.2], Es=[1000, 500], dz=[2, 2]
        # summation = 0.4/1000*2 + 0.2/500*2 = 0.0008 + 0.0008 = 0.0016
        # s = 1.0 * 1.0 * (2000-500) * 0.0016 = 1500 * 0.0016 = 2.4
        result = schmertmann_settlement(1.0, 1.0, 2000.0, 500.0, [0.4, 0.2], [1000.0, 500.0], [2.0, 2.0])
        assert result == pytest.approx(2.4, rel=1e-4)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Iz, Es, and dz must have the same length"):
            schmertmann_settlement(1.0, 1.0, 2000.0, 500.0, [0.4], [1000.0, 500.0], [2.0])

    def test_empty_layers_raises(self):
        with pytest.raises(ValueError, match="At least one layer must be provided"):
            schmertmann_settlement(1.0, 1.0, 2000.0, 500.0, [], [], [])

    def test_zero_modulus_raises(self):
        with pytest.raises(ValueError, match="All Es values must be positive"):
            schmertmann_settlement(1.0, 1.0, 2000.0, 500.0, [0.4], [0.0], [2.0])

    def test_negative_modulus_raises(self):
        with pytest.raises(ValueError, match="All Es values must be positive"):
            schmertmann_settlement(1.0, 1.0, 2000.0, 500.0, [0.4], [-100.0], [2.0])


# ===========================================================================
# 5-8  schmertmann_embedment_correction
# ===========================================================================

class TestSchmertmannEmbedmentCorrection:
    """Tests for Equation 5-8: C1 = 1 - 0.5 * sigma_z0 / (q0 - sigma_z0), min 0.5."""

    def test_basic(self):
        # sigma_z0=500, q0=2000 => net=1500
        # C1 = 1 - 0.5*(500/1500) = 1 - 0.1667 = 0.8333
        expected = 1.0 - 0.5 * (500.0 / 1500.0)
        assert schmertmann_embedment_correction(500.0, 2000.0) == pytest.approx(expected, rel=1e-4)

    def test_minimum_clamp(self):
        # sigma_z0=1800, q0=2000 => net=200
        # raw = 1 - 0.5*(1800/200) = 1 - 4.5 = -3.5 => clamped to 0.5
        assert schmertmann_embedment_correction(1800.0, 2000.0) == pytest.approx(0.5, rel=1e-4)

    def test_zero_embedment(self):
        # sigma_z0=0, q0=2000 => net=2000
        # C1 = 1 - 0.5*(0/2000) = 1.0
        assert schmertmann_embedment_correction(0.0, 2000.0) == pytest.approx(1.0, rel=1e-4)

    def test_q0_equals_sigma_z0_raises(self):
        with pytest.raises(ValueError, match="q0 must be greater than sigma_z0"):
            schmertmann_embedment_correction(1000.0, 1000.0)

    def test_q0_less_than_sigma_z0_raises(self):
        with pytest.raises(ValueError, match="q0 must be greater than sigma_z0"):
            schmertmann_embedment_correction(2000.0, 1000.0)


# ===========================================================================
# 5-9  schmertmann_time_correction
# ===========================================================================

class TestSchmertmannTimeCorrection:
    """Tests for Equation 5-9: C2 = 1 + 0.2 * log10(t / 0.1)."""

    def test_at_reference_time(self):
        # t=0.1 => log10(0.1/0.1) = log10(1) = 0 => C2 = 1.0
        assert schmertmann_time_correction(0.1) == pytest.approx(1.0, rel=1e-4)

    def test_one_year(self):
        # t=1 => log10(1/0.1) = log10(10) = 1 => C2 = 1 + 0.2 = 1.2
        assert schmertmann_time_correction(1.0) == pytest.approx(1.2, rel=1e-4)

    def test_ten_years(self):
        # t=10 => log10(10/0.1) = log10(100) = 2 => C2 = 1 + 0.4 = 1.4
        assert schmertmann_time_correction(10.0) == pytest.approx(1.4, rel=1e-4)

    def test_short_time(self):
        # t=0.01 => log10(0.01/0.1) = log10(0.1) = -1 => C2 = 1 - 0.2 = 0.8
        assert schmertmann_time_correction(0.01) == pytest.approx(0.8, rel=1e-4)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="t_years must be positive"):
            schmertmann_time_correction(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="t_years must be positive"):
            schmertmann_time_correction(-1.0)


# ===========================================================================
# 5-10  modified_recompression_index
# ===========================================================================

class TestModifiedRecompressionIndex:
    """Tests for Equation 5-10: C_epsilon_r = Cr / (1 + e0)."""

    def test_basic(self):
        # Cr=0.05, e0=0.8 => C_epsilon_r = 0.05 / 1.8 = 0.027778
        assert modified_recompression_index(0.05, 0.8) == pytest.approx(0.05 / 1.8, rel=1e-4)

    def test_zero_void_ratio(self):
        # e0=0 => C_epsilon_r = Cr / 1.0 = Cr
        assert modified_recompression_index(0.04, 0.0) == pytest.approx(0.04, rel=1e-4)

    def test_zero_cr(self):
        # Cr=0 => C_epsilon_r = 0
        assert modified_recompression_index(0.0, 0.5) == pytest.approx(0.0, rel=1e-4)

    def test_e0_at_minus_one_raises(self):
        with pytest.raises(ValueError, match="e0 must be greater than -1"):
            modified_recompression_index(0.05, -1.0)

    def test_e0_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="e0 must be greater than -1"):
            modified_recompression_index(0.05, -2.0)

    def test_cr_negative_raises(self):
        with pytest.raises(ValueError, match="Cr must be non-negative"):
            modified_recompression_index(-0.01, 0.8)


# ===========================================================================
# 5-11  modified_compression_index
# ===========================================================================

class TestModifiedCompressionIndex:
    """Tests for Equation 5-11: C_epsilon_c = Cc / (1 + e0)."""

    def test_basic(self):
        # Cc=0.30, e0=1.0 => C_epsilon_c = 0.30 / 2.0 = 0.15
        assert modified_compression_index(0.30, 1.0) == pytest.approx(0.15, rel=1e-4)

    def test_high_void_ratio(self):
        # Cc=0.50, e0=2.5 => C_epsilon_c = 0.50 / 3.5 = 0.142857
        assert modified_compression_index(0.50, 2.5) == pytest.approx(0.50 / 3.5, rel=1e-4)

    def test_e0_at_minus_one_raises(self):
        with pytest.raises(ValueError, match="e0 must be greater than -1"):
            modified_compression_index(0.30, -1.0)

    def test_cc_negative_raises(self):
        with pytest.raises(ValueError, match="Cc must be non-negative"):
            modified_compression_index(-0.10, 0.8)


# ===========================================================================
# 5-12  primary_consolidation_settlement_nc
# ===========================================================================

class TestPrimaryConsolidationSettlementNc:
    """Tests for Equation 5-12: s_c = C_epsilon_c * log10((sigma_z0 + delta_sigma_z) / sigma_z0) * H."""

    def test_basic(self):
        # C_epsilon_c=0.15, sigma_z0=1000, delta_sigma_z=1000, H=10
        # s_c = 0.15 * log10(2000/1000) * 10 = 0.15 * log10(2) * 10
        # log10(2) = 0.30103
        # s_c = 0.15 * 0.30103 * 10 = 0.45155
        expected = 0.15 * math.log10(2000.0 / 1000.0) * 10.0
        result = primary_consolidation_settlement_nc(0.15, 1000.0, 1000.0, 10.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_small_stress_increment(self):
        # delta_sigma_z much smaller than sigma_z0
        # C_epsilon_c=0.20, sigma_z0=5000, delta_sigma_z=100, H=8
        # s_c = 0.20 * log10(5100/5000) * 8 = 0.20 * log10(1.02) * 8
        # log10(1.02) = 0.008600
        # s_c = 0.20 * 0.008600 * 8 = 0.01376
        expected = 0.20 * math.log10(5100.0 / 5000.0) * 8.0
        result = primary_consolidation_settlement_nc(0.20, 5000.0, 100.0, 8.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            primary_consolidation_settlement_nc(0.15, 0.0, 1000.0, 10.0)

    def test_sigma_z0_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            primary_consolidation_settlement_nc(0.15, -100.0, 1000.0, 10.0)

    def test_final_stress_not_positive_raises(self):
        # sigma_z0=100, delta_sigma_z=-200 => final stress = -100 <= 0
        with pytest.raises(ValueError, match="Final stress"):
            primary_consolidation_settlement_nc(0.15, 100.0, -200.0, 10.0)


# ===========================================================================
# 5-13  primary_consolidation_settlement_oc_recompression
# ===========================================================================

class TestPrimaryConsolidationSettlementOcRecompression:
    """Tests for Equation 5-13: s_c = C_epsilon_r * log10((sigma_z0 + delta_sigma_z) / sigma_z0) * H."""

    def test_basic(self):
        # C_epsilon_r=0.03, sigma_z0=2000, delta_sigma_z=500, H=12
        # s_c = 0.03 * log10(2500/2000) * 12 = 0.03 * log10(1.25) * 12
        # log10(1.25) = 0.09691
        # s_c = 0.03 * 0.09691 * 12 = 0.034888
        expected = 0.03 * math.log10(2500.0 / 2000.0) * 12.0
        result = primary_consolidation_settlement_oc_recompression(0.03, 2000.0, 500.0, 12.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            primary_consolidation_settlement_oc_recompression(0.03, 0.0, 500.0, 12.0)

    def test_final_stress_not_positive_raises(self):
        with pytest.raises(ValueError, match="Final stress"):
            primary_consolidation_settlement_oc_recompression(0.03, 100.0, -200.0, 12.0)


# ===========================================================================
# 5-14  primary_consolidation_settlement_oc_beyond
# ===========================================================================

class TestPrimaryConsolidationSettlementOcBeyond:
    """Tests for Equation 5-14: s_c = [C_er * log10(sigma_p/sigma_z0) + C_ec * log10(final/sigma_p)] * H."""

    def test_basic(self):
        # C_epsilon_r=0.03, C_epsilon_c=0.15, sigma_z0=1000, delta_sigma_z=2000, sigma_p=2000, H=10
        # final = 3000
        # recomp = 0.03 * log10(2000/1000) = 0.03 * 0.30103 = 0.0090309
        # virgin = 0.15 * log10(3000/2000) = 0.15 * 0.17609 = 0.0264135
        # s_c = (0.0090309 + 0.0264135) * 10 = 0.354444
        expected_recomp = 0.03 * math.log10(2000.0 / 1000.0)
        expected_virgin = 0.15 * math.log10(3000.0 / 2000.0)
        expected = (expected_recomp + expected_virgin) * 10.0
        result = primary_consolidation_settlement_oc_beyond(0.03, 0.15, 1000.0, 2000.0, 2000.0, 10.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            primary_consolidation_settlement_oc_beyond(0.03, 0.15, 0.0, 2000.0, 2000.0, 10.0)

    def test_sigma_p_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_p must be positive"):
            primary_consolidation_settlement_oc_beyond(0.03, 0.15, 1000.0, 2000.0, 0.0, 10.0)

    def test_sigma_p_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_p must be positive"):
            primary_consolidation_settlement_oc_beyond(0.03, 0.15, 1000.0, 2000.0, -100.0, 10.0)

    def test_final_not_exceeding_sigma_p_raises(self):
        # sigma_z0=1000, delta_sigma_z=500, sigma_p=2000 => final=1500 <= 2000
        with pytest.raises(ValueError, match="Final stress must exceed sigma_p"):
            primary_consolidation_settlement_oc_beyond(0.03, 0.15, 1000.0, 500.0, 2000.0, 10.0)

    def test_final_exactly_at_sigma_p_raises(self):
        # sigma_z0=1000, delta_sigma_z=1000, sigma_p=2000 => final=2000 == sigma_p
        with pytest.raises(ValueError, match="Final stress must exceed sigma_p"):
            primary_consolidation_settlement_oc_beyond(0.03, 0.15, 1000.0, 1000.0, 2000.0, 10.0)


# ===========================================================================
# 5-15  time_factor_vertical
# ===========================================================================

class TestTimeFactorVertical:
    """Tests for Equation 5-15: Tv = cv * t / Hdr^2."""

    def test_basic(self):
        # cv=0.5, t=100, Hdr=10 => Tv = 0.5*100/100 = 0.5
        assert time_factor_vertical(0.5, 100.0, 10.0) == pytest.approx(0.5, rel=1e-4)

    def test_double_drainage(self):
        # cv=0.1, t=365, Hdr=5 => Tv = 0.1*365/25 = 1.46
        assert time_factor_vertical(0.1, 365.0, 5.0) == pytest.approx(1.46, rel=1e-4)

    def test_zero_time(self):
        # t=0 => Tv=0
        assert time_factor_vertical(0.5, 0.0, 10.0) == pytest.approx(0.0, rel=1e-4)

    def test_zero_cv(self):
        # cv=0 => Tv=0
        assert time_factor_vertical(0.0, 100.0, 10.0) == pytest.approx(0.0, rel=1e-4)

    def test_Hdr_zero_raises(self):
        with pytest.raises(ValueError, match="Hdr must be positive"):
            time_factor_vertical(0.5, 100.0, 0.0)

    def test_Hdr_negative_raises(self):
        with pytest.raises(ValueError, match="Hdr must be positive"):
            time_factor_vertical(0.5, 100.0, -1.0)

    def test_cv_negative_raises(self):
        with pytest.raises(ValueError, match="cv must be non-negative"):
            time_factor_vertical(-0.1, 100.0, 10.0)

    def test_t_negative_raises(self):
        with pytest.raises(ValueError, match="t must be non-negative"):
            time_factor_vertical(0.5, -10.0, 10.0)


# ===========================================================================
# 5-16  equivalent_layer_thickness
# ===========================================================================

class TestEquivalentLayerThickness:
    """Tests for Equation 5-16: H'_n = Hn * sqrt(cv_i / cv_n)."""

    def test_basic(self):
        # cv_i=0.5, cv_n=2.0, Hn=10 => H'_n = 10 * sqrt(0.5/2.0) = 10 * sqrt(0.25) = 10 * 0.5 = 5.0
        assert equivalent_layer_thickness(0.5, 2.0, 10.0) == pytest.approx(5.0, rel=1e-4)

    def test_same_cv(self):
        # cv_i = cv_n => H'_n = Hn
        assert equivalent_layer_thickness(1.0, 1.0, 8.0) == pytest.approx(8.0, rel=1e-4)

    def test_reference_faster(self):
        # cv_i > cv_n => H'_n > Hn (equivalent thickness is larger)
        # cv_i=4.0, cv_n=1.0, Hn=6 => H'_n = 6 * sqrt(4/1) = 6 * 2 = 12
        assert equivalent_layer_thickness(4.0, 1.0, 6.0) == pytest.approx(12.0, rel=1e-4)

    def test_zero_Hn(self):
        # Hn=0 => H'_n = 0
        assert equivalent_layer_thickness(1.0, 2.0, 0.0) == pytest.approx(0.0, rel=1e-4)

    def test_cv_n_zero_raises(self):
        with pytest.raises(ValueError, match="cv_n must be positive"):
            equivalent_layer_thickness(1.0, 0.0, 10.0)

    def test_cv_n_negative_raises(self):
        with pytest.raises(ValueError, match="cv_n must be positive"):
            equivalent_layer_thickness(1.0, -0.5, 10.0)

    def test_cv_i_negative_raises(self):
        with pytest.raises(ValueError, match="cv_i must be non-negative"):
            equivalent_layer_thickness(-0.1, 1.0, 10.0)

    def test_Hn_negative_raises(self):
        with pytest.raises(ValueError, match="Hn must be non-negative"):
            equivalent_layer_thickness(1.0, 1.0, -5.0)


# ===========================================================================
# 5-17  total_transformed_thickness
# ===========================================================================

class TestTotalTransformedThickness:
    """Tests for Equation 5-17: H'_t = sum(H'_n)."""

    def test_single_layer(self):
        assert total_transformed_thickness([5.0]) == pytest.approx(5.0, rel=1e-4)

    def test_multiple_layers(self):
        # [5.0, 12.0, 8.0] => 25.0
        assert total_transformed_thickness([5.0, 12.0, 8.0]) == pytest.approx(25.0, rel=1e-4)

    def test_with_zeros(self):
        assert total_transformed_thickness([0.0, 10.0, 0.0]) == pytest.approx(10.0, rel=1e-4)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one layer must be provided"):
            total_transformed_thickness([])

    def test_negative_thickness_raises(self):
        with pytest.raises(ValueError, match="All thicknesses must be non-negative"):
            total_transformed_thickness([5.0, -1.0, 3.0])


# ===========================================================================
# 5-18  secondary_compression_settlement
# ===========================================================================

class TestSecondaryCompressionSettlement:
    """Tests for Equation 5-18: s_s = C_epsilon_alpha * log10(t / tp) * H0."""

    def test_basic(self):
        # C_epsilon_alpha=0.004, t=10, tp=1, H0=20
        # s_s = 0.004 * log10(10/1) * 20 = 0.004 * 1.0 * 20 = 0.08
        assert secondary_compression_settlement(0.004, 10.0, 1.0, 20.0) == pytest.approx(0.08, rel=1e-4)

    def test_at_tp(self):
        # t = tp => log10(1) = 0 => s_s = 0
        assert secondary_compression_settlement(0.004, 5.0, 5.0, 20.0) == pytest.approx(0.0, rel=1e-4)

    def test_hundred_times_tp(self):
        # C_epsilon_alpha=0.005, t=100, tp=1, H0=15
        # s_s = 0.005 * log10(100) * 15 = 0.005 * 2.0 * 15 = 0.15
        assert secondary_compression_settlement(0.005, 100.0, 1.0, 15.0) == pytest.approx(0.15, rel=1e-4)

    def test_tp_zero_raises(self):
        with pytest.raises(ValueError, match="tp must be positive"):
            secondary_compression_settlement(0.004, 10.0, 0.0, 20.0)

    def test_tp_negative_raises(self):
        with pytest.raises(ValueError, match="tp must be positive"):
            secondary_compression_settlement(0.004, 10.0, -1.0, 20.0)

    def test_t_less_than_tp_raises(self):
        with pytest.raises(ValueError, match="t must be greater than or equal to tp"):
            secondary_compression_settlement(0.004, 0.5, 1.0, 20.0)


# ===========================================================================
# 5-19  modified_secondary_compression_index_from_wn
# ===========================================================================

class TestModifiedSecondaryCompressionIndexFromWn:
    """Tests for Equation 5-19: C_epsilon_alpha = 1e-4 * wn."""

    def test_basic(self):
        # wn=40 => C_epsilon_alpha = 1e-4 * 40 = 0.004
        assert modified_secondary_compression_index_from_wn(40.0) == pytest.approx(0.004, rel=1e-4)

    def test_zero(self):
        assert modified_secondary_compression_index_from_wn(0.0) == pytest.approx(0.0, rel=1e-4)

    def test_high_water_content(self):
        # wn=200 (organic soil) => C_epsilon_alpha = 0.02
        assert modified_secondary_compression_index_from_wn(200.0) == pytest.approx(0.02, rel=1e-4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="wn must be non-negative"):
            modified_secondary_compression_index_from_wn(-5.0)


# ===========================================================================
# 5-20  surcharge_degree_of_consolidation
# ===========================================================================

class TestSurchargeDegreeOfConsolidation:
    """Tests for Equation 5-20: U_f+s = log10(1 + qf/sigma_z0) / log10(1 + (qf+qs)/sigma_z0)."""

    def test_basic(self):
        # qf=1000, sigma_z0=2000, qs=500
        # numerator = log10(1 + 1000/2000) = log10(1.5) = 0.17609
        # denominator = log10(1 + 1500/2000) = log10(1.75) = 0.24304
        # U = 0.17609 / 0.24304 = 0.72454
        expected = math.log10(1.5) / math.log10(1.75)
        assert surcharge_degree_of_consolidation(1000.0, 2000.0, 500.0) == pytest.approx(expected, rel=1e-4)

    def test_no_surcharge(self):
        # qs=0 => numerator = denominator => U = 1.0
        assert surcharge_degree_of_consolidation(1000.0, 2000.0, 0.0) == pytest.approx(1.0, rel=1e-4)

    def test_large_surcharge(self):
        # qf=500, sigma_z0=1000, qs=2000
        # num = log10(1 + 500/1000) = log10(1.5) = 0.17609
        # den = log10(1 + 2500/1000) = log10(3.5) = 0.54407
        # U = 0.17609/0.54407 = 0.32366
        expected = math.log10(1.5) / math.log10(3.5)
        assert surcharge_degree_of_consolidation(500.0, 1000.0, 2000.0) == pytest.approx(expected, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            surcharge_degree_of_consolidation(1000.0, 0.0, 500.0)

    def test_sigma_z0_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            surcharge_degree_of_consolidation(1000.0, -100.0, 500.0)

    def test_qf_negative_raises(self):
        with pytest.raises(ValueError, match="qf must be non-negative"):
            surcharge_degree_of_consolidation(-100.0, 2000.0, 500.0)

    def test_qs_negative_raises(self):
        with pytest.raises(ValueError, match="qs must be non-negative"):
            surcharge_degree_of_consolidation(1000.0, 2000.0, -100.0)

    def test_both_qf_and_qs_zero_raises(self):
        # qf=0, qs=0 => denominator = log10(1) = 0 => should raise
        with pytest.raises(ValueError, match="Denominator is zero"):
            surcharge_degree_of_consolidation(0.0, 2000.0, 0.0)


# ===========================================================================
# 5-21  surcharge_degree_of_consolidation_with_secondary
# ===========================================================================

class TestSurchargeDegreeOfConsolidationWithSecondary:
    """Tests for Equation 5-21: U_f+s = (log10(1+qf/sigma_z0) + C_alpha/Cc * log10(t/tp)) / log10(1+(qf+qs)/sigma_z0)."""

    def test_basic(self):
        # qf=1000, sigma_z0=2000, qs=500, C_alpha/Cc=0.04, t=50, tp=2
        # primary_term = log10(1.5) = 0.17609
        # secondary_term = 0.04 * log10(50/2) = 0.04 * log10(25) = 0.04 * 1.39794 = 0.055918
        # denominator = log10(1.75) = 0.24304
        # U = (0.17609 + 0.055918) / 0.24304 = 0.232008 / 0.24304 = 0.95459
        primary = math.log10(1.5)
        secondary = 0.04 * math.log10(25.0)
        denom = math.log10(1.75)
        expected = (primary + secondary) / denom
        result = surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, 500.0, 0.04, 50.0, 2.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_no_secondary(self):
        # t = tp => secondary term = 0, should match Eq 5-20
        result = surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, 500.0, 0.04, 2.0, 2.0)
        expected = surcharge_degree_of_consolidation(1000.0, 2000.0, 500.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_sigma_z0_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_z0 must be positive"):
            surcharge_degree_of_consolidation_with_secondary(1000.0, 0.0, 500.0, 0.04, 50.0, 2.0)

    def test_qf_negative_raises(self):
        with pytest.raises(ValueError, match="qf must be non-negative"):
            surcharge_degree_of_consolidation_with_secondary(-100.0, 2000.0, 500.0, 0.04, 50.0, 2.0)

    def test_qs_negative_raises(self):
        with pytest.raises(ValueError, match="qs must be non-negative"):
            surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, -100.0, 0.04, 50.0, 2.0)

    def test_tp_zero_raises(self):
        with pytest.raises(ValueError, match="tp must be positive"):
            surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, 500.0, 0.04, 50.0, 0.0)

    def test_tp_negative_raises(self):
        with pytest.raises(ValueError, match="tp must be positive"):
            surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, 500.0, 0.04, 50.0, -1.0)

    def test_t_less_than_tp_raises(self):
        with pytest.raises(ValueError, match="t must be greater than or equal to tp"):
            surcharge_degree_of_consolidation_with_secondary(1000.0, 2000.0, 500.0, 0.04, 1.0, 2.0)

    def test_zero_denominator_raises(self):
        with pytest.raises(ValueError, match="Denominator is zero"):
            surcharge_degree_of_consolidation_with_secondary(0.0, 2000.0, 0.0, 0.04, 2.0, 2.0)


# ===========================================================================
# 5-22  time_factor_radial
# ===========================================================================

class TestTimeFactorRadial:
    """Tests for Equation 5-22: Tr = (1/8) * (Fn + Fs + Fr) * ln(1/(1-Ur))."""

    def test_basic(self):
        # Ur=0.5, Fn=2.0, Fs=0, Fr=0
        # Tr = (1/8) * 2.0 * ln(1/0.5) = 0.25 * ln(2) = 0.25 * 0.693147 = 0.173287
        expected = (1.0 / 8.0) * 2.0 * math.log(2.0)
        assert time_factor_radial(0.5, 2.0) == pytest.approx(expected, rel=1e-4)

    def test_with_smear_and_resistance(self):
        # Ur=0.8, Fn=2.0, Fs=0.5, Fr=0.3
        # Tr = (1/8)*(2.0+0.5+0.3)*ln(1/0.2) = (1/8)*2.8*ln(5)
        # ln(5) = 1.60944
        # Tr = 0.35 * 1.60944 = 0.56330
        expected = (1.0 / 8.0) * 2.8 * math.log(5.0)
        assert time_factor_radial(0.8, 2.0, 0.5, 0.3) == pytest.approx(expected, rel=1e-4)

    def test_zero_ur(self):
        # Ur=0 => ln(1/1) = 0 => Tr = 0
        assert time_factor_radial(0.0, 2.0) == pytest.approx(0.0, rel=1e-4)

    def test_ur_negative_raises(self):
        with pytest.raises(ValueError, match="Ur must be in the range"):
            time_factor_radial(-0.1, 2.0)

    def test_ur_one_raises(self):
        with pytest.raises(ValueError, match="Ur must be in the range"):
            time_factor_radial(1.0, 2.0)

    def test_ur_above_one_raises(self):
        with pytest.raises(ValueError, match="Ur must be in the range"):
            time_factor_radial(1.5, 2.0)


# ===========================================================================
# 5-23  drain_spacing_factor
# ===========================================================================

class TestDrainSpacingFactor:
    """Tests for Equation 5-23: Fn = ln(n) - 0.75 (approx) or exact formula."""

    def test_approximate_basic(self):
        # n=20, approximate => Fn = ln(20) - 0.75 = 2.99573 - 0.75 = 2.24573
        expected = math.log(20.0) - 0.75
        assert drain_spacing_factor(20.0, approximate=True) == pytest.approx(expected, rel=1e-4)

    def test_approximate_default(self):
        # default is approximate=True
        expected = math.log(30.0) - 0.75
        assert drain_spacing_factor(30.0) == pytest.approx(expected, rel=1e-4)

    def test_exact_basic(self):
        # n=20, exact
        # n2 = 400
        # Fn = (400/399) * (ln(20) - (3*400 - 1)/(4*400))
        #    = 1.002506 * (2.995732 - 1199/1600)
        #    = 1.002506 * (2.995732 - 0.749375)
        #    = 1.002506 * 2.246357
        #    = 2.251988
        n = 20.0
        n2 = n ** 2
        expected = (n2 / (n2 - 1.0)) * (math.log(n) - (3.0 * n2 - 1.0) / (4.0 * n2))
        assert drain_spacing_factor(20.0, approximate=False) == pytest.approx(expected, rel=1e-4)

    def test_n_equals_one_raises(self):
        with pytest.raises(ValueError, match="n must be greater than 1"):
            drain_spacing_factor(1.0)

    def test_n_less_than_one_raises(self):
        with pytest.raises(ValueError, match="n must be greater than 1"):
            drain_spacing_factor(0.5)


# ===========================================================================
# 5-24  equivalent_drain_diameter
# ===========================================================================

class TestEquivalentDrainDiameter:
    """Tests for Equation 5-24: dw = 2*(a+b)/pi."""

    def test_basic(self):
        # a=4, b=0.25 => dw = 2*(4+0.25)/pi = 2*4.25/3.14159 = 8.5/3.14159 = 2.70563
        expected = 2.0 * (4.0 + 0.25) / math.pi
        assert equivalent_drain_diameter(4.0, 0.25) == pytest.approx(expected, rel=1e-4)

    def test_square_cross_section(self):
        # a=b=3 => dw = 2*6/pi = 12/pi = 3.81972
        expected = 12.0 / math.pi
        assert equivalent_drain_diameter(3.0, 3.0) == pytest.approx(expected, rel=1e-4)

    def test_zero_dimension(self):
        # a=0, b=5 => dw = 2*5/pi = 10/pi = 3.18310
        expected = 10.0 / math.pi
        assert equivalent_drain_diameter(0.0, 5.0) == pytest.approx(expected, rel=1e-4)

    def test_both_zero(self):
        assert equivalent_drain_diameter(0.0, 0.0) == pytest.approx(0.0, rel=1e-4)

    def test_a_negative_raises(self):
        with pytest.raises(ValueError, match="a must be non-negative"):
            equivalent_drain_diameter(-1.0, 0.25)

    def test_b_negative_raises(self):
        with pytest.raises(ValueError, match="b must be non-negative"):
            equivalent_drain_diameter(4.0, -0.5)


# ===========================================================================
# 5-25  smear_factor
# ===========================================================================

class TestSmearFactor:
    """Tests for Equation 5-25: Fs = (kh/ks) * ln(s)."""

    def test_basic(self):
        # kh=1e-6, ks=5e-7, s=3
        # Fs = (1e-6 / 5e-7) * ln(3) = 2.0 * 1.09861 = 2.19722
        expected = (1e-6 / 5e-7) * math.log(3.0)
        assert smear_factor(1e-6, 5e-7, 3.0) == pytest.approx(expected, rel=1e-4)

    def test_no_disturbance(self):
        # kh = ks => ratio = 1 => Fs = ln(s)
        # s=2 => Fs = ln(2) = 0.693147
        assert smear_factor(1e-6, 1e-6, 2.0) == pytest.approx(math.log(2.0), rel=1e-4)

    def test_ks_zero_raises(self):
        with pytest.raises(ValueError, match="ks must be positive"):
            smear_factor(1e-6, 0.0, 3.0)

    def test_ks_negative_raises(self):
        with pytest.raises(ValueError, match="ks must be positive"):
            smear_factor(1e-6, -1e-7, 3.0)

    def test_s_one_raises(self):
        with pytest.raises(ValueError, match="s must be greater than 1"):
            smear_factor(1e-6, 5e-7, 1.0)

    def test_s_less_than_one_raises(self):
        with pytest.raises(ValueError, match="s must be greater than 1"):
            smear_factor(1e-6, 5e-7, 0.5)


# ===========================================================================
# 5-26  well_resistance_factor
# ===========================================================================

class TestWellResistanceFactor:
    """Tests for Equation 5-26: Fr = pi * (kh/qw) * z * (Lm - z)."""

    def test_basic(self):
        # kh=1e-6, z=5, Lm=20, qw=0.001
        # Fr = pi * (1e-6 / 0.001) * 5 * (20-5)
        #    = pi * 0.001 * 5 * 15
        #    = pi * 0.075
        #    = 0.23562
        expected = math.pi * (1e-6 / 0.001) * 5.0 * 15.0
        assert well_resistance_factor(1e-6, 5.0, 20.0, 0.001) == pytest.approx(expected, rel=1e-4)

    def test_at_top(self):
        # z=0 => Fr = 0 (water enters at top, no resistance)
        assert well_resistance_factor(1e-6, 0.0, 20.0, 0.001) == pytest.approx(0.0, rel=1e-4)

    def test_at_bottom(self):
        # z = Lm => Fr = pi * (kh/qw) * Lm * 0 = 0
        assert well_resistance_factor(1e-6, 20.0, 20.0, 0.001) == pytest.approx(0.0, rel=1e-4)

    def test_midpoint(self):
        # z = Lm/2 = 10, Lm=20 => (Lm - z) = 10
        # Fr = pi * (1e-6/0.001) * 10 * 10 = pi * 0.001 * 100 = pi * 0.1 = 0.31416
        expected = math.pi * 0.001 * 100.0
        assert well_resistance_factor(1e-6, 10.0, 20.0, 0.001) == pytest.approx(expected, rel=1e-4)

    def test_qw_zero_raises(self):
        with pytest.raises(ValueError, match="qw must be positive"):
            well_resistance_factor(1e-6, 5.0, 20.0, 0.0)

    def test_qw_negative_raises(self):
        with pytest.raises(ValueError, match="qw must be positive"):
            well_resistance_factor(1e-6, 5.0, 20.0, -0.001)

    def test_z_negative_raises(self):
        with pytest.raises(ValueError, match="z must be non-negative"):
            well_resistance_factor(1e-6, -1.0, 20.0, 0.001)

    def test_z_exceeds_Lm_raises(self):
        with pytest.raises(ValueError, match="z must not exceed Lm"):
            well_resistance_factor(1e-6, 25.0, 20.0, 0.001)


# ===========================================================================
# 5-27  combined_degree_of_consolidation
# ===========================================================================

class TestCombinedDegreeOfConsolidation:
    """Tests for Equation 5-27: Uc = 100 - (100-Uz)*(100-Ur)/100."""

    def test_basic(self):
        # Uz=50, Ur=60
        # Uc = 100 - (50*40)/100 = 100 - 2000/100 = 100 - 20 = 80
        assert combined_degree_of_consolidation(50.0, 60.0) == pytest.approx(80.0, rel=1e-4)

    def test_no_vertical(self):
        # Uz=0 => Uc = 100 - (100*40)/100 = 100 - 40 = 60 = Ur
        assert combined_degree_of_consolidation(0.0, 60.0) == pytest.approx(60.0, rel=1e-4)

    def test_no_radial(self):
        # Ur=0 => Uc = 100 - (50*100)/100 = 100 - 50 = 50 = Uz
        assert combined_degree_of_consolidation(50.0, 0.0) == pytest.approx(50.0, rel=1e-4)

    def test_both_zero(self):
        assert combined_degree_of_consolidation(0.0, 0.0) == pytest.approx(0.0, rel=1e-4)

    def test_both_hundred(self):
        # Uc = 100 - 0*0/100 = 100
        assert combined_degree_of_consolidation(100.0, 100.0) == pytest.approx(100.0, rel=1e-4)

    def test_high_consolidation(self):
        # Uz=90, Ur=95 => Uc = 100 - (10*5)/100 = 100 - 0.5 = 99.5
        assert combined_degree_of_consolidation(90.0, 95.0) == pytest.approx(99.5, rel=1e-4)

    def test_Uz_negative_raises(self):
        with pytest.raises(ValueError, match="Uz must be between 0 and 100"):
            combined_degree_of_consolidation(-1.0, 50.0)

    def test_Uz_above_100_raises(self):
        with pytest.raises(ValueError, match="Uz must be between 0 and 100"):
            combined_degree_of_consolidation(101.0, 50.0)

    def test_Ur_negative_raises(self):
        with pytest.raises(ValueError, match="Ur must be between 0 and 100"):
            combined_degree_of_consolidation(50.0, -1.0)

    def test_Ur_above_100_raises(self):
        with pytest.raises(ValueError, match="Ur must be between 0 and 100"):
            combined_degree_of_consolidation(50.0, 101.0)


# ===========================================================================
# 5-28  required_drain_diameter
# ===========================================================================

class TestRequiredDrainDiameter:
    """Tests for Equation 5-28: dc = sqrt(ch * t / Tr)."""

    def test_basic(self):
        # ch=0.5, t=365, Tr=0.2
        # dc = sqrt(0.5 * 365 / 0.2) = sqrt(912.5) = 30.2076
        expected = math.sqrt(0.5 * 365.0 / 0.2)
        assert required_drain_diameter(0.5, 365.0, 0.2) == pytest.approx(expected, rel=1e-4)

    def test_zero_ch(self):
        # ch=0 => dc = 0
        assert required_drain_diameter(0.0, 365.0, 0.2) == pytest.approx(0.0, rel=1e-4)

    def test_zero_time(self):
        # t=0 => dc = 0
        assert required_drain_diameter(0.5, 0.0, 0.2) == pytest.approx(0.0, rel=1e-4)

    def test_large_Tr(self):
        # Large Tr => smaller dc
        # ch=1.0, t=100, Tr=10 => dc = sqrt(100/10) = sqrt(10) = 3.16228
        assert required_drain_diameter(1.0, 100.0, 10.0) == pytest.approx(math.sqrt(10.0), rel=1e-4)

    def test_Tr_zero_raises(self):
        with pytest.raises(ValueError, match="Tr must be positive"):
            required_drain_diameter(0.5, 365.0, 0.0)

    def test_Tr_negative_raises(self):
        with pytest.raises(ValueError, match="Tr must be positive"):
            required_drain_diameter(0.5, 365.0, -0.1)

    def test_ch_negative_raises(self):
        with pytest.raises(ValueError, match="ch must be non-negative"):
            required_drain_diameter(-0.1, 365.0, 0.2)

    def test_t_negative_raises(self):
        with pytest.raises(ValueError, match="t must be non-negative"):
            required_drain_diameter(0.5, -10.0, 0.2)
