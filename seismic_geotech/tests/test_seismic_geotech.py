"""
Tests for seismic geotechnical analysis module.

Covers: site classification, site coefficients, Mononobe-Okabe,
CSR/CRR/liquefaction, and residual strength.

References:
    AASHTO LRFD Section 3.10.3
    Youd et al. (2001)
    FHWA GEC-3
"""

import math
import pytest

from seismic_geotech.site_class import (
    compute_vs30, compute_n_bar, compute_su_bar,
    classify_site, site_coefficients,
)
from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, mononobe_okabe_KPE, seismic_earth_pressure,
)
from seismic_geotech.liquefaction import (
    stress_reduction_rd, magnitude_scaling_factor, compute_CSR,
    fines_correction, CRR_from_N160cs, evaluate_liquefaction,
)
from seismic_geotech.residual_strength import (
    Sr_seed_harder, Sr_idriss_boulanger, post_liquefaction_strength,
)
from seismic_geotech.results import (
    SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
)


# ================================================================
# Site classification
# ================================================================
class TestSiteClassification:
    def test_class_A(self):
        assert classify_site(vs30=1600) == "A"

    def test_class_B(self):
        assert classify_site(vs30=800) == "B"

    def test_class_C_vs30(self):
        assert classify_site(vs30=400) == "C"

    def test_class_D_vs30(self):
        assert classify_site(vs30=250) == "D"

    def test_class_E_vs30(self):
        assert classify_site(vs30=150) == "E"

    def test_class_C_nbar(self):
        assert classify_site(n_bar=55) == "C"

    def test_class_D_nbar(self):
        assert classify_site(n_bar=25) == "D"

    def test_class_E_nbar(self):
        assert classify_site(n_bar=10) == "E"

    def test_class_C_subar(self):
        assert classify_site(su_bar=120) == "C"

    def test_class_D_subar(self):
        assert classify_site(su_bar=75) == "D"

    def test_class_E_subar(self):
        assert classify_site(su_bar=30) == "E"

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            classify_site()


class TestVs30Computation:
    def test_uniform_profile(self):
        """Single layer with Vs=300 m/s for 30m -> Vs30=300."""
        vs30 = compute_vs30([30.0], [300.0])
        assert abs(vs30 - 300.0) < 0.1

    def test_two_layer(self):
        """10m at 200 m/s + 20m at 400 m/s -> harmonic mean."""
        vs30 = compute_vs30([10.0, 20.0], [200.0, 400.0])
        expected = 30.0 / (10.0 / 200.0 + 20.0 / 400.0)  # 300.0
        assert abs(vs30 - expected) < 0.1

    def test_clips_to_30m(self):
        vs30 = compute_vs30([20.0, 20.0], [200.0, 400.0])
        expected = 30.0 / (20.0 / 200.0 + 10.0 / 400.0)
        assert abs(vs30 - expected) < 0.1


class TestNBarComputation:
    def test_uniform(self):
        n_bar = compute_n_bar([30.0], [20.0])
        assert abs(n_bar - 20.0) < 0.1

    def test_two_layer(self):
        n_bar = compute_n_bar([15.0, 15.0], [10.0, 30.0])
        expected = 30.0 / (15.0 / 10.0 + 15.0 / 30.0)
        assert abs(n_bar - expected) < 0.1


class TestSiteCoefficients:
    def test_class_B_all_ones(self):
        result = site_coefficients("B", Ss=0.5, S1=0.2)
        assert result.Fa == 1.0
        assert result.Fv == 1.0
        assert result.site_class == "B"

    def test_class_D_low_Ss(self):
        result = site_coefficients("D", Ss=0.25, S1=0.10)
        assert result.Fa == 1.6
        assert result.Fv == 2.4

    def test_class_E_high_Ss(self):
        result = site_coefficients("E", Ss=1.25, S1=0.50)
        assert result.Fa == 0.9
        assert result.Fv == 2.4

    def test_interpolation(self):
        """Ss=0.375 for class D: interpolate between 0.25(1.6) and 0.50(1.4)."""
        result = site_coefficients("D", Ss=0.375, S1=0.15)
        assert abs(result.Fa - 1.5) < 0.01

    def test_class_F_raises(self):
        with pytest.raises(ValueError):
            site_coefficients("F", Ss=0.5, S1=0.2)

    def test_summary_and_dict(self):
        result = site_coefficients("D", Ss=0.5, S1=0.2)
        assert "Site Class: D" in result.summary()
        d = result.to_dict()
        assert "Fa" in d
        assert "SDS_g" in d


# ================================================================
# Mononobe-Okabe
# ================================================================
class TestMononobeOkabe:
    def test_KAE_greater_than_KA(self):
        """Seismic active > static active."""
        Ka_static = math.tan(math.pi / 4 - math.radians(30) / 2) ** 2
        KAE = mononobe_okabe_KAE(30, 15, 0.2)
        assert KAE > Ka_static

    def test_KAE_zero_kh_equals_coulomb(self):
        """With kh=0, KAE should equal Coulomb Ka."""
        KAE = mononobe_okabe_KAE(30, 15, 0.0)
        # Coulomb Ka with delta=15, vertical wall, horizontal backfill
        from sheet_pile.earth_pressure import coulomb_Ka
        Ka_coulomb = coulomb_Ka(30, 15)
        assert abs(KAE - Ka_coulomb) < 0.001

    def test_KAE_increases_with_kh(self):
        KAE_low = mononobe_okabe_KAE(30, 15, 0.1)
        KAE_high = mononobe_okabe_KAE(30, 15, 0.3)
        assert KAE_high > KAE_low

    def test_KPE_less_than_KP(self):
        """Seismic passive < static Coulomb passive."""
        from sheet_pile.earth_pressure import coulomb_Kp
        Kp_static = coulomb_Kp(30, 15)
        KPE = mononobe_okabe_KPE(30, 15, 0.2)
        assert KPE < Kp_static

    def test_KPE_zero_kh_equals_coulomb(self):
        KPE = mononobe_okabe_KPE(30, 15, 0.0)
        from sheet_pile.earth_pressure import coulomb_Kp
        Kp_coulomb = coulomb_Kp(30, 15)
        assert abs(KPE - Kp_coulomb) < 0.01

    # ---- Tests with non-zero wall batter (beta) and backfill slope (i) ----
    # Reference values computed via AASHTO LRFD alpha-form (sin-based) of M-O.

    def test_KAE_battered_wall(self):
        """phi=30, delta=15, kh=0.2, beta=10 deg wall batter."""
        KAE = mononobe_okabe_KAE(30, 15, 0.2, beta_deg=10)
        assert abs(KAE - 0.3799) < 0.001

    def test_KAE_sloping_backfill(self):
        """phi=30, delta=15, kh=0.2, i=10 deg backfill slope."""
        KAE = mononobe_okabe_KAE(30, 15, 0.2, i_deg=10)
        assert abs(KAE - 0.5619) < 0.001

    def test_KAE_battered_and_sloping(self):
        """phi=30, delta=15, kh=0.2, beta=10, i=5."""
        KAE = mononobe_okabe_KAE(30, 15, 0.2, beta_deg=10, i_deg=5)
        assert abs(KAE - 0.4153) < 0.001

    def test_KAE_large_beta(self):
        """phi=35, delta=20, kh=0.15, beta=15."""
        KAE = mononobe_okabe_KAE(35, 20, 0.15, beta_deg=15)
        assert abs(KAE - 0.2375) < 0.001

    def test_KAE_with_kv_beta_i(self):
        """phi=30, delta=15, kh=0.2, kv=0.1, beta=10, i=5."""
        KAE = mononobe_okabe_KAE(30, 15, 0.2, kv=0.1, beta_deg=10, i_deg=5)
        assert abs(KAE - 0.4404) < 0.001

    def test_KPE_battered_wall(self):
        """phi=30, delta=15, kh=0.2, beta=10."""
        KPE = mononobe_okabe_KPE(30, 15, 0.2, beta_deg=10)
        assert abs(KPE - 3.1601) < 0.01

    def test_KPE_sloping_backfill(self):
        """phi=30, delta=15, kh=0.2, i=10."""
        KPE = mononobe_okabe_KPE(30, 15, 0.2, i_deg=10)
        assert abs(KPE - 5.3948) < 0.01

    def test_KPE_battered_and_sloping(self):
        """phi=30, delta=15, kh=0.2, beta=10, i=5."""
        KPE = mononobe_okabe_KPE(30, 15, 0.2, beta_deg=10, i_deg=5)
        assert abs(KPE - 4.1166) < 0.01

    def test_KPE_large_beta(self):
        """phi=35, delta=20, kh=0.15, beta=15."""
        KPE = mononobe_okabe_KPE(35, 20, 0.15, beta_deg=15)
        assert abs(KPE - 5.5431) < 0.01

    def test_KPE_with_kv_beta_i(self):
        """phi=30, delta=15, kh=0.2, kv=0.1, beta=10, i=5."""
        KPE = mononobe_okabe_KPE(30, 15, 0.2, kv=0.1, beta_deg=10, i_deg=5)
        assert abs(KPE - 3.9437) < 0.01

    def test_KAE_zero_kh_battered_equals_coulomb(self):
        """With kh=0, KAE(beta=10) should equal Coulomb Ka(alpha=100)."""
        from sheet_pile.earth_pressure import coulomb_Ka
        for beta_d in [5, 10, 15]:
            KAE = mononobe_okabe_KAE(30, 15, 0.0, beta_deg=beta_d)
            Ka_c = coulomb_Ka(30, 15, alpha_deg=90 + beta_d)
            assert abs(KAE - Ka_c) < 0.001, (
                f"beta={beta_d}: KAE={KAE:.4f} != Ka_coulomb={Ka_c:.4f}"
            )

    def test_KPE_zero_kh_battered_equals_coulomb(self):
        """With kh=0, KPE(beta=10) should equal Coulomb Kp(alpha=100)."""
        from sheet_pile.earth_pressure import coulomb_Kp
        for beta_d in [5, 10, 15]:
            KPE = mononobe_okabe_KPE(30, 15, 0.0, beta_deg=beta_d)
            Kp_c = coulomb_Kp(30, 15, alpha_deg=90 + beta_d)
            assert abs(KPE - Kp_c) < 0.01, (
                f"beta={beta_d}: KPE={KPE:.4f} != Kp_coulomb={Kp_c:.4f}"
            )

    def test_KAE_zero_kh_sloping_equals_coulomb(self):
        """With kh=0, KAE(i=10) should equal Coulomb Ka(beta=10)."""
        from sheet_pile.earth_pressure import coulomb_Ka
        for i_d in [5, 10]:
            KAE = mononobe_okabe_KAE(30, 15, 0.0, i_deg=i_d)
            Ka_c = coulomb_Ka(30, 15, beta_deg=i_d)
            assert abs(KAE - Ka_c) < 0.001, (
                f"i={i_d}: KAE={KAE:.4f} != Ka_coulomb={Ka_c:.4f}"
            )


class TestSeismicPressure:
    def test_increment_positive(self):
        Ka = math.tan(math.pi / 4 - math.radians(30) / 2) ** 2
        KAE = mononobe_okabe_KAE(30, 15, 0.2)
        result = seismic_earth_pressure(18.0, 6.0, KAE, Ka)
        assert result["delta_PAE_kN_per_m"] > 0

    def test_application_height(self):
        result = seismic_earth_pressure(18.0, 6.0, 0.5, 0.33)
        assert abs(result["height_of_application_m"] - 3.6) < 0.01

    def test_static_force(self):
        Ka = 0.333
        result = seismic_earth_pressure(18.0, 6.0, 0.5, Ka)
        expected = 0.5 * 18.0 * 6.0**2 * Ka
        assert abs(result["PA_static_kN_per_m"] - expected) < 0.1


# ================================================================
# Liquefaction CSR/CRR
# ================================================================
class TestStressReduction:
    def test_rd_surface(self):
        assert abs(stress_reduction_rd(0) - 1.0) < 0.001

    def test_rd_5m(self):
        expected = 1.0 - 0.00765 * 5.0
        assert abs(stress_reduction_rd(5.0) - expected) < 0.001

    def test_rd_15m(self):
        expected = 1.174 - 0.0267 * 15.0
        assert abs(stress_reduction_rd(15.0) - expected) < 0.001

    def test_rd_decreasing(self):
        assert stress_reduction_rd(10) < stress_reduction_rd(5)


class TestCSR:
    def test_basic_csr(self):
        """CSR = 0.65 * 0.3 * (180/90) * rd at 10m."""
        rd = stress_reduction_rd(10.0)
        CSR = compute_CSR(0.3, 180, 90, 10.0)
        expected = 0.65 * 0.3 * (180 / 90) * rd
        assert abs(CSR - expected) < 0.001

    def test_magnitude_scaling(self):
        """M=6.0 should give different CSR than M=7.5."""
        CSR_75 = compute_CSR(0.3, 180, 90, 5.0, M=7.5)
        CSR_60 = compute_CSR(0.3, 180, 90, 5.0, M=6.0)
        # M=6.0 has larger MSF, so CSR_M7.5 is smaller
        assert CSR_60 < CSR_75

    def test_msf_at_7_5(self):
        """MSF at M=7.5 should be approximately 1.0."""
        msf = magnitude_scaling_factor(7.5)
        assert abs(msf - 1.0) < 0.15  # approximately 1


class TestCRR:
    def test_low_N(self):
        """Low N160cs -> low CRR."""
        CRR = CRR_from_N160cs(5)
        assert 0.04 < CRR < 0.15

    def test_medium_N(self):
        CRR = CRR_from_N160cs(15)
        assert 0.1 < CRR < 0.3

    def test_high_N(self):
        """N160cs approaching 30 -> CRR increases sharply."""
        CRR = CRR_from_N160cs(28)
        assert CRR > 0.3

    def test_dense_sand(self):
        """N160cs >= 30 -> too dense to liquefy."""
        assert CRR_from_N160cs(30) == 2.0
        assert CRR_from_N160cs(40) == 2.0

    def test_monotonically_increasing(self):
        prev = 0
        for N in range(0, 30, 2):
            crr = CRR_from_N160cs(N)
            assert crr >= prev
            prev = crr


class TestFinesCorrection:
    def test_clean_sand(self):
        """FC <= 5%: no correction."""
        assert fines_correction(15, 3) == 15.0

    def test_moderate_fines(self):
        """FC = 20%: some correction."""
        N160cs = fines_correction(15, 20)
        assert N160cs > 15.0

    def test_high_fines(self):
        """FC >= 35%: alpha=5, beta=1.2."""
        N160cs = fines_correction(15, 40)
        expected = 5.0 + 1.2 * 15
        assert abs(N160cs - expected) < 0.01


# ================================================================
# Full liquefaction evaluation
# ================================================================
class TestLiquefactionEvaluation:
    def test_basic_evaluation(self):
        depths = [3, 6, 9, 12]
        N160 = [10, 12, 8, 25]
        FC = [5, 10, 15, 5]
        gamma = [18, 18, 18, 18]
        results = evaluate_liquefaction(depths, N160, FC, gamma, 0.3, 2.0)
        assert len(results) == 4
        for r in results:
            assert "FOS_liq" in r
            assert "liquefiable" in r

    def test_dense_layer_not_liquefiable(self):
        """Dense sand (N160=35) should not liquefy."""
        results = evaluate_liquefaction([5], [35], [3], [18], 0.3, 2.0)
        assert results[0]["liquefiable"] is False
        assert results[0]["FOS_liq"] > 1.0

    def test_loose_layer_liquefiable(self):
        """Loose sand (N160=5) at shallow depth should liquefy at high PGA."""
        results = evaluate_liquefaction([4], [5], [3], [18], 0.4, 2.0)
        assert results[0]["liquefiable"] is True

    def test_above_gwt_not_liquefiable(self):
        """Layer above GWT: sigma_v_eff = sigma_v, CSR lower."""
        results = evaluate_liquefaction([2], [10], [5], [18], 0.3, 5.0)
        # Above GWT, sigma_v/sigma_v' = 1.0, CSR is lower
        assert results[0]["sigma_v_eff_kPa"] == results[0]["sigma_v_kPa"]

    def test_result_container(self):
        layer_results = evaluate_liquefaction(
            [3, 6], [8, 25], [5, 5], [18, 18], 0.3, 2.0
        )
        result = LiquefactionResult(
            layer_results=layer_results, amax_g=0.3, magnitude=7.5, gwt_depth=2.0
        )
        assert result.n_liquefiable >= 0
        assert result.min_FOS > 0
        summary = result.summary()
        assert "LIQUEFACTION" in summary
        d = result.to_dict()
        assert "min_FOS_liq" in d


# ================================================================
# Residual strength
# ================================================================
class TestResidualStrength:
    def test_seed_harder_zero(self):
        assert Sr_seed_harder(0) == 0.0

    def test_seed_harder_moderate(self):
        Sr = Sr_seed_harder(12)
        assert abs(Sr - 10.0) < 0.1

    def test_seed_harder_increases(self):
        assert Sr_seed_harder(20) > Sr_seed_harder(10)

    def test_idriss_boulanger(self):
        Sr = Sr_idriss_boulanger(15, 100.0)
        assert Sr > 0
        assert Sr < 100.0  # Sr < sigma_v' (ratio capped at 0.6)

    def test_idriss_increases_with_N(self):
        Sr_low = Sr_idriss_boulanger(5, 100)
        Sr_high = Sr_idriss_boulanger(20, 100)
        assert Sr_high > Sr_low

    def test_dispatch_function(self):
        Sr1 = post_liquefaction_strength(12, method="seed_harder")
        Sr2 = post_liquefaction_strength(12, sigma_v_eff=100, method="idriss_boulanger")
        assert Sr1 > 0
        assert Sr2 > 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            post_liquefaction_strength(10, method="unknown")
