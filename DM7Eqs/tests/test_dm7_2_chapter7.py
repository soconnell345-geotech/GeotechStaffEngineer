"""Comprehensive tests for geotech.dm7_2.chapter7 module.

UFC 3-220-20, Chapter 7: Probability and Reliability in Geotechnical Engineering
Equations 7-1 through 7-16 plus convenience/composite functions.
"""

import math

import pytest

from geotech.dm7_2.chapter7 import *


# ============================================================================
# 1. sample_mean
# ============================================================================


class TestSampleMean:
    """Tests for sample_mean (Table 7-1)."""

    def test_basic_valid(self):
        # mean of [2, 4, 6, 8] = 20/4 = 5.0
        assert sample_mean([2, 4, 6, 8]) == pytest.approx(5.0, rel=1e-4)

    def test_single_value(self):
        # mean of a single value is itself
        assert sample_mean([7.5]) == pytest.approx(7.5, rel=1e-4)

    def test_negative_values(self):
        # mean of [-3, -1, 1, 3] = 0/4 = 0.0
        assert sample_mean([-3, -1, 1, 3]) == pytest.approx(0.0, abs=1e-10)

    def test_identical_values(self):
        assert sample_mean([4.0, 4.0, 4.0]) == pytest.approx(4.0, rel=1e-4)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one value"):
            sample_mean([])


# ============================================================================
# 2. sample_variance
# ============================================================================


class TestSampleVariance:
    """Tests for sample_variance (Table 7-1)."""

    def test_basic_valid(self):
        # x = [2, 4, 6, 8], mean = 5.0
        # deviations: -3, -1, 1, 3 -> squares: 9, 1, 1, 9 -> sum = 20
        # variance = 20 / 3 = 6.6667
        assert sample_variance([2, 4, 6, 8]) == pytest.approx(
            20.0 / 3.0, rel=1e-4
        )

    def test_two_values(self):
        # x = [3, 7], mean = 5. deviations: -2, 2 -> squares: 4, 4 -> sum=8
        # variance = 8 / 1 = 8.0
        assert sample_variance([3, 7]) == pytest.approx(8.0, rel=1e-4)

    def test_identical_values(self):
        # All same => variance is 0
        assert sample_variance([5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-10)

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            sample_variance([10.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            sample_variance([])


# ============================================================================
# 3. sample_standard_deviation
# ============================================================================


class TestSampleStandardDeviation:
    """Tests for sample_standard_deviation (Table 7-1)."""

    def test_basic_valid(self):
        # Same dataset: variance = 20/3, std = sqrt(20/3) = 2.58199
        assert sample_standard_deviation([2, 4, 6, 8]) == pytest.approx(
            math.sqrt(20.0 / 3.0), rel=1e-4
        )

    def test_two_values(self):
        # variance = 8.0, std = sqrt(8) = 2.82843
        assert sample_standard_deviation([3, 7]) == pytest.approx(
            math.sqrt(8.0), rel=1e-4
        )

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            sample_standard_deviation([42.0])


# ============================================================================
# 4. coefficient_of_variation
# ============================================================================


class TestCoefficientOfVariation:
    """Tests for coefficient_of_variation (Table 7-1)."""

    def test_basic_valid(self):
        # x = [2, 4, 6, 8], mean = 5.0, std = sqrt(20/3)
        # COV = sqrt(20/3) / 5.0
        expected = math.sqrt(20.0 / 3.0) / 5.0
        assert coefficient_of_variation([2, 4, 6, 8]) == pytest.approx(
            expected, rel=1e-4
        )

    def test_zero_mean_raises(self):
        # [-1, 1] has mean 0 -> COV undefined
        with pytest.raises(ValueError, match="mean is zero"):
            coefficient_of_variation([-1, 1])

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least two values"):
            coefficient_of_variation([5.0])


# ============================================================================
# 5. coefficient_of_variation_from_params
# ============================================================================


class TestCoefficientOfVariationFromParams:
    """Tests for coefficient_of_variation_from_params (Table 7-1)."""

    def test_basic_valid(self):
        # COV = 2.5 / 10.0 = 0.25
        assert coefficient_of_variation_from_params(2.5, 10.0) == pytest.approx(
            0.25, rel=1e-4
        )

    def test_zero_std_dev(self):
        # COV = 0 / 5.0 = 0.0
        assert coefficient_of_variation_from_params(0.0, 5.0) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="std_dev must be non-negative"):
            coefficient_of_variation_from_params(-1.0, 10.0)

    def test_zero_mean_raises(self):
        with pytest.raises(ValueError, match="mean must be non-zero"):
            coefficient_of_variation_from_params(2.0, 0.0)


# ============================================================================
# 6. cumulative_mass_function (Equation 7-1)
# ============================================================================


class TestCumulativeMassFunction:
    """Tests for cumulative_mass_function (Equation 7-1)."""

    def test_basic_valid(self):
        # PMF = [0.2, 0.3, 0.5], CMF at index 1 = 0.2 + 0.3 = 0.5
        assert cumulative_mass_function([0.2, 0.3, 0.5], 1) == pytest.approx(
            0.5, rel=1e-4
        )

    def test_first_index(self):
        # CMF at index 0 = first PMF value
        assert cumulative_mass_function([0.2, 0.3, 0.5], 0) == pytest.approx(
            0.2, rel=1e-4
        )

    def test_last_index(self):
        # CMF at last index should be 1.0
        assert cumulative_mass_function([0.2, 0.3, 0.5], 2) == pytest.approx(
            1.0, rel=1e-4
        )

    def test_single_outcome(self):
        # One outcome with probability 1.0
        assert cumulative_mass_function([1.0], 0) == pytest.approx(1.0, rel=1e-4)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            cumulative_mass_function([], 0)

    def test_pmf_value_out_of_range_negative_raises(self):
        with pytest.raises(ValueError, match="range \\[0, 1\\]"):
            cumulative_mass_function([-0.1, 0.6, 0.5], 0)

    def test_pmf_value_out_of_range_above_one_raises(self):
        with pytest.raises(ValueError, match="range \\[0, 1\\]"):
            cumulative_mass_function([1.1, 0.0, -0.1], 0)

    def test_pmf_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            cumulative_mass_function([0.2, 0.3, 0.3], 0)

    def test_index_negative_raises(self):
        with pytest.raises(ValueError, match="index must be in the range"):
            cumulative_mass_function([0.5, 0.5], -1)

    def test_index_too_large_raises(self):
        with pytest.raises(ValueError, match="index must be in the range"):
            cumulative_mass_function([0.5, 0.5], 2)


# ============================================================================
# 7. cumulative_density_function (Equation 7-2)
# ============================================================================


class TestCumulativeDensityFunction:
    """Tests for cumulative_density_function (Equation 7-2)."""

    def test_uniform_pdf(self):
        # Uniform PDF on [0, 1]: f(x) = 1 for x in [0,1], 0 otherwise
        # CDF at x=0.5 with lower_bound=0 should be ~0.5
        def uniform_pdf(x):
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        result = cumulative_density_function(uniform_pdf, 0.5, lower_bound=0.0)
        assert result == pytest.approx(0.5, rel=1e-3)

    def test_uniform_pdf_full_range(self):
        # CDF at x=1.0 should be ~1.0
        def uniform_pdf(x):
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        result = cumulative_density_function(uniform_pdf, 1.0, lower_bound=0.0)
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_standard_normal_at_zero(self):
        # Standard normal CDF at 0 should be ~0.5
        def std_normal_pdf(x):
            return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)

        result = cumulative_density_function(
            std_normal_pdf, 0.0, lower_bound=-10.0, n_steps=5000
        )
        assert result == pytest.approx(0.5, rel=1e-3)

    def test_x0_below_lower_bound_returns_zero(self):
        # When x0 < lower_bound, function returns 0.0
        def dummy_pdf(x):
            return 1.0

        result = cumulative_density_function(dummy_pdf, -20.0, lower_bound=-10.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_n_steps_less_than_one_raises(self):
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            cumulative_density_function(lambda x: 0.0, 0.0, n_steps=0)


# ============================================================================
# 8. pdf_from_cdf (Equation 7-3)
# ============================================================================


class TestPdfFromCdf:
    """Tests for pdf_from_cdf (Equation 7-3)."""

    def test_linear_cdf(self):
        # CDF = x => PDF = 1.0 everywhere
        result = pdf_from_cdf(lambda x: x, 5.0)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_quadratic_cdf(self):
        # CDF = x^2 => PDF = 2x. At x = 3, PDF = 6
        result = pdf_from_cdf(lambda x: x ** 2, 3.0)
        assert result == pytest.approx(6.0, rel=1e-3)

    def test_standard_normal_cdf_at_zero(self):
        # PDF of standard normal at 0 = 1/sqrt(2*pi) ~ 0.39894
        def std_normal_cdf(x):
            return 0.5 * math.erfc(-x / math.sqrt(2.0))

        result = pdf_from_cdf(std_normal_cdf, 0.0)
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert result == pytest.approx(expected, rel=1e-3)

    def test_dx_not_positive_raises(self):
        with pytest.raises(ValueError, match="dx must be positive"):
            pdf_from_cdf(lambda x: x, 0.0, dx=0.0)

    def test_dx_negative_raises(self):
        with pytest.raises(ValueError, match="dx must be positive"):
            pdf_from_cdf(lambda x: x, 0.0, dx=-0.001)


# ============================================================================
# 9. probability_over_interval (Equation 7-4)
# ============================================================================


class TestProbabilityOverInterval:
    """Tests for probability_over_interval (Equation 7-4)."""

    def test_uniform_full_interval(self):
        # Uniform on [0, 1], probability over [0, 1] = 1.0
        def uniform_pdf(x):
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        result = probability_over_interval(uniform_pdf, 0.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_uniform_half_interval(self):
        # Uniform on [0, 1], probability over [0, 0.5] = 0.5
        def uniform_pdf(x):
            return 1.0 if 0.0 <= x <= 1.0 else 0.0

        result = probability_over_interval(uniform_pdf, 0.0, 0.5)
        assert result == pytest.approx(0.5, rel=1e-3)

    def test_equal_bounds_returns_zero(self):
        # x1 == x2 => probability is 0
        result = probability_over_interval(lambda x: 1.0, 3.0, 3.0)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_x2_less_than_x1_raises(self):
        with pytest.raises(ValueError, match="x2 must be greater than or equal to x1"):
            probability_over_interval(lambda x: 1.0, 5.0, 3.0)

    def test_n_steps_less_than_one_raises(self):
        with pytest.raises(ValueError, match="n_steps must be at least 1"):
            probability_over_interval(lambda x: 1.0, 0.0, 1.0, n_steps=0)


# ============================================================================
# 10. combined_cov (Equation 7-5)
# ============================================================================


class TestCombinedCov:
    """Tests for combined_cov (Equation 7-5)."""

    def test_basic_valid(self):
        # sqrt(0.1^2 + 0.2^2 + 0.3^2) = sqrt(0.01 + 0.04 + 0.09)
        # = sqrt(0.14) = 0.37417
        assert combined_cov(0.1, 0.2, 0.3) == pytest.approx(
            math.sqrt(0.14), rel=1e-4
        )

    def test_one_source_dominates(self):
        # sqrt(0^2 + 0^2 + 0.5^2) = 0.5
        assert combined_cov(0.0, 0.0, 0.5) == pytest.approx(0.5, rel=1e-4)

    def test_all_zero(self):
        assert combined_cov(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)

    def test_equal_components(self):
        # sqrt(3 * 0.2^2) = 0.2 * sqrt(3) = 0.34641
        assert combined_cov(0.2, 0.2, 0.2) == pytest.approx(
            0.2 * math.sqrt(3.0), rel=1e-4
        )

    def test_negative_cov_w_raises(self):
        with pytest.raises(ValueError, match="cov_w must be non-negative"):
            combined_cov(-0.1, 0.2, 0.3)

    def test_negative_cov_e_raises(self):
        with pytest.raises(ValueError, match="cov_e must be non-negative"):
            combined_cov(0.1, -0.2, 0.3)

    def test_negative_cov_t_raises(self):
        with pytest.raises(ValueError, match="cov_t must be non-negative"):
            combined_cov(0.1, 0.2, -0.3)


# ============================================================================
# 11. std_dev_from_range (Equation 7-6)
# ============================================================================


class TestStdDevFromRange:
    """Tests for std_dev_from_range (Equation 7-6)."""

    def test_basic_valid_default_n(self):
        # sigma = (100 - 40) / 6 = 60 / 6 = 10.0
        assert std_dev_from_range(100.0, 40.0) == pytest.approx(10.0, rel=1e-4)

    def test_custom_n(self):
        # sigma = (100 - 40) / 4 = 60 / 4 = 15.0
        assert std_dev_from_range(100.0, 40.0, n=4.0) == pytest.approx(
            15.0, rel=1e-4
        )

    def test_small_range(self):
        # sigma = (10.5 - 10.0) / 6 = 0.5 / 6 = 0.08333
        assert std_dev_from_range(10.5, 10.0) == pytest.approx(
            0.5 / 6.0, rel=1e-4
        )

    def test_hcv_equal_lcv_raises(self):
        with pytest.raises(ValueError, match="hcv must be greater than lcv"):
            std_dev_from_range(50.0, 50.0)

    def test_hcv_less_than_lcv_raises(self):
        with pytest.raises(ValueError, match="hcv must be greater than lcv"):
            std_dev_from_range(40.0, 100.0)

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            std_dev_from_range(100.0, 40.0, n=0.0)

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="n must be positive"):
            std_dev_from_range(100.0, 40.0, n=-3.0)


# ============================================================================
# 12. reliability_index (Equation 7-7)
# ============================================================================


class TestReliabilityIndex:
    """Tests for reliability_index (Equation 7-7)."""

    def test_basic_valid(self):
        # beta = 10.0 / 2.5 = 4.0
        assert reliability_index(10.0, 2.5) == pytest.approx(4.0, rel=1e-4)

    def test_negative_mu_g(self):
        # beta = -5.0 / 2.0 = -2.5 (valid: negative beta means mean is
        # on the failure side)
        assert reliability_index(-5.0, 2.0) == pytest.approx(-2.5, rel=1e-4)

    def test_zero_mu_g(self):
        # beta = 0 / 3.0 = 0.0
        assert reliability_index(0.0, 3.0) == pytest.approx(0.0, abs=1e-10)

    def test_sigma_g_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_g must be positive"):
            reliability_index(10.0, 0.0)

    def test_sigma_g_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_g must be positive"):
            reliability_index(10.0, -1.0)


# ============================================================================
# 13. fosm_mean (Equation 7-8)
# ============================================================================


class TestFosmMean:
    """Tests for fosm_mean (Equation 7-8)."""

    def test_basic_valid(self):
        # g(x, y) = x - y, means = [10, 4] => g(10, 4) = 6
        def g(x, y):
            return x - y

        assert fosm_mean(g, [10.0, 4.0]) == pytest.approx(6.0, rel=1e-4)

    def test_single_variable(self):
        # g(x) = 2*x + 3, mean = [5] => g(5) = 13
        def g(x):
            return 2.0 * x + 3.0

        assert fosm_mean(g, [5.0]) == pytest.approx(13.0, rel=1e-4)

    def test_nonlinear_function(self):
        # g(x, y) = x*y, means = [3, 4] => g(3, 4) = 12
        def g(x, y):
            return x * y

        assert fosm_mean(g, [3.0, 4.0]) == pytest.approx(12.0, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="at least one value"):
            fosm_mean(lambda: 0.0, [])


# ============================================================================
# 14. fosm_variance_analytical (Equation 7-9)
# ============================================================================


class TestFosmVarianceAnalytical:
    """Tests for fosm_variance_analytical (Equation 7-9)."""

    def test_basic_valid(self):
        # partials = [2, 3], std_devs = [1.0, 2.0]
        # variance = (2^2)(1^2) + (3^2)(2^2) = 4 + 36 = 40
        assert fosm_variance_analytical([2.0, 3.0], [1.0, 2.0]) == pytest.approx(
            40.0, rel=1e-4
        )

    def test_single_variable(self):
        # partials = [5], std_devs = [3]
        # variance = 25 * 9 = 225
        assert fosm_variance_analytical([5.0], [3.0]) == pytest.approx(
            225.0, rel=1e-4
        )

    def test_zero_partial(self):
        # partials = [0, 4], std_devs = [10, 2]
        # variance = 0 + 16*4 = 64
        assert fosm_variance_analytical([0.0, 4.0], [10.0, 2.0]) == pytest.approx(
            64.0, rel=1e-4
        )

    def test_zero_std_dev(self):
        # partials = [3, 5], std_devs = [0, 2]
        # variance = 0 + 25*4 = 100
        assert fosm_variance_analytical([3.0, 5.0], [0.0, 2.0]) == pytest.approx(
            100.0, rel=1e-4
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fosm_variance_analytical([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fosm_variance_analytical([1.0, 2.0], [3.0])

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fosm_variance_analytical([1.0], [-2.0])


# ============================================================================
# 15. fosm_central_difference (Equation 7-10)
# ============================================================================


class TestFosmCentralDifference:
    """Tests for fosm_central_difference (Equation 7-10)."""

    def test_linear_function_var0(self):
        # g(x, y) = 3*x + 2*y, means = [5, 10], std_devs = [1, 2]
        # dg_0 = g(5+1, 10) - g(5-1, 10) = g(6,10) - g(4,10)
        #       = (18+20) - (12+20) = 38 - 32 = 6
        def g(x, y):
            return 3.0 * x + 2.0 * y

        result = fosm_central_difference(g, [5.0, 10.0], [1.0, 2.0], 0)
        assert result == pytest.approx(6.0, rel=1e-4)

    def test_linear_function_var1(self):
        # dg_1 = g(5, 10+2) - g(5, 10-2) = g(5,12) - g(5,8)
        #       = (15+24) - (15+16) = 39 - 31 = 8
        def g(x, y):
            return 3.0 * x + 2.0 * y

        result = fosm_central_difference(g, [5.0, 10.0], [1.0, 2.0], 1)
        assert result == pytest.approx(8.0, rel=1e-4)

    def test_single_variable(self):
        # g(x) = x^2, means = [3], std_devs = [1]
        # dg = g(4) - g(2) = 16 - 4 = 12
        def g(x):
            return x ** 2

        result = fosm_central_difference(g, [3.0], [1.0], 0)
        assert result == pytest.approx(12.0, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fosm_central_difference(lambda x: x, [], [], 0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fosm_central_difference(lambda x: x, [1.0], [1.0, 2.0], 0)

    def test_index_out_of_range_negative_raises(self):
        with pytest.raises(ValueError, match="var_index must be in range"):
            fosm_central_difference(lambda x: x, [1.0], [1.0], -1)

    def test_index_out_of_range_too_large_raises(self):
        with pytest.raises(ValueError, match="var_index must be in range"):
            fosm_central_difference(lambda x: x, [1.0], [1.0], 1)

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            fosm_central_difference(lambda x: x, [1.0], [-1.0], 0)


# ============================================================================
# 16. fosm_variance_numerical (Equation 7-11)
# ============================================================================


class TestFosmVarianceNumerical:
    """Tests for fosm_variance_numerical (Equation 7-11)."""

    def test_linear_function(self):
        # g(x, y) = 3*x + 2*y, means=[5, 10], std_devs=[1, 2]
        # dg_0 = 6.0 (computed above), dg_1 = 8.0
        # variance = (6/2)^2 + (8/2)^2 = 9 + 16 = 25
        def g(x, y):
            return 3.0 * x + 2.0 * y

        result = fosm_variance_numerical(g, [5.0, 10.0], [1.0, 2.0])
        assert result == pytest.approx(25.0, rel=1e-4)

    def test_single_variable(self):
        # g(x) = x^2, means=[3], std_devs=[1]
        # dg = 12, variance = (12/2)^2 = 36
        def g(x):
            return x ** 2

        result = fosm_variance_numerical(g, [3.0], [1.0])
        assert result == pytest.approx(36.0, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            fosm_variance_numerical(lambda x: x, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fosm_variance_numerical(lambda x: x, [1.0], [1.0, 2.0])


# ============================================================================
# 17. reliability_index_lognormal (Equation 7-12)
# ============================================================================


class TestReliabilityIndexLognormal:
    """Tests for reliability_index_lognormal (Equation 7-12)."""

    def test_basic_valid(self):
        # mu_g = 2.0, sigma_g = 0.5
        # COV_g = 0.5/2.0 = 0.25, COV_g^2 = 0.0625
        # ln_term = ln(1.0625) = 0.060625
        # beta_LN = (ln(2) - 0.5 * 0.060625) / sqrt(0.060625)
        #         = (0.693147 - 0.030312) / 0.246363
        #         = 0.662835 / 0.246363
        #         = 2.6907
        mu_g = 2.0
        sigma_g = 0.5
        cov_g = sigma_g / mu_g
        ln_term = math.log(1.0 + cov_g ** 2)
        expected = (math.log(mu_g) - 0.5 * ln_term) / math.sqrt(ln_term)
        result = reliability_index_lognormal(mu_g, sigma_g)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_small_cov(self):
        # mu_g = 10.0, sigma_g = 0.1 -> COV = 0.01
        mu_g = 10.0
        sigma_g = 0.1
        cov_g = sigma_g / mu_g
        ln_term = math.log(1.0 + cov_g ** 2)
        expected = (math.log(mu_g) - 0.5 * ln_term) / math.sqrt(ln_term)
        result = reliability_index_lognormal(mu_g, sigma_g)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_mu_g_zero_raises(self):
        with pytest.raises(ValueError, match="mu_g must be positive"):
            reliability_index_lognormal(0.0, 1.0)

    def test_mu_g_negative_raises(self):
        with pytest.raises(ValueError, match="mu_g must be positive"):
            reliability_index_lognormal(-1.0, 1.0)

    def test_sigma_g_zero_raises(self):
        with pytest.raises(ValueError, match="sigma_g must be positive"):
            reliability_index_lognormal(2.0, 0.0)

    def test_sigma_g_negative_raises(self):
        with pytest.raises(ValueError, match="sigma_g must be positive"):
            reliability_index_lognormal(2.0, -0.5)


# ============================================================================
# 18. point_estimate_mean (Equation 7-13)
# ============================================================================


class TestPointEstimateMean:
    """Tests for point_estimate_mean (Equation 7-13)."""

    def test_linear_function(self):
        # g(x) = 2*x + 3, means=[5], std_devs=[1]
        # 2 cases: g(5-1)=g(4)=11, g(5+1)=g(6)=15
        # mean = (11 + 15) / 2 = 13.0
        def g(x):
            return 2.0 * x + 3.0

        result = point_estimate_mean(g, [5.0], [1.0])
        assert result == pytest.approx(13.0, rel=1e-4)

    def test_quadratic_function_one_var(self):
        # g(x) = x^2, means=[3], std_devs=[1]
        # cases: g(2)=4, g(4)=16
        # mean = (4 + 16) / 2 = 10.0
        def g(x):
            return x ** 2

        result = point_estimate_mean(g, [3.0], [1.0])
        assert result == pytest.approx(10.0, rel=1e-4)

    def test_two_variables(self):
        # g(x, y) = x + y, means=[2, 3], std_devs=[1, 1]
        # 4 cases (bit pattern: 0=minus, 1=plus):
        #   case 0 (00): g(2-1, 3-1) = g(1,2) = 3
        #   case 1 (01): g(2+1, 3-1) = g(3,2) = 5
        #   case 2 (10): g(2-1, 3+1) = g(1,4) = 5
        #   case 3 (11): g(2+1, 3+1) = g(3,4) = 7
        # mean = (3+5+5+7)/4 = 20/4 = 5.0
        def g(x, y):
            return x + y

        result = point_estimate_mean(g, [2.0, 3.0], [1.0, 1.0])
        assert result == pytest.approx(5.0, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            point_estimate_mean(lambda: 0.0, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            point_estimate_mean(lambda x: x, [1.0], [1.0, 2.0])

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            point_estimate_mean(lambda x: x, [1.0], [-1.0])


# ============================================================================
# 19. point_estimate_variance (Equation 7-14)
# ============================================================================


class TestPointEstimateVariance:
    """Tests for point_estimate_variance (Equation 7-14)."""

    def test_linear_function(self):
        # g(x) = 2*x + 3, means=[5], std_devs=[1]
        # cases: g(4)=11, g(6)=15
        # sum_g = (11+15)/2 = 13, sum_g2 = (121+225)/2 = 173
        # variance = 173 - 13^2 = 173 - 169 = 4.0
        def g(x):
            return 2.0 * x + 3.0

        result = point_estimate_variance(g, [5.0], [1.0])
        assert result == pytest.approx(4.0, rel=1e-4)

    def test_quadratic_function(self):
        # g(x) = x^2, means=[3], std_devs=[1]
        # cases: g(2)=4, g(4)=16
        # sum_g = (4+16)/2 = 10, sum_g2 = (16+256)/2 = 136
        # variance = 136 - 100 = 36.0
        def g(x):
            return x ** 2

        result = point_estimate_variance(g, [3.0], [1.0])
        assert result == pytest.approx(36.0, rel=1e-4)

    def test_two_variables_product(self):
        # g(x, y) = x * y, means=[2, 3], std_devs=[1, 1]
        # 4 cases:
        #   case 0 (00): g(1, 2) = 2
        #   case 1 (01): g(3, 2) = 6
        #   case 2 (10): g(1, 4) = 4
        #   case 3 (11): g(3, 4) = 12
        # w = 0.25
        # sum_g = 0.25*(2+6+4+12) = 0.25*24 = 6
        # sum_g2 = 0.25*(4+36+16+144) = 0.25*200 = 50
        # variance = 50 - 36 = 14.0
        def g(x, y):
            return x * y

        result = point_estimate_variance(g, [2.0, 3.0], [1.0, 1.0])
        assert result == pytest.approx(14.0, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            point_estimate_variance(lambda: 0.0, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            point_estimate_variance(lambda x: x, [1.0], [1.0, 2.0])

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            point_estimate_variance(lambda x: x, [1.0], [-1.0])


# ============================================================================
# 20. rate_of_exceedance (Equation 7-15)
# ============================================================================


class TestRateOfExceedance:
    """Tests for rate_of_exceedance (Equation 7-15)."""

    def test_basic_valid(self):
        # lambda = 1/100 = 0.01
        assert rate_of_exceedance(100.0) == pytest.approx(0.01, rel=1e-4)

    def test_one_year(self):
        assert rate_of_exceedance(1.0) == pytest.approx(1.0, rel=1e-4)

    def test_475_year(self):
        # Common seismic return period
        assert rate_of_exceedance(475.0) == pytest.approx(
            1.0 / 475.0, rel=1e-4
        )

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="return_period must be positive"):
            rate_of_exceedance(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="return_period must be positive"):
            rate_of_exceedance(-50.0)


# ============================================================================
# 21. rate_of_exceedance_from_probability (Equation 7-16)
# ============================================================================


class TestRateOfExceedanceFromProbability:
    """Tests for rate_of_exceedance_from_probability (Equation 7-16)."""

    def test_ten_percent_in_50_years(self):
        # lambda = -ln(1 - 0.10) / 50 = -ln(0.90) / 50
        # = 0.10536 / 50 = 0.0021072
        expected = -math.log(1.0 - 0.10) / 50.0
        result = rate_of_exceedance_from_probability(0.10, 50.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_two_percent_in_50_years(self):
        # lambda = -ln(0.98) / 50
        expected = -math.log(0.98) / 50.0
        result = rate_of_exceedance_from_probability(0.02, 50.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_fifty_percent_in_75_years(self):
        # lambda = -ln(0.50) / 75 = 0.693147 / 75 = 0.009242
        expected = -math.log(0.50) / 75.0
        result = rate_of_exceedance_from_probability(0.50, 75.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_prob_zero_raises(self):
        with pytest.raises(ValueError, match="prob_exceedance must be in the range"):
            rate_of_exceedance_from_probability(0.0, 50.0)

    def test_prob_one_raises(self):
        with pytest.raises(ValueError, match="prob_exceedance must be in the range"):
            rate_of_exceedance_from_probability(1.0, 50.0)

    def test_prob_negative_raises(self):
        with pytest.raises(ValueError, match="prob_exceedance must be in the range"):
            rate_of_exceedance_from_probability(-0.1, 50.0)

    def test_prob_above_one_raises(self):
        with pytest.raises(ValueError, match="prob_exceedance must be in the range"):
            rate_of_exceedance_from_probability(1.5, 50.0)

    def test_exposure_zero_raises(self):
        with pytest.raises(ValueError, match="exposure_period must be positive"):
            rate_of_exceedance_from_probability(0.1, 0.0)

    def test_exposure_negative_raises(self):
        with pytest.raises(ValueError, match="exposure_period must be positive"):
            rate_of_exceedance_from_probability(0.1, -10.0)


# ============================================================================
# 22. probability_of_failure_normal
# ============================================================================


class TestProbabilityOfFailureNormal:
    """Tests for probability_of_failure_normal (Section 7-4.2)."""

    def test_beta_zero(self):
        # Phi(-0) = Phi(0) = 0.5
        assert probability_of_failure_normal(0.0) == pytest.approx(0.5, rel=1e-4)

    def test_beta_positive(self):
        # Phi(-3) ~ 0.001350
        # Using erfc: 0.5 * erfc(3/sqrt(2))
        expected = 0.5 * math.erfc(3.0 / math.sqrt(2.0))
        result = probability_of_failure_normal(3.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_beta_negative(self):
        # Phi(-(-2)) = Phi(2) ~ 0.97725
        expected = 0.5 * math.erfc(-2.0 / math.sqrt(2.0))
        result = probability_of_failure_normal(-2.0)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_large_beta(self):
        # For large beta, P_u should be very small
        result = probability_of_failure_normal(5.0)
        assert result < 1e-5

    def test_known_value_beta_1_645(self):
        # Phi(-1.645) ~ 0.05
        result = probability_of_failure_normal(1.645)
        assert result == pytest.approx(0.05, rel=1e-2)


# ============================================================================
# 23. first_order_second_moment
# ============================================================================


class TestFirstOrderSecondMoment:
    """Tests for first_order_second_moment (composite FOSM analysis)."""

    def test_linear_function(self):
        # g(x, y) = x - y, means=[10, 4], std_devs=[2, 1]
        # mu_g = g(10, 4) = 6.0
        # dg_0 = g(12, 4) - g(8, 4) = 8 - 4 = 4
        # dg_1 = g(10, 5) - g(10, 3) = 5 - 7 = -2
        # (Wait: g(10,5) = 10-5 = 5, g(10,3) = 10-3 = 7, so dg_1 = 5-7 = -2)
        # variance = (4/2)^2 + (-2/2)^2 = 4 + 1 = 5
        # sigma_g = sqrt(5) = 2.23607
        # beta = 6 / sqrt(5) = 2.68328
        # p_u = Phi(-beta)
        def g(x, y):
            return x - y

        mu_g, sigma_g, beta, p_u = first_order_second_moment(
            g, [10.0, 4.0], [2.0, 1.0]
        )
        assert mu_g == pytest.approx(6.0, rel=1e-4)
        assert sigma_g == pytest.approx(math.sqrt(5.0), rel=1e-4)
        assert beta == pytest.approx(6.0 / math.sqrt(5.0), rel=1e-4)
        expected_pu = 0.5 * math.erfc((6.0 / math.sqrt(5.0)) / math.sqrt(2.0))
        assert p_u == pytest.approx(expected_pu, rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="at least one value"):
            first_order_second_moment(lambda: 0.0, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            first_order_second_moment(lambda x: x, [1.0], [1.0, 2.0])


# ============================================================================
# 24. point_estimate_method
# ============================================================================


class TestPointEstimateMethod:
    """Tests for point_estimate_method (composite PEM analysis)."""

    def test_linear_function(self):
        # g(x) = 2*x + 3, means=[5], std_devs=[1]
        # PEM mean = 13 (computed above)
        # PEM variance = 4 (computed above)
        # sigma_g = 2
        # beta = 13/2 = 6.5
        # p_u = Phi(-6.5) ~ very small
        def g(x):
            return 2.0 * x + 3.0

        mu_g, sigma_g, beta, p_u = point_estimate_method(g, [5.0], [1.0])
        assert mu_g == pytest.approx(13.0, rel=1e-4)
        assert sigma_g == pytest.approx(2.0, rel=1e-4)
        assert beta == pytest.approx(6.5, rel=1e-4)
        expected_pu = 0.5 * math.erfc(6.5 / math.sqrt(2.0))
        assert p_u == pytest.approx(expected_pu, rel=1e-4)

    def test_two_variables(self):
        # g(x, y) = x * y, means=[2, 3], std_devs=[1, 1]
        # PEM mean = 6.0, PEM variance = 14.0 (computed above)
        # sigma_g = sqrt(14) = 3.7417
        # beta = 6 / sqrt(14)
        def g(x, y):
            return x * y

        mu_g, sigma_g, beta, p_u = point_estimate_method(
            g, [2.0, 3.0], [1.0, 1.0]
        )
        assert mu_g == pytest.approx(6.0, rel=1e-4)
        assert sigma_g == pytest.approx(math.sqrt(14.0), rel=1e-4)
        assert beta == pytest.approx(6.0 / math.sqrt(14.0), rel=1e-4)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            point_estimate_method(lambda: 0.0, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            point_estimate_method(lambda x: x, [1.0], [1.0, 2.0])

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            point_estimate_method(lambda x: x, [1.0], [-1.0])


# ============================================================================
# 25. monte_carlo_simulation
# ============================================================================


class TestMonteCarloSimulation:
    """Tests for monte_carlo_simulation (Section 7-4.2.4)."""

    def test_normal_distribution_seeded(self):
        # g(x) = x, with mean=10, std_dev=2, should produce
        # mu_g ~ 10, sigma_g ~ 2, p_u ~ Phi(-5) (very small)
        def g(x):
            return x

        mu_g, sigma_g, beta, p_u = monte_carlo_simulation(
            g, [10.0], [2.0], n_trials=50000, seed=42
        )
        assert mu_g == pytest.approx(10.0, rel=0.05)
        assert sigma_g == pytest.approx(2.0, rel=0.1)
        # With mean=10, sigma=2, almost no failures (g<0)
        assert p_u < 0.01

    def test_high_failure_rate(self):
        # g(x) = x - 10, mean=10, std_dev=5
        # mu_g ~ 0, about 50% failures
        def g(x):
            return x - 10.0

        mu_g, sigma_g, beta, p_u = monte_carlo_simulation(
            g, [10.0], [5.0], n_trials=50000, seed=123
        )
        assert mu_g == pytest.approx(0.0, abs=0.5)
        assert p_u == pytest.approx(0.5, abs=0.05)

    def test_lognormal_distribution(self):
        # g(x) = x, mean=5, std_dev=1, lognormal
        def g(x):
            return x

        mu_g, sigma_g, beta, p_u = monte_carlo_simulation(
            g, [5.0], [1.0], n_trials=50000, seed=99, distribution="lognormal"
        )
        # Should recover approximately the specified mean
        assert mu_g == pytest.approx(5.0, rel=0.05)
        # All values positive => p_u = 0 (g is always > 0 for lognormal)
        assert p_u == pytest.approx(0.0, abs=0.01)

    def test_reproducibility_with_seed(self):
        def g(x):
            return x

        result1 = monte_carlo_simulation(g, [5.0], [1.0], n_trials=1000, seed=7)
        result2 = monte_carlo_simulation(g, [5.0], [1.0], n_trials=1000, seed=7)
        assert result1[0] == pytest.approx(result2[0], rel=1e-10)
        assert result1[1] == pytest.approx(result2[1], rel=1e-10)

    def test_empty_means_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            monte_carlo_simulation(lambda: 0.0, [], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            monte_carlo_simulation(lambda x: x, [1.0], [1.0, 2.0])

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            monte_carlo_simulation(lambda x: x, [1.0], [-1.0])

    def test_n_trials_zero_raises(self):
        with pytest.raises(ValueError, match="n_trials must be at least 1"):
            monte_carlo_simulation(lambda x: x, [1.0], [1.0], n_trials=0)

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="distribution must be"):
            monte_carlo_simulation(
                lambda x: x, [1.0], [1.0], distribution="uniform"
            )

    def test_lognormal_negative_mean_raises(self):
        with pytest.raises(ValueError, match="positive for lognormal"):
            monte_carlo_simulation(
                lambda x: x, [-1.0], [1.0], distribution="lognormal"
            )

    def test_two_variables_normal(self):
        # g(x, y) = x - y, means=[10, 4], std_devs=[2, 1]
        # Expected: mu_g ~ 6, sigma_g ~ sqrt(5) ~ 2.236
        def g(x, y):
            return x - y

        mu_g, sigma_g, beta, p_u = monte_carlo_simulation(
            g, [10.0, 4.0], [2.0, 1.0], n_trials=50000, seed=55
        )
        assert mu_g == pytest.approx(6.0, rel=0.05)
        assert sigma_g == pytest.approx(math.sqrt(5.0), rel=0.1)
