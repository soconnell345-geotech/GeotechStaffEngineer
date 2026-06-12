"""Stats / index utilities — resurrected DM7-validated expected values.

Expected values carried over from the retired DM7Eqs chapter-7 suite
(git show 450320b~1:DM7Eqs/tests/test_dm7_2_chapter7.py), which encoded
UFC 3-220-20 ch. 7 worked numbers.
"""

import math

import pytest

from reliability.stats import (
    beta_from_pf, beta_lognormal, beta_normal, combined_cov, cov_from_params,
    pf_from_beta, rate_of_exceedance, rate_of_exceedance_from_probability,
    sample_cov, sample_mean, sample_std, sample_variance, std_from_range,
)


class TestSampleStats:
    def test_mean(self):
        assert sample_mean([2, 4, 6, 8]) == pytest.approx(5.0)

    def test_mean_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            sample_mean([])

    def test_variance_unbiased(self):
        # DM7 suite: var([2,4,6,8]) = 20/3
        assert sample_variance([2, 4, 6, 8]) == pytest.approx(20.0 / 3.0)

    def test_variance_two_values(self):
        assert sample_variance([3, 7]) == pytest.approx(8.0)

    def test_variance_needs_two(self):
        with pytest.raises(ValueError, match="at least two"):
            sample_variance([10.0])

    def test_std(self):
        assert sample_std([2, 4, 6, 8]) == pytest.approx(
            math.sqrt(20.0 / 3.0))

    def test_cov_from_data(self):
        expected = math.sqrt(20.0 / 3.0) / 5.0
        assert sample_cov([2, 4, 6, 8]) == pytest.approx(expected)

    def test_cov_zero_mean_raises(self):
        with pytest.raises(ValueError, match="zero"):
            sample_cov([-1, 1])

    def test_cov_from_params(self):
        assert cov_from_params(2.5, 10.0) == pytest.approx(0.25)

    def test_cov_from_params_errors(self):
        with pytest.raises(ValueError):
            cov_from_params(-1.0, 10.0)
        with pytest.raises(ValueError):
            cov_from_params(2.0, 0.0)


class TestRangeRule:
    def test_six_sigma_default(self):
        # UFC Eq. 7-6: sigma = (100-40)/6 = 10
        assert std_from_range(100.0, 40.0) == pytest.approx(10.0)

    def test_custom_n(self):
        assert std_from_range(100.0, 40.0, n_sigma=4.0) == pytest.approx(15.0)

    def test_errors(self):
        with pytest.raises(ValueError, match="greater than"):
            std_from_range(40.0, 100.0)
        with pytest.raises(ValueError, match="positive"):
            std_from_range(100.0, 40.0, n_sigma=0.0)


class TestCombinedCov:
    def test_three_components(self):
        # DM7 suite: sqrt(0.1^2+0.2^2+0.3^2) = sqrt(0.14)
        assert combined_cov(0.1, 0.2, 0.3) == pytest.approx(math.sqrt(0.14))

    def test_single_component(self):
        assert combined_cov(0.0, 0.0, 0.5) == pytest.approx(0.5)

    def test_with_variance_reduction(self):
        # Gamma^2 = 0.25 applied to inherent component only:
        # sqrt(0.25*0.4^2 + 0.1^2) = sqrt(0.04+0.01)
        assert combined_cov(0.4, 0.1, variance_reduction=0.25) == \
            pytest.approx(math.sqrt(0.05))

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            combined_cov(-0.1, 0.2, 0.3)

    def test_bad_reduction_raises(self):
        with pytest.raises(ValueError, match="variance_reduction"):
            combined_cov(0.1, variance_reduction=1.5)


class TestReliabilityIndices:
    def test_beta_normal_margin(self):
        # DM7 suite: beta = 10/2.5 = 4
        assert beta_normal(10.0, 2.5) == pytest.approx(4.0)

    def test_beta_normal_fos_threshold(self):
        assert beta_normal(1.5, 0.25, threshold=1.0) == pytest.approx(2.0)

    def test_beta_normal_sigma_positive(self):
        with pytest.raises(ValueError):
            beta_normal(10.0, 0.0)

    def test_beta_lognormal_matches_ufc_formula(self):
        # DM7 suite expected: mu=2, sigma=0.5 -> formula value
        mu, sigma = 2.0, 0.5
        cov = sigma / mu
        ln_term = math.log(1.0 + cov ** 2)
        expected = (math.log(mu) - 0.5 * ln_term) / math.sqrt(ln_term)
        assert beta_lognormal(mu, cov) == pytest.approx(expected)

    def test_duncan_2000_anchor(self):
        # Duncan (2000): F=1.5, COV=0.17 -> beta_LN = 2.32, pf ~ 1%
        b = beta_lognormal(1.5, 0.17)
        assert b == pytest.approx(2.32, abs=0.005)
        assert pf_from_beta(b) == pytest.approx(0.0102, abs=0.0005)

    def test_beta_lognormal_errors(self):
        with pytest.raises(ValueError):
            beta_lognormal(0.0, 0.1)
        with pytest.raises(ValueError):
            beta_lognormal(1.5, 0.0)


class TestPfBeta:
    def test_pf_beta_zero(self):
        assert pf_from_beta(0.0) == pytest.approx(0.5)

    def test_pf_beta_3(self):
        assert pf_from_beta(3.0) == pytest.approx(
            0.5 * math.erfc(3.0 / math.sqrt(2.0)), rel=1e-10)

    def test_pf_known_value(self):
        # PHI(-1.645) ~ 0.05
        assert pf_from_beta(1.645) == pytest.approx(0.05, rel=1e-2)

    def test_roundtrip(self):
        assert beta_from_pf(pf_from_beta(2.5)) == pytest.approx(2.5)

    def test_beta_from_pf_range(self):
        with pytest.raises(ValueError):
            beta_from_pf(0.0)


class TestHazardRates:
    def test_return_period(self):
        assert rate_of_exceedance(475.0) == pytest.approx(1.0 / 475.0)
        with pytest.raises(ValueError):
            rate_of_exceedance(0.0)

    def test_from_probability(self):
        # DM7 suite: 10% in 50 yr -> 0.0021072 (return period ~475 yr)
        lam = rate_of_exceedance_from_probability(0.10, 50.0)
        assert lam == pytest.approx(-math.log(0.90) / 50.0)
        assert 1.0 / lam == pytest.approx(474.6, abs=0.1)

    def test_from_probability_errors(self):
        with pytest.raises(ValueError):
            rate_of_exceedance_from_probability(0.0, 50.0)
        with pytest.raises(ValueError):
            rate_of_exceedance_from_probability(0.1, 0.0)
