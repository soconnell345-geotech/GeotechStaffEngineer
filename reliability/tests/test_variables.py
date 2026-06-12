"""RandomVariable model + correlation assembly tests."""

import math

import numpy as np
import pytest

from reliability.variables import (
    RandomVariable, build_correlation, variables_from_spec,
)


class TestConstruction:
    def test_normal_from_cov(self):
        rv = RandomVariable("phi", mean=30.0, cov=0.10)
        assert rv.std == pytest.approx(3.0)
        assert rv.dist == "normal"

    def test_normal_from_std(self):
        rv = RandomVariable("phi", mean=30.0, std=3.0)
        assert rv.cov == pytest.approx(0.10)

    def test_conflicting_std_cov_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            RandomVariable("x", mean=10.0, std=1.0, cov=0.5)

    def test_missing_dispersion_raises(self):
        with pytest.raises(ValueError, match="positive std"):
            RandomVariable("x", mean=10.0)

    def test_missing_mean_raises(self):
        with pytest.raises(ValueError, match="mean is required"):
            RandomVariable("x", std=1.0)

    def test_bad_dist_raises(self):
        with pytest.raises(ValueError, match="dist must be one of"):
            RandomVariable("x", mean=1.0, std=0.1, dist="gumbel")

    def test_lognormal_underlying_params(self):
        rv = RandomVariable("su", mean=50.0, cov=0.30, dist="lognormal")
        v2 = 0.09
        assert rv.sigma_ln == pytest.approx(math.sqrt(math.log(1 + v2)))
        assert rv.mu_ln == pytest.approx(
            math.log(50.0) - 0.5 * math.log(1 + v2))
        # frozen dist must reproduce the arithmetic moments
        assert rv._base.mean() == pytest.approx(50.0)
        assert rv._base.std() == pytest.approx(15.0)

    def test_lognormal_needs_positive_mean(self):
        with pytest.raises(ValueError, match="mean > 0"):
            RandomVariable("x", mean=-5.0, std=1.0, dist="lognormal")

    def test_uniform_from_bounds(self):
        rv = RandomVariable("k", lower=2.0, upper=8.0, dist="uniform")
        assert rv.mean == pytest.approx(5.0)
        assert rv.std == pytest.approx(6.0 / math.sqrt(12.0))

    def test_uniform_from_moments(self):
        rv = RandomVariable("k", mean=5.0, std=1.0, dist="uniform")
        assert rv.lower == pytest.approx(5.0 - math.sqrt(3.0))
        assert rv.upper == pytest.approx(5.0 + math.sqrt(3.0))
        assert rv._base.std() == pytest.approx(1.0)

    def test_triangular_from_bounds(self):
        rv = RandomVariable("gamma", lower=17.0, upper=21.0, mode=19.0,
                            dist="triangular")
        assert rv.mean == pytest.approx(19.0)
        assert rv._base.mean() == pytest.approx(19.0)
        assert rv._base.std() == pytest.approx(rv.std)

    def test_triangular_from_moments_symmetric(self):
        rv = RandomVariable("gamma", mean=19.0, std=0.8, dist="triangular")
        assert rv.mode == pytest.approx(19.0)
        assert rv._base.std() == pytest.approx(0.8)

    def test_triangular_mode_outside_raises(self):
        with pytest.raises(ValueError, match="mode"):
            RandomVariable("x", lower=0.0, upper=1.0, mode=2.0,
                           dist="triangular")


class TestDistributionInterface:
    def test_normal_ppf_cdf_roundtrip(self):
        rv = RandomVariable("x", mean=10.0, std=2.0)
        for q in (0.05, 0.5, 0.95):
            assert rv.cdf(rv.ppf(q)) == pytest.approx(q)

    def test_lognormal_median(self):
        rv = RandomVariable("x", mean=10.0, cov=0.5, dist="lognormal")
        assert float(rv.ppf(0.5)) == pytest.approx(math.exp(rv.mu_ln))

    def test_sampling_moments(self):
        rv = RandomVariable("x", mean=20.0, cov=0.25, dist="lognormal")
        rng = np.random.default_rng(7)
        s = rv.sample(rng, 200_000)
        assert s.mean() == pytest.approx(20.0, rel=0.01)
        assert s.std(ddof=1) == pytest.approx(5.0, rel=0.02)
        assert (s > 0).all()

    def test_truncated_normal_respects_bounds(self):
        rv = RandomVariable("phi", mean=30.0, std=8.0, lower=10.0, upper=45.0)
        rng = np.random.default_rng(3)
        s = rv.sample(rng, 50_000)
        assert s.min() >= 10.0
        assert s.max() <= 45.0
        assert float(rv.cdf(10.0)) == pytest.approx(0.0, abs=1e-12)
        assert float(rv.cdf(45.0)) == pytest.approx(1.0, abs=1e-12)

    def test_truncation_zero_mass_raises(self):
        with pytest.raises(ValueError, match="probability"):
            RandomVariable("x", mean=0.0, std=1.0, lower=50.0, upper=60.0)


class TestEquivalentNormal:
    def test_normal_is_identity(self):
        rv = RandomVariable("x", mean=10.0, std=2.0)
        mu_eq, sig_eq = rv.equivalent_normal(13.0)
        assert mu_eq == pytest.approx(10.0)
        assert sig_eq == pytest.approx(2.0)

    def test_lognormal_closed_form(self):
        # Rackwitz-Fiessler closed form for lognormal:
        # sigma_eq = zeta*x ; mu_eq = x*(1 - ln x + lambda)
        rv = RandomVariable("x", mean=10.0, cov=0.3, dist="lognormal")
        x = 8.0
        mu_eq, sig_eq = rv.equivalent_normal(x)
        assert sig_eq == pytest.approx(rv.sigma_ln * x, rel=1e-6)
        assert mu_eq == pytest.approx(
            x * (1.0 - math.log(x) + rv.mu_ln), rel=1e-6)

    def test_matches_cdf_and_pdf(self):
        # the equivalent normal must match F and f at x (definition)
        from scipy import stats as sps
        rv = RandomVariable("x", lower=2.0, upper=8.0, dist="uniform")
        x = 4.0
        mu_eq, sig_eq = rv.equivalent_normal(x)
        assert sps.norm(mu_eq, sig_eq).cdf(x) == pytest.approx(
            float(rv.cdf(x)), rel=1e-9)
        assert sps.norm(mu_eq, sig_eq).pdf(x) == pytest.approx(
            float(rv.pdf(x)), rel=1e-9)


class TestSpecParsing:
    def test_dict_spec(self):
        vs = variables_from_spec({
            "phi": {"mean": 33.0, "cov": 0.08},
            "c": {"mean": 5.0, "std": 2.0, "dist": "lognormal"},
        })
        assert [v.name for v in vs] == ["phi", "c"]
        assert vs[1].dist == "lognormal"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown spec key"):
            variables_from_spec({"phi": {"mean": 33.0, "stdev": 2.0}})

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            variables_from_spec({})

    def test_duplicate_names_raise(self):
        a = RandomVariable("x", mean=1.0, std=0.1)
        b = RandomVariable("x", mean=2.0, std=0.1)
        with pytest.raises(ValueError, match="Duplicate"):
            variables_from_spec([a, b])


class TestCorrelation:
    def _vars(self):
        return [RandomVariable("a", mean=1.0, std=0.1),
                RandomVariable("b", mean=2.0, std=0.2),
                RandomVariable("c", mean=3.0, std=0.3)]

    def test_none_gives_identity(self):
        R = build_correlation(self._vars(), None)
        assert np.allclose(R, np.eye(3))

    def test_pairwise_dict(self):
        R = build_correlation(self._vars(), {("a", "b"): -0.5})
        assert R[0, 1] == pytest.approx(-0.5)
        assert R[1, 0] == pytest.approx(-0.5)
        assert R[2, 2] == 1.0

    def test_string_key(self):
        R = build_correlation(self._vars(), {"a,c": 0.3})
        assert R[0, 2] == pytest.approx(0.3)

    def test_full_matrix(self):
        M = [[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]]
        R = build_correlation(self._vars(), M)
        assert R[0, 1] == pytest.approx(0.2)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown variable"):
            build_correlation(self._vars(), {("a", "zz"): 0.5})

    def test_not_pd_raises(self):
        bad = {("a", "b"): 0.95, ("b", "c"): 0.95, ("a", "c"): -0.95}
        with pytest.raises(ValueError, match="positive definite"):
            build_correlation(self._vars(), bad)

    def test_asymmetric_matrix_raises(self):
        M = [[1.0, 0.2, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValueError, match="symmetric"):
            build_correlation(self._vars(), M)

    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError, match="in \\(-1, 1\\)"):
            build_correlation(self._vars(), {("a", "b"): 1.0})
