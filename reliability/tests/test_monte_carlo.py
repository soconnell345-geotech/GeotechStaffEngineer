"""Monte Carlo engine tests — exact linear anchors, LHS, correlation, CI."""

import math

import numpy as np
import pytest

from reliability import monte_carlo


def margin_RS(v):
    return v["R"] - v["S"]


RS_SPEC = {"R": {"mean": 15.0, "std": 2.0}, "S": {"mean": 10.0, "std": 1.5}}
# exact: beta = 5/sqrt(4+2.25) = 2.0, pf = PHI(-2) = 0.02275


class TestExactLinearAnchor:
    def test_normal_margin_pf(self):
        res = monte_carlo(margin_RS, RS_SPEC, n=200_000, seed=42,
                          convention="margin")
        assert res.g_mean == pytest.approx(5.0, rel=0.01)
        assert res.g_std == pytest.approx(2.5, rel=0.01)
        assert res.pf == pytest.approx(0.02275, rel=0.05)
        assert res.beta_normal == pytest.approx(2.0, rel=0.02)
        # CI must bracket the exact value
        assert res.pf_ci95[0] < 0.02275 < res.pf_ci95[1]

    def test_lhs_matches(self):
        res = monte_carlo(margin_RS, RS_SPEC, n=50_000, seed=1,
                          sampling="lhs", convention="margin")
        assert res.pf == pytest.approx(0.02275, rel=0.08)
        assert res.sampling == "lhs"
        # LHS stratification: sample moments essentially exact
        assert res.g_mean == pytest.approx(5.0, rel=0.002)


class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = monte_carlo(margin_RS, RS_SPEC, n=2000, seed=7,
                         convention="margin")
        r2 = monte_carlo(margin_RS, RS_SPEC, n=2000, seed=7,
                         convention="margin")
        assert r1.pf == r2.pf
        assert r1.g_mean == r2.g_mean

    def test_different_seed_differs(self):
        r1 = monte_carlo(margin_RS, RS_SPEC, n=2000, seed=7,
                         convention="margin")
        r2 = monte_carlo(margin_RS, RS_SPEC, n=2000, seed=8,
                         convention="margin")
        assert r1.g_mean != r2.g_mean


class TestCorrelation:
    def test_positive_rho_reduces_margin_variance(self):
        # Var[R-S] = sR^2 + sS^2 - 2 rho sR sS
        res = monte_carlo(margin_RS, RS_SPEC, n=100_000, seed=3,
                          correlation={("R", "S"): 0.8},
                          convention="margin")
        exact_var = 4.0 + 2.25 - 2 * 0.8 * 2.0 * 1.5
        assert res.g_std ** 2 == pytest.approx(exact_var, rel=0.02)
        assert res.correlated

    def test_lognormal_marginals_with_correlation_keep_moments(self):
        spec = {"R": {"mean": 20.0, "cov": 0.2, "dist": "lognormal"},
                "S": {"mean": 10.0, "cov": 0.25, "dist": "lognormal"}}
        res = monte_carlo(lambda v: v["R"], spec, n=100_000, seed=5,
                          correlation={("R", "S"): 0.5},
                          convention="margin")
        # marginal transform preserves the marginal moments
        assert res.g_mean == pytest.approx(20.0, rel=0.01)
        assert res.g_std == pytest.approx(4.0, rel=0.02)


class TestFOSConvention:
    def test_duncan_anchor_lognormal_fos(self):
        # F lognormal (1.5, COV 0.17): exact pf = PHI(-2.32) = 1.02%
        res = monte_carlo(lambda v: v["F"],
                          {"F": {"mean": 1.5, "cov": 0.17,
                                 "dist": "lognormal"}},
                          n=200_000, seed=11)
        assert res.pf == pytest.approx(0.0102, rel=0.05)
        assert res.beta_lognormal == pytest.approx(2.32, abs=0.02)
        assert res.pf_lognormal == pytest.approx(0.0102, rel=0.05)

    def test_truncated_variable_respected(self):
        res = monte_carlo(lambda v: v["phi"],
                          {"phi": {"mean": 30.0, "std": 10.0,
                                   "lower": 20.0, "upper": 45.0}},
                          n=20_000, seed=2, convention="margin")
        assert res.g_min >= 20.0
        assert res.g_max <= 45.0


class TestDiagnostics:
    def test_convergence_trace(self):
        res = monte_carlo(margin_RS, RS_SPEC, n=10_000, seed=4,
                          convention="margin")
        assert len(res.convergence) == 10
        ns = [c[0] for c in res.convergence]
        assert ns == sorted(ns)
        assert ns[-1] == res.n
        # final trace point equals reported pf
        assert res.convergence[-1][1] == pytest.approx(res.pf)

    def test_histogram_and_percentiles(self):
        res = monte_carlo(margin_RS, RS_SPEC, n=5000, seed=9, n_bins=20,
                          convention="margin")
        assert len(res.histogram_counts) == 20
        assert len(res.histogram_bins) == 21
        assert sum(res.histogram_counts) == res.n
        assert res.percentiles["p50"] == pytest.approx(res.g_median)

    def test_keep_samples(self):
        res = monte_carlo(margin_RS, RS_SPEC, n=500, seed=10,
                          keep_samples=True, convention="margin")
        assert len(res.samples) == 500

    def test_nonfinite_g_dropped(self):
        def g(v):
            return float("nan") if v["x"] > 10.0 else v["x"]
        res = monte_carlo(g, {"x": {"mean": 10.0, "std": 1.0}},
                          n=4000, seed=12, convention="margin")
        assert res.n < 4000
        assert res.g_max <= 10.0

    def test_to_dict_json_safe(self):
        import json
        res = monte_carlo(margin_RS, RS_SPEC, n=1000, seed=1,
                          convention="margin")
        json.dumps(res.to_dict())


class TestErrors:
    def test_bad_sampling(self):
        with pytest.raises(ValueError, match="sampling"):
            monte_carlo(margin_RS, RS_SPEC, n=100, sampling="sobol")

    def test_n_too_small(self):
        with pytest.raises(ValueError, match="at least 2"):
            monte_carlo(margin_RS, RS_SPEC, n=1)
