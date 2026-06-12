"""Rosenblueth PEM tests — DM7-2 ch7 expected values + correlated weights."""

import math

import pytest

from reliability import pem


class TestDM7Values:
    def test_linear_single_var(self):
        # g = 2x+3, mu=5, sigma=1 -> mean 13, var 4 (UFC Eqs. 7-13/7-14)
        res = pem(lambda v: 2.0 * v["x"] + 3.0,
                  {"x": {"mean": 5.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(13.0)
        assert res.g_std ** 2 == pytest.approx(4.0)
        assert res.beta_normal == pytest.approx(6.5)

    def test_quadratic(self):
        # g = x^2, mu=3, sigma=1 -> mean 10, var 36
        res = pem(lambda v: v["x"] ** 2,
                  {"x": {"mean": 3.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(10.0)
        assert res.g_std ** 2 == pytest.approx(36.0)

    def test_product_two_vars(self):
        # g = x*y, mu=[2,3], sigma=[1,1] -> mean 6, var 14
        res = pem(lambda v: v["x"] * v["y"],
                  {"x": {"mean": 2.0, "std": 1.0},
                   "y": {"mean": 3.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(6.0)
        assert res.g_std ** 2 == pytest.approx(14.0)
        assert res.beta_normal == pytest.approx(6.0 / math.sqrt(14.0))
        assert res.n_points == 4
        assert res.scheme == "full_2n"

    def test_sum_two_vars_mean(self):
        # g = x+y, mu=[2,3] -> mean 5 (DM7 suite)
        res = pem(lambda v: v["x"] + v["y"],
                  {"x": {"mean": 2.0, "std": 1.0},
                   "y": {"mean": 3.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(5.0)


class TestCorrelatedWeights:
    def test_correlated_sum_exact(self):
        # Rosenblueth correlated weights reproduce
        # Var[x+y] = s1^2+s2^2+2 rho s1 s2 exactly for linear g
        res = pem(lambda v: v["x"] + v["y"],
                  {"x": {"mean": 0.0, "std": 1.0},
                   "y": {"mean": 0.0, "std": 1.0}},
                  correlation={("x", "y"): 0.5},
                  convention="margin")
        assert res.g_std ** 2 == pytest.approx(2.0 + 2 * 0.5)
        assert res.correlated


class TestMultiplicativeScheme:
    def test_matches_full_for_product(self):
        # For g = x*y the 2n+1 multiplicative scheme is exact:
        # mean 6, var 14 — same as the full 2^n scheme
        spec = {"x": {"mean": 2.0, "std": 1.0}, "y": {"mean": 3.0, "std": 1.0}}
        res = pem(lambda v: v["x"] * v["y"], spec, convention="margin",
                  scheme="multiplicative")
        assert res.g_mean == pytest.approx(6.0)
        assert res.g_std ** 2 == pytest.approx(14.0)
        assert res.n_points == 5  # 2n + 1
        assert res.scheme == "multiplicative_2n_plus_1"

    def test_rejects_correlation(self):
        with pytest.raises(ValueError, match="uncorrelated"):
            pem(lambda v: v["x"] * v["y"],
                {"x": {"mean": 2.0, "std": 1.0},
                 "y": {"mean": 3.0, "std": 1.0}},
                correlation={("x", "y"): 0.5}, scheme="multiplicative")

    def test_zero_mean_value_raises(self):
        with pytest.raises(ValueError, match="g\\(means\\)"):
            pem(lambda v: v["x"], {"x": {"mean": 0.0, "std": 1.0}},
                convention="margin", scheme="multiplicative")


class TestFOSConvention:
    def test_duncan_anchor(self):
        res = pem(lambda v: v["F"],
                  {"F": {"mean": 1.5, "cov": 0.17, "dist": "lognormal"}})
        # PEM at mu +/- sigma: mean 1.5, std 0.255 exactly for identity g
        assert res.g_mean == pytest.approx(1.5)
        assert res.g_cov == pytest.approx(0.17)
        assert res.beta_lognormal == pytest.approx(2.32, abs=0.005)

    def test_to_dict_summary(self):
        res = pem(lambda v: v["F"], {"F": {"mean": 2.0, "cov": 0.3}})
        assert res.to_dict()["engine"] == "pem"
        assert "ROSENBLUETH" in res.summary()


class TestErrors:
    def test_too_many_vars_full(self):
        spec = {f"x{i}": {"mean": 1.0, "std": 0.1} for i in range(21)}
        with pytest.raises(ValueError, match="2\\^21"):
            pem(lambda v: sum(v.values()), spec, convention="margin")

    def test_bad_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            pem(lambda v: 1.0, {"a": {"mean": 1.0, "std": 0.1}},
                scheme="reduced")
