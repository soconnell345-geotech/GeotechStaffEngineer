"""FOSM engine tests — DM7-2 ch7 expected values + exact linear anchors."""

import math

import pytest

from reliability import fosm


class TestDM7Values:
    """Expected values resurrected from the retired DM7Eqs ch7 suite."""

    def test_linear_margin(self):
        # g = x - y, mu=[10,4], sigma=[2,1] -> mu_g=6, sigma_g=sqrt(5),
        # beta = 6/sqrt(5) = 2.683 (UFC Eqs. 7-8/7-10/7-11/7-7)
        res = fosm(lambda v: v["x"] - v["y"],
                   {"x": {"mean": 10.0, "std": 2.0},
                    "y": {"mean": 4.0, "std": 1.0}},
                   convention="margin")
        assert res.g_mean == pytest.approx(6.0)
        assert res.g_std == pytest.approx(math.sqrt(5.0))
        assert res.beta_normal == pytest.approx(6.0 / math.sqrt(5.0))
        expected_pu = 0.5 * math.erfc((6.0 / math.sqrt(5.0)) / math.sqrt(2.0))
        assert res.pf_normal == pytest.approx(expected_pu, rel=1e-6)
        assert res.beta_lognormal is None  # margin convention

    def test_linear_two_vars_variance(self):
        # g = 3x + 2y, mu=[5,10], sigma=[1,2] -> var = 3^2*1 + 2^2*4 = 25
        res = fosm(lambda v: 3.0 * v["x"] + 2.0 * v["y"],
                   {"x": {"mean": 5.0, "std": 1.0},
                    "y": {"mean": 10.0, "std": 2.0}},
                   convention="margin")
        assert res.g_std ** 2 == pytest.approx(25.0)
        assert res.variable_deltas["x"] == pytest.approx(6.0)
        assert res.variable_deltas["y"] == pytest.approx(8.0)

    def test_quadratic_single_var(self):
        # g = x^2 at mu=3, sigma=1: Delta g = 16-4 = 12, var = 36
        res = fosm(lambda v: v["x"] ** 2,
                   {"x": {"mean": 3.0, "std": 1.0}}, convention="margin")
        assert res.g_std ** 2 == pytest.approx(36.0)


class TestCorrelated:
    def test_correlated_sum_exact(self):
        # Var[x+y] = s1^2 + s2^2 + 2 rho s1 s2 (exact for linear g)
        res = fosm(lambda v: v["x"] + v["y"],
                   {"x": {"mean": 0.0, "std": 2.0},
                    "y": {"mean": 10.0, "std": 3.0}},
                   correlation={("x", "y"): 0.5},
                   convention="margin")
        assert res.g_std ** 2 == pytest.approx(4 + 9 + 2 * 0.5 * 2 * 3)
        assert res.correlated

    def test_negative_correlation_reduces_variance(self):
        spec = {"x": {"mean": 0.0, "std": 2.0}, "y": {"mean": 10.0, "std": 3.0}}
        v_neg = fosm(lambda v: v["x"] + v["y"], spec,
                     correlation={("x", "y"): -0.5},
                     convention="margin").g_std ** 2
        assert v_neg == pytest.approx(4 + 9 - 6)


class TestVarianceContributions:
    def test_sum_to_100(self):
        res = fosm(lambda v: v["a"] * v["b"] + v["c"],
                   {"a": {"mean": 2.0, "cov": 0.1},
                    "b": {"mean": 3.0, "cov": 0.2},
                    "c": {"mean": 1.0, "std": 0.5}},
                   convention="margin")
        assert sum(res.variance_contributions_pct.values()) == \
            pytest.approx(100.0)

    def test_dominant_variable_identified(self):
        # y has 4x the sensitivity-weighted sigma of x -> 16x the variance
        res = fosm(lambda v: v["x"] + v["y"],
                   {"x": {"mean": 1.0, "std": 1.0},
                    "y": {"mean": 1.0, "std": 4.0}},
                   convention="margin")
        assert res.variance_contributions_pct["y"] == pytest.approx(
            100.0 * 16.0 / 17.0)


class TestFOSConvention:
    def test_duncan_anchor(self):
        # F ~ (1.5, COV 0.17): beta_LN = 2.32, pf ~ 1% (Duncan 2000)
        res = fosm(lambda v: v["F"],
                   {"F": {"mean": 1.5, "cov": 0.17, "dist": "lognormal"}})
        assert res.g_mean == pytest.approx(1.5)
        assert res.g_cov == pytest.approx(0.17)
        assert res.beta_normal == pytest.approx(0.5 / 0.255, rel=1e-6)
        assert res.beta_lognormal == pytest.approx(2.32, abs=0.005)
        assert res.pf_lognormal == pytest.approx(0.0102, abs=0.0005)

    def test_summary_and_to_dict(self):
        res = fosm(lambda v: v["F"], {"F": {"mean": 2.0, "cov": 0.2}})
        d = res.to_dict()
        assert d["engine"] == "fosm"
        assert d["convention"] == "fos"
        assert "beta_lognormal" in d
        assert "Variance contributions" in res.summary()

    def test_n_g_calls(self):
        calls = []
        res = fosm(lambda v: calls.append(1) or 1.5,
                   {"a": {"mean": 1.0, "std": 0.1},
                    "b": {"mean": 1.0, "std": 0.1}})
        assert res.n_g_calls == 5  # 2n + 1
        assert len(calls) == 5


class TestErrors:
    def test_bad_convention(self):
        with pytest.raises(ValueError, match="convention"):
            fosm(lambda v: 1.0, {"a": {"mean": 1.0, "std": 0.1}},
                 convention="weird")
