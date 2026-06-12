"""Native FORM tests — exact anchors, R-F transform, pystra cross-check,
and four-engine agreement on a shared problem."""

import math

import pytest

from reliability import form, fosm, monte_carlo, pem


def exact_lognormal_beta(mu_r, cov_r, mu_s, cov_s):
    """Exact beta for g = R - S with R, S lognormal (failure iff R/S < 1).

    beta = ln[(mu_R/mu_S) sqrt((1+V_S^2)/(1+V_R^2))]
           / sqrt(ln[(1+V_R^2)(1+V_S^2)])

    (USACE ETL 1110-2-547 App. B; Ang & Tang.)
    """
    num = math.log((mu_r / mu_s)
                   * math.sqrt((1 + cov_s ** 2) / (1 + cov_r ** 2)))
    den = math.sqrt(math.log((1 + cov_r ** 2) * (1 + cov_s ** 2)))
    return num / den


class TestExactNormalAnchor:
    SPEC = {"R": {"mean": 15.0, "std": 2.0}, "S": {"mean": 10.0, "std": 1.5}}

    def test_linear_normal_beta_exact(self):
        # exact beta = 5/2.5 = 2.0, FORM is exact for linear normal g
        res = form(lambda v: v["R"] - v["S"], self.SPEC, convention="margin")
        assert res.converged
        assert res.beta == pytest.approx(2.0, abs=1e-4)
        assert res.pf == pytest.approx(0.02275, rel=1e-3)

    def test_alphas_and_design_point(self):
        res = form(lambda v: v["R"] - v["S"], self.SPEC, convention="margin")
        # alpha_R = -sigma_R/sigma_g = -0.8 (resistance), alpha_S = +0.6
        assert res.alphas["R"] == pytest.approx(-0.8, abs=1e-3)
        assert res.alphas["S"] == pytest.approx(0.6, abs=1e-3)
        # design point sits on the limit state: R* = S*
        assert res.design_point["R"] == pytest.approx(
            res.design_point["S"], rel=1e-4)
        assert abs(res.g_at_design_point) < 1e-4

    def test_correlated_linear_exact(self):
        # beta = (muR-muS)/sqrt(sR^2+sS^2-2 rho sR sS)
        rho = 0.5
        res = form(lambda v: v["R"] - v["S"], self.SPEC,
                   correlation={("R", "S"): rho}, convention="margin")
        exact = 5.0 / math.sqrt(4.0 + 2.25 - 2 * rho * 2.0 * 1.5)
        assert res.beta == pytest.approx(exact, abs=1e-3)


class TestLognormalAnchor:
    def test_lognormal_margin_matches_closed_form(self):
        # failure surface {R=S} identical for R-S and ln R - ln S, so FORM
        # must reproduce the exact lognormal beta
        spec = {"R": {"mean": 20.0, "cov": 0.2, "dist": "lognormal"},
                "S": {"mean": 10.0, "cov": 0.25, "dist": "lognormal"}}
        res = form(lambda v: v["R"] - v["S"], spec, convention="margin")
        exact = exact_lognormal_beta(20.0, 0.2, 10.0, 0.25)
        assert res.converged
        assert res.beta == pytest.approx(exact, abs=1e-3)

    def test_duncan_anchor_single_lognormal_fos(self):
        # F lognormal (1.5, 0.17), failure F < 1: exact beta is the Duncan
        # lognormal index 2.32 — FORM transform is exact here
        res = form(lambda v: v["F"],
                   {"F": {"mean": 1.5, "cov": 0.17, "dist": "lognormal"}})
        assert res.beta == pytest.approx(2.32, abs=0.005)
        assert res.pf == pytest.approx(0.0102, abs=0.0005)
        # design point at the failure threshold F* = 1
        assert res.design_point["F"] == pytest.approx(1.0, abs=1e-4)


class TestNonNormalMarginals:
    def test_uniform_variable_single(self):
        # S ~ U(8, 12), failure S > 11 -> margin g = 11 - S,
        # exact pf = 0.25 -> beta = -PHI^-1(0.25)
        from scipy import stats as sps
        res = form(lambda v: 11.0 - v["S"],
                   {"S": {"lower": 8.0, "upper": 12.0, "dist": "uniform"}},
                   convention="margin")
        assert res.converged
        assert res.pf == pytest.approx(0.25, abs=1e-3)
        assert res.beta == pytest.approx(float(-sps.norm.ppf(0.25)),
                                         abs=1e-3)

    def test_mean_in_failure_domain_gives_negative_beta(self):
        res = form(lambda v: v["F"],
                   {"F": {"mean": 0.8, "cov": 0.2, "dist": "lognormal"}})
        assert res.beta < 0
        assert res.pf > 0.5


class TestEngineAgreement:
    """Shared problem through all four engines (cross-engine sanity)."""
    SPEC = {"R": {"mean": 15.0, "std": 2.0}, "S": {"mean": 10.0, "std": 1.5}}

    @staticmethod
    def g(v):
        return v["R"] - v["S"]

    def test_all_engines_agree_linear_normal(self):
        r_fosm = fosm(self.g, self.SPEC, convention="margin")
        r_pem = pem(self.g, self.SPEC, convention="margin")
        r_form = form(self.g, self.SPEC, convention="margin")
        r_mc = monte_carlo(self.g, self.SPEC, n=200_000, seed=42,
                           convention="margin")
        assert r_fosm.beta_normal == pytest.approx(2.0, abs=1e-6)
        assert r_pem.beta_normal == pytest.approx(2.0, abs=1e-6)
        assert r_form.beta == pytest.approx(2.0, abs=1e-3)
        assert r_mc.beta_normal == pytest.approx(2.0, rel=0.02)
        assert r_mc.pf == pytest.approx(r_form.pf, rel=0.06)

    def test_fosm_form_diverge_for_nonnormal_as_expected(self):
        # for lognormal R,S FOSM (moments only) and FORM (full dists)
        # both run; FORM matches the exact result
        spec = {"R": {"mean": 20.0, "cov": 0.2, "dist": "lognormal"},
                "S": {"mean": 10.0, "cov": 0.25, "dist": "lognormal"}}
        exact = exact_lognormal_beta(20.0, 0.2, 10.0, 0.25)
        r_form = form(self.g, spec, convention="margin")
        r_mc = monte_carlo(self.g, spec, n=300_000, seed=1,
                           convention="margin")
        assert r_form.beta == pytest.approx(exact, abs=1e-3)
        from reliability.stats import pf_from_beta
        assert r_mc.pf == pytest.approx(pf_from_beta(exact), rel=0.10)


class TestPystraCrossCheck:
    def _has_pystra(self):
        try:
            from pystra_agent import has_pystra
            return has_pystra()
        except Exception:
            return False

    def test_form_vs_pystra_nonlinear(self):
        if not self._has_pystra():
            pytest.skip("pystra not installed")
        from pystra_agent import analyze_form

        spec = {"R": {"mean": 200.0, "std": 30.0, "dist": "lognormal"},
                "B": {"mean": 1.0, "std": 0.05, "dist": "normal"},
                "S": {"mean": 100.0, "std": 20.0, "dist": "lognormal"}}
        ours = form(lambda v: v["R"] * v["B"] - v["S"], spec,
                    convention="margin")
        theirs = analyze_form(
            variables=[
                {"name": "R", "dist": "lognormal", "mean": 200.0,
                 "stdv": 30.0},
                {"name": "B", "dist": "normal", "mean": 1.0, "stdv": 0.05},
                {"name": "S", "dist": "lognormal", "mean": 100.0,
                 "stdv": 20.0},
            ],
            limit_state="R*B - S")
        assert ours.converged
        assert ours.beta == pytest.approx(theirs.beta, abs=5e-3)
        assert ours.pf == pytest.approx(theirs.pf, rel=0.02)

    def test_form_vs_pystra_normal_linear(self):
        if not self._has_pystra():
            pytest.skip("pystra not installed")
        from pystra_agent import analyze_form
        ours = form(lambda v: v["R"] - v["S"],
                    {"R": {"mean": 200.0, "std": 20.0},
                     "S": {"mean": 100.0, "std": 30.0}},
                    convention="margin")
        theirs = analyze_form(
            variables=[
                {"name": "R", "dist": "normal", "mean": 200.0, "stdv": 20.0},
                {"name": "S", "dist": "normal", "mean": 100.0, "stdv": 30.0},
            ],
            limit_state="R - S")
        assert ours.beta == pytest.approx(theirs.beta, abs=2e-3)


class TestResultShape:
    def test_summary_and_to_dict(self):
        res = form(lambda v: v["F"], {"F": {"mean": 1.5, "cov": 0.2}})
        d = res.to_dict()
        assert d["engine"] == "form"
        assert "alpha_squared_pct" in d
        assert "FORM" in res.summary()

    def test_alpha_squared_sums_to_one(self):
        res = form(lambda v: v["R"] - v["S"],
                   {"R": {"mean": 15.0, "std": 2.0},
                    "S": {"mean": 10.0, "std": 1.5}},
                   convention="margin")
        assert sum(a * a for a in res.alphas.values()) == \
            pytest.approx(1.0, abs=1e-6)


class TestErrors:
    def test_zero_gradient_raises(self):
        with pytest.raises(ValueError, match="gradient"):
            form(lambda v: 5.0, {"x": {"mean": 1.0, "std": 0.1}},
                 convention="margin")
