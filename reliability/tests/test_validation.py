"""Runnable validation anchors — ours vs published (mirrors VALIDATION.md).

Every row of reliability/VALIDATION.md that can be computed offline is
asserted here, so the validation table cannot silently rot.
"""

import math

import pytest

from reliability import (
    beta_lognormal, form, fosm, monte_carlo, pem, pf_from_beta,
)


class TestDuncan2000Anchors:
    """Duncan (2000), J. Geotech. Geoenviron. Eng. 126(4) worked examples."""

    def test_retaining_wall_sliding(self):
        # Paper: F_MLV = 1.50, COV_F = 17% -> pf ~ 1% (reliability ~99%)
        b = beta_lognormal(1.50, 0.17)
        assert b == pytest.approx(2.32, abs=0.005)
        assert pf_from_beta(b) == pytest.approx(0.01, abs=0.002)

    def test_lash_terminal_cut_slope(self):
        # Paper: F_MLV = 1.17, COV_F = 16% -> pf = 18%
        # (observed: ~22% of the 2000-ft slope length failed)
        b = beta_lognormal(1.17, 0.16)
        assert pf_from_beta(b) == pytest.approx(0.18, abs=0.005)

    def test_consolidation_settlement_exceedance(self):
        # Paper: most likely settlement 1.07 ft, COV 21%;
        # settlement with 1% exceedance probability ~ 1.6x the most likely
        # value (SR ~ 1.6). Lognormal settlement model:
        from scipy import stats as sps
        cov = 0.21
        s_ln = math.sqrt(math.log(1 + cov ** 2))
        mu_ln = math.log(1.07) - 0.5 * s_ln ** 2
        s99 = float(sps.lognorm(s=s_ln, scale=math.exp(mu_ln)).ppf(0.99))
        assert s99 / 1.07 == pytest.approx(1.6, abs=0.05)


class TestUFC_DM7_Anchors:
    """UFC 3-220-20 ch. 7 worked numbers (resurrected DM7Eqs test suite)."""

    def test_fosm_linear(self):
        res = fosm(lambda v: v["x"] - v["y"],
                   {"x": {"mean": 10.0, "std": 2.0},
                    "y": {"mean": 4.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(6.0)
        assert res.g_std == pytest.approx(math.sqrt(5.0))
        assert res.beta_normal == pytest.approx(2.683, abs=0.001)

    def test_pem_product(self):
        res = pem(lambda v: v["x"] * v["y"],
                  {"x": {"mean": 2.0, "std": 1.0},
                   "y": {"mean": 3.0, "std": 1.0}}, convention="margin")
        assert res.g_mean == pytest.approx(6.0)
        assert res.g_std ** 2 == pytest.approx(14.0)

    def test_seismic_rate_of_exceedance(self):
        from reliability import rate_of_exceedance_from_probability
        lam = rate_of_exceedance_from_probability(0.10, 50.0)
        assert lam == pytest.approx(0.002107, abs=2e-6)


class TestExactClosedForms:
    """USACE ETL 1110-2-547 App. B / Ang & Tang exact solutions."""

    def test_normal_margin_all_engines(self):
        spec = {"R": {"mean": 15.0, "std": 2.0},
                "S": {"mean": 10.0, "std": 1.5}}
        exact_beta = 5.0 / math.sqrt(4.0 + 2.25)  # = 2.0

        def g(v):
            return v["R"] - v["S"]

        assert fosm(g, spec, convention="margin").beta_normal == \
            pytest.approx(exact_beta, abs=1e-9)
        assert pem(g, spec, convention="margin").beta_normal == \
            pytest.approx(exact_beta, abs=1e-9)
        assert form(g, spec, convention="margin").beta == \
            pytest.approx(exact_beta, abs=1e-3)
        mc = monte_carlo(g, spec, n=200_000, seed=42, convention="margin")
        assert mc.pf == pytest.approx(pf_from_beta(exact_beta), rel=0.05)

    def test_lognormal_quotient_exact(self):
        mu_r, cov_r, mu_s, cov_s = 20.0, 0.2, 10.0, 0.25
        num = math.log((mu_r / mu_s)
                       * math.sqrt((1 + cov_s ** 2) / (1 + cov_r ** 2)))
        den = math.sqrt(math.log((1 + cov_r ** 2) * (1 + cov_s ** 2)))
        exact_beta = num / den
        spec = {"R": {"mean": mu_r, "cov": cov_r, "dist": "lognormal"},
                "S": {"mean": mu_s, "cov": cov_s, "dist": "lognormal"}}
        res = form(lambda v: v["R"] - v["S"], spec, convention="margin")
        assert res.beta == pytest.approx(exact_beta, abs=1e-3)


class TestEngineAgreementSharedProblem:
    """Bearing-capacity FOS through all four engines (VALIDATION.md table).

    Square footing B=2 m, Df=1.5 m on sand: phi' lognormal (32 deg,
    COV 8% — Duncan Table 1 upper range), gamma normal (18, COV 5%),
    q_applied = 700 kPa.
    """
    SPEC = {"friction_angle": {"mean": 32.0, "cov": 0.08,
                               "dist": "lognormal"},
            "unit_weight": {"mean": 18.0, "cov": 0.05}}

    @staticmethod
    def g(v):
        from bearing_capacity import (
            BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer,
        )
        res = BearingCapacityAnalysis(
            footing=Footing(width=2.0, depth=1.5, shape="square"),
            soil=BearingSoilProfile(layer1=SoilLayer(
                cohesion=0.0, friction_angle=v["friction_angle"],
                unit_weight=v["unit_weight"]))).compute()
        return res.q_ultimate / 700.0

    def test_engines_agree(self):
        r_fosm = fosm(self.g, self.SPEC)
        r_pem = pem(self.g, self.SPEC)
        r_form = form(self.g, self.SPEC)
        r_mc = monte_carlo(self.g, self.SPEC, n=20_000, seed=42)

        # moment methods agree (g strongly nonlinear in phi via exp(Nq):
        # PEM mean sits above g(means) by the convexity correction)
        assert r_pem.g_mean == pytest.approx(r_fosm.g_mean, rel=0.10)
        assert r_pem.g_cov == pytest.approx(r_fosm.g_cov, rel=0.10)
        # MC lognormal-fit beta vs FORM beta (both full-distribution)
        assert r_mc.beta_lognormal == pytest.approx(r_form.beta, rel=0.15)
        # FOSM lognormal beta vs FORM (first-order moments -> looser)
        assert r_fosm.beta_lognormal == pytest.approx(r_form.beta,
                                                      rel=0.25)
        # FORM pf falls inside the Monte Carlo 95% confidence interval
        assert r_mc.pf_ci95[0] <= r_form.pf <= r_mc.pf_ci95[1]
        assert r_mc.pf_ci95[0] <= r_mc.pf <= r_mc.pf_ci95[1]
