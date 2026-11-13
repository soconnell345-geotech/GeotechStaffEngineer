"""GM-322: a pystra result says what it is.

- a failed SORM returns no second-order numbers (never FORM's under the
  SORM names) and says it failed;
- ``converged`` is pystra's own flag, not "beta > 0";
- nothing falls back to an empty value without a line in ``warnings``;
- the Monte Carlo seed is local to the call, every sample asked for is run,
  and a run with no failure reports no beta and no COV;
- a limit state may use the math functions the compiler allows.

All values are made up.
"""

import numpy as np
import pytest

from pystra_agent import (
    analyze_form, analyze_monte_carlo, analyze_sorm, has_pystra,
)
from pystra_agent import reliability as rel
from pystra_agent.pystra_utils import _Run

pytestmark = pytest.mark.skipif(not has_pystra(), reason="Requires pystra")


def N(name, mean, stdv, dist="normal"):
    return {"name": name, "dist": dist, "mean": mean, "stdv": stdv}


SAFE = [N("R", 200, 20), N("S", 100, 30)]
#: the mean point is already in the failure domain (R < S on average)
UNSAFE = [N("R", 80, 10), N("S", 100, 10)]


class TestConvergedIsPystrasFlag:
    def test_converged_run_with_negative_beta_is_converged(self):
        """beta < 0 (mean point in the failure domain) used to read as
        'not converged'."""
        r = analyze_form(UNSAFE, "R - S")
        assert r.beta == pytest.approx(-20 / np.hypot(10, 10), abs=1e-6)
        assert r.pf == pytest.approx(0.92135, abs=1e-4)
        assert r.converged is True
        assert r.converged_source
        assert r.alpha and r.design_point_x and r.design_point_u

    def test_unconverged_run_with_positive_beta_is_not_converged(
            self, monkeypatch):
        """An unconverged last iterate with beta > 0 (pystra 1.x keeps it)
        used to read as 'converged'."""
        def fake(sm, lsf, with_sorm=False):
            run = _Run()
            run.beta, run.pf = 2.5, 0.0062
            run.alpha = np.array([0.6, -0.8])
            run.x = np.array([170.0, 150.0])
            run.u = np.array([-1.5, 2.0])
            run.iterations = 100
            run.converged = False
            run.converged_source = "test"
            run.message = "FORM iteration limit (100) reached"
            return run
        monkeypatch.setattr(rel, "run_form", fake)
        r = analyze_form(SAFE, "R - S")
        assert r.converged is False
        assert r.beta == 2.5
        assert any("did not converge" in w and "last iterate" in w
                   for w in r.warnings)
        assert r.to_dict()["converged"] is False

    def test_unconverged_run_without_values_says_so(self, monkeypatch):
        """pystra 2.x gives no index or design point for an unconverged
        run: none is invented, and each missing quantity is named."""
        def fake(sm, lsf, with_sorm=False):
            run = _Run()
            run.iterations = 100
            run.converged = False
            run.message = "FORM iteration limit reached"
            run.n_evaluations = 700
            if with_sorm:
                run.sorm_converged = False
                run.sorm_message = "SORM was not run"
            return run
        monkeypatch.setattr(rel, "run_form", fake)
        r = analyze_form(SAFE, "R - S")
        assert r.converged is False
        assert r.beta is None and r.pf is None
        assert r.alpha == {} and r.design_point_x == {}
        text = " ".join(r.warnings)
        assert "did not converge" in text
        assert "no sensitivity factors" in text
        assert "no design point (physical space)" in text
        assert "not reported" in r.summary()
        s = analyze_sorm(SAFE, "R - S")
        assert s.converged is False and s.sorm_failed is True
        assert s.beta_breitung is None and s.pf_breitung is None
        assert "not reported" in s.summary()

    def test_mismatched_array_is_refused_not_mispaired(self, monkeypatch):
        def fake(sm, lsf, with_sorm=False):
            run = _Run()
            run.beta, run.pf, run.iterations = 2.0, 0.0228, 3
            run.converged = True
            run.alpha = np.array([1.0])              # one value, two names
            run.x = np.array([150.0, 150.0])
            run.u = np.array([-1.0, 1.0])
            run.n_evaluations = 12
            return run
        monkeypatch.setattr(rel, "run_form", fake)
        r = analyze_form(SAFE, "R - S")
        assert r.alpha == {}
        assert any("cannot" in w and "alpha" in w for w in r.warnings)
        assert r.design_point_x == {"R": 150.0, "S": 150.0}


class TestFunctionCalls:
    def test_count_is_pystras_or_flagged_as_an_estimate(self):
        r = analyze_form(SAFE, "R - S")
        d = r.to_dict()
        if d["n_function_calls_counted"]:
            assert d["n_function_calls"] > d["n_iterations"]
            assert not any("estimate" in w for w in d["warnings"])
        else:
            assert d["n_function_calls"] == d["n_iterations"] * 2
            assert any("estimate" in w for w in d["warnings"])


class TestSormFailure:
    def test_real_failed_sorm_returns_no_sorm_numbers(self):
        """With the mean point in the failure domain beta < 0 and
        Breitung's formula is undefined: pystra gives no second-order
        value. The FORM values used to come back under the SORM names."""
        r = analyze_sorm(UNSAFE, "R - S")
        assert r.beta_form == pytest.approx(-1.41421, abs=1e-4)
        assert r.pf_form == pytest.approx(0.92135, abs=1e-4)
        assert r.beta_breitung is None
        assert r.pf_breitung is None
        assert r.sorm_failed is True and r.sorm_converged is False
        assert r.form_converged is True
        assert r.converged is False
        assert "undefined" in r.sorm_message
        assert any(w.startswith("SORM FAILED") for w in r.warnings)
        d = r.to_dict()
        assert d["beta_breitung"] is None and d["pf_breitung"] is None
        assert d["sorm_failed"] is True
        assert "FAILED" in r.summary()

    def test_unconverged_form_runs_no_sorm(self):
        """g never reaches 0, so FORM does not converge. SORM is then not
        fitted on either pystra version (1.x used to fit it at the last
        iterate)."""
        r = analyze_sorm([N("R", 200, 20)], "(R - 190)**2 + 1")
        if r.form_converged:
            pytest.skip("FORM converged on this pystra version")
        assert r.sorm_failed is True and r.sorm_converged is False
        assert r.beta_breitung is None and r.pf_breitung is None
        assert r.sorm_message.startswith("SORM was not run")

    def test_good_sorm_is_not_flagged(self):
        r = analyze_sorm(SAFE, "R - S")
        assert r.sorm_failed is False and r.converged is True
        assert r.beta_breitung == pytest.approx(r.beta_form, abs=1e-6)
        assert r.warnings == [] or all("estimate" in w for w in r.warnings)

    def test_sorm_curvature_correction_matches_sampling(self):
        """A curved limit state: SORM moves pf from FORM's 0.0186 to
        0.0195, where crude Monte Carlo puts it."""
        variables = [N("R", 200, 20), N("S", 100, 30, "lognormal")]
        expr = "R - sqrt(S)*14"
        f = analyze_form(variables, expr)
        s = analyze_sorm(variables, expr)
        m = analyze_monte_carlo(variables, expr, n_samples=200_000, seed=1)
        assert f.pf == pytest.approx(0.018614, rel=1e-3)
        assert s.pf_breitung == pytest.approx(0.019528, rel=1e-3)
        assert s.kappa[0] == pytest.approx(-0.04386, abs=2e-4)
        assert m.pf == pytest.approx(s.pf_breitung, rel=0.05)
        assert abs(m.pf - s.pf_breitung) < abs(m.pf - f.pf)


class TestLimitStateFunctions:
    @pytest.mark.parametrize("expr", [
        "R - S if R > 0 else -1", "R - max(S, 50)", "R - abs(S)",
        "R - exp(log(S))"])
    def test_functions_and_conditionals_evaluate(self, expr):
        """pystra hands the limit state arrays of points; math functions
        and conditionals take one number, so they failed on every call."""
        variables = [N("R", 200, 20), N("S", 100, 30, "lognormal")]
        plain = analyze_form(variables, "R - S")
        r = analyze_form(variables, expr)
        assert r.beta == pytest.approx(plain.beta, rel=1e-5)
        m = analyze_monte_carlo(variables, expr, n_samples=20_000, seed=2)
        ref = analyze_monte_carlo(variables, "R - S", n_samples=20_000,
                                  seed=2)
        assert m.pf == ref.pf


class TestMonteCarloSeedAndCounts:
    def test_seed_repeats_and_differs(self):
        a = analyze_monte_carlo(SAFE, "R - S", n_samples=20_000, seed=11)
        b = analyze_monte_carlo(SAFE, "R - S", n_samples=20_000, seed=11)
        c = analyze_monte_carlo(SAFE, "R - S", n_samples=20_000, seed=12)
        assert a.pf == b.pf and a.seed == 11
        assert a.pf != c.pf
        assert not any("seed" in w for w in a.warnings)

    def test_no_seed_is_said(self):
        r = analyze_monte_carlo(SAFE, "R - S", n_samples=5_000)
        assert r.seed is None
        assert any("No seed was given" in w for w in r.warnings)

    def test_global_generator_neither_read_nor_changed(self):
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        a = analyze_monte_carlo(SAFE, "R - S", n_samples=5_000, seed=4)
        assert (np.random.get_state()[1] == before).all()
        np.random.seed(999)                # another global state ...
        b = analyze_monte_carlo(SAFE, "R - S", n_samples=5_000, seed=4)
        assert a.pf == b.pf                # ... the same seeded run

    @pytest.mark.parametrize("seed", [-1, 1.5, "7", True])
    def test_bad_seed_refused(self, seed):
        with pytest.raises(ValueError, match="seed must be"):
            analyze_monte_carlo(SAFE, "R - S", n_samples=1000, seed=seed)

    def test_every_sample_asked_for_is_run(self):
        """pystra stops by default once the COV of pf reaches 0.05 (here
        after about 61,000 samples); the count reported was the count asked
        for, and n_failures was pf times that."""
        r = analyze_monte_carlo(SAFE, "R - 150 + 0*S", n_samples=200_000,
                                seed=3)
        assert r.n_samples == r.n_samples_requested == 200_000
        assert r.n_failures == round(r.pf * 200_000)
        assert r.pf == pytest.approx(0.00621, rel=0.08)   # PHI(-2.5)
        assert r.cov_pf == pytest.approx(
            np.sqrt((1 - r.pf) / (200_000 * r.pf)))
        assert r.cov_pf < 0.05

    def test_no_failures_reports_no_beta_and_no_cov(self):
        r = analyze_monte_carlo(SAFE, "R - S + 500", n_samples=2_000,
                                seed=1)
        assert r.pf == 0.0 and r.n_failures == 0
        assert r.beta is None
        assert r.cov_pf is None
        assert r.pf_upper95_no_failures == pytest.approx(
            1 - 0.025 ** (1 / 2000))
        assert any("No failure in 2000 samples" in w for w in r.warnings)
        d = r.to_dict()
        assert d["beta"] is None and d["cov_pf"] is None
        assert "not reported" in r.summary()

    def test_every_sample_failed_reports_no_beta_and_no_cov(self):
        """GM-322 review: with pf = 1 the COV of pf is undefined, as at
        pf = 0; it used to read 0.0."""
        r = analyze_monte_carlo(SAFE, "R - S - 500", n_samples=2_000,
                                seed=1)
        assert r.pf == 1.0 and r.n_failures == 2000
        assert r.beta is None and r.cov_pf is None
        assert r.to_dict()["cov_pf"] is None
        assert any("Every sample failed" in w for w in r.warnings)
