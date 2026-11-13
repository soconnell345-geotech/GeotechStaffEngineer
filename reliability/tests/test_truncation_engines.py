"""One made-up truncated case through all four engines (GM-322).

F = R / S, failure at F < 1. R is normal (mean 200, std 40) truncated to
[130, 270]; S is lognormal (mean 100, std 20). The values are invented.

What each engine does with the truncation:

- Monte Carlo samples the truncated R: pf falls from about 0.0138
  (untruncated) to about 0.0036.
- FORM reads the truncated R through its CDF: pf falls from about 0.0126
  to about 0.0049. It moves the same way as Monte Carlo and stays within a
  factor of 1.5 of it (FORM is first order, and the truncated variable's
  transform is strongly curved near the bound).
- FOSM and PEM evaluate g at mean +/- std and take no truncation: they
  refuse the truncated case. On the untruncated variables they run, and
  give the untruncated answer (0.0105 and 0.0066, lognormal index), which
  is three times and twice the truncated Monte Carlo pf: the answer they
  used to return, unmarked, for the truncated case.
"""

import pytest

from reliability import form, fosm, monte_carlo, pem


def g(v):
    return v["R"] / v["S"]


S = {"mean": 100.0, "std": 20.0, "dist": "lognormal"}
PLAIN = {"R": {"mean": 200.0, "std": 40.0}, "S": S}
TRUNCATED = {"R": {"mean": 200.0, "std": 40.0,
                   "lower": 130.0, "upper": 270.0}, "S": S}


class TestTruncatedCase:
    @pytest.mark.parametrize("engine,name", [(fosm, "FOSM"), (pem, "PEM")])
    def test_moment_engines_refuse(self, engine, name):
        with pytest.raises(ValueError) as exc:
            engine(g, TRUNCATED)
        msg = str(exc.value)
        assert f"{name} takes no truncation" in msg
        assert "'R'" in msg and "lower=130.0" in msg
        assert "monte_carlo or form" in msg

    def test_pem_multiplicative_refuses_too(self):
        with pytest.raises(ValueError, match="takes no truncation"):
            pem(g, TRUNCATED, scheme="multiplicative")

    @pytest.mark.parametrize("bound", ["lower", "upper"])
    def test_one_sided_truncation_refused(self, bound):
        spec = {"R": {"mean": 200.0, "std": 40.0,
                      bound: 130.0 if bound == "lower" else 270.0}, "S": S}
        with pytest.raises(ValueError, match="takes no truncation"):
            fosm(g, spec)

    def test_monte_carlo_and_form_read_the_truncation(self):
        mc = monte_carlo(g, TRUNCATED, n=200_000, seed=7)
        fo = form(g, TRUNCATED)
        assert fo.converged
        assert mc.pf == pytest.approx(0.0036, rel=0.10)
        assert fo.pf == pytest.approx(0.0049, rel=0.02)
        assert 1.0 < fo.pf / mc.pf < 1.5
        # both well below their own untruncated answers
        mc0 = monte_carlo(g, PLAIN, n=200_000, seed=7)
        fo0 = form(g, PLAIN)
        assert mc0.pf == pytest.approx(0.0138, rel=0.06)
        assert fo0.pf == pytest.approx(0.0126, rel=0.02)
        assert mc.pf < 0.35 * mc0.pf
        assert fo.pf < 0.45 * fo0.pf

    def test_moment_engines_run_untruncated_and_differ(self):
        fo = fosm(g, PLAIN)
        pe = pem(g, PLAIN)
        assert fo.pf_lognormal == pytest.approx(0.0105, rel=0.02)
        assert pe.pf_lognormal == pytest.approx(0.0066, rel=0.02)
        mc = monte_carlo(g, TRUNCATED, n=200_000, seed=7)
        # the untruncated moment answers are not the truncated case's
        assert fo.pf_lognormal > 2.5 * mc.pf
        assert pe.pf_lognormal > 1.6 * mc.pf

    def test_uniform_and_triangular_bounds_accepted(self):
        spec = {"R": {"lower": 150.0, "upper": 250.0, "dist": "uniform"},
                "S": {"lower": 60.0, "mode": 100.0, "upper": 140.0,
                      "dist": "triangular"}}
        assert fosm(g, spec).g_mean == pytest.approx(2.0)
        assert pem(g, spec).n_points == 4
