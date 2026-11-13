"""GM-323: the counts are checked against the design the samples imply,
and the expression compiler takes the reliability engine's limits.

All values are made up.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from salib_agent import (
    has_salib, morris_analyze, morris_sample, sobol_analyze, sobol_sample,
)
from salib_agent import expression as ex

pytestmark = pytest.mark.skipif(not has_salib(), reason="Requires SALib")

NAMES = ["a", "b", "c"]
BOUNDS = [[0, 1], [0, 2], [1, 3]]


def model(X):
    return X[:, 0] + 2.0 * X[:, 1] * X[:, 2]


class TestSobolCounts:
    @pytest.mark.parametrize("second,per", [(True, 8), (False, 5)])
    def test_n_samples_matching_the_design_is_accepted(self, second, per):
        X = sobol_sample(NAMES, BOUNDS, n_samples=16,
                         calc_second_order=second)
        assert len(X) == 16 * per
        r = sobol_analyze(NAMES, BOUNDS, model(X), n_samples=16,
                          calc_second_order=second)
        assert r.n_samples == 16 * per          # outputs analysed
        assert r.n_base_samples == 16           # the design's N
        assert r.to_dict()["n_base_samples"] == 16

    def test_n_samples_left_out_is_taken_from_y(self):
        X = sobol_sample(NAMES, BOUNDS, n_samples=16)
        r = sobol_analyze(NAMES, BOUNDS, model(X))
        assert r.n_base_samples == 16
        given = sobol_analyze(NAMES, BOUNDS, model(X), n_samples=16)
        assert r.S1 == given.S1 and r.ST == given.ST

    def test_n_samples_that_disagrees_with_y_is_refused(self):
        X = sobol_sample(NAMES, BOUNDS, n_samples=16)
        with pytest.raises(ValueError) as exc:
            sobol_analyze(NAMES, BOUNDS, model(X), n_samples=32)
        msg = str(exc.value)
        assert "n_samples=32 does not match Y" in msg
        assert "256 rows" in msg and "128 values (N=16)" in msg

    def test_y_that_is_not_a_design_is_refused(self):
        X = sobol_sample(NAMES, BOUNDS, n_samples=16)
        with pytest.raises(ValueError, match="not a Sobol design"):
            sobol_analyze(NAMES, BOUNDS, model(X)[:-1])

    def test_second_order_mismatch_is_named(self):
        """A sample made without second-order indices (N*(D+2) = 90 rows)
        analysed with calc_second_order=True (N*(2D+2))."""
        X = sobol_sample(NAMES, BOUNDS, n_samples=18,
                         calc_second_order=False)
        assert len(X) == 90
        with pytest.raises(ValueError) as exc:
            sobol_analyze(NAMES, BOUNDS, model(X), calc_second_order=True)
        assert "calc_second_order=False" in str(exc.value)

    def test_y_must_be_flat(self):
        with pytest.raises(ValueError, match="flat list"):
            sobol_analyze(NAMES, BOUNDS, np.zeros((8, 2)))


class TestMorrisCounts:
    def setup_method(self):
        self.X = morris_sample(NAMES, BOUNDS, n_trajectories=6)
        self.Y = model(self.X)

    def test_trajectories_are_counted_from_x(self):
        r = morris_analyze(NAMES, BOUNDS, self.X, self.Y)
        assert r.n_trajectories == 6
        r = morris_analyze(NAMES, BOUNDS, self.X, self.Y, n_trajectories=6)
        assert r.n_trajectories == 6

    def test_n_trajectories_that_disagrees_is_refused(self):
        """It used to be copied into the result as the run's count."""
        with pytest.raises(ValueError) as exc:
            morris_analyze(NAMES, BOUNDS, self.X, self.Y, n_trajectories=20)
        msg = str(exc.value)
        assert "n_trajectories=20 does not match X" in msg
        assert "80 rows" in msg and "X has 24 (6 trajectories)" in msg

    def test_y_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="Y length"):
            morris_analyze(NAMES, BOUNDS, self.X, self.Y[:-1])

    def test_x_that_is_not_a_design_is_refused(self):
        with pytest.raises(ValueError, match="not a Morris design"):
            morris_analyze(NAMES, BOUNDS, self.X[:-1], self.Y[:-1])
        with pytest.raises(ValueError, match="one column per variable"):
            morris_analyze(NAMES, BOUNDS, self.X[:, :2], self.Y)


class TestExpressionLimits:
    def test_limits_are_the_reliability_engines(self):
        """The numbers are restated in three places (an analysis module
        imports no other module's code); this keeps them one rule."""
        core = pytest.importorskip("aec_agent_core.safe_expr")
        from pystra_agent import pystra_utils as pu
        for name in ("_MAX_POW_BITS", "_MAX_EXPR_CHARS", "_MAX_NESTING"):
            assert getattr(ex, name) == getattr(core, name) \
                == getattr(pu, name), name

    def test_huge_integer_power_is_refused_not_computed(self):
        child = textwrap.dedent("""
            import time
            from salib_agent.expression import compile_expression
            t0 = time.time()
            for expr, values in (("9**9**9 + a", {"a": 1.0}),
                                 ("floor(a)**floor(a)**floor(a)",
                                  {"a": 9.0})):
                try:
                    compile_expression(expr, ["a"])(values)
                    print("no error")
                except OverflowError as exc:
                    print("OverflowError:", exc)
            print(time.time() - t0)
        """)
        proc = subprocess.run([sys.executable, "-c", child],
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-2000:]
        lines = proc.stdout.strip().splitlines()
        assert all(l.startswith("OverflowError") and "1e308" in l
                   for l in lines[:2]), lines
        assert float(lines[2]) < 5

    def test_ordinary_powers_still_work(self):
        f = ex.compile_expression("a**2 - b + 2**10", ["a", "b"])
        assert f({"a": 3.0, "b": 1.0}) == 1032.0
        assert ex.compile_expression("tan(radians(45 - a/2))**2", ["a"])(
            {"a": 30.0}) == pytest.approx(1 / 3)
        assert ex.compile_expression("a**b**2", ["a", "b"])(
            {"a": 2.0, "b": 2.0}) == 16.0
        # a variable cannot shadow the bounded power
        assert ex.compile_expression("__gse_pow__**2", ["__gse_pow__"])(
            {"__gse_pow__": 3.0}) == 9.0
        assert ex._bounded_pow(2, 1000) == 2 ** 1000
        with pytest.raises(OverflowError):
            ex._bounded_pow(2, 1100)

    def test_overlong_and_overdeep_expressions_are_refused(self):
        with pytest.raises(ValueError, match="too long"):
            ex.compile_expression("+".join(["a"] * 6000), ["a"])
        with pytest.raises(ValueError, match="too long or too deeply"):
            ex.compile_expression("+".join(["a"] * 250), ["a"])
        assert ex.compile_expression("+".join(["a"] * 150), ["a"])(
            {"a": 1.0}) == 150.0
        assert ex.compile_expression("+".join(["a**2"] * 150), ["a"])(
            {"a": 2.0}) == 600.0

    def test_one_call_analysis_reports_the_refusal(self):
        from salib_agent import sobol_analysis
        with pytest.raises(ValueError, match="1e308"):
            sobol_analysis(["a", "b"], [[1, 2], [1, 2]],
                           expression="floor(a + 8)**floor(a + 8)"
                                      "**floor(a + 8) + b",
                           n_samples=8)
