"""
Tests for salib_agent — sensitivity analysis wrapper.

Tier 1: No SALib required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires SALib (integration tests with Ishigami function)
"""

import json
import math
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from salib_agent.salib_utils import has_salib
from salib_agent.results import SobolResult, MorrisResult

requires_salib = pytest.mark.skipif(
    not has_salib(), reason="SALib not installed"
)


# =====================================================================
# Ishigami test function: Y = sin(x1) + a*sin(x2)^2 + b*x3^4*sin(x1)
# =====================================================================

def _ishigami(X, a=7.0, b=0.1):
    """Ishigami function — standard SA test function."""
    return (np.sin(X[:, 0]) + a * np.sin(X[:, 1])**2 +
            b * X[:, 2]**4 * np.sin(X[:, 0]))


# =====================================================================
# Tier 1: SobolResult defaults
# =====================================================================

class TestSobolResultDefaults:

    def test_default_construction(self):
        r = SobolResult()
        assert r.n_samples == 0
        assert r.n_vars == 0
        assert r.S1 == []
        assert r.ST == []

    def test_construction_with_values(self):
        r = SobolResult(
            n_samples=2048,
            n_vars=3,
            var_names=["x1", "x2", "x3"],
            S1=[0.3, 0.4, 0.0],
            S1_conf=[0.01, 0.01, 0.01],
            ST=[0.5, 0.4, 0.2],
            ST_conf=[0.02, 0.02, 0.02],
        )
        assert r.n_vars == 3
        assert len(r.S1) == 3

    def test_summary_contains_variables(self):
        r = SobolResult(
            n_samples=2048, n_vars=3,
            var_names=["phi", "c", "gamma"],
            S1=[0.5, 0.3, 0.1],
            ST=[0.6, 0.3, 0.2],
        )
        s = r.summary()
        assert "phi" in s
        assert "c" in s
        assert "gamma" in s

    def test_to_dict_keys(self):
        r = SobolResult(
            n_samples=2048, n_vars=3,
            var_names=["x1", "x2", "x3"],
            S1=[0.3, 0.4, 0.0],
            S1_conf=[0.01, 0.01, 0.01],
            ST=[0.5, 0.4, 0.2],
            ST_conf=[0.02, 0.02, 0.02],
        )
        d = r.to_dict()
        assert "S1" in d
        assert "ST" in d
        assert "var_names" in d
        assert "n_samples" in d

    def test_to_dict_json_serializable(self):
        r = SobolResult(
            n_samples=2048, n_vars=3,
            var_names=["x1", "x2", "x3"],
            S1=[0.3, 0.4, 0.0],
            S1_conf=[0.01, 0.01, 0.01],
            ST=[0.5, 0.4, 0.2],
            ST_conf=[0.02, 0.02, 0.02],
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: MorrisResult defaults
# =====================================================================

class TestMorrisResultDefaults:

    def test_default_construction(self):
        r = MorrisResult()
        assert r.n_trajectories == 0
        assert r.n_vars == 0

    def test_construction_with_values(self):
        r = MorrisResult(
            n_trajectories=20, n_vars=3,
            var_names=["x1", "x2", "x3"],
            mu_star=[10.0, 5.0, 1.0],
            sigma=[8.0, 2.0, 0.5],
            mu_star_conf=[1.0, 0.5, 0.1],
        )
        assert r.n_vars == 3

    def test_summary_contains_variables(self):
        r = MorrisResult(
            n_trajectories=20, n_vars=3,
            var_names=["phi", "c", "gamma"],
            mu_star=[10.0, 5.0, 1.0],
            sigma=[8.0, 2.0, 0.5],
        )
        s = r.summary()
        assert "phi" in s
        assert "Morris" in s.upper() or "MORRIS" in s

    def test_to_dict_keys(self):
        r = MorrisResult(
            n_trajectories=20, n_vars=3,
            var_names=["x1", "x2", "x3"],
            mu_star=[10.0, 5.0, 1.0],
            sigma=[8.0, 2.0, 0.5],
            mu_star_conf=[1.0, 0.5, 0.1],
        )
        d = r.to_dict()
        assert "mu_star" in d
        assert "sigma" in d
        assert "var_names" in d

    def test_to_dict_json_serializable(self):
        r = MorrisResult(
            n_trajectories=20, n_vars=3,
            var_names=["x1", "x2", "x3"],
            mu_star=[10.0, 5.0, 1.0],
            sigma=[8.0, 2.0, 0.5],
            mu_star_conf=[1.0, 0.5, 0.1],
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke tests
# =====================================================================

class TestPlotSmoke:

    def test_sobol_plot(self):
        r = SobolResult(
            n_samples=2048, n_vars=3,
            var_names=["x1", "x2", "x3"],
            S1=[0.3, 0.4, 0.0],
            S1_conf=[0.01, 0.01, 0.01],
            ST=[0.5, 0.4, 0.2],
            ST_conf=[0.02, 0.02, 0.02],
        )
        ax = r.plot_sensitivity(show=False)
        assert ax is not None

    def test_morris_plot(self):
        r = MorrisResult(
            n_trajectories=20, n_vars=3,
            var_names=["x1", "x2", "x3"],
            mu_star=[10.0, 5.0, 1.0],
            sigma=[8.0, 2.0, 0.5],
            mu_star_conf=[1.0, 0.5, 0.1],
        )
        ax = r.plot_screening(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestValidation:

    def test_too_few_variables(self):
        from salib_agent.sensitivity import _validate_problem
        with pytest.raises(ValueError, match="at least 2"):
            _validate_problem(["x1"], [[0, 1]])

    def test_mismatched_lengths(self):
        from salib_agent.sensitivity import _validate_problem
        with pytest.raises(ValueError, match="same length"):
            _validate_problem(["x1", "x2"], [[0, 1]])

    def test_bad_bounds_length(self):
        from salib_agent.sensitivity import _validate_problem
        with pytest.raises(ValueError, match="2 values"):
            _validate_problem(["x1", "x2"], [[0, 1, 2], [0, 1]])

    def test_bad_bounds_order(self):
        from salib_agent.sensitivity import _validate_problem
        with pytest.raises(ValueError, match="min.*< max"):
            _validate_problem(["x1", "x2"], [[1, 0], [0, 1]])


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:

    def test_has_salib_returns_bool(self):
        assert isinstance(has_salib(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:

    def test_list_methods_all(self):
        from foundry.salib_agent_foundry import salib_list_methods
        result = json.loads(salib_list_methods(""))
        assert "Sobol" in result
        assert "Morris" in result

    def test_list_methods_filtered(self):
        from foundry.salib_agent_foundry import salib_list_methods
        result = json.loads(salib_list_methods("Sobol"))
        assert "sobol_analyze" in result["Sobol"]

    def test_list_methods_bad_category(self):
        from foundry.salib_agent_foundry import salib_list_methods
        result = json.loads(salib_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_sobol(self):
        from foundry.salib_agent_foundry import salib_describe_method
        result = json.loads(salib_describe_method("sobol_analyze"))
        assert "parameters" in result
        assert "var_names" in result["parameters"]

    def test_describe_morris(self):
        from foundry.salib_agent_foundry import salib_describe_method
        result = json.loads(salib_describe_method("morris_analyze"))
        assert "parameters" in result

    def test_describe_unknown(self):
        from foundry.salib_agent_foundry import salib_describe_method
        result = json.loads(salib_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from foundry.salib_agent_foundry import salib_agent
        result = json.loads(salib_agent("sobol_analyze", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from foundry.salib_agent_foundry import salib_agent
        result = json.loads(salib_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Sobol integration (requires SALib)
# =====================================================================

@requires_salib
class TestSobolIntegration:

    def test_sobol_sample_shape(self):
        from salib_agent import sobol_sample
        X = sobol_sample(
            var_names=["x1", "x2", "x3"],
            bounds=[[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
            n_samples=256,
        )
        # Expected: N*(2D+2) = 256*(2*3+2) = 2048
        assert X.shape == (2048, 3)

    def test_sobol_full_analysis(self):
        from salib_agent import sobol_sample, sobol_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = sobol_sample(names, bounds, n_samples=512, seed=42)
        Y = _ishigami(X)
        r = sobol_analyze(names, bounds, Y, n_samples=512, seed=42)
        assert r.n_vars == 3
        assert len(r.S1) == 3
        assert len(r.ST) == 3
        # x1 should be most important (S1 > 0.2)
        assert r.S1[0] > 0.2
        # x3 has high total-order but low first-order (interaction with x1)
        assert r.ST[2] > r.S1[2]

    def test_sobol_to_dict_json(self):
        from salib_agent import sobol_sample, sobol_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = sobol_sample(names, bounds, n_samples=256, seed=42)
        Y = _ishigami(X)
        r = sobol_analyze(names, bounds, Y, n_samples=256, seed=42)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_sobol_plot_integration(self):
        from salib_agent import sobol_sample, sobol_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = sobol_sample(names, bounds, n_samples=256, seed=42)
        Y = _ishigami(X)
        r = sobol_analyze(names, bounds, Y, n_samples=256, seed=42)
        ax = r.plot_sensitivity(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Morris integration (requires SALib)
# =====================================================================

@requires_salib
class TestMorrisIntegration:

    def test_morris_sample_shape(self):
        from salib_agent import morris_sample
        X = morris_sample(
            var_names=["x1", "x2", "x3"],
            bounds=[[-3.14, 3.14]] * 3,
            n_trajectories=20,
        )
        # Expected: n_trajectories * (n_vars + 1) = 20 * 4 = 80
        assert X.shape == (80, 3)

    def test_morris_full_analysis(self):
        from salib_agent import morris_sample, morris_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = morris_sample(names, bounds, n_trajectories=30, seed=42)
        Y = _ishigami(X)
        r = morris_analyze(names, bounds, X, Y, n_trajectories=30, seed=42)
        assert r.n_vars == 3
        assert len(r.mu_star) == 3
        assert len(r.sigma) == 3
        # All mu_star should be non-negative
        assert all(m >= 0 for m in r.mu_star)

    def test_morris_to_dict_json(self):
        from salib_agent import morris_sample, morris_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = morris_sample(names, bounds, n_trajectories=20, seed=42)
        Y = _ishigami(X)
        r = morris_analyze(names, bounds, X, Y, n_trajectories=20, seed=42)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_morris_plot_integration(self):
        from salib_agent import morris_sample, morris_analyze
        names = ["x1", "x2", "x3"]
        bounds = [[-3.14, 3.14]] * 3
        X = morris_sample(names, bounds, n_trajectories=20, seed=42)
        Y = _ishigami(X)
        r = morris_analyze(names, bounds, X, Y, n_trajectories=20, seed=42)
        ax = r.plot_screening(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Foundry integration (requires SALib)
# =====================================================================

@requires_salib
class TestFoundryIntegration:

    def test_foundry_sobol(self):
        from foundry.salib_agent_foundry import salib_agent
        params = {
            "var_names": ["x1", "x2", "x3"],
            "bounds": [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
            "n_samples": 256,
            "test_function": "ishigami",
        }
        result = json.loads(salib_agent("sobol_analyze", json.dumps(params)))
        assert "error" not in result
        assert "S1" in result
        assert len(result["S1"]) == 3

    def test_foundry_morris(self):
        from foundry.salib_agent_foundry import salib_agent
        params = {
            "var_names": ["x1", "x2", "x3"],
            "bounds": [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
            "n_trajectories": 20,
            "test_function": "ishigami",
        }
        result = json.loads(salib_agent("morris_analyze", json.dumps(params)))
        assert "error" not in result
        assert "mu_star" in result
        assert len(result["mu_star"]) == 3


# =====================================================================
# One-call analyses (live smoke G9, 2026-10-08): the model is evaluated
# here, only the indices come back. Ishigami (a=7, b=0.1, uniform on
# [-pi, pi]^3) analytic indices: S1 = 0.3139, 0.4424, 0; ST = 0.5576,
# 0.4424, 0.2437 (Saltelli et al. 2008, Global Sensitivity Analysis).
# =====================================================================

_ISHIGAMI_EXPR = "sin(x1) + 7*sin(x2)**2 + 0.1*x3**4*sin(x1)"
_PI_BOUNDS = [[-math.pi, math.pi]] * 3


@requires_salib
class TestOneCallAnalyses:

    def test_sobol_expression_reproduces_ishigami(self):
        from salib_agent import sobol_analysis
        r = sobol_analysis(["x1", "x2", "x3"], _PI_BOUNDS,
                           expression=_ISHIGAMI_EXPR, n_samples=1024)
        assert r.S1 == pytest.approx([0.3139, 0.4424, 0.0], abs=0.05)
        assert r.ST == pytest.approx([0.5576, 0.4424, 0.2437], abs=0.05)
        d = r.to_dict()
        assert d["n_samples"] == 1024 * (3 + 2)   # no second order
        assert "sample_matrix" not in d
        assert d["output_mean"] == pytest.approx(3.5, abs=0.2)  # E[Y] = a/2

    def test_sobol_callable_linear_model(self):
        """Y = 2 x1 + x2 on [0,1]^2: Var = 4/12 + 1/12 -> S1 = 0.8, 0.2."""
        from salib_agent import sobol_analysis
        r = sobol_analysis(["x1", "x2"], [[0, 1], [0, 1]],
                           model=lambda v: 2 * v["x1"] + v["x2"],
                           n_samples=512)
        assert r.S1 == pytest.approx([0.8, 0.2], abs=0.03)
        assert r.ST == pytest.approx([0.8, 0.2], abs=0.03)

    def test_morris_expression_ranks_ishigami(self):
        from salib_agent import morris_analysis
        r = morris_analysis(["x1", "x2", "x3"], _PI_BOUNDS,
                            expression=_ISHIGAMI_EXPR, n_trajectories=50)
        d = r.to_dict()
        assert len(d["mu_star"]) == 3
        assert "sample_matrix" not in d
        assert all(m > 0 for m in d["mu_star"])

    def test_exactly_one_model_source(self):
        from salib_agent import sobol_analysis
        with pytest.raises(ValueError, match="exactly one"):
            sobol_analysis(["a", "b"], [[0, 1], [0, 1]])
        with pytest.raises(ValueError, match="exactly one"):
            sobol_analysis(["a", "b"], [[0, 1], [0, 1]],
                           model=lambda v: 0.0, expression="a+b")

    def test_scientific_notation_accepted(self):
        from salib_agent import compile_expression
        f = compile_expression("1e-3*a + 2.5E2*b", ["a", "b"])
        assert f({"a": 1000.0, "b": 0.004}) == pytest.approx(2.0)

    def test_unsafe_expression_refused(self):
        from salib_agent import sobol_analysis
        for bad in ("__import__('os')", "a.__class__", "[a for a in b]",
                    "open('x')", "'a'*3"):
            with pytest.raises(ValueError):
                sobol_analysis(["a", "b"], [[0, 1], [0, 1]], expression=bad)

    def test_non_finite_output_is_named(self):
        from salib_agent import sobol_analysis
        with pytest.raises(ValueError, match="non-finite"):
            sobol_analysis(["a", "b"], [[0, 1], [0, 1]],
                           expression="a * 1e308 * 1e10 + b", n_samples=64)
        # A model that raises is reported with the point it failed at.
        with pytest.raises(ValueError, match="failed at sample"):
            sobol_analysis(["a", "b"], [[0, 1], [0, 1]],
                           expression="log(a - 0.5) + b", n_samples=64)

    def test_evaluation_limit(self):
        from salib_agent import sobol_analysis
        with pytest.raises(ValueError, match="model evaluations"):
            sobol_analysis(["a", "b"], [[0, 1], [0, 1]], expression="a+b",
                           n_samples=1024, max_evaluations=100)
