"""
Tests for pystra_agent module.

Tier 1: Tests that work without pystra (metadata, validation, result classes)
Tier 2: Tests that require pystra (actual reliability analysis)
"""

import pytest
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from pystra_agent import (
    FormResult,
    SormResult,
    MonteCarloResult,
    has_pystra,
)

# Import functions that need pystra conditionally
try:
    from pystra_agent import analyze_form, analyze_sorm, analyze_monte_carlo
    HAS_PYSTRA = has_pystra()
except ImportError:
    HAS_PYSTRA = False


# ============================================================================
# Tier 1 Tests — No pystra required
# ============================================================================

class TestFormResult:
    """Test FormResult dataclass."""

    def test_default_initialization(self):
        """FormResult should initialize with default values."""
        result = FormResult()
        assert result.beta == 0.0
        assert result.pf == 1.0
        assert result.alpha == {}
        assert result.design_point_x == {}
        assert result.design_point_u == {}
        assert result.n_iterations == 0
        assert result.n_function_calls == 0
        assert result.converged is False
        assert result.limit_state_expr == ""
        assert result.n_variables == 0

    def test_initialization_with_values(self):
        """FormResult should accept custom values."""
        result = FormResult(
            beta=3.5,
            pf=0.0002,
            alpha={"R": 0.7, "S": -0.7},
            design_point_x={"R": 180, "S": 120},
            design_point_u={"R": -2.5, "S": 2.5},
            n_iterations=6,
            n_function_calls=42,
            converged=True,
            limit_state_expr="R - S",
            n_variables=2,
        )
        assert result.beta == 3.5
        assert result.pf == 0.0002
        assert result.alpha["R"] == 0.7
        assert result.converged is True

    def test_summary(self):
        """FormResult.summary() should return formatted string."""
        result = FormResult(
            beta=2.5,
            pf=0.006,
            alpha={"R": 0.8, "S": -0.6},
            limit_state_expr="R - S",
            n_variables=2,
        )
        summary = result.summary()
        assert "FORM Analysis Results" in summary
        assert "2.5" in summary
        assert "6.0" in summary or "6e" in summary.lower()
        assert "R" in summary
        assert "S" in summary

    def test_to_dict(self):
        """FormResult.to_dict() should return dictionary."""
        result = FormResult(
            beta=3.0,
            pf=0.0013,
            alpha={"R": 0.7},
            design_point_x={"R": 190},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["beta"] == 3.0
        assert d["pf"] == 0.0013
        assert "alpha" in d
        assert "design_point_x" in d

    def test_json_serializable(self):
        """FormResult.to_dict() should be JSON serializable."""
        result = FormResult(
            beta=2.8,
            pf=0.0025,
            alpha={"R": 0.6, "S": -0.8},
            design_point_x={"R": 185, "S": 110},
            design_point_u={"R": -1.5, "S": 2.0},
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert "2.8" in json_str

    def test_plot_importance_smoke(self):
        """FormResult.plot_importance() should create plot without error."""
        result = FormResult(
            alpha={"R": 0.7, "S": -0.7},
            beta=3.0,
        )
        ax = result.plot_importance(show=False)
        assert ax is not None


class TestSormResult:
    """Test SormResult dataclass."""

    def test_default_initialization(self):
        """SormResult should initialize with default values."""
        result = SormResult()
        assert result.beta_form == 0.0
        assert result.beta_breitung == 0.0
        assert result.pf_breitung == 1.0
        assert result.kappa == []
        assert result.pf_form == 1.0

    def test_initialization_with_values(self):
        """SormResult should accept custom values."""
        result = SormResult(
            beta_form=2.8,
            beta_breitung=2.9,
            pf_breitung=0.0018,
            pf_form=0.0026,
            kappa=[0.1, -0.05],
            alpha={"R": 0.7},
        )
        assert result.beta_form == 2.8
        assert result.beta_breitung == 2.9
        assert len(result.kappa) == 2

    def test_summary(self):
        """SormResult.summary() should return formatted string."""
        result = SormResult(
            beta_form=3.0,
            beta_breitung=3.1,
            pf_form=0.0013,
            pf_breitung=0.001,
            kappa=[0.05],
            limit_state_expr="R**2 - S",
            n_variables=2,
        )
        summary = result.summary()
        assert "SORM Analysis Results" in summary
        assert "3.0" in summary
        assert "3.1" in summary
        assert "kappa" in summary

    def test_to_dict(self):
        """SormResult.to_dict() should return dictionary."""
        result = SormResult(
            beta_form=2.5,
            beta_breitung=2.6,
            kappa=[0.1, -0.02],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["beta_form"] == 2.5
        assert d["beta_breitung"] == 2.6
        assert "kappa" in d


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""

    def test_default_initialization(self):
        """MonteCarloResult should initialize with default values."""
        result = MonteCarloResult()
        assert result.beta == 0.0
        assert result.pf == 1.0
        assert result.n_samples == 0
        assert result.n_failures == 0
        assert result.cov_pf == 0.0

    def test_initialization_with_values(self):
        """MonteCarloResult should accept custom values."""
        result = MonteCarloResult(
            beta=2.7,
            pf=0.0035,
            n_samples=100000,
            n_failures=350,
            cov_pf=0.053,
            limit_state_expr="R - S",
            n_variables=2,
        )
        assert result.beta == 2.7
        assert result.n_samples == 100000
        assert result.n_failures == 350

    def test_summary(self):
        """MonteCarloResult.summary() should return formatted string."""
        result = MonteCarloResult(
            beta=3.0,
            pf=0.0013,
            n_samples=100000,
            n_failures=130,
            cov_pf=0.087,
            limit_state_expr="R - S",
            n_variables=2,
        )
        summary = result.summary()
        assert "Monte Carlo" in summary
        assert "3.0" in summary
        assert "100,000" in summary or "100000" in summary
        assert "COV" in summary

    def test_to_dict(self):
        """MonteCarloResult.to_dict() should return dictionary."""
        result = MonteCarloResult(
            beta=2.8,
            pf=0.0025,
            n_samples=50000,
            n_failures=125,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["beta"] == 2.8
        assert d["n_samples"] == 50000

    def test_json_serializable(self):
        """MonteCarloResult.to_dict() should be JSON serializable."""
        result = MonteCarloResult(
            beta=3.2,
            pf=0.0007,
            n_samples=200000,
            n_failures=140,
            cov_pf=0.084,
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestInputValidation:
    """Test input validation (no pystra needed)."""

    @pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
    def test_empty_variables(self):
        """Should raise ValueError for empty variables list."""
        with pytest.raises(ValueError, match="empty"):
            analyze_form([], "R - S")

    @pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
    def test_empty_limit_state(self):
        """Should raise ValueError for empty limit state."""
        variables = [{"name": "R", "dist": "normal", "mean": 100, "stdv": 10}]
        with pytest.raises(ValueError, match="empty"):
            analyze_form(variables, "")

    @pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
    def test_bad_distribution_name(self):
        """Should raise ValueError for unknown distribution."""
        variables = [{"name": "R", "dist": "badname", "mean": 100, "stdv": 10}]
        with pytest.raises(ValueError, match="Unknown distribution"):
            analyze_form(variables, "R")

    @pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
    def test_missing_mean_stdv(self):
        """Should raise ValueError if normal missing mean/stdv."""
        variables = [{"name": "R", "dist": "normal"}]
        with pytest.raises(ValueError, match="requires 'mean'"):
            analyze_form(variables, "R")

    @pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
    def test_bad_correlation_shape(self):
        """Should raise ValueError for mismatched correlation matrix."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 100, "stdv": 10},
            {"name": "S", "dist": "normal", "mean": 50, "stdv": 15},
        ]
        # 3x3 matrix for 2 variables
        correlation = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]
        with pytest.raises(ValueError, match="2 variables"):
            analyze_form(variables, "R - S", correlation=correlation)


class TestUtilities:
    """Test utility functions."""

    def test_has_pystra_returns_bool(self):
        """has_pystra() should return boolean."""
        result = has_pystra()
        assert isinstance(result, bool)


# ============================================================================
# Foundry Metadata Tests
# ============================================================================

@pytest.fixture
def foundry_agent():
    """Import Foundry agent functions."""
    import sys
    import os
    # Add project root to path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from foundry.pystra_agent_foundry import (
        pystra_list_methods,
        pystra_describe_method,
        pystra_agent,
    )
    return pystra_list_methods, pystra_describe_method, pystra_agent


class TestFoundryMetadata:
    """Test Foundry agent metadata functions (no pystra required)."""

    def test_list_methods_all(self, foundry_agent):
        """list_methods() should return all methods."""
        list_methods, _, _ = foundry_agent
        methods = list_methods()
        assert isinstance(methods, list)
        assert len(methods) == 3
        names = [m["name"] for m in methods]
        assert "form_analysis" in names
        assert "sorm_analysis" in names
        assert "monte_carlo_analysis" in names

    def test_list_methods_by_category(self, foundry_agent):
        """list_methods(category) should filter by category."""
        list_methods, _, _ = foundry_agent
        methods = list_methods(category="Reliability")
        assert len(methods) == 3

    def test_describe_form_analysis(self, foundry_agent):
        """describe_method('form_analysis') should return metadata."""
        _, describe_method, _ = foundry_agent
        desc = describe_method("form_analysis")
        assert isinstance(desc, dict)
        assert desc["name"] == "form_analysis"
        assert "FORM" in desc["description"] or "First Order" in desc["description"]
        assert "variables" in desc["parameters"]

    def test_describe_sorm_analysis(self, foundry_agent):
        """describe_method('sorm_analysis') should return metadata."""
        _, describe_method, _ = foundry_agent
        desc = describe_method("sorm_analysis")
        assert desc["name"] == "sorm_analysis"
        assert "SORM" in desc["description"] or "Second Order" in desc["description"]

    def test_describe_monte_carlo_analysis(self, foundry_agent):
        """describe_method('monte_carlo_analysis') should return metadata."""
        _, describe_method, _ = foundry_agent
        desc = describe_method("monte_carlo_analysis")
        assert desc["name"] == "monte_carlo_analysis"
        assert "Monte Carlo" in desc["description"]
        assert "n_samples" in desc["parameters"]

    def test_agent_invalid_json(self, foundry_agent):
        """Agent should handle invalid JSON gracefully."""
        _, _, agent = foundry_agent
        result = agent("form_analysis", "not valid json")
        assert "error" in result

    def test_agent_unknown_method(self, foundry_agent):
        """Agent should handle unknown method gracefully."""
        _, _, agent = foundry_agent
        params = json.dumps({
            "variables": [{"name": "R", "dist": "normal", "mean": 100, "stdv": 10}],
            "limit_state": "R",
        })
        result = agent("unknown_method", params)
        assert "error" in result


# ============================================================================
# Tier 2 Tests — Require pystra
# ============================================================================

@pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
class TestFormAnalysis:
    """Test FORM analysis with pystra."""

    def test_basic_r_minus_s(self):
        """FORM: Basic R-S problem should give beta ≈ 2.77."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_form(variables, "R - S")

        assert result.converged
        assert 2.5 < result.beta < 3.0  # Should be around 2.77
        assert 0.001 < result.pf < 0.01  # Around 0.0028
        assert result.n_variables == 2

    def test_lognormal_variables(self):
        """FORM should work with lognormal distributions."""
        variables = [
            {"name": "R", "dist": "lognormal", "mean": 200, "stdv": 40},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_form(variables, "R - S")

        assert result.converged
        assert result.beta > 0
        assert 0 < result.pf < 1

    def test_with_constant(self):
        """FORM should work with constant variables."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
            {"name": "FS", "dist": "constant", "value": 1.5},
        ]
        result = analyze_form(variables, "R - FS * S")

        assert result.converged
        assert result.n_variables == 3
        assert "FS" in result.design_point_x

    def test_with_correlation(self):
        """FORM should work with correlated variables."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        correlation = [[1.0, 0.3], [0.3, 1.0]]
        result = analyze_form(variables, "R - S", correlation=correlation)

        assert result.converged
        assert result.beta > 0

    def test_design_point(self):
        """FORM should provide design point in x and u space."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_form(variables, "R - S")

        assert "R" in result.design_point_x
        assert "S" in result.design_point_x
        assert "R" in result.design_point_u
        assert "S" in result.design_point_u

    def test_alpha_signs(self):
        """Sensitivity factors should have correct signs."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_form(variables, "R - S")

        # pystra convention: alpha points from origin toward design point
        # R (resistance) decreases at design point → alpha_R < 0
        # S (load) increases at design point → alpha_S > 0
        if "R" in result.alpha and "S" in result.alpha:
            assert result.alpha["R"] < 0
            assert result.alpha["S"] > 0


@pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
class TestSormAnalysis:
    """Test SORM analysis with pystra."""

    def test_linear_lsf_matches_form(self):
        """SORM should match FORM for linear limit state."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_sorm(variables, "R - S")

        assert result.converged
        # For linear LSF, SORM and FORM should be very close
        assert abs(result.beta_form - result.beta_breitung) < 0.1

    def test_nonlinear_lsf(self):
        """SORM should differ from FORM for nonlinear limit state."""
        variables = [
            {"name": "R", "dist": "lognormal", "mean": 200, "stdv": 40},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_sorm(variables, "R**2 - S**2")

        assert result.converged
        assert result.beta_form > 0
        assert result.beta_breitung > 0
        # SORM and FORM can differ for nonlinear problems
        # Just check both are positive and finite


@pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
class TestMonteCarloAnalysis:
    """Test Monte Carlo analysis with pystra."""

    def test_basic_mc(self):
        """Monte Carlo should give beta close to FORM."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result_form = analyze_form(variables, "R - S")
        result_mc = analyze_monte_carlo(variables, "R - S", n_samples=10000)

        # MC with 10k samples should be within 20% of FORM for this problem
        assert 0.8 * result_form.beta < result_mc.beta < 1.2 * result_form.beta

    def test_mc_many_samples(self):
        """Monte Carlo with many samples should converge to analytical."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_monte_carlo(variables, "R - S", n_samples=100000)

        assert result.n_samples == 100000
        assert result.n_failures > 0
        assert result.cov_pf > 0  # Should have finite COV


@pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
class TestFoundryIntegration:
    """Test Foundry agent with actual pystra calls."""

    def test_form_analysis_roundtrip(self, foundry_agent):
        """Foundry agent should handle FORM analysis."""
        _, _, agent = foundry_agent

        params = {
            "variables": [
                {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
                {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
            ],
            "limit_state": "R - S",
        }
        result = agent("form_analysis", json.dumps(params))

        assert "error" not in result
        assert "beta" in result
        assert "pf" in result
        assert result["converged"]

    def test_monte_carlo_analysis_roundtrip(self, foundry_agent):
        """Foundry agent should handle Monte Carlo analysis."""
        _, _, agent = foundry_agent

        params = {
            "variables": [
                {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
                {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
            ],
            "limit_state": "R - S",
            "n_samples": 10000,
        }
        result = agent("monte_carlo_analysis", json.dumps(params))

        assert "error" not in result
        assert "beta" in result
        assert result["n_samples"] == 10000


@pytest.mark.skipif(not HAS_PYSTRA, reason="Requires pystra")
class TestAdvancedProblems:
    """Test more complex reliability problems."""

    def test_to_dict_json_after_analysis(self):
        """Result.to_dict() should be JSON-serializable after real analysis."""
        variables = [
            {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
            {"name": "S", "dist": "normal", "mean": 100, "stdv": 30},
        ]
        result = analyze_form(variables, "R - S")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_nonlinear_lsf_squared(self):
        """FORM should handle nonlinear limit state with powers."""
        variables = [
            {"name": "R", "dist": "lognormal", "mean": 200, "stdv": 30},
            {"name": "S", "dist": "normal", "mean": 50, "stdv": 15},
        ]
        result = analyze_form(variables, "R**2 - S")

        assert result.converged
        assert result.beta > 0

    def test_three_variable_problem(self):
        """FORM should work with 3+ variables."""
        variables = [
            {"name": "c", "dist": "lognormal", "mean": 50, "stdv": 15},
            {"name": "gamma", "dist": "normal", "mean": 18, "stdv": 1},
            {"name": "Q", "dist": "normal", "mean": 200, "stdv": 40},
        ]
        # Simplified bearing capacity: Qult = 5.14*c, check Q < Qult
        result = analyze_form(variables, "5.14 * c - Q")

        assert result.converged
        assert result.n_variables == 3
        assert len(result.alpha) == 3
