"""Tests for the reliability adapter (funhouse dispatch surface)."""

import json
import math

import pytest

from funhouse_agent.adapters.reliability_adapter import (
    METHOD_INFO, METHOD_REGISTRY,
)
from funhouse_agent.dispatch import call_agent

RS_PARAMS = {
    "variables": {"R": {"mean": 15.0, "std": 2.0},
                  "S": {"mean": 10.0, "std": 1.5}},
    "g_expression": "R - S",
    "convention": "margin",
}


class TestRegistration:
    def test_module_registered(self):
        from funhouse_agent.adapters import MODULE_REGISTRY
        assert "reliability" in MODULE_REGISTRY
        assert MODULE_REGISTRY["reliability"]["adapter"] == \
            "funhouse_agent.adapters.reliability_adapter"

    def test_method_info_covers_registry(self):
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)
        for name, info in METHOD_INFO.items():
            assert info["brief"], name
            assert info["category"], name
            assert "parameters" in info and "returns" in info, name

    def test_analysis_module_not_reference(self):
        from funhouse_agent.dispatch import ANALYSIS_MODULES
        assert "reliability" in ANALYSIS_MODULES

    def test_results_json_serializable(self):
        r = call_agent("reliability", "fosm", RS_PARAMS)
        json.dumps(r)


class TestEngines:
    def test_fosm_exact_linear(self):
        r = call_agent("reliability", "fosm", RS_PARAMS)
        assert "error" not in r
        assert r["beta_normal"] == pytest.approx(2.0, abs=1e-3)
        assert r["variance_contributions_pct"]["R"] == pytest.approx(
            100 * 4.0 / 6.25, abs=0.1)

    def test_pem(self):
        r = call_agent("reliability", "pem", RS_PARAMS)
        assert r["beta_normal"] == pytest.approx(2.0, abs=1e-3)
        assert r["n_points"] == 4

    def test_monte_carlo_seeded(self):
        params = dict(RS_PARAMS, n=50000, seed=42)
        r = call_agent("reliability", "monte_carlo", params)
        assert r["pf"] == pytest.approx(0.02275, rel=0.1)
        assert r["pf_ci95"][0] < 0.02275 < r["pf_ci95"][1]
        r2 = call_agent("reliability", "monte_carlo", params)
        assert r2["pf"] == r["pf"]  # reproducible

    def test_form_lognormal_duncan_anchor(self):
        r = call_agent("reliability", "form", {
            "variables": {"F": {"mean": 1.5, "cov": 0.17,
                                "dist": "lognormal"}},
            "g_expression": "F",
            "convention": "fos",
        })
        assert r["beta"] == pytest.approx(2.32, abs=0.01)
        assert r["converged"] is True
        assert r["design_point"]["F"] == pytest.approx(1.0, abs=1e-3)

    def test_correlation_pairwise_string_keys(self):
        params = dict(RS_PARAMS, correlation={"R,S": 0.5})
        r = call_agent("reliability", "fosm", params)
        exact = 5.0 / math.sqrt(4 + 2.25 - 2 * 0.5 * 2.0 * 1.5)
        assert r["beta_normal"] == pytest.approx(exact, abs=1e-3)

    def test_math_functions_in_expression(self):
        r = call_agent("reliability", "fosm", {
            "variables": {"phi": {"mean": 32.0, "cov": 0.08},
                          "tau": {"mean": 40.0, "cov": 0.15}},
            "g_expression": "100*tan(radians(phi))/tau",
        })
        assert "error" not in r
        assert r["g_mean"] == pytest.approx(
            100 * math.tan(math.radians(32.0)) / 40.0, rel=0.01)


class TestKnowledgeBase:
    def test_cov_guidance_duncan(self):
        r = call_agent("reliability", "cov_guidance",
                       {"property": "friction_angle",
                        "category": "inherent"})
        assert r["n_entries"] == 1
        e = r["entries"][0]
        assert (e["cov_min_pct"], e["cov_max_pct"]) == (2, 13)
        assert "Duncan (2000)" in e["source"]

    def test_cov_guidance_filters(self):
        r = call_agent("reliability", "cov_guidance",
                       {"property": "su", "soil_type": "clay",
                        "category": "site_specific"})
        assert r["entries"][0]["cov_mean_pct"] == pytest.approx(28.2)

    def test_combined_cov(self):
        r = call_agent("reliability", "combined_cov",
                       {"cov_inherent": 0.3, "cov_measurement": 0.1,
                        "cov_transformation": 0.05})
        assert r["cov_total"] == pytest.approx(
            math.sqrt(0.09 + 0.01 + 0.0025), abs=1e-6)

    def test_variance_reduction_with_delta(self):
        r = call_agent("reliability", "variance_reduction",
                       {"L": 2.0, "delta": 2.0})
        assert r["gamma_squared"] == pytest.approx(
            0.5 * (1 + math.exp(-2.0)), abs=1e-6)

    def test_variance_reduction_soil_type_lookup(self):
        r = call_agent("reliability", "variance_reduction",
                       {"L": 10.0, "soil_type": "clay"})
        assert "error" not in r
        assert r["delta"] > 0
        assert "delta_guidance" in r
        assert 0 < r["gamma_squared"] < 1


class TestErrors:
    def test_missing_params(self):
        r = call_agent("reliability", "fosm", {})
        assert "missing required parameter" in r["error"]

    def test_unknown_identifier_in_expression(self):
        r = call_agent("reliability", "fosm", {
            "variables": {"R": {"mean": 1.5, "cov": 0.1}},
            "g_expression": "R - Sneaky"})
        assert "Unknown identifier 'Sneaky'" in r["error"]

    def test_no_builtins_escape(self):
        r = call_agent("reliability", "fosm", {
            "variables": {"R": {"mean": 1.5, "cov": 0.1}},
            "g_expression": "__import__('os').system('echo hi')"})
        assert "error" in r

    def test_bad_convention(self):
        r = call_agent("reliability", "fosm",
                       dict(RS_PARAMS, convention="vibes"))
        assert "convention" in r["error"]

    def test_bad_sampling(self):
        r = call_agent("reliability", "monte_carlo",
                       dict(RS_PARAMS, sampling="sobol"))
        assert "sampling" in r["error"]

    def test_unknown_property_guidance(self):
        r = call_agent("reliability", "cov_guidance",
                       {"property": "swagger"})
        assert "No COV guidance" in r["error"]

    def test_variance_reduction_needs_delta_or_soil(self):
        r = call_agent("reliability", "variance_reduction", {"L": 5.0})
        assert "delta" in r["error"]
