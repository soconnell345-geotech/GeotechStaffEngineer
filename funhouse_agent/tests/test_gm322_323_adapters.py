"""GM-322 / GM-323 through dispatch: what the pystra and salib tools say
they return is in the result, and the refusals reach the agent.

All values are made up.
"""

import pytest

from funhouse_agent.dispatch import call_agent

pystra = pytest.importorskip("pystra")
SALib = pytest.importorskip("SALib")


def N(name, mean, stdv):
    return {"name": name, "dist": "normal", "mean": mean, "stdv": stdv}


SAFE = {"variables": [N("R", 200, 20), N("S", 100, 30)],
        "limit_state": "R - S"}
UNSAFE = {"variables": [N("R", 80, 10), N("S", 100, 10)],
          "limit_state": "R - S"}


class TestPystraTools:
    @pytest.mark.parametrize("method,params", [
        ("form_analysis", SAFE),
        ("sorm_analysis", SAFE),
        ("sorm_analysis", UNSAFE),          # the failed-SORM shape
        ("monte_carlo_analysis", dict(SAFE, n_samples=2000, seed=1)),
        ("monte_carlo_analysis", dict(SAFE, n_samples=2000)),
    ])
    def test_every_key_the_tool_promises_is_in_the_result(self, method,
                                                          params):
        from funhouse_agent.adapters.pystra_adapter import METHOD_INFO
        r = call_agent("pystra", method, params)
        assert "error" not in r, r
        for key in METHOD_INFO[method]["returns"]:
            assert key in r, key

    def test_failed_sorm_reaches_the_agent_as_failed(self):
        r = call_agent("pystra", "sorm_analysis", UNSAFE)
        assert r["sorm_failed"] is True
        assert r["beta_breitung"] is None and r["pf_breitung"] is None
        assert r["beta_form"] == pytest.approx(-1.4142, abs=1e-3)
        assert r["form_converged"] is True and r["converged"] is False
        assert any("SORM FAILED" in w for w in r["warnings"])

    def test_negative_beta_form_is_converged(self):
        r = call_agent("pystra", "form_analysis", UNSAFE)
        assert r["beta"] < 0 and r["converged"] is True

    def test_seed_repeats_a_run(self):
        p = dict(SAFE, n_samples=20000, seed=5)
        a = call_agent("pystra", "monte_carlo_analysis", p)
        b = call_agent("pystra", "monte_carlo_analysis", p)
        assert a["pf"] == b["pf"] and a["seed"] == 5
        assert a["n_samples"] == a["n_samples_requested"] == 20000

    def test_seed_is_a_documented_input(self):
        from funhouse_agent.adapters.pystra_adapter import METHOD_INFO
        assert "seed" in METHOD_INFO["monte_carlo_analysis"]["parameters"]


class TestReliabilityPackage:
    def test_monte_carlo_package_says_what_was_dropped(self, tmp_path):
        """GM-322 review: the package's note says how many realizations
        were dropped; the agent-facing result says it too."""
        r = call_agent("calc_package", "reliability_package", {
            "engine": "monte_carlo", "convention": "margin",
            "g_expression": "x - 8.5 + 0*sqrt(12 - x)",
            "variables": {"x": {"mean": 10.0, "std": 1.0}},
            "n": 4000, "seed": 12,
            "output_path": str(tmp_path / "rel.html")})
        assert r["n_requested"] == 4000
        assert r["n_dropped"] > 0
        assert r["n"] == 4000 - r["n_dropped"]
        lo, hi = r["pf_bounds_dropped"]
        assert lo <= r["pf"] <= hi
        assert r["pf_ci95_method"] == "normal_approximation"
        assert any("dropped" in w or "not finite" in w
                   for w in r["warnings"])


class TestSalibTools:
    NAMES = ["a", "b"]
    BOUNDS = [[0, 1], [0, 1]]

    def _sobol(self, **extra):
        X = call_agent("salib", "sobol_sample", {
            "var_names": self.NAMES, "bounds": self.BOUNDS,
            "n_samples": 8})["sample_matrix"]
        Y = [row[0] + 2 * row[1] for row in X]
        return call_agent("salib", "sobol_analyze", {
            "var_names": self.NAMES, "bounds": self.BOUNDS, "Y": Y,
            **extra})

    def _morris(self, **extra):
        X = call_agent("salib", "morris_sample", {
            "var_names": self.NAMES, "bounds": self.BOUNDS,
            "n_trajectories": 4})["sample_matrix"]
        Y = [row[0] + 2 * row[1] for row in X]
        return call_agent("salib", "morris_analyze", {
            "var_names": self.NAMES, "bounds": self.BOUNDS, "X": X, "Y": Y,
            **extra})

    def test_sobol_counts(self):
        r = self._sobol(n_samples=8)
        assert r["n_samples"] == 48 and r["n_base_samples"] == 8
        assert self._sobol()["n_base_samples"] == 8
        refused = self._sobol(n_samples=1024)
        assert "n_samples=1024 does not match Y" in refused["error"]

    def test_morris_counts(self):
        assert self._morris(n_trajectories=4)["n_trajectories"] == 4
        assert self._morris()["n_trajectories"] == 4
        refused = self._morris(n_trajectories=20)
        assert "n_trajectories=20 does not match X" in refused["error"]

    @pytest.mark.parametrize("method", ["sobol_analyze", "morris_analyze",
                                        "sobol_analysis", "morris_analysis"])
    def test_every_key_the_tool_promises_is_in_the_result(self, method):
        from funhouse_agent.adapters.salib_adapter import METHOD_INFO
        if method == "sobol_analyze":
            r = self._sobol()
        elif method == "morris_analyze":
            r = self._morris()
        else:
            r = call_agent("salib", method, {
                "var_names": self.NAMES, "bounds": self.BOUNDS,
                "expression": "a + 2*b"})
        assert "error" not in r, r
        missing = [k for k in METHOD_INFO[method]["returns"] if k not in r]
        # S2 is returned only with second-order indices
        assert [k for k in missing if k != "S2"] == []

    def test_expression_limit_reaches_the_agent(self):
        r = call_agent("salib", "sobol_analysis", {
            "var_names": self.NAMES, "bounds": [[1, 2], [1, 2]],
            "expression": "floor(a + 8)**floor(a + 8)**floor(a + 8) + b",
            "n_samples": 8})
        assert "1e308" in r["error"]
