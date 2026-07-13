"""Cross-module SELECTION mis-routing redirects (v5.4 E11-routing).

The agent repeatedly guessed Rankine/Coulomb earth-pressure coefficients on the
WRONG module (bearing_capacity / seismic_geotech). These offline tests verify
the redirect returns a did-you-mean pointing at retaining_walls.
earth_pressure_coefficient WITHOUT executing a different module, that the same
name on the RIGHT module auto-routes to a real result, and that genuinely
unknown methods still get the plain "Unknown method" error.
"""

import pytest

from funhouse_agent.dispatch import call_agent, _cross_module_redirect


class TestWrongModuleRedirects:
    def test_bearing_capacity_rankine_redirects(self):
        r = call_agent("bearing_capacity", "rankine_coefficients", {"phi_deg": 30})
        assert "error" in r
        assert "retaining_walls" in r["error"]
        assert "earth_pressure_coefficient" in r["error"]
        # a did-you-mean, NOT an execution of a different module
        assert "K" not in r and "coefficient" not in r

    def test_seismic_rankine_earth_pressure_redirects(self):
        r = call_agent("seismic_geotech", "rankine_earth_pressure", {})
        assert "error" in r and "retaining_walls" in r["error"]

    def test_active_earth_pressure_variant_redirects(self):
        r = call_agent("bearing_capacity", "active_earth_pressure", {})
        assert "error" in r and "earth_pressure_coefficient" in r["error"]

    def test_names_the_wrong_and_right_module(self):
        r = call_agent("bearing_capacity", "rankine_ka", {})
        assert "bearing_capacity" in r["error"]      # what you called
        assert "retaining_walls" in r["error"]       # where it lives

    def test_seismic_caquot_redirects_to_earth_pressure(self):
        """Eval EPC-3: 'caquot'/'log_spiral' guessed on seismic_geotech (the
        agent used M-O with kh=0 for a static Caquot question) points at
        retaining_walls.earth_pressure_coefficient — not an execution."""
        for guess in ("caquot", "caquot_kerisel", "log_spiral", "passive_coefficient"):
            r = call_agent("seismic_geotech", guess, {"phi_deg": 35})
            assert "error" in r, guess
            assert "retaining_walls" in r["error"], guess
            assert "earth_pressure_coefficient" in r["error"], guess
            assert "K" not in r and "coefficient" not in r, guess

    def test_newmark_redirects_to_slope_stability(self):
        """Eval NMK-2: 'newmark'/'sliding_block' guessed on the wrong module
        points at slope_stability.newmark_displacement (the integrator), so the
        agent never concludes no Newmark integrator exists."""
        for guess in ("newmark", "newmark_analysis", "sliding_block"):
            r = call_agent("seismic_geotech", guess, {"ky": 0.2})
            assert "error" in r, guess
            assert "slope_stability" in r["error"], guess
            assert "newmark_displacement" in r["error"], guess


class TestRightModuleStillExecutes:
    def test_same_module_alias_executes(self):
        """The SAME guessed name on retaining_walls auto-routes to the real
        earth_pressure_coefficient and returns a coefficient (not an error)."""
        r = call_agent("retaining_walls", "rankine_coefficients", {"phi_deg": 30})
        assert "error" not in r
        assert "K" in r and "coefficient" in r

    def test_canonical_method_unaffected(self):
        r = call_agent("retaining_walls", "earth_pressure_coefficient",
                       {"phi_deg": 30, "state": "passive"})
        assert "error" not in r and "K" in r


class TestNoFalsePositives:
    def test_genuinely_unknown_is_generic(self):
        r = call_agent("bearing_capacity", "totally_bogus_xyz", {})
        assert "error" in r and "Unknown method" in r["error"]
        assert "retaining_walls" not in r["error"]

    def test_redirect_suppressed_when_target_not_visible(self):
        """If retaining_walls is out of scope, don't point at it — fall back to
        the generic unknown-method error."""
        r = call_agent("bearing_capacity", "rankine_coefficients", {},
                       allowed_agents=["bearing_capacity"])
        assert "error" in r and "Unknown method" in r["error"]


class TestRedirectHelper:
    def test_maps_guess_to_right_target(self):
        assert _cross_module_redirect("bearing_capacity", "rankine_ka") == (
            "retaining_walls", "earth_pressure_coefficient")

    def test_same_module_returns_none(self):
        # on the right module it is a same-module alias, not a cross redirect
        assert _cross_module_redirect(
            "retaining_walls", "rankine_coefficients") is None

    def test_unknown_name_returns_none(self):
        assert _cross_module_redirect("bearing_capacity", "some_method") is None
