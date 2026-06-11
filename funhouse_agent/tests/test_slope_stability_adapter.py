"""le-modern P9 tests: slope_stability adapter — new methods + params.

Covers the modernized LE surface exposed through the funhouse adapter:
- METHOD_INFO/REGISTRY contract + allowed_values (gle/janbu, f_interslice,
  entry_exit / noncircular_de surface types, strength models)
- analyze_slope with the rigorous GLE engine (thrust line, interslice E/X)
- compare_methods_table (F&K-style one-surface/all-methods table)
- entry-exit critical-surface search
- reinforcement pass-through (nails / anchors / geosynthetics)
- SHANSEP and Hoek-Brown per-layer strength models, ponded water
- probabilistic: fosm_fos + monte_carlo_fos
- clear ValueError on bad enum values
"""

import pytest

from funhouse_agent.adapters.slope_stability import (
    METHOD_INFO, METHOD_REGISTRY,
)


# ----------------------------------------------------------------------------
# Shared geometry: 10-m-high 2H:1V slope, single drained layer, GWT below toe
# ----------------------------------------------------------------------------

def _base(**over):
    p = {
        "surface_points": [[0, 10], [20, 10], [40, 20], [70, 20]],
        "soil_layers": [{"top_elevation": 20, "bottom_elevation": -15,
                         "gamma": 18, "gamma_sat": 20,
                         "phi": 25, "c_prime": 5}],
        "gwt_points": [[0, 8], [70, 8]],
        "xc": 30.0, "yc": 32.0, "radius": 26.0,
    }
    p.update(over)
    return p


def _no_circle(p):
    return {k: v for k, v in p.items() if k not in ("xc", "yc", "radius")}


# ----------------------------------------------------------------------------
# METHOD_INFO contract
# ----------------------------------------------------------------------------

class TestMethodInfo:
    def test_keys_match_registry(self):
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())
        for name, info in METHOD_INFO.items():
            for field in ("category", "brief", "parameters", "returns"):
                assert field in info, f"{name} missing {field}"

    def test_expected_methods(self):
        assert set(METHOD_INFO.keys()) == {
            "analyze_slope", "search_critical_surface",
            "compare_methods_table", "fosm_fos", "monte_carlo_fos",
        }

    def test_method_allowed_values_modernized(self):
        av = METHOD_INFO["analyze_slope"]["parameters"]["method"]["allowed_values"]
        for m in ("fellenius", "bishop", "janbu", "spencer",
                  "morgenstern_price", "gle"):
            assert m in av

    def test_f_interslice_allowed_values(self):
        av = METHOD_INFO["analyze_slope"]["parameters"]["f_interslice"]["allowed_values"]
        assert set(av) == {"constant", "half_sine", "clipped_sine",
                           "trapezoidal"}

    def test_surface_type_allowed_values(self):
        av = METHOD_INFO["search_critical_surface"]["parameters"]["surface_type"]["allowed_values"]
        for s in ("circular", "entry_exit", "noncircular", "noncircular_de"):
            assert s in av

    def test_defaults_in_allowed_values(self):
        for m, mi in METHOD_INFO.items():
            for p, pi in mi["parameters"].items():
                if "allowed_values" in pi and pi.get("default") is not None:
                    assert pi["default"] in pi["allowed_values"], f"{m}.{p}"


class TestDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("slope_stability")
        names = [m for v in result.values() for m in v]
        for m in ("analyze_slope", "compare_methods_table", "fosm_fos",
                  "monte_carlo_fos"):
            assert m in names

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("slope_stability", "analyze_slope")
        assert "f_interslice" in info["parameters"]
        assert "nails" in info["parameters"]


# ----------------------------------------------------------------------------
# analyze_slope: rigorous methods + outputs
# ----------------------------------------------------------------------------

class TestAnalyzeSlope:
    def test_gle_with_slice_data_and_thrust(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(
            method="gle", include_slice_data=True))
        assert 1.5 < r["FOS"] < 2.5
        assert "thrust_line" in r
        row = r["slice_data"][len(r["slice_data"]) // 2]
        for key in ("N_eff_kN_per_m", "S_mob_kN_per_m", "U_base_kN_per_m",
                    "E_left_kN_per_m", "X_right_kN_per_m", "alpha_deg"):
            assert key in row

    def test_janbu_reports_corrected_and_uncorrected(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(method="janbu"))
        assert r["FOS"] == pytest.approx(r["FOS_janbu_corrected"])
        assert r["FOS_janbu_uncorrected"] < r["FOS_janbu_corrected"]
        assert r["janbu_f0"] > 1.0

    def test_constant_f_reproduces_spencer(self):
        r_sp = METHOD_REGISTRY["analyze_slope"](_base(method="spencer"))
        r_mp = METHOD_REGISTRY["analyze_slope"](_base(
            method="morgenstern_price", f_interslice="constant"))
        assert r_mp["FOS"] == pytest.approx(r_sp["FOS"], rel=1e-3)

    def test_compare_methods_table(self):
        t = METHOD_REGISTRY["compare_methods_table"](_base())
        names = [row["method"] for row in t["rows"]]
        assert "Fellenius (OMS)" in names
        assert "Spencer" in names
        assert any(n.startswith("Morgenstern-Price") for n in names)
        assert "Method comparison" in t["summary"]


# ----------------------------------------------------------------------------
# Search options
# ----------------------------------------------------------------------------

class TestSearch:
    def test_entry_exit_search(self):
        r = METHOD_REGISTRY["search_critical_surface"](_no_circle(_base(
            surface_type="entry_exit",
            x_entry_range=[2, 18], x_exit_range=[42, 68],
            nx=4, ny=4, method="bishop", n_slices=20)))
        assert r["critical"]["FOS"] > 0.5
        assert r["n_surfaces_evaluated"] > 0

    def test_invalid_surface_type(self):
        with pytest.raises(ValueError, match="surface_type 'spiral'"):
            METHOD_REGISTRY["search_critical_surface"](_no_circle(_base(
                surface_type="spiral")))


# ----------------------------------------------------------------------------
# Reinforcement pass-through
# ----------------------------------------------------------------------------

class TestReinforcement:
    def test_nail_increases_fos(self):
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="gle"))
        nailed = METHOD_REGISTRY["analyze_slope"](_base(
            method="gle",
            nails=[{"x_head": 30, "z_head": 15, "length": 25,
                    "bond_stress": 150, "spacing_h": 1.0}]))
        assert nailed["FOS"] > plain["FOS"]

    def test_anchor_and_geosynthetic_accepted(self):
        plain = METHOD_REGISTRY["analyze_slope"](_base(method="bishop"))
        r = METHOD_REGISTRY["analyze_slope"](_base(
            method="bishop",
            anchors=[{"x_head": 30, "z_head": 15, "length": 25,
                      "T_allow": 100}],
            geosynthetics=[{"elevation": 12.0, "T_allow": 50}]))
        assert r["FOS"] > plain["FOS"]

    def test_nail_missing_required_key(self):
        with pytest.raises(ValueError, match=r"nails\[\].*length"):
            METHOD_REGISTRY["analyze_slope"](_base(
                nails=[{"x_head": 30, "z_head": 15}]))


# ----------------------------------------------------------------------------
# Strength models + ponded water
# ----------------------------------------------------------------------------

class TestStrengthModels:
    def test_shansep_ocr_raises_fos(self):
        def run(ocr):
            return METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
                "top_elevation": 20, "bottom_elevation": -15,
                "gamma": 18, "gamma_sat": 20, "analysis_mode": "undrained",
                "strength_model": "shansep", "shansep_S": 0.25,
                "shansep_m": 0.8, "ocr": ocr}]))["FOS"]
        assert run(2.0) > run(1.0)

    def test_hoek_brown_runs(self):
        r = METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
            "top_elevation": 20, "bottom_elevation": -15, "gamma": 24,
            "strength_model": "hoek_brown", "hb_sigci": 5000,
            "hb_gsi": 45, "hb_mi": 10}]))
        assert r["FOS"] > 1.0

    def test_invalid_strength_model(self):
        with pytest.raises(ValueError, match="strength_model 'tresca'"):
            METHOD_REGISTRY["analyze_slope"](_base(soil_layers=[{
                "top_elevation": 20, "bottom_elevation": -15, "gamma": 18,
                "strength_model": "tresca"}]))

    def test_ponded_water_buttresses(self):
        dry_toe = METHOD_REGISTRY["analyze_slope"](_base())
        ponded = METHOD_REGISTRY["analyze_slope"](_base(
            gwt_points=[[0, 14], [70, 14]]))   # 4 m pond over the toe bench
        assert ponded["FOS"] != pytest.approx(dry_toe["FOS"], rel=1e-3)


# ----------------------------------------------------------------------------
# Probabilistic
# ----------------------------------------------------------------------------

class TestProbabilistic:
    def test_fosm(self):
        r = METHOD_REGISTRY["fosm_fos"](_base(
            method="bishop",
            variables={"phi": {"cov": 0.10}, "c_prime": {"cov": 0.30}}))
        assert r["FOS_mean_values"] > 1.5
        assert r["beta_lognormal"] > 0
        assert 0 <= r["pf_lognormal"] <= 1
        assert sum(r["variable_variance_pct"].values()) == pytest.approx(
            100.0, abs=0.5)

    def test_fosm_requires_variables(self):
        with pytest.raises(ValueError, match="variables"):
            METHOD_REGISTRY["fosm_fos"](_base(method="bishop"))

    def test_monte_carlo_seeded_reproducible(self):
        kw = _base(method="bishop", n=150, seed=42,
                   variables={"phi": {"cov": 0.10}})
        a = METHOD_REGISTRY["monte_carlo_fos"](kw)
        b = METHOD_REGISTRY["monte_carlo_fos"](kw)
        assert a["FOS_mean"] == b["FOS_mean"]
        assert a["n_realizations"] == 150
        assert "histogram_counts" in a


# ----------------------------------------------------------------------------
# Error clarity
# ----------------------------------------------------------------------------

class TestErrors:
    def test_invalid_method(self):
        with pytest.raises(ValueError, match=r"method 'sarma'.*Allowed"):
            METHOD_REGISTRY["analyze_slope"](_base(method="sarma"))

    def test_invalid_f_interslice(self):
        with pytest.raises(ValueError, match="f_interslice 'parabolic'"):
            METHOD_REGISTRY["analyze_slope"](_base(f_interslice="parabolic"))
