"""Tool-surface usability fixes from the 2026-10 Foundry eval (106 questions,
GPT-5.4, package 5.31.0). Offline: no model, no network.

What the run showed, by kind: a ``list_methods`` category used as a topic
matched nothing and sent the agent guessing; guessed method names got no
pointer to the nearest real one; unknown parameter names got no did-you-mean;
``list_methods`` with no module was a schema error; and several validation
errors said what was wrong but not what to pass. None of the fixes changes a
calculated number — the alias tests below assert identical results.
"""

import json

import pytest

from funhouse_agent.adapters import (
    MODULE_REGISTRY, reject_unknown_params, suggest_params,
)
from funhouse_agent.dispatch import (
    ANALYSIS_MODULES, _closest_methods, _load_adapter, call_agent,
    describe_method, list_agents, list_methods,
)


# ---------------------------------------------------------------------------
# list_methods: a category that is not a category, and no module at all
# ---------------------------------------------------------------------------

class TestListMethodsCategory:
    def test_exact_category_still_filters(self):
        out = list_methods("settlement", category="consolidation")
        assert list(out) == ["Consolidation"]
        assert "note" not in out

    def test_topic_finds_the_methods_that_mention_it(self):
        out = list_methods("fem2d", category="slope")
        assert "error" not in out
        names = {m for ms in out["methods"].values() for m in ms}
        assert "fem2d_slope_srm" in names
        assert "fem2d_consolidation" not in names
        assert "is not a 'fem2d' category" in out["note"]
        assert out["available_categories"] == ["FEM 2D"]

    def test_topic_matches_case_insensitively(self):
        out = list_methods("subsurface", category="DIGGS")
        names = {m for ms in out["methods"].values() for m in ms}
        assert {"parse_diggs", "validate_diggs_schema"} <= names
        assert "read_ags4" not in names

    def test_topic_reaches_parameter_descriptions(self):
        # stabilizing piles are a parameter of the LE analyses, not a method
        out = list_methods("slope_stability", category="stabilizing pile")
        names = {m for ms in out["methods"].values() for m in ms}
        assert "analyze_slope" in names

    def test_nothing_matches_lists_everything_with_a_note(self):
        out = list_methods("worked_examples", category="settlement")
        assert "error" not in out
        names = {m for ms in out["methods"].values() for m in ms}
        assert names == {"find_worked_examples", "get_worked_example"}
        assert "showing ALL" in out["note"]
        assert "parameters" in out["note"]       # where a topic belongs

    def test_partial_category_name(self):
        out = list_methods("settlement", category="granular")
        assert list(out["methods"]) == ["Granular Settlement"]

    def test_no_module_lists_the_modules(self):
        out = list_methods("")
        assert "agent_name" in out["error"]
        assert "bearing_capacity" in out["modules"]

    def test_no_module_respects_scope(self):
        out = list_methods("", allowed_agents=ANALYSIS_MODULES)
        assert "dm7" not in out["modules"]
        assert "slope_stability" in out["modules"]


# ---------------------------------------------------------------------------
# unknown method names: the closest real methods, every path
# ---------------------------------------------------------------------------

class TestUnknownMethodNames:
    @pytest.mark.parametrize("agent, guess, expected", [
        ("lateral_pile", "composite_ei", "composite_section_ei"),
        ("downdrag", "downdrag_analyses", "downdrag_analysis"),
        ("slope_stability", "newmark_jibson", "newmark_jibson2007"),
        ("retaining_walls", "mse_wall_design", "mse_wall"),
    ])
    def test_closest_is_first(self, agent, guess, expected):
        assert _closest_methods(_load_adapter(agent), guess)[0] == expected

    def test_closest_never_empty(self):
        assert len(_closest_methods(_load_adapter("liquepy"), "zzz")) == 3

    def test_describe_unknown_names_closest_before_full_list(self):
        out = describe_method("lateral_pile", "composite_ei")
        assert out["closest"][0] == "composite_section_ei"
        assert "Closest" in out["error"]
        assert list(out).index("closest") < list(out).index("available")

    def test_describe_unknown_points_at_the_other_module(self):
        out = describe_method("retaining_walls", "apparent_earth_pressure")
        assert out["redirect"] == {"agent_name": "soe",
                                   "method": "apparent_pressure"}
        assert "Unknown method" in out["error"]

    def test_call_agent_unknown_always_says_did_you_mean(self):
        out = call_agent("lateral_pile", "composite_ei", {})
        assert "Did you mean: ['composite_section_ei'" in out["error"]

    def test_call_agent_cross_module_never_executes(self):
        out = call_agent("retaining_walls", "apparent_earth_pressure",
                         {"H": 6, "gamma": 18, "phi": 30})
        assert "error" in out and "'soe'" in out["error"]
        assert "apparent_pressure" in out["error"]

    @pytest.mark.parametrize("agent, guess, real", [
        ("axial_pile", "alpha_method", "axial_pile_capacity"),
        ("axial_pile", "driven_pile_capacity", "axial_pile_capacity"),
        ("drilled_shaft", "beta_method_capacity", "drilled_shaft_capacity"),
        ("retaining_walls", "cantilever_wall_analysis", "cantilever_wall"),
        ("downdrag", "analyze_downdrag", "downdrag_analysis"),
        ("liquepy", "spt_bi2014_triggering", "spt_liquefaction"),
        ("slope_stability", "infinite_slope_analysis", "infinite_slope_fos"),
    ])
    def test_obvious_synonyms_resolve(self, agent, guess, real):
        from funhouse_agent.dispatch import _resolve_unknown_method
        mod = _load_adapter(agent)
        assert _resolve_unknown_method(mod, agent, guess, {}) == (real, {})


# ---------------------------------------------------------------------------
# unknown parameter names: did-you-mean, and the obvious synonyms accepted
# ---------------------------------------------------------------------------

class TestParameterSuggestions:
    @pytest.mark.parametrize("unknown, valid, expected", [
        ("x_label", ["xlabel", "ylabel", "title"], ["xlabel"]),
        ("fc_mpa", ["b", "h", "fc", "fy"], ["fc"]),
        ("b_mm", ["b", "h", "fc"], ["b"]),
        ("water_table_depth", ["depth", "gwt_depth"], ["gwt_depth"]),
        ("unit_wieght", ["unit_weight", "width"], ["unit_weight"]),
    ])
    def test_suggests(self, unknown, valid, expected):
        assert suggest_params(unknown, valid) == expected

    def test_does_not_offer_a_different_quantity(self):
        # sharing a word is not a spelling slip: base vs load inclination
        assert suggest_params("base_inclination",
                              ["load_inclination", "base_tilt"]) == []

    def test_message_keeps_its_shape_and_adds_the_hint(self):
        with pytest.raises(ValueError) as exc:
            reject_unknown_params({"x_label": 1, "bogus": 2},
                                  ("xlabel", "ylabel"), method="plot_data")
        msg = str(exc.value)
        assert msg.startswith("plot_data: unknown parameter(s) ['bogus', "
                              "'x_label'].")
        assert "Did you mean: x_label -> xlabel?" in msg
        assert "Valid parameters: ['xlabel', 'ylabel']." in msg
        assert "describe_method" in msg


class TestParameterAliasesChangeNoNumbers:
    def test_plot_axis_labels(self, tmp_path):
        out = call_agent("profile_figure", "plot_data", {
            "series": [{"x": [0, 1, 2], "y": [0, 1, 4], "label": "s"}],
            "x_label": "x (m)", "y_label": "y (kPa)",
            "output_path": str(tmp_path / "p.png"), "interactive": False})
        assert "error" not in out, out

    def test_rc_section_unit_suffixed_names(self):
        base = {"b": 300, "h": 500, "fc": 30, "fy": 420, "n_bot": 3,
                "dia_bot": 20, "cover": 40}
        suffixed = {"b_mm": 300, "h_mm": 500, "fc_MPa": 30, "fy_mpa": 420,
                    "n_bot": 3, "dia_bot_mm": 20, "cover_mm": 40}
        a = call_agent("concrete_props", "rc_rectangular_section", base)
        b = call_agent("concrete_props", "rc_rectangular_section", suffixed)
        assert "error" not in a and a == b

    def test_bearing_water_table_and_base_inclination(self):
        base = {"width": 2, "unit_weight": 18, "friction_angle": 30,
                "depth": 1}
        a = call_agent("bearing_capacity", "bearing_capacity_analysis",
                       {**base, "gwt_depth": 1.5, "base_tilt": 5})
        b = call_agent("bearing_capacity", "bearing_capacity_analysis",
                       {**base, "water_table_depth": 1.5,
                        "base_inclination": 5})
        assert "error" not in a and a == b

    def test_combined_settlement_footing_names(self):
        a = call_agent("settlement", "combined_settlement_analysis",
                       {"q_applied": 100, "B": 2, "L": 3, "Es": 20000})
        b = call_agent("settlement", "combined_settlement_analysis",
                       {"foundation_pressure": 100, "foundation_width": 2,
                        "foundation_length": 3, "Es": 20000})
        assert "error" not in a and a == b

    def test_worked_examples_query_is_the_topic(self):
        a = call_agent("worked_examples", "find_worked_examples",
                       {"topic": "rapid drawdown dam"})
        b = call_agent("worked_examples", "find_worked_examples",
                       {"query": "rapid drawdown dam"})
        assert "error" not in a and a == b


# ---------------------------------------------------------------------------
# validation errors that now say what to pass
# ---------------------------------------------------------------------------

class TestValidationSaysWhatToPass:
    def test_hansen_is_not_a_factor_method(self):
        out = call_agent("bearing_capacity", "bearing_capacity_analysis",
                         {"width": 2, "unit_weight": 18, "friction_angle": 30,
                          "factor_method": "hansen"})
        assert "ngamma_method='hansen'" in out["error"]
        info = _load_adapter("bearing_capacity").METHOD_INFO
        allowed = info["bearing_capacity_analysis"]["parameters"][
            "factor_method"]["allowed_values"]
        assert "hansen" not in allowed      # it was advertised and always failed

    def test_hansen_ngamma_still_runs(self):
        out = call_agent("bearing_capacity", "bearing_capacity_analysis",
                         {"width": 2, "unit_weight": 18, "friction_angle": 30,
                          "ngamma_method": "hansen"})
        assert "error" not in out

    def test_pile_group_6dof_names_the_stiffness(self):
        out = call_agent("pile_group", "pile_group_6dof",
                         {"n_rows": 2, "n_cols": 3, "spacing": 1.2,
                          "Vz": 4000, "Vx": 400})
        assert "axial_stiffness (kN/m)" in out["error"]
        assert "pile_group_simple" in out["error"]
        params = _load_adapter("pile_group").METHOD_INFO[
            "pile_group_6dof"]["parameters"]
        assert "axial_stiffness" in params and "lateral_stiffness" in params

    def test_pile_group_6dof_with_stiffness_runs(self):
        out = call_agent("pile_group", "pile_group_6dof",
                         {"n_rows": 2, "n_cols": 3, "spacing": 1.2,
                          "Vz": 4000, "Vx": 400, "axial_stiffness": 2e5,
                          "lateral_stiffness": 2e4})
        assert "error" not in out

    def test_downdrag_layer_error_names_the_layer_and_the_fix(self):
        out = call_agent("downdrag", "downdrag_analysis", {
            "pile_length": 20, "pile_diameter": 0.5,
            "layers": [{"thickness": 5, "soil_type": "cohesionless",
                        "unit_weight": 18, "phi": 30, "settling": True,
                        "description": "fill"}]})
        assert "layers[0] (fill)" in out["error"]
        assert "E_s (elastic modulus, kPa" in out["error"]
        doc = _load_adapter("downdrag").METHOD_INFO["downdrag_analysis"][
            "parameters"]["layers"]["description"]
        assert "E_s" in doc

    def test_signal_processing_says_what_to_give(self, monkeypatch):
        from funhouse_agent.adapters import seismic_signals_adapter as ss
        monkeypatch.setattr(ss, "_check_eqsig", lambda: None)
        with pytest.raises(ValueError, match=r"bandpass=\[f_low, f_high\]"):
            ss._run_signal_processing({"motion": "el_centro"})

    def test_find_worked_examples_lists_valid(self):
        out = call_agent("worked_examples", "find_worked_examples",
                         {"domain": "slope_stability"})
        assert "Valid parameters: ['domain', 'limit', 'topic']" in out["error"]

    def test_liquefaction_missing_input_says_what_it_is(self):
        out = call_agent("liquefaction", "liquefaction_analysis", {
            "depth": [3], "N160": [10], "gamma": [18], "gwt_depth": 1,
            "amax_g": 0.3, "method": "nceer2001"})
        assert "FC (Fines content (%)" in out["error"]

    def test_unknown_reinforcement_name_is_named(self):
        from funhouse_agent.adapters.retaining_walls import (
            _build_reinforcement)
        with pytest.raises(ValueError, match="'geogrid' is not a built-in"):
            _build_reinforcement({"reinforcement_name": "geogrid"})


# ---------------------------------------------------------------------------
# module briefs name the capabilities agents looked for elsewhere
# ---------------------------------------------------------------------------

class TestCatalogNamesItsCapabilities:
    def test_slope_stability_brief(self):
        brief = MODULE_REGISTRY["slope_stability"]["brief"]
        for word in ("Newmark", "rapid drawdown", "infinite slope"):
            assert word in brief

    def test_pavement_brief_names_ufc(self):
        assert "UFC 3-250-01" in MODULE_REGISTRY["pavement_design"]["brief"]

    def test_lateral_pile_brief_names_composite_ei(self):
        assert "composite" in MODULE_REGISTRY["lateral_pile"]["brief"]

    def test_section_tools_point_at_composite_pile_ei(self):
        for agent, method in (("concrete_props", "rc_rectangular_section"),
                              ("section_props", "section_properties")):
            brief = _load_adapter(agent).METHOD_INFO[method]["brief"]
            assert "lateral_pile.composite_section_ei" in brief

    def test_catalog_still_fits(self):
        assert len(json.dumps(list_agents())) < 8000
