"""BACKLOG GM-115: a result says what it chose. Dispatch adds
``defaults_used`` (the declared parameters a call left out, with the default
each took); the module results carry ``choices``."""

import pytest

from funhouse_agent import dispatch as D

_PILE = {"pile_type": "pipe_closed", "diameter": 0.6, "wall_thickness": 0.012,
         "pile_length": 15,
         "layers": [{"thickness": 10, "soil_type": "cohesionless",
                     "unit_weight": 18, "friction_angle": 32},
                    {"thickness": 10, "soil_type": "cohesive",
                     "unit_weight": 17, "cohesion": 60}]}


def test_defaults_used_lists_what_the_call_left_out():
    r = D.call_agent("axial_pile", "axial_pile_capacity", dict(_PILE))
    # cohesive_phi (beta only) and the uplift fraction (uplift only) are
    # not used by this run, so not listed (GM-119); the clay layer's Fig.
    # 7-17 curve is (GM-121), and the sand-toe limit is not: the toe is in
    # clay
    assert r["defaults_used"] == {
        "head_depth": 0.0, "factor_of_safety": 2.5, "method": "auto",
        "include_uplift": False, "adhesion_curve": "auto"}
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**_PILE, "factor_of_safety": 3.0, "method": "auto"})
    assert "factor_of_safety" not in r["defaults_used"]
    assert "method" not in r["defaults_used"]


def test_a_parameter_given_by_another_spelling_counts_as_given():
    info = {"parameters": {"stickup": {"default": 0.0},
                           "Vt": {"default": 0.0}, "n": {"default": None},
                           "x": {"type": "float"}},
            "parameter_aliases": {"free_length": "stickup"}}

    class Mod:
        METHOD_INFO = {"m": info}
    assert D._defaults_used(Mod, "m", {"free_length"}) == {"Vt": 0.0}
    assert D._defaults_used(Mod, "nope", set()) == {}


def test_empty_defaults_and_package_headers_are_left_out():
    info = {"parameters": {"project_name": {"default": "Project"},
                           "engineer": {"default": ""},
                           "format": {"default": "html"},
                           "label": {"default": ""},
                           "FOS_required": {"default": 1.5}}}

    from types import ModuleType
    Package = ModuleType("funhouse_agent.adapters.calc_package")
    Package.METHOD_INFO = {"p": info}
    Other = ModuleType("funhouse_agent.adapters.other")
    Other.METHOD_INFO = {"p": info}
    assert D._defaults_used(Package, "p", set()) == {"FOS_required": 1.5}
    # outside the calc packages "format" may be a real choice: kept
    assert D._defaults_used(Other, "p", set()) == {
        "project_name": "Project", "format": "html", "FOS_required": 1.5}
    from funhouse_agent.adapters import calc_package
    for method, info in calc_package.METHOD_INFO.items():
        used = D._defaults_used(calc_package, method, set())
        assert not set(used) & D._PACKAGE_HEADER_PARAMS, method
        assert "" not in used.values(), method


def test_mo_without_a_solution_warns_in_the_result():
    r = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                     {"phi": 20, "kh": 0.5, "kv": 0.0, "gamma": 19, "H": 6})
    (w,) = r["warnings"]
    assert "no solution" in w and "trial wedge" in w
    assert r["choices"][0]["chosen"].startswith("Ka + 0.75 kh")
    kv = {c["what"]: c for c in r["choices"]}["kv"]
    assert kv["how"] == "given"          # typed, though it is 0
    ok = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                      {"phi": 32, "kh": 0.15})
    assert "warnings" not in ok


def test_errors_and_other_results_are_left_alone():
    class Mod:
        METHOD_INFO = {"m": {"parameters": {"a": {"default": 1}}}}
    assert D._with_defaults_used(Mod, "m", set(), {"error": "x"}) == {
        "error": "x"}
    assert D._with_defaults_used(Mod, "m", set(), [1, 2]) == [1, 2]
    assert D._with_defaults_used(Mod, "m", set(), "text") == "text"
    assert D._with_defaults_used(Mod, "m", {"a"}, {"v": 1}) == {"v": 1}
    assert D._with_defaults_used(Mod, "m", set(), {"v": 1}) == {
        "v": 1, "defaults_used": {"a": 1}}
    bad = D.call_agent("axial_pile", "axial_pile_capacity", {"pile_length": 1})
    assert "error" in bad and "defaults_used" not in bad


def test_axial_choices_reach_the_agent_with_the_tool_default_named():
    r = D.call_agent("axial_pile", "axial_pile_capacity", dict(_PILE))
    by = {c["what"]: c for c in r["choices"]}
    assert by["side resistance method"]["chosen"] == \
        "Nordlund 0-10 m; Tomlinson alpha 10-15 m"
    # a layer with no delta_phi_ratio takes GEC-12 Fig. 7-9 (GM-121)
    assert by["Nordlund delta/phi"] == {**by["Nordlund delta/phi"],
        "chosen": "0.80 (Fig. 7-9 curve a, pipe piles)", "how": "module"}
    given = [dict(_PILE["layers"][0], delta_phi_ratio=0.8),
             _PILE["layers"][1]]
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**_PILE, "layers": given})
    by = {c["what"]: c for c in r["choices"]}
    assert by["Nordlund delta/phi"] == {**by["Nordlund delta/phi"],
                                        "chosen": "0.8", "how": "given"}


_SPT = {"depth": [3, 5, 7], "N160": [10, 15, 20], "FC": [10, 10, 25],
        "gamma": [18, 19, 19], "amax_g": 0.3, "gwt_depth": 2.0, "m_w": 7.0}


@pytest.mark.parametrize("method, how, other", [
    (None, "default", "nceer2001"), ("bi2014", "given", "nceer2001"),
    ("nceer2001", "given", "bi2014")])
def test_liquefaction_says_which_procedure_and_whether_it_was_named(
        method, how, other):
    from liquepy_agent import has_liquepy
    if method != "nceer2001" and not has_liquepy():
        pytest.skip("liquepy not installed")
    params = dict(_SPT, **({"method": method} if method else {}))
    r = D.call_agent("liquefaction", "liquefaction_analysis", params)
    first = r["choices"][0]
    assert first["what"] == "triggering procedure" and first["how"] == how
    assert other in first["alternatives"][0]
    assert all("rd" in row and "MSF" in row for row in r["layer_results"])


def test_seismic_earth_pressure_and_consolidation_report_their_choices():
    r = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                     {"phi": 32, "kh": 0.15, "gamma": 19, "H": 6})
    assert [c["what"] for c in r["choices"]] == [
        "KAE", "kh", "kv", "wall friction delta", "static KA",
        "height of delta-PAE"]
    r = D.call_agent("settlement", "consolidation_settlement", {
        "layers": [{"thickness": 4.0, "depth_to_center": 4.0, "e0": 0.9,
                    "Cc": 0.3, "Cr": 0.05, "sigma_v0": 60.0}],
        "q_net": 100, "B": 2})
    assert r["choices"][0]["what"] == "stress increase with depth"
    given = D.call_agent("settlement", "consolidation_settlement", {
        "layers": [{"thickness": 4.0, "depth_to_center": 4.0, "e0": 0.9,
                    "Cc": 0.3, "Cr": 0.05, "sigma_v0": 60.0}],
        "delta_sigma": 50})
    assert "choices" not in given      # the stress was given


# ---------------------------------------------------------------------------
# GM-118: a typed value equal to the default is "given"
# ---------------------------------------------------------------------------

_CLAY = [{"thickness": 4.0, "depth_to_center": 4.0, "e0": 0.9, "Cc": 0.3,
          "Cr": 0.05, "sigma_v0": 60.0}]


def test_a_typed_value_equal_to_the_default_is_given():
    """The stress method typed as '2:1' (the default) is 'given', and not
    in defaults_used; left out, it is 'default' and listed."""
    typed = D.call_agent("settlement", "consolidation_settlement", {
        "layers": _CLAY, "q_net": 100, "B": 2, "stress_method": "2:1"})
    # (the footing length L = B is the other choice, GM-142)
    c = {x["what"]: x for x in typed["choices"]}["stress increase with depth"]
    assert c["how"] == "given"
    assert "stress_method" not in typed.get("defaults_used", {})
    left = D.call_agent("settlement", "consolidation_settlement", {
        "layers": _CLAY, "q_net": 100, "B": 2})
    assert left["choices"][0]["how"] == "default"
    assert left["defaults_used"]["stress_method"] == "2:1"


_BETA_CHOICE = "phi' of cohesive layers given no friction angle (beta method)"


def test_axial_defaults_typed_at_their_values_are_given():
    r = D.call_agent("axial_pile", "axial_pile_capacity", {
        **_PILE, "method": "beta", "include_uplift": True,
        "cohesive_phi": 25.0, "uplift_skin_fraction": 0.75})
    by = {c["what"]: c for c in r["choices"]}
    assert by[_BETA_CHOICE]["how"] == "given"
    assert by["uplift capacity"]["how"] == "given"
    assert not {"cohesive_phi", "uplift_skin_fraction"} & set(
        r["defaults_used"])
    left = D.call_agent("axial_pile", "axial_pile_capacity", {
        **_PILE, "method": "beta", "include_uplift": True})
    by = {c["what"]: c for c in left["choices"]}
    assert by[_BETA_CHOICE]["how"] == "default"
    assert by["uplift capacity"]["how"] == "default"
    assert {"cohesive_phi", "uplift_skin_fraction"} <= set(
        left["defaults_used"])


# ---------------------------------------------------------------------------
# GM-119: defaults_used lists only the defaults the run used
# ---------------------------------------------------------------------------

def _used(agent, method, params):
    mod = D._load_adapter(agent)
    return set(D._defaults_used(mod, method, set(params), params))


def test_an_spt_run_lists_no_cpt_settings():
    from funhouse_agent.adapters.liquefaction_adapter import METHOD_INFO
    cpt_only = {"a_ratio", "i_c_limit", "cfc", "unit_wt_method",
                "gamma_predrill", "s_g", "p_a"}
    assert cpt_only <= set(METHOD_INFO["liquefaction_analysis"]["parameters"])
    spt = _used("liquefaction", "liquefaction_analysis", _SPT)
    assert not spt & cpt_only
    assert "c_0" in spt and "gamma_w" not in spt         # B&I 2014 SPT
    nceer = _used("liquefaction", "liquefaction_analysis",
                  {**_SPT, "method": "nceer2001"})
    assert "gamma_w" in nceer and "c_0" not in nceer
    cpt = _used("liquefaction", "liquefaction_analysis",
                {"depth": [1], "q_c": [5000], "f_s": [50], "gwl": 1.5,
                 "pga": 0.3, "m_w": 7.5})
    assert cpt_only <= cpt and not {"c_0", "gamma_w"} & cpt


def test_the_seismic_package_lists_only_its_own_analysis():
    mo = _used("calc_package", "seismic_package",
               {"analysis_type": "seismic_earth_pressure", "phi": 30,
                "kh": 0.15})
    # KA has no declared default: the run works it out from phi
    # (GM-120), so it is not a default the call left in force
    assert "kv" in mo and "KA" not in mo and "magnitude" not in mo
    liq = _used("calc_package", "seismic_package",
                {"analysis_type": "liquefaction", "layers": []})
    assert "magnitude" in liq and not {"KA", "kv"} & liq


def test_a_pipe_pile_lists_no_h_pile_or_filled_pipe_settings():
    params = {"pile_type": "pipe", "pile_diameter": 0.6, "pile_length": 15,
              "Vt": 50, "layers": []}
    pipe = _used("lateral_pile", "lateral_pile_analysis", params)
    assert not {"axis", "fc"} & pipe
    assert "axis" in _used("lateral_pile", "lateral_pile_analysis",
                           {**params, "pile_type": "h_pile"})
    assert "fc" in _used("lateral_pile", "lateral_pile_analysis",
                         {**params, "pile_type": "filled_pipe"})


@pytest.mark.parametrize("surface_type, used, unused", [
    ("circular", {"nx", "ny"}, {"n_trials", "n_points", "bishop_screen"}),
    ("noncircular_de", {"n_points"}, {"nx", "ny", "n_trials",
                                      "bishop_screen"}),
    ("weak_layer", {"n_trials", "n_points"}, {"nx", "ny"}),
])
def test_a_slope_search_lists_only_its_own_settings(surface_type, used,
                                                     unused):
    base = {"surface_points": [], "soil_layers": []}
    got = _used("slope_stability", "search_critical_surface",
                {**base, "surface_type": surface_type})
    assert used <= got and not unused & got
    assert not {"tension_crack_water_depth", "tension_crack_side",
                "tension_crack_model"} & got           # no crack
    rigorous = _used("slope_stability", "search_critical_surface",
                     {**base, "method": "spencer"})
    assert "bishop_screen" in rigorous


def test_a_default_a_module_reported_is_listed_whatever_its_condition():
    """choices and defaults_used never contradict: a name how_given called
    'default' is in defaults_used even where applies_when says no."""
    class Mod:
        METHOD_INFO = {"m": {"parameters": {
            "a": {"default": 1, "applies_when": False},
            "b": {"default": 2, "applies_when": False}}}}
    assert D._defaults_used(Mod, "m", set(), {}) == {}
    assert D._defaults_used(Mod, "m", set(), {}, reported={"a"}) == {"a": 1}


def test_applies_reads_values_set_unset_and_any_of():
    A = D.applies
    assert A(None, {}) and A({}, {}) and not A(False, {})
    assert A({"m": ["Beta"]}, {"m": "beta"})              # no case
    assert not A({"m": ["beta"]}, {"m": "auto"})
    assert A({"t": [None, "spt"]}, {})                    # not given
    assert A({"q": "set"}, {"q": [1]}) and not A({"q": "set"}, {"q": []})
    assert not A({"q": "set"}, {"q": 0}) and A({"q": "unset"}, {})
    either = [{"m": ["gle"]}, {"c": [True]}]
    assert A(either, {"c": True}) and not A(either, {"m": "bishop"})
    assert A({"m": ["x"], "q": "set"}, {"m": "x", "q": 1})
    assert not A({"m": ["x"], "q": "set"}, {"m": "x"})


def test_describe_method_leaves_applies_when_out():
    doc = D.describe_method("liquefaction", "liquefaction_analysis")
    assert not any("applies_when" in spec
                   for spec in doc["parameters"].values())
    from funhouse_agent.adapters.liquefaction_adapter import METHOD_INFO
    assert "applies_when" in METHOD_INFO["liquefaction_analysis"][
        "parameters"]["a_ratio"]


def test_a_condition_naming_no_parameter_fails_on_import():
    from funhouse_agent.adapters import declare_applies_when
    info = {"m": {"parameters": {"a": {"default": 1}, "b": {}}}}
    with pytest.raises(KeyError, match="no parameter 'z'"):
        declare_applies_when(info, {"m": {"z": False}})
    with pytest.raises(KeyError, match="applies_when names no parameter"):
        declare_applies_when(info, {"m": {"a": [{"b": "set"}, {"y": "set"}]}})
    declare_applies_when(info, {"m": {"a": {"b": "set"}}})
    assert info["m"]["parameters"]["a"]["applies_when"] == {"b": "set"}


_TAGGED = ("axial_pile", "lateral_pile", "settlement", "liquefaction",
           "calc_package", "slope_stability")


def test_every_declared_condition_names_declared_parameters():
    """Every applies_when names only its method's parameters, with values
    the conditions understand."""
    seen = 0
    for agent in _TAGGED:
        mod = D._load_adapter(agent)
        for method, info in mod.METHOD_INFO.items():
            params = info.get("parameters") or {}
            for name, spec in params.items():
                cond = spec.get("applies_when")
                if cond is None or cond is False:
                    continue
                seen += 1
                for c in (cond if isinstance(cond, list) else [cond]):
                    for other, want in c.items():
                        assert other in params, (agent, method, name, other)
                        assert want in ("set", "unset") or isinstance(
                            want, list), (agent, method, name, other)
    assert seen > 40


_SEARCHES = ("search_critical_surface", "search_rapid_drawdown",
             "monte_carlo_fos")


def _agree_on_worked_examples(monkeypatch, searches: bool) -> int:
    """Run the worked examples' calls (the searches, or everything else)
    and check each: every input a module called "default" is in
    defaults_used, and no input the call named is. Returns the calls run."""
    from contextlib import contextmanager
    import geotech_common.choices as gc
    from funhouse_agent import worked_examples as we
    records = []
    real = gc.call_inputs

    @contextmanager
    def spy(given, defaults=()):
        with real(given, defaults) as inputs:
            records.append(inputs)
            yield inputs
    monkeypatch.setattr(gc, "call_inputs", spy)
    n = reported = 0
    for entry in we.load_examples():
        for call in entry["dispatch_calls"]:
            if (call["method"] in _SEARCHES) != searches:
                continue
            records.clear()
            r = D.call_agent(call["module"], call["method"],
                             dict(call["params"]))
            if not isinstance(r, dict) or "error" in r:
                continue
            used = set(r.get("defaults_used") or {})
            (inputs,) = records
            assert inputs.reported <= used, entry["id"]
            assert not inputs.given & used, entry["id"]
            reported += len(inputs.reported)
            n += 1
    # the searches name their method, kind, budget and m_alpha rule
    assert reported or not searches, "no module answered by name"
    return n


def test_choices_and_defaults_used_agree_on_the_worked_examples(
        monkeypatch):
    """GM-118/119 on real calls, the searches apart (slow)."""
    assert _agree_on_worked_examples(monkeypatch, searches=False) >= 30


@pytest.mark.slow
def test_choices_and_defaults_used_agree_on_the_worked_example_searches(
        monkeypatch):
    assert _agree_on_worked_examples(monkeypatch, searches=True) >= 5


def test_axial_gec12_limits_are_listed_where_the_soil_uses_them():
    """GM-121: the toe-stress limit, the open-pipe toe limit, the Fig. 7-17
    settings and the beta method's clay Nt depend on the soil, so they are
    listed when the result's choices report them as defaults."""
    sand = [{"thickness": 30, "soil_type": "cohesionless",
             "unit_weight": 19, "friction_angle": 36}]
    clay = [{"thickness": 30, "soil_type": "cohesive",
             "unit_weight": 18, "cohesion": 4.0 * 47.88}]
    pipe = {"pile_type": "pipe_open", "diameter": 0.6,
            "wall_thickness": 0.012, "pile_length": 20}
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**pipe, "layers": sand})
    used = set(r["defaults_used"])
    assert {"toe_stress_limit", "open_pipe_toe_limit",
            "inside_friction_ratio", "open_pipe_method"} <= used
    assert not {"adhesion_curve", "adhesion_beyond_chart",
                "cohesive_Nt", "h_pile_toe_area"} & used
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**pipe, "layers": clay, "toe_stress_limit": None})
    used = set(r["defaults_used"])
    assert {"adhesion_curve", "adhesion_beyond_chart"} <= used
    assert not {"toe_stress_limit", "open_pipe_toe_limit"} & used
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**pipe, "layers": clay, "method": "beta"})
    assert "cohesive_Nt" in r["defaults_used"]
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**pipe, "layers": clay, "method": "beta",
                      "cohesive_Nt": 9})
    assert "cohesive_Nt" not in r["defaults_used"]
    by = {c["what"]: c for c in r["choices"]}
    assert by["Nt of the cohesive toe layer (beta method)"]["how"] == "given"


def test_axial_gec12_options_reach_the_module():
    """Each GM-121 option the tool declares changes the run it is given to."""
    sand = [{"thickness": 30, "soil_type": "cohesionless",
             "unit_weight": 19, "friction_angle": 38}]
    pipe = {"pile_type": "pipe_open", "diameter": 0.6,
            "wall_thickness": 0.012, "pile_length": 20, "layers": sand}
    closed = {**pipe, "pile_type": "pipe_closed"}
    for call, name, value in ((closed, "toe_stress_limit", None),
                              (pipe, "inside_friction_ratio", 0.5),
                              (pipe, "open_pipe_toe_limit", None),
                              (pipe, "displaced_volume", 0.2)):
        base = D.call_agent("axial_pile", "axial_pile_capacity", dict(call))
        r = D.call_agent("axial_pile", "axial_pile_capacity",
                         {**call, name: value})
        assert "error" not in r, (name, r)

        def cases(res):   # the plugged and unplugged totals, if any
            return [c["basis"] for c in res["choices"]
                    if c["what"] == "open-ended pipe"]
        assert (r["Q_ultimate_kN"], cases(r)) != (base["Q_ultimate_kN"],
                                                  cases(base)), name
    h = D.call_agent("axial_pile", "axial_pile_capacity", {
        "pile_type": "h_pile", "designation": "HP12x74", "pile_length": 20,
        "layers": sand, "h_pile_toe_area": "steel"})
    assert h["Q_tip_kN"] < D.call_agent("axial_pile", "axial_pile_capacity", {
        "pile_type": "h_pile", "designation": "HP12x74", "pile_length": 20,
        "layers": sand})["Q_tip_kN"]
    stiff = {**closed, "layers": [{"thickness": 30, "soil_type": "cohesive",
                                   "unit_weight": 19,
                                   "cohesion": 4.0 * 47.88}]}
    end = D.call_agent("axial_pile", "axial_pile_capacity", dict(stiff))
    api = D.call_agent("axial_pile", "axial_pile_capacity",
                       {**stiff, "adhesion_beyond_chart": "api"})
    assert api["Q_skin_kN"] > end["Q_skin_kN"]
    bad = D.call_agent("axial_pile", "axial_pile_capacity",
                       {**pipe, "open_pipe_method": "nope"})
    assert "error" in bad
