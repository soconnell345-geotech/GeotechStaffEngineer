"""BACKLOG GM-115: a result says what it chose. Dispatch adds
``defaults_used`` (the declared parameters a call left out, with the default
each took); the module results carry ``choices``."""

import copy
import math
import random
import re

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.tests import method_probes as MP

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


def test_the_weak_rock_curve_uses_rqd_and_lists_it():
    """GM-159: alpha_r follows RQD (GEC 9 p. 231), so the weak-rock probe
    lists RQD, k_rm and the rock surface as defaults it used; gamma_r is
    still not used (the criterion has no unit weight), and other models
    list none of them."""
    probe = MP.PROBES["lateral_pile|get_py_curve#WeakRock"]["params"]
    r = D.call_agent("lateral_pile", "get_py_curve", dict(probe))
    assert {"RQD": 100.0, "krm": 0.0005, "rock_top": 0.0}.items() <= \
        r["defaults_used"].items()
    assert "gamma_r" not in r["defaults_used"]
    fractured = D.call_agent("lateral_pile", "get_py_curve",
                             {**probe, "RQD": 0})
    assert max(fractured["p_kN_per_m"]) == pytest.approx(
        3 * max(r["p_kN_per_m"]))
    sand = _used("lateral_pile", "get_py_curve",
                 MP.PROBES["lateral_pile|get_py_curve#SandAPI"]["params"])
    assert not {"RQD", "krm", "rock_top", "gamma_r"} & sand


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


# ---------------------------------------------------------------------------
# GM-158 / GM-150: METHOD_INFO and defaults_used say what each method does.
# Every analysis-side method runs on its probe (method_probes.py); the
# defaults the run lists are then given back by name, and nothing may move.
# A default that depends on another input is declared with
# default_depends_on and listed at the value the call used.

#: Result keys that differ between two identical runs (time, files) or
#: only in how an input was given ("given" or "default").
_VOLATILE = frozenset({
    "defaults_used", "output_note", "output_path", "file_size_bytes",
    "html_length", "elapsed_s", "runtime_s", "seconds_per_evaluation",
    "dxf_base64", "notes", "warnings", "note"})
_STAMP = re.compile(r"\d{8}_\d{6}(_\d+)?")


def _analysis_methods() -> set:
    out = set()
    for agent in sorted(D.MODULE_REGISTRY):
        if agent in D.REFERENCE_MODULES:
            continue
        mod = D._load_adapter(agent)
        out |= {f"{agent}|{m}" for m in mod.METHOD_REGISTRY}
    return out


def _run(agent: str, method: str, params: dict):
    """One call with the random generators reset, so a sampling method
    answers the same twice."""
    import numpy as np
    random.seed(0)
    np.random.seed(0)
    return D.call_agent(agent, method, copy.deepcopy(params))


def _norm(x, where: str):
    if isinstance(x, str):
        return _STAMP.sub("<t>", x.replace(where, "<cwd>"))
    if isinstance(x, dict):
        return {k: _norm(v, where) for k, v in x.items()
                if k not in _VOLATILE}
    if isinstance(x, list):
        return [_norm({k: w for k, w in v.items() if k != "how"}
                      if isinstance(v, dict) and "what" in v and "how" in v
                      else v, where) for v in x]
    return x


def _diff(a, b, path=""):
    """Where ``a`` and ``b`` first differ, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k} in one result only"
            d = _diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} items != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if (math.isnan(a) and math.isnan(b)) or math.isclose(
                a, b, rel_tol=1e-9, abs_tol=1e-12):
            return None
    return None if a == b else f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"


def _same(echo, value) -> bool:
    if isinstance(echo, str) and isinstance(value, str):
        return echo.strip().lower() == value.strip().lower()
    return _diff(echo, value) is None


def test_every_method_has_a_probe_or_a_reason():
    methods = _analysis_methods()
    probed = {key.split("#")[0] for key in MP.PROBES}
    assert sorted(methods - probed - set(MP.EXCLUDED)) == []
    assert sorted(probed - methods) == []
    assert sorted(set(MP.EXCLUDED) & probed) == []
    assert sorted(set(MP.EXCLUDED) - methods) == []


def test_a_probe_names_only_declared_parameters_and_every_required_one():
    """Each probe runs, so every parameter a method declares required is
    in it; and every name in it is one the method declares."""
    problems = []
    for key, probe in MP.PROBES.items():
        agent, method = key.split("#")[0].split("|")
        mod = D._load_adapter(agent)
        info = mod.METHOD_INFO[method]
        declared = info.get("parameters") or {}
        aliases = info.get("parameter_aliases") or {}
        of = info.get("parameters_of")   # render_figures: a package's too
        if of:
            other = mod.METHOD_INFO[probe["params"][of]]
            declared = {**(other.get("parameters") or {}), **declared}
            aliases = {**(other.get("parameter_aliases") or {}), **aliases}
        given = {aliases.get(k, k) for k in probe["params"]}
        problems += [(key, "not declared", k) for k in probe["params"]
                     if k not in declared and k not in aliases]
        problems += [(key, "required but not needed", name)
                     for name, spec in declared.items()
                     if spec.get("required") and name not in given]
    assert problems == []


def test_every_parameter_says_whether_it_is_required():
    problems = []
    for key in sorted(_analysis_methods()):
        agent, method = key.split("|")
        info = D._load_adapter(agent).METHOD_INFO[method]
        for name, spec in (info.get("parameters") or {}).items():
            if not isinstance(spec.get("required"), bool):
                problems.append((agent, method, name))
    assert problems == []


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=pytest.mark.slow) if k in MP.SLOW else k
    for k in sorted(MP.PROBES)])
def test_the_defaults_a_run_lists_are_the_values_it_used(key, tmp_path,
                                                         monkeypatch):
    """GM-158: give a run its own defaults_used by name and the result
    does not move; where the result echoes a listed default, it echoes
    the same value."""
    monkeypatch.chdir(tmp_path)
    base = key.split("#")[0]
    agent, method = base.split("|")
    probe = MP.PROBES[key]
    if probe.get("needs"):
        pytest.importorskip(probe["needs"])
    for before in probe.get("before", ()):
        D.call_agent(before[0], before[1], copy.deepcopy(before[2]))
    params = probe["params"]
    r0 = _run(agent, method, params)
    assert isinstance(r0, dict) and "error" not in r0, (key, r0)
    used = r0.get("defaults_used") or {}
    for name, value in used.items():
        if (base, name) in MP.KNOWN or (base, name) in \
                MP.ECHO_IS_NOT_THE_INPUT:
            continue
        echo = r0.get(name)
        if name in r0 and not isinstance(echo, (dict, list)):
            assert _same(echo, value), (
                f"{key}: defaults_used says {name}={value!r}, the result "
                f"says {echo!r}")
    if not used:
        return
    r1 = _run(agent, method, {**params, **used})
    assert "error" not in r1, (key, r1["error"])
    where = str(tmp_path)
    moved = _diff(_norm(r0, where), _norm(r1, where))
    assert moved is None, f"{key}: giving defaults_used back moved {moved}"


def _other_value(spec: dict, value):
    """A value other than ``value`` for an input declared by ``spec``: the
    next of its allowed values, the other boolean, a number moved by half
    (by 1 for an integer), or a string with a suffix; None where none can
    be made (a list, a dict, None)."""
    allowed = spec.get("allowed_values")
    if allowed:
        def same(a):
            return (a.strip().lower() == value.strip().lower()
                    if isinstance(a, str) and isinstance(value, str)
                    else a == value)
        return next((a for a in allowed if not same(a)), None)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 if value else 0.5
    if isinstance(value, str):
        return value + "_other"
    return None


def _without(x, name: str):
    """``x`` without ``choices`` and without any key named ``name``, at
    any depth: what an input's own echo leaves of a result."""
    if isinstance(x, dict):
        return {k: _without(v, name) for k, v in x.items()
                if k != name and k != "choices"}
    if isinstance(x, list):
        return [_without(v, name) for v in x]
    return x


def _moved_by(name: str, r0, r1, where: str = ""):
    """Where the result moved when ``name`` took another value, or None.
    A run the other value refuses has moved."""
    if isinstance(r1, dict) and "error" in r1:
        return "refused: " + str(r1["error"])[:80]
    return _diff(_without(_norm(r0, where), name),
                 _without(_norm(r1, where), name))


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=pytest.mark.slow) if k in MP.SLOW else k
    for k in sorted(MP.PROBES)])
def test_every_default_a_run_lists_changes_its_result(key, tmp_path,
                                                      monkeypatch):
    """GM-160: an input that changes nothing is not advertised. Each
    default a probe's run lists (``defaults_used``: the declared inputs
    that apply to the run) is given another value; the result must move,
    past the input's own echo and the ``choices``. Inputs that are a label
    only are in ``MP.LABEL_ONLY``; inputs whose effect this probe does not
    reach are in ``MP.NOT_REACHED``, each with why."""
    monkeypatch.chdir(tmp_path)
    base = key.split("#")[0]
    agent, method = base.split("|")
    probe = MP.PROBES[key]
    if probe.get("needs"):
        pytest.importorskip(probe["needs"])

    def run(params):
        for before in probe.get("before", ()):
            D.call_agent(before[0], before[1], copy.deepcopy(before[2]))
        return _run(agent, method, params)

    r0 = run(probe["params"])
    assert isinstance(r0, dict) and "error" not in r0, (key, r0)
    info = D._load_adapter(agent).METHOD_INFO[method].get("parameters") or {}
    where = str(tmp_path)
    still = []
    for name, value in (r0.get("defaults_used") or {}).items():
        if (base, name) in MP.LABEL_ONLY or (key, name) in MP.LABEL_ONLY \
                or (key, name) in MP.NOT_REACHED \
                or (base, name) in MP.KNOWN:
            continue
        other = _other_value(info.get(name) or {}, value)
        if other is None:
            continue
        if _moved_by(name, r0, run({**probe["params"], name: other}),
                     where) is None:
            still.append(f"{name}: {value!r} -> {other!r}")
    assert still == [], (
        f"{key}: these inputs changed nothing in the result; use them as "
        f"described, or say what they do and stop advertising them: {still}")


def test_the_allow_lists_name_declared_inputs_of_probed_methods():
    for table in (MP.LABEL_ONLY, MP.NOT_REACHED):
        for (key, name), why in table.items():
            assert key in MP.PROBES or key in {
                k.split("#")[0] for k in MP.PROBES}, key
            agent, method = key.split("#")[0].split("|")
            params = D._load_adapter(agent).METHOD_INFO[method][
                "parameters"]
            assert name in params and why, (key, name)


def test_a_wrapped_function_and_its_method_info_agree():
    """A method that passes its inputs straight to a function (the
    reference wrappers): every name, required flag and default
    METHOD_INFO gives is the function's own (inspect.signature)."""
    import inspect
    problems, checked = [], 0
    for agent in sorted(D.MODULE_REGISTRY):
        mod = D._load_adapter(agent)
        for method, handler in mod.METHOD_REGISTRY.items():
            fn = getattr(handler, "__wrapped__", None)
            if fn is None:
                continue
            checked += 1
            declared = (mod.METHOD_INFO.get(method) or {}).get(
                "parameters") or {}
            sig = inspect.signature(fn).parameters
            catch_all = any(p.kind == p.VAR_KEYWORD for p in sig.values())
            named = {n: p for n, p in sig.items()
                     if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
            if not catch_all:
                problems += [(agent, method, n, "not taken")
                             for n in set(declared) - set(named)]
            for n, p in named.items():
                spec = declared.get(n)
                if spec is None:
                    problems.append((agent, method, n, "not listed"))
                elif (p.default is p.empty) != bool(spec.get("required")):
                    problems.append((agent, method, n, "required flag"))
                elif p.default is not p.empty and p.default is not None \
                        and spec.get("default") != p.default:
                    problems.append((agent, method, n, "default"))
    assert checked > 1000
    assert problems == []


def test_a_default_that_depends_on_another_input_says_so():
    seen = 0
    for agent in sorted(D.MODULE_REGISTRY):
        mod = D._load_adapter(agent)
        for method, info in mod.METHOD_INFO.items():
            params = info.get("parameters") or {}
            for name, spec in params.items():
                dep = spec.get("default_depends_on") \
                    if isinstance(spec, dict) else None
                if not dep:
                    continue
                seen += 1
                where = (agent, method, name)
                other = dep["parameter"]
                assert other in params, where
                assert "default" not in spec, where
                assert f"Default depends on {other}" in spec[
                    "description"], where
                allowed = spec.get("allowed_values")
                if allowed:
                    assert set(dep["values"].values()) | {
                        dep["otherwise"]} <= set(allowed), where
                theirs = params[other].get("allowed_values")
                if theirs:
                    assert set(dep["values"]) <= set(theirs), where
    # bearing_capacity_analysis and the package that runs it
    assert seen >= 2


def test_default_in_force_follows_the_other_input():
    spec = {"default_depends_on": {"parameter": "factor_method",
                                   "values": {"terzaghi": "kumbhojkar"},
                                   "otherwise": "vesic"}}
    assert D.default_in_force(spec, {"factor_method": "Terzaghi"}) \
        == "kumbhojkar"
    assert D.default_in_force(spec, {"factor_method": "meyerhof"}) \
        == "vesic"
    assert D.default_in_force(spec, {}) == "vesic"
    assert D.default_in_force({"default": 3}, {}) == 3


def test_terzaghi_lists_the_n_gamma_it_used(tmp_path, monkeypatch):
    """GM-158: with factor_method='terzaghi' the N-gamma default is
    Kumbhojkar's, and defaults_used says so, on the tool and the package."""
    monkeypatch.chdir(tmp_path)
    footing = {"width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0}
    r = D.call_agent("bearing_capacity", "bearing_capacity_analysis",
                     {**footing, "factor_method": "terzaghi"})
    assert r["defaults_used"]["ngamma_method"] == "kumbhojkar"
    assert r["ngamma_method"] == "kumbhojkar"
    assert r["defaults_used"]["failure_mode"] == "general"
    r = D.call_agent("bearing_capacity", "bearing_capacity_analysis",
                     dict(footing))
    assert r["defaults_used"]["ngamma_method"] == "vesic"
    assert "failure_mode" not in r["defaults_used"]
    r = D.call_agent("bearing_capacity", "bearing_capacity_factors",
                     {"friction_angle": 30.0})
    assert set(r["defaults_used"]) == {"method"}
    r = D.call_agent("calc_package", "bearing_capacity_package",
                     {**footing, "factor_method": "terzaghi"})
    assert "error" not in r, r
    assert r["defaults_used"]["ngamma_method"] == "kumbhojkar"
