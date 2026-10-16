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
    assert r["defaults_used"] == {
        "head_depth": 0.0, "factor_of_safety": 2.5, "method": "auto",
        "include_uplift": False, "cohesive_phi": 25.0,
        "uplift_skin_fraction": 0.75}
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
    # the adapter gives a layer with no delta_phi_ratio 0.75: its default
    assert by["Nordlund delta/phi"]["how"] == "default"
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
