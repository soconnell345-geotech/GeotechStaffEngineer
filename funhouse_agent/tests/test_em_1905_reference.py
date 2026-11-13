"""USACE EM 1110-1-1905 (shallow foundations on soils, 31 July 2025)
through the reference tools.

The library's package ``em_1905`` (2026-11-10 reference review, REF-72):
66 lookups in the manual's US units, the chapter text by its printed Army
paragraph numbers ('2-7a(2)(c)1', 'B-1b') and a figure catalog. These check
what the references helper is told it has: every lookup listed, a call
answered with the printed value, a refusal reaching the helper whole, and
the Army ids passing through the text tools unchanged, all through
``dispatch.call_agent`` as the helper calls them. The values are the
manual's own (the table printed on Figure 7-16) or made up.
"""

import inspect
import json
import re

import pytest

pytest.importorskip("geotech_references.em_1905")

from funhouse_agent import dispatch  # noqa: E402

REF = "em_1905"


def _call(agent, method, params):
    out = dispatch.call_agent(agent, method, params,
                              allowed_agents=dispatch.REFERENCE_MODULES)
    json.dumps(out)
    return out


def _ok(agent, method, params):
    out = _call(agent, method, params)
    assert "error" not in out, out
    return out


def test_registered_as_a_reference_module():
    assert REF in dispatch.REFERENCE_MODULES
    assert REF not in dispatch.ANALYSIS_MODULES
    assert REF in dispatch.FOUNDATIONS_REFERENCES
    from funhouse_agent.adapters import MODULE_REGISTRY
    assert "EM 1110-1-1905" in MODULE_REGISTRY[REF]["title"]
    assert MODULE_REGISTRY[REF]["title"] in dispatch.held_references()


def test_every_lookup_of_the_package_is_listed():
    from geotech_references.em_1905 import (
        equations, figures, settlement, tables)
    held = {n for m in (equations, figures, settlement, tables)
            for n, f in inspect.getmembers(m, inspect.isfunction)
            if not n.startswith("_") and f.__module__ == m.__name__}
    listed = {n for group in dispatch.list_methods(REF).values()
              for n in (group if isinstance(group, (list, dict)) else [])}
    assert len(held) == 66
    assert held <= listed, sorted(held - listed)
    text = {"retrieve_section", "search_sections", "list_chapters",
            "load_chapter"}
    assert listed == held | text


def test_a_printed_value_comes_back_with_its_page():
    out = _ok(REF, "figure_7_16_time_factor", {"U_percent": 50.0, "T_v": None})
    assert out["T_v"] == pytest.approx(0.197)
    assert out["interpolated"] is False
    assert "Figure 7-16, p. 185" in out["source"]


def test_a_refusal_reaches_the_helper_whole():
    """A refusal the page prints no condition for says it is the
    library's reading; the helper gets the library's whole message."""
    from geotech_references.em_1905 import settlement
    with pytest.raises(ValueError) as raised:
        settlement.equation_7_34_time_factor(R3=0.3, Rt=0.2, t_years=1.0)
    out = _call(REF, "equation_7_34_time_factor",
                {"R3": 0.3, "Rt": 0.2, "t_years": 1.0})
    assert str(raised.value) in out["error"]
    assert "the library's reading, not a condition the page prints" \
        in out["error"]


@pytest.mark.parametrize("sid", ["2-7a(2)", "2-7a(2)(c)1", "B-1b", "B"])
def test_army_section_ids_pass_through_unchanged(sid):
    own = _ok(REF, "retrieve_section", {"section_id": sid})
    assert own["section_id"] == sid
    via_db = _ok("reference_db", "reference_get",
                 {"reference": REF, "section_id": sid})
    via_db = via_db.get("result", via_db)
    assert via_db["section_id"] == sid


def test_children_and_under_carry_army_ids():
    sec = _ok(REF, "retrieve_section", {"section_id": "2-7a(2)"})
    kids = [c["section_id"] for c in sec["children"]]
    assert kids[:3] == ["2-7a(2)(a)", "2-7a(2)(b)", "2-7a(2)(c)"]
    for kid in kids:
        assert _ok(REF, "retrieve_section",
                   {"section_id": kid})["section_id"] == kid
    deep = _ok(REF, "retrieve_section", {"section_id": "2-7a(2)(c)1"})
    assert deep["parent_id"] == "2-7a(2)(c)"
    assert deep["under"].startswith("2-7 ")


def test_search_hits_carry_army_ids():
    own = _ok(REF, "search_sections", {"query": "Burland Burbidge"})
    ids = [h["section_id"] for h in own["result"]]
    assert "7-13" in ids
    via_db = _ok("reference_db", "reference_search",
                 {"query": "Burland Burbidge", "reference": REF})
    assert {h["section_id"] for h in via_db["result"]} & set(ids)
    for sid in ids[:3]:
        assert _ok(REF, "retrieve_section",
                   {"section_id": sid})["section_id"] == sid


def test_the_section_id_examples_are_army_ids_it_holds():
    spec = dispatch.describe_method(REF, "retrieve_section")
    desc = spec["parameters"]["section_id"]["description"]
    examples = [p.split("'")[0] for p in desc.split("'")[1::2]]
    assert any("(" in e for e in examples), desc
    for sid in examples:
        assert _ok(REF, "retrieve_section",
                   {"section_id": sid})["section_id"] == sid


def test_aliases_name_the_method_they_reach():
    # stripped of their numbers these would mislead; curated names stand in
    for gone in ("time_factor", "shape_factor", "factor_of_safety"):
        assert "Unknown method" in _call(REF, gone, {})["error"]
    out = _ok(REF, "consolidation_time_factor",
              {"U_percent": 50.0, "T_v": None})
    assert out["T_v"] == pytest.approx(0.197)
    out = _ok(REF, "burland_burbidge_time_factor",
              {"R3": 0.3, "Rt": 0.2, "t_years": 30.0})
    assert out["C3"] == pytest.approx(1.5)
    assert _ok(REF, "bearing_factor_of_safety",
               {"q_ult": 15.0, "q_eq": 5.0})["FS"] == pytest.approx(3.0)


#: A call with null for each required parameter the library takes as None
#: (a continuous footing's L or L', the given one of U and Tv, sigma'_p of a
#: normally consolidated sand, theta and d_g where nothing needs them).
#: Required on purpose: a helper that leaves L_eff out gets an error, not a
#: continuous footing by silence. Made-up values.
NULL_CALLS = {
    "equation_5_2_equivalent_bearing_pressure":
        {"Q": 1.2, "Wf": 0.6, "B_eff": 2, "L_eff": None, "u_d": 0},
    "equation_5_4_effective_dimensions":
        {"B": 8, "L": None, "V": 200, "M_B": 100, "M_L": 0},
    "equation_5_17_punching_capacity":
        {"q_ult_1": 20.0, "q_ult_2": 8.0, "H_s": 3.0, "B_eff": 6.0,
         "L_eff": None, "B": 6.0, "L": None},
    "equation_6_8_influence_factor":
        {"shape": "square", "B": 8.0, "z": 14.0, "L": None},
    "equation_6_12_two_to_one_influence_factor":
        {"shape": "circle", "B": 6.0, "z": 6.0, "L": None},
    "equation_7_19_strain_influence_factor":
        {"footing": "axisymmetric", "z": 2.0, "B": 6.0, "I_zp": 0.6,
         "L": None},
    "equation_7_29_burland_burbidge_settlement":
        {"state": "normally consolidated", "q": 3.0, "sigma_zo_mean": 0.35,
         "sigma_p": None, "Z_I": 6.0, "I_c": 0.01, "C1": 1.0, "C2": 1.0,
         "C3": 1.0},
    "equation_7_32_shape_factor": {"footing": "strip", "L": None, "B": None},
    "figure_7_16_time_factor:U": {"U_percent": None, "T_v": 0.2},
    "figure_7_16_time_factor:T": {"U_percent": 70, "T_v": None},
    "table_5_2_aashto_shape_depth_factors":
        {"phi_deg": 30, "B_eff": 2, "L_eff": None, "Df": 40},
    "table_5_2_usace_shape_depth_factors":
        {"flow_number": 3, "B_eff": 8, "L_eff": None, "Df": 2.5},
    "table_5_3_aashto_load_inclination_factors":
        {"H": 10, "V": 100, "B_eff": 5, "L_eff": None, "c": 0,
         "phi_deg": 30, "theta_deg": 0},
    "table_5_4_undrained_factors":
        {"method": "usace", "B_eff": 2, "L_eff": None, "Df": 6, "H": 0,
         "V": 10, "su": 1, "theta_deg": None, "alpha_b_deg": 0,
         "alpha_g_deg": 0, "d_g": None},
    "ultimate_bearing_capacity_drained":
        {"method": "aashto", "phi_deg": 30, "c": 0, "q0": 4.0,
         "gamma_eff": 0.06, "B_eff": 2, "L_eff": None, "Df": 40, "H": 0,
         "V": 100, "theta_deg": None, "alpha_b_deg": 0, "alpha_g_deg": 0,
         "d_g": None},
    "ultimate_bearing_capacity_undrained":
        {"method": "usace", "su": 1.0, "q0": 0.5, "B_eff": 2, "L_eff": None,
         "Df": 6, "H": 0, "V": 10, "theta_deg": None, "alpha_b_deg": 0,
         "alpha_g_deg": 0, "d_g": None},
}

#: Parameters whose library docstring says when they are needed but not, in
#: words, to pass None otherwise; the docstrings are to be changed in the
#: library (the lead's call, 2026-11-12). The test below allows these only.
SAYS_ONLY_WHEN_NEEDED = {
    ("table_5_4_undrained_factors", "theta_deg"),
    ("table_5_4_undrained_factors", "d_g"),
    ("ultimate_bearing_capacity_drained", "theta_deg"),
    ("ultimate_bearing_capacity_drained", "d_g"),
    ("ultimate_bearing_capacity_undrained", "theta_deg"),
    ("ultimate_bearing_capacity_undrained", "d_g"),
}


def _nullable_required():
    """(method, parameter) for every listed lookup's required parameter
    that may be null, as describe shows it."""
    out = set()
    for group in dispatch.list_methods(REF).values():
        for method in (group if isinstance(group, (list, dict)) else []):
            d = dispatch.describe_method(REF, method)
            for p, spec in (d.get("parameters") or {}).items():
                if spec.get("required") and "None" in str(spec.get("type")):
                    out.add((method, p))
    return out


def test_a_required_parameter_that_may_be_null_says_when():
    """Each required parameter the library takes as None says in words,
    as describe shows it, when to pass null; each answers a call with it
    null."""
    nullable = _nullable_required()
    assert len(nullable) == 24
    called = {(m.split(":")[0], p) for m, c in NULL_CALLS.items()
              for p, v in c.items() if v is None}
    assert nullable == called, sorted(nullable ^ called)
    for key, params in NULL_CALLS.items():
        _ok(REF, key.split(":")[0], params)
    silent = set()
    for method, p in nullable:
        desc = dispatch.describe_method(REF, method)["parameters"][p]
        if not re.search(r"\b(None|null)\b", desc.get("description", "")):
            silent.add((method, p))
    assert silent <= SAYS_ONLY_WHEN_NEEDED, sorted(
        silent - SAYS_ONLY_WHEN_NEEDED)


def test_its_figures_are_in_the_catalog():
    fig = _ok("figure_db", "figure_get",
              {"reference": REF, "figure_number": "7-16"})
    fig = fig.get("result", fig)
    assert fig["printed_page"] in (185, "185")
    hits = _ok("figure_db", "figure_search",
               {"query": "influence chart vertical stress", "reference": REF})
    assert any(h.get("reference") == REF for h in hits["result"])
