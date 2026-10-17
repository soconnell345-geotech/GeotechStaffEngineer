"""BACKLOG GM-133: a liquefaction run never uses a water table nobody gave.

Until 2026-10-17 the CPT route (the unified ``liquefaction`` tool and
``liquepy.cpt_liquefaction``) took a 1.0 m water table and a 0.25 g PGA
without saying so: on the unified tool neither was a declared default, so
neither ``defaults_used`` nor ``choices`` showed them, and the water table
drives the result. The groundwater level is a site input (GEC 5,
FHWA-NHI-16-072, p. 12-23: "establish the location of the groundwater
level"), so every route now requires it, as the SPT routes always did, and a
CPT result lists it among its choices as given.
"""

import pytest

from funhouse_agent import dispatch as D
from liquepy_agent import has_liquepy

_CPT = {"depth": [0.5 * i for i in range(1, 21)],
        "q_c": [3000 + 400 * i for i in range(20)],
        "f_s": [30 + 2 * i for i in range(20)]}
_SPT = {"depth": [3, 5, 7], "N160": [10, 15, 20], "FC": [10, 10, 25],
        "gamma": [18, 19, 19], "amax_g": 0.3}
_NCEER = {"depths": [3, 5, 7], "N160": [10, 15, 20], "FC": [10, 10, 25],
          "gamma": [18, 19, 19], "amax_g": 0.3}
_PACKAGE = {"analysis_type": "liquefaction", "layer_depths": [3, 5, 7],
            "layer_N160": [10, 15, 20], "layer_FC": [10, 10, 25],
            "layer_gamma": [18, 19, 19], "amax_g": 0.3}

#: Every liquefaction route: agent, method, the call without its water
#: table, the water table's name, whether the route needs liquepy.
_ROUTES = [
    ("liquefaction", "liquefaction_analysis", {**_CPT, "pga": 0.3}, "gwl",
     True),
    ("liquefaction", "liquefaction_analysis", _SPT, "gwt_depth", True),
    ("liquefaction", "liquefaction_analysis",
     {**_SPT, "method": "nceer2001"}, "gwt_depth", False),
    ("liquepy", "cpt_liquefaction", {**_CPT, "pga": 0.3}, "gwl", True),
    ("liquepy", "spt_liquefaction", _SPT, "gwt_depth", True),
    ("liquepy", "field_correlations", dict(_CPT), "gwl", True),
    ("seismic_geotech", "liquefaction_evaluation", _NCEER, "gwt_depth",
     False),
    ("calc_package", "seismic_package", _PACKAGE, "gwt_depth", False),
]


def _skip_without_liquepy(needs):
    if needs and not has_liquepy():
        pytest.skip("liquepy not installed")


@pytest.mark.parametrize("agent, method, params, name, needs", _ROUTES)
def test_every_route_refuses_a_run_without_a_water_table(agent, method,
                                                         params, name, needs):
    _skip_without_liquepy(needs)
    out = D.call_agent(agent, method, dict(params))
    assert "error" in out, out
    assert name in out["error"]


@pytest.mark.parametrize("agent, method, params, name, needs",
                         [r for r in _ROUTES if r[0] != "calc_package"])
def test_every_route_runs_with_the_water_table_given(agent, method, params,
                                                     name, needs):
    _skip_without_liquepy(needs)
    out = D.call_agent(agent, method, {**params, name: 1.5})
    assert "error" not in out, out


@pytest.mark.parametrize("agent, method", [
    ("liquefaction", "liquefaction_analysis"), ("liquepy", "cpt_liquefaction")])
def test_a_cpt_refusal_says_why(agent, method):
    """The plain reason: the references treat the groundwater level as a
    site input, not a default."""
    _skip_without_liquepy(True)
    err = D.call_agent(agent, method, {**_CPT, "pga": 0.3})["error"]
    assert "water table" in err and "site input" in err
    assert "12-23" in err
    assert "1.0" not in err


@pytest.mark.parametrize("agent, method", [
    ("liquefaction", "liquefaction_analysis"), ("liquepy", "cpt_liquefaction")])
def test_a_cpt_run_without_a_pga_is_refused(agent, method):
    """The other undeclared CPT default: 0.25 g, refused like the SPT
    routes refuse a missing amax_g."""
    _skip_without_liquepy(True)
    err = D.call_agent(agent, method, {**_CPT, "gwl": 1.5})["error"]
    assert "peak ground acceleration" in err and "pga" in err


_SITE_INPUTS = {"gwl", "gwt_depth", "pga", "amax_g"}


@pytest.mark.parametrize("agent, method, params", [
    ("liquefaction", "liquefaction_analysis", {"gwl": 1.5, "pga": 0.3}),
    ("liquefaction", "liquefaction_analysis",
     {"gwt_depth": 1.5, "amax_g": 0.3}),
    ("liquepy", "cpt_liquefaction", {"gwl": 1.5, "pga": 0.3})])
def test_a_cpt_result_lists_the_water_table_as_given(agent, method, params):
    _skip_without_liquepy(True)
    r = D.call_agent(agent, method, {**_CPT, **params})
    by = {c["what"]: c for c in r["choices"]}
    water = by["water table"]
    assert water["chosen"] == "1.5 m below the ground surface"
    assert water["how"] == "given"
    assert r["choices"][1]["what"] == "water table"      # second, prominent
    assert not _SITE_INPUTS & set(r.get("defaults_used", {}))
    assert r["gwl_m"] == 1.5 and r["pga_g"] == 0.3


def test_the_cpt_crr_curve_is_said():
    """C0 = 2.8 (liquepy's value) is not an input on the CPT route; the
    result says it was used."""
    _skip_without_liquepy(True)
    r = D.call_agent("liquefaction", "liquefaction_analysis",
                     {**_CPT, "gwl": 1.5, "pga": 0.3})
    c0 = {c["what"]: c for c in r["choices"]}["CRR curve C0"]
    assert c0["chosen"] == "2.8" and c0["how"] == "module"


@pytest.mark.parametrize("agent, method, required", [
    ("liquefaction", "liquefaction_analysis", {"gwt_depth", "amax_g"}),
    ("liquepy", "cpt_liquefaction", {"gwl", "pga"}),
    ("liquepy", "spt_liquefaction", {"gwt_depth", "amax_g"}),
    ("liquepy", "field_correlations", {"gwl"}),
    ("seismic_geotech", "liquefaction_evaluation", {"gwt_depth", "amax_g"}),
    ("calc_package", "seismic_package", set())])
def test_no_route_declares_a_water_table_default(agent, method, required):
    params = D._load_adapter(agent).METHOD_INFO[method]["parameters"]
    for name in _SITE_INPUTS & set(params):
        assert "default" not in params[name], (agent, method, name)
    for name in required:
        assert params[name]["required"] is True, (agent, method, name)


def test_describe_method_says_the_water_table_is_a_site_input():
    doc = D.describe_method("liquefaction", "liquefaction_analysis")
    wt = doc["parameters"]["gwt_depth"]
    assert "REQUIRED" in wt["description"]
    assert "site input" in wt["description"]
    assert "default 1.0" not in str(doc)


def test_no_worked_example_leaves_the_water_table_out():
    """A worked example that relied on the 1.0 m would now fail to run;
    this names the cause directly."""
    from funhouse_agent import worked_examples as we
    names = {(a, m): n for a, m, _, n, _ in _ROUTES}
    for entry in we.load_examples():
        for call in entry.get("dispatch_calls", []):
            name = names.get((call["module"], call["method"]))
            if name is None:
                continue
            p = call.get("params", {})
            if call["method"] == "seismic_package" \
                    and p.get("analysis_type") != "liquefaction":
                continue
            assert {"gwl", "gwt_depth"} & set(p), entry["id"]
