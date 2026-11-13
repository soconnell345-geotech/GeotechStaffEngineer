"""A value outside a parameter's ``allowed_values`` is refused at dispatch
(BACKLOG GM-317; module audit 2026-11-12 §2.2 X2 and §6 T1 (c)). Made-up
values only."""

import copy

import pytest

from aec_agent_core.method_call import _check_parameter_values
from funhouse_agent import dispatch as D
from funhouse_agent.tests import method_probes as MP

NOT_A_VALUE = "gm317_not_a_value"
REFUSED = "is not one of the values it takes"


def _probe(key):
    return copy.deepcopy(MP.PROBES[key]["params"])


def test_a_trend_type_it_does_not_fit_is_refused():
    r = D.call_agent("subsurface", "compute_trend",
                     {"depths": [1.0, 2.0, 3.0, 4.0],
                      "values": [10.0, 12.0, 15.0, 19.0],
                      "trend_type": "exponential"})
    assert REFUSED in r["error"]
    assert "['linear', 'log_linear']" in r["error"]


def test_a_calc_package_format_it_does_not_write_is_refused(tmp_path,
                                                            monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = "calc_package|bearing_capacity_package"
    for fmt in ("docx", "PDF", " pdf"):
        r = D.call_agent("calc_package", "bearing_capacity_package",
                         {**_probe(key), "format": fmt})
        assert REFUSED in r.get("error", ""), (fmt, r)
    assert not list(tmp_path.glob("*.docx"))


def test_the_refusal_names_the_parameter_a_near_value_and_the_list():
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**_probe("axial_pile|axial_pile_capacity"),
                      "method": "Beta"})
    assert r["error"] == (
        "ValueError: axial_pile_capacity: method='Beta' is not one of the "
        "values it takes. Did you mean 'beta'? Valid values: ['auto', "
        "'beta']. describe_method gives each one's meaning.")
    r = D.call_agent("axial_pile", "axial_pile_capacity",
                     {**_probe("axial_pile|axial_pile_capacity"),
                      "open_pipe_method": "paikowsky-whitman"})
    assert "Did you mean 'paikowsky_whitman'?" in r["error"]


def test_any_case_matches_as_the_code_does():
    """bearing_capacity lower-cases its shape: 'Square' runs; axial_pile
    compares exactly: 'Beta' is refused (above)."""
    r = D.call_agent("bearing_capacity", "bearing_capacity_analysis",
                     {**_probe("bearing_capacity|bearing_capacity_analysis"),
                      "shape": "Square"})
    assert "error" not in r, r
    # the AASHTO tables strip, lower-case and read a space as "_": drainage
    # quality "Very Poor" is very_poor; "very-poor" is not
    mod = D._load_adapter("calc_package")
    _check_parameter_values(mod, "pavement_design_package",
                            {"drainage_quality": " Very Poor"})
    with pytest.raises(ValueError, match=REFUSED):
        _check_parameter_values(mod, "pavement_design_package",
                                {"drainage_quality": "very-poor"})


def test_a_padded_value_the_code_does_not_strip_is_refused():
    """GM-316 review B2: Schmertmann lower-cases its shape without
    stripping it, so " strip " ran as a square footing (39.67 mm in place
    of 60.84 mm). It is refused now; "STRIP" still runs as a strip."""
    params = {"q_net": 150.0, "B": 2.0, "gamma_soil": 18.0,
              "layers": [{"depth_top": 0.0, "depth_bottom": 4.0,
                          "Es": 12000.0},
                         {"depth_top": 4.0, "depth_bottom": 8.0,
                          "Es": 20000.0}]}
    r = D.call_agent("settlement", "schmertmann_settlement",
                     {**params, "shape": " strip "})
    assert REFUSED in r.get("error", ""), r
    assert "Did you mean 'strip'?" in r["error"]
    upper = D.call_agent("settlement", "schmertmann_settlement",
                         {**params, "shape": "STRIP"})
    strip = D.call_agent("settlement", "schmertmann_settlement",
                         {**params, "shape": "strip"})
    square = D.call_agent("settlement", "schmertmann_settlement",
                          {**params, "shape": "square"})
    assert upper["schmertmann_settlement_mm"] == strip[
        "schmertmann_settlement_mm"] != square["schmertmann_settlement_mm"]
    # a handler that strips takes the padded value: bearing factor_method
    mod = D._load_adapter("bearing_capacity")
    _check_parameter_values(mod, "bearing_capacity_analysis",
                            {"factor_method": " Meyerhof "})


def test_an_alias_is_read_but_listed_apart_from_the_values():
    """GM-316 review B3: the other spellings a method reads are aliases of
    a listed value, never values of their own. describe_method gives them
    as "also read as"; a misleading one ("grid" for a steel grid, which an
    agent would pick for a geogrid) is read but never offered; the refusal
    lists and suggests listed values only."""
    mod = D._load_adapter("retaining_walls")
    spec = mod.METHOD_INFO["mse_wall"]["parameters"]["reinforcement_type"]
    for alias in ("barmat", "steel_grid", "grid", "strip"):
        _check_parameter_values(mod, "mse_wall",
                                {"reinforcement_type": alias})
    with pytest.raises(ValueError, match=REFUSED) as err:
        _check_parameter_values(mod, "mse_wall",
                                {"reinforcement_type": "steel-grid"})
    text = str(err.value)
    assert "Did you mean 'metallic_grid'" in text
    assert f"Valid values: {spec['allowed_values']}." in text
    assert "barmat" not in text and "'grid'" not in text
    doc = D.describe_method("retaining_walls", "mse_wall")
    shown = doc["parameters"]["reinforcement_type"]
    assert "'barmat' = 'metallic_grid'" in shown["values_note"]
    assert "'grid'" not in shown["values_note"]
    for key in ("value_aliases", "hidden_value_aliases", "any_case",
                "refused_values"):
        assert key not in shown
    # a calc package's older selector spellings are its aliases too
    pkg = D._load_adapter("calc_package")
    engine = pkg.METHOD_INFO["reliability_package"]["parameters"]["engine"]
    assert engine["allowed_values"] == ["fosm", "pem", "monte_carlo",
                                        "form"]
    assert engine["value_aliases"]["monte carlo"] == "monte_carlo"
    _check_parameter_values(pkg, "reliability_package",
                            {"engine": " Monte Carlo"})


def test_describe_method_gives_the_value_rules_in_words():
    """Review C14: any_case and refused_values are given in words, not as
    raw keys; a refusal reason carries no backlog ID."""
    doc = D.describe_method("bearing_capacity", "bearing_capacity_analysis")
    fm = doc["parameters"]["factor_method"]
    assert "any_case" not in fm and "refused_values" not in fm
    assert fm["values_note"].startswith(
        "Case and surrounding spaces ignored. Not taken: 'hansen' (")
    doc = D.describe_method("soe", "design_ground_anchor")
    note = doc["parameters"]["soil_type"]["values_note"]
    assert "'gravel' (" in note and "GM-" not in note


def test_a_package_whose_tools_share_a_list_reads_it_one_way():
    """Review C20: in a package that runs several tools, a parameter's
    any_case agrees across the tools that list it, and its aliases and
    refusals are every tool's, not the first one's."""
    pkg = D._load_adapter("calc_package")
    from funhouse_agent.adapters.calc_package import (
        _PACKAGE_TOOLS, _tool_adapter)
    from aec_agent_core.method_call import value_aliases
    for package, spec in _PACKAGE_TOOLS.items():
        if len(spec["tools"]) < 2:
            continue
        info = _tool_adapter(package).METHOD_INFO
        merged = pkg.METHOD_INFO[package]["parameters"]
        by = {}
        for tool in spec["tools"].values():
            for name, p in info[tool]["parameters"].items():
                if isinstance(p, dict) and p.get("allowed_values"):
                    by.setdefault(name, []).append(p)
        for name, specs in by.items():
            assert len({repr(p.get("any_case")) for p in specs}) == 1, (
                package, name)
            want = {}
            for p in specs:
                want.update(value_aliases(p))
            assert value_aliases(merged[name]) == want, (package, name)


def _probed_lists():
    """(agent, method, parameter, spec) of every declared list on a probed
    method's adapter (calc packages included)."""
    out, seen = [], set()
    for key in sorted(MP.PROBES):
        agent = key.split("#")[0].split("|")[0]
        if agent in seen:
            continue
        seen.add(agent)
        mod = D._load_adapter(agent)
        for method, info in mod.METHOD_INFO.items():
            for name, spec in (info.get("parameters") or {}).items():
                if isinstance(spec, dict) and spec.get("allowed_values"):
                    out.append((agent, method, name, spec))
    return out


def test_every_any_case_is_a_fold_the_check_knows():
    """Each ``any_case`` names one of the folds in ``VALUE_FOLDS`` (each the
    way one handler reads a string), never one the check would guess."""
    from aec_agent_core.method_call import VALUE_FOLDS, VALUE_FOLD_WORDS
    assert set(VALUE_FOLDS) == set(VALUE_FOLD_WORDS)
    bad = [(a, m, n, s["any_case"]) for a, m, n, s in _probed_lists()
           if s.get("any_case") and s["any_case"] not in VALUE_FOLDS]
    assert bad == []


class _Mod:
    METHOD_INFO = {"m": {"parameters": {
        "kind": {"allowed_values": ["a", "b"]},
        "kinds": {"type": "array", "allowed_values": ["a", "b"]},
        "flag": {"allowed_values": [False, True]},
        "n": {"type": "int", "allowed_values": [3, 6]},
        "fold": {"allowed_values": ["x_y"], "any_case": True},
        "strip": {"allowed_values": ["x_y"], "any_case": "strip"},
        "nosp": {"allowed_values": ["2:1"], "any_case": "no_spaces"},
        "nosph": {"allowed_values": ["ab 1"], "any_case": "no_spaces_hyphens"},
        "under": {"allowed_values": ["x_y"],
                  "any_case": "spaces_as_underscores"},
    }, "parameter_aliases": {"k": "kind"}}}


@pytest.mark.parametrize("params,refused", [
    ({"kind": "a"}, False),
    ({"kind": "c"}, True),
    ({"kind": "A"}, True),                 # exact unless any_case
    ({"k": "c"}, True),                    # an alias is checked as its name
    ({"n": 3, "k": "c"}, True),            # ... after a listed one too (N1)
    ({"n": 3, "k": "a"}, False),
    ({"kind": None}, False),               # None: the method's default
    ({"kind": 3}, False),                  # another kind: left to the method
    ({"kinds": ["a", "b"]}, False),        # each item of a list
    ({"kinds": ["a", "z"]}, True),
    ({"flag": True}, False),
    ({"flag": 1}, False),                  # an int is not a bool: left alone
    ({"n": 6}, False),
    ({"n": 4}, True),
    ({"n": "3"}, False),                   # the adapter converts it
    ({"fold": "X_Y"}, False),              # case only
    ({"fold": " X_Y "}, True),             # not stripped (review B2)
    ({"strip": " X_Y "}, False),
    ({"strip": "x y"}, True),
    ({"nosp": " 2 : 1 "}, False),
    ({"nosp": "2_:_1"}, True),
    ({"nosph": "AB-1"}, False),
    ({"nosph": "ab\t1"}, False),
    ({"nosph": "ab_1"}, True),
    ({"under": " X Y "}, False),
    ({"under": "x-y"}, True),
    ({"under": "xy"}, True),
    ({"other": "anything"}, False),        # no list declared
])
def test_the_shapes_enforced(params, refused):
    if refused:
        with pytest.raises(ValueError, match=REFUSED):
            _check_parameter_values(_Mod, "m", params)
    else:
        _check_parameter_values(_Mod, "m", params)


def test_a_parameter_named_by_its_alias_after_a_listed_one_is_checked():
    """GM-316 re-check N1: the value check kept the parameter-name aliases
    in a variable it then reused for the value aliases, so after the first
    listed parameter a parameter given under its alias name went
    unchecked. The reviewer's case: footing_shape (the settlement
    package's alias of shape) after immediate_method ran "circle footing"
    as a rectangular footing."""
    pkg = D._load_adapter("calc_package")
    for shape in ("circle footing", NOT_A_VALUE):
        with pytest.raises(ValueError, match=REFUSED):
            _check_parameter_values(pkg, "settlement_package", {
                "immediate_method": "elastic", "footing_shape": shape,
                "L": 3})
    for shape in ("circular", "circle", "Square"):
        _check_parameter_values(pkg, "settlement_package", {
            "immediate_method": "elastic", "footing_shape": shape, "L": 3})


class _Applies:
    METHOD_INFO = {"m": {"parameters": {
        "theory": {"allowed_values": ["rankine", "log_spiral"],
                   "default": "rankine", "any_case": "strip",
                   "value_aliases": {"log-spiral": "log_spiral"},
                   "hidden_value_aliases": {"spiral": "log_spiral"}},
        "delta": {"default": 0.66,
                  "applies_when": {"theory": ["log_spiral"]}},
    }}}


@pytest.mark.parametrize("theory,listed", [
    ("log_spiral", True), ("log-spiral", True), ("spiral", True),
    (" Log_Spiral ", True), ("rankine", False)])
def test_a_default_that_applies_to_a_value_applies_to_its_spellings(
        theory, listed):
    """GM-316 re-check N2: an ``applies_when`` naming a listed value holds
    for every spelling the value check takes (a shown or hidden alias, the
    parameter's fold), so the default it governs stays in defaults_used
    (and a package's Assumed Inputs)."""
    from aec_agent_core.method_call import _defaults_used
    used = _defaults_used(_Applies, "m", {"theory"}, {"theory": theory})
    assert ("delta" in used) is listed, used


def test_the_soe_wall_friction_default_is_listed_for_log_spiral_aliases(
        tmp_path, monkeypatch):
    """The reviewer's case: soe cantilever_excavation with passive_theory
    'log-spiral' or 'logspiral' lists passive_wall_friction_deg as for
    'log_spiral'."""
    monkeypatch.chdir(tmp_path)
    got = {}
    for theory in ("log_spiral", "log-spiral", "logspiral"):
        r = D.call_agent("soe", "cantilever_excavation", {
            **_probe("soe|cantilever_excavation"),
            "passive_theory": theory})
        assert "error" not in r, r
        got[theory] = r["defaults_used"].get("passive_wall_friction_deg")
    assert got["log_spiral"] is not None
    assert got["log-spiral"] == got["logspiral"] == got["log_spiral"]


def _made_up(spec):
    """A value outside the parameter's list, of the list's kind, or None
    where none can be made (a list of the two booleans)."""
    allowed = spec["allowed_values"]
    if all(isinstance(a, str) for a in allowed):
        return [NOT_A_VALUE] if spec.get("type") == "array" else NOT_A_VALUE
    if all(isinstance(a, bool) for a in allowed):
        return None
    if all(isinstance(a, int) for a in allowed):
        return max(allowed) + 1
    return None


@pytest.mark.parametrize("key", sorted(MP.PROBES))
def test_every_allowed_values_parameter_refuses_a_made_up_value(
        key, tmp_path, monkeypatch):
    """T1 (c): for every probed method, each parameter declaring
    ``allowed_values`` refuses a value outside it, before the method
    runs."""
    if MP.skip_reason(key):
        pytest.skip(MP.skip_reason(key))
    monkeypatch.chdir(tmp_path)
    agent, method = key.split("#")[0].split("|")
    mod = D._load_adapter(agent)
    spec_of = dict(mod.METHOD_INFO[method].get("parameters") or {})
    for name, spec in sorted(spec_of.items()):
        if not isinstance(spec, dict) or not spec.get("allowed_values"):
            continue
        value = _made_up(spec)
        if value is None:
            continue
        r = D.call_agent(agent, method, {**_probe(key), name: value})
        assert isinstance(r, dict) and REFUSED in str(r.get("error")), (
            key, name, r)
