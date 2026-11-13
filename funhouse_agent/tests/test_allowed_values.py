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
    # "separators": drainage quality "Very Poor" is very_poor
    mod = D._load_adapter("calc_package")
    _check_parameter_values(mod, "pavement_design_package",
                            {"drainage_quality": "Very Poor"})


class _Mod:
    METHOD_INFO = {"m": {"parameters": {
        "kind": {"allowed_values": ["a", "b"]},
        "kinds": {"type": "array", "allowed_values": ["a", "b"]},
        "flag": {"allowed_values": [False, True]},
        "n": {"type": "int", "allowed_values": [3, 6]},
        "fold": {"allowed_values": ["x_y"], "any_case": True},
        "sep": {"allowed_values": ["x_y"], "any_case": "separators"},
    }, "parameter_aliases": {"k": "kind"}}}


@pytest.mark.parametrize("params,refused", [
    ({"kind": "a"}, False),
    ({"kind": "c"}, True),
    ({"kind": "A"}, True),                 # exact unless any_case
    ({"k": "c"}, True),                    # an alias is checked as its name
    ({"kind": None}, False),               # None: the method's default
    ({"kind": 3}, False),                  # another kind: left to the method
    ({"kinds": ["a", "b"]}, False),        # each item of a list
    ({"kinds": ["a", "z"]}, True),
    ({"flag": True}, False),
    ({"flag": 1}, False),                  # an int is not a bool: left alone
    ({"n": 6}, False),
    ({"n": 4}, True),
    ({"n": "3"}, False),                   # the adapter converts it
    ({"fold": " X_Y "}, False),
    ({"fold": "x y"}, True),
    ({"sep": "X Y"}, False),
    ({"sep": "x-y"}, False),
    ({"sep": "xz"}, True),
    ({"other": "anything"}, False),        # no list declared
])
def test_the_shapes_enforced(params, refused):
    if refused:
        with pytest.raises(ValueError, match=REFUSED):
            _check_parameter_values(_Mod, "m", params)
    else:
        _check_parameter_values(_Mod, "m", params)


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
