"""The lateral-pile tool's cyclic stiff clay above the water table (backlog
GM-202): the number of cycles N reaches the p-y model as ``n_cycles``, and
loading='cyclic' without it is refused rather than run as static."""

import pytest

from funhouse_agent.dispatch import call_agent

_PILE = {"pile_diameter": 0.9, "pile_length": 15.0, "pile_E": 25e6,
         "Vt": 300.0}
_CLAY = {"top": 0.0, "bottom": 16.0, "model": "StiffClayAboveWT",
         "c": 100.0, "gamma": 19.0, "eps50": 0.005}


def _run(**layer):
    return call_agent("lateral_pile", "lateral_pile_analysis",
                      dict(_PILE, layers=[dict(_CLAY, **layer)]))


def test_cyclic_runs_with_the_number_of_cycles():
    static = _run()
    cyclic = _run(loading="cyclic", n_cycles=100)
    assert "error" not in cyclic, cyclic.get("error")
    assert cyclic["y_top_m"] > 1.3 * static["y_top_m"]
    assert cyclic["max_moment_kNm"] > static["max_moment_kNm"]
    n = next(c for c in cyclic["choices"]
             if c["what"].startswith(
                 "number of cycles of load application"))
    assert n["chosen"] == "100 (y50 x 2.2)" and n["how"] == "given"


def test_cyclic_without_the_number_of_cycles_is_refused():
    r = _run(loading="cyclic")
    assert "n_cycles" in r["error"] and "no default" in r["error"]


def test_n_cycles_on_another_model_is_refused():
    r = call_agent("lateral_pile", "lateral_pile_analysis", dict(
        _PILE, layers=[{"top": 0.0, "bottom": 16.0,
                        "model": "SoftClayMatlock", "c": 30.0,
                        "gamma": 8.0, "eps50": 0.01, "loading": "cyclic",
                        "n_cycles": 100}]))
    assert "n_cycles" in r["error"]


def test_get_py_curve_takes_the_number_of_cycles():
    base = {"model": "StiffClayAboveWT", "c": 100.0, "gamma": 19.0,
            "eps50": 0.005, "z": 2.0, "pile_diameter": 0.9}
    static = call_agent("lateral_pile", "get_py_curve", base)
    cyclic = call_agent("lateral_pile", "get_py_curve",
                        dict(base, loading="cyclic", n_cycles=100))
    assert "error" not in cyclic, cyclic.get("error")
    # the same plateau, reached 2.2 times further out
    assert cyclic["p_kN_per_m"][-1] == pytest.approx(
        static["p_kN_per_m"][-1])
    assert cyclic["y_m"][-1] == pytest.approx(2.2 * static["y_m"][-1])
