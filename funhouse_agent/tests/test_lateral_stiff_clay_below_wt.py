"""The lateral-pile tool's stiff clay below the water table (backlog
GM-208): ks for static loading and kc for cyclic reach the p-y model
(COM624P Table 3.2), and each loading refuses the other's k."""

import pytest

from funhouse_agent.dispatch import call_agent

_PILE = {"pile_diameter": 0.9, "pile_length": 15.0, "pile_E": 25e6,
         "Vt": 300.0}
_CLAY = {"top": 0.0, "bottom": 16.0, "model": "StiffClayBelowWT",
         "c": 100.0, "gamma": 9.0, "eps50": 0.005}


def _run(**layer):
    return call_agent("lateral_pile", "lateral_pile_analysis",
                      dict(_PILE, layers=[dict(_CLAY, **layer)]))


def test_static_with_ks_and_cyclic_with_kc():
    static = _run(ks=271_400.0)
    cyclic = _run(loading="cyclic", kc=108_600.0)
    assert "error" not in static, static.get("error")
    assert "error" not in cyclic, cyclic.get("error")
    assert cyclic["y_top_m"] > static["y_top_m"]
    assert cyclic["max_moment_kNm"] > static["max_moment_kNm"]
    k = next(c for c in cyclic["choices"]
             if c["what"].startswith("k of the initial line, kc"))
    assert k["chosen"] == "108600 kN/m3" and k["how"] == "given"


@pytest.mark.parametrize("layer, needs", [
    ({}, "ks"),
    ({"loading": "cyclic"}, "kc"),
    ({"loading": "cyclic", "ks": 271_400.0}, "kc"),
    ({"ks": 271_400.0, "kc": 108_600.0}, "ks only"),
])
def test_the_wrong_k_is_refused(layer, needs):
    r = _run(**layer)
    assert needs in r["error"]


def test_get_py_curve_takes_kc():
    base = {"model": "StiffClayBelowWT", "c": 100.0, "gamma": 9.0,
            "eps50": 0.005, "z": 1.5, "pile_diameter": 0.6}
    static = call_agent("lateral_pile", "get_py_curve",
                        dict(base, ks=271_400.0))
    cyclic = call_agent("lateral_pile", "get_py_curve",
                        dict(base, loading="cyclic", kc=108_600.0))
    assert "error" not in cyclic, cyclic.get("error")
    # the residuals (Eqs. 3.21, 3.26): 47.44 and 87.27 kN/m at x/b 2.5
    assert static["p_kN_per_m"][-1] == pytest.approx(47.44, abs=0.01)
    assert cyclic["p_kN_per_m"][-1] == pytest.approx(87.27, abs=0.01)
