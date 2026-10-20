"""BACKLOG GM-142: a default the tools used without saying so is now
either correct, refused or stated in the result's ``choices`` (settlement,
drilled shaft, aggregate piers, seismic earth pressure), through the
agent's own call path."""

import pytest

from funhouse_agent import dispatch as D


def _by_what(choices):
    return {c["what"]: c for c in choices}


def _error(result):
    return str(result.get("error", "")) if isinstance(result, dict) else ""


# ----------------------------------------------------------- settlement

_CLAY = {"thickness": 2.0, "depth_to_center": 4.0, "e0": 0.9, "Cc": 0.3,
         "Cr": 0.05, "sigma_v0": 60.0}


def test_consolidation_tool_needs_the_load_and_width_without_delta_sigma():
    # It took B = 1.0 m and q_net = 0 without saying so (GM-142).
    r = D.call_agent("settlement", "consolidation_settlement",
                     {"layers": [dict(_CLAY)]})
    assert "q_net" in _error(r) and "B" in _error(r)
    r = D.call_agent("settlement", "consolidation_settlement",
                     {"layers": [dict(_CLAY)], "q_net": 100.0, "B": 2.0})
    assert r["consolidation_settlement_mm"] > 0
    assert _by_what(r["choices"])["footing length"]["how"] == "default"
    # with delta_sigma given neither is needed
    r = D.call_agent("settlement", "consolidation_settlement",
                     {"layers": [dict(_CLAY)], "delta_sigma": 30.0})
    assert r["consolidation_settlement_mm"] > 0


def test_combined_tool_takes_a_strip_as_a_strip():
    r = D.call_agent("settlement", "combined_settlement_analysis",
                     {"q_net": 100.0, "B": 2.0, "shape": "strip",
                      "consolidation_layers": [dict(_CLAY)]})
    length = _by_what(r["choices"])["footing length"]
    assert length["how"] == "module" and "plane strain" in length["chosen"]


# -------------------------------------------------------- drilled shaft

def test_a_cohesionless_tip_needs_its_N60():
    layers = [{"thickness": 15, "soil_type": "cohesionless",
               "unit_weight": 19, "phi": 33}]
    r = D.call_agent("drilled_shaft", "drilled_shaft_capacity",
                     {"diameter": 1.0, "length": 12, "layers": layers})
    assert "N60" in _error(r)
    layers[0]["N60"] = 25
    r = D.call_agent("drilled_shaft", "drilled_shaft_capacity",
                     {"diameter": 1.0, "length": 12, "layers": layers})
    assert r["Q_tip_kN"] > 0


# ------------------------------------------------------ aggregate piers

def test_aggregate_piers_state_n_and_the_moduli():
    r = D.call_agent("ground_improvement", "aggregate_piers",
                     {"column_diameter": 0.6, "spacing": 1.8,
                      "S_unreinforced": 100.0})
    c = _by_what(r["choices"])
    n = c["stress concentration ratio n"]
    assert n["chosen"] == "5" and n["how"] == "default"
    assert "GEC-13" in n["basis"]
    assert "column modulus E_c" not in c            # GM-179: refused
    assert c["soil modulus E_s"]["how"] == "default"
    # the defaults the choices call "default" are among defaults_used
    for name in ("n", "E_soil", "pattern"):
        assert name in r["defaults_used"]
    r = D.call_agent("ground_improvement", "aggregate_piers",
                     {"column_diameter": 0.6, "spacing": 1.8, "n": 5.0})
    assert _by_what(r["choices"])["stress concentration ratio n"]["how"] \
        == "given"


# ------------------------------------------------- seismic earth pressure

def test_mo_states_both_readings_of_the_static_KA():
    from seismic_geotech.mononobe_okabe import mononobe_okabe_KAE
    r = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                     {"phi": 34, "kh": 0.15, "i": 10, "beta": 5,
                      "gamma": 19, "H": 6})
    ka = _by_what(r["choices"])["static KA"]
    # the default is unchanged: Rankine from phi
    assert ka["chosen"].startswith("Rankine, 0.2827")
    assert r["KA_static"] == pytest.approx(0.2827, abs=1e-4)
    coulomb = mononobe_okabe_KAE(34, 2.0 / 3.0 * 34, 0.0, 0.0, 5, 10)
    assert f"Coulomb, {coulomb:.4f}" in ka["alternatives"][0]
    assert "beta 5 deg" in ka["alternatives"][0]
    assert f"{coulomb:.4f}" in ka["basis"]
