"""GM-251, GM-252: the drilled shaft base through the tools.

A rock layer takes ``GSI`` and ``m_i``, the inputs of GEC-10's base for a
fractured rock mass (Eqs. 10-24 to 10-29, pp. 10-23, 10-24); lrfd_capacity
states the resistance factor for it, 0.50 (p. 10-24; Table 8-4, p. 8-14,
'Eq. 10-23 or Eq. 10-29'). The clay base's su is the mean over 2B below the
base (p. 10-19).
"""

import json

import pytest

from funhouse_agent import dispatch as D

_ROCK = {
    "diameter": 1.2, "shaft_length": 14.0,
    "layers": [{"thickness": 8.0, "soil_type": "cohesive",
                "unit_weight": 18.5, "cu": 75.0},
               {"thickness": 10.0, "soil_type": "rock", "unit_weight": 23.0,
                "qu": 20000.0, "RQD": 40.0, "GSI": 45.0, "m_i": 10.0}]}


def _call(method, params):
    r = D.call_agent("drilled_shaft", method, params)
    return json.loads(r) if isinstance(r, str) else r


def test_gsi_and_m_i_reach_the_base_and_its_factor():
    r = _call("lrfd_capacity", _ROCK)
    assert "error" not in r, r
    (base,) = [c for c in r["choices"]
               if c["what"] == "base resistance in rock"
               and str(c["chosen"]).startswith("GEC-10 (2018) Eqs. 10-24")]
    (phi,) = [c for c in r["choices"] if c["what"]
              == "resistance factor for base resistance in rock"]
    assert phi["chosen"] == 0.5
    assert "Equation 10-29 is phi = 0.50" in phi["basis"]
    assert "'Eq. 10-23 or Eq. 10-29'" in phi["basis"]
    assert r["lrfd"]["phi_tip"] == 0.5


def test_without_them_the_refusal_names_them():
    layers = [dict(_ROCK["layers"][0]),
              {k: v for k, v in _ROCK["layers"][1].items()
               if k not in ("GSI", "m_i")}]
    r = _call("drilled_shaft_capacity", {**_ROCK, "layers": layers})
    assert "error" in r
    assert "GSI" in r["error"] and "m_i" in r["error"]


def test_a_misspelt_mi_is_caught():
    layers = [dict(_ROCK["layers"][0]),
              {**{k: v for k, v in _ROCK["layers"][1].items()
                  if k != "m_i"}, "mi": 10.0}]
    r = _call("drilled_shaft_capacity", {**_ROCK, "layers": layers})
    assert "error" in r and "m_i" in r["error"]


def test_the_clay_base_su_over_2b_through_the_tool():
    r = _call("drilled_shaft_capacity", {
        "diameter": 1.0, "shaft_length": 12.0, "gwt_depth": 2.0,
        "layers": [{"thickness": 13.0, "soil_type": "cohesive",
                    "unit_weight": 18.0, "cu": 100.0},
                   {"thickness": 12.0, "soil_type": "cohesive",
                    "unit_weight": 17.0, "cu": 40.0}]})
    assert "error" not in r, r
    assert r["base_steps"]["su_kPa"] == pytest.approx(70.0)


def test_gsi_on_a_layer_no_base_takes_is_refused():
    """GM-251 review: GSI and m_i on the socket rock above the base were
    accepted and dropped; refused as GM-179 refuses an unused input."""
    layers = [dict(_ROCK["layers"][0]),
              {"thickness": 3.0, "soil_type": "rock", "unit_weight": 23.0,
               "qu": 20000.0, "GSI": 30.0, "m_i": 7.0},
              {"thickness": 7.0, "soil_type": "rock", "unit_weight": 23.0,
               "qu": 20000.0}]
    for method in ("drilled_shaft_capacity", "lrfd_capacity"):
        r = _call(method, {**_ROCK, "layers": layers})
        assert "does not take GSI and m_i" in r.get("error", ""), r


def test_conditions_left_out_at_rqd_100_take_eq_10_23():
    """GM-251 review: with the rock at its default RQD 100 and GSI and m_i
    given, Eq. 10-23 as GEC-10 p. 10-22 recommends, the Hoek-Brown base
    beside it."""
    layers = [dict(_ROCK["layers"][0]),
              {k: v for k, v in _ROCK["layers"][1].items() if k != "RQD"}]
    r = _call("drilled_shaft_capacity", {**_ROCK, "layers": layers})
    assert r["base_steps"]["equation"] == "10-23"
    (base,) = [c for c in r["choices"]
               if c["what"] == "base resistance in rock"]
    assert any("GSI 45 and m_i 10" in a for a in base["alternatives"])
