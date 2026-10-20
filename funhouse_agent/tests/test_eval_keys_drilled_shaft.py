"""The drilled-shaft eval keys (DS-1, DS-2) against the tool (GM-241).

DS-2's key is a hand derivation from GEC-10 (2018) as printed, recorded in
its ``notes``. DS-1's is still the module value of 2026-06-10 (AASHTO
alpha 0.55; GEC-10 (2018) Eq. 10-18 gives about 6 % less, inside its
tolerance). The tool is run here on each question's inputs: a value
outside the key's ``rtol`` means the tool and the key now disagree.
Re-derive the key by hand and find out why; never copy the tool's number
into the key.
"""

import json
import os

import pytest

_SUITE = os.path.join(os.path.dirname(__file__), os.pardir,
                      "geotech_test_suite.json")

# The question's inputs as the key read them: the soil continuing below
# the base, no water table; DS-2's N60 = 15 along the shaft as at the tip.
_INPUTS = {
    "DS-1": {"diameter": 1.0, "shaft_length": 12.0,
             "layers": [{"thickness": 30.0, "soil_type": "cohesive",
                         "unit_weight": 18.0, "cu": 80.0}]},
    "DS-2": {"diameter": 1.2, "shaft_length": 15.0,
             "layers": [{"thickness": 30.0, "soil_type": "cohesionless",
                         "unit_weight": 19.0, "phi": 34.0, "N60": 15.0}]},
}

# The key's names for the tool's results.
_RESULT = {"Q_side_kN": "Q_skin_kN", "Q_base_kN": "Q_tip_kN",
           "Q_ult_kN": "Q_ultimate_kN"}


def _key(qid):
    with open(_SUITE, encoding="utf-8") as f:
        q = {q["id"]: q for q in json.load(f)}[qid]
    return q["expected"]


def test_ds2_key_is_a_hand_derivation():
    exp = _key("DS-2")
    assert "hand derivation" in exp["source"]
    assert "GEC-10 (FHWA-NHI-18-024)" in exp["source"]
    assert "Step 5" in exp["notes"] and "Eq. 10-13" in exp["notes"]


@pytest.mark.parametrize("qid", sorted(_INPUTS))
def test_tool_agrees_with_the_key(qid):
    from funhouse_agent.dispatch import call_agent
    out = call_agent("drilled_shaft", "drilled_shaft_capacity", _INPUTS[qid])
    if isinstance(out, str):
        out = json.loads(out)
    assert "error" not in out, out.get("error")
    for entry in _key(qid)["values"]:
        got = out[_RESULT[entry["name"]]]
        value, rtol = entry["value"], entry["rtol"]
        assert abs(got - value) <= rtol * value, (
            f"{qid} {entry['name']}: the tool gives {got} kN, the key "
            f"{value} kN (rtol {rtol})")
