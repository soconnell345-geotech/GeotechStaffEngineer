"""The driven-pile eval keys (AP-1, AP-2) against the tool (GM-238).

Each key is a hand derivation from GEC-12 as printed, recorded in its
``notes``. The tool is run here on the question's inputs: a value outside
the key's ``rtol`` means the tool and the hand derivation now disagree.
Re-derive the key by hand and find out why; never copy the tool's number
into the key.
"""

import json
import os

import pytest

_SUITE = os.path.join(os.path.dirname(__file__), os.pardir,
                      "geotech_test_suite.json")

# The question's inputs as the hand derivation read them (see each key's
# notes): a circular section, the head at the ground surface, the soil
# continuing below the toe.
_INPUTS = {
    "AP-1": {"pile_type": "concrete_circular", "width": 0.4,
             "pile_length": 15.0, "gwt_depth": 3.0,
             "layers": [{"thickness": 30.0, "soil_type": "cohesionless",
                         "unit_weight": 19.0, "friction_angle": 32.0}]},
    "AP-2": {"pile_type": "concrete_circular", "width": 0.45,
             "pile_length": 18.0,
             "layers": [{"thickness": 30.0, "soil_type": "cohesive",
                         "unit_weight": 18.0, "cohesion": 60.0}]},
}


def _key(qid):
    with open(_SUITE, encoding="utf-8") as f:
        q = {q["id"]: q for q in json.load(f)}[qid]
    return q["expected"]


@pytest.mark.parametrize("qid", sorted(_INPUTS))
def test_key_is_a_hand_derivation(qid):
    exp = _key(qid)
    assert "hand derivation" in exp["source"]
    assert "GEC-12" in exp["source"]
    assert "Step 1" in exp["notes"]


@pytest.mark.parametrize("qid", sorted(_INPUTS))
def test_tool_agrees_with_the_hand_derivation(qid):
    from funhouse_agent.dispatch import call_agent
    out = call_agent("axial_pile", "axial_pile_capacity", _INPUTS[qid])
    if isinstance(out, str):
        out = json.loads(out)
    assert "error" not in out, out.get("error")
    entry = _key(qid)["values"][0]
    got, value, rtol = out["Q_ultimate_kN"], entry["value"], entry["rtol"]
    assert abs(got - value) <= rtol * value, (
        f"{qid}: the tool gives {got} kN, the hand-derived key {value} kN "
        f"(rtol {rtol})")
