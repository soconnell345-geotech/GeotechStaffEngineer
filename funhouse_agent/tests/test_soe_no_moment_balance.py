"""GM-20: the SOE tools say plainly when free earth support has no solution.

A 6 m cut in soft clay (phi = 0, c = 20 kPa, q = 10 kPa) has no moment
balance at any embedment; the adapter must hand the agent nulls and a
warning, not the 6H search limit dressed up as a design depth.
"""

import json

from funhouse_agent.dispatch import call_agent

_CLAY = [{"thickness": 40, "unit_weight": 18, "friction_angle": 0,
          "cohesion": 20, "soil_type": "soft_clay"}]


def test_cantilever_excavation_reports_no_solution():
    out = call_agent("soe", "cantilever_excavation",
                     {"excavation_depth": 6.0, "layers": _CLAY})
    assert "error" not in out
    assert out["converged"] is False
    assert out["required_embedment_m"] is None
    assert out["max_moment_kNm_per_m"] is None
    assert "cannot stand as a cantilever" in out["notes"][0]
    json.dumps(out)


def test_braced_excavation_reports_no_solution():
    out = call_agent("soe", "braced_excavation",
                     {"excavation_depth": 6.0, "layers": _CLAY,
                      "supports": [{"depth": 2.0}], "FOS_passive": 1.3})
    assert "error" not in out
    assert out["converged"] is False
    assert out["required_embedment_m"] is None
    assert out["free_earth"]["embedment_D_FS1_m"] is None
    # the hinge method's single support takes free earth support's load,
    # not found here (GM-290); its load from above is
    assert out["support_reactions"][0]["load_kN_per_m"] is None
    assert out["support_reactions"][0]["from_above_kN_per_m"] > 0
    json.dumps(out)


def test_single_anchor_wall_reports_no_solution():
    out = call_agent("soe", "fhwa_apparent_pressure_anchored_wall",
                     {"H": 6.0, "anchor_depths": [2.0], "gamma": 18.0,
                      "phi": 5.0, "Kp": 1.0})
    assert "error" not in out
    assert out["converged"] is False
    assert out["anchors"][0]["TH_kN_per_m"] is None
    assert out["embedment_D_m"] is None
    json.dumps(out)
