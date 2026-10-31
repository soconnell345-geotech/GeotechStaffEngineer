"""GM-182: three SOE statements.

* a layer's ``soil_type`` picks the apparent-pressure envelope and used to
  default to sand unstated: braced_excavation and apparent_pressure now
  refuse a layer above the excavation base without one; the cantilever
  does not read it;
* ``FOS_embedment`` and ``Kp`` belonged to the FHWA wall's single-anchor
  route; since GM-290 (GM-193) two or more anchors have an embedment below
  the lowest anchor too, so both routes list, accept and state them;
* the ``gwt_depth_excavation`` default is in ``defaults_used``.
"""

import pytest

from funhouse_agent.dispatch import call_agent

_SAND = {"thickness": 15.0, "unit_weight": 18.0, "friction_angle": 32.0}
_CLAY = {"thickness": 15.0, "unit_weight": 18.0, "friction_angle": 0.0,
         "cohesion": 30.0}
_SUPPORTS = [{"depth": 2.0}, {"depth": 4.5}]


# ---------------------------------------------------------------------------
# soil_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method, extra", [
    ("braced_excavation", {"supports": _SUPPORTS}),
    ("apparent_pressure", {}),
])
def test_a_layer_above_the_base_must_give_its_soil_type(method, extra):
    out = call_agent("soe", method, {"excavation_depth": 6.0,
                                     "layers": [dict(_SAND)], **extra})
    assert "layers[0]" in out["error"] and "soil_type" in out["error"]
    assert "envelope" in out["error"]


def test_a_layer_below_the_base_needs_none():
    out = call_agent("soe", "braced_excavation", {
        "excavation_depth": 6.0, "supports": _SUPPORTS,
        "layers": [dict(_SAND, thickness=6.0, soil_type="sand"),
                   dict(_SAND, thickness=10.0)]})
    assert "error" not in out, out


def test_the_soil_type_given_picks_the_envelope():
    sand = call_agent("soe", "apparent_pressure", {
        "excavation_depth": 6.0, "layers": [dict(_CLAY, soil_type="sand")]})
    clay = call_agent("soe", "apparent_pressure", {
        "excavation_depth": 6.0, "layers": [dict(_CLAY, soil_type="clay")]})
    assert sand["type"] == "sand"
    assert clay["type"] in ("soft_clay", "stiff_clay")


def test_the_cantilever_does_not_read_it():
    base = {"excavation_depth": 4.0}
    left_out = call_agent("soe", "cantilever_excavation",
                          {**base, "layers": [dict(_SAND)]})
    clay = call_agent("soe", "cantilever_excavation",
                      {**base, "layers": [dict(_SAND, soil_type="clay")]})
    assert "error" not in left_out, left_out
    assert left_out["required_embedment_m"] == clay["required_embedment_m"]


def test_the_package_refuses_a_braced_wall_without_it(tmp_path):
    out = call_agent("calc_package", "soe_package", {
        "wall_type": "braced", "excavation_depth": 6.0,
        "layers": [dict(_SAND)], "supports": _SUPPORTS,
        "output_path": str(tmp_path / "soe.html")})
    assert "soil_type" in out["error"]


# ---------------------------------------------------------------------------
# FOS_embedment and Kp: both routes (GM-290; a single anchor's alone before)
# ---------------------------------------------------------------------------

_TWO = {"H": 8.0, "anchor_depths": [2.0, 5.0], "gamma": 18.0, "phi": 32.0}
_ONE = {"H": 6.0, "anchor_depths": [2.0], "gamma": 18.0, "phi": 32.0}


def test_two_anchors_list_both():
    out = call_agent("soe", "fhwa_apparent_pressure_anchored_wall", _TWO)
    assert "error" not in out, out
    assert out["defaults_used"]["FOS_embedment"] == 1.3
    assert out["defaults_used"]["Kp"].startswith("Rankine Kp from phi")
    assert "Ka" in out["defaults_used"]          # both routes read Ka
    how = {c["what"]: c["how"] for c in out["choices"]}
    assert how == {"embedment factor of safety": "default",
                   "passive coefficient below the excavation": "default"}
    assert out["embedment_D_m"] > 0


@pytest.mark.parametrize("name, value", [("FOS_embedment", 1.5),
                                         ("Kp", 4.0)])
def test_two_anchors_take_them(name, value):
    out = call_agent("soe", "fhwa_apparent_pressure_anchored_wall",
                     {**_TWO, name: value})
    assert "error" not in out, out
    assert name not in out.get("defaults_used", {})
    base = call_agent("soe", "fhwa_apparent_pressure_anchored_wall", _TWO)
    assert out["embedment_D_m"] != base["embedment_D_m"]


def test_one_anchor_lists_them_and_states_them():
    out = call_agent("soe", "fhwa_apparent_pressure_anchored_wall", _ONE)
    assert out["defaults_used"]["FOS_embedment"] == 1.3
    assert out["defaults_used"]["Kp"].startswith("Rankine Kp from phi")
    how = {c["what"]: c["how"] for c in out["choices"]}
    assert how == {"embedment factor of safety": "default",
                   "passive coefficient below the excavation": "default"}
    given = call_agent("soe", "fhwa_apparent_pressure_anchored_wall",
                       {**_ONE, "FOS_embedment": 1.3, "Kp": 4.0})
    assert "FOS_embedment" not in given.get("defaults_used", {})
    assert {c["how"] for c in given["choices"]} == {"given"}
    assert given["Kp"] == 4.0


def test_the_package_follows_the_route(tmp_path):
    out = {"wall_type": "fhwa_anchored",
           "output_path": str(tmp_path / "soe.html")}
    two = call_agent("calc_package", "soe_package", {**out, **_TWO})
    one = call_agent("calc_package", "soe_package", {**out, **_ONE})
    assert two["status"] == "success" and one["status"] == "success"
    assert two["defaults_used"]["FOS_embedment"] == 1.3
    assert one["defaults_used"]["FOS_embedment"] == 1.3
    given = call_agent("calc_package", "soe_package",
                       {**out, **_TWO, "FOS_embedment": 1.3})
    assert given["status"] == "success"


# ---------------------------------------------------------------------------
# gwt_depth_excavation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent, method, extra", [
    ("soe", "braced_excavation", {"supports": _SUPPORTS}),
    ("soe", "cantilever_excavation", {}),
    ("calc_package", "soe_package", {"wall_type": "braced",
                                     "supports": _SUPPORTS}),
    ("calc_package", "soe_package", {"wall_type": "cantilever"}),
])
def test_the_excavation_side_water_default_is_listed(agent, method, extra,
                                                    tmp_path):
    layers = [dict(_SAND, soil_type="sand")]
    if agent == "calc_package":
        extra = {**extra, "output_path": str(tmp_path / "soe.html")}
    with_water = call_agent(agent, method, {
        "excavation_depth": 6.0, "layers": layers, "gwt_depth": 3.0, **extra})
    assert with_water["defaults_used"]["gwt_depth_excavation"] == 3.0
    dry = call_agent(agent, method, {
        "excavation_depth": 6.0, "layers": layers, **extra})
    assert dry["defaults_used"]["gwt_depth_excavation"].startswith(
        "no water table")
    given = call_agent(agent, method, {
        "excavation_depth": 6.0, "layers": layers, "gwt_depth": 3.0,
        "gwt_depth_excavation": 7.0, **extra})
    assert "gwt_depth_excavation" not in given["defaults_used"]
