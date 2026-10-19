"""BACKLOG GM-144 through the agent's own call path: axial_pile_capacity
takes a custom pile section again (the calc package took one before
GM-141), capacity_vs_depth takes the GEC-12 options axial_pile_capacity
takes, and the new options and layer keys reach the module."""

import pytest

from funhouse_agent import dispatch as D

_SAND = [{"thickness": 30, "soil_type": "cohesionless", "unit_weight": 19,
          "friction_angle": 36}]
_SAND_CLAY = [{"thickness": 5, "soil_type": "cohesionless", "unit_weight": 18,
               "friction_angle": 32},
              {"thickness": 20, "soil_type": "cohesive", "unit_weight": 19,
               "cohesion": 100}]


def _call(method, params):
    return D.call_agent("axial_pile", method, params)


def _by_what(r):
    return {c["what"]: c for c in r["choices"]}


# ----------------------------------------------------------- custom section

def test_a_custom_h_section_runs_as_the_database_shape_does():
    """HP12x74 by its properties (box perimeter and box toe area) gives the
    designation's run."""
    from axial_pile import make_h_pile
    hp = make_h_pile("HP12x74")
    by_name = _call("axial_pile_capacity", {
        "pile_type": "h_pile", "designation": "HP12x74", "pile_length": 20,
        "layers": _SAND})
    custom = _call("axial_pile_capacity", {
        "pile_type": "h_pile", "pile_area": hp.area,
        "pile_perimeter": hp.perimeter, "pile_tip_area": hp.tip_area,
        "pile_width": hp.width, "pile_depth": hp.depth,
        "pile_name": "HP12x74 by its properties", "pile_length": 20,
        "layers": _SAND})
    assert "error" not in custom, custom
    for key in ("Q_ultimate_kN", "Q_skin_kN", "Q_tip_kN"):
        assert custom[key] == by_name[key]
    assert custom["pile_name"] == "HP12x74 by its properties"


def test_a_solid_custom_section_takes_its_area_at_the_toe():
    square = _call("axial_pile_capacity", {
        "pile_type": "concrete_square", "width": 0.4, "pile_length": 15,
        "layers": _SAND_CLAY})
    custom = _call("axial_pile_capacity", {
        "pile_type": "concrete_square", "pile_area": 0.16,
        "pile_perimeter": 1.6, "pile_width": 0.4, "pile_length": 15,
        "layers": _SAND_CLAY})
    assert custom["Q_ultimate_kN"] == pytest.approx(square["Q_ultimate_kN"])
    # a timber pile is reached through a custom section (Fig. 7-9 curve b)
    timber = _call("axial_pile_capacity", {
        "pile_type": "timber", "pile_area": 0.0707, "pile_perimeter": 0.942,
        "pile_width": 0.3, "pile_length": 12, "layers": _SAND})
    assert "error" not in timber, timber
    dphi = _by_what(timber)["Nordlund delta/phi"]["chosen"]
    assert "curve b, timber piles" in dphi


@pytest.mark.parametrize("params, said", [
    ({"pile_type": "h_pile", "pile_area": 0.0141, "pile_perimeter": 1.24,
      "pile_width": 0.31}, "pile_tip_area"),
    ({"pile_type": "pipe_open", "pile_area": 0.0141, "pile_perimeter": 1.24,
      "pile_width": 0.4}, "diameter and wall_thickness"),
    ({"pile_type": "h_pile", "designation": "HP12x74", "pile_area": 0.0141,
      "pile_perimeter": 1.24, "pile_width": 0.31, "pile_tip_area": 0.09},
     "not both"),
    ({"pile_type": "timber"}, "custom section"),
    ({"pile_type": "concrete_square", "pile_area": 0.16,
      "pile_perimeter": 0.0, "pile_width": 0.4}, "pile_perimeter must be > 0"),
])
def test_a_custom_section_says_what_it_lacks(params, said):
    r = _call("axial_pile_capacity", {**params, "pile_length": 10,
                                      "layers": _SAND})
    assert said in r["error"]


def test_make_pile_section_reports_a_custom_section():
    r = _call("make_pile_section", {
        "pile_type": "pipe_closed", "pile_area": 0.0113,
        "pile_perimeter": 1.885, "pile_tip_area": 0.2827, "pile_width": 0.6})
    assert r["tip_area_m2"] == pytest.approx(0.2827)
    assert r["pile_type"] == "pipe_closed"


# --------------------------------------------------- the new GEC-12 options

def test_the_gm144_options_reach_the_module():
    clay = {"pile_type": "concrete_square", "width": 0.4, "pile_length": 15,
            "layers": _SAND_CLAY}
    base = _call("axial_pile_capacity", dict(clay))
    for name, value in (("adhesion_chart", "fig_7_18"),
                        ("shrink_swell_depth", 7.0)):
        r = _call("axial_pile_capacity", {**clay, name: value})
        assert r["Q_skin_kN"] != base["Q_skin_kN"], name
    beta = {**clay, "method": "beta"}
    r = _call("axial_pile_capacity", {**beta,
                                      "beta_coefficients": "table_7_9"})
    assert r["Q_skin_kN"] != _call("axial_pile_capacity",
                                   dict(beta))["Q_skin_kN"]
    layers = [dict(_SAND_CLAY[0]),
              dict(_SAND_CLAY[1], fig_7_18_case="stiff_clay")]
    r = _call("axial_pile_capacity", {**clay, "layers": layers,
                                      "adhesion_chart": "fig_7_18"})
    case = _by_what(r)["Fig. 7-18 stratigraphy case"]
    assert case["how"] == "given" and "bottom graph" in case["chosen"]
    layers = [dict(_SAND_CLAY[0], soil_class="gravel"), dict(_SAND_CLAY[1])]
    r = _call("axial_pile_capacity", {**beta, "layers": layers,
                                      "beta_coefficients": "table_7_9"})
    assert "gravel" in _by_what(r)["beta and Nt (beta method)"]["chosen"]


def test_the_new_defaults_are_listed_where_the_run_uses_them():
    sand = _call("axial_pile_capacity", {
        "pile_type": "pipe_closed", "diameter": 0.6, "wall_thickness": 0.012,
        "pile_length": 15, "layers": _SAND})
    assert sand["defaults_used"]["toe_phi_zone"] == "toe_layer"
    assert not {"adhesion_chart", "beta_coefficients"} & set(
        sand["defaults_used"])
    beta = _call("axial_pile_capacity", {
        "pile_type": "pipe_closed", "diameter": 0.6, "wall_thickness": 0.012,
        "pile_length": 15, "layers": _SAND_CLAY, "method": "beta"})
    assert beta["defaults_used"]["beta_coefficients"] == "from_phi"
    assert "adhesion_chart" not in beta["defaults_used"]


# --------------------------------------------- capacity_vs_depth (item 5)

def test_capacity_vs_depth_takes_the_gec12_options():
    """capacity_vs_depth took the module's defaults only (GM-121's note):
    now each option changes its curve as it changes a single run."""
    sand = {"pile_type": "pipe_open", "diameter": 0.6,
            "wall_thickness": 0.012, "layers": _SAND, "depth_min": 10,
            "depth_max": 20, "n_points": 3}
    base = _call("capacity_vs_depth", dict(sand))
    assert "error" not in base, base
    assert base["choices"]

    def curve(r):
        return [p["Q_ultimate_kN"] for p in r["capacity_vs_depth"]]
    closed = {**sand, "pile_type": "pipe_closed",
              "layers": [dict(_SAND[0], friction_angle=38)]}
    two = [dict(_SAND[0], thickness=19), dict(_SAND[0], friction_angle=40)]
    for call, name, value in ((closed, "toe_stress_limit", None),
                              (sand, "inside_friction_ratio", 0.5),
                              (sand, "open_pipe_toe_limit", None),
                              (sand, "displaced_volume", 0.2),
                              ({**closed, "layers": two, "depth_min": 17.5,
                                "depth_max": 18.5}, "toe_phi_zone",
                               "average_3b")):
        r = _call("capacity_vs_depth", {**call, name: value})
        assert "error" not in r, (name, r)
        assert curve(r) != curve(_call("capacity_vs_depth", dict(call))), \
            name
    clay = {**sand, "pile_type": "concrete_square", "width": 0.4,
            "layers": _SAND_CLAY}
    clay.pop("diameter"), clay.pop("wall_thickness")
    for name, value in (("adhesion_chart", "smaller"),
                        ("shrink_swell_depth", 8.0),
                        ("adhesion_curve", "smooth_steel")):
        r = _call("capacity_vs_depth", {**clay, name: value})
        assert curve(r) != curve(_call("capacity_vs_depth", dict(clay))), name
    # the beta method's cohesive_phi and Table 7-9
    beta = {**clay, "method": "beta"}
    for name, value in (("cohesive_phi", 30.0),
                        ("beta_coefficients", "table_7_9")):
        r = _call("capacity_vs_depth", {**beta, name: value})
        assert curve(r) != curve(_call("capacity_vs_depth", dict(beta))), name
    assert "cohesive_phi" in _call("capacity_vs_depth",
                                   dict(beta))["defaults_used"]


def test_capacity_vs_depth_takes_a_custom_section():
    r = _call("capacity_vs_depth", {
        "pile_type": "timber", "pile_area": 0.0707, "pile_perimeter": 0.942,
        "pile_width": 0.3, "layers": _SAND, "n_points": 2})
    assert "error" not in r and len(r["capacity_vs_depth"]) == 2
    r = _call("capacity_vs_depth", {"layers": _SAND, "diameter": 0.6})
    assert "pile_type" in r["error"]
