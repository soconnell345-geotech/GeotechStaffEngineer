"""GM-266: mse_wall takes a reinforcement per level (reinforcement_per_level),
as GEC-11 Vol II Example E4 changes its bar mats with depth (Table E4-7.4,
p. E4-18). The module tests are in
retaining_walls/tests/test_mse_per_level_reinforcement.py; E4 against the
table is validation_examples/test_published_v009_v011.py."""

import pytest

from funhouse_agent import dispatch as D

_WALL = dict(wall_height=3.0, reinforcement_spacing=0.75, gamma_backfill=19.0,
             phi_backfill=34.0, reinforcement_horizontal_spacing=1.524)
_PER = ["welded_wire_grid_w11", "bar_mat_e4_3w11_w11x0_5",
        "bar_mat_e4_4w15_w11x1",
        {"Tallowable": 150.0, "type": "bar_mat", "width": 0.4572,
         "thickness": 0.0095, "transverse_spacing": 0.457}]


def _mse(**extra):
    return D.call_agent("retaining_walls", "mse_wall", {**_WALL, **extra})


def _tal(r):
    (c,) = [c for c in r["choices"] if "rupture" in c["what"]]
    return c


def test_each_level_takes_its_entry():
    r = _mse(reinforcement_per_level=_PER)
    assert "error" not in r, r.get("error")
    names = ["Bar mat 4W11 + W11x0.5' (GEC-11 Vol II Example E4)",
             "Bar mat 3W11 + W11x0.5' (GEC-11 Vol II Example E4)",
             "Bar mat 4W15 + W11x1' (GEC-11 Vol II Example E4)",
             "Custom metallic_grid"]
    assert r["reinforcement_name"] == names
    levels = r["internal_results"]
    assert [lv["reinforcement"] for lv in levels] == names
    assert [lv["Tallowable_kN_per_m"] for lv in levels] == pytest.approx(
        [96.44, 108.49, 138.43, 150.0], abs=0.005)
    # Rc = b / Sh, each level's own b
    assert [lv["Rc"] for lv in levels] == [0.3, 0.2, 0.3, 0.3]
    c = _tal(r)
    assert c["what"].endswith("per level")
    assert "Custom metallic_grid: Tal 150 kN/m of mat width at level 4" \
        in c["chosen"]
    assert c["how"] == "given"


def test_built_ins_alone_are_the_modules_values():
    r = _mse(reinforcement_per_level=_PER[:3] + ["bar_mat_e4_4w15_w11x1_5"])
    assert _tal(r)["how"] == "module"
    # the prefix of the single-reinforcement inputs is taken too
    custom = {"reinforcement_Tallowable": 150.0,
              "reinforcement_type": "metallic_grid", "reinforcement_width":
              0.4572, "reinforcement_thickness": 0.0095,
              "reinforcement_transverse_spacing": 0.457}
    r2 = _mse(reinforcement_per_level=_PER[:3] + [custom])
    assert r2["internal_results"] == _mse(
        reinforcement_per_level=_PER)["internal_results"]


def test_one_reinforcement_for_the_wall_is_unchanged():
    r = _mse(reinforcement_name="welded_wire_grid_w11")
    assert r["reinforcement_name"] == \
        "Bar mat 4W11 + W11x0.5' (GEC-11 Vol II Example E4)"
    assert all("reinforcement" not in lv for lv in r["internal_results"])
    assert not _tal(r)["what"].endswith("per level")


def test_a_list_of_the_wrong_length_is_refused_naming_the_levels():
    err = _mse(reinforcement_per_level=_PER[:3])["error"]
    for words in ("gives 3 reinforcements", "this wall has 4 levels",
                  "0.38, 1.12, 1.88, 2.62 m", "reinforcement_spacing 0.75 m"):
        assert words in err, words


@pytest.mark.parametrize("extra, words", [
    ({"reinforcement_name": "welded_wire_grid_w11"},
     "does not take reinforcement_name with reinforcement_per_level"),
    ({"reinforcement_Tallowable": 96.0},
     "does not take reinforcement_Tallowable"),
    ({"reinforcement_horizontal_spacing": [1.5, 1.5]},
     "gives 2 values, one per level from the top"),
])
def test_what_is_refused_beside_it(extra, words):
    assert words in _mse(reinforcement_per_level=_PER, **extra)["error"]


def test_a_list_given_as_reinforcement_name_points_to_it():
    err = _mse(reinforcement_name=["welded_wire_grid_w11"] * 4)["error"]
    assert "for a reinforcement per level give reinforcement_per_level" in err


@pytest.mark.parametrize("entry, words", [
    ({"Tallowble": 3, "type": "bar_mat"},
     "Did you mean: Tallowble -> Tallowable?"),
    ({"Tallowable": 3}, "a custom reinforcement needs its type"),
    ("nope", "'nope' is not a built-in"),
    ({"name": "welded_wire_grid_w11", "Tallowable": 3},
     "gives Tallowable with the built-in 'welded_wire_grid_w11'"),
    ({"name": "welded_wire_grid_w11", "horizontal_spacing": 1.0},
     "reinforcement_horizontal_spacing or coverage_ratio as a list"),
    (3.0, "must be a built-in name or a dict"),
])
def test_an_entry_is_checked_and_named_by_its_level(entry, words):
    err = _mse(reinforcement_per_level=_PER[:3] + [entry])["error"]
    assert "reinforcement_per_level[3] (level 4 from the top)" in err
    assert words in err, err


def test_steel_and_geosynthetic_in_one_wall_are_refused():
    err = _mse(reinforcement_per_level=_PER[:3]
               + ["geogrid_gec11_e1_gg1"])["error"]
    for words in ("p. 4-33", "Fig. 4-9", "steel at levels 1-3",
                  "geosynthetic at level 4"):
        assert words in err, words


def test_geogrid_grades_per_level():
    """Example E1 takes GG-I at Layers 1 to 4 and GG-II below (Table E1-7.4,
    p. E1-16); a geosynthetic per level needs no spacing."""
    r = D.call_agent("retaining_walls", "mse_wall", {
        "wall_height": 3.0, "reinforcement_spacing": 0.75,
        "gamma_backfill": 19.0, "phi_backfill": 34.0,
        "reinforcement_per_level": ["geogrid_gec11_e1_gg1"] * 2
        + ["geogrid_gec11_e1_gg2"] * 2})
    assert "error" not in r, r.get("error")
    levels = r["internal_results"]
    assert {lv["FOS_rupture_required"] for lv in levels} == {1.5}
    assert levels[0]["Tallowable_kN_per_m"] < levels[3][
        "Tallowable_kN_per_m"]


def test_the_tool_text_offers_it():
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    params = METHOD_INFO["mse_wall"]["parameters"]
    spec = params["reinforcement_per_level"]
    assert spec["type"] == "list" and not spec["required"]
    for words in ("Table E4-7.4, p. E4-18", "Fig. 4-9",
                  "a list of another length is refused"):
        assert words in spec["description"], words
    assert "reinforcement_per_level" in params["reinforcement_name"][
        "description"]
