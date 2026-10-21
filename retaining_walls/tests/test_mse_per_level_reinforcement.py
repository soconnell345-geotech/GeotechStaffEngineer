"""GM-266: a different reinforcement per level.

GEC-11 Vol II Example E4 (FHWA-NHI-10-025) changes its bar mats with depth:
4W11 + W11x0.5' at Levels 1, 3 and 4, 3W11 + W11x0.5' at Level 2, 4W15 +
W11x1' at Levels 5 to 7 and 4W15 + W11x1.5' at Levels 8 to 10 (Table
E4-7.4, p. E4-18). Before GM-266 the module took one reinforcement for the
wall. One reinforcement holds at every level, as before; a list gives one
a level from the top down, each level's pullout and rupture checks taking
its own Tal, width, Rc, F* and alpha. E4 in one run, against the table, is
validation_examples/test_published_v009_v011.py (test_v010_e4_mats_per_level).
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, level_reinforcements,
)
from retaining_walls.reinforcement import (
    BAR_MAT_E4_3W11, BAR_MAT_E4_4W15_ST1, BAR_MAT_E4_4W15_ST1_5,
    GEOGRID_GEC11_E1_GG1, GEOGRID_GEC11_E1_GG2, RIBBED_STEEL_STRIP_50x4,
    WELDED_WIRE_GRID_W11, Reinforcement,
)

FT = 0.3048


def _wall():
    """Four levels at Sv 0.75 m (0.375 to 2.625 m)."""
    return MSEWallGeometry(wall_height=3.0, reinforcement_length=2.5,
                           reinforcement_spacing=0.75)


def _sh(r, Sh=5 * FT):
    return dataclasses.replace(r, horizontal_spacing=Sh)


_MATS = [_sh(WELDED_WIRE_GRID_W11), _sh(BAR_MAT_E4_3W11),
         _sh(BAR_MAT_E4_4W15_ST1), _sh(BAR_MAT_E4_4W15_ST1_5)]


def test_e4s_mats_are_built_ins_as_printed():
    """Table E4-7.4 (p. E4-18) with p. E4-14 (D75 0.318 in. W11, 0.381 in.
    W15), p. E4-15 (Fy 65 ksi, t 0.374 in.) and p. E4-17 (Sl 0.5 ft): Tal =
    0.48 Fy Ac N / b (GEC-11 Vol I Eq. 3-11a, p. 3-22; p. 4-77)."""
    assert BAR_MAT_E4_3W11.width == pytest.approx(1.0 * FT)
    assert BAR_MAT_E4_4W15_ST1.width == pytest.approx(1.5 * FT)
    # 3 W11 wires over 1 ft carry what 4 carry over 1.5 ft
    assert BAR_MAT_E4_3W11.Tallowable * BAR_MAT_E4_3W11.width == \
        pytest.approx(0.75 * WELDED_WIRE_GRID_W11.Tallowable
                      * WELDED_WIRE_GRID_W11.width, rel=1e-9)
    assert BAR_MAT_E4_4W15_ST1.Tallowable == pytest.approx(138.4, abs=0.05)
    assert BAR_MAT_E4_4W15_ST1_5.Tallowable == \
        BAR_MAT_E4_4W15_ST1.Tallowable
    # the transverse W11 at 1 ft and 1.5 ft
    assert BAR_MAT_E4_4W15_ST1.t_over_St == pytest.approx(0.374 / 12,
                                                          abs=2e-4)
    assert BAR_MAT_E4_4W15_ST1_5.t_over_St == pytest.approx(0.374 / 18,
                                                            abs=2e-4)
    for r in (BAR_MAT_E4_3W11, BAR_MAT_E4_4W15_ST1, BAR_MAT_E4_4W15_ST1_5):
        assert r.type == "metallic_grid"
        assert "Table E4-7.4, p. E4-18" in r.basis


def test_each_level_takes_its_own_reinforcement():
    levels = check_internal_stability(_wall(), 19.0, 34.0, _MATS)
    for i, r in enumerate(_MATS):
        alone = check_internal_stability(_wall(), 19.0, 34.0, r)[i]
        assert levels[i] == dict(alone, reinforcement=r.name)
    assert [lv["reinforcement"] for lv in levels] == [r.name for r in _MATS]
    assert [lv["Tallowable_kN_per_m"] for lv in levels] == pytest.approx(
        [96.44, 108.49, 138.43, 138.43], abs=0.005)


def test_one_reinforcement_holds_at_every_level_as_before():
    one = check_internal_stability(_wall(), 19.0, 34.0, _MATS[0])
    assert one == [{k: v for k, v in lv.items() if k != "reinforcement"}
                   for lv in check_internal_stability(_wall(), 19.0, 34.0,
                                                      [_MATS[0]] * 4)]
    assert all("reinforcement" not in lv for lv in one)


def test_a_strip_and_a_mat_level_take_their_own_curves():
    """Kr/Ka 1.7 -> 1.2 for strips, 2.5 -> 1.2 for bar mats; F* 1.2 + log
    Cu -> tan phi for strips, 20 t/St -> 10 t/St for grids (GEC-11 Vol I
    Figs. 3-9 and 3-11)."""
    rs = [_sh(RIBBED_STEEL_STRIP_50x4, 0.75)] * 2 + _MATS[2:]
    levels = check_internal_stability(_wall(), 19.0, 34.0, rs)
    strip = check_internal_stability(_wall(), 19.0, 34.0, rs[0])
    mat = check_internal_stability(_wall(), 19.0, 34.0, rs[2])
    assert levels[0]["Kr_Ka"] == strip[0]["Kr_Ka"] < 1.7
    assert levels[2]["Kr_Ka"] == mat[2]["Kr_Ka"] > 2.0
    assert levels[1]["F_star"] == strip[1]["F_star"]
    assert levels[2]["F_star"] == mat[2]["F_star"]


def test_a_list_of_the_wrong_length_is_refused_naming_the_levels():
    with pytest.raises(ValueError) as e:
        analyze_mse_wall(_wall(), 19.0, 34.0, _MATS[:3])
    for words in ("gives 3 reinforcements", "this wall has 4 levels",
                  "0.38, 1.12, 1.88, 2.62 m"):
        assert words in str(e.value), words


def test_steel_and_geosynthetic_levels_in_one_wall_are_refused():
    rs = _MATS[:2] + [GEOGRID_GEC11_E1_GG1, GEOGRID_GEC11_E1_GG2]
    with pytest.raises(ValueError) as e:
        analyze_mse_wall(_wall(), 19.0, 34.0, rs)
    for words in ("p. 4-33", "Fig. 4-9", "steel at levels 1-2",
                  "geosynthetic at levels 3-4"):
        assert words in str(e.value), words


def test_what_else_is_refused_per_level():
    with pytest.raises(ValueError, match="one Reinforcement for every level"):
        level_reinforcements(["welded_wire_grid_w11"] * 4, [0.4] * 4)
    with pytest.raises(ValueError, match="level 2's reinforcement"):
        level_reinforcements(
            [_MATS[0], _sh(_MATS[1], [1.5] * 4)] + _MATS[2:], [0.4] * 4)


def test_geosynthetic_grades_per_level_as_e1():
    """Example E1 takes GG-I at Layers 1 to 4 and GG-II below (Table
    E1-7.4, p. E1-16): each level's rupture check takes its own grade."""
    rs = [GEOGRID_GEC11_E1_GG2] * 2 + [GEOGRID_GEC11_E1_GG1] * 2
    levels = check_internal_stability(_wall(), 19.0, 34.0, rs)
    assert [lv["Tallowable_kN_per_m"] for lv in levels] == pytest.approx(
        [GEOGRID_GEC11_E1_GG2.Tallowable] * 2
        + [GEOGRID_GEC11_E1_GG1.Tallowable] * 2, abs=0.01)
    assert {lv["FOS_rupture_required"] for lv in levels} == {1.5}


def _tal(result):
    (c,) = [c for c in result.choices if "rupture" in c["what"]]
    return c


def test_the_choice_gives_tal_by_runs_of_levels():
    rs = [_MATS[0], _MATS[1], _MATS[0], _MATS[2]]
    c = _tal(analyze_mse_wall(_wall(), 19.0, 34.0, rs))
    assert c["what"].endswith("per level")
    assert c["chosen"].startswith("Tal per level, from the top: ")
    for words in ("4W11 + W11x0.5'", "at levels 1, 3", "3W11 + W11x0.5'",
                  "at level 2", "4W15 + W11x1'", "at level 4"):
        assert words in c["chosen"], words
    assert c["how"] == "module"
    custom = Reinforcement(name="Mat X", type="metallic_grid",
                           Tallowable=120.0, width=0.4572,
                           thickness=0.0095, transverse_spacing=0.3,
                           horizontal_spacing=5 * FT)
    c = _tal(analyze_mse_wall(_wall(), 19.0, 34.0, rs[:3] + [custom]))
    assert c["how"] == "given"
    assert "Mat X: Tal 120 kN/m of mat width at level 4" in c["chosen"]


def test_the_summary_names_each_levels_reinforcement():
    lines = [line for line in analyze_mse_wall(
        _wall(), 19.0, 34.0, _MATS).summary().splitlines() if "z=" in line]
    assert len(lines) == 4
    for line, r in zip(lines, _MATS):
        assert f"m {r.name}: Tmax=" in line, line
    one = analyze_mse_wall(_wall(), 19.0, 34.0, _MATS[0]).summary()
    assert "Example E4): Tmax" not in one


def test_the_calc_package_keys_each_level_to_its_reinforcement():
    rs = [_MATS[0], _MATS[1], _MATS[0], _MATS[2]]
    analysis = {"geom": _wall(), "gamma_backfill": 19.0, "phi_backfill": 34.0,
                "reinforcement": rs}
    result = analyze_mse_wall(_wall(), 19.0, 34.0, rs)
    items = {i.name: i for i in get_input_summary(result, analysis)}
    assert items["Reinf."].value.startswith("A, B, A, C (A = Bar mat 4W11")
    assert "T_al 96.44 kN/m" in items["Reinf."].value
    assert items["T_al"].value == "96.4, 108.5, 96.4, 138.4"
    assert items["T_al"].description.endswith("per level from the top")
    assert items["R_c"].value == "0.300, 0.200, 0.300 × 2"
    text = repr(get_calc_steps(result, analysis))
    assert "'Reinf.'" in text
    assert "Reinforcement per level: A = " in text
    one = {"geom": _wall(), "gamma_backfill": 19.0, "phi_backfill": 34.0,
           "reinforcement": _MATS[0]}
    items = {i.name: i for i in get_input_summary(
        analyze_mse_wall(_wall(), 19.0, 34.0, _MATS[0]), one)}
    # one reinforcement: its name, as before
    assert items["Reinf."].value == _MATS[0].name
