"""GM-273: mse_wall takes the reinforcement levels' depths (level_depths),
as GEC-11 Vol II's examples print Z, each level then over its
contributory height (GEC-11 Vol I pp. 4-46, 4-47), sigma_H averaged over
it as sigma_H_average says. The module tests are in
retaining_walls/tests/test_mse_level_depths.py; E4, E1 and E3 against
their tables are in validation_examples."""

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters.retaining_walls import METHOD_INFO

_WALL = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
             reinforcement_name="ribbed_steel_strip_50x4",
             reinforcement_horizontal_spacing=0.75)
_Z = [0.4, 1.15, 1.9, 2.65, 3.4, 4.15, 4.9, 5.65]


def _mse(**extra):
    return D.call_agent("retaining_walls", "mse_wall", {**_WALL, **extra})


def _choice(r):
    (c,) = [c for c in r["choices"] if "contributory height" in c["what"]]
    return c


def test_the_levels_sit_at_the_depths_given():
    r = _mse(level_depths=_Z)
    assert "error" not in r, r.get("error")
    levels = r["internal_results"]
    assert r["n_levels"] == 8
    assert [lv["depth_m"] for lv in levels] == _Z
    assert [lv["Svt_m"] for lv in levels] == pytest.approx(
        [0.775] + [0.75] * 6 + [0.725], abs=1e-3)
    assert all(lv["Tmax_kN_per_m"] == pytest.approx(
        lv["sigma_H_kPa"] * lv["Svt_m"], abs=0.01) for lv in levels)
    c = _choice(r)
    assert c["how"] == "default"
    assert r["defaults_used"]["sigma_H_average"] == "trapezoidal"
    # placed by the depths, so no spacing and no case for the depths
    assert "reinforcement_spacing" not in r["defaults_used"]
    assert "level_depths" not in r["defaults_used"]


def test_the_alias_and_the_midpoint_average():
    trap = _mse(level_depths=_Z)
    mid = _mse(reinforcement_depths=_Z, sigma_H_average="midpoint")
    assert "error" not in mid, mid.get("error")
    assert _choice(mid)["how"] == "given"
    assert "sigma_H at the midpoint of Svt" in _choice(mid)["chosen"]
    # steel: Kr/Ka changes with depth, so the two averages differ
    assert mid["internal_results"][0]["Tmax_kN_per_m"] != \
        trap["internal_results"][0]["Tmax_kN_per_m"]


def test_without_depths_the_spacing_places_them_and_says_so():
    r = _mse()
    assert r["n_levels"] == 10
    assert "Svt_m" not in r["internal_results"][0]
    assert r["defaults_used"]["reinforcement_spacing"] == 0.6
    assert "sigma_H_average" not in r["defaults_used"]
    case = r["defaults_used"]["level_depths"]
    assert case.startswith("the levels at reinforcement_spacing (i + 1/2)")
    for words in ("the bottom level's contributory height down to the "
                  "base", "p. 4-46", "p. 4-47 and Fig. 4-14",
                  "Give level_depths"):
        assert words in case, words
    # GM-277: the layout's contributory heights, the default
    (c,) = [c for c in r["choices"] if "contributory height" in c["what"]]
    assert c["what"].startswith("Tmax at each level, the levels equally "
                                "spaced at reinforcement_spacing")
    assert c["how"] == "default"
    assert "the bottom level's to its base" in c["chosen"]


def test_without_depths_the_bottom_level_runs_to_the_base():
    """GM-277: 6.32 m at Sv 0.6 m is ten levels, the bottom one at 5.7 m
    carrying the wall from 5.4 m down to the base, 0.92 m; 4.8 m at Sv
    0.8 m is six levels, not the five floating-point rounding gave."""
    r = _mse(wall_height=6.32)
    assert r["n_levels"] == 10
    (c,) = [c for c in r["choices"] if "contributory height" in c["what"]]
    assert ("from the midpoint above it at 5.4 m down to the base at 6.32 "
            "m, 0.92 m") in c["chosen"]
    whole = _mse()
    assert r["internal_results"][-1]["Tmax_kN_per_m"] > \
        whole["internal_results"][-1]["Tmax_kN_per_m"] * 0.92 / 0.6
    short = _mse(wall_height=4.8, reinforcement_spacing=0.8)
    assert short["n_levels"] == 6
    assert short["internal_results"][-1]["depth_m"] == 4.4
    refused = _mse(wall_height=0.5, reinforcement_spacing=1.2)
    assert "at or below the base of the wall" in refused["error"]


@pytest.mark.parametrize("extra, words", [
    (dict(level_depths=_Z, reinforcement_spacing=0.75),
     "takes level_depths or reinforcement_spacing, not both"),
    (dict(level_depths=[1.0, 0.5]),
     "level 2 at z = 0.5 m is not below level 1"),
    (dict(level_depths=[0.5, 6.5]),
     "level 2 at z = 6.5 m is outside the wall"),
    (dict(level_depths=[]), "level_depths takes the depths z"),
    (dict(sigma_H_average="midpoint"),
     "sigma_H_average applies to level depths given"),
    (dict(level_depths=_Z, sigma_H_average="mean"),
     "takes 'trapezoidal'"),
])
def test_what_is_refused(extra, words):
    r = _mse(**extra)
    assert words in r.get("error", ""), r


def test_a_reinforcement_list_needs_one_entry_a_depth():
    wall = {k: v for k, v in _WALL.items() if k != "reinforcement_name"}
    r = D.call_agent("retaining_walls", "mse_wall", {
        **wall, "level_depths": [0.5, 2.0, 4.0],
        "reinforcement_per_level": ["ribbed_steel_strip_50x4"] * 4})
    assert ("reinforcement_per_level gives 4 reinforcements" in r["error"]
            and "has 3 levels, at depths 0.50, 2.00, 4.00 m (level_depths "
            "gives them: give one entry of reinforcement_per_level for each "
            "depth there, or one depth for each entry)" in r["error"]), r
    ok = D.call_agent("retaining_walls", "mse_wall", {
        **wall, "level_depths": [0.5, 2.0, 4.0],
        "reinforcement_per_level": ["ribbed_steel_strip_50x4"] * 3})
    assert "error" not in ok and ok["n_levels"] == 3


@pytest.mark.parametrize("reinforcement", [
    {"reinforcement_name": "geogrid_gec11_e1_gg1"},
    {"reinforcement_per_level": ["geogrid_gec11_e1_gg1"] * 10}])
def test_a_rupture_shortfall_points_at_the_input_in_use(reinforcement):
    # GM-273 review: with the depths given, closer levels are level_depths;
    # reinforcement_spacing would be refused beside them
    wall = dict(wall_height=6.0, gamma_backfill=19.0, phi_backfill=34.0,
                **reinforcement)

    def advice(r):
        (c,) = [c for c in r["choices"] if "Tal" in c["what"]]
        return " ".join(a for a in c["alternatives"] if "closer" in a)

    spaced = D.call_agent("retaining_walls", "mse_wall", wall)
    placed = D.call_agent("retaining_walls", "mse_wall", {
        **wall, "level_depths": [0.3 + 0.6 * i for i in range(10)]})
    assert "a closer vertical spacing (reinforcement_spacing)" in \
        advice(spaced)
    assert "the levels closer together (level_depths)" in advice(placed)
    assert "reinforcement_spacing" not in advice(placed)


def test_the_tool_text_offers_it():
    params = METHOD_INFO["mse_wall"]["parameters"]
    words = params["level_depths"]["description"]
    for w in ("instead of reinforcement_spacing", "Table E4-7.1, p. E4-11",
              "p. 4-46", "p. 4-47", "Svt_m", "Alias: reinforcement_depths"):
        assert w in words, w
    avg = params["sigma_H_average"]
    assert avg["default"] == "trapezoidal"
    assert avg["allowed_values"] == ["trapezoidal", "midpoint"]
    assert "Refused without level_depths" in avg["description"]
    assert "Not taken with level_depths" in \
        params["reinforcement_spacing"]["description"]
    assert avg["applies_when"] == {"level_depths": "set"}
    assert params["reinforcement_spacing"]["applies_when"] == {
        "level_depths": "unset"}
    returns = METHOD_INFO["mse_wall"]["returns"]
    assert "Svt_m" in returns["internal_results"]
    assert "sigma_H_average" in returns["choices"]
