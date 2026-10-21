"""GM-273: the reinforcement levels' depths as an input.

Without them the module places H / Sv levels (rounded down) at Sv (i +
1/2) below the top, each taking Sv and sigma_H at its depth, the bottom
level's contributory height down to the base (GM-277,
test_mse_even_layout.py). GEC-11 gives each level
its contributory height: "the midpoint between vertically adjacent
reinforcement elevations, except for the top and bottom layers
reinforcement" (Vol I p. 4-46), whose Sv "is the distance from top or
bottom of wall, respectively, to the midpoint between the first and second
layer" (p. 4-47), and Tmax is that height times "the average factored
horizontal stress acting upon that contributory height", the average "of
the stress at top and at the bottom of the contributory height or at the
midpoint" (p. 4-47, Fig. 4-14). Vol II Example E4 places Level 1 at Z =
1.87 ft, so its Svt is 3.12 ft (Table E4-7.1, p. E4-11), and averages
sigma_H over Z- and Z+ (p. E4-13). E4, E1 and E3 in one run each, against
their tables, are in validation_examples (test_v010_e4_in_one_run_at_its_
depths, test_v057_e1_in_one_run_at_its_layers, test_v056_e3_at_its_depths).
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    Kr_Ka_ratio, SIGMA_H_NEEDS_DEPTHS, Tmax_at_level, analyze_mse_wall,
    check_internal_stability, contributory_Tmax, level_depths_choices,
)
from retaining_walls.earth_pressure import rankine_Ka
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG1, RIBBED_STEEL_STRIP_50x4, WELDED_WIRE_GRID_W11,
)

STRIP = dataclasses.replace(RIBBED_STEEL_STRIP_50x4, horizontal_spacing=0.75)
DEPTHS = [0.4, 1.15, 1.9, 2.65, 3.4, 4.15, 4.9, 5.65]


def _wall(**kw):
    kw.setdefault("level_depths", DEPTHS)
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0, **kw)


def test_each_level_takes_its_contributory_height():
    """Svt from the midpoints, the top level's from the top of the wall
    and the bottom level's to its base (GEC-11 Vol I pp. 4-46, 4-47)."""
    g = _wall()
    assert g.n_reinforcement_levels == 8
    assert g.reinforcement_depths == DEPTHS
    heights = g.contributory_heights
    assert heights[0] == pytest.approx((0.0, 0.775))
    assert heights[1] == pytest.approx((0.775, 1.525))
    assert heights[-1] == pytest.approx((5.275, 6.0))
    # the heights tile the wall
    assert sum(b - a for a, b in heights) == pytest.approx(6.0)
    rows = check_internal_stability(g, 19.0, 34.0, STRIP)
    assert [r["Svt_m"] for r in rows] == pytest.approx(
        [0.775] + [0.75] * 6 + [0.725], abs=1e-3)
    one = _wall(level_depths=[2.0])
    assert one.contributory_heights == [(0.0, 6.0)]


def test_tmax_is_sigma_H_averaged_over_svt_trapezoidal_by_default():
    """sigma_H the average of its values at the top and the bottom of Svt
    (p. 4-47's tributary trapezoidal area), Kr/Ka at each, as E4 takes
    Level 4: 0.5 (sigma_H(Z-) + sigma_H(Z+)) x Svt (p. E4-13)."""
    g = _wall(surcharge=10.0)
    Ka = rankine_Ka(34.0)

    def T(z, Sv):
        return Tmax_at_level(z, 19.0, Ka, Kr_Ka_ratio(z, "metallic"), Sv,
                             10.0)
    rows = check_internal_stability(g, 19.0, 34.0, STRIP)
    for r, (a, b) in zip(rows, g.contributory_heights):
        expected = 0.5 * (T(a, b - a) + T(b, b - a))
        assert r["Tmax_kN_per_m"] == pytest.approx(expected, abs=0.006)
        assert r["sigma_H_kPa"] == pytest.approx(expected / (b - a),
                                                 abs=0.006)
    assert contributory_Tmax(0.775, 1.525, 19.0, Ka, "metallic", 10.0) == \
        pytest.approx(0.5 * (T(0.775, 0.75) + T(1.525, 0.75)))


def test_the_midpoint_average_takes_sigma_H_at_the_middle_of_svt():
    g = _wall()
    Ka = rankine_Ka(34.0)
    trap = check_internal_stability(g, 19.0, 34.0, STRIP)
    mid = check_internal_stability(g, 19.0, 34.0, STRIP,
                                   sigma_H_average="midpoint")
    for r, (a, b) in zip(mid, g.contributory_heights):
        z = 0.5 * (a + b)
        assert r["Tmax_kN_per_m"] == pytest.approx(Tmax_at_level(
            z, 19.0, Ka, Kr_Ka_ratio(z, "metallic"), b - a), abs=0.006)
    # Kr/Ka changes with depth for steel, so the two differ (the
    # trapezoid lies under the concave sigma_H)
    assert mid[0]["Tmax_kN_per_m"] != trap[0]["Tmax_kN_per_m"]
    # for a geosynthetic Kr/Ka = 1 and sigma_H is linear: they agree
    gg = [check_internal_stability(g, 19.0, 34.0, GEOGRID_GEC11_E1_GG1,
                                   sigma_H_average=a)
          for a in ("trapezoidal", "midpoint")]
    assert [r["Tmax_kN_per_m"] for r in gg[0]] == pytest.approx(
        [r["Tmax_kN_per_m"] for r in gg[1]], abs=0.011)


def test_evenly_spaced_depths_at_a_whole_number_of_spacings_match_sv():
    """At Sv (i + 1/2) with H a whole number of spacings each Svt is Sv
    and, with Kr/Ka constant, sigma_H over it is sigma_H at the level: a
    geosynthetic wall comes out as the module's own layout."""
    even = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.75)
    given = _wall(level_depths=even.reinforcement_depths)
    a = analyze_mse_wall(even, 19.0, 34.0, GEOGRID_GEC11_E1_GG1)
    b = analyze_mse_wall(given, 19.0, 34.0, GEOGRID_GEC11_E1_GG1)
    for ra, rb in zip(a.internal_results, b.internal_results):
        assert rb["Svt_m"] == pytest.approx(0.75)
        assert rb["Tmax_kN_per_m"] == pytest.approx(ra["Tmax_kN_per_m"],
                                                    abs=0.011)


def test_depths_not_increasing_or_outside_the_wall_are_refused():
    with pytest.raises(ValueError, match="must increase from the top down"
                       ".*level 3 at z = 1 m is not below level 2"):
        _wall(level_depths=[0.5, 1.5, 1.0])
    with pytest.raises(ValueError, match="level 2 at z = 1.5 m is not "
                       "below level 1"):
        _wall(level_depths=[1.5, 1.5])
    with pytest.raises(ValueError, match="level 3 at z = 6 m is outside "
                       "the wall.*z < wall_height = 6 m"):
        _wall(level_depths=[0.5, 3.0, 6.0])
    with pytest.raises(ValueError, match="level 1 at z = 0 m is outside"):
        _wall(level_depths=[0.0, 3.0])
    for bad in ([], "0.5, 1.5", 2.0, [0.5, None], [True, 2.0],
                [0.5, float("nan")]):
        with pytest.raises(ValueError, match="level_depths takes the "
                           "depths z"):
            _wall(level_depths=bad)


def test_a_reinforcement_list_of_another_length_is_refused_naming_depths():
    with pytest.raises(ValueError, match=r"gives 3 reinforcements.*has 8 "
                       r"levels, at depths 0\.40, 1\.15.*5\.65 m "
                       r"\(level_depths gives them: give one entry for "
                       r"each depth there, or one depth for each entry\)"):
        analyze_mse_wall(_wall(), 19.0, 34.0, [STRIP] * 3)
    with pytest.raises(ValueError, match=r"gives 2 values of Sh.*\(level_"
                       r"depths gives them"):
        analyze_mse_wall(_wall(), 19.0, 34.0, dataclasses.replace(
            STRIP, horizontal_spacing=(0.75, 0.5)))
    # one a depth runs
    r = analyze_mse_wall(_wall(), 19.0, 34.0, [STRIP] * 8)
    assert r.n_levels == 8


def test_sigma_H_average_needs_the_depths_and_one_of_two_values():
    even = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
    for call in (analyze_mse_wall, check_internal_stability):
        with pytest.raises(ValueError, match="applies to level depths "
                           "given"):
            call(even, 19.0, 34.0, STRIP, sigma_H_average="trapezoidal")
    assert "p. 4-47" in SIGMA_H_NEEDS_DEPTHS
    with pytest.raises(ValueError, match="takes 'trapezoidal'.*or "
                       "'midpoint'.*got 'mean'"):
        analyze_mse_wall(_wall(), 19.0, 34.0, STRIP, sigma_H_average="mean")


def test_without_depths_the_levels_take_sv():
    """No depths: no Svt or sigma_H key and no sigma_H average; each level
    of a wall a whole number of spacings high takes Sv, and the choice
    says so (GM-277, test_mse_even_layout.py)."""
    even = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.75)
    r = analyze_mse_wall(even, 19.0, 34.0, STRIP)
    assert all("Svt_m" not in lv and "sigma_H_kPa" not in lv
               for lv in r.internal_results)
    (c,) = level_depths_choices(even)
    assert c["what"].startswith("Tmax at each level, the levels equally "
                                "spaced")
    assert "sigma_H_average" not in c["chosen"]
    assert [c] == [c for c in r.choices if "contributory" in c["what"]]
    assert [v for ab in even.contributory_heights for v in ab] == \
        pytest.approx([v for i in range(8) for v in (0.75 * i,
                                                     0.75 * (i + 1))])


def test_the_choice_says_svt_and_which_average_with_the_other():
    r = analyze_mse_wall(_wall(), 19.0, 34.0, STRIP)
    (c,) = [c for c in r.choices if "contributory height" in c["what"]]
    assert c["how"] == "default"
    assert ("the top level's from the top of the wall (0.775 m) and the "
            "bottom level's to its base (0.725 m)") in c["chosen"]
    assert "(sigma_H_average 'trapezoidal')" in c["chosen"]
    for words in ("p. 4-46", "p. 4-47", "Fig. 4-14", "E4-7.1, p. E4-11",
                  "E3-7.1, p. E3-13", "E4-13", "E3-15"):
        assert words in c["basis"], words
    assert "sigma_H_average 'midpoint'" in c["alternatives"][0]
    m = analyze_mse_wall(_wall(), 19.0, 34.0, STRIP,
                         sigma_H_average="midpoint")
    (c,) = [c for c in m.choices if "contributory height" in c["what"]]
    assert c["how"] == "given"
    assert "sigma_H at the midpoint of Svt" in c["chosen"]
    assert "sigma_H_average 'trapezoidal'" in c["alternatives"][0]


def test_the_calc_package_shows_the_depths_and_svt():
    g = _wall()
    analysis = {"geom": g, "gamma_backfill": 19.0, "phi_backfill": 34.0,
                "reinforcement": STRIP}
    result = analyze_mse_wall(g, 19.0, 34.0, STRIP)
    items = {i.name: i for i in get_input_summary(result, analysis)}
    assert "S_v" not in items
    assert items["z"].value == "0.40, 1.15, 1.90, 2.65, 3.40, 4.15, 4.90, " \
                               "5.65"
    assert items["N_levels"].value == 8
    assert "average of its values at the top and the bottom" in \
        items["σ_H"].value
    text = repr(get_calc_steps(result, analysis))
    assert "S_vt (m)" in text
    assert "Eq. 4-32a, pp. 4-46, 4-47" in text
    even = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.75)
    one = {"geom": even, "gamma_backfill": 19.0, "phi_backfill": 34.0,
           "reinforcement": STRIP}
    plain = analyze_mse_wall(even, 19.0, 34.0, STRIP)
    items = {i.name: i for i in get_input_summary(plain, one)}
    assert items["S_v"].value == "0.75" and "z" not in items
    assert "S_vt" not in repr(get_calc_steps(plain, one))


def test_a_bar_mat_wall_at_e4s_depths_carries_level_1_as_e4():
    """E4's Level 1 at Z = 1.87 ft takes Svt 3.12 ft (Table E4-7.1),
    sigma_H 0.40 ksf and Tmax 6.25 k a 5 ft panel factored (Table E4-7.4,
    p. E4-18); at Sv (i + 1/2) the module put it at 1.25 ft with Sv 2.5
    ft. The whole of E4 is test_v010_e4_in_one_run_at_its_depths."""
    FT, KSF, KIP = 0.3048, 47.88, 4.448
    g = MSEWallGeometry(wall_height=25.64 * FT, reinforcement_length=18 * FT,
                        surcharge=0.25 * KSF,
                        level_depths=[(1.87 + 2.5 * i) * FT
                                      for i in range(10)])
    mat = dataclasses.replace(WELDED_WIRE_GRID_W11, horizontal_spacing=5 * FT)
    lv = check_internal_stability(g, 125 * 0.157087, 34.0, mat)[0]
    assert lv["Svt_m"] / FT == pytest.approx(3.12, abs=0.005)
    assert 1.35 * lv["sigma_H_kPa"] / KSF == pytest.approx(0.40, abs=0.005)
    assert 1.35 * lv["Tmax_kN_per_m"] * 5 * FT / KIP == pytest.approx(
        6.25, rel=0.003)
