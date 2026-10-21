"""GM-256: the coverage ratio Rc = b/Sh, and F* and alpha from pullout tests.

GEC-11 Vol I (FHWA-NHI-10-024), read on the pages:
  - Eq. 3-1 (p. 3-12): Rc = b/Sh, b "the gross width of the strip, sheet,
    or grid", Sh the "center-to-center horizontal spacing"; p. 3-13: "Rc = 1
    in the case of continuous reinforcement";
  - Eq. 4-35 (p. 4-49): pullout per unit width of wall, F* alpha sigma_v Le
    C Rc; p. 4-47: the tensile resistance is "the product of the nominal
    long-term strength, coverage ratio, and applicable resistance factor",
    Tal per unit width of reinforcement (Eq. 3-11a, p. 3-22);
  - Table 3-6 note (i) (p. 3-17): the empirical F* "in the absence of
    product and backfill specific test data"; p. 3-16: alpha "can be
    obtained from pullout tests".
Before GM-256 every reinforcement took Rc = 1: a 9 m wall of 75 mm strips
showed FOS_pullout 28.8 to 42.1, and the strip's 43.1 kN was compared with
the tension per metre of wall at no spacing. Since GM-267 the built-in
strip is GEC-11 Example E3's 50 mm strip, so the strip wall's pins below are
2/3 of the 75 mm strip's.
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, coverage_choice,
    coverage_ratio, F_star_from_tests,
)
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG1, RIBBED_STEEL_STRIP_50x4, Reinforcement,
    WELDED_WIRE_GRID_W11,
)


def _strip_wall():
    """The 9 m strip wall of GM-246 and GM-256."""
    return MSEWallGeometry(wall_height=9.0, reinforcement_length=7.0,
                           reinforcement_spacing=0.75)


def _strip(Sh=0.75):
    return dataclasses.replace(RIBBED_STEEL_STRIP_50x4, horizontal_spacing=Sh)


def test_the_strip_wall_before_and_after():
    """At Sh 0.75 m, Rc = 0.05/0.75 = 0.0667 for GEC-11's 50 mm strip
    (GM-267): pullout 0.0667 of the Rc = 1 reading (28.762 to 42.101
    before GM-256), and the rupture resistance 31.9 kN a strip / 0.75 m =
    42.6 kN per metre of wall. Before GM-256: 0.982 at the bottom level,
    43.1 against Tmax 43.89 kN/m; 1.309 at 43.1 kN a 75 mm strip / 0.75 m
    until the GM-262 review, which took the strip's Tal from GEC-11's
    print, 0.55 Fy Ec at Example E3's corroded thickness, 638.6 kN/m of
    strip width; 1.455 at 47.9 kN a 75 mm strip until GM-267, and with
    pullout 2.876 to 4.210 (Rc 0.1). The 75 mm width was the tool's own:
    every pin moves by 50/75 = 2/3, and the bottom level now fails
    rupture, 0.970."""
    levels = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip())
    po = [lv["FOS_pullout"] for lv in levels]
    ru = [lv["FOS_rupture"] for lv in levels]
    assert min(po) == pytest.approx(2.876 * 2 / 3, abs=1e-3)      # 1.917
    assert max(po) == pytest.approx(4.210 * 2 / 3, abs=1e-3)      # 2.807
    assert min(ru) == pytest.approx(1.455 * 2 / 3, abs=1e-3)      # 0.970
    assert levels[-1]["Tmax_kN_per_m"] == pytest.approx(43.89, abs=0.01)
    assert levels[-1]["Tal_Rc_kN_per_m"] == pytest.approx(31.93 / 0.75,
                                                          abs=0.01)
    full = check_internal_stability(
        _strip_wall(), 20.0, 34.0,
        dataclasses.replace(RIBBED_STEEL_STRIP_50x4, coverage_ratio=1.0))
    for a, b in zip(levels, full):
        assert a["Rc"] == pytest.approx(0.0667)
        assert a["Pr_kN_per_m"] == pytest.approx(
            0.05 / 0.75 * b["Pr_kN_per_m"], abs=0.01)
    assert min(lv["FOS_pullout"] for lv in full) == pytest.approx(
        28.76, abs=0.01)


def test_pullout_and_rupture_fall_in_proportion_to_the_spacing():
    a = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(0.5))
    b = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(1.0))
    for x, y in zip(a, b):
        assert x["FOS_pullout"] == pytest.approx(2 * y["FOS_pullout"],
                                                 rel=2e-3)
        assert x["FOS_rupture"] == pytest.approx(2 * y["FOS_rupture"],
                                                 rel=2e-3)


def test_a_geogrid_wall_keeps_Rc_1():
    """A geosynthetic given no spacing is a continuous layer, Rc = 1, as
    before GM-256. FOS_pullout is 8.394 to 20.633: 10.493 to 25.792 before,
    times the alpha 0.8 GEC-11 Table 3-6 prints for a geogrid (p. 3-17; the
    GM-256 alpha commit, tests/test_mse_scale_effect_alpha.py). FOS_rupture
    is 0.862 at the bottom level with Example E1's GG-I (Tal 15.83 kN/m,
    GM-271); it was 1.192 with the retired built-in geogrid's 21.9 kN/m,
    whose source was not recorded."""
    geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)
    r = analyze_mse_wall(geom, 19.0, 34.0, GEOGRID_GEC11_E1_GG1)
    po = [lv["FOS_pullout"] for lv in r.internal_results]
    assert min(po) == pytest.approx(8.394, abs=1e-3)
    assert max(po) == pytest.approx(20.633, abs=1e-3)
    assert min(lv["FOS_rupture"] for lv in r.internal_results) == \
        pytest.approx(1.192 * 15.83 / 21.9, abs=1e-3)          # 0.862
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "default"
    assert c["chosen"].startswith("Rc 1, a continuous geosynthetic layer")
    assert "p. 3-13" in c["basis"] and "Eq. 3-1" in c["basis"]
    assert "b/Sh" in c["alternatives"][0]


def test_a_geosynthetic_width_with_no_spacing_is_said_to_be_unused():
    """GM-256 review: GEC-11 p. 3-12 describes geosynthetic STRIPS 'by the
    width of the strips and the center-to-center horizontal distance
    between them'. A width with no spacing leaves Rc unknown: the library
    keeps Rc = 1 and its choice says the width was not used (the retired
    Foundry wrapper gives every custom reinforcement a 0.05 m width), and
    the tool refuses it (funhouse_agent test_mse_coverage_ratio_adapter)."""
    strips = Reinforcement("geogrid strips", "geogrid", 30.0, width=0.5)
    assert coverage_ratio(strips) == 1.0
    assert "its width b 0.5 m is not used" in coverage_choice(strips)[
        "chosen"]
    assert coverage_ratio(dataclasses.replace(
        strips, horizontal_spacing=1.0)) == pytest.approx(0.5)


def test_a_steel_strip_or_grid_with_no_spacing_is_refused():
    with pytest.raises(ValueError) as err:
        analyze_mse_wall(_strip_wall(), 20.0, 34.0, RIBBED_STEEL_STRIP_50x4)
    text = str(err.value)
    for words in ("coverage_ratio", "reinforcement_horizontal_spacing",
                  "reinforcement_width", "Eq. 3-1", "Sh alone will do",
                  "the tool assumes none of them", "E3-19"):
        assert words in text, words
    grid = Reinforcement("a grid", "metallic_grid", 100.0)
    with pytest.raises(ValueError, match="outside longitudinal bars"):
        coverage_ratio(grid)
    with pytest.raises(ValueError, match="needs the steel grid's width b"):
        coverage_ratio(dataclasses.replace(grid, horizontal_spacing=1.5))
    # GM-262: the built-in mat carries its own b, as the strip does
    with pytest.raises(ValueError,
                       match="this grid is 0.4572 m wide, so Sh alone"):
        coverage_ratio(WELDED_WIRE_GRID_W11)
    with pytest.raises(ValueError, match="not both"):
        coverage_ratio(dataclasses.replace(_strip(), coverage_ratio=0.1))
    with pytest.raises(ValueError, match="at most 1"):
        coverage_ratio(_strip(0.04))
    for Rc in (0.0, 1.2):
        with pytest.raises(ValueError, match="more than 0 and at most 1"):
            Reinforcement("s", "metallic_strip", 500.0, coverage_ratio=Rc)


def test_the_built_in_strip_strength_is_per_metre_of_strip_width():
    """Eq. 3-11a: Tal = Fy Ac / b, per metre of strip width; per metre of
    wall at Sh it is the strip's strength / Sh. Since the GM-262 review
    the built-in's Tal is 0.55 Fy Ec (p. 4-77; Ac = b Ec, Fig. 3-5a), Ec =
    0.102 in. (GEC-11 Vol II Example E3, p. E3-16), Fy 65 ksi: 638.6 kN/m,
    31.9 kN a 50 mm strip since GM-267 (47.9 kN a 75 mm strip before; 43.1
    kN a strip, 574.67 kN/m, of no recorded source, before the GM-262
    review). Tal per strip and Rc agree: Tal x b / Sh = Tal x Rc."""
    assert RIBBED_STEEL_STRIP_50x4.width == 0.05
    assert RIBBED_STEEL_STRIP_50x4.Tallowable * 0.05 == pytest.approx(
        31.930, abs=1e-3)
    lv = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(0.6))[0]
    assert lv["Tal_Rc_kN_per_m"] == pytest.approx(31.930 / 0.6, abs=0.01)
    assert lv["Tal_Rc_kN_per_m"] == pytest.approx(
        lv["Tallowable_kN_per_m"] * lv["Rc"], rel=1e-3)
    assert lv["Tallowable_kN_per_m"] == pytest.approx(638.60, abs=0.01)


def test_the_built_in_strip_is_GEC11s_50_mm_strip_as_printed():
    """GM-267: E3's strip, 'Grade 65 (Fy = 65 ksi), 1.969 in. x 0.157 in.
    (50 mm wide x 4 mm)', Ec = 0.102 in. after 75 years, Tn = 65 ksi x
    0.200 in.^2 = 13.00 k a strip (p. E3-16), the strip GEC-11 Vol I p.
    2-16 gives as 'currently commercially available'. The built-in's 0.55
    Fy Ec over b = 50 mm is 31.93 kN a strip, against 0.55 Tn = 7.15 k =
    31.80 kN on E3's rounded 0.200 in.^2 (0.4 %). It was 75 mm wide, the
    tool's own width; the choice no longer offers the 50 mm strip as the
    other reading, since it is the built-in."""
    from retaining_walls.reinforcement import (
        STRIP_ALLOWABLE_FY, steel_strip_Tal)
    IN, KSI, KIP = 0.0254, 6894.757, 4.4482216
    assert steel_strip_Tal(0.102 * IN, 65 * KSI) * 1.969 * IN / KIP == \
        pytest.approx(13.00, rel=0.005)
    assert RIBBED_STEEL_STRIP_50x4.Tallowable == pytest.approx(
        STRIP_ALLOWABLE_FY * 13.00 * KIP / (1.969 * IN), rel=0.005)
    assert RIBBED_STEEL_STRIP_50x4.Tallowable * 0.05 / KIP == \
        pytest.approx(0.55 * 13.00, rel=0.005)
    assert RIBBED_STEEL_STRIP_50x4.width == pytest.approx(1.969 * IN,
                                                          rel=5e-4)
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip())
    (c,) = [c for c in r.choices if c["what"].startswith("Long-term")]
    assert c["how"] == "module"
    assert c["chosen"].startswith(
        "Ribbed steel strip 50 x 4 mm (GEC-11 Vol II Example E3): Tal 638.6 "
        "kN/m of strip width (31.9 kN a 0.05 m strip), the same at every "
        "level; FOS_rupture is below 1.0 at z = 8.62 m (lowest 0.97)")
    for words in ("0.55 Fy", "p. 4-77", "p. E3-16", "0.102 in.",
                  "p. 2-16", "50 mm wide x 4 mm", "13.00 k a strip"):
        assert words in c["basis"], words
    assert "tool's own" not in c["basis"]
    alt, specified = c["alternatives"]
    assert alt.startswith("a stronger reinforcement, or a closer spacing")
    # GM-271 review: the built-in is E3's, not a product
    assert specified.startswith(
        "the reinforcement specified for the wall, where there is one: "
        "Ribbed steel strip 50 x 4 mm (GEC-11 Vol II Example E3) is the "
        "worked example's")
    for words in ("reinforcement_type 'metallic_strip' with "
                  "reinforcement_Tallowable and reinforcement_width",
                  "Eq. 3-11a, p. 3-22", "0.55 Fy for strips"):
        assert words in specified, words
    with pytest.raises(ValueError, match="positive corroded"):
        steel_strip_Tal(0.0, 65 * KSI)


def test_the_choice_says_how_Rc_was_set():
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip())
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "given"
    # GM-267: the 50 mm strip (Rc 0.1 and 63.86 kN/m at 75 mm)
    assert c["chosen"].startswith("Rc 0.0667 = b 0.05 m / Sh 0.75 m")
    assert "Tal x Rc 42.57 kN/m of wall" in c["chosen"]
    grid = dataclasses.replace(WELDED_WIRE_GRID_W11, coverage_ratio=0.3)
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, grid)
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "given" and c["chosen"].startswith("Rc 0.3, given")


def test_a_tested_F_star_replaces_the_empirical_one():
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip(),
                         F_star_tested=1.1)
    assert {lv["F_star"] for lv in r.internal_results} == {1.1}
    assert not [c for c in r.choices if "(Cu_backfill)" in c["what"]]
    (c,) = [c for c in r.choices if "(F_star_tested)" in c["what"]]
    assert c["how"] == "given"
    assert c["chosen"] == "F* 1.1 at every level, from pullout tests"
    assert "note (i)" in c["basis"] and "Appendix B" in c["basis"]
    (alt,) = c["alternatives"]
    empirical = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip())
    assert alt.startswith("the empirical F*, Eqs. 3-4 and 3-5 at the Cu = 4")
    assert (f"lowest {min(lv['FOS_pullout'] for lv in empirical):.3g}"
            in alt)


def test_a_Cu_beside_a_tested_F_star_is_said_to_be_in_no_check():
    """GM-256 review: with a tested F* the Cu is in no check. The F*
    choice says so and what it bears on: a strip's empirical F* given
    beside the tested one, and GEC-11 Table 3-6 note (i)'s check on the
    empirical F* (p. 3-17); a geogrid's empirical F* takes no Cu."""
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip(),
                         F_star_tested=1.1, Cu_backfill=7.0)
    assert r.internal_results == analyze_mse_wall(
        _strip_wall(), 20.0, 34.0, _strip(), F_star_tested=1.1).internal_results
    assert not [c for c in r.choices if "(Cu_backfill)" in c["what"]]
    (c,) = [c for c in r.choices if "(F_star_tested)" in c["what"]]
    assert c["chosen"].startswith(
        "F* 1.1 at every level, from pullout tests. Cu_backfill 7 is not "
        "used in the checks: the tested F* replaces the empirical F* of "
        "Eqs. 3-4 and 3-5, which is what takes Cu.")
    assert "at least 4, within the fill" in c["chosen"]
    assert c["alternatives"][0].startswith(
        "the empirical F*, Eqs. 3-4 and 3-5 at Cu 7 (F* 2.00 at the top)")
    geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)
    g = analyze_mse_wall(geom, 19.0, 34.0, GEOGRID_GEC11_E1_GG1,
                         F_star_tested=0.6, Cu_backfill=2.5)
    assert not [c for c in g.choices if "(Cu_backfill)" in c["what"]]
    (c,) = [c for c in g.choices if "(F_star_tested)" in c["what"]]
    for words in ("Cu_backfill 2.5 is not used in the checks",
                  "this reinforcement's empirical F* takes no Cu",
                  "It serves only as GEC-11 Table 3-6 note (i)'s check",
                  "below 4, outside the fill", "tests must be run"):
        assert words in c["chosen"], words


def test_a_tested_F_star_by_depth_is_linear_between():
    pts = [[1.0, 1.5], [5.0, 0.7]]
    assert F_star_from_tests(pts, 0.2) == 1.5
    assert F_star_from_tests(pts, 3.0) == pytest.approx(1.1)
    assert F_star_from_tests(pts, 8.0) == 0.7
    levels = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(),
                                      F_star_tested=pts)
    for lv in levels:
        assert lv["F_star"] == pytest.approx(
            F_star_from_tests(pts, lv["depth_m"]), abs=2e-3)


def test_a_tested_alpha_scales_the_pullout():
    base = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip())
    tested = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(),
                                      alpha_tested=0.8)
    for a, b in zip(base, tested):
        assert a["alpha"] == 1.0 and b["alpha"] == 0.8
        assert b["Pr_kN_per_m"] == pytest.approx(0.8 * a["Pr_kN_per_m"],
                                                 abs=0.01)
        assert b["FOS_rupture"] == a["FOS_rupture"]
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip(),
                         alpha_tested=0.8)
    (c,) = [c for c in r.choices if "(alpha_tested)" in c["what"]]
    assert c["chosen"] == "alpha 0.8, from pullout tests"
    assert "p. 3-16" in c["basis"] and "note (ii)" in c["basis"]
    assert c["alternatives"][0].startswith(
        "alpha 1, Table 3-6's default value for inextensible strips")


@pytest.mark.parametrize("kwargs, words", [
    ({"alpha_tested": 0.0}, "more than 0 and at most 1"),
    ({"alpha_tested": 1.2}, "more than 0 and at most 1"),
    ({"F_star_tested": -0.5}, "must be positive"),
    ({"F_star_tested": [[3.0, 1.0], [1.0, 1.2]]}, "increasing order"),
    ({"F_star_tested": [[1.0]]}, "one number for every level"),
    ({"F_star_tested": "high"}, "one number for every level"),
])
def test_a_tested_value_that_is_not_one_is_refused(kwargs, words):
    with pytest.raises(ValueError, match=words):
        check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(),
                                 **kwargs)


def test_the_calc_steps_carry_Rc():
    strip = _strip()
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, strip,
                         F_star_tested=[[0.0, 1.6], [6.0, 0.8]])
    analysis = {"wall_type": "mse", "geom": _strip_wall(),
                "gamma_backfill": 20.0, "phi_backfill": 34.0,
                "reinforcement": strip,
                "F_star_tested": [[0.0, 1.6], [6.0, 0.8]]}
    inputs = {i.name: i.value for i in get_input_summary(r, analysis)}
    # GM-267: the 50 mm strip, b 0.05 m / Sh 0.75 m (0.100 at 75 mm)
    assert inputs["R_c"] == "0.067" and inputs["S_h"] == "0.750"
    items = [it for sec in get_calc_steps(r, analysis) for it in sec.items]
    by_title = {getattr(it, "title", ""): it for it in items}
    assert by_title["Pullout Resistance"].equation.endswith("× C × R_c")
    assert "R_c = 0.067" in by_title["Pullout Resistance"].substitution
    # 31.9 kN a 50 mm strip / 0.75 m since GM-267 (47.9 kN a 75 mm strip
    # since the GM-262 review, 63.9; 43.1 before)
    assert by_title["Tensile Resistance per Metre of Wall"].result_value == \
        "42.6"
    tested = by_title["Pullout Resistance Factor F* (from pullout tests)"]
    assert tested.substitution == "F* 1.6 at 0 m, 0.8 at 6 m (linear between)"
    table = by_title["Internal Stability per Reinforcement Level"]
    assert "T_al R_c (kN/m)" in table.headers
    assert table.rows[0][3] == "42.6"
