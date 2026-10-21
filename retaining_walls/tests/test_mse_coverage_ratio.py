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
the tension per metre of wall at no spacing.
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, coverage_ratio,
    F_star_from_tests,
)
from retaining_walls.reinforcement import (
    GEOGRID_UX1600, RIBBED_STEEL_STRIP_75x4, Reinforcement,
    WELDED_WIRE_GRID_W11,
)


def _strip_wall():
    """The 9 m strip wall of GM-246 and GM-256."""
    return MSEWallGeometry(wall_height=9.0, reinforcement_length=7.0,
                           reinforcement_spacing=0.75)


def _strip(Sh=0.75):
    return dataclasses.replace(RIBBED_STEEL_STRIP_75x4, horizontal_spacing=Sh)


def test_the_strip_wall_before_and_after():
    """At Sh 0.75 m, Rc = 0.075/0.75 = 0.1: pullout a tenth of the Rc = 1
    reading (28.762 to 42.101 before GM-256), and the rupture resistance
    43.1 kN a strip / 0.75 m = 57.5 kN per metre of wall (0.982 at the
    bottom level before, 43.1 against Tmax 43.89 kN/m)."""
    levels = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip())
    po = [lv["FOS_pullout"] for lv in levels]
    ru = [lv["FOS_rupture"] for lv in levels]
    assert min(po) == pytest.approx(2.876, abs=1e-3)
    assert max(po) == pytest.approx(4.210, abs=1e-3)
    assert min(ru) == pytest.approx(1.309, abs=1e-3)
    assert levels[-1]["Tmax_kN_per_m"] == pytest.approx(43.89, abs=0.01)
    assert levels[-1]["Tal_Rc_kN_per_m"] == pytest.approx(43.1 / 0.75,
                                                          abs=0.01)
    full = check_internal_stability(
        _strip_wall(), 20.0, 34.0,
        dataclasses.replace(RIBBED_STEEL_STRIP_75x4, coverage_ratio=1.0))
    for a, b in zip(levels, full):
        assert a["Rc"] == pytest.approx(0.1)
        assert a["Pr_kN_per_m"] == pytest.approx(0.1 * b["Pr_kN_per_m"],
                                                 abs=0.01)
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
    before GM-256, and FOS_rupture is unchanged, 1.192 at the bottom level.
    FOS_pullout is 8.394 to 20.633: 10.493 to 25.792 before, times the
    alpha 0.8 GEC-11 Table 3-6 prints for a geogrid (p. 3-17; the GM-256
    alpha commit, tests/test_mse_scale_effect_alpha.py)."""
    geom = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0,
                           reinforcement_spacing=0.6)
    r = analyze_mse_wall(geom, 19.0, 34.0, GEOGRID_UX1600)
    po = [lv["FOS_pullout"] for lv in r.internal_results]
    assert min(po) == pytest.approx(8.394, abs=1e-3)
    assert max(po) == pytest.approx(20.633, abs=1e-3)
    assert min(lv["FOS_rupture"] for lv in r.internal_results) == \
        pytest.approx(1.192, abs=1e-3)
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "default"
    assert c["chosen"].startswith("Rc 1, a continuous geosynthetic layer")
    assert "p. 3-13" in c["basis"] and "Eq. 3-1" in c["basis"]
    assert "b/Sh" in c["alternatives"][0]


def test_a_steel_strip_or_grid_with_no_spacing_is_refused():
    with pytest.raises(ValueError) as err:
        analyze_mse_wall(_strip_wall(), 20.0, 34.0, RIBBED_STEEL_STRIP_75x4)
    text = str(err.value)
    for words in ("coverage_ratio", "reinforcement_horizontal_spacing",
                  "reinforcement_width", "Eq. 3-1", "Sh alone will do",
                  "For scale only, not taken", "E3-19"):
        assert words in text, words
    with pytest.raises(ValueError, match="outside longitudinal bars"):
        coverage_ratio(WELDED_WIRE_GRID_W11)
    with pytest.raises(ValueError, match="needs the steel grid's width b"):
        coverage_ratio(dataclasses.replace(WELDED_WIRE_GRID_W11,
                                           horizontal_spacing=1.5))
    with pytest.raises(ValueError, match="not both"):
        coverage_ratio(dataclasses.replace(_strip(), coverage_ratio=0.1))
    with pytest.raises(ValueError, match="at most 1"):
        coverage_ratio(_strip(0.05))
    for Rc in (0.0, 1.2):
        with pytest.raises(ValueError, match="more than 0 and at most 1"):
            Reinforcement("s", "metallic_strip", 500.0, coverage_ratio=Rc)


def test_the_built_in_strip_strength_is_per_metre_of_strip_width():
    """Eq. 3-11a: Tal = Fy Ac / b. The built-in's 43.1 kN a strip over b =
    75 mm; per metre of wall at Sh it is 43.1 / Sh."""
    assert RIBBED_STEEL_STRIP_75x4.Tallowable * 0.075 == pytest.approx(43.1)
    lv = check_internal_stability(_strip_wall(), 20.0, 34.0, _strip(0.6))[0]
    assert lv["Tal_Rc_kN_per_m"] == pytest.approx(43.1 / 0.6, abs=0.01)
    assert lv["Tallowable_kN_per_m"] == pytest.approx(574.67, abs=0.01)


def test_the_choice_says_how_Rc_was_set():
    r = analyze_mse_wall(_strip_wall(), 20.0, 34.0, _strip())
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "given"
    assert c["chosen"].startswith("Rc 0.1 = b 0.075 m / Sh 0.75 m")
    assert "Tal x Rc 57.47 kN/m of wall" in c["chosen"]
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
    assert inputs["R_c"] == "0.100" and inputs["S_h"] == "0.750"
    items = [it for sec in get_calc_steps(r, analysis) for it in sec.items]
    by_title = {getattr(it, "title", ""): it for it in items}
    assert by_title["Pullout Resistance"].equation.endswith("× C × R_c")
    assert "R_c = 0.100" in by_title["Pullout Resistance"].substitution
    assert by_title["Tensile Resistance per Metre of Wall"].result_value == \
        "57.5"
    tested = by_title["Pullout Resistance Factor F* (from pullout tests)"]
    assert tested.substitution == "F* 1.6 at 0 m, 0.8 at 6 m (linear between)"
    table = by_title["Internal Stability per Reinforcement Level"]
    assert "T_al R_c (kN/m)" in table.headers
    assert table.rows[0][3] == "57.5"
