"""GM-277: the levels the module places at Sv (i + 1/2), no level_depths.

GEC-11 Vol I, read on the rendered pages: "The term Sv is equal to the
vertical reinforcement spacing for a layer where vertically adjacent
reinforcements are equally spaced from the layer under consideration. In
this case, sigma_H, calculated at the level of the reinforcement, is at
the center of the contributory height" (p. 4-46); "For the top and bottom
layers of reinforcement, Sv is the distance from top or bottom of wall,
respectively, to the midpoint between the first and second layer" (p.
4-47, Fig. 4-14, Sv(1) running down to the base), their TMAX "the product
of the contributory height and the average factored horizontal stress
acting upon that contributory height", the average on the tributary
trapezoid "or at the midpoint of the contributory height" (p. 4-47).

Two defects, both on the unsafe side: H / Sv rounded down in floating
point dropped a level on walls a whole number of spacings high (4.8 m at
Sv 0.8 m got 5 levels, 0.8 m of wall in none), and on a wall that is not
a whole number of spacings the bottom level took Sv, the strip below it
in no level. The count now takes a tolerance, and the bottom level's
contributory height runs to the base with sigma_H at its middle.
"""

import dataclasses
import math

import pytest

from retaining_walls.calc_steps import get_calc_steps
from retaining_walls.earth_pressure import rankine_Ka
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    Kr_Ka_ratio, Tmax_at_level, analyze_mse_wall, backslope_geometry,
    check_internal_stability, contributory_Tmax,
)
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG1, RIBBED_STEEL_STRIP_50x4, WELDED_WIRE_GRID_W11,
)

FT = 0.3048
STRIP = dataclasses.replace(RIBBED_STEEL_STRIP_50x4, horizontal_spacing=0.75)
MAT = dataclasses.replace(WELDED_WIRE_GRID_W11, horizontal_spacing=1.5)
KA = rankine_Ka(34.0)

#: Walls a whole number of spacings high whose H / Sv falls under the
#: whole number in floating point (the GM-273 review's cases): (H, Sv) in
#: m, and in ft, entered as the metres they convert to.
SI_SHORT = [(1.2, 0.4), (2.4, 0.4), (2.8, 0.4), (4.8, 0.4), (5.6, 0.4),
            (7.6, 0.4), (9.2, 0.4), (9.6, 0.4), (11.2, 0.4), (11.6, 0.4),
            (13.2, 0.4), (15.2, 0.4), (2.4, 0.8), (4.8, 0.8), (5.6, 0.8),
            (9.6, 0.8), (11.2, 0.8), (15.2, 0.8), (18.4, 0.8), (19.2, 0.8),
            (11.7, 0.9), (18.9, 0.9)]
US_SHORT_FT = [(7.0, 1.0), (14.0, 1.0), (7.5, 1.5), (10.5, 1.5),
               (15.0, 1.5), (30.0, 1.5), (14.0, 2.0), (28.0, 2.0),
               (15.0, 3.0), (30.0, 3.0)]


def _sweep(spacings, n_max, unit):
    """Every H = k Sv up to n_max spacings, as typed (rounded to the
    hundredth of the unit) and converted to metres."""
    for Sv in spacings:
        for k in range(1, n_max + 1):
            yield round(k * Sv, 2) * unit, Sv * unit, k


@pytest.mark.parametrize("unit, spacings", [
    (1.0, (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)),
    (FT, (1.0, 1.5, 2.0, 2.5, 3.0))])
def test_a_wall_a_whole_number_of_spacings_high_gets_every_level(
        unit, spacings):
    """k spacings high is k levels, the bottom at (k - 1/2) Sv, and the
    contributory heights tile the wall, each Sv, in metres and in feet."""
    for H, Sv, k in _sweep(spacings, 40, unit):
        g = MSEWallGeometry(wall_height=H, reinforcement_spacing=Sv)
        assert g.n_reinforcement_levels == k, (H, Sv)
        assert g.whole_spacings, (H, Sv)
        assert g.reinforcement_depths[-1] == pytest.approx(
            (k - 0.5) * Sv, abs=6e-4)
        heights = g.contributory_heights
        assert heights[0][0] == 0.0 and heights[-1][1] == H
        assert all(b - a == pytest.approx(Sv) for a, b in heights)


@pytest.mark.parametrize("H, Sv", SI_SHORT
                         + [(h * FT, s * FT) for h, s in US_SHORT_FT])
def test_the_walls_that_lost_a_level(H, Sv):
    """H / Sv is under the whole number in floating point; the level the
    rounding dropped is back and the bottom 1 Sv of wall is in it."""
    assert int(H / Sv) == round(H / Sv) - 1      # the old count
    g = MSEWallGeometry(wall_height=H, reinforcement_spacing=Sv)
    assert g.n_reinforcement_levels == round(H / Sv)
    assert sum(b - a for a, b in g.contributory_heights) == \
        pytest.approx(H)
    rows = check_internal_stability(g, 19.0, 34.0, STRIP)
    z = g.reinforcement_depths[-1]
    assert rows[-1]["Tmax_kN_per_m"] == pytest.approx(Tmax_at_level(
        z, 19.0, KA, Kr_Ka_ratio(z, "metallic"), Sv), abs=0.006)


def test_the_four_point_eight_metre_wall():
    """The backlog's case: 4.8 m at Sv 0.8 m, six levels to 4.4 m."""
    g = MSEWallGeometry(wall_height=4.8, reinforcement_spacing=0.8)
    assert g.reinforcement_depths == [0.4, 1.2, 2.0, 2.8, 3.6, 4.4]
    r = analyze_mse_wall(g, 19.0, 34.0, STRIP)
    assert r.n_levels == 6


@pytest.mark.parametrize("H, Sv", [(6.32, 0.6), (25.64 * FT, 2.5 * FT),
                                   (5.0, 0.75), (3.9, 0.8)])
@pytest.mark.parametrize("reinforcement, kind", [
    (STRIP, "metallic"), (MAT, "metallic_grid"),
    (GEOGRID_GEC11_E1_GG1, "geosynthetic")])
def test_the_bottom_level_runs_to_the_base(H, Sv, reinforcement, kind):
    """Not a whole number of spacings: the bottom level's contributory
    height from (n - 1) Sv down to H (Fig. 4-14), sigma_H at its middle;
    every other level Sv with sigma_H at the level, as before."""
    g = MSEWallGeometry(wall_height=H, reinforcement_spacing=Sv,
                        surcharge=10.0)
    n = g.n_reinforcement_levels
    assert n == math.floor(H / Sv) and not g.whole_spacings
    a, b = g.contributory_heights[-1]
    assert (a, b) == (pytest.approx((n - 1) * Sv), H)
    assert b - a > Sv
    rows = check_internal_stability(g, 19.0, 34.0, reinforcement)
    for z, lv in zip(g.reinforcement_depths[:-1], rows):
        assert lv["Tmax_kN_per_m"] == round(Tmax_at_level(
            z, 19.0, KA, Kr_Ka_ratio(z, kind), Sv, 10.0), 2)
    mid = 0.5 * (a + b)
    assert rows[-1]["Tmax_kN_per_m"] == round(Tmax_at_level(
        mid, 19.0, KA, Kr_Ka_ratio(mid, kind), b - a, 10.0), 2)
    assert rows[-1]["Tmax_kN_per_m"] == round(contributory_Tmax(
        a, b, 19.0, KA, kind, 10.0, "midpoint"), 2)
    # more than Sv and sigma_H at the level gave: the strip is carried
    z = g.reinforcement_depths[-1]
    assert rows[-1]["Tmax_kN_per_m"] > Tmax_at_level(
        z, 19.0, KA, Kr_Ka_ratio(z, kind), Sv, 10.0)
    # pullout does not depend on the contributory height
    assert rows[-1]["depth_m"] == round(z, 2)


def test_the_top_level_is_in_the_middle_of_its_contributory_height():
    """The top level at Sv/2: its contributory height from the top of the
    wall to the midpoint between it and the second level is Sv (p. 4-47),
    and sigma_H at the level is sigma_H at its middle."""
    g = MSEWallGeometry(wall_height=6.32, reinforcement_spacing=0.6)
    assert g.contributory_heights[0] == (0.0, 0.6)
    top = check_internal_stability(g, 19.0, 34.0, STRIP)[0]
    assert top["depth_m"] == 0.3
    assert top["Tmax_kN_per_m"] == round(contributory_Tmax(
        0.0, 0.6, 19.0, KA, "metallic", 0.0, "midpoint"), 2)


def test_a_sloping_backfill_carries_sigma_2_to_the_base():
    """Under a 15 deg slope the bottom level's sigma_H takes sigma_2 = S_eq
    gamma (GEC-11 Eq. 4-29) over its contributory height to the base."""
    g = MSEWallGeometry(wall_height=6.32, reinforcement_spacing=0.6,
                        reinforcement_length=5.0, backfill_slope=15.0)
    sigma_2 = 19.0 * backslope_geometry(g)["S_eq"]
    a, b = g.contributory_heights[-1]
    rows = check_internal_stability(g, 19.0, 34.0, STRIP)
    assert rows[-1]["Tmax_kN_per_m"] == round(contributory_Tmax(
        a, b, 19.0, KA, "metallic", sigma_2, "midpoint"), 2)


def test_one_level_takes_the_whole_wall():
    """Under two spacings high: one level, at Sv/2, its contributory height
    the whole wall (top and bottom layer at once, p. 4-47), less than Sv
    when the wall is lower than Sv."""
    for H in (1.0, 0.5):
        g = MSEWallGeometry(wall_height=H, reinforcement_spacing=0.6,
                            reinforcement_length=2.5)
        assert g.n_reinforcement_levels == 1
        assert g.contributory_heights == [(0.0, H)]
        (lv,) = check_internal_stability(g, 19.0, 34.0, STRIP)
        assert lv["Tmax_kN_per_m"] == round(contributory_Tmax(
            0.0, H, 19.0, KA, "metallic", 0.0, "midpoint"), 2)
        (c,) = [c for c in analyze_mse_wall(g, 19.0, 34.0, STRIP).choices
                if c["what"].startswith("Tmax at each level")]
        assert c["chosen"].startswith("one level, at Sv/2 = 0.3 m")
        assert f"the whole wall, {H:g} m" in c["chosen"]


def test_a_spacing_that_puts_the_first_level_outside_the_wall_is_refused():
    with pytest.raises(ValueError, match=r"Sv / 2 = 0\.6 m, at or below "
                       r"the base of the wall \(wall_height 0\.5 m\).*"
                       r"level_depths"):
        MSEWallGeometry(wall_height=0.5, reinforcement_spacing=1.2)
    # the depths given place the levels: the spacing is not used
    MSEWallGeometry(wall_height=0.5, reinforcement_spacing=1.2,
                    level_depths=[0.25])


def _layout_choice(r):
    (c,) = [c for c in r.choices if c["what"].startswith(
        "Tmax at each level, the levels equally spaced")]
    return c


def test_the_choice_says_what_the_layout_assumed():
    whole = analyze_mse_wall(MSEWallGeometry(wall_height=6.0), 19.0, 34.0,
                             STRIP)
    c = _layout_choice(whole)
    assert c["how"] == "default"
    assert c["chosen"].startswith(
        "10 levels equally spaced at Sv 0.6 m, the top level at Sv/2 = 0.3 "
        "m below the top of the wall; each level's contributory height Sv "
        "with the level at its middle")
    assert c["chosen"].endswith("the top level's from the top of the wall "
                                "and the bottom level's to its base")
    for words in ("p. 4-46", "equally spaced from the layer under "
                  "consideration", "center of the contributory height",
                  "p. 4-47", "Fig. 4-14", "from top or bottom of wall",
                  "at the midpoint of the contributory height"):
        assert words in c["basis"], words
    (alt,) = c["alternatives"]
    assert "level_depths" in alt and "E3-15, E4-13" in alt
    part = analyze_mse_wall(MSEWallGeometry(wall_height=6.32), 19.0, 34.0,
                            STRIP)
    assert ("the bottom level's (z = 5.7 m) from the midpoint above it at "
            "5.4 m down to the base at 6.32 m, 0.92 m, with sigma_H at its "
            "middle (5.86 m): the wall height is not a whole number of "
            "spacings") in _layout_choice(part)["chosen"]
    assert "down to the base at 6.32 m" in part.summary()


def test_the_calc_package_says_where_the_bottom_level_ends():
    g = MSEWallGeometry(wall_height=6.32, reinforcement_length=5.0)
    analysis = {"geom": g, "gamma_backfill": 19.0, "phi_backfill": 34.0,
                "reinforcement": STRIP}
    text = repr(get_calc_steps(analyze_mse_wall(g, 19.0, 34.0, STRIP),
                               analysis))
    assert ("the bottom level's from 5.40 m down to the base, S_vt = "
            "0.92 m") in text
    assert "GEC-11 Vol I Eq. 4-32a, pp. 4-46, 4-47 (Fig. 4-14)" in text
    assert "Eq. 4-10" not in text
    whole = MSEWallGeometry(wall_height=6.0, reinforcement_length=5.0)
    text = repr(get_calc_steps(analyze_mse_wall(whole, 19.0, 34.0, STRIP),
                               dict(analysis, geom=whole)))
    assert "the bottom level's to its base" in text and "S_vt" not in text
