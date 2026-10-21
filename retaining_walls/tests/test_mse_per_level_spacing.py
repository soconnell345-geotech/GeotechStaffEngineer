"""GM-261: the horizontal spacing Sh, or the coverage ratio Rc, per
reinforcement level.

GEC-11 Vol II Example E3 (FHWA-NHI-10-025) changes the spacing with depth:
"The last column of Table E3-7.3 provides horizontal spacing of the
reinforcing strips which is obtained by dividing the panel width, wp, by
the governing number of strips, Ng" (p. E3-19): 2 strips a 5 ft panel at
Levels 1 to 8, 3 at Levels 9 to 12. Before GM-261 the module took one Rc
for the wall, so a wall like E3 was run once per spacing. A single value
holds at every level, as before; E3 in one run is V-056
(validation_examples/test_published_v056_v057_gec11_slope.py).
"""

import dataclasses

import pytest

from retaining_walls.calc_steps import get_calc_steps, get_input_summary
from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_internal_stability, coverage_choice,
    coverage_ratio, coverage_ratios,
)
from retaining_walls.reinforcement import (
    GEOGRID_UX1600, RIBBED_STEEL_STRIP_75x4, Reinforcement,
)


def _wall():
    """The 9 m strip wall of GM-246 and GM-256: 12 levels at Sv 0.75 m."""
    return MSEWallGeometry(wall_height=9.0, reinforcement_length=7.0,
                           reinforcement_spacing=0.75)


_SH = [0.75] * 8 + [0.5] * 4


def _strip(Sh=_SH):
    return dataclasses.replace(RIBBED_STEEL_STRIP_75x4, horizontal_spacing=Sh)


def test_each_level_takes_its_own_spacing():
    levels = check_internal_stability(_wall(), 20.0, 34.0, _strip())
    wide = check_internal_stability(_wall(), 20.0, 34.0, _strip(0.75))
    close = check_internal_stability(_wall(), 20.0, 34.0, _strip(0.5))
    assert [lv["Rc"] for lv in levels] == [0.1] * 8 + [0.15] * 4
    assert levels[:8] == wide[:8]
    assert levels[8:] == close[8:]
    assert levels[-1]["Tal_Rc_kN_per_m"] == pytest.approx(43.1 / 0.5,
                                                          abs=0.01)


def test_a_single_value_holds_at_every_level_as_before():
    one = check_internal_stability(_wall(), 20.0, 34.0, _strip(0.75))
    assert one == check_internal_stability(_wall(), 20.0, 34.0,
                                           _strip([0.75] * 12))
    assert not _strip(0.75).per_level and _strip().per_level
    assert coverage_ratio(_strip(0.75)) == pytest.approx(0.1)


def test_the_ratio_per_level():
    Rc = [0.1] * 8 + [0.15] * 4
    r = dataclasses.replace(RIBBED_STEEL_STRIP_75x4, coverage_ratio=Rc)
    levels = check_internal_stability(_wall(), 20.0, 34.0, r)
    assert levels == check_internal_stability(_wall(), 20.0, 34.0, _strip())
    assert coverage_ratios(r, _wall().reinforcement_depths) == Rc
    geogrid = dataclasses.replace(GEOGRID_UX1600, coverage_ratio=[1.0] * 10
                                  + [0.5] * 2)
    lv = check_internal_stability(_wall(), 20.0, 34.0, geogrid)
    assert lv[-1]["FOS_rupture"] == pytest.approx(
        0.5 * lv[-1]["Tallowable_kN_per_m"] / lv[-1]["Tmax_kN_per_m"],
        abs=1e-3)


def test_a_list_of_the_wrong_length_is_refused_naming_the_levels():
    with pytest.raises(ValueError) as err:
        analyze_mse_wall(_wall(), 20.0, 34.0, _strip(_SH[:10]))
    text = str(err.value)
    for words in ("reinforcement_horizontal_spacing gives 10 values",
                  "this wall has 12 levels", "0.38, 1.12",
                  "from the top down"):
        assert words in text, words
    with pytest.raises(ValueError, match="coverage_ratio gives 3 values"):
        check_internal_stability(_wall(), 20.0, 34.0, dataclasses.replace(
            RIBBED_STEEL_STRIP_75x4, coverage_ratio=[0.1, 0.1, 0.1]))


def test_what_is_refused_per_level():
    with pytest.raises(ValueError, match="per reinforcement level"):
        coverage_ratio(_strip())
    with pytest.raises(ValueError, match="not both"):
        coverage_ratios(dataclasses.replace(_strip(), coverage_ratio=0.1),
                        _wall().reinforcement_depths)
    narrow = _strip([0.75] * 11 + [0.05])
    with pytest.raises(ValueError,
                       match="Sh 0.05 m at level 12 from the top"):
        check_internal_stability(_wall(), 20.0, 34.0, narrow)
    for bad, words in (([], "at least one value"),
                       ([0.75, "x"], "a list of numbers"),
                       ([0.75, -0.5], "must be positive")):
        with pytest.raises(ValueError, match=words):
            _strip(bad)
    with pytest.raises(ValueError, match="more than 0 and at most 1"):
        Reinforcement("s", "metallic_strip", 500.0, coverage_ratio=[0.1, 2])
    no_b = Reinforcement("s", "metallic_strip", 500.0,
                         horizontal_spacing=[0.75] * 12)
    with pytest.raises(ValueError, match=r"\[0.75, .*\] m per level needs"):
        check_internal_stability(_wall(), 20.0, 34.0, no_b)


def test_the_choice_lists_the_spacings_by_level():
    r = analyze_mse_wall(_wall(), 20.0, 34.0, _strip())
    (c,) = [c for c in r.choices if c["what"].startswith("Coverage ratio")]
    assert c["how"] == "given"
    assert c["chosen"].startswith(
        "Rc = b 0.075 m / Sh per level, from the top: Sh 0.75 m at levels "
        "1-8 (Rc 0.1); Sh 0.5 m at levels 9-12 (Rc 0.15)")
    assert "Tal x Rc 57.47 to 86.2 kN/m of wall" in c["chosen"]
    assert "Table E3-7.3" in c["basis"] and "p. E3-19" in c["basis"]
    given = coverage_choice(dataclasses.replace(
        RIBBED_STEEL_STRIP_75x4, coverage_ratio=[0.1] * 11 + [0.2]))
    assert given["chosen"] == ("Rc given per level, from the top: 0.1 at "
                               "levels 1-11; 0.2 at level 12")


def test_the_calc_package_shows_the_spacing_per_level():
    analysis = {"geom": _wall(), "gamma_backfill": 20.0, "phi_backfill": 34.0,
                "reinforcement": _strip()}
    result = analyze_mse_wall(_wall(), 20.0, 34.0, _strip())
    items = {i.name: i for i in get_input_summary(result, analysis)}
    assert items["S_h"].value == "0.750 × 8, 0.500 × 4"
    assert "per level from the top" in items["S_h"].description
    assert items["R_c"].value == "0.100 × 8, 0.150 × 4"
    text = repr(get_calc_steps(result, analysis))
    assert "R_c = 0.100 × 8, 0.150 × 4 per level from the top" \
        in text
    assert "57.5 to 86.2" in text
    one = {"geom": _wall(), "gamma_backfill": 20.0, "phi_backfill": 34.0,
           "reinforcement": _strip(0.75)}
    items = {i.name: i for i in get_input_summary(
        analyze_mse_wall(_wall(), 20.0, 34.0, _strip(0.75)), one)}
    assert items["S_h"].value == "0.750"
    assert items["R_c"].value == "0.100"
